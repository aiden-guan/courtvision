"""CourtVision command line: ``courtvision <command>`` (also ``python -m courtvision``).

Commands
--------
``simulate``   offline simulator run (truth3d | mono | video), evaluated against simulator truth
``serve``      live dashboard: uvicorn + the FastAPI app, with the :class:`~courtvision.pipeline.LiveGame`
               running as a task on the server's own event loop
``run``        offline processing of a video file (calibration + detector + roster)
``calibrate``  a geometry ``Calibration`` JSON from a points file, or from the simulator camera
``render``     a synthetic demo video (mp4) from a simulation, plus its calibration / roster / truth
``eval``       compare a predicted ``events.jsonl`` with ``truth_events.jsonl``
``config``     print the effective YAML configuration
``latency``    the cloud latency harness (``tools/latency_harness.py``; real, billed calls)

Every command that builds a configuration takes ``--config FILE`` and repeatable
``--set section.key=value`` (see ``courtvision/config/default.yaml``). Explicit flags win over
``--set``, which wins over the file.

Credentials are detected by name only and never printed: ``ANTHROPIC_API_KEY`` (or an
``ant auth login`` profile) enables Claude colour lines in real-time ``serve``; ``TYPESAFE_API_KEY``
enables live Jev. Without them narration is template-only and Jev runs its deterministic fallbacks.
Offline commands (``simulate``, ``run``) default Jev to ``off`` so their results are reproducible
(source time runs far ahead of wall time); pass ``--jev live`` / ``--jev stub`` to change that.

Exit codes: 0 success, 1 runtime failure, 2 usage / configuration error.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import importlib.util
import ipaddress
import json
import logging
import math
import os
import re
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # heavy modules are imported inside the commands so `--help` stays fast
    from courtvision.config import AppConfig
    from courtvision.pipeline import LiveGame, RunResult

__all__ = [
    "CLIError",
    "DetectorSpec",
    "RedactTokens",
    "ServeApp",
    "attach_game",
    "build_parser",
    "build_uvicorn_server",
    "create_serve_app",
    "format_summary",
    "load_cli_config",
    "load_points_file",
    "main",
    "points_file_text",
    "parse_detector",
    "read_events_jsonl",
    "simulate_config",
    "unused_serve_flags",
]

log = logging.getLogger("courtvision.cli")

EXIT_OK, EXIT_ERROR, EXIT_USAGE = 0, 1, 2
SIM_MODES = ("truth3d", "mono", "video", "broadcast")
SOURCES = ("sim", "video", "camera")
JEV_MODES = ("live", "stub", "off")
TTS_KINDS = ("none", "say", "avspeech")
LEAGUES = ("nba", "fiba", "ncaa_m", "ncaa_w", "nfhs")
RFDETR_VARIANTS = ("nano", "small", "medium")
LATENCY_CONFIGS = ("C1", "C2", "C3", "C4", "J1", "J2", "J3", "J4", "J5")
ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"  # source checkout only: never part of the installed package

AGPL_WARNING = (
    "WARNING: rnd-yolo uses Ultralytics, which is AGPL-3.0 (R&D ONLY, RESEARCH.md D9). AGPL section 13 "
    "covers network use: never ship this path or weights trained with it without an Ultralytics "
    "Enterprise licence. The production detector is RF-DETR (--detector rfdetr / coreml)."
)


class CLIError(Exception):
    """A user-facing error: printed without a traceback, exit code 2 by default."""

    def __init__(self, message: str, exit_code: int = EXIT_USAGE) -> None:
        super().__init__(message)
        self.exit_code = exit_code


# --------------------------------------------------------------------------- argument parsing


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {text!r}")
    return value


def _non_negative_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"expected a number >= 0, got {text!r}")
    return value


def _port(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a port number, got {text!r}") from exc
    if not 0 < value < 65536:
        raise argparse.ArgumentTypeError(f"port must be 1-65535, got {value}")
    return value


def _override(text: str) -> str:
    if "=" not in text or not text.split("=", 1)[0].strip():
        raise argparse.ArgumentTypeError(f"expected section.key=value, got {text!r}")
    return text


def _detector_arg(text: str) -> DetectorSpec:
    try:
        return parse_detector(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _config_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("configuration")
    g.add_argument("--config", metavar="FILE", help="YAML config file (keys as in courtvision/config/default.yaml)")
    g.add_argument("--set", metavar="KEY=VALUE", action="append", default=[], type=_override,
                   help="override one config value, e.g. --set events.assist_window_s=4 (repeatable)")
    return p


def _sim_args(p: argparse.ArgumentParser, *, seconds_default: float | None = None) -> None:
    g = p.add_argument_group("simulation")
    g.add_argument("--seconds", type=_positive_float, default=seconds_default,
                   help="video seconds to simulate (default: sim.seconds)"
                   + (f"; {seconds_default:g} here" if seconds_default else ""))
    g.add_argument("--seed", type=int, default=None, help="simulator seed (default: sim.seed)")
    g.add_argument("--league", choices=LEAGUES, default=None, help="court + rules (default: game.league)")


def _perception_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("perception")
    g.add_argument("--calibration", metavar="FILE", help="Calibration JSON (courtvision calibrate)")
    g.add_argument("--scoreboard-roi", metavar="FILE", help="ScoreboardROI JSON for the venue scoreboard reader")
    g.add_argument("--detector", type=_detector_arg, default=None, metavar="SPEC",
                   help="colorblob (synthetic renders; also turns on the template jersey reader) | "
                        "rfdetr[:<weights>|nano|small|medium] | coreml:<mlpackage> | rnd-yolo:<weights> "
                        "(R&D only, AGPL-3.0)")
    g.add_argument("--roster", metavar="FILE", help="roster YAML/JSON (docs/examples/roster.yaml)")
    g.add_argument("--allow-download", action="store_true",
                   help="let RF-DETR fetch rfdetr's pretrained COCO checkpoint (~370 MB) when no weights file "
                        "is given; off by default so nothing is downloaded silently")


def build_parser() -> argparse.ArgumentParser:
    """The ``courtvision`` argument parser (every subcommand)."""
    parser = argparse.ArgumentParser(
        prog="courtvision",
        description="Real-time basketball CV -> events -> stats -> Jev decisions -> grounded narration.",
        epilog="Run `courtvision <command> -h` for a command's options.")
    from courtvision import __version__

    parser.add_argument("--version", action="version", version=f"courtvision {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="more logging (-vv for debug)")
    parser.add_argument("-q", "--quiet", action="store_true", help="errors only")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    cfgp = _config_parent()

    studio = sub.add_parser("studio", parents=[cfgp], help="open the local video analysis workspace")
    studio.add_argument("--host", default="127.0.0.1")
    studio.add_argument("--port", type=_port, default=8000)
    studio.add_argument("--open", action="store_true", help="open the workspace in your browser")
    studio.add_argument("--data-dir", default="runs/studio", help="local footage and analysis directory")
    studio.add_argument("--auth-token", default=None)
    studio.set_defaults(func=cmd_studio)

    # simulate
    p = sub.add_parser("simulate", parents=[cfgp], help="offline simulator run evaluated against truth",
                       description="Simulate a game, run it through the whole stack as fast as possible "
                                   "(source-time clocks: deterministic) and evaluate it against the "
                                   "simulator's truth. Prints the report summary; the run directory holds "
                                   "events.jsonl, truth_events.jsonl, transcript.jsonl, insights.jsonl, "
                                   "stats.json, report.json, report.md and config.yaml.")
    _sim_args(p, seconds_default=120.0)
    p.add_argument("--mode", choices=SIM_MODES, default="truth3d",
                   help="truth3d: 3-D truth frames; mono: noisy monocular observations; video: render + "
                        "perception (slow: ~20-30 ms/frame)")
    p.add_argument("--out", metavar="DIR", help="run directory (default: runs/sim-<mode>-seed<seed>-<time>)")
    p.add_argument("--jev", choices=JEV_MODES, default=None, help="Jev mode (default off for offline runs)")
    p.add_argument("--json", action="store_true", help="print the summary as JSON instead of markdown")
    p.set_defaults(func=cmd_simulate)

    # serve
    p = sub.add_parser("serve", parents=[cfgp], help="live dashboard (uvicorn + LiveGame)",
                       description="Serve the live dashboard. The game runs on the server's event loop; "
                                   "open http://HOST:PORT/ in a browser.")
    p.add_argument("--source", choices=SOURCES, default=None, help="frame source (default: source.kind)")
    p.add_argument("--mode", choices=SIM_MODES, default=None, help="sim source mode (default: source.sim_mode)")
    p.add_argument("--video", metavar="PATH", help="video file for --source video")
    p.add_argument("--camera", metavar="INDEX_OR_URL", help="camera index or stream URL for --source camera")
    p.add_argument("--speed", type=_positive_float, default=None, help="playback speed for sim / video sources")
    p.add_argument("--host", default=None, help="bind address (default 127.0.0.1: local only)")
    p.add_argument("--port", type=_port, default=None, help="port (default 8765)")
    p.add_argument("--jev", choices=JEV_MODES, default=None,
                   help="Jev mode (default: jev.mode; live needs TYPESAFE_API_KEY, else fallbacks)")
    p.add_argument("--no-llm", action="store_true", help="template-only narration (no Claude)")
    p.add_argument("--tts", choices=TTS_KINDS, default=None, help="voice (default: narration.tts)")
    p.add_argument("--auth-token", metavar="TOKEN", default=None,
                   help="require this bearer token (prefer the COURTVISION_TOKEN env var: argv is visible to ps)")
    p.add_argument("--insecure-no-auth", action="store_true",
                   help="allow a non-loopback --host without a token (exposes rosters on the network)")
    p.add_argument("--open", action="store_true", help="open the dashboard in a browser once the server is up")
    _sim_args(p)
    _perception_args(p)
    p.set_defaults(func=cmd_serve)

    # run
    p = sub.add_parser("run", parents=[cfgp], help="offline processing of a video file",
                       description="Process a video file offline (as fast as possible) through perception, "
                                   "events, stats and narration; writes a run directory.")
    p.add_argument("--video", metavar="PATH", required=True, help="video file")
    _perception_args(p)
    p.add_argument("--league", choices=LEAGUES, default=None, help="court + rules (default: game.league)")
    p.add_argument("--start", type=_non_negative_float, default=None, help="skip this many seconds first")
    p.add_argument("--duration", type=_positive_float, default=None, help="process this many seconds")
    p.add_argument("--out", metavar="DIR", help="run directory (default: runs/run-<video>-<time>)")
    p.add_argument("--jev", choices=JEV_MODES, default=None, help="Jev mode (default off for offline runs)")
    p.add_argument("--json", action="store_true", help="print the summary as JSON instead of markdown")
    p.set_defaults(func=cmd_run)

    # calibrate
    p = sub.add_parser("calibrate", parents=[cfgp], help="build a Calibration JSON (homography + rims)",
                       description="Fit the floor homography from clicked image points and court points or "
                                   "landmark names (docs/examples/calibration_points.json), or from the "
                                   "simulator camera, and print the fit RMSE in court feet.")
    p.add_argument("points", nargs="?", metavar="POINTS", help="points file (JSON or YAML)")
    p.add_argument("--from-sim", action="store_true", help="calibrate the simulator camera (sim.* camera settings)")
    p.add_argument("--out", metavar="FILE", default="calibration.json", help="output (default calibration.json)")
    p.add_argument("--league", choices=LEAGUES, default=None, help="court for landmark names (default: file / game.league)")
    p.add_argument("--image-size", nargs=2, type=int, metavar=("W", "H"), help="image size when the file has none")
    p.add_argument("--ransac-thresh", type=_positive_float, default=None, help="RANSAC threshold in ft (default 1.0)")
    p.add_argument("--points-out", metavar="FILE", help="--from-sim: also write the projected landmarks as a points file")
    p.add_argument("--roi-out", metavar="FILE", help="--from-sim: also write the renderer's scoreboard ROI JSON")
    p.set_defaults(func=cmd_calibrate)

    # render
    p = sub.add_parser("render", parents=[cfgp], help="write a synthetic demo video from a simulation",
                       description="Simulate a game and write the synthetic broadcast view as a video "
                                   "(H.264 for .mp4; lossless FFV1 for .avi / .mkv). Next to it go "
                                   "<stem>.calibration.json, <stem>.roster.yaml, <stem>.scoreboard_roi.json "
                                   "and <stem>.truth_events.jsonl, so `courtvision run` and `courtvision eval` "
                                   "work on the result.")
    _sim_args(p, seconds_default=30.0)
    p.add_argument("--out", metavar="FILE", default="runs/demo.mp4", help="output video (default runs/demo.mp4)")
    p.add_argument("--start", type=_non_negative_float, default=0.0, help="first second to render")
    p.add_argument("--no-sidecars", action="store_true", help="write only the video")
    p.set_defaults(func=cmd_render)

    # eval
    p = sub.add_parser("eval", help="compare predicted events with truth",
                       description="Match a predicted events.jsonl against truth_events.jsonl (greedy by "
                                   "time within the tolerance, per kind) and print the markdown report.")
    p.add_argument("pred", metavar="PRED", help="events.jsonl, or a run directory holding one")
    p.add_argument("--truth", metavar="FILE", help="truth_events.jsonl (default: next to PRED)")
    p.add_argument("--tol", type=_positive_float, default=1.0, help="matching tolerance in seconds (default 1.0)")
    p.add_argument("--kinds", metavar="K1,K2", help="only these event kinds")
    p.add_argument("--no-player", action="store_true", help="ignore player attribution")
    p.add_argument("--json", metavar="FILE", help="also write the report as JSON")
    p.add_argument("--out", metavar="FILE", help="also write the markdown report")
    p.set_defaults(func=cmd_eval)

    # ingest / train / promote: the learning loop (courtvision/learning, docs/TRAINING.md §7)
    p = sub.add_parser("ingest", parents=[cfgp], help="video + play-by-play -> an aligned, labelled game dataset",
                       description="Run a game video through perception and the event engine, fit the clock "
                                   "timeline from the scoreboard / scorebug reads, align the play-by-play, gate the "
                                   "game (median |residual| on made FGs < 0.5 s) and write a game dataset (cached "
                                   "tracks, labels, clips, cue and shot rows, manifest with rights). --sim-seed "
                                   "ingests a simulated game with a pseudo play-by-play instead (no footage needed).")
    p.add_argument("--video", metavar="PATH", help="game video (real footage)")
    p.add_argument("--pbp", metavar="FILE", help="play-by-play: nba_api PlayByPlayV3 / live JSON, or our CSV / JSON schema")
    p.add_argument("--sim-seed", type=int, default=None, help="ingest a simulated game instead (pseudo-PBP from its truth)")
    p.add_argument("--mode", choices=SIM_MODES, default="mono", help="--sim-seed: sim mode (default mono)")
    p.add_argument("--seconds", type=_positive_float, default=None, help="--sim-seed: game length (default sim.seconds)")
    p.add_argument("--game-id", default=None, help="dataset name (default: video stem / sim-<mode>-<seed>)")
    p.add_argument("--out", metavar="DIR", default="datasets", help="datasets root (default datasets/)")
    p.add_argument("--rights", choices=("own", "licensed", "research_only"), default="research_only",
                   help="rights of the footage (default research_only: NBA footage is R&D only; simulated games: own)")
    p.add_argument("--training-allowed", action="store_true", help="the game's owner consented to training on it")
    p.add_argument("--gold", action="store_true", help="a frozen evaluation game (never trains)")
    p.add_argument("--broadcast", action="store_true", help="TV footage: court tracking, replay gating, scorebug")
    p.add_argument("--scorebug-layout", metavar="FILE", help="ScorebugLayout JSON for --broadcast")
    _perception_args(p)
    p.add_argument("--league", choices=LEAGUES, default=None, help="court + rules (default: game.league)")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("train", help="train a candidate model bundle from ingested games",
                       description="Tune the event engine on the games' cached tracks against their PBP labels, refit "
                                   "the cue and shot models, and write a model bundle (pipeline.model_bundle). Only "
                                   "accepted, non-gold games train; the bundle records its games and rights.")
    p.add_argument("datasets", nargs="+", metavar="DATASET", help="game dataset directories (or roots holding them)")
    p.add_argument("--out", metavar="DIR", required=True, help="candidate bundle directory")
    p.add_argument("--base", metavar="BUNDLE", help="start from this bundle (e.g. models/current)")
    p.add_argument("--rounds", type=int, default=1, help="coordinate-search passes (default 1)")
    p.add_argument("--params", metavar="P1,P2", help="tune only these engine settings (default: the search space)")
    p.add_argument("--glyph-bank", metavar="FILE", help="scorebug GlyphBank JSON to ship in the bundle")
    p.add_argument("--workers", type=int, default=None, help="parallel games (default: CPU count)")
    p.add_argument("--no-cues", action="store_true", help="do not refit the cue models")
    p.add_argument("--no-shots", action="store_true", help="do not refit the shot model")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("promote", help="promote a candidate bundle if it beats the current one on gold games",
                       description="Score the candidate and the current bundle on frozen gold games; the candidate "
                                   "replaces <models>/current only with a macro-F1 gain and no per-kind regression "
                                   "(the ship channel also needs shippable rights). Every decision is logged in "
                                   "<models>/promotions.jsonl.")
    p.add_argument("candidate", metavar="BUNDLE", help="candidate bundle directory (courtvision train)")
    p.add_argument("--gold", nargs="+", required=True, metavar="DATASET", help="gold game datasets")
    p.add_argument("--models", metavar="DIR", default="models", help="models root (default models/)")
    p.add_argument("--channel", choices=("research", "ship"), default="research")
    p.add_argument("--min-gain", type=float, default=None, help="required macro-F1 gain (default 0.005)")
    p.add_argument("--dry-run", action="store_true", help="decide and print, change nothing")
    p.set_defaults(func=cmd_promote)

    # config
    p = sub.add_parser("config", parents=[cfgp], help="print the effective YAML configuration")
    p.add_argument("--default", action="store_true", help="print the commented default.yaml instead")
    p.add_argument("--section", metavar="NAME", help="only this top-level section")
    p.set_defaults(func=cmd_config)

    # latency
    p = sub.add_parser("latency", help="cloud latency harness (real, billed calls; needs credentials)",
                       description="Thin wrapper over tools/latency_harness.py: measures Claude (C1-C4) and "
                                   "Jev (J1-J5) latency to set every cloud deadline from the measured p95. "
                                   "Refuses to run without the credentials the selected configs need.")
    p.add_argument("--configs", default=",".join(LATENCY_CONFIGS), help="comma list of C1-C4, J1-J5")
    p.add_argument("--n", type=int, default=50, help="calls per configuration (default 50)")
    p.add_argument("--network", default="dev-mac", help="network label (dev-mac, gym-wifi, lte-hotspot, ...)")
    p.add_argument("--out", default="runs/latency", help="output directory")
    p.add_argument("--packet", default=None, help="FactPacket JSON to send instead of the built-in one")
    p.add_argument("--pause", type=_non_negative_float, default=0.5, help="seconds between calls")
    p.add_argument("--no-prewarm", action="store_true", help="skip the max_tokens=0 cache pre-warm")
    p.set_defaults(func=cmd_latency)
    return parser


# --------------------------------------------------------------------------- config helpers


def _ov(key: str, value: Any) -> str:
    """One ``key=value`` override; the value is JSON (a YAML subset), so paths with ':' stay strings."""
    return f"{key}={json.dumps(value)}"


def load_cli_config(args: argparse.Namespace, flags: Iterable[str] = ()) -> AppConfig:
    """``--config`` file <- ``--set`` overrides <- the command's explicit flags."""
    from courtvision.config import load_config

    return load_config(getattr(args, "config", None), [*(getattr(args, "set", None) or []), *flags])


def _offline_jev_default(cfg: AppConfig, explicit: str | None) -> None:
    """Offline runs: Jev off unless a mode was chosen (flag, --set or file). Source time races ahead
    of wall time, so live answers would make offline results timing-dependent."""
    from courtvision.decision.client import JevConfig

    if explicit is None and cfg.jev.mode == JevConfig().mode:
        cfg.jev.mode = "off"


def _sim_flags(args: argparse.Namespace) -> list[str]:
    out = []
    if getattr(args, "seconds", None) is not None:
        out.append(_ov("sim.seconds", float(args.seconds)))
    if getattr(args, "seed", None) is not None:
        out.append(_ov("sim.seed", int(args.seed)))
    if getattr(args, "league", None):
        out.append(_ov("game.league", args.league))
    return out


def _sim_config(cfg: AppConfig) -> Any:
    """``cfg.sim`` with the authoritative ``game.*`` league / rules / names (as ``make_source`` does)."""
    import dataclasses

    g = cfg.game
    return dataclasses.replace(cfg.sim, league=g.league, ruleset=g.ruleset, rule_overrides=list(g.rule_overrides),
                               home_name=g.home_name or cfg.sim.home_name, away_name=g.away_name or cfg.sim.away_name)


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


# --------------------------------------------------------------------------- detectors


@dataclass(frozen=True)
class DetectorSpec:
    """A parsed ``--detector`` value."""

    kind: str  # "colorblob" | "rfdetr" | "coreml" | "rnd-yolo" | "auto"
    arg: str | None = None  # weights path / variant / mlpackage

    def overrides(self) -> list[str]:
        """Config overrides for the detectors the pipeline builds itself (all but rnd-yolo)."""
        if self.kind == "auto":
            return [_ov("perception.detector", "auto")]
        if self.kind == "colorblob":
            # colour blobs only make sense on synthetic renders, whose digits the template reader reads
            return [_ov("perception.detector", "color_blob"), _ov("perception.jersey_reader", "template")]
        if self.kind == "rfdetr":
            out = [_ov("perception.detector", "rfdetr"), _ov("perception.rfdetr.backend", "torch")]
            if self.arg in RFDETR_VARIANTS:
                out.append(_ov("perception.rfdetr.variant", self.arg))
            elif self.arg:
                out.append(_ov("perception.rfdetr.weights", self.arg))
            return out
        if self.kind == "coreml":
            return [_ov("perception.detector", "rfdetr"), _ov("perception.rfdetr.backend", "coreml"),
                    _ov("perception.rfdetr.coreml_path", self.arg)]
        return []

    def check_available(self) -> None:
        """Fail early (CLIError) when the detector's optional extra or file is missing."""
        if self.kind == "rfdetr":
            if self.arg and self.arg not in RFDETR_VARIANTS and not Path(self.arg).exists():
                raise CLIError(f"RF-DETR weights not found: {self.arg}")
            if importlib.util.find_spec("rfdetr") is None:
                raise CLIError("the RF-DETR detector needs the `cv` extra: .venv/bin/pip install -e '.[cv]'")
        elif self.kind == "coreml":
            if not self.arg or not Path(self.arg).exists():
                raise CLIError(f"CoreML package not found: {self.arg}")
            if importlib.util.find_spec("coremltools") is None:
                raise CLIError("the CoreML detector needs coremltools (extra `cv`: pip install -e '.[cv]')")
        elif self.kind == "rnd-yolo":
            if not self.arg:
                raise CLIError("rnd-yolo needs weights: --detector rnd-yolo:<weights.pt>")
            if importlib.util.find_spec("ultralytics") is None:
                raise CLIError("rnd-yolo needs the R&D-only `rnd-agpl` extra (AGPL-3.0): pip install -e '.[rnd-agpl]'")

    def build(self) -> Any | None:
        """A ready Detector for kinds the pipeline does not build from config (rnd-yolo), else None.

        Shipped code never imports the AGPL adapter (D9): rnd-yolo goes through
        ``tools/rnd_detectors.py``, which exists only in a source checkout, loaded by path after the
        warning is printed."""
        if self.kind != "rnd-yolo":
            return None
        print(AGPL_WARNING, file=sys.stderr)
        return _load_tool("rnd_detectors", "R&D detectors").build_ultralytics(str(self.arg))


def parse_detector(text: str) -> DetectorSpec:
    """``colorblob`` | ``rfdetr[:<weights or nano|small|medium>]`` | ``coreml:<mlpackage>`` |
    ``rnd-yolo:<weights>`` | ``auto`` -> :class:`DetectorSpec` (ValueError on anything else)."""
    raw = (text or "").strip()
    kind, _, arg = raw.partition(":")
    kind = kind.strip().lower().replace("_", "")
    arg = arg.strip() or None
    if kind in ("colorblob", "blob"):
        if arg:
            raise ValueError("colorblob takes no argument")
        return DetectorSpec("colorblob")
    if kind == "auto":
        return DetectorSpec("auto")
    if kind == "rfdetr":
        return DetectorSpec("rfdetr", arg)
    if kind == "coreml":
        if not arg:
            raise ValueError("coreml needs a package path: coreml:<model.mlpackage>")
        return DetectorSpec("coreml", arg)
    if kind == "rnd-yolo":  # deliberately no shorter alias: opting into AGPL code must be explicit
        if not arg:
            raise ValueError("rnd-yolo needs weights: rnd-yolo:<weights.pt>")
        return DetectorSpec("rnd-yolo", arg)
    raise ValueError(f"unknown detector {text!r}; use colorblob, rfdetr[:<weights|nano|small|medium>], "
                     "coreml:<mlpackage> or rnd-yolo:<weights>")


def _check_rfdetr_weights(cfg: AppConfig, spec: DetectorSpec | None, real_video: bool) -> None:
    """Fail before loading anything when RF-DETR would have to download weights (never silently)."""
    if spec is not None and spec.kind == "rnd-yolo":
        return  # a prebuilt detector replaces the configured one
    pc = cfg.perception
    if not (pc.detector == "rfdetr" or (pc.detector == "auto" and real_video)):
        return
    from courtvision.perception.rfdetr_adapter import PretrainedDownloadNotAllowed, check_weights_available

    try:
        check_weights_available(pc.rfdetr)
    except PretrainedDownloadNotAllowed as exc:
        raise CLIError(f"{exc}\nFor synthetic renders use --detector colorblob.") from None


def _perception_flags(args: argparse.Namespace) -> list[str]:
    out = []
    if getattr(args, "allow_download", False):
        out.append(_ov("perception.rfdetr.allow_pretrained_download", True))
    if getattr(args, "calibration", None):
        if not Path(args.calibration).is_file():
            raise CLIError(f"calibration file not found: {args.calibration}")
        out.append(_ov("perception.calibration_path", str(args.calibration)))
    if getattr(args, "scoreboard_roi", None):
        if not Path(args.scoreboard_roi).is_file():
            raise CLIError(f"scoreboard ROI file not found: {args.scoreboard_roi}")
        out.append(_ov("perception.scoreboard_roi_path", str(args.scoreboard_roi)))
    if getattr(args, "roster", None):
        if not Path(args.roster).is_file():
            raise CLIError(f"roster file not found: {args.roster}")
        out.append(_ov("game.roster_path", str(args.roster)))
    spec: DetectorSpec | None = getattr(args, "detector", None)
    if spec is not None:
        out += spec.overrides()
    return out


def _check_roster(cfg: AppConfig) -> None:
    if cfg.game.roster_path:
        from courtvision.roster import load_roster

        load_roster(cfg.game.roster_path)  # RosterError (a ValueError) -> exit 2 with the entry named


# --------------------------------------------------------------------------- summaries


def _f(value: Any, nd: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


def _narration_line(result: RunResult) -> str:
    nar = result.narrator or {}
    jev = result.jev.get("mode") if isinstance(result.jev, Mapping) else None
    by = nar.get("by_source") or {}
    llm = "Claude + templates" if nar.get("candidates") is not None else "templates only"
    return (f"Narration: {llm} ({by.get('template', 0)} template / {by.get('llm', 0)} Claude lines); "
            f"Jev: {jev or result.config.jev.mode}")


def format_summary(result: RunResult, title: str = "CourtVision run") -> str:
    """A short markdown summary of a :class:`~courtvision.pipeline.RunResult` (the full report is
    ``report.md`` in the run directory)."""
    from courtvision.types import EventStatus

    lat = result.latency or {}
    lines = [f"# {title}: {result.source_kind}/{result.source_mode}", ""]
    if result.run_dir is not None:
        lines.append(f"- Run directory: `{result.run_dir}` (full report: `{result.run_dir / 'report.md'}`)")
    rt = lat.get("realtime_factor")
    lines.append(f"- Frames: {result.frames} ({_f(lat.get('source_s'), 1)} s of source in "
                 f"{_f(lat.get('wall_s'), 1)} s wall" + (f", {rt:.1f}x real time)" if rt else ")"))
    n_events = sum(1 for e in result.events if e.status is not EventStatus.RETRACTED)
    lines.append(f"- Events: {n_events}; utterances: {len(result.transcript)}; insights: {len(result.insights)}")
    lines.append(f"- {_narration_line(result)}")
    score = result.stats.score if result.stats is not None else {}
    if score:
        lines.append("- Final score: " + ", ".join(f"{t.value} {v}" for t, v in score.items())
                     + f" (source: {getattr(result.stats, 'score_source', 'n/a')})")
    ev = result.event_report
    if ev is not None:
        lines += ["", f"## Events vs truth (tolerance {ev.tol_s:g} s)", "",
                  f"Headline macro-F1 **{_f(ev.headline_macro_f1)}** (attributed {_f(ev.headline_attributed_f1)}) "
                  f"over {', '.join(ev.headline_kinds) or 'no kinds'}.", "",
                  "| kind | truth | pred | precision | recall | F1 | player acc |",
                  "|---|---|---|---|---|---|---|"]
        for kind, r in [*ev.per_kind.items(), *ev.separate.items()]:
            lines.append(f"| {kind} | {r.n_truth} | {r.n_pred} | {_f(r.precision)} | {_f(r.recall)} | {_f(r.f1)} "
                         f"| {_f(r.player_accuracy)} |")
        s = ev.shots
        lines += ["", f"Shots: made-FG F1 {_f(s.made_fg_f1)}, FG outcome accuracy {_f(s.fg_outcome_accuracy)} "
                      f"(n={s.fg_outcome_n}), 2 vs 3 accuracy {_f(s.value_accuracy)} (n={s.value_n})."]
    box = result.box_report
    if box is not None:
        lines += ["", "## Box score vs truth", "",
                  f"Mean per-stat MAE {_f(box.mean_mae)}; player points MAE "
                  f"{_f(box.per_stat['pts'].mae if 'pts' in box.per_stat else None)}; exact stat lines "
                  f"{_f(box.line_exact_rate)}; team points error {_f(box.team_points_abs_error)}; "
                  f"score match {'yes' if box.score_match else 'no'}."]
    tr = result.track_report
    if tr is not None:
        lines += ["", "## Tracking vs truth", "",
                  f"MOTA {_f(tr.mota)}, IDF1 {_f(tr.idf1)}, RMSE {_f(tr.rmse_ft, 2)} ft, ID switches {tr.idsw}, "
                  f"identity accuracy {_f(tr.identity_accuracy)} over {tr.n_frames} frames."]
    rows = [("perception (ms/frame)", "perception_ms_per_frame"), ("whole frame step (ms)", "frame_step_ms"),
            ("event t -> engine emission (ms)", "detection_delay_ms"),
            ("engine emission -> utterance (ms)", "event_to_utterance_ms"),
            ("event t -> utterance (ms)", "occurrence_to_utterance_ms")]
    shown = [(label, lat.get(key)) for label, key in rows if lat.get(key)]
    if shown:
        lines += ["", "## Latency", "", "| stage | n | p50 | p95 |", "|---|---|---|---|"]
        lines += [f"| {label} | {s['n']} | {s['p50']:.1f} | {s['p95']:.1f} |" for label, s in shown]
    return "\n".join(lines) + "\n"


def _print_result(result: RunResult, args: argparse.Namespace, title: str) -> None:
    if getattr(args, "json", False):
        summary = result.summary()
        summary["run_dir"] = str(result.run_dir) if result.run_dir else None
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(format_summary(result, title), end="")


# --------------------------------------------------------------------------- simulate


def simulate_config(mode: str, *, seed: int | None = None, seconds: float | None = None,
                    league: str | None = None, jev: str | None = None, config: str | None = None,
                    overrides: Sequence[str] = ()) -> AppConfig:
    """The configuration ``courtvision simulate`` runs with: ``config`` file <- ``overrides`` (``--set``) <-
    the explicit flags, and Jev off unless a mode was chosen. ``tools/accuracy_bench.py`` builds its runs
    with this same function, so the benchmark and the CLI share one code path."""
    if mode not in SIM_MODES:
        raise CLIError(f"unknown sim mode {mode!r} (choose from {', '.join(SIM_MODES)})")
    ns = argparse.Namespace(config=config, set=list(overrides), seconds=seconds, seed=seed, league=league)
    flags = [_ov("source.kind", "sim"), _ov("source.sim_mode", mode), *_sim_flags(ns)]
    if jev:
        flags.append(_ov("jev.mode", jev))
    cfg = load_cli_config(ns, flags)
    _offline_jev_default(cfg, jev)
    return cfg


def cmd_simulate(args: argparse.Namespace) -> int:
    from courtvision.pipeline import run_offline

    cfg = simulate_config(args.mode, seed=args.seed, seconds=args.seconds, league=args.league, jev=args.jev,
                          config=args.config, overrides=args.set or [])
    out = Path(args.out) if args.out else Path(cfg.pipeline.runs_dir) / (
        f"sim-{args.mode}-seed{cfg.sim.seed}-{_stamp()}")
    log.info("simulating %.0f s (seed %d, %s) in %s mode -> %s", cfg.sim.seconds, cfg.sim.seed, cfg.game.league,
             args.mode, out)
    result = run_offline(cfg, out_dir=out)
    _print_result(result, args, "CourtVision simulate")
    return EXIT_OK


# --------------------------------------------------------------------------- learning loop


def cmd_ingest(args: argparse.Namespace) -> int:
    from courtvision.learning.ingest import IngestConfig, ingest_game, sim_pbp
    from courtvision.pipeline import make_source
    from courtvision.training.align_pbp import load_pbp

    icfg = IngestConfig(rights_status=args.rights, training_allowed=args.training_allowed, gold=args.gold)
    if args.sim_seed is not None:
        if args.video or args.pbp:
            raise CLIError("--sim-seed ingests a simulated game: drop --video / --pbp")
        from courtvision.sim.simulator import GameSimulator

        cfg = simulate_config(args.mode, seed=args.sim_seed, seconds=args.seconds, league=args.league,
                              config=args.config, overrides=args.set or [])
        result = GameSimulator(_sim_config(cfg)).run()
        source = make_source(cfg, sim_result=result, realtime=False)
        pbp = sim_pbp(result.events, result.frames, icfg.sim_pbp_lag_s)
        game_id = args.game_id or f"sim-{args.mode}-{args.sim_seed}"
        if args.rights == "research_only" and not args.training_allowed:
            icfg.rights_status, icfg.training_allowed = "own", True  # simulator output is ours
        ds = ingest_game(cfg, source, pbp, Path(args.out) / game_id, game_id, icfg)
    else:
        if not args.video or not args.pbp:
            raise CLIError("ingest needs --video and --pbp (or --sim-seed for a simulated game)")
        video, pbp_path = Path(args.video), Path(args.pbp)
        for f in (video, pbp_path):
            if not f.is_file():
                raise CLIError(f"file not found: {f}")
        flags = [_ov("source.kind", "video"), _ov("source.path", str(video)), _ov("source.realtime", False),
                 *_perception_flags(args)]
        if args.league:
            flags.append(_ov("game.league", args.league))
        if args.broadcast:
            flags.append(_ov("perception.broadcast", "on"))
        if args.scorebug_layout:
            flags.append(_ov("perception.scorebug_layout_path", str(args.scorebug_layout)))
        cfg = load_cli_config(args, flags)
        _offline_jev_default(cfg, None)
        _check_roster(cfg)
        if not cfg.perception.calibration_path or not Path(cfg.perception.calibration_path).is_file():
            raise CLIError("ingest needs a calibration of the (game) camera: --calibration calibration.json")
        spec: DetectorSpec | None = args.detector
        if spec is not None:
            spec.check_available()
        _check_rfdetr_weights(cfg, spec, real_video=True)
        source = make_source(cfg, realtime=False)
        game_id = args.game_id or video.stem
        ds = ingest_game(cfg, source, load_pbp(pbp_path), Path(args.out) / game_id, game_id, icfg,
                         video=str(video), pbp_path=str(pbp_path))
    m = ds.manifest
    print(f"ingested {m.game_id} -> {ds.path}: {m.frames} frames, alignment "
          f"{'accepted' if m.accepted else 'REJECTED'} (median made-FG residual "
          f"{m.alignment.get('median_made_residual_s')} s over {m.alignment.get('n_made_checked')} makes), "
          f"rights {m.rights_status}{', gold' if m.gold else ''}")
    return EXIT_OK


def cmd_train(args: argparse.Namespace) -> int:
    from courtvision.learning.bundle import ModelBundle
    from courtvision.learning.dataset import load_datasets
    from courtvision.learning.train import SEARCH_SPACE, TrainConfig, train

    search = dict(SEARCH_SPACE)
    if args.params:
        names = [n.strip() for n in args.params.split(",") if n.strip()]
        unknown = [n for n in names if n not in SEARCH_SPACE]
        if unknown:
            raise CLIError(f"unknown --params {unknown}; tunable: {sorted(SEARCH_SPACE)}")
        search = {n: SEARCH_SPACE[n] for n in names}
    cfg = TrainConfig(search=search, rounds=args.rounds, workers=args.workers, fit_cues=not args.no_cues,
                      fit_shots=not args.no_shots)
    base = ModelBundle.load(args.base) if args.base else None
    bundle = train(load_datasets(args.datasets), args.out, cfg, base, args.glyph_bank)
    eng, start = bundle.metrics["engine"], bundle.metrics["engine_start"]
    print(f"bundle {bundle.id} ({bundle.rights}) -> {bundle.path}: engine macro F1 "
          f"{start['macro_f1'] or 0:.4f} -> {eng['macro_f1'] or 0:.4f} over {eng['games']} game(s); "
          f"settings {bundle.engine_overrides}")
    return EXIT_OK


def cmd_promote(args: argparse.Namespace) -> int:
    from courtvision.learning.bundle import ModelBundle
    from courtvision.learning.dataset import load_datasets
    from courtvision.learning.promote import PromoteConfig, promote

    pc = PromoteConfig(channel=args.channel)
    if args.min_gain is not None:
        pc.min_gain = float(args.min_gain)
    d = promote(ModelBundle.load(args.candidate), args.models, load_datasets(args.gold), pc, dry_run=args.dry_run)
    c, b = d["candidate_score"]["macro_f1"] or 0.0, d["current_score"]["macro_f1"] or 0.0
    verdict = "PROMOTED" if d["promoted"] else "not promoted: " + "; ".join(d["reasons"])
    print(f"candidate {d['candidate']} vs current {d['current'] or 'defaults'} on {len(d['gold'])} gold game(s): "
          f"macro F1 {c:.4f} vs {b:.4f} -> {verdict}{' (dry run)' if args.dry_run else ''}")
    return EXIT_OK if d["promoted"] else EXIT_ERROR


# --------------------------------------------------------------------------- run (video file)


def cmd_run(args: argparse.Namespace) -> int:
    from courtvision.geometry.homography import Calibration
    from courtvision.pipeline import run_offline

    video = Path(args.video)
    if not video.is_file():
        raise CLIError(f"video file not found: {video}")
    flags = [_ov("source.kind", "video"), _ov("source.path", str(video)), _ov("source.realtime", False),
             *_perception_flags(args)]
    if args.league:
        flags.append(_ov("game.league", args.league))
    if args.start is not None:
        flags.append(_ov("source.start_s", float(args.start)))
    if args.duration is not None:
        flags.append(_ov("source.duration_s", float(args.duration)))
    if args.jev:
        flags.append(_ov("jev.mode", args.jev))
    cfg = load_cli_config(args, flags)
    _offline_jev_default(cfg, args.jev)
    _check_roster(cfg)
    calib_path = cfg.perception.calibration_path
    if not calib_path or not Path(calib_path).is_file():
        raise CLIError("run needs a calibration: --calibration calibration.json "
                       "(build one with `courtvision calibrate POINTS` or `courtvision calibrate --from-sim`)")
    calib = Calibration.load(calib_path)
    if calib.league != cfg.game.league:
        log.warning("calibration league %r differs from game.league %r", calib.league, cfg.game.league)
    spec: DetectorSpec | None = args.detector
    if spec is not None:
        spec.check_available()
    elif importlib.util.find_spec("rfdetr") is None and cfg.perception.detector in ("auto", "rfdetr"):
        raise CLIError("no detector available: real video uses RF-DETR (extra `cv`: pip install -e '.[cv]'); "
                       "for synthetic renders pass --detector colorblob")
    _check_rfdetr_weights(cfg, spec, real_video=True)
    detector = spec.build() if spec is not None else None
    out = Path(args.out) if args.out else Path(cfg.pipeline.runs_dir) / f"run-{video.stem}-{_stamp()}"
    result = run_offline(cfg, out_dir=out, detector=detector)
    if result.frames == 0:
        raise CLIError(f"no frames processed from {video} (unreadable video or calibration never applied)",
                       EXIT_ERROR)
    _print_result(result, args, "CourtVision run")
    truth = video.with_name(f"{video.stem}.truth_events.jsonl")
    if truth.is_file() and not args.json:
        print(f"\nCompare with the render's truth: courtvision eval {out / 'events.jsonl'} --truth {truth}")
    return EXIT_OK


# --------------------------------------------------------------------------- serve


@dataclass
class ServeApp:
    """What :func:`create_serve_app` built: the ASGI app, its game and hub."""

    app: Any
    game: LiveGame
    hub: Any


async def _run_game(game: LiveGame) -> None:
    try:
        await game.run()
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - the dashboard keeps serving the last state
        log.exception("the game stopped with an error; the dashboard keeps serving its last state")


async def stop_game(game: LiveGame, task: asyncio.Task[None] | None, timeout_s: float = 10.0) -> None:
    """Ask the game to stop (no narration tail) and wait for it to finish; cancel after ``timeout_s``."""
    game.request_stop(drain=False)
    if task is None:
        return
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
    except TimeoutError:
        log.warning("the game did not stop within %.0f s; cancelling it", timeout_s)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def attach_game(app: Any, game: LiveGame, stop_timeout_s: float = 10.0) -> None:
    """Run ``game`` as a task on the app's event loop for the app's lifetime (startup -> shutdown),
    inside the app's own lifespan (which binds the hub to that loop and closes clients last)."""
    inner = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(a: Any):  # type: ignore[no-untyped-def]
        async with inner(a) as state:
            task = asyncio.create_task(_run_game(game), name="courtvision-game")
            a.state.game_task = task
            try:
                yield state
            finally:
                await stop_game(game, task, stop_timeout_s)

    app.router.lifespan_context = lifespan
    app.state.game = game


def cmd_studio(args: argparse.Namespace) -> int:
    from courtvision.server.studio import create_studio_app

    cfg = load_cli_config(args)
    if not _is_loopback(args.host) and not args.auth_token:
        raise CLIError("Studio needs --auth-token when listening outside localhost.")
    app = create_studio_app(cfg, root=args.data_dir, auth_token=args.auth_token)
    url = f"http://{_url_host(args.host)}:{args.port}/"
    print(f"CourtVision Studio: {url}\nFootage and analysis: {Path(args.data_dir).resolve()}")
    server = build_uvicorn_server(app, args.host, args.port)
    return EXIT_OK if asyncio.run(_serve(server, url if args.open else None)) else EXIT_ERROR


def create_serve_app(cfg: AppConfig, *, source: Any = None, env: Mapping[str, str] | None = None,
                     auth_token: str | None = None, jev: Any = None, llm: Any = None, tts: Any = None,
                     detector: Any = None, stop_timeout_s: float = 10.0) -> ServeApp:
    """The dashboard app (``server.app.create_app``) with a real-time :class:`LiveGame` attached."""
    from courtvision.pipeline import LiveGame, make_source
    from courtvision.server.app import create_app
    from courtvision.server.hub import Hub

    hub = Hub(cfg.hub)
    src = source if source is not None else make_source(cfg, realtime=True)
    game = LiveGame(cfg, src, hub, realtime=True, jev=jev, llm=llm, tts=tts, env=env, detector=detector)
    app = create_app(hub, game.controller, auth_token=auth_token or None, config=cfg.server)
    attach_game(app, game, stop_timeout_s)
    return ServeApp(app=app, game=game, hub=hub)


def _is_loopback(host: str) -> bool:
    h = host.strip().strip("[]").lower()
    if h in ("localhost", "ip6-localhost"):
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _url_host(host: str) -> str:
    h = host.strip()
    if h in ("0.0.0.0", "::", ""):
        return "127.0.0.1"
    return f"[{h}]" if ":" in h and not h.startswith("[") else h


def serve_flags(args: argparse.Namespace) -> list[str]:
    """Config overrides for ``serve`` flags (also used by the tests)."""
    flags = [_ov("source.realtime", True), *_sim_flags(args), *_perception_flags(args)]
    if args.source:
        flags.append(_ov("source.kind", args.source))
    if args.mode:
        flags.append(_ov("source.sim_mode", args.mode))
    if args.video:
        flags.append(_ov("source.path", str(args.video)))
        if not args.source:
            flags.append(_ov("source.kind", "video"))
    if args.camera is not None:
        flags.append(_ov("source.camera", str(args.camera)))
        if not args.source:
            flags.append(_ov("source.kind", "camera"))
    if args.speed is not None:
        flags.append(_ov("source.speed", float(args.speed)))
    if args.host:
        flags.append(_ov("serve.host", args.host))
    if args.port is not None:
        flags.append(_ov("serve.port", int(args.port)))
    if args.jev:
        flags.append(_ov("jev.mode", args.jev))
    if args.no_llm:
        flags.append(_ov("narration.llm_enabled", False))
    if args.tts:
        flags.append(_ov("narration.tts", args.tts))
    return flags


def unused_serve_flags(args: argparse.Namespace, cfg: AppConfig) -> list[str]:
    """``serve`` flags that the chosen source never reads (each with the reason), so they are not
    silently ignored: the simulator makes its own roster, only ``--mode video`` runs perception,
    and ``--seconds`` / ``--seed`` only drive the simulator."""
    kind = cfg.source.kind
    out = []
    if kind == "sim":
        if getattr(args, "roster", None):
            out.append("--roster (the simulator makes its own roster)")
        if cfg.source.sim_mode != "video":
            for flag, attr in (("--calibration", "calibration"), ("--scoreboard-roi", "scoreboard_roi"),
                               ("--detector", "detector")):
                if getattr(args, attr, None):
                    out.append(f"{flag} (sim mode {cfg.source.sim_mode} runs no perception; use --mode video)")
    else:
        for flag, attr in (("--seconds", "seconds"), ("--seed", "seed")):
            if getattr(args, attr, None) is not None:
                out.append(f"{flag} (only the simulator source uses it)")
    return out


def cmd_serve(args: argparse.Namespace) -> int:
    from courtvision.pipeline import anthropic_credential_source, typesafe_credential_source

    cfg = load_cli_config(args, serve_flags(args))
    _check_roster(cfg)
    for note in unused_serve_flags(args, cfg):
        print(f"courtvision serve: warning: ignoring {note}", file=sys.stderr)
    kind = cfg.source.kind
    if kind == "video" and not Path(cfg.source.path or "").is_file():
        raise CLIError(f"video file not found: {cfg.source.path}")
    spec: DetectorSpec | None = args.detector
    if spec is not None:
        spec.check_available()
    _check_rfdetr_weights(cfg, spec, real_video=kind in ("video", "camera"))
    if kind in ("video", "camera") and not cfg.perception.calibration_path:
        log.warning("no --calibration: perception waits until you calibrate in the dashboard's Calibration tab")
    token = args.auth_token or (os.environ.get(cfg.serve.auth_token_env) or "").strip() or None
    host, port = cfg.serve.host, cfg.serve.port
    if not _is_loopback(host) and token is None and not args.insecure_no_auth:
        raise CLIError(f"refusing to serve on {host} without an access token: the API carries rosters. Set "
                       f"${cfg.serve.auth_token_env} (or --auth-token), or pass --insecure-no-auth.")
    detector = spec.build() if spec is not None else None
    srv = create_serve_app(cfg, auth_token=token, detector=detector)
    game = srv.game
    url = f"http://{_url_host(host)}:{port}/"
    if game.llm is not None:
        claude = "on (dead-ball colour)"
    elif anthropic_credential_source() is None:
        claude = "off (no Anthropic credentials)"
    elif cfg.narration.use_jev_validator and not getattr(game.jev, "available", False):
        claude = "off (spoken Claude lines need the Jev line guard: TYPESAFE_API_KEY)"
    else:
        claude = "off"
    jev_mode = getattr(game.jev.client, "mode", None) if game.jev is not None else None
    jev_note = " (no TYPESAFE_API_KEY: fallbacks)" if cfg.jev.mode == "live" and typesafe_credential_source() is None else ""
    print(f"CourtVision dashboard: {url}")
    print(f"  source {game.source.kind}/{game.source.mode} at {cfg.source.speed:g}x; Claude {claude}; "
          f"Jev {jev_mode}{jev_note}; voice {cfg.narration.tts}; "
          f"auth {'bearer token required' if token else 'none (loopback only)' if _is_loopback(host) else 'NONE'}")
    if token:
        print(f"  open {url}#token=<your token> once to store it in the browser session")
    print("  Ctrl-C to stop.", flush=True)
    log_level = "warning" if args.quiet else ("debug" if args.verbose > 1 else "info")
    server = build_uvicorn_server(srv.app, host, port, log_level)
    open_url = f"{url}#token={token}" if token else url  # the hash never reaches the server or the logs
    # The simulator source holds the whole pre-simulated game (2.3 M objects for 48 min). Every
    # full collection then walked them all: ~0.5 s stalls of the event loop the game, the hub and
    # the WebSockets share. Freeze what exists now (the game's fixed data) out of later collections.
    gc.collect()
    gc.freeze()
    try:
        started = asyncio.run(_serve(server, open_url if args.open else None))
    except KeyboardInterrupt:
        return 130
    finally:
        gc.unfreeze()
    return EXIT_OK if started else EXIT_ERROR


_TOKEN_RE = re.compile(r"(token=)[^&\s\"']+")


class RedactTokens(logging.Filter):
    """Masks ``token=...`` in log lines (uvicorn logs WebSocket paths with their query string)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            text = record.getMessage()
        except Exception:  # noqa: BLE001 - a malformed record is left to the handler
            return True
        if "token=" in text:
            record.msg, record.args = _TOKEN_RE.sub(r"\1REDACTED", text), None
        return True


def build_uvicorn_server(app: Any, host: str, port: int, log_level: str = "info") -> Any:
    """``uvicorn.Server`` for the serve app: lifespan on (the game starts and stops with the server),
    a short graceful shutdown, and access / error logs with tokens redacted."""
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level=log_level, lifespan="on",
                            timeout_graceful_shutdown=5)
    for name in ("uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(name)
        if not any(isinstance(f, RedactTokens) for f in logger.filters):
            logger.addFilter(RedactTokens())
    return uvicorn.Server(config)


async def _serve(server: Any, open_url: str | None) -> bool:
    task = asyncio.create_task(server.serve())
    if open_url:
        while not server.started and not task.done():
            await asyncio.sleep(0.05)
        if server.started:
            import webbrowser

            await asyncio.to_thread(webbrowser.open, open_url)
    await task
    return bool(server.started)


# --------------------------------------------------------------------------- calibrate


def load_points_file(path: str | Path) -> dict[str, Any]:
    """Read a calibration points file (JSON or YAML) and normalise it to
    ``{"image_points", "court_points" | "landmarks", "image_size", "league", "rims", "ransac_thresh_ft"}``.

    Accepted shapes: ``"points": [{"image": [u, v], "landmark": name} | {"image": [u, v], "court": [x, y]}]``
    (one kind per file), or the ``POST /api/calibration`` arrays ``image_points`` + ``court_points`` /
    ``landmarks``. Optional: ``image_size`` [w, h], ``league``, ``rims`` {side: [x1, y1, x2, y2]},
    ``ransac_thresh_ft``."""
    import yaml

    p = Path(path)
    if not p.is_file():
        raise CLIError(f"points file not found: {p}")
    try:
        data = json.loads(p.read_text()) if p.suffix.lower() == ".json" else yaml.safe_load(p.read_text())
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise CLIError(f"{p}: cannot parse points file: {exc}") from exc
    if not isinstance(data, Mapping):
        raise CLIError(f"{p}: expected a mapping with 'points' or 'image_points'")
    out: dict[str, Any] = {k: data[k] for k in ("image_size", "league", "rims", "ransac_thresh_ft") if k in data}
    if data.get("points") is not None:
        rows = data["points"]
        if not isinstance(rows, list):
            raise CLIError(f"{p}: 'points' must be a list")
        image, court, names = [], [], []
        for i, row in enumerate(rows):
            if not isinstance(row, Mapping) or "image" not in row:
                raise CLIError(f"{p}: points[{i}] needs 'image': [u, v]")
            image.append(row["image"])
            if row.get("landmark") is not None:
                names.append(str(row["landmark"]))
            elif row.get("court") is not None:
                court.append(row["court"])
            else:
                raise CLIError(f"{p}: points[{i}] needs 'landmark' or 'court'")
        if names and court:
            raise CLIError(f"{p}: use either landmark names or court points for every point, not both")
        out["image_points"] = image
        if names:
            out["landmarks"] = names
        else:
            out["court_points"] = court
    else:
        out["image_points"] = data.get("image_points")
        for key in ("landmarks", "court_points"):
            if data.get(key) is not None:
                out[key] = data[key]
    if not out.get("image_points"):
        raise CLIError(f"{p}: no image points")
    if "landmarks" not in out and "court_points" not in out:
        raise CLIError(f"{p}: give court_points or landmarks for the image points")
    return out


def points_file_text(league: str, image_size: Sequence[int], rows: Sequence[Mapping[str, Any]],
                     rims: Mapping[str, Sequence[float]] | None = None) -> str:
    """A points file as JSON with one point per line (what ``--points-out`` writes)."""
    body = ",\n".join(f"    {json.dumps(dict(r))}" for r in rows)
    out = [
        "{",
        f'  "league": {json.dumps(league)},',
        f'  "image_size": {json.dumps([int(v) for v in image_size])},',
        '  "points": [',
        body,
        "  ]" + ("," if rims else ""),
    ]
    if rims:
        out.append(f'  "rims": {json.dumps({k: list(v) for k, v in rims.items()})}')
    out.append("}")
    return "\n".join(out) + "\n"


def _bbox(value: Any, where: str) -> Any:
    from courtvision.types import BBox

    try:
        if isinstance(value, Mapping):
            return BBox(float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"]))
        x1, y1, x2, y2 = (float(v) for v in value)
        return BBox(x1, y1, x2, y2)
    except (KeyError, TypeError, ValueError) as exc:
        raise CLIError(f"{where}: a rim box is [x1, y1, x2, y2]") from exc


def _sim_landmarks(camera: Any, court: Any) -> tuple[list[list[float]], list[str]]:
    import numpy as np

    pts, names = [], []
    for name, (x, y) in court.landmarks().items():
        uv = camera.project(np.array([x, y, 0.0]))
        if np.all(np.isfinite(uv)) and bool(camera.in_image(uv, 5.0)[0]):
            pts.append([float(uv[0]), float(uv[1])])
            names.append(name)
    return pts, names


def cmd_calibrate(args: argparse.Namespace) -> int:
    import numpy as np

    from courtvision.geometry.camera import PinholeCamera
    from courtvision.geometry.court import court_for
    from courtvision.geometry.homography import Calibration, Homography, fit_landmarks

    if bool(args.points) == bool(args.from_sim):
        raise CLIError("give a POINTS file or --from-sim (one of them)")
    flags = [_ov("game.league", args.league)] if args.league else []
    cfg = load_cli_config(args, flags)
    if args.from_sim:
        sim = _sim_config(cfg)
        camera = PinholeCamera.look_at(sim.camera_position, sim.camera_target, tuple(sim.image_size), sim.hfov_deg)
        court = court_for(sim.league)
        pts, names = _sim_landmarks(camera, court)
        request: dict[str, Any] = {"image_points": pts, "landmarks": names}
        image_size = tuple(int(v) for v in sim.image_size)
        rims = {}
        for side in ("left", "right"):
            bx, by = court.basket_xy(side)
            rims[side] = camera.bbox_of_circle(bx, by, court.rim_height, court.rim_radius)
        league = sim.league
    else:
        if args.points_out or args.roi_out:
            raise CLIError("--points-out / --roi-out need --from-sim")
        request = load_points_file(args.points)
        league = args.league or request.get("league") or cfg.game.league
        size = args.image_size or request.get("image_size")
        if not size:
            raise CLIError("the points file has no image_size: pass --image-size W H")
        image_size = (int(size[0]), int(size[1]))
        rims = {str(side): _bbox(box, f"rims.{side}") for side, box in (request.get("rims") or {}).items()}
        court = court_for(league)
    thresh = args.ransac_thresh if args.ransac_thresh is not None else float(request.get("ransac_thresh_ft", 1.0))
    img = np.asarray(request["image_points"], dtype=float)
    if img.ndim != 2 or img.shape[1] != 2 or len(img) < 4:
        raise CLIError("need >= 4 image points [u, v]")
    if request.get("landmarks"):
        hom, mask, rmse = fit_landmarks(court, img, list(request["landmarks"]), ransac_thresh_ft=thresh)
    else:
        court_pts = np.asarray(request["court_points"], dtype=float)
        if court_pts.shape != img.shape:
            raise CLIError(f"{len(court_pts)} court points for {len(img)} image points")
        hom, mask, rmse = Homography.fit(img, court_pts, ransac_thresh_ft=thresh)
    calib = Calibration(homography=hom, image_size=image_size, league=league, rims=rims)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    calib.save(out)
    inliers = int(np.asarray(mask).sum())
    print(f"rmse {float(rmse):.3f} ft over {inliers}/{len(img)} inlier points ({league}, "
          f"{image_size[0]}x{image_size[1]}, rims: {', '.join(sorted(rims)) or 'none'}) -> {out}")
    if float(rmse) > 1.0:
        print("warning: rmse above 1 ft; check the clicked points and landmark names", file=sys.stderr)
    if args.from_sim and args.points_out:
        rows = [{"image": [round(u, 1), round(v, 1)], "landmark": n} for (u, v), n in zip(pts, names)]
        rim_rows = {s: [round(b.x1, 1), round(b.y1, 1), round(b.x2, 1), round(b.y2, 1)] for s, b in rims.items()}
        Path(args.points_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.points_out).write_text(points_file_text(league, image_size, rows, rim_rows))
        print(f"points -> {args.points_out}")
    if args.from_sim and args.roi_out:
        from courtvision.sim.render import scoreboard_roi

        Path(args.roi_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.roi_out).write_text(json.dumps(scoreboard_roi(image_size), indent=2) + "\n")
        print(f"scoreboard ROI -> {args.roi_out}")
    return EXIT_OK


# --------------------------------------------------------------------------- render


def _video_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[Any, str]:
    """A ``cv2.VideoWriter`` and its codec: H.264 (``avc1``, falling back to ``mp4v``) for .mp4 / .m4v /
    .mov, lossless FFV1 otherwise (.avi / .mkv). Measured on a 60 s seed-7 render through
    ``courtvision run``: FFV1 and avc1 detect every shot; mp4v's artifacts lose the free throws."""
    import cv2

    codecs = ("avc1", "mp4v") if path.suffix.lower() in (".mp4", ".m4v", ".mov") else ("FFV1",)
    for fourcc in codecs:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), float(fps), size)
        if writer.isOpened():
            return writer, fourcc
        writer.release()
    raise CLIError(f"cannot open a {'/'.join(codecs)} video writer for {path} (try an .avi output)", EXIT_ERROR)


def cmd_render(args: argparse.Namespace) -> int:
    from courtvision.geometry.homography import Calibration
    from courtvision.roster import save_roster
    from courtvision.server.hub import dumps
    from courtvision.sim.render import SyntheticRenderer, scoreboard_roi
    from courtvision.sim.simulator import GameSimulator

    cfg = load_cli_config(args, _sim_flags(args))
    sim_cfg = _sim_config(cfg)
    start = float(args.start or 0.0)
    if start >= sim_cfg.seconds:
        raise CLIError(f"--start {start:g} is past the end of a {sim_cfg.seconds:g} s simulation")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = GameSimulator(sim_cfg).run()
    renderer = SyntheticRenderer(result.camera, result.court, cfg.render)
    size = tuple(int(v) for v in result.camera.image_size)
    i0 = min(len(result.frames), int(round(start * sim_cfg.fps)))
    writer, codec = _video_writer(out, sim_cfg.fps, size)  # type: ignore[arg-type]
    n = 0
    try:
        for i in range(i0, len(result.frames)):
            refs = result.referees[i] if i < len(result.referees) else None
            hands = result.hands[i] if i < len(result.hands) else None  # reaching arms (as SimSource.render)
            writer.write(renderer.render(result.frames[i], referees=refs, hands=hands))
            n += 1
    finally:
        writer.release()
    wall = time.perf_counter() - t0
    print(f"{n} frames ({n / sim_cfg.fps:.1f} s at {sim_cfg.fps:g} fps, {size[0]}x{size[1]}, {codec}, seed "
          f"{sim_cfg.seed}) -> {out} in {wall:.1f} s")
    if args.no_sidecars:
        return EXIT_OK
    stem = out.with_suffix("")
    rims = {r.side: r.bbox for r in (result.frames[0].rims if result.frames else [])}
    calib = Calibration(homography=result.camera.floor_homography(), image_size=size, league=result.court.league,
                        rims=rims)
    calib_path = Path(f"{stem}.calibration.json")
    calib.save(calib_path)
    roster_path = save_roster(result.roster, f"{stem}.roster.yaml")
    roi_path = Path(f"{stem}.scoreboard_roi.json")
    roi_path.write_text(json.dumps(scoreboard_roi(size), indent=2) + "\n")
    t_lo = result.frames[i0].t - 1e-6 if i0 < len(result.frames) else math.inf
    offset = result.frames[i0].t if i0 < len(result.frames) else 0.0
    truth_path = Path(f"{stem}.truth_events.jsonl")
    with truth_path.open("w", encoding="utf-8") as fh:
        for e in result.events:
            if e.t >= t_lo:
                d = e.to_dict()
                d["t"] = max(0.0, round(e.t - offset, 6))  # video time of the rendered file starts at 0
                d["frame_idx"] = e.frame_idx - i0
                fh.write(dumps(d) + "\n")
    print(f"calibration -> {calib_path}\nroster -> {roster_path}\nscoreboard ROI -> {roi_path}\n"
          f"truth events -> {truth_path}")
    print(f"\nProcess it: courtvision run --video {out} --calibration {calib_path} --roster {roster_path} "
          f"--scoreboard-roi {roi_path} --detector colorblob")
    return EXIT_OK


# --------------------------------------------------------------------------- eval


def read_events_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Event dicts from a JSON-lines file (``Event.to_dict()`` per line; blank lines skipped)."""
    p = Path(path)
    if not p.is_file():
        raise CLIError(f"events file not found: {p}")
    rows = []
    with p.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CLIError(f"{p}:{n}: not JSON: {exc}") from exc
            if not isinstance(row, dict) or "kind" not in row or "t" not in row:
                raise CLIError(f"{p}:{n}: not an event (needs kind and t)")
            rows.append(row)
    return rows


def cmd_eval(args: argparse.Namespace) -> int:
    from courtvision.eval.events import match_events
    from courtvision.eval.report import format_report
    from courtvision.server.hub import dumps

    pred_path = Path(args.pred)
    if pred_path.is_dir():
        pred_path = pred_path / "events.jsonl"
    truth_path = Path(args.truth) if args.truth else pred_path.with_name("truth_events.jsonl")
    if not args.truth and not truth_path.is_file():
        raise CLIError(f"no truth file next to {pred_path}: pass --truth truth_events.jsonl")
    pred, truth = read_events_jsonl(pred_path), read_events_jsonl(truth_path)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()] if args.kinds else None
    # A run directory's own eval protocol (stream window, operator lag) so the re-score matches the run.
    window, ecfg = None, None
    saved = pred_path.with_name("report.json")
    if saved.is_file():
        try:
            proto = json.loads(saved.read_text()).get("eval", {}).get("events", {}) or {}
        except (OSError, ValueError):
            proto = {}
        if proto.get("window") is not None:
            window = tuple(proto["window"])
        if proto.get("operator_lag_s"):
            from courtvision.eval.events import EventEvalConfig

            ecfg = EventEvalConfig(operator_lag_s={str(k): float(v) for k, v in proto["operator_lag_s"].items()})
    try:
        report = match_events(pred, truth, kinds=kinds, tol_s=args.tol, match_player=not args.no_player,
                              config=ecfg, window=window)
    except (KeyError, TypeError, ValueError) as exc:
        raise CLIError(f"cannot evaluate: {exc}") from exc
    text = format_report(report)
    print(f"Predicted {len(pred)} event rows ({pred_path}) vs truth {len(truth)} ({truth_path}).\n")
    print(text, end="")
    if args.out:
        Path(args.out).write_text(text)
    if args.json:
        Path(args.json).write_text(dumps(report.to_dict()))
    return EXIT_OK


# --------------------------------------------------------------------------- config


def cmd_config(args: argparse.Namespace) -> int:
    import yaml

    from courtvision.config import config_to_yaml, dump_config, render_default_yaml

    if args.default:
        if args.config or args.set or args.section:
            raise CLIError("--default prints the shipped defaults; drop --config / --set / --section")
        print(render_default_yaml(), end="")
        return EXIT_OK
    cfg = load_cli_config(args)
    if args.section:
        data = dump_config(cfg)
        if args.section not in data:
            raise CLIError(f"unknown section {args.section!r}; sections: {', '.join(data)}")
        print(yaml.safe_dump({args.section: data[args.section]}, sort_keys=False, default_flow_style=False,
                             width=120), end="")
    else:
        print(config_to_yaml(cfg), end="")
    return EXIT_OK


# --------------------------------------------------------------------------- latency


def latency_credentials_missing(configs: Sequence[str], env: Mapping[str, str] | None = None) -> list[str]:
    """Credential names the selected harness configs need but cannot find (names only)."""
    from courtvision.pipeline import anthropic_credential_source, typesafe_credential_source

    missing = []
    if any(c.startswith("C") for c in configs) and anthropic_credential_source(env) is None:
        missing.append("Anthropic (ANTHROPIC_API_KEY or `ant auth login`)")
    if any(c.startswith("J") for c in configs) and typesafe_credential_source(env) is None:
        missing.append("TypeSafe (TYPESAFE_API_KEY)")
    return missing


def _load_tool(stem: str, what: str) -> Any:
    """Load ``tools/<stem>.py`` by path (source checkouts only; ``tools/`` is not part of the package)."""
    path = TOOLS_DIR / f"{stem}.py"
    if not path.is_file():
        raise CLIError(f"{what} not found at {path}: they are only available from a source checkout", EXIT_ERROR)
    name = f"courtvision_tools_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - a regular file always has a loader
        raise CLIError(f"cannot load {path}", EXIT_ERROR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses in the module resolve it through sys.modules
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_harness() -> Any:
    return _load_tool("latency_harness", "the latency harness")


def cmd_latency(args: argparse.Namespace, env: Mapping[str, str] | None = None,
                harness_main: Callable[[list[str]], int] | None = None) -> int:
    configs = [c.strip().upper() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in LATENCY_CONFIGS]
    if unknown or not configs:
        raise CLIError(f"unknown latency configs {unknown or configs}; choose from {', '.join(LATENCY_CONFIGS)}")
    if args.n <= 0:
        raise CLIError("--n must be > 0")
    missing = latency_credentials_missing(configs, env)
    if missing:
        raise CLIError("refusing to run the latency harness (it makes real, billed calls) without credentials: "
                       + "; ".join(missing))
    argv = ["--configs", ",".join(configs), "--n", str(args.n), "--network", args.network, "--out", args.out,
            "--pause", str(args.pause)]
    if args.packet:
        argv += ["--packet", args.packet]
    if args.no_prewarm:
        argv.append("--no-prewarm")
    run = harness_main or _load_harness().main
    return int(run(argv) or 0)


# --------------------------------------------------------------------------- entry point


def _setup_logging(args: argparse.Namespace) -> None:
    if args.quiet:
        level = logging.ERROR
    elif args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1 or getattr(args, "command", None) == "serve":
        level = logging.INFO
    else:
        level = logging.WARNING
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    root.setLevel(level)


def main(argv: Sequence[str] | None = None) -> int:
    """``courtvision`` entry point; returns the exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "func", None):
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    _setup_logging(args)
    from courtvision.config import ConfigError
    from courtvision.roster import RosterError

    try:
        return int(args.func(args) or 0)
    except CLIError as exc:
        print(f"courtvision {args.command}: error: {exc}", file=sys.stderr)
        return exc.exit_code
    except (ConfigError, RosterError, FileNotFoundError) as exc:
        print(f"courtvision {args.command}: error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except ValueError as exc:  # bad roster / calibration / video: say what, keep the trace for -v
        log.debug("details", exc_info=True)
        print(f"courtvision {args.command}: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
