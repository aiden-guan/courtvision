"""Integration layer: frame sources -> perception -> events -> stats -> insights -> narration -> hub.

Frame sources are async iterators of :class:`FrameItem` ``(frame_idx, t, image | None, FrameState | None)``:

* :class:`SimSource` replays a :class:`~courtvision.sim.simulator.SimResult` as ``"truth3d"`` (truth
  frames), ``"mono"`` (``NoiseModel`` observed copies, monocular ball) or ``"video"`` (rendered with
  ``SyntheticRenderer``; the game then runs ``PerceptionPipeline`` with ``ColorBlobDetector``, the sim
  camera's floor homography + rim boxes, and ``render.SCOREBOARD_ROI``).
* :class:`VideoFileSource` (``cv2.VideoCapture``, PTS-based ``t``) and :class:`CameraSource`
  (device index or URL, wall-clock ``t``).

:class:`LiveGame` orchestrates one game. Per frame: perception (a worker thread for image sources
when running in real time, so the asyncio loop never blocks) -> ``EventEngine.process`` -> event
enrichment -> ``StatsEngine.on_events`` / ``on_frame`` -> ``InsightEngine.observe_frame`` (every
frame) / ``update`` -> ``Narrator.on_events`` / ``on_insights`` / ``tick`` (with the engine's
``DeadBallDetector``) -> hub messages (frames throttled, stats ~1 Hz, events / utterances / insights /
captions at once). Simulated operator entries (lineups, and the kinds the CV engine never infers)
are applied at their time. :class:`GameController` implements the server's ``Controller`` protocol.

Integration rules honoured here (module reviewers' notes):

* The sim camera is ``geometry.camera.RECOMMENDED_SIM_CAMERA`` (``SimConfig``'s default already uses
  it; :func:`make_source` warns when a configured camera does not see the full court).
* ``FrameState.scoreboard`` is set only on fresh reads (5 Hz OCR). The game keeps the last confirmed
  reading (``EventEngine.scoreboard`` confirmed score / period + last observed clocks) for clock
  display, the narrator's facts and the hub ``frame`` message; the engine and stats only ever see
  fresh reads.
* P(3) gate: the engine puts ``p_three`` on the SHOT_ATTEMPT only and the narrator usually receives
  the make in a later batch, so the enricher copies ``p_three`` onto SHOT_MADE / SHOT_MISSED of the
  same ``shot_id`` (``narration.facts.event_trusted`` then applies the 0.97 three gate).
* Identity tiers: events carry ``attrs["id_conf"]`` / ``attrs["jersey_conf"]`` (perception: the
  actor's ``PlayerState.confidence`` and the jersey vote share; sim / manual / feed: 1.0), so names
  are spoken only at >= 0.95. With two actors the lower confidence is used (the narrator applies one
  value to both).
* One ``PseudonymMap`` per game, ``narration.facts.pseudonyms_for(roster, seed)``, shared by facts,
  candidates and rendering.
* Priorities: ``Insight.priority`` is "higher = more important" in ``stats.insights``,
  ``JevDecider.pick_insight`` (fallback = max) and the narrator's coach board; the director's
  ``Cue.priority`` is 0 = highest (0..3) and speech ``Utterance.priority = 3 - cue.priority``. The
  pipeline passes insights through unchanged and orders them with :func:`insight_order`. Insights are
  coach-board cards only: they never become director cues (live calls).
* Claude runs rarely (timeouts, period breaks): the prompt cache uses a 1 h TTL and is pre-warmed at
  start and periodically. No Anthropic credentials => Claude disabled (template-only); no
  ``TYPESAFE_API_KEY`` => Jev "off" (deterministic fallbacks). Credentials are detected by name only,
  never read or printed.

Clocks: in real time the narrator runs on a :class:`SpeedClock` (``time.monotonic`` at the playback speed,
so its holds and TTLs stay in video seconds at any speed). Offline (``realtime=False``) every
component runs on a :class:`VirtualClock` that follows source time, TTS lines occupy source time
(:class:`VideoTimeTTS`) and perception runs inline, so offline runs are deterministic and their
transcripts match what a real-time run would say.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
import os
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

import numpy as np

from courtvision.config import AppConfig, config_to_yaml
from courtvision.decision.advisor import JevDecider
from courtvision.decision.client import JevClient, JevConfig
from courtvision.eval.boxscore import BoxReport, compare_box, edge_contributions
from courtvision.eval.events import EventEvalConfig, EventReport, match_events
from courtvision.eval.report import format_report, report_dict
from courtvision.eval.tracking import TrackReport, tracking_report
from courtvision.events.engine import EventEngine
from courtvision.geometry.camera import sees_full_court
from courtvision.geometry.court import CourtSpec, court_for
from courtvision.geometry.homography import Calibration, Homography, fit_landmarks
from courtvision.geometry.rules import RuleSpec, rules_for
from courtvision.narration.captions import CaptionEmitter, SidecarSink, WebVTTWriter
from courtvision.narration.candidates import CandidateConfig
from courtvision.narration.director import Director, DirectorConfig
from courtvision.narration.llm import ClaudeConfig, ClaudeNarrationLLM, NarrationLLM
from courtvision.narration.narrator import Narrator
from courtvision.narration.pseudonym import pseudonyms_for, team_from_player_id
from courtvision.narration.tts import TTS, NullTTS, SpokenLine, TTSConfig, estimate_duration_s, make_tts
from courtvision.roster import load_roster
from courtvision.server.hub import Hub, dumps, frame_msg, meta_msg, stats_msg, status_msg
from courtvision.sim import palette
from courtvision.sim.noise import NoiseConfig, NoiseModel, ScoreboardConfig, ScoreboardModel
from courtvision.sim.operator import SimOperator, SimOperatorConfig
from courtvision.sim.render import RenderConfig, SyntheticRenderer, scoreboard_roi
from courtvision.sim.simulator import GameSimulator, SimResult
from courtvision.stats.decision import DecisionModel
from courtvision.stats.engine import GameStats, StatsEngine
from courtvision.stats.insights import InsightEngine
from courtvision.stats.shot_quality import ShotQualityModel
from courtvision.types import (
    Event,
    EventKind,
    EventStatus,
    FrameState,
    Insight,
    Roster,
    ScoreboardReading,
    Team,
    Utterance,
)

__all__ = [
    "CameraSource",
    "FrameItem",
    "FrameSource",
    "GameController",
    "LiveGame",
    "MuteableTTS",
    "Pacer",
    "RunResult",
    "SimSource",
    "TieredPseudonyms",
    "VideoFileSource",
    "VideoTimeTTS",
    "SpeedClock",
    "VirtualClock",
    "anthropic_credential_source",
    "arun_offline",
    "build_jev",
    "build_llm",
    "director_overrides",
    "hub_stats_msg",
    "insight_order",
    "load_roster",
    "make_source",
    "run_offline",
    "typesafe_credential_source",
]

log = logging.getLogger(__name__)

T = TypeVar("T")

SIM_MODES = ("truth3d", "mono", "video", "broadcast")
_TRUSTED_SOURCES = frozenset({"sim", "manual", "feed"})  # identity is known: no tiering
_QUIET_KINDS = frozenset({EventKind.DRIBBLE, EventKind.PASS})  # do not trigger snapshots / insight updates alone


# --------------------------------------------------------------------------- credentials (names only)


def anthropic_credential_source(env: Mapping[str, str] | None = None, config_dir: str | Path | None = None) -> str | None:
    """Where Anthropic credentials would come from, or None. Never reads or returns a secret.

    Resolution order of the SDK: ``ANTHROPIC_API_KEY`` -> ``ANTHROPIC_AUTH_TOKEN`` -> an
    ``ant auth login`` profile (``$ANTHROPIC_CONFIG_DIR`` or ``~/.config/anthropic``,
    ``credentials/<profile>.json``) -> Workload Identity Federation env vars.
    """
    env = os.environ if env is None else env
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        if (env.get(name) or "").strip():
            return name
    base = Path(config_dir) if config_dir is not None else Path(
        env.get("ANTHROPIC_CONFIG_DIR") or Path(env.get("HOME") or Path.home()) / ".config" / "anthropic")
    creds = base / "credentials"
    profile = (env.get("ANTHROPIC_PROFILE") or "").strip()
    try:
        if profile:
            if (creds / f"{profile}.json").is_file():
                return f"profile:{profile}"
        elif creds.is_dir() and any(p.suffix == ".json" for p in creds.iterdir()):
            return "profile"
    except OSError:
        pass
    wif = ("ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID", "ANTHROPIC_SERVICE_ACCOUNT_ID")
    if all((env.get(n) or "").strip() for n in wif) and (
            (env.get("ANTHROPIC_IDENTITY_TOKEN_FILE") or "").strip() or (env.get("ANTHROPIC_IDENTITY_TOKEN") or "").strip()):
        return "workload_identity"
    return None


def typesafe_credential_source(env: Mapping[str, str] | None = None) -> str | None:
    """``"TYPESAFE_API_KEY"`` when that variable is set (never its value), else None."""
    env = os.environ if env is None else env
    return "TYPESAFE_API_KEY" if (env.get("TYPESAFE_API_KEY") or "").strip() else None


# --------------------------------------------------------------------------- priorities


def insight_order(insights: Iterable[Insight]) -> list[Insight]:
    """Most important first: ``Insight.priority`` is higher = more important everywhere
    (``InsightEngine.active``, ``JevDecider.pick_insight`` fallback, the narrator's coach board)."""
    return sorted(insights, key=lambda i: (-(i.priority or 0), -i.t, i.insight_id))


# --------------------------------------------------------------------------- frame sources


class FrameItem(NamedTuple):
    """One source frame: an image to perceive, or a ready ``FrameState`` (sim truth / noise)."""

    frame_idx: int
    t: float
    image: np.ndarray | None
    state: FrameState | None


class FrameSource:
    """Base class. Subclasses implement :meth:`_iterate`; ``async for item in source`` yields
    :class:`FrameItem`. ``realtime`` asks the game to pace frames at ``t / speed``; ``live`` sources
    (cameras) arrive in real time on their own."""

    kind = "base"
    mode = "frames"
    synthetic = False  # images come from the synthetic renderer (colour-blob detector, template jerseys)

    def __init__(self, realtime: bool = False, speed: float = 1.0) -> None:
        self.realtime = bool(realtime)
        self.live = False
        self.speed = float(speed)
        self.fps = 30.0
        self.image_size: tuple[int, int] | None = None
        self.roster: Roster | None = None
        self.court: CourtSpec | None = None
        self.rules: RuleSpec | None = None
        self.truth: SimResult | None = None
        self.start_t = 0.0
        self.end_t: float | None = None

    def __aiter__(self) -> AsyncIterator[FrameItem]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[FrameItem]:  # pragma: no cover - abstract
        raise NotImplementedError
        yield  # noqa: unreachable - makes this an async generator

    def bind(self, court: CourtSpec, rules: RuleSpec, roster: Roster) -> None:
        """Give sources without their own game data (video, camera) the configured court / rules / roster."""
        self.court = self.court or court
        self.rules = self.rules or rules
        self.roster = self.roster or roster

    # hints for perception on synthetic renders (None / {} for real footage)
    def calibration(self) -> Calibration | None:
        return None

    def scoreboard_roi(self) -> dict[str, Any] | None:
        return None

    def jersey_candidates(self) -> list[str] | None:
        return None

    def perception_hints(self) -> dict[str, Any]:
        return {}

    def render(self, frame_idx: int) -> np.ndarray | None:
        """An image of ``frame_idx`` for sources without images (sim truth modes); None otherwise."""
        return None

    def truth_frames(self) -> list[FrameState] | None:
        return None

    def close(self) -> None:
        return None


class SimSource(FrameSource):
    """Replays a simulation: ``mode`` "truth3d" | "mono" | "video" (see the module docstring)."""

    kind = "sim"

    def __init__(self, result: SimResult, mode: str = "truth3d", noise: NoiseConfig | None = None,
                 realtime: bool = False, speed: float = 1.0, *, noise_seed: int = 1, start_s: float = 0.0,
                 duration_s: float | None = None, scoreboard: str = "auto",
                 render_config: RenderConfig | None = None, board: ScoreboardConfig | None = None) -> None:
        super().__init__(realtime, speed)
        if mode not in SIM_MODES:
            raise ValueError(f"SimSource mode must be one of {SIM_MODES}, got {mode!r}")
        if scoreboard not in ("auto", "truth", "operator", "ocr_5hz", "none"):
            raise ValueError(f"scoreboard must be 'auto', 'truth', 'operator', 'ocr_5hz' or 'none', got {scoreboard!r}")
        if scoreboard == "auto":  # truth3d keeps the exact board; mono / video see the operator-run board
            scoreboard = "truth" if mode == "truth3d" else "operator"
        self.mode = mode
        self.synthetic = mode in ("video", "broadcast")
        self.result = self.truth = result
        self.roster, self.court, self.rules = result.roster, result.court, result.rules
        self.fps = float(result.config.fps)
        self.image_size = tuple(result.camera.image_size)  # type: ignore[assignment]
        self.scoreboard_policy = scoreboard
        n = len(result.frames)
        self._i0 = min(n, max(0, int(round(start_s * self.fps))))
        self._i1 = n if duration_s is None else min(n, self._i0 + max(0, int(round(duration_s * self.fps))))
        if self._i1 > self._i0:
            self.start_t = result.frames[self._i0].t
            self.end_t = result.frames[self._i1 - 1].t
        self._render_config = render_config
        self._renderer: SyntheticRenderer | None = None
        self._noise_seed = noise_seed
        self._broadcast: Any = None  # (BroadcastPlan, BroadcastRenderer), built on first use
        self._observed: list[FrameState] | None = None
        if mode == "mono":
            ncfg = dataclasses.replace(noise or NoiseConfig(), monocular=True, fps=self.fps)
            feet = result.feet_z[self._i0:self._i1] if result.feet_z else None
            # (referees and substitutes waiting at the table: people a detector can mistake for players)
            self._observed = NoiseModel(ncfg, result.camera).apply(
                result.frames[self._i0:self._i1], seed=noise_seed, feet_z=feet,
                referees=result.referees[self._i0:self._i1] or None, waiting=result.waiting[self._i0:self._i1] or None)
        self._ocr_every = max(1, int(round(self.fps / 5.0)))  # D14: 5 Hz
        # The operator-run venue board (sim.noise.ScoreboardModel): OCR readings for mono, the displayed
        # board (MM:SS panel) for video renders. Computed over the whole game so lags carry across a window.
        self._board_reads: list[ScoreboardReading | None] | None = None
        self._board_shown: list[ScoreboardReading | None] | None = None
        if scoreboard == "operator":
            model = ScoreboardModel(board)
            boards = [f.scoreboard for f in result.frames]
            if mode == "video":
                self._board_shown = model.displayed(boards, self.fps, seed=noise_seed, tenths=False)
            elif mode == "broadcast":  # the scorebug shows tenths in the last minute
                self._board_shown = model.displayed(boards, self.fps, seed=noise_seed, tenths=True)
            else:
                self._board_reads = model.readings(boards, self.fps, seed=noise_seed)

    @classmethod
    def simulate(cls, sim_config: Any, mode: str = "truth3d", **kwargs: Any) -> SimSource:
        """Run ``GameSimulator(sim_config)`` and wrap the result."""
        return cls(GameSimulator(sim_config).run(), mode, **kwargs)

    @property
    def window(self) -> tuple[int, int]:
        return self._i0, self._i1

    def _board(self, i: int, state: FrameState) -> FrameState:
        policy = self.scoreboard_policy
        if policy == "operator":
            if self._board_shown is not None:  # video: the panel shows the operator-run board
                return dataclasses.replace(state, scoreboard=self._board_shown[i])
            return dataclasses.replace(state, scoreboard=self._board_reads[i] if self._board_reads else None)
        if policy == "truth" or state.scoreboard is None:
            return state
        if policy == "ocr_5hz" and (i - self._i0) % self._ocr_every == 0:
            return state
        return dataclasses.replace(state, scoreboard=None)

    def renderer(self) -> SyntheticRenderer:
        if self._renderer is None:
            self._renderer = SyntheticRenderer(self.result.camera, self.result.court, self._render_config)
        return self._renderer

    def broadcast(self) -> Any:
        """(BroadcastPlan, BroadcastRenderer) of the whole game (``sim/broadcast.py``; the scorebug shows the
        operator-run board)."""
        if self._broadcast is None:
            from courtvision.sim.broadcast import BroadcastConfig, BroadcastRenderer, plan_broadcast

            r = self.result
            plan = plan_broadcast(r.frames, self.fps, r.court.length, BroadcastConfig(seed=self._noise_seed))
            shown = [self._board(i, f) for i, f in enumerate(r.frames)]
            names = r.roster.team_names
            abbr = (names.get(Team.AWAY, "Away")[:3].upper(), names.get(Team.HOME, "Home")[:3].upper())
            self._broadcast = (plan, BroadcastRenderer(shown, r.court, plan, r.feet_z or None, r.referees or None,
                                                       self._render_config, abbr=abbr, seed=self._noise_seed))
        return self._broadcast

    def render(self, frame_idx: int) -> np.ndarray | None:
        if not 0 <= frame_idx < len(self.result.frames):
            return None
        if self.mode == "broadcast":
            return self.broadcast()[1].render(frame_idx)
        truth = self._board(frame_idx, self.result.frames[frame_idx])
        refs = self.result.referees[frame_idx] if frame_idx < len(self.result.referees) else None
        waiting = self.result.waiting[frame_idx] if frame_idx < len(self.result.waiting) else []
        jersey = {e.player_id: e.jersey for e in self.result.roster.entries}
        extras = [(Team(team), x, y, jersey.get(pid)) for pid, team, x, y in waiting]  # substitutes at the table
        hands = self.result.hands[frame_idx] if frame_idx < len(self.result.hands) else None  # reaching arms
        return self.renderer().render(truth, referees=refs, extras=extras, hands=hands)

    async def _iterate(self) -> AsyncIterator[FrameItem]:
        for k, i in enumerate(range(self._i0, self._i1)):
            truth = self.result.frames[i]
            if self.mode in ("video", "broadcast"):
                yield FrameItem(i, truth.t, self.render(i), None)
            else:
                state = self._observed[k] if self._observed is not None else truth
                yield FrameItem(i, truth.t, None, self._board(i, state))

    def calibration(self) -> Calibration:
        if self.mode == "broadcast":  # the operator calibrates the game camera on the first game-camera frame
            plan = self.broadcast()[0]
            cam = next((f.camera for f in plan.frames[self._i0:] if f.kind == "live"), plan.frames[0].camera)
            rims = {}
            for side in ("left", "right"):
                bx, by = self.court.basket_xy(side)
                box = cam.bbox_of_circle(bx, by, self.court.rim_height, self.court.rim_radius)
                if np.all(np.isfinite([box.x1, box.y1, box.x2, box.y2])):
                    rims[side] = box
            return Calibration(homography=cam.floor_homography(), image_size=cam.image_size, league=self.court.league,
                               rims=rims)
        cam = self.result.camera
        rims = {r.side: r.bbox for r in (self.result.frames[0].rims if self.result.frames else [])}
        return Calibration(homography=cam.floor_homography(), image_size=cam.image_size, league=self.court.league,
                           rims=rims)

    def scoreboard_roi(self) -> dict[str, Any] | None:
        if self.mode == "broadcast":
            return None  # no venue board in view: the scorebug carries the score
        return scoreboard_roi(self.result.camera.image_size)

    def scorebug_layout(self) -> dict[str, Any] | None:
        if self.mode != "broadcast":
            return None
        from courtvision.sim.broadcast import scorebug_layout

        return scorebug_layout(self.result.camera.image_size)

    def jersey_candidates(self) -> list[str]:
        return sorted({e.jersey for e in self.result.roster.entries})

    def perception_hints(self) -> dict[str, Any]:
        return {"foot_radius_ft": float(self.result.config.player_radius_ft)}

    def truth_frames(self) -> list[FrameState]:
        return self.result.frames[self._i0:self._i1]


class VideoFileSource(FrameSource):
    """A video file through ``cv2.VideoCapture``; ``t`` is the decoder's PTS (CAP_PROP_POS_MSEC)."""

    kind = "video"
    mode = "video"

    def __init__(self, path: str | Path, realtime: bool = True, speed: float = 1.0, *, start_s: float = 0.0,
                 duration_s: float | None = None, fps: float | None = None, roster: Roster | None = None,
                 court: CourtSpec | None = None, rules: RuleSpec | None = None) -> None:
        super().__init__(realtime, speed)
        import cv2  # core dependency; local import keeps module import light

        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"video file not found: {self.path}")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise ValueError(f"cannot open video {self.path}")
        file_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = float(fps) if fps else (file_fps if file_fps > 0 else 30.0)
        w, h = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.image_size = (w, h) if w > 0 and h > 0 else None
        self.start_s, self.duration_s = float(start_s), duration_s
        self.start_t = self.start_s
        self.roster, self.court, self.rules = roster, court, rules
        self._lock = threading.Lock()

    def _read(self) -> tuple[bool, np.ndarray | None, float]:
        import cv2

        with self._lock:
            ok, img = self._cap.read()
            pos = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        return bool(ok), img if ok else None, pos

    async def _iterate(self) -> AsyncIterator[FrameItem]:
        import cv2

        if self.start_s > 0:
            with self._lock:
                self._cap.set(cv2.CAP_PROP_POS_MSEC, self.start_s * 1000.0)
        idx, last_t, t0 = 0, None, None
        dt = 1.0 / self.fps
        while True:
            ok, img, pos = await asyncio.to_thread(self._read)
            if not ok or img is None:
                break
            # PTS when the decoder reports one that advances, else the previous t + 1/fps.
            t = pos if math.isfinite(pos) and (last_t is None or pos > last_t + 1e-6) else (
                (last_t + dt) if last_t is not None else 0.0)
            if t0 is None:
                t0 = t
            if self.duration_s is not None and t - t0 > self.duration_s + 1e-9:
                break
            last_t = t
            self.end_t = t
            if self.image_size is None:
                self.image_size = (int(img.shape[1]), int(img.shape[0]))
            yield FrameItem(idx, float(t), img, None)
            idx += 1

    def close(self) -> None:
        with self._lock:
            self._cap.release()


class CameraSource(FrameSource):
    """A live camera (device index) or stream URL; ``t`` is wall-clock seconds since the first frame."""

    kind = "camera"
    mode = "video"

    def __init__(self, device: int | str = 0, *, roster: Roster | None = None, court: CourtSpec | None = None,
                 rules: RuleSpec | None = None, duration_s: float | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(realtime=True, speed=1.0)
        import cv2

        self.live = True
        if isinstance(device, str) and device.strip().isdigit():
            device = int(device.strip())
        self.device = device
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise ValueError(f"cannot open camera {device!r}")
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if fps > 0 else 30.0
        self.roster, self.court, self.rules = roster, court, rules
        self.duration_s = duration_s
        self._clock = clock
        self._lock = threading.Lock()

    def _read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            ok, img = self._cap.read()
        return bool(ok), img if ok else None

    async def _iterate(self) -> AsyncIterator[FrameItem]:
        idx, t0 = 0, None
        while True:
            ok, img = await asyncio.to_thread(self._read)
            if not ok or img is None:
                break
            now = self._clock()
            t0 = now if t0 is None else t0
            t = now - t0
            if self.duration_s is not None and t > self.duration_s:
                break
            if self.image_size is None:
                self.image_size = (int(img.shape[1]), int(img.shape[0]))
            self.end_t = t
            yield FrameItem(idx, t, img, None)
            idx += 1

    def close(self) -> None:
        with self._lock:
            self._cap.release()


def make_source(cfg: AppConfig, sim_result: SimResult | None = None, realtime: bool | None = None) -> FrameSource:
    """Build the configured source (``realtime`` overrides ``cfg.source.realtime``)."""
    s = cfg.source
    rt = s.realtime if realtime is None else bool(realtime)
    if s.kind == "sim":
        if sim_result is None:
            g = cfg.game
            sim_cfg = dataclasses.replace(
                cfg.sim, league=g.league, ruleset=g.ruleset, rule_overrides=list(g.rule_overrides),
                home_name=g.home_name or cfg.sim.home_name, away_name=g.away_name or cfg.sim.away_name)
            sim_result = GameSimulator(sim_cfg).run()
        if not sees_full_court(sim_result.camera, sim_result.court):
            # image boxes, rims and the monocular ball all come from this camera (renders too)
            log.warning("sim camera does not frame the whole court; use geometry.camera.RECOMMENDED_SIM_CAMERA")
        return SimSource(sim_result, s.sim_mode, cfg.noise, rt, s.speed, noise_seed=s.noise_seed, start_s=s.start_s,
                         duration_s=s.duration_s, scoreboard=s.scoreboard, render_config=cfg.render, board=cfg.board)
    if s.kind == "video":
        if not s.path:
            raise ValueError("source.path is required for a video source")
        return VideoFileSource(s.path, rt, s.speed, start_s=s.start_s, duration_s=s.duration_s, fps=s.fps)
    if s.kind == "camera":
        return CameraSource(s.camera, duration_s=s.duration_s)
    raise ValueError(f"unknown source kind {s.kind!r}")


# --------------------------------------------------------------------------- clocks, pacing, TTS


class VirtualClock:
    """Source-time clock for offline runs: follows the newest frame time (never goes back)."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def set(self, t: float) -> None:
        if t > self.t:
            self.t = float(t)

    def advance(self, dt: float) -> None:
        self.t += max(0.0, float(dt))


_BOARD_GROUPS = (("home_score", "away_score"), ("period",), ("game_clock",), ("shot_clock",))


class SpeedClock:
    """Real-time narrator clock that runs at the playback speed (``speed`` x monotonic time), so the
    narrator's holds, cooldowns and TTLs keep their meaning in video seconds when a file or sim plays faster
    or slower than real time (on wall time, at speed 2 a 3 s make hold spanned 6 s of video and the
    narration fell behind). Speed changes re-anchor it: it never jumps or runs backwards."""

    def __init__(self, speed: float = 1.0, clock: Callable[[], float] = time.monotonic) -> None:
        if not speed > 0:
            raise ValueError("speed must be > 0")
        self._clock = clock
        self._speed = float(speed)
        self._wall0 = clock()
        self._t0 = self._wall0

    @property
    def speed(self) -> float:
        return self._speed

    def __call__(self) -> float:
        return self._t0 + (self._clock() - self._wall0) * self._speed

    def set_speed(self, speed: float) -> None:
        if not speed > 0:
            raise ValueError("speed must be > 0")
        now = self._clock()
        self._t0 = self._t0 + (now - self._wall0) * self._speed
        self._wall0, self._speed = now, float(speed)


class Pacer:
    """Releases frames at ``t / speed`` against a monotonic clock, with pause / resume / speed.

    Speed changes and resumes re-anchor at the current frame so playback never bursts. Disabled
    pacing still honours pause. :meth:`wait` returns how late the frame is (seconds)."""

    def __init__(self, enabled: bool, speed: float = 1.0, clock: Callable[[], float] = time.monotonic) -> None:
        if speed <= 0:
            raise ValueError("speed must be > 0")
        self.enabled = bool(enabled)
        self.speed = float(speed)
        self.paused = False
        self.lag_s = 0.0
        self._clock = clock
        self._anchor: tuple[float, float] | None = None  # (wall, source t)
        self._last_t: float | None = None
        self._changed: asyncio.Event | None = None

    def _event(self) -> asyncio.Event:
        if self._changed is None:
            self._changed = asyncio.Event()
        return self._changed

    def _notify(self) -> None:
        if self._changed is not None:
            self._changed.set()
        self._changed = None

    def pause(self) -> None:
        self.paused = True
        self._notify()

    def resume(self) -> None:
        if self.paused:
            self.paused = False
            self._anchor = None  # re-anchor at the next frame: no catch-up burst
        self._notify()

    def set_speed(self, speed: float) -> None:
        if not speed > 0:
            raise ValueError("speed must be > 0")
        self.speed = float(speed)
        if self._last_t is not None:
            self._anchor = (self._clock(), self._last_t)
        self._notify()

    async def wait(self, t: float) -> float:
        while True:
            if self.paused:
                await self._event().wait()
                continue
            self._last_t = t
            if not self.enabled:
                return 0.0
            now = self._clock()
            if self._anchor is None:
                self._anchor = (now, t)
            wall0, t0 = self._anchor
            delay = wall0 + (t - t0) / self.speed - now
            if delay <= 0:
                self.lag_s = -delay
                return -delay
            ev = self._event()
            try:
                await asyncio.wait_for(ev.wait(), timeout=delay)
            except TimeoutError:
                self.lag_s = 0.0
                return 0.0


class VideoTimeTTS:
    """Silent TTS whose lines occupy *source* time on a :class:`VirtualClock` (offline runs)."""

    def __init__(self, clock: Callable[[], float], config: TTSConfig | None = None) -> None:
        self.config = config or TTSConfig()
        self._clock = clock
        self.spoken: list[SpokenLine] = []
        self._stop = False
        self._current: SpokenLine | None = None

    @property
    def speaking(self) -> bool:
        return self._current is not None

    async def speak(self, text: str, priority: int = 1, *, prosody: str | None = None) -> None:
        cfg = self.config
        rec = SpokenLine(text, priority, prosody, self._clock(), estimate_duration_s(text, cfg.words_per_s,
                                                                                     cfg.min_line_s))
        self.spoken.append(rec)
        self._stop, self._current = False, rec
        end = rec.started + rec.duration_s
        try:
            while self._clock() < end:
                if self._stop:
                    rec.interrupted = True
                    break
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            rec.interrupted = True
            raise
        finally:
            if self._current is rec:
                self._current = None

    async def stop(self) -> None:
        self._stop = True


class MuteableTTS:
    """Wraps the voice: muted lines play on a silent engine of the same duration, so pacing holds."""

    def __init__(self, inner: Any, silent: Any) -> None:
        self.inner, self.silent = inner, silent
        self.muted = False
        self._inner_prosody = _accepts_prosody(inner)

    async def speak(self, text: str, priority: int = 1, *, prosody: str | None = None) -> None:
        if self.muted:
            await self.silent.speak(text, priority, prosody=prosody)
        elif self._inner_prosody:
            await self.inner.speak(text, priority, prosody=prosody)
        else:
            await self.inner.speak(text, priority)

    async def stop(self) -> None:
        for engine in (self.inner, self.silent) if self.silent is not self.inner else (self.inner,):
            try:
                await engine.stop()
            except Exception:  # noqa: BLE001 - one engine failing must not keep the other playing
                log.exception("TTS stop failed")

    async def set_muted(self, muted: bool) -> None:
        if muted and not self.muted and self.inner is not self.silent:
            try:
                await self.inner.stop()  # cut the line now playing
            except Exception:  # noqa: BLE001
                log.exception("TTS stop failed while muting")
        self.muted = bool(muted)

    async def aclose(self) -> None:
        for engine in {id(self.inner): self.inner, id(self.silent): self.silent}.values():
            closer = getattr(engine, "aclose", None)
            if callable(closer):
                try:
                    await closer()
                except Exception:  # noqa: BLE001
                    log.exception("TTS close failed")


def _accepts_prosody(tts: Any) -> bool:
    import inspect

    try:
        params = inspect.signature(tts.speak).parameters
    except (AttributeError, TypeError, ValueError):
        return False
    return "prosody" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


# --------------------------------------------------------------------------- Jev / Claude factories


def build_jev(cfg: JevConfig, env: Mapping[str, str] | None = None) -> JevDecider:
    """A JevDecider; "live" without ``TYPESAFE_API_KEY`` (or a disabled / unloadable config) runs
    in "off" mode, i.e. deterministic fallbacks, never a network call."""
    jc = dataclasses.replace(cfg, deadlines_s=dict(cfg.deadlines_s), thresholds=dict(cfg.thresholds),
                             fallback_rules=dict(cfg.fallback_rules))
    if not jc.enabled:
        jc.mode = "off"
    elif jc.mode == "live" and typesafe_credential_source(env) is None:
        log.warning("Jev live mode without TYPESAFE_API_KEY: running on deterministic fallbacks (mode off)")
        jc.mode = "off"
    elif jc.mode == "replay" and not jc.replay_path:
        log.warning("Jev replay mode without jev.replay_path: mode off")
        jc.mode = "off"
    try:
        if jc.mode == "live":
            env = os.environ if env is None else env
            # The key goes straight to the SDK client; it is never logged or stored elsewhere.
            return JevDecider(jc, client=JevClient(jc, api_key=(env.get("TYPESAFE_API_KEY") or "").strip()))
        return JevDecider(jc)
    except Exception as exc:  # noqa: BLE001 - never crash the game over the decision layer
        log.error("Jev unavailable (%s); mode off", type(exc).__name__)
        return JevDecider(dataclasses.replace(jc, mode="off"))


def build_llm(cfg: AppConfig, env: Mapping[str, str] | None = None, realtime: bool = True,
              jev_available: bool | None = None) -> NarrationLLM | None:
    """``ClaudeNarrationLLM`` only with Anthropic credentials and ``narration.llm_enabled`` (broadcast
    mode, real time unless ``llm.offline``); otherwise None => template-only narration.

    ``jev_available=False`` with ``narration.use_jev_validator`` (D4) also returns None: Claude lines
    are spoken only after a passing Jev line guard, so without Jev no line could ever be used and the
    client (and its periodic prompt-cache pre-warm) would only cost money."""
    n = cfg.narration
    if not n.llm_enabled or not n.cloud_narration_ok or n.mode != "broadcast":
        return None
    if not realtime and not cfg.llm.offline:
        log.info("Claude off for a non-realtime run (llm.offline=false); templates only")
        return None
    source = anthropic_credential_source(env)
    if source is None:
        log.warning("no Anthropic credentials found; Claude disabled (template-only narration)")
        return None
    if jev_available is False and n.use_jev_validator:
        log.warning("Claude off: spoken Claude lines need the Jev line guard (D4) and Jev is unavailable "
                    "(set TYPESAFE_API_KEY, or jev.mode=stub for demos); template-only narration")
        return None
    log.info("Claude colour enabled (credentials from %s)", source)
    return ClaudeNarrationLLM(ClaudeConfig(model=n.model, max_tokens=n.max_tokens, effort=n.effort,
                                           fallbacks=cfg.llm.fallbacks, cache_ttl=cfg.llm.cache_ttl,
                                           answer_directly=cfg.llm.answer_directly, timeout_s=cfg.llm.timeout_s,
                                           max_retries=cfg.llm.max_retries))


def director_overrides(director: DirectorConfig | None) -> dict[str, Any]:
    """``director.*`` fields set away from the ``DirectorConfig`` defaults.

    The narrator builds its own director from ``narration.*`` (cooldowns, min gap) and its mode
    (accessibility reads every possession event: no latest-wins, no flush, AD TTL). Passing the
    whole ``director`` section would silently replace all of that with defaults, so only explicit
    changes are applied on top of the narrator's director."""
    if director is None:
        return {}
    base = DirectorConfig()
    return {f.name: getattr(director, f.name) for f in dataclasses.fields(DirectorConfig)
            if getattr(director, f.name) != getattr(base, f.name)}


def _prewarm_interval(cfg: AppConfig) -> float:
    if cfg.llm.prewarm_interval_s is not None:
        return max(30.0, float(cfg.llm.prewarm_interval_s))
    return 55 * 60.0 if (cfg.llm.cache_ttl or "").strip() == "1h" else 4 * 60.0


# --------------------------------------------------------------------------- event enrichment


class EventEnricher:
    """Adds ``p_three`` to shot results and identity confidences (``id_conf`` / ``jersey_conf``)."""

    def __init__(self, trusted_identity: bool) -> None:
        self.trusted_identity = trusted_identity
        self._p3: dict[str, float] = {}
        self._ident: dict[str, tuple[float, float]] = {}
        self._cache: dict[tuple[str, str | None, str | None], tuple[float, float]] = {}

    def observe(self, frame: FrameState, jersey_conf: Mapping[str, float] | None = None) -> None:
        jc = jersey_conf or {}
        for p in frame.players:
            if self.trusted_identity:
                self._ident[p.player_id] = (1.0, 1.0)
            else:
                conf = float(p.confidence) if math.isfinite(p.confidence) else 0.0
                self._ident[p.player_id] = (min(1.0, max(0.0, conf)), float(jc.get(p.player_id, 0.0)))

    def identity(self, player_id: str) -> tuple[float, float] | None:
        return self._ident.get(player_id)

    def enrich(self, ev: Event) -> Event:
        attrs = dict(ev.attrs)
        changed = False
        if ev.kind is EventKind.SHOT_ATTEMPT and ev.shot_id and attrs.get("p_three") is not None:
            self._p3[ev.shot_id] = float(attrs["p_three"])
        if ev.kind in (EventKind.SHOT_MADE, EventKind.SHOT_MISSED) and ev.shot_id and "p_three" not in attrs:
            p3 = self._p3.get(ev.shot_id)
            if p3 is not None:
                attrs["p_three"], changed = p3, True
        actors = [pid for pid in (ev.player_id, ev.other_player_id) if pid]
        if actors and (self.trusted_identity or ev.source in _TRUSTED_SOURCES):
            # Sim truth / operator / official feed: the identity is known. Always 1.0, also over
            # confidences an operator correction copied from the CV event it revises (the dashboard
            # sends the old attrs back; they describe the old, possibly different, actor).
            if attrs.get("id_conf") != 1.0 or attrs.get("jersey_conf") != 1.0:
                attrs["id_conf"], attrs["jersey_conf"] = 1.0, 1.0
                changed = True
        elif actors and "id_conf" not in attrs:
            # The first revision's value sticks for later engine revisions with the same actors.
            key = (ev.event_id, ev.player_id, ev.other_player_id)
            conf = self._cache.get(key)
            if conf is None:
                vals = [self._ident.get(pid, (0.0, 0.0)) for pid in actors]
                conf = self._cache[key] = (min(v[0] for v in vals), min(v[1] for v in vals))
            attrs["id_conf"], attrs["jersey_conf"] = round(conf[0], 4), round(conf[1], 4)
            changed = True
        return dataclasses.replace(ev, attrs=attrs) if changed else ev


class TieredPseudonyms:
    """The game's shared :class:`~courtvision.narration.pseudonym.PseudonymMap` (same tokens, same
    state) whose :meth:`render` applies the identity tiers to EVERY player a line names.

    ``Narrator._render`` passes confidences only for the actors of the cue's events, and the
    pseudonym map counts a missing confidence as 1.0 (a name). Colour lines (templates from
    ``derived`` phrases such as "{P07} is 4-for-6", Claude candidates) and coach cards carry no
    events, so on video / camera sources they would name players below the 0.95 identity tier. Here
    a player without a confidence from the caller gets the latest perceived identity
    (``EventEnricher.identity``: ``PlayerState.confidence`` and the jersey vote share), and a player
    perception has never identified gets ``default`` (0.0 for perception sources: the team name;
    1.0 for sim truth, where every identity is known)."""

    def __init__(self, shared: Any, identity: Callable[[str], tuple[float, float] | None],
                 default: tuple[float, float]) -> None:
        self.shared = shared
        self._identity = identity
        self._default = (float(default[0]), float(default[1]))

    def __getattr__(self, name: str) -> Any:  # token / player / tokens / pseudonymise / display / config
        return getattr(self.shared, name)

    def __contains__(self, player_id: object) -> bool:
        return player_id in self.shared

    def confidences(self, text: str, id_conf: Mapping[str, float] | None = None,
                    jersey_conf: Mapping[str, float] | None = None) -> tuple[dict[str, float], dict[str, float]]:
        """(id_conf, jersey_conf) covering every token in ``text``: the caller's value first (the
        event's own confidence), else the latest perceived identity, else ``default``."""
        from courtvision.narration.pseudonym import TOKEN_RE

        ic, jc = dict(id_conf or {}), dict(jersey_conf or {})
        for tok in TOKEN_RE.findall(text):
            pid = self.shared.player(tok)
            if pid is None:
                continue
            ident = self._identity(pid) if (pid not in ic or pid not in jc) else None
            seen = ident if ident is not None else self._default
            if pid not in ic and tok not in ic:
                ic[pid] = seen[0]
            if pid not in jc and tok not in jc:
                jc[pid] = seen[1]
        return ic, jc

    def render(self, text: str, roster: Roster, id_conf: dict[str, float] | None = None,
               jersey_conf: dict[str, float] | None = None) -> str:
        ic, jc = self.confidences(text, id_conf, jersey_conf)
        return self.shared.render(text, roster, ic or None, jc or None)


# --------------------------------------------------------------------------- latency


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


@dataclass
class _Latency:
    perception_ms: list[float] = field(default_factory=list)
    engine_ms: list[float] = field(default_factory=list)
    stats_ms: list[float] = field(default_factory=list)
    step_ms: list[float] = field(default_factory=list)
    detect_to_utterance_ms: list[float] = field(default_factory=list)  # source time: engine emission -> line out
    occur_to_utterance_ms: list[float] = field(default_factory=list)  # source time: event t -> line out
    detect_to_utterance_wall_ms: list[float] = field(default_factory=list)
    utterance_ready_ms: list[float] = field(default_factory=list)  # Utterance.latency_ms (narrator)
    detection_delay_ms: list[float] = field(default_factory=list)  # source time: event t -> engine emission

    def report(self, source_s: float, wall_s: float) -> dict[str, Any]:
        return {
            "perception_ms_per_frame": _summary(self.perception_ms),
            "engine_ms_per_frame": _summary(self.engine_ms),
            "stats_ms_per_frame": _summary(self.stats_ms),
            "frame_step_ms": _summary(self.step_ms),
            "detection_delay_ms": _summary(self.detection_delay_ms),
            "event_to_utterance_ms": _summary(self.detect_to_utterance_ms),
            "occurrence_to_utterance_ms": _summary(self.occur_to_utterance_ms),
            "event_to_utterance_wall_ms": _summary(self.detect_to_utterance_wall_ms),
            "utterance_ready_ms": _summary(self.utterance_ready_ms),
            "source_s": source_s,
            "wall_s": wall_s,
            "realtime_factor": (source_s / wall_s) if wall_s > 0 else None,
        }


# --------------------------------------------------------------------------- LiveGame


def _slim_decision(d: Mapping[str, Any]) -> dict[str, Any]:
    """A decision grade for the dashboard: no per-option list, no option ``detail``."""
    out = {k: v for k, v in d.items() if k != "options"}
    for key in ("chosen", "best"):
        opt = out.get(key)
        if isinstance(opt, Mapping):
            out[key] = {k: v for k, v in opt.items() if k != "detail"}
    return out


def hub_stats_msg(stats: GameStats, max_decisions: int = 40) -> dict[str, Any]:
    """The hub ``stats`` message: ``GameStats.to_dict()`` with only the newest ``max_decisions``
    decision grades, slimmed (``decisions_total`` counts them all).

    The full list grows by ~5 KB per graded pass / shot (every option with its detail): late in a
    game the 1 Hz message was megabytes, encoded on the event loop (100-200 ms stalls per second
    at 15 min, frame steps up to seconds) and pushed to every client. ``stats.json`` /
    ``RunResult.stats`` keep every grade."""
    decs = list(stats.decisions or [])
    keep = [_slim_decision(d) for d in decs[-max_decisions:]] if max_decisions > 0 else []
    msg = stats_msg(dataclasses.replace(stats, decisions=keep))
    msg["stats"]["decisions_total"] = len(decs)
    return msg


def _hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = (int(v) for v in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def _encode_jpeg(image: np.ndarray, quality: int) -> bytes | None:
    import cv2

    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None


class LiveGame:
    """One game: sources in, events / stats / insights / narration / hub messages out."""

    def __init__(self, config: AppConfig, source: FrameSource, hub: Hub | None = None, *,
                 realtime: bool | None = None, jev: JevDecider | None = None, llm: NarrationLLM | None = None,
                 tts: TTS | None = None, env: Mapping[str, str] | None = None, detector: Any | None = None,
                 on_frame: Callable[[FrameState], None] | None = None) -> None:
        self._on_frame = on_frame  # sees every FrameState the event engine consumes (courtvision ingest caches them)
        if config.pipeline.model_bundle:  # what `courtvision train` learned and `promote` accepted
            from courtvision.learning.bundle import apply_bundle

            config = apply_bundle(config, config.pipeline.model_bundle)
        cfg = self.config = config
        self.source = source
        self.hub = hub
        self.env = os.environ if env is None else env
        self.realtime = bool(source.realtime or source.live) if realtime is None else bool(realtime)
        self.court: CourtSpec = source.court or court_for(cfg.game.league)
        self.rules: RuleSpec = source.rules or rules_for(cfg.game.ruleset or cfg.game.league,
                                                         list(cfg.game.rule_overrides) or None)
        self.roster: Roster = source.roster or load_roster(cfg.game.roster_path, cfg.game.home_name,
                                                           cfg.game.away_name)
        source.bind(self.court, self.rules, self.roster)
        self.is_sim = source.truth is not None
        # offline: source time; real time: monotonic time at the playback speed (a live camera plays at 1x)
        self.clock: Callable[[], float] = SpeedClock(1.0 if source.live else source.speed) if self.realtime \
            else VirtualClock()
        self.pacer = Pacer(enabled=self.realtime and not source.live, speed=source.speed)

        # --- events / stats / insights
        ecfg = cfg.events
        if self.is_sim and cfg.pipeline.sim_assume_calibrated and not ecfg.assume_calibrated:
            ecfg = dataclasses.replace(ecfg, assume_calibrated=True)
        self.engine = EventEngine(self.court, self.roster, ecfg, rules=self.rules)
        self.shot_model = self._make_shot_model()
        self.decision_model = (DecisionModel(self.court, self.shot_model, cfg.decision_model.config)
                               if cfg.decision_model.enabled and self.shot_model is not None else None)
        self.stats = StatsEngine(self.court, self.roster, cfg.stats, shot_model=self.shot_model,
                                 decision_model=self.decision_model, rules=self.rules)
        self.insights = InsightEngine(self.roster, self.court, cfg.insights, rules=self.rules)
        self.enricher = EventEnricher(trusted_identity=self.is_sim and not source.synthetic)  # renders: perceived ids

        # --- decision layer, Claude, voice, captions, narrator
        self.pseudo = pseudonyms_for(self.roster, seed=cfg.narration.seed)
        trusted = self.enricher.trusted_identity
        self.tiered_pseudo = TieredPseudonyms(self.pseudo, self.enricher.identity,
                                              (1.0, 1.0) if trusted else (0.0, 0.0))
        self.jev = jev if jev is not None else build_jev(cfg.jev, self.env)
        self.llm = llm if llm is not None else build_llm(cfg, self.env, self.realtime,
                                                         jev_available=bool(getattr(self.jev, "available", False)))
        self.tts = MuteableTTS(*self._make_voice(tts))
        self.captions = self._make_captions()
        ncfg = cfg.narration
        if not self.realtime:
            ncfg = dataclasses.replace(ncfg, poll_s=0.0)  # cooperative stepping on the source clock
        # director_config=None: the narrator derives its own (narration.cooldown_s / min_gap_s and the
        # accessibility FIFO policy); explicit director.* settings are applied on top of it below.
        self.narrator = Narrator(
            self.roster, self.court, ncfg, llm=self.llm, jev=self.jev, on_utterance=self._on_utterance,
            tts=self.tts, rules=self.rules, dead_ball=self.engine.deadball, captions=self.captions, clock=self.clock,
            director_config=None, candidate_config=self._candidate_config(), facts_config=cfg.facts,
            template_config=cfg.templates, pseudo=self.tiered_pseudo)  # type: ignore[arg-type]
        overrides = director_overrides(cfg.director)
        if overrides:
            self.narrator.director = Director(dataclasses.replace(self.narrator.director.config, **overrides))

        # --- simulated operator
        self.operator: SimOperator | None = None
        self._manual_queue: list[Event] = []
        if self.is_sim:
            assert source.truth is not None
            self.operator = SimOperator(source.truth, SimOperatorConfig(entry_delay_s=cfg.operator.entry_delay_s,
                                                                        lineup_t=cfg.operator.lineup_t))
            kinds = {EventKind(k) for k in cfg.operator.replay_kinds}
            lo = source.start_t - 1e-9
            self._manual_queue = sorted((e for e in source.truth.events if e.kind in kinds and e.t >= lo),
                                        key=lambda e: (e.t, e.frame_idx))

        # --- perception (lazy: built on the first image)
        self.perception: Any = None
        self._detector = detector  # a ready Detector (e.g. the CLI's R&D adapter); None = perception.detector
        self._calibration: Calibration | None = None
        self._auto_center: np.ndarray | None = None  # broadcast camera centre (automatic court registration)
        self._auto_params: Any = None
        self._auto_frames: list[tuple[float, np.ndarray]] = []
        self._auto_tries = 0
        self._auto_probes = 0  # failed probes at a saved camera centre
        self._auto_probe_t: float | None = None
        self._pending_roi: dict[str, Any] | None = None
        self._needs_calibration = False
        self._calib_hold = False  # real-time file playback paused until a calibration arrives
        self._perc_lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None

        # --- state
        self._lock = threading.RLock()  # engine / stats / insights / stores (controller may call from another thread)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._events: dict[str, Event] = {}
        self._order: dict[str, int] = {}
        self._emitted: dict[str, tuple[float, float]] = {}  # event_id -> (source t, wall) of its latest revision
        self._insights_out: list[Insight] = []
        self._ins_pending: list[Event] = []
        self._utterances: list[Utterance] = []
        self._kept: list[FrameState] = []
        self._board: ScoreboardReading | None = None  # last confirmed reading (display / facts / hub)
        self._pending_board: ScoreboardReading | None = None
        # manual scoreboard fields held over the source: field -> (manual value, source value when set). A field
        # follows the source again once the source shows something else (a new basket, the clock running).
        self._board_hold: dict[str, tuple[Any, Any]] = {}
        self._raw_board: ScoreboardReading | None = None  # the source's own last reading
        self._latest_frame: FrameState | None = None
        self._latest_image: tuple[int, np.ndarray] | None = None
        self._snap: GameStats | None = None
        self._snap_version = -1
        self._version = 0  # bumps on every ingest; snapshot cache key
        self._t = source.start_t
        self._frame_idx = 0
        self._last: dict[str, float] = {}
        self._lat = _Latency()
        self._frames_done = 0
        self._dropped = 0
        self._errors = 0
        self._started = False
        self._finished = False
        self._stop_requested = False
        self._skip_tail = False
        self._wall0: float | None = None
        self._wall_s = 0.0
        self._prewarm_task: asyncio.Task[None] | None = None
        self.final_stats: GameStats | None = None
        self.controller = GameController(self)

    # ------------------------------------------------------------------ construction helpers

    def _make_shot_model(self) -> ShotQualityModel | None:
        sm = self.config.shot_model
        if not sm.enabled:
            return None
        if sm.path:
            return ShotQualityModel.from_json(json.loads(Path(sm.path).read_text()))
        return ShotQualityModel(self.court.league, sm.priors)

    def _make_voice(self, tts: TTS | None) -> tuple[Any, Any]:
        tcfg = self.config.tts
        if not self.realtime:
            voice = tts if tts is not None else VideoTimeTTS(self.clock, tcfg)
            return voice, VideoTimeTTS(self.clock, tcfg) if tts is not None else voice
        silent = NullTTS(tcfg, time_scale=1.0)
        if tts is not None:
            return tts, silent
        kind = self.config.narration.tts
        if self.config.narration.mode == "coach":
            kind = "none"
        try:
            voice = make_tts(kind, tcfg)
        except Exception as exc:  # noqa: BLE001 - narration keeps running without audio
            log.error("TTS %r unavailable (%s); narrating silently", kind, exc)
            voice = silent
        return voice, silent

    def _make_captions(self) -> CaptionEmitter | None:
        sinks: list[Any] = []
        if self.hub is not None and self.config.pipeline.captions_sidecar:
            sinks.append(SidecarSink(self.hub.publish))
        if self.config.pipeline.webvtt_path:
            sinks.append(WebVTTWriter(self.config.pipeline.webvtt_path))
        return CaptionEmitter(sinks, self.config.captions) if sinks else None

    def _candidate_config(self) -> CandidateConfig | None:
        cfg = self.config
        if cfg.candidates is not None:
            return cfg.candidates
        n = cfg.narration
        return CandidateConfig(p95_total_s=n.p95_total_s, candidate_deadline_s=n.candidate_deadline_s,
                               max_tokens=n.max_tokens, max_words=int(n.max_words.get("color", 35)),
                               use_jev_guard=n.use_jev_validator, answer_directly=cfg.llm.answer_directly)

    # ------------------------------------------------------------------ public views

    @property
    def current_t(self) -> float:
        return self._t

    @property
    def events(self) -> list[Event]:
        """Latest revision of every event (enriched), sorted by t (RETRACTED included)."""
        with self._lock:
            return sorted(self._events.values(), key=lambda e: (e.t, self._order[e.event_id]))

    @property
    def transcript(self) -> list[Utterance]:
        return list(self._utterances)

    @property
    def insights_emitted(self) -> list[Insight]:
        return list(self._insights_out)

    @property
    def kept_frames(self) -> list[FrameState]:
        return list(self._kept)

    @property
    def latest_board(self) -> ScoreboardReading | None:
        return self._board

    def latency_report(self) -> dict[str, Any]:
        source_s = max(0.0, self._t - self.source.start_t) if self._frames_done else 0.0
        return self._lat.report(source_s, self._wall_s)

    def snapshot(self) -> GameStats:
        """A fresh ``GameStats`` (the frame step itself reuses one for up to ``1 / stats_hz``)."""
        with self._lock:
            return self._snapshot(force=True)

    def status(self) -> dict[str, Any]:
        return {
            "source": self.source.kind, "mode": self.source.mode, "realtime": self.realtime,
            "paused": self.pacer.paused, "speed": self.pacer.speed, "muted": self.tts.muted,
            "lag_s": round(self.pacer.lag_s, 3), "frames": self._frames_done, "dropped": self._dropped,
            "needs_calibration": self._needs_calibration, "finished": self._finished,
            "narration": self.config.narration.mode, "llm": self.narrator.pool is not None,
            "jev": getattr(self.jev.client, "mode", None) if self.jev is not None else None,
        }

    def request_stop(self, drain: bool = True) -> None:
        """Stop after the current frame; ``drain=False`` also skips the narration tail (shutdown)."""
        self._stop_requested = True
        if not drain:
            self._skip_tail = True
        self.pacer.resume()

    # ------------------------------------------------------------------ lifecycle

    async def run(self) -> None:
        """Start, consume the whole source (or until :meth:`request_stop`), then finish."""
        await self.start()
        frames = self.source.__aiter__()
        try:
            async for item in frames:
                if self._stop_requested:
                    break
                await self.process(item)
        finally:
            closer = getattr(frames, "aclose", None)
            if closer is not None:
                await closer()  # stop the source generator now, not at garbage collection
            await self.finish()

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._loop = asyncio.get_running_loop()
        self._wall0 = time.perf_counter()
        if self.hub is not None:
            if self.hub.loop is None or self.hub.loop.is_closed() or not self.hub.loop.is_running():
                self.hub.bind_loop(self._loop)
            colors = {"home": _hex(palette.HOME_BGR), "away": _hex(palette.AWAY_BGR)} if self.is_sim else None
            await self.hub.publish(meta_msg(self.roster, self.court, self.rules, t=self._t, colors=colors))
            await self.hub.publish(status_msg(self._t, **self.status()))
        if self.realtime and self.source.mode in ("video", "broadcast"):
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="perception")
        self.narrator.start()
        if self.narrator.pool is not None and self.config.llm.prewarm:
            self._prewarm_task = asyncio.create_task(self._prewarm_loop(), name="claude-prewarm")

    async def _prewarm_loop(self) -> None:
        interval = _prewarm_interval(self.config)
        while True:
            try:
                await self.narrator.prewarm()
            except Exception:  # noqa: BLE001 - a failed pre-warm only costs one cache write later
                log.exception("Claude prompt-cache pre-warm failed")
            await asyncio.sleep(interval)

    async def process(self, item: FrameItem) -> None:
        """Pace (real time), perceive (image items) and run one frame through the stack."""
        if not self._started:
            await self.start()
        lateness = await self.pacer.wait(item.t)
        drop = self.config.source.drop_late_s
        if self.realtime and drop is not None and lateness > drop:
            self._dropped += 1
            return
        try:
            if item.state is not None:
                if item.image is not None:
                    self._latest_image = (item.frame_idx, item.image)
                await self._process_frame(item.state)
            elif item.image is not None:
                self._latest_image = (item.frame_idx, item.image)
                result = await self._perceive_async(item)
                if result is None:  # no calibration yet: keep serving the image for calibration
                    self._t, self._frame_idx = item.t, item.frame_idx
                    if not self.realtime:
                        self.clock.set(item.t)  # type: ignore[attr-defined]
                    await self._await_calibration(item.t)
                    return
                state, jersey = result
                await self._process_frame(state, jersey)
        except Exception:
            self._errors += 1
            if not self.realtime:
                raise
            log.exception("frame %d failed; continuing", item.frame_idx)
        if not self.realtime:
            await asyncio.sleep(0)  # let the narrator's speaking loop and background tasks step

    async def _await_calibration(self, t: float) -> None:
        """Perception has no calibration yet. Tell the dashboard (status ``needs_calibration``: the
        frame step, which normally publishes status, does not run) and, for a file played in real
        time, hold playback until ``POST /api/calibration`` arrives instead of consuming the video
        unseen. A live camera keeps going (its frames cannot wait)."""
        first = False
        if self.realtime and not self.source.live and not self._calib_hold:
            self._calib_hold = first = True
            self.pacer.pause()
            log.warning("holding %s playback at t=%.2f until a calibration arrives (dashboard Calibration tab)",
                        self.source.kind, t)
        if self.hub is not None and (first or self._tick_due_named("status_msg", t, self.config.pipeline.status_hz)):
            await self.hub.publish(status_msg(t, **self.status()))

    def _release_calibration_hold(self) -> bool:
        """Resume playback held by :meth:`_await_calibration` (True if it was held)."""
        if not self._calib_hold:
            return False
        self._calib_hold = False
        self.pacer.resume()
        return True

    async def finish(self) -> None:
        """Flush the engine, let narration finish, publish final stats and close everything."""
        if self._finished:
            return
        self._finished = True
        try:
            with self._lock:
                batch = self._ingest(self.engine.flush())
                if batch:
                    self.stats.on_events(batch)
                    self._ins_pending.extend(batch)
                snap = self._snapshot(force=True)
                new_ins = self._update_insights(snap, force=True)
            await self._publish_events(batch)
            if batch:
                await self.narrator.on_events(batch, snap, self._display_frame())
            if new_ins:
                await self._emit_insights(new_ins, snap)
            await self._tail()
            with self._lock:
                self.final_stats = self._snapshot(force=True)
            if self.hub is not None:
                await self.hub.publish(self._stats_msg(self.final_stats))
        finally:
            self._wall_s = time.perf_counter() - (self._wall0 or time.perf_counter())
            await self._close()
            if self.hub is not None:
                await self.hub.publish(status_msg(self._t, **self.status()))

    async def _tail(self) -> None:
        tail = max(0.0, float(self.config.pipeline.tail_s))
        if tail <= 0 or self._skip_tail:
            return
        if self.realtime:
            await self.narrator.drain(timeout=tail)
            return
        clock = self.clock
        assert isinstance(clock, VirtualClock)
        dt = 1.0 / max(1.0, self.source.fps)
        end = clock.t + tail
        drain = asyncio.ensure_future(self.narrator.drain(timeout=tail + 1.0))
        try:
            while not drain.done() and clock.t < end:
                clock.advance(dt)
                await asyncio.sleep(0)
        finally:
            if not drain.done():
                drain.cancel()
                await asyncio.gather(drain, return_exceptions=True)

    async def _close(self) -> None:
        if self._prewarm_task is not None:
            self._prewarm_task.cancel()
            await asyncio.gather(self._prewarm_task, return_exceptions=True)
        for name, closer in (("narrator", self.narrator.aclose), ("jev", getattr(self.jev, "aclose", None)),
                             ("llm", getattr(self.llm, "aclose", None)), ("tts", self.tts.aclose),
                             ("captions", getattr(self.captions, "aclose", None))):
            if closer is None:
                continue
            try:
                await closer()
            except Exception:  # noqa: BLE001 - shutdown must finish
                log.exception("closing %s failed", name)
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        try:
            self.source.close()
        except Exception:  # noqa: BLE001
            log.exception("closing the source failed")

    # ------------------------------------------------------------------ perception

    def _ensure_perception(self, image: np.ndarray | None = None, t: float = 0.0) -> Any:
        if self.perception is not None:
            return self.perception
        from courtvision.perception.jersey import TemplateJerseyReader
        from courtvision.perception.pipeline import PerceptionPipeline
        from courtvision.perception.scoreboard_reader import ScoreboardROI

        pc = self.config.perception
        hints = pc.use_source_hints and self.source.synthetic
        calib = self._calibration
        if calib is None and pc.calibration_path and Path(pc.calibration_path).is_file():
            calib = Calibration.load(pc.calibration_path)
        if calib is None and hints:
            calib = self.source.calibration()
        if calib is None and image is not None and self._auto_court_wanted():
            calib = self._auto_court(image, t)
        if calib is None:
            if not self._needs_calibration:
                log.warning("no calibration for %s; POST /api/calibration to start perception", self.source.kind)
            self._needs_calibration = True
            return None
        self._needs_calibration = False
        self._calibration = calib
        pcfg = dataclasses.replace(pc.pipeline)
        defaults = type(pcfg)()
        if hints:
            for key, value in self.source.perception_hints().items():
                if getattr(pcfg, key) == getattr(defaults, key):
                    setattr(pcfg, key, value)
        roi: Any = self._pending_roi
        if roi is None and pc.scoreboard_roi_path and Path(pc.scoreboard_roi_path).is_file():
            roi = json.loads(Path(pc.scoreboard_roi_path).read_text())
        if roi is None and pcfg.scoreboard_roi is None and hints:
            roi = self.source.scoreboard_roi()
        if roi is not None:
            pcfg.scoreboard_roi = ScoreboardROI.from_dict(roi).to_dict()
        kind = pc.detector if pc.detector != "auto" else ("color_blob" if self.source.synthetic else "rfdetr")
        if self._detector is not None:
            detector: Any = self._detector
            kind = type(detector).__name__
        elif kind == "color_blob":
            from courtvision.perception.detector import ColorBlobDetector

            detector = ColorBlobDetector(pc.color_blob)
        else:
            from courtvision.perception.rfdetr_adapter import RFDETRDetector

            detector = RFDETRDetector(pc.rfdetr)
        reader_kind = pc.jersey_reader if pc.jersey_reader != "auto" else ("template" if self.source.synthetic else "none")
        reader = None
        if reader_kind == "template":
            cands = self.source.jersey_candidates() or sorted({e.jersey for e in self.roster.entries}) or None
            reader = TemplateJerseyReader(pc.jersey, candidates=cands)
        self.perception = PerceptionPipeline(detector, calib, roster=self.roster, config=pcfg, jersey_reader=reader)
        if pc.broadcast == "on" or (pc.broadcast == "auto" and self.source.mode == "broadcast"):
            self.perception = self._broadcast_perception(self.perception, calib, hints)
        self._pending_roi = None
        log.info("perception ready: %s detector, %s jersey reader, calibration %s", kind, reader_kind,
                 "from the sim camera" if calib is not None and hints else "from file / operator")
        return self.perception

    def _broadcast_perception(self, inner: Any, calib: Calibration, hints: bool) -> Any:
        """Wrap the pipeline for broadcast footage (perception/broadcast.py) with the scorebug reader of the
        configured layout and glyph bank (sim renders: the simulator's layout, glyphs from its font)."""
        from courtvision.perception.broadcast import BroadcastPerception
        from courtvision.perception.scorebug import GlyphBank, ScorebugLayout, ScorebugReader

        pc = self.config.perception
        layout: Any = None
        if pc.scorebug_layout_path and Path(pc.scorebug_layout_path).is_file():
            layout = json.loads(Path(pc.scorebug_layout_path).read_text())
        elif hints and hasattr(self.source, "scorebug_layout"):
            layout = self.source.scorebug_layout()
        reader = None
        if layout is not None:
            lay = ScorebugLayout.from_dict(layout)
            if pc.glyph_bank_path and Path(pc.glyph_bank_path).is_file():
                bank = GlyphBank.from_dict(json.loads(Path(pc.glyph_bank_path).read_text()))
            else:  # bootstrap from the font the simulator draws with (real broadcasts: learn it, ingest/train)
                scale = float(layout.get("scale", 1.0))
                bank = GlyphBank.from_font(lay.fields["game_clock"][3], scale=0.9 * scale,
                                           thickness=max(1, int(round(2 * scale))))
            reader = ScorebugReader(lay, bank)
        if reader is None and pc.scorebug_ocr and not self.source.synthetic:
            from courtvision.perception import vision_ocr

            if vision_ocr.available():
                from courtvision.perception.scorebug_layout import load_saved_profile
                from courtvision.perception.scorebug_ocr import ESPN_2024, OCRScorebugReader

                # the layout tools/nba_scan_scorebug.py discovered for this video (next to it or in its game
                # folder), else the hand-measured ESPN bug
                reader = OCRScorebugReader(load_saved_profile(self.config.source.path, pc.calibration_path)
                                           or ESPN_2024)
        tracker = None
        if self._auto_center is not None:
            from courtvision.perception.broadcast_court import BroadcastCourtTracker

            tracker = BroadcastCourtTracker(self.court, calib.image_size, self._auto_center)
            if self._auto_params is not None:
                tracker.reset(self._auto_params)
        log.info("broadcast perception: court tracking (%s), cut / replay gate, scorebug %s",
                 "automatic" if tracker is not None else "from the operator calibration",
                 "on" if reader is not None else "off (no layout)")
        return BroadcastPerception(inner, calib, reader, pc.broadcast_config, court_tracker=tracker)

    def _auto_court_wanted(self) -> bool:
        pc = self.config.perception
        return bool(pc.auto_court and pc.broadcast == "on" and not self.source.synthetic and self._auto_tries < pc.auto_court_tries)

    def _auto_court(self, image: np.ndarray, t: float) -> Calibration | None:
        """Automatic court registration of a broadcast game camera (``perception/broadcast_court.py``): frames are
        collected every ``auto_court_every_s``; with ``auto_court_frames`` of them the camera centre is estimated
        (a few seconds, once). A centre saved next to the calibration file is reused."""
        from courtvision.perception.broadcast_court import estimate_center
        from courtvision.perception.court_search import CameraParams

        pc = self.config.perception
        saved = Path(pc.calibration_path).with_name("broadcast_camera.json") if pc.calibration_path else None
        size = (int(image.shape[1]), int(image.shape[0]))
        if self._auto_center is None and saved is not None and saved.is_file():
            try:
                d = json.loads(saved.read_text())
                self._auto_center = np.asarray(d["center"], float)
            except (ValueError, KeyError, OSError):
                log.warning("could not read %s", saved)
        if self._auto_center is not None:
            from courtvision.perception.broadcast_court import BroadcastCourtTracker

            # one probe (a full search at the known centre) per auto_court_every_s; every auto_court_frames failed
            # probes count as one try, the time budget of a centre estimate
            if self._auto_probe_t is not None and t - self._auto_probe_t < pc.auto_court_every_s:
                return None
            self._auto_probe_t = t
            probe = BroadcastCourtTracker(self.court, size, self._auto_center)
            fix = probe.update(image, cut=True)
            if fix.params is None:
                self._auto_probes += 1
                if self._auto_probes % max(1, pc.auto_court_frames) == 0:
                    self._auto_tries += 1
                return None
            self._auto_params = fix.params
            return Calibration(homography=fix.homography(size), image_size=size, league=self.court.league)
        if self._auto_frames and t - self._auto_frames[-1][0] < pc.auto_court_every_s:
            return None
        self._auto_frames.append((t, image.copy()))
        if len(self._auto_frames) < pc.auto_court_frames:
            return None
        self._auto_tries += 1
        log.info("estimating the broadcast camera from %d frames", len(self._auto_frames))
        center, cams = estimate_center([f for _, f in self._auto_frames], self.court)
        last = next((c for c in reversed(cams) if c is not None), None)
        self._auto_frames = self._auto_frames[-max(1, pc.auto_court_frames // 2):]
        if center is None or last is None:
            log.info("no game camera found yet")
            return None
        self._auto_center = center
        if saved is not None:
            saved.write_text(json.dumps({"center": [float(v) for v in center]}))
        fix_params = CameraParams(center, last.pan, last.tilt, last.roll, last.f)
        self._auto_params = fix_params
        log.info("broadcast camera centre %s", np.round(center, 1))
        return Calibration(homography=Homography(np.linalg.inv(fix_params.G(size))), image_size=size,
                           league=self.court.league)

    def _perceive(self, image: np.ndarray, frame_idx: int, t: float) -> tuple[FrameState, dict[str, float]] | None:
        with self._perc_lock:
            pipe = self._ensure_perception(image, t)
            if pipe is None:
                return None
            a = time.perf_counter()
            state = pipe.process(image, frame_idx, t)
            self._lat.perception_ms.append((time.perf_counter() - a) * 1000.0)
            return state, self._jersey_shares(pipe, state)

    def _jersey_shares(self, pipe: Any, state: FrameState) -> dict[str, float]:
        """Jersey vote share of each roster-identified player's own number (0 when it is not leading)."""
        voter = getattr(getattr(pipe, "identity", None), "voter", None)
        out: dict[str, float] = {}
        if voter is None:
            return out
        for p in state.players:
            entry = self.roster.get(p.player_id)
            if entry is None or p.track_id is None:
                continue
            allowed = {e.jersey for e in self.roster.entries if e.team is entry.team}
            number, share, _n = voter.best(p.track_id, allowed)
            out[p.player_id] = float(share) if number == entry.jersey else 0.0
        return out

    async def _perceive_async(self, item: FrameItem) -> tuple[FrameState, dict[str, float]] | None:
        assert item.image is not None
        if self._executor is None:  # offline: inline, deterministic
            return self._perceive(item.image, item.frame_idx, item.t)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._perceive, item.image, item.frame_idx, item.t)

    async def _in_perception(self, fn: Callable[[], T]) -> T:
        """Run ``fn`` serialized with frame perception (the worker thread when there is one)."""
        def guarded() -> T:
            with self._perc_lock:
                return fn()

        if self._executor is not None:
            return await asyncio.get_running_loop().run_in_executor(self._executor, guarded)
        return guarded()

    # ------------------------------------------------------------------ the frame step

    async def _process_frame(self, frame: FrameState, jersey: Mapping[str, float] | None = None) -> None:
        cfg = self.config.pipeline
        a = time.perf_counter()
        t = frame.t
        if not self.realtime:
            self.clock.set(t)  # type: ignore[attr-defined]
        with self._lock:
            self._t, self._frame_idx = t, frame.frame_idx
            self._latest_frame = frame
            batch = self._operator_due(t)
            if frame.scoreboard is not None and frame.scoreboard.source != "manual":
                self._raw_board = frame.scoreboard
                if self._board_hold:
                    frame = dataclasses.replace(frame, scoreboard=self._held_board(frame.scoreboard))
            if self._pending_board is not None:
                frame = dataclasses.replace(frame, scoreboard=self._pending_board)
                self._pending_board = None
            self.enricher.observe(frame, jersey)
            if self._on_frame is not None:
                self._on_frame(frame)
            e0 = time.perf_counter()
            batch += self._ingest(self.engine.process(frame))
            e1 = time.perf_counter()
            self.stats.on_events(batch)
            self.stats.on_frame(frame, self.engine.holder_id, self.engine.possession_team, self.engine.attacking_side)
            self.insights.observe_frame(frame)
            e2 = time.perf_counter()
            if frame.scoreboard is not None:
                self._update_board(frame.scoreboard)
            if (not self.realtime and self.source.synthetic and self.is_sim and cfg.keep_frames
                    and len(self._kept) < cfg.max_kept_frames):
                self._kept.append(frame)  # only offline runs build the tracking report from these
            self._frames_done += 1
            self._ins_pending.extend(batch)
            snap = self._snapshot()  # recomputed after new events, else at most stats_hz
            new_ins = self._update_insights(snap, force=False)
            display = self._display_frame(frame)
        self._lat.engine_ms.append((e1 - e0) * 1000.0)
        self._lat.stats_ms.append((e2 - e1) * 1000.0)
        await self._publish_events(batch)
        if batch:
            await self.narrator.on_events(batch, snap, display)
        if new_ins:
            await self._emit_insights(new_ins, snap)
        if self._tick_due(t, cfg.tick_hz):
            await self.narrator.tick(t, snap, self.engine.deadball)
        if self.hub is not None:
            if self._tick_due_named("frame_msg", t, cfg.frame_hz):
                await self.hub.publish(frame_msg(display, self.roster, self.engine.holder_id, self.engine.possession_team))
            if self._tick_due_named("stats_msg", t, cfg.stats_hz):
                await self.hub.publish(self._stats_msg(snap))
            if self._tick_due_named("status_msg", t, cfg.status_hz):
                await self.hub.publish(status_msg(t, **self.status()))
        self._lat.step_ms.append((time.perf_counter() - a) * 1000.0)

    def _stats_msg(self, snap: GameStats) -> dict[str, Any]:
        return hub_stats_msg(snap, self.config.pipeline.hub_decisions)

    def _due(self, name: str, t: float, hz: float) -> bool:
        """True when ``1 / hz`` source seconds passed since the last time ``name`` fired (no state change)."""
        last = self._last.get(name)
        return last is None or t - last >= 1.0 / hz - 1e-9

    def _tick_due_named(self, name: str, t: float, hz: float) -> bool:
        if self._due(name, t, hz):
            self._last[name] = t
            return True
        return False

    def _tick_due(self, t: float, hz: float) -> bool:
        return self._tick_due_named("tick", t, hz)

    def _snapshot(self, force: bool = False) -> GameStats:
        if force or self._snap is None or self._snap_version != self._version or self._due("snapshot", self._t,
                                                                                        self.config.pipeline.stats_hz):
            self._snap = self.stats.snapshot()
            self._snap_version = self._version
            self._last["snapshot"] = self._t
        return self._snap

    def _update_insights(self, snap: GameStats, force: bool) -> list[Insight]:
        pending = self._ins_pending
        loud = any(e.kind not in _QUIET_KINDS for e in pending)
        if not (force or loud or self._due("insights", self._t, self.config.pipeline.insights_hz)):
            return []
        self._ins_pending = []
        self._last["insights"] = self._t
        new = self.insights.update(snap, [e for e in pending if e.kind not in _QUIET_KINDS], self._t)
        self._insights_out.extend(new)
        return new

    async def _emit_insights(self, new: list[Insight], snap: GameStats) -> None:
        if self.hub is not None:
            for ins in insight_order(new):  # local dashboard: names rendered here, never sent to a cloud API
                await self.hub.publish(dataclasses.replace(ins, text=self.insights.render(ins)))
        await self.narrator.on_insights(new, snap)

    def _ingest(self, events: Iterable[Event]) -> list[Event]:
        """Enrich and store new events / revisions (latest revision wins); returns the enriched batch."""
        out: list[Event] = []
        wall = time.perf_counter()
        for ev in events:
            cur = self._events.get(ev.event_id)
            if cur is not None and ev.revision < cur.revision:
                continue
            e = self.enricher.enrich(ev)
            if e.event_id not in self._order:
                self._order[e.event_id] = len(self._order)
                self._lat.detection_delay_ms.append(max(0.0, self._t - e.t) * 1000.0)
            self._events[e.event_id] = e
            self._emitted[e.event_id] = (self._t, wall)
            out.append(e)
        if out:
            self._version += 1
        return out

    def _operator_due(self, t: float) -> list[Event]:
        """Simulated operator entries due by ``t`` (lineups via StatsEngine; FOUL / TIMEOUT as manual events)."""
        out: list[Event] = []
        if self.operator is not None:
            for entry in self.operator.due(t):
                out += self._apply_lineup(entry)
        delay = self.config.operator.replay_delay_s
        while self._manual_queue and self._manual_queue[0].t + delay <= t + 1e-9:
            truth = self._manual_queue.pop(0)
            ev = dataclasses.replace(truth, event_id="", status=EventStatus.CONFIRMED, source="manual", revision=0,
                                     attrs={**truth.attrs, "entry": "sim_operator"})
            out += self._ingest(self.engine.add_manual(ev))
        return out

    def _apply_lineup(self, entry: Mapping[str, Any]) -> list[Event]:
        """One operator lineup entry (``POST /api/lineup`` shape); returns the manual events it made."""
        team = Team(entry["team"])
        t = float(entry["t"]) if entry.get("t") is not None else self._t
        if entry.get("players") is not None:
            players = [str(p) for p in entry["players"]]
            self.stats.set_lineup(team, players, t)
            return []
        out_id, in_id = str(entry["out"]), str(entry["in"])
        if out_id == in_id:
            raise ValueError("a substitution needs distinct out and in player ids")
        self.stats.substitute(team, out_id, in_id, t)
        if not self.config.operator.lineup_events:
            return []
        ev = Event(event_id="", kind=EventKind.SUBSTITUTION, t=t, frame_idx=self._frame_idx, team=team,
                   player_id=in_id, other_player_id=out_id, status=EventStatus.CONFIRMED, source="manual",
                   attrs={"entry": "operator"})
        return self._ingest(self.engine.add_manual(ev))

    # ------------------------------------------------------------------ scoreboard

    def _held_board(self, raw: ScoreboardReading) -> ScoreboardReading:
        """The source's reading with the operator's held fields (``_board_hold``) applied; a field whose source
        value has changed since the override is released (the score pair together)."""
        for group in _BOARD_GROUPS:  # the two scores are one unit: either changing releases both
            keys = [k for k in group if k in self._board_hold]
            if any(getattr(raw, k) is not None and getattr(raw, k) != self._board_hold[k][1] for k in keys):
                for k in keys:
                    del self._board_hold[k]
        held = {k: manual for k, (manual, _) in self._board_hold.items()}
        return dataclasses.replace(raw, **held) if held else raw

    def _update_board(self, reading: ScoreboardReading) -> None:
        """Keep the last confirmed reading: confirmed score / period from the engine's confirmer,
        last observed clocks; trusted sources (sim / manual / feed) as read."""
        if reading.source in _TRUSTED_SOURCES:
            self._board = reading
            return
        sb = self.engine.scoreboard
        score = sb.score or {}
        self._board = ScoreboardReading(
            t=reading.t, home_score=score.get(Team.HOME), away_score=score.get(Team.AWAY),
            period=sb.period if sb.period is not None else (self._board.period if self._board else None),
            game_clock=sb.game_clock, shot_clock=sb.shot_clock, source=reading.source, confidence=reading.confidence)

    def _display_frame(self, frame: FrameState | None = None) -> FrameState | None:
        frame = frame if frame is not None else self._latest_frame
        if frame is None:
            return None
        if frame.scoreboard is None and self._board is not None:
            return dataclasses.replace(frame, scoreboard=self._board)
        if frame.scoreboard is not None and frame.scoreboard.source not in _TRUSTED_SOURCES and self._board is not None:
            return dataclasses.replace(frame, scoreboard=self._board)
        return frame

    # ------------------------------------------------------------------ publishing

    async def _publish_events(self, batch: list[Event]) -> None:
        if self.hub is not None:
            for e in batch:
                await self.hub.publish(e)

    async def _on_utterance(self, utt: Utterance) -> None:
        self._utterances.append(utt)
        times = [self._emitted[i] for i in utt.event_ids if i in self._emitted]
        if times:
            now_src = self.clock() if not self.realtime else self._t  # source time the line goes out
            self._lat.detect_to_utterance_ms.append(max(0.0, now_src - max(x[0] for x in times)) * 1000.0)
            self._lat.detect_to_utterance_wall_ms.append(
                max(0.0, time.perf_counter() - max(x[1] for x in times)) * 1000.0)
            occ = [self._events[i].t for i in utt.event_ids if i in self._events]
            if occ:
                self._lat.occur_to_utterance_ms.append(max(0.0, now_src - max(occ)) * 1000.0)
        if utt.latency_ms is not None:
            self._lat.utterance_ready_ms.append(float(utt.latency_ms))
        if self.hub is not None:
            await self.hub.publish(utt)

    async def _propagate(self, batch: list[Event], snap: GameStats, new_ins: list[Insight]) -> None:
        """Hub + narration for events made outside the frame step (operator writes)."""
        await self._publish_events(batch)
        if self.hub is not None:
            await self.hub.publish(self._stats_msg(snap))
            display = self._display_frame()
            if display is not None:
                await self.hub.publish(frame_msg(display, self.roster, self.engine.holder_id,
                                                 self.engine.possession_team))
        if batch and not self._finished:
            await self.narrator.on_events(batch, snap, self._display_frame())
        if new_ins:
            await self._emit_insights(new_ins, snap)

    async def on_game_loop(self, fn: Callable[[], Awaitable[T]]) -> T:
        """Await ``fn()`` on the game's loop while the game runs (hops threads when called from
        another loop); before start / after finish it runs on the caller's loop."""
        loop = self._loop
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - always called from a coroutine
            current = None
        live = self._started and not self._finished
        if not live or loop is None or loop is current or loop.is_closed() or not loop.is_running():
            return await fn()
        fut = asyncio.run_coroutine_threadsafe(fn(), loop)  # type: ignore[arg-type]
        return await asyncio.wrap_future(fut)


# --------------------------------------------------------------------------- controller


def _event_from_payload(d: Mapping[str, Any], default_t: float, default_frame: int) -> Event:
    known = {f.name for f in dataclasses.fields(Event)}
    data = {k: v for k, v in d.items() if k in known}
    data["event_id"] = str(data.get("event_id") or "")
    if data.get("t") is None:
        data["t"] = default_t
    if data.get("frame_idx") is None:
        data["frame_idx"] = default_frame
    data["attrs"] = dict(data.get("attrs") or {})
    data["source"] = str(d.get("source") or "manual")
    data.setdefault("status", EventStatus.CONFIRMED.value)
    for k in ("kind", "status", "team"):
        if isinstance(data.get(k), (EventKind, EventStatus, Team)):
            data[k] = data[k].value
    try:
        return Event.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"bad manual event: {exc}") from exc


_SHOT_FAMILY = frozenset({EventKind.SHOT_ATTEMPT, EventKind.SHOT_MADE, EventKind.SHOT_MISSED})


def _team_of_id(roster: Roster, player_id: str | None) -> Team | None:
    """Roster team, else the team a provisional id names (\"home-t12\")."""
    if not player_id:
        return None
    return roster.team_of(player_id) or team_from_player_id(player_id)


def _sync_shot_family(engine: EventEngine, produced: list[Event], roster: Roster) -> list[Event]:
    """Apply an operator's change of scorer on a shot's result to the whole shot.

    The dashboard corrects ONE event, and the natural one is the make ("who scored"). The box score
    files a shot under its ``shot_id`` and credits the attempt's shooter, while the narrator (and
    its correction line) names the result's player, so revising only the make moved no points:
    "Correction: the basket belongs to X" was spoken with the points still on Y. Here a manual make
    / miss whose shooter (or team) differs from the shot's attempt revises the attempt too, and the
    shot's ASSIST / BLOCK follow the new shooter as ``other_player_id`` (retracted when they no
    longer fit: an assist to oneself or across teams, a block by a teammate). A corrected attempt
    already moves the box-score credit by itself and is left as sent.
    """
    out: list[Event] = []
    for rev in produced:
        if (rev.source != "manual" or rev.kind not in (EventKind.SHOT_MADE, EventKind.SHOT_MISSED)
                or not rev.shot_id or not rev.player_id or rev.status is EventStatus.RETRACTED):
            continue
        team = rev.team if rev.team is not None else _team_of_id(roster, rev.player_id)
        for sib in engine.events:
            if sib.shot_id != rev.shot_id or sib.event_id == rev.event_id or sib.status is EventStatus.RETRACTED:
                continue
            attrs = {**sib.attrs, "synced_from": rev.event_id}
            if sib.kind in _SHOT_FAMILY:
                if sib.player_id == rev.player_id and (team is None or sib.team is team):
                    continue
                fixed = dataclasses.replace(sib, player_id=rev.player_id, team=team if team is not None else sib.team,
                                            source="manual", attrs=attrs)
            elif sib.kind in (EventKind.ASSIST, EventKind.BLOCK):
                if sib.other_player_id == rev.player_id:
                    continue
                actor_team = sib.team if sib.team is not None else _team_of_id(roster, sib.player_id)
                same = None if team is None or actor_team is None else actor_team is team
                fits = sib.player_id != rev.player_id and (same is None or same == (sib.kind is EventKind.ASSIST))
                fixed = dataclasses.replace(sib, other_player_id=rev.player_id, source="manual", attrs=attrs) if fits \
                    else dataclasses.replace(sib, source="manual", status=EventStatus.RETRACTED,
                                             attrs={**attrs, "reason": f"shooter corrected to {rev.player_id}"})
            else:
                continue
            out += engine.add_manual(fixed)
    return out


class GameController:
    """``server.app.Controller`` for a :class:`LiveGame`. Every write runs its state change under the
    game lock (safe from any thread) and its narration / hub side effects on the game's loop."""

    def __init__(self, game: LiveGame) -> None:
        self.game = game

    # -- events / scoreboard / lineups ------------------------------------------------------------

    async def add_manual(self, event: dict[str, Any]) -> dict[str, Any]:
        g = self.game
        ev = _event_from_payload(event, g.current_t, g._frame_idx)
        with g._lock:
            known = bool(ev.event_id) and any(x.event_id == ev.event_id for x in g.engine.events)
            produced = g.engine.add_manual(ev)
            produced += _sync_shot_family(g.engine, produced, g.roster)
            for e in produced:
                # A substitution typed into the event dialog is a lineup entry too: without this the
                # feed and the narration showed it while on-court lineups, minutes and +/- ignored it
                # (stats take lineups only from set_lineup / substitute). New entries only: lineup
                # entries cannot be withdrawn, so revisions / retractions leave the lineup as is.
                if (e.kind is EventKind.SUBSTITUTION and not known and e.status is not EventStatus.RETRACTED
                        and e.player_id and e.other_player_id and e.player_id != e.other_player_id):
                    team = e.team if e.team is not None else _team_of_id(g.roster, e.player_id)
                    if team is not None:
                        g.stats.substitute(team, e.other_player_id, e.player_id, e.t)
            batch = g._ingest(produced)
            g.stats.on_events(batch)
            g._ins_pending.extend(batch)
            snap = g._snapshot(force=True)
            new_ins = g._update_insights(snap, force=True)
        await g.on_game_loop(lambda: g._propagate(batch, snap, new_ins))
        return {"events": [{"event_id": e.event_id, "kind": e.kind.value, "revision": e.revision,
                            "status": e.status.value} for e in batch]}

    async def set_scoreboard(self, reading: dict[str, Any]) -> dict[str, Any]:
        g = self.game
        with g._lock:
            base = g._board
            fields = ("home_score", "away_score", "period", "game_clock", "shot_clock")
            values = {k: reading.get(k, getattr(base, k) if base is not None else None) for k in fields}
            if all(v is None for v in values.values()):
                raise ValueError("set at least one of " + ", ".join(fields))
            r = ScoreboardReading(t=float(reading.get("t") if reading.get("t") is not None else g.current_t),
                                  source=str(reading.get("source") or "manual"), confidence=1.0, **values)
            g._pending_board = r  # the engine sees it with the next frame
            raw = g._raw_board
            for k in fields:
                if reading.get(k) is not None:  # held over later source readings until the source changes it
                    g._board_hold[k] = (values[k], getattr(raw, k) if raw is not None else None)
            g.stats.on_scoreboard(r)
            g._board = r
            snap = g._snapshot(force=True)
        await g.on_game_loop(lambda: g._propagate([], snap, []))
        return {"t": r.t, **values}

    async def set_lineup(self, entry: dict[str, Any]) -> dict[str, Any]:
        g = self.game
        with g._lock:
            batch = g._apply_lineup(entry)
            if batch:
                g.stats.on_events(batch)
                g._ins_pending.extend(batch)
            snap = g._snapshot(force=True)
            team = Team(entry["team"])
            on_court = g.stats.on_court(team)
        await g.on_game_loop(lambda: g._propagate(batch, snap, []))
        return {"team": team.value, "on_court": on_court, "events": [e.event_id for e in batch]}

    # -- calibration / scoreboard ROI ---------------------------------------------------------

    async def calibrate(self, request: dict[str, Any]) -> dict[str, Any]:
        g = self.game
        pts = np.asarray(request.get("image_points") or [], dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 4:
            raise ValueError("image_points needs >= 4 [u, v] pairs")
        src_size = g.source.image_size or (g._calibration.image_size if g._calibration else None)
        clicked = request.get("image_size")
        if clicked and src_size and tuple(int(v) for v in clicked) != tuple(src_size):
            pts = pts * np.array([src_size[0] / float(clicked[0]), src_size[1] / float(clicked[1])])
        thresh = request.get("ransac_thresh_ft", 1.0)
        if request.get("landmarks"):
            hom, mask, rmse = fit_landmarks(g.court, pts, list(request["landmarks"]), ransac_thresh_ft=thresh)
        elif request.get("court_points"):
            hom, mask, rmse = Homography.fit(pts, np.asarray(request["court_points"], dtype=float),
                                             ransac_thresh_ft=thresh)
        else:
            raise ValueError("send court_points or landmarks")
        size = tuple(src_size) if src_size else (int(pts[:, 0].max()) + 1, int(pts[:, 1].max()) + 1)
        old = g._calibration
        calib = Calibration(homography=hom, image_size=size, league=g.court.league,
                            rims=dict(old.rims) if old is not None else {})
        if not calib.rims:
            from courtvision.perception.footpoint import camera_from_homography

            camera = camera_from_homography(hom, size, (g.court.length, g.court.width))
            if camera is not None:
                for side in ("left", "right"):
                    x, y = g.court.basket_xy(side)
                    box = camera.bbox_of_circle(x, y, g.court.rim_height, g.court.rim_radius)
                    if (all(math.isfinite(v) for v in box.as_tuple()) and box.w > 0 and box.h > 0
                            and box.x2 > 0 and box.x1 < size[0] and box.y2 > 0 and box.y1 < size[1]):
                        calib.rims[side] = box

        def swap() -> bool:
            g._calibration = calib
            if g.perception is not None:
                g.perception.calibration = calib
                return True
            return False

        applied = await g._in_perception(swap)
        pc = g.config.perception
        if pc.save_calibration and pc.calibration_path:
            await asyncio.to_thread(calib.save, pc.calibration_path)
        if g._needs_calibration:
            g._needs_calibration = False

        async def release() -> None:
            g._release_calibration_hold()  # file playback held for this calibration continues
            if g.hub is not None:
                await g.hub.publish(status_msg(g.current_t, **g.status()))

        await g.on_game_loop(release)
        log.info("calibration fitted: rmse %.3f ft over %d/%d points (applied=%s)", rmse, int(mask.sum()), len(pts),
                 applied)
        return {"rmse_ft": float(rmse), "inliers": int(mask.sum()), "n_points": int(len(pts)), "applied": applied,
                "image_size": list(size)}

    async def set_scoreboard_roi(self, roi: dict[str, Any]) -> dict[str, Any]:
        from courtvision.perception.scoreboard_reader import ScoreboardROI

        g = self.game
        roi = dict(roi)
        clicked = roi.pop("image_size", None)
        src_size = g.source.image_size
        if clicked and src_size and tuple(int(v) for v in clicked) != tuple(src_size):
            sx, sy = src_size[0] / float(clicked[0]), src_size[1] / float(clicked[1])
            roi["corners"] = [[float(u) * sx, float(v) * sy] for u, v in roi["corners"]]
            roi["cells"] = {k: [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for k, b in roi["cells"].items()}
        parsed = ScoreboardROI.from_dict(roi)  # ValueError -> HTTP 400

        def swap() -> bool:
            if g.perception is not None:
                g.perception.set_scoreboard_roi(parsed)
                return True
            g._pending_roi = parsed.to_dict()
            return False

        applied = await g._in_perception(swap)
        pc = g.config.perception
        if pc.save_calibration and pc.scoreboard_roi_path:
            await asyncio.to_thread(Path(pc.scoreboard_roi_path).write_text, json.dumps(parsed.to_dict(), indent=2))
        return {"cells": {k: len(v) for k, v in parsed.cells.items()}, "applied": applied}

    # -- playback ----------------------------------------------------------------------------

    async def control(self, command: dict[str, Any]) -> dict[str, Any]:
        g = self.game
        action = command.get("action")
        if action == "pause":
            g.pacer.pause()
        elif action == "resume":
            g.pacer.resume()
        elif action == "speed":
            if g.source.live:  # a camera arrives in real time: the pacer is off, the speed would only be shown
                raise ValueError("a live camera plays in real time; speed does not apply")
            g.pacer.set_speed(float(command["value"]))
            if isinstance(g.clock, SpeedClock):
                g.clock.set_speed(g.pacer.speed)  # the narrator's holds and TTLs follow the video
        elif action in ("mute", "unmute"):
            await g.on_game_loop(lambda: g.tts.set_muted(action == "mute"))
        elif action == "dead_ball":
            with g._lock:
                g.engine.deadball.mark_dead_ball(g.current_t)
        else:
            raise ValueError(f"unknown control action {action!r}")
        status = g.status()
        if g.hub is not None:
            g.hub.publish_threadsafe(status_msg(g.current_t, **status))
        return status

    async def latest_frame_jpeg(self) -> bytes | None:
        g = self.game
        latest = g._latest_image
        quality = g.config.pipeline.jpeg_quality
        if latest is not None:
            return await asyncio.to_thread(_encode_jpeg, latest[1], quality)
        frame = g._latest_frame
        if frame is None:
            return None

        def render() -> bytes | None:
            img = g.source.render(frame.frame_idx)
            return _encode_jpeg(img, quality) if img is not None else None

        return await asyncio.to_thread(render)


# --------------------------------------------------------------------------- offline runs


@dataclass
class RunResult:
    """Everything an offline run produced (see :func:`run_offline`)."""

    config: AppConfig
    events: list[Event]  # latest revision of every event (RETRACTED included)
    truth_events: list[Event] | None
    stats: GameStats
    transcript: list[Utterance]
    insights: list[Insight]
    latency: dict[str, Any]
    event_report: EventReport | None = None
    box_report: BoxReport | None = None
    track_report: TrackReport | None = None
    run_dir: Path | None = None
    frames: int = 0
    source_kind: str = ""
    source_mode: str = ""
    narrator: dict[str, Any] = field(default_factory=dict)
    jev: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        ev = self.event_report
        box = self.box_report
        lat = self.latency
        return {
            "source": f"{self.source_kind}/{self.source_mode}",
            "frames": self.frames,
            "source_s": lat.get("source_s"),
            "wall_s": lat.get("wall_s"),
            "events": sum(1 for e in self.events if e.status is not EventStatus.RETRACTED),
            "utterances": len(self.transcript),
            "insights": len(self.insights),
            "headline_f1": ev.headline_macro_f1 if ev else None,
            "per_kind_f1": {k: r.f1 for k, r in ev.per_kind.items()} if ev else None,
            "box_mean_mae": box.mean_mae if box else None,
            "team_points_abs_error": box.team_points_abs_error if box else None,
            "mota": self.track_report.mota if self.track_report else None,
        }


async def arun_offline(cfg: AppConfig, source: FrameSource | None = None, out_dir: str | Path | None = None, *,
                       hub: Hub | None = None, jev: JevDecider | None = None, llm: NarrationLLM | None = None,
                       write: bool = True, env: Mapping[str, str] | None = None,
                       detector: Any | None = None, on_frame: Callable[[FrameState], None] | None = None) -> RunResult:
    """Async :func:`run_offline`."""
    src = source if source is not None else make_source(cfg, realtime=False)
    game = LiveGame(cfg, src, hub=hub, realtime=False, jev=jev, llm=llm, env=env, detector=detector, on_frame=on_frame)
    await game.run()
    result = _collect(game)
    if write:
        result.run_dir = write_run_dir(result, out_dir if out_dir is not None else _default_run_dir(cfg))
    return result


def run_offline(cfg: AppConfig, source: FrameSource | None = None, out_dir: str | Path | None = None, *,
                hub: Hub | None = None, jev: JevDecider | None = None, llm: NarrationLLM | None = None,
                write: bool = True, env: Mapping[str, str] | None = None, detector: Any | None = None,
                on_frame: Callable[[FrameState], None] | None = None) -> RunResult:
    """Run a whole source as fast as possible (source-time clocks: deterministic) and evaluate it.

    With a simulator source the result carries the truth events and the ``courtvision.eval``
    reports (events always; box score when the run covers the whole simulation; tracking for the
    ``"video"`` mode). Results are written to ``out_dir`` (default ``cfg.pipeline.runs_dir/run-<time>``)
    unless ``write=False``.
    """
    return asyncio.run(arun_offline(cfg, source, out_dir, hub=hub, jev=jev, llm=llm, write=write, env=env,
                                    detector=detector, on_frame=on_frame))


def _default_run_dir(cfg: AppConfig) -> Path:
    return Path(cfg.pipeline.runs_dir) / time.strftime("run-%Y%m%d-%H%M%S")


def _collect(game: LiveGame) -> RunResult:
    cfg, src = game.config, game.source
    stats = game.final_stats or game.snapshot()
    events = game.events
    truth_events = event_report = box_report = track_report = None
    if src.truth is not None:
        lo, hi = src.start_t - 1e-6, (src.end_t if src.end_t is not None else math.inf) + 1e-6
        truth_events = [e for e in src.truth.events if lo <= e.t <= hi]
        # The stream is cut at its end (and at its start when it begins mid-game): unmatched events at a
        # cut edge are not scored. Operator-entered kinds are matched with the operator's modelled lag.
        window = (src.start_t if src.start_t > 1e-6 else None, src.end_t)
        ecfg = EventEvalConfig(operator_lag_s={"substitution": float(cfg.operator.entry_delay_s)})
        event_report = match_events(events, truth_events, tol_s=cfg.eval.tol_s, match_player=cfg.eval.match_player,
                                    config=ecfg, window=window)
        whole = isinstance(src, SimSource) and src.window == (0, len(src.truth.frames))
        if whole:
            # the game is cut at the stream end: truth from its last moments (a basket the operator board
            # shows only after the stop) is "don't care" for the box score, as for events
            guard = float(cfg.eval.tol_s)
            if getattr(src, "scoreboard_policy", "truth") == "operator":
                guard = max(guard, float(cfg.board.score_lag_s[1]) + 2.0 / float(cfg.board.ocr_hz))
            hi_t = src.end_t if src.end_t is not None else math.inf
            edge = edge_contributions([e for e in truth_events if e.t > hi_t - guard])
            box_report = compare_box(stats, src.truth.truth_box, src.truth.truth_score, game.roster, edge=edge)
        kept = game.kept_frames
        truth_frames = src.truth_frames()
        if kept and truth_frames:
            track_report = tracking_report(kept, truth_frames, match_radius_ft=cfg.eval.track_match_radius_ft)
    insights = insight_order(game.insights_emitted)
    return RunResult(
        config=cfg, events=events, truth_events=truth_events, stats=stats, transcript=game.transcript,
        insights=insights, latency=game.latency_report(), event_report=event_report, box_report=box_report,
        track_report=track_report, frames=game._frames_done, source_kind=src.kind, source_mode=src.mode,
        narrator=game.narrator.stats(), jev=game.jev.stats() if game.jev is not None else None)


def _jsonl(path: Path, rows: Iterable[Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(dumps(row) + "\n")


def _latency_table(lat: Mapping[str, Any]) -> list[str]:
    rows = ["## Latency", "", "| stage | n | mean | p50 | p95 | max |", "|---|---|---|---|---|---|"]
    labels = {
        "perception_ms_per_frame": "perception (ms/frame)",
        "engine_ms_per_frame": "event engine (ms/frame)",
        "stats_ms_per_frame": "stats + insights frame fold (ms/frame)",
        "frame_step_ms": "whole frame step (ms)",
        "detection_delay_ms": "event t -> engine emission (source ms)",
        "event_to_utterance_ms": "engine emission -> utterance (source ms)",
        "occurrence_to_utterance_ms": "event t -> utterance (source ms)",
        "event_to_utterance_wall_ms": "engine emission -> utterance (wall ms)",
        "utterance_ready_ms": "narrator ready latency (ms)",
    }
    for key, label in labels.items():
        s = lat.get(key)
        if not s:
            rows.append(f"| {label} | 0 | n/a | n/a | n/a | n/a |")
            continue
        rows.append(f"| {label} | {s['n']} | {s['mean']:.2f} | {s['p50']:.2f} | {s['p95']:.2f} | {s['max']:.2f} |")
    rt = lat.get("realtime_factor")
    rows += ["", f"Source {lat.get('source_s', 0.0):.1f} s processed in {lat.get('wall_s', 0.0):.1f} s wall"
             + (f" ({rt:.1f}x real time)." if rt else "."), ""]
    return rows


def write_run_dir(result: RunResult, out_dir: str | Path) -> Path:
    """events.jsonl, truth_events.jsonl, transcript.jsonl, insights.jsonl, stats.json, report.json,
    report.md (eval report + latency table) and config.yaml."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    _jsonl(d / "events.jsonl", (e.to_dict() for e in result.events))
    if result.truth_events is not None:
        _jsonl(d / "truth_events.jsonl", (e.to_dict() for e in result.truth_events))
    _jsonl(d / "transcript.jsonl", result.transcript)
    _jsonl(d / "insights.jsonl", result.insights)
    (d / "stats.json").write_text(dumps(result.stats.to_dict()))
    reports = report_dict(result.event_report, result.box_report, result.track_report)
    (d / "report.json").write_text(dumps({"summary": result.summary(), "eval": reports, "latency": result.latency,
                                          "narrator": result.narrator, "jev": result.jev}))
    lines = [f"Source: {result.source_kind} / {result.source_mode}, {result.frames} frames, "
             f"{len(result.events)} events, {len(result.transcript)} utterances, {len(result.insights)} insights.", ""]
    if result.event_report or result.box_report or result.track_report:
        lines.append(format_report(result.event_report, result.box_report, result.track_report))
    else:
        lines.append("# CourtVision run\n\nNo simulator truth: evaluation skipped.\n")
    lines += _latency_table(result.latency)
    (d / "report.md").write_text("\n".join(lines))
    (d / "config.yaml").write_text(config_to_yaml(result.config))
    log.info("run written to %s", d)
    return d
