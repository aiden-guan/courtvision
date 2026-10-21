"""Application configuration: one :class:`AppConfig` composing every module's config dataclass.

``load_config(path, overrides)`` reads an optional YAML file and ``"a.b=value"`` overrides and builds
the dataclass tree by field name (ARCHITECTURE "Global conventions"):

* unknown keys raise :class:`ConfigError` naming the dotted path (``events.ball_min_conff``);
* values are coerced to the annotated field type: ints/floats/bools/strings, ``tuple[...]`` from
  YAML lists, ``list``/``dict`` element types, ``X | None``, ``Literal`` and nested dataclasses;
* nested dataclasses are partial (missing keys keep their defaults) and ``dict`` fields merge key
  by key onto their default dict, so ``narration.max_words.color=40`` keeps the other channels;
  lists and scalars replace.

``dump_config(cfg)`` is the inverse (plain dicts/lists, YAML-ready). ``render_default_yaml()``
renders the defaults with comments taken from the dataclass sources; it is what
``courtvision/config/default.yaml`` contains, so the file cannot drift from the code
(``tests/test_integration_core_config.py`` checks both).

Integration-owned sections (``game``, ``source``, ``operator``, ``perception``, ``shot_model``,
``decision_model``, ``llm``, ``serve``, ``pipeline``, ``eval``) are defined here; every other
section is the module's own config dataclass, unchanged.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import io
import logging
import re
import tokenize
import types
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from courtvision.decision.client import JevConfig
from courtvision.events.engine import EventEngineConfig
from courtvision.narration.captions import CaptionConfig
from courtvision.narration.candidates import CandidateConfig
from courtvision.narration.director import DirectorConfig
from courtvision.narration.facts import FactsConfig
from courtvision.narration.narrator import NarrationConfig
from courtvision.narration.templates import TemplateConfig
from courtvision.narration.tts import TTSConfig
from courtvision.perception.detector import DetectorConfig
from courtvision.perception.jersey import TemplateJerseyConfig
from courtvision.perception.pipeline import PerceptionConfig
from courtvision.perception.rfdetr_adapter import RFDETRConfig
from courtvision.server.app import ServerConfig
from courtvision.server.hub import HubConfig
from courtvision.sim.noise import NoiseConfig, ScoreboardConfig
from courtvision.sim.render import RenderConfig
from courtvision.sim.simulator import SimConfig
from courtvision.stats.decision import DecisionConfig
from courtvision.stats.engine import StatsConfig
from courtvision.stats.insights import InsightConfig
from courtvision.stats.shot_quality import ShotQualityConfig

__all__ = [
    "AppConfig",
    "ConfigError",
    "DEFAULT_CONFIG_PATH",
    "DecisionModelSettings",
    "EvalSettings",
    "GameConfig",
    "LLMSettings",
    "OperatorConfig",
    "PerceptionSettings",
    "PipelineSettings",
    "ServeSettings",
    "ShotModelSettings",
    "SourceConfig",
    "apply_overrides",
    "build_dataclass",
    "config_to_yaml",
    "dump_config",
    "load_config",
    "render_default_yaml",
    "write_default_yaml",
]

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "default.yaml"

SOURCE_KINDS = ("sim", "video", "camera")
SIM_MODES = ("truth3d", "mono", "video")
SIM_SCOREBOARDS = ("auto", "truth", "operator", "ocr_5hz", "none")
DETECTORS = ("auto", "color_blob", "rfdetr")
JERSEY_READERS = ("auto", "template", "none")


class ConfigError(ValueError):
    """Invalid configuration: unknown key, wrong type or invalid value (message names the path)."""


# --------------------------------------------------------------------------- integration sections


@dataclass
class GameConfig:
    """Who plays and under which rules (authoritative over sim.league / sim.ruleset / sim.rule_overrides)."""

    league: str = "nba"  # court geometry: "nba" | "fiba" | "ncaa_m" | "ncaa_w" | "nfhs"
    ruleset: str | None = None  # RuleSpec id in config/rules/; None = the league's own rules
    rule_overrides: list[str] = field(default_factory=list)  # names in config/rules/overrides/ or YAML paths
    roster_path: str | None = None  # JSON/YAML roster for video/camera sources (sim makes its own)
    home_name: str | None = None  # team display names (None = roster file / simulator names)
    away_name: str | None = None


@dataclass
class SourceConfig:
    """Where frames come from."""

    kind: str = "sim"  # "sim" | "video" (file) | "camera" (index or URL)
    sim_mode: str = "truth3d"  # sim: "truth3d" (3-D truth) | "mono" (NoiseModel, monocular) | "video" (render + perception)
    path: str | None = None  # video file for kind="video"
    camera: str = "0"  # device index ("0") or stream URL for kind="camera"
    realtime: bool = True  # pace frames at t / speed (run_offline always runs as fast as possible)
    speed: float = 1.0  # playback speed multiplier for realtime pacing
    start_s: float = 0.0  # skip source time before this (sim window / video seek)
    duration_s: float | None = None  # stop after this many source seconds (None = whole source)
    noise_seed: int = 1  # NoiseModel seed for sim_mode="mono"
    # sim scoreboard: "auto" = "truth" for truth3d, "operator" for mono / video | "truth" (exact, every
    # frame, source "sim") | "operator" (operator-run board read by 5 Hz OCR: board.* settings) |
    # "ocr_5hz" (exact values at 5 Hz) | "none"
    scoreboard: str = "auto"
    drop_late_s: float | None = None  # realtime: drop frames more than this late (None = never drop)
    fps: float | None = None  # override the container fps for video files (None = from the file)


@dataclass
class OperatorConfig:
    """Simulated operator (sim sources): lineups and the kinds the CV engine never infers."""

    entry_delay_s: float = 2.0  # operator lag after a true substitution (SimOperatorConfig); design choice
    lineup_t: float = 0.0  # video time the starting fives are entered
    lineup_events: bool = True  # operator lineup entries also become manual SUBSTITUTION events
    replay_kinds: list[str] = field(default_factory=lambda: ["foul", "timeout"])  # truth kinds entered as manual events
    replay_delay_s: float = 0.0  # delay of those manual entries after the truth event


@dataclass
class PerceptionSettings:
    """Detector choice, the perception pipeline and the venue calibration files."""

    detector: str = "auto"  # "auto" (colour blobs for sim renders, RF-DETR for real video) | "color_blob" | "rfdetr"
    color_blob: DetectorConfig = field(default_factory=DetectorConfig)  # ColorBlobDetector (synthetic renders)
    rfdetr: RFDETRConfig = field(default_factory=RFDETRConfig)  # RFDETRDetector (extra `cv`)
    pipeline: PerceptionConfig = field(default_factory=PerceptionConfig)  # PerceptionPipeline settings
    jersey_reader: str = "auto"  # "auto" (template reader on sim renders, none otherwise) | "template" | "none"
    jersey: TemplateJerseyConfig = field(default_factory=TemplateJerseyConfig)  # TemplateJerseyReader
    calibration_path: str | None = None  # geometry Calibration JSON (homography + rims) for video/camera
    scoreboard_roi_path: str | None = None  # ScoreboardROI JSON (overrides pipeline.scoreboard_roi)
    save_calibration: bool = False  # write POST /api/calibration and /api/scoreboard_roi results back to the paths
    use_source_hints: bool = True  # sim renders: take calibration, ROI, jersey candidates and foot radius from the sim


@dataclass
class ShotModelSettings:
    """Shot-quality (xFG) model."""

    enabled: bool = True
    path: str | None = None  # ShotQualityModel.to_json() file from training/fit_shot_model.py; None = priors
    priors: ShotQualityConfig = field(default_factory=ShotQualityConfig)  # used when path is None


@dataclass
class DecisionModelSettings:
    """EPV-lite decision grades (needs the shot model)."""

    enabled: bool = True
    config: DecisionConfig = field(default_factory=DecisionConfig)


@dataclass
class LLMSettings:
    """Claude client knobs not in NarrationConfig (model / effort / max_tokens come from narration.*)."""

    cache_ttl: str | None = "1h"  # prompt-cache TTL: Claude runs rarely (timeouts, breaks), so keep the prefix 1 h
    prewarm: bool = True  # pre-warm the cached system prefix at start and periodically
    prewarm_interval_s: float | None = None  # None = auto: 55 min with a 1 h TTL, 4 min with the 5 min default
    answer_directly: bool = False  # C2 system line (RESEARCH §7A.3); keep off until measured
    fallbacks: str = "default"  # server-side refusal fallback (opus-class models)
    timeout_s: float = 30.0  # HTTP timeout; candidate deadlines are enforced by the narrator
    max_retries: int = 0  # speculative lines are never retried
    offline: bool = False  # allow Claude in non-realtime runs (video time races ahead of wall time)


@dataclass
class ServeSettings:
    """HTTP server binding (used by the CLI)."""

    host: str = "127.0.0.1"  # local in-gym server by default
    port: int = 8765  # dashboard port (`courtvision serve --port`)
    auth_token_env: str = "COURTVISION_TOKEN"  # bearer token read from this env var (never from the file)


@dataclass
class PipelineSettings:
    """LiveGame orchestration rates and outputs."""

    frame_hz: float = 15.0  # frame messages to the hub (video-time rate)
    stats_hz: float = 1.0  # stats snapshots to the hub
    insights_hz: float = 1.0  # InsightEngine.update between event batches
    tick_hz: float = 10.0  # Narrator.tick (dead-ball colour scheduling)
    sim_assume_calibrated: bool = True  # sim sources are simulator evaluations: cue models report "assumed_sim"
    keep_frames: bool = True  # keep perceived frames of offline sim video runs for the tracking report
    max_kept_frames: int = 36000  # cap on kept frames (20 min at 30 fps)
    tail_s: float = 15.0  # source-time seconds narration may finish after the last frame
    runs_dir: str = "runs"  # run_offline output root when no directory is given
    jpeg_quality: int = 80  # GET /api/frame.jpg quality
    captions_sidecar: bool = True  # caption / AD cues -> hub "caption" messages
    webvtt_path: str | None = None  # also write an HLS WebVTT caption file
    status_hz: float = 1.0  # status messages (pause / speed / lag) to the hub
    hub_decisions: int = 40  # newest decision grades per hub stats message, slimmed (stats.json keeps all)


@dataclass
class EvalSettings:
    """Evaluation against simulator truth."""

    tol_s: float = 1.0  # event matching tolerance (ARCHITECTURE)
    match_player: bool = True
    track_match_radius_ft: float = 3.0  # tracking_report match radius


@dataclass
class AppConfig:
    """Every CourtVision setting. Sections map 1:1 to YAML top-level keys."""

    game: GameConfig = field(default_factory=GameConfig)  # league, rules and roster
    source: SourceConfig = field(default_factory=SourceConfig)  # frame source and pacing
    sim: SimConfig = field(default_factory=SimConfig)  # simulator (league/ruleset come from game.*)
    noise: NoiseConfig = field(default_factory=NoiseConfig)  # observation noise for source.sim_mode="mono"
    render: RenderConfig = field(default_factory=RenderConfig)  # synthetic renderer for source.sim_mode="video"
    board: ScoreboardConfig = field(default_factory=ScoreboardConfig)  # sim venue scoreboard: operator lag, 5 Hz OCR
    operator: OperatorConfig = field(default_factory=OperatorConfig)  # simulated operator entries
    perception: PerceptionSettings = field(default_factory=PerceptionSettings)  # detector + PerceptionPipeline
    events: EventEngineConfig = field(default_factory=EventEngineConfig)  # event engine thresholds
    stats: StatsConfig = field(default_factory=StatsConfig)  # box score / tracking / lineups
    shot_model: ShotModelSettings = field(default_factory=ShotModelSettings)  # xFG model
    decision_model: DecisionModelSettings = field(default_factory=DecisionModelSettings)  # EPV-lite grades
    insights: InsightConfig = field(default_factory=InsightConfig)  # coach cards
    jev: JevConfig = field(default_factory=JevConfig)  # TypeSafe Jev (live needs TYPESAFE_API_KEY; "stub" for demos)
    narration: NarrationConfig = field(default_factory=NarrationConfig)  # narrator (mode, Claude model, holds)
    director: DirectorConfig = field(default_factory=DirectorConfig)  # speak-now / priority / preemption
    facts: FactsConfig = field(default_factory=FactsConfig)  # FactPacket gates
    templates: TemplateConfig = field(default_factory=TemplateConfig)  # template lines
    candidates: CandidateConfig | None = None  # Claude candidate pool (None = derived from narration.*)
    tts: TTSConfig = field(default_factory=TTSConfig)  # voice (engine chosen by narration.tts)
    captions: CaptionConfig = field(default_factory=CaptionConfig)  # caption / AD cue timing
    llm: LLMSettings = field(default_factory=LLMSettings)  # Claude client (needs Anthropic credentials)
    server: ServerConfig = field(default_factory=ServerConfig)  # FastAPI app
    hub: HubConfig = field(default_factory=HubConfig)  # WebSocket hub
    serve: ServeSettings = field(default_factory=ServeSettings)  # host / port / token env var
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)  # LiveGame rates and outputs
    eval: EvalSettings = field(default_factory=EvalSettings)  # truth comparison (sim sources)

    def validate(self) -> AppConfig:
        """Cross-field checks that types alone cannot express; returns ``self``."""
        _choice("source.kind", self.source.kind, SOURCE_KINDS)
        _choice("source.sim_mode", self.source.sim_mode, SIM_MODES)
        _choice("source.scoreboard", self.source.scoreboard, SIM_SCOREBOARDS)
        _choice("perception.detector", self.perception.detector, DETECTORS)
        _choice("perception.jersey_reader", self.perception.jersey_reader, JERSEY_READERS)
        _choice("narration.mode", self.narration.mode, ("broadcast", "coach", "accessibility"))
        _choice("narration.tts", self.narration.tts, ("avspeech", "say", "none", "null"))
        _choice("jev.mode", self.jev.mode, ("live", "stub", "replay", "off"))
        if self.source.speed <= 0:
            raise ConfigError(f"source.speed must be > 0, got {self.source.speed}")
        if self.source.kind == "video" and not self.source.path:
            raise ConfigError("source.path is required when source.kind is 'video'")
        for name in ("frame_hz", "stats_hz", "insights_hz", "tick_hz", "status_hz"):
            if getattr(self.pipeline, name) <= 0:
                raise ConfigError(f"pipeline.{name} must be > 0")
        from courtvision.geometry.court import court_for
        from courtvision.geometry.rules import rules_for

        try:
            court_for(self.game.league)
            rules_for(self.game.ruleset or self.game.league, list(self.game.rule_overrides) or None)
        except (ValueError, FileNotFoundError) as exc:
            raise ConfigError(f"game: {exc}") from exc
        unknown = [k for k in self.operator.replay_kinds if k not in {e.value for e in _event_kinds()}]
        if unknown:
            raise ConfigError(f"operator.replay_kinds: unknown event kind(s) {unknown}")
        return self


def _event_kinds() -> type[enum.Enum]:
    from courtvision.types import EventKind

    return EventKind


def _choice(path: str, value: Any, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ConfigError(f"{path} must be one of {list(choices)}, got {value!r}")


# --------------------------------------------------------------------------- dict -> dataclass

_HINTS: dict[type, dict[str, Any]] = {}


def _hints(cls: type) -> dict[str, Any]:
    hints = _HINTS.get(cls)
    if hints is None:
        try:
            hints = typing.get_type_hints(cls)
        except Exception:  # noqa: BLE001 - a TYPE_CHECKING-only name: resolve field by field
            hints = {}
            module = inspect.getmodule(cls)
            ns = dict(vars(module)) if module else {}
            for f in fields(cls):
                ann = f.type
                if isinstance(ann, str):
                    try:
                        ann = eval(ann, ns)  # noqa: S307 - annotations of our own dataclasses
                    except Exception:  # noqa: BLE001
                        ann = Any
                hints[f.name] = ann
        _HINTS[cls] = hints
    return hints


def _join(path: str, key: Any) -> str:
    return f"{path}.{key}" if path else str(key)


def build_dataclass(cls: type, data: Any, path: str = "", base: Any = None) -> Any:
    """Build dataclass ``cls`` from a mapping (partial: missing keys keep defaults / ``base``)."""
    if isinstance(data, cls):
        return dataclasses.replace(data)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path or cls.__name__}: expected a mapping for {cls.__name__}, got {type(data).__name__}")
    hints = _hints(cls)
    names = {f.name: f for f in fields(cls) if f.init}
    unknown = sorted(str(k) for k in data if k not in names)
    if unknown:
        raise ConfigError(f"unknown config key(s) {[_join(path, k) for k in unknown]}; "
                          f"{cls.__name__} has: {sorted(names)}")
    current = base if base is not None else _default_instance(cls, path)
    kwargs: dict[str, Any] = {}
    for name in names:
        default_value = getattr(current, name)
        if name in data:
            kwargs[name] = _coerce(data[name], hints.get(name, Any), _join(path, name), default_value)
        else:
            kwargs[name] = default_value
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path or cls.__name__}: {exc}") from exc


def _default_instance(cls: type, path: str) -> Any:
    try:
        return cls()
    except TypeError as exc:
        raise ConfigError(f"{path or cls.__name__}: {cls.__name__} has required fields: {exc}") from exc


def _is_optional(tp: Any) -> bool:
    origin = typing.get_origin(tp)
    return (origin is typing.Union or origin is types.UnionType) and type(None) in typing.get_args(tp)


def _coerce(value: Any, tp: Any, path: str, default: Any = None) -> Any:
    """Coerce ``value`` to type ``tp`` (``default`` is the current value, used to merge dicts)."""
    if tp is Any or tp is object:
        return value
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is types.UnionType:
        args = typing.get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{path}: null is not allowed (expected {_tname(tp)})")
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(value, arg, path, default if _fits(default, arg) else None)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(errors[0] if len(errors) == 1 else f"{path}: {value!r} does not fit {_tname(tp)}")
    if value is None:
        raise ConfigError(f"{path}: null is not allowed (expected {_tname(tp)})")
    if origin is typing.Literal:
        if value not in typing.get_args(tp):
            raise ConfigError(f"{path}: {value!r} is not one of {list(typing.get_args(tp))}")
        return value
    if isinstance(tp, type) and is_dataclass(tp):
        base = default if isinstance(default, tp) else None
        return build_dataclass(tp, value, path, base=base)
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        try:
            return tp(value.value if isinstance(value, enum.Enum) else value)
        except ValueError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
    if tp is bool:
        return _to_bool(value, path)
    if tp is int:
        return _to_int(value, path)
    if tp is float:
        return _to_float(value, path)
    if tp is str:
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            return str(value)
        hint = " (YAML reads unquoted on/off/yes/no as booleans: quote the value, e.g. 'off')" if isinstance(
            value, bool) else ""
        raise ConfigError(f"{path}: expected a string, got {type(value).__name__}{hint}")
    if origin in (list, tuple, dict) or tp in (list, tuple, dict):
        return _coerce_container(value, tp, origin or tp, path, default)
    if isinstance(tp, type) and isinstance(value, tp):
        return value
    raise ConfigError(f"{path}: cannot coerce {type(value).__name__} to {_tname(tp)}")


def _coerce_container(value: Any, tp: Any, origin: Any, path: str, default: Any) -> Any:
    args = typing.get_args(tp)
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        kt, vt = args if len(args) == 2 else (Any, Any)
        out: dict[Any, Any] = dict(default) if isinstance(default, dict) else {}  # merge onto the default
        for k, v in value.items():
            key = _coerce(k, kt, _join(path, k))
            out[key] = _coerce(v, vt, _join(path, k), out.get(key))
        return out
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
    if origin is list:
        et = args[0] if args else Any
        return [_coerce(v, et, f"{path}[{i}]") for i, v in enumerate(value)]
    # tuple
    if not args:
        return tuple(value)
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(_coerce(v, args[0], f"{path}[{i}]") for i, v in enumerate(value))
    if len(value) != len(args):
        raise ConfigError(f"{path}: expected {len(args)} items, got {len(value)}")
    return tuple(_coerce(v, a, f"{path}[{i}]") for i, (v, a) in enumerate(zip(value, args)))


def _fits(value: Any, tp: Any) -> bool:
    if value is None:
        return False
    origin = typing.get_origin(tp) or tp
    try:
        return isinstance(value, origin)
    except TypeError:
        return False


def _to_bool(value: Any, path: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "yes", "on", "1", "false", "no", "off", "0"):
        return value.strip().lower() in ("true", "yes", "on", "1")
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ConfigError(f"{path}: expected true/false, got {value!r}")


def _to_int(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{path}: expected an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise ConfigError(f"{path}: expected an integer, got {value!r}")


def _to_float(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{path}: expected a number, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    raise ConfigError(f"{path}: expected a number, got {value!r}")


def _tname(tp: Any) -> str:
    return getattr(tp, "__name__", None) or str(tp).replace("typing.", "")


# --------------------------------------------------------------------------- overrides


_YAML_BOOL_WORDS = frozenset({"true", "false"})


def _parse_value(text: str) -> Any:
    """An override value as YAML, except that YAML 1.1 boolean words other than true / false
    (``off``, ``on``, ``yes``, ``no``, ...) stay strings: ``jev.mode=off`` means the string "off"
    (bool fields still accept those words, see ``_to_bool``)."""
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    if isinstance(value, bool) and text.strip().lower() not in _YAML_BOOL_WORDS:
        return text.strip()
    return value


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``"a.b.c=value"`` overrides to a raw config dict (values parsed as YAML scalars/flow)."""
    out = data
    for item in overrides or []:
        if not isinstance(item, str) or "=" not in item:
            raise ConfigError(f"override {item!r} must look like 'section.key=value'")
        key, _, raw = item.partition("=")
        parts = [p for p in key.strip().split(".")]
        if not parts or any(not p for p in parts):
            raise ConfigError(f"override {item!r}: empty key segment")
        _check_override_path(parts, item)
        node = out
        for p in parts[:-1]:
            nxt = node.get(p)
            if nxt is None:
                nxt = node[p] = {}
            elif not isinstance(nxt, dict):
                raise ConfigError(f"override {item!r}: {p!r} is not a section")
            node = nxt
        node[parts[-1]] = _parse_value(raw.strip())
    return out


def _check_override_path(parts: list[str], item: str) -> None:
    """Walk the AppConfig type tree so a misspelt override fails with its dotted path."""
    tp: Any = AppConfig
    walked: list[str] = []
    for p in parts:
        while _is_optional(tp):
            tp = next(a for a in typing.get_args(tp) if a is not type(None))
        if isinstance(tp, type) and is_dataclass(tp):
            hints = _hints(tp)
            if p not in {f.name for f in fields(tp)}:
                raise ConfigError(f"unknown config key {'.'.join([*walked, p])!r} in override {item!r}")
            tp = hints.get(p, Any)
        elif typing.get_origin(tp) is dict or tp is dict:
            args = typing.get_args(tp)
            tp = args[1] if len(args) == 2 else Any
        else:
            raise ConfigError(f"override {item!r}: {'.'.join(walked)!r} is not a section")
        walked.append(p)


# --------------------------------------------------------------------------- public API


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> AppConfig:
    """Defaults <- optional YAML file <- ``overrides``; validated."""
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        loaded = yaml.safe_load(p.read_text())
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{p}: top level must be a mapping, got {type(loaded).__name__}")
        data = loaded
    data = apply_overrides(data, overrides)
    cfg = build_dataclass(AppConfig, data)
    return cfg.validate()


def dump_config(cfg: Any) -> Any:
    """Dataclass tree -> plain dicts / lists / scalars (YAML / JSON ready)."""
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return {f.name: dump_config(getattr(cfg, f.name)) for f in fields(cfg)}
    if isinstance(cfg, enum.Enum):
        return cfg.value
    if isinstance(cfg, dict):
        return {(k.value if isinstance(k, enum.Enum) else k): dump_config(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [dump_config(v) for v in cfg]
    if isinstance(cfg, Path):
        return str(cfg)
    return cfg


def config_to_yaml(cfg: AppConfig) -> str:
    """Plain YAML of a config (no comments), e.g. the run directory's ``config.yaml``."""
    return yaml.safe_dump(dump_config(cfg), sort_keys=False, default_flow_style=False, width=120)


# --------------------------------------------------------------------------- default.yaml

_HEADER = """\
# CourtVision default configuration -- GENERATED from the config dataclasses.
# Do not edit by hand: run `python -m courtvision.config` after changing a default.
# tests/test_integration_core_config.py fails when this file drifts from the code.
#
# Load with courtvision.config.load_config(path, overrides=["section.key=value", ...]).
# Copy only the keys you change into your own file; missing keys keep these defaults.
# Nested sections are partial, dict-valued keys merge key by key, lists replace.
# game.league / game.ruleset / game.rule_overrides drive the court, the rules AND the simulator
# (sim.league / sim.ruleset / sim.rule_overrides are overwritten from game.*).
# Credentials never live here: ANTHROPIC_API_KEY (or an `ant auth login` profile) enables Claude,
# TYPESAFE_API_KEY enables live Jev; without them narration is template-only and Jev falls back.
"""

_COMMENT_MAX = 110


def _field_comments(cls: type) -> dict[str, str]:
    """Trailing ``# comment`` of each field declaration in the dataclass source (else the comment
    block right above it)."""
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return {}
    names = {f.name for f in fields(cls)}
    comments: dict[int, str] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                comments.setdefault(tok.start[0], tok.string.lstrip("#").strip())
    except (tokenize.TokenError, IndentationError):
        return {}
    lines = src.splitlines()
    decl = re.compile(r"^(\s+)([A-Za-z_]\w*)\s*:")
    body_indent: str | None = None
    out: dict[str, str] = {}
    for i, line in enumerate(lines, start=1):
        m = decl.match(line)
        if not m:
            continue
        indent, name = m.group(1), m.group(2)
        if body_indent is None:
            body_indent = indent
        if indent != body_indent or name not in names or name in out:
            continue
        text = comments.get(i)
        if text is None:  # comment block above the field
            block = []
            j = i - 1
            while j >= 1 and lines[j - 1].strip().startswith("#"):
                block.insert(0, lines[j - 1].strip().lstrip("#").strip())
                j -= 1
            text = " ".join(block) if block else None
        if text:
            out[name] = text if len(text) <= _COMMENT_MAX else text[: _COMMENT_MAX - 3].rstrip() + "..."
    return out


def _flow(key: str, value: Any) -> str:
    text = yaml.safe_dump({key: value}, default_flow_style=True, sort_keys=False, width=1_000_000,
                          allow_unicode=True).strip()
    return text[1:-1] if text.startswith("{") and text.endswith("}") else text


def _emit(obj: Any, indent: int, out: list[str]) -> None:
    pad = " " * indent
    comments = _field_comments(type(obj))
    for f in fields(obj):
        value = getattr(obj, f.name)
        note = comments.get(f.name)
        suffix = f"  # {note}" if note else ""
        if is_dataclass(value) and not isinstance(value, type):
            out.append(f"{pad}{f.name}:{suffix}")
            _emit(value, indent + 2, out)
        else:
            out.append(f"{pad}{_flow(f.name, dump_config(value))}{suffix}")


def render_default_yaml() -> str:
    """The commented YAML of ``AppConfig()`` (the content of ``courtvision/config/default.yaml``)."""
    cfg = AppConfig()
    out: list[str] = [_HEADER.rstrip("\n")]
    section_notes = _field_comments(AppConfig)
    for f in fields(cfg):
        value = getattr(cfg, f.name)
        out.append("")
        note = section_notes.get(f.name)
        if note:
            out.append(f"# {f.name}: {note}")
        if is_dataclass(value):
            out.append(f"{f.name}:")
            _emit(value, 2, out)
        else:
            out.append(_flow(f.name, dump_config(value)))
    return "\n".join(out) + "\n"


def write_default_yaml(path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_default_yaml())
    return p


if __name__ == "__main__":  # pragma: no cover - regenerate config/default.yaml
    print(write_default_yaml())
