"""Core data model shared by every CourtVision module.

This file is the contract between perception, the event engine, stats, the Jev
decision layer, narration, the simulator and the server. Change it deliberately.

Conventions
-----------
* Court coordinates are in FEET. Origin is the corner where the left baseline
  meets the bottom sideline. ``x`` runs along the court length (0 = left
  baseline), ``y`` along the width (0 = bottom sideline). ``z`` is height above
  the floor. FIBA courts are converted to feet at the geometry boundary.
* Image coordinates are pixels, origin top-left, ``u`` right, ``v`` down.
* ``t`` is video time in seconds since the start of the source (monotonic).
  Game clock / shot clock are *remaining* seconds and only come from a
  scoreboard reading or an official feed.
* Every inferred quantity carries a ``confidence`` in [0, 1].
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

Side = Literal["left", "right"]


class Team(str, enum.Enum):
    HOME = "home"
    AWAY = "away"

    def other(self) -> Team:
        return Team.AWAY if self is Team.HOME else Team.HOME


class ObjClass(str, enum.Enum):
    PLAYER = "player"
    REFEREE = "referee"
    BALL = "ball"
    RIM = "rim"


# --------------------------------------------------------------------------- image space


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned pixel box (x1, y1) top-left to (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-centre point: the standard ground-contact proxy for homography."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def iou(self, other: BBox) -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def expand(self, frac: float) -> BBox:
        dx, dy = self.w * frac / 2.0, self.h * frac / 2.0
        return BBox(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy)

    def contains(self, u: float, v: float) -> bool:
        return self.x1 <= u <= self.x2 and self.y1 <= v <= self.y2

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(slots=True)
class Detection:
    """One detector output in one frame."""

    bbox: BBox
    cls: ObjClass
    score: float
    embedding: np.ndarray | None = None  # appearance feature (colour histogram, SigLIP, ...)
    jersey: str | None = None  # per-frame jersey-number read, if available
    jersey_conf: float = 0.0


@dataclass(slots=True)
class FrameDetections:
    frame_idx: int
    t: float
    image_size: tuple[int, int]  # (width, height)
    detections: list[Detection] = field(default_factory=list)

    def of(self, cls: ObjClass) -> list[Detection]:
        return [d for d in self.detections if d.cls is cls]


@dataclass(slots=True)
class Track:
    """A short-term tracker output (one per live track per frame)."""

    track_id: int
    cls: ObjClass
    bbox: BBox
    score: float
    hits: int = 1  # number of matched detections so far
    age: int = 1  # frames since the track was created
    time_since_update: int = 0  # 0 when matched this frame
    embedding: np.ndarray | None = None  # running-average appearance feature
    jersey: str | None = None  # last per-frame jersey read attached to this track
    jersey_conf: float = 0.0


@dataclass(frozen=True, slots=True)
class RimObservation:
    """A rim in image space (from a detector or from a fixed-camera calibration)."""

    side: Side
    bbox: BBox
    confidence: float = 1.0


# --------------------------------------------------------------------------- roster


@dataclass(frozen=True, slots=True)
class RosterEntry:
    player_id: str  # stable id, e.g. "home-23"
    team: Team
    jersey: str  # as printed: "0" and "00" are different numbers
    name: str
    position: str | None = None


@dataclass(slots=True)
class Roster:
    entries: list[RosterEntry]
    team_names: dict[Team, str] = field(default_factory=lambda: {Team.HOME: "Home", Team.AWAY: "Away"})

    def get(self, player_id: str | None) -> RosterEntry | None:
        if player_id is None:
            return None
        for e in self.entries:
            if e.player_id == player_id:
                return e
        return None

    def by_jersey(self, team: Team, jersey: str) -> RosterEntry | None:
        for e in self.entries:
            if e.team is team and e.jersey == jersey:
                return e
        return None

    def team_of(self, player_id: str | None) -> Team | None:
        e = self.get(player_id)
        return e.team if e else None

    def display_name(self, player_id: str | None) -> str:
        e = self.get(player_id)
        if e is None:
            return "unknown player" if player_id is None else player_id
        return e.name


# --------------------------------------------------------------------------- world state per frame


@dataclass(slots=True)
class PlayerState:
    """A player (never a referee) located on the court at one instant."""

    player_id: str  # roster id when identified, else provisional "<team>-t<track_id>" / "unk-t<track_id>"
    team: Team | None
    x: float  # court ft
    y: float  # court ft
    vx: float = 0.0  # ft/s
    vy: float = 0.0  # ft/s
    jersey: str | None = None
    track_id: int | None = None
    bbox: BBox | None = None  # image bbox when produced from video
    confidence: float = 1.0

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))


@dataclass(slots=True)
class BallState:
    """Ball observation at one instant.

    ``image_xy`` is authoritative when it exists (monocular video). ``court_xy``
    is the floor-plane projection of the image point (only physically right when
    the ball is near the floor) or the true ground position in simulations /
    3-D tracking feeds. ``z`` is only set when height is actually known.
    """

    visible: bool
    image_xy: tuple[float, float] | None = None
    court_xy: tuple[float, float] | None = None
    z: float | None = None
    radius_px: float | None = None
    confidence: float = 0.0
    interpolated: bool = False  # True when filled by the ball tracker, not observed


@dataclass(frozen=True, slots=True)
class ScoreboardReading:
    """Scoreboard / official-feed snapshot (OCR, manual operator, or data feed)."""

    t: float
    home_score: int | None = None
    away_score: int | None = None
    period: int | None = None
    game_clock: float | None = None  # seconds remaining in period
    shot_clock: float | None = None  # seconds remaining
    source: str = "manual"  # "ocr" | "manual" | "feed" | "sim"
    confidence: float = 1.0


@dataclass(slots=True)
class FrameState:
    """Everything known about one instant. Produced by perception or the simulator."""

    frame_idx: int
    t: float
    players: list[PlayerState]
    ball: BallState | None
    rims: list[RimObservation] = field(default_factory=list)  # empty when there is no camera
    scoreboard: ScoreboardReading | None = None
    image_size: tuple[int, int] | None = None
    # Camera centre (court ft, z up) when perception knows it exactly (a broadcast game camera tracked about its
    # fixed centre); the event engine then uses it for monocular depth instead of estimating a fixed camera.
    camera_xyz: tuple[float, float, float] | None = None
    # The detector's ball candidates this frame, (n, 3) image x, y, score (None: not kept). The engine tracks one ball;
    # the shot judge (nba/live_judge.py) reads every candidate about the rims.
    balls: Any = None
    # Each rim in view cut out square about itself (nba/rim_clips.crop_rim; None: not cut), for the shot judge's rim model
    rim_crops: Any = None

    def player(self, player_id: str) -> PlayerState | None:
        for p in self.players:
            if p.player_id == player_id:
                return p
        return None


# --------------------------------------------------------------------------- events


class EventKind(str, enum.Enum):
    POSSESSION = "possession"  # a team gains possession; attrs["cause"]
    PASS = "pass"  # player_id passer -> other_player_id receiver
    DRIBBLE = "dribble"  # one bounce by player_id (high-frequency, not narrated)
    SHOT_ATTEMPT = "shot_attempt"  # at release; value = 1/2/3 potential points
    SHOT_MADE = "shot_made"  # shot_id links to the attempt; value = points scored
    SHOT_MISSED = "shot_missed"
    BLOCK = "block"  # player_id blocker, other_player_id shooter
    REBOUND = "rebound"  # player_id None => team rebound; attrs["offensive"]
    ASSIST = "assist"  # player_id assister, other_player_id scorer
    STEAL = "steal"  # player_id stealer, other_player_id victim
    TURNOVER = "turnover"  # player_id committer (None => team turnover); attrs["type"]
    FOUL = "foul"  # manual / feed only
    SUBSTITUTION = "substitution"  # player_id in, other_player_id out
    TIMEOUT = "timeout"
    PERIOD_START = "period_start"
    PERIOD_END = "period_end"
    SCORE_CORRECTION = "score_correction"  # reconciliation against scoreboard


class EventStatus(str, enum.Enum):
    PROVISIONAL = "provisional"  # inferred, may still change
    CONFIRMED = "confirmed"  # corroborated (scoreboard, enough evidence, or manual)
    RETRACTED = "retracted"  # withdrawn; consumers must undo its effects


@dataclass(slots=True)
class Event:
    """A basketball event.

    Revisions: an engine may re-emit an event with the same ``event_id`` and a
    higher ``revision`` (changed fields or ``status=RETRACTED``). Consumers must
    keep only the latest revision per ``event_id`` (stats are recomputed from
    the latest revisions, never accumulated blindly).
    """

    event_id: str
    kind: EventKind
    t: float
    frame_idx: int
    team: Team | None = None
    player_id: str | None = None
    other_player_id: str | None = None
    x: float | None = None  # court ft where it happened (shot release point for shots)
    y: float | None = None
    value: int = 0
    shot_id: str | None = None
    confidence: float = 1.0
    status: EventStatus = EventStatus.PROVISIONAL
    source: str = "cv"  # "cv" | "scoreboard" | "manual" | "sim" | "feed"
    revision: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["status"] = self.status.value
        d["team"] = self.team.value if self.team else None
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Event:
        d = dict(d)
        d["kind"] = EventKind(d["kind"])
        d["status"] = EventStatus(d.get("status", "provisional"))
        d["team"] = Team(d["team"]) if d.get("team") else None
        d.setdefault("attrs", {})
        return Event(**d)


# Well-known ``Event.attrs`` keys (use these spellings everywhere):
#   SHOT_ATTEMPT: "shot_type" in {"jumper","layup","dunk","hook","tip","free_throw"},
#                 "distance_ft", "closest_defender_ft", "catch_and_shoot" (bool),
#                 "dribbles_before" (int), "xfg" (0..1), "basket" (Side), "zone" (str)
#   REBOUND:      "offensive" (bool)
#   TURNOVER:     "type" in {"steal","bad_pass","out_of_bounds","shot_clock","travel","offensive_foul","other"}
#   POSSESSION:   "cause" in {"jump_ball","inbound","defensive_rebound","offensive_rebound","steal",
#                             "turnover","made_shot","free_throw","period_start","other"}
#   PASS:         "distance_ft", "completed" (bool)
#   any:          "reason" (str, human-readable evidence), "needs_review" (bool)


# --------------------------------------------------------------------------- narration output


@dataclass(slots=True)
class Utterance:
    """One unit of narration (fan, coach, or accessibility channel)."""

    utterance_id: str
    text: str
    channel: Literal["play_by_play", "color", "coach", "accessibility", "correction"]
    t: float  # video time it refers to
    event_ids: list[str] = field(default_factory=list)
    source: Literal["llm", "template"] = "template"
    priority: int = 0  # higher preempts lower
    latency_ms: float | None = None  # event -> text ready
    validated: bool = True


@dataclass(slots=True)
class Insight:
    """A coach/staff-facing observation backed by computed numbers."""

    insight_id: str
    t: float
    kind: str  # e.g. "foul_trouble", "hot_hand", "run", "shot_quality", "lineup", "fatigue", "decision"
    text: str
    priority: int = 0
    team: Team | None = None
    player_id: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
