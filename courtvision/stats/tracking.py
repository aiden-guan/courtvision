"""Player tracking stats from per-frame court positions plus pass / dribble events.

Frame-derived (``TrackingAccumulator.update``, called from ``StatsEngine.on_frame``):

* distance: sum of frame-to-frame steps. A step whose implied speed exceeds ``max_speed_fts``
  (30 ft/s, the perception clamp in ARCHITECTURE) is an identity jump, not running: it adds no
  distance and no tracked time, and the track restarts at the new position (``clamped_steps``).
  A gap longer than ``max_gap_s`` also restarts the track.
* top speed: max of the windowed speed, i.e. the straight-line displacement over the last
  ``speed_window_s`` divided by that time (only once a track has been followed that long). It is
  exact for steady motion and averages position jitter out instead of reading it as motion.
  Average speed: distance / seconds tracked.
* ``position_smoothing_tau_s`` > 0 additionally measures distance on EMA-smoothed positions (off by
  default: perception already smooths court positions, and simulator positions are exact).
* time of possession: each frame interval is credited to the holder at its start.
* touches: holder transitions to a player; a brief loss and regain by the same player within
  ``touch_regain_gap_s`` (holder flicker) is not a new touch. A paint touch is a touch that starts
  inside the paint of the basket the holder's team attacks.

Event-derived (``event_counts``, recomputed from the latest revisions): passes made (every PASS
thrown), passes received (completed passes only) and dribbles.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any

from courtvision.types import Event, EventKind, EventStatus, FrameState, Side, Team

if TYPE_CHECKING:
    from courtvision.geometry.court import CourtSpec

__all__ = ["TrackingLine", "TrackingAccumulator", "event_counts"]

log = logging.getLogger(__name__)

AttackFn = Callable[[Team], "Side | None"]


@dataclass
class TrackingLine:
    """Per-player tracking totals."""

    player_id: str
    distance_ft: float = 0.0
    top_speed_fts: float = 0.0
    avg_speed_fts: float = 0.0
    seconds_tracked: float = 0.0
    time_of_possession_s: float = 0.0
    touches: int = 0
    paint_touches: int = 0
    passes_made: int = 0
    passes_received: int = 0
    dribbles: int = 0
    clamped_steps: int = 0  # steps ignored by the speed clamp (likely identity switches)

    def to_dict(self) -> dict[str, Any]:
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrackingLine:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class _Track:
    last_t: float
    x: float  # last raw position (identity-jump test)
    y: float
    sx: float  # position used for distance (smoothed when enabled)
    sy: float
    distance: float = 0.0
    seconds: float = 0.0
    top_speed: float = 0.0
    window: deque[tuple[float, float, float]] = field(default_factory=deque)  # (t, sx, sy) since restart
    clamped: int = 0

    def restart(self, t: float, x: float, y: float) -> None:
        self.sx, self.sy = x, y
        self.window.clear()
        self.window.append((t, x, y))


class TrackingAccumulator:
    """Accumulates frame-derived tracking stats; parameters are documented on ``StatsConfig``."""

    def __init__(
        self,
        court: CourtSpec | None = None,
        *,
        max_speed_fts: float = 30.0,
        speed_window_s: float = 0.5,
        max_gap_s: float = 0.5,
        touch_regain_gap_s: float = 0.5,
        position_smoothing_tau_s: float = 0.0,
    ) -> None:
        self.court = court
        self.max_speed_fts = max_speed_fts
        self.speed_window_s = speed_window_s
        self.position_smoothing_tau_s = position_smoothing_tau_s
        self.max_gap_s = max_gap_s
        self.touch_regain_gap_s = touch_regain_gap_s
        self._tracks: dict[str, _Track] = {}
        self._top: Counter[str] = Counter()
        self._touches: Counter[str] = Counter()
        self._paint_touches: Counter[str] = Counter()
        self._prev_t: float | None = None
        self._prev_holder: str | None = None
        self._last_holder: str | None = None
        self._last_holder_t: float = -math.inf

    # ------------------------------------------------------------------ per frame

    def update(self, frame: FrameState, holder_id: str | None, attacking: AttackFn | None = None) -> None:
        """Fold one frame in. Frames must arrive in increasing ``t``; stale frames are ignored."""
        if self._prev_t is not None and frame.t <= self._prev_t:
            log.debug("tracking: frame %d at t=%.3f is not newer than %.3f; skipped",
                      frame.frame_idx, frame.t, self._prev_t)
            return
        for p in frame.players:
            self._step(p.player_id, frame.t, p.x, p.y)
        self._possession(frame, holder_id, attacking)
        self._prev_t = frame.t
        self._prev_holder = holder_id

    def _step(self, pid: str, t: float, x: float, y: float) -> None:
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        tr = self._tracks.get(pid)
        if tr is None:
            tr = self._tracks[pid] = _Track(last_t=t, x=x, y=y, sx=x, sy=y)
            tr.restart(t, x, y)
            return
        dt = t - tr.last_t
        if dt <= 0:
            return
        if dt > self.max_gap_s:  # lost too long: restart without a step
            tr.restart(t, x, y)
        elif math.hypot(x - tr.x, y - tr.y) / dt > self.max_speed_fts:  # identity jump
            tr.clamped += 1
            tr.restart(t, x, y)
        else:
            a = _alpha(dt, self.position_smoothing_tau_s)
            nx, ny = tr.sx + a * (x - tr.sx), tr.sy + a * (y - tr.sy)
            tr.distance += math.hypot(nx - tr.sx, ny - tr.sy)
            tr.seconds += dt
            tr.sx, tr.sy = nx, ny
            self._window_speed(tr, t)
        tr.last_t, tr.x, tr.y = t, x, y

    def _window_speed(self, tr: _Track, t: float) -> None:
        w = tr.window
        w.append((t, tr.sx, tr.sy))
        while len(w) >= 2 and t - w[1][0] >= self.speed_window_s - 1e-9:
            w.popleft()
        t0, x0, y0 = w[0]
        span = t - t0
        if span > 0 and span >= self.speed_window_s - 1e-9:
            tr.top_speed = max(tr.top_speed, math.hypot(tr.sx - x0, tr.sy - y0) / span)

    def _possession(self, frame: FrameState, holder_id: str | None, attacking: AttackFn | None) -> None:
        if self._prev_t is not None and self._prev_holder is not None:
            dt = frame.t - self._prev_t
            if 0 < dt <= self.max_gap_s:
                self._top[self._prev_holder] += dt
        if holder_id is not None and holder_id != self._prev_holder:
            regained = holder_id == self._last_holder and frame.t - self._last_holder_t <= self.touch_regain_gap_s
            if not regained:
                self._touches[holder_id] += 1
                if self._in_attacking_paint(frame, holder_id, attacking):
                    self._paint_touches[holder_id] += 1
        if holder_id is not None:
            self._last_holder, self._last_holder_t = holder_id, frame.t

    def _in_attacking_paint(self, frame: FrameState, holder_id: str, attacking: AttackFn | None) -> bool:
        if self.court is None or attacking is None:
            return False
        p = frame.player(holder_id)
        if p is None or p.team is None:
            return False
        side = attacking(p.team)
        return side is not None and self.court.in_paint(p.x, p.y, side)

    # ------------------------------------------------------------------ output

    def lines(self, counts: dict[str, Counter[str]] | None = None) -> dict[str, TrackingLine]:
        """Tracking lines for every player seen in frames or in ``counts`` (see ``event_counts``)."""
        counts = counts or {}
        ids = set(self._tracks) | set(self._top) | set(self._touches) | set(counts)
        out: dict[str, TrackingLine] = {}
        for pid in sorted(ids):
            tr = self._tracks.get(pid)
            c = counts.get(pid, Counter())
            line = TrackingLine(
                player_id=pid,
                time_of_possession_s=self._top.get(pid, 0.0),
                touches=self._touches.get(pid, 0),
                paint_touches=self._paint_touches.get(pid, 0),
                passes_made=c.get("passes_made", 0),
                passes_received=c.get("passes_received", 0),
                dribbles=c.get("dribbles", 0),
            )
            if tr is not None:
                line.distance_ft = tr.distance
                line.seconds_tracked = tr.seconds
                line.top_speed_fts = tr.top_speed
                line.avg_speed_fts = tr.distance / tr.seconds if tr.seconds > 0 else 0.0
                line.clamped_steps = tr.clamped
            out[pid] = line
        return out


def _alpha(dt: float, tau: float) -> float:
    """EMA gain for a step of ``dt`` with time constant ``tau`` (``tau <= 0``: no smoothing)."""
    return 1.0 if tau <= 0 else 1.0 - math.exp(-dt / tau)


def event_counts(events: Iterable[Event]) -> dict[str, Counter[str]]:
    """Passes made / received and dribbles per player from active events."""
    out: dict[str, Counter[str]] = {}

    def bump(pid: str | None, key: str) -> None:
        if pid is not None:
            out.setdefault(pid, Counter())[key] += 1

    for e in events:
        if e.status is EventStatus.RETRACTED:
            continue
        if e.kind is EventKind.PASS:
            bump(e.player_id, "passes_made")
            if e.attrs.get("completed", True) is not False:
                bump(e.other_player_id, "passes_received")
        elif e.kind is EventKind.DRIBBLE:
            bump(e.player_id, "dribbles")
    return out
