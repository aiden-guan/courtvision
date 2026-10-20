"""Audio-free dead-ball detector (RESEARCH.md §7A.5).

``dead_ball`` fires on any of:

1. the OCR game clock is unchanged for >= 2 consecutive reads while it should be running;
2. free-throw formation: one player stationary at the FT line and >= 4 players set on the lane
   lines, held for >= 1.5 s;
3. global stillness: median player speed < 2 ft/s for >= 2 s, with the ball held by a referee
   (no player holder), out of bounds, or not visible;
4. the operator hotkey (``mark_dead_ball``).

Callers (the event engine) add typed hints with ``note_stoppage`` (made basket, out of bounds,
timeout, foul, period break) and live-play evidence with ``note_live``. ``expected_window_s``
turns the stoppage type into the remaining expected dead time using ``RuleSpec`` (timeout
lengths, period intervals) and unmeasured placeholders for the rest (§7A.3).
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from courtvision.geometry.court import CourtSpec
from courtvision.geometry.rules import RuleSpec
from courtvision.types import FrameState, PlayerState, Side

__all__ = ["STOPPAGES", "DeadBallConfig", "DeadBallDetector", "DeadBallState", "Formation", "ft_formation"]

log = logging.getLogger(__name__)

STOPPAGES: tuple[str, ...] = ("timeout", "free_throw", "period_break", "out_of_bounds", "foul", "made_basket", "unknown")
# Lower = more specific / longer; used to label a dead ball that several rules support.
_PRIORITY = {"period_break": 0, "timeout": 1, "free_throw": 2, "foul": 3, "out_of_bounds": 4, "made_basket": 5, "unknown": 6}
# Stoppages that stop the game clock, so a running clock is live-play evidence against them.
_CLOCK_STOPPING = frozenset({"timeout", "free_throw", "period_break", "out_of_bounds", "foul", "unknown"})


@dataclass
class DeadBallConfig:
    """Thresholds of the four rules and the expected-window placeholders."""

    clock_stopped_reads: int = 2  # rule 1: unchanged for >= 2 consecutive reads (§7A.5); sourced (proposal)
    clock_stopped_min_s: float = 1.2  # a MM:SS display repeats a value for up to 1 s while running; tuned
    ft_formation_hold_s: float = 1.5  # rule 2 hold (§7A.5); sourced (proposal)
    ft_spot_radius_ft: float = 3.0  # FT shooter within this of the FT-line centre; tuned
    ft_spot_inside_ft: float = 1.0  # ... at most this far inside the line toward the basket; tuned
    ft_shooter_max_speed_fts: float = 2.0  # "stationary" FT shooter; tuned
    ft_lane_min_players: int = 4  # rule 2: >= 4 players set on the lane lines (§7A.5); sourced (proposal)
    ft_lane_outside_ft: float = 4.0  # lane-spot band outside the lane line; tuned
    ft_lane_inside_ft: float = 1.0  # ... and inside it; tuned
    ft_lane_near_baseline_ft: float = 4.0  # lane spots start this far from the baseline; tuned
    ft_lane_max_speed_fts: float = 3.0  # "set" lane players move slower than this; tuned
    stillness_speed_fts: float = 2.0  # rule 3: median speed < 2 ft/s (§7A.5); sourced (proposal)
    stillness_hold_s: float = 2.0  # rule 3 hold (§7A.5); sourced (proposal)
    stillness_min_players: int = 4  # need this many players for a median; tuned
    live_speed_fts: float = 4.0  # median speed above this ... ; tuned
    live_hold_s: float = 0.5  # ... for this long ends a stillness dead ball; tuned
    hotkey_max_s: float = 120.0  # operator dead ball lasts until live evidence or this; tuned
    hint_max_s: float = 60.0  # hinted stoppages expire after this without live evidence; tuned
    window_free_throw_s: float = 10.0  # FT dead time is unmeasured (§7A.3); unverified
    window_out_of_bounds_s: float = 6.0  # unmeasured; unverified
    window_foul_s: float = 12.0  # unmeasured; unverified
    window_made_basket_s: float = 4.0  # unmeasured; NFHS clock keeps running [241]; unverified
    window_unknown_s: float = 5.0  # unmeasured; unverified
    window_timeout_s: float | None = None  # None -> RuleSpec timeouts.length_s (NBA 75, FIBA/NFHS 60); sourced
    window_period_break_s: float = 120.0  # fallback when RuleSpec has no intervals; unverified
    history: int = 64  # past dead intervals kept for is_dead(t) queries


@dataclass(frozen=True, slots=True)
class Formation:
    """A free-throw formation: shooter at ``side``'s FT line, ``lane_ids`` set on the lane lines."""

    side: Side
    shooter_id: str
    lane_ids: tuple[str, ...]
    since_t: float = 0.0


@dataclass(frozen=True, slots=True)
class DeadBallState:
    dead: bool
    since_t: float | None
    stoppage: str | None
    reasons: tuple[str, ...]


def ft_formation(court: CourtSpec, players: Sequence[PlayerState], cfg: DeadBallConfig,
                 shooter_id: str | None = None) -> Formation | None:
    """Free-throw formation now (no hold time), or None. ``shooter_id`` restricts the shooter."""
    for side in ("left", "right"):
        f = _formation_at(court, players, cfg, side, shooter_id)  # type: ignore[arg-type]
        if f is not None:
            return f
    return None


def _formation_at(court: CourtSpec, players: Sequence[PlayerState], cfg: DeadBallConfig, side: Side,
                  shooter_id: str | None) -> Formation | None:
    fx, fy = court.free_throw_xy(side)
    sign = 1.0 if side == "left" else -1.0
    shooters = []
    for p in players:
        if shooter_id is not None and p.player_id != shooter_id:
            continue
        inside = (fx - p.x) * sign  # > 0: toward the basket from the FT line
        if inside <= cfg.ft_spot_inside_ft and ((p.x - fx) ** 2 + (p.y - fy) ** 2) ** 0.5 <= cfg.ft_spot_radius_ft \
                and p.speed <= cfg.ft_shooter_max_speed_fts:
            shooters.append(p)
    if not shooters:
        return None
    shooter = min(shooters, key=lambda p: (p.x - fx) ** 2 + (p.y - fy) ** 2)
    baseline_x = 0.0 if side == "left" else court.length
    half_lane = court.lane_width / 2.0
    lane = []
    for p in players:
        if p.player_id == shooter.player_id or p.speed > cfg.ft_lane_max_speed_fts:
            continue
        along = (p.x - baseline_x) * sign
        lateral = abs(p.y - court.basket_y)
        if cfg.ft_lane_near_baseline_ft <= along <= court.free_throw_line_from_baseline \
                and half_lane - cfg.ft_lane_inside_ft <= lateral <= half_lane + cfg.ft_lane_outside_ft:
            lane.append(p.player_id)
    if len(lane) < cfg.ft_lane_min_players:
        return None
    return Formation(side=side, shooter_id=shooter.player_id, lane_ids=tuple(lane))


class DeadBallDetector:
    """Dead-ball state from the scoreboard clock, formations, stillness, hints and the hotkey."""

    def __init__(self, court: CourtSpec, rules: RuleSpec | None = None, config: DeadBallConfig | None = None) -> None:
        self.court = court
        self.rules = rules
        self.config = config or DeadBallConfig()
        self._reasons: dict[str, tuple[float, str]] = {}  # reason -> (since_t, stoppage)
        self._last_t: float | None = None
        self._period: int | None = None
        self._ended_period: int | None = None
        self._clock_value: float | None = None
        self._clock_since: float | None = None
        self._clock_reads = 0
        self._still_since: float | None = None
        self._moving_since: float | None = None
        self.formation: Formation | None = None  # current formation (with since_t), even before the hold
        self._intervals: deque[tuple[float, float, str]] = deque(maxlen=self.config.history)

    # ------------------------------------------------------------------ inputs

    def mark_dead_ball(self, t: float, stoppage: str = "unknown") -> None:
        """Operator hotkey (rule 4)."""
        self._set("hotkey", t, _checked(stoppage))

    def note_stoppage(self, t: float, stoppage: str) -> None:
        """Typed hint from the engine or a manual event (made basket, OOB, timeout, foul, period break)."""
        stoppage = _checked(stoppage)
        if stoppage == "period_break" and self._period is not None:
            self._ended_period = self._period
        self._set("hint", t, stoppage)

    def note_live(self, t: float) -> None:
        """Live-play evidence (inbound touched, jump ball, ball released in play)."""
        for reason in ("hint", "hotkey", "stillness", "clock"):
            self._clear(reason, t)

    def update(self, frame: FrameState, holder_id: str | None = None, ball_out: bool = False,
               ball_visible: bool | None = None) -> DeadBallState:
        """Advance with one frame; ``holder_id`` is the engine's confirmed player holder."""
        t = frame.t
        self._last_t = t
        visible = ball_visible if ball_visible is not None else bool(frame.ball and frame.ball.visible)
        self._rule_clock(frame, t)
        self._rule_formation(frame.players, t)
        self._rule_stillness(frame.players, t, holder_id is None or not visible or ball_out)
        self._expire(t)
        return self.state

    # ------------------------------------------------------------------ queries

    @property
    def state(self) -> DeadBallState:
        if not self._reasons:
            return DeadBallState(False, None, None, ())
        since = min(s for s, _ in self._reasons.values())
        return DeadBallState(True, since, self._label(), tuple(sorted(self._reasons)))

    def is_dead(self, t: float | None = None) -> bool:
        if t is None or self._last_t is None or t >= self._last_t - 1e-9:
            return bool(self._reasons)
        if self._reasons and min(s for s, _ in self._reasons.values()) <= t:
            return True
        return any(a <= t <= b for a, b, _ in self._intervals)

    def stoppage(self) -> str | None:
        return self._label() if self._reasons else None

    def expected_window_s(self, t: float | None = None) -> float:
        """Remaining expected dead time (s) at ``t``; 0.0 when the ball is live."""
        st = self.state
        if not st.dead or st.since_t is None or st.stoppage is None:
            return 0.0
        now = t if t is not None else (self._last_t or st.since_t)
        return max(0.0, self.expected_total_s(st.stoppage) - max(0.0, now - st.since_t))

    def expected_total_s(self, stoppage: str) -> float:
        """Expected total length of a stoppage of this type (RuleSpec where it defines one)."""
        cfg, rules = self.config, self.rules
        if stoppage == "timeout":
            if cfg.window_timeout_s is not None:
                return cfg.window_timeout_s
            length = (rules.timeouts or {}).get("length_s") if rules else None
            return float(length) if length else 60.0
        if stoppage == "period_break":
            return self._period_break_s()
        return {
            "free_throw": cfg.window_free_throw_s, "out_of_bounds": cfg.window_out_of_bounds_s,
            "foul": cfg.window_foul_s, "made_basket": cfg.window_made_basket_s,
        }.get(stoppage, cfg.window_unknown_s)

    # ------------------------------------------------------------------ rules

    def _rule_clock(self, frame: FrameState, t: float) -> None:
        sb = frame.scoreboard
        if sb is None or sb.game_clock is None:
            return
        if sb.period is not None and sb.period != self._period:
            if self._period is not None:
                self._ended_period = self._period
            self._period = sb.period
        gc = float(sb.game_clock)
        if self._clock_value is not None and abs(gc - self._clock_value) < 1e-6:
            self._clock_reads += 1
        else:
            running = self._clock_value is not None and gc < self._clock_value
            self._clock_value, self._clock_since, self._clock_reads = gc, t, 1
            if running:
                self._clear("clock", t)
                self._clear("clock_zero", t)
                for reason in ("hint", "hotkey"):
                    if reason in self._reasons and self._reasons[reason][1] in _CLOCK_STOPPING:
                        self._clear(reason, t)
        cfg = self.config
        if gc <= 0.05:
            # like rule 1, a zero needs >= 2 identical reads: one misread "0:00" is not a period break
            if self._clock_reads >= cfg.clock_stopped_reads:
                if "clock_zero" not in self._reasons:
                    self._ended_period = self._period
                self._set("clock_zero", self._clock_since or t, "period_break")
            return
        if self._clock_reads >= cfg.clock_stopped_reads and t - (self._clock_since or t) >= cfg.clock_stopped_min_s:
            self._set("clock", self._clock_since or t, self._hint_stoppage() or "unknown")

    def _rule_formation(self, players: Sequence[PlayerState], t: float) -> None:
        f = ft_formation(self.court, players, self.config)
        if f is None:
            self.formation = None
            self._clear("formation", t)
            return
        prev = self.formation
        since = prev.since_t if prev is not None and (prev.side, prev.shooter_id) == (f.side, f.shooter_id) else t
        self.formation = Formation(f.side, f.shooter_id, f.lane_ids, since)
        if t - since >= self.config.ft_formation_hold_s:
            self._set("formation", since, "free_throw")

    def _rule_stillness(self, players: Sequence[PlayerState], t: float, ball_idle: bool) -> None:
        cfg = self.config
        if len(players) < cfg.stillness_min_players:
            return
        med = statistics.median(p.speed for p in players)
        if med < cfg.stillness_speed_fts:
            self._moving_since = None
            if self._still_since is None:
                self._still_since = t
            if ball_idle and t - self._still_since >= cfg.stillness_hold_s:
                self._set("stillness", self._still_since, self._hint_stoppage() or "unknown")
            return
        self._still_since = None
        if med > cfg.live_speed_fts:
            if self._moving_since is None:
                self._moving_since = t
            if t - self._moving_since >= cfg.live_hold_s:
                self._clear("stillness", t)

    def _expire(self, t: float) -> None:
        cfg = self.config
        for reason, limit in (("hotkey", cfg.hotkey_max_s), ("hint", cfg.hint_max_s)):
            if reason in self._reasons and t - self._reasons[reason][0] > limit:
                self._clear(reason, t)

    # ------------------------------------------------------------------ helpers

    def _set(self, reason: str, since: float, stoppage: str) -> None:
        cur = self._reasons.get(reason)
        if cur is not None and cur[1] == stoppage:
            return
        was_dead = bool(self._reasons)
        self._reasons[reason] = (min(since, cur[0]) if cur else since, stoppage)
        if not was_dead:
            log.debug("dead ball (%s, %s) from t=%.2f", reason, stoppage, since)

    def _clear(self, reason: str, t: float) -> None:
        if reason not in self._reasons:
            return
        label = self._label()
        since = min(s for s, _ in self._reasons.values())
        del self._reasons[reason]
        if not self._reasons:
            self._intervals.append((since, t, label))

    def _hint_stoppage(self) -> str | None:
        for reason in ("hotkey", "hint"):
            if reason in self._reasons:
                return self._reasons[reason][1]
        return None

    def _label(self) -> str:
        labels = [s for _, s in self._reasons.values()]
        return min(labels, key=lambda s: _PRIORITY.get(s, 99)) if labels else "unknown"

    def _period_break_s(self) -> float:
        rules, cfg = self.rules, self.config
        if rules is None or not rules.intervals:
            return cfg.window_period_break_s
        ended = self._ended_period or self._period or 1
        iv = rules.intervals
        half = 1 if rules.halves else rules.n_periods // 2
        if ended == half and "halftime_s" in iv:
            return float(iv["halftime_s"])
        if ended >= rules.n_periods and "before_ot_s" in iv:
            return float(iv["before_ot_s"])
        return float(iv.get("between_periods_s", cfg.window_period_break_s))


def _checked(stoppage: str) -> str:
    if stoppage not in STOPPAGES:
        raise ValueError(f"unknown stoppage {stoppage!r}; expected one of {STOPPAGES}")
    return stoppage
