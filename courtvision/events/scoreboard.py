"""Scoreboard confirmation (RESEARCH.md D14, §4.2, §7A.5).

``ScoreboardConfirmer`` takes raw ``ScoreboardReading``s (OCR at 5 Hz, a data feed, the operator,
or the simulator) and turns them into confirmed changes:

* a score change from OCR is confirmed by 2 identical consecutive reads whose per-team delta is
  +1, +2 or +3 and whose game clock is consistent (non-increasing, and not falling faster than
  real time); other deltas (negative = scorer correction, > 3 = missed reads) need
  ``jump_confirm_reads`` identical reads and are flagged ``needs_review``;
* ``manual`` / ``feed`` / ``sim`` sources confirm immediately;
* period numbers confirm the same way and produce ``PeriodChange``s.

The confirmer also tracks the game clock so callers can tell whether it is running.
``ClockZeroWatch`` confirms that a game or shot clock ran out (2 OCR reads at zero after the clock
was seen running, as D14 does for scores).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Union

from courtvision.types import ScoreboardReading, Team

__all__ = ["ClockZeroWatch", "PeriodChange", "ScoreChange", "ScoreboardConfig", "ScoreboardConfirmer", "ScoreboardUpdate"]

log = logging.getLogger(__name__)


@dataclass
class ScoreboardConfig:
    """Confirmation rules; every value documents its source and status."""

    confirm_reads: int = 2  # identical consecutive reads for a +1/+2/+3 change (D14); sourced
    jump_confirm_reads: int = 5  # reads for any other delta (corrections, missed reads); tuned
    valid_deltas: tuple[int, ...] = (1, 2, 3)  # D14; sourced
    immediate_sources: tuple[str, ...] = ("manual", "feed", "sim")  # trusted sources confirm at once (ARCHITECTURE); sourced
    min_ocr_conf: float = 0.5  # OCR reads below this confidence are ignored; tuned
    clock_tol_s: float = 0.6  # the clock may fall by at most elapsed + this between reads; tuned (MM:SS rounding)
    clock_rise_tol_s: float = 0.05  # a clock that rises by more than this is inconsistent (same period); tuned
    stream_timeout_s: float = 1.5  # "streaming" needs a read within this; tuned
    stream_window_s: float = 3.0  # ... and >= stream_min_reads reads within this window; tuned
    stream_min_reads: int = 3  # tuned (5 Hz OCR gives 15 reads in 3 s)
    zero_s: float = 0.01  # a countdown clock at or below this reads "expired" (displays show 0.0); tuned
    zero_confirm_reads: int = 2  # OCR reads at zero before a clock counts as expired (as D14 for scores); tuned
    zero_jump_tol_s: float = 3.0  # the last running value must be able to reach 0 by then (+ this); tuned


@dataclass(slots=True)
class ClockZeroWatch:
    """Confirms that a countdown clock (game or shot clock) ran out.

    The clock must first be seen running above ``zero_s`` (since the last ``reset``, e.g. a new
    period), then show zero on ``zero_confirm_reads`` consecutive reads (one for trusted sources),
    and the drop must be plausible: the last running value could have counted down to zero by the
    first zero read (+ ``zero_jump_tol_s``). A single OCR misread "0:00" or a period number that
    advances before the clock is reset therefore never ends a period or a shot clock.
    """

    config: ScoreboardConfig
    last_running: tuple[float, float] | None = None  # (t, value) of the last read above zero
    zero_reads: int = 0
    first_zero_t: float | None = None
    fired: bool = False

    def reset(self) -> None:
        self.last_running, self.zero_reads, self.first_zero_t, self.fired = None, 0, None, False

    def update(self, t: float, value: float | None, trusted: bool) -> float | None:
        """Feed one read; returns the time of the first zero read once expiry is confirmed."""
        cfg = self.config
        if value is None or value != value:  # missing or nan: no evidence either way
            return None
        if value > cfg.zero_s:
            self.last_running, self.zero_reads, self.first_zero_t, self.fired = (t, value), 0, None, False
            return None
        if self.last_running is None or self.fired:
            return None
        self.zero_reads += 1
        if self.first_zero_t is None:
            self.first_zero_t = t
        if self.zero_reads < (1 if trusted else cfg.zero_confirm_reads):
            return None
        last_t, last_v = self.last_running
        if last_v - (self.first_zero_t - last_t) > cfg.zero_jump_tol_s:
            return None  # jumped to zero from far away: a misread or an operator reset, not expiry
        self.fired = True
        return self.first_zero_t


@dataclass(frozen=True, slots=True)
class ScoreChange:
    """A confirmed change of one team's score."""

    t: float  # time the new score first appeared (first of the confirming reads)
    team: Team
    delta: int
    home: int
    away: int
    period: int | None
    game_clock: float | None
    source: str
    confirmed_t: float  # time of the read that confirmed it
    needs_review: bool = False
    reason: str = ""


@dataclass(frozen=True, slots=True)
class PeriodChange:
    """A confirmed period number change (``old`` is None for the first period seen)."""

    t: float
    old: int | None
    new: int
    game_clock: float | None
    source: str


ScoreboardUpdate = Union[ScoreChange, PeriodChange]


@dataclass(slots=True)
class _Candidate:
    value: tuple[int, int] | int
    reads: list[tuple[float, float | None]]  # (t, game_clock) of each identical read


class ScoreboardConfirmer:
    """Confirms score and period changes from raw scoreboard readings."""

    def __init__(self, config: ScoreboardConfig | None = None) -> None:
        self.config = config or ScoreboardConfig()
        self.score: dict[Team, int] | None = None  # confirmed score
        self.period: int | None = None  # confirmed period
        self.game_clock: float | None = None  # last observed game clock
        self.shot_clock: float | None = None  # last observed shot clock
        self.source: str | None = None  # source of the last accepted reading
        self._score_cand: _Candidate | None = None
        self._period_cand: _Candidate | None = None
        self._confirmed_clock: float | None = None  # clock at the last confirmed score
        self._read_times: deque[float] = deque(maxlen=64)
        self._clock_value: float | None = None
        self._clock_since_t: float | None = None  # first time the current clock value was read
        self._clock_reads: int = 0  # consecutive reads with the current clock value

    # ------------------------------------------------------------------ public API

    def update(self, reading: ScoreboardReading) -> list[ScoreboardUpdate]:
        """Feed one reading; returns the changes it confirms (period first, then scores)."""
        cfg = self.config
        immediate = reading.source in cfg.immediate_sources
        if not immediate and reading.confidence < cfg.min_ocr_conf:
            return []
        if reading.home_score is not None and reading.away_score is not None:
            self._read_times.append(reading.t)  # only score reads can show that no delta came
        self.source = reading.source
        if reading.shot_clock is not None:
            self.shot_clock = reading.shot_clock
        out: list[ScoreboardUpdate] = []
        period_change = self._update_period(reading, immediate)
        if period_change is not None:
            out.append(period_change)
            self._confirmed_clock = None  # a new period resets the clock-consistency anchor
        out.extend(self._update_score(reading, immediate))
        self._update_clock(reading)
        return out

    def is_streaming(self, t: float) -> bool:
        """True while score readings keep arriving (OCR / feed / sim), i.e. a make should see its
        delta. Reads without both scores (clock-only, unreadable digits) do not count."""
        cfg = self.config
        if not self._read_times or t - self._read_times[-1] > cfg.stream_timeout_s:
            return False
        recent = sum(1 for rt in self._read_times if t - rt <= cfg.stream_window_s)
        return recent >= cfg.stream_min_reads

    def clock_stopped(self, t: float, min_reads: int = 2, min_s: float = 0.0) -> bool | None:
        """Is the game clock unchanged for >= ``min_reads`` reads and >= ``min_s`` seconds?

        None when no clock has been read.
        """
        if self._clock_value is None or self._clock_since_t is None:
            return None
        return self._clock_reads >= min_reads and (t - self._clock_since_t) >= min_s

    @property
    def clock_stopped_since(self) -> float | None:
        """Time the current (unchanged) clock value was first read."""
        return self._clock_since_t

    # ------------------------------------------------------------------ internals

    def _update_period(self, r: ScoreboardReading, immediate: bool) -> PeriodChange | None:
        if r.period is None:
            return None
        if r.period == self.period:
            self._period_cand = None
            return None
        cand = self._period_cand
        if cand is None or cand.value != r.period:
            cand = self._period_cand = _Candidate(r.period, [])
        cand.reads.append((r.t, r.game_clock))
        if not immediate and len(cand.reads) < self.config.confirm_reads:
            return None
        old, self.period, self._period_cand = self.period, int(r.period), None
        log.info("period %s -> %s at t=%.2f (%s)", old, self.period, cand.reads[0][0], r.source)
        return PeriodChange(t=cand.reads[0][0], old=old, new=self.period, game_clock=r.game_clock, source=r.source)

    def _update_score(self, r: ScoreboardReading, immediate: bool) -> list[ScoreChange]:
        if r.home_score is None or r.away_score is None:
            return []
        value = (int(r.home_score), int(r.away_score))
        if self.score is not None and value == (self.score[Team.HOME], self.score[Team.AWAY]):
            self._score_cand = None
            return []
        cand = self._score_cand
        if cand is None or cand.value != value:
            cand = self._score_cand = _Candidate(value, [])
        cand.reads.append((r.t, r.game_clock))
        if self.score is None:  # baseline: no change is emitted
            if immediate or len(cand.reads) >= self.config.confirm_reads:
                self.score = {Team.HOME: value[0], Team.AWAY: value[1]}
                self._confirmed_clock = r.game_clock
                self._score_cand = None
            return []
        deltas = {Team.HOME: value[0] - self.score[Team.HOME], Team.AWAY: value[1] - self.score[Team.AWAY]}
        valid = all(d == 0 or d in self.config.valid_deltas for d in deltas.values())
        if not immediate:
            needed = self.config.confirm_reads if valid else self.config.jump_confirm_reads
            if len(cand.reads) < needed or not self._clock_consistent(cand.reads[-needed:]):
                return []
        changes = [
            ScoreChange(
                t=cand.reads[0][0], team=team, delta=d, home=value[0], away=value[1], period=self.period,
                game_clock=cand.reads[0][1], source=r.source, confirmed_t=r.t, needs_review=not valid,
                reason="" if valid else f"score jump {d:+d} (not +1/+2/+3)",
            )
            for team, d in deltas.items() if d != 0
        ]
        self.score = {Team.HOME: value[0], Team.AWAY: value[1]}
        self._confirmed_clock = cand.reads[-1][1]
        self._score_cand = None
        for c in changes:
            log.info("score %s %+d -> %d-%d at t=%.2f (%s)", c.team.value, c.delta, c.home, c.away, c.t, c.source)
        return changes

    def _clock_consistent(self, reads: list[tuple[float, float | None]]) -> bool:
        cfg = self.config
        clocks = [(t, c) for t, c in reads if c is not None]
        if self._confirmed_clock is not None and clocks and clocks[0][1] > self._confirmed_clock + cfg.clock_rise_tol_s:
            return False
        for (t0, c0), (t1, c1) in zip(clocks, clocks[1:]):
            if c1 > c0 + cfg.clock_rise_tol_s:
                return False
            if c0 - c1 > (t1 - t0) + cfg.clock_tol_s:
                return False
        return True

    def _update_clock(self, r: ScoreboardReading) -> None:
        if r.game_clock is None:
            return
        self.game_clock = r.game_clock
        if self._clock_value is not None and abs(r.game_clock - self._clock_value) < 1e-6:
            self._clock_reads += 1
            return
        self._clock_value = r.game_clock
        self._clock_since_t = r.t
        self._clock_reads = 1
