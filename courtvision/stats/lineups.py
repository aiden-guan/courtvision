"""Operator lineups, stints, player minutes / plus-minus and the elapsed-game-time timeline.

Lineups, ``PlayerLine.seconds`` and ``plus_minus`` come ONLY from operator entries
(``set_lineup`` / ``substitute``; RESEARCH.md §7A.8). Without entries they stay empty / zero.

Entries are kept as a log and stints are rebuilt on every snapshot, so a late entry (operator
latency) or a revised basket is reflected everywhere. A stint is the half-open video-time interval
``[start_t, end_t)`` between two entries for one team; a scoring play at time ``t`` belongs to the
lineup on the floor at ``t``. Stint seconds are elapsed *game* seconds from
``GameClockTimeline`` (game clock when the scoreboard provides it, else live video time).
"""

from __future__ import annotations

import bisect
import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from courtvision.stats.boxscore import Possession, ScoringPlay
from courtvision.types import ScoreboardReading, Team

__all__ = [
    "GameClockTimeline",
    "LineupEntry",
    "LineupTracker",
    "LineupResult",
    "Stint",
    "compute_lineups",
    "lineup_id",
]

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- elapsed game time


class GameClockTimeline:
    """Cumulative elapsed game seconds as a function of video time ``t``.

    Per frame: with a game-clock reading, elapsed = full lengths of earlier periods + (period
    length - remaining). Between readings the value is interpolated; if no reading arrived for
    ``clock_stale_s`` (or none ever), live video dt is added instead (capped at ``max_dt_s`` per
    frame, and only while ``live``). The timeline never decreases (backward OCR glitches are
    ignored). The game clock cannot run faster than video time, so an untrusted (OCR) reading
    that jumps more than ``jump_tol_s`` ahead of it is held until the next reading agrees with it;
    readings from ``trusted_sources`` are applied at once.
    """

    def __init__(
        self,
        period_length_s: Callable[[int], float],
        *,
        clock_stale_s: float = 3.0,
        max_dt_s: float = 0.5,
        jump_tol_s: float = 2.0,
        trusted_sources: Iterable[str] = ("manual", "feed", "sim"),
    ) -> None:
        self._period_length_s = period_length_s
        self.clock_stale_s = clock_stale_s
        self.max_dt_s = max_dt_s
        self.jump_tol_s = jump_tol_s
        self.trusted_sources = frozenset(trusted_sources)
        self._ts: list[float] = []
        self._cum: list[float] = []
        self._last_frame_t: float | None = None
        self._last_clock_t: float | None = None
        self._jump: tuple[float, float, float] | None = None  # (t, elapsed, reading t) of a held jump
        self._bases: dict[int, float] = {}
        self.used_clock = False
        self.used_video = False

    @property
    def basis(self) -> str:
        """"game_clock", "video", "mixed" or "none" (no frames yet)."""
        if self.used_clock and self.used_video:
            return "mixed"
        if self.used_clock:
            return "game_clock"
        return "video" if self.used_video else "none"

    @property
    def total(self) -> float | None:
        return self._cum[-1] if self._cum else None

    def _period_base(self, period: int) -> float:
        if period not in self._bases:
            self._bases[period] = sum(float(self._period_length_s(p)) for p in range(1, period))
        return self._bases[period]

    def _append(self, t: float, cum: float) -> None:
        cum = max(cum, self._cum[-1]) if self._cum else cum
        if self._ts and t <= self._ts[-1]:
            self._cum[-1] = max(self._cum[-1], cum)
            return
        self._ts.append(t)
        self._cum.append(cum)

    def update(self, t: float, reading: ScoreboardReading | None = None, live: bool = True) -> bool:
        """Fold in one frame at video time ``t`` with its scoreboard reading (if any).

        Returns True when the reading's game clock was applied (False: no clock, or held / stale).
        """
        if self._last_frame_t is not None and t < self._last_frame_t:
            return False
        applied = self._apply_clock(t, reading)
        if not applied and (self._last_clock_t is None or t - self._last_clock_t > self.clock_stale_s):
            prev = self._last_frame_t
            dt = 0.0 if prev is None else min(max(t - prev, 0.0), self.max_dt_s)
            base = self._cum[-1] if self._cum else 0.0
            self._append(t, base + (dt if live else 0.0))
            self.used_video = True
        self._last_frame_t = t
        return applied

    def _apply_clock(self, t: float, reading: ScoreboardReading | None) -> bool:
        """Append the reading's elapsed time if it is usable; False when absent or held."""
        clock = _period_clock(reading)
        if clock is None or reading is None:
            return False
        period, remaining = clock
        length = float(self._period_length_s(period))
        cum = self._period_base(period) + min(max(length - remaining, 0.0), length)
        if reading.source not in self.trusted_sources and not self._plausible(t, cum, reading.t):
            return False
        self._jump = None
        self._append(t, cum)
        self._last_clock_t = t
        self.used_clock = True
        return True

    def _plausible(self, t: float, cum: float, read_t: float) -> bool:
        """Not ahead of video time, or confirmed by a second distinct read consistent with it."""
        if not self._cum or cum <= self._cum[-1] + max(t - self._ts[-1], 0.0) + self.jump_tol_s:
            return True
        if self._jump is not None:
            jt, jcum, jread_t = self._jump
            if read_t != jread_t and jcum - self.jump_tol_s <= cum <= jcum + max(t - jt, 0.0) + self.jump_tol_s:
                return True
        log.debug("clock reading at t=%.2f jumps elapsed to %.1f s; held for confirmation", t, cum)
        self._jump = (t, cum, read_t)
        return False

    def elapsed_at(self, t: float) -> float | None:
        """Elapsed game seconds at video time ``t`` (linear between samples, clamped at the ends)."""
        if not self._ts:
            return None
        i = bisect.bisect_right(self._ts, t)
        if i == 0:
            return self._cum[0]
        if i >= len(self._ts):
            return self._cum[-1]
        t0, t1 = self._ts[i - 1], self._ts[i]
        c0, c1 = self._cum[i - 1], self._cum[i]
        return c0 + (c1 - c0) * (t - t0) / (t1 - t0)


def _period_clock(reading: ScoreboardReading | None) -> tuple[int, float] | None:
    """(period, remaining seconds) when the reading carries both."""
    if reading is None or reading.game_clock is None or not reading.period or reading.period < 1:
        return None
    if not math.isfinite(float(reading.game_clock)):
        return None  # a nan clock would poison every later elapsed value
    return int(reading.period), float(reading.game_clock)


# --------------------------------------------------------------------------- operator entries


@dataclass(frozen=True)
class LineupEntry:
    t: float
    seq: int
    team: Team
    kind: str  # "set" | "sub"
    players: tuple[str, ...] = ()  # "set": the five
    out_id: str | None = None  # "sub"
    in_id: str | None = None  # "sub"


@dataclass(frozen=True)
class Stint:
    team: Team
    players: tuple[str, ...]
    start_t: float
    end_t: float | None  # None = still on the floor


def lineup_id(team: Team, players: Iterable[str]) -> str:
    return f"{team.value}:" + "+".join(sorted(players))


class LineupTracker:
    """Log of operator lineup entries; builds stints on demand."""

    def __init__(self) -> None:
        self._entries: list[LineupEntry] = []

    def set_lineup(self, team: Team, player_ids: list[str], t: float) -> None:
        players = tuple(dict.fromkeys(player_ids))
        if len(players) != len(player_ids):
            log.warning("set_lineup(%s): duplicate player ids dropped", team.value)
        self._entries.append(LineupEntry(t=float(t), seq=len(self._entries), team=team, kind="set", players=players))

    def substitute(self, team: Team, out_id: str, in_id: str, t: float) -> None:
        floor = self.on_court(team, t)
        if out_id not in floor:
            log.warning("substitute(%s): %s is not on the floor at t=%.2f", team.value, out_id, t)
        if in_id in floor:
            log.warning("substitute(%s): %s is already on the floor at t=%.2f", team.value, in_id, t)
        self._entries.append(
            LineupEntry(t=float(t), seq=len(self._entries), team=team, kind="sub", out_id=out_id, in_id=in_id)
        )

    @property
    def entries(self) -> list[LineupEntry]:
        return sorted(self._entries, key=lambda e: (e.t, e.seq))

    def has_entries(self, team: Team | None = None) -> bool:
        return any(team is None or e.team is team for e in self._entries)

    def on_court(self, team: Team, t: float | None = None) -> list[str]:
        """The team's lineup after every entry at or before ``t`` (all entries when None)."""
        current: list[str] = []
        for e in self.entries:
            if e.team is team and (t is None or e.t <= t):
                current = _apply(current, e)
        return current

    def stints(self) -> list[Stint]:
        """Stints of both teams (zero-length ones dropped); the last stint per team is open."""
        out: list[Stint] = []
        for team in Team:
            current: list[str] = []
            start: float | None = None
            for e in (x for x in self.entries if x.team is team):
                if start is not None and current and e.t > start:
                    out.append(Stint(team, tuple(current), start, e.t))
                current = _apply(current, e)
                start = e.t
            if start is not None and current:
                out.append(Stint(team, tuple(current), start, None))
        return out


def _apply(current: list[str], e: LineupEntry) -> list[str]:
    if e.kind == "set":
        return list(e.players)
    nxt = list(current)  # invalid entries were already warned about when entered
    if e.out_id in nxt:
        nxt.remove(e.out_id)
    if e.in_id is not None and e.in_id not in nxt:
        nxt.append(e.in_id)
    return nxt


# --------------------------------------------------------------------------- aggregation


@dataclass
class LineupResult:
    lineups: dict[str, dict[str, Any]] = field(default_factory=dict)
    player_seconds: dict[str, float] = field(default_factory=dict)
    player_plus_minus: dict[str, int] = field(default_factory=dict)


def _in_stint(t: float, stint: Stint) -> bool:
    return stint.start_t <= t and (stint.end_t is None or t < stint.end_t)


def compute_lineups(
    stints: list[Stint],
    elapsed_at: Callable[[float], float],
    now_t: float,
    scoring: list[ScoringPlay],
    possessions: list[Possession],
    min_possessions: int = 20,
) -> LineupResult:
    """Lineup records plus per-player seconds and plus-minus from operator stints.

    Each record: team, players, seconds, plus_minus, pts_for, pts_against, off_possessions,
    def_possessions, possessions, stints, ``current`` (on the floor now), off_rtg / def_rtg /
    net_rtg (per 100) and ``small_sample`` (possessions < ``min_possessions``) so consumers can
    hide noisy lineups. ``possessions`` is the per-side count used for lineup samples and pace
    (B-Ref style mean of offensive and defensive possessions [31]), not offense + defense: a five
    with 10 trips each way has played 10 possessions, which the ~20-possession small-sample gate
    (narration.md §7.2) must not read as 20.
    """
    res = LineupResult()
    for st in stints:
        end = now_t if st.end_t is None else st.end_t
        seconds = max(0.0, elapsed_at(end) - elapsed_at(st.start_t)) if end > st.start_t else 0.0
        pts_for = sum(s.points for s in scoring if s.team is st.team and _in_stint(s.t, st))
        pts_against = sum(s.points for s in scoring if s.team is not st.team and _in_stint(s.t, st))
        off = sum(1 for p in possessions if p.team is st.team and _in_stint(p.start_t, st))
        dfn = sum(1 for p in possessions if p.team is not st.team and _in_stint(p.start_t, st))
        rec = res.lineups.setdefault(
            lineup_id(st.team, st.players),
            {"team": st.team, "players": sorted(st.players), "seconds": 0.0, "plus_minus": 0, "pts_for": 0,
             "pts_against": 0, "off_possessions": 0, "def_possessions": 0, "possessions": 0, "stints": 0,
             "current": False},
        )
        rec["seconds"] += seconds
        rec["pts_for"] += pts_for
        rec["pts_against"] += pts_against
        rec["plus_minus"] += pts_for - pts_against
        rec["off_possessions"] += off
        rec["def_possessions"] += dfn
        rec["stints"] += 1
        rec["current"] = rec["current"] or st.end_t is None
        for pid in st.players:
            res.player_seconds[pid] = res.player_seconds.get(pid, 0.0) + seconds
            res.player_plus_minus[pid] = res.player_plus_minus.get(pid, 0) + pts_for - pts_against
    for rec in res.lineups.values():
        off, dfn = rec["off_possessions"], rec["def_possessions"]
        rec["off_rtg"] = 100.0 * rec["pts_for"] / off if off else None
        rec["def_rtg"] = 100.0 * rec["pts_against"] / dfn if dfn else None
        rec["net_rtg"] = rec["off_rtg"] - rec["def_rtg"] if off and dfn else None
        rec["possessions"] = (off + dfn) / 2.0
        rec["small_sample"] = rec["possessions"] < min_possessions
    return res
