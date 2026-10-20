"""Scoreboard reconciliation of shot results (RESEARCH.md §4.2, §7A.5; ARCHITECTURE events/).

The scoreboard is authoritative. For each confirmed score change (team T, delta d) the
``Reconciler`` decides, in order:

1. **confirm** an unconfirmed SHOT_MADE of T worth d near the change;
2. **fix_value** of an unconfirmed SHOT_MADE of T when d and its value are 2 and 3;
3. (hold while T has a shot in flight, so a make the engine is still deciding can match);
4. **flip_to_made** a recent SHOT_MISSED of T worth d (or 2 <-> 3);
5. **correction** (SCORE_CORRECTION, needs_review) once ``reconcile_hold_s`` has passed.

Separately, a SHOT_MADE with no delta within ``make_confirm_wait_s`` while a trusted scoreboard is
streaming becomes **make_to_miss** (needs_review). The engine applies the actions (revisions,
rebound retraction, assist re-derivation) because it owns the event store.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from courtvision.types import Team

from .scoreboard import ScoreChange

if TYPE_CHECKING:
    from .engine import EventEngineConfig

__all__ = ["Action", "Reconciler", "ShotEntry"]

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ShotEntry:
    """Engine ledger row for one shot (field goal or free throw)."""

    shot_id: str
    team: Team | None
    shooter_id: str | None
    value: int  # potential points: 1 / 2 / 3
    attempt_id: str
    t_release: float
    is_ft: bool = False
    result_id: str | None = None
    made: bool | None = None  # None while unresolved
    result_t: float | None = None
    confirmed: bool = False  # result corroborated (scoreboard or operator): never revised to a miss / flipped
    score_matched: bool = False  # a scoreboard delta was consumed by this shot (it cannot match another)
    rebound_id: str | None = None
    possession_id: str | None = None  # POSSESSION caused by the rebound that followed a miss
    assist_id: str | None = None
    block_id: str | None = None
    assist_candidate: tuple[str, Team | None] | None = None  # (passer, team) qualifying at release
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Action:
    kind: Literal["confirm", "fix_value", "flip_to_made", "correction", "make_to_miss"]
    shot_id: str | None
    change: ScoreChange | None
    new_value: int | None = None


class Reconciler:
    """Matches confirmed scoreboard deltas to the engine's shot results."""

    def __init__(self, cfg: EventEngineConfig) -> None:
        self.cfg = cfg
        self.pending: list[ScoreChange] = []

    def add(self, change: ScoreChange) -> None:
        self.pending.append(change)

    def step(self, now: float, shots: Sequence[ShotEntry], streaming: bool, inflight: Iterable[Team | None] = (),
             force: bool = False) -> list[Action]:
        """Decide what to do with pending changes and stale makes at time ``now``.

        ``inflight``: teams with an unresolved shot. ``force``: end of stream, no more holding.
        """
        actions: list[Action] = []
        busy = set(inflight)
        used: set[str] = set()
        for ch in list(self.pending):
            act = self._match(ch, shots, now, busy, used, force)
            if act is not None:
                actions.append(act)
                self.pending.remove(ch)
                if act.shot_id is not None:
                    used.add(act.shot_id)
        if streaming:
            waiting = {c.team for c in self.pending}
            for s in shots:
                if s.made and not s.confirmed and not s.score_matched and s.result_t is not None and s.shot_id not in used \
                        and s.team not in waiting and now - s.result_t > self.cfg.make_confirm_wait_s:
                    actions.append(Action("make_to_miss", s.shot_id, None))
        return actions

    def _match(self, ch: ScoreChange, shots: Sequence[ShotEntry], now: float, busy: set[Team | None],
               used: set[str], force: bool) -> Action | None:
        cfg = self.cfg
        d = ch.delta
        if ch.needs_review or d not in (1, 2, 3):
            return Action("correction", None, ch)
        lo, hi = ch.t - cfg.reconcile_match_back_s, ch.t + cfg.reconcile_match_fwd_s
        mine = [s for s in shots if s.team is ch.team and s.shot_id not in used and s.result_t is not None]
        # an operator-confirmed make still absorbs its delta (only a consumed delta excludes a shot)
        makes = sorted((s for s in mine if s.made and not s.score_matched and lo <= s.result_t <= hi),  # type: ignore[operator]
                       key=lambda s: abs(s.result_t - ch.t))  # type: ignore[operator]
        exact = [s for s in makes if s.value == d]
        if exact:
            return Action("confirm", exact[0].shot_id, ch)
        swap = [s for s in makes if {s.value, d} == {2, 3}]
        if swap:
            return Action("fix_value", swap[0].shot_id, ch, new_value=d)
        age = now - ch.confirmed_t
        if not force and ch.team in busy and age < cfg.reconcile_inflight_max_s:
            return None
        flip_lo = ch.t - cfg.reconcile_flip_window_s
        misses = sorted((s for s in mine if s.made is False and not s.confirmed and not s.score_matched  # type: ignore[operator]
                         and flip_lo <= s.result_t <= hi  # type: ignore[operator]
                         and (s.value == d or {s.value, d} == {2, 3})),
                        key=lambda s: (s.value != d, -s.result_t))  # type: ignore[operator]
        if misses:
            return Action("flip_to_made", misses[0].shot_id, ch, new_value=d)
        if force or age >= cfg.reconcile_hold_s:
            return Action("correction", None, ch)
        return None
