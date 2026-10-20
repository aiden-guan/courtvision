"""Replays a simulation's true lineups as operator entries (ARCHITECTURE stats/: lineups, minutes
and +/- come only from operator entries).

Entries use the ``POST /api/lineup`` shapes: a starting five ``{"t", "team", "players"}`` and a
tap-to-sub ``{"t", "team", "out", "in"}``. Substitutions arrive ``entry_delay_s`` after the true
entry frame (the operator's reaction time).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from courtvision.sim.simulator import SimResult
from courtvision.types import Team

__all__ = ["LineupSink", "SimOperator", "SimOperatorConfig"]

log = logging.getLogger(__name__)


@dataclass
class SimOperatorConfig:
    entry_delay_s: float = 2.0  # operator lag after the true substitution; design choice
    lineup_t: float = 0.0  # video time the starting fives are entered (before tip-off)


class LineupSink(Protocol):
    """The part of ``stats.engine.StatsEngine`` the operator drives."""

    def set_lineup(self, team: Team, player_ids: list[str], t: float) -> None: ...

    def substitute(self, team: Team, out_id: str, in_id: str, t: float) -> None: ...


class SimOperator:
    """Operator lineup entries derived from a :class:`SimResult`."""

    def __init__(self, result: SimResult, config: SimOperatorConfig | None = None) -> None:
        self.result = result
        self.config = config or SimOperatorConfig()
        self._entries = self._build()
        self._next = 0

    def _build(self) -> list[dict[str, Any]]:
        c = self.config
        out: list[dict[str, Any]] = []
        for team in (Team.HOME, Team.AWAY):
            players = list(self.result.starters.get(team, []))
            if players:
                out.append({"t": float(c.lineup_t), "team": team.value, "players": players})
        for sub in self.result.substitutions:
            t = float(sub["t"]) + c.entry_delay_s
            out.append({"t": t, "team": sub["team"], "out": sub["out"], "in": sub["in"]})
        out.sort(key=lambda e: (e["t"], "players" not in e))  # stable: true order kept for equal times
        return out

    def entries(self) -> list[dict[str, Any]]:
        """All entries sorted by time (copies)."""
        return [dict(e) for e in self._entries]

    def due(self, t: float) -> list[dict[str, Any]]:
        """Entries with time <= ``t`` not returned by an earlier call (for streaming replays)."""
        out = []
        while self._next < len(self._entries) and self._entries[self._next]["t"] <= t + 1e-9:
            out.append(dict(self._entries[self._next]))
            self._next += 1
        return out

    def apply(self, sink: LineupSink, t: float) -> int:
        """Send the entries due by ``t`` to ``sink`` (``set_lineup`` / ``substitute``); returns the count."""
        due = self.due(t)
        for e in due:
            team = Team(e["team"])
            if "players" in e:
                sink.set_lineup(team, list(e["players"]), e["t"])
            else:
                sink.substitute(team, e["out"], e["in"], e["t"])
        return len(due)
