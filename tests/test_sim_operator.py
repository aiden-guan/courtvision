"""SimOperator lineup entries and fictional rosters."""

from __future__ import annotations

import numpy as np
import pytest

from courtvision.sim.names import LAST_NAMES, TEAM_NAMES, make_roster
from courtvision.sim.operator import SimOperator, SimOperatorConfig
from courtvision.sim.simulator import GameSimulator, SimConfig, SimResult
from courtvision.types import Team


@pytest.fixture(scope="module")
def sim() -> SimResult:
    return GameSimulator(SimConfig(seconds=400.0, fps=10.0, seed=13, period_s=90.0)).run()


class _Sink:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def set_lineup(self, team: Team, player_ids: list[str], t: float) -> None:
        self.calls.append(("lineup", team, tuple(player_ids), t))

    def substitute(self, team: Team, out_id: str, in_id: str, t: float) -> None:
        self.calls.append(("sub", team, out_id, in_id, t))


def test_entries_match_truth_substitutions(sim: SimResult) -> None:
    op = SimOperator(sim, SimOperatorConfig(entry_delay_s=2.0))
    entries = op.entries()
    lineups = [e for e in entries if "players" in e]
    subs = [e for e in entries if "out" in e]
    assert {e["team"]: e["players"] for e in lineups} == {t.value: v for t, v in sim.starters.items()}
    assert all(e["t"] == 0.0 for e in lineups)
    assert len(subs) == len(sim.substitutions) > 0
    for e, s in zip(subs, sim.substitutions, strict=True):
        assert (e["team"], e["out"], e["in"]) == (s["team"], s["out"], s["in"])
        assert e["t"] == pytest.approx(s["t"] + 2.0)
    assert [e["t"] for e in entries] == sorted(e["t"] for e in entries)


def test_entries_replay_to_the_true_lineup(sim: SimResult) -> None:
    on = {t.value: set(v) for t, v in sim.starters.items()}
    for e in SimOperator(sim).entries():
        if "out" in e:
            assert e["out"] in on[e["team"]] and e["in"] not in on[e["team"]]
            on[e["team"]] = (on[e["team"]] - {e["out"]}) | {e["in"]}
    last = sim.frames[-1].players
    assert on == {t.value: {p.player_id for p in last if p.team is t} for t in (Team.HOME, Team.AWAY)}


def test_due_and_apply_stream_entries(sim: SimResult) -> None:
    op = SimOperator(sim, SimOperatorConfig(entry_delay_s=1.5))
    sink = _Sink()
    total = 0
    for t in np.arange(0.0, sim.frames[-1].t + 5.0, 1.0):
        total += op.apply(sink, float(t))
    assert total == len(op.entries()) == len(sink.calls)
    assert sink.calls[0][0] == "lineup" and sink.calls[1][0] == "lineup"
    assert op.due(1e9) == []
    subs = [c for c in sink.calls if c[0] == "sub"]
    assert subs[0][4] == pytest.approx(sim.substitutions[0]["t"] + 1.5)


def test_rosters_are_fictional_and_unique() -> None:
    rng = np.random.default_rng(0)
    roster = make_roster(rng, size=12)
    assert len(roster.entries) == 24
    ids = [e.player_id for e in roster.entries]
    assert len(set(ids)) == 24
    names = [e.name for e in roster.entries]
    assert len(set(names)) == 24
    assert all(n.split(" ", 1)[1] in LAST_NAMES for n in names)
    assert set(roster.team_names.values()) <= set(TEAM_NAMES)
    for team in (Team.HOME, Team.AWAY):
        mine = [e for e in roster.entries if e.team is team]
        assert len({e.jersey for e in mine}) == 12
        assert all(e.player_id == f"{team.value}-{e.jersey}" for e in mine)
        assert [e.position for e in mine[:5]] == ["G", "G", "F", "F", "C"]
    again = make_roster(np.random.default_rng(0), size=12)
    assert [e.name for e in again.entries] == names
    with pytest.raises(ValueError):
        make_roster(rng, size=4)
