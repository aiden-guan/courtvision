"""Tests for courtvision.eval.boxscore with a local GameStats fake (stats/ is built in parallel)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import pytest

from courtvision.eval.boxscore import BoxEvalConfig, compare_box
from courtvision.types import Roster, RosterEntry, Team

STATS = ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf")


@dataclass
class FakeLine:
    """Mirrors the ARCHITECTURE PlayerLine counting fields."""

    player_id: str
    team: Team | None
    pts: int = 0
    fgm: int = 0
    fga: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    oreb: int = 0
    dreb: int = 0
    ast: int = 0
    stl: int = 0
    blk: int = 0
    tov: int = 0
    pf: int = 0
    provisional: bool = False

    @property
    def reb(self) -> int:
        return self.oreb + self.dreb


@dataclass
class FakeTeamLine:
    pts: int = 0


@dataclass
class FakeGameStats:
    players: dict[str, FakeLine]
    score: dict[Team, int]
    teams: dict[Team, FakeTeamLine] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        players = {}
        for pid, line in self.players.items():
            d = asdict(line)
            d["team"] = line.team.value if line.team else None
            d["reb"] = line.reb
            players[pid] = d
        return {
            "players": players,
            "score": {t.value: v for t, v in self.score.items()},
            "teams": {t.value: {"pts": tl.pts} for t, tl in self.teams.items()},
        }


def truth_box() -> dict[str, dict[str, int]]:
    def line(**kw: int) -> dict[str, int]:
        return {s: kw.get(s, 0) for s in STATS}

    return {
        "home-1": line(pts=7, fgm=3, fga=6, fg3m=1, fg3a=2, dreb=3, ast=2),
        "home-2": line(pts=4, fgm=2, fga=3, oreb=1, stl=1, tov=1),
        "away-5": line(pts=5, fgm=2, fga=5, fg3m=1, fg3a=3, dreb=4, blk=1, pf=2),
    }


def perfect_stats() -> FakeGameStats:
    players = {}
    for pid, line in truth_box().items():
        team = Team.HOME if pid.startswith("home") else Team.AWAY
        players[pid] = FakeLine(player_id=pid, team=team, **line)
    teams = {Team.HOME: FakeTeamLine(11), Team.AWAY: FakeTeamLine(5)}
    return FakeGameStats(players, {Team.HOME: 11, Team.AWAY: 5}, teams)


def test_perfect_box() -> None:
    rep = compare_box(perfect_stats(), truth_box(), {Team.HOME: 11, Team.AWAY: 5})
    assert rep.stats == list(STATS[:9]) + ["reb"] + list(STATS[9:])
    assert all(e.mae == 0.0 and e.exact_rate == 1.0 for e in rep.per_stat.values())
    assert rep.per_stat["reb"].truth_total == 8  # derived from oreb + dreb on both sides
    assert rep.line_exact_rate == 1.0 and rep.mean_mae == 0.0
    assert rep.score_match and rep.team_points_abs_error == 0 and rep.points_source == "score"
    assert rep.missing_players == [] and rep.unmatched_players == {} and rep.player_errors == {}
    assert rep.team_totals["home"]["pts"] == {"pred": 11.0, "truth": 11.0, "error": 0.0}
    assert rep.team_totals_mae == 0.0
    json.dumps(rep.to_dict())


def test_mismatches_mae_exact_rate_and_bias() -> None:
    stats = perfect_stats()
    stats.players["home-1"].pts = 9  # +2
    stats.players["home-1"].ast = 1  # -1
    stats.players["away-5"].pts = 2  # -3
    stats.score = {Team.HOME: 13, Team.AWAY: 2}
    rep = compare_box(stats, truth_box(), {Team.HOME: 11, Team.AWAY: 5})
    pts = rep.per_stat["pts"]
    assert pts.mae == pytest.approx(5 / 3)
    assert pts.bias == pytest.approx(-1 / 3)
    assert pts.exact_rate == pytest.approx(1 / 3)
    assert (pts.pred_total, pts.truth_total) == (15, 16)
    assert rep.per_stat["ast"].mae == pytest.approx(1 / 3)
    assert rep.player_errors == {"home-1": {"pts": 2, "ast": -1}, "away-5": {"pts": -3}}
    assert rep.line_exact_rate == pytest.approx(1 / 3)
    assert rep.team_points == {
        "home": {"pred": 13, "truth": 11, "error": 2},
        "away": {"pred": 2, "truth": 5, "error": -3},
    }
    assert rep.team_points_abs_error == 5 and not rep.score_match


def test_missing_and_provisional_players() -> None:
    stats = perfect_stats()
    lost = stats.players.pop("home-2")
    # The engine could not identify home-2: its stats went to a provisional track id.
    stats.players["home-t17"] = FakeLine(player_id="home-t17", team=Team.HOME, pts=lost.pts, fgm=lost.fgm,
                                         fga=lost.fga, oreb=lost.oreb, provisional=True)
    rep = compare_box(stats, truth_box(), {Team.HOME: 11, Team.AWAY: 5})
    assert rep.missing_players == ["home-2"] and rep.n_matched_players == 2
    assert rep.unmatched_players == {
        "home-t17": {"pts": 4.0, "fgm": 2.0, "fga": 3.0, "oreb": 1.0, "reb": 1.0, "team": "home", "provisional": True}
    }
    assert rep.unattributed["pts"] == 4.0
    assert rep.per_stat["pts"].mae == pytest.approx(4 / 3)
    # Team totals include the provisional line, so only stl/tov (never credited) differ.
    home = rep.team_totals["home"]
    assert home["pts"]["error"] == 0.0 and home["stl"]["error"] == -1.0 and home["tov"]["error"] == -1.0
    assert rep.score_match


def test_dict_form_matches_object_form() -> None:
    stats = perfect_stats()
    stats.players["away-5"].blk = 0
    stats.players["home-x"] = FakeLine(player_id="home-x", team=None, stl=1)
    a = compare_box(stats, truth_box(), {Team.HOME: 11, Team.AWAY: 5}).to_dict()
    b = compare_box(stats.to_dict(), truth_box(), {"home": 11, "away": 5}).to_dict()
    assert a == b
    assert a["per_stat"]["blk"]["mae"] == pytest.approx(1 / 3)


def test_points_fall_back_to_team_lines_then_players() -> None:
    stats = perfect_stats()
    stats.score = {}
    rep = compare_box(stats, truth_box(), {Team.HOME: 11, Team.AWAY: 5})
    assert rep.points_source == "teams" and rep.score_match
    stats.teams = {}
    rep = compare_box(stats, truth_box(), {Team.HOME: 12, Team.AWAY: 5})
    assert rep.points_source == "players"
    assert rep.team_points["home"] == {"pred": 11, "truth": 12, "error": -1} and not rep.score_match


def test_player_points_fallback_uses_roster_teams() -> None:
    # No score / teams in the prediction and lines without a team field: the roster
    # must still assign each player's points to a team.
    roster = Roster([RosterEntry("p1", Team.HOME, "1", "A"), RosterEntry("p2", Team.AWAY, "2", "B")])
    stats = {"players": {"p1": {"pts": 5}, "p2": {"pts": 3}}}
    rep = compare_box(stats, {"p1": {"pts": 5}, "p2": {"pts": 3}}, {"home": 5, "away": 3}, roster=roster)
    assert rep.points_source == "players"
    assert rep.team_points["home"]["pred"] == 5 and rep.team_points["away"]["pred"] == 3
    assert rep.score_match


def test_roster_resolves_teams_and_stat_subset() -> None:
    roster = Roster([RosterEntry("p1", Team.HOME, "1", "A"), RosterEntry("p2", Team.AWAY, "2", "B")])
    truth = {"p1": {"pts": 3, "ast": 1}, "p2": {"pts": 2}}
    stats = {"players": {"p1": {"pts": 3, "ast": 0, "team": "home"}, "p2": {"pts": 2, "team": "away"}},
             "score": {"home": 3, "away": 2}}
    rep = compare_box(stats, truth, {"home": 3, "away": 2}, roster=roster)
    assert rep.stats == ["pts", "ast"]  # only what truth reports
    assert rep.team_totals["home"]["ast"] == {"pred": 0.0, "truth": 1.0, "error": -1.0}
    assert rep.team_totals["away"]["pts"]["truth"] == 2.0
    only_pts = compare_box(stats, truth, {"home": 3, "away": 2}, config=BoxEvalConfig(stats=("pts",)))
    assert only_pts.stats == ["pts"] and only_pts.line_exact_rate == 1.0
