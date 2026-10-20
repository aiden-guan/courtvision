"""InsightEngine: each coach-card rule triggers and stays quiet, cooldown / dedupe, <= 12 words,
NBA-only xeFG gate, team fouls / bonus with RuleSpec. GameStats is faked from the ARCHITECTURE
contract so these tests do not depend on stats.engine."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pytest

from courtvision.geometry.court import court_for
from courtvision.geometry.rules import rules_for
from courtvision.stats.insights import InsightConfig, InsightEngine, penalty_state, word_count
from courtvision.types import Event, EventKind, EventStatus, FrameState, PlayerState, Roster, RosterEntry, Team

H, A = Team.HOME, Team.AWAY


# --------------------------------------------------------------------------- contract fakes


@dataclass
class FakePlayerLine:
    player_id: str
    team: Team | None
    pf: int = 0
    fga: int = 0


@dataclass
class FakeTeamLine:
    fga: int = 0
    fta: int = 0
    oreb: int = 0
    dreb: int = 0
    second_chance_pts: int = 0


@dataclass
class FakeShot:
    shot_id: str
    player_id: str
    team: Team
    t: float
    value: int
    made: bool | None
    zone: str | None = None
    xfg: float | None = None
    shot_type: str = "jumper"


@dataclass
class FakeStats:
    t: float = 0.0
    period: int | None = 1
    game_clock: float | None = 600.0
    players: dict[str, FakePlayerLine] = field(default_factory=dict)
    teams: dict[Team, FakeTeamLine] = field(default_factory=lambda: {H: FakeTeamLine(), A: FakeTeamLine()})
    shots: list[FakeShot] = field(default_factory=list)
    run: dict[str, Any] = field(default_factory=lambda: {"team": None, "points": 0, "opp_points": 0, "since_t": 0.0})
    lineups: dict[str, dict[str, Any]] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    tracking: dict[str, Any] = field(default_factory=dict)
    score: dict[Team, int] = field(default_factory=lambda: {H: 0, A: 0})


ROSTER = Roster(
    entries=[RosterEntry("h1", H, "23", "Marcus Hill"), RosterEntry("h2", H, "5", "Dee Park"),
             RosterEntry("a1", A, "11", "Sam Ortiz"), RosterEntry("a2", A, "0", "Jo Reyes")],
    team_names={H: "Riverton", A: "Summit"},
)
NAMES = ("Marcus", "Hill", "Dee", "Park", "Sam", "Ortiz", "Jo", "Reyes", "Riverton", "Summit")


def engine(league: str = "nba", rules=None, **cfg) -> InsightEngine:
    return InsightEngine(ROSTER, court_for(league), InsightConfig(**cfg), rules=rules)


def kinds(insights) -> list[str]:
    return [i.kind for i in insights]


def shots(team: Team, pid: str, made: list[bool], *, value: int = 2, zone: str | None = "paint",
          xfg: float | None = 0.4, t0: float = 0.0, prefix: str = "s") -> list[FakeShot]:
    return [FakeShot(f"{prefix}{team.value}{i}", pid, team, t0 + i, value, m, zone, xfg) for i, m in enumerate(made)]


def ev(eid: str, kind: EventKind, t: float, team: Team | None = None, pid: str | None = None, **kw) -> Event:
    return Event(eid, kind, t, int(t * 30), team=team, player_id=pid, **kw)


# --------------------------------------------------------------------------- foul trouble


def test_foul_trouble_q_plus_one_nba() -> None:
    e = engine()
    quiet = FakeStats(period=2, players={"h1": FakePlayerLine("h1", H, pf=2)})
    assert e.update(quiet, [], 1.0) == []
    out = e.update(FakeStats(period=2, players={"h1": FakePlayerLine("h1", H, pf=3)}), [], 2.0)
    assert [i.text for i in out] == ["{player} 3 fouls in Q2 (Q+1)"]
    assert out[0].player_id == "h1" and out[0].team is H and out[0].evidence["n"] == 3
    q4 = e.update(FakeStats(period=4, players={"h1": FakePlayerLine("h1", H, pf=5)}), [], 3.0)
    assert q4[0].text == "{player} 5 fouls in Q4, one from fouling out"
    assert q4[0].priority > out[0].priority


def test_foul_trouble_adapts_to_rulespec_foul_out() -> None:
    e = engine("fiba", rules=rules_for("fiba"))  # foul-out 5: Q4 threshold min(5, 4) = 4
    assert e.update(FakeStats(period=4, players={"a1": FakePlayerLine("a1", A, pf=3)}), [], 1.0) == []
    out = e.update(FakeStats(period=4, players={"a1": FakePlayerLine("a1", A, pf=4)}), [], 2.0)
    assert out[0].text == "{player} 4 fouls in Q4, one from fouling out"
    assert out[0].evidence["foul_out"] == 5 and out[0].evidence["threshold"] == 4


def test_foul_trouble_halves_use_quarter_equivalents() -> None:
    e = engine("ncaa_m", rules=rules_for("ncaa_m"))  # 2 x 20 min, foul-out 5
    early = e.update(FakeStats(period=1, game_clock=1100.0, players={"h1": FakePlayerLine("h1", H, pf=2)}), [], 1.0)
    assert early[0].text == "{player} 2 fouls in H1 (adapted Q+1)" and early[0].evidence["quarter_equivalent"] == 1
    late_h2 = engine("ncaa_m", rules=rules_for("ncaa_m"))
    st = FakeStats(period=2, game_clock=300.0, players={"h1": FakePlayerLine("h1", H, pf=3)})
    assert late_h2.update(st, [], 1.0) == []  # Q-equivalent 4: threshold 4
    st.players["h1"].pf = 4
    assert "one from fouling out" in late_h2.update(st, [], 2.0)[0].text


# --------------------------------------------------------------------------- hot hand


@pytest.mark.parametrize("made, fires", [([False, True, True, True], True), ([True, True, True, False], False),
                                         ([False, True, False, True], False), ([True, True, True], False)])
def test_hot_hand(made: list[bool], fires: bool) -> None:
    out = engine().update(FakeStats(shots=shots(H, "h1", made)), [], 1.0)
    hot = [i for i in out if i.kind == "hot_hand"]
    assert bool(hot) is fires
    if fires:
        assert hot[0].text == "{player} made 3 of last 4 shots" and hot[0].evidence["n"] == 4


# --------------------------------------------------------------------------- runs


@pytest.mark.parametrize("pts, opp, fires", [(8, 0, True), (7, 0, False), (10, 2, True), (10, 3, False), (9, 1, False)])
def test_scoring_run_thresholds_and_wording(pts: int, opp: int, fires: bool) -> None:
    out = engine().update(FakeStats(run={"team": A, "points": pts, "opp_points": opp, "since_t": 3.0}), [], 50.0)
    runs = [i for i in out if i.kind == "run"]
    assert bool(runs) is fires
    if fires:
        article = "an" if pts == 8 else "a"
        assert runs[0].text == f"{{team}} on {article} {pts}-{opp} run" and runs[0].team is A
        assert "timeout" not in runs[0].text.lower() and runs[0].evidence["duration_s"] == pytest.approx(47.0)


@pytest.mark.parametrize("window, leader", [({H: 10, A: 2}, H), ({"home": 10, "away": 2}, "home")])
def test_ten_two_run_from_stats_core_window(window: dict, leader: Any) -> None:
    # stats-core: "points"/"opp_points" is the unanswered run; 10-2 runs show up in the window keys
    run = {"team": A, "points": 2, "opp_points": 0, "since_t": 40.0, "window_s": 300.0, "window": window,
           "window_leader": leader, "window_diff": 8}
    out = [i for i in engine().update(FakeStats(run=run), [], 50.0) if i.kind == "run"]
    assert [(i.text, i.team) for i in out] == [("{team} on a 10-2 run", H)]
    assert out[0].evidence["source"] == "window" and out[0].evidence["duration_s"] == 300.0
    quiet = dict(run, window={H: 10, A: 4})
    assert [i for i in engine().update(FakeStats(run=quiet), [], 50.0) if i.kind == "run"] == []


def test_a_new_run_gets_a_new_card() -> None:
    e = engine()
    run = lambda p, since: FakeStats(run={"team": H, "points": p, "opp_points": 0, "since_t": since})  # noqa: E731
    assert kinds(e.update(run(8, 1.0), [], 10.0)) == ["run"]
    e.update(FakeStats(), [], 20.0)  # run broken
    assert kinds(e.update(run(8, 30.0), [], 40.0)) == ["run"]


def test_run_updates_only_every_four_points() -> None:
    e = engine()
    run = lambda p: FakeStats(run={"team": H, "points": p, "opp_points": 0, "since_t": 1.0})  # noqa: E731
    assert kinds(e.update(run(8), [], 10.0)) == ["run"]
    assert e.update(run(9), [], 11.0) == [] and e.update(run(11), [], 12.0) == []
    assert kinds(e.update(run(12), [], 13.0)) == ["run"]


# --------------------------------------------------------------------------- shot quality (NBA-only xeFG)


def _sq_stats(n_home: int = 16) -> FakeStats:
    home = shots(H, "h1", [True] * 12 + [False] * (n_home - 12), xfg=0.4)
    away = shots(A, "a1", [True] * 5 + [False] * 11, xfg=0.4)
    return FakeStats(shots=home + away)


def test_xefg_card_nba() -> None:
    out = [i for i in engine().update(_sq_stats(), [], 1.0) if i.kind == "shot_quality"]
    assert [i.text for i in out] == ["{team} eFG 75% on expected 40% (n=16)"]
    assert out[0].team is H and out[0].evidence["xefg"] == pytest.approx(0.4)
    assert not [i for i in engine().update(_sq_stats(n_home=14), [], 1.0) if i.kind == "shot_quality" and i.team is H]


def test_xefg_needs_fifteen_shots_and_a_gap() -> None:
    few = FakeStats(shots=shots(H, "h1", [True] * 12 + [False] * 2, xfg=0.4))
    assert [i for i in engine().update(few, [], 1.0) if i.kind == "shot_quality"] == []
    close = FakeStats(shots=shots(H, "h1", [True] * 7 + [False] * 9, xfg=0.45))
    assert [i for i in engine().update(close, [], 1.0) if i.kind == "shot_quality"] == []


@pytest.mark.parametrize("league", ["fiba", "ncaa_m", "ncaa_w", "nfhs"])
def test_non_nba_shows_raw_efg_with_n_never_expected(league: str) -> None:
    out = engine(league).update(_sq_stats(), [], 1.0)
    sq = [i for i in out if i.kind == "shot_quality"]
    assert [i.text for i in sq] == ["{team} eFG 75% vs opponent 31% (n=16)"]
    assert all("expected" not in i.text and "xeFG" not in i.text for i in out)
    assert "xefg" not in sq[0].evidence and sq[0].evidence["opp_n"] == 16


# --------------------------------------------------------------------------- shot diet, OREB, FT rate


def test_shot_diet_midrange_share() -> None:
    mid = shots(H, "h1", [True, False, False, True, False, False], zone="midrange", prefix="m")
    rest = shots(H, "h2", [True] * 4 + [False] * 6, zone="restricted_area", prefix="r", t0=10)
    out = [i for i in engine().update(FakeStats(shots=mid + rest), [], 1.0) if i.kind == "shot_diet"]
    assert [i.text for i in out] == ["{team} 6 of 16 shots from midrange, 2 made"]
    assert out[0].evidence["by_zone"]["midrange"] == {"fga": 6, "fgm": 2}
    light = shots(H, "h1", [False] * 3, zone="midrange", prefix="m") + rest + shots(H, "h2", [True] * 3, zone="paint", prefix="p", t0=30)
    assert [i for i in engine().update(FakeStats(shots=light), [], 1.0) if i.kind == "shot_diet"] == []


@pytest.mark.parametrize("oreb, opp_dreb, fires", [(7, 8, True), (3, 12, False), (4, 4, False)])
def test_offensive_rebounding_battle(oreb: int, opp_dreb: int, fires: bool) -> None:
    st = FakeStats(teams={H: FakeTeamLine(oreb=oreb), A: FakeTeamLine(dreb=opp_dreb)})
    out = [i for i in engine().update(st, [], 1.0) if i.kind == "oreb"]
    assert bool(out) is fires
    if fires:
        assert out[0].text == "{team} 7 of 15 offensive rebound chances (47%)" and out[0].evidence["n"] == 15


@pytest.mark.parametrize("fta, fga, fires", [(9, 20, True), (4, 20, False), (9, 12, False)])
def test_ft_rate(fta: int, fga: int, fires: bool) -> None:
    st = FakeStats(teams={H: FakeTeamLine(), A: FakeTeamLine(fga=fga, fta=fta)})
    out = [i for i in engine().update(st, [], 1.0) if i.kind == "ft_rate"]
    assert bool(out) is fires
    if fires:
        assert out[0].text == "{team} FT rate 45%: 9 FTA on 20 FGA" and out[0].team is A


# --------------------------------------------------------------------------- turnovers


def _possessions_and_turnovers(n_to: int) -> list[Event]:
    evs = []
    for k in range(6):
        evs.append(ev(f"p{k}", EventKind.POSSESSION, 10.0 * k, team=H, pid="h1"))
    for k in range(n_to):
        evs.append(ev(f"t{k}", EventKind.TURNOVER, 10.0 * (5 - k) + 5.0, team=H, pid="h1", attrs={"type": "bad_pass"}))
    return evs


def test_turnover_burst_in_last_five_possessions() -> None:
    out = [i for i in engine().update(FakeStats(), _possessions_and_turnovers(3), 60.0) if i.kind == "turnovers"]
    assert [i.text for i in out] == ["{team} 3 turnovers in last 5 possessions"]
    assert out[0].evidence["types"] == {"bad_pass": 3} and out[0].evidence["n"] == 5
    assert [i for i in engine().update(FakeStats(), _possessions_and_turnovers(2), 60.0) if i.kind == "turnovers"] == []


def test_turnover_retraction_clears_the_card() -> None:
    e = engine(default_cooldown_s=0.0)
    evs = _possessions_and_turnovers(3)
    assert kinds(e.update(FakeStats(), evs, 60.0)) == ["turnovers"]
    last = evs[-1]
    retracted = Event(last.event_id, last.kind, last.t, last.frame_idx, team=H, player_id="h1", revision=1,
                      status=EventStatus.RETRACTED)
    assert e.update(FakeStats(), [retracted], 61.0) == [] and e.active() == []
    stale = Event(last.event_id, last.kind, last.t, last.frame_idx, team=H, player_id="h1", revision=0)
    assert e.update(FakeStats(), [stale], 62.0) == [] and e.active() == []


def test_turnover_time_window_fallback_without_possessions() -> None:
    evs = [ev(f"t{k}", EventKind.TURNOVER, 100.0 + 20 * k, team=A, pid="a1") for k in range(3)]
    out = [i for i in engine().update(FakeStats(), evs, 150.0) if i.kind == "turnovers"]
    assert [i.text for i in out] == ["{team} 3 turnovers in last 3 minutes"]
    assert [i for i in engine().update(FakeStats(), evs, 400.0) if i.kind == "turnovers"] == []


# --------------------------------------------------------------------------- lineups


@pytest.mark.parametrize("lineup, fires", [
    ({"team": "home", "players": ["h1", "h2"], "plus_minus": 8, "possessions": 22}, True),
    ({"team": "home", "players": ["h1", "h2"], "plus_minus": 8, "possessions": 15}, False),
    ({"team": "home", "players": ["h1", "h2"], "plus_minus": 8}, False),
    ({"team": "home", "players": ["h1", "h2"], "plus_minus": 2, "possessions": 30}, False),
])
def test_lineup_plus_minus_needs_twenty_possessions(lineup: dict, fires: bool) -> None:
    out = [i for i in engine().update(FakeStats(lineups={"L1": lineup}), [], 1.0) if i.kind == "lineup"]
    assert bool(out) is fires
    if fires:
        assert out[0].text == "{team} lineup +8 over 22 possessions" and out[0].evidence["n"] == 22


# --------------------------------------------------------------------------- fatigue


def _feed_speeds(e: InsightEngine, early_top: float, late_top: float, minutes: int = 9) -> None:
    for k in range(int(minutes * 60 * 2) + 4):
        t = k * 0.5
        top = early_top if t < 360.0 else late_top
        speed = top * (0.3 + 0.7 * abs(math.sin(k * 0.37)))
        e.observe_frame(FrameState(k, t, [PlayerState("h1", H, 30.0, 25.0, vx=speed)], None))


def test_fatigue_top_speed_decline() -> None:
    e = engine()
    _feed_speeds(e, early_top=20.0, late_top=15.0)
    out = [i for i in e.update(FakeStats(), [], 545.0) if i.kind == "fatigue"]
    assert len(out) == 1 and out[0].text.startswith("{player} top speed down ") and out[0].player_id == "h1"
    assert out[0].evidence["drop"] == pytest.approx(0.25, abs=0.03) and out[0].evidence["n"] == 9


def test_fatigue_quiet_when_speed_holds_or_without_frames() -> None:
    e = engine()
    _feed_speeds(e, early_top=20.0, late_top=19.5)
    assert [i for i in e.update(FakeStats(), [], 545.0) if i.kind == "fatigue"] == []
    assert [i for i in engine().update(FakeStats(), [], 545.0) if i.kind == "fatigue"] == []


# --------------------------------------------------------------------------- decision quality


@pytest.mark.parametrize("labels, fires", [(["good", "poor", "neutral", "poor", "poor"], True),
                                           (["good", "poor", "neutral", "poor", "good"], False)])
def test_decision_quality(labels: list[str], fires: bool) -> None:
    decs = [{"t": float(i), "player_id": "a2", "label": lab, "delta": -0.3 if lab == "poor" else 0.0,
             "event_id": f"e{i}", "best": {"kind": "pass"}} for i, lab in enumerate(labels)]
    out = [i for i in engine().update(FakeStats(decisions=decs), [], 9.0) if i.kind == "decision"]
    assert bool(out) is fires
    if fires:
        assert out[0].text == "{player} 3 low-value choices in last 5 graded"
        assert out[0].team is A and out[0].evidence["best_alternatives"] == {"pass": 3}


# --------------------------------------------------------------------------- team fouls / bonus


def _fouls(team: Team, n: int, period: int, start: int = 0, game_clock: float | None = None) -> list[Event]:
    attrs = {"period": period} | ({} if game_clock is None else {"game_clock": game_clock})
    fouler = "h1" if team is H else "a1"
    return [ev(f"f{team.value}{period}{start + k}", EventKind.FOUL, 100.0 + start + k, team=team, pid=fouler,
               attrs=dict(attrs)) for k in range(n)]


def test_team_fouls_need_rules_and_foul_events() -> None:
    assert [i for i in engine().update(FakeStats(period=3), _fouls(H, 4, 3), 1.0) if i.kind == "team_fouls"] == []
    assert [i for i in engine(rules=rules_for("nba")).update(FakeStats(period=3), [], 1.0)] == []


def test_team_fouls_nba_limit_and_bonus() -> None:
    e = engine(rules=rules_for("nba"))
    out = [i for i in e.update(FakeStats(period=3), _fouls(H, 3, 2) + _fouls(H, 4, 3), 1.0) if i.kind == "team_fouls"]
    assert [i.text for i in out] == ["{team} 4 team fouls in Q3, next is two shots"]
    out = [i for i in e.update(FakeStats(period=3), _fouls(H, 1, 3, start=10), 2.0) if i.kind == "team_fouls"]
    assert [i.text for i in out] == ["{team} 5 team fouls in Q3, opponent in the bonus"]
    technical = ev("tech1", EventKind.FOUL, 3.0, team=H, pid="h1", attrs={"period": 3, "type": "technical"})
    assert [i for i in e.update(FakeStats(period=3), [technical], 3.0) if i.kind == "team_fouls"] == []


def test_penalty_state_rule_variants() -> None:
    nba, ncaa, fiba = rules_for("nba"), rules_for("ncaa_m"), rules_for("fiba")
    # NCAA men: one-and-one from the 7th, two shots from the 10th, per half
    assert penalty_state(ncaa, 1, None, [(1, None)] * 5).award == "none"
    assert penalty_state(ncaa, 1, None, [(1, None)] * 6).award == "one_and_one"
    st = penalty_state(ncaa, 1, None, [(1, None)] * 7)
    assert st.award == "one_and_one" and st.in_bonus
    assert penalty_state(ncaa, 1, None, [(1, None)] * 9).award == "two_shots"
    # NBA last 2:00: under the limit, the 2nd foul inside the window gives two shots
    late = penalty_state(nba, 2, 90.0, [(2, 500.0), (2, 100.0)])
    assert late.two_shots_from == 3 and late.award == "two_shots"
    assert penalty_state(nba, 2, 500.0, [(2, 600.0), (2, 550.0)]).award == "none"
    # NBA OT: own count, penalty from the 4th; FIBA OT counts toward Q4
    assert penalty_state(nba, 5, 200.0, [(4, 300.0)] * 4 + [(5, 250.0)] * 3).award == "two_shots"
    assert penalty_state(nba, 5, 200.0, [(4, 300.0)] * 4 + [(5, 250.0)] * 2).award == "none"
    assert penalty_state(fiba, 5, 200.0, [(4, 300.0)] * 3 + [(5, 250.0)]).fouls == 4


def test_team_fouls_ncaa_one_and_one_wording() -> None:
    e = engine("ncaa_m", rules=rules_for("ncaa_m"))
    out = [i for i in e.update(FakeStats(period=1), _fouls(A, 6, 1), 1.0) if i.kind == "team_fouls"]
    assert [(i.text, i.team) for i in out] == [("{team} 6 team fouls in H1, next is one-and-one", A)]


# --------------------------------------------------------------------------- dedupe, cooldown, active


def test_identical_update_is_deduped() -> None:
    e = engine()
    st = FakeStats(period=2, players={"h1": FakePlayerLine("h1", H, pf=3)}, shots=shots(H, "h2", [False, True, True, True]))
    assert sorted(kinds(e.update(st, [], 1.0))) == ["foul_trouble", "hot_hand"]
    assert e.update(st, [], 2.0) == []


def test_cooldown_uses_game_clock_and_resets_each_period() -> None:
    e = engine()
    base = shots(H, "h1", [False, True, True, True])
    assert kinds(e.update(FakeStats(period=2, game_clock=600.0, shots=base), [], 1.0)) == ["hot_hand"]
    more = base + shots(H, "h1", [True], t0=10, prefix="x")
    assert e.update(FakeStats(period=2, game_clock=520.0, shots=more), [], 200.0) == []  # 80 s of game clock
    assert kinds(e.update(FakeStats(period=2, game_clock=470.0, shots=more), [], 260.0)) == ["hot_hand"]
    more2 = more + shots(H, "h1", [True], t0=20, prefix="y")
    assert kinds(e.update(FakeStats(period=3, game_clock=719.0, shots=more2), [], 270.0)) == ["hot_hand"]


def test_cooldown_falls_back_to_video_time_without_clock() -> None:
    e = engine()
    base = shots(H, "h1", [False, True, True, True])
    assert kinds(e.update(FakeStats(period=None, game_clock=None, shots=base), [], 10.0)) == ["hot_hand"]
    more = base + shots(H, "h1", [True], t0=10, prefix="x")
    assert e.update(FakeStats(period=None, game_clock=None, shots=more), [], 60.0) == []
    assert kinds(e.update(FakeStats(period=None, game_clock=None, shots=more), [], 131.0)) == ["hot_hand"]


def test_active_keeps_top_three_by_priority_and_drops_cleared_cards() -> None:
    e = engine()
    st = FakeStats(period=4, players={"h1": FakePlayerLine("h1", H, pf=5), "a1": FakePlayerLine("a1", A, pf=5),
                                      "h2": FakePlayerLine("h2", H, pf=1)},
                   run={"team": A, "points": 9, "opp_points": 0, "since_t": 1.0},
                   shots=shots(H, "h2", [False, True, True, True]),
                   teams={H: FakeTeamLine(fga=20, fta=9), A: FakeTeamLine()})
    out = e.update(st, [], 5.0)
    assert sorted(kinds(out)) == ["foul_trouble", "foul_trouble", "ft_rate", "hot_hand", "run"]
    assert [i.priority for i in out] == sorted((i.priority for i in out), reverse=True)
    assert kinds(e.active()) == ["foul_trouble", "foul_trouble", "run"]
    st.run = {"team": None, "points": 0, "opp_points": 0, "since_t": 9.0}
    st.players["a1"].pf = 2
    e.update(st, [], 6.0)
    assert kinds(e.active()) == ["foul_trouble", "hot_hand", "ft_rate"]


def test_news_cards_expire_after_ttl_but_state_cards_stay() -> None:
    e = engine(active_ttl_s=100.0)
    st = FakeStats(period=2, players={"h1": FakePlayerLine("h1", H, pf=3)}, shots=shots(H, "h2", [False, True, True, True]))
    e.update(st, [], 1.0)
    e.update(st, [], 200.0)
    assert kinds(e.active()) == ["foul_trouble"]


def test_every_card_is_short_has_n_and_uses_placeholders_only() -> None:
    e = engine(rules=rules_for("nba"))
    decs = [{"t": float(i), "player_id": "h1", "label": "poor", "delta": -0.4, "best": {"kind": "shoot"}} for i in range(3)]
    st = FakeStats(
        period=2, players={"h1": FakePlayerLine("h1", H, pf=3), "a1": FakePlayerLine("a1", A, pf=5)},
        run={"team": A, "points": 10, "opp_points": 2, "since_t": 0.0},
        shots=(shots(H, "h1", [False] * 4 + [True] * 12, zone="midrange", xfg=0.35)
               + shots(A, "a1", [True] * 5 + [False] * 11, zone="above_break_three", value=3, xfg=0.36)),
        teams={H: FakeTeamLine(fga=16, fta=9, oreb=8), A: FakeTeamLine(fga=16, dreb=6)},
        lineups={"L1": {"team": "away", "players": ["a1"], "plus_minus": -9, "possessions": 25, "current": True}},
        decisions=decs,
    )
    _feed_speeds(e, 20.0, 14.0)
    out = e.update(st, _possessions_and_turnovers(3) + _fouls(H, 4, 2), 545.0)
    assert set(kinds(out)) == {"foul_trouble", "run", "team_fouls", "turnovers", "shot_quality", "decision",
                               "oreb", "hot_hand", "lineup", "fatigue", "ft_rate", "shot_diet"}
    for ins in out:
        assert word_count(ins.text) <= 12, ins.text
        assert not any(name in ins.text for name in NAMES), ins.text
        assert "n" in ins.evidence and "n_unit" in ins.evidence
        assert "timeout" not in ins.text.lower()
    assert len({i.insight_id for i in out}) == len(out) and len(e.active()) == 3


def test_render_inserts_names_locally() -> None:
    e = engine()
    out = e.update(FakeStats(period=2, players={"h1": FakePlayerLine("h1", H, pf=3)},
                             run={"team": A, "points": 8, "opp_points": 0, "since_t": 0.0}), [], 1.0)
    rendered = {i.kind: e.render(i) for i in out}
    assert rendered == {"foul_trouble": "Marcus Hill 3 fouls in Q2 (Q+1)", "run": "Summit on an 8-0 run"}
