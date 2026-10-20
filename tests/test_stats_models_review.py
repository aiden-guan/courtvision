"""Regression tests for defects found in the adversarial review of stats_models.

Each test names the defect it pins down. Hand-built inputs only; deterministic, no network.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from courtvision.geometry.court import court_for
from courtvision.stats.decision import DecisionConfig, DecisionModel
from courtvision.stats.insights import InsightConfig, InsightEngine
from courtvision.stats.shot_quality import ShotFeatures, ShotQualityModel
from courtvision.types import (
    Event,
    EventKind,
    FrameState,
    PlayerState,
    Roster,
    RosterEntry,
    ScoreboardReading,
    Team,
)

H, A = Team.HOME, Team.AWAY
COURT = court_for("nba")
BX, BY = COURT.basket_xy("left")


def _p(pid: str, team: Team, along: float, lateral: float, vx: float = 0.0, vy: float = 0.0) -> PlayerState:
    """Player at (along, lateral) ft from the left basket centre."""
    return PlayerState(pid, team, BX + along, BY + lateral, vx, vy)


def _frame(players: list[PlayerState], sb: ScoreboardReading | None = None) -> FrameState:
    return FrameState(frame_idx=30, t=1.0, players=players, ball=None, scoreboard=sb)


def _pass(passer: str, receiver: str, **kw) -> Event:
    return Event("e1", EventKind.PASS, 1.0, 30, team=H, player_id=passer, other_player_id=receiver, **kw)


@pytest.fixture(scope="module")
def nba() -> ShotQualityModel:
    return ShotQualityModel("nba")


@pytest.fixture(scope="module")
def model(nba: ShotQualityModel) -> DecisionModel:
    return DecisionModel(COURT, nba)


# --------------------------------------------------------------------------- shot quality


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_defender_distance_is_treated_as_unknown(nba: ShotQualityModel, bad: float) -> None:
    # Defect: NaN closest_defender_ft made xfg() return NaN (the [0.01, 0.99] clip passes NaN).
    for d, v in ((3.0, 2), (18.0, 2), (25.0, 3)):
        f = ShotFeatures(d, v, closest_defender_ft=bad)
        assert nba.xfg(f) == nba.xfg(ShotFeatures(d, v))
        assert np.all(np.isfinite(nba.residual_features(f)))
        assert math.isfinite(nba.expected_points(f))


def test_missing_shot_type_prices_as_a_jumper(nba: ShotQualityModel) -> None:
    # Defect: shot_type=None (attrs without a type) silently dropped the catch-and-shoot effect.
    none_type = ShotFeatures(24.5, 3, shot_type=None, catch_and_shoot=True)  # type: ignore[arg-type]
    assert nba.xfg(none_type) == nba.xfg(ShotFeatures(24.5, 3, catch_and_shoot=True))


def test_invalid_value_raises_value_error_not_type_error(nba: ShotQualityModel) -> None:
    with pytest.raises(ValueError):
        nba.xfg(ShotFeatures(10.0, None))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        nba.xfg(ShotFeatures(None, 2))  # type: ignore[arg-type]


def test_fit_skips_unlabelled_attempts_instead_of_counting_misses() -> None:
    # Defect: made=None (a pending ShotRecord) became bool(None) = a miss.
    rng = np.random.default_rng(1)
    feats = [ShotFeatures(float(d), 2) for d in rng.uniform(0, 20, 60)]
    made = [bool(x) for x in rng.integers(0, 2, 60)]
    ref = ShotQualityModel("nba")
    ref.fit(feats, made)
    m = ShotQualityModel("nba")
    m.fit(feats + [ShotFeatures(5.0, 2)] * 40, made + [None] * 40)  # type: ignore[list-item]
    assert m.fit_info["n"] == 60 and m.fit_info["n_unlabelled_skipped"] == 40
    np.testing.assert_allclose(m.coef, ref.coef)
    with pytest.raises(ValueError):
        ShotQualityModel("nba").fit([ShotFeatures(5.0, 2)], [None])  # type: ignore[list-item]


# --------------------------------------------------------------------------- decision model


def _reversal_frame() -> FrameState:
    # Holder at the top, on-ball defender 4 ft in front; the receiver is far out on the wing
    # (a poor catch-and-shoot) and loosely guarded: an ordinary ball reversal.
    return _frame([_p("h1", H, 28.0, 0.0), _p("a1", A, 24.0, 0.0), _p("h2", H, 26.0, 18.0), _p("a2", A, 23.0, 16.0)])


def test_routine_reset_pass_is_not_graded_poor(model: DecisionModel, nba: ShotQualityModel) -> None:
    # Defect: a pass was priced as a forced catch-and-shoot, so every reversal to a player who
    # should not shoot graded "poor" (sim: 90 of 154 passes poor, median delta -0.33).
    g = model.grade(_reversal_frame(), _pass("h1", "h2"), H, "left")
    assert g is not None and g.label != "poor"
    hold = next(o for o in g.options if o.kind == "hold")
    assert g.chosen.detail["receiver_action"] == "reset"
    assert g.chosen.xpts_if_success == pytest.approx(hold.value, rel=0.02)
    assert g.chosen.value == pytest.approx(g.chosen.success_prob * g.chosen.xpts_if_success)
    old = DecisionModel(COURT, nba, DecisionConfig(receiver_can_reset=False))
    assert old.grade(_reversal_frame(), _pass("h1", "h2"), H, "left").label == "poor"


def test_open_shot_after_pass_still_values_the_shot(model: DecisionModel) -> None:
    fr = _frame([_p("h1", H, 25.0, 0.0), _p("a1", A, 24.0, 0.0), _p("h2", H, 2.0, -22.5), _p("a2", A, 2.0, 2.5)])
    o = next(o for o in model.options(fr, "h1", H, "left") if o.target_id == "h2")
    assert o.detail["receiver_action"] == "shoot"
    assert o.xpts_if_success == pytest.approx(o.detail["receiver_shot_value"])


def test_on_ball_defender_in_front_of_passer_is_not_a_lane_defender(model: DecisionModel) -> None:
    # Defect: the lane started 3 ft from the passer, so the on-ball defender guarding from ~4 ft
    # was the "lane defender" on most simulated passes.
    def p_complete(m: DecisionModel, guarded: bool) -> float:
        on_ball = [_p("a1", A, 22.0, 0.0)] if guarded else []  # 4 ft out, between passer and basket
        fr = _frame([_p("h1", H, 26.0, 0.0), _p("h2", H, 14.0, 18.0), _p("a9", A, 60.0, 40.0)] + on_ball)
        return next(o for o in m.options(fr, "h1", H, "left") if o.target_id == "h2").success_prob

    assert p_complete(model, True) == pytest.approx(p_complete(model, False), abs=0.01)
    legacy = DecisionModel(COURT, model.shot_model, DecisionConfig(lane_ignore_near_passer_ft=3.0))
    assert p_complete(legacy, True) < p_complete(legacy, False) - 0.015


@pytest.mark.parametrize("offset, risky", [(2.5, True), (-2.5, False)])
def test_receiver_defender_counts_only_when_fronting(model: DecisionModel, offset: float, risky: bool) -> None:
    # Defect: the lane ended at the catch point, so a defender beside / behind the receiver was
    # "in the lane". A defender fronting the receiver (toward the passer) still is.
    px, py, rx, ry = 24.0, 0.0, 20.0, -15.0
    length = math.hypot(px - rx, py - ry)
    ux, uy = (px - rx) / length, (py - ry) / length
    fr = _frame([_p("h1", H, px, py), _p("h2", H, rx, ry), _p("a2", A, rx + ux * offset, ry + uy * offset)])
    o = next(o for o in model.options(fr, "h1", H, "left") if o.target_id == "h2")
    if risky:
        assert o.detail["lane_defender_ft"] < 0.5 and o.success_prob < 0.8
    else:
        assert o.detail["lane_defender_ft"] > 4.0 and o.success_prob > 0.95


def test_hold_value_runs_down_with_the_game_clock(model: DecisionModel, nba: ShotQualityModel) -> None:
    # Defect: with the shot clock off (or longer than the game clock) a reset kept full league PPP
    # with 6 s left in the period.
    players = [_p("h1", H, 20.0, 0.0)]

    def hold(sb: ScoreboardReading, m: DecisionModel = model):
        return next(o for o in m.options(_frame(players, sb), "h1", H, "left") if o.kind == "hold")

    assert hold(ScoreboardReading(t=1.0, game_clock=6.5)).value == pytest.approx(1.160 * 0.5)
    assert hold(ScoreboardReading(t=1.0, game_clock=6.5, shot_clock=20.0)).value == pytest.approx(1.160 * 0.5)
    assert hold(ScoreboardReading(t=1.0, game_clock=300.0, shot_clock=6.5)).value == pytest.approx(1.160 * 0.5)
    off = DecisionModel(COURT, nba, DecisionConfig(hold_uses_game_clock=False))
    assert hold(ScoreboardReading(t=1.0, game_clock=6.5), off).value == pytest.approx(1.160)


def test_non_finite_positions_are_dropped(model: DecisionModel) -> None:
    nan = float("nan")
    fr = _frame([_p("h1", H, 25.0, 0.0), PlayerState("h2", H, nan, nan), _p("h3", H, 10.0, 15.0),
                 PlayerState("a1", A, nan, 10.0), _p("a2", A, 12.0, 14.0)])
    opts = model.options(fr, "h1", H, "left")
    assert [(o.kind, o.target_id) for o in opts] == [("shoot", None), ("pass", "h3"), ("hold", None)]
    assert all(math.isfinite(o.value) for o in opts)
    assert model.options(_frame([PlayerState("h1", H, nan, 1.0)]), "h1", H, "left") == []


def test_holder_located_by_event_xy_is_not_offered_a_pass_to_themself(model: DecisionModel) -> None:
    # Defect: when the event's player id is not in the frame (the holder is there under a
    # provisional track id) the holder's own track became a zero-length "pass" target.
    fr = _frame([_p("home-t7", H, 25.0, 0.0), _p("h2", H, 10.0, 15.0), _p("a2", A, 12.0, 14.0)])
    g = model.grade(fr, _pass("h1", "h2", x=BX + 25.0, y=BY), H, "left")
    assert g is not None and {o.target_id for o in g.options if o.kind == "pass"} == {"h2"}


# --------------------------------------------------------------------------- insights


ROSTER = Roster(entries=[RosterEntry("h1", H, "23", "Ann Lee"), RosterEntry("h2", H, "5", "Bo Kim"),
                         RosterEntry("a1", A, "11", "Cy Dor")])


class _Stats:
    """Minimal GameStats stand-in (attribute access per the ARCHITECTURE contract)."""

    def __init__(self, **kw) -> None:
        self.period, self.game_clock = kw.pop("period", 2), kw.pop("game_clock", 600.0)
        self.players, self.shots = kw.pop("players", {}), kw.pop("shots", [])
        self.decisions, self.teams, self.lineups = kw.pop("decisions", []), {}, {}
        self.run = {"team": None, "points": 0, "opp_points": 0, "since_t": None}
        assert not kw


class _Line:
    def __init__(self, pid: str, team: Team | None, pf: int) -> None:
        self.player_id, self.team, self.pf = pid, team, pf


def _shot(pid: str | None, i: int, made: bool, team: Team = H) -> dict:
    return {"shot_id": f"s{pid}{i}", "player_id": pid, "team": team, "t": float(i), "value": 2,
            "made": made, "zone": "paint", "xfg": 0.5}


def _streak(pid: str | None, team: Team = H) -> list[dict]:
    return [_shot(pid, i, m, team) for i, m in enumerate([False, True, True, True])]


def _engine(**cfg) -> InsightEngine:
    return InsightEngine(ROSTER, COURT, InsightConfig(**cfg))


def test_cooldown_is_per_card_type_not_per_subject() -> None:
    # Defect: the cooldown was kept per (type, subject), so the same card type posted for several
    # players within seconds (sim: 5 decision cards inside 30 s). Spec: per-kind cooldown.
    e = _engine()
    both = _Stats(shots=_streak("h1") + _streak("h2"))
    first = [i for i in e.update(both, [], 1.0) if i.kind == "hot_hand"]
    assert len(first) == 1
    assert [i for i in e.update(_Stats(shots=both.shots, game_clock=530.0), [], 70.0) if i.kind == "hot_hand"] == []
    later = [i for i in e.update(_Stats(shots=both.shots, game_clock=470.0), [], 140.0) if i.kind == "hot_hand"]
    assert len(later) == 1 and later[0].player_id != first[0].player_id
    per_subject = _engine(cooldown_per_subject=True)
    assert len([i for i in per_subject.update(both, [], 1.0) if i.kind == "hot_hand"]) == 2


def test_card_whose_condition_returns_is_shown_again() -> None:
    # Defect: a foul retraction cleared the Q+1 card, and when the foul came back the dedupe key
    # was already "seen", so a player in foul trouble showed no card at all.
    e = _engine()
    out = e.update(_Stats(players={"h1": _Line("h1", H, 3)}), [], 1.0)
    assert [i.kind for i in out] == ["foul_trouble"]
    assert e.update(_Stats(players={"h1": _Line("h1", H, 2)}), [], 2.0) == [] and e.active() == []
    assert e.update(_Stats(players={"h1": _Line("h1", H, 3)}), [], 3.0) == []  # not re-sent
    assert [i.insight_id for i in e.active()] == [out[0].insight_id]  # but visible again


def test_fouled_out_card_expires_while_foul_trouble_stays() -> None:
    # Defect: "fouled out" is a state kind, so it held one of the 3 visible slots for the rest of
    # the game (priority 85, above runs).
    e = _engine(active_ttl_s=100.0)
    st = _Stats(period=4, players={"h1": _Line("h1", H, 6), "a1": _Line("a1", A, 5)})
    assert sorted(i.text for i in e.update(st, [], 1.0)) == ["{player} 5 fouls in Q4, one from fouling out",
                                                           "{player} fouled out with 6 fouls"]
    e.update(st, [], 200.0)
    assert [i.player_id for i in e.active()] == ["a1"]


def test_hot_hand_ignores_attempts_without_a_shooter() -> None:
    # Defect: shots with player_id None were grouped under the player id "None".
    out = _engine().update(_Stats(shots=_streak(None)), [], 1.0)
    assert [i for i in out if i.kind == "hot_hand"] == []


def test_decision_window_uses_time_order() -> None:
    # Defect: "last 8 graded" took the list order; with an out-of-order list, 3 old poor grades
    # still triggered a card after 8 newer good ones.
    old = [{"t": float(i), "player_id": "h1", "label": "poor", "delta": -0.4, "event_id": f"p{i}",
            "best": {"kind": "pass"}} for i in range(3)]
    new = [{"t": 10.0 + i, "player_id": "h1", "label": "good", "delta": 0.0, "event_id": f"g{i}",
            "best": {"kind": "shoot"}} for i in range(8)]
    out = _engine().update(_Stats(decisions=list(reversed(old + new))), [], 20.0)
    assert [i for i in out if i.kind == "decision"] == []
    fired = _engine().update(_Stats(decisions=list(reversed(new[:4] + old))), [], 20.0)
    assert [i.kind for i in fired if i.kind == "decision"] == ["decision"]


def test_provisional_ids_resolve_their_team_from_the_id_prefix() -> None:
    # Defect: "away-t7" style ids (types.py PlayerState) got team None on decision / fatigue cards,
    # unlike stats-core's TeamResolver.
    decs = [{"t": float(i), "player_id": "away-t7", "label": "poor", "delta": -0.4, "event_id": f"d{i}",
             "best": {"kind": "pass"}} for i in range(3)]
    out = [i for i in _engine().update(_Stats(decisions=decs), [], 5.0) if i.kind == "decision"]
    assert [i.team for i in out] == [A]


def test_turnover_burst_falls_back_when_possession_events_are_missing() -> None:
    # Defect: 3 turnovers after a single POSSESSION event read "3 turnovers in last 1 possessions".
    evs = [Event("p0", EventKind.POSSESSION, 0.0, 0, team=H, player_id="h1")]
    evs += [Event(f"t{k}", EventKind.TURNOVER, 10.0 * (k + 1), 30 * (k + 1), team=H, player_id="h1") for k in range(3)]
    out = [i for i in _engine().update(_Stats(), evs, 40.0) if i.kind == "turnovers"]
    assert [i.text for i in out] == ["{team} 3 turnovers in last 3 minutes"]


class _TLine(_Line):
    def __init__(self, pid: str, team: Team, pf: int, tf: int) -> None:
        super().__init__(pid, team, pf)
        self.tf = tf


@pytest.mark.parametrize("league, counts", [("nfhs", True), ("fiba", True), ("nba", False), ("ncaa_m", False)])
def test_technicals_count_toward_foul_trouble_where_the_rules_say_so(league: str, counts: bool) -> None:
    # Defect: NFHS (Case 4.14.1) and FIBA count technicals toward foul-out (RuleSpec
    # foul_out_counts_technicals), but the rule read PlayerLine.pf only (technicals live in tf).
    from courtvision.geometry.rules import rules_for

    st = _Stats(period=4, players={"h1": _TLine("h1", H, 3, 1)})
    for rules in (rules_for(league), None):
        e = InsightEngine(ROSTER, court_for(league), rules=rules)
        out = [i for i in e.update(st, [], 1.0) if i.kind == "foul_trouble"]
        foul_out = 6 if league == "nba" else 5
        if counts:
            assert [i.evidence["n"] for i in out] == [4] and out[0].evidence["tf"] == 1
            assert "one from fouling out" in out[0].text and foul_out == 5
        else:
            assert out == []  # 3 PF in Q4 is under the Q+1 threshold; the technical does not count


def test_xefg_needs_an_nba_ruleset_too() -> None:
    # Defect: the gate read only court.league, so an NBA court with an NFHS RuleSpec got xeFG.
    from courtvision.geometry.rules import rules_for

    shots = [_shot("h1", i, i < 12) for i in range(16)]
    nba = InsightEngine(ROSTER, COURT, rules=rules_for("nba")).update(_Stats(shots=shots), [], 1.0)
    assert [i.evidence.get("model") for i in nba if i.kind == "shot_quality"] == ["xeFG (NBA priors)"]
    hs = InsightEngine(ROSTER, COURT, rules=rules_for("nfhs")).update(_Stats(shots=shots), [], 1.0)
    assert all("expected" not in i.text and "xefg" not in i.evidence for i in hs)


def test_window_run_is_not_reannounced_when_it_requalifies() -> None:
    # Defect: an old basket leaving stats-core's 300 s window dropped a 10-2 below the bar; the
    # next basket re-qualified it as a "new" run and the same 10-2 was announced again.
    def st(home: int, away: int) -> _Stats:
        s = _Stats()
        s.run = {"team": H, "points": 2, "opp_points": 0, "since_t": 1.0, "window_s": 300.0,
                 "window": {H: home, A: away}, "window_leader": H if home > away else A}
        return s

    e = _engine()
    assert [i.text for i in e.update(st(10, 2), [], 100.0)] == ["{team} on a 10-2 run"]
    assert e.update(st(8, 2), [], 130.0) == []
    assert e.update(st(10, 2), [], 160.0) == [] and [i.kind for i in e.active()] == ["run"]
    assert [i.text for i in e.update(st(12, 2), [], 170.0)] == ["{team} on a 12-2 run"]  # +4 update
    e.update(st(0, 0), [], 480.0)  # the window has turned over: those baskets are gone
    assert [i.text for i in e.update(st(10, 2), [], 600.0)] == ["{team} on a 10-2 run"]  # a later run


def test_sub_run_is_not_announced_after_the_bigger_run() -> None:
    # Defect (sim, 40 min): "on a 13-2 run" then, 9 s later, "on an 8-0 run": the unanswered view
    # was preferred over the bigger window view of the same stretch.
    e = _engine()
    s = _Stats()
    s.run = {"team": H, "points": 2, "opp_points": 0, "since_t": 90.0, "window_s": 300.0,
             "window": {H: 13, A: 2}, "window_leader": H}
    assert [i.text for i in e.update(s, [], 100.0)] == ["{team} on a 13-2 run"]
    s.run = dict(s.run, points=8, since_t=60.0)  # the last 8 unanswered, inside the same 13-2
    assert e.update(s, [], 109.0) == [] and [i.text for i in e.active()] == ["{team} on a 13-2 run"]
    s.run = dict(s.run, window={H: 9, A: 2})  # the window lost the early baskets; the 8-0 remains
    assert e.update(s, [], 140.0) == [] and [i.text for i in e.active()] == ["{team} on a 13-2 run"]
    s.run = dict(s.run, points=13, window={H: 13, A: 0})  # "13-0" right after "13-2": same stretch
    assert e.update(s, [], 150.0) == []
    s.run = dict(s.run, points=8, since_t=200.0, window={H: 8, A: 4})  # a new 8-0 after A answered
    assert [i.text for i in e.update(s, [], 230.0)] == ["{team} on an 8-0 run"]


def test_rulespec_without_a_penalty_threshold_gives_no_bonus_card(caplog: pytest.LogCaptureFixture) -> None:
    # Defect: a custom RuleSpec with no bonus.two_shots_from raised KeyError inside the rule on
    # every update (logged as an exception each time).
    import dataclasses

    from courtvision.geometry.rules import rules_for

    rules = dataclasses.replace(rules_for("nba"), bonus={"scope": "period"})
    e = InsightEngine(ROSTER, COURT, rules=rules)
    fouls = [Event(f"f{k}", EventKind.FOUL, 1.0 + k, 30 + k, team=H, player_id="h1", attrs={"period": 2})
             for k in range(6)]
    with caplog.at_level("ERROR"):
        out = e.update(_Stats(), fouls, 10.0)
    assert [i for i in out if i.kind == "team_fouls"] == [] and not caplog.records
