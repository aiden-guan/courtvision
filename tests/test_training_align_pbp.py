"""Tests for courtvision.training.align_pbp (clock parsing, PBP parsing, timeline fit, alignment, clips)."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from courtvision.events.engine import EventEngine
from courtvision.events.scoreboard import ScoreChange
from courtvision.sim.simulator import GameSimulator, SimConfig, SimResult
from courtvision.training.align_pbp import (
    AlignConfig,
    ClockTimeline,
    LabeledClip,
    PBPAction,
    TimelineConfig,
    align_pbp,
    evaluate_against_pbp,
    format_clock,
    infer_team_map,
    load_pbp,
    make_clips,
    parse_clock,
    parse_pbp_v3,
    pbp_from_events,
    read_jsonl,
    score_changes,
    write_jsonl,
    write_pbp_csv,
)
from courtvision.types import EventKind, ScoreboardReading, Team

# =============================================================================================== clock text


@pytest.mark.parametrize("text,expected", [
    ("PT11M32.00S", 692.0), ("PT00M23.40S", 23.4), ("PT12M00.00S", 720.0), ("PT05M", 300.0), ("PT45.5S", 45.5),
    ("11:32", 692.0), ("0:23.4", 23.4), ("23.4", 23.4), (45, 45.0), (0.0, 0.0),
])
def test_parse_clock(text: object, expected: float) -> None:
    assert parse_clock(text) == pytest.approx(expected)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["", "PT", "abc", "-3", None, True, "11:xx", float("nan")])
def test_parse_clock_rejects(bad: object) -> None:
    with pytest.raises(ValueError):
        parse_clock(bad)  # type: ignore[arg-type]


def test_format_clock_round_trip() -> None:
    assert format_clock(692.0) == "PT11M32.00S"
    assert format_clock(23.4) == "PT00M23.40S"
    assert parse_clock(format_clock(23.4)) == pytest.approx(23.4)


# =============================================================================================== PBP parsing

BOS, NYK = 1610612738, 1610612752
V3_ROWS = [
    {"actionNumber": 1, "clock": "PT12M00.00S", "period": 1, "actionType": "period", "subType": "start",
     "description": "Start of 1st Period", "personId": 0},
    {"actionNumber": 4, "clock": "PT11M40.00S", "period": 1, "actionType": "Made Shot", "subType": "Jump Shot",
     "shotResult": "Made", "isFieldGoal": 1, "teamTricode": "BOS", "teamId": BOS, "personId": 1628369,
     "playerName": "Tatum", "location": "h", "scoreHome": "3", "scoreAway": "0",
     "description": "Tatum 25' 3PT Jump Shot (3 PTS) (Brown 1 AST)", "xLegacy": -120, "yLegacy": 220},
    {"actionNumber": 6, "clock": "PT11M20.00S", "period": 1, "actionType": "Missed Shot", "subType": "Driving Layup Shot",
     "shotResult": "Missed", "teamTricode": "NYK", "teamId": NYK, "personId": 1629628, "playerName": "Barrett",
     "location": "v", "description": "MISS Barrett 2' Driving Layup"},
    {"actionNumber": 7, "clock": "PT11M18.00S", "period": 1, "actionType": "Rebound", "subType": "",
     "teamTricode": "BOS", "teamId": BOS, "personId": 1627759, "playerName": "Brown", "location": "h",
     "description": "Brown REBOUND (Off:0 Def:1)"},
    {"actionNumber": 9, "clock": "PT10M58.00S", "period": 1, "actionType": "Turnover", "subType": "Bad Pass",
     "teamTricode": "BOS", "teamId": BOS, "personId": 1628369, "playerName": "Tatum", "location": "h",
     "description": "Tatum Bad Pass Turnover (P1.T1)"},
    {"actionNumber": 10, "clock": "PT10M58.00S", "period": 1, "actionType": "", "subType": "",
     "teamTricode": "NYK", "teamId": NYK, "personId": 1629628, "playerName": "Barrett", "location": "v",
     "description": "Barrett STEAL (1 STL)"},
    {"actionNumber": 12, "clock": "PT10M40.00S", "period": 1, "actionType": "Foul", "subType": "Shooting",
     "teamTricode": "NYK", "teamId": NYK, "personId": 203944, "playerName": "Randle", "location": "v",
     "description": "Randle S.FOUL (P1.T1)"},
    {"actionNumber": 13, "clock": "PT10M40.00S", "period": 1, "actionType": "Free Throw", "subType": "Free Throw 1 of 2",
     "shotResult": "Missed", "teamTricode": "BOS", "teamId": BOS, "personId": 1627759, "playerName": "Brown",
     "location": "h", "description": "MISS Brown Free Throw 1 of 2"},
    {"actionNumber": 14, "clock": "PT10M40.00S", "period": 1, "actionType": "Rebound", "subType": "",
     "teamTricode": "BOS", "teamId": BOS, "personId": BOS, "playerName": "", "location": "h", "description": "BOS Rebound"},
    {"actionNumber": 15, "clock": "PT10M40.00S", "period": 1, "actionType": "Free Throw", "subType": "Free Throw 2 of 2",
     "shotResult": "Made", "teamTricode": "BOS", "teamId": BOS, "personId": 1627759, "playerName": "Brown",
     "location": "h", "scoreHome": "4", "scoreAway": "0", "description": "Brown Free Throw 2 of 2 (1 PTS)"},
    {"actionNumber": 17, "clock": "PT10M20.00S", "period": 1, "actionType": "Missed Shot", "subType": "Jump Shot",
     "shotResult": "Missed", "teamTricode": "NYK", "teamId": NYK, "personId": 203944, "playerName": "Randle",
     "location": "v", "description": "MISS Randle 18' Jump Shot"},
    {"actionNumber": 18, "clock": "PT10M18.00S", "period": 1, "actionType": "Rebound", "subType": "",
     "teamTricode": "NYK", "teamId": NYK, "personId": 203944, "playerName": "Randle", "location": "v",
     "description": "Randle REBOUND (Off:1 Def:0)"},
    {"actionNumber": 20, "clock": "PT10M00.00S", "period": 1, "actionType": "Violation", "subType": "Kicked Ball",
     "teamTricode": "NYK", "location": "v", "description": "Randle Violation:Kicked Ball"},
]


def _by_number(actions: list[PBPAction]) -> dict[tuple[int, bool], PBPAction]:
    return {(a.action_number, bool(a.attrs.get("derived"))): a for a in actions}


def test_parse_v3_rows() -> None:
    acts = parse_pbp_v3(V3_ROWS)
    a = _by_number(acts)
    assert a[(1, False)].kind is EventKind.PERIOD_START
    three = a[(4, False)]
    assert (three.kind, three.value, three.made, three.team) == (EventKind.SHOT_MADE, 3, True, Team.HOME)
    assert three.player_id == "1628369" and three.attrs["shot_type"] == "jumper" and three.x == -120
    assert three.attrs["xy_units"].startswith("legacy")
    assist = a[(4, True)]
    assert assist.kind is EventKind.ASSIST and assist.player_name == "Brown" and assist.attrs["parent_action"] == 4
    layup = a[(6, False)]
    assert (layup.kind, layup.value, layup.attrs["shot_type"], layup.team) == (EventKind.SHOT_MISSED, 2, "layup", Team.AWAY)
    assert a[(7, False)].attrs["offensive"] is False  # BOS rebounds a NYK miss
    assert a[(9, False)].attrs["type"] == "bad_pass"
    assert a[(10, False)].kind is EventKind.STEAL and a[(10, False)].team is Team.AWAY
    ft1, ft2 = a[(13, False)], a[(15, False)]
    assert ft1.is_free_throw and ft1.made is False and (ft1.attrs["ft_index"], ft1.attrs["ft_total"]) == (1, 2)
    assert ft2.made is True and ft2.value == 1 and ft2.attrs["ft_index"] == 2
    team_reb = a[(14, False)]
    assert team_reb.player_id is None and team_reb.synthetic  # team rebound after FT 1 of 2 is bookkeeping
    assert a[(18, False)].attrs["offensive"] is True
    assert a[(20, False)].kind is None


def test_parse_v3_infers_teams_without_location() -> None:
    rows = [{k: v for k, v in r.items() if k != "location"} for r in V3_ROWS]
    acts = parse_pbp_v3(rows)
    assert infer_team_map(acts) == {"BOS": Team.HOME, "NYK": Team.AWAY}
    assert {a.team_tricode: a.team for a in acts if a.team_tricode} == {"BOS": Team.HOME, "NYK": Team.AWAY}


def test_infer_team_map_from_first_basket_only() -> None:
    rows = [
        {"actionNumber": 2, "clock": "PT11M50.00S", "period": 1, "actionType": "Made Shot", "shotResult": "Made",
         "teamTricode": "NYK", "scoreHome": "0", "scoreAway": "2", "description": "Brunson 12' Jump Shot (2 PTS)"},
        {"actionNumber": 3, "clock": "PT11M30.00S", "period": 1, "actionType": "Missed Shot", "shotResult": "Missed",
         "teamTricode": "BOS", "description": "MISS Tatum 3PT"},
    ]
    assert infer_team_map(parse_pbp_v3(rows)) == {"NYK": Team.AWAY, "BOS": Team.HOME}


def test_parse_generic_free_throw_aliases() -> None:
    from courtvision.training.align_pbp import parse_pbp_generic

    acts = parse_pbp_generic([
        {"action_number": 1, "period": 1, "clock": "5:00", "kind": "free_throw_made", "team": "home", "ft_index": 1, "ft_total": 2},
        {"action_number": 2, "period": 1, "clock": "5:00", "kind": "free_throw_missed", "team": "home", "ft_index": 2, "ft_total": 2},
        {"action_number": 3, "period": 1, "clock": "4:58", "kind": "rebound", "team": "away"},
    ])
    assert [a.value for a in acts[:2]] == [1, 1] and all(a.is_free_throw for a in acts[:2])
    assert acts[0].made is True and acts[1].made is False
    assert acts[2].attrs["offensive"] is False and not acts[2].synthetic


def test_empty_timeline_leaves_actions_unplaced() -> None:
    tl = ClockTimeline.fit([])
    assert tl.periods == [] and tl.clock_at(10.0) is None and tl.video_t(1, 100.0) is None
    res = align_pbp(parse_pbp_v3(V3_ROWS), tl)
    assert not res.accepted and all(al.t is None and al.method == "unplaced" for al in res.aligned)


def test_parse_live_feed_person_fields() -> None:
    doc = {"game": {"actions": [
        {"actionNumber": 2, "clock": "PT11M50.00S", "period": 1, "actionType": "2pt", "subType": "Layup",
         "shotResult": "Made", "teamTricode": "BOS", "personId": 1, "location": "h", "assistPersonId": 2},
        {"actionNumber": 3, "clock": "PT11M30.00S", "period": 1, "actionType": "3pt", "subType": "Jump Shot",
         "shotResult": "Missed", "teamTricode": "NYK", "personId": 7, "location": "v", "blockPersonId": 3},
        {"actionNumber": 5, "clock": "PT11M10.00S", "period": 1, "actionType": "turnover", "subType": "lostball",
         "teamTricode": "NYK", "personId": 8, "location": "v", "stealPersonId": 4},
        {"actionNumber": 6, "clock": "PT11M10.00S", "period": 1, "actionType": "steal", "teamTricode": "BOS",
         "personId": 4, "location": "h"},
    ]}}
    acts = parse_pbp_v3(doc)
    kinds = Counter(a.kind for a in acts)
    assert kinds[EventKind.ASSIST] == 1 and kinds[EventKind.BLOCK] == 1
    assert kinds[EventKind.STEAL] == 1  # explicit steal row: no duplicate from stealPersonId
    block = next(a for a in acts if a.kind is EventKind.BLOCK)
    assert block.player_id == "3" and block.team is Team.HOME
    miss = next(a for a in acts if a.kind is EventKind.SHOT_MISSED)
    assert miss.value == 3
    assert next(a for a in acts if a.kind is EventKind.TURNOVER).attrs["type"] == "steal"


# =============================================================================================== timeline

# (t_start, t_end, clock_at_start, running) for a 150 s period, then period 2. Stops sit at non-integer
# clock values (a whistle can come at any instant).
_SEGMENTS = {
    1: [(0.0, 4.0, 150.0, False), (4.0, 40.4, 150.0, True), (40.4, 52.0, 113.6, False), (52.0, 80.0, 113.6, True),
        (80.0, 80.8, 85.6, False), (80.8, 100.8, 85.6, True), (100.8, 120.8, 65.6, False), (120.8, 170.5, 65.6, True),
        (170.5, 176.5, 15.9, False), (176.5, 192.4, 15.9, True), (192.4, 205.0, 0.0, False)],
    2: [(205.0, 214.0, 150.0, False), (214.0, 244.0, 150.0, True)],
}


def _true_clock(t: float) -> tuple[int, float]:
    for period, segs in _SEGMENTS.items():
        for t0, t1, c0, running in segs:
            if t0 <= t < t1 or (period == 2 and t == t1):
                return period, max(0.0, c0 - (t - t0)) if running else c0
    raise ValueError(t)


def _display(c: float) -> float:
    return float(math.floor(c + 1e-9)) if c >= 60.0 else math.floor(c * 10 + 1e-9) / 10.0


def _noisy_reads(seed: int = 0) -> list[ScoreboardReading]:
    rng = np.random.default_rng(seed)
    reads = []
    for t in np.arange(0.0, 244.0, 0.2):
        if 105.0 < t < 115.0:  # replay: scorebug hidden
            continue
        period, c = _true_clock(float(t))
        disp, per, conf = _display(c), period, float(rng.uniform(0.85, 1.0))
        u = rng.random()
        if u < 0.03:
            disp = max(0.0, disp + float(rng.choice([-60.0, -10.0, 10.0, 60.0, 100.0])))  # digit misreads
        elif u < 0.045:
            per = period + 1
        elif u < 0.06:
            conf = 0.2
        reads.append(ScoreboardReading(t=round(float(t), 3), period=per, game_clock=disp, source="ocr", confidence=conf))
    return reads


@pytest.fixture(scope="module")
def fitted() -> ClockTimeline:
    return ClockTimeline.fit(_noisy_reads(), TimelineConfig(clock_display="floor"))


def test_timeline_drops_noise(fitted: ClockTimeline) -> None:
    s = fitted.stats
    assert s["dropped_conf"] > 0 and s["dropped_period"] > 0 and s["dropped_monotone"] > 0
    assert fitted.periods == [1, 2]


def test_timeline_clock_accuracy(fitted: ClockTimeline) -> None:
    errs = []
    for t in np.arange(0.5, 243.5, 0.7):
        period, c = _true_clock(float(t))
        cp = fitted.clock_at(float(t))
        assert cp is not None and cp.period == period, t
        errs.append(abs(cp.clock - c))
    errs_a = np.array(errs)
    assert np.median(errs_a) < 0.2
    assert errs_a.max() < 0.75


@pytest.mark.parametrize("display,period,t_stop,t_restart", [
    (113.0, 1, 40.4, 52.0), (65.0, 1, 100.8, 120.8), (15.9, 1, 170.5, 176.5), (0.0, 1, 192.4, 205.0),
])
def test_timeline_finds_stoppages(fitted: ClockTimeline, display: float, period: int, t_stop: float, t_restart: float) -> None:
    iv = fitted.stop_interval(period, display)
    assert iv is not None
    assert iv[0] == pytest.approx(t_stop, abs=0.6)
    assert iv[1] == pytest.approx(t_restart, abs=0.6)
    assert fitted.video_t(period, display, "first") == pytest.approx(iv[0])
    assert fitted.video_t(period, display, "last") == pytest.approx(iv[1])
    assert fitted.clock_at((iv[0] + iv[1]) / 2).running is False


def test_timeline_inverse_on_running_clock(fitted: ClockTimeline) -> None:
    # display 100 (floor) = true clock in [100, 101): midpoint 100.5 is shown at t = 52 + (113.6 - 100.5)
    assert fitted.video_t(1, 100.0) == pytest.approx(52.0 + 13.1, abs=0.3)
    assert fitted.video_t(2, 140.0) == pytest.approx(214.0 + 9.5, abs=0.3)
    assert fitted.clock_at(66.0).running is True
    assert fitted.video_t(1, 400.0) is None  # far outside the period


def test_timeline_serialisation_round_trip(fitted: ClockTimeline) -> None:
    back = ClockTimeline.from_dict(fitted.to_dict())
    assert back.periods == fitted.periods
    assert back.config.clock_display == "floor"
    for t in (10.0, 45.0, 130.0, 230.0):
        assert back.clock_at(t) == fitted.clock_at(t)


def test_exact_timeline_matches_truth() -> None:
    reads = [ScoreboardReading(t=float(t), period=_true_clock(float(t))[0], game_clock=_true_clock(float(t))[1], source="sim")
             for t in np.arange(0.0, 244.0, 0.1)]
    tl = ClockTimeline.from_exact(reads)
    assert tl.exact
    assert tl.clock_at(60.0).clock == pytest.approx(_true_clock(60.0)[1], abs=1e-6)
    assert tl.stop_interval(1, 113.6) == pytest.approx((40.4, 52.0), abs=0.11)


# =============================================================================================== sim alignment


@pytest.fixture(scope="module")
def sim() -> SimResult:
    return GameSimulator(SimConfig(seconds=600.0, fps=10.0, seed=5, period_s=240.0)).run()


def _ocr_reads(sim: SimResult, seed: int = 1) -> list[ScoreboardReading]:
    """5 Hz OCR of the sim scoreboard: truncated display, digit misreads, period flips, a replay gap."""
    rng = np.random.default_rng(seed)
    reads = []
    for f in sim.frames[::2]:
        sb = f.scoreboard
        assert sb is not None and sb.game_clock is not None
        disp, per = _display(sb.game_clock), sb.period
        u = rng.random()
        if u < 0.03:
            disp = max(0.0, disp + float(rng.choice([-60.0, -10.0, -1.0, 10.0, 60.0])))
        elif u < 0.04:
            per = (per or 1) + 1
        if 200.0 < f.t < 208.0:
            continue
        reads.append(ScoreboardReading(t=f.t, home_score=sb.home_score, away_score=sb.away_score, period=per,
                                       game_clock=disp, source="ocr", confidence=float(rng.uniform(0.85, 1.0))))
    return reads


@pytest.fixture(scope="module")
def pseudo(sim: SimResult) -> dict[str, object]:
    truth_tl = ClockTimeline.from_exact([f.scoreboard for f in sim.frames if f.scoreboard is not None])
    pbp = pbp_from_events(sim.events, truth_tl, lag_s=1.5)
    reads = _ocr_reads(sim)
    tl = ClockTimeline.fit(reads)
    return {"truth_tl": truth_tl, "pbp": pbp, "tl": tl, "changes": score_changes(reads)}


def _truth_errors(result, sim: SimResult) -> dict[str, list[float]]:  # noqa: ANN001
    by_id = {e.event_id: e for e in sim.events}
    out: dict[str, list[float]] = {}
    for al in result.aligned:
        e = by_id.get(al.action.attrs.get("source_event_id"))
        if e is None or al.t is None:
            continue
        key = ("ft_" if al.action.is_free_throw else "") + al.action.kind.value
        out.setdefault(key, []).append(abs(al.t - e.t))
    return out


def test_pseudo_pbp_from_sim(pseudo: dict, sim: SimResult) -> None:
    pbp: list[PBPAction] = pseudo["pbp"]  # type: ignore[assignment]
    kinds = Counter(a.kind for a in pbp)
    truth = Counter(e.kind for e in sim.events)
    assert kinds[EventKind.SHOT_MADE] == truth[EventKind.SHOT_MADE]
    assert kinds[EventKind.SHOT_MISSED] == truth[EventKind.SHOT_MISSED]
    assert kinds[EventKind.REBOUND] == truth[EventKind.REBOUND]
    assert all(a.attrs.get("source_event_id") for a in pbp)
    assert [a.action_number for a in pbp] == list(range(1, len(pbp) + 1))
    fts = [a for a in pbp if a.is_free_throw]
    assert fts and all(a.attrs["ft_total"] >= a.attrs["ft_index"] >= 1 for a in fts)
    whole = [a.clock for a in pbp if a.clock >= 60.0]
    assert all(c == math.floor(c) for c in whole)  # feed resolution: whole seconds above the last minute
    assists = [a for a in pbp if a.kind is EventKind.ASSIST]
    assert assists and all("parent_action" in a.attrs for a in assists)


def test_alignment_on_pseudo_pbp(pseudo: dict, sim: SimResult) -> None:
    res = align_pbp(pseudo["pbp"], pseudo["tl"], score_changes=pseudo["changes"])  # type: ignore[arg-type]
    assert res.accepted
    assert res.median_made_residual_s is not None and res.median_made_residual_s < 0.5
    assert res.lag_estimated and res.lag_s == pytest.approx(1.5, abs=0.4)
    errs = _truth_errors(res, sim)
    assert np.median(errs["shot_made"]) < 0.5
    assert np.median(errs["ft_shot_made"]) < 0.3
    live = errs["shot_missed"] + errs["rebound"] + errs.get("turnover", [])
    assert np.median(live) < 0.75
    assert all(al.t is not None for al in res.aligned if al.action.kind is not None)


def test_alignment_against_exact_timeline(pseudo: dict, sim: SimResult) -> None:
    changes = score_changes([f.scoreboard for f in sim.frames if f.scoreboard is not None])
    res = align_pbp(pseudo["pbp"], pseudo["truth_tl"], score_changes=changes)  # type: ignore[arg-type]
    assert res.accepted
    assert np.median(_truth_errors(res, sim)["shot_made"]) < 0.5


def test_alignment_snaps_to_engine(pseudo: dict, sim: SimResult) -> None:
    engine = EventEngine(sim.court, sim.roster)
    for f in sim.frames:
        engine.process(f)
    engine.flush()
    res = align_pbp(pseudo["pbp"], pseudo["tl"], engine.events, pseudo["changes"])  # type: ignore[arg-type]
    assert res.accepted
    made = [al for al in res.aligned if al.action.kind is EventKind.SHOT_MADE and al.action.is_field_goal]
    snapped = [al for al in made if al.method == "snap:shot_made"]
    assert len(snapped) >= 0.8 * len(made)
    eng = {e.event_id: e for e in engine.events}
    assert all(al.t == eng[al.anchor_event_id].t for al in snapped)  # type: ignore[index]
    assert all(al.t_release is not None and al.t_release < al.t for al in snapped)  # type: ignore[operator]
    agree = evaluate_against_pbp(engine.events, res)
    assert agree.totals["fgm"]["diff"] == 0 and agree.totals["ftm"]["diff"] == 0
    assert abs(agree.totals["fga"]["diff"]) <= 3


def test_evaluate_truth_agrees_exactly(pseudo: dict, sim: SimResult) -> None:
    res = align_pbp(pseudo["pbp"], pseudo["tl"], score_changes=pseudo["changes"])  # type: ignore[arg-type]
    agree = evaluate_against_pbp(sim.events, res)
    assert agree.agreement == 1.0 and agree.abs_error == 0
    assert agree.periods == sorted({a.action.period for a in res.aligned if a.action.is_shot})
    assert agree.totals["fga"]["pbp"] == sum(1 for e in sim.events if e.kind is EventKind.SHOT_ATTEMPT and e.value in (2, 3))
    assert {r["stat"] for r in agree.rows} == {"fga", "fgm", "fg3a", "fg3m", "fta", "ftm"}


def test_gate_rejects_bad_alignment(pseudo: dict) -> None:
    rng = np.random.default_rng(3)
    noisy = []
    for a in pseudo["pbp"]:  # type: ignore[attr-defined]
        b = PBPAction(**{**a.__dict__, "attrs": dict(a.attrs)})
        if b.is_field_goal:  # a scorer clock that is off by 2-4 s either way
            b.clock = max(0.0, b.clock + float(rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 4.0)))
        noisy.append(b)
    res = align_pbp(noisy, pseudo["tl"], score_changes=pseudo["changes"], config=AlignConfig(score_window_s=6.0))  # type: ignore[arg-type]
    assert not res.accepted
    assert res.median_made_residual_s is not None and res.median_made_residual_s > 0.5


def test_gate_needs_enough_makes(pseudo: dict) -> None:
    res = align_pbp(pseudo["pbp"], pseudo["tl"], score_changes=pseudo["changes"], config=AlignConfig(accept_min_made=10_000))  # type: ignore[arg-type]
    assert not res.accepted


def test_plateau_group_order_and_anchors() -> None:
    # Clock runs 300 -> 280 over t 0-20, stops at 280 for t 20-40 (foul + 2 FTs), then runs again.
    tl = ClockTimeline({1: [(0.0, 300.0), (20.0, 280.0), (40.0, 280.0), (60.0, 260.0)]}, stats={"exact": False})
    acts = [
        PBPAction(1, 1, 280.0, EventKind.FOUL, Team.AWAY, "a1"),
        PBPAction(2, 1, 280.0, EventKind.SHOT_MADE, Team.HOME, "h1", value=1, made=True,
                  attrs={"shot_type": "free_throw", "ft_index": 1, "ft_total": 2}),
        PBPAction(3, 1, 280.0, EventKind.SUBSTITUTION, Team.AWAY, "a2"),
        PBPAction(4, 1, 280.0, EventKind.SHOT_MISSED, Team.HOME, "h1", value=1, made=False,
                  attrs={"shot_type": "free_throw", "ft_index": 2, "ft_total": 2}),
        PBPAction(5, 1, 280.0, EventKind.REBOUND, Team.AWAY, "a3", attrs={"offensive": False}),
    ]
    change = ScoreChange(t=27.0, team=Team.HOME, delta=1, home=1, away=0, period=1, game_clock=280.0, source="ocr", confirmed_t=27.2)
    res = align_pbp(acts, tl, score_changes=[change], config=AlignConfig(estimate_lag=False))
    t = [al.t for al in res.aligned]
    assert t[0] == pytest.approx(20.0) and res.aligned[0].method == "plateau_start"
    assert t[1] == pytest.approx(27.0) and res.aligned[1].method == "score_delta"
    assert t[4] == pytest.approx(40.0) and res.aligned[4].method == "plateau_end"  # clock restarts on the rebound touch
    assert t == sorted(t)
    assert all(al.in_plateau for al in res.aligned)


# =============================================================================================== clips / IO


def test_clips_and_jsonl(pseudo: dict, tmp_path) -> None:  # noqa: ANN001
    res = align_pbp(pseudo["pbp"], pseudo["tl"], score_changes=pseudo["changes"])  # type: ignore[arg-type]
    clips = make_clips(res, game_id="g1", video_id="v1")
    labels = Counter(c.label for c in clips)
    assert labels["fg2_made"] + labels["fg3_made"] > 0 and labels["rebound_def"] > 0 and labels["ft_made"] > 0
    assert all(c.rights_status == "research_only" and c.label_origin == "pbp" for c in clips)
    assert all(c.t_start <= c.t_anchor <= c.t_end for c in clips)
    shots = [c for c in clips if c.label.startswith("fg") and c.t_anchor > 3.0]
    assert all(c.t_end - c.t_anchor == pytest.approx(1.5, abs=1e-3) and c.t_anchor - c.t_start == pytest.approx(3.0, abs=1e-3) for c in shots)
    misses = {round(c.t_anchor, 3) for c in clips if c.label.endswith("_missed")}
    rebounds = [c for c in clips if c.label.startswith("rebound")]
    assert any(c.t_start in misses for c in rebounds)  # rebound clips start at the miss
    assert not any(c.label == "substitution" for c in clips)
    path = tmp_path / "clips.jsonl"
    assert write_jsonl(clips, path) == len(clips)
    back = [LabeledClip.from_dict(d) for d in read_jsonl(path)]
    assert back == clips


def test_generic_csv_round_trip(pseudo: dict, tmp_path) -> None:  # noqa: ANN001
    pbp: list[PBPAction] = pseudo["pbp"]  # type: ignore[assignment]
    path = tmp_path / "pbp.csv"
    assert write_pbp_csv(pbp, path) == len(pbp)
    back = load_pbp(path)
    assert [(a.action_number, a.period, a.clock, a.kind, a.team, a.value, a.made) for a in back] == \
           [(a.action_number, a.period, a.clock, a.kind, a.team, a.value, a.made) for a in pbp]
    assert [a.attrs.get("offensive") for a in back if a.kind is EventKind.REBOUND] == \
           [a.attrs.get("offensive") for a in pbp if a.kind is EventKind.REBOUND]


def test_load_v3_json(tmp_path) -> None:  # noqa: ANN001
    import json

    path = tmp_path / "pbp.json"
    path.write_text(json.dumps({"game": {"actions": V3_ROWS}}))
    acts = load_pbp(path)
    assert sum(a.kind is EventKind.SHOT_MADE for a in acts) == 2


# =============================================================================================== review regressions


def test_parse_v3_unicode_names_in_descriptions() -> None:
    # NBA feeds keep diacritics; an ASCII-only name pattern silently dropped these assists / blocks / steals.
    rows = [
        {"actionNumber": 4, "clock": "PT11M40.00S", "period": 1, "actionType": "Made Shot", "subType": "Jump Shot",
         "shotResult": "Made", "teamTricode": "DEN", "personId": 1, "playerName": "Murray", "location": "h",
         "description": "Murray 25' 3PT Jump Shot (3 PTS) (Jokić 1 AST)"},
        {"actionNumber": 6, "clock": "PT11M20.00S", "period": 1, "actionType": "Missed Shot", "subType": "Layup",
         "shotResult": "Missed", "teamTricode": "DAL", "personId": 2, "playerName": "Irving", "location": "v",
         "description": "MISS Irving 2' Layup Nurkić BLOCK (1 BLK)"},
        {"actionNumber": 8, "clock": "PT11M00.00S", "period": 1, "actionType": "Turnover", "subType": "Bad Pass",
         "teamTricode": "DEN", "personId": 1, "playerName": "Murray", "location": "h",
         "description": "Murray Bad Pass Turnover (P1.T1) Dončić STEAL (1 STL)"},
        {"actionNumber": 9, "clock": "PT10M40.00S", "period": 1, "actionType": "Made Shot", "subType": "Jump Shot",
         "shotResult": "Made", "teamTricode": "DAL", "personId": 3, "playerName": "Irving", "location": "v",
         "description": "Irving 15' Jump Shot (2 PTS) (Jalen Williams Jr. 2 AST)"},
    ]
    derived = {(a.kind, a.player_name) for a in parse_pbp_v3(rows) if a.attrs.get("derived")}
    assert derived == {(EventKind.ASSIST, "Jokić"), (EventKind.BLOCK, "Nurkić"), (EventKind.STEAL, "Dončić"),
                       (EventKind.ASSIST, "Jalen Williams Jr.")}


def test_foul_after_leading_live_row_goes_to_the_stop() -> None:
    # A miss whose rebound is fouled: the miss is recorded at the stopped clock, and the foul is the whistle
    # that stopped it. It must not be spaced towards the first FT anchor (was 6.5 s late).
    tl = ClockTimeline({1: [(0.0, 300.0), (20.0, 280.0), (40.0, 280.0), (60.0, 260.0)]})
    ft = {"shot_type": "free_throw", "ft_total": 2}
    acts = [
        PBPAction(1, 1, 280.0, EventKind.SHOT_MISSED, Team.HOME, "h2", value=2, made=False),
        PBPAction(2, 1, 280.0, EventKind.FOUL, Team.HOME, "h3"),
        PBPAction(3, 1, 280.0, EventKind.SHOT_MADE, Team.AWAY, "a1", value=1, made=True, attrs={**ft, "ft_index": 1}),
        PBPAction(4, 1, 280.0, EventKind.SHOT_MADE, Team.AWAY, "a1", value=1, made=True, attrs={**ft, "ft_index": 2}),
        PBPAction(5, 1, 280.0, EventKind.TIMEOUT, Team.HOME),
    ]
    changes = [ScoreChange(t=t, team=Team.AWAY, delta=1, home=0, away=k + 1, period=1, game_clock=280.0, source="ocr",
                           confirmed_t=t + 0.2) for k, t in enumerate((33.0, 37.0))]
    res = align_pbp(acts, tl, score_changes=changes, config=AlignConfig(estimate_lag=False))
    t = [al.t for al in res.aligned]
    assert t[0] == pytest.approx(20.0) and t[1] == pytest.approx(20.0) and res.aligned[1].method == "plateau_start"
    assert t[2:4] == [pytest.approx(33.0), pytest.approx(37.0)]
    assert 37.0 < t[4] < 40.0 and t == sorted(t)  # a later timeout is spaced by order, not pulled to the stop


def test_timeline_drops_misread_at_period_start() -> None:
    # The monotone filter keeps a high misread on the first reads of a period (it breaks no monotonicity,
    # and wins the longest-subsequence tie); the fit then had a 300 -> 240 "running" clock and the
    # period-start stoppage began 1.4 s late.
    reads = [ScoreboardReading(t=0.0, period=2, game_clock=240.0, source="ocr"),
             ScoreboardReading(t=0.2, period=2, game_clock=300.0, source="ocr")]
    reads += [ScoreboardReading(t=round(0.4 + 0.2 * k, 3), period=2, game_clock=240.0, source="ocr") for k in range(20)]
    reads += [ScoreboardReading(t=round(4.4 + 0.2 * k, 3), period=2, game_clock=float(math.floor(239.999 - 0.2 * k)),
                                source="ocr") for k in range(50)]
    tl = ClockTimeline.fit(reads)
    assert max(c for _, c in tl.knots(2)) == 240.0
    assert tl.stats["dropped_rate"] == 1
    iv = tl.stop_interval(2, 240.0)
    assert iv is not None and iv[0] <= 0.4 + 1e-9
    cp = tl.clock_at(0.3)
    assert cp is not None and cp.clock == pytest.approx(240.0) and not cp.running
    # a low misread on the last read of a period is trimmed the same way
    tail = reads[:60] + [ScoreboardReading(t=16.4, period=2, game_clock=150.0, source="ocr")]
    assert min(c for _, c in ClockTimeline.fit(tail).knots(2)) > 200.0


def test_timeline_keeps_real_edges() -> None:
    # a real period-start plateau and the last running reads are never trimmed
    reads = [ScoreboardReading(t=0.2 * k, period=1, game_clock=720.0, source="ocr") for k in range(10)]
    reads += [ScoreboardReading(t=2.0 + 0.2 * k, period=1, game_clock=float(math.floor(719.999 - 0.2 * k)), source="ocr")
              for k in range(30)]
    tl = ClockTimeline.fit(reads)
    assert tl.stats["dropped_rate"] == 0 and tl.knots(1)[0] == (0.0, 720.0)


def test_format_clock_rounds_into_the_next_minute() -> None:
    assert format_clock(59.999) == "PT01M00.00S"
    assert format_clock(119.994) == "PT01M59.99S"
    assert parse_clock(format_clock(59.999)) == pytest.approx(60.0)


def test_clips_skip_actions_before_the_video_start() -> None:
    from courtvision.training.align_pbp import AlignedAction

    shot = PBPAction(1, 1, 700.0, EventKind.SHOT_MADE, Team.HOME, "h1", value=2, made=True)
    later = PBPAction(2, 1, 690.0, EventKind.SHOT_MISSED, Team.AWAY, "a1", value=2, made=False)
    clips = make_clips([AlignedAction(shot, -0.5, -0.5, "clock", 0.7), AlignedAction(later, 9.0, 9.0, "clock", 0.7)], "g", "v")
    assert [c.action_number for c in clips] == [2]
    assert all(0.0 <= c.t_start <= c.t_anchor <= c.t_end for c in clips)
