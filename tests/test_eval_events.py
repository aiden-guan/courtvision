"""Tests for courtvision.eval.events (synthetic, deterministic)."""

from __future__ import annotations

import copy
import itertools
import json
import random

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from courtvision.eval.events import (
    EventEvalConfig,
    gated_assignment,
    latest_revisions,
    match_events,
    match_times,
)
from courtvision.types import Event, EventKind, EventStatus, Team

K = EventKind
_ids = itertools.count(1)


def ev(kind: EventKind, t: float, player: str | None = "home-1", **kw) -> Event:
    kw.setdefault("event_id", f"e{next(_ids):06d}")
    kw.setdefault("frame_idx", int(round(t * 30)))
    kw.setdefault("team", Team.HOME if player and player.startswith("home") else Team.AWAY)
    return Event(kind=kind, t=t, player_id=player, **kw)


def shot(t: float, shooter: str, made: bool, value: int = 2, sid: str = "s0001", assister: str | None = None,
         t_res: float | None = None) -> list[Event]:
    """Attempt + result (+ assist) for one shot, linked by shot_id."""
    t_res = t + 1.0 if t_res is None else t_res
    out = [ev(K.SHOT_ATTEMPT, t, shooter, value=value, shot_id=sid,
              attrs={"shot_type": "free_throw" if value == 1 else "jumper"})]
    out.append(ev(K.SHOT_MADE if made else K.SHOT_MISSED, t_res, shooter, shot_id=sid, value=value if made else 0))
    if assister is not None:
        out.append(ev(K.ASSIST, t_res, assister, other_player_id=shooter, shot_id=sid))
    return out


def game() -> list[Event]:
    """A small truth game covering the core kinds."""
    events: list[Event] = []
    events += [ev(K.PASS, 1.0, "home-1", other_player_id="home-2", attrs={"completed": True})]
    events += shot(3.0, "home-2", True, 2, "s0001", assister="home-1")
    events += shot(10.0, "away-5", False, 3, "s0002")
    events += [ev(K.REBOUND, 12.0, "home-3", attrs={"offensive": False})]
    events += shot(20.0, "home-3", True, 3, "s0003")
    events += [ev(K.STEAL, 30.0, "away-6", other_player_id="home-4"),
               ev(K.TURNOVER, 30.0, "home-4", attrs={"type": "steal"})]
    events += shot(40.0, "away-7", False, 2, "s0004")
    events += [ev(K.BLOCK, 40.2, "home-5", other_player_id="away-7", shot_id="s0004"),
               ev(K.REBOUND, 42.0, None, team=Team.HOME, attrs={"offensive": False})]
    events += shot(50.0, "away-8", True, 1, "s0005")  # free throw
    events += [ev(K.DRIBBLE, 60.0 + 0.5 * i, "home-1") for i in range(6)]
    return events


def as_pred(events: list[Event], dt: float = 0.0) -> list[Event]:
    """Independent copies with their own ids (pred ids never equal truth ids in practice)."""
    out = []
    for e in events:
        c = copy.deepcopy(e)
        c.event_id = "p" + e.event_id
        c.t = e.t + dt
        out.append(c)
    return out


# --------------------------------------------------------------------------- perfect / shifts


def test_perfect_match() -> None:
    truth = game()
    rep = match_events(as_pred(truth), truth)
    assert rep.headline_macro_f1 == pytest.approx(1.0)
    assert rep.headline_attributed_f1 == pytest.approx(1.0)
    assert set(rep.headline_kinds) == {"pass", "shot_attempt", "shot_made", "shot_missed", "rebound",
                                       "assist", "steal", "turnover", "block"}
    for kr in rep.per_kind.values():
        assert (kr.precision, kr.recall, kr.f1) == (1.0, 1.0, 1.0)
        assert kr.player_accuracy == 1.0 and kr.mean_abs_dt == 0.0
    assert rep.per_kind["rebound"].attr_accuracy["offensive"] == 1.0
    s = rep.shots
    assert s.n_paired == s.n_truth_shots == 5
    assert s.fg_outcome_accuracy == 1.0 and s.fg_outcome_n == 4
    assert s.ft_outcome_accuracy == 1.0 and s.ft_outcome_n == 1
    assert (s.made_fg_precision, s.made_fg_recall) == (1.0, 1.0)
    assert s.n_truth_made_fg == 2  # the made free throw is not a field goal
    assert s.value_accuracy == 1.0 and s.value_n == 4
    assert s.assist_accuracy == 1.0 and s.assist_n == 2  # assisted + unassisted made FGs
    json.dumps(rep.to_dict())


def test_shift_within_and_beyond_tolerance() -> None:
    truth = game()
    within = match_events(as_pred(truth, dt=0.2), truth)
    assert within.headline_macro_f1 == pytest.approx(1.0)
    assert within.per_kind["pass"].mean_abs_dt == pytest.approx(0.2)
    assert within.per_kind["pass"].mean_dt == pytest.approx(0.2)

    beyond = match_events(as_pred(truth, dt=1.5), truth)
    assert beyond.headline_macro_f1 == 0.0
    for kr in beyond.per_kind.values():
        assert kr.tp == 0 and kr.precision == 0.0 and kr.recall == 0.0 and kr.f1 == 0.0
        assert kr.mean_abs_dt is None
    assert beyond.shots.n_paired == 0
    assert beyond.shots.made_fg_precision == 0.0 and beyond.shots.made_fg_recall == 0.0


def test_tolerance_boundary_is_inclusive() -> None:
    truth = [ev(K.PASS, 5.0, "home-1")]
    assert match_events([ev(K.PASS, 6.0, "home-1")], truth).per_kind["pass"].tp == 1
    assert match_events([ev(K.PASS, 6.01, "home-1")], truth).per_kind["pass"].tp == 0
    assert match_events([ev(K.PASS, 6.4, "home-1")], truth, tol_s=1.5).per_kind["pass"].tp == 1


def test_inclusive_boundary_survives_frame_quantised_timestamps() -> None:
    # t = frame_idx / 30: an exact 30-frame (1.0 s) offset can be 1.0000000000000002 in
    # floating point (e.g. 33/30 - 3/30). The inclusive bound must still accept it.
    for a in range(0, 300):
        truth = [ev(K.PASS, a / 30, "home-1")]
        pred = [ev(K.PASS, (a + 30) / 30, "home-1")]
        assert match_events(pred, truth).per_kind["pass"].tp == 1, a
        half = [ev(K.PASS, (a + 15) / 30, "home-1")]
        assert match_events(half, truth, tol_s=0.5).per_kind["pass"].tp == 1, a
    # ...and in the time-blocked matcher with neighbours on both sides of the gap.
    assert match_times([33 / 30, 40.0], [3 / 30, 39.0], 1.0) == [(0, 0), (1, 1)]


def test_infinite_tolerance_with_player_keys() -> None:
    # tol_s=inf used to make the player tie-break weight inf and the cost matrix NaN.
    truth = [ev(K.PASS, 50.0, "home-1"), ev(K.PASS, 60.0, "home-2")]
    pred = [ev(K.PASS, 1.0, "home-2")]
    kr = match_events(pred, truth, tol_s=float("inf")).per_kind["pass"]
    assert kr.tp == 1 and kr.player_accuracy == 1.0 and kr.mean_abs_dt == pytest.approx(59.0)


# --------------------------------------------------------------------------- players


def test_wrong_player_counts_for_detection_not_attribution() -> None:
    truth = [ev(K.REBOUND, 5.0, "home-3"), ev(K.REBOUND, 15.0, "away-2")]
    pred = [ev(K.REBOUND, 5.1, "home-4"), ev(K.REBOUND, 15.0, "away-2")]
    rep = match_events(pred, truth)
    kr = rep.per_kind["rebound"]
    assert (kr.tp, kr.fp, kr.fn) == (2, 0, 0)
    assert kr.f1 == 1.0
    assert kr.player_accuracy == 0.5
    assert kr.attributed_f1 == pytest.approx(0.5)
    assert kr.wrong_player == [(pred[0].event_id, truth[0].event_id)]
    assert rep.headline_macro_f1 == 1.0 and rep.headline_attributed_f1 == pytest.approx(0.5)


def test_assignment_prefers_correct_player_over_smaller_dt() -> None:
    truth = [ev(K.PASS, 10.0, "home-1"), ev(K.PASS, 10.6, "home-2")]
    # Time alone pairs the crossed predictions (total |dt| 0); the player-aware
    # assignment accepts |dt| 0.6 each to get both actors right.
    pred = [ev(K.PASS, 10.6, "home-1"), ev(K.PASS, 10.0, "home-2")]
    rep = match_events(pred, truth)
    assert rep.per_kind["pass"].player_accuracy == 1.0
    assert rep.per_kind["pass"].mean_abs_dt == pytest.approx(0.6)
    no_player = match_events(pred, truth, match_player=False)
    kr = no_player.per_kind["pass"]
    assert kr.tp == 2 and kr.player_accuracy is None and kr.attributed_f1 is None
    assert no_player.headline_attributed_f1 is None


def test_team_rebound_is_attributed_to_the_team_and_other_player_scored() -> None:
    # Team rebounds / team turnovers are first-class team entities (rules note §0.5):
    # a team rebound credited to the wrong team is a wrong attribution, not None == None.
    truth = [ev(K.REBOUND, 5.0, None, team=Team.AWAY), ev(K.STEAL, 9.0, "away-1", other_player_id="home-2")]
    pred = [ev(K.REBOUND, 5.2, None, team=Team.HOME), ev(K.STEAL, 9.0, "away-1", other_player_id="home-3")]
    rep = match_events(pred, truth)
    reb = rep.per_kind["rebound"]
    assert reb.tp == 1 and reb.player_accuracy == 0.0 and reb.attributed_f1 == 0.0
    assert reb.team_accuracy == 0.0
    assert rep.per_kind["steal"].player_accuracy == 1.0
    assert rep.per_kind["steal"].other_player_accuracy == 0.0
    right_team = match_events([ev(K.REBOUND, 5.2, None, team=Team.AWAY)], truth[:1]).per_kind["rebound"]
    assert right_team.player_accuracy == 1.0 and right_team.attributed_f1 == 1.0
    named = match_events([ev(K.REBOUND, 5.2, "away-4", team=Team.AWAY)], truth[:1]).per_kind["rebound"]
    assert named.player_accuracy == 0.0  # a player credited with a team rebound is wrong


def test_team_turnover_pairing_prefers_the_right_team() -> None:
    # Two simultaneous-ish team turnovers (shot-clock style, player None): actor = team.
    truth = [ev(K.TURNOVER, 10.0, None, team=Team.HOME), ev(K.TURNOVER, 10.6, None, team=Team.AWAY)]
    pred = [ev(K.TURNOVER, 10.6, None, team=Team.HOME), ev(K.TURNOVER, 10.0, None, team=Team.AWAY)]
    kr = match_events(pred, truth).per_kind["turnover"]
    assert kr.tp == 2 and kr.player_accuracy == 1.0 and kr.mean_abs_dt == pytest.approx(0.6)


def test_attributed_f1_uses_its_own_same_actor_matching() -> None:
    # Detection takes 2 pairs (both wrong actor); a same-actor matching still finds B-B.
    truth = [ev(K.PASS, 10.0, "home-1"), ev(K.PASS, 11.0, "home-2")]
    pred = [ev(K.PASS, 10.5, "home-2"), ev(K.PASS, 11.8, "home-9")]
    kr = match_events(pred, truth).per_kind["pass"]
    assert kr.tp == 2 and kr.player_accuracy == 0.0
    assert kr.attributed_precision == 0.5 and kr.attributed_recall == 0.5
    assert kr.attributed_f1 == pytest.approx(0.5)
    assert kr.attributed_f1 <= kr.f1


# --------------------------------------------------------------------------- revisions


def test_revisions_latest_wins_and_retractions_dropped() -> None:
    truth = [ev(K.SHOT_ATTEMPT, 5.0, "home-1", shot_id="s1", value=2),
             ev(K.SHOT_MISSED, 6.0, "home-1", shot_id="s1")]
    base = ev(K.SHOT_ATTEMPT, 5.0, "home-9", shot_id="s1", value=2, event_id="e1")
    fixed = copy.deepcopy(base)
    fixed.revision, fixed.player_id = 1, "home-1"
    made = ev(K.SHOT_MADE, 6.0, "home-1", shot_id="s1", value=2, event_id="e2")
    made_retracted = copy.deepcopy(made)
    made_retracted.revision, made_retracted.status = 1, EventStatus.RETRACTED
    missed = ev(K.SHOT_MISSED, 6.0, "home-1", shot_id="s1", event_id="e3")
    # Out of order on purpose: the higher revision must win wherever it appears.
    pred = [fixed, made_retracted, base, made, missed]
    rep = match_events(pred, truth)
    assert rep.per_kind["shot_attempt"].player_accuracy == 1.0
    assert "shot_made" not in rep.per_kind  # retracted, and truth has none
    assert rep.per_kind["shot_missed"].f1 == 1.0
    assert rep.shots.fg_outcome_accuracy == 1.0
    counts = rep.input_counts["pred"]
    assert counts == {"input": 5, "unique_ids": 3, "retracted": 1, "revised_ids": 2, "kept": 2}


def test_retraction_can_be_reversed_by_a_later_revision() -> None:
    a = ev(K.STEAL, 3.0, "away-1", event_id="e9")
    r1 = copy.deepcopy(a)
    r1.revision, r1.status = 1, EventStatus.RETRACTED
    r2 = copy.deepcopy(a)
    r2.revision, r2.status = 2, EventStatus.CONFIRMED
    assert [e.revision for e in latest_revisions([a, r1, r2])] == [2]
    assert latest_revisions([a, r1]) == []


def test_dict_inputs_are_accepted() -> None:
    truth = game()
    rep = match_events([e.to_dict() for e in as_pred(truth)], [e.to_dict() for e in truth])
    assert rep.headline_macro_f1 == pytest.approx(1.0)


def test_dict_input_with_null_attrs() -> None:
    truth = [ev(K.REBOUND, 5.0, "home-1", attrs={"offensive": True})]
    d = ev(K.REBOUND, 5.0, "home-1").to_dict()
    d["attrs"] = None  # e.g. JSON from a manual/feed source
    kr = match_events([d], truth).per_kind["rebound"]
    assert kr.tp == 1 and kr.attr_accuracy == {"offensive": 0.0}


def test_single_kind_argument_is_not_split_into_characters() -> None:
    truth = [ev(K.PASS, 5.0, "home-1"), ev(K.STEAL, 7.0, "away-1")]
    pred = [ev(K.PASS, 5.0, "home-1")]
    for kinds in ("pass", K.PASS, ["pass"]):
        assert set(match_events(pred, truth, kinds=kinds).per_kind) == {"pass"}


# --------------------------------------------------------------------------- missing / extra


def test_missing_and_extra_events() -> None:
    truth = [ev(K.TURNOVER, float(t), "home-1", attrs={"type": "bad_pass"}) for t in (10, 20, 30, 40)]
    pred = [ev(K.TURNOVER, 10.1, "home-1", attrs={"type": "bad_pass"}),
            ev(K.TURNOVER, 20.3, "home-1", attrs={"type": "travel"}),
            ev(K.TURNOVER, 55.0, "home-1"),
            ev(K.TURNOVER, 70.0, "home-1")]
    kr = match_events(pred, truth).per_kind["turnover"]
    assert (kr.tp, kr.fp, kr.fn) == (2, 2, 2)
    assert kr.precision == 0.5 and kr.recall == 0.5 and kr.f1 == 0.5
    assert kr.fp_ids == [pred[2].event_id, pred[3].event_id]
    assert kr.fn_ids == [truth[2].event_id, truth[3].event_id]
    assert kr.attr_accuracy["type"] == 0.5
    assert kr.mean_abs_dt == pytest.approx(0.2)


def test_kind_only_on_one_side_scores_zero_and_kinds_filter() -> None:
    truth = [ev(K.BLOCK, 5.0, "home-1"), ev(K.PASS, 8.0, "home-1")]
    pred = [ev(K.STEAL, 5.0, "home-1"), ev(K.PASS, 8.0, "home-1"),
            ev(K.SCORE_CORRECTION, 9.0, None)]
    rep = match_events(pred, truth)
    assert rep.per_kind["block"].f1 == 0.0 and rep.per_kind["block"].precision is None
    assert rep.per_kind["steal"].f1 == 0.0 and rep.per_kind["steal"].recall is None
    assert "score_correction" not in rep.per_kind
    assert rep.headline_macro_f1 == pytest.approx(1.0 / 3.0)
    only = match_events(pred, truth, kinds=["pass", K.TURNOVER])
    assert set(only.per_kind) == {"pass", "turnover"}
    assert only.per_kind["turnover"].f1 is None
    assert only.headline_kinds == ["pass"] and only.headline_macro_f1 == 1.0


def test_dribble_reported_separately_with_its_own_tolerance() -> None:
    truth = [ev(K.DRIBBLE, 1.0 + 0.5 * i, "home-1") for i in range(10)]
    pred = [ev(K.DRIBBLE, 1.0 + 0.5 * i + 0.1, "home-1") for i in range(10)]
    pred.append(ev(K.DRIBBLE, 2.25, "home-1"))  # spurious bounce between two true ones
    rep = match_events(pred, truth + [ev(K.PASS, 9.0, "home-1")])
    assert "dribble" not in rep.per_kind and rep.dribble is not None
    assert rep.dribble.tp == 10 and rep.dribble.fp == 1 and rep.dribble.tol_s == 0.25
    assert rep.headline_kinds == ["pass"]
    loose = match_events(pred, truth, config=EventEvalConfig(kind_tol_s={}))
    assert loose.dribble is not None and loose.dribble.tol_s == 1.0


# --------------------------------------------------------------------------- shots


def test_make_miss_confusion() -> None:
    truth = (shot(5.0, "home-1", True, 2, "s1") + shot(15.0, "home-2", False, 2, "s2")
             + shot(25.0, "home-3", True, 3, "s3"))
    pred = (shot(5.0, "home-1", False, 2, "p1") + shot(15.0, "home-2", True, 2, "p2")
            + shot(25.0, "home-3", True, 3, "p3"))
    rep = match_events(pred, truth)
    s = rep.shots
    assert s.fg_outcome_accuracy == pytest.approx(1 / 3)
    assert s.outcome_confusion == {"made": {"missed": 1, "made": 1}, "missed": {"made": 1}}
    assert s.made_fg_tp == 1 and s.n_pred_made_fg == 2 and s.n_truth_made_fg == 2
    assert s.made_fg_precision == 0.5 and s.made_fg_recall == 0.5
    assert rep.per_kind["shot_made"].f1 == 0.5
    assert rep.per_kind["shot_attempt"].f1 == 1.0


def test_unresolved_shot_and_ft_not_counted_as_made_fg() -> None:
    truth = shot(5.0, "home-1", True, 2, "s1") + shot(9.0, "home-1", True, 1, "s2")
    pred = [ev(K.SHOT_ATTEMPT, 5.0, "home-1", value=2, shot_id="p1")] + shot(9.0, "home-1", True, 1, "p2")
    s = match_events(pred, truth).shots
    assert s.outcome_confusion == {"made": {"unresolved": 1}}
    assert s.fg_outcome_accuracy == 0.0 and s.ft_outcome_accuracy == 1.0
    assert s.n_pred_made_fg == 0 and s.made_fg_precision is None and s.made_fg_recall == 0.0


def test_two_vs_three_value_and_assist_attribution() -> None:
    truth = (shot(5.0, "home-1", True, 3, "s1", assister="home-2")
             + shot(15.0, "home-3", True, 2, "s2", assister="home-4")
             + shot(25.0, "home-5", True, 2, "s3"))
    pred = (shot(5.0, "home-1", True, 2, "p1", assister="home-2")  # value wrong
            + shot(15.0, "home-3", True, 2, "p2", assister="home-1")  # wrong assister
            + shot(25.0, "home-5", True, 2, "p3", assister="home-4"))  # assist invented
    s = match_events(pred, truth).shots
    assert s.value_accuracy == pytest.approx(2 / 3)
    assert s.value_confusion == {"3": {"2": 1}, "2": {"2": 2}}
    assert s.assist_accuracy == pytest.approx(1 / 3) and s.assist_n == 3


def test_shot_paired_by_result_when_attempt_missing() -> None:
    truth = shot(5.0, "home-1", True, 2, "s1", t_res=6.0)
    pred = [ev(K.SHOT_MADE, 6.1, "home-1", value=2, shot_id="p9")]  # attempt never detected
    s = match_events(pred, truth).shots
    assert s.n_paired == 1 and s.n_paired_by_result == 1
    assert s.fg_outcome_accuracy == 1.0 and s.made_fg_precision == 1.0 and s.value_accuracy == 1.0


def test_shot_ids_are_namespaced_per_side() -> None:
    # Same shot_id strings on both sides refer to different shots: linking is per side.
    truth = shot(5.0, "home-1", True, 2, "s1") + shot(20.0, "home-2", False, 2, "s2")
    pred = shot(5.0, "home-1", True, 2, "s2") + shot(20.0, "home-2", False, 2, "s1")
    s = match_events(pred, truth).shots
    assert s.fg_outcome_accuracy == 1.0 and s.n_paired == 2


# --------------------------------------------------------------------------- assignment core


def test_gated_assignment_maximises_cardinality_first() -> None:
    cost = np.array([[0.1, 0.9], [0.2, 5.0]])
    ok = np.array([[True, True], [True, False]])
    # Min-cost alone would choose (0,0)+(1,1); (1,1) is infeasible, so take (0,1)+(1,0).
    assert sorted(gated_assignment(cost, ok)) == [(0, 1), (1, 0)]
    assert gated_assignment(np.zeros((0, 3)), np.zeros((0, 3), bool)) == []


def test_match_times_blocks_agree_with_full_matrix() -> None:
    rng = random.Random(7)
    for _ in range(25):
        pt = sorted(rng.uniform(0, 30) for _ in range(rng.randint(0, 25)))
        tt = sorted(rng.uniform(0, 30) for _ in range(rng.randint(0, 25)))
        pairs = match_times(pt, tt, 1.0)
        assert len({i for i, _ in pairs}) == len(pairs) == len({j for _, j in pairs})
        assert all(abs(pt[i] - tt[j]) <= 1.0 for i, j in pairs)
        if pt and tt:
            d = np.abs(np.subtract.outer(pt, tt))
            big = 1e6
            r, c = linear_sum_assignment(np.where(d <= 1.0, d, big))
            ref = [(i, j) for i, j in zip(r, c) if d[i, j] <= 1.0]
            assert len(pairs) == len(ref)
            assert sum(d[i, j] for i, j in pairs) == pytest.approx(sum(d[i, j] for i, j in ref))


def _brute_lexicographic(pt, tt, tol, pk, tk) -> tuple[int, int, float]:
    """Exhaustive (most pairs, most equal keys, least total |dt|) over all matchings."""
    best: tuple[int, int, float] | None = None

    def rec(i: int, used: set[int], pairs: list[tuple[int, int]]) -> None:
        nonlocal best
        if i == len(pt):
            key = (-len(pairs), -sum(pk[a] == tk[b] for a, b in pairs), sum(abs(pt[a] - tt[b]) for a, b in pairs))
            if best is None or key < best:
                best = key
            return
        rec(i + 1, used, pairs)
        for j in range(len(tt)):
            if j not in used and abs(pt[i] - tt[j]) <= tol + 1e-9:
                rec(i + 1, used | {j}, pairs + [(i, j)])

    rec(0, set(), [])
    assert best is not None
    return best


def test_match_times_with_keys_is_lexicographically_optimal() -> None:
    rng = random.Random(3)
    for _ in range(150):
        pt = [round(rng.uniform(0, 5), 1) for _ in range(rng.randint(0, 5))]
        tt = [round(rng.uniform(0, 5), 1) for _ in range(rng.randint(0, 5))]
        pk = [rng.choice("abc") for _ in pt]
        tk = [rng.choice("abc") for _ in tt]
        pairs = match_times(pt, tt, 1.0, pk, tk)
        got = (-len(pairs), -sum(pk[a] == tk[b] for a, b in pairs), sum(abs(pt[a] - tt[b]) for a, b in pairs))
        ref = _brute_lexicographic(pt, tt, 1.0, pk, tk)
        assert got[:2] == ref[:2] and got[2] == pytest.approx(ref[2])


def test_orphan_assist_is_not_a_shot_and_explicit_dribble_kind() -> None:
    truth = shot(5.0, "home-1", True, 2, "s1", assister="home-2")
    pred = shot(5.0, "home-1", True, 2, "p1", assister="home-2")
    pred.append(ev(K.ASSIST, 30.0, "home-2", other_player_id="home-1", shot_id="p77"))
    rep = match_events(pred, truth)
    assert rep.shots.n_pred_shots == 1 and rep.shots.assist_accuracy == 1.0
    assert rep.per_kind["assist"].fp == 1
    only = match_events(pred, truth, kinds=["dribble"])
    assert only.per_kind == {} and only.headline_macro_f1 is None and only.dribble is not None


# --------------------------------------------------------------------------- stream boundaries / operator lag


def test_cut_stream_end_does_not_score_unresolvable_events() -> None:
    """A stream cut mid-flight: the engine's flush-time SHOT_MISSED for a shot still in the air, and a truth
    attempt too close to the end to be seen, are left out instead of scoring FP / FN; matched pairs count."""
    end = 300.0
    truth = [*shot(100.0, "home-2", True, sid="s1"), ev(K.SHOT_ATTEMPT, 299.93, "away-5", value=2, shot_id="s2")]
    pred = [*shot(100.1, "home-2", True, sid="p1"),
            ev(K.SHOT_MISSED, 299.97, "home-4", shot_id="p9", attrs={"reason": "unresolved at end of tracking"}),
            ev(K.PASS, 299.2, "home-4")]
    plain = match_events(pred, truth)
    assert plain.per_kind["shot_missed"].fp == 1 and plain.per_kind["shot_attempt"].fn == 1
    assert plain.per_kind["pass"].fp == 1
    cut = match_events(pred, truth, window=(None, end))
    assert cut.per_kind["shot_missed"].n_pred == 0 and cut.per_kind["shot_missed"].f1 is None
    att = cut.per_kind["shot_attempt"]
    assert (att.tp, att.fp, att.fn) == (1, 0, 0)  # the near-end FN falls in the guard band
    assert cut.per_kind["pass"].n_pred == 0
    assert cut.boundary_excluded == {"shot_attempt": {"pred": 0, "truth": 1}, "shot_missed": {"pred": 1, "truth": 0},
                                     "pass": {"pred": 1, "truth": 0}}
    assert cut.headline_macro_f1 == pytest.approx(1.0)
    # a pair that matches inside the band always counts; events outside the band are untouched
    late = match_events([ev(K.PASS, 299.6, "home-1")], [ev(K.PASS, 299.9, "home-1")], window=(None, end))
    assert late.per_kind["pass"].tp == 1
    early = match_events([ev(K.PASS, 150.0, "home-1")], [ev(K.PASS, 160.0, "home-1")], window=(None, end))
    assert (early.per_kind["pass"].fp, early.per_kind["pass"].fn) == (1, 1)
    # a window that starts mid-game guards its start edge too
    start = match_events([ev(K.PASS, 30.4, "home-1")], [], window=(30.0, end))
    assert start.per_kind["pass"].n_pred == 0
    narrow = match_events([ev(K.PASS, 299.5, "home-1")], [], window=(None, end),
                          config=EventEvalConfig(boundary_guard_s=0.2))
    assert narrow.per_kind["pass"].fp == 1  # outside a 0.2 s guard


def test_operator_entered_kinds_match_with_the_operator_lag() -> None:
    """SUBSTITUTION truth is the entry frame; the operator enters it entry_delay_s later (2.0 s): matched
    within tolerance + lag, attribution still by player."""
    truth = [ev(K.SUBSTITUTION, 263.17, "home-9", other_player_id="home-3"),
             ev(K.SUBSTITUTION, 264.53, "home-7", other_player_id="home-4")]
    pred = [ev(K.SUBSTITUTION, 265.17, "home-9", other_player_id="home-3", source="manual"),
            ev(K.SUBSTITUTION, 266.53, "home-7", other_player_id="home-4", source="manual")]
    plain = match_events(pred, truth)
    assert plain.per_kind["substitution"].tp == 1  # only the cross pairing 265.17 ~ 264.53 fits +-1 s
    lagged = match_events(pred, truth, config=EventEvalConfig(operator_lag_s={"substitution": 2.0}))
    sub = lagged.per_kind["substitution"]
    assert sub.tp == 2 and sub.player_accuracy == 1.0 and sub.tol_s == pytest.approx(3.0)
    assert lagged.per_kind["substitution"].f1 == 1.0
