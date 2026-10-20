"""Decoders: usable(), order-ensembling, thresholds and the deterministic fallback tables (design §7.4)."""

from __future__ import annotations

import pytest

from courtvision.decision import questions as Q
from courtvision.decision import states as S
from courtvision.decision.client import JevClient, JevConfig, JevResult, StubAnswers, stub_choice, stub_noul, stub_score
from courtvision.decision.decoders import (
    FallbackRules,
    Thresholds,
    decide_release_intent,
    decide_shot_outcome,
    decode_cards,
    decode_guard,
    decode_play_type,
    decode_stakes,
    ensembled_choice,
    fallback_highlight,
    fallback_play_type,
    fallback_stakes,
    stakes_level,
    usable,
)

TH = Thresholds()
B5_STATE = {"shot": {"view_quality": "clear view of the rim"}}


def answered(overrides: dict, questions: dict, model: str | None = None) -> JevResult:
    client = JevClient(JevConfig(mode="stub"), stub=StubAnswers(overrides=overrides, model=model))
    r = client.ask("B1" if "play_type" in questions else "B5", {"x": "y"}, questions)
    client.close()
    return r


def shot(fwd: dict[str, float], rev: dict[str, float] | None = None, model: str | None = None) -> JevResult:
    rev = fwd if rev is None else rev
    return answered({"shot_outcome": stub_choice(fwd), "shot_outcome_rev": stub_choice(dict(reversed(list(rev.items()))))},
                    Q.B5_SHOT_OUTCOME, model)


def test_usable_requires_ok_answer_from_the_pinned_model() -> None:
    assert usable(shot({"make": 0.9, "miss": 0.05, "unclear": 0.05}))
    assert not usable(shot({"make": 0.9, "miss": 0.05, "unclear": 0.05}, model="jev-1.14.0"))
    assert not usable(JevResult(bundle="B5", ok=False, source="fallback", error="deadline"))
    assert usable(shot({"make": 0.9, "miss": 0.05, "unclear": 0.05}, model="jev-1.14.0"), pinned="jev-1.14.0")


def test_ensembled_choice_averages_both_orders() -> None:
    r = shot({"make": 0.9, "miss": 0.06, "unclear": 0.04}, {"make": 0.8, "miss": 0.1, "unclear": 0.1})
    ens = ensembled_choice(r.response, "shot_outcome", "shot_outcome_rev")
    assert ens.agree and ens.label == "make" and ens.p_max == pytest.approx(0.85)
    assert ens.probabilities["miss"] == pytest.approx(0.08)
    assert ens.confidence == pytest.approx((0.85 - 1 / 3) / (2 / 3))


def test_ensembled_choice_returns_unclear_when_orders_disagree() -> None:
    r = shot({"make": 0.6, "miss": 0.3, "unclear": 0.1}, {"make": 0.3, "miss": 0.6, "unclear": 0.1})
    ens = ensembled_choice(r.response, "shot_outcome", "shot_outcome_rev")
    assert not ens.agree and ens.label == "unclear" and ens.raw_top in {"make", "miss"}


@pytest.mark.parametrize(("fwd", "rev", "lean", "forbidden", "expected"), [
    ({"make": 0.9, "miss": 0.05, "unclear": 0.05}, None, "make", set(), "provisional_make"),
    ({"make": 0.05, "miss": 0.9, "unclear": 0.05}, None, "miss", set(), "provisional_miss"),
    ({"make": 0.9, "miss": 0.05, "unclear": 0.05}, None, "miss", set(), "pending"),  # disagrees with the lean
    ({"make": 0.8, "miss": 0.15, "unclear": 0.05}, None, "make", set(), "pending"),  # p_max below 0.85
    ({"make": 0.9, "miss": 0.05, "unclear": 0.05}, None, "make", {"make"}, "pending"),  # hard constraint
    ({"make": 0.9, "miss": 0.05, "unclear": 0.05}, {"make": 0.4, "miss": 0.5, "unclear": 0.1}, "make", set(),
     "pending"),  # orders disagree
    ({"make": 0.05, "miss": 0.05, "unclear": 0.9}, None, "make", set(), "pending"),  # abstain
])
def test_decide_shot_outcome_commit_rule(fwd, rev, lean, forbidden, expected) -> None:
    assert decide_shot_outcome(shot(fwd, rev), lean, forbidden, TH) == expected


def test_decide_shot_outcome_rejects_unpinned_model() -> None:
    r = shot({"make": 0.99, "miss": 0.005, "unclear": 0.005}, model="jev-1.14.0")
    assert decide_shot_outcome(r, "make", set(), TH) == "pending"


def test_decide_release_intent() -> None:
    probs = {"shot_attempt": 0.92, "pass": 0.05, "unclear": 0.03}
    r = answered({"release_intent": stub_choice(probs), "release_intent_rev": stub_choice(probs)}, Q.B5_RELEASE)
    assert decide_release_intent(r, "shot_attempt", set(), TH) == "provisional_shot_attempt"
    assert decide_release_intent(r, "pass", set(), TH) == "pending"


def _b1(play: dict, play_rev: dict | None = None, clear: float = 0.9, stakes=(0.0, 0.1, 0.8, 0.1),
        highlight: float = 0.5) -> JevResult:
    play_rev = play if play_rev is None else play_rev
    return answered({"play_type": stub_choice(play), "play_type_rev": stub_choice(play_rev),
                     "summary_is_clear": stub_noul(clear), "moment_stakes": stub_score(stakes),
                     "highlight_play": stub_noul(highlight)}, Q.B1_POSSESSION_END)


def _peaked(label: str, p: float) -> dict[str, float]:
    rest = (1.0 - p) / (len(Q.PLAY_TYPE_LABELS) - 1)
    return {k: (p if k == label else rest) for k in Q.PLAY_TYPE_LABELS}


def test_decode_play_type_thresholds() -> None:
    assert decode_play_type(_b1(_peaked("PRBallHandler", 0.8)), TH).label == "PRBallHandler"
    assert decode_play_type(_b1(_peaked("PRBallHandler", 0.8), clear=0.5), TH) is None  # summary unclear
    assert decode_play_type(_b1(_peaked("PRBallHandler", 0.5)), TH) is None  # confidence < 0.60
    assert decode_play_type(_b1(_peaked("PRBallHandler", 0.8), _peaked("Isolation", 0.8)), TH) is None
    assert decode_play_type(_b1(_peaked("unclear", 0.95)), TH) is None  # abstain is not accepted
    j = decode_play_type(_b1(_peaked("Cut", 0.7)), TH)
    assert j is not None and j.confidence < TH.play_type_coach_conf
    assert decode_play_type(_b1(_peaked("Cut", 0.7)), TH, coach=True) is None


def test_decode_stakes_rounds_and_normalises() -> None:
    j = decode_stakes(_b1(_peaked("Cut", 0.7), stakes=(0.0, 0.05, 0.15, 0.8)), TH)
    assert j.label == "decisive" and j.value == pytest.approx(2.75 / 3) and j.probabilities["decisive"] == 0.8
    assert decode_stakes(_b1(_peaked("Cut", 0.7), stakes=(0.25, 0.25, 0.25, 0.25)), TH) is None


def test_decode_guard_thresholds_per_sentence() -> None:
    overrides = {"supported_s1": stub_noul(0.95), "unlisted_s1": stub_noul(0.1), "speculation_s1": stub_noul(0.1),
                 "supported_s2": stub_noul(0.95), "unlisted_s2": stub_noul(0.1), "speculation_s2": stub_noul(0.4)}
    r = answered(overrides, Q.b4_line_guard(2))
    v1, v2 = decode_guard(r, ["a.", "b."], TH)
    assert v1.passed and not v2.passed and v2.speculation == pytest.approx(0.4)
    assert [v.passed for v in decode_guard(answered({}, Q.b4_line_guard(2)), ["a.", "b."], TH)] == [False, False]


def test_decode_cards() -> None:
    cards = {"card_a": "x", "card_b": "y"}
    qs = Q.b3_dead_ball_coach(cards)
    good = {"relevant__card_a": stub_noul(0.9), "relevant__card_b": stub_noul(0.3),
            "top_card": stub_choice({"card_a": 0.8, "card_b": 0.1, "none": 0.1})}
    assert decode_cards(answered(good, qs), list(cards), TH).top == "card_a"
    weak = {**good, "relevant__card_a": stub_noul(0.7)}
    assert decode_cards(answered(weak, qs), list(cards), TH).top is None
    none = {**good, "top_card": stub_choice({"card_a": 0.1, "card_b": 0.1, "none": 0.8})}
    assert decode_cards(answered(none, qs), list(cards), TH).top == "none"


# --------------------------------------------------------------------------- fallback tables

RULES = FallbackRules()


@pytest.mark.parametrize(("local", "label"), [
    ({"end": "made_shot", "seconds_since_oreb": 1.5, "start_cause": "inbound"}, "OffRebound"),
    ({"end": "missed_shot", "seconds_since_oreb": 2.5, "start_cause": "inbound", "passes": 2}, "unclear"),
    ({"end": "made_shot", "start_cause": "defensive_rebound", "seconds_since_start": 6.5}, "Transition"),
    ({"end": "turnover", "start_cause": "steal", "seconds_since_start": 3.0}, "Transition"),
    ({"end": "made_shot", "start_cause": "inbound", "seconds_since_start": 6.0, "passes": 1}, "unclear"),
    ({"end": "made_shot", "start_cause": "steal", "seconds_since_start": 7.5, "passes": 1}, "unclear"),
    ({"end": "made_shot", "start_cause": "defensive_rebound", "seconds_since_start": 5.0,
      "seconds_since_oreb": 3.0}, "unclear"),  # the reset after an offensive rebound is not transition
    ({"end": "missed_shot", "start_cause": "inbound", "seconds_since_start": 14.0, "passes": 0,
      "terminal_dribbles": 4}, "Isolation"),
    ({"end": "missed_shot", "start_cause": "inbound", "seconds_since_start": 14.0, "passes": 0,
      "terminal_dribbles": 3}, "unclear"),
    ({"end": "made_shot", "start_cause": "inbound", "seconds_since_start": 14.0, "passes": 2,
      "catch_and_shoot": True}, "Spotup"),
    ({"end": "missed_shot", "shot_distance_ft": 55.0, "game_clock_at_end_s": 0.8, "start_cause": "steal",
      "seconds_since_start": 1.0}, "Misc"),
    ({}, "unclear"),
])
def test_fallback_play_type_rules(local: dict, label: str) -> None:
    j = fallback_play_type(local, RULES)
    assert j.label == label and j.source == "fallback"
    assert j.confidence == (0.0 if label == "unclear" else 1.0)


def test_play_type_rules_from_a_real_possession_summary() -> None:
    from courtvision.types import Event, EventKind, Team
    evs = [Event("e1", EventKind.POSSESSION, 0.0, 0, team=Team.AWAY, player_id="away-1", attrs={"cause": "steal"}),
           Event("e2", EventKind.SHOT_ATTEMPT, 3.0, 90, team=Team.AWAY, player_id="away-1", value=2, shot_id="s1",
                 attrs={"shot_type": "layup"})]
    p = S.summarize_possession(evs, Team.AWAY, "Summit")
    assert fallback_play_type(S.possession_local(p), RULES).label == "Transition"


@pytest.mark.parametrize(("codes", "level"), [
    ({"clutch": True, "margin": "one_possession", "time": "final_minute", "run": False, "lead_change": False,
      "milestone": False}, 3),
    ({"clutch": True, "margin": "two_possession", "time": "final_minute", "run": False, "lead_change": False,
      "milestone": False}, 2),
    ({"clutch": True, "margin": "tied", "time": "final_five", "run": False, "lead_change": False,
      "milestone": False}, 2),
    ({"clutch": False, "margin": "double_digits", "time": "earlier", "run": True, "lead_change": False,
      "milestone": False}, 1),
    ({"clutch": False, "margin": "tied", "time": "earlier", "run": False, "lead_change": False,
      "milestone": False}, 1),
    ({"clutch": False, "margin": "comfortable", "time": "earlier", "run": False, "lead_change": False,
      "milestone": False}, 0),
])
def test_stakes_rule_table(codes: dict, level: int) -> None:
    assert stakes_level(codes)[0] == level


def test_stakes_fallback_reads_buckets_with_or_without_local() -> None:
    sit = S.GameSituation(home="Harbor", away="Summit", home_score=90, away_score=89, period=4, clock_s=30.0)
    block = S.situation_block(sit)
    with_local = fallback_stakes(block, S.situation_local(sit))
    from_text = fallback_stakes(block, {})
    assert with_local.label == from_text.label == "decisive" and with_local.value == 1.0
    calm = S.GameSituation(home_score=40, away_score=20, period=2, clock_s=600.0)
    assert fallback_stakes(S.situation_block(calm), {}).label == "routine"


def test_highlight_fallback_from_event_flags() -> None:
    assert fallback_highlight({"flags": ["dunk"]}, {}).label == "yes"
    assert fallback_highlight({}, {"flags": ["block"]}).label == "yes"
    assert fallback_highlight({"flags": ["drew contact"]}, {"flags": ["steal"]}).label == "no"


def test_threshold_and_rule_overrides() -> None:
    assert Thresholds.from_overrides({"event_pmax": 0.9, "nonsense": 1}).event_pmax == 0.9
    assert FallbackRules.from_overrides({"transition_window_s": 5.0, "crisp_labels": ["Transition"]}).crisp_labels == (
        "Transition",)
