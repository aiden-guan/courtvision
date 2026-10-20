"""JevDecider façade: B1-B5 through the real SDK in stub mode, fallbacks first, never raising."""

from __future__ import annotations

import json
import time

import pytest

from courtvision.decision import questions as Q
from courtvision.decision import states as S
from courtvision.decision.advisor import JevConfig, JevDecider, Judgement, LineVerdict, PossessionAssessment
from courtvision.decision.client import JevClient, StubAnswers, stub_choice, stub_noul, stub_score
from courtvision.types import Insight, Team

NAMES = ["Jalen Brooks", "Marcus Hill", "Theo Okafor"]


def decider(stub: StubAnswers | None = None, **cfg) -> tuple[JevDecider, StubAnswers]:
    stub = stub or StubAnswers()
    config = JevConfig(mode=cfg.pop("mode", "stub"), **cfg)
    return JevDecider(config, JevClient(config, stub=stub)), stub


def peaked(labels, label: str, p: float) -> dict[str, float]:
    rest = (1.0 - p) / (len(labels) - 1)
    return {k: (p if k == label else rest) for k in labels}


def b1_state(**poss) -> dict:
    sit = S.GameSituation(home="Harbor", away="Summit", home_score=70, away_score=60, period=3, clock_s=400.0,
                          foul_note="Theo Okafor has four fouls")
    p = S.PossessionSummary(**{"offense": "Harbor", "start_cause": "inbound", "end": "made_shot", "shot_value": 2,
                               "passes": 3, "terminal_dribbles": 2, "seconds_since_start": 15.0, **poss})
    return S.b1_possession_end(sit, p, description="Jalen Brooks drove and scored", names=NAMES)


def confident_b1(label: str = "PRBallHandler") -> dict:
    probs = peaked(Q.PLAY_TYPE_LABELS, label, 0.85)
    return {"play_type": stub_choice(probs), "play_type_rev": stub_choice(probs),
            "summary_is_clear": stub_noul(0.9), "moment_stakes": stub_score([0.05, 0.85, 0.05, 0.05]),
            "highlight_play": stub_noul(0.92)}


# --------------------------------------------------------------------------- B1


async def test_b1_confident_answers_are_used() -> None:
    d, stub = decider(StubAnswers(overrides=confident_b1()))
    a = await d.possession_end(b1_state())
    assert isinstance(a, PossessionAssessment)
    assert (a.play_type.label, a.play_type.source, a.play_type.model) == ("PRBallHandler", "stub", "jev-1.13.0")
    assert a.summary_clear.label == "yes" and a.summary_clear.value == pytest.approx(0.9)
    assert a.stakes.label == "notable" and a.stakes.value == pytest.approx(1.1 / 3)  # E[level] / 3
    assert a.highlight.label == "yes" and a.highlight.confidence == pytest.approx(0.84)
    body = json.dumps(stub.last_body)
    assert "_local" not in body and not any(n.split()[-1] in body for n in NAMES)
    await d.aclose()


async def test_b1_uncertain_answers_take_every_fallback() -> None:
    d, _ = decider()
    a = await d.possession_end(b1_state())
    for j in (a.play_type, a.summary_clear, a.stakes, a.highlight):
        assert j.source == "fallback" and j.model is None
    assert a.play_type.label == "unclear" and a.stakes.label == "routine" and a.highlight.label == "no"
    assert d.stats()["methods"]["possession_end"]["below_threshold"]["play_type"] == 1


async def test_b1_crisp_rule_wins_over_jev() -> None:
    d, _ = decider(StubAnswers(overrides=confident_b1("Spotup")))
    # A putback: the rebounder shoots 1 s after the offensive rebound, no pass and no dribble.
    a = await d.possession_end(b1_state(seconds_since_oreb=1.0, passes=0, terminal_dribbles=0))
    assert a.play_type.label == "OffRebound" and a.play_type.source == "fallback"
    assert a.stakes.source == "stub"  # the other parts still use Jev
    assert d.stats()["methods"]["possession_end"]["rule_wins"] == {"play_type": 1}


async def test_b1_order_disagreement_falls_back() -> None:
    over = confident_b1("PRBallHandler")
    over["play_type_rev"] = stub_choice(peaked(Q.PLAY_TYPE_LABELS, "Isolation", 0.85))
    d, _ = decider(StubAnswers(overrides=over))
    a = await d.possession_end(b1_state(catch_and_shoot=True))
    assert a.play_type.label == "Spotup" and a.play_type.source == "fallback"


async def test_b1_unpinned_model_falls_back_entirely() -> None:
    d, _ = decider(StubAnswers(overrides=confident_b1(), model="jev-1.14.0"))
    a = await d.possession_end(b1_state())
    assert {j.source for j in (a.play_type, a.summary_clear, a.stakes, a.highlight)} == {"fallback"}
    st = d.stats()
    assert st["methods"]["possession_end"]["fallbacks"] == {"unpinned_model": 1}
    assert st["models_seen"] == ["jev-1.14.0"]


async def test_b1_hand_built_state_without_local_still_gets_bucket_fallbacks() -> None:
    d, _ = decider(mode="off")
    state = {"situation": {"clutch": S.CLUTCH_YES, "margin": "Summit leads; a one-possession game",
                           "time_bucket": "final minute of the game"},
             "last_play": {"description": "Summit's shooter made a dunk", "event": "made two-point shot",
                           "flags": ["dunk"]}}
    a = await d.possession_end(state)
    assert a.stakes.label == "decisive" and a.highlight.label == "yes" and a.play_type.label == "unclear"


# --------------------------------------------------------------------------- B2


def b2_state(due: list[bool] | None = None) -> dict:
    sit = S.GameSituation(home="Harbor", away="Summit", home_score=50, away_score=50, period=2, clock_s=200.0)
    return S.b2_dead_ball_fan(sit, stoppage="a timeout",
                              material={"stat_note": "{P03} has twenty points", "tendency": "Summit doubles the post"},
                              storylines=["{P07} is two assists from a milestone"],
                              storylines_due_now=due or [], dead_ball_id="db-1")


async def test_b2_confident_answer() -> None:
    over = {"next_line_kind": stub_choice({"stat_note": 0.05, "tendency": 0.85, "storyline": 0.05, "silence": 0.05}),
            "storyline_due_0": stub_noul(0.8)}
    d, stub = decider(StubAnswers(overrides=over))
    kind, stories = await d.next_line_kind(b2_state(), {"stat_note", "tendency", "storyline"},
                                           ["{P07} is two assists from a milestone"])
    assert kind.label == "tendency" and kind.source == "stub"
    assert [s.label for s in stories] == ["yes"] and stories[0].source == "stub"
    assert list(stub.last_body["questions"]["next_line_kind"]["criteria"]) == [
        "stat_note", "tendency", "storyline", "silence"]


async def test_b2_fallback_rotation_is_deterministic() -> None:
    d, _ = decider(mode="off")
    kinds = {"stat_note", "tendency"}
    labels = [(await d.next_line_kind(b2_state(), kinds, []))[0].label for _ in range(6)]
    assert labels == ["stat_note", "silence", "tendency", "silence", "stat_note", "silence"]
    kind, stories = await d.next_line_kind(b2_state([True]), {"storyline", "stat_note"},
                                           ["{P07} is two assists from a milestone"])
    assert kind.label == "storyline" and stories[0].label == "yes" and kind.confidence == 1.0
    assert (await d.next_line_kind(b2_state(), set(), []))[0].label == "silence"


async def test_b2_jev_choice_outside_offered_kinds_is_ignored() -> None:
    over = {"next_line_kind": stub_choice({"stat_note": 0.05, "tendency": 0.85, "storyline": 0.05, "silence": 0.05})}
    d, _ = decider(StubAnswers(overrides=over))
    kind, _ = await d.next_line_kind(b2_state(), {"stat_note"}, [])
    assert kind.source == "fallback" and kind.label == "stat_note"


# --------------------------------------------------------------------------- B3


def cards() -> list[Insight]:
    return [Insight("i1", 10.0, "run", "Summit eight-nothing run, two turnovers", priority=2, team=Team.AWAY),
            Insight("i2", 10.0, "foul_trouble", "Marcus Hill three fouls in the second quarter", priority=5),
            Insight("i3", 10.0, "shot_quality", "Summit eFG well above expected, small sample", priority=1)]


async def test_b3_fallback_is_highest_priority_card() -> None:
    d, _ = decider(mode="off")
    assert (await d.pick_insight(cards(), {})).insight_id == "i2"
    assert await d.pick_insight([], {}) is None
    d2, _ = decider()  # uncertain stub answers -> fallback
    assert (await d2.pick_insight(cards(), {"situation": {"margin": "the game is tied"}})).insight_id == "i2"


async def test_b3_confident_pick_and_none() -> None:
    probe, stub = decider()
    ctx = {"situation": {"margin": "the game is tied"}, "staff_team": "Harbor",
           "_local": {"dead_ball_id": "db-9", "redact_names": NAMES}}
    await probe.pick_insight(cards(), ctx)
    body = stub.last_body
    assert "Hill" not in json.dumps(body) and body["state"]["situation"]["staff_team"] == "Harbor"
    key = next(k for k, v in body["state"]["cards"].items() if v.startswith("Summit eFG"))
    keys = list(body["state"]["cards"])
    over = {f"relevant__{k}": stub_noul(0.9 if k == key else 0.2) for k in keys}
    over["top_card"] = stub_choice({**{k: (0.8 if k == key else 0.2 / len(keys)) for k in keys}, "none": 0.0})
    d, _ = decider(StubAnswers(overrides=over))
    assert (await d.pick_insight(cards(), ctx)).insight_id == "i3"
    over["top_card"] = stub_choice({**{k: 0.02 for k in keys}, "none": 0.94})
    d, _ = decider(StubAnswers(overrides=over))
    assert await d.pick_insight(cards(), ctx) is None


async def test_b3_accepts_a_game_situation_in_ctx() -> None:
    d, stub = decider()
    sit = S.GameSituation(home="Harbor", away="Summit", home_score=60, away_score=61, period=4, clock_s=90.0)
    await d.pick_insight(cards(), {"situation": sit, "staff_team": "Harbor", "bonus": "Summit is in the bonus"})
    situation = stub.last_body["state"]["situation"]
    assert situation["clutch"] == S.CLUTCH_YES and situation["bonus"] == "Summit is in the bonus"
    assert len(stub.last_body["state"]["cards"]) == 3


# --------------------------------------------------------------------------- B4

FACTS = {"score": "Harbor 98, Summit 95", "clock": "fourth quarter, under two minutes",
         "recent_events": ["{P07} made a driving layup"], "allowed_numbers": ["98", "95"]}


async def test_b4_guard_fallback_is_not_passed() -> None:
    for d, _ in (decider(mode="off"), decider()):  # off, and uncertain stub answers
        verdicts = await d.guard_lines(FACTS, ["{P07} attacks the rim.", "Summit needs a stop."])
        assert all(isinstance(v, LineVerdict) and not v.passed for v in verdicts)
    d, _ = decider(StubAnswers(delay_s=2.0), deadlines_s={"B4": 0.1})
    assert [v.passed for v in await d.guard_lines(FACTS, ["One."])] == [False]
    assert await d.guard_lines(FACTS, []) == []


async def test_b4_confident_guard_passes_and_splits_long_lists() -> None:
    over = {}
    for i in (1, 2):
        over |= {f"supported_s{i}": stub_noul(0.97), f"unlisted_s{i}": stub_noul(0.05),
                 f"speculation_s{i}": stub_noul(0.05)}
    d, stub = decider(StubAnswers(overrides=over))
    lines = ["{P07} attacks the rim.", "Summit needs a stop.", "Harbor leads by three."]
    verdicts = await d.guard_lines(FACTS, lines)
    assert [v.sentence for v in verdicts] == lines and all(v.passed for v in verdicts)
    assert stub.hits == 2  # two requests: 2 + 1 sentences
    assert "allowed_numbers" not in json.dumps(stub.last_body["state"])


# --------------------------------------------------------------------------- B5

CUES = {"rim_plane_crossing": "not observed: the ball was hidden behind a player at the rim",
        "drop_speed_frac": 0.5, "net_motion_z": 3.5, "rim_contact": True}


def b5_stub(fwd: dict[str, float], rev: dict[str, float] | None = None, **kw) -> StubAnswers:
    rev = fwd if rev is None else rev
    return StubAnswers(overrides={"shot_outcome": stub_choice(fwd), "shot_outcome_rev": stub_choice(rev)}, **kw)


async def test_b5_commit_only_with_agreement_lean_and_pmax() -> None:
    sure = {"make": 0.92, "miss": 0.05, "unclear": 0.03}
    d, _ = decider(b5_stub(sure))
    j = await d.ambiguity("shot_outcome", CUES, "make")
    assert isinstance(j, Judgement) and j.label == "make" and j.source == "stub"
    assert j.probabilities["make"] == pytest.approx(0.92)
    assert (await d.ambiguity("shot_outcome", CUES, "miss")).label == "unclear"
    d, _ = decider(b5_stub(sure, {"make": 0.3, "miss": 0.6, "unclear": 0.1}))
    assert (await d.ambiguity("shot_outcome", CUES, "make")).source == "fallback"
    d, _ = decider(b5_stub(sure))
    forbidden = {**CUES, "_local": {"forbidden": ["make"]}}
    assert (await d.ambiguity("shot_outcome", forbidden, "make")).label == "unclear"


async def test_b5_bad_arguments_and_release_intent() -> None:
    d, stub = decider()
    assert (await d.ambiguity("foul", CUES, "make")).label == "unclear"
    assert (await d.ambiguity("shot_outcome", CUES, "unclear")).label == "unclear"
    assert stub.hits == 0
    probs = {"shot_attempt": 0.04, "pass": 0.93, "unclear": 0.03}
    d, _ = decider(StubAnswers(overrides={"release_intent": stub_choice(probs), "release_intent_rev": stub_choice(probs)}))
    j = await d.ambiguity("release_intent", {"direction": "toward a teammate", "arc": "high"}, "pass")
    assert j.label == "pass" and j.source == "stub"


async def test_b5_slow_answer_falls_back_within_deadline() -> None:
    d, _ = decider(b5_stub({"make": 0.95, "miss": 0.03, "unclear": 0.02}, delay_s=3.0), deadlines_s={"B5": 0.15})
    t0 = time.perf_counter()
    j = await d.ambiguity("shot_outcome", CUES, "make")
    assert j.label == "unclear" and j.source == "fallback"
    assert time.perf_counter() - t0 < 0.15 + 0.2 and j.latency_ms >= 100  # deadline + slack; mock takes 3 s
    assert d.stats()["bundles"]["B5"]["fallbacks"] == {"deadline": 1}


# --------------------------------------------------------------------------- modes, auth, stats


async def test_off_mode_returns_fallbacks_without_calls() -> None:
    d, stub = decider(mode="off")
    assert d.available is False
    a = await d.possession_end(b1_state())
    kind, _ = await d.next_line_kind(b2_state(), {"stat_note"}, [])
    pick = await d.pick_insight(cards(), {})
    verdicts = await d.guard_lines(FACTS, ["One."])
    amb = await d.ambiguity("shot_outcome", CUES, "make")
    assert a.play_type.source == kind.source == amb.source == "fallback"
    assert pick.insight_id == "i2" and not verdicts[0].passed and stub.hits == 0
    methods = d.stats()["methods"]
    assert all(m["fallbacks"].get("mode_off", 0) >= 1 for m in methods.values())


async def test_401_disables_the_session_for_every_method() -> None:
    d, stub = decider(StubAnswers(status=401))
    a = await d.possession_end(b1_state())
    assert a.play_type.source == "fallback" and d.available is False
    stub.status = 200
    stub.overrides = confident_b1()
    assert (await d.possession_end(b1_state())).stakes.source == "fallback"
    assert (await d.ambiguity("shot_outcome", CUES, "make")).source == "fallback"
    assert stub.hits == 1
    assert d.stats()["disabled_reason"] == "TypeSafeAuthenticationError"


async def test_breaker_opens_and_recovers_through_the_facade() -> None:
    class Clock:
        t = 0.0

        def __call__(self) -> float:
            return self.t

    clock = Clock()
    stub = b5_stub({"make": 0.95, "miss": 0.03, "unclear": 0.02}, status=503)
    config = JevConfig(mode="stub", breaker_threshold=2, breaker_cooldown_s=10.0)
    d = JevDecider(config, JevClient(config, stub=stub, clock=clock))
    for _ in range(3):
        assert (await d.ambiguity("shot_outcome", CUES, "make")).source == "fallback"
    assert stub.hits == 2
    stub.status = 200
    assert (await d.ambiguity("shot_outcome", CUES, "make")).source == "fallback"  # still open
    clock.t += 10.5
    assert (await d.ambiguity("shot_outcome", CUES, "make")).label == "make"
    assert d.stats()["methods"]["ambiguity"]["fallbacks"] == {"TypeSafeInternalServerError": 2, "circuit_open": 2}


async def test_stats_shape() -> None:
    d, _ = decider(StubAnswers(overrides=confident_b1()))
    for _ in range(3):
        await d.possession_end(b1_state())
    st = d.stats()
    b1 = st["bundles"]["B1"]
    assert b1["calls"] == b1["answered"] == 3 and b1["p50_ms"] is not None and b1["p95_ms"] >= b1["p50_ms"]
    assert b1["input_tokens"] > 0 and st["models_seen"] == ["jev-1.13.0"]
    assert st["methods"]["possession_end"]["calls"] == 3
    assert st["methods"]["possession_end"]["jev_parts"]["play_type"] == 3
    await d.aclose()
