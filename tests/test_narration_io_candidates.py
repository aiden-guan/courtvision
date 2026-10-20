"""CandidatePool: window policy, validator + Jev guard, version binding, deadlines, D16 checks."""

from __future__ import annotations

import asyncio

import pytest

from courtvision.narration.candidates import STATE_INDEPENDENT, CandidateConfig, CandidatePool
from courtvision.narration.facts import build_facts
from courtvision.narration.llm import NarrationLLMError
from courtvision.narration.pseudonym import PseudonymMap
from tests.test_narration_io_support import COURT, FakeJev, FakeLLM, FakeStats, made, make_roster

ROSTER = make_roster()


def _setup(reply: object = "SILENCE", *, jev: FakeJev | None = None, delay_s: float = 0.0,
           **cfg: object) -> tuple[CandidatePool, FakeLLM, dict, str]:
    pseudo = PseudonymMap(ROSTER, seed=0)
    facts = build_facts([made(3)], FakeStats(), ROSTER, COURT, None, pseudo, state_version=1)
    tok = pseudo.token("home-23")
    llm = FakeLLM(reply(tok) if callable(reply) else reply, delay_s=delay_s)
    pool = CandidatePool(llm, ROSTER, pseudo, jev=jev if jev is not None else FakeJev(), config=CandidateConfig(**cfg))
    pool.set_version(1)
    return pool, llm, facts, tok


def test_window_policy_threshold() -> None:
    pool, *_ = _setup()
    # p95 12 s + speaking a 35-word line at 3.5 words/s (10 s) + 1 s (RESEARCH §7A.3)
    assert pool.tts_s == pytest.approx(10.0) and pool.threshold_s == pytest.approx(12.0 + 10.0 + 1.0)
    assert not pool.window_ok(10.0)  # free throw: too short until p95 is measured
    assert not pool.window_ok(20.0)  # a 35-word line requested here would still be talking at 23 s
    assert pool.window_ok(30.0) and pool.window_ok(60.0)  # NFHS 30 s / FIBA 60 s timeouts
    short, *_ = _setup(max_words=14)  # a tighter word cap needs a shorter window
    assert short.threshold_s == pytest.approx(12.0 + 4.0 + 1.0)
    fixed, *_ = _setup(tts_s=4.0)
    assert fixed.threshold_s == pytest.approx(17.0)


async def test_short_window_requests_nothing() -> None:
    pool, llm, facts, _ = _setup()
    assert pool.request(facts, state_version=1, t_occ=299.0, expected_window_s=8.0) == 0
    assert llm.prompts == [] and pool.counts["skipped:short_window"] == 1


async def test_two_parallel_candidates_pass_validator_and_guard() -> None:
    jev = FakeJev()
    pool, llm, facts, tok = _setup(lambda t: f"{{{t}}} has 12 points tonight.", jev=jev)
    assert pool.request(facts, state_version=1, t_occ=299.0, expected_window_s=60.0) == 2
    assert pool.request(facts, state_version=1, t_occ=299.0, expected_window_s=60.0) == 0  # enough pending
    await pool.wait(2.0)
    assert len(llm.prompts) == 2 and {p.splitlines()[0] for p in llm.prompts} == {"Line kind: stat_note",
                                                                               "Line kind: tendency"}
    assert len(jev.guard_calls) == 2 and jev.guard_calls[0][1] == [f"{{{tok}}} has 12 points tonight."]
    cand = pool.take(kinds=["tendency"])
    assert cand is not None and cand.kind == "tendency"
    assert cand.text_tokens == f"{{{tok}}} has 12 points tonight."
    assert cand.text == "Riley Thorsen has 12 points tonight."  # rendered locally, after both checks
    assert all(v.passed for v in cand.verdicts)
    # D16: neither the prompt nor the guard input carries a roster name
    assert not any(e.name in p for p in llm.prompts for e in ROSTER.entries)


async def test_validator_failure_is_not_kept() -> None:
    pool, _, facts, _ = _setup(lambda t: f"{{{t}}} has 99 points tonight.")
    pool.request(facts, state_version=1, t_occ=299.0, expected_window_s=60.0)
    await pool.wait(2.0)
    assert pool.take() is None and pool.counts["rejected:validator"] == 2


async def test_guard_failure_is_not_kept() -> None:
    jev = FakeJev(guard_pass=False)
    pool, _, facts, _ = _setup(lambda t: f"{{{t}}} has 12 points tonight.", jev=jev)
    pool.request(facts, state_version=1, t_occ=299.0, expected_window_s=60.0)
    await pool.wait(2.0)
    assert pool.take() is None and pool.counts["rejected:guard"] == 2


async def test_no_jev_means_no_request_d4() -> None:
    pseudo = PseudonymMap(ROSTER)
    llm = FakeLLM("Riverside leads.")
    pool = CandidatePool(llm, ROSTER, pseudo, jev=None)
    assert pool.request({}, state_version=0, t_occ=0.0, expected_window_s=60.0) == 0
    assert llm.prompts == [] and pool.counts["skipped:no_guard"] == 1
    unavailable = CandidatePool(llm, ROSTER, pseudo, jev=FakeJev(available=False))
    assert unavailable.request({}, state_version=0, t_occ=0.0, expected_window_s=60.0) == 0


async def test_stale_pending_candidates_are_cancelled() -> None:
    pool, _, facts, _ = _setup(lambda t: f"{{{t}}} has 12 points tonight.", delay_s=0.3)
    pool.request(facts, state_version=1, t_occ=299.0, expected_window_s=60.0)
    await asyncio.sleep(0.05)
    assert pool.set_version(2) == 2
    await pool.wait(1.0)
    assert pool.take() is None and pool.pending() == 0


async def test_stale_ready_candidates_are_dropped() -> None:
    pool, _, facts, _ = _setup(lambda t: f"{{{t}}} has 12 points tonight.")
    pool.request(facts, state_version=1, t_occ=299.0, expected_window_s=60.0)
    await pool.wait(2.0)
    assert len(pool.ready()) == 2
    pool.set_version(2)
    assert pool.ready() == [] and pool.take() is None


async def test_state_independent_candidates_survive_version_changes() -> None:
    pseudo = PseudonymMap(ROSTER, seed=0)
    facts = build_facts([], FakeStats(), ROSTER, COURT, None, pseudo, state_version=STATE_INDEPENDENT,
                        storylines=["Riley Thorsen is two points from 1,000 career points"])
    tok = pseudo.token("home-23")
    assert f"{{{tok}}}" in facts["storylines"][0]  # names pseudonymised before any cloud call
    llm = FakeLLM(f"{{{tok}}} is two points from 1,000 career points.")
    pool = CandidatePool(llm, ROSTER, pseudo, jev=FakeJev())
    assert pool.request(facts, state_version=0, t_occ=0.0, state_independent=True, kinds=("storyline",)) == 2
    await pool.wait(2.0)
    pool.set_version(7)
    cand = pool.take(kinds=["storyline"])
    assert cand is not None and cand.state_independent and cand.text.startswith("Riley Thorsen is two points")


async def test_deadline_and_llm_errors() -> None:
    pool, *_rest = _setup("Riverside leads.", delay_s=0.5, candidate_deadline_s=0.05)
    facts = _rest[1]
    pool.request(facts, state_version=1, t_occ=0.0, expected_window_s=60.0)
    await pool.wait(2.0)
    assert pool.counts["failed:deadline"] == 2
    pool2, _, facts2, _ = _setup(NarrationLLMError("refusal"))
    pool2.request(facts2, state_version=1, t_occ=0.0, expected_window_s=60.0)
    await pool2.wait(2.0)
    assert pool2.counts["failed:refusal"] == 2 and pool2.take() is None


async def test_silence_and_too_many_sentences() -> None:
    pool, _, facts, _ = _setup("SILENCE")
    pool.request(facts, state_version=1, t_occ=0.0, expected_window_s=60.0)
    await pool.wait(2.0)
    assert pool.counts["silence"] == 2
    pool2, _, facts2, _ = _setup("Riverside leads.\nLakeport trails.\nRiverside leads still.")
    pool2.request(facts2, state_version=1, t_occ=0.0, expected_window_s=60.0)
    await pool2.wait(2.0)
    assert pool2.counts["rejected:validator"] == 2


async def test_prompt_with_a_name_is_blocked() -> None:
    pool, llm, facts, _ = _setup("Riverside leads.")
    leaky = dict(facts, derived=["Riley Thorsen is hot"])
    pool.request(leaky, state_version=1, t_occ=0.0, expected_window_s=60.0)
    await pool.wait(2.0)
    assert llm.prompts == [] and pool.counts["blocked:names"] == 2


async def test_take_discards_lines_repeating_recent_speech() -> None:
    pool, _, facts, tok = _setup(lambda t: f"{{{t}}} has 12 points tonight.")
    pool.request(facts, state_version=1, t_occ=0.0, expected_window_s=60.0)
    await pool.wait(2.0)
    assert pool.take(recent_lines=[f"{{{tok}}} has 12 points tonight."]) is None
    assert pool.counts["dropped:repeat"] == 2


async def test_aclose_cancels_pending() -> None:
    pool, _, facts, _ = _setup("Riverside leads.", delay_s=5.0)
    pool.request(facts, state_version=1, t_occ=0.0, expected_window_s=60.0)
    await asyncio.sleep(0)
    await pool.aclose()
    assert pool.pending() == 0
