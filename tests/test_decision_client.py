"""JevClient over the real SDK + httpx2.MockTransport: modes, deadlines, breakers, auth, cache, replay.

Network-free: stub and replay modes never leave the process; live mode is only used with no key.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from courtvision.decision import questions as Q
from courtvision.decision import states as S
from courtvision.decision.client import (
    BUNDLES,
    Breaker,
    JevClient,
    JevConfig,
    StubAnswers,
    TTLCache,
    parse_replay_lines,
    request_key,
    stub_choice,
    stub_noul,
    wire_questions,
)

B5_STATE = {"shot": {"rim_plane_crossing": "not observed", "view_quality": "clear view of the rim"}}
MARGIN_S = 0.2  # scheduling slack on a loaded machine; the slow mocks take 2 s or more


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def bundle_cases() -> list[tuple[str, dict, dict]]:
    sit = S.GameSituation(home="Harbor", away="Summit", home_score=50, away_score=48, period=2, clock_s=400.0)
    poss = S.PossessionSummary(offense="Harbor", start_cause="inbound", end="made_shot", shot_value=2)
    b3_state, _ = S.b3_dead_ball_coach(sit, ["{P05} has three fouls", "Summit eight-nothing run"], seed="x")
    return [
        ("B1", S.b1_possession_end(sit, poss), Q.B1_POSSESSION_END),
        ("B2", S.b2_dead_ball_fan(sit, stoppage="a timeout", material={"stat_note": "Harbor shoots well"},
                                  storylines=["{P07} nears a milestone"]),
         Q.b2_dead_ball_fan({"stat_note", "storyline"}, 1)),
        ("B3", b3_state, Q.b3_dead_ball_coach(b3_state["cards"])),
        ("B4", S.b4_line_guard({"score": "Harbor 50, Summit 48"}, ["Harbor leads.", "{P07} scored."]),
         Q.b4_line_guard(2)),
        ("B5", S.b5_event_ambiguity("shot_outcome", B5_STATE["shot"]), Q.B5_SHOT_OUTCOME),
    ]


@pytest.mark.parametrize(("key", "state", "questions"), bundle_cases(), ids=[c[0] for c in bundle_cases()])
def test_sync_stub_round_trip_through_real_sdk(key: str, state: dict, questions: dict) -> None:
    stub = StubAnswers()
    client = JevClient(JevConfig(mode="stub"), stub=stub)
    r = client.ask(key, state, questions, tag="v1")
    assert r.ok and r.source == "stub" and r.model == "jev-1.13.0" and r.tag == "v1"
    assert set(r.response.answers) == set(questions)
    assert r.request_id and r.request_id.startswith("stub-") and r.input_tokens is not None
    assert "_local" not in json.dumps(stub.last_body)
    assert stub.last_body["model"] == "jev-1.13.0"
    assert stub.last_body["questions"] == wire_questions(questions)
    client.close()


@pytest.mark.parametrize(("key", "state", "questions"), bundle_cases(), ids=[c[0] for c in bundle_cases()])
async def test_async_stub_round_trip_through_real_sdk(key: str, state: dict, questions: dict) -> None:
    client = JevClient(JevConfig(mode="stub"))
    r = await client.aask(key, state, questions)
    assert r.ok and r.source == "stub" and set(r.response.answers) == set(questions)
    for name, q in r.response.choices.items():  # default stub answers are maximally uncertain
        assert q.confidence == 0.0
    assert client.stats()["bundles"][key]["answered"] == 1
    await client.aclose()


async def test_stub_overrides_are_parsed_into_typed_answers() -> None:
    stub = StubAnswers(overrides={"shot_outcome": stub_choice({"make": 0.9, "miss": 0.05, "unclear": 0.05}),
                                  "warm": stub_noul(0.8)})
    client = JevClient(JevConfig(mode="stub"), stub=stub)
    r = await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)
    ans = r.response.choices["shot_outcome"]
    assert ans.choice == "make" and ans.confidence == pytest.approx((0.9 - 1 / 3) / (2 / 3))
    assert await client.warm_up() is True
    await client.aclose()


async def test_slow_answer_falls_back_within_the_hard_deadline() -> None:
    stub = StubAnswers(delay_s=2.0)
    client = JevClient(JevConfig(mode="stub", deadlines_s={"B5": 0.15, "B1": 0.2}), stub=stub)
    for key, qs, state, deadline in (("B5", Q.B5_SHOT_OUTCOME, B5_STATE, 0.15),
                                     ("B1", Q.B1_POSSESSION_END, {"possession": {"end": "x"}}, 0.2)):
        t0 = time.perf_counter()
        r = await client.aask(key, state, qs)
        elapsed = time.perf_counter() - t0
        assert not r.ok and r.source == "fallback" and r.error == "deadline"
        assert elapsed < deadline + MARGIN_S
        assert r.latency_ms >= deadline * 1000 * 0.9
    await client.aclose()


async def test_semaphore_queue_wait_counts_toward_the_deadline() -> None:
    stub = StubAnswers(delay_s=0.3)
    client = JevClient(JevConfig(mode="stub", max_in_flight=1, deadlines_s={"B5": 0.4}), stub=stub)
    t0 = time.perf_counter()
    first, second = await asyncio.gather(client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME),
                                         client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME))
    assert first.ok and not second.ok and second.error == "deadline"
    assert time.perf_counter() - t0 < 0.4 + MARGIN_S
    await client.aclose()


async def test_breaker_opens_after_n_failures_and_recovers_after_cooldown() -> None:
    clock = FakeClock()
    stub = StubAnswers(status=503)
    client = JevClient(JevConfig(mode="stub", breaker_threshold=3, breaker_cooldown_s=30.0), stub=stub, clock=clock)
    for _ in range(3):
        r = await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)
        assert r.error == "TypeSafeInternalServerError"
    assert stub.hits == 3  # B5 has max_retries=0
    r = await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)
    assert r.error == "circuit_open" and stub.hits == 3 and client.breaker("B5").state == "open"

    clock.t += 30.5  # half-open: one failure re-opens immediately
    r = await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)
    assert r.error == "TypeSafeInternalServerError" and client.breaker("B5").state == "open"

    clock.t += 30.5
    stub.status = 200
    r = await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)
    assert r.ok and client.breaker("B5").state == "closed"
    stats = client.stats()["bundles"]["B5"]
    assert stats["fallbacks"] == {"TypeSafeInternalServerError": 4, "circuit_open": 1} and stats["answered"] == 1
    await client.aclose()


async def test_one_breaker_per_bundle() -> None:
    clock = FakeClock()
    stub = StubAnswers(status=503)
    client = JevClient(JevConfig(mode="stub", breaker_threshold=2), stub=stub, clock=clock)
    for _ in range(2):
        await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)
    stub.status = 200
    assert (await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)).error == "circuit_open"
    for key, state, qs in bundle_cases():
        if key != "B5":
            assert (await client.aask(key, state, qs)).ok, key
    r = await client.aask("B6", {"frame": "x"}, Q.B6_SETUP)
    assert r.error == "images_disabled"  # B6 never runs unless images_enabled
    await client.aclose()


@pytest.mark.parametrize("status", [401, 403])
async def test_auth_error_disables_the_session(status: int) -> None:
    stub = StubAnswers(status=status)
    client = JevClient(JevConfig(mode="stub"), stub=stub)
    r = await client.aask("B1", {"possession": {}}, Q.B1_POSSESSION_END)
    assert not r.ok and r.error in {"TypeSafeAuthenticationError", "TypeSafePermissionDeniedError"}
    assert stub.hits == 1  # never retried
    assert client.available is False
    stub.status = 200
    for key, state, qs in bundle_cases():
        assert (await client.aask(key, state, qs)).error == "session_disabled"
    assert stub.hits == 1
    await client.aclose()


@pytest.mark.parametrize("status", [408, 413, 415])
async def test_other_4xx_are_never_retried(status: int) -> None:
    stub = StubAnswers(status=status)
    client = JevClient(JevConfig(mode="stub"), stub=stub)
    r = await client.aask("B1", {"possession": {}}, Q.B1_POSSESSION_END)  # B1 allows one retry
    assert not r.ok and r.error == "TypeSafeAPIError" and stub.hits == 1
    await client.aclose()


async def test_5xx_is_retried_within_the_bundle_policy() -> None:
    stub = StubAnswers(status=529)
    client = JevClient(JevConfig(mode="stub"), stub=stub)
    r = await client.aask("B1", {"possession": {}}, Q.B1_POSSESSION_END)
    assert r.error == "TypeSafeInternalServerError" and stub.hits == 2
    await client.aclose()


@pytest.mark.parametrize("status", [400, 404, 422])
async def test_request_bugs_disable_only_that_bundle_and_are_not_retried(status: int) -> None:
    stub = StubAnswers(status=status)
    client = JevClient(JevConfig(mode="stub"), stub=stub)
    r = await client.aask("B1", {"possession": {}}, Q.B1_POSSESSION_END)  # B1 allows one retry for 5xx/429
    assert not r.ok and stub.hits == 1
    stub.status = 200
    assert (await client.aask("B1", {"possession": {}}, Q.B1_POSSESSION_END)).error == "bundle_disabled"
    assert (await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)).ok
    assert client.available
    await client.aclose()


async def test_rate_limit_trips_the_breaker_for_retry_after() -> None:
    clock = FakeClock()
    stub = StubAnswers(status=429, headers={"retry-after-ms": "5"})
    client = JevClient(JevConfig(mode="stub", rate_limit_min_open_s=2.0), stub=stub, clock=clock)
    r = await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)
    assert r.error == "TypeSafeRateLimitError"
    stub.status = 200
    assert (await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)).error == "circuit_open"
    clock.t += 2.1
    assert (await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)).ok
    await client.aclose()


async def test_off_mode_and_disabled_config_never_touch_the_transport() -> None:
    for cfg in (JevConfig(mode="off"), JevConfig(mode="stub", enabled=False)):
        stub = StubAnswers()
        client = JevClient(cfg, stub=stub)
        assert client.available is False and client.disabled_reason == "mode_off"
        assert (await client.aask("B1", {}, Q.B1_POSSESSION_END)).error == "mode_off"
        assert client.ask("B5", B5_STATE, Q.B5_SHOT_OUTCOME).error == "mode_off"
        assert stub.hits == 0
    with pytest.raises(ValueError):
        JevClient(JevConfig(mode="banana"))


async def test_live_mode_without_a_key_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    client = JevClient(JevConfig(mode="live"))
    assert client.available is False
    r = await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)
    assert not r.ok and r.error == "session_disabled"
    assert client.ask("B5", B5_STATE, Q.B5_SHOT_OUTCOME).error == "session_disabled"
    fresh = JevClient(JevConfig(mode="live"))  # first touch is a call, not the property
    assert (await fresh.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)).error == "client_unavailable"
    assert fresh.available is False
    await client.aclose()
    await fresh.aclose()


def test_request_key_is_order_preserving() -> None:
    fwd = wire_questions({"q": Q.B5_SHOT_OUTCOME["shot_outcome"]})
    rev = wire_questions({"q": Q.B5_SHOT_OUTCOME["shot_outcome_rev"]})
    assert request_key("jev-1.13.0", B5_STATE, fwd) != request_key("jev-1.13.0", B5_STATE, rev)
    assert request_key("jev-1.13.0", B5_STATE, fwd) == request_key("jev-1.13.0", dict(B5_STATE), dict(fwd))
    assert request_key("jev-1.13.0", B5_STATE, fwd) != request_key("jev-1.14.0", B5_STATE, fwd)


async def test_cache_is_off_by_default_and_works_when_enabled() -> None:
    stub = StubAnswers()
    client = JevClient(JevConfig(mode="stub"), stub=stub)
    for _ in range(2):
        assert (await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)).source == "stub"
    assert stub.hits == 2

    clock = FakeClock()
    stub = StubAnswers()
    client = JevClient(JevConfig(mode="stub", cache_enabled=True, cache_ttl_s=10.0), stub=stub, clock=clock)
    a = await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME, tag=1)
    b = await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME, tag=2)
    assert (a.source, b.source, b.tag, stub.hits) == ("stub", "cache", 2, 1)
    assert b.response is a.response
    reordered = {"shot_outcome": Q.B5_SHOT_OUTCOME["shot_outcome_rev"], "shot_outcome_rev": Q.B5_SHOT_OUTCOME["shot_outcome"]}
    assert (await client.aask("B5", B5_STATE, reordered)).source == "stub" and stub.hits == 2
    clock.t += 11.0
    assert (await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)).source == "stub"
    await client.aclose()


def test_ttl_cache_lru_eviction() -> None:
    clock = FakeClock()
    cache = TTLCache(maxsize=2, ttl_s=5.0, clock=clock)
    from courtvision.decision.client import JevResult
    for k in "abc":
        cache.put(k, JevResult(bundle="B1", ok=True, source="stub"))
    assert len(cache) == 2 and cache.get("a") is None and cache.get("c") is not None


async def test_recorded_responses_replay_through_the_same_mock() -> None:
    records: list[dict] = []
    stub = StubAnswers(overrides={"shot_outcome": stub_choice({"make": 0.7, "miss": 0.2, "unclear": 0.1})})
    rec_client = JevClient(JevConfig(mode="stub"), stub=stub, record=records.append)
    original = await rec_client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)
    assert len(records) == 1 and records[0]["bundle"] == "B5" and records[0]["model"] == "jev-1.13.0"

    lines = [json.dumps(r) for r in records]
    replay = JevClient(JevConfig(mode="replay"), replay=parse_replay_lines(lines))
    again = await replay.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)
    assert again.ok and again.source == "replay"
    assert again.response.choices["shot_outcome"] == original.response.choices["shot_outcome"]
    sync_again = replay.ask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)
    assert sync_again.ok and sync_again.source == "replay"
    miss = await replay.aask("B5", {"shot": {"view_quality": "blocked"}}, Q.B5_SHOT_OUTCOME)
    assert miss.error == "replay_miss"
    await rec_client.aclose()
    await replay.aclose()


async def test_replay_path_loads_recorded_jsonl(tmp_path) -> None:
    records: list[dict] = []
    rec = JevClient(JevConfig(mode="stub"), record=records.append)
    await rec.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)
    path = tmp_path / "jev.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    replay = JevClient(JevConfig(mode="replay", replay_path=str(path)))
    assert (await replay.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)).source == "replay"
    await rec.aclose()
    await replay.aclose()


async def test_unpinned_model_is_flagged_but_answered() -> None:
    stub = StubAnswers(model="jev-1.14.0")
    client = JevClient(JevConfig(mode="stub"), stub=stub)
    r = await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)
    assert r.ok and r.model == "jev-1.14.0" and client.stats()["models_seen"] == ["jev-1.14.0"]
    await client.aclose()


def test_breaker_unit() -> None:
    clock = FakeClock()
    br = Breaker(threshold=2, cooldown_s=5.0, clock=clock)
    br.failure()
    assert br.allow() and br.state == "closed"
    br.success()
    br.failure()
    assert br.allow()  # success reset the count
    br.failure()
    assert not br.allow() and br.state == "open" and br.trips == 1
    clock.t += 5.0
    assert br.allow() and br.state == "half_open"
    br.success()
    assert br.state == "closed"


def test_bundles_match_the_design_table() -> None:
    assert {k: b.deadline_s for k, b in BUNDLES.items()} == JevConfig().deadlines_s
    assert all(b.model == "jev-1.13.0" for k, b in BUNDLES.items() if k != "B6")
    assert BUNDLES["B6"].images and not BUNDLES["B6"].cacheable and BUNDLES["B6"].model == "jev-1.14.0"
    assert BUNDLES["B5"].retry.max_retries == 0
    assert all(b.http_timeout_s < b.deadline_s for b in BUNDLES.values())
