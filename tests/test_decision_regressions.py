"""Regression tests for defects found in the adversarial review of the decision layer.

Each test names the defect it pins. Network-free and deterministic.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from pathlib import Path

import numpy as np
import pytest
from typesafe_sdk import Image

from courtvision.decision import questions as Q
from courtvision.decision import states as S
from courtvision.decision.advisor import JevConfig, JevDecider
from courtvision.decision.client import JevClient, StubAnswers, sanitize, state_bytes, stub_noul, stub_score
from courtvision.decision.decoders import FallbackRules, Thresholds, decode_stakes, fallback_highlight, fallback_play_type
from courtvision.types import Event, EventKind, Team

ROOT = Path(__file__).resolve().parents[1]
B5_STATE = {"shot": {"view_quality": "clear view of the rim"}}


def ev(eid: str, kind: EventKind, t: float, team: Team | None = Team.HOME, pid: str | None = None,
       other: str | None = None, value: int = 0, shot_id: str | None = None, **attrs) -> Event:
    return Event(eid, kind, t, int(t * 30), team=team, player_id=pid, other_player_id=other, value=value,
                 shot_id=shot_id, attrs=attrs)


def run_with_timeout(fn, timeout_s: float = 5.0):
    """Run ``fn`` in a daemon thread; fail (instead of hanging the suite) if it does not return."""
    out: dict = {}

    def target() -> None:
        out["value"] = fn()

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout_s)
    assert not th.is_alive(), "did not terminate (infinite loop)"
    return out["value"]


# --------------------------------------------------------------------------- fit_state termination


def test_fit_state_terminates_on_text_without_spaces() -> None:
    """Defect: _truncate_longest re-produced the same 25-char string forever when the cut window had
    no space, so fit_state spun in the event loop on an over-budget state."""
    st = {"cards": {f"k{i}": "x" * 25 for i in range(60)}}
    run_with_timeout(lambda: S.fit_state(st, 1024))
    assert all(len(v) <= 24 for v in st["cards"].values())


def test_b4_builder_terminates_with_long_protected_sentences() -> None:
    facts = {"score": "Harbor 98, Summit 95", "labels": {"last_play_type": "pick-and-roll-ball-handler-drive"}}
    sentences = ["word " * 110, "more " * 110]
    st = run_with_timeout(lambda: S.b4_line_guard(facts, sentences))
    assert st["lines"] == {"s1": sentences[0], "s2": sentences[1]}  # sentences under review untouched


def test_fit_state_still_cuts_on_word_boundaries() -> None:
    st = {"a": "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 4}
    S.fit_state(st, 80)
    assert state_bytes(st) <= 80 and st["a"].endswith("…") and " " in st["a"]


# --------------------------------------------------------------------------- numpy / sets / enums


def test_sanitize_converts_numpy_sets_and_enum_keys() -> None:
    """Defect: numpy scalars/arrays and sets passed through sanitize; the SDK cannot encode them."""
    img = Image(b"\x89PNG\r\n\x1a\n")
    out = sanitize({"b": np.bool_(True), "i": np.int64(3), "f": np.float32("nan"), "s": np.str_("x"),
                    "a": np.array([1.5, 2.0]), "set": {"b", "a"}, Team.HOME: 1, "obj": object(), "img": img})
    assert out == {"b": True, "i": 3, "f": None, "s": "x", "a": [1.5, 2.0], "set": ["a", "b"], "home": 1,
                   "obj": None, "img": img}
    assert type(out["b"]) is bool and type(out["s"]) is str
    json.dumps({k: v for k, v in out.items() if k != "img"})  # JSON-safe


async def test_numpy_bool_in_state_does_not_disable_the_bundle() -> None:
    """Defect: one np.bool_ made the SDK raise "could not be encoded", which disabled B5 for the session."""
    stub = StubAnswers()
    client = JevClient(JevConfig(mode="stub"), stub=stub)
    r = await client.aask("B5", {"shot": {"rim_contact": np.bool_(True), "frames": np.int64(2)}}, Q.B5_SHOT_OUTCOME)
    assert r.ok and stub.last_body["state"] == {"shot": {"rim_contact": True, "frames": 2}}
    assert client.stats()["disabled_bundles"] == []
    assert (await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)).ok
    await client.aclose()


async def test_facade_never_raises_on_odd_state_values() -> None:
    """Defect: possession_end / guard_lines raised TypeError (json) on numpy, set or array values."""
    config = JevConfig(mode="stub")
    d = JevDecider(config, JevClient(config))
    for st in ({"situation": {"flag": np.bool_(True)}}, {"situation": {"tags": {"a"}}},
               {"situation": {"xy": np.array([1.0, 2.0])}}):
        a = await d.possession_end(st)
        assert a.play_type.label == "unclear"
    facts = {"score": "Harbor 50, Summit 48", "players": {"P01": {"pts": np.int64(3), "hot": np.bool_(True)}},
             "recent_events": np.array(["x", "y"])}
    verdicts = await d.guard_lines(facts, ["{P01} scores.", "Harbor leads.", "Summit calls timeout."])
    assert len(verdicts) == 3 and not any(v.passed for v in verdicts)
    assert d.stats()["disabled_bundles"] == []
    await d.aclose()


async def test_facade_last_resort_guard_returns_plain_fallbacks() -> None:
    d = JevDecider(JevConfig(mode="off"))
    a = await d.possession_end({"situation": 5, "last_play": "dunk"})  # malformed blocks
    assert (a.play_type.label, a.stakes.label, a.highlight.label) == ("unclear", "routine", "no")
    kind, stories = await d.next_line_kind({"_local": {"storylines_due_now": 7}}, {"stat_note"}, ["s"])
    assert kind.label == "silence" and [s.label for s in stories] == ["no"]
    assert (await d.ambiguity("shot_outcome", {"_local": {"forbidden": 3}}, "make")).label == "unclear"


def test_b5_keeps_numpy_bool_cues() -> None:
    """Defect: isinstance(np.bool_, bool) is False, so rim_contact was dropped as a raw number."""
    st = S.b5_event_ambiguity("shot_outcome", {"rim_contact": np.True_, "drop_speed_frac": np.float64("nan"),
                                               "net_motion_z": np.float32(3.5)})
    assert st["shot"] == {"rim_contact": "yes", "ball_after_rim": "not observed",
                          "net_motion": "strong net movement right after the ball reached the rim"}


def test_guard_facts_stringify_numpy_numbers_and_enum_keys() -> None:
    f = S.guard_facts({"teams": {Team.HOME: {"pts": np.int64(50), "pct": np.float64(0.5)}}})
    assert f == {"teams": {"home": {"pts": "50", "pct": "0.5"}}}


# --------------------------------------------------------------------------- play-type rules


RULES = FallbackRules()


def test_spotup_fires_for_numpy_bool_catch_and_shoot() -> None:
    local = {"end": "made_shot", "start_cause": "inbound", "seconds_since_start": 14.0, "passes": 2,
             "catch_and_shoot": np.True_}
    assert fallback_play_type(local, RULES).label == "Spotup"


def test_kick_out_after_offensive_rebound_is_not_a_putback() -> None:
    """Defect: OffRebound (a crisp rule that overrides Jev) fired for any shot <= 2 s after an
    offensive rebound, including a kick-out catch-and-shoot three (design: putback "with no dribble")."""
    base = [ev("e1", EventKind.POSSESSION, 10.0, pid="home-3", cause="inbound"),
            ev("e2", EventKind.SHOT_ATTEMPT, 18.0, pid="home-3", value=2, shot_id="s1"),
            ev("e3", EventKind.SHOT_MISSED, 19.0, pid="home-3", shot_id="s1"),
            ev("e4", EventKind.REBOUND, 19.5, pid="home-7", offensive=True)]
    kick = [*base, ev("e5", EventKind.PASS, 20.0, pid="home-7", other="home-5", completed=True),
            ev("e6", EventKind.SHOT_ATTEMPT, 21.2, pid="home-5", value=3, shot_id="s2", catch_and_shoot=True)]
    p = S.summarize_possession(kick, Team.HOME, "Harbor")
    assert p.seconds_since_oreb == pytest.approx(1.7) and p.passes == 1
    assert fallback_play_type(S.possession_local(p), RULES).label == "Spotup"
    dribble = [*base, ev("e5", EventKind.DRIBBLE, 19.9, pid="home-7"),
               ev("e6", EventKind.SHOT_ATTEMPT, 20.8, pid="home-7", value=2, shot_id="s2", dribbles_before=1)]
    p = S.summarize_possession(dribble, Team.HOME, "Harbor")
    assert fallback_play_type(S.possession_local(p), RULES).label != "OffRebound"
    tip = [*base, ev("e5", EventKind.SHOT_ATTEMPT, 20.4, pid="home-7", value=2, shot_id="s2", shot_type="tip")]
    p = S.summarize_possession(tip, Team.HOME, "Harbor")
    assert fallback_play_type(S.possession_local(p), RULES).label == "OffRebound"


def test_nan_and_numpy_attrs_do_not_break_summaries_or_rules() -> None:
    evs = [ev("e1", EventKind.POSSESSION, 0.0, pid="home-3", cause="inbound"),
           ev("e2", EventKind.SHOT_ATTEMPT, 9.0, pid="home-3", value=2, shot_id="s1",
              distance_ft=float("nan"), dribbles_before=float("nan"), catch_and_shoot=np.False_)]
    p = S.summarize_possession(evs, Team.HOME, "Harbor")  # int(nan) used to raise ValueError
    assert p.shot_distance_ft is None and p.catch_and_shoot is False and type(p.catch_and_shoot) is bool
    assert fallback_play_type(S.possession_local(p), RULES).label == "unclear"
    assert S.time_bucket(True, float("nan"), 24) == "game clock unknown"
    assert S.shot_clock_text(float("nan")) is None and not S.is_clutch(True, float("nan"), 1)


# --------------------------------------------------------------------------- possession boundaries


def test_next_possession_events_do_not_leak_into_this_one() -> None:
    """Defect: events after the opponent's POSSESSION were part of this possession: their dunk
    flagged our missed possession as a highlight."""
    evs = [ev("e1", EventKind.POSSESSION, 10.0, pid="home-3", cause="inbound"),
           ev("e2", EventKind.SHOT_ATTEMPT, 18.0, pid="home-3", value=2, shot_id="s1"),
           ev("e3", EventKind.SHOT_MISSED, 19.0, pid="home-3", shot_id="s1"),
           ev("e4", EventKind.REBOUND, 19.4, team=Team.AWAY, pid="away-11", offensive=False),
           ev("e5", EventKind.POSSESSION, 19.4, team=Team.AWAY, pid="away-11", cause="defensive_rebound"),
           ev("e6", EventKind.SHOT_ATTEMPT, 23.0, team=Team.AWAY, pid="away-11", value=2, shot_id="s2",
              shot_type="dunk"),
           ev("e7", EventKind.SHOT_MADE, 23.3, team=Team.AWAY, pid="away-11", value=2, shot_id="s2"),
           ev("e8", EventKind.TURNOVER, 30.0, team=None, type="other")]
    p = S.summarize_possession(evs, Team.HOME, "Harbor")
    assert p.end == "missed_shot" and "dunk" not in p.flags
    assert fallback_highlight({}, {"flags": p.flags}).label == "no"


def test_made_shot_stays_the_end_despite_a_later_foul() -> None:
    """Defect: a defensive foul after a made basket (outside a 1 s window of the make) became the
    possession's end."""
    evs = [ev("e1", EventKind.POSSESSION, 10.0, pid="home-3", cause="inbound"),
           ev("e2", EventKind.SHOT_ATTEMPT, 15.0, pid="home-3", value=2, shot_id="s1"),
           ev("e3", EventKind.SHOT_MADE, 15.8, pid="home-3", value=2, shot_id="s1"),
           ev("e4", EventKind.FOUL, 19.0, team=Team.AWAY, pid="away-11", other="home-7")]
    assert S.summarize_possession(evs, Team.HOME, "Harbor").end == "made_shot"


def test_and_one_on_a_jumper_with_long_flight() -> None:
    """Defect: the and-one window was measured only from SHOT_MADE, so a foul at the release of a
    three that took over 1 s to drop was missed (and could even end the possession as free throws)."""
    evs = [ev("e1", EventKind.POSSESSION, 10.0, pid="home-3", cause="inbound"),
           ev("e2", EventKind.SHOT_ATTEMPT, 15.0, pid="home-3", value=3, shot_id="s1", distance_ft=25.0),
           ev("e3", EventKind.FOUL, 15.3, team=Team.AWAY, pid="away-11", other="home-3"),
           ev("e4", EventKind.SHOT_MADE, 16.5, pid="home-3", value=3, shot_id="s1"),
           ev("e5", EventKind.SHOT_ATTEMPT, 40.0, pid="home-3", value=1, shot_id="s2", shot_type="free_throw")]
    p = S.summarize_possession(evs, Team.HOME, "Harbor")
    assert p.end == "made_shot" and p.shot_value == 3 and "and_one" in p.flags
    other_player = [*evs[:2], ev("e3", EventKind.FOUL, 15.3, team=Team.AWAY, pid="away-11", other="home-7"), evs[3]]
    assert "and_one" not in S.flags_from_events(other_player)


def test_terminal_block_keeps_the_terminal_action_last() -> None:
    """Defect: "a defender blocked the shot" was appended after the terminal action, so an over-budget
    state (which keeps only the last action) lost the terminal action; an earlier block also read as
    if it blocked the final shot."""
    evs = [ev("e1", EventKind.POSSESSION, 10.0, pid="home-3", cause="inbound"),
           ev("e2", EventKind.SHOT_ATTEMPT, 14.0, pid="home-3", value=2, shot_id="s1"),
           ev("e3", EventKind.BLOCK, 14.2, team=Team.AWAY, pid="away-11", other="home-3", shot_id="s1"),
           ev("e4", EventKind.SHOT_MISSED, 14.3, pid="home-3", shot_id="s1"),
           ev("e5", EventKind.REBOUND, 15.0, pid="home-7", offensive=True),
           ev("e6", EventKind.SHOT_ATTEMPT, 20.0, pid="home-7", value=2, shot_id="s2"),
           ev("e7", EventKind.BLOCK, 20.1, team=Team.AWAY, pid="away-11", other="home-7", shot_id="s2"),
           ev("e8", EventKind.SHOT_MISSED, 20.2, pid="home-7", shot_id="s2")]
    p = S.summarize_possession(evs, Team.HOME, "Harbor")
    assert p.actions[-1].startswith("the player who ended the possession took") and "blocked it" in p.actions[-1]
    assert p.actions.index("a defender blocked a shot") < p.actions.index("a teammate grabbed an offensive rebound")
    sit = S.GameSituation(home="Harbor", away="Summit", home_score=10, away_score=8, period=1, clock_s=500.0)
    st = S.b1_possession_end(sit, p, cfg=S.StateConfig(max_state_bytes={"B1": 560}))  # full state is ~650 B
    assert st["possession"]["actions"] == [p.actions[-1]]  # oldest dropped, terminal action kept intact


# --------------------------------------------------------------------------- client: queue, latency


async def test_queue_wait_timeout_does_not_charge_the_breaker() -> None:
    """Defect: a call that expired while queued behind max_in_flight counted as a bundle failure,
    so one bundle's burst could open another bundle's circuit while the service was healthy."""
    stub = StubAnswers(delay_s=0.6)
    config = JevConfig(mode="stub", max_in_flight=1, breaker_threshold=1, deadlines_s={"B1": 2.0, "B5": 0.15})
    client = JevClient(config, stub=stub)
    holder = asyncio.create_task(client.aask("B1", {"possession": {}}, Q.B1_POSSESSION_END))
    await asyncio.sleep(0.05)  # B1 now holds the only slot
    r = await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)
    assert r.error == "queue_timeout" and client.breaker("B5").state == "closed"
    assert (await holder).ok
    assert (await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME, tag="x")).error == "deadline"  # in flight
    assert client.breaker("B5").state == "open"
    await client.aclose()


def test_stats_availability_matches_the_property(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    client = JevClient(JevConfig(mode="live"))
    assert client.stats()["available"] is False and client.available is False
    assert client.stats()["disabled_reason"] == "client_unavailable"


async def test_deadline_overruns_are_in_the_latency_percentiles() -> None:
    """Defect: p50/p95 covered answered calls only, hiding the tail that deadlines are reset from."""
    client = JevClient(JevConfig(mode="stub", deadlines_s={"B5": 0.1}), stub=StubAnswers(delay_s=1.0))
    await client.aask("B5", B5_STATE, Q.B5_SHOT_OUTCOME)
    b5 = client.stats()["bundles"]["B5"]
    assert b5["answered"] == 0 and b5["p50_ms"] is not None and b5["p50_ms"] >= 90
    await client.aclose()


# --------------------------------------------------------------------------- decoders


def _stakes(probs: list[float]):
    client = JevClient(JevConfig(mode="stub"), stub=StubAnswers(overrides={"moment_stakes": stub_score(probs)}))
    r = client.ask("B1", {"x": "y"}, {"moment_stakes": Q.B1_POSSESSION_END["moment_stakes"]})
    client.close()
    return decode_stakes(r, Thresholds())


def test_stakes_level_rounds_half_up() -> None:
    """Defect: round() is half-to-even, so E[level] 2.5 -> 2 but 1.5 -> 2 and 0.5 -> 0."""
    assert _stakes([0.0, 0.0, 0.5, 0.5]).label == "decisive"
    assert _stakes([0.5, 0.5, 0.0, 0.0]).label == "notable"


# --------------------------------------------------------------------------- B2 material


async def test_b2_offers_only_kinds_that_have_material() -> None:
    """Defect: kinds without material were offered to Jev (design §6.2 removes them from the options)."""
    stub = StubAnswers()
    config = JevConfig(mode="stub")
    d = JevDecider(config, JevClient(config, stub=stub))
    sit = S.GameSituation(home="Harbor", away="Summit", home_score=50, away_score=50, period=2, clock_s=200.0)
    st = S.b2_dead_ball_fan(sit, stoppage="a timeout", material={"tendency": "Summit doubles the post"})
    await d.next_line_kind(st, {"stat_note", "tendency"}, [])
    assert list(stub.last_body["questions"]["next_line_kind"]["criteria"]) == ["tendency", "silence"]
    bare = S.b2_dead_ball_fan(sit, stoppage="a timeout")
    hits = stub.hits
    kind, _ = await d.next_line_kind(bare, {"stat_note"}, [])
    assert stub.hits == hits and kind.source == "fallback"  # nothing Jev could choose besides silence
    await d.aclose()


# --------------------------------------------------------------------------- questions stay verbatim


def test_questions_are_verbatim_from_the_design_note() -> None:
    """The spec requires §5 verbatim: rebuild the design note's code block and compare wire forms."""
    src = (ROOT / "docs/research/jev_integration_design.md").read_text(encoding="utf-8")
    sec = src[src.index("## 5. The questions we ship (verbatim)"):]
    code = sec[sec.index("```python") + len("```python"):]
    ns: dict = {}
    exec(code[: code.index("```")], ns)  # noqa: S102 - our own design note, typesafe_sdk constructors only
    from courtvision.decision.client import wire_questions

    def same(a: dict, b: dict) -> bool:
        wa, wb = wire_questions(a), wire_questions(b)
        return json.dumps(wa) == json.dumps(wb)  # order-sensitive: option order matters to Jev

    for name in ("B1_POSSESSION_END", "B5_SHOT_OUTCOME", "B5_RELEASE", "B6_RIM_FRAMES", "B6_SETUP"):
        assert same(ns[name], getattr(Q, name)), name
    for kinds in (set(), {"stat_note"}, {"stat_note", "tendency", "storyline"}):
        for n in range(4):
            assert same(ns["b2_dead_ball_fan"](kinds, n), Q.b2_dead_ball_fan(kinds, n))
    for n in range(1, len(Q.CARD_KEYS) + 1):
        cards = {k: "t" for k in Q.CARD_KEYS[:n]}
        assert same(ns["b3_dead_ball_coach"](cards), Q.b3_dead_ball_coach(cards))
    for n in (1, 2):
        assert same(ns["b4_line_guard"](n), Q.b4_line_guard(n))
    assert not re.search(r"\d", "".join(Q.LINE_KINDS.values()))
    assert stub_noul(0.5)["type"] == "noul"
