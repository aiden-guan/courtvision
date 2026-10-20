"""Regression tests for the integration-boundary fixes (each names the defect it pins down)."""

from __future__ import annotations

import asyncio
import time

import pytest

from courtvision.decision.client import JevConfig
from courtvision.geometry.court import court_for
from courtvision.narration.facts import pseudonyms_for, trust_map
from courtvision.narration.narrator import NarrationConfig, Narrator
from courtvision.narration.pseudonym import PseudonymMap
from courtvision.narration.tts import NullTTS
from courtvision.pipeline import (
    EventEnricher,
    MuteableTTS,
    Pacer,
    VideoTimeTTS,
    VirtualClock,
    build_jev,
    cue_priority_for_insight,
    insight_order,
)
from courtvision.stats.insights import InsightConfig
from courtvision.types import Event, EventKind, FrameState, Insight, PlayerState, Roster, RosterEntry, Team

COURT = court_for("nba")


def _roster() -> Roster:
    return Roster([RosterEntry("home-23", Team.HOME, "23", "Ada Brindlecombe"),
                   RosterEntry("home-5", Team.HOME, "5", "Cyrus Dellacourt"),
                   RosterEntry("away-11", Team.AWAY, "11", "Ezra Farrowby")],
                  {Team.HOME: "Riverside", Team.AWAY: "Lakeport"})


def _calibrated(kind: EventKind, eid: str, pid: str, value: int, **attrs) -> Event:
    return Event(eid, kind, 1.0, 30, team=Team.HOME, player_id=pid, value=value, shot_id="s0001", confidence=0.99,
                 attrs={"p_status": "platt", **attrs})


# --------------------------------------------------------------------------- one PseudonymMap per game


def test_narrator_uses_the_shared_pseudonym_map() -> None:
    """Defect: the narrator built a private PseudonymMap, so any other packet builder using
    ``pseudonyms_for(roster)`` issued different tokens (provisional ids especially)."""
    roster = _roster()
    shared = pseudonyms_for(roster, seed=0)
    nar = Narrator(roster, COURT, NarrationConfig(tts="none", llm_enabled=False), pseudo=shared)
    assert nar.pseudo is shared
    ev = Event("e1", EventKind.REBOUND, 1.0, 30, team=Team.HOME, player_id="home-t12", source="manual",
               attrs={"offensive": False})
    facts = nar._facts([ev])
    assert facts["events"][0]["actor"] == shared.token("home-t12")  # lazily issued provisional token agrees
    # standalone narrators keep a private map (unchanged behaviour)
    alone = Narrator(roster, COURT, NarrationConfig(tts="none", llm_enabled=False))
    assert isinstance(alone.pseudo, PseudonymMap) and alone.pseudo is not shared


# --------------------------------------------------------------------------- P(3) gate


def test_made_three_inherits_p_three_from_its_earlier_attempt() -> None:
    """Defect: the engine puts ``p_three`` on the SHOT_ATTEMPT only and the narrator gets the make in a
    later batch, so ``trust_map([made])`` skipped the 0.97 three gate."""
    enricher = EventEnricher(trusted_identity=True)
    att = _calibrated(EventKind.SHOT_ATTEMPT, "e1", "home-5", 3, p_three=0.80)
    made = _calibrated(EventKind.SHOT_MADE, "e2", "home-5", 3)
    assert trust_map([made])["e2"] is True  # the gap: alone, the make passes
    enricher.enrich(att)
    made_e = enricher.enrich(made)
    assert made_e.attrs["p_three"] == 0.80 and "p_three" not in made.attrs  # engine object untouched
    assert trust_map([made_e])["e2"] is False
    sure = EventEnricher(trusted_identity=True)
    sure.enrich(_calibrated(EventKind.SHOT_ATTEMPT, "e1", "home-5", 3, p_three=0.99))
    assert trust_map([sure.enrich(made)])["e2"] is True


# --------------------------------------------------------------------------- identity tiers


def test_identity_confidences_drive_the_spoken_name() -> None:
    """Defect: events carried no ``id_conf`` / ``jersey_conf``, so the renderer treated every video
    identity as certain (missing => 1.0) and spoke names below the 0.95 gate."""
    roster = _roster()
    nar = Narrator(roster, COURT, NarrationConfig(tts="none", llm_enabled=False), pseudo=PseudonymMap(roster))
    enricher = EventEnricher(trusted_identity=False)
    frame = FrameState(0, 1.0, [PlayerState("home-23", Team.HOME, 10.0, 20.0, confidence=0.90),
                                PlayerState("home-5", Team.HOME, 30.0, 20.0, confidence=0.97)], None)
    enricher.observe(frame, {"home-23": 0.97, "home-5": 1.0})

    def spoken(pid: str, source: str = "cv") -> str:
        ev = enricher.enrich(Event(f"x-{pid}-{source}", EventKind.REBOUND, 1.0, 30, team=Team.HOME, player_id=pid,
                                   source=source, attrs={"offensive": False}))
        return nar._render("{" + nar.pseudo.token(pid) + "} grabs it.", [ev])

    assert spoken("home-5") == "Cyrus Dellacourt grabs it."  # id 0.97 >= 0.95: name
    assert spoken("home-23") == "Riverside's number 23 grabs it."  # 0.80 <= id < 0.95 and jersey >= 0.95
    assert spoken("away-11") == "Lakeport grabs it."  # never seen by perception: team only
    assert spoken("home-23", source="manual") == "Ada Brindlecombe grabs it."  # operator identity is certain
    two = enricher.enrich(Event("ast", EventKind.ASSIST, 1.0, 30, team=Team.HOME, player_id="home-5",
                                other_player_id="home-23"))
    assert two.attrs["id_conf"] == 0.90  # one value covers both actors: the lower one


# --------------------------------------------------------------------------- priorities


async def test_insight_priority_direction_is_consistent_across_consumers() -> None:
    """Insight.priority is higher = more important in stats.insights, JevDecider.pick_insight and the
    narrator's coach board; Cue.priority is 0 = highest. Pin both conventions at the boundary."""
    cards = [Insight("i1", 10.0, "shot_diet", "{team} 12 of 30 from three.", priority=20, team=Team.AWAY),
             Insight("i2", 10.0, "run", "{team} 9-0 run.", priority=80, team=Team.HOME),
             Insight("i3", 10.0, "turnovers", "{team} 4 turnovers in 6 possessions.", priority=60, team=Team.AWAY)]
    assert [i.insight_id for i in insight_order(cards)] == ["i2", "i3", "i1"]
    jev = build_jev(JevConfig(mode="off"), {})
    assert (await jev.pick_insight(cards, {})).insight_id == "i2"  # fallback = highest priority
    nar = Narrator(_roster(), COURT, NarrationConfig(mode="coach", tts="none", llm_enabled=False))
    await nar.on_insights(cards, None)  # type: ignore[arg-type]
    assert [u.text for u in nar.coach_cards][0] == "Riverside 9-0 run."
    await nar.aclose()
    pri = InsightConfig().priorities
    assert pri["foul_trouble"] > pri["run"] > pri["shot_diet"]  # higher = more important
    levels = {k: cue_priority_for_insight(v) for k, v in pri.items()}
    assert levels["foul_trouble"] == levels["run"] == levels["team_fouls"] == 2 and levels["shot_diet"] == 3
    assert all(lv in (2, 3) for lv in levels.values())  # cards never become P0 / P1 live calls


# --------------------------------------------------------------------------- pacing and TTS helpers


async def test_pacer_paces_pauses_and_reanchors() -> None:
    off = Pacer(enabled=False)
    t0 = time.perf_counter()
    for i in range(50):
        assert await off.wait(i / 30.0) == 0.0
    assert time.perf_counter() - t0 < 0.05
    p = Pacer(enabled=True, speed=10.0)
    t0 = time.perf_counter()
    for i in range(16):
        await p.wait(i / 30.0)  # 0.5 s of source at 10x
    assert 0.04 <= time.perf_counter() - t0 < 0.3
    p.pause()
    waiter = asyncio.create_task(p.wait(0.6))
    await asyncio.sleep(0.1)
    assert not waiter.done()
    p.resume()
    await asyncio.wait_for(waiter, 1.0)
    t1 = time.perf_counter()
    await p.wait(0.6 + 1.0)  # re-anchored at resume: one source second at 10x, not a burst
    assert time.perf_counter() - t1 >= 0.08
    p.set_speed(100.0)
    t2 = time.perf_counter()
    await p.wait(2.6)
    assert time.perf_counter() - t2 < 0.05
    with pytest.raises(ValueError):
        p.set_speed(0.0)


async def test_video_time_tts_occupies_source_time() -> None:
    clock = VirtualClock(10.0)
    tts = VideoTimeTTS(clock)
    task = asyncio.create_task(tts.speak("one two three four five six seven", 1))
    for _ in range(5):
        await asyncio.sleep(0)
    assert not task.done() and tts.speaking
    while not task.done():
        clock.advance(0.1)
        await asyncio.sleep(0)
    assert clock() >= 12.0 and not tts.spoken[0].interrupted  # 7 words at 3.5 words/s
    task = asyncio.create_task(tts.speak("a long line that will be cut", 1))
    await asyncio.sleep(0)
    await tts.stop()
    await asyncio.wait_for(task, 1.0)
    assert tts.spoken[1].interrupted


async def test_muteable_tts_routes_muted_lines_to_the_silent_engine() -> None:
    voice, silent = NullTTS(time_scale=0.0), NullTTS(time_scale=0.0)
    tts = MuteableTTS(voice, silent)
    await tts.speak("first", 1, prosody="up")
    await tts.set_muted(True)
    await tts.speak("second", 1)
    await tts.set_muted(False)
    await tts.speak("third", 1)
    assert voice.texts == ["first", "third"] and silent.texts == ["second"]
    assert voice.spoken[0].prosody == "up"
