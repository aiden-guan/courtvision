"""Narrator flows with fake LLM / Jev / TTS: live calls, hedging, LLM colour, preemption, corrections, modes."""

from __future__ import annotations

import asyncio

import pytest

from courtvision.events.deadball import DeadBallDetector
from courtvision.narration.captions import CaptionEmitter, CaptionConfig, WebVTTWriter
from courtvision.narration.narrator import DeadBallInfo, NarrationConfig, Narrator
from courtvision.narration.validator import OUTCOME_RE
from courtvision.types import EventKind, EventStatus, Insight, Team
from tests.test_narration_io_support import (COURT, FakeJev, FakeLLM, ev, made, make_rig, make_roster, revise)

TIMEOUT = DeadBallInfo(True, since_t=299.5, expected_window_s=60.0, stoppage="timeout")
FREE_THROW = DeadBallInfo(True, since_t=299.5, expected_window_s=10.0, stoppage="free_throw")


async def _finish(rig, timeout: float = 3.0) -> None:  # noqa: ANN001
    assert await rig.narrator.drain(timeout)


def _lines(rig, channel: str | None = None) -> list[str]:  # noqa: ANN001
    return [u.text for u in rig.utterances if channel is None or u.channel == channel]


async def test_made_shot_template_line_fast_with_name() -> None:
    rig = make_rig()
    await rig.narrator.on_events([made(3)], rig.stats, None)
    await _finish(rig)
    [u] = rig.utterances
    assert u.channel == "play_by_play" and u.source == "template" and u.validated
    assert "Riley Thorsen" in u.text and "{P" not in u.text  # tokens rendered locally
    assert u.latency_ms is not None and u.latency_ms < 500  # 0.2 s confirmation hold + loop tick
    assert rig.tts.texts == [u.text] and rig.narrator.transcript == [u]
    await rig.narrator.aclose()


async def test_uncalibrated_make_is_hedged_until_confirmed() -> None:
    rig = make_rig()
    shot = made(3, source="cv", status=EventStatus.PROVISIONAL, confidence=0.99, attrs={"p_status": "uncalibrated"})
    await rig.narrator.on_events([shot], rig.stats, None)
    await asyncio.sleep(0.4)  # hedged line spoken; the outcome call waits for confirmation
    [hedged] = rig.utterances
    assert not OUTCOME_RE.search(hedged.text.replace("Riley Thorsen", "")), hedged.text  # no outcome, no value
    await rig.narrator.on_events([revise(shot, status=EventStatus.CONFIRMED)], rig.stats, None)
    await _finish(rig)
    assert len(rig.utterances) == 2
    assert OUTCOME_RE.search(rig.utterances[1].text) and "Riley Thorsen" in rig.utterances[1].text
    await rig.narrator.aclose()


async def test_unconfirmed_make_goes_silent_after_the_wait() -> None:
    rig = make_rig(make_wait_s=0.3)
    attempt = ev(EventKind.SHOT_ATTEMPT, team=Team.HOME, player_id="home-23", value=3, shot_id="s9001",
                 x=30.0, y=8.0, source="cv", status=EventStatus.PROVISIONAL, attrs={"p_status": "uncalibrated"})
    await rig.narrator.on_events([attempt], rig.stats, None)
    shot = made(3, shot_id="s9001", source="cv", status=EventStatus.PROVISIONAL, attrs={"p_status": "uncalibrated"})
    await rig.narrator.on_events([shot], rig.stats, None)
    await _finish(rig)
    assert len(rig.utterances) == 1  # only the release call; the hedge would repeat it, the outcome never came
    assert not OUTCOME_RE.search(rig.utterances[0].text.replace("Riley Thorsen", ""))
    await rig.narrator.aclose()


async def test_calibrated_make_below_the_band_is_silent() -> None:
    rig = make_rig()
    shot = made(2, source="cv", status=EventStatus.PROVISIONAL, confidence=0.6, attrs={"p_status": "platt"})
    await rig.narrator.on_events([shot], rig.stats, None)
    await _finish(rig)
    assert rig.utterances == []
    await rig.narrator.aclose()


def _llm_reply(rig):  # noqa: ANN001
    tok = rig.narrator.pseudo.token("home-23")
    return tok, (lambda prompt: f"{{{tok}}} has 12 points tonight.")


async def test_llm_candidate_spoken_with_names_rendered() -> None:
    jev = FakeJev()
    rig = make_rig(llm=FakeLLM(), jev=jev)
    tok, reply = _llm_reply(rig)
    rig.narrator.llm.reply = reply  # type: ignore[union-attr]
    await rig.narrator.on_events([made(3)], rig.stats, None)
    await _finish(rig)
    await rig.narrator.tick(300.0, rig.stats, TIMEOUT)
    await asyncio.sleep(0.2)
    await rig.narrator.tick(300.2, rig.stats, TIMEOUT)
    await _finish(rig)
    colour = [u for u in rig.utterances if u.channel == "color"]
    assert colour and colour[0].source == "llm"
    assert colour[0].text == "Riley Thorsen has 12 points tonight."
    assert jev.guard_calls and jev.guard_calls[0][1] == [f"{{{tok}}} has 12 points tonight."]
    assert jev.b2_calls, "dead-ball line kind asked of Jev (B2)"
    await rig.narrator.aclose()


@pytest.mark.parametrize("bad", ["validator", "guard"])
async def test_failed_llm_line_falls_back_to_template(bad: str) -> None:
    jev = FakeJev(guard_pass=bad != "guard")
    rig = make_rig(llm=FakeLLM(), jev=jev)
    tok = rig.narrator.pseudo.token("home-23")
    rig.narrator.llm.reply = (f"{{{tok}}} has 99 points tonight." if bad == "validator"  # type: ignore[union-attr]
                              else f"{{{tok}}} has 12 points tonight.")
    await rig.narrator.on_events([made(3)], rig.stats, None)
    await _finish(rig)
    for i in range(6):
        await rig.narrator.tick(300.0 + i * 0.05, rig.stats, TIMEOUT)
        await asyncio.sleep(0.05)
    await _finish(rig)
    colour = [u for u in rig.utterances if u.channel == "color"]
    assert colour, "a template colour line replaces the rejected LLM line"
    assert all(u.source == "template" for u in colour)
    assert "99" not in " ".join(u.text for u in rig.utterances)
    key = "rejected:validator" if bad == "validator" else "rejected:guard"
    assert rig.narrator.pool is not None and rig.narrator.pool.counts[key] == 2
    await rig.narrator.aclose()


async def test_short_window_uses_templates_only() -> None:
    llm = FakeLLM("Riverside leads.")
    rig = make_rig(llm=llm, jev=FakeJev())
    await rig.narrator.on_events([made(3)], rig.stats, None)
    await _finish(rig)
    await rig.narrator.tick(300.0, rig.stats, FREE_THROW)
    await _finish(rig)
    assert llm.prompts == []  # 10 s < p95 12 + tts 10 + 1
    assert [u.source for u in rig.utterances if u.channel == "color"] == ["template"]
    await rig.narrator.aclose()


async def test_stale_candidate_dropped_on_state_change() -> None:
    llm = FakeLLM(delay_s=0.4)
    rig = make_rig(llm=llm, jev=FakeJev())
    _tok, reply = _llm_reply(rig)
    llm.reply = reply
    await rig.narrator.on_events([made(3)], rig.stats, None)
    await _finish(rig)
    await rig.narrator.tick(300.0, rig.stats, TIMEOUT)
    await asyncio.sleep(0.05)
    sub = ev(EventKind.SUBSTITUTION, team=Team.AWAY, player_id="away-5", other_player_id="away-0", t=300.1,
             source="manual")
    await rig.narrator.on_events([sub], rig.stats, None)  # the fact store moved: pending candidates are stale
    await asyncio.sleep(0.6)
    await rig.narrator.tick(300.8, rig.stats, TIMEOUT)
    await _finish(rig)
    assert not [u for u in rig.utterances if u.source == "llm"]
    pool = rig.narrator.pool
    assert pool is not None and pool.counts["dropped:stale"] >= 2 and pool.ready() == []
    await rig.narrator.aclose()


async def test_higher_priority_cue_preempts_colour() -> None:
    rig = make_rig(time_scale=1.0)
    rig.stats.run = {"team": Team.HOME, "points": 9, "opp_points": 0, "since_t": 250.0}
    await rig.narrator.on_events([ev(EventKind.TIMEOUT, team=Team.AWAY, source="manual", t=299.0)], rig.stats, None)
    await _finish(rig)
    await rig.narrator.tick(299.6, rig.stats, TIMEOUT)
    for _ in range(50):
        await asyncio.sleep(0.02)
        if rig.narrator.director.current is not None and rig.narrator.director.current.priority == 3:
            break
    assert rig.narrator.director.current is not None and rig.narrator.director.current.priority == 3
    await rig.narrator.on_events([made(3, t=300.0)], rig.stats, None)
    await _finish(rig, 6.0)
    colour = next(s for s in rig.tts.spoken if s.priority == 0)
    assert colour.interrupted, "the colour line was cut"
    assert "Riley Thorsen" in rig.tts.spoken[-1].text and not rig.tts.spoken[-1].interrupted
    await rig.narrator.aclose()


async def test_correction_after_retraction_of_a_spoken_basket() -> None:
    rig = make_rig()
    shot = made(3)
    await rig.narrator.on_events([shot], rig.stats, None)
    await _finish(rig)
    rig.stats.score = {Team.HOME: 39, Team.AWAY: 40}
    await rig.narrator.on_events([revise(shot, status=EventStatus.RETRACTED)], rig.stats, None)
    await _finish(rig)
    corr = [u for u in rig.utterances if u.channel == "correction"]
    assert len(corr) == 1 and corr[0].text.startswith("Correction:") and corr[0].priority == 3
    # never twice for the same event
    await rig.narrator.on_events([revise(revise(shot, status=EventStatus.RETRACTED), status=EventStatus.RETRACTED)],
                                 rig.stats, None)
    await _finish(rig)
    assert len([u for u in rig.utterances if u.channel == "correction"]) == 1
    await rig.narrator.aclose()


async def test_no_correction_when_only_a_hedge_was_spoken() -> None:
    rig = make_rig(make_wait_s=0.2)
    shot = made(3, source="cv", status=EventStatus.PROVISIONAL, attrs={"p_status": "uncalibrated"})
    await rig.narrator.on_events([shot], rig.stats, None)
    await _finish(rig)
    assert len(rig.utterances) == 1
    await rig.narrator.on_events([revise(shot, status=EventStatus.RETRACTED)], rig.stats, None)
    await _finish(rig)
    assert [u.channel for u in rig.utterances] == ["play_by_play"]
    await rig.narrator.aclose()


async def test_revision_before_speech_is_silent() -> None:
    rig = make_rig(hold_s={"shot_made": 0.3})
    shot = made(3)
    await rig.narrator.on_events([shot], rig.stats, None)
    await rig.narrator.on_events([revise(shot, status=EventStatus.RETRACTED)], rig.stats, None)
    await _finish(rig)
    assert rig.utterances == []
    await rig.narrator.aclose()


async def test_coach_mode_cards() -> None:
    jev = FakeJev()
    rig = make_rig("coach", jev=jev)
    cards = [
        Insight("i0001", 300.0, "run", "{team} 9-0 run, 2:41 left in Q2.", priority=80, team=Team.HOME),
        Insight("i0002", 300.0, "hot_hand", "{player} 5 of 6 from the field, n = 6.", priority=35,
                team=Team.HOME, player_id="home-23"),
        Insight("i0003", 300.0, "turnovers", "{team} 4 turnovers in 6 possessions.", priority=60, team=Team.AWAY),
        Insight("i0004", 300.0, "shot_diet", "{team} 12 of 30 shots from three.", priority=20, team=Team.AWAY),
        Insight("i0005", 300.0, "foul_trouble", "{player} 3 fouls in Q2 (Q+1).", priority=90,
                team=Team.AWAY, player_id="away-11"),
    ]
    await rig.narrator.on_insights(cards, rig.stats)
    await _finish(rig)
    board = rig.narrator.coach_cards
    assert len(board) == 3 and all(u.channel == "coach" for u in board)
    assert all(len(u.text.split()) <= 12 for u in rig.utterances)
    assert board[0].text.startswith("Emerson Duquesne")  # Q+1 foul alert posts immediately, on top
    sent = jev.pick_calls[0][0]
    assert not any(e.name in c.text for c in sent for e in make_roster().entries)  # Jev sees tokens only
    assert any("{P" in c.text for c in sent)
    assert board[1].text.startswith("Lakeport 12 of 30")  # Jev's pick (lowest priority here) ranks first
    assert rig.tts.spoken == []  # cards are visual
    await rig.narrator.aclose()


async def test_coach_mode_without_jev_ranks_by_priority() -> None:
    rig = make_rig("coach")
    cards = [Insight("i1", 1.0, "shot_diet", "{team} 12 of 30 from three.", priority=20, team=Team.AWAY),
             Insight("i2", 1.0, "run", "{team} 9-0 run.", priority=80, team=Team.HOME)]
    await rig.narrator.on_insights(cards, rig.stats)
    await _finish(rig)
    assert [u.text for u in rig.narrator.coach_cards] == ["Riverside 9-0 run.", "Lakeport 12 of 30 from three."]
    await rig.narrator.aclose()


async def test_accessibility_mode_lines_and_score() -> None:
    rig = make_rig("accessibility")
    poss = ev(EventKind.POSSESSION, team=Team.HOME, player_id="home-3", attrs={"cause": "inbound"}, t=290.0)
    att = ev(EventKind.SHOT_ATTEMPT, team=Team.HOME, player_id="home-23", value=3, shot_id="s7001", x=30.0, y=8.0,
             attrs={"shot_type": "jumper", "basket": "left"}, t=298.0)
    await rig.narrator.on_events([poss], rig.stats, None)
    await rig.narrator.on_events([att], rig.stats, None)
    await rig.narrator.on_events([made(3, shot_id="s7001", t=299.0)], rig.stats, None)
    await _finish(rig)
    lines = _lines(rig, "accessibility")
    assert lines[0] == "Riverside ball."
    assert lines[1].startswith("Riley Thorsen shot")
    assert "good" in lines[2] and lines[2].endswith("Riverside 42, Lakeport 40.")  # score after the possession
    await rig.narrator.aclose()


async def test_captions_keyed_to_source_pts() -> None:
    vtt = WebVTTWriter(map_pts_s=10.0)
    captions = CaptionEmitter([vtt], CaptionConfig(delay_s=1.0, pts_offset_s=10.0), wall_clock=lambda: 1_800_000_000.0)
    rig = make_rig(captions=captions)
    await rig.narrator.on_events([made(3, t=299.0)], rig.stats, None)
    await _finish(rig)
    [cue] = captions.cues
    assert cue.channel == "caption" and cue.pts == pytest.approx(309.0)
    assert cue.show_at == pytest.approx(310.0)  # pts(t_occ) + D: confirmation (t=300) is within D
    doc = vtt.render()
    assert "X-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:900000" in doc
    assert "00:04:59.000 --> " in doc and "Riley Thorsen" in doc
    await rig.narrator.aclose()


async def test_possession_end_stakes_become_prosody() -> None:
    jev = FakeJev(stakes="decisive")
    rig = make_rig(jev=jev)
    await rig.narrator.on_events([ev(EventKind.POSSESSION, team=Team.HOME, player_id="home-3",
                                     attrs={"cause": "inbound"}, t=280.0)], rig.stats, None)
    await rig.narrator.on_events([ev(EventKind.POSSESSION, team=Team.AWAY, player_id="away-0",
                                     attrs={"cause": "made_shot"}, t=290.0)], rig.stats, None)
    await _finish(rig)
    assert jev.b1_states and "situation" in jev.b1_states[0]
    await rig.narrator.on_events([made(3, t=299.0)], rig.stats, None)
    await _finish(rig)
    assert rig.tts.spoken[-1].prosody == "big"
    await rig.narrator.aclose()


async def test_real_deadball_detector_drives_colour() -> None:
    det = DeadBallDetector(COURT)
    rig = make_rig(dead_ball=det)
    rig.stats.run = {"team": Team.HOME, "points": 9, "opp_points": 0, "since_t": 250.0}
    await rig.narrator.on_events([ev(EventKind.TIMEOUT, team=Team.AWAY, source="manual", t=299.0)], rig.stats, None)
    await rig.narrator.tick(299.2, rig.stats)
    assert [u.channel for u in rig.utterances if u.channel == "color"] == []  # live ball: no colour
    det.mark_dead_ball(299.4, "timeout")
    await rig.narrator.tick(299.5, rig.stats)
    await _finish(rig)
    assert any(u.channel == "color" and "9-0" in u.text for u in rig.utterances)
    await rig.narrator.aclose()


async def test_template_only_without_consent() -> None:
    llm = FakeLLM("Riverside leads.")
    rig = make_rig(llm=llm, jev=FakeJev(), cloud_narration_ok=False)
    assert rig.narrator.pool is None
    await rig.narrator.on_events([made(3)], rig.stats, None)
    await rig.narrator.tick(300.0, rig.stats, TIMEOUT)
    await _finish(rig)
    assert llm.prompts == []
    await rig.narrator.aclose()


async def test_aclose_cancels_everything() -> None:
    rig = make_rig(llm=FakeLLM(delay_s=5.0), jev=FakeJev(), time_scale=1.0)
    await rig.narrator.on_events([made(3)], rig.stats, None)
    await rig.narrator.tick(300.0, rig.stats, TIMEOUT)
    await asyncio.sleep(0.05)
    await rig.narrator.aclose()
    others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert all(t.done() for t in others), others
    with pytest.raises(RuntimeError):
        rig.narrator.start()


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError):
        Narrator(make_roster(), COURT, NarrationConfig(mode="radio"))


async def test_late_assist_merges_into_the_held_make() -> None:
    rig = make_rig(hold_s={"shot_made": 0.3})
    shot = made(3)
    await rig.narrator.on_events([shot], rig.stats, None)
    assist = ev(EventKind.ASSIST, team=Team.HOME, player_id="home-3", other_player_id="home-23", shot_id=shot.shot_id)
    await rig.narrator.on_events([assist], rig.stats, None)
    await _finish(rig)
    [u] = rig.utterances  # one made call, not a separate assist call
    assert "Riley Thorsen" in u.text and set(u.event_ids) == {shot.event_id, assist.event_id}
    await rig.narrator.aclose()


async def test_release_call_needs_its_gate_when_calibrated() -> None:
    rig = make_rig()
    low = ev(EventKind.SHOT_ATTEMPT, team=Team.HOME, player_id="home-23", value=2, x=15.0, y=20.0, source="cv",
             status=EventStatus.PROVISIONAL, confidence=0.7, attrs={"p_status": "platt", "basket": "left"})
    await rig.narrator.on_events([low], rig.stats, None)
    await _finish(rig)
    assert rig.utterances == []  # P(shot) < 0.90: no release call
    high = ev(EventKind.SHOT_ATTEMPT, team=Team.HOME, player_id="home-23", value=2, x=15.0, y=20.0, source="cv",
              status=EventStatus.PROVISIONAL, confidence=0.97, attrs={"p_status": "platt", "basket": "left"})
    await rig.narrator.on_events([high], rig.stats, None)
    await _finish(rig)
    assert len(rig.utterances) == 1 and "Riley Thorsen" in rig.utterances[0].text
    await rig.narrator.aclose()


async def test_state_independent_storyline_fills_a_short_window() -> None:
    jev = FakeJev(line_kind="storyline")
    llm = FakeLLM()
    rig = make_rig(llm=llm, jev=jev, storylines=["Riley Thorsen is two points from 1,000 career points"])
    tok = rig.narrator.pseudo.token("home-23")
    llm.reply = f"{{{tok}}} is two points from 1,000 career points."
    await rig.narrator.tick(100.0, rig.stats, None)  # live ball: storyline candidates are pre-generated
    assert rig.narrator.pool is not None
    await rig.narrator.pool.wait(2.0)
    assert llm.prompts and not any(e.name in p for p in llm.prompts for e in make_roster().entries)
    await rig.narrator.on_events([made(2, t=101.0)], rig.stats, None)  # state changes; storylines survive
    await _finish(rig)
    await rig.narrator.tick(102.0, rig.stats, DeadBallInfo(True, since_t=102.0, expected_window_s=10.0,
                                                           stoppage="free_throw"))
    for _ in range(10):
        await asyncio.sleep(0.02)
        await rig.narrator.tick(102.1, rig.stats, DeadBallInfo(True, since_t=102.0, expected_window_s=10.0,
                                                               stoppage="free_throw"))
    await _finish(rig)
    colour = [u for u in rig.utterances if u.channel == "color"]
    assert [u.source for u in colour] == ["llm"]
    assert colour[0].text == "Riley Thorsen is two points from 1,000 career points."
    await rig.narrator.aclose()
