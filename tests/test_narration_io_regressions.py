"""Regression tests for defects found in the narration-io review (fake LLM / Jev / TTS; no network)."""

from __future__ import annotations

import asyncio
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest

from courtvision.events.deadball import DeadBallDetector
from courtvision.narration import captions as captions_mod
from courtvision.narration.captions import CaptionConfig, CaptionEmitter, WebVTTWriter
from courtvision.narration.narrator import DeadBallInfo, NarrationConfig, Narrator
from courtvision.narration.tts import AVSpeechTTS, TTSConfig
from courtvision.narration.validator import OUTCOME_RE
from courtvision.types import EventKind, EventStatus, Insight, Team
from tests.test_narration_io_support import COURT, FakeJev, FakeLLM, FakeStats, ev, made, make_rig, make_roster, revise


def _uncalibrated_miss(shot_id: str) -> Any:
    return ev(EventKind.SHOT_MISSED, team=Team.HOME, player_id="home-23", value=3, x=30.0, y=8.0, shot_id=shot_id,
              attrs={"shot_type": "jumper", "basket": "left", "p_status": "uncalibrated"},
              source="cv", status=EventStatus.PROVISIONAL)


def _flip_to_make(miss: Any) -> Any:
    return revise(miss, kind=EventKind.SHOT_MADE.value, status=EventStatus.CONFIRMED)


# --------------------------------------------------------------------------- TTS contract


class ContractTTS:
    """Exactly ARCHITECTURE's ``TTS``: ``speak(text, priority)`` and ``stop()``, no ``prosody``."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def speak(self, text: str, priority: int) -> None:
        self.texts.append(text)

    async def stop(self) -> None:
        return None


async def test_contract_tts_without_prosody_keyword_still_speaks() -> None:
    tts = ContractTTS()
    n = Narrator(make_roster(), COURT, NarrationConfig(tts="none", min_gap_s=0.0), tts=tts)
    await n.on_events([made(3)], FakeStats(), None)
    assert await n.drain(3.0)
    assert tts.texts and "Riley Thorsen" in tts.texts[0]
    assert [u.text for u in n.transcript] == tts.texts
    await n.aclose()


class StuckTTS:
    """A voice that ignores stop() (the line keeps going until cancelled)."""

    def __init__(self) -> None:
        self.cancelled = False

    async def speak(self, text: str, priority: int = 1, *, prosody: str | None = None) -> None:
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def stop(self) -> None:
        return None


async def test_aclose_while_speaking_cancels_the_voice_task() -> None:
    tts = StuckTTS()
    n = Narrator(make_roster(), COURT, NarrationConfig(tts="none", min_gap_s=0.0), tts=tts)
    await n.on_events([made(3)], FakeStats(), None)
    for _ in range(50):
        await asyncio.sleep(0.02)
        if n.director.current is not None:
            break
    assert n.director.current is not None, "the line is playing"
    await n.aclose()
    others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert others == [] and tts.cancelled


async def test_no_background_work_after_aclose() -> None:
    jev = FakeJev()
    rig = make_rig(jev=jev)
    await rig.narrator.aclose()
    await rig.narrator.on_events([ev(EventKind.POSSESSION, team=Team.HOME, player_id="home-3",
                                     attrs={"cause": "inbound"}, t=280.0)], rig.stats, None)
    await rig.narrator.on_events([ev(EventKind.POSSESSION, team=Team.AWAY, player_id="away-0",
                                     attrs={"cause": "made_shot"}, t=290.0)], rig.stats, None)
    await rig.narrator.tick(291.0, rig.stats, DeadBallInfo(True, 291.0, 60.0, "timeout"))
    others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert others == [] and jev.b1_states == [] and rig.utterances == []


# --------------------------------------------------------------------------- corrections vs hedges


async def test_no_correction_when_the_hedge_is_still_playing() -> None:
    rig = make_rig(time_scale=1.0, make_wait_s=0.2)
    shot = made(3, source="cv", status=EventStatus.PROVISIONAL, attrs={"p_status": "uncalibrated"})
    await rig.narrator.on_events([shot], rig.stats, None)
    for _ in range(50):
        await asyncio.sleep(0.02)
        if rig.narrator.director.current is not None:
            break
    assert rig.narrator.director.current is not None and rig.narrator.director.current.hedge
    await rig.narrator.on_events([revise(shot, status=EventStatus.RETRACTED)], rig.stats, None)
    await rig.narrator.drain(5.0)
    assert [u.channel for u in rig.utterances] == ["play_by_play"], [u.text for u in rig.utterances]
    await rig.narrator.aclose()


async def test_correction_when_the_stated_line_is_still_playing() -> None:
    rig = make_rig(time_scale=1.0)
    shot = made(3)
    await rig.narrator.on_events([shot], rig.stats, None)
    for _ in range(50):
        await asyncio.sleep(0.02)
        if rig.narrator.director.current is not None:
            break
    assert rig.narrator.director.current is not None and not rig.narrator.director.current.hedge
    rig.stats.score = {Team.HOME: 39, Team.AWAY: 40}
    await rig.narrator.on_events([revise(shot, status=EventStatus.RETRACTED)], rig.stats, None)
    await rig.narrator.drain(6.0)
    assert [u.channel for u in rig.utterances] == ["play_by_play", "correction"]
    await rig.narrator.aclose()


@pytest.mark.parametrize("mode, channel", [("broadcast", "play_by_play"), ("accessibility", "accessibility")])
async def test_hedged_miss_flipped_to_a_make_gets_the_outcome_call(mode: str, channel: str) -> None:
    rig = make_rig(mode)
    miss = _uncalibrated_miss("s5101")
    await rig.narrator.on_events([miss], rig.stats, None)
    assert await rig.narrator.drain(3.0)
    [hedge] = rig.utterances
    assert not OUTCOME_RE.search(hedge.text.replace("Riley Thorsen", ""))
    await rig.narrator.on_events([_flip_to_make(miss)], rig.stats, None)  # scoreboard: it went in
    assert await rig.narrator.drain(3.0)
    assert [u.channel for u in rig.utterances] == [channel, channel]  # an outcome call, not a "Correction:"
    call = rig.utterances[1].text
    assert "good" in call if mode == "accessibility" else OUTCOME_RE.search(call), call
    await rig.narrator.aclose()


async def test_flip_confirmed_after_the_make_wait_stays_silent() -> None:
    rig = make_rig(make_wait_s=0.2)
    miss = _uncalibrated_miss("s5102")
    await rig.narrator.on_events([miss], rig.stats, None)
    assert await rig.narrator.drain(3.0)
    await asyncio.sleep(0.3)
    await rig.narrator.on_events([_flip_to_make(miss)], rig.stats, None)
    assert await rig.narrator.drain(3.0)
    assert len(rig.utterances) == 1  # too late to call it; no correction for a hedge either
    await rig.narrator.aclose()


async def _queue_behind_a_timeout_line(rig: Any) -> None:
    """Speak a line, so the director's 1.5 s minimum gap keeps the next call queued."""
    await rig.narrator.on_events([ev(EventKind.TIMEOUT, team=Team.AWAY, source="manual", t=280.0)], rig.stats, None)
    await asyncio.sleep(0.1)


@pytest.mark.parametrize("case", ["flip", "scorer"])
async def test_revision_of_a_queued_basket_call_is_re_rendered(case: str) -> None:
    rig = make_rig(min_gap_s=1.5, hold_s={"shot_missed": 0.0, "shot_made": 0.0})
    await _queue_behind_a_timeout_line(rig)
    first = _uncalibrated_miss("s6001") if case == "flip" else made(2, t=299.0)
    await rig.narrator.on_events([first], rig.stats, None)
    await asyncio.sleep(0.1)
    assert len(rig.narrator.director.pending()) == 1  # the call is queued, not yet heard
    new = _flip_to_make(first) if case == "flip" else revise(first, player_id="home-3")
    await rig.narrator.on_events([new], rig.stats, None)
    assert await rig.narrator.drain(4.0)
    calls = [u.text for u in rig.utterances[1:]]
    assert len(calls) == 1, calls  # the basket is called, once
    assert first.event_id in rig.narrator._unhedged  # ...as an outcome call (made family), not a hedge
    assert ("Riley Thorsen" if case == "flip" else "Jordan Okafor") in calls[0]  # from the new version
    assert not any(u.channel == "correction" for u in rig.utterances)
    await rig.narrator.aclose()


# --------------------------------------------------------------------------- dead-ball window upgrade


async def test_window_relabelled_as_timeout_gets_claude_colour() -> None:
    det = DeadBallDetector(COURT)
    llm = FakeLLM()
    jev = FakeJev()
    rig = make_rig(llm=llm, jev=jev, dead_ball=det)
    tok = rig.narrator.pseudo.token("home-23")
    llm.reply = f"{{{tok}}} has 12 points tonight."
    await rig.narrator.on_events([made(3)], rig.stats, None)
    assert await rig.narrator.drain(3.0)
    det.mark_dead_ball(299.5, "unknown")  # stillness-style stoppage: ~5 s expected, too short for Claude
    await rig.narrator.tick(300.0, rig.stats)
    assert llm.prompts == []
    det.note_stoppage(300.4, "timeout")  # same dead ball, now known to be a timeout
    await rig.narrator.tick(300.5, rig.stats)
    assert rig.narrator.pool is not None and rig.narrator.pool.pending() == 2, "the upgrade requests the pair"
    for i in range(10):
        await asyncio.sleep(0.03)
        await rig.narrator.tick(300.6 + i * 0.03, rig.stats)
    assert await rig.narrator.drain(3.0)
    assert len(llm.prompts) == 2
    colour = [u for u in rig.utterances if u.channel == "color"]
    assert colour and colour[0].source == "llm" and colour[0].text == "Riley Thorsen has 12 points tonight."
    await rig.narrator.aclose()


async def test_window_upgrade_happens_once() -> None:
    jev = FakeJev()
    rig = make_rig(jev=jev)
    await rig.narrator.on_events([made(3)], rig.stats, None)
    assert await rig.narrator.drain(3.0)
    await rig.narrator.tick(300.0, rig.stats, DeadBallInfo(True, 299.5, 3.0, "unknown"))
    for i in range(5):
        await rig.narrator.tick(300.1 + i * 0.1, rig.stats, DeadBallInfo(True, 299.5, 30.0, "timeout"))
    await asyncio.sleep(0.05)
    assert len(jev.b2_calls) == 1  # the 3 s window asked nothing; the upgrade asked once
    await rig.narrator.aclose()


async def _free_throw_window(rig: Any, t: float) -> None:
    await rig.narrator.tick(t, rig.stats, None)  # live ball: storyline candidates are stocked
    await asyncio.sleep(0.1)
    await rig.narrator.on_events([made(2, t=t + 1)], rig.stats, None)
    assert await rig.narrator.drain(3.0)
    fw = DeadBallInfo(True, since_t=t + 2, expected_window_s=10.0, stoppage="free_throw")
    for i in range(8):
        await rig.narrator.tick(t + 2 + i * 0.05, rig.stats, fw)
        await asyncio.sleep(0.03)
    assert await rig.narrator.drain(3.0)
    await rig.narrator.tick(t + 3, rig.stats, None)


async def test_storyline_is_not_repeated_in_every_short_window() -> None:
    llm = FakeLLM()
    rig = make_rig(llm=llm, jev=FakeJev(), storylines=["Riley Thorsen is two points from 1,000 career points"],
                   storyline_refresh_s=0.05)
    tok = rig.narrator.pseudo.token("home-23")
    phrasings = iter([f"{{{tok}}} is two points from 1,000 career points.",
                      f"Two points from 1,000 career points for {{{tok}}}.",
                      f"{{{tok}}} sits just two points shy of 1,000 career points.",
                      f"{{{tok}}} needs two points for 1,000 career points."] * 3)
    llm.reply = lambda prompt: next(phrasings)

    def stories() -> list[str]:
        return [u.text for u in rig.utterances if u.channel == "color" and "career points" in u.text]

    for k in range(3):  # three free-throw windows 20 s apart
        await _free_throw_window(rig, 400.0 + 20.0 * k)
    assert len(stories()) == 1, stories()  # narration.md §9.5: same colour topic >= 5 min apart
    await _free_throw_window(rig, 400.0 + 320.0)  # five minutes later it may come back
    assert len(stories()) == 2, stories()
    await rig.narrator.aclose()


# --------------------------------------------------------------------------- coach cards


async def test_coach_card_cap_counts_rendered_words() -> None:
    rig = make_rig("coach")
    # 10 words with the token, 11 rendered: fits. 12 with the token (the old check passed it), 13 rendered: dropped.
    fits = Insight("i1", 1.0, "hot_hand", "{player} made 5 of last 6 shots from the wing.",
                   priority=50, team=Team.HOME, player_id="home-23")
    long = Insight("i2", 1.0, "hot_hand", "{player} made 5 of last 6 shots, all from the left wing.",
                   priority=40, team=Team.HOME, player_id="home-23")
    await rig.narrator.on_insights([fits, long], rig.stats)
    assert await rig.narrator.drain(2.0)
    assert [u.text for u in rig.narrator.coach_cards] == ["Riley Thorsen made 5 of last 6 shots from the wing."]
    assert all(len(u.text.split()) <= 12 for u in rig.utterances)
    await rig.narrator.aclose()


# --------------------------------------------------------------------------- WebVTT writes


async def test_webvtt_concurrent_sends_keep_every_cue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    async def uneven_to_thread(fn: Any, *args: Any) -> Any:
        calls["n"] += 1
        await asyncio.sleep(0.05 if calls["n"] == 1 else 0.0)  # the first write finishes last
        return fn(*args)

    monkeypatch.setattr(captions_mod.asyncio, "to_thread", uneven_to_thread)
    path = tmp_path / "live.vtt"
    vtt = WebVTTWriter(path)
    em = CaptionEmitter([vtt], CaptionConfig(delay_s=1.0))
    await asyncio.gather(em.emit("Riverside ball.", 10.0), em.emit("Lakeport ball.", 20.0))
    doc = path.read_text()
    assert "Riverside ball." in doc and "Lakeport ball." in doc


# --------------------------------------------------------------------------- AVSpeech (fake PyObjC)


class _FakeUtterance:
    def __init__(self, text: str) -> None:
        self.text, self.rate, self.pitch, self.volume, self.voice = text, None, None, None, None

    def setRate_(self, r: float) -> None:
        self.rate = r

    def setPitchMultiplier_(self, p: float) -> None:
        self.pitch = p

    def setVolume_(self, v: float) -> None:
        self.volume = v

    def setVoice_(self, v: Any) -> None:
        self.voice = v


class _FakeSynth:
    line_s = 0.3

    def __init__(self) -> None:
        self.until = 0.0
        self.log: list[tuple[str, Any]] = []

    @classmethod
    def alloc(cls) -> _FakeSynth:
        return cls()

    def init(self) -> _FakeSynth:
        return self

    def speakUtterance_(self, u: _FakeUtterance) -> None:
        self.log.append(("speak", u.text))
        self.until = time.monotonic() + (0.0 if u.volume == 0.0 else self.line_s)

    def isSpeaking(self) -> bool:
        return time.monotonic() < self.until

    def stopSpeakingAtBoundary_(self, boundary: int) -> None:
        self.log.append(("stop", boundary))
        self.until = 0.0


def _fake_pyobjc(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    pumps = {"n": 0}

    class _RunLoop:
        def runMode_beforeDate_(self, mode: str, date: Any) -> bool:
            pumps["n"] += 1
            return True

    av = types.ModuleType("AVFoundation")
    av.AVSpeechSynthesizer = _FakeSynth  # type: ignore[attr-defined]
    av.AVSpeechUtterance = types.SimpleNamespace(speechUtteranceWithString_=_FakeUtterance)  # type: ignore[attr-defined]
    av.AVSpeechSynthesisVoice = types.SimpleNamespace(  # type: ignore[attr-defined]
        voiceWithIdentifier_=lambda ident: None, voiceWithLanguage_=lambda lang: f"voice:{lang}")
    av.AVSpeechUtteranceDefaultSpeechRate = 0.5  # type: ignore[attr-defined]
    av.AVSpeechUtteranceMaximumSpeechRate = 1.0  # type: ignore[attr-defined]
    av.AVSpeechBoundaryImmediate, av.AVSpeechBoundaryWord = 0, 1  # type: ignore[attr-defined]
    fnd = types.ModuleType("Foundation")
    fnd.NSRunLoop = types.SimpleNamespace(currentRunLoop=_RunLoop)  # type: ignore[attr-defined]
    fnd.NSDate = types.SimpleNamespace(date=lambda: 0.0)  # type: ignore[attr-defined]
    fnd.NSDefaultRunLoopMode = "kCFRunLoopDefaultMode"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "AVFoundation", av)
    monkeypatch.setitem(sys.modules, "Foundation", fnd)
    return pumps


async def test_avspeech_speak_stop_and_cancel_with_fake_pyobjc(monkeypatch: pytest.MonkeyPatch) -> None:
    pumps = _fake_pyobjc(monkeypatch)
    tts = AVSpeechTTS(TTSConfig(poll_s=0.01))
    synth = tts._synth
    assert synth.log == [("speak", " ")]  # warmed with a silent utterance
    t0 = time.monotonic()
    await tts.speak("Riverside ball.", 1, prosody="big")
    assert 0.25 <= time.monotonic() - t0 < 1.0 and pumps["n"] > 0  # waited for the line, ran the run loop
    task = asyncio.create_task(tts.speak("A long colour line about the run.", 3))
    await asyncio.sleep(0.05)
    await tts.stop()
    await asyncio.wait_for(task, 1.0)
    assert synth.log[-1] == ("stop", 1)  # word boundary
    task = asyncio.create_task(tts.speak("Another line that gets cancelled.", 3))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert synth.log[-1] == ("stop", 0) and not synth.isSpeaking()  # silenced at once, not left talking
