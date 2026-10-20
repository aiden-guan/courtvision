"""TTS adapters: NullTTS timing and stop, MacSayTTS process control (fake executable), AVSpeech import error."""

from __future__ import annotations

import asyncio
import stat
import sys
import time
from pathlib import Path

import pytest

from courtvision.narration.tts import TTS, AVSpeechTTS, MacSayTTS, NullTTS, TTSConfig, estimate_duration_s, make_tts


def test_duration_from_word_count() -> None:
    assert estimate_duration_s("one two three four five six seven", 3.5) == pytest.approx(2.0)
    assert estimate_duration_s("Good.", 3.5, min_s=0.4) == pytest.approx(0.4)


async def test_null_tts_records_and_simulates_time() -> None:
    tts = NullTTS(time_scale=0.05)
    assert isinstance(tts, TTS)
    t0 = time.monotonic()
    await tts.speak("one two three four five six seven", 1, prosody="up")
    assert 0.05 <= time.monotonic() - t0 < 1.0  # 2.0 s simulated x 0.05
    [rec] = tts.spoken
    assert rec.text.startswith("one") and rec.priority == 1 and rec.prosody == "up" and not rec.interrupted
    assert rec.duration_s == pytest.approx(2.0)


async def test_null_tts_stop_interrupts() -> None:
    tts = NullTTS(time_scale=1.0)
    task = asyncio.create_task(tts.speak("a long colour line about the run that keeps going and going", 0))
    await asyncio.sleep(0.05)
    assert tts.speaking
    t0 = time.monotonic()
    await tts.stop()
    await task
    assert time.monotonic() - t0 < 0.5 and tts.spoken[0].interrupted and not tts.speaking


def _fake_say(tmp_path: Path, body: str) -> str:
    script = tmp_path / "fake-say"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script stands in for `say`")
async def test_mac_say_args_and_stop(tmp_path: Path) -> None:
    out = tmp_path / "args.txt"
    exe = _fake_say(tmp_path, f'printf "%s|" "$@" > "{out}"\nexec sleep 5')
    tts = MacSayTTS(TTSConfig(rate_wpm=200, voice="Samantha"), executable=exe)
    task = asyncio.create_task(tts.speak("-Riverside ball.", 1, prosody="big"))
    for _ in range(100):
        await asyncio.sleep(0.02)
        if out.exists() and out.read_text():
            break
    t0 = time.monotonic()
    await tts.stop()
    await asyncio.wait_for(task, 2.0)
    assert time.monotonic() - t0 < 1.5  # the process was terminated, not waited out
    args = out.read_text().split("|")
    assert args[:4] == ["-r", "224", "-v", "Samantha"]  # 200 wpm x 1.12 for "big"
    assert args[4] == " -Riverside ball."  # a leading dash is not parsed as an option


def test_mac_say_missing_executable() -> None:
    with pytest.raises(RuntimeError, match="not found"):
        MacSayTTS(executable="definitely-not-a-say-binary")


def test_avspeech_needs_pyobjc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "AVFoundation", None)  # makes `import AVFoundation` fail
    with pytest.raises(RuntimeError, match="pyobjc-framework-AVFoundation"):
        AVSpeechTTS()


def test_make_tts() -> None:
    assert isinstance(make_tts("none"), NullTTS)
    with pytest.raises(ValueError):
        make_tts("espeak")
