"""Text-to-speech adapters (D5; narration.md §6, §11.7; RESEARCH.md §2.7).

* :class:`TTS` protocol: ``await speak(text, priority, prosody=None)`` returns when the line has
  finished playing or was cut; ``await stop()`` cuts the current line (at the next word boundary
  where the engine supports it).
* :class:`NullTTS`: records lines and simulates playback time from the word count. Tests, headless
  servers and dashboards without audio use it.
* :class:`MacSayTTS`: one ``say`` process per line. **Dev-only** (D5): ``say`` measured 0.78-2.81 s
  per line, so it is never the production voice. ``stop()`` terminates the process (no word boundary).
* :class:`AVSpeechTTS`: production adapter (``AVSpeechSynthesizer`` via ``pyobjc-framework-AVFoundation``),
  warmed at boot, ``stopSpeakingAtBoundary_(AVSpeechBoundaryWord)`` word-boundary cut. **Untested in
  CI** (needs macOS audio and PyObjC). It uses the synthesizer's own ``speakUtterance_`` playback;
  the write-path mixer (``writeUtterance_toBufferCallback_`` + word -> sample-offset map, narration.md
  §3.6) is not built yet.

Prosody tags ("calm" / "up" / "big", narration.md §3.4) map to rate and pitch multipliers.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)

PROSODY_TAGS = ("calm", "up", "big")


@runtime_checkable
class TTS(Protocol):
    """A speech engine the narrator drives (one line at a time)."""

    async def speak(self, text: str, priority: int = 1, *, prosody: str | None = None) -> None:
        """Play ``text``; return when it finished or was stopped."""
        ...

    async def stop(self) -> None:
        """Cut the current line (next word boundary when supported)."""
        ...


@dataclass
class TTSConfig:
    """Voice settings shared by the adapters."""

    words_per_s: float = 3.5  # macOS default voice measured 3.5-3.7 words/s (narration.md §4.1 [M]); sourced
    min_line_s: float = 0.4  # shortest simulated line (one-word calls); tuned
    rate_wpm: int = 210  # `say -r`: Samantha measured ~210-220 wpm (narration.md §4.1 [M]); sourced
    voice: str | None = None  # None = system default voice (licensed stock voices only, RESEARCH §2.7)
    # Prosody multipliers (narration.md §3.4 "calm / up / big"; values are design proposals); unverified
    prosody_rate: dict[str, float] = field(default_factory=lambda: {"calm": 0.95, "up": 1.05, "big": 1.12})
    prosody_pitch: dict[str, float] = field(default_factory=lambda: {"calm": 1.0, "up": 1.05, "big": 1.1})
    poll_s: float = 0.02  # completion polling interval for engines without a completion future; tuned
    start_timeout_s: float = 1.5  # AVSpeech: give up if speaking never starts (cold start 0.6-1.0 s [M]); sourced
    stop_grace_s: float = 1.0  # how long stop() may take before speak() is forced to return; tuned


def estimate_duration_s(text: str, words_per_s: float = 3.5, min_s: float = 0.4) -> float:
    """Spoken duration from the word count (~3.5 words/s measured TTS rate, narration.md §9.2)."""
    words = len(text.split())
    return max(min_s, words / max(words_per_s, 1e-6))


@dataclass
class SpokenLine:
    """One line handed to :class:`NullTTS`."""

    text: str
    priority: int
    prosody: str | None
    started: float
    duration_s: float
    interrupted: bool = False


class NullTTS:
    """Silent TTS: records every line and simulates its playback time.

    ``time_scale`` multiplies the simulated duration (0 = return immediately). ``stop()`` ends the
    current line early and marks it ``interrupted``.
    """

    def __init__(self, config: TTSConfig | None = None, time_scale: float = 1.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.config = config or TTSConfig()
        self.time_scale = max(0.0, float(time_scale))
        self._clock = clock
        self.spoken: list[SpokenLine] = []
        self._stop_evt: asyncio.Event | None = None
        self._current: SpokenLine | None = None

    @property
    def texts(self) -> list[str]:
        return [s.text for s in self.spoken]

    @property
    def speaking(self) -> bool:
        return self._current is not None

    async def speak(self, text: str, priority: int = 1, *, prosody: str | None = None) -> None:
        cfg = self.config
        duration = estimate_duration_s(text, cfg.words_per_s, cfg.min_line_s)
        rec = SpokenLine(text, priority, prosody, self._clock(), duration)
        self.spoken.append(rec)
        stop = asyncio.Event()
        self._stop_evt, self._current = stop, rec
        try:
            wait = duration * self.time_scale
            if wait > 0:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=wait)
                    rec.interrupted = True
                except TimeoutError:
                    pass
            elif stop.is_set():
                rec.interrupted = True
        except asyncio.CancelledError:
            rec.interrupted = True
            raise
        finally:
            if self._current is rec:
                self._current, self._stop_evt = None, None

    async def stop(self) -> None:
        if self._stop_evt is not None:
            self._stop_evt.set()

    async def aclose(self) -> None:
        await self.stop()


class MacSayTTS:
    """macOS ``say`` per line. Dev-only (D5): 0.78-2.81 s per call [M]; ``stop()`` kills the process."""

    def __init__(self, config: TTSConfig | None = None, executable: str = "say") -> None:
        self.config = config or TTSConfig()
        path = shutil.which(executable)
        if path is None:
            raise RuntimeError(f"MacSayTTS: {executable!r} not found (macOS only; dev-only fallback, D5)")
        self._exe = path
        self._proc: asyncio.subprocess.Process | None = None
        log.warning("MacSayTTS is a dev-only fallback (D5): use AVSpeechTTS in production")

    def _args(self, text: str, prosody: str | None) -> list[str]:
        cfg = self.config
        rate = int(round(cfg.rate_wpm * cfg.prosody_rate.get(prosody or "", 1.0)))
        args = [self._exe, "-r", str(rate)]
        if cfg.voice:
            args += ["-v", cfg.voice]
        # A leading "-" would be parsed as an option.
        args.append(text if not text.startswith("-") else " " + text)
        return args

    async def speak(self, text: str, priority: int = 1, *, prosody: str | None = None) -> None:
        proc = await asyncio.create_subprocess_exec(*self._args(text, prosody), stdin=asyncio.subprocess.DEVNULL,
                                                    stdout=asyncio.subprocess.DEVNULL,
                                                    stderr=asyncio.subprocess.DEVNULL)
        self._proc = proc
        try:
            await proc.wait()
        except asyncio.CancelledError:
            _terminate(proc)
            raise
        finally:
            if self._proc is proc:
                self._proc = None

    async def stop(self) -> None:
        proc = self._proc
        if proc is not None and proc.returncode is None:
            _terminate(proc)
            try:
                await asyncio.wait_for(proc.wait(), timeout=self.config.stop_grace_s)
            except TimeoutError:
                proc.kill()

    async def aclose(self) -> None:
        await self.stop()


def _terminate(proc: asyncio.subprocess.Process) -> None:
    try:
        proc.terminate()
    except ProcessLookupError:
        pass


class AVSpeechTTS:
    """``AVSpeechSynthesizer`` via PyObjC (D5). UNTESTED: requires macOS and pyobjc-framework-AVFoundation.

    The synthesizer is created and warmed (one silent utterance) at construction, because the cold
    first call costs 0.6-1.0 s and a warm one 12-26 ms [M]. Completion is detected by polling
    ``isSpeaking()``; each poll also runs this thread's NSRunLoop once without blocking, because
    the synthesizer needs a run loop on its thread (narration.md §6) and asyncio does not run one.
    Use one instance from the event-loop thread that created it.
    """

    def __init__(self, config: TTSConfig | None = None, warm: bool = True) -> None:
        self.config = config or TTSConfig()
        try:
            import AVFoundation  # type: ignore[import-not-found]  # lazy: optional `tts` extra
        except ImportError as err:  # pragma: no cover - depends on the host
            raise RuntimeError(
                "AVSpeechTTS needs pyobjc-framework-AVFoundation 12.2.2 (install the `tts` extra: "
                "pip install 'courtvision[tts]'); use NullTTS or the dev-only MacSayTTS instead"
            ) from err
        try:  # pragma: no cover - needs PyObjC; Foundation ships with pyobjc-framework-AVFoundation
            import Foundation  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover
            Foundation = None
        self._av: Any = AVFoundation
        self._foundation: Any = Foundation
        self._synth = AVFoundation.AVSpeechSynthesizer.alloc().init()
        self._voice = self._pick_voice()
        if warm:
            self._synth.speakUtterance_(self._utterance(" ", None, volume=0.0))
            self._pump()

    def _pump(self) -> None:  # pragma: no cover - needs AVFoundation
        """Run the current NSRunLoop once, non-blocking (services the synthesizer's callbacks)."""
        fnd = self._foundation
        if fnd is None:
            return
        try:
            fnd.NSRunLoop.currentRunLoop().runMode_beforeDate_(fnd.NSDefaultRunLoopMode, fnd.NSDate.date())
        except Exception:  # noqa: BLE001 - polling isSpeaking() still works without it on some hosts
            log.debug("NSRunLoop pump failed", exc_info=True)
            self._foundation = None

    def _pick_voice(self) -> Any:  # pragma: no cover - needs AVFoundation
        av, ident = self._av, self.config.voice
        voice = av.AVSpeechSynthesisVoice.voiceWithIdentifier_(ident) if ident else None
        return voice or av.AVSpeechSynthesisVoice.voiceWithLanguage_("en-US")

    def _utterance(self, text: str, prosody: str | None, volume: float = 1.0) -> Any:  # pragma: no cover
        av, cfg = self._av, self.config
        u = av.AVSpeechUtterance.speechUtteranceWithString_(text)
        rate = float(av.AVSpeechUtteranceDefaultSpeechRate) * cfg.prosody_rate.get(prosody or "", 1.0)
        u.setRate_(min(float(av.AVSpeechUtteranceMaximumSpeechRate), rate))
        u.setPitchMultiplier_(cfg.prosody_pitch.get(prosody or "", 1.0))
        u.setVolume_(volume)
        if self._voice is not None:
            u.setVoice_(self._voice)
        return u

    async def speak(self, text: str, priority: int = 1, *, prosody: str | None = None) -> None:  # pragma: no cover
        cfg = self.config
        self._synth.speakUtterance_(self._utterance(text, prosody))
        try:
            waited = 0.0
            while not self._synth.isSpeaking() and waited < cfg.start_timeout_s:
                self._pump()
                await asyncio.sleep(cfg.poll_s)
                waited += cfg.poll_s
            while self._synth.isSpeaking():
                self._pump()
                await asyncio.sleep(cfg.poll_s)
        except asyncio.CancelledError:
            # The caller gave up on this line (stop grace exceeded, shutdown): silence it now,
            # or the synthesizer keeps talking over the next line.
            self._synth.stopSpeakingAtBoundary_(self._av.AVSpeechBoundaryImmediate)
            raise

    async def stop(self) -> None:  # pragma: no cover
        if self._synth.isSpeaking():
            self._synth.stopSpeakingAtBoundary_(self._av.AVSpeechBoundaryWord)

    async def aclose(self) -> None:  # pragma: no cover
        if self._synth.isSpeaking():
            self._synth.stopSpeakingAtBoundary_(self._av.AVSpeechBoundaryImmediate)


def make_tts(kind: str, config: TTSConfig | None = None, **kwargs: Any) -> TTS:
    """``"avspeech"`` (production, D5) | ``"say"`` (dev-only) | ``"none"`` (silent)."""
    if kind == "avspeech":
        return AVSpeechTTS(config, **kwargs)
    if kind == "say":
        return MacSayTTS(config, **kwargs)
    if kind in ("none", "null"):
        return NullTTS(config, **kwargs)
    raise ValueError(f"unknown TTS {kind!r}; expected 'avspeech', 'say' or 'none'")
