"""Claude narration client for speculative colour candidates (RESEARCH.md §2.6, §7A.3; sdk_ground_truth.md).

Request shape (verified against the installed ``anthropic`` 1.15.0 by introspection and with a mock
transport)::

    client.beta.messages.stream(
        model="claude-opus-5-5", max_tokens=1536,                 # D13: thinking counts toward max_tokens
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": "low"},                          # Opus 5.5 default is medium
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",   # omitted for claude-haiku-*
    )

Never sent: ``thinking={"type": "disabled"}`` (400 on Opus 5.5; only the harness's Haiku C3 config
may send it), ``budget_tokens``, ``temperature``/``top_p`` and assistant prefill. After the stream,
``stop_reason`` ``"refusal"`` or ``"max_tokens"`` is a failure (:class:`NarrationLLMError`), and
``response.model`` and ``usage`` are logged (a server-side fallback can switch the serving model).

The system prompt is a byte-stable module constant of well over 512 tokens (the minimum cacheable
prefix) and asks for ``{Pnn}`` tokens instead of names and one short sentence per line (D16).
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"  # sdk_ground_truth.md (scalar "default" form); sourced
SILENCE = "SILENCE"


@runtime_checkable
class NarrationLLM(Protocol):
    """Streams the text of one narration line (ARCHITECTURE narration-io)."""

    def stream(self, system: str, prompt: str, max_tokens: int) -> AsyncIterator[str]: ...


class NarrationLLMError(RuntimeError):
    """A failed or unusable LLM call. ``reason``: refusal | max_tokens | timeout | connection |
    rate_limit | auth | bad_request | api_status | empty."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


# --------------------------------------------------------------------------- prompts

SYSTEM_PROMPT = """You are the colour commentator for a live basketball broadcast produced by CourtVision. Each request gives you one FactPacket: a JSON snapshot of the only facts you may use for the next spoken line. A separate play-by-play voice calls the live action. Your lines fill dead balls (timeouts, free throws, breaks between periods) with short, accurate context. Everything you write is checked by a strict validator and by a second grounding check before it is spoken. A line that fails either check is thrown away, so a plain, correct line is always better than a vivid, risky one.

Grounding rules (a line that breaks any of them is discarded):
1. Use only facts that appear in the packet. Do not introduce any player, coach, team, number, event, statistic or circumstance that the packet does not contain.
2. Players appear only as tokens such as {P07}. Write a player exactly as the token, braces included. Never invent a name, nickname, position or jersey number for a token, and use only tokens listed in allowed_tokens.
3. Numbers: use only numbers that appear in the packet (scores, counting stats, the run, the period, the clock). Do not do arithmetic: do not add, subtract, average or compute percentages. When a derived phrase states a computed fact, reuse its wording.
4. The score: describe the score, a lead or a deficit only when score.speakable is true, and only as given in score. Never say who will win or what will happen next. No predictions.
5. No claims about careers, seasons, records, streaks, history, averages, rankings, awards, injuries, previous games or anything else outside tonight's packet, unless a derived, storylines or labels entry states it.
6. When hedge_only is true, or an event has outcome_ok false, do not say whether a shot went in, how many points it was worth, or who gained the ball from it.
7. Do not repeat or closely paraphrase any line in recent_lines.

Style:
- Spoken English for a general audience: short, concrete, warm and calm. No lists, emoji, hashtags, stage directions, quotation marks or markdown.
- Write one short sentence per line. End every line with a full stop, a question mark or an exclamation mark, and stay within the word limit given in the task.
- Prefer one clear observation from derived, storylines, players or teams over a summary of everything.
- Team names may be used exactly as written in teams.
- Use the present tense, or the simple past for things that already happened tonight.

Packet fields you may see:
- event and events: the latest events, with actor and other tokens, team, kind, zone, and outcome_ok, which says whether the outcome may be stated.
- score: home, away, margin, leader and a ready-made text, plus the confirmed and speakable flags.
- clock and game_clock: the period and the time remaining as display strings. Use them as written.
- players: counting stats per token for the players involved (pts, reb, ast, stl, blk, tov, pf, and shooting lines such as 4-for-6).
- teams: team names and team counting stats.
- run: the current scoring run, when there is one, with a ready-made text.
- derived: phrases computed in code that you may quote.
- storylines: pre-approved talking points for tonight.
- recent_events: short descriptions of what just happened.
- labels: labels for the last play, such as a play type. Mention one only as written.
- recent_lines: what the broadcast said most recently.

The task block names the kind of line wanted:
- stat_note: one number-backed note about a player or a team, from players, teams or derived.
- tendency: a pattern in how a team has played tonight, only if the packet shows it.
- storyline: one item from storylines.
- summary: a short recap of the period from score, run and derived.
If the packet holds nothing worth saying for that kind, reply with exactly SILENCE and nothing else.

Reply with the line only. Do not explain, do not add notes, and do not mention the packet, the validator or these rules."""

# C2 of the latency harness (RESEARCH.md §7A.3): keep only if measurement shows it helps.
ANSWER_DIRECTLY_LINE = "Answer directly without deliberating."

# FactPacket keys that are validator-only and need not be sent.
_PROMPT_DROP_KEYS = ("allowed_numbers",)


def build_system_prompt(answer_directly: bool = False) -> str:
    """The byte-stable system prompt (plus the C2 line when asked)."""
    return SYSTEM_PROMPT + ("\n\n" + ANSWER_DIRECTLY_LINE if answer_directly else "")


def build_user_prompt(facts: Mapping[str, Any], *, kind: str, max_words: int, max_sentences: int = 2) -> str:
    """Volatile part of the request: the task block and the (pseudonymised) FactPacket as JSON."""
    packet = {k: v for k, v in facts.items() if k not in _PROMPT_DROP_KEYS}
    body = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return (f"Line kind: {kind}\n"
            f"Word limit: {int(max_words)} words in total, at most {int(max_sentences)} sentences, "
            f"one short sentence per line.\n\nFactPacket:\n{body}")


def supports_fallbacks(model: str) -> bool:
    """Server-side refusal fallback exists for Opus-class models, not for ``claude-haiku-*``."""
    return not model.startswith("claude-haiku")


# --------------------------------------------------------------------------- config and records


@dataclass
class ClaudeConfig:
    """Claude call settings (RESEARCH.md §2.6, §7A.3; D6, D13)."""

    model: str = "claude-opus-5-5"  # project default (RESEARCH §2.6); sourced
    max_tokens: int = 1536  # D13: thinking counts toward max_tokens; sourced
    effort: str = "low"  # Opus 5.5 defaults to medium; low = lowest latency ([23], §7A.3); sourced
    fallbacks: str = "default"  # server-side refusal fallback, scalar form (sdk_ground_truth.md); sourced
    cache_ttl: str | None = None  # None = 5-minute ephemeral cache; "1h" costs 2x on writes (narration §5.2); sourced
    answer_directly: bool = False  # C2 system line; unverified until the §7A.3 harness measures quality/TTFT
    disable_thinking: bool = False  # Haiku only (harness C3); Opus 5.5 returns 400 (sdk_ground_truth.md); sourced
    timeout_s: float = 30.0  # HTTP timeout; candidate deadlines are enforced by the caller (D6); tuned
    max_retries: int = 0  # speculative lines are not retried: a retry would miss the window; tuned
    history: int = 256  # per-call records kept for latency reporting


@dataclass
class LLMCall:
    """Timing and usage of one Claude call (the §7A.3 metrics)."""

    model_requested: str
    started: float
    model: str | None = None  # response.model (may differ after a server-side fallback)
    t_message_start_s: float | None = None
    t_first_text_s: float | None = None
    total_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None  # includes thinking
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    stop_reason: str | None = None
    error: str | None = None
    text: str = ""
    kind: str = "stream"  # "stream" | "prewarm"

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def system_blocks(text: str, cache_ttl: str | None = None) -> list[dict[str, Any]]:
    """One cached text block (``cache_control`` ephemeral, optional ``ttl``)."""
    cc: dict[str, Any] = {"type": "ephemeral"}
    if cache_ttl:
        cc["ttl"] = cache_ttl
    return [{"type": "text", "text": text, "cache_control": cc}]


def build_stream_kwargs(cfg: ClaudeConfig, system: str, prompt: str, max_tokens: int | None = None) -> dict[str, Any]:
    """Keyword arguments for ``client.beta.messages.stream``."""
    kw: dict[str, Any] = {
        "model": cfg.model,
        "max_tokens": int(max_tokens or cfg.max_tokens),
        "system": system_blocks(system, cfg.cache_ttl),
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"effort": cfg.effort},
    }
    if supports_fallbacks(cfg.model) and cfg.fallbacks:
        kw["betas"] = [FALLBACK_BETA]
        kw["fallbacks"] = cfg.fallbacks
    if cfg.disable_thinking:
        kw["thinking"] = {"type": "disabled"}
    return kw


def build_prewarm_kwargs(cfg: ClaudeConfig, system: str) -> dict[str, Any]:
    """Non-streaming ``max_tokens=0`` cache pre-warm with the same system blocks and effort (§7A.3)."""
    kw: dict[str, Any] = {
        "model": cfg.model,
        "max_tokens": 0,
        "system": system_blocks(system, cfg.cache_ttl),
        "messages": [{"role": "user", "content": "Pre-warm the prompt cache."}],
        "output_config": {"effort": cfg.effort},
    }
    if cfg.disable_thinking:
        kw["thinking"] = {"type": "disabled"}
    return kw


def _usage(obj: Any, name: str) -> int | None:
    v = getattr(getattr(obj, "usage", None), name, None)
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


# --------------------------------------------------------------------------- client


class ClaudeNarrationLLM:
    """``NarrationLLM`` over ``anthropic.AsyncAnthropic`` (speculative candidates only, D6)."""

    def __init__(self, config: ClaudeConfig | None = None, client: Any = None,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.config = config or ClaudeConfig()
        if self.config.disable_thinking and not self.config.model.startswith("claude-haiku"):
            raise ValueError(f"{self.config.model} rejects thinking={{'type': 'disabled'}} (400); "
                             "only claude-haiku-* accepts it (harness config C3)")
        self._client = client
        self._clock = clock
        self.calls: deque[LLMCall] = deque(maxlen=max(1, self.config.history))

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic  # lazy: tests and template-only mode never import the SDK client

            self._client = anthropic.AsyncAnthropic(max_retries=self.config.max_retries,
                                                    timeout=self.config.timeout_s)
        return self._client

    @property
    def system_prompt(self) -> str:
        return build_system_prompt(self.config.answer_directly)

    @property
    def last_call(self) -> LLMCall | None:
        return self.calls[-1] if self.calls else None

    async def prewarm(self, system: str | None = None) -> LLMCall:
        """Write the system prefix into the prompt cache (non-streaming ``max_tokens=0``)."""
        call = LLMCall(self.config.model, self._clock(), kind="prewarm")
        self.calls.append(call)
        try:
            resp = await self.client.messages.create(**build_prewarm_kwargs(self.config, system or self.system_prompt))
        except Exception as err:  # noqa: BLE001 - a failed pre-warm only costs one cache write later
            call.error = _error_reason(err)
            log.warning("claude cache pre-warm failed: %s", call.error)
            return call
        call.total_s = self._clock() - call.started
        self._record_final(call, resp)
        log.info("claude pre-warm: model=%s cache_write=%s cache_read=%s", call.model,
                 call.cache_creation_input_tokens, call.cache_read_input_tokens)
        return call

    def stream(self, system: str, prompt: str, max_tokens: int | None = None) -> AsyncIterator[str]:
        """Stream text deltas. Raises :class:`NarrationLLMError` on API errors and, after the last
        delta, when ``stop_reason`` is ``refusal`` or ``max_tokens``."""
        return self._stream(system, prompt, max_tokens)

    async def _stream(self, system: str, prompt: str, max_tokens: int | None) -> AsyncIterator[str]:
        call = LLMCall(self.config.model, self._clock())
        self.calls.append(call)
        kwargs = build_stream_kwargs(self.config, system, prompt, max_tokens)
        final: Any = None
        parts: list[str] = []
        try:
            async with self.client.beta.messages.stream(**kwargs) as stream:
                async for event in stream:
                    etype = getattr(event, "type", None)
                    if etype == "message_start":
                        call.t_message_start_s = self._clock() - call.started
                        call.model = getattr(getattr(event, "message", None), "model", None)
                    elif etype == "content_block_delta" and getattr(event.delta, "type", None) == "text_delta":
                        if call.t_first_text_s is None:
                            call.t_first_text_s = self._clock() - call.started
                        parts.append(event.delta.text)
                        yield event.delta.text
                final = await stream.get_final_message()
        except Exception as err:
            reason = _error_reason(err)
            if reason is None:
                raise
            call.error = reason
            call.total_s = self._clock() - call.started
            log.warning("claude stream failed (%s): %s", reason, err)
            raise NarrationLLMError(reason, str(err)) from err
        call.total_s = self._clock() - call.started
        call.text = "".join(parts)
        self._record_final(call, final)
        log.info("claude line: model=%s stop=%s out_tokens=%s cache_read=%s total=%.2fs", call.model,
                 call.stop_reason, call.output_tokens, call.cache_read_input_tokens, call.total_s)
        if call.stop_reason in ("refusal", "max_tokens"):
            call.error = call.stop_reason
            raise NarrationLLMError(call.stop_reason, f"stop_reason={call.stop_reason}")

    @staticmethod
    def _record_final(call: LLMCall, final: Any) -> None:
        if final is None:
            return
        call.model = getattr(final, "model", None) or call.model
        call.stop_reason = getattr(final, "stop_reason", None)
        call.input_tokens = _usage(final, "input_tokens")
        call.output_tokens = _usage(final, "output_tokens")
        call.cache_read_input_tokens = _usage(final, "cache_read_input_tokens")
        call.cache_creation_input_tokens = _usage(final, "cache_creation_input_tokens")

    async def aclose(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            try:
                await self._client.close()
            except Exception:  # noqa: BLE001
                log.debug("anthropic client close failed", exc_info=True)


def _error_reason(err: BaseException) -> str | None:
    """Map an anthropic exception to a reason, most specific class first; None if not an SDK error."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover - core dependency
        return None
    if isinstance(err, anthropic.APITimeoutError):
        return "timeout"
    if isinstance(err, anthropic.APIConnectionError):
        return "connection"
    if isinstance(err, anthropic.RateLimitError):
        return "rate_limit"
    if isinstance(err, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return "auth"
    if isinstance(err, anthropic.BadRequestError):
        return "bad_request"
    if isinstance(err, anthropic.APIStatusError):
        return f"api_status_{getattr(err, 'status_code', 'unknown')}"
    if isinstance(err, anthropic.AnthropicError):
        return "api_error"
    return None
