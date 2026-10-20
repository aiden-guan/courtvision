"""ClaudeNarrationLLM: request shape, prompt rules, stop reasons and error mapping (no network)."""

from __future__ import annotations

import json
import logging

import anthropic
import httpx2
import pytest

from courtvision.narration.llm import (ANSWER_DIRECTLY_LINE, FALLBACK_BETA, SYSTEM_PROMPT, ClaudeConfig,
                                       ClaudeNarrationLLM, NarrationLLM, NarrationLLMError, build_system_prompt,
                                       build_user_prompt)
from tests.test_narration_io_support import FakeAnthropic, make_roster

FORBIDDEN_KEYS = ("thinking", "temperature", "top_p", "top_k", "budget_tokens")


async def _drain(llm: ClaudeNarrationLLM, prompt: str = "Line kind: stat_note") -> str:
    out = []
    async for t in llm.stream(llm.system_prompt, prompt, 1536):
        out.append(t)
    return "".join(out)


async def test_opus_request_kwargs() -> None:
    fake = FakeAnthropic(text="{P07} has twelve points.")
    llm = ClaudeNarrationLLM(client=fake)
    assert isinstance(llm, NarrationLLM)
    text = await _drain(llm)
    assert text.strip() == "{P07} has twelve points."
    kw = fake.stream_calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["max_tokens"] == 1536
    assert kw["output_config"] == {"effort": "low"}
    assert kw["betas"] == [FALLBACK_BETA] == ["server-side-fallback-2026-07-01"]
    assert kw["fallbacks"] == "default"
    assert kw["system"] == [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    assert kw["messages"] == [{"role": "user", "content": "Line kind: stat_note"}]  # no assistant prefill
    assert not any(k in kw for k in FORBIDDEN_KEYS)


async def test_haiku_omits_betas_and_fallbacks() -> None:
    fake = FakeAnthropic(served_model="claude-haiku-5-5")
    await _drain(ClaudeNarrationLLM(ClaudeConfig(model="claude-haiku-5-5"), client=fake))
    kw = fake.stream_calls[0]
    assert kw["model"] == "claude-haiku-5-5"
    assert "betas" not in kw and "fallbacks" not in kw
    assert not any(k in kw for k in FORBIDDEN_KEYS)


async def test_disabled_thinking_only_for_haiku() -> None:
    with pytest.raises(ValueError, match="rejects thinking"):
        ClaudeNarrationLLM(ClaudeConfig(model="claude-opus-5-5", disable_thinking=True), client=FakeAnthropic())
    fake = FakeAnthropic(served_model="claude-haiku-5-5")
    await _drain(ClaudeNarrationLLM(ClaudeConfig(model="claude-haiku-5-5", disable_thinking=True), client=fake))
    assert fake.stream_calls[0]["thinking"] == {"type": "disabled"}


def test_system_prompt_is_byte_stable_and_long_enough() -> None:
    assert build_system_prompt() is SYSTEM_PROMPT or build_system_prompt() == SYSTEM_PROMPT
    assert build_system_prompt() == build_system_prompt()
    # >= 512 tokens: English runs ~4 chars/token, so 3,000+ chars and 600+ words clear it with margin.
    assert len(SYSTEM_PROMPT) >= 3000 and len(SYSTEM_PROMPT.split()) >= 600
    assert "{P07}" in SYSTEM_PROMPT and "one short sentence per line" in SYSTEM_PROMPT
    assert "SILENCE" in SYSTEM_PROMPT
    assert build_system_prompt(answer_directly=True).endswith(ANSWER_DIRECTLY_LINE)
    assert ANSWER_DIRECTLY_LINE not in SYSTEM_PROMPT


def test_user_prompt_is_tokenised_and_drops_validator_keys() -> None:
    facts = {"allowed_numbers": ["12", "twelve"], "allowed_tokens": ["P07"], "players": {"P07": {"pts": 12}},
             "storylines": ["{P07} reaches 10 points"]}
    prompt = build_user_prompt(facts, kind="stat_note", max_words=35, max_sentences=2)
    assert "allowed_numbers" not in prompt and '"P07"' in prompt
    assert "Line kind: stat_note" in prompt and "35 words" in prompt and "one short sentence per line" in prompt
    assert not any(e.name in prompt for e in make_roster().entries)
    assert json.loads(prompt.split("FactPacket:\n", 1)[1])["players"]["P07"]["pts"] == 12


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
async def test_refusal_and_max_tokens_fail(stop: str) -> None:
    fake = FakeAnthropic(text="Partial line", stop_reason=stop)
    llm = ClaudeNarrationLLM(client=fake)
    with pytest.raises(NarrationLLMError) as info:
        await _drain(llm)
    assert info.value.reason == stop
    assert llm.last_call is not None and llm.last_call.stop_reason == stop and llm.last_call.error == stop


async def test_usage_and_model_are_recorded_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeAnthropic(served_model="claude-opus-4-8", output_tokens=333)  # a server-side fallback served it
    llm = ClaudeNarrationLLM(client=fake)
    with caplog.at_level(logging.INFO, logger="courtvision.narration.llm"):
        await _drain(llm)
    call = llm.last_call
    assert call is not None
    assert call.model == "claude-opus-4-8" and call.output_tokens == 333 and call.cache_read_input_tokens == 880
    assert call.t_message_start_s is not None and call.t_first_text_s is not None and call.total_s is not None
    assert call.t_message_start_s <= call.t_first_text_s <= call.total_s
    assert "claude-opus-4-8" in caplog.text and "out_tokens=333" in caplog.text


async def test_prewarm_is_non_streaming_with_same_system() -> None:
    fake = FakeAnthropic()
    llm = ClaudeNarrationLLM(client=fake)
    call = await llm.prewarm()
    kw = fake.create_calls[0]
    assert kw["max_tokens"] == 0 and "stream" not in kw
    assert kw["system"] == [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    assert kw["output_config"] == {"effort": "low"} and kw["model"] == "claude-opus-5-5"
    assert call.cache_creation_input_tokens == 880 and call.kind == "prewarm"


def _req() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize("err, reason", [
    (anthropic.APITimeoutError(request=_req()), "timeout"),
    (anthropic.APIConnectionError(request=_req()), "connection"),
    (anthropic.RateLimitError("slow down", response=httpx2.Response(429, request=_req()), body=None), "rate_limit"),
    (anthropic.AuthenticationError("bad key", response=httpx2.Response(401, request=_req()), body=None), "auth"),
    (anthropic.BadRequestError("bad", response=httpx2.Response(400, request=_req()), body=None), "bad_request"),
    (anthropic.InternalServerError("boom", response=httpx2.Response(500, request=_req()), body=None), "api_status_500"),
])
async def test_sdk_errors_map_most_specific_first(err: Exception, reason: str) -> None:
    llm = ClaudeNarrationLLM(client=FakeAnthropic(raise_on_enter=err))
    with pytest.raises(NarrationLLMError) as info:
        await _drain(llm)
    assert info.value.reason == reason


async def test_real_sdk_accepts_the_request_shape() -> None:
    """The installed anthropic SDK builds the beta request and parses an SSE stream (mock transport)."""
    events = [
        {"type": "message_start", "message": {"id": "m1", "type": "message", "role": "assistant",
                                              "model": "claude-opus-5-5", "content": [], "stop_reason": None,
                                              "stop_sequence": None,
                                              "usage": {"input_tokens": 9, "output_tokens": 1,
                                                        "cache_read_input_tokens": 870}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Riverside "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "leads."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": {"output_tokens": 57}},
        {"type": "message_stop"},
    ]
    sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
    seen: dict = {}

    async def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(url=str(request.url), beta=request.headers.get("anthropic-beta"), body=json.loads(request.content))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=sse)

    client = anthropic.AsyncAnthropic(api_key="test", max_retries=0,
                                      http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    llm = ClaudeNarrationLLM(client=client)
    assert (await _drain(llm)) == "Riverside leads."
    assert seen["url"].endswith("/v1/messages?beta=true") and seen["beta"] == FALLBACK_BETA
    body = seen["body"]
    assert body["fallbacks"] == "default" and body["output_config"] == {"effort": "low"} and body["stream"] is True
    assert body["max_tokens"] == 1536 and body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert not any(k in body for k in FORBIDDEN_KEYS)
    assert llm.last_call is not None and llm.last_call.output_tokens == 57 and llm.last_call.stop_reason == "end_turn"
    await client.close()
