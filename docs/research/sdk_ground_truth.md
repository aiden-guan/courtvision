# SDK ground truth (verified against installed packages, 2026-10-20)

These facts were read from the installed package source in `.venv` and exercised
offline. They override anything in other notes if they disagree.

**Scope of the override (2026-10-20).** This note wins on SDK *API surface* facts: names,
signatures, parameters and error types. It does not decide configuration values. Those are set
in `docs/RESEARCH.md` §1A. In particular, the `max_tokens` in the Claude example below was 400
and is now 1536 (D13), because thinking counts toward `max_tokens`.

## typesafe-sdk 0.7.5 (Jev)

```python
from typesafe_sdk import (
    TypeSafeClient, AsyncTypeSafeClient,            # sync / async clients
    Choice, Score, Noul, Box, NoulCriteria,          # question objects (pydantic, extra="forbid")
    SystemOneResponse, ChoiceAnswer, ScoreAnswer, NoulAnswer, BoxAnswer, Usage,
    RetryPolicy,
    TypeSafeError, TypeSafeAPIError, TypeSafeAPIConnectionError, TypeSafeAPITimeoutError,
    TypeSafeRateLimitError, TypeSafeBadRequestError, TypeSafeAuthenticationError,
    TypeSafePermissionDeniedError, TypeSafeNotFoundError, TypeSafeUnprocessableEntityError,
    TypeSafeInternalServerError, TypeSafeAPIResponseValidationError,
)
```

Client constructors (keyword-only, identical for `AsyncTypeSafeClient`):

```python
TypeSafeClient(*, api_key: str | None = None, model: str | None = None,
               retry: RetryPolicy | None = None, timeout: float | httpx2.Timeout | None = None,
               headers: Mapping[str, str] | None = None, transport: httpx2.BaseTransport | None = None,
               http_client: httpx2.Client | None = None, base_url: str | None = None,
               compression: Literal["zstd", False] | None = None)
```
* `TypeSafeError` is raised at construction when no API key is available
  (`TYPESAFE_API_KEY`). Construct lazily and treat that as "Jev disabled".
* Env vars: `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL` (default `https://api.typesafe.ai`),
  `TYPESAFE_DEFAULT_MODEL` (default `jev-latest`), `TYPESAFE_LOG_LEVEL`.
* Default per-HTTP-operation timeout 10.0 s. Sync: context manager / `.close()`.
  Async: `async with` / `await client.aclose()`. The async client takes
  `transport: httpx2.AsyncBaseTransport` / `http_client: httpx2.AsyncClient`.

Call:

```python
resp = client.system_one(state, questions, *, model=None, assets=None, retry=None,
                         timeout=None, extra_headers=None, extra_body=None, response_model=None)
resp = await aclient.system_one(...)   # same signature
```
* `state`: str | dict | list | asset. `questions`: non-empty `Mapping[str, Choice|Score|Noul|Box|dict]`.
* `Choice(instructions=..., criteria={"label": "description" or None, ...})` (1..255 labels)
* `Score(instructions=..., criteria=["level 0 desc", "level 1 desc", ...])` (1..10 levels, ordered from 0)
* `Noul(instructions=..., criteria=NoulCriteria(true=..., false=...) | None)`
* Wire request body: `{"state":..., "model":..., "questions": {name: {"type": "noul"|"choice"|"score", ...}}}`
  POST `{base_url}/v1/systemone`, header `Authorization: Bearer <key>`.

Response (`SystemOneResponse`, frozen pydantic):
* `.model: str` (versioned id that answered, e.g. `jev-1.13.0`), `.usage: Usage(input_tokens, output_tokens)`
* `.answers: dict[str, Answer]` and typed views `.nouls`, `.choices`, `.scores`, `.boxes`
* `NoulAnswer.noul: float` — P(yes) in [0,1]; **no confidence field**.
* `ChoiceAnswer.choice: str`, `.confidence: float`, `.probabilities: dict[str, float]`
* `ScoreAnswer.score: float` (probability-weighted expected level, may be fractional),
  `.confidence: float`, `.legend: dict[int, ...]`, `.probabilities: dict[int, float]` (int keys!)

`RetryPolicy(max_retries=2, backoff_initial=0.5, backoff_max=5.0, backoff_jitter=0.25,
http_statuses={408, 429, 500..599}, respect_retry_after=True, api_connection_error=True,
api_timeout_error=True, exceptions=set(), predicate=None, timeout=30.0)` — for live use set
`RetryPolicy(max_retries=0)` or a tight `timeout` so a slow call never blocks narration.

Offline testing: pass `transport=httpx2.MockTransport(handler)` (sync) or
`httpx2.MockTransport(async_handler)` to the async client, `compression=False`, a dummy
`api_key`. Verified working: the real SDK code path builds the request and parses a
documented-shape response.

## anthropic 1.15.0 (Claude)

* Built on `httpx2`. `anthropic.AsyncAnthropic()` resolves credentials from env / `ant auth` profile.
* Streaming with server-side refusal fallback (recommended default for `claude-opus-5-5`):

```python
async with client.beta.messages.stream(
    model="claude-opus-5-5",
    max_tokens=1536,                          # D13: thinking counts toward max_tokens; 400 risked stop_reason="max_tokens"
    system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
    messages=[{"role": "user", "content": facts_prompt}],
    output_config={"effort": "low"},          # Opus 5.5 thinking cannot be disabled; low effort = lowest latency
    betas=["server-side-fallback-2026-07-01"],
    fallbacks="default",
) as stream:
    async for text in stream.text_stream:
        ...
    final = await stream.get_final_message()
    # always check final.stop_reason == "refusal" before trusting content
```
* Do NOT send `thinking={"type": "disabled"}` or `budget_tokens` to `claude-opus-5-5` (400).
  Do not send `temperature`/`top_p` (removed). No assistant prefill.
* Errors: `anthropic.APIConnectionError`, `anthropic.RateLimitError`, `anthropic.APIStatusError`
  (catch most-specific first), `anthropic.APITimeoutError`.
* Model is configurable; default `claude-opus-5-5`. (`claude-haiku-5-5` is the lowest-latency
  option a user may choose; it has no server-side fallback, so do not send `fallbacks` with it.)

## Addendum (2026-10-20 gap pass): Claude model facts that shape configuration

From Anthropic's API reference as bundled in the Claude Code `claude-api` skill (cached
2026-10-06) [G1]. These are model facts, not SDK-surface facts, so they are listed separately.

* `claude-opus-5-5`: thinking is always on. `thinking={"type": "disabled"}` and
  `{"type": "enabled", "budget_tokens": N}` both return 400 at every effort level. Effort defaults to
  `medium`, so send `output_config={"effort": "low"}` explicitly. Thinking counts toward
  `max_tokens` even though its text is omitted by default (`display: "omitted"`); read blocks by
  `type`, not position. Price: $4 input / $20 output per MTok; cache reads $0.20/MTok. Minimum
  cacheable prefix: 512 tokens.
* `claude-haiku-5-5`: adaptive thinking is on by default. `thinking={"type": "disabled"}` is accepted
  at effort `low`, `medium` and `high`, and returns 400 at `xhigh`/`max`. Omit sampling parameters: the only
  accepted values are `temperature=1` or `top_p=0.99`, sending both is a 400, and any `top_k` is a 400. Prefill is a 400. No
  server-side refusal fallback, so do not send `fallbacks`. Price: $0.10 / $0.50 per MTok for prompts
  of ≤ 100K tokens. Minimum cacheable prefix: 512 tokens.
* Prompt-cache pre-warm: a non-streaming request with `max_tokens: 0` and the same system blocks.
* Data retention: the bundled docs say nothing new about retention for Opus 5.5 and say to treat it
  like Opus 5, which is available under zero data retention (ZDR). The Fable 5.x models require
  30-day retention. Confirm ZDR for Opus 5.5 with Anthropic before relying on it (unverified on a live
  page). Anthropic's public retention page: API inputs/outputs are deleted within 30 days unless
  otherwise agreed [G2].

Sources for this addendum:
[G1] Anthropic Claude API reference, bundled in the Claude Code `claude-api` skill (models table,
thinking and effort tables, prompt-caching minimums, Opus 5.5 and Haiku 5.5 migration sections). Public
docs root: https://platform.claude.com/docs
[G2] Anthropic Privacy Center, "How long do you store my organization's data?" (2026-07-01).
https://privacy.claude.com/en/articles/7996866-how-long-do-you-store-my-organization-s-data
