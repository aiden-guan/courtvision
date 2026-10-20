# TypeSafe Jev: exact API reference for the CourtVision Python integration

Researched 2026-10-20. Every page on the requested list was fetched as raw `.md` (the
JSX signature widgets were converted back to plain code). I also read
`https://api.typesafe.ai/openapi.json` (spec version `0.2.0`), the PyPI JSON metadata, and
the `v0.7.5` tag of the SDK source on GitHub. Behaviour marked **[verified offline]** was
run against the installed `typesafe-sdk 0.7.5` in this repo's `.venv`, using an
`httpx2.MockTransport`. No real API calls were made. Code blocks under "documented
signature" are copied verbatim from the official reference pages.

---

## 0. Decisions for CourtVision (TL;DR)

| Decision | Value | Why |
|---|---|---|
| Package | `typesafe-sdk==0.7.5`, optionally the `[zstd]` extra | Latest release, 2026-10-14 [37][25]. On Python 3.12, zstd compression only works with the extra [24][38] |
| Import root | `typesafe_sdk` | [22][38] |
| Clients | `TypeSafeClient` (sync), `AsyncTypeSafeClient` (async). Create one and reuse it | [26][27][24] |
| Model | Pin `"jev-1.13.0"`. Use `"jev-1.14.0"` only for Box/image experiments | The `jev-latest` and `jev-preview` aliases move between releases; Box needs 1.14 [17][9] |
| Call | `client.system_one(state, questions, *, model=..., timeout=..., retry=...)` | [26] |
| Questions | `Choice`, `Score`, `Noul`, plus `Box` (images only), as keyword arguments | [28] |
| Answers | `resp.choices[id]`, `resp.scores[id]`, `resp.nouls[id]`, `resp.boxes[id]` | [30] |
| Live retry policy | Tight `RetryPolicy` plus `asyncio.wait_for` as a hard deadline | The default policy can block for about 30 s or slightly more (the budget never cancels an in-flight attempt) [31][38] |
| Images | Do **not** put live video frames on the hot path | Images are "dreamer-ready", need an opt-in header, have strict size limits and no token-cost guidance [11][9] |
| Numbers | Compute all stats, counts, clocks and margins in Python. Send them to Jev as named buckets or text | Jaggedness page [18] |

---

## 1. What Jev / System One is

- **System One** is TypeSafe's class of models that make "fast, structured decisions"
  for software. You send a **state** (string, JSON object or array) and typed
  **questions**, and get back typed answers with probability distributions. There is no
  text generation [2][1].
- **Jev** is "TypeSafe's flagship model and the first System One model" [1][2]. It is
  trained with RLCD for calibrated probabilities [17]. TypeSafe notes that calibration holds
  across groups of predictions and does not guarantee any single answer [2].
- All questions in one request see the same state. Each is evaluated **independently and
  in parallel**, so one answer never becomes context for another [1][5].
- Question IDs (the dict keys) are **not sent to the model**. Put the full question in
  `instructions` [5][6].
- Jev does not generate replies, code or explanations. It is not an agent [2][4][18].
- Jev is not fine-tuned per customer. You adapt it through `state`, `instructions` and
  `criteria` [17].
- Data handling: Jev is not trained on customer requests or responses. ZDR is available
  for enterprise customers [17].

### Latency (every number the docs give)

| Source | Number | Context |
|---|---|---|
| How-to-build page [4] | "Most queries complete in about 100 ms" | Marketing-level claim with no percentile |
| Twitch chat cookbook [35] | Median **about 0.4 s** per request | `jev-1.13.0` on 2026-09-25: 3 questions, a small state, measured client-side with `perf_counter`, 8 concurrent workers |
| Parallel-questions cookbook [39] | **0.27 s** for 1 call with 13 questions vs 2.71 s for 13 sequential calls (10.0x faster, 12.2x cheaper) | `jev-1.12`, ~54,000-character article. The primitives page quotes 11.5x / 9.6x for the same cookbook [5] |
| Models page [17] | Long context is "slower and less accurate" | No number given |
| SDK default [34] | `DEFAULT_TIMEOUT = 10.0` s per HTTP operation | — |

Implications for CourtVision: budget about 150–500 ms end-to-end per call from a home
network (this is my estimate from [4][35], not a documented SLA). Never block narration on
Jev. Batch every question for a given state into **one** request, since extra questions
barely change latency [1][5][13].

### Pricing [17]

- **$42 per billion input tokens ($0.042 per million). Output tokens are free.** Jev 1.14
  and 1.13 have the same price [17]. Usage reports `output_tokens` as well, but they are
  "currently free of charge" [36].
- Images are billed as input tokens at the same rate. TypeSafe has not yet published a
  way to estimate image tokens [11].
- Example (my arithmetic): 500 input tokens × 4 requests/s × 2.5 h ≈ 18 M tokens ≈
  **$0.76 per game**.

### Rate limits [17][21]

- Limits are set per organization, in tokens/s and requests/s. Your org's values are in
  the console (`https://console.typesafe.ai/rate-limits`). The docs give no default
  numbers (unverified).
- Going over a limit returns `429`. `529 Overloaded` means TypeSafe is overloaded. The SDKs
  retry both with exponential backoff and honour `retry-after` [17][21][31].

### Models and aliases [17]

| Model ID | Price | Context | Input | Notes |
|---|---|---|---|---|
| `jev-1.14.0` | $0.042/Mtok | 64k per request / 32k per question | "Text" per the Models table [17]; images accepted per the Images page (dreamer-ready, opt-in) [11] | Newest. Available from **2026-10-14**. Required for `Box` [9] |
| `jev-1.13.0` | $0.042/Mtok | 64k per request / 32k per question | "Text" per the Models table [17]; images accepted per the Images page (dreamer-ready, opt-in) [11] | Current stable release (target of `jev-latest`) |

| Alias | Points to (as of the docs) | Meaning |
|---|---|---|
| `jev-latest` | `jev-1.13.0` | Most recent stable, official release. **SDK default** (`DEFAULT_MODEL = 'jev-latest'`) [34] |
| `jev-preview` | `jev-1.14.0` | Most recent release, official or not |

- `resp.model` returns the **versioned ID that answered**, even when you sent an alias
  [17][36]. Log it with every decision.
- **Pinning:** the docs say that if you have tuned confidence thresholds against one
  version, you should send that version's ID instead of the alias [17]. For CourtVision,
  pin `JEV_MODEL = "jev-1.13.0"` and re-tune thresholds before moving to `jev-1.14.0`.
- `GET /v1/models` currently lists only aliases. Versioned IDs are accepted whether or
  not they appear in the list [17].
- Other model strings appear in the docs but are **not** in the Models table, so avoid
  them: `model="jev"` (SDK basic usage [23]), `model="jev-1.13"` (jaggedness example [18]),
  `"jev-1.12"` (parallel cookbook [39]). Gateway IDs: OpenRouter `"~typesafe/jev-latest"`,
  Vercel `"typesafe-ai/jev"` [23].

---

## 2. Python SDK (`typesafe-sdk`)

### 2.1 Package facts

| Item | Value |
|---|---|
| PyPI name | `typesafe-sdk` [22][37] |
| Import name | `typesafe_sdk` [22] |
| Current version | **0.7.5**, uploaded 2026-10-14. Earlier releases: 0.7.4 (10-09), 0.7.3 (10-09), 0.7.2 (09-26), 0.7.1 (09-21), 0.7.0 (09-18), 0.6.0 (09-15), 0.5.7 (09-11), 0.0.1a0 [37] |
| Python | `requires-python = ">=3.10"`. Classifiers cover 3.10–3.14 [37][38]. **3.12 is fine** |
| License | MIT [38] |
| Runtime deps | `anyio>=4.0.0`, `httpx2>=2.0.0`, `packaging>=24.0`, `pydantic>=2.12.0`, `pydantic-core>=2.41.1`, `tenacity>=9.0.0`, `truststore>=0.10.0` (not on emscripten), `typing-extensions>=4.13.0` [37] |
| Extras | `http2` (`httpx2[http2]`), `pil` (`pillow>=11.0.0`), `zstd` (`backports-zstd>=1.0.0` on Python < 3.14) [22][37] |
| HTTP stack | **`httpx2`**, not `httpx`. A custom `http_client` must be an `httpx2.Client` or `httpx2.AsyncClient` [26][27] |
| Breaking history | 0.6.0: `Score.criteria` became an ordered sequence instead of an int-keyed dict. 0.7.0: serialization moved from `msgspec` to `pydantic`, and `response_model` was added [25] |

Install:

```sh
uv add "typesafe-sdk[zstd]==0.7.5"     # zstd extra so request bodies >= 1 KiB get compressed on 3.12
```

**[verified offline]** The repo's `.venv` has `typesafe_sdk 0.7.5` and `httpx2 2.15.1`, but
**not** `backports.zstd`. With `compression=None`, the SDK therefore sends uncompressed
bodies without any error. Passing `compression="zstd"` explicitly raises `TypeSafeError`
with the message "zstd compression requires Python 3.14+ or the `typesafe-sdk[zstd]`
extra." [38]

### 2.2 Public names (`typesafe_sdk.__all__`, v0.7.5) [38]

```python
from typesafe_sdk import (
    # clients and resources
    TypeSafeClient, AsyncTypeSafeClient, Models, AsyncModels,
    # questions
    Choice, Score, Noul, NoulCriteria, Box,
    ChoiceModel, ScoreModel, NoulModel, BoxModel, Question, QuestionModel, Questions,
    # assets and content types
    Image, Asset, Content, ContentValue, JSONContent, JSONValue,
    # responses
    SystemOneResponse, Answer, ChoiceAnswer, ScoreAnswer, NoulAnswer, BoxAnswer, BoxCoordinates,
    Usage, ListModelsResponse, ModelMetadata,
    # retries
    RetryPolicy,
    # errors
    TypeSafeError, TypeSafeAPIError, TypeSafeBadRequestError, TypeSafeAuthenticationError,
    TypeSafePermissionDeniedError, TypeSafeNotFoundError, TypeSafeUnprocessableEntityError,
    TypeSafeRateLimitError, TypeSafeInternalServerError, TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError, TypeSafeAPIResponseValidationError,
)
from typesafe_sdk import constants        # API_KEY_ENV, DEFAULT_MODEL, ...
```

### 2.3 Environment variables and constants (documented) [34][23]

```python
API_KEY_ENV = 'TYPESAFE_API_KEY'
BASE_URL_ENV = 'TYPESAFE_BASE_URL'
DEFAULT_MODEL_ENV = 'TYPESAFE_DEFAULT_MODEL'
LOG_LEVEL_ENV = 'TYPESAFE_LOG_LEVEL'
DEFAULT_BASE_URL = 'https://api.typesafe.ai'
DEFAULT_MODEL = 'jev-latest'
DEFAULT_TIMEOUT = 10.0
```

| Variable | Configures | Default |
|---|---|---|
| `TYPESAFE_API_KEY` | API key (required) | — |
| `TYPESAFE_BASE_URL` | API root URL | `https://api.typesafe.ai` |
| `TYPESAFE_DEFAULT_MODEL` | Default model | `jev-latest` |
| `TYPESAFE_LOG_LEVEL` | Level of the `typesafe_sdk` logger, applied **once at import** (`debug`, `info`, `warning`, `error`, `off`) | unset |

- Explicit constructor arguments override environment variables. Empty or
  whitespace-only environment values are ignored [26].
- Keys have leading and trailing whitespace stripped. Empty keys, internal whitespace,
  control characters and non-ASCII characters are rejected **at construction**. An
  explicitly empty key does not fall back to the environment [23]. **[verified offline]**
  `TypeSafeClient(api_key="")` raises `TypeSafeError`, so construct the client lazily and
  treat that error as "Jev disabled".
- Logging: `info` writes one line per request. `debug` also logs headers and **bodies,
  which are not redacted**. Secret headers are redacted [24].

### 2.4 Sync client: `TypeSafeClient` (documented signature) [26]

```python
TypeSafeClient(
    *,
    api_key: str | None = None,
    model: str | None = None,
    retry: RetryPolicy | None = None,
    timeout: float
    | httpx2.Timeout
    | None = None,
    headers: Mapping[str, str] | None = None,
    transport: httpx2.BaseTransport
    | None = None,
    http_client: httpx2.Client | None = None,
    base_url: str | None = None,
    compression: Literal["zstd", False]
    | None = None,
)
```

| Param | Meaning [26] |
|---|---|
| `api_key` | Required. Falls back to `TYPESAFE_API_KEY` |
| `model` | Default model. Falls back to `TYPESAFE_DEFAULT_MODEL`, then `jev-latest` |
| `retry` | `RetryPolicy`. Pass `RetryPolicy(max_retries=0)` to disable retries |
| `timeout` | Timeout for each HTTP operation, in seconds or as an `httpx2.Timeout`. Inherits `http_client.timeout` when one is supplied, otherwise 10.0 |
| `headers` | Extra headers on every request. Example: `{"TypeSafe-Danger-Zone-Images": "true"}` [23] |
| `transport` | Custom `httpx2.BaseTransport`. Closed when the client closes |
| `http_client` | Custom `httpx2.Client`. Cannot be combined with `transport` (raises `ValueError`). Closed when the client closes |
| `base_url` | API root. Falls back to `TYPESAFE_BASE_URL` |
| `compression` | `"zstd"`, or `False` for uncompressed. The default `None` means zstd when available. Only bodies of 1 KiB or more are compressed [24][38] |

Raises `TypeSafeError` when the API key is missing or invalid, the timeout is invalid, or
zstd was requested but is unavailable. Raises `ValueError` when both `transport` and
`http_client` are given [26].

Lifecycle: use `with TypeSafeClient() as client:` (it has `__enter__` and `__exit__`
[38]), or call `client.close()`. Other members: `client.models` (a cached property of type
`Models`) and `client.attach_assets(...)` [26].

Performance notes [24]:
- Reuse one client. A new client means new connections and TLS handshakes.
- The default pool is 100 connections, all kept alive for 30 s.
- The sync client can be shared across threads, but under heavy load the docs recommend
  one client per thread (`threading.local`).
- HTTP/1.1 is the default and is recommended for high request rates. HTTP/2 needs the
  `[http2]` extra and `http_client=httpx2.Client(http2=True)`.

### 2.5 Async client: `AsyncTypeSafeClient` (documented signature) [27]

```python
AsyncTypeSafeClient(
    *,
    api_key: str | None = None,
    model: str | None = None,
    retry: RetryPolicy | None = None,
    timeout: float
    | httpx2.Timeout
    | None = None,
    headers: Mapping[str, str] | None = None,
    transport: httpx2.AsyncBaseTransport
    | None = None,
    http_client: httpx2.AsyncClient
    | None = None,
    base_url: str | None = None,
    compression: Literal["zstd", False]
    | None = None,
)
```

- Use `async with AsyncTypeSafeClient() as client:` (`__aenter__`/`__aexit__` [38]) or
  `await client.aclose()`.
- `await client.system_one(...)` and `await client.models.list()`. `client.models` has
  type `AsyncModels` [27].
- `client.attach_assets(...)` returns a **sync** `AbstractContextManager[None]`, so call
  it with a plain `with`, even in async code [27].
- For high request rates in async code, the docs suggest an aiohttp-backed client [24]:

```python
from httpx_aiohttp.httpx2 import Httpx2AiohttpClient

from typesafe_sdk import AsyncTypeSafeClient

client = AsyncTypeSafeClient(http_client=Httpx2AiohttpClient())
```

(Install with `uv add "httpx-aiohttp[httpx2]"`.)

### 2.6 `system_one(...)`: full documented signature [26][27]

```python
system_one(
    state: Content,
    questions: Mapping[str, Question],
    *,
    model: str | None = None,
    assets: Mapping[str, Asset] | None = None,
    retry: RetryPolicy | None = None,
    timeout: float
    | httpx2.Timeout
    | None = None,
    extra_headers: Mapping[str, str]
    | None = None,
    extra_body: Mapping[str, JSONValue | None]
    | None = None,
    response_model: type[ResponseT]
    | None = None,
) -> SystemOneResponse | ResponseT
```

There are two overloads. `response_model: None = None` returns `SystemOneResponse`.
`response_model: type[ResponseT]` returns `ResponseT`. The async version has the same
signature and is a coroutine.

| Param | Meaning [26] |
|---|---|
| `state` | `Content`: text, an asset, a JSON object, or an array. **Cannot be `None`**, but values inside an object may be [28] |
| `questions` | **Non-empty** mapping from names to question objects or raw dicts. An empty mapping raises `TypeSafeError("At least one question is required.")` **[verified offline]** |
| `model` | Per-call override. `None` uses the client default |
| `assets` | Named image references: `{"<ref>": Image(...)}` |
| `retry` | Per-call `RetryPolicy` override |
| `timeout` | Per-call HTTP-operation timeout override, in seconds |
| `extra_headers` | Per-call headers, e.g. the danger-zone headers |
| `extra_body` | Top-level body fields shallow-merged in after `state`, `model`, `questions` and `assets`. Last write wins |
| `response_model` | Optional pydantic `BaseModel` type for the JSON body |

Raises `TypeSafeError` for invalid requests before sending (e.g. empty questions or an
unreadable asset), `TypeSafeAPIError` (or a subclass) for non-2xx responses after retries,
`TypeSafeAPIConnectionError` or `TypeSafeAPITimeoutError` when the server can't be reached
or the call times out after retries, and `TypeSafeAPIResponseValidationError` when the body
doesn't match the response model [26].

Positional use is documented: `client.system_one(state, questions)` [23].

### 2.7 `models.list()` [26][27]

```python
list(
    *,
    retry: RetryPolicy | None = None,
    timeout: float
    | httpx2.Timeout
    | None = None,
    extra_headers: Mapping[str, str]
    | None = None,
) -> ListModelsResponse
```

`ListModelsResponse.models: tuple[ModelMetadata, ...]`. Each `ModelMetadata` has
`name: str`, `description: str` and `release_date: str` (`YYYY-MM-DD`) [30]. Usage:
`for m in client.models.list().models: print(m.name, m.release_date, m.description)` [17].

### 2.8 Question classes [28]

All question classes are pydantic models with `extra="forbid"` [38]. Pass fields as
**keyword arguments**. A wrong field, or a wrong shape for `criteria`, raises
`pydantic.ValidationError` **at construction**, not `TypeSafeError` **[verified offline]**.
`Content` is `str | Mapping[str, ContentValue | None] | Sequence[ContentValue | None] | Asset`,
and `ContentValue` covers JSON scalars, lists, maps and assets [32]. In other words,
`instructions` and every criterion can be a string, an object, an array, or an `Image`.

#### `Noul` (yes/no)

Documented overloads:

```python
Noul(
    *,
    type: Literal["noul"] = ...,
    instructions: Content,
    criteria: NoulCriteria | None = ...,
)
```
```python
Noul(
    *,
    type: Literal["noul"] = ...,
    instructions: None = ...,
    criteria: _DescribedTrue | _DescribedFalse,
)
```

Fields:

```python
instructions: Content | None = None
```
```python
criteria: NoulCriteria | None = None
```

`NoulCriteria` is a `TypedDict` with optional keys `true: Content | None` and
`false: Content | None`. Build it either way:
`NoulCriteria(true="...", false="...")` or `{"true": "...", "false": "..."}` [8][28].
Unknown keys are rejected **[verified offline]**.

Note **[verified offline]**: `Noul()` with neither `instructions` nor `criteria` passes the
pydantic constructor (only the type checker flags it), **but `system_one` rejects it before
sending**: `normalize_questions` in `_core/questions.py` raises
`TypeSafeError('Noul question "<name>" has neither instructions nor criteria; at least one is required.')`.
The same happens when `instructions` is `None` and every `criteria` value is `None`. It never
reaches the server [38]. Always pass `instructions`.

#### `Choice` (pick one label)

Fields:

```python
criteria: Mapping[str, Content | None]
```
```python
instructions: Content | None = None
```

- `criteria` is a **dict** from label to description. A value of `None` means "label
  only", e.g. `{"calm": None, "angry": None}` [22][28]. A **list is rejected**
  (`ValidationError`) **[verified offline]**.
- The API allows 1–**255** options (`minProperties: 1`, `maxProperties: 255`) [36][6]. The
  SDK does **not** check the 255 maximum: 256 options are sent to the server
  **[verified offline]**, which should reject them (expect 422; unverified, since no live
  call was made). An **empty** dict is different. `Choice(criteria={})` constructs, but
  `system_one` raises
  `TypeSafeError('Choice question "<name>" has no criteria; at least one choice is required.')`
  before sending **[verified offline]** [38].
- Labels are object keys, so they are never replaced by image references. The model sees
  **both** the label and its description [6][11].

#### `Score` (ordered rubric)

Fields:

```python
criteria: Sequence[Content]
```
```python
instructions: Content | None = None
```

- `criteria` is an **ordered list**. Index 0 is score 0, so `["low", "medium", "high"]` is
  0..2 [7][28]. The SDK enforces `min_length=1` [38]. The API allows 1–**10** levels
  (`maxItems: 10`) [36], and the docs say a scale "should have at least two levels" [7].
  The SDK does not check the 10-level maximum: an 11-level Score is sent to the server
  **[verified offline]**, where it should fail (expect 422; unverified). An empty list
  fails at construction with `pydantic.ValidationError` (object form) or with
  `TypeSafeError` in `system_one` (dict form) [38].

#### `Box` (image localisation; needs images and `jev-1.14`)

```python
instructions: Content | None = None
```

Box has no `criteria` [9][28].

#### Dict forms

Questions can also be plain dicts with a `type` key of `"noul"`, `"choice"`, `"score"` or
`"box"`. These are `NoulModel`, `ChoiceModel`, `ScoreModel` and `BoxModel` (TypedDicts).
You can mix dicts and objects in one call [28].

```python
Question: TypeAlias = (
    Noul | Choice | Score | Box | QuestionModel
)
```
```python
Questions: TypeAlias = Mapping[str, Question]
```

Serialising question objects (as in the Twitch cookbook [35], useful for caching and
logging): `q.model_dump(mode="json", exclude_none=True)`. **[verified offline]** This
produces exactly the wire dict, e.g.
`{'type': 'choice', 'instructions': 'pick', 'criteria': {'a': None, 'b': 'desc'}}`.

### 2.9 Response objects [30]

`SystemOneResponse` is a frozen pydantic model (`extra="ignore", frozen=True, strict=True`):

| Attribute | Type | Meaning |
|---|---|---|
| `model` | `str` | Versioned ID that answered (e.g. `"jev-1.13.0"`) |
| `usage` | `Usage` | `input_tokens: int \| None`, `output_tokens: int \| None` |
| `answers` | `dict[str, Answer]` | Every answer keyed by question name. `Answer` is a discriminated union on `type` |
| `nouls` | `dict[str, NoulAnswer]` | Cached property: typed view of Noul answers |
| `choices` | `dict[str, ChoiceAnswer]` | Cached property |
| `scores` | `dict[str, ScoreAnswer]` | Cached property |
| `boxes` | `dict[str, BoxAnswer]` | Cached property |
| `request_id` | `str` | The `x-typesafe-request-id` header (cached property). **Raises `TypeSafeError("The response did not include a request ID.")` when the header is absent**, e.g. behind a proxy or gateway that strips it, or with a `MockTransport` handler that doesn't set it [38] **[verified offline]**. For a non-raising read use `resp.raw_http_response.headers.get("x-typesafe-request-id")` |
| `raw_http_response` | `httpx2.Response` | Raw status, headers and body |

Prefer `resp.choices["x"]` and friends over `resp.answers["x"]` so the type checker sees
concrete types. The docs use both styles.

Per-answer fields:

| Class | Fields (exact names and types) |
|---|---|
| `NoulAnswer` | `type: Literal['noul']`, `noul: float`. This is P(yes/true) in 0..1. There is **no `confidence`** |
| `ChoiceAnswer` | `type: Literal['choice']`, `choice: str` (the argmax label), `confidence: float` (0..1), `probabilities: dict[str, float]` (keyed by label, sums to about 1) |
| `ScoreAnswer` | `type: Literal['score']`, `score: float` (probability-weighted expected level, can be fractional), `confidence: float`, `legend: dict[int, str \| dict[str, Any] \| list[Any]]`, `probabilities: dict[int, float]`. **The SDK keys both by `int`; the HTTP JSON keys them by string** [7] **[verified offline]** |
| `BoxAnswer` | `type: Literal['box']`, `box: BoxCoordinates`, `confidence: float \| None` (`None` when the model provides none) |
| `BoxCoordinates` | `top: float`, `bottom: float`, `left: float`, `right: float`. Fractions of image height (top, bottom) and width (left, right), measured from the top-left corner |

- Do not rely on the key order of `probabilities`. The docs' example responses list labels
  in a different order from the criteria [6].
- Unknown answer kinds are skipped with a logged warning. Read them through
  `resp.raw_http_response.json()["answers"]` [24]. **[verified offline]** An unknown
  `"type"` was dropped from `answers`.
- Unknown extra fields on known responses are ignored [24].

### 2.10 Typed responses (`response_model`) [23]

```python
class BillingResponse(SystemOneResponse):
    billing: NoulAnswer
# result = client.system_one(..., response_model=BillingResponse); result.billing.noul
```

A plain `pydantic.BaseModel` with an `answers: <Model>` field also works [23].

### 2.11 Exceptions [33][38]

```
TypeSafeError(Exception)                        # base; also raised for client-side request errors
├── TypeSafeAPIError                            # non-2xx; .status .body .headers .endpoint .request_id
│   ├── TypeSafeBadRequestError                 # 400
│   ├── TypeSafeAuthenticationError             # 401
│   ├── TypeSafePermissionDeniedError           # 403
│   ├── TypeSafeNotFoundError                   # 404
│   ├── TypeSafeUnprocessableEntityError        # 422
│   ├── TypeSafeRateLimitError                  # 429; .retry_after_ms (float | None)
│   ├── TypeSafeInternalServerError             # any status >= 500 (incl. 529 Overloaded)
│   └── TypeSafeAPIResponseValidationError      # 2xx whose body is invalid; .field_path e.g. "answers.tone.confidence"
└── TypeSafeAPIConnectionError(TypeSafeError, ConnectionError)   # no HTTP response
    └── TypeSafeAPITimeoutError(..., TimeoutError)               # .timeout
```

- Other statuses (e.g. 408, 413, 415) are raised as a plain `TypeSafeAPIError`, per the
  `STATUS_ERROR_TYPES` table in `_core/errors.py` [38].
- **[verified offline]** A mocked `429` with `retry-after-ms: 5` raised
  `TypeSafeRateLimitError` with `status=429` and `retry_after_ms=5.0` after one retry. A
  mocked `529` raised `TypeSafeInternalServerError`.

What the default `RetryPolicy` retries [31]:

| Error | Retried by default? |
|---|---|
| HTTP 408, 429, 500–599 (incl. 529) | **Yes** (`http_statuses`) |
| `TypeSafeAPIConnectionError` | Yes (`api_connection_error=True`) |
| `TypeSafeAPITimeoutError` | Yes (`api_timeout_error=True`) |
| 400, 401, 403, 404, 413, 415, 422 | No. These are bugs in the request or config. Fix them, don't retry |
| `TypeSafeAPIResponseValidationError` | No (it's a 2xx) |
| Construction errors (`TypeSafeError`, pydantic `ValidationError`, `ValueError`) | Not applicable |

### 2.12 `RetryPolicy` (documented signature) [31]

```python
RetryPolicy(
    max_retries: int = 2,
    backoff_initial: float = 0.5,
    backoff_max: float = 5.0,
    backoff_jitter: float = 0.25,
    http_statuses: set[int] = (
        lambda: {408, 429, *range(500, 600)}
    )(),
    respect_retry_after: bool = True,
    api_connection_error: bool = True,
    api_timeout_error: bool = True,
    exceptions: set[
        type[BaseException]
    ] = set(),
    predicate: Callable[[BaseException], bool]
    | None = None,
    timeout: float | None = 30.0,
)
```

| Field | Meaning |
|---|---|
| `max_retries` | Retries after the first attempt. `0` disables retries |
| `backoff_initial` | First delay in seconds, doubled each attempt up to `backoff_max`. `0` disables backoff |
| `backoff_max` | Maximum delay in seconds |
| `backoff_jitter` | Random fraction (0..1) subtracted from each delay |
| `http_statuses` | Status codes to retry |
| `respect_retry_after` | Honour the `Retry-After` and `retry-after-ms` headers |
| `exceptions`, `predicate` | Extra retry triggers |
| `timeout` | **Total retry budget** per SDK call, including the first attempt and the delays. `None` means no limit. The SDK stops *before* a retry whose delay would reach the budget, then re-raises |

How the timeouts interact (from source [38], tenacity `stop_after_attempt | stop_before_delay`):
- The budget does **not** cancel a request already in flight. The per-operation
  `timeout` (default 10 s) bounds each attempt.
- With the defaults, one call can block for many seconds: up to 3 attempts × 10 s per
  operation plus about 1.5 s of backoff, so roughly 31.5 s. That is slightly **over** the
  30 s budget, because the third attempt starts at about 21.5 s and is not cancelled
  (my arithmetic from `_core/retry.py` [38]). `httpx2.Timeout(10.0)` applies 10 s to each
  phase separately (connect, read, write, pool), not to the whole request, so a server that
  trickles bytes can make one attempt last longer still (httpx2 `Timeout` docstring).
- For live use, set both values tight, and add `asyncio.wait_for` as a hard deadline (see
  §7).

---

## 3. HTTP API (for reference, debugging and gateways)

Base URL `https://api.typesafe.ai`. OpenAPI spec `https://api.typesafe.ai/openapi.json`,
`info.version = 0.2.0` [36].

| Endpoint | Purpose |
|---|---|
| `POST /v1/systemone` | Evaluate the questions against the state (`operationId: systemone_v1_systemone_post`) |
| `GET /v1/models` | List the models and aliases available to the account (`ModelMetadataList`) |

### Headers

| Header | Value | When |
|---|---|---|
| `Authorization` | `Bearer <API_KEY>` | Always [36] |
| `Content-Type` | `application/json`, or `multipart/form-data` for image parts | Always |
| `Content-Encoding` | `gzip`, `x-gzip` or `zstd` (one layer only; zstd window ≤ 8 MiB) | Optional [21] |
| `TypeSafe-Danger-Zone-Images` | `true` | **Required** on any request with assets or a Box question. Without it you get `400` [36][11] |
| `TypeSafe-Danger-Zone-Long-Context` | `true` | Raises both context limits to 262k tokens. Same price, slower, less accurate [17] |
| `x-typesafe-request-id` | Response header | Log it [30] |
| `retry-after`, `retry-after-ms` | Response headers on 429 | The SDK honours them [31][38] |

The SDK also sends `User-Agent: typesafe-sdk/0.7.5`, `X-TypeSafe-SDK: typesafe-sdk/0.7.5`,
`X-TypeSafe-Runtime` and `Accept: application/json`. Retries add `X-TypeSafe-Retry-Count`
[38] **[verified offline]**.

### Request body (`SystemOneRequest`) [36]

- Required: `model` (string), `state` (string, object or array) and `questions` (an
  object with at least 1 entry).
- Optional: `assets` (an object with at least 1 entry).

```json
{
  "model": "jev-1.13.0",
  "state": {"situation": {"period": "4th quarter", "margin": "home leads by 2"}},
  "questions": {
    "is_clutch":   {"type": "noul",   "instructions": "Is `situation` a clutch moment?",
                    "criteria": {"true": "Late in a close game", "false": "Anything else"}},
    "tone":        {"type": "choice", "instructions": "Which tone fits `situation`?",
                    "criteria": {"calm": null, "excited": "Big moment", "tense": {"what": "Close and late"}}},
    "drama":       {"type": "score",  "instructions": "How dramatic is `situation`?",
                    "criteria": ["Routine", "Notable", "Decisive"]},
    "rim":         {"type": "box",    "instructions": "the basketball rim in `frame`"}
  }
}
```

The `box` question in that example also needs the images header, an asset and `jev-1.14`.
Per-type schemas [36]:

- `noul`: `instructions` is a string, object, array or null. `criteria` is null or
  `{"true": ..., "false": ...}`.
- `choice`: `criteria` is required, with 1–255 properties. Each value is a string,
  object, array or null.
- `score`: `criteria` is required, an array of 1–10 items. Each item is a string, object
  or array (not null).
- `box`: `instructions` only.

### Response body (`SystemOneResponse`) [36][19]

```json
{"model": "jev-1.13.0",
 "answers": {
   "billing": {"type": "noul", "noul": 0.98},
   "tone": {"type": "choice", "choice": "calm", "confidence": 0.7,
            "probabilities": {"angry": 0.15, "calm": 0.8, "excited": 0.05}},
   "urgency": {"type": "score", "score": 1.7, "confidence": 0.55,
               "legend": {"0": "Can wait", "1": "Needs attention this week", "2": "Needs attention today"},
               "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8}},
   "red_rectangle": {"type": "box", "box": {"top": 0.2, "bottom": 0.66, "left": 0.19, "right": 0.68},
                     "confidence": null}},
 "usage": {"input_tokens": 120, "output_tokens": 12}}
```

(This combines the four documented per-type examples. The `box` example came from
`jev-preview`.)

### Limits and request rules [21][17][11]

| Rule | Limit | Error |
|---|---|---|
| Body size | ≤ 32 MiB (headers excluded). Applies to both the compressed and the decompressed body | 413 |
| Encoding | UTF-8 JSON without a BOM | 400 |
| Nesting | ≤ 64 levels in each of `state`, `instructions` and `criteria` | 400 |
| Numbers | Finite only. `NaN`, `Infinity` and `1e400` are rejected. Send IDs and large exact values as strings | 400 |
| Context | 64k tokens per request (state plus all questions) and 32k per question (state plus one question). 262k for both with the long-context header | 400 `max_tokens_exceeded` |
| Questions per request | No count limit. Only the context limits apply | — |
| Choice options | 1–255 | 422 for > 255 (unverified: inferred from the OpenAPI constraint, not seen live). The Python SDK raises `TypeSafeError` before sending when there are 0 [38] |
| Score levels | 1–10 (use ≥ 2) | 422 for > 10 (unverified, as above). The SDK rejects 0 before sending [38] |
| Compression | Unsupported encoding, or more than one layer | 415. A malformed compressed body is 400 |

Note **[verified offline]**: SDK 0.7.4+ serialises `float('nan')` as the string `"NaN"`
rather than failing [25]. Sanitise NaN and inf from CV pipelines **before** building the
state.

### Status codes [21]

| Code | Meaning |
|---|---|
| 400 | A request rule is broken, or `max_tokens_exceeded` |
| 401 | Missing or invalid key |
| 413 | Too large |
| 415 | Bad `Content-Encoding` |
| 422 | Validation error. The body is `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}` |
| 429 | Rate limited |
| 529 | Overloaded. Retry with backoff |

---

## 4. Confidence semantics [12][8][6][7]

- `confidence` is derived **only** from the answer's own `probabilities`. It is 1 when all
  the mass is on one outcome and 0 when the mass is spread evenly [12].
- **Choice** with n options: `confidence = (p_max − 1/n) / (1 − 1/n)`. Only the top
  probability counts. Check: probabilities 0.8/0.15/0.05 give (0.8−⅓)/(⅔) = 0.70, which
  matches the API example [12][36].
- **Score** with n levels and m the most likely level:
  `confidence = max(0, 1 − Σ pᵢ·|i−m| / MAD_unif)`, where `MAD_unif = (1/n)·Σ|i − (n−1)/2|`.
  Mass on neighbouring levels costs less than mass far away. Check: 0.1/0.1/0.8 gives
  1 − 0.3/(2/3) = 0.55, which matches the API example [12][36].
- **Noul** has no `confidence`. The `noul` probability already carries the uncertainty.
  For a confidence-style number, use `|2p − 1|`, which puts Noul on the same scale as
  Choice [12][8].
- Other statistics the docs suggest: top probability `p_max` (threshold per question,
  because its meaning depends on n) and the top-to-second ratio `p_max/p_second` [12].
- **Box** confidence may be `None`. There is no recommended threshold yet [9].
- Confidence describes how peaked the answer is, not whether it is correct. Calibration is
  measured over groups of predictions [2][7].

Routing thresholds shown in the docs. They are all illustrative, so tune ours on labelled
game footage:

| Where | Rule |
|---|---|
| Confidence page [12] | Three bands: act / caution / don't act. A **< 0.5 floor** goes to a human. High-stakes actions need **> 0.9** |
| Confidence-routing pattern [14] | **< 0.6** routes to a human. Low-stakes actions run at ≥ 0.6. High-stakes actions need **> 0.85**; otherwise confirm with the user |
| Intent-routing pattern [16] | Intent confidence **< 0.5** goes to a human. Also escalate when complexity `score > 1` or its confidence < 0.5 |
| Choice page [6] | `department.confidence < 0.3` goes to manual triage. `resolution.confidence < 0.5` means ask. A second label with probability **> 0.25** gets a copy |
| How-to-build page [4] | `confidence < 0.8` goes to review (example). Score is acted on when `confidence >= 0.7 and score >= 1.5` |
| Noul page [8] | Threshold depends on the cost of each error: 0.5 when symmetric, higher when false-yes is costly, lower when a miss is costly. Use a review band such as `NO=0.2 < p < YES=0.8` |
| Twitch cookbook [35] | `TOXIC_AT = 0.7`, `SPOILER_AT = 0.7` |

CourtVision policy (my recommendation):
- Stats-affecting judgments (anything that would change the live box score) are never
  auto-applied from Jev alone. A Choice must reach ≥ 0.85 **and** agree with the CV event
  engine; anything else is flagged.
- Narration-flavour judgments (tone, highlight-worthiness) can act at ≥ 0.6 / `noul` ≥ 0.7.
- Below the floor, fall back to deterministic templates.
- Re-tune every threshold when the pinned model changes [17].

---

## 5. Images, frames and the Box primitive [11][9][29][23]

**Status: "Dreamer-ready"**, meaning "not yet up to our standards to call ready for
production". The request format, answers, limits and accuracy may change between releases
[11][9].

- **Opt-in:** send the header `TypeSafe-Danger-Zone-Images: true` on every request that
  contains an image or a Box question. Otherwise you get `400` [11]. In the SDK, use
  `TypeSafeClient(headers={"TypeSafe-Danger-Zone-Images": "true"})` or the per-call
  `extra_headers={"TypeSafe-Danger-Zone-Images": "true"}` [23][11].
- **Models:** Jev 1.13 and 1.14 both accept images. **Box needs Jev 1.14**
  (`jev-preview` / `jev-1.14.0`) [11][9].
- **SDK asset type:** `Image(data: Buffer | PathLike[str])`, with constructors
  `Image.from_base64(data: str | Buffer)` and
  `Image.from_pil(image: PIL.Image.Image, format: str = "jpeg")`. `from_pil` needs the
  `pil` extra and encodes JPEG at quality 95. Use `format="png"` for RGBA or P mode [29].
- `Buffer` means the bytes of an **encoded image file** (PNG or JPEG), not a raw pixel
  array. Encode a NumPy frame first, e.g. with `cv2.imencode(".jpg", frame)[1]`.
- `Asset` is currently just `Image` [29]. An `Image` is loaded once and reused [29]. The
  API detects the format from the bytes [29].
- **Three ways to reference an image** [23]:
  1. Put the `Image` object directly in the state or questions, e.g.
     `state={"frame": Image(...)}`. The SDK auto-generates a reference: `str(img)` is
     `<image:{uuid4hex}>` [38] **[verified offline]**.
  2. Use an f-string inside `with client.attach_assets(img):`.
  3. Pass named references through `assets={"<frame>": Image(...)}`.
- Requests with assets go out as `multipart/form-data`, one raw-bytes part per image [23].
- **Placement:** images in `state` are shared by all questions. Images in `instructions`
  are re-sent with every question that uses them. Images can also go in criteria, e.g. as
  example pictures for Choice options. Every use of a reference adds the image to what the
  model reads again [11].
- **Limits** [11]:

  | Limit | Value |
  |---|---|
  | Format | PNG or JPEG stills (animated PNGs are rejected) |
  | File size | ≤ 16 MiB per image |
  | Dimensions | **280–1568 px on each side** |
  | Total pixels | 78,400–2,458,624 |
  | Request body, all images together, and expanded request | 32 MiB each |
  | Reference length | ≤ 1,024 bytes each, ≤ 65,536 bytes in total |

  Out-of-range images are **rejected, never resized**. An image below a minimum returns
  `400`, and one over a size limit (file size, width, height, total pixels, body, all images
  together, expanded request, reference lengths) returns `413` [11]. One bad image fails the
  whole request. The API strips EXIF and does **not** apply EXIF orientation. Every asset must be
  referenced, or you get 400 [11].
- **Box semantics** [9]:
  - The answer is one box per question, with no "not found". Pair it with a Noul such as
    "Is there an X in `frame`?".
  - The docs advise asking one Box per target and cropping or tiling to find small
    targets.
  - TypeSafe's edge-question check caught 30 of 36 truncated crops in their test [9].
- **Frame history** is a documented pattern: a state such as `{"frames": [{"seconds_ago":
  2, "frame": ref}, ...]}`, asking "what changed" [11].

**Is this practical for real-time video? Not on the hot path.**

| Problem | Detail |
|---|---|
| (a) Not production-ready | Explicitly labelled pre-production; the format may change [11] |
| (b) Latency | Text calls alone take ~0.1–0.4 s [4][35]. Image upload size and image-token cost are unknown ("guidance … soon") [11]. 30 fps is impossible, and even 1–2 fps is speculative |
| (c) Minimum size | 280 px per side, so crops of players or the ball must be upscaled or padded [11] |
| (d) Errors | All-or-nothing rejection [11] |
| (e) Rate limits | Images count against the same token rate limits [11] |

What to do instead:
- Keep detection and tracking local (the YOLO / RF-DETR backends in `pyproject.toml`, on
  MPS).
- Send Jev the **text/JSON** event state derived from CV.
- Images are acceptable only for **offline or low-rate experiments**, behind a feature flag
  and pinned to `jev-1.14.0`. Examples: checking whether a keyframe shows a made basket
  when the CV engine is uncertain, or a `Box` for the scoreboard or rim during calibration.

---

## 6. Known weaknesses: Jev 1.13 jaggedness [18] and our rules

The page applies to `jev-1.13` and was last reviewed 2026-10-02 [18].

| # | Failure mode | Doc's fix | CourtVision rule |
|---|---|---|---|
| 1 | Literal reading | State the exact condition and put boundary cases in criteria | Write instructions as exact conditions: say "shot released before the shot clock expired", never "good shot" |
| 2 | **Math, numbers and counting**: "Jev is not a calculator"; it "does not count reliably". Hex/RGB values and other numeric representations underperform, and you should not interpolate Score levels for magnitudes | Keep arithmetic in code. To count, ask one Noul per item and sum in code | **All** points, fouls, runs, percentages, margins, clock arithmetic and distances are computed in Python. Send results as **named buckets or text** (e.g. `"margin": "one-possession game"`, `"time_bucket": "under two minutes"`). Never ask "how many…" |
| 3 | **Dates and times** are read as text, not ordered quantities | Extract components with Choice, compare in code | Game clock, shot clock and timestamps are compared in Python. Send "last 24 seconds of the quarter", not `"00:23.4"` vs `"00:47.1"` |
| 4 | Indirection and double negatives | Reduce hops and point to state paths | Reference fields by backticked paths, e.g. `` `last_play.description` `` [5]. One hop per question |
| 5 | **Large, irrelevant state** causes context rot | Filter first and send only what's needed | Keep the state compact: the current situation, the last 1–3 events and only the players involved. No full box score or play-by-play dump |
| 6 | Adversarial content in state can move answers | Explicit criteria, testing | Never put fan chat or social text in the same state as stats decisions |
| 7 | Contradictory instructions and criteria | Align them | `true` must mean the "yes" reading of the instruction |
| 8 | **Choice option order** bias toward the first option | Reorder and check consistency | Our test suite should shuffle option order and assert the answer stays stable |
| 9 | Generation | Use a generative model | Claude writes the narration. Jev only judges |

Also from the primitives and Score pages [5][7][8]:
- One snap judgment per question.
- One condition per Noul ("angry **and** wants a refund" should be two Nouls).
- Phrase questions so that a high value means yes.
- Score levels must describe **situations, not degrees**. Numbers-only levels like
  `["0","1","2"]` perform badly [7].
- Keep each Score to one dimension. Normalise with `score / (len(criteria) - 1)` before
  weighting [7][15].
- Add `other` / `none` options, or a companion Noul to check that anything fits at all [6].
- Ask **speculative questions** in the same request and ignore the irrelevant ones in code
  [13].
- Questions within one request can't depend on each other. Chain a second request only
  when it truly needs the first answer [5].

---

## 7. How CourtVision should call Jev

Design:
1. The CV and event engine compute every number.
2. A builder turns them into a **compact, bucketed text/JSON state**.
3. **One** Jev request per new event carries every narrow question (speculative fan-out).
4. Code applies confidence gates.
5. Claude narrates from the verified facts plus Jev's labels.
6. Jev can also be used as a grounding check on Claude's non-numeric claims. Numeric claims
   are checked in Python.

### Shared questions and state builder (documented names only)

```python
from typesafe_sdk import Choice, Noul, NoulCriteria, Score

JEV_MODEL = "jev-1.13.0"  # pinned versioned ID; `jev-latest` currently points here

# Built once at import. Question IDs are never sent to the model, so each instruction is a full question.
NARRATION_QUESTIONS = {
    "play_type": Choice(
        instructions="Which description best fits the play in `last_play.description`?",
        criteria={
            "fast_break": "The scoring team pushed the ball up court before the defense was set",
            "half_court": "The offense attacked a defense that was already set",
            "putback": "A shot right after an offensive rebound close to the rim",
            "free_throws": "Points scored only from the free-throw line",
            "other": "None of the descriptions above fits the play",
        },
    ),
    "highlight": Noul(
        instructions="Would a typical fan call `last_play.description` a highlight play?",
        criteria=NoulCriteria(
            true="Spectacular or game-changing: a dunk in traffic, a buzzer-beater, a chase-down block",
            false="A routine basket, free throw, or ordinary possession",
        ),
    ),
    "drama": Score(
        instructions="How much is at stake in the moment described by `situation`?",
        criteria=[
            "Routine stretch of the game with no special stakes",
            "Notable stretch: a scoring run, a lead change, or foul trouble for a key player",
            "Decisive moment: the final minutes of a one- or two-possession game",
        ],
    ),
    "mention_fouls": Noul(
        instructions="Should a commentator mention the foul situation in `situation.foul_note` right now?",
    ),
}


def build_state(snapshot: dict) -> dict:
    """All numbers are computed and bucketed in Python before they reach Jev."""
    return {
        "situation": {
            "period": snapshot["period_label"],          # e.g. "4th quarter"
            "time_bucket": snapshot["time_bucket"],      # e.g. "under two minutes left" (computed from the clock)
            "margin": snapshot["margin_text"],           # e.g. "home leads by 2, a one-possession game"
            "run": snapshot["run_text"],                 # e.g. "away team on an 8-0 run" (counted in Python)
            "foul_note": snapshot["foul_note"],          # e.g. "home #23 has five fouls; one more fouls him out"
        },
        "last_play": {"description": snapshot["last_play_text"]},  # templated from CV events
    }
```

### Minimal sync example

```python
import logging

from typesafe_sdk import RetryPolicy, TypeSafeClient, TypeSafeError

log = logging.getLogger("courtvision.jev")

# One long-lived client. Construction raises TypeSafeError if TYPESAFE_API_KEY is missing or invalid.
client = TypeSafeClient(
    model=JEV_MODEL,
    timeout=1.0,  # seconds per HTTP operation (SDK default is 10.0)
    retry=RetryPolicy(max_retries=1, backoff_initial=0.05, backoff_max=0.2, timeout=1.5),
)


def judge(snapshot: dict) -> dict | None:
    try:
        resp = client.system_one(build_state(snapshot), NARRATION_QUESTIONS)
    except TypeSafeError as err:  # base of API, connection, timeout, and response-validation errors
        log.warning("jev unavailable: %r", err)
        return None  # caller falls back to deterministic narration templates

    play = resp.choices["play_type"]
    drama = resp.scores["drama"]
    return {
        "play_type": play.choice if play.confidence >= 0.6 else None,
        "highlight": resp.nouls["highlight"].noul >= 0.8,
        "drama_level": round(drama.score) if drama.confidence >= 0.5 else None,  # 0..2
        "mention_fouls": resp.nouls["mention_fouls"].noul >= 0.7,
        "jev_model": resp.model,                 # versioned ID that answered
        "input_tokens": resp.usage.input_tokens,  # int | None
        # `resp.request_id` raises TypeSafeError if the header is missing, and this line is outside the try,
        # so read the header without raising:
        "request_id": resp.raw_http_response.headers.get("x-typesafe-request-id"),
    }

# On shutdown: client.close()   (or create it with `with TypeSafeClient(...) as client:`)
```

### Minimal async example (live loop, hard deadline, bounded concurrency)

```python
import asyncio
import logging

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, SystemOneResponse, TypeSafeError

log = logging.getLogger("courtvision.jev")
JEV_DEADLINE_S = 1.2   # hard wall-clock budget; RetryPolicy.timeout does not cancel in-flight attempts


async def judge_async(client: AsyncTypeSafeClient, sem: asyncio.Semaphore, snapshot: dict) -> SystemOneResponse | None:
    async with sem:
        try:
            return await asyncio.wait_for(
                client.system_one(build_state(snapshot), NARRATION_QUESTIONS),
                timeout=JEV_DEADLINE_S,
            )
        except (TypeSafeError, TimeoutError) as err:  # TypeSafeAPITimeoutError is also a TimeoutError
            log.warning("jev skipped: %r", err)
            return None


async def run(events: asyncio.Queue) -> None:
    async with AsyncTypeSafeClient(
        model=JEV_MODEL,
        timeout=1.0,
        retry=RetryPolicy(max_retries=1, backoff_initial=0.05, backoff_max=0.2, timeout=1.5),
    ) as client:
        sem = asyncio.Semaphore(8)  # the Twitch cookbook ran 8 concurrent requests
        # NOTE: as written, each call is awaited inline, so only one request is ever in flight and the
        # semaphore never limits anything. To get real concurrency, start each judge_async(...) with
        # asyncio.create_task (or a TaskGroup) and handle the results as they complete.
        while True:
            snapshot = await events.get()
            resp = await judge_async(client, sem, snapshot)
            if resp is None or snapshot.get("superseded"):
                continue  # stale or failed: narration uses templates
            play = resp.choices["play_type"]
            if play.confidence >= 0.6 and resp.nouls["highlight"].noul >= 0.8:
                ...  # hand {play.choice, resp.scores["drama"].score} to the Claude narrator
```

### Grounding check for Claude's narration (non-numeric claims only)

Claude returns its line plus a list of atomic claims. Check numeric claims (points, fouls,
times) against the stats store **in Python**. Ask Jev only about semantic claims, one Noul
per claim, all in one request (the pattern from the Noul page's structured instructions
[8]):

```python
from typesafe_sdk import Noul

def grounding_questions(claims: list[str]) -> dict[str, Noul]:
    return {
        f"claim_{i}": Noul(
            instructions={
                "claim": claim,
                "question": "Is `claim` directly supported by `events`?",
            },
        )
        for i, claim in enumerate(claims)
    }

# resp = client.system_one({"events": recent_event_texts}, grounding_questions(claims))
# unsupported = [i for i in range(len(claims)) if resp.nouls[f"claim_{i}"].noul < 0.5]  # regenerate or drop those
```

Operational notes:
- Log `resp.model`, `resp.request_id` and `resp.usage.input_tokens` for every call.
- Keep the state under about 1 KiB where possible. Below 1 KiB compression is skipped, so
  latency is lower.
- Never set `TYPESAFE_LOG_LEVEL=debug` in production, because request bodies are logged
  unredacted [24].
- Offline tests: use `TypeSafeClient(api_key="test", transport=httpx2.MockTransport(handler),
  compression=False)`. **[verified offline]** this exercises the real request build and
  response parse. The handler should set an `x-typesafe-request-id` response header, because
  otherwise `resp.request_id` raises `TypeSafeError` [38].

---

## 8. Documentation inconsistencies and gaps

1. **Images vs "text only".** The System One concept page says Jev "accepts text input
   only… Images… are not supported (yet)" [2], and the Models table says Input: "Text"
   [17]. The Images page says Jev 1.13 and 1.14 accept images (dreamer-ready, opt-in) [11].
   Treat images as experimental.
2. **Odd date.** SDK basic usage says "As of November 14, 2026, image support is only
   dreamer-ready" [23]. That date is in the future relative to today (2026-10-20), so it
   is probably a typo.
3. **Parallel cookbook figures.** The primitives page says 11.5x cheaper / 9.6x faster
   [5]; the cookbook and llms.txt say 12.2x / 10.0x [39][41].
4. **Undocumented model strings** in examples: `"jev"` [23], `"jev-1.13"` [18],
   `"jev-1.12"` [39]. Only `jev-1.14.0`, `jev-1.13.0`, `jev-latest` and `jev-preview` are
   documented on the Models page [17].
5. **No percentile latency SLA** and no published default rate limits [4][17].
6. **Image token cost** is not yet estimable [11].

---

## Sources

1. https://docs.typesafe.ai/introduction.md
2. https://docs.typesafe.ai/concepts/system-one.md
3. https://docs.typesafe.ai/concepts/state.md
4. https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md
5. https://docs.typesafe.ai/primitives.md
6. https://docs.typesafe.ai/primitives/choice.md
7. https://docs.typesafe.ai/primitives/score.md
8. https://docs.typesafe.ai/primitives/noul.md
9. https://docs.typesafe.ai/primitives/box.md
10. https://docs.typesafe.ai/primitives/advanced.md
11. https://docs.typesafe.ai/images.md
12. https://docs.typesafe.ai/confidence.md
13. https://docs.typesafe.ai/patterns/fan-out.md
14. https://docs.typesafe.ai/patterns/confidence-routing.md
15. https://docs.typesafe.ai/patterns/composite-scoring.md
16. https://docs.typesafe.ai/patterns/intent-routing.md
17. https://docs.typesafe.ai/models.md
18. https://docs.typesafe.ai/model-jaggedness/jev-1.13.md
19. https://docs.typesafe.ai/api/system-one.md
20. https://docs.typesafe.ai/api/list-models.md
21. https://docs.typesafe.ai/api/errors.md
22. https://docs.typesafe.ai/sdk/python.md
23. https://docs.typesafe.ai/sdk/python/usage/basic.md
24. https://docs.typesafe.ai/sdk/python/usage/advanced.md
25. https://docs.typesafe.ai/sdk/python/changelog.md
26. https://docs.typesafe.ai/sdk/python/api/clients/sync.md
27. https://docs.typesafe.ai/sdk/python/api/clients/async.md
28. https://docs.typesafe.ai/sdk/python/api/types/questions.md
29. https://docs.typesafe.ai/sdk/python/api/types/assets.md
30. https://docs.typesafe.ai/sdk/python/api/types/responses.md
31. https://docs.typesafe.ai/sdk/python/api/retries.md
32. https://docs.typesafe.ai/sdk/python/api/types/common.md
33. https://docs.typesafe.ai/sdk/python/api/exceptions.md
34. https://docs.typesafe.ai/sdk/python/api/constants.md
35. https://docs.typesafe.ai/cookbooks/twitch_chat_filter.md
36. https://api.typesafe.ai/openapi.json
37. https://pypi.org/project/typesafe-sdk/ (read via https://pypi.org/pypi/typesafe-sdk/json)
38. https://github.com/typesafe-ai/typesafe-sdk-python (tag v0.7.5: `src/typesafe_sdk/__init__.py`, `_core/config.py`, `_core/compression.py`, `_core/constants.py`, `_core/errors.py`, `_core/retry.py`, `_core/question_types.py`, `_core/assets.py`, `pyproject.toml`)
39. https://docs.typesafe.ai/cookbooks/parallel_questions.md
40. https://docs.typesafe.ai/introduction/quickstart.md
41. https://docs.typesafe.ai/llms.txt
42. https://github.com/typesafe-ai/typesafe-sdk-python/blob/v0.7.5/src/typesafe_sdk/_core/questions.py (client-side question validation; added by the fact-check)
43. https://github.com/typesafe-ai/typesafe-sdk-python/blob/v0.7.5/src/typesafe_sdk/_core/schemas/base.py (`request_id` / `raw_http_response`; added by the fact-check)

---

## Verification log

Adversarial fact-check run on 2026-10-20. I re-fetched every page below as raw `.md` or JSON,
read the SDK source at tag `v0.7.5` (commit `f36e6854`) from GitHub, and re-ran behaviour
checks offline against the repo's `.venv` (`typesafe-sdk 0.7.5`, `httpx2 2.15.1`, no
`backports.zstd`) using `httpx2.MockTransport`. No live API calls were made.

### Confirmed as written

| Claim | Evidence |
|---|---|
| PyPI `typesafe-sdk` 0.7.5, uploaded 2026-10-14T21:21:41. `requires_python >=3.10`, `license_expression: MIT`. Deps include `httpx2>=2.0.0`. Extras `http2`, `pil`, `zstd` (`backports-zstd` only when `python_full_version < "3.14"`). Release dates of 0.0.1a0 through 0.7.4 | https://pypi.org/pypi/typesafe-sdk/json |
| `TypeSafeClient` / `AsyncTypeSafeClient` keyword-only signatures, including the async `httpx2.AsyncBaseTransport` / `httpx2.AsyncClient` types. `close()` / `__enter__` / `__exit__` on sync; `aclose()` / `__aenter__` / `__aexit__` on async. `ValueError` when both `transport` and `http_client` are passed | https://docs.typesafe.ai/sdk/python/api/clients/sync.md, https://docs.typesafe.ai/sdk/python/api/clients/async.md, `_core/client/{sync,aio}/client.py` |
| `system_one(state, questions, *, model, assets, retry, timeout, extra_headers, extra_body, response_model)` and its 2 overloads. Empty mapping raises `TypeSafeError("At least one question is required.")` (re-run offline) | sync.md; `_core/questions.py` |
| Question classes: `extra="forbid"`. `Choice.criteria: Mapping[str, Content \| None]`. `Score.criteria: Sequence[Content] = Field(min_length=1)` (an empty list raises `pydantic.ValidationError`, re-run offline). `Noul.criteria: NoulCriteria \| None` (TypedDict, `total=False`, `closed=True`). `Box` has only `instructions` | `_core/question_types.py`; https://docs.typesafe.ai/sdk/python/api/types/questions.md |
| `SystemOneResponse` fields and cached views. Answer field names and types. `ScoreAnswer.legend` / `probabilities` are int-keyed in the SDK (re-run offline: keys `[0, 1, 2]`) and string-keyed in the JSON. `Usage.input_tokens/output_tokens: int \| None` (re-run offline: missing fields give `None`). Config is `extra="ignore", frozen=True, strict=True` | `_core/response_types.py`, `_core/schemas/base.py`; https://docs.typesafe.ai/sdk/python/api/types/responses.md; https://docs.typesafe.ai/primitives/score.md |
| Exception tree and `STATUS_ERROR_TYPES` (400/401/403/404/422/429). `>= 500` maps to `TypeSafeInternalServerError` (re-run offline: 529 gives that, after 2 attempts with `max_retries=1`). 408 gives a plain `TypeSafeAPIError` (re-run). `retry_after_ms` | https://docs.typesafe.ai/sdk/python/api/exceptions.md; `_core/errors.py` |
| `RetryPolicy` defaults (`2, 0.5, 5.0, 0.25, {408, 429, *range(500, 600)}, True, True, True, set(), None, 30.0`). Stop rule `stop_after_attempt(max_retries + 1) \| stop_before_delay(timeout)` | https://docs.typesafe.ai/sdk/python/api/retries.md; `_core/retry.py` |
| Constants: `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`, `TYPESAFE_LOG_LEVEL`, `https://api.typesafe.ai`, `jev-latest`, `DEFAULT_TIMEOUT = 10.0`. The log level is applied at import (`setup_logging()` runs at module level) | https://docs.typesafe.ai/sdk/python/api/constants.md; `_core/logging.py` |
| Models: `jev-1.14.0` (from October 14, 2026) and `jev-1.13.0`. `jev-latest` points to `jev-1.13.0`, `jev-preview` to `jev-1.14.0`. $42/Btok = $0.042/Mtok, output free. 64k / 32k context, 262k with `TypeSafe-Danger-Zone-Long-Context: true`. Pin the versioned ID after tuning thresholds. `/v1/models` lists aliases | https://docs.typesafe.ai/models.md |
| HTTP: `POST /v1/systemone` (`systemone_v1_systemone_post`). `required: [model, state, questions]`. `questions` / `assets` `minProperties: 1`. Choice `minProperties 1` / `maxProperties 255`. Score `minItems 1` / `maxItems 10`. `info.version 0.2.0`. 32 MiB → 413. 64 nesting levels, finite numbers, UTF-8 without BOM → 400. gzip / x-gzip / zstd, zstd window ≤ 8 MiB, 415. 529 Overloaded | https://api.typesafe.ai/openapi.json; https://docs.typesafe.ai/api/errors.md |
| Confidence formulas for Choice, Score (`MAD_unif`) and Noul (`\|2p-1\|`). Worked checks 0.70 and 0.55 recomputed. Thresholds: 0.5 / 0.9 (confidence page), 0.6 / 0.85 (confidence-routing), 0.5 and `score > 1` (intent-routing), 0.3 / 0.5 / 0.25 (Choice), 0.8 and `>= 0.7 and >= 1.5` (how-to-build), YES=0.8 / NO=0.2 (Noul), `TOXIC_AT` / `SPOILER_AT = 0.7` (Twitch) | https://docs.typesafe.ai/confidence.md and the pattern / primitive pages in Sources |
| Images: dreamer-ready. `TypeSafe-Danger-Zone-Images: true` on every request with an image or Box, otherwise 400. PNG/JPEG stills, 16 MiB per image, 280–1568 px per side, 78,400–2,458,624 px. 32 MiB ×3. References ≤ 1,024 B each / 65,536 B total. Never resized. EXIF stripped. Box needs Jev 1.14. 30 of 36 truncated crops caught | https://docs.typesafe.ai/images.md; https://docs.typesafe.ai/primitives/box.md |
| Latency: "Most queries complete in about 100 ms". Twitch median "about 0.4 seconds" on `jev-1.13.0`, 2026-09-25, `max_workers=8`, 3 questions. Parallel cookbook 0.27 s vs 2.71 s, 12.2x cheaper / 10.0x faster, `jev-1.12`, 53,777 characters. Primitives page says 11.5x / 9.6x | https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md; https://docs.typesafe.ai/cookbooks/twitch_chat_filter.md; https://docs.typesafe.ai/cookbooks/parallel_questions.md; https://docs.typesafe.ai/primitives.md |
| Jaggedness (applies to `jev-1.13`, last reviewed 2026-10-02): the 9 failure modes, counting / math / numeric representations / dates, context rot, first-option bias | https://docs.typesafe.ai/model-jaggedness/jev-1.13.md |
| Compression: `MIN_COMPRESSED_BODY_LENGTH = 1024` with a `>=` check. `compression=None` silently resolves to no compression without zstd. `"zstd"` raises the exact quoted `TypeSafeError` (re-run). NaN / inf are sent as the strings `"NaN"` / `"Infinity"` (`inf_nan_mode="strings"`, re-run). The 0.7.4 changelog entry | `_core/compression.py`, `_core/transport.py`, `_core/json.py`; https://docs.typesafe.ai/sdk/python/changelog.md |
| Example code in §7: re-executed offline against a mock. The questions serialise to the expected wire types, and `judge()` returns the expected dict | local run |

### Changed

1. **Empty `Choice` criteria.** The note said an empty dict "passes client-side" and is
   rejected by the server. In fact `system_one` raises `TypeSafeError('Choice question "<name>" has no criteria; at least one choice is required.')`
   before sending (re-run offline). Only the 255 upper bound goes unchecked by the SDK [42].
2. **`Noul()` with neither field.** The note said the server's behaviour was unverified. In
   fact `system_one` rejects it client-side with `TypeSafeError('Noul question "<name>" has neither instructions nor criteria; at least one is required.')`
   (re-run offline) [42].
3. **`resp.request_id` can raise.** It raises `TypeSafeError("The response did not include a request ID.")`
   when the `x-typesafe-request-id` header is absent (re-run offline) [43]. The §7 sync
   example read it outside its `try`, so I changed it to
   `resp.raw_http_response.headers.get("x-typesafe-request-id")`. I also added a note to
   the offline-testing tip and to the §2.9 table.
4. **Retry worst case.** The note said "within a 30 s budget". With the defaults the worst
   case is about 31.5 s, because the in-flight attempt is not cancelled, and
   `httpx2.Timeout(10.0)` applies per phase rather than per request. I also tightened the
   TL;DR row to match.
5. **Models table "Input" column.** The Models page says only "Text". I reworded the cells to
   show both sources instead of attributing image support to the Models table.
6. **Limits table.** I marked the 422 status for >255 options and >10 levels as
   (unverified), since it is inferred from the OpenAPI constraints and was never seen live.
   I also noted that 0 options or levels are rejected client-side.
7. **Additions.** Image statuses (400 below a minimum, 413 over a size limit) [11]. The
   extra SDK headers `X-TypeSafe-Runtime` and `Accept: application/json` [38]. An RLCD
   citation [17]. The Score empty-list and over-10-levels behaviour. A comment in the async
   example: as written it awaits each call inline, so the `Semaphore(8)` never allows more
   than one request in flight.

### Still unverified (not testable without a live key)

- The server's status code for 256 Choice options or 11 Score levels (expected 422).
- Default per-organization rate-limit numbers, which are not published.
- Real-world latency from CourtVision's network. The 150–500 ms budget remains the author's
  estimate.
- Image token cost: TypeSafe says guidance is coming "soon".
