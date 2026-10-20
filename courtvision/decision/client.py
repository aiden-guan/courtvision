"""TypeSafe Jev client: bundles, hard deadlines, per-bundle breakers and the live/stub/replay/off modes.

Design: ``docs/research/jev_integration_design.md`` §4 and §7. Every SDK name used here is listed in
``docs/research/sdk_ground_truth.md`` (typesafe-sdk 0.7.5). The client never raises into callers:
every failure becomes a ``JevResult`` with ``source="fallback"`` and an ``error`` reason.

Modes
-----
* ``live``: the real SDK against ``TYPESAFE_BASE_URL`` (key from ``TYPESAFE_API_KEY``).
* ``stub``: the REAL SDK request build and response parse over ``httpx2.MockTransport``, answered
  by ``StubAnswers`` (documented wire shape).
* ``replay``: recorded raw responses (see ``record=`` and ``load_replay``) served through the same mock.
* ``off``: no calls; every request returns a fallback result immediately.

Keys starting with ``_`` anywhere in a state are local metadata (fallback inputs, redaction lists)
and are stripped by ``sanitize`` before anything leaves the process.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import hashlib
import json
import logging
import math
import numbers
import threading
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx2
import numpy as np
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Image,
    RetryPolicy,
    SystemOneResponse,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAuthenticationError,
    TypeSafeBadRequestError,
    TypeSafeClient,
    TypeSafeError,
    TypeSafeNotFoundError,
    TypeSafePermissionDeniedError,
    TypeSafeRateLimitError,
    TypeSafeUnprocessableEntityError,
)

log = logging.getLogger(__name__)

JEV_MODEL_TEXT = "jev-1.13.0"  # pinned (D2): thresholds are tuned per version
JEV_MODEL_IMAGES = "jev-1.14.0"  # Box needs 1.14; cold path only
IMAGES_HEADER = {"TypeSafe-Danger-Zone-Images": "true"}  # required on any request with images/Box
REQUEST_ID_HEADER = "x-typesafe-request-id"
STUB_API_KEY = "stub-key-not-a-secret"  # dummy key for the mock transport; never sent anywhere

MODES = ("live", "stub", "replay", "off")
Mode = Literal["live", "stub", "replay", "off"]
Source = Literal["jev", "cache", "stub", "replay", "fallback"]


# --------------------------------------------------------------------------- configuration


def _default_deadlines() -> dict[str, float]:
    # Initial per-bundle hard deadlines (D7; design §4). Reset each from the measured p95 (§7A.3).
    return {"B1": 1.5, "B2": 1.0, "B3": 1.0, "B4": 0.8, "B5": 0.7, "B6": 5.0}


@dataclass
class JevConfig:
    """Jev decision-layer settings (plain fields so YAML can load them by name)."""

    enabled: bool = True
    mode: str = "live"  # "live" | "stub" | "replay" | "off"
    model: str = JEV_MODEL_TEXT  # pinned text model (D2); answers from any other model are untuned
    image_model: str = JEV_MODEL_IMAGES  # B6 only
    deadlines_s: dict[str, float] = field(default_factory=_default_deadlines)  # hard wall clock per bundle
    max_in_flight: int = 8  # design §2: one asyncio pool, <= 8 requests in flight
    cache_enabled: bool = False  # §7A.9: off until the 20-identical-requests determinism test passes
    images_enabled: bool = False  # B6 only when enabled; images are "dreamer-ready" (design §3 site 8)
    cache_maxsize: int = 2048
    cache_ttl_s: float = 120.0
    breaker_threshold: int = 3  # consecutive failures that open a bundle's circuit (design §7.2 proposal)
    breaker_cooldown_s: float = 30.0  # open duration before a trial call (design §7.2 proposal)
    rate_limit_min_open_s: float = 1.0  # floor for a 429 trip when retry-after is missing/short
    replay_path: str | None = None  # recorded JSONL for mode="replay" when no mapping is passed in
    thresholds: dict[str, float] = field(default_factory=dict)  # overrides of decoders.Thresholds fields
    fallback_rules: dict[str, Any] = field(default_factory=dict)  # overrides of decoders.FallbackRules fields


# --------------------------------------------------------------------------- bundles


@dataclass(frozen=True)
class Bundle:
    """One request type: its questions share one state and one request (design §4)."""

    key: str  # "B1".."B6"
    name: str
    deadline_s: float  # hard wall clock (asyncio.wait_for); RetryPolicy.timeout does not cancel in flight
    http_timeout_s: float  # SDK per-HTTP-operation timeout
    retry: RetryPolicy  # SDK retry budget; 400/401/403/404/413/415/422 are never retried by the SDK
    model: str = JEV_MODEL_TEXT
    images: bool = False
    cacheable: bool = True


# Only 429 and 5xx (incl. 529) are retried: the SDK default also retries 408, but no other 4xx is
# ever retried here (400/401/403/404/413/415/422 are request or config bugs).
RETRY_STATUSES = frozenset({429, *range(500, 600)})


def _retry(max_retries: int, backoff_initial: float = 0.05, backoff_max: float = 0.2,
           timeout: float | None = None) -> RetryPolicy:
    return RetryPolicy(max_retries=max_retries, backoff_initial=backoff_initial, backoff_max=backoff_max,
                       timeout=timeout, http_statuses=set(RETRY_STATUSES))


# Initial values from design §4 (D7); reset each from the measured p95 (RESEARCH.md §7A.3).
BUNDLES: dict[str, Bundle] = {
    "B1": Bundle("B1", "possession_end", 1.5, 1.0, _retry(1, timeout=1.5)),
    "B2": Bundle("B2", "dead_ball_fan", 1.0, 0.7, _retry(1, timeout=1.0)),
    "B3": Bundle("B3", "dead_ball_coach", 1.0, 0.7, _retry(1, timeout=1.0)),
    "B4": Bundle("B4", "line_guard", 0.8, 0.6, _retry(1, backoff_max=0.1, timeout=0.8)),
    "B5": Bundle("B5", "event_ambiguity", 0.7, 0.6, _retry(0, timeout=0.7)),
    "B6": Bundle("B6", "cold_images", 5.0, 4.0, _retry(1, backoff_initial=0.5, backoff_max=5.0, timeout=5.0),
                 model=JEV_MODEL_IMAGES, images=True, cacheable=False),
}


@dataclass(frozen=True)
class JevResult:
    """Outcome of one bundle request. ``ok`` is False for every fallback."""

    bundle: str  # bundle key, "B1".."B6"
    ok: bool
    source: Source
    response: SystemOneResponse | None = None
    error: str | None = None  # fallback reason, e.g. "deadline", "circuit_open", "TypeSafeInternalServerError"
    latency_ms: float = 0.0
    model: str | None = None  # resp.model that answered
    request_id: str | None = None
    input_tokens: int | None = None
    tag: Any = None  # caller's state_version / event id, echoed back for staleness checks


# --------------------------------------------------------------------------- helpers


def _key_text(k: Any) -> str:
    return str(k.value) if isinstance(k, enum.Enum) else str(k)


def sanitize(obj: Any) -> Any:
    """JSON-safe copy of a state: NaN/inf -> None, enums -> values, tuples/sets -> lists, numpy
    scalars/arrays -> Python values, ``_keys`` dropped.

    The SDK would otherwise send the strings "NaN"/"Infinity", and it cannot encode numpy or set
    values at all (one such value would fail the request as a client-side bug and disable the
    bundle). SDK image assets pass through; any other object is dropped (``None``) rather than
    sent as its repr, which could carry names.
    """
    if obj is None or type(obj) is bool or type(obj) is str:
        return obj
    if isinstance(obj, enum.Enum):
        return sanitize(obj.value)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, Mapping):
        return {_key_text(k): sanitize(v) for k, v in obj.items() if not _key_text(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [sanitize(v) for v in sorted(obj, key=repr)]  # deterministic order
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        return sanitize(obj.item())
    if isinstance(obj, Image):
        return obj
    log.warning("jev state value of type %s dropped: not JSON-serialisable", type(obj).__name__)
    return None


def wire_questions(questions: Mapping[str, Any]) -> dict[str, Any]:
    """The exact wire dict of each question (``model_dump(mode="json", exclude_none=True)``)."""
    return {k: (q if isinstance(q, dict) else q.model_dump(mode="json", exclude_none=True))
            for k, q in questions.items()}


def request_key(model: str, state: Any, wire_q: Mapping[str, Any]) -> str:
    """Cache / replay key. Order-PRESERVING (no sort_keys): reversed-option questions must not collide."""
    blob = json.dumps({"model": model, "state": state, "questions": wire_q},
                      separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def state_bytes(state: Any) -> int:
    """Size of the sanitized state as compact UTF-8 JSON (the budget unit of design §4)."""
    return len(json.dumps(sanitize(state), separators=(",", ":"), ensure_ascii=False, default=str).encode())


def parse_replay_lines(lines: Iterable[str]) -> dict[str, dict]:
    """Recorded JSONL lines (as written from ``record=`` dicts) -> {request_key: raw response body}."""
    out: dict[str, dict] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("key") and isinstance(rec.get("raw"), dict):
            out[rec["key"]] = rec["raw"]
    return out


def load_replay(path: str | Path) -> dict[str, dict]:
    """Read a recorded JSONL file for ``mode="replay"`` (read-only)."""
    with open(path, encoding="utf-8") as fh:
        return parse_replay_lines(fh)


def _request_id(resp: SystemOneResponse) -> str | None:
    # resp.request_id raises when the header is absent; read the raw header instead.
    try:
        return resp.raw_http_response.headers.get(REQUEST_ID_HEADER)
    except TypeSafeError:
        return None


def _percentile(values: Iterable[float], q: float) -> float | None:
    s = sorted(values)
    if not s:
        return None
    idx = min(len(s) - 1, max(0, math.ceil(q / 100.0 * len(s)) - 1))  # nearest-rank
    return round(s[idx], 1)


# --------------------------------------------------------------------------- cache and breaker


class TTLCache:
    """LRU with a TTL, keyed by ``request_key``. Present but disabled by default (§7A.9)."""

    def __init__(self, maxsize: int = 2048, ttl_s: float = 120.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._d: OrderedDict[str, tuple[float, JevResult]] = OrderedDict()
        self.maxsize, self.ttl_s, self._clock = maxsize, ttl_s, clock

    def get(self, key: str) -> JevResult | None:
        hit = self._d.get(key)
        if hit is None or self._clock() - hit[0] > self.ttl_s:
            self._d.pop(key, None)
            return None
        self._d.move_to_end(key)
        return hit[1]

    def put(self, key: str, value: JevResult) -> None:
        self._d[key] = (self._clock(), value)
        self._d.move_to_end(key)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)

    def __len__(self) -> int:
        return len(self._d)


class Breaker:
    """Opens after ``threshold`` consecutive failures for ``cooldown_s`` (or the server's retry-after).

    After the cooldown the circuit is half-open: calls go through, a success closes it and the
    first failure re-opens it immediately.
    """

    def __init__(self, threshold: int = 3, cooldown_s: float = 30.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.threshold, self.cooldown_s, self._clock = threshold, cooldown_s, clock
        self._fails = 0
        self._open_until = 0.0
        self._tripped = False
        self.trips = 0

    @property
    def state(self) -> str:
        if not self._tripped:
            return "closed"
        return "open" if self._clock() < self._open_until else "half_open"

    def allow(self) -> bool:
        return self._clock() >= self._open_until

    def success(self) -> None:
        self._fails, self._tripped = 0, False

    def failure(self) -> None:
        if self._tripped and self.allow():  # half-open trial failed
            self.trip(self.cooldown_s)
            return
        self._fails += 1
        if self._fails >= self.threshold:
            self.trip(self.cooldown_s)

    def trip(self, seconds: float) -> None:
        self._open_until = self._clock() + seconds
        self._fails, self._tripped = 0, True
        self.trips += 1


# --------------------------------------------------------------------------- stub answers


def choice_confidence(probs: Mapping[str, float]) -> float:
    """Choice confidence (p_max - 1/n) / (1 - 1/n), as the API computes it (jev_api.md §4)."""
    n = len(probs)
    if n < 2:
        return 0.0
    p = max(probs.values())
    return max(0.0, (p - 1.0 / n) / (1.0 - 1.0 / n))


def score_confidence(probs: Mapping[int, float]) -> float:
    """Score confidence 1 - MAD/MAD_uniform around the modal level (jev_api.md §4)."""
    n = len(probs)
    if n < 2:
        return 0.0
    m = max(probs, key=lambda k: probs[k])
    mad_unif = sum(abs(i - (n - 1) / 2.0) for i in range(n)) / n
    mad = sum(p * abs(int(i) - int(m)) for i, p in probs.items())
    return 0.0 if mad_unif == 0 else max(0.0, 1.0 - mad / mad_unif)


def stub_noul(p: float) -> dict:
    """A Noul answer in the documented wire shape."""
    return {"type": "noul", "noul": float(p)}


def stub_choice(probabilities: Mapping[str, float]) -> dict:
    """A Choice answer (argmax label + API confidence) in the documented wire shape."""
    probs = {str(k): float(v) for k, v in probabilities.items()}
    top = max(probs, key=lambda k: probs[k])
    return {"type": "choice", "choice": top, "confidence": choice_confidence(probs), "probabilities": probs}


def stub_score(probabilities: Iterable[float]) -> dict:
    """A Score answer (expected level + API confidence); the legend is filled from the question."""
    probs = {i: float(p) for i, p in enumerate(probabilities)}
    score = sum(i * p for i, p in probs.items())
    return {"type": "score", "score": score, "confidence": score_confidence(probs),
            "probabilities": {str(i): p for i, p in probs.items()}}


@dataclass
class StubAnswers:
    """Canned answers for ``mode="stub"``. The default is maximally uncertain, so decoders fall back.

    Tests register confident answers per question name in ``overrides`` and can force latency,
    an HTTP error status, extra headers, or a different answering ``model``.
    """

    overrides: dict[str, dict] = field(default_factory=dict)  # question name -> wire answer dict
    model: str | None = None  # response "model"; None echoes the requested model
    delay_s: float = 0.0  # handler sleeps this long before answering
    status: int = 200  # non-2xx returns an error body (error-path tests)
    headers: dict[str, str] = field(default_factory=dict)  # extra response headers, e.g. retry-after-ms
    hits: int = 0  # requests that reached the mock transport
    last_body: dict | None = None  # the last wire request body (tests inspect what left the process)

    def build(self, body: Mapping[str, Any]) -> dict:
        """Response body for one wire request body."""
        answers: dict[str, dict] = {}
        for name, q in body["questions"].items():
            ans = dict(self.overrides[name]) if name in self.overrides else self._uncertain(q)
            if ans.get("type") == "score" and "legend" not in ans:
                ans["legend"] = {str(i): c for i, c in enumerate(q.get("criteria", []))}
            answers[name] = ans
        return {"model": self.model or body["model"], "answers": answers,
                "usage": {"input_tokens": len(json.dumps(body)) // 4, "output_tokens": 0}}

    @staticmethod
    def _uncertain(q: Mapping[str, Any]) -> dict:
        kind = q.get("type")
        if kind == "choice":
            labels = list(q["criteria"])
            return {"type": "choice", "choice": labels[0], "confidence": 0.0,
                    "probabilities": {k: 1.0 / len(labels) for k in labels}}
        if kind == "score":
            n = len(q["criteria"])
            return {"type": "score", "score": (n - 1) / 2.0, "confidence": 0.0,
                    "probabilities": {str(i): 1.0 / n for i in range(n)}}
        return {"type": "noul", "noul": 0.5}


# --------------------------------------------------------------------------- stats


@dataclass
class _BundleStats:
    calls: int = 0
    answered: int = 0
    fallbacks: Counter = field(default_factory=Counter)
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=4096))
    input_tokens: int = 0
    models: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {"calls": self.calls, "answered": self.answered, "fallbacks": dict(self.fallbacks),
                "p50_ms": _percentile(self.latencies_ms, 50), "p95_ms": _percentile(self.latencies_ms, 95),
                "input_tokens": self.input_tokens, "models": dict(self.models)}


# --------------------------------------------------------------------------- client


class JevClient:
    """One instance per process. Never raises into callers: every failure becomes a fallback result."""

    def __init__(self, config: JevConfig | None = None, *, stub: StubAnswers | None = None,
                 replay: Mapping[str, dict] | None = None, record: Callable[[dict], None] | None = None,
                 cache: TTLCache | None = None, breakers: Mapping[str, Breaker] | None = None,
                 api_key: str | None = None, base_url: str | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.config = config or JevConfig()
        mode = self.config.mode if self.config.enabled else "off"
        if mode not in MODES:
            raise ValueError(f"JevConfig.mode must be one of {MODES}, got {mode!r}")
        self.mode: Mode = mode  # type: ignore[assignment]
        self.stub = stub or StubAnswers()
        if replay is None and self.mode == "replay" and self.config.replay_path:
            replay = load_replay(self.config.replay_path)
        self._replay = dict(replay or {})
        self._record = record
        self._clock = clock
        self._cache = cache or TTLCache(self.config.cache_maxsize, self.config.cache_ttl_s, clock)
        self._breakers: dict[str, Breaker] = dict(breakers or {})
        self._bundles = {k: self._configured(b) for k, b in BUNDLES.items()}
        self._api_key, self._base_url = api_key, base_url
        self._sem = asyncio.Semaphore(max(1, self.config.max_in_flight))
        self._disabled_bundles: set[str] = set()
        self._session_disabled: str | None = None  # reason; set on 401/403 or client construction failure
        self._local = threading.local()  # one sync client per thread
        self._sync_clients: list[TypeSafeClient] = []
        self._aclient: AsyncTypeSafeClient | None = None
        self._stats: dict[str, _BundleStats] = {k: _BundleStats() for k in BUNDLES}
        self._served = 0

    # ---------------------------------------------------------------- public API

    @property
    def available(self) -> bool:
        """False when off, session-disabled (401/403) or the client cannot be built (no key)."""
        if self.mode == "off" or self._session_disabled:
            return False
        if self._aclient is None:
            try:
                self._async()
            except TypeSafeError as err:
                self._mark_unbuildable(err)
                return False
        return True

    @property
    def disabled_reason(self) -> str | None:
        """Why the client is unavailable ("mode_off", an auth error name, "client_unavailable"), or None."""
        if self.mode == "off":
            return "mode_off"
        return self._session_disabled

    def bundle(self, b: Bundle | str) -> Bundle:
        """The configured bundle (deadline and pinned model from ``JevConfig``) for a key or Bundle."""
        return self._bundles[b] if isinstance(b, str) else b

    def breaker(self, b: Bundle | str) -> Breaker:
        """The bundle's own circuit breaker (created on first use)."""
        key = self.bundle(b).key
        if key not in self._breakers:
            self._breakers[key] = Breaker(self.config.breaker_threshold, self.config.breaker_cooldown_s,
                                          self._clock)
        return self._breakers[key]

    def ask(self, b: Bundle | str, state: Any, questions: Mapping[str, Any], *,
            assets: Mapping[str, Any] | None = None, tag: Any = None) -> JevResult:
        """Sync call for offline evaluation and scripts. No hard cancel: worst case is bounded by
        (max_retries + 1) x http_timeout + backoff."""
        bundle = self.bundle(b)
        state = sanitize(state)
        key, early = self._precheck(bundle, state, questions, assets, tag)
        if early is not None:
            return early
        t0 = time.perf_counter()
        try:
            client = self._sync()
        except TypeSafeError as err:
            return self._unbuildable(bundle, err, t0, tag)
        try:
            resp = client.system_one(state, questions, **self._kwargs(bundle, assets))
        except Exception as err:  # noqa: BLE001 - never raise into callers
            return self._failed(bundle, err, t0, tag)
        return self._succeeded(bundle, key, state, questions, resp, t0, tag)

    async def aask(self, b: Bundle | str, state: Any, questions: Mapping[str, Any], *,
                   assets: Mapping[str, Any] | None = None, tag: Any = None) -> JevResult:
        """Async call with a hard wall-clock deadline (``asyncio.wait_for``) covering the queue wait
        for the in-flight semaphore and every SDK attempt."""
        bundle = self.bundle(b)
        state = sanitize(state)
        key, early = self._precheck(bundle, state, questions, assets, tag)
        if early is not None:
            return early
        t0 = time.perf_counter()
        try:
            client = self._async()
        except TypeSafeError as err:
            return self._unbuildable(bundle, err, t0, tag)
        slot = [False]  # set once the in-flight semaphore is acquired
        try:
            resp = await asyncio.wait_for(self._guarded(client, bundle, state, questions, assets, slot),
                                          timeout=bundle.deadline_s)
        except Exception as err:  # noqa: BLE001 - TimeoutError (deadline) and every SDK error
            if isinstance(err, TimeoutError) and not isinstance(err, TypeSafeError) and not slot[0]:
                # Expired while queued behind max_in_flight: local overload, not evidence that the
                # service is failing, so the bundle's breaker is not charged.
                ms = (time.perf_counter() - t0) * 1000
                log.warning("jev %s fallback after %.0f ms queued for a free slot", bundle.key, ms)
                return self._fallback(bundle, "queue_timeout", tag, ms)
            return self._failed(bundle, err, t0, tag)
        return self._succeeded(bundle, key, state, questions, resp, t0, tag)

    async def warm_up(self) -> bool:
        """One cheap B5-sized call at boot to open the connection pool; True when it answered."""
        r = await self.aask("B5", {"shot": {"view_quality": "clear view of the rim"}},
                            {"warm": {"type": "noul", "instructions": "Is `shot.view_quality` about a rim?"}})
        return r.ok

    def stats(self) -> dict[str, Any]:
        """Per-bundle calls, answers, fallback reasons, p50/p95 latency (answered calls plus
        deadline overruns, counted at the time they were cut), tokens and models seen."""
        models: Counter = Counter()
        for s in self._stats.values():
            models.update(s.models)
        return {
            "mode": self.mode,
            "available": self.available,  # same answer as the property (a live client with no key is not)
            "disabled_reason": self.disabled_reason,
            "disabled_bundles": sorted(self._disabled_bundles),
            "breakers": {k: br.state for k, br in sorted(self._breakers.items())},
            "bundles": {k: {"name": self._bundles[k].name, **s.as_dict()} for k, s in self._stats.items()},
            "models_seen": sorted(models),
        }

    def close(self) -> None:
        """Close every sync client this instance created."""
        for c in self._sync_clients:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                log.debug("jev sync client close failed", exc_info=True)
        self._sync_clients.clear()
        self._local = threading.local()

    async def aclose(self) -> None:
        """Close the async client (and any sync clients)."""
        if self._aclient is not None:
            client, self._aclient = self._aclient, None
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                log.debug("jev async client close failed", exc_info=True)
        self.close()

    # ---------------------------------------------------------------- construction (lazy)

    def _configured(self, b: Bundle) -> Bundle:
        model = self.config.image_model if b.images else self.config.model
        deadline = float(self.config.deadlines_s.get(b.key, b.deadline_s))
        return dataclasses.replace(b, model=model, deadline_s=deadline)

    def _sync(self) -> TypeSafeClient:
        c = getattr(self._local, "client", None)
        if c is None:
            if self.mode == "live":
                c = TypeSafeClient(api_key=self._api_key, base_url=self._base_url, model=self.config.model,
                                   timeout=1.0, retry=RetryPolicy(max_retries=0))
            else:
                c = TypeSafeClient(api_key=STUB_API_KEY, model=self.config.model, compression=False,
                                   transport=httpx2.MockTransport(self._handle))
            self._local.client = c
            self._sync_clients.append(c)
        return c

    def _async(self) -> AsyncTypeSafeClient:
        if self._aclient is None:
            if self.mode == "live":
                self._aclient = AsyncTypeSafeClient(api_key=self._api_key, base_url=self._base_url,
                                                    model=self.config.model, timeout=1.0,
                                                    retry=RetryPolicy(max_retries=0))
            else:  # async MockTransport with an async handler (sdk_ground_truth.md)
                self._aclient = AsyncTypeSafeClient(api_key=STUB_API_KEY, model=self.config.model,
                                                    compression=False,
                                                    transport=httpx2.MockTransport(self._ahandle))
        return self._aclient

    def _mark_unbuildable(self, err: BaseException) -> None:
        if not self._session_disabled:
            self._session_disabled = "client_unavailable"
            log.warning("jev disabled: client could not be constructed (%s)", type(err).__name__)

    # ---------------------------------------------------------------- internals

    async def _guarded(self, client: AsyncTypeSafeClient, b: Bundle, state: Any, questions: Mapping[str, Any],
                       assets: Mapping[str, Any] | None, slot: list[bool]) -> SystemOneResponse:
        async with self._sem:
            slot[0] = True
            return await client.system_one(state, questions, **self._kwargs(b, assets))

    def _kwargs(self, b: Bundle, assets: Mapping[str, Any] | None) -> dict[str, Any]:
        kw: dict[str, Any] = {"model": b.model, "retry": b.retry, "timeout": b.http_timeout_s}
        if assets:
            kw["assets"] = assets
        if b.images:
            kw["extra_headers"] = IMAGES_HEADER
        return kw

    def _fallback(self, b: Bundle, why: str, tag: Any, ms: float = 0.0) -> JevResult:
        st = self._stats[b.key]
        st.calls += 1
        st.fallbacks[why] += 1
        return JevResult(bundle=b.key, ok=False, source="fallback", error=why, latency_ms=ms, tag=tag)

    def _cache_on(self, b: Bundle, assets: Mapping[str, Any] | None) -> bool:
        return self.config.cache_enabled and b.cacheable and not assets

    def _precheck(self, b: Bundle, state: Any, questions: Mapping[str, Any],
                  assets: Mapping[str, Any] | None, tag: Any) -> tuple[str | None, JevResult | None]:
        if self.mode == "off":
            return None, self._fallback(b, "mode_off", tag)
        if self._session_disabled:
            return None, self._fallback(b, "session_disabled", tag)
        if b.images and not self.config.images_enabled:
            return None, self._fallback(b, "images_disabled", tag)
        if b.key in self._disabled_bundles:
            return None, self._fallback(b, "bundle_disabled", tag)
        if not self.breaker(b).allow():
            return None, self._fallback(b, "circuit_open", tag)
        key = None
        if self._cache_on(b, assets) or self.mode == "replay" or self._record is not None:
            try:
                key = request_key(b.model, state, wire_questions(questions))
            except Exception as err:  # noqa: BLE001 - malformed question objects: our bug
                log.error("jev %s request key failed: %r", b.key, err)
                self._disabled_bundles.add(b.key)
                return None, self._fallback(b, "bad_request", tag)
        if self.mode == "replay" and key not in self._replay:
            return key, self._fallback(b, "replay_miss", tag)
        if key is not None and self._cache_on(b, assets):
            hit = self._cache.get(key)
            if hit is not None:
                st = self._stats[b.key]
                st.calls += 1
                st.answered += 1
                return key, dataclasses.replace(hit, source="cache", tag=tag, latency_ms=0.0)
        return key, None

    def _unbuildable(self, b: Bundle, err: BaseException, t0: float, tag: Any) -> JevResult:
        self._mark_unbuildable(err)
        return self._fallback(b, "client_unavailable", tag, (time.perf_counter() - t0) * 1000)

    def _failed(self, b: Bundle, err: BaseException, t0: float, tag: Any) -> JevResult:
        ms = (time.perf_counter() - t0) * 1000
        reason = type(err).__name__
        if isinstance(err, (TypeSafeAuthenticationError, TypeSafePermissionDeniedError)):
            self._session_disabled = reason  # 401/403: config bug, stop for the session
            log.error("jev disabled for the session after %s", reason)
        elif isinstance(err, (TypeSafeBadRequestError, TypeSafeUnprocessableEntityError, TypeSafeNotFoundError)):
            self._disabled_bundles.add(b.key)  # our request is wrong: fix it, never retry
            log.error("jev bundle %s disabled after %s: %s", b.key, reason, err)
        elif isinstance(err, TypeSafeRateLimitError):
            wait_s = (err.retry_after_ms or 0.0) / 1000.0
            self.breaker(b).trip(max(self.config.rate_limit_min_open_s, wait_s))
        elif isinstance(err, TypeSafeError) and not isinstance(err, (TypeSafeAPIError, TypeSafeAPIConnectionError)):
            self._disabled_bundles.add(b.key)  # client-side request validation error: our bug
            log.error("jev bundle %s disabled after client-side error: %s", b.key, err)
        else:
            if isinstance(err, TimeoutError) and not isinstance(err, TypeSafeError):
                reason = "deadline"
                # A deadline overrun is a right-censored latency sample: keep it in p50/p95 so the
                # measured tail used to reset deadlines (design §11.5) is not biased low.
                self._stats[b.key].latencies_ms.append(ms)
            elif not isinstance(err, TypeSafeError):
                log.warning("jev %s unexpected error", b.key, exc_info=err)
            self.breaker(b).failure()  # deadline, connection, 5xx/529, 408/413/415, invalid 2xx body
        log.warning("jev %s fallback after %.0f ms: %s", b.key, ms, reason)
        return self._fallback(b, reason, tag, ms)

    def _succeeded(self, b: Bundle, key: str | None, state: Any, questions: Mapping[str, Any],
                   resp: SystemOneResponse, t0: float, tag: Any) -> JevResult:
        self.breaker(b).success()
        ms = (time.perf_counter() - t0) * 1000
        rid = _request_id(resp)
        tokens = resp.usage.input_tokens
        source: Source = "jev" if self.mode == "live" else self.mode  # type: ignore[assignment]
        r = JevResult(bundle=b.key, ok=True, source=source, response=resp, latency_ms=ms,
                      model=resp.model, request_id=rid, input_tokens=tokens, tag=tag)
        st = self._stats[b.key]
        st.calls += 1
        st.answered += 1
        st.latencies_ms.append(ms)
        st.input_tokens += tokens or 0
        st.models[resp.model] += 1
        log.info("jev %s model=%s request_id=%s input_tokens=%s questions=%d latency_ms=%.0f",
                 b.key, resp.model, rid, tokens, len(questions), ms)
        if resp.model != b.model:
            log.error("jev %s answered by %s, pinned %s: decoders treat it as untuned", b.key, resp.model, b.model)
        if key is not None and self._cache_on(b, None):
            self._cache.put(key, r)
        if self._record is not None and self.mode != "replay":
            self._record_response(b, key, state, questions, resp, r)
        return r

    def _record_response(self, b: Bundle, key: str | None, state: Any, questions: Mapping[str, Any],
                         resp: SystemOneResponse, r: JevResult) -> None:
        try:
            raw = resp.raw_http_response.json()
            self._record({"key": key or request_key(b.model, state, wire_questions(questions)),
                          "bundle": b.key, "model": resp.model, "request_id": r.request_id,
                          "latency_ms": round(r.latency_ms, 1), "raw": raw})
        except Exception:  # noqa: BLE001 - recording must never break a call
            log.warning("jev %s record failed", b.key, exc_info=True)

    # ---------------------------------------------------------------- mock transport (stub / replay)

    def _respond(self, request: httpx2.Request) -> httpx2.Response:
        self._served += 1
        headers = {REQUEST_ID_HEADER: f"{self.mode}-{self._served:06d}"}
        try:
            body = json.loads(request.content)  # compression=False, so the body is plain JSON
        except ValueError:  # multipart image bodies are never stubbed (design §7.2)
            return httpx2.Response(503, json={"detail": "mock cannot answer multipart requests"}, headers=headers)
        if self.mode == "replay":
            raw = self._replay.get(request_key(body["model"], body["state"], body["questions"]))
            if raw is None:
                return httpx2.Response(503, json={"detail": "replay miss"}, headers=headers)
            return httpx2.Response(200, json=raw, headers=headers)
        self.stub.hits += 1
        self.stub.last_body = body
        headers.update(self.stub.headers)
        if not 200 <= self.stub.status < 300:
            return httpx2.Response(self.stub.status, json={"detail": "stub error"}, headers=headers)
        return httpx2.Response(200, json=self.stub.build(body), headers=headers)

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        if self.mode == "stub" and self.stub.delay_s > 0:
            time.sleep(self.stub.delay_s)
        return self._respond(request)

    async def _ahandle(self, request: httpx2.Request) -> httpx2.Response:
        if self.mode == "stub" and self.stub.delay_s > 0:
            await asyncio.sleep(self.stub.delay_s)
        return self._respond(request)
