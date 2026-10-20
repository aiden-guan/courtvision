"""Async pub/sub hub between the pipeline and dashboard WebSocket clients.

The pipeline publishes JSON-able messages (ARCHITECTURE "server/" schema: every message has
``type`` and ``t``). The hub

* encodes each message once (numpy, enums, dataclasses, NaN -> null),
* keeps replay state for late joiners: the latest ``meta`` / ``stats`` / ``frame`` (and
  ``status``) plus the last N events (latest revision per ``event_id``), utterances, insights
  and captions,
* fans messages out to per-client bounded queues (drop-oldest, superseded snapshots first),
* throttles snapshot types per client (``frame`` <= 15 Hz) with latest-wins coalescing.

All state mutation and fan-out run on the bound asyncio loop. ``publish_threadsafe`` may be
called from any thread (e.g. the perception thread); it snapshots and encodes the message in
the caller's thread and hands it to the loop with ``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import datetime as _dt
import enum
import itertools
import json
import logging
import math
import threading
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any

import numpy as np

from courtvision.types import Event, FrameState, Insight, Roster, Team, Utterance

log = logging.getLogger(__name__)

__all__ = [
    "Hub",
    "HubClient",
    "HubConfig",
    "HubJSONEncoder",
    "caption_msg",
    "dumps",
    "event_msg",
    "frame_msg",
    "insight_msg",
    "message",
    "meta_msg",
    "stats_msg",
    "status_msg",
    "to_jsonable",
    "utterance_msg",
]


# --------------------------------------------------------------------------- config


@dataclass
class HubConfig:
    """Hub tuning. All values are engineering defaults (status: tuned unless noted)."""

    # Per-client live queue capacity; on overflow the oldest superseded snapshot is dropped,
    # else the oldest message (tuned: ~8 s of 15 Hz frames plus events for a stalled tab).
    queue_size: int = 256
    # Events / utterances / insights / captions replayed to a late joiner
    # (ARCHITECTURE "server/" and task spec: last 200; sourced).
    replay_history: int = 200
    # Per-client max delivery rate per message type, latest-wins (frame <= 15 Hz: task spec,
    # sourced; stats 4 Hz: tuned safety cap above the ~1 Hz producer rate from ARCHITECTURE).
    max_rate_hz: dict[str, float] = field(default_factory=lambda: {"frame": 15.0, "stats": 4.0})
    # Message types whose latest instance is kept and replayed (snapshot semantics).
    latest_types: list[str] = field(default_factory=lambda: ["meta", "status", "stats", "frame"])
    # Event kinds left out of the WS replay (high-frequency, never narrated; ARCHITECTURE
    # "Shared event semantics": DRIBBLE "not narrated"). They stay in the REST event log.
    replay_exclude_event_kinds: list[str] = field(default_factory=lambda: ["dribble"])
    # REST event log cap (latest revision per event id). A full game is a few thousand events
    # including dribbles (tuned).
    event_log_max: int = 20_000
    # REST transcript cap (utterances), tuned.
    transcript_max: int = 5_000
    # Close a client whose queue had to drop a non-superseded message, so the browser
    # reconnects and receives a consistent replay instead of silently missing events (tuned).
    disconnect_on_overflow: bool = True


# --------------------------------------------------------------------------- JSON encoding

_MAX_DEPTH = 64


def _json_key(k: Any) -> str:
    if isinstance(k, enum.Enum):
        k = k.value
    if isinstance(k, np.generic):
        k = k.item()
    if isinstance(k, str):
        return k
    if isinstance(k, float) and not math.isfinite(k):
        return "null"
    if k is None or isinstance(k, bool):
        return json.dumps(k)
    return str(k)


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Convert ``obj`` to plain JSON types.

    numpy scalars/arrays -> Python numbers/lists, Enum -> value, dataclass -> ``to_dict()`` when
    it has one else its fields, sets/tuples/deques -> lists, NaN/inf -> None, mapping keys ->
    strings (Enum keys -> their value).
    """
    if _depth > _MAX_DEPTH:
        raise ValueError("object nesting too deep (cycle?) for JSON encoding")
    d = _depth + 1
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, enum.Enum):  # before str/int: Team is a str-mixin Enum
        return to_jsonable(obj.value, d)
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item(), d)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist(), d)
    if isinstance(obj, Mapping):
        return {_json_key(k): to_jsonable(v, d) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _dataclass_jsonable(obj, d)
    if isinstance(obj, (set, frozenset)):
        items = [to_jsonable(v, d) for v in obj]
        try:
            return sorted(items)
        except TypeError:
            return items
    if isinstance(obj, (list, tuple, deque)):
        return [to_jsonable(v, d) for v in obj]
    return _other_jsonable(obj, d)


def _dataclass_jsonable(obj: Any, depth: int) -> Any:
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_jsonable(to_dict(), depth)
        except TypeError:
            pass  # to_dict with a different signature; fall back to fields
    return {f.name: to_jsonable(getattr(obj, f.name), depth) for f in dataclasses.fields(obj)}


def _other_jsonable(obj: Any, depth: int) -> Any:
    model_dump = getattr(obj, "model_dump", None)  # pydantic v2 models
    if callable(model_dump):
        return to_jsonable(model_dump(), depth)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(obj)).decode("ascii")
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        return obj.isoformat()
    if isinstance(obj, PurePath):
        return str(obj)
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_jsonable(to_dict(), depth)
    if isinstance(obj, Iterable):
        return [to_jsonable(v, depth) for v in obj]
    log.debug("JSON fallback to str() for %s", type(obj).__name__)
    return str(obj)


def dumps(obj: Any) -> str:
    """Compact, strict JSON (no NaN tokens) for anything :func:`to_jsonable` accepts."""
    return json.dumps(to_jsonable(obj), allow_nan=False, separators=(",", ":"), ensure_ascii=False)


class HubJSONEncoder(json.JSONEncoder):
    """``json.JSONEncoder`` that applies :func:`to_jsonable` first (so NaN becomes null)."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["allow_nan"] = False
        super().__init__(**kwargs)

    def encode(self, o: Any) -> str:
        return super().encode(to_jsonable(o))

    def iterencode(self, o: Any, _one_shot: bool = False) -> Iterable[str]:
        return super().iterencode(to_jsonable(o), _one_shot)

    def default(self, o: Any) -> Any:  # pragma: no cover - to_jsonable already converted
        return to_jsonable(o)


# --------------------------------------------------------------------------- message builders


def message(type_: str, t: float | None, **fields: Any) -> dict[str, Any]:
    """Build a raw message ``{"type": type_, "t": t, **fields}``."""
    return {"type": type_, "t": t, **fields}


def event_msg(event: Event) -> dict[str, Any]:
    return message("event", event.t, event=event.to_dict())


def utterance_msg(utterance: Utterance) -> dict[str, Any]:
    return message("utterance", utterance.t, utterance=to_jsonable(utterance))


def insight_msg(insight: Insight) -> dict[str, Any]:
    return message("insight", insight.t, insight=to_jsonable(insight))


def stats_msg(stats: Any) -> dict[str, Any]:
    """``stats`` is a ``GameStats`` (anything with ``to_dict()`` and ``t``) or a dict."""
    payload = stats.to_dict() if callable(getattr(stats, "to_dict", None)) else stats
    t = getattr(stats, "t", None)
    if t is None and isinstance(payload, Mapping):
        t = payload.get("t")
    return message("stats", t, stats=payload)


def caption_msg(t: float, pts: float | None, text: str, channel: str = "caption") -> dict[str, Any]:
    """Sidecar caption / audio-description cue keyed to source video PTS (RESEARCH §7A.8)."""
    return message("caption", t, pts=pts, text=text, channel=channel)


def status_msg(t: float | None, **fields: Any) -> dict[str, Any]:
    """Pipeline status snapshot (e.g. ``paused``, ``speed``, ``muted``, ``source``)."""
    return message("status", t, **fields)


def _fields_of(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return dict(obj)
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


def meta_msg(
    roster: Roster | None,
    court: Any,
    rules: Any = None,
    t: float | None = 0.0,
    colors: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Game metadata: roster, team names, CourtSpec fields, RuleSpec fields, landmarks."""
    entries = roster.entries if roster is not None else []
    names = roster.team_names if roster is not None else {Team.HOME: "Home", Team.AWAY: "Away"}
    landmarks_fn = getattr(court, "landmarks", None)
    landmarks = landmarks_fn() if callable(landmarks_fn) else {}
    msg = message(
        "meta",
        t,
        roster=[_fields_of(e) for e in entries],
        teams={"home": names.get(Team.HOME, "Home"), "away": names.get(Team.AWAY, "Away")},
        court=_fields_of(court),
        rules=_fields_of(rules) if rules is not None else None,
        landmarks={str(k): list(v) for k, v in landmarks.items()},
    )
    if colors:
        msg["colors"] = dict(colors)
    return msg


def _r(v: float | None, nd: int = 2) -> float | None:
    if v is None:
        return None
    f = float(v)
    return round(f, nd) if math.isfinite(f) else None


def frame_msg(
    frame: FrameState,
    roster: Roster | None = None,
    holder_id: str | None = None,
    possession: Team | None = None,
) -> dict[str, Any]:
    """Compact top-down state for the live court (ARCHITECTURE WebSocket ``frame`` schema)."""
    players = []
    for p in frame.players:
        entry = roster.get(p.player_id) if roster is not None else None
        players.append(
            {
                "id": p.player_id,
                "team": p.team.value if p.team else None,
                "jersey": p.jersey if p.jersey is not None else (entry.jersey if entry else None),
                "name": entry.name if entry else None,
                "x": _r(p.x),
                "y": _r(p.y),
                "speed": _r(p.speed),
            }
        )
    ball = frame.ball
    bx, by = (ball.court_xy if ball is not None and ball.court_xy is not None else (None, None))
    sb = frame.scoreboard
    return message(
        "frame",
        frame.t,
        frame_idx=frame.frame_idx,
        players=players,
        ball={
            "x": _r(bx),
            "y": _r(by),
            "z": _r(ball.z) if ball is not None else None,
            "visible": bool(ball.visible) if ball is not None else False,
            "holder": holder_id,
        },
        possession=possession.value if possession else None,
        clock={
            "period": sb.period if sb else None,
            "game": _r(sb.game_clock, 1) if sb else None,
            "shot": _r(sb.shot_clock, 1) if sb else None,
        },
    )


# --------------------------------------------------------------------------- internals


@dataclass(slots=True)
class _Prepared:
    mtype: str
    msg: dict[str, Any]
    text: str


@dataclass(slots=True)
class _Stored:
    seq: int
    msg: dict[str, Any]
    text: str


class HubClient:
    """One subscriber: replay buffer, bounded live queue, per-type rate limits."""

    def __init__(
        self,
        client_id: int,
        maxsize: int,
        min_interval_s: Mapping[str, float],
        snapshot_types: Iterable[str],
        loop: asyncio.AbstractEventLoop,
        clock: Callable[[], float],
    ) -> None:
        self.id = client_id
        self.dropped = 0
        self.overflowed = False  # a non-superseded message was dropped
        self.closed = False
        self._maxsize = max(1, int(maxsize))
        self._min_interval = dict(min_interval_s)
        self._snapshot_types = frozenset(snapshot_types) | frozenset(self._min_interval)
        self._loop = loop
        self._clock = clock
        self._replay: deque[str] = deque()
        self._queue: deque[tuple[str, str]] = deque()
        self._type_counts: Counter[str] = Counter()
        self._wakeup = asyncio.Event()
        self._pending: dict[str, str] = {}
        self._last_sent: dict[str, float] = {}
        self._timers: dict[str, asyncio.TimerHandle] = {}

    # -- producer side (loop thread) --------------------------------------------------------

    def load_replay(self, texts: Iterable[str], types: Iterable[str] = ()) -> None:
        now = self._clock()
        self._replay.extend(texts)
        for mtype in types:  # replayed snapshots count against the rate limit
            if mtype in self._min_interval:
                self._last_sent[mtype] = now
        self._wakeup.set()

    def offer(self, mtype: str, text: str) -> None:
        """Queue a message, applying the per-type rate limit (latest wins)."""
        if self.closed:
            return
        interval = self._min_interval.get(mtype)
        if not interval:
            self._enqueue(mtype, text)
            return
        now = self._clock()
        last = self._last_sent.get(mtype, -math.inf)
        if mtype not in self._timers and now - last >= interval:
            self._last_sent[mtype] = now
            self._enqueue(mtype, text)
            return
        self._pending[mtype] = text
        if mtype not in self._timers:
            delay = max(0.0, interval - (now - last))
            self._timers[mtype] = self._loop.call_later(delay, self._flush, mtype)

    def flush_pending(self) -> None:
        """Deliver every coalesced message now (ignores the rate limit)."""
        for mtype in list(self._pending):
            timer = self._timers.pop(mtype, None)
            if timer is not None:
                timer.cancel()
            self._flush(mtype)

    def _flush(self, mtype: str) -> None:
        self._timers.pop(mtype, None)
        text = self._pending.pop(mtype, None)
        if text is None or self.closed:
            return
        self._last_sent[mtype] = self._clock()
        self._enqueue(mtype, text)

    def _enqueue(self, mtype: str, text: str) -> None:
        if len(self._queue) >= self._maxsize:
            self._drop_one()
        self._queue.append((mtype, text))
        self._type_counts[mtype] += 1
        self._wakeup.set()

    def _drop_one(self) -> None:
        # Prefer the oldest snapshot that a newer queued message of the same type supersedes.
        for i, (mtype, _) in enumerate(self._queue):
            if mtype in self._snapshot_types and self._type_counts[mtype] > 1:
                del self._queue[i]
                self._type_counts[mtype] -= 1
                self.dropped += 1
                return
        mtype, _ = self._queue.popleft()
        self._type_counts[mtype] -= 1
        self.dropped += 1
        self.overflowed = True
        log.warning("hub client %d overflowed: dropped a %s message", self.id, mtype)

    def close(self) -> None:
        self.closed = True
        for timer in self._timers.values():
            timer.cancel()
        self._timers.clear()
        self._pending.clear()
        self._wakeup.set()

    # -- consumer side ------------------------------------------------------------------------

    def qsize(self) -> int:
        return len(self._replay) + len(self._queue)

    def get_nowait(self) -> str | None:
        """Next encoded message, replay first; None when nothing is queued."""
        if self._replay:
            return self._replay.popleft()
        if self._queue:
            mtype, text = self._queue.popleft()
            self._type_counts[mtype] -= 1
            return text
        return None

    async def get(self) -> str | None:
        """Wait for the next encoded message; None once the client is closed and drained."""
        while True:
            text = self.get_nowait()
            if text is not None:
                return text
            if self.closed:
                return None
            self._wakeup.clear()
            await self._wakeup.wait()


# --------------------------------------------------------------------------- hub


class Hub:
    """Async pub/sub with replay for late joiners (see module docstring)."""

    def __init__(self, config: HubConfig | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.config = config or HubConfig()
        self._clock = clock
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.RLock()
        self._seq = itertools.count(1)
        self._client_ids = itertools.count(1)
        self._clients: dict[int, HubClient] = {}
        self._latest: dict[str, _Stored] = {}
        self._event_log: OrderedDict[str, _Stored] = OrderedDict()
        self._event_replay: OrderedDict[str, _Stored] = OrderedDict()
        self._transcript: deque[_Stored] = deque(maxlen=max(1, self.config.transcript_max))
        self._insights: OrderedDict[str, _Stored] = OrderedDict()
        self._captions: deque[_Stored] = deque(maxlen=max(1, self.config.replay_history))
        self._latest_types = frozenset(self.config.latest_types)
        self._replay_exclude = frozenset(self.config.replay_exclude_event_kinds)
        self._min_interval = {k: 1.0 / v for k, v in self.config.max_rate_hz.items() if v and v > 0}
        self.published: Counter[str] = Counter()

    # -- loop binding ---------------------------------------------------------------------------

    def bind_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Bind the loop that owns client queues (defaults to the running loop)."""
        self._loop = loop or asyncio.get_running_loop()

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        return self._loop

    # -- publishing ---------------------------------------------------------------------------

    async def publish(self, msg: Mapping[str, Any] | Event | Utterance | Insight) -> None:
        """Publish from a coroutine on the hub's loop."""
        prepared = self._prepare(msg)
        running = asyncio.get_running_loop()
        bound = self._loop
        if bound is not None and bound is not running and bound.is_running() and not bound.is_closed():
            bound.call_soon_threadsafe(self._ingest, prepared)  # foreign loop: hand over
            return
        if bound is None or bound.is_closed() or not bound.is_running():
            self._loop = running
        self._ingest(prepared)

    def publish_threadsafe(self, msg: Mapping[str, Any] | Event | Utterance | Insight) -> None:
        """Publish from any thread. The message is snapshotted and encoded immediately."""
        prepared = self._prepare(msg)
        loop = self._loop
        # A bound loop that is stopped (not closed) would accept the callback but never run it,
        # silently losing the message; only hand over to a loop that is running.
        if loop is not None and loop.is_running() and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._ingest, prepared)
                return
            except RuntimeError:  # loop closed between the check and the call
                pass
        # No running loop means no client is being served: just record the state for replay.
        self._ingest(prepared, fanout=False)

    def _prepare(self, msg: Mapping[str, Any] | Event | Utterance | Insight) -> _Prepared:
        if isinstance(msg, Event):
            msg = event_msg(msg)
        elif isinstance(msg, Utterance):
            msg = utterance_msg(msg)
        elif isinstance(msg, Insight):
            msg = insight_msg(msg)
        if not isinstance(msg, Mapping):
            raise TypeError(f"hub messages must be mappings, got {type(msg).__name__}")
        mtype = msg.get("type")
        if not isinstance(mtype, str) or not mtype:
            raise ValueError("hub message needs a non-empty string 'type'")
        normalized = to_jsonable(msg)
        normalized.setdefault("t", None)
        text = json.dumps(normalized, allow_nan=False, separators=(",", ":"), ensure_ascii=False)
        return _Prepared(mtype, normalized, text)

    def _ingest(self, p: _Prepared, fanout: bool = True) -> None:
        with self._lock:
            stored = _Stored(next(self._seq), p.msg, p.text)
            if not self._store(p.mtype, stored):
                return
            self.published[p.mtype] += 1
            clients = list(self._clients.values()) if fanout else []
        for client in clients:
            client.offer(p.mtype, p.text)

    def _store(self, mtype: str, stored: _Stored) -> bool:
        """Record replay/REST state. Returns False for a stale event revision."""
        if mtype in self._latest_types:
            self._latest[mtype] = stored
        if mtype == "event":
            return self._store_event(stored)
        if mtype == "utterance":
            self._transcript.append(stored)
        elif mtype == "insight":
            ins = stored.msg.get("insight") or {}
            key = str(ins.get("insight_id") or f"_seq{stored.seq}")
            _put_capped(self._insights, key, stored, self.config.replay_history)
        elif mtype == "caption":
            self._captions.append(stored)
        return True

    def _store_event(self, stored: _Stored) -> bool:
        ev = stored.msg.get("event")
        if not isinstance(ev, Mapping) or not ev.get("event_id"):
            log.warning("event message without event.event_id; broadcast but not stored")
            return True
        eid = str(ev["event_id"])
        old = self._event_log.get(eid)
        if old is not None and _revision(old.msg) > _revision(stored.msg):
            log.debug("ignoring stale revision %s of event %s", _revision(stored.msg), eid)
            return False
        _put_capped(self._event_log, eid, stored, self.config.event_log_max)
        if ev.get("kind") in self._replay_exclude:
            self._event_replay.pop(eid, None)
        else:
            _put_capped(self._event_replay, eid, stored, self.config.replay_history)
        return True

    # -- subscribers --------------------------------------------------------------------------

    def subscribe(self) -> HubClient:
        """Register a client (call on the hub's loop). Its replay is queued atomically."""
        loop = asyncio.get_running_loop()
        self._loop = loop
        with self._lock:
            client = HubClient(
                next(self._client_ids),
                self.config.queue_size,
                self._min_interval,
                self._latest_types,
                loop,
                self._clock,
            )
            replay = self._replay_items()
            client.load_replay((s.text for _, s in replay), (t for t, _ in replay))
            self._clients[client.id] = client
        log.info("hub client %d subscribed (%d replay messages)", client.id, len(replay))
        return client

    def unsubscribe(self, client: HubClient) -> None:
        with self._lock:
            self._clients.pop(client.id, None)
        client.close()
        log.info("hub client %d unsubscribed (dropped %d)", client.id, client.dropped)

    def close_clients(self) -> None:
        """Close every client (server shutdown); their WebSocket handlers then exit."""
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            client.close()

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def _replay_items(self) -> list[tuple[str, _Stored]]:
        head = [(t, self._latest[t]) for t in ("meta", "status", "stats") if t in self._latest]
        head += [
            (t, s) for t, s in sorted(self._latest.items(), key=lambda kv: kv[1].seq)
            if t not in ("meta", "status", "stats", "frame")
        ]
        n = max(0, self.config.replay_history)
        history: list[tuple[str, _Stored]] = [("event", s) for s in self._event_replay.values()]
        history += [("utterance", s) for s in list(self._transcript)[-n:]] if n else []
        history += [("insight", s) for s in self._insights.values()]
        history += [("caption", s) for s in self._captions]
        history.sort(key=lambda ts: ts[1].seq)
        tail = [("frame", self._latest["frame"])] if "frame" in self._latest else []
        return head + history + tail

    def replay_messages(self) -> list[dict[str, Any]]:
        """What a client joining now would receive first (decoded)."""
        with self._lock:
            return [s.msg for _, s in self._replay_items()]

    # -- REST read helpers ----------------------------------------------------------------------

    def latest(self, mtype: str) -> dict[str, Any] | None:
        with self._lock:
            stored = self._latest.get(mtype)
            return stored.msg if stored else None

    def events(
        self, kinds: Iterable[str] | None = None, since: float | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Latest revision of each logged event (``Event.to_dict()`` form), sorted by t."""
        wanted = frozenset(kinds) if kinds else None
        with self._lock:
            evs = [s.msg["event"] for s in self._event_log.values()]
        if wanted is not None:
            evs = [e for e in evs if e.get("kind") in wanted]
        if since is not None:
            evs = [e for e in evs if _num(e.get("t")) >= since]
        evs.sort(key=lambda e: (_num(e.get("t")), str(e.get("event_id"))))
        return evs[-limit:] if limit else evs

    def transcript(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = [s.msg.get("utterance") for s in self._transcript]
        items = [u for u in items if u is not None]
        return items[-limit:] if limit else items

    def insights(self) -> list[dict[str, Any]]:
        with self._lock:
            return [s.msg.get("insight") for s in self._insights.values() if s.msg.get("insight")]

    def captions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [s.msg for s in self._captions]

    def info(self) -> dict[str, Any]:
        """Counters for health / state endpoints."""
        with self._lock:
            return {
                "clients": len(self._clients),
                "published": dict(self.published),
                "dropped": sum(c.dropped for c in self._clients.values()),
                "events_logged": len(self._event_log),
            }


def _put_capped(store: OrderedDict[str, _Stored], key: str, value: _Stored, cap: int) -> None:
    store[key] = value
    store.move_to_end(key)
    while len(store) > max(0, cap):
        store.popitem(last=False)


def _revision(msg: Mapping[str, Any]) -> int:
    ev = msg.get("event") or {}
    try:
        return int(ev.get("revision") or 0)
    except (TypeError, ValueError):
        return 0


def _num(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return -math.inf
    return f if math.isfinite(f) else -math.inf
