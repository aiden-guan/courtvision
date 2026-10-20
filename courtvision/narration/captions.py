"""Caption and audio-description cue emitter keyed to source video PTS (RESEARCH.md §7A.8, narration.md §11.5).

Timing rule. Every cue carries the source PTS of ``t_occ`` (``pts = pts_offset_s + t_occ``). It is
shown at ``pts + D`` (D = the stream's applied delay: ~0.8-1.0 s overlay delay, or a 30-60 s
caption / anti-wagering delay) and **never before the event was confirmed**: when the line is
ready only at video time ``t_confirm`` and ``t_confirm > t_occ + D``, the cue moves to
``pts(t_confirm)``. In media-timeline terms (WebVTT) the cue starts at ``start_pts = show_at - D``.

Sinks (one per ingest target):

* :class:`WebVTTWriter`: HLS WebVTT subtitle rendition with ``X-TIMESTAMP-MAP=LOCAL:..,MPEGTS:..``.
* :class:`SidecarSink`: callback with ``{"type": "caption", "t", "pts", "show_at", "text", "channel"}``
  (e.g. ``Hub.publish``) for our dashboard and the AD app.
* :class:`OBSCaptionSink`: obs-websocket v5 ``Identify`` (+ SHA-256 auth challenge) and
  ``SendStreamCaption`` (CEA-608 over the stream output). Sent on emit: OBS injects it into its live
  input, and any OBS stream delay then applies to video and captions alike.
* :class:`YouTubeCaptionSink`: HTTP POST to the event's Captions Ingestion URL with ``&seq=N`` and
  body ``"YYYY-MM-DDTHH:MM:SS.mmm\\ntext\\n"`` (UTC of the frame the cue belongs to).

This module performs network/file I/O by design (it is the caption transport); a failing sink is
logged and never blocks the other sinks.
"""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import hashlib
import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)

MPEGTS_HZ = 90_000  # MPEG-2 transport stream clock (RFC 8216 X-TIMESTAMP-MAP); sourced
MPEGTS_WRAP = 1 << 33  # 33-bit PTS rollover; sourced
CHANNELS = ("caption", "ad")


@dataclass
class CaptionConfig:
    """Cue timing. ``delay_s`` must equal the delay actually applied to the outgoing stream."""

    delay_s: float = 1.0  # D: overlay 0.8-1.0 s, caption/anti-wagering 30-60 s (RESEARCH §7A.8); sourced range, set per stream
    pts_offset_s: float = 0.0  # source PTS (seconds) at video t = 0; set from the demuxer; tuned per source
    words_per_s: float = 3.5  # cue duration from word count (narration.md §4.1 [M]); sourced
    min_cue_s: float = 1.0  # shortest on-screen cue; tuned
    max_cue_s: float = 7.0  # longest on-screen cue; tuned
    video_epoch_s: float | None = None  # Unix time of video t = 0 (live sources); None = anchor at the first cue


@dataclass
class CaptionCue:
    """One caption / AD cue. Times are source-PTS seconds unless named otherwise."""

    seq: int
    text: str
    channel: str  # "caption" | "ad"
    t_occ: float  # video time of the event
    pts: float  # source PTS of t_occ (the key)
    start_pts: float  # media time the cue may appear at (never before confirmation)
    end_pts: float
    show_at: float  # start_pts + delay: when the viewer of the delayed stream sees it
    utc: dt.datetime  # wall-clock UTC of the frame at start_pts (wall-clock sinks)
    event_ids: list[str] = field(default_factory=list)


@runtime_checkable
class CaptionSink(Protocol):
    channels: tuple[str, ...]

    async def send(self, cue: CaptionCue) -> None: ...

    async def aclose(self) -> None: ...


# --------------------------------------------------------------------------- time helpers


def format_vtt_time(seconds: float) -> str:
    """WebVTT timestamp ``HH:MM:SS.mmm`` (hours zero-padded to 2+ digits; negative clamps to 0)."""
    ms = int(round(max(0.0, float(seconds)) * 1000.0))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def mpegts_ticks(pts_s: float) -> int:
    """90 kHz MPEG-2 timestamp of ``pts_s`` (33-bit wrap)."""
    return int(round(float(pts_s) * MPEGTS_HZ)) % MPEGTS_WRAP


def format_youtube_time(t: dt.datetime) -> str:
    """``YYYY-MM-DDTHH:MM:SS.mmm`` in UTC (YouTube caption ingestion body)."""
    t = t.astimezone(dt.UTC)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}"


def cue_duration_s(text: str, cfg: CaptionConfig) -> float:
    words = len(text.split())
    return min(cfg.max_cue_s, max(cfg.min_cue_s, words / max(cfg.words_per_s, 1e-6)))


# --------------------------------------------------------------------------- emitter


class CaptionEmitter:
    """Builds :class:`CaptionCue` objects with the timing rule and fans them out to the sinks."""

    def __init__(self, sinks: Iterable[CaptionSink] = (), config: CaptionConfig | None = None,
                 wall_clock: Callable[[], float] = time.time) -> None:
        self.config = config or CaptionConfig()
        self.sinks: list[CaptionSink] = list(sinks)
        self._wall = wall_clock
        self._epoch = self.config.video_epoch_s
        self._seq = 0
        self.cues: list[CaptionCue] = []

    def add_sink(self, sink: CaptionSink) -> None:
        self.sinks.append(sink)

    def pts_of(self, t: float) -> float:
        return self.config.pts_offset_s + float(t)

    def make_cue(self, text: str, t_occ: float, *, channel: str = "caption", t_confirm: float | None = None,
                 duration_s: float | None = None, event_ids: Sequence[str] = ()) -> CaptionCue:
        """Apply the timing rule: ``show_at = max(pts(t_occ) + D, pts(t_confirm))``."""
        if channel not in CHANNELS:
            raise ValueError(f"caption channel must be one of {CHANNELS}, got {channel!r}")
        cfg = self.config
        pts = self.pts_of(t_occ)
        show_at = pts + cfg.delay_s
        if t_confirm is not None:
            show_at = max(show_at, self.pts_of(t_confirm))
        start = show_at - cfg.delay_s
        dur = duration_s if duration_s is not None else cue_duration_s(text, cfg)
        if self._epoch is None:  # anchor live wall time to the newest video time we know
            self._epoch = self._wall() - float(t_confirm if t_confirm is not None else t_occ)
        utc = dt.datetime.fromtimestamp(self._epoch + (start - cfg.pts_offset_s), tz=dt.UTC)
        self._seq += 1
        return CaptionCue(self._seq, text, channel, float(t_occ), pts, start, start + dur, show_at, utc,
                          list(event_ids))

    async def emit(self, text: str, t_occ: float, *, channel: str = "caption", t_confirm: float | None = None,
                   duration_s: float | None = None, event_ids: Sequence[str] = ()) -> CaptionCue | None:
        """Build one cue and send it to every sink that carries ``channel`` (failures are logged)."""
        if not text.strip():
            return None
        cue = self.make_cue(text, t_occ, channel=channel, t_confirm=t_confirm, duration_s=duration_s,
                            event_ids=event_ids)
        self.cues.append(cue)
        targets = [s for s in self.sinks if cue.channel in getattr(s, "channels", CHANNELS)]
        results = await asyncio.gather(*(s.send(cue) for s in targets), return_exceptions=True)
        for sink, res in zip(targets, results):
            if isinstance(res, BaseException):
                log.warning("caption sink %s failed: %r", type(sink).__name__, res)
        return cue

    async def aclose(self) -> None:
        for s in self.sinks:
            try:
                await s.aclose()
            except Exception:  # noqa: BLE001 - shutdown must not fail on one sink
                log.exception("caption sink %s failed to close", type(s).__name__)


# --------------------------------------------------------------------------- WebVTT (HLS)


class WebVTTWriter:
    """WebVTT subtitle rendition for HLS (``EXT-X-MEDIA TYPE=SUBTITLES``; RFC 8216 X-TIMESTAMP-MAP).

    The header maps cue time ``LOCAL`` (00:00:00.000) to ``MPEGTS`` = 90 kHz ticks of ``map_pts_s``,
    so a cue at local time ``x`` plays at source PTS ``map_pts_s + x``. With ``path`` set, the file is
    rewritten on every cue (small live files; segment with :meth:`render`).
    """

    def __init__(self, path: str | Path | None = None, *, map_pts_s: float = 0.0,
                 channels: tuple[str, ...] = ("caption",)) -> None:
        self.path = Path(path) if path is not None else None
        self.map_pts_s = float(map_pts_s)
        self.channels = channels
        self.cues: list[CaptionCue] = []
        self._write_lock = asyncio.Lock()  # concurrent emits must not let an older document land last

    def header(self) -> str:
        return f"WEBVTT\nX-TIMESTAMP-MAP=LOCAL:{format_vtt_time(0.0)},MPEGTS:{mpegts_ticks(self.map_pts_s)}\n"

    def cue_block(self, cue: CaptionCue) -> str:
        start = cue.start_pts - self.map_pts_s
        end = max(start + 0.001, cue.end_pts - self.map_pts_s)
        text = "\n".join(line.strip() for line in cue.text.replace("-->", "->").splitlines() if line.strip())
        return f"{cue.seq}\n{format_vtt_time(start)} --> {format_vtt_time(end)}\n{text}\n"

    def render(self, start_pts: float | None = None, end_pts: float | None = None) -> str:
        """The document (optionally only cues overlapping ``[start_pts, end_pts)``, e.g. one segment)."""
        blocks = [self.cue_block(c) for c in self.cues
                  if (start_pts is None or c.end_pts > start_pts) and (end_pts is None or c.start_pts < end_pts)]
        return self.header() + "".join("\n" + b for b in blocks)

    async def send(self, cue: CaptionCue) -> None:
        self.cues.append(cue)
        if self.path is not None:
            async with self._write_lock:  # render inside the lock: the last write holds every cue
                await asyncio.to_thread(self.path.write_text, self.render(), "utf-8")

    async def aclose(self) -> None:
        return None


# --------------------------------------------------------------------------- sidecar WebSocket JSON


def caption_message(cue: CaptionCue) -> dict[str, Any]:
    """Sidecar message (ARCHITECTURE WebSocket schema) plus ``show_at`` and ``seq``."""
    return {"type": "caption", "t": cue.t_occ, "pts": cue.pts, "show_at": cue.show_at, "text": cue.text,
            "channel": cue.channel, "seq": cue.seq}


class SidecarSink:
    """Hands each cue to ``callback`` (sync or async), e.g. ``hub.publish``."""

    def __init__(self, callback: Callable[[dict[str, Any]], Awaitable[None] | None],
                 channels: tuple[str, ...] = CHANNELS) -> None:
        self._cb = callback
        self.channels = channels

    async def send(self, cue: CaptionCue) -> None:
        res = self._cb(caption_message(cue))
        if inspect.isawaitable(res):
            await res

    async def aclose(self) -> None:
        return None


# --------------------------------------------------------------------------- OBS (obs-websocket v5)


def obs_auth(password: str, salt: str, challenge: str) -> str:
    """obs-websocket v5 auth string: base64(sha256(base64(sha256(password + salt)) + challenge))."""
    secret = base64.b64encode(hashlib.sha256((password + salt).encode()).digest()).decode()
    return base64.b64encode(hashlib.sha256((secret + challenge).encode()).digest()).decode()


class _WS(Protocol):
    async def send(self, message: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


async def _default_connect(url: str) -> _WS:
    from websockets.asyncio.client import connect  # lazy: only the OBS sink needs it

    return await connect(url, open_timeout=5.0)


class OBSCaptionSink:
    """``SendStreamCaption`` over obs-websocket v5 (added in v5.0.0; RESEARCH §7A.8 [469]).

    Op codes: Hello 0, Identify 1, Identified 2, Request 6, RequestResponse 7. The connection is
    opened lazily and re-opened after a failure.
    """

    def __init__(self, url: str = "ws://localhost:4455", password: str | None = None, *,
                 connect: Callable[[str], Awaitable[_WS]] | None = None, channels: tuple[str, ...] = ("caption",),
                 timeout_s: float = 5.0) -> None:
        self.url = url
        self._password = password
        self._connect = connect or _default_connect
        self.channels = channels
        self.timeout_s = timeout_s
        self._ws: _WS | None = None
        self._lock = asyncio.Lock()
        self._req = 0
        self.sent = 0
        self.failed = 0

    async def _recv_json(self, ws: _WS) -> dict[str, Any]:
        raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout_s)
        return json.loads(raw.decode() if isinstance(raw, bytes) else raw)

    async def _open(self) -> _WS:
        ws = await self._connect(self.url)
        hello = await self._recv_json(ws)
        if hello.get("op") != 0:
            raise ConnectionError(f"OBS: expected Hello (op 0), got {hello.get('op')}")
        d = hello.get("d") or {}
        identify: dict[str, Any] = {"rpcVersion": int(d.get("rpcVersion", 1)), "eventSubscriptions": 0}
        auth = d.get("authentication")
        if auth:
            if self._password is None:
                raise ConnectionError("OBS requires a password (authentication challenge received)")
            identify["authentication"] = obs_auth(self._password, auth["salt"], auth["challenge"])
        await ws.send(json.dumps({"op": 1, "d": identify}))
        ident = await self._recv_json(ws)
        if ident.get("op") != 2:
            raise ConnectionError(f"OBS: Identify failed (op {ident.get('op')})")
        return ws

    async def send(self, cue: CaptionCue) -> None:
        async with self._lock:
            try:
                if self._ws is None:
                    self._ws = await self._open()
                self._req += 1
                rid = f"cv-caption-{self._req}"
                req = {"op": 6, "d": {"requestType": "SendStreamCaption", "requestId": rid,
                                      "requestData": {"captionText": cue.text}}}
                await self._ws.send(json.dumps(req))
                while True:
                    msg = await self._recv_json(self._ws)
                    if msg.get("op") == 7 and (msg.get("d") or {}).get("requestId") == rid:
                        break
                status = msg["d"].get("requestStatus") or {}
                if not status.get("result"):
                    self.failed += 1
                    log.warning("OBS SendStreamCaption failed: code %s %s", status.get("code"), status.get("comment"))
                    return
                self.sent += 1
            except Exception:
                self.failed += 1
                await self._drop()
                raise

    async def _drop(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001 - already broken
                pass

    async def aclose(self) -> None:
        await self._drop()


# --------------------------------------------------------------------------- YouTube Live


class YouTubeCaptionSink:
    """HTTP POST captions to a YouTube Live Captions Ingestion URL (RESEARCH §7A.8 [470]).

    One caption track; YouTube advises a 30 s or 60 s broadcast delay for caption software, so set
    ``CaptionConfig.delay_s`` accordingly. ``seq`` increments per POST.
    """

    def __init__(self, ingestion_url: str, *, client: Any = None, timeout_s: float = 5.0, start_seq: int = 1,
                 channels: tuple[str, ...] = ("caption",)) -> None:
        self.ingestion_url = ingestion_url
        self._client = client
        self._own_client = client is None
        self.timeout_s = timeout_s
        self.seq = int(start_seq)
        self.channels = channels
        self._lock = asyncio.Lock()
        self.sent = 0

    def _http(self) -> Any:
        if self._client is None:
            import httpx2  # lazy: only this sink needs it

            self._client = httpx2.AsyncClient(timeout=self.timeout_s)
        return self._client

    def url_for(self, seq: int) -> str:
        sep = "&" if "?" in self.ingestion_url else "?"
        return f"{self.ingestion_url}{sep}seq={seq}"

    @staticmethod
    def body(cue: CaptionCue) -> str:
        return f"{format_youtube_time(cue.utc)}\n{cue.text}\n"

    async def send(self, cue: CaptionCue) -> None:
        async with self._lock:
            seq = self.seq
            self.seq += 1
            resp = await self._http().post(self.url_for(seq), content=self.body(cue).encode("utf-8"),
                                           headers={"Content-Type": "text/plain; charset=utf-8"})
            if resp.status_code >= 400:
                raise RuntimeError(f"YouTube caption POST seq={seq} failed: HTTP {resp.status_code}")
            self.sent += 1

    async def aclose(self) -> None:
        if self._own_client and self._client is not None:
            await self._client.aclose()
            self._client = None
