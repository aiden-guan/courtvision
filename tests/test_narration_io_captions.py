"""Caption emitter: timing rule, WebVTT formatting, sidecar, OBS v5 and YouTube transports (no network)."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
from typing import Any

import httpx2
import pytest

from courtvision.narration.captions import (CaptionConfig, CaptionEmitter, OBSCaptionSink, SidecarSink,
                                            WebVTTWriter, YouTubeCaptionSink, format_vtt_time, format_youtube_time,
                                            mpegts_ticks, obs_auth)


@pytest.mark.parametrize("secs, text", [
    (0.0, "00:00:00.000"), (1.5, "00:00:01.500"), (61.25, "00:01:01.250"), (3661.2344, "01:01:01.234"),
    (59.9996, "00:01:00.000"), (-3.0, "00:00:00.000"), (360000.0, "100:00:00.000"),
])
def test_vtt_time_format(secs: float, text: str) -> None:
    assert format_vtt_time(secs) == text


def test_mpegts_ticks_wrap() -> None:
    assert mpegts_ticks(10.0) == 900_000
    assert mpegts_ticks((1 << 33) / 90_000 + 1.0) == 90_000


def test_timing_rule_never_before_confirmation() -> None:
    em = CaptionEmitter(config=CaptionConfig(delay_s=1.0, pts_offset_s=10.0), wall_clock=lambda: 1000.0)
    on_time = em.make_cue("Riverside ball.", 100.0, t_confirm=100.4)
    assert on_time.pts == pytest.approx(110.0) and on_time.show_at == pytest.approx(111.0)
    assert on_time.start_pts == pytest.approx(110.0)  # media time of the event itself
    late = em.make_cue("Good.", 100.0, t_confirm=103.0)  # confirmed 3 s after the event, D = 1 s
    assert late.show_at == pytest.approx(113.0) and late.start_pts == pytest.approx(112.0)
    assert late.show_at >= em.pts_of(103.0)
    long_delay = CaptionEmitter(config=CaptionConfig(delay_s=30.0))  # YouTube-style caption delay
    cue = long_delay.make_cue("Good.", 100.0, t_confirm=103.0)
    assert cue.show_at == pytest.approx(130.0) and cue.start_pts == pytest.approx(100.0)
    assert cue.end_pts - cue.start_pts >= 1.0
    with pytest.raises(ValueError):
        em.make_cue("x", 1.0, channel="subtitles")


async def test_webvtt_writer_document(tmp_path: Any) -> None:
    path = tmp_path / "captions.vtt"
    vtt = WebVTTWriter(path, map_pts_s=10.0)
    em = CaptionEmitter([vtt], CaptionConfig(delay_s=1.0, pts_offset_s=10.0, words_per_s=2.0), wall_clock=lambda: 0.0)
    await em.emit("Riley Thorsen buries the three!", 3661.25, t_confirm=3661.5)
    await em.emit("Riverside 42, Lakeport 40.", 3665.0)
    doc = path.read_text()
    assert doc == vtt.render()
    lines = doc.splitlines()
    assert lines[0] == "WEBVTT"
    assert lines[1] == "X-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:900000"
    assert lines[2] == "" and lines[3] == "1"
    assert lines[4] == "01:01:01.250 --> 01:01:03.750"  # 5 words at 2 words/s
    assert lines[5] == "Riley Thorsen buries the three!"
    assert "01:01:05.000 --> 01:01:07.000" in doc
    segment = vtt.render(start_pts=3674.0, end_pts=3680.0)  # source PTS window of one HLS segment
    assert "Riverside 42" in segment and "buries" not in segment


async def test_sidecar_sync_and_async_callbacks() -> None:
    got: list[dict] = []

    async def publish(msg: dict) -> None:
        got.append(msg)

    em = CaptionEmitter([SidecarSink(got.append), SidecarSink(publish, channels=("ad",))],
                        CaptionConfig(delay_s=0.8))
    await em.emit("Riverside ball.", 50.0, channel="ad")
    await em.emit("Good from deep.", 51.0)
    assert [m["channel"] for m in got] == ["ad", "ad", "caption"]
    m = got[0]
    assert m["type"] == "caption" and m["t"] == 50.0 and m["pts"] == 50.0 and m["show_at"] == pytest.approx(50.8)
    assert m["text"] == "Riverside ball."


async def test_failing_sink_does_not_block_others() -> None:
    class Broken:
        channels = ("caption",)

        async def send(self, cue: Any) -> None:
            raise ConnectionError("down")

        async def aclose(self) -> None:
            return None

    got: list[dict] = []
    em = CaptionEmitter([Broken(), SidecarSink(got.append)])
    assert await em.emit("Riverside ball.", 1.0) is not None
    assert len(got) == 1


class FakeWS:
    """obs-websocket v5 server side, scripted."""

    def __init__(self, password: str | None, result: bool = True) -> None:
        self.password, self.result = password, result
        self.salt, self.challenge = "c2FsdHlzYWx0", "Y2hhbGxlbmdl"
        self.sent: list[dict] = []
        self._out: list[str] = [json.dumps({"op": 0, "d": {"obsWebSocketVersion": "5.5.0", "rpcVersion": 1,
                                                           **({"authentication": {"salt": self.salt,
                                                                                  "challenge": self.challenge}}
                                                              if password else {})}})]
        self.closed = False

    async def send(self, message: str) -> None:
        msg = json.loads(message)
        self.sent.append(msg)
        if msg["op"] == 1:
            self._out.append(json.dumps({"op": 2, "d": {"negotiatedRpcVersion": 1}}))
        elif msg["op"] == 6:
            self._out.append(json.dumps({"op": 5, "d": {"eventType": "Noise"}}))  # unrelated event first
            self._out.append(json.dumps({"op": 7, "d": {"requestType": msg["d"]["requestType"],
                                                        "requestId": msg["d"]["requestId"],
                                                        "requestStatus": {"result": self.result,
                                                                          "code": 100 if self.result else 501}}}))

    async def recv(self) -> str:
        return self._out.pop(0)

    async def close(self) -> None:
        self.closed = True


def _expected_auth(password: str, salt: str, challenge: str) -> str:
    secret = base64.b64encode(hashlib.sha256(f"{password}{salt}".encode()).digest())
    return base64.b64encode(hashlib.sha256(secret + challenge.encode()).digest()).decode()


async def test_obs_identify_auth_and_send_caption() -> None:
    ws = FakeWS("hunter22")
    urls: list[str] = []

    async def connect(url: str) -> FakeWS:
        urls.append(url)
        return ws

    sink = OBSCaptionSink("ws://127.0.0.1:4455", "hunter22", connect=connect)
    em = CaptionEmitter([sink])
    await em.emit("Riley Thorsen buries the three!", 10.0)
    await em.emit("Riverside 45, Lakeport 40.", 12.0)
    identify = ws.sent[0]
    assert identify["op"] == 1 and identify["d"]["rpcVersion"] == 1
    assert identify["d"]["authentication"] == _expected_auth("hunter22", ws.salt, ws.challenge)
    assert identify["d"]["authentication"] == obs_auth("hunter22", ws.salt, ws.challenge)
    reqs = [m for m in ws.sent if m["op"] == 6]
    assert [r["d"]["requestType"] for r in reqs] == ["SendStreamCaption"] * 2
    assert reqs[0]["d"]["requestData"] == {"captionText": "Riley Thorsen buries the three!"}
    assert urls == ["ws://127.0.0.1:4455"] and sink.sent == 2  # one connection, reused


async def test_obs_without_auth_and_failed_request() -> None:
    ws = FakeWS(None, result=False)

    async def connect(url: str) -> FakeWS:
        return ws

    sink = OBSCaptionSink(connect=connect)
    await CaptionEmitter([sink]).emit("Riverside ball.", 1.0)
    assert "authentication" not in ws.sent[0]["d"] and sink.failed == 1 and sink.sent == 0


async def test_obs_password_required() -> None:
    async def connect(url: str) -> FakeWS:
        return FakeWS("secret")

    sink = OBSCaptionSink(connect=connect)
    with pytest.raises(ConnectionError):
        await sink.send(CaptionEmitter().make_cue("x.", 1.0))


async def test_youtube_post_seq_and_body() -> None:
    seen: list[httpx2.Request] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200)

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    url = "http://upload.youtube.com/closedcaption?cid=abc-123"
    sink = YouTubeCaptionSink(url, client=client)
    epoch = dt.datetime(2026, 10, 20, 19, 30, 0, tzinfo=dt.UTC).timestamp()
    em = CaptionEmitter([sink], CaptionConfig(delay_s=30.0, video_epoch_s=epoch))
    await em.emit("Riverside ball.", 12.345)
    await em.emit("Good from deep.", 20.0)
    assert [str(r.url) for r in seen] == [url + "&seq=1", url + "&seq=2"]
    assert seen[0].content.decode() == "2026-10-20T19:30:12.345\nRiverside ball.\n"
    assert seen[0].headers["content-type"].startswith("text/plain")
    assert sink.sent == 2
    await client.aclose()


async def test_youtube_http_error_is_reported() -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(403)

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    sink = YouTubeCaptionSink("http://upload.youtube.com/closedcaption?cid=x", client=client)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        await sink.send(CaptionEmitter().make_cue("x.", 1.0))
    assert sink.seq == 2  # a failed POST still consumes its sequence number
    await client.aclose()


def test_youtube_time_format() -> None:
    t = dt.datetime(2026, 1, 2, 3, 4, 5, 678_900, tzinfo=dt.UTC)
    assert format_youtube_time(t) == "2026-01-02T03:04:05.678"
