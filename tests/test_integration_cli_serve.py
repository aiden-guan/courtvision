"""``courtvision serve`` smoke tests: the dashboard app (server.app.create_app) with a real-time LiveGame
attached to its lifespan, driven through Starlette's TestClient WebSocket; bearer tokens; a game
that fails mid-run; the uvicorn wiring (loopback, ephemeral port); the CLI's refusal to expose an
unauthenticated server and token redaction in logs. No network beyond 127.0.0.1."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from collections import Counter

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import courtvision.cli as cli
from courtvision.cli import RedactTokens, build_uvicorn_server, create_serve_app, main
from courtvision.config import load_config
from courtvision.pipeline import FrameItem, FrameSource, SimSource
from courtvision.sim.simulator import GameSimulator

SEED = 7
NO_KEYS: dict[str, str] = {}
TOKEN = "cv-test-token-6f1d2a"


def _cfg(*extra: str):
    return load_config(None, ["sim.seconds=20", f"sim.seed={SEED}", "narration.tts=none", "pipeline.tail_s=1",
                              "jev.mode=off", "hub.queue_size=100000", "hub.disconnect_on_overflow=false", *extra])


@pytest.fixture(scope="module")
def result():
    return GameSimulator(_cfg().sim).run()


def _source(result, speed: float = 10.0) -> SimSource:
    return SimSource(result, "truth3d", realtime=True, speed=speed)


def _collect(ws, done, limit: int = 50_000) -> tuple[Counter, list[dict]]:
    """Receive until ``done(counts)`` or the game reports it finished (never blocks past the game)."""
    counts: Counter = Counter()
    msgs: list[dict] = []
    for _ in range(limit):
        m = ws.receive_json()
        counts[m["type"]] += 1
        msgs.append(m)
        if done(counts) or (m["type"] == "status" and m.get("finished")):
            break
    return counts, msgs


def test_serve_smoke_meta_frames_events_utterances_then_clean_shutdown(result) -> None:
    srv = create_serve_app(_cfg(), source=_source(result), env=NO_KEYS)
    game = srv.game
    assert game.realtime and game.llm is None and game.jev.client.mode == "off"  # no keys: templates + fallbacks
    t0 = time.perf_counter()
    with TestClient(srv.app) as client:
        task = srv.app.state.game_task
        assert not task.done()
        with client.websocket_connect("/ws") as ws:
            counts, msgs = _collect(ws, lambda c: c["meta"] and c["frame"] >= 10 and c["event"] and c["utterance"])
            r = client.get("/api/state")
            assert r.status_code == 200
        assert counts["meta"] == 1 and msgs[0]["type"] == "meta", counts
        assert counts["frame"] >= 10 and counts["event"] >= 1 and counts["utterance"] >= 1, counts
        meta = msgs[0]
        assert len(meta["roster"]) == len(result.roster.entries) and meta["teams"]["home"]
        frames = [m for m in msgs if m["type"] == "frame"]
        assert frames[-1]["t"] > frames[0]["t"]  # the game is running in real time on the server loop
        assert all(len(f["players"]) == 10 for f in frames[-3:])
        utt = next(m for m in msgs if m["type"] == "utterance")["utterance"]
        assert utt["text"] and utt["source"] == "template"
        ev = next(m for m in msgs if m["type"] == "event")["event"]
        assert ev["event_id"] and ev["kind"]
        assert game._loop is srv.hub.loop  # LiveGame and the hub share the server's event loop
        assert not game._finished
    # leaving the client runs the lifespan shutdown: the game stops without its narration tail
    assert task.done() and not task.cancelled() and task.exception() is None
    assert game._finished and game._errors == 0 and game.final_stats is not None
    assert srv.hub.client_count == 0
    assert game._frames_done < len(result.frames)  # stopped early, not played to the end
    assert time.perf_counter() - t0 < 15.0


def test_serve_plays_to_the_end_and_keeps_serving(result) -> None:
    srv = create_serve_app(_cfg("sim.seconds=20"), source=SimSource(result, "truth3d", realtime=True, speed=16.0,
                                                                   duration_s=4.0), env=NO_KEYS)
    with TestClient(srv.app) as client:
        with client.websocket_connect("/ws") as ws:
            counts, msgs = _collect(ws, lambda c: False)  # until the finished status
        assert msgs[-1]["type"] == "status" and msgs[-1]["finished"] is True
        assert srv.game._frames_done == 120
        # the finished game's state is still served
        assert client.get("/api/boxscore").status_code == 200
        assert client.get("/api/events").json()["events"]
        with client.websocket_connect("/ws") as late:  # a late joiner gets the replay
            first = late.receive_json()
            assert first["type"] == "meta"


def test_serve_controls_and_bearer_token(result) -> None:
    srv = create_serve_app(_cfg(), source=_source(result, speed=4.0), env=NO_KEYS, auth_token=TOKEN)
    with TestClient(srv.app) as client:
        assert client.get("/api/state").status_code == 401
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws") as ws:
                ws.receive_json()
        auth = {"Authorization": f"Bearer {TOKEN}"}
        r = client.post("/api/control", json={"action": "pause"}, headers=auth)
        assert r.status_code == 200 and r.json()["result"]["paused"] is True
        n = srv.game._frames_done
        time.sleep(0.3)
        assert srv.game._frames_done - n <= 1  # nothing advances while paused
        r = client.post("/api/control", json={"action": "resume"}, headers=auth)
        assert r.json()["result"]["paused"] is False
        with client.websocket_connect(f"/ws?token={TOKEN}") as ws:
            counts, _ = _collect(ws, lambda c: c["frame"] >= 3)
            assert counts["frame"] >= 3
    assert srv.game._finished  # shutdown also stops a game that was paused and resumed


class _FailingSource(FrameSource):
    """Yields a few sim frames, then the camera 'disconnects' with an exception."""

    def __init__(self, result) -> None:
        super().__init__(realtime=True, speed=1.0)
        self.result = result
        self.roster, self.court, self.rules = result.roster, result.court, result.rules

    async def _iterate(self):
        for f in self.result.frames[:5]:
            yield FrameItem(f.frame_idx, f.t, None, f)
        raise RuntimeError("camera disconnected")


def test_a_failing_game_does_not_take_the_dashboard_down(result, caplog) -> None:
    caplog.set_level(logging.ERROR, logger="courtvision.cli")
    srv = create_serve_app(_cfg(), source=_FailingSource(result), env=NO_KEYS)
    with TestClient(srv.app) as client:
        task = srv.app.state.game_task
        for _ in range(200):
            if task.done():
                break
            time.sleep(0.01)
        assert task.done() and task.exception() is None  # logged, not raised into the server
        assert client.get("/api/state").status_code == 200
        assert client.get("/healthz").status_code == 200
    assert srv.game._finished
    assert "the game stopped with an error" in caplog.text


def _loopback_ok() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _loopback_ok(), reason="no loopback sockets in this sandbox")
def test_uvicorn_runs_the_game_on_its_own_loop(result, monkeypatch) -> None:
    websockets = pytest.importorskip("websockets")
    srv = create_serve_app(_cfg(), source=_source(result), env=NO_KEYS, auth_token=TOKEN)
    server = build_uvicorn_server(srv.app, "127.0.0.1", 0, "warning")
    opened: list[str] = []
    import webbrowser

    monkeypatch.setattr(webbrowser, "open", lambda url, *a, **k: opened.append(url) or True)

    async def scenario() -> tuple[bool, set[str], bool]:
        task = asyncio.create_task(cli._serve(server, f"http://127.0.0.1/#token={TOKEN}"))
        for _ in range(500):
            if server.started:
                break
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        same_loop = srv.game._loop is asyncio.get_running_loop()
        types: set[str] = set()
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws?token={TOKEN}") as ws:
            while not {"meta", "frame", "event"} <= types:
                types.add(json.loads(await asyncio.wait_for(ws.recv(), 10))["type"])
        server.should_exit = True
        return await asyncio.wait_for(task, 20), types, same_loop

    started, types, same_loop = asyncio.run(scenario())
    assert started and same_loop and {"meta", "frame", "event"} <= types
    assert opened == [f"http://127.0.0.1/#token={TOKEN}"]  # --open: after startup, token only in the hash
    assert srv.game._finished


def test_cli_serve_refuses_an_open_server_without_a_token(monkeypatch, capsys) -> None:
    monkeypatch.delenv("COURTVISION_TOKEN", raising=False)
    called: list[object] = []
    monkeypatch.setattr(cli, "_serve", lambda *a: called.append(a))
    assert main(["serve", "--host", "0.0.0.0", "--seconds", "5", "--tts", "none"]) == 2
    assert "without an access token" in capsys.readouterr().err and called == []


def test_cli_serve_wiring_never_prints_the_token(monkeypatch, capsys, tmp_path) -> None:
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("COURTVISION_TOKEN", TOKEN)
    seen: dict = {}

    async def fake_serve(server, open_url):
        seen["config"] = server.config
        seen["open_url"] = open_url
        return True

    monkeypatch.setattr(cli, "_serve", fake_serve)
    code = main(["serve", "--host", "0.0.0.0", "--port", "9123", "--seconds", "5", "--tts", "none", "--no-llm",
                 "--jev", "off", "--speed", "3", "--open"])
    out = capsys.readouterr()
    assert code == 0, out.err
    assert "CourtVision dashboard: http://127.0.0.1:9123/" in out.out and "bearer token required" in out.out
    assert TOKEN not in out.out and TOKEN not in out.err
    config = seen["config"]
    assert (config.host, config.port, config.lifespan) == ("0.0.0.0", 9123, "on")
    assert seen["open_url"] == f"http://127.0.0.1:9123/#token={TOKEN}"
    game = config.app.state.game
    assert game.source.speed == 3.0 and game.llm is None and game.jev.client.mode == "off"


def test_redact_tokens_filter() -> None:
    record = logging.LogRecord("uvicorn.error", logging.INFO, __file__, 1, '%s - "WebSocket %s" [accepted]',
                               ("127.0.0.1:5", f"/ws?token={TOKEN}&x=1"), None)
    assert RedactTokens().filter(record)
    assert record.getMessage() == '127.0.0.1:5 - "WebSocket /ws?token=REDACTED&x=1" [accepted]'
    plain = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, "GET %s", ("/api/state",), None)
    assert RedactTokens().filter(plain) and plain.getMessage() == "GET /api/state"
