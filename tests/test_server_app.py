"""FastAPI app: REST endpoints (200/400/401/403/413/503), WebSocket replay + live, auth, static."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from courtvision.server.app import Controller, ServerConfig, _pump, create_app
from courtvision.server.hub import Hub, HubConfig
from courtvision.types import Event, EventKind, Team, Utterance

STATIC = Path(__file__).resolve().parents[1] / "courtvision" / "server" / "static"
TOKEN = "s3cret-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
SQUARE = [[0, 0], [100, 0], [100, 100], [0, 100]]


class FakeController:
    """Records every call; returns JSON-hostile values (numpy) on purpose."""

    def __init__(self, jpeg: bytes | None = b"\xff\xd8\xff\xe0fake-jpeg") -> None:
        self.calls: list[tuple[str, Any]] = []
        self.jpeg = jpeg

    def add_manual(self, event: dict) -> dict:
        self.calls.append(("add_manual", event))
        return {"event_id": event.get("event_id") or "m000001"}

    def set_scoreboard(self, reading: dict) -> dict:
        self.calls.append(("set_scoreboard", reading))
        return reading

    def calibrate(self, request: dict) -> dict:
        self.calls.append(("calibrate", request))
        return {"rmse_ft": np.float64(0.42), "inliers": np.ones(len(request["image_points"]), dtype=bool)}

    def control(self, command: dict) -> dict:
        self.calls.append(("control", command))
        return {"applied": command["action"]}

    def latest_frame_jpeg(self) -> bytes | None:
        self.calls.append(("latest_frame_jpeg", None))
        return self.jpeg

    def set_lineup(self, entry: dict) -> dict:
        self.calls.append(("set_lineup", entry))
        return entry

    def set_scoreboard_roi(self, roi: dict) -> dict:
        self.calls.append(("set_scoreboard_roi", roi))
        return {"cells": len(roi["cells"])}


META = {
    "type": "meta", "t": 0.0,
    "teams": {"home": "Hawks", "away": "Owls"},
    "roster": [
        {"player_id": "home-1", "team": "home", "jersey": "1", "name": "A"},
        {"player_id": "home-2", "team": "home", "jersey": "2", "name": "B"},
        {"player_id": "away-3", "team": "away", "jersey": "3", "name": "C"},
    ],
    "court": {"length": 94.0, "width": 50.0},
    "landmarks": {"center": [47, 25], "lb": [0, 0], "lt": [0, 50], "rb": [94, 0], "rt": [94, 50]},
}


@pytest.fixture
def hub() -> Hub:
    return Hub()


@pytest.fixture
def ctrl() -> FakeController:
    return FakeController()


@pytest.fixture
def client(hub: Hub, ctrl: FakeController):
    with TestClient(create_app(hub, ctrl)) as c:
        yield c


VALID_POSTS = [
    ("/api/events", {"kind": "shot_made", "t": 12.5, "team": "home", "player_id": "home-1", "value": 3}, "add_manual"),
    ("/api/events", {"event_id": "e000007", "kind": "rebound", "t": 3.0, "status": "retracted"}, "add_manual"),
    ("/api/scoreboard", {"home_score": 10, "away_score": 8, "game_clock": 431.5}, "set_scoreboard"),
    ("/api/lineup", {"team": "home", "out": "home-1", "in": "home-2", "t": 100.0}, "set_lineup"),
    ("/api/lineup", {"team": "home", "players": ["home-1", "home-2"]}, "set_lineup"),
    ("/api/scoreboard_roi", {"corners": SQUARE, "cells": {"home_ones": [10, 10, 20, 30]}}, "set_scoreboard_roi"),
    ("/api/calibration", {"image_points": SQUARE, "court_points": [[0, 0], [94, 0], [94, 50], [0, 50]]}, "calibrate"),
    ("/api/calibration", {"image_points": SQUARE, "landmarks": ["lb", "rb", "rt", "lt"]}, "calibrate"),
    ("/api/control", {"action": "pause"}, "control"),
    ("/api/control", {"action": "speed", "value": 2.0}, "control"),
    ("/api/control", {"action": "mute"}, "control"),
]

INVALID_POSTS = [
    ("/api/events", {"kind": "not_a_kind", "t": 1.0}),
    ("/api/events", {"kind": "shot_made", "t": 1.0, "player_id": "home-1", "value": 0}),  # made shot needs 1-3
    ("/api/events", {"kind": "assist", "t": 1.0}),  # needs player_id
    ("/api/events", {"kind": "steal", "t": 1.0, "player_id": "p1", "other_player_id": "p1"}),
    ("/api/events", {"kind": "turnover", "t": -1.0}),
    ("/api/events", {"kind": "turnover", "t": 1.0, "status": "provisional"}),
    ("/api/events", {"kind": "turnover", "t": 1.0, "status": "retracted"}),  # retraction needs an id
    ("/api/events", {"kind": "turnover", "t": 1.0, "player_id": "bad id with spaces"}),
    ("/api/events", {"kind": "turnover", "t": 1.0, "source": "cv"}),  # extra fields are refused
    ("/api/scoreboard", {}),
    ("/api/scoreboard", {"home_score": -1}),
    ("/api/lineup", {"team": "home", "out": "home-1"}),
    ("/api/lineup", {"team": "home", "out": "home-1", "in": "home-1"}),
    ("/api/lineup", {"team": "home", "players": ["a", "b", "c", "d", "e", "f"]}),
    ("/api/lineup", {"team": "home", "players": ["a", "a"]}),
    ("/api/lineup", {"team": "visitors", "players": ["a"]}),
    ("/api/scoreboard_roi", {"corners": SQUARE[:3], "cells": {"a": [0, 0, 1, 1]}}),
    ("/api/scoreboard_roi", {"corners": [[0, 0]] * 4, "cells": {"a": [0, 0, 1, 1]}}),
    ("/api/scoreboard_roi", {"corners": SQUARE, "cells": {"a": [5, 5, 1, 1]}}),
    ("/api/scoreboard_roi", {"corners": SQUARE, "cells": {}}),
    ("/api/calibration", {"image_points": SQUARE[:3], "court_points": SQUARE[:3]}),
    ("/api/calibration", {"image_points": SQUARE}),
    ("/api/calibration", {"image_points": SQUARE, "court_points": SQUARE, "landmarks": ["a", "b", "c", "d"]}),
    ("/api/calibration", {"image_points": SQUARE, "court_points": SQUARE[:3] + [[1, 2], [3, 4]]}),
    ("/api/control", {"action": "rewind"}),
    ("/api/control", {"action": "speed"}),
    ("/api/control", {"action": "speed", "value": 0}),
    ("/api/control", {"action": "speed", "value": 100}),
]


# --------------------------------------------------------------------------- basics / static


def test_controller_protocol_is_runtime_checkable(ctrl: FakeController):
    assert isinstance(ctrl, Controller)


def test_healthz_is_open(hub: Hub, ctrl: FakeController):
    with TestClient(create_app(hub, ctrl, auth_token=TOKEN)) as c:
        r = c.get("/healthz")
    assert r.status_code == 200 and r.json()["ok"] is True and r.json()["controller"] is True


def test_dashboard_index_and_assets_are_served(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "CourtVision" in r.text and 'src="static/app.js?v=' in r.text
    csp = r.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp
    assert r.headers["x-content-type-options"] == "nosniff"
    js = client.get("/static/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    css = client.get("/static/styles.css")
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")


@pytest.mark.parametrize("path", [
    "/static/../app.py", "/static/%2e%2e/app.py", "/static/..%2fhub.py", "/static/%2e%2e%2f%2e%2e/types.py",
    "/static//etc/passwd", "/static/....//app.py",
])
def test_static_path_traversal_is_refused(client: TestClient, path: str):
    r = client.get(path)
    assert r.status_code == 404
    assert "def create_app" not in r.text and "class Hub" not in r.text


def test_dashboard_is_self_contained_offline():
    for name in ("index.html", "app.js", "styles.css"):
        text = (STATIC / name).read_text()
        urls = re.findall(r"https?://[^\s'\"<>)]+", text)
        assert all("www.w3.org/2000/svg" in u for u in urls), (name, urls)  # SVG namespace only
        assert "@import" not in text


def test_every_dom_id_used_by_the_script_exists():
    html = (STATIC / "index.html").read_text()
    js = (STATIC / "app.js").read_text()
    ids = set(re.findall(r'id="([^"]+)"', html))
    used = set(re.findall(r'byId\("([^"]+)"\)', js))
    for template in re.findall(r"byId\(`([^`]+)`\)", js):
        pattern = "^" + re.sub(r"\\\$\\\{[^}]+\\\}", "[a-z]+", re.escape(template)) + "$"
        assert any(re.match(pattern, i) for i in ids), template
    assert used and used <= ids, used - ids
    assert 'aria-live="polite"' in html  # transcript is announced politely


# --------------------------------------------------------------------------- reads


def test_get_endpoints_read_hub_state_and_encode_nan_as_null(hub: Hub, client: TestClient):
    hub.publish_threadsafe(META)
    hub.publish_threadsafe({"type": "frame", "t": 5.0, "frame_idx": 150, "players": []})
    hub.publish_threadsafe({"type": "stats", "t": 5.0, "stats": {"score": {"home": 2}, "efg": float("nan"),
                                                                  "arr": np.arange(3)}})
    hub.publish_threadsafe(Event("e000001", EventKind.SHOT_ATTEMPT, 4.0, 120, team=Team.HOME, value=2))
    hub.publish_threadsafe(Event("e000002", EventKind.SHOT_MADE, 4.8, 144, team=Team.HOME, value=2))
    hub.publish_threadsafe(Utterance("u0001", "Hawks score.", "play_by_play", 4.8))
    hub.publish_threadsafe({"type": "insight", "t": 5.0, "insight": {"insight_id": "i1", "text": "Run"}})

    state = client.get("/api/state").json()
    assert state["t"] == 5.0 and state["meta"]["teams"]["home"] == "Hawks" and state["frame"]["frame_idx"] == 150
    box = client.get("/api/boxscore")
    assert box.status_code == 200 and "NaN" not in box.text
    assert box.json() == {"t": 5.0, "stats": {"score": {"home": 2}, "efg": None, "arr": [0, 1, 2]}}
    events = client.get("/api/events").json()["events"]
    assert [e["event_id"] for e in events] == ["e000001", "e000002"]
    made = client.get("/api/events", params={"kind": "shot_made"}).json()["events"]
    assert [e["kind"] for e in made] == ["shot_made"]
    assert len(client.get("/api/events", params={"limit": 1}).json()["events"]) == 1
    assert client.get("/api/events", params={"kind": "bogus"}).status_code == 400
    assert client.get("/api/transcript").json()["utterances"][0]["text"] == "Hawks score."
    assert client.get("/api/insights").json()["insights"][0]["text"] == "Run"


def test_empty_hub_reads_are_null_not_errors(client: TestClient):
    assert client.get("/api/boxscore").json() == {"t": None, "stats": None}
    assert client.get("/api/state").json()["frame"] is None
    assert client.get("/api/events").json() == {"events": []}


def test_frame_jpeg(hub: Hub):
    with TestClient(create_app(hub, FakeController())) as c:
        r = c.get("/api/frame.jpg")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert r.content.startswith(b"\xff\xd8") and r.headers["cache-control"] == "no-store"
    with TestClient(create_app(hub, FakeController(jpeg=None))) as c:
        assert c.get("/api/frame.jpg").status_code == 404


# --------------------------------------------------------------------------- writes


@pytest.mark.parametrize(("path", "body", "method"), VALID_POSTS)
def test_post_endpoints_forward_to_controller(client: TestClient, ctrl: FakeController, path: str, body: dict, method: str):
    r = client.post(path, json=body)
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True
    name, payload = ctrl.calls[-1]
    assert name == method
    if path == "/api/events":
        assert payload["source"] == "manual" and payload["kind"] == body["kind"]
        assert payload["status"] == body.get("status", "confirmed")
        assert set(payload) >= {"event_id", "kind", "t", "frame_idx", "team", "player_id", "value", "attrs"}
        Event.from_dict({k: v for k, v in payload.items()})  # controller can rebuild an Event
    if path == "/api/scoreboard":
        assert payload == {**body, "source": "manual"}
    if path == "/api/lineup":
        assert payload == body


def test_calibration_result_is_json_safe(client: TestClient):
    r = client.post("/api/calibration", json={"image_points": SQUARE, "court_points": SQUARE})
    assert r.json()["result"] == {"rmse_ft": 0.42, "inliers": [True, True, True, True]}


@pytest.mark.parametrize(("path", "body"), INVALID_POSTS)
def test_invalid_bodies_get_400(client: TestClient, ctrl: FakeController, path: str, body: dict):
    r = client.post(path, json=body)
    assert r.status_code == 400, (path, body, r.text)
    assert r.json()["detail"]
    assert not ctrl.calls


def test_malformed_and_non_finite_json_get_400(client: TestClient):
    headers = {"content-type": "application/json"}
    assert client.post("/api/control", content=b"{not json", headers=headers).status_code == 400
    r = client.post("/api/control", content=b'{"action": "speed", "value": NaN}', headers=headers)
    assert r.status_code == 400
    r = client.post("/api/calibration", content=json.dumps(
        {"image_points": [[0, 0], [1, 0], [1, 1], [0, 1]], "court_points": SQUARE}).replace("1]", "Infinity]", 1),
        headers=headers)
    assert r.status_code == 400


def test_roster_checks_when_meta_is_known(hub: Hub, client: TestClient, ctrl: FakeController):
    hub.publish_threadsafe(META)
    assert client.post("/api/lineup", json={"team": "home", "out": "home-1", "in": "away-3"}).status_code == 400
    assert client.post("/api/lineup", json={"team": "home", "players": ["home-1", "ghost-9"]}).status_code == 400
    assert client.post("/api/events", json={"kind": "rebound", "t": 1.0, "team": "home", "player_id": "away-3"}).status_code == 400
    ok = client.post("/api/events", json={"kind": "rebound", "t": 1.0, "team": "home", "player_id": "home-t12"})
    assert ok.status_code == 200  # provisional track ids are allowed in events
    bad = client.post("/api/calibration", json={"image_points": SQUARE, "landmarks": ["lb", "rb", "rt", "nowhere"]})
    assert bad.status_code == 400 and "nowhere" in bad.json()["detail"]
    assert [c[0] for c in ctrl.calls] == ["add_manual"]


@pytest.mark.parametrize(("path", "body", "_m"), VALID_POSTS[::2])
def test_writes_without_controller_get_503(hub: Hub, path: str, body: dict, _m: str):
    with TestClient(create_app(hub)) as c:
        assert c.post(path, json=body).status_code == 503
        assert c.get("/api/frame.jpg").status_code == 503
        assert c.get("/api/state").status_code == 200  # reads still work from the hub


def test_controller_errors_map_to_http(hub: Hub):
    class Failing(FakeController):
        def set_scoreboard(self, reading: dict) -> dict:
            raise ValueError("period out of range for this ruleset")

        def control(self, command: dict) -> dict:
            raise RuntimeError("boom")

    with TestClient(create_app(hub, Failing()), raise_server_exceptions=False) as c:
        r = c.post("/api/scoreboard", json={"period": 9})
        assert r.status_code == 400 and "ruleset" in r.json()["detail"]
        assert c.post("/api/control", json={"action": "pause"}).status_code == 500


def test_async_controller_methods_are_awaited(hub: Hub):
    class AsyncCtrl(FakeController):
        async def control(self, command: dict) -> dict:
            await asyncio.sleep(0)
            return {"async": True}

    with TestClient(create_app(hub, AsyncCtrl())) as c:
        assert c.post("/api/control", json={"action": "resume"}).json()["result"] == {"async": True}


def test_cross_origin_writes_and_large_bodies_are_refused(client: TestClient, ctrl: FakeController):
    r = client.post("/api/control", json={"action": "pause"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    same = client.post("/api/control", json={"action": "pause"}, headers={"Origin": "http://testserver"})
    assert same.status_code == 200
    big = client.post("/api/events", content=b"{" + b" " * 70_000 + b"}", headers={"content-type": "application/json"})
    assert big.status_code == 413
    assert [c[0] for c in ctrl.calls] == ["control"]


# --------------------------------------------------------------------------- auth


def test_auth_token_enforced_on_writes_and_reads(hub: Hub, ctrl: FakeController):
    with TestClient(create_app(hub, ctrl, auth_token=TOKEN)) as c:
        body = {"action": "pause"}
        r = c.post("/api/control", json=body)
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
        assert c.post("/api/control", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.post("/api/control", json=body, headers={"Authorization": TOKEN}).status_code == 401
        assert c.post("/api/control", json=body, headers=AUTH).status_code == 200
        assert c.get("/api/events").status_code == 401
        assert c.get("/api/frame.jpg").status_code == 401
        assert c.get("/api/events", headers=AUTH).status_code == 200
        assert c.get("/").status_code == 200 and c.get("/static/app.js").status_code == 200
    assert [x[0] for x in ctrl.calls] == ["control"]


def test_reads_can_be_left_open(hub: Hub, ctrl: FakeController):
    app = create_app(hub, ctrl, auth_token=TOKEN, config=ServerConfig(protect_reads=False))
    with TestClient(app) as c:
        assert c.get("/api/boxscore").status_code == 200
        assert c.post("/api/control", json={"action": "pause"}).status_code == 401


def test_websocket_requires_token_query_param(hub: Hub, ctrl: FakeController):
    with TestClient(create_app(hub, ctrl, auth_token=TOKEN)) as c:
        for url in ("/ws", "/ws?token=wrong"):
            with pytest.raises(WebSocketDisconnect) as exc:
                with c.websocket_connect(url) as ws:
                    ws.receive_text()
            assert exc.value.code == 1008
        hub.publish_threadsafe({"type": "status", "t": 1.0, "paused": False})
        with c.websocket_connect(f"/ws?token={TOKEN}") as ws:
            assert ws.receive_json()["type"] == "status"


def test_websocket_refuses_foreign_origin(client: TestClient):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws", headers={"Origin": "https://evil.example"}) as ws:
            ws.receive_text()
    assert exc.value.code == 1008


# --------------------------------------------------------------------------- websocket stream


def test_websocket_receives_replay_then_live_messages(hub: Hub, client: TestClient):
    hub.publish_threadsafe(META)
    hub.publish_threadsafe({"type": "stats", "t": 1.0, "stats": {"score": {"home": 0, "away": 0}}})
    hub.publish_threadsafe(Event("e000001", EventKind.POSSESSION, 0.5, 15, team=Team.AWAY))
    hub.publish_threadsafe({"type": "frame", "t": 1.0, "frame_idx": 30, "players": []})
    with client.websocket_connect("/ws") as ws:
        replay = [ws.receive_json() for _ in range(4)]
        assert [m["type"] for m in replay] == ["meta", "stats", "event", "frame"]
        hub.publish_threadsafe(Event("e000002", EventKind.SHOT_ATTEMPT, 2.0, 60, team=Team.AWAY, value=np.int64(3)))
        live = ws.receive_json()
        assert live["type"] == "event" and live["event"]["value"] == 3
        hub.publish_threadsafe(Event("e000002", EventKind.SHOT_ATTEMPT, 2.0, 60, team=Team.AWAY, value=2, revision=1))
        assert ws.receive_json()["event"]["revision"] == 1
        hub.publish_threadsafe(Utterance("u0001", "Owls for three.", "play_by_play", 2.0, latency_ms=float("nan")))
        utt = ws.receive_json()
        assert utt["utterance"]["text"] == "Owls for three." and utt["utterance"]["latency_ms"] is None
    assert hub.client_count == 0


def test_websocket_ping_gets_pong(client: TestClient):
    with client.websocket_connect("/ws") as ws:
        ws.send_text("not json")  # ignored
        ws.send_json({"type": "ping", "id": 7})
        pong = ws.receive_json()
        assert pong["type"] == "pong" and pong["id"] == 7 and isinstance(pong["server_time"], float)


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed_with: int | None = None

    async def send_text(self, text: str) -> None:
        self.sent.append(text)

    async def close(self, code: int = 1000) -> None:
        self.closed_with = code


async def test_overflowed_client_is_closed_for_resync():
    hub = Hub(HubConfig(queue_size=2, max_rate_hz={}))
    client = hub.subscribe()
    for i in range(4):
        await hub.publish(Utterance(f"u{i}", "x", "color", float(i)))
    assert client.overflowed
    sock = _FakeSocket()
    async with asyncio.timeout(1):
        await _pump(sock, client, hub)  # type: ignore[arg-type]
    assert sock.closed_with == 1013 and len(sock.sent) == 1


async def test_pump_exits_when_hub_closes():
    hub = Hub()
    client = hub.subscribe()
    sock = _FakeSocket()
    task = asyncio.create_task(_pump(sock, client, hub))  # type: ignore[arg-type]
    await asyncio.sleep(0)
    hub.close_clients()
    async with asyncio.timeout(1):
        await task
    assert sock.closed_with == 1001


def test_websocket_ignores_binary_frames(client: TestClient):
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(b"\x00\x01")
        ws.send_json({"type": "ping", "id": "a"})
        assert ws.receive_json()["id"] == "a"
