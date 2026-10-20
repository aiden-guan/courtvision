"""Regression tests for defects found in the server review (one test group per defect)."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from courtvision.server import app as app_mod
from courtvision.server.app import ServerConfig, create_app
from courtvision.server.hub import Hub
from courtvision.types import Event, EventKind
from tests.test_server_app import META, SQUARE, FakeController

STATIC = Path(__file__).resolve().parents[1] / "courtvision" / "server" / "static"


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


def _post_event(client: TestClient, **body):
    return client.post("/api/events", json={"t": 5.0, **body})


# --------------------------------------------------------------------------- Event.value per kind


@pytest.mark.parametrize("body", [
    {"kind": "score_correction", "team": "home", "value": -2},  # take back a phantom basket
    {"kind": "score_correction", "team": "away", "value": 5},
    {"kind": "period_start", "value": 4},  # stats reads the period number from value
    {"kind": "period_end", "value": 5},
    {"kind": "shot_made", "player_id": "home-1", "value": 1, "attrs": {"shot_type": "free_throw"}},
    {"kind": "shot_attempt", "player_id": "home-1", "value": 3, "attrs": {"shot_type": "jumper"}},
])
def test_kind_specific_values_are_accepted(client: TestClient, ctrl: FakeController, body: dict):
    r = _post_event(client, **body)
    assert r.status_code == 200, r.text
    payload = ctrl.calls[-1][1]
    assert payload["value"] == body["value"]
    assert Event.from_dict(payload).value == body["value"]


@pytest.mark.parametrize("body", [
    {"kind": "score_correction", "team": "home", "value": 0},
    {"kind": "period_start", "value": 25},
    {"kind": "rebound", "team": "home", "value": 5},
    {"kind": "rebound", "team": "home", "value": -1},
    {"kind": "shot_made", "player_id": "home-1", "value": 2, "attrs": {"shot_type": "free_throw"}},
    {"kind": "shot_attempt", "player_id": "home-1", "value": 1, "attrs": {"shot_type": "layup"}},
])
def test_inconsistent_values_get_400(client: TestClient, ctrl: FakeController, body: dict):
    assert _post_event(client, **body).status_code == 400
    assert not ctrl.calls


# --------------------------------------------------------------------------- team semantics


@pytest.mark.parametrize("kind", ["rebound", "possession", "turnover", "score_correction"])
def test_events_without_any_team_get_400(client: TestClient, ctrl: FakeController, kind: str):
    body = {"kind": kind, "value": 2} if kind == "score_correction" else {"kind": kind}
    r = _post_event(client, **body)
    assert r.status_code == 400 and "team" in r.json()["detail"]
    assert not ctrl.calls


def test_timeout_and_period_events_may_be_teamless(client: TestClient):
    assert _post_event(client, kind="timeout").status_code == 200
    assert _post_event(client, kind="period_end").status_code == 200


def test_team_is_inferred_from_roster_and_id_prefix(hub: Hub, client: TestClient, ctrl: FakeController):
    assert _post_event(client, kind="rebound", player_id="away-t9").status_code == 200
    assert ctrl.calls[-1][1]["team"] == "away"  # provisional id prefix
    hub.publish_threadsafe({**META, "roster": [{"player_id": "p23", "team": "home", "jersey": "23", "name": "A"}]})
    assert _post_event(client, kind="steal", player_id="p23", other_player_id="away-t4").status_code == 200
    assert ctrl.calls[-1][1]["team"] == "home"  # roster lookup


@pytest.mark.parametrize("body", [
    {"kind": "assist", "player_id": "home-1", "other_player_id": "away-3"},  # scorer must be a teammate
    {"kind": "substitution", "team": "home", "player_id": "home-1", "other_player_id": "away-3"},
    {"kind": "steal", "player_id": "home-1", "other_player_id": "home-2"},  # victim must be an opponent
    {"kind": "block", "player_id": "away-3", "other_player_id": "away-t7"},
    {"kind": "foul", "player_id": "home-1", "other_player_id": "home-2"},
    {"kind": "rebound", "team": "home", "player_id": "away-t9"},  # id prefix contradicts team
])
def test_cross_team_relations_are_checked_without_a_roster(client: TestClient, ctrl: FakeController, body: dict):
    assert _post_event(client, **body).status_code == 400
    assert not ctrl.calls


def test_intercepted_pass_may_name_an_opponent(client: TestClient):
    r = _post_event(client, kind="pass", player_id="home-1", other_player_id="away-3", attrs={"completed": False})
    assert r.status_code == 200


def test_lineup_prefix_mismatch_without_roster(client: TestClient, ctrl: FakeController):
    r = client.post("/api/lineup", json={"team": "home", "out": "away-3", "in": "home-2", "t": 10.0})
    assert r.status_code == 400 and not ctrl.calls


# --------------------------------------------------------------------------- scoreboard ROI geometry


@pytest.mark.parametrize("corners", [
    [[0, 0], [0, 100], [100, 100], [100, 0]],  # counter-clockwise: rectified image would be mirrored
    [[100, 0], [100, 100], [0, 100], [0, 0]],  # starts at TR: rectified image would be rotated
    [[0, 0], [100, 0], [40, 40], [0, 100]],  # concave
])
def test_roi_corner_order_and_convexity(client: TestClient, ctrl: FakeController, corners: list):
    r = client.post("/api/scoreboard_roi", json={"corners": corners, "cells": {"a": [10, 10, 20, 20]}})
    assert r.status_code == 400 and not ctrl.calls


def test_roi_cells_and_image_bounds(client: TestClient, ctrl: FakeController):
    base = {"corners": SQUARE, "cells": {"a": [10, 10, 20, 20]}}
    assert client.post("/api/scoreboard_roi", json={**base, "cells": {"a": [500, 500, 520, 520]}}).status_code == 400
    assert client.post("/api/scoreboard_roi", json={**base, "image_size": [0, 720]}).status_code == 400
    assert client.post("/api/scoreboard_roi", json={**base, "image_size": [50, 50]}).status_code == 400
    assert not ctrl.calls
    skewed = [[10, 5], [110, 15], [105, 60], [8, 50]]  # a perspective quad is fine
    r = client.post("/api/scoreboard_roi", json={"corners": skewed, "cells": {"a": [40, 20, 50, 40]},
                                                 "image_size": [1280, 720]})
    assert r.status_code == 200, r.text


# --------------------------------------------------------------------------- calibration image size


def test_calibration_forwards_image_size_and_checks_bounds(client: TestClient, ctrl: FakeController):
    body = {"image_points": SQUARE, "court_points": [[0, 0], [94, 0], [94, 50], [0, 50]]}
    assert client.post("/api/calibration", json={**body, "image_size": [1280, 720]}).status_code == 200
    assert ctrl.calls[-1][1]["image_size"] == [1280, 720]
    assert client.post("/api/calibration", json={**body, "image_size": [50, 50]}).status_code == 400
    assert client.post("/api/calibration", json={**body, "image_size": [0, 720]}).status_code == 400


# --------------------------------------------------------------------------- DNS rebinding (Host)


@pytest.mark.parametrize("host", ["localhost:8000", "127.0.0.1:8000", "[::1]:8000", "192.168.1.20",
                                  "courtvision-mac", "courtvision-mac.local", "gym.lan:8080"])
def test_local_hosts_are_accepted(hub: Hub, ctrl: FakeController, host: str):
    with TestClient(create_app(hub, ctrl), base_url=f"http://{host}") as c:
        assert c.get("/api/state").status_code == 200
        with c.websocket_connect(f"ws://{host}/ws") as ws:  # websocket_connect ignores base_url
            ws.send_json({"type": "ping", "id": 1})
            assert ws.receive_json()["type"] == "pong"


def test_public_host_names_are_refused(hub: Hub, ctrl: FakeController):
    with TestClient(create_app(hub, ctrl), base_url="http://rebind.attacker.example") as c:
        assert c.get("/api/state").status_code == 400
        assert c.post("/api/control", json={"action": "pause"},
                      headers={"Origin": "http://rebind.attacker.example"}).status_code == 400
        with pytest.raises(WebSocketDisconnect) as exc:
            with c.websocket_connect("ws://rebind.attacker.example/ws") as ws:
                ws.receive_text()
        assert exc.value.code == 1008
    assert not ctrl.calls


def test_allowed_hosts_extends_and_star_disables(hub: Hub, ctrl: FakeController):
    cfg = ServerConfig(allowed_hosts=["*.school.example"])
    with TestClient(create_app(hub, ctrl, config=cfg), base_url="https://cv.school.example") as c:
        assert c.get("/api/state").status_code == 200
    with TestClient(create_app(hub, ctrl, config=ServerConfig(allowed_hosts=["*"])),
                    base_url="http://anything.example.com") as c:
        assert c.get("/api/state").status_code == 200


# --------------------------------------------------------------------------- body cap, traversal, jpeg


def test_chunked_body_over_the_cap_gets_413(client: TestClient, ctrl: FakeController):
    def chunks():
        for _ in range(100):
            yield b" " * 1000  # no Content-Length: capped while streaming

    r = client.post("/api/events", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 413 and not ctrl.calls


async def _raw_get(app, path: str) -> tuple[int, bytes]:
    """Send ``path`` to the ASGI app untouched (httpx would normalise ``..`` away)."""
    sent: list[dict] = []
    first = True

    async def receive() -> dict:
        nonlocal first
        if first:
            first = False
            return {"type": "http.request", "body": b"", "more_body": False}
        await asyncio.Event().wait()  # never disconnect while the response is produced
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
             "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
             "headers": [(b"host", b"testserver")], "server": ("testserver", 80),
             "client": ("127.0.0.1", 5000), "root_path": ""}
    await asyncio.wait_for(app(scope, receive, send), 5)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return status, b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")


@pytest.mark.parametrize("path", ["/static/../app.py", "/static/../../types.py", "/static/../server/hub.py"])
async def test_raw_dotdot_paths_never_reach_files_outside_static(path: str):
    status, body = await _raw_get(create_app(Hub()), path)
    assert status == 404
    assert b"def create_app" not in body and b"class Hub" not in body and b"class Team" not in body


async def test_raw_static_path_still_serves_assets():
    status, body = await _raw_get(create_app(Hub()), "/static/app.js")
    assert status == 200 and body.startswith(b"/* CourtVision")


def test_frame_jpeg_accepts_an_imencode_ndarray(hub: Hub):
    buf = np.frombuffer(b"\xff\xd8\xff\xe0ndarray-jpeg", dtype=np.uint8)  # what cv2.imencode returns
    with TestClient(create_app(hub, FakeController(jpeg=buf))) as c:  # type: ignore[arg-type]
        r = c.get("/api/frame.jpg")
        assert r.status_code == 200 and r.content == buf.tobytes()
    empty = np.zeros(0, dtype=np.uint8)
    with TestClient(create_app(hub, FakeController(jpeg=empty))) as c:  # type: ignore[arg-type]
        assert c.get("/api/frame.jpg").status_code == 404


# --------------------------------------------------------------------------- hub: stopped loop


def test_publish_threadsafe_on_a_stopped_loop_is_recorded_not_lost():
    loop = asyncio.new_event_loop()
    try:
        hub = Hub()
        hub.bind_loop(loop)  # bound but not running (e.g. before run_forever / after stop)
        hub.publish_threadsafe({"type": "meta", "t": 0.0, "roster": []})
        hub.publish_threadsafe(Event("e000001", EventKind.POSSESSION, 1.0, 30))
        assert [m["type"] for m in hub.replay_messages()] == ["meta", "event"]
    finally:
        loop.close()


async def test_async_publish_rebinds_from_a_stopped_loop():
    stopped = asyncio.new_event_loop()
    try:
        hub = Hub()
        hub.bind_loop(stopped)
        await hub.publish({"type": "status", "t": 1.0, "paused": True})
        assert hub.loop is asyncio.get_running_loop()
        client = hub.subscribe()
        hub.publish_threadsafe({"type": "status", "t": 2.0, "paused": False})
        async with asyncio.timeout(1):
            assert '"paused":true' in (await client.get() or "")
            assert '"paused":false' in (await client.get() or "")
    finally:
        stopped.close()


# --------------------------------------------------------------------------- dashboard mirrors server rules


def _js_set(js: str, name: str) -> set[str]:
    m = re.search(rf"const {name} = new Set\(\[([^\]]*)\]\)", js)
    assert m, name
    return set(re.findall(r'"([a-z_]+)"', m.group(1)))


def test_dashboard_event_rules_match_the_server():
    js = (STATIC / "app.js").read_text()
    assert _js_set(js, "TEAMLESS_OK") == {k.value for k in app_mod._TEAMLESS_OK}
    assert _js_set(js, "SAME_TEAM_PAIR") == {k.value for k in app_mod._SAME_TEAM_PAIR}
    assert _js_set(js, "OPPONENT_PAIR") == {k.value for k in app_mod._OPPONENT_PAIR}
    assert _js_set(js, "NEEDS_PLAYER") == {k.value for k in app_mod._NEEDS_ACTOR}
    assert _js_set(js, "FG_SHOT_TYPES") == set(app_mod._FG_SHOT_TYPES)
    html = (STATIC / "index.html").read_text()
    value_input = re.search(r'<input[^>]*id="ev-value"[^>]*>', html)
    assert value_input and 'min="-' in value_input.group(0)  # signed score corrections are enterable
    assert "image_size" in js  # calibration and ROI posts carry the clicked frame's size
    # The top bar holds the last scoreboard reading (OCR reaches ~1 frame in 6) instead of
    # reading clk.shot straight off each frame, which flickered to "--" between reads.
    assert 'heldClock("shot")' in js and "fmtShot(clk.shot)" not in js
