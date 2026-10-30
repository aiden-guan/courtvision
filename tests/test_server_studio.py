"""Exercise the video workflow with actual decoding and a deterministic, injected detector."""
from __future__ import annotations

import asyncio
import json
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from courtvision.config import load_config
from courtvision.server import studio
from courtvision.server.studio import create_studio_app
from courtvision.types import BBox, Detection, Event, EventKind, FrameDetections, ObjClass, Team


class PeopleDetector:
    def detect(self, image, frame_idx, t):
        h, w = image.shape[:2]
        return FrameDetections(frame_idx, t, (w, h), [
            Detection(BBox(w*.2, h*.15, w*.4, h*.9), ObjClass.PLAYER, .95),
            Detection(BBox(w*.5, h*.4, w*.53, h*.45), ObjClass.BALL, .9),
        ])


@pytest.fixture
def video_bytes(tmp_path):
    # Random pixels keep this over the ordinary JSON body limit, exercising streamed upload.
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30., (160, 96))
    rng = np.random.default_rng(7)
    for _ in range(60):
        writer.write(rng.integers(0, 255, (96, 160, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = load_config(None, ["narration.llm_enabled=false", "narration.tts=none", "jev.mode=off"])
    return create_studio_app(cfg, root=tmp_path / "workspace", auth_token="local-test", detector=PeopleDetector(),
                             lab_runs=tmp_path / "runs")


AUTH = {"Authorization": "Bearer local-test"}


def wait_for(client, predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        state = client.get("/api/studio", headers=AUTH).json()
        if predicate(state):
            return state
        time.sleep(.025)
    pytest.fail(f"Workspace did not reach expected state: {state}")


def test_upload_track_pause_export_and_reopen_without_cross_game_data(app, video_bytes):
    with TestClient(app) as client:
        response = client.post("/api/studio/upload?name=Game.avi", content=video_bytes, headers=AUTH)
        assert response.status_code == 200, response.text
        video = response.json()
        assert video["width"] == 160 and video["duration_s"] == pytest.approx(2)
        started = client.post("/api/studio/start", headers=AUTH,
                              json={"video_id": video["id"], "broadcast": False})
        assert started.status_code == 200, started.text
        state = wait_for(client, lambda s: s["vision"].get("people", 0) > 0 and s["status"]["frames"] > 0)
        assert state["vision"]["ball_visible"] and not state["vision"]["calibrated"]
        assert state["status"]["frames"] > 0 and state["status"]["needs_calibration"]
        # Pixel tracking must never manufacture basketball events or a floor map.
        assert client.get("/api/events", headers=AUTH).json()["events"] == []
        assert client.post("/api/control", json={"action":"pause"}, headers=AUTH).status_code == 200
        snapshot = client.get("/api/studio/frame.jpg?overlay=true", headers=AUTH)
        assert snapshot.status_code == 200 and snapshot.headers["content-type"] == "image/jpeg"
        assert cv2.imdecode(np.frombuffer(snapshot.content, np.uint8), cv2.IMREAD_COLOR) is not None
        exported = client.get("/api/studio/export", headers=AUTH)
        assert exported.status_code == 200 and "attachment" in exported.headers["content-disposition"]
        assert exported.json()["events"] == [] and exported.json()["video"]["id"] == video["id"]
        assert client.post("/api/scoreboard", json={"home_score":42,"away_score":37}, headers=AUTH).status_code == 200
        assert client.get("/api/boxscore", headers=AUTH).json()["stats"]["score"]["home"] == 42
        second = client.post("/api/studio/start", headers=AUTH,
                             json={"video_id":video["id"], "home_name":"Home two", "broadcast":False})
        assert second.status_code == 200
        wait_for(client, lambda s: bool(s["vision"]))
        # Reopening resets REST, replay, and the entire event engine, never carrying a prior score.
        assert client.get("/api/boxscore", headers=AUTH).json()["stats"] is None
        assert client.get("/api/events", headers=AUTH).json()["events"] == []
        assert (app.state.workspace.root / video["id"] / "source.video").is_file()
    assert list((app.state.workspace.root / "analyses").glob("*/events.jsonl"))


def test_upload_guards_auth_origin_and_json_size(app, video_bytes):
    with TestClient(app) as client:
        assert len(video_bytes) > 65536
        assert client.post("/api/studio/upload", content=video_bytes).status_code == 401
        assert client.get("/api/studio").status_code == 401
        assert client.get("/api/studio/frame.jpg").status_code == 401
        assert client.get("/api/studio/export").status_code == 401
        assert client.post("/api/studio/prepare").status_code == 401
        assert client.post("/api/studio/start", json={"demo":True}).status_code == 401
        foreign = {**AUTH, "Origin":"https://attacker.example"}
        assert client.post("/api/studio/upload", content=video_bytes, headers=foreign).status_code == 403
        assert client.post("/api/control", content=video_bytes, headers=AUTH).status_code == 413
        valid = client.post("/api/studio/upload?name=../../outside.avi", content=video_bytes, headers=AUTH)
        assert valid.status_code == 200 and valid.json()["name"] == "outside.avi"
        assert not (app.state.workspace.root.parent / "outside.avi").exists()


def test_bad_upload_and_missing_video_do_not_start_analysis(app):
    with TestClient(app) as client:
        assert client.post("/api/studio/upload", content=b"not a video", headers=AUTH).status_code == 400
        assert client.get("/api/studio", headers=AUTH).json()["videos"] == []
        for body, expected in [({},400), ({"video_id":"0"*32},404), ({"video_id":"../secrets"},400)]:
            assert client.post("/api/studio/start", json=body, headers=AUTH).status_code == expected
        assert app.state.workspace.task is None


def test_calibration_persists_and_reopens_on_same_footage(app, video_bytes):
    with TestClient(app) as client:
        video = client.post("/api/studio/upload", content=video_bytes, headers=AUTH).json()
        assert client.post("/api/studio/start", json={"video_id":video["id"], "broadcast":False}, headers=AUTH).status_code == 200
        wait_for(client, lambda s: s["status"]["needs_calibration"])
        client.post("/api/control", json={"action":"pause"}, headers=AUTH)
        body = {"image_points":[[0,0],[160,0],[160,96],[0,96]],
                "court_points":[[0,50],[94,50],[94,0],[0,0]], "image_size":[160,96]}
        calibration = client.post("/api/calibration", json=body, headers=AUTH)
        assert calibration.status_code == 200, calibration.text
        path = app.state.workspace.root / video["id"] / "calibration.json"
        assert path.is_file()
        before = json.loads(path.read_text())
        assert client.post("/api/studio/start", json={"video_id":video["id"], "broadcast":False}, headers=AUTH).status_code == 200
        state = wait_for(client, lambda s: s["vision"].get("calibrated"))
        assert not state["status"]["needs_calibration"] and json.loads(path.read_text()) == before
        state = wait_for(client, lambda s: bool(s["frame"]))
        assert state["frame"]["players"]  # reconnecting clients can restore the final tracking summary


def test_detector_prepare_is_idempotent_and_does_not_download_when_ready(app, monkeypatch):
    async def forbidden():
        raise AssertionError("An injected ready detector must not load or download a model")
    monkeypatch.setattr(app.state.workspace, "prepare", forbidden)
    with TestClient(app) as client:
        assert client.post("/api/studio/prepare", headers=AUTH).json() == {"state":"ready"}
        assert app.state.workspace.model_task is None


def test_hub_reset_removes_pending_and_replay_from_previous_game():
    from courtvision.server.hub import Hub
    async def check():
        hub = Hub()
        hub.bind_loop()
        client = hub.subscribe()
        await hub.publish({"type":"stats", "t":1, "stats":{"score":{"home":99}}})
        await hub.publish({"type":"caption", "t":1, "text":"Old game"})
        await hub.reset()
        assert hub.latest("stats") is None and hub.captions() == []
        assert all('99' not in text for _, text in client._queue)
        hub.close_clients()
    asyncio.run(check())


def test_monitor_focus_uses_same_frame_and_never_changes_clean_footage(app):
    from courtvision.server.studio import StudioGame
    game = StudioGame.__new__(StudioGame)
    image = np.full((180, 320, 3), 40, np.uint8)
    game.visual = (image, [
        {"box":(40, 30, 100, 150), "label":"Track 1", "team":None, "confidence":.94},
        {"box":(180, 25, 245, 155), "label":"Track 2", "team":None, "confidence":.81},
    ], 1.25)
    game._jpeg_cache = {}
    app.state.workspace.game = game
    with TestClient(app) as client:
        clean = client.get("/api/studio/frame.jpg?overlay=false", headers=AUTH)
        all_tracks = client.get("/api/studio/frame.jpg", headers=AUTH)
        focused = client.get("/api/studio/frame.jpg?focus=Track%201", headers=AUTH)
        clean_focused = client.get("/api/studio/frame.jpg?overlay=false&focus=Track%201", headers=AUTH)
        assert clean.content == clean_focused.content
        assert all_tracks.content != focused.content != clean.content
        assert focused.headers["x-video-time"] == "1.25"
        # The backing source image is never drawn on.
        assert np.all(image == 40)
    app.state.workspace.game = None


def test_studio_reads_saved_key_for_next_session_without_exposing_it(app, tmp_path, monkeypatch):
    from courtvision.credentials import save_jev_key
    monkeypatch.chdir(tmp_path)
    with TestClient(app) as client:
        assert not client.get("/api/studio", headers=AUTH).json()["capabilities"]["jev"]
        save_jev_key("test-only-saved-key", tmp_path / ".env")
        state = client.get("/api/studio", headers=AUTH)
        assert state.json()["capabilities"]["jev"]
        assert "test-only-saved-key" not in state.text
        assert client.post("/api/studio/start", json={"demo":True}, headers=AUTH).status_code == 200
        assert app.state.workspace.game.env["TYPESAFE_API_KEY"] == "test-only-saved-key"


def test_an_analysed_video_serves_its_possession_timeline_and_shots(app, video_bytes, monkeypatch):
    monkeypatch.setattr(studio, "ANALYSIS_RECHECK_S", 0.0)
    with TestClient(app) as client:
        video = client.post("/api/studio/upload", content=video_bytes, headers=AUTH).json()
        assert client.post("/api/studio/start", json={"video_id": video["id"], "home_name": "Warriors",
                                                      "broadcast": False}, headers=AUTH).status_code == 200
        assert wait_for(client, lambda s: s["status"] is not None)["analysis"] is None
        assert client.get("/api/studio/analysis", headers=AUTH).status_code == 404
        # tools/nba_analyze.py <game> timeline, on this video
        ws = app.state.workspace
        run = ws.nba_root / "2025-01-01-AAA-BBB" / "runs" / "full"
        run.mkdir(parents=True)
        (run.parents[1] / "game.json").write_text(json.dumps({"video": str(ws.root / video["id"] / "source.video")}))
        (run / "timeline.json").write_text(json.dumps({"dt_s": .25, "segments": [[0, 1, "home"], [1.25, 2, "away"]],
                                                       "back_court": [[1.25, 1.5]]}))
        made = Event("e1", EventKind.SHOT_MADE, 1.0, 30, team=Team.HOME, value=2, attrs={"makes": {"p_made": .9}})
        (run / "timeline.jsonl").write_text(json.dumps(made.to_dict()) + "\n")
        state = client.get("/api/studio", headers=AUTH).json()
        assert state["analysis"]["game"] == "2025-01-01-AAA-BBB" and state["analysis"]["tag"] == "full"
        doc = client.get("/api/studio/analysis", headers=AUTH).json()
        assert doc["segments"] == [[0, 1, "home"], [1.25, 2, "away"]] and doc["back_court"] == [[1.25, 1.5]]
        assert [(e["kind"], e["team"], e["value"], e["p_made"]) for e in doc["events"]] == [("shot_made", "home", 2, .9)]
        assert doc["teams"]["home"]["name"] == "Warriors"  # no ESPN summary: the names given when opening it
        assert client.get("/api/studio/analysis").status_code == 401
        demo = client.post("/api/studio/start", json={"demo": True}, headers=AUTH)
        assert demo.status_code == 200 and demo.json()["analysis"] is None


def test_the_recording_streams_in_ranges_for_watching_with_the_token_in_the_query(app, video_bytes):
    with TestClient(app) as client:
        video = client.post("/api/studio/upload?name=Game.avi", content=video_bytes, headers=AUTH).json()
        url = f"/api/studio/video/{video['id']}"
        assert client.get(url).status_code == 401 and client.get(url + "?token=wrong").status_code == 401
        part = client.get(url + "?token=local-test", headers={"Range": "bytes=0-99"})  # a <video> sends no header
        assert part.status_code == 206 and part.content == video_bytes[:100]
        assert part.headers["content-type"] == "video/x-msvideo"
        full = client.get(url, headers=AUTH)
        assert full.status_code == 200 and full.content == video_bytes
        for missing in ("0" * 32, "g" * 32, "workspace"):
            assert client.get(f"/api/studio/video/{missing}", headers=AUTH).status_code == 404


def test_video_titles_read_like_names():
    assert studio.video_title("YTDown.com_YouTube_Media_5U9k1U6nN-g_Lakers-vs-Warriors-WILD-Christmas-Day-Ending-"
                              "NBA-Classic-Game_001_1080p.mp4") == "Lakers vs Warriors WILD Christmas Day Ending NBA Classic Game"
    assert studio.video_title("2024-12-25 LAL-GSW.mp4") == "2024-12-25 LAL GSW"
    assert studio.video_title("basketball-sample.mp4") == "Basketball sample"
    assert studio.video_title(".mp4") == ".mp4" or studio.video_title(".mp4")  # never empty


def test_library_keeps_one_copy_renames_and_deletes(app, video_bytes, tmp_path):
    from courtvision.nba.results import fingerprint

    with TestClient(app) as client:
        first = client.post("/api/studio/upload?name=YTDown.com_YouTube_Media_abcdefghijk_Big-Game_001_720p.avi",
                            content=video_bytes, headers=AUTH).json()
        assert first["title"] == "Big Game" and not first.get("duplicate")
        again = client.post("/api/studio/upload?name=copy.avi", content=video_bytes, headers=AUTH).json()
        assert again["duplicate"] and again["id"] == first["id"]  # the same video is kept once
        root = app.state.workspace.root
        assert len(list(root.glob("*/video.json"))) == 1
        size, digest = fingerprint(root / first["id"] / "source.video")
        found = client.post("/api/studio/find", json={"bytes": size, "digest": digest}, headers=AUTH).json()
        assert found["video"]["id"] == first["id"]
        assert client.post("/api/studio/find", json={"bytes": size, "digest": "0" * 40}, headers=AUTH).json() == {"video": None}

        renamed = client.post(f"/api/studio/video/{first['id']}/rename", json={"title": "  Finals   game 7 "}, headers=AUTH)
        assert renamed.status_code == 200 and renamed.json()["title"] == "Finals game 7"
        assert client.get("/api/studio", headers=AUTH).json()["videos"][0]["title"] == "Finals game 7"

        # a live analysis of it, finished: its saved results belong to the video
        client.post("/api/studio/start", headers=AUTH, json={"video_id": first["id"], "broadcast": False})
        wait_for(client, lambda s: (s["status"] or {}).get("finished"), seconds=10)
        time.sleep(0.3)
        (root / first["id"] / "replays" / "run1" / "chunks").mkdir(parents=True)
        (root / first["id"] / "replays" / "run1" / "status.json").write_text('{"state": "done"}')
        lib = client.get("/api/studio/library", headers=AUTH).json()
        (video,) = lib["videos"]
        assert video["open"] and video["saved_runs"] == 1 and video["replay_bytes"] > 0
        assert video["disk_bytes"] >= len(video_bytes) and video["used_by"] == [] and video["copies"] == []

        assert client.delete(f"/api/studio/replay/{first['id']}", headers=AUTH).json()["freed_bytes"] > 0
        assert not (root / first["id"] / "replays").exists()

        # a game folder that names this file keeps it
        game = tmp_path / "runs" / "nba" / "2024-01-01-AAA-BBB"
        game.mkdir(parents=True)
        (game / "game.json").write_text(json.dumps({"video": str(root / first["id"] / "source.video")}))
        refused = client.delete(f"/api/studio/video/{first['id']}", headers=AUTH)
        assert refused.status_code == 409 and "2024-01-01-AAA-BBB" in refused.json()["detail"]
        (game / "game.json").unlink()

        gone = client.delete(f"/api/studio/video/{first['id']}", headers=AUTH)
        assert gone.status_code == 200 and gone.json()["freed_bytes"] >= len(video_bytes)
        state = client.get("/api/studio", headers=AUTH).json()
        assert state["active"] is None and state["videos"] == []  # the open video was closed first
        assert not (root / first["id"]).exists() and list((root / "analyses").glob("*/")) == []
        assert client.delete(f"/api/studio/video/{first['id']}", headers=AUTH).status_code == 404
        assert client.delete("/api/studio/video/..%2F..", headers=AUTH).status_code in (404, 405)
        assert client.delete(f"/api/studio/video/{first['id']}").status_code == 401


def test_saved_results_of_the_demo_and_old_runs_are_deleted_together(app):
    with TestClient(app) as client:
        root = app.state.workspace.root
        for run in ("a" * 32, "b" * 32):
            (root / "analyses" / run).mkdir(parents=True)
            (root / "analyses" / run / "events.jsonl").write_text("{}\n")
        lib = client.get("/api/studio/library", headers=AUTH).json()
        assert lib["other_runs"] == 2 and lib["other_bytes"] == 6
        assert client.delete("/api/studio/analyses", headers=AUTH).json() == {"deleted_runs": 2, "freed_bytes": 6}
        assert list((root / "analyses").glob("*/")) == []


def test_public_demo_is_watch_only_and_never_uses_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-used")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-should-not-be-used")
    cfg = load_config(None, ["narration.tts=none"])
    app = create_studio_app(cfg, root=tmp_path / "workspace", lab_runs=tmp_path / "runs", public_demo=True)
    with TestClient(app) as client:
        state = client.get("/api/studio").json()
        assert state["public_demo"] and state["capabilities"] == {"jev": False, "claude": False}
        assert client.post("/api/studio/upload", content=b"x" * 1024).status_code == 403
        assert client.post("/api/studio/prepare").status_code == 403
        assert client.post("/api/control", json={"action": "pause"}).status_code == 403
        assert client.post("/api/scoreboard", json={"home_score": 1, "away_score": 0}).status_code == 403
        assert client.post("/api/studio/start", json={"video_id": "0" * 32}).status_code == 403
        # the write endpoints added since: the video library, replays, the model lab, Ask
        for path in ("/api/studio/find", f"/api/studio/video/{'0' * 32}/rename", f"/api/studio/replay/{'0' * 32}/start",
                     "/api/lab/run", "/api/lab/promote", "/api/ask"):
            assert client.post(path, json={}).status_code == 403, path
        for path in (f"/api/studio/video/{'0' * 32}", f"/api/studio/replay/{'0' * 32}", "/api/studio/analyses"):
            assert client.delete(path).status_code == 403, path
        first = client.post("/api/studio/start", json={"demo": True})
        assert first.status_code == 200, first.text
        status = first.json()["status"]
        assert status["llm"] is False and status["jev"] in (None, "off")
        run_id = first.json()["active"]["run_id"]
        game = app.state.workspace.game
        assert not set(studio.CREDENTIAL_ENV) & set(game.env)  # no keys at all: the local engine only
        assert not game.config.narration.llm_enabled and game.config.jev.mode == "off"
        # A second viewer joins the running demo instead of restarting it for everyone.
        assert client.post("/api/studio/start", json={"demo": True}).json()["active"]["run_id"] == run_id
