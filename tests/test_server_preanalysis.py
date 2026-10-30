"""Analysing a video ahead and replaying it (courtvision/server/preanalysis.py, the Studio's replay endpoints)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from courtvision.config import load_config
from courtvision.server import preanalysis as pa
from courtvision.server.studio import create_studio_app
from courtvision.types import BBox, Detection, FrameDetections, ObjClass

AUTH = {"Authorization": "Bearer local-test"}


class PeopleDetector:
    def detect(self, image, frame_idx, t):
        h, w = image.shape[:2]
        return FrameDetections(frame_idx, t, (w, h), [
            Detection(BBox(w*.2, h*.15, w*.4, h*.9), ObjClass.PLAYER, .95),
            Detection(BBox(w*.5, h*.4, w*.53, h*.45), ObjClass.BALL, .9),
        ])


def write_video(path: Path, frames: int = 60, fps: float = 30.) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 96))
    rng = np.random.default_rng(7)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (96, 160, 3), dtype=np.uint8))
    writer.release()
    return path


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def box(label: str, team: str | None = "home") -> dict:
    return {"box": (10.4, 20.6, 50.0, 90.0), "label": label, "team": team, "confidence": 0.876}


def test_the_recorder_cuts_minutes_and_keeps_the_sparse_messages_apart(tmp_path):
    rec = pa.Recorder(tmp_path)
    rec.message("meta", json.dumps({"type": "meta", "t": None, "teams": {"home": "H"}}))
    rec.frame(59.9, [box("home-t1"), box("Ball", "ball")])
    rec.message("event", json.dumps({"type": "event", "t": 59.9, "event": {"event_id": "e1", "revision": 0}}))
    rec.message("status", json.dumps({"type": "status", "t": 59.9}))  # the pipeline's state, not the game's
    rec.frame(60.1, [box("home-t1")])
    rec.message("utterance", json.dumps({"type": "utterance", "t": 60.1, "utterance": {"utterance_id": "u1"}}))
    rec.close()
    first, second = lines(pa.chunk_path(tmp_path, 0)), lines(pa.chunk_path(tmp_path, 1))
    assert first[0] == {"vt": 0.0, "m": {"type": "meta", "t": None, "teams": {"home": "H"}}}
    assert first[1] == {"vt": 59.9, "b": [[10, 21, 50, 90, "home-t1", "home", 0.88], [10, 21, 50, 90, "Ball", "ball", 0.88]]}
    assert [r["m"]["type"] for r in first[2:]] == ["event"] and len(second) == 2
    assert [r["m"]["type"] for r in lines(tmp_path / "sparse.jsonl")] == ["event", "utterance"]
    assert json.loads((tmp_path / "meta_msg.json").read_text())["teams"] == {"home": "H"}
    assert rec.frames == 2 and rec.first_t == 59.9 and rec.t == 60.1


def test_stats_are_kept_without_tracking_every_two_seconds_and_at_each_score_change(tmp_path):
    rec = pa.Recorder(tmp_path)

    def stats(t: float, home: int) -> None:
        rec.frame(t, [])
        rec.message("stats", json.dumps({"type": "stats", "t": t, "stats": {"score": {"home": home, "away": 0},
                                                                         "tracking": {"t1": {"m": 1}}}}))
    for t, home in ((0.0, 0), (1.0, 0), (1.5, 2), (2.0, 2), (3.6, 2), (4.0, 2)):
        stats(t, home)
    rec.close()
    kept = [r for r in lines(pa.chunk_path(tmp_path, 0)) if "m" in r]
    assert [(r["vt"], r["m"]["stats"]["score"]["home"]) for r in kept] == [(0.0, 0), (1.5, 2), (3.6, 2)]
    assert all("tracking" not in r["m"]["stats"] for r in kept)


def test_the_view_at_a_moment_has_the_newest_revisions_and_the_last_frame_and_stats(tmp_path):
    rec = pa.Recorder(tmp_path)
    rec.message("meta", json.dumps({"type": "meta", "t": None}))

    def at(t: float, *msgs: dict) -> None:
        rec.frame(t, [box("home-t1")])
        for m in msgs:
            rec.message(m["type"], json.dumps(m))
    at(10.0, {"type": "event", "t": 10.0, "event": {"event_id": "e1", "revision": 0, "kind": "shot_made"}},
       {"type": "stats", "t": 10.0, "stats": {"score": {"home": 2, "away": 0}}}, {"type": "frame", "t": 10.0})
    at(59.0, {"type": "event", "t": 10.0, "event": {"event_id": "e1", "revision": 1, "kind": "shot_missed"}},
       {"type": "vision", "t": 59.0})
    at(61.0, {"type": "frame", "t": 61.0}, {"type": "event", "t": 61.0, "event": {"event_id": "e2", "revision": 0}},
       {"type": "utterance", "t": 61.0, "utterance": {"utterance_id": "u1"}})
    at(90.0, {"type": "frame", "t": 90.0})
    rec.close()
    st = pa.replay_state(tmp_path, 61.5)  # stats and vision from the minute before
    assert st["meta"] == {"type": "meta", "t": None}
    assert [(m["event"]["event_id"], m["event"]["revision"]) for m in st["events"]] == [("e1", 1), ("e2", 0)]
    assert st["frame"]["t"] == 61.0 and st["stats"]["t"] == 10.0 and st["vision"]["t"] == 59.0
    assert len(st["utterances"]) == 1
    early = pa.replay_state(tmp_path, 30.0)
    assert [m["event"]["revision"] for m in early["events"]] == [0] and early["frame"]["t"] == 10.0
    assert early["utterances"] == [] and early["vision"] is None


def test_a_strided_source_reads_every_nth_frame(tmp_path):
    import asyncio

    source = pa.StridedVideoSource(write_video(tmp_path / "v.avi", 30), 3)

    async def times() -> list[float]:
        return [item.t async for item in source._iterate()]
    ts = asyncio.run(times())
    assert len(ts) == 10 and source.fps == pytest.approx(10)
    assert np.allclose(np.diff(ts), 0.1, atol=0.01)


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = load_config(None, ["narration.llm_enabled=false", "narration.tts=none", "jev.mode=off"])
    return create_studio_app(cfg, root=tmp_path / "workspace", auth_token="local-test", detector=PeopleDetector(),
                             lab_runs=tmp_path / "runs", detector_factory=PeopleDetector)


def wait_replay(client, video_id: str, states: tuple[str, ...], seconds: float = 20.) -> dict:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        status = client.get(f"/api/studio/replay/{video_id}", headers=AUTH).json()
        if status.get("state") in states:
            return status
        time.sleep(.05)
    pytest.fail(f"pre-analysis did not reach {states}: {status}")


def test_an_upload_is_analysed_ahead_and_replayed_from_any_moment(app, tmp_path):
    data = write_video(tmp_path / "clip.avi").read_bytes()
    with TestClient(app) as client:
        video = client.post("/api/studio/upload?name=Game.avi", content=data, headers=AUTH).json()
        assert video["replay"]["state"] in ("queued", "loading", "running")  # started by the upload
        status = wait_replay(client, video["id"], ("done",))
        assert status["frames"] == 60 and status["t_done"] == pytest.approx(59 / 30, abs=.05) and status["stride"] == 1
        chunk = client.get(f"/api/studio/replay/{video['id']}/chunk/0", headers=AUTH)
        assert chunk.status_code == 200 and chunk.headers["content-type"] == "application/x-ndjson"
        recs = [json.loads(x) for x in chunk.text.splitlines()]
        boxes = [r for r in recs if "b" in r]
        assert len(boxes) == 60 and any(b[5] == "ball" for b in boxes[0]["b"])
        assert {r["m"]["type"] for r in recs if "m" in r} >= {"meta", "vision"}
        view = client.get(f"/api/studio/replay/{video['id']}/state?t=1.0", headers=AUTH).json()
        assert view["meta"]["type"] == "meta" and view["vision"]["t"] <= 1.0
        assert client.get(f"/api/studio/replay/{video['id']}/chunk/1", headers=AUTH).status_code == 404
        assert client.get("/api/studio", headers=AUTH).json()["videos"][0]["replay"]["state"] == "done"
        # again, every 2nd frame: a new run beside the first; stopping one that is not running is refused
        again = client.post(f"/api/studio/replay/{video['id']}/start", json={"stride": 2}, headers=AUTH)
        assert again.status_code == 200
        assert wait_replay(client, video["id"], ("done",))["frames"] == 30
        assert len(list((app.state.workspace.root / video["id"] / "replays").iterdir())) == 2
        assert client.post(f"/api/studio/replay/{video['id']}/stop", headers=AUTH).status_code == 409
        assert client.post(f"/api/studio/replay/{video['id']}/start", json={"stride": 4}, headers=AUTH).status_code == 400
        assert client.get(f"/api/studio/replay/{video['id']}/state?t=1").status_code == 401
        assert client.post(f"/api/studio/replay/{'0' * 32}/start", json={}, headers=AUTH).status_code == 404
        assert client.get("/api/studio/replay/..%2Fmodels/chunk/0", headers=AUTH).status_code == 404


def test_a_queued_analysis_can_be_cancelled_and_a_running_one_stopped(app, tmp_path):
    data = write_video(tmp_path / "clip.avi", 600).read_bytes()  # 20 s: still running when stopped
    other = write_video(tmp_path / "clip2.avi", 601).read_bytes()  # another video (a copy would be kept once)
    with TestClient(app) as client:
        first = client.post("/api/studio/upload?name=One.avi", content=data, headers=AUTH).json()
        second = client.post("/api/studio/upload?name=Two.avi", content=other, headers=AUTH).json()
        assert second["replay"]["state"] == "queued"  # one at a time
        assert client.post(f"/api/studio/replay/{second['id']}/stop", headers=AUTH).json()["state"] != "queued"
        wait_replay(client, first["id"], ("running",))
        assert client.post(f"/api/studio/replay/{first['id']}/start", json={}, headers=AUTH).status_code == 409
        client.post(f"/api/studio/replay/{first['id']}/stop", headers=AUTH)
        stopped = wait_replay(client, first["id"], ("stopped",))
        assert 0 < stopped["frames"] < 600
        assert client.get(f"/api/studio/replay/{second['id']}", headers=AUTH).json()["state"] == "none"


def test_an_upload_does_not_start_one_that_would_download_a_detector(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    cfg = load_config(None, ["narration.llm_enabled=false", "narration.tts=none", "jev.mode=off"])
    app = create_studio_app(cfg, root=tmp_path / "workspace", auth_token="local-test", detector=PeopleDetector(),
                            lab_runs=tmp_path / "runs")
    data = write_video(tmp_path / "clip.avi").read_bytes()
    with TestClient(app) as client:
        video = client.post("/api/studio/upload?name=Game.avi", content=data, headers=AUTH).json()
        assert video["replay"] is None and app.state.workspace.preanalysis is None
