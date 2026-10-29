"""Local video workspace. Uploaded footage stays on disk; one analysis owns the hub at a time.

Uncalibrated footage produces image-space tracks only. Spatial events are enabled only after
operator calibration. COCO is an explicitly prepared starter detector, not an NBA-trained model.
"""
from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from courtvision.config import AppConfig, load_config
from courtvision.credentials import runtime_env
from courtvision.perception.rfdetr_adapter import RFDETRConfig, RFDETRDetector
from courtvision.perception.tracker import ByteTracker
from courtvision.perception.broadcast import ShotBoundaryDetector
from courtvision.pipeline import (FrameItem, LiveGame, VideoFileSource, _collect, anthropic_credential_source, make_source,
                                  write_run_dir)
from courtvision.server.app import HubJSONResponse, create_app
from courtvision.server.hub import Hub, status_msg
from courtvision.server.lab import Lab, register_lab
from courtvision.types import ObjClass

log = logging.getLogger(__name__)
UPLOAD_LIMIT = 8 * 1024**3


def probe_video(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    try:
        ok, image = cap.read()
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not ok or image is None or not math.isfinite(fps) or not 0 < fps <= 1000:
            raise ValueError("This file could not be decoded as video. Try an MP4 encoded with H.264.")
        h, w = image.shape[:2]
        if not 16 <= w <= 16384 or not 16 <= h <= 16384:
            raise ValueError("Video dimensions must be between 16 and 16384 pixels.")
        return {"width": w, "height": h, "fps": fps,
                "duration_s": count / fps if math.isfinite(count) and count > 0 else None}
    finally:
        cap.release()


class StudioGame(LiveGame):
    """Run useful image-space detection while awaiting a real court calibration."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.preview_tracker = ByteTracker()
        self.preview_cuts = ShotBoundaryDetector()
        self.visual: tuple[np.ndarray, list[dict[str, Any]], float] | None = None
        self.vision: dict[str, Any] = {}
        self.failure: str | None = None
        self._vision_t = -1.0
        self._consecutive_failures = 0
        self._jpeg_cache: dict[tuple[float, bool, str], bytes] = {}

    def _perceive(self, image: np.ndarray, frame_idx: int, t: float) -> Any:
        started = time.perf_counter()
        result = super()._perceive(image, frame_idx, t)
        boxes: list[dict[str, Any]] = []
        gate = getattr(self.perception, "last_gate", None)
        if result is None:
            # No invented homography: until setup, display tracks in pixels and emit no stats.
            fd = self._detector.detect(image, frame_idx, t)
            if self.preview_cuts.update(image):
                self.preview_tracker = ByteTracker()
            people = [d for d in fd.of(ObjClass.PLAYER) if d.bbox.h >= image.shape[0] * .045]
            for tr in self.preview_tracker.update(people, frame_idx):
                if tr.time_since_update == 0:
                    boxes.append({"box": tr.bbox.as_tuple(), "label": f"Track {tr.track_id}",
                                  "team": None, "confidence": tr.score})
            for ball in fd.of(ObjClass.BALL):
                boxes.append({"box": ball.bbox.as_tuple(), "label": "Ball", "team": "ball",
                              "confidence": ball.score})
        else:
            frame, _ = result
            for p in frame.players:
                if p.bbox is not None:
                    boxes.append({"box": p.bbox.as_tuple(), "label": p.player_id,
                                  "team": p.team.value if p.team else None, "confidence": p.confidence})
            if frame.ball is not None and frame.ball.visible and frame.ball.image_xy is not None:
                u, v = frame.ball.image_xy
                boxes.append({"box": (u - 9, v - 9, u + 9, v + 9), "label": "Ball", "team": "ball"})
        ms = (time.perf_counter() - started) * 1000
        if result is None:
            self._lat.perception_ms.append(ms)
        self._consecutive_failures = 0
        self.visual = (image, boxes, t)
        self.vision = {"type": "vision", "t": t, "people": sum(b["team"] != "ball" for b in boxes),
                       "ball_visible": any(b["team"] == "ball" for b in boxes),
                       "inference_ms": round(ms, 1), "fps": round(1000 / max(ms, 1), 1),
                       "calibrated": result is not None, "gate": gate.reason if gate else None,
                       "tracks": [{"id": b["label"], "team": b["team"],
                                   "confidence": b.get("confidence")}
                                  for b in boxes if b["team"] != "ball"]}
        return result

    async def _await_calibration(self, t: float) -> None:
        # Let pixel tracks continue; the setup UI pauses and freezes the chosen calibration frame.
        self._t = t
        self._frames_done += 1
        if self.hub is not None and t - self._vision_t >= .25:
            await self.hub.publish(status_msg(t, **self.status()))

    async def process(self, item: FrameItem) -> None:
        before = self._errors
        await super().process(item)
        if self._errors > before:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                self.failure = "Analysis failed on consecutive frames. Reopen the video or check the detector setup."
                self.request_stop(drain=False)
        if self.hub is not None and item.t - self._vision_t >= .25:
            self._vision_t = item.t
            if self.vision:
                await self.hub.publish(self.vision)
            await self.hub.publish(status_msg(self.current_t, **self.status(), error=self.failure))

    async def monitor_jpeg(self, overlay: bool, focus: str = "") -> bytes | None:
        visual = self.visual
        if visual is None:
            return await self.controller.latest_frame_jpeg()
        image, boxes, t = visual
        key = (t, overlay, focus)
        if key in self._jpeg_cache:
            return self._jpeg_cache[key]

        def encode() -> bytes | None:
            scale = min(1., 1440 / image.shape[1])
            frame = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image.copy()
            if overlay:
                for b in boxes:
                    x1, y1, x2, y2 = (round(v * scale) for v in b["box"])
                    h, w = frame.shape[:2]
                    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
                    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    color = {"home": (237, 187, 120), "away": (117, 191, 246),
                             "ball": (105, 224, 248)}.get(b["team"], (151, 235, 209))
                    muted = bool(focus and focus != b["label"] and b["team"] != "ball")
                    if muted:
                        color = (106, 112, 110)
                    if b["team"] == "ball":
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        cv2.circle(frame, (cx, cy), max(9, (x2-x1)//2+4), color, 2, cv2.LINE_AA)
                        cv2.line(frame, (cx-4, cy), (cx+4, cy), color, 1, cv2.LINE_AA)
                        cv2.line(frame, (cx, cy-4), (cx, cy+4), color, 1, cv2.LINE_AA)
                        continue
                    # Corner brackets keep bodies readable. The bottom marker is the box base,
                    # not a measured joint or calibrated foot position.
                    length = max(3, min(20, (x2-x1)//3, (y2-y1)//4))
                    thickness = 3 if focus == b["label"] else 2
                    for x, dx in ((x1, 1), (x2, -1)):
                        for y, dy in ((y1, 1), (y2, -1)):
                            cv2.line(frame, (x, y), (x+dx*length, y), color, thickness, cv2.LINE_AA)
                            cv2.line(frame, (x, y), (x, y+dy*length), color, thickness, cv2.LINE_AA)
                    if muted:
                        continue
                    cv2.ellipse(frame, ((x1+x2)//2, y2), (max(6, (x2-x1)//3), 4), 0, 0, 360, color, 1, cv2.LINE_AA)
                    cv2.circle(frame, ((x1+x2)//2, y2), 2, color, -1, cv2.LINE_AA)
                    label = str(b["label"])
                    confidence = b.get("confidence")
                    if confidence is not None:
                        label += f" / {confidence:.0%}"
                    font_scale = .42 if w < 800 else .5
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                    xx = max(0, min(x1, w - tw - 12))
                    yy = min(h - 1, max(th + 10, y1 - 5))
                    cv2.rectangle(frame, (xx, yy-th-9), (min(w-1, xx+tw+12), yy), (24, 30, 27), -1)
                    cv2.line(frame, (xx, yy-th-9), (xx, yy), color, 2)
                    cv2.putText(frame, label, (xx+6, yy-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                color, 1, cv2.LINE_AA)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buf.tobytes() if ok else None

        encoded = await asyncio.to_thread(encode)
        self._jpeg_cache = {key: encoded} if encoded else {}
        return encoded


class StartIn(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    video_id: str | None = Field(None, pattern=r"^[a-f0-9]{32}$")
    demo: bool = False
    home_name: str = Field("Home", min_length=1, max_length=48)
    away_name: str = Field("Away", min_length=1, max_length=48)
    broadcast: bool = True


class Workspace:
    def __init__(self, cfg: AppConfig, root: Path, detector: Any = None):
        self.cfg, self.root = cfg, root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.hub = Hub(cfg.hub)
        self.game: LiveGame | None = None
        self.task: asyncio.Task[Any] | None = None
        self.active: dict[str, Any] | None = None
        self.detector = detector
        self.model_state = "ready" if detector is not None else "idle"
        self.model_error: str | None = None
        self.model_name = ("Configured RF-DETR checkpoint" if cfg.perception.rfdetr.weights or
                           cfg.perception.rfdetr.coreml_path else
                           "NBA broadcast detector" if self.student_weights() is not None else "RF-DETR Nano · COCO starter")
        self.model_task: asyncio.Task[Any] | None = None
        self.lock = asyncio.Lock()
        self.upload_lock = asyncio.Lock()

    def videos(self) -> list[dict[str, Any]]:
        videos = []
        for path in self.root.glob("*/video.json"):
            try:
                videos.append(json.loads(path.read_text()))
            except (ValueError, OSError):
                log.warning("Could not read video metadata %s", path)
        return sorted(videos, key=lambda x: x.get("created", 0), reverse=True)

    def state(self) -> dict[str, Any]:
        game = self.game
        return {"enabled": True, "active": self.active, "videos": self.videos(),
                "model": {"state": self.model_state, "error": self.model_error,
                          "name": self.model_name},
                "status": game.status() if game else None,
                "frame": self.hub.latest("frame"),
                "vision": getattr(game, "vision", None),
                "error": getattr(game, "failure", None),
                "capabilities": {"jev": bool(runtime_env().get("TYPESAFE_API_KEY")),
                                 "claude": anthropic_credential_source() is not None}}

    def student_weights(self) -> Path | None:
        """The NBA broadcast student detector (training/train_student.py), when one is installed in the data folder's
        models (``models/nba-student.pt``); it replaces the COCO starter unless an RF-DETR checkpoint is configured."""
        p = self.root / "models" / "nba-student.pt"
        rf = self.cfg.perception.rfdetr
        return p if p.is_file() and not (rf.weights or rf.coreml_path) else None

    async def prepare(self) -> None:
        self.model_state, self.model_error = "loading", None
        student = self.student_weights()
        if student is not None:
            try:
                def load_student() -> Any:
                    from courtvision.perception.student import StudentConfig, StudentDetector

                    det = StudentDetector(StudentConfig(weights=str(student)))
                    det.detect(np.zeros((540, 960, 3), np.uint8), 0, 0.)
                    det.reset()
                    return det
                self.detector = await asyncio.to_thread(load_student)
                self.model_state = "ready"
                return
            except Exception:
                log.exception("Could not load the NBA student detector; falling back to the COCO starter")
        try:
            def load() -> RFDETRDetector:
                import torch
                weights = self.root / "models" / "rf-detr-nano.pth"
                weights.parent.mkdir(exist_ok=True)
                if self.cfg.perception.rfdetr.weights or self.cfg.perception.rfdetr.coreml_path:
                    config = copy.deepcopy(self.cfg.perception.rfdetr)
                else:
                    config = RFDETRConfig(variant="nano", weights=str(weights), allow_pretrained_download=True)
                if config.device is None:
                    config.device = "mps" if torch.backends.mps.is_available() else "cpu"
                detector = RFDETRDetector(config)
                detector.detect(np.zeros((384, 640, 3), np.uint8), 0, 0.)
                return detector
            self.detector = await asyncio.to_thread(load)
            self.model_state = "ready"
        except Exception:
            log.exception("Could not prepare starter detector")
            self.model_state = "error"
            self.model_error = "Detector setup failed. Check your connection and install courtvision[cv]. Then retry."

    async def stop(self) -> None:
        if self.game is not None and self.task is not None:
            from courtvision.cli import stop_game
            await stop_game(self.game, self.task)
        self.task = None

    async def start(self, body: StartIn) -> dict[str, Any]:
        async with self.lock:
            cfg = copy.deepcopy(self.cfg)
            cfg.narration.tts = "none"
            cfg.source.drop_late_s = None  # process every frame, report measured throughput honestly
            cfg.game.home_name, cfg.game.away_name = body.home_name.strip(), body.away_name.strip()
            cfg.source.kind = "sim" if body.demo else "video"
            if body.demo:
                cfg.source.sim_mode, cfg.sim.seconds = "truth3d", 180.
                source = await asyncio.to_thread(make_source, cfg, realtime=True)
                active = {"id": uuid.uuid4().hex, "name": "Studio demo", "duration_s": 180., "demo": True}
            else:
                if body.video_id is None:
                    raise HTTPException(400, "Choose a video first.")
                path = self.root / body.video_id / "video.json"
                if not path.is_file():
                    raise HTTPException(404, "Video not found.")
                if self.model_state != "ready" or self.detector is None:
                    raise HTTPException(409, "Prepare the detector before starting video analysis.")
                active = json.loads(path.read_text())
                cfg.source.path = str(self.root / body.video_id / "source.video")
                cfg.perception.calibration_path = str(self.root / body.video_id / "calibration.json")
                cfg.perception.scoreboard_roi_path = str(self.root / body.video_id / "scoreboard_roi.json")
                cfg.perception.save_calibration = True
                cfg.perception.broadcast = "on" if body.broadcast else "off"
                # The scorebug must be explicitly configured before it can gate live play. Without
                # one, court visibility gates cutaways; replay footage still needs operator review.
                cfg.perception.broadcast_config.live_requires_scorebug = False
                if type(self.detector).__name__ == "StudentDetector":
                    # its person scores are centre-heatmap peaks (lower than a classifier's): ByteTrack thresholds to
                    # match, and boxes re-drawn from the tracked camera (perception/broadcast.py)
                    cfg.perception.pipeline.tracker = {"high_thresh": 0.3, "low_thresh": 0.1, "new_track_thresh": 0.35,
                                                       **cfg.perception.pipeline.tracker}
                    cfg.perception.broadcast_config.rectify_boxes = True
                    self.detector.reset()
                source = await asyncio.to_thread(VideoFileSource, cfg.source.path, realtime=True)
            await self.stop()
            await self.hub.reset()
            self.active = {**active, "run_id": uuid.uuid4().hex}
            self.game = StudioGame(cfg, source, self.hub, realtime=True,
                                   detector=None if body.demo else self.detector, env=runtime_env())
            self.task = asyncio.create_task(self.run(self.game, dict(self.active)), name="studio-analysis")
            return self.state()

    async def run(self, game: LiveGame, active: dict[str, Any]) -> None:
        try:
            await game.run()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Video analysis stopped")
            game.failure = "Analysis stopped. Reopen this video to retry."
        finally:
            await self.hub.publish(status_msg(game.current_t, **game.status(),
                                             error=getattr(game, "failure", None)))
            # Each run has its own folder. Reopening footage never overwrites previous analysis.
            try:
                result = _collect(game)
                await asyncio.to_thread(write_run_dir, result, self.root / "analyses" / active["run_id"])
            except Exception:
                log.exception("Could not persist analysis output")

    def controller(self) -> Any:
        if self.game is None:
            raise HTTPException(409, "Open a video or start the demo first.")
        return self.game.controller

    async def add_manual(self, data: dict[str, Any]) -> Any:
        return await self.controller().add_manual(data)

    async def set_scoreboard(self, data: dict[str, Any]) -> Any:
        return await self.controller().set_scoreboard(data)

    async def calibrate(self, data: dict[str, Any]) -> Any:
        return await self.controller().calibrate(data)

    async def control(self, data: dict[str, Any]) -> Any:
        return await self.controller().control(data)

    async def latest_frame_jpeg(self) -> Any:
        return await self.controller().latest_frame_jpeg()

    async def set_lineup(self, data: dict[str, Any]) -> Any:
        return await self.controller().set_lineup(data)

    async def set_scoreboard_roi(self, data: dict[str, Any]) -> Any:
        return await self.controller().set_scoreboard_roi(data)


def create_studio_app(config: AppConfig | None = None, *, root: Path | str = "runs/studio",
                      auth_token: str | None = None, detector: Any = None, lab_runs: Path | None = None) -> Any:
    workspace = Workspace(config or load_config(), Path(root), detector)
    app = create_app(workspace.hub, workspace, auth_token, workspace.cfg.server, upload_max_body=UPLOAD_LIMIT)
    app.state.workspace = workspace
    inner = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(a: Any):
        async with inner(a) as state:
            try:
                yield state
            finally:
                app.state.lab.shutdown()
                await workspace.stop()
                if workspace.model_task is not None:
                    # A model load runs in a thread: wait for it before releasing the process.
                    await workspace.model_task
    app.router.lifespan_context = lifespan
    read, write = Depends(app.state.require_read), Depends(app.state.require_write)
    app.state.asker.studio_state = workspace.state
    lab = Lab(lab_runs)
    app.state.lab = lab
    register_lab(app, lab, read, write)

    @app.get("/api/studio", dependencies=[read])
    async def studio_state() -> Any:
        return workspace.state()

    @app.post("/api/studio/prepare", dependencies=[write])
    async def prepare() -> Any:
        if workspace.model_state not in ("ready", "loading"):
            workspace.model_state = "loading"
            workspace.model_task = asyncio.create_task(workspace.prepare(), name="studio-detector")
        return {"state": workspace.model_state}

    @app.post("/api/studio/upload", dependencies=[write])
    async def upload(request: Request, name: str = "Game video") -> Any:
        # Stream raw file bytes; do not buffer an NBA game in RAM or require multipart dependencies.
        async with workspace.upload_lock:
            video_id = uuid.uuid4().hex
            directory = workspace.root / video_id
            directory.mkdir()
            path = directory / "source.video"
            size = 0
            try:
                with path.open("wb") as f:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > UPLOAD_LIMIT:
                            raise HTTPException(413, "Video exceeds the 8 GB upload limit.")
                        await asyncio.to_thread(f.write, chunk)
                metadata = await asyncio.to_thread(probe_video, path)
                record = {"id": video_id, "name": Path(name.replace("\\", "/")).name[:160] or "Game video",
                          "bytes": size, "created": time.time(), **metadata}
                (directory / "video.json").write_text(json.dumps(record))
                return record
            except BaseException as exc:
                # Only this incomplete upload is removed; valid footage and past analyses are kept.
                path.unlink(missing_ok=True)
                directory.rmdir()
                if isinstance(exc, ValueError):
                    raise HTTPException(400, str(exc)) from exc
                raise

    @app.post("/api/studio/start", dependencies=[write])
    async def start(body: StartIn) -> Any:
        return await workspace.start(body)

    @app.get("/api/studio/frame.jpg", dependencies=[read])
    async def monitor(overlay: bool = True, focus: str = Query("", max_length=128)) -> Response:
        game = workspace.game
        if game is None:
            raise HTTPException(404, "No video is open.")
        t = game.visual[2] if game.visual else game.current_t
        image = await game.monitor_jpeg(overlay, focus)
        if image is None:
            raise HTTPException(404, "Waiting for the first frame.")
        return Response(image, media_type="image/jpeg", headers={"Cache-Control": "no-store",
                                                                 "X-Video-Time": str(t)})

    @app.get("/api/studio/export", dependencies=[read])
    async def export() -> Response:
        if workspace.game is None:
            raise HTTPException(404, "Open a video before exporting.")
        game = workspace.game
        content = {"video": workspace.active, "status": game.status(),
                   "events": [e.to_dict() for e in game.events], "stats": game.snapshot().to_dict(),
                   "transcript": workspace.hub.transcript(), "vision": getattr(game, "vision", {}),
                   "latency": game.latency_report(),
                   "notes": "Computer vision estimates. Review events and player identities before use."}
        return HubJSONResponse(content, headers={"Content-Disposition": 'attachment; filename="courtvision-analysis.json"'})

    return app
