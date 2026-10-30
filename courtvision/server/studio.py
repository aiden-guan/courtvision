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
import re
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from courtvision.config import AppConfig, load_config
from courtvision.credentials import runtime_env
from courtvision.nba import results as game_results
from courtvision.perception.rfdetr_adapter import RFDETRConfig, RFDETRDetector
from courtvision.pipeline import (LiveGame, VideoFileSource, _collect, anthropic_credential_source, make_source,
                                  write_run_dir)
from courtvision.server.app import HubJSONResponse, create_app
from courtvision.server.hub import Hub, status_msg
from courtvision.server import preanalysis
from courtvision.server.lab import REPO, Lab, register_lab
from courtvision.server.preanalysis import Preanalysis
from courtvision.server.studio_game import StudioGame

log = logging.getLogger(__name__)
UPLOAD_LIMIT = 8 * 1024**3
VIDEO_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
               ".mkv": "video/x-matroska", ".avi": "video/x-msvideo"}
ANALYSIS_RECHECK_S = 10.0  # how often the open video's analysis is looked for again (a run may finish meanwhile)


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


def video_title(name: str) -> str:
    """A readable title from an uploaded file's name: no extension, underscores and joining hyphens as spaces, and
    without a downloader's prefix ("YTDown.com_YouTube_Media_<video id>_") or trailing copy and resolution tags
    ("_001_1080p")."""
    s = re.sub(r"_+", " ", Path(name).stem)
    s = re.sub(r"^\S+\.(?:com|net|org|io|app|to|cc)\b\s*", "", s, flags=re.I)
    s = re.sub(r"^(?:(?:youtube|media|video|download)\s+)+[\w-]{11}\s+", "", s, flags=re.I)
    s = re.sub(r"(?:\s+(?:\d{3,4}p|\d{1,3}|hd|4k|uhd|\d{2}fps))+$", "", s, flags=re.I)
    s = re.sub(r"(?<=[^\W\d_])-(?=[^\W\d_])", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:1].upper() + s[1:])[:120] if s else (Path(name).stem or "Game video")


def folder_bytes(path: Path) -> int:
    """Bytes of the files under ``path`` (0 when it does not exist)."""
    total = 0
    for p in path.rglob("*") if path.is_dir() else ():
        with contextlib.suppress(OSError):
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
    return total


class StartIn(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    video_id: str | None = Field(None, pattern=r"^[a-f0-9]{32}$")
    demo: bool = False
    home_name: str = Field("Home", min_length=1, max_length=48)
    away_name: str = Field("Away", min_length=1, max_length=48)
    broadcast: bool = True


class RenameIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(..., min_length=1, max_length=120)


class FindIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bytes: int = Field(..., ge=0)
    digest: str = Field(..., pattern=r"^[a-f0-9]{40}$")  # SHA-1 of the first and last MiB (``results.fingerprint``)


class PreanalyseIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stride: int = Field(1, ge=1, le=3)  # analyse every frame, or every 2nd or 3rd (2-3x sooner done)


class Workspace:
    def __init__(self, cfg: AppConfig, root: Path, detector: Any = None, nba_root: Path | None = None,
                 repo: Path = REPO, detector_factory: Callable[[], Any] | None = None, auto_preanalyze: bool = True):
        self.cfg, self.root = cfg, root.resolve()
        # analysing videos ahead (server/preanalysis.py): one at a time, each with its own detector
        self.detector_factory, self.auto_preanalyze = detector_factory, auto_preanalyze
        self.preanalysis: Preanalysis | None = None
        self.preanalysis_task: asyncio.Task[Any] | None = None
        self.preanalysis_queue: list[tuple[str, int]] = []
        # analysed games (tools/nba_analyze.py): runs/nba/<game>; a game.json's relative video path is from the repo
        self.nba_root, self.repo = (nba_root or repo / "runs" / "nba").resolve(), repo
        self._analysis: tuple[str, float, tuple[Path, str, float] | None] | None = None  # (video id, looked at, run)
        self._analysis_doc: tuple[tuple[Path, str, float], dict[str, Any]] | None = None
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
                record = json.loads(path.read_text())
                videos.append({**record, "title": record.get("title") or video_title(str(record.get("name", ""))),
                               "replay": self.replay_status(path.parent.name)})
            except (ValueError, OSError):
                log.warning("Could not read video metadata %s", path)
        return sorted(videos, key=lambda x: x.get("created", 0), reverse=True)

    # ------------------------------------------------------------------ the video library: sizes, renaming, deleting

    def video_dir(self, video_id: str) -> Path:
        """The folder of an uploaded video (404 for anything else)."""
        if not re.fullmatch(r"[a-f0-9]{32}", video_id) or not (self.root / video_id / "video.json").is_file():
            raise HTTPException(404, "Video not found.")
        return self.root / video_id

    def used_by(self, video_id: str) -> list[str]:
        """Game folders under ``nba_root`` whose ``game.json`` names this upload's file: deleting it would break them
        (a copy of the same video elsewhere is not this file)."""
        source = (self.root / video_id / "source.video").resolve()
        out = []
        for meta in sorted(self.nba_root.glob("*/game.json")):
            try:
                named = Path(str(json.loads(meta.read_text())["video"]))
                if (named if named.is_absolute() else self.repo / named).resolve() == source:
                    out.append(meta.parent.name)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return out

    def saved_runs(self) -> list[tuple[Path, str | None]]:
        """The saved results of live analyses (``analyses/<run>``) and the video each was of (None: the demo, or
        saved before runs recorded it)."""
        out = []
        for run in sorted((self.root / "analyses").glob("*/")):
            try:
                video = json.loads((run / "source.json").read_text()).get("video_id")
            except (OSError, ValueError, AttributeError):
                video = None
            out.append((run, video))
        return out

    def library(self) -> dict[str, Any]:
        """Every video with what it keeps on disk and whether it can be deleted, and the saved live results (blocking:
        it reads the folders and each file's first and last MiB to find copies of the same video)."""
        active = self.active.get("id") if self.active and not self.active.get("demo") else None
        running = self.active_run_id()
        saved = self.saved_runs()
        videos, prints = [], {}
        for v in self.videos():
            folder = self.root / v["id"]
            with contextlib.suppress(OSError):
                prints[v["id"]] = game_results.fingerprint(folder / "source.video")
            runs = [run for run, vid in saved if vid == v["id"] and run.name != running]
            videos.append({**v, "disk_bytes": folder_bytes(folder), "replay_bytes": folder_bytes(folder / "replays"),
                           "saved_runs": len(runs), "saved_bytes": sum(folder_bytes(r) for r in runs),
                           "used_by": self.used_by(v["id"]), "open": v["id"] == active})
        for v in videos:  # copies of the same file, oldest first: the later ones say which they repeat
            same = [w for w in videos if w["id"] != v["id"] and prints.get(w["id"]) is not None
                    and prints.get(w["id"]) == prints.get(v["id"])]
            v["copies"] = [w["id"] for w in same]
        other = [run for run, vid in saved if (vid is None or not (self.root / vid).is_dir()) and run.name != running]
        return {"videos": videos, "other_runs": len(other), "other_bytes": sum(folder_bytes(r) for r in other),
                "total_bytes": sum(v["disk_bytes"] for v in videos) + folder_bytes(self.root / "analyses")}

    def active_run_id(self) -> str | None:
        """The run a live analysis is still writing (its results are saved when it ends)."""
        return self.active.get("run_id") if self.active and self.task is not None and not self.task.done() else None

    def find_copy(self, size: int, digest: str) -> dict[str, Any] | None:
        """An uploaded video with this size and first-and-last-MiB digest (``results.fingerprint``), if any."""
        for v in self.videos():
            source = self.root / v["id"] / "source.video"
            with contextlib.suppress(OSError):
                if source.stat().st_size == size and game_results.fingerprint(source) == (size, digest):
                    return v
        return None

    def rename(self, video_id: str, title: str) -> dict[str, Any]:
        path = self.video_dir(video_id) / "video.json"
        record = {**json.loads(path.read_text()), "title": " ".join(title.split())[:120]}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record))
        tmp.replace(path)
        if self.active and self.active.get("id") == video_id:
            self.active = {**self.active, "title": record["title"]}
        return {**record, "replay": self.replay_status(video_id)}

    async def _end_preanalysis(self, video_id: str) -> None:
        """Take a video out of the queue and stop its running analysis ahead, waiting until it has stopped writing."""
        self.preanalysis_queue = [(v, s) for v, s in self.preanalysis_queue if v != video_id]
        job = self.preanalysis
        if job is not None and job.video_id == video_id:
            if job.finished is None:
                job.stop()
                await job.wait_finished()
            if self.preanalysis is job:
                self.preanalysis = None

    async def delete_video(self, video_id: str) -> dict[str, Any]:
        """Delete an upload: its file, calibration, analyses ahead and saved live results. A video that is open is
        closed first and one being analysed ahead is stopped; one a game folder uses is refused."""
        folder = self.video_dir(video_id)
        users = self.used_by(video_id)
        if users:
            raise HTTPException(409, f"The game folder {', '.join(users)} (runs/nba) uses this video file, so it is "
                                     "kept. Delete a copy of it instead, or point that game.json at another copy.")
        await self._end_preanalysis(video_id)
        if self.active and self.active.get("id") == video_id:
            async with self.lock:
                await self.stop()
                if self.task is not None:
                    with contextlib.suppress(Exception):
                        await self.task
                self.active, self.game, self.task = None, None, None
                await self.hub.reset()
        runs = [run for run, vid in self.saved_runs() if vid == video_id]
        freed = folder_bytes(folder) + sum(folder_bytes(r) for r in runs)
        await asyncio.to_thread(shutil.rmtree, folder)
        for run in runs:
            await asyncio.to_thread(shutil.rmtree, run, True)
        return {"deleted": video_id, "freed_bytes": freed}

    async def delete_replays(self, video_id: str) -> dict[str, Any]:
        """Delete a video's analyses ahead (stopping a running one); the video itself stays."""
        folder = self.video_dir(video_id) / "replays"
        await self._end_preanalysis(video_id)
        freed = folder_bytes(folder)
        if folder.is_dir():
            await asyncio.to_thread(shutil.rmtree, folder)
        return {"video_id": video_id, "freed_bytes": freed}

    async def delete_saved_runs(self, video_id: str | None = None) -> dict[str, Any]:
        """Delete the saved results of past live analyses: of one video, or all of them (``video_id`` None). The run
        a live analysis is still writing is kept."""
        running = self.active_run_id()
        runs = [run for run, vid in self.saved_runs() if run.name != running and (video_id is None or vid == video_id)]
        freed = sum(folder_bytes(r) for r in runs)
        for run in runs:
            await asyncio.to_thread(shutil.rmtree, run, True)
        return {"deleted_runs": len(runs), "freed_bytes": freed}

    def state(self) -> dict[str, Any]:
        game, run = self.game, self.analysis_run()
        return {"enabled": True, "active": self.active, "videos": self.videos(),
                "model": {"state": self.model_state, "error": self.model_error,
                          "name": self.model_name},
                "status": game.status() if game else None,
                "frame": self.hub.latest("frame"),
                "vision": getattr(game, "vision", None),
                "error": getattr(game, "failure", None),
                "capabilities": {"jev": bool(runtime_env().get("TYPESAFE_API_KEY")),
                                 "claude": anthropic_credential_source() is not None},
                "analysis": {"game": run[0].name, "tag": run[1], "version": run[2]} if run else None,
                "replay": self.replay_status(self.active["id"]) if self.active and not self.active.get("demo") else None}

    def analysis_run(self) -> tuple[Path, str, float] | None:
        """The open video's analysed run (``courtvision/nba/results.py``): None for the demo or a video no game folder
        analyses. Looked for again every ``ANALYSIS_RECHECK_S``, so a run finished meanwhile appears."""
        active = self.active
        if not active or active.get("demo"):
            return None
        vid, now = str(active["id"]), time.monotonic()
        if self._analysis is None or self._analysis[0] != vid or now - self._analysis[1] > ANALYSIS_RECHECK_S:
            try:
                run = game_results.find_run(self.root / vid / "source.video", self.nba_root, self.repo)
                if run is None:  # the possession analysis its analysis ahead built (nba/auto_analysis.py)
                    run = self.auto_analysis_run(vid)
            except OSError:
                log.exception("Could not look for this video's analysis")
                run = None
            self._analysis = (vid, now, run)
        return self._analysis[2]

    def auto_analysis_run(self, video_id: str) -> tuple[Path, str, float] | None:
        """(folder, tag, time) of the newest possession analysis built by an analysis ahead of ``video_id``."""
        from courtvision.nba.auto_analysis import TAG

        found = sorted(((p.stat().st_mtime, p) for p in (self.root / video_id / "replays").glob(
            f"*/possession/runs/{TAG}/timeline.json") if p.with_suffix(".jsonl").is_file()), reverse=True)
        if not found:
            return None
        mtime, doc = found[0]
        return doc.parents[2], TAG, mtime

    def analysis(self) -> dict[str, Any]:
        """The open video's possession timeline and decided events, read once per run. Without the game's ESPN
        summary the teams are the names given when the video was opened."""
        run = self.analysis_run()
        if run is None:
            raise HTTPException(404, "No possession analysis for this video. Add it as a game (tools/nba_new_game.py) "
                                     "and run the analysis through nba_analyze.py <game> timeline (docs/NBA.md).")
        if self._analysis_doc is None or self._analysis_doc[0] != run:
            self._analysis_doc = (run, game_results.load(run[0], run[1]))
        doc = self._analysis_doc[1]
        if doc["teams"] is None:
            names = self.game.config.game if self.game is not None else self.cfg.game
            doc = {**doc, "teams": {"home": {"abbr": names.home_name, "name": names.home_name},
                                    "away": {"abbr": names.away_name, "name": names.away_name}}}
        return doc

    def student_weights(self) -> Path | None:
        """The NBA broadcast student detector (training/train_student.py), when one is installed in the data folder's
        models (``models/nba-student.pt``); it replaces the COCO starter unless an RF-DETR checkpoint is configured."""
        p = self.root / "models" / "nba-student.pt"
        rf = self.cfg.perception.rfdetr
        return p if p.is_file() and not (rf.weights or rf.coreml_path) else None

    def new_detector(self) -> Any:
        """A fresh, warmed detector (blocking: call it in a thread): the NBA student when installed, else the
        configured RF-DETR checkpoint or the COCO starter (downloaded on first use)."""
        student = self.student_weights()
        if student is not None:
            try:
                from courtvision.perception.student import StudentConfig, StudentDetector

                det = StudentDetector(StudentConfig(weights=str(student)))
                det.detect(np.zeros((540, 960, 3), np.uint8), 0, 0.)
                det.reset()
                return det
            except Exception:
                log.exception("Could not load the NBA student detector; falling back to the COCO starter")
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

    async def prepare(self) -> None:
        self.model_state, self.model_error = "loading", None
        try:
            self.detector = await asyncio.to_thread(self.new_detector)
            self.model_state = "ready"
        except Exception:
            log.exception("Could not prepare starter detector")
            self.model_state = "error"
            self.model_error = "Detector setup failed. Check your connection and install courtvision[cv]. Then retry."

    def video_config(self, video_id: str, detector: Any, *, broadcast: bool = True, home_name: str = "Home",
                     away_name: str = "Away") -> AppConfig:
        """The settings an uploaded video is analysed with, live or ahead of time."""
        cfg = copy.deepcopy(self.cfg)
        cfg.narration.tts = "none"
        cfg.source.drop_late_s = None  # process every frame, report measured throughput honestly
        cfg.game.home_name, cfg.game.away_name = home_name.strip(), away_name.strip()
        cfg.source.kind = "video"
        cfg.source.path = str(self.root / video_id / "source.video")
        cfg.perception.calibration_path = str(self.root / video_id / "calibration.json")
        cfg.perception.scoreboard_roi_path = str(self.root / video_id / "scoreboard_roi.json")
        cfg.perception.save_calibration = True
        cfg.perception.broadcast = "on" if broadcast else "off"
        # The scorebug must be explicitly configured before it can gate live play. Without
        # one, court visibility gates cutaways; replay footage still needs operator review.
        cfg.perception.broadcast_config.live_requires_scorebug = False
        if type(detector).__name__ == "StudentDetector":
            # its person scores are centre-heatmap peaks (lower than a classifier's): ByteTrack thresholds to
            # match, and boxes re-drawn from the tracked camera (perception/broadcast.py)
            cfg.perception.pipeline.tracker = {"high_thresh": 0.3, "low_thresh": 0.1, "new_track_thresh": 0.35,
                                               **cfg.perception.pipeline.tracker}
            # the two team clusters refit as player crops accumulate: a first fit on a few frames can split on
            # something other than the teams and would hold all game (perception/teams.py TeamConfig.refit_at)
            cfg.perception.pipeline.team = {"refit_at": (150, 600, 1500, 3000), **cfg.perception.pipeline.team}
            cfg.perception.broadcast_config.rectify_boxes = True
            # Made or missed from the video: the scorebug's scores no longer confirm, veto or invent shots, and the
            # shot judge (models/shot-judge.json, tools/nba_live_judge.py) looks again at the rim after each call
            cfg.events.score_reconcile = False
            judge = self.shot_judge_model()
            if judge is not None and not cfg.pipeline.shot_judge:
                cfg.pipeline.shot_judge = str(judge)
        return cfg

    def shot_models_file(self) -> Path | None:
        """The shot models the possession analysis of a video analysed ahead decides its shots with
        (``models/shot-models.json``, tools/nba_shot_models.py), when installed."""
        p = self.root / "models" / "shot-models.json"
        return p if p.is_file() else None

    def shot_judge_model(self) -> Path | None:
        """The shot judge's model (``models/shot-judge.json`` in the data folder), when one is installed."""
        p = self.root / "models" / "shot-judge.json"
        return p if p.is_file() else None

    # ------------------------------------------------------------------ analysing ahead (server/preanalysis.py)

    def replay_dir(self, video_id: str) -> Path | None:
        """The video's newest pre-analysis (``<video>/replays/<run>``), finished or not."""
        runs = sorted((self.root / video_id / "replays").glob("*/status.json"))
        return runs[-1].parent if runs else None

    def replay_status(self, video_id: str) -> dict[str, Any] | None:
        job = self.preanalysis
        if job is not None and job.video_id == video_id:
            return job.status()
        if any(v == video_id for v, _ in self.preanalysis_queue):
            return {"video_id": video_id, "state": "queued"}
        out = self.replay_dir(video_id)
        status = preanalysis.read_status(out) if out is not None else None
        if status is not None and status.get("state") in ("queued", "loading", "running"):
            status["state"] = "stopped"  # the Studio closed while it ran
        return status

    def replay_info(self, video_id: str) -> dict[str, Any]:
        return self.replay_status(video_id) or {"video_id": video_id, "state": "none"}

    def preanalysis_ready(self) -> bool:
        """Whether a pre-analysis can load its detector without a download (the NBA student or a checkpoint on disk,
        or one injected), so an upload may start one unasked."""
        rf = self.cfg.perception.rfdetr
        return (self.detector_factory is not None or self.student_weights() is not None
                or bool(rf.weights or rf.coreml_path) or (self.root / "models" / "rf-detr-nano.pth").is_file())

    def enqueue_preanalysis(self, video_id: str, stride: int = 1) -> dict[str, Any]:
        if not (self.root / video_id / "video.json").is_file():
            raise HTTPException(404, "Video not found.")
        job = self.preanalysis
        if (job is not None and job.video_id == video_id and job.state in ("loading", "running")) or any(
                v == video_id for v, _ in self.preanalysis_queue):
            raise HTTPException(409, "This video is already being analysed ahead.")
        self.preanalysis_queue.append((video_id, stride))
        if self.preanalysis_task is None or self.preanalysis_task.done():
            self.preanalysis_task = asyncio.create_task(self._preanalyse(), name="studio-preanalysis")
        return self.replay_info(video_id)

    async def _preanalyse(self) -> None:
        while self.preanalysis_queue:
            video_id, stride = self.preanalysis_queue.pop(0)
            try:
                meta = json.loads((self.root / video_id / "video.json").read_text())
            except (OSError, ValueError):
                log.warning("Could not read video %s to analyse it ahead", video_id)
                continue

            def config_for(detector: Any, video_id: str = video_id) -> AppConfig:
                cfg = self.video_config(video_id, detector)
                cfg.perception.save_calibration = False  # never replace a calibration the operator saved
                cfg.jev.mode, cfg.narration.llm_enabled = "off", False
                cfg.pipeline.court_worker = True  # the court tracked ahead in a process of its own: the same fixes
                return cfg

            out = self.root / video_id / "replays" / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            # "every frame" is every frame of a 30 fps video: a 60 fps video is read at every 2nd frame
            stride *= preanalysis.rate_stride(meta.get("fps"))
            self.preanalysis = Preanalysis(video_id=video_id, video=self.root / video_id / "source.video", out=out,
                                           config_for=config_for, detector_factory=self.detector_factory or self.new_detector,
                                           duration_s=meta.get("duration_s"), stride=stride,
                                           shot_models=self.shot_models_file(),
                                           title=meta.get("title") or video_title(str(meta.get("name", ""))))
            await self.preanalysis.run()

    def stop_preanalysis(self, video_id: str) -> dict[str, Any]:
        queued = len(self.preanalysis_queue)
        self.preanalysis_queue = [(v, s) for v, s in self.preanalysis_queue if v != video_id]
        job = self.preanalysis
        if job is not None and job.video_id == video_id and job.state in ("loading", "running"):
            job.stop()
        elif len(self.preanalysis_queue) == queued:
            raise HTTPException(409, "This video is not being analysed ahead.")
        return self.replay_info(video_id)

    async def stop_preanalyses(self) -> None:
        self.preanalysis_queue.clear()
        if self.preanalysis is not None:
            self.preanalysis.stop()
        if self.preanalysis_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self.preanalysis_task

    async def stop(self) -> None:
        if self.game is not None and self.task is not None:
            from courtvision.cli import stop_game
            await stop_game(self.game, self.task)
        self.task = None

    async def start(self, body: StartIn) -> dict[str, Any]:
        async with self.lock:
            if body.demo:
                cfg = copy.deepcopy(self.cfg)
                cfg.narration.tts = "none"
                cfg.source.drop_late_s = None
                cfg.game.home_name, cfg.game.away_name = body.home_name.strip(), body.away_name.strip()
                cfg.source.kind = "sim"
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
                active["title"] = active.get("title") or video_title(str(active.get("name", "")))
                cfg = self.video_config(body.video_id, self.detector, broadcast=body.broadcast,
                                        home_name=body.home_name, away_name=body.away_name)
                step = preanalysis.rate_stride(active.get("fps"))  # a 60 fps video is analysed at 30 frames a second
                source = await asyncio.to_thread(preanalysis.StridedVideoSource, cfg.source.path, step, realtime=True) \
                    if step > 1 else await asyncio.to_thread(VideoFileSource, cfg.source.path, realtime=True)
            await self.stop()
            if not body.demo and type(self.detector).__name__ == "StudentDetector":
                self.detector.reset()
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
                out = self.root / "analyses" / active["run_id"]
                await asyncio.to_thread(write_run_dir, result, out)
                # which video it was of, so the library can count and delete it with the video
                (out / "source.json").write_text(json.dumps({"video_id": None if active.get("demo") else active.get("id"),
                                                             "title": active.get("title") or active.get("name")}))
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
                      auth_token: str | None = None, detector: Any = None, lab_runs: Path | None = None,
                      detector_factory: Callable[[], Any] | None = None, auto_preanalyze: bool = True) -> Any:
    lab = Lab(lab_runs)
    workspace = Workspace(config or load_config(), Path(root), detector, nba_root=lab.runs / "nba", repo=lab.repo,
                          detector_factory=detector_factory, auto_preanalyze=auto_preanalyze)
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
                await workspace.stop_preanalyses()
                await workspace.stop()
                if workspace.model_task is not None:
                    # A model load runs in a thread: wait for it before releasing the process.
                    await workspace.model_task
    app.router.lifespan_context = lifespan
    read, write = Depends(app.state.require_read), Depends(app.state.require_write)
    app.state.asker.studio_state = workspace.state
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
                same = await asyncio.to_thread(workspace.find_copy, size, game_results.fingerprint(path)[1])
                if same is not None:  # this video is here already: keep one copy
                    await asyncio.to_thread(shutil.rmtree, directory)
                    return {**same, "duplicate": True}
                filename = Path(name.replace("\\", "/")).name[:160] or "Game video"
                record = {"id": video_id, "name": filename, "title": video_title(filename),
                          "bytes": size, "created": time.time(), **metadata}
                (directory / "video.json").write_text(json.dumps(record))
                if workspace.auto_preanalyze and workspace.preanalysis_ready():
                    # analysed ahead in the background, so it can be replayed at full speed (server/preanalysis.py)
                    workspace.enqueue_preanalysis(video_id)
                return {**record, "replay": workspace.replay_status(video_id)}
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

    @app.get("/api/studio/library", dependencies=[read])
    async def library() -> Any:
        return await asyncio.to_thread(workspace.library)

    @app.post("/api/studio/find", dependencies=[read])
    async def find(body: FindIn) -> Any:
        # a video the browser is about to upload that is here already (by size and first and last MiB)
        return {"video": await asyncio.to_thread(workspace.find_copy, body.bytes, body.digest)}

    @app.post("/api/studio/video/{video_id}/rename", dependencies=[write])
    async def rename(video_id: str, body: RenameIn) -> Any:
        return await asyncio.to_thread(workspace.rename, video_id, body.title)

    @app.delete("/api/studio/video/{video_id}", dependencies=[write])
    async def delete_video(video_id: str) -> Any:
        return await workspace.delete_video(video_id)

    @app.delete("/api/studio/replay/{video_id}", dependencies=[write])
    async def delete_replays(video_id: str) -> Any:
        return await workspace.delete_replays(video_id)

    @app.delete("/api/studio/analyses", dependencies=[write])
    async def delete_saved_runs(video_id: str | None = Query(None, pattern=r"^[a-f0-9]{32}$")) -> Any:
        return await workspace.delete_saved_runs(video_id)

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

    @app.get("/api/studio/analysis", dependencies=[read])
    async def analysis() -> Any:
        return await asyncio.to_thread(workspace.analysis)

    @app.get("/api/studio/video/{video_id}", dependencies=[Depends(app.state.require_media)])
    async def video(video_id: str) -> Response:
        # The footage itself, for watching an analysed game at full speed in the browser (static/possession.js);
        # ranges are served, so it seeks.
        meta = workspace.root / video_id / "video.json"
        if not re.fullmatch(r"[a-f0-9]{32}", video_id) or not meta.is_file():
            raise HTTPException(404, "Video not found.")
        name = str(json.loads(meta.read_text()).get("name", ""))
        return FileResponse(workspace.root / video_id / "source.video",
                            media_type=VIDEO_TYPES.get(Path(name).suffix.lower(), "video/mp4"),
                            headers={"Cache-Control": "private, max-age=86400"})

    def replay_out(video_id: str) -> Path:
        out = workspace.replay_dir(video_id) if re.fullmatch(r"[a-f0-9]{32}", video_id) else None
        if out is None:
            raise HTTPException(404, "This video has not been analysed ahead.")
        return out

    @app.get("/api/studio/replay/{video_id}", dependencies=[read])
    async def replay_status(video_id: str) -> Any:
        if not re.fullmatch(r"[a-f0-9]{32}", video_id):
            raise HTTPException(404, "Video not found.")
        return workspace.replay_info(video_id)

    @app.post("/api/studio/replay/{video_id}/start", dependencies=[write])
    async def replay_start(video_id: str, body: PreanalyseIn) -> Any:
        if not re.fullmatch(r"[a-f0-9]{32}", video_id):
            raise HTTPException(404, "Video not found.")
        return workspace.enqueue_preanalysis(video_id, body.stride)

    @app.post("/api/studio/replay/{video_id}/stop", dependencies=[write])
    async def replay_stop(video_id: str) -> Any:
        if not re.fullmatch(r"[a-f0-9]{32}", video_id):
            raise HTTPException(404, "Video not found.")
        return workspace.stop_preanalysis(video_id)

    @app.get("/api/studio/replay/{video_id}/chunk/{k}", dependencies=[read])
    async def replay_chunk(video_id: str, k: int) -> Response:
        # one minute of the recording (server/preanalysis.py), as written so far
        path = preanalysis.chunk_path(replay_out(video_id), k)
        if not 0 <= k < 100_000 or not path.is_file():
            raise HTTPException(404, "Not analysed this far yet.")
        return Response(await asyncio.to_thread(path.read_bytes), media_type="application/x-ndjson",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/studio/replay/{video_id}/state", dependencies=[read])
    async def replay_view(video_id: str, t: float = Query(..., ge=0, le=1e6)) -> Any:
        return await asyncio.to_thread(preanalysis.replay_state, replay_out(video_id), t)

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
