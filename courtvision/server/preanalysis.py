"""Analysing a whole video ahead of time, so the Studio can replay its live analysis at full speed.

The live view analyses as it plays, at the pipeline's speed (about 20 frames a second for the NBA detector on an
M3 Pro, two thirds of real time). A pre-analysis runs the same pipeline (:class:`StudioGame`, the same settings) over
the whole video unpaced, in the background and with no cloud calls, and records what the live view shows:

* each analysed frame's boxes (the people and the ball the feed draws), and
* every hub message (the game's meta, court-lens frames, events, stats, vision, narration, insights) with the video
  time it was published at. Stats snapshots are kept every ``STATS_EVERY_S`` and at each score change, without their
  per-track running totals (``tracking``, most of each snapshot, which the browser does not show).

The recording is cut into minutes (``chunks/NNNN.jsonl``, one JSON object per line: ``{"vt", "b"}`` for a frame's
boxes, ``{"vt", "m"}`` for a message) so the browser fetches the part it plays; the sparse messages (events,
narration, insights, captions) are also kept together (``sparse.jsonl``) so the view at any moment can be rebuilt for
a seek (:func:`replay_state`). ``status.json`` says how far it has got. The Studio plays the recording itself and
replays these at the video's time (``static/replay.js``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from courtvision.config import AppConfig
from courtvision.pipeline import VideoFileSource, _collect, write_run_dir
from courtvision.server.hub import Hub
from courtvision.server.studio_game import StudioGame

log = logging.getLogger(__name__)
CHUNK_S = 60.0  # seconds of video per recording file
SPARSE = frozenset({"event", "utterance", "insight", "caption"})
SKIP = frozenset({"status", "pong", "reset"})  # the pipeline's own state, not the game's
FLUSH_EVERY = 30  # frames between flushes, so a replay can read a running analysis
STATS_EVERY_S = 2.0  # a stats snapshot is kept this often, and whenever the score changes


def chunk_path(out: Path, k: int) -> Path:
    return out / "chunks" / f"{k:04d}.jsonl"


class Recorder:
    """Writes a pre-analysis (see the module docstring). Thread-safe: perception and the hub both write."""

    def __init__(self, out: Path) -> None:
        self.out = out
        (out / "chunks").mkdir(parents=True, exist_ok=True)
        self.t = 0.0  # video time of the frame being analysed: messages published meanwhile carry it
        self.first_t: float | None = None
        self.frames = 0
        self._lock = threading.Lock()
        self._chunk = -1
        self._fh: Any = None
        self._sparse = (out / "sparse.jsonl").open("a", encoding="utf-8")
        self._stats: tuple[float, str] | None = None  # (video time, score) of the last stats snapshot kept

    def _line(self, t: float, line: str) -> None:
        k = int(max(0.0, t) // CHUNK_S)
        if k != self._chunk:
            if self._fh is not None:
                self._fh.close()
            self._chunk, self._fh = k, chunk_path(self.out, k).open("a", encoding="utf-8")
        self._fh.write(line + "\n")

    def frame(self, t: float, boxes: Sequence[Mapping[str, Any]]) -> None:
        compact = [[round(float(v)) for v in b["box"]] + [b.get("label"), b.get("team"),
                                                           None if b.get("confidence") is None
                                                           else round(float(b["confidence"]), 2)] for b in boxes]
        with self._lock:
            self.t = float(t)
            if self.first_t is None:
                self.first_t = self.t
            self.frames += 1
            self._line(self.t, json.dumps({"vt": round(self.t, 3), "b": compact}, separators=(",", ":")))
            if self.frames % FLUSH_EVERY == 0:
                self.flush()

    def message(self, mtype: str, text: str) -> None:
        if mtype in SKIP:
            return
        if mtype == "stats":
            msg = json.loads(text)
            stats = msg.get("stats") or {}
            stats.pop("tracking", None)
            score = json.dumps(stats.get("score"), sort_keys=True)
            last = self._stats
            if last is not None and last[1] == score and self.t - last[0] < STATS_EVERY_S:
                return
            self._stats = (self.t, score)
            text = json.dumps(msg, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            line = f'{{"vt":{round(self.t, 3)},"m":{text}}}'
            self._line(self.t, line)
            if mtype in SPARSE:
                self._sparse.write(line + "\n")
            elif mtype == "meta":
                (self.out / "meta_msg.json").write_text(text, encoding="utf-8")

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()
        self._sparse.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            self._sparse.close()


class RecordingHub(Hub):
    """A hub that also hands every message it takes to a :class:`Recorder` (no browser subscribes to it)."""

    def __init__(self, config: Any, recorder: Recorder) -> None:
        super().__init__(config)
        self.recorder = recorder

    def _ingest(self, p: Any, fanout: bool = True) -> None:  # every published message passes here, in order
        self.recorder.message(p.mtype, p.text)
        super()._ingest(p, fanout=fanout)


MAX_FAILED_FRAMES = 90  # consecutive frames that fail before an analysis ahead gives up (3 s at 30 fps)


class PreanalysisGame(StudioGame):
    """The Studio's game, recording each analysed frame's boxes. Unpaced, the live pipeline raises a frame's error
    (``LiveGame.process``); here one bad frame is logged with its traceback, counted and passed over, so a rare error
    does not end the analysis of a whole game, and only ``MAX_FAILED_FRAMES`` failures in a row stop it."""

    def __init__(self, *args: Any, recorder: Recorder, possession: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.recorder = recorder
        self.possession = possession  # nba/auto_analysis.Recorder: the camera and ball of each frame, or None
        self.failed_frames = 0
        self.first_error: str | None = None
        self._failed_in_a_row = 0

    async def process(self, item: Any) -> None:
        try:
            await super().process(item)
            self._failed_in_a_row = 0
        except Exception as exc:
            self.failed_frames += 1
            self._failed_in_a_row += 1
            if self.first_error is None:
                self.first_error = f"frame {item.frame_idx} at {item.t:.2f} s: {type(exc).__name__}: {exc}"
            log.exception("Analysis ahead: frame %d (%.2f s) failed; continuing", item.frame_idx, item.t)
            if self._failed_in_a_row >= MAX_FAILED_FRAMES:
                self.failure = f"The analysis failed on {self._failed_in_a_row} frames in a row ({self.first_error})."
                self.request_stop(drain=False)

    def _perceive(self, image: Any, frame_idx: int, t: float) -> Any:
        result = super()._perceive(image, frame_idx, t)
        if self.visual is not None:
            self.recorder.frame(t, self.visual[1])
        if self.possession is not None:
            self._record_court(frame_idx, t, result)
        return result

    def _record_court(self, frame_idx: int, t: float, result: Any) -> None:
        """The frame's camera (while the broadcast court tracker has one) and ball candidates, for the possession
        analysis built when the analysis ends (``nba/auto_analysis.py``)."""
        pipe = getattr(self, "perception", None)
        auto = getattr(pipe, "auto", None)
        if auto is None or result is None:
            return
        gate = getattr(pipe, "last_gate", None)
        params = auto.params if gate is not None and gate.live else None
        status = (getattr(auto, "status", None) or "tracking") if params is not None else "lost"
        self.possession.frame(frame_idx, t, auto.center, pipe.base.image_size, self.source.fps,
                              getattr(self.source, "stride", 1), status, params, result[0].balls)


def rate_stride(fps: float | None, base_fps: float = 29.97) -> int:
    """Frames of a ``fps`` video per analysed frame so that about ``base_fps`` frames a second are analysed: 1 up to
    44 fps, 2 at 50 / 59.94 / 60 fps, 4 at 119.88 (1 for an unknown rate). The detector, the tracker and the event
    engine are set up for 29.97 fps; a 60 fps video read at every frame costs twice as much and gives the same
    analysis."""
    if not fps or not np.isfinite(fps) or fps <= 0:
        return 1
    return max(1, int(np.floor(fps / base_fps + 0.5)))


class StridedVideoSource(VideoFileSource):
    """A video read every ``stride``-th frame (the others are skipped without decoding)."""

    def __init__(self, path: str | Path, stride: int = 1, realtime: bool = False, **kwargs: Any) -> None:
        super().__init__(path, realtime=realtime, **kwargs)
        self.stride = max(1, int(stride))
        self.fps = self.fps / self.stride

    def _read(self) -> tuple[bool, Any, float]:
        with self._lock:
            for _ in range(self.stride - 1):
                if not self._cap.grab():
                    return False, None, 0.0
            ok, img = self._cap.read()
            pos = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        return bool(ok), img if ok else None, pos


class Preanalysis:
    """One background analysis of a whole video (see the module docstring)."""

    def __init__(self, *, video_id: str, video: Path, out: Path, config_for: Callable[[Any], AppConfig],
                 detector_factory: Callable[[], Any], duration_s: float | None = None, stride: int = 1,
                 shot_models: Path | None = None, title: str | None = None) -> None:
        self.video_id, self.video, self.out, self.config_for = video_id, video, out, config_for
        self.detector_factory, self.duration_s, self.stride = detector_factory, duration_s, max(1, int(stride))
        self.shot_models, self.title = shot_models, title  # the possession analysis's shot models (auto_analysis)
        self.state = "queued"
        self.error: str | None = None
        self.started: float | None = None  # wall time it started analysing frames
        self.finished: float | None = None
        self.game: PreanalysisGame | None = None
        self.recorder: Recorder | None = None
        self.analysed_fps: float | None = None  # frames analysed per video second (the video's rate / stride)
        self._stop = False
        self._finished = asyncio.Event()

    def status(self) -> dict[str, Any]:
        rec, now = self.recorder, time.time()
        frames = rec.frames if rec is not None else 0
        t_done = rec.t if rec is not None and frames else 0.0
        spent = ((self.finished or now) - self.started) if self.started else 0.0
        video_s = t_done - (rec.first_t or 0.0) if rec is not None and frames else 0.0
        speed = video_s / spent if spent > 0 and video_s > 0 else None  # video seconds per wall second
        left = max(0.0, (self.duration_s or 0.0) - t_done)
        return {"run": self.out.name, "video_id": self.video_id, "state": self.state, "stride": self.stride,
                "fps": None if self.analysed_fps is None else round(self.analysed_fps, 2),
                "frames": frames, "t_done": round(t_done, 3), "duration_s": self.duration_s,
                "speed": None if speed is None else round(speed, 4),
                "eta_s": round(left / speed) if speed and self.state == "running" else None,
                "started": self.started, "finished": self.finished, "error": self.error,
                "failed_frames": self.game.failed_frames if self.game is not None else 0,
                "first_error": self.game.first_error if self.game is not None else None}

    def write_status(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        tmp = self.out / "status.json.tmp"
        tmp.write_text(json.dumps(self.status()))
        tmp.replace(self.out / "status.json")

    def stop(self) -> None:
        self._stop = True
        if self.game is not None:
            self.game.request_stop(drain=False)

    async def wait_finished(self) -> None:
        """Wait until the worker has closed its recording and written its final status."""
        await self._finished.wait()

    async def run(self) -> None:
        self.state = "loading"
        try:
            self.write_status()
            detector = await asyncio.to_thread(self.detector_factory)
            if self._stop:
                self.state = "stopped"
                return
            cfg = self.config_for(detector)  # the live view's settings for this video and detector
            source = await asyncio.to_thread(StridedVideoSource, self.video, self.stride)
            self.analysed_fps = source.fps
            self.recorder = Recorder(self.out)
            hub = RecordingHub(cfg.hub, self.recorder)
            # No keys: Jev, Claude and spoken narration stay off however the Studio is set up.
            possession = None
            if cfg.perception.broadcast == "on":  # the court is tracked: the possession analysis can be built after
                from courtvision.nba.auto_analysis import Recorder as CourtRecorder

                possession = CourtRecorder(self.out / "possession")
            self.game = PreanalysisGame(cfg, source, hub, realtime=False, detector=detector, env={},
                                        recorder=self.recorder, possession=possession)
            self.state, self.started = "running", time.time()
            self.write_status()
            ticker = asyncio.create_task(self._tick(), name="preanalysis-status")
            try:
                await self.game.run()
            finally:
                ticker.cancel()
            self.error = self.game.failure
            self.state = "stopped" if self._stop else "failed" if self.error else "done"
            try:
                await asyncio.to_thread(write_run_dir, _collect(self.game), self.out / "analysis")
            except Exception:
                log.exception("Could not write the pre-analysis's events")
            await self._possession()
        except asyncio.CancelledError:
            self.state = "stopped"
            raise
        except Exception:
            log.exception("Pre-analysis of %s failed", self.video_id)
            self.state, self.error = "failed", "The analysis stopped with an error. See the Studio's log."
        finally:
            try:
                if self.recorder is not None:
                    self.recorder.close()
            finally:
                self.finished = time.time()
                try:
                    self.write_status()
                finally:
                    self._finished.set()

    async def _possession(self) -> None:
        """The possession analysis from the recorded camera, ball and events (``nba/auto_analysis.py``), for Watch."""
        game = self.game
        rec = game.possession if game is not None else None
        if rec is None:
            return
        rec.close()
        if not rec.usable or self._stop:
            return
        from courtvision.nba import auto_analysis

        try:  # the replay is ready meanwhile; Watch finds the analysis when it is written
            await asyncio.to_thread(auto_analysis.build, rec.folder, game.engine.events, rec.balls, str(self.video),
                                    self.shot_models, self.title)
        except Exception:
            log.exception("Could not build the possession analysis of %s", self.video_id)

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(2.0)
            if self.recorder is not None:
                with self.recorder._lock:
                    self.recorder.flush()
            self.write_status()


# --------------------------------------------------------------------------- reading a recording


def _records(path: Path) -> list[dict[str, Any]]:
    out = []
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:  # the line being written by a running analysis
                    break
    return out


def replay_state(out: Path, t: float) -> dict[str, Any]:
    """What the live view showed at video time ``t``: the game's meta, the newest revision of every event published
    by then, the narration, insights and captions so far, and the last court-lens frame, stats and vision message.
    The browser applies these as if they had arrived live, then plays on from ``t``."""
    meta_path = out / "meta_msg.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None
    events: dict[str, dict[str, Any]] = {}
    insights: dict[str, dict[str, Any]] = {}
    utterances: list[dict[str, Any]] = []
    captions: list[dict[str, Any]] = []
    for rec in _records(out / "sparse.jsonl"):
        if rec["vt"] > t:
            break
        m = rec["m"]
        if m["type"] == "event":
            eid = str(m["event"]["event_id"])
            old = events.get(eid)
            if old is None or int(old["event"].get("revision") or 0) <= int(m["event"].get("revision") or 0):
                events[eid] = m
        elif m["type"] == "insight":
            insights[str((m.get("insight") or {}).get("insight_id") or len(insights))] = m
        elif m["type"] == "utterance":
            utterances.append(m)
        elif m["type"] == "caption":
            captions.append(m)
    latest: dict[str, dict[str, Any] | None] = {"frame": None, "stats": None, "vision": None}
    k = int(max(0.0, t) // CHUNK_S)
    for kk in (k, k - 1):  # stats come each second and frames 15 times a second: the last minute has them
        if kk < 0 or all(v is not None for v in latest.values()):
            break
        found: dict[str, dict[str, Any]] = {}
        for rec in _records(chunk_path(out, kk)):
            if rec["vt"] > t:
                break
            m = rec.get("m")
            if m is not None and m["type"] in latest:
                found[m["type"]] = m
        for mtype, m in found.items():
            if latest[mtype] is None:
                latest[mtype] = m
    return {"t": t, "meta": meta, "events": list(events.values()), "insights": list(insights.values()),
            "utterances": utterances[-60:], "captions": captions[-3:], **latest}


def read_status(out: Path) -> dict[str, Any] | None:
    try:
        return json.loads((out / "status.json").read_text())
    except (OSError, ValueError):
        return None
