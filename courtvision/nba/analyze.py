"""Offline analysis of an NBA broadcast: camera track, detections, ball path and 3-D flights, players, events.

The game is processed in passes whose outputs are cached under the game folder, so the later passes (ball,
players, events) can be re-run quickly while they are improved:

1. ``tools/nba_scan_scorebug.py``: scorebug reads (period, game clock, scores, shot clock) at 3 Hz.
2. ``tools/nba_track_court.py``: the game camera per frame (``CameraTrack``).
3. ``detect_video``: the student detector (``perception/student.py``) on every frame with the court in view:
   ball candidates and person boxes (``DetectionCache``).
4. ``link_ball``: one ball path through the candidates (a dynamic programme like ``training/ball_labels.py``,
   with the camera's turn between frames removed from the motion cost), then ``ball3d.find_flights`` for the
   ballistic parts of the path (passes, shots) in 3-D.
5. ``perceive``: ``FrameState`` per frame: players (ByteTrack on the cached boxes with camera-motion
   compensation, team colours, referees left out, court positions through the frame's camera), the ball (image
   point; floor point and height inside fitted flights) and the rims projected through the camera. Frames
   without the game camera (replays, close-ups) carry only the scoreboard.
6. ``run_engine``: the event engine (``events/engine.py``) on those states.

**Other frame rates and sizes.** The settings were tuned on one ESPN game at 29.97 fps and 1920x1080
(``REF_FPS``, ``REF_HEIGHT``). Settings counted in frames are frames at ``REF_FPS`` and are converted to the camera
track's rate (``rate_ratio``: the same time; exactly the tuned numbers at 29.97 / 30 fps); pixel settings are px of a
1080-line frame, scaled with the frame height. Time is ``CameraTrack.time``: frame / fps, or the decoder's timestamps
when the tracker found a variable frame rate. ``probe_video`` checks a file before frame / fps is trusted. Limits:
a variable-frame-rate file is read without seeking (OpenCV's seeks land on the wrong frame there), which is slow:
converting it to a constant rate first (``ffmpeg -i in.mp4 -fps_mode cfr -r 30000/1001 out.mp4``) is better; an
interlaced source shows combing on a fast ball (deinterlace first, e.g. ``-vf bwdif=mode=send_field`` at twice the
rate); a letterboxed or pillarboxed picture scales the pixel settings by the frame, not the picture, and moves the
scorebug (crop first, e.g. ``-vf crop=W:H:X:Y``); non-square pixels break the camera model (``-vf scale=iw*sar:ih``).

Research use only (NBA footage and statistics; docs/TRAINING.md §1).
"""

from __future__ import annotations

import json
import logging
import math
import queue
import shutil
import subprocess
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeVar

import cv2
import numpy as np

from courtvision.geometry.court import CourtSpec, court_for
from courtvision.geometry.homography import Calibration, Homography
from courtvision.perception.ball3d import Flight3D, FlightSearchConfig, find_flights
from courtvision.perception.court_search import CameraParams
from courtvision.types import (
    BallState,
    BBox,
    Detection,
    FrameDetections,
    FrameState,
    ObjClass,
    ScoreboardReading,
)

__all__ = ["REF_FPS", "REF_HEIGHT", "BallTrack", "CameraTrack", "DetectionCache", "VideoInfo", "detect_video",
           "link_ball", "load_scorebug", "prefetch", "probe_video", "rate_ratio", "scale_frames"]

log = logging.getLogger(__name__)
T = TypeVar("T")


# --------------------------------------------------------------------------- frame rate and frame size

REF_FPS = 29.97  # the rate the frame-count settings were tuned at (ESPN, Lakers @ Warriors 2024-12-25)
REF_HEIGHT = 1080  # the pixel settings are px of a 1080-line frame


def rate_ratio(fps: float) -> float:
    """Frames of a ``fps`` video per frame of the tuning footage (``fps / REF_FPS``). Exactly 1.0 within 0.5 % of
    ``REF_FPS`` (29.97 and 30 fps, NTSC 30000/1001 and integer, are one rate here, so the tuned settings apply
    unchanged) and for an unknown rate (0 or not finite)."""
    if not (isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0):
        return 1.0
    r = fps / REF_FPS
    return 1.0 if abs(r - 1.0) <= 0.005 else r


def scale_frames(n: int, ratio: float, minimum: int = 1) -> int:
    """A frame count of the tuning footage -> about the same time in frames of a video ``ratio`` times its rate
    (``n`` itself at ratio 1)."""
    if ratio == 1.0:
        return n
    return max(minimum, math.floor(n * ratio + 0.5))


def _scale_odd(n: int, ratio: float) -> int:
    """``scale_frames`` for a centred window: the odd count nearest ``n * ratio`` (``n`` itself at ratio 1)."""
    if ratio == 1.0:
        return n
    return max(1, 2 * math.floor(n * ratio / 2.0) + 1)


@dataclass
class VideoInfo:
    """What a video file says about its timing and picture (``probe_video``)."""

    path: str
    fps: float  # OpenCV's rate (the container's mean rate on a variable-rate file)
    n_frames: int  # OpenCV's frame count
    size: tuple[int, int]
    duration_s: float | None = None  # ffprobe
    times: np.ndarray | None = None  # each frame's presentation time (s from the first; ffprobe's packets)
    vfr: bool = False  # frame / fps strays more than half a frame from ``times``
    interlaced: bool = False
    warnings: list[str] = field(default_factory=list)

    def time(self, frame: int) -> float:
        if self.vfr and self.times is not None and 0 <= frame < len(self.times):
            return float(self.times[frame])
        return frame / self.fps

    def frame_at(self, t: float) -> int:
        """The frame shown at video time ``t`` (the nearest; frame / fps unless the rate is variable)."""
        if self.vfr and self.times is not None and len(self.times):
            j = int(np.searchsorted(self.times, t))
            if j > 0 and (j == len(self.times) or t - self.times[j - 1] <= self.times[j] - t):
                j -= 1
            return j
        return round(t * self.fps)


def _ffprobe(path: str, entries: str, csv: bool = False) -> str | None:
    exe = shutil.which("ffprobe")
    if exe is None:
        return None
    cmd = [exe, "-v", "error", "-select_streams", "v:0", "-show_entries", entries, "-of", "csv=p=0" if csv else "json",
           path]
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _rate(text: Any) -> float | None:
    try:
        a, _, b = str(text).partition("/")
        v = float(a) / float(b or 1)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def black_bars(frames: list[np.ndarray], level: int = 24) -> tuple[int, int, int, int]:
    """Rows (top, bottom) and columns (left, right) at the frame edges that are black in every frame (all channels
    at most ``level``): a letterboxed or pillarboxed picture."""
    if not frames:
        return 0, 0, 0, 0
    peak = np.max(np.stack([f.max(axis=2) for f in frames]), axis=0)
    rows, cols = peak.max(axis=1) <= level, peak.max(axis=0) <= level

    def run(flags: np.ndarray) -> int:
        return int(np.argmin(flags)) if not flags.all() else len(flags)

    return run(rows), run(rows[::-1]), run(cols), run(cols[::-1])


def probe_video(path: Path | str, timestamps: bool = True, samples: int = 5) -> VideoInfo:
    """Rate, frame count and size of a video (OpenCV), checked against ffprobe when it is installed: the stream's
    duration, its field order, its pixel aspect and (``timestamps``) every frame's presentation time; ``samples``
    frames spread over the video are checked for black bars. ``warnings`` says what would make frame / fps, the frame
    numbers or the pixel settings wrong (see the module docstring for what to do)."""
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[np.ndarray] = []
    for k in range(samples if n > 0 else 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * n / samples))
        ok, img = cap.read()
        if ok:
            frames.append(img)
    cap.release()
    info = VideoInfo(path, fps, n, (w, h))
    warn = info.warnings
    if not (math.isfinite(fps) and 0 < fps <= 1000):
        warn.append(f"no usable frame rate ({fps}): times would be wrong")
    raw = _ffprobe(path, "stream=avg_frame_rate,r_frame_rate,duration,field_order,sample_aspect_ratio:format=duration")
    if raw:
        doc = json.loads(raw)
        st = (doc.get("streams") or [{}])[0]
        for d in (st.get("duration"), (doc.get("format") or {}).get("duration")):
            if info.duration_s is None and _rate(d) is not None:
                info.duration_s = _rate(d)
        if st.get("field_order") in ("tt", "bb", "tb", "bt"):
            info.interlaced = True
            warn.append(f"interlaced ({st['field_order']}): each frame weaves two fields {500 / max(fps, 1e-9):.1f} ms "
                        "apart, so a fast ball shows combing; deinterlace first (ffmpeg -vf bwdif=mode=send_field "
                        "doubles the rate)")
        sar = st.get("sample_aspect_ratio")
        if sar and sar not in ("1:1", "0:1", "N/A"):
            warn.append(f"non-square pixels (sample aspect {sar}): the camera model needs square pixels "
                        "(ffmpeg -vf scale=iw*sar:ih,setsar=1)")
    if info.duration_s and fps > 0 and n > 0:
        expect = info.duration_s * fps
        if abs(n - expect) > max(2.0, 0.005 * expect):
            warn.append(f"frame count {n} and rate {fps:.3f} fps disagree with the duration {info.duration_s:.2f} s "
                        f"({expect:.0f} frames): frame / fps is not the video time")
    if timestamps:
        pts = _ffprobe(path, "packet=pts_time,flags", csv=True)
        if pts:
            vals = []
            for line in pts.splitlines():
                fields = line.split(",")
                if len(fields) > 1 and "D" in fields[1]:
                    continue  # pre-roll before an edit list's start: never shown (a stream-copied trim)
                try:
                    vals.append(float(fields[0]))
                except ValueError:
                    continue  # N/A (no timestamp) or an empty line
            ts = np.sort(np.array(vals, float))
            if len(ts):
                info.times = ts - ts[0]
                if n > 0 and abs(len(ts) - n) > 1:
                    warn.append(f"OpenCV counts {n} frames, the stream has {len(ts)}")
                if fps > 0 and len(ts) > 2:
                    drift = float(np.max(np.abs(info.times - np.arange(len(ts)) / fps)))
                    if drift > 0.5 / fps:
                        info.vfr = True
                        d = np.diff(info.times)
                        warn.append(f"variable frame rate (frame intervals {1000 * d.min():.1f}-{1000 * d.max():.1f} "
                                    f"ms; frame / fps strays up to {drift:.2f} s from the frame times): OpenCV's seeks "
                                    "land on the wrong frames, so the tools read such a file from its start; "
                                    "converting it to a constant rate is faster (ffmpeg -fps_mode cfr -r <rate>)")
    if frames and all(float(f.std()) < 2.0 for f in frames):
        warn.append(f"the {len(frames)} sampled frames decode as flat images (OpenCV cannot decode this file: "
                    "seen with interlaced H.264 and MPEG-2 in OpenCV 5.0.0); re-encode it with ffmpeg first (e.g. "
                    "-vf bwdif -c:v libx264)")
        return info
    top, bottom, left, right = black_bars(frames)
    if h and w and (top + bottom >= 0.02 * h or left + right >= 0.02 * w):
        warn.append(f"black bars (top {top}, bottom {bottom}, left {left}, right {right} px): the pixel settings "
                    "scale with the frame, not the picture, and the scorebug is not where its layout expects; crop "
                    f"first (ffmpeg -vf crop={w - left - right}:{h - top - bottom}:{left}:{top})")
    return info


# --------------------------------------------------------------------------- camera


@dataclass
class CameraTrack:
    """The game camera per frame (full-resolution pixels): fixed centre, per-frame pan / tilt / roll / focal.
    ``times``: each tracked frame's decoder timestamp (s) when the video's frame rate is variable (the tracker's
    header says ``"vfr"``); None: a frame's time is frame / fps."""

    center: np.ndarray
    image_size: tuple[int, int]
    fps: float
    params: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)  # frame -> pan, tilt, roll, f
    times: dict[int, float] | None = None

    # frames (at REF_FPS) of the smoothing window and of the longest gap filled (``_smooth_runs``)
    SMOOTH_FRAMES = 5
    MAX_FILL_FRAMES = 3

    @classmethod
    def load(cls, paths: list[Path], smooth: int | None = None, max_fill: int | None = None) -> CameraTrack:
        """The parts of ``tools/nba_track_court.py`` (each starts with a header: centre, frame size, rate). The fits
        are smoothed by ``_smooth_runs`` over ``smooth`` frames after gaps of up to ``max_fill`` frames are filled; by
        default ``SMOOTH_FRAMES`` and ``MAX_FILL_FRAMES`` at REF_FPS (0.17 s, 0.1 s) in the track's own frames."""
        center, size, fps, vfr = None, None, None, False
        raw: dict[int, tuple[float, float, float, float]] = {}
        stamps: dict[int, float] = {}
        for p in sorted(paths):
            for line in p.open():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # a line still being written by the tracker
                if r.get("header"):
                    center, size, fps = np.array(r["center"], float), tuple(r["image_size"]), float(r["fps"])
                    vfr = vfr or bool(r.get("vfr"))
                    continue
                if vfr and r.get("t") is not None and r.get("frame") is not None:  # (the header comes first)
                    stamps[int(r["frame"])] = float(r["t"])
                q = r.get("params")
                if q is not None and r.get("status") != "held":
                    raw[int(r["frame"])] = (q["pan"], q["tilt"], q["roll"], q["f"])
        if center is None or size is None or fps is None:
            raise ValueError("no camera header")
        ratio = rate_ratio(fps)
        track = cls(center, (int(size[0]), int(size[1])), fps, times=stamps if vfr and stamps else None)
        track.params = _smooth_runs(raw, _scale_odd(cls.SMOOTH_FRAMES, ratio) if smooth is None else smooth,
                                    scale_frames(cls.MAX_FILL_FRAMES, ratio) if max_fill is None else max_fill)
        return track

    @property
    def vfr(self) -> bool:
        """The video's frame rate is variable: frame / fps is not its time, and a seek lands on the wrong frame."""
        return self.times is not None

    def _table(self) -> tuple[np.ndarray, np.ndarray]:
        """The tracked frames and their times as sorted arrays (rebuilt when ``times`` changes)."""
        times = self.times or {}
        key = (id(self.times), len(times))
        if getattr(self, "_tab_key", None) != key:
            fr = np.array(sorted(times), np.int64)
            self._tab = (fr, np.array([times[int(f)] for f in fr], float))
            self._tab_key = key
        return self._tab

    def time(self, frame: int) -> float:
        """Video time (s) of ``frame``: frame / fps; on a variable-rate video its decoder timestamp (interpolated
        between tracked frames, at ``fps`` beyond them)."""
        if self.times is None:
            return frame / self.fps
        t = self.times.get(frame)
        if t is not None:
            return t
        fr, ts = self._table()
        if not len(fr):
            return frame / self.fps
        if frame < fr[0]:
            return float(ts[0] - (fr[0] - frame) / self.fps)
        if frame > fr[-1]:
            return float(ts[-1] + (frame - fr[-1]) / self.fps)
        return float(np.interp(frame, fr, ts))

    def frame_at(self, t: float) -> int:
        """The frame shown at video time ``t`` (the nearest: round(t * fps) at a constant rate)."""
        if self.times is None:
            return round(t * self.fps)
        fr, ts = self._table()
        if not len(fr):
            return round(t * self.fps)
        if t <= ts[0]:
            return int(fr[0] - round((ts[0] - t) * self.fps))
        if t >= ts[-1]:
            return int(fr[-1] + round((t - ts[-1]) * self.fps))
        j = int(np.searchsorted(ts, t))
        a, b = j - 1, j
        if fr[b] - fr[a] > 1:  # an untracked stretch: in proportion
            return int(fr[a] + round((t - ts[a]) / (ts[b] - ts[a]) * (fr[b] - fr[a])))
        return int(fr[a] if t - ts[a] <= ts[b] - t else fr[b])

    def get(self, frame: int) -> CameraParams | None:
        q = self.params.get(frame)
        if q is None:
            return None
        return CameraParams(self.center, q[0], q[1], q[2], q[3])

    def homography(self, frame: int) -> Homography | None:
        cam = self.get(frame)
        return None if cam is None else Homography(np.linalg.inv(cam.G(self.image_size)))

    def frame_to_frame(self, a: int, b: int) -> np.ndarray | None:
        """Image a -> image b homography of the camera's turn and zoom about its centre (exact for any point)."""
        ca, cb = self.get(a), self.get(b)
        if ca is None or cb is None:
            return None
        return cb.K(self.image_size) @ cb.R() @ ca.R().T @ np.linalg.inv(ca.K(self.image_size))


def _smooth_runs(raw: dict[int, tuple[float, float, float, float]], k: int,
                 max_fill: int) -> dict[int, tuple[float, float, float, float]]:
    """Fill gaps of at most ``max_fill`` frames by interpolation, then within each run of consecutive frames a
    centred moving median of width ``k`` (a single bad fit is dropped, not smeared over its neighbours) followed
    by a moving average of width ``k`` (the per-frame fits jitter by about a pixel)."""
    frames = sorted(raw)
    out: dict[int, tuple[float, float, float, float]] = {}
    runs: list[list[int]] = []
    for f in frames:
        if runs and f - runs[-1][-1] <= max_fill + 1:
            runs[-1].append(f)
        else:
            runs.append([f])
    for run in runs:
        full = np.arange(run[0], run[-1] + 1)
        arr = np.array([raw[f] for f in run], float)
        arr[:, 3] = np.log(arr[:, 3])
        vals = np.column_stack([np.interp(full, run, arr[:, j]) for j in range(4)])
        if k > 1 and len(full) >= k:
            pad = k // 2
            padded = np.vstack([vals[:1].repeat(pad, 0), vals, vals[-1:].repeat(pad, 0)])
            vals = np.median(np.lib.stride_tricks.sliding_window_view(padded, k, axis=0), axis=2)
            padded = np.vstack([vals[:1].repeat(pad, 0), vals, vals[-1:].repeat(pad, 0)])
            kernel = np.ones(k) / k
            vals = np.column_stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(4)])
        for f, v in zip(full, vals, strict=True):
            out[int(f)] = (float(v[0]), float(v[1]), float(v[2]), float(math.exp(v[3])))
    return out


# --------------------------------------------------------------------------- scorebug


def load_scorebug(path: Path) -> list[ScoreboardReading]:
    out = []
    for r in map(json.loads, path.open()):
        if r["game_clock"] is None and r["period"] is None:
            continue
        out.append(ScoreboardReading(t=r["t"], home_score=r["home_score"], away_score=r["away_score"],
                                     period=r["period"], game_clock=r["game_clock"],
                                     shot_clock=None if r["shot_clock"] is None else float(r["shot_clock"]),
                                     source="ocr", confidence=0.95))
    return out


# --------------------------------------------------------------------------- detections


@dataclass
class DetectionCache:
    """Per-frame ball candidates (x, y, score) and person boxes (x1, y1, x2, y2, score), full-resolution px."""

    balls: dict[int, np.ndarray] = field(default_factory=dict)
    persons: dict[int, np.ndarray] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        frames = np.array(sorted(self.balls), np.int32)
        b = [self.balls[f] for f in frames]
        p = [self.persons.get(int(f), np.zeros((0, 5), np.float32)) for f in frames]
        np.savez_compressed(path, frames=frames, ball_n=np.array([len(x) for x in b], np.int32),
                            ball=np.concatenate(b) if b else np.zeros((0, 3), np.float32),
                            person_n=np.array([len(x) for x in p], np.int32),
                            person=np.concatenate(p) if p else np.zeros((0, 5), np.float32))

    @classmethod
    def load(cls, path: Path) -> DetectionCache:
        # each z[name] decompresses the whole array again, and a frame's slice keeps its copy alive: read once
        with np.load(path) as z:
            frames, ball, person = z["frames"], z["ball"], z["person"]
            bi = np.concatenate([[0], np.cumsum(z["ball_n"])])
            pi = np.concatenate([[0], np.cumsum(z["person_n"])])
        out = cls()
        for k, f in enumerate(frames):
            out.balls[int(f)] = ball[bi[k]:bi[k + 1]]
            out.persons[int(f)] = person[pi[k]:pi[k + 1]]
        return out

    def merge(self, other: DetectionCache) -> None:
        self.balls.update(other.balls)
        self.persons.update(other.persons)


def iter_frames(video: Path, start: int = 0, end: int | None = None, exact: bool = False
                ) -> Iterator[tuple[int, np.ndarray]]:
    """Frames [start, end) of ``video`` with their numbers (the order the decoder gives them, from 0). OpenCV's frame
    seek is exact on a constant-frame-rate file but lands on another frame of a variable-rate one (it turns the
    number into a time with the mean rate: a seek to frame 200 of a test file returned its 245th frame); ``exact``
    decodes and drops the frames before ``start`` instead."""
    cap = cv2.VideoCapture(str(video))
    if exact:
        idx = 0
        while idx < start and cap.grab():
            idx += 1
        if idx < start:
            cap.release()
            return
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
    while end is None or idx < end:
        ok, img = cap.read()
        if not ok:
            break
        yield idx, img
        idx += 1
    cap.release()


def prefetch(items: Iterable[T], depth: int = 8) -> Iterator[T]:
    """``items`` produced ahead in a background thread, at most ``depth`` waiting, in their order: the decoder and the
    numpy / OpenCV work release the GIL, so decoding overlaps the caller's work. An exception of the producer is
    raised here; leaving the loop stops the producer (which closes ``items``)."""
    q: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=max(1, depth))
    stop = threading.Event()

    def put(item: tuple[bool, Any]) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def run() -> None:
        it = iter(items)
        try:
            for x in it:
                if not put((True, x)):
                    return
            put((False, None))
        except BaseException as exc:  # noqa: BLE001 - handed to the consumer
            put((False, exc))
        finally:
            close = getattr(it, "close", None)
            if callable(close):
                close()

    th = threading.Thread(target=run, name="prefetch", daemon=True)
    th.start()
    try:
        while True:
            more, x = q.get()
            if not more:
                if x is not None:
                    raise x
                return
            yield x
    finally:
        stop.set()
        th.join(timeout=5.0)


def detect_video(video: Path, camera: CameraTrack, weights: Path, start: int = 0, end: int | None = None,
                 batch: int = 4, progress: bool = True, backend: str = "auto") -> DetectionCache:
    """Student detections on every frame of [start, end) whose court is in view (the camera is tracked). The
    student sees each frame with two earlier ones spaced as in its training clips (``student.history_step``: the
    previous two frames at 29.97 fps, every second frame at 50-60 fps). Frames are decoded and preprocessed in a
    background thread (``prefetch``) while the network runs (``backend``: ``StudentConfig.backend``; "torch" gives
    the detections of the MPS / CPU network bit for bit)."""
    import time

    from courtvision.perception.student import (
        StudentConfig,
        StudentDetector,
        history_step,
        preprocess_frames,
    )

    det = StudentDetector(StudentConfig(weights=str(weights), ball_thresh=0.04, person_thresh=0.25, backend=backend))
    n_in, step = det.config.frames, history_step(camera.fps)
    cache = DetectionCache()
    hist: list[np.ndarray] = []  # each frame's own preprocessed (3, H, W) slice, oldest first
    pending: list[tuple[int, np.ndarray, tuple[int, int]]] = []
    t0 = time.time()
    n = 0

    def frames() -> Iterator[tuple[int, tuple[int, int], np.ndarray]]:
        for idx, img in iter_frames(video, start, end, exact=camera.vfr):
            yield idx, (img.shape[1], img.shape[0]), preprocess_frames([img])

    def flush() -> None:
        nonlocal n
        if not pending:
            return
        res = det.infer_arrays(np.stack([x for _, x, _ in pending]), [wh for _, _, wh in pending])
        for (f, _, _), (balls, persons) in zip(pending, res, strict=True):
            cache.balls[f] = np.asarray(balls, np.float32).reshape(-1, 3)
            cache.persons[f] = np.asarray(persons, np.float32).reshape(-1, 5)
        n += len(pending)
        pending.clear()

    for idx, wh, x in prefetch(frames()):
        if not hist and wh != tuple(camera.image_size):
            raise ValueError(f"{video} is {wh[0]}x{wh[1]}, the camera track "
                             f"{camera.image_size[0]}x{camera.image_size[1]}: detections would be misplaced")
        hist = (hist + [x])[-((n_in - 1) * step + 1):]
        if camera.get(idx) is None:
            continue
        fs = [hist[max(0, len(hist) - 1 - j * step)] for j in range(n_in - 1, -1, -1)]  # oldest first; the
        #   first frames of the range repeat the earliest one
        pending.append((idx, np.concatenate(fs, axis=0), wh))
        if len(pending) >= batch:
            flush()
            if progress and n % 2000 < batch:
                log.info("detect: %d frames (frame %d), %.1f ms/frame", n, idx, 1000 * (time.time() - t0) / n)
    flush()
    return cache


# --------------------------------------------------------------------------- ball


@dataclass
class BallLinkConfig:
    """Ball path settings in the units of the tuning footage: frames at ``REF_FPS`` and px of a ``REF_HEIGHT``-line
    frame. ``link_ball`` applies them in the camera track's own frames and px (``for_video``)."""

    min_score: float = 0.08
    max_candidates: int = 6
    bias: float = 0.25
    max_gap: int = 6  # frames a chain may skip (0.2 s)
    max_step_px: float = 90.0  # per frame, after removing the camera's turn (2700 px/s)
    free_step_px: float = 30.0  # per frame: slower steps cost nothing
    speed_cost: float = 0.004  # per px / frame above free_step_px, per link
    gap_cost: float = 0.1  # per skipped frame
    restart_cost: float = 0.5

    def for_video(self, fps: float, height: float) -> BallLinkConfig:
        """These settings in the frames and px of a ``fps`` video of ``height`` lines (``self`` itself at REF_FPS
        and REF_HEIGHT). The same ball speeds (px / s of a 1080-line frame) and gap times pass the gates. A frame's
        score counts as before, so with r = ``rate_ratio(fps)`` times the frames a second a path's evidence grows r
        times as fast; the costs keep pace: a link costs what a link of the same speed cost before (there are r
        times as many), a restart r times as much, and a skip ``gap_cost`` per skipped frame beyond r - 1
        (``link_ball``), so one stretch of time without the ball costs r times as much as before."""
        r, s = rate_ratio(fps), height / REF_HEIGHT
        if r == 1.0 and s == 1.0:
            return self
        return replace(self, max_gap=scale_frames(self.max_gap, r), max_step_px=self.max_step_px * s / r,
                       free_step_px=self.free_step_px * s / r, speed_cost=self.speed_cost * r / s,
                       restart_cost=self.restart_cost * r)


def _video_of(camera: Any) -> tuple[float, float]:
    """(fps, frame height) of a camera track; a stand-in without them (tools/nba_ball_bench.py) is taken for the
    tuning footage."""
    size = getattr(camera, "image_size", None)
    return float(getattr(camera, "fps", REF_FPS)), float(size[1]) if size is not None else float(REF_HEIGHT)


def flight_config_for(fps: float, config: FlightSearchConfig | None = None) -> FlightSearchConfig | None:
    """``ball3d.find_flights`` settings counted in path points (one a frame: ``min_obs``, ``seed_obs``,
    ``max_skip``, given for REF_FPS) for a ``fps`` video, so a flight needs as long a stretch as before; ``config``
    itself (None: find_flights' defaults) at REF_FPS."""
    r = rate_ratio(fps)
    if r == 1.0:
        return config
    cfg = config or FlightSearchConfig()
    return replace(cfg, min_obs=scale_frames(cfg.min_obs, r), seed_obs=tuple(scale_frames(n, r) for n in cfg.seed_obs),
                   max_skip=scale_frames(cfg.max_skip, r, minimum=0))


@dataclass
class BallTrack:
    """The linked ball path (frame -> x, y, score) and the 3-D flights along it."""

    path: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    flights: list[Flight3D] = field(default_factory=list)

    def flight_at(self, frame: int) -> Flight3D | None:
        for f in self.flights:  # few hundred per game; linear scan is fine per frame
            if f.frames[0] <= frame <= f.frames[1]:
                return f
        return None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"path": {str(k): [round(v, 2) for v in xy] for k, xy in self.path.items()},
                                    "flights": [f.to_dict() for f in self.flights]}))

    @classmethod
    def load(cls, path: Path) -> BallTrack:
        d = json.loads(path.read_text())
        flights = [Flight3D(f["t0"], f["t1"], np.array(f["p0"]), np.array(f["v0"]), f["rms_px"], f["n_obs"],
                            tuple(f["frames"])) for f in d["flights"]]  # type: ignore[arg-type]
        return cls({int(k): tuple(v) for k, v in d["path"].items()}, flights)  # type: ignore[misc]


def _segments(frames: list[int]) -> list[list[int]]:
    segs: list[list[int]] = []
    for f in frames:
        if segs and f == segs[-1][-1] + 1:
            segs[-1].append(f)
        else:
            segs.append([f])
    return segs


def _warp(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h = pts @ M[:, :2].T + M[:, 2]
    return h[:, :2] / h[:, 2:3]


def link_ball(cache: DetectionCache, camera: CameraTrack, config: BallLinkConfig | None = None) -> dict[int, tuple[float, float, float]]:
    """One ball path per camera segment (see ``training/ball_labels.link_ball_path``), with candidate motion measured
    after mapping the earlier frame's candidates through the camera's frame-to-frame homography. ``config`` is in
    the units of the tuning footage (``BallLinkConfig.for_video`` converts it to the camera track's)."""
    fps, height = _video_of(camera)
    cfg = (config or BallLinkConfig()).for_video(fps, height)
    gap_free = rate_ratio(fps)  # a link spanning up to one frame of the tuning footage skips nothing
    path: dict[int, tuple[float, float, float]] = {}
    for seg in _segments(sorted(f for f in cache.balls if camera.get(f) is not None)):
        T = len(seg)
        cands = []
        for f in seg:
            b = cache.balls[f]
            b = b[b[:, 2] >= cfg.min_score][: cfg.max_candidates]
            cands.append(b)
        V: list[np.ndarray] = []
        back: list[list[tuple[str, tuple[int, int] | None]]] = []
        prefix: list[tuple[float, tuple[int, int] | None]] = [(0.0, None)]
        for t in range(T):
            un = cands[t][:, 2] - cfg.bias
            base, base_node = prefix[t]
            best = base - cfg.restart_cost + un
            ptr: list[tuple[str, tuple[int, int] | None]] = [("restart", base_node)] * len(un)
            for g in range(1, cfg.max_gap + 1):
                tp = t - g
                if tp < 0 or len(un) == 0:
                    break
                if len(cands[tp]) == 0:
                    continue
                M = camera.frame_to_frame(seg[tp], seg[t])
                prev = _warp(M, cands[tp][:, :2].astype(float)) if M is not None else cands[tp][:, :2]
                D = np.linalg.norm(cands[t][:, None, :2] - prev[None, :, :], axis=2) / g
                val = V[tp][None, :] + un[:, None] - cfg.gap_cost * max(0.0, g - gap_free) \
                    - cfg.speed_cost * np.maximum(0.0, D - cfg.free_step_px)
                val[D > cfg.max_step_px] = -np.inf
                k = np.argmax(val, axis=1)
                v = val[np.arange(len(un)), k]
                for i in np.flatnonzero(v > best):
                    best[i] = v[i]
                    ptr[i] = ("chain", (tp, int(k[i])))
            V.append(best)
            back.append(ptr)
            bv, bn = prefix[t]
            if len(best):
                i = int(np.argmax(best))
                if best[i] > bv:
                    bv, bn = float(best[i]), (t, i)
            prefix.append((bv, bn))
        node = prefix[T][1]
        while node is not None:
            t, k = node
            x, y, s = cands[t][k]
            path[seg[t]] = (float(x), float(y), float(s))
            node = back[t][k][1]
    return path


def build_ball_track(cache: DetectionCache, camera: CameraTrack, config: BallLinkConfig | None = None,
                     flight_config: FlightSearchConfig | None = None, court: CourtSpec | None = None) -> BallTrack:
    """The linked ball path and its 3-D flights; flights that come down at a rim are refitted through it (the
    rim pins the depth a single camera sees poorly: ``ball3d.find_flights``). Both configs are in the units of the
    tuning footage (``BallLinkConfig.for_video``, ``flight_config_for``); the flights are fitted on the frames'
    times (``CameraTrack.time``)."""
    cfg = config or BallLinkConfig()
    court = court or court_for("nba")
    baskets = {side: court.basket_xy(side) for side in ("left", "right")}
    path = link_ball(cache, camera, cfg)
    max_gap = cfg.for_video(camera.fps, camera.image_size[1]).max_gap
    min_run = scale_frames(6, rate_ratio(camera.fps))
    flight_config = flight_config_for(camera.fps, flight_config)
    flights: list[Flight3D] = []
    frames = sorted(path)
    # one run per stretch of the path whose gaps the linker itself bridges (a missed detection mid-flight must not
    # cut the flight in two: find_flights skips such gaps itself)
    runs: list[list[int]] = []
    for f in frames:
        if runs and f - runs[-1][-1] <= max_gap and all(camera.get(g) is not None for g in range(runs[-1][-1] + 1, f)):
            runs[-1].append(f)
        else:
            runs.append([f])
    for seg in runs:
        if len(seg) < min_run:
            continue
        t = np.array([camera.time(f) for f in seg])
        uv = np.array([path[f][:2] for f in seg])
        cams = [camera.get(f) for f in seg]
        flights += find_flights(t, uv, cams, camera.image_size, frames=np.array(seg), config=flight_config,  # type: ignore[arg-type]
                                court_wh=(court.length, court.width), baskets=baskets, rim_height=court.rim_height)
    return BallTrack(path, flights)


# --------------------------------------------------------------------------- per-frame states


def rim_boxes(camera: CameraParams, court: CourtSpec, size: tuple[int, int]) -> dict[str, BBox]:
    cam = camera.camera(size)
    out: dict[str, BBox] = {}
    for side in ("left", "right"):
        bx, by = court.basket_xy(side)  # type: ignore[arg-type]
        box = cam.bbox_of_circle(bx, by, court.rim_height, court.rim_radius)
        if np.all(np.isfinite([box.x1, box.y1, box.x2, box.y2])) and box.x2 > 0 and box.x1 < size[0] \
                and box.y2 > 0 and box.y1 < size[1]:
            out[side] = box
    return out


def is_referee(frame_bgr: np.ndarray, box: tuple[float, float, float, float]) -> bool:
    """NBA referees wear grey shirts with black stripes: an achromatic, mid-bright torso with dark stripes (white
    and coloured jerseys fail it). Thresholds from torso statistics of on-court people on ESPN frames."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    crop = frame_bgr[int(y1 + 0.2 * bh):int(y1 + 0.5 * bh), int(x1 + 0.25 * bw):int(x2 - 0.25 * bw)]
    if crop.size < 30:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    s, v = _OrderStats(hsv[:, 1]), _OrderStats(hsv[:, 2])
    return bool(s.median() < 70 and v.median() < 175 and v.percentile(20) < 75)


class _OrderStats:
    """``np.median`` / ``np.percentile`` (linear) of 8-bit values from their counts: the same float64 results without
    sorting (``is_referee`` asks for three of them per person box)."""

    def __init__(self, values: np.ndarray) -> None:
        self.n = len(values)
        self.cum = np.cumsum(np.bincount(values, minlength=256))

    def kth(self, k: int) -> float:
        """The k-th smallest value (from 0)."""
        return float(np.searchsorted(self.cum, k, side="right"))

    def median(self) -> float:
        return (self.kth((self.n - 1) // 2) + self.kth(self.n // 2)) / 2.0  # numpy: the mean of the middle one or two

    def percentile(self, q: float) -> float:
        virtual = (self.n - 1) * (q / 100)  # numpy's "linear" method, step by step
        if virtual >= self.n - 1:
            return self.kth(self.n - 1)
        prev = math.floor(virtual)
        a, b, t = self.kth(prev), self.kth(prev + 1), virtual - prev
        return b - (b - a) * (1 - t) if t >= 0.5 else a + (b - a) * t


def rectify_box(box: tuple[float, float, float, float], cam: CameraParams, size: tuple[int, int],
                body_h: float = 6.6, center_frac: float = 0.5) -> tuple[float, float, float, float] | None:
    """The box of a standing player of height ``body_h`` whose body centre is the box centre: the centre's ray
    meets the height ``center_frac * body_h`` above a floor point, and the box spans that point's feet and head
    (width scaled with the height). A detector whose boxes are well centred but sized off (the student's
    undersized boxes) then still gives the right feet. None when the ray does not come down to that height."""
    x1, y1, x2, y2 = box
    u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    R = cam.R()
    d = R.T @ np.array([(u - size[0] / 2.0) / cam.f, (v - size[1] / 2.0) / cam.f, 1.0])
    C = np.asarray(cam.center, float)
    zc = center_frac * body_h
    if d[2] >= -1e-6:
        return None
    s = (zc - C[2]) / d[2]
    X = C + s * d

    def proj(p: np.ndarray) -> tuple[float, float]:
        q = R @ (p - C)
        return (cam.f * q[0] / q[2] + size[0] / 2.0, cam.f * q[1] / q[2] + size[1] / 2.0)

    _, foot_v = proj(np.array([X[0], X[1], 0.0]))
    _, head_v = proj(np.array([X[0], X[1], body_h]))
    h_new = foot_v - head_v
    if not h_new > 1.0:
        return None
    half_w = 0.5 * (x2 - x1) * h_new / max(1.0, y2 - y1)
    return (u - half_w, head_v, u + half_w, foot_v)


class CachedDetector:
    """``Detector`` replaying the cache (persons; referees relabelled; no balls: the ball is set afterwards). With
    ``camera``, person boxes are rectified to a standing player at their centre (``rectify_box``)."""

    def __init__(self, cache: DetectionCache, min_person: float = 0.3, camera: CameraTrack | None = None) -> None:
        self.cache = cache
        self.min_person = min_person
        self.camera = camera

    def detect(self, frame_bgr: np.ndarray, frame_idx: int, t: float) -> FrameDetections:
        h, w = frame_bgr.shape[:2]
        cam = self.camera.get(frame_idx) if self.camera is not None else None
        dets = []
        for x1, y1, x2, y2, sc in self.cache.persons.get(frame_idx, np.zeros((0, 5))):
            if sc < self.min_person:
                continue
            cls = ObjClass.REFEREE if is_referee(frame_bgr, (x1, y1, x2, y2)) else ObjClass.PLAYER
            box = (float(x1), float(y1), float(x2), float(y2))
            if cam is not None:
                box = rectify_box(box, cam, (w, h)) or box
            dets.append(Detection(BBox(*box), cls, float(sc)))
        return FrameDetections(frame_idx=frame_idx, t=t, image_size=(w, h), detections=dets)


def ball_state(frame: int, t: float, track: BallTrack, camera: CameraTrack, use_3d: bool) -> BallState:
    """The frame's ball: inside a fitted flight its 3-D point (``use_3d``: with height; else the image point and its
    floor projection, like the monocular simulator); on the linked path the detected point; else not visible."""
    cam = camera.get(frame)
    if cam is None:
        return BallState(visible=False)
    size = camera.image_size
    fl = track.flight_at(frame)
    p = track.path.get(frame)
    if fl is not None and use_3d:
        X = fl.at(t)
        pc = cam.R() @ (X - camera.center)
        uv = (float(cam.f * pc[0] / pc[2] + size[0] / 2.0), float(cam.f * pc[1] / pc[2] + size[1] / 2.0))
        return BallState(visible=True, image_xy=uv, court_xy=(float(X[0]), float(X[1])), z=float(max(0.0, X[2])),
                         radius_px=None, confidence=0.9, interpolated=p is None)
    if p is None:
        return BallState(visible=False)
    H = Homography(np.linalg.inv(cam.G(size)))
    xy = H.image_to_court(np.array(p[:2]), front_only=True)
    court_xy = (float(xy[0]), float(xy[1])) if np.all(np.isfinite(xy)) else None
    return BallState(visible=True, image_xy=(p[0], p[1]), court_xy=court_xy, z=None, radius_px=None,
                     confidence=float(min(1.0, 0.4 + p[2])), interpolated=False)


def scoreboard_by_frame(reads: list[ScoreboardReading], fps: float,
                        frame_at: Callable[[float], int] | None = None) -> dict[int, ScoreboardReading]:
    """The scorebug reads by the frame shown at their time: round(t * fps), or ``frame_at`` (``CameraTrack.frame_at``,
    which also knows a variable frame rate)."""
    if frame_at is None:
        return {int(round(r.t * fps)): r for r in reads}
    return {frame_at(r.t): r for r in reads}


def perception_rate_overrides(fps: float) -> dict[str, Any]:
    """``PerceptionConfig`` overrides for the pipeline settings counted in frames (its own fields and its ``tracker``
    and ``identity`` dicts), so that on a ``fps`` video they keep the times they have at REF_FPS; {} there (the
    tuned pipeline unchanged). They are: ByteTrack's lost-track buffer (its own ``frame_rate`` rule) and how long a
    lost track is still reported, the team-colour sampling interval, the age that makes a track established, the
    per-track memory, the per-frame decay of a coasting track's confidence, and the identity layer's young-track age,
    its per-frame coasting penalty and its rate. Taken from the pipeline's current defaults."""
    r = rate_ratio(fps)
    if r == 1.0:
        return {}
    from courtvision.perception.identity import IdentityConfig
    from courtvision.perception.pipeline import PerceptionConfig
    from courtvision.perception.tracker import TrackerConfig

    p, tc, ic = PerceptionConfig(), TrackerConfig(), IdentityConfig()
    return {"team_observe_every": scale_frames(p.team_observe_every, r),
            "duplicate_new_frames": scale_frames(p.duplicate_new_frames, r),
            "track_memory_frames": scale_frames(p.track_memory_frames, r),
            "coast_confidence_decay": p.coast_confidence_decay ** (1.0 / r),
            "tracker": {"frame_rate": float(fps), "emit_lost_frames": scale_frames(tc.emit_lost_frames, r, 0)},
            "identity": {"relink_new_track_frames": scale_frames(ic.relink_new_track_frames, r),
                         "coast_penalty": ic.coast_penalty / r, "fps": float(fps)}}


@dataclass
class PerceiveConfig:
    use_3d_flights: bool = False
    rectify_boxes: bool = True  # person boxes re-drawn as a standing player at their centre (``rectify_box``)
    home_color_bgr: tuple[int, int, int] | None = None
    away_color_bgr: tuple[int, int, int] | None = None
    team_refit_at: tuple[int, ...] = ()  # TeamConfig.refit_at (nba_analyze.py events --team-refit)
    league: str = "nba"
    # ByteTrack thresholds for the student detector, whose person scores sit lower than RF-DETR's (a centre
    # heatmap peak, not a classification score); tuned on Lakers @ Warriors frames
    tracker: dict[str, Any] = field(default_factory=lambda: {"high_thresh": 0.3, "low_thresh": 0.1,
                                                             "new_track_thresh": 0.35})


def perceive(video: Path, camera: CameraTrack, cache: DetectionCache, ball: BallTrack,
             reads: list[ScoreboardReading], config: PerceiveConfig | None = None, start: int = 0,
             end: int | None = None) -> Iterator[FrameState]:
    """FrameStates of [start, end) (see the module docstring). Frame times are ``camera.time``; the pipeline's
    frame-count settings keep their times at the camera's rate (``perception_rate_overrides``). The event engine's
    own frame counts are not converted (``run_engine``)."""
    from courtvision.perception.pipeline import PerceptionConfig, PerceptionPipeline

    cfg = config or PerceiveConfig()
    court = court_for(cfg.league)
    size = camera.image_size
    team = {}
    if cfg.home_color_bgr is not None:
        team["home_color_bgr"] = cfg.home_color_bgr
    if cfg.away_color_bgr is not None:
        team["away_color_bgr"] = cfg.away_color_bgr
    if cfg.team_refit_at:
        team["refit_at"] = tuple(cfg.team_refit_at)
    rate = perception_rate_overrides(camera.fps)
    pcfg = PerceptionConfig(team=team, foot_direct=True, court_margin_ft=4.0, compute_embeddings=True,
                            tracker={**rate.pop("tracker", {}), **cfg.tracker}, identity=rate.pop("identity", {}),
                            **rate)
    first_cam = next(iter(sorted(camera.params)))
    calib0 = Calibration(homography=camera.homography(first_cam), image_size=size, league=cfg.league)  # type: ignore[arg-type]
    pipe = PerceptionPipeline(CachedDetector(cache, camera=camera if cfg.rectify_boxes else None), calib0, config=pcfg)
    board = scoreboard_by_frame(reads, camera.fps, camera.frame_at)
    prev_cam_frame: int | None = None
    for idx, img in iter_frames(video, start, end, exact=camera.vfr):
        t = camera.time(idx)
        sb = board.get(idx)
        cam = camera.get(idx)
        if cam is None:
            if prev_cam_frame is not None:
                pipe.reset_tracks()
                prev_cam_frame = None
            yield FrameState(idx, t, [], None, [], sb, size)
            continue
        if prev_cam_frame is not None and prev_cam_frame == idx - 1:
            M = camera.frame_to_frame(prev_cam_frame, idx)
            if M is not None:
                pipe.warp_image_state(M)
        prev_cam_frame = idx
        hom = Homography(np.linalg.inv(cam.G(size)))
        pipe.calibration = Calibration(homography=hom, image_size=size, league=cfg.league,
                                       rims=rim_boxes(cam, court, size))
        st = pipe.process(img, idx, t)
        st.ball = ball_state(idx, t, ball, camera, cfg.use_3d_flights)
        st.scoreboard = sb
        st.camera_xyz = (float(camera.center[0]), float(camera.center[1]), float(camera.center[2]))
        yield st


def run_engine(frames: Iterator[FrameState], league: str = "nba", config: dict[str, Any] | None = None) -> list[Any]:
    """The event engine on the frame states. Its settings in seconds follow the frames' times, but several are
    counts of observed frames tuned at 30 fps (``ball_reacquire_frames``, ``holder_vote_loose_n``,
    ``holder_clear_frames``, ``toss_min_frames``, ``holder_oob_frames``, ...): at another rate pass converted
    values in ``config``."""
    from courtvision.events.engine import EventEngine, EventEngineConfig
    from courtvision.geometry.rules import rules_for
    from courtvision.perception.pipeline import _sub

    ecfg = _sub(EventEngineConfig, config) if config else EventEngineConfig()
    engine = EventEngine(court_for(league), None, ecfg, rules=rules_for(league))
    events = []
    for fs in frames:
        events.extend(engine.process(fs))
    events.extend(engine.flush())
    return events
