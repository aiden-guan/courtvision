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

Research use only (NBA footage and statistics; docs/TRAINING.md §1).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

__all__ = ["BallTrack", "CameraTrack", "DetectionCache", "detect_video", "link_ball", "load_scorebug"]

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- camera


@dataclass
class CameraTrack:
    """The game camera per frame (full-resolution pixels): fixed centre, per-frame pan / tilt / roll / focal."""

    center: np.ndarray
    image_size: tuple[int, int]
    fps: float
    params: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)  # frame -> pan, tilt, roll, f

    @classmethod
    def load(cls, paths: list[Path], smooth: int = 5, max_fill: int = 3) -> CameraTrack:
        center, size, fps = None, (1920, 1080), 29.97
        raw: dict[int, tuple[float, float, float, float]] = {}
        for p in sorted(paths):
            for line in p.open():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # a line still being written by the tracker
                if r.get("header"):
                    center, size, fps = np.array(r["center"], float), tuple(r["image_size"]), float(r["fps"])
                    continue
                q = r.get("params")
                if q is not None and r.get("status") != "held":
                    raw[int(r["frame"])] = (q["pan"], q["tilt"], q["roll"], q["f"])
        if center is None:
            raise ValueError("no camera header")
        track = cls(center, (int(size[0]), int(size[1])), fps)
        track.params = _smooth_runs(raw, smooth, max_fill)
        return track

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
        z = np.load(path)
        out = cls()
        bi = np.concatenate([[0], np.cumsum(z["ball_n"])])
        pi = np.concatenate([[0], np.cumsum(z["person_n"])])
        for k, f in enumerate(z["frames"]):
            out.balls[int(f)] = z["ball"][bi[k]:bi[k + 1]]
            out.persons[int(f)] = z["person"][pi[k]:pi[k + 1]]
        return out

    def merge(self, other: DetectionCache) -> None:
        self.balls.update(other.balls)
        self.persons.update(other.persons)


def iter_frames(video: Path, start: int = 0, end: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    while end is None or idx < end:
        ok, img = cap.read()
        if not ok:
            break
        yield idx, img
        idx += 1
    cap.release()


def detect_video(video: Path, camera: CameraTrack, weights: Path, start: int = 0, end: int | None = None,
                 batch: int = 4, progress: bool = True) -> DetectionCache:
    """Student detections on every frame of [start, end) whose court is in view (the camera is tracked)."""
    import time

    from courtvision.perception.student import StudentConfig, StudentDetector

    from courtvision.perception.student import preprocess_frames

    det = StudentDetector(StudentConfig(weights=str(weights), ball_thresh=0.04, person_thresh=0.25))
    cache = DetectionCache()
    hist: list[np.ndarray] = []  # each frame's own preprocessed (3, H, W) slice, oldest first
    pending: list[tuple[int, np.ndarray, int]] = []
    t0 = time.time()
    n = 0

    def flush() -> None:
        nonlocal n
        if not pending:
            return
        res = det.infer_arrays(np.stack([x for _, x, _ in pending]), [w for _, _, w in pending])
        for (f, _, _), (balls, persons) in zip(pending, res, strict=True):
            cache.balls[f] = np.asarray(balls, np.float32).reshape(-1, 3)
            cache.persons[f] = np.asarray(persons, np.float32).reshape(-1, 5)
        n += len(pending)
        pending.clear()

    for idx, img in iter_frames(video, start, end):
        hist = (hist + [preprocess_frames([img])])[-3:]
        if camera.get(idx) is None:
            continue
        fs = list(hist)
        while len(fs) < 3:
            fs.insert(0, fs[0])
        pending.append((idx, np.concatenate(fs, axis=0), img.shape[1]))
        if len(pending) >= batch:
            flush()
            if progress and n % 2000 < batch:
                log.info("detect: %d frames (frame %d), %.1f ms/frame", n, idx, 1000 * (time.time() - t0) / n)
    flush()
    return cache


# --------------------------------------------------------------------------- ball


@dataclass
class BallLinkConfig:
    min_score: float = 0.08
    max_candidates: int = 6
    bias: float = 0.25
    max_gap: int = 6
    max_step_px: float = 90.0  # per frame, after removing the camera's turn
    free_step_px: float = 30.0
    speed_cost: float = 0.004
    gap_cost: float = 0.1
    restart_cost: float = 0.5


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
    after mapping the earlier frame's candidates through the camera's frame-to-frame homography."""
    cfg = config or BallLinkConfig()
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
                val = V[tp][None, :] + un[:, None] - cfg.gap_cost * (g - 1) - cfg.speed_cost * np.maximum(0.0, D - cfg.free_step_px)
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
                     flight_config: FlightSearchConfig | None = None) -> BallTrack:
    cfg = config or BallLinkConfig()
    path = link_ball(cache, camera, cfg)
    flights: list[Flight3D] = []
    frames = sorted(path)
    # one run per stretch of the path whose gaps the linker itself bridges (a missed detection mid-flight must not
    # cut the flight in two: find_flights skips such gaps itself)
    runs: list[list[int]] = []
    for f in frames:
        if runs and f - runs[-1][-1] <= cfg.max_gap and all(camera.get(g) is not None for g in range(runs[-1][-1] + 1, f)):
            runs[-1].append(f)
        else:
            runs.append([f])
    for seg in runs:
        if len(seg) < 6:
            continue
        t = np.array([f / camera.fps for f in seg])
        uv = np.array([path[f][:2] for f in seg])
        cams = [camera.get(f) for f in seg]
        flights += find_flights(t, uv, cams, camera.image_size, frames=np.array(seg), config=flight_config)  # type: ignore[arg-type]
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
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    s_med, v_med, v20 = np.median(hsv[:, 1]), np.median(hsv[:, 2]), np.percentile(hsv[:, 2], 20)
    return bool(s_med < 70 and v_med < 175 and v20 < 75)


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


def scoreboard_by_frame(reads: list[ScoreboardReading], fps: float) -> dict[int, ScoreboardReading]:
    return {int(round(r.t * fps)): r for r in reads}


@dataclass
class PerceiveConfig:
    use_3d_flights: bool = False
    rectify_boxes: bool = True  # person boxes re-drawn as a standing player at their centre (``rectify_box``)
    home_color_bgr: tuple[int, int, int] | None = None
    away_color_bgr: tuple[int, int, int] | None = None
    league: str = "nba"
    # ByteTrack thresholds for the student detector, whose person scores sit lower than RF-DETR's (a centre
    # heatmap peak, not a classification score); tuned on Lakers @ Warriors frames
    tracker: dict[str, Any] = field(default_factory=lambda: {"high_thresh": 0.3, "low_thresh": 0.1,
                                                             "new_track_thresh": 0.35})


def perceive(video: Path, camera: CameraTrack, cache: DetectionCache, ball: BallTrack,
             reads: list[ScoreboardReading], config: PerceiveConfig | None = None, start: int = 0,
             end: int | None = None) -> Iterator[FrameState]:
    """FrameStates of [start, end) (see the module docstring)."""
    from courtvision.perception.pipeline import PerceptionConfig, PerceptionPipeline

    cfg = config or PerceiveConfig()
    court = court_for(cfg.league)
    size = camera.image_size
    team = {}
    if cfg.home_color_bgr is not None:
        team["home_color_bgr"] = cfg.home_color_bgr
    if cfg.away_color_bgr is not None:
        team["away_color_bgr"] = cfg.away_color_bgr
    pcfg = PerceptionConfig(team=team, foot_direct=True, court_margin_ft=4.0, compute_embeddings=True,
                            tracker=dict(cfg.tracker))
    first_cam = next(iter(sorted(camera.params)))
    calib0 = Calibration(homography=camera.homography(first_cam), image_size=size, league=cfg.league)  # type: ignore[arg-type]
    pipe = PerceptionPipeline(CachedDetector(cache, camera=camera if cfg.rectify_boxes else None), calib0, config=pcfg)
    board = scoreboard_by_frame(reads, camera.fps)
    prev_cam_frame: int | None = None
    for idx, img in iter_frames(video, start, end):
        t = idx / camera.fps
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
