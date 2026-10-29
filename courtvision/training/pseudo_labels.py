"""Pseudo labels for the student detector from a game it has analysed (self-training).

    .venv/bin/python -m courtvision.training.pseudo_labels runs/nba/<game> --out runs/nba/<game>/pseudo \
        [--max-frames 2000] [--min-score 0.5] [--det-dir DIR] [--per-minute 20] [--dry-run]

After ``tools/nba_analyze.py <game> detect`` and ``ball``, the game folder holds the student's detections
(``detections*/*.npz``), the linked ball path with its 3-D flights (``ball_track.json``) and the camera track
(``camera_parts/*.jsonl``). Without ``--det-dir`` the detection folder the ball track was linked from is found
(``resolve_detections``: at least 90% of a sample of the path's points are candidates there). A frame gets a ball
label only where that evidence pins the ball down:

* **flight**: the frame lies inside a fitted 3-D flight (reprojection rms within ``flight_max_rms_px``), with path
  points that agree with the flight (``flight_inlier_px``) at most ``flight_bracket`` frames before and after it.
  The label is the flight's point projected through the frame's camera, so frames where the student missed the
  blurred ball, or picked something else, are labelled too (``filled``: a lower weight than ``confirmed``, where
  the student's own point agrees);
* **path**: a path point that a quadratic through at least two path points on each side (within
  ``motion_window`` frames; fitted to camera-independent rays, so a pan is no motion) predicts within
  ``motion_tol_px``. It is trusted when its own score reaches ``--min-score`` and no far candidate scores nearly
  as high (``scored``), or, with a lower weight, when a neighbour on each side reaches it (``anchored``: the
  student was unsure, the path's motion is not). Points that stay put for a second either side (a logo, a
  ball on the scorer's table) are left out.

Frames near a cut the court tracker flagged or the start / end of a camera run (``cut_margin``: the student's
three input frames must show one shot), frames without the scorebug (replays, close-ups) and labels under the
ESPN scorebug or at the image border are left out.

Sampling: candidates are binned by minute of video. Each minute offers its frames hardest first (hardness: one
minus the student's score at the label, plus the best score of a candidate elsewhere, so missed balls and
distractors come first), at most ``per_minute`` frames, ``min_spacing`` frames apart and ``per_flight`` per
flight; the minutes take turns until ``--max-frames``, which spreads the set over the game. The video is read
once, front to back, one frame at a time.

Output, a teacher-format folder that ``train_student.py`` reads like a teacher folder: ``frames/<clip>/<frame>.jpg``
(``save_width`` px wide), ``detections.jsonl`` (the cached persons and ball candidates, full-resolution px, with
the frame's ``size``) and ``ball_labels.jsonl``. Each labelled frame is a clip with its two predecessors, stored as
``"context": true`` rows (the student's earlier input frames, not samples). ``meta.json`` marks the folder
``"kind": "pseudo"``. Label weights are at most ``max_weight`` (``train_student.PSEUDO_MAX_WEIGHT``, 0.5: the weight
of the teacher's filled frames; its detections weigh 0.75-1.0), and the cached persons, the student's own boxes,
train with ``person_weight`` (0 by default: its misses would become negatives): the student's own errors weigh
less than the teacher's labels, and the held-out score that picks ``best.pt`` stays teacher-only. Thresholds are
design choices, not measurements.

Research use only (NBA footage; docs/TRAINING.md §1).
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from courtvision.nba.analyze import BallTrack, CameraTrack, DetectionCache
from courtvision.perception.court_search import CameraParams
from courtvision.training.train_student import PSEUDO_MAX_WEIGHT

__all__ = ["Candidate", "FrameGate", "InputError", "PseudoConfig", "build_candidates", "flight_candidates",
           "load_cuts", "path_candidates", "resolve_detections", "scorebug_live", "select_targets", "write_folder"]

# the ESPN scorebug and the row under it (x1, y1, x2, y2 in 1080p px; the court tracker's exclusion box): a ball
# there is hidden by the graphic
BUG_BOX_1080 = (468.0, 944.0, 1468.0, 1069.0)
BALL_RADIUS_1080 = 12.0  # StudentConfig.ball_radius_px: the box drawn around a cached ball point


class InputError(Exception):
    """A game folder or output folder this tool cannot use (the message says why)."""


@dataclass
class PseudoConfig:
    min_score: float = 0.5  # a ball point trusted on its own score, and the score an anchoring neighbour needs
    max_frames: int = 2000  # labelled frames (each stored with its two previous frames)
    per_minute: int = 20  # labelled frames per minute of video
    min_spacing: int = 10  # frames between two labelled frames
    per_flight: int = 3
    cut_margin: int = 15  # frames from a flagged cut or the start / end of a camera run
    bug_check: bool = True  # with scorebug.jsonl: only frames where the bug is shown within bug_tol_s
    bug_tol_s: float = 1.0
    flight_max_rms_px: float = 3.0  # 1080-line px; the fitted flight's reprojection rms
    flight_inlier_px: float = 5.0  # a path point this close to the flight's projection agrees with it
    flight_min_agree: int = 6  # agreeing path points a flight needs
    flight_bracket: int = 4  # a flight frame needs agreeing points at most this many frames before and after
    filled_share: float = 0.7  # weight share of a flight frame without an agreeing point (confirmed: 1)
    motion_window: int = 3  # path neighbours within this many frames on each side
    motion_min_side: int = 2  # neighbours needed on each side
    motion_tol_px: float = 6.0  # 1080-line px: the quadratic's fit to the neighbours and its prediction
    static_window: int = 30  # frames on each side
    static_px: float = 6.0  # 1080-line px: a path point within this of itself over the window is static
    ambiguity: float = 0.85  # a far candidate scoring this share of a scored point makes it ambiguous
    near_px: float = 10.0  # 1080-line px: candidates this close to a label are the student's score there
    far_px: float = 24.0  # 1080-line px: candidates farther than this are other objects
    scored_share: float = 0.8  # weight shares of path labels
    anchored_share: float = 0.6
    image_margin_px: float = 8.0  # 1080-line px
    max_weight: float = PSEUDO_MAX_WEIGHT
    person_weight: float = 0.0  # training weight of the cached person boxes (written to meta.json)
    context: int = 2  # previous frames stored with each labelled frame (the student sees three)
    save_width: int = 960


@dataclass(frozen=True)
class Candidate:
    frame: int
    xy: tuple[float, float]  # full-resolution px
    weight: float  # label weight, at most max_weight
    source: str  # "flight" | "path"
    kind: str  # flight: "confirmed" | "filled"; path: "scored" | "anchored"
    hardness: float  # 1 - the student's score at the label + the best score elsewhere (sampling priority)
    flight: int | None = None  # index of the flight in ball_track.json


# --------------------------------------------------------------------------- inputs


def load_cache(folder: Path) -> DetectionCache:
    """All ``*.npz`` parts of a detection folder (``tools/nba_analyze.py detect``)."""
    cache = DetectionCache()
    for p in sorted(folder.glob("*.npz")):
        cache.merge(DetectionCache.load(p))
    return cache


def load_cuts(paths: Sequence[Path]) -> list[int]:
    """Frames the court tracker flagged as a cut (``"cut": true`` rows of ``tools/nba_track_court.py``)."""
    out: list[int] = []
    for p in sorted(paths):
        for line in p.open():
            try:
                r = json.loads(line)
            except ValueError:
                continue  # a line still being written
            if not r.get("header") and r.get("cut"):
                out.append(int(r["frame"]))
    return out


def scorebug_live(path: Path, tol_s: float = 1.0) -> Callable[[float], bool]:
    """``live(t)``: a scorebug read within ``tol_s`` of video time ``t`` shows the game clock (the bug is shown:
    live play, free throws, dead balls; replays and close-ups hide it)."""
    shown = []
    for line in path.open():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("game_clock") is not None:
            shown.append(float(r["t"]))
    ts = np.array(sorted(shown))

    def live(t: float) -> bool:
        j = int(np.searchsorted(ts, t))
        return any(0 <= k < len(ts) and abs(ts[k] - t) <= tol_s for k in (j - 1, j))

    return live


class FrameGate:
    """Frames fit for a label: inside a run of consecutive camera frames and at least ``margin`` frames from its ends
    and from every cut the court tracker flagged; with ``live``, the scorebug shown at the frame's time."""

    def __init__(self, camera_frames: Iterable[int], cuts: Iterable[int], margin: int,
                 live: Callable[[float], bool] | None = None, fps: float = 29.97) -> None:
        f = np.array(sorted(int(x) for x in camera_frames), np.int64)
        if len(f):
            br = np.flatnonzero(np.diff(f) != 1)
            self._start, self._end = f[np.r_[0, br + 1]], f[np.r_[br, len(f) - 1]]
        else:
            self._start = self._end = np.zeros(0, np.int64)
        self._cuts = np.array(sorted({int(c) for c in cuts}), np.int64)
        self.margin, self.live, self.fps = margin, live, fps

    def __call__(self, frame: int) -> bool:
        i = int(np.searchsorted(self._start, frame, side="right")) - 1
        if i < 0 or frame > self._end[i]:
            return False
        if frame - self._start[i] < self.margin or self._end[i] - frame < self.margin:
            return False
        j = int(np.searchsorted(self._cuts, frame))
        if any(0 <= k < len(self._cuts) and abs(int(self._cuts[k]) - frame) <= self.margin for k in (j - 1, j)):
            return False
        return self.live is None or self.live(frame / self.fps)


# --------------------------------------------------------------------------- geometry


def _to_px(d: np.ndarray, cam: CameraParams, size: tuple[int, int]) -> tuple[float, float] | None:
    """Image point of the camera-centred world direction (or offset) ``d``; None behind the camera."""
    pc = cam.R() @ d
    if pc[2] <= 1e-9:
        return None
    return (float(cam.f * pc[0] / pc[2] + size[0] / 2.0), float(cam.f * pc[1] / pc[2] + size[1] / 2.0))


def _ray(uv: Sequence[float], cam: CameraParams, size: tuple[int, int]) -> np.ndarray:
    d = cam.R().T @ np.array([(uv[0] - size[0] / 2.0) / cam.f, (uv[1] - size[1] / 2.0) / cam.f, 1.0])
    return d / np.linalg.norm(d)


def _placeable(xy: tuple[float, float], size: tuple[int, int], cfg: PseudoConfig) -> bool:
    """Inside the image (with a margin) and not under the scorebug."""
    s = size[1] / 1080.0
    m = cfg.image_margin_px * s
    if not (m <= xy[0] <= size[0] - 1 - m and m <= xy[1] <= size[1] - 1 - m):
        return False
    sx = size[0] / 1920.0
    x1, y1, x2, y2 = BUG_BOX_1080
    return not (x1 * sx <= xy[0] <= x2 * sx and y1 * s <= xy[1] <= y2 * s)


def _scores_near(cands: np.ndarray | None, xy: tuple[float, float], near: float, far: float) -> tuple[float, float]:
    """(best candidate score within ``near`` px of ``xy``, best score farther than ``far`` px)."""
    if cands is None or not len(cands):
        return 0.0, 0.0
    d = np.hypot(cands[:, 0] - xy[0], cands[:, 1] - xy[1])
    a, b = cands[d <= near, 2], cands[d > far, 2]
    return (float(a.max()) if len(a) else 0.0, float(b.max()) if len(b) else 0.0)


# --------------------------------------------------------------------------- candidates


def flight_candidates(track: BallTrack, camera: CameraTrack, cache: DetectionCache, gate: Callable[[int], bool],
                      cfg: PseudoConfig) -> tuple[list[Candidate], set[int]]:
    """Labels from the fitted 3-D flights (see the module docstring) -> (candidates, frames of the flights used)."""
    size = camera.image_size
    s = size[1] / 1080.0
    out: list[Candidate] = []
    spans: set[int] = set()
    for k, fl in enumerate(track.flights):
        if not fl.rms_px <= cfg.flight_max_rms_px * s:
            continue
        uv: dict[int, tuple[float, float]] = {}
        agree: list[int] = []
        for f in range(int(fl.frames[0]), int(fl.frames[1]) + 1):
            cam = camera.get(f)
            if cam is None:
                continue
            p = _to_px(fl.at(f / camera.fps) - np.asarray(cam.center, float), cam, size)
            if p is None:
                continue
            uv[f] = p
            ob = track.path.get(f)
            if ob is not None and math.hypot(ob[0] - p[0], ob[1] - p[1]) <= cfg.flight_inlier_px * s:
                agree.append(f)
        if len(agree) < cfg.flight_min_agree:
            continue
        spans.update(range(int(fl.frames[0]), int(fl.frames[1]) + 1))
        for f, p in uv.items():
            i, j = bisect.bisect_right(agree, f) - 1, bisect.bisect_left(agree, f)
            if i < 0 or j >= len(agree) or f - agree[i] > cfg.flight_bracket or agree[j] - f > cfg.flight_bracket:
                continue
            if not gate(f) or not _placeable(p, size, cfg):
                continue
            confirmed = agree[i] == f
            own, far = _scores_near(cache.balls.get(f), p, cfg.near_px * s, cfg.far_px * s)
            out.append(Candidate(f, p, round(cfg.max_weight * (1.0 if confirmed else cfg.filled_share), 4), "flight",
                                 "confirmed" if confirmed else "filled", round(1.0 - min(1.0, own) + far, 4), k))
    return out, spans


def path_candidates(track: BallTrack, camera: CameraTrack, cache: DetectionCache, gate: Callable[[int], bool],
                    cfg: PseudoConfig, skip: set[int] | None = None) -> list[Candidate]:
    """Labels from points of the linked path whose motion the path itself confirms (see the module docstring);
    frames in ``skip`` (those of the flights used) are left to the flights."""
    size = camera.image_size
    s = size[1] / 1080.0
    skip = skip or set()
    frames = np.array(sorted(track.path), np.int64)
    if len(frames) < 2 * cfg.motion_min_side + 1:
        return []
    pts = np.array([track.path[int(f)][:3] for f in frames], float)
    cams = [camera.get(int(f)) for f in frames]
    has = np.array([c is not None for c in cams])
    d = np.zeros((len(frames), 3))
    for i, cam in enumerate(cams):
        if cam is not None:
            d[i] = _ray(pts[i], cam, size)
    out: list[Candidate] = []
    for i, f in enumerate(frames.tolist()):
        cam = cams[i]
        if cam is None or f in skip or not gate(f):
            continue
        own = float(pts[i, 2])
        lo = int(np.searchsorted(frames, f - cfg.motion_window))
        hi = int(np.searchsorted(frames, f + cfg.motion_window, side="right"))
        nb = np.array([j for j in range(lo, hi) if j != i and has[j]], int)
        if len(nb) < 2 * cfg.motion_min_side:
            continue
        before, after = nb[frames[nb] < f], nb[frames[nb] > f]
        if len(before) < cfg.motion_min_side or len(after) < cfg.motion_min_side:
            continue
        anchored = pts[before, 2].max() >= cfg.min_score and pts[after, 2].max() >= cfg.min_score
        if own < cfg.min_score and not anchored:
            continue
        dt = (frames[nb] - f).astype(float)
        A = np.column_stack([np.ones_like(dt), dt, dt * dt])
        coef, *_ = np.linalg.lstsq(A, d[nb], rcond=None)
        if np.linalg.norm(A @ coef - d[nb], axis=1).max() * cam.f > cfg.motion_tol_px * s:
            continue  # the neighbours themselves do not move smoothly
        pred = _to_px(coef[0], cam, size)
        if pred is None or math.hypot(pred[0] - pts[i, 0], pred[1] - pts[i, 1]) > cfg.motion_tol_px * s:
            continue
        lo = int(np.searchsorted(frames, f - cfg.static_window))
        hi = int(np.searchsorted(frames, f + cfg.static_window, side="right"))
        win = d[lo:hi][has[lo:hi]]
        if np.arccos(np.clip(win @ d[i], -1.0, 1.0)).max() * cam.f < cfg.static_px * s:
            continue  # it never moves: not the game ball
        xy = (float(pts[i, 0]), float(pts[i, 1]))
        if not _placeable(xy, size, cfg):
            continue
        _, far = _scores_near(cache.balls.get(f), xy, cfg.near_px * s, cfg.far_px * s)
        scored = own >= cfg.min_score
        if scored and far >= cfg.ambiguity * own:
            continue  # another ball-like thing scores nearly as high
        share = cfg.scored_share if scored else cfg.anchored_share
        out.append(Candidate(f, xy, round(cfg.max_weight * share, 4), "path", "scored" if scored else "anchored",
                             round(1.0 - min(1.0, own) + far, 4)))
    return out


def select_targets(cands: Sequence[Candidate], cfg: PseudoConfig, fps: float) -> list[Candidate]:
    """A bounded, spread-out set (see the module docstring): one candidate per frame (the higher weight), minutes
    of video taking turns, each offering its hardest remaining candidate that keeps ``min_spacing`` to every
    chosen frame and ``per_flight``; at most ``per_minute`` per minute and ``max_frames`` in all."""
    best: dict[int, Candidate] = {}
    for c in cands:
        if c.frame not in best or c.weight > best[c.frame].weight:
            best[c.frame] = c
    bins: dict[int, list[Candidate]] = defaultdict(list)
    for c in best.values():
        bins[int(c.frame // (60.0 * fps))].append(c)
    for lst in bins.values():  # ties in a scrambled but fixed order, not from the start of the minute
        lst.sort(key=lambda c: (-c.hardness, (c.frame * 2654435761) % 4294967296))
    order = sorted(bins)
    pos = dict.fromkeys(order, 0)
    per_bin: Counter[int] = Counter()
    per_flight: Counter[int] = Counter()
    taken: list[int] = []
    chosen: list[Candidate] = []
    progress = True
    while progress and len(chosen) < cfg.max_frames:
        progress = False
        for b in order:
            if len(chosen) >= cfg.max_frames:
                break
            if per_bin[b] >= cfg.per_minute:
                continue
            lst = bins[b]
            while pos[b] < len(lst):
                c = lst[pos[b]]
                pos[b] += 1
                if c.flight is not None and per_flight[c.flight] >= cfg.per_flight:
                    continue
                k = bisect.bisect_left(taken, c.frame)
                if (k > 0 and c.frame - taken[k - 1] < cfg.min_spacing) or \
                        (k < len(taken) and taken[k] - c.frame < cfg.min_spacing):
                    continue
                taken.insert(k, c.frame)
                chosen.append(c)
                per_bin[b] += 1
                if c.flight is not None:
                    per_flight[c.flight] += 1
                progress = True
                break
    return sorted(chosen, key=lambda c: c.frame)


def track_matches_cache(track: BallTrack, cache: DetectionCache, n: int = 500) -> float:
    """Share of (a sample of) the ball path's points that are candidates of the detection cache: ~1 when
    ``ball_track.json`` was linked from this cache."""
    frames = sorted(track.path)
    if not frames:
        return 1.0
    sample = frames[:: max(1, len(frames) // n)]
    hit = 0
    for f in sample:
        b = cache.balls.get(f)
        x, y = track.path[f][:2]
        if b is not None and len(b) and float(np.hypot(b[:, 0] - x, b[:, 1] - y).min()) <= 0.05:
            hit += 1
    return hit / len(sample)


def resolve_detections(root: Path, track: BallTrack, det_dir: str | None = None,
                       ball_track: str = "ball_track.json") -> tuple[str, DetectionCache]:
    """The detection folder the ball track was linked from, and its cache. ``det_dir`` None tries the folder
    ``ball_track.loop.json`` names (courtvision/nba/loop.py stamps it), then ``detections``, then the other
    ``detections*`` folders; a folder counts when ``track_matches_cache`` is at least 0.9."""
    names = [det_dir] if det_dir is not None else []
    if det_dir is None:
        try:
            stamp = json.loads((root / "ball_track.loop.json").read_text()) if ball_track == "ball_track.json" else {}
        except (OSError, ValueError):
            stamp = {}
        if isinstance(stamp, dict) and isinstance(stamp.get("det_dir"), str):
            names.append(stamp["det_dir"])
        names += ["detections"] + sorted(p.name for p in root.glob("detections*") if p.is_dir())
    tried = []
    for name in dict.fromkeys(names):
        if not (root / name).is_dir():
            continue
        cache = load_cache(root / name)
        share = track_matches_cache(track, cache) if cache.balls else 0.0
        if share >= 0.9:
            return name, cache
        tried.append(f"{name}: {share:.0%}")
    if not tried:
        raise InputError(f"{root}: no detection folder {names} (tools/nba_analyze.py {root} detect)")
    raise InputError(f"{root / ball_track} was not linked from {', '.join(tried)} of its points are candidates "
                     f"there: run tools/nba_analyze.py {root} ball --det-dir <folder>, or name it with --det-dir")


def build_candidates(root: Path, cfg: PseudoConfig, det_dir: str | None = None,
                     ball_track: str = "ball_track.json") -> tuple[list[Candidate], dict[str, Any]]:
    """Every labelable frame of the game folder ``root`` -> (candidates, context for ``write_folder``)."""
    parts = sorted((root / "camera_parts").glob("part*.jsonl"))
    if not parts:
        raise InputError(f"{root}/camera_parts/part*.jsonl: no camera track (tools/nba_track_court.py)")
    camera = CameraTrack.load(parts)
    track = BallTrack.load(root / ball_track)
    det_dir, cache = resolve_detections(root, track, det_dir, ball_track)
    bug = root / "scorebug.jsonl"
    live = scorebug_live(bug, cfg.bug_tol_s) if cfg.bug_check and bug.is_file() else None
    gate = FrameGate(camera.params.keys(), load_cuts(parts), cfg.cut_margin, live, camera.fps)
    fc, spans = flight_candidates(track, camera, cache, gate, cfg)
    pc = path_candidates(track, camera, cache, gate, cfg, skip=spans)
    ctx = {"camera": camera, "cache": cache, "det_dir": det_dir, "bug_check": live is not None,
           "n_flights": len(track.flights), "n_path": len(track.path)}
    return fc + pc, ctx


# --------------------------------------------------------------------------- output


def _clips(targets: Sequence[Candidate], context: int) -> list[list[int]]:
    """Runs of consecutive frames: each target with its ``context`` predecessors (touching runs merge)."""
    clips: list[list[int]] = []
    for c in sorted(targets, key=lambda c: c.frame):
        fr = list(range(c.frame - context, c.frame + 1))
        if clips and fr[0] <= clips[-1][-1] + 1:
            clips[-1].extend(f for f in fr if f > clips[-1][-1])
        else:
            clips.append(fr)
    return clips


def write_folder(out: Path, video: Path, targets: Sequence[Candidate], cache: DetectionCache, cfg: PseudoConfig,
                 meta: dict[str, Any], progress: bool = True) -> dict[str, int]:
    """Read ``video`` once and write the teacher-format folder (see the module docstring) -> counts."""
    by_frame = {c.frame: c for c in targets}
    clips = _clips(targets, cfg.context)
    where = {f: ci for ci, fr in enumerate(clips) for f in fr}
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({**meta, "complete": False}, indent=1))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or float(meta.get("fps") or 29.97)
    expect = tuple(meta["image_size"]) if meta.get("image_size") else None
    saved: dict[int, tuple[int, int]] = {}  # frame -> its (w, h)
    first, last = min(where), max(where)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    t0 = time.time()
    for idx in range(first, last + 1):
        if idx not in where:
            if not cap.grab():
                break
            continue
        ok, img = cap.read()
        if not ok:
            break
        h, w = img.shape[:2]
        if expect is not None and (w, h) != expect:
            cap.release()
            raise InputError(f"{video} is {w}x{h}, the camera track {expect[0]}x{expect[1]}: labels would be misplaced")
        small = cv2.resize(img, (cfg.save_width, round(h * cfg.save_width / w)), interpolation=cv2.INTER_AREA)
        d = out / "frames" / f"{where[idx]:04d}"
        d.mkdir(exist_ok=True)
        cv2.imwrite(str(d / f"{idx:06d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved[idx] = (w, h)
        if progress and len(saved) % 1000 == 0:
            print(f"{len(saved)} frames written (frame {idx}, {time.time() - t0:.0f}s)", flush=True)
    cap.release()
    det_rows: list[dict[str, Any]] = []
    lab_rows: list[dict[str, Any]] = []
    n_clips = dropped = 0
    for ci, fr in enumerate(clips):
        if not all(f in saved for f in fr):  # the video ended early or a frame did not decode
            shutil.rmtree(out / "frames" / f"{ci:04d}", ignore_errors=True)
            dropped += 1
            continue
        n_clips += 1
        for f in fr:
            w, h = saved[f]
            r = BALL_RADIUS_1080 * h / 1080.0
            balls = cache.balls.get(f, np.zeros((0, 3), np.float32))
            persons = cache.persons.get(f, np.zeros((0, 5), np.float32))
            row: dict[str, Any] = {"clip": ci, "frame": f, "t": round(f / fps, 4), "size": [w, h],
                                   "persons": [[round(float(v), 1) for v in p[:4]] + [round(float(p[4]), 3)]
                                               for p in persons],
                                   "balls": [[round(float(x - r), 1), round(float(y - r), 1), round(float(x + r), 1),
                                              round(float(y + r), 1), round(float(sc), 3)] for x, y, sc in balls]}
            c = by_frame.get(f)
            if c is None:
                row["context"] = True
                lab_rows.append({"clip": ci, "frame": f, "xy": None, "r": None, "w": 0.0, "source": "none"})
            else:
                lab_rows.append({"clip": ci, "frame": f, "xy": [round(c.xy[0], 2), round(c.xy[1], 2)],
                                 "r": round(r, 1), "w": c.weight, "source": f"{c.source}:{c.kind}",
                                 "hardness": c.hardness})
            det_rows.append(row)
    for name, rows in (("detections.jsonl", det_rows), ("ball_labels.jsonl", lab_rows)):
        tmp = out / (name + ".tmp")
        tmp.write_text("".join(json.dumps(x) + "\n" for x in rows))
        os.replace(tmp, out / name)
    counts = {"clips": n_clips, "clips_dropped": dropped, "frames": len(det_rows),
              "labelled": sum(1 for x in lab_rows if x["w"] > 0)}
    (out / "meta.json").write_text(json.dumps({**meta, "counts": {**meta.get("counts", {}), **counts},
                                               "complete": True}, indent=1))
    return counts


def _prepare_out(out: Path) -> None:
    """An empty folder, or an earlier pseudo folder (cleared); anything else is refused."""
    if not out.exists() or not any(out.iterdir()):
        return
    try:
        kind = json.loads((out / "meta.json").read_text()).get("kind")
    except (OSError, ValueError):
        kind = None
    if kind != "pseudo":
        raise InputError(f"{out} exists and is not a pseudo-label folder: choose another --out")
    shutil.rmtree(out / "frames", ignore_errors=True)
    for name in ("detections.jsonl", "ball_labels.jsonl", "meta.json"):
        (out / name).unlink(missing_ok=True)


def _counts(cands: Iterable[Candidate]) -> dict[str, int]:
    return dict(sorted(Counter(f"{c.source}:{c.kind}" for c in cands).items()))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="game folder (game.json, camera_parts/, detections/, ball_track.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=PseudoConfig.max_frames,
                    help="labelled frames (each also stores its two previous frames)")
    ap.add_argument("--min-score", type=float, default=PseudoConfig.min_score)
    ap.add_argument("--per-minute", type=int, default=PseudoConfig.per_minute)
    ap.add_argument("--det-dir", default=None,
                    help="the detection folder ball_track.json was linked from (default: found, see resolve_detections)")
    ap.add_argument("--ball-track", default="ball_track.json")
    ap.add_argument("--max-weight", type=float, default=PseudoConfig.max_weight)
    ap.add_argument("--person-weight", type=float, default=PseudoConfig.person_weight)
    ap.add_argument("--save-width", type=int, default=PseudoConfig.save_width)
    ap.add_argument("--no-bug-check", action="store_true", help="also label frames without the scorebug")
    ap.add_argument("--dry-run", action="store_true", help="select and report; write nothing, read no video")
    args = ap.parse_args(argv)
    if not 0.0 < args.max_weight <= PSEUDO_MAX_WEIGHT:
        ap.error(f"--max-weight must be in (0, {PSEUDO_MAX_WEIGHT}]: pseudo labels stay below the teacher's")
    if not 0.0 <= args.person_weight <= args.max_weight:
        ap.error("--person-weight must be in [0, --max-weight]")
    cfg = PseudoConfig(min_score=args.min_score, max_frames=args.max_frames, per_minute=args.per_minute,
                       max_weight=args.max_weight, person_weight=args.person_weight, save_width=args.save_width,
                       bug_check=not args.no_bug_check)
    try:
        return _run(Path(args.root), Path(args.out), cfg, args)
    except (InputError, FileNotFoundError) as e:
        print(f"pseudo_labels: {e}", file=sys.stderr)
        return 2


def _run(root: Path, out: Path, cfg: PseudoConfig, args: argparse.Namespace) -> int:
    t0 = time.time()
    cands, ctx = build_candidates(root, cfg, args.det_dir, args.ball_track)
    camera: CameraTrack = ctx["camera"]
    targets = select_targets(cands, cfg, camera.fps)
    minutes = Counter(int(c.frame // (60 * camera.fps)) for c in targets)
    det_dir = ctx["det_dir"]
    print(f"{det_dir}: ball path {ctx['n_path']} frames, {ctx['n_flights']} flights; candidates {_counts(cands)}; "
          f"selected {len(targets)} {_counts(targets)} over {len(minutes)} minutes "
          f"(scorebug check {'on' if ctx['bug_check'] else 'off'}; {time.time() - t0:.0f}s)", flush=True)
    if args.dry_run or not targets:
        return 0 if targets else 1
    game = json.loads((root / "game.json").read_text())
    video = Path(game["video"])
    if not video.is_absolute() and not video.exists() and (root / video).exists():
        video = root / video
    _prepare_out(out)
    meta = {"kind": "pseudo", "version": 1, "label_origin": "pseudo", "rights_status": "research_only",
            "game_root": str(root.resolve()), "video": str(video.resolve()), "det_dir": det_dir,
            "detections": sorted(p.name for p in (root / det_dir).glob("*.npz")),
            "ball_track": args.ball_track, "fps": camera.fps, "image_size": list(camera.image_size),
            "max_weight": cfg.max_weight, "person_weight": cfg.person_weight, "config": asdict(cfg),
            "counts": {"candidates": _counts(cands), "selected": _counts(targets)},
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    counts = write_folder(out, video, targets, ctx["cache"], cfg, meta)
    print(f"{out}: {counts['labelled']} labelled frames in {counts['clips']} clips, {counts['frames']} frames "
          f"({counts['clips_dropped']} clips dropped; {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
