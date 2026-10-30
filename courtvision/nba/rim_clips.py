"""Rim clips: short runs of frames cut around a hoop so the rim is always the same size and in the same place.

A made shot is easiest to see at the hoop itself: the ball drops through the ring and the net, where a miss bounces
off the ring or the board and leaves upwards or sideways. The camera track (``analyze.CameraTrack``) projects each
rim into every frame of the game camera, so a crop can follow the rim through pans and zooms: ``CLIP_RIMS`` rim widths
square, its centre ``BELOW_RIMS`` rim widths below the rim's centre (the ball's arrival above and the net below are
both inside), resized to ``CLIP_PX``. A rim seen 30 px wide in 720p and 45 px wide in 1080p gives the same crop.

``clip_at`` cuts ``CLIP_OFFSETS`` (every other frame from 1 s before to 1.4 s after a moment, at 29.97 fps; scaled to
other rates) around one rim. Frames where the camera is lost (replays, close-ups) take the rim box of the nearest
frame that has one, within ``HOLD_S``; beyond that they are blank and marked invalid. ``rim_model.py`` learns made /
missed / no shot from these clips.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from courtvision.types import BBox

from .analyze import rim_boxes

__all__ = ["BELOW_RIMS", "CLIP_OFFSETS", "CLIP_PX", "CLIP_RIMS", "HOLD_S", "SERIES_SPAN", "ZONES", "RimClip",
           "ball_zone_series", "clip_at", "clip_frames", "zone_scores",
           "crop_rim", "rim_box_near"]

CLIP_OFFSETS = tuple(range(-30, 43, 2))  # frames at 29.97 fps: 37 frames, -1.0 s to +1.4 s, every other frame
CLIP_PX = 96  # side of a cut frame
CLIP_RIMS = 4.8  # side of the cut in rim widths
BELOW_RIMS = 0.5  # the cut's centre this many rim widths below the rim's centre
HOLD_S = 0.5  # a frame without a camera takes the rim box of a frame at most this far away


@dataclass
class RimClip:
    """One clip: ``frames`` (T, CLIP_PX, CLIP_PX, 3) uint8 BGR, ``valid`` (T,) whether each had a rim box, the frame
    it is centred on and the rim's side."""

    frames: np.ndarray
    valid: np.ndarray
    centre: int
    side: str


def crop_rim(img: np.ndarray, box: BBox, px: int = CLIP_PX, rims: float = CLIP_RIMS,
             below: float = BELOW_RIMS) -> np.ndarray:
    """The square cut about the rim ``box`` (``rims`` rim widths, centred ``below`` rim widths under the rim's
    centre), resized to ``px``; parts outside the image are black."""
    rw = max(float(box.x2 - box.x1), 1.0)
    side = rims * rw
    cx, cy = (box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0 + below * rw
    s = px / side
    m = np.array([[s, 0.0, px / 2.0 - s * cx], [0.0, s, px / 2.0 - s * cy]], np.float32)
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    return cv2.warpAffine(img, m, (px, px), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def clip_frames(centre: int, fps: float, offsets: Sequence[int] = CLIP_OFFSETS) -> list[int]:
    """The video frames of a clip centred on ``centre`` (``offsets`` are at 29.97 fps)."""
    return [centre + round(o * fps / 29.97) for o in offsets]


def rim_box_near(camera: Any, court: Any, frame: int, side: str, hold: int) -> BBox | None:
    """``side``'s rim box on ``frame``, else on the nearest frame within ``hold`` frames that has one."""
    size = camera.image_size
    for d in range(hold + 1):
        for f in ((frame,) if d == 0 else (frame - d, frame + d)):
            cam = camera.get(f)
            if cam is not None:
                box = rim_boxes(cam, court, size).get(side)
                if box is not None:
                    return box
    return None


def clip_at(frames: Any, camera: Any, court: Any, centre: int, side: str,
            offsets: Sequence[int] = CLIP_OFFSETS) -> RimClip | None:
    """The clip of ``side``'s rim centred on frame ``centre`` (``frames.get(f)``: the video's frame ``f``, as
    ``rim_jev.Frames``). None when no frame of it has a rim box, or a frame cannot be read."""
    fs = clip_frames(centre, camera.fps, offsets)
    hold = max(1, round(HOLD_S * camera.fps))
    boxes = [rim_box_near(camera, court, f, side, hold) for f in fs]
    if not any(b is not None for b in boxes):
        return None
    out = np.zeros((len(fs), CLIP_PX, CLIP_PX, 3), np.uint8)
    valid = np.zeros(len(fs), bool)
    for i, (f, box) in enumerate(zip(fs, boxes, strict=True)):
        if box is None:
            continue
        img = frames.get(f)
        if img is None:
            return None
        out[i] = crop_rim(img, box)
        valid[i] = True
    return RimClip(out, valid, centre, side)


# --------------------------------------------------------------------------- the ball about the rim, from detections

# Zones about a rim in rim widths (x across, y down; the rim's centre at 0, 0): name, x range of |dx|, y range
ZONES = (("above", (0.0, 0.6), (-2.2, -0.25)),  # coming down onto the ring
         ("ring", (0.0, 0.8), (-0.25, 0.3)),  # at the ring
         ("net", (0.0, 0.5), (0.3, 1.7)),  # inside the net, under the ring
         ("under", (0.5, 2.4), (0.3, 2.9)),  # below, outside the ring's span
         ("side", (0.8, 2.4), (-1.2, 0.3)),  # beside the ring (off the rim, off the board)
         ("high", (0.0, 2.4), (-4.0, -2.2)))  # well above (a rebound bouncing high, the shot's arc)
SERIES_SPAN = (-30, 42)  # frames at 29.97 fps about the centre: -1.0 s to +1.4 s


def zone_scores(balls: np.ndarray | None, box: BBox, min_score: float = 0.0) -> np.ndarray:
    """The best score of the ball candidates ``balls`` ((n, 3) x, y, score, image px) in each of ``ZONES`` about the
    rim ``box``, 0 where none; candidates below ``min_score`` are ignored."""
    out = np.zeros(len(ZONES), np.float32)
    if balls is None or not len(balls):
        return out
    b = balls[balls[:, 2] >= min_score] if min_score > 0 else balls
    if not len(b):
        return out
    rw = max(float(box.x2 - box.x1), 1.0)
    dx = np.abs(b[:, 0] - (box.x1 + box.x2) / 2.0) / rw
    dy = (b[:, 1] - (box.y1 + box.y2) / 2.0) / rw
    for z, (_, (x0, x1), (y0, y1)) in enumerate(ZONES):
        m = (dx >= x0) & (dx < x1) & (dy >= y0) & (dy < y1)
        if m.any():
            out[z] = float(b[m, 2].max())
    return out


def ball_zone_series(balls: Any, camera: Any, court: Any, centre: int, side: str,
                     span: tuple[int, int] = SERIES_SPAN, min_score: float = 0.0) -> np.ndarray:
    """Per frame of ``span`` about ``centre`` (every frame, rate-scaled), the best ball score of the detector in each of
    ``ZONES`` of ``side``'s rim (``balls``: frame -> (n, 3) x, y, score, as ``analyze.DetectionCache.balls``; scores
    below ``min_score`` ignored): (T, len(ZONES)), 0 where none; a frame without a rim box is all -1."""
    r = camera.fps / 29.97
    fs = range(centre + round(span[0] * r), centre + round(span[1] * r) + 1)
    hold = max(1, round(HOLD_S * camera.fps))
    out = np.zeros((len(fs), len(ZONES)), np.float32)
    for i, f in enumerate(fs):
        box = rim_box_near(camera, court, f, side, hold)
        if box is None:
            out[i] = -1.0
            continue
        out[i] = zone_scores(balls.get(f), box, min_score)
    return out
