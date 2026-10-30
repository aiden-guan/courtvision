"""Synthetic benchmark of automatic broadcast court registration across NBA arenas (research use).

    .venv/bin/python tools/court_bench.py --out runs/court_bench/<tag>.jsonl [--modes single,center,track]
        [--families espn,geometry,...] [--n 12] [--games 2] [--runs 1] [--workers 4] [--seed 0]
    .venv/bin/python tools/court_bench.py --summary runs/court_bench/base.jsonl [runs/court_bench/new.jsonl]
    .venv/bin/python tools/court_bench.py --show mixed --n 4 --frames-dir /tmp/frames   # sample frames (JPEG)

Every frame is drawn from a known game camera (``court_search.CameraParams``: centre, pan, tilt, roll, focal), so
each result is scored against the truth: the RMS reprojection error of a 2-ft grid of floor points in view, in
1080-line pixels. A camera is **correct** within ``--tol`` px (default 4).

Arenas (``sample_arena``): light, medium or dark wood with strip and board grain; white, dark or coloured lines;
painted keys, a stain inside the 3-point line, a painted or dark out-of-bounds border, a filled centre circle; a
large centre-court logo and baseline / sideline wordmarks; lane space marks and the dashed half of the free-throw
circles (real markings the court model does not hold); glare blobs and the far stands mirrored in the floor. Around
the floor: LED boards, tiered stands of seated fans, courtside fans at the near sideline, basket stanchions and
backboards. Per frame: players and referees, motion blur, a scorebug and sponsor graphics, exposure, lens
softness, noise and JPEG compression, at 720p, 1080p or 4K.

Cameras: centre 45-130 ft behind the near sideline, 18-60 ft up (elevation to the court centre 8-42 deg), up to
+-10 ft from half court; horizontal field of view 12-60 deg; aimed across the play; roll within a degree.

Modes:

* ``single``: ``calibrate_frame`` from the prior on one frame (the evidence at the 960-px work width, as
  ``estimate_center`` does). **accepted** = passes ``estimate_center``'s gate (quality and ``plausible``).
* ``center``: ``estimate_center`` on ``--frames`` game frames of one game plus a close-up and a replay. Scored by
  the cameras it returns and by the centre itself: the worst RMS error, over 9 views across the court, of the best
  camera at the estimated centre (``center_fit``: the error tracking from that centre cannot go below).
* ``track``: ``BroadcastCourtTracker`` (the true centre) through a play of ~9 s at 30 fps with a flagged false cut,
  a camera flash, a fast pan, a cut to a close-up and back, and a replay from another camera and back. Cut flags
  come from ``ShotBoundaryDetector`` as in ``tools/nba_track_court.py``, plus the scripted false cut.

Scenario families (``FAMILIES``) change one factor at a time from an ESPN / Chase-Center-like reference
(``espn``: the arena the defaults were tuned on); ``mixed`` draws everything at random. Renders are design choices,
not measurements of real arenas: the benchmark ranks variants, real footage decides.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if "--workers" in sys.argv:  # one BLAS / OpenCV thread per worker process (set before numpy loads)
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))  # appended: a PYTHONPATH copy of the package wins

from courtvision.geometry.court import court_for  # noqa: E402
from courtvision.perception.broadcast import ShotBoundaryDetector  # noqa: E402
from courtvision.perception.broadcast_court import (  # noqa: E402
    BroadcastCourtTracker,
    BroadcastTrackerConfig,
    estimate_center,
)
from courtvision.perception.court_search import (  # noqa: E402
    CameraParams,
    EvidenceConfig,
    calibrate_frame,
    court_evidence,
    plausible,
)

COURT = court_for("nba")
L, W = COURT.length, COURT.width
ESPN_EXCLUDE = [(468, 944, 1000, 125)]  # tools/nba_track_court.py: the ESPN bug and its lower row (1920x1080 px)
SIZES = {"720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}

# --------------------------------------------------------------------------- colours (BGR)

TEAM = {
    "navy": (90, 40, 20), "royal": (170, 75, 20), "red": (40, 35, 200), "wine": (55, 20, 120),
    "purple": (125, 40, 85), "gold": (40, 175, 235), "green": (55, 115, 25), "black": (28, 28, 28),
    "orange": (25, 110, 235), "teal": (135, 125, 0), "silver": (165, 165, 165), "skyblue": (215, 160, 95),
    "maroon": (40, 25, 110), "white": (238, 238, 238),
}
WOOD = {"light": (132, 184, 222), "medium": (84, 134, 186), "dark": (48, 80, 124), "grey": (112, 116, 120)}
SKIN = [(60, 90, 140), (85, 125, 185), (120, 160, 215), (40, 62, 95), (100, 140, 200)]


def _team(rng: np.random.Generator, avoid: tuple[str, ...] = ()) -> tuple[int, int, int]:
    names = [k for k in TEAM if k not in avoid]
    return TEAM[names[int(rng.integers(len(names)))]]


# --------------------------------------------------------------------------- scene description


@dataclass
class Arena:
    """One arena's floor and surroundings (deterministic for its ``seed``)."""

    seed: int
    wood: tuple[int, int, int] = WOOD["light"]
    grain: float = 0.5  # strength of the strip / board / grain texture (0-1.5)
    line: tuple[int, int, int] = (90, 40, 20)
    line_ft: float = 2.0 / 12.0
    key: tuple[int, int, int] | None = None  # painted lane
    key_alpha: float = 0.9
    key_lines: tuple[int, int, int] | None = None  # lane lines in another colour (e.g. the paint's: a step edge)
    stain: tuple[int, int, int] | None = None  # tinted wood inside the 3-point line
    stain_alpha: float = 0.35
    border: tuple[int, int, int] | None = None  # painted out-of-bounds floor
    border_lines: bool = True  # boundary lines drawn (False: the border's edge is the line)
    circle: tuple[int, int, int] | None = None  # filled centre circle
    logo: dict[str, Any] | None = None  # centre-court logo: r (ft), c1, c2, text
    words: list[dict[str, Any]] = field(default_factory=list)  # wordmarks: text, x, y, h, rot, color, stroke
    marks: bool = True  # lane space marks and the dashed half of the free-throw circles
    division_gap: bool = False  # the division line is not painted across the logo
    apron: tuple[float, float, float] = (8.0, 9.0, 10.0)  # floor beyond the near / far sideline, the baselines
    crowd: tuple[int, int, int] = (55, 50, 50)  # stands background
    stands: str = "dense"  # "dense": a full, lit crowd; "dark": dim, half-empty stands
    crowd_colors: list[tuple[int, int, int]] = field(default_factory=list)  # shirts
    led: list[tuple[int, int, int]] = field(default_factory=list)  # LED board panel colours
    baskets: bool = True
    glare: int = 1  # specular blobs on the floor
    reflect: float = 0.1  # strength of the far stands mirrored in the floor
    sheen: float = 0.0  # broad lighting gradient over the floor


@dataclass
class Look:
    """Per-frame clutter and photometry."""

    players: int = 8
    refs: int = 2
    blur_px: float = 0.0  # motion blur length (1080-line px)
    blur_deg: float = 0.0
    overlays: list[str] = field(default_factory=list)
    noise: float = 2.0
    jpeg: int = 88
    gain: float = 1.0
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    vignette: float = 0.1
    flash: float = 0.0
    soft: float = 0.6  # lens softness (Gaussian sigma, 1080-line px)


# --------------------------------------------------------------------------- deterministic hashing


def _hash(seed: int, *ks: np.ndarray | int) -> np.ndarray:
    """Pseudo-random floats in [0, 1) from integer keys (splitmix64)."""
    h = np.uint64(seed * 0x9E3779B97F4A7C15 % (1 << 64))
    out = np.full(np.broadcast_shapes(*[np.shape(k) for k in ks]) if ks else (), h, np.uint64)
    with np.errstate(over="ignore"):
        for i, k in enumerate(ks):
            out = out ^ (np.asarray(k).astype(np.int64).astype(np.uint64) + np.uint64(0x632BE59BD9B4E019 * (i + 1)
                                                                                     % (1 << 64)))
            out = out * np.uint64(0xBF58476D1CE4E5B9)
            out = out ^ (out >> np.uint64(31))
            out = out * np.uint64(0x94D049BB133111EB)
            out = out ^ (out >> np.uint64(29))
    return (out >> np.uint64(11)).astype(np.float64) / float(1 << 53)


# --------------------------------------------------------------------------- projection helpers


def _P(cam: CameraParams, size: tuple[int, int]) -> np.ndarray:
    R = cam.R()
    return cam.K(size) @ np.column_stack([R, -R @ np.asarray(cam.center, float)])


def _proj3(P: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = X @ P[:, :3].T + P[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        return h[:, :2] / h[:, 2:3], h[:, 2]


def _rays(cam: CameraParams, size: tuple[int, int], uv: np.ndarray) -> np.ndarray:
    """World directions of the rays through image points (N, 2)."""
    Kinv = np.linalg.inv(cam.K(size))
    p = np.column_stack([uv, np.ones(len(uv))]) @ Kinv.T
    return p @ cam.R()  # R^T p as rows


def _plane_H(cam: CameraParams, size: tuple[int, int], origin: np.ndarray, au: np.ndarray, av: np.ndarray,
             res: float) -> np.ndarray:
    """Texture pixel (c, r) -> image homography of the plane X = origin + c res au + r res av."""
    R = cam.R()
    M = np.column_stack([au * res, av * res, np.asarray(origin, float) - np.asarray(cam.center, float)])
    return cam.K(size) @ R @ M


def _front(cam: CameraParams, origin: np.ndarray, au: np.ndarray, av: np.ndarray, extent: tuple[float, float]) -> bool:
    """All four corners of a plane patch in front of the camera."""
    fwd = cam.R()[2]
    for a in (0.0, extent[0]):
        for b in (0.0, extent[1]):
            X = np.asarray(origin, float) + a * au + b * av
            if float(fwd @ (X - cam.center)) <= 1.0:
                return False
    return True


def _scale1080(size: tuple[int, int]) -> float:
    return size[1] / 1080.0


# --------------------------------------------------------------------------- textures


def _crowd_texture(arena: Arena, width_ft: float, height_ft: float, res: float, seed: int,
                   rows_ft: float = 2.8, seat_ft: float = 1.7, fill: float = 0.85, alpha: bool = False) -> np.ndarray:
    """Tiered stands of seated fans (row 0 at the bottom of the stands): seats with armrests, aisles and rails;
    fans with patterned shirts, arms, faces and hair, as busy in thin detail as a real crowd. ``arena.stands ==
    "dark"``: dim stands, half empty. ``alpha``: BGRA, transparent above the top row's seats except where its fans
    are (the front row of courtside fans, with the floor seen between their heads)."""
    rng = np.random.default_rng(seed)
    dark = arena.stands == "dark"
    w, h = int(width_ft / res), int(height_ft / res)
    tex = np.empty((h, w, 3), np.uint8)
    tex[:] = arena.crowd
    am = np.full((h, w), 255, np.uint8)
    colors = arena.crowd_colors or [(200, 200, 200), (40, 40, 40), (60, 60, 160), (150, 90, 40)]
    seat = tuple(int(v) for v in np.clip(np.asarray(arena.crowd, int) + 25, 0, 255))
    px = max(1, int(round(0.15 / res)))  # thin details: armrests, stripes, arms
    n_rows, n_seats = int(height_ft / rows_ft), int(width_ft / seat_ft)
    aisle = int(rng.integers(14, 24))
    if alpha:
        am[: max(0, h - int(((n_rows - 1) * rows_ft + 1.9) / res))] = 0
    if dark:
        fill *= 0.5
    for r in range(n_rows):
        y_base = h - int((r * rows_ft) / res)  # drawn upright: rows go up the image, flipped below
        top_row = alpha and r == n_rows - 1
        tex[max(0, y_base - int(0.25 / res)):y_base] = (35, 35, 40)  # riser edge
        if not dark and r % 9 == 4:  # a hand rail
            cv2.line(tex, (0, y_base - int(2.3 / res)), (w, y_base - int(2.3 / res)), (150, 150, 155), px)
        for k in range(n_seats):
            x = int((k + 0.5 + rng.normal(0, 0.08)) * seat_ft / res)
            if k % aisle == 0:  # stairs of an aisle
                cv2.rectangle(tex, (x - int(0.5 * seat_ft / res), y_base - int(rows_ft / res)),
                              (x + int(0.5 * seat_ft / res), y_base), (95, 95, 100), -1)
                cv2.line(tex, (x - int(0.5 * seat_ft / res), y_base - int(0.5 * rows_ft / res)),
                         (x + int(0.5 * seat_ft / res), y_base - int(0.5 * rows_ft / res)), (150, 150, 150), px)
                continue
            sw = int(0.45 * seat_ft / res)
            cv2.rectangle(tex, (x - sw, y_base - int(1.9 / res)), (x + sw, y_base - int(0.3 / res)), seat, -1)
            cv2.rectangle(tex, (x - sw, y_base - int(1.0 / res)), (x - sw + px, y_base - int(0.3 / res)),
                          (20, 20, 22), -1)  # armrest
            if rng.random() > fill:
                continue
            body = colors[int(rng.integers(len(colors)))]
            body = tuple(int(np.clip(v * rng.uniform(0.7, 1.15), 0, 255)) for v in body)
            skin = SKIN[int(rng.integers(len(SKIN)))]
            bw, bh = int(0.5 * seat_ft / res), int(1.5 / res)
            top = y_base - int(0.35 / res) - bh
            cv2.rectangle(tex, (x - bw, top), (x + bw, y_base - int(0.35 / res)), body, -1)
            pat = rng.random()
            if pat < 0.3:  # stripes
                c2 = colors[int(rng.integers(len(colors)))]
                for yy in range(top + px, y_base - int(0.35 / res), 3 * px):
                    cv2.line(tex, (x - bw, yy), (x + bw, yy), c2, px)
            elif pat < 0.6:  # a logo or number
                c2 = (240, 240, 240) if sum(body) < 380 else (25, 25, 25)
                cv2.rectangle(tex, (x - bw // 3, top + bh // 4), (x + bw // 3, top + bh // 2), c2, -1)
            for side in (-1, 1):  # arms
                ax = x + side * (bw - px)
                c = skin if rng.random() < 0.5 else tuple(int(v * 0.8) for v in body)
                cv2.rectangle(tex, (ax - px // 2, top + px), (ax + px // 2 + 1, top + int(0.8 * bh)), c, -1)
            if rng.random() < 0.15:  # a raised hand, a phone
                hx = x + int(rng.uniform(-bw, bw))
                cv2.rectangle(tex, (hx, top - int(0.9 / res)), (hx + px, top), skin, -1)
            hr = max(1, int(0.33 / res))
            cv2.circle(tex, (x, top - hr + 1), hr, skin, -1)
            if rng.random() < 0.8:
                hair = (25, 25, 30) if rng.random() < 0.7 else (40, 80, 120)
                cv2.ellipse(tex, (x, top - hr - max(1, hr // 3)), (hr, max(1, int(hr * 0.6))), 0, 180, 360, hair, -1)
            if rng.random() < 0.3:  # a cap
                cv2.rectangle(tex, (x - hr, top - 2 * hr), (x + hr, top - 2 * hr + px), colors[0], -1)
            if top_row:
                cv2.rectangle(am, (x - bw, top), (x + bw, y_base - int(0.35 / res)), 255, -1)
                cv2.circle(am, (x, top - hr + 1), hr, 255, -1)
    noise = rng.integers(-10, 11, (h, w, 1)).astype(np.int16)
    tex = np.clip(tex.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if dark:
        tex = (tex.astype(np.float32) * 0.45).astype(np.uint8)
    if alpha:
        tex = np.dstack([tex, am])
    return np.ascontiguousarray(tex[::-1])  # row index grows up the stands


def _led_texture(arena: Arena, width_ft: float, height_ft: float, res: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w, h = int(width_ft / res), max(2, int(height_ft / res))
    tex = np.zeros((h, w, 3), np.uint8)
    colors = arena.led or [(30, 30, 200), (200, 200, 200), (20, 20, 20)]
    x = 0
    words = ["NBA", "TICKETS", "COURTSIDE", "BANK", "AIR", "CLOUD", "MOTORS", "ENERGY", "MOBILE", "SPORTS"]
    while x < w:
        pw = int(rng.uniform(8.0, 22.0) / res)
        c = colors[int(rng.integers(len(colors)))]
        tex[:, x:x + pw] = c
        txt = words[int(rng.integers(len(words)))]
        fg = (255, 255, 255) if sum(c) < 450 else (20, 20, 20)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        sc = 0.55 * h / max(th, 1)
        cv2.putText(tex, txt, (x + int(0.08 * pw), int(0.78 * h)), cv2.FONT_HERSHEY_DUPLEX, sc, fg,
                    max(1, int(sc * 2)), cv2.LINE_AA)
        x += pw
    return np.ascontiguousarray(tex[::-1])


@dataclass
class _Surround:
    """Per-arena textures of the stands, LED boards and courtside fans (cached)."""

    far_stands: np.ndarray
    end_stands: np.ndarray
    far_led: np.ndarray
    end_led: np.ndarray
    near: np.ndarray


_SURROUND: dict[int, _Surround] = {}
STAND_RES = 0.05  # ft per texel of the stands (fine enough for 2-in details)
_X0, _X1 = -70.0, L + 70.0  # extent of the far stands along the court length
_YS0, _YS1 = -50.0, W + 50.0  # extent of the end stands across


def _surround(arena: Arena) -> _Surround:
    if arena.seed not in _SURROUND:
        s = arena.seed
        _SURROUND.clear()  # one arena per worker at a time
        _SURROUND[s] = _Surround(
            far_stands=_crowd_texture(arena, _X1 - _X0, 110.0, STAND_RES, s + 1),
            end_stands=_crowd_texture(arena, _YS1 - _YS0, 110.0, STAND_RES, s + 2),
            far_led=_led_texture(arena, _X1 - _X0, 3.0, 0.05, s + 3),
            end_led=_led_texture(arena, _YS1 - _YS0, 3.0, 0.05, s + 4),
            near=_crowd_texture(arena, _X1 - _X0, 35.2, STAND_RES, s + 5, rows_ft=2.2, seat_ft=1.6, fill=0.95,
                                alpha=True),
        )
    return _SURROUND[arena.seed]


def _wood(arena: Arena, x0: float, y0: float, w: int, h: int, res: float) -> np.ndarray:
    """Hardwood: strips 2.25 in wide along the court length, boards 2-8 ft long, grain streaks (float32 BGR).
    A function of the court position and the arena seed only, so every frame shows the same floor."""
    s = arena.seed
    xs = (x0 + np.arange(w) * res).astype(np.float32)
    ys = y0 + np.arange(h) * res
    strip = np.floor(ys / 0.1875).astype(np.int64)
    js = np.arange(strip.min(), strip.max() + 1)
    jr = strip - js[0]
    tone_strip = (_hash(s, js) - 0.5).astype(np.float32)[jr]
    blen = (2.0 + 6.0 * _hash(s + 7, js)).astype(np.float32)[jr]
    boff = (_hash(s + 11, js) * 8.0).astype(np.float32)[jr]
    phase = (_hash(s + 17, js) * 2 * math.pi).astype(np.float32)[jr]
    b = np.floor((xs[None, :] + boff[:, None]) / blen[:, None]).astype(np.int64)
    b_lo, b_hi = int(b.min()), int(b.max())
    table = (_hash(s + 13, js[:, None], np.arange(b_lo, b_hi + 1)[None, :]) - 0.5).astype(np.float32)
    tone_board = table[jr[:, None], b - b_lo]
    grain = np.sin(xs[None, :] * np.float32(2 * math.pi / 3.1) + phase[:, None])
    grain += 0.5 * np.sin(xs[None, :] * np.float32(2 * math.pi / 0.83) + np.float32(2.3) * phase[:, None])
    d = np.abs(((ys + 0.09375) % 0.1875) - 0.09375)  # distance to the nearest strip edge (ft)
    seam = (np.clip(1.0 - d / max(res, 0.012), 0.0, 1.0) * min(1.0, 0.012 / res)).astype(np.float32)
    g = np.float32(arena.grain)
    v = g * (9.0 * tone_strip[:, None] + 7.0 * tone_board + 2.5 * grain - 10.0 * seam[:, None])
    v *= np.float32(1.0 / 160.0)
    v += np.float32(1.0)
    return v[..., None] * np.asarray(arena.wood, np.float32)[None, None, :]


def _poly(tex: np.ndarray, pts_ft: np.ndarray, x0: float, y0: float, res: float, color: tuple, alpha: float = 1.0,
          mask_only: bool = False) -> np.ndarray | None:
    """Fill a court-ft polygon on the texture (alpha blend, within its bounding box); returns the full-size mask
    when ``mask_only``."""
    q = (np.asarray(pts_ft, float) - [x0, y0]) / res
    th, tw = tex.shape[:2]
    c0, c1 = int(max(0, math.floor(q[:, 0].min()) - 1)), int(min(tw, math.ceil(q[:, 0].max()) + 2))
    r0, r1 = int(max(0, math.floor(q[:, 1].min()) - 1)), int(min(th, math.ceil(q[:, 1].max()) + 2))
    if mask_only:
        m = np.zeros((th, tw), np.uint8)
        c0, r0, c1, r1 = 0, 0, tw, th
    elif c0 >= c1 or r0 >= r1:
        return None
    else:
        m = np.zeros((r1 - r0, c1 - c0), np.uint8)
    qi = np.round((q - [c0, r0]) * 8.0)
    qi = np.clip(qi, -(2**28), 2**28).astype(np.int32)
    cv2.fillPoly(m, [qi], 255, cv2.LINE_AA, shift=3)
    if mask_only:
        return m
    a = (m.astype(np.float32) * np.float32(alpha / 255.0))[..., None]
    sub = tex[r0:r1, c0:c1]
    sub *= 1.0 - a
    sub += a * np.asarray(color, np.float32)
    return None


def _text_mask(text: str, h_px: float, stroke_px: float) -> np.ndarray:
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), base = cv2.getTextSize(text, font, 1.0, 1)
    sc = h_px / max(th, 1)
    t = max(1, int(round(stroke_px)))
    (tw, th), base = cv2.getTextSize(text, font, sc, t)
    m = np.zeros((th + base + 2 * t + 4, tw + 2 * t + 4), np.uint8)
    cv2.putText(m, text, (t + 2, th + t + 2), font, sc, 255, t, cv2.LINE_AA)
    return m


def _stamp(tex: np.ndarray, m: np.ndarray, cx: float, cy: float, x0: float, y0: float, res: float,
           color: tuple) -> None:
    """Blend a mask centred at court (cx, cy) into the texture (clipped)."""
    h, w = m.shape
    c0 = int(round((cx - x0) / res - w / 2))
    r0 = int(round((cy - y0) / res - h / 2))
    th, tw = tex.shape[:2]
    a0, b0 = max(0, r0), max(0, c0)
    a1, b1 = min(th, r0 + h), min(tw, c0 + w)
    if a0 >= a1 or b0 >= b1:
        return
    a = m[a0 - r0:a1 - r0, b0 - c0:b1 - c0].astype(np.float32)[..., None] / 255.0
    tex[a0:a1, b0:b1] = tex[a0:a1, b0:b1] * (1.0 - a) + a * np.asarray(color, np.float32)


def _floor_texture(arena: Arena, x0: float, y0: float, w: int, h: int, res: float) -> np.ndarray:
    tex = _wood(arena, x0, y0, w, h, res)
    ft = COURT.free_throw_line_from_baseline
    if arena.border is not None:
        inner = _poly(tex, np.array([[0, 0], [L, 0], [L, W], [0, W]]), x0, y0, res, (0, 0, 0), mask_only=True)
        a = (1.0 - inner.astype(np.float32) / 255.0)[..., None] * 0.92
        tex[:] = tex * (1 - a) + a * np.asarray(arena.border, np.float32)
    for side in ("left", "right"):
        sgn = 1.0 if side == "left" else -1.0
        bx = 0.0 if side == "left" else L
        if arena.stain is not None:
            three = np.asarray(COURT._three_point_line(side, 48))
            _poly(tex, three, x0, y0, res, arena.stain, arena.stain_alpha)
        if arena.key is not None:
            lane = np.array([[bx, 25 - 8], [bx + sgn * ft, 25 - 8], [bx + sgn * ft, 25 + 8], [bx, 25 + 8]])
            _poly(tex, lane, x0, y0, res, arena.key, arena.key_alpha)
    if arena.circle is not None:
        a = np.linspace(0, 2 * math.pi, 97)
        _poly(tex, np.column_stack([47 + 6 * np.cos(a), 25 + 6 * np.sin(a)]), x0, y0, res, arena.circle, 0.95)
    if arena.logo is not None:
        lg = arena.logo
        a = np.linspace(0, 2 * math.pi, 97)
        r = lg["r"]
        _poly(tex, np.column_stack([47 + r * np.cos(a), 25 + r * np.sin(a)]), x0, y0, res, lg["c1"], 0.95)
        ring = np.column_stack([47 + 0.86 * r * np.cos(a), 25 + 0.86 * r * np.sin(a)])
        _poly(tex, ring, x0, y0, res, lg["c2"], 0.95)
        _poly(tex, np.column_stack([47 + 0.80 * r * np.cos(a), 25 + 0.80 * r * np.sin(a)]), x0, y0, res, lg["c1"],
              0.95)
        k = np.arange(10)
        rad = np.where(k % 2 == 0, 0.55 * r, 0.25 * r)  # a star emblem
        star = np.column_stack([47 + rad * np.cos(k * math.pi / 5 - math.pi / 2),
                                25 - 0.12 * r + rad * np.sin(k * math.pi / 5 - math.pi / 2)])
        _poly(tex, star, x0, y0, res, lg["c2"], 0.95)
        m = _text_mask(lg["text"], 0.26 * r / res, 0.05 * r / res)
        if m.shape[1] * res > 1.5 * r:
            m = cv2.resize(m, (int(1.5 * r / res), m.shape[0]), interpolation=cv2.INTER_AREA)
        _stamp(tex, m, 47.0, 25.0 + 0.45 * r, x0, y0, res, lg.get("c3", (240, 240, 240)))
    for wd in arena.words:
        m = _text_mask(wd["text"], wd["h"] / res, wd["stroke"] / res)
        if wd.get("rot", 0):
            m = np.rot90(m, 1 if wd["rot"] > 0 else -1)
        _stamp(tex, np.ascontiguousarray(m), wd["x"], wd["y"], x0, y0, res, wd["color"])
    return tex


_FLOOR: dict[int, list[tuple[np.ndarray, float]]] = {}
FLOOR_RES = 0.02  # ft per texel of the finest floor level


def _floor_levels(arena: Arena) -> tuple[list[tuple[np.ndarray, float]], float, float]:
    """The whole floor (court and aprons) as a uint8 texture pyramid [(texture, ft per texel)], built in bands
    (cached per arena), and its origin (court ft of texel (0, 0))."""
    an, af, ae = arena.apron
    x0, y0 = -ae, -an
    if arena.seed not in _FLOOR:
        _FLOOR.clear()
        w = int(math.ceil((L + 2 * ae) / FLOOR_RES)) + 1
        h = int(math.ceil((W + an + af) / FLOOR_RES)) + 1
        tex = np.empty((h, w, 3), np.uint8)
        for r in range(0, h, 600):
            band = _floor_texture(arena, x0, y0 + r * FLOOR_RES, w, min(600, h - r), FLOOR_RES)
            tex[r:r + band.shape[0]] = np.clip(band, 0, 255).astype(np.uint8)
        levels = [(tex, FLOOR_RES)]
        for _ in range(3):
            t, res = levels[-1]
            levels.append((cv2.resize(t, ((t.shape[1] + 1) // 2, (t.shape[0] + 1) // 2),
                                      interpolation=cv2.INTER_AREA), res * 2.0))
        _FLOOR[arena.seed] = levels
    return _FLOOR[arena.seed], x0, y0


# --------------------------------------------------------------------------- court lines (vector, in the image)


def _line_polys(arena: Arena) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    """Painted court markings as (closed court-ft polygons, colour); the lines centred on the model polylines."""
    hw = arena.line_ft / 2.0
    out: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    lines = COURT.line_segments()
    for i, line in enumerate(lines):
        pts = np.asarray(line, float)
        color = arena.line
        if i == 0 and not arena.border_lines:
            continue  # the painted border's edge is the boundary
        if i in (3, 7) and arena.key_lines is not None:  # the lanes
            color = arena.key_lines
        if i == 1 and arena.division_gap and arena.logo is not None:  # division line broken by the logo
            r = arena.logo["r"]
            for ya, yb in ((0.0, 25 - r), (25 + r, W)):
                if yb > ya:
                    out.append((_segment_quad(np.array([47.0, ya]), np.array([47.0, yb]), hw), color))
            continue
        for a, b in zip(pts[:-1], pts[1:], strict=True):
            if np.hypot(*(b - a)) > 1e-9:
                out.append((_segment_quad(a, b, hw, extend=hw if len(pts) > 2 else 0.0), color))
    if arena.marks:  # lane space marks and the dashed free-throw semicircle (not in the court model)
        for side in ("left", "right"):
            sgn = 1.0 if side == "left" else -1.0
            bx = 0.0 if side == "left" else L
            for d, depth in ((7.0, 1.0), (8.0 + 1.0 / 6.0, 1.0 / 6.0), (11.0, 1.0 / 6.0), (14.0, 1.0 / 6.0)):
                for lat in (-1.0, 1.0):
                    yy = 25 + lat * 8.0
                    xa = bx + sgn * d
                    quad = np.array([[xa, yy], [xa + sgn * depth, yy], [xa + sgn * depth, yy + lat * 0.5],
                                     [xa, yy + lat * 0.5]])
                    out.append((quad, arena.line))
            cx = bx + sgn * COURT.free_throw_line_from_baseline
            for k in range(8):  # dashes of the half circle inside the lane
                th = np.linspace(math.pi / 2 + k * math.pi / 8 + 0.04, math.pi / 2 + (k + 0.55) * math.pi / 8, 5)
                arc = np.column_stack([cx + sgn * 6.0 * np.cos(th), 25.0 + 6.0 * np.sin(th)])
                for a, b in zip(arc[:-1], arc[1:], strict=True):
                    out.append((_segment_quad(a, b, hw), arena.line))
    return out


def _segment_quad(a: np.ndarray, b: np.ndarray, hw: float, extend: float = 0.0) -> np.ndarray:
    d = (b - a) / max(np.hypot(*(b - a)), 1e-12)
    n = np.array([-d[1], d[0]])
    a2, b2 = a - d * extend, b + d * extend
    return np.array([a2 + n * hw, b2 + n * hw, b2 - n * hw, a2 - n * hw])


def _fill_image_polys(img: np.ndarray, polys: list[np.ndarray], color: tuple, P: np.ndarray, size: tuple[int, int],
                      z: float | list[float] = 0.0) -> None:
    w, h = size
    pts_all = []
    for q in polys:
        X = np.column_stack([q, np.full(len(q), z) if np.isscalar(z) else np.asarray(z)])
        uv, wz = _proj3(P, X)
        if np.any(wz <= 0.1) or not np.all(np.isfinite(uv)):
            continue
        if uv[:, 0].max() < -10 or uv[:, 0].min() > w + 10 or uv[:, 1].max() < -10 or uv[:, 1].min() > h + 10:
            continue
        if np.abs(uv).max() > 1e5:
            continue
        pts_all.append(np.round(uv * 16.0).astype(np.int32))
    if pts_all:
        cv2.fillPoly(img, pts_all, color, cv2.LINE_AA, shift=4)


# --------------------------------------------------------------------------- the renderer


def _visible_floor(cam: CameraParams, size: tuple[int, int], rect: tuple[float, float, float, float]
                   ) -> tuple[float, float, float, float, float] | None:
    """Bounding box (court ft) of the floor in view, clipped to ``rect``, and the largest magnification (px/ft)."""
    w, h = size
    gu, gv = np.meshgrid(np.linspace(0, w - 1, 49), np.linspace(0, h - 1, 28))
    uv = np.column_stack([gu.ravel(), gv.ravel()])
    d = _rays(cam, size, uv)
    C = np.asarray(cam.center, float)
    down = d[:, 2] < -1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(down, -C[2] / d[:, 2], 1e5)
    far = ~down | (t > 1e5)
    t = np.where(far, 1e5, t)
    hx = d[:, 0] / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-12)
    hy = d[:, 1] / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-12)
    x = np.where(far, C[0] + 1e4 * hx, C[0] + t * d[:, 0])
    y = np.where(far, C[1] + 1e4 * hy, C[1] + t * d[:, 1])
    x0, x1, y0, y1 = rect
    inside = (~far) & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    xc, yc = np.clip(x, x0, x1), np.clip(y, y0, y1)
    bx0, bx1, by0, by1 = xc.min() - 1, xc.max() + 1, yc.min() - 1, yc.max() + 1
    bx0, bx1, by0, by1 = max(bx0, x0), min(bx1, x1), max(by0, y0), min(by1, y1)
    if bx1 - bx0 < 0.5 or by1 - by0 < 0.5:
        return None
    G = cam.G(size)
    pts = np.column_stack([x[inside], y[inside]]) if inside.any() else np.array([[bx0, by0], [bx1, by1]])
    mag = 1.0
    for dx, dy in ((0.5, 0.0), (0.0, 0.5)):
        a = pts @ G[:, :2].T + G[:, 2]
        b = (pts + [dx, dy]) @ G[:, :2].T + G[:, 2]
        ok = (a[:, 2] > 0) & (b[:, 2] > 0)
        if ok.any():
            step = np.hypot(*(a[ok, :2] / a[ok, 2:3] - b[ok, :2] / b[ok, 2:3]).T) / 0.5
            mag = max(mag, float(step.max()))
    return bx0, bx1, by0, by1, mag


def _quad_mask(H: np.ndarray, tw: int, th: int, size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]
                                                                              ] | None:
    """Anti-aliased image mask of a texture's extent under ``H`` within its bounding box (x0, y0, x1, y1)."""
    w, h = size
    q = np.array([[-0.5, -0.5, 1], [tw - 0.5, -0.5, 1], [tw - 0.5, th - 0.5, 1], [-0.5, th - 0.5, 1]]) @ H.T
    if np.any(q[:, 2] <= 1e-9):
        return None
    uv = q[:, :2] / q[:, 2:3]
    x0, y0 = int(max(0, math.floor(uv[:, 0].min()))), int(max(0, math.floor(uv[:, 1].min())))
    x1, y1 = int(min(w, math.ceil(uv[:, 0].max()) + 1)), int(min(h, math.ceil(uv[:, 1].max()) + 1))
    if x0 >= x1 or y0 >= y1:
        return None
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    pts = np.clip(np.round((uv - [x0, y0]) * 16.0), -(2**26), 2**26).astype(np.int32)
    cv2.fillPoly(m, [pts], 255, cv2.LINE_AA, shift=4)
    return m.astype(np.float32) * np.float32(1.0 / 255.0), (x0, y0, x1, y1)


def _warp(img: np.ndarray, tex: np.ndarray, H: np.ndarray, size: tuple[int, int], alpha: float = 1.0) -> None:
    """Warp a texture (in front of the camera) into the float image within its bounding box."""
    th, tw = tex.shape[:2]
    qm = _quad_mask(H, tw, th, size)
    if qm is None:
        return
    m, (x0, y0, x1, y1) = qm
    T = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]]) @ H
    warped = cv2.warpPerspective(tex, T, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    if warped.shape[2] == 4:  # per-texel alpha
        m = m * warped[..., 3] * np.float32(1.0 / 255.0)
        warped = warped[..., :3]
    a = (m * np.float32(alpha))[..., None]
    sub = img[y0:y1, x0:x1]
    sub *= 1.0 - a
    sub += a * warped


def _horizon_mask(cam: CameraParams, size: tuple[int, int], normal: np.ndarray, offset: float) -> np.ndarray:
    """Pixels whose ray meets the plane n . X = offset in front of the camera (float 0/1)."""
    w, h = size
    Kinv = np.linalg.inv(cam.K(size))
    A = np.asarray(normal, float) @ cam.R().T @ Kinv  # n . (R^T K^-1 [u, v, 1])
    s = offset - float(np.asarray(normal, float) @ np.asarray(cam.center, float))
    val = A[0] * np.arange(w)[None, :] + A[1] * np.arange(h)[:, None] + A[2]
    return ((val * s) > 0).astype(np.float32)


def render(arena: Arena, cam: CameraParams, size: tuple[int, int], look: Look, seed: int,
           near_fans: bool = True) -> np.ndarray:
    """One broadcast frame of ``arena`` seen by ``cam`` (full-resolution camera)."""
    rng = np.random.default_rng(seed)
    w, h = size
    s = _scale1080(size)
    P = _P(cam, size)
    G = cam.G(size)
    sur = _surround(arena)
    an, af, ae = arena.apron
    yf, yn = W + af, -an
    xl, xr = -ae, L + ae
    img = np.empty((h, w, 3), np.float32)
    img[:] = (20, 18, 18)  # the arena's dark upper levels
    alpha_stand = math.radians(33.0)
    ca, sa = math.cos(alpha_stand), math.sin(alpha_stand)
    # far stands, then LED boards along the far sideline
    surfaces = [
        (sur.far_stands, np.array([_X0, yf + 2.5, 3.3]), np.array([1.0, 0, 0]), np.array([0, ca, sa]), STAND_RES),
        (sur.far_led, np.array([_X0, yf, 0.0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), 0.05),
        # end stands and boards (left: facing +x; right: facing -x)
        (sur.end_stands, np.array([xl - 2.5, _YS0, 3.3]), np.array([0, 1.0, 0]), np.array([-ca, 0, sa]),
         STAND_RES),
        (sur.end_stands, np.array([xr + 2.5, _YS0, 3.3]), np.array([0, 1.0, 0]), np.array([ca, 0, sa]), STAND_RES),
        (sur.end_led, np.array([xl, _YS0, 0.0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), 0.05),
        (sur.end_led, np.array([xr, _YS0, 0.0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), 0.05),
    ]
    for tex, o, au, av, res in surfaces:
        ext = (tex.shape[1] * res, tex.shape[0] * res)
        if not _front(cam, o, au, av, ext):
            continue
        _warp(img, tex, _plane_H(cam, size, o, au, av, res), size)
    # the floor
    rect = (xl, xr, yn, yf)
    vis = _visible_floor(cam, size, rect)
    floor_mask = np.zeros((h, w), np.float32)
    if vis is not None:
        mag = vis[4]
        levels, fx0, fy0 = _floor_levels(arena)
        k = 0
        while k + 1 < len(levels) and levels[k + 1][1] * mag <= 1.1:  # the coarsest level not blurring the floor
            k += 1
        tex, res = levels[k]
        if arena.reflect > 0:  # the far boards and the first rows of the stands mirrored in the glossy floor
            refl = np.zeros((h, w, 3), np.float32)
            for t2, o, au, av, r2 in surfaces[:2]:
                t2 = t2[: int(round(18.0 / r2))]
                om = o * np.array([1, 1, -1.0])
                avm = av * np.array([1, 1, -1.0])
                ext = (t2.shape[1] * r2, t2.shape[0] * r2)
                if _front(cam, om, au, avm, ext):
                    _warp(refl, t2, _plane_H(cam, size, om, au, avm, r2), size)
            kk = max(3, int(14 * s) | 1)
            refl = cv2.GaussianBlur(refl, (3, kk * 2 + 1), 0)
        A = np.array([[res, 0, fx0], [0, res, fy0], [0, 0, 1.0]])
        hm = _horizon_mask(cam, size, np.array([0, 0, 1.0]), 0.0)
        floor = cv2.warpPerspective(tex, G @ A, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        rect_ft = np.array([[xl, yn, 1.0], [xr, yn, 1.0], [xr, yf, 1.0], [xl, yf, 1.0]])
        q = rect_ft @ G.T
        if np.all(q[:, 2] > 1e-9):
            uv = q[:, :2] / q[:, 2:3]
            mm = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mm, [np.clip(np.round(uv * 16), -(2**26), 2**26).astype(np.int32)], 255, cv2.LINE_AA,
                         shift=4)
            floor_mask = mm.astype(np.float32) * np.float32(1.0 / 255.0) * hm
        else:  # a camera standing on the floor (replays): the horizon bounds it
            floor_mask = hm
        # court lines, drawn in the image at their true width
        groups: dict[tuple, list[np.ndarray]] = {}
        for q, c in _line_polys(arena):
            groups.setdefault(c, []).append(q)
        for c, qs in groups.items():
            _fill_image_polys(floor, qs, c, P, size)
        if arena.reflect > 0:
            floor += arena.reflect * refl
        if arena.sheen > 0:
            gy = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None, None]
            gx = np.cos(np.linspace(-1.2, 1.2, w, dtype=np.float32))[None, :, None]
            floor *= 1.0 + arena.sheen * (gx * (1 - gy) - 0.3)
        for _ in range(arena.glare):  # specular highlights of the arena lights
            fx, fy = rng.uniform(-5, L + 5), rng.uniform(10, W + 6)
            q = G @ np.array([fx, fy, 1.0])
            if q[2] <= 0:
                continue
            u, v = q[0] / q[2], q[1] / q[2]
            if not (-200 * s < u < w + 200 * s and -200 * s < v < h + 200 * s):
                continue
            sx, sy = rng.uniform(15, 50) * s, rng.uniform(40, 140) * s
            x0b, x1b = int(max(0, u - 3 * sx)), int(min(w, u + 3 * sx))
            y0b, y1b = int(max(0, v - 3 * sy)), int(min(h, v + 3 * sy))
            if x1b <= x0b or y1b <= y0b:
                continue
            gxs = np.exp(-0.5 * ((np.arange(x0b, x1b) - u) / sx) ** 2)
            gys = np.exp(-0.5 * ((np.arange(y0b, y1b) - v) / sy) ** 2)
            amp = rng.uniform(25, 90)
            floor[y0b:y1b, x0b:x1b] += (amp * gys[:, None] * gxs[None, :])[..., None]
        a = floor_mask[..., None]
        img *= 1.0 - a
        img += floor * a
    # basket stanchions and backboards
    if arena.baskets:
        _baskets(img, cam, size, P, arena, rng)
    # players and referees, far ones first
    _people(img, cam, size, P, look, rng, arena)
    # courtside fans along the near sideline (in front of the floor)
    if near_fans:
        o = np.array([_X0, yn - 3.0, -31.0])  # the front row a few feet behind the apron's edge, heads ~4 ft up
        au, av = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
        tex = sur.near
        if _front(cam, o, au, av, (tex.shape[1] * STAND_RES, tex.shape[0] * STAND_RES)):
            _warp(img, tex, _plane_H(cam, size, o, au, av, STAND_RES), size)
    # motion blur, exposure, lens
    if look.blur_px > 0.5:
        n = max(2, int(round(look.blur_px * s)))
        kern = np.zeros((n | 1, n | 1), np.float32)
        c = (n | 1) // 2
        dx, dy = math.cos(math.radians(look.blur_deg)), math.sin(math.radians(look.blur_deg))
        for t in np.linspace(-c, c, 4 * n):
            kern[int(round(c + t * dy)), int(round(c + t * dx))] = 1.0
        img = cv2.filter2D(img, -1, kern / kern.sum())
    if look.soft > 0:
        img = cv2.GaussianBlur(img, (0, 0), look.soft * s)
    img *= look.gain * np.asarray(look.tint, np.float32)
    if look.vignette > 0:
        xx = ((np.arange(w, dtype=np.float32) - w / 2) / (w / 2)) ** 2
        yy = ((np.arange(h, dtype=np.float32) - h / 2) / (h / 2)) ** 2
        img *= (1.0 - np.float32(look.vignette * 0.5) * (yy[:, None] + xx[None, :]))[..., None]
    if look.flash > 0:
        img = img * (1.0 + look.flash) + 60.0 * look.flash
    out = np.clip(img, 0, 255).astype(np.uint8)
    for ov in look.overlays:
        _overlay(out, ov, size, rng)
    noise = rng.standard_normal(out.shape, dtype=np.float32)
    noise *= np.float32(look.noise)
    noise += out
    out = np.clip(noise, 0, 255).astype(np.uint8)
    if look.jpeg < 100:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(look.jpeg)])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def _quad3(base: np.ndarray, wdir: np.ndarray, hw: float, z0: float, z1: float, off: float = 0.0) -> np.ndarray:
    b = base + off * wdir
    return np.array([[*(b - hw * wdir), z0], [*(b + hw * wdir), z0], [*(b + hw * wdir), z1], [*(b - hw * wdir), z1]])


def _draw3(img: np.ndarray, P: np.ndarray, quads: list[np.ndarray], color: tuple) -> None:
    pts = []
    for X in quads:
        uv, wz = _proj3(P, X)
        if np.any(wz <= 0.5) or np.abs(uv).max() > 1e5:
            continue
        pts.append(np.round(uv * 16).astype(np.int32))
    if pts:
        cv2.fillPoly(img, pts, color, cv2.LINE_AA, shift=4)


def _people(img: np.ndarray, cam: CameraParams, size: tuple[int, int], P: np.ndarray, look: Look,
            rng: np.random.Generator, arena: Arena) -> None:
    """Players (two teams) and referees standing on the floor near the aim point, with shadows."""
    C = np.asarray(cam.center, float)
    fwd = cam.R()[2]
    t = -C[2] / fwd[2] if fwd[2] < -1e-6 else 60.0
    ax, ay = float(np.clip(C[0] + t * fwd[0], 0, L)), float(np.clip(C[1] + t * fwd[1], 0, W))
    wdir = np.array([cam.R()[0][0], cam.R()[0][1]])
    wdir /= max(np.hypot(*wdir), 1e-9)
    home, away = (238, 238, 238), _team(rng, ("white", "silver"))
    shorts = {home: (200, 200, 200), away: tuple(int(v * 0.8) for v in away)}
    people = []
    for i in range(look.players):
        x = ax + rng.uniform(-22, 22)
        y = float(np.clip(ay + rng.uniform(-20, 20), -2, W + 2))
        people.append(("player", np.array([x, y]), rng.uniform(6.1, 7.2), home if i % 2 else away))
    for _ in range(look.refs):
        where = rng.integers(3)
        if where == 0:
            p = np.array([ax + rng.uniform(-15, 15), rng.uniform(-3, 1)])
        elif where == 1:
            p = np.array([ax + rng.uniform(-15, 15), W + rng.uniform(-1, 3)])
        else:
            p = np.array([rng.choice([-2.0, L + 2.0]) + rng.uniform(-1, 1), rng.uniform(8, 42)])
        people.append(("ref", p, rng.uniform(5.8, 6.4), (150, 150, 150)))
    people.sort(key=lambda q: -float(np.hypot(*(q[1] - C[:2]))))
    for kind, p, hgt, col in people:
        # shadow on the floor
        a = np.linspace(0, 2 * math.pi, 20)
        sh = np.column_stack([p[0] + 1.3 * np.cos(a), p[1] + 0.8 * np.sin(a), np.zeros(20)])
        uv, wz = _proj3(P, sh)
        if np.all(wz > 0.5) and np.abs(uv).max() < 1e5:
            m = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(m, [np.round(uv * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
            img *= (1.0 - 0.35 * m.astype(np.float32) / 255.0)[..., None]
        skin = SKIN[int(rng.integers(len(SKIN)))]
        legs = skin if kind == "player" else (25, 25, 25)
        torso = col
        q = []
        for off in (-0.33, 0.33):
            q.append(_quad3(p, wdir, 0.2, 0.0, 0.48 * hgt, off))
        _draw3(img, P, q, legs)
        _draw3(img, P, [_quad3(p, wdir, 0.25, 0.0, 0.25, o) for o in (-0.33, 0.33)],
               (240, 240, 240) if rng.random() < 0.5 else (20, 20, 20))
        if kind == "player":
            _draw3(img, P, [_quad3(p, wdir, 0.75, 0.40 * hgt, 0.56 * hgt)], shorts[col])
        _draw3(img, P, [_quad3(p, wdir, 0.78, 0.53 * hgt, 0.82 * hgt)], torso)
        if kind == "ref":
            _draw3(img, P, [_quad3(p, wdir, 0.06, 0.53 * hgt, 0.82 * hgt, o) for o in (-0.45, -0.15, 0.15, 0.45)],
                   (20, 20, 20))
        up = rng.random() < 0.3
        for side in (-1.0, 1.0):
            z0, z1 = (0.78 * hgt, 1.12 * hgt) if up and side > 0 else (0.45 * hgt, 0.8 * hgt)
            _draw3(img, P, [_quad3(p, wdir, 0.13, z0, z1, side * 0.92)], skin)
        X = np.array([[*p, 0.91 * hgt], [*(p + 0.36 * wdir), 0.91 * hgt]])
        uv, wz = _proj3(P, X)
        if np.all(wz > 0.5) and np.abs(uv).max() < 1e5:
            r = max(1.0, float(np.hypot(*(uv[1] - uv[0]))))
            cv2.circle(img, (int(round(uv[0, 0] * 16)), int(round(uv[0, 1] * 16))), int(round(r * 16)), skin, -1,
                       cv2.LINE_AA, shift=4)


def _baskets(img: np.ndarray, cam: CameraParams, size: tuple[int, int], P: np.ndarray, arena: Arena,
             rng: np.random.Generator) -> None:
    pad = arena.border if arena.border is not None else (40, 40, 40)
    for side in (-1.0, 1.0):  # -1: left basket (x = 4), +1: right (x = 90)
        bx = 4.0 if side < 0 else L - 4.0
        base_x = -6.5 if side < 0 else L + 6.5
        wy = np.array([0.0, 1.0])
        # padded base behind the baseline and the arm up to the board
        box = [_quad3(np.array([base_x, 25.0]), wy, 2.5, 0.0, 4.0)]
        _draw3(img, P, box, pad)
        arm = np.array([[base_x, 24.4, 4.0], [base_x, 25.6, 4.0], [bx - side * 1.2, 25.6, 12.6],
                        [bx - side * 1.2, 24.4, 12.6]])
        _draw3(img, P, [arm], pad)
        # backboard: glass tint, frame and the inner rectangle
        board = np.array([[bx, 22.0, 9.4], [bx, 28.0, 9.4], [bx, 28.0, 12.9], [bx, 22.0, 12.9]])
        uv, wz = _proj3(P, board)
        if np.all(wz > 0.5) and np.abs(uv).max() < 1e5:
            m = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(m, [np.round(uv * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
            img *= (1.0 - 0.15 * m.astype(np.float32) / 255.0)[..., None]
        frame_c = (235, 235, 235)
        e = 0.17
        segs = [((22.0, 9.4), (28.0, 9.4)), ((28.0, 9.4), (28.0, 12.9)), ((28.0, 12.9), (22.0, 12.9)),
                ((22.0, 12.9), (22.0, 9.4)), ((24.0, 10.0), (26.0, 10.0)), ((26.0, 10.0), (26.0, 11.5)),
                ((26.0, 11.5), (24.0, 11.5)), ((24.0, 11.5), (24.0, 10.0))]
        quads = []
        for (ya, za), (yb, zb) in segs:
            d = np.array([yb - ya, zb - za]) / max(math.hypot(yb - ya, zb - za), 1e-9)
            n = np.array([-d[1], d[0]]) * e / 2
            quads.append(np.array([[bx, ya + n[0], za + n[1]], [bx, yb + n[0], zb + n[1]],
                                   [bx, yb - n[0], zb - n[1]], [bx, ya - n[0], za - n[1]]]))
        _draw3(img, P, quads, frame_c)
        # rim
        rx = 5.25 if side < 0 else L - 5.25
        a = np.linspace(0, 2 * math.pi, 24)
        rim = np.column_stack([rx + 0.75 * np.cos(a), 25 + 0.75 * np.sin(a), np.full(24, 10.0)])
        uv, wz = _proj3(P, rim)
        if np.all(wz > 0.5) and np.abs(uv).max() < 1e5:
            cv2.polylines(img, [np.round(uv * 16).astype(np.int32)], True, (20, 80, 230),
                          max(1, int(round(3 * _scale1080(size)))), cv2.LINE_AA, shift=4)


def _overlay(img: np.ndarray, kind: str, size: tuple[int, int], rng: np.random.Generator) -> None:
    """Broadcast graphics in 1920x1080 layout coordinates (scaled)."""
    w, h = size
    sx, sy = w / 1920.0, h / 1080.0
    boxes = {"espn": (468, 944, 1000, 125), "tl": (40, 36, 560, 92), "bl": (40, 958, 700, 84),
             "br": (1430, 900, 450, 130), "tr": (1540, 40, 340, 110), "ticker": (0, 1022, 1920, 58)}
    x, y, bw, bh = boxes[kind]
    x0, y0, x1, y1 = int(x * sx), int(y * sy), int((x + bw) * sx), int((y + bh) * sy)
    bg = np.array(_team(rng), np.float32) * 0.5 if kind != "espn" else np.array((60, 30, 20), np.float32)
    img[y0:y1, x0:x1] = np.clip(img[y0:y1, x0:x1] * 0.15 + bg * 0.85, 0, 255).astype(np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    if kind == "espn":
        img[int(y0 + 0.08 * (y1 - y0)):int(y0 + 0.55 * (y1 - y0)), x0:x1] = (90, 45, 30)
        for i, (txt, c) in enumerate([("LAL", (130, 40, 85)), ("98", None), ("GSW", (170, 75, 20)), ("102", None)]):
            bx0 = x0 + int((0.12 + 0.17 * i) * (x1 - x0))
            if c is not None:
                img[y0 + 4:y0 + int(0.55 * (y1 - y0)), bx0:bx0 + int(0.15 * (x1 - x0))] = c
            cv2.putText(img, txt, (bx0 + 6, y0 + int(0.45 * (y1 - y0))), font, 1.3 * sy, (255, 255, 255),
                        max(1, int(2 * sy)), cv2.LINE_AA)
        cv2.putText(img, "4TH  2:41  14", (x0 + int(0.72 * (x1 - x0)), y0 + int(0.45 * (y1 - y0))), font,
                    1.2 * sy, (255, 255, 255), max(1, int(2 * sy)), cv2.LINE_AA)
        cv2.putText(img, "BONUS   TO 3", (x0 + int(0.2 * (x1 - x0)), y0 + int(0.88 * (y1 - y0))), font, 0.9 * sy,
                    (220, 220, 220), max(1, int(2 * sy)), cv2.LINE_AA)
        return
    words = {"tl": "BOS 87  NYK 85  3RD 4:12", "bl": "DEN 101  PHX 99   4TH 0:48  :09", "br": "PRESENTED BY",
             "tr": "LIVE", "ticker": "NBA SCORES  MIA 99 ORL 95 FINAL   CHI 77 CLE 80 3RD"}
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (235, 235, 235), max(1, int(2 * sy)), cv2.LINE_AA)
    cv2.putText(img, words[kind], (x0 + int(12 * sx), y0 + int(0.66 * (y1 - y0))), font, 1.0 * sy,
                (255, 255, 255), max(1, int(2 * sy)), cv2.LINE_AA)
    if kind == "br":
        cv2.circle(img, (x1 - int(70 * sx), (y0 + y1) // 2), int(40 * sy), (40, 40, 220), -1, cv2.LINE_AA)


def render_closeup(arena: Arena, size: tuple[int, int], seed: int, t: int = 0, noise_seed: int | None = None
                   ) -> np.ndarray:
    """A close-up (a player against the blurred stands): no floor lines. One shot per ``seed``; ``t`` (frames into
    the shot) drifts the view slowly, as a camera following the player does."""
    rng = np.random.default_rng(seed)
    w, h = size
    sur = _surround(arena)
    tex = sur.far_stands
    y0 = int(rng.integers(0, tex.shape[0] // 2))
    x0 = int(rng.integers(0, tex.shape[1] - 600)) + int(round(2 * t))
    crop = tex[y0:y0 + 240, x0:x0 + 400][::-1]
    img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    img = cv2.GaussianBlur(img, (0, 0), 6 * _scale1080(size))
    jersey = _team(rng)
    cx = int(w * rng.uniform(0.35, 0.65)) + int(round(w * 0.002 * t))
    cv2.ellipse(img, (cx, int(h * 1.05)), (int(w * 0.22), int(h * 0.6)), 0, 0, 360, jersey, -1, cv2.LINE_AA)
    skin = SKIN[int(rng.integers(len(SKIN)))]
    cv2.circle(img, (cx, int(h * 0.32)), int(h * 0.17), skin, -1, cv2.LINE_AA)
    cv2.putText(img, str(int(rng.integers(0, 99))), (cx - int(0.06 * w), int(h * 0.85)), cv2.FONT_HERSHEY_DUPLEX,
                4.0 * _scale1080(size), (240, 240, 240), max(2, int(10 * _scale1080(size))), cv2.LINE_AA)
    nrng = np.random.default_rng(seed + 7919 * t if noise_seed is None else noise_seed)
    return np.clip(img.astype(np.float32) + nrng.normal(0, 2.0, img.shape).astype(np.float32), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- scenario families

FAMILIES = ("espn", "geometry", "zoom", "floors", "paint", "logos", "glare", "clutter", "resolution", "mixed")


def _jit(c: tuple[int, int, int], rng: np.random.Generator, d: float = 8.0) -> tuple[int, int, int]:
    return tuple(int(np.clip(v + rng.normal(0, d), 0, 255)) for v in c)  # type: ignore[return-value]


def _espn_arena(seed: int, rng: np.random.Generator) -> Arena:
    """Light wood, dark blue lines, a centre logo and wordmarks: like Chase Center (where the defaults were tuned)."""
    navy = _jit(TEAM["navy"], rng, 4)
    return Arena(
        seed=seed, wood=_jit(WOOD["light"], rng, 5), grain=float(rng.uniform(0.4, 0.8)), line=navy,
        logo={"r": float(rng.uniform(5.5, 7.5)), "c1": TEAM["royal"], "c2": TEAM["gold"], "c3": (240, 240, 240),
              "text": "DUBS"},
        words=[{"text": "WARRIORS", "x": -5.0, "y": 25.0, "h": 3.2, "rot": 1, "color": navy, "stroke": 0.4},
               {"text": "WARRIORS", "x": L + 5.0, "y": 25.0, "h": 3.2, "rot": -1, "color": navy, "stroke": 0.4},
               {"text": "GOLDEN STATE", "x": 47.0, "y": W + 4.5, "h": 2.6, "rot": 0, "color": TEAM["royal"],
                "stroke": 0.35}],
        apron=(8.0, 9.0, 10.0), crowd=(50, 46, 46),
        crowd_colors=[TEAM["royal"], TEAM["gold"], (230, 230, 230), (40, 40, 40), (60, 60, 150)],
        led=[TEAM["royal"], (30, 30, 30), (220, 220, 220), (20, 20, 190)],
        glare=int(rng.integers(0, 3)), reflect=float(rng.uniform(0.04, 0.12)))


def _random_arena(seed: int, rng: np.random.Generator, wood: bool = True, paint: bool = True, logos: bool = True,
                  glare: bool = True) -> Arena:
    a = _espn_arena(seed, rng)
    t1, t2 = _team(rng, ("white", "silver")), _team(rng)
    a.crowd_colors = [t1, t2, (230, 230, 230), (40, 40, 40), _team(rng)]
    a.led = [t1, t2, (20, 20, 20), (220, 220, 220)]
    a.logo = None
    a.words = []
    if wood:
        kind = str(rng.choice(["light", "light", "medium", "medium", "dark", "grey"]))
        a.wood = _jit(WOOD[kind], rng, 6)
        a.grain = float(rng.uniform(0.3, 1.5))
        lk = rng.random()
        a.line = ((240, 240, 240) if lk < 0.45 else (25, 25, 25) if lk < 0.6 else _jit(TEAM["navy"], rng, 4)
                  if lk < 0.75 else t1)
        if kind in ("dark", "grey") and sum(a.line) < 300:
            a.line = (240, 240, 240)
    if paint:
        if rng.random() < 0.75:
            a.key = t1
            a.key_alpha = float(rng.uniform(0.8, 1.0))
            if rng.random() < 0.3:
                a.key_lines = t1  # lane lines in the paint's colour: only an edge
        if rng.random() < 0.4:
            a.stain = t2 if rng.random() < 0.5 else (30, 50, 80)
            a.stain_alpha = float(rng.uniform(0.15, 0.45))
        if rng.random() < 0.7:
            a.border = t1 if rng.random() < 0.6 else (25, 25, 25)
            a.border_lines = bool(rng.random() < 0.5)
        if rng.random() < 0.4:
            a.circle = t2
    if logos:
        if rng.random() < 0.85:
            a.logo = {"r": float(rng.uniform(6.0, 14.0)), "c1": t1, "c2": t2, "c3": _team(rng),
                      "text": str(rng.choice(["NETS", "HEAT", "BULLS", "KINGS", "MAGIC", "SUNS"]))}
            a.division_gap = bool(rng.random() < 0.5)
        name = str(rng.choice(["CELTICS", "LAKERS", "WARRIORS", "BUCKS", "RAPTORS", "PISTONS", "SPURS"]))
        wc = t1 if a.border is None else (240, 240, 240)
        st = float(rng.uniform(0.3, 0.7))
        a.words += [{"text": name, "x": -5.0, "y": 25.0, "h": float(rng.uniform(3, 5)), "rot": 1, "color": wc,
                     "stroke": st},
                    {"text": name, "x": L + 5.0, "y": 25.0, "h": float(rng.uniform(3, 5)), "rot": -1, "color": wc,
                     "stroke": st}]
        if rng.random() < 0.7:
            a.words.append({"text": str(rng.choice(["SPONSOR", "BANK OF", "AIRLINES", "ARENA"])), "x": 47.0,
                            "y": W + 4.5, "h": 3.0, "rot": 0, "color": wc, "stroke": st})
        if rng.random() < 0.5:
            a.words.append({"text": name, "x": float(rng.choice([25.0, 69.0])), "y": W - 4.0, "h": 2.5, "rot": 0,
                            "color": t1, "stroke": 0.35})  # on the court along the far sideline
        if rng.random() < 0.5:
            a.words.append({"text": "NBA CUP", "x": 47.0, "y": -4.0, "h": 2.5, "rot": 0, "color": wc,
                            "stroke": 0.35})
    if glare:
        a.glare = int(rng.integers(0, 6))
        a.reflect = float(rng.uniform(0.0, 0.35))
        a.sheen = float(rng.uniform(0.0, 0.2))
    a.apron = (float(rng.uniform(5, 12)), float(rng.uniform(5, 12)), float(rng.uniform(7, 14)))
    a.baskets = bool(rng.random() < 0.9)
    if rng.random() < 0.2:
        a.stands = "dark"
    return a


def sample_arena(family: str, rng: np.random.Generator, seed: int) -> Arena:
    if family in ("espn", "geometry", "zoom", "clutter", "resolution"):
        return _espn_arena(seed, rng)
    if family == "floors":
        return _random_arena(seed, rng, paint=False, logos=False, glare=False)
    if family == "paint":
        a = _random_arena(seed, rng, logos=False, glare=False)
        a.key = a.key or _team(rng)
        if a.border is None and rng.random() < 0.6:
            a.border = _team(rng)
        return a
    if family == "logos":
        a = _random_arena(seed, rng, wood=False, paint=False, glare=False)
        if a.logo is None:
            a.logo = {"r": float(rng.uniform(6.0, 14.0)), "c1": TEAM["red"], "c2": TEAM["black"],
                      "c3": TEAM["white"], "text": "HEAT"}
        return a
    if family == "glare":
        a = _espn_arena(seed, rng)
        a.glare = int(rng.integers(3, 9))
        a.reflect = float(rng.uniform(0.2, 0.45))
        a.sheen = float(rng.uniform(0.1, 0.25))
        return a
    return _random_arena(seed, rng)  # mixed


def sample_center(family: str, rng: np.random.Generator, size: tuple[int, int] | None = None) -> np.ndarray:
    """A game camera centre; with ``size``, one from which a framed view of the court lines exists."""
    while True:
        c = _draw_center(family, rng)
        if size is None:
            return c
        try:
            sample_view(rng, c, size, _fov_range(family, rng))
            return c
        except RuntimeError:
            continue


def _draw_center(family: str, rng: np.random.Generator) -> np.ndarray:
    if family in ("geometry", "mixed"):
        while True:
            back, up = rng.uniform(45.0, 130.0), rng.uniform(18.0, 60.0)
            elev = math.degrees(math.atan2(up, back + 25.0))
            if 8.0 <= elev <= 42.0:
                return np.array([47.0 + rng.uniform(-10.0, 10.0), -back, up])
    return np.array([47.0 + rng.normal(0, 0.7), -86.0 + rng.normal(0, 3.0), 33.0 + rng.normal(0, 1.5)])


def sample_size(family: str, rng: np.random.Generator) -> tuple[int, int]:
    if family == "resolution":
        return SIZES[str(rng.choice(["720p", "4k"]))]
    if family == "mixed":
        return SIZES[str(rng.choice(["720p", "1080p", "1080p", "4k"]))]
    return SIZES["1080p"]


def _fov_range(family: str, rng: np.random.Generator) -> tuple[float, float]:
    if family == "zoom":
        return (12.0, 18.0) if rng.random() < 0.5 else (48.0, 60.0)
    if family == "mixed":
        return (12.0, 60.0)
    return (20.0, 50.0)


def observable(cam: CameraParams, size: tuple[int, int], min_pts: int = 50) -> bool:
    """Enough court lines in view, in two directions at least (one straight line fixes no camera)."""
    from courtvision.perception.court_tracker import court_samples

    pts, dirs = court_samples(COURT, 1.0)
    G = cam.G(size)
    h = pts @ G[:, :2].T + G[:, 2]
    ok = h[:, 2] > 0
    uv = h[:, :2] / np.where(ok, h[:, 2], 1.0)[:, None]
    w, hh = size
    vis = ok & (uv[:, 0] > 0.02 * w) & (uv[:, 0] < 0.98 * w) & (uv[:, 1] > 0.02 * hh) & (uv[:, 1] < 0.98 * hh)
    if vis.sum() < min_pts:
        return False
    d = dirs[vis]
    ev = np.linalg.eigvalsh(d.T @ d)
    return bool(ev[0] / ev.sum() >= 0.12)


def view_toward(center: np.ndarray, ax: float, ay: float, hfov: float, size: tuple[int, int],
                roll: float = 0.0) -> CameraParams:
    pan = math.degrees(math.atan2(ax - center[0], ay - center[1]))
    tilt = math.degrees(math.atan2(center[2], math.hypot(ax - center[0], ay - center[1])))
    f = (size[0] / 2.0) / math.tan(math.radians(hfov) / 2.0)
    return CameraParams(np.asarray(center, float), pan, tilt, roll, f)


def sample_view(rng: np.random.Generator, center: np.ndarray, size: tuple[int, int],
                fov: tuple[float, float]) -> CameraParams:
    for _ in range(400):
        hfov = math.exp(rng.uniform(math.log(fov[0]), math.log(fov[1])))
        u = rng.random()
        ax = rng.uniform(2, 36) if u < 0.4 else rng.uniform(58, 92) if u < 0.8 else rng.uniform(30, 64)
        ay = rng.uniform(10, 42)
        cam = view_toward(center, ax, ay, hfov, size, float(np.clip(rng.normal(0, 0.25), -0.8, 0.8)))
        if observable(cam, size) and framed(cam, size):
            return cam
    raise RuntimeError("no observable view")


def framed(cam: CameraParams, size: tuple[int, int], near_ft: float = -14.0, max_share: float = 0.25) -> bool:
    """An operator frames the play: the stands beyond a few feet past the near sideline fill at most a quarter of
    the picture (down the centre column; wide shots from far back show some)."""
    w, h = size
    vs = np.linspace(0.0, h - 1.0, 65)
    d = _rays(cam, size, np.column_stack([np.full(len(vs), w / 2.0), vs]))
    down = d[:, 2] < -1e-9
    t = np.where(down, -cam.center[2] / np.where(down, d[:, 2], -1.0), np.inf)
    y = cam.center[1] + t * d[:, 1]
    beyond = down & (y < near_ft)
    return bool(beyond.mean() <= max_share)


def sample_look(family: str, rng: np.random.Generator) -> Look:
    lk = Look(players=int(rng.integers(4, 11)), refs=int(rng.integers(1, 4)), noise=float(rng.uniform(1.0, 3.0)),
              jpeg=int(rng.integers(72, 94)), gain=float(rng.uniform(0.92, 1.08)),
              tint=tuple(float(v) for v in rng.uniform(0.96, 1.04, 3)),  # type: ignore[arg-type]
              vignette=float(rng.uniform(0.0, 0.25)), soft=float(rng.uniform(0.4, 0.9)))
    if rng.random() < 0.15:
        lk.blur_px, lk.blur_deg = float(rng.uniform(4, 14)), float(rng.normal(0, 8))
    if family in ("espn", "resolution"):
        lk.overlays = ["espn"]
    elif family == "clutter":
        lk.players, lk.refs = 10, 3
        if rng.random() < 0.5:
            lk.blur_px, lk.blur_deg = float(rng.uniform(5, 30)), float(rng.normal(0, 10))
        kinds = ["tl", "bl", "br", "tr", "ticker"]
        lk.overlays = [str(k) for k in rng.choice(kinds, int(rng.integers(1, 4)), replace=False)]
    elif family == "mixed":
        u = rng.random()
        lk.overlays = ["espn"] if u < 0.3 else [] if u < 0.45 else [str(k) for k in rng.choice(
            ["tl", "bl", "br", "tr", "ticker"], int(rng.integers(1, 3)), replace=False)]
        if rng.random() < 0.25:
            lk.blur_px, lk.blur_deg = float(rng.uniform(4, 30)), float(rng.normal(0, 10))
    return lk


def evidence_config(look: Look) -> EvidenceConfig:
    """What the court tool is told: the ESPN bug's box when the broadcast is ESPN's, nothing otherwise."""
    return EvidenceConfig(exclude=list(ESPN_EXCLUDE) if "espn" in look.overlays else [])


# --------------------------------------------------------------------------- scoring

_GX, _GY = np.meshgrid(np.arange(-2.0, L + 2.01, 2.0), np.arange(-2.0, W + 2.01, 2.0))
GRID = np.column_stack([_GX.ravel(), _GY.ravel()])


def reproj_error(est: CameraParams | None, truth: CameraParams, size: tuple[int, int]) -> float:
    """RMS image distance (1080-line px) between the floor grid points in view under ``est`` and ``truth``."""
    if est is None:
        return math.inf
    Gt, Ge = truth.G(size), est.G(size)
    ht = GRID @ Gt[:, :2].T + Gt[:, 2]
    ok = ht[:, 2] > 0
    ut = ht[:, :2] / np.where(ok, ht[:, 2], 1.0)[:, None]
    w, h = size
    vis = ok & (ut[:, 0] >= 0) & (ut[:, 0] < w) & (ut[:, 1] >= 0) & (ut[:, 1] < h)
    if vis.sum() < 4:
        return math.inf
    he = GRID[vis] @ Ge[:, :2].T + Ge[:, 2]
    if np.any(he[:, 2] <= 0):
        return math.inf
    ue = he[:, :2] / he[:, 2:3]
    d = np.linalg.norm(ue - ut[vis], axis=1)
    return float(np.sqrt(np.mean(d**2)) * 1080.0 / h)


def best_fit_at(center: np.ndarray, truth: CameraParams, size: tuple[int, int]) -> float:
    """RMS error (1080-line px) of the best camera at ``center`` for the view ``truth`` (pan / tilt / roll / f)."""
    from scipy.optimize import least_squares

    Gt = truth.G(size)
    ht = GRID @ Gt[:, :2].T + Gt[:, 2]
    ok = ht[:, 2] > 0
    ut = ht[:, :2] / np.where(ok, ht[:, 2], 1.0)[:, None]
    w, h = size
    vis = ok & (ut[:, 0] >= 0) & (ut[:, 0] < w) & (ut[:, 1] >= 0) & (ut[:, 1] < h)
    X, U = GRID[vis], ut[vis]
    C = np.asarray(center, float)
    pan0 = math.degrees(math.atan2(*(np.mean(X, axis=0) - C[:2])))
    tilt0 = math.degrees(math.atan2(C[2], float(np.hypot(*(np.mean(X, axis=0) - C[:2])))))

    def res(v: np.ndarray) -> np.ndarray:
        G = CameraParams(C, v[0], v[1], v[2], math.exp(v[3])).G(size)
        q = X @ G[:, :2].T + G[:, 2]
        return np.where(q[:, 2:3] > 0, q[:, :2] / q[:, 2:3] - U, 1e3).ravel()

    sol = least_squares(res, np.array([pan0, tilt0, 0.0, math.log(truth.f)]), method="lm", max_nfev=400)
    d = sol.fun.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(d**2, axis=1))) * 1080.0 / h)


def _scaled(p: CameraParams, s: float) -> CameraParams:
    return CameraParams(np.asarray(p.center, float).copy(), p.pan, p.tilt, p.roll, p.f * s)


def _cam_dict(p: CameraParams | None) -> dict[str, Any] | None:
    if p is None:
        return None
    return {"center": [round(float(v), 3) for v in p.center], "pan": round(p.pan, 4), "tilt": round(p.tilt, 4),
            "roll": round(p.roll, 4), "f": round(p.f, 2)}


# --------------------------------------------------------------------------- modes


def run_single(family: str, idx: int, seed: int, tol: float) -> dict[str, Any]:
    rng = np.random.default_rng([seed, FAMILIES.index(family), 1, idx])
    arena = sample_arena(family, rng, int(rng.integers(1 << 30)))
    size = sample_size(family, rng)
    center = sample_center(family, rng, size)
    cam = sample_view(rng, center, size, _fov_range(family, rng))
    look = sample_look(family, rng)
    img = render(arena, cam, size, look, int(rng.integers(1 << 30)))
    t0 = time.perf_counter()
    scale = min(1.0, 960.0 / size[0])
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    ev = court_evidence(small, evidence_config(look))
    r = calibrate_frame(ev, COURT, free_center=True)
    dt = time.perf_counter() - t0
    est = None if r is None else _scaled(r.params, 1.0 / scale)
    err = reproj_error(est, cam, size)
    accepted = r is not None and r.quality >= 0.15 and plausible(r.params, ev.size)
    return {"mode": "single", "family": family, "idx": idx, "size": list(size), "err": err,
            "correct": err <= tol, "accepted": bool(accepted), "quality": None if r is None else round(r.quality, 4),
            "center_err": None if est is None else round(float(np.linalg.norm(est.center - cam.center)), 3),
            "hfov": round(2 * math.degrees(math.atan(size[0] / 2 / cam.f)), 2), "time_s": round(dt, 3),
            "truth": _cam_dict(cam), "est": _cam_dict(est)}


def run_center(family: str, idx: int, seed: int, tol: float, n_frames: int) -> dict[str, Any]:
    rng = np.random.default_rng([seed, FAMILIES.index(family), 2, idx])
    arena = sample_arena(family, rng, int(rng.integers(1 << 30)))
    size = sample_size(family, rng)
    center = sample_center(family, rng, size)
    look0 = sample_look(family, rng)
    frames, cams, kinds = [], [], []
    for _ in range(n_frames):
        cam = sample_view(rng, center, size, _fov_range(family, rng))
        look = sample_look(family, rng)
        look.overlays = look0.overlays  # one broadcaster per game
        frames.append(render(arena, cam, size, look, int(rng.integers(1 << 30))))
        cams.append(cam)
        kinds.append("game")
    frames.append(render_closeup(arena, size, int(rng.integers(1 << 30))))
    cams.append(None)
    kinds.append("closeup")
    rp = _replay_camera(rng, size)
    frames.append(render(arena, rp, size, Look(players=6, refs=1), int(rng.integers(1 << 30)), near_fans=False))
    cams.append(None)
    kinds.append("replay")
    order = rng.permutation(len(frames))
    frames = [frames[i] for i in order]
    cams = [cams[i] for i in order]
    kinds = [kinds[i] for i in order]
    t0 = time.perf_counter()
    est_c, est = estimate_center(frames, COURT, 960, evidence_config(look0))
    dt = time.perf_counter() - t0
    errs, false_cams = [], 0
    for c, e, k in zip(cams, est, kinds, strict=True):
        if k == "game":
            errs.append(reproj_error(e, c, size))
        elif e is not None:
            false_cams += 1
    fit = math.inf
    if est_c is not None:
        fit = max(best_fit_at(est_c, view_toward(center, ax, ay, 35.0, size), size)
                  for ax in (12.0, 47.0, 82.0) for ay in (12.0, 25.0, 38.0))
    found = [e for e in errs if math.isfinite(e)]
    return {"mode": "center", "family": family, "idx": idx, "size": list(size), "frames": n_frames,
            "center_found": est_c is not None,
            "center_err": None if est_c is None else round(float(np.linalg.norm(est_c - center)), 3),
            "center_fit": fit, "correct": bool(fit <= tol), "frame_errs": errs,
            "frames_correct": sum(e <= tol for e in errs), "frames_wrong": sum(tol < e < math.inf for e in errs),
            "frames_found": len(found), "false_cams": false_cams, "time_s": round(dt, 2),
            "truth_center": [round(float(v), 3) for v in center],
            "est_center": None if est_c is None else [round(float(v), 3) for v in est_c]}


def _replay_camera(rng: np.random.Generator, size: tuple[int, int]) -> CameraParams:
    """Another camera of the arena: behind a baseline or low on the near sideline."""
    for _ in range(100):
        u = rng.random()
        if u < 0.5:
            right = rng.random() < 0.5
            C = np.array([L + 9.0 if right else -9.0, 25.0 + rng.uniform(-10, 10), rng.uniform(8, 16)])
            ax, ay = (L - rng.uniform(10, 40) if right else rng.uniform(10, 40)), rng.uniform(15, 35)
        else:
            C = np.array([47.0 + rng.uniform(-30, 30), -rng.uniform(6, 14), rng.uniform(5, 12)])
            ax, ay = C[0] + rng.uniform(-25, 25), rng.uniform(12, 40)
        cam = view_toward(C, ax, ay, rng.uniform(30, 55), size)
        if observable(cam, size):
            return cam
    return view_toward(np.array([-9.0, 25.0, 12.0]), 20.0, 25.0, 45.0, size)


def _track_plan(rng: np.random.Generator, center: np.ndarray, size: tuple[int, int], fov: tuple[float, float],
                max_blur: float = 32.0) -> list[dict[str, Any]]:
    """Per-frame kind / camera / blur / flash / scripted cut of a 10 s broadcast segment at 30 fps. Motion blur is
    half a frame of the camera's turn (a 1/60 s shutter); an operator pans no faster than keeps it under
    ``max_blur`` (1080-line px) at the zoom in use, as tight shots follow slow play."""
    w, h = size

    def f_of(hf: float) -> float:
        return (w / 2.0) / math.tan(math.radians(hf) / 2.0)

    def peak_blur(a: tuple[float, float, float], b: tuple[float, float, float], frames: int) -> float:
        ca, cb_ = view_toward(center, a[0], a[1], a[2], size), view_toward(center, b[0], b[1], b[2], size)
        ang = math.hypot(cb_.pan - ca.pan, cb_.tilt - ca.tilt)
        return math.pi / 2.0 * ang / frames * math.pi / 180.0 * f_of(min(a[2], b[2])) * 0.5 * 1080.0 / h

    def capped(a: tuple[float, float, float], b: tuple[float, float, float], frames: int) -> tuple[float, float, float]:
        for _ in range(30):  # shorten the move until its peak blur is acceptable
            if peak_blur(a, b, frames) <= max_blur:
                break
            b = (a[0] + 0.8 * (b[0] - a[0]), a[1] + 0.8 * (b[1] - a[1]), b[2])
        return b

    def keyframes(n: int, frames: int) -> list[tuple[float, float, float]]:
        out: list[tuple[float, float, float]] = []
        for _ in range(n):
            for _ in range(100):
                ax, ay = rng.uniform(5, 89), rng.uniform(12, 38)
                hf = math.exp(rng.uniform(math.log(fov[0]), math.log(fov[1])))
                if out:  # the play moves on: at most ~30 ft a keyframe
                    ax = out[-1][0] + float(np.clip(ax - out[-1][0], -30, 30))
                    ax, ay, hf = capped(out[-1], (ax, ay, hf), frames)
                c = view_toward(center, ax, ay, hf, size)
                if observable(c, size) and framed(c, size):
                    break
            out.append((ax, ay, hf))
        return out

    def path(kf: list[tuple[float, float, float]], n: int) -> list[CameraParams]:
        cams = []
        seg = max(1, n // max(1, len(kf) - 1))
        for i in range(n):
            k = min(i // seg, len(kf) - 2)
            t = (i - k * seg) / seg
            t = 0.5 - 0.5 * math.cos(math.pi * min(t, 1.0))
            a, b = kf[k], kf[k + 1]
            ax, ay = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            hf = math.exp(math.log(a[2]) + t * (math.log(b[2]) - math.log(a[2])))
            cams.append(view_toward(center, ax, ay, hf, size, 0.0))
        return cams

    plan: list[dict[str, Any]] = []

    def game(cams: list[CameraParams]) -> None:
        for i, c in enumerate(cams):
            prev = cams[i - 1] if i else c
            dp = c.pan - prev.pan
            dtilt = c.tilt - prev.tilt
            blur = math.hypot(dp, dtilt) * math.pi / 180.0 * c.f * 0.5 * 1080.0 / size[1]
            plan.append({"kind": "game", "cam": c, "blur": blur,
                         "blur_deg": math.degrees(math.atan2(dtilt, dp)) if blur > 0 else 0.0})

    game(path(keyframes(4, 30), 90))
    plan[40]["false_cut"] = True  # the shot-boundary detector fires on the same camera
    plan[65]["flash"] = 0.9  # a photographer's flash
    plan += [{"kind": "closeup"} for _ in range(30)]
    # back to the game camera elsewhere, then a transition: a fast pan across half the court, zoomed out
    kf = keyframes(3, 20)
    a = kf[-1]
    far = (a[0] + (45.0 if a[0] < 47 else -45.0), a[1], max(a[2], 30.0))
    far = capped(a, far, 45)
    game(path(kf, 40) + path([a, far], 45) + path([far, far], 5))
    rp = _replay_camera(rng, size)
    plan += [{"kind": "replay", "cam": rp} for _ in range(30)]
    game(path(keyframes(3, 30), 60))
    return plan


def run_track(family: str, idx: int, seed: int, tol: float) -> dict[str, Any]:
    rng = np.random.default_rng([seed, FAMILIES.index(family), 3, idx])
    arena = sample_arena(family, rng, int(rng.integers(1 << 30)))
    size = sample_size(family, rng)
    center = sample_center(family, rng, size)
    look0 = sample_look(family, rng)
    plan = _track_plan(rng, center, size, _fov_range(family, rng))
    tracker = BroadcastCourtTracker(COURT, size, center, BroadcastTrackerConfig(evidence=evidence_config(look0)))
    cuts = ShotBoundaryDetector()
    rows = []
    shot_seed, shot_start = 0, 0
    for i, fr in enumerate(plan):
        fseed = int(rng.integers(1 << 30))
        if fr["kind"] == "closeup":
            if i == 0 or plan[i - 1]["kind"] != "closeup":  # one continuous shot
                shot_seed, shot_start = fseed, i
            img = render_closeup(arena, size, shot_seed, i - shot_start)
        else:
            lk = Look(players=look0.players, refs=look0.refs, overlays=look0.overlays if fr["kind"] == "game" else [],
                      noise=look0.noise, jpeg=look0.jpeg, gain=look0.gain, tint=look0.tint, vignette=look0.vignette,
                      soft=look0.soft, blur_px=fr.get("blur", 0.0), blur_deg=fr.get("blur_deg", 0.0),
                      flash=fr.get("flash", 0.0))
            img = render(arena, fr["cam"], size, lk, fseed, near_fans=fr["kind"] == "game")
        det = cuts.update(img)
        cut = bool(det or fr.get("false_cut", False) or i == 0)
        t0 = time.perf_counter()
        fix = tracker.update(img, cut=cut)
        dt = time.perf_counter() - t0
        out = fix.params if fix.status in ("tracking", "reacquired") else None
        err = reproj_error(out, fr["cam"], size) if fr["kind"] == "game" else None
        rows.append({"kind": fr["kind"], "status": fix.status, "cut": cut, "det": bool(det),
                     "err": None if err is None or not math.isfinite(err) else round(err, 3),
                     "out": out is not None, "ms": round(dt * 1000.0, 1),
                     "p": None if out is None else [round(out.pan, 5), round(out.tilt, 5), round(out.roll, 5),
                                                    round(out.f, 3)],
                     "t": None if fr["kind"] != "game" else [round(fr["cam"].pan, 5), round(fr["cam"].tilt, 5),
                                                             round(fr["cam"].roll, 5), round(fr["cam"].f, 3)]})
    live = [r for r in rows if r["kind"] == "game"]
    other = [r for r in rows if r["kind"] != "game"]
    ok = [r["out"] and r["err"] is not None and r["err"] <= tol for r in live]
    wrong = [r["out"] and (r["err"] is None or r["err"] > tol) for r in live]
    delays = []  # frames from each return to the game camera until a correct camera
    for i, r in enumerate(rows):
        if r["kind"] == "game" and (i == 0 or rows[i - 1]["kind"] != "game"):
            d = 0
            while i + d < len(rows) and rows[i + d]["kind"] == "game" and not (
                    rows[i + d]["out"] and rows[i + d]["err"] is not None and rows[i + d]["err"] <= tol):
                d += 1
            delays.append(d)
    sm = smoothed_errors(rows, center, size)
    sm_ok = [e is not None and e <= tol for e, r in zip(sm, rows, strict=True) if r["kind"] == "game"]
    sm_bad = [e is not None and e > tol for e, r in zip(sm, rows, strict=True) if r["kind"] == "game"]
    ms = [r["ms"] for r in rows]
    tr_ms = [r["ms"] for r in rows if r["status"] in ("tracking", "held")]
    return {"mode": "track", "family": family, "idx": idx, "size": list(size), "frames": len(rows),
            "live_correct": round(sum(ok) / len(live), 4), "live_wrong": round(sum(wrong) / len(live), 4),
            "smooth_correct": round(sum(sm_ok) / len(live), 4), "smooth_wrong": round(sum(sm_bad) / len(live), 4),
            "center": [float(v) for v in center], "frames_out": [r["p"] for r in rows],
            "frames_truth": [r["t"] for r in rows],
            "nongame_false": round(sum(r["out"] for r in other) / max(1, len(other)), 4),
            "reacq_delays": delays, "detector_cuts": sum(r["det"] for r in rows),
            "ms_mean": round(float(np.mean(ms)), 1), "ms_track_median": round(float(np.median(tr_ms)), 1)
            if tr_ms else None, "ms_max": round(float(np.max(ms)), 1),
            "errs_live": [r["err"] for r in live], "statuses": [r["status"][0] for r in rows]}


def smooth_runs(raw: dict[int, tuple[float, float, float, float]], k: int = 5,
                max_fill: int = 3) -> dict[int, tuple[float, float, float, float]]:
    """The camera track smoothing of ``courtvision.nba.analyze`` (``_smooth_runs``, copied): gaps of at most
    ``max_fill`` frames interpolated, then per run of frames a centred moving median and a moving average of width
    ``k`` over pan, tilt, roll and log focal."""
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


def smoothed_errors(rows: list[dict[str, Any]], center: np.ndarray, size: tuple[int, int]) -> list[float | None]:
    """Per frame, the error of the smoothed camera track (what the NBA analysis uses) on game frames."""
    raw = {i: tuple(r["p"]) for i, r in enumerate(rows) if r.get("p") is not None}
    sm = smooth_runs(raw)  # type: ignore[arg-type]
    out: list[float | None] = []
    for i, r in enumerate(rows):
        if r["kind"] != "game" or i not in sm:
            out.append(None)
            continue
        q, t = sm[i], r["t"]
        e = reproj_error(CameraParams(np.asarray(center, float), *q), CameraParams(np.asarray(center, float), *t), size)
        out.append(e if math.isfinite(e) else None)
    return out


# --------------------------------------------------------------------------- runner and summary


def _task(args: tuple) -> dict[str, Any]:  # noqa: ANN401
    mode, family, idx, seed, tol, n_frames = args
    cv2.setNumThreads(1)
    t0 = time.time()
    try:
        if mode == "single":
            out = run_single(family, idx, seed, tol)
        elif mode == "center":
            out = run_center(family, idx, seed, tol, n_frames)
        else:
            out = run_track(family, idx, seed, tol)
    except Exception as exc:  # noqa: BLE001 (one failed task must not end the benchmark)
        import traceback

        out = {"mode": mode, "family": family, "idx": idx, "error": f"{type(exc).__name__}: {exc}",
               "trace": traceback.format_exc()[-2000:]}
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def _fmt(v: float | None, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return f"{v:.{nd}f}"


def summarize(rows: list[dict[str, Any]], tol: float) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for mode in ("single", "center", "track"):
        for fam in FAMILIES:
            rs = [r for r in rows if r.get("mode") == mode and r.get("family") == fam and "error" not in r]
            nerr = sum(1 for r in rows if r.get("mode") == mode and r.get("family") == fam and "error" in r)
            if not rs and not nerr:
                continue
            s: dict[str, Any] = {"n": len(rs), "errors": nerr}
            if mode == "single":
                e = np.array([r["err"] for r in rs], float)
                acc = np.array([r["accepted"] for r in rs], bool)
                s["correct"] = float(np.mean(e <= tol)) if len(rs) else None
                s["acc_correct"] = float(np.mean(acc & (e <= tol))) if len(rs) else None
                s["acc_wrong"] = float(np.mean(acc & (e > tol))) if len(rs) else None
                s["err_p50"] = float(np.median(e)) if len(rs) else None
                ok = e[e <= tol]
                s["err_ok_p50"] = float(np.median(ok)) if len(ok) else None
                s["err_p90"] = float(np.quantile(e, 0.9)) if len(rs) else None
                s["time_s"] = float(np.mean([r["time_s"] for r in rs])) if rs else None
            elif mode == "center":
                fit = np.array([r["center_fit"] for r in rs], float)
                s["correct"] = float(np.mean(fit <= tol)) if len(rs) else None
                s["fit_p50"] = float(np.median(fit)) if len(rs) else None
                fe = [e for r in rs for e in r["frame_errs"]]
                s["frames_correct"] = float(np.mean([e <= tol for e in fe])) if fe else None
                s["frames_wrong"] = float(np.mean([tol < e < math.inf for e in fe])) if fe else None
                s["false_cams"] = sum(r["false_cams"] for r in rs)
                ce = [r["center_err"] for r in rs if r["center_err"] is not None]
                s["center_err_p50"] = float(np.median(ce)) if ce else None
                s["time_s"] = float(np.mean([r["time_s"] for r in rs])) if rs else None
            else:
                s["live_correct"] = float(np.mean([r["live_correct"] for r in rs])) if rs else None
                s["live_wrong"] = float(np.mean([r["live_wrong"] for r in rs])) if rs else None
                sc = [r["smooth_correct"] for r in rs if "smooth_correct" in r]
                s["smooth_correct"] = float(np.mean(sc)) if sc else None
                sw = [r["smooth_wrong"] for r in rs if "smooth_wrong" in r]
                s["smooth_wrong"] = float(np.mean(sw)) if sw else None
                s["nongame_false"] = float(np.mean([r["nongame_false"] for r in rs])) if rs else None
                d = [x for r in rs for x in r["reacq_delays"]]
                s["reacq_p50"] = float(np.median(d)) if d else None
                s["reacq_max"] = float(np.max(d)) if d else None
                s["ms_mean"] = float(np.mean([r["ms_mean"] for r in rs])) if rs else None
                tm = [r["ms_track_median"] for r in rs if r["ms_track_median"] is not None]
                s["ms_track"] = float(np.median(tm)) if tm else None
            out[(mode, fam)] = s
    return out


COLS = {
    "single": [("correct", "ok", 2), ("acc_correct", "acc_ok", 2), ("acc_wrong", "acc_bad", 2), ("err_p50", "e50", 1),
               ("err_ok_p50", "e50ok", 2), ("err_p90", "e90", 1), ("time_s", "s/frame", 1)],
    "center": [("correct", "ok", 2), ("fit_p50", "fit50", 2), ("frames_correct", "fr_ok", 2),
               ("frames_wrong", "fr_bad", 2), ("false_cams", "false", 0), ("center_err_p50", "c_err", 1),
               ("time_s", "s/game", 0)],
    "track": [("live_correct", "live_ok", 3), ("live_wrong", "live_bad", 3), ("smooth_correct", "sm_ok", 3),
              ("smooth_wrong", "sm_bad", 3), ("nongame_false", "false", 3),
              ("reacq_p50", "reacq50", 0), ("reacq_max", "reacqmx", 0), ("ms_mean", "ms/fr", 0),
              ("ms_track", "ms_trk", 0)],
}


def print_summary(paths: list[str], tol: float) -> None:
    sums = []
    for p in paths:
        rows = [json.loads(line) for line in open(p) if line.strip()]
        sums.append(summarize(rows, tol))
    for mode, cols in COLS.items():
        keys = sorted({k for s in sums for k in s if k[0] == mode}, key=lambda k: FAMILIES.index(k[1]))
        if not keys:
            continue
        print(f"\n== {mode} (tol {tol} px)" + ("" if len(paths) == 1 else "  [" + " | ".join(Path(p).stem for p in paths) + "]"))
        head = f"{'family':<11} {'n':>3} " + " ".join(f"{c[1]:>{8 * len(paths) + len(paths) - 1}}" for c in cols)
        print(head)
        for k in keys:
            n = "/".join(str(s.get(k, {}).get("n", 0)) for s in sums)
            cells = []
            for key, _, nd in cols:
                cells.append("|".join(f"{_fmt(s.get(k, {}).get(key), nd):>8}" for s in sums))
            print(f"{k[1]:<11} {n:>3} " + " ".join(cells))


def show(family: str, n: int, seed: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        rng = np.random.default_rng([seed, FAMILIES.index(family), 1, i])
        arena = sample_arena(family, rng, int(rng.integers(1 << 30)))
        size = sample_size(family, rng)
        center = sample_center(family, rng)
        cam = sample_view(rng, center, size, _fov_range(family, rng))
        look = sample_look(family, rng)
        t0 = time.time()
        img = render(arena, cam, size, look, int(rng.integers(1 << 30)))
        p = Path(out_dir) / f"{family}_{i}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(p, size, f"hfov {2 * math.degrees(math.atan(size[0] / 2 / cam.f)):.1f}", np.round(cam.center, 1),
              f"{time.time() - t0:.2f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="results (JSON lines); appended, finished tasks are skipped")
    ap.add_argument("--modes", default="single,center,track")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--n", type=int, default=12, help="single frames per family")
    ap.add_argument("--games", type=int, default=2, help="centre estimates per family")
    ap.add_argument("--frames", type=int, default=8, help="game frames per centre estimate")
    ap.add_argument("--runs", type=int, default=1, help="tracking runs per family")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=4.0, help="correct within this RMS error (1080-line px)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--summary", nargs="+", default=None, help="print the summary of result files (side by side)")
    ap.add_argument("--show", default=None, help="write --n sample frames of this family to --frames-dir")
    ap.add_argument("--frames-dir", default="court_bench_frames")
    args = ap.parse_args()
    if args.summary:
        print_summary(args.summary, args.tol)
        return 0
    if args.show:
        show(args.show, args.n, args.seed, args.frames_dir)
        return 0
    if not args.out:
        ap.error("--out is required")
    fams = [f for f in args.families.split(",") if f]
    for f in fams:
        if f not in FAMILIES:
            ap.error(f"unknown family {f!r}")
    tasks = []
    for mode in args.modes.split(","):
        count = {"single": args.n, "center": args.games, "track": args.runs}[mode]
        for fam in fams:
            for i in range(count):
                tasks.append((mode, fam, i, args.seed, args.tol, args.frames))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "error" not in r:
                done.add((r["mode"], r["family"], r["idx"]))
    tasks = [t for t in tasks if (t[0], t[1], t[2]) not in done]
    tasks.sort(key=lambda t: {"center": 0, "track": 1, "single": 2}[t[0]])  # the long ones first
    from courtvision.perception import court_search

    print(f"{len(tasks)} tasks ({len(done)} done) -> {out}; court_search: {court_search.__file__}", flush=True)
    t0 = time.time()
    with out.open("a") as fh:
        if args.workers > 1:
            import multiprocessing as mp

            with mp.get_context("spawn" if sys.platform == "darwin" else "fork").Pool(args.workers) as pool:
                for k, r in enumerate(pool.imap_unordered(_task, tasks), 1):
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    print(f"[{k}/{len(tasks)} {time.time() - t0:.0f}s] {r['mode']} {r['family']} {r['idx']}: "
                          + (r.get("error") or f"correct={r.get('correct', r.get('live_correct'))}"), flush=True)
        else:
            for k, t in enumerate(tasks, 1):
                r = _task(t)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                print(f"[{k}/{len(tasks)} {time.time() - t0:.0f}s] {r['mode']} {r['family']} {r['idx']}: "
                      + (r.get("error") or f"correct={r.get('correct', r.get('live_correct'))}"), flush=True)
    print_summary([str(out)], args.tol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
