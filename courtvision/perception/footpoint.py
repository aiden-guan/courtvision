"""Court position of a person from their image box: an upright cylinder seen in perspective.

The naive foot point is the box's bottom centre through the floor homography, pushed ``radius`` ft
"away from the camera" (the direction of moving up the image), because the box bottom is the front
edge of the feet. That is biased: verticals converge (towards the vertical vanishing point below the
image), so a figure off the image centre leans outwards; its box spans the head's column and the feet's
column and the box centre is not over the feet, and image-up is not the radial direction on the floor.
On the sim camera the bias is ~0.5 ft for an exact box (median 0.54 ft on perceived boxes).

``FootModel`` removes it. A box is modelled as the exact image box of an upright cylinder (radius
``radius_ft``, height ``height_ft``) standing on the floor, which a real person's box also is to first
order. The camera is recovered from the floor homography alone (square pixels, zero skew, principal
point at the image centre: a standard fixed-camera assumption, checked by reprojection), a correction
field ``true base - naive estimate`` is tabulated over the court once, and each naive estimate is
corrected by a two-step fixed-point lookup (bilinear). When the camera cannot be recovered the naive
estimate is returned unchanged.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from courtvision.geometry.camera import PinholeCamera
from courtvision.geometry.homography import Homography

__all__ = ["FootModel", "FootModelConfig", "camera_from_homography"]

log = logging.getLogger(__name__)


@dataclass
class FootModelConfig:
    """Settings of :class:`FootModel` (feet; see the module docstring)."""

    radius_ft: float = 0.0  # body radius: the box bottom is this far in front of the body centre
    height_ft: float = 6.5  # body height; NBA average ~6 ft 6 in; sourced (league average)
    margin_ft: float = 10.0  # the correction table covers the court plus this margin; design
    step_ft: float = 2.0  # table spacing (the field is smooth: bilinear error < 0.01 ft); design
    max_reproj_px: float = 1.0  # the recovered camera must reproduce the homography this well; design


def camera_from_homography(homography: Homography, image_size: tuple[int, int],
                           court_wh: tuple[float, float] = (94.0, 50.0),
                           max_reproj_px: float = 1.0) -> PinholeCamera | None:
    """Pinhole camera behind an image -> floor homography, or None.

    ``G = inv(H) ~ K [r1 r2 t]`` with ``K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]`` (square pixels, zero
    skew, principal point at the image centre). ``r1 . r2 = 0`` and ``|r1| = |r2|`` give two linear
    equations in ``1 / f^2`` (least squares: either alone degenerates for some poses). The camera must
    be above the floor, see the court in front of it and reproduce ``G`` on the court corners within
    ``max_reproj_px``.
    """
    try:
        G = np.linalg.inv(np.asarray(homography.H, dtype=float))
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(G)):
        return None
    G = G / np.linalg.norm(G)
    w_img, h_img = image_size
    cx, cy = w_img / 2.0, h_img / 2.0
    a = G[0, :2] - cx * G[2, :2]
    b = G[1, :2] - cy * G[2, :2]
    c = G[2, :2]
    p = np.array([a[0] * a[1] + b[0] * b[1], a[0] ** 2 + b[0] ** 2 - a[1] ** 2 - b[1] ** 2])
    q = np.array([c[0] * c[1], c[0] ** 2 - c[1] ** 2])
    if float(p @ p) <= 0:
        return None
    inv_f2 = -float(p @ q) / float(p @ p)
    if not inv_f2 > 0:
        return None
    f = 1.0 / math.sqrt(inv_f2)
    if not 0.1 * w_img < f < 20.0 * w_img:
        return None
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    M = np.linalg.solve(K, G)
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    corners = np.array([[0.0, 0.0], [court_wh[0], 0.0], [0.0, court_wh[1]], [court_wh[0], court_wh[1]]])
    for sign in (1.0, -1.0):
        r1, r2, t = sign * lam * M[:, 0], sign * lam * M[:, 1], sign * lam * M[:, 2]
        R0 = np.column_stack([r1, r2, np.cross(r1, r2)])
        U, _, Vt = np.linalg.svd(R0)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            continue
        cam = PinholeCamera(K=K, R=R, t=t, image_size=(int(w_img), int(h_img)))
        if float(cam.center[2]) <= 0:
            continue
        pts = np.c_[corners, np.zeros(len(corners))]
        if np.any(cam.depth(pts) <= 0):
            continue
        uv = cam.project(pts)
        g = np.c_[corners, np.ones(len(corners))] @ G.T
        ref = g[:, :2] / g[:, 2:]
        err = float(np.max(np.linalg.norm(uv - ref, axis=1)))
        if np.isfinite(err) and err <= max_reproj_px:
            return cam
    return None


class FootModel:
    """``court_points(boxes) -> (N, 2)`` court ft of person boxes (``(N, 4)`` xyxy px)."""

    def __init__(self, homography: Homography, image_size: tuple[int, int], court_wh: tuple[float, float],
                 config: FootModelConfig | None = None) -> None:
        self.config = config or FootModelConfig()
        self.homography = homography
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.court_wh = court_wh
        self.camera = camera_from_homography(homography, self.image_size, court_wh, self.config.max_reproj_px)
        self._grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        if self.camera is None:
            log.warning("foot model: no camera behind the floor homography; using the naive foot point")

    @property
    def corrected(self) -> bool:
        return self.camera is not None

    def naive(self, boxes: np.ndarray) -> np.ndarray:
        """Bottom centre through the homography, pushed ``radius_ft`` towards image-up."""
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        feet = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]]
        xy = self.homography.image_to_court(feet, front_only=True).reshape(-1, 2)
        r = self.config.radius_ft
        if r > 0:
            up = self.homography.image_to_court(feet - np.array([0.0, 1.0]), front_only=True).reshape(-1, 2)
            d = up - xy
            n = np.linalg.norm(d, axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                step = np.where(n > 1e-9, d / n, 0.0) * r
            xy = xy + np.nan_to_num(step)
        return xy

    def court_points(self, boxes: np.ndarray) -> np.ndarray:
        n = self.naive(boxes)
        if self.camera is None or len(n) == 0:
            return n
        b0 = n + self._correction(n)
        return n + self._correction(b0)

    # ------------------------------------------------------------------ internals

    def _table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._grid is None:
            cfg = self.config
            assert self.camera is not None
            L, W = self.court_wh
            xs = np.arange(-cfg.margin_ft, L + cfg.margin_ft + 1e-9, cfg.step_ft)
            ys = np.arange(-cfg.margin_ft, W + cfg.margin_ft + 1e-9, cfg.step_ft)
            gx, gy = np.meshgrid(xs, ys, indexing="ij")
            base = np.c_[gx.ravel(), gy.ravel()]
            boxes = np.array([self.camera.bbox_of_cylinder(x, y, cfg.height_ft, cfg.radius_ft).as_tuple()
                              for x, y in base])
            corr = base - self.naive(boxes)
            corr[~np.all(np.isfinite(corr), axis=1)] = 0.0  # behind the camera / off the image: no correction
            self._grid = (xs, ys, corr.reshape(len(xs), len(ys), 2))
        return self._grid

    def _correction(self, pts: np.ndarray) -> np.ndarray:
        xs, ys, corr = self._table()
        out = np.zeros_like(pts)
        ok = np.all(np.isfinite(pts), axis=1)
        if not ok.any():
            return out
        p = pts[ok]
        fx = np.clip((p[:, 0] - xs[0]) / (xs[1] - xs[0]), 0.0, len(xs) - 1.000001)
        fy = np.clip((p[:, 1] - ys[0]) / (ys[1] - ys[0]), 0.0, len(ys) - 1.000001)
        i, j = fx.astype(int), fy.astype(int)
        tx, ty = (fx - i)[:, None], (fy - j)[:, None]
        out[ok] = ((1 - tx) * (1 - ty) * corr[i, j] + tx * (1 - ty) * corr[i + 1, j]
                   + (1 - tx) * ty * corr[i, j + 1] + tx * ty * corr[i + 1, j + 1])
        return out
