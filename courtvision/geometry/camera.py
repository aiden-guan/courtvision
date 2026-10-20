"""Pinhole camera over the court (world = court feet, z up).

Camera axes follow OpenCV: x right, y down, z forward. ``X_cam = R @ X_world + t`` and
``[u, v, 1] ~ K @ X_cam``. Bounding boxes of spheres, horizontal circles and upright cylinders
are exact: each outline is the image of a quadric / circle, so its extreme tangent lines come from
the image dual conic (Hartley & Zisserman §8.3) instead of point sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from courtvision.types import BBox

from .court import CourtSpec
from .homography import Homography

__all__ = ["PinholeCamera", "RECOMMENDED_SIM_CAMERA", "sees_full_court"]

_EPS_DEPTH = 1e-9  # ft; points at or behind the camera plane project to nan
_NAN_BOX = BBox(math.nan, math.nan, math.nan, math.nan)

# Elevated sideline sim camera (pitch ~31 deg) that frames all 4 court corners, both rims and
# 6.7-ft players standing on the corners with >= 20 px margin at 1280x720, for every league in
# LEAGUES; checked by tests/test_geometry_camera.py. The ARCHITECTURE SimConfig default position
# (47, -38, 26) puts both near corners off-image (u = -279 and 1559); only the position changes.
RECOMMENDED_SIM_CAMERA: dict[str, Any] = {
    "camera_position": (47.0, -56.0, 48.0),
    "camera_target": (47.0, 25.0, 0.0),
    "hfov_deg": 72.0,
    "image_size": (1280, 720),
}


@dataclass
class PinholeCamera:
    """Pinhole camera; world coordinates are court feet with z up."""

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    image_size: tuple[int, int]  # (width, height) px

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=float).reshape(3, 3)
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=float).reshape(3)
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))

    def __eq__(self, other: object) -> bool:
        # The dataclass-generated __eq__ compares ndarray fields inside a tuple, which raises
        # "truth value of an array is ambiguous" for two distinct cameras.
        if not isinstance(other, PinholeCamera):
            return NotImplemented
        return (
            self.image_size == other.image_size
            and np.array_equal(self.K, other.K)
            and np.array_equal(self.R, other.R)
            and np.array_equal(self.t, other.t)
        )

    __hash__ = None  # type: ignore[assignment]  # mutable arrays: unhashable, as before

    # ------------------------------------------------------------------ construction

    @classmethod
    def look_at(
        cls,
        position: Any,
        target: Any,
        image_size: tuple[int, int] = (1280, 720),
        hfov_deg: float = 70.0,
    ) -> PinholeCamera:
        """Camera at ``position`` aimed at ``target``, level horizon (no roll).

        K has square pixels, ``fx = (W / 2) / tan(hfov / 2)`` and the principal point at the image
        centre ``(W / 2, H / 2)``.
        """
        if not 0.0 < hfov_deg < 180.0:
            raise ValueError("hfov_deg must be in (0, 180)")
        C = np.asarray(position, dtype=float).reshape(3)
        forward = np.asarray(target, dtype=float).reshape(3) - C
        if np.linalg.norm(forward) < 1e-9:
            raise ValueError("camera position and target coincide")
        R = _look_rotation(forward / np.linalg.norm(forward))
        w, h = int(image_size[0]), int(image_size[1])
        f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        K = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])
        return cls(K=K, R=R, t=-R @ C, image_size=(w, h))

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world (court) coordinates."""
        return -self.R.T @ self.t

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix ``K [R | t]``."""
        return self.K @ np.hstack([self.R, self.t[:, None]])

    # ------------------------------------------------------------------ projection

    def depth(self, xyz: Any) -> np.ndarray:
        """Camera-frame z (ft) of world points; (3,) -> scalar array, (N, 3) -> (N,)."""
        pts = np.asarray(xyz, dtype=float)
        return pts @ self.R[2] + self.t[2]

    def project(self, xyz: Any) -> np.ndarray:
        """World ft -> pixels: (N, 3) -> (N, 2), (3,) -> (2,); points behind the camera -> nan."""
        arr = np.asarray(xyz, dtype=float)
        single = arr.ndim == 1
        if arr.shape[-1:] != (3,) or arr.ndim > 2:
            raise ValueError(f"points must have shape (3,) or (N, 3), got {arr.shape}")
        cam = arr.reshape(-1, 3) @ self.R.T + self.t
        img = cam @ self.K.T
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = img[:, :2] / z[:, None]
        uv[~(z > _EPS_DEPTH)] = np.nan
        return uv[0] if single else uv

    def in_image(self, uv: Any, margin_px: float = 0.0) -> np.ndarray:
        """Bool mask of pixels inside the image shrunk by ``margin_px`` (nan -> False)."""
        p = np.asarray(uv, dtype=float).reshape(-1, 2)
        w, h = self.image_size
        with np.errstate(invalid="ignore"):
            return (p[:, 0] >= margin_px) & (p[:, 0] <= w - margin_px) & (p[:, 1] >= margin_px) & (p[:, 1] <= h - margin_px)

    def floor_homography(self) -> Homography:
        """Image -> court homography of the floor (z = 0): inverse of ``K [r1 r2 t]``."""
        G = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.t])
        H = np.linalg.inv(G)  # positive scale keeps w > 0 for floor points in front of the camera
        return Homography(H / np.linalg.norm(H))

    # ------------------------------------------------------------------ bounding boxes

    def bbox_of_sphere(self, x: float, y: float, z: float, radius: float) -> BBox:
        """Exact image bbox of a sphere; all-nan when it reaches the camera plane."""
        S = np.array([x, y, z], dtype=float)
        if not float(self.depth(S)) > radius + _EPS_DEPTH:
            return _NAN_BOX
        # With the world origin moved to S the dual quadric is diag(-r^2, -r^2, -r^2, 1) and the
        # camera is [K R | c] with c = P @ [S, 1], so A = K R.
        return _outline_bbox(self.P @ np.append(S, 1.0), self.K @ self.R, radius)

    def bbox_of_cylinder(self, x: float, y: float, height: float, radius: float) -> BBox:
        """Exact image bbox of an upright cylinder standing on the floor at (x, y).

        A vertical segment projects to a segment, so the extremes lie on the top or bottom circle;
        the result is the union of both circles' exact bboxes. All-nan when the cylinder reaches
        the camera plane.
        """
        boxes = [self.bbox_of_circle(x, y, z, radius) for z in (0.0, float(height))]
        if any(math.isnan(b.x1) for b in boxes):
            return _NAN_BOX
        a, b = boxes
        return BBox(min(a.x1, b.x1), min(a.y1, b.y1), max(a.x2, b.x2), max(a.y2, b.y2))

    def bbox_of_circle(self, x: float, y: float, z: float, radius: float) -> BBox:
        """Exact image bbox of the horizontal circle (e.g. a rim) of ``radius`` centred at (x, y, z).

        All-nan when the circle reaches the camera plane.
        """
        near = float(self.depth([x, y, z])) - radius * math.hypot(self.R[2, 0], self.R[2, 1])
        if not near > _EPS_DEPTH:
            return _NAN_BOX
        # Plane homography of z = const: (X, Y, 1) -> K (r1 X + r2 Y + r3 z + t). With the circle
        # centred at the origin its dual conic is diag(-r^2, -r^2, 1): A = K [r1 r2], c = image of centre.
        A = self.K @ self.R[:, :2]
        return _outline_bbox(self.P @ np.array([x, y, z, 1.0]), A, radius)

    # ------------------------------------------------------------------ serialisation

    def to_dict(self) -> dict[str, Any]:
        return {"K": self.K.tolist(), "R": self.R.tolist(), "t": self.t.tolist(), "image_size": list(self.image_size)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PinholeCamera:
        return cls(K=np.array(d["K"]), R=np.array(d["R"]), t=np.array(d["t"]), image_size=tuple(d["image_size"]))


def _look_rotation(forward: np.ndarray) -> np.ndarray:
    """World->camera rotation with rows (right, down, forward) for world z up."""
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-9:  # looking straight up/down: fix "up" to world +y
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.vstack([right, down, forward])


def _tangent_roots(a: float, b: float, c: float) -> tuple[float, float]:
    """Roots of a s^2 - 2 b s + c = 0 (sorted); nan when the outline is unbounded."""
    disc = b * b - a * c
    if a <= 0 or disc < 0:
        return math.nan, math.nan
    r = math.sqrt(disc)
    return (b - r) / a, (b + r) / a


def _outline_bbox(center_h: np.ndarray, A: np.ndarray, radius: float) -> BBox:
    """Exact bbox of the imaged outline of a sphere or circle of ``radius``.

    The image dual conic is ``D = c c^T - r^2 A A^T`` (H&Z §8.3) with ``c`` the homogeneous image
    of the centre and ``A`` the imaged axes. It is formed in pixel coordinates centred on the
    projected centre (``c -> (0, 0, 1)``), which keeps the quadratic well conditioned. Vertical
    tangent lines u = s satisfy ``(1, 0, -s) D (1, 0, -s)^T = 0``; likewise for v.
    """
    w = float(center_h[2])
    ou, ov = float(center_h[0]) / w, float(center_h[1]) / w
    T = np.array([[1.0, 0.0, -ou], [0.0, 1.0, -ov], [0.0, 0.0, 1.0]])
    TA = T @ A / w
    D = -(radius**2) * (TA @ TA.T)
    D[2, 2] += 1.0
    u1, u2 = _tangent_roots(float(D[2, 2]), float(D[0, 2]), float(D[0, 0]))
    v1, v2 = _tangent_roots(float(D[2, 2]), float(D[1, 2]), float(D[1, 1]))
    return BBox(u1 + ou, v1 + ov, u2 + ou, v2 + ov)


def sees_full_court(camera: PinholeCamera, court: CourtSpec, margin_px: float = 0.0) -> bool:
    """True when all 4 court corners and both rims (full ring bbox) are inside the image."""
    corners = np.array([[0.0, 0.0, 0.0], [court.length, 0.0, 0.0], [0.0, court.width, 0.0],
                        [court.length, court.width, 0.0]])
    if not bool(np.all(camera.in_image(camera.project(corners), margin_px))):
        return False
    for side in ("left", "right"):
        bx, by = court.basket_xy(side)
        box = camera.bbox_of_circle(bx, by, court.rim_height, court.rim_radius)
        pts = np.array([[box.x1, box.y1], [box.x2, box.y2]])
        if not bool(np.all(camera.in_image(pts, margin_px))):
            return False
    return True
