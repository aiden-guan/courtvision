"""Image <-> court floor-plane homography and fixed-camera calibration files.

``Homography.H`` maps image pixels (u, v) to court feet (x, y) on the floor plane (z = 0).
Fitting is a Hartley-normalised DLT in numpy; with a RANSAC threshold the inlier set comes from
``cv2.findHomography(img, court, cv2.RANSAC, ransac_thresh_ft)`` (threshold in court feet, the
destination plane; D1 gate = 1.0 ft) and the DLT is refit on the inliers.

Fitted matrices are *oriented*: the homogeneous scale ``w`` is positive for the fitted image
points, so ``w <= 0`` marks pixels beyond the horizon (``image_to_court(..., front_only=True)``).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from courtvision.types import BBox, RimObservation, Side

from .court import CourtSpec

__all__ = ["Calibration", "Homography", "HomographyConfig", "fit_landmarks"]

log = logging.getLogger(__name__)

_INF_TOL = 1e-12  # |w| below this fraction of the row magnitude is "at infinity"


@dataclass
class HomographyConfig:
    """Fitting knobs. Defaults match OpenCV's and need no tuning for calibration-sized inputs."""

    ransac_max_iters: int = 2000  # cv2.findHomography default; sourced (OpenCV docs)
    ransac_confidence: float = 0.995  # cv2.findHomography default; sourced (OpenCV docs)
    degenerate_tol: float = 1e-9  # relative singular-value floor for collinear / rank-deficient input; tuned
    refine_inliers: bool = True  # re-score inliers with the refit H and refit once more; tuned


class Homography:
    """Maps image px -> court ft on the floor plane (3x3, ``H @ [u, v, 1]``)."""

    def __init__(self, H: np.ndarray) -> None:
        M = np.asarray(H, dtype=float)
        if M.shape != (3, 3) or not np.all(np.isfinite(M)):
            raise ValueError(f"homography must be a finite 3x3 matrix, got shape {M.shape}")
        sv = np.linalg.svd(M, compute_uv=False)
        if sv[-1] <= 1e-12 * sv[0]:
            raise ValueError("homography matrix is singular")
        self.H: np.ndarray = M
        self.H_inv: np.ndarray = np.linalg.inv(M)

    def __repr__(self) -> str:
        return f"Homography({np.array2string(self.H, precision=6)})"

    # ------------------------------------------------------------------ fitting

    @classmethod
    def fit(
        cls,
        image_pts: Any,
        court_pts: Any,
        ransac_thresh_ft: float | None = 1.0,
        config: HomographyConfig | None = None,
    ) -> tuple[Homography, np.ndarray, float]:
        """Fit image->court from >= 4 correspondences.

        Returns ``(homography, inlier_mask, rmse_ft)``; ``rmse_ft`` is the court-space RMS error
        over the inliers. ``ransac_thresh_ft=None`` fits all points (mask all True).
        Raises ValueError for fewer than 4 points, mismatched shapes, or degenerate (collinear)
        input.
        """
        cfg = config or HomographyConfig()
        if ransac_thresh_ft is not None and not ransac_thresh_ft > 0:
            raise ValueError("ransac_thresh_ft must be > 0 or None")
        img, court = _as_pairs(image_pts, court_pts)
        _check_spread(img, "image", cfg.degenerate_tol)
        _check_spread(court, "court", cfg.degenerate_tol)
        n = len(img)
        if ransac_thresh_ft is None or n == 4:
            mask = np.ones(n, dtype=bool)
        else:
            mask = _ransac_mask(img, court, float(ransac_thresh_ft), cfg)
        H = _dlt(img[mask], court[mask], cfg.degenerate_tol)
        if ransac_thresh_ft is not None and cfg.refine_inliers and n > 4:
            H, mask = _refine(H, img, court, mask, float(ransac_thresh_ft), cfg.degenerate_tol)
        hom = cls(H)
        err = _court_errors(hom, img[mask], court[mask])
        rmse = float(np.sqrt(np.mean(err**2)))
        log.debug("homography fit: %d/%d inliers, rmse %.4f ft", int(mask.sum()), n, rmse)
        return hom, mask, rmse

    # ------------------------------------------------------------------ mapping

    def image_to_court(self, pts: Any, front_only: bool = False) -> np.ndarray:
        """Pixels -> court ft. (2,) -> (2,), (N, 2) -> (N, 2); nan where the point maps to infinity.

        ``front_only=True`` also returns nan for pixels beyond the horizon (``w <= 0``); it needs an
        oriented matrix, as produced by ``fit`` and ``PinholeCamera.floor_homography``.
        """
        return _apply(self.H, pts, front_only)

    def court_to_image(self, pts: Any, front_only: bool = False) -> np.ndarray:
        """Court ft -> pixels. Same shapes and nan rules as ``image_to_court``."""
        return _apply(self.H_inv, pts, front_only)

    # ------------------------------------------------------------------ serialisation

    def to_json(self) -> dict[str, Any]:
        return {"H": self.H.tolist()}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Homography:
        if "H" not in d:
            raise ValueError("homography JSON needs an 'H' entry")
        return cls(np.asarray(d["H"], dtype=float))


# ---------------------------------------------------------------------------- fitting helpers


def _as_pairs(image_pts: Any, court_pts: Any) -> tuple[np.ndarray, np.ndarray]:
    img = np.asarray(image_pts, dtype=float)
    court = np.asarray(court_pts, dtype=float)
    for name, arr in (("image_pts", img), ("court_pts", court)):
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"{name} must have shape (N, 2), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
    if len(img) != len(court):
        raise ValueError(f"got {len(img)} image points but {len(court)} court points")
    if len(img) < 4:
        raise ValueError(f"a homography needs >= 4 point pairs, got {len(img)}")
    return img, court


def _check_spread(pts: np.ndarray, name: str, tol: float) -> None:
    centred = pts - pts.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)
    if sv[0] <= 0 or sv[1] <= tol * sv[0]:
        raise ValueError(f"degenerate input: {name} points are collinear or coincident")


def _normaliser(pts: np.ndarray) -> np.ndarray:
    """Hartley similarity: centroid to the origin, mean distance sqrt(2)."""
    c = pts.mean(axis=0)
    mean_dist = float(np.mean(np.linalg.norm(pts - c, axis=1)))
    s = math.sqrt(2.0) / mean_dist
    return np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])


def _dlt(src: np.ndarray, dst: np.ndarray, tol: float) -> np.ndarray:
    """Normalised DLT: H with dst ~ H @ src, oriented (w > 0 on src) and unit Frobenius norm."""
    if len(src) < 4:
        raise ValueError(f"a homography needs >= 4 inliers, got {len(src)}")
    # Inlier subsets are not checked by ``fit``; coincident points would divide by zero below.
    _check_spread(src, "image", tol)
    _check_spread(dst, "court", tol)
    Ts, Td = _normaliser(src), _normaliser(dst)
    s = _homogeneous(src) @ Ts.T
    d = _homogeneous(dst) @ Td.T
    zeros = np.zeros((len(s), 3))
    rows_u = np.hstack([s, zeros, -d[:, :1] * s])
    rows_v = np.hstack([zeros, s, -d[:, 1:2] * s])
    A = np.vstack([rows_u, rows_v])
    _, sv, vt = np.linalg.svd(A)
    if sv[7] <= tol * sv[0]:  # null space of dimension > 1: the correspondences do not fix H
        raise ValueError("degenerate input: correspondences do not determine a unique homography")
    H = np.linalg.inv(Td) @ vt[-1].reshape(3, 3) @ Ts
    hsv = np.linalg.svd(H, compute_uv=False)
    if hsv[-1] <= tol * hsv[0]:
        raise ValueError("degenerate input: fitted homography is singular (collinear points?)")
    w = _homogeneous(src) @ H[2]
    if np.median(w) < 0:
        H = -H
    return H / np.linalg.norm(H)


def _ransac_mask(img: np.ndarray, court: np.ndarray, thresh: float, cfg: HomographyConfig) -> np.ndarray:
    import cv2  # lazy: keeps court-only consumers free of the OpenCV import

    H, mask = cv2.findHomography(
        img.astype(np.float64),
        court.astype(np.float64),
        cv2.RANSAC,
        thresh,
        maxIters=cfg.ransac_max_iters,
        confidence=cfg.ransac_confidence,
    )
    if H is None or mask is None:
        raise ValueError("RANSAC could not find a homography (degenerate input?)")
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 4:
        raise ValueError(f"RANSAC kept only {int(inliers.sum())} inliers; need >= 4")
    return inliers


def _refine(
    H: np.ndarray, img: np.ndarray, court: np.ndarray, mask: np.ndarray, thresh: float, tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Re-score every point against the refit H; refit once if the inlier set changed.

    Keeps the RANSAC solution when the re-scored set is too small or degenerate, so this optional
    polish step can never turn a valid fit into an error.
    """
    err = _court_errors(Homography(H), img, court)
    new_mask = np.nan_to_num(err, nan=np.inf) <= thresh
    if new_mask.sum() < 4 or np.array_equal(new_mask, mask):
        return H, mask
    try:
        return _dlt(img[new_mask], court[new_mask], tol), new_mask
    except ValueError as exc:
        log.debug("homography refine skipped: %s", exc)
        return H, mask


def _court_errors(hom: Homography, img: np.ndarray, court: np.ndarray) -> np.ndarray:
    return np.linalg.norm(hom.image_to_court(img) - court, axis=1)


def _homogeneous(pts: np.ndarray) -> np.ndarray:
    return np.hstack([pts, np.ones((len(pts), 1))])


def _apply(M: np.ndarray, pts: Any, front_only: bool) -> np.ndarray:
    arr = np.asarray(pts, dtype=float)
    single = arr.ndim == 1
    if arr.shape[-1:] != (2,) or arr.ndim > 2:
        raise ValueError(f"points must have shape (2,) or (N, 2), got {arr.shape}")
    P = arr.reshape(-1, 2)
    hom = P @ M[:, :2].T + M[:, 2]
    w = hom[:, 2]
    row_mag = np.abs(P[:, 0] * M[2, 0]) + np.abs(P[:, 1] * M[2, 1]) + abs(M[2, 2])
    bad = ~(np.abs(w) > _INF_TOL * row_mag)
    if front_only:
        bad |= w <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = hom[:, :2] / w[:, None]
    out[bad] = np.nan
    return out[0] if single else out


# ---------------------------------------------------------------------------- landmarks


def fit_landmarks(
    court: CourtSpec,
    image_points: Any,
    names: list[str],
    ransac_thresh_ft: float | None = 1.0,
    config: HomographyConfig | None = None,
) -> tuple[Homography, np.ndarray, float]:
    """``Homography.fit`` against named ``court.landmarks()``; ValueError on an unknown name."""
    table = court.landmarks()
    unknown = [n for n in names if n not in table]
    if unknown:
        raise ValueError(f"unknown landmark(s) {unknown}; known: {sorted(table)}")
    court_pts = np.array([table[n] for n in names], dtype=float)
    return Homography.fit(image_points, court_pts, ransac_thresh_ft, config)


# ---------------------------------------------------------------------------- calibration file

_SIDES: tuple[Side, ...] = ("left", "right")


@dataclass
class Calibration:
    """Fixed-camera calibration: floor homography, image size, league and rim boxes."""

    homography: Homography
    image_size: tuple[int, int]  # (width, height) px
    league: str = "nba"
    rims: dict[str, BBox] = field(default_factory=dict)  # side -> image bbox (fixed camera)

    def __post_init__(self) -> None:
        bad = [k for k in self.rims if k not in _SIDES]
        if bad:
            raise ValueError(f"rim keys must be 'left'/'right', got {bad}")
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))

    def rim_observations(self) -> list[RimObservation]:
        """Calibrated rims as ``RimObservation``s (confidence 1)."""
        return [RimObservation(side=side, bbox=self.rims[side]) for side in _SIDES if side in self.rims]

    def to_dict(self) -> dict[str, Any]:
        return {
            "homography": self.homography.to_json(),
            "image_size": list(self.image_size),
            "league": self.league,
            "rims": {side: _bbox_to_dict(b) for side, b in self.rims.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Calibration:
        """Parse ``to_dict`` output; ValueError (never KeyError/TypeError) on a malformed dict."""
        try:
            rims = {side: _bbox_from_json(b) for side, b in (d.get("rims") or {}).items()}
            w, h = d["image_size"]
            image_size = (int(w), int(h))
            homography = Homography.from_json(d["homography"])
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"malformed calibration: {exc!r}") from exc
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError(f"calibration image_size must be positive, got {image_size}")
        return cls(homography=homography, image_size=image_size, league=str(d.get("league", "nba")), rims=rims)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        return cls.from_dict(json.loads(Path(path).read_text()))


def _bbox_to_dict(b: BBox) -> dict[str, float]:
    return {"x1": float(b.x1), "y1": float(b.y1), "x2": float(b.x2), "y2": float(b.y2)}


def _bbox_from_json(b: Any) -> BBox:
    if isinstance(b, dict):
        return BBox(float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]))
    x1, y1, x2, y2 = b  # tolerate [x1, y1, x2, y2]
    return BBox(float(x1), float(y1), float(x2), float(y2))
