"""Appearance features: an HSV histogram of a player's torso with court colours suppressed.

The feature is the cheap fallback for team assignment and the optional appearance term of
the tracker (cv_pipeline.md §4.1: "an HSV colour histogram of the torso crop ... compared to
two reference histograms"; TrackID3x3 used colour histograms). Chromatic pixels vote into a
hue x saturation grid; achromatic pixels (low saturation, where hue is noise) vote into value
bins, so black referees and white jerseys still get a distinct feature. The vector is
L1-normalised; an all-zero vector means "no usable pixels".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from courtvision.sim.palette import FLOOR_BGR, LINE_BGR, NUMBER_BGR, PAINT_BGR
from courtvision.types import BBox

__all__ = ["AppearanceConfig", "color_feature", "feature_distance", "feature_size", "torso_feature"]

log = logging.getLogger(__name__)


@dataclass
class AppearanceConfig:
    """Torso-histogram settings. All values are design choices tuned on synthetic renders."""

    h_bins: int = 18  # hue bins over OpenCV's 0-180 hue range; tuned
    s_bins: int = 3  # saturation bins for chromatic pixels; tuned
    v_bins: int = 4  # value bins for achromatic pixels; tuned
    achromatic_s: int = 50  # S (0-255) below this => hue unreliable, vote by value; tuned
    # Central part of the box (Roboflow keeps "the central part" of each crop, cv_pipeline.md §1.1);
    # rows skip the head (top ~20 %) and the legs. Fractions of the bbox; tuned.
    torso_x: tuple[float, float] = (0.2, 0.8)
    torso_y: tuple[float, float] = (0.25, 0.65)
    # Court floor, paint, lines and the white jersey number are not jersey colour; tuned.
    suppress_bgr: tuple[tuple[int, int, int], ...] = (FLOOR_BGR, PAINT_BGR, LINE_BGR, NUMBER_BGR)
    suppress_tol: float = 30.0  # BGR Euclidean distance to a suppressed colour; tuned
    min_pixels: int = 4  # fewer usable pixels => zero vector; tuned


def feature_size(config: AppearanceConfig | None = None) -> int:
    """Length of the feature vector for ``config``."""
    cfg = config or AppearanceConfig()
    return cfg.h_bins * cfg.s_bins + cfg.v_bins


def torso_feature(
    frame_bgr: np.ndarray, bbox: BBox, config: AppearanceConfig | None = None, mask: np.ndarray | None = None
) -> np.ndarray:
    """HSV histogram of the torso region of ``bbox`` (court colours suppressed, L1-normalised).

    ``mask`` (frame-sized, non-zero = this person's pixels) restricts the histogram to the person's
    own pixels, so an occluder in front of the torso does not leak into the feature. Returns a
    float32 vector of length ``feature_size(config)``; all zeros when the region is empty or has
    fewer than ``min_pixels`` usable pixels.
    """
    cfg = config or AppearanceConfig()
    window = _torso_window(frame_bgr.shape[:2], bbox, cfg)
    if window is None:
        return np.zeros(feature_size(cfg), dtype=np.float32)
    y1, y2, x1, x2 = window
    pixels = frame_bgr[y1:y2, x1:x2].reshape(-1, 3)
    if mask is not None:
        pixels = pixels[mask[y1:y2, x1:x2].reshape(-1) > 0]
    return _histogram(pixels, cfg)


def color_feature(bgr: tuple[int, int, int] | np.ndarray, config: AppearanceConfig | None = None) -> np.ndarray:
    """Feature of a uniform patch of colour ``bgr`` (used for team colour anchors).

    Court-colour suppression is skipped, so an anchor always has a non-zero feature.
    """
    cfg = config or AppearanceConfig()
    patch = np.asarray(bgr, dtype=np.uint8).reshape(1, 3)
    return _histogram(patch, cfg, suppress=False, min_pixels=1)


def feature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance clipped to [0, 1]; 1.0 when either vector is zero."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 1.0
    return float(np.clip(1.0 - float(np.dot(a, b)) / (na * nb), 0.0, 1.0))


# ---------------------------------------------------------------------------- internals


def _torso_window(shape: tuple[int, int], bbox: BBox, cfg: AppearanceConfig) -> tuple[int, int, int, int] | None:
    h_img, w_img = shape
    x1 = bbox.x1 + cfg.torso_x[0] * bbox.w
    x2 = bbox.x1 + cfg.torso_x[1] * bbox.w
    y1 = bbox.y1 + cfg.torso_y[0] * bbox.h
    y2 = bbox.y1 + cfg.torso_y[1] * bbox.h
    ix1, iy1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    ix2, iy2 = min(w_img, int(np.ceil(x2))), min(h_img, int(np.ceil(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return iy1, iy2, ix1, ix2


def _histogram(
    pixels_bgr: np.ndarray, cfg: AppearanceConfig, suppress: bool = True, min_pixels: int | None = None
) -> np.ndarray:
    out = np.zeros(feature_size(cfg), dtype=np.float32)
    px = pixels_bgr.astype(np.float32)
    if suppress and cfg.suppress_bgr:
        keep = np.ones(len(px), dtype=bool)
        for c in cfg.suppress_bgr:
            keep &= np.linalg.norm(px - np.asarray(c, dtype=np.float32), axis=1) > cfg.suppress_tol
        pixels_bgr = pixels_bgr[keep]
    if len(pixels_bgr) < (cfg.min_pixels if min_pixels is None else min_pixels):
        return out
    hsv = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = hsv[:, 0].astype(int), hsv[:, 1].astype(int), hsv[:, 2].astype(int)
    chrom = s >= cfg.achromatic_s
    hb = np.minimum(h * cfg.h_bins // 180, cfg.h_bins - 1)
    sb = np.minimum((s - cfg.achromatic_s) * cfg.s_bins // max(1, 256 - cfg.achromatic_s), cfg.s_bins - 1)
    np.add.at(out, (hb * cfg.s_bins + sb)[chrom], 1.0)
    vb = np.minimum(v * cfg.v_bins // 256, cfg.v_bins - 1)
    np.add.at(out, cfg.h_bins * cfg.s_bins + vb[~chrom], 1.0)
    total = float(out.sum())
    return out / total if total > 0 else out
