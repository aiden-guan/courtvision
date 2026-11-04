"""Referee or player, from a person box's colours: a logistic regression on shirt and trouser statistics.

The detector has one person class; the pipeline tells referees apart afterwards (``nba.analyze.is_referee``). Its
first rule (an achromatic, mid-bright shirt with dark stripes) misses about a quarter of the referees and calls
players in dark or grey jerseys referees. This classifier reads two parts of the box, the shirt (rows 20-50 %, the
middle half) and the trousers (rows 55-85 %, the middle 40 %), each resampled to a fixed size (``SHIRT_HW``,
``TROUSERS_HW``: the frame's resolution does not change the statistics): saturation and brightness statistics, the
share of dark, of bright grey and of coloured pixels, the coloured pixels' hues, and the brightness changes across
and down (stripes). All boxes of a frame are read at once (``referee_probs``).

Weights (``nba.referee_model``) come from ``tools/referee_classifier.py``, fitted on human labels of 8 NBA games
(Roboflow Universe "basketball-player-detection-2", CC BY 4.0; NBA footage, research use: docs/TRAINING.md §1).
Left out one game at a time it found 94.3 % of the referees against the rule's 73.9 % and called fewer players
referees (1,730 against 2,464 of 22,440): docs/research/cross_game_detector.md §11.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

SHIRT = (0.2, 0.5, 0.25, 0.75)  # rows from, to; columns from, to (shares of the box)
TROUSERS = (0.55, 0.85, 0.3, 0.7)
SHIRT_HW = (29, 20)  # each region resampled to this height and width before any statistic
TROUSERS_HW = (29, 16)
_STATS = ("s_median", "v_median", "v_p10", "v_p20", "v_p90", "dark", "bright_grey", "coloured", "grad_x", "grad_y",
          "hue_0", "hue_1", "hue_2", "hue_3", "hue_4", "hue_5")
FEATURE_NAMES = tuple(f"shirt_{n}" for n in _STATS) + tuple(f"trousers_{n}" for n in _STATS[:8])


def _region(frame_bgr: np.ndarray, box: tuple[float, float, float, float], part: tuple[float, float, float, float],
            hw: tuple[int, int]) -> np.ndarray | None:
    """``part`` of a box (shares of its height and width, the box clipped to the frame), resampled to ``hw``."""
    x1, y1, x2, y2 = box
    h, w = frame_bgr.shape[:2]
    xa, ya, xb, yb = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
    if xb - xa < 4 or yb - ya < 8:
        return None
    fy0, fy1, fx0, fx1 = part
    bw, bh = xb - xa, yb - ya
    r0, r1 = int(ya + fy0 * bh), max(int(ya + fy0 * bh) + 1, int(ya + fy1 * bh))
    c0, c1 = int(xa + fx0 * bw), max(int(xa + fx0 * bw) + 1, int(xa + fx1 * bw))
    return cv2.resize(frame_bgr[r0:r1, c0:c1], (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


def _quantiles(vals: np.ndarray, qs: tuple[float, ...]) -> np.ndarray:
    """``np.percentile(vals, q, axis=1)`` (the linear method) of N rows of 8-bit values for each ``q`` (100 * 0.5 is
    the median), from the rows' value counts: no sort. Returns (len(qs), N)."""
    n, px = vals.shape
    counts = np.bincount((vals.astype(np.int32) + 256 * np.arange(n, dtype=np.int32)[:, None]).ravel(),
                         minlength=256 * n).reshape(n, 256)
    cum = np.cumsum(counts, axis=1)

    def kth(k: int) -> np.ndarray:  # each row's k-th smallest value (from 0)
        return (cum <= k).sum(axis=1).astype(np.float64)

    out = []
    for q in qs:
        virtual = (px - 1) * (q / 100)
        if virtual >= px - 1:
            out.append(kth(px - 1))
            continue
        prev = math.floor(virtual)
        a, b, t = kth(prev), kth(prev + 1), virtual - prev
        out.append(b - (b - a) * (1 - t) if t >= 0.5 else a + (b - a) * t)
    return np.array(out)


def _stats(hsv: np.ndarray) -> np.ndarray:
    """The ``_STATS`` of one region of N boxes at once: ``hsv`` (N, rows, cols, 3), OpenCV's 8-bit HSV."""
    n = hsv.shape[0]
    flat = hsv.reshape(n, -1, 3)
    hh, s, v = flat[..., 0], flat[..., 1], flat[..., 2]
    px = flat.shape[1]
    col = s > 80
    hue = np.stack([np.count_nonzero(((hh // 30) == i) & col, axis=1) for i in range(6)], axis=1) / px
    sq = _quantiles(s, (50.0,))
    vq = _quantiles(v, (50.0, 10.0, 20.0, 90.0))
    vi = hsv[..., 2].astype(np.int16)
    gx = np.abs(np.diff(vi, axis=2)).mean(axis=(1, 2)) / 255
    gy = np.abs(np.diff(vi, axis=1)).mean(axis=(1, 2)) / 255
    return np.column_stack([sq[0] / 255, vq[0] / 255, vq[1] / 255, vq[2] / 255, vq[3] / 255,
                            np.count_nonzero(v < 60, axis=1) / px, np.count_nonzero((v > 160) & (s < 60), axis=1) / px,
                            np.count_nonzero(col, axis=1) / px, gx, gy, hue])


def referee_features_batch(frame_bgr: np.ndarray, boxes: Any) -> tuple[np.ndarray, np.ndarray]:
    """The ``FEATURE_NAMES`` of several person boxes of one frame (frame px) at once: (features of the readable
    boxes, their indices in ``boxes``); a box under 4 x 8 px inside the frame cannot be read."""
    shirts, trousers, keep = [], [], []
    for i, box in enumerate(boxes):
        sh = _region(frame_bgr, box, SHIRT, SHIRT_HW)
        if sh is None:
            continue
        shirts.append(sh)
        trousers.append(_region(frame_bgr, box, TROUSERS, TROUSERS_HW))
        keep.append(i)
    if not keep:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0, int)

    def hsv(crops: list[np.ndarray], hw: tuple[int, int]) -> np.ndarray:
        stack = np.stack(crops)
        return cv2.cvtColor(stack.reshape(-1, hw[1], 3), cv2.COLOR_BGR2HSV).reshape(stack.shape)

    return np.column_stack([_stats(hsv(shirts, SHIRT_HW)), _stats(hsv(trousers, TROUSERS_HW))[:, :8]]), \
        np.array(keep, int)


def referee_features(frame_bgr: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    """The ``FEATURE_NAMES`` of one person box (frame px), or None for a box under 4 x 8 px inside the frame."""
    feats, keep = referee_features_batch(frame_bgr, [box])
    return feats[0] if len(keep) else None


def referee_probs(frame_bgr: np.ndarray, boxes: Any, model: Any = None) -> np.ndarray:
    """``referee_prob`` of several boxes of one frame at once (NaN for a box too small to read)."""
    if model is None:
        from courtvision.nba import referee_model as model
    out = np.full(len(boxes), np.nan)
    feats, keep = referee_features_batch(frame_bgr, boxes)
    if len(keep):
        z = (feats - np.asarray(model.MEAN)) / np.asarray(model.STD)
        out[keep] = 1.0 / (1.0 + np.exp(-(z @ np.asarray(model.COEF) + float(model.BIAS))))
    return out


def referee_prob(frame_bgr: np.ndarray, box: tuple[float, float, float, float], model: Any = None) -> float | None:
    """The probability that a person box shows a referee (``model``: a module or object with ``MEAN``, ``STD``,
    ``COEF``, ``BIAS``; default ``nba.referee_model``), or None when the box is too small to tell."""
    p = float(referee_probs(frame_bgr, [box], model)[0])
    return None if np.isnan(p) else p


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 300
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """L2-regularised logistic regression on standardised features, the two classes weighted equally (so that 0.5 is
    the even point); Newton steps. Returns (mean, std, coefficients, bias)."""
    y = np.asarray(y, bool)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z1 = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.zeros(Z1.shape[1])
    sw = np.where(y, (len(y) - y.sum()) / max(1, y.sum()), 1.0)
    reg = np.diag(np.r_[np.ones(len(w) - 1), 0.0])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Z1 @ w))
        grad = Z1.T @ (sw * (p - y)) / len(y) + l2 * (reg @ w)
        hess = (Z1 * (sw * p * (1 - p))[:, None]).T @ Z1 / len(y) + l2 * reg
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return mu, sd, w[:-1], float(w[-1])


def referee_calls(frame_bgr: np.ndarray, boxes: Any, method: str = "shirt", model: Any = None) -> list[bool]:
    """Which person boxes of one frame show a referee: ``shirt`` (the default: ``is_referee_shirt``) or ``learned``
    (``referee_probs`` at the weights' threshold; ``model``: another fit, else the packaged ``nba.referee_model``; a
    box too small to read is not a referee). One entry point for the desktop pipeline (``analyze.are_referees``),
    the Studio and the browser app (``StudentConfig.referees``, ``browser.runtime``). The shirt rule is the default:
    the learned check reads the Cavaliers' black jerseys on the README's 2017 game as referees' shirts
    (``docs/research/cross_game_detector.md`` §10.10-10.12)."""
    if method == "shirt":
        return [is_referee_shirt(frame_bgr, b) for b in boxes]
    if method != "learned":
        raise ValueError(f"unknown referee method {method!r}")
    if model is None:
        from courtvision.nba import referee_model as model
    probs = referee_probs(frame_bgr, boxes, model) if len(boxes) else np.zeros(0)
    return [bool(p >= model.THRESHOLD) for p in probs]  # NaN (unreadable) compares False


def is_referee_shirt(frame_bgr: np.ndarray, box: tuple[float, float, float, float]) -> bool:
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
