"""Single-ball tracker: image-space Kalman filter with a vertical acceleration (gravity) term.

State ``[u, v, vu, vv, av]`` in px, px/s and px/s^2 (cv_pipeline.md §6.1: "a Kalman filter with
state (x, y, vx, vy, ay) in image space, where ay absorbs projected gravity during flight").
``av`` is a random walk because projected gravity depends on depth and is zero while the ball is
held. Each frame the tracker:

1. predicts to ``t``;
2. keeps the detections inside the gate (Mahalanobis on the innovation, or within
   ``gate_min_px`` for bounces and catches, never beyond ``gate_max_px``) and updates on the best;
3. otherwise coasts: for up to ``max_gap_frames`` frames the predicted point is reported with
   ``interpolated=True`` and a decaying confidence, then the ball is reported not visible;
4. re-acquires on the highest-score detection after ``reacquire_after`` frames without a gated
   match (a ball that jumped, or a stale track), or at once when the frame has a single confident
   ball detection (a release or a fast pass that outran the gate).

Only one ball is tracked, so spare balls and false positives outside the gate are ignored.
``court_xy`` is the floor-homography projection of the image point (exact only near the floor).

Online, a gap can only be extrapolated. ``fill_gaps`` is the two-sided version for near-live
(delayed) or offline use: it re-fills short gaps by a quadratic fit through the observations on
both sides.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from courtvision.geometry.homography import Homography
from courtvision.types import BallState, Detection, ObjClass

__all__ = ["BallTracker", "BallTrackerConfig", "fill_gaps"]

log = logging.getLogger(__name__)


@dataclass
class BallTrackerConfig:
    """Ball Kalman settings (pixels and seconds). All tuned on synthetic flights unless noted."""

    meas_std_px: float = 0.7  # detector centre noise; tuned (synthetic blobs are sub-pixel accurate)
    accel_std_px_s2: float = 3000.0  # white acceleration on u and v (catches, bounces); tuned
    jerk_std_px_s3: float = 1500.0  # random walk of the vertical acceleration term; tuned
    init_vel_std_px_s: float = 600.0  # tuned
    init_acc_std_px_s2: float = 500.0  # tuned
    gate_chi2: float = 13.8  # chi-square, 2 dof, 99.9 %; sourced (standard table)
    gate_min_px: float = 30.0  # always accept within this distance of the prediction; tuned
    gate_max_px: float = 200.0  # never accept beyond this distance; tuned
    max_gap_frames: int = 6  # coast (interpolated=True) at most this many frames (0.2 s at 30 fps); tuned
    reacquire_after: int = 3  # frames without a gated match before jumping to a new detection; tuned
    reacquire_single_score: float = 0.85  # a lone detection this confident is taken at once (releases,
    #   fast passes); weaker or competing out-of-gate detections wait reacquire_after frames; tuned
    min_score: float = 0.2  # ignore weaker ball detections; tuned
    confidence_decay: float = 0.85  # per coasted frame; tuned
    radius_momentum: float = 0.7  # EMA of the detected radius; tuned
    fallback_dt_s: float = 1.0 / 30.0  # used when t does not advance; design


class BallTracker:
    """``update(ball_dets, frame_idx, t, homography | None) -> BallState``."""

    def __init__(self, config: BallTrackerConfig | None = None) -> None:
        self.config = config or BallTrackerConfig()
        self._x: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._t: float | None = None
        self._frames_since_obs = 0
        self._misses_with_dets = 0
        self._radius: float | None = None
        self._conf = 0.0

    def reset(self) -> None:
        """Forget the ball (e.g. at a broadcast cut)."""
        self._x, self._P, self._t = None, None, None
        self._frames_since_obs = self._misses_with_dets = 0
        self._radius, self._conf = None, 0.0

    @property
    def predicted_xy(self) -> tuple[float, float] | None:
        """Last filtered image position (for ROI re-detection), or None when not tracking."""
        if self._x is None:
            return None
        return float(self._x[0]), float(self._x[1])

    def update(
        self, ball_dets: list[Detection], frame_idx: int, t: float, homography: Homography | None = None
    ) -> BallState:
        """Advance to ``t`` with this frame's BALL detections and return the ball state."""
        cfg = self.config
        # A non-finite box would turn the filter state into NaN for good (every later detection
        # passes a NaN gate), so such detections are dropped here.
        dets = [d for d in ball_dets if d.cls is ObjClass.BALL and d.score >= cfg.min_score
                and all(math.isfinite(v) for v in d.bbox.as_tuple())]
        if self._x is not None and self._P is not None and self._t is not None:
            dt = t - self._t if t > self._t else cfg.fallback_dt_s
            self._x, self._P = self._predict(self._x, self._P, dt)
        self._t = t

        chosen = self._choose(dets)
        lone_confident = len(dets) == 1 and dets[0].score >= cfg.reacquire_single_score
        if chosen is None and dets and (
            self._x is None or lone_confident or self._misses_with_dets + 1 >= cfg.reacquire_after
        ):
            chosen = max(dets, key=lambda d: d.score)
            self._init(chosen)
        elif chosen is not None:
            self._correct(chosen)
        if chosen is not None:
            self._observed(chosen)
            return self._state(homography, interpolated=False)

        self._misses_with_dets = self._misses_with_dets + 1 if dets else 0
        self._frames_since_obs += 1
        if self._x is None or self._frames_since_obs > cfg.max_gap_frames:
            if self._x is not None and self._frames_since_obs > cfg.max_gap_frames:
                log.debug("frame %d: ball lost after %d frames", frame_idx, self._frames_since_obs)
                self._x, self._P = None, None
            return BallState(visible=False)
        self._conf *= cfg.confidence_decay
        return self._state(homography, interpolated=True)

    # ------------------------------------------------------------------ filter

    def _predict(self, x: np.ndarray, P: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        F = np.eye(5)
        F[0, 2] = dt
        F[1, 3] = dt
        F[1, 4] = 0.5 * dt * dt
        F[3, 4] = dt
        Q = np.zeros((5, 5))
        qa = cfg.accel_std_px_s2**2
        acc = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * qa
        Q[np.ix_([0, 2], [0, 2])] += acc
        Q[np.ix_([1, 3], [1, 3])] += acc
        qj = cfg.jerk_std_px_s3**2
        jerk = np.array([
            [dt**5 / 20, dt**4 / 8, dt**3 / 6],
            [dt**4 / 8, dt**3 / 3, dt**2 / 2],
            [dt**3 / 6, dt**2 / 2, dt],
        ]) * qj
        Q[np.ix_([1, 3, 4], [1, 3, 4])] += jerk
        return F @ x, F @ P @ F.T + Q

    def _innovation(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._x is not None and self._P is not None
        S = self._P[:2, :2] + np.eye(2) * self.config.meas_std_px**2
        return z - self._x[:2], S

    def _choose(self, dets: list[Detection]) -> Detection | None:
        if self._x is None or not dets:
            return None
        cfg = self.config
        best: tuple[float, Detection] | None = None
        for d in dets:
            y, S = self._innovation(np.array(d.bbox.center))
            dist_px = float(np.hypot(*y))
            m2 = float(y @ np.linalg.solve(S, y))
            if dist_px > cfg.gate_max_px or (m2 > cfg.gate_chi2 and dist_px > cfg.gate_min_px):
                continue
            cost = m2 - 2.0 * math.log(max(d.score, 1e-3))
            if best is None or cost < best[0]:
                best = (cost, d)
        return None if best is None else best[1]

    def _init(self, det: Detection) -> None:
        cfg = self.config
        u, v = det.bbox.center
        self._x = np.array([u, v, 0.0, 0.0, 0.0])
        self._P = np.diag([cfg.meas_std_px**2, cfg.meas_std_px**2, cfg.init_vel_std_px_s**2,
                           cfg.init_vel_std_px_s**2, cfg.init_acc_std_px_s2**2])
        self._radius = None

    def _correct(self, det: Detection) -> None:
        assert self._x is not None and self._P is not None
        y, S = self._innovation(np.array(det.bbox.center))
        K = np.linalg.solve(S, self._P[:2, :]).T  # P H^T S^-1 with H = [I 0]
        self._x = self._x + K @ y
        self._P = self._P - K @ S @ K.T

    def _observed(self, det: Detection) -> None:
        cfg = self.config
        self._frames_since_obs = 0
        self._misses_with_dets = 0
        self._conf = float(det.score)
        r = (det.bbox.w + det.bbox.h) / 4.0
        m = cfg.radius_momentum
        self._radius = r if self._radius is None else m * self._radius + (1.0 - m) * r

    def _state(self, homography: Homography | None, interpolated: bool) -> BallState:
        assert self._x is not None
        uv = (float(self._x[0]), float(self._x[1]))
        court_xy = None
        if homography is not None:
            xy = homography.image_to_court(np.array(uv), front_only=True)
            if np.all(np.isfinite(xy)):
                court_xy = (float(xy[0]), float(xy[1]))
        return BallState(
            visible=True,
            image_xy=uv,
            court_xy=court_xy,
            z=None,
            radius_px=self._radius,
            confidence=float(np.clip(self._conf, 0.0, 1.0)),
            interpolated=interpolated,
        )


def fill_gaps(
    states: list[BallState],
    times: list[float],
    max_gap: int = 6,
    homography: Homography | None = None,
    context: int = 3,
) -> list[BallState]:
    """Two-sided interpolation of short gaps (near-live / offline).

    Runs of at most ``max_gap`` states that are interpolated or invisible, with observed states on
    both sides, are replaced by a per-axis quadratic in time fitted to up to ``context`` observed
    states on each side. Context comes only from the observed runs touching the gap, never from
    across another gap (an earlier flight phase would bend the fit). Filled states have
    ``interpolated=True``. Other states are returned as is.
    """
    if len(states) != len(times):
        raise ValueError("states and times must have the same length")
    observed = [s.visible and not s.interpolated and s.image_xy is not None for s in states]
    out = list(states)
    i = 0
    while i < len(states):
        if observed[i]:
            i += 1
            continue
        j = i
        while j < len(states) and not observed[j]:
            j += 1
        if 0 < i and j < len(states) and j - i <= max_gap:
            before = _observed_run(observed, i - 1, -1, context)
            after = _observed_run(observed, j, 1, context)
            out[i:j] = _quadratic_fill(states, times, sorted(before + after), range(i, j), homography)
        i = j
    return out


def _observed_run(observed: list[bool], start: int, step: int, limit: int) -> list[int]:
    """Up to ``limit`` consecutive observed indices from ``start`` in direction ``step``."""
    run: list[int] = []
    k = start
    while 0 <= k < len(observed) and observed[k] and len(run) < max(1, limit):
        run.append(k)
        k += step
    return run


def _quadratic_fill(
    states: list[BallState], times: list[float], support: list[int], gap: range, homography: Homography | None
) -> list[BallState]:
    t0 = times[support[0]]
    ts = np.array([times[k] - t0 for k in support])
    uv = np.array([states[k].image_xy for k in support], dtype=float)
    deg = min(2, len(support) - 1)
    cu, cv = np.polyfit(ts, uv[:, 0], deg), np.polyfit(ts, uv[:, 1], deg)
    radii = [states[k].radius_px for k in support if states[k].radius_px is not None]
    conf = min(states[support[0]].confidence, states[support[-1]].confidence)
    filled = []
    for k in gap:
        tk = times[k] - t0
        uv_k = (float(np.polyval(cu, tk)), float(np.polyval(cv, tk)))
        court_xy = None
        if homography is not None:
            xy = homography.image_to_court(np.array(uv_k), front_only=True)
            if np.all(np.isfinite(xy)):
                court_xy = (float(xy[0]), float(xy[1]))
        filled.append(BallState(visible=True, image_xy=uv_k, court_xy=court_xy, z=None,
                                radius_px=float(np.mean(radii)) if radii else None,
                                confidence=float(conf) * 0.8, interpolated=True))
    return filled
