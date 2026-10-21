"""Broadcast perception: TV footage (a panning / zooming game camera, cuts, replays, a scorebug) -> FrameState.

``PerceptionPipeline`` assumes one calibrated camera that sees the court in every frame. ``BroadcastPerception``
wraps it for broadcast footage:

* **Court homography per frame** from ``CourtTracker`` (pan / tilt / zoom of the game camera, tracked from
  the court lines, started from the operator's one calibration). The wrapped pipeline gets a fresh
  ``Calibration`` every frame: that homography and the rims projected through the tracked camera.
* **Cuts** (``ShotBoundaryDetector``): a large colour-histogram change between consecutive frames
  (Bhattacharyya distance > ``cut_threshold``) is a cut. Image-space tracking state is reset at every cut
  (``PerceptionPipeline.reset_tracks``), so tracks never jump across one.
* **Camera-motion compensation.** Between live frames the image-space track and ball states are moved by
  the image -> image homography of the camera's turn / zoom about its fixed centre (``K R R'^T K'^-1``), so
  the image-space motion models see the players' own motion only (BoT-SORT style).
* **Live gate.** A frame is *live* when the court tracker holds the game camera and the scorebug is shown
  (a replay hides it; ``live_requires_scorebug``). Non-live frames (replays, close-ups, crowd shots) yield a
  ``FrameState`` with no players and no ball, so the event engine sees nothing and emits nothing from them;
  ``live`` / ``gate_reason`` record why (``last_gate``).
* **Scorebug** (``ScorebugReader``): read at ``scoreboard_hz`` (D14: 5 Hz) on live frames.

Thresholds are design choices tuned on simulator broadcast renders (``sim/broadcast.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from courtvision.geometry.homography import Calibration, Homography
from courtvision.types import BBox, FrameState

from .court_tracker import CourtTracker, CourtTrackerConfig, TrackResult, _camera, _rotation
from .pipeline import PerceptionPipeline
from .scorebug import ScorebugReader

__all__ = ["BroadcastPerception", "BroadcastPerceptionConfig", "GateState", "ShotBoundaryDetector"]

log = logging.getLogger(__name__)


@dataclass
class BroadcastPerceptionConfig:
    tracker: dict[str, Any] = field(default_factory=dict)  # CourtTrackerConfig overrides
    cut_threshold: float = 0.35  # Bhattacharyya distance of consecutive HSV histograms; tuned on sim renders
    live_requires_scorebug: bool = True  # a replay hides the scorebug
    scoreboard_hz: float = 5.0  # D14


@dataclass(frozen=True)
class GateState:
    live: bool
    reason: str  # "live" | "court_lost" | "no_scorebug"
    cut: bool
    track: TrackResult


class ShotBoundaryDetector:
    """Hard cuts from the HSV histogram of consecutive (downscaled) frames."""

    def __init__(self, threshold: float = 0.35) -> None:
        self.threshold = threshold
        self._prev: np.ndarray | None = None

    def update(self, frame_bgr: np.ndarray) -> bool:
        small = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        prev, self._prev = self._prev, hist
        if prev is None:
            return False
        return float(cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA)) > self.threshold


class BroadcastPerception:
    """``process(frame_bgr, frame_idx, t) -> FrameState`` for broadcast footage (see the module docstring).

    ``calibration`` is the operator's calibration of the game camera on one frame (its homography starts the
    court tracker; its rims are re-projected per frame); ``scorebug`` reads the broadcast graphic (None: no
    scoreboard reads and no scorebug gate)."""

    def __init__(self, inner: PerceptionPipeline, calibration: Calibration, scorebug: ScorebugReader | None = None,
                 config: BroadcastPerceptionConfig | None = None) -> None:
        self.inner = inner
        inner.config.foot_direct = True  # a new homography every frame: no per-frame correction table
        self.config = config or BroadcastPerceptionConfig()
        self.base = calibration
        tcfg = CourtTrackerConfig(**self.config.tracker)
        self.tracker = CourtTracker(inner.court, calibration.image_size, calibration.homography, tcfg)
        self.cuts = ShotBoundaryDetector(self.config.cut_threshold)
        self.scorebug = scorebug
        self._last_read_t: float | None = None
        self._was_live = True
        self._prev_ptz: tuple[float, float, float] | None = None
        self.last_gate: GateState | None = None
        self.stats = {"frames": 0, "live": 0, "cuts": 0, "court_lost": 0, "no_scorebug": 0}

    @property
    def calibration(self) -> Calibration:
        return self.inner.calibration

    @calibration.setter
    def calibration(self, calib: Calibration) -> None:
        """A new operator calibration of the game camera restarts the court tracker from it."""
        self.base = calib
        self.inner.calibration = calib
        self.tracker.reset(calib.homography)

    def __getattr__(self, name: str) -> Any:  # identity, set_scoreboard_roi, ... of the wrapped pipeline
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def _calibration(self, hom: Homography) -> Calibration:
        """The frame's calibration: the tracked homography and the rims seen through the tracked camera."""
        rims: dict[str, BBox] = {}
        center, ptz = self.tracker.center, self.tracker.ptz
        if center is not None and ptz is not None:
            cam = _camera(center, *ptz, self.base.image_size)
            court = self.inner.court
            for side in ("left", "right"):
                bx, by = court.basket_xy(side)
                box = cam.bbox_of_circle(bx, by, court.rim_height, court.rim_radius)
                w, h = self.base.image_size
                if np.all(np.isfinite([box.x1, box.y1, box.x2, box.y2])) and box.x2 > 0 and box.x1 < w \
                        and box.y2 > 0 and box.y1 < h:
                    rims[side] = box
        return Calibration(homography=hom, image_size=self.base.image_size, league=self.base.league, rims=rims)

    def gate(self, frame_bgr: np.ndarray) -> GateState:
        cut = self.cuts.update(frame_bgr)
        track = self.tracker.update(frame_bgr)
        court = track.homography is not None and track.status != "lost"
        bug = self.scorebug.present(frame_bgr) if self.scorebug is not None else True
        if not court:
            reason = "court_lost"
        elif not bug and self.config.live_requires_scorebug:
            reason = "no_scorebug"
        else:
            reason = "live"
        return GateState(reason == "live", reason, cut, track)

    def process(self, frame_bgr: np.ndarray, frame_idx: int, t: float) -> FrameState:
        h, w = frame_bgr.shape[:2]
        g = self.gate(frame_bgr)
        self.last_gate = g
        st = self.stats
        st["frames"] += 1
        st["cuts"] += int(g.cut)
        if g.cut or (g.live and not self._was_live):
            self.inner.reset_tracks()  # no image track survives a cut
        self._was_live = g.live
        if not g.live:
            self._prev_ptz = None
            st[g.reason] = st.get(g.reason, 0) + 1
            return FrameState(frame_idx, t, [], None, [], None, (w, h))
        st["live"] += 1
        assert g.track.homography is not None
        ptz = self.tracker.ptz
        if self._prev_ptz is not None and ptz is not None and not g.cut and self.tracker.center is not None:
            self.inner.warp_image_state(_rotation_homography(self._prev_ptz, ptz, self.base.image_size))
        self._prev_ptz = ptz
        self.inner.calibration = self._calibration(g.track.homography)
        state = self.inner.process(frame_bgr, frame_idx, t)
        state.scoreboard = self._read_scoreboard(frame_bgr, t)
        center = self.tracker.center
        if center is not None:
            state.camera_xyz = (float(center[0]), float(center[1]), float(center[2]))
        return state

    def _read_scoreboard(self, frame_bgr: np.ndarray, t: float) -> Any:
        if self.scorebug is None:
            return None
        hz = self.config.scoreboard_hz
        if self._last_read_t is not None and hz > 0 and t - self._last_read_t < 1.0 / hz - 1e-9:
            return None
        self._last_read_t = t
        return self.scorebug.read(frame_bgr, t)


def _rotation_homography(a: tuple[float, float, float], b: tuple[float, float, float],
                         size: tuple[int, int]) -> np.ndarray:
    """Image -> image homography of a camera turning / zooming about its centre from pan / tilt / focal ``a`` to
    ``b``: ``K_b R_b R_a^T K_a^-1`` (exact for every point, not only the floor)."""
    def K(f: float) -> np.ndarray:
        return np.array([[f, 0.0, size[0] / 2.0], [0.0, f, size[1] / 2.0], [0.0, 0.0, 1.0]])

    return K(b[2]) @ _rotation(b[0], b[1]) @ _rotation(a[0], a[1]).T @ np.linalg.inv(K(a[2]))
