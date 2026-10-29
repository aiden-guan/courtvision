"""BroadcastPerception with automatic court tracking: cut handling, the multi-frame detector's history and operator
calibrations (fakes for the wrapped pipeline and the court tracker)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from courtvision.geometry.court import court_for
from courtvision.geometry.homography import Calibration, Homography
from courtvision.perception.broadcast import BroadcastPerception, BroadcastPerceptionConfig
from courtvision.perception.broadcast_court import BroadcastCourtTracker, CourtFix
from courtvision.perception.court_search import CameraParams
from courtvision.types import FrameState

COURT = court_for("nba")
CENTER = np.array([47.0, -94.0, 36.0])
SIZE = (1920, 1080)
CAM = CameraParams(CENTER, 5.0, 16.0, 0.0, 4400.0)


class FakeDetector:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


class FakeInner:
    def __init__(self, calib: Calibration) -> None:
        self.config = SimpleNamespace(foot_direct=False)
        self.court = COURT
        self.calibration = calib
        self.detector = FakeDetector()
        self.track_resets = 0

    def reset_tracks(self) -> None:
        self.track_resets += 1

    def warp_image_state(self, M: np.ndarray) -> None:
        pass

    def process(self, frame_bgr: np.ndarray, frame_idx: int, t: float) -> FrameState:
        return FrameState(frame_idx, t, [], None, image_size=SIZE)


class FakeAuto:
    """Returns the queued court fixes in turn."""

    def __init__(self, statuses: list[str]) -> None:
        self.statuses = list(statuses)
        self.center = CENTER
        self.params: CameraParams | None = None
        self.reset_with: Homography | None = None

    def update(self, frame_bgr: np.ndarray, cut: bool = False) -> CourtFix:
        status = self.statuses.pop(0)
        self.params = None if status == "lost" else CAM
        return CourtFix(status, self.params, 0.8, 100, 1.0)

    def reset_to_homography(self, hom: Homography, size: tuple[int, int] | None = None) -> CameraParams | None:
        self.reset_with = hom
        return CAM


def perception(statuses: list[str], cuts: list[bool]) -> tuple[BroadcastPerception, FakeInner, FakeAuto]:
    calib = Calibration(Homography(np.linalg.inv(CAM.G(SIZE))), SIZE)
    inner = FakeInner(calib)
    auto = FakeAuto(statuses)
    bp = BroadcastPerception(inner, calib, None, BroadcastPerceptionConfig(), court_tracker=auto)  # type: ignore[arg-type]
    flags = list(cuts)
    bp.cuts.update = lambda frame: flags.pop(0)  # type: ignore[method-assign]
    return bp, inner, auto


FRAME = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)


def test_a_flagged_cut_through_which_the_tracker_keeps_the_camera_is_no_cut() -> None:
    bp, inner, _ = perception(["tracking", "tracking", "tracking"], [False, True, False])
    for i in range(3):
        bp.process(FRAME, i, i / 30.0)
    assert bp.stats["cuts"] == 0 and inner.track_resets == 0 and inner.detector.resets == 0
    assert bp.last_gate is not None and bp.last_gate.live


def test_the_detector_history_is_reset_at_a_real_cut_and_on_non_live_frames() -> None:
    bp, inner, _ = perception(["tracking", "lost", "lost", "reacquired", "tracking"], [False, True, False, False, False])
    resets = []
    for i in range(5):
        bp.process(FRAME, i, i / 30.0)
        resets.append(inner.detector.resets)
    assert resets == [0, 1, 2, 2, 2]  # the cut to a replay, the replay frame; nothing once live again
    assert bp.stats["cuts"] == 1


def test_an_operator_calibration_restarts_the_automatic_tracker_from_it() -> None:
    bp, _, auto = perception([], [])
    new = Calibration(Homography(np.linalg.inv(CameraParams(CENTER, -8.0, 15.0, 0.0, 3000.0).G(SIZE))), SIZE)
    bp.calibration = new
    assert auto.reset_with is new.homography


def test_the_court_tracker_restarts_from_a_homography_at_its_centre() -> None:
    tracker = BroadcastCourtTracker(COURT, SIZE, CENTER)
    truth = CameraParams(CENTER, -13.0, 15.5, 0.4, 4400.0)
    p = tracker.reset_to_homography(Homography(-2.0 * np.linalg.inv(truth.G(SIZE))))  # any scale, sign included
    assert p is not None and tracker.status == "tracking"
    assert abs(p.pan - truth.pan) < 1e-3 and abs(p.tilt - truth.tilt) < 1e-3 and abs(p.f / truth.f - 1) < 1e-4
    other = CameraParams(np.array([30.0, -40.0, 20.0]), 10.0, 22.0, 0.0, 2500.0)  # another camera
    assert tracker.reset_to_homography(Homography(np.linalg.inv(other.G(SIZE)))) is None
    assert tracker.status == "lost" and tracker.params is None
