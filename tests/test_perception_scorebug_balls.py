"""Ball detections centred in the broadcast scorebug: where the OCR reader's bug covers a frame, and the live path
(``BroadcastPerceptionConfig.drop_scorebug_balls``, the Studio) dropping them only when switched on and while the
reader sees the bug (fakes for the wrapped pipeline and the court tracker, as in test_perception_broadcast_auto)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from courtvision.config import load_config
from courtvision.geometry.homography import Calibration, Homography
from courtvision.perception.broadcast import BroadcastPerception, BroadcastPerceptionConfig, drop_balls_in
from courtvision.perception.scorebug_cover import DROP_BALLS_IN_BUG, frame_boxes
from courtvision.perception.scorebug_ocr import ESPN_2024, OCRScorebugReader, bug_boxes
from courtvision.types import BBox, Detection, FrameDetections, FrameState, ObjClass
from test_perception_broadcast_auto import CAM, FRAME, SIZE, FakeAuto, FakeInner

LOGO = (530.0, 979.0)  # a team logo in the ESPN 2024 bug
BALL = (960.0, 600.0)  # the game ball
R = 12.0  # StudentDetector's ball box half-size at 1080 lines


def ball(xy: tuple[float, float], score: float) -> Detection:
    return Detection(BBox(xy[0] - R, xy[1] - R, xy[0] + R, xy[1] + R), ObjClass.BALL, score)


class BallDetector:
    parallel_court = True  # raw inference may run while the court is fitted (BroadcastPerception's parallel path)

    def reset(self) -> None:
        pass

    def detect(self, image: np.ndarray, idx: int, t: float) -> FrameDetections:
        player = Detection(BBox(500.0, 880.0, 560.0, 1060.0), ObjClass.PLAYER, 0.8)  # partly behind the bug
        return FrameDetections(idx, t, SIZE, [ball(LOGO, 0.7), ball(BALL, 0.45), player])


class Inner(FakeInner):
    def process(self, image: np.ndarray, idx: int, t: float, *,
                detections: FrameDetections | None = None) -> FrameState:
        fd = detections if detections is not None else self.detector.detect(image, idx, t)
        self.people = [d for d in fd.detections if d.cls is ObjClass.PLAYER]
        balls = [((d.bbox.x1 + d.bbox.x2) / 2.0, (d.bbox.y1 + d.bbox.y2) / 2.0, d.score) for d in fd.of(ObjClass.BALL)]
        return FrameState(idx, t, [], None, image_size=SIZE, balls=np.array(balls, np.float32).reshape(-1, 3))


class Reader:
    """A scorebug reader that sees its bug or not; the ESPN 2024 bug's boxes (OCRScorebugReader.bug_boxes)."""

    def __init__(self, shown: bool = True) -> None:
        self.shown = shown
        self.ocr = OCRScorebugReader()

    def present(self, image: np.ndarray) -> bool:
        return self.shown

    def bug_boxes(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        return self.ocr.bug_boxes(image)

    def read(self, image: np.ndarray, t: float) -> None:
        return None


def pipeline(drop: bool, shown: bool = True, parallel: bool = False,
             rectify: bool = False) -> tuple[BroadcastPerception, Inner]:
    calib = Calibration(Homography(np.linalg.inv(CAM.G(SIZE))), SIZE)
    inner = Inner(calib)
    inner.detector = BallDetector()  # type: ignore[assignment]
    auto = FakeAuto(["tracking"] * 3)
    auto.parallel_detection = parallel  # type: ignore[attr-defined]
    cfg = BroadcastPerceptionConfig(drop_scorebug_balls=drop, live_requires_scorebug=False, rectify_boxes=rectify)
    bp = BroadcastPerception(inner, calib, Reader(shown), cfg, court_tracker=auto)  # type: ignore[arg-type]
    bp.cuts.update = lambda image: False  # type: ignore[method-assign]
    return bp, inner


def centres(state: FrameState) -> list[tuple[float, float]]:
    assert state.balls is not None
    return [(float(x), float(y)) for x, y, _ in state.balls]


def test_the_bug_covers_espns_box_and_lower_row_and_a_discovered_bug_in_each_frame_size() -> None:
    espn = OCRScorebugReader()
    assert espn.bug_boxes(np.zeros((1080, 1920, 3), np.uint8)) == [(468, 944, 1000, 125)]
    assert espn.bug_boxes(np.zeros((720, 1280, 3), np.uint8)) == [(312, 629, 667, 83)]  # x 2/3, rounded
    # a bug found top right on a 720p copy (test_nba_track_court_tool's TOP_RIGHT_720): its text crop and its static
    # region, each 10 px (of 1080 lines) wider on both sides and half its height taller above and below
    top = dataclasses.replace(ESPN_2024, name="discovered", ocr_box=(1000, 26, 240, 34), logo_box=(946, 30, 48, 26),
                              ref_size=(1280, 720))
    reader = OCRScorebugReader(top)
    assert reader.bug_boxes(np.zeros((720, 1280, 3), np.uint8)) == [(993, 9, 253, 68), (939, 17, 61, 52)]
    assert reader.bug_boxes(np.zeros((1080, 1920, 3), np.uint8)) == bug_boxes(top) == [(1490, 14, 380, 102),
                                                                                        (1409, 26, 92, 78)]


def test_drop_balls_in_drops_only_ball_detections_centred_in_the_boxes() -> None:
    fd = BallDetector().detect(FRAME, 0, 0.0)
    boxes = frame_boxes(bug_boxes(ESPN_2024), SIZE)
    out = drop_balls_in(fd, boxes)
    assert [d.cls for d in out.detections] == [ObjClass.BALL, ObjClass.PLAYER]  # the player over the bug stays
    assert out.detections[0].score == 0.45 and len(fd.detections) == 3  # the input is not changed
    assert drop_balls_in(fd, []) is fd and drop_balls_in(out, boxes) is out  # nothing to drop: the same detections


@pytest.mark.parametrize("drop,want", [(False, [LOGO, BALL]), (True, [BALL])])
def test_live_ball_detections_in_the_bug_are_dropped_when_switched_on(drop: bool, want: list) -> None:
    bp, inner = pipeline(drop)
    try:
        state = bp.process(FRAME, 0, 0.0)
    finally:
        bp.close()
    assert bp.last_gate is not None and bp.last_gate.live and bp.last_gate.scorebug
    assert centres(state) == want and len(inner.people) == 1


def test_the_bugs_ball_detections_stay_while_the_reader_does_not_see_the_bug() -> None:
    bp, _ = pipeline(True, shown=False)  # live without the bug (live_requires_scorebug off, as in the Studio)
    try:
        state = bp.process(FRAME, 0, 0.0)
    finally:
        bp.close()
    assert bp.last_gate is not None and bp.last_gate.live and not bp.last_gate.scorebug
    assert centres(state) == [LOGO, BALL]


@pytest.mark.parametrize("parallel", [False, True])
def test_detections_inferred_alongside_the_court_fit_and_rectified_are_filtered_too(parallel: bool) -> None:
    bp, inner = pipeline(True, parallel=parallel, rectify=True)
    try:
        states = [bp.process(FRAME, i, i / 30.0) for i in range(3)]
    finally:
        bp.close()
    assert all(centres(s) == [BALL] for s in states)
    assert len(inner.people) == 1 and inner.people[0].bbox.as_tuple() != (500.0, 880.0, 560.0, 1060.0)  # rectified


def test_the_studio_switch_is_a_config_key_whose_default_follows_the_shared_one() -> None:
    assert BroadcastPerceptionConfig().drop_scorebug_balls == DROP_BALLS_IN_BUG
    assert load_config().perception.broadcast_config.drop_scorebug_balls == DROP_BALLS_IN_BUG
    on = load_config(overrides=["perception.broadcast_config.drop_scorebug_balls=true"])
    assert on.perception.broadcast_config.drop_scorebug_balls is True
