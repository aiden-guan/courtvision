"""Parallel Live inference keeps frame history, cut/reset behaviour and current-camera box rectification."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from courtvision.geometry.homography import Calibration, Homography
from courtvision.perception.broadcast import BroadcastPerception, BroadcastPerceptionConfig
from courtvision.types import BBox, Detection, FrameDetections, FrameState, ObjClass, ScoreboardReading
from test_perception_broadcast_auto import CAM, FRAME, SIZE, FakeAuto, FakeInner


class HistoryDetector:
    parallel_court = True

    def __init__(self, fail: int | None = None):
        self.history = []
        self.fail = fail
        self.started = None
        self.court_started = None

    def reset(self):
        self.history.clear()

    def detect(self, image, idx, t):
        if self.started is not None:
            self.started.set()
            assert self.court_started.wait(5)
        self.history = (self.history + [idx])[-3:]
        if idx == self.fail:
            raise ValueError("inference failed")
        return FrameDetections(idx, t, SIZE, [Detection(BBox(800, 400, 900, 700), ObjClass.PLAYER,
                                                       sum(self.history) / 100.0)])


class Inner(FakeInner):
    def process(self, image, idx, t, *, detections=None):
        fd = detections if detections is not None else self.detector.detect(image, idx, t)
        values = np.array([[d.bbox.x1, d.bbox.y1, d.score] for d in fd.detections])
        return FrameState(idx, t, [], None, image_size=SIZE, balls=values)


def make_pipeline(statuses, cuts, parallel, fail=None):
    calib = Calibration(Homography(np.linalg.inv(CAM.G(SIZE))), SIZE)
    inner = Inner(calib)
    inner.detector = HistoryDetector(fail)
    auto = FakeAuto(statuses)
    auto.parallel_detection = parallel
    pipe = BroadcastPerception(inner, calib, config=BroadcastPerceptionConfig(rectify_boxes=True), court_tracker=auto)
    flags = iter(cuts)
    pipe.cuts.update = lambda image: next(flags)
    return pipe, inner, auto


def results(pipe, n):
    out = []
    for i in range(n):
        try:
            frame = pipe.process(FRAME, i, i / 30)
            out.append((pipe.last_gate.reason, None if frame.balls is None else frame.balls.tolist()))
        except ValueError as exc:
            out.append(str(exc))
    return out


@pytest.mark.parametrize("fail,lost", [(None, True), (2, True), (2, False)])
def test_parallel_frames_equal_serial_through_cuts_losses_and_inference_errors(fail, lost):
    statuses = ["tracking", "tracking", "lost" if lost else "tracking", "lost" if lost else "tracking",
                "reacquired", "tracking", "tracking"]
    cuts = [False, True, False, True, False, False, False]  # first cut is a flash: tracking survives
    serial, _, _ = make_pipeline(statuses, cuts, False, fail)
    parallel, _, _ = make_pipeline(statuses, cuts, True, fail)
    try:
        assert results(parallel, len(statuses)) == results(serial, len(statuses))
        pool = parallel._detector_worker
        assert pool is not None
    finally:
        serial.close()
        parallel.close()
    assert not any(thread.is_alive() for thread in pool._threads)


def test_detector_and_court_actually_overlap_on_the_same_frame():
    pipe, inner, auto = make_pipeline(["tracking"], [False], True)
    started, court_started = threading.Event(), threading.Event()
    inner.detector.started, inner.detector.court_started = started, court_started
    update = auto.update

    def wait_for_detection(*args, **kwargs):
        court_started.set()
        assert started.wait(5)
        return update(*args, **kwargs)

    auto.update = wait_for_detection
    try:
        assert pipe.process(FRAME, 0, 0.0).balls is not None
    finally:
        pipe.close()


def test_prepared_ocr_only_updates_state_on_the_original_live_frames():
    class Reader:
        def __init__(self):
            self.applied = []
            self.recognized = 0

        def present(self, image):
            return True

        def read_fields(self, image):
            self.recognized += 1
            return {"home": 37}

        def read_prepared(self, image, t, fields):
            self.applied.append(t)
            return ScoreboardReading(t, home_score=fields["home"])

        def read(self, image, t):
            return self.read_prepared(image, t, self.read_fields(image))

    statuses = ["tracking", "tracking", "lost", "lost", "reacquired", "tracking", "tracking"]
    pipes = [make_pipeline(statuses, [False] * len(statuses), mode)[0] for mode in (False, True)]
    readers = [Reader(), Reader()]
    boards = []
    try:
        for pipe, reader in zip(pipes, readers):
            pipe.scorebug, pipe.config.scoreboard_hz = reader, 30.0
            boards.append([pipe.process(FRAME, i, i / 30).scoreboard for i in range(len(statuses))])
        assert boards[0] == boards[1]
        assert readers[0].applied == readers[1].applied == [0.0, 1 / 30, 4 / 30, 5 / 30, 6 / 30]
        assert readers[1].recognized == readers[0].recognized + 1
        assert pipes[0]._last_read_t == pipes[1]._last_read_t
    finally:
        for pipe in pipes:
            pipe.close()
