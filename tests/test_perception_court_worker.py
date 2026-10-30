"""The court tracker working ahead in a process of its own (perception/court_worker.py): the same fixes as in process,
and a fall back to in process when the frames disagree."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from courtvision.geometry.court import court_for
from courtvision.perception.broadcast import ShotBoundaryDetector
from courtvision.perception.broadcast_court import BroadcastCourtTracker
from courtvision.perception.court_worker import CourtTrackerProcess, frame_checksum


def write_video(path, frames: int = 40, size: tuple[int, int] = (320, 180)) -> list[np.ndarray]:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, size)
    rng = np.random.default_rng(3)
    for k in range(frames):
        img = np.full((size[1], size[0], 3), 40 + 4 * k, np.uint8)  # slow fades with a cut every 10 frames
        if k % 10 == 0:
            img[:] = rng.integers(0, 255, 3, dtype=np.uint8)
        writer.write(img)
    writer.release()
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, img = cap.read()
        if not ok:
            return out
        out.append(img)


@pytest.mark.parametrize("stride", [1, 2])
def test_the_worker_gives_the_in_process_fixes(tmp_path, stride):
    video = tmp_path / "v.avi"
    frames = write_video(video)[stride - 1::stride]  # as preanalysis.StridedVideoSource reads them: skip, then read
    court, center = court_for("nba"), np.array([47.0, -90.0, 35.0])
    local = BroadcastCourtTracker(court, (320, 180), center)
    worker = CourtTrackerProcess(court, (320, 180), center, video=str(video), stride=stride)
    cuts_a, cuts_b = ShotBoundaryDetector(), ShotBoundaryDetector()
    try:
        for k, img in enumerate(frames):
            a = local.update(img, cut=cuts_a.update(img))
            b = worker.update(img, cut=cuts_b.update(img), frame_idx=k)
            assert (a.status, a.params) == (b.status, b.params)
        assert worker.fallback_reason is None
    finally:
        worker.close()


def test_a_frame_the_worker_did_not_read_sends_the_tracking_back_in_process(tmp_path):
    video = tmp_path / "v.avi"
    frames = write_video(video, frames=12)
    court, center = court_for("nba"), np.array([47.0, -90.0, 35.0])
    worker = CourtTrackerProcess(court, (320, 180), center, video=str(video))
    cuts = ShotBoundaryDetector()
    try:
        for k in range(3):
            worker.update(frames[k], cut=cuts.update(frames[k]), frame_idx=k)
        other = frames[3].copy()
        other[:] = 255 - other  # not the frame the worker read
        assert frame_checksum(other) != frame_checksum(frames[3])
        worker.update(other, cut=cuts.update(other), frame_idx=3)
        assert worker.fallback_reason is not None and "not the analysis's frame" in worker.fallback_reason
        assert worker.update(frames[4], cut=False, frame_idx=4).status == "lost"  # in process from here on
    finally:
        worker.close()
