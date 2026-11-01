"""Live court fitting consumes only submitted frames, agrees with local fits and releases bounded resources."""

from __future__ import annotations

from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest

from courtvision.perception.broadcast import ShotBoundaryDetector
from courtvision.perception.broadcast_court import BroadcastCourtTracker
from courtvision.perception.court_search import CameraParams
from courtvision.perception.live_court_worker import LiveCourtTrackerProcess
from test_perception_court_search import CENTER, COURT, render

SIZE = (320, 180)


def as_tuple(fix):
    return (fix.status, None if fix.params is None else fix.params.to_dict(), fix.coverage, fix.inliers, fix.rmse_px)


def test_live_worker_matches_local_on_exactly_the_submitted_frames_and_cleans_up() -> None:
    cam = CameraParams(CENTER, 8.0, 16.0, 0.0, 800.0)
    local = BroadcastCourtTracker(COURT, SIZE, CENTER)
    worker = LiveCourtTrackerProcess(COURT, SIZE, CENTER)
    local.reset(cam)
    worker.reset(cam)
    cuts = ShotBoundaryDetector()
    try:
        for i, idx in enumerate((0, 1, 4, 7, 8, 12)):
            # Frame numbers can skip: no file exists for the worker to read instead of these actual pixels.
            img = render(CameraParams(CENTER, 8.0 - i * 0.2, 16.0, 0.0, 800.0), size=SIZE, seed=i)
            cut = cuts.update(img)
            assert as_tuple(worker.update(img, cut=cut, frame_idx=idx)) == as_tuple(local.update(img, cut=cut))
        assert worker.fallback_reason is None
        name = worker._memory.name
        process = worker._proc
    finally:
        worker.close()
        worker.close()
    assert not process.is_alive()
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name)


def test_dead_live_worker_falls_back_without_waiting_for_a_response() -> None:
    img = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    worker = LiveCourtTrackerProcess(COURT, SIZE, CENTER)
    try:
        assert worker.update(img, frame_idx=0).status == "lost"
        worker._proc.terminate()
        worker._proc.join(timeout=5)
        assert worker.update(img, frame_idx=1).status == "lost"
        assert worker.fallback_reason == "the live court worker stopped answering"
        assert worker._memory is None
    finally:
        worker.close()


def test_operator_reset_releases_the_live_worker_and_preserves_the_new_calibration() -> None:
    img = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    worker = LiveCourtTrackerProcess(COURT, SIZE, CENTER)
    try:
        worker.update(img, frame_idx=0)
        cam = CameraParams(CENTER, -8.0, 16.0, 0.0, 820.0)
        worker.reset(cam)
        assert worker._memory is None and worker._local is not None
        assert worker.params.to_dict() == cam.to_dict()
    finally:
        worker.close()


def test_spawn_failure_releases_an_unstarted_process_and_shared_memory(monkeypatch) -> None:
    import multiprocessing as mp

    worker = LiveCourtTrackerProcess(COURT, SIZE, CENTER)
    names = []

    def fail_start(first):
        worker._memory = SharedMemory(create=True, size=64)
        names.append(worker._memory.name)
        worker._proc = mp.get_context("spawn").Process(target=print)
        raise RuntimeError("spawn unavailable")

    monkeypatch.setattr(worker, "_start", fail_start)
    try:
        assert worker.update(np.zeros((SIZE[1], SIZE[0], 3), np.uint8), frame_idx=0).status == "lost"
        assert "could not submit live frame" in worker.fallback_reason
        assert worker._proc is None and worker._memory is None
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=names[0])
    finally:
        worker.close()


@pytest.mark.parametrize("change", ["none", "replace", "skip", "cut"])
def test_arrived_frame_prefetch_commits_only_matching_frames_and_restores_complete_state(change) -> None:
    size = (960, 540)
    cam = CameraParams(CENTER, 8.0, 16.0, 0.0, 2300.0)
    images = [render(CameraParams(CENTER, 8.0 - i * 0.2, 16.0, 0.0, 2300.0), size=size, seed=i)
              for i in range(10)]
    local = BroadcastCourtTracker(COURT, size, CENTER)
    worker = LiveCourtTrackerProcess(COURT, size, CENTER)
    local.reset(cam)
    worker.reset(cam)
    try:
        for i in range(9):
            actual_idx = i * 2 if change == "skip" else i
            assert as_tuple(worker.update(images[i], frame_idx=actual_idx)) == as_tuple(local.update(images[i]))
            assert worker.status == "tracking"  # exercise camera velocity/history, not merely matching lost states
            assert worker.prefetch_available
            params = worker.params.to_dict()
            hypothetical = images[i + 1] if change in ("none", "cut") else np.zeros_like(images[i])
            assert worker.prefetch(hypothetical, actual_idx + 1, cut=change == "cut")
            assert not worker.prefetch_available
            assert not worker.prefetch(images[-1], 100)
            assert worker.params.to_dict() == params  # speculative camera/velocity is never published
        assert worker.fallback_reason is None
    finally:
        worker.close()
