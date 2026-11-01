"""Bounded inference ahead must preserve temporal evidence, including cuts, skips and replacements."""

import numpy as np
import pytest

from courtvision.perception import student


def detector(monkeypatch, step):
    monkeypatch.setattr(student, "INPUT_HW", (2, 2))

    def preprocess(fs, *, out=None):
        if out is None:
            out = np.empty((3, 2, 2), np.float32)
        out.fill(fs[0][0, 0, 0])
        return out

    monkeypatch.setattr(student, "preprocess_frames", preprocess)
    det = student.StudentDetector(student.StudentConfig(frame_step=step))
    seen = []

    def infer(x, sizes):
        seen.append(x.copy())
        return [([(float(x.sum()), 50.0, 0.9)], [])]

    det.infer_arrays = infer
    return det, seen


@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("change", ["none", "cut", "skip", "replace", "error"])
def test_prefetch_inputs_and_detections_equal_normal_stream(monkeypatch, step, change):
    serial, normal_inputs = detector(monkeypatch, step)
    ahead, ahead_inputs = detector(monkeypatch, step)
    images = [np.full((4, 6, 3), i % 255 + 1, np.uint8) for i in range(30)]
    for i in range(3):
        assert ahead.detect(images[i], i, i / 30).detections == serial.detect(images[i], i, i / 30).detections
    if change == "error":
        infer = ahead.infer_arrays

        def fail(*args):
            raise RuntimeError("transient prefetch failure")

        ahead.infer_arrays = fail
        ahead.prefetch(images[3], 3, 0.1)
        ahead.infer_arrays = infer
        assert not ahead._prefetched
    else:
        ahead.prefetch(images[3], 3, 0.1)
    if change == "cut":
        ahead.reset()
        serial.reset()
    elif change == "replace":
        images[3] = np.full((4, 6, 3), 19, np.uint8)
    start = 4 if change == "skip" else 3
    for i in range(start, len(images)):
        assert ahead.detect(images[i], i, i / 30).detections == serial.detect(images[i], i, i / 30).detections
        np.testing.assert_array_equal(ahead_inputs[-1], normal_inputs[-1])
    if change == "none":
        assert len(ahead_inputs) == len(normal_inputs)  # prefetch replaces an inference, not its temporal evidence
    assert len(ahead._frame_pool) <= ahead._buf.maxlen + 1


@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("change", ["none", "cut", "skip", "replace", "error"])
def test_three_frame_prefetch_rolls_back_from_last_consumed_frame(monkeypatch, step, change):
    serial, normal_inputs = detector(monkeypatch, step)
    ahead, ahead_inputs = detector(monkeypatch, step)
    images = [np.full((4, 6, 3), i + 1, np.uint8) for i in range(30)]
    assert ahead.detect(images[0], 0, 0) == serial.detect(images[0], 0, 0)
    for i in range(1, 4):
        assert ahead.prefetch(images[i], i, i / 30)
        assert ahead.last_raw == serial.last_raw  # future evidence is not published as the current result
    assert not ahead.prefetch(images[4], 4, 4 / 30)
    assert len(ahead._prefetched) == 3
    assert ahead.detect(images[1], 1, 1 / 30) == serial.detect(images[1], 1, 1 / 30)
    assert ahead.last_raw == serial.last_raw
    if change == "cut":
        ahead.reset()
        serial.reset()
    elif change == "replace":
        images[2] = np.full((4, 6, 3), 99, np.uint8)
    elif change == "error":
        infer = ahead.infer_arrays

        def fail(*args):
            raise RuntimeError("transient prefetch failure")

        ahead.infer_arrays = fail
        assert not ahead.prefetch(images[4], 4, 4 / 30)
        ahead.infer_arrays = infer
    start = 3 if change == "skip" else 2
    for i in range(start, len(images)):
        assert ahead.detect(images[i], i, i / 30) == serial.detect(images[i], i, i / 30)
        assert ahead.last_raw == serial.last_raw
        # Keep a full lookahead across repeated pool reuse. Identity matches must not infer twice.
        n = len(ahead_inputs)
        for j in range(i + 1, min(i + 4, len(images))):
            assert ahead.prefetch(images[j], j, j / 30)
            assert ahead.prefetch(images[j], j, j / 30)
        assert len(ahead_inputs) - n <= 3
    if change == "none":
        np.testing.assert_array_equal(ahead_inputs, normal_inputs)
    assert len(ahead._frame_pool) <= ahead._buf.maxlen + ahead.max_prefetch
