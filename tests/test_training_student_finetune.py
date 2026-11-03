"""tools/student_finetune.py (data folders, augmentation) and tools/quantize_student.py (calibration reader)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import quantize_student as qs  # noqa: E402
import student_finetune as sf  # noqa: E402

from courtvision.training.train_student import ClipDataset, plan_data  # noqa: E402


def _folder(root: Path, kind: str, video: str, ball_frames: set[int], persons: bool) -> Path:
    """A ClipDataset folder: one clip of 6 frames (960x540), samples at frames 3 and 6."""
    root.mkdir(parents=True)
    (root / "frames" / "0000").mkdir(parents=True)
    rows, labels = [], []
    for f in range(1, 7):
        img = np.full((540, 960, 3), (40, 120, 200), np.uint8)
        cv2.imwrite(str(root / "frames" / "0000" / f"{f:06d}.jpg"), img)
        row = {"clip": 0, "frame": f, "size": [1280, 720], "balls": [],
               "persons": [[100, 100, 160, 260, 1.0]] if persons else []}
        if f % 3:
            row["context"] = True
        rows.append(row)
        labels.append({"clip": 0, "frame": f, "xy": [400.0, 300.0] if f in ball_frames else None,
                       "w": 0.5 if f in ball_frames else 0.0})
    (root / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "ball_labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labels))
    meta = {"kind": kind, "video": video}
    if kind == "pseudo":
        meta["person_weight"] = 0.0
    (root / "meta.json").write_text(json.dumps(meta))
    return root


def test_finetune_dataset_drops_label_free_samples_and_turns_hues(tmp_path: Path) -> None:
    human = _folder(tmp_path / "human", "human", "a", set(), persons=True)
    pseudo = _folder(tmp_path / "pseudo", "pseudo", "b", {3}, persons=False)  # frame 6 has no ball label
    plan = plan_data([str(human), str(pseudo)])
    assert [r["guarded_clips"] for r in plan.summary()] == [0, 0]  # distinct videos: nothing guarded
    cls = sf.finetune_dataset(ClipDataset, 40.0)
    ds = cls(plan.folders, plan.train | plan.val, train=True, weights=plan.weights)
    kept = sorted((fi, frames[-1]) for fi, _, frames, _, _ in ds.items)
    assert kept == [(0, 3), (0, 6), (1, 3)] and ds.dropped == 1  # the pseudo frame without a ball is gone
    random.seed(0)
    hues = {}
    for i, (fi, _, frames, _, _) in enumerate(ds.items):
        ds[i]
        hues[(fi, frames[-1])] = ds._hue
    assert hues[(1, 3)] == 0.0  # a frame with a ball label keeps its colours
    assert hues[(0, 3)] != 0.0 and abs(hues[(0, 3)]) <= 40.0
    eval_ds = cls(plan.folders, plan.train | plan.val, train=False, weights=plan.weights)
    eval_ds[0]
    assert eval_ds._hue == 0.0  # never at evaluation


def test_hue_turn_keeps_brightness_and_moves_hue(tmp_path: Path) -> None:
    human = _folder(tmp_path / "human", "human", "a", set(), persons=True)
    plan = plan_data([str(human)])
    ds = sf.finetune_dataset(ClipDataset, 40.0)(plan.folders, plan.train | plan.val, train=True,
                                                 weights=plan.weights)
    plain = ds._img(0, 0, 1)
    ds._hue = 30.0
    turned = ds._img(0, 0, 1)
    h0 = cv2.cvtColor(plain, cv2.COLOR_BGR2HSV)[..., 0].astype(int)
    h1 = cv2.cvtColor(turned, cv2.COLOR_BGR2HSV)[..., 0].astype(int)
    assert np.median((h1 - h0) % 180) == pytest.approx(15, abs=1)  # OpenCV hue is in half degrees
    v0 = cv2.cvtColor(plain, cv2.COLOR_BGR2HSV)[..., 2].astype(int)
    v1 = cv2.cvtColor(turned, cv2.COLOR_BGR2HSV)[..., 2].astype(int)
    assert abs(int(np.median(v1)) - int(np.median(v0))) <= 2


def test_calibration_reader_serves_ranges() -> None:
    xs = [np.full((1, 2), i, np.float32) for i in range(5)]
    r = qs._Reader(xs, "frames")
    assert len(r) == 5
    assert [int(r.get_next()["frames"][0, 0]) for _ in range(5)] == [0, 1, 2, 3, 4]
    assert r.get_next() is None
    r.set_range(2, 4)
    assert [int(r.get_next()["frames"][0, 0]) for _ in range(2)] == [2, 3] and r.get_next() is None
    r.rewind()
    assert int(r.get_next()["frames"][0, 0]) == 0
