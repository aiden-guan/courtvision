"""tools/detector_bench.py: box matching, average precision and the SportsMOT reader."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import detector_bench as db  # noqa: E402


def test_iou_matrix() -> None:
    a = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], float)
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    iou = db.iou_matrix(a, b)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[1, 0] == pytest.approx(25 / 175)
    assert iou[:, 1].tolist() == [0.0, 0.0]
    assert db.iou_matrix(np.zeros((0, 4)), b).shape == (0, 2)


def test_greedy_match_highest_score_claims_the_truth() -> None:
    gt = np.array([[0, 0, 10, 10]], float)
    pred = np.array([[0, 0, 10, 10], [0, 0, 10, 9]], float)
    tp = db.greedy_match(pred, np.array([0.4, 0.9]), gt)
    assert tp.tolist() == [False, True]  # the 0.9 box takes the only truth box; the other is a duplicate
    assert db.greedy_match(pred, np.array([0.4, 0.9]), np.zeros((0, 4))).tolist() == [False, False]


def test_average_precision() -> None:
    # two truths; ranked hits at 1 and 3, a false positive between them
    scores = np.array([0.9, 0.8, 0.7])
    tp = np.array([True, False, True])
    # all-point interpolation: recall 0.5 at precision 1, recall 1.0 at precision 2/3
    assert db.average_precision(scores, tp, 2) == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))
    assert db.average_precision(np.array([]), np.array([], bool), 3) == 0.0
    assert db.average_precision(scores, tp, 0) == 0.0


def test_tally_summary_counts_at_the_threshold() -> None:
    t = db.Tally(scores=[0.9, 0.5, 0.2], tp=[True, False, True], n_truth=4, ref_kept=[True, False, True])
    s = t.summary(0.3)
    assert s["recall"] == pytest.approx(1 / 4)
    assert s["precision"] == pytest.approx(1 / 2)
    assert s["n_pred_noref"] == 1 and s["precision_noref"] == pytest.approx(1.0)


def test_load_sequences_reads_mot_ground_truth(tmp_path: Path) -> None:
    seq = tmp_path / "val" / "v_x_c001"
    (seq / "img1").mkdir(parents=True)
    (seq / "gt").mkdir()
    for i in range(1, 4):
        cv2.imwrite(str(seq / "img1" / f"{i:06d}.jpg"), np.zeros((72, 128, 3), np.uint8))
    (seq / "img1" / "._000001.jpg").write_bytes(b"resource fork")  # macOS tar residue: ignored
    (seq / "gt" / "gt.txt").write_text("1, 0, 10, 20, 30, 40, 1, 1, 1\n3, 0, 0, 0, 5, 5, 1, 1, 1\n")
    seqs = db.load_sequences(tmp_path, ["val"], None)
    assert len(seqs) == 1 and len(seqs[0].frames) == 3 and seqs[0].size == (128, 72)
    np.testing.assert_allclose(seqs[0].gt[1], [[10, 20, 40, 60]])
    assert sorted(seqs[0].gt) == [1, 3]
    assert db.load_sequences(tmp_path, ["val"], {"other"}) == []
