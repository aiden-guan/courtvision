"""tools/detector_bench.py: box matching, average precision and the SportsMOT reader."""

from __future__ import annotations

import json
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


def test_mcnemar_counts_discordant_frames_and_exact_p() -> None:
    a = [1, 1, 0, 0, 1, 0, 1, 1]
    b = [1, 0, 1, 1, 1, 1, 1, 1]
    r = db.mcnemar(a, b)
    assert (r["only_first"], r["only_second"]) == (1, 3)
    # two-sided exact binomial test, n = 4 discordant, k = 1: 2 * (C(4,0) + C(4,1)) / 2^4
    assert r["p_value"] == pytest.approx(2 * (1 + 4) / 16)
    assert db.mcnemar([1, 0], [1, 0])["p_value"] == 1.0


class _FakeSparse:
    """A ``run_model``-compatible model: fixed detections in frame px."""

    sparse = True

    def __init__(self, balls, persons) -> None:  # noqa: ANN001
        self.balls, self.persons = balls, persons

    def detect(self, x, scale, ball_thresh, person_floor):  # noqa: ANN001, ANN201
        return self.balls, self.persons


def test_coco_bench_scores_players_people_and_ball(tmp_path: Path) -> None:
    imgs = tmp_path / "heldout"
    imgs.mkdir()
    images = []
    for i in (1, 2):
        name = f"g-clip_mp4-000{i}_jpg.rf.h{i}.jpg"
        cv2.imwrite(str(imgs / name), np.zeros((108, 108, 3), np.uint8))
        images.append({"id": i, "file_name": name, "width": 108, "height": 108})
    cats = [{"id": 1, "name": "player"}, {"id": 2, "name": "referee"}, {"id": 3, "name": "ball"}]
    anns = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 40]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [60, 10, 20, 40]},
            {"id": 3, "image_id": 1, "category_id": 3, "bbox": [40, 40, 4, 4]}]
    ann = tmp_path / "heldout_ann.json"
    ann.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
    good = _FakeSparse([(42.0, 42.0, 0.9)], [(10, 10, 30, 50, 0.9), (60, 10, 80, 50, 0.8)])
    out = db.coco_bench(ann, imgs, {"good": good}, 0.3, 0.08, 15.0, "static", quiet=True)["good"]
    assert out["players"]["recall"] == pytest.approx(1.0) and out["people"]["recall"] == pytest.approx(1.0)
    # the same two boxes on both frames: one matches a player (frame 1), the referee box counts against players
    assert out["players"]["precision"] == pytest.approx(1 / 4)
    assert out["people"]["precision"] == pytest.approx(2 / 4)
    b = out["ball"]
    assert b["frames"] == 1 and b["hit_rate"] == pytest.approx(1.0)
    assert b["no_ball_frames"] == 1 and b["false_alarm_rate"] == pytest.approx(1.0)


def test_ball_bench_scores_top_candidates_and_linked_paths(tmp_path: Path) -> None:
    folder = tmp_path / "teacher"
    (folder / "frames" / "0000").mkdir(parents=True)
    rows, labels = [], []
    for f in range(1, 9):
        cv2.imwrite(str(folder / "frames" / "0000" / f"{f:06d}.jpg"), np.zeros((54, 96, 3), np.uint8))
        rows.append({"clip": 0, "frame": f, "size": [960, 540], "persons": [], "balls": []})
        labels.append({"clip": 0, "frame": f, "xy": [100.0 + 10 * f, 200.0], "w": 1.0, "source": "detected"})
    (folder / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (folder / "ball_labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labels))

    class Moving(_FakeSparse):
        """The ball where the label is, plus a brighter still distractor far away."""

        def __init__(self) -> None:
            super().__init__([], [])
            self.k = 0

        def detect(self, x, scale, ball_thresh, person_floor):  # noqa: ANN001, ANN201
            self.k += 1
            return [(800.0, 500.0, 0.9), (100.0 + 10 * ((self.k - 1) % 8 + 1), 200.0, 0.6)], []

    top = db.ball_bench([folder], {"m": Moving()}, 0.08, 15.0, 0.5, quiet=True)["m"]
    assert top["n"] == 6 and top["hit_rate"] == 0.0 and top["wrong_rate"] == 1.0  # the distractor scores higher
    linked = db.ball_bench([folder], {"m": Moving()}, 0.08, 15.0, 0.5, quiet=True, link=True)["m"]
    assert linked["n"] == 6
    assert linked["hit"] + linked["wrong"] + linked["miss"] == 6
