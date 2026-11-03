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


def _coco(tmp_path: Path) -> tuple[Path, Path]:
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    images, anns = [], []
    for i, name in enumerate(("gameA-clip_mp4-0000_jpg.rf.x.jpg", "gameA-clip_mp4-0001_jpg.rf.y.jpg",
                              "gameB-clip_mp4-0000_jpg.rf.z.jpg"), start=1):
        cv2.imwrite(str(imgs / name), np.full((96, 96, 3), 60 * i, np.uint8))
        images.append({"id": i, "file_name": name, "width": 96, "height": 96})
    cats = [{"id": 1, "name": "player"}, {"id": 2, "name": "referee"}, {"id": 3, "name": "ball"},
            {"id": 4, "name": "player-dribble"}, {"id": 5, "name": "number"}]
    for aid, (img, cat, box) in enumerate([(1, 1, [10, 10, 20, 40]), (1, 2, [50, 10, 20, 40]), (1, 3, [30, 30, 6, 6]),
                                           (2, 4, [5, 5, 20, 40]), (2, 5, [8, 8, 4, 4]), (3, 1, [1, 1, 9, 9])], start=1):
        anns.append({"id": aid, "image_id": img, "category_id": cat, "bbox": box})
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
    return ann, imgs


def test_prep_coco_keeps_players_and_human_balls(tmp_path: Path) -> None:
    ann, imgs = _coco(tmp_path)
    out = tmp_path / "coco"
    assert sf.prep_coco(ann, imgs, out, ["gameA"], []) == 2
    rows = [json.loads(line) for line in (out / "detections.jsonl").open()]
    labels = [json.loads(line) for line in (out / "ball_labels.jsonl").open()]
    assert [len(r["persons"]) for r in rows] == [1, 1]  # the referee and the jersey number are not people
    assert rows[0]["persons"][0][:4] == [10, 10, 30, 50] and rows[1]["persons"][0][:4] == [5, 5, 25, 45]
    assert labels[0]["xy"] == [33.0, 33.0] and labels[0]["w"] == 1.0
    assert labels[1]["xy"] is None and labels[1]["w"] == 0.0
    assert (out / "frames" / "0001" / "000001.jpg").is_file()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["kind"] == "human" and meta["video"].startswith("coco:")
    assert sf.prep_coco(ann, imgs, tmp_path / "none", [], ["gameA", "gameB"]) == 0


def test_temporal_dropout_repeats_the_current_frame(tmp_path: Path) -> None:
    human = _folder(tmp_path / "human", "human", "a", set(), persons=True)
    for f in range(1, 7):  # frames that differ, so a repeated frame is visible in the input
        img = np.full((540, 960, 3), 30 * f, np.uint8)
        cv2.imwrite(str(human / "frames" / "0000" / f"{f:06d}.jpg"), img)
    plan = plan_data([str(human)])
    always = sf.finetune_dataset(ClipDataset, 0.0, static_p=1.0)(plan.folders, plan.train | plan.val, train=True,
                                                                 weights=plan.weights, zoom_range=(1.0, 1.0))
    never = sf.finetune_dataset(ClipDataset, 0.0, static_p=0.0)(plan.folders, plan.train | plan.val, train=True,
                                                                weights=plan.weights, zoom_range=(1.0, 1.0))
    random.seed(3)
    x, _ = always[1]  # the sample at frame 6: inputs 4, 5, 6 unless dropped
    np.testing.assert_allclose(x[0:3], x[6:9], atol=1e-5)
    assert always.items[1][2] == [4, 5, 6]  # the item itself is left as it was
    random.seed(3)
    y, _ = never[1]
    assert not np.allclose(y[0:3], y[6:9], atol=1e-3)


def test_soup_averages_float_tensors_and_interpolates_with_the_start(tmp_path: Path) -> None:
    import torch

    base = {"w": torch.zeros(2, 2), "n": torch.tensor(7)}
    a = {"w": torch.full((2, 2), 2.0), "n": torch.tensor(7)}
    b = {"w": torch.full((2, 2), 4.0), "n": torch.tensor(7)}
    assert torch.equal(sf.soup_states([a, b])["w"], torch.full((2, 2), 3.0))
    wise = sf.soup_states([a, b], base, alpha=0.5)
    assert torch.equal(wise["w"], torch.full((2, 2), 1.5)) and int(wise["n"]) == 7
    with pytest.raises(ValueError):
        sf.soup_states([a, {"w": torch.zeros(3), "n": torch.tensor(7)}])
    for name, s in (("a", a), ("b", b), ("base", base)):
        torch.save({"model": s, "step": 1}, tmp_path / f"{name}.pt")
    out = tmp_path / "soup.pt"
    assert sf.main(["soup", str(tmp_path / "a.pt"), str(tmp_path / "b.pt"), "--base", str(tmp_path / "base.pt"),
                    "--alpha", "0.5", "--out", str(out)]) == 0
    got = torch.load(out, weights_only=True)
    assert torch.equal(got["model"]["w"], torch.full((2, 2), 1.5)) and got["alpha"] == 0.5


def test_prep_video_appends_batches_to_one_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import courtvision.perception.student_onnx as so

    class _Fake:
        def __init__(self, path: Path, threads: int = 4) -> None:
            pass

        def detect(self, x, scale, ball_thresh, person_thresh):  # noqa: ANN001, ANN201
            return [(640.0, 360.0, 0.9)], []

    monkeypatch.setattr(so, "OnnxStudent", _Fake)
    src, out = tmp_path / "clips", tmp_path / "folder"

    def video(name: str) -> None:
        (src / "g").mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(src / "g" / name), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 720))
        for _ in range(8):
            vw.write(np.full((720, 1280, 3), 90, np.uint8))
        vw.release()

    video("1.mp4")
    assert sf.prep_video(src, out, tmp_path / "m.onnx", 3, 6, 1) == 1
    identity = json.loads((out / "meta.json").read_text())["video"]
    (src / "g" / "1.mp4").unlink()  # a batch's videos deleted before the next is fetched
    video("2.mp4")
    sf.prep_video(src, out, tmp_path / "m.onnx", 3, 6, 1, append=True)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["video"] == identity and [Path(c["video"]).name for c in meta["clips"]] == ["1.mp4", "2.mp4"]
    rows = [json.loads(line) for line in (out / "detections.jsonl").open()]
    assert sorted({r["clip"] for r in rows}) == [0, 1] and len(rows) == 12
    assert (out / "frames" / "0001" / "000006.jpg").is_file()
    labels = [json.loads(line) for line in (out / "ball_labels.jsonl").open()]
    assert sorted({r["clip"] for r in labels}) == [0, 1]
    with pytest.raises(SystemExit):  # another labeller into the same folder
        sf.prep_video(src, out, tmp_path / "other.onnx", 3, 6, 1, append=True)


def test_prep_coco_can_count_referees_as_people(tmp_path: Path) -> None:
    ann, imgs = _coco(tmp_path)
    out = tmp_path / "coco_refs"
    assert sf.prep_coco(ann, imgs, out, ["gameA"], [], referees=True) == 2
    rows = [json.loads(line) for line in (out / "detections.jsonl").open()]
    assert [len(r["persons"]) for r in rows] == [2, 1]  # the referee is a person now; the jersey number still is not
    assert rows[0]["persons"][1][:4] == [50, 10, 70, 50]
    assert json.loads((out / "meta.json").read_text())["persons"] == "players and referees"


def test_foot_hull_keeps_people_among_the_players_and_drops_the_stands() -> None:
    players = np.array([[100, 300, 140, 400], [500, 320, 540, 420], [300, 500, 340, 600]], float)
    found = np.array([[300, 330, 340, 430],    # a referee between the players: unlabelled, inside the foot hull
                      [300, 50, 330, 120],     # a fan in the stands: far above the hull
                      [101, 301, 141, 401],    # one of the players again (overlaps a label)
                      [700, 330, 740, 430]],   # beyond the hull by more than the margin
                     float)
    boxes = sf.foot_hull_ignore(players, found, np.array([0.9, 0.9, 0.9, 0.9]), (1280, 720), margin=0.05)
    assert boxes == [[300.0, 330.0, 340.0, 430.0]]
    assert sf.foot_hull_ignore(players, found, np.array([0.2, 0.9, 0.9, 0.9]), (1280, 720)) == []  # below 0.3
    assert sf.foot_hull_ignore(players[:2], found, np.ones(4), (1280, 720)) == []  # too few players for a hull


def test_ignore_boxes_leave_person_cells_out_of_the_loss(tmp_path: Path) -> None:
    import torch

    from courtvision.training.train_student import ClipDataset, focal_loss

    human = _folder(tmp_path / "human", "human", "a", set(), persons=True)  # the person: (100, 100)-(160, 260)
    rows = [json.loads(line) for line in (human / "detections.jsonl").open()]
    for r in rows:
        r["ignore"] = [[600, 100, 700, 300]]
    (human / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    plan = plan_data([str(human)])
    ds = ClipDataset(plan.folders, plan.train | plan.val, train=False, weights=plan.weights)
    _, t = ds[0]
    valid = t["person_valid"][0]
    s = 960 / 1280 / 4  # label px -> output cells (stored at 960 px wide, stride 4)
    assert valid[int(200 * s), int(650 * s)] == 0.0  # inside the ignore box
    assert valid[int(180 * s), int(130 * s)] == 1.0 and valid[10, 10] == 1.0  # the labelled person, the background
    logits = torch.zeros((1, 1, 4, 4))
    target = torch.zeros((1, 1, 4, 4))
    target[0, 0, 0, 0] = 1.0
    w = torch.ones((1, 1, 4, 4))
    full = float(focal_loss(logits, target, None, w))
    w[0, 0, 2:, 2:] = 0.0
    assert float(focal_loss(logits, target, None, w)) < full  # four background cells no longer count
    assert float(focal_loss(logits, target)) == pytest.approx(full)


def test_mark_ignore_writes_ignore_boxes_and_links_the_frames(tmp_path: Path) -> None:
    src = tmp_path / "mot"
    (src / "frames" / "0000").mkdir(parents=True)
    rows = []
    players = [[100, 300, 140, 400, 1.0], [500, 320, 540, 420, 1.0], [300, 500, 340, 600, 1.0]]
    for f in range(1, 4):
        cv2.imwrite(str(src / "frames" / "0000" / f"{f:06d}.jpg"), np.full((540, 960, 3), 90, np.uint8))
        row = {"clip": 0, "frame": f, "size": [1280, 720], "persons": players, "balls": []}
        if f < 3:
            row["context"] = True
        rows.append(row)
    (src / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (src / "ball_labels.jsonl").write_text("".join(json.dumps({"clip": 0, "frame": f, "xy": None, "w": 0.0}) + "\n"
                                                   for f in range(1, 4)))
    (src / "meta.json").write_text(json.dumps({"kind": "human", "video": "mot:x"}))

    class Everyone:
        sparse = True

        def detect(self, x, scale, ball_thresh, person_floor):  # noqa: ANN001, ANN201
            return [], [tuple(p[:4]) + (0.9,) for p in players] + [(300, 330, 340, 430, 0.8), (300, 50, 330, 120, 0.8)]

    out = tmp_path / "mot_ign"
    assert sf.mark_ignore(src, out, Path("app.onnx"), model=Everyone()) == 1
    got = [json.loads(line) for line in (out / "detections.jsonl").open()]
    assert got[2]["ignore"] == [[300.0, 330.0, 340.0, 430.0]] and "ignore" not in got[0]  # samples only
    assert (out / "frames").is_symlink() and (out / "frames" / "0000" / "000003.jpg").is_file()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["video"] == "mot:x" and meta["ignore"]["boxes"] == 1
