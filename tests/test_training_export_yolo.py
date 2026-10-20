"""Tests for courtvision.training.export_yolo (YOLO format, data.yaml, manifest, rights gate, review sampling)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from courtvision.sim.simulator import GameSimulator, SimConfig
from courtvision.training.export_yolo import (
    CLASS_NAMES,
    ExportConfig,
    ExportItem,
    ReviewConfig,
    RightsError,
    detections_from_frame_state,
    export_yolo,
    select_for_review,
    uncertainty_score,
    yolo_lines,
)
from courtvision.types import BBox, Detection, FrameDetections, ObjClass

W, H = 64, 48
_LINE = re.compile(r"^[0-3] (0|1|0\.\d{6}|1\.0{6}) (0|1|0\.\d{6}|1\.0{6}) (0\.\d{6}|1\.0{6}) (0\.\d{6}|1\.0{6})$")


def _dets() -> list[Detection]:
    return [
        Detection(BBox(10, 5, 20, 40), ObjClass.PLAYER, 0.9, embedding=np.ones(4)),
        Detection(BBox(30, 10, 34, 14), ObjClass.BALL, 0.35),
        Detection(BBox(50, 2, 58, 8), ObjClass.RIM, 0.95),
        Detection(BBox(40, 5, 48, 40), ObjClass.REFEREE, 0.6),
        Detection(BBox(0, 0, 5, 5), ObjClass.PLAYER, 0.2),  # below the pseudo-label threshold
        Detection(BBox(55, 30, 80, 60), ObjClass.PLAYER, 0.8),  # partly outside: clipped
    ]


def _items(rights: str = "own", allowed: bool = True, n_games: int = 3) -> list[ExportItem]:
    rng = np.random.default_rng(0)
    return [ExportItem(image=rng.integers(0, 255, (H, W, 3), dtype=np.uint8), detections=_dets(), game_id=f"game{g}",
                       video_id=f"vid{g}", frame_idx=10 * k, t=k / 30, rights_status=rights, label_origin="pseudo",
                       training_allowed=allowed, source="test")
            for g in range(n_games) for k in range(2)]


def test_yolo_lines_format() -> None:
    lines, kept = yolo_lines(_dets(), (W, H))
    assert len(lines) == len(kept) == 5  # low-score player dropped
    assert all(_LINE.match(line) for line in lines), lines
    cls, cx, cy, w, h = lines[0].split()
    assert int(cls) == 0 and float(cx) == pytest.approx(15 / W) and float(h) == pytest.approx(35 / H)
    clipped = lines[-1].split()
    assert float(clipped[1]) + float(clipped[3]) / 2 == pytest.approx(1.0)  # right edge clipped to the image
    human, _ = yolo_lines(_dets(), (W, H), label_origin="human")
    assert len(human) == 6  # human labels keep every box
    assert [int(line.split()[0]) for line in lines] == [0, 2, 3, 1, 0]
    with pytest.raises(ValueError):
        yolo_lines(_dets(), (0, H))


def test_export_layout_yaml_and_manifest(tmp_path: Path) -> None:
    res = export_yolo(_items(), tmp_path / "ds")
    root = Path(res.root)
    data = yaml.safe_load((root / "data.yaml").read_text())
    assert data["names"] == list(CLASS_NAMES) and data["nc"] == 4
    assert data["train"] == "train/images" and data["val"] == "valid/images"
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == res.n_images == 6
    for r in rows:
        assert {"rights_status", "label_origin", "label_conf", "training_allowed", "game_id", "frame_idx", "uncertainty"} <= set(r)
        assert (root / r["image"]).exists() and (root / r["label"]).exists()
        lines = (root / r["label"]).read_text().splitlines()
        assert len(lines) == r["n_boxes"] == 5 and all(_LINE.match(x) for x in lines)
        assert r["label_conf"] == pytest.approx(0.35)  # weakest kept box (the ball)
    splits = {}
    for r in rows:
        splits.setdefault(r["game_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in splits.values())  # a game never straddles splits
    assert {"train", "val"} == {next(iter(s)) for s in splits.values()}
    meta = json.loads((root / "export_meta.json").read_text())
    sha = hashlib.sha256((root / "manifest.jsonl").read_bytes()).hexdigest()
    assert meta["manifest_sha256"] == sha == res.manifest_sha256 and sha in (root / "data.yaml").read_text()
    assert res.class_counts == {"player": 12, "ball": 6, "rim": 6, "referee": 6}


def test_export_from_paths_and_frame_detections(tmp_path: Path) -> None:
    import cv2

    img_path = tmp_path / "src.png"
    cv2.imwrite(str(img_path), np.zeros((H, W, 3), dtype=np.uint8))
    item = ExportItem(image=img_path, detections=FrameDetections(0, 0.0, (W, H), _dets()), game_id="g", video_id="v",
                      frame_idx=0, rights_status="licensed", training_allowed=True)
    res = export_yolo([item], tmp_path / "ds", for_shipping=True)
    assert res.for_shipping and res.per_split == {"train": 1}
    assert (tmp_path / "ds" / "train" / "images" / "g_v_0000000.png").exists()


def test_rights_gate_refuses_shipping(tmp_path: Path) -> None:
    items = _items()
    items[3].rights_status = "research_only"
    out = tmp_path / "ship"
    with pytest.raises(RightsError, match="research_only"):
        export_yolo(items, out, for_shipping=True)
    assert not out.exists()  # nothing written
    no_consent = _items(allowed=False)
    with pytest.raises(RightsError, match="training_allowed"):
        export_yolo(no_consent, out, ExportConfig(for_shipping=True))
    ok = export_yolo(_items(rights="licensed"), out, for_shipping=True)
    assert ok.for_shipping and ok.rights_statuses == {"licensed": 6}
    rnd = export_yolo(items, tmp_path / "rnd")  # research export is allowed
    assert not rnd.for_shipping and rnd.rights_statuses == {"own": 5, "research_only": 1}


def test_unknown_vocabulary_always_refused(tmp_path: Path) -> None:
    items = _items()
    items[0].rights_status = "youtube"
    with pytest.raises(RightsError, match="unknown rights_status"):
        export_yolo(items, tmp_path / "x")
    items = _items()
    items[0].label_origin = "guess"
    with pytest.raises(RightsError, match="unknown label_origin"):
        export_yolo(items, tmp_path / "x")


def test_non_empty_directory_refused(tmp_path: Path) -> None:
    out = tmp_path / "ds"
    export_yolo(_items(rights="research_only", allowed=False), out)  # an R&D export
    with pytest.raises(FileExistsError):  # its images must not leak into a shipping export
        export_yolo(_items(rights="own"), out, for_shipping=True)


def test_duplicate_frames_refused(tmp_path: Path) -> None:
    items = _items(n_games=1)
    items[1].frame_idx = items[0].frame_idx
    with pytest.raises(ValueError, match="duplicate"):
        export_yolo(items, tmp_path / "x")


def test_uncertainty_score() -> None:
    sure = [Detection(BBox(0, 0, 5, 5), ObjClass.PLAYER, 0.95), Detection(BBox(0, 0, 2, 2), ObjClass.BALL, 0.9)]
    unsure = [Detection(BBox(0, 0, 5, 5), ObjClass.PLAYER, 0.5), Detection(BBox(0, 0, 5, 5), ObjClass.PLAYER, 0.7)]
    assert uncertainty_score(sure) == 0.0
    assert uncertainty_score(unsure) == pytest.approx(1.0 + 0.1)
    assert uncertainty_score(sure[:1], live=True) == pytest.approx(1.0)  # live play without a ball
    assert uncertainty_score(sure, live=True) == 0.0


def test_select_for_review_ranks_and_spreads() -> None:
    def item(frame: int, score: float, video: str = "v") -> ExportItem:
        return ExportItem(np.zeros((2, 2, 3), np.uint8), [Detection(BBox(0, 0, 1, 1), ObjClass.PLAYER, score)], "g", video, frame)

    items = [item(0, 0.5), item(5, 0.5), item(40, 0.6), item(80, 0.95), item(100, 0.3), item(3, 0.5, "w")]
    picked = select_for_review(items, k=10, config=ReviewConfig(min_gap_frames=15))
    # index 1 (frame 5 of v) is within 15 frames of frame 0 and is skipped; index 5 is video w; the
    # confident frame (index 3) is never picked
    assert picked == [0, 5, 2, 4]
    assert select_for_review(items, k=2) == [0, 5]
    assert select_for_review(items, k=10) == select_for_review(items, k=10)  # deterministic


def test_detections_from_sim_frame(tmp_path: Path) -> None:
    sim = GameSimulator(SimConfig(seconds=4.0, fps=10.0, seed=2)).run()
    frame = sim.frames[-1]
    dets = detections_from_frame_state(frame)
    classes = {d.cls for d in dets}
    assert ObjClass.PLAYER in classes and ObjClass.RIM in classes
    size = frame.image_size or sim.config.image_size
    lines, _ = yolo_lines(dets, size, label_origin="human")
    assert lines and all(_LINE.match(x) for x in lines)


def test_export_loads_in_rfdetr(tmp_path: Path) -> None:
    pytest.importorskip("rfdetr")
    from rfdetr.datasets.yolo import is_valid_yolo_dataset

    res = export_yolo(_items(rights="own"), tmp_path / "ds", for_shipping=True)
    assert is_valid_yolo_dataset(res.root)


def test_non_finite_boxes_are_not_labels(tmp_path: Path) -> None:
    # NaN survives min/max clipping: it was written as "0 nan 0.468750 nan 0.729167".
    nan, inf = float("nan"), float("inf")
    dets = [Detection(BBox(nan, 5, 20, 40), ObjClass.PLAYER, 0.9), Detection(BBox(10, 5, inf, 40), ObjClass.PLAYER, 0.9),
            Detection(BBox(1, 1, 10, 10), ObjClass.BALL, nan), Detection(BBox(30, 10, 34, 14), ObjClass.BALL, 0.6)]
    for origin in ("pseudo", "human"):
        lines, kept = yolo_lines(dets, (W, H), label_origin=origin)
        assert all(_LINE.match(x) for x in lines) and not any("nan" in x or "inf" in x for x in lines)
        assert all(np.isfinite(d.bbox.as_tuple()).all() for d in kept)
    item = ExportItem(np.zeros((H, W, 3), np.uint8), dets, "g", "v", 0, rights_status="own", training_allowed=True)
    export_yolo([item], tmp_path / "ds")
    row = json.loads((tmp_path / "ds" / "manifest.jsonl").read_text())
    assert row["n_boxes"] == 1 and row["label_conf"] == pytest.approx(0.6)  # finite, from the one real box
