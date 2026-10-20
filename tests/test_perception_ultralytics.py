"""R&D-only Ultralytics adapter: results parsing (no weights, no network) and the import fence."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from courtvision.perception.rnd.ultralytics_adapter import COCO_CLASS_MAP, UltralyticsDetector, parse_results
from courtvision.types import ObjClass

ROOT = Path(__file__).resolve().parents[1]


def test_parse_results_maps_names_and_drops_unknown() -> None:
    boxes = SimpleNamespace(
        xyxy=np.array([[10, 20, 50, 120], [60, 60, 70, 70], [0, 0, 5, 5], [9, 9, 9, 20]], float),
        conf=np.array([0.9, 0.4, 0.8, 0.7]),
        cls=np.array([0, 32, 2, 0]),
    )
    result = SimpleNamespace(boxes=_Sized(boxes, 4), names={0: "person", 32: "sports ball", 2: "car"})
    dets = parse_results(result, COCO_CLASS_MAP)
    assert [(d.cls, d.score) for d in dets] == [(ObjClass.PLAYER, 0.9), (ObjClass.BALL, 0.4)]
    assert parse_results(SimpleNamespace(boxes=None), COCO_CLASS_MAP) == []


class _Sized:
    def __init__(self, inner: SimpleNamespace, n: int) -> None:
        self._inner, self._n = inner, n

    def __len__(self) -> int:
        return self._n

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def test_yolo_from_yaml_on_dummy_frame() -> None:
    os.environ.setdefault("YOLO_OFFLINE", "1")  # skip Ultralytics' connectivity probe
    os.environ.setdefault("YOLO_CONFIG_DIR", tempfile.mkdtemp(prefix="yolo-cfg-"))
    pytest.importorskip("ultralytics")
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)  # YAML build = random init; seed it
    # A YAML-built model names its 80 classes "0".."79"; map them all so every box is parsed.
    class_map = {str(i): ObjClass.BALL if i == 32 else ObjClass.PLAYER for i in range(80)}
    det = UltralyticsDetector("yolo11n.yaml", class_map=class_map, conf=1e-6, imgsz=320, device="cpu")
    frame = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    fd = det.detect(frame, 4, 0.5)
    assert fd.frame_idx == 4 and fd.image_size == (320, 240)
    assert len(fd.detections) > 0  # untrained weights only clear a tiny threshold
    for d in fd.detections:
        assert d.cls in (ObjClass.PLAYER, ObjClass.BALL)
        assert 0 <= d.bbox.x1 < d.bbox.x2 <= 320 and 0 <= d.bbox.y1 < d.bbox.y2 <= 240
        assert 0.0 <= d.score <= 1.0


def test_nothing_outside_rnd_imports_ultralytics() -> None:
    pattern = re.compile(r"^\s*(from|import)\s+(ultralytics|courtvision\.perception\.rnd)\b|from\s+\.+rnd\b", re.M)
    offenders = []
    for path in (ROOT / "courtvision").rglob("*.py"):
        if "rnd" in path.relative_to(ROOT / "courtvision").parts:
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
    assert "AGPL" in (ROOT / "courtvision/perception/rnd/ultralytics_adapter.py").read_text().split('"""')[1]
