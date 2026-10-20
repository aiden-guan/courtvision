"""Ultralytics YOLO detector adapter. **R&D ONLY: Ultralytics is AGPL-3.0.**

AGPL-3.0 section 13 extends the source-offer obligation to network use, so this adapter belongs to
the ``rnd-agpl`` extra (RESEARCH.md D9). Shipped CourtVision code must never import it, and weights
trained with it must not ship without an Ultralytics Enterprise licence. The production detector is
``courtvision.perception.rfdetr_adapter.RFDETRDetector`` (Apache-2.0).

``ultralytics`` is imported lazily in ``__init__``. ``YOLO(weights).predict(frame_bgr, ...)``
returns ``Results`` whose ``boxes`` carry ``xyxy``, ``conf`` and ``cls`` tensors and whose
``names`` maps class ids to names (verified against ultralytics 8.4.183). Numpy frames are taken
as BGR, so frames pass through unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from courtvision.types import BBox, Detection, FrameDetections, ObjClass

__all__ = ["COCO_CLASS_MAP", "UltralyticsDetector", "parse_results"]

log = logging.getLogger(__name__)

COCO_CLASS_MAP: dict[str, ObjClass] = {"person": ObjClass.PLAYER, "sports ball": ObjClass.BALL}


def parse_results(result: Any, class_map: dict[str, ObjClass], names: dict[int, str] | None = None) -> list[Detection]:
    """Convert one Ultralytics ``Results`` into detections (classes missing from ``class_map`` drop)."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    names = names if names is not None else getattr(result, "names", {}) or {}
    xyxy = _numpy(boxes.xyxy).reshape(-1, 4)
    conf = _numpy(boxes.conf).reshape(-1)
    cls = _numpy(boxes.cls).reshape(-1).astype(int)
    out: list[Detection] = []
    for (x1, y1, x2, y2), score, cid in zip(xyxy, conf, cls):
        obj = class_map.get(str(names.get(int(cid), cid)))
        if obj is None or x2 <= x1 or y2 <= y1:
            continue
        out.append(Detection(bbox=BBox(float(x1), float(y1), float(x2), float(y2)), cls=obj, score=float(score)))
    return out


class UltralyticsDetector:
    """``Detector`` over an Ultralytics YOLO model (R&D only, see module docstring)."""

    def __init__(
        self,
        weights: str,
        class_map: dict[str, ObjClass] | None = None,
        conf: float = 0.25,  # Ultralytics default; sourced
        imgsz: int = 1280,  # native-ish resolution for the small ball; design
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        if model is None:
            from ultralytics import YOLO  # lazy: extra `rnd-agpl` (AGPL-3.0)

            model = YOLO(weights)
        self.model = model
        self.class_map = dict(class_map) if class_map is not None else dict(COCO_CLASS_MAP)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        log.warning("UltralyticsDetector is R&D only (AGPL-3.0); do not ship it")

    def detect(self, frame_bgr: np.ndarray, frame_idx: int, t: float) -> FrameDetections:
        kwargs: dict[str, Any] = {"conf": self.conf, "imgsz": self.imgsz, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model.predict(frame_bgr, **kwargs)
        dets: list[Detection] = []
        for r in results:
            dets += parse_results(r, self.class_map, getattr(self.model, "names", None))
        h, w = frame_bgr.shape[:2]
        return FrameDetections(frame_idx=frame_idx, t=t, image_size=(w, h), detections=dets)


def _numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)
