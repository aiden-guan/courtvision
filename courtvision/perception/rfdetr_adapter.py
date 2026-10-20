"""RF-DETR detector: the production ``Detector`` (Apache-2.0; RESEARCH.md D9/D10, §7A.10).

Two backends behind one ``detect`` call:

* ``backend="torch"``: ``rfdetr.RFDETRNano/Small/Medium(pretrain_weights=...)`` (keyword-only
  ``ModelConfig`` fields, verified against rfdetr 1.11.2) and ``model.predict(rgb, threshold=...)``,
  which returns ``supervision.Detections`` (class names in ``data["class_name"]``).
* ``backend="coreml"``: ``coremltools.models.MLModel(path, compute_units=CPU_AND_NE)``. The input
  name and its fixed (1, 3, H, W) shape come from the spec. Preprocessing: RGB, [0, 1], bilinear
  resize without antialiasing (``cv2.INTER_LINEAR``, half-pixel centres, like
  ``predict``'s ``antialias=False``), ImageNet mean/std. Outputs are matched by position:
  ``(boxes, logits)``; a leading decoder-layer dimension, if present, is reduced to the last layer.
  ``postprocess`` is a numpy port of ``rfdetr.models.postprocess.PostProcess``: sigmoid, top
  ``num_select`` over the flattened (query x class) grid, ``query = idx // C``, ``label = idx % C``,
  cxcywh -> xyxy scaled by (W, H, W, H) and clamped.

Class ids map to ``ObjClass`` through ``class_names`` + ``class_map``; per-class thresholds then
filter. Nothing here imports ``rfdetr``, ``torch``, ``coremltools`` or ``supervision`` at module
import time, and tests inject fake models, so no weights are ever downloaded in tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from courtvision.types import BBox, Detection, FrameDetections, ObjClass

__all__ = ["RFDETRConfig", "RFDETRDetector", "detections_from_arrays", "postprocess", "preprocess"]

log = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)  # sourced (rfdetr export preprocessing)
IMAGENET_STD = (0.229, 0.224, 0.225)  # sourced (rfdetr export preprocessing)
_VARIANTS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium"}
_DEFAULT_RESOLUTION = {"nano": 384, "small": 512, "medium": 576}  # sourced (rfdetr config defaults)


@dataclass
class RFDETRConfig:
    """RF-DETR adapter settings."""

    variant: str = "small"  # "nano" | "small" | "medium"; Small is the default pick (RESEARCH §2.1)
    backend: str = "torch"  # "torch" | "coreml"
    weights: str | None = None  # fine-tuned checkpoint (torch); None => rfdetr's pretrained COCO weights
    coreml_path: str | None = None  # .mlpackage (coreml); one package per input size
    compute_units: str = "CPU_AND_NE"  # fp16 packages reach the ANE only with CPU_AND_NE/ALL; sourced
    device: str | None = None  # torch device; None => rfdetr's choice
    # Model class order (fine-tuned 0-based ids). COCO checkpoints report names in data["class_name"].
    class_names: list[str] = field(default_factory=lambda: ["player", "referee", "ball", "rim"])
    class_map: dict[str, str] = field(default_factory=lambda: {
        "player": "player", "referee": "referee", "ball": "ball", "rim": "rim",
        "person": "player", "sports ball": "ball",  # COCO fallback
    })
    thresholds: dict[str, float] = field(default_factory=lambda: {
        "player": 0.5, "referee": 0.5, "ball": 0.3, "rim": 0.5,  # ball lower: small target; tuned
    })
    num_select: int = 300  # sourced (rfdetr PostProcess default)
    background_class_id: int | None = -1  # exported final-slot background (rfdetr decode default); None for COCO


def preprocess(frame_bgr: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """BGR frame -> (1, 3, H, W) float32 ImageNet-normalised tensor."""
    h, w = size_hw
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    norm = (resized - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(IMAGENET_STD, np.float32)
    return np.ascontiguousarray(norm.transpose(2, 0, 1)[None], dtype=np.float32)


def postprocess(
    boxes_cxcywh: np.ndarray,
    logits: np.ndarray,
    image_size: tuple[int, int],
    num_select: int = 300,
    background_class_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of rfdetr ``PostProcess`` for one image. Returns (xyxy px, scores, labels), best first.

    ``boxes_cxcywh`` is (Q, 4) normalised; ``logits`` is (Q, C). Leading singleton batch dims are
    squeezed and a leading decoder-layer dim keeps the last layer. ``image_size`` is (W, H) of the
    original frame. ``background_class_id`` (e.g. -1) removes that class slot before selection.
    """
    boxes = _squeeze_to(np.asarray(boxes_cxcywh, dtype=np.float64), 2)
    logit = _squeeze_to(np.asarray(logits, dtype=np.float64), 2)
    if boxes.shape[0] != logit.shape[0] or boxes.shape[1] != 4:
        raise ValueError(f"boxes {boxes.shape} and logits {logit.shape} do not describe the same queries")
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -88.0, 88.0)))
    n_classes = prob.shape[1]
    class_ids = np.arange(n_classes)
    if background_class_id is not None:
        keep = np.ones(n_classes, dtype=bool)
        keep[background_class_id] = False
        prob, class_ids = prob[:, keep], class_ids[keep]
    flat = prob.reshape(-1)
    k = min(num_select, flat.size)
    order = np.argsort(-flat, kind="stable")[:k]  # descending score, ties by flattened index
    scores = flat[order]
    query = order // prob.shape[1]
    labels = class_ids[order % prob.shape[1]]
    cx, cy, bw, bh = boxes[query].T
    W, H = image_size
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1) * np.array([W, H, W, H])
    xyxy = np.clip(xyxy, 0.0, np.array([W, H, W, H], dtype=float))
    return xyxy, scores, labels.astype(np.int64)


def detections_from_arrays(
    xyxy: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None,
    class_map: dict[str, str],
    thresholds: dict[str, float],
    names: list[str] | np.ndarray | None = None,
) -> list[Detection]:
    """Map class ids/names to ``ObjClass`` and apply per-class thresholds (unmapped classes drop)."""
    out: list[Detection] = []
    for i in range(len(scores)):
        name = str(names[i]) if names is not None else _name_of(int(labels[i]), class_names)
        mapped = class_map.get(name)
        if mapped is None:
            continue
        cls = ObjClass(mapped)
        if float(scores[i]) < thresholds.get(cls.value, 0.5):
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(Detection(bbox=BBox(x1, y1, x2, y2), cls=cls, score=float(scores[i])))
    return out


class RFDETRDetector:
    """RF-DETR ``Detector`` (torch or CoreML backend). ``model`` injects a loaded model or a fake."""

    def __init__(self, config: RFDETRConfig | None = None, model: Any | None = None) -> None:
        self.config = config or RFDETRConfig()
        if self.config.variant not in _VARIANTS:
            raise ValueError(f"unknown RF-DETR variant {self.config.variant!r}; use one of {sorted(_VARIANTS)}")
        if self.config.backend not in ("torch", "coreml"):
            raise ValueError(f"unknown backend {self.config.backend!r}; use 'torch' or 'coreml'")
        self._model = model
        self._input_name: str | None = None
        self._input_hw: tuple[int, int] | None = None
        self._output_names: list[str] | None = None

    def detect(self, frame_bgr: np.ndarray, frame_idx: int, t: float) -> FrameDetections:
        h, w = frame_bgr.shape[:2]
        dets = self._detect_coreml(frame_bgr) if self.config.backend == "coreml" else self._detect_torch(frame_bgr)
        return FrameDetections(frame_idx=frame_idx, t=t, image_size=(w, h), detections=dets)

    # ------------------------------------------------------------------ torch backend

    def _detect_torch(self, frame_bgr: np.ndarray) -> list[Detection]:
        cfg = self.config
        model = self._torch_model()
        threshold = min(cfg.thresholds.values()) if cfg.thresholds else 0.5
        sv_dets = model.predict(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), threshold=threshold,
                                include_source_image=False)
        return self.from_supervision(sv_dets)

    def from_supervision(self, sv_dets: Any) -> list[Detection]:
        """Convert ``supervision.Detections`` (xyxy, confidence, class_id, data["class_name"])."""
        cfg = self.config
        n = len(sv_dets.xyxy)
        if n == 0:
            return []
        conf = np.ones(n) if sv_dets.confidence is None else np.asarray(sv_dets.confidence)
        labels = np.zeros(n, dtype=np.int64) if sv_dets.class_id is None else np.asarray(sv_dets.class_id)
        names = sv_dets.data.get("class_name") if getattr(sv_dets, "data", None) else None
        return detections_from_arrays(np.asarray(sv_dets.xyxy), conf, labels, cfg.class_names, cfg.class_map,
                                      cfg.thresholds, names)

    def _torch_model(self) -> Any:
        if self._model is None:
            import rfdetr  # lazy: extra `cv`

            cls = getattr(rfdetr, _VARIANTS[self.config.variant])
            kwargs: dict[str, Any] = {}
            if self.config.weights is not None:
                kwargs["pretrain_weights"] = self.config.weights
            if self.config.device is not None:
                kwargs["device"] = self.config.device
            log.info("loading %s (%s)", cls.__name__, self.config.weights or "pretrained COCO")
            self._model = cls(**kwargs)
        return self._model

    # ------------------------------------------------------------------ CoreML backend

    def _detect_coreml(self, frame_bgr: np.ndarray) -> list[Detection]:
        cfg = self.config
        model = self._coreml_model()
        assert self._input_name is not None and self._input_hw is not None
        outputs = model.predict({self._input_name: preprocess(frame_bgr, self._input_hw)})
        boxes, logits = self._match_outputs(outputs)
        h, w = frame_bgr.shape[:2]
        xyxy, scores, labels = postprocess(boxes, logits, (w, h), cfg.num_select, cfg.background_class_id)
        return detections_from_arrays(xyxy, scores, labels, cfg.class_names, cfg.class_map, cfg.thresholds)

    def _coreml_model(self) -> Any:
        if self._model is None:
            if self.config.coreml_path is None:
                raise ValueError("backend='coreml' needs RFDETRConfig.coreml_path")
            import coremltools as ct  # lazy: extra `cv` (rfdetr[coreml])

            units = getattr(ct.ComputeUnit, self.config.compute_units)
            log.info("loading CoreML package %s (compute_units=%s)", self.config.coreml_path, self.config.compute_units)
            self._model = ct.models.MLModel(self.config.coreml_path, compute_units=units)
        if self._input_name is None:
            self._read_spec(self._model)
        return self._model

    def _read_spec(self, model: Any) -> None:
        desc = model.get_spec().description
        inp = desc.input[0]
        self._input_name = inp.name
        shape = list(inp.type.multiArrayType.shape)
        if len(shape) == 4:
            self._input_hw = (int(shape[2]), int(shape[3]))
        else:
            r = _DEFAULT_RESOLUTION[self.config.variant]
            self._input_hw = (r, r)
            log.warning("CoreML input %s has shape %s; assuming %dx%d", inp.name, shape, r, r)
        self._output_names = [o.name for o in desc.output]

    def _match_outputs(self, outputs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Outputs by position: first the boxes ("dets"), then the logits ("labels")."""
        names = self._output_names or list(outputs)
        if len(names) < 2:
            raise ValueError(f"expected (boxes, logits) outputs, got {names}")
        a, b = np.asarray(outputs[names[0]]), np.asarray(outputs[names[1]])
        if a.shape[-1] != 4 and b.shape[-1] == 4:  # tolerate swapped order
            a, b = b, a
        return a, b


def _squeeze_to(arr: np.ndarray, ndim: int) -> np.ndarray:
    while arr.ndim > ndim:
        arr = arr[0] if arr.shape[0] == 1 else arr[-1]  # batch of 1, or decoder layers -> last
    return arr


def _name_of(label: int, class_names: list[str] | None) -> str:
    if class_names is not None and 0 <= label < len(class_names):
        return class_names[label]
    return str(label)
