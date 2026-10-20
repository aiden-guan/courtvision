"""RF-DETR adapter: PostProcess port on synthetic logits/boxes, CoreML and torch paths with fakes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from courtvision.perception.detector import Detector
from courtvision.perception.rfdetr_adapter import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    RFDETRConfig,
    RFDETRDetector,
    detections_from_arrays,
    postprocess,
    preprocess,
)
from courtvision.types import ObjClass


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def test_postprocess_topk_labels_and_box_scaling() -> None:
    boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.1, 0.1, 0.4, 0.4], [0.9, 0.5, 0.4, 0.2]])
    logits = np.array([
        [logit(0.9), logit(0.1), logit(0.2)],
        [logit(0.3), logit(0.8), logit(0.05)],
        [logit(0.05), logit(0.05), logit(0.7)],
    ])
    xyxy, scores, labels = postprocess(boxes, logits, (200, 100), num_select=4)
    assert np.allclose(scores, [0.9, 0.8, 0.7, 0.3])
    assert labels.tolist() == [0, 1, 2, 0]
    assert np.allclose(xyxy[0], [80, 30, 120, 70])  # query 0
    assert np.allclose(xyxy[1], [0, 0, 60, 30])  # query 1, clamped at the image origin
    assert np.allclose(xyxy[2], [140, 40, 200, 60])  # query 2, clamped at the right edge
    assert np.allclose(xyxy[3], xyxy[1])  # second class of query 1


def test_postprocess_background_slot_and_leading_dims() -> None:
    rng = np.random.default_rng(0)
    boxes = rng.uniform(0.2, 0.8, (2, 1, 6, 4))  # (decoder layers, batch, Q, 4): last layer is used
    logits = rng.normal(-3, 1, (2, 1, 6, 4))
    logits[-1, 0, 2, 3] = 9.0  # background slot (last) fires: must be ignored
    logits[-1, 0, 4, 1] = 5.0
    xyxy, scores, labels = postprocess(boxes, logits, (100, 100), num_select=5, background_class_id=-1)
    assert labels[0] == 1 and scores[0] == pytest.approx(1 / (1 + np.exp(-5.0)))
    assert 3 not in labels.tolist()
    with pytest.raises(ValueError):
        postprocess(np.zeros((5, 4)), np.zeros((6, 3)), (10, 10))


def test_postprocess_matches_rfdetr_reference() -> None:
    torch = pytest.importorskip("torch")
    pp_mod = pytest.importorskip("rfdetr.models.postprocess")
    rng = np.random.default_rng(1)
    q, c = 300, 5
    boxes = 1 / (1 + np.exp(-rng.normal(0, 1, (1, q, 4))))
    logits = rng.normal(-2, 1.5, (1, q, c))
    ref = pp_mod.PostProcess(num_select=300)(
        {"pred_logits": torch.tensor(logits, dtype=torch.float32), "pred_boxes": torch.tensor(boxes, dtype=torch.float32)},
        torch.tensor([[720, 1280]]),
    )[0]
    xyxy, scores, labels = postprocess(boxes, logits, (1280, 720), num_select=300)
    assert np.allclose(scores, ref["scores"].numpy(), atol=1e-6)
    assert np.array_equal(labels, ref["labels"].numpy())
    assert np.allclose(xyxy, ref["boxes"].numpy(), atol=1e-3)


def test_class_map_and_per_class_thresholds() -> None:
    xyxy = np.array([[0, 0, 10, 10], [5, 5, 9, 9], [1, 1, 3, 3], [0, 0, 5, 5], [2, 2, 2, 8]], float)
    scores = np.array([0.6, 0.35, 0.25, 0.9, 0.9])
    labels = np.array([0, 2, 2, 7, 0])
    cfg = RFDETRConfig()
    dets = detections_from_arrays(xyxy, scores, labels, cfg.class_names, cfg.class_map, cfg.thresholds)
    assert [(d.cls, d.score) for d in dets] == [(ObjClass.PLAYER, 0.6), (ObjClass.BALL, 0.35)]
    named = detections_from_arrays(xyxy[:2], scores[:2], labels[:2], None, cfg.class_map, cfg.thresholds,
                                   names=["person", "sports ball"])
    assert [d.cls for d in named] == [ObjClass.PLAYER, ObjClass.BALL]


def test_preprocess_layout_and_normalisation() -> None:
    frame = np.zeros((40, 60, 3), np.uint8)
    frame[:] = (0, 0, 255)  # pure red in BGR
    x = preprocess(frame, (32, 48))
    assert x.shape == (1, 3, 32, 48) and x.dtype == np.float32
    assert np.allclose(x[0, 0], (1 - IMAGENET_MEAN[0]) / IMAGENET_STD[0], atol=1e-5)  # R channel first
    assert np.allclose(x[0, 2], (0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2], atol=1e-5)


class FakeMLModel:
    """Duck-typed coremltools MLModel: spec + predict(dict)."""

    def __init__(self, boxes: np.ndarray, logits: np.ndarray) -> None:
        self.boxes, self.logits, self.calls = boxes, logits, []

    def get_spec(self):
        inp = SimpleNamespace(name="tensors", type=SimpleNamespace(multiArrayType=SimpleNamespace(shape=[1, 3, 64, 96])))
        outs = [SimpleNamespace(name="dets"), SimpleNamespace(name="labels")]
        return SimpleNamespace(description=SimpleNamespace(input=[inp], output=outs))

    def predict(self, data):
        self.calls.append(data)
        return {"dets": self.boxes, "labels": self.logits}


def test_coreml_path_with_fake_model() -> None:
    q = 10
    boxes = np.full((1, q, 4), 0.5)
    boxes[0, 0] = [0.25, 0.5, 0.1, 0.4]
    boxes[0, 1] = [0.75, 0.25, 0.02, 0.04]
    logits = np.full((1, q, 5), -8.0)  # 4 classes + background
    logits[0, 0, 0] = 4.0  # player
    logits[0, 1, 2] = 0.0  # ball, p = 0.5
    logits[0, 2, 4] = 8.0  # background: ignored
    fake = FakeMLModel(boxes, logits)
    det = RFDETRDetector(RFDETRConfig(backend="coreml"), model=fake)
    assert isinstance(det, Detector)
    fd = det.detect(np.zeros((360, 640, 3), np.uint8), 3, 0.1)
    (inp,) = fake.calls
    assert inp["tensors"].shape == (1, 3, 64, 96)
    assert [d.cls for d in fd.detections] == [ObjClass.PLAYER, ObjClass.BALL]
    p = fd.detections[0].bbox
    assert (p.x1, p.y1, p.x2, p.y2) == pytest.approx((128.0, 108.0, 192.0, 252.0))
    assert fd.image_size == (640, 360) and fd.frame_idx == 3


def test_torch_path_with_fake_model_uses_rgb_and_class_names() -> None:
    sv = pytest.importorskip("supervision")

    class FakeRFDETR:
        def predict(self, image, threshold, include_source_image):
            assert image[0, 0].tolist() == [255, 0, 0]  # the BGR red frame arrives as RGB
            assert threshold == 0.3 and include_source_image is False
            return sv.Detections(xyxy=np.array([[1, 2, 30, 80], [5, 5, 9, 9], [0, 0, 4, 4]], float),
                                 confidence=np.array([0.8, 0.4, 0.9]), class_id=np.array([1, 37, 0]),
                                 data={"class_name": np.array(["person", "sports ball", "car"])})

    frame = np.zeros((100, 100, 3), np.uint8)
    frame[:] = (0, 0, 255)  # red in BGR -> (255, 0, 0) in RGB
    det = RFDETRDetector(RFDETRConfig(), model=FakeRFDETR())
    dets = det.detect(frame, 0, 0.0).detections
    assert [(d.cls, d.score) for d in dets] == [(ObjClass.PLAYER, 0.8), (ObjClass.BALL, 0.4)]


def test_bad_config_is_rejected_without_imports() -> None:
    with pytest.raises(ValueError):
        RFDETRDetector(RFDETRConfig(variant="huge"))
    with pytest.raises(ValueError):
        RFDETRDetector(RFDETRConfig(backend="onnx"))
    with pytest.raises(ValueError, match="coreml_path"):
        RFDETRDetector(RFDETRConfig(backend="coreml")).detect(np.zeros((8, 8, 3), np.uint8), 0, 0.0)
