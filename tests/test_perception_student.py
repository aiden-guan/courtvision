"""The student detector's input (perception/student.py) and its backends: Core ML on the Neural Engine, else torch."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from courtvision.perception import student


def _float_maths(frames: list[np.ndarray]) -> np.ndarray:
    """``preprocess_frames`` as it was written before its lookup tables."""
    h_in, w_in = student.INPUT_HW
    chans = []
    for f in frames:
        s = student.input_scale(f.shape[1], f.shape[0])
        wide = s == w_in / f.shape[1]
        size = (w_in if wide else round(f.shape[1] * s), round(f.shape[0] * s))
        img = cv2.resize(f, size, interpolation=cv2.INTER_AREA) if s != 1.0 else f
        if img.shape[0] < h_in or img.shape[1] < w_in:
            img = cv2.copyMakeBorder(img, 0, max(0, h_in - img.shape[0]), 0, max(0, w_in - img.shape[1]),
                                     cv2.BORDER_CONSTANT)
        rgb = cv2.cvtColor(img[:h_in, :w_in], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chans.append(((rgb - student.MEAN) / student.STD).transpose(2, 0, 1))
    return np.concatenate(chans, axis=0)


def test_preprocessing_is_the_float_maths_bit_for_bit() -> None:
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in ((1080, 1920), (720, 1280), (1080, 1440),
                                                                            (544, 960), (90, 160))]
    for f in frames:
        x = student.preprocess_frames([f])
        assert x.dtype == np.float32 and x.flags.c_contiguous and np.array_equal(x, _float_maths([f]))
    assert np.array_equal(student.preprocess_frames(frames[:3]), _float_maths(frames[:3]))


def test_auto_backend_falls_back_to_torch_and_an_unknown_one_is_refused(monkeypatch: pytest.MonkeyPatch,
                                                                        tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    weights = tmp_path / "w.pt"
    torch.save(torch.nn.Conv2d(9, 1, 1).state_dict(), weights)
    monkeypatch.setattr(student, "build_net", lambda frames, pretrained=True: torch.nn.Conv2d(9, 1, 1))

    def no_coreml(*a: object, **k: object) -> None:
        raise RuntimeError("no Core ML here")

    monkeypatch.setattr(student, "coreml_model", no_coreml)
    assert student.StudentDetector(student.StudentConfig(weights=str(weights), device="cpu")).backend == "torch"
    with pytest.raises(RuntimeError, match="no Core ML"):
        student.StudentDetector(student.StudentConfig(weights=str(weights), backend="coreml")).backend
    with pytest.raises(ValueError, match="unknown student backend"):
        student.StudentDetector(student.StudentConfig(weights=str(weights), backend="tpu")).backend


@pytest.mark.skipif(sys.platform != "darwin", reason="Core ML runs on macOS")
def test_coreml_export_is_cached_and_matches_the_torch_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    ct = pytest.importorskip("coremltools")
    torch.manual_seed(0)
    net = student.build_net(3, pretrained=False).eval()
    weights = tmp_path / "w.pt"
    torch.save({"model": net.state_dict()}, weights)
    ml = student.coreml_model(weights, cache=tmp_path / "cache")
    assert len(list((tmp_path / "cache").glob("student-*.mlpackage"))) == 1

    def no_second_export(*a: object, **k: object) -> None:
        raise AssertionError("exported again")

    monkeypatch.setattr(ct, "convert", no_second_export)
    student.coreml_model(weights, cache=tmp_path / "cache")  # loaded from the cache
    assert len(list((tmp_path / "cache").iterdir())) == 1
    x = student.preprocess_frames([np.random.default_rng(1).integers(0, 256, (540, 960, 3), dtype=np.uint8)] * 3)
    with torch.no_grad():
        want = {k: v.numpy() for k, v in net(torch.from_numpy(x[None])).items()}
    got = ml.predict({"x": x[None]})
    for k in student.OUTPUTS:  # fp16 on the Neural Engine: heatmap probabilities within a few thousandths
        a, b = np.asarray(got[k], np.float32), want[k]
        if k.endswith("_hm"):
            a, b = 1 / (1 + np.exp(-a)), 1 / (1 + np.exp(-b))
        assert a.shape == b.shape and float(np.abs(a - b).max()) < 0.02
