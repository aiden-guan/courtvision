"""The student on ONNX Runtime: sparse offset/size heads give the full network's detections; ONNX -> torch weights."""

from __future__ import annotations

import numpy as np
import pytest

from courtvision.perception.student_onnx import SparseHead, decode_sparse, default_threads

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from courtvision.perception.student import INPUT_HW, OUTPUTS, build_net, decode  # noqa: E402
from courtvision.perception.student_onnx import OnnxStudent  # noqa: E402
from courtvision.training.student_from_onnx import _bn_after, load_onnx_weights, module_path  # noqa: E402


def _export(net, path) -> None:  # noqa: ANN001
    """The browser export's wrapper (``tools/export_student_browser.py``): node names under ``/student/``."""

    class Export(torch.nn.Module):
        def forward(self, x):  # noqa: ANN001, ANN202
            out = net(x)
            return tuple(out[name] for name in OUTPUTS)

    wrapper = Export()
    wrapper.add_module("student", net)
    wrapper.eval()
    torch.onnx.export(wrapper, torch.zeros((1, 9, *INPUT_HW)), str(path), input_names=["frames"],
                      output_names=list(OUTPUTS), opset_version=17, dynamo=False)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):  # noqa: ANN001, ANN201
    torch.manual_seed(0)
    net = build_net(3, pretrained=False).eval()
    with torch.no_grad():  # non-trivial batch norms, so folding them matters; heatmap biases low enough for peaks
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.uniform_(-0.2, 0.2)
                m.running_var.uniform_(0.5, 2.0)
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.1, 0.1)
        net.ball_hm[-1].bias.fill_(0.0)  # sigmoid ~0.5: plenty of peaks for the decoder
        net.person_hm[-1].bias.fill_(0.0)
    path = tmp_path_factory.mktemp("student") / "student.onnx"
    _export(net, path)
    x = np.random.default_rng(1).standard_normal((1, 9, *INPUT_HW)).astype(np.float32)
    return net, path, x


def test_sparse_heads_reproduce_the_full_network(exported) -> None:  # noqa: ANN001
    _, path, x = exported
    sparse, dense = OnnxStudent(path, threads=2, sparse=True), OnnxStudent(path, threads=2, sparse=False)
    full = dense.run(x)
    rebuilt = sparse.dense(x)
    for k in OUTPUTS:
        assert rebuilt[k].shape == full[k].shape
        np.testing.assert_allclose(rebuilt[k], full[k], atol=2e-4, rtol=1e-4)
    for thr in (0.3, 0.5):
        b1, p1 = sparse.detect(x, 0.5, ball_thresh=thr, person_thresh=thr)
        b2, p2 = dense.detect(x, 0.5, ball_thresh=thr, person_thresh=thr)
        assert len(b1) == len(b2) and len(p1) == len(p2)
        assert len(p1) > 0  # the test exercises the decoder
        np.testing.assert_allclose(np.array(b1).reshape(-1, 3), np.array(b2).reshape(-1, 3), atol=1e-2)
        np.testing.assert_allclose(np.array(p1).reshape(-1, 5), np.array(p2).reshape(-1, 5), atol=1e-2)


def test_sparse_head_pads_like_the_convolution() -> None:
    rng = np.random.default_rng(2)
    c, h, w = 6, 5, 7
    feat = rng.standard_normal((c, h, w)).astype(np.float32)
    w0 = rng.standard_normal((4, c, 3, 3)).astype(np.float32)
    b0 = rng.standard_normal(4).astype(np.float32)
    w2 = rng.standard_normal((2, 4)).astype(np.float32)
    b2 = rng.standard_normal(2).astype(np.float32)
    head = SparseHead(w0.reshape(4, -1), b0, w2, b2)
    pad = np.pad(feat, ((0, 0), (1, 1), (1, 1)))
    rows, cols = np.array([0, 0, 4, 2, 4]), np.array([0, 6, 0, 3, 6])  # corners, an edge and the middle
    got = head.at(feat, rows, cols)
    for i, (r, cc) in enumerate(zip(rows, cols)):
        patch = pad[:, r:r + 3, cc:cc + 3]
        hid = np.maximum(np.tensordot(w0, patch, axes=([1, 2, 3], [0, 1, 2])) + b0, 0)
        np.testing.assert_allclose(got[:, i], w2 @ hid + b2, rtol=1e-5, atol=1e-5)
    assert head.at(feat, np.array([], int), np.array([], int)).shape == (2, 0)


def test_decode_sparse_matches_decode() -> None:
    rng = np.random.default_rng(3)
    h, w = 20, 30
    feat = rng.standard_normal((8, h, w)).astype(np.float32)
    heads = {k: SparseHead(rng.standard_normal((5, 72)).astype(np.float32), rng.standard_normal(5).astype(np.float32),
                           rng.standard_normal((2, 5)).astype(np.float32), rng.standard_normal(2).astype(np.float32))
             for k in ("ball_off", "person_wh", "person_off")}
    ball_hm = rng.random((h, w)).astype(np.float32) ** 4
    person_hm = rng.random((h, w)).astype(np.float32) ** 3
    rows, cols = np.divmod(np.arange(h * w), w)
    dense = {"ball_hm": ball_hm[None], "person_hm": person_hm[None]}
    for k, head in heads.items():
        dense[k] = head.at(feat, rows, cols).reshape(2, h, w)
    b1, p1 = decode_sparse(ball_hm, person_hm, feat, heads, 0.5, ball_thresh=0.2, person_thresh=0.3)
    b2, p2 = decode(dense, 0.5, ball_thresh=0.2, person_thresh=0.3)
    assert len(b1) == len(b2) > 0 and len(p1) == len(p2) > 0
    np.testing.assert_allclose(np.array(b1), np.array(b2), rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(np.array(p1), np.array(p2), rtol=1e-4, atol=1e-3)  # random weights: sizes up to 1e17


def test_onnx_weights_round_trip_into_build_net(exported) -> None:  # noqa: ANN001
    net, path, x = exported
    copy = build_net(3, pretrained=False)
    counts = load_onnx_weights(copy, path)
    assert counts["conv"] == 35 and counts["folded_bn"] == 21
    copy.eval()
    with torch.inference_mode():
        a = net(torch.from_numpy(x))
        b = copy(torch.from_numpy(x))
    for k in OUTPUTS:
        np.testing.assert_allclose(b[k].numpy(), a[k].numpy(), atol=2e-4, rtol=1e-4)


@pytest.mark.parametrize(("node", "path", "bn"), [
    ("/student/stem/stem.0/Conv", "stem.0", "stem.1"),
    ("/student/l1/l1.0/conv1/Conv", "l1.0.conv1", "l1.0.bn1"),
    ("/student/l3/l3.1/conv2/Conv", "l3.1.conv2", "l3.1.bn2"),
    ("/student/l2/l2.0/downsample/downsample.0/Conv", "l2.0.downsample.0", "l2.0.downsample.1"),
    ("/student/smooth/smooth.0/Conv", "smooth.0", "smooth.1"),
])
def test_module_paths_of_folded_convolutions(node: str, path: str, bn: str) -> None:
    assert module_path(node) == path
    assert _bn_after(path) == bn


def test_module_paths_of_biased_convolutions() -> None:
    assert module_path("/student/lat.2/Conv") == "lat.2"
    assert module_path("/student/person_wh/person_wh.2/Conv") == "person_wh.2"


def test_default_threads(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("COURTVISION_THREADS", "3")
    assert default_threads() == 3
    monkeypatch.delenv("COURTVISION_THREADS")
    assert 1 <= default_threads() <= 8


def test_student_detector_runs_an_onnx_export_like_torch(exported, tmp_path) -> None:  # noqa: ANN001
    from courtvision.perception.student import StudentConfig, StudentDetector

    net, path, _ = exported
    pt = tmp_path / "student.pt"
    torch.save({"model": net.state_dict()}, pt)
    rng = np.random.default_rng(4)
    frames = [rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(3)]
    onnx_det = StudentDetector(StudentConfig(weights=str(path), ball_thresh=0.2, person_thresh=0.2))
    torch_det = StudentDetector(StudentConfig(weights=str(pt), ball_thresh=0.2, person_thresh=0.2, backend="torch",
                                              device="cpu"))
    assert onnx_det.backend == "onnx" and torch_det.backend == "torch"
    for i, f in enumerate(frames):
        a, b = onnx_det.detect(f, i, i / 30), torch_det.detect(f, i, i / 30)
        assert len(a.detections) == len(b.detections)
        for da, dbx in zip(sorted(a.detections, key=lambda d: (d.cls.value, -d.score)),
                           sorted(b.detections, key=lambda d: (d.cls.value, -d.score))):
            assert da.cls == dbx.cls
            assert abs(da.score - dbx.score) < 1e-3
            assert abs(da.bbox.x1 - dbx.bbox.x1) < 0.05 and abs(da.bbox.y2 - dbx.bbox.y2) < 0.05
    with pytest.raises(ValueError, match="cannot run the ONNX export"):
        StudentDetector(StudentConfig(weights=str(path), backend="torch")).backend
