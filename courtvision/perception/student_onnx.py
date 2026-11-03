"""The student detector on ONNX Runtime (CPU), with the size and offset heads evaluated only where they are read.

``BallPersonNet`` ends in five heads, each a 3x3 convolution (128 -> 64 channels) and a 1x1 convolution, run at
stride 4 over the whole 136 x 240 output grid. The decoder (``student.decode``) reads the ball offset, the person
offset and the person box size **only at the heatmap peaks** it keeps (at most ``ball_k`` + ``person_k`` cells of
32,640), so three of the five heads are computed almost entirely for nothing: about a fifth of the network's
multiply-adds. ``OnnxStudent`` runs the network up to the shared stride-4 feature map plus the two heatmap heads, then
evaluates the other three heads at the peak cells in NumPy, from the same weights. The decoded detections are those
of the full network (the same arithmetic per cell, summed in a different order: differences of ~1e-6).

The model file is the browser/CPU export (``tools/export_student_browser.py``: input ``frames`` (1, 9, 544, 960),
outputs ``student.OUTPUTS``), fp32 or QDQ-quantized (``tools/quantize_student.py``, which leaves those three heads in
fp32). The split is found from the graph itself: the first convolution of each sparse head names the feature map it
reads. Needs ``onnxruntime`` and ``onnx`` (``pip install -e ".[onnx]"``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from courtvision.perception.student import OUTPUTS, STRIDE

__all__ = ["SPARSE_HEADS", "OnnxStudent", "SparseHead", "decode_sparse", "default_threads", "initializer_reader"]

DENSE_HEADS = ("ball_hm", "person_hm")
SPARSE_HEADS = ("ball_off", "person_wh", "person_off")


def default_threads() -> int:
    """Intra-op threads: ``COURTVISION_THREADS`` when set, else every core but one (at least 1, at most 8)."""
    env = os.environ.get("COURTVISION_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(8, (os.cpu_count() or 2) - 1))


@dataclass
class SparseHead:
    """One head's 3x3 convolution (``w0``: out x in*9, ``b0``) and 1x1 convolution (``w2``: n x out, ``b2``)."""

    w0: np.ndarray
    b0: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def at(self, feat: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """The head's (n, k) outputs at cells (``rows``, ``cols``) of the (C, H, W) feature map, zero padded as the
        convolution is."""
        k = len(rows)
        if k == 0:
            return np.zeros((len(self.b2), 0), np.float32)
        c, h, w = feat.shape
        d = np.arange(-1, 2)
        rr = rows[:, None, None] + d[None, :, None]  # (k, 3, 1)
        cc = cols[:, None, None] + d[None, None, :]  # (k, 1, 3)
        rr, cc = np.broadcast_arrays(rr, cc)
        inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        patch = feat[:, np.clip(rr, 0, h - 1), np.clip(cc, 0, w - 1)]  # (C, k, 3, 3)
        patch = np.where(inside[None], patch, 0.0).transpose(1, 0, 2, 3).reshape(k, c * 9)
        hid = np.maximum(patch @ self.w0.T + self.b0, 0.0)
        return (hid @ self.w2.T + self.b2).T.astype(np.float32)


def _peaks(hm: np.ndarray, k: int, thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``student._peaks`` as arrays: rows, cols, scores of the top-k 3x3 local maxima above ``thresh``."""
    pooled = cv2.dilate(hm, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero((hm >= pooled) & (hm >= thresh))
    order = np.argsort(-hm[ys, xs])[:k]
    return ys[order], xs[order], hm[ys[order], xs[order]]


def decode_sparse(ball_hm: np.ndarray, person_hm: np.ndarray, feat: np.ndarray, heads: dict[str, SparseHead],
                  scale: float, ball_k: int = 8, ball_thresh: float = 0.05, person_k: int = 60,
                  person_thresh: float = 0.25) -> tuple[list[tuple[float, float, float]],
                                                       list[tuple[float, float, float, float, float]]]:
    """``student.decode`` from the two heatmaps (sigmoid applied, (H, W)) and the feature map: the offset and size
    heads are evaluated at the kept peaks only. Same outputs as ``decode`` on the full network's outputs."""
    balls: list[tuple[float, float, float]] = []
    r, c, s = _peaks(ball_hm, ball_k, ball_thresh)
    if len(r):
        off = heads["ball_off"].at(feat, r, c)
        for i in range(len(r)):
            balls.append(((c[i] + off[0, i]) * STRIDE / scale, (r[i] + off[1, i]) * STRIDE / scale, float(s[i])))
    persons: list[tuple[float, float, float, float, float]] = []
    r, c, s = _peaks(person_hm, person_k, person_thresh)
    if len(r):
        off = heads["person_off"].at(feat, r, c)
        wh = heads["person_wh"].at(feat, r, c)
        for i in range(len(r)):
            cx, cy = (c[i] + off[0, i]) * STRIDE / scale, (r[i] + off[1, i]) * STRIDE / scale
            w, h = np.exp(wh[0, i]) * STRIDE / scale, np.exp(wh[1, i]) * STRIDE / scale
            persons.append((float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2), float(s[i])))
    return balls, persons


def initializer_reader(model: Any, where: str = "model") -> Any:
    """``name -> float32 array`` for a weight of ``model``: an initializer, or one reached through ``Identity``
    (the exporter shares identical tensors that way) or ``DequantizeLinear`` (an int8 QDQ weight)."""
    from onnx import numpy_helper

    inits = {i.name: i for i in model.graph.initializer}
    by_out = {o: n for n in model.graph.node for o in n.output}

    def weight(name: str) -> np.ndarray:
        for _ in range(16):
            if name in inits:
                return numpy_helper.to_array(inits[name]).astype(np.float32)
            node = by_out.get(name)
            if node is not None and node.op_type == "Identity":
                name = node.input[0]
                continue
            if node is not None and node.op_type == "DequantizeLinear":
                q, sc = weight_raw(node.input[0]), weight_raw(node.input[1])
                zp = weight_raw(node.input[2]) if len(node.input) > 2 and node.input[2] else 0
                axis = next((a.i for a in node.attribute if a.name == "axis"), 1)
                shape = [1] * q.ndim
                if np.ndim(sc):
                    shape[axis] = -1
                return ((q.astype(np.float32) - np.reshape(zp, shape)) * np.reshape(sc, shape)).astype(np.float32)
            break
        raise ValueError(f"{where}: weight {name!r} not found")

    def weight_raw(name: str) -> np.ndarray:
        if name not in inits:
            raise ValueError(f"{where}: quantized tensor {name!r} not found")
        return numpy_helper.to_array(inits[name])

    return weight


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class OnnxStudent:
    """The student network on ONNX Runtime. ``sparse`` (default): the trunk and both heatmap heads run in ONNX
    Runtime, the offset and size heads at the decoded peaks (``decode_sparse``); otherwise the whole exported graph
    runs and ``student.decode`` reads its outputs."""

    def __init__(self, path: str | Path, threads: int | None = None, sparse: bool = True) -> None:
        import onnx
        import onnxruntime as ort

        self.path = Path(path)
        model = onnx.load(str(self.path))
        self.sparse = sparse
        self.heads: dict[str, SparseHead] = {}
        self.feat_name: str | None = None
        if sparse:
            model = self._split(model)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or default_threads()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def _split(self, model: Any) -> Any:
        """Read the three sparse heads' weights and cut the graph after the feature map they share."""
        from onnx.utils import Extractor

        weight = initializer_reader(model, str(self.path))
        feats = set()
        for head in SPARSE_HEADS:
            conv0 = next((n for n in model.graph.node if n.op_type == "Conv" and n.name.endswith(f"/{head}.0/Conv")),
                         None)
            conv2 = next((n for n in model.graph.node if n.op_type == "Conv" and n.name.endswith(f"/{head}.2/Conv")),
                         None)
            if conv0 is None or conv2 is None:
                raise ValueError(f"{self.path}: head {head!r} not found (not a BallPersonNet export?)")
            w0, b0 = weight(conv0.input[1]), weight(conv0.input[2])
            w2, b2 = weight(conv2.input[1]), weight(conv2.input[2])
            self.heads[head] = SparseHead(w0.reshape(w0.shape[0], -1), b0, w2.reshape(w2.shape[0], -1), b2)
            feats.add(conv0.input[0])
        if len(feats) != 1:
            raise ValueError(f"{self.path}: the sparse heads read different feature maps {sorted(feats)}")
        self.feat_name = feats.pop()
        inits = {i.name for i in model.graph.initializer}
        graph_in = [i.name for i in model.graph.input if i.name not in inits]
        from onnx.shape_inference import infer_shapes

        return Extractor(infer_shapes(model)).extract_model(graph_in, [*DENSE_HEADS, self.feat_name])

    def run(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Raw outputs for a (1, 9, 544, 960) float32 input: logit heatmaps (and ``feat`` when sparse), batch dim
        dropped."""
        res = self.session.run(None, {self.input_name: np.ascontiguousarray(x, np.float32)})
        out = {("feat" if k == self.feat_name else k): v[0] for k, v in zip(self.output_names, res)}
        return out

    def dense(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """All five outputs over the whole grid (the sparse heads evaluated at every cell): for tests and tools."""
        out = self.run(x)
        if not self.sparse:
            return out
        feat = out.pop("feat")
        h, w = feat.shape[1:]
        rows, cols = np.divmod(np.arange(h * w), w)
        for k, head in self.heads.items():
            out[k] = head.at(feat, rows, cols).reshape(-1, h, w)
        return out

    def detect(self, x: np.ndarray, scale: float, *, ball_thresh: float, person_thresh: float, ball_k: int = 8,
               person_k: int = 60) -> tuple[list, list]:
        """Decoded ball points and person boxes for one input (``student.decode``'s outputs)."""
        out = self.run(x)
        ball_hm, person_hm = _sigmoid(out["ball_hm"][0]), _sigmoid(out["person_hm"][0])
        if self.sparse:
            return decode_sparse(ball_hm, person_hm, out["feat"], self.heads, scale, ball_k=ball_k,
                                 ball_thresh=ball_thresh, person_k=person_k, person_thresh=person_thresh)
        from courtvision.perception.student import decode

        one = {k: out[k] for k in OUTPUTS}
        one["ball_hm"], one["person_hm"] = ball_hm[None], person_hm[None]
        return decode(one, scale, ball_k=ball_k, ball_thresh=ball_thresh, person_k=person_k,
                      person_thresh=person_thresh)
