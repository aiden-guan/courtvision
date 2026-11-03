"""Recover a trainable ``BallPersonNet`` checkpoint from its ONNX export.

The browser/CPU export (``tools/export_student_browser.py``) folds every batch normalisation into the convolution
before it: a convolution ``W x`` followed by ``BN`` becomes ``W' x + b'``. The original running statistics are gone,
but the folded network is exactly representable by ``build_net`` with each such BN set to the identity plus a bias:
``running_mean = 0``, ``running_var = 1 - eps`` (so ``sqrt(var + eps) = 1``), ``weight = 1``, ``bias = b'``. In
eval mode the result computes what the ONNX graph computes (float rounding apart), and its state dict loads wherever
the original checkpoint did (``StudentDetector``, Core ML export, ``export_student_browser.py``).

For fine-tuning, keep those BN layers in eval mode (frozen statistics): in train mode they would renormalise by
batch statistics and undo the folding. ``freeze_bn`` does that.

    .venv/bin/python -m courtvision.training.student_from_onnx courtvision/server/static/nba-student.onnx \\
        --out runs/models/student-from-onnx.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["freeze_bn", "load_onnx_weights", "module_path"]


def module_path(node_name: str) -> str:
    """``/student/l2/l2.0/downsample/downsample.0/Conv`` -> ``l2.0.downsample.0`` (the exporter's scope names:
    a child of a container repeats its parent's name with ``.index``)."""
    segs = [s for s in node_name.strip("/").split("/")[:-1]]
    if segs and segs[0] == "student":
        segs = segs[1:]
    path: list[str] = []
    for s in segs:
        if path and s.startswith(path[-1] + "."):
            path[-1] = s
        else:
            path.append(s)
    return ".".join(path)


def _bn_after(conv_path: str) -> str:
    """The batch norm that follows a bias-free convolution in ``build_net``."""
    head, _, last = conv_path.rpartition(".")
    if last.startswith("conv"):
        return f"{head}.bn{last[4:]}"
    return f"{head}.{int(last) + 1}" if head else str(int(last) + 1)


def load_onnx_weights(net: Any, onnx_path: str | Path) -> dict[str, int]:
    """Copy the ONNX export's (folded) convolution weights into ``net`` (``build_net``); returns counts."""
    import onnx
    import torch

    from courtvision.perception.student_onnx import initializer_reader

    model = onnx.load(str(onnx_path))
    weight = initializer_reader(model, str(onnx_path))
    modules = dict(net.named_modules())
    counts = {"conv": 0, "folded_bn": 0}
    seen: set[str] = set()
    with torch.no_grad():
        for node in model.graph.node:
            if node.op_type != "Conv":
                continue
            path = module_path(node.name)
            conv = modules.get(path)
            if conv is None or not isinstance(conv, torch.nn.Conv2d):
                raise ValueError(f"ONNX conv {node.name} -> {path!r}: no such Conv2d in the network")
            w = torch.from_numpy(weight(node.input[1]))
            b = torch.from_numpy(weight(node.input[2])) if len(node.input) > 2 and node.input[2] else None
            if tuple(conv.weight.shape) != tuple(w.shape):
                raise ValueError(f"{path}: weight shape {tuple(w.shape)} != network {tuple(conv.weight.shape)}")
            conv.weight.copy_(w)
            counts["conv"] += 1
            seen.add(path)
            if conv.bias is not None:
                conv.bias.copy_(b if b is not None else torch.zeros_like(conv.bias))
            else:
                bn = modules[_bn_after(path)]
                if not isinstance(bn, torch.nn.BatchNorm2d):
                    raise ValueError(f"{path}: expected a BatchNorm2d after it, found {type(bn).__name__}")
                bn.weight.fill_(1.0)
                bn.bias.copy_(b if b is not None else torch.zeros_like(bn.bias))
                bn.running_mean.zero_()
                bn.running_var.fill_(1.0 - bn.eps)
                counts["folded_bn"] += 1
    missing = [n for n, m in modules.items() if isinstance(m, torch.nn.Conv2d) and n not in seen]
    if missing:
        raise ValueError(f"convolutions without ONNX weights: {missing}")
    return counts


def freeze_bn(net: Any) -> None:
    """Every BatchNorm2d to eval mode with its statistics frozen (call after ``net.train()``)."""
    import torch

    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", action=argparse.BooleanOptionalAction, default=True,
                    help="compare the network's outputs with ONNX Runtime on a random input")
    a = ap.parse_args(argv)
    import torch

    from courtvision.perception.student import INPUT_HW, OUTPUTS, build_net

    net = build_net(3, pretrained=False)
    counts = load_onnx_weights(net, a.onnx)
    net.eval()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "source_onnx": str(a.onnx)}, a.out)
    print(f"{counts['conv']} convolutions ({counts['folded_bn']} with a folded batch norm) -> {a.out}")
    if a.check:
        import onnxruntime as ort

        x = np.random.default_rng(0).standard_normal((1, 9, *INPUT_HW)).astype(np.float32)
        sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
        ref = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {sess.get_inputs()[0].name: x})))
        with torch.inference_mode():
            got = {k: v.numpy() for k, v in net(torch.from_numpy(x)).items()}
        diff = {k: float(np.abs(got[k] - ref[k]).max()) for k in OUTPUTS}
        print("max |torch - onnxruntime|:", {k: f"{v:.2e}" for k, v in diff.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
