"""Export the existing NBA checkpoint, without training, and verify ONNX parity on NBA footage.

Optional export dependencies: pip install onnx onnxruntime (plus the desktop torch/torchvision).
Run from the repository root: python tools/export_student_browser.py --video <NBA clip>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch

from courtvision.perception.student import INPUT_HW, OUTPUTS, build_net, decode, input_scale, preprocess_frames


def write_browser_reference(session, output: Path, model_hash: str) -> None:
    """A deterministic, non-video CPU reference to verify every visitor's inference backend."""
    indices = np.arange(9 * 544 * 960, dtype=np.uint32)
    x = (((indices * 13 + (indices >> 10) * 7) % 251).astype(np.float32) - 125) / 128
    outputs = session.run(None, {"frames": x.reshape(1, 9, *INPUT_HW)})
    heads = {}
    for name, value in zip(OUTPUTS, outputs, strict=True):
        flat = value.ravel()
        probes = [j * 9973 % len(flat) for j in range(128)]
        heads[name] = {"indices": probes, "values": [float(flat[i]) for i in probes]}
    reference = {"onnx_sha256": model_hash, "input": "((i*13+(i>>>10)*7)%251-125)/128",
                 "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(), "tolerance": .003, "outputs": heads}
    output.with_name("nba-validation.json").write_text(json.dumps(reference, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("runs/studio/models/nba-student.pt"))
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("courtvision/server/static/nba-student.onnx"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    net = build_net(3, pretrained=False).eval()
    net.load_state_dict(checkpoint.get("model", checkpoint))

    class Export(torch.nn.Module):
        def forward(self, x):
            out = net(x)
            return tuple(out[name] for name in OUTPUTS)

    wrapper = Export()
    wrapper.add_module("student", net)
    wrapper.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper, torch.zeros((1, 9, *INPUT_HW)), str(args.output),
                      input_names=["frames"], output_names=list(OUTPUTS), opset_version=17,
                      dynamo=False)
    onnx.checker.check_model(onnx.load(args.output))
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode {args.video}")
    checks = []
    for at in (0., 3., 6.):
        cap.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
        frames = []
        for _ in range(3):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("The validation clip must have at least seven seconds")
            frames.append(frame)
        x = preprocess_frames(frames)[None]
        with torch.no_grad():
            expected = net(torch.from_numpy(x))
        actual = dict(zip(OUTPUTS, session.run(None, {"frames": x})))
        errors = {name: float(np.max(np.abs(expected[name].numpy() - actual[name]))) for name in OUTPUTS}
        if max(errors.values()) > .003:
            raise RuntimeError(f"ONNX output differs from the checkpoint: {errors}")
        decoded = []
        for outputs in ({k: v.numpy()[0] for k, v in expected.items()}, {k: v[0] for k, v in actual.items()}):
            for name in ("ball_hm", "person_hm"):
                outputs[name] = 1 / (1 + np.exp(-outputs[name]))
            decoded.append(decode(outputs, input_scale(frame.shape[1], frame.shape[0]), ball_thresh=.08, person_thresh=.3))
        for left, right in zip(*decoded):
            if not np.allclose(left, right, atol=.05, rtol=1e-4):
                raise RuntimeError("Decoded detections differ from the checkpoint")
        checks.append({"t": at, "max_absolute_error": errors, "balls": len(decoded[0][0]), "people": len(decoded[0][1])})
    cap.release()
    manifest = {"architecture": "BallPersonNet", "training": "existing NBA student checkpoint; no new training",
                "checkpoint_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                "onnx_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "epoch": checkpoint.get("epoch"), "input": [1, 9, *INPUT_HW], "outputs": list(OUTPUTS),
                "temporal_spacing_s": 1 / 29.97, "ball_threshold": .08, "person_threshold": .3,
                "validation_clip": args.video.name, "parity": checks}
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_browser_reference(session, args.output, manifest["onnx_sha256"])
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
