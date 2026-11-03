"""Time the student detector's CPU paths on real frames: one detection (network + decoding), median of N.

    .venv/bin/python tools/student_speed.py --frames data/bard/split/check --threads 4 \\
        --onnx courtvision/server/static/nba-student.onnx --int8 runs/models/student-int8.onnx \\
        [--pt runs/models/student-from-onnx.pt] [--json runs/bench/speed.json]

Paths: ``torch`` (``StudentDetector``'s CPU fallback, from a ``.pt`` checkpoint), ``onnx-dense`` (the whole exported
graph on ONNX Runtime, then ``student.decode``), ``onnx-sparse`` (``student_onnx``: the offset and size heads only
at the decoded peaks) and ``int8-sparse`` (the same on the int8 export). Every path decodes the same inputs; the
first three give the same detections, so their times compare like for like. Run it on an otherwise idle machine.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from courtvision.perception.student import input_scale  # noqa: E402


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _time(fn: Any, inputs: list[tuple[np.ndarray, tuple[int, int]]], repeats: int) -> dict[str, float]:
    for x, size in inputs[:3]:
        fn(x, size)
    times = []
    for i in range(repeats):
        x, size = inputs[i % len(inputs)]
        t0 = time.perf_counter()
        fn(x, size)
        times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(times)), "p10_ms": float(np.percentile(times, 10)),
            "p90_ms": float(np.percentile(times, 90)), "n": repeats}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", type=Path, required=True, help="folder of videos / .jpg sequences")
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--int8", type=Path, default=None)
    ap.add_argument("--pt", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--inputs", type=int, default=12)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)

    sys.path.insert(0, str(ROOT / "tools"))
    from quantize_student import sample_inputs  # noqa: E402

    from courtvision.perception.student import decode
    from courtvision.perception.student_onnx import OnnxStudent

    inputs = sample_inputs(a.frames, a.inputs, 7)
    res: dict[str, Any] = {"threads": a.threads, "cpu": _cpu_name(),
                           "inputs": len(inputs), "paths": {}}

    def thr(model: OnnxStudent):  # noqa: ANN202
        return lambda x, size: model.detect(x, input_scale(*size), ball_thresh=0.08, person_thresh=0.3)

    dense = OnnxStudent(a.onnx, threads=a.threads, sparse=False)
    res["paths"]["onnx-dense"] = _time(thr(dense), inputs, a.repeats)
    res["paths"]["onnx-sparse"] = _time(thr(OnnxStudent(a.onnx, threads=a.threads, sparse=True)), inputs, a.repeats)
    if a.int8:
        res["paths"]["int8-sparse"] = _time(thr(OnnxStudent(a.int8, threads=a.threads, sparse=True)), inputs,
                                            a.repeats)
    if a.pt:
        import torch

        from courtvision.perception.student import build_net

        torch.set_num_threads(a.threads)
        net = build_net(3, pretrained=False).eval()
        state = torch.load(a.pt, map_location="cpu", weights_only=True)
        net.load_state_dict(state.get("model", state))

        def run_torch(x: np.ndarray, size: tuple[int, int]) -> Any:
            with torch.inference_mode():
                out = {k: v[0].numpy() for k, v in net(torch.from_numpy(x)).items()}
            for k in ("ball_hm", "person_hm"):
                out[k] = 1.0 / (1.0 + np.exp(-out[k]))
            return decode(out, input_scale(*size), ball_thresh=0.08, person_thresh=0.3)

        res["paths"]["torch"] = _time(run_torch, inputs, a.repeats)
    print(f"{res['cpu']}, {a.threads} threads, {len(inputs)} inputs, {a.repeats} timed detections each\n")
    print("| path | median ms | p10 | p90 |")
    print("|---|---:|---:|---:|")
    for k, v in res["paths"].items():
        print(f"| {k} | {v['median_ms']:.1f} | {v['p10_ms']:.1f} | {v['p90_ms']:.1f} |")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
