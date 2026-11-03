"""INT8 student detector: static post-training quantization of the browser/CPU ONNX export, checked against fp32.

The fp32 export (``courtvision/server/static/nba-student.onnx``, ``tools/export_student_browser.py``) spends nearly
all of its time in 35 convolutions. On CPUs with 8-bit dot-product instructions (x86 AVX512-VNNI / AVX-VNNI / AMX,
Arm i8mm) ONNX Runtime runs QDQ-quantized convolutions in integer arithmetic. This tool:

1. calibrates activation ranges on real broadcast inputs (three consecutive frames each, exactly as the detector
   sees them: ``preprocess_frames``), read from videos and/or image-sequence folders that are **not** the frames
   it is checked on;
2. writes a QDQ model: weights int8 per output channel, activations 8-bit per tensor; the five 1x1 output
   convolutions (heatmap logits, sub-cell offsets, box sizes) stay fp32 by default (``--fp32-heads``): they cost
   0.02% of the compute and carry the values the decoder thresholds and reads at sub-pixel precision;
3. checks it (run with the sparse heads of ``student_onnx``) against the fp32 network on held-out inputs: the
   decoded detections at the pipeline's thresholds (ball points within 15 px of 1080p, person boxes at IoU 0.5), the
   largest heatmap change, and the time per frame.

    .venv/bin/python tools/quantize_student.py --calib data/sportsmot/train --check data/bard/clips \\
        --out runs/models/student-int8.onnx --report runs/bench/int8.json

``--calib`` and ``--check`` take folders: every ``*.mp4``/``*.mov``/``*.mkv`` below them is a video, every folder
holding ``.jpg`` frames (``img1``) an image sequence. Rights: calibration frames do not end up in the model beyond
the 8-bit ranges of its activations, but research-only footage stays research-only (docs/TRAINING.md §1).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from courtvision.perception.student import (OUTPUTS, TRAIN_FPS, history_step, input_scale,  # noqa: E402
                                            preprocess_frames)
from courtvision.perception.student_onnx import SPARSE_HEADS, OnnxStudent  # noqa: E402

DEFAULT_SRC = ROOT / "courtvision" / "server" / "static" / "nba-student.onnx"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
HEAD_OUTPUT_CONVS = tuple(f"/student/{h}/{h}.2/Conv" for h in OUTPUTS)
# the offset and size heads run at the decoded peaks only (``student_onnx``): left in fp32, they are never quantized
SPARSE_HEAD_CONVS = tuple(f"/student/{h}/{h}.{i}/Conv" for h in SPARSE_HEADS for i in (0, 2))


# --------------------------------------------------------------------------- inputs


def _sources(folder: Path) -> list[Path]:
    """Videos and image-sequence folders under ``folder`` (symlinked folders followed)."""
    vids, seqs = [], []
    for d, _, files in os.walk(folder, followlinks=True):
        names = [f for f in files if not f.startswith("._")]
        vids += [Path(d) / f for f in names if Path(f).suffix.lower() in VIDEO_EXT]
        if any(f.endswith(".jpg") for f in names):
            seqs.append(Path(d))
    return sorted(vids) + sorted(seqs)


def _read_video(path: Path, n: int, rng: random.Random) -> Iterator[tuple[np.ndarray, tuple[int, int]]]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = history_step(cap.get(cv2.CAP_PROP_FPS) or TRAIN_FPS)
    if total < 2 * step + 2:
        cap.release()
        return
    starts = sorted(rng.sample(range(0, total - 2 * step - 1), min(n, total - 2 * step - 1)))
    for s in starts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        frames = []
        for k in range(2 * step + 1):
            ok, f = cap.read()
            if not ok:
                break
            if k % step == 0:
                frames.append(f)
        if len(frames) == 3:
            yield preprocess_frames(frames)[None], (frames[-1].shape[1], frames[-1].shape[0])
    cap.release()


def _read_sequence(folder: Path, n: int, rng: random.Random) -> Iterator[tuple[np.ndarray, tuple[int, int]]]:
    files = sorted(p for p in folder.glob("*.jpg") if not p.name.startswith("._"))
    if len(files) < 3:
        return
    for s in sorted(rng.sample(range(0, len(files) - 2), min(n, len(files) - 2))):
        frames = [cv2.imread(str(files[s + k])) for k in range(3)]
        yield preprocess_frames(frames)[None], (frames[-1].shape[1], frames[-1].shape[0])


def sample_inputs(folder: Path, total: int, seed: int) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """``total`` network inputs spread evenly over every video / image sequence under ``folder``."""
    rng = random.Random(seed)
    srcs = _sources(folder)
    if not srcs:
        raise SystemExit(f"no videos or .jpg sequences under {folder}")
    per = max(1, -(-total // len(srcs)))
    out: list[tuple[np.ndarray, tuple[int, int]]] = []
    for src in srcs:
        reader = _read_video if src.is_file() else _read_sequence
        out.extend(reader(src, per, rng))
    rng.shuffle(out)
    return out[:total]


# --------------------------------------------------------------------------- quantization


class _Reader:
    """Calibration inputs for onnxruntime, one range at a time (``CalibStridedMinMax``: the calibrator then holds the
    intermediate outputs of one input, not of all of them)."""

    def __init__(self, inputs: list[np.ndarray], name: str) -> None:
        self.inputs = inputs
        self.name = name
        self.set_range(0, len(inputs))

    def __len__(self) -> int:
        return len(self.inputs)

    def set_range(self, start_index: int, end_index: int) -> None:
        self.it = iter(self.inputs[start_index:end_index])

    def get_next(self) -> dict[str, np.ndarray] | None:
        x = next(self.it, None)
        return None if x is None else {self.name: x}

    def rewind(self) -> None:
        self.set_range(0, len(self.inputs))


def quantize(src: Path, out: Path, calib: list[np.ndarray], *, method: str, activation: str, fp32_heads: bool,
             percentile: float, fp32_nodes: tuple[str, ...] = ()) -> None:
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType, quant_pre_process,
                                          quantize_static)

    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "pre.onnx"
        quant_pre_process(str(src), str(pre), skip_symbolic_shape=False)
        methods = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
                   "percentile": CalibrationMethod.Percentile}
        # A calibrator keeps the intermediate outputs of every input it is given (~0.2 GB per 544x960 input), so it
        # gets one input at a time and merges ranges / histograms between them (``CalibStridedMinMax``; despite the
        # name every calibration method takes this path). ``CalibMaxIntermediateOutputs`` is no substitute: in
        # onnxruntime 1.31 its chunks are cleared without being merged.
        extra: dict[str, Any] = {"CalibStridedMinMax": 1}
        if method == "percentile":
            extra["CalibPercentile"] = percentile
        if method == "minmax":
            extra["CalibMovingAverage"] = True
        quantize_static(
            str(pre), str(out), _Reader(calib, "frames"),
            quant_format=QuantFormat.QDQ, per_channel=True,
            activation_type=QuantType.QUInt8 if activation == "u8" else QuantType.QInt8,
            weight_type=QuantType.QInt8, calibrate_method=methods[method],
            nodes_to_exclude=sorted(set(SPARSE_HEAD_CONVS) | set(fp32_nodes)
                                    | (set(HEAD_OUTPUT_CONVS) if fp32_heads else set())),
            extra_options=extra)


# --------------------------------------------------------------------------- checking


def runner(path: Path, threads: int, sparse: bool) -> OnnxStudent:
    return OnnxStudent(path, threads=threads, sparse=sparse)


def _logits(model: OnnxStudent, x: np.ndarray) -> dict[str, np.ndarray]:
    out = model.run(x)
    return {k: out[k] for k in ("ball_hm", "person_hm")}


def _decode(model: OnnxStudent, x: np.ndarray, size: tuple[int, int]) -> tuple[list, list]:
    return model.detect(x, input_scale(*size), ball_thresh=0.08, person_thresh=0.3)


def _iou(a: tuple, b: tuple) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def compare(ref: Any, cand: Any, inputs: list[tuple[np.ndarray, tuple[int, int]]]) -> dict[str, Any]:
    ball = [0, 0, 0]  # matched, only reference, only candidate
    person = [0, 0, 0]
    ball_err: list[float] = []
    hm_diff = {"ball_hm": 0.0, "person_hm": 0.0}
    for x, size in inputs:
        r, c = _logits(ref, x), _logits(cand, x)
        for k in hm_diff:
            sr, sc = 1 / (1 + np.exp(-r[k])), 1 / (1 + np.exp(-c[k]))
            hm_diff[k] = max(hm_diff[k], float(np.abs(sr - sc).max()))
        (rb, rp), (cb, cp) = _decode(ref, x, size), _decode(cand, x, size)
        px = 15.0 * size[1] / 1080.0
        used: set[int] = set()
        for b in rb:
            d = [(np.hypot(b[0] - q[0], b[1] - q[1]), j) for j, q in enumerate(cb) if j not in used]
            best = min(d, default=(np.inf, -1))
            if best[0] <= px:
                used.add(best[1])
                ball[0] += 1
                ball_err.append(best[0] * 1080.0 / size[1])
            else:
                ball[1] += 1
        ball[2] += len(cb) - len(used)
        used = set()
        for b in rp:
            cands = [(_iou(b, q), j) for j, q in enumerate(cp) if j not in used]
            best = max(cands, default=(0.0, -1))
            if best[0] >= 0.5:
                used.add(best[1])
                person[0] += 1
            else:
                person[1] += 1
        person[2] += len(cp) - len(used)

    def rate(m: list[int]) -> float | None:
        return m[0] / sum(m) if sum(m) else None

    return {"inputs": len(inputs), "ball": {"matched": ball[0], "only_fp32": ball[1], "only_int8": ball[2],
                                            "agreement": rate(ball),
                                            "median_px_1080": float(np.median(ball_err)) if ball_err else None},
            "person": {"matched": person[0], "only_fp32": person[1], "only_int8": person[2],
                       "agreement": rate(person)},
            "max_heatmap_change": hm_diff}


def speed(model: OnnxStudent, x: np.ndarray, size: tuple[int, int], n: int) -> float:
    """Median ms per frame of network + decoding (``OnnxStudent.detect``)."""
    for _ in range(3):
        _decode(model, x, size)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        _decode(model, x, size)
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", type=Path, default=DEFAULT_SRC, help="fp32 ONNX export")
    p.add_argument("--out", type=Path, required=True, help="INT8 ONNX to write")
    p.add_argument("--calib", type=Path, required=True, help="folder of videos / .jpg sequences for calibration")
    p.add_argument("--check", type=Path, required=True, help="folder of held-out videos / sequences for the check")
    p.add_argument("--n-calib", type=int, default=96)
    p.add_argument("--n-check", type=int, default=120)
    p.add_argument("--method", choices=("minmax", "entropy", "percentile"), default="minmax")
    p.add_argument("--percentile", type=float, default=99.999)
    p.add_argument("--activation", choices=("u8", "s8"), default="u8")
    p.add_argument("--fp32-heads", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fp32-nodes", default="",
                   help="comma list of further nodes left in fp32, e.g. /student/ball_hm/ball_hm.0/Conv")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", type=Path, default=None)
    a = p.parse_args(argv)

    t0 = time.time()
    calib = [x for x, _ in sample_inputs(a.calib, a.n_calib, a.seed)]
    print(f"{len(calib)} calibration inputs ({time.time() - t0:.0f} s)", file=sys.stderr)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fp32_nodes = tuple(n.strip() for n in a.fp32_nodes.split(",") if n.strip())
    quantize(a.src, a.out, calib, method=a.method, activation=a.activation, fp32_heads=a.fp32_heads,
             percentile=a.percentile, fp32_nodes=fp32_nodes)
    del calib  # 19 MB an input: sample the check inputs only once these are gone
    print(f"quantized in {time.time() - t0:.0f} s -> {a.out}", file=sys.stderr)
    check = sample_inputs(a.check, a.n_check, a.seed + 1)
    ref, cand = runner(a.src, a.threads, sparse=False), runner(a.out, a.threads, sparse=True)
    report = {"src": str(a.src), "out": str(a.out), "method": a.method, "activation": a.activation,
              "fp32_heads": a.fp32_heads, "fp32_nodes": list(fp32_nodes), "calib": str(a.calib), "check": str(a.check),
              "size_mb": {"fp32": a.src.stat().st_size / 1e6, "int8": a.out.stat().st_size / 1e6},
              "agreement": compare(ref, cand, check)}
    x, size = check[0]
    fp32_sparse = runner(a.src, a.threads, sparse=True)
    report["ms_per_frame"] = {"fp32": speed(ref, x, size, 10), "fp32_sparse": speed(fp32_sparse, x, size, 10),
                              "int8_sparse": speed(cand, x, size, 10), "threads": a.threads}
    print(json.dumps(report, indent=1))
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
