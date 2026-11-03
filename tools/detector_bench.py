"""Cross-game benchmark of the NBA student detector on human-labelled broadcast footage (SportsMOT basketball).

The student's own held-out numbers (``train_student.evaluate``) come from teacher labels on the two games it was
trained on. This tool scores it on games it has never seen, against human annotation: the basketball sequences of
SportsMOT (Cui et al., ICCV 2023; 720p NBA broadcast clips, every player on the court boxed in every frame; CC BY-NC
4.0, research use only, see docs/TRAINING.md §1). SportsMOT boxes players only, not referees, coaches or the crowd, so
**player recall** is the clean number; precision counts every unlabelled person the detector finds as a false positive
and is reported for comparisons between models, not as an absolute.

Data layout (what ``data/sportsmot`` holds after extraction): ``<split>/<sequence>/img1/000001.jpg`` and
``<split>/<sequence>/gt/gt.txt`` (MOT format: frame, id, x, y, w, h, conf, class, visibility). Only the basketball
sequences (``splits_txt/basketball.txt``) are used; ``train`` and ``val`` have labels.

For every scored frame t the network sees frames t-2, t-1, t, as in the pipeline (SportsMOT is 25 fps, so its input
frames are 40 ms apart instead of the 33 ms of training). Matching is greedy by score at IoU >= 0.5 (PASCAL VOC).
Reported per model, pooled over the scored frames of every sequence:

* AP50 (all-point interpolated, detections decoded down to score 0.05);
* recall, precision and F1 at the pipeline's operating threshold (``StudentConfig.person_thresh``, 0.3);
* the same after the pipeline's referee filter (``nba.analyze.is_referee``), which only removes boxes;
* inference ms per frame (network and decoding) and, with two or more models, their agreement with the first model:
  ball points within 15 px (of 1080p) and person boxes at IoU 0.5, both at the operating thresholds.

    .venv/bin/python tools/detector_bench.py --models courtvision/server/static/nba-student.onnx --stride 10
    .venv/bin/python tools/detector_bench.py --models a.onnx,b.onnx --split val --stride 5 --json runs/bench/det.json

Models are ONNX files with the browser export's interface (input ``frames`` (1, 9, 544, 960), outputs
``OUTPUTS``) or PyTorch checkpoints (``.pt``) of ``BallPersonNet``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from courtvision.perception.student import INPUT_HW, OUTPUTS, decode, input_scale, preprocess_frames  # noqa: E402

IOU_MATCH = 0.5
DECODE_FLOOR = 0.05  # lowest person score kept for AP
BALL_AGREE_PX_1080 = 15.0


# --------------------------------------------------------------------------- data


@dataclass
class Sequence:
    name: str
    split: str
    frames: list[Path]  # index i is frame i + 1
    gt: dict[int, np.ndarray]  # frame (1-based) -> (n, 4) x1, y1, x2, y2
    size: tuple[int, int]  # width, height


def load_sequences(root: Path, splits: list[str], names: set[str] | None) -> list[Sequence]:
    out = []
    for split in splits:
        for d in sorted((root / split).glob("*")):
            if not d.is_dir() or (names is not None and d.name not in names):
                continue
            gt_file = d / "gt" / "gt.txt"
            if not gt_file.is_file():
                continue
            frames = sorted(p for p in (d / "img1").glob("*.jpg") if not p.name.startswith("._"))
            if not frames:
                continue
            rows = np.loadtxt(gt_file, delimiter=",", ndmin=2)
            gt: dict[int, list[list[float]]] = {}
            for r in rows:
                fr, x, y, w, h = int(r[0]), r[2], r[3], r[4], r[5]
                gt.setdefault(fr, []).append([x, y, x + w, y + h])
            im = cv2.imread(str(frames[0]))
            out.append(Sequence(d.name, split, frames, {k: np.asarray(v, np.float64) for k, v in gt.items()},
                                (im.shape[1], im.shape[0])))
    return out


# --------------------------------------------------------------------------- models


class OnnxModel:
    def __init__(self, path: Path, threads: int) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.names = [o.name for o in self.sess.get_outputs()]

    def __call__(self, x: np.ndarray) -> dict[str, np.ndarray]:
        res = self.sess.run(None, {self.input: x})
        return {k: v[0] for k, v in zip(self.names, res)}


class TorchModel:
    def __init__(self, path: Path, threads: int) -> None:
        import torch

        from courtvision.perception.student import build_net

        torch.set_num_threads(threads)
        self.torch = torch
        self.net = build_net(3, pretrained=False)
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state.get("model", state))
        self.net.eval()

    def __call__(self, x: np.ndarray) -> dict[str, np.ndarray]:
        with self.torch.inference_mode():
            out = self.net(self.torch.from_numpy(x))
        return {k: v[0].float().numpy() for k, v in out.items()}


class SparseOnnxModel:
    """``student_onnx.OnnxStudent``: the offset and size heads only at the decoded peaks (same detections)."""

    sparse = True

    def __init__(self, path: Path, threads: int) -> None:
        from courtvision.perception.student_onnx import OnnxStudent

        self.model = OnnxStudent(path, threads=threads, sparse=True)

    def detect(self, x: np.ndarray, scale: float, ball_thresh: float, person_floor: float) -> tuple[list, list]:
        return self.model.detect(x, scale, ball_thresh=ball_thresh, person_thresh=person_floor, person_k=100)


def load_model(path: Path, threads: int, sparse: bool = False) -> Any:
    if path.suffix in (".pt", ".pth"):
        return TorchModel(path, threads)
    return SparseOnnxModel(path, threads) if sparse else OnnxModel(path, threads)


def run_model(model: Any, x: np.ndarray, size: tuple[int, int], ball_thresh: float, person_floor: float
              ) -> tuple[list, list]:
    """Decoded (balls, persons) of one input in frame px."""
    if getattr(model, "sparse", False):
        return model.detect(x, input_scale(*size), ball_thresh, person_floor)
    return detections(model(x), size, ball_thresh, person_floor)


def detections(raw: dict[str, np.ndarray], size: tuple[int, int], ball_thresh: float, person_floor: float
               ) -> tuple[list, list]:
    one = dict(raw)
    for k in ("ball_hm", "person_hm"):
        one[k] = 1.0 / (1.0 + np.exp(-raw[k]))
    return decode(one, input_scale(*size), ball_thresh=ball_thresh, person_thresh=person_floor, person_k=100)


# --------------------------------------------------------------------------- metrics


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def greedy_match(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, thr: float = IOU_MATCH) -> np.ndarray:
    """True positive flag per prediction (highest score first claims a truth box)."""
    tp = np.zeros(len(pred), bool)
    if len(pred) == 0 or len(gt) == 0:
        return tp
    iou = iou_matrix(pred, gt)
    used = np.zeros(len(gt), bool)
    for i in np.argsort(-scores, kind="stable"):
        cand = np.where(~used & (iou[i] >= thr))[0]
        if len(cand):
            j = cand[np.argmax(iou[i, cand])]
            used[j] = True
            tp[i] = True
    return tp


def average_precision(scores: np.ndarray, tp: np.ndarray, n_truth: int) -> float:
    if n_truth == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tps = np.cumsum(tp[order])
    fps = np.cumsum(~tp[order])
    rec = tps / n_truth
    prec = tps / np.maximum(tps + fps, 1)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@dataclass
class Tally:
    scores: list[float] = field(default_factory=list)
    tp: list[bool] = field(default_factory=list)
    n_truth: int = 0
    ref_kept: list[bool] = field(default_factory=list)  # survives the referee filter
    ms: list[float] = field(default_factory=list)
    agree_ball: list[int] = field(default_factory=lambda: [0, 0, 0])  # matched, only-ref, only-this
    agree_person: list[int] = field(default_factory=lambda: [0, 0, 0])

    def summary(self, thresh: float) -> dict[str, Any]:
        s, tp, kept = np.asarray(self.scores), np.asarray(self.tp, bool), np.asarray(self.ref_kept, bool)
        at = s >= thresh
        out: dict[str, Any] = {"n_truth": self.n_truth, "ap50": average_precision(s, tp, self.n_truth)}
        for tag, sel in (("", at), ("_noref", at & kept)):
            n_tp, n_pred = int((tp & sel).sum()), int(sel.sum())
            r = n_tp / self.n_truth if self.n_truth else 0.0
            p = n_tp / n_pred if n_pred else 0.0
            out.update({f"recall{tag}": r, f"precision{tag}": p, f"f1{tag}": 2 * p * r / (p + r) if p + r else 0.0,
                        f"n_pred{tag}": n_pred})
        out["ms_per_frame"] = float(np.median(self.ms)) if self.ms else None
        for k, (m, a, b) in (("ball_agreement", self.agree_ball), ("person_agreement", self.agree_person)):
            out[k] = {"matched": m, "only_reference": a, "only_this": b,
                      "rate": m / (m + a + b) if (m + a + b) else None}
        return out


def _ball_agreement(ref: list, mine: list, px: float) -> tuple[int, int, int]:
    ref_xy = np.array([b[:2] for b in ref]).reshape(-1, 2)
    my_xy = np.array([b[:2] for b in mine]).reshape(-1, 2)
    if len(ref_xy) == 0 or len(my_xy) == 0:
        return 0, len(ref_xy), len(my_xy)
    d = np.linalg.norm(ref_xy[:, None] - my_xy[None], axis=2)
    used, m = set(), 0
    for i in np.argsort(d.min(axis=1)):
        j = int(np.argmin(np.where([k in used for k in range(len(my_xy))], np.inf, d[i])))
        if d[i, j] <= px and j not in used:
            used.add(j)
            m += 1
    return m, len(ref_xy) - m, len(my_xy) - m


# --------------------------------------------------------------------------- run


def run(seqs: list[Sequence], models: dict[str, Any], stride: int, thresh: float, ball_thresh: float,
        max_frames: int | None, quiet: bool, zoom: float = 1.0) -> tuple[dict[str, dict[str, Tally]], dict[str, Tally]]:
    from courtvision.nba.analyze import is_referee

    per_seq: dict[str, dict[str, Tally]] = {}
    pooled = {m: Tally() for m in models}
    names = list(models)
    for si, seq in enumerate(seqs):
        tallies = {m: Tally() for m in models}
        frames_scored = [f for f in range(3, len(seq.frames) + 1, stride) if f in seq.gt][:max_frames]
        cache: dict[int, np.ndarray] = {}
        x = np.empty((1, 9, *INPUT_HW), np.float32)
        t_seq = time.perf_counter()
        for f in frames_scored:
            imgs = {}
            for k, fr in enumerate((f - 2, f - 1, f)):
                if fr not in cache:
                    img = cv2.imread(str(seq.frames[fr - 1]))
                    imgs[fr] = img
                    if zoom != 1.0:  # the frame shrunk inside the network input (padded): a scale probe
                        small = cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA)
                        canvas = np.zeros((round(img.shape[0] * 960 / img.shape[1]), 960, 3), np.uint8)
                        sm = cv2.resize(small, None, fx=960 / img.shape[1], fy=960 / img.shape[1],
                                        interpolation=cv2.INTER_AREA)
                        canvas[:sm.shape[0], :sm.shape[1]] = sm
                        img = canvas
                    cache[fr] = preprocess_frames([img])
                x[0, 3 * k:3 * k + 3] = cache[fr]
            cur = imgs.get(f)
            if cur is None:
                cur = cv2.imread(str(seq.frames[f - 1]))
            for fr in [k for k in cache if k < f - 2]:
                del cache[fr]
            gt = seq.gt[f]
            outs = {}
            for m in names:
                t0 = time.perf_counter()
                if zoom != 1.0:
                    b, ps = run_model(models[m], x, (960, round(seq.size[1] * 960 / seq.size[0])), ball_thresh,
                                      DECODE_FLOOR)
                    k = seq.size[0] / 960 / zoom
                    outs[m] = ([(u * k, v * k, sc) for u, v, sc in b],
                               [(a * k, b_ * k, c * k, d * k, sc) for a, b_, c, d, sc in ps])
                else:
                    outs[m] = run_model(models[m], x, seq.size, ball_thresh, DECODE_FLOOR)
                tallies[m].ms.append((time.perf_counter() - t0) * 1000)
            for m in names:
                balls, persons = outs[m]
                boxes = np.array([p[:4] for p in persons]).reshape(-1, 4)
                sc = np.array([p[4] for p in persons])
                tp = greedy_match(boxes, sc, gt)
                tl = tallies[m]
                tl.scores += sc.tolist()
                tl.tp += tp.tolist()
                tl.n_truth += len(gt)
                tl.ref_kept += [not is_referee(cur, tuple(b)) for b in boxes]
                if m != names[0]:
                    ref_b, ref_p = outs[names[0]]
                    px = BALL_AGREE_PX_1080 * seq.size[1] / 1080.0
                    a = _ball_agreement(ref_b, balls, px)
                    rb = np.array([p[:4] for p in ref_p if p[4] >= thresh]).reshape(-1, 4)
                    rs = np.array([p[4] for p in ref_p if p[4] >= thresh])
                    mb = boxes[sc >= thresh]
                    pm = greedy_match(mb, sc[sc >= thresh], rb)
                    tl.agree_ball = [u + v for u, v in zip(tl.agree_ball, a)]
                    tl.agree_person = [u + v for u, v in zip(tl.agree_person, (int(pm.sum()), len(rb) - int(pm.sum()),
                                                                               len(mb) - int(pm.sum())))]
                    del rs
        per_seq[seq.name] = tallies
        for m in names:
            p, t = pooled[m], tallies[m]
            p.scores += t.scores
            p.tp += t.tp
            p.n_truth += t.n_truth
            p.ref_kept += t.ref_kept
            p.ms += t.ms
            p.agree_ball = [u + v for u, v in zip(p.agree_ball, t.agree_ball)]
            p.agree_person = [u + v for u, v in zip(p.agree_person, t.agree_person)]
        if not quiet:
            msg = " ".join(f"{m}: R {tallies[m].summary(thresh)['recall']:.3f}" for m in names)
            print(f"[{si + 1}/{len(seqs)}] {seq.split}/{seq.name} {len(frames_scored)} frames "
                  f"{time.perf_counter() - t_seq:.0f}s  {msg}", file=sys.stderr, flush=True)
    return per_seq, pooled


def ball_bench(folder: Path, models: dict[str, Any], ball_thresh: float, tol_1080: float, min_weight: float,
               quiet: bool) -> dict[str, dict[str, Any]]:
    """Each model's top ball on every labelled frame of a teacher folder (``ball_labels.jsonl`` weight >=
    ``min_weight``: the teacher's linked path, ``training.ball_labels``): within ``tol_1080`` px of 1080p of the label
    (hit), elsewhere (wrong) or nothing above ``ball_thresh`` (miss). The network sees the frame and the two stored
    frames before it, as in training (``train_student.ClipDataset``)."""
    rows = {(r["clip"], r["frame"]): r for r in map(json.loads, (folder / "detections.jsonl").open())}
    labels = {(r["clip"], r["frame"]): r for r in map(json.loads, (folder / "ball_labels.jsonl").open())}
    by_clip: dict[int, list[int]] = {}
    for c, f in rows:
        by_clip.setdefault(c, []).append(f)
    out = {m: {"hit": 0, "wrong": 0, "miss": 0, "err_px_1080": []} for m in models}
    x = np.empty((1, 9, *INPUT_HW), np.float32)
    for clip, frames in sorted(by_clip.items()):
        frames.sort()
        cache: dict[int, np.ndarray] = {}
        for i, f in enumerate(frames):
            lab = labels.get((clip, f))
            if not lab or lab.get("xy") is None or lab.get("w", 0) < min_weight or i < 2:
                continue
            for k, g in enumerate(frames[i - 2:i + 1]):
                if g not in cache:
                    cache[g] = preprocess_frames([cv2.imread(str(folder / "frames" / f"{clip:04d}" / f"{g:06d}.jpg"))])
                x[0, 3 * k:3 * k + 3] = cache[g]
            size = tuple(rows[(clip, f)]["size"])
            tol = tol_1080 * size[1] / 1080.0
            for m, model in models.items():
                balls, _ = run_model(model, x, size, ball_thresh, 0.99)
                o = out[m]
                if not balls:
                    o["miss"] += 1
                    continue
                bx, by, _ = balls[0]
                d = float(np.hypot(bx - lab["xy"][0], by - lab["xy"][1]))
                o["hit" if d <= tol else "wrong"] += 1
                if d <= tol:
                    o["err_px_1080"].append(d * 1080.0 / size[1])
        if not quiet:
            msg = " ".join(f"{Path(m).name}: {o['hit']}/{o['hit'] + o['wrong'] + o['miss']}" for m, o in out.items())
            print(f"clip {clip}: {msg}", file=sys.stderr, flush=True)
    for o in out.values():
        n = o["hit"] + o["wrong"] + o["miss"]
        o.update({"n": n, "hit_rate": o["hit"] / n if n else None, "wrong_rate": o["wrong"] / n if n else None,
                  "miss_rate": o["miss"] / n if n else None,
                  "median_err_px_1080": float(np.median(o["err_px_1080"])) if o["err_px_1080"] else None})
        del o["err_px_1080"]
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, default=ROOT / "data" / "sportsmot")
    p.add_argument("--split", default="val,train", help="comma list of SportsMOT splits with labels")
    p.add_argument("--models", required=True, help="comma list of .onnx / .pt files; the first is the reference")
    p.add_argument("--seqs", default=None, help="comma list of sequence names (default: every basketball sequence)")
    p.add_argument("--exclude", default=None, help="file with sequence names to leave out (e.g. used in training)")
    p.add_argument("--stride", type=int, default=10, help="score every Nth frame (default 10)")
    p.add_argument("--max-frames", type=int, default=None, help="at most this many scored frames per sequence")
    p.add_argument("--thresh", type=float, default=0.3, help="person operating threshold (StudentConfig: 0.3)")
    p.add_argument("--ball-thresh", type=float, default=0.08, help="ball threshold (StudentConfig: 0.08)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--sparse", action="store_true",
                   help="run ONNX models with student_onnx's sparse heads (same detections, less compute)")
    p.add_argument("--zoom", type=float, default=1.0, help="probe: shrink each frame by this inside the network input")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--ball-folder", type=Path, default=None,
                   help="score the ball instead: a teacher folder (frames, detections.jsonl, ball_labels.jsonl)")
    p.add_argument("--ball-tol", type=float, default=15.0, help="ball hit radius in px of a 1080-line frame")
    p.add_argument("--ball-min-weight", type=float, default=0.5)
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)

    if a.ball_folder is not None:
        paths = [Path(m.strip()) for m in a.models.split(",") if m.strip()]
        models = {str(pth): load_model(pth, a.threads, a.sparse) for pth in paths}
        res = ball_bench(a.ball_folder, models, a.ball_thresh, a.ball_tol, a.ball_min_weight, a.quiet)
        print(f"\nball within {a.ball_tol:g} px of 1080p, labelled frames of {a.ball_folder} (weight >= "
              f"{a.ball_min_weight}), threshold {a.ball_thresh}\n")
        print("| model | frames | hit | wrong | missed | median error px (1080p) |")
        print("|---|---|---|---|---|---|")
        for m, o in res.items():
            med = o["median_err_px_1080"]
            print(f"| {Path(m).name} | {o['n']} | {o['hit_rate']:.4f} | {o['wrong_rate']:.4f} | {o['miss_rate']:.4f} | "
                  f"{'—' if med is None else round(med, 2)} |")
        if a.json:
            a.json.parent.mkdir(parents=True, exist_ok=True)
            a.json.write_text(json.dumps({"ball_folder": str(a.ball_folder), "tol_px_1080": a.ball_tol,
                                          "threshold": a.ball_thresh, "models": res}, indent=1))
        return 0

    names: set[str] | None = None
    basket = a.data / "basketball.txt"
    if a.seqs:
        names = {s.strip() for s in a.seqs.split(",") if s.strip()}
    elif basket.is_file():
        names = {s.strip() for s in basket.read_text().split() if s.strip()}
    if a.exclude:
        names = (names or set()) - {s.strip() for s in Path(a.exclude).read_text().split() if s.strip()}
    seqs = load_sequences(a.data, [s.strip() for s in a.split.split(",") if s.strip()], names)
    if not seqs:
        print(f"no labelled sequences under {a.data}", file=sys.stderr)
        return 2
    paths = [Path(m.strip()) for m in a.models.split(",") if m.strip()]
    models = {str(pth): load_model(pth, a.threads, a.sparse) for pth in paths}
    per_seq, pooled = run(seqs, models, a.stride, a.thresh, a.ball_thresh, a.max_frames, a.quiet, a.zoom)

    report = {
        "data": str(a.data), "splits": a.split, "stride": a.stride, "thresh": a.thresh,
        "sequences": [s.name for s in seqs], "frames_scored": len(pooled[str(paths[0])].ms),
        "pooled": {m: t.summary(a.thresh) for m, t in pooled.items()},
        "per_sequence": {s: {m: t.summary(a.thresh) for m, t in ts.items()} for s, ts in per_seq.items()},
    }
    print(f"\n{len(seqs)} sequences, {report['frames_scored']} scored frames, IoU {IOU_MATCH}, "
          f"operating threshold {a.thresh}\n")
    print("| model | AP50 | recall | precision | F1 | recall, refs out | precision, refs out | ms/frame |"
          " ball agree | person agree |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m, s in report["pooled"].items():
        ba, pa = s["ball_agreement"]["rate"], s["person_agreement"]["rate"]
        print(f"| {Path(m).name} | {s['ap50']:.4f} | {s['recall']:.4f} | {s['precision']:.4f} | {s['f1']:.4f} | "
              f"{s['recall_noref']:.4f} | {s['precision_noref']:.4f} | {s['ms_per_frame']:.1f} | "
              f"{'—' if ba is None else f'{ba:.4f}'} | {'—' if pa is None else f'{pa:.4f}'} |")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
