"""Fine-tune the NBA student detector on more varied broadcasts, starting from its published ONNX export.

The student (``perception/student.py``) was distilled from an RF-DETR teacher on two NBA games. On human-labelled
clips of seven other broadcasts (SportsMOT basketball, ``tools/detector_bench.py``) it finds 61% of the players.
This tool adds varied footage without losing what the network already does well:

* **players** from human boxes: SportsMOT basketball sequences (``prep-mot``; players only, so the person head also
  learns to leave the crowd, the benches and the referees alone);
* **the ball** from the network's own linked detections on NBA footage (``prep-video``): its ball candidates on
  every frame are linked into one path per clip (``training.ball_labels.link_ball_path``, which also fills fast
  flights the detector misses) and kept as pseudo labels (weight at most 0.5, ``train_student.PSEUDO_MAX_WEIGHT``).
  This anchors the ball head while the shared trunk adapts (self-training with temporal consistency, after Noisy
  Student, Xie et al. 2020);
* training (``train``) from ``student_from_onnx``'s exact copy of the export, batch norms frozen (they hold the folded
  statistics), the stem and first stage frozen, AdamW with cosine decay, ``train_student``'s data loading,
  augmentation and losses. ``export`` writes the ONNX export (browser interface) and a ``.pt`` checkpoint.

Both data folders use ``train_student.ClipDataset``'s layout (``frames/<clip>/<frame>.jpg`` at 960 px wide,
``detections.jsonl``, ``ball_labels.jsonl``, ``meta.json``), so ``python -m courtvision.training.train_student`` can
read them too.

    .venv/bin/python tools/student_finetune.py prep-mot data/sportsmot/train data/ft/mot_train
    .venv/bin/python tools/student_finetune.py prep-video data/bard/split/calib data/ft/bard_calib
    .venv/bin/python tools/student_finetune.py train --data data/ft/mot_train --data data/ft/bard_calib \\
        --init runs/models/student-from-onnx.pt --out runs/models/student-ft --steps 3000
    .venv/bin/python tools/student_finetune.py export runs/models/student-ft/last.pt --out runs/models/student-ft.onnx

Research use only: SportsMOT is CC BY-NC 4.0, and NBA footage is research-only (docs/TRAINING.md §1).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from courtvision.perception.student import (INPUT_HW, OUTPUTS, TRAIN_FPS, history_step, input_scale,  # noqa: E402
                                            preprocess_frames)

SAVE_WIDTH = 960
DEFAULT_MODEL = ROOT / "courtvision" / "server" / "static" / "nba-student.onnx"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


def _save(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    small = cv2.resize(img, (SAVE_WIDTH, round(h * SAVE_WIDTH / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 92])


# --------------------------------------------------------------------------- prep-mot


def prep_mot(src: Path, out: Path, sample_every: int, names: set[str] | None) -> int:
    """SportsMOT sequences -> a ClipDataset folder: every frame stored (each sample needs its two predecessors),
    every ``sample_every``-th frame a sample with the human boxes, the others context rows."""
    seqs = sorted(d for d in src.iterdir() if (d / "gt" / "gt.txt").is_file() and (names is None or d.name in names))
    det_rows, clips = [], []
    for ci, seq in enumerate(seqs):
        files = sorted(p for p in (seq / "img1").glob("*.jpg") if not p.name.startswith("._"))
        gt: dict[int, list[list[float]]] = defaultdict(list)
        for r in np.loadtxt(seq / "gt" / "gt.txt", delimiter=",", ndmin=2):
            gt[int(r[0])].append([float(r[2]), float(r[3]), float(r[2] + r[4]), float(r[3] + r[5]), 1.0])
        size = None
        for fi, f in enumerate(files, start=1):
            img = cv2.imread(str(f))
            size = size or [img.shape[1], img.shape[0]]
            _save(img, out / "frames" / f"{ci:04d}" / f"{fi:06d}.jpg")
            row: dict[str, Any] = {"clip": ci, "frame": fi, "t": (fi - 1) / 25.0, "size": size,
                                   "persons": gt.get(fi, []), "balls": []}
            if fi < 3 or fi % sample_every or fi not in gt:
                row["context"] = True
            det_rows.append(row)
        clips.append({"clip": ci, "sequence": seq.name, "frames": len(files)})
        print(f"{seq.name}: {len(files)} frames", file=sys.stderr, flush=True)
    (out / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in det_rows))
    (out / "ball_labels.jsonl").write_text("")  # no ball labels: the ball loss skips these frames
    (out / "meta.json").write_text(json.dumps({
        # "video": an identity of its own, so train_student.plan_data never takes these frames for frames of another
        # folder's video (its guard against near-duplicates of held-out clips)
        "kind": "human", "video": f"sportsmot:{src.resolve()}", "source": f"SportsMOT basketball ({src})", "fps": 25.0,
        "rights_status": "research_only",
        "license": "CC BY-NC 4.0", "label_origin": "human", "persons": "players on the court only",
        "clips": clips}, indent=1))
    return len(seqs)


# --------------------------------------------------------------------------- prep-video


def _videos(folder: Path) -> list[Path]:
    out = []
    for d, _, files in os.walk(folder, followlinks=True):
        out += [Path(d) / f for f in files if Path(f).suffix.lower() in VIDEO_EXT and not f.startswith("._")]
    return sorted(out)


def prep_video(src: Path, out: Path, model_path: Path, sample_every: int, max_frames: int, threads: int) -> int:
    """Videos -> a pseudo ClipDataset folder with the model's own linked ball labels (no person labels)."""
    from courtvision.perception.student_onnx import OnnxStudent
    from courtvision.training.ball_labels import label_folder

    model = OnnxStudent(model_path, threads=threads)
    det_rows, clips = [], []
    vids = _videos(src)
    for ci, vid in enumerate(vids):
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS) or TRAIN_FPS
        step = history_step(fps)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        keep = min(max_frames, total // step)
        first = max(0, (total // step - keep) // 2) * step  # the middle of the clip
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        hist: list[np.ndarray] = []
        t0 = time.time()
        n = 0
        for k in range(keep * step):
            ok, img = cap.read()
            if not ok:
                break
            if k % step:
                continue
            n += 1
            fi = n
            hist = (hist + [preprocess_frames([img])])[-3:]
            x = np.concatenate([hist[max(0, len(hist) - 3 + j)] for j in range(3)])[None]
            balls, _ = model.detect(x, input_scale(img.shape[1], img.shape[0]), ball_thresh=0.05, person_thresh=0.99)
            r = 12.0 * img.shape[0] / 1080.0
            _save(img, out / "frames" / f"{ci:04d}" / f"{fi:06d}.jpg")
            row: dict[str, Any] = {"clip": ci, "frame": fi, "t": (first + k) / fps,
                                   "size": [img.shape[1], img.shape[0]], "persons": [],
                                   "balls": [[bx - r, by - r, bx + r, by + r, sc] for bx, by, sc in balls]}
            if fi < 3 or fi % sample_every:
                row["context"] = True
            det_rows.append(row)
        cap.release()
        clips.append({"clip": ci, "video": str(vid), "first_frame": first, "frames": n, "frame_step": step})
        print(f"[{ci + 1}/{len(vids)}] {vid.parent.name}/{vid.name}: {n} frames {time.time() - t0:.0f}s",
              file=sys.stderr, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in det_rows))
    (out / "meta.json").write_text(json.dumps({
        "kind": "pseudo", "video": f"videos:{src.resolve()}", "person_weight": 0.0,
        "source": f"{model_path.name} on {src}", "fps": TRAIN_FPS,
        "rights_status": "research_only", "label_origin": "pseudo", "clips": clips}, indent=1))
    rows = label_folder(out)
    (out / "ball_labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    labelled = sum(1 for r in rows if r["w"] > 0)
    print(f"{len(rows)} frames, {labelled} with a ball label", file=sys.stderr)
    return len(vids)


# --------------------------------------------------------------------------- prep-teacher


def prep_teacher(src: Path, out: Path, frames: int, weights: str | None, shape: str) -> int:
    """Videos -> a teacher folder labelled as ``tools/nba_teacher_labels.py`` labels a game: RF-DETR Small (COCO) on
    every stored frame (the middle ``frames`` of each video, at the student's input spacing) at ``shape``, persons
    from 0.25, every ball candidate, then the linked ball labels. For held-out ball checks (``detector_bench.py
    --ball-folder``) on footage the student never trained on."""
    import warnings

    import torch

    from courtvision.training.ball_labels import label_folder

    warnings.filterwarnings("ignore")
    import rfdetr

    threads = os.environ.get("OMP_NUM_THREADS", "")
    torch.set_num_threads(int(threads) if threads.isdigit() and int(threads) > 0 else max(1, os.cpu_count() or 2))
    model = rfdetr.RFDETRSmall(pretrain_weights=weights, device="cpu") if weights else rfdetr.RFDETRSmall(device="cpu")
    hw = tuple(int(v) for v in shape.split(","))
    det_rows, clips = [], []
    vids = _videos(src)
    t_all = time.time()
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "detections.partial.jsonl"  # one line per frame as it is labelled: an interrupted run resumes
    done = {(r["clip"], r["frame"]): r for r in map(json.loads, partial.open())} if partial.is_file() else {}
    sink = partial.open("a")
    for ci, vid in enumerate(vids):
        if any(c == ci for c, _ in done) and sum(1 for c, _ in done if c == ci) >= frames:
            det_rows += [r for (c, _), r in sorted(done.items()) if c == ci]
            clips.append({"clip": ci, "video": str(vid), "resumed": True})
            continue
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS) or TRAIN_FPS
        step = history_step(fps)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        keep = min(frames, total // step)
        first = max(0, (total // step - keep) // 2) * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        n = 0
        for k in range(keep * step):
            ok, img = cap.read()
            if not ok:
                break
            if k % step:
                continue
            n += 1
            ten = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)
            with torch.inference_mode():
                det = model.predict(ten, threshold=0.04, shape=hw, include_source_image=False)
            names = det.data.get("class_name")
            persons, balls = [], []
            for i in range(len(det.xyxy)):
                box = [round(float(v), 1) for v in det.xyxy[i]]
                sc = round(float(det.confidence[i]), 3)
                if names[i] == "person" and sc >= 0.25:
                    persons.append(box + [sc])
                elif names[i] == "sports ball":
                    balls.append(box + [sc])
            _save(img, out / "frames" / f"{ci:04d}" / f"{n:06d}.jpg")
            row = {"clip": ci, "frame": n, "t": round((first + k) / fps, 4), "size": [img.shape[1], img.shape[0]],
                   "persons": persons, "balls": balls}
            det_rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
        cap.release()
        clips.append({"clip": ci, "video": str(vid), "first_frame": first, "frames": n, "frame_step": step})
        el = time.time() - t_all
        print(f"[{ci + 1}/{len(vids)}] {vid.parent.name}/{vid.name}: {n} frames, "
              f"{el / max(1, len(det_rows)) * 1000:.0f} ms/frame", file=sys.stderr, flush=True)
    sink.close()
    (out / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in det_rows))
    (out / "meta.json").write_text(json.dumps({
        "kind": "teacher", "video": f"videos:{src.resolve()}", "teacher": f"RF-DETR Small (COCO) at {shape}",
        "source": str(src), "fps": TRAIN_FPS,
        "rights_status": "research_only", "label_origin": "cv_auto", "clips": clips}, indent=1))
    rows = label_folder(out)
    (out / "ball_labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} frames, {sum(1 for r in rows if r['w'] >= 0.5)} with a ball label of weight >= 0.5",
          file=sys.stderr)
    return len(vids)


# --------------------------------------------------------------------------- train


def _quick_mot_eval(net: Any, seqs: list[Any], frames_per_seq: int) -> dict[str, float]:
    """Person AP50 / recall at 0.3 on a few SportsMOT frames (``detector_bench``'s matching)."""
    import torch

    sys.path.insert(0, str(ROOT / "tools"))
    from detector_bench import average_precision, greedy_match  # noqa: E402

    from courtvision.perception.student import decode

    scores, tps, n_truth = [], [], 0
    net.eval()
    with torch.inference_mode():
        for seq in seqs:
            fr = [f for f in sorted(seq.gt) if f >= 3]
            for f in fr[:: max(1, len(fr) // frames_per_seq)][:frames_per_seq]:
                imgs = [cv2.imread(str(seq.frames[g - 1])) for g in (f - 2, f - 1, f)]
                x = torch.from_numpy(preprocess_frames(imgs)[None])
                out = {k: v[0].numpy() for k, v in net(x).items()}
                for k in ("ball_hm", "person_hm"):
                    out[k] = 1.0 / (1.0 + np.exp(-out[k]))
                _, persons = decode(out, input_scale(*seq.size), person_thresh=0.05, person_k=100)
                boxes = np.array([p[:4] for p in persons]).reshape(-1, 4)
                sc = np.array([p[4] for p in persons])
                tps += greedy_match(boxes, sc, seq.gt[f]).tolist()
                scores += sc.tolist()
                n_truth += len(seq.gt[f])
    s, t = np.array(scores), np.array(tps, bool)
    return {"ap50": round(average_precision(s, t, n_truth), 4),
            "recall@0.3": round(float((t & (s >= 0.3)).sum()) / max(1, n_truth), 4)}


def finetune_dataset(base: Any, hue_deg: float) -> Any:
    """``train_student.ClipDataset`` with domain randomisation for the person head: a sample without a ball label
    has its three frames' hues turned by one random angle in [-hue_deg, hue_deg] (jerseys, floors and adverts
    change colour from game to game; a frame with a ball label keeps its colours, the ball's orange being part
    of what the network knows a ball by)."""

    class FineTuneDataset(base):
        _hue = 0.0

        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            # a sample of a folder without person labels and with no ball label of its own trains nothing
            before = len(self.items)
            self.items = [it for it in self.items
                          if self.weights[it[0]].person > 0 or (it[4].get("xy") and it[4].get("w", 0) > 0)]
            self.dropped = before - len(self.items)

        def __getitem__(self, idx: int) -> Any:
            lab = self.items[idx][4]
            has_ball = bool(lab.get("xy")) and lab.get("w", 0) > 0
            self._hue = random.uniform(-hue_deg, hue_deg) if self.train and hue_deg > 0 and not has_ball else 0.0
            return super().__getitem__(idx)

        def _img(self, fi: int, clip: int, frame: int) -> np.ndarray:
            img = super()._img(fi, clip, frame)
            if self._hue and img is not None:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + round(self._hue / 2.0)) % 180).astype(np.uint8)
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return img

    return FineTuneDataset


def train(args: argparse.Namespace) -> int:
    import torch

    from courtvision.perception.student import build_net
    from courtvision.training.student_from_onnx import freeze_bn
    from courtvision.training.train_student import ClipDataset, _collate, focal_loss, masked_l1, plan_data

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    plan = plan_data(args.data)
    ds = finetune_dataset(ClipDataset, args.hue)(plan.folders, plan.train | plan.val if args.no_val else plan.train,
                                                 train=True, weights=plan.weights, crop=args.crop,
                                                 zoom_range=(args.zoom_min, args.zoom_max))
    for row in plan.summary():
        print(f"{row['kind']} {row['path']}: {row['train_clips']} train clips, {row['val_clips']} held out",
              flush=True)
    print(f"{len(ds)} training samples ({ds.dropped} without any label left out)", flush=True)
    net = build_net(3, pretrained=False)
    state = torch.load(args.init, map_location="cpu", weights_only=True)
    net.load_state_dict(state.get("model", state))
    frozen = [m for m in args.freeze.split(",") if m]
    for name in frozen:
        for p in getattr(net, name).parameters():
            p.requires_grad_(False)
    for m in net.modules():  # folded statistics: never updated, their affine part frozen too
        if isinstance(m, torch.nn.BatchNorm2d):
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(args.train_bn_bias)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                         drop_last=True, persistent_workers=args.workers > 0, collate_fn=_collate)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mot_seqs = []
    if args.eval_mot:
        sys.path.insert(0, str(ROOT / "tools"))
        from detector_bench import load_sequences  # noqa: E402

        names = {s.strip() for s in (Path(args.eval_mot).parent / "basketball.txt").read_text().split()} \
            if (Path(args.eval_mot).parent / "basketball.txt").is_file() else None
        mot_seqs = load_sequences(Path(args.eval_mot).parent, [Path(args.eval_mot).name], names)
    log = (out / "log.jsonl").open("a")
    if mot_seqs:
        ev = _quick_mot_eval(net, mot_seqs, args.eval_frames)
        print(f"step 0 eval {ev}", flush=True)
        log.write(json.dumps({"step": 0, "eval": ev}) + "\n")
    step, t0, run = 0, time.time(), defaultdict(float)
    best = -1.0
    while step < args.steps:
        for x, tg in loader:
            net.train()
            freeze_bn(net)
            o = net(x)
            l_ball = focal_loss(o["ball_hm"], tg["ball_hm"], tg["ball_w"][:, 0])
            l_boff = masked_l1(o["ball_off"], tg["ball_off"], tg["ball_mask"])
            l_phm = focal_loss(o["person_hm"], tg["person_hm"], tg["person_w"][:, 0])
            l_pwh = masked_l1(o["person_wh"], tg["person_wh"], tg["person_mask"])
            l_poff = masked_l1(o["person_off"], tg["person_off"], tg["person_mask"])
            loss = args.ball_weight * (2.0 * l_ball + l_boff) + l_phm + 0.5 * l_pwh + l_poff
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in (("loss", loss), ("ball", l_ball), ("person", l_phm), ("wh", l_pwh)):
                run[k] += float(v.detach())
            run["n"] += 1
            if step % args.log_every == 0:
                n = run.pop("n")
                msg = {k: round(v / n, 4) for k, v in run.items()}
                print(f"step {step}/{args.steps} {time.time() - t0:.0f}s lr {sched.get_last_lr()[0]:.2e} {msg}",
                      flush=True)
                run = defaultdict(float)
            if step % args.eval_every == 0 or step >= args.steps:
                ev = _quick_mot_eval(net, mot_seqs, args.eval_frames) if mot_seqs else {}
                print(f"step {step} eval {ev}", flush=True)
                log.write(json.dumps({"step": step, "eval": ev}) + "\n")
                log.flush()
                state = {"model": net.state_dict(), "step": step, "eval": ev, "init": str(args.init),
                         "data": args.data}
                torch.save(state, out / "last.pt")
                if ev and ev["ap50"] > best:
                    best = ev["ap50"]
                    torch.save(state, out / "best.pt")
            if step >= args.steps:
                break
    (out / "summary.json").write_text(json.dumps({"steps": step, "best_eval_ap50": best, "args": vars(args)},
                                                 indent=1, default=str))
    return 0


# --------------------------------------------------------------------------- export


def export(weights: Path, out: Path) -> int:
    """``.pt`` -> the browser/CPU ONNX export (input ``frames``, outputs ``OUTPUTS``; as
    ``tools/export_student_browser.py`` writes it), checked against torch on a random input."""
    import onnx
    import onnxruntime as ort
    import torch

    from courtvision.perception.student import build_net

    net = build_net(3, pretrained=False).eval()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    net.load_state_dict(state.get("model", state))

    class Export(torch.nn.Module):
        def forward(self, x: Any) -> tuple[Any, ...]:
            o = net(x)
            return tuple(o[name] for name in OUTPUTS)

    wrapper = Export()
    wrapper.add_module("student", net)
    wrapper.eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper, torch.zeros((1, 9, *INPUT_HW)), str(out), input_names=["frames"],
                      output_names=list(OUTPUTS), opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(out)))
    x = np.random.default_rng(0).standard_normal((1, 9, *INPUT_HW)).astype(np.float32)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = dict(zip(OUTPUTS, sess.run(None, {"frames": x})))
    with torch.inference_mode():
        ref = {k: v.numpy() for k, v in net(torch.from_numpy(x)).items()}
    print(f"{out}: max |onnx - torch| " + ", ".join(f"{k} {np.abs(got[k] - ref[k]).max():.1e}" for k in OUTPUTS))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep-mot", help="SportsMOT sequences -> a training folder")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--sample-every", type=int, default=3)
    p.add_argument("--seqs", default=None, help="comma list of sequence names (default: every labelled one)")
    p = sub.add_parser("prep-video", help="videos -> a pseudo-labelled training folder (ball only)")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--sample-every", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=120, help="frames per clip (at ~30 fps), from its middle")
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("prep-teacher", help="videos -> an RF-DETR teacher folder (held-out ball checks)")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--frames", type=int, default=48, help="frames per video (at ~30 fps), from its middle")
    p.add_argument("--weights", default=None, help="RF-DETR Small checkpoint (default: rfdetr's COCO download)")
    p.add_argument("--shape", default="736,1312", help="teacher input height,width (nba_teacher_labels default)")
    p = sub.add_parser("train", help="fine-tune")
    p.add_argument("--data", action="append", required=True, help="a training folder, DIR or DIR:WEIGHT")
    p.add_argument("--init", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--freeze", default="stem,l1", help="comma list of modules kept fixed")
    p.add_argument("--train-bn-bias", action="store_true", help="also train the folded batch-norm biases")
    p.add_argument("--ball-weight", type=float, default=1.0)
    p.add_argument("--hue", type=float, default=0.0, help="hue rotation (degrees) of samples without a ball label")
    p.add_argument("--zoom-min", type=float, default=0.8)
    p.add_argument("--zoom-max", type=float, default=1.25)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-val", action="store_true", help="train on the held-out clips too")
    p.add_argument("--eval-mot", default=None, help="SportsMOT split folder to score during training")
    p.add_argument("--eval-frames", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p = sub.add_parser("export", help=".pt -> ONNX")
    p.add_argument("weights", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "prep-mot":
        names = {s.strip() for s in a.seqs.split(",")} if a.seqs else None
        return 0 if prep_mot(a.src, a.out, a.sample_every, names) else 2
    if a.cmd == "prep-video":
        return 0 if prep_video(a.src, a.out, a.model, a.sample_every, a.max_frames, a.threads) else 2
    if a.cmd == "prep-teacher":
        return 0 if prep_teacher(a.src, a.out, a.frames, a.weights, a.shape) else 2
    if a.cmd == "train":
        return train(a)
    return export(a.weights, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
