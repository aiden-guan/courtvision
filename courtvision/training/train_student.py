"""Train the basketball student detector (``perception/student.py``) on teacher-labelled broadcast clips.

    .venv/bin/python -m courtvision.training.train_student --data runs/nba/<game>/teacher \
        [--data runs/nba/<other game>/teacher ...] [--data runs/nba/<game>/pseudo[:WEIGHT] ...] \
        --out runs/models/student [--epochs 14] [--resume]

Each ``--data`` folder holds ``frames/<clip>/<frame>.jpg`` (stored smaller than the video, e.g. 960 px wide),
``detections.jsonl`` (persons in full-resolution px, with the frame's ``size``) and ``ball_labels.jsonl``
(``ball_labels.link_ball_path``). A sample is a clip frame and its two predecessors; rows marked
``"context": true`` are only predecessors, never samples. Targets: Gaussian centre heatmaps at stride 4 (ball sigma
``ball_sigma`` cells; persons sized by CenterNet's radius rule), offsets and log box sizes. The ball loss is weighted
by the label weight and skipped on frames without a ball label (the teacher's silence is not evidence of no ball).

Folders are teacher folders (``tools/nba_teacher_labels.py``) unless their ``meta.json`` says ``"kind": "pseudo"``
(``pseudo_labels.py``: the student's own analysis of a game). ``DIR:WEIGHT`` scales a folder's label weights (default
1). A pseudo folder's ball labels never weigh more than ``PSEUDO_MAX_WEIGHT`` (the teacher's filled frames; its
detections weigh 0.75-1.0) and its persons train with the ``person_weight`` of its ``meta.json``. Every 10th clip of
each teacher folder is held out; pseudo folders are never held out or evaluated, so the score stays teacher-only.
A training clip within ``VAL_GUARD_S`` of a held-out clip of another folder of the same video is left out (two
folders without ``meta.json`` are split as before). ``best.pt`` keeps the epoch with the best held-out ball F1
(within 12 full-resolution px). ``--resume`` continues from ``last.pt`` in ``--out`` with the same ``--epochs`` and
data: the learning-rate schedule is replayed to the step it reached and the optimiser state restored (a ``last.pt``
saved without it restarts the optimiser moments).

Research use only: the labels derive from NBA broadcasts (docs/TRAINING.md §1).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from courtvision.perception.student import INPUT_HW, MEAN, STD, STRIDE, build_net, decode

TEACHER_SCALE = 0.5  # stored px per label px for rows without a ``size`` (960-px frames of 1920-px video)
OUT_HW = (INPUT_HW[0] // STRIDE, INPUT_HW[1] // STRIDE)
PSEUDO_MAX_WEIGHT = 0.5  # cap of a pseudo label's weight: the teacher's filled frames (its detections: 0.75-1.0)
VAL_GUARD_S = 2.0  # training clips this close to a held-out clip of the same video are left out


# --------------------------------------------------------------------------- data folders


def parse_data(spec: str) -> tuple[Path, float | None]:
    """``DIR`` or ``DIR:WEIGHT`` -> (folder, weight or None)."""
    head, sep, tail = spec.rpartition(":")
    if sep and head:
        try:
            return Path(head), float(tail)
        except ValueError:
            pass
    return Path(spec), None


def folder_meta(folder: Path) -> dict[str, Any]:
    """The folder's ``meta.json`` ({} when it has none: a teacher folder of the first format)."""
    try:
        meta = json.loads((Path(folder) / "meta.json").read_text())
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def is_pseudo(meta: dict[str, Any]) -> bool:
    return meta.get("kind") == "pseudo"


@dataclass(frozen=True)
class FolderWeights:
    ball: float  # scales the label weights
    ball_cap: float  # upper bound of a label's weight (pseudo folders)
    person: float  # weight of the person losses
    pseudo: bool


def folder_weights(meta: dict[str, Any], weight: float = 1.0) -> FolderWeights:
    if not is_pseudo(meta):
        return FolderWeights(weight, math.inf, weight, False)
    cap = min(float(meta.get("max_weight", PSEUDO_MAX_WEIGHT)), PSEUDO_MAX_WEIGHT)
    return FolderWeights(weight, cap, min(float(meta.get("person_weight", 0.0)) * weight, cap), True)


def _may_share_video(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether two folders can hold frames of one video: unknown counts as yes, except between two folders without
    ``meta.json`` (the split of such data stays as it was)."""
    if not a and not b:
        return False
    va, vb = a.get("video"), b.get("video")
    if va and vb:
        return os.path.realpath(str(va)) == os.path.realpath(str(vb))
    return True


@dataclass
class DataPlan:
    """The ``--data`` folders, their weights and ``meta.json``, and the split by clip ((folder, clip) pairs)."""

    folders: list[Path]
    weights: list[float]
    metas: list[dict[str, Any]]
    train: set[tuple[int, int]]
    val: set[tuple[int, int]]
    guarded: set[tuple[int, int]]  # left out: near a held-out clip of the same video

    def summary(self) -> list[dict[str, Any]]:
        def n(clips: set[tuple[int, int]], fi: int) -> int:
            return sum(1 for c in clips if c[0] == fi)

        return [{"path": str(f), "kind": "pseudo" if is_pseudo(self.metas[fi]) else "teacher",
                 "weight": self.weights[fi], "train_clips": n(self.train, fi), "val_clips": n(self.val, fi),
                 "guarded_clips": n(self.guarded, fi)} for fi, f in enumerate(self.folders)]


def plan_data(specs: Sequence[str], guard_s: float = VAL_GUARD_S) -> DataPlan:
    """Read the ``--data`` folders and split their clips: every 10th clip of each teacher folder is held out, pseudo
    folders train only, and training clips within ``guard_s`` of a held-out clip of another folder that may show
    the same video (``_may_share_video``) are left out."""
    folders: list[Path] = []
    weights: list[float] = []
    metas: list[dict[str, Any]] = []
    for spec in specs:
        path, w = parse_data(spec)
        meta = folder_meta(path)
        if is_pseudo(meta) and meta.get("complete") is False:
            raise ValueError(f"{path}: incomplete pseudo-label folder (run courtvision.training.pseudo_labels again)")
        folders.append(path)
        weights.append(1.0 if w is None else w)
        metas.append(meta)
    if all(is_pseudo(m) for m in metas):
        raise ValueError("--data has no teacher folder: the held-out score is measured on teacher clips only")
    spans: list[dict[int, tuple[int, int]]] = []  # per folder: clip -> (first, last frame)
    for folder in folders:
        sp: dict[int, tuple[int, int]] = {}
        for r in map(json.loads, (folder / "detections.jsonl").open()):
            a, b = sp.get(r["clip"], (r["frame"], r["frame"]))
            sp[r["clip"]] = (min(a, r["frame"]), max(b, r["frame"]))
        spans.append(sp)
    all_clips = {(fi, c) for fi, sp in enumerate(spans) for c in sp}
    val = {(fi, c) for fi, c in all_clips if not is_pseudo(metas[fi]) and c % 10 == 5}
    held: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for fi, c in val:
        held[fi].append(spans[fi][c])
    guarded: set[tuple[int, int]] = set()
    for fa, ca in all_clips - val:
        a0, a1 = spans[fa][ca]
        for fb, ranges in held.items():
            if fb == fa or not _may_share_video(metas[fa], metas[fb]):
                continue
            g = guard_s * float(metas[fa].get("fps") or metas[fb].get("fps") or 29.97)
            if any(a0 <= b1 + g and a1 >= b0 - g for b0, b1 in ranges):
                guarded.add((fa, ca))
                break
    return DataPlan(folders, weights, metas, all_clips - val - guarded, val, guarded)


def gaussian_radius(h: float, w: float, min_overlap: float = 0.7) -> float:
    """CenterNet's radius rule (``h``, ``w`` in output cells)."""
    a1, b1, c1 = 1, h + w, w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1**2 - 4 * a1 * c1)) / 2
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(b2**2 - 4 * a2 * c2)) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(b3**2 - 4 * a3 * c3)) / 2
    return max(0.0, min(r1, r2, r3))


def draw_gaussian(hm: np.ndarray, cx: float, cy: float, sigma: float) -> None:
    h, w = hm.shape
    r = int(math.ceil(3 * sigma))
    x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xs - int(cx)) ** 2 + (ys - int(cy)) ** 2) / (2 * sigma * sigma))
    np.maximum(hm[y0:y1, x0:x1], g, out=hm[y0:y1, x0:x1])


class ClipDataset:
    """Samples (input tensor, targets) from teacher and pseudo folders; ``train`` turns on augmentation;
    ``weights``: one per folder (``DIR:WEIGHT``, default 1)."""

    def __init__(self, folders: list[Path], clips: set[tuple[int, int]] | None, train: bool, frames: int = 3,
                 ball_sigma: float = 1.2, person_min_score: float = 0.4, crop: int = 512,
                 ball_crop_share: float = 0.7, weights: Sequence[float] | None = None) -> None:
        self.train, self.frames, self.ball_sigma, self.person_min = train, frames, ball_sigma, person_min_score
        self.crop, self.ball_crop_share = crop, ball_crop_share  # training crops (eval: the whole frame)
        self.items: list[tuple[int, int, list[int], dict, dict]] = []
        self.folders = folders
        self.weights = [folder_weights(folder_meta(f), 1.0 if weights is None else float(weights[fi]))
                        for fi, f in enumerate(folders)]
        for fi, folder in enumerate(folders):
            dets = {(r["clip"], r["frame"]): r for r in map(json.loads, (folder / "detections.jsonl").open())}
            labels = {(r["clip"], r["frame"]): r for r in map(json.loads, (folder / "ball_labels.jsonl").open())}
            by_clip: dict[int, list[int]] = defaultdict(list)
            for clip, frame in dets:
                by_clip[clip].append(frame)
            for clip, fr in by_clip.items():
                if clips is not None and (fi, clip) not in clips:
                    continue
                fr.sort()
                for i, f in enumerate(fr):
                    if dets[(clip, f)].get("context"):
                        continue  # only an earlier input frame of the next sample
                    prev = [fr[max(0, i - k)] for k in range(frames - 1, 0, -1)]
                    self.items.append((fi, clip, prev + [f], dets[(clip, f)], labels.get((clip, f), {})))

    def __len__(self) -> int:
        return len(self.items)

    def _img(self, fi: int, clip: int, frame: int) -> np.ndarray:
        return cv2.imread(str(self.folders[fi] / "frames" / f"{clip:04d}" / f"{frame:06d}.jpg"))

    def __getitem__(self, idx: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        fi, clip, frames, det, lab = self.items[idx]
        imgs = [self._img(fi, clip, f) for f in frames]
        h_in, w_in = INPUT_HW
        h0, w0 = imgs[0].shape[:2]
        s = w0 / float(det["size"][0]) if det.get("size") else TEACHER_SCALE  # label px -> stored image px
        flip = self.train and random.random() < 0.5
        zoom = random.uniform(0.8, 1.25) if self.train else 1.0
        ox = oy = 0.0
        if self.train:  # a crop x crop window of the zoomed frame, often around the ball
            h_in = w_in = self.crop
            if lab.get("xy") and lab.get("w", 0) > 0 and random.random() < self.ball_crop_share:
                bx, by = lab["xy"][0] * s * zoom, lab["xy"][1] * s * zoom
                cx, cy = bx + random.uniform(-0.4, 0.4) * w_in, by + random.uniform(-0.4, 0.4) * h_in
            else:
                cx, cy = random.uniform(0, w0 * zoom), random.uniform(0, h0 * zoom)
            ox, oy = w_in / 2 - cx, h_in / 2 - cy
        M = np.array([[zoom, 0.0, ox], [0.0, zoom, oy]], np.float32)
        if flip:
            M = np.array([[-1.0, 0.0, w_in - 1.0], [0.0, 1.0, 0.0]], np.float32) @ np.vstack([M, [0, 0, 1]])
        jitter = None
        if self.train:
            jitter = (random.uniform(0.85, 1.15), random.uniform(-12, 12), random.uniform(0.85, 1.15))
        chans = []
        for img in imgs:
            warped = cv2.warpAffine(img, M, (w_in, h_in), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            if jitter is not None:
                c, b, sat = jitter
                warped = np.clip(warped.astype(np.float32) * c + b, 0, 255).astype(np.uint8)
                hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
                warped = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            chans.append(((rgb - MEAN) / STD).transpose(2, 0, 1))
        x = np.concatenate(chans, axis=0).astype(np.float32)

        def tf(px: float, py: float) -> tuple[float, float]:
            return (M[0, 0] * px * s + M[0, 1] * py * s + M[0, 2], M[1, 0] * px * s + M[1, 1] * py * s + M[1, 2])

        oh, ow = h_in // STRIDE, w_in // STRIDE
        fw = self.weights[fi]
        t = {"ball_hm": np.zeros((1, oh, ow), np.float32), "ball_off": np.zeros((2, oh, ow), np.float32),
             "ball_mask": np.zeros((1, oh, ow), np.float32), "ball_w": np.zeros(1, np.float32),
             "person_hm": np.zeros((1, oh, ow), np.float32), "person_wh": np.zeros((2, oh, ow), np.float32),
             "person_off": np.zeros((2, oh, ow), np.float32), "person_mask": np.zeros((1, oh, ow), np.float32),
             "person_w": np.array([fw.person], np.float32), "scale": np.array([s], np.float32)}
        if lab.get("w", 0) > 0 and lab.get("xy"):
            u, v = tf(*lab["xy"])
            cu, cv = u / STRIDE, v / STRIDE
            bw = min(float(lab["w"]) * fw.ball, fw.ball_cap)
            t["ball_w"][0] = bw  # the frame counts for the ball loss (even with the ball cropped out)
            if 0 <= cu < ow and 0 <= cv < oh:
                draw_gaussian(t["ball_hm"][0], cu, cv, self.ball_sigma)
                t["ball_off"][:, int(cv), int(cu)] = (cu - int(cu), cv - int(cv))
                t["ball_mask"][0, int(cv), int(cu)] = bw if fw.pseudo else fw.ball
        for x1, y1, x2, y2, sc in det["persons"] if fw.person > 0 else ():
            if sc < self.person_min:
                continue
            a, b = tf(x1, y1)
            c, d = tf(x2, y2)
            bx1, bx2 = sorted((a, c))
            by1, by2 = sorted((b, d))
            bx1, by1, bx2, by2 = max(0, bx1), max(0, by1), min(w_in - 1, bx2), min(h_in - 1, by2)
            if bx2 - bx1 < 4 or by2 - by1 < 8:
                continue
            cu, cv = (bx1 + bx2) / 2 / STRIDE, (by1 + by2) / 2 / STRIDE
            wc, hc = (bx2 - bx1) / STRIDE, (by2 - by1) / STRIDE
            draw_gaussian(t["person_hm"][0], cu, cv, max(0.6, (2 * gaussian_radius(hc, wc) + 1) / 6))
            t["person_wh"][:, int(cv), int(cu)] = (math.log(wc), math.log(hc))
            t["person_off"][:, int(cv), int(cu)] = (cu - int(cu), cv - int(cv))
            t["person_mask"][0, int(cv), int(cu)] = fw.person
        return x, t


def focal_loss(logits, target, weight=None):  # noqa: ANN001, ANN201 (torch)
    """CenterNet focal loss (alpha 2, beta 4), summed over the batch and divided by the positive count; ``weight``
    (B,) scales each sample."""
    import torch

    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1.0 - pos
    pos_l = torch.log(p) * (1 - p) ** 2 * pos
    neg_l = torch.log(1 - p) * p**2 * (1 - target) ** 4 * neg
    per = -(pos_l + neg_l).flatten(1).sum(1)
    if weight is not None:
        per = per * weight
    n = pos.flatten(1).sum(1)
    if weight is not None:
        n = n * (weight > 0).float()
    return per.sum() / n.sum().clamp(min=1.0)


def masked_l1(pred, target, mask):  # noqa: ANN001, ANN201 (torch)
    m = mask.expand_as(pred)
    return (abs(pred - target) * m).sum() / m.sum().clamp(min=1.0)


def evaluate(net, ds: ClipDataset, device: str, tol_px: float = 12.0) -> dict[str, float]:  # noqa: ANN001
    """Held-out ball precision / recall / F1 at several thresholds (labelled frames; within ``tol_px``
    full-resolution px) and person recall at IoU 0.5."""
    import torch

    net.eval()
    found: list[tuple[float, bool]] = []  # top ball score per labelled frame, and whether it hit the label
    n_lab = 0
    p_hit = p_n = 0
    with torch.no_grad():
        for i in range(0, len(ds), 16):
            batch = [ds[j] for j in range(i, min(len(ds), i + 16))]
            x = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
            out = {k: v.float().cpu().numpy() for k, v in net(x).items()}
            for j, (_, tg) in enumerate(batch):
                one = {k: v[j] for k, v in out.items()}
                for k in ("ball_hm", "person_hm"):
                    one[k] = 1.0 / (1.0 + np.exp(-one[k]))
                balls, persons = decode(one, float(tg["scale"][0]), ball_thresh=0.02, person_thresh=0.3)
                _, item_clip, frames, det, lab = ds.items[i + j]
                if lab.get("w", 0) >= 0.5 and lab.get("xy"):
                    n_lab += 1
                    if balls:
                        bx, by, sc = balls[0]
                        found.append((sc, math.hypot(bx - lab["xy"][0], by - lab["xy"][1]) <= tol_px))
                gts = [p for p in det["persons"] if p[4] >= 0.5]
                p_n += len(gts)
                for g in gts:
                    if any(_iou(g, q) >= 0.5 for q in persons):
                        p_hit += 1
    res: dict[str, float] = {"person_recall": p_hit / max(1, p_n), "n_ball": n_lab}
    best_f1 = 0.0
    for th in (0.1, 0.2, 0.3, 0.4, 0.5):
        tp = sum(1 for sc, ok in found if sc >= th and ok)
        fp = sum(1 for sc, ok in found if sc >= th and not ok)
        prec, rec = tp / max(1, tp + fp), tp / max(1, n_lab)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        res[f"ball_p@{th}"], res[f"ball_r@{th}"] = round(prec, 3), round(rec, 3)
        best_f1 = max(best_f1, f1)
    res["ball_f1"] = round(best_f1, 4)
    net.train()
    return res


def _iou(a, b) -> float:  # noqa: ANN001
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", required=True, help="a teacher or pseudo folder, DIR or DIR:WEIGHT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init", default=None, help="start from these weights")
    ap.add_argument("--resume", action="store_true", help="continue from last.pt in --out (same --epochs)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smoke test)")
    args = ap.parse_args()
    try:
        plan = plan_data(args.data)
    except ValueError as e:
        ap.error(str(e))

    import torch

    random.seed(0)
    torch.manual_seed(0)
    folders = plan.folders
    train_ds = ClipDataset(folders, plan.train, train=True, weights=plan.weights)
    val_ds = ClipDataset(folders, plan.val, train=False, weights=plan.weights)
    for row in plan.summary():
        print(f"{row['kind']} {row['path']} (weight {row['weight']}): {row['train_clips']} train clips, "
              f"{row['val_clips']} held out, {row['guarded_clips']} left out near held-out clips", flush=True)
    print(f"train {len(train_ds)} frames, val {len(val_ds)} frames", flush=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = build_net(3, pretrained=args.init is None).to(device)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True)["model"])
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                         drop_last=True, persistent_workers=args.workers > 0, collate_fn=_collate)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.08)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    step = 0
    start_epoch = 0
    if args.resume:
        start_epoch, step, best = _resume(out, net, opt, sched, len(loader))
        print(f"resumed after epoch {start_epoch}/{args.epochs} (best ball F1 {best})", flush=True)
    for epoch in range(start_epoch, args.epochs):
        t0, run = time.time(), defaultdict(float)
        for x, tg in loader:
            x = x.to(device)
            tg = {k: v.to(device) for k, v in tg.items()}
            o = net(x)
            l_ball = focal_loss(o["ball_hm"], tg["ball_hm"], tg["ball_w"][:, 0])
            l_boff = masked_l1(o["ball_off"], tg["ball_off"], tg["ball_mask"])
            l_phm = focal_loss(o["person_hm"], tg["person_hm"], tg["person_w"][:, 0])
            l_pwh = masked_l1(o["person_wh"], tg["person_wh"], tg["person_mask"])
            l_poff = masked_l1(o["person_off"], tg["person_off"], tg["person_mask"])
            loss = 2.0 * l_ball + l_boff + l_phm + 0.5 * l_pwh + l_poff
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            for k, v in (("loss", loss), ("ball", l_ball), ("person", l_phm), ("wh", l_pwh)):
                run[k] += float(v)
            run["n"] += 1
            if args.max_steps and step >= args.max_steps:
                break
        n = max(1.0, run.pop("n"))
        msg = {k: round(v / n, 4) for k, v in run.items()}
        ev = evaluate(net, val_ds, device)
        print(f"epoch {epoch + 1}/{args.epochs} {time.time() - t0:.0f}s train {msg} val {ev}", flush=True)
        state = {"model": net.state_dict(), "epoch": epoch + 1, "val": ev}
        torch.save({**state, "opt": opt.state_dict(), "step": step, "epochs": args.epochs}, out / "last.pt")
        if ev["ball_f1"] > best:
            best = ev["ball_f1"]
            torch.save(state, out / "best.pt")
        if args.max_steps and step >= args.max_steps:
            break
    (out / "train_summary.json").write_text(json.dumps({"best_ball_f1": best, "epochs": args.epochs,
                                                        "data": args.data, "folders": plan.summary()}, indent=1))
    return 0


def _resume(out: Path, net, opt, sched, steps_per_epoch: int) -> tuple[int, int, float]:  # noqa: ANN001
    """Load ``last.pt`` (weights, and the optimiser state when saved) and replay the schedule to its step ->
    (epochs done, steps done, best F1). The schedule is replayed, not loaded: older torch versions pickle
    OneCycleLR's state with a bound method, which ``weights_only`` loading refuses."""
    import warnings

    import torch

    last = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(last["model"])
    epoch = int(last["epoch"])
    step = int(last.get("step", epoch * steps_per_epoch))
    if "opt" in last:
        opt.load_state_dict(last["opt"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "lr_scheduler.step() before optimizer.step()"
        for _ in range(min(step, sched.total_steps - 1)):
            sched.step()
    best = -1.0
    if (out / "best.pt").is_file():
        best = float(torch.load(out / "best.pt", map_location="cpu", weights_only=True)["val"]["ball_f1"])
    return epoch, step, best


def _collate(batch):  # noqa: ANN001, ANN202 (torch)
    import torch

    x = torch.from_numpy(np.stack([b[0] for b in batch]))
    keys = batch[0][1].keys()
    return x, {k: torch.from_numpy(np.stack([b[1][k] for b in batch])) for k in keys}


if __name__ == "__main__":
    raise SystemExit(main())
