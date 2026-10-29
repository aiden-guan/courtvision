"""Ball labels from noisy per-frame detections: link one ball path through a clip, fill short flight gaps.

A COCO detector run frame by frame on broadcast basketball finds the ball when it is large and still (held,
dribbled) and misses it in flight; it also fires on round things in the crowd. ``link_ball_path`` turns such
candidates into one consistent path per clip (the single game ball):

* each candidate scores ``score + w_color * color - bias`` (``color``: how close its centre colour is to a
  basketball's, ``ball_color_likelihood``);
* a dynamic programme over the clip's frames picks the best chain: consecutive picks at most ``max_gap`` frames
  apart and at most ``max_step_px`` per frame apart, paying for skipped frames and for speed above
  ``free_step_px``; a chain may also restart (``restart_cost``) when the ball leaves the view;
* frames between two picks of a fast chain (``flight_px`` per frame or more: a pass or a shot, where the
  detector misses the blurred ball) are filled by linear interpolation and labelled with a lower weight; slow
  gaps (the ball held behind a body) are left unlabelled.

``BallLabel.weight`` is 0 for "no label" frames (the ball loss ignores them), else the label's confidence.
Thresholds are design choices checked on ESPN frames of Lakers @ Warriors (2024-12-25): frames at 29.97 fps and px
of a 1080-line frame. ``link_ball_path`` applies them at a clip's own rate (``fps``: the video's rate over the
clip's frame step) and frame height (``LinkConfig.for_clip``); ``label_folder`` reads both from the folder.

``ball_labels.jsonl`` of a teacher folder (``tools/nba_teacher_labels.py``), one row per frame of each clip:

    .venv/bin/python -m courtvision.training.ball_labels runs/nba/<game>/teacher-hard [--force]

The candidates are the teacher's "sports ball" boxes (centre, half the longer side, score); their colour is read
from the stored frames (``label_folder``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from courtvision.nba.analyze import REF_FPS, REF_HEIGHT, rate_ratio, scale_frames

__all__ = ["BallLabel", "LinkConfig", "ball_color_likelihood", "label_folder", "link_ball_path"]

# median Lab (OpenCV 8-bit) of the centre of confident COCO "sports ball" detections on ESPN NBA frames
BALL_LAB = (95.0, 149.0, 150.0)
BALL_AB_SIGMA = 7.0


def ball_color_likelihood(img_bgr: np.ndarray, center: tuple[float, float], radius: float) -> float:
    """In [0, 1]: how close the median colour of the disc's centre is to a basketball's (Lab a / b)."""
    h, w = img_bgr.shape[:2]
    r = max(1, int(round(radius * 0.5)))
    cx, cy = int(round(center[0])), int(round(center[1]))
    patch = img_bgr[max(0, cy - r):min(h, cy + r + 1), max(0, cx - r):min(w, cx + r + 1)]
    if patch.size == 0:
        return 0.0
    lab = np.median(cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float), axis=0)
    d2 = ((lab[1] - BALL_LAB[1]) ** 2 + (lab[2] - BALL_LAB[2]) ** 2) / BALL_AB_SIGMA**2
    dark = 1.0 if 40.0 <= lab[0] <= 175.0 else 0.3
    return float(math.exp(-0.5 * d2) * dark)


@dataclass
class LinkConfig:
    """Settings in the units of the tuning footage: frames at 29.97 fps (``REF_FPS``), px of a 1080-line frame."""

    min_score: float = 0.05
    max_candidates: int = 10
    color_weight: float = 0.6
    bias: float = 0.3  # a candidate with no colour support needs score > bias to help a chain
    max_gap: int = 6  # frames
    max_step_px: float = 90.0  # per frame, full-resolution px (ball speed in the image + camera pan)
    free_step_px: float = 25.0
    speed_cost: float = 0.004  # per px/frame above free_step_px
    gap_cost: float = 0.12  # per skipped frame
    restart_cost: float = 0.6
    flight_px: float = 14.0  # mean speed of a gap's ends at or above which the gap is filled
    fill_weight: float = 0.5
    min_pick_weight: float = 0.25  # picks below this unary are kept in the chain but not labelled

    def for_clip(self, fps: float, height: float) -> LinkConfig:
        """These settings in the frames and px of a clip at ``fps`` (its frames per second) with ``height``-line
        frames (``self`` itself at 29.97 fps and 1080 lines), on the rule of ``analyze.BallLinkConfig.for_video``:
        the same speeds and gap times, a restart r times as dear with r times the frames a second."""
        r, s = rate_ratio(fps), height / REF_HEIGHT
        if r == 1.0 and s == 1.0:
            return self
        return replace(self, max_gap=scale_frames(self.max_gap, r), max_step_px=self.max_step_px * s / r,
                       free_step_px=self.free_step_px * s / r, speed_cost=self.speed_cost * r / s,
                       restart_cost=self.restart_cost * r, flight_px=self.flight_px * s / r)


@dataclass(frozen=True)
class BallLabel:
    xy: tuple[float, float] | None  # full-resolution px
    radius: float | None
    weight: float  # 0 = no label
    source: str  # "detected" | "filled" | "none"


def link_ball_path(candidates: Sequence[Sequence[tuple[float, float, float, float]]],
                   colors: Sequence[Sequence[float]] | None = None,
                   config: LinkConfig | None = None, *, fps: float = REF_FPS,
                   height: float = REF_HEIGHT) -> list[BallLabel]:
    """``candidates[t]``: (cx, cy, radius, score) per frame; ``colors[t]``: their colour likelihoods (default 0.5).
    ``fps``: the clip's frames per second; ``height``: the lines of the frames its px are in (``config`` is in
    the units of the tuning footage: ``LinkConfig.for_clip``). Returns one ``BallLabel`` per frame."""
    cfg = (config or LinkConfig()).for_clip(fps, height)
    gap_free = rate_ratio(fps)  # a link spanning up to one frame of the tuning footage skips nothing
    T = len(candidates)
    cands: list[list[tuple[float, float, float, float, float]]] = []  # cx, cy, r, unary, raw score
    for t in range(T):
        row = []
        for k, (cx, cy, r, sc) in enumerate(candidates[t]):
            if sc < cfg.min_score:
                continue
            col = colors[t][k] if colors is not None else 0.5
            row.append((cx, cy, r, sc + cfg.color_weight * col - cfg.bias, sc))
        row.sort(key=lambda c: -c[3])
        cands.append(row[: cfg.max_candidates])
    # V[t][k]: best total of chains ending with candidate k of frame t. Its back pointer is ("chain", t', k')
    # (the previous pick of the same chain) or ("restart", node) (a new chain after the best total that
    # ended before t; node None = nothing before). prefix[t]: (best total ending before t, its node).
    V: list[list[float]] = [[-math.inf] * len(c) for c in cands]
    back: list[list[tuple[str, tuple[int, int] | None]]] = [[("restart", None)] * len(c) for c in cands]
    prefix: list[tuple[float, tuple[int, int] | None]] = [(0.0, None)]
    for t in range(T):
        base, base_node = prefix[t]
        for k, (cx, cy, _, un, _) in enumerate(cands[t]):
            v, ptr = base - cfg.restart_cost + un, ("restart", base_node)
            for g in range(1, cfg.max_gap + 1):
                tp = t - g
                if tp < 0:
                    break
                for kp, (px, py, _, _, _) in enumerate(cands[tp]):
                    step = math.hypot(cx - px, cy - py) / g
                    if step > cfg.max_step_px:
                        continue
                    val = V[tp][kp] + un - cfg.gap_cost * max(0.0, g - gap_free) \
                        - cfg.speed_cost * max(0.0, step - cfg.free_step_px)
                    if val > v:
                        v, ptr = val, ("chain", (tp, kp))
            V[t][k], back[t][k] = v, ptr
        bv, bn = prefix[t]
        for k in range(len(cands[t])):
            if V[t][k] > bv:
                bv, bn = V[t][k], (t, k)
        prefix.append((bv, bn))
    chains: list[list[tuple[int, int]]] = []
    node = prefix[T][1]
    current: list[tuple[int, int]] = []
    while node is not None:
        current.append(node)
        kind, nxt = back[node[0]][node[1]]
        if kind == "restart":
            chains.append(current)
            current = []
        node = nxt
    if current:
        chains.append(current)
    labels = [BallLabel(None, None, 0.0, "none") for _ in range(T)]
    for chain in chains:
        chain.sort()
        for t, k in chain:
            cx, cy, r, un, sc = cands[t][k]
            if un >= cfg.min_pick_weight:
                labels[t] = BallLabel((cx, cy), r, float(min(1.0, 0.5 + un)), "detected")
        for (t0, k0), (t1, k1) in zip(chain[:-1], chain[1:], strict=True):
            if t1 - t0 < 2:
                continue
            a, b = cands[t0][k0], cands[t1][k1]
            speed = math.hypot(b[0] - a[0], b[1] - a[1]) / (t1 - t0)
            if speed < cfg.flight_px:
                continue
            for t in range(t0 + 1, t1):
                s = (t - t0) / (t1 - t0)
                labels[t] = BallLabel((a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])), 0.5 * (a[2] + b[2]),
                                      cfg.fill_weight, "filled")
    return labels


def clip_rate(folder: Path) -> float:
    """Frames per second of a teacher folder's clips: the video's rate (``meta.json``) over the clip's frame step
    (``tools/nba_teacher_labels.py`` stores every ``frame_step``-th frame); 29.97 for a folder without them."""
    try:
        meta = json.loads((Path(folder) / "meta.json").read_text())
    except (OSError, ValueError):
        meta = {}
    meta = meta if isinstance(meta, dict) else {}
    fps = meta.get("fps")
    fps = float(fps) if isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0 else REF_FPS
    step = meta.get("frame_step")
    return fps / step if isinstance(step, int) and step > 1 else fps


def label_folder(folder: Path, config: LinkConfig | None = None) -> list[dict[str, Any]]:
    """``ball_labels.jsonl`` rows of a teacher folder: ``link_ball_path`` over the frames of each clip (in frame
    order, at the folder's ``clip_rate`` and the rows' frame height), with each candidate's colour read from the
    stored frame (``frames/<clip>/<frame>.jpg``, scaled from the full-resolution ``size`` of its row; 0.5 when the
    image is missing)."""
    fps = clip_rate(folder)
    by_clip: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in map(json.loads, (folder / "detections.jsonl").open()):
        by_clip[int(r["clip"])].append(r)
    out: list[dict[str, Any]] = []
    for clip, rows in sorted(by_clip.items()):
        rows.sort(key=lambda r: r["frame"])
        cands, colors = [], []
        for r in rows:
            img = cv2.imread(str(folder / "frames" / f"{clip:04d}" / f"{int(r['frame']):06d}.jpg"))
            s = img.shape[1] / float(r["size"][0]) if img is not None and r.get("size") else 0.5
            row, col = [], []
            for x1, y1, x2, y2, sc in r.get("balls", []):
                cx, cy, rad = (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(x2 - x1, y2 - y1) / 2.0
                row.append((cx, cy, rad, float(sc)))
                col.append(ball_color_likelihood(img, (cx * s, cy * s), rad * s) if img is not None else 0.5)
            cands.append(row)
            colors.append(col)
        height = float(rows[0]["size"][1]) if rows[0].get("size") else REF_HEIGHT
        for r, lab in zip(rows, link_ball_path(cands, colors, config, fps=fps, height=height), strict=True):
            out.append({"clip": clip, "frame": int(r["frame"]),
                        "xy": None if lab.xy is None else [round(lab.xy[0], 2), round(lab.xy[1], 2)],
                        "r": None if lab.radius is None else round(lab.radius, 2), "w": round(lab.weight, 4),
                        "source": lab.source})
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write ball_labels.jsonl for a teacher folder (link_ball_path per clip).")
    ap.add_argument("folder")
    ap.add_argument("--force", action="store_true", help="replace an existing ball_labels.jsonl")
    args = ap.parse_args(argv)
    folder = Path(args.folder)
    try:
        kind = json.loads((folder / "meta.json").read_text()).get("kind")
    except (OSError, ValueError):
        kind = None
    if kind == "pseudo":
        ap.error(f"{folder} is a pseudo-label folder: its labels come from courtvision.training.pseudo_labels")
    out = folder / "ball_labels.jsonl"
    if out.exists() and not args.force:
        ap.error(f"{out} exists (--force replaces it)")
    rows = label_folder(folder)
    tmp = folder / "ball_labels.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    os.replace(tmp, out)
    n: dict[str, int] = defaultdict(int)
    for r in rows:
        n[r["source"]] += 1
    print(f"{out}: {len(rows)} frames, {dict(sorted(n.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
