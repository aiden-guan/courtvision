"""Ball-tracking benchmark on held-out teacher clips: the studio's starter tracker vs the NBA student + path linking.

    .venv/bin/python tools/nba_ball_bench.py runs/nba/<game>/teacher [--student runs/models/student/best.pt]

Labels are the teacher's linked ball path (``training/ball_labels.py``) on the held-out clips (every 10th clip,
never trained on). For each labelled frame (weight >= 0.5) a tracker either reports the ball within ``--tol`` px
(label px: the video frame's own pixels) of the label (hit), elsewhere (wrong) or not at all (miss). Trackers:

* ``starter``: RF-DETR Nano COCO at 384 px (the studio default) + the Kalman ``BallTracker``;
* ``student``: the NBA student detector + the dynamic-programme path linking of ``nba/analyze.link_ball``
  (no camera compensation here: clips are short).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from courtvision.perception.ball import BallTracker  # noqa: E402
from courtvision.types import BBox, Detection, ObjClass  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("teacher")
    ap.add_argument("--student", default="runs/models/student/best.pt")
    ap.add_argument("--starter", default="runs/studio/models/rf-detr-nano.pth")
    ap.add_argument("--tol", type=float, default=15.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    root = Path(args.teacher)
    labels = {(r["clip"], r["frame"]): r for r in map(json.loads, (root / "ball_labels.jsonl").open())}
    # the frame's own size: labels are in its pixels, the stored frames are smaller
    sizes = {(r["clip"], r["frame"]): tuple(r.get("size") or (1920, 1080))
             for r in map(json.loads, (root / "detections.jsonl").open())}
    clips: dict[int, list[int]] = defaultdict(list)
    for clip, frame in labels:
        if clip % 10 == 5:
            clips[clip].append(frame)

    def load(clip: int, frame: int) -> np.ndarray:
        img = cv2.imread(str(root / "frames" / f"{clip:04d}" / f"{frame:06d}.jpg"))
        w, h = sizes.get((clip, frame), (1920, 1080))
        return cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)

    results: dict[str, dict[tuple[int, int], tuple[float, float] | None]] = {"starter": {}, "student": {}}
    import torch
    import rfdetr

    from courtvision.nba.analyze import BallLinkConfig, DetectionCache
    from courtvision.perception.student import StudentConfig, StudentDetector

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    nano = rfdetr.RFDETRNano(pretrain_weights=args.starter, device=dev)
    student = StudentDetector(StudentConfig(weights=args.student, ball_thresh=0.04))
    for clip, frames in sorted(clips.items()):
        frames.sort()
        imgs = [load(clip, f) for f in frames]
        tracker = BallTracker()
        cache = DetectionCache()
        for k, (f, img) in enumerate(zip(frames, imgs, strict=True)):
            det = nano.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), threshold=0.3)
            names = det.data.get("class_name")
            balls = [Detection(BBox(*map(float, det.xyxy[i])), ObjClass.BALL, float(det.confidence[i]))
                     for i in range(len(det.xyxy)) if names[i] == "sports ball"]
            people = [BBox(*map(float, det.xyxy[i])) for i in range(len(det.xyxy)) if names[i] == "person"]
            st = tracker.update(balls, f, f / 29.97, None, people=people)
            results["starter"][(clip, f)] = st.image_xy if st.visible and not st.interpolated else None
            hist = imgs[max(0, k - 2):k + 1]
            while len(hist) < 3:
                hist.insert(0, hist[0])
            b, _ = student.infer([hist])[0]
            cache.balls[f] = np.asarray(b, np.float32).reshape(-1, 3)
            cache.persons[f] = np.zeros((0, 5), np.float32)
        path = _link_no_camera(cache, BallLinkConfig())
        for f in frames:
            p = path.get(f)
            results["student"][(clip, f)] = None if p is None else (p[0], p[1])
        print(f"clip {clip}: {len(frames)} frames", flush=True)
    report = {}
    for name, res in results.items():
        hit = wrong = miss = 0
        for key, lab in labels.items():
            if key not in res or lab["w"] < 0.5 or not lab["xy"]:
                continue
            p = res[key]
            if p is None:
                miss += 1
            elif math.hypot(p[0] - lab["xy"][0], p[1] - lab["xy"][1]) <= args.tol:
                hit += 1
            else:
                wrong += 1
        n = hit + wrong + miss
        report[name] = {"labelled_frames": n, "hit": round(hit / n, 3), "wrong": round(wrong / n, 3),
                        "miss": round(miss / n, 3)}
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


def _link_no_camera(cache, cfg):  # noqa: ANN001, ANN202
    """``link_ball`` with an identity camera (short clips; camera motion stays in the motion cost)."""
    from courtvision.nba import analyze

    class _Ident:
        def get(self, f: int) -> object:
            return object()

        def frame_to_frame(self, a: int, b: int) -> None:
            return None

    return analyze.link_ball(cache, _Ident(), cfg)  # type: ignore[arg-type]


if __name__ == "__main__":
    raise SystemExit(main())
