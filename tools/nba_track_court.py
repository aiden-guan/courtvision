"""Track the broadcast game camera through a video: per-frame court registration (pan / tilt / roll / zoom).

    .venv/bin/python tools/nba_track_court.py VIDEO --scorebug runs/nba/<game>/scorebug.jsonl \
        --out runs/nba/<game>/camera.jsonl [--center X,Y,Z] [--stride 1] [--start S --duration D]

Without ``--center`` the camera centre is estimated first from ``--center-frames`` frames where the scorebug shows
the game clock (``broadcast_court.estimate_center``). Every ``--stride``-th frame is then tracked
(``BroadcastCourtTracker``; hard cuts from ``ShotBoundaryDetector`` drop the camera). One JSON line per tracked
frame: ``frame``, ``t``, ``status`` and, when the court is in view, ``params`` (full-resolution camera) with
``coverage`` / ``inliers`` / ``rmse``. The first line is a header with the centre.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from courtvision.geometry.court import court_for  # noqa: E402
from courtvision.perception.broadcast import ShotBoundaryDetector  # noqa: E402
from courtvision.perception.broadcast_court import (  # noqa: E402
    BroadcastCourtTracker,
    BroadcastTrackerConfig,
    estimate_center,
)
from courtvision.perception.court_search import EvidenceConfig  # noqa: E402
from courtvision.perception.scorebug_ocr import ESPN_2024  # noqa: E402

ESPN_EXCLUDE = [(ESPN_2024.logo_box[0] - 10, ESPN_2024.ocr_box[1] - 6, 1000, 125)]  # the bug and its lower row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--scorebug", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--center", default=None)
    ap.add_argument("--center-frames", type=int, default=24)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    args = ap.parse_args()

    court = court_for("nba")
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cfg = BroadcastTrackerConfig(evidence=EvidenceConfig(exclude=ESPN_EXCLUDE))
    rows = [json.loads(line) for line in open(args.scorebug)]
    if args.center:
        center = np.array([float(v) for v in args.center.split(",")])
    else:
        live = [r["t"] for r in rows if r["game_clock"] is not None]
        picks = np.quantile(live, np.linspace(0.02, 0.98, args.center_frames))
        frames = []
        for t in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, img = cap.read()
            if ok:
                frames.append(img)
        t0 = time.time()
        center, cams = estimate_center(frames, court, cfg.work_width, cfg.evidence)
        if center is None:
            print("camera centre not found", file=sys.stderr)
            return 1
        print(f"centre {np.round(center, 2)} from {sum(c is not None for c in cams)}/{len(frames)} frames "
              f"({time.time() - t0:.0f}s)", flush=True)
    tracker = BroadcastCourtTracker(court, (W, H), center, cfg)
    cuts = ShotBoundaryDetector()
    first = int(args.start * fps)
    last = n_frames if args.duration is None else min(n_frames, int((args.start + args.duration) * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    counts: dict[str, int] = {}
    with out.open("w") as fh:
        fh.write(json.dumps({"header": True, "center": [float(v) for v in center], "image_size": [W, H],
                             "fps": fps, "stride": args.stride}) + "\n")
        for idx in range(first, last):
            ok = cap.grab()
            if not ok:
                break
            if (idx - first) % args.stride:
                continue
            ok, img = cap.retrieve()
            if not ok:
                break
            cut = cuts.update(img)
            fix = tracker.update(img, cut=cut)
            counts[fix.status] = counts.get(fix.status, 0) + 1
            row = {"frame": idx, "t": round(idx / fps, 4), "status": fix.status, "cut": bool(cut)}
            if fix.params is not None:
                p = fix.params
                row.update(params={"pan": round(p.pan, 4), "tilt": round(p.tilt, 4), "roll": round(p.roll, 4),
                                   "f": round(p.f, 2)}, coverage=round(fix.coverage, 3), inliers=fix.inliers,
                           rmse=None if fix.rmse_px is None else round(fix.rmse_px, 3))
            fh.write(json.dumps(row) + "\n")
            n = sum(counts.values())
            if n % 1000 == 0:
                el = time.time() - t0
                print(f"{n} frames (t={idx / fps:.0f}s) {el / n * 1000:.1f} ms/frame {counts}", flush=True)
    print(f"done in {time.time() - t0:.0f}s: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
