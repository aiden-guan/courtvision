"""Teacher detections on sampled clips of a broadcast game (training data for the student detector).

    .venv/bin/python tools/nba_teacher_labels.py VIDEO --scorebug runs/nba/<game>/scorebug.jsonl \
        --out runs/nba/<game>/teacher [--clips 150 --clip-frames 36]

Clips of consecutive frames are spread over the game: most where the scorebug is shown (live play, free throws,
dead balls), a share where it is not (replays, close-ups: negatives and other views). Every frame of a clip is
run through RF-DETR Small (COCO) at a high input resolution (balls are 20-35 px in 1080p) with a low threshold.
Frames are stored downscaled to ``--save-width`` (JPEG) under ``frames/<clip>/<frame>.jpg``; detections in
full-resolution pixels go to ``detections.jsonl`` (one line per frame: clip, frame index, t, persons and ball
candidates with scores). Research use only (docs/TRAINING.md §1).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def plan_clips(scorebug: Path, duration_s: float, fps: float, n_clips: int, clip_frames: int,
               bugless_share: float, seed: int) -> list[int]:
    """Start frames of the clips: spread evenly over the bug-shown time, plus ``bugless_share`` elsewhere."""
    rows = [json.loads(line) for line in scorebug.open()]
    shown = np.array([r["t"] for r in rows if r["game_clock"] is not None])
    hidden = np.array([r["t"] for r in rows if r["game_clock"] is None])
    rng = np.random.default_rng(seed)
    n_hidden = int(round(n_clips * bugless_share))
    n_shown = n_clips - n_hidden
    picks = list(np.quantile(shown, np.linspace(0.002, 0.998, n_shown)))
    if len(hidden) and n_hidden:
        picks += list(rng.choice(hidden, n_hidden, replace=False))
    starts = sorted({int(t * fps) for t in picks})
    out: list[int] = []
    for s in starts:  # no overlaps
        if not out or s >= out[-1] + clip_frames:
            out.append(min(s, int(duration_s * fps) - clip_frames - 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--scorebug", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=150)
    ap.add_argument("--clip-frames", type=int, default=36)
    ap.add_argument("--bugless-share", type=float, default=0.12)
    ap.add_argument("--weights", default=str(Path.home() / ".roboflow/models/rf-detr-small.pth"))
    ap.add_argument("--shape", default="736,1312")
    ap.add_argument("--save-width", type=int, default=960)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    import torch
    import rfdetr

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    starts = plan_clips(Path(args.scorebug), n_frames / fps, fps, args.clips, args.clip_frames,
                        args.bugless_share, args.seed)
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    shape = tuple(int(v) for v in args.shape.split(","))
    model = rfdetr.RFDETRSmall(pretrain_weights=args.weights, device="mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    done = 0
    with (out / "detections.jsonl").open("w") as fh:
        for ci, s in enumerate(starts):
            cap.set(cv2.CAP_PROP_POS_FRAMES, s)
            clip_dir = out / "frames" / f"{ci:04d}"
            clip_dir.mkdir(exist_ok=True)
            for k in range(args.clip_frames):
                ok, img = cap.read()
                if not ok:
                    break
                idx = s + k
                h, w = img.shape[:2]
                ten = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)
                det = model.predict(ten.to(model.model.device), threshold=0.04, shape=shape, include_source_image=False)
                names = det.data.get("class_name")
                persons, balls = [], []
                for i in range(len(det.xyxy)):
                    box = [round(float(v), 1) for v in det.xyxy[i]]
                    sc = round(float(det.confidence[i]), 3)
                    if names[i] == "person" and sc >= 0.25:
                        persons.append(box + [sc])
                    elif names[i] == "sports ball":
                        balls.append(box + [sc])
                small = cv2.resize(img, (args.save_width, int(round(h * args.save_width / w))), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(clip_dir / f"{idx:06d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
                fh.write(json.dumps({"clip": ci, "frame": idx, "t": round(idx / fps, 4), "size": [w, h],
                                     "persons": persons, "balls": balls}) + "\n")
                done += 1
            fh.flush()
            el = time.time() - t0
            print(f"clip {ci + 1}/{len(starts)}: {done} frames, {el / max(done, 1) * 1000:.0f} ms/frame", flush=True)
    print(f"done: {done} frames in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
