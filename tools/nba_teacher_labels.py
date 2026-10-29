"""Teacher detections on sampled clips of a broadcast game (training data for the student detector).

    .venv/bin/python tools/nba_teacher_labels.py VIDEO --scorebug runs/nba/<game>/scorebug.jsonl \
        --out runs/nba/<game>/teacher [--clips 150 --clip-frames 36]
    .venv/bin/python tools/nba_teacher_labels.py VIDEO --times runs/nba/<game>/hard.json \
        --out runs/nba/<game>/teacher-hard [--clips 0]

Clips of consecutive frames are spread over the game: most where the scorebug is shown (live play, free throws,
dead balls), a share where it is not (replays, close-ups: negatives and other views). With ``--times`` (a JSON file
of clip start times in video seconds, e.g. ``tools/nba_hard_examples.py``'s list of where the analysis went wrong)
exactly those clips are labelled, plus ``--clips`` random ones (0 by default with ``--times``) that overlap none
of them. Every frame of a clip is run through RF-DETR Small (COCO) at a high input resolution (balls are 20-35 px
in 1080p) with a low threshold. Frames are stored downscaled to ``--save-width`` (JPEG) under
``frames/<clip>/<frame>.jpg``; detections in full-resolution pixels go to ``detections.jsonl`` (one line per frame:
clip, frame index, t, persons and ball candidates with scores); ``meta.json`` names the video and each clip's start
and origin (``times`` / ``random``). At the end the ball labels are linked (``ball_labels.jsonl``,
``courtvision.training.ball_labels``; ``--no-ball-labels`` leaves them out). Research use only (docs/TRAINING.md §1).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_CLIPS = 150


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


def load_times(path: Path) -> list[float]:
    """Clip start times (video s) of a ``--times`` file: a JSON list of seconds, a list of objects with ``t``, or an
    object whose ``clips`` is such a list (``tools/nba_hard_examples.py``) or whose ``times`` is a list of seconds."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("clips", data.get("times"))
    if not isinstance(data, list):
        raise TypeError(f"{path}: expected a list of clip start times, or an object with 'clips' or 'times'")
    out = []
    for item in data:
        t = item.get("t") if isinstance(item, dict) else item
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not np.isfinite(t):
            raise TypeError(f"{path}: not a time in seconds: {item!r}")
        out.append(float(t))
    return out


def starts_from_times(times: Sequence[float], fps: float, clip_frames: int, n_frames: int) -> list[int]:
    """Start frames of the listed clips (each once, kept inside the video), sorted."""
    last = max(0, n_frames - clip_frames - 1)
    return sorted({min(max(0, round(t * fps)), last) for t in times})


def merge_starts(fixed: Sequence[int], extra: Sequence[int], clip_frames: int) -> list[tuple[int, str]]:
    """Every ``fixed`` start ("times") and the ``extra`` starts ("random") whose clip overlaps none of them, sorted."""
    fixed = sorted(set(fixed))
    out = [(s, "times") for s in fixed]
    for s in extra:
        if all(abs(s - f) >= clip_frames for f in fixed):
            out.append((s, "random"))
    return sorted(out)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--scorebug", default=None, help="scorebug.jsonl: needed for random clips")
    ap.add_argument("--out", required=True)
    ap.add_argument("--times", default=None, help="JSON of clip start times (s) to label (tools/nba_hard_examples.py)")
    ap.add_argument("--clips", type=int, default=None,
                    help=f"random clips (default {DEFAULT_CLIPS}; 0 with --times)")
    ap.add_argument("--clip-frames", type=int, default=36)
    ap.add_argument("--bugless-share", type=float, default=0.12)
    ap.add_argument("--weights", default=str(Path.home() / ".roboflow/models/rf-detr-small.pth"))
    ap.add_argument("--shape", default="736,1312")
    ap.add_argument("--save-width", type=int, default=960)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ball-labels", action="store_true", help="do not write ball_labels.jsonl at the end")
    return ap


def plan(args: argparse.Namespace, fps: float, n_frames: int) -> list[tuple[int, str]]:
    """The clips to label: (start frame, origin) in time order."""
    n_random = args.clips if args.clips is not None else (0 if args.times else DEFAULT_CLIPS)
    fixed = starts_from_times(load_times(Path(args.times)), fps, args.clip_frames, n_frames) if args.times else []
    rand: list[int] = []
    if n_random > 0:
        if not args.scorebug:
            raise ValueError("random clips need --scorebug (or pass --clips 0 with --times)")
        rand = plan_clips(Path(args.scorebug), n_frames / fps, fps, n_random, args.clip_frames, args.bugless_share,
                          args.seed)
    return merge_starts(fixed, rand, args.clip_frames)


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not args.times and not args.scorebug:
        ap.error("--scorebug is required (random clips) unless --times lists the clips")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        clips = plan(args, fps, n_frames)
    except (TypeError, ValueError) as e:
        ap.error(str(e))
    if not clips:
        ap.error("no clips to label")
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({
        "kind": "teacher", "video": str(Path(args.video).resolve()), "fps": fps, "n_frames": n_frames,
        "clip_frames": args.clip_frames, "shape": args.shape, "save_width": args.save_width,
        "times": str(Path(args.times).resolve()) if args.times else None, "label_origin": "cv_auto",
        "rights_status": "research_only",
        "clips": [{"clip": ci, "start": s, "origin": o} for ci, (s, o) in enumerate(clips)]}, indent=1))
    n_listed = sum(o == "times" for _, o in clips)
    print(f"{len(clips)} clips: {n_listed} listed, {len(clips) - n_listed} random", flush=True)

    warnings.filterwarnings("ignore")
    import torch
    import rfdetr

    shape = tuple(int(v) for v in args.shape.split(","))
    model = rfdetr.RFDETRSmall(pretrain_weights=args.weights, device="mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    done = 0
    starts = [s for s, _ in clips]
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
    cap.release()
    print(f"done: {done} frames in {time.time() - t0:.0f}s")
    if not args.no_ball_labels:  # this run's detections replace the old ones, so their labels go too
        from courtvision.training import ball_labels

        return ball_labels.main([str(out), "--force"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
