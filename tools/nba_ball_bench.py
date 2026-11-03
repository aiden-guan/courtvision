"""Ball-tracking benchmark on held-out teacher clips: the studio's starter tracker vs the NBA student + path linking.

    .venv/bin/python tools/nba_ball_bench.py runs/nba/<game>/teacher [--student runs/models/student/best.pt]
        [--scorebug-profile NAME|PATH]

Labels are the teacher's linked ball path (``training/ball_labels.py``) on the held-out clips (every 10th clip,
never trained on). For each labelled frame (weight >= 0.5) a tracker either reports the ball within ``--tol`` px
(label px: the video frame's own pixels) of the label (hit), elsewhere (wrong) or not at all (miss). Trackers:

* ``starter``: RF-DETR Nano COCO at 384 px (the studio default) + the Kalman ``BallTracker``;
* ``student``: the NBA student detector + the dynamic-programme path linking of ``nba/analyze.link_ball``
  (no camera compensation here: clips are short);
* ``student_scorebug_filter``: the same detections and linking without the candidates centred in the scorebug
  (``analyze.without_balls_in``, as ``nba_analyze.py ball --scorebug-filter on``): the boxes tools/nba_track_court.py
  masks, from the game folder above the teacher folder (its ``scorebug_profile.json``, else the ESPN 2024 bug;
  ``--scorebug-profile`` names another).

The report's ``scorebug`` counts the held-out clips' frames whose best student candidate lies in the bug, those
whose linked ``student`` ball does, and the scored labels that do (the filter can only lose those).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from courtvision.perception.ball import BallTracker  # noqa: E402
from courtvision.perception.scorebug_cover import frame_boxes, in_boxes  # noqa: E402
from courtvision.types import BBox, Detection, ObjClass  # noqa: E402

FILTERED = "student_scorebug_filter"
Key = tuple[int, int]  # (clip, frame)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("teacher")
    ap.add_argument("--student", default="runs/models/student/best.pt")
    ap.add_argument("--starter", default="runs/studio/models/rf-detr-nano.pth")
    ap.add_argument("--tol", type=float, default=15.0)
    ap.add_argument("--scorebug-profile", default=None,
                    help="student_scorebug_filter: this profile's scorebug (a name such as espn-2024, or a saved "
                         "JSON) instead of the game's")
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
    bug = game_scorebug(root, args.scorebug_profile)
    print(f"scorebug (x, y, w, h of a 1920x1080 frame): {bug}", flush=True)

    def load(clip: int, frame: int) -> np.ndarray:
        img = cv2.imread(str(root / "frames" / f"{clip:04d}" / f"{frame:06d}.jpg"))
        w, h = sizes.get((clip, frame), (1920, 1080))
        return cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)

    results: dict[str, dict[Key, tuple[float, float] | None]] = {"starter": {}, "student": {}, FILTERED: {}}
    check = {"boxes": bug, "frames": 0, "best_candidate_inside": 0, "linked_inside": 0}
    import torch
    import rfdetr

    from courtvision.nba.analyze import BallLinkConfig, DetectionCache, without_balls_in
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
        boxes = frame_boxes(bug, sizes.get((clip, frames[0]), (1920, 1080)))  # one video: one frame size
        kept = _link_no_camera(without_balls_in(cache, boxes), BallLinkConfig())
        for f in frames:
            p, q = path.get(f), kept.get(f)
            results["student"][(clip, f)] = None if p is None else (p[0], p[1])
            results[FILTERED][(clip, f)] = None if q is None else (q[0], q[1])
        tally(check, cache.balls, path, boxes, frames, BallLinkConfig().min_score)
        print(f"clip {clip}: {len(frames)} frames", flush=True)
    report: dict[str, Any] = score(results, labels, args.tol)
    check["labels_inside"] = labels_inside(labels, results["student"], bug, sizes)
    report["scorebug"] = check
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


def game_scorebug(teacher: Path, profile: str | None = None) -> list[tuple[int, int, int, int]]:
    """The scorebug of the teacher folder's game (x, y, w, h of a 1920x1080 frame): tools/nba_track_court.py's
    ``bug_exclude`` for the game folder above it (its reads and the video its game.json names: the saved
    scorebug_profile.json, else the ESPN 2024 bug), or ``profile``'s."""
    from tools.nba_track_court import bug_exclude

    game = Path(os.path.abspath(teacher)).parent  # ("." in the teacher folder too; links not followed)
    video = None
    try:
        video = json.loads((game / "game.json").read_text()).get("video")
    except (OSError, ValueError, AttributeError):
        pass
    return bug_exclude(str(game / "scorebug.jsonl"), video, profile)


def scored(labels: dict[Key, dict], res: dict[Key, Any]) -> list[tuple[Key, dict]]:
    """The labels a tracker is scored on: labelled frames of its clips with weight 0.5 or more and a ball."""
    return [(key, lab) for key, lab in labels.items() if key in res and lab["w"] >= 0.5 and lab["xy"]]


def score(results: dict[str, dict[Key, tuple[float, float] | None]], labels: dict[Key, dict],
          tol: float) -> dict[str, dict[str, Any]]:
    """Each tracker's share of scored labels with its ball within ``tol`` px (hit), elsewhere (wrong) or none (miss)."""
    report = {}
    for name, res in results.items():
        hit = wrong = miss = 0
        for key, lab in scored(labels, res):
            p = res[key]
            if p is None:
                miss += 1
            elif math.hypot(p[0] - lab["xy"][0], p[1] - lab["xy"][1]) <= tol:
                hit += 1
            else:
                wrong += 1
        n = hit + wrong + miss
        report[name] = {"labelled_frames": n, "hit": round(hit / n, 3), "wrong": round(wrong / n, 3),
                        "miss": round(miss / n, 3)}
    return report


def tally(check: dict[str, Any], balls: dict[int, np.ndarray], path: dict[int, tuple[float, float, float]],
          boxes: list[tuple[int, int, int, int]], frames: list[int], floor: float) -> None:
    """Count a clip's frames, those whose best candidate scoring ``floor`` or more lies in ``boxes`` (frame px) and
    those whose linked ball does."""
    for f in frames:
        check["frames"] += 1
        b = balls.get(f, np.zeros((0, 3), np.float32))
        b = b[b[:, 2] >= floor]
        if len(b) and in_boxes(b[np.argmax(b[:, 2])][None], boxes)[0]:
            check["best_candidate_inside"] += 1
        p = path.get(f)
        if p is not None and in_boxes(np.array([p[:2]]), boxes)[0]:
            check["linked_inside"] += 1


def labels_inside(labels: dict[Key, dict], res: dict[Key, Any], bug: list[tuple[int, int, int, int]],
                  sizes: dict[Key, tuple[int, int]]) -> int:
    """Scored labels (``scored``) that lie in the scorebug (``bug``: boxes of a 1920x1080 frame, in each label's frame
    size)."""
    n = 0
    for key, lab in scored(labels, res):
        size = tuple(sizes.get(key, (1920, 1080)))
        n += int(in_boxes(np.array([lab["xy"][:2]], float), frame_boxes(bug, size))[0])  # type: ignore[arg-type]
    return n


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
