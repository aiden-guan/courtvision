r"""Compare court-processing speed and exact outputs with a local Git revision (no cloud calls).

    .venv/bin/python tools/court_processing_bench.py VIDEO --start 1200 --frames 300 \
        --reference-ref REV [--center X,Y,Z] [--repeats 3] [--out report.json]

The centre defaults to broadcast_camera.json beside VIDEO. Each trial decodes the same frames and runs the
broadcast court tracker and cut detector. Imports and the first frame warm up before timing; trial order
alternates. Decoding keeps one frame at a time; stored court results are small records for the clip. This measures
decoding and court processing, not detection, OCR, events, or the complete Studio pipeline. Any output difference fails the
benchmark even when the candidate is faster. The reference revision's Python source is executed locally.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from courtvision.geometry.court import court_for  # noqa: E402
from courtvision.perception import broadcast_court  # noqa: E402
from courtvision.perception.broadcast import ShotBoundaryDetector  # noqa: E402
from tools.nba_track_court import bug_exclude  # noqa: E402


def reference_modules(ref: str) -> Any:
    """Load only the reference's two court modules; all other dependencies are shared with the candidate."""
    for filename in ("court_search", "broadcast_court"):
        source = subprocess.check_output(
            ["git", "show", f"{ref}:courtvision/perception/{filename}.py"], cwd=ROOT, text=True,
        )
        source = source.replace("from .court_search import (", "from ._bench_court_search import (")
        name = f"courtvision.perception._bench_{filename}"
        spec = importlib.util.spec_from_loader(name, loader=None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses look up their owning module
        exec(compile(source, f"{ref}/{filename}.py", "exec"), module.__dict__)
    return module


def trial(module: Any, video: Path, start: int, count: int, center: np.ndarray,
          exclude: list[tuple[int, int, int, int]]) -> tuple[float, list[dict[str, Any]]]:
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open {video}")
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        cfg = module.BroadcastTrackerConfig(evidence=module.EvidenceConfig(exclude=exclude))
        tracker = module.BroadcastCourtTracker(court_for("nba"), size, center, cfg)
        cuts = ShotBoundaryDetector()
        results = []
        t0 = time.perf_counter()
        for i in range(count):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Video ended after {i} of {count} requested frames")
            cut = cuts.update(frame)
            fix = tracker.update(frame, cut=cut or i == 0)
            results.append({"status": fix.status, "params": None if fix.params is None else fix.params.to_dict(),
                            "coverage": fix.coverage, "inliers": fix.inliers, "rmse_px": fix.rmse_px})
        return (time.perf_counter() - t0) * 1000 / count, results
    finally:
        cap.release()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--reference-ref", required=True)
    ap.add_argument("--center", help="X,Y,Z in court feet; default: sidecar broadcast_camera.json")
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds at the video's nominal rate")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--scorebug-profile")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.frames < 1 or args.repeats < 1 or not np.isfinite(args.start) or args.start < 0:
        ap.error("frames/repeats must be positive; start must be finite and nonnegative")
    center = np.array([float(v) for v in args.center.split(",")] if args.center else
                      json.loads((args.video.parent / "broadcast_camera.json").read_text())["center"])
    if center.shape != (3,) or not np.isfinite(center).all():
        ap.error("center must contain three finite numbers")
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not np.isfinite(fps) or fps <= 0:
        ap.error("cannot read the video's frame rate")
    start = round(args.start * fps)
    exclude = bug_exclude(str(args.video.parent / "scorebug.jsonl"), str(args.video), args.scorebug_profile)
    reference = reference_modules(args.reference_ref)
    modules = {"before": reference, "after": broadcast_court}
    for module in modules.values():
        trial(module, args.video, start, 1, center, exclude)
    times: dict[str, list[float]] = {label: [] for label in modules}
    expected = None
    for repeat in range(args.repeats):
        order = list(modules) if repeat % 2 == 0 else list(reversed(modules))
        for label in order:
            ms, results = trial(modules[label], args.video, start, args.frames, center, exclude)
            if expected is None:
                expected = results
            elif results != expected:
                frame = next(i for i, (a, b) in enumerate(zip(expected, results, strict=True)) if a != b)
                raise AssertionError(f"{label} trial {repeat + 1} changed frame {start + frame}: "
                                     f"{expected[frame]} != {results[frame]}")
            times[label].append(ms)
            print(f"{label} {repeat + 1}: {ms:.3f} ms/frame; exact outputs agree", file=sys.stderr, flush=True)
    medians = {label: statistics.median(values) for label, values in times.items()}
    report = {"video": str(args.video), "reference_ref": args.reference_ref, "start_frame": start,
              "frames": args.frames, "repeats": args.repeats, "exact_outputs": True,
              "statuses": dict(Counter(r["status"] for r in expected)), "ms_per_frame": times,
              "median_ms_per_frame": medians, "speedup": medians["before"] / medians["after"]}
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
