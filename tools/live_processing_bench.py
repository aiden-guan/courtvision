"""Measure the complete Studio video pipeline without cloud calls or frame dropping.

Examples:
  .venv/bin/python tools/live_processing_bench.py --start 2400 --seconds 20 --out runs/bench/live-base.json
  .venv/bin/python tools/live_processing_bench.py --worker --paced --seconds 60 --out runs/bench/live-paced.json

Uses local research footage/models. --code-root can select a saved source tree for output comparisons.
Initialises the camera from an existing camera track, separating one-time venue setup from steady processing.
Both OCR and the video-only shot judge remain enabled. --audit saves frame states, rim-crop checksums and events;
--profile profiles perception, including its worker thread. The benchmark does not include browser rendering.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import dataclasses
import json
import os
import pstats
import sys
import time
import zlib
from pathlib import Path


async def run(args: argparse.Namespace) -> None:
    import numpy as np
    import cv2

    from courtvision.config import AppConfig
    from courtvision.geometry.homography import Calibration, Homography
    from courtvision.nba.analyze import CameraTrack
    from courtvision.perception.student import StudentConfig, StudentDetector
    from courtvision.pipeline import VideoFileSource
    from courtvision.server.hub import Hub, dumps
    from courtvision.server.studio import Workspace
    from courtvision.server.studio_game import StudioGame

    if args.court_threads is not None:
        os.environ["COURTVISION_THREADS"] = str(args.court_threads)
    if args.opencv_threads is not None:
        cv2.setNumThreads(args.opencv_threads)
    det = StudentDetector(StudentConfig(weights=str(args.weights.resolve())))
    det.detect(np.zeros((540, 960, 3), np.uint8), 0, 0.0)
    det.reset()
    workspace = Workspace(AppConfig(), args.root.resolve())
    cfg = workspace.video_config(args.video_id, det)
    cfg.jev.mode, cfg.narration.llm_enabled, cfg.narration.tts = "off", False, "none"
    cfg.pipeline.tail_s, cfg.pipeline.court_worker = 0.0, args.worker
    cfg.pipeline.parallel_live = args.worker
    if args.lookahead is not None:
        cfg.pipeline.live_lookahead_frames = args.lookahead
    cfg.perception.save_calibration = False
    camera = CameraTrack.load(sorted(args.camera.glob("*.jsonl")))
    cam = camera.get(camera.frame_at(args.start))
    if cam is None:
        raise ValueError("Choose a start frame where the saved camera is tracked")
    class BenchmarkSource(VideoFileSource):
        async def _iterate(self):
            first_t, first_wall = None, None
            async for item in super()._iterate():
                if args.causal and args.paced:
                    if first_t is None:
                        first_t, first_wall = item.t, time.perf_counter()
                    await asyncio.sleep(max(0.0, first_wall + item.t - first_t - time.perf_counter()))
                yield item

    source = BenchmarkSource(cfg.source.path, realtime=args.paced, start_s=args.start, duration_s=args.seconds)
    source.fps = camera.fps
    source.live = args.causal
    hub = Hub(cfg.hub)
    frames = []
    times = []
    profile = cProfile.Profile()

    def observe(frame):
        if args.audit:
            record = json.loads(dumps(dataclasses.replace(frame, rim_crops=None)))
            record["rim_crc"] = {side: zlib.crc32(crop.tobytes()) for side, crop in (frame.rim_crops or {}).items()}
            frames.append(record)

    class MeasuredGame(StudioGame):
        def _perceive(self, *a, **kw):
            if args.profile:
                profile.enable()
            try:
                return super()._perceive(*a, **kw)
            finally:
                if args.profile:
                    profile.disable()

        async def process(self, item):
            t0 = time.perf_counter()
            await super().process(item)
            times.append((item.t, (time.perf_counter() - t0) * 1000.0, time.perf_counter() - started))

    game = MeasuredGame(cfg, source, hub, realtime=args.paced or args.live_path, detector=det, env={}, on_frame=observe)
    if args.live_path and not args.paced:
        game.pacer.enabled = False
    game._auto_center, game._auto_params = camera.center.copy(), cam
    game._calibration = Calibration(Homography(np.linalg.inv(cam.G(camera.image_size))), camera.image_size)
    warm_started = time.perf_counter()
    if game.shot_judge is not None:
        warmup = getattr(game.shot_judge, "warmup", None)
        if warmup is not None:
            await asyncio.to_thread(warmup)
    warm_seconds = time.perf_counter() - warm_started
    started = time.perf_counter()
    await game.run()
    elapsed = time.perf_counter() - started
    steady = [row for row in times if row[0] >= args.start + args.warmup]
    playback_wall = times[-1][2] - times[0][2] if len(times) > 1 else 0.0
    playback_source = times[-1][0] - times[0][0] if len(times) > 1 else 0.0
    auto = getattr(game.perception, "auto", None)
    report = {"seconds": elapsed, "frames": game._frames_done, "fps": game._frames_done / elapsed,
              "source_fps": source.fps, "paced": args.paced, "worker": args.worker, "backend": det.backend,
              "live_path": args.live_path,
              "lookahead_frames": getattr(cfg.pipeline, "live_lookahead_frames", None),
              "court_search_threads": os.environ.get("COURTVISION_THREADS", "auto"),
              "causal": args.causal, "model_warmup_s": warm_seconds,
              "opencv_threads_requested": getattr(cfg.pipeline, "opencv_threads", None),
              "opencv_threads_reported": cv2.getNumThreads(),
              "steady_fps": ((len(steady) - 1) / (steady[-1][2] - steady[0][2])) if len(steady) > 1 else None,
              "first_frame_s": times[0][2] if times else None,
              "post_first_frame_fps": (len(times) - 1) / playback_wall if playback_wall > 0 else None,
              "post_first_frame_realtime_factor": playback_source / playback_wall if playback_wall > 0 else None,
              "lag_growth_s": playback_wall - playback_source,
              "last_frame_lag_s": times[-1][2] - (times[-1][0] - args.start) if times else None,
              "dropped": game._dropped, "errors": game._errors,
              "worker_fallback": getattr(auto, "fallback_reason", None), "latency": game.latency_report(),
              "frame_times": times, "events": [e.to_dict() for e in game.engine.events]}
    if args.audit:
        report["states"] = frames
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps(report))
    if game._errors or game._dropped:
        raise AssertionError(f"Invalid throughput sample: {game._errors} errors, {game._dropped} dropped frames")
    if args.profile:
        pstats.Stats(profile).strip_dirs().sort_stats("cumtime").print_stats(35)
    print(json.dumps({k: v for k, v in report.items() if k not in ("states", "frame_times", "events")}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("runs/studio"))
    ap.add_argument("--video-id", default="72e1591f5d4d4fdfb639b20e56396fce")
    ap.add_argument("--camera", type=Path, default=Path("runs/nba/2024-12-25-LAL-GSW/camera_parts"))
    ap.add_argument("--weights", type=Path, default=Path("runs/studio/models/nba-student.pt"))
    ap.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--start", type=float, default=2400.0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=3.0, help="source seconds excluded from the steady-rate measurement")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--lookahead", type=int, choices=(1, 2, 3), help="maximum decoded/speculative frames in Live")
    ap.add_argument("--court-threads", type=int, choices=(1, 2, 4, 8), help="court-search threads; same candidate grid")
    ap.add_argument("--causal", action="store_true", help="give the worker only incoming pixels, as in Live")
    ap.add_argument("--opencv-threads", type=int)
    ap.add_argument("--paced", action="store_true")
    ap.add_argument("--live-path", action="store_true", help="exercise Live's executor/decoder pipeline without pacing")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.seconds <= 0 or args.start < 0:
        ap.error("seconds must be positive; start must be nonnegative")
    sys.path.insert(0, str(args.code_root.resolve()))
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
