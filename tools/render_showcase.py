"""Export a short public recording from saved NBA detections and timeline analysis.

No model inference or cloud calls run during export. Requires an existing nba_analyze
run, its detection cache, source footage, and ffmpeg. Only the finished MP4 and poster
belong in the public showcase; source video and model weights remain local.

    .venv/bin/python tools/render_showcase.py runs/nba/2024-12-25-LAL-GSW \
        --start 981 --seconds 30 --out showcase/media
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from nba_analyze import load_game
from nba_overlay import load_overlay

from courtvision.nba.analyze import CachedDetector, DetectionCache
from courtvision.perception.tracker import ByteTracker, TrackerConfig
from courtvision.types import EventKind, ObjClass


def text(image: np.ndarray, value: str, xy: tuple[int, int], size: float = .5,
         color: tuple[int, int, int] = (228, 231, 234)) -> None:
    cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def brackets(image: np.ndarray, bbox: tuple[float, ...], scale: float, track_id: int) -> None:
    x1, y1, x2, y2 = [round(v * scale) for v in bbox]
    y1 += 52
    y2 += 52
    color = (166, 225, 206)
    length = max(4, min(16, (x2 - x1) // 3, (y2 - y1) // 4))
    for x, dx in ((x1, 1), (x2, -1)):
        for y, dy in ((y1, 1), (y2, -1)):
            cv2.line(image, (x, y), (x + dx * length, y), color, 1, cv2.LINE_AA)
            cv2.line(image, (x, y), (x, y + dy * length), color, 1, cv2.LINE_AA)
    label = f"T{track_id:02d}"
    (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .35, 1)
    yy = max(52 + height + 8, y1 - 5)
    cv2.rectangle(image, (x1, yy - height - 5), (x1 + width + 8, yy + 3), (17, 24, 23), -1)
    text(image, label, (x1 + 4, yy), .35, color)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game", type=Path)
    parser.add_argument("--tag", default="full")
    parser.add_argument("--start", type=float, default=981)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--out", type=Path, default=Path("showcase/media"))
    args = parser.parse_args()
    if not 0 < args.seconds <= 30 or args.start < 0:
        parser.error("choose a positive recording up to 30 seconds and a non-negative start")
    game, camera = load_game(args.game)
    if camera.vfr:
        parser.error("use constant-frame-rate footage so cached detections match the video")
    source = Path(game["video"])
    overlay = load_overlay(args.game, args.tag, game)
    first, stop = camera.frame_at(args.start), camera.frame_at(args.start + args.seconds)
    cache = DetectionCache()
    cache_paths = []
    for path in sorted((args.game / "detections-full").glob("*.npz")):
        lower, upper = [int(v) for v in path.stem.split("-")]
        if lower < stop and upper > first - round(camera.fps):
            cache.merge(DetectionCache.load(path))
            cache_paths.append(path)
    if not cache.persons:
        raise SystemExit("no saved detections cover this recording")
    detector = CachedDetector(cache, camera=camera)
    tracker = ByteTracker(TrackerConfig(high_thresh=.3, low_thresh=.1, new_track_thresh=.35))
    # BallTrack owns its storage format; never infer a path from raw candidates.
    from courtvision.nba.analyze import BallTrack
    ball_path = BallTrack.load(args.game / "ball_track.json").path
    width, height = 1280, 720
    scale = width / camera.image_size[0]
    if round(camera.image_size[1] * scale) != height:
        raise SystemExit("the public recording expects 16:9 source footage")
    args.out.mkdir(parents=True, exist_ok=True)
    temporary = args.out / ".recording.mp4"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-s", "1280x816", "-r", str(camera.fps), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "26", "-maxrate", "2M",
           "-bufsize", "4M", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(temporary)]
    encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    capture = cv2.VideoCapture(str(source))
    warmup = max(0, first - round(camera.fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, warmup)
    count = 0
    poster = None
    try:
        for index in range(warmup, stop):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"source ended at frame {index}, expected {stop}")
            t = camera.time(index)
            detected = detector.detect(image, index, t)
            players = detected.of(ObjClass.PLAYER)
            tracks = tracker.update(players, index)
            if index < first:
                continue
            frame = np.full((816, 1280, 3), (14, 13, 12), np.uint8)
            frame[52:772] = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            text(frame, "COURTVISION", (22, 32), .65)
            text(frame, "LAKERS / WARRIORS", (230, 31), .43, (156, 160, 166))
            text(frame, "RECORDED ANALYSIS", (1062, 31), .43, (112, 167, 238))
            for track in tracks:
                if track.time_since_update == 0:
                    brackets(frame, track.bbox.as_tuple(), scale, track.track_id)
            ball = ball_path.get(index)
            if ball is not None:
                xy = (round(ball[0] * scale), round(ball[1] * scale) + 52)
                cv2.circle(frame, xy, 11, (79, 222, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, xy, 2, (79, 222, 255), -1, cv2.LINE_AA)
            team = overlay.team_at(t)
            possession = overlay.names[team] if team is not None else "UNSETTLED"
            text(frame, f"POSSESSION  {possession}", (22, 799), .44)
            recent = [e for e in overlay.recent(t, 3) if e.kind in
                      (EventKind.SHOT_MADE, EventKind.SHOT_MISSED, EventKind.STEAL, EventKind.REBOUND)]
            if recent:
                event = recent[-1]
                name = overlay.names.get(event.team, "")
                label = event.kind.value.replace("_", " ").upper()
                text(frame, f"{name}  {label}", (340, 799), .44, (166, 225, 206))
            else:
                text(frame, "PLAYER TRACKS / BALL PATH", (340, 799), .40, (156, 160, 166))
            text(frame, f"{(index-first)/camera.fps:04.1f}s / {args.seconds:.0f}s", (1134, 799), .43)
            if count == round(camera.fps * 2):
                poster = frame.copy()
            assert encoder.stdin is not None
            encoder.stdin.write(frame.tobytes())
            count += 1
            if count % 150 == 0:
                print(f"Rendered {count}/{stop-first} frames", flush=True)
    finally:
        capture.release()
        assert encoder.stdin is not None
        encoder.stdin.close()
        code = encoder.wait()
    if code or count != stop - first or poster is None:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"incomplete recording: encoder={code}, frames={count}/{stop-first}")
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    video = args.out / f"nba-analysis-{digest[:12]}.mp4"
    temporary.rename(video)
    poster_path = args.out / f"nba-analysis-{digest[:12]}.webp"
    if not cv2.imwrite(str(poster_path), poster, [cv2.IMWRITE_WEBP_QUALITY, 88]):
        raise SystemExit("could not write the recording poster")
    provenance = {"game": game.get("title"), "source_start_s": first / camera.fps,
                  "duration_s": count / camera.fps, "frames": count, "fps": camera.fps,
                  "analysis_tag": args.tag, "detector": json.loads((args.game / "detections-full/loop.json").read_text()),
                  "video": str(video), "video_sha256": digest, "poster": str(poster_path),
                  "audio": False, "detection_caches": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in cache_paths},
                  "timeline_sha256": hashlib.sha256((args.game / "runs" / args.tag / "timeline.jsonl").read_bytes()).hexdigest()}
    Path("docs/showcase-recording.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
