"""Track the broadcast game camera through a video: per-frame court registration (pan / tilt / roll / zoom).

    .venv/bin/python tools/nba_track_court.py VIDEO --scorebug runs/nba/<game>/scorebug.jsonl \
        --out runs/nba/<game>/camera.jsonl [--center X,Y,Z] [--stride 1] [--start S --duration D] \
        [--scorebug-profile NAME|PATH] [--track-margin-ft FT] [--no-reacquire-after-loss]

Without ``--center`` the camera centre is estimated first from ``--center-frames`` frames where the scorebug shows
the game clock (``broadcast_court.estimate_center``). Every ``--stride``-th frame is then tracked
(``BroadcastCourtTracker``; hard cuts from ``ShotBoundaryDetector`` drop the camera). One JSON line per tracked
frame: ``frame``, ``t``, ``status`` and, when the court is in view, ``params`` (full-resolution camera) with
``coverage`` / ``inliers`` / ``rmse``. The first line is a header with the centre.

The video is checked first (``analyze.probe_video``; its warnings are printed). On a variable-frame-rate video the
header says ``"vfr": true``, ``t`` is each frame's decoder timestamp (``CameraTrack.time`` uses them), the frames are
numbered in decoding order from the start (it is read from its start: OpenCV's seeks land on other frames there)
and ``--start`` / ``--duration`` cut parts at those timestamps. ``CameraTrack.load`` fills gaps of up to 0.1 s
between tracked frames, so a ``--stride`` beyond 3 at 29.97 fps (6 at 59.94) leaves every tracked frame alone.

The scorebug is kept out of the court-line evidence (``bug_exclude``): the one ``--scorebug-profile`` names (a
built-in name such as ``espn-2024``, or a saved JSON), else the one saved for the game (``scorebug_profile.json``
beside the ``--scorebug`` reads, beside the video, or in the ``runs/nba/<game>/`` folder whose ``game.json`` names
the video), else the ESPN 2024 bug.
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
from courtvision.nba.analyze import CameraTrack, probe_video, rate_ratio, scale_frames  # noqa: E402
from courtvision.perception.broadcast import ShotBoundaryDetector  # noqa: E402
from courtvision.perception.broadcast_court import (  # noqa: E402
    BroadcastCourtTracker,
    BroadcastTrackerConfig,
    estimate_center,
)
from courtvision.perception.court_search import EvidenceConfig  # noqa: E402
from courtvision.perception.scorebug_ocr import ESPN_2024  # noqa: E402

ESPN_EXCLUDE = [(ESPN_2024.logo_box[0] - 10, ESPN_2024.ocr_box[1] - 6, 1000, 125)]  # the bug and its lower row


def bug_exclude(scorebug_path: str, video: str | None = None,
                profile: str | None = None) -> list[tuple[int, int, int, int]]:
    """Boxes (x, y, w, h of a 1920x1080 frame) the court evidence ignores: the scorebug of ``profile`` (a name or a
    JSON path), else the one ``nba_scan_scorebug.py`` found for this video (``find_saved_profile``: beside its reads
    first), with a margin of 10 px of a 1080-line frame left and right and half its height above and below for a
    second row; else the ESPN 2024 bug."""
    from courtvision.perception.scorebug_layout import find_saved_profile, load_profile, resolve_profile

    if profile:
        prof = resolve_profile(profile)  # asked for: one that cannot be read is an error, not the ESPN bug
    else:
        path = find_saved_profile(scorebug_path, video)
        if path is None:
            return ESPN_EXCLUDE
        try:
            prof = load_profile(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"warning: {path}: {exc}; masking the ESPN 2024 bug", file=sys.stderr, flush=True)
            return ESPN_EXCLUDE
    if prof.name == ESPN_2024.name:
        return ESPN_EXCLUDE
    sx, sy = 1920.0 / prof.ref_size[0], 1080.0 / prof.ref_size[1]
    out = []
    for x, y, w, h in (prof.ocr_box, prof.logo_box):
        mx, my = 10.0 / sy, 0.5 * h  # profile px: the same mask whatever size the profile was measured at
        out.append((int(round((x - mx) * sx)), int(round((y - my) * sy)), int(round((w + 2 * mx) * sx)),
                    int(round((h + 2 * my) * sy))))
    return out


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
    ap.add_argument("--scorebug-profile", default=None)
    ap.add_argument("--track-margin-ft", type=float, default=None,
                    help="tracked frames read lines around the projected court plus this margin (default: "
                         "BroadcastTrackerConfig's); 0: the frame-wide floor mask of before")
    ap.add_argument("--no-reacquire-after-loss", action="store_true",
                    help="re-acquire after a loss in play only on the regular schedule (as before)")
    args = ap.parse_args()

    court = court_for("nba")
    info = probe_video(args.video)
    for w in info.warnings:
        print(f"warning: {args.video}: {w}", file=sys.stderr, flush=True)
    vfr = info.vfr and info.times is not None
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fill = scale_frames(CameraTrack.MAX_FILL_FRAMES, rate_ratio(fps))
    if args.stride - 1 > fill:
        print(f"warning: --stride {args.stride} leaves {args.stride - 1} frames between tracked frames; the camera "
              f"track fills gaps of up to {fill} at {fps:.2f} fps, so no two tracked frames would be joined",
              file=sys.stderr, flush=True)
    exclude = bug_exclude(args.scorebug, args.video, args.scorebug_profile)
    print(f"scorebug mask (x, y, w, h of a 1920x1080 frame): {exclude}", flush=True)
    cfg = BroadcastTrackerConfig(evidence=EvidenceConfig(exclude=exclude))
    if args.track_margin_ft is not None:
        cfg.track_margin_ft = args.track_margin_ft
    if args.no_reacquire_after_loss:
        cfg.reacquire_after_loss = ()
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
    if vfr:  # parts cut at timestamps; frames numbered from the start
        assert info.times is not None
        first = int(np.searchsorted(info.times, args.start - 1e-6))
        last = n_frames if args.duration is None else \
            min(n_frames, int(np.searchsorted(info.times, args.start + args.duration - 1e-6)))
        cap.release()
        cap = cv2.VideoCapture(args.video)
        k = 0
        while k < first and cap.grab():
            k += 1
    else:
        first = int(args.start * fps)
        last = n_frames if args.duration is None else min(n_frames, int((args.start + args.duration) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    counts: dict[str, int] = {}
    with out.open("w") as fh:
        header = {"header": True, "center": [float(v) for v in center], "image_size": [W, H], "fps": fps,
                  "stride": args.stride}
        if vfr:
            header["vfr"] = True
        fh.write(json.dumps(header) + "\n")
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
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if vfr else idx / fps  # the decoder's timestamp
            row = {"frame": idx, "t": round(t, 4), "status": fix.status, "cut": bool(cut)}
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
