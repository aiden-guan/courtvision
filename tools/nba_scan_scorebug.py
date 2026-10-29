"""Read a broadcast's scorebug through a whole video: video time -> period, game clock, scores, shot clock.

    .venv/bin/python tools/nba_scan_scorebug.py VIDEO --out runs/nba/<game>/scorebug.jsonl [--hz 3]

ffmpeg decodes only the bug's strip at ``--hz``; each strip is read with ``OCRScorebugReader`` (Apple Vision).
One JSON line per sample: ``t``, the parsed fields (None when unread) and ``logo``, the bug-logo correlation
(the presence signal; 1.0 until the logo is learned). The output feeds ``ClockTimeline.fit`` (training/align_pbp.py).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from courtvision.perception.scorebug_ocr import ESPN_2024, OCRScorebugReader  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hz", type=float, default=3.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    args = ap.parse_args()

    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "stream=width,height", "-of", "json", args.video],
                                      check=True, capture_output=True, text=True).stdout)
    W, H = probe["streams"][0]["width"], probe["streams"][0]["height"]
    prof = ESPN_2024.scaled((W, H))
    # the strip holding the logo and the OCR box, full frame width (keeps frame x coordinates)
    y0 = min(prof.ocr_box[1], prof.logo_box[1])
    y1 = max(prof.ocr_box[1] + prof.ocr_box[3], prof.logo_box[1] + prof.logo_box[3])
    y0 -= y0 % 2
    h = y1 - y0 + ((y1 - y0) % 2)
    cmd = ["ffmpeg", "-loglevel", "error", "-hwaccel", "videotoolbox", "-ss", str(args.start), "-i", args.video]
    if args.duration:
        cmd += ["-t", str(args.duration)]
    cmd += ["-vf", f"fps={args.hz},crop={W}:{h}:0:{y0}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    reader = OCRScorebugReader(ESPN_2024)
    frame_bytes = W * h * 3
    canvas = np.zeros((H, W, 3), np.uint8)  # the strip pasted at its frame position: profile boxes apply as is
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n, t0 = 0, time.time()
    assert proc.stdout is not None
    with out.open("w") as fh:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            canvas[y0:y0 + h] = np.frombuffer(buf, np.uint8).reshape(h, W, 3)
            t = args.start + n / args.hz
            f = reader.read_fields(canvas)
            logo = reader.presence(canvas)
            if reader._logo is None and f["game_clock"] is not None and f["period"] is not None:
                reader.read(canvas, t)  # learns the logo
            fh.write(json.dumps({"t": round(t, 3), "logo": round(logo, 3), **f}) + "\n")
            n += 1
            if n % 1000 == 0:
                print(f"{n} samples, t={t:.0f}s, {n / (time.time() - t0):.1f}/s", flush=True)
    reader.close()
    print(f"done: {n} samples in {time.time() - t0:.0f}s -> {out}")
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
