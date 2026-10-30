"""Read a broadcast's scorebug through a whole video: video time -> period, game clock, scores, shot clock.

    .venv/bin/python tools/nba_scan_scorebug.py VIDEO --out runs/nba/<game>/scorebug.jsonl [--hz 3]
        [--profile auto|espn-2024|PROFILE.json] [--teams AWY,HOM] [--samples 30] [--burst 2]

Where the bug draws its fields comes from ``--profile``:

* ``auto`` (default): ``--samples`` moments spread over the video (``--burst`` frames a second apart at each) are
  decoded with ffmpeg and read whole with Apple Vision; ``scorebug_layout.discover`` finds the layout from how the
  tokens behave (a clock that counts down, scores that never decrease, ...; perception/scorebug_layout.py). The team
  tricodes (``--teams``, else ``espn_summary.json`` next to ``--out``) tell home from away; without them the left
  (top) score is assumed to be the away team. With ``espn_summary.json`` the play-by-play scores check the sides,
  and swap them when the reads show them the other way round. The presence region is then chosen on the pixels of
  the sampled moments. The profile is saved as ``scorebug_profile.json`` next to ``--out`` and printed; when
  discovery fails the scan falls back to ``espn-2024`` with a warning.
* ``espn-2024``: the ESPN / ABC 2024-25 bug, measured by hand on one game.
* a JSON path: a saved profile (``scorebug_profile.json`` of an earlier run).

ffmpeg then decodes only the bug's strip at ``--hz``; each strip is read with ``OCRScorebugReader`` (Apple Vision).
One JSON line per sample: ``t``, the parsed fields (None when unread) and ``logo``, the bug-presence correlation
(1.0 until the reference is learned). The output feeds ``ClockTimeline.fit`` (training/align_pbp.py).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from courtvision.perception.scorebug_layout import (  # noqa: E402
    PROFILE_FILE,
    Discovery,
    LayoutError,
    discover,
    ocr_frame,
    pbp_score_lookup,
    presence_box,
    presence_score,
    resolve_profile,
    save_profile,
    side_votes,
    swap_sides,
)
from courtvision.perception.scorebug_ocr import ESPN_2024, Box, OCRScorebugReader, ScorebugProfile  # noqa: E402


def make_ocr() -> Any:
    from courtvision.perception.vision_ocr import VisionOCR

    return VisionOCR()


def probe(video: str) -> tuple[int, int, float | None]:
    """Width, height and duration (seconds; None when the container does not say)."""
    out = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                     "stream=width,height:format=duration", "-of", "json", video],
                                    check=True, capture_output=True, text=True).stdout)
    st = out["streams"][0]
    dur = out.get("format", {}).get("duration")
    try:
        duration = float(dur) if dur is not None else None
    except ValueError:
        duration = None
    return int(st["width"]), int(st["height"]), duration


def _ffmpeg(video: str, t: float) -> list[str]:
    cmd = ["ffmpeg", "-loglevel", "error"]
    if sys.platform == "darwin":
        cmd += ["-hwaccel", "videotoolbox"]
    return cmd + ["-ss", f"{max(0.0, t):.3f}", "-i", video]


def grab(video: str, t: float, size: tuple[int, int], crop: Box | None = None) -> np.ndarray | None:
    """The frame at ``t`` (BGR), or its ``crop`` (x, y, w, h, even numbers); None past the end."""
    w, h = size
    cmd = _ffmpeg(video, t) + ["-frames:v", "1"]
    if crop is not None:
        x, y, w, h = crop
        cmd += ["-vf", f"crop={w}:{h}:{x}:{y}"]
    buf = subprocess.run(cmd + ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], capture_output=True,
                         check=False).stdout
    n = w * h * 3
    return np.frombuffer(buf[:n], np.uint8).reshape(h, w, 3).copy() if len(buf) >= n else None


def sample_times(t0: float, t1: float, n: int, burst: int, gap: float = 1.0) -> list[list[float]]:
    """``n`` moments spread over [t0, t1] (3% kept clear of either end), ``burst`` frames ``gap`` s apart each."""
    span = max(0.0, t1 - t0 - (burst - 1) * gap)
    a, b = t0 + 0.03 * span, t0 + 0.97 * span
    return [[a + (b - a) * (k + 0.5) / n + j * gap for j in range(burst)] for k in range(n)]


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def summary_teams(summary: Any) -> tuple[str, str] | None:
    """(away, home) abbreviations of an ESPN game summary (the fields courtvision.nba.espn.teams reads)."""
    try:
        side = {c["homeAway"]: str(c["team"]["abbreviation"])
                for c in summary["header"]["competitions"][0]["competitors"]}
        return side["away"], side["home"]
    except (KeyError, IndexError, TypeError):
        return None


def summary_scores(summary: Any) -> list[tuple[int, float, int, int]]:
    """(period, clock seconds left, away score, home score) after each ESPN play that carries the score."""
    rows = []
    for p in (summary or {}).get("plays", []) if isinstance(summary, dict) else []:
        try:
            text = str(p["clock"]["displayValue"]).strip()
            m = re.fullmatch(r"(\d+):(\d+(?:\.\d+)?)", text)
            clock = 60.0 * int(m.group(1)) + float(m.group(2)) if m else float(text)
            rows.append((int(p["period"]["number"]), clock, int(p["awayScore"]), int(p["homeScore"])))
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def parse_teams(spec: str | None) -> tuple[str, str] | None:
    if not spec:
        return None
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != 2:
        raise SystemExit(f"--teams wants AWY,HOM (two team codes), got {spec!r}")
    return parts[0], parts[1]


def discover_video(video: str, size: tuple[int, int], t0: float, t1: float, ocr: Any, teams: tuple[str, str] | None,
                   n: int, burst: int) -> tuple[Discovery, list[float]]:
    """Whole frames at ``n`` moments read by ``ocr``, and the layout they show (raises LayoutError)."""
    samples, times = [], []
    for moment in sample_times(t0, t1, n, burst):
        for t in moment:
            frame = grab(video, t, size)
            if frame is not None:
                samples.append(ocr_frame(ocr, frame))
                times.append(t)
    return discover(samples, size, teams, times=times), times


def check_sides(profile: ScorebugProfile, disc: Discovery, summary: Any) -> ScorebugProfile:
    """Home / away checked against the ESPN play-by-play scores at the periods and clocks the bug showed."""
    rows = summary_scores(summary)
    if not rows:
        if disc.sides_by == "position":
            print("  home/away: no play-by-play scores (espn_summary.json) to check the assumption")
        return profile
    agree, swapped = side_votes(disc.reads, pbp_score_lookup(rows))
    print(f"  home/away against the ESPN play-by-play scores: {agree} reads agree, {swapped} read swapped")
    if swapped >= 3 and swapped >= 3 * agree:
        print("  home/away: SWAPPED to match the play-by-play scores")
        return swap_sides(profile, f"home/away swapped by the ESPN play-by-play scores ({swapped} reads swapped, "
                                   f"{agree} agree)")
    if agree >= 3 and agree >= 3 * swapped:
        notes = tuple(n for n in profile.notes if not n.startswith("home/away ASSUMED"))
        note = f"home/away confirmed by the ESPN play-by-play scores ({agree} reads agree, {swapped} swapped)"
        return replace(profile, notes=(*notes, note))
    print("WARNING: the play-by-play scores neither confirm nor contradict home/away")
    return profile


def presence_region(profile: ScorebugProfile, size: tuple[int, int]) -> Box:
    """Where to look for the presence box: the bug's crop and current presence box, widened by four field heights
    across and one down (logos and tricodes can sit beyond the numbers); even numbers."""
    x, y, w, h = profile.ocr_box
    lx, ly, lw, lh = profile.logo_box
    th = (profile.away_score_box or profile.ocr_box)[3]
    x0, y0 = max(0, min(x, lx) - 4 * th), max(0, min(y, ly) - th)
    x1, y1 = min(size[0], max(x + w, lx + lw) + 4 * th), min(size[1], max(y + h, ly + lh) + th)
    x0, y0 = x0 - x0 % 2, y0 - y0 % 2
    w2, h2 = (x1 - x0) - (x1 - x0) % 2, (y1 - y0) - (y1 - y0) % 2
    return (x0, y0, w2, h2)


def refine_presence(video: str, profile: ScorebugProfile, disc: Discovery, times: list[float],
                    size: tuple[int, int]) -> ScorebugProfile:
    """The presence box checked on the pixels of the sampled moments showing the bug and those without: the
    discovered one while it tells them apart (shown >= 0.9, a margin of 0.5 over hidden), else the window of the
    bug that does best (``scorebug_layout.presence_box``) when it beats the discovered one."""
    shown_t, hidden_t, seen = [], [], set()
    for t, r, m in zip(times, disc.reads, disc.moments, strict=True):
        if m in seen:
            continue
        seen.add(m)
        if r is None:
            hidden_t.append(t)
        elif r["game_clock"] is not None and (r["away_score"] is not None or r["home_score"] is not None):
            shown_t.append(t)
    region = presence_region(profile, size)
    shown = [c for c in (grab(video, t, size, region) for t in shown_t) if c is not None]
    hidden = [c for c in (grab(video, t, size, region) for t in hidden_t) if c is not None]
    if len(shown) < 3:
        print(f"  presence: kept ({len(shown)} sampled moments show the bug, too few to check on pixels)")
        return profile
    cur = presence_score(shown, hidden, region[:2], profile.logo_box)
    best = presence_box(shown, hidden, region[:2], profile)

    def fmt(c: Any) -> str:
        return f"shown >= {c.shown:.2f}" + ("" if c.hidden is None else f", hidden <= {c.hidden:.2f}")

    good = cur is not None and cur.shown >= 0.9 and cur.margin >= 0.5  # the discovered box works: keep it
    if not good and best is not None and best.shown >= 0.8 and (cur is None or best.margin > cur.margin + 0.05):
        print(f"  presence: box {best.box} chosen on pixels of {len(shown)} + {len(hidden)} moments ({fmt(best)})"
              + ("" if cur is None else f"; the discovered box gave {fmt(cur)}"))
        notes = tuple(n for n in profile.notes if not n.startswith("presence region GUESSED"))
        note = (f"presence box chosen on pixels of {len(shown)} moments with the bug and {len(hidden)} without "
                f"({fmt(best)})")
        return replace(profile, logo_box=best.box, notes=(*notes, note))
    if cur is not None:
        print(f"  presence: box {profile.logo_box} kept ({fmt(cur)} over {len(shown)} + {len(hidden)} moments)")
    return profile


def choose_profile(args: argparse.Namespace, size: tuple[int, int], duration: float | None, out: Path,
                   ocr: Any) -> ScorebugProfile:
    if args.profile != "auto":
        prof = resolve_profile(args.profile)
        print(f"scorebug profile {prof.name} ({args.profile})")
        return prof
    summary = read_json(out.parent / "espn_summary.json")
    teams = parse_teams(args.teams) or summary_teams(summary)
    t0 = args.start
    t1 = t0 + args.duration if args.duration else duration
    if t1 is None or t1 - t0 < 10.0:
        print("WARNING: the video's duration is unknown or too short to sample; scorebug profile espn-2024")
        return ESPN_2024
    print(f"scorebug discovery: {args.samples} moments x {args.burst} frames, {size[0]}x{size[1]}, teams "
          + (f"{teams[0]} (away) @ {teams[1]} (home)" if teams else "unknown"), flush=True)
    t_start = time.time()
    try:
        disc, times = discover_video(args.video, size, t0, t1, ocr, teams, args.samples, args.burst)
    except LayoutError as exc:
        print(f"WARNING: scorebug discovery failed: {exc}")
        print("WARNING: falling back to espn-2024, measured on one ESPN game: another broadcaster's bug will read "
              "wrong or not at all")
        return ESPN_2024
    for line in disc.report:
        print("  " + line)
    prof = check_sides(disc.profile, disc, summary)
    prof = refine_presence(args.video, prof, disc, times, size)
    for note in prof.notes:
        print("  note: " + note)
    path = save_profile(prof, out.parent / PROFILE_FILE, {"discovery": {
        "video": args.video, "range_s": [round(t0, 1), round(t1, 1)], "samples": args.samples, "burst": args.burst,
        "teams": list(teams) if teams else None, "report": disc.report}})
    print(f"scorebug profile saved to {path} ({time.time() - t_start:.0f}s); rerun with --profile {path} to reuse it",
          flush=True)
    return prof


def scan(video: str, out: Path, profile: ScorebugProfile, size: tuple[int, int], hz: float, start: float,
         duration: float | None, ocr: Any) -> int:
    W, H = size
    prof = profile.scaled((W, H))
    # the strip holding the presence box and the OCR box, full frame width (keeps frame x coordinates)
    y0 = max(0, min(prof.ocr_box[1], prof.logo_box[1]))
    y1 = min(H, max(prof.ocr_box[1] + prof.ocr_box[3], prof.logo_box[1] + prof.logo_box[3]))
    y0 -= y0 % 2
    y1 = min(y1 + y1 % 2, H - H % 2)
    h = y1 - y0
    cmd = _ffmpeg(video, start)
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-vf", f"fps={hz},crop={W}:{h}:0:{y0}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    reader = OCRScorebugReader(profile, ocr=ocr)
    frame_bytes = W * h * 3
    canvas = np.zeros((H, W, 3), np.uint8)  # the strip pasted at its frame position: profile boxes apply as is
    out.parent.mkdir(parents=True, exist_ok=True)
    n, t0 = 0, time.time()
    assert proc.stdout is not None
    with out.open("w") as fh:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            canvas[y0:y0 + h] = np.frombuffer(buf, np.uint8).reshape(h, W, 3)
            t = start + n / hz
            f = reader.read_fields(canvas)
            logo = reader.presence(canvas)
            if reader._logo is None and f["game_clock"] is not None and f["period"] is not None:
                reader.read(canvas, t)  # learns the presence reference
            fh.write(json.dumps({"t": round(t, 3), "logo": round(logo, 3), **f}) + "\n")
            n += 1
            if n % 1000 == 0:
                print(f"{n} samples, t={t:.0f}s, {n / (time.time() - t0):.1f}/s", flush=True)
    print(f"done: {n} samples in {time.time() - t0:.0f}s -> {out}")
    return proc.wait()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hz", type=float, default=3.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--profile", default="auto", help="auto (discover the layout; default), espn-2024, or a profile "
                                                      "JSON path")
    ap.add_argument("--teams", default=None, help="AWY,HOM: the away and home team codes as the bug may show them "
                                                  "(default: from espn_summary.json next to --out)")
    ap.add_argument("--samples", type=int, default=30, help="moments sampled for --profile auto")
    ap.add_argument("--burst", type=int, default=2, help="frames a second apart at each sampled moment")
    args = ap.parse_args(argv)

    W, H, duration = probe(args.video)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ocr = make_ocr()
    try:
        profile = choose_profile(args, (W, H), duration, out, ocr)
        return scan(args.video, out, profile, (W, H), args.hz, args.start, args.duration, ocr)
    finally:
        ocr.close()


if __name__ == "__main__":
    raise SystemExit(main())
