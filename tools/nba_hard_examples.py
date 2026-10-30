"""Teacher clips where the analysis of a game went wrong (active learning for the student detector).

    .venv/bin/python tools/nba_hard_examples.py runs/nba/<game> --tag <run> --out runs/nba/<game>/hard.json \
        [--file fused.jsonl] [--kinds shot,rebound,turnover] [--outcomes miss,extra,team] [--max-clips 150]
    .venv/bin/python tools/nba_teacher_labels.py VIDEO --times runs/nba/<game>/hard.json --out runs/nba/<game>/teacher-hard

The run's events (``--file``: ``fused.jsonl`` when the run has it, else ``events.jsonl``) are matched to the aligned
ESPN play-by-play as tools/nba_diagnose.py matches them: per kind, each true event to the nearest unused prediction
of that kind within ``--tol`` s, greedily in time order (``ok`` / ``team`` / ``result`` / ``value`` / ``miss``), and
the predictions left over are ``extra``. Each event with one of ``--outcomes`` asks for a window around its time
(``WINDOWS``: shots from 2.5 s before to 1 s after, for the release, the flight and the rim; rebounds and turnovers
1.5 s either side), inside the run's window. Events are taken kind by kind in ``--kinds`` order (shots and rebounds
first); within a kind, those whose window the ball path covers least come first (the detector most likely lost the
ball), then in time order; an event is taken while the clips of all windows taken stay within ``--max-clips``.
Overlapping windows merge, and each merged window is cut into back-to-back clips of ``--clip-frames`` frames
``--frame-step`` apart (as tools/nba_teacher_labels.py stores them: by default the student's input spacing, every
second frame at 50-60 fps, so a clip covers the same time at any rate). Events with the game camera tracked on less
than ``--min-camera-share`` of their window are skipped, and so are such clips: the student only runs where the
camera is, so those errors are not the detector's.

The frame rate and frame times are the camera track's (``CameraTrack.time``: decoder timestamps on a
variable-frame-rate video); without a camera track, the video's rate (``game.json``), else 29.97 fps with a warning.

Writes ``--out`` (JSON): the video, fps and clip length, one entry per clip (start ``t`` in video seconds and its
``frame``, the events that asked for it, camera and ball-path shares) and the outcome counts. Research use only
(docs/TRAINING.md §1).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nba_diagnose import KINDS, _fmt  # noqa: E402  (the diagnosis' kind groups and row format)

from courtvision.eval.events import latest_revisions  # noqa: E402
from courtvision.types import Event, EventStatus  # noqa: E402

# seconds before / after an event's time that its clips cover
WINDOWS: dict[str, tuple[float, float]] = {"shot": (2.5, 1.0), "rebound": (1.5, 1.5), "turnover": (1.5, 1.5),
                                           "assist": (2.0, 1.0), "steal": (1.5, 1.5)}


@dataclass(frozen=True)
class Outcome:
    name: str  # kind group of tools/nba_diagnose.py ("shot", "rebound", ...)
    outcome: str  # ok / team / result / value / miss / extra
    t: float  # the true event's time (the prediction's for an extra)
    truth: Event | None
    pred: Event | None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.name, "outcome": self.outcome, "t": round(self.t, 2),
                "truth": _fmt(self.truth) if self.truth else "", "pred": _fmt(self.pred) if self.pred else ""}


def match(truth: Sequence[Event], pred: Sequence[Event], name: str, tol: float) -> list[Outcome]:
    """tools/nba_diagnose.py's matching of one kind group: each true event (in time order) to the nearest unused
    prediction of the group within ``tol`` s; then the unmatched predictions as extras."""
    kinds = KINDS[name]
    T = sorted((e for e in truth if e.kind in kinds), key=lambda e: e.t)
    P = sorted((e for e in pred if e.kind in kinds), key=lambda e: e.t)
    used: set[int] = set()
    out: list[Outcome] = []
    for e in T:
        best = None
        for k, p in enumerate(P):
            if k in used or abs(p.t - e.t) > tol:
                continue
            if best is None or abs(p.t - e.t) < abs(P[best].t - e.t):
                best = k
        if best is None:
            out.append(Outcome(name, "miss", e.t, e, None))
            continue
        used.add(best)
        p = P[best]
        outcome = "ok" if p.team is e.team else "team"
        if name == "shot" and outcome == "ok" and (p.kind is not e.kind or p.value != e.value):
            outcome = "result" if p.kind is not e.kind else "value"
        out.append(Outcome(name, outcome, e.t, e, p))
    for k, p in enumerate(P):
        if k not in used:
            out.append(Outcome(name, "extra", p.t, None, p))
    return out


def _share(lo: int, hi: int, frames: set[int] | None) -> float | None:
    if frames is None:
        return None
    return sum(1 for f in range(lo, hi + 1) if f in frames) / max(1, hi - lo + 1)


def _tiles(windows: Sequence[tuple[int, int, int]], clip_frames: int, camera_frames: set[int] | None,
           min_camera_share: float) -> list[tuple[int, list[int]]]:
    """Clips covering ``windows`` (first frame, last frame, id): windows merge while one's clips would reach into the
    next, each merged window is cut into back-to-back clips from its start, and clips with too little camera are
    dropped -> (start frame, ids of the windows the clip overlaps)."""
    merged: list[list[Any]] = []  # [lo, hi, ids]
    for lo, hi, i in sorted(windows):
        if merged:
            m = merged[-1]
            end = m[0] + math.ceil((m[1] - m[0] + 1) / clip_frames) * clip_frames  # past its last clip
            if lo < end:
                m[1] = max(m[1], hi)
                m[2].append((lo, hi, i))
                continue
        merged.append([lo, hi, [(lo, hi, i)]])
    out: list[tuple[int, list[int]]] = []
    for lo, hi, members in merged:
        for s in range(lo, hi + 1, clip_frames):
            share = _share(s, s + clip_frames - 1, camera_frames)
            if share is not None and share < min_camera_share:
                continue
            ids = [i for a, b, i in members if a <= s + clip_frames - 1 and b >= s]
            out.append((s, ids))
    return out


def plan_clips(errors: Sequence[Outcome], fps: float, clip_frames: int, max_clips: int, kinds: Sequence[str],
               camera_frames: set[int] | None = None, ball_frames: set[int] | None = None,
               bounds: tuple[float, float] = (0.0, math.inf), min_camera_share: float = 0.5,
               windows: dict[str, tuple[float, float]] | None = None, frame_step: int = 1,
               camera: Any | None = None) -> tuple[list[dict[str, Any]], list[Outcome], dict[str, int]]:
    """Teacher clips for ``errors`` (see the module docstring) -> (clips, the errors they cover, skip counts). A clip
    takes ``clip_frames * frame_step`` video frames. ``camera``: a variable-frame-rate ``CameraTrack`` maps times and
    frames (``time`` / ``frame_at``); otherwise frame = time * ``fps``."""
    win = windows or WINDOWS
    vfr = camera is not None and getattr(camera, "vfr", False)
    span = clip_frames * frame_step
    order = {k: i for i, k in enumerate(kinds)}
    cand = []
    skipped = {"off_camera": 0, "over_cap": 0}
    for e in errors:
        pre, post = win[e.name]
        if vfr:
            lo = max(0, camera.frame_at(max(bounds[0], e.t - pre)))
            hi = camera.frame_at(min(bounds[1], e.t + post))
        else:
            lo = max(0, math.floor(max(bounds[0], e.t - pre) * fps))
            hi = math.ceil(min(bounds[1], e.t + post) * fps)
        if hi < lo:
            continue
        cam = _share(lo, hi, camera_frames)
        if cam is not None and cam < min_camera_share:
            skipped["off_camera"] += 1
            continue
        on = [f for f in range(lo, hi + 1) if camera_frames is None or f in camera_frames]
        ball = None if ball_frames is None else sum(1 for f in on if f in ball_frames) / max(1, len(on))
        cand.append((order.get(e.name, len(order)), 0.0 if ball is None else ball, e.t, lo, hi, e, cam, ball))
    cand.sort(key=lambda c: c[:3])
    taken: list[tuple[int, int, int]] = []
    info: list[tuple[Outcome, float | None, float | None]] = []
    for _, _, _, lo, hi, e, cam, ball in cand:
        trial = [*taken, (lo, hi, len(taken))]
        if len(_tiles(trial, span, camera_frames, min_camera_share)) > max_clips:
            skipped["over_cap"] += 1
            continue
        taken = trial
        info.append((e, cam, ball))
    clips = []
    for s, ids in _tiles(taken, span, camera_frames, min_camera_share):
        clips.append({"t": round(camera.time(s) if vfr else s / fps, 3), "frame": s,
                      "events": [info[i][0].to_dict() for i in sorted(ids)],
                      "camera_share": _share(s, s + span - 1, camera_frames),
                      "ball_share": None if ball_frames is None else round(
                          sum(1 for f in range(s, s + span) if f in ball_frames) / span, 3)})
    return clips, [e for e, _, _ in info], skipped


def video_fps(root: Path, video: str | None) -> float:
    """The frame rate of the game's video (``game.json``), for a game folder without a camera track; 29.97 fps, with
    a warning, when it cannot be read."""
    import cv2

    if video:
        p = Path(video)
        if not p.is_absolute() and not p.exists() and (root / p).exists():
            p = root / p
        cap = cv2.VideoCapture(str(p))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if cap.isOpened() else 0.0
        cap.release()
        if math.isfinite(fps) and fps > 0:
            return fps
    print(f"warning: {root}: no camera track and no readable video: frames counted at 29.97 fps", file=sys.stderr)
    return 29.97


def load_truth(root: Path, lo: float, hi: float) -> list[Event]:
    """The aligned play-by-play of the game folder as truth events in [lo, hi] (as tools/nba_diagnose.py)."""
    from nba_analyze import truth

    from courtvision.nba.evaluate import truth_events

    _, aligned = truth(root)
    return [e for e in truth_events(aligned.aligned) if lo <= e.t <= hi]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--file", default=None, help="events of the run (default fused.jsonl when present, else events.jsonl)")
    ap.add_argument("--kinds", default="shot,rebound,turnover", help="in priority order")
    ap.add_argument("--outcomes", default="miss,extra,team")
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--max-clips", type=int, default=150)
    ap.add_argument("--clip-frames", type=int, default=36, help="frames tools/nba_teacher_labels.py stores per clip")
    ap.add_argument("--frame-step", type=int, default=0,
                    help="video frames between two stored frames (0: the student's input spacing, as "
                         "tools/nba_teacher_labels.py's default)")
    ap.add_argument("--min-camera-share", type=float, default=0.5)
    args = ap.parse_args(argv)
    kinds = [k for k in args.kinds.split(",") if k]
    if any(k not in KINDS or k not in WINDOWS for k in kinds):
        ap.error(f"--kinds: choose from {', '.join(KINDS)}")
    outcomes = {o for o in args.outcomes.split(",") if o}
    root = Path(args.root)
    run_dir = root / "runs" / args.tag
    run = json.loads((run_dir / "run.json").read_text())
    lo, hi = float(run.get("start") or 0.0), float(run.get("end") or 1e9)
    name = args.file or ("fused.jsonl" if (run_dir / "fused.jsonl").exists() else "events.jsonl")
    pred = [e for e in latest_revisions(Event.from_dict(json.loads(line)) for line in (run_dir / name).open())
            if e.status is not EventStatus.RETRACTED and lo <= e.t <= hi]
    truth = load_truth(root, lo, hi)
    rows = [o for k in kinds for o in match(truth, pred, k, args.tol)]
    errors = [o for o in rows if o.outcome in outcomes]

    from courtvision.nba.analyze import BallTrack, CameraTrack
    from courtvision.perception.student import history_step

    camera, camera_frames, ball_frames = None, None, None
    game = json.loads((root / "game.json").read_text()) if (root / "game.json").exists() else {}
    parts = sorted((root / "camera_parts").glob("part*.jsonl"))
    if parts:
        camera = CameraTrack.load(parts)
        fps, camera_frames = camera.fps, set(camera.params)
    else:
        fps = video_fps(root, game.get("video"))
    if (root / "ball_track.json").exists():
        ball_frames = set(BallTrack.load(root / "ball_track.json").path)
    step = args.frame_step if args.frame_step > 0 else history_step(fps)
    clips, chosen, skipped = plan_clips(errors, fps, args.clip_frames, args.max_clips, kinds, camera_frames,
                                        ball_frames, (lo, hi), args.min_camera_share, frame_step=step, camera=camera)
    doc = {"game_root": str(root.resolve()), "video": game.get("video"), "tag": args.tag, "file": name, "fps": fps,
           "clip_frames": args.clip_frames, "kinds": kinds, "outcomes": sorted(outcomes), "tol_s": args.tol,
           "summary": dict(sorted(Counter(f"{o.name}:{o.outcome}" for o in rows).items())),
           "selected": dict(sorted(Counter(f"{o.name}:{o.outcome}" for o in chosen).items())),
           "skipped": skipped, "camera": camera_frames is not None, "ball_path": ball_frames is not None,
           "clips": clips}
    if step > 1:
        doc["frame_step"] = step
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    for c in clips:
        ev = "; ".join(f"{e['kind']} {e['outcome']} {e['t']:.1f}" for e in c["events"])
        print(f"{c['t']:8.2f}  ball {'-' if c['ball_share'] is None else format(c['ball_share'], '.2f')}  {ev}")
    print(f"{len(errors)} errors {doc['selected']} -> {len(clips)} clips of {args.clip_frames} frames "
          f"(skipped {skipped}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
