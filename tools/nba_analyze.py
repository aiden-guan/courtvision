"""Analyse a whole NBA broadcast and score it against the official box score (research use).

    .venv/bin/python tools/nba_analyze.py runs/nba/<game> detect [--workers 2]
    .venv/bin/python tools/nba_analyze.py runs/nba/<game> ball
    .venv/bin/python tools/nba_analyze.py runs/nba/<game> events [--tag baseline] [--3d] [--start S --end S]
    .venv/bin/python tools/nba_analyze.py runs/nba/<game> eval --tag baseline

The game folder holds ``game.json`` (``video``, ``espn_event``, optional ``home_color_bgr`` / ``away_color_bgr``),
``scorebug.jsonl`` (tools/nba_scan_scorebug.py), ``camera_parts/*.jsonl`` (tools/nba_track_court.py) and
``espn_summary.json``. ``detect`` writes ``detections/*.npz`` (``--det-dir``: another folder, e.g. for new
weights; ``ball`` and ``events`` read the same one), ``ball`` writes ``ball_track.json``, ``events`` writes
``runs/<tag>/events.jsonl`` and ``eval`` writes ``runs/<tag>/report.json`` / ``report.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from courtvision.nba import espn  # noqa: E402
from courtvision.nba.analyze import (  # noqa: E402
    BallTrack,
    CameraTrack,
    DetectionCache,
    PerceiveConfig,
    build_ball_track,
    detect_video,
    load_scorebug,
    perceive,
    run_engine,
)
from courtvision.nba.evaluate import compare  # noqa: E402
from courtvision.training.align_pbp import AlignConfig, ClockTimeline, align_pbp, score_changes  # noqa: E402

# ESPN's scorebug shows a basket about 2 s after the play-by-play clock time of the make (median over the made
# field goals of Lakers @ Warriors 2024-12-25); its play-by-play clock is the event's own time (no entry lag)
ESPN_ALIGN = AlignConfig(score_update_lag_s=2.0, lag_s={"nba": 0.0}, estimate_lag=False)
from courtvision.types import Event, Team  # noqa: E402


def load_game(root: Path) -> tuple[dict, CameraTrack]:
    game = json.loads((root / "game.json").read_text())
    camera = CameraTrack.load(sorted((root / "camera_parts").glob("part*.jsonl")))
    return game, camera


def cmd_detect(root: Path, args: argparse.Namespace) -> None:
    import multiprocessing as mp

    game, camera = load_game(root)
    out = root / args.det_dir
    out.mkdir(exist_ok=True)
    frames = sorted(camera.params)
    lo = max(frames[0], int(args.start * camera.fps)) if args.start else frames[0]
    hi = min(frames[-1] + 1, int(args.end * camera.fps)) if args.end else frames[-1] + 1
    n = max(args.workers, args.chunks)  # chunks already written are skipped, so an interrupted run resumes
    bounds = [lo + (hi - lo) * k // n for k in range(n + 1)]
    jobs = [(str(root), game["video"], args.weights, bounds[k], bounds[k + 1], str(out / f"{bounds[k]:07d}-{bounds[k + 1]:07d}.npz"))
            for k in range(n) if not (out / f"{bounds[k]:07d}-{bounds[k + 1]:07d}.npz").exists()]
    if args.workers == 1:
        for j in jobs:
            _detect_part(*j)
    else:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            pool.starmap(_detect_part, jobs)


def _detect_part(root: str, video: str, weights: str, start: int, end: int, out: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    _, camera = load_game(Path(root))
    t0 = time.time()
    cache = detect_video(Path(video), camera, Path(weights), start, end)
    cache.save(Path(out))
    print(f"{out}: {len(cache.balls)} frames in {time.time() - t0:.0f}s", flush=True)


def load_cache(root: Path, det_dir: str = "detections") -> DetectionCache:
    cache = DetectionCache()
    for p in sorted((root / det_dir).glob("*.npz")):
        cache.merge(DetectionCache.load(p))
    return cache


def cmd_ball(root: Path, args: argparse.Namespace) -> None:
    _, camera = load_game(root)
    t0 = time.time()
    track = build_ball_track(load_cache(root, args.det_dir), camera)
    track.save(root / "ball_track.json")
    print(f"ball path {len(track.path)} frames, {len(track.flights)} flights ({time.time() - t0:.0f}s)")


def cmd_events(root: Path, args: argparse.Namespace) -> None:
    game, camera = load_game(root)
    cache = load_cache(root, args.det_dir)
    ball = BallTrack.load(root / "ball_track.json")
    reads = load_scorebug(root / "scorebug.jsonl")
    cfg = PerceiveConfig(use_3d_flights=args.use_3d, home_color_bgr=tuple(game["home_color_bgr"]) if game.get("home_color_bgr") else None,
                         away_color_bgr=tuple(game["away_color_bgr"]) if game.get("away_color_bgr") else None)
    start = int(args.start * camera.fps) if args.start else 0
    end = int(args.end * camera.fps) if args.end else None
    t0 = time.time()
    frames = perceive(Path(game["video"]), camera, cache, ball, reads, cfg, start, end)
    events = run_engine(frames, config=json.loads(args.engine) if args.engine else None)
    out = root / "runs" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    with (out / "events.jsonl").open("w") as fh:
        for e in events:
            fh.write(json.dumps(e.to_dict()) + "\n")
    (out / "run.json").write_text(json.dumps({"start": args.start, "end": args.end, "use_3d": args.use_3d,
                                              "engine": args.engine, "seconds": time.time() - t0}))
    print(f"{len(events)} events in {time.time() - t0:.0f}s -> {out}")


def truth(root: Path):  # noqa: ANN201
    summary = espn.load_summary(root / "espn_summary.json")
    reads = load_scorebug(root / "scorebug.jsonl")
    tl = ClockTimeline.fit(reads)
    return summary, align_pbp(espn.pbp_actions(summary), tl, score_changes=score_changes(reads), config=ESPN_ALIGN)


def cmd_fuse(root: Path, args: argparse.Namespace) -> None:
    from courtvision.nba.fusion import FusionConfig, PlayDirection, fuse
    from courtvision.nba.scorebug_events import scorebug_events

    _, camera = load_game(root)
    out = root / "runs" / args.tag
    run = json.loads((out / "run.json").read_text())
    lo, hi = float(run.get("start") or 0.0), float(run.get("end") or 1e9)
    reads = load_scorebug(root / "scorebug.jsonl")
    bug = [b for b in scorebug_events(reads) if lo <= b.t <= hi]
    periods = [(r.t, r.period) for r in reads if r.period is not None]
    events = [Event.from_dict(json.loads(line)) for line in (out / "events.jsonl").open()]
    cfg = FusionConfig(**(json.loads(args.fusion) if args.fusion else {}))
    flights = BallTrack.load(root / "ball_track.json").flights if (root / "ball_track.json").exists() else []
    from courtvision.geometry.court import court_for

    fused = fuse(events, bug, cfg, PlayDirection.from_camera(camera), periods, flights, court_for("nba"))
    with (out / "fused.jsonl").open("w") as fh:
        for e in fused:
            fh.write(json.dumps(e.to_dict()) + "\n")
    print(f"{len(events)} engine events -> {len(fused)} fused events")


def nba_checks(root: Path, events: list[Event], window: tuple[float, float] | None) -> dict:
    """Extra checks against NBA.com data when the game folder has it: shot locations (``nba_pbp.csv``) and
    passes per team (``nba_tracking.json``, Second Spectrum)."""
    from courtvision.eval.events import latest_revisions
    from courtvision.nba.nba_stats import load_shots, load_tracking, match_shot_locations
    from courtvision.types import EventKind, EventStatus

    out: dict = {}
    evs = [e for e in latest_revisions(events) if e.status is not EventStatus.RETRACTED]
    lo, hi = window if window is not None else (0.0, 1e9)
    if (root / "nba_pbp.csv").exists():
        tl = ClockTimeline.fit(load_scorebug(root / "scorebug.jsonl"))
        placed = []
        for s in load_shots(root / "nba_pbp.csv"):
            t = tl.video_t(s.period, s.clock)
            if t is not None and lo <= t <= hi:
                placed.append((t, s))
        att = [e for e in evs if e.kind is EventKind.SHOT_ATTEMPT and e.value in (2, 3)]  # the release point
        out["shot_locations"] = match_shot_locations(att, placed, tol_s=3.0)
    if (root / "nba_tracking.json").exists() and window is None:
        trk = load_tracking(root / "nba_tracking.json")
        passes = {t.value: sum(1 for e in evs if e.kind is EventKind.PASS and e.team is t) for t in (Team.HOME, Team.AWAY)}
        out["passes"] = {"pred": passes, "nba": {"home": trk["home"]["totals"]["passes"],
                                                 "away": trk["away"]["totals"]["passes"]}}
    return out


def cmd_eval(root: Path, args: argparse.Namespace) -> None:
    summary, aligned = truth(root)
    out = root / "runs" / args.tag
    run = json.loads((out / "run.json").read_text())
    events = [Event.from_dict(json.loads(line)) for line in (out / args.file).open()]
    window = None
    if run.get("start") or run.get("end"):
        window = (float(run.get("start") or 0.0), float(run.get("end") or 1e9))
    rep = compare(events, aligned.aligned, espn.box_score(summary), tol_s=args.tol, window=window)
    rep.update(nba_checks(root, events, window))
    stem = Path(args.file).stem
    (out / f"report_{stem}.json").write_text(json.dumps(rep, indent=1))
    abbr = {v["side"]: v["abbr"] for v in espn.teams(summary).values()}
    lines = [f"# {args.tag} ({stem})", ""]
    if "box" in rep:
        lines += [f"Box-score error (sum |pred - truth| / sum truth): **{rep['box']['error']:.3f}**, "
                  f"exact cells {rep['box']['exact_share']:.0%}", ""]
        lines.append("| stat | " + " | ".join(f"{abbr[t]} pred | {abbr[t]} truth" for t in (Team.AWAY, Team.HOME)) + " |")
        lines.append("|---|---:|---:|---:|---:|")
        by = {(r["team"], r["stat"]): r for r in rep["box"]["rows"]}
        for s in ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov"):
            lines.append(f"| {s} | " + " | ".join(f"{by[(t.value, s)]['pred']} | {by[(t.value, s)]['truth']}"
                                                for t in (Team.AWAY, Team.HOME)) + " |")
    else:
        lines.append(f"Window {window}: " + json.dumps(rep["box_window"]))
    if "shot_locations" in rep:
        lines.append("")
        lines.append(f"Shot locations vs NBA.com: {json.dumps(rep['shot_locations'])}")
    if "passes" in rep:
        lines.append(f"Passes vs NBA tracking: {json.dumps(rep['passes'])}")
    lines += ["", "| event | F1 | precision | recall | pred | truth |", "|---|---:|---:|---:|---:|---:|"]
    for k, r in rep["events"].items():
        f = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        lines.append(f"| {k} | {f(r['f1'])} | {f(r['precision'])} | {f(r['recall'])} | {r['n_pred']} | {r['n_truth']} |")
    (out / f"report_{stem}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("cmd", choices=("detect", "ball", "events", "fuse", "eval"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=1, help="detect: split the range into this many files (resumable)")
    ap.add_argument("--det-dir", default="detections", help="detect / ball / events: detection folder in the game folder")
    ap.add_argument("--weights", default="runs/models/student/best.pt")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--3d", dest="use_3d", action="store_true")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    ap.add_argument("--engine", default=None, help="EventEngineConfig overrides (JSON)")
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--file", default="events.jsonl", help="eval: events file of the run (events.jsonl / fused.jsonl)")
    ap.add_argument("--fusion", default=None, help="fuse: FusionConfig overrides (JSON)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    root = Path(args.root)
    {"detect": cmd_detect, "ball": cmd_ball, "events": cmd_events, "fuse": cmd_fuse, "eval": cmd_eval}[args.cmd](root, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
