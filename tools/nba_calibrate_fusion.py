"""Calibrate the scorebug-fusion settings (``FusionConfig``) on games with ESPN truth (research use; docs/NBA.md).

    .venv/bin/python tools/nba_calibrate_fusion.py runs/nba/<game> [runs/nba/<game2> ...] --tag q1 \\
        --out runs/nba/fusion_q1.json [--params score_lag_s,make_window_s,rim_misses] [--budget 600] [--workers 4]
    .venv/bin/python tools/nba_analyze.py runs/nba/<game> fuse --tag q1 --fusion-file runs/nba/fusion_q1.json

Each game folder needs ``runs/<tag>/run.json`` and ``events.jsonl`` (``nba_analyze.py events``), ``scorebug.jsonl``,
``camera_parts/``, ``espn_summary.json`` and, when there is one, ``ball_track.json``; no video is read. Each game is
loaded once, as ``nba_fusion_ab.py`` loads it; a setting then costs one ``fuse`` + ``compare`` per game, scored by
``--workers`` processes (default: up to 4; the result does not depend on it).

The search (``courtvision/nba/fusion_tune.py``: the objective, the parameter ranges and the guards against
overfitting) starts from the defaults and keeps a change only when it lowers the pooled objective (box error plus a
small event-F1 term, over all the games) by ``--margin``. With two or more games each game is also held out in
turn. With one game the result is in-sample.

``--out`` gets ``config`` (the FusionConfig overrides: ``nba_analyze.py fuse --fusion-file`` applies them),
``games``, ``tag``, ``objective``, ``before`` / ``after`` (box error and event F1 per game and summed, with the
defaults / with ``config``), ``leave_one_out`` (game -> its box error with the setting tuned on the other games;
empty with one game), ``search`` (space, moves, held-out details, timings) and ``metrics`` (the headline numbers).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from courtvision.geometry.court import court_for  # noqa: E402
from courtvision.nba import espn  # noqa: E402
from courtvision.nba.analyze import BallTrack, load_scorebug  # noqa: E402
from courtvision.nba.fusion import PlayDirection  # noqa: E402
from courtvision.nba.fusion_tune import (  # noqa: E402
    BUDGET,
    F1_WEIGHT,
    MARGIN,
    MAX_ROUNDS,
    GameCase,
    calibrate,
    resolve,
)
from courtvision.nba.scorebug_events import scorebug_events  # noqa: E402
from courtvision.types import Event  # noqa: E402
from nba_analyze import load_game, truth  # noqa: E402


def load_case(root: Path, tag: str, name: str | None = None, tol_s: float = 2.5) -> GameCase:
    """A game's fuse inputs and truth, read as ``nba_fusion_ab.py`` / ``nba_analyze.py fuse`` / ``eval`` read them."""
    out = root / "runs" / tag
    run = json.loads((out / "run.json").read_text())
    window = None
    if run.get("start") or run.get("end"):
        window = (float(run.get("start") or 0.0), float(run.get("end") or 1e9))
    lo, hi = window if window is not None else (0.0, 1e9)
    _, camera = load_game(root)
    reads = load_scorebug(root / "scorebug.jsonl")
    bug = [b for b in scorebug_events(reads) if lo <= b.t <= hi]
    periods = [(r.t, r.period) for r in reads if r.period is not None]
    with (out / "events.jsonl").open() as fh:
        events = [Event.from_dict(json.loads(line)) for line in fh if line.strip()]
    flights = BallTrack.load(root / "ball_track.json").flights if (root / "ball_track.json").exists() else []
    summary, aligned = truth(root)
    return GameCase(name or root.name, events, bug, aligned.aligned, espn.box_score(summary), window,
                    PlayDirection.from_camera(camera), periods, flights, court_for("nba"), tol_s)


def game_names(roots: list[Path]) -> list[str]:
    """The folder names, or the paths as given where two folders share a name."""
    names = [r.name for r in roots]
    return [str(r) if names.count(r.name) > 1 else r.name for r in roots]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path, help="game folders (runs/nba/<game>)")
    ap.add_argument("--tag", required=True, help="the run of each game (runs/<tag>/events.jsonl)")
    ap.add_argument("--out", required=True, type=Path, help="calibration JSON to write")
    ap.add_argument("--params", default=None,
                    help="comma-separated coordinates to search (default: all; a window field such as make_window_s "
                         "means both of its ends, make_window_s.lo / .hi one)")
    ap.add_argument("--budget", type=int, default=BUDGET, help="settings each search may score")
    ap.add_argument("--margin", type=float, default=MARGIN, help="objective gain a change must bring")
    ap.add_argument("--f1-weight", type=float, default=F1_WEIGHT, help="weight of (1 - event F1) in the objective")
    ap.add_argument("--rounds", type=int, default=MAX_ROUNDS, help="coordinate-descent rounds at most")
    ap.add_argument("--workers", type=int, default=0,
                    help="processes scoring settings in parallel (default 0: min(4, CPU count); same result)")
    ap.add_argument("--tol", type=float, default=2.5, help="event matching tolerance (s), as nba_analyze.py eval")
    ap.add_argument("--loo-guard", action="store_true",
                    help="keep the defaults when the held-out games do worse with the settings tuned without them")
    args = ap.parse_args(argv)

    def say(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    params = args.params.split(",") if args.params else None
    try:
        resolve(params)  # a misspelt name fails before the games load
    except ValueError as exc:
        ap.error(str(exc))
    names = game_names(args.roots)
    cases = []
    for root, name in zip(args.roots, names, strict=True):
        cases.append(load_case(root, args.tag, name, args.tol))
        c = cases[-1]
        say(f"{name}: {len(c.events)} engine events, {len(c.bug)} scorebug events, window {c.window}")
    workers = args.workers if args.workers > 0 else min(4, os.cpu_count() or 1)
    rep = calibrate(cases, params, budget=args.budget, margin=args.margin, f1_weight=args.f1_weight,
                    max_rounds=args.rounds, loo_guard=args.loo_guard, workers=workers, log=say)
    rep["search"]["roots"] = {n: str(r) for n, r in zip(names, args.roots, strict=True)}
    out = {"config": rep["config"], "games": rep["games"], "tag": args.tag, **{k: v for k, v in rep.items()
                                                                               if k not in ("config", "games")}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")

    b, a, loo = rep["before"], rep["after"], rep["leave_one_out"]
    f = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
    print("| game | box error, defaults | calibrated | held out | event F1, defaults | calibrated |")
    print("|---|---:|---:|---:|---:|---:|")
    for n in names:
        print(f"| {n} | {f(b['per_game'][n]['box_error'])} | {f(a['per_game'][n]['box_error'])} | {f(loo.get(n))} | "
              f"{f(b['per_game'][n]['event_f1'])} | {f(a['per_game'][n]['event_f1'])} |")
    print(f"| all | {f(b['summed']['box_error'])} | {f(a['summed']['box_error'])} | "
          f"{f(rep['search']['leave_one_out']['summed']['held_out']['box_error']) if loo else '-'} | "
          f"{f(b['summed']['event_f1'])} | {f(a['summed']['event_f1'])} |")
    print(f"\nconfig: {json.dumps(rep['config'])}")
    s = rep["search"]
    if s["leave_one_out"]:
        print(s["leave_one_out"]["verdict"])
    if s["loo_guard"]["applied"]:
        print(f"leave-one-out guard: kept the defaults instead of {json.dumps(s['tuned_config'])}")
    print(f"note: {s['note']}")
    print(f"{s['evaluations']} settings in the search, {s['scored']} fuse + compare runs, {s['seconds']:.0f} s "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
