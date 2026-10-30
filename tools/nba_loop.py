"""The continuous NBA loop: every game analysed with the current models, scored against its box score and mined for
training data; new models fitted on the games seen so far; a model made current only when it does better on games it
was not fitted on (research use: docs/NBA.md, docs/TRAINING.md §1; the layout is in courtvision/nba/loop.py).

    .venv/bin/python tools/nba_loop.py status [--json]
    .venv/bin/python tools/nba_loop.py next [--game G]
    .venv/bin/python tools/nba_loop.py run [--max-steps N] [--game G]
    .venv/bin/python tools/nba_loop.py train-cues | calibrate-fusion | train-student [--holdout N] [--max-steps N]
    .venv/bin/python tools/nba_loop.py eval [--candidate V] [--game G] [--max-steps N]
    .venv/bin/python tools/nba_loop.py promote V [--force]
    .venv/bin/python tools/nba_loop.py adopt G --det-dir DIR [--student V]

``run`` takes every game's missing stages with the current models; without ``--game`` it then fits the models whose
inputs are new (cue models, fusion config, student) and evaluates every open candidate on every game (``--game``:
that game's stages and the candidates on it). ``next`` prints the step ``run`` would start next and its exact
command, without running it. ``promote`` makes a candidate current only when its mean box error over the games it
was not fitted on is lower than the current models' on the same games (``--force``: only when no game is held out).
``adopt`` declares an existing detection folder the whole-game detections of a student, so it is not detected again.

Steps run one at a time from the repository root; each logs to runs/nba/<game>/logs/ (global steps: runs/nba/logs/).
The runner refuses to start while another CourtVision job runs or with less than --min-free % of memory free, and
a watchdog stops the step's whole process tree when free memory falls below --watch-free %. Re-running resumes.
Exit codes: 0 done, 1 a step failed (the others went on), 2 bad input, 3 refused, 4 stopped by the watchdog,
130 interrupted; promote: 0 promoted, 1 not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from courtvision.nba import loop as nl  # noqa: E402

SCOPES = {"train-cues": "cue", "calibrate-fusion": "fusion", "train-student": "student"}


def default_python() -> str:
    return ".venv/bin/python" if (REPO / ".venv" / "bin" / "python").exists() else sys.executable


def parse(argv: list[str] | None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--runs", default=None, help="the runs folder (default: <repo>/runs)")
    common.add_argument("--python", default=None, help="interpreter of the steps (default: .venv/bin/python)")
    common.add_argument("--min-free", type=float, default=25.0, help="%% of free memory needed to start a step")
    common.add_argument("--watch-free", type=float, default=8.0, help="%% of free memory below which a step is stopped")
    common.add_argument("--poll", type=float, default=5.0, help="watchdog period, s")
    common.add_argument("--holdout", type=int, default=1, help="newest games left out of every fit")
    common.add_argument("--epochs", type=int, default=12, help="train-student --epochs")
    common.add_argument("--from-scratch", action="store_true", help="train-student without --init (current student)")
    common.add_argument("--camera-chunk-s", type=float, default=1800.0, help="video seconds per camera-tracking step")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("status", parents=[common], help="every game's stages, the models and the leaderboard")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("next", parents=[common], help="the next step and its command, without running it")
    p.add_argument("--game", default=None)
    p = sub.add_parser("run", parents=[common], help="run the planned steps one at a time")
    p.add_argument("--game", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    for name, kind in SCOPES.items():
        p = sub.add_parser(name, parents=[common], help=f"fit a new {kind} version on every game but the held-out ones")
        p.add_argument("--max-steps", type=int, default=None)
    p = sub.add_parser("eval", parents=[common], help="analyse every game with each open candidate")
    p.add_argument("--candidate", default=None)
    p.add_argument("--game", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p = sub.add_parser("promote", parents=[common], help="make a candidate current if it is better on held-out games")
    p.add_argument("version")
    p.add_argument("--force", action="store_true", help="promote although no game is held out (a single game)")
    p = sub.add_parser("adopt", parents=[common], help="declare an existing detection folder a student's")
    p.add_argument("game")
    p.add_argument("--det-dir", required=True)
    p.add_argument("--student", default=None, help="default: the current student")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    cfg = nl.LoopConfig(python=args.python or default_python(), min_free=args.min_free, watch_free=args.watch_free,
                        poll_s=args.poll, camera_chunk_s=args.camera_chunk_s, epochs=args.epochs,
                        holdout=args.holdout, init_student=not args.from_scratch)
    runs = Path(args.runs).resolve() if args.runs else REPO / "runs"
    try:
        loop = nl.Loop(runs, REPO, cfg)
        if args.cmd == "status":
            print(json.dumps(nl.status(loop), indent=1, default=str) if args.json else nl.format_status(loop))
            return 0
        if args.cmd == "next":
            print(nl.format_next(loop, game=args.game))
            return 0
        if args.cmd == "run":
            return nl.run_loop(loop, game=args.game, max_steps=args.max_steps)
        if args.cmd in SCOPES:
            return nl.run_loop(loop, scope=SCOPES[args.cmd], max_steps=args.max_steps)
        if args.cmd == "eval":
            return nl.run_loop(loop, scope="eval", candidate=args.candidate, game=args.game, max_steps=args.max_steps)
        if args.cmd == "promote":
            d = nl.promote(loop, args.version, force=args.force)
            print(f"{d['candidate']} ({d['kind']}): {d['outcome']}{' (forced)' if d['forced'] else ''}: {d['reason']}")
            for gid, p in (d.get("per_game") or (d.get("in_sample") or {}).get("per_game") or {}).items():
                print(f"  {p['game']} ({gid}): box error {p['candidate']} vs {p['current']} (current)")
            if d.get("note"):
                print(f"  note: {d['note']}")
            print(f"  recorded in {loop.rel(loop.registry.path)}")
            return 0 if d["promoted"] else 1
        if args.cmd == "adopt":
            stamp = nl.adopt_detections(loop, args.game, args.det_dir, args.student)
            print(f"{args.game}/{args.det_dir}: the detections of student {stamp['student']} ({stamp['weights']})")
            return 0
    except (KeyError, ValueError) as exc:
        print(f"nba_loop: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
