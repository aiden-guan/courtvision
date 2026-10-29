"""Where an NBA analysis goes wrong: true shots, rebounds and turnovers next to the predicted ones, in time order.

    .venv/bin/python tools/nba_diagnose.py runs/nba/<game> --tag <run> [--fused] [--kinds shot,rebound]

Each true event is matched to the nearest predicted event of the same kind within ``--tol`` s (greedy, in time
order) and printed as ``ok`` / ``team`` (wrong team) / ``miss``; unmatched predictions are printed as ``extra``.
A summary counts each outcome per kind.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from courtvision.eval.events import latest_revisions  # noqa: E402
from courtvision.nba.evaluate import truth_events  # noqa: E402
from courtvision.types import Event, EventKind, EventStatus  # noqa: E402
from nba_analyze import truth  # noqa: E402

KINDS = {"shot": (EventKind.SHOT_MADE, EventKind.SHOT_MISSED), "rebound": (EventKind.REBOUND,),
         "turnover": (EventKind.TURNOVER,), "assist": (EventKind.ASSIST,), "steal": (EventKind.STEAL,)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--file", default="events.jsonl")
    ap.add_argument("--kinds", default="shot,rebound,turnover")
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    run = json.loads((root / "runs" / args.tag / "run.json").read_text())
    lo, hi = float(run.get("start") or 0.0), float(run.get("end") or 1e9)
    _, aligned = truth(root)
    tev = [e for e in truth_events(aligned.aligned) if lo <= e.t <= hi]
    pev = [e for e in latest_revisions(Event.from_dict(json.loads(line))
                                       for line in (root / "runs" / args.tag / args.file).open())
           if e.status is not EventStatus.RETRACTED and lo <= e.t <= hi]
    summary: Counter[str] = Counter()
    rows = []
    for name in args.kinds.split(","):
        kinds = KINDS[name]
        T = sorted((e for e in tev if e.kind in kinds), key=lambda e: e.t)
        P = sorted((e for e in pev if e.kind in kinds), key=lambda e: e.t)
        used: set[int] = set()
        for e in T:
            best = None
            for k, p in enumerate(P):
                if k in used or abs(p.t - e.t) > args.tol:
                    continue
                if best is None or abs(p.t - e.t) < abs(P[best].t - e.t):
                    best = k
            if best is None:
                outcome = "miss"
            else:
                used.add(best)
                p = P[best]
                outcome = "ok" if p.team is e.team else "team"
                if name == "shot" and outcome == "ok" and (p.kind is not e.kind or p.value != e.value):
                    outcome = "result" if p.kind is not e.kind else "value"
            summary[f"{name}:{outcome}"] += 1
            rows.append((e.t, name, outcome, _fmt(e), _fmt(P[best]) if best is not None else ""))
        for k, p in enumerate(P):
            if k not in used:
                summary[f"{name}:extra"] += 1
                rows.append((p.t, name, "extra", "", _fmt(p)))
    if not args.quiet:
        for t, name, outcome, a, b in sorted(rows):
            print(f"{t:8.2f} {name:8s} {outcome:6s} | {a:38s} | {b}")
    print(dict(sorted(summary.items())))
    return 0


def _fmt(e: Event) -> str:
    v = f" {e.value}" if e.kind in (EventKind.SHOT_MADE, EventKind.SHOT_MISSED) else ""
    off = " off" if e.kind is EventKind.REBOUND and e.attrs.get("offensive") else ""
    return f"{e.t:7.2f} {e.kind.value}{v}{off} {e.team.value if e.team else '-'} {e.player_id or ''}"


if __name__ == "__main__":
    raise SystemExit(main())
