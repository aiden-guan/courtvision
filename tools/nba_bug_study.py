"""How the scorebug's shot clock lines up with the real play: next to each shot-clock event (``reset24``,
``reset14``, ``run24``; ``nba/scorebug_events.py``), what the aligned play-by-play says happened around it, so the
fusion rules built on them (``nba/fusion.py``: rim misses, shot-clock rebounds, offensive-rebound misses) can be
checked on a real game (research use; docs/NBA.md).

    .venv/bin/python tools/nba_bug_study.py runs/nba/<game> [--start S --end S]

For each kind (``reset24`` split by whether the game clock runs, ``reset14`` likewise, ``run24``) it counts the true
action nearest to the event in a window: missed / made field goal, missed / made free throw, offensive / defensive /
team rebound, steal, turnover, block, foul, or none. It also gives the time from a running ``reset24`` to the next
``run24`` for each of the true causes (a rim touch then a defensive rebound waits for the rebound; a steal starts
at once). Writes ``bug_study.json`` in the game folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from courtvision.nba.analyze import load_scorebug  # noqa: E402
from courtvision.nba.scorebug_events import BugEvent, scorebug_events  # noqa: E402
from courtvision.types import EventKind  # noqa: E402
from nba_analyze import truth  # noqa: E402

# where to look for each kind's cause, relative to the bug event (s): the play-by-play times are about 1-2 s off
WINDOWS = {"reset24": (-4.0, 1.5), "reset14": (-4.0, 1.5), "run24": (-5.0, 1.0)}


def category(a) -> str | None:  # noqa: ANN001 (align_pbp.PBPAction)
    if a.kind in (EventKind.SHOT_MADE, EventKind.SHOT_MISSED):
        ft = a.is_free_throw
        return f"{'ft' if ft else 'fg'}_{'make' if a.made else 'miss'}"
    if a.kind is EventKind.REBOUND:
        if a.player_id is None:
            return "team_reb"
        return "oreb" if a.attrs.get("offensive") else "dreb"
    return {EventKind.STEAL: "steal", EventKind.TURNOVER: "turnover", EventKind.BLOCK: "block",
            EventKind.FOUL: "foul"}.get(a.kind)  # type: ignore[arg-type]


def running_at(bug: list[BugEvent]):  # noqa: ANN201
    clock = sorted((b.t, b.kind == "clock_start") for b in bug if b.kind in ("clock_start", "clock_stop"))
    ts = np.array([c[0] for c in clock])

    def running(t: float) -> bool:
        i = int(np.searchsorted(ts, t, side="right")) - 1
        return i < 0 or clock[i][1]

    return running


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    args = ap.parse_args()
    root = Path(args.root)
    lo, hi = args.start, (args.end or 1e9)
    bug = [b for b in scorebug_events(load_scorebug(root / "scorebug.jsonl")) if lo <= b.t <= hi]
    _, aligned = truth(root)
    acts = sorted(((al.t, category(al.action)) for al in aligned.aligned
                   if al.t is not None and not al.action.synthetic and category(al.action) is not None),
                  key=lambda r: r[0])
    running = running_at(bug)
    counts: dict[str, Counter] = defaultdict(Counter)
    gaps: dict[str, list[float]] = defaultdict(list)
    runs = [b.t for b in bug if b.kind == "run24"]
    for b in bug:
        if b.kind not in WINDOWS:
            continue
        key = b.kind if b.kind == "run24" else f"{b.kind} ({'running' if running(b.t) else 'stopped'})"
        a, z = WINDOWS[b.kind]
        near = [(t, c) for t, c in acts if a <= t - b.t <= z]
        if b.kind == "run24":
            # what gave the ball: the latest change of possession before the count (a rebound, a steal, a make)
            cause = max(near, key=lambda r: r[0])[1] if near else "none"
        else:
            cause = min(near, key=lambda r: abs(r[0] - b.t))[1] if near else "none"
        counts[key][cause] += 1
        if b.kind == "reset24" and running(b.t):
            nxt = next((t for t in runs if 0.0 < t - b.t <= 6.0), None)
            if nxt is not None:
                gaps[cause].append(round(nxt - b.t, 2))
    out = {"counts": {k: dict(v.most_common()) for k, v in counts.items()},
           "reset24_to_run24_s": {k: {"n": len(v), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                                      "p90": float(np.percentile(v, 90))} for k, v in gaps.items() if v}}
    (root / "bug_study.json").write_text(json.dumps(out, indent=1))
    for k, v in sorted(out["counts"].items()):
        n = sum(v.values())
        print(f"{k} ({n}): " + ", ".join(f"{c} {m} ({m / n:.0%})" for c, m in v.items()))
    print("\nrunning reset24 -> next run24 (s), by the nearest true action:")
    for k, v in sorted(out["reset24_to_run24_s"].items()):
        print(f"  {k}: n={v['n']} median {v['median']:.2f} (p10 {v['p10']:.2f}, p90 {v['p90']:.2f})")
    print(f"\n-> {root / 'bug_study.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
