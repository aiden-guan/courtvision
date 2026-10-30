"""Measure each scorebug-fusion pass on a game: the box score and event F1 with every switch of ``FusionConfig``
flipped on its own, next to the current defaults and the engine alone (research use; docs/NBA.md).

    .venv/bin/python tools/nba_fusion_ab.py runs/nba/<game> --tag q1 [--set '{"bug_rebound_window_s": 6}']

Needs the run's ``events.jsonl`` (``nba_analyze.py events``) and the game's truth files; no video is read, so it
takes seconds. Box error is sum |pred - truth| / sum truth over the box stats of both teams (in the run's window
when it has one). Writes ``runs/<tag>/fusion_ab.json`` and prints a table.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from courtvision.geometry.court import court_for  # noqa: E402
from courtvision.nba import espn  # noqa: E402
from courtvision.nba.analyze import BallTrack, load_scorebug  # noqa: E402
from courtvision.nba.evaluate import BOX_STATS, compare  # noqa: E402
from courtvision.nba.fusion import FusionConfig, PlayDirection, fuse  # noqa: E402
from courtvision.nba.scorebug_events import scorebug_events  # noqa: E402
from courtvision.types import Event, Team  # noqa: E402
from nba_analyze import load_game, truth  # noqa: E402

SHOWN = ("fga", "fg3a", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov")


def box_rows(rep: dict) -> dict[tuple[str, str], tuple[int, int]]:
    """(team, stat) -> (pred, truth), from a whole-game or a windowed report."""
    if "box" in rep:
        return {(r["team"], r["stat"]): (r["pred"], r["truth"]) for r in rep["box"]["rows"]}
    w = rep["box_window"]
    return {(t, s): (int(w["pred"][t].get(s, 0)), int(w["truth"][t].get(s, 0)))
            for t in (Team.HOME.value, Team.AWAY.value) for s in BOX_STATS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", default=None, help="FusionConfig overrides applied to every variant (JSON)")
    ap.add_argument("--tol", type=float, default=2.5)
    args = ap.parse_args()
    root = Path(args.root)
    out = root / "runs" / args.tag
    run = json.loads((out / "run.json").read_text())
    window = None
    if run.get("start") or run.get("end"):
        window = (float(run.get("start") or 0.0), float(run.get("end") or 1e9))
    lo, hi = window if window is not None else (0.0, 1e9)
    _, camera = load_game(root)
    reads = load_scorebug(root / "scorebug.jsonl")
    bug = [b for b in scorebug_events(reads) if lo <= b.t <= hi]
    periods = [(r.t, r.period) for r in reads if r.period is not None]
    events = [Event.from_dict(json.loads(line)) for line in (out / "events.jsonl").open()]
    flights = BallTrack.load(root / "ball_track.json").flights if (root / "ball_track.json").exists() else []
    direction = PlayDirection.from_camera(camera)
    court = court_for("nba")
    summary, aligned = truth(root)
    box = espn.box_score(summary)
    base = json.loads(args.set) if args.set else {}
    flags = [f.name for f in dataclasses.fields(FusionConfig) if f.type in (bool, "bool")]
    variants: list[tuple[str, dict | None]] = [("engine only", None), ("defaults", {})]
    for name in flags:
        default = getattr(FusionConfig(**base), name)
        variants.append((f"{name}={'off' if default else 'on'}", {name: not default}))
    results = []
    for label, change in variants:
        if change is None:
            evs = events
        else:
            evs = fuse(events, bug, FusionConfig(**{**base, **change}), direction, periods, flights, court)
        rep = compare(evs, aligned.aligned, box, tol_s=args.tol, window=window)
        rows = box_rows(rep)
        err = sum(abs(p - q) for p, q in rows.values()) / max(1, sum(q for _, q in rows.values()))
        results.append({"variant": label, "box_error": round(err, 4),
                        "diff": {f"{t}.{s}": p - q for (t, s), (p, q) in sorted(rows.items())},
                        "f1": {k: r["f1"] for k, r in rep["events"].items()}})
    (out / "fusion_ab.json").write_text(json.dumps(results, indent=1))
    teams = (Team.AWAY.value, Team.HOME.value)
    head = "| variant | box error | " + " | ".join(f"{s} a/h" for s in SHOWN) + " | shot F1 | rebound F1 |"
    print(head)
    print("|---|---:|" + "---:|" * (len(SHOWN) + 2))
    for r in results:
        cells = [f"{r['diff'][f'{teams[0]}.{s}']:+d}/{r['diff'][f'{teams[1]}.{s}']:+d}" for s in SHOWN]
        f1 = r["f1"]
        shot = f1.get("shot_attempt")
        reb = f1.get("rebound")
        fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        print(f"| {r['variant']} | {r['box_error']:.3f} | " + " | ".join(cells) + f" | {fmt(shot)} | {fmt(reb)} |")
    print(f"\n(diff = pred - truth, away/home; {out / 'fusion_ab.json'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
