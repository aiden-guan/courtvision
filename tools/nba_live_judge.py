"""Fit the shot judge's model (``courtvision/nba/live_judge.py``) on analysed games, and check it on held-out data
(research use).

    .venv/bin/python tools/nba_live_judge.py --games 2024-12-25-LAL-GSW[,2017-06-12-CLE-GSW] --tag <run>
        [--det-dir detections] [--out runs/models/rim/live_judge.json] [--min-score 0.08]
        [--rim NAME --rim-model runs/models/rim/<model>.pt]

For every shot result of the engine in the run (``runs/<tag>/events.jsonl``) that the play-by-play matches to a real
shot (``makes.label_candidates``), the judge's evidence as it would see it live: the engine's rim cues and where the
detector's ball candidates (scores of at least ``--min-score``, the Studio detector's threshold) were about the rim
from 1 s before to 1.4 s after the linked ball's pass nearest it (``makes.zone_features``). A logistic model of made or
missed over those (``live_judge.JUDGE_FEATURES``) is scored held out (each game fitted on one half and scored on the
other; with two or more games, also each game scored by a model fitted on the others) beside the engine's own cue
"crossed inside the rim", then fitted on everything and written to ``--out``. A game folder's detections may be in
another folder than ``detections`` (``--det-dir``; the first that exists of ``--det-dir`` and ``detections-full`` is
used).

``--rim NAME`` adds the rim model's read of each shot's rim clip (``live_judge.CNN_FEATURES``) from its held-out
predictions ``<game>/rim_clips/oof_NAME.jsonl`` (``tools/nba_rim_model.py``: fitted without the game, or without its
half); ``--rim-model`` names the weights the judge then runs in the Studio (fitted on every clip: ``--save``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from courtvision.eval.events import latest_revisions  # noqa: E402
from courtvision.geometry.court import court_for  # noqa: E402
from courtvision.nba import makes as mk  # noqa: E402
from courtvision.nba.analyze import BallTrack  # noqa: E402
from courtvision.nba.live_judge import CNN_FEATURES, JUDGE_FEATURES, cue_features  # noqa: E402
from courtvision.nba.rim_clips import ball_zone_series  # noqa: E402
from courtvision.nba.rim_jev import rim_centre  # noqa: E402
from courtvision.types import Event, EventStatus  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nba_analyze import load_cache, load_game, truth  # noqa: E402
from nba_rim_clips import truth_shots  # noqa: E402

NBA = Path("runs/nba")


def game_rows(root: Path, tag: str, det_dir: str, min_score: float,
              rim: str | None = None) -> list[tuple[dict, bool, float]]:
    """(features, made, time) of each engine result the play-by-play matches to a real shot; ``rim``: the name of the
    rim model's held-out predictions to read ``CNN_FEATURES`` from."""
    oof = {}
    if rim:
        for line in (root / "rim_clips" / f"oof_{rim}.jsonl").open():
            r = json.loads(line)
            oof[r["key"]] = r
    _, camera = load_game(root)
    court = court_for("nba")
    folder = next((d for d in (det_dir, "detections-full", "detections") if (root / d).is_dir()), det_dir)
    cache = load_cache(root, folder)
    ball = BallTrack.load(root / "ball_track.json")
    path = {f: (p[0], p[1]) for f, p in ball.path.items()}
    events = [Event.from_dict(json.loads(line)) for line in (root / "runs" / tag / "events.jsonl").open()]
    events = [e for e in latest_revisions(events) if e.status is not EventStatus.RETRACTED]
    shots = truth_shots(truth(root)[1])
    cands = [c for c in mk.candidates(events, mk.rim_visits(ball.path, camera, court)) if c.result is not None]
    labels = mk.label_candidates(cands, [(s["t"], s["ft"], s["made"]) for s in shots])
    rows = []
    for c in cands:
        lab = labels[c.key]
        if lab == "false":
            continue
        found = rim_centre(c.t, camera, court, path, side=c.side)
        series = ball_zone_series(cache.balls, camera, court, found[0], found[1], min_score=min_score) \
            if found is not None else None
        r = c.result
        f = {**cue_features(r), **mk.zone_features(series),
             "is_ft": float(r.value == 1 or r.attrs.get("shot_type") == "free_throw")}
        o = oof.get(c.key)
        f.update(cnn_made=o["p_made"] / max(o["p_made"] + o["p_missed"], 1e-9) if o else 0.5,
                 cnn_missing=0.0 if o else 1.0)
        rows.append((f, lab.endswith("made"), c.t))
    return rows


def score(rows: list[tuple[dict, bool, float]], model: mk.Logistic) -> float:
    return float(np.mean([(model.p(f) >= 0.5) == made for f, made, _ in rows])) if rows else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--det-dir", default="detections")
    ap.add_argument("--min-score", type=float, default=0.08)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--out", default="runs/models/rim/live_judge.json")
    ap.add_argument("--rim", default=None, help="the rim model's held-out predictions to read (oof_<name>.jsonl)")
    ap.add_argument("--rim-model", default=None, help="the rim model's weights the judge runs (with --rim)")
    args = ap.parse_args()
    games = [g.strip() for g in args.games.split(",") if g.strip()]
    data = {g: game_rows(NBA / g, args.tag, args.det_dir, args.min_score, args.rim) for g in games}
    names = JUDGE_FEATURES + (CNN_FEATURES if args.rim else ())
    report: dict = {}
    for g, rows in data.items():
        mid = float(np.median([t for _, _, t in rows]))
        right, crossed = [], []
        for first in (True, False):
            train = [r for r in rows if (r[2] < mid) != first]
            test = [r for r in rows if (r[2] < mid) == first]
            m = mk.Logistic.fit([f for f, _, _ in train], [y for _, y, _ in train], names, args.l2)
            right += [(m.p(f) >= 0.5) == y for f, y, _ in test]
            crossed += [bool(f["crossed_inside"]) == y for f, y, _ in test]
        report[g] = {"shots": len(rows), "made": int(sum(y for _, y, _ in rows)),
                     "halves": round(float(np.mean(right)), 4), "engine_crossed_inside": round(float(np.mean(crossed)), 4)}
        if len(games) > 1:
            others = [r for h, rs in data.items() if h != g for r in rs]
            m = mk.Logistic.fit([f for f, _, _ in others], [y for _, y, _ in others], names, args.l2)
            report[g]["other_games"] = round(score(rows, m), 4)
    allrows = [r for rs in data.values() for r in rs]
    model = mk.Logistic.fit([f for f, _, _ in allrows], [y for _, y, _ in allrows], names, args.l2)
    doc = {"made": model.to_dict(), "min_score": args.min_score, "fitted_on": games, "tag": args.tag,
           "shots": len(allrows), "held_out": report}
    if args.rim and args.rim_model:
        doc["rim_model"] = str(Path(args.rim_model).resolve())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps(report, indent=1))
    print(f"fitted on {len(allrows)} shots of {len(games)} game(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
