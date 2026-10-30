"""Fit the shot models of ``nba/makes.py`` across games and score each game with models fitted on the others
(research use).

    .venv/bin/python tools/nba_shot_models.py --runs 2024-12-25-LAL-GSW:zr1,2017-06-12-CLE-GSW:zr1 [--zones] [--rim]
        [--out runs/models/shot_models.json]

Each ``GAME:TAG`` names a run of ``nba_analyze.py <game> timeline --tag TAG`` with the play-by-play (it writes every
candidate's features and label to ``runs/<tag>/shot_rows.jsonl``). For each game the three models (a real shot or
not, a free throw or a field goal, made or missed) are fitted on the other games' candidates and scored on this one:
the accuracy a new game, never seen in fitting, gets. Then they are fitted on every game and written to ``--out``,
for games without play-by-play (``nba_analyze.py <game> timeline --fit none --shot-models <out>``) and the Studio.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from courtvision.nba import makes as mk  # noqa: E402

NBA = Path("runs/nba")


def load(game: str, tag: str) -> list[dict]:
    return [json.loads(line) for line in (NBA / game / "runs" / tag / "shot_rows.jsonl").open()]


def score(models: mk.ShotModels, rows: list[dict]) -> dict:
    real = [r for r in rows if r["label"] != "false"]
    ok_real = sum((models.real.p(r["features"]) >= 0.5) == (r["label"] != "false") for r in rows)
    ok_ft = ok_made = 0
    for r in real:
        ft = models.ft.p(r["features"]) >= 0.5
        ok_ft += ft == r["label"].startswith("ft")
        ok_made += (models.made.p({**r["features"], "is_ft": float(ft)}) >= 0.5) == r["label"].endswith("made")
    return {"candidates": len(rows), "real_shots": len(real), "real_accuracy": round(ok_real / max(1, len(rows)), 4),
            "ft_accuracy": round(ok_ft / max(1, len(real)), 4), "made_accuracy": round(ok_made / max(1, len(real)), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="GAME:TAG,GAME:TAG,...")
    ap.add_argument("--zones", action="store_true", help="the made model reads the ball-zone features")
    ap.add_argument("--rim", action="store_true", help="the made model reads the rim model's features")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--out", default="runs/models/shot_models.json")
    args = ap.parse_args()
    runs = [tuple(x.split(":", 1)) for x in args.runs.split(",") if x.strip()]
    data = {g: load(g, t) for g, t in runs}

    def fit(rows: list[dict]) -> mk.ShotModels:
        return mk.fit_shot_models([r["features"] for r in rows], [r["label"] for r in rows], l2=args.l2,
                                  rim=args.rim, zones=args.zones)

    report = {}
    if len(data) > 1:
        for g, rows in data.items():
            report[g] = score(fit([r for h, rs in data.items() if h != g for r in rs]), rows)
    every = [r for rs in data.values() for r in rs]
    models = fit(every)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**models.to_dict(), "fitted_on": [f"{g}:{t}" for g, t in runs],
                               "candidates": len(every), "held_out": report}, indent=1))
    print(json.dumps({"scored by models fitted on the other games": report}, indent=1))
    print(f"fitted on {len(every)} candidates of {len(data)} game(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
