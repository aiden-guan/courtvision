"""Train and score the rim model (``courtvision/nba/rim_model.py``) on the rim clips of one or more games
(``tools/nba_rim_clips.py``), research use.

    .venv/bin/python tools/nba_rim_model.py --games 2024-12-25-LAL-GSW[,2017-06-12-CLE-GSW] --folds halves
        [--name rim-v1] [--epochs 24] [--px 96] [--save runs/models/rim/rim-v1.pt]

``--folds``:

* ``halves``: each game cut at the middle of its clips' times; a fold trains on every clip but those of one half of
  one game and predicts that half (with one game: fitted on one half, scored on the other, both ways, as
  ``nba_analyze.py timeline`` scores the shot models);
* ``games``: train on the other games, predict one (needs two or more);
* ``none``: no scoring, only ``--save``.

Every clip gets its held-out probabilities, written to ``<game>/rim_clips/oof_<name>.jsonl`` (``key``, ``t``,
``p_made``, ``p_missed``, ``p_none``) for ``nba/makes.py`` to read; the scores printed are on them: made or missed on
the real shots (the candidates' and the play-by-play's), shot or no shot on the candidates, both as accuracy and
area under the ROC curve. ``--save`` trains once more on every clip and saves the model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from courtvision.nba import rim_model as rm  # noqa: E402

NBA = Path("runs/nba")


def auc(scores: np.ndarray, pos: np.ndarray) -> float:
    """Area under the ROC curve (ties counted half)."""
    s, p = np.asarray(scores, float), np.asarray(pos, bool)
    if p.all() or not p.any():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks of ties
        m = s == v
        ranks[m] = ranks[m].mean()
    return float((ranks[p].sum() - p.sum() * (p.sum() + 1) / 2) / (p.sum() * (~p).sum()))


def report(meta: list[dict], probs: np.ndarray) -> dict:
    lab = np.array([m["label"] for m in meta])
    src = np.array([m["source"] for m in meta])
    real = np.isin(lab, ("made", "missed"))
    pm = probs[:, 0] / np.maximum(probs[:, 0] + probs[:, 1], 1e-9)
    out = {}
    for name, sel in (("real shots", real), ("real shots, candidates", real & (src == "candidate")),
                      ("field goals", real & ~np.array([m["ft"] for m in meta])),
                      ("free throws", real & np.array([m["ft"] for m in meta]))):
        if sel.any():
            out[f"made or missed, {name}"] = {"n": int(sel.sum()),
                                              "accuracy": round(float(np.mean((pm[sel] >= 0.5) == (lab[sel] == "made"))), 4),
                                              "auc": round(auc(pm[sel], lab[sel] == "made"), 4)}
    cand = src == "candidate"
    if not (lab == "none").any():
        return out
    shot = 1.0 - probs[:, 2]
    out["shot or not, candidates"] = {"n": int(cand.sum()),
                                      "accuracy": round(float(np.mean((shot[cand] >= 0.5) == real[cand])), 4),
                                      "auc": round(auc(shot[cand], real[cand]), 4)}
    out["shot or not, all clips"] = {"n": len(lab), "accuracy": round(float(np.mean((shot >= 0.5) == real)), 4),
                                     "auc": round(auc(shot, real), 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", required=True)
    ap.add_argument("--folds", choices=("halves", "games", "none"), default="halves")
    ap.add_argument("--name", default="rim-v1")
    ap.add_argument("--clips", default="rim_clips", help="the clips folder in each game folder")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--px", type=int, default=None)
    ap.add_argument("--config", default=None, help="RimConfig overrides (JSON)")
    ap.add_argument("--save", default=None, help="train on every clip and save the model here")
    ap.add_argument("--real-only", action="store_true",
                    help="train on the real shots only (made or missed); every held-out clip is still predicted")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    games = [g.strip() for g in args.games.split(",") if g.strip()]
    folders = [NBA / g / args.clips for g in games]
    frames, _, meta = rm.load_clips(folders)
    over = json.loads(args.config) if args.config else {}
    if args.epochs:
        over["epochs"] = args.epochs
    if args.px:
        over["px"] = args.px
    cfg = rm.RimConfig(**over)
    y = np.array([rm.CLASSES.index(m["label"]) for m in meta])
    game_of = np.array([m["game"] for m in meta])
    t = np.array([m["t"] for m in meta])
    print(f"{len(meta)} clips from {len(games)} game(s); classes {dict(zip(rm.CLASSES, np.bincount(y, minlength=3).tolist()))}")
    folds: list[tuple[str, np.ndarray]] = []
    if args.folds == "halves":
        for g in games:
            mid = float(np.median(t[game_of == g]))
            folds += [(f"{g} first half", (game_of == g) & (t < mid)), (f"{g} second half", (game_of == g) & (t >= mid))]
    elif args.folds == "games":
        if len(games) < 2:
            raise SystemExit("--folds games needs two or more games")
        folds = [(g, game_of == g) for g in games]
    probs = np.full((len(meta), len(rm.CLASSES)), np.nan, np.float32)
    t0 = time.time()
    trainable = np.array([not args.real_only or m["label"] in ("made", "missed") for m in meta])
    for name, test in folds:
        tr = np.flatnonzero(~test & trainable)
        te = np.flatnonzero(test)
        print(f"fold {name}: train {len(tr)}, test {len(te)}")
        net = rm.train(frames[tr], y[tr], cfg, val=(frames[te], y[te]))
        probs[te] = rm.predict(net, frames[te], cfg)
        print(json.dumps(report([meta[i] for i in te], probs[te])))
    if folds:
        print(f"held-out, all folds ({time.time() - t0:.0f}s):")
        print(json.dumps(report(meta, probs), indent=1))
        for g, d in zip(games, folders, strict=True):
            with (d / f"oof_{args.name}.jsonl").open("w") as fh:
                for i in np.flatnonzero(game_of == g):
                    m = meta[i]
                    fh.write(json.dumps({"key": m["key"], "source": m["source"], "t": m["t"], "label": m["label"],
                                         "ft": m["ft"], **{f"p_{c}": round(float(probs[i, k]), 5)
                                                           for k, c in enumerate(rm.CLASSES)}}) + "\n")
    if args.save:
        net = rm.train(frames[trainable], y[trainable], cfg)
        rm.save_model(net, cfg, Path(args.save), {"games": games, "clips": len(meta), "name": args.name})
        print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
