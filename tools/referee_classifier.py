"""Fit and check the pipeline's referee classifier (``nba/referee.py``) on COCO-format human labels.

    .venv/bin/python tools/referee_classifier.py \\
        --source data/hfcoco_repo/train/_annotations.coco.json:data/hfcoco_repo/train \\
        --source data/hfcoco/valid_ann.json:data/hfcoco/valid --source data/hfcoco/test_ann.json:data/hfcoco/test \\
        --out courtvision/nba/referee_model.py

Every labelled ``referee`` and ``player`` / ``player-<action>`` box of every image is a sample (plus ``--jitter``
copies with the box moved and resized by about 6 %, as a detector's boxes are). ``--players`` adds folders whose people
are all players (``student_finetune.py prep-mot``: SportsMOT boxes the players on the court only), every
``--players-every``-th labelled frame, each broadcast a game of its own: more uniforms that are not a referee's (the
Roboflow games' dark jerseys are few; research note ``cross_game_detector.md`` §10.10). Games are read from the file names
(``...-game-N``). First each game is left out in turn: a model fitted on the others calls that game's boxes at 0.5,
beside the earlier shirt rule (``analyze.is_referee(..., method="shirt")``), with an exact McNemar test on the boxes
only one of the two gets right. Then one model is fitted on every game and written to ``--out`` as Python constants.
Research use: the labels are CC BY 4.0, the footage is the NBA's (docs/TRAINING.md §1).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from courtvision.nba.analyze import is_referee  # noqa: E402
from courtvision.nba.referee import FEATURE_NAMES, fit_logistic, referee_features  # noqa: E402

THRESHOLD = 0.5  # the classes are weighted equally in the fit, so 0.5 is the even point; fixed, never tuned


def game_of(file_name: str) -> str:
    m = re.match(r".*?-game-\d+", file_name)
    return m.group(0) if m else "other"


def load(ann: Path, images: Path, jitter: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str], list]:
    """(features, is a referee, game, shirt rule's call on the labelled box or None for a jittered copy)."""
    rng = np.random.default_rng(seed)
    d = json.loads(ann.read_text())
    cats = {c["id"]: c["name"] for c in d["categories"]}
    by_img: dict[int, list[dict]] = defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    X, y, games, rule = [], [], [], []
    for im in d["images"]:
        img = cv2.imread(str(images / im["file_name"]))
        if img is None:
            continue
        game = game_of(im["file_name"])
        for a in by_img[im["id"]]:
            c = cats[a["category_id"]]
            if c != "referee" and not (c == "player" or c.startswith("player-")):
                continue
            x, yy, w, h = a["bbox"]
            boxes = [(x, yy, x + w, yy + h)]
            for _ in range(jitter):
                dx, dy, sw, sh = rng.normal(0, 0.06, 4)
                boxes.append((x + dx * w, yy + dy * h, x + w * (1 + sw) + dx * w, yy + h * (1 + sh) + dy * h))
            for k, b in enumerate(boxes):
                f = referee_features(img, b)
                if f is None:
                    continue
                X.append(f)
                y.append(c == "referee")
                games.append(game)
                rule.append(is_referee(img, b, method="shirt") if k == 0 else None)
    return np.array(X).reshape(-1, len(FEATURE_NAMES)), np.array(y, bool), games, rule


def load_players(folder: Path, every: int, jitter: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str], list]:
    """``load``'s arrays for a ClipDataset folder whose labelled people are all players (``prep-mot``): its stored
    frames (scaled from the rows' ``size``), every ``every``-th labelled row; the game is the SportsMOT broadcast of the
    clip's sequence (``v_<id>``)."""
    rng = np.random.default_rng(seed)
    meta = json.loads((folder / "meta.json").read_text())
    seq = {c["clip"]: c.get("sequence", str(c["clip"])) for c in meta.get("clips", [])}
    X, y, games, rule = [], [], [], []
    n = 0
    for r in map(json.loads, (folder / "detections.jsonl").open()):
        if r.get("context") or not r.get("persons"):
            continue
        n += 1
        if n % every:
            continue
        img = cv2.imread(str(folder / "frames" / f"{r['clip']:04d}" / f"{r['frame']:06d}.jpg"))
        if img is None:
            continue
        k = img.shape[1] / float(r["size"][0])
        game = "sportsmot:" + seq.get(r["clip"], str(r["clip"])).rsplit("_c", 1)[0]
        for q in r["persons"]:
            x1, y1, x2, y2 = (v * k for v in q[:4])
            w, h = x2 - x1, y2 - y1
            boxes = [(x1, y1, x2, y2)]
            for _ in range(jitter):
                dx, dy, sw, sh = rng.normal(0, 0.06, 4)
                boxes.append((x1 + dx * w, y1 + dy * h, x2 + w * sw + dx * w, y2 + h * sh + dy * h))
            for j, b in enumerate(boxes):
                f = referee_features(img, b)
                if f is None:
                    continue
                X.append(f)
                y.append(False)
                games.append(game)
                rule.append(is_referee(img, b, method="shirt") if j == 0 else None)
    return np.array(X).reshape(-1, len(FEATURE_NAMES)), np.array(y, bool), games, rule


def prob(model: tuple, X: np.ndarray) -> np.ndarray:
    mu, sd, coef, bias = model
    return 1.0 / (1.0 + np.exp(-(((X - mu) / sd) @ coef + bias)))


def counts(called: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    tp, fp = int((called & truth).sum()), int((called & ~truth).sum())
    fn = int((~called & truth).sum())
    return {"referees": tp + fn, "referees_called": tp, "players": int((~truth).sum()), "players_called_referee": fp,
            "recall": tp / (tp + fn) if tp + fn else None, "precision": tp / (tp + fp) if tp + fp else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    only_a, only_b = int((a & ~b).sum()), int((b & ~a).sum())
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0  # exact, any n
    return {"only_rule": only_a, "only_learned": only_b, "p_value": p}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", action="append", required=True, help="ANN.json:IMAGES_DIR (repeat)")
    ap.add_argument("--players", action="append", default=[], type=Path,
                    help="a folder whose people are all players (prep-mot; repeat): more uniforms that are not a referee's")
    ap.add_argument("--players-every", type=int, default=10, help="--players: every Nth labelled frame")
    ap.add_argument("--jitter", type=int, default=2)
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--out", type=Path, default=None, help="write the fitted model here (a Python module)")
    ap.add_argument("--json", type=Path, default=None, help="the leave-one-game-out results")
    a = ap.parse_args(argv)
    parts = []
    for i, src in enumerate(a.source, start=1):
        ann, images = src.split(":", 1)
        parts.append(load(Path(ann), Path(images), a.jitter, seed=i))
    for i, folder in enumerate(a.players, start=len(a.source) + 1):
        parts.append(load_players(folder, a.players_every, a.jitter, seed=i))
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    g = np.array(sum((p[2] for p in parts), []))
    rule = sum((p[3] for p in parts), [])
    labelled = np.array([r is not None for r in rule])
    rule_call = np.array([bool(r) for r in rule])
    games = sorted(x for x in set(g) if x != "other")
    print(f"{int(labelled.sum())} labelled boxes ({len(X)} with the jittered copies), {len(games)} games")
    per_game, right_rule, right_learned = {}, [], []
    pooled = {"rule": (np.zeros(0, bool), np.zeros(0, bool)), "learned": (np.zeros(0, bool), np.zeros(0, bool))}
    extra = {"rule": (np.zeros(0, bool), np.zeros(0, bool)), "learned": (np.zeros(0, bool), np.zeros(0, bool))}
    print("| game left out | rule: referees called / players called referee, F1 | learned: the same |")
    print("|---|---|---|")
    for game in games:
        tr, te = (g != game) & (g != "other"), (g == game) & labelled
        model = fit_logistic(X[tr], y[tr], a.l2)
        learned = prob(model, X[te]) >= THRESHOLD
        truth, by_rule = y[te], rule_call[te]
        per_game[game] = {"rule": counts(by_rule, truth), "learned": counts(learned, truth)}
        if game.startswith("sportsmot:"):  # players only: reported apart, so the NBA games' figures keep their meaning
            for name, c in (("rule", by_rule), ("learned", learned)):
                extra[name] = (np.r_[extra[name][0], c], np.r_[extra[name][1], truth])
        else:
            for name, c in (("rule", by_rule), ("learned", learned)):
                pooled[name] = (np.r_[pooled[name][0], c], np.r_[pooled[name][1], truth])
            right_rule.append(by_rule == truth)
            right_learned.append(learned == truth)
        r, L = per_game[game]["rule"], per_game[game]["learned"]
        print(f"| {game} | {r['referees_called']}/{r['referees']} / {r['players_called_referee']}/{r['players']}, "
              f"{r['f1']:.3f} | {L['referees_called']}/{L['referees']} / {L['players_called_referee']}/{L['players']},"
              f" {L['f1']:.3f} |")
    summary = {name: counts(*pooled[name]) for name in pooled}
    test = mcnemar(np.concatenate(right_rule), np.concatenate(right_learned))
    for name, s in summary.items():
        print(f"pooled {name}: referees {s['referees_called']}/{s['referees']} ({s['recall']:.3f}), players called "
              f"referee {s['players_called_referee']}/{s['players']}, precision {s['precision']:.3f}, F1 {s['f1']:.3f}")
    print(f"boxes only the rule gets right {test['only_rule']}, only the learned classifier {test['only_learned']}, "
          f"exact McNemar p {test['p_value']:.3g}")
    players_only = {name: counts(*extra[name]) for name in extra} if len(extra["rule"][0]) else None
    if players_only:
        print("players-only broadcasts, each left out: players called referee: rule "
              f"{players_only['rule']['players_called_referee']}/{players_only['rule']['players']}, learned "
              f"{players_only['learned']['players_called_referee']}/{players_only['learned']['players']}")
    result = {"games": games, "per_game": per_game, "pooled": summary, "mcnemar": test, "threshold": THRESHOLD,
              "l2": a.l2, "jitter": a.jitter, "sources": a.source, "players": [str(f) for f in a.players],
              "players_every": a.players_every, "players_only": players_only}
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(result, indent=1))
    if a.out:
        keep = g != "other"
        mu, sd, coef, bias = fit_logistic(X[keep], y[keep], a.l2)
        fmt = lambda v: "(" + ", ".join(f"{float(x):.10g}" for x in v) + ",)"  # noqa: E731
        s = summary
        a.out.write_text(
            '"""The referee classifier\'s weights (``nba/referee.py``), written by tools/referee_classifier.py: do not\n'
            'edit by hand. Fitted on human labels of these NBA games (Roboflow Universe "basketball-player-detection-2",\n'
            f'CC BY 4.0; research use): {", ".join(games)}.\n'
            f"Each game left out in turn: referees {s['learned']['referees_called']}/{s['learned']['referees']}, players "
            f"called referee {s['learned']['players_called_referee']}/{s['learned']['players']} (the shirt rule\n"
            f"{s['rule']['referees_called']}/{s['rule']['referees']} and {s['rule']['players_called_referee']}), "
            f"F1 {s['learned']['f1']:.3f} against {s['rule']['f1']:.3f}.\n"
            '"""\n\n'
            f"FEATURES = {tuple(FEATURE_NAMES)!r}\n"
            f"MEAN = {fmt(mu)}\nSTD = {fmt(sd)}\nCOEF = {fmt(coef)}\nBIAS = {float(bias):.10g}\n"
            f"THRESHOLD = {THRESHOLD}\n")
        print(f"{a.out}: fitted on {int(keep.sum())} samples of {len(games)} games")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
