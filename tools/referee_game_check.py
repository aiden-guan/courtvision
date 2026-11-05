"""Score the in-game referee check (``nba/referee.py``: ``fit_game``) against the shirt rule and the learned check, per
game, on COCO-format human labels.

    .venv/bin/python tools/referee_game_check.py --coco data/ebard/calib_ann.json:data/ebard/images
    .venv/bin/python tools/referee_game_check.py --games roboflow \\
        --coco data/hfcoco/heldout_ann.json:data/hfcoco/heldout

Every labelled ``referee`` and ``player`` / ``player-<action>`` box is a sample. The boxes of one game are the in-game
check's context (E-BARD: a game is the file name before its last ``_``, 30 frames from a whole game; Roboflow:
``...-game-N``), without tracks. Per game and pooled over the games: referees called, players called referee,
precision, recall and F1 of the shirt rule (``analyze``'s default), the learned check (``nba.referee_model``) and the
in-game check; whether the in-game check fell back to the shirt rule; exact McNemar tests on per-box correctness, the
in-game check against each of the other two. ``--dedup`` keeps one copy of a frame Roboflow exported twice (its
training split holds flipped, re-lit copies, which a nearest-neighbour graph would pair). The settings are
``GameConfig``'s defaults, fixed on the 30 E-BARD training games (``calib_ann.json``) before the held-out ones
(``check_ann.json``) were scored. Research use: the labels are CC BY 4.0 (E-BARD, Roboflow Universe), the footage is
the NBA's (docs/TRAINING.md §1).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from courtvision.nba.referee import (  # noqa: E402
    GAME_FEATURE_NAMES,
    game_features_batch,
    game_referee_calls,
    referee_calls,
)
from tools.referee_classifier import counts, mcnemar  # noqa: E402

METHODS = ("shirt", "learned", "game")


def game_of(file_name: str, games: str) -> str:
    if games == "ebard":
        return file_name.rsplit("_", 1)[0]
    m = re.match(r".*?-game-\d+", file_name)
    return m.group(0) if m else "other"


def load(ann: Path, images: Path, games: str, dedup: bool) -> dict[str, dict[str, list]]:
    """Per game: each labelled person box's features, frame, truth, and the shirt rule's and learned check's calls."""
    d = json.loads(ann.read_text())
    cats = {c["id"]: c["name"] for c in d["categories"]}
    by_img: dict[int, list[dict]] = defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    out: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    seen: set[str] = set()
    for im in sorted(d["images"], key=lambda im: im["file_name"]):
        base = re.sub(r"\.rf\.[0-9a-f]+\.jpg$", "", im["file_name"])
        if dedup and base in seen:
            continue
        seen.add(base)
        people = [a for a in by_img[im["id"]]
                  if cats[a["category_id"]] in ("referee", "player") or cats[a["category_id"]].startswith("player-")]
        if not people:
            continue
        img = cv2.imread(str(images / im["file_name"]))
        if img is None:
            print(f"missing image {im['file_name']}", file=sys.stderr)
            continue
        boxes = [(a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]) for a in people]
        feats = np.full((len(boxes), len(GAME_FEATURE_NAMES)), np.nan)
        f, keep = game_features_batch(img, boxes)
        feats[keep] = f
        g = out[game_of(im["file_name"], games)]
        g["features"] += list(feats)
        g["frame"] += [im["id"]] * len(boxes)
        g["truth"] += [cats[a["category_id"]] == "referee" for a in people]
        g["shirt"] += referee_calls(img, boxes, "shirt")
        g["learned"] += referee_calls(img, boxes, "learned")
    return out


def score(per_game: dict[str, dict[str, list]]) -> dict:
    games = sorted(g for g in per_game if g != "other")
    rows, pooled = {}, {m: ([], []) for m in METHODS}
    right = {m: [] for m in METHODS}
    for game in games:
        g = per_game[game]
        truth = np.array(g["truth"], bool)
        calls = {m: np.array(g[m], bool) for m in ("shirt", "learned")}
        calls["game"], fit = game_referee_calls(np.array(g["features"]), g["frame"], calls["shirt"])
        rows[game] = {m: counts(calls[m], truth) for m in METHODS}
        rows[game]["fit"] = {"accepted": fit.accepted, "reason": fit.reason, "per_frame": fit.per_frame,
                             "silhouette": fit.silhouette, "graph_parts": fit.parts, "cluster_sizes": list(fit.sizes),
                             "frames": len(set(g["frame"]))}
        for m in METHODS:
            pooled[m][0].append(calls[m])
            pooled[m][1].append(truth)
            right[m].append(calls[m] == truth)
    summary = {m: counts(np.concatenate(pooled[m][0]), np.concatenate(pooled[m][1])) for m in METHODS}
    tests = {}
    for other in ("shirt", "learned"):
        t = mcnemar(np.concatenate(right["game"]), np.concatenate(right[other]))
        tests[f"game_vs_{other}"] = {"only_game": t["only_rule"], f"only_{other}": t["only_learned"],
                                     "p_value": t["p_value"]}
    fell = [g for g in games if not rows[g]["fit"]["accepted"]]
    return {"games": games, "per_game": rows, "pooled": summary, "mcnemar": tests, "fell_back": fell}


def _f(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def report(name: str, r: dict) -> None:
    print(f"\n{name}: {len(r['games'])} games, the in-game check fell back to the shirt rule on {len(r['fell_back'])}"
          + (f" ({', '.join(r['fell_back'])})" if r["fell_back"] else ""))
    print("| game | referees | players | shirt rule: referees called, players called referee, F1 | learned check: the "
          "same | in-game check: the same | referee cluster: a frame, silhouette |")
    print("|---|---:|---:|---|---|---|---:|")
    for game in r["games"]:
        row = r["per_game"][game]
        cells = [f"{row[m]['referees_called']}, {row[m]['players_called_referee']}, {_f(row[m]['f1'])}"
                 for m in METHODS]
        fit = row["fit"]
        size = f"{fit['per_frame']:.2f}, {fit['silhouette']:.3f}" + \
            ("" if fit["accepted"] else f" (fell back: {fit['reason']})")
        print(f"| {game} | {row['game']['referees']} | {row['game']['players']} | " + " | ".join(cells)
              + f" | {size} |")
    print("\n| pooled | referees called / referees | players called referee / players | precision | recall | F1 |")
    print("|---|---:|---:|---:|---:|---:|")
    for m in METHODS:
        s = r["pooled"][m]
        print(f"| {m} | {s['referees_called']} / {s['referees']} | {s['players_called_referee']} / {s['players']} | "
              f"{_f(s['precision'])} | {_f(s['recall'])} | {_f(s['f1'])} |")
    for key, t in r["mcnemar"].items():
        other = key.split("_vs_")[1]
        print(f"{key}: boxes only the in-game check gets right {t['only_game']}, only the {other} "
              f"{t[f'only_{other}']}, exact McNemar p {t['p_value']:.3g}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--coco", action="append", required=True, help="ANN.json:IMAGES_DIR (repeat: each scored apart)")
    ap.add_argument("--games", choices=("ebard", "roboflow"), default="ebard",
                    help="how a game is read from a file name: E-BARD's name before the last '_', Roboflow's -game-N")
    ap.add_argument("--dedup", action="store_true", help="one copy of a frame Roboflow exported more than once")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    cv2.setNumThreads(1)
    results = {}
    for src in a.coco:
        ann, images = src.split(":", 1)
        results[ann] = score(load(Path(ann), Path(images), a.games, a.dedup))
        report(Path(ann).name, results[ann])
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(results, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
