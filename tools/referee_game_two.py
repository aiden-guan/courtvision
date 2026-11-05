"""The in-game referee check (``nba.referee.fit_game``, ``GameConfig``'s defaults) on the README's two games, scored on
the 160 boxes labelled by eye (``docs/research/referee_check_two_games.json``, research note §10.10) beside the shirt
rule's and the learned check's calls stored there.

    .venv/bin/python tools/referee_game_two.py [out.json]

Each game's context is the population §10.10 drew its boxes from (the teacher's people at 0.5 or more placed on the
court on the test frames: the Lakers @ Warriors held-out clips, the 60 Cavaliers @ Warriors clips), taken as
``RefereeCheck("game")`` observes a game: frames at least ``observe_every`` video frames apart within each clip,
clustered once, without tracks. A labelled box is placed by the fit's vote (``GameFit.predict``), or takes the shirt
rule's call when the fit is not accepted. Exact McNemar tests on per-box correctness against the other two checks.
The labelled boxes were drawn where the shirt rule and the learned check disagree (60 a game) and where both call a
referee (20), so the counts describe those hard boxes, not a game's boxes at large. Needs the two games' teacher
folders and camera tracks (on the owner's Mac). Research use (docs/TRAINING.md §1)."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from courtvision.nba.analyze import CameraTrack, are_referees  # noqa: E402
from courtvision.nba.referee import GAME_FEATURE_NAMES, GameConfig, fit_game, game_features_batch  # noqa: E402

lab = json.loads((ROOT / "docs/research/referee_check_two_games.json").read_text())["boxes"]
games = [("lal", ROOT / "runs/nba/2024-12-25-LAL-GSW/teacher", ROOT / "runs/nba/2024-12-25-LAL-GSW", True),
         ("cle", ROOT / "runs/nba/2017-06-12-CLE-GSW/teacher-check", ROOT / "runs/nba/2017-06-12-CLE-GSW", False)]


def mcnemar(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


out = {}
for name, folder, game, held in games:
    cam = CameraTrack.load(sorted((game / "camera_parts").glob("*.jsonl")))
    feats, frames, shirt, keys = [], [], [], []
    every = GameConfig().observe_every
    want = defaultdict(list)
    for x in lab:
        if x["game"] == name:
            want[(x["clip"], x["frame"])].append(x)
    last = {}
    lab_feat = {}
    for fi, r in enumerate(map(json.loads, (folder / "detections.jsonl").open())):
        if held and r["clip"] % 10 != 5:
            continue
        key = (r["clip"], r["frame"])
        if key in want:
            img = cv2.imread(str(folder / "frames" / f"{r['clip']:04d}" / f"{r['frame']:06d}.jpg"))
            full = cv2.resize(img, tuple(r["size"]))
            bxs = [tuple(x["box"]) for x in want[key]]
            f = np.full((len(bxs), len(GAME_FEATURE_NAMES)), np.nan)
            g, keep = game_features_batch(full, bxs)
            f[keep] = g
            for x, row in zip(want[key], f):
                lab_feat[(x["clip"], x["frame"], tuple(x["box"]))] = row
        if r["clip"] in last and r["frame"] - last[r["clip"]] < every:
            continue
        H = cam.homography(int(r["frame"]))
        if H is None:
            continue
        boxes = []
        for q in r["persons"]:
            if q[4] < 0.5:
                continue
            X, Y = H.image_to_court(np.array([[0.5 * (q[0] + q[2]), q[3]]]))[0]
            if -2 <= X <= 96 and -2 <= Y <= 52:
                boxes.append(tuple(q[:4]))
        if not boxes:
            continue
        last[r["clip"]] = r["frame"]
        img = cv2.imread(str(folder / "frames" / f"{r['clip']:04d}" / f"{r['frame']:06d}.jpg"))
        full = cv2.resize(img, tuple(r["size"]))
        f = np.full((len(boxes), len(GAME_FEATURE_NAMES)), np.nan)
        g, keep = game_features_batch(full, boxes)
        f[keep] = g
        feats.append(f)
        frames += [fi] * len(boxes)
        shirt += list(are_referees(full, boxes, method="shirt"))
        keys += [(r["clip"], r["frame"], tuple(round(v, 1) for v in b)) for b in boxes]
    fit = fit_game(np.concatenate(feats), np.array(frames))
    calls = fit.calls if fit.accepted else np.array(shirt)
    n_frames = len(set(frames))
    mine = [x for x in lab if x["game"] == name]
    rows = []
    for x in mine:
        k = (x["clip"], x["frame"], tuple(x["box"]))
        truth = x["label"] == "referee"
        call = bool(fit.predict(lab_feat[k][None])[0]) if fit.accepted else x["rule"]
        rows.append((truth, call, x["rule"], x["learned"]))
    found = [r for r in rows if r[1] is not None]
    g_ok = [r[1] == r[0] for r in found]
    s_ok = [r[2] == r[0] for r in found]
    l_ok = [r[3] == r[0] for r in found]
    only_g, only_s = sum(a and not b for a, b in zip(g_ok, s_ok)), sum(b and not a for a, b in zip(g_ok, s_ok))
    only_gl, only_l = sum(a and not b for a, b in zip(g_ok, l_ok)), sum(b and not a for a, b in zip(g_ok, l_ok))
    res = {"context_boxes": len(keys), "frames": n_frames, "accepted": bool(fit.accepted), "reason": fit.reason,
           "fit_per_frame": float(fit.per_frame), "silhouette": float(fit.silhouette), "sizes": list(fit.sizes), "parts": fit.parts,
           "called_referee_per_frame": float(calls.sum()) / n_frames, "shirt_referee_per_frame": float(np.sum(shirt)) / n_frames,
           "labelled_found": len(found), "of": len(rows),
           "right": {"game": int(sum(g_ok)), "shirt": int(sum(s_ok)), "learned": int(sum(l_ok))},
           "vs_shirt": [only_g, only_s, mcnemar(only_g, only_s)], "vs_learned": [only_gl, only_l, mcnemar(only_gl, only_l)],
           "by_label": {lb: {"n": sum(1 for x in mine if x["label"] == lb),
                             "called_referee": sum(1 for x, r in zip(mine, rows) if x["label"] == lb and r[1])} for lb in ("referee", "player", "other")}}
    out[name] = res
    print(name, json.dumps(res), flush=True)
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
