"""Cut the rim clips of a game for the rim model: every shot candidate, every play-by-play shot, and moments with no
shot, each labelled from the play-by-play (research use: docs/NBA.md).

    .venv/bin/python tools/nba_rim_clips.py runs/nba/<game> --tag <run> [--negatives 1.0] [--seed 0]

Reads the game's camera track, ball track, the run's ``events.jsonl`` and the play-by-play placed on video time
(``nba_analyze.truth``). Writes ``runs/nba/<game>/rim_clips/clips.npz`` (``frames`` N x T x 96 x 96 x 3 uint8 BGR,
``valid`` N x T) and ``meta.jsonl`` (one line per clip, in the same order):

* ``source`` "candidate": the candidates of ``makes.candidates`` (the engine's shot results the camera saw and the
  linked ball's rim visits), centred where the linked ball passes nearest the rim (``rim_jev.rim_centre``), labelled
  by ``makes.label_candidates``: "made" / "missed" (``ft`` says whether a free throw) or "none" (no shot there);
* ``source`` "pbp": the play-by-play shots no candidate matched, centred the same way around their aligned time at
  the rim the shooting team attacks (the attack map of the run);
* ``source`` "random": moments with a rim in view and no play-by-play shot within ``--clear-s``, ``--negatives``
  times as many as the shots, labelled "none".

``rim_clips.py`` says how the frames are cut; ``tools/nba_rim_model.py`` trains on them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from courtvision.eval.events import latest_revisions  # noqa: E402
from courtvision.geometry.court import court_for  # noqa: E402
from courtvision.nba import makes as mk  # noqa: E402
from courtvision.nba import timeline as tlm  # noqa: E402
from courtvision.nba.analyze import BallTrack, rim_boxes  # noqa: E402
from courtvision.nba.fusion import PlayDirection  # noqa: E402
from courtvision.nba.rim_clips import clip_at  # noqa: E402
from courtvision.nba.rim_jev import Frames, rim_centre  # noqa: E402
from courtvision.types import Event, EventStatus  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nba_analyze import load_game, truth  # noqa: E402


def truth_shots(aligned) -> list[dict]:  # noqa: ANN001
    out = []
    for al in aligned.aligned:
        a = al.action
        if al.t is None or not a.is_shot or a.synthetic or a.made is None:
            continue
        out.append({"t": float(al.t), "ft": bool(a.is_free_throw), "made": bool(a.made), "team": a.team,
                    "value": int(a.value or 0), "method": al.method, "text": a.description})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--negatives", type=float, default=1.0, help="random no-shot clips per play-by-play shot")
    ap.add_argument("--clear-s", type=float, default=4.0, help="a random clip has no play-by-play shot this near")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rim_clips")
    args = ap.parse_args()
    root = Path(args.root)
    t0 = time.time()
    game, camera = load_game(root)
    court = court_for("nba")
    events = [Event.from_dict(json.loads(line)) for line in (root / "runs" / args.tag / "events.jsonl").open()]
    events = [e for e in latest_revisions(events) if e.status is not EventStatus.RETRACTED]
    ball = BallTrack.load(root / "ball_track.json")
    path = {f: (p[0], p[1]) for f, p in ball.path.items()}
    shots = truth_shots(truth(root)[1])
    attack = tlm.attack_map(events, PlayDirection.from_camera(camera))

    visits = mk.rim_visits(ball.path, camera, court)
    cands = mk.candidates(events, visits)
    labels = mk.label_candidates(cands, [(s["t"], s["ft"], s["made"]) for s in shots])
    matched = set()
    for c in cands:
        lab = labels[c.key]
        if lab != "false":  # the play-by-play shot it took: the nearest one with the same label
            j = min((j for j, s in enumerate(shots) if j not in matched and abs(s["t"] - c.t) <= 2.5
                     and lab == f"{'ft' if s['ft'] else 'fg'}_{'made' if s['made'] else 'missed'}"),
                    key=lambda j: abs(shots[j]["t"] - c.t), default=None)
            if j is not None:
                matched.add(j)

    todo: list[dict] = []
    for c in cands:
        lab = labels[c.key]
        todo.append({"source": "candidate", "key": c.key, "t": c.t, "side": c.side, "engine": c.result is not None,
                     "label": "none" if lab == "false" else lab.split("_")[1], "ft": lab.startswith("ft")})
    for j, s in enumerate(shots):
        if j in matched:
            continue
        side = attack.side(s["team"], s["t"]) if s["team"] is not None else None
        todo.append({"source": "pbp", "key": f"pbp-{s['t']:.2f}", "t": s["t"], "side": side, "engine": False,
                     "label": "made" if s["made"] else "missed", "ft": s["ft"], "method": s["method"]})
    rng = np.random.default_rng(args.seed)
    shot_t = np.array(sorted(s["t"] for s in shots))
    frames_with_rim = [f for f in sorted(camera.params)[::15]
                       if rim_boxes(camera.get(f), court, camera.image_size)]
    want = int(round(args.negatives * len(shots)))
    for f in rng.permutation(frames_with_rim):
        if want <= 0:
            break
        t = camera.time(int(f))
        if len(shot_t) and np.min(np.abs(shot_t - t)) < args.clear_s:
            continue
        sides = list(rim_boxes(camera.get(int(f)), court, camera.image_size))
        todo.append({"source": "random", "key": f"rnd-{t:.2f}", "t": t, "side": sides[int(rng.integers(len(sides)))],
                     "engine": False, "label": "none", "ft": False})
        want -= 1

    reader = Frames(game["video"])
    clips, valid, meta = [], [], []
    skipped = 0
    for item in sorted(todo, key=lambda d: d["t"]):
        if item["source"] == "random":
            centre, side, how = camera.frame_at(item["t"]), item["side"], "time"
        else:
            found = rim_centre(item["t"], camera, court, path, side=item["side"])
            if found is None:
                skipped += 1
                continue
            centre, side, how = found
        clip = clip_at(reader, camera, court, centre, side)
        if clip is None or clip.valid.sum() < len(clip.valid) // 2:
            skipped += 1
            continue
        clips.append(clip.frames)
        valid.append(clip.valid)
        meta.append({**item, "side": side, "centre": centre, "t_centre": round(camera.time(centre), 3), "how": how,
                     "game": root.name})
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "clips.npz", frames=np.stack(clips), valid=np.stack(valid))
    with (out / "meta.jsonl").open("w") as fh:
        for m in meta:
            fh.write(json.dumps(m) + "\n")
    counts: dict[str, int] = {}
    for m in meta:
        k = f"{m['source']}:{m['label']}{':ft' if m['ft'] else ''}"
        counts[k] = counts.get(k, 0) + 1
    print(f"{len(meta)} clips ({skipped} skipped: no rim in view) in {time.time() - t0:.0f}s -> {out}")
    print(json.dumps(dict(sorted(counts.items())), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
