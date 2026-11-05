"""Players on the court (teacher boxes, court registration, not referees): each model's hit per box, then paired
comparisons per box (McNemar) and per clip (sign test, and a bootstrap over clips of the recall difference)."""
import json, math, sys
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import detector_bench as db
sys.path.insert(0, str(Path(__file__).parent)); import mps_student; mps_student.patch_bench()  # .pt models on MPS
from courtvision.perception.student import INPUT_HW, preprocess_frames
from courtvision.nba.analyze import CameraTrack, is_referee

folder, game, models, held = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(","), sys.argv[4] == "held"
cam = CameraTrack.load(sorted((game / "camera_parts").glob("*.jsonl")))
ms = {m: db.load_model(Path(m), 4, sparse=True) for m in models}
rows = {(r["clip"], r["frame"]): r for r in map(json.loads, (folder / "detections.jsonl").open())}
labels = {(r["clip"], r["frame"]): r for r in map(json.loads, (folder / "ball_labels.jsonl").open())}
by_clip = defaultdict(list)
for c, f in rows: by_clip[c].append(f)
hits = {m: [] for m in models}; clip_of = []
x = np.empty((1, 9, *INPUT_HW), np.float32)
for clip, frames in sorted(by_clip.items()):
    if held and clip % 10 != 5: continue
    frames.sort()
    for i, f in enumerate(frames):
        lab = labels.get((clip, f))
        if i < 2 or not lab or lab.get("xy") is None or lab.get("w", 0) < 0.5: continue
        H = cam.homography(f)
        if H is None: continue
        imgs = [cv2.imread(str(folder / "frames" / f"{clip:04d}" / f"{g:06d}.jpg")) for g in frames[i - 2:i + 1]]
        for k, im in enumerate(imgs): x[0, 3 * k:3 * k + 3] = preprocess_frames([im])
        size = tuple(rows[(clip, f)]["size"])
        full = cv2.resize(imgs[-1], size)
        players = []
        for q in rows[(clip, f)]["persons"]:
            if q[4] < 0.5: continue
            X, Y = H.image_to_court(np.array([[0.5 * (q[0] + q[2]), q[3]]]))[0]
            if -2 <= X <= 96 and -2 <= Y <= 52 and not is_referee(full, tuple(q[:4])):
                players.append(q[:4])
        if not players: continue
        teach = np.array(players).reshape(-1, 4)
        for m, model in ms.items():
            _, persons = db.run_model(model, x, size, 0.08, 0.3)
            mine = np.array([p[:4] for p in persons]).reshape(-1, 4)
            h = (db.iou_matrix(teach, mine) >= 0.5).any(axis=1) if len(mine) else np.zeros(len(teach), bool)
            hits[m] += h.astype(int).tolist()
        clip_of += [clip] * len(teach)
clip_of = np.array(clip_of)
def mcnemar(a, b):
    n = a + b; k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
H = {m: np.array(v) for m, v in hits.items()}
print(f"{len(clip_of)} player boxes on {len(set(clip_of.tolist()))} clips")
for m in models: print(f"  {Path(m).name}: {H[m].mean():.4f}")
rng = np.random.default_rng(0)
clips = sorted(set(clip_of.tolist()))
idx = {c: np.flatnonzero(clip_of == c) for c in clips}
names = list(models)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = H[names[i]], H[names[j]]
        oa, ob = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        per = np.array([b[idx[c]].mean() - a[idx[c]].mean() for c in clips])
        up, down = int((per > 0).sum()), int((per < 0).sum())
        sign_p = mcnemar(up, down)
        boots = []
        for _ in range(2000):
            pick = rng.choice(len(clips), len(clips))
            sel = np.concatenate([idx[clips[k]] for k in pick])
            boots.append(b[sel].mean() - a[sel].mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  {Path(names[j]).name} - {Path(names[i]).name}: {b.mean() - a.mean():+.4f} (clip bootstrap 95% CI {lo:+.4f} to {hi:+.4f}); "
              f"boxes only it / only other {ob} / {oa} (McNemar p {mcnemar(oa, ob):.2g}); clips up / down {up} / {down} (sign p {sign_p:.2g})")
