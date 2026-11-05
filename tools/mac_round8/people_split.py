"""Teacher people on held-out teacher clips, split by where they stand (court registration) and whether they look
like a referee; each model's recall per group (IoU 0.5, model boxes at 0.3)."""
import json, sys
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
stat = {m: defaultdict(lambda: [0, 0]) for m in models}
x = np.empty((1, 9, *INPUT_HW), np.float32)
for clip, frames in sorted(by_clip.items()):
    if held and clip % 10 != 5: continue
    frames.sort()
    for i, f in enumerate(frames):
        lab = labels.get((clip, f))
        if i < 2 or not lab or lab.get("xy") is None or lab.get("w", 0) < 0.5: continue
        imgs = [cv2.imread(str(folder / "frames" / f"{clip:04d}" / f"{g:06d}.jpg")) for g in frames[i - 2:i + 1]]
        for k, im in enumerate(imgs): x[0, 3 * k:3 * k + 3] = preprocess_frames([im])
        size = tuple(rows[(clip, f)]["size"])
        full = cv2.resize(imgs[-1], size)
        H = cam.homography(f)
        groups = []
        for q in rows[(clip, f)]["persons"]:
            if q[4] < 0.5: continue
            if H is None: g = "no court fit"
            else:
                X, Y = H.image_to_court(np.array([[0.5 * (q[0] + q[2]), q[3]]]))[0]
                on = -2 <= X <= 96 and -2 <= Y <= 52
                g = ("referee on court" if is_referee(full, tuple(q[:4])) else "player on court") if on else "off court"
            groups.append((g, q[:4]))
        teach = np.array([b for _, b in groups]).reshape(-1, 4)
        for m, model in ms.items():
            _, persons = db.run_model(model, x, size, 0.08, 0.3)
            mine = np.array([p[:4] for p in persons]).reshape(-1, 4)
            hit = (db.iou_matrix(teach, mine) >= 0.5).any(axis=1) if len(mine) and len(teach) else np.zeros(len(teach), bool)
            for (g, _), h in zip(groups, hit):
                stat[m][g][0] += int(h); stat[m][g][1] += 1
            stat[m]["model boxes"][1] += len(mine)
for m in models:
    print(Path(m).name, {g: (f"{h}/{n} = {h / n:.3f}" if g != "model boxes" else n) for g, (h, n) in sorted(stat[m].items())})
