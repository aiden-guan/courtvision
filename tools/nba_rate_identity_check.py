"""A/B on a game folder: analyze.py / pseudo_labels.py / ball_labels.py of commit --rev (default 063e638, before the
frame-rate work) against the working tree, in one process with the same dependencies, on the stored camera track,
detections, ball track and teacher folders.
    .venv/bin/python ab_rate_check.py runs/nba/<game> [--det-dir detections] [--perceive START_S DURATION_S]
        [--detect START_S DURATION_S --weights runs/models/student/best.pt] [--rev 063e638]"""
import argparse
import gc
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from courtvision.nba import analyze as new  # noqa: E402
from courtvision.training import ball_labels as new_bl  # noqa: E402
from courtvision.training import pseudo_labels as new_pl  # noqa: E402


def at_head(path: str, name: str):  # noqa: ANN201
    src = subprocess.run(["git", "show", f"{REV}:{path}"], check=True, capture_output=True, text=True).stdout
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    sys.modules[name] = mod
    exec(compile(src, f"{REV}:{path}", "exec"), mod.__dict__)  # noqa: S102
    return mod


ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--det-dir", default="detections")
ap.add_argument("--perceive", nargs=2, type=float, metavar=("START_S", "DURATION_S"))
ap.add_argument("--detect", nargs=2, type=float, metavar=("START_S", "DURATION_S"))
ap.add_argument("--weights", default="runs/models/student/best.pt")
ap.add_argument("--rev", default="063e638")
args = ap.parse_args()
REV = args.rev
G = Path(args.root)
old = at_head("courtvision/nba/analyze.py", "analyze_head")
parts = sorted((G / "camera_parts").glob("part*.jsonl"))
cam_new, cam_old = new.CameraTrack.load(parts), old.CameraTrack.load(parts)
assert cam_new.params == cam_old.params, "camera track differs"
print(f"camera identical: {len(cam_new.params)} frames at {cam_new.fps} fps, {cam_new.image_size}", flush=True)
cache = new.DetectionCache()
for p in sorted((G / args.det_dir).glob("*.npz")):
    cache.merge(new.DetectionCache.load(p))
b_new = new.build_ball_track(cache, cam_new)
b_old = old.build_ball_track(cache, cam_old)
assert b_new.path == b_old.path, "ball path differs"
assert [f.to_dict() for f in b_new.flights] == [f.to_dict() for f in b_old.flights], "3-D flights differ"
print(f"ball path identical ({len(b_new.path)} frames), flights identical ({len(b_new.flights)})", flush=True)
if args.perceive or args.detect:
    game = json.loads((G / "game.json").read_text())
    video = Path(game["video"])
    video = G / video if not video.exists() and (G / video).exists() else video
if args.perceive:
    a, b = round(args.perceive[0] * cam_new.fps), round(sum(args.perceive) * cam_new.fps)
    reads = new.load_scorebug(G / "scorebug.jsonl")
    track = new.BallTrack.load(G / "ball_track.json") if (G / "ball_track.json").exists() else b_new

    def summary(fs):  # noqa: ANN001, ANN202
        bl = fs.ball
        return (fs.frame_idx, fs.t, None if fs.scoreboard is None else fs.scoreboard.t,
                [(p.player_id, None if p.team is None else p.team.value, p.x, p.y) for p in fs.players],
                None if bl is None else (bl.visible, bl.image_xy, bl.court_xy, bl.z))

    n = 0
    for x, y in zip(new.perceive(video, cam_new, cache, track, reads, start=a, end=b),
                    old.perceive(video, cam_old, cache, track, reads, start=a, end=b), strict=True):
        assert summary(x) == summary(y), f"frame {x.frame_idx} differs"
        n += 1
    print(f"perceive identical on {n} frames", flush=True)
del cache, b_new, b_old
gc.collect()
if args.detect:
    a, b = round(args.detect[0] * cam_new.fps), round(sum(args.detect) * cam_new.fps)
    d_new = new.detect_video(video, cam_new, Path(args.weights), a, b, progress=False)
    d_old = old.detect_video(video, cam_old, Path(args.weights), a, b, progress=False)
    assert sorted(d_new.balls) == sorted(d_old.balls), "detected frames differ"
    shapes = all(d_new.balls[f].shape == d_old.balls[f].shape for f in d_new.balls)
    worst = max((float(np.abs(d_new.balls[f] - d_old.balls[f]).max()) for f in d_new.balls
                 if d_new.balls[f].shape == d_old.balls[f].shape and d_new.balls[f].size), default=0.0)
    print(f"detect: {len(d_new.balls)} frames; same candidate counts: {shapes}; largest difference {worst:.6f}")
if (G / "ball_track.json").exists():
    old_pl = at_head("courtvision/training/pseudo_labels.py", "pseudo_labels_head")
    c_new, ctx = new_pl.build_candidates(G, new_pl.PseudoConfig())  # the folder the ball track was linked from
    del ctx
    gc.collect()
    c_old, ctx = old_pl.build_candidates(G, old_pl.PseudoConfig())
    del ctx
    gc.collect()
    assert [asdict(c) for c in c_new] == [asdict(c) for c in c_old], "pseudo-label candidates differ"
    t_new = new_pl.select_targets(c_new, new_pl.PseudoConfig(), cam_new.fps)
    t_old = old_pl.select_targets(c_old, old_pl.PseudoConfig(), cam_old.fps)
    assert [asdict(c) for c in t_new] == [asdict(c) for c in t_old], "pseudo-label selection differs"
    print(f"pseudo-label candidates identical ({len(c_new)}), selection identical ({len(t_new)})", flush=True)
old_bl = at_head("courtvision/training/ball_labels.py", "ball_labels_head")
for folder in sorted(G.glob("teacher*")):
    if (folder / "detections.jsonl").exists():
        assert new_bl.label_folder(folder) == old_bl.label_folder(folder), f"{folder.name}: ball labels differ"
        print(f"{folder.name}: teacher ball labels identical", flush=True)
print("identical")
