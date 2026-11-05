"""prep_mps.py against the tool's own prep_video with the network on MPS one frame at a time (validates the
batching and the rewritten loop), and the tool's ONNX network against MPS on a few frames (validates the backend).
usage (from the repository root): prep_check.py WORK, where WORK/t_src/<folder>/ holds a few clips; WORK/t_tool and
WORK/t_mps are written."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from mps_student import MpsNet
import student_finetune as sf
import courtvision.perception.student_onnx as so
from courtvision.perception.student import decode
import prep_mps

P = Path(sys.argv[1])
V2, PT = Path("courtvision/server/static/nba-student.onnx"), Path("runs/models/v2.pt")


class OneAtATime:
    def __init__(self, path, threads=None):
        self.net = MpsNet(PT)

    def detect(self, x, scale, *, ball_thresh, person_thresh, ball_k=8, person_k=60):
        raw = self.net(x)
        one = dict(raw)
        for k in ("ball_hm", "person_hm"):
            one[k] = 1.0 / (1.0 + np.exp(-raw[k]))
        b, q = decode(one, scale, ball_k=ball_k, ball_thresh=ball_thresh, person_k=person_k, person_thresh=person_thresh)
        return [tuple(map(float, v)) for v in b], [tuple(map(float, v)) for v in q]


orig = so.OnnxStudent
so.OnnxStudent = OneAtATime
sf.prep_video(P / "t_src", P / "t_tool", V2, 3, 60, 1)
so.OnnxStudent = orig
prep_mps.prep_video(P / "t_src", P / "t_mps", V2, PT, 3, 60, False, 92, 8)
a = [json.loads(l) for l in (P / "t_tool" / "detections.jsonl").open()]
b = [json.loads(l) for l in (P / "t_mps" / "detections.jsonl").open()]
la = [json.loads(l) for l in (P / "t_tool" / "ball_labels.jsonl").open()]
lb = [json.loads(l) for l in (P / "t_mps" / "ball_labels.jsonl").open()]
ma, mb = json.loads((P / "t_tool" / "meta.json").read_text()), json.loads((P / "t_mps" / "meta.json").read_text())
mb.pop("backend")
print("rows", len(a), len(b), "keys equal", all(x.keys() == y.keys() for x, y in zip(a, b)))
print("frames/t/size/context equal", all((x["clip"], x["frame"], x["t"], x["size"], x.get("context")) == (y["clip"], y["frame"], y["t"], y["size"], y.get("context")) for x, y in zip(a, b)))
nb = [(len(x["balls"]), len(y["balls"])) for x, y in zip(a, b)]
print("same number of balls per frame", sum(p == q for p, q in nb), "of", len(nb))
d = max((abs(u - v) for x, y in zip(a, b) for bx, by in zip(x["balls"], y["balls"]) for u, v in zip(bx, by)), default=0)
print("max ball coordinate/score difference", d)
print("ball labels equal", la == lb, "meta equal", ma == mb)
import filecmp
fa = sorted((P / "t_tool" / "frames").rglob("*.jpg")); fb = sorted((P / "t_mps" / "frames").rglob("*.jpg"))
print("stored frames identical", len(fa) == len(fb) and all(filecmp.cmp(x, y, shallow=False) for x, y in zip(fa, fb)))
# backend: the tool's ONNX network against MPS, three frames
on = orig(V2, threads=4); mn = OneAtATime(None)
import cv2
from courtvision.perception.student import preprocess_frames, input_scale
imgs = [cv2.imread(str(p)) for p in fa[20:23]]
x = np.concatenate([preprocess_frames([im]) for im in imgs])[None]
s = input_scale(imgs[0].shape[1], imgs[0].shape[0])
t = time.time()
bo, po = on.detect(x, s, ball_thresh=0.05, person_thresh=0.3)
bm, pm = mn.detect(x, s, ball_thresh=0.05, person_thresh=0.3)
print("onnx vs mps:", len(bo), len(bm), "balls;", len(po), len(pm), "persons; max diff",
      max([abs(u - v) for p, q in zip(bo, bm) for u, v in zip(p, q)] + [abs(u - v) for p, q in zip(po, pm) for u, v in zip(p, q)]),
      f"{time.time() - t:.0f}s")
