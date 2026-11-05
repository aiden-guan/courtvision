"""court-persons (tools/student_finetune.py) with the per-clip court fits run in parallel processes and the labeller
on the GPU (MPS fp32, the export's student_from_onnx copy). clip_courts is a
function of one clip's stored frames only, so the fits, and the labels, are those of the serial tool.
Args: SRC OUT MODEL WORKERS THREADS [extra court_persons kwargs as k=v]"""
import json, sys, time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))


def fit(paths):
    import cv2
    cv2.setNumThreads(1)
    import student_finetune as sf
    return paths, sf.clip_courts([Path(p) for p in paths])


if __name__ == "__main__":
    import student_finetune as sf
    src, out, model = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    workers, threads = int(sys.argv[4]), int(sys.argv[5])
    kw = {k: float(v) for k, v in (a.split("=") for a in sys.argv[6:])}
    rows = [json.loads(line) for line in (src / "detections.jsonl").open()]
    frames = defaultdict(list)
    for r in rows:
        frames[r["clip"]].append(r["frame"])
    jobs = [[str(src / "frames" / f"{c:04d}" / f"{f:06d}.jpg") for f in sorted(fs)] for c, fs in sorted(frames.items())]
    homs, t0 = {}, time.time()
    with get_context("spawn").Pool(workers) as pool:
        for k, (ps, hs) in enumerate(pool.imap(fit, jobs)):
            homs.update(zip(ps, hs))
            if k % 20 == 0 or k == len(jobs) - 1:
                print(f"court fits {k + 1}/{len(jobs)}: {sum(h is not None for h in hs)} of {len(hs)} frames, "
                      f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    sys.path.insert(0, str(Path(__file__).parent))
    from mps_student import MpsNet  # the labeller on the GPU: its student_from_onnx copy (.pt beside the .onnx)
    net = MpsNet(model.with_suffix(".pt"))
    sf.court_persons(src, out, model, threads=threads, model=net,
                     court_fn=lambda paths: [homs[str(p)] for p in paths], **kw)
