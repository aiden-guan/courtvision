"""tools/student_finetune.py prep-video, line for line, with the network on the GPU in batches (mps_student.MpsNet)
instead of ONNX Runtime on the CPU: the same frames, inputs, decoding (student.decode, the dense form of the sparse
decoding: the same detections), rows, meta and ball linking. --pt is the PyTorch copy of the --model export
(student_from_onnx); meta records --model, as the tool does.
usage: prep_mps.py SRC OUT --model M.onnx --pt M.pt [--sample-every 3] [--max-frames 120] [--append] [--jpeg-quality 92] [--batch 8]"""
import argparse, json, sys, time
from pathlib import Path
from typing import Any
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from mps_student import MpsNet  # noqa: E402
import student_finetune as sf  # noqa: E402
from courtvision.perception.student import decode  # noqa: E402


def prep_video(src, out, model_path, pt, sample_every, max_frames, append=False, quality=92, batch=8):
    from courtvision.training.ball_labels import label_folder

    model = MpsNet(pt)
    det_rows, clips, old_meta = [], [], None
    if append and (out / "meta.json").is_file():
        old_meta = json.loads((out / "meta.json").read_text())
        det_rows = [json.loads(line) for line in (out / "detections.jsonl").open()]
        clips = list(old_meta["clips"])
        if old_meta.get("model") not in (None, str(model_path)):
            raise SystemExit(f"{out} was labelled by {old_meta['model']}, not {model_path}")
        if old_meta.get("jpeg_quality", 92) != quality:
            raise SystemExit(f"{out} stores frames at JPEG quality {old_meta.get('jpeg_quality', 92)}, not {quality}")
    done = {c["video"] for c in clips}
    vids = [v for v in sf._videos(src) if str(v) not in done]
    n_clips = len(clips) + len(vids)
    for ci, vid in enumerate(vids, start=len(clips)):
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS) or sf.TRAIN_FPS
        step = sf.history_step(fps)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        keep = min(max_frames, total // step)
        first = max(0, (total // step - keep) // 2) * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        t0 = time.time()
        kept = []  # (k, img)
        for k in range(keep * step):
            ok, img = cap.read()
            if not ok:
                break
            if k % step:
                continue
            kept.append((k, img))
        cap.release()
        pre = [sf.preprocess_frames([img]) for _, img in kept]
        def inp(i):  # the tool's hist: the frame and the two before it, the first frames repeated
            hist = pre[max(0, i - 2):i + 1]
            return np.concatenate([hist[max(0, len(hist) - 3 + j)] for j in range(3)])

        raws = []  # inputs built a batch at a time (60 of them would hold 1.1 GB)
        for b in range(0, len(kept), batch):
            raws += model.batch(np.stack([inp(i) for i in range(b, min(b + batch, len(kept)))]))
        del pre
        import torch
        torch.mps.empty_cache()
        for i, ((k, img), raw) in enumerate(zip(kept, raws)):
            fi = i + 1
            one = dict(raw)
            for key in ("ball_hm", "person_hm"):
                one[key] = 1.0 / (1.0 + np.exp(-raw[key]))
            balls, _ = decode(one, sf.input_scale(img.shape[1], img.shape[0]), ball_k=8, ball_thresh=0.05,
                              person_k=60, person_thresh=0.99)
            balls = [(float(bx), float(by), float(sc)) for bx, by, sc in balls]  # JSON (decode_sparse gives floats)
            r = 12.0 * img.shape[0] / 1080.0
            sf._save(img, out / "frames" / f"{ci:04d}" / f"{fi:06d}.jpg", quality)
            row: dict[str, Any] = {"clip": ci, "frame": fi, "t": (first + k) / fps,
                                   "size": [img.shape[1], img.shape[0]], "persons": [],
                                   "balls": [[bx - r, by - r, bx + r, by + r, sc] for bx, by, sc in balls]}
            if fi < 3 or fi % sample_every:
                row["context"] = True
            det_rows.append(row)
        n = len(kept)
        clips.append({"clip": ci, "video": str(vid), "first_frame": first, "frames": n, "frame_step": step})
        print(f"[{ci + 1}/{n_clips}] {vid.parent.name}/{vid.name}: {n} frames {time.time() - t0:.0f}s",
              file=sys.stderr, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in det_rows))
    meta = {"kind": "pseudo", "video": f"videos:{src.resolve()}", "person_weight": 0.0,
            "source": f"{model_path.name} on {src}", "model": str(model_path), "jpeg_quality": quality, "fps": sf.TRAIN_FPS,
            "rights_status": "research_only", "label_origin": "pseudo", "clips": clips,
            "backend": f"PyTorch MPS fp32 ({pt}, the student_from_onnx copy of {model_path}), batches of {batch}"}
    if old_meta is not None:
        meta["video"] = old_meta["video"]
        meta["source"] = "; ".join(dict.fromkeys(old_meta["source"].split("; ") + [meta["source"]]))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    rows = label_folder(out)
    (out / "ball_labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} frames, {sum(1 for r in rows if r['w'] > 0)} with a ball label", file=sys.stderr)
    return len(vids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--model", type=Path, required=True); ap.add_argument("--pt", type=Path, required=True)
    ap.add_argument("--sample-every", type=int, default=3); ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--append", action="store_true"); ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    prep_video(a.src, a.out, a.model, a.pt, a.sample_every, a.max_frames, a.append, a.jpeg_quality, a.batch)
