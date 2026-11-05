"""Fine-tune the NBA student detector on more varied broadcasts, starting from its published ONNX export.

The student (``perception/student.py``) was distilled from an RF-DETR teacher on two NBA games. On human-labelled
clips of seven other broadcasts (SportsMOT basketball, ``tools/detector_bench.py``) it finds 61% of the players.
This tool adds varied footage without losing what the network already does well:

* **players** from human boxes: SportsMOT basketball sequences (``prep-mot``; players only, so the person head also
  learns to leave the crowd, the benches and the referees alone);
* **the ball** from the network's own linked detections on NBA footage (``prep-video``): its ball candidates on
  every frame are linked into one path per clip (``training.ball_labels.link_ball_path``, which also fills fast
  flights the detector misses) and kept as pseudo labels (weight at most 0.5, ``train_student.PSEUDO_MAX_WEIGHT``).
  This anchors the ball head while the shared trunk adapts (self-training with temporal consistency, after Noisy
  Student, Xie et al. 2020);
* **the people on the court** of NBA footage from the camera geometry (``court-persons``, on a ``prep-video``
  folder): the court is fitted on each clip's stored frames and tracked through them as the browser runtime does
  (``estimate_center``, ``BroadcastCourtTracker``), a labeller that boxes everyone (the app's original model) runs on
  each sample, and its people are placed on the floor by their foot points: within ``--margin-in`` ft of the court a
  person label (players, referees), beyond ``--margin-out`` ft background (the benches, the crowd), in between an
  ``ignore`` box. A sample without a court fit (close-ups, replays) trains no person loss (``"person_weight": 0`` on
  its row). The person head learns to box the people on the court and to leave the spectators alone;
* training (``train``) from ``student_from_onnx``'s exact copy of the export, batch norms frozen (they hold the folded
  statistics), the stem and first stage frozen, AdamW with cosine decay, ``train_student``'s data loading,
  augmentation and losses. ``export`` writes the ONNX export (browser interface) and a ``.pt`` checkpoint;
* ``soup`` averages fine-tunes of the same start in weight space (optionally pulled back towards that start): one
  network, so no extra cost per frame.

The data folders use ``train_student.ClipDataset``'s layout (``frames/<clip>/<frame>.jpg`` at 960 px wide,
``detections.jsonl``, ``ball_labels.jsonl``, ``meta.json``), so ``python -m courtvision.training.train_student`` can
read them too.

    .venv/bin/python tools/student_finetune.py prep-mot data/sportsmot/train data/ft/mot_train
    .venv/bin/python tools/student_finetune.py prep-video data/bard/split/calib data/ft/bard_calib
    .venv/bin/python tools/student_finetune.py court-persons data/ft/bard_calib data/ft/bard_calib_court \\
        --model runs/models/app-v1.onnx
    .venv/bin/python tools/student_finetune.py train --data data/ft/mot_train --data data/ft/bard_calib \\
        --init runs/models/student-from-onnx.pt --out runs/models/student-ft --steps 3000
    .venv/bin/python tools/student_finetune.py export runs/models/student-ft/last.pt --out runs/models/student-ft.onnx

Research use only: SportsMOT is CC BY-NC 4.0, and NBA footage is research-only (docs/TRAINING.md §1).
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from courtvision.perception.student import (INPUT_HW, OUTPUTS, TRAIN_FPS, history_step, input_scale,  # noqa: E402
                                            preprocess_frames)

SAVE_WIDTH = 960
DEFAULT_MODEL = ROOT / "courtvision" / "server" / "static" / "nba-student.onnx"
APP_V1_MODEL = ROOT / "runs" / "models" / "app-v1.onnx"  # the first app model (checkpoint 8bdf229a): boxes everyone
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


def _save(img: np.ndarray, path: Path, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    small = cv2.resize(img, (SAVE_WIDTH, round(h * SAVE_WIDTH / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, quality])


# --------------------------------------------------------------------------- prep-mot


def prep_mot(src: Path, out: Path, sample_every: int, names: set[str] | None) -> int:
    """SportsMOT sequences -> a ClipDataset folder: every frame stored (each sample needs its two predecessors),
    every ``sample_every``-th frame a sample with the human boxes, the others context rows."""
    seqs = sorted(d for d in src.iterdir() if (d / "gt" / "gt.txt").is_file() and (names is None or d.name in names))
    det_rows, clips = [], []
    for ci, seq in enumerate(seqs):
        files = sorted(p for p in (seq / "img1").glob("*.jpg") if not p.name.startswith("._"))
        gt: dict[int, list[list[float]]] = defaultdict(list)
        for r in np.loadtxt(seq / "gt" / "gt.txt", delimiter=",", ndmin=2):
            gt[int(r[0])].append([float(r[2]), float(r[3]), float(r[2] + r[4]), float(r[3] + r[5]), 1.0])
        size = None
        for fi, f in enumerate(files, start=1):
            img = cv2.imread(str(f))
            size = size or [img.shape[1], img.shape[0]]
            _save(img, out / "frames" / f"{ci:04d}" / f"{fi:06d}.jpg")
            row: dict[str, Any] = {"clip": ci, "frame": fi, "t": (fi - 1) / 25.0, "size": size,
                                   "persons": gt.get(fi, []), "balls": []}
            if fi < 3 or fi % sample_every or fi not in gt:
                row["context"] = True
            det_rows.append(row)
        clips.append({"clip": ci, "sequence": seq.name, "frames": len(files)})
        print(f"{seq.name}: {len(files)} frames", file=sys.stderr, flush=True)
    (out / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in det_rows))
    (out / "ball_labels.jsonl").write_text("")  # no ball labels: the ball loss skips these frames
    (out / "meta.json").write_text(json.dumps({
        # "video": an identity of its own, so train_student.plan_data never takes these frames for frames of another
        # folder's video (its guard against near-duplicates of held-out clips)
        "kind": "human", "video": f"sportsmot:{src.resolve()}", "source": f"SportsMOT basketball ({src})", "fps": 25.0,
        "rights_status": "research_only",
        "license": "CC BY-NC 4.0", "label_origin": "human", "persons": "players on the court only",
        "clips": clips}, indent=1))
    return len(seqs)


# --------------------------------------------------------------------------- prep-video


def _videos(folder: Path) -> list[Path]:
    out = []
    for d, _, files in os.walk(folder, followlinks=True):
        out += [Path(d) / f for f in files if Path(f).suffix.lower() in VIDEO_EXT and not f.startswith("._")]
    return sorted(out)


def prep_video(src: Path, out: Path, model_path: Path, sample_every: int, max_frames: int, threads: int,
               append: bool = False, quality: int = 92) -> int:
    """Videos -> a pseudo ClipDataset folder with the model's own linked ball labels (no person labels). With
    ``append`` the clips are added after those already in ``out`` (one folder, and one ``video`` identity, filled
    batch by batch: the videos of a batch can then be deleted before the next is fetched), and every clip's labels
    are linked again."""
    from courtvision.perception.student_onnx import OnnxStudent
    from courtvision.training.ball_labels import label_folder

    model = OnnxStudent(model_path, threads=threads)
    det_rows: list[dict[str, Any]] = []
    clips: list[dict[str, Any]] = []
    old_meta: dict[str, Any] | None = None
    if append and (out / "meta.json").is_file():
        old_meta = json.loads((out / "meta.json").read_text())
        det_rows = [json.loads(line) for line in (out / "detections.jsonl").open()]
        clips = list(old_meta["clips"])
        if old_meta.get("model") not in (None, str(model_path)):
            raise SystemExit(f"{out} was labelled by {old_meta['model']}, not {model_path}")
        if old_meta.get("jpeg_quality", 92) != quality:
            raise SystemExit(f"{out} stores frames at JPEG quality {old_meta.get('jpeg_quality', 92)}, not {quality}")
    done = {c["video"] for c in clips}
    vids = [v for v in _videos(src) if str(v) not in done]
    n_clips = len(clips) + len(vids)
    for ci, vid in enumerate(vids, start=len(clips)):
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS) or TRAIN_FPS
        step = history_step(fps)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        keep = min(max_frames, total // step)
        first = max(0, (total // step - keep) // 2) * step  # the middle of the clip
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        hist: list[np.ndarray] = []
        t0 = time.time()
        n = 0
        for k in range(keep * step):
            ok, img = cap.read()
            if not ok:
                break
            if k % step:
                continue
            n += 1
            fi = n
            hist = (hist + [preprocess_frames([img])])[-3:]
            x = np.concatenate([hist[max(0, len(hist) - 3 + j)] for j in range(3)])[None]
            balls, _ = model.detect(x, input_scale(img.shape[1], img.shape[0]), ball_thresh=0.05, person_thresh=0.99)
            r = 12.0 * img.shape[0] / 1080.0
            _save(img, out / "frames" / f"{ci:04d}" / f"{fi:06d}.jpg", quality)
            row: dict[str, Any] = {"clip": ci, "frame": fi, "t": (first + k) / fps,
                                   "size": [img.shape[1], img.shape[0]], "persons": [],
                                   "balls": [[bx - r, by - r, bx + r, by + r, sc] for bx, by, sc in balls]}
            if fi < 3 or fi % sample_every:
                row["context"] = True
            det_rows.append(row)
        cap.release()
        clips.append({"clip": ci, "video": str(vid), "first_frame": first, "frames": n, "frame_step": step})
        print(f"[{ci + 1}/{n_clips}] {vid.parent.name}/{vid.name}: {n} frames {time.time() - t0:.0f}s",
              file=sys.stderr, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in det_rows))
    meta = {"kind": "pseudo", "video": f"videos:{src.resolve()}", "person_weight": 0.0,
            "source": f"{model_path.name} on {src}", "model": str(model_path), "jpeg_quality": quality, "fps": TRAIN_FPS,
            "rights_status": "research_only", "label_origin": "pseudo", "clips": clips}
    if old_meta is not None:  # the folder keeps its identity; each batch's source is listed
        meta["video"] = old_meta["video"]
        meta["source"] = "; ".join(dict.fromkeys(old_meta["source"].split("; ") + [meta["source"]]))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    rows = label_folder(out)
    (out / "ball_labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    labelled = sum(1 for r in rows if r["w"] > 0)
    print(f"{len(rows)} frames, {labelled} with a ball label", file=sys.stderr)
    return len(vids)


# --------------------------------------------------------------------------- prep-coco


def prep_coco(ann: Path, images: Path, out: Path, games: list[str], exclude: list[str],
              referees: bool = False) -> int:
    """COCO-format human labels (a Roboflow export: ``player``, ``player-<action>``, ``referee``, ``ball``, ``rim``)
    -> a ClipDataset folder: every image its own one-frame clip (the network's earlier input frames are then the frame
    itself: the export's frames are too far apart for motion), players (every ``player*`` class) as person labels with
    score 1, the human ball box centre as a ball label of weight 1. Referees are left out, as in SportsMOT, unless
    ``referees``: then they are people too (the crowd, benches and coaches, never labelled, stay background).
    ``games``: file-name prefixes to keep (all when empty); ``exclude``: prefixes to drop."""
    import shutil

    d = json.loads(ann.read_text())
    cats = {c["id"]: c["name"] for c in d["categories"]}
    by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    det_rows, lab_rows, n = [], [], 0
    for im in sorted(d["images"], key=lambda i: i["file_name"]):
        name = im["file_name"]
        if (games and not any(name.startswith(g) for g in games)) or any(name.startswith(g) for g in exclude):
            continue
        src = images / name
        if not src.is_file():
            continue
        ci = n
        n += 1
        dst = out / "frames" / f"{ci:04d}" / f"{1:06d}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        persons, ball = [], None
        for a in by_img.get(im["id"], []):
            x, y, w, h = a["bbox"]
            c = cats[a["category_id"]]
            if c == "player" or c.startswith("player-") or (referees and c == "referee"):
                persons.append([x, y, x + w, y + h, 1.0])
            elif c == "ball" and ball is None:
                ball = [x + w / 2.0, y + h / 2.0]
        det_rows.append({"clip": ci, "frame": 1, "size": [im["width"], im["height"]], "persons": persons,
                         "balls": [], "source": name})
        lab_rows.append({"clip": ci, "frame": 1, "xy": ball, "w": 1.0 if ball else 0.0,
                         "source": "human" if ball else "none"})
    out.mkdir(parents=True, exist_ok=True)
    (out / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in det_rows))
    (out / "ball_labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lab_rows))
    (out / "meta.json").write_text(json.dumps({
        "kind": "human", "video": f"coco:{ann.resolve()}:{','.join(games) or 'all'}", "source": str(ann),
        "games": games, "exclude": exclude, "fps": TRAIN_FPS, "rights_status": "research_only",
        "license": "labels CC BY 4.0 (Roboflow Universe); footage NBA", "label_origin": "human",
        "persons": "players and referees" if referees else "players only (referees left out)"}, indent=1))
    print(f"{n} images, {sum(1 for r in lab_rows if r['w'] > 0)} with a ball", file=sys.stderr)
    return n


# --------------------------------------------------------------------------- court-people


COURT_LENGTH_FT, COURT_WIDTH_FT = 94.0, 50.0


def court_zone(foot_ft: tuple[float, float], keep_ft: float = 3.0, ignore_ft: float = 10.0) -> str:
    """Where a person's foot point (court ft: origin at a baseline-sideline corner) stands: "court" within ``keep_ft``
    of the lines (players, referees, a player inbounding), "near" up to ``ignore_ft`` (courtside seats, the bench's
    edge: unsure), else "off" (the benches and the crowd)."""
    x, y = foot_ft
    out = max(-x, x - COURT_LENGTH_FT, -y, y - COURT_WIDTH_FT, 0.0)
    return "court" if out <= keep_ft else "near" if out <= ignore_ft else "off"


def court_people(src: Path, game: Path, out: Path, keep_ft: float = 3.0, ignore_ft: float = 10.0,
                 min_score: float = 0.4) -> dict[str, int]:
    """A teacher folder of a game the camera track covers (``tools/nba_teacher_labels.py``: the teacher boxes everyone,
    the crowd and benches included) -> the same folder (``out``: frames linked, not copied) whose people are the ones
    on the court: each teacher box at ``min_score`` or more is placed through the frame's camera (its foot point,
    ``court_zone``) and kept ("court"), left out of the person loss as an ``ignore`` box ("near"), or dropped, so that
    it is background ("off"). On a frame without a camera fit (replays, close-ups) every box is an ``ignore`` box.
    Boxes under ``min_score`` (``ClipDataset``'s ``person_min_score``: never a target) are dropped. The ball labels
    are copied unchanged. Returns the counts per kind."""
    import shutil

    from courtvision.nba.analyze import CameraTrack

    cam = CameraTrack.load(sorted((game / "camera_parts").glob("*.jsonl")))
    counts = {"court": 0, "near": 0, "off": 0, "no_fit": 0, "weak": 0}
    rows = []
    for r in map(json.loads, (src / "detections.jsonl").open()):
        H = cam.homography(int(r["frame"]))
        keep, ignore = [], list(r.get("ignore") or [])
        for q in r.get("persons", []):
            if q[4] < min_score:
                counts["weak"] += 1
                continue
            if H is None:
                counts["no_fit"] += 1
                ignore.append([round(float(v), 1) for v in q[:4]])
                continue
            fx, fy = H.image_to_court(np.array([[0.5 * (q[0] + q[2]), q[3]]], float))[0]
            zone = court_zone((float(fx), float(fy)), keep_ft, ignore_ft)
            counts[zone] += 1
            if zone == "court":
                keep.append(q)
            elif zone == "near":
                ignore.append([round(float(v), 1) for v in q[:4]])
        r["persons"] = keep
        if ignore:
            r["ignore"] = ignore
        rows.append(r)
    out.mkdir(parents=True, exist_ok=True)
    (out / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    shutil.copyfile(src / "ball_labels.jsonl", out / "ball_labels.jsonl")
    if not (out / "frames").exists():
        (out / "frames").symlink_to((src / "frames").resolve(), target_is_directory=True)
    meta = json.loads((src / "meta.json").read_text()) if (src / "meta.json").is_file() else {"kind": "teacher"}
    meta.setdefault("video", f"teacher:{src.resolve()}")
    meta["court_people"] = {"game": str(game), "keep_ft": keep_ft, "ignore_ft": ignore_ft, "min_score": min_score,
                            "counts": counts, "what": "people on the court kept; near it ignored; the crowd background"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{out}: {counts}", file=sys.stderr)
    return counts


# --------------------------------------------------------------------------- court-persons


def clip_courts(paths: list[Path], court: Any = None) -> list[Any]:
    """The court in each stored frame of a clip (``paths``, in order), found as the browser runtime finds it
    (``browser/runtime.py``): ``estimate_center`` on six frames spread over the clip (the browser's choice for a clip
    of up to 5 s), a ``BroadcastCourtTracker`` restarted from the first camera found on each frame until it fits one,
    then tracked through the frames with the cuts of ``ShotBoundaryDetector`` (as ``BroadcastPerception.gate`` drives
    it). Per frame the image -> court homography (stored px -> court ft), or None: no camera for the clip (close-ups,
    replays, too few court views), the court lost, or a camera only held from an earlier frame (``held``: this frame's
    lines did not confirm it)."""
    from courtvision.geometry.court import court_for
    from courtvision.perception.broadcast import ShotBoundaryDetector
    from courtvision.perception.broadcast_court import BroadcastCourtTracker, estimate_center

    court = court or court_for("nba")
    if not paths:
        return []
    views = [cv2.imread(str(paths[i])) for i in sorted({round(k * (len(paths) - 1) / 5) for k in range(6)})]
    center, cams = estimate_center(views, court)
    initial = next((c for c in cams if c is not None), None)
    if center is None or initial is None:
        return [None] * len(paths)
    size = (views[0].shape[1], views[0].shape[0])
    tracker, cuts, out = None, ShotBoundaryDetector(), []
    for p in paths:
        img = cv2.imread(str(p))
        if tracker is None:  # BrowserPipeline.prepare: the first camera, restarted on each frame until it fits
            first = BroadcastCourtTracker(court, size, center)
            first.reset(initial)
            if first.update(img, cut=True).homography(size) is None:
                out.append(None)
                continue
            tracker = first  # then the broadcast gate updates it on that frame too
        fix = tracker.update(img, cut=cuts.update(img))
        out.append(fix.homography(size) if fix.status in ("tracking", "reacquired") else None)
    return out


def court_split(persons: list[Any], hom: Any, scale: float, margin_in: float = 2.0,
                margin_out: float = 8.0) -> dict[str, list[list[float]]]:
    """``persons`` (x1, y1, x2, y2, ... in label px) by where each one's foot point (its box's bottom centre) stands
    on the floor; ``hom`` maps stored px (``scale`` of them per label px) to court ft. "court": inside the court
    grown by ``margin_in`` ft; "off": more than ``margin_out`` ft outside it; "near": in between; "unplaced": not on
    the floor in front of the camera (above the horizon). Boxes as [x1, y1, x2, y2] in label px (0.1 px)."""
    zones: dict[str, list[list[float]]] = {"court": [], "near": [], "off": [], "unplaced": []}
    for q in persons:
        foot = np.array([[0.5 * (q[0] + q[2]) * scale, q[3] * scale]], float)
        fx, fy = (float(v) for v in hom.image_to_court(foot, front_only=True)[0])
        zone = court_zone((fx, fy), margin_in, margin_out) if math.isfinite(fx) and math.isfinite(fy) else "unplaced"
        zones[zone].append([round(float(v), 1) for v in q[:4]])
    return zones


def court_persons(src: Path, out: Path, model_path: Path, score: float = 0.3, margin_in: float = 2.0,
                  margin_out: float = 8.0, threads: int = 1, model: Any = None,
                  court_fn: Callable[[list[Path]], list[Any]] | None = None, weak: float = 0.1) -> dict[str, int]:
    """A pseudo folder of NBA broadcast clips (``prep-video``: ball labels, no person labels) -> the same folder
    (``out``: its frames linked, not copied) with person labels from the camera geometry. The court is fitted on each
    clip's stored frames and tracked through them (``court_fn``, default ``clip_courts``: per frame a stored px ->
    court ft mapping or None). On every sample with a court, ``model_path`` (a detector that boxes everyone, e.g. the
    app's original model) runs on the sample's own network input (``ClipDataset``: its two earlier frames are the real
    ones), and each person at ``score`` or more is placed by its foot point (``court_split``): on the court grown by
    ``margin_in`` ft (players, referees, a player inbounding) a person label of score 1; more than ``margin_out`` ft
    off it (the benches, the crowd) left out, so background; in between, or above the horizon, an ``ignore`` box. A
    person the labeller is unsure of (``weak`` to ``score``; often a player it half misses) is an ``ignore`` box
    wherever it stands, so it is not taught as background. A sample without a court (close-ups, replays, no fit) gets ``"person_weight": 0`` on its row: no person loss
    (``train_student.ClipDataset``). The folder's persons weigh ``PSEUDO_MAX_WEIGHT``; the ball labels are copied
    unchanged. ``model``: an already loaded ``detector_bench`` model (tests). Returns the counts."""
    import shutil

    sys.path.insert(0, str(ROOT / "tools"))
    from detector_bench import load_model, run_model  # noqa: E402

    from courtvision.training.train_student import PSEUDO_MAX_WEIGHT, ClipDataset, folder_meta, is_pseudo

    meta = folder_meta(src)
    rows = [json.loads(line) for line in (src / "detections.jsonl").open()]
    if not is_pseudo(meta) or any(r.get("persons") for r in rows):
        raise SystemExit(f"{src}: court-persons labels a pseudo folder without person labels (prep-video's)")
    if out.resolve() == src.resolve():
        raise SystemExit("OUT must be a new folder, not SRC")
    if not 0.0 <= margin_in <= margin_out:
        raise SystemExit(f"--margin-in ({margin_in}) must be between 0 and --margin-out ({margin_out})")
    if not 0.0 <= weak <= score:
        raise SystemExit(f"--weak ({weak}) must be between 0 and --score ({score})")
    if model is None:
        model = load_model(model_path, threads, sparse=True)
    court_fn = court_fn or clip_courts
    ds = ClipDataset([src], None, train=False)
    by_key = {(r["clip"], r["frame"]): r for r in rows}
    frames: dict[int, list[int]] = defaultdict(list)  # clip -> its stored frames
    for clip, frame in by_key:
        frames[clip].append(frame)
    samples: dict[int, list[int]] = defaultdict(list)  # clip -> its samples (indices of ds.items)
    for i, item in enumerate(ds.items):
        samples[item[1]].append(i)
    counts = {"samples": len(ds), "fit": 0, "persons": 0, "ignored": 0, "unplaced": 0, "off_court": 0, "weak": 0}
    t0 = time.time()
    for k, clip in enumerate(sorted(samples)):
        fr = sorted(frames[clip])
        homs = dict(zip(fr, court_fn([src / "frames" / f"{clip:04d}" / f"{f:06d}.jpg" for f in fr]), strict=True))
        fit = 0
        for i in samples[clip]:
            _, _, inputs, det, _ = ds.items[i]
            row, hom = by_key[(clip, inputs[-1])], homs[inputs[-1]]
            if hom is None:
                row["person_weight"] = 0.0  # no court: the sample trains no person loss
                continue
            x, t = ds[i]
            _, found = run_model(model, x[None], tuple(det["size"]), 0.99, min(weak, score))
            zones = court_split([q for q in found if q[4] >= score], hom, float(t["scale"][0]), margin_in, margin_out)
            unsure = [[round(float(v), 1) for v in q[:4]] for q in found if weak <= q[4] < score]
            ignore = zones["near"] + zones["unplaced"] + unsure
            row["persons"] = [b + [1.0] for b in zones["court"]]
            if ignore:
                row["ignore"] = list(row.get("ignore") or []) + ignore
            fit += 1
            counts["persons"] += len(zones["court"])
            counts["ignored"] += len(ignore)
            counts["unplaced"] += len(zones["unplaced"])
            counts["off_court"] += len(zones["off"])
            counts["weak"] += len(unsure)
        counts["fit"] += fit
        print(f"[{k + 1}/{len(samples)}] clip {clip}: a court on {fit} of {len(samples[clip])} samples; in all "
              f"{counts['persons']} people on it, {counts['ignored']} ignored, {counts['off_court']} off it, "
              f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    shutil.copyfile(src / "ball_labels.jsonl", out / "ball_labels.jsonl")
    if not (out / "frames").exists():
        (out / "frames").symlink_to((src / "frames").resolve(), target_is_directory=True)
    meta["person_weight"] = PSEUDO_MAX_WEIGHT  # the most a pseudo folder's persons weigh (train_student.folder_weights)
    meta["court_persons"] = {
        "model": str(model_path), "score": score, "weak": weak, "margin_in": margin_in, "margin_out": margin_out,
        **counts,
        "what": "people whose foot point is on the court grown by margin_in ft (players, referees) labelled through "
                "the clip's own court fit; margin_in to margin_out ft off it, above the horizon, or scored weak to "
                "score, ignored; farther "
                "(the benches, the crowd) background; samples without a court fit train no person loss"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{out}: {counts}", file=sys.stderr)
    return counts


# --------------------------------------------------------------------------- mark-ignore


def foot_hull_ignore(players: np.ndarray, found: np.ndarray, scores: np.ndarray, size: tuple[int, int],
                     iou: float = 0.3, score: float = 0.3, margin: float = 0.05,
                     referee: np.ndarray | None = None) -> list[list[float]]:
    """The boxes of ``found`` (people another model detected, label px, with ``scores``) that a player-only label set
    leaves out although they are on the court: at ``score`` or more, overlapping no labelled player at ``iou`` or
    more, and either standing among the players (foot point, the bottom centre, inside the convex hull of the
    labelled players' foot points grown by ``margin`` of the frame height) or read as a referee (``referee``: one
    flag per box, ``nba.referee``; referees stand at the sidelines, mostly outside that hull). People in the stands
    stay background. Fewer than 3 players: only the referees."""
    if len(found) == 0:
        return []
    if len(players) < 3:
        if referee is None or len(players) == 0:
            return []
        sys.path.insert(0, str(ROOT / "tools"))
        from detector_bench import iou_matrix  # noqa: E402

        over = iou_matrix(found, players).max(axis=1)
        return [[round(float(v), 1) for v in box[:4]] for box, sc, o, r in zip(found, scores, over, referee)
                if sc >= score and o < iou and r]
    sys.path.insert(0, str(ROOT / "tools"))
    from detector_bench import iou_matrix  # noqa: E402

    feet = np.stack([(players[:, 0] + players[:, 2]) / 2.0, players[:, 3]], 1).astype(np.float32)
    hull = cv2.convexHull(feet)
    over = iou_matrix(found, players).max(axis=1) if len(players) else np.zeros(len(found))
    out = []
    for k, (box, sc, o) in enumerate(zip(found, scores, over)):
        if sc < score or o >= iou:
            continue
        foot = (float((box[0] + box[2]) / 2.0), float(box[3]))
        if cv2.pointPolygonTest(hull, foot, True) >= -margin * size[1] or (referee is not None and referee[k]):
            out.append([round(float(v), 1) for v in box[:4]])
    return out


@dataclass(frozen=True)
class Promote:
    """Which unlabelled people ``mark-ignore --promote`` makes person labels (``promote_officials``). With these
    defaults and the first app model as the labeller, 674 of 679 random promoted boxes of SportsMOT's 7 broadcasts
    (up to 100 a broadcast, judged by eye) were officials, the rest coaches or bench staff at the sideline. Combined
    with the learned check (``either``), about 12 of 100 drawn on one Lakers at New Orleans clip were fans seated
    courtside; under a labeller score of 0.4 the shirt rule let in a bench in dark tracksuits and coaches in suits."""

    referee: str = "shirt"  # the referee check: shirt, learned, either or both (``nba.referee.referee_calls``)
    iou: float = 0.3  # a box overlapping a labelled player at this IoU or more is that player
    score: float = 0.4  # the labeller's score a box needs (``ClipDataset``'s person_min_score for machine boxes)
    ratio: tuple[float, float] = (0.8, 1.25)  # box height over ``standing_height`` at its foot row
    above: float = 0.1  # foot row at most this share of the frame height above the farthest player's
    border: float = 0.01  # the box keeps this share of the frame height clear of every image edge
    min_players: int = 4  # labelled players a frame needs for the height fit


def standing_height(players: np.ndarray, min_players: int = 4) -> tuple[float, float] | None:
    """How tall a person standing among the labelled ``players`` of one frame (label px) is at each foot row: (a, b)
    of ``height = a + b * foot_row`` (the box bottom), a Theil-Sen line (the median of the pairwise slopes, then the
    median intercept), which a few crouching, jumping or occluded players do not move. Over a flat floor a standing
    person's image height grows about linearly with the row the feet stand on (Hoiem et al. 2006, "Putting Objects
    in Perspective"). None with fewer than ``min_players`` players or no spread of foot rows."""
    if len(players) < max(2, min_players):
        return None
    y = players[:, 3].astype(float)
    h = (players[:, 3] - players[:, 1]).astype(float)
    i, j = np.triu_indices(len(players), 1)
    dy = y[j] - y[i]
    ok = np.abs(dy) > 1.0
    if not ok.any():
        return None
    b = float(np.median((h[j] - h[i])[ok] / dy[ok]))
    return float(np.median(h - b * y)), b


def promote_officials(players: np.ndarray, found: np.ndarray, scores: np.ndarray, size: tuple[int, int],
                      calls: Any, rule: Promote = Promote()) -> list[str]:
    """Which boxes of ``found`` (people another model detected, label px, with ``scores``) a player-only label set
    leaves out although they are officials: one entry per box, "" for a box to promote, else the first test it fails.
    A box is promoted when it overlaps no labelled player (IoU under ``rule.iou``; else "player"), the labeller is
    sure enough of it (``rule.score``; else "score"), the frame has a height fit (``standing_height``; else "fit"),
    its foot row is at most ``rule.above`` of the frame height above the farthest player's (else "above": the
    stands), its height is within ``rule.ratio`` of the fit's at its foot row (else "height": seated fans, the
    benches, the raised stands), it keeps ``rule.border`` of the frame height clear of every image edge (else
    "border": people cut off by the frame) and the referee check calls it a referee (``calls``: one flag per box;
    else "referee")."""
    if len(found) == 0:
        return []
    calls = list(calls)
    if len(calls) != len(found) or len(scores) != len(found):
        raise ValueError(f"{len(calls)} referee calls and {len(scores)} scores for {len(found)} boxes")
    sys.path.insert(0, str(ROOT / "tools"))
    from detector_bench import iou_matrix  # noqa: E402

    over = iou_matrix(found, players).max(axis=1) if len(players) else np.zeros(len(found))
    fit = standing_height(players, rule.min_players)
    top = float(players[:, 3].min()) - rule.above * size[1] if len(players) else -math.inf
    pad = rule.border * size[1]
    out = []
    for box, sc, o, ref in zip(found, scores, over, calls):
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        want = fit[0] + fit[1] * y2 if fit is not None else 0.0
        if o >= rule.iou:
            out.append("player")
        elif sc < rule.score:
            out.append("score")
        elif fit is None or want <= 1.0:
            out.append("fit")
        elif y2 < top:
            out.append("above")
        elif not rule.ratio[0] <= (y2 - y1) / want <= rule.ratio[1]:
            out.append("height")
        elif x1 < pad or y1 < pad or x2 > size[0] - pad or y2 > size[1] - pad:
            out.append("border")
        elif not ref:
            out.append("referee")
        else:
            out.append("")
    return out


def referee_flags(frame_bgr: np.ndarray, boxes: list[tuple[float, float, float, float]], method: str) -> np.ndarray:
    """``nba.referee.referee_calls`` for ``Promote.referee``: ``shirt``, ``learned``, or the two combined (``either``:
    one of them calls a referee; ``both``: the two do)."""
    from courtvision.nba.referee import referee_calls

    if method in ("shirt", "learned"):
        return np.array(referee_calls(frame_bgr, boxes, method), bool).reshape(-1)
    if method not in ("either", "both"):
        raise ValueError(f"unknown referee method {method!r}")
    shirt = np.array(referee_calls(frame_bgr, boxes, "shirt"), bool).reshape(-1)
    learned = np.array(referee_calls(frame_bgr, boxes, "learned"), bool).reshape(-1)
    return shirt | learned if method == "either" else shirt & learned


def mark_ignore(src: Path, out: Path, model_path: Path, iou: float = 0.3, score: float = 0.3, margin: float = 0.05,
                threads: int = 4, model: Any = None, referees: bool = True, promote: Promote | None = None) -> int:
    """A player-only folder (SportsMOT, ``prep-mot``) -> the same folder (``out``: its frames linked, not copied)
    whose rows carry ``ignore`` boxes (``foot_hull_ignore``) found by ``model_path`` (default: the app's export,
    which boxes everyone) on the network's own input of every sample: ``train_student.ClipDataset`` then leaves
    those people out of the person heatmap loss, so the referees the labels skip are not taught as background.
    ``referees``: also the people the referee check (``nba.referee``) reads as referees, wherever they stand.
    ``promote``: the people ``promote_officials`` passes become person labels instead (score 1, their indices in the
    row's ``promoted``), whether they would have been ignore boxes or background; everyone else is treated as without
    it. ``model``: an already loaded ``detector_bench`` model (tests). Returns the number of ignore boxes."""
    import shutil

    sys.path.insert(0, str(ROOT / "tools"))
    from detector_bench import load_model, run_model  # noqa: E402

    from courtvision.training.train_student import ClipDataset, plan_data

    if model is None:
        model = load_model(model_path, threads, sparse=True)
    plan = plan_data([str(src)])
    ds = ClipDataset(plan.folders, plan.train | plan.val, train=False, weights=plan.weights)
    src_meta = json.loads((src / "meta.json").read_text())
    sequence = {c["clip"]: c.get("sequence", str(c["clip"])) for c in src_meta.get("clips", [])}
    marks: dict[tuple[int, int], list[list[float]]] = {}
    ups: dict[tuple[int, int], list[list[float]]] = {}
    why: Counter[str] = Counter()
    by_seq: Counter[str] = Counter()
    up_ignored = 0  # promoted boxes that would have been ignore boxes (the rest would have been background)
    n_rows = 0
    for i, (fi, clip, frames, det, _) in enumerate(ds.items):
        players = np.array([q[:4] for q in det["persons"]], float).reshape(-1, 4)
        if len(players) == 0:
            continue
        x, _ = ds[i]
        size = tuple(det.get("size") or (1280, 720))
        _, persons = run_model(model, x[None], size, 0.99, score)
        found = np.array([q[:4] for q in persons], float).reshape(-1, 4)
        scores = np.array([q[4] for q in persons], float)
        ref = None
        img, stored = None, []
        if (referees or promote is not None) and len(found):
            img = ds._img(fi, clip, frames[-1])
            k = img.shape[1] / float(size[0])  # stored px per label px
            stored = [tuple(float(v) * k for v in b) for b in found]
        if referees and len(found):
            from courtvision.nba import referee_model
            from courtvision.nba.referee import referee_probs

            ref = referee_probs(img, stored, referee_model) >= referee_model.THRESHOLD
        up = np.zeros(len(found), bool)
        if promote is not None and len(found):
            tests = promote_officials(players, found, scores, size, referee_flags(img, stored, promote.referee),
                                      promote)
            why.update(t or "promoted" for t in tests)
            up = np.array([t == "" for t in tests])
            if up.any():
                ups[(clip, frames[-1])] = [[round(float(v), 1) for v in b[:4]] + [1.0] for b in found[up]]
                by_seq[sequence.get(clip, str(clip))] += int(up.sum())
                up_ignored += len(foot_hull_ignore(players, found[up], scores[up], size, iou, score, margin,
                                                   None if ref is None else ref[up]))
        boxes = foot_hull_ignore(players, found[~up], scores[~up], size, iou, score, margin,
                                 None if ref is None else ref[~up])
        n_rows += 1
        if boxes:
            marks[(clip, frames[-1])] = boxes
        if n_rows % 500 == 0:
            print(f"{n_rows} samples, {sum(len(v) for v in marks.values())} ignore boxes, "
                  f"{sum(len(v) for v in ups.values())} promoted", file=sys.stderr, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (src / "detections.jsonl").open()]
    for r in rows:
        if (r["clip"], r["frame"]) in marks:
            r["ignore"] = marks[(r["clip"], r["frame"])]
        if (r["clip"], r["frame"]) in ups:
            n_players = len(r["persons"])
            r["persons"] = r["persons"] + ups[(r["clip"], r["frame"])]
            r["promoted"] = list(range(n_players, len(r["persons"])))
    (out / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    shutil.copyfile(src / "ball_labels.jsonl", out / "ball_labels.jsonl")
    if not (out / "frames").exists():
        (out / "frames").symlink_to((src / "frames").resolve(), target_is_directory=True)
    meta = src_meta
    n = sum(len(v) for v in marks.values())
    meta["ignore"] = {"model": str(model_path), "iou": iou, "score": score, "margin": margin, "referees": referees,
                      "boxes": n, "rows": len(marks), "samples": n_rows,
                      "what": "unlabelled people among the labelled players or read as referees: no person loss"}
    if promote is not None:
        n_up = sum(len(v) for v in ups.values())
        meta["persons"] = f"{meta.get('persons', 'players')}; and the officials promoted from {Path(model_path).name}"
        meta["promote"] = {
            "model": str(model_path), "rule": asdict(promote), "boxes": n_up, "rows": len(ups),
            "were_ignore": up_ignored, "were_background": n_up - up_ignored,
            "tests": {k: why[k] for k in ("promoted", "player", "score", "fit", "above", "height", "border",
                                          "referee")},
            "by_sequence": dict(sorted(by_seq.items())),
            "what": "unlabelled people standing among the players, clear of the image border, called referees: person "
                    "labels with score 1 (the row's promoted indices into persons)"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{out}: {n} ignore boxes on {len(marks)} of {n_rows} samples"
          + (f", {sum(len(v) for v in ups.values())} promoted on {len(ups)}" if promote is not None else ""),
          file=sys.stderr)
    return n


# --------------------------------------------------------------------------- prep-teacher


def prep_teacher(src: Path, out: Path, frames: int, weights: str | None, shape: str, device: str = "cpu") -> int:
    """Videos -> a teacher folder labelled as ``tools/nba_teacher_labels.py`` labels a game: RF-DETR Small (COCO) on
    every stored frame (the middle ``frames`` of each video, at the student's input spacing) at ``shape``, persons
    from 0.25, every ball candidate, then the linked ball labels. For held-out ball checks (``detector_bench.py
    --ball-folder``) on footage the student never trained on."""
    import warnings

    import torch

    from courtvision.training.ball_labels import label_folder

    warnings.filterwarnings("ignore")
    import rfdetr

    threads = os.environ.get("OMP_NUM_THREADS", "")
    torch.set_num_threads(int(threads) if threads.isdigit() and int(threads) > 0 else max(1, os.cpu_count() or 2))
    model = rfdetr.RFDETRSmall(pretrain_weights=weights, device=device) if weights else rfdetr.RFDETRSmall(device=device)
    hw = tuple(int(v) for v in shape.split(","))
    det_rows, clips = [], []
    vids = _videos(src)
    t_all = time.time()
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "detections.partial.jsonl"  # one line per frame as it is labelled: an interrupted run resumes
    done = {(r["clip"], r["frame"]): r for r in map(json.loads, partial.open())} if partial.is_file() else {}
    sink = partial.open("a")
    for ci, vid in enumerate(vids):
        if any(c == ci for c, _ in done) and sum(1 for c, _ in done if c == ci) >= frames:
            det_rows += [r for (c, _), r in sorted(done.items()) if c == ci]
            clips.append({"clip": ci, "video": str(vid), "resumed": True})
            continue
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS) or TRAIN_FPS
        step = history_step(fps)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        keep = min(frames, total // step)
        first = max(0, (total // step - keep) // 2) * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        n = 0
        for k in range(keep * step):
            ok, img = cap.read()
            if not ok:
                break
            if k % step:
                continue
            n += 1
            ten = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)
            with torch.inference_mode():
                det = model.predict(ten, threshold=0.04, shape=hw, include_source_image=False)
            names = det.data.get("class_name")
            persons, balls = [], []
            for i in range(len(det.xyxy)):
                box = [round(float(v), 1) for v in det.xyxy[i]]
                sc = round(float(det.confidence[i]), 3)
                if names[i] == "person" and sc >= 0.25:
                    persons.append(box + [sc])
                elif names[i] == "sports ball":
                    balls.append(box + [sc])
            _save(img, out / "frames" / f"{ci:04d}" / f"{n:06d}.jpg")
            row = {"clip": ci, "frame": n, "t": round((first + k) / fps, 4), "size": [img.shape[1], img.shape[0]],
                   "persons": persons, "balls": balls}
            det_rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
        cap.release()
        clips.append({"clip": ci, "video": str(vid), "first_frame": first, "frames": n, "frame_step": step})
        el = time.time() - t_all
        print(f"[{ci + 1}/{len(vids)}] {vid.parent.name}/{vid.name}: {n} frames, "
              f"{el / max(1, len(det_rows)) * 1000:.0f} ms/frame", file=sys.stderr, flush=True)
    sink.close()
    (out / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in det_rows))
    (out / "meta.json").write_text(json.dumps({
        "kind": "teacher", "video": f"videos:{src.resolve()}", "teacher": f"RF-DETR Small (COCO) at {shape}",
        "source": str(src), "fps": TRAIN_FPS,
        "rights_status": "research_only", "label_origin": "cv_auto", "clips": clips}, indent=1))
    rows = label_folder(out)
    (out / "ball_labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} frames, {sum(1 for r in rows if r['w'] >= 0.5)} with a ball label of weight >= 0.5",
          file=sys.stderr)
    return len(vids)


# --------------------------------------------------------------------------- train


def _quick_mot_eval(net: Any, seqs: list[Any], frames_per_seq: int) -> dict[str, float]:
    """Person AP50 / recall at 0.3 on a few SportsMOT frames (``detector_bench``'s matching)."""
    import torch

    sys.path.insert(0, str(ROOT / "tools"))
    from detector_bench import average_precision, greedy_match  # noqa: E402

    from courtvision.perception.student import decode

    scores, tps, n_truth = [], [], 0
    net.eval()
    with torch.inference_mode():
        for seq in seqs:
            fr = [f for f in sorted(seq.gt) if f >= 3]
            for f in fr[:: max(1, len(fr) // frames_per_seq)][:frames_per_seq]:
                imgs = [cv2.imread(str(seq.frames[g - 1])) for g in (f - 2, f - 1, f)]
                x = torch.from_numpy(preprocess_frames(imgs)[None]).to(next(net.parameters()).device)
                out = {k: v[0].float().cpu().numpy() for k, v in net(x).items()}
                for k in ("ball_hm", "person_hm"):
                    out[k] = 1.0 / (1.0 + np.exp(-out[k]))
                _, persons = decode(out, input_scale(*seq.size), person_thresh=0.05, person_k=100)
                boxes = np.array([p[:4] for p in persons]).reshape(-1, 4)
                sc = np.array([p[4] for p in persons])
                tps += greedy_match(boxes, sc, seq.gt[f]).tolist()
                scores += sc.tolist()
                n_truth += len(seq.gt[f])
    s, t = np.array(scores), np.array(tps, bool)
    return {"ap50": round(average_precision(s, t, n_truth), 4),
            "recall@0.3": round(float((t & (s >= 0.3)).sum()) / max(1, n_truth), 4)}


def finetune_dataset(base: Any, hue_deg: float, static_p: float = 0.0) -> Any:
    """``train_student.ClipDataset`` with domain randomisation for the person head: a sample without a ball label
    has its three frames' hues turned by one random angle in [-hue_deg, hue_deg] (jerseys, floors and adverts
    change colour from game to game; a frame with a ball label keeps its colours, the ball's orange being part
    of what the network knows a ball by). ``base`` must be ``ClipDataset``; the class is defined at module level
    (``FineTuneDataset``) so that DataLoader workers started by spawn (macOS) can unpickle it."""
    from courtvision.training.train_student import ClipDataset

    if base is not ClipDataset:
        raise TypeError("finetune_dataset extends train_student.ClipDataset only")
    return functools.partial(FineTuneDataset, hue_deg=hue_deg, static_p=static_p)


def _clip_dataset() -> Any:
    from courtvision.training.train_student import ClipDataset

    return ClipDataset


class FineTuneDataset(_clip_dataset()):  # type: ignore[misc]
    """``finetune_dataset``'s dataset: label-free samples dropped, hue turns, temporal dropout."""

    _hue = 0.0

    def __init__(self, *a: Any, hue_deg: float = 0.0, static_p: float = 0.0, **k: Any) -> None:
        super().__init__(*a, **k)
        self.hue_deg, self.static_p = hue_deg, static_p
        # a sample without person loss (its folder's or its row's person weight 0) and with no ball label of its own
        # trains nothing
        before = len(self.items)
        self.items = [it for it in self.items
                      if self.weights[it[0]].row_person(it[3]) > 0 or (it[4].get("xy") and it[4].get("w", 0) > 0)]
        self.dropped = before - len(self.items)

    def __getitem__(self, idx: int) -> Any:
        item = self.items[idx]
        lab = item[4]
        has_ball = bool(lab.get("xy")) and lab.get("w", 0) > 0
        hue_deg, static_p = self.hue_deg, self.static_p
        self._hue = random.uniform(-hue_deg, hue_deg) if self.train and hue_deg > 0 and not has_ball else 0.0
        if not (self.train and static_p > 0 and random.random() < static_p):
            return super().__getitem__(idx)
        # temporal dropout: the earlier input frames replaced by the frame itself, as at the start of a stream or
        # after a cut (StudentDetector.reset) and on frames sampled too sparsely for motion
        fi, clip, frames, det, lab = item
        self.items[idx] = (fi, clip, [frames[-1]] * len(frames), det, lab)
        try:
            return super().__getitem__(idx)
        finally:
            self.items[idx] = item

    def _img(self, fi: int, clip: int, frame: int) -> np.ndarray:
        img = super()._img(fi, clip, frame)
        if self._hue and img is not None:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + round(self._hue / 2.0)) % 180).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img


def ema_update(ema: dict[str, Any], state: dict[str, Any], decay: float, step: int) -> float:
    """One step of the exponential moving average of the weights (Polyak averaging): every floating tensor of ``ema``
    moves towards ``state``'s by ``1 - d``, with ``d = min(decay, (1 + step) / (10 + step))`` so the first steps do not
    pin the average to the start; other tensors are copied. Returns ``d``."""
    import torch

    d = min(decay, (1.0 + step) / (10.0 + step))
    with torch.no_grad():
        for k, v in state.items():
            if v.is_floating_point():
                ema[k].mul_(d).add_(v.detach().to(ema[k].dtype), alpha=1.0 - d)
            else:
                ema[k].copy_(v)
    return d


def train(args: argparse.Namespace) -> int:
    import torch

    from courtvision.perception.student import build_net
    from courtvision.training.student_from_onnx import freeze_bn
    from courtvision.training.train_student import ClipDataset, _collate, focal_loss, masked_l1, plan_data

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    plan = plan_data(args.data)
    ds = finetune_dataset(ClipDataset, args.hue, args.static_p)(plan.folders,
                                                                plan.train | plan.val if args.no_val else plan.train,
                                                 train=True, weights=plan.weights, crop=args.crop,
                                                 zoom_range=(args.zoom_min, args.zoom_max))
    for row in plan.summary():
        print(f"{row['kind']} {row['path']}: {row['train_clips']} train clips, {row['val_clips']} held out",
              flush=True)
    print(f"{len(ds)} training samples ({ds.dropped} without any label left out)", flush=True)
    net = build_net(3, pretrained=False)
    state = torch.load(args.init, map_location="cpu", weights_only=True)
    net.load_state_dict(state.get("model", state))
    net = net.to(args.device)
    if args.channels_last:  # oneDNN's preferred layout for CPU convolutions
        net = net.to(memory_format=torch.channels_last)
    frozen = [m for m in args.freeze.split(",") if m]
    for name in frozen:
        for p in getattr(net, name).parameters():
            p.requires_grad_(False)
    for m in net.modules():  # folded statistics: never updated, their affine part frozen too
        if isinstance(m, torch.nn.BatchNorm2d):
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(args.train_bn_bias)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    sampler = None
    if args.balance:  # every data folder an equal share of the samples, whatever its size
        sizes: dict[int, int] = defaultdict(int)
        for it in ds.items:
            sizes[it[0]] += 1
        w = torch.tensor([1.0 / sizes[it[0]] for it in ds.items], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(ds), replacement=True,
                                                         generator=torch.Generator().manual_seed(args.seed))
        print("balanced sampling: " + ", ".join(f"{plan.folders[fi]} {n} samples" for fi, n in sorted(sizes.items())),
              flush=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=sampler is None, sampler=sampler,
                                         num_workers=args.workers, drop_last=True,
                                         persistent_workers=args.workers > 0, collate_fn=_collate)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()} if args.ema > 0 else None
    mot_seqs = []
    if args.eval_mot:
        sys.path.insert(0, str(ROOT / "tools"))
        from detector_bench import load_sequences  # noqa: E402

        names = {s.strip() for s in (Path(args.eval_mot).parent / "basketball.txt").read_text().split()} \
            if (Path(args.eval_mot).parent / "basketball.txt").is_file() else None
        mot_seqs = load_sequences(Path(args.eval_mot).parent, [Path(args.eval_mot).name], names)
    log = (out / "log.jsonl").open("a")
    if mot_seqs:
        ev = _quick_mot_eval(net, mot_seqs, args.eval_frames)
        print(f"step 0 eval {ev}", flush=True)
        log.write(json.dumps({"step": 0, "eval": ev}) + "\n")
    step, t0, run = 0, time.time(), defaultdict(float)
    best = -1.0
    while step < args.steps:
        for x, tg in loader:
            net.train()
            freeze_bn(net)
            x = x.to(args.device, non_blocking=True)
            tg = {k: v.to(args.device, non_blocking=True) for k, v in tg.items()}
            if args.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            # bf16 autocast runs the convolutions on AMX / AVX512-BF16 where the CPU has them (on MPS: the GPU's
            # bf16); --fp16 is MPS's half precision; the losses stay fp32
            with torch.autocast(args.device, dtype=torch.float16 if args.fp16 else torch.bfloat16,
                                enabled=args.bf16 or args.fp16):
                o = net(x)
            o = {k: v.float() for k, v in o.items()}
            l_ball = focal_loss(o["ball_hm"], tg["ball_hm"], tg["ball_w"][:, 0])
            l_boff = masked_l1(o["ball_off"], tg["ball_off"], tg["ball_mask"])
            l_phm = focal_loss(o["person_hm"], tg["person_hm"], tg["person_w"][:, 0], tg.get("person_valid"))
            l_pwh = masked_l1(o["person_wh"], tg["person_wh"], tg["person_mask"])
            l_poff = masked_l1(o["person_off"], tg["person_off"], tg["person_mask"])
            loss = args.ball_weight * (2.0 * l_ball + l_boff) + l_phm + 0.5 * l_pwh + l_poff
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            if ema is not None:
                ema_update(ema, net.state_dict(), args.ema, step)
            step += 1
            for k, v in (("loss", loss), ("ball", l_ball), ("person", l_phm), ("wh", l_pwh)):
                run[k] += float(v.detach())
            run["n"] += 1
            if step % args.log_every == 0:
                n = run.pop("n")
                msg = {k: round(v / n, 4) for k, v in run.items()}
                print(f"step {step}/{args.steps} {time.time() - t0:.0f}s lr {sched.get_last_lr()[0]:.2e} {msg}",
                      flush=True)
                run = defaultdict(float)
            if step % args.eval_every == 0 or step >= args.steps:
                ev = _quick_mot_eval(net, mot_seqs, args.eval_frames) if mot_seqs else {}
                print(f"step {step} eval {ev}", flush=True)
                log.write(json.dumps({"step": step, "eval": ev}) + "\n")
                log.flush()
                state = {"model": {k: v.cpu() for k, v in net.state_dict().items()}, "step": step, "eval": ev, "init": str(args.init),
                         "data": args.data}
                if ema is not None:  # the averaged weights beside the last ones (export --ema)
                    state["ema"] = {k: v.cpu() for k, v in ema.items()}
                    state["ema_decay"] = args.ema
                torch.save(state, out / "last.pt")
                if ev and ev["ap50"] > best:
                    best = ev["ap50"]
                    torch.save(state, out / "best.pt")
            if step >= args.steps:
                break
    (out / "summary.json").write_text(json.dumps({"steps": step, "best_eval_ap50": best, "args": vars(args)},
                                                 indent=1, default=str))
    return 0


# --------------------------------------------------------------------------- soup


def soup_states(states: list[dict[str, Any]], base: dict[str, Any] | None = None,
                alpha: float = 1.0) -> dict[str, Any]:
    """Weight-space ensemble of fine-tunes that share one initialisation: their uniform average (a "uniform soup",
    Wortsman et al. 2022, arXiv:2203.05482), interpolated with that initialisation when ``base`` is given:
    ``(1 - alpha) * base + alpha * average`` (WiSE-FT, Wortsman et al. 2022, arXiv:2109.01903). The result is one
    network of the same shape, so a frame costs what it cost before. Integer buffers (batch-norm counters) come from
    the first state; floating tensors are averaged in float64."""
    import torch

    keys = list(states[0])
    for i, s in enumerate(states[1:] + ([base] if base is not None else []), start=1):
        if set(s) != set(keys) or any(tuple(s[k].shape) != tuple(states[0][k].shape) for k in keys):
            raise ValueError(f"state {i} is not the same network as state 0")
    out = {}
    for k in keys:
        t0 = states[0][k]
        if not t0.is_floating_point():
            out[k] = t0.clone()
            continue
        m = torch.stack([s[k].double() for s in states]).mean(0)
        if base is not None:
            m = (1.0 - alpha) * base[k].double() + alpha * m
        out[k] = m.to(t0.dtype).contiguous()
    return out


def soup(weights: list[Path], out: Path, base: Path | None = None, alpha: float = 1.0) -> int:
    """``soup_states`` over checkpoints on disk; prints how far each one moved from ``base`` (relative L2 norm of
    the trained tensors) and writes a checkpoint ``export`` reads."""
    import torch

    def load(p: Path) -> dict[str, Any]:
        s = torch.load(p, map_location="cpu", weights_only=True)
        return s.get("model", s)

    states = [load(w) for w in weights]
    ref = load(base) if base is not None else None
    mixed = soup_states(states, ref, alpha)
    if ref is not None:
        def dist(s: dict[str, Any]) -> float:
            num = sum(float((s[k].double() - ref[k].double()).pow(2).sum()) for k in ref if ref[k].is_floating_point())
            den = sum(float(ref[k].double().pow(2).sum()) for k in ref if ref[k].is_floating_point())
            return math.sqrt(num / den) if den > 0 else float("inf")
        for w, s in zip(weights, states):
            print(f"{w}: {dist(s):.4f} of the base's norm away from it")
        print(f"soup: {dist(mixed):.4f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": mixed, "soup": [str(w) for w in weights], "base": str(base) if base else None,
                "alpha": alpha}, out)
    print(f"{out}: {len(weights)} checkpoints" + (f", alpha {alpha} from {base}" if base else ""))
    return 0


# --------------------------------------------------------------------------- export


def export(weights: Path, out: Path, ema: bool = False) -> int:
    """``.pt`` -> the browser/CPU ONNX export (input ``frames``, outputs ``OUTPUTS``; as
    ``tools/export_student_browser.py`` writes it), checked against torch on a random input."""
    import onnx
    import onnxruntime as ort
    import torch

    from courtvision.perception.student import build_net

    net = build_net(3, pretrained=False).eval()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if ema and "ema" not in state:
        raise SystemExit(f"{weights}: no averaged weights (train with --ema)")
    net.load_state_dict(state["ema"] if ema else state.get("model", state))

    class Export(torch.nn.Module):
        def forward(self, x: Any) -> tuple[Any, ...]:
            o = net(x)
            return tuple(o[name] for name in OUTPUTS)

    wrapper = Export()
    wrapper.add_module("student", net)
    wrapper.eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper, torch.zeros((1, 9, *INPUT_HW)), str(out), input_names=["frames"],
                      output_names=list(OUTPUTS), opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(out)))
    x = np.random.default_rng(0).standard_normal((1, 9, *INPUT_HW)).astype(np.float32)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = dict(zip(OUTPUTS, sess.run(None, {"frames": x})))
    with torch.inference_mode():
        ref = {k: v.numpy() for k, v in net(torch.from_numpy(x)).items()}
    print(f"{out}: max |onnx - torch| " + ", ".join(f"{k} {np.abs(got[k] - ref[k]).max():.1e}" for k in OUTPUTS))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep-mot", help="SportsMOT sequences -> a training folder")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--sample-every", type=int, default=3)
    p.add_argument("--seqs", default=None, help="comma list of sequence names (default: every labelled one)")
    p = sub.add_parser("prep-video", help="videos -> a pseudo-labelled training folder (ball only)")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--sample-every", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=120, help="frames per clip (at ~30 fps), from its middle")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--append", action="store_true",
                   help="add these videos' clips to the folder already in OUT (videos already in it are skipped)")
    p.add_argument("--jpeg-quality", type=int, default=92, help="stored frames' JPEG quality (lower: a smaller folder)")
    p = sub.add_parser("prep-coco", help="COCO-format human labels -> a training folder (one-frame clips)")
    p.add_argument("ann", type=Path, help="the COCO annotation file")
    p.add_argument("images", type=Path, help="the folder of its images")
    p.add_argument("out", type=Path)
    p.add_argument("--games", default="", help="comma list of file-name prefixes to keep (default: all)")
    p.add_argument("--exclude", default="", help="comma list of file-name prefixes to drop")
    p.add_argument("--referees", action="store_true", help="referees are people too (default: players only)")
    p = sub.add_parser("court-people", help="a teacher folder -> the same with only the people on the court as people "
                                          "(near it ignored, the crowd background)")
    p.add_argument("src", type=Path)
    p.add_argument("game", type=Path, help="the game folder whose camera_parts place the boxes on the court")
    p.add_argument("out", type=Path)
    p.add_argument("--keep-ft", type=float, default=3.0, help="people within this of the lines are kept")
    p.add_argument("--ignore-ft", type=float, default=10.0, help="people up to this off the lines are ignored")
    p.add_argument("--min-score", type=float, default=0.4, help="teacher boxes under this are dropped (never targets)")
    p = sub.add_parser("court-persons", help="a pseudo NBA folder -> the same with person labels from its own court "
                                           "fit: the people on the court labelled, the benches and crowd background")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--model", type=Path, required=True,
                   help="the labeller: a detector that boxes everyone (e.g. runs/models/app-v1.onnx)")
    p.add_argument("--score", type=float, default=0.3, help="labeller boxes at this or more are placed on the court")
    p.add_argument("--weak", type=float, default=0.1,
                   help="labeller boxes from this to --score are ignored wherever they stand (unsure: not background)")
    p.add_argument("--margin-in", type=float, default=2.0,
                   help="feet within this many ft outside the court lines are on the court: person labels")
    p.add_argument("--margin-out", type=float, default=8.0,
                   help="feet farther than this many ft outside are background; in between: ignored")
    p.add_argument("--threads", type=int, default=1, help="ONNX Runtime and OpenCV threads")
    p = sub.add_parser("mark-ignore", help="player-only folder -> the same with the unlabelled people among the "
                                         "players (mostly referees) left out of the person loss")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--model", type=Path, default=None,
                   help=f"the model that finds everyone (default: the app's export {DEFAULT_MODEL.relative_to(ROOT)}; "
                        f"with --promote the first app model {APP_V1_MODEL.relative_to(ROOT)})")
    p.add_argument("--iou", type=float, default=0.3, help="overlap with a labelled player that makes a box that player")
    p.add_argument("--score", type=float, default=0.3)
    p.add_argument("--margin", type=float, default=0.05, help="growth of the players' foot hull, share of frame height")
    p.add_argument("--referees", action=argparse.BooleanOptionalAction, default=True,
                   help="also ignore the people the referee check reads as referees, wherever they stand")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--promote", action="store_true",
                   help="make the unlabelled officials person labels (promote_officials): no labelled player within "
                        "--iou, standing among the players, clear of the image border, called a referee")
    p.add_argument("--promote-referee", choices=("shirt", "learned", "either", "both"), default=Promote.referee,
                   help="the referee check of --promote")
    p.add_argument("--promote-score", type=float, default=Promote.score,
                   help="the labeller's score a promoted box needs")
    p.add_argument("--promote-ratio", default=",".join(str(v) for v in Promote.ratio),
                   help="LOW,HIGH: box height over the players' fitted height at its foot row")
    p.add_argument("--promote-above", type=float, default=Promote.above,
                   help="foot row at most this share of the frame height above the farthest player's")
    p.add_argument("--promote-border", type=float, default=Promote.border,
                   help="clearance from every image edge, share of the frame height")
    p.add_argument("--promote-min-players", type=int, default=Promote.min_players,
                   help="labelled players a frame needs for the height fit")
    p = sub.add_parser("prep-teacher", help="videos -> an RF-DETR teacher folder (held-out ball checks)")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--frames", type=int, default=48, help="frames per video (at ~30 fps), from its middle")
    p.add_argument("--weights", default=None, help="RF-DETR Small checkpoint (default: rfdetr's COCO download)")
    p.add_argument("--shape", default="736,1312", help="teacher input height,width (nba_teacher_labels default)")
    p.add_argument("--device", default="cpu", choices=("cpu", "mps"),
                   help="where RF-DETR runs (mps: faster on a Mac; float rounding differs from the CPU's)")
    p = sub.add_parser("train", help="fine-tune")
    p.add_argument("--data", action="append", required=True, help="a training folder, DIR or DIR:WEIGHT")
    p.add_argument("--init", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--freeze", default="stem,l1", help="comma list of modules kept fixed")
    p.add_argument("--train-bn-bias", action="store_true", help="also train the folded batch-norm biases")
    p.add_argument("--ball-weight", type=float, default=1.0)
    p.add_argument("--hue", type=float, default=0.0, help="hue rotation (degrees) of samples without a ball label")
    p.add_argument("--zoom-min", type=float, default=0.8)
    p.add_argument("--zoom-max", type=float, default=1.25)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bf16", action="store_true", help="bf16 autocast for the forward pass (CPU AMX / AVX512-BF16)")
    p.add_argument("--fp16", action="store_true", help="fp16 autocast for the forward pass (--device mps)")
    p.add_argument("--device", default="cpu", choices=("cpu", "mps"), help="where to train (mps: Apple GPU)")
    p.add_argument("--balance", action="store_true", help="sample every --data folder equally often")
    p.add_argument("--ema", type=float, default=0.0,
                   help="keep an exponential moving average of the weights with this decay (e.g. 0.999; 0: off)")
    p.add_argument("--static-p", type=float, default=0.0,
                   help="temporal dropout: share of samples whose earlier input frames are the frame itself")
    p.add_argument("--channels-last", action="store_true", help="channels-last memory format (faster CPU convolutions)")
    p.add_argument("--no-val", action="store_true", help="train on the held-out clips too")
    p.add_argument("--eval-mot", default=None, help="SportsMOT split folder to score during training")
    p.add_argument("--eval-frames", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p = sub.add_parser("soup", help="average fine-tunes of one start (and interpolate with that start)")
    p.add_argument("weights", type=Path, nargs="+", help=".pt checkpoints fine-tuned from the same --init")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base", type=Path, default=None, help="their shared start, for (1 - alpha) * base + alpha * avg")
    p.add_argument("--alpha", type=float, default=1.0, help="weight of the average against --base (1: plain soup)")
    p = sub.add_parser("export", help=".pt -> ONNX")
    p.add_argument("weights", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ema", action="store_true", help="export the averaged weights (a checkpoint trained with --ema)")
    a = ap.parse_args(argv)
    if a.cmd == "soup":
        return soup(a.weights, a.out, a.base, a.alpha)
    if a.cmd == "prep-mot":
        names = {s.strip() for s in a.seqs.split(",")} if a.seqs else None
        return 0 if prep_mot(a.src, a.out, a.sample_every, names) else 2
    if a.cmd == "prep-video":
        return 0 if prep_video(a.src, a.out, a.model, a.sample_every, a.max_frames, a.threads, a.append,
                               a.jpeg_quality) else 2
    if a.cmd == "prep-coco":
        games = [g.strip() for g in a.games.split(",") if g.strip()]
        excl = [g.strip() for g in a.exclude.split(",") if g.strip()]
        return 0 if prep_coco(a.ann, a.images, a.out, games, excl, a.referees) else 2
    if a.cmd == "court-people":
        court_people(a.src, a.game, a.out, a.keep_ft, a.ignore_ft, a.min_score)
        return 0
    if a.cmd == "court-persons":
        cv2.setNumThreads(a.threads)  # the court fit's image work
        court_persons(a.src, a.out, a.model, a.score, a.margin_in, a.margin_out, a.threads, weak=a.weak)
        return 0
    if a.cmd == "mark-ignore":
        rule = None
        if a.promote:
            lo, hi = (float(v) for v in a.promote_ratio.split(","))
            rule = Promote(a.promote_referee, a.iou, a.promote_score, (lo, hi), a.promote_above, a.promote_border,
                           a.promote_min_players)
        model = a.model or (APP_V1_MODEL if a.promote else DEFAULT_MODEL)
        return 0 if mark_ignore(a.src, a.out, model, a.iou, a.score, a.margin, a.threads,
                                referees=a.referees, promote=rule) >= 0 else 2
    if a.cmd == "prep-teacher":
        return 0 if prep_teacher(a.src, a.out, a.frames, a.weights, a.shape, a.device) else 2
    if a.cmd == "train":
        return train(a)
    return export(a.weights, a.out, a.ema)


if __name__ == "__main__":
    raise SystemExit(main())
