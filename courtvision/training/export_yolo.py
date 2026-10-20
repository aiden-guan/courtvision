"""Export frames + YOLO labels + data.yaml from detections / pseudo-labels, with a rights-aware manifest.

Layout (the Roboflow YOLO layout, which rfdetr 1.11.2 detects as ``dataset_file="yolo"`` and Ultralytics
reads through ``data.yaml``)::

    out/data.yaml                  names: [player, referee, ball, rim]; train: train/images; val: valid/images
    out/train/images/<stem>.jpg    out/train/labels/<stem>.txt   ("cls cx cy w h", normalised to [0, 1])
    out/valid/images/...           out/valid/labels/...
    out/manifest.jsonl             one row per image: rights_status, label_origin, label_conf, game, frame, ...
    out/export_meta.json           counts, for_shipping, manifest_sha256 (embed it in the weights, §8.4)

Rules (training_data §5, §8.4; RESEARCH §5.1, §7A.11):

* every row carries ``rights_status`` in {own, licensed, research_only} and ``label_origin`` in
  {pbp, ocr, cv_auto, pseudo, jev, human};
* ``for_shipping=True`` refuses the whole export (``RightsError``, nothing written) unless every row is
  ``own`` / ``licensed`` and its game has ``training_allowed=True``;
* splits are by game, never inside a game (SportsMOT / DeepSportradar practice, training_data §5 step 4).

``select_for_review`` ranks frames for human labelling by detector uncertainty (scores near the
decision boundary, a missing ball in live play) with a per-video minimum frame gap for diversity
(training_data §2.5, priority 3 and 4).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from courtvision.types import BBox, Detection, FrameDetections, FrameState, ObjClass

__all__ = [
    "CLASS_NAMES",
    "LABEL_ORIGINS",
    "RIGHTS_STATUSES",
    "SHIPPABLE_RIGHTS",
    "ExportConfig",
    "ExportItem",
    "ExportResult",
    "ReviewConfig",
    "RightsError",
    "check_rights",
    "detections_from_frame_state",
    "export_yolo",
    "select_for_review",
    "uncertainty_score",
    "yolo_lines",
]

log = logging.getLogger(__name__)

CLASS_NAMES: tuple[str, ...] = ("player", "referee", "ball", "rim")
_CLASS_ID: dict[ObjClass, int] = {ObjClass.PLAYER: 0, ObjClass.REFEREE: 1, ObjClass.BALL: 2, ObjClass.RIM: 3}
RIGHTS_STATUSES: tuple[str, ...] = ("own", "licensed", "research_only")
SHIPPABLE_RIGHTS: tuple[str, ...] = ("own", "licensed")
LABEL_ORIGINS: tuple[str, ...] = ("pbp", "ocr", "cv_auto", "pseudo", "jev", "human")
_SPLIT_DIRS = {"train": "train", "val": "valid"}


class RightsError(RuntimeError):
    """A shipping export contains rows without training rights."""


@dataclass
class ExportConfig:
    """Export settings."""

    val_frac: float = 0.2  # share of games in valid/; tuned
    seed: int = 0  # game -> split hash salt
    image_ext: str = ".jpg"  # for ndarray frames
    jpeg_quality: int = 95  # cv2 JPEG quality; high so compression artefacts do not hide the small ball; tuned
    # Pseudo-label keep thresholds per class (detector score); tuned, unverified. Human labels keep everything.
    min_score: dict[str, float] = field(default_factory=lambda: {"player": 0.5, "referee": 0.5, "ball": 0.3, "rim": 0.5})
    min_box_px: float = 2.0  # boxes narrower / shorter than this after clipping are dropped; tuned
    for_shipping: bool = False


@dataclass
class ExportItem:
    """One frame to export. ``image`` is a BGR array or a path to an image file."""

    image: np.ndarray | str | Path
    detections: Sequence[Detection] | FrameDetections
    game_id: str
    video_id: str
    frame_idx: int
    t: float = 0.0
    rights_status: str = "research_only"
    label_origin: str = "pseudo"
    training_allowed: bool = False  # per-game training consent, separate from processing consent (§5.1)
    source: str = ""
    live: bool | None = None  # ball in play (for the "ball missing in live play" uncertainty signal)
    image_size: tuple[int, int] | None = None  # (width, height); read from the image when None

    def dets(self) -> list[Detection]:
        return list(self.detections.detections if isinstance(self.detections, FrameDetections) else self.detections)


@dataclass
class ExportResult:
    root: str
    n_images: int
    n_boxes: int
    per_split: dict[str, int]
    class_counts: dict[str, int]
    manifest_sha256: str
    for_shipping: bool
    rights_statuses: dict[str, int]


# ---------------------------------------------------------------------------------------------- rights


def check_rights(items: Iterable[ExportItem], for_shipping: bool) -> list[str]:
    """Problems that block the export (unknown vocab always; missing rights when shipping)."""
    problems: list[str] = []
    for it in items:
        where = f"{it.game_id}/{it.video_id}#{it.frame_idx}"
        if it.rights_status not in RIGHTS_STATUSES:
            problems.append(f"{where}: unknown rights_status {it.rights_status!r}")
        if it.label_origin not in LABEL_ORIGINS:
            problems.append(f"{where}: unknown label_origin {it.label_origin!r}")
        if for_shipping and it.rights_status not in SHIPPABLE_RIGHTS:
            problems.append(f"{where}: rights_status {it.rights_status!r} cannot ship (needs own/licensed)")
        if for_shipping and not it.training_allowed:
            problems.append(f"{where}: game has training_allowed=False")
    return problems


# ---------------------------------------------------------------------------------------------- labels


def _keep(d: Detection, origin: str, cfg: ExportConfig) -> bool:
    return d.cls in _CLASS_ID and (origin == "human" or d.score >= cfg.min_score.get(d.cls.value, 0.5))


def yolo_lines(dets: Sequence[Detection], image_size: tuple[int, int], config: ExportConfig | None = None,
               label_origin: str = "pseudo") -> tuple[list[str], list[Detection]]:
    """YOLO label lines ("cls cx cy w h", normalised, clipped to the image) and the detections kept."""
    cfg = config or ExportConfig()
    w, h = image_size
    if w <= 0 or h <= 0:
        raise ValueError(f"bad image size {image_size!r}")
    lines: list[str] = []
    kept: list[Detection] = []
    for d in dets:
        if not _keep(d, label_origin, cfg):
            continue
        if not all(math.isfinite(v) for v in d.bbox.as_tuple()):
            # NaN survives min/max clipping and would be written as a "nan" label line (e.g. a camera
            # projection behind the image plane); such a box is not a label.
            log.debug("yolo_lines: dropping non-finite %s box %s", d.cls.value, d.bbox)
            continue
        x1, y1 = min(max(d.bbox.x1, 0.0), w), min(max(d.bbox.y1, 0.0), h)
        x2, y2 = min(max(d.bbox.x2, 0.0), w), min(max(d.bbox.y2, 0.0), h)
        if x2 - x1 < cfg.min_box_px or y2 - y1 < cfg.min_box_px:
            continue
        lines.append(f"{_CLASS_ID[d.cls]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        kept.append(d)
    return lines, kept


def detections_from_frame_state(frame: FrameState, referee_boxes: Sequence[BBox] = (), ball_min_radius_px: float = 1.0) -> list[Detection]:
    """Pseudo-label boxes from a FrameState: player bboxes, the visible ball (image_xy + radius), rims."""
    dets = [Detection(p.bbox, ObjClass.PLAYER, float(p.confidence)) for p in frame.players if p.bbox is not None]
    dets += [Detection(b, ObjClass.REFEREE, 1.0) for b in referee_boxes]
    b = frame.ball
    if b is not None and b.visible and not b.interpolated and b.image_xy is not None and b.radius_px:
        r = max(float(b.radius_px), ball_min_radius_px)
        u, v = b.image_xy
        dets.append(Detection(BBox(u - r, v - r, u + r, v + r), ObjClass.BALL, float(b.confidence or 1.0)))
    dets += [Detection(rim.bbox, ObjClass.RIM, float(rim.confidence)) for rim in frame.rims]
    return dets


# ---------------------------------------------------------------------------------------------- export


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-") or "x"


def _game_split(games: Sequence[str], val_frac: float, seed: int) -> dict[str, str]:
    """Deterministic game -> "train"/"val"; at least one val game when there are two or more games."""
    uniq = sorted(set(games))
    ranked = sorted(uniq, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    n_val = 0 if len(uniq) < 2 or val_frac <= 0 else min(len(uniq) - 1, max(1, round(val_frac * len(uniq))))
    val = set(ranked[:n_val])
    return {g: "val" if g in val else "train" for g in uniq}


def _image_size(it: ExportItem) -> tuple[int, int]:
    if it.image_size is not None:
        return it.image_size
    if isinstance(it.detections, FrameDetections) and it.detections.image_size:
        return it.detections.image_size
    if isinstance(it.image, np.ndarray):
        return int(it.image.shape[1]), int(it.image.shape[0])
    import cv2  # lazy: only for path inputs without a known size

    img = cv2.imread(str(it.image))
    if img is None:
        raise ValueError(f"cannot read image {it.image}")
    return int(img.shape[1]), int(img.shape[0])


def _write_image(it: ExportItem, dest_stem: Path, cfg: ExportConfig) -> Path:
    if isinstance(it.image, np.ndarray):
        import cv2

        dest = dest_stem.with_suffix(cfg.image_ext)
        params = [int(cv2.IMWRITE_JPEG_QUALITY), cfg.jpeg_quality] if cfg.image_ext.lower() in (".jpg", ".jpeg") else []
        if not cv2.imwrite(str(dest), it.image, params):
            raise OSError(f"cv2.imwrite failed for {dest}")
        return dest
    src = Path(it.image)
    dest = dest_stem.with_suffix(src.suffix.lower() or cfg.image_ext)
    shutil.copy2(src, dest)
    return dest


def _label_conf(kept: Sequence[Detection], dropped: Sequence[Detection], origin: str) -> float:
    """Lowest kept score; for a frame with no kept box, confidence that nothing was missed."""
    if origin == "human":
        return 1.0
    kept_s = [float(d.score) for d in kept if math.isfinite(d.score)]
    if kept:
        return round(min(kept_s), 4) if kept_s else 0.0  # a kept box of unknown score: no confidence
    return round(1.0 - max((float(d.score) for d in dropped if math.isfinite(d.score)), default=0.0), 4)


def _data_yaml(manifest_sha: str, for_shipping: bool) -> str:
    names = ", ".join(CLASS_NAMES)
    return ("# CourtVision detector export (courtvision.training.export_yolo)\n"
            f"# for_shipping: {str(for_shipping).lower()}\n# manifest_sha256: {manifest_sha}\n"
            "train: train/images\nval: valid/images\n"
            f"nc: {len(CLASS_NAMES)}\nnames: [{names}]\n")


def export_yolo(items: Sequence[ExportItem], out_dir: str | Path, config: ExportConfig | None = None,
                for_shipping: bool | None = None, review_config: ReviewConfig | None = None) -> ExportResult:
    """Write the YOLO dataset into a new or empty ``out_dir``.

    Raises ``RightsError`` (before writing anything) when the rights gate fails, and ``FileExistsError``
    when ``out_dir`` already holds files (stale images would escape the manifest).
    """
    cfg = config or ExportConfig()
    shipping = cfg.for_shipping if for_shipping is None else bool(for_shipping)
    problems = check_rights(items, shipping)
    if problems:
        shown = "; ".join(problems[:10]) + (f"; ... {len(problems) - 10} more" if len(problems) > 10 else "")
        raise RightsError(f"export refused ({len(problems)} problem(s)): {shown}")
    stems = [f"{_safe(it.game_id)}_{_safe(it.video_id)}_{it.frame_idx:07d}" for it in items]
    dupes = [s for s, n in Counter(stems).items() if n > 1]
    if dupes:
        raise ValueError(f"duplicate frames in export: {dupes[:5]}")
    root = Path(out_dir)
    if root.exists() and any(root.iterdir()):
        # Leftover images would be trained on without a manifest row, bypassing the rights gate.
        raise FileExistsError(f"export directory {root} is not empty; export into a fresh directory")
    split = _game_split([it.game_id for it in items], cfg.val_frac, cfg.seed)
    if len(set(split.values())) < 2 and items:
        log.warning("only one game: everything goes to train/, valid/ stays empty")
    for d in _SPLIT_DIRS.values():
        (root / d / "images").mkdir(parents=True, exist_ok=True)
        (root / d / "labels").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    classes: Counter[str] = Counter()
    for it, stem in zip(items, stems, strict=True):
        sub = _SPLIT_DIRS[split[it.game_id]]
        size = _image_size(it)
        dets = it.dets()
        lines, kept = yolo_lines(dets, size, cfg, it.label_origin)
        img = _write_image(it, root / sub / "images" / stem, cfg)
        (root / sub / "labels" / f"{stem}.txt").write_text("".join(f"{line}\n" for line in lines))
        counts = Counter(d.cls.value for d in kept)
        classes.update(counts)
        kept_ids = {id(d) for d in kept}  # identity: Detection equality would compare embedding arrays
        rows.append({
            "image": img.relative_to(root).as_posix(), "label": f"{sub}/labels/{stem}.txt", "split": split[it.game_id],
            "game_id": it.game_id, "video_id": it.video_id, "frame_idx": it.frame_idx, "t": round(float(it.t), 3),
            "rights_status": it.rights_status, "label_origin": it.label_origin,
            "label_conf": _label_conf(kept, [d for d in dets if id(d) not in kept_ids], it.label_origin),
            "training_allowed": it.training_allowed, "source": it.source, "n_boxes": len(kept),
            "class_counts": dict(counts), "uncertainty": round(uncertainty_score(dets, it.live, review_config), 4),
        })
    manifest = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    sha = hashlib.sha256(manifest.encode()).hexdigest()
    (root / "manifest.jsonl").write_text(manifest)
    (root / "data.yaml").write_text(_data_yaml(sha, shipping))
    result = ExportResult(str(root), len(rows), sum(r["n_boxes"] for r in rows), dict(Counter(r["split"] for r in rows)),
                          dict(classes), sha, shipping, dict(Counter(r["rights_status"] for r in rows)))
    (root / "export_meta.json").write_text(json.dumps({**asdict(result), "class_names": list(CLASS_NAMES)}, indent=2, sort_keys=True))
    log.info("YOLO export: %d images, %d boxes, splits %s, shipping=%s, manifest %s", result.n_images, result.n_boxes,
             result.per_split, shipping, sha[:12])
    return result


# ---------------------------------------------------------------------------------------------- active learning


@dataclass
class ReviewConfig:
    """Uncertainty sampling for human review (training_data §2.5)."""

    band: tuple[float, float] = (0.2, 0.8)  # detector scores in this band are uncertain; tuned
    ball_missing_weight: float = 1.0  # live-play frame without a confident ball (§2.5 priority 3); tuned
    count_weight: float = 0.1  # extra weight per additional uncertain box; tuned
    min_gap_frames: int = 15  # diversity: picks of one video are at least this many frames apart; tuned


def uncertainty_score(dets: Sequence[Detection], live: bool | None = None, config: ReviewConfig | None = None) -> float:
    """max(1 - |2s - 1|) over in-band detections, + count bonus, + a missing-ball term in live play."""
    cfg = config or ReviewConfig()
    lo, hi = cfg.band
    margins = sorted((1.0 - abs(2.0 * d.score - 1.0) for d in dets if lo <= d.score <= hi), reverse=True)
    score = (margins[0] + cfg.count_weight * (len(margins) - 1)) if margins else 0.0
    if live and not any(d.cls is ObjClass.BALL and d.score > hi for d in dets):
        score += cfg.ball_missing_weight
    return float(score)


def select_for_review(items: Sequence[ExportItem], k: int, config: ReviewConfig | None = None) -> list[int]:
    """Indices of the ``k`` most uncertain frames, at most one per ``min_gap_frames`` window of a video.

    Ties break on (video_id, frame_idx) so the selection is deterministic. Frames scoring 0 are never picked.
    """
    cfg = config or ReviewConfig()
    scored = sorted(((uncertainty_score(it.dets(), it.live, cfg), i) for i, it in enumerate(items)),
                    key=lambda s: (-s[0], items[s[1]].video_id, items[s[1]].frame_idx))
    picked: list[int] = []
    taken: dict[str, list[int]] = {}
    for score, i in scored:
        if len(picked) >= k or score <= 0.0:
            break
        it = items[i]
        if any(abs(it.frame_idx - f) < cfg.min_gap_frames for f in taken.get(it.video_id, [])):
            continue
        picked.append(i)
        taken.setdefault(it.video_id, []).append(it.frame_idx)
    return picked
