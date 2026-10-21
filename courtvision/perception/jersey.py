"""Jersey-number reading and per-track temporal voting.

* ``JerseyReader`` protocol: ``read(frame_bgr, bbox) -> (number | None, confidence)``.
* ``NullJerseyReader`` never reads (practice pinnies, or no reader configured).
* ``TemplateJerseyReader`` reads the synthetic renderer's numbers. ``sim/render.py`` draws them
  with ``cv2.putText(img, number, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
  thickness=2)``, ``scale = max(0.35, bbox_height / 70)``, centred horizontally on the bbox and
  vertically at 40 % of its height. The reader renders each candidate the same way and scores it by
  the best IoU between the template's white pixels and the frame's white pixels over a small
  shift/scale search (``cv2.matchTemplate``), so "1" inside "11" loses to "11". OpenCV 5 draws
  Hershey text anti-aliased, so the template alpha is binarised at ``template_alpha``.
* ``JerseyVoter``: confidence-weighted votes per track; a number is confirmed after >= 3 agreeing
  reads holding >= 60 % of the vote mass (optionally also by 3 confident identical reads in a row,
  Roboflow's rule, cv_pipeline.md §1.1, §4.2), only among the numbers the caller allows (the team's
  roster). A one-digit read that is contained in a voted two-digit number is down-weighted (the
  Koshkina aggregation bias, cv_pipeline.md §4.2). An optional window keeps only recent reads.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from courtvision.types import BBox

__all__ = [
    "ALL_JERSEY_NUMBERS",
    "JerseyReader",
    "JerseyVoter",
    "NullJerseyReader",
    "TemplateJerseyConfig",
    "TemplateJerseyReader",
]

log = logging.getLogger(__name__)

ALL_JERSEY_NUMBERS: tuple[str, ...] = tuple(str(i) for i in range(100)) + ("00",)


@runtime_checkable
class JerseyReader(Protocol):
    def read(self, frame_bgr: np.ndarray, bbox: BBox) -> tuple[str | None, float]: ...


class NullJerseyReader:
    """Reads nothing."""

    def read(self, frame_bgr: np.ndarray, bbox: BBox) -> tuple[str | None, float]:
        return None, 0.0


@dataclass
class TemplateJerseyConfig:
    """Template reader settings. Rendering constants follow the sim/render.py convention."""

    candidates: tuple[str, ...] = ALL_JERSEY_NUMBERS  # restrict to the roster when known
    base_height_px: float = 70.0  # scale = max(min_scale, bbox_h / base_height); sourced (render convention)
    min_scale: float = 0.35  # sourced (render convention)
    thickness: int = 2  # sourced (render convention)
    vertical_frac: float = 0.40  # text centre at 40 % of the bbox height; sourced (render convention)
    scale_factors: tuple[float, ...] = (0.94, 0.97, 1.0, 1.03, 1.06)  # absorbs bbox-height error; tuned
    search_px: int = 4  # extra shift search around the expected text box; tuned
    white_min: int = 200  # all BGR channels >= this => white pixel; tuned
    min_bbox_height_px: float = 20.0  # legibility gate (text below ~8 px is unreadable); tuned
    min_iou: float = 0.45  # weaker best matches are "no read"; tuned
    margin_norm: float = 0.15  # IoU margin over the runner-up that gives full confidence; tuned
    pad_frac: float = 0.35  # template padding (x glyph height) that penalises unexplained white; tuned
    # OpenCV 5 anti-aliases Hershey text whatever the lineType, so templates are alpha maps; a pixel
    # over a palette background reaches white_min at alpha ~0.5-0.75. Binarise at this alpha; tuned.
    template_alpha: float = 0.6
    refine_top: int = 3  # candidates refined over all scale factors after the nominal-scale pass; tuned


class TemplateJerseyReader:
    """Template-matching reader for synthetic renders (white Hershey digits)."""

    def __init__(self, config: TemplateJerseyConfig | None = None, candidates: Iterable[str] | None = None) -> None:
        self.config = config or TemplateJerseyConfig()
        self.candidates = tuple(dict.fromkeys(candidates if candidates is not None else self.config.candidates))
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    def read(self, frame_bgr: np.ndarray, bbox: BBox) -> tuple[str | None, float]:
        cfg = self.config
        if bbox.h < cfg.min_bbox_height_px or not self.candidates:
            return None, 0.0
        base = max(cfg.min_scale, bbox.h / cfg.base_height_px)
        scales = sorted({max(cfg.min_scale, base * f) for f in cfg.scale_factors})
        window, _origin = self._white_window(frame_bgr, bbox, max(scales))
        if window is None or not window.any():
            return None, 0.0
        msums: dict[tuple[int, int], np.ndarray] = {}  # white-pixel box sums depend on template shape only
        nominal = min(scales, key=lambda sc: abs(sc - base))
        best = {n: self._best_iou(window, self._template(n, nominal), msums) for n in self.candidates}
        # Coarse to fine: only the leading candidates are refined over the other scales.
        for number in sorted(best, key=lambda k: (-best[k], k))[: cfg.refine_top]:
            for sc in scales:
                if sc != nominal:
                    best[number] = max(best[number], self._best_iou(window, self._template(number, sc), msums))
        ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
        top, top_iou = ranked[0]
        if top_iou < cfg.min_iou:
            return None, 0.0
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        conf = top_iou * min(1.0, max(0.0, top_iou - runner) / cfg.margin_norm)
        return top, float(np.clip(conf, 0.0, 1.0))

    # ------------------------------------------------------------------ internals

    def _white_window(
        self, frame: np.ndarray, bbox: BBox, max_scale: float
    ) -> tuple[np.ndarray | None, tuple[int, int]]:
        cfg = self.config
        (tw, th), base = cv2.getTextSize("00", cv2.FONT_HERSHEY_SIMPLEX, max_scale, cfg.thickness)
        pad = int(np.ceil(cfg.pad_frac * th)) + cfg.thickness
        cx = (bbox.x1 + bbox.x2) / 2.0
        cy = bbox.y1 + cfg.vertical_frac * bbox.h
        half_w = tw / 2.0 + pad + cfg.search_px
        half_h = th + base + pad + cfg.search_px  # centre-vs-baseline convention is not pinned down
        h_img, w_img = frame.shape[:2]
        x1, x2 = max(0, int(cx - half_w)), min(w_img, int(np.ceil(cx + half_w)))
        y1, y2 = max(0, int(cy - half_h)), min(h_img, int(np.ceil(cy + half_h)))
        if x2 <= x1 or y2 <= y1:
            return None, (x1, y1)
        crop = frame[y1:y2, x1:x2]
        white = np.all(crop >= cfg.white_min, axis=2).astype(np.float32)
        return white, (x1, y1)

    def _template(self, number: str, scale: float) -> np.ndarray:
        key = (number, int(round(scale * 100)))
        tpl = self._cache.get(key)
        if tpl is None:
            s = key[1] / 100.0
            t = self.config.thickness
            (tw, th), base = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, s, t)
            pad_x = int(np.ceil(self.config.pad_frac * th))
            pad_y = 2
            canvas = np.zeros((th + base + 2 * pad_y + 2 * t, tw + 2 * pad_x + 2 * t), dtype=np.uint8)
            cv2.putText(canvas, number, (pad_x + t, pad_y + t + th), cv2.FONT_HERSHEY_SIMPLEX, s, 255, t)
            tpl = (canvas >= 255.0 * self.config.template_alpha).astype(np.float32)
            self._cache[key] = tpl
        return tpl

    @staticmethod
    def _best_iou(window: np.ndarray, tpl: np.ndarray, msums: dict[tuple[int, int], np.ndarray]) -> float:
        """Max over placements of IoU(template white, frame white) inside the template box."""
        wh, ww = window.shape
        th, tw = tpl.shape
        if th > wh or tw > ww:
            window = np.pad(window, ((0, max(0, th - wh)), (0, max(0, tw - ww))))
        inter = cv2.matchTemplate(window, tpl, cv2.TM_CCORR)
        msum = msums.get(tpl.shape)
        if msum is None or msum.shape != inter.shape:
            msum = cv2.matchTemplate(window, np.ones_like(tpl), cv2.TM_CCORR)
            msums[tpl.shape] = msum
        tsum = float(tpl.sum())
        iou = inter / np.maximum(tsum + msum - inter, 1e-6)
        return float(iou.max())


class JerseyVoter:
    """Per-track confidence-weighted voting over a window of recent reads, with a roster constraint.

    A number is confirmed when it has >= ``min_votes`` reads and >= ``min_share`` of the
    (confidence-weighted, Koshkina-adjusted) vote mass. With ``streak=True`` the last ``min_votes``
    reads agreeing also confirm it (Roboflow's "3 identical predictions in a row"; it wins over an
    older majority, so an identity recovers quickly after a tracker swap), provided each of those
    reads has confidence >= ``streak_min_conf`` and the streak is not a one-digit partial read of a
    two-digit number that holds votes in the window.
    """

    def __init__(
        self,
        min_votes: int = 3,  # Roboflow: confirm after 3 identical reads; sourced
        min_share: float = 0.6,  # share of vote mass for the leader; spec (ARCHITECTURE JerseyVoter)
        single_digit_weight: float = 0.5,  # Koshkina down-weighting of 1-digit reads; sourced (factor tuned)
        min_read_conf: float = 0.05,  # weaker reads are ignored; tuned
        window: int | None = None,  # keep only the newest reads per track (None = all); tuned
        streak: bool = False,  # also confirm on min_votes identical consecutive reads (bypasses min_share);
        #   off by default so confirmation follows the spec (>= 3 reads AND share >= 0.6); design
        streak_min_conf: float = 0.5,  # every read of a confirming streak must be at least this sure; tuned
    ) -> None:
        self.min_votes = min_votes
        self.min_share = min_share
        self.single_digit_weight = single_digit_weight
        self.min_read_conf = min_read_conf
        self.window = window
        self.streak = streak
        self.streak_min_conf = streak_min_conf
        self._reads: dict[int, deque[tuple[str, float]]] = {}

    def add(self, track_id: int, jersey: str | None, confidence: float) -> None:
        """Record one per-frame read for ``track_id`` (None / weak reads are ignored)."""
        if jersey is None or confidence < self.min_read_conf:
            return
        self._reads.setdefault(track_id, deque(maxlen=self.window)).append((jersey, float(confidence)))

    def best(self, track_id: int, allowed: Collection[str] | None = None) -> tuple[str | None, float, int]:
        """(leading number, its vote share, its read count) among ``allowed`` numbers."""
        weights, counts = self._tally(track_id, allowed)
        total = sum(weights.values())
        if total <= 0:
            return None, 0.0, 0
        top = min(weights, key=lambda k: (-weights[k], k))
        return top, weights[top] / total, counts[top]

    def result(self, track_id: int, allowed: Collection[str] | None = None) -> tuple[str | None, float]:
        """Confirmed number and its share, or (None, share-of-leader) when not confirmed.

        A streak of ``min_votes`` identical reads wins over an older majority in the window.
        """
        run = self._streak(track_id, allowed)
        if run is not None:
            weights, _ = self._tally(track_id, allowed)
            return run, weights[run] / max(sum(weights.values()), 1e-12)
        top, share, count = self.best(track_id, allowed)
        if top is not None and count >= self.min_votes and share >= self.min_share:
            return top, share
        return None, share

    def weight(self, track_id: int, jersey: str) -> float:
        return sum(c for n, c in self._reads.get(track_id, ()) if n == jersey)

    def merge(self, src: int, dst: int, weight: float = 1.0) -> None:
        """Move ``src``'s reads into ``dst`` as older evidence (identity re-link), their confidences
        scaled by ``weight``: a proximity re-link is a hypothesis, so inherited reads can be made
        to count less than reads ``dst`` makes itself."""
        old = self._reads.pop(src, None)
        if not old:
            return
        merged: deque[tuple[str, float]] = deque(((n, c * weight) for n, c in old), maxlen=self.window)
        merged.extend(self._reads.get(dst, ()))
        self._reads[dst] = merged

    def discount(self, track_id: int, factor: float) -> None:
        """Scale ``track_id``'s reads so far by ``factor``: after an occlusion or a same-team crossing
        the track may be on another body, so older reads are weak evidence about who it is now and a
        few fresh agreeing reads must be able to outvote them."""
        reads = self._reads.get(track_id)
        if reads:
            self._reads[track_id] = deque(((n, c * factor) for n, c in reads), maxlen=self.window)

    def forget(self, track_id: int) -> None:
        self._reads.pop(track_id, None)

    def _streak(self, track_id: int, allowed: Collection[str] | None) -> str | None:
        reads = self._reads.get(track_id)
        if not self.streak or not reads or len(reads) < self.min_votes:
            return None
        last = list(reads)[-self.min_votes:]
        number = last[0][0]
        if any(n != number or c < self.streak_min_conf for n, c in last):
            return None
        if allowed is not None and number not in allowed:
            return None
        if len(number) == 1 and any(len(n) == 2 and number in n and (allowed is None or n in allowed)
                                    for n, _ in reads):
            return None  # "2" x3 while "23" holds votes: a partial read, left to the weighted vote
        return number

    def _tally(self, track_id: int, allowed: Collection[str] | None) -> tuple[dict[str, float], dict[str, int]]:
        weights: dict[str, float] = {}
        counts: dict[str, int] = {}
        for n, c in self._reads.get(track_id, ()):
            if allowed is None or n in allowed:
                weights[n] = weights.get(n, 0.0) + c
                counts[n] = counts.get(n, 0) + 1
        return self._effective(weights), counts

    def _effective(self, weights: dict[str, float]) -> dict[str, float]:
        w = dict(weights)
        two_digit = [k for k in w if len(k) == 2]
        for k in list(w):
            if len(k) == 1 and any(k in t for t in two_digit):
                w[k] *= self.single_digit_weight
        return w
