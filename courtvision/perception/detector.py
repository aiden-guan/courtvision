"""Detector protocol and the colour-blob detector for synthetic renders.

``ColorBlobDetector`` segments the shared synthetic palette (``courtvision.sim.palette``):
jersey colours -> PLAYER, referee black -> REFEREE, orange -> BALL, red rim -> RIM. Colour
masks use per-channel boxes (``cv2.inRange``, i.e. an L-infinity tolerance).

* A player box is the jersey blob extended up to the skin-coloured head (and down to skin-coloured
  legs when present), so it matches the drawn figure's bbox.
* Overlapping same-team players merge into one jersey blob. Each head belongs to the blob it sits
  on; a blob carrying two or more heads is split into one box per head, using body/head proportions
  learned online from single figures (feet from the blob when they agree with the proportions,
  else predicted).
* A single figure whose blob ends on another person (lower body hidden behind them) gets
  predicted feet.
* Embeddings are torso histograms of the blob's own pixels, so occluders do not leak in.
* The scoreboard panel (near-black background, red lit segments) is masked out for people and
  rims with ``mask_polygons`` / ``add_mask_polygon`` / ``remove_mask_polygon`` (the ball may fly
  in front of it, so balls are not masked); blobs also have shape filters as a second guard.

Production detection is ``RFDETRDetector`` (``rfdetr_adapter.py``); both satisfy ``Detector``.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol, runtime_checkable

import cv2
import numpy as np

from courtvision.sim import palette
from courtvision.types import BBox, Detection, FrameDetections, ObjClass, Team

from .appearance import AppearanceConfig, torso_feature

__all__ = ["ColorBlobDetector", "Detector", "DetectorConfig"]

log = logging.getLogger(__name__)


@runtime_checkable
class Detector(Protocol):
    """Anything that turns a BGR frame into detections."""

    def detect(self, frame_bgr: np.ndarray, frame_idx: int, t: float) -> FrameDetections: ...


@dataclass
class DetectorConfig:
    """``ColorBlobDetector`` settings. Colours come from the shared palette; tolerances are tuned
    on noise-free synthetic renders (status: tuned) and are per-channel (L-infinity) in BGR."""

    home_bgr: tuple[int, int, int] = palette.HOME_BGR
    away_bgr: tuple[int, int, int] = palette.AWAY_BGR
    ref_bgr: tuple[int, int, int] = palette.REF_BGR
    ball_bgr: tuple[int, int, int] = palette.BALL_BGR
    rim_bgr: tuple[int, int, int] = palette.RIM_BGR
    skin_bgr: tuple[int, int, int] = palette.SKIN_BGR
    floor_bgr: tuple[int, int, int] = palette.FLOOR_BGR
    paint_bgr: tuple[int, int, int] = palette.PAINT_BGR
    jersey_tol: int = 40  # nearest other palette colour is > 100 away; tuned
    ref_tol: int = 20  # scoreboard background (20,20,20) is only 5 away: mask the panel; tuned
    ball_tol: int = 40  # rim red is 110 away in G; tuned
    rim_tol: int = 30  # lit scoreboard segments (0,0,255) are 35 away; tuned
    skin_tol: int = 7  # paint (110,140,190) is only 10 away from skin; tuned
    min_person_area_px: int = 25  # jersey/shirt blob pixels; tuned
    min_person_aspect: float = 0.6  # jersey blob h/w; rejects wide panels and court stripes; tuned
    min_ball_area_px: int = 3  # tuned
    max_ball_area_px: int = 4000  # tuned
    max_ball_aspect: float = 2.5  # max(w, h) / min(w, h) of a ball blob; tuned
    min_rim_area_px: int = 6  # tuned
    rim_merge_px: int = 5  # dilation that joins pieces of one rim ring split by net/ball; tuned
    close_kernel_wh: tuple[int, int] = (3, 7)  # bridges white number strokes across the jersey; tuned
    head_search_frac: float = 0.8  # search for a head up to this x jersey height above the blob; tuned
    leg_search_frac: float = 0.8  # same below the blob (legs drawn in skin colour, if any); tuned
    touch_gap_frac: float = 0.08  # max gap (x jersey height) between blob and head/legs; tuned
    fallback_head_frac: float = 0.25  # no head found: extend up by this x jersey height; tuned
    full_conf_area_px: float = 300.0  # blobs at least this large get score 0.95; tuned
    split_merged: bool = True  # split a same-colour blob that carries >= 2 heads (overlapping players)
    # Body / head-width ratios used until learned online; measured on sim/render.py output
    # (head radius 0.07 x height; the anti-aliased head edge trims ~1 px); tuned.
    default_width_per_head: float = 3.2
    default_height_per_head: float = 6.7
    proportion_samples: int = 200  # single-figure blobs kept for the online proportion medians; tuned
    split_bottom_tol: float = 0.12  # observed feet within this x height of the predicted ones are used; tuned
    court_tol: int = 6  # floor / paint pixels (feet visible below a blob); skin is 10 from paint; tuned
    occlusion_scan_px: int = 8  # look this far below a blob, past white / blended pixels; tuned
    mask_polygons: list[list[list[float]]] = field(default_factory=list)  # image polygons to ignore
    mask_margin_px: int = 4  # dilation of the ignore polygons; tuned
    embed: bool = True  # attach torso_feature embeddings to players and referees


class ColorBlobDetector:
    """Palette segmentation + morphology + connected components (synthetic renders only)."""

    def __init__(self, config: DetectorConfig | None = None, appearance: AppearanceConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.appearance = appearance or AppearanceConfig()
        self._polygons: list[np.ndarray] = [np.asarray(p, dtype=np.float32) for p in self.config.mask_polygons]
        self._ignore: np.ndarray | None = None
        self._ignore_shape: tuple[int, int] | None = None
        self._width_per_head: deque[float] = deque(maxlen=self.config.proportion_samples)
        self._height_per_head: deque[float] = deque(maxlen=self.config.proportion_samples)

    @property
    def team_colors(self) -> dict[Team, tuple[int, int, int]]:
        """Jersey colours this detector segments (team-classifier anchors)."""
        return {Team.HOME: self.config.home_bgr, Team.AWAY: self.config.away_bgr}

    def add_mask_polygon(self, polygon: list[list[float]] | np.ndarray) -> None:
        """Ignore an image polygon from now on (e.g. the scoreboard panel's 4 corners)."""
        self._polygons.append(np.asarray(polygon, dtype=np.float32).reshape(-1, 2))
        self._ignore = None

    def remove_mask_polygon(self, polygon: list[list[float]] | np.ndarray) -> bool:
        """Stop ignoring a polygon added before (e.g. a scoreboard ROI the operator replaced)."""
        p = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        for i, q in enumerate(self._polygons):
            if q.shape == p.shape and np.allclose(q, p):
                del self._polygons[i]
                self._ignore = None
                return True
        return False

    def detect(self, frame_bgr: np.ndarray, frame_idx: int, t: float) -> FrameDetections:
        """Detect players, referees, the ball and rims in one BGR frame."""
        h, w = frame_bgr.shape[:2]
        ignore = self._ignore_mask((h, w))
        cfg = self.config
        skin = self._mask(frame_bgr, cfg.skin_bgr, cfg.skin_tol, ignore)
        groups = [
            self._blobs(frame_bgr, cfg.home_bgr, cfg.jersey_tol, ObjClass.PLAYER, ignore),
            self._blobs(frame_bgr, cfg.away_bgr, cfg.jersey_tol, ObjClass.PLAYER, ignore),
            self._blobs(frame_bgr, cfg.ref_bgr, cfg.ref_tol, ObjClass.REFEREE, ignore),
        ]
        owners: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        if cfg.split_merged:
            heads = cv2.connectedComponentsWithStats(skin, connectivity=8)[2]
            owners = self._assign_heads(heads, groups)
        people = skin > 0  # any person pixel: shirts of every colour and skin
        for group in groups:
            people |= group.labels > 0
        court = (self._mask(frame_bgr, cfg.floor_bgr, cfg.court_tol, None) > 0) | (
            self._mask(frame_bgr, cfg.paint_bgr, cfg.court_tol, None) > 0)
        people_court = (people, court)
        dets: list[Detection] = []
        for g, group in enumerate(groups):
            dets += self._people(frame_bgr, g, group, owners, skin, people_court)
        dets += self._balls(frame_bgr, None)  # orange never occurs in the panel; a ball may pass in front
        dets += self._rims(frame_bgr, ignore)
        return FrameDetections(frame_idx=frame_idx, t=t, image_size=(w, h), detections=dets)

    # ------------------------------------------------------------------ masks

    def _ignore_mask(self, shape: tuple[int, int]) -> np.ndarray | None:
        if not self._polygons:
            return None
        if self._ignore is None or self._ignore_shape != shape:
            m = np.zeros(shape, dtype=np.uint8)
            for poly in self._polygons:
                cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255)
            k = 2 * self.config.mask_margin_px + 1
            if self.config.mask_margin_px > 0:
                m = cv2.dilate(m, np.ones((k, k), np.uint8))
            self._ignore, self._ignore_shape = m, shape
        return self._ignore

    @staticmethod
    def _mask(frame: np.ndarray, bgr: tuple[int, int, int], tol: int, ignore: np.ndarray | None) -> np.ndarray:
        c = np.asarray(bgr, dtype=int)
        lo = np.clip(c - tol, 0, 255).astype(np.uint8)
        hi = np.clip(c + tol, 0, 255).astype(np.uint8)
        m = cv2.inRange(frame, lo, hi)
        if ignore is not None:
            m[ignore > 0] = 0
        return m

    # ------------------------------------------------------------------ people

    def _blobs(
        self, frame: np.ndarray, colour: tuple[int, int, int], tol: int, cls: ObjClass, ignore: np.ndarray | None
    ) -> _Blobs:
        kw, kh = self.config.close_kernel_wh
        mask = self._mask(frame, colour, tol, ignore)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        return _Blobs(cls, n, labels, stats)

    def _people(
        self,
        frame: np.ndarray,
        g: int,
        group: _Blobs,
        owners: dict[tuple[int, int], list[tuple[int, int, int, int]]],
        skin: np.ndarray,
        people_court: tuple[np.ndarray, np.ndarray],
    ) -> list[Detection]:
        cfg = self.config
        out: list[Detection] = []
        for i in range(1, group.n):
            x, y, bw, bh, area = (int(v) for v in group.stats[i])
            if area < cfg.min_person_area_px or bh < cfg.min_person_aspect * bw:
                continue
            blob = group.labels[y:y + bh, x:x + bw] == i
            on_blob = _dedupe_heads(owners.get((g, i), []))
            if len(on_blob) >= 2:
                boxes = [_pixel_box(*b) for b in self._split(blob, x, y, on_blob)]
                score = 0.8 * self._area_score(area / len(on_blob))
            else:
                x1, y1, x2, y2 = self._extend_with_skin(skin, x, y, x + bw, y + bh)
                hidden_feet = self._ends_on_someone(*people_court, blob, x, y)
                if len(on_blob) == 1 and hidden_feet:  # lower body behind another person: predict the feet
                    hx1, hy1, hx2, _ = on_blob[0]
                    y2 = max(y2, int(round(hy1 + self._proportions()[1] * (hx2 - hx1))))
                elif len(on_blob) == 1 and area >= 0.75 * bw * bh:  # clean single figure: learn proportions
                    hx1, hy1, hx2, _ = on_blob[0]
                    self._width_per_head.append(bw / (hx2 - hx1))
                    self._height_per_head.append((y2 - hy1) / (hx2 - hx1))
                boxes, score = [_pixel_box(x1, y1, x2, y2)], self._area_score(area)
            own = None
            if cfg.embed:  # the blob's own pixels only: an occluder must not leak into the colour feature
                own = np.zeros(group.labels.shape, dtype=np.uint8)
                own[y:y + bh, x:x + bw] = blob
            for box in boxes:
                emb = torso_feature(frame, box, self.appearance, mask=own) if cfg.embed else None
                out.append(Detection(bbox=box, cls=group.cls, score=score, embedding=emb))
        return out

    def _assign_heads(
        self, head_stats: np.ndarray, groups: list[_Blobs]
    ) -> dict[tuple[int, int], list[tuple[int, int, int, int]]]:
        """Each head-shaped skin component goes to the one blob (any colour) with the most pixels
        directly under it, i.e. the body it sits on. Returns (group, label) -> head boxes."""
        owners: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        for j in range(1, len(head_stats)):
            hx, hy, hw, hh, area = (int(v) for v in head_stats[j])
            if area < 4 or not 0.5 <= hw / max(hh, 1) <= 2.0:
                continue
            bottom = hy + hh
            band = (slice(max(0, bottom - 1), bottom + max(2, hh // 2 + 1)), slice(hx, hx + hw))
            best: tuple[int, int, int] | None = None  # (columns covered, group, label)
            for g, group in enumerate(groups):
                lab = group.labels[band]
                if lab.size == 0:
                    continue
                for label in np.unique(lab[lab > 0]):
                    cover = int((lab == label).any(axis=0).sum())
                    if best is None or cover > best[0]:
                        best = (cover, g, int(label))
            if best is None or best[0] < 0.3 * hw:
                continue
            _, g, label = best
            bx, by, bw, bh, _ = (int(v) for v in groups[g].stats[label])
            gap = max(2, int(math.ceil(self.config.touch_gap_frac * bh)))
            if bx <= hx + hw / 2.0 < bx + bw and by - gap <= bottom <= by + 0.7 * bh:
                owners.setdefault((g, label), []).append((hx, hy, hx + hw, bottom))
        return owners

    def _proportions(self) -> tuple[float, float]:
        """(body width, body height) per head width: online medians, else the defaults."""
        cfg = self.config
        if len(self._width_per_head) >= 5:
            return float(np.median(self._width_per_head)), float(np.median(self._height_per_head))
        return cfg.default_width_per_head, cfg.default_height_per_head

    def _ends_on_someone(self, people: np.ndarray, court: np.ndarray, blob: np.ndarray, bx: int, by: int) -> bool:
        """True when the blob's bottom edge rests on another person: in each column, scan down from
        the lowest blob pixel past neutral pixels (white numbers, blends) to the first decisive one;
        another shirt or skin means occluded, floor or paint means the feet are visible."""
        cols = np.nonzero(blob.any(axis=0))[0]
        if cols.size == 0:
            return False
        lowest = blob.shape[0] - 1 - np.argmax(blob[::-1, cols], axis=0)
        rows = by + lowest[None, :] + 1 + np.arange(self.config.occlusion_scan_px)[:, None]
        inside = rows < people.shape[0]
        rows = np.minimum(rows, people.shape[0] - 1)
        xs = np.broadcast_to(bx + cols[None, :], rows.shape)
        person = people[rows, xs] & inside
        floor = court[rows, xs] & inside
        decisive = person | floor
        has = decisive.any(axis=0)
        if has.sum() < 3:
            return False
        first = np.argmax(decisive, axis=0)
        occluded = person[first, np.arange(cols.size)] & has
        return bool(occluded.sum() >= 0.5 * has.sum())

    def _split(
        self, blob: np.ndarray, bx: int, by: int, heads: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """One box per head in a merged blob, from body proportions learned on single figures."""
        cfg = self.config
        wph, hph = self._proportions()
        bh, bw = blob.shape
        out = []
        for hx1, hy1, hx2, _ in heads:
            head_w = hx2 - hx1
            cx = (hx1 + hx2) / 2.0
            x1 = max(bx, int(round(cx - wph * head_w / 2.0)))
            x2 = min(bx + bw, int(round(cx + wph * head_w / 2.0)))
            pred_bottom = hy1 + hph * head_w
            c1, c2 = int(round(cx - wph * head_w / 4.0)) - bx, int(round(cx + wph * head_w / 4.0)) - bx
            rows = np.nonzero(blob[:, max(0, c1):max(c1 + 1, min(bw, c2))].any(axis=1))[0]
            obs_bottom = by + int(rows.max()) + 1 if rows.size else pred_bottom
            tol = cfg.split_bottom_tol * (pred_bottom - hy1)
            bottom = obs_bottom if abs(obs_bottom - pred_bottom) <= tol else int(round(pred_bottom))
            out.append((x1, hy1, x2, int(bottom)))
        return out

    def _extend_with_skin(self, skin: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
        """Grow a jersey blob box (exclusive x2/y2) to the head above and legs below."""
        cfg = self.config
        jh = y2 - y1
        gap = max(2, int(math.ceil(cfg.touch_gap_frac * jh)))
        above = y1 - int(math.ceil(cfg.head_search_frac * jh))
        below = y2 + int(math.ceil(cfg.leg_search_frac * jh))
        top = self._skin_run(skin, x1, x2, above, y1 + gap, y1 - gap, up=True)
        bottom = self._skin_run(skin, x1, x2, y2 - gap, below, y2 + gap, up=False)
        nx1, nx2 = x1, x2
        if top is None:
            ny1 = int(round(y1 - cfg.fallback_head_frac * jh))
        else:
            ny1 = top[0]
            nx1, nx2 = min(nx1, top[1]), max(nx2, top[2])
        ny2 = y2
        if bottom is not None:
            ny2 = bottom[0]
            nx1, nx2 = min(nx1, bottom[1]), max(nx2, bottom[2])
        return nx1, max(0, ny1), nx2, ny2

    @staticmethod
    def _skin_run(
        skin: np.ndarray, x1: int, x2: int, wy1: int, wy2: int, touch_y: int, up: bool
    ) -> tuple[int, int, int] | None:
        """Skin components in rows [wy1, wy2) that touch the blob edge ``touch_y`` and are centred
        over its columns. Up: the single component nearest the blob centre (one head per body; a
        neighbour's head is not merged). Down: the union (two legs). Returns (extreme row, min x,
        max x exclusive) or None."""
        h, w = skin.shape
        pad = max(2, (x2 - x1) // 3)
        wx1, wx2 = max(0, x1 - pad), min(w, x2 + pad)
        wy1, wy2 = max(0, wy1), min(h, wy2)
        if wy2 <= wy1 or wx2 <= wx1:
            return None
        win = skin[wy1:wy2, wx1:wx2]
        if not win.any():
            return None
        n, _, stats, _ = cv2.connectedComponentsWithStats(win, connectivity=8)
        centre = (x1 + x2) / 2.0
        found: list[tuple[float, int, int, int]] = []  # (|offset|, edge row, x1, x2)
        for i in range(1, n):
            cx, cy, cw, ch, _area = (int(v) for v in stats[i])
            gx1, gx2 = wx1 + cx, wx1 + cx + cw
            gy1, gy2 = wy1 + cy, wy1 + cy + ch  # exclusive
            mid = (gx1 + gx2) / 2.0
            if not x1 <= mid <= x2:
                continue
            if (up and gy2 < touch_y) or (not up and gy1 > touch_y):
                continue
            found.append((abs(mid - centre), gy1 if up else gy2, gx1, gx2))
        if not found:
            return None
        if up:
            _, edge, gx1, gx2 = min(found)
            return edge, gx1, gx2
        return max(f[1] for f in found), min(f[2] for f in found), max(f[3] for f in found)

    def _area_score(self, area: float) -> float:
        return float(min(0.95, 0.3 + 0.65 * min(1.0, area / self.config.full_conf_area_px)))

    # ------------------------------------------------------------------ ball and rims

    def _balls(self, frame: np.ndarray, ignore: np.ndarray | None) -> list[Detection]:
        cfg = self.config
        mask = self._mask(frame, cfg.ball_bgr, cfg.ball_tol, ignore)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out: list[Detection] = []
        for i in range(1, n):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            if not cfg.min_ball_area_px <= area <= cfg.max_ball_area_px:
                continue
            if max(bw, bh) > cfg.max_ball_aspect * max(1, min(bw, bh)):
                continue
            fill = area / float(bw * bh)  # a disc fills pi/4 of its box
            score = 0.9 if 0.55 <= fill <= 0.95 else 0.5
            out.append(Detection(bbox=_pixel_box(x, y, x + bw, y + bh), cls=ObjClass.BALL, score=score))
        out.sort(key=lambda d: -d.score)
        return out

    def _rims(self, frame: np.ndarray, ignore: np.ndarray | None) -> list[Detection]:
        cfg = self.config
        mask = self._mask(frame, cfg.rim_bgr, cfg.rim_tol, ignore)
        if not mask.any():
            return []
        k = 2 * cfg.rim_merge_px + 1
        joined = cv2.dilate(mask, np.ones((k, k), np.uint8)) if cfg.rim_merge_px > 0 else mask
        n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
        out: list[Detection] = []
        for i in range(1, n):
            x, y, bw, bh, _ = (int(v) for v in stats[i])
            sub = (labels[y:y + bh, x:x + bw] == i) & (mask[y:y + bh, x:x + bw] > 0)
            ys, xs = np.nonzero(sub)
            ys, xs = ys + y, xs + x
            if len(xs) < cfg.min_rim_area_px:
                continue
            box = _pixel_box(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            out.append(Detection(bbox=box, cls=ObjClass.RIM, score=0.9))
        return out


def _pixel_box(x1: int, y1: int, x2: int, y2: int) -> BBox:
    """Box of pixel columns [x1, x2) and rows [y1, y2) in continuous image coordinates, where
    pixel centres are integers (OpenCV and ``PinholeCamera.project``): edges sit at +-0.5."""
    return BBox(x1 - 0.5, y1 - 0.5, x2 - 0.5, y2 - 0.5)


class _Blobs(NamedTuple):
    """Connected components of one shirt colour in one frame."""

    cls: ObjClass
    n: int
    labels: np.ndarray
    stats: np.ndarray


def _dedupe_heads(heads: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Sort heads top-down and merge pieces of one head (e.g. split by the ball)."""
    merged: list[tuple[int, int, int, int]] = []
    for b in sorted(heads, key=lambda b: (b[1], b[0])):
        if merged:
            m = merged[-1]
            if abs((b[0] + b[2]) - (m[0] + m[2])) / 2.0 < 0.6 * (b[2] - b[0]) and b[1] < m[3]:
                merged[-1] = (min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3]))
                continue
        merged.append(b)
    return merged
