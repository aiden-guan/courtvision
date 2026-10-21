"""Detector protocol and the colour-blob detector for synthetic renders.

``ColorBlobDetector`` segments the shared synthetic palette (``courtvision.sim.palette``):
jersey colours -> PLAYER, referee black -> REFEREE, orange -> BALL, red rim -> RIM. Colour
masks use per-channel boxes (``cv2.inRange``, i.e. an L-infinity tolerance).

* A player box is the jersey blob extended up to the skin-coloured head (and down to skin-coloured
  legs when present), so it matches the drawn figure's bbox.
* Overlapping same-team players merge into one jersey blob. Each head belongs to the blob it sits
  on; a blob carrying two or more heads is split into one box per head. Feet and width come from
  the fixed camera's figure model (``FigureModel``: head top -> feet homography and width, learned
  online from figures detected whole; body/head-width proportions until it is fitted), and the
  blob's own feet are used when they agree with it.
* Heads are discs: a skin component much taller than wide is heads touching vertically (a player
  right behind a teammate) and is cut into one head each; two whole discs are never merged as
  "pieces of one head".
* A single figure whose blob ends on another person (lower body hidden behind them) gets
  predicted feet. Until the figure model has its spread, a feet-row model (feet row affine in the
  head top) predicts them; in a split blob one observed bottom row belongs to one head only.
* Skin discs (heads) below a blob are never taken for its legs, and shirt pieces that one ball blob
  touches are one blob (a ball in front of a body can cut its visible shirt in two).
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

__all__ = ["ColorBlobDetector", "Detector", "DetectorConfig", "FigureModel"]

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
    # Heads (skin discs) are never legs: the leg search ignores them. On sim train seed 3, 46 % of the
    # boxes the leg search had extended ended 5-15 px too low, on a nearer player's head; design.
    legs_skip_heads: bool = True
    # A ball in front of a body can cut its visible shirt in two (an upper piece with the head whose box
    # ends at the ball, and a headless phantom below it): shirt pieces one ball blob touches are one
    # blob; design (seen on sim train seed 3 with the holder hidden behind a defender).
    bridge_ball: bool = True
    touch_gap_frac: float = 0.08  # max gap (x jersey height) between blob and head/legs; tuned
    fallback_head_frac: float = 0.25  # no head found: extend up by this x jersey height; tuned
    full_conf_area_px: float = 300.0  # blobs at least this large get score 0.95; tuned
    split_merged: bool = True  # split a same-colour blob that carries >= 2 heads (overlapping players)
    # Heads are drawn as discs; a skin component taller than this x its width is two heads touching
    # vertically (a player right behind a teammate). Cut, not merged, so the shared blob is split
    # into both players instead of one box carrying two ids; design (sim renders), None = off.
    stacked_head_ratio: float | None = 1.25
    # Heads touching at an angle: a skin component filling less than this share of its box is several
    # discs (one disc fills pi/4 = 0.785), split at the peaks of its distance transform; design (sim
    # renders draw heads as discs), None = off.
    head_disc_fill: float | None = 0.65
    # A blob with two or more heads hands one to a touching headless blob (another body side by side);
    # design (sim renders: the nearer shirt covers most of the neighbour's head bottom).
    headless_takes_head: bool = True
    # Body / head-width ratios used until learned online; measured on sim/render.py output
    # (head radius 0.07 x height; the anti-aliased head edge trims ~1 px); tuned.
    default_width_per_head: float = 3.2
    default_height_per_head: float = 6.7
    proportion_samples: int = 200  # single-figure blobs kept for the online proportion medians; tuned
    split_bottom_tol: float = 0.12  # observed feet within this x height of the predicted ones are used; tuned
    # Fixed-camera figure model, learned online from clean single figures (``FigureModel``): head top ->
    # feet homography and box width as a quadratic in the foot point. Split and hidden-feet boxes take
    # their feet / width from it instead of the head-width proportions (whose 1-px head quantisation
    # moved predicted feet by up to ~10 px). On truth boxes of a sim game the feet residual is 0.7 px
    # mean / 1.9 px p95 vs 10 px p90 for the proportions; design (sim train seeds).
    figure_model: bool = True
    figure_samples: int = 400  # clean single figures kept for the fit; tuned
    figure_min_samples: int = 40  # the model is used only after this many samples; tuned
    figure_refit_every: int = 25  # refit after this many new samples; tuned
    figure_max_resid_px: float = 2.5  # ... and only while its median feet residual is below this; tuned
    figure_min_cells: int = 12  # ... and its feet samples cover this many 64-px image cells (spread); tuned
    figure_bottom_tol: float = 0.06  # observed feet within this x height of the model's are used; tuned
    # Until the full model has its spread: feet row = a + b * head-top row + c * head-top column (a fixed
    # camera without roll; on the sim camera this fits standing figures to 0.3 px), from this many clean
    # figures whose head tops span this many rows. The head-width proportions it replaces put the feet
    # 4-12 px off (a 1-px head-width error is ~6 px of feet), which moved a hidden player behind a
    # teammate by ~1.5 ft or onto the teammate's feet; design (pinhole geometry), counts tuned.
    figure_row_min_samples: int = 10
    figure_row_min_span_px: float = 40.0
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
        self._split_heads: set[tuple[int, int, int, int]] = set()  # this frame's heads cut from one component
        self._width_per_head: deque[float] = deque(maxlen=self.config.proportion_samples)
        self._height_per_head: deque[float] = deque(maxlen=self.config.proportion_samples)
        cfg = self.config
        self.figure = FigureModel(cfg.figure_samples, cfg.figure_min_samples, cfg.figure_refit_every,
                                  cfg.figure_max_resid_px, cfg.figure_min_cells,
                                  row_min_samples=cfg.figure_row_min_samples,
                                  row_min_span_px=cfg.figure_row_min_span_px) if cfg.figure_model else None

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
        bridge = self._mask(frame_bgr, cfg.ball_bgr, cfg.ball_tol, ignore) if cfg.bridge_ball else None
        groups = [
            self._blobs(frame_bgr, cfg.home_bgr, cfg.jersey_tol, ObjClass.PLAYER, ignore, bridge),
            self._blobs(frame_bgr, cfg.away_bgr, cfg.jersey_tol, ObjClass.PLAYER, ignore, bridge),
            self._blobs(frame_bgr, cfg.ref_bgr, cfg.ref_tol, ObjClass.REFEREE, ignore, bridge),
        ]
        owners: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        head_stats = self._head_components(skin)
        if cfg.split_merged:
            owners = self._assign_heads(head_stats, groups)
        legs = self._without_heads(skin, head_stats) if cfg.legs_skip_heads else skin
        people = skin > 0  # any person pixel: shirts of every colour and skin
        for group in groups:
            people |= group.labels > 0
        court = (self._mask(frame_bgr, cfg.floor_bgr, cfg.court_tol, None) > 0) | (
            self._mask(frame_bgr, cfg.paint_bgr, cfg.court_tol, None) > 0)
        people_court = (people, court)
        dets: list[Detection] = []
        for g, group in enumerate(groups):
            dets += self._people(frame_bgr, g, group, owners, (skin, legs), people_court)
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
        self, frame: np.ndarray, colour: tuple[int, int, int], tol: int, cls: ObjClass, ignore: np.ndarray | None,
        bridge: np.ndarray | None = None,
    ) -> _Blobs:
        kw, kh = self.config.close_kernel_wh
        mask = self._mask(frame, colour, tol, ignore)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if bridge is not None and n > 2 and bridge.any():
            self._bridge(labels, stats, bridge)
        return _Blobs(cls, n, labels, stats)

    @staticmethod
    def _bridge(labels: np.ndarray, stats: np.ndarray, bridge: np.ndarray) -> None:
        """Join the shirt components that one ball-coloured component touches: a ball in front of a
        body (held, dribbled, or in flight) can cut its visible shirt in two, leaving the head on an
        upper piece whose box ends at the ball (feet several ft off) and a headless lower piece (a
        phantom player). The ball pixels stay out of the blob; the joined pieces keep the lowest
        label and the union box, the others get area 0 (skipped). Blobs of a team that the ball cuts
        between two teammates join too: that is the merged-blob case the head split handles."""
        nb, blabels, bstats, _ = cv2.connectedComponentsWithStats(bridge, connectivity=8)
        h, w = labels.shape
        for j in range(1, nb):
            x, y, bw, bh, _ = (int(v) for v in bstats[j])
            y0, y1, x0, x1 = max(0, y - 1), min(h, y + bh + 1), max(0, x - 1), min(w, x + bw + 1)
            ring = cv2.dilate((blabels[y0:y1, x0:x1] == j).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            touched = np.unique(labels[y0:y1, x0:x1][ring])
            touched = [int(k) for k in touched if k > 0 and stats[k, cv2.CC_STAT_AREA] > 0]
            if len(touched) < 2:
                continue
            keep = min(touched)
            for k in touched:
                if k == keep:
                    continue
                kx, ky, kw_, kh_, ka = (int(v) for v in stats[k])
                ox, oy, ow, oh, oa = (int(v) for v in stats[keep])
                nx1, ny1 = min(ox, kx), min(oy, ky)
                nx2, ny2 = max(ox + ow, kx + kw_), max(oy + oh, ky + kh_)
                labels[ky:ky + kh_, kx:kx + kw_][labels[ky:ky + kh_, kx:kx + kw_] == k] = keep
                stats[keep] = (nx1, ny1, nx2 - nx1, ny2 - ny1, oa + ka)
                stats[k, cv2.CC_STAT_AREA] = 0

    def _people(
        self,
        frame: np.ndarray,
        g: int,
        group: _Blobs,
        owners: dict[tuple[int, int], list[tuple[int, int, int, int]]],
        skins: tuple[np.ndarray, np.ndarray],
        people_court: tuple[np.ndarray, np.ndarray],
    ) -> list[Detection]:
        cfg = self.config
        skin, legs = skins
        out: list[Detection] = []
        for i in range(1, group.n):
            x, y, bw, bh, area = (int(v) for v in group.stats[i])
            if area < cfg.min_person_area_px or bh < cfg.min_person_aspect * bw:
                continue
            blob = group.labels[y:y + bh, x:x + bw] == i
            on_blob = _dedupe_heads(owners.get((g, i), []), cfg.stacked_head_ratio, self._split_heads)
            if len(on_blob) >= 2:
                boxes = [_pixel_box(*b) for b in self._split(blob, x, y, on_blob)]
                score = 0.8 * self._area_score(area / len(on_blob))
            else:
                x1, y1, x2, y2 = self._extend_with_skin(skin, x, y, x + bw, y + bh, legs)
                hidden_feet = self._ends_on_someone(*people_court, blob, x, y)
                if len(on_blob) == 1 and hidden_feet:  # lower body behind another person: predict the feet
                    hx1, hy1, hx2, _ = on_blob[0]
                    y2 = max(y2, int(round(self._predict_figure(hx1, hy1, hx2)[0])))
                elif len(on_blob) == 1 and area >= 0.75 * bw * bh:  # clean single figure: learn proportions
                    hx1, hy1, hx2, _ = on_blob[0]
                    self._width_per_head.append(bw / (hx2 - hx1))
                    self._height_per_head.append((y2 - hy1) / (hx2 - hx1))
                    h_img, w_img = group.labels.shape
                    if self.figure is not None and x1 > 0 and y1 > 0 and x2 < w_img and y2 < h_img:
                        self.figure.add(((hx1 + hx2) / 2.0, float(hy1)), ((x1 + x2) / 2.0, float(y2)), float(x2 - x1))
                boxes, score = [_pixel_box(x1, y1, x2, y2)], self._area_score(area)
            own = None
            if cfg.embed:  # the blob's own pixels only: an occluder must not leak into the colour feature
                own = np.zeros(group.labels.shape, dtype=np.uint8)
                own[y:y + bh, x:x + bw] = blob
            for box in boxes:
                emb = torso_feature(frame, box, self.appearance, mask=own) if cfg.embed else None
                out.append(Detection(bbox=box, cls=group.cls, score=score, embedding=emb))
        return out

    def _head_components(self, skin: np.ndarray) -> np.ndarray:
        """Skin components as ``connectedComponentsWithStats`` rows (row 0 = background). A head is a
        disc, so a component more than ``stacked_head_ratio`` times taller than wide is heads touching
        vertically (a player right behind a teammate): it is cut into round(h / w) heads, each with
        the column extent of its own rows."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
        self._split_heads = set()
        ratio = self.config.stacked_head_ratio
        if ratio is None or ratio <= 0:
            return stats
        rows = [stats[0]]
        for j in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[j])
            if w < 3 or h <= ratio * w:
                discs = self._disc_peaks(labels[y:y + h, x:x + w] == j, x, y) if w >= 3 else []
                if len(discs) >= 2:
                    rows += [np.array(d, dtype=stats.dtype) for d in discs]
                    self._split_heads.update((dx, dy, dx + dw, dy + dh) for dx, dy, dw, dh, _ in discs)
                else:
                    rows.append(stats[j])
                continue
            k = max(2, int(round(h / w)))
            sub = labels[y:y + h, x:x + w] == j
            for i in range(k):
                r0, r1 = int(round(i * h / k)), int(round((i + 1) * h / k))
                cols = np.nonzero(sub[r0:r1].any(axis=0))[0]
                if cols.size == 0:
                    continue
                rows.append(np.array([x + cols[0], y + r0, cols[-1] - cols[0] + 1, r1 - r0,
                                      int(sub[r0:r1].sum())], dtype=stats.dtype))
        return np.asarray(rows)

    def _disc_peaks(self, sub: np.ndarray, x: int, y: int) -> list[tuple[int, int, int, int, int]]:
        """Heads touching at an angle (two players overlapping diagonally) form one skin component that
        a disc fills badly (a disc fills pi/4 of its box). Each head is a peak of the distance
        transform; returns one stats row per head (box of the disc around its peak, clipped to the
        component), or [] for a single disc. Without this the merged shirt blob kept one head, was not
        split, and the tracker swapped the two teammates when they parted."""
        fill_max = self.config.head_disc_fill
        h, w = sub.shape
        area = int(sub.sum())
        if fill_max is None or min(w, h) < 5 or area >= fill_max * w * h:
            return []
        pad = np.pad(sub.astype(np.uint8), 1)
        dist = cv2.distanceTransform(pad, cv2.DIST_L2, 5)[1:-1, 1:-1]
        top = float(dist.max())
        if top < 2.0:
            return []
        peak = (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)) - 1e-6) & (dist >= 0.6 * top)
        cand = sorted(((float(dist[r, c]), r, c) for r, c in zip(*np.nonzero(peak))), reverse=True)
        kept: list[tuple[float, int, int]] = []
        for v, r, c in cand:
            if all(math.hypot(r - kr, c - kc) >= 1.2 * top for _, kr, kc in kept):
                kept.append((v, r, c))
        if len(kept) < 2:
            return []
        out = []
        for v, r, c in kept[:3]:
            rad = int(math.ceil(top))
            r0, r1 = max(0, r - rad), min(h, r + rad + 1)
            c0, c1 = max(0, c - rad), min(w, c + rad + 1)
            out.append((x + c0, y + r0, c1 - c0, r1 - r0, int(sub[r0:r1, c0:c1].sum())))
        return out

    @staticmethod
    def _without_heads(skin: np.ndarray, head_stats: np.ndarray) -> np.ndarray:
        """``skin`` with every head-shaped component (a disc: the ``_assign_heads`` shape test) erased,
        for the leg search below a blob: a nearer player's head just under a blob's bottom edge is not
        that player's legs (it moved the box bottom onto the other head, 1.5-4 ft)."""
        out = skin.copy()
        for j in range(1, len(head_stats)):
            hx, hy, hw, hh, area = (int(v) for v in head_stats[j])
            if area >= 4 and 0.5 <= hw / max(hh, 1) <= 2.0:
                out[hy:hy + hh, hx:hx + hw] = 0
        return out

    def _assign_heads(
        self, head_stats: np.ndarray, groups: list[_Blobs]
    ) -> dict[tuple[int, int], list[tuple[int, int, int, int]]]:
        """Each head-shaped skin component goes to the one blob (any colour) with the most pixels
        directly under it, i.e. the body it sits on. Returns (group, label) -> head boxes."""
        owners: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        alternatives: dict[tuple[int, int, int, int], list[tuple[int, int, int]]] = {}
        for j in range(1, len(head_stats)):
            hx, hy, hw, hh, area = (int(v) for v in head_stats[j])
            if area < 4 or not 0.5 <= hw / max(hh, 1) <= 2.0:
                continue
            bottom = hy + hh
            band = (slice(max(0, bottom - 1), bottom + max(2, hh // 2 + 1)), slice(hx, hx + hw))
            cands: list[tuple[int, int, int]] = []  # (columns covered, group, label)
            for g, group in enumerate(groups):
                lab = group.labels[band]
                if lab.size == 0:
                    continue
                for label in np.unique(lab[lab > 0]):
                    cands.append((int((lab == label).any(axis=0).sum()), g, int(label)))
            cands.sort(key=lambda c: -c[0])
            if not cands or cands[0][0] < 0.3 * hw:
                continue

            def fits(c: tuple[int, int, int]) -> bool:
                bx, by, bw, bh, _ = (int(v) for v in groups[c[1]].stats[c[2]])
                gap = max(2, int(math.ceil(self.config.touch_gap_frac * bh)))
                return bx <= hx + hw / 2.0 < bx + bw and by - gap <= bottom <= by + 0.7 * bh

            if fits(cands[0]):
                head = (hx, hy, hx + hw, bottom)
                owners.setdefault((cands[0][1], cands[0][2]), []).append(head)
                alternatives[head] = [c for c in cands[1:] if c[0] >= max(2, 0.2 * hw) and fits(c)]
        if self.config.headless_takes_head:
            self._give_heads_to_headless(owners, alternatives)
        return owners

    @staticmethod
    def _give_heads_to_headless(owners: dict[tuple[int, int], list[tuple[int, int, int, int]]],
                                alternatives: dict[tuple[int, int, int, int], list[tuple[int, int, int]]]) -> None:
        """A blob carrying two or more heads gives one to a touching blob of another body that has no
        head: side by side at one depth, the nearer shirt covers more of a neighbour's head bottom than
        the neighbour's own visible shirt, so the neighbour lost its head (its box then spans the
        wrong figure) and the nearer blob was split into a phantom of the wrong team."""
        for key in sorted(owners, key=lambda k: -len(owners[k])):
            heads = owners[key]
            for head in sorted(heads, key=lambda h: -max((c[0] for c in alternatives.get(h, [])), default=0)):
                if len(heads) < 2:
                    break
                free = [c for c in alternatives.get(head, []) if (c[1], c[2]) != key and not owners.get((c[1], c[2]))]
                if not free:
                    continue
                _, g, label = free[0]
                heads.remove(head)
                owners[(g, label)] = [head]

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

    def _predict_figure(self, hx1: int, hy1: int, hx2: int) -> tuple[float, float, float]:
        """(feet row, box width, feet tolerance in px) of the figure whose head spans columns
        [hx1, hx2) from row hy1: the online figure model when it is ready, else its row model for the
        feet (width from the proportions), else the proportions."""
        cfg = self.config
        wph, hph = self._proportions()
        head_w = hx2 - hx1
        top = ((hx1 + hx2) / 2.0, float(hy1))
        pred = self.figure.predict(top) if self.figure is not None else None
        if pred is not None and pred[0] > hy1:
            bottom, width = pred
            width = float(np.clip(width, 0.5 * wph * head_w, 2.0 * wph * head_w))  # guard extrapolation
            return bottom, width, cfg.figure_bottom_tol * (bottom - hy1)
        row = self.figure.predict_feet_row(top) if self.figure is not None else None
        if row is not None and row > hy1:
            return row, wph * head_w, cfg.figure_bottom_tol * (row - hy1)
        bottom = hy1 + hph * head_w
        return bottom, wph * head_w, cfg.split_bottom_tol * (bottom - hy1)

    def _split(
        self, blob: np.ndarray, bx: int, by: int, heads: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """One box per head in a merged blob, from the figure model (or body proportions) learned on
        single figures; observed feet are used when they agree with the predicted ones."""
        bh, bw = blob.shape
        rows_out = []
        for hx1, hy1, hx2, _ in heads:
            cx = (hx1 + hx2) / 2.0
            pred_bottom, width, tol = self._predict_figure(hx1, hy1, hx2)
            x1 = max(bx, int(round(cx - width / 2.0)))
            x2 = min(bx + bw, int(round(cx + width / 2.0)))
            c1, c2 = int(round(cx - width / 4.0)) - bx, int(round(cx + width / 4.0)) - bx
            rows = np.nonzero(blob[:, max(0, c1):max(c1 + 1, min(bw, c2))].any(axis=1))[0]
            obs_bottom = by + int(rows.max()) + 1 if rows.size else None
            ok = obs_bottom is not None and abs(obs_bottom - pred_bottom) <= tol
            rows_out.append([x1, hy1, x2, pred_bottom, obs_bottom if ok else None])
        # One pair of feet per body: heads whose columns end at the same observed bottom row (a player
        # right behind a teammate) cannot both stand there; only the head whose prediction is nearest
        # keeps it, the others take their predicted feet (else the hidden player sits on the teammate).
        for i, a in enumerate(rows_out):
            if a[4] is None:
                continue
            for j, b in enumerate(rows_out):
                if j != i and b[4] is not None and abs(b[4] - a[4]) <= 1 and (
                        abs(b[3] - b[4]) < abs(a[3] - a[4]) or (abs(b[3] - b[4]) == abs(a[3] - a[4]) and j < i)):
                    a[4] = None
                    break
        return [(x1, hy1, x2, int(obs if obs is not None else round(pred))) for x1, hy1, x2, pred, obs in rows_out]

    def _extend_with_skin(self, skin: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                          legs: np.ndarray | None = None) -> tuple[int, int, int, int]:
        """Grow a jersey blob box (exclusive x2/y2) to the head above and legs below (``legs``: the
        skin mask searched below, default ``skin``)."""
        cfg = self.config
        jh = y2 - y1
        gap = max(2, int(math.ceil(cfg.touch_gap_frac * jh)))
        above = y1 - int(math.ceil(cfg.head_search_frac * jh))
        below = y2 + int(math.ceil(cfg.leg_search_frac * jh))
        top = self._skin_run(skin, x1, x2, above, y1 + gap, y1 - gap, up=True)
        bottom = self._skin_run(skin if legs is None else legs, x1, x2, y2 - gap, below, y2 + gap, up=False)
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


class FigureModel:
    """Figure geometry of a fixed camera, learned online from clean single figures.

    For people of one height standing on a plane, the image of the head top and the image of the
    feet are related by a plane-to-plane homography (both planes map to the image by homographies),
    so ``feet = G(head_top)``; box width is a smooth function of the foot point (here a quadratic).
    Both are fitted (normalised DLT / least squares, then refitted once on the inliers so jumping
    players and partly hidden figures do not bend the fit) from (head top, feet, width) samples of
    figures detected whole, and used for figures whose feet are hidden or merged with another body.
    """

    def __init__(self, max_samples: int = 400, min_samples: int = 40, refit_every: int = 25,
                 max_resid_px: float = 2.5, min_cells: int = 12, cell_px: float = 64.0,
                 row_min_samples: int = 10, row_min_span_px: float = 40.0) -> None:
        self.min_samples = min_samples
        self.refit_every = max(1, refit_every)
        self.max_resid_px = max_resid_px
        self.min_cells = min_cells  # feet in at least this many distinct cell_px grid cells: a homography
        self.cell_px = cell_px  # fitted on a few clustered spots fits them exactly and extrapolates badly
        self.row_min_samples = row_min_samples  # the feet-row fallback (``predict_feet_row``) needs this many
        self.row_min_span_px = row_min_span_px  # ... samples whose head tops span this many image rows
        self._samples: deque[tuple[float, float, float, float, float]] = deque(maxlen=max_samples)
        self._since_fit = 0
        self._since_row_fit = 0
        self._G: np.ndarray | None = None
        self._wc: np.ndarray | None = None
        self._row: np.ndarray | None = None  # feet row = [1, head u, head v] @ _row
        self._scale = 1.0
        self.residual_px: float | None = None  # median in-sample feet residual of the current fit
        self.row_residual_px: float | None = None  # same for the feet-row fallback

    @property
    def ready(self) -> bool:
        return self._G is not None

    def add(self, head_top: tuple[float, float], feet: tuple[float, float], width: float) -> None:
        """One clean single figure: head-top centre, feet (box bottom centre), box width (px)."""
        self._samples.append((head_top[0], head_top[1], feet[0], feet[1], width))
        self._since_fit += 1
        self._since_row_fit += 1
        if len(self._samples) >= self.min_samples and (self._G is None or self._since_fit >= self.refit_every):
            self.fit()
        if self._G is None and len(self._samples) >= self.row_min_samples and (
                self._row is None or self._since_row_fit >= self.refit_every):
            self.fit_rows()

    def fit(self) -> bool:
        """Refit on the stored samples; returns whether the model is usable."""
        self._since_fit = 0
        S = np.asarray(self._samples, dtype=float)
        cells = {(int(u // self.cell_px), int(v // self.cell_px)) for u, v in S[:, 2:4]}
        if len(S) < max(8, self.min_samples) or len(cells) < self.min_cells:
            self._G = self._wc = None
            return False
        self._scale = float(max(np.abs(S[:, :4]).max(), 1.0))
        keep = np.ones(len(S), dtype=bool)
        G = wc = None
        resid = np.zeros(len(S))
        for _ in range(2):  # fit, then refit on the inliers
            G = _dlt(S[keep, 0:2], S[keep, 2:4])
            if G is None:
                break
            resid = np.abs(_apply_h(G, S[:, 0:2])[:, 1] - S[:, 3])
            wc, *_ = np.linalg.lstsq(self._quad(S[keep, 2:4]), S[keep, 4], rcond=None)
            med = float(np.median(resid[keep]))
            keep = resid <= max(3.0 * med, 1.5)
            if keep.sum() < 8:
                break
        if G is None or wc is None or not np.all(np.isfinite(G)):
            self._G = self._wc = None
            self.residual_px = None
            return False
        self.residual_px = float(np.median(resid[keep])) if keep.any() else math.inf
        ok = self.residual_px <= self.max_resid_px
        self._G, self._wc = (G, wc) if ok else (None, None)
        return ok

    def fit_rows(self) -> bool:
        """Fit the feet-row fallback ``feet v = a + b * head u + c * head v`` (refitted once on the
        inliers). For a fixed camera without roll and people of one height it holds to well under a
        pixel; it needs less spread than the homography (head tops spanning ``row_min_span_px`` rows),
        so it covers the first seconds of a stream. Returns whether it is usable."""
        self._since_row_fit = 0
        self._row, self.row_residual_px = None, None
        S = np.asarray(self._samples, dtype=float)
        if len(S) < max(4, self.row_min_samples) or float(np.ptp(S[:, 1])) < self.row_min_span_px:
            return False
        A = np.c_[np.ones(len(S)), S[:, 0], S[:, 1]]
        keep = np.ones(len(S), dtype=bool)
        coef = None
        resid = np.zeros(len(S))
        for _ in range(2):  # fit, then refit on the inliers (jumps, partly hidden figures)
            coef, *_ = np.linalg.lstsq(A[keep], S[keep, 3], rcond=None)
            resid = np.abs(A @ coef - S[:, 3])
            med = float(np.median(resid[keep]))
            keep = resid <= max(3.0 * med, 1.5)
            if keep.sum() < 4:
                return False
        if coef is None or not np.all(np.isfinite(coef)):
            return False
        self.row_residual_px = float(np.median(resid[keep]))
        if self.row_residual_px > self.max_resid_px:
            return False
        self._row = coef
        return True

    def predict_feet_row(self, head_top: tuple[float, float]) -> float | None:
        """Feet row for a head top from the full model when ready, else the row fallback, else None."""
        full = self.predict(head_top)
        if full is not None:
            return full[0]
        if self._row is None:
            return None
        v = float(np.array([1.0, head_top[0], head_top[1]]) @ self._row)
        return v if math.isfinite(v) else None

    def predict(self, head_top: tuple[float, float]) -> tuple[float, float] | None:
        """(feet row, box width) for a head top, or None before a usable fit."""
        if self._G is None or self._wc is None:
            return None
        feet = _apply_h(self._G, np.array([head_top], dtype=float))[0]
        if not np.all(np.isfinite(feet)):
            return None
        width = float(self._quad(np.array([[head_top[0], feet[1]]]))[0] @ self._wc)
        return float(feet[1]), width

    def _quad(self, P: np.ndarray) -> np.ndarray:
        u, v = P[:, 0] / self._scale, P[:, 1] / self._scale
        return np.c_[np.ones(len(P)), u, v, u * u, u * v, v * v]


def _dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Homography src -> dst by the normalised direct linear transform (Hartley), or None."""
    if len(src) < 4:
        return None

    def norm(P: np.ndarray) -> np.ndarray:
        c = P.mean(axis=0)
        s = math.sqrt(2.0) / max(float(np.mean(np.linalg.norm(P - c, axis=1))), 1e-9)
        return np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])

    Ts, Td = norm(src), norm(dst)
    a = _apply_h(Ts, src)
    b = _apply_h(Td, dst)
    A = np.zeros((2 * len(a), 9))
    A[0::2, 0:2], A[0::2, 2] = -a, -1.0
    A[0::2, 6:8], A[0::2, 8] = a * b[:, :1], b[:, 0]
    A[1::2, 3:5], A[1::2, 5] = -a, -1.0
    A[1::2, 6:8], A[1::2, 8] = a * b[:, 1:], b[:, 1]
    try:
        _, _, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    H = np.linalg.inv(Td) @ vt[-1].reshape(3, 3) @ Ts
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else None


def _apply_h(H: np.ndarray, P: np.ndarray) -> np.ndarray:
    q = np.c_[P, np.ones(len(P))] @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return q[:, :2] / q[:, 2:]


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


def _dedupe_heads(heads: list[tuple[int, int, int, int]], max_ratio: float | None = None,
                  apart: set[tuple[int, int, int, int]] | None = None) -> list[tuple[int, int, int, int]]:
    """Sort heads top-down and merge pieces of one head (e.g. split by the ball). Pieces whose union
    would be more than ``max_ratio`` times taller than wide are two stacked heads, not one, and two
    heads in ``apart`` (discs separated on purpose, ``_disc_peaks``) are never merged."""
    merged: list[tuple[int, int, int, int]] = []
    for b in sorted(heads, key=lambda b: (b[1], b[0])):
        if merged:
            m = merged[-1]
            if apart and m in apart and b in apart:
                merged.append(b)
                continue
            u = (min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3]))
            disc = max_ratio is None or max_ratio <= 0 or (
                u[3] - u[1] <= max_ratio * (u[2] - u[0]) and not (_whole_disc(m) and _whole_disc(b)
                                                              and m[3] - b[1] < 0.5 * min(m[3] - m[1], b[3] - b[1])))
            if abs((b[0] + b[2]) - (m[0] + m[2])) / 2.0 < 0.6 * (b[2] - b[0]) and b[1] < m[3] and disc:
                merged[-1] = u
                continue
        merged.append(b)
    return merged


def _whole_disc(b: tuple[int, int, int, int]) -> bool:
    """A head box as wide as it is tall: a whole head, not a piece cut off by the ball."""
    w, h = b[2] - b[0], b[3] - b[1]
    return abs(w - h) <= max(2.0, 0.25 * max(w, h))
