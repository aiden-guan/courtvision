"""Court registration through a broadcast game: one fixed camera centre, pan / tilt / roll / zoom per frame.

``BroadcastCourtTracker.update(frame) -> CourtFix`` follows the game camera (``court_search``):

* **Tracking.** The line evidence of the frame (at ``work_width``) refines the previous frame's camera, predicted
  forward with its velocity (pan / tilt / roll / focal only: the centre is fixed for the game). The evidence is
  seen through the predicted camera (``court_search.restrict``): the projected court plus ``track_margin_ft`` is
  the floor, so the stands and boards drop out and lines close together (the key) no longer mask themselves out
  of the frame-wide clutter test (synthetic benchmark, ``tools/court_bench.py``). A fit is accepted
  when enough of the court's lines land on line pixels (``min_coverage``) and the change is one a camera
  operator makes in one frame; otherwise the last camera is held for ``hold_frames`` and then the court is lost.
* **Cuts.** On a ``cut`` (from a shot-boundary detector) the last camera is refined first and kept when it still
  fits as well as a re-acquisition must (most flagged cuts in a game are fast pans or flashes, not cuts); only
  then is it dropped.
* **Re-acquisition** while lost (also ``reacquire_after_loss`` frames after a loss in play, e.g. the blur of a
  fast pan): every ``reacquire_every`` frames a pan / tilt / zoom search near the last good
  camera (``reacquire_around_deg``), then over the whole prior; accepted with ``reacquire_quality`` (coverage x
  explained lines) and only for a camera the game camera can be (``court_search.plausible``: focal length in the
  prior's field of view, looking down into the aim box, nearly level, a tight fit). Replays, close-ups and other
  cameras (a different centre) fail it and stay "lost", which the broadcast gate treats as not live.

``params_from_homography`` fits the camera at the known centre to an operator's calibration (the tracker restarts
from it). ``estimate_center`` finds the game camera's centre from sample frames (``court_search.calibrate_frame`` with
the centre free, then ``fit_center`` on the good ones). Thresholds are design choices checked on ESPN frames of
Lakers @ Warriors (2024-12-25).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from courtvision.geometry.court import CourtSpec
from courtvision.geometry.homography import Homography

from .court_search import (
    BroadcastPrior,
    CameraParams,
    Evidence,
    EvidenceConfig,
    RefineConfig,
    _angles_to,
    _project,
    calibrate_frame,
    court_evidence,
    explained,
    fit_center,
    focal_range,
    plausible,
    refine,
    restrict,
    with_lines,
)

__all__ = ["BroadcastCourtTracker", "BroadcastTrackerConfig", "CourtFix", "estimate_center", "params_from_homography"]

log = logging.getLogger(__name__)


@dataclass
class BroadcastTrackerConfig:
    work_width: int = 960  # evidence resolution (the full-resolution camera is scaled back)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    track_refine: RefineConfig = field(default_factory=lambda: RefineConfig(search_px=(40, 16, 6, 3), max_nfev=25))
    # Tracked frames see the floor through the predicted camera (``court_search.restrict``): the projected court
    # grown by ``track_margin_ft``, minus clutter denser than ``track_density`` (0 margin: the frame-wide mask).
    track_margin_ft: float = 4.0
    track_density: float = 0.30
    min_coverage: float = 0.40  # tracked fit: share of the court samples in view on a line
    min_inliers: int = 20
    max_rmse_px: float = 2.5  # 1080-line px
    max_pan_step: float = 3.0  # degrees per frame
    max_tilt_step: float = 1.5
    max_roll_step: float = 0.5
    max_zoom_step: float = 0.08  # relative focal change per frame
    hold_frames: int = 4
    # Re-acquisition while lost: at a cut and a few frames after it (the game camera comes back on a cut), then
    # every ``reacquire_every`` frames; a frame must show some floor with lines first (close-ups do not).
    reacquire_after_cut: tuple[int, ...] = (0, 2, 6, 15)
    reacquire_after_loss: tuple[int, ...] = (1, 4, 10, 20)  # lost frames after a loss in play (the blur of a pan)
    reacquire_every: int = 30
    min_floor_share: float = 0.12
    min_line_share: float = 0.002
    reacquire_around_deg: float = 25.0
    full_search_every: int = 3  # every n-th re-acquisition attempt searches the whole prior
    memory: int = 24  # recent good cameras (distinct views) tried first on re-acquisition, cheap refinements
    memory_min_sep_deg: float = 4.0
    reacquire_quality: float = 0.10
    reacquire_coverage: float = 0.5
    max_roll_deg: float = 3.0  # plausibility of a re-acquired camera (court_search.plausible)


@dataclass(frozen=True)
class CourtFix:
    """The court in one frame: ``params`` in full-resolution pixels (None while lost) and how it was found."""

    status: str  # "tracking" | "held" | "reacquired" | "lost"
    params: CameraParams | None
    coverage: float = 0.0
    inliers: int = 0
    rmse_px: float | None = None

    def homography(self, size: tuple[int, int]) -> Homography | None:
        """Image -> court floor homography (full resolution)."""
        if self.params is None:
            return None
        return Homography(np.linalg.inv(self.params.G(size)))


def _scaled(p: CameraParams, s: float) -> CameraParams:
    return CameraParams(np.asarray(p.center, float).copy(), p.pan, p.tilt, p.roll, p.f * s)


class BroadcastCourtTracker:
    """``update(frame_bgr, cut=False, people=None) -> CourtFix`` (see the module docstring)."""

    def __init__(self, court: CourtSpec, image_size: tuple[int, int], center: np.ndarray,
                 config: BroadcastTrackerConfig | None = None, prior: BroadcastPrior | None = None) -> None:
        self.court = court
        self.config = config or BroadcastTrackerConfig()
        self.prior = prior or BroadcastPrior()
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.center = np.asarray(center, float)
        self.scale = min(1.0, self.config.work_width / self.image_size[0])
        self._last: CameraParams | None = None  # work resolution
        self._last_good: CameraParams | None = None
        self._vel = np.zeros(4)  # pan, tilt, roll, log f per frame
        self._misses = 0
        self._lost_frames = 0
        self._since_cut = 10**6
        self._loss_in_play = False  # lost without a cut (the blur of a fast pan, a flash)
        self._attempts = 0
        self._memory: list[CameraParams] = []  # work resolution, most recent first
        self.status = "lost"

    def reset(self, params: CameraParams | None = None) -> None:
        """Forget the camera (or start from ``params``, full resolution)."""
        self._last = None if params is None else _scaled(params, self.scale)
        self._last_good = self._last
        self._vel[:] = 0.0
        self._misses = self._lost_frames = self._attempts = 0
        self.status = "lost" if params is None else "tracking"

    def reset_to_homography(self, hom: Homography, size: tuple[int, int] | None = None) -> CameraParams | None:
        """Restart from an operator's calibration (image -> court homography): the camera at the known centre that
        matches it (``params_from_homography``), or from scratch (lost) when none does. Returns that camera."""
        p = params_from_homography(hom, self.center, size or self.image_size, self.court, self.prior)
        self.reset(p)
        return p

    @property
    def params(self) -> CameraParams | None:
        """The current camera at full resolution (None while lost)."""
        return None if self._last is None or self.status == "lost" else _scaled(self._last, 1.0 / self.scale)

    def evidence(self, frame_bgr: np.ndarray, people: list[tuple[float, float, float, float]] | None = None,
                 lines: bool = True) -> Evidence:
        img = frame_bgr
        boxes = people
        if self.scale < 1.0:
            w = int(round(frame_bgr.shape[1] * self.scale))
            h = int(round(frame_bgr.shape[0] * self.scale))
            img = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
            boxes = None if people is None else [tuple(v * self.scale for v in b) for b in people]  # type: ignore[misc]
        return court_evidence(img, self.config.evidence, boxes, lines=lines)  # type: ignore[arg-type]

    def update(self, frame_bgr: np.ndarray, cut: bool = False,
               people: list[tuple[float, float, float, float]] | None = None) -> CourtFix:
        cfg = self.config
        ev = self.evidence(frame_bgr, people, lines=False)  # the frame-wide line pixels only when needed
        self._since_cut = 0 if cut else self._since_cut + 1
        if cut and self.status != "lost" and self._last is not None:
            pred = self._predict()
            r = refine(self._local(ev, pred), self.court, pred, config=cfg.track_refine)
            if r is not None and self._acceptable(r, pred) and self._sound(with_lines(ev, cfg.evidence), r):
                return self._accept(r, "tracking")  # not a cut: the same camera, still turning
            self.status = "lost"
            self._lost_frames = 0
            self._loss_in_play = False
        if self.status != "lost" and self._last is not None:
            pred = self._predict()
            r = refine(self._local(ev, pred), self.court, pred, config=cfg.track_refine)
            if r is not None and self._acceptable(r, pred):
                return self._accept(r, "tracking")
            self._misses += 1
            if self._misses <= cfg.hold_frames:
                self.status = "held"
                return CourtFix("held", self.params)
            self.status = "lost"
            self._lost_frames = 0
            self._loss_in_play = True
            log.debug("court lost")
        ev = with_lines(ev, cfg.evidence)
        self._lost_frames += 1
        due = (self._since_cut in cfg.reacquire_after_cut or (self._lost_frames - 1) % cfg.reacquire_every == 0
               or (self._loss_in_play and self._lost_frames in cfg.reacquire_after_loss))
        if due and ev.floor.mean() >= cfg.min_floor_share and ev.lines.mean() >= cfg.min_line_share:
            r = self._reacquire(ev)
            if r is not None:
                self._vel[:] = 0.0
                return self._accept(r, "reacquired")
        return CourtFix("lost", None)

    def _local(self, ev: Evidence, p: CameraParams) -> Evidence:
        """The frame's evidence seen through a predicted camera (``court_search.restrict``)."""
        cfg = self.config
        if cfg.track_margin_ft <= 0:
            return with_lines(ev, cfg.evidence)
        return restrict(ev, self.court, p, cfg.evidence, cfg.track_margin_ft, cfg.track_density)

    def _predict(self) -> CameraParams:
        assert self._last is not None
        k = self._misses + 1
        p = self._last
        return CameraParams(self.center * 1.0, p.pan + k * self._vel[0], p.tilt + k * self._vel[1],
                            p.roll + k * self._vel[2], p.f * math.exp(k * self._vel[3]))

    def _acceptable(self, r, pred: CameraParams) -> bool:  # noqa: ANN001 (RefineResult)
        cfg = self.config
        if r.coverage < cfg.min_coverage or r.inliers < cfg.min_inliers or r.rmse_px > cfg.max_rmse_px:
            return False
        last = self._last
        assert last is not None
        k = self._misses + 1
        p = r.params
        return (abs(p.pan - last.pan) <= cfg.max_pan_step * k and abs(p.tilt - last.tilt) <= cfg.max_tilt_step * k
                and abs(p.roll - last.roll) <= cfg.max_roll_step * k
                and abs(math.log(p.f / last.f)) <= cfg.max_zoom_step * k)

    def _sound(self, ev: Evidence, r) -> bool:  # noqa: ANN001 (RefineResult)
        """A fit good enough to stand without the previous frame: a re-acquisition's coverage, quality and inliers,
        a tight fit and a camera the game camera can be (sets ``r.quality``)."""
        cfg = self.config
        if r.coverage < cfg.reacquire_coverage or r.inliers < cfg.min_inliers or r.rmse_px > cfg.max_rmse_px:
            return False
        p = CameraParams(self.center * 1.0, r.params.pan, r.params.tilt, r.params.roll, r.params.f)
        if not plausible(p, ev.size, self.prior, max_roll_deg=cfg.max_roll_deg):
            return False
        r.quality = r.coverage * explained(ev, self.court, r.params)
        return r.quality >= cfg.reacquire_quality

    def _accept(self, r, status: str) -> CourtFix:  # noqa: ANN001 (RefineResult)
        p = r.params
        if self._last is not None and status == "tracking":
            k = self._misses + 1
            v = np.array([(p.pan - self._last.pan) / k, (p.tilt - self._last.tilt) / k, (p.roll - self._last.roll) / k,
                          math.log(p.f / self._last.f) / k])
            self._vel = 0.5 * self._vel + 0.5 * v
        self._last = CameraParams(self.center * 1.0, p.pan, p.tilt, p.roll, p.f)
        self._last_good = self._last
        self._remember(self._last)
        self._misses = self._lost_frames = 0
        self.status = "tracking" if status == "tracking" else "reacquired"
        return CourtFix(status, self.params, r.coverage, r.inliers, r.rmse_px)

    def _remember(self, p: CameraParams) -> None:
        cfg = self.config
        keep = [q for q in self._memory if abs(q.pan - p.pan) > cfg.memory_min_sep_deg
                or abs(q.tilt - p.tilt) > cfg.memory_min_sep_deg / 2 or abs(math.log(q.f / p.f)) > 0.15]
        self._memory = [p, *keep][: cfg.memory]

    def _reacquire(self, ev: Evidence):  # noqa: ANN202 (RefineResult | None)
        cfg = self.config
        best = None
        for q in self._memory:  # the game camera usually comes back to a recent view
            r = refine(ev, self.court, q, config=cfg.track_refine)
            if r is not None and r.coverage >= cfg.reacquire_coverage and r.inliers >= cfg.min_inliers \
                    and (best is None or r.coverage * r.inliers > best.coverage * best.inliers):
                best = r
        if best is not None and self._sound(ev, best):
            best.params = CameraParams(self.center * 1.0, best.params.pan, best.params.tilt, best.params.roll,
                                       best.params.f)
            return best
        self._attempts += 1
        around = None
        if self._last_good is not None and self._attempts % cfg.full_search_every != 0:
            around = self._last_good
        r = calibrate_frame(ev, self.court, self.prior, centers=[self.center], free_center=False, around=around,
                            around_deg=cfg.reacquire_around_deg, top=4)
        if r is None or not self._sound(ev, r):
            return None
        r.params = CameraParams(self.center * 1.0, r.params.pan, r.params.tilt, r.params.roll, r.params.f)
        return r


def params_from_homography(hom: Homography, center: np.ndarray, size: tuple[int, int], court: CourtSpec,
                           prior: BroadcastPrior | None = None, max_rmse_px: float = 3.0) -> CameraParams | None:
    """The camera at ``center`` (pan / tilt / roll / focal) whose floor-to-image map best matches ``hom`` (image px
    -> court ft) on the court points it shows, or None when none matches to ``max_rmse_px`` (1080-line px): an
    operator calibration of another camera, or one that disagrees with the centre."""
    from scipy.optimize import least_squares

    pr = prior or BroadcastPrior()
    w, h = size
    xs, ys = np.meshgrid(np.linspace(0.0, court.length, 13), np.linspace(0.0, court.width, 7))
    X = np.column_stack([xs.ravel(), ys.ravel()])
    try:
        G = np.linalg.inv(hom.H)  # court -> image
    except np.linalg.LinAlgError:
        return None
    uv, wz = _project(G, X)
    if np.median(wz) < 0:  # a homography is defined up to scale, sign included
        wz = -wz
    ok = (wz > 0) & np.all(np.isfinite(uv), axis=1)
    ok &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if ok.sum() < 6:
        return None
    X, uv = X[ok], uv[ok]
    C = np.asarray(center, float)
    aim = hom.H @ np.array([w / 2.0, h / 2.0, 1.0])
    ax, ay = (aim[:2] / aim[2]) if abs(aim[2]) > 1e-12 else (court.length / 2.0, court.width / 2.0)
    if not (np.isfinite(ax) and np.isfinite(ay)) or math.hypot(ax - C[0], ay - C[1]) > 400.0:
        ax, ay = court.length / 2.0, court.width / 2.0
    pan0, tilt0 = _angles_to(C, float(ax), float(ay))

    def res(v: np.ndarray) -> np.ndarray:
        q, wq = _project(CameraParams(C, v[0], v[1], v[2], math.exp(v[3])).G(size), X)
        return np.where(wq[:, None] > 0, q - uv, 1e3).ravel()

    best: tuple[float, CameraParams] | None = None
    f_lo, f_hi = focal_range(w, pr)
    for f0 in np.geomspace(f_lo, f_hi, 6):
        sol = least_squares(res, np.array([pan0, tilt0, 0.0, math.log(f0)]), loss="soft_l1", f_scale=5.0,
                            max_nfev=200)
        rmse = float(np.sqrt(np.mean(sol.fun ** 2) * 2.0))  # per point, both coordinates
        if best is None or rmse < best[0]:
            best = (rmse, CameraParams(C.copy(), float(sol.x[0]), float(sol.x[1]), float(sol.x[2]),
                                       float(math.exp(sol.x[3]))))
    assert best is not None
    return best[1] if best[0] * 1080.0 / h <= max_rmse_px else None


def estimate_center(frames: list[np.ndarray], court: CourtSpec, work_width: int = 960,
                    evidence: EvidenceConfig | None = None, prior: BroadcastPrior | None = None,
                    min_quality: float = 0.15) -> tuple[np.ndarray | None, list[CameraParams | None]]:
    """The game camera's centre (court ft) from sample frames, and each frame's camera (full resolution; None
    where the court was not found). Frames without the game camera (close-ups, replays) are skipped."""
    evs: list[Evidence] = []
    starts: list[CameraParams] = []
    idx: list[int] = []
    scale = 1.0
    for i, img in enumerate(frames):
        scale = min(1.0, work_width / img.shape[1])
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
        ev = court_evidence(small, evidence)
        r = calibrate_frame(ev, court, prior, free_center=True)
        if r is None or r.quality < min_quality or not plausible(r.params, ev.size, prior):
            continue
        evs.append(ev)
        starts.append(r.params)
        idx.append(i)
    if len(evs) < 2:
        return None, [None] * len(frames)
    center, results = fit_center(evs, court, starts)
    out: list[CameraParams | None] = [None] * len(frames)
    for i, ev, r in zip(idx, evs, results, strict=True):
        if r is not None and r.coverage * explained(ev, court, r.params) >= 0.5 * min_quality:
            out[i] = _scaled(r.params, 1.0 / scale)
    return center, out
