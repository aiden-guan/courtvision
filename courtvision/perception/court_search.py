"""Automatic court registration for a broadcast game camera (no operator clicks).

An NBA broadcast's main camera sits at a fixed spot high behind the near sideline at half court and pans, tilts
and zooms to follow the play. Its image of the floor is fixed by seven numbers: the camera centre (3, fixed for
the game), pan, tilt, roll and focal length. ``search`` finds them for one frame from the painted lines alone:

1. **Line evidence** (``court_evidence``). Court lines are thin and differ in brightness from the floor around
   them, brighter (white lines) or darker (the blue lines of Chase Center): a top-hat and a black-hat of the Lab
   lightness with a kernel a little wider than a line, whichever is stronger. The crowd, benches and ad boards
   are dense with such thin structures while the floor carries only its lines, so the **floor mask** keeps the
   pixels whose neighbourhood has a low density of responses. Line pixels are the strong responses on the
   floor; their distance transform scores candidate cameras.
2. **Grid search** over a prior of broadcast cameras (``BroadcastPrior``: centre positions around half court,
   the floor point the optical axis aims at, field of view; no roll). A candidate projects points sampled along
   every court line (``court_tracker.court_samples``); on the floor mask each point scores ``g = 1 - d / tau``
   (``d`` its distance to the nearest line pixel, clipped) and ``-lam * (1 - g)`` for the miss. Points off
   the floor mask (hidden by the crowd at the bottom of the frame, by the scorebug) score nothing.
3. **Refinement** (``refine``) of the best candidates by the line search of ``CourtTracker`` (matches along the
   projected lines' normals, robust least squares over pan / tilt / roll / log focal and, optionally, the
   centre), with shrinking search radii.

``fit_center`` pools several frames: one camera centre for the game, each frame its own pan / tilt / roll /
focal. ``CourtTracker`` then follows the game from that centre and re-acquires after cuts.

The prior and thresholds are design choices checked on ESPN frames of Lakers @ Warriors (2024-12-25); other
arenas may need a wider prior.

Speed. The chamfer scores of a level are computed on several threads (``COURTVISION_THREADS``, by default up to 8;
numpy releases the GIL) and the local grids of all the candidates a level refines are scored in one call per centre.
A candidate's score does not depend on the batch or the thread it is scored in (checked bit for bit on broadcast
frames), so the search returns what one thread scoring one grid at a time did, 3.3 times faster on an M3 Pro.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

from courtvision.geometry.camera import PinholeCamera
from courtvision.geometry.court import CourtSpec

from .court_tracker import court_samples

__all__ = [
    "BroadcastPrior",
    "CameraParams",
    "Evidence",
    "EvidenceConfig",
    "court_evidence",
    "court_region",
    "coverage",
    "explained",
    "fit_center",
    "plausible",
    "refine",
    "restrict",
    "rotation",
    "search",
    "with_lines",
]

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- camera model


def rotation(pan: float, tilt: float, roll: float = 0.0) -> np.ndarray:
    """World -> camera rotation (rows right, down, forward) for pan / tilt / roll in degrees. Pan 0 looks along
    +y (across the court from the near sideline), positive tilt looks down, roll turns the image clockwise."""
    p, t = math.radians(pan), math.radians(tilt)
    f0, f1, f2 = math.cos(t) * math.sin(p), math.cos(t) * math.cos(p), -math.sin(t)
    n = max(math.hypot(f0, f1), 1e-12)
    r0, r1, r2 = f1 / n, -f0 / n, 0.0
    # down = fwd x right, written out (the products np.cross forms; it costs more than the rest of a refine step)
    R = np.array([[r0, r1, r2], [f1 * r2 - f2 * r1, f2 * r0 - f0 * r2, f0 * r1 - f1 * r0], [f0, f1, f2]])
    if roll:
        r = math.radians(roll)
        c, s = math.cos(r), math.sin(r)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ R
    return R


@dataclass
class CameraParams:
    """A broadcast camera: centre (court ft), pan / tilt / roll (deg), focal length (px)."""

    center: np.ndarray
    pan: float
    tilt: float
    roll: float
    f: float

    def K(self, size: tuple[int, int]) -> np.ndarray:
        return np.array([[self.f, 0.0, size[0] / 2.0], [0.0, self.f, size[1] / 2.0], [0.0, 0.0, 1.0]])

    def R(self) -> np.ndarray:
        return rotation(self.pan, self.tilt, self.roll)

    def G(self, size: tuple[int, int]) -> np.ndarray:
        """Court floor (x, y, 1) -> image homography."""
        R = self.R()
        t = -R @ np.asarray(self.center, float)
        return self.K(size) @ np.column_stack([R[:, 0], R[:, 1], t])

    def camera(self, size: tuple[int, int]) -> PinholeCamera:
        R = self.R()
        return PinholeCamera(K=self.K(size), R=R, t=-R @ np.asarray(self.center, float), image_size=size)

    def vector(self) -> np.ndarray:
        return np.array([*np.asarray(self.center, float), self.pan, self.tilt, self.roll, math.log(self.f)])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> CameraParams:
        return cls(np.array(v[:3], float), float(v[3]), float(v[4]), float(v[5]), float(math.exp(v[6])))

    def to_dict(self) -> dict[str, float | list[float]]:
        return {"center": [float(c) for c in self.center], "pan": self.pan, "tilt": self.tilt, "roll": self.roll,
                "f": self.f}

    @classmethod
    def from_dict(cls, d: dict) -> CameraParams:  # type: ignore[type-arg]
        return cls(np.asarray(d["center"], float), float(d["pan"]), float(d["tilt"]), float(d.get("roll", 0.0)),
                   float(d["f"]))


def _project(G: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = pts @ G[:, :2].T + G[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = h[:, :2] / h[:, 2:3]
    return uv, h[:, 2]


# --------------------------------------------------------------------------- evidence

N_BINS = 8  # line orientation bins over [0, 180) degrees


@dataclass
class EvidenceConfig:
    """Line evidence settings, in pixels of a 1080-line frame (scaled with the frame height)."""

    kernel_px: int = 11  # top-hat / black-hat width: wider than a line (2 in, 3-8 px at broadcast zoom)
    line_min: float = 22.0  # response (Lab lightness levels) of a line pixel; tuned on ESPN frames
    density_px: int = 61  # neighbourhood of the floor test
    strong: float = 30.0  # a "strong" response for the density count
    max_density: float = 0.16  # floor: below this share of strong responses around (lines are sparse)
    min_floor_frac: float = 0.002  # a floor component smaller than this share of the image is dropped
    min_line_len_px: float = 28.0  # line pixel components shorter than this (wood grain, text bits) drop
    dist_clip_px: float = 60.0  # 30 px at the trackers' 960-px work width, where the thresholds were tuned
    exclude: list[tuple[int, int, int, int]] = field(default_factory=list)  # boxes (x, y, w, h) of a 1920x1080
    #   frame that never show the floor (the scorebug)


@dataclass
class Evidence:
    resp: np.ndarray  # float32 ridge response (masked to the floor)
    floor: np.ndarray  # bool floor mask
    lines: np.ndarray  # bool line pixels (long components)
    dist: np.ndarray  # float32 distance (px) to the nearest line pixel, clipped
    sigma: float = 2.0  # structure-tensor scale of ``orient``
    _orient: np.ndarray | None = None
    raw: np.ndarray | None = None  # float32 ridge response before the floor mask (for ``restrict``)
    density: np.ndarray | None = None  # float32 share of strong responses around each pixel
    blocked: np.ndarray | None = None  # bool: excluded boxes and people

    @property
    def size(self) -> tuple[int, int]:
        return (self.resp.shape[1], self.resp.shape[0])

    @property
    def orient(self) -> np.ndarray:
        """uint8 orientation bin of each pixel's line direction (N_BINS over 180 degrees); computed on first use
        (only the camera search needs it, not frame-to-frame tracking)."""
        if self._orient is None:
            self._orient = line_orientation(self.resp, self.sigma)
        return self._orient


def ridge_response(img_bgr: np.ndarray, kernel_px: int) -> np.ndarray:
    """Thin bright or dark structures: max of the top-hat and the black-hat of the Lab lightness."""
    L = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_px, kernel_px))
    th = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, se)
    bh = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, se)
    return np.maximum(th, bh).astype(np.float32)


def line_orientation(resp: np.ndarray, sigma: float) -> np.ndarray:
    """Orientation bin of the line through each pixel: the structure tensor of the response gives the normal."""
    gx = cv2.Sobel(resp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(resp, cv2.CV_32F, 0, 1, ksize=3)
    k = (0, 0)
    jxx = cv2.GaussianBlur(gx * gx, k, sigma)
    jyy = cv2.GaussianBlur(gy * gy, k, sigma)
    jxy = cv2.GaussianBlur(gx * gy, k, sigma)
    normal = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    direction = np.mod(normal + np.pi / 2.0, np.pi)
    return (np.floor(direction / np.pi * N_BINS + 0.5).astype(np.int32) % N_BINS).astype(np.uint8)


def court_evidence(img_bgr: np.ndarray, config: EvidenceConfig | None = None,
                   people: list[tuple[float, float, float, float]] | None = None, lines: bool = True) -> Evidence:
    """Line evidence of one frame (see the module docstring). ``people``: person boxes (x1, y1, x2, y2), masked
    out of the floor (a body's outline is no court line). ``lines=False`` leaves the frame-wide line pixels for
    ``with_lines`` (a tracked frame needs only ``restrict``)."""
    cfg = config or EvidenceConfig()
    h, w = img_bgr.shape[:2]
    s = h / 1080.0

    def odd(v: float) -> int:
        k = max(3, int(round(v)))
        return k if k % 2 else k + 1

    resp = ridge_response(img_bgr, odd(cfg.kernel_px * s))
    strong = (resp >= cfg.strong).astype(np.float32)
    k = odd(cfg.density_px * s)
    density = cv2.boxFilter(strong, -1, (k, k))
    floor = density < cfg.max_density
    sx = w / 1920.0
    blocked = np.zeros((h, w), bool)

    def clear(x0: float, y0: float, x1: float, y1: float) -> None:
        # clipped to the frame first: a negative end would wrap round to the far edge in a slice
        i0, i1 = int(np.clip(y0, 0, h)), int(np.clip(y1, 0, h))
        j0, j1 = int(np.clip(x0, 0, w)), int(np.clip(x1, 0, w))
        blocked[i0:i1, j0:j1] = True

    for x, y, bw, bh in cfg.exclude:  # boxes in pixels of a 1920x1080 frame
        x0, y0 = round(x * sx), round(y * s)
        clear(x0, y0, x0 + round(bw * sx), y0 + round(bh * s))
    for x1, y1, x2, y2 in people or ():
        clear(math.floor(x1), math.floor(y1), math.floor(x2) + 1, math.floor(y2) + 1)
    floor &= ~blocked
    n, lab, stats, _ = cv2.connectedComponentsWithStats(floor.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= cfg.min_floor_frac * h * w
    floor = keep[lab]
    ev = Evidence(resp, floor, np.zeros((0, 0), bool), np.zeros((0, 0), np.float32), 2.0 * s, raw=resp,
                  density=density, blocked=blocked)
    return with_lines(ev, cfg) if lines else ev


def with_lines(ev: Evidence, config: EvidenceConfig | None = None) -> Evidence:
    """``ev`` with its frame-wide line pixels, distance map and floor-masked response (computed once)."""
    if ev.lines.shape == ev.floor.shape or ev.raw is None:
        return ev
    cfg = config or EvidenceConfig()
    ev.lines, ev.dist = _line_pixels(ev.raw, ev.floor, cfg, ev.size[1] / 1080.0)
    ev.resp = np.where(ev.floor, ev.raw, 0.0).astype(np.float32)
    return ev


def _line_pixels(resp: np.ndarray, floor: np.ndarray, cfg: EvidenceConfig, s: float) -> tuple[np.ndarray, np.ndarray]:
    """Line pixels (strong responses on the floor in long components) and their clipped distance map."""
    lines = (resp >= cfg.line_min) & floor
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lines.astype(np.uint8), connectivity=8)
    span = np.hypot(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = span >= cfg.min_line_len_px * s
    keep[0] = False
    lines = keep[lab]
    dist = cv2.distanceTransform((~lines).astype(np.uint8), cv2.DIST_L2, 3)
    np.minimum(dist, cfg.dist_clip_px * s, out=dist)
    return lines, dist


def court_region(court: CourtSpec, params: CameraParams, size: tuple[int, int], margin_ft: float) -> np.ndarray:
    """bool image mask of the court rectangle grown by ``margin_ft`` under ``params`` (the part in front of the
    camera)."""
    w, h = size
    m = margin_ft
    xs = np.linspace(-m, court.length + m, 48)
    ys = np.linspace(-m, court.width + m, 32)
    pts = np.concatenate([np.column_stack([xs, np.full_like(xs, -m)]),
                          np.column_stack([np.full_like(ys, court.length + m), ys]),
                          np.column_stack([xs[::-1], np.full_like(xs, court.width + m)]),
                          np.column_stack([np.full_like(ys, -m), ys[::-1]])])
    uv, wz = _project(params.G(size), pts)
    mask = np.zeros((h, w), np.uint8)
    ok = wz > 1e-6
    if ok.sum() >= 3:
        q = np.clip(uv[ok], -8.0 * max(w, h), 8.0 * max(w, h))
        cv2.fillPoly(mask, [np.round(q * 8.0).astype(np.int32)], 1, cv2.LINE_8, shift=3)
    return mask.astype(bool)


def restrict(ev: Evidence, court: CourtSpec, params: CameraParams, config: EvidenceConfig | None = None,
             margin_ft: float = 4.0, max_density: float = 0.30) -> Evidence:
    """The evidence of a frame seen by a known (predicted) camera: the floor is the projected court grown by
    ``margin_ft`` (the stands, boards and graphics outside it drop out) minus clutter denser than ``max_density``,
    a looser test than the frame-wide floor mask (lines close together, as round the key, no longer mask
    themselves out; players still do)."""
    if ev.raw is None or ev.density is None or ev.blocked is None:
        return ev
    cfg = config or EvidenceConfig()
    s = ev.size[1] / 1080.0
    floor = (ev.density < max_density) & ~ev.blocked & court_region(court, params, ev.size, margin_ft)
    lines, dist = _line_pixels(ev.raw, floor, cfg, s)
    return Evidence(np.where(floor, ev.raw, 0.0).astype(np.float32), floor, lines, dist, ev.sigma, raw=ev.raw,
                    density=ev.density, blocked=ev.blocked)


def _oriented_dist(ev: Evidence, k: int, clip: float) -> tuple[np.ndarray, np.ndarray]:
    """Floor mask and per-orientation-bin line distance (full-resolution px) at 1/k resolution: (bins, h, w).
    A line pixel counts for its bin and the two neighbouring bins (orientation tolerance about 34 degrees)."""
    h, w = ev.lines.shape
    hh, ww = h // k, w // k
    floor = ev.floor[:hh * k, :ww * k].reshape(hh, k, ww, k).mean(axis=(1, 3)) > 0.5
    lines = ev.lines[:hh * k, :ww * k]
    orient = ev.orient[:hh * k, :ww * k]
    # the line pixels of each bin at 1/k; bin b takes those of b - 1, b and b + 1 (orientation distance <= 1)
    own = [(lines & (orient == o)).reshape(hh, k, ww, k).any(axis=(1, 3)) for o in range(N_BINS)]

    def one(b: int) -> np.ndarray:
        mask = own[(b - 1) % N_BINS] | own[b] | own[(b + 1) % N_BINS]
        dt = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3) * k
        return np.minimum(dt, clip)

    workers = _workers()
    bins = _pool(workers).map(one, range(N_BINS)) if workers > 1 else map(one, range(N_BINS))
    return floor, np.stack(list(bins)).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- search


@dataclass
class BroadcastPrior:
    """Candidate broadcast game cameras (court ft, degrees). Design choices: an NBA main camera sits at half court,
    50-120 ft behind the near sideline and 20-50 ft up (the Chase Center camera: about 86 ft back, 33 ft up), with
    a 12-60 degree horizontal field of view, aimed at a floor point near the court."""

    center_x: tuple[float, ...] = (47.0,)
    center_y: tuple[float, ...] = (-62.0, -95.0)
    center_z: tuple[float, ...] = (26.0, 40.0)
    aim_x: tuple[float, float] = (-6.0, 100.0)  # the optical axis meets the floor in this box
    aim_y: tuple[float, float] = (-4.0, 54.0)
    hfov_deg: tuple[float, float] = (12.0, 60.0)
    focal_step: float = 0.1  # relative focal-length step of the coarse grid
    levels: tuple[tuple[int, float, float], ...] = ((8, 80.0, 80.0), (4, 20.0, 24.0), (1, 5.0, 8.0))
    #   (downsample, pan / tilt step as an image shift px, soft match radius px) per pyramid level
    keep: tuple[int, ...] = (40, 12)  # candidates carried from level 0 -> 1 -> 2
    miss_weight: float = 0.5  # cost of a model point on the floor with no line near it
    step_ft: float = 2.5  # model sample spacing
    inverse_points: int = 300  # line pixels cast onto the floor for the inverse (image -> model) term
    inverse_weight: float = 1.0


def _rotations(pans: np.ndarray, tilts: np.ndarray) -> np.ndarray:
    pp, tt = np.radians(pans), np.radians(tilts)
    fwd = np.stack([np.cos(tt) * np.sin(pp), np.cos(tt) * np.cos(pp), -np.sin(tt)], axis=1)
    right = np.stack([fwd[:, 1], -fwd[:, 0], np.zeros_like(pp)], axis=1)
    right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd], axis=1).astype(np.float32)


@dataclass(frozen=True)
class CourtField:
    """Distance (ft) from each floor cell to the nearest court line, on a grid covering the court and a margin."""

    origin: tuple[float, float]
    res: float
    dist: np.ndarray

    @classmethod
    def of(cls, court: CourtSpec, res: float = 0.25, margin: float = 20.0) -> CourtField:
        x0, y0 = -margin, -margin
        w = int(math.ceil((court.length + 2 * margin) / res))
        h = int(math.ceil((court.width + 2 * margin) / res))
        mask = np.zeros((h, w), np.uint8)
        for line in court.line_segments():
            pts = np.asarray(line, float)
            for a, b in zip(pts[:-1], pts[1:], strict=True):
                n = max(1, int(math.hypot(*(b - a)) / (res / 2.0)))
                seg = a + (b - a) * (np.arange(n + 1)[:, None] / n)
                j = np.clip(((seg[:, 0] - x0) / res).astype(int), 0, w - 1)
                i = np.clip(((seg[:, 1] - y0) / res).astype(int), 0, h - 1)
                mask[i, j] = 1
        dist = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 3) * res
        return cls((x0, y0), res, dist.astype(np.float32))

    def lookup(self, x: np.ndarray, y: np.ndarray, far: float) -> np.ndarray:
        h, w = self.dist.shape
        j = ((x - self.origin[0]) / self.res).astype(np.int32)
        i = ((y - self.origin[1]) / self.res).astype(np.int32)
        inside = (j >= 0) & (j < w) & (i >= 0) & (i < h)
        return np.where(inside, self.dist[np.clip(i, 0, h - 1), np.clip(j, 0, w - 1)], far)


_BATCH = 256  # candidates projected together in ``_score_part``
_POOL: tuple[int, int, ThreadPoolExecutor] | None = None  # (process id, workers, executor)
_POOL_LOCK = threading.Lock()


def _workers() -> int:
    """Threads scoring camera candidates: ``COURTVISION_THREADS``, else one per core but one, at most 8."""
    try:
        n = int(os.environ.get("COURTVISION_THREADS") or min(8, (os.cpu_count() or 2) - 1))
    except ValueError:
        n = 1
    return max(1, n)


def _pool(n: int) -> ThreadPoolExecutor:
    """The process's scoring threads (made again in a forked child, whose copy has no threads)."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None or _POOL[0] != os.getpid() or _POOL[1] != n:
            _POOL = (os.getpid(), n, ThreadPoolExecutor(n, thread_name_prefix="court-search"))
        return _POOL[2]


def _scores(floor: np.ndarray, odist: np.ndarray, k: int, size: tuple[int, int], P: np.ndarray, D: np.ndarray,
            C: np.ndarray, pans: np.ndarray, tilts: np.ndarray, focals: np.ndarray, tau: float,
            lam: float, ev_pts: np.ndarray | None = None, field: CourtField | None = None,
            beta: float = 1.0) -> np.ndarray:
    """Bidirectional chamfer scores of level cameras at ``C`` (one per (pan, tilt, focal) triple): ``_score_part``
    over runs of whole batches on the scoring threads (``_workers``) when there are enough candidates."""
    n, workers = len(pans), _workers()
    if workers <= 1 or n <= 2 * _BATCH:
        return _score_part(floor, odist, k, size, P, D, C, pans, tilts, focals, tau, lam, ev_pts, field, beta)
    per = _BATCH * math.ceil(n / (_BATCH * workers))
    jobs = [_pool(workers).submit(_score_part, floor, odist, k, size, P, D, C, pans[a:a + per], tilts[a:a + per],
                                  focals[a:a + per], tau, lam, ev_pts, field, beta) for a in range(0, n, per)]
    return np.concatenate([j.result() for j in jobs])


def _score_part(floor: np.ndarray, odist: np.ndarray, k: int, size: tuple[int, int], P: np.ndarray, D: np.ndarray,
                C: np.ndarray, pans: np.ndarray, tilts: np.ndarray, focals: np.ndarray, tau: float,
                lam: float, ev_pts: np.ndarray | None = None, field: CourtField | None = None,
                beta: float = 1.0) -> np.ndarray:
    """Bidirectional chamfer scores of level cameras at ``C`` (one per (pan, tilt, focal) triple).

    Forward: each projected model point on the floor mask scores ``g - lam (1 - g)`` with ``g = 1 - d / tau`` and
    ``d`` its distance to a line pixel of the same orientation. Inverse (with ``ev_pts`` and ``field``): each line
    pixel, cast onto the floor through the candidate, scores the same way by its distance to the nearest court line
    (ft, turned into px by the range); the inverse sum is scaled to the forward point count, times ``beta``."""
    Rs = _rotations(pans, tilts)
    A = (P - C).astype(np.float32)
    B = (P + 0.75 * D - C).astype(np.float32)
    out = np.empty(len(Rs), np.float32)
    nb, hh, ww = odist.shape
    if ev_pts is not None and field is not None and len(ev_pts):
        e_uv = (ev_pts - np.array([size[0] / 2.0, size[1] / 2.0])).astype(np.float32)
        inv_scale = beta * len(P) / len(ev_pts)
    for a in range(0, len(Rs), _BATCH):
        R = Rs[a:a + _BATCH]
        Rt = R.transpose(0, 2, 1)
        f = focals[a:a + _BATCH, None].astype(np.float32)
        ca = A @ Rt  # (a, n, 3): each candidate's rotation applied to every model point (BLAS, not einsum's loop)
        cb = B @ Rt
        with np.errstate(divide="ignore", invalid="ignore"):
            ua, va = f * ca[..., 0] / ca[..., 2], f * ca[..., 1] / ca[..., 2]
            ub, vb = f * cb[..., 0] / cb[..., 2], f * cb[..., 1] / cb[..., 2]
        ang = np.mod(np.arctan2(vb - va, ub - ua), np.pi)
        bins = (np.floor(ang / np.pi * nb + 0.5).astype(np.int32)) % nb
        u = (ua + size[0] / 2.0) / k
        v = (va + size[1] / 2.0) / k
        ok = (ca[..., 2] > 1e-3) & (cb[..., 2] > 1e-3) & (u >= 0) & (u <= ww - 1) & (v >= 0) & (v <= hh - 1)
        ui = np.where(ok, u, 0).astype(np.int32)
        vi = np.where(ok, v, 0).astype(np.int32)
        on = ok & floor[vi, ui]
        g = np.clip(1.0 - odist[bins, vi, ui] / tau, 0.0, 1.0)
        sc = np.where(on, g - lam * (1.0 - g), 0.0).sum(axis=1)
        if ev_pts is not None and field is not None and len(ev_pts):
            # rays of the line pixels in world coordinates: R^T [u / f, v / f, 1] (row vectors: v @ R)
            rays = np.concatenate(
                [e_uv[None, :, :] / f[:, :, None], np.ones((len(R), len(e_uv), 1), np.float32)], axis=2) @ R
            with np.errstate(divide="ignore", invalid="ignore"):
                s = -C[2] / rays[..., 2]
            down = rays[..., 2] < -1e-6
            x = C[0] + s * rays[..., 0]
            y = C[1] + s * rays[..., 1]
            rng = s * np.linalg.norm(rays, axis=2)
            d_ft = field.lookup(np.where(down, x, 1e6), np.where(down, y, 1e6), 1e3)
            gi = np.clip(1.0 - d_ft * f / np.maximum(rng, 1.0) / tau, 0.0, 1.0)
            gi = np.where(down, gi, 0.0)
            sc += inv_scale * (gi - lam * (1.0 - gi)).sum(axis=1)
        out[a:a + _BATCH] = sc
    return out


def _line_points(ev: Evidence, k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Up to ``n`` line pixels (full-resolution px), one per occupied cell of a 1/k grid, picked at random."""
    ys, xs = np.nonzero(ev.lines)
    if len(xs) == 0:
        return np.zeros((0, 2))
    cells = (ys // k) * (ev.lines.shape[1] // k + 1) + xs // k
    _, first = np.unique(cells, return_index=True)
    pts = np.column_stack([xs[first], ys[first]]).astype(float) + 0.5
    if len(pts) > n:
        pts = pts[rng.choice(len(pts), n, replace=False)]
    return pts


_FIELDS: dict[CourtSpec, CourtField] = {}


def _court_field(court: CourtSpec) -> CourtField:
    if court not in _FIELDS:
        _FIELDS[court] = CourtField.of(court)
    return _FIELDS[court]


def _angles_to(C: np.ndarray, x: float, y: float) -> tuple[float, float]:
    return (math.degrees(math.atan2(x - C[0], y - C[1])),
            math.degrees(math.atan2(C[2], math.hypot(x - C[0], y - C[1]))))


def aim_range(C: np.ndarray, prior: BroadcastPrior) -> tuple[float, float, float, float]:
    """(pan_lo, pan_hi, tilt_lo, tilt_hi) of the optical axis from centre ``C`` through the prior's aim box. Pan
    is extreme at a corner (the box lies in front of the camera); tilt is lowest at the farthest corner and highest
    at the box point nearest the camera, which is on an edge when the camera is level with the box (half court)."""
    corners = [_angles_to(C, x, y) for x in prior.aim_x for y in prior.aim_y]
    nx = min(max(float(C[0]), min(prior.aim_x)), max(prior.aim_x))
    ny = min(max(float(C[1]), min(prior.aim_y)), max(prior.aim_y))
    near = 90.0 if math.hypot(nx - C[0], ny - C[1]) < 1e-9 else _angles_to(C, nx, ny)[1]
    return (min(c[0] for c in corners), max(c[0] for c in corners), min(c[1] for c in corners), near)


def focal_range(width: int, prior: BroadcastPrior) -> tuple[float, float]:
    """Focal lengths (px) of the prior's widest and narrowest horizontal field of view for an image ``width``."""
    return ((width / 2.0) / math.tan(math.radians(prior.hfov_deg[1]) / 2.0),
            (width / 2.0) / math.tan(math.radians(prior.hfov_deg[0]) / 2.0))


def plausible(params: CameraParams, size: tuple[int, int], prior: BroadcastPrior | None = None,
              max_roll_deg: float = 3.0, margin_deg: float = 3.0, focal_margin: float = 0.15) -> bool:
    """Whether a fitted camera is one a broadcast game camera can be: a focal length in the prior's field of view
    (``focal_margin`` relative slack), looking down (tilt > 0) within the prior's aim box (``margin_deg``), and
    nearly level (``|roll| <= max_roll_deg``). A fit on crowd texture or a graphic can land lines elsewhere
    (one such fit came back with tilt -21, roll 33 and f = 48 px)."""
    pr = prior or BroadcastPrior()
    f_lo, f_hi = focal_range(size[0], pr)
    if not (f_lo * (1.0 - focal_margin) <= params.f <= f_hi * (1.0 + focal_margin)):
        return False
    _, _, tilt_lo, tilt_hi = aim_range(np.asarray(params.center, float), pr)
    return (params.tilt > 0.0 and tilt_lo - margin_deg <= params.tilt <= tilt_hi + margin_deg
            and abs(params.roll) <= max_roll_deg)


def _nms(order: np.ndarray, pans: np.ndarray, tilts: np.ndarray, focals: np.ndarray, step0: float,
         focal_step: float, n: int) -> list[int]:
    """Up to ``n`` indices of ``order`` (best first), skipping a candidate within 1.5 grid steps in pan, tilt and
    log focal of one already kept: the next level searches about one step around each, so such neighbours would
    only repeat the same peak and crowd out the others."""
    kept: list[int] = []
    for i in order[: 50 * n]:
        f = float(focals[i])
        win = 1.5 * math.degrees(step0 / f)
        if all(abs(float(pans[i]) - float(pans[j])) > win or abs(float(tilts[i]) - float(tilts[j])) > win
               or abs(math.log(f / float(focals[j]))) > 1.5 * focal_step for j in kept):
            kept.append(int(i))
            if len(kept) >= n:
                break
    return kept


def search(ev: Evidence, court: CourtSpec, prior: BroadcastPrior | None = None, top: int = 6,
           centers: list[np.ndarray] | None = None, around: CameraParams | None = None,
           around_deg: float = 12.0) -> list[tuple[float, CameraParams]]:
    """Best cameras for one frame's evidence, best first: a coarse-to-fine (pan, tilt, focal) grid per candidate
    centre, scored by oriented chamfer matching at each pyramid level. ``around``: search only near this camera
    (re-acquisition after a cut: the same centre, pan within ``around_deg``)."""
    pr = prior or BroadcastPrior()
    size = ev.size
    pts, dirs = court_samples(court, pr.step_ft)
    P = np.column_stack([pts, np.zeros(len(pts))])
    D = np.column_stack([dirs, np.zeros(len(dirs))])
    if around is not None:
        centers = [np.asarray(around.center, float)]
    elif centers is None:
        centers = [np.array([cx, cy, cz]) for cx in pr.center_x for cy in pr.center_y for cz in pr.center_z]
    scale = size[1] / 1080.0  # levels are given for a 1080-line frame
    levels = [(max(1, int(round(k * scale))), step * scale, tau * scale) for k, step, tau in pr.levels]
    pyramid = [(k, *_oriented_dist(ev, k, 4.0 * tau), step, tau) for k, step, tau in levels]
    field = _court_field(court)
    rng = np.random.default_rng(0)
    ev_pts = [_line_points(ev, max(k, 2), pr.inverse_points, rng) for k, _, _ in levels]
    f_lo, f_hi = focal_range(size[0], pr)
    # level 0: the full grid
    k0, floor0, od0, step0, tau0 = pyramid[0]
    cand: list[tuple[float, np.ndarray, float, float, float]] = []
    for C in centers:
        if around is not None:
            pan_lo, pan_hi = around.pan - around_deg, around.pan + around_deg
            tilt_lo, tilt_hi = around.tilt - around_deg / 2.0, around.tilt + around_deg / 2.0
        else:
            pan_lo, pan_hi, tilt_lo, tilt_hi = aim_range(C, pr)
        pp, tt, ff = [], [], []
        for f in np.exp(np.arange(math.log(f_lo), math.log(f_hi) + 1e-9, pr.focal_step)):
            step = math.degrees(step0 / f)
            pans = np.arange(pan_lo, pan_hi + 1e-9, step)
            tilts = np.arange(tilt_lo, tilt_hi + 1e-9, step)
            gp, gt = np.meshgrid(pans, tilts, indexing="ij")
            pp.append(gp.ravel())
            tt.append(gt.ravel())
            ff.append(np.full(gp.size, f))
        pans_a, tilts_a, focals_a = np.concatenate(pp), np.concatenate(tt), np.concatenate(ff)
        sc = _scores(floor0, od0, k0, size, P, D, C, pans_a, tilts_a, focals_a, tau0, pr.miss_weight,
                     ev_pts[0], field, pr.inverse_weight)
        for i in _nms(np.argsort(-sc), pans_a, tilts_a, focals_a, step0, pr.focal_step, pr.keep[0]):
            cand.append((float(sc[i]), C, float(pans_a[i]), float(tilts_a[i]), float(focals_a[i])))
    cand.sort(key=lambda r: -r[0])
    cand = cand[: pr.keep[0]]
    # finer levels: a local grid around each carried candidate (the grids of one centre scored in one call)
    prev_step = step0
    for lvl, (k, floor, od, step, tau) in enumerate(pyramid[1:], start=1):
        grids: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for _, C, pan, tilt, f in cand:
            half = prev_step / f
            offs = np.degrees(np.arange(-half, half + 1e-9, step / f))
            gp, gt, gf = np.meshgrid(pan + offs, tilt + offs, f * np.array([0.94, 1.0, 1.06]), indexing="ij")
            grids.append((C, gp.ravel(), gt.ravel(), gf.ravel()))
        by_centre: dict[int, list[int]] = {}
        for j, g in enumerate(grids):
            by_centre.setdefault(id(g[0]), []).append(j)
        scored: list[np.ndarray] = [np.empty(0, np.float32)] * len(grids)
        for js in by_centre.values():
            pans_a, tilts_a, focals_a = (np.concatenate([grids[j][a] for j in js]) for a in (1, 2, 3))
            sc = _scores(floor, od, k, size, P, D, grids[js[0]][0], pans_a, tilts_a, focals_a, tau, pr.miss_weight,
                         ev_pts[lvl], field, pr.inverse_weight)
            start = 0
            for j in js:
                scored[j] = sc[start:start + len(grids[j][1])]
                start += len(grids[j][1])
        nxt: list[tuple[float, np.ndarray, float, float, float]] = []
        for (C, gp, gt, gf), sc in zip(grids, scored, strict=True):
            i = int(np.argmax(sc))
            nxt.append((float(sc[i]), C, float(gp[i]), float(gt[i]), float(gf[i])))
        nxt.sort(key=lambda r: -r[0])
        cand = nxt[: pr.keep[lvl] if lvl < len(pr.keep) else top * 2]
        prev_step = step
    out: list[tuple[float, CameraParams]] = []
    for sc, C, pan, tilt, f in cand:
        p = CameraParams(np.array(C, float), pan, tilt, 0.0, f)
        if all(abs(p.pan - q.pan) > 1.0 or abs(p.tilt - q.tilt) > 1.0 or abs(math.log(p.f / q.f)) > 0.1
               or np.linalg.norm(p.center - q.center) > 1.0 for _, q in out):
            out.append((sc, p))
        if len(out) >= top:
            break
    return out


_DENSE: dict[CourtSpec, list[np.ndarray]] = {}


def _dense_lines(court: CourtSpec) -> list[np.ndarray]:
    """Each court line as points at most 0.5 ft apart (its corners kept), for drawing it through a camera."""
    if court not in _DENSE:
        out = []
        for line in court.line_segments():
            pts = np.asarray(line, float)
            dense = [pts[0]]
            for a, b in zip(pts[:-1], pts[1:], strict=True):
                n = max(1, int(math.hypot(*(b - a)) / 0.5))
                dense.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
            out.append(np.asarray(dense))
        _DENSE[court] = out
    return _DENSE[court]


def explained(ev: Evidence, court: CourtSpec, params: CameraParams, tol_px: float = 4.0) -> float:
    """Share of the frame's line pixels within ``tol_px`` (1080-line px) of a projected court line (the image ->
    model half of a bidirectional match: a wrong camera that puts the court on some lines leaves the others
    unexplained)."""
    size = ev.size
    tol_px *= size[1] / 1080.0
    mask = np.zeros(ev.lines.shape, np.uint8)
    G = params.G(size)
    for dense in _dense_lines(court):
        uv, wz = _project(G, dense)
        ok = (wz > 0) & np.all(np.abs(uv) < 1e5, axis=1)
        # the segments between consecutive points both in front: one polyline per run of such points (a thin
        # polyline draws each segment as cv2.line does)
        q = np.round(np.where(ok[:, None], uv, 0.0)).astype(np.int32)
        cuts = np.flatnonzero(np.diff(ok.astype(np.int8)) != 0) + 1
        runs = [r for r in np.split(np.arange(len(ok)), cuts) if len(r) >= 2 and ok[r[0]]]
        if runs:
            cv2.polylines(mask, [q[r] for r in runs], False, 255, 1)
    d = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    n = int(ev.lines.sum())
    return float((d[ev.lines] <= tol_px).sum() / n) if n else 0.0


# --------------------------------------------------------------------------- refinement


@dataclass
class RefineConfig:
    """Line-search refinement; pixel values are for a 1080-line frame (scaled with the frame height)."""

    search_px: tuple[int, ...] = (30, 16, 8, 4, 3)
    sample_step_ft: float = 1.0
    min_matches: int = 30
    huber_px: float = 3.0
    cover_px: float = 2.5  # a model point within this of a line pixel is covered
    max_nfev: int = 60
    fix_roll: bool = False
    chamfer_pass: bool = True  # a wide-basin alignment on the line distance map before the line matches


@dataclass
class RefineResult:
    params: CameraParams
    matches: int
    inliers: int
    rmse_px: float
    coverage: float  # share of the model points on the floor that are on a line (tight match)
    quality: float = 0.0  # coverage x explained (set by calibrate_frame)


def _match(ev: Evidence, G: np.ndarray, pts: np.ndarray, dirs: np.ndarray, radius: int,
           min_resp: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Along each projected line's normal, the strongest response within ``radius``: (court pts, image pts,
    weights)."""
    h_img, w_img = ev.resp.shape
    uv, wz = _project(G, pts)
    uv2, wz2 = _project(G, pts + 0.5 * dirs)
    ok = (wz > 0) & (wz2 > 0) & np.all(np.isfinite(uv), axis=1) & np.all(np.isfinite(uv2), axis=1)
    m = radius + 2
    ok &= (uv[:, 0] >= m) & (uv[:, 0] < w_img - m) & (uv[:, 1] >= m) & (uv[:, 1] < h_img - m)
    if not ok.any():
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    ui, vi = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
    ok_idx = np.flatnonzero(ok)[ev.floor[vi, ui]]
    if len(ok_idx) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    q, q2, P = uv[ok_idx], uv2[ok_idx], pts[ok_idx]
    d = q2 - q
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    offs = np.arange(-radius, radius + 1, dtype=float)
    xs = np.clip(np.rint(q[:, None, 0] + offs[None, :] * n[:, None, 0]).astype(int), 0, w_img - 1)
    ys = np.clip(np.rint(q[:, None, 1] + offs[None, :] * n[:, None, 1]).astype(int), 0, h_img - 1)
    vals = ev.resp[ys, xs]
    k = np.argmax(vals, axis=1)
    best = vals[np.arange(len(k)), k]
    keep = best >= min_resp
    km, kp = np.clip(k - 1, 0, len(offs) - 1), np.clip(k + 1, 0, len(offs) - 1)
    a, b, c = vals[np.arange(len(k)), km], best, vals[np.arange(len(k)), kp]
    den = a - 2.0 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(np.abs(den) > 1e-6, 0.5 * (a - c) / den, 0.0)
    s = offs[k] + np.clip(frac, -0.5, 0.5)
    img = q + s[:, None] * n
    return P[keep], img[keep], best[keep]


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = img.shape
    u = np.clip(u, 0.0, w - 1.001)
    v = np.clip(v, 0.0, h - 1.001)
    x0, y0 = u.astype(np.int32), v.astype(np.int32)
    fx, fy = u - x0, v - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def _chamfer_pass(ev: Evidence, pts: np.ndarray, x: np.ndarray, free: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Coarse alignment: minimise the (clipped) distance of every projected court point on the floor to the nearest
    line pixel in any direction. Unlike the matches along each line's normal it sees a tilt error on lines that run
    across the image and a pan error on lines that run up it alike, so it pulls in from farther away."""
    from scipy.optimize import least_squares

    h_img, w_img = ev.dist.shape
    clip = float(ev.dist.max())
    uv0, wz = _project(CameraParams.from_vector(x).G(size), pts)
    ok = (wz > 0) & (uv0[:, 0] >= 0) & (uv0[:, 0] < w_img) & (uv0[:, 1] >= 0) & (uv0[:, 1] < h_img)
    ok[ok] = ev.floor[uv0[ok, 1].astype(int), uv0[ok, 0].astype(int)]
    P = pts[ok]
    if len(P) < 20:
        return x

    def res(p: np.ndarray) -> np.ndarray:
        v = x.copy()
        v[free] = p
        uv, w = _project(CameraParams.from_vector(v).G(size), P)
        d = _bilinear(ev.dist, uv[:, 0], uv[:, 1])
        return np.where((w > 0) & np.isfinite(d), d, clip)

    scale = np.array([5.0, 5.0, 5.0, 0.5, 0.5, 0.5, 0.02])[free]
    h = scale * 0.02  # absolute steps: scipy's diff_step is relative to |x|, so a centre at y = -62 ft would step
    #   6 ft and a pan or roll near 0 by next to nothing

    def jac(p: np.ndarray) -> np.ndarray:  # central differences over the bilinear distance map
        J = np.empty((len(P), len(p)))
        for i, hi in enumerate(h):
            e = np.zeros_like(p)
            e[i] = hi
            J[:, i] = (res(p + e) - res(p - e)) / (2.0 * hi)
        return J

    sol = least_squares(res, x[free], jac=jac, loss="soft_l1", f_scale=0.25 * clip, max_nfev=40, x_scale=scale)
    out = x.copy()
    out[free] = sol.x
    return out


def refine(ev: Evidence, court: CourtSpec, start: CameraParams, free_center: bool = False,
           config: RefineConfig | None = None, min_resp: float | None = None) -> RefineResult | None:
    """Line-search refinement of ``start`` (pan / tilt / roll / focal, and the centre when ``free_center``)."""
    from scipy.optimize import least_squares

    cfg = config or RefineConfig()
    size = ev.size
    pts, dirs = court_samples(court, cfg.sample_step_ft)
    x = start.vector()
    free = np.array([free_center] * 3 + [True, True, not cfg.fix_roll, True])
    thr = EvidenceConfig().line_min if min_resp is None else min_resp
    sc = size[1] / 1080.0
    if cfg.chamfer_pass:
        x = _chamfer_pass(ev, pts, x, free, size)
    last = None
    for k, radius in enumerate(cfg.search_px):
        G = CameraParams.from_vector(x).G(size)
        cp, ip, _ = _match(ev, G, pts, dirs, max(2, int(round(radius * sc))), thr)
        if len(cp) < cfg.min_matches:
            return None
        # the robust scale shrinks with the search radius: with a tight scale on a wide first pass the few lines
        # that see a pan error (across their direction) would be outliers next to the many that slide along it
        huber = max(cfg.huber_px, 0.6 * radius) * sc

        def res(p: np.ndarray, cp: np.ndarray = cp, ip: np.ndarray = ip) -> np.ndarray:
            v = x.copy()
            v[free] = p
            uv, _ = _project(CameraParams.from_vector(v).G(size), cp)
            return (uv - ip).ravel()

        scale = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 0.05])[free]
        sol = least_squares(res, x[free], loss="huber", f_scale=huber, max_nfev=cfg.max_nfev, x_scale=scale)
        x = x.copy()
        x[free] = sol.x
        last = (cp, ip)
    assert last is not None
    params = CameraParams.from_vector(x)
    G = params.G(size)
    cp, ip = last
    uv, _ = _project(G, cp)
    err = np.linalg.norm(uv - ip, axis=1)
    inl = err <= cfg.huber_px * sc
    rmse = float(np.sqrt(np.mean(err[inl] ** 2))) / sc if inl.any() else math.inf  # 1080-line px
    return RefineResult(params, len(err), int(inl.sum()), rmse, coverage(ev, court, params, cfg))


def coverage(ev: Evidence, court: CourtSpec, params: CameraParams, config: RefineConfig | None = None) -> float:
    """Share of the projected court samples on the floor mask that lie on (next to) a line pixel."""
    cfg = config or RefineConfig()
    pts, _ = court_samples(court, cfg.sample_step_ft)
    uvp, wz = _project(params.G(ev.size), pts)
    wimg, himg = ev.size
    ok = (wz > 0) & (uvp[:, 0] >= 0) & (uvp[:, 0] < wimg) & (uvp[:, 1] >= 0) & (uvp[:, 1] < himg)
    ui, vi = uvp[ok, 0].astype(int), uvp[ok, 1].astype(int)
    on = ev.floor[vi, ui]
    return float((ev.dist[vi, ui][on] <= cfg.cover_px * himg / 1080.0).mean()) if on.sum() >= 20 else 0.0


def calibrate_frame(ev: Evidence, court: CourtSpec, prior: BroadcastPrior | None = None,
                    centers: list[np.ndarray] | None = None, top: int = 5, free_center: bool = True,
                    around: CameraParams | None = None, around_deg: float = 12.0) -> RefineResult | None:
    """Search + refinement of the best candidates; the result whose court lines land on the most lines and explain
    the most line pixels (``coverage x explained``). The free-centre refinement is kept only when it scores
    better than the fixed-centre one (a centre that slides along a line can raise coverage and explain less)."""
    best: tuple[float, RefineResult] | None = None
    for _, cand in search(ev, court, prior, top=top, centers=centers, around=around, around_deg=around_deg):
        r = refine(ev, court, cand)
        if r is None:
            continue
        q = r.coverage * explained(ev, court, r.params)
        if free_center and around is None and r.coverage > 0.3:
            rc = refine(ev, court, r.params, free_center=True)
            if rc is not None:
                qc = rc.coverage * explained(ev, court, rc.params)
                if qc > q:
                    r, q = rc, qc
        r.quality = q
        if best is None or q > best[0]:
            best = (q, r)
    return None if best is None else best[1]


def fit_center(evidences: list[Evidence], court: CourtSpec, starts: list[CameraParams],
               config: RefineConfig | None = None) -> tuple[np.ndarray, list[RefineResult | None]]:
    """One camera centre for several frames (each with its own pan / tilt / roll / focal): alternate per-frame
    refinement at the shared centre with a centre step over all frames' line matches."""
    from scipy.optimize import least_squares

    cfg = config or RefineConfig()
    center = np.median(np.array([s.center for s in starts]), axis=0)
    params = [CameraParams(center.copy(), s.pan, s.tilt, s.roll, s.f) for s in starts]
    pts, dirs = court_samples(court, cfg.sample_step_ft)
    results: list[RefineResult | None] = [None] * len(starts)
    for _ in range(4):
        results = [refine(ev, court, CameraParams(center.copy(), p.pan, p.tilt, p.roll, p.f), config=cfg)
                   for ev, p in zip(evidences, params, strict=True)]
        good = [(ev, r) for ev, r in zip(evidences, results, strict=True) if r is not None and r.coverage > 0.3]
        if len(good) < 2:
            break
        params = [r.params if r is not None else p for r, p in zip(results, params, strict=True)]
        corr = []
        for ev, r in good:
            cp, ip, _ = _match(ev, r.params.G(ev.size), pts, dirs, max(2, int(round(3 * ev.size[1] / 1080.0))),
                               EvidenceConfig().line_min)
            corr.append((ev.size, r.params, cp, ip))

        def res(v: np.ndarray) -> np.ndarray:
            c = v[:3]
            out = []
            for k, (size, p, cp, ip) in enumerate(corr):
                q = v[3 + 4 * k: 7 + 4 * k]
                cam = CameraParams(c, q[0], q[1], q[2], math.exp(q[3]))
                uv, _ = _project(cam.G(size), cp)
                out.append((uv - ip).ravel())
            return np.concatenate(out)

        v0 = np.concatenate([center] + [np.array([p.pan, p.tilt, p.roll, math.log(p.f)]) for _, p, _, _ in corr])
        sol = least_squares(res, v0, loss="huber", f_scale=2.0, max_nfev=200)
        step = float(np.linalg.norm(sol.x[:3] - center))
        center = sol.x[:3]
        log.debug("centre %s (step %.2f ft)", np.round(center, 2), step)
        if step < 0.05:
            break
    results = [refine(ev, court, CameraParams(center.copy(), p.pan, p.tilt, p.roll, p.f), config=cfg)
               for ev, p in zip(evidences, params, strict=True)]
    return center, results
