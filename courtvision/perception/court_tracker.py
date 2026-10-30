"""Per-frame court homography for a moving (pan / tilt / zoom) broadcast camera.

The venue pipeline assumes one operator calibration for the whole game. A broadcast game camera pans and
zooms to follow the play, so the image -> court homography changes every frame. ``CourtTracker`` tracks it
from the painted court lines, starting from one calibration (the operator's 4+ landmark clicks, or the
previous frame):

1. **Line response.** Court lines are thin, bright and unsaturated: a white top-hat of the grey image
   (structures narrower than ``tophat_px``), kept where the saturation is low (``line_response``).
2. **Model-based search** (ICP-style). Points sampled every ``sample_step_ft`` along every court line
   (``CourtSpec.line_segments``) are projected with the current estimate; along each projected line's
   normal the strongest response within ``search_px`` is the matched image point (sub-pixel by a parabola
   through the peak). ``cv2.findHomography`` (RANSAC, ``ransac_px``) refits; the search repeats with
   shrinking radii, which also removes the along-line (aperture) bias of point-to-point matches.
3. **Temporal gating.** A fit is accepted only with enough inlier matches (``min_matches``,
   ``min_inlier_frac``), a small reprojection error (``max_rmse_px``) and a plausible change since the last
   accepted frame: the camera centre recovered from the homography (``footpoint.camera_from_homography``)
   stays within ``max_center_ft`` of its running estimate (a broadcast game camera does not move; it
   pans, tilts and zooms), the pan within ``max_pan_deg`` per frame and the zoom within ``max_zoom``. A
   rejected frame keeps the last homography ("held"); ``lost_after`` rejections in a row mean the court is
   not in view (a cut to a replay, a close-up) and the state is "lost".
4. **Re-acquisition.** While lost, every frame tries the pan / tilt / zoom family of the known camera
   centre: a grid over pan (``reacquire_pan_deg``), tilt and focal length around the last good camera is
   scored by the mean line response at the projected court samples; the best candidate is refined by the
   search above and accepted through the same gates (the motion gate is relaxed).

All thresholds are design choices tuned on simulator broadcast renders (``sim/broadcast.py``); real footage
needs a check on gold frames.
"""

from __future__ import annotations

import functools
import logging
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from courtvision.geometry.camera import PinholeCamera
from courtvision.geometry.court import CourtSpec
from courtvision.geometry.homography import Homography

from .footpoint import camera_from_homography

__all__ = ["CourtTracker", "CourtTrackerConfig", "TrackResult", "court_samples", "line_response"]

log = logging.getLogger(__name__)


@dataclass
class CourtTrackerConfig:
    sample_step_ft: float = 1.0
    search_px: tuple[int, ...] = (24, 10, 5, 3)  # radius per refinement pass
    tophat_px: int = 9  # lines narrower than this (px) respond
    sat_max: int = 70  # HSV saturation above this is not a painted line
    min_response: float = 25.0  # top-hat grey levels at a matched point
    edge_margin_px: int = 3
    min_matches: int = 40
    min_inlier_frac: float = 0.6
    ransac_px: float = 3.0
    max_rmse_px: float = 2.0
    max_center_ft: float = 4.0  # homography mode: the recovered camera centre vs the initial one
    max_pan_deg: float = 4.0  # per frame
    max_zoom: float = 0.08  # relative focal-length change per frame
    lost_after: int = 3
    # coarse grid around the last good camera, then a fine grid around the best coarse candidates
    reacquire_pan_deg: tuple[float, float, float] = (-50.0, 50.0, 2.0)  # (from, to, step) around the last pan
    reacquire_zoom: tuple[float, ...] = (0.8, 1.0, 1.25)
    reacquire_fine_pan_deg: tuple[float, ...] = (-1.5, -0.75, 0.0, 0.75, 1.5)
    reacquire_fine_tilt_deg: tuple[float, ...] = (-3.0, -1.5, 0.0, 1.5, 3.0)
    reacquire_fine_zoom: tuple[float, ...] = (0.93, 1.0, 1.07)
    reacquire_top: int = 3
    reacquire_every: int = 2  # while lost, try every this many frames
    reacquire_min_score: float = 0.35  # share of the projected samples on a line response


@dataclass(frozen=True)
class TrackResult:
    homography: Homography | None  # image -> court (None while lost)
    status: str  # "tracking" | "held" | "lost" | "reacquired"
    matches: int = 0
    inliers: int = 0
    rmse_px: float | None = None


def line_response(img_bgr: np.ndarray, tophat_px: int = 9, sat_max: int = 70) -> np.ndarray:
    """Float32 response of thin bright unsaturated structures (painted lines)."""
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_px, tophat_px))
    th = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, k).astype(np.float32)
    sat = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    th[sat > sat_max] = 0.0
    return th


def court_samples(court: CourtSpec, step_ft: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Points every ``step_ft`` along all court lines and their unit directions (court ft), (N, 2) each. Cached per
    (court, step): callers must not modify the arrays."""
    return _court_samples(court, float(step_ft))


@functools.lru_cache(maxsize=32)
def _court_samples(court: CourtSpec, step_ft: float) -> tuple[np.ndarray, np.ndarray]:
    pts, dirs = [], []
    for line in court.line_segments():
        for (x0, y0), (x1, y1) in zip(line[:-1], line[1:], strict=True):
            d = math.hypot(x1 - x0, y1 - y0)
            if d < 1e-9:
                continue
            n = max(1, int(round(d / step_ft)))
            u = ((x1 - x0) / d, (y1 - y0) / d)
            for k in range(n):
                s = (k + 0.5) / n
                pts.append((x0 + s * (x1 - x0), y0 + s * (y1 - y0)))
                dirs.append(u)
    p, d = np.asarray(pts, dtype=float), np.asarray(dirs, dtype=float)
    p.setflags(write=False)
    d.setflags(write=False)
    return p, d


def _project(G: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = pts @ G[:, :2].T + G[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = h[:, :2] / h[:, 2:3]
    return uv, h[:, 2]


def _ptz(cam: PinholeCamera) -> tuple[float, float, float]:
    """(pan deg, tilt deg, focal px) of a camera without roll."""
    fwd = cam.R[2]
    return (math.degrees(math.atan2(fwd[0], fwd[1])), math.degrees(math.asin(max(-1.0, min(1.0, -fwd[2])))),
            float(cam.K[0, 0]))


def _rotation(pan: float, tilt: float) -> np.ndarray:
    """World -> camera rotation (rows right, down, forward; no roll) for pan / tilt in degrees."""
    p, t = math.radians(pan), math.radians(tilt)
    fwd = np.array([math.cos(t) * math.sin(p), math.cos(t) * math.cos(p), -math.sin(t)])
    right = np.array([fwd[1], -fwd[0], 0.0]) / max(math.hypot(fwd[0], fwd[1]), 1e-12)
    return np.vstack([right, np.cross(fwd, right), fwd])


def _ptz_G(center: np.ndarray, ptz: tuple[float, float, float], size: tuple[int, int]) -> np.ndarray:
    """Court -> image homography of the camera at ``center`` with (pan deg, tilt deg, focal px) ``ptz``."""
    R = _rotation(ptz[0], ptz[1])
    t = -R @ center
    K = np.array([[ptz[2], 0.0, size[0] / 2.0], [0.0, ptz[2], size[1] / 2.0], [0.0, 0.0, 1.0]])
    return K @ np.column_stack([R[:, 0], R[:, 1], t])


def _camera(center: np.ndarray, pan: float, tilt: float, f: float, size: tuple[int, int]) -> PinholeCamera:
    R = _rotation(pan, tilt)
    K = np.array([[f, 0.0, size[0] / 2.0], [0.0, f, size[1] / 2.0], [0.0, 0.0, 1.0]])
    return PinholeCamera(K=K, R=R, t=-R @ center, image_size=size)


def _G(cam: PinholeCamera) -> np.ndarray:
    return cam.K @ np.column_stack([cam.R[:, 0], cam.R[:, 1], cam.t])


_Fit = tuple[np.ndarray, int, int, float, "tuple[float, float, float] | None"]  # G, matches, inliers, rmse, ptz


class CourtTracker:
    """``update(frame_bgr) -> TrackResult`` (see the module docstring)."""

    def __init__(self, court: CourtSpec, image_size: tuple[int, int], initial: Homography | None = None,
                 config: CourtTrackerConfig | None = None) -> None:
        self.court = court
        self.config = config or CourtTrackerConfig()
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self._pts, self._dirs = court_samples(court, self.config.sample_step_ft)
        self._G: np.ndarray | None = None  # court -> image of the last accepted frame
        self._center: np.ndarray | None = None  # the fixed camera centre (pan / tilt / zoom mode)
        self._ptz: tuple[float, float, float] | None = None  # last accepted (pan, tilt, focal px)
        self._vel = np.zeros(3)  # per-frame (pan, tilt, log focal) velocity
        self.misses = 0
        self.status = "lost"
        self._lost_frames = 0
        if initial is not None:
            self.reset(initial)

    # ------------------------------------------------------------------ state

    def reset(self, homography: Homography) -> None:
        """Start (again) from a known homography, e.g. an operator calibration of the game camera: its camera
        centre is taken as fixed from then on (pan / tilt / zoom tracking); without a camera behind the
        homography the tracker fits a general homography per frame."""
        self._G = np.linalg.inv(homography.H)
        self._G /= self._G[2, 2] if abs(self._G[2, 2]) > 1e-12 else 1.0
        cam = self._camera_of(homography)
        if cam is not None:
            self._center, self._ptz = cam.center.copy(), _ptz(cam)
        self._vel[:] = 0.0
        self.misses, self.status, self._lost_frames = 0, "tracking", 0

    @property
    def center(self) -> np.ndarray | None:
        """The fixed camera centre (court ft), when known."""
        return None if self._center is None else self._center.copy()

    @property
    def ptz(self) -> tuple[float, float, float] | None:
        return self._ptz

    @property
    def homography(self) -> Homography | None:
        return None if self._G is None or self.status == "lost" else Homography(np.linalg.inv(self._G))

    # ------------------------------------------------------------------ per frame

    def _predicted(self) -> tuple[float, float, float] | None:
        if self._ptz is None:
            return None
        k = self.misses + 1
        pan, tilt, f = self._ptz
        return pan + k * self._vel[0], tilt + k * self._vel[1], f * math.exp(k * self._vel[2])

    def update(self, frame_bgr: np.ndarray) -> TrackResult:
        cfg = self.config
        resp = line_response(frame_bgr, cfg.tophat_px, cfg.sat_max)
        if self._G is None:
            return TrackResult(None, "lost")
        if self.status != "lost":
            pred = self._predicted() if self._center is not None else None
            fit = self._fit(resp, self._G if pred is None else None, pred, self.misses + 1)
            if fit is not None and self._plausible(fit, relaxed=False):
                return self._accept(fit, "tracking")
            self.misses += 1
            if self.misses < cfg.lost_after:
                self.status = "held"
                return TrackResult(self.homography, "held", *(fit[1:4] if fit is not None else (0, 0, None)))
            self.status = "lost"
            log.debug("court tracker lost after %d rejected frames", self.misses)
        self._lost_frames += 1
        start = self._reacquire(resp) if self._lost_frames % cfg.reacquire_every == 1 else None
        if start is not None:
            fit = self._fit(resp, None, start, 1)
            if fit is not None and self._plausible(fit, relaxed=True):
                self._vel[:] = 0.0
                return self._accept(fit, "reacquired")
        return TrackResult(None, "lost")

    def _accept(self, fit: _Fit, status: str) -> TrackResult:
        G, matches, inliers, rmse, ptz = fit
        self._G = G / G[2, 2]
        if ptz is not None and self._ptz is not None and status == "tracking":
            k = self.misses + 1
            v = np.array([(ptz[0] - self._ptz[0]) / k, (ptz[1] - self._ptz[1]) / k, math.log(ptz[2] / self._ptz[2]) / k])
            self._vel = 0.5 * self._vel + 0.5 * v
        if ptz is not None:
            self._ptz = ptz
        self.misses, self.status, self._lost_frames = 0, "tracking", 0
        return TrackResult(Homography(np.linalg.inv(self._G)), status, matches, inliers, rmse)

    def _fit(self, resp: np.ndarray, G: np.ndarray | None, ptz: tuple[float, float, float] | None,
             frames: int) -> _Fit | None:
        """Search + fit passes from ``ptz`` (pan / tilt / zoom of the fixed centre) or, without a centre, from
        the homography ``G``. ``frames``: frames since the start estimate (widens the first search)."""
        from scipy.optimize import least_squares

        cfg = self.config
        last = len(cfg.search_px) - 1
        x = None if ptz is None else np.array([ptz[0], ptz[1], math.log(ptz[2])])
        out: _Fit | None = None
        for k, radius in enumerate(cfg.search_px):
            r = radius * min(frames, 3) if k == 0 else radius
            Gk = G if x is None else _ptz_G(self._center, (x[0], x[1], math.exp(x[2])), self.image_size)
            m = self._match(resp, Gk, int(r))
            if m is None:
                return None
            court_pts, img_pts = m
            # wide early passes also catch other thin bright things (jersey numbers): robust losses drop them,
            # and the inlier share is judged on the last, tight pass only
            thresh = cfg.ransac_px if k == last else 2.0 * cfg.ransac_px
            if x is None:
                Gn, mask = cv2.findHomography(court_pts, img_pts, cv2.RANSAC, thresh)
                if Gn is None or mask is None:
                    return None
                G = Gn
                uv, _ = _project(Gn, court_pts)
            else:
                cp = court_pts.astype(float)
                ip = img_pts.astype(float)

                def res(p: np.ndarray) -> np.ndarray:
                    uv_, _ = _project(_ptz_G(self._center, (p[0], p[1], math.exp(p[2])), self.image_size), cp)
                    return (uv_ - ip).ravel()

                sol = least_squares(res, x, loss="huber", f_scale=thresh, max_nfev=30, x_scale=[1.0, 1.0, 0.05])
                x = sol.x
                Gn = _ptz_G(self._center, (x[0], x[1], math.exp(x[2])), self.image_size)
                uv, _ = _project(Gn, cp)
            err = np.linalg.norm(uv - img_pts, axis=1)
            inl_mask = err <= thresh
            inl = int(inl_mask.sum())
            if inl < cfg.min_matches or (k == last and inl < cfg.min_inlier_frac * len(err)):
                return None
            rmse = float(np.sqrt(np.mean(err[inl_mask] ** 2)))
            ptz_k = None if x is None else (float(x[0]), float(x[1]), float(math.exp(x[2])))
            out = (Gn, len(err), inl, rmse, ptz_k)
        if out is None or out[3] > cfg.max_rmse_px:
            return None
        return out

    def _match(self, resp: np.ndarray, G: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray] | None:
        cfg = self.config
        h_img, w_img = resp.shape
        uv, w = _project(G, self._pts)
        uv2, w2 = _project(G, self._pts + 0.5 * self._dirs)
        ok = (w > 0) & (w2 > 0) & np.all(np.isfinite(uv), axis=1) & np.all(np.isfinite(uv2), axis=1)
        m = cfg.edge_margin_px + radius
        ok &= (uv[:, 0] >= m) & (uv[:, 0] < w_img - m) & (uv[:, 1] >= m) & (uv[:, 1] < h_img - m)
        if ok.sum() < cfg.min_matches:
            return None
        P, q, q2 = self._pts[ok], uv[ok], uv2[ok]
        d = q2 - q
        n = np.stack([-d[:, 1], d[:, 0]], axis=1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        offs = np.arange(-radius, radius + 1, dtype=float)
        xs = np.rint(q[:, None, 0] + offs[None, :] * n[:, None, 0]).astype(int)
        ys = np.rint(q[:, None, 1] + offs[None, :] * n[:, None, 1]).astype(int)
        np.clip(xs, 0, w_img - 1, out=xs)
        np.clip(ys, 0, h_img - 1, out=ys)
        vals = resp[ys, xs]
        k = np.argmax(vals, axis=1)
        best = vals[np.arange(len(k)), k]
        keep = best >= cfg.min_response
        if keep.sum() < cfg.min_matches:
            return None
        # sub-pixel peak: parabola through the neighbours
        km, kp = np.clip(k - 1, 0, len(offs) - 1), np.clip(k + 1, 0, len(offs) - 1)
        a, b, c = vals[np.arange(len(k)), km], best, vals[np.arange(len(k)), kp]
        den = a - 2.0 * b + c
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(np.abs(den) > 1e-6, 0.5 * (a - c) / den, 0.0)
        s = offs[k] + np.clip(frac, -0.5, 0.5)
        img = q + s[:, None] * n
        return P[keep].astype(np.float32), img[keep].astype(np.float32)

    def _plausible(self, fit: _Fit, relaxed: bool) -> bool:
        """Pan / tilt / zoom mode: the change since the last accepted frame is one a camera operator can make
        (``max_pan_deg`` per frame, a fifth of it in tilt, ``max_zoom``); relaxed after a cut. Homography mode:
        the recovered camera centre agrees with the last one (``max_center_ft``)."""
        cfg = self.config
        ptz = fit[4]
        if ptz is None:  # general homography: the camera behind it must not jump
            cam = self._camera_of(Homography(np.linalg.inv(fit[0] / fit[0][2, 2])))
            if cam is None:
                return False
            ok = self._center is None or float(np.linalg.norm(cam.center - self._center)) <= cfg.max_center_ft
            return relaxed or ok
        if relaxed or self._ptz is None:
            return True
        k = self.misses + 1
        dpan = (ptz[0] - self._ptz[0] + 180.0) % 360.0 - 180.0
        return abs(dpan) <= cfg.max_pan_deg * k and abs(ptz[1] - self._ptz[1]) <= 0.2 * cfg.max_pan_deg * k \
            and abs(math.log(ptz[2] / self._ptz[2])) <= cfg.max_zoom * k

    def _camera_of(self, hom: Homography) -> PinholeCamera | None:
        """The camera behind a fitted homography (square pixels, centred principal point). No reprojection
        check: a fit from the lines in view need not reproduce far-off court corners to the pixel."""
        return camera_from_homography(hom, self.image_size, (self.court.length, self.court.width), math.inf)

    def _score(self, on: np.ndarray, G: np.ndarray) -> float:
        """Share of the court samples projected by ``G`` that land on (or next to) a line pixel."""
        h_img, w_img = on.shape
        uv, w = _project(G, self._pts)
        ok = (w > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w_img - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h_img - 1)
        if ok.sum() < self.config.min_matches:
            return 0.0
        xs, ys = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
        return float((on[ys, xs] | on[ys, xs + 1] | on[ys + 1, xs]).mean())

    def _reacquire(self, resp: np.ndarray) -> tuple[float, float, float] | None:
        """Best pan / tilt / zoom of the fixed camera centre for this frame's line response, or None."""
        cfg = self.config
        if self._center is None or self._ptz is None:
            return None
        pan0, tilt0, f0 = self._ptz
        on = resp >= cfg.min_response
        a, b, step = cfg.reacquire_pan_deg
        size, c = self.image_size, self._center
        coarse = []
        for zoom in cfg.reacquire_zoom:
            for dp in np.arange(a, b + 1e-9, step):
                cand = (pan0 + float(dp), tilt0, f0 * zoom)
                coarse.append((self._score(on, _ptz_G(c, cand, size)), cand))
        coarse.sort(key=lambda q: -q[0])
        best, best_score = None, cfg.reacquire_min_score
        for _, (pan, tilt, f) in coarse[: cfg.reacquire_top]:
            for dp in cfg.reacquire_fine_pan_deg:
                for dt in cfg.reacquire_fine_tilt_deg:
                    for z in cfg.reacquire_fine_zoom:
                        cand = (pan + dp, tilt + dt, f * z)
                        score = self._score(on, _ptz_G(c, cand, size))
                        if score > best_score:
                            best, best_score = cand, score
        return best

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "misses": self.misses, "ptz": self._ptz,
                "center": None if self._center is None else [float(v) for v in self._center]}
