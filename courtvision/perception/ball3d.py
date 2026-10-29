"""3-D ball flights from one broadcast camera: gravity fixes the depth a single view cannot see.

A ball in free flight follows ``X(t) = p0 + v0 (t - t0) + g (t - t0)^2 / 2`` with ``g = (0, 0, -32.17) ft/s^2``.
Every image observation puts the ball on the ray from the camera centre ``C`` through its pixel, so
``(X(t_i) - C) x d_i = 0``. That is linear in ``(p0, v0)``: two equations per frame, six unknowns. Without gravity
the solution would only be known up to a scale about ``C`` (a flight twice as far and twice as fast looks the
same); the known ``g`` sets the scale, so a flight seen for a few tenths of a second gives its depth. The game
camera of a broadcast turns about a fixed centre (``broadcast_court``), so all rays share ``C`` and the frames'
rotations and zoom only change the ray directions.

``fit_flight`` solves the linear system (rows weighted by the inverse depth so that every frame counts in pixels),
then refines ``(p0, v0)`` by robust least squares on the reprojection error. ``find_flights`` slides over a ball
track (frame times, pixels, cameras), grows windows that fit a parabola to within ``max_rms_px`` and checks that
the fitted flight is physical (above the floor, plausible speed, inside the arena). ``rim_pass`` reports how a
flight passes a basket: the closest horizontal approach while descending through rim height, and whether it went
through the ring.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .court_search import CameraParams

__all__ = ["G_FTS2", "Flight3D", "FlightSearchConfig", "find_flights", "fit_flight", "rim_pass", "rays"]

G_FTS2 = 32.174
_G = np.array([0.0, 0.0, -G_FTS2])


@dataclass
class Flight3D:
    t0: float
    t1: float
    p0: np.ndarray  # position at t0 (court ft)
    v0: np.ndarray  # velocity at t0 (ft/s)
    rms_px: float
    n_obs: int
    frames: tuple[int, int] = (0, 0)  # first / last frame index of the observations used

    def at(self, t: float | np.ndarray) -> np.ndarray:
        tau = np.asarray(t, float) - self.t0
        return self.p0 + np.multiply.outer(tau, self.v0) + 0.5 * np.multiply.outer(tau**2, _G)

    def velocity(self, t: float) -> np.ndarray:
        return self.v0 + _G * (t - self.t0)

    @property
    def apex_z(self) -> float:
        tz = self.v0[2] / G_FTS2  # time of the apex after t0
        if tz <= 0:
            return float(self.p0[2])
        return float(self.p0[2] + self.v0[2] * tz - 0.5 * G_FTS2 * tz * tz)

    @property
    def speed0(self) -> float:
        return float(np.linalg.norm(self.v0))

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        return {"t0": self.t0, "t1": self.t1, "p0": [float(v) for v in self.p0], "v0": [float(v) for v in self.v0],
                "rms_px": self.rms_px, "n_obs": self.n_obs, "frames": list(self.frames)}


def rays(uv: np.ndarray, cams: Sequence[CameraParams], size: tuple[int, int]) -> np.ndarray:
    """Unit world directions of pixels ``uv`` (N, 2) seen by ``cams`` (one per pixel)."""
    out = np.empty((len(uv), 3))
    cx, cy = size[0] / 2.0, size[1] / 2.0
    for i, ((u, v), cam) in enumerate(zip(uv, cams, strict=True)):
        d = cam.R().T @ np.array([(u - cx) / cam.f, (v - cy) / cam.f, 1.0])
        out[i] = d / np.linalg.norm(d)
    return out


def _skew(d: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]])


def _project(X: np.ndarray, cam: CameraParams, size: tuple[int, int]) -> np.ndarray:
    pc = cam.R() @ (X - np.asarray(cam.center, float))
    return np.array([cam.f * pc[0] / pc[2] + size[0] / 2.0, cam.f * pc[1] / pc[2] + size[1] / 2.0])


def fit_flight(t: np.ndarray, uv: np.ndarray, cams: Sequence[CameraParams], size: tuple[int, int],
               refine: bool = True, huber_px: float = 3.0) -> Flight3D | None:
    """Ballistic flight through pixel observations ``uv`` at times ``t`` (see the module docstring)."""
    n = len(t)
    if n < 4:
        return None
    t = np.asarray(t, float)
    t0 = float(t[0])
    tau = t - t0
    C = np.asarray(cams[0].center, float)
    d = rays(np.asarray(uv, float), cams, size)
    A = np.zeros((3 * n, 6))
    b = np.zeros(3 * n)
    # depth weights: start from a nominal 120 ft, re-weight once with the first solution's depths
    w = np.full(n, 1.0 / 120.0)
    sol = None
    for _ in range(2):
        for i in range(n):
            S = _skew(d[i])
            A[3 * i:3 * i + 3, :3] = S * w[i]
            A[3 * i:3 * i + 3, 3:] = S * tau[i] * w[i]
            b[3 * i:3 * i + 3] = S @ (C - 0.5 * _G * tau[i] ** 2) * w[i]
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        X = sol[:3] + np.outer(tau, sol[3:]) + 0.5 * np.outer(tau**2, _G)
        depth = np.einsum("ij,ij->i", X - C, d)
        if np.any(depth <= 1.0):
            return None
        w = 1.0 / depth
    assert sol is not None
    p0, v0 = sol[:3], sol[3:]
    if refine:
        from scipy.optimize import least_squares

        uvf = np.asarray(uv, float)

        def res(x: np.ndarray) -> np.ndarray:
            X = x[:3] + np.outer(tau, x[3:]) + 0.5 * np.outer(tau**2, _G)
            return np.concatenate([_project(X[i], cams[i], size) - uvf[i] for i in range(n)])

        r = least_squares(res, sol, loss="huber", f_scale=huber_px, max_nfev=40)
        p0, v0 = r.x[:3], r.x[3:]
    X = p0 + np.outer(tau, v0) + 0.5 * np.outer(tau**2, _G)
    err = np.array([np.linalg.norm(_project(X[i], cams[i], size) - np.asarray(uv[i], float)) for i in range(n)])
    return Flight3D(t0, float(t[-1]), np.asarray(p0, float), np.asarray(v0, float), float(np.sqrt(np.mean(err**2))), n)


@dataclass
class FlightSearchConfig:
    min_obs: int = 8  # inlier observations in a flight
    seed_obs: tuple[int, ...] = (10, 7)  # windows that seed a flight, longest first
    max_rms_px: float = 3.0  # 1080-line px
    max_point_px: float = 7.0  # an observation this close to the growing flight's prediction joins it
    max_skip: int = 2  # observations off the flight skipped while growing (a wrong pick of the linker)
    max_gap_s: float = 0.25  # between consecutive observations
    max_speed_fts: float = 65.0
    min_speed_fts: float = 6.0  # slower "flights" are held or rolling balls
    z_range: tuple[float, float] = (-1.0, 40.0)
    xy_margin_ft: float = 12.0  # the flight stays above the court plus this margin


def _physical(f: Flight3D, cfg: FlightSearchConfig, court_wh: tuple[float, float]) -> bool:
    if not cfg.min_speed_fts <= f.speed0 <= cfg.max_speed_fts:
        return False
    ts = np.linspace(f.t0, f.t1, 6)
    X = f.at(ts)
    m = cfg.xy_margin_ft
    if np.any(X[:, 2] < cfg.z_range[0]) or np.any(X[:, 2] > cfg.z_range[1]):
        return False
    return bool(np.all((X[:, 0] > -m) & (X[:, 0] < court_wh[0] + m) & (X[:, 1] > -m) & (X[:, 1] < court_wh[1] + m)))


def find_flights(t: np.ndarray, uv: np.ndarray, cams: Sequence[CameraParams], size: tuple[int, int],
                 frames: np.ndarray | None = None, config: FlightSearchConfig | None = None,
                 court_wh: tuple[float, float] = (94.0, 50.0)) -> list[Flight3D]:
    """Ballistic segments of a ball track (observations in time order). A window of ``seed_obs`` observations that
    fits a physical flight seeds one; it grows forward and backward, taking every observation within
    ``max_point_px`` of the refitted flight and skipping up to ``max_skip`` that are not (a wrong pick of the path
    linker, the ball behind a body); the flight is kept when it has ``min_obs`` inliers and fits within
    ``max_rms_px`` after a final robust refinement. Seeds are not re-used."""
    cfg = config or FlightSearchConfig()
    s = size[1] / 1080.0
    t = np.asarray(t, float)
    uv = np.asarray(uv, float)
    fr = np.arange(len(t)) if frames is None else np.asarray(frames)
    n = len(t)
    used = np.zeros(n, bool)
    out: list[Flight3D] = []

    def fit(idx: list[int], refine: bool = False) -> Flight3D | None:
        return fit_flight(t[idx], uv[idx], [cams[k] for k in idx], size, refine=refine)

    def ok(f: Flight3D | None) -> bool:
        return f is not None and f.rms_px <= cfg.max_rms_px * s and _physical(f, cfg, court_wh)

    def grow(f: Flight3D, idx: list[int], step: int) -> tuple[Flight3D, list[int]]:
        k = idx[-1] + 1 if step > 0 else idx[0] - 1
        skips = 0
        while 0 <= k < n and not used[k] and skips <= cfg.max_skip:
            prev = idx[-1] if step > 0 else idx[0]
            if abs(t[k] - t[prev]) > cfg.max_gap_s * (skips + 1):
                break
            pred = _project(f.at(t[k]), cams[k], size)
            if np.linalg.norm(pred - uv[k]) <= cfg.max_point_px * s:
                cand = idx + [k] if step > 0 else [k] + idx
                g = fit(cand)
                if ok(g):
                    f, idx, skips = g, cand, 0  # type: ignore[assignment]
                else:
                    skips += 1
            else:
                skips += 1
            k += step
        return f, idx

    i = 0
    while i < n:
        seeded = None
        for w in cfg.seed_obs:
            if i + w > n or used[i:i + w].any() or np.any(np.diff(t[i:i + w]) > cfg.max_gap_s):
                continue
            idx = list(range(i, i + w))
            f = fit(idx)
            if ok(f):
                seeded = (f, idx)
                break
        if seeded is None:
            i += 1
            continue
        f, idx = seeded
        f, idx = grow(f, idx, +1)
        f, idx = grow(f, idx, -1)
        final = fit(idx, refine=True)
        if len(idx) >= cfg.min_obs and ok(final):
            assert final is not None
            final.frames = (int(fr[idx[0]]), int(fr[idx[-1]]))
            out.append(final)
            used[idx[0]:idx[-1] + 1] = True
            i = idx[-1] + 1
        else:
            i += 1
    return out


@dataclass(frozen=True)
class RimPass:
    side: str
    t: float  # when the flight descends through rim height (or its closest approach)
    dist_ft: float  # horizontal distance from the rim centre at that moment
    through: bool  # inside the ring (centre within rim radius - ball radius + slack)
    apex_ft: float


def rim_pass(f: Flight3D, baskets: dict[str, tuple[float, float]], rim_height: float = 10.0,
             rim_radius: float = 0.75, ball_radius: float = 0.39, slack_ft: float = 0.25) -> RimPass | None:
    """The basket the flight comes nearest to at rim height on its way down (None if it never descends through
    rim height within its span, e.g. a pass)."""
    a, b, c = 0.5 * _G[2], f.v0[2], f.p0[2] - rim_height
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    roots = sorted(((-b + sq) / (2 * a), (-b - sq) / (2 * a)) for sq in (math.sqrt(disc),))[0]
    t_down = f.t0 + max(roots)  # the later crossing is the descending one
    if t_down < f.t0 - 1e-6 or t_down > f.t1 + 0.6:  # extrapolate a little past the last frame (net occlusion)
        return None
    X = f.at(t_down)
    best = None
    for side, (bx, by) in baskets.items():
        dist = math.hypot(X[0] - bx, X[1] - by)
        if best is None or dist < best.dist_ft:
            best = RimPass(side, float(t_down), float(dist), dist <= rim_radius - ball_radius + slack_ft, f.apex_z)
    return best
