"""3-D ball flights from one broadcast camera: gravity fixes the depth a single view cannot see.

A ball in free flight follows ``X(t) = p0 + v0 (t - t0) + g (t - t0)^2 / 2`` with ``g = (0, 0, -32.17) ft/s^2``.
Every image observation puts the ball on the ray from the camera centre ``C`` through its pixel, so
``(X(t_i) - C) x d_i = 0``. That is linear in ``(p0, v0)``: two equations per frame, six unknowns. Without gravity
the solution would only be known up to a scale about ``C`` (a flight twice as far and twice as fast looks the
same); the known ``g`` sets the scale, so a flight seen for a few tenths of a second gives its depth. The game
camera of a broadcast turns about a fixed centre (``broadcast_court``), so all rays share ``C`` and the frames'
rotations and zoom only change the ray directions.

``fit_flight`` starts from a depth sweep (``_sweep_init``): the equations above measure a miss in feet at the ball,
so on noisy pixels their least-squares solution comes out too near the camera (a few tenths of a second of a
flight 100 ft away land at 0-50 ft, gravity's small curvature being all that holds them out); with the depth of
one observation fixed the system is linear, and the depth whose flight reprojects best is taken. It then refines
``(p0, v0)`` by robust least squares on the reprojection error, optionally with a prior that the flight comes down
through a given rim. ``find_flights`` slides over a ball track (frame times, pixels, cameras), grows windows that
fit a parabola to within ``max_rms_px`` and checks that the fitted flight is physical (above the floor, plausible
speed, inside the arena); given the baskets, it refits each flight that is consistent with coming down through a
rim through it (a shot: under broadcast noise gravity fixes the depth to several feet, the rim pins the flight's
end). ``rim_pass`` reports how a flight passes a basket: the closest horizontal approach while descending through
rim height, and whether it went through the ring. ``release_point`` follows a flight back from its first
observation to a typical release height (the ball is usually lost in the shooter's hands for a few frames).
``tools/ball3d_bench.py`` measures all of it on synthetic broadcast shots and passes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .court_search import CameraParams

__all__ = ["G_FTS2", "Flight3D", "FlightSearchConfig", "RimPass", "find_flights", "fit_flight", "rays", "release_point",
           "rim_pass"]

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


class _Obs:
    """Observations of a ball track as arrays: times, pixels, unit rays, camera rotations and focal lengths (all
    cameras share the centre of the first)."""

    def __init__(self, t: np.ndarray, uv: np.ndarray, cams: Sequence[CameraParams], size: tuple[int, int]) -> None:
        self.t = np.asarray(t, float)
        self.uv = np.asarray(uv, float).reshape(-1, 2)
        self.R = np.stack([c.R() for c in cams]) if len(cams) else np.zeros((0, 3, 3))
        self.f = np.array([c.f for c in cams], float)
        self.C = np.asarray(cams[0].center, float) if len(cams) else np.zeros(3)
        self.half = np.array([size[0] / 2.0, size[1] / 2.0])
        q = np.column_stack([(self.uv - self.half) / self.f[:, None], np.ones(len(self.t))])
        d = np.einsum("nji,nj->ni", self.R, q)  # R^T q
        self.d = d / np.linalg.norm(d, axis=1, keepdims=True)

    def project(self, X: np.ndarray, idx: np.ndarray | slice = slice(None)) -> np.ndarray:
        """Pixels of points ``X`` (n, 3), or (k, n, 3) for k candidate flights, in the views ``idx``."""
        pc = np.einsum("nij,...nj->...ni", self.R[idx], X - self.C)
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.f[idx, None] * pc[..., :2] / pc[..., 2:3] + self.half


def _skews(d: np.ndarray) -> np.ndarray:
    """Cross-product matrices of the rows of ``d`` (n, 3) -> (n, 3, 3)."""
    S = np.zeros((len(d), 3, 3))
    S[:, 0, 1], S[:, 0, 2], S[:, 1, 2] = -d[:, 2], d[:, 1], -d[:, 0]
    S[:, 1, 0], S[:, 2, 0], S[:, 2, 1] = d[:, 2], -d[:, 1], d[:, 0]
    return S


_DEPTHS = np.exp(np.linspace(math.log(8.0), math.log(400.0), 50))  # depth sweep of the initial solution (ft)


def _sweep_init(obs: _Obs, idx: np.ndarray, huber_px: float) -> np.ndarray | None:
    """Initial (p0, v0) by a depth sweep. The cross-product equations ``d_i x (X(t_i) - C) = 0`` measure a miss in
    feet at the ball, so their least-squares solution shrinks toward the camera when the pixels are noisy (the
    rays' cross products are smaller nearer ``C``), with only gravity's small curvature to hold it out: a short
    window of noisy observations lands near or even behind the camera. With the depth of the middle observation
    fixed at ``D`` the system is linear and its solution affine in ``D``; the ``D`` (log-spaced 8-400 ft, refined
    by a parabola through the best three) whose flight reprojects best (Huber cost in pixels, which has no such
    bias) is taken."""
    n = len(idx)
    t, d = obs.t[idx], obs.d[idx]
    tau = t - t[0]
    S = _skews(d)
    A = np.zeros((3 * n + 1, 6))
    B = np.zeros((3 * n + 1, 2))
    A[:3 * n, :3] = S.reshape(3 * n, 3)
    A[:3 * n, 3:] = (S * tau[:, None, None]).reshape(3 * n, 3)
    B[:3 * n, 0] = np.einsum("nij,nj->ni", S, obs.C - 0.5 * np.outer(tau**2, _G)).ravel()
    m, W = n // 2, 1e3
    A[3 * n, :3], A[3 * n, 3:] = W * d[m], W * tau[m] * d[m]
    B[3 * n] = (W * float(d[m] @ (obs.C - 0.5 * _G * tau[m] ** 2)), W)
    sol, *_ = np.linalg.lstsq(A, B, rcond=None)
    xa, xb = sol[:, 0], sol[:, 1]
    Xa = xa[:3] + np.outer(tau, xa[3:]) + 0.5 * np.outer(tau**2, _G)
    Xb = xb[:3] + np.outer(tau, xb[3:])
    D = _DEPTHS
    X = Xa[None] + D[:, None, None] * Xb[None]
    pc_z = np.einsum("nj,...nj->...n", obs.R[idx][:, 2], X - obs.C)
    r = obs.project(X, idx) - obs.uv[idx]
    cost = np.sum(_huber_res(np.nan_to_num(r, nan=1e6, posinf=1e6, neginf=-1e6), huber_px) ** 2, axis=(1, 2))
    cost[np.any(pc_z <= 1.0, axis=1)] = np.inf
    k = int(np.argmin(cost))
    if not np.isfinite(cost[k]):
        return None
    lD = math.log(D[k])
    if 0 < k < len(D) - 1 and np.isfinite(cost[k - 1]) and np.isfinite(cost[k + 1]):
        c0, c1, c2 = cost[k - 1], cost[k], cost[k + 1]
        den = c0 - 2.0 * c1 + c2
        if den > 0:
            h = math.log(D[k + 1]) - lD
            lD += float(np.clip(0.5 * (c0 - c2) / den, -1.0, 1.0)) * h
    return xa + math.exp(lD) * xb


def _linear_init(obs: _Obs, idx: np.ndarray) -> np.ndarray | None:
    """The original initial solution: the cross-product equations weighted by inverse depth (a nominal 120 ft,
    re-weighted once with the first solution's depths). Biased toward the camera on noisy pixels (``_sweep_init``)."""
    n = len(idx)
    t = obs.t[idx]
    tau = t - t[0]
    C = obs.C
    d = obs.d[idx]
    A = np.zeros((3 * n, 6))
    b = np.zeros(3 * n)
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
    return sol


def _huber_res(r: np.ndarray, k: float) -> np.ndarray:
    """Residuals whose squares sum to twice the Huber loss (scale ``k``) of ``r``."""
    a = np.abs(r)
    return np.where(a <= k, r, np.sign(r) * np.sqrt(np.maximum(2.0 * k * a - k * k, 0.0)))


def _rim_miss(x: np.ndarray, rim: tuple[float, float, float]) -> tuple[np.ndarray, float]:
    """Offset (dx, dy, dz) from the rim centre of the flight ``x = (p0, v0)`` where it comes down through rim
    height, and when (after ``t0``); a flight whose apex stays below the rim: its apex and the height it lacks."""
    p0z, vz = float(x[2]), float(x[5])
    disc = vz * vz + 2.0 * G_FTS2 * (p0z - rim[2])
    if disc >= 0.0:
        tc, dz = (vz + math.sqrt(disc)) / G_FTS2, 0.0
    else:
        tc = vz / G_FTS2
        dz = p0z + vz * tc - 0.5 * G_FTS2 * tc * tc - rim[2]
    X = x[:3] + x[3:] * tc + 0.5 * _G * tc * tc
    return np.array([X[0] - rim[0], X[1] - rim[1], dz]), tc


def _fit(obs: _Obs, idx: np.ndarray, refine: bool = True, huber_px: float = 3.0,
         rim: tuple[float, float, float] | None = None, rim_sigma_ft: float = 0.6, start: Flight3D | None = None,
         init: str = "sweep") -> Flight3D | None:
    """``fit_flight`` on the observations ``idx`` of ``obs``."""
    idx = np.asarray(idx)
    n = len(idx)
    if n < 4:
        return None
    t = obs.t[idx]
    t0 = float(t[0])
    tau = t - t0
    if start is not None:
        sol: np.ndarray | None = np.concatenate([start.at(t0), start.velocity(t0)])
    elif init == "linear":
        sol = _linear_init(obs, idx)
    else:
        sol = _sweep_init(obs, idx, huber_px)
    if sol is None:
        return None
    p0, v0 = sol[:3], sol[3:]
    uvf = obs.uv[idx]
    if refine or rim is not None:
        from scipy.optimize import least_squares

        def res(x: np.ndarray) -> np.ndarray:
            X = x[:3] + np.outer(tau, x[3:]) + 0.5 * np.outer(tau**2, _G)
            return (obs.project(X, idx) - uvf).ravel()

        if rim is None:
            r = least_squares(res, sol, loss="huber", f_scale=huber_px, max_nfev=40)
        else:
            w_rim = huber_px / rim_sigma_ft  # a miss of one sigma costs as much as a pixel residual at the Huber scale

            def res_rim(x: np.ndarray) -> np.ndarray:
                return np.concatenate([_huber_res(res(x), huber_px), w_rim * _rim_miss(x, rim)[0]])

            r = least_squares(res_rim, sol, max_nfev=80)
        p0, v0 = r.x[:3], r.x[3:]
    X = p0 + np.outer(tau, v0) + 0.5 * np.outer(tau**2, _G)
    err = np.linalg.norm(obs.project(X, idx) - uvf, axis=1)
    if not np.all(np.isfinite(err)):
        return None
    return Flight3D(t0, float(t[-1]), np.asarray(p0, float), np.asarray(v0, float), float(np.sqrt(np.mean(err**2))), n)


def fit_flight(t: np.ndarray, uv: np.ndarray, cams: Sequence[CameraParams], size: tuple[int, int],
               refine: bool = True, huber_px: float = 3.0, rim: tuple[float, float, float] | None = None,
               rim_sigma_ft: float = 0.6, start: Flight3D | None = None, init: str = "sweep") -> Flight3D | None:
    """Ballistic flight through pixel observations ``uv`` at times ``t`` (see the module docstring).

    ``init``: the initial solution, ``"sweep"`` (``_sweep_init``: the depth that reprojects best, unbiased on noisy
    pixels) or ``"linear"`` (the original inverse-depth weighted cross-product solution, which comes out too near
    the camera, or behind it (None), on noisy pixels). ``rim``: (x, y, z) of a rim centre the flight is known to
    come down through (a shot that reached the rim); the refinement then also minimises the flight's horizontal
    miss of that point where it descends through height z (as ``rim_pass`` measures it), as a Gaussian prior of
    ``rim_sigma_ft``: a shot's ball crosses rim height within about a foot of the rim centre, which pins the depth
    of the flight's end that gravity alone fixes only loosely. ``start``: a flight to start the refinement from
    instead of the initial solution."""
    if len(t) < 4:
        return None
    return _fit(_Obs(t, uv, cams, size), np.arange(len(t)), refine, huber_px, rim, rim_sigma_ft, start, init)


@dataclass
class FlightSearchConfig:
    """Flight search settings. The original ones were ``max_rms_px=3.0, max_point_px=7.0, init="linear",
    grow_physical=True``: on tools/ball3d_bench.py's broadcast model (2 px detection noise, outliers and dropouts,
    cameras about 0.04 deg off) they gave 5% of the shots a rim pass within 3 ft of their rim; these give 45%, and
    87% with ``baskets`` (300 shots, seed 0)."""

    min_obs: int = 8  # inlier observations in a flight
    seed_obs: tuple[int, ...] = (10, 7)  # windows that seed a flight, longest first
    max_rms_px: float = 6.0  # 1080-line px; 2 px detection noise plus 0.04 deg camera error (2.8 px at f = 4000)
    #   give a good flight an rms of 3-6 px
    max_point_px: float = 12.0  # an observation this close to the growing flight's prediction joins it
    max_skip: int = 2  # observations off the flight skipped while growing (a wrong pick of the linker)
    max_gap_s: float = 0.25  # between consecutive observations
    max_speed_fts: float = 65.0
    min_speed_fts: float = 6.0  # slower "flights" are held or rolling balls
    z_range: tuple[float, float] = (-1.0, 40.0)
    xy_margin_ft: float = 12.0  # the flight stays above the court plus this margin
    init: str = "sweep"  # initial solution of each fit (``fit_flight``): "sweep" or the original "linear"
    grow_physical: bool = False  # growing, also require each refit to be physical (True originally): the loose
    #   depth of a short window then stops a flight at observations that fit it in the image
    # with ``baskets`` (find_flights): a flight whose refit through a rim centre stays consistent with its pixels
    # is kept as that refit (a shot that reached the rim)
    rim_sigma_ft: float = 0.6  # a shot comes down through rim height this close to the rim centre (sd)
    rim_max_chi2: float = 16.0  # the refit may cost this much more than the free fit, in units of its noise variance
    rim_window_s: tuple[float, float] = (-0.3, 0.6)  # the refit's rim crossing, from the flight's last observation


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
                 court_wh: tuple[float, float] = (94.0, 50.0), baskets: dict[str, tuple[float, float]] | None = None,
                 rim_height: float = 10.0) -> list[Flight3D]:
    """Ballistic segments of a ball track (observations in time order). A window of ``seed_obs`` observations that
    fits a physical flight seeds one; it grows forward and backward, taking every observation within
    ``max_point_px`` of the refitted flight and skipping up to ``max_skip`` that are not (a wrong pick of the path
    linker, the ball behind a body); the flight is kept when it has ``min_obs`` inliers and fits within
    ``max_rms_px`` after a final robust refinement. Seeds are not re-used.

    ``baskets`` (side -> rim centre (x, y), e.g. ``{s: court.basket_xy(s) for s in ("left", "right")}``) with
    ``rim_height``: each flight that rises to about rim height is refitted through each rim centre
    (``fit_flight(..., rim=...)``); the refit is kept when it costs at most ``rim_max_chi2`` more (pixel residuals
    plus the rim prior, in units of the free fit's residual variance) and comes down through rim height within
    ``rim_window_s`` of the flight's last observation. Under broadcast noise gravity fixes a free flight's depth only
    to several feet, so a shot's rim pass often lands feet from its rim; refitted, it comes down within about a foot
    of it (tools/ball3d_bench.py). A refitted flight's rim distance then reflects the prior more than the data, so it
    no longer tells a make from a miss (``RimPass.through``). Passes fail the test: forcing them through a rim costs
    far more (a lob seen only on its way up can pass it)."""
    cfg = config or FlightSearchConfig()
    s = size[1] / 1080.0
    t = np.asarray(t, float)
    uv = np.asarray(uv, float)
    fr = np.arange(len(t)) if frames is None else np.asarray(frames)
    n = len(t)
    used = np.zeros(n, bool)
    out: list[Flight3D] = []
    if n < 4:
        return out
    obs = _Obs(t, uv, cams, size)

    def fit(idx: list[int], refine: bool = False) -> Flight3D | None:
        return _fit(obs, np.asarray(idx), refine=refine, init=cfg.init)

    def ok(f: Flight3D | None) -> bool:
        return f is not None and f.rms_px <= cfg.max_rms_px * s and _physical(f, cfg, court_wh)

    def fits(f: Flight3D | None) -> bool:
        return ok(f) if cfg.grow_physical else f is not None and f.rms_px <= cfg.max_rms_px * s

    def grow(f: Flight3D, idx: list[int], step: int) -> tuple[Flight3D, list[int]]:
        k = idx[-1] + 1 if step > 0 else idx[0] - 1
        skips = 0
        while 0 <= k < n and not used[k] and skips <= cfg.max_skip:
            prev = idx[-1] if step > 0 else idx[0]
            if abs(t[k] - t[prev]) > cfg.max_gap_s * (skips + 1):
                break
            pred = obs.project(f.at(t[k])[None], [k])[0]
            if np.linalg.norm(pred - uv[k]) <= cfg.max_point_px * s:
                cand = idx + [k] if step > 0 else [k] + idx
                g = fit(cand)
                if fits(g):
                    f, idx, skips = g, cand, 0  # type: ignore[assignment]
                else:
                    skips += 1
            else:
                skips += 1
            k += step
        return f, idx

    def through_rim(f: Flight3D, idx: list[int]) -> Flight3D:
        """``f`` refitted through the rim it is consistent with (see the docstring), else ``f`` itself."""
        assert baskets is not None
        if f.apex_z < rim_height - 1.5:
            return f
        ia = np.asarray(idx)

        def cost(g: Flight3D) -> float:
            return float(np.sum(_huber_res((obs.project(g.at(t[ia]), ia) - uv[ia]).ravel(), 3.0) ** 2))

        c0 = cost(f)
        var = max(c0 / max(2 * len(idx) - 6, 1), 1e-6)
        best: tuple[float, Flight3D] | None = None
        for bx, by in baskets.values():
            rim = (float(bx), float(by), rim_height)
            g = _fit(obs, ia, rim=rim, rim_sigma_ft=cfg.rim_sigma_ft, start=f)
            if g is None or not ok(g):
                continue
            miss, tc = _rim_miss(np.concatenate([g.p0, g.v0]), rim)
            if not cfg.rim_window_s[0] <= g.t0 + tc - g.t1 <= cfg.rim_window_s[1]:
                continue
            chi2 = (cost(g) - c0) / var + float(miss @ miss) / cfg.rim_sigma_ft**2
            if chi2 <= cfg.rim_max_chi2 and (best is None or chi2 < best[0]):
                best = (chi2, g)
        return f if best is None else best[1]

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
            if baskets:
                final = through_rim(final, idx)
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


def release_point(f: Flight3D, z_release: float = 8.5) -> tuple[float, float]:
    """Floor point (x, y) under the ball where it left the shooter's hand: the fitted flight followed back from its
    first observation (``t0``) to the height ``z_release`` on its way up.

    The detector usually loses the ball for a few frames at the release (hands, head, arms), so ``f.p0`` is the ball
    already some way along the flight, closer to the basket (about two thirds of a foot per frame on a three). A
    first observation no higher than ``z_release`` is taken as it is (the ball was seen at the release).
    ``z_release``: the middle of the 7-10 ft of jump-shot releases (an assumption); a release a foot higher or lower
    moves the point about a foot along the flight's direction (a 45-degree launch)."""
    z0, vz = float(f.p0[2]), float(f.v0[2])
    if z0 <= z_release:
        return float(f.p0[0]), float(f.p0[1])
    tau = (vz - math.sqrt(vz * vz + 2.0 * G_FTS2 * (z0 - z_release))) / G_FTS2  # the earlier crossing (<= 0)
    X = f.at(f.t0 + tau)
    return float(X[0]), float(X[1])
