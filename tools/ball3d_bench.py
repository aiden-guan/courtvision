"""Synthetic benchmark of the 3-D ball flights (``perception/ball3d.py``) seen by a panning broadcast camera.

    .venv/bin/python tools/ball3d_bench.py                      # 300 shots + 150 passes, seed 0: about 2 min
    .venv/bin/python tools/ball3d_bench.py --seed 1 --variants baseline,recommended --json out.json
    .venv/bin/python tools/ball3d_bench.py --noise px=3,cam=1.5  # another observation model ('clean': none)

Each sample is one clip (about 1.5-3.5 s at 29.97 fps, 1920x1080) of one shot or one pass, with the ball held or
dribbled before it and after it. The clip's ball track goes through the steps of ``nba/analyze.build_ball_track``
(the path cut where it misses more than 6 frames, ``find_flights`` on each part with at least 6 observations) and
of ``nba/fusion._flight_values``, re-implemented here (fusion.py is not imported): the flights whose ``rim_pass``
is within 3 ft of a rim; for a missed shot at time t the one whose rim pass is nearest t within 1.5 s; 3 points
when its release point is beyond the 3-point line (``court.is_three``) by at least 0.5 ft (``_arc_margin``), else
2. The missed shot's time is the true rim time (as from a shot-clock reset at the rim touch). Every shot is scored
as if it were missed (makes and misses differ only after the rim).

Variants (``--variants``, comma-separated): ``baseline`` is the flight search as it was before this benchmark
(``FlightSearchConfig(max_rms_px=3, max_point_px=7, init="linear", grow_physical=True)``) with fusion's release
point ``f.p0``; ``defaults`` today's ``FlightSearchConfig()`` with ``f.p0`` and no baskets (what
``build_ball_track`` and fusion give unchanged); ``recommended`` the same with ``find_flights(..., baskets=...)``
and ``ball3d.release_point``. A variant can be changed with ``+`` parts: ``+field=value`` of
``FlightSearchConfig`` (``+init='linear'``, ``+max_rms_px=3.0``, ``+rim_window_s=(-0.3,0.4)``), ``+rim`` /
``+norim`` (baskets or none), ``+release`` / ``+release=Z`` (``release_point`` with ``z_release=Z``) /
``+release=p0`` / ``+release=sym`` (the flight's rise through 8.5 ft whether before or after its first
observation), e.g. ``baseline+init='sweep'+max_rms_px=6.0+max_point_px=12.0``.

Everything is synthetic; these are the assumptions (design choices, not measurements):

* Camera: centre x = 47 +- 1, y in [-95, -65], z in [26, 40] ft per clip (a real one: about (47, -86, 33)),
  fixed in the clip and known exactly. The operator aims at the ball's position low-passed with a 0.2-0.45 s time
  constant, the aim height pulled toward 4 ft (a high arc can leave the top of a tight frame), plus a framing
  offset of up to 10% / 8% of the horizontal / vertical field of view; pan at most 3 deg/frame; roll up to 0.3
  deg; focal length log-uniform in 2000-5000 px at the start of the clip, drifting by up to 20%/s, within
  2000-6000 px.
* The pipeline's cameras are slightly wrong: pan and tilt each off by AR(1) noise (0.9 per frame, sd 0.03 deg)
  plus white noise (sd 0.025 deg), about 0.04 deg in all; roll 0.02 + 0.02 deg; focal length 0.25% + 0.15%.
* Detections: pixel noise N(0, 2 px) per axis. Outliers (the linker took a wrong candidate) in runs of mean 2
  frames, starting on 3.5% of frames, 15-80 px off in a random direction (+-3 px jitter). Dropouts: 3% of
  frames at random, occlusions of 2-6 frames starting on 0.6% of frames, the ball outside the image, and for
  85% of shots the ball lost at the release (from 0-2 frames before it to 1-4 frames after) and at the rim (from
  0-4 frames before the rim time to 1-8 frames after); passes lose 0-2 frames at the release and the catch half
  the time. The report prints the share of frames with a detection within 15 px of the truth (about 88% outside
  those release / rim / catch windows, as measured for the student detector on labelled broadcast frames).
* Shots: either basket. The shooter's feet: 2-6 ft from the basket 12%, 6-14 ft 12%, 14-19 ft 10%, long twos
  (0.1-4.75 ft inside the arc, or 0.1-3 ft inside a corner line) 16%, corner threes (0.1-2.6 ft beyond the 22-ft
  line) 14%, above-the-break threes (0.1-3.25 ft beyond 23.75 ft) 28%, deep threes (27-30 ft) 8%: half are
  threes. The ball is released 0.5 +- 0.35 ft in front of the feet toward the rim (sim/simulator.py
  ``_release_xy`` uses 0.5 ft) and +-0.25 ft sideways, 7-10 ft high, with an entry angle of 38-50 deg (45-58
  deg within 8 ft): flights of 0.5-1.5 s with the apex at 11-20.5 ft. The truth value is ``court.is_three`` at
  the feet. 45% go through the ring (rim-height crossing within 0.3 ft of its centre), 55% hit the rim (crossing
  0.45-1.3 ft from the centre) and rebound for up to 1.1 s. Before the release: a dribble (half the clips) and
  a 0.35-0.65 s gather that ends in a 0.08-0.15 s push to the release velocity (not ballistic).
* Passes (the negatives): chest 40%, overhead 25%, lob (apex 11-17 ft) 15%, bounce 20%, 8-45 ft long, caught
  at least 6 ft from either basket; a pass whose true path comes down through rim height within 4.5 ft of a rim
  is drawn again. Held or dribbled before the throw and held after the catch.
* Not modelled: an error in the camera centre, lens distortion, rolling shutter, motion-blur bias of the
  detections, a ball hidden by a player for longer than 6 frames mid-flight.

Reported per variant: recall (shots with a flight whose rim pass is within 3 ft of a rim and 1.5 s of the true rim
time), false positives (passes with a flight whose rim pass is within 3 ft of a rim), the error of the release
floor point used for the 2 / 3 call against the true one (median and 90th percentile, ft; also of its distance to
the basket, and along / across the camera's line of sight), and the 2 / 3 accuracy of the shots the fusion rule
gives a value (all, and those whose feet are within 3 ft of the 3-point line), also with the call made at other
distances beyond the line. ``oracle`` applies the same rules to the true release point: what the 0.5-ft margin
costs given the assumed ball-ahead-of-the-feet offset. Diagnostics: whether the matched flight is the shot's own,
starts before the release (it took held-ball frames), and "impure" flights (less than 80% of their time span in
one free-flight piece of the truth: the gather or a dribble fitted as a flight, or a shot glued to its rebound).
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import dataclasses
import functools
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from courtvision.geometry.court import court_for  # noqa: E402
from courtvision.perception import ball3d  # noqa: E402
from courtvision.perception.court_search import CameraParams  # noqa: E402

FPS = 29.97
SIZE = (1920, 1080)
COURT = court_for("nba")
BASKETS = {s: COURT.basket_xy(s) for s in ("left", "right")}
G = np.array([0.0, 0.0, -ball3d.G_FTS2])
BALL_R = 0.39
LINK_MAX_GAP = 6  # analyze.BallLinkConfig.max_gap: build_ball_track cuts the path at longer gaps
RIM_GATE_FT = 3.0  # fusion: a flight's rim pass this close to a rim
MATCH_S = 1.5  # fusion.FusionConfig.flight_match_s
THREE_MARGIN_FT = 0.5  # fusion.FusionConfig.three_margin_ft
MARGINS = (-1.0, -0.5, 0.0, 0.5, 1.0)  # other margins (``line_margin`` >= m: fusion's rule when m >= 0), reported


# --------------------------------------------------------------------------- fusion's 2 / 3 rule (re-implemented)


def arc_margin(x: float, y: float, side: str) -> float:
    """fusion._arc_margin: how far (ft) the point lies beyond the 3-point line (negative inside)."""
    bx, by = COURT.basket_xy(side)  # type: ignore[arg-type]
    lateral = abs(y - by)
    if lateral >= COURT.corner_three_dist and abs(x - bx) <= COURT.three_junction_along:
        return lateral - COURT.corner_three_dist
    return math.hypot(x - bx, y - by) - COURT.three_pt_radius


def fusion_value(x: float, y: float, side: str, margin_ft: float = THREE_MARGIN_FT) -> int:
    """fusion._flight_values: 3 when beyond the line and at least ``margin_ft`` beyond the arc, else 2."""
    return 3 if COURT.is_three(x, y, side) and arc_margin(x, y, side) >= margin_ft else 2  # type: ignore[arg-type]


def margin_value(x: float, y: float, side: str, margin_ft: float) -> int:
    """3 when the point is at least ``margin_ft`` beyond the 3-point line (``line_margin``; the same as fusion's rule
    for a margin >= 0, which cannot go below 0 since it asks ``court.is_three`` first)."""
    return 3 if line_margin(x, y, side) >= margin_ft else 2


def line_margin(x: float, y: float, side: str) -> float:
    """Signed distance (ft) beyond the 3-point line in the line's own terms (corner segments below the junction,
    the arc beyond it): positive exactly when ``court.is_three``."""
    bx, by = COURT.basket_xy(side)  # type: ignore[arg-type]
    along = (x - bx) * (1.0 if side == "left" else -1.0)
    if along <= COURT.three_junction_along:
        return abs(y - by) - COURT.corner_three_dist
    return math.hypot(x - bx, y - by) - COURT.three_pt_radius


# --------------------------------------------------------------------------- ball paths


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _perp(u: np.ndarray) -> np.ndarray:
    return np.array([-u[1], u[0]])


class Path3D:
    """A ball path made of pieces (t0, t1, fn) with fn(times) -> (n, 3) positions."""

    def __init__(self) -> None:
        self.pieces: list[tuple[float, float, Callable[[np.ndarray], np.ndarray]]] = []

    def add(self, t0: float, t1: float, fn: Callable[[np.ndarray], np.ndarray]) -> float:
        self.pieces.append((t0, t1, fn))
        return t1

    @property
    def end(self) -> float:
        return self.pieces[-1][1]

    def ballistic_spans(self) -> list[tuple[float, float]]:
        """The free-flight pieces (a flight that spans more than one of them, or other motion, is impure)."""
        return [(t0, t1) for t0, t1, fn in self.pieces if getattr(fn, "ballistic", False) and t1 > t0]

    def at(self, t: np.ndarray) -> np.ndarray:
        out = np.full((len(t), 3), np.nan)
        for t0, t1, fn in self.pieces:
            m = (t >= t0) & (t < t1)
            if m.any():
                out[m] = fn(t[m])
        return out


def ballistic(p: np.ndarray, v: np.ndarray, t0: float) -> Callable[[np.ndarray], np.ndarray]:
    p, v = np.asarray(p, float), np.asarray(v, float)

    def fn(t: np.ndarray) -> np.ndarray:
        return p + np.outer(t - t0, v) + 0.5 * np.outer((t - t0) ** 2, G)

    fn.ballistic = True  # type: ignore[attr-defined]
    return fn


def ease(a: np.ndarray, b: np.ndarray, t0: float, t1: float) -> Callable[[np.ndarray], np.ndarray]:
    """From ``a`` to ``b`` with zero speed at both ends (a hand moving the ball)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return lambda t: a + np.outer(0.5 - 0.5 * np.cos(math.pi * np.clip((t - t0) / (t1 - t0), 0.0, 1.0)), b - a)


def push(q: np.ndarray, v: np.ndarray, t0: float, tp: float) -> Callable[[np.ndarray], np.ndarray]:
    """From rest at ``q`` to velocity ``v`` in ``tp`` s at a constant acceleration (the release push)."""
    q, a = np.asarray(q, float), np.asarray(v, float) / tp
    return lambda t: q + 0.5 * np.outer((t - t0) ** 2, a)


def hold(p: np.ndarray, v: np.ndarray, t0: float) -> Callable[[np.ndarray], np.ndarray]:
    p, v = np.asarray(p, float), np.asarray(v, float)
    return lambda t: p + np.outer(t - t0, v)


def dribble(body0: np.ndarray, body_v: np.ndarray, off: np.ndarray, h: float, period: float,
            t0: float) -> Callable[[np.ndarray], np.ndarray]:
    def fn(t: np.ndarray) -> np.ndarray:
        tau = t - t0
        xy = body0[None, :] + np.outer(tau, body_v) + off[None, :]
        z = BALL_R + (h - BALL_R) * np.abs(np.cos(math.pi * tau / period))
        return np.column_stack([xy, z])
    return fn


def descend_time(z0: float, vz: float, z: float) -> float | None:
    """Time after which a ball at height z0 with vertical speed vz comes down through height z."""
    a, b, c = -0.5 * ball3d.G_FTS2, vz, z0 - z
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    r = max((-b + math.sqrt(disc)) / (2 * a), (-b - math.sqrt(disc)) / (2 * a))
    return r if r > 0 else None


def fall_and_bounce(path: Path3D, p: np.ndarray, v: np.ndarray, t0: float, t_end: float, e: float = 0.6) -> None:
    """Ballistic from (p, v) at t0 with floor bounces (restitution ``e``) until ``t_end``."""
    t, p, v = t0, np.asarray(p, float), np.asarray(v, float)
    for _ in range(4):
        tf = descend_time(float(p[2]), float(v[2]), BALL_R)
        if tf is None or t + tf >= t_end:
            path.add(t, t_end, ballistic(p, v, t))
            return
        path.add(t, t + tf, ballistic(p, v, t))
        p = p + v * tf + 0.5 * G * tf * tf
        v = np.array([0.85 * v[0], 0.85 * v[1], -e * (v[2] - ball3d.G_FTS2 * tf)])
        p[2] = BALL_R
        t += tf
    path.add(t, t_end, hold(p, np.zeros(3), t))


def rim_crossing(p: np.ndarray, v: np.ndarray) -> tuple[float, np.ndarray] | None:
    """When / where a flight from (p, v) comes down through rim height."""
    tc = descend_time(float(p[2]), float(v[2]), COURT.rim_height)
    if tc is None:
        return None
    return tc, p + v * tc + 0.5 * G * tc * tc


# --------------------------------------------------------------------------- shots and passes

SHOT_MIX = (("rim", 0.12), ("short", 0.12), ("mid", 0.10), ("long2", 0.16), ("corner3", 0.14), ("ab3", 0.28),
            ("deep3", 0.08))
PHI_J = math.atan2(COURT.corner_three_dist, COURT.three_junction_along)  # the arc meets the corner lines


def _shot_spot(rng: np.random.Generator, side: str) -> tuple[str, np.ndarray]:
    """(category, feet (x, y)) of a shot at ``side``'s basket."""
    bx, by = BASKETS[side]
    sgn = 1.0 if side == "left" else -1.0
    names = [c for c, _ in SHOT_MIX]
    probs = np.array([p for _, p in SHOT_MIX])
    cat = str(rng.choice(names, p=probs / probs.sum()))
    for _ in range(200):
        if cat in ("rim", "short", "mid"):
            lo, hi = {"rim": (2.0, 6.0), "short": (6.0, 14.0), "mid": (14.0, 19.0)}[cat]
            d, phi = rng.uniform(lo, hi), rng.uniform(-math.radians(85), math.radians(85))
            along, lat = d * math.cos(phi), d * math.sin(phi)
        elif cat == "corner3" or (cat == "long2" and rng.random() < 0.3):
            m = rng.uniform(0.1, 2.6) if cat == "corner3" else rng.uniform(-3.0, -0.1)
            along = rng.uniform(-1.0, COURT.three_junction_along - 0.05)
            lat = (COURT.corner_three_dist + m) * (1.0 if rng.random() < 0.5 else -1.0)
        else:
            if cat == "ab3":
                r = COURT.three_pt_radius + rng.uniform(0.1, 3.25)
            elif cat == "deep3":
                r = rng.uniform(27.0, 30.0)
            else:
                r = COURT.three_pt_radius + rng.uniform(-4.75, -0.1)
            phi = rng.uniform(-PHI_J, PHI_J)
            along, lat = r * math.cos(phi), r * math.sin(phi)
        x, y = bx + sgn * along, by + lat
        if 0.5 <= x <= COURT.length - 0.5 and 0.4 <= y <= COURT.width - 0.4:
            if cat in ("ab3", "deep3") and not COURT.is_three(x, y, side):  # type: ignore[arg-type]
                continue
            return cat, np.array([x, y])
    raise RuntimeError("no shot spot")


@dataclass
class Truth:
    kind: str  # "shot" | "pass"
    t_rel: float
    rel: np.ndarray  # release point (x, y, z)
    side: str = ""
    feet: np.ndarray = field(default_factory=lambda: np.zeros(2))
    t_rim: float = 0.0
    made: bool = False
    value: int = 0
    margin: float = 0.0  # feet beyond the 3-point line (ft, negative inside)
    cat: str = ""
    t_end_flight: float = 0.0  # end of the ballistic stretch of the shot / pass
    spans: list[tuple[float, float]] = field(default_factory=list)  # every free-flight piece of the clip
    cam_c: np.ndarray = field(default_factory=lambda: np.zeros(3))  # the camera centre of the clip


def simulate_shot(rng: np.random.Generator) -> tuple[Path3D, Truth, dict[str, Any]]:
    side = "left" if rng.random() < 0.5 else "right"
    basket = np.array(BASKETS[side])
    cat, feet = _shot_spot(rng, side)
    to_b = _unit(basket - feet)
    d_feet = float(np.linalg.norm(basket - feet))
    fwd = min(float(np.clip(rng.normal(0.5, 0.35), -0.4, 1.4)), 0.4 * d_feet)
    rel_xy = feet + to_b * fwd + _perp(to_b) * rng.normal(0.0, 0.25)
    z_rel = rng.uniform(7.0, 10.0)
    rel = np.array([*rel_xy, z_rel])
    made = bool(rng.random() < 0.45)
    if made:
        r, a = 0.3 * math.sqrt(rng.random()), rng.uniform(0.0, 2 * math.pi)
    else:
        r = rng.uniform(0.45, 1.3)
        front = math.atan2(*(rel_xy - basket)[::-1])  # toward the shooter
        u = rng.random()
        base = front if u < 0.5 else (front + math.pi if u < 0.85 else front + math.copysign(math.pi / 2, rng.normal()))
        a = base + rng.normal(0.0, math.radians(35))
    cross = np.array([*(basket + r * np.array([math.cos(a), math.sin(a)])), COURT.rim_height])
    D = float(np.linalg.norm(cross[:2] - rel_xy))
    for _ in range(50):
        theta = math.radians(rng.uniform(45.0, 58.0) if d_feet < 8.0 else rng.uniform(38.0, 50.0))
        T = math.sqrt(2.0 * max(COURT.rim_height - z_rel + D * math.tan(theta), 0.05) / ball3d.G_FTS2)
        v0 = (cross - rel - 0.5 * G * T * T) / T
        apex = z_rel + max(v0[2], 0.0) ** 2 / (2 * ball3d.G_FTS2)
        if 11.0 <= apex <= 20.5:
            break
    path = Path3D()
    # before the release: a dribble or a catch, then the gather and the push
    t = 0.0
    side_off = _perp(to_b) * 0.9
    h = rng.uniform(2.8, 4.2)
    start = np.array([*(feet + to_b * 0.4 + side_off * 0.5), h])
    if rng.random() < 0.5:
        period = rng.uniform(0.4, 0.55)
        n_b = int(rng.integers(1, 3))
        body_v = _unit(rng.normal(size=2)) * rng.uniform(0.0, 6.0)
        Td = n_b * period
        t = path.add(0.0, Td, dribble(feet - body_v * Td, body_v, side_off, h, period, 0.0))
        start = np.array([*(feet + side_off), h])
    else:
        th = rng.uniform(0.2, 0.4)
        t = path.add(0.0, th, hold(start - np.array([*(to_b * 0.6), 0.0]), np.array([*(to_b * 0.6 / th), 0.0]), 0.0))
    tg, tp = rng.uniform(0.35, 0.65), rng.uniform(0.08, 0.15)
    q = rel - 0.5 * v0 * tp
    t = path.add(t, t + tg - tp, ease(start, q, t, t + tg - tp))
    t_rel = path.add(t, t + tp, push(q, v0, t, tp))
    t_rim = t_rel + T
    if made:
        vz_rim = v0[2] - ball3d.G_FTS2 * T
        dt_in = descend_time(COURT.rim_height, float(vz_rim), 9.4) or 0.02
        t = path.add(t_rel, t_rim + dt_in, ballistic(rel, v0, t_rel))
        p_in = rel + v0 * (t_rim + dt_in - t_rel) + 0.5 * G * (t_rim + dt_in - t_rel) ** 2
        v_net = rng.uniform(5.0, 9.0)
        p_out = np.array([basket[0], basket[1], 8.0]) + 0.3 * (p_in - np.array([*basket, p_in[2]]))
        p_out[2] = 8.0
        t_out = t + (p_in[2] - 8.0) / v_net
        t = path.add(t, t_out, ease(p_in, p_out, t, t_out))
        end = t_rim + rng.uniform(0.6, 1.0)
        fall_and_bounce(path, p_out, np.array([*rng.normal(0.0, 0.7, 2), -v_net]), t, max(end, t + 0.2))
        t_end_flight = t_rim + dt_in
    else:
        t = path.add(t_rel, t_rim, ballistic(rel, v0, t_rel))
        out = _unit(cross[:2] - basket)
        ang = rng.uniform(-math.radians(60), math.radians(60))
        ca, sa = math.cos(ang), math.sin(ang)
        dirn = np.array([out[0] * ca - out[1] * sa, out[0] * sa + out[1] * ca])
        if (1.0 if side == "left" else -1.0) * dirn[0] < 0:  # toward the board: it comes back to the court
            dirn[0] = -dirn[0]
        v_reb = np.array([*(dirn * rng.uniform(3.0, 14.0)), rng.uniform(2.0, 12.0)])
        z_catch = rng.uniform(6.0, 9.0)
        t_c = min(descend_time(COURT.rim_height, float(v_reb[2]), z_catch) or 1.1, 1.1)
        t = path.add(t_rim, t_rim + t_c, ballistic(cross, v_reb, t_rim))
        p_c = cross + v_reb * t_c + 0.5 * G * t_c * t_c
        down = p_c.copy()
        down[2] = 4.5
        t = path.add(t, t + 0.4, ease(p_c, down, t, t + 0.4))
        path.add(t, t + rng.uniform(0.0, 0.3), hold(down, np.zeros(3), t))
        t_end_flight = t_rim
    value = 3 if COURT.is_three(float(feet[0]), float(feet[1]), side) else 2  # type: ignore[arg-type]
    truth = Truth("shot", t_rel, rel, side, feet, t_rim, made, value, line_margin(float(feet[0]), float(feet[1]), side),
                  cat, t_end_flight)
    return path, truth, {"apex": apex, "T": T}


PASS_MIX = (("chest", 0.40), ("overhead", 0.25), ("lob", 0.15), ("bounce", 0.20))


def simulate_pass(rng: np.random.Generator) -> tuple[Path3D, Truth, dict[str, Any]]:
    names = [c for c, _ in PASS_MIX]
    probs = np.array([p for _, p in PASS_MIX])
    kind = str(rng.choice(names, p=probs / probs.sum()))
    for _ in range(1000):
        passer = np.array([rng.uniform(3.0, 91.0), rng.uniform(2.0, 48.0)])
        dist = rng.uniform(*{"chest": (10.0, 35.0), "overhead": (18.0, 45.0), "lob": (12.0, 40.0),
                             "bounce": (8.0, 22.0)}[kind])
        a = rng.uniform(0.0, 2 * math.pi)
        catch = passer + dist * np.array([math.cos(a), math.sin(a)])
        if not (1.0 <= catch[0] <= COURT.length - 1.0 and 1.0 <= catch[1] <= COURT.width - 1.0):
            continue
        if min(math.dist(catch, b) for b in BASKETS.values()) < 6.0:
            continue
        legs: list[tuple[np.ndarray, np.ndarray, float]] = []  # (p, v, duration) of each ballistic leg
        if kind == "bounce":
            z_rel, speed, frac = rng.uniform(3.0, 4.5), rng.uniform(18.0, 30.0), rng.uniform(0.55, 0.7)
            rel = np.array([*passer, z_rel])
            T1 = frac * dist / speed
            hv = (catch - passer) / dist * speed
            v1 = np.array([*hv, (BALL_R - z_rel + 0.5 * ball3d.G_FTS2 * T1 * T1) / T1])
            legs.append((rel, v1, T1))
            pb = rel + v1 * T1 + 0.5 * G * T1 * T1
            v2 = np.array([*(0.85 * hv), -0.75 * (v1[2] - ball3d.G_FTS2 * T1)])
            T2 = (1.0 - frac) * dist / (0.85 * speed)
            z_catch = float(pb[2] + v2[2] * T2 - 0.5 * ball3d.G_FTS2 * T2 * T2)
            if not 1.5 <= z_catch <= 5.0:
                continue
            legs.append((pb, v2, T2))
        else:
            if kind == "chest":
                z_rel, z_catch = rng.uniform(3.8, 5.2), rng.uniform(3.5, 5.5)
                T = dist / rng.uniform(22.0, 40.0)
            elif kind == "overhead":
                z_rel, z_catch = rng.uniform(6.5, 8.0), rng.uniform(5.0, 7.5)
                T = dist / rng.uniform(20.0, 34.0)
            else:
                z_rel, z_catch = rng.uniform(6.0, 8.0), rng.uniform(6.0, 9.5)
                vz = math.sqrt(2 * ball3d.G_FTS2 * (rng.uniform(11.0, 17.0) - z_rel))
                T = (vz + math.sqrt(max(vz * vz - 2 * ball3d.G_FTS2 * (z_catch - z_rel), 0.0))) / ball3d.G_FTS2
            rel = np.array([*passer, z_rel])
            v = (np.array([*catch, z_catch]) - rel - 0.5 * G * T * T) / T
            if not 8.0 <= math.hypot(v[0], v[1]) <= 50.0:
                continue
            if kind == "overhead" and z_rel + max(v[2], 0.0) ** 2 / (2 * ball3d.G_FTS2) > 11.5:
                continue
            legs.append((rel, v, T))
        crossing = rim_crossing(legs[-1][0], legs[-1][1])
        if crossing is not None and crossing[0] <= legs[-1][2] + 0.05 and \
                min(math.dist(crossing[1][:2], b) for b in BASKETS.values()) < 4.5:
            continue
        break
    else:
        raise RuntimeError("no pass")
    path = Path3D()
    p_rel, v_rel, _ = legs[0]
    holder = np.array([*(passer - _unit(catch - passer) * 0.5), rng.uniform(3.0, 4.5)])
    t = 0.0
    if rng.random() < 0.5:
        period = rng.uniform(0.4, 0.55)
        t = path.add(0.0, period, dribble(holder[:2], np.zeros(2), np.zeros(2), float(holder[2]), period, 0.0))
    else:
        t = path.add(0.0, rng.uniform(0.3, 0.6), hold(holder, np.array([*rng.normal(0.0, 1.0, 2), 0.0]), 0.0))
        holder = path.at(np.array([t - 1e-9]))[0]
    tw, tp = rng.uniform(0.12, 0.25), rng.uniform(0.06, 0.1)
    q = p_rel - 0.5 * v_rel * tp
    t = path.add(t, t + tw - tp, ease(holder, q, t, t + tw - tp))
    t_rel = path.add(t, t + tp, push(q, v_rel, t, tp))
    t = t_rel
    for p, v, T in legs:
        t = path.add(t, t + T, ballistic(p, v, t))
    t_end_flight = t
    p_c, v_c, T_c = legs[-1]
    end = p_c + v_c * T_c + 0.5 * G * T_c * T_c
    settle = end + np.array([*(0.03 * v_c[:2]), -0.3])
    t = path.add(t, t + 0.2, ease(end, settle, t, t + 0.2))
    path.add(t, t + rng.uniform(0.2, 0.5), hold(settle, np.array([*rng.normal(0.0, 1.0, 2), 0.0]), t))
    return path, Truth("pass", t_rel, p_rel, t_end_flight=t_end_flight, cat=kind), {}


# --------------------------------------------------------------------------- camera and detections


def _ar1(rng: np.random.Generator, n: int, rho: float, sd: float) -> np.ndarray:
    e = np.empty(n)
    e[0] = rng.normal(0.0, sd)
    k = math.sqrt(1.0 - rho * rho) * sd
    for i in range(1, n):
        e[i] = rho * e[i - 1] + rng.normal(0.0, k)
    return e


@dataclass(frozen=True)
class Noise:
    """The observation model (defaults: the assumptions of the module docstring); ``--noise key=value,...``."""

    px: float = 2.0  # detection noise per axis (1080-line px)
    outlier_start: float = 0.035  # an outlier run starts on this share of frames (runs of mean 2 frames)
    drop: float = 0.03  # random dropouts
    occlusion_start: float = 0.006  # occlusions of 2-6 frames start on this share of frames
    release_loss: float = 0.85  # share of shots that lose the ball at the release (passes: 0.5 when > 0)
    rim_loss: float = 0.85  # share of shots that lose the ball at the rim (passes: 0.5 at the catch when > 0)
    cam: float = 1.0  # scale of the camera parameter errors (1: about 0.04 deg pan / tilt, 0.3% focal length)
    cam_rho: float = 0.9  # frame-to-frame correlation of the slow part of the camera errors (0: white)


CLEAN = Noise(px=0.0, outlier_start=0.0, drop=0.0, occlusion_start=0.0, release_loss=0.0, rim_loss=0.0, cam=0.0)


def film(path: Path3D, truth: Truth, rngs: list[np.random.Generator], noise: Noise | None = None) -> dict[str, Any]:
    """The broadcast camera following the ball, the detector's observations and the pipeline's cameras. One random
    stream per part of the model (operator, camera errors, dropouts, outliers, pixel noise), so switching one part
    off leaves the others' draws as they were."""
    noise = noise or Noise()
    rng, rng_cam, rng_drop, rng_out, rng_px = rngs
    n = math.floor(path.end * FPS)
    tk = np.arange(n) / FPS
    B = path.at(tk)
    C = np.array([47.0 + rng.uniform(-1.0, 1.0), rng.uniform(-95.0, -65.0), rng.uniform(26.0, 40.0)])
    tau = rng.uniform(0.2, 0.45)
    alpha = 1.0 - math.exp(-1.0 / (FPS * tau))
    kappa = rng.uniform(0.35, 0.8)
    f0, zr = math.exp(rng.uniform(math.log(2000.0), math.log(5000.0))), rng.uniform(-0.2, 0.2)
    off_p, off_t = rng.uniform(-0.1, 0.1), rng.uniform(-0.08, 0.08)
    roll = rng.uniform(-0.3, 0.3)
    A = B[0].copy()
    true_cams, pans = [], []
    for k in range(n):
        target = B[k].copy()
        target[2] = 4.0 + kappa * (target[2] - 4.0)
        A = A + alpha * (target - A)
        d = A - C
        f = float(np.clip(f0 * math.exp(zr * tk[k]), 2000.0, 6000.0))
        hfov = 2 * math.degrees(math.atan(SIZE[0] / 2 / f))
        vfov = 2 * math.degrees(math.atan(SIZE[1] / 2 / f))
        pan = math.degrees(math.atan2(d[0], d[1])) + off_p * hfov
        if pans:
            pan = pans[-1] + float(np.clip(pan - pans[-1], -3.0, 3.0))
        pans.append(pan)
        tilt = math.degrees(math.atan2(-d[2], math.hypot(d[0], d[1]))) + off_t * vfov
        true_cams.append(CameraParams(C, pan, tilt, roll, f))
    s, rho = noise.cam, noise.cam_rho
    e_pan = s * (_ar1(rng_cam, n, rho, 0.03) + rng_cam.normal(0.0, 0.025, n))
    e_tilt = s * (_ar1(rng_cam, n, rho, 0.03) + rng_cam.normal(0.0, 0.025, n))
    e_roll = s * (_ar1(rng_cam, n, rho, 0.02) + rng_cam.normal(0.0, 0.02, n))
    e_f = s * (_ar1(rng_cam, n, rho, 0.0025) + rng_cam.normal(0.0, 0.0015, n))
    est_cams = [CameraParams(C, c.pan + e_pan[k], c.tilt + e_tilt[k], c.roll + e_roll[k], c.f * math.exp(e_f[k]))
                for k, c in enumerate(true_cams)]
    uv_true = np.array([ball3d._project(B[k], true_cams[k], SIZE) for k in range(n)])
    in_view = (uv_true[:, 0] >= 10) & (uv_true[:, 0] <= SIZE[0] - 10) & (uv_true[:, 1] >= 10) & \
        (uv_true[:, 1] <= SIZE[1] - 10)
    for k in range(n):  # in front of the camera
        pc = true_cams[k].R() @ (B[k] - C)
        in_view[k] &= pc[2] > 1.0
    drop = rng_drop.random(n) < noise.drop
    occ_u, occ_len = rng_drop.random(n), rng_drop.integers(2, 7, n)
    k = 0
    while k < n:
        if occ_u[k] < noise.occlusion_start:
            drop[k:k + occ_len[k]] = True
            k += int(occ_len[k])
        k += 1

    special = np.zeros(n, bool)  # the release / rim / catch windows

    def lose(t_mid: float, before: int, after: int) -> None:
        k0 = math.ceil(t_mid * FPS - 1e-9)
        drop[max(0, k0 - before):max(0, k0 + after + 1)] = True
        special[max(0, k0 - before):max(0, k0 + after + 1)] = True

    u = rng_drop.random(4)
    w = rng_drop.integers(0, 3), rng_drop.integers(1, 5), rng_drop.integers(0, 5), rng_drop.integers(1, 9)
    if truth.kind == "shot":
        if u[0] < noise.release_loss:
            lose(truth.t_rel, int(w[0]), int(w[1]))
        if u[1] < noise.rim_loss:
            lose(truth.t_rim, int(w[2]), int(w[3]))
    else:
        if noise.release_loss > 0 and u[2] < 0.5:
            lose(truth.t_rel, int(w[0]) // 2, int(w[1]) // 2)
        if noise.rim_loss > 0 and u[3] < 0.5:
            lose(truth.t_end_flight, int(w[2]) // 2, int(w[3]) // 4)
    outlier = np.zeros(n, bool)
    out_off = np.zeros((n, 2))
    out_u, out_more = rng_out.random(n), rng_out.random((n, 8))
    out_mag, out_ang, out_jit = rng_out.uniform(15.0, 80.0, n), rng_out.uniform(0.0, 2 * math.pi, n), \
        rng_out.normal(0.0, 3.0, (n, 2))
    k = 0
    while k < n:
        if out_u[k] < noise.outlier_start:
            L = 1 + int(np.argmax(out_more[k] >= 0.5)) if (out_more[k] >= 0.5).any() else 9
            for j in range(k, min(n, k + L)):
                outlier[j] = True
                out_off[j] = out_mag[k] * np.array([math.cos(out_ang[k]), math.sin(out_ang[k])]) + out_jit[j]
            k += L
        k += 1
    px = noise.px * rng_px.normal(0.0, 1.0, (n, 2))
    obs = in_view & (outlier | ~drop)
    uv = np.where(outlier[:, None], uv_true + out_off, uv_true + px)
    idx = np.flatnonzero(obs)
    good = np.zeros(n, bool)
    good[idx] = np.linalg.norm(uv[idx] - uv_true[idx], axis=1) <= 15.0
    ordinary = in_view & ~special
    return {"frames": idx, "t": tk[idx], "uv": uv[idx], "cams": [est_cams[i] for i in idx],
            "stats": {"frames": int(in_view.sum()), "within15": int(good.sum()),
                      "outliers": int((outlier & in_view).sum()), "observed": len(idx),
                      "ordinary": int(ordinary.sum()), "ordinary_within15": int((good & ordinary).sum()),
                      "frames_total": n, "max_pan_step": float(np.max(np.abs(np.diff(pans)))) if n > 1 else 0.0},
            "f0": f0, "C": C}


# --------------------------------------------------------------------------- the pipeline under test


@dataclass
class Variant:
    """One way of running the pipeline: the flight search settings and the release point used for 2 / 3."""

    name: str
    config: Callable[[], ball3d.FlightSearchConfig]
    release: Callable[[ball3d.Flight3D], tuple[float, float]]
    find_kwargs: dict[str, Any] = field(default_factory=dict)


def _p0(f: ball3d.Flight3D) -> tuple[float, float]:
    return float(f.p0[0]), float(f.p0[1])


def _release(f: ball3d.Flight3D, z: float = 8.5) -> tuple[float, float]:
    return ball3d.release_point(f, z_release=z)


def _release_sym(f: ball3d.Flight3D, z: float = 8.5) -> tuple[float, float]:
    """Where the fitted flight rises through ``z``, before or after its first observation."""
    z0, vz = float(f.p0[2]), float(f.v0[2])
    disc = vz * vz + 2.0 * ball3d.G_FTS2 * (z0 - z)
    if disc < 0 or vz <= 0 and z0 < z:
        return float(f.p0[0]), float(f.p0[1])
    tau = (vz - math.sqrt(disc)) / ball3d.G_FTS2
    X = f.at(f.t0 + tau)
    return float(X[0]), float(X[1])


def _baseline_config() -> ball3d.FlightSearchConfig:
    """The flight search as it was before this benchmark (the original defaults and initial solution)."""
    return ball3d.FlightSearchConfig(min_obs=8, seed_obs=(10, 7), max_rms_px=3.0, max_point_px=7.0, max_skip=2,
                                     max_gap_s=0.25, max_speed_fts=65.0, min_speed_fts=6.0, z_range=(-1.0, 40.0),
                                     xy_margin_ft=12.0, init="linear", grow_physical=True)


VARIANTS: dict[str, Variant] = {
    "baseline": Variant("baseline", _baseline_config, _p0),
    "defaults": Variant("defaults", ball3d.FlightSearchConfig, _p0),
    "recommended": Variant("recommended", ball3d.FlightSearchConfig, ball3d.release_point, {"baskets": BASKETS}),
}
RELEASES: dict[str, Callable[[ball3d.Flight3D], tuple[float, float]]] = {"p0": _p0, "sym": _release_sym}


@functools.cache
def resolve(spec: str) -> Variant:
    """A registered variant, or one changed by ``+`` parts: ``field=value`` of ``FlightSearchConfig``; ``rim`` (pass
    the baskets to ``find_flights``) or ``norim``; ``release`` (``ball3d.release_point``), ``release=Z`` (its
    ``z_release``) or ``release=p0``; any other ``key=value`` is a keyword argument of ``find_flights``. E.g.
    ``baseline+max_rms_px=5+rim+release``."""
    base, *mods = spec.split("+")
    v = VARIANTS[base]
    over: dict[str, Any] = {}
    kw, release = dict(v.find_kwargs), v.release
    fields = {f.name for f in dataclasses.fields(ball3d.FlightSearchConfig)}
    for m in mods:
        key, _, val = m.partition("=")
        if key == "rim":
            kw["baskets"] = BASKETS
        elif key == "norim":
            kw.pop("baskets", None)
        elif key == "release":
            if val in RELEASES:
                release = RELEASES[val]
            else:
                z = float(val) if val else 8.5
                release = functools.partial(_release, z=z)
        elif key in fields:
            over[key] = ast.literal_eval(val)
        else:
            kw[key] = ast.literal_eval(val)
    base_cfg = v.config
    return Variant(spec, (lambda: dataclasses.replace(base_cfg(), **over)) if over else base_cfg, release, kw)


def run_flights(obs: dict[str, Any], variant: Variant) -> list[ball3d.Flight3D]:
    """``analyze.build_ball_track`` after the linker: runs of the path, ``find_flights`` on each."""
    frames = obs["frames"]
    runs: list[list[int]] = []
    for i, fr in enumerate(frames):
        if runs and fr - frames[runs[-1][-1]] <= LINK_MAX_GAP:
            runs[-1].append(i)
        else:
            runs.append([i])
    out: list[ball3d.Flight3D] = []
    for run in runs:
        if len(run) < 6:
            continue
        out += ball3d.find_flights(obs["t"][run], obs["uv"][run], [obs["cams"][i] for i in run], SIZE,
                                   frames=frames[run], config=variant.config(), **variant.find_kwargs)
    return out


def purity(f: ball3d.Flight3D, spans: list[tuple[float, float]]) -> float:
    """Share of the flight's time span inside the one free-flight piece it overlaps most."""
    best = max((max(0.0, min(f.t1, b) - max(f.t0, a)) for a, b in spans), default=0.0)
    return best / max(f.t1 - f.t0, 1e-6)


def evaluate(truth: Truth, flights: list[ball3d.Flight3D], variant: Variant) -> dict[str, Any]:
    impure = [f for f in flights if purity(f, truth.spans) < 0.8]
    passes = []
    for f in flights:
        rp = ball3d.rim_pass(f, BASKETS, COURT.rim_height, COURT.rim_radius)
        if rp is not None and rp.dist_ft <= RIM_GATE_FT:
            passes.append((rp, f))
    base = {"n_flights": len(flights), "n_impure": len(impure),
            "impure_rim": sum(1 for _, f in passes if purity(f, truth.spans) < 0.8)}
    if truth.kind == "pass":
        return {"fp": bool(passes), **base}
    r: dict[str, Any] = {"hit": any(abs(rp.t - truth.t_rim) <= MATCH_S for rp, _ in passes), **base}
    near = [(rp, f) for rp, f in passes if abs(rp.t - truth.t_rim) <= MATCH_S]
    if not near:
        r["matched"] = False
        return r
    rp, f = min(near, key=lambda p: abs(p[0].t - truth.t_rim))
    x, y = variant.release(f)
    bx, by = BASKETS[rp.side]
    d_used = math.hypot(x - bx, y - by)
    d_true = math.hypot(truth.rel[0] - bx, truth.rel[1] - by)
    overlap = max(0.0, min(f.t1, truth.t_end_flight) - max(f.t0, truth.t_rel)) / max(f.t1 - f.t0, 1e-6)
    los = _unit(truth.rel[:2] - truth.cam_c[:2])
    e = np.array([x, y]) - truth.rel[:2]
    r.update({"matched": True, "pred": fusion_value(x, y, rp.side), "side_ok": rp.side == truth.side,
              "pred_m": {str(mg): margin_value(x, y, rp.side, mg) for mg in MARGINS},
              "err": math.hypot(x - truth.rel[0], y - truth.rel[1]), "radial": d_used - d_true,
              "err_los": float(e @ los), "err_cross": float(e @ _perp(los)),
              "shot_flight": overlap >= 0.6, "t0_delay": f.t0 - truth.t_rel, "rim_dist": rp.dist_ft,
              "rim_dt": rp.t - truth.t_rim, "n_obs": f.n_obs, "dur": f.t1 - f.t0})
    return r


# --------------------------------------------------------------------------- driver


def run_sample(args: tuple[int, str, int, list[str], Noise]) -> dict[str, Any]:
    seed, kind, i, names, noise = args
    streams = np.random.SeedSequence([seed, 0 if kind == "shot" else 1, i]).spawn(6)
    world, *rngs = [np.random.default_rng(s) for s in streams]
    path, truth, _ = simulate_shot(world) if kind == "shot" else simulate_pass(world)
    truth.spans = path.ballistic_spans()
    obs = film(path, truth, rngs, noise)
    truth.cam_c = obs["C"]
    out: dict[str, Any] = {"kind": kind, "stats": obs["stats"], "f0": obs["f0"]}
    if kind == "shot":
        out.update({"value": truth.value, "margin": truth.margin, "cat": truth.cat, "made": truth.made,
                    "oracle": fusion_value(float(truth.rel[0]), float(truth.rel[1]), truth.side),
                    "oracle_m": {str(mg): margin_value(float(truth.rel[0]), float(truth.rel[1]), truth.side, mg)
                                 for mg in MARGINS}})
    else:
        out["cat"] = truth.cat
    for name in names:
        t0 = time.perf_counter()
        variant = resolve(name)
        flights = run_flights(obs, variant)
        out[name] = evaluate(truth, flights, variant)
        out[name]["sec"] = time.perf_counter() - t0
    return out


def _pct(v: list[float], q: float) -> float:
    return float(np.percentile(v, q)) if v else float("nan")


def _share(v: list[bool]) -> float:
    return float(np.mean(v)) if v else float("nan")


def summarize(results: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    shots = [r for r in results if r["kind"] == "shot"]
    passes = [r for r in results if r["kind"] == "pass"]
    st = {k: sum(r["stats"][k] for r in results) for k in ("frames", "within15", "outliers", "observed", "ordinary",
                                                             "ordinary_within15", "frames_total")}
    near = [r for r in shots if abs(r["margin"]) <= 3.0]
    rep: dict[str, Any] = {
        "n_shots": len(shots), "n_passes": len(passes),
        "share_threes": float(np.mean([r["value"] == 3 for r in shots])),
        "detections": {"within_15px": st["within15"] / st["frames"], "outlier": st["outliers"] / st["frames"],
                       "observed": st["observed"] / st["frames"],
                       "within_15px_ordinary": st["ordinary_within15"] / st["ordinary"],
                       "in_view": st["frames"] / st["frames_total"],
                       "max_pan_step_deg": max(r["stats"]["max_pan_step"] for r in results)},
        "oracle_accuracy": float(np.mean([r["oracle"] == r["value"] for r in shots])),
        "oracle_accuracy_near_arc": float(np.mean([r["oracle"] == r["value"] for r in near])), "n_near_arc": len(near),
        "oracle_by_margin": {mg: (float(np.mean([r["oracle_m"][mg] == r["value"] for r in shots])),
                                  float(np.mean([r["oracle_m"][mg] == r["value"] for r in near])))
                             for mg in (str(x) for x in MARGINS)},
        "variants": {}}
    for name in names:
        m = [r for r in shots if r[name]["matched"]]
        mn = [r for r in m if abs(r["margin"]) <= 3.0]
        miss = [r for r in shots if not r["made"]]
        rep["variants"][name] = {
            "recall": float(np.mean([r[name]["hit"] for r in shots])),
            "recall_missed": float(np.mean([r[name]["hit"] for r in miss])),
            "matched": len(m) / len(shots),
            "matched_shot_flight": float(np.mean([r[name]["shot_flight"] for r in m])) if m else float("nan"),
            "false_pos": float(np.mean([r[name]["fp"] for r in passes])) if passes else float("nan"),
            "err_median_ft": _pct([r[name]["err"] for r in m], 50), "err_p90_ft": _pct([r[name]["err"] for r in m], 90),
            "radial_median_ft": _pct([r[name]["radial"] for r in m], 50),
            "radial_abs_p90_ft": _pct([abs(r[name]["radial"]) for r in m], 90),
            "accuracy": float(np.mean([r[name]["pred"] == r["value"] for r in m])) if m else float("nan"),
            "accuracy_near_arc": float(np.mean([r[name]["pred"] == r["value"] for r in mn])) if mn else float("nan"),
            "n_matched_near_arc": len(mn),
            "radial_abs_median_ft": _pct([abs(r[name]["radial"]) for r in m], 50),
            "accuracy_by_margin": {mg: (_share([r[name]["pred_m"][mg] == r["value"] for r in m]),
                                        _share([r[name]["pred_m"][mg] == r["value"] for r in mn]))
                                   for mg in (str(x) for x in MARGINS)},
            "err_los_abs_median_ft": _pct([abs(r[name]["err_los"]) for r in m], 50),
            "err_cross_abs_median_ft": _pct([abs(r[name]["err_cross"]) for r in m], 50),
            "starts_before_release": float(np.mean([r[name]["t0_delay"] < -0.05 for r in m])) if m else float("nan"),
            "first_obs_delay_median_s": _pct([r[name]["t0_delay"] for r in m], 50),
            "threes_called_two": int(sum(r["value"] == 3 and r[name]["pred"] == 2 for r in m)),
            "twos_called_three": int(sum(r["value"] == 2 and r[name]["pred"] == 3 for r in m)),
            "impure_per_shot": float(np.mean([r[name]["n_impure"] for r in shots])),
            "impure_per_pass": float(np.mean([r[name]["n_impure"] for r in passes])) if passes else float("nan"),
            "impure_rim_per_clip": float(np.mean([r[name]["impure_rim"] for r in results])),
            "flights_per_clip": float(np.mean([r[name]["n_flights"] for r in results])),
            "sec_per_clip": float(np.mean([r[name]["sec"] for r in results])),
        }
    return rep


def print_report(rep: dict[str, Any], names: list[str]) -> None:
    d = rep["detections"]
    print(f"{rep['n_shots']} shots ({rep['share_threes']:.0%} threes, {rep['n_near_arc']} within 3 ft of the line), "
          f"{rep['n_passes']} passes")
    print(f"ball in view on {d['in_view']:.1%} of frames; a detection within 15 px on {d['within_15px']:.1%} of "
          f"them ({d['within_15px_ordinary']:.1%} outside the release / rim / catch windows), outliers "
          f"{d['outlier']:.1%}; largest pan step {d['max_pan_step_deg']:.2f} deg/frame")
    print(f"oracle (true release point, same rule): accuracy {rep['oracle_accuracy']:.3f}, "
          f"within 3 ft of the line {rep['oracle_accuracy_near_arc']:.3f}")
    rows = [("recall (rim pass <= 3 ft, <= 1.5 s)", "recall", "{:.3f}"),
            ("  recall, missed shots", "recall_missed", "{:.3f}"),
            ("shots given a value (matched)", "matched", "{:.3f}"),
            ("  matched flight is the shot's", "matched_shot_flight", "{:.3f}"),
            ("false positives (passes)", "false_pos", "{:.3f}"),
            ("release point error median (ft)", "err_median_ft", "{:.2f}"),
            ("release point error p90 (ft)", "err_p90_ft", "{:.2f}"),
            ("distance-to-basket error median (ft)", "radial_median_ft", "{:+.2f}"),
            ("|distance-to-basket error| median (ft)", "radial_abs_median_ft", "{:.2f}"),
            ("|distance-to-basket error| p90 (ft)", "radial_abs_p90_ft", "{:.2f}"),
            ("  |error| along the line of sight, median", "err_los_abs_median_ft", "{:.2f}"),
            ("  |error| across the line of sight, median", "err_cross_abs_median_ft", "{:.2f}"),
            ("  matched flight starts > 0.05 s before release", "starts_before_release", "{:.3f}"),
            ("  first observation after release, median (s)", "first_obs_delay_median_s", "{:+.3f}"),
            ("2/3 accuracy (matched)", "accuracy", "{:.3f}"),
            ("2/3 accuracy within 3 ft of the line", "accuracy_near_arc", "{:.3f}"),
            ("  threes called two", "threes_called_two", "{:d}"),
            ("  twos called three", "twos_called_three", "{:d}"),
            ("flights per clip", "flights_per_clip", "{:.2f}"),
            ("impure flights per shot clip", "impure_per_shot", "{:.3f}"),
            ("impure flights per pass clip", "impure_per_pass", "{:.3f}"),
            ("impure flights with a rim pass per clip", "impure_rim_per_clip", "{:.3f}"),
            ("seconds per clip", "sec_per_clip", "{:.3f}")]
    labels = [n if len(n) <= 12 else f"v{i}" for i, n in enumerate(names)]
    for lab, n in zip(labels, names, strict=True):
        if lab != n:
            print(f"{lab} = {n}")
    w = max(len(r[0]) for r in rows)
    print(" " * w + "".join(f"{lab:>14}" for lab in labels))
    for label, key, fmt in rows:
        print(label.ljust(w) + "".join(f"{fmt.format(rep['variants'][n][key]):>14}" for n in names))
    print("2/3 accuracy (all / within 3 ft of the line) with the call at a distance beyond the line of at least:")
    for mg in (str(x) for x in MARGINS):
        o = rep["oracle_by_margin"][mg]
        print(f"  {float(mg):+.1f} ft".ljust(w) + "".join(
            "{:>7.3f}/{:.3f}".format(*rep["variants"][n]["accuracy_by_margin"][mg]).rjust(14) for n in names)
            + f"   oracle {o[0]:.3f}/{o[1]:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, default=300)
    ap.add_argument("--passes", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--variants", default="baseline,defaults,recommended",
                    help=f"comma-separated: {', '.join(VARIANTS)}, each with optional +changes (module docstring)")
    ap.add_argument("--noise", default="", help="observation model changes, e.g. 'cam=0,px=1' ('clean': none at all)")
    ap.add_argument("--json", default=None, help="write the summary (and per-clip results) here")
    args = ap.parse_args()
    names, depth, cur = [], 0, ""
    for ch in args.variants + ",":  # commas inside parentheses belong to a tuple value
        depth += (ch == "(") - (ch == ")")
        if ch == "," and depth == 0:
            if cur:
                names.append(cur)
            cur = ""
        else:
            cur += ch
    for n in names:
        try:
            resolve(n)
        except (KeyError, ValueError, SyntaxError) as e:
            ap.error(f"bad variant {n!r}: {e}")
    noise = Noise()
    for item in (x for x in args.noise.split(",") if x):
        if item == "clean":
            noise = CLEAN
            continue
        key, _, val = item.partition("=")
        if key not in Noise.__dataclass_fields__:
            ap.error(f"unknown noise field {key!r}")
        noise = dataclasses.replace(noise, **{key: float(val)})
    jobs = [(args.seed, "shot", i, names, noise) for i in range(args.shots)] + \
        [(args.seed, "pass", i, names, noise) for i in range(args.passes)]
    t0 = time.time()
    if args.workers > 1:
        with cf.ProcessPoolExecutor(args.workers) as ex:
            results = list(ex.map(run_sample, jobs, chunksize=8))
    else:
        results = [run_sample(j) for j in jobs]
    rep = summarize(results, names)
    print_report(rep, names)
    print(f"({time.time() - t0:.0f} s)")
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": rep, "clips": results}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
