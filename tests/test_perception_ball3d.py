"""3-D ball flights from one broadcast camera (ball3d.py)."""

from __future__ import annotations

import math

import numpy as np

from courtvision.perception.ball3d import (
    G_FTS2,
    Flight3D,
    _project,
    find_flights,
    fit_flight,
    release_point,
    rim_pass,
)
from courtvision.perception.court_search import CameraParams

SIZE = (1920, 1080)
C = np.array([47.0, -94.0, 36.0])
BASKETS = {"left": (5.25, 25.0), "right": (88.75, 25.0)}
G = np.array([0.0, 0.0, -G_FTS2])


def shot(p0: np.ndarray, target: np.ndarray, T: float, noise: float, seed: int = 0):  # noqa: ANN201
    rng = np.random.default_rng(seed)
    v0 = (target - p0 - 0.5 * np.array([0.0, 0.0, -G_FTS2]) * T**2) / T
    ts = np.arange(0.0, T + 1e-9, 1 / 29.97)
    cams, uv = [], []
    for t in ts:
        X = p0 + v0 * t + 0.5 * np.array([0.0, 0.0, -G_FTS2]) * t * t
        cam = CameraParams(C, 15.0 + 2.0 * t, 15.0 - 0.5 * t, 0.1, 4200.0 * (1 + 0.02 * t))  # panning, zooming
        cams.append(cam)
        uv.append(_project(X, cam, SIZE) + rng.normal(0.0, noise, 2))
    return ts, np.array(uv), cams, v0


def observe(p0: np.ndarray, target: np.ndarray, T: float, px: float, seed: int = 0, cam_deg: float = 0.0,
            t_from: float = 0.0):  # noqa: ANN201
    """``shot`` with the pipeline's cameras off by ``cam_deg`` (pan and tilt, per frame) as well as pixel noise
    ``px``, observed from ``t_from`` on (the frames before are lost in the shooter's hands)."""
    rng = np.random.default_rng(seed)
    v0 = (target - p0 - 0.5 * G * T**2) / T
    ts = np.arange(0.0, T + 1e-9, 1 / 29.97)
    out_t, uv, cams = [], [], []
    for t in ts:
        cam = CameraParams(C, 15.0 + 2.0 * t, 15.0 - 0.5 * t, 0.1, 4200.0 * (1 + 0.02 * t))
        e = rng.normal(0.0, 1.0, 4) * [px, px, cam_deg, cam_deg]
        if t < t_from - 1e-9:
            continue
        out_t.append(t)
        uv.append(_project(p0 + v0 * t + 0.5 * G * t * t, cam, SIZE) + e[:2])
        cams.append(CameraParams(C, cam.pan + e[2], cam.tilt + e[3], cam.roll, cam.f))
    return np.array(out_t), np.array(uv), cams, v0


def test_fit_recovers_a_shot_with_a_panning_camera() -> None:
    p0 = np.array([70.0, 22.0, 7.5])
    ts, uv, cams, v0 = shot(p0, np.array([88.75, 25.0, 10.0]), 1.1, noise=1.0)
    f = fit_flight(ts, uv, cams, SIZE)
    assert f is not None
    assert np.linalg.norm(f.p0 - p0) < 1.5
    assert np.linalg.norm(f.v0 - v0) < 1.0
    assert f.rms_px < 2.0
    assert 13.0 < f.apex_z < 14.5


def test_rim_pass_tells_a_make_from_a_miss() -> None:
    p0 = np.array([70.0, 22.0, 7.5])
    ts, uv, cams, _ = shot(p0, np.array([88.75, 25.0, 10.0]), 1.1, noise=0.5)
    made = rim_pass(fit_flight(ts, uv, cams, SIZE), BASKETS)
    assert made is not None and made.side == "right" and made.through
    ts, uv, cams, _ = shot(p0, np.array([88.75, 27.5, 10.0]), 1.1, noise=0.5, seed=1)  # 2.5 ft wide
    missed = rim_pass(fit_flight(ts, uv, cams, SIZE), BASKETS)
    assert missed is not None and missed.side == "right" and not missed.through and missed.dist_ft > 1.5


def test_find_flights_skips_the_dribble_before_the_shot() -> None:
    p0 = np.array([70.0, 22.0, 7.5])
    ts, uv, cams, _ = shot(p0, np.array([88.75, 25.0, 10.0]), 1.1, noise=0.5)
    held_t = np.arange(-0.6, 0.0, 1 / 29.97)
    held = [_project(p0 + np.array([0.0, 0.0, -2.0 + 2.0 * abs(math.sin(9 * t))]), cams[0], SIZE) for t in held_t]
    t_all = np.concatenate([held_t, ts])
    uv_all = np.vstack([held, uv])
    cams_all = [cams[0]] * len(held_t) + cams
    flights = find_flights(t_all, uv_all, cams_all, SIZE)
    assert len(flights) == 1
    assert flights[0].t0 >= -0.05 and flights[0].n_obs >= len(ts) - 3


def test_a_ball_held_still_is_no_flight() -> None:
    cam = CameraParams(C, 10.0, 15.0, 0.0, 3000.0)
    t = np.arange(0.0, 1.0, 1 / 29.97)
    uv = np.array([_project(np.array([60.0, 20.0, 4.0]), cam, SIZE)] * len(t))
    assert find_flights(t, uv, [cam] * len(t), SIZE) == []


def test_depth_sweep_is_unbiased_where_the_linear_solution_shrinks_toward_the_camera() -> None:
    """A short window (12 frames) of noisy pixels: the cross-product equations' least squares lands too near the
    camera (its miss is measured in feet at the ball); the depth that reprojects best does not."""
    p0, target = np.array([70.0, 22.0, 7.5]), np.array([88.75, 25.0, 10.0])
    sweep, linear = [], []
    for seed in range(30):
        ts, uv, cams, v0 = observe(p0, target, 1.1, px=3.0, seed=seed)
        w, mid = slice(10, 22), ts[16]
        depth = np.linalg.norm(p0 + v0 * mid + 0.5 * G * mid**2 - C)
        for out, init in ((sweep, "sweep"), (linear, "linear")):
            f = fit_flight(ts[w], uv[w], cams[w], SIZE, refine=False, init=init)
            out.append(0.0 if f is None else np.linalg.norm(f.at(mid) - C) / depth)
    assert 0.9 < np.median(sweep) < 1.1
    assert np.median(linear) < 0.85


def test_find_flights_follows_a_shot_seen_with_broadcast_noise() -> None:
    """2 px of pixel noise and cameras 0.04 degrees off in pan and tilt: one flight over the shot's frames."""
    p0, target = np.array([70.0, 22.0, 7.5]), np.array([88.75, 25.0, 10.0])
    for seed in (1, 2, 3, 4):
        ts, uv, cams, _ = observe(p0, target, 1.1, px=2.0, seed=seed, cam_deg=0.04)
        flights = find_flights(ts, uv, cams, SIZE)
        assert len(flights) == 1 and flights[0].n_obs >= len(ts) - 3


def test_rim_refit_pins_a_noisy_shot_to_its_rim_and_leaves_passes_alone() -> None:
    p0, target = np.array([70.0, 22.0, 7.5]), np.array([88.75, 25.0, 10.0])
    free, pinned = [], []
    for seed in (1, 2, 3, 4):
        ts, uv, cams, _ = observe(p0, target, 1.1, px=2.0, seed=seed, cam_deg=0.04)
        (f,) = find_flights(ts, uv, cams, SIZE)
        (g,) = find_flights(ts, uv, cams, SIZE, baskets=BASKETS)
        assert g.frames == f.frames and g.n_obs == f.n_obs  # the same observations, refitted
        free.append(rim_pass(f, BASKETS).dist_ft)  # type: ignore[union-attr]
        rp = rim_pass(g, BASKETS)
        assert rp is not None and rp.side == "right"
        pinned.append(rp.dist_ft)
    assert max(pinned) < 0.3 and np.mean(free) > 0.8  # gravity alone leaves the depth loose
    for a, b, T in (((60.0, 30.0, 5.0), (75.0, 18.0, 4.5), 0.6),  # a chest pass
                    ((70.0, 38.0, 7.0), (82.0, 27.0, 9.0), 1.0)):  # a lob caught 7 ft from the rim
        ts, uv, cams, _ = observe(np.array(a), np.array(b), T, px=2.0, seed=0, cam_deg=0.04)
        plain = find_flights(ts, uv, cams, SIZE)
        with_rims = find_flights(ts, uv, cams, SIZE, baskets=BASKETS)
        assert len(plain) == len(with_rims) == 1
        assert np.allclose(plain[0].p0, with_rims[0].p0) and np.allclose(plain[0].v0, with_rims[0].v0)


def test_fit_flight_with_a_rim_prior_comes_down_through_the_rim() -> None:
    p0, target = np.array([70.0, 22.0, 7.5]), np.array([88.75, 25.0, 10.0])
    ts, uv, cams, _ = observe(p0, target, 1.1, px=2.0, seed=3, cam_deg=0.04)
    free = fit_flight(ts, uv, cams, SIZE)
    pinned = fit_flight(ts, uv, cams, SIZE, rim=(88.75, 25.0, 10.0), start=free)
    assert free is not None and pinned is not None
    assert rim_pass(pinned, BASKETS).dist_ft < 0.3 < rim_pass(free, BASKETS).dist_ft  # type: ignore[union-attr]
    assert pinned.rms_px < free.rms_px + 0.5  # still fits the pixels


def test_release_point_follows_the_flight_back_to_release_height() -> None:
    p0, target = np.array([66.0, 21.0, 8.5]), np.array([88.75, 25.0, 10.0])
    ts, uv, cams, _ = observe(p0, target, 1.2, px=0.0, t_from=4 / 29.97)  # the first 4 frames lost in the hands
    f = fit_flight(ts, uv, cams, SIZE)
    assert f is not None
    assert np.linalg.norm(f.p0[:2] - p0[:2]) > 2.0  # the first observation is already well on its way
    assert math.dist(release_point(f), p0[:2]) < 0.05
    assert math.dist(release_point(f, z_release=7.5), p0[:2]) > 0.5  # a lower release lies further back
    low = Flight3D(0.0, 1.0, np.array([66.0, 21.0, 7.0]), f.v0, 1.0, 30)  # seen below the release height
    assert release_point(low) == (66.0, 21.0)


def test_rectify_box_recovers_a_standing_player_from_a_shrunken_centred_box() -> None:
    from courtvision.nba.analyze import rectify_box

    cam = CameraParams(C, 12.0, 15.0, 0.2, 3500.0)
    feet = _project(np.array([70.0, 20.0, 0.0]), cam, SIZE)
    head = _project(np.array([70.0, 20.0, 6.6]), cam, SIZE)
    true = (feet[0] - 30.0, head[1], feet[0] + 30.0, feet[1])
    cx, cy, h = (true[0] + true[2]) / 2, (true[1] + true[3]) / 2, true[3] - true[1]
    small = (cx - 20.0, cy - 0.33 * h, cx + 20.0, cy + 0.33 * h)  # centred but only two thirds the size
    out = rectify_box(small, cam, SIZE)
    assert out is not None
    assert abs(out[3] - true[3]) < 3.0 and abs(out[1] - true[1]) < 3.0
    assert abs((out[2] - out[0]) - 60.0) < 3.0
