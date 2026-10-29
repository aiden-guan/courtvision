"""3-D ball flights from one broadcast camera (ball3d.py)."""

from __future__ import annotations

import math

import numpy as np

from courtvision.perception.ball3d import G_FTS2, _project, find_flights, fit_flight, rim_pass
from courtvision.perception.court_search import CameraParams

SIZE = (1920, 1080)
C = np.array([47.0, -94.0, 36.0])
BASKETS = {"left": (5.25, 25.0), "right": (88.75, 25.0)}


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
