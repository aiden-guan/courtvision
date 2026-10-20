"""NoiseModel: observed copies of truth frames."""

from __future__ import annotations

import math

import numpy as np
import pytest

from courtvision.sim.noise import NoiseConfig, NoiseModel
from courtvision.sim.simulator import GameSimulator, SimConfig, SimResult


@pytest.fixture(scope="module")
def sim() -> SimResult:
    return GameSimulator(SimConfig(seconds=60.0, fps=30.0, seed=9)).run()


def _held(f, radius: float = 2.5) -> bool:
    b = f.ball
    return b.z <= 6.0 and any(math.hypot(p.x - b.court_xy[0], p.y - b.court_xy[1]) <= radius for p in f.players)


def test_deterministic_and_input_untouched(sim: SimResult) -> None:
    before = [(p.x, p.y, p.bbox) for p in sim.frames[5].players]
    a = NoiseModel(camera=sim.camera).apply(sim.frames, seed=1)
    b = NoiseModel(camera=sim.camera).apply(sim.frames, seed=1)
    c = NoiseModel(camera=sim.camera).apply(sim.frames, seed=2)
    assert [(p.x, p.y) for f in a for p in f.players] == [(p.x, p.y) for f in b for p in f.players]
    assert [(p.x, p.y) for f in a for p in f.players] != [(p.x, p.y) for f in c for p in f.players]
    assert [(p.x, p.y, p.bbox) for p in sim.frames[5].players] == before
    assert len(a) == len(sim.frames)
    assert all(o.scoreboard == t.scoreboard and o.frame_idx == t.frame_idx for o, t in zip(a, sim.frames, strict=True))


def test_player_noise_levels(sim: SimResult) -> None:
    cfg = NoiseConfig()
    obs = NoiseModel(cfg, camera=sim.camera).apply(sim.frames, seed=3)
    dx, dbox, n_truth, n_obs = [], [], 0, 0
    for o, t in zip(obs, sim.frames, strict=True):
        truth = {p.player_id: p for p in t.players}
        n_truth += len(t.players)
        n_obs += len(o.players)
        for p in o.players:
            q = truth[p.player_id]
            dx.append(p.x - q.x)
            dbox.append(p.bbox.x1 - q.bbox.x1)
    assert abs(np.std(dx) - cfg.pos_sigma_ft) < 0.05
    assert abs(np.std(dbox) - cfg.bbox_sigma_px) < 0.2
    assert abs((1 - n_obs / n_truth) - cfg.drop_player_rate) < 0.01


def test_ball_dropout_rates(sim: SimResult) -> None:
    cfg = NoiseConfig(ball_fp_rate=0.0)
    obs = NoiseModel(cfg, camera=sim.camera).apply(sim.frames, seed=4)
    held = [o.ball.visible for o, t in zip(obs, sim.frames, strict=True) if t.ball.visible and _held(t)]
    flight = [o.ball.visible for o, t in zip(obs, sim.frames, strict=True) if t.ball.visible and not _held(t)]
    assert len(held) > 200 and len(flight) > 100
    assert abs((1 - np.mean(held)) - cfg.ball_miss_rate_held) < 0.05
    assert abs((1 - np.mean(flight)) - cfg.ball_miss_rate) < 0.06


def test_monocular_ball_projection(sim: SimResult) -> None:
    obs = NoiseModel(NoiseConfig(ball_fp_rate=0.0, ball_px_sigma=0.0), camera=sim.camera).apply(sim.frames, seed=5)
    H = sim.camera.floor_homography()
    checked = 0
    for o, t in zip(obs, sim.frames, strict=True):
        if not o.ball.visible:
            continue
        assert o.ball.z is None
        expect = H.image_to_court(np.array(o.ball.image_xy), front_only=True)
        assert np.allclose(o.ball.court_xy, expect)
        # the camera ray through the ball centre meets the floor behind the ball (away from the camera)
        cam = sim.camera.center
        ball = np.array([*t.ball.court_xy, t.ball.z])
        floor = cam + (ball - cam) * (cam[2] / (cam[2] - ball[2]))
        assert np.allclose(o.ball.court_xy, floor[:2], atol=1e-6)
        if t.ball.z > 3.0:
            assert o.ball.court_xy[1] > t.ball.court_xy[1]
            checked += 1
    assert checked > 0


def test_three_d_mode_keeps_height(sim: SimResult) -> None:
    obs = NoiseModel(NoiseConfig(monocular=False, ball_fp_rate=0.0), camera=None).apply(sim.frames, seed=6)
    vis = [o.ball for o in obs if o.ball.visible]
    assert vis and all(b.z is not None and b.z >= 0.0 and b.court_xy is not None for b in vis)


def test_false_positives_and_rims(sim: SimResult) -> None:
    obs = NoiseModel(NoiseConfig(ball_fp_rate=1.0), camera=sim.camera).apply(sim.frames[:50], seed=7)
    w, h = sim.config.image_size
    for o in obs:
        assert o.ball.visible and 0 <= o.ball.image_xy[0] <= w and 0.3 * h <= o.ball.image_xy[1] <= h
        assert o.ball.confidence < 0.7
    obs = NoiseModel(NoiseConfig(rim_px_sigma=0.5), camera=sim.camera).apply(sim.frames[:200], seed=8)
    d = [o.rims[0].bbox.x1 - t.rims[0].bbox.x1 for o, t in zip(obs, sim.frames[:200], strict=True)]
    assert 0.3 < np.std(d) < 0.7


def test_identity_swaps(sim: SimResult) -> None:
    cfg = NoiseConfig(id_swap_rate=0.2, id_swap_frames=10, drop_player_rate=0.0, id_swap_max_ft=8.0)
    obs = NoiseModel(cfg, camera=sim.camera).apply(sim.frames, seed=9)
    swapped = 0
    for o, t in zip(obs, sim.frames, strict=True):
        for p, q in zip(o.players, t.players, strict=True):
            if p.player_id != q.player_id:
                swapped += 1
                assert p.team is q.team
                assert abs(p.x - q.x) < 2.0  # positions stay with the true player; only the identity moves
    assert swapped > 0
    clean = NoiseModel(NoiseConfig(drop_player_rate=0.0), camera=sim.camera).apply(sim.frames, seed=9)
    assert all(p.player_id == q.player_id for o, t in zip(clean, sim.frames, strict=True)
               for p, q in zip(o.players, t.players, strict=True))
