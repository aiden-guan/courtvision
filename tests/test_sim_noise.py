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


# Position / bbox / drop noise on its own: no occlusion drops, no identity swaps, no team-label flips, no
# track fragments (every player keeps his track id and identity) and no phantom people (only the ten on-court
# players, in truth order).
_NO_FRAG = dict(frag_after_drop_1=0.0, frag_per_frame=0.0, frag_hazard=0.0)
_NO_PHANTOMS = dict(dup_hazard=0.0, ref_hazard=0.0, waiting_detect=0.0)
_ONE_PER_PLAYER = dict(**_NO_FRAG, **_NO_PHANTOMS)
_PLAIN = dict(occluded_drop_rate=0.0, occlusion_swap_prob=0.0, team_flip_share=0.0, **_ONE_PER_PLAYER)


def test_player_noise_levels(sim: SimResult) -> None:
    """One error source: the foot point (bbox bottom centre) is off by a slow AR(1) pixel error and the court
    point is its floor projection, so the court error is that box offset through the floor homography (up
    to the box's own edge jitter), grows with depth and lies mostly along the depth direction; pos_sigma_ft is
    its scale at mid-court. (It used to be an isotropic court error independent of depth and of the box.)"""
    cfg = NoiseConfig(**_PLAIN)
    model = NoiseModel(cfg, camera=sim.camera)
    obs = model.apply(sim.frames, seed=3)
    model._P = np.asarray(sim.camera.P, dtype=float)  # (the per-call camera state, for the Jacobian below)
    dtop, n_truth, n_obs = [], 0, 0
    pred, real = [], []
    band: dict[str, list[tuple[float, float]]] = {"near": [], "mid": [], "far": []}
    series: dict[str, list[tuple[int, float]]] = {}
    for o, t in zip(obs, sim.frames, strict=True):
        truth = {p.player_id: p for p in t.players}
        n_truth += len(t.players)
        n_obs += len(o.players)
        for p in o.players:
            q = truth[p.player_id]
            err = (p.x - q.x, p.y - q.y)
            dtop.append(p.bbox.y1 - q.bbox.y1)  # the top edge carries only the box jitter
            off = np.array([(p.bbox.x1 + p.bbox.x2 - q.bbox.x1 - q.bbox.x2) / 2.0, p.bbox.y2 - q.bbox.y2])
            pred.append(model._floor_jacobian(q.x, q.y) @ off)
            real.append(err)
            band["near" if q.y < 15.0 else ("mid" if q.y < 35.0 else "far")].append(err)
            series.setdefault(p.player_id, []).append((o.frame_idx, err[0]))
    assert abs(np.std(dtop) - cfg.bbox_sigma_px) < 0.2
    assert abs((1 - n_obs / n_truth) - cfg.drop_player_rate) < 0.01
    pred_a, real_a = np.array(pred), np.array(real)
    for axis in (0, 1):  # the box offset predicts the court error (independent sources would give ~0)
        assert float(np.corrcoef(pred_a[:, axis], real_a[:, axis])[0, 1]) > 0.7, axis
    mid = np.array(band["mid"])
    assert abs(math.sqrt(float(np.std(mid[:, 0]) * np.std(mid[:, 1]))) - cfg.pos_sigma_ft) < 0.08
    assert np.std(mid[:, 1]) > 1.5 * np.std(mid[:, 0])  # mostly along the depth direction
    med = {k: float(np.median(np.hypot(*np.array(v).T))) for k, v in band.items() if len(v) > 100}
    assert med["far"] > 1.2 * med["near"], med  # the floor scale grows with depth
    # AR(1): consecutive errors are correlated (pos_rho), so they do not average out frame to frame
    a = [(e0, e1) for v in series.values() for (i0, e0), (i1, e1) in zip(v, v[1:]) if i1 == i0 + 1]
    rho = float(np.corrcoef(np.array(a).T)[0, 1])
    assert abs(rho - cfg.pos_rho) < 0.05, rho


def test_occluded_players_drop_in_clusters(sim: SimResult) -> None:
    cfg = NoiseConfig(drop_player_rate=0.0, occluded_drop_rate=0.5, occlusion_swap_prob=0.0, team_flip_share=0.0,
                      **_ONE_PER_PLAYER)
    model = NoiseModel(cfg, camera=sim.camera)
    obs = model.apply(sim.frames, seed=3)
    hidden = [0, 0]
    free = [0, 0]
    for o, t in zip(obs, sim.frames, strict=True):
        seen = {p.player_id for p in o.players}
        for p, cover in zip(t.players, model._covered(t.players), strict=True):
            row = hidden if cover >= cfg.occluded_cover else free
            row[0] += 1
            row[1] += p.player_id not in seen
    assert hidden[0] > 100
    assert abs(hidden[1] / hidden[0] - cfg.occluded_drop_rate) < 0.08
    assert free[1] == 0  # only bodies behind a nearer body drop


def test_velocities_come_from_noisy_positions(sim: SimResult) -> None:
    """Mono must not pass the simulator's velocities through: they are alpha-beta estimates from the
    observed (noisy) positions, the same smoothing the video pipeline uses."""
    cfg = NoiseConfig(**_PLAIN, drop_player_rate=0.0)
    obs = NoiseModel(cfg, camera=sim.camera).apply(sim.frames, seed=11)
    err, exact = [], 0
    state: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for o, t in zip(obs, sim.frames, strict=True):
        truth = {p.player_id: p for p in t.players}
        for p in o.players:
            q = truth[p.player_id]
            err.append(math.hypot(p.vx - q.vx, p.vy - q.vy))
            z = np.array([p.x, p.y])
            if p.player_id not in state:  # independent re-computation of the alpha-beta filter
                state[p.player_id] = (z, np.zeros(2), o.t)
                v = np.zeros(2)
            else:
                xf, vf, t0 = state[p.player_id]
                dt = o.t - t0
                pred = xf + vf * dt
                r = z - pred
                if math.hypot(*r) > cfg.vel_reset_ft:
                    xf, v = z, np.zeros(2)
                else:
                    xf, v = pred + cfg.vel_alpha * r, vf + cfg.vel_beta / dt * r
                sp = math.hypot(*v)
                if sp > cfg.vel_max_fts:
                    v = v * cfg.vel_max_fts / sp
                state[p.player_id] = (xf, v, o.t)
            exact += math.isclose(p.vx, v[0], abs_tol=1e-9) and math.isclose(p.vy, v[1], abs_tol=1e-9)
    n = len(err)
    assert exact == n
    assert sum(e < 1e-6 for e in err) < 0.05 * n  # never the truth velocity (except standing-start zeros)
    assert 0.5 < float(np.median(err)) < 5.0  # error consistent with 0.4 ft AR(1) position noise at 30 fps


def test_jumping_players_foot_point_is_lifted(sim: SimResult) -> None:
    """cv_pipeline §11 #8: a jump lifts the foot point, so its floor projection lands behind the player."""
    cfg = NoiseConfig(**_PLAIN, drop_player_rate=0.0, pos_sigma_ft=0.0)
    assert sim.feet_z and len(sim.feet_z) == len(sim.frames)
    plain = NoiseModel(cfg, camera=sim.camera).apply(sim.frames, seed=1)
    lifted = NoiseModel(cfg, camera=sim.camera).apply(sim.frames, seed=1, feet_z=sim.feet_z)
    cam = sim.camera.center
    checked = 0
    for i, (a, b, t) in enumerate(zip(plain, lifted, sim.frames, strict=True)):
        for k, (pa, pb, q) in enumerate(zip(a.players, b.players, t.players, strict=True)):
            h = sim.feet_z[i][k]
            assert pa.x == q.x and pa.y == q.y  # no feet_z: the floor point
            if h <= 0.0:
                assert (pb.x, pb.y) == (q.x, q.y)
                continue
            foot = np.array([q.x, q.y, h])
            floor = cam + (foot - cam) * (cam[2] / (cam[2] - h))  # the camera ray through the lifted foot
            assert np.allclose((pb.x, pb.y), floor[:2], atol=1e-4)
            assert math.hypot(*(floor[:2] - cam[:2])) > math.hypot(q.x - cam[0], q.y - cam[1])  # behind him
            checked += h > 1.0
    assert checked > 10


def test_ball_dropouts_are_bursty_and_occlusion_driven(sim: SimResult) -> None:
    cfg = NoiseConfig(ball_fp_rate=0.0)
    model = NoiseModel(cfg, camera=sim.camera)
    obs = model.apply(sim.frames, seed=4)
    held, flight, behind, clear = [], [], [], []
    runs, cur = [], 0
    for o, t in zip(obs, sim.frames, strict=True):
        if not t.ball.visible:
            continue
        seen = o.ball.visible
        (held if _held(t) else flight).append(seen)
        (behind if model._behind_body(t) else clear).append(seen)
        if not seen:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    assert len(held) > 200 and len(flight) > 100 and len(behind) > 50
    assert 0.25 < 1 - np.mean(held) < 0.5  # ~0.37 on train seeds: body + hands + detector
    assert 0.06 < 1 - np.mean(flight) < 0.2  # ~0.13
    assert (1 - np.mean(behind)) > 0.6 > (1 - np.mean(clear))  # a nearer body hides the ball
    # Gilbert-Elliott bursts: i.i.d. misses at these rates would give mean runs of ~1.2-1.5 frames
    assert np.mean(runs) > 2.0 and max(runs) >= 8


def test_a_ball_hidden_behind_a_body_stays_hidden() -> None:
    """Occlusion is a state, as in video (RESEARCH §2.1, cv_pipeline §6.2): while the ball's image point is
    behind a nearer upper body for >= 0.5 s it is detected on only a few percent of those frames, and many
    such episodes have no detection at all. (At the old i.i.d. 0.75 per-frame miss a hidden ball was still
    seen on ~24% of its frames and no >= 0.5 s episode stayed fully hidden.)"""
    r = GameSimulator(SimConfig(seconds=150.0, fps=30.0, seed=1)).run()
    model = NoiseModel(NoiseConfig(ball_fp_rate=0.0), camera=r.camera)
    obs = model.apply(r.frames, seed=4, feet_z=r.feet_z)
    runs: list[list[bool]] = []
    cur: list[bool] = []
    for o, t in zip(obs, r.frames, strict=True):
        if t.ball.visible and model._behind_body(t):
            cur.append(bool(o.ball.visible))
            continue
        if len(cur) >= 15:
            runs.append(cur)
        cur = []
    assert len(runs) >= 5
    seen = sum(sum(x) for x in runs) / sum(len(x) for x in runs)
    assert seen <= 0.06, seen
    assert sum(1 for x in runs if not any(x)) >= len(runs) / 4
    assert max(sum(x) / len(x) for x in runs) <= 0.2


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
    # Only false positives (the true ball is never detected): persistent heads and static blobs.
    cfg = NoiseConfig(ball_fp_rate=1.0, ball_miss_rate=1.0, ball_miss_rate_held=1.0, ball_good_miss=1.0,
                      ball_fp_present=1.0)
    obs = NoiseModel(cfg, camera=sim.camera).apply(sim.frames[:300], seed=7)
    w, h = sim.config.image_size
    lo, hi = cfg.ball_fp_conf
    heads = static = 0
    for o, t in zip(obs, sim.frames[:300], strict=True):
        assert o.ball.visible and 0 <= o.ball.image_xy[0] <= w
        assert lo - 0.15 <= o.ball.confidence <= hi + 0.15
        u, v = o.ball.image_xy
        at_head = any(abs(u - (p.bbox.x1 + p.bbox.x2) / 2) < 5 and abs(v - (p.bbox.y1 + 0.06 * p.bbox.h)) < 5
                      for p in t.players)
        heads += at_head
        static += (not at_head) and 0.3 * h - 3 <= v <= h + 3
    assert heads > 30 and static > 30 and heads + static >= 290
    # persistence: the source stays put / follows its player for several frames (not one-frame noise)
    steps = [math.hypot(a.ball.image_xy[0] - b.ball.image_xy[0], a.ball.image_xy[1] - b.ball.image_xy[1])
             for a, b in zip(obs, obs[1:])]
    assert np.median(steps) < 3.0
    # confidences overlap true detections' (cfg.ball_conf 0.5-0.95): no clean confidence cut
    conf = [o.ball.confidence for o in obs]
    assert max(conf) > 0.65 and min(conf) < 0.5
    obs = NoiseModel(NoiseConfig(rim_px_sigma=0.5), camera=sim.camera).apply(sim.frames[:200], seed=8)
    d = [o.rims[0].bbox.x1 - t.rims[0].bbox.x1 for o, t in zip(obs, sim.frames[:200], strict=True)]
    assert 0.3 < np.std(d) < 0.7


def test_identity_swaps(sim: SimResult) -> None:
    cfg = NoiseConfig(id_swap_rate=0.2, id_swap_frames=10, drop_player_rate=0.0, id_swap_max_ft=8.0, **_PLAIN)
    obs = NoiseModel(cfg, camera=sim.camera).apply(sim.frames, seed=9)
    swapped = 0
    for o, t in zip(obs, sim.frames, strict=True):
        for p, q in zip(o.players, t.players, strict=True):
            if p.player_id != q.player_id:
                swapped += 1
                assert p.team is q.team
                assert abs(p.x - q.x) < 2.0  # positions stay with the true player; only the identity moves
    assert swapped > 0
    clean = NoiseModel(NoiseConfig(drop_player_rate=0.0, **_PLAIN), camera=sim.camera).apply(sim.frames, seed=9)
    assert all(p.player_id == q.player_id for o, t in zip(clean, sim.frames, strict=True)
               for p, q in zip(o.players, t.players, strict=True))


def test_occlusion_swaps_are_same_team_and_bounded(sim: SimResult) -> None:
    """Identity errors come from image occlusions between teammates (cv_pipeline §3.2/§3.4); positions
    stay with the bodies, and the identity accuracy stays near the video pipeline's (~0.92)."""
    cfg = NoiseConfig(drop_player_rate=0.0, occluded_drop_rate=0.0, pos_sigma_ft=0.0, team_flip_share=0.0,
                      **_ONE_PER_PLAYER)
    obs = NoiseModel(cfg, camera=sim.camera).apply(sim.frames, seed=12)
    ok = n = 0
    for o, t in zip(obs, sim.frames, strict=True):
        for p, q in zip(o.players, t.players, strict=True):
            assert (p.x, p.y) == (q.x, q.y) and p.team is q.team
            n += 1
            ok += p.player_id == q.player_id
            if p.player_id != q.player_id:
                assert sim.roster.team_of(p.player_id) is q.team
    assert 0.8 < ok / n < 0.995


def test_team_label_flips_are_rare_brief_and_unidentified(sim: SimResult) -> None:
    """A track's team vote flips for short episodes (team accuracy ~0.99, as the video pipeline measures);
    the flipped detection has no roster identity (a provisional "<team>-t<track>" id, no jersey). Detection
    confidences are below 1 and lower for covered players (mono used to report team accuracy 1.0 and
    confidence 1.0 always)."""
    cfg = NoiseConfig(drop_player_rate=0.0, occluded_drop_rate=0.0, occlusion_swap_prob=0.0, pos_sigma_ft=0.0,
                      **_ONE_PER_PLAYER)
    model = NoiseModel(cfg, camera=sim.camera)
    obs = model.apply(sim.frames, seed=21)
    wrong = n = 0
    runs: dict[str, list[int]] = {}
    conf_free, conf_covered = [], []
    for o, t in zip(obs, sim.frames, strict=True):
        covered = dict(zip((q.player_id for q in t.players), model._covered(t.players), strict=True))
        for p, q in zip(o.players, t.players, strict=True):
            n += 1
            if p.team is not q.team:
                wrong += 1
                assert p.player_id == f"{p.team.value}-t{q.track_id}" and p.jersey is None
                runs.setdefault(q.player_id, []).append(o.frame_idx)
            else:
                (conf_covered if covered[q.player_id] >= 0.5 else conf_free).append(p.confidence)
    assert 0.003 < wrong / n < 0.02, wrong / n
    lengths = []
    for frames in runs.values():
        cur = 1
        for a, b in zip(frames, frames[1:]):
            if b == a + 1:
                cur += 1
            else:
                lengths.append(cur)
                cur = 1
        lengths.append(cur)
    assert np.mean(lengths) >= 5  # episodes, not single-frame coin flips
    assert max(conf_free) < 1.0 and np.mean(conf_covered) < np.mean(conf_free)

