"""BallTracker: gravity Kalman, gating against false positives, short-gap fill, re-acquisition."""

from __future__ import annotations

import numpy as np
from test_perception_synth import make_camera

from courtvision.perception.ball import BallTracker, BallTrackerConfig, fill_gaps
from courtvision.types import BallState, BBox, Detection, ObjClass

FPS = 30.0
G_PX = 700.0  # projected gravity in px/s^2 for this synthetic flight


def ball_det(u: float, v: float, r: float = 5.0, score: float = 0.9) -> Detection:
    return Detection(bbox=BBox(u - r, v - r, u + r, v + r), cls=ObjClass.BALL, score=score)


def flight(k: int) -> tuple[float, float]:
    t = k / FPS
    return 200.0 + 300.0 * t, 500.0 - 600.0 * t + 0.5 * G_PX * t * t


def test_parabolic_flight_gap_is_interpolated_close_to_truth() -> None:
    rng = np.random.default_rng(1)
    bt = BallTracker()
    gap = range(20, 24)
    errs_obs, errs_gap = [], []
    for k in range(45):
        u, v = flight(k)
        dets = [] if k in gap else [ball_det(u + rng.normal(0, 0.3), v + rng.normal(0, 0.3))]
        st = bt.update(dets, k, k / FPS)
        assert st.visible and st.image_xy is not None
        err = float(np.hypot(st.image_xy[0] - u, st.image_xy[1] - v))
        if k in gap:
            assert st.interpolated and st.confidence < 0.9
            errs_gap.append(err)
        else:
            assert not st.interpolated
            if k > 5:
                errs_obs.append(err)
    assert max(errs_obs) < 1.5
    assert max(errs_gap) < 3.0  # the acceleration state carries the arc through the gap
    assert abs(bt._x[4] - G_PX) < 150  # learned projected gravity (white-box check)


def test_far_false_positive_is_gated_out() -> None:
    bt = BallTracker()
    for k in range(30):
        u, v = flight(k)
        dets = [ball_det(u, v, score=0.6)]
        if k >= 10:
            dets.append(ball_det(u + 400.0, v - 50.0, score=0.95))  # spare ball / head false positive
        st = bt.update(dets, k, k / FPS)
        assert np.hypot(st.image_xy[0] - u, st.image_xy[1] - v) < 2.0, k


def test_long_gap_reports_not_visible_then_reacquires() -> None:
    cfg = BallTrackerConfig(max_gap_frames=6)
    bt = BallTracker(cfg)
    for k in range(10):
        bt.update([ball_det(*flight(k))], k, k / FPS)
    states = [bt.update([], k, k / FPS) for k in range(10, 20)]
    assert [s.visible for s in states] == [True] * 6 + [False] * 4
    assert all(s.interpolated for s in states[:6])
    st = bt.update([ball_det(900.0, 300.0)], 20, 20 / FPS)
    assert st.visible and not st.interpolated and st.image_xy == (900.0, 300.0)


def test_jump_outside_gate_is_reacquired_after_a_few_frames() -> None:
    cfg = BallTrackerConfig(reacquire_after=3)
    bt = BallTracker(cfg)
    for k in range(10):
        bt.update([ball_det(300.0, 300.0)], k, k / FPS)  # held ball
    outs = [bt.update([ball_det(800.0, 250.0, score=0.6)], k, k / FPS) for k in range(10, 14)]
    assert all(o.interpolated for o in outs[:2])  # weak lone detection: treated as a false positive first
    assert outs[2].image_xy == (800.0, 250.0) and not outs[2].interpolated


def test_lone_confident_detection_is_taken_at_once() -> None:
    bt = BallTracker()
    for k in range(10):
        bt.update([ball_det(300.0, 330.0)], k, k / FPS)  # held at hand height
    st = bt.update([ball_det(300.0, 290.0, score=0.9)], 10, 10 / FPS)  # release: 40 px in one frame
    assert not st.interpolated and st.image_xy == (300.0, 290.0)
    two = BallTracker()
    for k in range(10):
        two.update([ball_det(300.0, 330.0)], k, k / FPS)
    st = two.update([ball_det(300.0, 290.0, score=0.9), ball_det(700.0, 200.0, score=0.9)], 10, 10 / FPS)
    assert st.interpolated  # two out-of-gate candidates: wait for evidence


def test_dribble_bounce_stays_locked() -> None:
    bt = BallTracker()
    v, vel = 400.0, 0.0
    for k in range(60):
        vel += 1200.0 / FPS
        v += vel / FPS
        if v > 520.0:  # floor contact: bounce back up with 80 % speed
            v, vel = 520.0, -0.8 * vel
        st = bt.update([ball_det(500.0 + k, v)], k, k / FPS)
        assert not st.interpolated and abs(st.image_xy[1] - v) < 2.0


def test_court_xy_from_floor_homography_and_radius() -> None:
    cam = make_camera()
    H = cam.floor_homography()
    uv = cam.project(np.array([30.0, 20.0, 0.0]))
    bt = BallTracker()
    st = bt.update([ball_det(float(uv[0]), float(uv[1]), r=4.0)], 0, 0.0, H)
    assert st.court_xy is not None
    assert np.allclose(st.court_xy, (30.0, 20.0), atol=1e-6)
    assert st.radius_px == 4.0 and st.z is None
    sky = BallTracker().update([ball_det(640.0, -400.0)], 0, 0.0, H)  # above the horizon (v ~ -169)
    assert sky.visible and sky.court_xy is None


def test_no_detections_ever_is_invisible() -> None:
    bt = BallTracker()
    assert not bt.update([], 0, 0.0).visible
    assert bt.predicted_xy is None
    bt.update([ball_det(10, 10, score=0.1)], 1, 1 / FPS)  # below min_score
    assert bt.predicted_xy is None


def test_fill_gaps_interpolates_two_sided() -> None:
    rng = np.random.default_rng(3)
    bt = BallTracker()
    states, times, truth = [], [], []
    for k in range(40):  # gap right after a "release": online extrapolation is weakest here
        u, v = flight(k)
        dets = [] if 8 <= k < 13 else [ball_det(u + rng.normal(0, 0.3), v + rng.normal(0, 0.3))]
        states.append(bt.update(dets, k, k / FPS))
        times.append(k / FPS)
        truth.append((u, v))
    filled = fill_gaps(states, times, max_gap=6)
    for k in range(8, 13):
        assert filled[k].interpolated
        assert np.hypot(filled[k].image_xy[0] - truth[k][0], filled[k].image_xy[1] - truth[k][1]) < 1.0
    assert filled[20] is states[20]
    lead = [BallState(visible=False)] * 2 + states[2:]
    assert fill_gaps(lead, times)[:2] == lead[:2]  # no left context: unchanged


def test_non_finite_detection_does_not_poison_the_filter() -> None:
    """Regression: one NaN box used to turn the state into NaN for good (a NaN gate accepts every
    later detection), so the ball was reported visible at (nan, nan) forever."""
    nan = float("nan")
    bt = BallTracker()
    st = bt.update([Detection(bbox=BBox(nan, nan, nan, nan), cls=ObjClass.BALL, score=0.9)], 0, 0.0)
    assert not st.visible
    for k in range(1, 5):
        u, v = flight(k)
        dets = [ball_det(u, v), Detection(bbox=BBox(nan, 1.0, 2.0, 3.0), cls=ObjClass.BALL, score=0.95)]
        st = bt.update(dets, k, k / FPS)
        assert st.visible and np.all(np.isfinite(st.image_xy))
        assert np.hypot(st.image_xy[0] - u, st.image_xy[1] - v) < 2.0


def test_fill_gaps_takes_context_only_next_to_the_gap() -> None:
    """Regression: context used to reach across an earlier long gap into another flight phase
    (a dribble 1 s before a shot arc), bending the quadratic by 2-3 px."""

    def truth(k: int) -> tuple[float, float]:
        if k < 10:
            return 100.0 + k, 650.0  # low dribble on the left
        t = (k - 40) / FPS
        return 900.0 + 500.0 * t, 200.0 - 200.0 * t + 0.5 * G_PX * t * t  # later: a shot arc

    observed = set(range(10)) | {40} | set(range(44, 50))
    states = [BallState(visible=True, image_xy=truth(k), confidence=0.9) if k in observed
              else BallState(visible=False) for k in range(50)]
    times = [k / FPS for k in range(50)]
    filled = fill_gaps(states, times, max_gap=6)
    for k in (41, 42, 43):
        assert filled[k].interpolated
        assert np.hypot(filled[k].image_xy[0] - truth(k)[0], filled[k].image_xy[1] - truth(k)[1]) < 0.5
    assert not any(filled[k].visible for k in range(10, 40))  # the long gap stays unfilled
