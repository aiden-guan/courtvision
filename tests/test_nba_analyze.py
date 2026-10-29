"""NBA broadcast analysis passes (nba/analyze.py) and the box-score comparison (nba/evaluate.py)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from courtvision.nba.analyze import (
    BallLinkConfig,
    CameraTrack,
    DetectionCache,
    _smooth_runs,
    is_referee,
    link_ball,
    rectify_box,
)
from courtvision.nba.espn import TeamBox
from courtvision.nba.evaluate import BOX_STATS, box_error
from courtvision.perception.court_search import CameraParams
from courtvision.types import Team

C = np.array([47.0, -94.0, 36.0])
SIZE = (1920, 1080)


def track(params: dict[int, tuple[float, float, float, float]]) -> CameraTrack:
    return CameraTrack(center=C, image_size=SIZE, fps=30.0, params=params)


def jersey(bgr: tuple[int, int, int], stripes: bool = False) -> np.ndarray:
    """A frame filled with one shirt colour, optionally with black vertical stripes over a third of it."""
    img = np.full((300, 200, 3), bgr, np.uint8)
    if stripes:
        img[:, np.arange(200) % 9 < 3] = 0
    return img


# --------------------------------------------------------------------------- camera track


def test_smooth_runs_fill_gaps_of_up_to_max_fill_frames_and_keep_longer_gaps_empty() -> None:
    frames = [*range(0, 10), *range(13, 21), *range(25, 30)]  # 3 frames missing at 10-12, 4 at 21-24
    raw = {f: (0.5 * f, 15.0, 0.0, 2000.0 if f < 10 else 4000.0) for f in frames}
    out = _smooth_runs(raw, k=1, max_fill=3)
    assert sorted(out) == [*range(0, 21), *range(25, 30)]
    pan, tilt, roll, focal = out[11]
    assert pan == pytest.approx(5.5) and tilt == pytest.approx(15.0) and roll == 0.0
    assert focal == pytest.approx(math.sqrt(2000.0 * 4000.0))  # the zoom is interpolated in log focal length
    assert sorted(_smooth_runs(raw, k=1, max_fill=2)) == frames


def test_smooth_runs_average_jitter_within_a_run_but_not_across_a_long_gap() -> None:
    raw = {f: (5.0 if f == 5 else 0.0, 15.0, 0.0, 3000.0) for f in range(11)}  # one jittery fit
    raw.update({f: (10.0, 15.0, 0.0, 3000.0) for f in range(20, 31)})  # the next run, after a cut
    out = _smooth_runs(raw, k=5, max_fill=3)
    assert [round(out[f][0], 6) for f in range(2, 9)] == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    assert all(out[f][0] == pytest.approx(10.0) for f in range(20, 31))
    assert out[0][3] == pytest.approx(3000.0)


def test_frame_to_frame_maps_a_point_to_the_same_world_direction_in_the_other_frame() -> None:
    cam = track({0: (5.0, 16.0, 0.4, 2600.0), 1: (9.0, 18.5, -0.3, 3400.0)})  # a turn, a tilt, a roll and a zoom
    H = cam.frame_to_frame(0, 1)
    assert H is not None
    for X in ((60.0, 20.0, 0.0), (52.0, 30.0, 10.0), (40.0, 10.0, 6.5)):  # on the floor, at rim height, a head
        a, b = (cam.get(f).camera(SIZE).project(np.array(X)) for f in (0, 1))  # type: ignore[union-attr]
        h = H @ np.array([a[0], a[1], 1.0])
        assert np.allclose(h[:2] / h[2], b, atol=1e-6)
    back = cam.frame_to_frame(1, 0)
    assert back is not None and np.allclose(back @ H / (back @ H)[2, 2], np.eye(3), atol=1e-9)
    assert cam.frame_to_frame(0, 7) is None  # no camera in frame 7


# --------------------------------------------------------------------------- detections and the ball path


def test_detection_cache_save_and_load_round_trip(tmp_path: Path) -> None:
    balls = {3: np.array([[100.0, 200.0, 0.9], [50.0, 60.0, 0.2]], np.float32), 4: np.zeros((0, 3), np.float32),
             9: np.array([[1.5, 2.5, 0.5]], np.float32)}
    persons = {3: np.array([[10.0, 20.0, 30.0, 80.0, 0.8]], np.float32),
               4: np.array([[1.0, 2.0, 3.0, 4.0, 0.5], [5.0, 6.0, 7.0, 8.0, 0.4]], np.float32)}  # none in frame 9
    path = tmp_path / "detections.npz"
    DetectionCache(balls, persons).save(path)
    back = DetectionCache.load(path)
    assert sorted(back.balls) == sorted(back.persons) == [3, 4, 9]
    for f in (3, 4, 9):
        np.testing.assert_array_equal(back.balls[f], balls[f])
    for f in (3, 4):
        np.testing.assert_array_equal(back.persons[f], persons[f])
    assert back.balls[4].shape == (0, 3) and back.persons[9].shape == (0, 5)


def test_link_ball_follows_a_moving_ball_rather_than_a_static_distractor() -> None:
    cam = track({f: (0.0, 15.0, 0.0, 3000.0) for f in range(100, 115)})  # a still camera
    ball: dict[int, tuple[float, float, float]] = {}
    cache = DetectionCache()
    for f in range(100, 115):
        k = f - 100
        ball[f] = (400.0 + 40.0 * k, 600.0 + 8.0 * k, 0.3 if k in (6, 7) else 0.6)  # its score dips in two frames
        # a static ball-like thing scoring 0.5, and a candidate below the minimum score
        cache.balls[f] = np.array([[1500.0, 200.0, 0.5], ball[f], [900.0, 900.0, 0.05]], np.float32)
    path = link_ball(cache, cam)
    assert sorted(path) == list(range(100, 115))
    assert all(np.allclose(path[f], ball[f]) for f in path)


def test_link_ball_keeps_the_chain_through_a_fast_camera_pan() -> None:
    params = {f: (-12.0 + 2.5 * f, 18.0, 0.3, 3000.0) for f in range(12)}  # the camera turns 2.5 degrees a frame
    cam = track(params)
    X = np.array([50.0, 22.0, 4.0])  # the ball, held still
    uv = {f: cam.get(f).camera(SIZE).project(X) for f in params}  # type: ignore[union-attr]
    # in the image the ball jumps further per frame than a ball may move
    assert min(np.linalg.norm(uv[f + 1] - uv[f]) for f in range(11)) > BallLinkConfig().max_step_px
    logo = (300.0, 950.0)  # a ball-like graphic fixed in the image, scoring higher than the ball
    cache = DetectionCache({f: np.array([[*logo, 0.55], [*uv[f], 0.5]], np.float32) for f in params})
    path = link_ball(cache, cam)
    assert sorted(path) == list(range(12))
    assert all(np.allclose(path[f][:2], uv[f], atol=0.01) for f in path)
    # told the camera stood still, the same candidates link the logo: removing the pan is what keeps the ball
    still = link_ball(cache, track({f: params[0] for f in params}))
    assert sorted(still) == list(range(12)) and all(np.allclose(still[f][:2], logo) for f in still)


# --------------------------------------------------------------------------- person boxes


def test_rectify_box_is_none_when_the_box_centre_ray_does_not_come_down() -> None:
    cam = CameraParams(C, 0.0, 3.0, 0.0, 3000.0)  # nearly level: the horizon is at v = 540 - 3000 tan 3 deg = 383
    assert rectify_box((900.0, 150.0, 960.0, 250.0), cam, SIZE) is None  # someone in the stands, above the horizon
    assert rectify_box((900.0, 760.0, 960.0, 840.0), cam, SIZE) is not None


def test_is_referee_on_synthetic_torsos() -> None:
    box = (40.0, 20.0, 160.0, 280.0)
    assert is_referee(jersey((128, 128, 128), stripes=True), box)  # grey with black stripes
    assert not is_referee(jersey((255, 255, 255)), box)  # a white jersey
    assert not is_referee(jersey((150, 30, 10), stripes=True), box)  # saturated blue, even with dark stripes
    assert not is_referee(jersey((128, 128, 128)), box)  # plain grey: no dark stripes


# --------------------------------------------------------------------------- box score


def test_box_error_totals_and_exact_share_on_a_tiny_box() -> None:
    truth = {Team.HOME: TeamBox(Team.HOME, "HOM", {"pts": 10, "fgm": 4, "fga": 8, "ftm": 2, "fta": 2}),
             Team.AWAY: TeamBox(Team.AWAY, "AWY", {"pts": 6, "fgm": 3, "fga": 7})}  # 42 official counts in all
    pred = {Team.HOME: {"pts": 12, "fgm": 5, "fga": 8, "ftm": 2, "fta": 2},
            Team.AWAY: {"pts": 6, "fgm": 3, "fga": 5, "oreb": 1}}  # off by +2, +1, -2 and +1
    cmp = box_error(pred, truth)
    assert cmp.error == round(6 / 42, 4)
    assert len(cmp.rows) == 2 * len(BOX_STATS) and cmp.exact_share == round(22 / 26, 4)
    rows = {(r["team"], r["stat"]): r for r in cmp.rows}
    assert rows[("home", "pts")] == {"team": "home", "stat": "pts", "pred": 12, "truth": 10, "diff": 2}
    assert rows[("away", "fga")]["diff"] == -2 and rows[("away", "oreb")]["diff"] == 1
    assert rows[("away", "blk")] == {"team": "away", "stat": "blk", "pred": 0, "truth": 0, "diff": 0}
