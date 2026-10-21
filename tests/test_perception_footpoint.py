"""Court position from a person box: camera from the floor homography, cylinder foot model.

Regression for the round-2 court-position root cause: the naive foot point (bottom centre through
the homography, pushed foot_radius_ft towards image-up) ignores the perspective lean of off-centre
figures; on sim train seeds 1-3 it was biased by ~0.5 ft (median court error of perceived players
0.54 ft, 0.12 ft with the cylinder model on the same boxes).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from courtvision.geometry.camera import RECOMMENDED_SIM_CAMERA, PinholeCamera
from courtvision.geometry.court import court_for
from courtvision.geometry.homography import Calibration, Homography
from courtvision.perception.footpoint import FootModel, FootModelConfig, camera_from_homography
from courtvision.perception.pipeline import PerceptionConfig, PerceptionPipeline
from courtvision.types import BallState, FrameState, PlayerState, Team

HEIGHT_FT, RADIUS_FT = 6.5, 1.0
SIM = RECOMMENDED_SIM_CAMERA


def _sim_camera() -> PinholeCamera:
    return PinholeCamera.look_at(SIM["camera_position"], SIM["camera_target"], SIM["image_size"], SIM["hfov_deg"])


@pytest.mark.parametrize("pos,target,size,hfov", [
    (SIM["camera_position"], SIM["camera_target"], SIM["image_size"], SIM["hfov_deg"]),
    ((10.0, -30.0, 25.0), (60.0, 30.0, 0.0), (1920, 1080), 60.0),  # oblique corner camera
    ((47.0, -20.0, 18.0), (47.0, 25.0, 3.0), (1280, 720), 90.0),  # low wide sideline camera
])
def test_camera_is_recovered_from_the_floor_homography(pos, target, size, hfov) -> None:
    cam = PinholeCamera.look_at(pos, target, size, hfov)
    for H in (cam.floor_homography(), Homography(-3.7 * cam.floor_homography().H)):  # any scale or sign
        rec = camera_from_homography(H, size)
        assert rec is not None
        assert np.allclose(rec.center, pos, atol=1e-3)
        assert rec.K[0, 0] == pytest.approx(cam.K[0, 0], rel=1e-6)


def test_homography_without_a_pinhole_camera_behind_it_falls_back_to_the_naive_point() -> None:
    cam = _sim_camera()
    affine = Homography(np.array([[0.1, 0.0, -10.0], [0.0, -0.2, 120.0], [0.0, 0.0, 1.0]]))  # orthographic: no camera
    mirrored = Homography(cam.floor_homography().H @ np.diag([-1.0, 1.0, 1.0]))  # a mirrored image
    for H in (affine, mirrored):
        assert camera_from_homography(H, cam.image_size) is None
        fm = FootModel(H, cam.image_size, (94.0, 50.0), FootModelConfig(radius_ft=RADIUS_FT))
        assert not fm.corrected
        box = np.array([[600.0, 300.0, 625.0, 365.0]])
        assert np.allclose(fm.court_points(box), fm.naive(box), equal_nan=True)


def test_cylinder_model_removes_the_perspective_bias_of_the_naive_foot_point() -> None:
    cam = _sim_camera()
    fm = FootModel(cam.floor_homography(), cam.image_size, (94.0, 50.0),
                   FootModelConfig(radius_ft=RADIUS_FT, height_ft=HEIGHT_FT))
    assert fm.corrected
    rng = np.random.default_rng(2)
    pts = np.c_[rng.uniform(0, 94, 400), rng.uniform(0, 50, 400)]
    boxes = np.array([cam.bbox_of_cylinder(x, y, HEIGHT_FT, RADIUS_FT).as_tuple() for x, y in pts])
    naive = np.linalg.norm(fm.naive(boxes) - pts, axis=1)
    fixed = np.linalg.norm(fm.court_points(boxes) - pts, axis=1)
    assert naive.mean() > 0.3 and naive.max() > 0.8  # the bias this replaces
    assert fixed.max() < 0.02
    # a player 0.5 ft taller or shorter than the model moves by a few hundredths of a foot only
    tall = np.array([cam.bbox_of_cylinder(x, y, HEIGHT_FT + 0.5, RADIUS_FT).as_tuple() for x, y in pts])
    assert np.linalg.norm(fm.court_points(tall) - pts, axis=1).max() < 0.15


def test_pipeline_locates_off_centre_players_with_the_cylinder_model() -> None:
    render_mod = pytest.importorskip("courtvision.sim.render")
    from courtvision.perception.detector import ColorBlobDetector

    cam = _sim_camera()
    renderer = render_mod.SyntheticRenderer(cam, court_for("nba"))
    calib = Calibration(homography=cam.floor_homography(), image_size=cam.image_size, league="nba")
    spots = [(6.0, 6.0), (88.0, 5.0), (8.0, 44.0), (86.0, 45.0), (30.0, 20.0), (64.0, 32.0)]
    errors = {}
    for model in ("cylinder", "push"):
        pipe = PerceptionPipeline(ColorBlobDetector(), calib,
                                  config=PerceptionConfig(foot_radius_ft=RADIUS_FT, foot_model=model))
        players = [PlayerState(f"p{i}", Team.HOME if i % 2 else Team.AWAY, x, y,
                               bbox=cam.bbox_of_cylinder(x, y, HEIGHT_FT, RADIUS_FT)) for i, (x, y) in enumerate(spots)]
        img = renderer.render(FrameState(frame_idx=0, t=0.0, players=players, ball=BallState(visible=False)))
        state = pipe.process(img, 0, 0.0)
        assert len(state.players) == len(spots)
        errors[model] = [min(math.hypot(p.x - x, p.y - y) for p in state.players) for x, y in spots]
    # the far corners are ~0.3 ft per pixel; the renderer's pixel rounding leaves up to ~0.25 ft there
    assert max(errors["cylinder"]) < 0.3 and np.mean(errors["cylinder"]) < 0.15, errors
    assert max(errors["push"]) > 0.5, errors  # the corners lean most
