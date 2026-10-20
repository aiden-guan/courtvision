"""PinholeCamera: look_at axes, projection, floor homography, exact bboxes, sim camera framing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from courtvision.geometry.camera import RECOMMENDED_SIM_CAMERA, PinholeCamera, sees_full_court
from courtvision.geometry.court import LEAGUES, court_for

NBA = court_for("nba")
W, H = 1280, 720


def _sim_camera(position=(47.0, -38.0, 26.0), target=(47.0, 25.0, 0.0), hfov=72.0) -> PinholeCamera:
    return PinholeCamera.look_at(position, target, (W, H), hfov)


def _recommended() -> PinholeCamera:
    rc = RECOMMENDED_SIM_CAMERA
    return PinholeCamera.look_at(rc["camera_position"], rc["camera_target"], rc["image_size"], rc["hfov_deg"])


def test_look_at_axes_and_intrinsics() -> None:
    cam = _sim_camera()
    assert np.allclose(cam.R @ cam.R.T, np.eye(3), atol=1e-12) and np.linalg.det(cam.R) == pytest.approx(1.0)
    assert np.allclose(cam.center, [47.0, -38.0, 26.0])
    f = (W / 2) / math.tan(math.radians(36.0))
    assert np.allclose(cam.K, [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    assert np.allclose(cam.project(np.array([47.0, 25.0, 0.0])), [W / 2, H / 2])
    right = cam.project(np.array([57.0, 25.0, 0.0]))  # world +x is image right for a camera looking +y
    up = cam.project(np.array([47.0, 25.0, 5.0]))  # world +z is image up
    assert right[0] > W / 2 and abs(right[1] - H / 2) < 1e-9
    assert up[1] < H / 2 and abs(up[0] - W / 2) < 1e-9
    assert cam.R[2] @ np.array([0.0, 63.0, -26.0]) > 0  # camera z is forward


def test_look_at_straight_down_and_validation() -> None:
    cam = PinholeCamera.look_at((47.0, 25.0, 80.0), (47.0, 25.0, 0.0))
    assert np.isfinite(cam.R).all()
    assert np.allclose(cam.project(np.array([47.0, 25.0, 0.0])), [640.0, 360.0])
    with pytest.raises(ValueError):
        PinholeCamera.look_at((0, 0, 0), (0, 0, 0))
    with pytest.raises(ValueError):
        PinholeCamera.look_at((0, -10, 5), (0, 0, 0), hfov_deg=180.0)


def test_project_shapes_and_points_behind_camera() -> None:
    cam = _sim_camera()
    pts = np.array([[47.0, 25.0, 0.0], [47.0, -60.0, 0.0], [10.0, 10.0, 0.0]])  # middle one is behind
    uv = cam.project(pts)
    assert uv.shape == (3, 2)
    assert np.isfinite(uv[0]).all() and np.isnan(uv[1]).all() and np.isfinite(uv[2]).all()
    assert cam.project(pts[0]).shape == (2,)
    assert np.isnan(cam.project(np.array(cam.center))).all()  # on the camera centre
    with pytest.raises(ValueError):
        cam.project(np.zeros((4, 2)))


def test_floor_homography_agrees_with_projection() -> None:
    rng = np.random.default_rng(0)
    for cam in (_sim_camera(), _recommended()):
        court = np.column_stack([rng.uniform(-6, 100, 400), rng.uniform(-6, 56, 400)])
        xyz = np.column_stack([court, np.zeros(len(court))])
        hom = cam.floor_homography()
        projected = cam.project(xyz)
        assert np.abs(hom.court_to_image(court) - projected).max() < 1e-6
        assert np.abs(hom.image_to_court(projected) - court).max() < 1e-6
        assert np.isfinite(hom.image_to_court(projected, front_only=True)).all()


def _sample_sphere(x: float, y: float, z: float, r: float, n: int = 400) -> np.ndarray:
    th, ph = np.meshgrid(np.linspace(0, np.pi, n), np.linspace(0, 2 * np.pi, 2 * n))
    return np.column_stack([
        x + r * (np.sin(th) * np.cos(ph)).ravel(),
        y + r * (np.sin(th) * np.sin(ph)).ravel(),
        z + r * np.cos(th).ravel(),
    ])


def _sample_cylinder(x: float, y: float, h: float, r: float, n: int = 4096) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.column_stack([x + r * np.cos(a), y + r * np.sin(a)])
    return np.vstack([np.column_stack([ring, np.zeros(n)]), np.column_stack([ring, np.full(n, h)])])


def _bbox_of(uv: np.ndarray) -> tuple[float, float, float, float]:
    return uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()


@pytest.mark.parametrize("ball", [(47.0, 25.0, 4.0), (5.25, 25.0, 10.5), (90.0, 3.0, 1.0), (20.0, 45.0, 14.0)])
def test_bbox_of_sphere_matches_dense_sampling(ball) -> None:
    cam = _recommended()
    r = 0.39  # size-7 ball radius, ft
    exact = cam.bbox_of_sphere(*ball, r).as_tuple()
    sampled = _bbox_of(cam.project(_sample_sphere(*ball, r)))
    assert np.allclose(exact, sampled, atol=0.02)
    for e, s in zip(exact[:2], sampled[:2]):
        assert e <= s + 1e-9  # sampling can only shrink the box
    for e, s in zip(exact[2:], sampled[2:]):
        assert e >= s - 1e-9
    u, v = cam.project(np.array(ball))
    box = cam.bbox_of_sphere(*ball, r)
    assert box.contains(u, v)
    point = cam.bbox_of_sphere(*ball, 0.0)
    assert point.x1 == pytest.approx(u) and point.x2 == pytest.approx(u) and point.y1 == pytest.approx(v)


@pytest.mark.parametrize("player", [(0.0, 0.0), (47.0, 25.0), (94.0, 50.0), (30.0, 10.0)])
def test_bbox_of_cylinder_matches_dense_sampling(player) -> None:
    cam = _recommended()
    exact = cam.bbox_of_cylinder(*player, 6.5, 0.9).as_tuple()
    sampled = _bbox_of(cam.project(_sample_cylinder(*player, 6.5, 0.9)))
    assert np.allclose(exact, sampled, atol=1e-3)
    box = cam.bbox_of_cylinder(*player, 6.5, 0.9)
    foot_u, foot_v = cam.project(np.array([*player, 0.0]))
    assert box.y2 >= foot_v and box.x1 <= foot_u <= box.x2


def test_bboxes_are_nan_when_reaching_the_camera_plane() -> None:
    cam = _sim_camera()
    assert math.isnan(cam.bbox_of_sphere(47.0, -38.0, 26.0, 0.4).x1)  # sphere around the camera
    assert math.isnan(cam.bbox_of_cylinder(47.0, -60.0, 6.5, 0.9).x1)  # behind the camera
    assert math.isnan(cam.bbox_of_circle(47.0, -38.0, 26.0, 0.75).x1)


def test_dict_round_trip() -> None:
    cam = _recommended()
    restored = PinholeCamera.from_dict(cam.to_dict())
    assert np.allclose(restored.P, cam.P) and restored.image_size == cam.image_size


def test_default_sim_camera_cuts_off_the_near_corners() -> None:
    # Finding reported to the sim agent: ARCHITECTURE's SimConfig default does not frame the court.
    cam = _sim_camera()
    corners = cam.project(np.array([[0.0, 0.0, 0.0], [94.0, 0.0, 0.0]]))
    assert corners[0, 0] < 0 and corners[1, 0] > W
    assert not sees_full_court(cam, NBA)


@pytest.mark.parametrize("league", LEAGUES)
def test_recommended_sim_camera_frames_the_court(league: str) -> None:
    cam = _recommended()
    court = court_for(league)
    assert sees_full_court(cam, court, margin_px=20.0)
    # Players standing on each corner are fully inside too (not just their feet).
    for x, y in ((0.0, 0.0), (court.length, 0.0), (0.0, court.width), (court.length, court.width)):
        box = cam.bbox_of_cylinder(x, y, 6.7, 0.9)
        assert cam.in_image([[box.x1, box.y1], [box.x2, box.y2]], margin_px=20.0).all()
    assert RECOMMENDED_SIM_CAMERA["hfov_deg"] == 72.0 and RECOMMENDED_SIM_CAMERA["image_size"] == (W, H)


def test_camera_equality_does_not_raise() -> None:
    # Before: the dataclass __eq__ compared ndarray fields in a tuple and raised ValueError.
    a, b = _recommended(), _recommended()
    assert a == b and a is not b
    assert a != _sim_camera()
    assert (a == "camera") is False
    with pytest.raises(TypeError):
        hash(a)
