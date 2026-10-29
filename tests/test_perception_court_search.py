"""Automatic broadcast court registration (court_search.py, broadcast_court.py) on synthetic renders."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from courtvision.geometry.court import court_for
from courtvision.perception.broadcast_court import BroadcastCourtTracker, BroadcastTrackerConfig
from courtvision.perception.court_search import (
    CameraParams,
    CourtField,
    EvidenceConfig,
    calibrate_frame,
    court_evidence,
    explained,
    refine,
    rotation,
)

COURT = court_for("nba")
CENTER = np.array([47.0, -94.0, 36.0])
SIZE = (960, 540)


def render(cam: CameraParams, size: tuple[int, int] = SIZE, dark_lines: bool = True, seed: int = 0) -> np.ndarray:
    """A floor-coloured image with the court's lines drawn through ``cam`` (dark blue, like Chase Center), a crowd
    of random texture above the far sideline and image noise."""
    rng = np.random.default_rng(seed)
    w, h = size
    img = np.full((h, w, 3), (120, 170, 215), np.uint8)  # light wood (BGR)
    G = cam.G(size)
    far = []
    for x in np.linspace(-10, 104, 60):  # the stands behind the far sideline: busy texture
        q = G @ np.array([x, 53.0, 1.0])
        if q[2] > 0:
            far.append((q[0] / q[2], q[1] / q[2]))
    if far:
        top = int(np.clip(min(p[1] for p in far), 0, h))
        img[:top] = rng.integers(0, 255, (top, w, 3), dtype=np.uint8)
    color = (110, 50, 30) if dark_lines else (245, 245, 245)
    for line in COURT.line_segments():
        pts = np.asarray(line, float)
        for a, b in zip(pts[:-1], pts[1:], strict=True):
            n = max(2, int(math.hypot(*(b - a)) / 0.5))
            seg = a + (b - a) * (np.arange(n + 1)[:, None] / n)
            hom = seg @ G[:, :2].T + G[:, 2]
            ok = hom[:, 2] > 0
            uv = hom[:, :2] / np.where(ok, hom[:, 2], 1.0)[:, None]
            for p, q, oa, ob in zip(uv[:-1], uv[1:], ok[:-1], ok[1:], strict=True):
                if oa and ob and np.all(np.abs(p) < 1e4) and np.all(np.abs(q) < 1e4):
                    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 2, cv2.LINE_AA)
    noise = rng.normal(0, 3, img.shape)
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)


def test_rotation_is_orthonormal_and_pan_zero_looks_across_the_court() -> None:
    R = rotation(12.0, 18.0, 1.5)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    fwd = rotation(0.0, 0.0)[2]
    assert np.allclose(fwd, [0.0, 1.0, 0.0])


def test_camera_params_round_trip() -> None:
    p = CameraParams(CENTER, 10.0, 15.0, 0.3, 2100.0)
    q = CameraParams.from_vector(p.vector())
    assert np.allclose(p.G(SIZE), q.G(SIZE))
    r = CameraParams.from_dict(p.to_dict())
    assert np.allclose(p.G(SIZE), r.G(SIZE))


def test_court_field_is_zero_on_lines() -> None:
    f = CourtField.of(COURT, res=0.25)
    d = f.lookup(np.array([47.0, 47.0, 30.0]), np.array([0.0, 25.0, 10.0]), 99.0)
    assert d[0] < 0.3 and d[1] < 0.3  # sideline, division line
    assert d[2] > 4.0  # open floor (5 ft outside the left arc)


@pytest.mark.parametrize("dark", [True, False])
def test_evidence_finds_bright_and_dark_lines(dark: bool) -> None:
    cam = CameraParams(CENTER, 14.0, 15.5, 0.0, 2300.0)
    ev = court_evidence(render(cam, dark_lines=dark))
    assert ev.lines.sum() > 500
    assert explained(ev, COURT, cam) > 0.6  # most line pixels lie on the true court lines


def test_fixed_centre_search_recovers_the_camera() -> None:
    truth = CameraParams(CENTER, -13.0, 15.5, 0.2, 2200.0)
    ev = court_evidence(render(truth))
    r = calibrate_frame(ev, COURT, centers=[CENTER], free_center=False)
    assert r is not None
    assert abs(r.params.pan - truth.pan) < 0.3
    assert abs(r.params.tilt - truth.tilt) < 0.3
    assert abs(math.log(r.params.f / truth.f)) < 0.02
    assert r.coverage > 0.6


def test_refine_converges_from_a_nearby_start() -> None:
    truth = CameraParams(CENTER, 8.0, 16.0, 0.0, 2500.0)
    ev = court_evidence(render(truth))
    start = CameraParams(CENTER, 9.0, 15.4, 0.0, 2400.0)
    r = refine(ev, COURT, start)
    assert r is not None
    assert abs(r.params.pan - truth.pan) < 0.15 and abs(r.params.tilt - truth.tilt) < 0.15


def test_tracker_follows_a_pan_and_drops_a_cut_to_a_closeup() -> None:
    cfg = BroadcastTrackerConfig(work_width=960, evidence=EvidenceConfig())
    tracker = BroadcastCourtTracker(COURT, SIZE, CENTER, cfg)
    pans = [12.0 - 0.8 * k for k in range(6)]
    statuses = []
    for k, pan in enumerate(pans):
        fix = tracker.update(render(CameraParams(CENTER, pan, 15.0, 0.0, 2300.0), seed=k), cut=k == 0)
        statuses.append(fix.status)
        if fix.params is not None:
            assert abs(fix.params.pan - pan) < 0.4
    assert statuses[0] == "reacquired" and statuses[-1] == "tracking"
    rng = np.random.default_rng(1)
    closeup = rng.integers(0, 255, (SIZE[1], SIZE[0], 3), dtype=np.uint8)  # no floor at all
    assert tracker.update(closeup, cut=True).status == "lost"
