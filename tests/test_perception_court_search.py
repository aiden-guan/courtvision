"""Automatic broadcast court registration (court_search.py, broadcast_court.py) on synthetic renders."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from courtvision.geometry.court import court_for
from courtvision.perception.broadcast_court import BroadcastCourtTracker, BroadcastTrackerConfig
from courtvision.perception import court_search
from courtvision.perception.court_search import (
    BroadcastPrior,
    CameraParams,
    CourtField,
    EvidenceConfig,
    aim_range,
    calibrate_frame,
    court_evidence,
    explained,
    plausible,
    refine,
    rotation,
)
from courtvision.perception.court_tracker import court_samples

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


def test_tracker_keeps_the_camera_through_a_false_cut() -> None:
    tracker = BroadcastCourtTracker(COURT, SIZE, CENTER, BroadcastTrackerConfig(work_width=960))
    pans = [6.0 - 0.6 * k for k in range(5)]
    for k, pan in enumerate(pans):
        fix = tracker.update(render(CameraParams(CENTER, pan, 15.0, 0.0, 2300.0), seed=k), cut=k in (0, 3))
        if k == 3:  # the shot-boundary detector fired, but it is the same camera still turning
            assert fix.status == "tracking"
            assert fix.params is not None and abs(fix.params.pan - pan) < 0.4


def test_plausible_rejects_an_upside_down_fit_and_keeps_a_game_camera() -> None:
    assert plausible(CameraParams(CENTER, -13.0, 15.5, 0.2, 2200.0), (1920, 1080))
    assert not plausible(CameraParams(CENTER, 5.0, -21.0, 33.0, 48.0), (1920, 1080))  # seen on a real frame
    assert not plausible(CameraParams(CENTER, 5.0, 15.0, 4.0, 2200.0), (1920, 1080))  # rolled
    assert not plausible(CameraParams(CENTER, 5.0, -2.0, 0.0, 2200.0), (1920, 1080))  # looking up
    assert not plausible(CameraParams(CENTER, 5.0, 15.0, 0.0, 400.0), (1920, 1080))  # wider than the prior
    assert not plausible(CameraParams(CENTER, 5.0, 15.0, 0.0, 30000.0), (1920, 1080))  # tighter than the prior


def test_aim_range_tilt_peaks_at_the_nearest_edge_point_not_a_corner() -> None:
    pr = BroadcastPrior()
    C = np.array([47.0, -62.0, 26.0])
    pan_lo, pan_hi, tilt_lo, tilt_hi = aim_range(C, pr)
    assert pan_lo < 0.0 < pan_hi
    nearest = math.degrees(math.atan2(26.0, -4.0 + 62.0))  # (47, -4): the middle of the near edge of the box
    assert tilt_hi == pytest.approx(nearest)
    corners = [math.degrees(math.atan2(26.0, math.hypot(x - 47.0, y + 62.0))) for x in pr.aim_x for y in pr.aim_y]
    assert tilt_hi > max(corners) + 1.0 and tilt_lo == pytest.approx(min(corners))


def test_people_boxes_partly_off_the_frame_do_not_wrap_round() -> None:
    img = render(CameraParams(CENTER, 14.0, 15.5, 0.0, 2300.0))
    base = court_evidence(img)
    ev = court_evidence(img, people=[(-50.0, -80.0, 30.0, -10.0), (900.0, 500.0, 1200.0, 700.0)])
    assert np.array_equal(ev.floor[:, :890], base.floor[:, :890])  # the box above the frame cleared nothing
    assert not ev.floor[500:, 900:].any()
    cfg = EvidenceConfig(exclude=[(-100, -100, 50, 50)])  # an exclusion box wholly off the frame
    assert np.array_equal(court_evidence(img, cfg).floor, base.floor)


def test_line_distance_is_clipped_at_a_resolution_free_distance() -> None:
    cam = CameraParams(CENTER, 14.0, 15.5, 0.0, 2300.0)
    small = court_evidence(render(cam))  # 540 lines
    big = court_evidence(render(cam, size=(1920, 1080)))
    assert small.dist.max() == pytest.approx(EvidenceConfig().dist_clip_px / 2.0)
    assert big.dist.max() == pytest.approx(EvidenceConfig().dist_clip_px)


def test_chamfer_pass_with_the_centre_free_pulls_in_a_pan_tilt_and_zoom_error() -> None:
    """scipy's diff_step is relative to |x|: with it the centre (y = -94 ft) stepped by 9 ft and a zero pan or roll
    by about 1e-8, and this start ended about 12 px off (fixed absolute steps: under 2 px)."""
    truth = CameraParams(CENTER, 0.0, 15.5, 0.0, 2300.0)  # pan and roll exactly 0
    ev = court_evidence(render(truth))
    pts, _ = court_samples(COURT, 1.0)
    start = CameraParams(CENTER + np.array([0.0, 1.0, 0.0]), 0.8, 15.9, 0.0, 2300.0 * 1.03)
    out = CameraParams.from_vector(court_search._chamfer_pass(ev, pts, start.vector(), np.array([True] * 7), SIZE))

    def err(p: CameraParams) -> float:
        a, _ = court_search._project(p.G(SIZE), pts)
        b, wb = court_search._project(truth.G(SIZE), pts)
        ok = (wb > 0) & (b[:, 0] >= 0) & (b[:, 0] < SIZE[0]) & (b[:, 1] >= 0) & (b[:, 1] < SIZE[1])
        return float(np.median(np.linalg.norm(a[ok] - b[ok], axis=1)))

    assert err(start) > 30.0 and err(out) < 4.0


def test_calibrate_frame_passes_the_reacquisition_window_to_the_search(monkeypatch: pytest.MonkeyPatch) -> None:
    got: dict[str, float] = {}

    def fake_search(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        got["around_deg"] = kwargs["around_deg"]
        return []

    monkeypatch.setattr(court_search, "search", fake_search)
    ev = court_evidence(render(CameraParams(CENTER, 0.0, 15.5, 0.0, 2300.0)))
    assert calibrate_frame(ev, COURT, around=CameraParams(CENTER, 0.0, 15.5, 0.0, 2300.0), around_deg=25.0) is None
    assert got["around_deg"] == 25.0


def test_level_zero_suppression_keeps_distinct_peaks() -> None:
    pans = np.array([0.0, 0.1, 0.2, 10.0, 10.1, 20.0])
    tilts = np.full(6, 15.0)
    focals = np.full(6, 2000.0)
    order = np.arange(6)  # already best first
    kept = court_search._nms(order, pans, tilts, focals, step0=40.0, focal_step=0.1, n=3)
    assert kept == [0, 3, 5]
