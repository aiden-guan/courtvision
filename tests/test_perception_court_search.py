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


def test_restrict_keeps_the_court_region_and_drops_the_stands() -> None:
    cam = CameraParams(CENTER, 14.0, 15.5, 0.0, 2300.0)
    ev = court_evidence(render(cam))
    loc = court_search.restrict(ev, COURT, cam)
    region = court_search.court_region(COURT, cam, ev.size, 4.0)
    assert loc.lines.any() and not (loc.lines & ~region).any()  # nothing beyond the court and its margin
    assert court_search.coverage(loc, COURT, cam) >= court_search.coverage(ev, COURT, cam)  # no court line lost
    lazy = court_evidence(render(cam), lines=False)  # the tracker's frame: lines only on demand
    assert lazy.lines.size == 0 and court_search.with_lines(lazy).lines.sum() == ev.lines.sum()


# ---------------------------------------------------------------- speed-ups that must not change a result


@pytest.mark.parametrize("loss_in_play", [False, True])
def test_lost_frames_only_build_evidence_at_the_unchanged_reacquisition_times(
    monkeypatch: pytest.MonkeyPatch, loss_in_play: bool,
) -> None:
    tracker = BroadcastCourtTracker(COURT, SIZE, CENTER)
    tracker._loss_in_play = loss_in_play
    evidence = tracker.evidence
    calls = []
    frame_idx = 0

    def record_evidence(*args, **kwargs):
        calls.append(frame_idx)
        return evidence(*args, **kwargs)

    monkeypatch.setattr(tracker, "evidence", record_evidence)
    blank = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    for frame_idx in range(100):
        assert tracker.update(blank, cut=frame_idx == 43).status == "lost"
    scheduled = {0, 30, 60, 90, 43, 45, 49, 58}  # regular attempts plus 0, 2, 6, 15 frames after the cut
    if loss_in_play:
        scheduled |= {3, 9, 19}  # 1, 4, 10, 20 lost frames since an in-play loss (frame 0 is already scheduled)
    assert calls == sorted(scheduled)
    assert tracker._lost_frames == 100 and tracker._since_cut == 56


@pytest.mark.parametrize("free_center,fix_roll", [(False, False), (False, True), (True, False), (True, True)])
def test_batched_chamfer_probes_keep_the_serial_jacobian_and_fit(
    monkeypatch: pytest.MonkeyPatch, free_center: bool, fix_roll: bool,
) -> None:
    """Compare against independent scalar central differences, including probes outside the image. Both the
    derivatives and the complete multi-pass fit must agree exactly, so solver convergence cannot hide drift."""
    from scipy import optimize

    truth = CameraParams(CENTER, 0.0, 15.5, 0.0, 2300.0)
    ev = court_evidence(render(truth))
    start = CameraParams(CENTER + [0.0, 1.0, 0.0], 0.8, 15.9, 0.0, 2369.0)
    cfg = court_search.RefineConfig(fix_roll=fix_roll)
    batched = refine(ev, COURT, start, free_center=free_center, config=cfg)
    solve = optimize.least_squares
    checked = []

    def serial_solve(fun, x0, **kwargs):
        if callable(kwargs.get("jac")) and kwargs.get("loss") == "soft_l1":
            batch_jac = kwargs["jac"]
            steps = kwargs["x_scale"] * 0.02

            def serial_jac(p):
                columns = []
                for i, step in enumerate(steps):
                    delta = np.zeros_like(p)
                    delta[i] = step
                    columns.append((fun(p + delta) - fun(p - delta)) / (2.0 * step))
                return np.column_stack(columns)

            for delta in (0.0, 0.1, 40.0):
                p = x0.copy()
                p[3 if free_center else 0] += delta  # pan: exercise clipped/off-image samples too
                np.testing.assert_array_equal(batch_jac(p), serial_jac(p), strict=True)
            checked.append(True)
            kwargs["jac"] = serial_jac
        return solve(fun, x0, **kwargs)

    monkeypatch.setattr(optimize, "least_squares", serial_solve)
    serial = refine(ev, COURT, start, free_center=free_center, config=cfg)
    assert checked and batched is not None and serial is not None
    assert batched.params.to_dict() == serial.params.to_dict()
    assert (batched.matches, batched.inliers, batched.rmse_px, batched.coverage) == \
        (serial.matches, serial.inliers, serial.rmse_px, serial.coverage)


@pytest.mark.parametrize("free_center,fix_roll", [(False, False), (False, True), (True, False), (True, True)])
def test_batched_line_match_jacobian_and_fit_equal_scipy_differences(
    monkeypatch: pytest.MonkeyPatch, free_center: bool, fix_roll: bool,
) -> None:
    from scipy import optimize
    from scipy.optimize._numdiff import approx_derivative

    truth = CameraParams(CENTER, 0.0, 15.5, 0.0, 2300.0)
    ev = court_evidence(render(truth))
    start = CameraParams(CENTER + [0.0, 1.0, 0.0], 0.8, 15.9, 0.0, 2369.0)
    cfg = court_search.RefineConfig(fix_roll=fix_roll)
    batched = refine(ev, COURT, start, free_center=free_center, config=cfg)
    solve = optimize.least_squares
    checked = []

    def scipy_solve(fun, x0, **kwargs):
        if callable(kwargs.get("jac")) and kwargs.get("loss") == "huber":
            # Exercise the cache both at its last residual and at a point it has not evaluated.
            for delta in (0.0, -0.1, 0.1):
                p = x0.copy()
                p[0] += delta
                reference = approx_derivative(fun, p, method="2-point")
                np.testing.assert_array_equal(kwargs["jac"](p), reference, strict=True)
            checked.append(True)
            kwargs.pop("jac")  # the actual SciPy 2-point solver, not a copy of its implementation
        return solve(fun, x0, **kwargs)

    monkeypatch.setattr(optimize, "least_squares", scipy_solve)
    serial = refine(ev, COURT, start, free_center=free_center, config=cfg)
    assert checked and batched is not None and serial is not None
    assert batched.params.to_dict() == serial.params.to_dict()
    assert (batched.matches, batched.inliers, batched.rmse_px, batched.coverage) == \
        (serial.matches, serial.inliers, serial.rmse_px, serial.coverage)


@pytest.mark.parametrize("blank", [False, True])
def test_deferred_global_floor_is_unused_locally_and_identical_when_needed(
    monkeypatch: pytest.MonkeyPatch, blank: bool,
) -> None:
    cam = CameraParams(CENTER, 14.0, 15.5, 0.0, 2300.0)
    img = np.zeros((SIZE[1], SIZE[0], 3), np.uint8) if blank else render(cam)
    cfg = EvidenceConfig(exclude=[(400, 900, 1000, 160)])
    people = [(-20.0, -10.0, 30.0, 80.0), (850.0, 100.0, 990.0, 300.0)]
    eager = court_evidence(img, cfg, people)
    eager_local = court_search.restrict(eager, COURT, cam, cfg)
    floor_mask = court_search._floor_mask
    calls = []

    def count_mask(*args):
        calls.append(True)
        return floor_mask(*args)

    monkeypatch.setattr(court_search, "_floor_mask", count_mask)
    lazy = court_evidence(img, cfg, people, lines=False, defer_floor=True)
    assert not calls and lazy.floor.size == lazy.lines.size == 0
    local = court_search.restrict(lazy, COURT, cam, cfg)
    assert not calls  # the local fit does not build the global mask
    for field in ("resp", "floor", "lines", "dist"):
        np.testing.assert_array_equal(getattr(local, field), getattr(eager_local, field), strict=True)
    assert court_search.with_lines(lazy, cfg) is lazy
    assert len(calls) == 1
    for field in ("resp", "floor", "lines", "dist", "orient"):
        np.testing.assert_array_equal(getattr(lazy, field), getattr(eager, field), strict=True)
    court_search.with_lines(lazy, cfg)
    assert len(calls) == 1  # re-acquisition and cut checks share the same materialised mask


def _old_oriented_dist(ev, k: int, clip: float):  # noqa: ANN001, ANN202 - court_search before the speed-up
    h, w = ev.lines.shape
    hh, ww = h // k, w // k
    floor = ev.floor[:hh * k, :ww * k].reshape(hh, k, ww, k).mean(axis=(1, 3)) > 0.5
    lines, orient = ev.lines[:hh * k, :ww * k], ev.orient[:hh * k, :ww * k]
    nb = court_search.N_BINS
    out = np.empty((nb, hh, ww), np.float32)
    for b in range(nb):
        d = np.abs(((orient.astype(np.int32) - b + nb // 2) % nb) - nb // 2)
        mask = (lines & (d <= 1)).reshape(hh, k, ww, k).any(axis=(1, 3))
        out[b] = np.minimum(cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3) * k, clip)
    return floor, out


def _old_explained(ev, court, params, tol_px: float = 4.0) -> float:  # noqa: ANN001 - court_search before
    size = ev.size
    tol_px *= size[1] / 1080.0
    mask = np.zeros(ev.lines.shape, np.uint8)
    G = params.G(size)
    for line in court.line_segments():
        pts = np.asarray(line, float)
        dense = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:], strict=True):
            n = max(1, int(math.hypot(*(b - a)) / 0.5))
            dense.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
        uv, wz = court_search._project(G, np.asarray(dense))
        ok = (wz > 0) & np.all(np.abs(uv) < 1e5, axis=1)
        for p, q, oa, ob in zip(uv[:-1], uv[1:], ok[:-1], ok[1:], strict=True):
            if oa and ob:
                cv2.line(mask, (int(round(p[0])), int(round(p[1]))), (int(round(q[0])), int(round(q[1]))), 255, 1)
    d = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    n = int(ev.lines.sum())
    return float((d[ev.lines] <= tol_px).sum() / n) if n else 0.0


def test_scores_do_not_depend_on_the_thread_or_batch_a_candidate_is_scored_in(monkeypatch: pytest.MonkeyPatch) -> None:
    ev = court_evidence(render(CameraParams(CENTER, 14.0, 15.5, 0.0, 2300.0)))
    pr = BroadcastPrior()
    pts, dirs = court_samples(COURT, pr.step_ft)
    P, D = np.column_stack([pts, np.zeros(len(pts))]), np.column_stack([dirs, np.zeros(len(dirs))])
    floor, od = court_search._oriented_dist(ev, 4, 40.0)
    ev_pts = court_search._line_points(ev, 4, pr.inverse_points, np.random.default_rng(0))
    rng = np.random.default_rng(1)
    pans, tilts = rng.uniform(-30, 30, 1300), rng.uniform(8, 25, 1300)
    focals = np.exp(rng.uniform(math.log(500), math.log(3000), 1300))

    def scores(p: np.ndarray, t: np.ndarray, f: np.ndarray) -> np.ndarray:
        return court_search._scores(floor, od, 4, ev.size, P, D, CENTER, p, t, f, 10.0, pr.miss_weight, ev_pts,
                                    court_search._court_field(COURT), pr.inverse_weight)

    monkeypatch.setenv("COURTVISION_THREADS", "1")
    one = scores(pans, tilts, focals)
    alone = [scores(pans[i:i + 1], tilts[i:i + 1], focals[i:i + 1])[0] for i in range(0, 1300, 97)]
    shifted = scores(np.r_[pans[:37], pans], np.r_[tilts[:37], tilts], np.r_[focals[:37], focals])[37:]
    monkeypatch.setenv("COURTVISION_THREADS", "4")
    assert np.array_equal(scores(pans, tilts, focals), one)  # threads
    assert np.array_equal(shifted, one) and np.array_equal(alone, one[::97])  # another batch, or none
    found = court_search.search(ev, COURT, pr, centers=[CENTER])
    monkeypatch.setenv("COURTVISION_THREADS", "1")
    assert [(s, p.to_dict()) for s, p in court_search.search(ev, COURT, pr, centers=[CENTER])] == \
        [(s, p.to_dict()) for s, p in found]


def test_oriented_distance_and_explained_lines_match_the_direct_computation() -> None:
    cam = CameraParams(CENTER, 14.0, 15.5, 0.0, 2300.0)
    ev = court_evidence(render(cam))
    for k in (4, 2, 1):
        new, old = court_search._oriented_dist(ev, k, 10.0 * k), _old_oriented_dist(ev, k, 10.0 * k)
        assert np.array_equal(new[0], old[0]) and np.array_equal(new[1], old[1]) and new[1].dtype == old[1].dtype
    rng = np.random.default_rng(2)
    for _ in range(40):  # cameras that show part of the court, all of it, or point away
        p = CameraParams(CENTER, rng.uniform(-60, 60), rng.uniform(-5, 40), rng.uniform(-5, 5),
                         float(np.exp(rng.uniform(math.log(150), math.log(4000)))))
        assert explained(ev, COURT, p) == _old_explained(ev, COURT, p)


# ---------------------------------------------------------------- estimate_center: weak fits pooled only when they agree


def test_center_in_box_holds_game_camera_positions_only() -> None:
    assert court_search.center_in_box(CENTER)
    assert court_search.center_in_box(np.array([47.0, -62.0, 26.0]))  # every grid centre of the prior
    assert not court_search.center_in_box(np.array([-20.0, 25.0, 15.0]))  # behind a baseline, low
    assert not court_search.center_in_box(np.array([47.0, -30.0, 36.0]))  # at the sideline
    assert not court_search.center_in_box(np.array([47.0, -94.0, 80.0]))  # in the roof
    assert not court_search.center_in_box(np.array([47.0, np.nan, 36.0]))
    assert not court_search.center_in_box(np.array([47.0, -94.0]))


def _estimate(monkeypatch: pytest.MonkeyPatch, qualities: list[float | None], centre: np.ndarray,
              at_centre: dict[int, float], more: int = 0, **kwargs):
    """``estimate_center`` on one blank frame per entry of ``qualities`` (the last ``more`` of them passed as
    ``more``) with the search, the joint fit and the explained-lines share faked: frame i's own fit scores
    qualities[i] (None: no fit), ``fit_center`` returns ``centre`` and, for each pooled frame i, a camera scoring
    at_centre[i] there. Returns (centre, cameras, the frames pooled by the last joint fit, the frames searched)."""
    from types import SimpleNamespace

    from courtvision.perception import broadcast_court as module

    size = (960, 540)
    cam = CameraParams(CENTER, 0.0, 15.5, 0.0, 2300.0)  # a plausible game camera at this size
    frames = [np.full((size[1], size[0], 3), i, np.uint8) for i in range(len(qualities))]
    pooled: list[int] = []
    searched: list[int] = []
    monkeypatch.setattr(module, "court_evidence", lambda img, *a: SimpleNamespace(size=size, frame=int(img[0, 0, 0])))

    def calibrate_frame(ev, *a, **k):
        searched.append(ev.frame)
        q = qualities[ev.frame]
        return None if q is None else SimpleNamespace(quality=q, params=cam, coverage=q)

    def fit_center(evs, court, starts):
        pooled[:] = [ev.frame for ev in evs]
        return centre, [SimpleNamespace(params=cam, coverage=at_centre.get(ev.frame, 0.0)) for ev in evs]

    monkeypatch.setattr(module, "calibrate_frame", calibrate_frame)
    monkeypatch.setattr(module, "fit_center", fit_center)
    monkeypatch.setattr(module, "explained", lambda *a: 1.0)  # a frame scores its coverage
    first = frames[:len(frames) - more]
    found, cams = module.estimate_center(first, COURT, more=frames[len(first):] or None, **kwargs)
    return found, cams, pooled, searched


def test_weak_fits_that_agree_on_a_game_camera_centre_calibrate(monkeypatch: pytest.MonkeyPatch) -> None:
    found, cams, pooled, _ = _estimate(monkeypatch, [0.02, 0.12, None, 0.11, 0.09], CENTER, {1: 0.11, 3: 0.105})
    assert pooled == [1, 3]  # the weak fits, not the poorer ones
    np.testing.assert_array_equal(found, CENTER)
    assert [c is not None for c in cams] == [False, True, False, True, False]
    assert cams[1].f == pytest.approx(2300.0)  # the search ran at the frame's own width (no rescaling)


@pytest.mark.parametrize("centre,at_centre", [
    (np.array([-20.0, 25.0, 15.0]), {1: 0.12, 3: 0.12}),  # a centre no game camera has (behind the baseline)
    (CENTER, {1: 0.12, 3: 0.06}),  # one of the two no longer fits at the shared centre
])
def test_weak_fits_that_disagree_do_not_calibrate(monkeypatch: pytest.MonkeyPatch, centre: np.ndarray,
                                                  at_centre: dict[int, float]) -> None:
    found, cams, pooled, _ = _estimate(monkeypatch, [0.02, 0.12, None, 0.11], centre, at_centre)
    assert pooled == [1, 3] and found is None and cams == [None] * 4


def test_a_single_weak_fit_or_weak_pooling_switched_off_does_not_calibrate(monkeypatch: pytest.MonkeyPatch) -> None:
    found, cams, pooled, _ = _estimate(monkeypatch, [0.02, 0.14, 0.05], CENTER, {1: 0.14})
    assert found is None and pooled == []  # nothing to agree with
    found, cams, pooled, _ = _estimate(monkeypatch, [0.13, 0.14, 0.12], CENTER, {0: 0.13, 1: 0.14, 2: 0.12},
                                       weak_quality=None)
    assert found is None and pooled == [] and cams == [None] * 3


def test_two_strong_fits_are_pooled_alone_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    # the centre is not checked against the box here (unchanged behaviour), and a camera stays at half min_quality
    away = np.array([47.0, -140.0, 60.0])
    found, cams, pooled, _ = _estimate(monkeypatch, [0.2, 0.12, 0.3, 0.16], away, {0: 0.08, 2: 0.07, 3: 0.2})
    assert pooled == [0, 2, 3]
    np.testing.assert_array_equal(found, away)
    assert [c is not None for c in cams] == [True, False, False, True]


def test_more_frames_are_searched_only_for_a_game_view_the_first_frames_could_not_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # one weak fit among the first three: the next frames are searched, in order, until two agree
    found, cams, pooled, searched = _estimate(monkeypatch, [0.02, 0.12, 0.03, 0.05, 0.11, 0.3], CENTER,
                                              {1: 0.12, 4: 0.11}, more=3)
    assert searched == [0, 1, 2, 3, 4] and pooled == [1, 4]  # the last frame is never searched
    np.testing.assert_array_equal(found, CENTER)
    assert len(cams) == 6 and [c is not None for c in cams] == [False, True, False, False, True, False]
    # no fit worth pooling among the first frames (close-ups, another camera): the others are not searched
    found, cams, pooled, searched = _estimate(monkeypatch, [0.02, 0.05, 0.03, 0.2, 0.3], CENTER, {3: 0.2, 4: 0.3},
                                              more=2)
    assert searched == [0, 1, 2] and found is None and cams == [None] * 5
    # the first frames fix the centre: the others are not searched either
    found, cams, pooled, searched = _estimate(monkeypatch, [0.2, 0.3, 0.02, 0.2], CENTER, {0: 0.2, 1: 0.3}, more=1)
    assert searched == [0, 1, 2] and pooled == [0, 1] and len(cams) == 4 and cams[3] is None

