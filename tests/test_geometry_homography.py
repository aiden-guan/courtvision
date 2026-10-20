"""Homography: exact DLT, noise, RANSAC with outliers, degenerate input, mapping, JSON, Calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from courtvision.geometry.court import court_for
from courtvision.geometry.homography import Calibration, Homography, HomographyConfig, fit_landmarks
from courtvision.types import BBox, RimObservation

# Court ft -> image px, a broadcast-like perspective (w in [1, 1.34] over an NBA court).
G_TRUE = np.array([[9.0, 1.2, 150.0], [0.3, -5.0, 600.0], [0.0004, 0.006, 1.0]])
NBA = court_for("nba")


def _to_image(court_xy: np.ndarray) -> np.ndarray:
    p = np.hstack([court_xy, np.ones((len(court_xy), 1))]) @ G_TRUE.T
    return p[:, :2] / p[:, 2:]


def _landmark_arrays() -> tuple[list[str], np.ndarray]:
    lm = NBA.landmarks()
    names = sorted(lm)
    return names, np.array([lm[n] for n in names])


def _grid() -> np.ndarray:
    xs, ys = np.meshgrid(np.linspace(0, 94, 12), np.linspace(0, 50, 7))
    return np.column_stack([xs.ravel(), ys.ravel()])


@pytest.mark.parametrize("thresh", [1.0, None])
def test_exact_fit_recovers_mapping(thresh) -> None:
    _, court = _landmark_arrays()
    img = _to_image(court)
    hom, mask, rmse = Homography.fit(img, court, ransac_thresh_ft=thresh)
    assert mask.dtype == bool and mask.all()
    assert rmse < 1e-9
    grid = _grid()
    assert np.abs(hom.image_to_court(_to_image(grid)) - grid).max() < 1e-8
    assert np.abs(hom.court_to_image(grid) - _to_image(grid)).max() < 1e-7
    expected = np.linalg.inv(G_TRUE)
    assert np.allclose(hom.H / hom.H[2, 2], expected / expected[2, 2], atol=1e-10)


def test_exact_fit_with_four_points() -> None:
    court = np.array([[0.0, 0.0], [94.0, 0.0], [94.0, 50.0], [0.0, 50.0]])
    hom, mask, rmse = Homography.fit(_to_image(court), court)
    assert mask.all() and rmse < 1e-9
    assert np.abs(hom.image_to_court(_to_image(_grid())) - _grid()).max() < 1e-8


def test_noisy_fit_is_accurate() -> None:
    rng = np.random.default_rng(7)
    _, court = _landmark_arrays()
    img = _to_image(court) + rng.normal(0.0, 0.5, size=(len(court), 2))
    hom, mask, rmse = Homography.fit(img, court, ransac_thresh_ft=1.0)
    assert mask.all()
    assert 0.0 < rmse < 0.3
    err = np.linalg.norm(hom.image_to_court(_to_image(_grid())) - _grid(), axis=1)
    assert err.mean() < 0.15 and err.max() < 0.5


def test_ransac_rejects_30_percent_outliers() -> None:
    rng = np.random.default_rng(11)
    n, n_out = 60, 18
    court = np.column_stack([rng.uniform(0, 94, n), rng.uniform(0, 50, n)])
    img = _to_image(court) + rng.normal(0.0, 0.2, size=(n, 2))
    outlier = np.zeros(n, dtype=bool)
    outlier[rng.choice(n, n_out, replace=False)] = True
    angles = rng.uniform(0, 2 * np.pi, n_out)
    img[outlier] += rng.uniform(40, 150, n_out)[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])

    hom, mask, rmse = Homography.fit(img, court, ransac_thresh_ft=1.0)
    assert np.array_equal(mask, ~outlier)
    assert rmse < 0.1
    err = np.linalg.norm(hom.image_to_court(_to_image(_grid())) - _grid(), axis=1)
    assert err.max() < 0.2

    plain, _, _ = Homography.fit(img, court, ransac_thresh_ft=None)
    plain_err = np.linalg.norm(plain.image_to_court(_to_image(_grid())) - _grid(), axis=1)
    assert plain_err.max() > 1.0  # without RANSAC the outliers wreck the fit


def test_ransac_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    court = np.column_stack([rng.uniform(0, 94, 30), rng.uniform(0, 50, 30)])
    img = _to_image(court)
    img[:8] += 80.0
    a = Homography.fit(img, court)
    b = Homography.fit(img, court)
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[0].H, b[0].H)


@pytest.mark.parametrize(
    ("img", "court", "match"),
    [
        ([[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 0], [0, 1]], ">= 4"),
        ([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], [[0, 0], [1, 0], [0, 1], [1, 1], [2, 3]], "collinear"),
        ([[0, 0], [1, 0], [0, 1], [1, 1]], [[0, 0], [10, 0], [20, 0], [30, 0]], "collinear"),
        ([[0, 0], [1, 0], [2, 0], [0, 1]], [[0, 0], [1, 0], [2, 0], [0, 1]], "degenerate"),
        ([[0, 0], [1, 0], [0, 1], [1, 1]], [[0, 0], [10, 0], [20, 0], [0, 10]], "degenerate"),
        ([[5, 5], [5, 5], [5, 5], [5, 5]], [[0, 0], [1, 0], [0, 1], [1, 1]], "coincident"),
        ([[0, 0], [1, 0], [0, 1], [1, 1]], [[0, 0], [1, 0], [0, 1]], "court points"),
        ([[0, 0], [1, 0], [0, 1], [np.nan, 1]], [[0, 0], [1, 0], [0, 1], [1, 1]], "non-finite"),
        ([0, 0, 1, 0], [0, 0, 1, 0], r"\(N, 2\)"),
    ],
)
def test_degenerate_and_invalid_input(img, court, match) -> None:
    for thresh in (1.0, None):
        with pytest.raises(ValueError, match=match):
            Homography.fit(np.array(img, dtype=float), np.array(court, dtype=float), ransac_thresh_ft=thresh)


def test_degenerate_input_with_ransac_extra_points() -> None:
    xs = np.linspace(0, 94, 10)
    court = np.column_stack([xs, np.zeros_like(xs)])  # one sideline: all collinear
    with pytest.raises(ValueError, match="collinear"):
        Homography.fit(_to_image(court), court)


def test_mapping_shapes_and_points_at_infinity() -> None:
    hom = Homography(np.linalg.inv(G_TRUE))
    one = hom.image_to_court(np.array([600.0, 400.0]))
    assert one.shape == (2,)
    many = hom.image_to_court([[600.0, 400.0], [700.0, 300.0]])
    assert many.shape == (2, 2) and np.allclose(many[0], one)
    assert hom.image_to_court(np.zeros((0, 2))).shape == (0, 2)
    assert hom.court_to_image((47.0, 25.0)).shape == (2,)
    # A pixel on the vanishing line (w = 0) maps to infinity -> nan.
    h = hom.H
    u = 600.0
    v = -(h[2, 0] * u + h[2, 2]) / h[2, 1]
    out = hom.image_to_court([[u, v], [600.0, 400.0]])
    assert np.isnan(out[0]).all() and np.isfinite(out[1]).all()
    with pytest.raises(ValueError):
        hom.image_to_court(np.zeros((3, 3)))


def test_front_only_rejects_points_beyond_the_horizon() -> None:
    court = np.array([[0.0, 0.0], [94.0, 0.0], [94.0, 50.0], [0.0, 50.0], [47.0, 25.0]])
    hom, _, _ = Homography.fit(_to_image(court), court)
    h = hom.H
    u = 600.0
    v_horizon = -(h[2, 0] * u + h[2, 2]) / h[2, 1]
    beyond = [u, v_horizon - 50.0 * np.sign(h[2, 1])]  # w < 0 side of the vanishing line
    assert np.isfinite(hom.image_to_court(beyond)).all()  # mathematically a (mirrored) point
    assert np.isnan(hom.image_to_court(beyond, front_only=True)).all()
    assert np.isfinite(hom.image_to_court(_to_image(court), front_only=True)).all()


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        Homography(np.eye(2))
    with pytest.raises(ValueError):
        Homography(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        Homography(np.full((3, 3), np.nan))


def test_json_round_trip() -> None:
    _, court = _landmark_arrays()
    hom, _, _ = Homography.fit(_to_image(court), court)
    restored = Homography.from_json(json.loads(json.dumps(hom.to_json())))
    assert np.array_equal(restored.H, hom.H)
    with pytest.raises(ValueError):
        Homography.from_json({})


def test_fit_landmarks_by_name() -> None:
    names, court = _landmark_arrays()
    hom, mask, rmse = fit_landmarks(NBA, _to_image(court), names)
    assert mask.all() and rmse < 1e-9
    with pytest.raises(ValueError, match="unknown landmark"):
        fit_landmarks(NBA, _to_image(court[:4]), ["L_corner_bottom", "R_corner_top", "center", "nope"])


def test_config_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2

    import courtvision.geometry.homography as homography_mod

    rng = np.random.default_rng(5)
    court = np.column_stack([rng.uniform(0, 94, 20), rng.uniform(0, 50, 20)])
    img = _to_image(court)
    img[:3] += 60.0
    seen: dict = {}
    real_find = cv2.findHomography

    def spy_find(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return real_find(*args, **kwargs)

    refine_calls: list[int] = []
    real_refine = homography_mod._refine

    def spy_refine(*args, **kwargs):
        refine_calls.append(1)
        return real_refine(*args, **kwargs)

    monkeypatch.setattr(cv2, "findHomography", spy_find)
    monkeypatch.setattr(homography_mod, "_refine", spy_refine)
    cfg = HomographyConfig(refine_inliers=False, ransac_max_iters=500, ransac_confidence=0.9)
    _, mask, _ = Homography.fit(img, court, ransac_thresh_ft=0.75, config=cfg)
    assert mask.sum() == 17
    assert seen["args"][2] == cv2.RANSAC and seen["args"][3] == 0.75  # threshold in court ft
    assert seen["kwargs"] == {"maxIters": 500, "confidence": 0.9}
    assert refine_calls == []
    Homography.fit(img, court, config=HomographyConfig(refine_inliers=True))
    assert refine_calls == [1]


def test_calibration_save_load(tmp_path: Path) -> None:
    _, court = _landmark_arrays()
    hom, _, _ = Homography.fit(_to_image(court), court)
    rims = {"left": BBox(100.0, 200.0, 120.0, 206.0), "right": BBox(1100.0, 200.0, 1120.0, 206.0)}
    cal = Calibration(homography=hom, image_size=(1280, 720), league="fiba", rims=rims)
    path = tmp_path / "cal.json"
    cal.save(path)
    raw = json.loads(path.read_text())
    assert raw["rims"]["left"] == {"x1": 100.0, "y1": 200.0, "x2": 120.0, "y2": 206.0}
    assert raw["image_size"] == [1280, 720] and raw["league"] == "fiba"
    loaded = Calibration.load(path)
    assert loaded.rims == rims and loaded.image_size == (1280, 720) and loaded.league == "fiba"
    assert np.array_equal(loaded.homography.H, hom.H)
    assert loaded.rim_observations() == [RimObservation("left", rims["left"]), RimObservation("right", rims["right"])]


def test_calibration_defaults_and_validation() -> None:
    hom = Homography(np.linalg.inv(G_TRUE))
    cal = Calibration.from_dict({"homography": hom.to_json(), "image_size": [640, 360], "rims": {"left": [1, 2, 3, 4]}})
    assert cal.league == "nba" and cal.rims["left"] == BBox(1.0, 2.0, 3.0, 4.0)
    assert Calibration(homography=hom, image_size=(640, 360)).rims == {}
    with pytest.raises(ValueError):
        Calibration(homography=hom, image_size=(640, 360), rims={"middle": BBox(0, 0, 1, 1)})


# ---------------------------------------------------------------------------- review regressions


def test_negative_threshold_is_rejected_even_with_four_points() -> None:
    # Before: the threshold was validated only on the RANSAC branch, so n == 4 accepted -1.
    court = np.array([[0.0, 0.0], [94.0, 0.0], [94.0, 50.0], [0.0, 50.0]])
    for thresh in (-1.0, 0.0):
        with pytest.raises(ValueError, match="ransac_thresh_ft"):
            Homography.fit(_to_image(court), court, ransac_thresh_ft=thresh)


def test_dlt_on_coincident_subset_raises_value_error() -> None:
    from courtvision.geometry.homography import _dlt

    # Before: an inlier subset of identical points hit a ZeroDivisionError in the normaliser.
    same = np.full((4, 2), 5.0)
    other = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="coincident"):
        _dlt(same, other, 1e-9)
    with pytest.raises(ValueError, match="coincident"):
        _dlt(other, same, 1e-9)


def test_refine_keeps_ransac_solution_when_rescored_set_is_degenerate() -> None:
    from courtvision.geometry.homography import _refine

    # Points on y = 0 fit H exactly; the off-line ones miss by 10-30 ft, so re-scoring keeps only
    # four collinear points. Before: the refit raised and aborted an otherwise valid fit.
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [0.0, 10.0], [10.0, 20.0], [30.0, 30.0]])
    H = np.diag([1.0, 2.0, 1.0])
    mask = np.ones(len(pts), dtype=bool)
    H_out, mask_out = _refine(H, pts, pts, mask, 1.0, 1e-9)
    assert H_out is H and np.array_equal(mask_out, mask)


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"homography": {"H": np.eye(3).tolist()}},
        {"homography": {"H": np.eye(3).tolist()}, "image_size": [640]},
        {"homography": {"H": np.eye(3).tolist()}, "image_size": [0, 360]},
        {"homography": {}, "image_size": [640, 360]},
        {"homography": {"H": np.eye(3).tolist()}, "image_size": [640, 360], "rims": {"left": {"x1": 1}}},
        {"homography": {"H": np.eye(3).tolist()}, "image_size": [640, 360], "rims": {"middle": [1, 2, 3, 4]}},
    ],
)
def test_calibration_from_malformed_dict_raises_value_error(bad) -> None:
    # Before: missing keys surfaced as KeyError / TypeError (a 500 instead of a 400 at the server).
    with pytest.raises(ValueError):
        Calibration.from_dict(bad)
