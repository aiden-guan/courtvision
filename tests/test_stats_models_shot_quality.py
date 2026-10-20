"""ShotQualityModel: anchors, monotonicity, bounds, defender / catch-and-shoot effects, fit, JSON."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from courtvision.stats.shot_quality import FEATURE_NAMES, ShotFeatures, ShotQualityConfig, ShotQualityModel
from courtvision.types import Event, EventKind


@pytest.fixture(scope="module")
def nba() -> ShotQualityModel:
    return ShotQualityModel("nba")


def _mean_xfg(m: ShotQualityModel, lo: float, hi: float, value: int = 2, **kw) -> float:
    return float(np.mean([m.xfg(ShotFeatures(d, value, **kw)) for d in np.linspace(lo, hi, 401)]))


# --------------------------------------------------------------------------- prior shape


def test_distance_anchors_reproduced_exactly(nba: ShotQualityModel) -> None:
    for d, p in nba.config.distance_anchors:
        value = 3 if d >= 23.75 else 2
        assert nba.xfg(ShotFeatures(d, value)) == pytest.approx(p, abs=1e-9)


def test_xfg_monotone_non_increasing_in_distance(nba: ShotQualityModel) -> None:
    grid = np.linspace(0.0, 94.0, 941)
    for kw in ({}, {"closest_defender_ft": 3.0}, {"closest_defender_ft": 8.0, "catch_and_shoot": True},
               {"catch_and_shoot": False}):
        for value in (2, 3):
            vals = [nba.xfg(ShotFeatures(float(d), value, **kw)) for d in grid]
            assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))


def test_outputs_clipped_to_bounds() -> None:
    rng = np.random.default_rng(3)
    m = ShotQualityModel("nba", ShotQualityConfig(shot_type_logit={"jumper": 0.0, "dunk": 12.0, "tip": -12.0}))
    assert m.xfg(ShotFeatures(0.5, 2, shot_type="dunk")) == pytest.approx(0.99)
    assert m.xfg(ShotFeatures(80.0, 3, shot_type="tip", closest_defender_ft=0.0)) == pytest.approx(0.01)
    for _ in range(300):
        f = ShotFeatures(float(rng.uniform(0, 94)), int(rng.choice([2, 3])), str(rng.choice(["jumper", "layup", "dunk", "hook"])),
                         float(rng.uniform(0, 30)), bool(rng.integers(2)))
        assert 0.01 <= m.xfg(f) <= 0.99


def test_zone_bucket_means_close_to_pbpstats_priors(nba: ShotQualityModel) -> None:
    # pbpstats 2025-26 zone FG% (rules note §4c) with distance uniform inside each zone.
    assert _mean_xfg(nba, 0.0, 4.0) == pytest.approx(0.673, abs=0.02)
    assert _mean_xfg(nba, 4.0, 14.0) == pytest.approx(0.444, abs=0.035)
    assert _mean_xfg(nba, 14.0, 22.0) == pytest.approx(0.412, abs=0.01)


def test_corner_three_beats_above_break_and_matches_prior(nba: ShotQualityModel) -> None:
    corner = nba.xfg(ShotFeatures(22.5, 3, zone="corner_three"))
    assert corner == pytest.approx(0.387, abs=1e-9)
    assert corner > nba.xfg(ShotFeatures(22.5, 3, zone="above_break_three"))
    assert nba.xfg(ShotFeatures(26.0, 3, zone="above_break_three")) == pytest.approx(0.345, abs=1e-9)


# --------------------------------------------------------------------------- defender and catch/pull


def test_defender_shift_monotone_and_reproduces_tracking_splits(nba: ShotQualityModel) -> None:
    shifts = [nba.defender_shift(d) for d in np.linspace(0, 20, 201)]
    assert all(b >= a - 1e-12 for a, b in zip(shifts, shifts[1:]))
    ref = math.log(0.36 / 0.64)
    edges = nba.config.defender_bucket_edges_ft
    for (lo, hi), target in zip(zip(edges[:-1], edges[1:]), nba.config.defender_bucket_fg3):
        mean = np.mean([1.0 / (1.0 + math.exp(-(ref + nba.defender_shift(d)))) for d in np.linspace(lo, hi, 81)])
        assert mean == pytest.approx(target, abs=1e-6)


def test_tight_defense_lowers_xfg_for_twos_and_threes(nba: ShotQualityModel) -> None:
    for d, v in ((2.0, 2), (16.0, 2), (25.0, 3)):
        tight = nba.xfg(ShotFeatures(d, v, closest_defender_ft=1.0))
        unknown = nba.xfg(ShotFeatures(d, v))
        open_ = nba.xfg(ShotFeatures(d, v, closest_defender_ft=9.0))
        assert tight < unknown < open_


def test_catch_and_shoot_beats_pull_up_for_jumpers_only(nba: ShotQualityModel) -> None:
    cs = nba.xfg(ShotFeatures(24.5, 3, catch_and_shoot=True))
    pu = nba.xfg(ShotFeatures(24.5, 3, catch_and_shoot=False))
    unk = nba.xfg(ShotFeatures(24.5, 3))
    assert pu < unk < cs
    assert pu < 0.40  # a typical pull-up three never prices above ~.40 (rules note §6.5)
    # no effect inside 10 ft or for non-jumpers
    assert nba.xfg(ShotFeatures(6.0, 2, catch_and_shoot=True)) == nba.xfg(ShotFeatures(6.0, 2, catch_and_shoot=False))
    assert nba.xfg(ShotFeatures(12.0, 2, "hook", catch_and_shoot=True)) == nba.xfg(ShotFeatures(12.0, 2, "hook"))


def test_shot_types_order_and_rim_mix(nba: ShotQualityModel) -> None:
    dunk, layup, tip = (nba.xfg(ShotFeatures(1.0, 2, t)) for t in ("dunk", "layup", "tip"))
    assert dunk > layup > tip
    # 6.1% dunks of 27.7% at-rim FGA (rules note §4b/§4c): the dunk/layup mix keeps the <4 ft prior
    share = 0.061 / 0.277
    mix = share * _mean_xfg(nba, 0, 4, shot_type="dunk") + (1 - share) * _mean_xfg(nba, 0, 4, shot_type="layup")
    assert mix == pytest.approx(0.673, abs=0.02)


# --------------------------------------------------------------------------- free throws, leagues, points


def test_free_throws_use_league_ft_pct() -> None:
    assert ShotQualityModel("nba").xfg(ShotFeatures(13.75, 1, "free_throw")) == pytest.approx(0.783)
    assert ShotQualityModel("ncaa_m").xfg(ShotFeatures(13.75, 1)) == pytest.approx(0.71)
    ft = ShotFeatures(13.75, 1, "free_throw", closest_defender_ft=0.5)
    assert ShotQualityModel("nba").expected_points(ft) == pytest.approx(0.783)


def test_expected_points_is_xfg_times_value(nba: ShotQualityModel) -> None:
    for f in (ShotFeatures(2.0, 2, "layup"), ShotFeatures(25.0, 3, closest_defender_ft=5.0)):
        assert nba.expected_points(f) == pytest.approx(nba.xfg(f) * f.value)


def test_non_nba_leagues_are_unverified_placeholders(nba: ShotQualityModel) -> None:
    ncaa = ShotQualityModel("ncaa_m")
    assert nba.prior_status == "sourced" and ncaa.prior_status == "unverified"
    f = ShotFeatures(22.5, 3, closest_defender_ft=4.0)
    assert ncaa.xfg(f) < nba.xfg(f)
    with pytest.raises(ValueError):
        ShotQualityModel("euroleague")


def test_invalid_features_raise(nba: ShotQualityModel) -> None:
    with pytest.raises(ValueError):
        nba.xfg(ShotFeatures(10.0, 4))
    with pytest.raises(ValueError):
        nba.xfg(ShotFeatures(float("nan"), 2))


def test_features_from_event_attrs() -> None:
    e = Event("e1", EventKind.SHOT_ATTEMPT, 1.0, 30, value=3,
              attrs={"distance_ft": 24.1, "shot_type": "jumper", "closest_defender_ft": 3.2,
                     "catch_and_shoot": True, "zone": "above_break_three"})
    f = ShotFeatures.from_event(e)
    assert (f.distance_ft, f.value, f.closest_defender_ft, f.catch_and_shoot, f.zone) == (24.1, 3, 3.2, True,
                                                                                          "above_break_three")
    with pytest.raises(ValueError):
        ShotFeatures.from_event(Event("e2", EventKind.SHOT_ATTEMPT, 1.0, 30, value=2))


# --------------------------------------------------------------------------- fit


def _synthetic_shots(rng: np.random.Generator, n: int) -> list[ShotFeatures]:
    out = []
    for _ in range(n):
        d = float(rng.uniform(0.0, 28.0))
        value = 3 if d > 23.75 or (d > 22.2 and rng.random() < 0.5) else 2
        zone = "corner_three" if value == 3 and d < 23.75 else None
        if d < 4.0:
            st = str(rng.choice(["jumper", "layup", "dunk", "tip"]))
        elif d < 10.0:
            st = str(rng.choice(["jumper", "hook"]))
        else:
            st = "jumper"
        cd = None if rng.random() < 0.2 else float(rng.uniform(0.0, 10.0))
        cs = None if rng.random() < 0.2 else bool(rng.integers(2))
        out.append(ShotFeatures(d, value, st, cd, cs, zone))
    return out


def test_fit_recovers_known_coefficients() -> None:
    rng = np.random.default_rng(20261020)
    m = ShotQualityModel("nba")
    feats = _synthetic_shots(rng, 30000)
    true = np.array([0.2, -0.15, 0.1, 0.3, 0.15, 0.4, -0.2, 0.3, -0.25, 0.2])
    z = np.array([m.prior_logit(f) + true @ m.residual_features(f) for f in feats])
    made = list(rng.random(len(z)) < 1.0 / (1.0 + np.exp(-z)))
    m.fit(feats, made, l2=1.0)
    assert m.fit_info["converged"] and m.fit_info["n"] == 30000
    assert m.fit_info["logloss_fit"] < m.fit_info["logloss_prior"]
    np.testing.assert_allclose(m.coef, true, atol=0.12)


def test_fit_with_strong_l2_stays_at_prior_and_skips_free_throws() -> None:
    rng = np.random.default_rng(5)
    m = ShotQualityModel("nba")
    feats = _synthetic_shots(rng, 40) + [ShotFeatures(13.75, 1, "free_throw")] * 5
    made = [bool(x) for x in rng.integers(0, 2, len(feats))]
    m.fit(feats, made, l2=1e6)
    assert m.fit_info["n_free_throws_skipped"] == 5
    assert np.max(np.abs(m.coef)) < 1e-3
    with pytest.raises(ValueError):
        m.fit(feats, made[:-1])
    with pytest.raises(ValueError):
        m.fit([ShotFeatures(13.75, 1)], [True])


# --------------------------------------------------------------------------- JSON


def test_json_round_trip_preserves_predictions() -> None:
    rng = np.random.default_rng(11)
    m = ShotQualityModel("fiba")
    feats = _synthetic_shots(rng, 2000)
    m.fit(feats, [bool(x) for x in rng.integers(0, 2, len(feats))], l2=2.0)
    doc = json.loads(json.dumps(m.to_json()))
    m2 = ShotQualityModel.from_json(doc)
    assert m2.league == "fiba" and m2.fit_info == m.fit_info
    assert doc["feature_names"] == list(FEATURE_NAMES)
    np.testing.assert_array_equal(m2.coef, m.coef)
    for f in feats[:200] + [ShotFeatures(13.75, 1)]:
        assert m2.xfg(f) == m.xfg(f)


def test_from_json_rejects_bad_documents(nba: ShotQualityModel) -> None:
    doc = nba.to_json()
    with pytest.raises(ValueError):
        ShotQualityModel.from_json({**doc, "feature_names": ["intercept"]})
    with pytest.raises(ValueError):
        ShotQualityModel.from_json({**doc, "config": {**doc["config"], "bogus": 1}})
    with pytest.raises(ValueError):
        ShotQualityModel.from_json({**doc, "coef": [0.0, 1.0]})
