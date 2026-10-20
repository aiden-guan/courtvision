"""Tests for courtvision.training.fit_shot_model (CLI on synthetic CSV, calibration metrics, splits)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from courtvision.events.calibration import ece as ece_point
from courtvision.stats.shot_quality import ShotFeatures, ShotQualityModel
from courtvision.training.fit_shot_model import (
    SHOT_COLUMNS,
    _ece_fast,
    calibration_report,
    main,
    read_shots_csv,
    split_by_game,
)

TRUE_SHIFT = 0.35  # our league shoots better than the NBA prior by this many log-odds


def _write_csv(path: Path, n_games: int = 40, per_game: int = 60, seed: int = 0, n_ft: int = 30) -> int:
    rng = np.random.default_rng(seed)
    prior = ShotQualityModel("nba")
    rows = []
    for g in range(n_games):
        for _ in range(per_game):
            d = float(rng.uniform(0.5, 28.0))
            value = 3 if d > 23.75 else 2
            kind = "dunk" if d < 3 and rng.random() < 0.3 else ("layup" if d < 4 else "jumper")
            f = ShotFeatures(d, value, kind, float(rng.uniform(0.5, 10.0)), bool(rng.random() < 0.5) if kind == "jumper" else None,
                             "corner_three" if value == 3 and rng.random() < 0.25 else None)
            p = 1.0 / (1.0 + math.exp(-(prior.prior_logit(f) + TRUE_SHIFT)))
            rows.append({"distance_ft": round(d, 2), "value": value, "made": int(rng.random() < p), "shot_type": kind,
                         "closest_defender_ft": round(f.closest_defender_ft or 0, 2),
                         "catch_and_shoot": "" if f.catch_and_shoot is None else int(f.catch_and_shoot),
                         "zone": f.zone or "", "game_id": f"g{g:03d}"})
    for i in range(n_ft):
        rows.append({"distance_ft": 15.0, "value": 1, "made": int(rng.random() < 0.78), "shot_type": "free_throw",
                     "closest_defender_ft": "", "catch_and_shoot": "", "zone": "", "game_id": f"g{i % n_games:03d}"})
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(SHOT_COLUMNS))
        w.writeheader()
        w.writerows(rows)
    return n_ft


def test_cli_fits_and_reports(tmp_path: Path) -> None:
    csv_path, out = tmp_path / "shots.csv", tmp_path / "model.json"
    n_ft = _write_csv(csv_path)
    assert main(["--csv", str(csv_path), "--out", str(out), "--n-boot", "200"]) == 0
    model = ShotQualityModel.from_json(json.loads(out.read_text()))
    assert model.coef[0] == pytest.approx(TRUE_SHIFT, abs=0.2)  # intercept recovers the shift
    report = json.loads((tmp_path / "model.report.json").read_text())
    assert report["split"] == "by_game" and report["n_free_throws_skipped"] == n_ft
    assert set(report["games_train"]).isdisjoint(report["games_holdout"]) and report["games_holdout"]
    fitted, prior = report["fitted"], report["prior"]
    assert fitted["n"] == report["n_holdout"] == sum(r["n"] for r in fitted["reliability"])
    assert len(fitted["reliability"]) == 15 and fitted["bins"] == 15
    lo, hi = fitted["ece_ci95"]
    assert 0.0 <= lo <= hi and fitted["bootstrap_unit"] == "game"
    assert fitted["log_loss"] < prior["log_loss"] and fitted["brier"] <= prior["brier"]
    assert fitted["ece"] < prior["ece"]
    assert {"mce", "base_rate", "mean_p", "target_ece", "meets_target"} <= set(fitted)


def test_cli_report_path_and_determinism(tmp_path: Path) -> None:
    csv_path = tmp_path / "shots.csv"
    _write_csv(csv_path, n_games=10, per_game=40)
    reps = []
    for i in range(2):
        rep = tmp_path / f"r{i}.json"
        assert main(["--csv", str(csv_path), "--out", str(tmp_path / f"m{i}.json"), "--report", str(rep), "--n-boot", "50"]) == 0
        reps.append(json.loads(rep.read_text()))
    assert reps[0]["fitted"] == reps[1]["fitted"]


def test_cli_rejects_bad_csv(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("distance_ft,value\n10,2\n")
    assert main(["--csv", str(bad), "--out", str(tmp_path / "m.json")]) == 2
    worse = tmp_path / "worse.csv"
    worse.write_text("distance_ft,value,made\n10,4,1\n")
    with pytest.raises(ValueError, match="worse.csv:2"):
        read_shots_csv(worse)


def test_in_sample_without_game_ids(tmp_path: Path) -> None:
    path = tmp_path / "s.csv"
    path.write_text("distance_ft,value,made\n" + "".join(f"{d},2,{d % 2}\n" for d in range(1, 30)))
    out = tmp_path / "m.json"
    assert main(["--csv", str(path), "--out", str(out), "--n-boot", "20"]) == 0
    rep = json.loads((tmp_path / "m.report.json").read_text())
    assert rep["split"] == "in_sample" and rep["n_holdout"] == 0


def test_calibration_report_hand_example() -> None:
    rep = calibration_report([0.1, 0.1, 0.9, 0.9], [0, 1, 1, 1], n_boot=0)
    assert rep["ece"] == pytest.approx(0.5 * 0.4 + 0.5 * 0.1)
    assert rep["mce"] == pytest.approx(0.4)
    assert rep["brier"] == pytest.approx((0.01 + 0.81 + 0.01 + 0.01) / 4)
    assert rep["log_loss"] == pytest.approx(-(math.log(0.9) * 3 + math.log(0.1)) / 4, rel=1e-4)
    assert rep["ece_ci95"] is None
    filled = [r for r in rep["reliability"] if r["n"]]
    assert [(r["bin"], r["n"], r["frac_pos"]) for r in filled] == [(1, 2, 0.5), (13, 2, 1.0)]


def test_calibrated_predictions_have_small_ece() -> None:
    rng = np.random.default_rng(4)
    p = rng.uniform(0.05, 0.95, 4000)
    y = rng.random(4000) < p
    rep = calibration_report(p, y, n_boot=200, seed=1)
    assert rep["ece"] < 0.04 and rep["meets_target"] in (True, False)
    lo, hi = rep["ece_ci95"]
    assert lo <= rep["ece"] <= hi + 0.01 and rep["bootstrap_unit"] == "event"
    assert calibration_report(p, y, n_boot=200, seed=1)["ece_ci95"] == rep["ece_ci95"]  # seeded


def test_fast_ece_matches_shared_definition() -> None:
    rng = np.random.default_rng(2)
    for _ in range(5):
        p = rng.random(500)
        y = (rng.random(500) < p).astype(float)
        assert _ece_fast(p, y, 15) == pytest.approx(ece_point(p, y, 15), abs=1e-12)


def test_split_by_game() -> None:
    games = [f"g{i % 7}" for i in range(70)]
    train, test = split_by_game(games, 0.3, seed=0)
    assert sorted(train + test) == list(range(70))
    assert {games[i] for i in train}.isdisjoint({games[i] for i in test})
    assert len({games[i] for i in test}) == 2
    assert split_by_game(["g1"] * 5, 0.5) == (list(range(5)), [])
    assert split_by_game(["g1", None], 0.5) == ([0, 1], [])


def test_calibration_report_rejects_non_finite() -> None:
    # NaN used to fall into a garbage bin and turn every metric (and the JSON report) into NaN.
    with pytest.raises(ValueError, match="non-finite"):
        calibration_report([0.2, float("nan"), 0.9], [0, 1, 1], n_boot=10)
    with pytest.raises(ValueError, match="non-finite"):
        calibration_report([0.2, float("inf")], [0, 1], n_boot=0)


def test_headline_ece_interval_covers_the_population_ece() -> None:
    # Binned ECE is biased upward at a few hundred events and a bootstrap resample more so: the percentile
    # interval covered the population ECE in ~20% of replicates here (0% for a calibrated model). The
    # headline interval is the bias-corrected (basic) one.
    rng = np.random.default_rng(11)

    def sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        p = rng.uniform(0.02, 0.98, n)
        q = 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + 0.4)))  # true odds 1.5x the stated odds
        return p, (rng.random(n) < q).astype(float)

    truth = _ece_fast(*sample(400_000), 15)
    hits = {"basic": 0, "percentile": 0}
    for r in range(40):
        p, y = sample(300)
        rep = calibration_report(p, y, n_boot=200, seed=r, groups=[f"g{i % 10}" for i in range(300)])
        assert rep["ci_method"] == "basic" and rep["bootstrap_unit"] == "game"
        lo, hi = rep["ece_ci95"]
        plo, phi = rep["ece_ci95_percentile"]
        assert 0.0 <= lo <= hi and hi == pytest.approx(max(0.0, 2 * rep["ece"] - plo), abs=2e-5)
        assert rep["ece_debiased"] == pytest.approx(max(0.0, rep["ece"] - rep["ece_bootstrap_bias"]), abs=2e-5)
        hits["basic"] += lo <= truth <= hi
        hits["percentile"] += plo <= truth <= phi
    assert hits["basic"] >= 30 and hits["basic"] > hits["percentile"] + 10, hits
