"""Tests for courtvision.training.fit_cue_models (status rules, ECE + CI, rule-of-three gates, sim datasets)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from courtvision.events.calibration import CueModelSet, default_cue_models
from courtvision.events.engine import EventEngine
from courtvision.sim.simulator import GameSimulator, SimConfig
from courtvision.training.fit_cue_models import (
    CueFitConfig,
    build_cue_dataset,
    certify_gate,
    error_upper_bound,
    fit_cue_kind,
    fit_cue_models,
    load_cue_rows,
    main,
    rule_of_three_n,
    save_cue_models,
    write_cue_rows,
)
from courtvision.types import Event, EventKind, EventStatus, Team

# ---------------------------------------------------------------------------------------------- certification math


@pytest.mark.parametrize("eps,n", [(0.02, 150), (0.01, 300), (0.05, 60), (0.03, 100), (0.10, 30), (0.08, 38)])
def test_rule_of_three_n(eps: float, n: int) -> None:
    assert rule_of_three_n(eps) == n
    assert 3.0 / n <= eps + 1e-12 and 3.0 / (n - 1) > eps


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_rule_of_three_rejects(bad: float) -> None:
    with pytest.raises(ValueError):
        rule_of_three_n(bad)


def test_error_upper_bound() -> None:
    assert error_upper_bound(0, 150) == pytest.approx(0.02)
    assert error_upper_bound(0, 0) == 1.0
    one = error_upper_bound(1, 150)
    assert 0.02 < one < 0.05  # Clopper-Pearson 95% one-sided, 1 error in 150 ~ 0.031
    assert error_upper_bound(5, 5) == 1.0


def test_certify_gate_rules() -> None:
    p, y = [0.99] * 150, [1] * 150
    ok = certify_gate(p, y, 0.95, 0.02, "platt", "make", "gold")
    assert ok.certified and ok.valid_for_live and ok.n_accepted == 150 and ok.n_required == 150
    assert ok.error_upper_95 == pytest.approx(0.02)
    short = certify_gate(p[:149], y[:149], 0.95, 0.02, "platt", "make")
    assert not short.certified and "150 needed" in short.reason
    one_err = certify_gate(p + [0.97], y + [0], 0.95, 0.02, "isotonic", "make")
    assert not one_err.certified and one_err.n_errors == 1
    below = certify_gate(p + [0.90] * 10, y + [0] * 10, 0.95, 0.02, "platt", "make")
    assert below.certified and below.n_accepted == 150  # errors below the threshold are not accepted events
    cold = certify_gate(p, y, 0.95, 0.02, "uncalibrated", "make")
    assert not cold.certified and "hedge-only" in cold.reason
    sim = certify_gate(p, y, 0.95, 0.02, "platt", "make", "sim")
    assert sim.certified and not sim.valid_for_live and "sim-derived" in sim.reason


# ---------------------------------------------------------------------------------------------- fitting

_TRUE = {"crossed_inside": 2.5, "drop_in_band": 1.0, "drop_is_freefall": -3.0, "net_motion_z": 0.6}


def _make_rows(n_games: int, per_game: int, seed: int = 0, source: str = "gold") -> list[dict]:
    """Make-cue rows whose labels follow a logistic model different from the cold-start weights."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        for k in range(per_game):
            f = {"crossed_inside": int(rng.random() < 0.55), "drop_in_band": int(rng.random() < 0.5),
                 "drop_is_freefall": int(rng.random() < 0.1), "net_motion_z": float(rng.uniform(0, 3)),
                 "occluded_frames": int(rng.integers(0, 5))}
            z = -1.8 + sum(w * f[name] for name, w in _TRUE.items())
            rows.append({"kind": "make", "features": f, "label": bool(rng.random() < 1 / (1 + math.exp(-z))),
                         "game_id": f"g{g:03d}", "calibration_source": source, "event_id": f"e{g:03d}{k:03d}"})
    return rows


def test_too_few_rows_stay_uncalibrated() -> None:
    model, rep = fit_cue_kind(default_cue_models()["make"], _make_rows(10, 10), CueFitConfig(n_boot=50))
    assert rep.status == "uncalibrated" == model.status
    assert not rep.gate.certified and rep.n_holdout > 0


def _true_p(rows: list[dict]) -> list[float]:
    return [1 / (1 + math.exp(-(-1.8 + sum(w * r["features"][n] for n, w in _TRUE.items())))) for r in rows]


@pytest.mark.parametrize("n_games,per_game,seed,status,min_train", [(30, 20, 1, "platt", 300), (30, 50, 2, "isotonic", 1000)])
def test_platt_then_isotonic(n_games: int, per_game: int, seed: int, status: str, min_train: int) -> None:
    base = default_cue_models()["make"]
    rows = _make_rows(n_games, per_game, seed=seed)
    model, rep = fit_cue_kind(base, rows, CueFitConfig(n_boot=100))
    assert rep.status == status == model.status and rep.n_train >= min_train and rep.n_ambiguous_train >= 50
    assert base.status == "uncalibrated"  # the input model is not mutated
    held = set(rep.games_holdout)
    assert held and held.isdisjoint({r["game_id"] for r in rows} - held)
    test = [r for r in rows if r["game_id"] in held]
    cal = rep.calibration
    assert cal["n"] == rep.n_holdout == len(test) and len(cal["reliability"]) == 15 and cal["bootstrap_unit"] == "game"
    lo, hi = cal["ece_ci95"]
    assert 0.0 <= lo <= hi
    # 15-bin ECE is biased upward on a few hundred events; the true probabilities show the same bias.
    from courtvision.training.fit_shot_model import calibration_report

    oracle = calibration_report(_true_p(test), [r["label"] for r in test], n_boot=0)["ece"]
    assert cal["ece"] <= oracle + 0.03
    y = np.array([r["label"] for r in test], dtype=float)

    def ll(p: np.ndarray) -> float:
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    assert ll(model.predict_many([r["features"] for r in test])) < ll(base.predict_many([r["features"] for r in test]))


def test_fit_set_save_and_load(tmp_path: Path) -> None:
    rows = _make_rows(30, 20, seed=3) + [{**r, "kind": "rebound"} for r in _make_rows(5, 5, seed=4)]
    models, reports = fit_cue_models(rows, config=CueFitConfig(n_boot=50))
    assert set(reports) == {"make", "rebound"}
    assert models.models["make"].status == "platt" and models.models["shot"].status == "uncalibrated"
    path = tmp_path / "cue.json"
    doc = save_cue_models(models, reports, path)
    assert doc["for_live"] is True and doc["models"]["make"]["calibration_source"] == "gold"
    loaded = CueModelSet.load(path)
    cues = rows[0]["features"]
    assert loaded.predict("make", cues) == pytest.approx(models.predict("make", cues))
    assert loaded.status("make") == "platt"


def test_jsonl_round_trip_and_validation(tmp_path: Path) -> None:
    rows = _make_rows(2, 3)
    path = tmp_path / "rows.jsonl"
    assert write_cue_rows(rows, path) == 6
    back = load_cue_rows([path])
    assert [r["label"] for r in back] == [r["label"] for r in rows]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"kind": "make", "label": 1}) + "\n")
    with pytest.raises(ValueError, match="features"):
        load_cue_rows([bad])
    bad.write_text(json.dumps({"kind": "make", "features": {}, "label": "maybe"}) + "\n")
    with pytest.raises(ValueError, match="label"):
        load_cue_rows([bad])


def test_cli_refuses_sim_without_flag(tmp_path: Path) -> None:
    path = tmp_path / "sim.jsonl"
    write_cue_rows(_make_rows(30, 20, seed=5, source="sim"), path)
    out = tmp_path / "cue.json"
    assert main(["--jsonl", str(path), "--out", str(out), "--n-boot", "20"]) == 3
    assert not out.exists()
    assert main(["--jsonl", str(path), "--out", str(out), "--n-boot", "20", "--allow-sim"]) == 0
    doc = json.loads(out.read_text())
    assert doc["for_live"] is False and doc["models"]["make"]["calibration_source"] == "sim"
    assert doc["calibration"]["make"]["gate"]["valid_for_live"] is False
    report = json.loads((tmp_path / "cue.report.json").read_text())
    assert report["make"]["status"] in ("platt", "isotonic")


def test_cli_bad_input(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jsonl"
    assert main(["--jsonl", str(missing), "--out", str(tmp_path / "o.json")]) == 2


# ---------------------------------------------------------------------------------------------- sim datasets


def _ev(eid: str, kind: EventKind, t: float, team: Team | None = Team.HOME, **kw: object) -> Event:
    return Event(event_id=eid, kind=kind, t=t, frame_idx=int(t * 30), team=team, **kw)  # type: ignore[arg-type]


def test_build_cue_dataset_labels() -> None:
    cues = {"entered_rim_zone": 1}
    engine = [
        _ev("e1", EventKind.SHOT_ATTEMPT, 10.0, value=3, shot_id="s1", attrs={"cues": cues, "three_cues": {"foot_margin_ft": 0.5}}),
        _ev("e2", EventKind.SHOT_MADE, 11.0, value=3, shot_id="s1", attrs={"cues": {"crossed_inside": 1}}),
        _ev("e3", EventKind.SHOT_ATTEMPT, 30.0, value=2, shot_id="s2", attrs={"cues": cues}),
        _ev("e4", EventKind.REBOUND, 50.0, Team.AWAY, attrs={"cues": {"control_duration_s": 1.0}}),
        _ev("e5", EventKind.REBOUND, 70.0, attrs={"cues": {"control_duration_s": 0.2}}),
        _ev("e6", EventKind.REBOUND, 90.0, attrs={"cues": {}}, status=EventStatus.RETRACTED),
    ]
    truth = [
        _ev("t1", EventKind.SHOT_ATTEMPT, 10.2, value=3, shot_id="s9"),
        _ev("t2", EventKind.SHOT_MISSED, 11.4, shot_id="s9"),
        _ev("t3", EventKind.REBOUND, 50.3, Team.AWAY),
        _ev("t4", EventKind.REBOUND, 70.1, Team.AWAY),  # wrong team for e5
    ]
    rows = build_cue_dataset(engine, truth, "g1")
    got = {(r["kind"], r["event_id"]): r["label"] for r in rows}
    assert got == {("shot", "e1"): True, ("shot", "e3"): False, ("three", "e1"): True, ("make", "e2"): False,
                   ("rebound", "e4"): True, ("rebound", "e5"): False}
    assert all(r["calibration_source"] == "sim" and r["game_id"] == "g1" for r in rows)
    assert build_cue_dataset(engine, truth, "g1", kinds=["make"]) == [r for r in rows if r["kind"] == "make"]


def test_build_cue_dataset_from_engine_on_sim() -> None:
    sim = GameSimulator(SimConfig(seconds=300.0, fps=10.0, seed=9, period_s=240.0)).run()
    eng = EventEngine(sim.court, sim.roster)
    for f in sim.frames:
        eng.process(f)
    eng.flush()
    rows = build_cue_dataset(eng.events, sim.events, "sim9")
    kinds = {r["kind"] for r in rows}
    assert {"shot", "make", "rebound"} <= kinds
    make = [r for r in rows if r["kind"] == "make"]
    assert make and np.mean([r["label"] for r in make]) > 0.2  # some makes, some misses
    assert np.mean([r["label"] for r in rows if r["kind"] == "shot"]) > 0.8  # engine attempts are mostly real


# ---------------------------------------------------------------------------------------------- review regressions


def _models(path: Path) -> dict:
    return json.loads(path.read_text())["models"]


def test_sim_calibration_survives_a_base_round_trip(tmp_path: Path) -> None:
    # A sim-calibrated --base file used to come out as for_live=True: kinds without rows dropped their
    # calibration_source, and a kind with too few gold rows (not refit) was relabelled "gold".
    sim_rows, gold_reb, gold_make = tmp_path / "sim.jsonl", tmp_path / "reb.jsonl", tmp_path / "make.jsonl"
    write_cue_rows(_make_rows(30, 20, seed=5, source="sim"), sim_rows)
    write_cue_rows([{**r, "kind": "rebound"} for r in _make_rows(6, 5, seed=7)], gold_reb)
    write_cue_rows(_make_rows(6, 5, seed=8), gold_make)
    base = tmp_path / "base.json"
    assert main(["--jsonl", str(sim_rows), "--out", str(base), "--n-boot", "20", "--allow-sim"]) == 0
    assert _models(base)["make"] == {**_models(base)["make"], "status": "platt", "calibration_source": "sim"}
    for rows in (gold_reb, gold_make):
        out = tmp_path / f"out_{rows.stem}.json"
        assert main(["--jsonl", str(rows), "--out", str(out), "--n-boot", "20", "--base", str(base)]) == 3
        assert not out.exists()
        assert main(["--jsonl", str(rows), "--out", str(out), "--n-boot", "20", "--base", str(base), "--allow-sim"]) == 0
        doc = json.loads(out.read_text())
        assert doc["for_live"] is False
        assert doc["models"]["make"]["status"] == "platt" and doc["models"]["make"]["calibration_source"] == "sim"
    rep = json.loads((tmp_path / "out_make.json").read_text())["calibration"]["make"]
    assert rep["refit"] is False and rep["calibration_source"] == "sim" and rep["gate"]["valid_for_live"] is False


def test_refit_on_gold_replaces_a_sim_base(tmp_path: Path) -> None:
    sim_rows, gold = tmp_path / "sim.jsonl", tmp_path / "gold.jsonl"
    write_cue_rows(_make_rows(30, 20, seed=5, source="sim"), sim_rows)
    write_cue_rows(_make_rows(30, 20, seed=6), gold)
    base, out = tmp_path / "base.json", tmp_path / "out.json"
    assert main(["--jsonl", str(sim_rows), "--out", str(base), "--n-boot", "20", "--allow-sim"]) == 0
    assert main(["--jsonl", str(gold), "--out", str(out), "--n-boot", "20", "--base", str(base)]) == 0
    doc = json.loads(out.read_text())
    assert doc["for_live"] is True and doc["models"]["make"]["calibration_source"] == "gold"
    assert doc["calibration"]["make"]["refit"] is True


def test_miss_side_gate_is_certified_separately() -> None:
    # Narration says "missed" when 1 - p_make >= 0.90; that tail gets its own rule-of-three certificate.
    rng = np.random.default_rng(0)
    rows = []
    for g in range(40):
        for k in range(30):
            made = bool(rng.random() < 0.5)
            f = {"crossed_inside": int(made), "crossed_inside_3d": int(made), "drop_in_band": int(made),
                 "net_motion_z": float(rng.uniform(2, 3)) if made else 0.0, "bib_conf_max": 1.0 if made else 0.0}
            rows.append({"kind": "make", "features": f, "label": made, "game_id": f"g{g:03d}", "calibration_source": "gold"})
    _, rep = fit_cue_kind(default_cue_models()["make"], rows, CueFitConfig(n_boot=20))
    assert rep.status == "platt" and rep.gate_low is not None and rep.gate_low.kind == "make:negative"
    assert rep.gate_low.threshold == 0.90 and rep.gate_low.n_required == 30
    test = [r for r in rows if r["game_id"] in set(rep.games_holdout)]
    assert rep.gate_low.n_accepted + rep.gate.n_accepted <= len(test)
    assert rep.gate_low.n_accepted >= 30 and rep.gate_low.n_errors == 0 and rep.gate_low.certified and rep.gate.certified
    # an error on the miss side (a make that the model scores ~0) blocks only that certificate
    p, y = [0.01] * 40 + [0.99] * 200, [False] * 39 + [True] + [True] * 200
    low = certify_gate([1 - v for v in p], [not v for v in y], 0.90, 0.10, "platt", "make:negative")
    high = certify_gate(p, y, 0.95, 0.02, "platt", "make")
    assert not low.certified and low.n_errors == 1 and high.certified


def test_certify_gate_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        certify_gate([0.99, float("nan")], [1, 1], 0.95, 0.02, "platt")
    with pytest.raises(ValueError, match="length"):
        certify_gate([0.99], [1, 1], 0.95, 0.02, "platt")
