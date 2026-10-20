"""Cue models: cold-start weights, JSON round trip, fit + status transitions, engine wiring."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from courtvision.events.calibration import GATED_KINDS, CalibrationConfig, CueModel, CueModelSet, default_cue_models, ece
from courtvision.events.engine import EventEngineConfig
from courtvision.types import EventKind

from test_events_kit import live, one, run_engine, two_team_scene

TEXTBOOK_MAKE = {"crossed_inside": True, "drop_in_band": True, "drop_is_freefall": False, "occluded_frames": 0}
CONFLICT_MAKE = {"crossed_inside": True, "drop_in_band": False, "drop_is_freefall": True, "occluded_frames": 4}


def test_cold_start_models_cover_every_gated_kind() -> None:
    models = default_cue_models()
    assert set(models) == set(GATED_KINDS)
    assert all(m.status == "uncalibrated" for m in models.values())
    make = models["make"]
    assert make.logit(TEXTBOOK_MAKE) == pytest.approx(-2.0 + 3.0 + 1.5)  # RESEARCH §7A.2 example
    assert make.predict(TEXTBOOK_MAKE) > 0.9
    assert 0.05 < make.predict(CONFLICT_MAKE) < 0.5
    assert make.predict({}) < 0.2  # no cues: no make
    three = models["three"]
    assert three.predict({"foot_margin_ft": 2.0}) > 0.99 and three.predict({"foot_margin_ft": -3.0}) < 0.01
    assert 0.3 < three.predict({"foot_margin_ft": 0.0}) < 0.7  # on the line: genuinely ambiguous


def test_json_round_trip(tmp_path: Path) -> None:
    m = default_cue_models()["make"]
    path = tmp_path / "make.json"
    m.save(path)
    m2 = CueModel.load(path)
    assert m2.to_dict() == m.to_dict()
    for cues in (TEXTBOOK_MAKE, CONFLICT_MAKE, {}):
        assert m2.predict(cues) == m.predict(cues)
    ms = CueModelSet()
    ms.save(tmp_path / "all.json")
    loaded = CueModelSet.load(tmp_path / "all.json")
    assert json.dumps(loaded.to_dict(), sort_keys=True) == json.dumps(ms.to_dict(), sort_keys=True)


def _synthetic(n: int, seed: int) -> tuple[list[dict[str, object]], list[bool]]:
    """Make-cue samples whose true P(make) differs from the cold-start weights."""
    rng = random.Random(seed)
    cues, labels = [], []
    for _ in range(n):
        c = {"crossed_inside": rng.random() < 0.6, "drop_in_band": rng.random() < 0.5,
             "drop_is_freefall": rng.random() < 0.2, "occluded_frames": rng.randint(0, 5)}
        z = -1.0 + 2.0 * c["crossed_inside"] + 0.5 * c["drop_in_band"] - 1.0 * c["drop_is_freefall"]
        labels.append(rng.random() < 1.0 / (1.0 + np.exp(-z)))
        cues.append(c)
    return cues, labels


@pytest.mark.parametrize(("n", "status"), [(120, "uncalibrated"), (400, "platt"), (1200, "isotonic")])
def test_fit_status_transitions(n: int, status: str) -> None:
    m = default_cue_models()["make"]
    cues, labels = _synthetic(n, seed=n)
    before = m.to_dict()
    assert m.fit(cues, labels) == status and m.status == status
    if status == "uncalibrated":
        assert m.to_dict() == before  # too few labels: untouched
        return
    assert m.n_labelled == n and m.n_ambiguous >= CalibrationConfig().platt_min_ambiguous
    test_cues, test_labels = _synthetic(3000, seed=99)
    fresh = default_cue_models()["make"]
    assert ece(m.predict_many(test_cues), test_labels) < ece(fresh.predict_many(test_cues), test_labels)
    assert ece(m.predict_many(test_cues), test_labels) < 0.06
    m2 = CueModel.from_dict(json.loads(json.dumps(m.to_dict())))
    assert m2.status == status and np.allclose(m2.predict_many(test_cues), m.predict_many(test_cues))


def test_fit_needs_enough_ambiguous_cases() -> None:
    m = default_cue_models()["make"]
    cues = [dict(TEXTBOOK_MAKE) for _ in range(400)]  # cold-start P = 0.92 for every one of them
    labels = [True] * 399 + [False]
    cfg_strict = CalibrationConfig(ambiguous_band=(0.05, 0.9))
    strict = CueModel("make", default_cue_models()["make"].features, -2.0, config=cfg_strict)
    assert strict.fit(cues, labels) == "uncalibrated"
    assert m.fit(cues, labels) == "platt"


def test_engine_reports_p_status_and_assumed_sim() -> None:
    s = two_team_scene()
    s.add("home-1", 18.0, 30.0)
    s.start("home-1").hold("home-1", 0.5).shoot("left").drop_through("left").set_score(2, 0).rest(0.3)
    eng = run_engine(s.frames_3d())
    made = one(eng.events, EventKind.SHOT_MADE)
    assert made.attrs["p_status"] == "uncalibrated" and made.confidence > 0.9
    eng2 = run_engine(s.frames_3d(), EventEngineConfig(assume_calibrated=True))
    assert all(e.attrs["p_status"] == "assumed_sim" for e in live(eng2.events) if "p_status" in e.attrs)
    assert one(eng2.events, EventKind.SHOT_ATTEMPT).attrs["p_status"] == "assumed_sim"


def test_engine_loads_cue_models_from_json(tmp_path: Path) -> None:
    ms = CueModelSet()
    ms.models["shot"].bias = 5.0
    ms.models["shot"].status = "platt"
    ms.models["shot"].platt = (1.0, 0.0)
    path = tmp_path / "cues.json"
    ms.save(path)
    s = two_team_scene()
    s.add("home-1", 18.0, 30.0)
    s.start("home-1").hold("home-1", 0.5).shoot("left").drop_through("left").rest(0.3)
    eng = run_engine(s.frames_3d(), EventEngineConfig(cue_model_path=str(path)))
    att = one(eng.events, EventKind.SHOT_ATTEMPT)
    assert att.attrs["p_status"] == "platt" and att.confidence > 0.99
