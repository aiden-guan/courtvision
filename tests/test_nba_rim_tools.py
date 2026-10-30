"""The rim tools' scoring (tools/nba_rim_model.py, tools/nba_shot_models.py)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from courtvision.nba import makes as mk

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def tool(name: str) -> ModuleType:
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_auc_counts_ties_half_and_needs_both_classes():
    rm = tool("nba_rim_model")
    assert rm.auc([0.1, 0.4, 0.35, 0.8], [False, False, True, True]) == pytest.approx(0.75)
    assert rm.auc([0.5, 0.5], [True, False]) == pytest.approx(0.5)
    assert np.isnan(rm.auc([0.2, 0.3], [True, True]))


def test_the_rim_report_scores_made_or_missed_on_real_shots_and_shot_or_not_on_candidates():
    rm = tool("nba_rim_model")
    meta = [{"label": "made", "source": "candidate", "ft": False}, {"label": "missed", "source": "candidate", "ft": False},
            {"label": "none", "source": "candidate", "ft": False}, {"label": "made", "source": "pbp", "ft": True}]
    probs = np.array([[0.7, 0.2, 0.1], [0.3, 0.6, 0.1], [0.2, 0.2, 0.6], [0.2, 0.7, 0.1]], np.float32)
    rep = rm.report(meta, probs)
    assert rep["made or missed, real shots"]["n"] == 3 and rep["made or missed, real shots"]["accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    assert rep["made or missed, real shots, candidates"]["accuracy"] == 1.0
    assert rep["shot or not, candidates"] == {"n": 3, "accuracy": 1.0, "auc": 1.0}


def test_shot_models_are_scored_on_a_game_they_were_not_fitted_on(tmp_path, monkeypatch):
    sm = tool("nba_shot_models")
    rng = np.random.default_rng(0)

    def rows(n: int) -> list[dict]:
        out = []
        for _ in range(n):
            made = bool(rng.integers(2))
            f = {name: 0.0 for name in mk.CUE_FEATURES + mk.VISIT_FEATURES + mk.TIMELINE_FEATURES + mk.FT_FEATURES}
            f["crossed_inside"] = float(made if rng.random() < 0.9 else not made)
            f["visit"] = 1.0
            out.append({"key": "k", "t": 0.0, "label": "fg_made" if made else "fg_missed", "features": f})
        return out

    data = {"g1": rows(80), "g2": rows(80)}
    monkeypatch.setattr(sm, "load", lambda game, tag: data[game])
    monkeypatch.setattr(sys, "argv", ["nba_shot_models.py", "--runs", "g1:t,g2:t", "--out", str(tmp_path / "m.json")])
    assert sm.main() == 0
    doc = json.loads((tmp_path / "m.json").read_text())
    assert doc["held_out"]["g1"]["made_accuracy"] > 0.8 and doc["held_out"]["g2"]["real_accuracy"] == 1.0
    assert mk.ShotModels.from_dict(doc).made.names[0] == "crossed_inside" and doc["candidates"] == 160
