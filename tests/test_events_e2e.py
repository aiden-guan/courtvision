"""End-to-end smoke against the simulator (skipped when sim / eval are not importable).

Truth 3-D frames and NoiseModel monocular frames go through the engine; FOUL / TIMEOUT truth events
are replayed as operator entries (the CV engine does not infer fouls). Thresholds are deliberately
loose: they guard against regressions, not against the simulator's tuning.
"""

from __future__ import annotations

import pytest

from courtvision.events.engine import EventEngine, EventEngineConfig
from courtvision.types import Event, EventKind

sim = pytest.importorskip("courtvision.sim.simulator")
noise = pytest.importorskip("courtvision.sim.noise")
ev_eval = pytest.importorskip("courtvision.eval.events")

MANUAL = (EventKind.FOUL, EventKind.TIMEOUT)


@pytest.fixture(scope="module")
def result():  # type: ignore[no-untyped-def]
    return sim.GameSimulator(sim.SimConfig(seconds=150.0, seed=7)).run()


def _run(result, frames) -> list[Event]:  # type: ignore[no-untyped-def]
    eng = EventEngine(result.court, result.roster, EventEngineConfig(assume_calibrated=True), rules=result.rules)
    manual = sorted((e for e in result.events if e.kind in MANUAL), key=lambda e: e.t)
    k = 0
    for fr in frames:
        while k < len(manual) and manual[k].t <= fr.t:
            eng.add_manual(manual[k])
            k += 1
        eng.process(fr)
    eng.flush()
    return [e for e in eng.events if e.kind not in MANUAL]


def _truth(result) -> list[Event]:  # type: ignore[no-untyped-def]
    return [e for e in result.events if e.kind not in MANUAL and e.kind is not EventKind.SUBSTITUTION]


def test_e2e_truth_3d(result) -> None:  # type: ignore[no-untyped-def]
    rep = ev_eval.match_events(_run(result, result.frames), _truth(result))
    f1 = {k: r.f1 for k, r in rep.per_kind.items()}
    for kind in ("pass", "shot_attempt", "shot_made"):
        if kind in f1 and f1[kind] is not None:
            assert f1[kind] >= 0.8, (kind, f1)
    assert rep.separate["dribble"].f1 >= 0.9
    assert rep.headline_macro_f1 >= 0.5


def test_e2e_monocular(result) -> None:  # type: ignore[no-untyped-def]
    frames = noise.NoiseModel(noise.NoiseConfig(monocular=True), result.camera).apply(result.frames, seed=1)
    rep = ev_eval.match_events(_run(result, frames), _truth(result))
    f1 = {k: r.f1 for k, r in rep.per_kind.items()}
    for kind in ("pass", "shot_made"):
        if kind in f1 and f1[kind] is not None:
            assert f1[kind] >= 0.6, (kind, f1)
    assert rep.headline_macro_f1 >= 0.3
