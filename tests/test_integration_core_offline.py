"""run_offline end to end on a 120 s simulation: truth3d and mono modes, reports, run directory.

Measured on seed 7 / 120 s (this machine, deterministic): truth3d headline macro-F1 0.967 (shot_missed
0.80, every other kind 1.0) with an exact box score; mono 0.687 (possession 0.77, pass 0.89, shots
1.0, steal / turnover 0.0). Thresholds sit slightly below those values.
"""

from __future__ import annotations

import json

import pytest

from courtvision.config import load_config
from courtvision.pipeline import SimSource, make_source, run_offline
from courtvision.sim.simulator import GameSimulator
from courtvision.types import EventKind, EventStatus, Team

SECONDS, SEED = 120.0, 7
NO_KEYS: dict[str, str] = {}  # never pick up a developer's real credentials in tests


def _cfg(*extra: str):
    return load_config(None, [f"sim.seconds={SECONDS}", f"sim.seed={SEED}", "narration.tts=none", *extra])


@pytest.fixture(scope="module")
def sim_result():
    return GameSimulator(_cfg().sim).run()


@pytest.fixture(scope="module")
def truth_run(sim_result, tmp_path_factory):
    cfg = _cfg("source.sim_mode=truth3d")
    out = tmp_path_factory.mktemp("truth3d")
    return run_offline(cfg, make_source(cfg, sim_result=sim_result, realtime=False), out, env=NO_KEYS)


@pytest.fixture(scope="module")
def mono_run(sim_result, tmp_path_factory):
    cfg = _cfg("source.sim_mode=mono")
    out = tmp_path_factory.mktemp("mono")
    return run_offline(cfg, make_source(cfg, sim_result=sim_result, realtime=False), out, env=NO_KEYS)


def test_truth3d_events_match_truth(truth_run) -> None:
    rep = truth_run.event_report
    assert rep is not None and truth_run.truth_events
    f1 = {k: r.f1 for k, r in rep.per_kind.items()}
    print(f"\ntruth3d headline {rep.headline_macro_f1:.3f} per kind {f1}")
    assert rep.headline_macro_f1 >= 0.93
    for kind in ("pass", "shot_attempt", "shot_made", "rebound", "assist", "possession"):
        assert f1[kind] is not None and f1[kind] >= 0.95, (kind, f1)
    assert rep.separate["dribble"].f1 >= 0.95
    assert f1["foul"] == 1.0  # simulated operator entries (the CV engine never infers fouls)


def test_truth3d_box_score_points_match_truth_exactly(truth_run, sim_result) -> None:
    box = truth_run.box_report
    assert box is not None
    assert box.score_match and box.team_points_abs_error == 0
    assert box.per_stat["pts"].mae == 0.0 and box.per_stat["pts"].exact_rate == 1.0
    for pid, line in sim_result.truth_box.items():
        assert truth_run.stats.players[pid].pts == line["pts"], pid
    assert truth_run.stats.event_score == {Team.HOME: sim_result.truth_score[Team.HOME],
                                           Team.AWAY: sim_result.truth_score[Team.AWAY]}


def test_truth3d_lineups_come_from_the_simulated_operator(truth_run, sim_result) -> None:
    starters = sim_result.starters[Team.HOME]
    secs = [truth_run.stats.players[p].seconds for p in starters]
    assert all(s > 0 for s in secs)  # operator starting fives drive player seconds (ARCHITECTURE stats/)
    assert truth_run.stats.lineups


def test_mono_events_and_reports(mono_run) -> None:
    rep = mono_run.event_report
    assert rep is not None
    f1 = {k: r.f1 for k, r in rep.per_kind.items()}
    print(f"\nmono headline {rep.headline_macro_f1:.3f} per kind {f1}")
    assert rep.headline_macro_f1 >= 0.62
    assert f1["shot_attempt"] >= 0.9 and f1["shot_made"] >= 0.9 and f1["pass"] >= 0.8
    box = mono_run.box_report
    assert box is not None and box.score_match  # team points follow the confirmed scoreboard
    assert box.mean_mae < 0.2


def test_reports_latency_and_narration(truth_run, mono_run) -> None:
    for run in (truth_run, mono_run):
        lat = run.latency
        assert lat["engine_ms_per_frame"]["n"] == int(round(SECONDS * 30))
        assert lat["perception_ms_per_frame"] is None  # no images in these modes
        assert lat["event_to_utterance_ms"] is not None and lat["event_to_utterance_ms"]["n"] > 5
        assert lat["realtime_factor"] > 1.0
        assert len(run.transcript) >= 10
        assert all(u.source == "template" for u in run.transcript)  # no keys: template-only
        assert run.jev is not None and run.jev["mode"] == "off"  # no TYPESAFE_API_KEY: deterministic fallbacks


def test_run_directory_files(truth_run) -> None:
    d = truth_run.run_dir
    assert d is not None
    for name in ("events.jsonl", "truth_events.jsonl", "transcript.jsonl", "insights.jsonl", "stats.json",
                 "report.md", "report.json", "config.yaml"):
        assert (d / name).is_file(), name
    events = [json.loads(line) for line in (d / "events.jsonl").read_text().splitlines()]
    assert len(events) == len(truth_run.events) and len({e["event_id"] for e in events}) == len(events)
    assert len((d / "truth_events.jsonl").read_text().splitlines()) == len(truth_run.truth_events)
    assert len((d / "transcript.jsonl").read_text().splitlines()) == len(truth_run.transcript)
    stats = json.loads((d / "stats.json").read_text())
    assert stats["score"] == {"home": truth_run.stats.score[Team.HOME], "away": truth_run.stats.score[Team.AWAY]}
    report = (d / "report.md").read_text()
    assert "Headline macro-F1" in report and "## Box score" in report and "## Latency" in report
    assert "event engine (ms/frame)" in report
    assert load_config(d / "config.yaml") == truth_run.config


def test_events_are_enriched_for_narration(truth_run) -> None:
    live = [e for e in truth_run.events if e.status is not EventStatus.RETRACTED]
    with_actor = [e for e in live if e.player_id]
    assert with_actor and all(e.attrs.get("id_conf") == 1.0 and e.attrs.get("jersey_conf") == 1.0 for e in with_actor)
    attempts = {e.shot_id: e for e in live if e.kind is EventKind.SHOT_ATTEMPT and "p_three" in e.attrs}
    results = [e for e in live if e.kind in (EventKind.SHOT_MADE, EventKind.SHOT_MISSED) and e.shot_id in attempts]
    assert results and all(e.attrs["p_three"] == attempts[e.shot_id].attrs["p_three"] for e in results)


def test_offline_runs_are_deterministic(sim_result) -> None:
    cfg = _cfg("source.sim_mode=truth3d")
    a = run_offline(cfg, make_source(cfg, sim_result=sim_result, realtime=False), write=False, env=NO_KEYS)
    b = run_offline(cfg, SimSource(sim_result, "truth3d"), write=False, env=NO_KEYS)
    assert [(u.text, round(u.t, 3)) for u in a.transcript] == [(u.text, round(u.t, 3)) for u in b.transcript]
    assert [(e.event_id, e.kind, e.revision) for e in a.events] == [(e.event_id, e.kind, e.revision) for e in b.events]


def test_window_runs_skip_the_box_score(sim_result) -> None:
    cfg = _cfg("source.start_s=30", "source.duration_s=20")
    res = run_offline(cfg, make_source(cfg, sim_result=sim_result, realtime=False), write=False, env=NO_KEYS)
    assert res.box_report is None  # truth_box covers the whole simulation
    assert res.event_report is not None
    assert all(30.0 - 1e-6 <= e.t <= 50.0 for e in res.truth_events)
    assert res.frames == 600
