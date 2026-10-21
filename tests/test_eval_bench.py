"""tools/accuracy_bench.py and courtvision.eval.bench: pooling over seeds, the split guard, and parity
with ``courtvision simulate`` on tiny runs (one seed, a few simulated seconds)."""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import pytest

from courtvision.cli import main as cli_main
from courtvision.eval import bench
from courtvision.eval.boxscore import compare_box
from courtvision.eval.events import match_events
from courtvision.types import Event, EventKind, Team

_PATH = Path(__file__).resolve().parents[1] / "tools" / "accuracy_bench.py"
_spec = importlib.util.spec_from_file_location("accuracy_bench", _PATH)
assert _spec is not None and _spec.loader is not None
accuracy_bench = importlib.util.module_from_spec(_spec)
sys.modules["accuracy_bench"] = accuracy_bench
_spec.loader.exec_module(accuracy_bench)

K = EventKind
_ids = itertools.count(1)


def ev(kind: EventKind, t: float, player: str | None = "home-1", **kw) -> Event:
    kw.setdefault("event_id", f"b{next(_ids):06d}")
    kw.setdefault("frame_idx", int(round(t * 30)))
    kw.setdefault("team", Team.HOME if player and player.startswith("home") else Team.AWAY)
    return Event(kind=kind, t=t, player_id=player, **kw)


def shot(t: float, shooter: str, made: bool, value: int, sid: str) -> list[Event]:
    return [ev(K.SHOT_ATTEMPT, t, shooter, value=value, shot_id=sid, attrs={"shot_type": "jumper"}),
            ev(K.SHOT_MADE if made else K.SHOT_MISSED, t + 1.0, shooter, shot_id=sid, value=value if made else 0)]


def _record(pred: list[Event], truth: list[Event], seed: int, box_pred: dict | None = None,
            box_truth: tuple[dict, dict] | None = None) -> dict:
    box = compare_box(box_pred, *box_truth) if box_pred is not None and box_truth is not None else None
    rec = bench.record_from_reports(match_events(pred, truth), box)
    rec.update(mode="mono", seed=seed, seconds=10.0, frames=300, wall_s=0.5)
    return rec


def test_pooling_sums_counts_over_seeds() -> None:
    # seed 1: the one rebound is found (right player, right label); a made three is called a made two
    truth_a = [ev(K.REBOUND, 12.0, "home-3", attrs={"offensive": False}), *shot(20.0, "home-2", True, 3, "s1")]
    pred_a = [ev(K.REBOUND, 12.2, "home-3", attrs={"offensive": False}), *shot(20.1, "home-2", True, 2, "p1")]
    # seed 2: three rebounds, one found with the wrong player and label, plus one false positive
    truth_b = [ev(K.REBOUND, t, "away-5", attrs={"offensive": True}) for t in (5.0, 15.0, 25.0)]
    truth_b += shot(30.0, "away-6", False, 2, "s2")
    pred_b = [ev(K.REBOUND, 5.1, "away-7", attrs={"offensive": False}), ev(K.REBOUND, 40.0, "away-5"),
              *shot(30.0, "away-6", False, 2, "p2")]
    box_truth = ({"home-1": {"pts": 3, "reb": 1}}, {"home": 3, "away": 0})
    rec_a = _record(pred_a, truth_a, 1, {"players": {"home-1": {"pts": 2, "reb": 1}}, "score": {"home": 2, "away": 0}},
                    box_truth)
    rec_b = _record(pred_b, truth_b, 2, {"players": {"home-1": {"pts": 3, "reb": 0}}, "score": {"home": 3, "away": 0}},
                    box_truth)
    assert rec_a["per_kind_f1"]["rebound"] == 1.0 and rec_b["per_kind_f1"]["rebound"] == pytest.approx(0.4)

    pooled = bench.pool_mode([rec_b, rec_a])
    reb = pooled["per_kind"]["rebound"]
    assert (reb["tp"], reb["n_pred"], reb["n_truth"], reb["fp"], reb["fn"]) == (2, 3, 4, 1, 2)
    assert reb["f1"] == pytest.approx(4 / 7)  # 2*sum(TP)/(sum(pred)+sum(truth)), not the seed mean 0.7
    assert reb["player_accuracy"] == pytest.approx(0.5) and reb["attributed_f1"] == pytest.approx(2 / 7)
    assert reb["attr_accuracy"]["offensive"] == {"accuracy": 0.5, "n": 2}
    assert pooled["rebound"]["rebounder_accuracy"] == pytest.approx(0.5)
    head = pooled["headline_kinds"]
    assert head == ["shot_attempt", "shot_made", "shot_missed", "rebound"]
    assert pooled["headline_f1"] == pytest.approx(sum(pooled["per_kind"][k]["f1"] for k in head) / len(head))
    assert pooled["headline_f1_seed_mean"] == pytest.approx((rec_a["headline_f1"] + rec_b["headline_f1"]) / 2)
    assert pooled["seeds"] == [1, 2] and pooled["n_runs"] == 2
    # shot level: both outcomes right, one of two values right (the three called a two)
    assert pooled["shots"]["fg_make_miss"]["accuracy"] == 1.0 and pooled["shots"]["fg_make_miss"]["n"] == 2
    assert pooled["two_vs_three_accuracy"] == pytest.approx(0.5)
    assert pooled["shots"]["two_vs_three"]["confusion"] == {"3": {"2": 1}, "2": {"2": 1}}
    # box: pts errors 1 and 0, reb errors 0 and 1 over two games -> per-stat MAE 0.5, team points |error| 1 and 0
    box = pooled["box"]
    assert box["per_stat"]["pts"]["mae"] == pytest.approx(0.5) and box["per_stat"]["reb"]["mae"] == pytest.approx(0.5)
    assert box["mean_mae"] == pytest.approx(0.5) and pooled["box_mean_mae"] == pytest.approx(0.5)
    assert box["team_points_abs_error_mean"] == pytest.approx(0.5) and box["team_points_abs_error_max"] == 1
    assert box["score_match_rate"] == pytest.approx(0.5)

    report = bench.build_report([rec_a, rec_b], split="train", command="x")
    text = bench.format_markdown(report)
    assert "| mono | 1,2 | 10 | **" in text and "| rebound* | 0.571 [4] (" in text
    d = json.loads(json.dumps(report.to_dict()))  # JSON-ready
    assert d["modes"]["mono"]["runs"][0]["seed"] == 1 and d["protocol"]["train_seeds"] == [1, 2, 3, 4, 5, 6]
    assert "fp_ids" not in json.dumps(d)  # aggregate counts only: no event ids leave a run


def test_rare_false_positive_only_kind_moves_the_headline_but_not_its_companions() -> None:
    """A core kind with false positives and no truth (e.g. BLOCK on train) enters the strict headline as
    F1 0; the supported-kind macro ignores it and its false positives are reported per stream minute."""
    truth = [*shot(10.0, "home-2", True, 2, "s1"), *shot(30.0, "home-3", False, 2, "s2")]
    pred = [*shot(10.1, "home-2", True, 2, "p1"), *shot(30.2, "home-3", False, 2, "p2")]
    clean = _record(pred, truth, 1)
    noisy = _record(pred + [ev(K.BLOCK, 31.0, "away-9", shot_id="p2")], truth, 1)
    a, b = bench.pool_mode([clean]), bench.pool_mode([noisy])
    assert a["headline_f1"] == pytest.approx(1.0) and b["headline_f1"] == pytest.approx(0.75)  # block F1 0 of 4 kinds
    assert a["headline_supported_f1"] == b["headline_supported_f1"] == pytest.approx(1.0)
    assert b["supported_kinds"] == ["shot_attempt", "shot_made", "shot_missed"]
    assert b["micro_f1"] == pytest.approx(2 * 4 / (5 + 4))  # 4 TP, 5 predictions, 4 truths
    assert b["unsupported_fp_per_min"] == {"block": pytest.approx(1 / (10.0 / 60.0))}  # 1 FP in 10 s
    assert a["unsupported_fp_per_min"] == {}
    # eval's own per-run report carries the same companions
    rep = match_events(pred + [ev(K.BLOCK, 31.0, "away-9", shot_id="p2")], truth)
    assert rep.headline_macro_f1 == pytest.approx(0.75) and rep.supported_macro_f1 == pytest.approx(1.0)
    assert rep.unsupported_fp == {"block": 1} and rep.supported_kinds == ["shot_attempt", "shot_made", "shot_missed"]


def test_seed_bootstrap_intervals() -> None:
    truth = [*shot(10.0, "home-2", True, 2, "s1"), *shot(30.0, "home-3", False, 2, "s2")]
    good = _record([*shot(10.1, "home-2", True, 2, "p1"), *shot(30.2, "home-3", False, 2, "p2")], truth, 1)
    bad = _record([*shot(10.1, "home-2", True, 2, "p1")], truth, 2)
    pooled = bench.pool_mode([good, bad])
    ci = pooled["ci95"]
    assert ci["n_resamples"] == bench.BOOTSTRAP_RESAMPLES
    lo, hi = ci["headline_f1"]
    assert lo <= pooled["headline_f1"] <= hi and lo < hi  # the two games disagree: a real interval
    assert ci["per_kind_f1"]["shot_attempt"][0] <= pooled["per_kind"]["shot_attempt"]["f1"] <= ci["per_kind_f1"][
        "shot_attempt"][1]
    assert bench.pool_mode([good, bad])["ci95"] == ci  # deterministic
    assert bench.pool_mode([good])["ci95"] == {}  # one game: no interval
    text = bench.format_markdown(bench.build_report([good, bad], split="train"))
    assert "supported-kind macro F1" in text and "not comparable" in text


@pytest.mark.parametrize("argv", [
    ["--split", "train", "--seeds", "101"],  # held-out seed on the train split
    ["--split", "train", "--video-seeds", "7"],  # a seed outside both splits
    ["--split", "heldout", "--seeds", "1"],
    ["--split", "heldout", "--keep-runs", "runs/x"],  # held-out runs never keep per-event output
    ["--modes", "truth3d,stereo"],
    ["--modes", "mono,mono"],
    ["--seeds", "1,1"],
    ["--workers", "0"],
])
def test_usage_errors_run_nothing(argv: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accuracy_bench, "run_case", lambda case: pytest.fail("a run started"))
    out = tmp_path / "out.json"
    assert accuracy_bench.main([*argv, "--json", str(out), "-q"]) == 2
    assert not out.exists()


def test_plan_defaults_and_video_length() -> None:
    args = accuracy_bench.build_parser().parse_args([])
    cases = accuracy_bench.plan_cases(args)
    assert [(c.mode, c.seed) for c in cases] == [(m, s) for m in bench.MODES for s in bench.TRAIN_SEEDS]
    assert {c.seconds for c in cases if c.mode != "video"} == {bench.DEFAULT_SECONDS}
    assert {c.seconds for c in cases if c.mode == "video"} == {bench.DEFAULT_VIDEO_SECONDS}
    held = accuracy_bench.plan_cases(accuracy_bench.build_parser().parse_args(
        ["--split", "heldout", "--modes", "video", "--seconds", "60", "--video-seeds", "101,102"]))
    assert [(c.seed, c.seconds, c.split) for c in held] == [(101, 60.0, "heldout"), (102, 60.0, "heldout")]


def test_bench_is_the_simulate_code_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """One seed, 20 s, mono: the bench's run is `courtvision simulate`'s run, and one seed pools to itself."""
    assert cli_main(["-q", "simulate", "--mode", "mono", "--seed", "1", "--seconds", "20", "--json",
                     "--out", str(tmp_path / "sim")]) == 0
    summary = json.loads(capsys.readouterr().out)
    out = tmp_path / "bench.json"
    rc = accuracy_bench.main(["--modes", "mono", "--seeds", "1", "--seconds", "20", "--workers", "1", "-q",
                              "--json", str(out), "--md", str(tmp_path / "bench.md"),
                              "--keep-runs", str(tmp_path / "runs")])
    assert rc == 0
    printed = capsys.readouterr().out
    assert printed == (tmp_path / "bench.md").read_text() and "| mono | 1 | 20 |" in printed
    d = json.loads(out.read_text())
    assert d["split"] == "train" and d["errors"] == [] and list(d["modes"]) == ["mono"]
    m = d["modes"]["mono"]
    [run] = m["runs"]
    assert run["frames"] == summary["frames"] == 600
    assert run["headline_f1"] == summary["headline_f1"] and run["per_kind_f1"] == summary["per_kind_f1"]
    assert run["box_mean_mae"] == summary["box_mean_mae"]
    assert run["team_points_abs_error"] == summary["team_points_abs_error"]
    assert m["headline_f1"] == pytest.approx(summary["headline_f1"])
    assert m["per_kind_f1"] == pytest.approx(summary["per_kind_f1"])
    assert m["box_mean_mae"] == pytest.approx(summary["box_mean_mae"])
    assert m["team_points_error"] == summary["team_points_abs_error"]
    assert (tmp_path / "runs" / "sim-mono-seed1" / "report.json").is_file()  # --keep-runs (train only)
    [win] = m["windows"]  # one (partial) 60 s window: it reproduces the run's own matching
    assert win["n_runs"] == 1 and win["headline_f1"] == pytest.approx(m["headline_f1"])


def test_process_pool_matches_in_process(tmp_path: Path) -> None:
    out = tmp_path / "pool.json"
    rc = accuracy_bench.main(["--modes", "truth3d", "--seeds", "1,2", "--seconds", "5", "--workers", "2", "-q",
                              "--json", str(out)])
    assert rc == 0
    runs = json.loads(out.read_text())["modes"]["truth3d"]["runs"]
    assert [r["seed"] for r in runs] == [1, 2]
    direct = bench.run_case(bench.BenchCase(mode="truth3d", seed=2, seconds=5.0))
    assert runs[1]["headline_f1"] == direct["headline_f1"] and runs[1]["per_kind_f1"] == direct["per_kind_f1"]


def test_failed_run_is_reported_not_hidden(tmp_path: Path) -> None:
    out = tmp_path / "fail.json"
    rc = accuracy_bench.main(["--modes", "truth3d", "--seeds", "1", "--seconds", "5", "--workers", "1", "-q",
                              "--set", "events.no_such_key=1", "--json", str(out)])
    assert rc == 1
    d = json.loads(out.read_text())
    assert d["modes"] == {} and d["errors"][0]["seed"] == 1 and "no_such_key" in d["errors"][0]["error"]


# ---------------------------------------------------------------------------- time trend and comparisons


def test_window_counts_score_each_window_on_its_own() -> None:
    """Non-cumulative 60 s windows: each is matched on its own with its inner edges as cut edges (a pass
    whose detection lands across the edge is "don't care" on both sides, not an FP + an FN), and pooling
    the windows over runs gives a per-window headline (a flat line = no degradation over time)."""
    truth = [*shot(10.0, "home-2", True, 2, "s1"), *shot(70.0, "home-3", False, 2, "s2"),
             ev(K.PASS, 59.8, "home-2", other_player_id="home-3")]
    pred = [*shot(10.1, "home-2", True, 2, "p1"), *shot(70.2, "home-3", False, 2, "p2"),
            ev(K.PASS, 60.3, "home-2", other_player_id="home-3"), ev(K.STEAL, 90.0, "away-9")]
    whole = match_events(pred, truth, window=(None, 120.0))
    assert whole.per_kind["pass"].tp == 1  # the whole run pairs the pass
    wins = bench.window_counts(pred, truth, whole)
    assert [(w["start"], w["end"]) for w in wins] == [(0.0, 60.0), (60.0, 120.0)]
    first, second = (w["kinds"] for w in wins)
    assert first["pass"]["n_truth"] == 0 and first["pass"]["n_pred"] == 0  # excluded at the inner edge
    assert second["pass"]["n_truth"] == 0 and second["pass"]["n_pred"] == 0
    assert first["shot_made"]["tp"] == 1 and "steal" not in first
    assert second["shot_missed"]["tp"] == 1 and second["steal"]["n_pred"] == 1 and second["steal"]["tp"] == 0
    rec = bench.record_from_reports(whole, windows=wins)
    rec.update(mode="mono", seed=1, seconds=120.0, frames=3600, wall_s=1.0)
    rec2 = dict(rec, seed=2)
    pooled = bench.pool_mode([rec, rec2], bootstrap=0)
    w0, w1 = pooled["windows"]
    assert w0["n_runs"] == w1["n_runs"] == 2
    assert w0["headline_f1"] == pytest.approx(1.0)  # shot_attempt, shot_made
    assert w1["headline_f1"] == pytest.approx(2 / 3)  # shot_attempt, shot_missed, steal F1 0 (no truth)
    assert pooled["window_headline_spread"] == pytest.approx(1 / 3)
    text = bench.format_markdown(bench.build_report([rec, rec2], split="train"))
    assert "Time trend (mono)" in text and "1.000, 0.667" in text


def test_windows_of_a_real_run_reproduce_its_matching() -> None:
    """A 130 s truth3d run: three windows (60, 60, 10 s); their counts never exceed the whole run's, and
    only events next to an inner edge can drop out."""
    rec = bench.run_case(bench.BenchCase(mode="truth3d", seed=1, seconds=130.0))
    wins = rec["windows"]
    assert [round(w["end"] - w["start"]) for w in wins] == [60, 60, 10]
    for kind, whole in rec["kinds"].items():
        tp = sum(w["kinds"].get(kind, {}).get("tp", 0) for w in wins)
        n_truth = sum(w["kinds"].get(kind, {}).get("n_truth", 0) for w in wins)
        assert tp <= whole["tp"] and n_truth <= whole["n_truth"], kind
        assert whole["n_truth"] - n_truth <= 4, kind  # at most a couple of events per inner edge


def _cmp_report(head: float, secs: float, steal: float) -> dict:
    return {"command": f"bench {head}", "modes": {"mono": {
        "headline_f1": head, "seconds": secs,
        "ci95": {"headline_f1": [0.65, 0.75], "per_kind_f1": {"steal": [0.2, 0.6]}},
        "per_kind": {"steal": {"f1": steal, "n_truth": 8}}}}}


def test_comparison_reports_truth_n_and_treats_small_300s_changes_as_noise() -> None:
    before = _cmp_report(0.70, 300.0, 0.40)
    small = bench.compare_reports(before, _cmp_report(0.72, 300.0, 0.50))["modes"]["mono"]
    assert small["verdict"] == "noise" and small["delta"] == pytest.approx(0.02)
    assert small["kinds"]["steal"] == {"before": 0.40, "after": 0.50, "delta": pytest.approx(0.10),
                                       "n_truth_before": 8, "n_truth_after": 8, "ci_before": [0.2, 0.6],
                                       "ci_after": [0.2, 0.6], "verdict": "inside CI"}
    big = bench.compare_reports(before, _cmp_report(0.80, 300.0, 0.90))["modes"]["mono"]
    assert big["verdict"] == "change" and big["kinds"]["steal"]["verdict"] == "change"
    long = bench.compare_reports(_cmp_report(0.70, 1200.0, 0.4), _cmp_report(0.72, 1200.0, 0.4))["modes"]["mono"]
    assert long["verdict"] == "inside CI"  # 1200 s runs: not dismissed as noise, judged against the interval
    text = bench.format_comparison(bench.compare_reports(before, _cmp_report(0.72, 300.0, 0.50)))
    assert "| mono | 300 | 0.700 [0.650, 0.750] | 0.720 [0.650, 0.750] | +0.020 | noise |" in text
    assert "| steal | 0.400 [8] [0.200, 0.600] | 0.500 [8] [0.200, 0.600] | +0.100 | inside CI |" in text


def test_diff_compares_two_saved_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(_cmp_report(0.70, 300.0, 0.4)))
    b.write_text(json.dumps(_cmp_report(0.69, 300.0, 0.4)))
    assert accuracy_bench.main(["--diff", str(a), str(b)]) == 0
    assert "| -0.010 | noise |" in capsys.readouterr().out
    assert accuracy_bench.main(["--diff", str(a), str(tmp_path / "missing.json")]) == 2
