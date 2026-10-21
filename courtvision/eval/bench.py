"""Accuracy benchmark over seeded simulator games (CLI: ``tools/accuracy_bench.py``).

Every agent measures accuracy the same way: for each ``(mode, seed)`` case the benchmark runs
exactly what ``courtvision simulate --mode MODE --seed SEED --seconds T`` runs
(:func:`courtvision.cli.simulate_config` then :func:`courtvision.pipeline.run_offline`), keeps only
aggregate counts from the ``courtvision.eval`` reports (never event ids), and pools them over seeds:

- **per-kind F1 pooled over seeds**: sum TP / predictions / truths over the seeds, then
  ``F1 = 2 TP / (n_pred + n_truth)`` (``eval.events._f1``);
- **headline macro-F1**: eval's definition (mean F1 over the core kinds that occur in prediction or
  truth, ``eval.events.match_events``) applied to the pooled per-kind counts; the per-seed headline
  (eval's own number for each run) and its mean / sd across seeds are reported too. It stays strict (a
  core kind with false positives but no truth counts as F1 0), so it is reported next to the macro over
  the core kinds that have truth support, the pooled micro-F1 over core kinds, the false positives per
  stream minute of core kinds without truth, and 95% seed-bootstrap intervals (games resampled with
  replacement): with 0-15 truth events per rare kind, one event moves a kind's F1 by 0.05-0.15, so only
  the pooled multi-seed numbers of >= 180-300 s games are comparable, never single-seed <= 60 s ones;
- shot make/miss accuracy (field goals; free throws apart), 2-vs-3 accuracy, made-FG P/R/F1, shooter
  accuracy, all from the shot-level join (``ShotReport``), pooled through its confusion counts;
- rebound attribution (rebounder accuracy on matched rebounds, attributed F1, offensive/defensive
  label accuracy) and assist attribution (assister-label accuracy on made-FG pairs, where
  "unassisted" is a label; assist-event F1 / attributed F1);
- box score: per-stat MAE pooled over every truth player of every game, their mean (eval's
  ``mean_mae`` definition), and team points error (sum of |error| over both teams per game).

Split protocol (anti-overfitting): tune only on :data:`TRAIN_SEEDS`; :data:`HELDOUT_SEEDS` are for
the verifier, and held-out runs never keep per-event outputs.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from courtvision.eval.boxscore import BoxReport
from courtvision.eval.events import (CORE_KINDS, EventEvalConfig, EventLike, EventReport, KindReport,
                                    latest_revisions, match_events)
from courtvision.eval.tracking import TrackReport

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- protocol constants
# Protocol, not tunables: fixed by the accuracy programme (anti-overfitting split). Status: design choice.
TRAIN_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)  # the only seeds anyone tunes on
HELDOUT_SEEDS: tuple[int, ...] = (101, 102, 103, 104)  # verifier only; aggregate metrics only
SPLITS: dict[str, tuple[int, ...]] = {"train": TRAIN_SEEDS, "heldout": HELDOUT_SEEDS}
MODES: tuple[str, ...] = ("truth3d", "mono", "video")  # cli.SIM_MODES
DEFAULT_SECONDS = 300.0  # simulated video seconds per game (truth3d / mono: ~1.5 s wall each)
# Video mode renders + perceives every frame (~30 ms/frame on an M3 Pro, ~165 s wall for 180 s of
# video), so its default game length is shorter to bound runtime. Box scores of 180 s and 300 s games
# are not comparable across modes; event F1 / accuracies are rates and are.
DEFAULT_VIDEO_SECONDS = 180.0
BENCH_VERSION = 2
BOOTSTRAP_RESAMPLES = 1000  # seed-bootstrap resamples for the 95% intervals; protocol choice
BOOTSTRAP_SEED = 0  # deterministic intervals
# Non-cumulative windows for the time-trend line (protocol): a run is re-scored per WINDOW_S of stream, each
# window's inner edges treated as cut edges (match_events(window=...)), and the per-window counts are pooled
# over seeds. A real degradation over time shows up as a trend across windows; the strict headline of a
# single <= 60 s run does not (rare kinds enter or leave its average).
WINDOW_S = 60.0
# Comparison rule (protocol; RESEARCH accuracy programme round 2): with 0-15 truth events per rare kind, a
# change of the pooled headline of <= HEADLINE_NOISE_DELTA between two runs of games shorter than
# DECISION_SECONDS is treated as noise unless a DECISION_SECONDS run agrees; per kind, a change whose
# "after" F1 lies inside the "before" 95% seed-bootstrap interval is flagged "inside CI".
HEADLINE_NOISE_DELTA = 0.03
DECISION_SECONDS = 1200.0


@dataclass(frozen=True)
class BenchCase:
    """One benchmark run: ``courtvision simulate --mode MODE --seed SEED --seconds SECONDS``."""

    mode: str
    seed: int
    seconds: float
    split: str = "train"
    config: str | None = None  # --config FILE passed through to the run
    overrides: tuple[str, ...] = ()  # --set section.key=value passed through to the run
    keep_dir: str | None = None  # write the run directory under here (train split only)

    @property
    def label(self) -> str:
        return f"{self.mode} seed {self.seed} ({self.seconds:g} s)"


# --------------------------------------------------------------------------- one run


def _count(ratio: float | None, n: int) -> int:
    """Recover the integer numerator of ``ratio = k / n`` (eval stores ratios, not counts)."""
    return 0 if ratio is None or n <= 0 else int(round(ratio * n))


def _kind_counts(r: KindReport, match_player: bool) -> dict[str, Any]:
    return {
        "n_pred": r.n_pred,
        "n_truth": r.n_truth,
        "tp": r.tp,
        "player_ok": _count(r.player_accuracy, r.tp) if match_player else None,
        "attributed_tp": _count(r.attributed_precision, r.n_pred) if match_player else None,
        "abs_dt_sum": (r.mean_abs_dt or 0.0) * r.tp,
        "attr": {k: {"ok": _count(acc, r.attr_n.get(k, 0)), "n": r.attr_n.get(k, 0)}
                 for k, acc in r.attr_accuracy.items()},
    }


def _shot_counts(ev: EventReport) -> dict[str, Any]:
    s = ev.shots
    return {
        "n_truth_shots": s.n_truth_shots,
        "n_pred_shots": s.n_pred_shots,
        "n_paired": s.n_paired,
        "outcome_confusion": {k: dict(v) for k, v in s.outcome_confusion.items()},
        "value_confusion": {k: dict(v) for k, v in s.value_confusion.items()},
        "fg_outcome_n": s.fg_outcome_n,
        "ft_ok": _count(s.ft_outcome_accuracy, s.ft_outcome_n),
        "ft_n": s.ft_outcome_n,
        "value_n": s.value_n,
        "made_fg_tp": s.made_fg_tp,
        "n_pred_made_fg": s.n_pred_made_fg,
        "n_truth_made_fg": s.n_truth_made_fg,
        "assist_ok": _count(s.assist_accuracy, s.assist_n),
        "assist_n": s.assist_n,
        "shooter_ok": _count(s.shooter_accuracy, s.shooter_n),
        "shooter_n": s.shooter_n,
    }


def _box_counts(box: BoxReport) -> dict[str, Any]:
    return {
        "mean_mae": box.mean_mae,
        "per_stat": {k: {"n": e.n, "abs_sum": e.mae * e.n, "pred_total": e.pred_total, "truth_total": e.truth_total}
                     for k, e in box.per_stat.items()},
        "n_truth_players": box.n_truth_players,
        "exact_lines": _count(box.line_exact_rate, box.n_truth_players),
        "team_points_abs_error": box.team_points_abs_error,
        "team_points": {t: dict(r) for t, r in box.team_points.items()},
        "score_match": box.score_match,
    }


def _track_counts(tr: TrackReport) -> dict[str, Any]:
    return {"gt": tr.gt, "tp": tr.tp, "fp": tr.fp, "fn": tr.fn, "idsw": tr.idsw, "idtp": tr.idtp, "idfp": tr.idfp,
            "idfn": tr.idfn, "mota": tr.mota, "idf1": tr.idf1, "rmse_ft": tr.rmse_ft,
            "identity_accuracy": tr.identity_accuracy, "team_accuracy": tr.team_accuracy, "n_frames": tr.n_frames,
            "ball_detection_rate": tr.ball.detection_rate}


def window_counts(pred: Iterable[EventLike], truth: Iterable[EventLike], report: EventReport,
                  width_s: float = WINDOW_S) -> list[dict[str, Any]]:
    """Per-kind counts of one run in non-cumulative ``width_s`` windows of stream time (aggregates only).

    Each window is matched on its own with eval's settings of the whole run (tolerance, player matching,
    operator lags) and with its edges as cut edges, so an event whose partner lies across an inner edge is
    "don't care" there (``boundary_excluded``) instead of a false positive in one window and a false
    negative in the next. The run's own cut edges (``report.window``) stay cut edges."""
    pred, truth = latest_revisions(pred), latest_revisions(truth)  # (a revision may move an event's t)
    times = [e.t for e in [*pred, *truth]]
    if not times or width_s <= 0:
        return []
    lo_cut, hi_cut = report.window if report.window is not None else (None, None)
    start = float(lo_cut) if lo_cut is not None else 0.0
    end = float(hi_cut) if hi_cut is not None else max(times)
    cfg = EventEvalConfig(operator_lag_s=dict(report.operator_lag_s))
    out: list[dict[str, Any]] = []
    n = max(1, int(math.ceil((end - start) / width_s - 1e-9)))
    for k in range(n):
        w0, w1 = start + k * width_s, min(end, start + (k + 1) * width_s)
        last = k == n - 1
        hi = w1 + 1e-9 if last else w1 - 1e-12
        inside = lambda e, a=w0, b=hi: a - 1e-9 <= e.t <= b  # noqa: E731
        rep = match_events([e for e in pred if inside(e)], [e for e in truth if inside(e)], tol_s=report.tol_s,
                           match_player=report.match_player, config=cfg,
                           window=(w0 if (k > 0 or lo_cut is not None) else None,
                                   w1 if (not last or hi_cut is not None) else None))
        out.append({"start": w0, "end": w1,
                    "kinds": {kind: _kind_counts(r, rep.match_player) for kind, r in rep.per_kind.items()}})
    return out


def record_from_reports(event_report: EventReport | None, box_report: BoxReport | None = None,
                        track_report: TrackReport | None = None,
                        windows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The aggregate-only record of one run (JSON-ready): counts to pool, plus eval's own per-run numbers."""
    rec: dict[str, Any] = {"headline_f1": None, "headline_attributed_f1": None, "headline_kinds": [],
                           "per_kind_f1": {}, "kinds": {}, "separate": {}, "shots": None, "box": None,
                           "tracking": None, "windows": list(windows or [])}
    if event_report is not None:
        ev = event_report
        rec["headline_f1"] = ev.headline_macro_f1
        rec["headline_attributed_f1"] = ev.headline_attributed_f1
        rec["headline_kinds"] = list(ev.headline_kinds)
        rec["match_player"] = ev.match_player
        rec["tol_s"] = ev.tol_s
        rec["per_kind_f1"] = {k: r.f1 for k, r in ev.per_kind.items()}
        rec["kinds"] = {k: _kind_counts(r, ev.match_player) for k, r in ev.per_kind.items()}
        rec["separate"] = {k: _kind_counts(r, ev.match_player) for k, r in ev.separate.items()}
        rec["shots"] = _shot_counts(ev)
    if box_report is not None:
        rec["box"] = _box_counts(box_report)
    if track_report is not None:
        rec["tracking"] = _track_counts(track_report)
    return rec


def quiet_worker_logging() -> None:
    """Process-pool initializer: runs log warnings only (the parent prints progress)."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("courtvision").setLevel(logging.WARNING)


def run_case(case: BenchCase) -> dict[str, Any]:
    """Run one case through the ``courtvision simulate`` code path; returns its aggregate record.

    Module-level so a process pool can pickle it. Raises whatever the run raises.
    """
    from courtvision.cli import simulate_config
    from courtvision.pipeline import run_offline

    t0 = time.perf_counter()
    cfg = simulate_config(case.mode, seed=case.seed, seconds=case.seconds, config=case.config,
                          overrides=case.overrides)
    out: Path | None = None
    if case.keep_dir is not None:
        if case.split != "train":
            raise ValueError("held-out runs keep aggregate metrics only (no run directories)")
        out = Path(case.keep_dir) / f"sim-{case.mode}-seed{case.seed}"
    result = run_offline(cfg, out_dir=out, write=out is not None)
    windows = None
    if result.event_report is not None and result.truth_events is not None:
        windows = window_counts(result.events, result.truth_events, result.event_report)
    rec = record_from_reports(result.event_report, result.box_report, result.track_report, windows)
    lat = result.latency or {}
    rec.update(mode=case.mode, seed=case.seed, seconds=case.seconds, split=case.split, frames=result.frames,
               source_s=lat.get("source_s"), wall_s=time.perf_counter() - t0,
               run_dir=str(result.run_dir) if result.run_dir is not None else None)
    return rec


# --------------------------------------------------------------------------- pooling


def _ratio(num: float, den: float) -> float | None:
    return num / den if den > 0 else None


def _f1(tp: int, n_pred: int, n_truth: int) -> float | None:  # eval.events._f1
    return 2.0 * tp / (n_pred + n_truth) if n_pred + n_truth > 0 else None


def _sum_conf(tables: Iterable[Mapping[str, Mapping[str, int]]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for table in tables:
        for row, cols in table.items():
            for col, n in cols.items():
                out.setdefault(row, {})
                out[row][col] = out[row].get(col, 0) + int(n)
    return out


def _pool_kinds(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    kinds: list[str] = []
    for rec in records:
        kinds += [k for k in rec.get(key, {}) if k not in kinds]
    out: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        rows = [rec[key][kind] for rec in records if kind in rec.get(key, {})]
        tp = sum(r["tp"] for r in rows)
        n_pred = sum(r["n_pred"] for r in rows)
        n_truth = sum(r["n_truth"] for r in rows)
        attributed = None if any(r["attributed_tp"] is None for r in rows) else sum(r["attributed_tp"] for r in rows)
        player_ok = None if any(r["player_ok"] is None for r in rows) else sum(r["player_ok"] for r in rows)
        attrs: dict[str, dict[str, Any]] = {}
        for r in rows:
            for name, a in r["attr"].items():
                cur = attrs.setdefault(name, {"ok": 0, "n": 0})
                cur["ok"] += a["ok"]
                cur["n"] += a["n"]
        out[kind] = {
            "n_truth": n_truth,
            "n_pred": n_pred,
            "tp": tp,
            "fp": n_pred - tp,
            "fn": n_truth - tp,
            "precision": _ratio(tp, n_pred),
            "recall": _ratio(tp, n_truth),
            "f1": _f1(tp, n_pred, n_truth),
            "player_accuracy": _ratio(player_ok, tp) if player_ok is not None else None,
            "attributed_f1": _f1(attributed, n_pred, n_truth) if attributed is not None else None,
            "mean_abs_dt": _ratio(sum(r["abs_dt_sum"] for r in rows), tp),
            "attr_accuracy": {k: {"accuracy": _ratio(a["ok"], a["n"]), "n": a["n"]} for k, a in attrs.items()},
        }
    return out


def _pool_shots(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [rec["shots"] for rec in records if rec.get("shots")]
    conf = _sum_conf(r["outcome_confusion"] for r in rows)
    vconf = _sum_conf(r["value_confusion"] for r in rows)
    fg_n = sum(r["fg_outcome_n"] for r in rows)
    fg_ok = sum(conf.get(o, {}).get(o, 0) for o in ("made", "missed"))
    value_n = sum(r["value_n"] for r in rows)
    value_ok = sum(vconf.get(v, {}).get(v, 0) for v in ("2", "3"))
    tp = sum(r["made_fg_tp"] for r in rows)
    n_pred = sum(r["n_pred_made_fg"] for r in rows)
    n_truth = sum(r["n_truth_made_fg"] for r in rows)

    def acc(ok: str, n: str) -> dict[str, Any]:
        k, m = sum(r[ok] for r in rows), sum(r[n] for r in rows)
        return {"accuracy": _ratio(k, m), "n": m}

    return {
        "n_truth_shots": sum(r["n_truth_shots"] for r in rows),
        "n_pred_shots": sum(r["n_pred_shots"] for r in rows),
        "n_paired": sum(r["n_paired"] for r in rows),
        "fg_make_miss": {"accuracy": _ratio(fg_ok, fg_n), "n": fg_n, "confusion": conf},
        "ft_make_miss": acc("ft_ok", "ft_n"),
        "two_vs_three": {"accuracy": _ratio(value_ok, value_n), "n": value_n, "confusion": vconf},
        "made_fg": {"precision": _ratio(tp, n_pred), "recall": _ratio(tp, n_truth), "f1": _f1(tp, n_pred, n_truth),
                    "tp": tp, "n_pred": n_pred, "n_truth": n_truth},
        "assist_label": acc("assist_ok", "assist_n"),
        "shooter": acc("shooter_ok", "shooter_n"),
    }


def _pool_box(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    rows = [rec["box"] for rec in records if rec.get("box")]
    if not rows:
        return None
    stats: list[str] = []
    for r in rows:
        stats += [s for s in r["per_stat"] if s not in stats]
    per_stat = {}
    for s in stats:
        n = sum(r["per_stat"][s]["n"] for r in rows if s in r["per_stat"])
        per_stat[s] = {"mae": _ratio(sum(r["per_stat"][s]["abs_sum"] for r in rows if s in r["per_stat"]), n),
                       "n": n,
                       "pred_total": sum(r["per_stat"][s]["pred_total"] for r in rows if s in r["per_stat"]),
                       "truth_total": sum(r["per_stat"][s]["truth_total"] for r in rows if s in r["per_stat"])}
    maes = [v["mae"] for v in per_stat.values() if v["mae"] is not None]
    seed_maes = [r["mean_mae"] for r in rows if r["mean_mae"] is not None]
    errs = [r["team_points_abs_error"] for r in rows if r["team_points_abs_error"] is not None]
    players = sum(r["n_truth_players"] for r in rows)
    return {
        "n_games": len(rows),
        "mean_mae": sum(maes) / len(maes) if maes else None,  # eval's mean_mae over pooled per-stat MAE
        "mean_mae_seed_mean": statistics.fmean(seed_maes) if seed_maes else None,
        "per_stat": per_stat,
        "line_exact_rate": _ratio(sum(r["exact_lines"] for r in rows), players),
        "team_points_abs_error_mean": statistics.fmean(errs) if errs else None,
        "team_points_abs_error_max": max(errs) if errs else None,
        "team_points_abs_error_total": sum(errs) if errs else None,
        "score_match_rate": _ratio(sum(1 for r in rows if r["score_match"]), len(rows)),
    }


def _pool_tracking(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    rows = [rec["tracking"] for rec in records if rec.get("tracking")]
    if not rows:
        return None
    s = {k: sum(r[k] for r in rows) for k in ("gt", "tp", "fp", "fn", "idsw", "idtp", "idfp", "idfn", "n_frames")}

    def mean(key: str) -> float | None:
        vals = [r[key] for r in rows if r[key] is not None]
        return statistics.fmean(vals) if vals else None

    return {
        "mota": 1.0 - (s["fn"] + s["fp"] + s["idsw"]) / s["gt"] if s["gt"] else None,
        "idf1": _ratio(2 * s["idtp"], 2 * s["idtp"] + s["idfp"] + s["idfn"]),
        "idsw": s["idsw"],
        "rmse_ft_seed_mean": mean("rmse_ft"),
        "identity_accuracy_seed_mean": mean("identity_accuracy"),
        "team_accuracy_seed_mean": mean("team_accuracy"),
        "ball_detection_rate_seed_mean": mean("ball_detection_rate"),
        "n_frames": s["n_frames"],
    }


def _headlines(per_kind: Mapping[str, Mapping[str, Any]], core_kinds: Sequence[str]) -> dict[str, Any]:
    """The strict headline (eval's macro-F1) and its companions from pooled per-kind counts."""
    head = [k for k in core_kinds if k in per_kind and per_kind[k]["f1"] is not None]
    supported = [k for k in head if per_kind[k]["n_truth"] > 0]
    return {
        "head": head,
        "headline_f1": statistics.fmean(per_kind[k]["f1"] for k in head) if head else None,
        "supported_kinds": supported,
        "headline_supported_f1": statistics.fmean(per_kind[k]["f1"] for k in supported) if supported else None,
        "micro_f1": _f1(sum(per_kind[k]["tp"] for k in head), sum(per_kind[k]["n_pred"] for k in head),
                        sum(per_kind[k]["n_truth"] for k in head)),
        "unsupported_fp": {k: per_kind[k]["fp"] for k in head if per_kind[k]["n_truth"] == 0},
    }


def bootstrap_ci(records: Sequence[Mapping[str, Any]], core_kinds: Sequence[str] = CORE_KINDS,
                 n: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """95% percentile intervals from resampling the games (records) with replacement: the headline, the
    supported-kind macro, the micro-F1 and each kind's pooled F1 (None where a resample has no data)."""
    import random

    records = list(records)
    if len(records) < 2:
        return {}
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(n):
        pick = [records[rng.randrange(len(records))] for _ in records]
        per_kind = _pool_kinds(pick, "kinds")
        h = _headlines(per_kind, core_kinds)
        for key in ("headline_f1", "headline_supported_f1", "micro_f1"):
            if h[key] is not None:
                samples.setdefault(key, []).append(h[key])
        for k, v in per_kind.items():
            if v["f1"] is not None:
                samples.setdefault(f"kind:{k}", []).append(v["f1"])

    def ci(vals: list[float]) -> list[float]:
        vals = sorted(vals)
        lo = vals[int(0.025 * (len(vals) - 1))]
        hi = vals[int(round(0.975 * (len(vals) - 1)))]
        return [lo, hi]

    out: dict[str, Any] = {k: ci(v) for k, v in samples.items() if not k.startswith("kind:")}
    out["per_kind_f1"] = {k[5:]: ci(v) for k, v in samples.items() if k.startswith("kind:")}
    out["n_resamples"] = n
    return out


def pool_windows(records: Sequence[Mapping[str, Any]], core_kinds: Sequence[str] = CORE_KINDS
                 ) -> list[dict[str, Any]]:
    """The pooled headline (and its companions) per non-cumulative window index, over the runs that reach
    that window: ``[{"start", "end", "n_runs", "headline_f1", "headline_supported_f1", "micro_f1",
    "core_truth"}]``. A flat line across windows means no degradation over time."""
    n = max((len(r.get("windows") or []) for r in records), default=0)
    out: list[dict[str, Any]] = []
    for k in range(n):
        rows = [r["windows"][k] for r in records if len(r.get("windows") or []) > k]
        per_kind = _pool_kinds(rows, "kinds")
        h = _headlines(per_kind, core_kinds)
        out.append({"start": min(w["start"] for w in rows), "end": max(w["end"] for w in rows), "n_runs": len(rows),
                    "headline_f1": h["headline_f1"], "headline_supported_f1": h["headline_supported_f1"],
                    "micro_f1": h["micro_f1"],
                    "core_truth": {kk: per_kind[kk]["n_truth"] for kk in core_kinds if kk in per_kind}})
    return out


def _window_spread(windows: Sequence[Mapping[str, Any]], min_runs: int) -> float | None:
    """Max - min pooled headline over the windows every run reaches (a partial last window is left out)."""
    vals = [w["headline_f1"] for w in windows if w["n_runs"] >= min_runs and w["headline_f1"] is not None]
    return max(vals) - min(vals) if len(vals) >= 2 else None


def pool_mode(records: Sequence[Mapping[str, Any]], core_kinds: Sequence[str] = CORE_KINDS,
              bootstrap: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Pool the per-run records of one mode (one record per seed)."""
    records = sorted(records, key=lambda r: r.get("seed", 0))
    per_kind = _pool_kinds(records, "kinds")
    heads = _headlines(per_kind, core_kinds)
    head = heads["head"]
    headline = heads["headline_f1"]
    match_player = all(r.get("match_player", True) for r in records)  # eval: None without player matching
    attributed = statistics.fmean(per_kind[k]["attributed_f1"] or 0.0 for k in head) if head and match_player else None
    seed_head = [r["headline_f1"] for r in records if r.get("headline_f1") is not None]
    shots = _pool_shots(records)
    reb = per_kind.get("rebound", {})
    ast = per_kind.get("assist", {})
    box = _pool_box(records)
    seconds = sorted({float(r["seconds"]) for r in records if "seconds" in r})
    minutes = sum(float(r.get("seconds") or 0.0) for r in records) / 60.0
    return {
        "seeds": [r.get("seed") for r in records],
        "seconds": seconds[0] if len(seconds) == 1 else seconds,
        "n_runs": len(records),
        "headline_f1": headline,  # eval's macro-F1 definition on pooled per-kind counts
        "headline_kinds": head,
        "headline_attributed_f1": attributed,
        "headline_f1_seed_mean": statistics.fmean(seed_head) if seed_head else None,
        "headline_f1_seed_sd": statistics.stdev(seed_head) if len(seed_head) > 1 else None,
        "headline_f1_per_seed": {str(r.get("seed")): r.get("headline_f1") for r in records},
        "headline_supported_f1": heads["headline_supported_f1"],  # macro over core kinds with truth support
        "supported_kinds": heads["supported_kinds"],
        "micro_f1": heads["micro_f1"],  # 2 sum TP / (sum pred + sum truth) over the headline kinds
        "stream_minutes": minutes,
        # core kinds with no truth at all: their false positives per stream minute (each enters the
        # headline as F1 0)
        "unsupported_fp_per_min": {k: (fp / minutes if minutes > 0 else None)
                                   for k, fp in heads["unsupported_fp"].items()},
        "ci95": bootstrap_ci(records, core_kinds, n=bootstrap) if bootstrap > 0 else {},
        "per_kind_f1": {k: v["f1"] for k, v in per_kind.items()},
        "per_kind": per_kind,
        "separate": _pool_kinds(records, "separate"),
        "shots": shots,
        "shot_make_miss_accuracy": shots["fg_make_miss"]["accuracy"],
        "two_vs_three_accuracy": shots["two_vs_three"]["accuracy"],
        "rebound": {"f1": reb.get("f1"), "n_truth": reb.get("n_truth", 0),
                    "rebounder_accuracy": reb.get("player_accuracy"), "attributed_f1": reb.get("attributed_f1"),
                    "offensive_label": (reb.get("attr_accuracy") or {}).get("offensive")},
        "assist": {"f1": ast.get("f1"), "n_truth": ast.get("n_truth", 0),
                   "passer_accuracy": ast.get("player_accuracy"), "attributed_f1": ast.get("attributed_f1"),
                   "label_on_made_fg": shots["assist_label"]},
        "box": box,
        "box_mean_mae": box["mean_mae"] if box else None,
        "team_points_error": box["team_points_abs_error_mean"] if box else None,
        "tracking": _pool_tracking(records),
        # time trend: pooled headline per non-cumulative WINDOW_S window, and its spread (windows every run
        # reaches); compare the spread with the whole-run ci95 headline interval
        "window_s": WINDOW_S,
        "windows": (wins := pool_windows(records, core_kinds)),
        "window_headline_spread": _window_spread(wins, len(records)),
        "frames": sum(int(r.get("frames") or 0) for r in records),
        "wall_s": sum(float(r.get("wall_s") or 0.0) for r in records),
    }


# --------------------------------------------------------------------------- report


@dataclass
class BenchReport:
    """Everything one benchmark invocation produced. ``to_dict()`` is the JSON file."""

    split: str
    command: str
    modes: dict[str, dict[str, Any]]  # mode -> pooled summary (pool_mode) + "runs"
    errors: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    wall_s: float = 0.0
    created: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d = {"bench": "courtvision accuracy_bench", "version": BENCH_VERSION, **d}
        d["protocol"] = {"train_seeds": list(TRAIN_SEEDS), "heldout_seeds": list(HELDOUT_SEEDS),
                         "core_kinds": list(CORE_KINDS),
                         "pooling": "per-kind F1 = 2*sum(TP)/(sum(pred)+sum(truth)) over seeds; headline_f1 = "
                                    "mean pooled F1 over core kinds that occur (eval's macro-F1 definition); "
                                    "headline_supported_f1 = the same over core kinds with truth; micro_f1 = "
                                    "2*sum(TP)/(sum(pred)+sum(truth)) over the headline kinds; ci95 = 95% "
                                    "percentile intervals from resampling the games with replacement",
                         "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                         "comparability": "pooled multi-seed numbers of >= 180-300 s games only; single-seed "
                                          "<= 60 s headlines are not comparable (rare kinds come and go)"}
        return d


def build_report(records: Sequence[Mapping[str, Any]], *, split: str, command: str = "",
                 errors: Sequence[Mapping[str, Any]] = (), config: Mapping[str, Any] | None = None,
                 wall_s: float = 0.0) -> BenchReport:
    """Group per-run records by mode (in :data:`MODES` order) and pool each group."""
    modes: dict[str, dict[str, Any]] = {}
    for mode in [*MODES, *sorted({r["mode"] for r in records} - set(MODES))]:
        group = [r for r in records if r["mode"] == mode]
        if not group:
            continue
        pooled = pool_mode(group)
        runs = []
        for r in sorted(group, key=lambda r: r["seed"]):
            box = r.get("box") or {}
            shots = r.get("shots") or {}
            conf = shots.get("outcome_confusion") or {}
            vconf = shots.get("value_confusion") or {}
            tr = r.get("tracking") or {}
            runs.append({
                "seed": r["seed"], "seconds": r.get("seconds"), "frames": r.get("frames"), "wall_s": r.get("wall_s"),
                "headline_f1": r.get("headline_f1"), "headline_attributed_f1": r.get("headline_attributed_f1"),
                "per_kind_f1": dict(r.get("per_kind_f1") or {}),
                "fg_make_miss_accuracy": _ratio(sum(conf.get(o, {}).get(o, 0) for o in ("made", "missed")),
                                                shots.get("fg_outcome_n", 0)),
                "two_vs_three_accuracy": _ratio(sum(vconf.get(v, {}).get(v, 0) for v in ("2", "3")),
                                                shots.get("value_n", 0)),
                "box_mean_mae": box.get("mean_mae"), "team_points_abs_error": box.get("team_points_abs_error"),
                "score_match": box.get("score_match"), "mota": tr.get("mota"), "run_dir": r.get("run_dir"),
            })
        pooled["mode"] = mode
        pooled["split"] = split
        pooled["runs"] = runs
        modes[mode] = pooled
    return BenchReport(split=split, command=command, modes=modes, errors=[dict(e) for e in errors],
                       config=dict(config or {}), wall_s=wall_s, created=time.strftime("%Y-%m-%dT%H:%M:%S"))


def _f(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _acc(d: Mapping[str, Any] | None) -> str:
    if not d:
        return "n/a"
    return f"{_f(d.get('accuracy'))} (n={d.get('n', 0)})"


def format_markdown(report: BenchReport | Mapping[str, Any]) -> str:
    """The benchmark's markdown: a headline row per mode, then pooled per-kind F1 by mode."""
    d = report.to_dict() if isinstance(report, BenchReport) else dict(report)
    modes: dict[str, Any] = d["modes"]
    lines = [f"# CourtVision accuracy bench ({d['split']} split)", ""]
    if d.get("command"):
        lines += [f"`{d['command']}`", ""]
    lines += ["| mode | seeds | s/game | headline F1 (pooled) | seed mean ± sd | attr. F1 | FG make/miss | 2 vs 3 "
              "| made-FG F1 | rebound F1 / rebounder | assist F1 / label | box MAE | team pts err (mean / max) "
              "| score match | wall s |",
              "|" + "---|" * 15]
    companions = ["", "| mode | headline 95% CI | supported-kind macro F1 [95% CI] | micro F1 [95% CI] "
                      "| FP / min of core kinds without truth |", "|" + "---|" * 5]
    for mode, m in modes.items():
        sd = m.get("headline_f1_seed_sd")
        box = m.get("box") or {}
        reb, ast = m["rebound"], m["assist"]
        secs = m["seconds"] if isinstance(m["seconds"], list) else [m["seconds"]]
        cells = [
            mode, ",".join(str(s) for s in m["seeds"]), "/".join(f"{float(s):g}" for s in secs),
            f"**{_f(m['headline_f1'])}**",
            f"{_f(m['headline_f1_seed_mean'])} ± {_f(sd)}" if sd is not None else _f(m["headline_f1_seed_mean"]),
            _f(m["headline_attributed_f1"]),
            _acc(m["shots"]["fg_make_miss"]), _acc(m["shots"]["two_vs_three"]), _f(m["shots"]["made_fg"]["f1"]),
            f"{_f(reb['f1'])} / {_f(reb['rebounder_accuracy'])}",
            f"{_f(ast['f1'])} / {_f(ast['label_on_made_fg'].get('accuracy'))}",
            _f(box.get("mean_mae")),
            f"{_f(box.get('team_points_abs_error_mean'), 2)} / {_f(box.get('team_points_abs_error_max'))}",
            _f(box.get("score_match_rate"), 2), _f(m["wall_s"], 0),
        ]
        lines.append("| " + " | ".join(cells) + " |")
        ci = m.get("ci95") or {}

        def iv(key: str) -> str:
            v = ci.get(key)
            return f"[{_f(v[0])}, {_f(v[1])}]" if v else "n/a"

        fpm = m.get("unsupported_fp_per_min") or {}
        companions.append(" | ".join([
            f"| {mode}", iv("headline_f1"), f"{_f(m.get('headline_supported_f1'))} {iv('headline_supported_f1')}",
            f"{_f(m.get('micro_f1'))} {iv('micro_f1')}",
            ", ".join(f"{k} {_f(v, 2)}" for k, v in fpm.items()) or "none"]) + " |")
    lines += companions
    lines += ["", "Only pooled multi-seed numbers of >= 180-300 s games are comparable; a single-seed <= 60 s "
                  "headline is not comparable (rare kinds with 0-15 truth events enter or leave its average).", ""]
    for mode, m in modes.items():
        wins = m.get("windows") or []
        if len(wins) < 2:
            continue
        full = [w for w in wins if w["n_runs"] == m["n_runs"]]
        ci = (m.get("ci95") or {}).get("headline_f1")
        cells = ", ".join(f"{_f(w['headline_f1'])}" + ("" if w["n_runs"] == m["n_runs"] else f" (n={w['n_runs']})")
                          for w in wins)
        lines.append(f"Time trend ({mode}): pooled headline per non-cumulative {m.get('window_s', WINDOW_S):g} s "
                     f"window: {cells}; spread {_f(m.get('window_headline_spread'))} over {len(full)} full windows "
                     f"(whole-run 95% CI {f'[{_f(ci[0])}, {_f(ci[1])}]' if ci else 'n/a'}).")
    if any(len(m.get("windows") or []) >= 2 for m in modes.values()):
        lines.append("")
    kinds: list[str] = []
    for m in modes.values():
        kinds += [k for k in m["per_kind"] if k not in kinds]
    order = [k for k in CORE_KINDS if k in kinds] + [k for k in kinds if k not in CORE_KINDS]
    if order:
        names = list(modes)
        lines += ["## Per-kind F1 pooled over seeds (truth n in brackets, then the 95% seed-bootstrap interval; "
                  "* = headline kind)", "",
                  "| kind | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for k in order:
            cells = []
            for name in names:
                r = modes[name]["per_kind"].get(k)
                civ = ((modes[name].get("ci95") or {}).get("per_kind_f1") or {}).get(k)
                band = f" ({_f(civ[0], 2)}-{_f(civ[1], 2)})" if civ else ""
                cells.append(f"{_f(r['f1'])} [{r['n_truth']}]{band}" if r else "—")
            star = "*" if k in CORE_KINDS else ""
            lines.append(f"| {k}{star} | " + " | ".join(cells) + " |")
        lines.append("")
    tracked = {name: m["tracking"] for name, m in modes.items() if m.get("tracking")}
    for name, tr in tracked.items():
        lines += [f"Tracking ({name}, pooled): MOTA {_f(tr['mota'])}, IDF1 {_f(tr['idf1'])}, ID switches "
                  f"{tr['idsw']}, identity accuracy {_f(tr['identity_accuracy_seed_mean'])} (seed mean), "
                  f"RMSE {_f(tr['rmse_ft_seed_mean'], 2)} ft.", ""]
    for err in d.get("errors") or []:
        lines.append(f"- FAILED {err.get('mode')} seed {err.get('seed')}: {err.get('error')}")
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- before / after comparison


def compare_reports(before: BenchReport | Mapping[str, Any], after: BenchReport | Mapping[str, Any]
                    ) -> dict[str, Any]:
    """Deltas between two bench reports, mode by mode, with the truth n and the seed-bootstrap intervals
    next to every delta and a verdict (protocol: :data:`HEADLINE_NOISE_DELTA`, :data:`DECISION_SECONDS`):

    - headline ``noise``: ``|delta| <= HEADLINE_NOISE_DELTA`` on games shorter than ``DECISION_SECONDS``
      (decide on a ``DECISION_SECONDS`` run instead); ``inside CI``: the after headline lies in the before
      95% interval; otherwise ``change``;
    - per kind ``inside CI`` / ``change`` the same way (``n/a`` without an interval).
    """
    b = before.to_dict() if isinstance(before, BenchReport) else dict(before)
    a = after.to_dict() if isinstance(after, BenchReport) else dict(after)
    out: dict[str, Any] = {"before": b.get("command", ""), "after": a.get("command", ""), "modes": {}}

    def inside(v: float | None, iv: Sequence[float] | None) -> bool | None:
        if v is None or not iv:
            return None
        return iv[0] - 1e-12 <= v <= iv[1] + 1e-12

    for mode in [m for m in a.get("modes", {}) if m in b.get("modes", {})]:
        mb, ma = b["modes"][mode], a["modes"][mode]
        hb, ha = mb.get("headline_f1"), ma.get("headline_f1")
        delta = None if hb is None or ha is None else ha - hb
        secs = [float(x) for x in (mb["seconds"] if isinstance(mb["seconds"], list) else [mb["seconds"]])] + \
            [float(x) for x in (ma["seconds"] if isinstance(ma["seconds"], list) else [ma["seconds"]])]
        short = min(secs) < DECISION_SECONDS - 1e-9
        ci_b, ci_a = (mb.get("ci95") or {}), (ma.get("ci95") or {})
        if delta is None:
            verdict = "n/a"
        elif short and abs(delta) <= HEADLINE_NOISE_DELTA + 1e-12:
            verdict = "noise"
        elif inside(ha, ci_b.get("headline_f1")):
            verdict = "inside CI"
        else:
            verdict = "change"
        kinds = {}
        for k in [k for k in ma.get("per_kind", {}) if k in mb.get("per_kind", {})]:
            kb, ka = mb["per_kind"][k], ma["per_kind"][k]
            civ = (ci_b.get("per_kind_f1") or {}).get(k)
            d = None if kb["f1"] is None or ka["f1"] is None else ka["f1"] - kb["f1"]
            ins = inside(ka["f1"], civ)
            kinds[k] = {"before": kb["f1"], "after": ka["f1"], "delta": d, "n_truth_before": kb["n_truth"],
                        "n_truth_after": ka["n_truth"], "ci_before": civ,
                        "ci_after": (ci_a.get("per_kind_f1") or {}).get(k),
                        "verdict": "n/a" if ins is None or d is None else ("inside CI" if ins else "change")}
        out["modes"][mode] = {"seconds": sorted(set(secs)), "headline_before": hb, "headline_after": ha,
                              "delta": delta, "ci_before": ci_b.get("headline_f1"), "ci_after": ci_a.get("headline_f1"),
                              "verdict": verdict, "kinds": kinds}
    return out


def format_comparison(cmp: Mapping[str, Any]) -> str:
    """Markdown of :func:`compare_reports`."""
    def iv(v: Sequence[float] | None) -> str:
        return f"[{_f(v[0])}, {_f(v[1])}]" if v else "n/a"

    def sgn(v: float | None) -> str:
        return "n/a" if v is None else f"{v:+.3f}"

    lines = ["# Bench comparison", "", f"- before: `{cmp.get('before')}`", f"- after: `{cmp.get('after')}`", "",
             f"Rule: a headline change of <= {HEADLINE_NOISE_DELTA:g} on games shorter than {DECISION_SECONDS:g} s is "
             f"noise unless a {DECISION_SECONDS:g} s run agrees; \"inside CI\" = the after value lies in the "
             "before 95% seed-bootstrap interval.", "",
             "| mode | s/game | headline before [95% CI] | after [95% CI] | delta | verdict |", "|---|---|---|---|---|---|"]
    for mode, m in cmp["modes"].items():
        lines.append(f"| {mode} | {'/'.join(f'{s:g}' for s in m['seconds'])} | {_f(m['headline_before'])} "
                     f"{iv(m['ci_before'])} | {_f(m['headline_after'])} {iv(m['ci_after'])} | {sgn(m['delta'])} "
                     f"| {m['verdict']} |")
    for mode, m in cmp["modes"].items():
        lines += ["", f"## {mode}: per kind (truth n before / after)", "",
                  "| kind | before [n] (95% CI) | after [n] (95% CI) | delta | verdict |", "|---|---|---|---|---|"]
        for k, r in m["kinds"].items():
            lines.append(f"| {k} | {_f(r['before'])} [{r['n_truth_before']}] {iv(r['ci_before'])} | {_f(r['after'])} "
                         f"[{r['n_truth_after']}] {iv(r['ci_after'])} | {sgn(r['delta'])} | {r['verdict']} |")
    return "\n".join(lines) + "\n"
