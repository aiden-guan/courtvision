"""Refit the xFG shot model on our own labelled shots, with a calibration report.

    python -m courtvision.training.fit_shot_model --csv shots.csv --league nba --out shot_model.json

CSV columns (one row per attempt; ``SHOT_COLUMNS``): ``distance_ft``, ``value`` (2/3; 1 = free throw,
skipped by the fit), ``made`` (1/0, true/false, made/missed), optional ``shot_type``,
``closest_defender_ft``, ``catch_and_shoot``, ``zone`` and ``game_id``.

With ``game_id`` the rows are split **by game, never by shot** (training_data §3.5): the model is fit on
the training games and the report is computed on the held-out games. The report (RESEARCH §4.4, §7A.2;
Guo et al. [70]) has 15-bin ECE with a bias-corrected bootstrap 95% CI (resampling games when known),
MCE, Brier, log-loss and the reliability table, for both the prior and the refit model. Target: ECE <= 0.03.

``calibration_report`` is shared with ``fit_cue_models``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from courtvision.events.calibration import ece as ece_point
from courtvision.stats.shot_quality import ShotFeatures, ShotQualityModel

__all__ = [
    "SHOT_COLUMNS",
    "FitShotConfig",
    "ShotRow",
    "calibration_report",
    "fit_shot_model",
    "main",
    "read_shots_csv",
    "split_by_game",
]

log = logging.getLogger(__name__)

SHOT_COLUMNS: tuple[str, ...] = ("distance_ft", "value", "made", "shot_type", "closest_defender_ft",
                                 "catch_and_shoot", "zone", "game_id")
_REQUIRED = ("distance_ft", "value", "made")


@dataclass
class FitShotConfig:
    """Fit and report settings."""

    league: str = "nba"
    l2: float = 1.0  # ridge toward the prior (ShotQualityModel.fit default); tuned
    holdout_frac: float = 0.25  # share of games held out for the report; tuned
    seed: int = 0
    bins: int = 15  # M = 15 equal-width bins (Guo et al. [70]; RESEARCH §4.4); sourced
    n_boot: int = 1000  # bootstrap resamples for the ECE CI; tuned
    target_ece: float = 0.03  # promote only if ECE <= 0.03 (training_data §5 Phase 5; RESEARCH §4.4); sourced (proposal)


@dataclass
class ShotRow:
    features: ShotFeatures
    made: bool
    game_id: str | None


# ---------------------------------------------------------------------------------------------- metrics


def _reliability(p: np.ndarray, y: np.ndarray, bins: int) -> list[dict[str, Any]]:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    table = []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        table.append({"bin": b, "lo": round(b / bins, 4), "hi": round((b + 1) / bins, 4), "n": n,
                      "mean_p": round(float(p[m].mean()), 4) if n else None,
                      "frac_pos": round(float(y[m].mean()), 4) if n else None})
    return table


def _ece_fast(p: np.ndarray, y: np.ndarray, bins: int) -> float:
    """Vectorised equal-width-bin ECE (same definition as ``events.calibration.ece``) for the bootstrap."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    n = np.bincount(idx, minlength=bins)
    gap = np.abs(np.bincount(idx, weights=p, minlength=bins) - np.bincount(idx, weights=y, minlength=bins))
    return float(gap.sum() / max(len(p), 1)) if n.any() else 0.0


def _bootstrap_ece(p: np.ndarray, y: np.ndarray, groups: Sequence[str | None] | None, bins: int,
                   n_boot: int, seed: int) -> tuple[np.ndarray | None, str]:
    """Bootstrap ECE draws; resamples whole games when every row has a game id."""
    if n_boot <= 0 or len(p) < 2:
        return None, "none"
    rng = np.random.default_rng(seed)
    if groups is not None and all(g is not None for g in groups) and len(set(groups)) >= 2:
        garr = np.asarray(groups, dtype=object)
        keys = sorted(set(groups))  # type: ignore[arg-type]
        members = [np.flatnonzero(garr == k) for k in keys]
        draws = []
        for _ in range(n_boot):
            idx = np.concatenate([members[i] for i in rng.integers(0, len(keys), len(keys))])
            draws.append(_ece_fast(p[idx], y[idx], bins))
        unit = "game"
    else:
        draws = [_ece_fast(p[i], y[i], bins) for i in (rng.integers(0, len(p), len(p)) for _ in range(n_boot))]
        unit = "event"
    return np.asarray(draws, dtype=float), unit


def _ece_intervals(e: float, draws: np.ndarray | None) -> dict[str, Any]:
    """Bias-corrected ("basic") and percentile 95% intervals, bootstrap bias and the debiased ECE.

    Binned ECE is biased upward at a few hundred events, and a resample (with duplicates) is biased
    further, so the percentile interval sits above the point estimate and almost never covers the
    population ECE of a calibrated model (0% in our simulation at n = 300-1,000). The basic interval
    [2e - q97.5, 2e - q2.5] (clipped at 0) corrects that shift; it is the headline ``ece_ci95``.
    """
    if draws is None or len(draws) == 0:
        return {"ece_ci95": None, "ece_ci95_percentile": None, "ece_bootstrap_bias": None, "ece_debiased": None}
    lo, hi = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    bias = float(np.mean(draws)) - e
    return {"ece_ci95": [round(max(0.0, 2 * e - hi), 5), round(max(0.0, 2 * e - lo), 5)],
            "ece_ci95_percentile": [round(lo, 5), round(hi, 5)], "ece_bootstrap_bias": round(bias, 5),
            "ece_debiased": round(max(0.0, e - bias), 5)}


def calibration_report(p: Sequence[float], y: Sequence[bool | int], bins: int = 15, n_boot: int = 1000,
                       seed: int = 0, groups: Sequence[str | None] | None = None,
                       target_ece: float | None = 0.03) -> dict[str, Any]:
    """15-bin ECE (+ bootstrap 95% CI), MCE, Brier, log-loss and the reliability table.

    ECE = sum_m |B_m|/n * |acc(B_m) - conf(B_m)| with equal-width bins (Guo et al. [70]); the point
    estimate is ``events.calibration.ece``. ``groups`` (game ids) makes the bootstrap resample games.
    ``ece_ci95`` is the bias-corrected (basic) bootstrap interval; ``ece_ci95_percentile``,
    ``ece_bootstrap_bias`` and ``ece_debiased`` are reported next to it (``_ece_intervals``).
    """
    raw = np.asarray(p, dtype=float)
    if not np.all(np.isfinite(raw)):
        # NaN would land in a garbage bin and turn every metric (and the JSON report) into NaN silently.
        raise ValueError(f"{int(np.sum(~np.isfinite(raw)))} non-finite probabilities; fix the model before reporting")
    pa = np.clip(raw, 0.0, 1.0)
    ya = np.asarray([1.0 if bool(v) else 0.0 for v in y])
    if len(pa) != len(ya):
        raise ValueError("p and y differ in length")
    if groups is not None and len(groups) != len(pa):
        raise ValueError("groups and p differ in length")
    if len(pa) == 0:
        return {"n": 0, "ece": None, **_ece_intervals(0.0, None), "ci_method": "basic", "mce": None, "brier": None,
                "log_loss": None, "base_rate": None, "mean_p": None, "reliability": [], "bootstrap_unit": "none",
                "meets_target": False}
    table = _reliability(pa, ya, bins)
    gaps = [abs(r["mean_p"] - r["frac_pos"]) for r in table if r["n"]]
    draws, unit = _bootstrap_ece(pa, ya, groups, bins, n_boot, seed)
    eps = 1e-12
    e = ece_point(pa, ya, bins)
    out = {
        "n": int(len(pa)), "bins": bins, "ece": round(float(e), 5), **_ece_intervals(float(e), draws),
        "ci_method": "basic", "bootstrap_unit": unit,
        "n_boot": int(n_boot), "mce": round(float(max(gaps)), 5),
        "brier": round(float(np.mean((pa - ya) ** 2)), 5),
        "log_loss": round(float(-np.mean(ya * np.log(np.clip(pa, eps, 1)) + (1 - ya) * np.log(np.clip(1 - pa, eps, 1)))), 5),
        "base_rate": round(float(ya.mean()), 5), "mean_p": round(float(pa.mean()), 5), "reliability": table,
    }
    if target_ece is not None:
        out["target_ece"] = target_ece
        out["meets_target"] = bool(e <= target_ece)
    return out


# ---------------------------------------------------------------------------------------------- data


def _bool_cell(v: str | None) -> bool | None:
    if v is None or str(v).strip() == "":
        return None
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "made", "make"):
        return True
    if s in ("0", "false", "f", "no", "n", "missed", "miss"):
        return False
    raise ValueError(f"not a boolean: {v!r}")


def _float_cell(v: str | None) -> float | None:
    if v is None or str(v).strip() == "":
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def read_shots_csv(path: str | Path) -> list[ShotRow]:
    """Rows of a shots CSV; raises ValueError naming the first bad row or missing column."""
    rows: list[ShotRow] = []
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in _REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing required column(s) {missing}; expected {list(SHOT_COLUMNS)}")
        for i, r in enumerate(reader, start=2):
            try:
                made = _bool_cell(r.get("made"))
                if made is None:
                    raise ValueError("empty 'made'")
                f = ShotFeatures(
                    distance_ft=float(r["distance_ft"]), value=int(float(r["value"])),
                    shot_type=(r.get("shot_type") or "jumper").strip() or "jumper",
                    closest_defender_ft=_float_cell(r.get("closest_defender_ft")),
                    catch_and_shoot=_bool_cell(r.get("catch_and_shoot")), zone=(r.get("zone") or "").strip() or None)
                if f.value not in (1, 2, 3) or not math.isfinite(f.distance_ft):
                    raise ValueError(f"value must be 1/2/3 and distance finite, got {f.value!r}, {f.distance_ft!r}")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{i}: {exc}") from exc
            rows.append(ShotRow(f, made, (r.get("game_id") or "").strip() or None))
    return rows


def split_by_game(game_ids: Sequence[str | None], holdout_frac: float, seed: int = 0) -> tuple[list[int], list[int]]:
    """(train indices, held-out indices); whole games go to one side, chosen by a seeded hash.

    Fewer than two distinct games (or any row without a game id) -> everything is training data.
    """
    if holdout_frac <= 0 or any(g is None for g in game_ids):
        return list(range(len(game_ids))), []
    games = sorted(set(game_ids))  # type: ignore[arg-type]
    if len(games) < 2:
        return list(range(len(game_ids))), []
    def h(g: str) -> str:
        return hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()
    n_hold = min(len(games) - 1, max(1, round(holdout_frac * len(games))))
    held = set(sorted(games, key=h)[:n_hold])
    train = [i for i, g in enumerate(game_ids) if g not in held]
    test = [i for i, g in enumerate(game_ids) if g in held]
    return train, test


# ---------------------------------------------------------------------------------------------- fit


def fit_shot_model(rows: Sequence[ShotRow], config: FitShotConfig | None = None) -> tuple[ShotQualityModel, dict[str, Any]]:
    """Fit ``ShotQualityModel`` on the training games; report prior and fit on the held-out games."""
    cfg = config or FitShotConfig()
    fg = [r for r in rows if not r.features.is_free_throw]
    if not fg:
        raise ValueError("no field-goal attempts to fit (free throws use the league FT%)")
    train_idx, test_idx = split_by_game([r.game_id for r in fg], cfg.holdout_frac, cfg.seed)
    in_sample = not test_idx
    if in_sample:
        log.warning("no held-out games (need >= 2 games with game_id); the report is IN-SAMPLE")
        test_idx = train_idx
    model = ShotQualityModel(cfg.league)
    prior = ShotQualityModel(cfg.league)
    model.fit([fg[i].features for i in train_idx], [fg[i].made for i in train_idx], l2=cfg.l2)
    test = [fg[i] for i in test_idx]
    y = [r.made for r in test]
    groups = [r.game_id for r in test]

    def rep(m: ShotQualityModel) -> dict[str, Any]:
        return calibration_report([m.xfg(r.features) for r in test], y, cfg.bins, cfg.n_boot, cfg.seed, groups, cfg.target_ece)

    report = {
        "league": cfg.league, "prior_status": model.prior_status, "split": "in_sample" if in_sample else "by_game",
        "n_rows": len(rows), "n_free_throws_skipped": len(rows) - len(fg), "n_train": len(train_idx),
        "n_holdout": 0 if in_sample else len(test_idx),
        "games_train": sorted({fg[i].game_id for i in train_idx if fg[i].game_id}),
        "games_holdout": [] if in_sample else sorted({r.game_id for r in test if r.game_id}),
        "fit_info": model.fit_info, "coef": dict(zip(model.to_json()["feature_names"], [round(c, 5) for c in model.coef.tolist()], strict=True)),
        "prior": rep(prior), "fitted": rep(model), "config": asdict(cfg),
    }
    f = report["fitted"]
    log.info("shot model: n_train=%d holdout=%d ECE %.4f CI %s Brier %.4f (prior ECE %.4f)", report["n_train"],
             report["n_holdout"], f["ece"], f["ece_ci95"], f["brier"], report["prior"]["ece"])
    return model, report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code (0 ok, 2 bad input)."""
    ap = argparse.ArgumentParser(prog="python -m courtvision.training.fit_shot_model", description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", required=True, help="labelled shots CSV (columns: %s)" % ", ".join(SHOT_COLUMNS))
    ap.add_argument("--out", required=True, help="output model JSON (ShotQualityModel.to_json)")
    ap.add_argument("--report", help="calibration report JSON (default: <out stem>.report.json)")
    ap.add_argument("--league", default="nba")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--holdout-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    cfg = FitShotConfig(league=args.league, l2=args.l2, holdout_frac=args.holdout_frac, seed=args.seed, n_boot=args.n_boot)
    try:
        model, report = fit_shot_model(read_shots_csv(args.csv), cfg)
    except (OSError, ValueError) as exc:
        log.error("fit_shot_model: %s", exc)
        return 2
    out = Path(args.out)
    out.write_text(json.dumps(model.to_json(), indent=2, sort_keys=True))
    rep_path = Path(args.report) if args.report else out.with_suffix(".report.json")
    rep_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    log.info("wrote %s and %s", out, rep_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
