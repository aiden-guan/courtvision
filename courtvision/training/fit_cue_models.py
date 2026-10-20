"""Fit the event cue models from labelled JSONL and certify their live gates (RESEARCH §7A.2, §4.1, §4.4).

    python -m courtvision.training.fit_cue_models --jsonl make_rows.jsonl --out cue_models.json

Row schema, one JSON object per line (``CUE_ROW_KEYS``)::

    {"kind": "make", "features": {"crossed_inside": 1, "drop_in_band": 1, ...}, "label": 1,
     "game_id": "g001", "calibration_source": "gold", "event_id": "e000123", "t": 812.4}

``features`` (alias ``cues``) are the named cues the engine stores in ``Event.attrs["cues"]``; ``label``
is 1 when the event is real / the outcome is positive (made, three, ...).

* **Status** follows ``events.calibration.CueModel.fit``: ``uncalibrated`` until >= 300 labelled events with
  >= 50 in the ambiguous band (0.05, 0.95), then ``platt``; ``isotonic`` from >= 1,000 events. Only the
  training games count.
* **Calibration** is reported on held-out games: 15-bin ECE with a bootstrap 95% CI over games, Brier,
  log-loss and the reliability table (``fit_shot_model.calibration_report``). Target ECE <= 0.03.
* **Gate certification** (rule of three [481]): the gate at threshold tau with error budget eps is
  certified only if the held-out events with p >= tau show 0 errors in n >= 3/eps cases (the 95% upper
  bound on the error rate is then 3/n), and the model is not ``uncalibrated``. Example: made FG at
  precision 0.98 (eps = 0.02) needs >= 150 accepted makes without an error.
* **Sim-derived rows** (``calibration_source == "sim"``, e.g. from ``build_cue_dataset``) are for
  development only. The model is marked ``calibration_source: "sim"``, its certificate is never valid for
  live use, and the CLI writes it only with ``--allow-sim``. The mark travels with the weights: a model
  carried over from a ``--base`` file (not refit for lack of rows, or a kind without rows) keeps the
  base's ``calibration_source``, so a sim calibration cannot become ``for_live`` by a later run.
* **Negative-outcome gates** (``CueFitConfig.lower_gates``): "make" also certifies P(miss) >= 0.90, the
  narration gate for SHOT_MISSED (confidence = 1 - p_make).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta

from courtvision.events.calibration import GATED_KINDS, CueModel, CueModelSet, default_cue_models
from courtvision.training.fit_shot_model import calibration_report, split_by_game
from courtvision.types import Event, EventKind, EventStatus

__all__ = [
    "CUE_ROW_KEYS",
    "CueFitConfig",
    "CueFitReport",
    "GateCertificate",
    "build_cue_dataset",
    "certify_gate",
    "error_upper_bound",
    "fit_cue_kind",
    "fit_cue_models",
    "load_base_sources",
    "load_cue_rows",
    "main",
    "model_sources",
    "rule_of_three_n",
    "save_cue_models",
    "sim_calibrated",
    "write_cue_rows",
]

log = logging.getLogger(__name__)

CUE_ROW_KEYS: tuple[str, ...] = ("kind", "features", "label", "game_id", "calibration_source", "event_id", "t")
SIM_SOURCE = "sim"
COLD_START_SOURCE = "cold_start"  # hand-set weights (events.calibration.default_cue_models)


@dataclass
class CueFitConfig:
    """Hold-out, report and gate settings."""

    holdout_frac: float = 0.3  # share of games held out for ECE and certification; tuned
    seed: int = 0
    bins: int = 15  # 15-bin ECE (§7A.2); sourced
    n_boot: int = 1000  # bootstrap resamples for the ECE CI; tuned
    target_ece: float = 0.03  # §4.4 / §7A.2 target; sourced (proposal)
    # Live gates of RESEARCH §4.1 (block 0.92 as in narration FactsConfig.gates); sourced (proposal).
    gates: dict[str, float] = field(default_factory=lambda: {
        "shot": 0.90, "make": 0.95, "three": 0.97, "rebound": 0.90, "steal": 0.90, "block": 0.92, "player_id": 0.95})
    # Error budget eps per gate. make: made-FG precision >= 0.98 (§4.4 coach target) -> 0.02, sourced.
    # Others: 1 - gate (a calibrated p >= tau errs at most 1 - tau); proposal.
    error_budget: dict[str, float] = field(default_factory=lambda: {
        "shot": 0.10, "make": 0.02, "three": 0.03, "rebound": 0.10, "steal": 0.10, "block": 0.08, "player_id": 0.05})
    # Gates on the negative outcome, P(negative) = 1 - p >= tau: narration says "missed" when a SHOT_MISSED's
    # confidence (= 1 - p_make) passes FactsConfig.gates["shot_missed"] = 0.90, so that tail needs its own
    # certificate. eps = 1 - tau as for the other kinds; proposal.
    lower_gates: dict[str, float] = field(default_factory=lambda: {"make": 0.90})
    lower_error_budget: dict[str, float] = field(default_factory=lambda: {"make": 0.10})


# ---------------------------------------------------------------------------------------------- certification


def rule_of_three_n(eps: float) -> int:
    """Smallest n with 3/n <= eps: accepted events needed with 0 errors (95% bound, [481])."""
    if not 0.0 < eps < 1.0:
        raise ValueError(f"error budget must be in (0, 1), got {eps!r}")
    return int(math.ceil(3.0 / eps - 1e-9))  # 3 / 0.02 = 150.00000000000003 must give 150


def error_upper_bound(errors: int, n: int, conf: float = 0.95) -> float:
    """One-sided upper bound on the error rate: rule of three (3/n) at 0 errors, else Clopper-Pearson."""
    if n <= 0:
        return 1.0
    if errors <= 0:
        return min(1.0, 3.0 / n)
    if errors >= n:
        return 1.0
    return float(beta.ppf(conf, errors + 1, n - errors))


@dataclass
class GateCertificate:
    kind: str
    threshold: float
    error_budget: float
    n_accepted: int  # held-out events with p >= threshold
    n_errors: int  # ... whose label is negative
    n_required: int  # rule_of_three_n(error_budget)
    error_upper_95: float
    certified: bool
    valid_for_live: bool  # certified and not sim-derived
    reason: str


def certify_gate(p: Sequence[float], y: Sequence[bool | int], threshold: float, error_budget: float,
                 status: str, kind: str = "", calibration_source: str = "unknown") -> GateCertificate:
    """Rule-of-three certificate for one gate on held-out events (RESEARCH §7A.2)."""
    pa = np.asarray(p, dtype=float)
    ya = np.asarray([bool(v) for v in y], dtype=bool)
    if len(pa) != len(ya):
        raise ValueError("p and y differ in length")
    if not np.all(np.isfinite(pa)):
        raise ValueError("non-finite probabilities cannot certify a gate")
    accepted = pa >= threshold
    n_acc = int(accepted.sum())
    n_err = int((accepted & ~ya).sum())
    need = rule_of_three_n(error_budget)
    if status == "uncalibrated":
        ok, reason = False, "model is uncalibrated: the gate is never met (hedge-only, RESEARCH §4.1)"
    elif n_err > 0:
        ok, reason = False, f"{n_err} error(s) among {n_acc} accepted events"
    elif n_acc < need:
        ok, reason = False, f"only {n_acc} accepted events without error; {need} needed for eps={error_budget}"
    else:
        ok, reason = True, f"0 errors in {n_acc} >= {need} accepted events"
    live = ok and calibration_source != SIM_SOURCE
    if ok and not live:
        reason += "; sim-derived calibration is not valid for live use"
    return GateCertificate(kind, float(threshold), float(error_budget), n_acc, n_err, need,
                           round(error_upper_bound(n_err, n_acc), 5), ok, live, reason)


# ---------------------------------------------------------------------------------------------- rows


def _label(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "made"):
        return True
    if s in ("0", "false", "no", "missed"):
        return False
    raise ValueError(f"label must be 0/1 or boolean, got {v!r}")


def _validate_row(row: Mapping[str, Any], where: str) -> dict[str, Any]:
    feats = row.get("features", row.get("cues"))
    if not isinstance(feats, Mapping):
        raise ValueError(f"{where}: 'features' must be an object of named cues")
    if not row.get("kind"):
        raise ValueError(f"{where}: missing 'kind'")
    out = dict(row)
    out["features"] = dict(feats)
    out["label"] = _label(row.get("label"))
    out["calibration_source"] = str(row.get("calibration_source") or "unknown")
    out["game_id"] = None if row.get("game_id") in (None, "") else str(row["game_id"])
    return out


def load_cue_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for i, line in enumerate(Path(path).read_text().splitlines(), start=1):
            if line.strip():
                rows.append(_validate_row(json.loads(line), f"{path}:{i}"))
    return rows


def write_cue_rows(rows: Iterable[Mapping[str, Any]], path: str | Path) -> int:
    n = 0
    with Path(path).open("w") as fh:
        for r in rows:
            fh.write(json.dumps(dict(r), sort_keys=True, default=str) + "\n")
            n += 1
    return n


def _combine_sources(sources: Iterable[str]) -> str:
    found = set(sources)
    if SIM_SOURCE in found:
        return SIM_SOURCE  # any sim row contaminates the calibration
    if not found:
        return "unknown"
    return found.pop() if len(found) == 1 else "mixed"


def _source_of(rows: Sequence[Mapping[str, Any]]) -> str:
    return _combine_sources(str(r.get("calibration_source") or "unknown") for r in rows)


def _default_source(model: CueModel) -> str:
    """Source of a model nobody recorded: cold-start weights, or a calibration of unknown origin."""
    return COLD_START_SOURCE if model.status == "uncalibrated" else "unknown"


def load_base_sources(path: str | Path) -> dict[str, str]:
    """Per-kind ``calibration_source`` recorded in a cue-model file written by ``save_cue_models``.

    ``CueModelSet.load`` ignores this key, so a ``--base`` file would otherwise lose the fact that its
    weights were calibrated on simulator rows.
    """
    doc = json.loads(Path(path).read_text())
    return {str(k): str(m["calibration_source"]) for k, m in (doc.get("models") or {}).items()
            if isinstance(m, Mapping) and m.get("calibration_source")}


# ---------------------------------------------------------------------------------------------- fitting


@dataclass
class CueFitReport:
    kind: str
    status_before: str
    status: str
    n_rows: int
    n_train: int
    n_holdout: int
    n_ambiguous_train: int
    split: str  # "by_game" | "in_sample"
    games_holdout: list[str]
    calibration_source: str  # where the model's calibration came from (the base's when it was not refit)
    calibration: dict[str, Any]
    gate: GateCertificate
    refit: bool = True  # False: too few rows, the base model was kept and only evaluated
    gate_low: GateCertificate | None = None  # negative-outcome gate (P(miss) for "make"), when configured

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _copy(model: CueModel) -> CueModel:
    return CueModel.from_dict(model.to_dict(), model.config)


def fit_cue_kind(model: CueModel, rows: Sequence[Mapping[str, Any]], config: CueFitConfig | None = None,
                 base_source: str | None = None) -> tuple[CueModel, CueFitReport]:
    """Refit one cue model on the training games; report calibration and certify its gate on the rest.

    ``base_source`` is the ``calibration_source`` of ``model`` (``load_base_sources``). When the rows are
    too few to refit, the model keeps the base calibration, so the report keeps the base's source too
    (a sim-calibrated base never becomes "gold" by being evaluated on gold rows).
    """
    cfg = config or CueFitConfig()
    fitted = _copy(model)
    train_idx, test_idx = split_by_game([r.get("game_id") for r in rows], cfg.holdout_frac, cfg.seed)
    in_sample = not test_idx
    if in_sample:
        log.warning("cue model %s: no held-out games; calibration and gate are IN-SAMPLE (never certified)", model.kind)
        test_idx = train_idx
    train = [rows[i] for i in train_idx]
    before = fitted.status
    before_doc = fitted.to_dict()
    lo, hi = fitted.config.ambiguous_band
    p_now = fitted.predict_many([r["features"] for r in train])
    n_amb = int(np.sum((p_now > lo) & (p_now < hi)))
    status = fitted.fit([r["features"] for r in train], [r["label"] for r in train])
    refit = fitted.to_dict() != before_doc
    test = [rows[i] for i in test_idx]
    p = fitted.predict_many([r["features"] for r in test])
    y = [r["label"] for r in test]
    if refit:
        source = _source_of(rows)
    else:  # calibration inherited from the base; sim evaluation rows still taint the certificate
        inherited = [base_source or _default_source(model)]
        source = _combine_sources(inherited + ([SIM_SOURCE] if _source_of(rows) == SIM_SOURCE else []))
    report = calibration_report(p, y, cfg.bins, cfg.n_boot, cfg.seed, [r.get("game_id") for r in test], cfg.target_ece)
    gate_status = "uncalibrated" if in_sample else status
    gate = certify_gate(p, y, cfg.gates.get(model.kind, 0.95), cfg.error_budget.get(model.kind, 0.05),
                        gate_status, model.kind, source)
    gate_low = None
    if model.kind in cfg.lower_gates:
        gate_low = certify_gate(1.0 - np.asarray(p, dtype=float), [not v for v in y], cfg.lower_gates[model.kind],
                                cfg.lower_error_budget.get(model.kind, 1.0 - cfg.lower_gates[model.kind]),
                                gate_status, f"{model.kind}:negative", source)
    if in_sample:
        gate.reason = "no held-out games: " + gate.reason
        if gate_low is not None:
            gate_low.reason = "no held-out games: " + gate_low.reason
    rep = CueFitReport(model.kind, before, status, len(rows), len(train), 0 if in_sample else len(test), n_amb,
                       "in_sample" if in_sample else "by_game",
                       [] if in_sample else sorted({str(r["game_id"]) for r in test if r.get("game_id")}),
                       source, report, gate, refit, gate_low)
    log.info("cue model %s: %s -> %s (train %d, holdout %d), ECE %s CI %s, gate %s", model.kind, before, status,
             rep.n_train, rep.n_holdout, report.get("ece"), report.get("ece_ci95"), "certified" if gate.certified else "not certified")
    return fitted, rep


def fit_cue_models(rows: Sequence[Mapping[str, Any]], base: CueModelSet | None = None,
                   config: CueFitConfig | None = None, kinds: Iterable[str] | None = None,
                   base_sources: Mapping[str, str] | None = None) -> tuple[CueModelSet, dict[str, CueFitReport]]:
    """Fit every gated kind that has rows; other kinds keep the base (default: cold-start) model.

    ``base_sources`` (``load_base_sources``) carries each base model's ``calibration_source``.
    """
    models = {k: _copy(m) for k, m in (base.models if base is not None else default_cue_models()).items()}
    sources = dict(base_sources or {})
    wanted = set(kinds) if kinds is not None else set(GATED_KINDS)
    by_kind: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by_kind.setdefault(str(r["kind"]), []).append(r)
    reports: dict[str, CueFitReport] = {}
    for kind, krows in sorted(by_kind.items()):
        if kind not in wanted:
            continue
        if kind not in models:
            log.warning("no cue model for kind %r; %d rows skipped", kind, len(krows))
            continue
        models[kind], reports[kind] = fit_cue_kind(models[kind], krows, config, sources.get(kind))
    return CueModelSet(models), reports


def model_sources(models: CueModelSet, reports: Mapping[str, CueFitReport],
                  base_sources: Mapping[str, str] | None = None) -> dict[str, str]:
    """``calibration_source`` of every model in the set: this run's report, else the base file's record."""
    base = dict(base_sources or {})
    return {k: reports[k].calibration_source if k in reports else base.get(k, _default_source(m))
            for k, m in models.models.items()}


def sim_calibrated(models: CueModelSet, sources: Mapping[str, str]) -> list[str]:
    """Kinds whose calibrated (platt / isotonic) weights came from simulator rows."""
    return sorted(k for k, m in models.models.items() if m.status != "uncalibrated" and sources.get(k) == SIM_SOURCE)


def save_cue_models(models: CueModelSet, reports: Mapping[str, CueFitReport], path: str | Path,
                    base_sources: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Write a ``CueModelSet`` JSON (loadable by ``CueModelSet.load`` / ``EventEngineConfig.cue_model_path``)
    with every model's ``calibration_source`` and the reports under ``"calibration"``.

    ``for_live`` is False when any calibrated model, fitted now or carried over from the base, was
    calibrated on simulator rows.
    """
    doc = models.to_dict()
    sources = model_sources(models, reports, base_sources)
    for kind, src in sources.items():
        doc["models"][kind]["calibration_source"] = src
    doc["calibration"] = {k: r.to_dict() for k, r in sorted(reports.items())}
    doc["for_live"] = not sim_calibrated(models, sources)
    Path(path).write_text(json.dumps(doc, indent=2, sort_keys=True, default=str))
    return doc


# ---------------------------------------------------------------------------------------------- sim datasets


def _latest(events: Iterable[Event]) -> list[Event]:
    latest: dict[str, Event] = {}
    for e in events:
        cur = latest.get(e.event_id)
        if cur is None or e.revision >= cur.revision:
            latest[e.event_id] = e
    return sorted((e for e in latest.values() if e.status is not EventStatus.RETRACTED), key=lambda e: e.t)


def _same_team(a: Event, b: Event) -> bool:
    return a.team is None or b.team is None or a.team is b.team


def _match(pred: Sequence[Event], truth: Sequence[Event], tol_s: float,
           compat: Callable[[Event, Event], bool] = _same_team) -> dict[int, int]:
    """Greedy one-to-one matching by |dt| <= tol_s; pred index -> truth index."""
    pairs = sorted((abs(p.t - t.t), i, j) for i, p in enumerate(pred) for j, t in enumerate(truth)
                   if abs(p.t - t.t) <= tol_s and compat(p, t))
    out: dict[int, int] = {}
    used: set[int] = set()
    for _, i, j in pairs:
        if i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out


def _row(kind: str, e: Event, cues: Mapping[str, Any], label: bool, game_id: str, truth: Event | None) -> dict[str, Any]:
    return {"kind": kind, "features": dict(cues), "label": bool(label), "game_id": game_id, "event_id": e.event_id,
            "t": round(e.t, 3), "truth_event_id": truth.event_id if truth else None, "calibration_source": SIM_SOURCE}


def build_cue_dataset(engine_events: Iterable[Event], truth_events: Iterable[Event], game_id: str,
                      tol_s: float = 1.0, kinds: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Label engine events' cues against simulator truth (DEVELOPMENT ONLY; rows are calibration_source "sim").

    * shot: an engine attempt is real when it matches a truth attempt within ``tol_s`` (same team);
    * make / three: outcome / value of the matched truth shot (through the attempts' shot ids);
    * rebound / steal / block: real when matched to a truth event of the same kind and team.
    Engine results whose shot has no truth match are skipped (no outcome to label).
    """
    eng, tru = _latest(engine_events), _latest(truth_events)
    want = set(kinds) if kinds is not None else set(GATED_KINDS)
    rows: list[dict[str, Any]] = []
    att_e = [e for e in eng if e.kind is EventKind.SHOT_ATTEMPT]
    att_t = [e for e in tru if e.kind is EventKind.SHOT_ATTEMPT]
    pairs = _match(att_e, att_t, tol_s)
    truth_result = {e.shot_id: e for e in tru if e.kind in (EventKind.SHOT_MADE, EventKind.SHOT_MISSED) and e.shot_id}
    matched_attempt = {att_e[i].shot_id: att_t[j] for i, j in pairs.items() if att_e[i].shot_id}
    for i, e in enumerate(att_e):
        truth = att_t[pairs[i]] if i in pairs else None
        if "shot" in want and isinstance(e.attrs.get("cues"), Mapping):
            rows.append(_row("shot", e, e.attrs["cues"], truth is not None, game_id, truth))
        if "three" in want and truth is not None and truth.value in (2, 3) and isinstance(e.attrs.get("three_cues"), Mapping):
            rows.append(_row("three", e, e.attrs["three_cues"], truth.value == 3, game_id, truth))
    if "make" in want:
        for e in eng:
            if e.kind not in (EventKind.SHOT_MADE, EventKind.SHOT_MISSED) or not isinstance(e.attrs.get("cues"), Mapping):
                continue
            t_att = matched_attempt.get(e.shot_id)
            res = truth_result.get(t_att.shot_id) if t_att is not None else None
            if res is not None:
                rows.append(_row("make", e, e.attrs["cues"], res.kind is EventKind.SHOT_MADE, game_id, res))
    for kind, ek in (("rebound", EventKind.REBOUND), ("steal", EventKind.STEAL), ("block", EventKind.BLOCK)):
        if kind not in want:
            continue
        pe = [e for e in eng if e.kind is ek and isinstance(e.attrs.get("cues"), Mapping)]
        te = [e for e in tru if e.kind is ek]
        m = _match(pe, te, tol_s)
        rows.extend(_row(kind, e, e.attrs["cues"], i in m, game_id, te[m[i]] if i in m else None) for i, e in enumerate(pe))
    log.info("cue dataset %s: %s (calibration_source=sim; development only)", game_id, dict(Counter(r["kind"] for r in rows)))
    return rows


# ---------------------------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; exit codes: 0 ok, 2 bad input, 3 refused (sim-derived without --allow-sim)."""
    ap = argparse.ArgumentParser(prog="python -m courtvision.training.fit_cue_models", description=__doc__.split("\n\n")[0])
    ap.add_argument("--jsonl", action="append", required=True, help="labelled cue rows (repeatable)")
    ap.add_argument("--out", required=True, help="output CueModelSet JSON")
    ap.add_argument("--report", help="report JSON (default: <out stem>.report.json)")
    ap.add_argument("--base", help="existing CueModelSet JSON to start from (default: cold-start weights)")
    ap.add_argument("--kinds", help="comma-separated kinds to fit (default: all gated kinds present)")
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--allow-sim", action="store_true", help="allow writing models calibrated on simulator rows (development only)")
    args = ap.parse_args(argv)
    try:
        rows = load_cue_rows(args.jsonl)
        base = CueModelSet.load(args.base) if args.base else None
        base_sources = load_base_sources(args.base) if args.base else {}
    except (OSError, ValueError, KeyError) as exc:
        log.error("fit_cue_models: %s", exc)
        return 2
    cfg = CueFitConfig(holdout_frac=args.holdout_frac, seed=args.seed, n_boot=args.n_boot)
    kinds = [k.strip() for k in args.kinds.split(",")] if args.kinds else None
    models, reports = fit_cue_models(rows, base, cfg, kinds, base_sources)
    sim_kinds = sim_calibrated(models, model_sources(models, reports, base_sources))
    if not args.allow_sim and (sim_kinds or any(r.calibration_source == SIM_SOURCE for r in reports.values())):
        log.error("refusing to write sim-calibrated cue models (%s) without --allow-sim (development only)",
                  ", ".join(sim_kinds) or "sim evaluation rows")
        return 3
    out = Path(args.out)
    save_cue_models(models, reports, out, base_sources)
    rep_path = Path(args.report) if args.report else out.with_suffix(".report.json")
    rep_path.write_text(json.dumps({k: r.to_dict() for k, r in sorted(reports.items())}, indent=2, sort_keys=True, default=str))
    log.info("wrote %s and %s", out, rep_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
