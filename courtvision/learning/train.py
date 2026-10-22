"""``courtvision train``: ingested games -> a candidate model bundle.

What gets better with more footage:

1. **Event engine settings** (``tune_engine``). The engine re-runs on each game's cached tracks
   (``tracks.jsonl.gz``: no video, no perception) with candidate settings and is scored against the game's
   PBP labels: F1 per PBP-labelled kind (``dataset.PBP_KINDS``), counts pooled over games, macro over the kinds
   with labels, events matched within ``tol_s`` (PBP times carry the alignment error) without actors (PBP
   person ids are not roster ids). A coordinate search over ``SEARCH_SPACE`` keeps a value only when it raises
   the pooled score by ``min_gain``; ``rounds`` passes over the space.
2. **Cue models** (``fit_cue_models``): status calibration of the engine's events from the games' cue rows.
3. **Shot model** (``fit_shot_model``): xFG from the games' shot rows.
4. **Scorebug glyphs**: a glyph bank given by the caller (``glyph_bank``) is copied into the bundle.

Only accepted (alignment-gated), non-gold games train. The bundle records its training games and its rights:
"shippable" only when every game is ``own`` / ``licensed`` with ``training_allowed``.
"""

from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import json
import logging
import multiprocessing as mp
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from courtvision.eval.events import match_events
from courtvision.events.calibration import CueModelSet
from courtvision.events.engine import EventEngine, EventEngineConfig
from courtvision.geometry.court import court_for
from courtvision.geometry.rules import rules_for
from courtvision.types import Event

from .bundle import ModelBundle
from .dataset import PBP_KINDS, GameDataset

__all__ = ["SEARCH_SPACE", "TrainConfig", "evaluate_engine", "score_report", "train", "tune_engine"]

log = logging.getLogger(__name__)

# Engine settings the loop may tune, with the values it tries (each a monocular / broadcast sensitivity
# documented in events/engine.py). Values are absolute; the current value is always a candidate.
SEARCH_SPACE: dict[str, list[Any]] = {
    "steal_backdate_max_s": [0.5, 0.7, 1.0, 1.3],
    "steal_vote_frac": [0.5, 0.6, 0.7],
    "steal_sustain_s": [0.6, 0.9, 1.2],
    "steal_reach_ft": [5.0, 6.0, 8.0],
    "rebound_backdate_max_s": [0.7, 1.0, 1.3],
    "holder_hidden_confirm_s": [0.3, 0.4, 0.6],
    "holder_vote_loose_k": [4, 5, 6],
    "retro_block_turn_rise_frac": [0.5, 1.0],
    "block_reverse_dv_img_fts": [10.0, 12.0, 15.0, 18.0],
    "retro_block_min_loose_s": [0.3, 0.4, 0.6],
    "hold_block_confirm_s": [0.0, 0.5],  # the monocular hold-time block path: off unless labelled games show a gain
}


@dataclass
class TrainConfig:
    search: dict[str, list[Any]] = field(default_factory=lambda: dict(SEARCH_SPACE))
    rounds: int = 1
    tol_s: float = 2.0
    min_gain: float = 0.002
    kinds: tuple[str, ...] = PBP_KINDS
    workers: int | None = None
    fit_cues: bool = True
    fit_shots: bool = True
    allow_sim_cues: bool = True  # simulator games give sim-calibrated cue models (development only)


# --------------------------------------------------------------------------- scoring


def evaluate_engine(ds: GameDataset, overrides: Mapping[str, Any], base: EventEngineConfig | None = None,
                    cue_models: CueModelSet | None = None) -> list[Event]:
    """The engine's events on a game's cached tracks with ``overrides`` applied to ``base``."""
    cfg = dataclasses.replace(base or EventEngineConfig(), **dict(overrides))
    league = ds.manifest.league
    engine = EventEngine(court_for(league), ds.roster(), cfg, rules=rules_for(league), cue_models=cue_models)
    for f in ds.frames():
        engine.process(f)
    engine.flush()
    return engine.events


def score_report(pairs: Sequence[tuple[Sequence[Event], Sequence[Event]]], kinds: Sequence[str], tol_s: float) -> dict[str, Any]:
    """Pooled F1 over games: per kind (TP / predicted / labelled summed over games) and their macro over the
    kinds with labels. ``pairs`` is (engine events, label events) per game."""
    tp: dict[str, int] = dict.fromkeys(kinds, 0)
    npred, ntrue = dict(tp), dict(tp)
    for pred, labels in pairs:
        rep = match_events(pred, labels, kinds=list(kinds), tol_s=tol_s, match_player=False)
        for k in kinds:
            r = rep.per_kind.get(k)
            if r is not None:
                tp[k] += r.tp
                npred[k] += r.n_pred
                ntrue[k] += r.n_truth
    per = {}
    for k in kinds:
        den = npred[k] + ntrue[k]
        per[k] = {"f1": (2.0 * tp[k] / den) if den else None, "tp": tp[k], "pred": npred[k], "labels": ntrue[k]}
    supported = [per[k]["f1"] or 0.0 for k in kinds if ntrue[k] > 0]
    return {"macro_f1": sum(supported) / len(supported) if supported else None, "per_kind": per, "games": len(pairs)}


def _eval_task(args: tuple[str, dict[str, Any], dict[str, Any], str | None]) -> list[dict[str, Any]]:
    path, overrides, base, cue_path = args
    ds = GameDataset(path)
    cues = CueModelSet.load(cue_path) if cue_path else None
    events = evaluate_engine(ds, overrides, EventEngineConfig(**base) if base else None, cues)
    return [e.to_dict() for e in events]


class _Scorer:
    """Scores override sets over games, in a process pool (each game is independent)."""

    def __init__(self, datasets: Sequence[GameDataset], cfg: TrainConfig, base: EventEngineConfig | None) -> None:
        self.datasets, self.cfg = list(datasets), cfg
        self.base = dataclasses.asdict(base) if base is not None else {}
        self.labels = [ds.labels() for ds in self.datasets]
        n = cfg.workers if cfg.workers is not None else min(len(self.datasets), mp.cpu_count() or 1)
        self.pool = cf.ProcessPoolExecutor(n, mp_context=mp.get_context("spawn")) if n > 1 else None

    def __call__(self, overrides: Mapping[str, Any]) -> dict[str, Any]:
        tasks = [(str(ds.path), dict(overrides), self.base, None) for ds in self.datasets]
        runs = list(self.pool.map(_eval_task, tasks)) if self.pool is not None else [_eval_task(t) for t in tasks]
        pairs = [([Event.from_dict(d) for d in run], labels) for run, labels in zip(runs, self.labels, strict=True)]
        return score_report(pairs, self.cfg.kinds, self.cfg.tol_s)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()


def tune_engine(datasets: Sequence[GameDataset], config: TrainConfig | None = None, start: Mapping[str, Any] | None = None,
                base: EventEngineConfig | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Coordinate search (module docstring). Returns (overrides, history with the scores)."""
    cfg = config or TrainConfig()
    scorer = _Scorer(datasets, cfg, base)
    try:
        current = dict(start or {})
        best = scorer(current)
        history = {"start": {"overrides": dict(current), "score": best}, "steps": []}
        log.info("tune: start macro F1 %.4f over %d games", best["macro_f1"] or 0.0, len(datasets))
        defaults = base or EventEngineConfig()
        for rnd in range(max(1, cfg.rounds)):
            changed = False
            for name, values in cfg.search.items():
                if not hasattr(defaults, name):
                    raise ValueError(f"unknown engine setting {name!r} in the search space")
                cur_val = current.get(name, getattr(defaults, name))
                for v in values:
                    if v == cur_val:
                        continue
                    trial = {**current, name: v}
                    sc = scorer(trial)
                    step = {"round": rnd, "param": name, "value": v, "macro_f1": sc["macro_f1"]}
                    history["steps"].append(step)
                    if (sc["macro_f1"] or 0.0) > (best["macro_f1"] or 0.0) + cfg.min_gain:
                        log.info("tune: %s %s -> %s: macro F1 %.4f -> %.4f", name, cur_val, v, best["macro_f1"] or 0.0,
                                 sc["macro_f1"])
                        current, best, cur_val, changed = trial, sc, v, True
            if not changed:
                break
        history["final"] = {"overrides": dict(current), "score": best}
        return current, history
    finally:
        scorer.close()


# --------------------------------------------------------------------------- train


def _rights(datasets: Sequence[GameDataset]) -> str:
    return "shippable" if datasets and all(ds.manifest.shippable for ds in datasets) else "research_only"


def train(datasets: Sequence[GameDataset], out_dir: str | Path, config: TrainConfig | None = None,
          base_bundle: ModelBundle | None = None, glyph_bank: str | Path | None = None) -> ModelBundle:
    """Train a candidate bundle from accepted, non-gold games (module docstring)."""
    cfg = config or TrainConfig()
    gold = [ds.game_id for ds in datasets if ds.manifest.gold]
    if gold:
        raise ValueError(f"gold (frozen evaluation) games cannot train: {gold}")
    games = [ds for ds in datasets if ds.manifest.accepted]
    skipped = sorted(ds.game_id for ds in datasets if not ds.manifest.accepted)
    if skipped:
        log.warning("train: %d game(s) failed the alignment gate and are skipped: %s", len(skipped), skipped)
    if not games:
        raise ValueError("no accepted games to train on")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = dict(base_bundle.engine_overrides) if base_bundle is not None else {}
    overrides, history = tune_engine(games, cfg, start)
    metrics: dict[str, Any] = {"engine": history["final"]["score"], "engine_start": history["start"]["score"]}
    (out / "tuning.json").write_text(json.dumps(history, indent=1, default=str))
    if cfg.fit_cues:
        metrics["cues"] = _fit_cues(games, out, cfg, base_bundle)
    if cfg.fit_shots:
        metrics["shots"] = _fit_shots(games, out, base_bundle)
    if glyph_bank is not None:
        shutil.copyfile(glyph_bank, out / "glyph_bank.json")
    elif base_bundle is not None and base_bundle.file("glyph_bank.json") is not None:
        shutil.copyfile(base_bundle.file("glyph_bank.json"), out / "glyph_bank.json")  # type: ignore[arg-type]
    prov = {"games": [{"game_id": ds.game_id, "rights_status": ds.manifest.rights_status,
                       "training_allowed": ds.manifest.training_allowed, "source": ds.manifest.source,
                       "files": ds.manifest.files.get("labels.jsonl")} for ds in games],
            "skipped": skipped, "parent": base_bundle.id if base_bundle is not None else None,
            "config": {"rounds": cfg.rounds, "tol_s": cfg.tol_s, "kinds": list(cfg.kinds),
                       "search": {k: list(v) for k, v in cfg.search.items()}}}
    bundle = ModelBundle(out, overrides, _rights(games), metrics, prov).save()
    log.info("train: bundle %s (%s) -> %s; engine macro F1 %.4f (start %.4f)", bundle.id, bundle.rights, out,
             metrics["engine"]["macro_f1"] or 0.0, metrics["engine_start"]["macro_f1"] or 0.0)
    return bundle


def _fit_cues(games: Sequence[GameDataset], out: Path, cfg: TrainConfig, base: ModelBundle | None) -> dict[str, Any]:
    from courtvision.training.fit_cue_models import fit_cue_models, load_base_sources, save_cue_models

    rows = [r for ds in games for r in ds.jsonl("cue_rows.jsonl")]
    if not cfg.allow_sim_cues:
        rows = [r for r in rows if r.get("calibration_source") != "sim"]
    if not rows:
        return {"rows": 0}
    base_path = base.file("cue_models.json") if base is not None else None
    base_models = CueModelSet.load(base_path) if base_path is not None else None
    base_sources = load_base_sources(base_path) if base_path is not None else None
    models, reports = fit_cue_models(rows, base=base_models, base_sources=base_sources)
    doc = save_cue_models(models, reports, out / "cue_models.json", base_sources)
    return {"rows": len(rows), "for_live": doc.get("for_live"),
            "status": {k: m["status"] for k, m in doc["models"].items()}}


def _fit_shots(games: Sequence[GameDataset], out: Path, base: ModelBundle | None) -> dict[str, Any]:
    from courtvision.training.fit_shot_model import FitShotConfig, fit_shot_model, read_shots_csv

    rows = []
    for ds in games:
        p = ds.path / "shots.csv"
        if p.is_file() and len(p.read_text().splitlines()) > 1:
            rows.extend(read_shots_csv(p))
    fg = [r for r in rows if not r.features.is_free_throw]
    if len(fg) < 20:
        return {"rows": len(rows), "skipped": "fewer than 20 field-goal attempts"}
    model, report = fit_shot_model(rows, FitShotConfig(league=games[0].manifest.league))
    (out / "shot_model.json").write_text(json.dumps(model.to_json()))
    return {"rows": len(rows), "ece": report["fitted"]["ece"], "prior_ece": report["prior"]["ece"],
            "n_train": report["n_train"], "n_holdout": report["n_holdout"]}

