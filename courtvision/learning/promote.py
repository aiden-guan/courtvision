"""``courtvision promote``: a candidate bundle replaces the current one only if it is better on frozen gold sets.

Gold games are ingested with ``--gold`` and never train (``train`` refuses them). They must still be frozen:
every file must match its manifest hash (``GameDataset.verify``), else promotion stops.

Both bundles (the current one: the defaults when there is none) re-run the event engine on every gold game's
cached tracks and are scored against the gold labels (``train.score_report``: pooled F1 per PBP-labelled kind,
macro over the labelled kinds). The candidate is promoted when

* its macro F1 beats the current bundle's by at least ``min_gain``, and
* no kind with at least ``min_kind_labels`` gold labels loses more than ``max_kind_drop`` F1, and
* for the ``ship`` channel: the bundle's rights are "shippable" (every training game ``own`` / ``licensed`` with
  ``training_allowed``; TRAINING.md §1.5); research-only data never reaches shipped models.

On promotion the candidate is copied to ``<models>/current`` (the previous current moves to
``<models>/history/<id>``) and every decision, promoted or not, is appended to ``<models>/promotions.jsonl``.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from courtvision.events.calibration import CueModelSet

from .bundle import ModelBundle
from .dataset import PBP_KINDS, GameDataset
from .train import evaluate_engine, score_report

__all__ = ["PromoteConfig", "evaluate_bundle", "promote"]

log = logging.getLogger(__name__)


@dataclass
class PromoteConfig:
    min_gain: float = 0.005
    max_kind_drop: float = 0.05
    min_kind_labels: int = 5
    tol_s: float = 2.0
    kinds: tuple[str, ...] = PBP_KINDS
    channel: str = "research"  # "research" | "ship"


def evaluate_bundle(bundle: ModelBundle | None, gold: Sequence[GameDataset], config: PromoteConfig | None = None) -> dict[str, Any]:
    """Pooled gold-set scores of a bundle (None: the engine defaults)."""
    cfg = config or PromoteConfig()
    overrides = dict(bundle.engine_overrides) if bundle is not None else {}
    cue_path = bundle.file("cue_models.json") if bundle is not None else None
    cues = CueModelSet.load(cue_path) if cue_path is not None else None
    pairs = [(evaluate_engine(ds, overrides, cue_models=cues), ds.labels()) for ds in gold]
    return score_report(pairs, cfg.kinds, cfg.tol_s)


def _check_gold(gold: Sequence[GameDataset]) -> None:
    if not gold:
        raise ValueError("no gold games to evaluate on")
    not_gold = [ds.game_id for ds in gold if not ds.manifest.gold]
    if not_gold:
        raise ValueError(f"not gold (ingest them with --gold): {not_gold}")
    changed = {ds.game_id: bad for ds in gold if (bad := ds.verify())}
    if changed:
        raise ValueError(f"gold sets are not frozen (files changed since ingest): {changed}")


def promote(candidate: ModelBundle, models_dir: str | Path, gold: Sequence[GameDataset],
            config: PromoteConfig | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Decide (and, unless ``dry_run``, apply) a promotion; returns the decision record."""
    cfg = config or PromoteConfig()
    if cfg.channel not in ("research", "ship"):
        raise ValueError("channel must be 'research' or 'ship'")
    _check_gold(gold)
    root = Path(models_dir)
    cur_dir = root / "current"
    current = ModelBundle.load(cur_dir) if (cur_dir / "bundle.json").is_file() else None
    cand_rep = evaluate_bundle(candidate, gold, cfg)
    cur_rep = evaluate_bundle(current, gold, cfg)
    reasons = []
    c, b = cand_rep["macro_f1"] or 0.0, cur_rep["macro_f1"] or 0.0
    if c < b + cfg.min_gain:
        reasons.append(f"macro F1 {c:.4f} does not beat the current {b:.4f} by {cfg.min_gain}")
    for k in cfg.kinds:
        pc, pb = cand_rep["per_kind"][k], cur_rep["per_kind"][k]
        if pb["labels"] >= cfg.min_kind_labels and (pb["f1"] or 0.0) - (pc["f1"] or 0.0) > cfg.max_kind_drop:
            reasons.append(f"{k} F1 drops {pb['f1']:.3f} -> {pc['f1']:.3f}")
    if cfg.channel == "ship" and candidate.rights != "shippable":
        reasons.append(f"rights '{candidate.rights}': only own / licensed games with training_allowed can ship")
    decision = {"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "candidate": candidate.id, "candidate_path": str(candidate.path),
                "current": current.id if current is not None else None, "channel": cfg.channel,
                "gold": [ds.game_id for ds in gold], "candidate_score": cand_rep, "current_score": cur_rep,
                "promoted": not reasons, "reasons": reasons, "dry_run": dry_run}
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)
        if not reasons:
            if current is not None:
                hist = root / "history" / (current.id or "unknown")
                if hist.exists():
                    shutil.rmtree(hist)
                hist.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(cur_dir), str(hist))
            elif cur_dir.exists():
                shutil.rmtree(cur_dir)
            shutil.copytree(candidate.path, cur_dir)
        with (root / "promotions.jsonl").open("a") as fh:
            fh.write(json.dumps(decision, default=str) + "\n")
    log.info("promote %s over %s on %d gold game(s): macro F1 %.4f vs %.4f -> %s%s", candidate.id,
             current.id if current is not None else "defaults", len(gold), c, b,
             "PROMOTED" if not reasons else "kept the current bundle", "" if not reasons else f" ({'; '.join(reasons)})")
    return decision
