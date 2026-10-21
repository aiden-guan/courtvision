"""``courtvision ingest``: one game's video + play-by-play -> an aligned, labelled ``GameDataset``.

1. **Perceive + detect.** The video runs through the normal offline pipeline (``pipeline.run_offline``: perception,
   broadcast gating when configured, the event engine); every ``FrameState`` the engine consumes is cached
   (``tracks.jsonl.gz``), so later training re-runs the engine without the video.
2. **Clock timeline** from the scoreboard reads (OCR, 5 Hz) on those frames (``align_pbp.ClockTimeline.fit``).
3. **Align** the PBP to video time (``align_pbp.align_pbp``: clock mapping, entry lag, plateau ordering, score
   deltas, snapping to engine anchors) and apply the game gate (median |residual| on made FGs < 0.5 s). A
   rejected game is still written (``manifest.accepted = False``); ``train`` skips it.
4. **Labels**: the aligned actions as label events (``dataset.label_events``: clock-mapped, unsnapped times),
   labelled clips, cue rows (the engine's cues labelled by the PBP; ``calibration_source`` "pbp", or "sim"
   for simulator games), shot rows (engine shot location + nearest defender from the tracks, PBP make / miss).

The simulator stands in for real footage end to end (``sim_pbp``: a pseudo-PBP from the simulator's truth
events, entered with a 1.5 s scorer lag and feed-resolution clocks), so the whole loop is exercised without
NBA data. Rights default to ``research_only`` (NBA footage is R&D only, TRAINING.md §1).
"""

from __future__ import annotations

import csv
import dataclasses
import json
import datetime as dt
import io
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from courtvision.geometry.court import CourtSpec, court_for
from courtvision.training.align_pbp import (AlignConfig, ClockTimeline, PBPAction, align_pbp, make_clips,
                                            pbp_from_events, score_changes)
from courtvision.training.fit_cue_models import build_cue_dataset
from courtvision.training.fit_shot_model import SHOT_COLUMNS
from courtvision.types import Event, EventKind, EventStatus, FrameState

from .dataset import GameDataset, GameManifest, label_events, roster_to_dict

__all__ = ["IngestConfig", "ingest_game", "shot_rows", "sim_pbp"]

log = logging.getLogger(__name__)


@dataclass
class IngestConfig:
    rights_status: str = "research_only"
    training_allowed: bool = False
    gold: bool = False  # a frozen evaluation set: train refuses it
    label_tol_s: float = 2.0  # engine event <-> PBP label matching (PBP times carry the alignment error)
    sim_pbp_lag_s: float = 1.5  # simulator pseudo-PBP: the scorer's entry lag (BasketballBench: 1-2 s)
    align: dict[str, Any] = dataclasses.field(default_factory=dict)  # AlignConfig overrides (e.g. accept_min_made)


def sim_pbp(truth_events: Sequence[Event], truth_frames: Sequence[FrameState], lag_s: float = 1.5) -> list[PBPAction]:
    """A pseudo play-by-play of a simulated game (stands in for the real feed in sim studies), its clocks
    quantised like the simulated scoreboard shows them."""
    reads = [f.scoreboard for f in truth_frames if f.scoreboard is not None]
    timeline = ClockTimeline.from_exact(reads)
    # on the simulated board's display scale (ceil): the OCR timeline and the PBP then quantise alike
    return pbp_from_events(truth_events, timeline, lag_s=lag_s, clock_resolution="ceil")


def _latest(events: Sequence[Event]) -> list[Event]:
    latest: dict[str, Event] = {}
    for e in events:
        cur = latest.get(e.event_id)
        if cur is None or e.revision >= cur.revision:
            latest[e.event_id] = e
    return sorted((e for e in latest.values() if e.status is not EventStatus.RETRACTED), key=lambda e: e.t)


def shot_rows(engine_events: Sequence[Event], labels: Sequence[Event], frames: Sequence[FrameState], court: CourtSpec,
              game_id: str, tol_s: float = 2.0) -> list[dict[str, Any]]:
    """Shot-model rows: each engine field-goal attempt matched (same team, within ``tol_s``) to a PBP-labelled
    shot gets the PBP outcome; distance / zone from the engine's shot location, the nearest opponent from the
    cached tracks at the release. Free throws are left out (the model uses the league FT%)."""
    attempts = [e for e in engine_events if e.kind is EventKind.SHOT_ATTEMPT and e.x is not None and e.y is not None]
    results = {e.shot_id: e for e in labels if e.kind in (EventKind.SHOT_MADE, EventKind.SHOT_MISSED) and e.shot_id}
    label_att = [e for e in labels if e.kind is EventKind.SHOT_ATTEMPT and (e.value or 2) in (2, 3)]
    by_t = {f.frame_idx: f for f in frames}
    used: set[str] = set()
    rows = []
    for a in attempts:
        if (a.value or 2) == 1 or a.attrs.get("shot_type") == "free_throw":
            continue
        cands = [lab for lab in label_att if lab.event_id not in used and lab.team is a.team and abs(lab.t - a.t) <= tol_s]
        if not cands:
            continue
        lab = min(cands, key=lambda e: abs(e.t - a.t))
        used.add(lab.event_id)
        res = results.get(lab.shot_id)
        if res is None:
            continue
        side = court.nearest_basket(a.x, a.y)
        f = by_t.get(a.frame_idx)
        defender = None
        if f is not None and a.team is not None:
            ds = [math.hypot(p.x - a.x, p.y - a.y) for p in f.players if p.team is not None and p.team is not a.team]
            defender = round(min(ds), 2) if ds else None
        rows.append({"distance_ft": round(court.distance_to_basket(a.x, a.y, side), 2), "value": lab.value or a.value or 2,
                     "made": int(res.kind is EventKind.SHOT_MADE), "shot_type": a.attrs.get("shot_type") or "jumper",
                     "closest_defender_ft": defender, "catch_and_shoot": "", "zone": court.zone(a.x, a.y, side),
                     "game_id": game_id})
    return rows


def _csv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(columns))
    w.writeheader()
    for r in rows:
        w.writerow({k: "" if r.get(k) is None else r.get(k) for k in columns})
    return buf.getvalue()


def ingest_game(cfg: Any, source: Any, pbp: Sequence[PBPAction], out_dir: str | Path, game_id: str,
                config: IngestConfig | None = None, *, video: str | None = None, pbp_path: str | None = None,
                env: Mapping[str, str] | None = None) -> GameDataset:
    """Run, align and label one game (module docstring); returns the written dataset."""
    from courtvision.pipeline import run_offline

    icfg = config or IngestConfig()
    frames: list[FrameState] = []
    res = run_offline(cfg, source, write=False, env=env if env is not None else {}, on_frame=frames.append)
    engine = _latest(res.events)
    reads = [f.scoreboard for f in frames if f.scoreboard is not None]
    league = cfg.game.league
    court = court_for(league)
    acfg = AlignConfig(**{"league": league, "rights_status": icfg.rights_status, **icfg.align})
    is_sim = source.truth is not None
    timeline = ClockTimeline.fit(reads) if reads else None
    if timeline is None:
        raise ValueError(f"{game_id}: no scoreboard reads to fit a clock timeline (configure the scorebug / board ROI)")
    result = align_pbp(list(pbp), timeline, engine_events=engine, score_changes=score_changes(reads), config=acfg)
    labels = label_events(result.aligned)
    clips = make_clips(result, game_id=game_id, video_id=video or game_id, config=acfg, rights_status=icfg.rights_status)
    cue_rows = build_cue_dataset(engine, labels, game_id, tol_s=icfg.label_tol_s)
    for r in cue_rows:
        r["calibration_source"] = "sim" if is_sim else "pbp"
    shots = shot_rows(engine, labels, frames, court, game_id, icfg.label_tol_s)
    summary = result.summary()
    summary["timeline"] = {k: v for k, v in (timeline.stats or {}).items() if isinstance(v, (int, float, str, bool))}
    manifest = GameManifest(game_id=game_id, rights_status=icfg.rights_status, training_allowed=icfg.training_allowed,
                            gold=icfg.gold, source="sim" if is_sim else "video", video=video, pbp=pbp_path,
                            league=league, fps=float(getattr(source, "fps", 0.0) or 0.0) or None,
                            accepted=bool(result.accepted), alignment=summary,
                            created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    ds = GameDataset.write(out_dir, manifest, frames, engine, labels,
                           extra_jsonl={"clips.jsonl": (dataclasses.asdict(c) for c in clips),
                                        "cue_rows.jsonl": cue_rows,
                                        "scoreboard.jsonl": (dataclasses.asdict(r) for r in reads)},
                           extra_text={"shots.csv": _csv(shots, SHOT_COLUMNS),
                                       "roster.json": json.dumps(roster_to_dict(source.roster), indent=1)})
    log.info("ingested %s: %d frames, %d labels, %d clips, %d cue rows, %d shots; alignment %s (median made "
             "residual %s s)", game_id, len(frames), len(labels), len(clips), len(cue_rows), len(shots),
             "accepted" if result.accepted else "REJECTED", summary.get("median_made_residual_s"))
    return ds
