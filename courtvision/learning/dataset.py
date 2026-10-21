"""Ingested game datasets: what ``courtvision ingest`` writes and ``train`` / ``promote`` read.

One directory per game::

    <root>/<game_id>/
        manifest.json          GameManifest: rights, gold flag, alignment summary, sha256 of every file
        tracks.jsonl.gz        the perceived FrameState of every frame (what the event engine consumed), so the
                               engine can be re-run with other settings without the video or perception
        engine_events.jsonl    the engine's events on that run (latest revisions)
        labels.jsonl           PBP-aligned label events (Event dicts at the aligned times; ``attrs["align"]``)
        clips.jsonl            labelled clips (training/align_pbp.make_clips)
        cue_rows.jsonl         cue-model rows (engine cues labelled by the PBP; training/fit_cue_models)
        shots.csv              shot-model rows (engine shot locations, PBP make / miss; training/fit_shot_model)
        scoreboard.jsonl       the scoreboard reads (OCR) the clock timeline was fitted from

Rights travel with the data (TRAINING.md §1): ``rights_status`` in {own, licensed, research_only} and
``training_allowed``. Only datasets with ``own`` / ``licensed`` rights and ``training_allowed`` can make a
shippable model; NBA footage is ``research_only`` (the ingest default). ``gold`` datasets are frozen
evaluation sets: ``train`` refuses them, ``promote`` evaluates on them only.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from courtvision.types import (BallState, BBox, Event, EventKind, EventStatus, FrameState, PlayerState,
                               RimObservation, Roster, RosterEntry, ScoreboardReading, Team)

__all__ = ["GameDataset", "GameManifest", "PBP_KINDS", "RIGHTS", "frame_from_dict", "frame_to_dict",
           "label_events", "load_datasets", "roster_from_dict", "roster_to_dict"]

log = logging.getLogger(__name__)

RIGHTS = ("own", "licensed", "research_only")
# Kinds a play-by-play feed labels (no passes, possessions or dribbles): the training / gold objective.
PBP_KINDS = ("shot_attempt", "shot_made", "shot_missed", "rebound", "assist", "steal", "turnover", "block")


@dataclass
class GameManifest:
    game_id: str
    rights_status: str = "research_only"
    training_allowed: bool = False
    gold: bool = False
    source: str = "video"  # "video" | "sim"
    video: str | None = None
    pbp: str | None = None
    league: str = "nba"
    fps: float | None = None
    frames: int = 0
    accepted: bool = False  # the alignment gate (median |residual| on made FGs)
    alignment: dict[str, Any] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)  # name -> sha256
    created_utc: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.rights_status not in RIGHTS:
            raise ValueError(f"rights_status must be one of {RIGHTS}, got {self.rights_status!r}")

    @property
    def shippable(self) -> bool:
        return self.rights_status in ("own", "licensed") and self.training_allowed


# --------------------------------------------------------------------------- frame (de)serialisation


def _r(v: float | None, n: int = 3) -> float | None:
    return None if v is None else round(float(v), n)


def frame_to_dict(f: FrameState) -> dict[str, Any]:
    d: dict[str, Any] = {"i": f.frame_idx, "t": round(f.t, 4)}
    d["p"] = [[p.player_id, p.team.value if p.team else None, _r(p.x), _r(p.y), _r(p.vx), _r(p.vy), p.jersey,
               p.track_id, None if p.bbox is None else [_r(v, 1) for v in p.bbox.as_tuple()], _r(p.confidence)]
              for p in f.players]
    b = f.ball
    if b is not None:
        d["b"] = [b.visible, None if b.image_xy is None else [_r(v, 1) for v in b.image_xy],
                  None if b.court_xy is None else [_r(v) for v in b.court_xy], _r(b.z), _r(b.radius_px, 2),
                  _r(b.confidence), b.interpolated]
    if f.rims:
        d["r"] = [[r.side, [_r(v, 1) for v in r.bbox.as_tuple()], _r(r.confidence)] for r in f.rims]
    if f.scoreboard is not None:
        d["s"] = asdict(f.scoreboard)
    if f.image_size is not None:
        d["sz"] = list(f.image_size)
    if f.camera_xyz is not None:
        d["c"] = [_r(v) for v in f.camera_xyz]
    return d


def frame_from_dict(d: dict[str, Any]) -> FrameState:
    players = [PlayerState(pid, Team(team) if team else None, x, y, vx or 0.0, vy or 0.0, jersey, tid,
                           None if box is None else BBox(*box), conf if conf is not None else 1.0)
               for pid, team, x, y, vx, vy, jersey, tid, box, conf in d.get("p", [])]
    ball = None
    if d.get("b") is not None:
        vis, uv, xy, z, rad, conf, interp = d["b"]
        ball = BallState(vis, None if uv is None else tuple(uv), None if xy is None else tuple(xy), z, rad,
                         conf or 0.0, bool(interp))
    rims = [RimObservation(side, BBox(*box), conf) for side, box, conf in d.get("r", [])]
    sb = ScoreboardReading(**d["s"]) if d.get("s") else None
    return FrameState(d["i"], d["t"], players, ball, rims, sb, tuple(d["sz"]) if d.get("sz") else None,
                      tuple(d["c"]) if d.get("c") else None)


def roster_to_dict(roster: Roster) -> dict[str, Any]:
    return {"team_names": {t.value: n for t, n in roster.team_names.items()},
            "entries": [{"player_id": e.player_id, "team": e.team.value, "jersey": e.jersey, "name": e.name,
                         "position": e.position} for e in roster.entries]}


def roster_from_dict(d: dict[str, Any]) -> Roster:
    return Roster([RosterEntry(e["player_id"], Team(e["team"]), e["jersey"], e["name"], e.get("position"))
                   for e in d.get("entries", [])], {Team(k): v for k, v in d.get("team_names", {}).items()})


# --------------------------------------------------------------------------- labels from aligned PBP


def label_events(aligned: Iterable[Any], prefix: str = "L") -> list[Event]:
    """Label events from aligned PBP actions (training/align_pbp.AlignedAction) at their aligned times (``t``:
    snapped to an engine anchor or a score delta when possible, else clock-mapped). One shot row gives the attempt
    (at the snapped release when known) and its result; other kinds map one to one. Live-ball actions that were
    only clock-mapped are shifted by the median clock-vs-anchor residual of the game's snapped live-ball actions
    (the scorer's entry lag and the display quantisation, when at least three anchors exist); ``attrs["t_pbp"]``
    keeps the clock-mapped time."""
    aligned = list(aligned)
    res = sorted(a.residual_s for a in aligned if a.residual_s is not None and a.method.startswith("snap")
                 and a.action.live_ball)
    offset = res[len(res) // 2] if len(res) >= 3 else 0.0
    out: list[Event] = []
    n = 0
    for a in aligned:
        act = a.action
        if a.t is None or act.kind is None or act.synthetic:
            continue
        t = a.t - offset if a.method == "clock" and act.live_ball else a.t
        attrs = {"t_pbp": a.t_pbp, "align": a.method, "align_conf": a.align_conf, "pbp_action": act.action_number}
        if act.is_shot:
            n += 1
            sid = f"{prefix}s{n:05d}"
            t_att = a.t_release if a.t_release is not None else t
            kind_res = EventKind.SHOT_MADE if act.made else EventKind.SHOT_MISSED
            for kind, tt in ((EventKind.SHOT_ATTEMPT, t_att), (kind_res, t)):
                out.append(Event(f"{prefix}{len(out):06d}", kind, float(tt), 0, team=act.team, player_id=act.player_id,
                                 value=act.value, shot_id=sid, status=EventStatus.CONFIRMED, source="feed",
                                 attrs={**attrs, "shot_type": act.attrs.get("shot_type")}))
            continue
        if act.kind.value not in PBP_KINDS and act.kind is not EventKind.FOUL:
            continue
        out.append(Event(f"{prefix}{len(out):06d}", act.kind, float(t), 0, team=act.team, player_id=act.player_id,
                         other_player_id=act.attrs.get("other_player_id") or None, status=EventStatus.CONFIRMED,
                         source="feed", attrs={**attrs, **{k: v for k, v in act.attrs.items()
                                                          if k in ("offensive", "type", "cause")}}))
    return out


# --------------------------------------------------------------------------- dataset directory


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class GameDataset:
    """One ingested game directory (see the module docstring)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        mf = self.path / "manifest.json"
        if not mf.is_file():
            raise FileNotFoundError(f"no manifest.json in {self.path}")
        self.manifest = GameManifest(**json.loads(mf.read_text()))

    @property
    def game_id(self) -> str:
        return self.manifest.game_id

    # ------------------------------------------------------------------ writing

    @staticmethod
    def write(path: str | Path, manifest: GameManifest, frames: Iterable[FrameState], engine_events: Iterable[Event],
              labels: Iterable[Event], extra_jsonl: dict[str, Iterable[dict[str, Any]]] | None = None,
              extra_text: dict[str, str] | None = None) -> GameDataset:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        n = 0
        with gzip.open(root / "tracks.jsonl.gz", "wt") as fh:
            for f in frames:
                fh.write(json.dumps(frame_to_dict(f), separators=(",", ":")) + "\n")
                n += 1
        _write_jsonl(root / "engine_events.jsonl", (e.to_dict() for e in engine_events))
        _write_jsonl(root / "labels.jsonl", (e.to_dict() for e in labels))
        for name, rows in (extra_jsonl or {}).items():
            _write_jsonl(root / name, rows)
        for name, text in (extra_text or {}).items():
            (root / name).write_text(text)
        manifest.frames = n
        manifest.files = {p.name: _sha256(p) for p in sorted(root.iterdir()) if p.is_file() and p.name != "manifest.json"}
        (root / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True))
        return GameDataset(root)

    # ------------------------------------------------------------------ reading

    def frames(self) -> Iterator[FrameState]:
        with gzip.open(self.path / "tracks.jsonl.gz", "rt") as fh:
            for line in fh:
                yield frame_from_dict(json.loads(line))

    def roster(self) -> Roster | None:
        p = self.path / "roster.json"
        return roster_from_dict(json.loads(p.read_text())) if p.is_file() else None

    def labels(self) -> list[Event]:
        return [Event.from_dict(d) for d in _read_jsonl(self.path / "labels.jsonl")]

    def engine_events(self) -> list[Event]:
        return [Event.from_dict(d) for d in _read_jsonl(self.path / "engine_events.jsonl")]

    def jsonl(self, name: str) -> list[dict[str, Any]]:
        p = self.path / name
        return _read_jsonl(p) if p.is_file() else []

    def verify(self) -> list[str]:
        """Files whose sha256 no longer matches the manifest (a modified gold set is not frozen)."""
        bad = []
        for name, sha in self.manifest.files.items():
            p = self.path / name
            if not p.is_file() or _sha256(p) != sha:
                bad.append(name)
        return bad


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":"), default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_datasets(paths: Iterable[str | Path]) -> list[GameDataset]:
    """Game datasets from directories: each path is a game directory or a root holding game directories."""
    out: list[GameDataset] = []
    for p in paths:
        p = Path(p)
        if (p / "manifest.json").is_file():
            out.append(GameDataset(p))
        elif p.is_dir():
            out.extend(GameDataset(d) for d in sorted(p.iterdir()) if (d / "manifest.json").is_file())
        else:
            raise FileNotFoundError(f"no game dataset at {p}")
    return out
