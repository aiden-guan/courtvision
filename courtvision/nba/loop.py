"""The continuous NBA loop: analyse every game with the current models, score it against the box score, mine it for
training data, fit the next models on the games seen so far, and make a model current only when it does better on
games it was not fitted on (research use: NBA footage and statistics, docs/TRAINING.md §1).

    .venv/bin/python tools/nba_loop.py status | next | run | train-cues | calibrate-fusion | train-student | eval | promote

Everything lives under ``runs/`` (gitignored; the machine that has the footage):

* ``runs/nba/<game>/`` with ``game.json`` (``video``, ``espn_event``; docs/NBA.md). A game is its ESPN event id: a
  second folder of the same game (``nba_mac_jobs.sh`` links ``game.json`` into ``<game>-v2``) is a variant, listed
  by ``status`` and left alone unless named with ``--game``; ``"loop": false`` in a game.json leaves the folder out.
* ``runs/models/registry.json``: the versions of each kind of model (``student`` detector weights ``sNNN``, ``cue``
  models ``cNNN``, ``fusion`` configs ``fNNN``; ``c000`` / ``f000`` are the built-in defaults, and student weights
  found at ``runs/models/student/best.pt`` are imported as ``s001``), each with its provenance (the games and tags
  it was fitted on, the games held out, the command, the time, its metrics), the ``current`` version per kind and
  every promotion decision. Each version's files have their own folder ``runs/models/<kind>/<id>/``, written as
  ``<id>.partial`` and renamed when the job succeeds; a version is never overwritten.
* ``runs/nba/leaderboard.jsonl``: one row per evaluation of a model set on a game (box error, exact share,
  per-stat diffs, event F1 from ``report_fused.json``).

A model set (student, cue, fusion) analyses a game under the tag ``loop-<s>-<c>-<f>``. Per game, for the current
set (``GamePlanner``): ESPN summary, scorebug, camera (chunks of ``camera_chunk_s`` seconds, each its own step),
detections (``detections-<s>/``, stamped with the student), ball track (stamped too: ``ball_track.json`` has one
fixed path, so it is rebuilt whenever another student's detections are needed), events (re-used from another tag
with the same student and cue models), fusion, the two evaluations and the leaderboard row; then the training
data: cue rows, teacher labels and their ball labels, pseudo labels (``pseudo-<s>/``) and hard examples
(``teacher-hard-<s>/``). A detection run without a detected frame, or a whole-game events run without an event (a
video that cannot be decoded gives both, with exit code 0), fails instead of counting as done.
Global steps: ``train-cues``, ``calibrate-fusion`` and ``train-student`` fit a new version on every game except
the newest ``holdout`` ones the model has not seen (folder-name order: folders start with the date), unless a
version was already fitted on exactly these inputs; ``eval`` analyses every game with each open candidate swapped
into the current set. ``promote`` makes a candidate current only when its mean box error over the games it was not
fitted on is lower than the current set's on the same games. ``train-student`` takes the teacher-format folders
that have ``ball_labels.jsonl`` (a teacher folder made without them gets them from ``courtvision.training.ball_labels``)
and needs at least one teacher folder among them.

The runner starts one step at a time as a subprocess, output to ``runs/nba/<game>/logs/<step>.log`` (global steps:
``runs/nba/logs/``) with a JSON receipt next to it. It refuses to start while another CourtVision job runs, and a
step with less than ``min_free`` % of memory free (macOS ``memory_pressure -Q``, Linux ``/proc/meminfo``); while
a step runs, a watchdog stops its whole process tree when free memory falls below ``watch_free`` % (several heavy
jobs at once ran an 18 GB Mac out of memory). Re-running resumes: a stage is finished when its outputs exist.
Outputs whose path the loop chooses are written as ``*.partial`` and renamed when the step succeeds; the fixed
outputs of ``nba_analyze.py`` are written at the end of its run (``events.jsonl``, then ``run.json``), and an
interrupted ``detect`` resumes at its first missing chunk file. ``adopt`` stamps a detection folder made before
the loop with a student, so a whole game is not detected again.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CANDIDATE_TARGETS",
    "CURRENT_TARGETS",
    "KINDS",
    "STATUS_STAGES",
    "Game",
    "GamePlanner",
    "Loop",
    "LoopConfig",
    "ModelSet",
    "Registry",
    "Step",
    "StepResult",
    "adopt_detections",
    "assess",
    "discover_games",
    "execute",
    "format_next",
    "format_status",
    "free_memory_pct",
    "guard",
    "kill_tree",
    "leaderboard_row",
    "list_processes",
    "next_step",
    "other_jobs",
    "parse_meminfo",
    "parse_memory_pressure",
    "plan",
    "promote",
    "run_loop",
    "split_holdout",
    "status",
]

KINDS = ("student", "cue", "fusion")
PREFIX = {"student": "s", "cue": "c", "fusion": "f"}
KIND_OF = {v: k for k, v in PREFIX.items()}
MODEL_FILE = {"student": "best.pt", "cue": "cue_models.json", "fusion": "fusion.json"}
FIT_STAGE = {"cue": "train-cues", "fusion": "calibrate-fusion", "student": "train-student"}
VERSION_RE = re.compile(r"^[scf]\d{3}$")
TAG_PREFIX = "loop-"
DETECT_CHUNKS = 16  # detection files per game: an interrupted ``detect`` resumes at the first missing one
RIGHTS = "research_only"  # NBA footage and statistics (docs/TRAINING.md §1)
NBA_ANALYZE = "tools/nba_analyze.py"
FETCH_ESPN = "import sys; from courtvision.nba.espn import fetch_summary; fetch_summary(sys.argv[1], out=sys.argv[2])"
# the stages of one model set on one game, in the order ``status`` shows them
STATUS_STAGES = ("espn", "scorebug", "camera", "teacher", "ball_labels", "detect", "ball", "events", "fuse",
                 "eval_fused", "record", "eval_events", "cue_rows", "pseudo", "hard_teacher")
# what the current set needs on every game (evaluation first, then the training data), and what a candidate needs
CURRENT_TARGETS = ("espn", "record", "eval_events", "cue_rows", "hard_teacher", "pseudo", "teacher", "ball_labels")
CANDIDATE_TARGETS = ("record", "eval_events")
# another CourtVision job: a tool of tools/ (nba_mac_jobs.sh included) or a training module
JOB_RE = re.compile(r"(?:^|[\s/])(?:tools/[\w.-]+\.(?:py|sh)|courtvision\.training\.\w+)")
LOOP_READ_ONLY = {"status", "next"}


@dataclass
class LoopConfig:
    python: str = sys.executable  # the interpreter of every step (the CLI uses .venv/bin/python)
    min_free: float = 25.0  # % of memory that must be free to start a step
    watch_free: float = 8.0  # % of free memory below which the watchdog stops the running step
    poll_s: float = 5.0  # watchdog period
    grace_s: float = 10.0  # SIGTERM, then SIGKILL after this long
    camera_chunk_s: float = 1800.0  # video seconds per camera-tracking step
    epochs: int = 12  # train-student --epochs (nba_mac_jobs.sh EPOCHS)
    holdout: int = 1  # newest games left out of every fit (only while more than one game can be fitted)
    init_student: bool = True  # train-student starts from the current student's weights


# --------------------------------------------------------------------------- files


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    """The JSON document at ``path``; None when it is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def write_json(path: Path, data: Any) -> None:
    """Atomic write (a temporary file renamed over the target): a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a line cut short by a crash
        if isinstance(row, dict):
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


_SHA: dict[tuple[str, int, int], str] = {}


def file_sha256(path: Path) -> str | None:
    """sha256 of a file (None when missing), memoised on its path, size and modification time."""
    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _SHA:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _SHA[key] = h.hexdigest()
    return _SHA[key]


def partial(path: Path) -> Path:
    """Where a step writes ``path``; renamed to ``path`` when the step succeeds. (``partNN.jsonl.partial`` stays out
    of ``CameraTrack.load``'s ``part*.jsonl``.)"""
    return path.with_name(path.name + ".partial")


def inputs_key(inputs: Any) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else repr(float(v))


# --------------------------------------------------------------------------- games


@dataclass
class Game:
    name: str  # folder name under runs/nba
    root: Path
    meta: dict[str, Any]  # game.json
    game_id: str  # the ESPN event id (else the folder name): the same game whatever its folder
    primary: bool = True  # the folder the loop works on for this game id
    variant_of: str | None = None  # the primary folder of the same game
    excluded: bool = False  # "loop": false in game.json
    error: str | None = None  # game.json unreadable


def discover_games(nba: Path) -> list[Game]:
    """Every ``runs/nba/*/game.json``. Folders of one game id: the first whose game.json is a real file (not a link)
    is primary, the others are variants."""
    games: list[Game] = []
    for gj in sorted(Path(nba).glob("*/game.json")):
        root = gj.parent
        try:
            meta = json.loads(gj.read_text())
            err = None if isinstance(meta, dict) else "game.json is not an object"
        except (OSError, ValueError) as exc:
            meta, err = {}, f"game.json unreadable: {exc}"
        meta = meta if isinstance(meta, dict) else {}
        games.append(Game(root.name, root, meta, str(meta.get("espn_event") or root.name),
                          excluded=meta.get("loop") is False, error=err))
    groups: dict[str, list[Game]] = {}
    for g in games:
        if g.excluded or g.error:
            g.primary = False
        else:
            groups.setdefault(g.game_id, []).append(g)
    for group in groups.values():
        real = [g for g in group if not (g.root / "game.json").is_symlink()]
        first = (real or group)[0]
        for g in group:
            if g is not first:
                g.primary, g.variant_of = False, first.name
    return games


def scorebug_duration(path: Path) -> float | None:
    """Video seconds the scorebug scan covers: its last sample's ``t``."""
    try:
        with Path(path).open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 65536))
            tail = fh.read().decode(errors="replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        try:
            return float(json.loads(line)["t"])
        except (ValueError, KeyError, TypeError):
            continue
    return None


def camera_layout(duration_s: float, chunk_s: float) -> dict[str, Any]:
    """Camera tracking in chunks of ``chunk_s`` video seconds; the last one runs to the end of the video."""
    n = max(1, math.ceil(duration_s / chunk_s - 1e-9)) if chunk_s > 0 else 1
    width = max(2, len(str(n - 1)))
    parts = [{"name": f"part{k:0{width}d}.jsonl", "start": k * chunk_s if n > 1 else 0.0,
              "duration": chunk_s if k < n - 1 else None} for k in range(n)]
    return {"chunk_s": chunk_s, "duration_s": duration_s, "parts": parts}


def camera_center(part: Path) -> list[float] | None:
    """The camera centre in a camera part's header line (``nba_track_court.py``)."""
    try:
        with Path(part).open() as fh:
            head = json.loads(fh.readline())
        return [float(v) for v in head["center"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def find_det_dir(root: Path, student: str | None) -> str | None:
    """The detection folder stamped with ``student`` (``<folder>/loop.json``; ``adopt`` stamps older ones)."""
    if not student:
        return None
    found = [p.parent.name for p in sorted(Path(root).glob("detections*/loop.json"))
             if (read_json(p) or {}).get("student") == student]
    own = f"detections-{student}"
    return own if own in found else (found[0] if found else None)


def folder_kind(folder: Path) -> str:
    """``pseudo`` (``meta.json`` of pseudo_labels.py) or ``teacher`` (nba_teacher_labels.py, or no meta.json)."""
    meta = read_json(folder / "meta.json")
    return "pseudo" if isinstance(meta, dict) and meta.get("kind") == "pseudo" else "teacher"


def trainable(folder: Path) -> bool:
    """A teacher-format folder ``train_student`` takes: ``detections.jsonl`` and ``ball_labels.jsonl``, and not a
    pseudo folder whose ``meta.json`` says it is incomplete."""
    meta = read_json(folder / "meta.json")
    if isinstance(meta, dict) and meta.get("kind") == "pseudo" and meta.get("complete") is False:
        return False
    return (folder / "detections.jsonl").is_file() and (folder / "ball_labels.jsonl").is_file()


def training_folders(game: Game) -> list[Path]:
    """A game's student training data: ``teacher/``, every ``teacher-hard-<s>/`` and the newest ``pseudo-<s>/``."""
    out = [game.root / "teacher"] if trainable(game.root / "teacher") else []
    out += [d for d in sorted(game.root.glob("teacher-hard-s[0-9][0-9][0-9]")) if trainable(d)]
    pseudo = [d for d in game.root.glob("pseudo-s[0-9][0-9][0-9]") if trainable(d)]
    if pseudo:
        out.append(max(pseudo, key=lambda d: int(d.name[-3:])))
    return out


def events_complete(tdir: Path) -> bool:
    """``nba_analyze.py events`` writes ``events.jsonl``, then ``run.json``: both exist only after a whole run."""
    return (tdir / "events.jsonl").is_file() and (tdir / "run.json").is_file()


# --------------------------------------------------------------------------- versions and model sets


def kind_of(vid: str) -> str:
    if not VERSION_RE.match(vid or ""):
        raise ValueError(f"not a version id: {vid!r} (s001, c002, f003, ...)")
    return KIND_OF[vid[0]]


def vnum(vid: str) -> int:
    return int(vid[1:])


@dataclass(frozen=True)
class ModelSet:
    student: str | None
    cue: str
    fusion: str

    @property
    def key(self) -> str:
        return f"{self.student or 's000'}-{self.cue}-{self.fusion}"

    @property
    def tag(self) -> str:
        return TAG_PREFIX + self.key

    def swap(self, vid: str) -> ModelSet:
        return replace(self, **{kind_of(vid): vid})

    def as_dict(self) -> dict[str, str | None]:
        return {"student": self.student, "cue": self.cue, "fusion": self.fusion}


class Registry:
    """``runs/models/registry.json``: versions per kind, the current one per kind, and every decision."""

    def __init__(self, models: Path) -> None:
        self.dir = Path(models)
        self.path = self.dir / "registry.json"
        self.data = self._read()

    def _read(self) -> dict[str, Any]:
        d = read_json(self.path)
        if self.path.exists() and not isinstance(d, dict):  # never replaced by an empty registry
            raise ValueError(f"{self.path} is unreadable: restore it (registry.json.bak holds the previous one)")
        if not isinstance(d, dict):
            d = {}
        d.setdefault("schema", 1)
        d.setdefault("versions", {})
        d.setdefault("current", {})
        d.setdefault("decisions", [])
        for k in KINDS:
            d["versions"].setdefault(k, {})
            d["current"].setdefault(k, None)
        return d

    @contextlib.contextmanager
    def locked(self) -> Iterator[Registry]:
        """Exclusive access for a read-modify-write (``promote`` may run next to ``run``); reloads from disk."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with (self.dir / "registry.lock").open("a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                self.data = self._read()
                yield self
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def save(self) -> None:
        if self.path.is_file():
            shutil.copy2(self.path, self.path.with_name(self.path.name + ".bak"))
        write_json(self.path, self.data)

    def versions(self, kind: str) -> dict[str, dict[str, Any]]:
        return self.data["versions"][kind]

    def get(self, vid: str | None) -> dict[str, Any] | None:
        if not vid or not VERSION_RE.match(vid):
            return None
        return self.versions(kind_of(vid)).get(vid)

    def current_id(self, kind: str) -> str | None:
        return self.data["current"].get(kind)

    def add(self, entry: dict[str, Any]) -> None:
        kind, vid = entry["kind"], entry["id"]
        if vid in self.versions(kind):
            raise ValueError(f"version {vid} already exists: versions are never overwritten")
        self.versions(kind)[vid] = entry

    def set_current(self, kind: str, vid: str) -> None:
        if vid not in self.versions(kind):
            raise ValueError(f"unknown {kind} version {vid}")
        self.data["current"][kind] = vid

    def decide(self, decision: dict[str, Any]) -> None:
        self.data["decisions"].append(decision)

    def ids_on_disk(self, kind: str) -> list[str]:
        """Version folders of ``kind``, finished or ``.partial``."""
        out = []
        for p in (self.dir / kind).glob(f"{PREFIX[kind]}[0-9][0-9][0-9]*"):
            vid = p.name.removesuffix(".partial")
            if p.is_dir() and VERSION_RE.match(vid):
                out.append(vid)
        return out

    def next_id(self, kind: str) -> str:
        nums = [vnum(v) for v in self.versions(kind)] + [vnum(v) for v in self.ids_on_disk(kind)]
        return f"{PREFIX[kind]}{max(nums, default=0) + 1:03d}"

    def pending(self, kind: str) -> str | None:
        """The lowest ``<id>.partial`` folder with a ``loop.json``, not registered and without a finished folder."""
        ids = [p.name.removesuffix(".partial") for p in (self.dir / kind).glob("*.partial")]
        ids = [v for v in ids if VERSION_RE.match(v) and v not in self.versions(kind)
               and (self.dir / kind / f"{v}.partial" / "loop.json").is_file() and not (self.dir / kind / v).exists()]
        return min(ids, key=vnum) if ids else None

    def bootstrap(self, loop: Loop, persist: bool = False) -> None:
        """The defaults ``c000`` / ``f000`` and, when no student is registered, ``runs/models/student/best.pt`` as
        ``s001`` (copied into ``student/s001/`` when ``persist``: later training there never touches it)."""
        v = self.data["versions"]
        for kind, vid, note in (("cue", "c000", "the engine's hand-set cold-start cue models (no cue_model_path)"),
                                ("fusion", "f000", "FusionConfig defaults (no --fusion-file)")):
            if vid not in v[kind]:
                v[kind][vid] = {"id": vid, "kind": kind, "path": None, "sha256": None, "created": utc_now(),
                                "source": "default", "fit_games": [], "fit_folders": [], "holdout_games": [],
                                "games_seen": [], "tags": [], "command": None, "metrics": {}, "rights": RIGHTS,
                                "note": note}
            if self.data["current"][kind] is None:
                self.data["current"][kind] = vid
        legacy = self.dir / "student" / "best.pt"
        if not v["student"] and legacy.is_file():
            v["student"]["s001"] = _imported_student(loop, legacy, self.dir / "student" / "s001" / "best.pt")
            if self.data["current"]["student"] is None:
                self.data["current"]["student"] = "s001"
        if persist:
            for e in v["student"].values():
                if e.get("import_from"):
                    src, dst = loop.abs(e["import_from"]), loop.abs(e["path"])
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    for a, b in ((src, dst), (src.parent / "train_summary.json", dst.parent / "train_summary.json")):
                        if a.is_file() and not b.exists():  # through a temporary name: no half copy is ever b
                            shutil.copy2(a, partial(b))
                            os.replace(partial(b), b)
                    e["sha256"] = file_sha256(dst)
                    e["imported_from"] = e.pop("import_from")
            self.save()


def _imported_student(loop: Loop, legacy: Path, dst: Path) -> dict[str, Any]:
    summary = read_json(legacy.parent / "train_summary.json") or {}
    data = [data_folder(str(d)) for d in summary.get("data") or []]
    games = [loop.game_of_folder(d) for d in data]
    fit = sorted({g.game_id for g in games if g is not None})
    unknown = not data or any(g is None for g in games)
    seen = sorted(set(fit) | {g.game_id for g in loop.primaries()}) if unknown else fit
    note = "weights trained before the loop"
    if unknown:
        note += "; their training data is unknown, so every game present at import counts as seen (none held out)"
    metrics = {k: summary[k] for k in ("best_ball_f1", "epochs", "folders") if k in summary}
    return {"id": "s001", "kind": "student", "path": loop.rel(dst), "import_from": loop.rel(legacy), "sha256": None,
            "created": utc_now(), "source": "imported", "fit_games": fit, "fit_folders": data, "holdout_games": [],
            "games_seen": seen, "tags": [], "command": None, "metrics": metrics, "rights": RIGHTS, "note": note}


# --------------------------------------------------------------------------- the loop


class Loop:
    """The runs tree, the model registry and the leaderboard, with the paths steps are given."""

    def __init__(self, runs: Path | str, repo: Path | str, config: LoopConfig | None = None) -> None:
        self.repo = Path(repo).resolve()
        runs = Path(runs)
        self.runs = (runs if runs.is_absolute() else self.repo / runs).resolve()
        self.cfg = config or LoopConfig()
        self.nba = self.runs / "nba"
        self.models = self.runs / "models"
        self.leaderboard_path = self.nba / "leaderboard.jsonl"
        self.refresh()

    def refresh(self) -> None:
        self.games = discover_games(self.nba)
        self.registry = Registry(self.models)
        self.registry.bootstrap(self)
        self._board: list[dict[str, Any]] | None = None
        self.fit_notes: dict[str, str] = {}  # kind -> why the last plan_fit planned nothing

    def rel(self, p: Path | str) -> str:
        """A path as the steps get it: relative to the repository (their working directory) when inside it."""
        p = Path(p)
        if not p.is_absolute():
            return str(p)
        try:
            return str(p.relative_to(self.repo))
        except ValueError:
            return str(p)

    def abs(self, p: Path | str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.repo / p

    def game(self, name: str) -> Game:
        for g in self.games:
            if g.name == name or g.root == self.abs(name).resolve():
                return g
        raise KeyError(f"no game folder {name!r} under {self.nba} (with a game.json)")

    def primaries(self) -> list[Game]:
        return [g for g in self.games if g.primary]

    def game_of_folder(self, folder: str | Path) -> Game | None:
        p = self.abs(folder).resolve()
        hits = [g for g in self.games if p == g.root.resolve() or g.root.resolve() in p.parents]
        hits.sort(key=lambda g: not g.primary)
        return hits[0] if hits else None

    def current_set(self) -> ModelSet:
        cur = self.registry.data["current"]
        return ModelSet(cur.get("student"), cur.get("cue") or "c000", cur.get("fusion") or "f000")

    def model_path(self, vid: str | None) -> str | None:
        """The model file of a version as steps get it (None: the built-in default)."""
        ver = self.registry.get(vid)
        return None if ver is None else ver.get("path")

    def model_available(self, vid: str | None) -> bool:
        ver = self.registry.get(vid)
        if ver is None:
            return False
        if ver.get("path") is None:
            return True
        return self.abs(ver.get("import_from") or ver["path"]).is_file()

    def fitted_on(self, kind: str, key: str) -> str | None:
        """The version already fitted on these inputs, if any."""
        for vid, v in self.registry.versions(kind).items():
            if v.get("inputs_key") == key:
                return vid
        return None

    # leaderboard ------------------------------------------------------------

    def board(self) -> list[dict[str, Any]]:
        if self._board is None:
            self._board = read_jsonl(self.leaderboard_path)
        return self._board

    def latest_rows(self) -> dict[tuple[str, str], dict[str, Any]]:
        """(game id, model set key) -> its newest leaderboard row (primary folders only)."""
        prim = {g.name for g in self.primaries()}
        out: dict[tuple[str, str], dict[str, Any]] = {}
        for r in self.board():
            if r.get("game") in prim:
                out[(str(r.get("game_id")), str(r.get("set")))] = r
        return out

    def recorded(self, game: str, tag: str, sha: str | None) -> bool:
        return any(r.get("game") == game and r.get("tag") == tag and r.get("report_sha256") == sha for r in self.board())

    def record(self, game_name: str, mset: ModelSet) -> dict[str, Any]:
        g = self.game(game_name)
        rep = g.root / "runs" / mset.tag / "report_fused.json"
        doc = read_json(rep)
        if not isinstance(doc, dict):
            raise ValueError(f"{rep}: missing or unreadable")
        row = leaderboard_row(g, mset, doc, self.rel(rep), file_sha256(rep))
        append_jsonl(self.leaderboard_path, row)
        self._board = None
        return row

    # versions ---------------------------------------------------------------

    def register(self, kind: str, vid: str) -> dict[str, Any]:
        """Add a finished version folder to the registry (its provenance was written to ``loop.json`` when the job
        started). The first version of a kind becomes current."""
        vdir = self.models / kind / vid
        prov = read_json(vdir / "loop.json")
        if not isinstance(prov, dict):
            raise ValueError(f"{vdir / 'loop.json'} is missing: cannot register {vid}")
        main = vdir / MODEL_FILE[kind]
        if not main.is_file():
            raise ValueError(f"{main}: the job finished without writing it")
        entry = {**prov, "id": vid, "kind": kind, "path": self.rel(main), "sha256": file_sha256(main),
                 "created": utc_now(), "source": "fit", "metrics": version_metrics(kind, vdir)}
        with self.registry.locked() as reg:
            reg.bootstrap(self)
            reg.add(entry)
            if reg.current_id(kind) is None:
                reg.set_current(kind, vid)
                reg.decide({"time": utc_now(), "kind": kind, "candidate": vid, "current": None, "outcome": "promoted",
                            "promoted": True, "forced": False, "reason": f"the first {kind} version"})
            reg.save()
        return entry


def version_metrics(kind: str, vdir: Path) -> dict[str, Any]:
    """The metrics a job wrote next to a version: cue fit reports, the fusion calibration, the training summary."""
    if kind == "cue":
        doc = read_json(vdir / "cue_models.json") or {}
        out: dict[str, Any] = {"for_live": doc.get("for_live")}
        for k, r in (read_json(vdir / "cue_models.report.json") or {}).items():
            if isinstance(r, dict):
                cal, gate = r.get("calibration") or {}, r.get("gate") or {}
                out[k] = {"status": r.get("status"), "n_rows": r.get("n_rows"), "n_holdout": r.get("n_holdout"),
                          "split": r.get("split"), "ece": cal.get("ece"), "gate_certified": gate.get("certified")}
        return out
    if kind == "fusion":
        doc = read_json(vdir / "fusion.json") or {}
        out = {"config": doc.get("config", {})}
        out.update({k: v for k, v in doc.items() if k != "config" and isinstance(v, (int, float, str, bool))})
        if isinstance(doc.get("metrics"), dict):
            out["metrics"] = doc["metrics"]
        return out
    summary = read_json(vdir / "train_summary.json") or {}
    return {k: summary[k] for k in ("best_ball_f1", "epochs", "folders") if k in summary}


def data_folder(spec: str) -> str:
    """The folder of a ``train_student --data`` spec (``DIR`` or ``DIR:WEIGHT``)."""
    head, sep, tail = spec.rpartition(":")
    if sep and head:
        try:
            float(tail)
            return head
        except ValueError:
            pass
    return spec


def leaderboard_row(game: Game, mset: ModelSet, report: dict[str, Any], path: str, sha: str | None) -> dict[str, Any]:
    """A leaderboard row from ``report_fused.json`` (``nba_analyze.py eval --file fused.jsonl``)."""
    box = report.get("box") or {}
    return {"time": utc_now(), "game": game.name, "game_id": game.game_id, "tag": mset.tag, "set": mset.key,
            "models": mset.as_dict(), "box_error": box.get("error"), "exact_share": box.get("exact_share"),
            "diffs": {f"{r.get('team')}.{r.get('stat')}": r.get("diff") for r in box.get("rows", [])},
            "event_f1": {k: (v or {}).get("f1") for k, v in (report.get("events") or {}).items()},
            "report": path, "report_sha256": sha}


# --------------------------------------------------------------------------- steps


@dataclass
class Step:
    name: str  # unique: "<game>:<stage>[:<detail>]", or "<stage>:<version>" for a global step
    stage: str
    detail: str
    game: str | None
    heavy: bool
    why: str
    cmd: list[str] | None  # None: bookkeeping the runner does itself
    log: Path | None
    prepare: list[tuple[Any, ...]] = field(default_factory=list)  # actions before the command
    finish: list[tuple[Any, ...]] = field(default_factory=list)  # actions once it succeeded

    def shell(self) -> str:
        return shlex.join(self.cmd) if self.cmd else "(bookkeeping, no command)"

    def label(self) -> str:
        return f"{self.game or 'models'} {self.stage}" + (f" {self.detail}" if self.detail else "")


def apply_actions(loop: Loop, actions: Sequence[tuple[Any, ...]]) -> None:
    for a in actions:
        op = a[0]
        if op == "rename":
            src, dst = Path(a[1]), Path(a[2])
            if not src.exists():
                raise ValueError(f"{src}: the step finished without writing it")
            if dst.exists() and (src.is_dir() or dst.is_dir()):
                raise ValueError(f"{dst} already exists: not overwritten")
            os.replace(src, dst)
        elif op == "clean":  # a stale .partial folder of an interrupted step
            p = Path(a[1])
            if not p.name.endswith(".partial"):
                raise ValueError(f"refusing to remove {p}: not a .partial folder")
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        elif op == "mkdir":
            Path(a[1]).mkdir(parents=True, exist_ok=True)
        elif op == "require":
            if not Path(a[1]).is_file():
                raise ValueError(f"{a[1]}: the step finished without writing it")
        elif op == "drop_broken_npz":  # np.savez_compressed writes in place: a killed run can leave half a zip
            for f in sorted(Path(a[1]).glob("*.npz")):
                if not zipfile.is_zipfile(f):
                    f.unlink()
        elif op == "detections_sane":  # a video cv2 cannot read gives empty chunk files and exit 0
            files = sorted(Path(a[1]).glob("*.npz"))
            if not sum(_npz_frames(f) for f in files):
                for f in files:  # all empty: detect would skip them when it resumes
                    f.unlink()
                raise ValueError(f"{a[1]}: no frame was detected (could the video be read?)")
        elif op == "events_sane":  # likewise an empty whole-game run: never a box-score result
            tdir = Path(a[1])
            if not _rows(tdir / "events.jsonl"):
                os.replace(tdir / "run.json", tdir / "run.json.empty")  # the run does not count as done
                raise ValueError(f"{tdir / 'events.jsonl'} has no event (could the video be read?); run.json moved "
                                 "to run.json.empty")
        elif op == "write_json":
            write_json(Path(a[1]), a[2])
        elif op == "stamp":
            write_json(Path(a[1]), {**a[2], "time": utc_now()})
        elif op == "ball_stamp":
            root = Path(a[1])
            write_json(root / "ball_track.loop.json", {"student": a[2], "det_dir": a[3], "time": utc_now(),
                                                       "sha256": file_sha256(root / "ball_track.json")})
        elif op == "copy_events":
            src, dst = Path(a[1]), Path(a[2])
            dst.mkdir(parents=True, exist_ok=True)
            for name in ("events.jsonl", "run.json"):
                if not (dst / name).exists():
                    tmp = dst / f".{name}.copy"
                    shutil.copy2(src / name, tmp)
                    os.replace(tmp, dst / name)
            write_json(dst / "loop.json", {"events_from": src.name, "time": utc_now()})
        elif op == "record":
            loop.record(a[1], a[2])
        elif op == "register":
            loop.register(a[1], a[2])
        else:
            raise ValueError(f"unknown action {op!r}")


def describe(a: tuple[Any, ...], loop: Loop) -> str:
    op = a[0]
    if op == "rename":
        return f"rename {loop.rel(a[1])} -> {loop.rel(a[2])}"
    if op == "record":
        return f"append the {a[2].key} row of {a[1]} to {loop.rel(loop.leaderboard_path)}"
    if op == "register":
        return f"register {a[2]} in {loop.rel(loop.registry.path)}"
    if op in ("stamp", "write_json"):
        return f"write {loop.rel(a[1])}"
    if op == "ball_stamp":
        return f"stamp ball_track.json with student {a[2]} ({a[3]})"
    if op == "copy_events":
        return f"copy events.jsonl and run.json from {loop.rel(a[1])}"
    if op == "clean":
        return f"remove the stale {loop.rel(a[1])}"
    if op == "require":
        return f"check that {loop.rel(a[1])} was written"
    if op == "drop_broken_npz":
        return f"remove unreadable (cut short) chunk files in {loop.rel(a[1])}"
    if op == "detections_sane":
        return f"check that {loop.rel(a[1])} holds detected frames"
    if op == "events_sane":
        return f"check that {loop.rel(a[1])}/events.jsonl holds events"
    return f"{op} {' '.join(loop.rel(x) if isinstance(x, Path) else str(x) for x in a[1:])}"


# --------------------------------------------------------------------------- planning one game


class GamePlanner:
    """What one model set still needs on one game, found backwards from the wanted stages. Each stage ends up
    ``done``, ``ready`` (its step is in ``steps``), ``waiting`` (for an earlier stage), ``blocked: <reason>`` or
    ``failed`` (its step failed earlier in this run: ``skip``)."""

    def __init__(self, loop: Loop, game: Game, mset: ModelSet, skip: frozenset[str] | set[str] = frozenset()) -> None:
        self.loop, self.g, self.m, self.skip = loop, game, mset, skip
        self.root = game.root
        self.tdir = game.root / "runs" / mset.tag
        self.py = loop.cfg.python
        self.state: dict[str, str] = {}
        self.steps: list[Step] = []
        self.det_dir = find_det_dir(game.root, mset.student)
        self.events_src: Path | None = None  # another tag's events re-used (same student and cue models)

    def plan(self, targets: Sequence[str]) -> list[Step]:
        for t in targets:
            self.want(t)
        return self.steps

    def want(self, stage: str) -> bool:
        if stage not in self.state:
            self.state[stage] = "waiting"
            if getattr(self, "_" + stage)():
                self.state[stage] = "done"
        return self.state[stage] == "done"

    # helpers ----------------------------------------------------------------

    def _r(self, p: Path) -> str:
        return self.loop.rel(p)

    def _name(self, stage: str, detail: str = "") -> str:
        return f"{self.g.name}:{stage}" + (f":{detail}" if detail else "")

    def _blocked(self, stage: str, reason: str) -> bool:
        self.state[stage] = "blocked: " + reason
        return False

    def _waiting(self, stage: str) -> bool:
        self.state[stage] = "waiting"
        return False

    def _skipped(self, stage: str, detail: str = "") -> bool:
        if self._name(stage, detail) in self.skip:
            self.state[stage] = "failed"
            return True
        return False

    def _emit(self, stage: str, detail: str, why: str, cmd: list[str] | None, heavy: bool,
              prepare: Sequence[tuple[Any, ...]] = (), finish: Sequence[tuple[Any, ...]] = ()) -> bool:
        slug = f"{stage}-{detail}" if detail else stage
        log = self.root / "logs" / f"{slug}.log" if cmd is not None else None
        self.steps.append(Step(self._name(stage, detail), stage, detail, self.g.name, heavy, why, cmd, log,
                               list(prepare), list(finish)))
        self.state[stage] = "ready"
        return False

    def _video(self) -> tuple[str | None, str | None]:
        """(video as game.json gives it, None) or (None, why it cannot be read)."""
        v = self.g.meta.get("video")
        if not v:
            return None, "game.json has no video"
        if not self.loop.abs(str(v)).exists():
            return None, f"video not found: {v}"
        return str(v), None

    def _copy_events(self) -> list[tuple[Any, ...]]:
        if self.events_src is not None and not events_complete(self.tdir):
            return [("copy_events", self.events_src, self.tdir)]
        return []

    def _ball_ok(self) -> bool:
        st = read_json(self.root / "ball_track.loop.json") or {}
        return bool(self.det_dir and st.get("student") == self.m.student and st.get("det_dir") == self.det_dir
                    and st.get("sha256") == file_sha256(self.root / "ball_track.json"))

    # stages -----------------------------------------------------------------

    def _espn(self) -> bool:
        out = self.root / "espn_summary.json"
        if out.is_file():
            return True
        event = self.g.meta.get("espn_event")
        if not event:
            return self._blocked("espn", "game.json has no espn_event")
        if self._skipped("espn"):
            return False
        tmp = partial(out)
        return self._emit("espn", "", "no espn_summary.json (the ESPN box score and play-by-play)",
                          [self.py, "-c", FETCH_ESPN, str(event), self._r(tmp)], False, finish=[("rename", tmp, out)])

    def _scorebug(self) -> bool:
        out = self.root / "scorebug.jsonl"
        if out.is_file():
            return True
        video, problem = self._video()
        if problem:
            return self._blocked("scorebug", problem)
        if self._skipped("scorebug"):
            return False
        tmp = partial(out)
        return self._emit("scorebug", "", "no scorebug.jsonl", [self.py, "tools/nba_scan_scorebug.py", str(video),
                          "--out", self._r(tmp)], True, finish=[("rename", tmp, out)])

    def _camera(self) -> bool:
        d = self.root / "camera_parts"
        manifest = read_json(d / "loop.json")
        if not isinstance(manifest, dict):
            if any(d.glob("part*.jsonl")):
                return True  # tracked before the loop
            manifest = None
        elif all((d / p["name"]).is_file() for p in manifest["parts"]):
            return True
        video, problem = self._video()
        if problem:
            return self._blocked("camera", problem)
        if not self.want("scorebug"):
            return self._waiting("camera")
        layout = manifest
        if layout is None:
            duration = scorebug_duration(self.root / "scorebug.jsonl")
            if duration is None:
                return self._blocked("camera", "scorebug.jsonl has no samples")
            layout = camera_layout(duration, self.loop.cfg.camera_chunk_s)
        parts = layout["parts"]
        k = next(i for i, p in enumerate(parts) if not (d / p["name"]).is_file())
        part = parts[k]
        detail = part["name"].split(".")[0]
        if self._skipped("camera", detail):
            return False
        out = d / part["name"]
        tmp = partial(out)
        cmd = [self.py, "tools/nba_track_court.py", str(video), "--scorebug", self._r(self.root / "scorebug.jsonl"),
               "--out", self._r(tmp), "--start", _num(part["start"])]
        if part["duration"] is not None:
            cmd += ["--duration", _num(part["duration"])]
        if k > 0:  # the centre part 0 estimated (nba_mac_jobs.sh retrack-full does the same)
            center = camera_center(d / parts[0]["name"])
            if center is None:
                return self._blocked("camera", f"no camera centre in {parts[0]['name']}")
            cmd += ["--center", ",".join(str(v) for v in center)]
        end = "end" if part["duration"] is None else f"{part['start'] + part['duration']:.0f}s"
        return self._emit("camera", detail, f"camera part {k + 1}/{len(parts)} ({part['start']:.0f}s-{end})", cmd, True,
                          prepare=[] if manifest else [("write_json", d / "loop.json", layout)],
                          finish=[("rename", tmp, out)])

    def _detect(self) -> bool:
        s = self.m.student
        if self.det_dir is not None:
            return True
        if not s or self.loop.registry.get(s) is None:
            return self._blocked("detect", "no student weights (train-student, or runs/models/student/best.pt)")
        if not self.loop.model_available(s):
            return self._blocked("detect", f"weights of {s} missing: {self.loop.model_path(s)}")
        _, problem = self._video()
        if problem:
            return self._blocked("detect", problem)
        if self._skipped("detect", s):
            return False
        if not self.want("camera"):
            return self._waiting("detect")
        weights = str(self.loop.model_path(s))
        det = f"detections-{s}"
        cmd = [self.py, NBA_ANALYZE, self._r(self.root), "detect", "--weights", weights, "--det-dir", det,
               "--workers", "1", "--chunks", str(DETECT_CHUNKS)]
        stamp = {"student": s, "weights": weights, "weights_sha256": (self.loop.registry.get(s) or {}).get("sha256"),
                 "chunks": DETECT_CHUNKS}
        # an interrupted run resumes at the first missing chunk file; one cut short while written would be skipped
        return self._emit("detect", s, f"no detections of student {s}", cmd, True,
                          prepare=[("drop_broken_npz", self.root / det)],
                          finish=[("detections_sane", self.root / det), ("stamp", self.root / det / "loop.json", stamp)])

    def _ball(self) -> bool:
        if self._ball_ok():
            return True
        s = self.m.student or ""
        if self._skipped("ball", s):
            return False
        if not self.want("detect"):
            return self._waiting("ball")
        st = read_json(self.root / "ball_track.loop.json") or {}
        if not (self.root / "ball_track.json").is_file():
            why = "no ball_track.json"
        elif st.get("student"):
            why = f"ball_track.json is from {st['student']}'s detections ({st.get('det_dir')}) or changed since"
        else:
            why = "ball_track.json was built before the loop (detections unknown)"
        # heavy: it loads every detection of the game
        return self._emit("ball", s, why, [self.py, NBA_ANALYZE, self._r(self.root), "ball", "--det-dir",
                          str(self.det_dir)], True, finish=[("ball_stamp", self.root, s, self.det_dir)])

    def _events(self) -> bool:
        if events_complete(self.tdir):
            return True
        for d in sorted((self.root / "runs").glob(f"{TAG_PREFIX}{self.m.student}-{self.m.cue}-f[0-9][0-9][0-9]")):
            if d.name != self.m.tag and events_complete(d):
                self.events_src = d  # the engine's events do not depend on the fusion config
                return True
        _, problem = self._video()
        if problem:
            return self._blocked("events", problem)
        cue = self.loop.model_path(self.m.cue)
        if not self.loop.model_available(self.m.cue):
            return self._blocked("events", f"cue models {self.m.cue} missing: {cue}")
        if self._skipped("events", self.m.tag):
            return False
        if not self.want("ball"):  # the ball needs the detections, which need the camera and the scorebug
            return self._waiting("events")
        cmd = [self.py, NBA_ANALYZE, self._r(self.root), "events", "--tag", self.m.tag, "--det-dir", str(self.det_dir)]
        if cue:
            cmd += ["--engine", json.dumps({"cue_model_path": cue})]
        return self._emit("events", self.m.tag, f"no events of {self.m.key} (reads the video)", cmd, True,
                          finish=[("events_sane", self.tdir)])

    def _fuse(self) -> bool:
        if (self.tdir / "fused.jsonl").is_file():
            return True
        fusion = self.loop.model_path(self.m.fusion)
        if not self.loop.model_available(self.m.fusion):
            return self._blocked("fuse", f"fusion config {self.m.fusion} missing: {fusion}")
        if self._skipped("fuse", self.m.tag):
            return False
        if not (self.want("events") and self.want("ball")):  # fusion reads the ball track's flights
            return self._waiting("fuse")
        cmd = [self.py, NBA_ANALYZE, self._r(self.root), "fuse", "--tag", self.m.tag]
        if fusion:
            cmd += ["--fusion-file", fusion]
        why = "no fused.jsonl" + (f" (events of {self.events_src.name})" if self.events_src else "")
        return self._emit("fuse", self.m.tag, why, cmd, False, prepare=self._copy_events())

    def _eval_fused(self) -> bool:
        if (self.tdir / "report_fused.json").is_file():
            return True
        if self._skipped("eval_fused", self.m.tag):
            return False
        if not (self.want("fuse") and self.want("espn")):
            return self._waiting("eval_fused")
        cmd = [self.py, NBA_ANALYZE, self._r(self.root), "eval", "--tag", self.m.tag, "--file", "fused.jsonl"]
        return self._emit("eval_fused", self.m.tag, "no report_fused.json", cmd, False,
                          finish=[("record", self.g.name, self.m)])

    def _record(self) -> bool:
        if not self.want("eval_fused"):
            return self._waiting("record")
        rep = self.tdir / "report_fused.json"
        if self.loop.recorded(self.g.name, self.m.tag, file_sha256(rep)):
            return True
        if self._skipped("record", self.m.tag):
            return False
        return self._emit("record", self.m.tag, "report_fused.json is not on the leaderboard", None, False,
                          finish=[("record", self.g.name, self.m)])

    def _eval_events(self) -> bool:
        if (self.tdir / "report_events.json").is_file():
            return True
        if self._skipped("eval_events", self.m.tag):
            return False
        if not (self.want("events") and self.want("espn")):
            return self._waiting("eval_events")
        cmd = [self.py, NBA_ANALYZE, self._r(self.root), "eval", "--tag", self.m.tag]
        return self._emit("eval_events", self.m.tag, "no report_events.json (the engine alone)", cmd, False,
                          prepare=self._copy_events())

    def _cue_rows(self) -> bool:
        out = self.tdir / "cue_rows.jsonl"
        if out.is_file():
            return True
        if self._skipped("cue_rows", self.m.tag):
            return False
        if not (self.want("events") and self.want("espn")):
            return self._waiting("cue_rows")
        tmp = partial(out)
        cmd = [self.py, "-m", "courtvision.training.nba_cue_rows", self._r(self.root), "--tag", self.m.tag,
               "--out", self._r(tmp)]
        return self._emit("cue_rows", self.m.tag, "no cue rows (engine cues labelled by the play-by-play)", cmd, False,
                          prepare=self._copy_events(), finish=[("rename", tmp, out)])

    def _teacher(self) -> bool:
        t = self.root / "teacher"
        if (t / "detections.jsonl").is_file():
            return True
        if t.exists():
            return self._blocked("teacher", "teacher/ has no detections.jsonl (an unfinished run?): move it away")
        video, problem = self._video()
        if problem:
            return self._blocked("teacher", problem)
        if self._skipped("teacher"):
            return False
        if not self.want("scorebug"):
            return self._waiting("teacher")
        tmp = partial(t)
        cmd = [self.py, "tools/nba_teacher_labels.py", str(video), "--scorebug", self._r(self.root / "scorebug.jsonl"),
               "--out", self._r(tmp)]
        return self._emit("teacher", "", "no teacher labels (RF-DETR on sampled clips)", cmd, True,
                          prepare=[("clean", tmp)], finish=[("rename", tmp, t)])

    def _ball_labels(self) -> bool:
        """The ball labels of the teacher folders (``teacher/``, ``teacher-hard-<s>/``). nba_teacher_labels.py links
        them at the end of its run; a folder made without them gets them here (train-student needs them)."""
        if not self.want("teacher"):
            return self._waiting("ball_labels")
        folders = [self.root / "teacher", *sorted(self.root.glob("teacher-hard-s[0-9][0-9][0-9]"))]
        todo = [d for d in folders if (d / "detections.jsonl").is_file() and not (d / "ball_labels.jsonl").is_file()]
        if not todo:
            return True
        if self._skipped("ball_labels", todo[0].name):
            return False
        cmd = [self.py, "-m", "courtvision.training.ball_labels", self._r(todo[0])]
        return self._emit("ball_labels", todo[0].name, f"{todo[0].name}/ has no ball_labels.jsonl (train-student "
                          "needs them)", cmd, False)

    def _pseudo(self) -> bool:
        s = self.m.student
        if not s:
            return self._blocked("pseudo", "no student")
        d = self.root / f"pseudo-{s}"
        if d.is_dir():
            return True
        _, problem = self._video()
        if problem:
            return self._blocked("pseudo", problem)
        if self._skipped("pseudo", s):
            return False
        if not (self.want("record") and self.want("ball")):  # the whole analysis of this student, its ball track
            return self._waiting("pseudo")
        tmp = partial(d)
        cmd = [self.py, "-m", "courtvision.training.pseudo_labels", self._r(self.root), "--out", self._r(tmp),
               "--det-dir", str(self.det_dir)]  # the folder its ball track was linked from
        return self._emit("pseudo", s, f"no pseudo labels from {s}'s analysis (reads the video)", cmd, True,
                          prepare=[("clean", tmp)],
                          finish=[("rename", tmp, d), ("stamp", d / "loop.json", {"student": s, "tag": self.m.tag})])

    def _hard_times(self) -> bool:
        out = self.tdir / "hard_examples.json"
        if out.is_file():
            return True
        if self._skipped("hard_times", self.m.tag):
            return False
        if not (self.want("record") and self.want("espn")):
            return self._waiting("hard_times")
        tmp = partial(out)
        cmd = [self.py, "tools/nba_hard_examples.py", self._r(self.root), "--tag", self.m.tag, "--out", self._r(tmp)]
        return self._emit("hard_times", self.m.tag, "no hard-example times", cmd, False, finish=[("rename", tmp, out)])

    def _hard_teacher(self) -> bool:
        s = self.m.student
        if not s:
            return self._blocked("hard_teacher", "no student")
        d = self.root / f"teacher-hard-{s}"
        if d.is_dir():
            return True
        video, problem = self._video()
        if problem:
            return self._blocked("hard_teacher", problem)
        if self._skipped("hard_teacher", s):
            return False
        if not (self.want("hard_times") and self.want("scorebug")):
            return self._waiting("hard_teacher")
        tmp, times = partial(d), self.tdir / "hard_examples.json"
        doc = read_json(times)
        if isinstance(doc, dict) and doc.get("clips") == [] or doc == []:
            return True  # the analysis made no error the detector could be blamed for: nothing to label
        cmd = [self.py, "tools/nba_teacher_labels.py", str(video), "--scorebug", self._r(self.root / "scorebug.jsonl"),
               "--out", self._r(tmp), "--times", self._r(times)]
        return self._emit("hard_teacher", s, f"no teacher labels on the hard examples of {s}", cmd, True,
                          prepare=[("clean", tmp)],
                          finish=[("rename", tmp, d), ("stamp", d / "loop.json",
                                                       {"student": s, "tag": self.m.tag, "times": self._r(times)})])


# --------------------------------------------------------------------------- global steps


def split_holdout(games: Sequence[Game], n: int, seen: set[str] | frozenset[str] = frozenset()
                  ) -> tuple[list[Game], list[Game]]:
    """(held out, fitted on): the newest ``n`` games (folder-name order) the model has not seen, keeping at least
    one game to fit on."""
    ordered = sorted(games, key=lambda g: g.name)
    unseen = [g for g in ordered if g.game_id not in seen]
    k = max(0, min(n, len(ordered) - 1, len(unseen)))
    hold = unseen[len(unseen) - k:] if k else []
    return hold, [g for g in ordered if g not in hold]


def _global_step(loop: Loop, stage: str, vid: str, why: str, cmd: list[str] | None, heavy: bool,
                 prepare: Sequence[tuple[Any, ...]] = (), finish: Sequence[tuple[Any, ...]] = ()) -> Step:
    log = loop.nba / "logs" / f"{stage}-{vid}.log" if cmd is not None else None
    return Step(f"{stage}:{vid}", stage, vid, None, heavy, why, cmd, log, list(prepare), list(finish))


def _provenance(kind: str, vid: str, cur: ModelSet, fit: Sequence[Game], holdout: Sequence[Game],
                inputs: Any, key: str, cmd: list[str], seen: Sequence[str] = (), **extra: Any) -> dict[str, Any]:
    return {"kind": kind, "id": vid, "context": cur.key, "fit_games": sorted({g.game_id for g in fit}),
            "fit_folders": [g.name for g in fit], "holdout_games": sorted({g.game_id for g in holdout}),
            "games_seen": sorted(set(seen) | {g.game_id for g in fit}), "inputs": inputs, "inputs_key": key,
            "command": cmd, "started": utc_now(), "rights": RIGHTS, **extra}


def plan_registrations(loop: Loop) -> list[Step]:
    """Finished version folders not in the registry (a crash between the rename and the registration)."""
    steps = []
    for kind in KINDS:
        for d in sorted((loop.models / kind).glob(f"{PREFIX[kind]}[0-9][0-9][0-9]")):
            if d.is_dir() and loop.registry.get(d.name) is None and (d / "loop.json").is_file():
                steps.append(_global_step(loop, "register", d.name, f"{loop.rel(d)} is not registered", None, False,
                                          finish=[("register", kind, d.name)]))
    return steps


def plan_fit(loop: Loop, kind: str, skip: frozenset[str] | set[str] = frozenset()) -> list[Step]:
    """The steps a new version of ``kind`` needs now: its missing inputs, or the fit itself; nothing when a version
    was already fitted on exactly these inputs."""
    cur = loop.current_set()
    py = loop.cfg.python
    stage = FIT_STAGE[kind]

    def nothing(reason: str) -> list[Step]:
        loop.fit_notes[kind] = reason
        return []

    pending = loop.registry.pending(kind)  # a fit that started and did not finish
    if kind == "student" and pending is not None and (partial(loop.models / kind / pending) / "last.pt").is_file():
        return _resume_student(loop, pending, skip, nothing)  # epochs done: train_student --resume finishes them
    analysed = [g for g in loop.primaries() if (g.root / "runs" / cur.tag / "report_fused.json").is_file()]
    if kind != "student" and not analysed:
        return nothing(f"no game is analysed with the current set {cur.key} yet (nba_loop.py run)")
    if kind == "cue":
        before: list[Step] = []
        eligible = []
        for g in analysed:
            if (g.root / "runs" / cur.tag / "cue_rows.jsonl").is_file():
                eligible.append(g)
            else:
                gp = GamePlanner(loop, g, cur, skip)
                gp.want("cue_rows")
                before += gp.steps
        if before:
            return before
        if not eligible:
            return nothing(f"no game has cue rows of {cur.tag} (their step failed)")
        holdout, fit = split_holdout(eligible, loop.cfg.holdout)
        files = [g.root / "runs" / cur.tag / "cue_rows.jsonl" for g in fit]
        inputs: Any = [{"game": g.name, "game_id": g.game_id, "tag": cur.tag, "file": loop.rel(f), "sha256": file_sha256(f),
                        "rows": _rows(f)} for g, f in zip(fit, files, strict=True)]
        key = inputs_key(inputs)
        if done := loop.fitted_on(kind, key):
            return nothing(f"{done} was fitted on these inputs already ({len(fit)} game(s){_held_note(holdout)})")
        if not any(i["rows"] for i in inputs):  # nba_cue_rows exits 0 with no rows
            return nothing(f"the cue rows of {', '.join(g.name for g in fit)} are empty: nothing to fit")
        vid = pending or loop.registry.next_id(kind)
        tmp = partial(loop.models / kind / vid)
        cmd = [py, "-m", "courtvision.training.fit_cue_models"] + [a for f in files for a in ("--jsonl", loop.rel(f))]
        cmd += ["--out", loop.rel(tmp / MODEL_FILE[kind])]
        why = f"cue rows of {len(fit)} game(s) no version was fitted on" + _held_note(holdout)
        prov = _provenance(kind, vid, cur, fit, holdout, inputs, key, cmd, tags=[cur.tag])
        heavy = False
    elif kind == "fusion":
        eligible = [g for g in analysed if events_complete(g.root / "runs" / cur.tag)]
        if not eligible:
            return nothing(f"no game has the events of {cur.tag}")
        holdout, fit = split_holdout(eligible, loop.cfg.holdout)
        files = [g.root / "runs" / cur.tag / "events.jsonl" for g in fit]
        inputs = [{"game": g.name, "game_id": g.game_id, "tag": cur.tag, "file": loop.rel(f), "sha256": file_sha256(f)}
                  for g, f in zip(fit, files, strict=True)]
        key = inputs_key(inputs)
        if done := loop.fitted_on(kind, key):
            return nothing(f"{done} was calibrated on these inputs already ({len(fit)} game(s){_held_note(holdout)})")
        before = []
        for g in fit:  # the calibration re-fuses each game: its ball track must be the current student's
            gp = GamePlanner(loop, g, cur, skip)
            if not gp.want("ball"):
                if not gp.steps:
                    return nothing(f"the ball track of {g.name} cannot be rebuilt now ({gp.state.get('ball')})")
                before += gp.steps
        if before:
            return before
        vid = pending or loop.registry.next_id(kind)
        tmp = partial(loop.models / kind / vid)
        cmd = [py, "tools/nba_calibrate_fusion.py"] + [loop.rel(g.root) for g in fit]
        # one process: its default is up to four, each with every game loaded, next to nothing else on the Mac
        cmd += ["--tag", cur.tag, "--out", loop.rel(tmp / MODEL_FILE[kind]), "--workers", "1"]
        why = f"events of {len(fit)} game(s) no fusion version was calibrated on" + _held_note(holdout)
        prov = _provenance(kind, vid, cur, fit, holdout, inputs, key, cmd, tags=[cur.tag])
        heavy = False
    else:
        folders = {g.name: training_folders(g) for g in loop.primaries()}
        eligible = [g for g in loop.primaries() if folders[g.name]]
        if not eligible:
            return nothing("no game has a teacher-format folder with ball_labels.jsonl")
        init = cur.student if loop.cfg.init_student and cur.student else None
        init_ver = loop.registry.get(init) or {}
        seen = set(init_ver.get("games_seen") or [])
        holdout, fit = split_holdout(eligible, loop.cfg.holdout, seen)
        data = [d for g in fit for d in folders[g.name]]
        inputs = {"data": [{"game": g.name, "game_id": g.game_id, "folder": loop.rel(d),
                            "detections_sha256": file_sha256(d / "detections.jsonl"),
                            "ball_labels_sha256": file_sha256(d / "ball_labels.jsonl")}
                           for g in fit for d in folders[g.name]],
                  "epochs": loop.cfg.epochs, "init": init}
        key = inputs_key(inputs)
        if done := loop.fitted_on(kind, key):
            return nothing(f"{done} was trained on these inputs already ({len(data)} folder(s){_held_note(holdout)})")
        if all(folder_kind(d) == "pseudo" for d in data):  # its held-out score is measured on teacher clips
            return nothing("train_student needs a teacher folder, and the games it would train on have only "
                           f"pseudo labels{_held_note(holdout)}")
        if init and not loop.model_available(init):
            return nothing(f"the weights of {init} are missing: {loop.model_path(init)}")
        vid = pending or loop.registry.next_id(kind)  # a pending fit without a finished epoch starts over
        tmp = partial(loop.models / kind / vid)
        cmd = [py, "-m", "courtvision.training.train_student"] + [a for d in data for a in ("--data", loop.rel(d))]
        cmd += ["--out", loop.rel(tmp), "--epochs", str(loop.cfg.epochs)]
        if init:
            cmd += ["--init", str(loop.model_path(init))]
        why = (f"{len(data)} training folder(s) of {len(fit)} game(s) no student was trained on"
               + (f", starting from {init}" if init else "") + _held_note(holdout))
        prov = _provenance(kind, vid, cur, fit, holdout, inputs, key, cmd, seen=sorted(seen), init=init,
                           epochs=loop.cfg.epochs, data=[loop.rel(d) for d in data])
        heavy = True
    if f"{stage}:{vid}" in skip:
        return nothing(f"{stage} {vid} failed earlier in this run (see {loop.rel(loop.nba / 'logs' / f'{stage}-{vid}.log')})")
    loop.fit_notes.pop(kind, None)
    prepare: list[tuple[Any, ...]] = [("clean", tmp), ("mkdir", tmp), ("write_json", tmp / "loop.json", prov)]
    return [_global_step(loop, stage, vid, why, cmd, heavy, prepare=prepare, finish=_version_finish(loop, kind, vid))]


def _version_finish(loop: Loop, kind: str, vid: str) -> list[tuple[Any, ...]]:
    tmp = partial(loop.models / kind / vid)
    return [("require", tmp / MODEL_FILE[kind]), ("rename", tmp, loop.models / kind / vid), ("register", kind, vid)]


def _held_note(holdout: Sequence[Game]) -> str:
    return f"; held out: {', '.join(g.name for g in holdout)}" if holdout else "; no game held out"


def _rows(path: Path) -> int:
    try:
        with Path(path).open() as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def _npz_frames(path: Path) -> int:
    """Frames in a detection chunk file (``DetectionCache.save``); 0 when it cannot be read."""
    try:
        with np.load(path) as z:
            return int(z["frames"].size)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return 0


def _resume_student(loop: Loop, vid: str, skip: frozenset[str] | set[str],
                    nothing: Callable[[str], list[Step]]) -> list[Step]:
    """``train_student --resume`` with the command the training started with (the same data and ``--epochs``:
    it replays its schedule from ``last.pt``)."""
    tmp = loop.models / "student" / f"{vid}.partial"
    prov = read_json(tmp / "loop.json") or {}
    cmd = list(prov.get("command") or [])
    missing = [d for d in prov.get("data", []) if not loop.abs(data_folder(d)).is_dir()]
    if not cmd or missing:
        return nothing(f"the interrupted training {vid} cannot resume: its data {', '.join(missing) or '?'} is gone "
                       f"(remove {loop.rel(tmp)} to train afresh)")
    if f"train-student:{vid}" in skip:
        return nothing(f"train-student {vid} failed earlier in this run "
                       f"(see {loop.rel(loop.nba / 'logs' / f'train-student-{vid}.log')})")
    return [_global_step(loop, "train-student", vid, f"finish the interrupted training of {vid} (from its last.pt)",
                         [*cmd, "--resume"], True, finish=_version_finish(loop, "student", vid))]


def open_candidates(loop: Loop, kind: str) -> list[str]:
    """Versions newer than the current one that no decision rejected."""
    reg = loop.registry
    cur = reg.current_id(kind)
    rejected = {d.get("candidate") for d in reg.data["decisions"] if d.get("outcome") == "rejected"}
    return [v for v in sorted(reg.versions(kind), key=vnum)
            if (cur is None or vnum(v) > vnum(cur)) and v not in rejected and loop.model_available(v)]


def plan_candidate_evals(loop: Loop, games: Sequence[Game], candidate: str | None = None,
                         skip: frozenset[str] | set[str] = frozenset()) -> list[Step]:
    """Each open candidate (or ``candidate``) swapped into the current set, on every game: fusion candidates first
    (their events are re-used), then cue models, then students (new detections)."""
    cur = loop.current_set()
    if candidate is not None:
        if loop.registry.get(candidate) is None:
            if (loop.models / kind_of(candidate) / candidate / "loop.json").is_file():
                return []  # finished but not registered yet: plan_registrations comes first
            raise ValueError(f"unknown version {candidate}")
        cands = [candidate]
    else:
        cands = [v for k in ("fusion", "cue", "student") for v in open_candidates(loop, k)]
    steps: list[Step] = []
    for v in cands:
        mset = cur.swap(v)
        if mset.student is None or mset == cur:
            continue
        for g in games:
            steps += GamePlanner(loop, g, mset, skip).plan(CANDIDATE_TARGETS)
    return steps


def plan(loop: Loop, *, game: str | None = None, scope: str = "all", candidate: str | None = None,
         skip: frozenset[str] | set[str] = frozenset()) -> list[Step]:
    """The steps ready now, in the order the runner takes them. ``scope``: ``all`` (every game with the current
    set; without ``game`` also the fits; then the candidate evaluations), ``eval`` (candidate evaluations only) or
    a kind (that kind's fit only)."""
    games = [loop.game(game)] if game else loop.primaries()
    steps = plan_registrations(loop)
    if scope == "all":
        cur = loop.current_set()
        for g in games:
            steps += GamePlanner(loop, g, cur, skip).plan(CURRENT_TARGETS)
        if game is None:
            for kind in ("cue", "fusion", "student"):
                steps += plan_fit(loop, kind, skip)
    elif scope in KINDS:
        steps += plan_fit(loop, scope, skip)
    elif scope != "eval":
        raise ValueError(f"unknown scope {scope!r}")
    if scope in ("all", "eval"):
        steps += plan_candidate_evals(loop, games, candidate, skip)
    out: list[Step] = []
    names: set[str] = set()
    for s in steps:
        if s.name not in names and s.name not in skip:
            names.add(s.name)
            out.append(s)
    return out


def next_step(loop: Loop, **kw: Any) -> tuple[Step | None, list[Step]]:
    """(the next step with a command, the plan) of ``plan(loop, **kw)``."""
    steps = plan(loop, **kw)
    return next((s for s in steps if s.cmd is not None), None), steps


# --------------------------------------------------------------------------- promotion


def assess(loop: Loop, vid: str) -> dict[str, Any]:
    """The promotion rule for ``vid``: its mean box error over the games it was not fitted on, against the current
    set's on the same games (latest leaderboard rows). ``outcome``: promoted / rejected / refused."""
    reg = loop.registry
    ver = reg.get(vid)
    if ver is None:
        raise ValueError(f"unknown version {vid}")
    kind = kind_of(vid)
    cur_id = reg.current_id(kind)
    if vid == cur_id:
        raise ValueError(f"{vid} is already the current {kind}")
    cur = loop.current_set()
    cand = cur.swap(vid)
    rows = loop.latest_rows()
    seen = set(ver.get("games_seen") or [])
    games = loop.primaries()
    held = [g for g in games if g.game_id not in seen]
    out: dict[str, Any] = {"time": utc_now(), "kind": kind, "candidate": vid, "current": cur_id,
                           "current_set": cur.key, "candidate_set": cand.key,
                           "holdout_games": [g.game_id for g in held], "forced": False}

    def err(g: Game, m: ModelSet) -> float | None:
        r = rows.get((g.game_id, m.key))
        return None if r is None or r.get("box_error") is None else float(r["box_error"])

    def compare(gs: Sequence[Game]) -> dict[str, Any]:
        per = {g.game_id: {"game": g.name, "candidate": err(g, cand), "current": err(g, cur)} for g in gs}
        c = sum(p["candidate"] for p in per.values()) / len(per)
        b = sum(p["current"] for p in per.values()) / len(per)
        return {"games": [g.game_id for g in gs], "candidate_box_error": round(c, 5), "current_box_error": round(b, 5),
                "per_game": per}

    if not held:
        both = [g for g in games if err(g, cand) is not None and err(g, cur) is not None]
        if both:
            out["in_sample"] = compare(both)
        out.update(outcome="refused", reason=f"no held-out game: {vid} was fitted on every game "
                                             f"({', '.join(sorted(seen)) or '-'}), so nothing measures it on new games")
        return out
    evaluated = [g for g in held if err(g, cur) is not None]
    if not evaluated:
        out.update(outcome="refused", reason=f"the held-out game(s) {', '.join(g.name for g in held)} are not evaluated "
                                             f"with the current set {cur.key} yet (nba_loop.py run)")
        return out
    missing = [g for g in evaluated if err(g, cand) is None]
    if missing:
        out.update(outcome="refused", reason=f"{vid} is not evaluated on {', '.join(g.name for g in missing)} with the "
                                             f"current set yet (nba_loop.py eval --candidate {vid})")
        return out
    out.update(compare(evaluated))
    c, b = out["candidate_box_error"], out["current_box_error"]
    n = len(evaluated)
    if c < b:
        out.update(outcome="promoted", reason=f"mean box error {c:.4f} < {b:.4f} (current) on {n} held-out game(s)")
    else:
        out.update(outcome="rejected", reason=f"mean box error {c:.4f} >= {b:.4f} (current) on {n} held-out game(s)")
    biased = sorted(set((reg.get(cur_id) or {}).get("games_seen") or []) & set(out["games"]))
    if biased:
        out["note"] = f"the current {kind} {cur_id} was fitted on {', '.join(biased)}: the comparison favours it"
    return out


def promote(loop: Loop, vid: str, force: bool = False) -> dict[str, Any]:
    """Apply the promotion rule to ``vid`` and record the decision in the registry. ``force`` promotes only when no
    game is held out (a single game); a candidate measured on held-out games follows the rule."""
    with loop.registry.locked() as reg:
        reg.bootstrap(loop, persist=True)
        d = assess(loop, vid)
        if force:
            if not d["holdout_games"]:
                d.update(outcome="promoted", forced=True, reason=d["reason"] + "; promoted with --force")
            elif d["outcome"] != "promoted":
                d["reason"] += " (--force only applies when no game is held out)"
        elif d["outcome"] == "refused" and not d["holdout_games"]:
            d["reason"] += "; pass --force to promote it anyway"
        d["promoted"] = d["outcome"] == "promoted"
        reg.decide(d)
        if d["promoted"]:
            reg.set_current(d["kind"], vid)
        reg.save()
    loop.refresh()
    return d


def adopt_detections(loop: Loop, game: str, det_dir: str, student: str | None = None) -> dict[str, Any]:
    """Declare an existing detection folder (made before the loop) the whole-game detections of ``student``
    (default: the current one), so ``detect`` is not run again."""
    g = loop.game(game)
    d = g.root / det_dir
    if not d.is_dir() or not any(d.glob("*.npz")):
        raise ValueError(f"{d}: no detection files (*.npz)")
    with loop.registry.locked() as reg:
        reg.bootstrap(loop, persist=True)
    loop.refresh()
    s = student or loop.registry.current_id("student")
    ver = loop.registry.get(s)
    if ver is None:
        raise ValueError(f"unknown student {s!r}")
    old = read_json(d / "loop.json") or {}
    if old.get("student") not in (None, s):
        raise ValueError(f"{d} is already stamped with student {old['student']}")
    other = find_det_dir(g.root, s)
    if other not in (None, d.name):
        raise ValueError(f"{other} already holds the detections of {s}")
    stamp = {"student": s, "weights": ver.get("path"), "weights_sha256": ver.get("sha256"), "adopted": True,
             "time": utc_now(), "note": "declared by nba_loop.py adopt: made by these weights, over the whole game"}
    write_json(d / "loop.json", stamp)
    return stamp


# --------------------------------------------------------------------------- memory and process guard


def parse_memory_pressure(text: str) -> float | None:
    """``memory_pressure -Q`` (macOS): "System-wide memory free percentage: NN%"."""
    m = re.search(r"free percentage:\s*(\d+(?:\.\d+)?)\s*%", text)
    return float(m.group(1)) if m else None


def parse_meminfo(text: str) -> float | None:
    """``/proc/meminfo`` (Linux): MemAvailable / MemTotal, %."""
    vals: dict[str, float] = {}
    for line in text.splitlines():
        k, _, v = line.partition(":")
        try:
            vals[k.strip()] = float(v.split()[0])
        except (IndexError, ValueError):
            continue
    if vals.get("MemTotal", 0) > 0 and "MemAvailable" in vals:
        return 100.0 * vals["MemAvailable"] / vals["MemTotal"]
    return None


def free_memory_pct() -> float | None:
    """System-wide free memory, %; None when it cannot be read (the guard and the watchdog are then off)."""
    exe = shutil.which("memory_pressure")
    if exe:
        try:
            v = parse_memory_pressure(subprocess.run([exe, "-Q"], capture_output=True, text=True, timeout=60,
                                                     check=False).stdout)
        except (OSError, subprocess.SubprocessError):
            v = None
        if v is not None:
            return v
    try:
        return parse_meminfo(Path("/proc/meminfo").read_text())
    except OSError:
        return None


def list_processes() -> list[tuple[int, int, str]]:
    """(pid, parent pid, command line) of every process (``ps``; Linux ``/proc`` when ps is missing)."""
    try:
        out = subprocess.run(["ps", "-ww", "-Ao", "pid=,ppid=,command="], capture_output=True, text=True, timeout=60,
                             check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return _proc_scan()
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((int(parts[0]), int(parts[1]), parts[2] if len(parts) > 2 else ""))
    return rows


def _proc_scan() -> list[tuple[int, int, str]]:
    rows = []
    for d in Path("/proc").glob("[0-9]*"):
        try:
            ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except (OSError, ValueError, IndexError):
            continue
        rows.append((int(d.name), ppid, cmd))
    return rows


def descendants(pid: int, procs: Sequence[tuple[int, int, str]]) -> list[int]:
    children: dict[int, list[int]] = {}
    for p, pp, _ in procs:
        children.setdefault(pp, []).append(p)
    out, todo = [], [pid]
    while todo:
        for c in children.get(todo.pop(), []):
            if c not in out and c != pid:
                out.append(c)
                todo.append(c)
    return out


def is_job(cmd: str) -> bool:
    """A CourtVision job: a tool of tools/ (nba_mac_jobs.sh included) or a training module; ``nba_loop.py status``
    and ``next`` only read."""
    if not JOB_RE.search(cmd):
        return False
    if "nba_loop.py" in cmd:
        words = set(cmd.split())
        return not (words & LOOP_READ_ONLY) or bool(words & {"run", "eval", "train-cues", "calibrate-fusion",
                                                             "train-student", "promote", "adopt"})
    return True


def other_jobs(procs: Sequence[tuple[int, int, str]] | None = None, me: int | None = None) -> list[tuple[int, str]]:
    """CourtVision jobs running outside this process, its ancestors (a ``caffeinate`` wrapper) and its children."""
    procs = list_processes() if procs is None else procs
    me = os.getpid() if me is None else me
    parent = {p: pp for p, pp, _ in procs}
    mine = {me, *descendants(me, procs)}
    p = me
    while p in parent and parent[p] not in mine and parent[p] > 1:
        p = parent[p]
        mine.add(p)
    return [(pid, cmd) for pid, _, cmd in procs if pid not in mine and is_job(cmd)]


def guard(cfg: LoopConfig, memory: bool = True) -> str | None:
    """Why a step must not start now (another job, too little free memory), or None."""
    jobs = other_jobs()
    if jobs:
        return ("another CourtVision job is running (wait for it or stop it):\n"
                + "\n".join(f"  pid {p}: {c[:240]}" for p, c in jobs))
    if memory:
        free = free_memory_pct()
        if free is not None and free < cfg.min_free:
            return f"only {free:.0f}% of memory free (need {cfg.min_free:.0f}%): close other apps first"
    return None


def _signal(pid: int, sig: int, group: bool = False) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        (os.killpg if group else os.kill)(pid, sig)


def _alive(pid: int) -> bool:
    """Running (a zombie, killed but not yet reaped, is not; Linux tells them apart through /proc)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def kill_tree(proc: subprocess.Popen, grace_s: float = 10.0) -> None:
    """Stop a step and everything it started: SIGTERM to its process group and to each descendant (also those that
    left the group), SIGKILL to whatever still runs after ``grace_s``."""
    tree = descendants(proc.pid, list_processes())
    _signal(proc.pid, signal.SIGTERM, group=True)
    for p in tree:
        _signal(p, signal.SIGTERM)
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if proc.poll() is not None and not any(_alive(p) for p in tree):
            return
        time.sleep(0.05)
    _signal(proc.pid, signal.SIGKILL, group=True)
    for p in (proc.pid, *tree):
        _signal(p, signal.SIGKILL)


@dataclass
class StepResult:
    returncode: int
    seconds: float = 0.0
    killed: bool = False  # stopped by the memory watchdog
    interrupted: bool = False  # Ctrl-C or SIGTERM to the loop
    free_start: float | None = None
    free_min: float | None = None


def execute(step: Step, loop: Loop) -> StepResult:
    """Run a step's command in its own session (so its whole tree can be stopped), output appended to its log,
    while a watchdog polls free memory."""
    assert step.cmd is not None and step.log is not None
    cfg = loop.cfg
    step.log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    free0 = free_memory_pct()
    with step.log.open("a") as log:
        log.write(f"\n=== {utc_now()} {step.name}\n$ {step.shell()}\n")
        log.flush()
        proc = subprocess.Popen(step.cmd, cwd=loop.repo, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=dict(os.environ, PYTHONUNBUFFERED="1"), start_new_session=True)
        res = _watch(proc, cfg, log)
        log.write(f"=== exit {res.returncode}{' (stopped by the watchdog)' if res.killed else ''}"
                  f"{' (interrupted)' if res.interrupted else ''} after {time.time() - t0:.0f}s\n")
    res.seconds, res.free_start = time.time() - t0, free0
    return res


def _watch(proc: subprocess.Popen, cfg: LoopConfig, log: Any) -> StepResult:
    free_min: float | None = None
    try:
        while True:
            try:
                return StepResult(proc.wait(timeout=cfg.poll_s), free_min=free_min)
            except subprocess.TimeoutExpired:
                pass
            free = free_memory_pct()
            if free is None:
                continue
            free_min = free if free_min is None else min(free_min, free)
            if free < cfg.watch_free:
                msg = f"watchdog: only {free:.0f}% of memory free (< {cfg.watch_free:.0f}%): stopping the step"
                log.write(msg + "\n")
                log.flush()
                print(msg, file=sys.stderr, flush=True)
                return StepResult(_stop(proc, cfg.grace_s), killed=True, free_min=free_min)
    except KeyboardInterrupt:  # the step runs in its own session: Ctrl-C does not reach it
        return StepResult(_stop(proc, cfg.grace_s), interrupted=True, free_min=free_min)


def _stop(proc: subprocess.Popen, grace_s: float) -> int:
    """``kill_tree`` and wait, through further Ctrl-Cs: the step must not outlive the loop unwatched."""
    while True:
        try:
            kill_tree(proc, grace_s)
            return proc.wait()
        except KeyboardInterrupt:
            continue


def keep_awake() -> subprocess.Popen | None:
    """macOS: ``caffeinate`` keeps the Mac from sleeping while this process lives (like nba_mac_jobs.sh)."""
    exe = shutil.which("caffeinate")
    if not exe:
        return None
    try:
        return subprocess.Popen([exe, "-i", "-w", str(os.getpid())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return None


def _receipt(step: Step, res: StepResult, started: str, error: str | None = None) -> None:
    if step.log is None:
        return
    status = ("killed: low memory" if res.killed else "interrupted" if res.interrupted
              else "failed" if res.returncode != 0 else f"failed after it ran: {error}" if error else "ok")
    write_json(step.log.with_suffix(".json"), {"step": step.name, "stage": step.stage, "cmd": step.cmd,
                                               "started": started, "finished": utc_now(), "seconds": round(res.seconds, 1),
                                               "returncode": res.returncode, "status": status,
                                               "free_pct_start": res.free_start, "free_pct_min": res.free_min})


def _raise_interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


def run_loop(loop: Loop, *, game: str | None = None, scope: str = "all", candidate: str | None = None,
             max_steps: int | None = None, executor: Callable[[Step, Loop], StepResult] | None = None,
             out: Callable[[str], Any] = print) -> int:
    """Run the planned steps one at a time, re-planning after each. Exit codes: 0 done, 1 some step failed (the
    others went on), 3 refused (another job, too little memory), 4 stopped by the memory watchdog, 130 interrupted.
    ``max_steps`` counts steps with a command."""
    run = executor or execute
    refusal = guard(loop.cfg, memory=False)
    if refusal:
        out(refusal)
        return 3
    with loop.registry.locked() as reg:
        reg.bootstrap(loop, persist=True)
    if free_memory_pct() is None:
        out("warning: free memory cannot be read here (no memory_pressure, no /proc/meminfo): the memory guard is off")
    awake = keep_awake()
    # SIGTERM (even when inherited as ignored), or SIGHUP (the terminal closed) unless started under nohup: stop the
    # step, which runs in its own session and would otherwise go on without its memory watchdog. A handled signal
    # is reset to its default in the steps, so they can be stopped with SIGTERM too.
    old: dict[int, Any] = {}
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            if sig == signal.SIGTERM or signal.getsignal(sig) is not signal.SIG_IGN:
                old[sig] = signal.signal(sig, _raise_interrupt)
        except ValueError:  # not the main thread
            break
    failed: dict[str, int] = {}
    finished: set[str] = set()
    n = 0
    limited = False
    try:
        while True:
            loop.refresh()
            steps = plan(loop, game=game, scope=scope, candidate=candidate, skip=frozenset(failed))
            if not steps:
                break
            step = steps[0]
            if step.name in finished:
                failed[step.name] = -1
                out(f"{step.label()}: finished, yet still planned (its outputs are missing): skipped")
                continue
            if step.cmd is None:
                try:
                    apply_actions(loop, step.finish)
                except (OSError, ValueError) as exc:
                    failed[step.name] = -1
                    out(f"{step.label()}: {exc}")
                    continue
                finished.add(step.name)
                continue
            if max_steps is not None and n >= max_steps:
                limited = True
                break
            refusal = guard(loop.cfg)
            if refusal:
                out(refusal)
                return 3
            out(f"[{n + 1}] {step.label()}{' (heavy)' if step.heavy else ''}: {step.why}\n    $ {step.shell()}\n"
                f"    log: {loop.rel(step.log)}")  # type: ignore[arg-type]
            try:
                apply_actions(loop, step.prepare)
            except (OSError, ValueError) as exc:
                failed[step.name] = -1
                out(f"    could not prepare it: {exc}")
                continue
            started = utc_now()
            res = run(step, loop)
            error = None
            if res.returncode == 0 and not (res.killed or res.interrupted):
                try:
                    apply_actions(loop, step.finish)
                except (OSError, ValueError) as exc:
                    error = str(exc)
            _receipt(step, res, started, error)
            if res.interrupted:
                out("    interrupted: the step was stopped")
                return 130
            if res.killed:
                out(f"    stopped by the watchdog: less than {loop.cfg.watch_free:.0f}% of memory free")
                return 4
            if res.returncode != 0:
                failed[step.name] = res.returncode
                out(f"    FAILED (exit {res.returncode}) after {res.seconds:.0f}s: see {loop.rel(step.log)}")  # type: ignore[arg-type]
                continue
            if error is not None:
                failed[step.name] = -1
                out(f"    FAILED after it ran: {error}")
                continue
            finished.add(step.name)
            n += 1
            out(f"    ok in {res.seconds:.0f}s")
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
        if awake is not None:
            awake.terminate()
    if failed:
        out(f"{len(failed)} step(s) failed: " + ", ".join(sorted(failed)))
    if limited:
        out(f"stopped after {n} step(s); next: {step.label()} (nba_loop.py next)")
    elif not failed:
        out(f"{n} step(s) run; nothing else to do now" if n else "nothing to do")
    kinds = [scope] if scope in KINDS else list(KINDS) if scope == "all" and game is None else []
    for k in kinds:
        if k in loop.fit_notes and not limited:
            out(f"  {FIT_STAGE[k]}: {loop.fit_notes[k]}")
    return 1 if failed else 0


# --------------------------------------------------------------------------- status


CELL = {"done": "ok", "ready": "todo", "waiting": "wait", "failed": "FAIL"}
COLUMNS = (("espn", "espn"), ("scorebug", "scorebug"), ("camera", "camera"), ("teacher", "teacher"),
           ("ball_labels", "ball-lbl"), ("detect", "detect"), ("ball", "ball"), ("events", "events"),
           ("fuse", "fused"), ("eval_fused", "report"), ("cue_rows", "cue-rows"), ("pseudo", "pseudo"),
           ("hard_teacher", "hard"))


def _inventory(g: Game) -> dict[str, Any]:
    dets = {}
    for d in sorted(g.root.glob("detections*")):
        if d.is_dir() and not d.name.endswith(".partial"):
            dets[d.name] = {"student": (read_json(d / "loop.json") or {}).get("student"),
                            "files": len(list(d.glob("*.npz")))}
    tags = {}
    for t in sorted((g.root / "runs").glob("*")):
        if t.is_dir():
            rep = read_json(t / "report_fused.json") or {}
            tags[t.name] = {"events": events_complete(t), "fused": (t / "fused.jsonl").is_file(),
                            "report_events": (t / "report_events.json").is_file(),
                            "report_fused": (t / "report_fused.json").is_file(),
                            "cue_rows": (t / "cue_rows.jsonl").is_file(),
                            "hard_examples": (t / "hard_examples.json").is_file(),
                            "box_error": (rep.get("box") or {}).get("error")}
    parts = g.root / "camera_parts"
    manifest = read_json(parts / "loop.json")
    ball = read_json(g.root / "ball_track.loop.json") or {}
    return {"detections": dets, "tags": tags,
            "camera_parts": {"files": len(list(parts.glob("part*.jsonl"))),
                             "planned": len(manifest["parts"]) if isinstance(manifest, dict) else None},
            "ball_track": None if not (g.root / "ball_track.json").is_file() else (ball.get("student") or "unknown"),
            "pseudo": [d.name for d in sorted(g.root.glob("pseudo-s[0-9][0-9][0-9]")) if d.is_dir()],
            "hard": [d.name for d in sorted(g.root.glob("teacher-hard-s[0-9][0-9][0-9]")) if d.is_dir()],
            "training_folders": [d.name for d in training_folders(g)],
            "no_ball_labels": [d.name for d in [g.root / "teacher", *sorted(g.root.glob("teacher-hard-s[0-9][0-9][0-9]")),
                                                *sorted(g.root.glob("pseudo-s[0-9][0-9][0-9]"))]
                               if (d / "detections.jsonl").is_file() and not (d / "ball_labels.jsonl").is_file()]}


def status(loop: Loop) -> dict[str, Any]:
    """Every game's stages for the current set, its files, the registry and the leaderboard (``status --json``)."""
    cur = loop.current_set()
    games = []
    for g in loop.games:
        entry: dict[str, Any] = {"game": g.name, "game_id": g.game_id, "primary": g.primary,
                                 "variant_of": g.variant_of, "excluded": g.excluded, "error": g.error}
        if g.primary:
            gp = GamePlanner(loop, g, cur)
            gp.plan(STATUS_STAGES)
            failures = {}
            for s in gp.steps:
                rec = read_json(s.log.with_suffix(".json")) if s.log else None
                if rec and rec.get("status") != "ok":
                    gp.state[s.stage] = f"failed ({rec.get('status')}, exit {rec.get('returncode')})"
                    failures[s.stage] = loop.rel(s.log)  # type: ignore[arg-type]
            entry["stages"] = dict(gp.state)
            entry["failures"] = failures
        entry["inventory"] = _inventory(g)
        games.append(entry)
    rows = loop.latest_rows()
    by_set: dict[str, list[float]] = {}
    for (_, key), r in rows.items():
        if r.get("box_error") is not None:
            by_set.setdefault(key, []).append(float(r["box_error"]))
    reg = loop.registry
    return {"runs": str(loop.runs), "current": dict(reg.data["current"]), "current_set": cur.key, "tag": cur.tag,
            "games": games,
            "versions": {k: {v: {"source": e.get("source"), "fit_games": e.get("fit_games"),
                                 "holdout_games": e.get("holdout_games"), "games_seen": e.get("games_seen"),
                                 "created": e.get("created"), "path": e.get("path")}
                             for v, e in reg.versions(k).items()} for k in KINDS},
            "candidates": {k: open_candidates(loop, k) for k in KINDS},
            "decisions": reg.data["decisions"][-5:],
            "leaderboard": {"rows": len(loop.board()),
                            "mean_box_error": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_set.items())},
                            "games": {k: len(v) for k, v in sorted(by_set.items())}}}


def _cell(state: str) -> str:
    if state.startswith("blocked"):
        return "BLOCK"
    if state.startswith("failed"):
        return "FAIL"
    return CELL.get(state, state)


def format_status(loop: Loop) -> str:
    st = status(loop)
    cur = loop.current_set()
    reg = loop.registry
    lines = [f"CourtVision NBA loop: {loop.rel(loop.runs)} (research use only)", ""]
    for k in KINDS:
        vid = reg.current_id(k)
        e = reg.get(vid) or {}
        cands = st["candidates"][k]
        lines.append(f"{k:8s} current {vid or '-'} ({e.get('source', 'none')})"
                     + (f"; candidates {', '.join(cands)}" if cands else ""))
    lines += [f"current set {cur.key}: tag {cur.tag}" if cur.student else
              "no student yet: detection waits for one (train-student, or runs/models/student/best.pt to import)", ""]
    width = max([len(g["game"]) for g in st["games"]] + [4])
    cols = [max(len(h), 5) for _, h in COLUMNS]
    lines.append(f"{'game':{width}s}  " + " ".join(f"{h:{w}s}" for (_, h), w in zip(COLUMNS, cols, strict=True))
                 + " box err")
    notes: list[str] = []
    for g in st["games"]:
        if not g["primary"]:
            if g["error"]:
                why = g["error"]
            elif g["excluded"]:
                why = 'left out ("loop": false in game.json)'
            else:
                why = f"variant of {g['variant_of']} (left alone unless named with --game)"
            lines.append(f"{g['game']:{width}s}  {why}")
            continue
        stages = g["stages"]
        tag = g["inventory"]["tags"].get(cur.tag) or {}
        err = tag.get("box_error")
        lines.append(f"{g['game']:{width}s}  " + " ".join(f"{_cell(stages.get(s, '-')):{w}s}"
                                                         for (s, _), w in zip(COLUMNS, cols, strict=True))
                     + (f" {err:.3f}" if isinstance(err, (int, float)) else " -"))
        inv = g["inventory"]
        notes.append(f"{g['game']} (game id {g['game_id']}):")
        for s, state in stages.items():
            if state.startswith(("blocked", "failed")):
                notes.append(f"  {s}: {state}" + (f" - see {g['failures'][s]}" if s in g["failures"] else ""))
        dets = ", ".join(f"{k} ({v['student'] or 'student unknown'}, {v['files']} files)" for k, v in inv["detections"].items())
        notes.append(f"  detections: {dets or '-'}; ball track: {inv['ball_track'] or '-'}; "
                     f"training folders: {', '.join(inv['training_folders']) or '-'}")
        notes.append(f"  pseudo labels: {', '.join(inv['pseudo']) or '-'}; hard examples: {', '.join(inv['hard']) or '-'}")
        if inv["no_ball_labels"]:
            notes.append(f"  left out of train-student (no ball_labels.jsonl): {', '.join(inv['no_ball_labels'])}")
        tags = []
        for t, v in inv["tags"].items():
            have = [k for k in ("events", "fused", "report_events", "report_fused", "cue_rows", "hard_examples") if v[k]]
            box = f" box {v['box_error']:.3f}" if isinstance(v["box_error"], (int, float)) else ""
            tags.append(f"{t} [{' '.join(have) or 'empty'}{box}]")
        notes.append(f"  tags: {', '.join(tags) or '-'}")
    lines += ["", *notes, ""]
    lb = st["leaderboard"]
    lines.append(f"leaderboard {loop.rel(loop.leaderboard_path)}: {lb['rows']} rows")
    for k, v in lb["mean_box_error"].items():
        lines.append(f"  {k}: mean box error {v:.4f} over {lb['games'][k]} game(s) (latest rows)")
    for k in KINDS:
        for vid in st["candidates"][k]:
            d = assess(loop, vid)
            lines.append(f"  decide {vid}: would be {d['outcome']}: {d['reason']}")
    for d in st["decisions"]:
        lines.append(f"  decision {d.get('time')}: {d.get('candidate')} {d.get('outcome')}"
                     f"{' (forced)' if d.get('forced') else ''}: {d.get('reason')}")
    return "\n".join(lines)


def format_next(loop: Loop, **kw: Any) -> str:
    step, steps = next_step(loop, **kw)
    lines = []
    pending = [e for e in loop.registry.versions("student").values() if e.get("import_from")]
    for e in pending:
        lines.append(f"first, run imports {e['import_from']} as student {e['id']} ({e['path']})")
    book = [s for s in steps[: steps.index(step)] if s.cmd is None] if step else [s for s in steps if s.cmd is None]
    for s in book:
        lines.append(f"bookkeeping first: {s.label()}: " + "; ".join(describe(a, loop) for a in s.finish))
    if step is None:
        lines.append("nothing to run now")
        lines += [f"  {FIT_STAGE[k]}: {loop.fit_notes[k]}" for k in KINDS if k in loop.fit_notes]
        for k in KINDS:
            for vid in open_candidates(loop, k):
                d = assess(loop, vid)
                lines.append(f"  {vid}: promotion would be {d['outcome']}: {d['reason']}"
                             + (f" -> nba_loop.py promote {vid}" if d["outcome"] == "promoted" else ""))
        return "\n".join(lines)
    n_cmd = [s for s in steps if s.cmd is not None]
    lines += [f"next: {step.label()} ({'heavy' if step.heavy else 'light'})", f"why: {step.why}",
              f"command (from {loop.repo}):", f"  {step.shell()}", f"log: {loop.rel(step.log)}"]  # type: ignore[arg-type]
    for a in step.prepare:
        lines.append(f"before: {describe(a, loop)}")
    for a in step.finish:
        lines.append(f"after it succeeds: {describe(a, loop)}")
    heavy = [s for s in n_cmd if s.heavy]
    if not step.heavy and heavy:
        lines.append(f"next heavy step ready now: {heavy[0].label()}: {heavy[0].shell()}")
    lines.append(f"({len(n_cmd)} step(s) ready now, {len(heavy)} heavy; later stages appear as earlier ones finish. "
                 f"nba_loop.py run --max-steps 1 runs it with the memory guard)")
    return "\n".join(lines)
