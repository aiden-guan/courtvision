"""A game's analysis for watching: the possession timeline and the decided events of a run
(``tools/nba_analyze.py <game> timeline``), found from the video it was made from.

The Studio draws these on its feed (``courtvision/server/studio.py``, ``static/possession.js``). Everything shown comes
from the video except the run's ``reference``: the play-by-play, placed on video time through the scorebug clock and
shown beside the analysis to judge it by (docs/research/possession_decision_model.md).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from courtvision.nba import espn
from courtvision.nba.timeline import from_scorebug
from courtvision.types import Event, EventKind

K = EventKind
SHOWN = (K.SHOT_MADE, K.SHOT_MISSED, K.POSSESSION, K.REBOUND, K.STEAL, K.TURNOVER)
PROBE = 1 << 20  # bytes read from each end of a video to tell copies of it apart
_prints: dict[tuple[str, int, int], str] = {}


def fingerprint(path: Path) -> tuple[int, str]:
    """(size, digest of the first and last MiB): the same for every copy of a video (the Studio keeps one per upload)."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _prints:
        h = hashlib.sha1()
        with path.open("rb") as f:
            h.update(f.read(PROBE))
            f.seek(max(0, st.st_size - PROBE))
            h.update(f.read(PROBE))
        _prints[key] = h.hexdigest()
    return st.st_size, _prints[key]


def same_video(a: Path, b: Path) -> bool:
    try:
        if a.resolve() == b.resolve():
            return True
        return a.stat().st_size == b.stat().st_size and fingerprint(a) == fingerprint(b)
    except OSError:
        return False


def find_run(video: Path, nba: Path, repo: Path) -> tuple[Path, str, float] | None:
    """The newest run with a timeline (``<game>/runs/<tag>/timeline.json`` and ``.jsonl``) of the game folders under
    ``nba`` whose ``game.json`` names ``video`` or a copy of it (a relative path is from ``repo``): (game folder, run
    tag, the timeline's modification time)."""
    best: tuple[float, Path] | None = None
    for meta in sorted(nba.glob("*/game.json")):
        runs = [(p.stat().st_mtime, p) for p in meta.parent.glob("runs/*/timeline.json")
                if p.with_suffix(".jsonl").is_file()]
        if not runs or (best is not None and max(runs)[0] <= best[0]):
            continue
        try:
            source = Path(str(json.loads(meta.read_text())["video"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if same_video(video, source if source.is_absolute() else repo / source):
            best = max(runs)
    return (best[1].parents[2], best[1].parent.name, best[0]) if best is not None else None


def teams(game: Path) -> dict[str, dict[str, str]] | None:
    """{"home" | "away": {"abbr", "name"}} from the game's ESPN summary, when it has one."""
    path = game / "espn_summary.json"
    if not path.is_file():
        return None
    return {v["side"].value: {"abbr": v["abbr"], "name": v["name"]}
            for v in espn.teams(espn.load_summary(path)).values()}


def shown(e: Event) -> dict[str, Any]:
    """An event as the feed shows it: a shot's value, whether it was a free throw, whether it was found at a rim visit
    the engine reported nothing at, and P(made); a change of possession's cause; whether a rebound was offensive."""
    d: dict[str, Any] = {"t": round(e.t, 3), "kind": e.kind.value, "team": e.team.value if e.team else None}
    if e.kind in (K.SHOT_MADE, K.SHOT_MISSED):
        makes = e.attrs.get("makes")
        p = makes.get("p_made") if isinstance(makes, dict) else None
        d.update(value=int(e.value), free_throw=e.value == 1 or e.attrs.get("shot_type") == "free_throw",
                 found_at_rim=e.attrs.get("inferred") == "rim_visit", p_made=None if p is None else round(float(p), 3))
    elif e.kind is K.POSSESSION:
        d["cause"] = e.attrs.get("cause")
    elif e.kind is K.REBOUND:
        d["offensive"] = bool(e.attrs.get("offensive"))
    return d


def load(game: Path, tag: str) -> dict[str, Any]:
    """A run's possession timeline and decided events, as the Studio draws them (JSON-ready). ``measures`` are the
    run's held-out accuracies against the play-by-play, when it had one."""
    run = game / "runs" / tag
    doc = json.loads((run / "timeline.json").read_text())
    with (run / "timeline.jsonl").open() as fh:
        events = [Event.from_dict(json.loads(line)) for line in fh if line.strip()]
    shots = doc.get("shots") or {}
    measures = {"possession": (doc.get("timeline") or {}).get("accuracy"), "real": shots.get("real_accuracy"),
                "free_throw": shots.get("ft_accuracy"), "made": shots.get("made_accuracy")}
    return {"game": game.name, "tag": tag, "teams": teams(game), "dt_s": doc.get("dt_s"),
            "segments": doc["segments"], "back_court": doc.get("back_court", []),
            "events": sorted((shown(e) for e in events
                              if e.kind in SHOWN and e.team is not None and not from_scorebug(e)), key=lambda d: d["t"]),
            "reference": doc.get("reference"), "jev": bool(shots.get("jev")),
            "measures": {k: v for k, v in measures.items() if v is not None}}
