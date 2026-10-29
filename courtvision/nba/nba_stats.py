"""NBA.com data as extra ground truth: shot locations (play-by-play) and tracking totals (passes, distance).

``nba_pbp.csv`` is a compact export of ``cdn.nba.com/static/json/liveData/playbyplay/playbyplay_<gameId>.json``
(columns ``n, period, clock, team, person, type, sub, res, dist, x, y, side, poss, sh, sa, qual``; team / poss as
L / G for away / home here, ``x`` / ``y`` in percent of the court's length / width, ``side`` the basket). The NBA's
court frame is turned half a turn against the main broadcast camera's (the scorer's table side): a shot at
NBA (x %, y %) is at court ft ``(94 * (1 - x / 100), 50 * (1 - y / 100))`` in CourtVision's frame, which is the
camera's (x to the right, y away from it). ``nba_tracking.json`` holds the tracking box score (passes, touches,
distance per player and team).

Both are NBA statistics: research use only (docs/TRAINING.md §1).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from courtvision.types import Team

__all__ = ["NBAShot", "load_shots", "load_tracking", "match_shot_locations"]


@dataclass(frozen=True)
class NBAShot:
    period: int
    clock: float
    team: Team
    value: int
    made: bool
    x: float  # court ft, CourtVision (camera) frame
    y: float
    dist_ft: float


def load_shots(path: Path, home_code: str = "G", away_code: str = "L", length: float = 94.0,
               width: float = 50.0) -> list[NBAShot]:
    out = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r["type"] not in ("2", "3") or not r["x"]:
                continue
            team = Team.HOME if r["team"] == home_code else Team.AWAY if r["team"] == away_code else None
            if team is None:
                continue
            x, y = float(r["x"]), float(r["y"])
            out.append(NBAShot(int(r["period"]), float(r["clock"]), team, int(r["type"]), r["res"] == "M",
                               length * (1.0 - x / 100.0), width * (1.0 - y / 100.0), float(r["dist"] or 0.0)))
    return out


def load_tracking(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def match_shot_locations(pred: list[Any], truth: list[tuple[float, NBAShot]], tol_s: float = 2.5) -> dict[str, Any]:
    """Predicted field-goal results with a position (``Event`` with x / y) against NBA shots placed on video time
    (``(t, NBAShot)``): matched by team within ``tol_s``; the position error and the 2 / 3 agreement."""
    used: set[int] = set()
    errs: list[float] = []
    agree = n = 0
    for e in sorted(pred, key=lambda e: e.t):
        if e.x is None or e.y is None or e.team is None:
            continue
        best = None
        for k, (t, s) in enumerate(truth):
            if k in used or s.team is not e.team or abs(t - e.t) > tol_s:
                continue
            if best is None or abs(t - e.t) < abs(truth[best][0] - e.t):
                best = k
        if best is None:
            continue
        used.add(best)
        s = truth[best][1]
        errs.append(math.hypot(e.x - s.x, e.y - s.y))
        n += 1
        agree += int((e.value == 3) == (s.value == 3))
    errs.sort()
    return {"matched": n, "median_err_ft": round(errs[len(errs) // 2], 2) if errs else None,
            "p90_err_ft": round(errs[int(0.9 * (len(errs) - 1))], 2) if errs else None,
            "value_agreement": round(agree / n, 3) if n else None}
