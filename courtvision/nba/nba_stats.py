"""NBA.com data as extra ground truth: shot locations (play-by-play) and tracking totals (passes, distance).

``nba_pbp.csv`` is a compact export of ``cdn.nba.com/static/json/liveData/playbyplay/playbyplay_<gameId>.json``
(columns ``n, period, clock, team, person, type, sub, res, dist, x, y, side, poss, sh, sa, qual``; team / poss as
L / G for away / home here, ``x`` / ``y`` in percent of the court's length / width, ``side`` the basket). The NBA's
court frame is turned half a turn against the main broadcast camera's (the scorer's table side): a shot at
NBA (x %, y %) is at court ft ``(94 * (1 - x / 100), 50 * (1 - y / 100))`` in CourtVision's frame, which is the
camera's (x to the right, y away from it). ``nba_tracking.json`` holds the tracking box score (passes, touches,
distance per player and team).

Export (tools/nba_new_game.py): ``fetch_pbp`` downloads the live play-by-play of an NBA.com game id and
``export_pbp_csv`` writes ``nba_pbp.csv`` from it (``CSV_COLUMNS``; every action in the feed's order; ``team`` /
``poss`` as NBA.com tricodes, ``type`` 2 / 3 for field goals, else the feed's action type, ``res`` M / X, ``clock``
seconds left in the period, ``sh`` / ``sa`` the running home / away score). The game id comes from the season
schedule (``fetch_schedule``, cdn.nba.com: the current season) or a stats.nba.com scoreboard of the day
(``fetch_scoreboard``): ``find_game_id`` reads any of them. ``load_shots`` without codes takes the first export's
``G`` / ``L`` when the file has them, else the codes its running score gives (``csv_team_codes``).

Both are NBA statistics: research use only (docs/TRAINING.md §1).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from courtvision.nba.espn import get_json
from courtvision.training.align_pbp import PBPAction, infer_team_map, parse_clock
from courtvision.types import EventKind, Team

__all__ = ["CSV_COLUMNS", "NBAShot", "csv_team_codes", "export_pbp_csv", "fetch_pbp", "fetch_schedule",
           "fetch_scoreboard", "find_game_id", "load_shots", "load_tracking", "match_shot_locations", "pbp_csv_rows",
           "schedule_games"]

CSV_COLUMNS = ("n", "period", "clock", "team", "person", "type", "sub", "res", "dist", "x", "y", "side", "poss", "sh",
               "sa", "qual")
LEGACY_CODES = ("G", "L")  # (home, away) team codes of the first export (Lakers @ Warriors, 2024-12-25)
PBP_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
SCOREBOARD_URLS = {3: "https://stats.nba.com/stats/scoreboardv3?GameDate={date}&LeagueID=00",
                   2: "https://stats.nba.com/stats/scoreboardv2?GameDate={date}&LeagueID=00&DayOffset=0"}
# Browser-like request headers: stats.nba.com is known to stall or refuse requests without them and its two
# x-nba-stats headers (not verifiable from here: no network to NBA.com in this repo's CI)
NBA_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) "
                             "Version/17.5 Safari/605.1.15",
               "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9",
               "Referer": "https://www.nba.com/", "Origin": "https://www.nba.com"}
STATS_HEADERS = {**NBA_HEADERS, "x-nba-stats-origin": "stats", "x-nba-stats-token": "true"}
GAME_ID_RE = re.compile(r"^\d{10}$")


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


def load_shots(path: str | Path, home_code: str | None = None, away_code: str | None = None, length: float = 94.0,
               width: float = 50.0) -> list[NBAShot]:
    """The field-goal attempts of an ``nba_pbp.csv`` that have a location (``type`` 2 / 3 and an ``x``). ``home_code``
    / ``away_code``: the file's codes of the two teams; one left None is the first export's (``G`` home, ``L`` away)
    when the file uses those, else the one its running score gives (``csv_team_codes``)."""
    with Path(path).open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if home_code is None or away_code is None:
        home_code, away_code = _default_codes(rows, home_code, away_code)
    out = []
    for r in rows:
        if r["type"] not in ("2", "3") or not r["x"]:
            continue
        team = Team.HOME if r["team"] == home_code else Team.AWAY if r["team"] == away_code else None
        if team is None:
            continue
        x, y = float(r["x"]), float(r["y"])
        out.append(NBAShot(int(r["period"]), float(r["clock"]), team, int(r["type"]), r["res"] == "M",
                           length * (1.0 - x / 100.0), width * (1.0 - y / 100.0), float(r["dist"] or 0.0)))
    return out


def _default_codes(rows: list[dict[str, str]], home: str | None, away: str | None) -> tuple[str, str]:
    present = {(r.get("team") or "").strip() for r in rows}
    if not set(LEGACY_CODES) <= present:
        side = {t: c for c, t in csv_team_codes(rows).items()}
        home = home if home is not None else side.get(Team.HOME)
        away = away if away is not None else side.get(Team.AWAY)
    return (home if home is not None else LEGACY_CODES[0], away if away is not None else LEGACY_CODES[1])


def csv_team_codes(rows: Iterable[Mapping[str, Any]]) -> dict[str, Team]:
    """Team code -> Team of ``nba_pbp.csv`` rows from the running score (``sh`` / ``sa``): the side whose score moves on
    a team's made shots (``align_pbp.infer_team_map``). {} unless exactly two codes come out on different sides."""
    acts = []
    for k, r in enumerate(rows):
        try:
            period, home, away = int(r["period"]), int(r["sh"]), int(r["sa"])
        except (KeyError, TypeError, ValueError):
            continue
        typ, code = str(r.get("type") or "").strip().lower(), (r.get("team") or "").strip()
        value = int(typ) if typ in ("2", "3") else 1 if typ in ("freethrow", "1") else 0
        made = r.get("res") == "M" if value and r.get("res") in ("M", "X") else None
        try:
            number = int(r.get("n") or k)
        except ValueError:
            number = k
        acts.append(PBPAction(number, period, 0.0, EventKind.SHOT_MADE if made else None, value=value, made=made,
                              team_tricode=code or None, score_home=home, score_away=away))
    codes = infer_team_map(acts)
    return codes if len(codes) == 2 and len(set(codes.values())) == 2 else {}


# --------------------------------------------------------------------------- NBA.com: game ids, play-by-play, export


def fetch_schedule(timeout: float = 60.0) -> dict[str, Any]:
    """The NBA.com season schedule (current season only): ``leagueSchedule.gameDates[].games[]``."""
    return get_json(SCHEDULE_URL, timeout, NBA_HEADERS)


def fetch_scoreboard(date: str | dt.date, version: int = 3, timeout: float = 20.0) -> dict[str, Any]:
    """A stats.nba.com scoreboard of one day (any season): v3 ``scoreboard.games[]``, v2 ``resultSets`` GameHeader."""
    day = date if isinstance(date, dt.date) else dt.date.fromisoformat(str(date))
    return get_json(SCOREBOARD_URLS[version].format(date=day.isoformat()), timeout, STATS_HEADERS)


def fetch_pbp(game_id: str, out: str | Path | None = None, timeout: float = 30.0) -> dict[str, Any]:
    """The live play-by-play of an NBA.com game (``game.actions``; ``game_id`` the 10-digit id, e.g. 0022400447)."""
    if not GAME_ID_RE.match(str(game_id)):
        raise ValueError(f"not an NBA.com game id: {game_id!r} (10 digits)")
    data = get_json(PBP_URL.format(game_id=game_id), timeout, NBA_HEADERS)
    if out is not None:
        Path(out).write_text(json.dumps(data))
    return data


def _ymd(text: Any) -> str:
    """YYYY-MM-DD of "2024-12-25...", "20241225..." or "12/25/2024 ..."; "" when none."""
    s = str(text or "").strip()
    m = re.match(r"^(\d{4})-?(\d\d)-?(\d\d)", s)
    if m:
        return f"{m[1]}-{m[2]}-{m[3]}"
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})", s)
    return f"{m[3]}-{int(m[1]):02d}-{int(m[2]):02d}" if m else ""


def _live_game(g: Mapping[str, Any], day: str) -> dict[str, str]:
    code = re.match(r"^(\d{8})/([A-Z]{3})([A-Z]{3})$", str(g.get("gameCode") or ""))  # "20241225/LALGSW": away, home
    return {"game_id": str(g.get("gameId") or ""),
            "date": _ymd(code[1]) if code else _ymd(g.get("gameDateEst")) or day,
            "away": str((g.get("awayTeam") or {}).get("teamTricode") or (code[2] if code else "")),
            "home": str((g.get("homeTeam") or {}).get("teamTricode") or (code[3] if code else ""))}


def schedule_games(doc: Mapping[str, Any]) -> list[dict[str, str]]:
    """The games of an NBA.com document as ``{"game_id", "date" (YYYY-MM-DD, the US date), "away", "home"}``
    (tricodes): the season schedule (``leagueSchedule.gameDates[].games[]``), a scoreboard (``scoreboard.games[]``:
    stats.nba.com v3, the live feed) or the ``resultSets`` GameHeader of scoreboard v2 (``GAMECODE``)."""
    out: list[dict[str, str]] = []
    league = doc.get("leagueSchedule")
    if isinstance(league, Mapping):
        for gd in league.get("gameDates") or []:
            for g in gd.get("games") or []:
                out.append(_live_game(g, _ymd(gd.get("gameDate"))))
    board = doc.get("scoreboard")
    if isinstance(board, Mapping):
        for g in board.get("games") or []:
            out.append(_live_game(g, _ymd(board.get("gameDate"))))
    for rs in doc.get("resultSets") or []:
        if rs.get("name") != "GameHeader":
            continue
        cols = list(rs.get("headers") or [])
        for row in rs.get("rowSet") or []:
            r = dict(zip(cols, row, strict=False))
            code = re.match(r"^(\d{8})/([A-Z]{3})([A-Z]{3})$", str(r.get("GAMECODE") or ""))
            out.append({"game_id": str(r.get("GAME_ID") or ""),
                        "date": _ymd(code[1]) if code else _ymd(r.get("GAME_DATE_EST")),
                        "away": code[2] if code else "", "home": code[3] if code else ""})
    return [g for g in out if g["game_id"]]


def find_game_id(doc: Mapping[str, Any], date: str, teams: Iterable[str]) -> str | None:
    """The NBA.com game id of the game between two teams (tricodes, either order) on ``date`` (YYYY-MM-DD)."""
    want = set(teams)
    hits = {g["game_id"] for g in schedule_games(doc) if g["date"] == date and {g["away"], g["home"]} == want}
    return hits.pop() if len(hits) == 1 else None


def _cell(v: Any) -> Any:
    return "" if v is None else v


def pbp_csv_rows(pbp: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The ``nba_pbp.csv`` rows (``CSV_COLUMNS``) of a live play-by-play document (``fetch_pbp``): every action in the
    feed's order. Field goals (``actionType`` 2pt / 3pt) get ``type`` 2 / 3, other actions their type; x / y / side /
    shot distance as the feed gives them; ``poss`` is the tricode of the ``possession`` team id."""
    body = pbp.get("game")
    game: Mapping[str, Any] = body if isinstance(body, Mapping) else pbp
    actions = [a for a in game.get("actions") or [] if isinstance(a, Mapping)]
    tri = {str(a.get("teamId")): str(a["teamTricode"]) for a in actions if a.get("teamId") and a.get("teamTricode")}
    rows = []
    for a in actions:
        typ = str(a.get("actionType") or "")
        try:
            clock: Any = parse_clock(a.get("clock"))
        except ValueError:
            clock = ""
        rows.append({"n": _cell(a.get("actionNumber")), "period": _cell(a.get("period")), "clock": clock,
                     "team": a.get("teamTricode") or tri.get(str(a.get("teamId")), ""),
                     "person": a.get("personId") or "", "type": {"2pt": "2", "3pt": "3"}.get(typ.lower(), typ),
                     "sub": a.get("subType") or "",
                     "res": {"made": "M", "missed": "X"}.get(str(a.get("shotResult") or "").strip().lower(), ""),
                     "dist": _cell(a.get("shotDistance")), "x": _cell(a.get("x")), "y": _cell(a.get("y")),
                     "side": a.get("side") or "", "poss": tri.get(str(a.get("possession")), ""),
                     "sh": _cell(a.get("scoreHome")), "sa": _cell(a.get("scoreAway")),
                     "qual": ";".join(str(q) for q in a.get("qualifiers") or [])})
    return rows


def export_pbp_csv(pbp: Mapping[str, Any], out: str | Path) -> list[dict[str, Any]]:
    """Write ``nba_pbp.csv`` (``pbp_csv_rows``) atomically: a temporary file renamed over ``out``. Returns the rows."""
    rows = pbp_csv_rows(pbp)
    out = Path(out)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)
    return rows


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
