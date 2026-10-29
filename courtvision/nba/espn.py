"""ESPN game summaries as ground truth: play-by-play, box score and roster.

``site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=<id>`` returns one JSON document per game
with ``plays`` (period, clock, type, text, participants, scores, shot value) and ``boxscore`` (team and player
totals). ``fetch_summary`` downloads it; the parsers turn it into CourtVision's types:

* ``pbp_actions``: ``PBPAction`` rows in the generic schema of ``training/align_pbp.py``. A shot row carries its
  result; its assist, block and steal are separate child rows (``parent_action``), as in the NBA feeds. ESPN
  writes a blocked shot as one row whose second participant is the blocker, and a steal inside the turnover row.
  Team rebounds (no participant) are kept as team-level rebounds.
* ``box_score``: team and player counting stats (``TeamBox`` / ``PlayerBox``).
* ``roster``: a ``Roster`` with ESPN athlete ids, names and jersey numbers.

Finding a game (tools/nba_new_game.py): ``fetch_scoreboard`` downloads the scoreboard of a day
(``.../scoreboard?dates=YYYYMMDD``, the games ESPN lists on that US date) and ``scoreboard_games`` /
``summary_game`` reduce a scoreboard or a summary to its games (event id, US Eastern date, teams, status, national
TV). ``team_code`` turns any spelling of a team (NBA.com tricode, ESPN abbreviation, an alias such as GS / NY / SA /
NO / UTAH / WSH / BRK / PHO, or a name) into its NBA.com tricode, the code of the game folder names.

ESPN data is NBA statistics: research use only (docs/TRAINING.md §1), never shipped in a product.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import re
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from courtvision.training.align_pbp import PBPAction, parse_pbp_generic
from courtvision.types import Team

__all__ = ["TEAMS", "PlayerBox", "TeamBox", "box_score", "fetch_scoreboard", "fetch_summary", "find_games",
           "get_json", "load_summary", "parse_matchup", "pbp_actions", "roster", "scoreboard_games", "summary_game",
           "team_code", "teams", "us_date"]

SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event}"
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date}"
STATS = ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf")


def fetch_summary(event_id: str, out: str | Path | None = None, timeout: float = 30.0) -> dict[str, Any]:
    with urllib.request.urlopen(SUMMARY_URL.format(event=event_id), timeout=timeout) as resp:  # noqa: S310 (fixed host)
        data = json.loads(resp.read())
    if out is not None:
        Path(out).write_text(json.dumps(data))
    return data


def load_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def get_json(url: str, timeout: float = 30.0, headers: Mapping[str, str] | None = None) -> Any:
    """A JSON document over HTTPS (urllib; a gzip body is unpacked). Network errors propagate
    (``urllib.error.URLError`` / ``HTTPError``, ``TimeoutError``); a body that is not JSON raises ValueError."""
    req = urllib.request.Request(url, headers=dict(headers or {}))
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # fixed hosts (ESPN, NBA.com)
        body = resp.read()
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    return json.loads(body)


def fetch_scoreboard(date: str | dt.date, out: str | Path | None = None, timeout: float = 30.0) -> dict[str, Any]:
    """ESPN's NBA scoreboard of one day (``date``: YYYY-MM-DD or a date): the games it lists on that US date."""
    day = date if isinstance(date, dt.date) else dt.date.fromisoformat(str(date))
    data = get_json(SCOREBOARD_URL.format(date=day.strftime("%Y%m%d")), timeout)
    if out is not None:
        Path(out).write_text(json.dumps(data))
    return data


# --------------------------------------------------------------------------- teams

# NBA.com tricode -> (ESPN abbreviation, city, nickname, other names). The tricode names the game folders
# (``runs/nba/2024-12-25-LAL-GSW``); ESPN writes GS, NY, NO, SA, UTAH and WSH, other sources BRK, PHO, CHO, NOH, NJN.
TEAMS: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "ATL": ("ATL", "Atlanta", "Hawks", ()),
    "BKN": ("BKN", "Brooklyn", "Nets", ("BRK", "NJN", "New Jersey Nets")),
    "BOS": ("BOS", "Boston", "Celtics", ()),
    "CHA": ("CHA", "Charlotte", "Hornets", ("CHO", "Charlotte Bobcats", "Bobcats")),
    "CHI": ("CHI", "Chicago", "Bulls", ()),
    "CLE": ("CLE", "Cleveland", "Cavaliers", ("Cavs",)),
    "DAL": ("DAL", "Dallas", "Mavericks", ("Mavs",)),
    "DEN": ("DEN", "Denver", "Nuggets", ()),
    "DET": ("DET", "Detroit", "Pistons", ()),
    "GSW": ("GS", "Golden State", "Warriors", ()),
    "HOU": ("HOU", "Houston", "Rockets", ()),
    "IND": ("IND", "Indiana", "Pacers", ()),
    "LAC": ("LAC", "LA", "Clippers", ("Los Angeles Clippers",)),
    "LAL": ("LAL", "Los Angeles", "Lakers", ("LA Lakers",)),
    "MEM": ("MEM", "Memphis", "Grizzlies", ()),
    "MIA": ("MIA", "Miami", "Heat", ()),
    "MIL": ("MIL", "Milwaukee", "Bucks", ()),
    "MIN": ("MIN", "Minnesota", "Timberwolves", ("Wolves",)),
    "NOP": ("NO", "New Orleans", "Pelicans", ("NOH", "NOK")),
    "NYK": ("NY", "New York", "Knicks", ()),
    "OKC": ("OKC", "Oklahoma City", "Thunder", ()),
    "ORL": ("ORL", "Orlando", "Magic", ()),
    "PHI": ("PHI", "Philadelphia", "76ers", ("Sixers",)),
    "PHX": ("PHX", "Phoenix", "Suns", ("PHO",)),
    "POR": ("POR", "Portland", "Trail Blazers", ("Blazers",)),
    "SAC": ("SAC", "Sacramento", "Kings", ()),
    "SAS": ("SA", "San Antonio", "Spurs", ()),
    "TOR": ("TOR", "Toronto", "Raptors", ()),
    "UTA": ("UTAH", "Utah", "Jazz", ()),
    "WAS": ("WSH", "Washington", "Wizards", ()),
}


def _team_key(text: str) -> str:
    """Lower case, dots dropped (L.A.), other punctuation as spaces."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower().replace(".", "")).split())


def _team_index() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(codes, names): key -> tricodes. Codes are tricodes, ESPN abbreviations and short upper-case aliases; names
    are cities, nicknames, full names and the other aliases. "LA" / "Los Angeles" alone stay ambiguous."""
    codes: dict[str, set[str]] = {}
    names: dict[str, set[str]] = {}
    for tri, (abbr, city, nick, other) in TEAMS.items():
        for c in (tri, abbr, *(o for o in other if o.isupper() and len(o) <= 4)):
            codes.setdefault(_team_key(c), set()).add(tri)
        for n in (city, nick, f"{city} {nick}", *(o for o in other if not (o.isupper() and len(o) <= 4))):
            names.setdefault(_team_key(n), set()).add(tri)
    for both in ("la", "los angeles"):
        names[both] = {"LAL", "LAC"}
    return codes, names


TEAM_CODES, TEAM_NAMES = _team_index()


def team_code(text: str) -> str:
    """The NBA.com tricode of a team written as its tricode, ESPN abbreviation, an alias (GS, NY, SA, NO, UTAH, WSH,
    BRK, PHO, ...), its full name, nickname or city, in any case. ValueError when unknown or ambiguous."""
    key = _team_key(text)
    hits = TEAM_CODES.get(key) or TEAM_NAMES.get(key)
    if not hits:
        raise ValueError(f"unknown NBA team {text!r} (an NBA.com or ESPN abbreviation such as GSW or GS, or a name)")
    if len(hits) > 1:
        raise ValueError(f"{text!r} could be {' or '.join(sorted(hits))}: say which")
    return next(iter(hits))


def parse_matchup(text: str) -> tuple[str, str]:
    """Two teams as written, as NBA.com tricodes in the order given: ``AWY,HOM``, ``AWY@HOM``, ``AWY at HOM``,
    ``A vs B``, ``A/B``, ``AWY-HOM`` or two names separated by spaces (``Lakers Warriors``, ``LAL GSW``)."""
    s = text.strip()
    parts: list[str] | None = None
    for sep in (r"@", r",", r"\s+(?:at|vs\.?|v\.?|versus)\s+", r"/"):
        split = [p.strip() for p in re.split(sep, s, flags=re.IGNORECASE)]
        if len(split) == 2 and all(split):
            parts = split
            break
    if parts is not None:
        pair = (team_code(parts[0]), team_code(parts[1]))
    else:
        options: list[tuple[str, str]] = []
        dash = [p.strip() for p in s.split("-")]
        if len(dash) == 2 and all(dash):
            try:
                options.append((team_code(dash[0]), team_code(dash[1])))
            except ValueError:
                pass
        words = s.split()
        for i in range(1, len(words)):
            try:
                options.append((team_code(" ".join(words[:i])), team_code(" ".join(words[i:]))))
            except ValueError:
                continue
        options = list(dict.fromkeys(options))
        if len(options) != 1:
            raise ValueError(f"teams {text!r}: write them as AWY,HOM (for example LAL,GSW or Lakers@Warriors)")
        pair = options[0]
    if pair[0] == pair[1]:
        raise ValueError(f"teams {text!r}: the same team twice")
    return pair


def _code_or_none(text: Any) -> str | None:
    try:
        return team_code(str(text)) if text else None
    except ValueError:
        return None


# --------------------------------------------------------------------------- games of a scoreboard or a summary


def us_date(start: str) -> str | None:
    """The US Eastern date (YYYY-MM-DD) of an ESPN start time (UTC, "2024-12-26T01:00Z"): the day ESPN lists the
    game on and the game folders are named after. None when unreadable."""
    try:
        t = dt.datetime.fromisoformat(str(start).strip())  # Python 3.11+ reads the "Z"
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.UTC)
    try:
        from zoneinfo import ZoneInfo

        tz: dt.tzinfo = ZoneInfo("America/New_York")
    except (ImportError, KeyError, OSError, ValueError):  # no time zone database: EST (tip-offs are noon to 11 pm ET)
        tz = dt.timezone(dt.timedelta(hours=-5))
    return t.astimezone(tz).date().isoformat()


def _national_tv(comp: Mapping[str, Any]) -> list[str]:
    """National TV networks of a competition: ``broadcasts`` ``{"market": "national", "names": [...]}`` (scoreboard)
    or ``{"market": {"type": "National"}, "media": {"shortName": ...}}`` (summary header, ``geoBroadcasts``)."""
    names: list[str] = []
    for b in [*(comp.get("broadcasts") or []), *(comp.get("geoBroadcasts") or [])]:
        if not isinstance(b, Mapping):
            continue
        market = b.get("market")
        if isinstance(market, str) and market.lower() == "national":
            names += [str(n) for n in b.get("names") or [] if n]
        elif isinstance(market, Mapping) and str(market.get("type", "")).lower() == "national":
            name = (b.get("media") or {}).get("shortName")
            if name:
                names.append(str(name))
    return list(dict.fromkeys(names))


def _game(event_id: Any, start: Any, comp: Mapping[str, Any], status: Any) -> dict[str, Any]:
    sides: dict[str, dict[str, Any]] = {}
    for c in comp.get("competitors") or []:
        t = c.get("team") or {}
        abbr = str(t.get("abbreviation") or "")
        code = _code_or_none(abbr) or _code_or_none(t.get("displayName"))
        sides[str(c.get("homeAway"))] = {"abbr": abbr, "code": code, "name": str(t.get("displayName") or abbr),
                                         "score": c.get("score")}
    st = status.get("type") if isinstance(status, Mapping) else None
    st = st if isinstance(st, Mapping) else {}
    return {"event": str(event_id or ""), "start": start, "date": us_date(start) if start else None,
            "away": sides.get("away"), "home": sides.get("home"),
            "completed": st.get("completed") if isinstance(st.get("completed"), bool) else None,
            "status": str(st.get("description") or st.get("name") or ""), "national_tv": _national_tv(comp)}


def scoreboard_games(scoreboard: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The games of an ESPN scoreboard (``fetch_scoreboard``): ``{"event", "start" (UTC), "date" (US Eastern),
    "away" / "home": {"abbr" (ESPN), "code" (NBA.com tricode, None for a team outside the NBA), "name", "score"},
    "completed" (None: not given), "status", "national_tv"}``."""
    out = []
    for ev in scoreboard.get("events") or []:
        comp = (ev.get("competitions") or [{}])[0]
        out.append(_game(ev.get("id") or comp.get("id"), ev.get("date") or comp.get("date"), comp,
                         comp.get("status") or ev.get("status")))
    return out


def summary_game(summary: Mapping[str, Any]) -> dict[str, Any]:
    """The game of an ESPN summary (``fetch_summary``), as ``scoreboard_games`` gives it (from ``header``)."""
    head = summary.get("header") or {}
    comp = (head.get("competitions") or [{}])[0]
    return _game(head.get("id") or comp.get("id"), comp.get("date"), comp, comp.get("status"))


def find_games(games: Iterable[Mapping[str, Any]], teams: Iterable[str]) -> list[Mapping[str, Any]]:
    """The games between two teams (NBA.com tricodes, either order)."""
    want = set(teams)
    return [g for g in games if g.get("away") and g.get("home") and {g["away"]["code"], g["home"]["code"]} == want]


def teams(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """ESPN team id -> {"abbr", "name", "side": Team}."""
    out: dict[str, dict[str, Any]] = {}
    for c in summary["header"]["competitions"][0]["competitors"]:
        t = c["team"]
        out[str(t["id"])] = {"abbr": t["abbreviation"], "name": t.get("displayName", t["abbreviation"]),
                             "side": Team.HOME if c["homeAway"] == "home" else Team.AWAY}
    return out


def _athletes(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    side = {tid: v["side"] for tid, v in teams(summary).items()}
    for t in summary["boxscore"]["players"]:
        tid = str(t["team"]["id"])
        for grp in t["statistics"]:
            for a in grp["athletes"]:
                ath = a["athlete"]
                out[str(ath["id"])] = {"name": ath["displayName"], "jersey": str(ath.get("jersey", "")),
                                       "team": side[tid], "starter": bool(a.get("starter")),
                                       "dnp": bool(a.get("didNotPlay"))}
    return out


def _clock(text: str) -> float:
    m = re.fullmatch(r"(\d+):(\d+(?:\.\d+)?)", text.strip())
    if m:
        return 60.0 * int(m.group(1)) + float(m.group(2))
    return float(text)


def pbp_actions(summary: dict[str, Any]) -> list[PBPAction]:
    """ESPN plays -> ``PBPAction`` rows (generic schema; see the module docstring)."""
    tmap = {tid: v["side"] for tid, v in teams(summary).items()}
    ath = _athletes(summary)
    rows: list[dict[str, Any]] = []

    def name(pid: str | None) -> str:
        return ath.get(pid or "", {}).get("name", "") if pid else ""

    def side_of(pid: str | None, fallback: Team | None) -> Team | None:
        return ath[pid]["team"] if pid and pid in ath else fallback

    n = 0
    for p in summary["plays"]:
        n += 1
        typ = p["type"]["text"].replace("\n", " ").strip()
        text = p.get("text", "")
        low = text.lower()
        period = int(p["period"]["number"])
        clock = _clock(p["clock"]["displayValue"])
        team = tmap.get(str(p.get("team", {}).get("id", "")))
        parts = [str(x["athlete"]["id"]) for x in p.get("participants", [])]
        base = {"period": period, "clock": clock, "description": text, "score_home": p.get("homeScore"),
                "score_away": p.get("awayScore")}
        coord = p.get("coordinate") or {}
        number = int(p.get("sequenceNumber", n))

        def add(kind: str, number: int = number, **kw: Any) -> None:
            rows.append({**base, "action_number": number, "kind": kind, **kw})

        if p.get("shootingPlay"):
            free = typ.lower().startswith("free throw")
            made = bool(p.get("scoringPlay")) and "misses" not in low
            value = 1 if free else int(p.get("pointsAttempted") or (3 if "three point" in low else 2))
            shooter = parts[0] if parts else None
            kw: dict[str, Any] = {"team": side_of(shooter, team).value if side_of(shooter, team) else "",
                                  "player_id": shooter or "", "player_name": name(shooter), "value": value,
                                  "made": int(made)}
            if free:
                m = re.search(r"(\d) of (\d)", typ)
                if m:
                    kw.update(ft_index=int(m.group(1)), ft_total=int(m.group(2)))
            else:
                kw["shot_type"] = "three" if value == 3 else "two"
                if coord:
                    kw.update(x=coord.get("x"), y=coord.get("y"))
            add("shot_made" if made else "shot_missed", **kw)
            if made and " assists)" in low and len(parts) > 1:
                add("assist", number=number * 10 + 1, team=kw["team"], player_id=parts[1], player_name=name(parts[1]),
                    other_player_id=shooter or "", parent_action=number)
            if " blocks " in low and len(parts) > 1:
                blk_team = side_of(parts[1], None)
                add("block", number=number * 10 + 2, team=blk_team.value if blk_team else "", player_id=parts[1],
                    player_name=name(parts[1]), other_player_id=shooter or "", parent_action=number)
        elif "rebound" in typ.lower():
            off = typ.lower().startswith("offensive")
            pid = parts[0] if parts else None
            reb_team = side_of(pid, team)
            add("rebound", team=reb_team.value if reb_team else "", player_id=pid or "", player_name=name(pid),
                offensive=int(off), type="team" if pid is None else "player")
        elif "turnover" in typ.lower() or "turnover" in low:
            pid = parts[0] if parts else None
            to_team = side_of(pid, team)
            add("turnover", team=to_team.value if to_team else "", player_id=pid or "", player_name=name(pid),
                type=typ)
            if "steals)" in low and len(parts) > 1:
                st_team = side_of(parts[1], None)
                add("steal", number=number * 10 + 3, team=st_team.value if st_team else "", player_id=parts[1],
                    player_name=name(parts[1]), other_player_id=pid or "", parent_action=number)
        elif "foul" in typ.lower():
            pid = parts[0] if parts else None
            f_team = side_of(pid, team)
            add("foul", team=f_team.value if f_team else "", player_id=pid or "", player_name=name(pid), type=typ)
        elif "timeout" in typ.lower():
            add("timeout", team=team.value if team else "", type=typ)
        elif typ.lower() == "substitution":
            add("substitution", team=team.value if team else "", player_id=parts[0] if parts else "",
                other_player_id=parts[1] if len(parts) > 1 else "")
        elif typ.lower() == "jumpball":
            add("jump_ball", team=team.value if team else "", player_id=parts[0] if parts else "")
        elif typ.lower() == "end period":
            add("period_end")
    return parse_pbp_generic(rows)


@dataclass
class PlayerBox:
    player_id: str
    name: str
    team: Team
    jersey: str
    minutes: float = 0.0
    stats: dict[str, int] = field(default_factory=dict)


@dataclass
class TeamBox:
    team: Team
    abbr: str
    stats: dict[str, int] = field(default_factory=dict)
    players: list[PlayerBox] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["team"] = self.team.value
        for p in d["players"]:
            p["team"] = p["team"].value if isinstance(p["team"], Team) else p["team"]
        return d


def _pair(v: str) -> tuple[int, int]:
    a, b = v.split("-")
    return int(a), int(b)


def box_score(summary: dict[str, Any]) -> dict[Team, TeamBox]:
    """Official team and player totals."""
    tinfo = teams(summary)
    out: dict[Team, TeamBox] = {}
    for t in summary["boxscore"]["teams"]:
        info = tinfo[str(t["team"]["id"])]
        s = {x["name"]: x["displayValue"] for x in t["statistics"]}
        fgm, fga = _pair(s["fieldGoalsMade-fieldGoalsAttempted"])
        fg3m, fg3a = _pair(s["threePointFieldGoalsMade-threePointFieldGoalsAttempted"])
        ftm, fta = _pair(s["freeThrowsMade-freeThrowsAttempted"])
        stats = {"fgm": fgm, "fga": fga, "fg3m": fg3m, "fg3a": fg3a, "ftm": ftm, "fta": fta,
                 "oreb": int(s["offensiveRebounds"]), "dreb": int(s["defensiveRebounds"]),
                 "reb": int(s["totalRebounds"]), "ast": int(s["assists"]), "stl": int(s["steals"]),
                 "blk": int(s["blocks"]), "tov": int(s.get("totalTurnovers", s["turnovers"])), "pf": int(s["fouls"])}
        stats["pts"] = 2 * fgm + fg3m + ftm
        out[info["side"]] = TeamBox(info["side"], info["abbr"], stats)
    for t in summary["boxscore"]["players"]:
        info = tinfo[str(t["team"]["id"])]
        for grp in t["statistics"]:
            labels = grp["labels"]
            for a in grp["athletes"]:
                if a.get("didNotPlay") or not a.get("stats"):
                    continue
                v = dict(zip(labels, a["stats"], strict=False))
                fgm, fga = _pair(v["FG"])
                fg3m, fg3a = _pair(v["3PT"])
                ftm, fta = _pair(v["FT"])
                stats = {"pts": int(v["PTS"]), "fgm": fgm, "fga": fga, "fg3m": fg3m, "fg3a": fg3a, "ftm": ftm,
                         "fta": fta, "oreb": int(v["OREB"]), "dreb": int(v["DREB"]), "reb": int(v["REB"]),
                         "ast": int(v["AST"]), "stl": int(v["STL"]), "blk": int(v["BLK"]), "tov": int(v["TO"]),
                         "pf": int(v["PF"])}
                ath = a["athlete"]
                out[info["side"]].players.append(PlayerBox(str(ath["id"]), ath["displayName"], info["side"],
                                                           str(ath.get("jersey", "")), float(v.get("MIN") or 0),
                                                           stats))
    return out


def roster(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Players as roster rows: ``{"player_id", "name", "team", "jersey", "starter"}``."""
    return [{"player_id": pid, "name": a["name"], "team": a["team"].value, "jersey": a["jersey"],
             "starter": a["starter"]} for pid, a in _athletes(summary).items() if not a["dnp"]]
