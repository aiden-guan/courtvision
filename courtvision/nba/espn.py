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

ESPN data is NBA statistics: research use only (docs/TRAINING.md §1), never shipped in a product.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from courtvision.training.align_pbp import PBPAction, parse_pbp_generic
from courtvision.types import Team

__all__ = ["PlayerBox", "TeamBox", "box_score", "fetch_summary", "load_summary", "pbp_actions", "roster", "teams"]

SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event}"
STATS = ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf")


def fetch_summary(event_id: str, out: str | Path | None = None, timeout: float = 30.0) -> dict[str, Any]:
    with urllib.request.urlopen(SUMMARY_URL.format(event=event_id), timeout=timeout) as resp:  # noqa: S310 (fixed host)
        data = json.loads(resp.read())
    if out is not None:
        Path(out).write_text(json.dumps(data))
    return data


def load_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


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
