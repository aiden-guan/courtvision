"""Play-by-play <-> video alignment through a game-clock timeline (training_data.md §2.2-2.3, RESEARCH §5.3).

Pipeline (each step is a public function so it can be tested and reused on its own):

1. **Parse PBP.** ``parse_pbp_v3`` reads nba_api ``PlayByPlayV3`` / live-feed style rows (``period``,
   ``clock`` "PT11M32.00S", ``actionNumber``, ``actionType``, ``subType``, ``personId``/``playerName``,
   ``teamTricode``, ``shotResult``, x/y). ``parse_pbp_generic`` reads our own CSV/JSON schema
   (``GENERIC_COLUMNS``). Both give ``PBPAction`` rows whose ``kind`` is an ``EventKind``.
2. **Fit the clock timeline.** ``ClockTimeline.fit`` turns 5 Hz scoreboard reads (OCR noise, misreads,
   gaps) into a monotone piecewise-linear map between video time and (period, game clock): slope -1
   while the clock runs, plateaus while it is stopped. ``ClockTimeline.from_exact`` builds the same
   map from an exact feed (simulator, official data feed).
3. **Place actions** (``align_pbp``). Live-ball actions map through the clock and are shifted back by
   the PBP entry lag (1-2 s per BasketballBench [22]; estimated per game from anchors). Actions that
   share a stopped-clock plateau (fouls, free throws, substitutions, timeouts) are ordered by
   ``actionNumber`` and anchored on confirmed score deltas (+1 per made FT). Each placed action is then
   snapped to a nearby engine anchor (release / rim crossing / control) when one exists.
4. **Accept or reject the game**: median |residual| on made field goals < 0.5 s (training_data §5 step 7).
5. **Emit labelled clips** (``make_clips``) and write them as JSONL (``write_jsonl``).

``pbp_from_events`` turns engine or simulator events into a pseudo-PBP (for tests and sim studies);
``evaluate_against_pbp`` compares engine counts with the PBP per period (FGA, FGM, 3PA, ...).

PBP field names drift: the nba_api docs were last validated in 2023 (V3) and 2020 (live), so every
field read here has a fallback, and unverified semantics are marked (training_data §2.1). Basketball
semantics follow ARCHITECTURE "Shared event semantics" and basketball_rules_stats.md §2a (a fouled
shooter who misses has no FGA row; synthetic team rebounds after non-final FT misses and at period end
are not rebounds).
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from bisect import bisect_right
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from courtvision.events.scoreboard import ScoreboardConfig, ScoreboardConfirmer, ScoreChange
from courtvision.types import Event, EventKind, EventStatus, ScoreboardReading, Team

__all__ = [
    "GENERIC_COLUMNS",
    "AlignConfig",
    "AlignedAction",
    "AlignmentResult",
    "ClockPoint",
    "ClockTimeline",
    "LabeledClip",
    "PBPAction",
    "PBPAgreement",
    "TimelineConfig",
    "actions_to_rows",
    "align_pbp",
    "evaluate_against_pbp",
    "format_clock",
    "infer_team_map",
    "load_pbp",
    "make_clips",
    "parse_clock",
    "parse_pbp_generic",
    "parse_pbp_v3",
    "pbp_from_events",
    "read_jsonl",
    "score_changes",
    "write_jsonl",
    "write_pbp_csv",
]

log = logging.getLogger(__name__)

_EPS = 1e-6
_SHOT_KINDS = (EventKind.SHOT_MADE, EventKind.SHOT_MISSED)
_LIVE_KINDS = frozenset({EventKind.SHOT_MADE, EventKind.SHOT_MISSED, EventKind.REBOUND, EventKind.TURNOVER,
                         EventKind.STEAL, EventKind.BLOCK, EventKind.ASSIST, EventKind.POSSESSION})
_CHILD_KINDS = frozenset({EventKind.ASSIST, EventKind.BLOCK, EventKind.STEAL})  # follow their parent action

GENERIC_COLUMNS: tuple[str, ...] = (
    "action_number", "period", "clock", "kind", "team", "player_id", "player_name", "made", "value",
    "shot_type", "offensive", "type", "cause", "other_player_id", "ft_index", "ft_total", "parent_action",
    "x", "y", "score_home", "score_away", "description",
)


# =============================================================================================== clock text


_ISO_CLOCK = re.compile(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$", re.IGNORECASE)
_MMSS = re.compile(r"^(\d+):(\d{1,2}(?:\.\d+)?)$")


def parse_clock(value: str | float | int | None) -> float:
    """Seconds remaining from "PT11M32.00S", "11:32", "0:23.4", "23.4" or a number; ValueError otherwise."""
    if value is None or isinstance(value, bool):
        raise ValueError(f"not a clock value: {value!r}")
    if isinstance(value, (int, float)):
        secs = float(value)
    else:
        text = str(value).strip()
        iso = _ISO_CLOCK.match(text)
        mmss = _MMSS.match(text)
        if iso and (iso.group(1) or iso.group(2)):
            secs = int(iso.group(1) or 0) * 60 + float(iso.group(2) or 0.0)
        elif mmss:
            secs = int(mmss.group(1)) * 60 + float(mmss.group(2))
        else:
            try:
                secs = float(text)
            except ValueError as exc:
                raise ValueError(f"not a clock value: {value!r}") from exc
    if not math.isfinite(secs) or secs < 0:
        raise ValueError(f"not a clock value: {value!r}")
    return secs


def format_clock(seconds: float) -> str:
    """ISO-8601 duration in the nba_api style, e.g. 692.0 -> "PT11M32.00S"."""
    s = round(max(0.0, float(seconds)), 2)  # round first: 59.999 must give PT01M00.00S, not PT00M60.00S
    minutes = int(s // 60)
    return f"PT{minutes:02d}M{s - 60 * minutes:05.2f}S"


# =============================================================================================== PBP rows


@dataclass
class PBPAction:
    """One play-by-play action, normalised. ``kind`` is None for rows CourtVision does not label
    (violations, replays, ejections, ...). Shot rows carry the result: one row = attempt + result."""

    action_number: int
    period: int
    clock: float  # seconds remaining as recorded by the feed
    kind: EventKind | None
    team: Team | None = None
    player_id: str | None = None  # feed person id (generic / pseudo rows: our player id); None = team-level
    player_name: str | None = None
    value: int = 0  # shot rows: 1 (FT) / 2 / 3
    made: bool | None = None
    team_tricode: str | None = None
    action_type: str = ""
    sub_type: str = ""
    description: str = ""
    x: float | None = None
    y: float | None = None
    score_home: int | None = None
    score_away: int | None = None
    time_actual: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def is_shot(self) -> bool:
        return self.kind in _SHOT_KINDS

    @property
    def is_free_throw(self) -> bool:
        return self.is_shot and (self.value == 1 or self.attrs.get("shot_type") == "free_throw")

    @property
    def is_field_goal(self) -> bool:
        return self.is_shot and not self.is_free_throw

    @property
    def synthetic(self) -> bool:
        """Feed bookkeeping rows that are not basketball events (e.g. team rebound after FT 1 of 2)."""
        return bool(self.attrs.get("synthetic"))

    @property
    def live_ball(self) -> bool:
        """Happens while the ball is live (placed through the running clock, with the entry lag)."""
        return self.kind in _LIVE_KINDS and not self.is_free_throw

    def to_row(self) -> dict[str, Any]:
        """Generic-schema row (``GENERIC_COLUMNS``)."""
        a = self.attrs
        return {
            "action_number": self.action_number, "period": self.period, "clock": round(self.clock, 3),
            "kind": self.kind.value if self.kind else "", "team": self.team.value if self.team else "",
            "player_id": self.player_id or "", "player_name": self.player_name or "",
            "made": "" if self.made is None else int(self.made), "value": self.value,
            "shot_type": a.get("shot_type", ""), "offensive": "" if a.get("offensive") is None else int(bool(a["offensive"])),
            "type": a.get("type", ""), "cause": a.get("cause", ""), "other_player_id": a.get("other_player_id", ""),
            "ft_index": a.get("ft_index", ""), "ft_total": a.get("ft_total", ""), "parent_action": a.get("parent_action", ""),
            "x": "" if self.x is None else self.x, "y": "" if self.y is None else self.y,
            "score_home": "" if self.score_home is None else self.score_home,
            "score_away": "" if self.score_away is None else self.score_away, "description": self.description,
        }


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def _get(row: Mapping[str, Any], *names: str) -> Any:
    for n in names:
        v = row.get(n)
        if v is not None and v != "":
            return v
    return None


def _opt_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _opt_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _opt_bool(v: Any) -> bool | None:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "made", "make"):
        return True
    if s in ("0", "false", "f", "no", "n", "missed", "miss"):
        return False
    return None


def _person(v: Any, team_id: str | None) -> str | None:
    """Feed person id; 0 / empty / the team's own id mark a team-level row (team rebound / turnover)."""
    if v is None or v == "":
        return None
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s in ("0", "") or (team_id is not None and s == team_id):
        return None
    return s


def _shot_type(*texts: str) -> str:
    t = _norm(" ".join(texts))
    for key in ("dunk", "tip", "layup", "hook"):
        if key in t:
            return key
    return "jumper"


def _turnover_type(sub_type: str, description: str) -> str:
    """Map a feed turnover subtype to the types.py vocabulary (mapping unverified against live data)."""
    t = _norm(sub_type) or _norm(description)
    if "outofbounds" in t or "stepoutofbounds" in t:
        return "out_of_bounds"
    if "badpass" in t:
        return "bad_pass"
    if "lostball" in t:
        return "steal"
    if "travel" in t:
        return "travel"
    if "shotclock" in t:
        return "shot_clock"
    if "offensivefoul" in t or "offensive" in t or "charge" in t:
        return "offensive_foul"
    return "other"


_FT_OF = re.compile(r"(\d+)\s*of\s*(\d+)", re.IGNORECASE)
_THREE = re.compile(r"\b3PT\b", re.IGNORECASE)
# V3 descriptions append "(Name N AST)" to assisted makes; "Name BLOCK (N BLK)" / "Name STEAL (N STL)" may
# appear on their own rows. Formats are unverified against a live 2025-26 response (training_data §2.1).
# Names are matched as Unicode words: NBA feeds spell "Jokić", "Dončić", "Porziņģis" with diacritics.
_AST = re.compile(r"\(([^()]+?) \d+ AST\)")
_BLK = re.compile(r"([^\W\d_][\w.'\-]*) BLOCK \(\d+ BLK\)")
_STL = re.compile(r"([^\W\d_][\w.'\-]*) STEAL \(\d+ STL\)")

_V3_KINDS: dict[str, str | EventKind] = {
    "madeshot": "fg", "missedshot": "fg", "2pt": "fg", "3pt": "fg", "fieldgoal": "fg", "freethrow": "ft",
    "rebound": EventKind.REBOUND, "turnover": EventKind.TURNOVER, "steal": EventKind.STEAL,
    "block": EventKind.BLOCK, "foul": EventKind.FOUL, "substitution": EventKind.SUBSTITUTION,
    "timeout": EventKind.TIMEOUT, "jumpball": EventKind.POSSESSION, "period": "period",
}


def _v3_rows(data: Any) -> list[Mapping[str, Any]]:
    if isinstance(data, Mapping):
        if isinstance(data.get("game"), Mapping) and "actions" in data["game"]:
            return list(data["game"]["actions"])
        if "actions" in data:
            return list(data["actions"])
        for rs in data.get("resultSets") or []:
            if rs.get("name", "PlayByPlay") == "PlayByPlay":
                headers = list(rs["headers"])
                return [dict(zip(headers, r, strict=False)) for r in rs.get("rowSet", [])]
        raise ValueError("unrecognised play-by-play document (expected game.actions, actions or resultSets)")
    return list(data)


def _v3_kind(row: Mapping[str, Any]) -> tuple[EventKind | None, dict[str, Any]]:
    """Kind plus shot fields of one V3 / live row."""
    atype, sub, desc = str(row.get("actionType") or ""), str(row.get("subType") or ""), str(row.get("description") or "")
    key = _V3_KINDS.get(_norm(atype))
    if key is None and not atype and desc:  # V3 may log steals / blocks as untyped rows (unverified)
        key = EventKind.STEAL if " STEAL " in f" {desc.upper()} " else EventKind.BLOCK if " BLOCK " in f" {desc.upper()} " else None
    extra: dict[str, Any] = {}
    if key == "fg":
        result = _norm(row.get("shotResult")) or ("made" if _norm(atype) == "madeshot" else "missed" if _norm(atype) == "missedshot" else "")
        if not result:
            result = "missed" if desc.upper().startswith("MISS") else "made"
        sv = _opt_int(row.get("shotValue"))
        extra["value"] = sv if sv in (2, 3) else 3 if (_norm(atype) == "3pt" or _THREE.search(desc)) else 2
        extra["made"] = result == "made"
        extra["shot_type"] = _shot_type(sub, desc)
        return (EventKind.SHOT_MADE if extra["made"] else EventKind.SHOT_MISSED), extra
    if key == "ft":
        result = _norm(row.get("shotResult"))
        made = result == "made" if result else not desc.upper().startswith("MISS")
        m = _FT_OF.search(sub) or _FT_OF.search(desc)
        extra.update(value=1, made=made, shot_type="free_throw",
                     ft_index=int(m.group(1)) if m else 1, ft_total=int(m.group(2)) if m else 1)
        if "technical" in _norm(sub) or "technical" in _norm(desc):
            extra["technical"] = True
        return (EventKind.SHOT_MADE if made else EventKind.SHOT_MISSED), extra
    if key == "period":
        s = _norm(sub)
        return (EventKind.PERIOD_START if s == "start" else EventKind.PERIOD_END if s == "end" else None), extra
    if key is EventKind.REBOUND and _norm(sub) in ("offensive", "defensive"):
        extra["offensive"] = _norm(sub) == "offensive"
    if key is EventKind.TURNOVER:
        extra["type"] = _turnover_type(sub, desc)
    if key is EventKind.POSSESSION:
        extra["cause"] = "jump_ball"
    return (key if isinstance(key, EventKind) else None), extra


def _v3_team(row: Mapping[str, Any], team_map: Mapping[str, Team] | None) -> Team | None:
    loc = _norm(row.get("location"))
    if loc in ("h", "home"):
        return Team.HOME
    if loc in ("v", "a", "away", "visitor"):
        return Team.AWAY
    tri = row.get("teamTricode")
    if team_map and tri in team_map:
        return team_map[tri]
    return None


def _v3_action(row: Mapping[str, Any], team_map: Mapping[str, Team] | None) -> PBPAction | None:
    period = _opt_int(row.get("period"))
    clock_raw = row.get("clock")
    if period is None or clock_raw in (None, ""):
        return None
    try:
        clock = parse_clock(clock_raw)
    except ValueError:
        log.debug("skipping PBP row with unreadable clock %r", clock_raw)
        return None
    kind, extra = _v3_kind(row)
    team_id = None if row.get("teamId") in (None, "", 0, "0") else str(row.get("teamId")).removesuffix(".0")
    attrs: dict[str, Any] = {k: v for k, v in extra.items() if k not in ("value", "made")}
    x, y = _opt_float(_get(row, "xLegacy")), _opt_float(_get(row, "yLegacy"))
    if x is not None:
        attrs["xy_units"] = "legacy_tenths_ft_from_basket"  # semantics unverified (training_data §2.1)
    else:
        x, y = _opt_float(row.get("x")), _opt_float(row.get("y"))
        if x is not None:
            attrs["xy_units"] = "percent"  # live-feed court-percent floats; semantics unverified
    for key, name in (("assistPersonId", "assist_person_id"), ("blockPersonId", "block_person_id"),
                      ("stealPersonId", "steal_person_id")):
        pid = _person(row.get(key), team_id)
        if pid is not None:
            attrs[name] = pid
    if team_id is not None:
        attrs["team_id"] = team_id
    order = _opt_int(_get(row, "orderNumber"))
    if order is not None:
        attrs["order_number"] = order
    return PBPAction(
        action_number=_opt_int(_get(row, "actionNumber", "actionId", "orderNumber")) or 0, period=period, clock=clock,
        kind=kind, team=_v3_team(row, team_map), player_id=_person(row.get("personId"), team_id),
        player_name=_get(row, "playerName", "playerNameI"), value=int(extra.get("value", 0)), made=extra.get("made"),
        team_tricode=row.get("teamTricode") or None, action_type=str(row.get("actionType") or ""),
        sub_type=str(row.get("subType") or ""), description=str(row.get("description") or ""), x=x, y=y,
        score_home=_opt_int(row.get("scoreHome")), score_away=_opt_int(row.get("scoreAway")),
        time_actual=row.get("timeActual") or None, attrs=attrs,
    )


def infer_team_map(actions: Sequence[PBPAction]) -> dict[str, Team]:
    """Tricode -> Team from which running score moves on made shots (majority vote)."""
    votes: dict[str, Counter[Team]] = {}
    prev: tuple[int, int] | None = None
    for a in sorted(actions, key=_order_key):
        if a.score_home is None or a.score_away is None:
            continue
        cur = (a.score_home, a.score_away)
        if prev is None and a.made and sum(cur) == a.value:  # the game's first points
            prev = (0, 0)
        if prev is not None and a.made and a.team_tricode and cur != prev:
            side = Team.HOME if cur[0] > prev[0] else Team.AWAY if cur[1] > prev[1] else None
            if side is not None:
                votes.setdefault(a.team_tricode, Counter())[side] += 1
        prev = cur
    out = {tri: c.most_common(1)[0][0] for tri, c in votes.items()}
    if len(out) == 1:  # one team never scored: it is the other side
        others = {a.team_tricode for a in actions if a.team_tricode} - set(out)
        if len(others) == 1:
            out[others.pop()] = next(iter(out.values())).other()
    return out


def parse_pbp_v3(data: Any, team_map: Mapping[str, Team] | None = None) -> list[PBPAction]:
    """Parse nba_api PlayByPlayV3 / live play-by-play rows (list, ``{"game": {"actions"}}`` or resultSets).

    Team side comes from ``location`` ("h"/"v"), then ``team_map`` (tricode -> Team), then score moves.
    Assists, blocks and steals recorded on another row (person-id fields or description text) become
    their own ``ASSIST`` / ``BLOCK`` / ``STEAL`` actions with ``attrs["parent_action"]``.
    """
    actions = [a for a in (_v3_action(r, team_map) for r in _v3_rows(data)) if a is not None]
    if any(a.team is None and a.team_tricode for a in actions):
        inferred = {**infer_team_map(actions), **dict(team_map or {})}
        for a in actions:
            if a.team is None and a.team_tricode in inferred:
                a.team = inferred[a.team_tricode]
    actions.extend(_derived_actions(actions))
    _finalise(actions)
    return sorted(actions, key=_order_key)


def _derived_actions(actions: Sequence[PBPAction]) -> list[PBPAction]:
    explicit = {(a.kind, a.period, round(a.clock, 2)) for a in actions if a.kind in _CHILD_KINDS}
    out: list[PBPAction] = []
    for a in actions:
        specs: list[tuple[EventKind, str | None, str | None, Team | None]] = []
        if a.kind is EventKind.SHOT_MADE and a.is_field_goal:
            m = _AST.search(a.description)
            if a.attrs.get("assist_person_id") or m:
                specs.append((EventKind.ASSIST, a.attrs.get("assist_person_id"), m.group(1).strip() if m else None, a.team))
        if a.kind is EventKind.SHOT_MISSED and a.is_field_goal:
            m = _BLK.search(a.description)
            if a.attrs.get("block_person_id") or m:
                specs.append((EventKind.BLOCK, a.attrs.get("block_person_id"), m.group(1) if m else None, a.team.other() if a.team else None))
        if a.kind is EventKind.TURNOVER:
            m = _STL.search(a.description)
            if a.attrs.get("steal_person_id") or m:
                specs.append((EventKind.STEAL, a.attrs.get("steal_person_id"), m.group(1) if m else None, a.team.other() if a.team else None))
        for kind, pid, name, team in specs:
            if (kind, a.period, round(a.clock, 2)) in explicit:
                continue  # the live feed also logs this as its own row
            out.append(PBPAction(action_number=a.action_number, period=a.period, clock=a.clock, kind=kind, team=team,
                                 player_id=pid, player_name=name, description=a.description,
                                 attrs={"parent_action": a.action_number, "derived": True}))
    return out


def _finalise(actions: list[PBPAction]) -> None:
    """Offensive flags for rebounds, synthetic team rebounds, steal-typed turnovers."""
    steals = {(a.period, round(a.clock, 2)) for a in actions if a.kind is EventKind.STEAL}
    last_miss: PBPAction | None = None
    ordered = sorted(actions, key=_order_key)
    for i, a in enumerate(ordered):
        if a.kind is EventKind.PERIOD_START:
            last_miss = None
        if a.kind is EventKind.TURNOVER and a.attrs.get("type") in ("other", None) and (a.period, round(a.clock, 2)) in steals:
            a.attrs["type"] = "steal"
        if a.kind is EventKind.SHOT_MISSED:
            last_miss = a
        elif a.kind is EventKind.SHOT_MADE:
            last_miss = None
        if a.kind is not EventKind.REBOUND:
            continue
        team_level = a.player_id is None and not a.player_name
        nonfinal_ft = (last_miss is not None and last_miss.is_free_throw
                       and int(last_miss.attrs.get("ft_index", 1)) < int(last_miss.attrs.get("ft_total", 1)))
        at_buzzer = a.clock <= 0.0 + _EPS and any(b.kind is EventKind.PERIOD_END and b.period == a.period for b in ordered[i + 1:i + 4])
        if team_level and (nonfinal_ft or at_buzzer or last_miss is None):
            a.attrs["synthetic"] = True  # basketball_rules_stats §2a: feed bookkeeping, not a rebound
        if a.attrs.get("offensive") is None and last_miss is not None and a.team is not None and last_miss.team is not None:
            a.attrs["offensive"] = a.team is last_miss.team
        last_miss = None


_GENERIC_KINDS: dict[str, EventKind] = {k.value: k for k in EventKind} | {
    "made_shot": EventKind.SHOT_MADE, "missed_shot": EventKind.SHOT_MISSED, "jump_ball": EventKind.POSSESSION,
    "jumpball": EventKind.POSSESSION, "free_throw_made": EventKind.SHOT_MADE, "free_throw_missed": EventKind.SHOT_MISSED,
}


def parse_pbp_generic(rows: Iterable[Mapping[str, Any]]) -> list[PBPAction]:
    """Parse the generic schema (``GENERIC_COLUMNS``; team "home"/"away"; clock in any ``parse_clock`` form)."""
    out: list[PBPAction] = []
    for i, row in enumerate(rows):
        kind_raw = str(row.get("kind") or "").strip().lower()
        kind = _GENERIC_KINDS.get(kind_raw)
        if kind_raw and kind is None:
            log.debug("generic PBP row %d: unknown kind %r kept unlabelled", i, kind_raw)
        period = _opt_int(row.get("period"))
        if period is None:
            raise ValueError(f"generic PBP row {i}: missing period")
        made = _opt_bool(row.get("made"))
        if kind in _SHOT_KINDS:
            made = kind is EventKind.SHOT_MADE
        default_value = 1 if kind_raw.startswith("free_throw") else 2 if kind in _SHOT_KINDS else 0
        value = _opt_int(row.get("value")) or default_value
        attrs: dict[str, Any] = {}
        for key in ("shot_type", "type", "cause", "other_player_id"):
            if row.get(key) not in (None, ""):
                attrs[key] = str(row[key])
        for key in ("ft_index", "ft_total", "parent_action"):
            v = _opt_int(row.get(key))
            if v is not None:
                attrs[key] = v
        off = _opt_bool(row.get("offensive"))
        if off is not None:
            attrs["offensive"] = off
        if kind in _SHOT_KINDS and value == 1:
            attrs["shot_type"] = "free_throw"
        if kind is EventKind.POSSESSION and kind_raw in ("jump_ball", "jumpball"):
            attrs.setdefault("cause", "jump_ball")
        team_raw = str(row.get("team") or "").strip().lower()
        out.append(PBPAction(
            action_number=_opt_int(row.get("action_number")) or i + 1, period=period, clock=parse_clock(row.get("clock")),
            kind=kind, team=Team(team_raw) if team_raw in ("home", "away") else None,
            player_id=str(row["player_id"]) if row.get("player_id") not in (None, "") else None,
            player_name=str(row["player_name"]) if row.get("player_name") not in (None, "") else None,
            value=value, made=made, description=str(row.get("description") or ""),
            x=_opt_float(row.get("x")), y=_opt_float(row.get("y")),
            score_home=_opt_int(row.get("score_home")), score_away=_opt_int(row.get("score_away")), attrs=attrs,
        ))
    _finalise(out)
    return sorted(out, key=_order_key)


def actions_to_rows(actions: Iterable[PBPAction]) -> list[dict[str, Any]]:
    return [a.to_row() for a in actions]


def write_pbp_csv(actions: Iterable[PBPAction], path: str | Path) -> int:
    """Write actions in the generic CSV schema; returns the row count."""
    rows = actions_to_rows(actions)
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(GENERIC_COLUMNS))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def load_pbp(path: str | Path, team_map: Mapping[str, Team] | None = None) -> list[PBPAction]:
    """Load .csv (generic schema) or .json / .jsonl (V3 / live rows, or generic rows)."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        with p.open(newline="") as fh:
            return parse_pbp_generic(list(csv.DictReader(fh)))
    text = p.read_text()
    data: Any = [json.loads(line) for line in text.splitlines() if line.strip()] if p.suffix.lower() == ".jsonl" else json.loads(text)
    rows = _v3_rows(data)
    if rows and "kind" in rows[0] and "actionType" not in rows[0]:
        return parse_pbp_generic(rows)
    return parse_pbp_v3(rows, team_map)


def _order_key(a: PBPAction) -> tuple[int, int, int]:
    return (a.period, a.action_number, 1 if a.attrs.get("derived") else 0)


# =============================================================================================== timeline


@dataclass
class TimelineConfig:
    """Clock-timeline fit settings (training_data §2.2 step 3; RESEARCH §7A.5)."""

    min_conf: float = 0.5  # OCR reads below this confidence are dropped (= ScoreboardConfig.min_ocr_conf); tuned
    tenths_below_s: float = 60.0  # displays switch MM:SS -> SS.t in the last minute (rules note §3); sourced
    quantum_above_s: float = 1.0  # display resolution above tenths_below_s (MM:SS); sourced (rules note §3)
    quantum_below_s: float = 0.1  # display resolution in the last minute (SS.t); sourced (rules note §3)
    # How a display value relates to the true clock: "floor" (11:32 shown for 692.0-692.99), "ceil", "round"
    # or "none" (treat the value as exact). Venue conventions are unverified. "none" keeps OCR and PBP on the
    # same display scale, so the quantisation bias cancels when both are quantised the same way.
    clock_display: str = "none"
    read_hz: float = 5.0  # nominal OCR rate (D14); sourced
    plateau_slack_s: float = 0.5  # a value shown longer than one quantum + this = stopped clock; tuned
    segment_tol_s: float = 0.6  # tolerance on the intercept (clock + t) beyond one quantum in a running segment; tuned
    min_segment_reads: int = 3  # sustained shifts shorter than this are outliers, not a new segment; tuned
    ref_window: int = 25  # reads in the rolling-median intercept of a running segment; tuned
    period_window: int = 7  # reads in the majority filter on period labels; tuned
    max_slope_dev: float = 0.05  # warn when a long running segment's slope differs from -1 by more (video not 1x?); tuned
    extrapolate_s: float = 5.0  # clocks at most this far outside a period's observed range are extrapolated; tuned

    def quantum(self, clock: float) -> float:
        return self.quantum_below_s if clock < self.tenths_below_s else self.quantum_above_s

    def correct(self, clock: float) -> float:
        """Display value -> the value on the fitted line (midpoint of the display interval)."""
        q = self.quantum(clock)
        return clock + {"floor": 0.5 * q, "ceil": -0.5 * q}.get(self.clock_display, 0.0)


@dataclass(frozen=True, slots=True)
class ClockPoint:
    period: int
    clock: float
    running: bool
    extrapolated: bool = False


@dataclass(slots=True)
class _Read:
    t: float
    period: int
    clock: float  # display value
    conf: float


@dataclass(slots=True)
class _Plateau:
    value: float  # corrected clock value
    t_first: float
    t_last: float
    n: int
    quantum: float


@dataclass(slots=True)
class _Run:
    b: float  # intercept: clock + t (slope -1)
    t_first: float
    t_last: float
    n: int
    mad: float
    slope: float | None


def _longest_nondecreasing(vals: Sequence[float]) -> list[int]:
    """Indices of one longest non-decreasing subsequence (patience sorting, O(n log n))."""
    tails: list[float] = []
    tail_idx: list[int] = []
    prev = [-1] * len(vals)
    for i, v in enumerate(vals):
        k = bisect_right(tails, v)
        if k == len(tails):
            tails.append(v)
            tail_idx.append(i)
        else:
            tails[k] = v
            tail_idx[k] = i
        prev[i] = tail_idx[k - 1] if k > 0 else -1
    out: list[int] = []
    i = tail_idx[-1] if tail_idx else -1
    while i >= 0:
        out.append(i)
        i = prev[i]
    return out[::-1]


def _mode_filter(periods: Sequence[int], window: int) -> list[int]:
    half = max(window, 1) // 2
    out: list[int] = []
    for i, p in enumerate(periods):
        counts = Counter(periods[max(0, i - half): i + half + 1])
        best = max(counts.values())
        out.append(p if counts[p] == best else min(k for k, v in counts.items() if v == best))
    return out


def _clean_reads(readings: Iterable[ScoreboardReading], cfg: TimelineConfig) -> tuple[dict[int, list[_Read]], dict[str, int]]:
    raw = [r for r in readings if r.period is not None and r.game_clock is not None]
    stats = {"n_reads": len(raw)}
    reads = [_Read(float(r.t), int(r.period), float(r.game_clock), float(r.confidence)) for r in raw
             if math.isfinite(float(r.game_clock)) and r.game_clock >= 0 and r.confidence >= cfg.min_conf]
    stats["dropped_conf"] = len(raw) - len(reads)
    reads.sort(key=lambda r: r.t)
    smoothed = _mode_filter([r.period for r in reads], cfg.period_window)
    kept = [r for r, p in zip(reads, smoothed, strict=True) if r.period == p]
    keep_idx = _longest_nondecreasing([r.period for r in kept])
    stats["dropped_period"] = len(reads) - len(keep_idx)
    kept = [kept[i] for i in keep_idx]
    by_period: dict[int, list[_Read]] = {}
    for r in kept:
        by_period.setdefault(r.period, []).append(r)
    n_mono = n_rate = 0
    for p, rs in by_period.items():
        idx = _longest_nondecreasing([-r.clock for r in rs])
        n_mono += len(rs) - len(idx)
        kept_p = _trim_edges([rs[i] for i in idx], cfg)
        n_rate += len(idx) - len(kept_p)
        by_period[p] = kept_p
    stats["dropped_monotone"] = n_mono
    stats["dropped_rate"] = n_rate
    return by_period, stats


def _too_fast(a: _Read, b: _Read, cfg: TimelineConfig) -> bool:
    """The clock fell from ``a`` to ``b`` faster than real time allows (beyond one display quantum)."""
    return a.clock - b.clock > (b.t - a.t) + cfg.quantum(b.clock) + cfg.segment_tol_s


def _trim_edges(rs: list[_Read], cfg: TimelineConfig) -> list[_Read]:
    """Drop short runs at a period's first / last reads that no clock could produce.

    The monotone filter cannot see a misread at the edge of a period's reads: a high misread on the
    first reads ("5:00" before a "4:00" period-start plateau) or a low one on the last reads is still
    monotone, and on a tie the longest-subsequence pick may even prefer it. A run of fewer than
    ``min_segment_reads`` identical reads whose step to its neighbour is faster than real time is such a
    misread (a video cut at that exact read is the only legitimate cause, and it costs one short run).
    """
    def run_len(seq: Sequence[_Read], from_end: bool) -> int:
        ref = seq[-1].clock if from_end else seq[0].clock
        n = 0
        for r in (reversed(seq) if from_end else seq):
            if abs(r.clock - ref) > _EPS:
                break
            n += 1
        return n

    out = list(rs)
    while len(out) > 1:
        n = run_len(out, False)
        if n >= len(out) or n >= cfg.min_segment_reads or not _too_fast(out[n - 1], out[n], cfg):
            break
        out = out[n:]
    while len(out) > 1:
        n = run_len(out, True)
        if n >= len(out) or n >= cfg.min_segment_reads or not _too_fast(out[-n - 1], out[-n], cfg):
            break
        out = out[:-n]
    return out


def _runs_of_equal(reads: Sequence[_Read]) -> list[list[_Read]]:
    runs: list[list[_Read]] = []
    for r in reads:
        if runs and abs(runs[-1][-1].clock - r.clock) < _EPS:
            runs[-1].append(r)
        else:
            runs.append([r])
    return runs


def _intercept(r: _Read, cfg: TimelineConfig) -> float:
    return cfg.correct(r.clock) + r.t


def _fit_run(reads: Sequence[_Read], cfg: TimelineConfig) -> _Run:
    b = np.array([_intercept(r, cfg) for r in reads])
    med = float(np.median(b))
    slope = None
    ts = np.array([r.t for r in reads])
    if len(reads) >= 5 and ts[-1] - ts[0] >= 2.0:
        slope = float(np.polyfit(ts, b - ts, 1)[0])
    return _Run(b=med, t_first=reads[0].t, t_last=reads[-1].t, n=len(reads), mad=float(np.median(np.abs(b - med))), slope=slope)


def _running_segments(reads: Sequence[_Read], cfg: TimelineConfig) -> tuple[list[_Run], int]:
    """Split a running stretch where its intercept shifts (short stops, video cuts); drop transients."""
    if not reads:
        return [], 0
    segs: list[list[_Read]] = [[reads[0]]]
    pending: list[_Read] = []
    n_out = 0
    for r in reads[1:]:
        cur = segs[-1]
        ref = float(np.median([_intercept(x, cfg) for x in cur[-cfg.ref_window:]]))
        tol = cfg.quantum(r.clock) + cfg.segment_tol_s
        if abs(_intercept(r, cfg) - ref) <= tol:
            n_out += len(pending)
            pending = []
            cur.append(r)
            continue
        pending.append(r)
        pb = float(np.median([_intercept(x, cfg) for x in pending]))
        consistent = [x for x in pending if abs(_intercept(x, cfg) - pb) <= cfg.quantum(x.clock) + cfg.segment_tol_s]
        n_out += len(pending) - len(consistent)
        pending = consistent
        if len(pending) >= cfg.min_segment_reads:
            segs.append(pending)
            pending = []
    n_out += len(pending)
    good = [s for s in segs if len(s) >= cfg.min_segment_reads or len(segs) == 1]
    n_out += sum(len(s) for s in segs if not (len(s) >= cfg.min_segment_reads or len(segs) == 1))
    return [_fit_run(s, cfg) for s in good], n_out


def _pieces(reads: Sequence[_Read], cfg: TimelineConfig) -> tuple[list[_Plateau | _Run], int]:
    pieces: list[_Plateau | _Run] = []
    buf: list[_Read] = []
    n_out = 0
    dt = 1.0 / cfg.read_hz

    def flush() -> None:
        nonlocal n_out
        runs, out = _running_segments(buf, cfg)
        pieces.extend(runs)
        n_out += out
        buf.clear()

    for run in _runs_of_equal(reads):
        q = cfg.quantum(run[0].clock)
        if run[-1].t - run[0].t + dt > q + cfg.plateau_slack_s:
            flush()
            pieces.append(_Plateau(cfg.correct(run[0].clock), run[0].t, run[-1].t, len(run), q))
        else:
            buf.extend(run)
    flush()
    return pieces, n_out


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if lo > hi else min(max(v, lo), hi)


def _knots(pieces: Sequence[_Plateau | _Run], cfg: TimelineConfig) -> list[tuple[float, float]]:
    """Knots of the monotone piecewise-linear clock: plateaus meet the adjacent running lines."""
    dt = 1.0 / cfg.read_hz
    knots: list[tuple[float, float]] = []
    for i, pc in enumerate(pieces):
        prev = pieces[i - 1] if i > 0 else None
        nxt = pieces[i + 1] if i + 1 < len(pieces) else None
        if isinstance(pc, _Plateau):
            ts, te = pc.t_first, pc.t_last
            if isinstance(prev, _Run):
                ts = _clip(prev.b - pc.value, max(prev.t_last, pc.t_first - dt), pc.t_first + pc.quantum)
            if isinstance(nxt, _Run):
                te = _clip(nxt.b - pc.value, pc.t_last - pc.quantum, min(nxt.t_first, pc.t_last + dt))
            knots += [(ts, pc.value), (max(te, ts), pc.value)]
            continue
        if not isinstance(prev, _Plateau):
            knots.append((pc.t_first, pc.b - pc.t_first))
        if not isinstance(nxt, _Plateau):
            knots.append((pc.t_last, pc.b - pc.t_last))
    return knots


class ClockTimeline:
    """Monotone map between video time and (period, game clock), one knot list per period.

    Within a period the clock is piecewise linear in video time, non-increasing: slope -1 while running,
    0 on plateaus (stopped clock), linear interpolation across read gaps. Clock values live on the fitted
    line (``TimelineConfig.correct``); queries go through the same correction.
    """

    def __init__(self, knots: Mapping[int, Sequence[tuple[float, float]]], config: TimelineConfig | None = None,
                 stats: Mapping[str, Any] | None = None) -> None:
        self.config = config or TimelineConfig()
        self.stats: dict[str, Any] = dict(stats or {})
        self.exact = bool(self.stats.get("exact"))  # built from an exact feed, not from displayed values
        self._t: dict[int, np.ndarray] = {}
        self._c: dict[int, np.ndarray] = {}
        self._plateaus: dict[int, list[tuple[float, float, float]]] = {}
        for period, ks in sorted(knots.items()):
            if not ks:
                continue
            arr = np.array(sorted(ks, key=lambda k: k[0]), dtype=float)
            ts, cs = arr[:, 0], np.minimum.accumulate(arr[:, 1])
            self._t[int(period)], self._c[int(period)] = ts, cs
            self._plateaus[int(period)] = self._find_plateaus(ts, cs)

    # ------------------------------------------------------------------ construction

    @classmethod
    def fit(cls, readings: Iterable[ScoreboardReading], config: TimelineConfig | None = None) -> ClockTimeline:
        """Robust fit from noisy scoreboard reads (OCR at 5 Hz, D14).

        Drops low-confidence reads, period labels that disagree with their neighbours, and reads that break
        monotonicity (longest non-increasing subsequence per period); then splits each period into plateaus
        (a value held longer than one display quantum) and running segments (slope -1, robust intercept).
        """
        cfg = config or TimelineConfig()
        by_period, stats = _clean_reads(readings, cfg)
        knots: dict[int, list[tuple[float, float]]] = {}
        n_out = n_plat = n_run = 0
        for period, reads in sorted(by_period.items()):
            pieces, out = _pieces(reads, cfg)
            n_out += out
            n_plat += sum(isinstance(p, _Plateau) for p in pieces)
            for p in pieces:
                if isinstance(p, _Run):
                    n_run += 1
                    if p.slope is not None and p.n >= 20 and abs(p.slope + 1.0) > cfg.max_slope_dev:
                        log.warning("period %d: running segment slope %.3f (expected -1); video not real-time?", period, p.slope)
            if pieces:
                knots[period] = _knots(pieces, cfg)
        stats.update(dropped_outlier=n_out, n_plateaus=n_plat, n_running=n_run,
                     n_used=sum(len(v) for v in by_period.values()) - n_out)
        log.info("clock timeline: %s", stats)
        return cls(knots, cfg, stats)

    @classmethod
    def from_exact(cls, readings: Iterable[ScoreboardReading], config: TimelineConfig | None = None) -> ClockTimeline:
        """Timeline from an exact clock feed (simulator / official data feed): every read is a knot."""
        cfg = config or TimelineConfig(clock_display="none")
        knots: dict[int, list[tuple[float, float]]] = {}
        for r in sorted((r for r in readings if r.period is not None and r.game_clock is not None), key=lambda r: r.t):
            knots.setdefault(int(r.period), []).append((float(r.t), float(r.game_clock)))
        return cls(knots, cfg, {"n_reads": sum(len(v) for v in knots.values()), "exact": True})

    @staticmethod
    def _find_plateaus(ts: np.ndarray, cs: np.ndarray) -> list[tuple[float, float, float]]:
        out: list[tuple[float, float, float]] = []
        k = 0
        while k < len(ts) - 1:
            j = k
            while j + 1 < len(ts) and abs(cs[j + 1] - cs[k]) < _EPS:
                j += 1
            if j > k and ts[j] - ts[k] > _EPS:
                out.append((float(ts[k]), float(ts[j]), float(cs[k])))
            k = max(j, k + 1)
        return out

    # ------------------------------------------------------------------ queries

    @property
    def periods(self) -> list[int]:
        return sorted(self._t)

    def span(self, period: int) -> tuple[float, float] | None:
        ts = self._t.get(period)
        return None if ts is None else (float(ts[0]), float(ts[-1]))

    def knots(self, period: int) -> list[tuple[float, float]]:
        return [(float(t), float(c)) for t, c in zip(self._t.get(period, ()), self._c.get(period, ()), strict=True)]

    def plateaus(self, period: int) -> list[tuple[float, float, float]]:
        """(t_start, t_end, clock) of each stopped-clock interval in the period."""
        return list(self._plateaus.get(period, []))

    def clock_at(self, t: float) -> ClockPoint | None:
        """(period, clock, running) at video time ``t``; extrapolated within ``extrapolate_s`` of a period."""
        best: tuple[float, int] | None = None
        for p in self.periods:
            ts = self._t[p]
            if ts[0] - _EPS <= t <= ts[-1] + _EPS:
                return self._point_in(p, t)
            gap = ts[0] - t if t < ts[0] else t - ts[-1]
            if best is None or gap < best[0]:
                best = (gap, p)
        if best is None or best[0] > self.config.extrapolate_s:
            return None
        p = best[1]
        ts, cs = self._t[p], self._c[p]
        if t < ts[0]:
            return ClockPoint(p, float(cs[0]), False, True)
        running = len(cs) > 1 and cs[-2] - cs[-1] > _EPS
        return ClockPoint(p, float(max(cs[-1] - (t - ts[-1]) if running else cs[-1], 0.0)), running, True)

    def _point_in(self, p: int, t: float) -> ClockPoint:
        ts, cs = self._t[p], self._c[p]
        c = float(np.interp(t, ts, cs))
        k = int(np.searchsorted(ts, t, side="right")) - 1
        running = 0 <= k < len(ts) - 1 and cs[k] - cs[k + 1] > _EPS
        return ClockPoint(p, c, bool(running), False)

    def video_t(self, period: int, clock: float, where: str = "first") -> float | None:
        """Video time at which the clock shows ``clock`` ("first" / "last" instant on a plateau)."""
        ts, cs = self._t.get(period), self._c.get(period)
        if ts is None or cs is None:
            return None
        c = self.config.correct(float(clock))
        lim = self.config.extrapolate_s
        if c > cs[0] + _EPS:
            return float(ts[0] - (c - cs[0])) if c - cs[0] <= lim else None
        if c < cs[-1] - _EPS:
            return float(ts[-1] + (cs[-1] - c)) if cs[-1] - c <= lim else None
        neg = -cs
        if where == "last":
            k = int(np.searchsorted(neg, -c + _EPS, side="right")) - 1
            if k >= len(ts) - 1:
                return float(ts[-1])
            c0, c1, t0, t1 = cs[k], cs[k + 1], ts[k], ts[k + 1]
        else:
            k = int(np.searchsorted(neg, -c - _EPS, side="left"))
            if k <= 0:
                return float(ts[0])
            c0, c1, t0, t1 = cs[k - 1], cs[k], ts[k - 1], ts[k]
            if c0 - c1 < _EPS:
                return float(t1)
        if c0 - c1 < _EPS:
            return float(t0)
        return float(t0 + (c0 - c) / (c0 - c1) * (t1 - t0))

    def stop_interval(self, period: int, clock: float, tol: float = 0.0, upto: float = 0.0) -> tuple[float, float] | None:
        """(t_stop, t_restart) of the stopped-clock plateau whose value lies in [clock - tol, clock + upto + tol]
        (``upto`` > 0 matches a truncated display value against an exact feed), if there is one."""
        c = self.config.correct(float(clock))
        for ts, te, v in self._plateaus.get(period, []):
            if c - tol - _EPS <= v <= c + upto + tol + _EPS:
                return ts, te
        return None

    def support(self, period: int, t: float, radius_s: float = 3.0) -> bool:
        """True when a knot lies within ``radius_s`` of ``t`` (the map is fitted there, not interpolated)."""
        ts = self._t.get(period)
        return ts is not None and bool(np.any(np.abs(ts - t) <= radius_s))

    # ------------------------------------------------------------------ serialisation

    def to_dict(self) -> dict[str, Any]:
        return {"config": asdict(self.config), "stats": self.stats,
                "knots": {str(p): [[float(t), float(c)] for t, c in self.knots(p)] for p in self.periods}}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ClockTimeline:
        known = {f.name for f in fields(TimelineConfig)}
        cfg = TimelineConfig(**{k: v for k, v in (d.get("config") or {}).items() if k in known})
        knots = {int(p): [(float(t), float(c)) for t, c in ks] for p, ks in (d.get("knots") or {}).items()}
        return cls(knots, cfg, d.get("stats"))


def score_changes(readings: Iterable[ScoreboardReading], config: ScoreboardConfig | None = None) -> list[ScoreChange]:
    """Confirmed score changes (2 identical consecutive reads, +1/+2/+3; D14) via ``ScoreboardConfirmer``."""
    conf = ScoreboardConfirmer(config)
    out: list[ScoreChange] = []
    for r in sorted(readings, key=lambda r: r.t):
        out.extend(u for u in conf.update(r) if isinstance(u, ScoreChange))
    return out


# =============================================================================================== alignment


@dataclass
class AlignConfig:
    """PBP placement, snapping and acceptance settings (training_data §2.2 steps 4-7, §2.3)."""

    league: str = "nba"
    # PBP lags the visible event by roughly 1-2 s (BasketballBench [22]); 1.5 s is the midpoint. Per-league
    # values are unverified; the game's own lag is estimated from anchors when ``estimate_lag``.
    lag_s: dict[str, float] = field(default_factory=lambda: {
        "nba": 1.5, "fiba": 1.5, "ncaa_m": 1.5, "ncaa_w": 1.5, "nfhs": 1.5})
    estimate_lag: bool = True
    min_lag_samples: int = 5  # made FGs with an anchor before the lag estimate replaces the default; tuned
    max_lag_s: float = 4.0  # estimates are clipped to [0, max_lag_s]; tuned
    lag_search_s: float = 5.0  # anchors up to this before the clock-mapped time count for the lag estimate; tuned
    plateau_match_tol_s: float = 0.05  # a PBP clock this close to a plateau value is "on" the stoppage; tuned
    # Feed clocks are truncated display values (whole seconds, tenths in the last minute; "PT11M32.00S").
    # Only matters against an exact timeline (from_exact); against an OCR timeline both sides are display
    # values. Truncation (not rounding) is unverified.
    pbp_clock_truncated: bool = True
    score_update_lag_s: float = 0.0  # scorebug update after the ball goes in: no source (§7A.5); unverified
    score_window_s: float = 2.0  # a made shot's score delta must lie within this of its placed time; tuned
    plateau_slack_s: float = 3.0  # score deltas this far outside a stoppage still anchor its FTs; tuned
    snap_window_s: float = 2.0  # engine anchors within this of the placed time can be snapped to; tuned
    snap_player_bonus_s: float = 0.25  # cost bonus when the engine names the same player; tuned
    accept_median_residual_s: float = 0.5  # game gate: median |residual| on made FGs (training_data §5 step 7); sourced (proposal)
    accept_min_made: int = 5  # fewer anchored makes = gate undecided (rejected); tuned
    # Clip windows (seconds relative to the aligned time), training_data §2.3 (proposal; tune on the gold set).
    clip_windows: dict[str, list[float]] = field(default_factory=lambda: {
        "shot": [-3.0, 1.5], "ft": [-1.5, 1.5], "rebound": [-1.5, 2.0], "turnover": [-3.0, 1.0],
        "steal": [-3.0, 1.0], "foul": [-3.0, 1.0], "block": [-3.0, 1.5], "assist": [-5.0, 1.0], "jump_ball": [-1.0, 2.0]})
    rebound_from_miss_max_s: float = 5.0  # rebound clips start at the miss when it is at most this earlier; tuned
    # Conservative default: PBP-derived labels are research-only unless the caller holds the rights (§5.4).
    rights_status: str = "research_only"


@dataclass
class AlignedAction:
    """A PBP action placed on video time."""

    action: PBPAction
    t_pbp: float | None  # from the clock timeline (lag-corrected for live-ball actions)
    t: float | None  # final time: snapped / anchored when possible, else t_pbp
    method: str  # "clock" | "plateau_start" | "plateau_end" | "plateau_order" | "score_delta" | "snap:<kind>" | "parent" | "unplaced"
    align_conf: float = 0.0
    residual_s: float | None = None  # t_pbp - anchor time (snap or score delta)
    anchor_event_id: str | None = None
    t_release: float | None = None  # shots: release time of the snapped engine attempt
    in_plateau: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in ("t_pbp", "t", "method", "align_conf", "residual_s", "anchor_event_id", "t_release", "in_plateau")}
        d["action"] = self.action.to_row()
        return d


@dataclass
class AlignmentResult:
    aligned: list[AlignedAction]
    lag_s: float
    lag_estimated: bool
    n_lag_samples: int
    accepted: bool
    median_made_residual_s: float | None
    n_made_checked: int
    timeline: ClockTimeline | None = None

    def summary(self) -> dict[str, Any]:
        methods = Counter(a.method.split(":")[0] for a in self.aligned)
        return {"n_actions": len(self.aligned), "placed": sum(a.t is not None for a in self.aligned),
                "methods": dict(methods), "lag_s": round(self.lag_s, 3), "lag_estimated": self.lag_estimated,
                "n_lag_samples": self.n_lag_samples, "accepted": self.accepted,
                "median_made_residual_s": self.median_made_residual_s, "n_made_checked": self.n_made_checked}


def _latest(events: Iterable[Event] | None) -> list[Event]:
    latest: dict[str, Event] = {}
    for e in events or []:
        cur = latest.get(e.event_id)
        if cur is None or e.revision >= cur.revision:
            latest[e.event_id] = e
    return sorted((e for e in latest.values() if e.status is not EventStatus.RETRACTED), key=lambda e: e.t)


def _conf(method: str, timeline: ClockTimeline, a: PBPAction, t: float | None) -> float:
    """Heuristic align_conf (training_data §2.2 step 7); the numbers are tuned placeholders."""
    if t is None:
        return 0.0
    base = {"snap": 0.95, "score_delta": 0.9, "clock": 0.7, "plateau_start": 0.7, "plateau_end": 0.6,
            "parent": 0.6, "plateau_order": 0.45}.get(method.split(":")[0], 0.3)
    return round(base * (1.0 if timeline.support(a.period, t) else 0.8), 3)


def _place_plateau_group(group: list[AlignedAction], iv: tuple[float, float], changes: list[ScoreChange],
                         used: set[int], cfg: AlignConfig) -> None:
    """Order actions sharing one stoppage by actionNumber; anchor made FTs on +1 score deltas."""
    ts, te = iv
    n = len(group)
    fixed: dict[int, tuple[float, str]] = {}
    i = 0
    while i < n and group[i].action.live_ball:
        fixed[i] = (ts, "plateau_start")
        i += 1
    j = n - 1
    while j >= i and group[j].action.live_ball:
        fixed[j] = (te, "plateau_end")
        j -= 1
    for k in range(i, j + 1):
        kind = group[k].action.kind
        # The first dead-ball row opens the stoppage when it is the whistle (foul), a timeout or the period
        # start, also after leading live-ball rows (a foul on the rebounder, an and-one, a timeout after an
        # out-of-bounds turnover): it belongs at the stop, not spaced towards the next anchor.
        if kind is EventKind.PERIOD_END or (kind in (EventKind.FOUL, EventKind.TIMEOUT, EventKind.PERIOD_START) and k == i):
            fixed[k] = (ts, "plateau_start")
    last = ts - cfg.plateau_slack_s
    for k in range(i, j + 1):
        a = group[k].action
        if not (a.is_free_throw and a.made):
            continue
        for ci, ch in enumerate(changes):
            if ci in used or ch.delta != 1 or (a.team is not None and ch.team is not a.team):
                continue
            if last - _EPS <= ch.t and ts - cfg.plateau_slack_s <= ch.t <= te + cfg.plateau_slack_s:
                used.add(ci)
                fixed[k] = (ch.t - cfg.score_update_lag_s, "score_delta")
                last = ch.t
                break
    for k in range(n):
        if k in fixed:
            t, method = fixed[k]
        else:
            lo = max((x for x in fixed if x < k), default=-1)
            hi = min((x for x in fixed if x > k), default=n)
            t_lo = fixed[lo][0] if lo >= 0 else ts
            t_hi = fixed[hi][0] if hi < n else te
            t = t_lo + (k - lo) / (hi - lo) * (t_hi - t_lo)
            method = "plateau_order"
        group[k].t_pbp = group[k].t = t
        group[k].method = method
        group[k].in_plateau = True


def _stop_iv(timeline: ClockTimeline, a: PBPAction, cfg: AlignConfig) -> tuple[float, float] | None:
    upto = timeline.config.quantum(a.clock) if (timeline.exact and cfg.pbp_clock_truncated) else 0.0
    return timeline.stop_interval(a.period, a.clock, cfg.plateau_match_tol_s, upto)


def _clock_t(timeline: ClockTimeline, a: PBPAction, cfg: AlignConfig) -> float | None:
    """Video time of a PBP clock on a running clock (midpoint of a truncated value on an exact timeline)."""
    c = a.clock + (0.5 * timeline.config.quantum(a.clock) if (timeline.exact and cfg.pbp_clock_truncated) else 0.0)
    return timeline.video_t(a.period, c, "first")


def _place(actions: Sequence[PBPAction], timeline: ClockTimeline, lag: float, changes: list[ScoreChange],
           cfg: AlignConfig) -> tuple[list[AlignedAction], set[int]]:
    out = [AlignedAction(a, None, None, "unplaced") for a in actions]
    ivs = [_stop_iv(timeline, a, cfg) for a in actions]
    used: set[int] = set()
    k = 0
    while k < len(out):
        a, iv = out[k].action, ivs[k]
        if iv is None:
            t0 = _clock_t(timeline, a, cfg)
            if t0 is not None:
                out[k].t_pbp = out[k].t = t0 - (lag if a.live_ball else 0.0)
                out[k].method = "clock"
            k += 1
            continue
        j = k
        while j + 1 < len(out) and out[j + 1].action.period == a.period and ivs[j + 1] == iv:
            j += 1
        _place_plateau_group(out[k:j + 1], iv, changes, used, cfg)
        k = j + 1
    return out, used


_SNAP_KIND: dict[EventKind, EventKind] = {
    EventKind.SHOT_MADE: EventKind.SHOT_MADE, EventKind.SHOT_MISSED: EventKind.SHOT_MISSED,
    EventKind.REBOUND: EventKind.REBOUND, EventKind.TURNOVER: EventKind.TURNOVER, EventKind.STEAL: EventKind.STEAL,
    EventKind.BLOCK: EventKind.BLOCK, EventKind.ASSIST: EventKind.ASSIST, EventKind.FOUL: EventKind.FOUL,
    EventKind.POSSESSION: EventKind.POSSESSION,
}


def _is_ft_event(e: Event, attempts: Mapping[str, Event]) -> bool:
    att = attempts.get(e.shot_id or "")
    if att is not None:
        return att.value == 1 or att.attrs.get("shot_type") == "free_throw"
    return e.attrs.get("shot_type") == "free_throw" or (e.kind is EventKind.SHOT_MADE and e.value == 1)


def _compatible(a: PBPAction, e: Event, attempts: Mapping[str, Event]) -> bool:
    if _SNAP_KIND.get(a.kind) is not e.kind:  # type: ignore[arg-type]
        return False
    if a.team is not None and e.team is not None and a.team is not e.team:
        return False
    if a.is_shot:
        if a.is_free_throw != _is_ft_event(e, attempts):
            return False
        if a.kind is EventKind.SHOT_MADE and e.value not in (0, a.value):
            return False
    if a.kind is EventKind.POSSESSION:
        return e.attrs.get("cause") == a.attrs.get("cause", "jump_ball")
    return True


def _snap(aligned: list[AlignedAction], engine: Sequence[Event], cfg: AlignConfig) -> None:
    """One-to-one greedy snap of placed actions to the nearest compatible engine anchor."""
    attempts = {e.shot_id: e for e in engine if e.kind is EventKind.SHOT_ATTEMPT and e.shot_id}
    times = np.array([e.t for e in engine]) if engine else np.zeros(0)
    pairs: list[tuple[float, int, int]] = []
    for i, al in enumerate(aligned):
        if al.t_pbp is None or al.action.kind is None or al.action.synthetic or al.action.kind not in _SNAP_KIND:
            continue
        lo, hi = np.searchsorted(times, al.t_pbp - cfg.snap_window_s), np.searchsorted(times, al.t_pbp + cfg.snap_window_s, "right")
        for j in range(int(lo), int(hi)):
            e = engine[j]
            if not _compatible(al.action, e, attempts):
                continue
            cost = abs(e.t - al.t_pbp)
            if al.action.player_id is not None and al.action.player_id == e.player_id:
                cost -= cfg.snap_player_bonus_s
            pairs.append((cost, i, j))
    taken_a: set[int] = set()
    taken_e: set[int] = set()
    for _, i, j in sorted(pairs):
        if i in taken_a or j in taken_e:
            continue
        taken_a.add(i)
        taken_e.add(j)
        al, e = aligned[i], engine[j]
        al.residual_s = float(al.t_pbp - e.t)  # type: ignore[operator]
        al.t, al.method, al.anchor_event_id = float(e.t), f"snap:{e.kind.value}", e.event_id
        att = attempts.get(e.shot_id or "")
        if att is not None:
            al.t_release = float(att.attrs.get("release_t", att.t))


def _score_anchor(aligned: list[AlignedAction], changes: Sequence[ScoreChange], used: set[int], cfg: AlignConfig) -> None:
    """Made shots that were not snapped: nearest unused score delta of the right team and size."""
    for al in aligned:
        a = al.action
        if a.kind is not EventKind.SHOT_MADE or al.t_pbp is None or al.method.startswith(("snap", "score_delta")):
            continue
        best: tuple[float, int] | None = None
        for ci, ch in enumerate(changes):
            if ci in used or ch.delta != a.value or (a.team is not None and ch.team is not a.team):
                continue
            d = abs(ch.t - cfg.score_update_lag_s - al.t_pbp)
            if d <= cfg.score_window_s and (best is None or d < best[0]):
                best = (d, ci)
        if best is not None:
            used.add(best[1])
            t_anchor = changes[best[1]].t - cfg.score_update_lag_s
            al.residual_s = float(al.t_pbp - t_anchor)
            al.t, al.method = float(t_anchor), "score_delta"


def _estimate_lag(actions: Sequence[PBPAction], timeline: ClockTimeline, engine: Sequence[Event],
                  changes: Sequence[ScoreChange], cfg: AlignConfig) -> list[float]:
    """Clock-mapped time minus anchor time for made FGs on a running clock (the entry lag samples)."""
    made_engine = [e for e in engine if e.kind is EventKind.SHOT_MADE and e.value in (2, 3)]
    samples: list[float] = []
    for a in actions:
        if not (a.kind is EventKind.SHOT_MADE and a.is_field_goal) or _stop_iv(timeline, a, cfg):
            continue
        t0 = _clock_t(timeline, a, cfg)
        if t0 is None:
            continue
        lo, hi = t0 - cfg.lag_search_s, t0 + cfg.snap_window_s
        cands = [e.t for e in made_engine if lo <= e.t <= hi and (a.team is None or e.team is None or e.team is a.team)]
        if not cands:
            cands = [c.t - cfg.score_update_lag_s for c in changes
                     if lo <= c.t <= hi and c.delta == a.value and (a.team is None or c.team is a.team)]
        if cands:
            samples.append(t0 - min(cands, key=lambda x: abs(t0 - x)))
    return samples


def align_pbp(actions: Sequence[PBPAction], timeline: ClockTimeline, engine_events: Iterable[Event] | None = None,
              score_changes: Sequence[ScoreChange] | None = None, config: AlignConfig | None = None) -> AlignmentResult:
    """Place PBP actions on video time, snap them to engine anchors and run the acceptance gate."""
    cfg = config or AlignConfig()
    acts = sorted(actions, key=_order_key)
    engine = _latest(engine_events)
    changes = sorted(score_changes or [], key=lambda c: c.t)
    lag = float(cfg.lag_s.get(cfg.league, 1.5))
    samples = _estimate_lag(acts, timeline, engine, changes, cfg) if cfg.estimate_lag else []
    estimated = len(samples) >= cfg.min_lag_samples
    if estimated:
        lag = float(np.clip(np.median(samples), 0.0, cfg.max_lag_s))
    aligned, used = _place(acts, timeline, lag, changes, cfg)
    _snap(aligned, engine, cfg)
    _score_anchor(aligned, changes, used, cfg)
    _follow_parents(aligned)
    for al in aligned:
        al.align_conf = _conf(al.method, timeline, al.action, al.t)
    resid = [abs(al.residual_s) for al in aligned
             if al.action.kind is EventKind.SHOT_MADE and al.action.is_field_goal and al.residual_s is not None]
    med = float(np.median(resid)) if resid else None
    accepted = med is not None and len(resid) >= cfg.accept_min_made and med < cfg.accept_median_residual_s
    log.info("PBP alignment: lag %.2f s (%s, n=%d), %d made FGs checked, median |residual| %s -> %s",
             lag, "estimated" if estimated else "default", len(samples), len(resid),
             "n/a" if med is None else f"{med:.3f} s", "accepted" if accepted else "rejected")
    return AlignmentResult(aligned, lag, estimated, len(samples), accepted, med, len(resid), timeline)


def _follow_parents(aligned: list[AlignedAction]) -> None:
    """Assists / blocks / steals that were not snapped take their parent action's final time."""
    by_number: dict[tuple[int, int], AlignedAction] = {}
    for al in aligned:
        if not al.action.attrs.get("derived") and al.action.kind not in _CHILD_KINDS:
            by_number.setdefault((al.action.period, al.action.action_number), al)
    for al in aligned:
        parent_no = al.action.attrs.get("parent_action")
        if al.action.kind not in _CHILD_KINDS or parent_no is None or al.method.startswith("snap"):
            continue
        parent = by_number.get((al.action.period, int(parent_no)))
        if parent is not None and parent.t is not None:
            al.t, al.method = parent.t, "parent"
            al.t_release = parent.t_release


# =============================================================================================== clips


@dataclass
class LabeledClip:
    """A weakly labelled video window (training_data §2.3); one JSONL row."""

    clip_id: str
    game_id: str
    video_id: str
    t_start: float
    t_end: float
    t_anchor: float
    label: str  # e.g. "fg3_made", "ft_missed", "rebound_off", "turnover", "steal", "block", "assist", "foul"
    kind: str  # EventKind value
    team: str | None
    player_id: str | None
    player_name: str | None
    period: int
    clock: float
    action_number: int
    label_origin: str = "pbp"
    label_conf: float = 0.0
    align_method: str = ""
    residual_s: float | None = None
    rights_status: str = "research_only"
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LabeledClip:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def _label(a: PBPAction) -> tuple[str, str] | None:
    """(label, window key) for clip-worthy actions."""
    k = a.kind
    if a.is_shot:
        res = "made" if a.made else "missed"
        return (f"ft_{res}", "ft") if a.is_free_throw else (f"fg{a.value}_{res}", "shot")
    if k is EventKind.REBOUND:
        off = a.attrs.get("offensive")
        return ("rebound" if off is None else "rebound_off" if off else "rebound_def"), "rebound"
    if k is EventKind.POSSESSION and a.attrs.get("cause") == "jump_ball":
        return "jump_ball", "jump_ball"
    names = {EventKind.TURNOVER: "turnover", EventKind.STEAL: "steal", EventKind.BLOCK: "block",
             EventKind.ASSIST: "assist", EventKind.FOUL: "foul"}
    return (names[k], names[k]) if k in names else None


def make_clips(result: AlignmentResult | Sequence[AlignedAction], game_id: str, video_id: str,
               config: AlignConfig | None = None, rights_status: str | None = None) -> list[LabeledClip]:
    """Labelled clip windows for every placed, clip-worthy, non-synthetic action."""
    cfg = config or AlignConfig()
    aligned = result.aligned if isinstance(result, AlignmentResult) else list(result)
    rights = rights_status or cfg.rights_status
    clips: list[LabeledClip] = []
    last_miss_t: dict[int, float] = {}
    for al in aligned:
        a = al.action
        if a.kind is EventKind.SHOT_MISSED and al.t is not None:
            last_miss_t[a.period] = al.t
        lab = _label(a)
        if lab is None or al.t is None or a.synthetic:
            continue
        if al.t < 0.0:  # extrapolated before the first frame: the labelled moment is not in the video
            log.debug("make_clips: action %d at t=%.2f is before the video start; skipped", a.action_number, al.t)
            continue
        label, key = lab
        lo, hi = cfg.clip_windows.get(key, [-2.0, 1.0])
        start = al.t + lo
        if key == "rebound" and a.period in last_miss_t and 0.0 <= al.t - last_miss_t[a.period] <= cfg.rebound_from_miss_max_s:
            start = last_miss_t[a.period]
        attrs = {k: v for k, v in a.attrs.items() if k in ("shot_type", "offensive", "type", "ft_index", "ft_total", "cause", "technical")}
        if a.is_shot:
            attrs.update(value=a.value, made=a.made)
        if al.t_release is not None:
            attrs["t_release"] = round(al.t_release, 3)
        clips.append(LabeledClip(
            clip_id=f"c{len(clips) + 1:06d}", game_id=game_id, video_id=video_id, t_start=round(max(0.0, start), 3),
            t_end=round(al.t + hi, 3), t_anchor=round(al.t, 3), label=label, kind=a.kind.value,  # type: ignore[union-attr]
            team=a.team.value if a.team else None, player_id=a.player_id, player_name=a.player_name, period=a.period,
            clock=a.clock, action_number=a.action_number, label_conf=al.align_conf, align_method=al.method,
            residual_s=None if al.residual_s is None else round(al.residual_s, 3), rights_status=rights, attrs=attrs))
    return clips


def write_jsonl(records: Iterable[LabeledClip | AlignedAction | Mapping[str, Any]], path: str | Path) -> int:
    """Write one JSON object per line (sorted keys); returns the number of rows."""
    n = 0
    with Path(path).open("w") as fh:
        for r in records:
            d = r.to_dict() if isinstance(r, (LabeledClip, AlignedAction)) else dict(r)
            fh.write(json.dumps(d, sort_keys=True, default=str) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# =============================================================================================== pseudo-PBP


def _feed_clock(c: float, resolution: str, tenths_below_s: float) -> float:
    """Quantise like the feed: whole seconds above the last minute, tenths below ("exact" = untouched).
    "feed" truncates the whole seconds; "ceil" shows their ceiling, as the simulator's venue board and scorebug
    do (a pseudo-PBP on the board's display scale, so an OCR timeline and the PBP share one convention)."""
    if resolution == "exact":
        return float(c)
    if c >= tenths_below_s:
        return float(math.ceil(c - 1e-9) if resolution == "ceil" else math.floor(c + 1e-9))
    return math.floor(c * 10 + 1e-9) / 10.0


def pbp_from_events(events: Iterable[Event], timeline: ClockTimeline, lag_s: float = 0.0,
                    clock_resolution: str = "feed", tenths_below_s: float = 60.0) -> list[PBPAction]:
    """Pseudo-PBP from engine or simulator events (latest revisions, RETRACTED dropped).

    The clock of live-ball rows is read at ``t + lag_s`` (a scorer entering the play late); dead-ball rows
    at ``t``. ``timeline`` must map video time to the true clock (e.g. ``ClockTimeline.from_exact`` on sim
    scoreboard frames). ``clock_resolution="feed"`` floors to whole seconds above ``tenths_below_s`` and to
    tenths below. Steals, blocks and assists become their own rows with ``attrs["parent_action"]``.
    ``attrs["source_event_id"]`` keeps the originating event.
    """
    evs = _latest(events)
    attempts = {e.shot_id: e for e in evs if e.kind is EventKind.SHOT_ATTEMPT and e.shot_id}
    rows: list[PBPAction] = []
    score = {Team.HOME: 0, Team.AWAY: 0}
    trip: dict[str, Any] = {"shooter": None, "k": 0, "n": None}

    def add(e: Event, kind: EventKind, live: bool, **kw: Any) -> PBPAction | None:
        cp = timeline.clock_at(e.t + (lag_s if live else 0.0))
        if cp is None:
            log.debug("pbp_from_events: %s at t=%.2f is outside the timeline", e.event_id, e.t)
            return None
        attrs = dict(kw.pop("attrs", {}))
        attrs["source_event_id"] = e.event_id
        row = PBPAction(action_number=len(rows) + 1, period=cp.period, clock=_feed_clock(cp.clock, clock_resolution, tenths_below_s),
                        kind=kind, team=e.team, score_home=score[Team.HOME], score_away=score[Team.AWAY], attrs=attrs, **kw)
        rows.append(row)
        return row

    for e in evs:
        k = e.kind
        if k is EventKind.FOUL:
            n_ft = e.attrs.get("ft_awarded", e.attrs.get("n_ft"))
            trip.update(shooter=e.other_player_id, k=0, n=int(n_ft) if n_ft else None)
            add(e, k, False, player_id=e.player_id, attrs={"type": e.attrs.get("type", ""), "other_player_id": e.other_player_id or ""})
        elif k in _SHOT_KINDS:
            att = attempts.get(e.shot_id or "")
            src = {**(att.attrs if att is not None else {}), **e.attrs}
            value = att.value if att is not None else (e.value or 2)
            ft = value == 1 or src.get("shot_type") == "free_throw"
            attrs: dict[str, Any] = {"shot_type": "free_throw" if ft else str(src.get("shot_type") or "jumper")}
            if ft:
                if trip["shooter"] != e.player_id:
                    trip.update(shooter=e.player_id, k=0, n=None)
                trip["k"] += 1
                attrs.update(ft_index=int(src.get("ft_index", trip["k"])), ft_total=int(src.get("ft_total", trip["n"] or trip["k"])))
            if k is EventKind.SHOT_MADE and e.team is not None:
                score[e.team] += 1 if ft else value
            add(e, k, not ft, player_id=e.player_id, value=1 if ft else value, made=k is EventKind.SHOT_MADE,
                attrs={**attrs, "shot_id": e.shot_id or ""})
        elif k is EventKind.REBOUND:
            add(e, k, True, player_id=e.player_id, attrs={"offensive": bool(e.attrs.get("offensive"))})
        elif k is EventKind.TURNOVER:
            add(e, k, True, player_id=e.player_id, attrs={"type": e.attrs.get("type", "other")})
        elif k in _CHILD_KINDS:
            add(e, k, True, player_id=e.player_id, attrs={"other_player_id": e.other_player_id or "", "shot_id": e.shot_id or ""})
        elif k is EventKind.POSSESSION and e.attrs.get("cause") == "jump_ball":
            add(e, k, True, player_id=e.player_id, attrs={"cause": "jump_ball"})
        elif k in (EventKind.SUBSTITUTION, EventKind.TIMEOUT, EventKind.PERIOD_START, EventKind.PERIOD_END):
            add(e, k, False, player_id=e.player_id, attrs={"other_player_id": e.other_player_id or ""})
    _link_children(rows, {e.event_id: e for e in evs})
    _complete_ft_totals(rows)
    _finalise(rows)
    return rows


def _link_children(rows: list[PBPAction], by_id: Mapping[str, Event]) -> None:
    """Assist / block -> the shot row with the same shot_id; steal -> the turnover row at the same instant.
    Runs after all rows exist because the simulator emits STEAL / BLOCK before their parent."""
    shot_rows = {r.attrs["shot_id"]: r for r in rows if r.is_shot and r.attrs.get("shot_id")}
    turnovers = [(by_id[r.attrs["source_event_id"]].t, r) for r in rows if r.kind is EventKind.TURNOVER]
    for r in rows:
        parent: PBPAction | None = None
        if r.kind in (EventKind.ASSIST, EventKind.BLOCK):
            parent = shot_rows.get(r.attrs.get("shot_id") or "")
        elif r.kind is EventKind.STEAL and turnovers:
            t = by_id[r.attrs["source_event_id"]].t
            dt, cand = min(((abs(t0 - t), row) for t0, row in turnovers), key=lambda x: x[0])
            parent = cand if dt <= 0.5 else None
        if parent is not None:
            r.attrs["parent_action"] = parent.action_number


def _complete_ft_totals(rows: list[PBPAction]) -> None:
    """A trip without a known size: its total is the number of FTs the shooter actually took."""
    i = 0
    while i < len(rows):
        a = rows[i]
        if not a.is_free_throw or a.attrs.get("ft_index") != 1:
            i += 1
            continue
        j, trip = i, [a]
        while j + 1 < len(rows):
            b = rows[j + 1]
            if b.is_free_throw and b.player_id == a.player_id and int(b.attrs.get("ft_index", 0)) == len(trip) + 1:
                trip.append(b)
            elif b.kind not in (EventKind.SUBSTITUTION, EventKind.TIMEOUT, EventKind.FOUL) or b.is_free_throw:
                break
            j += 1
        total = max(len(trip), max(int(t.attrs.get("ft_total", 0)) for t in trip))
        for t in trip:
            t.attrs["ft_total"] = total
        i = j + 1


# =============================================================================================== evaluation


_STATS: tuple[str, ...] = ("fga", "fgm", "fg3a", "fg3m", "fta", "ftm")


@dataclass
class PBPAgreement:
    """Engine vs PBP counting stats per (period, team); diff = engine - pbp."""

    rows: list[dict[str, Any]]
    totals: dict[str, dict[str, int]]
    agreement: float  # share of (period, team, stat) cells with equal counts
    abs_error: int
    periods: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bump(table: dict[tuple[int, str, str], int], period: int, team: Team | None, stat: str) -> None:
    key = (period, team.value if team else "none", stat)
    table[key] = table.get(key, 0) + 1


def _pbp_counts(aligned: Sequence[AlignedAction]) -> dict[tuple[int, str, str], int]:
    out: dict[tuple[int, str, str], int] = {}
    for al in aligned:
        a = al.action
        if not a.is_shot or a.synthetic:
            continue
        pre = "ft" if a.is_free_throw else "fg"
        _bump(out, a.period, a.team, f"{pre}a")
        if a.made:
            _bump(out, a.period, a.team, f"{pre}m")
        if a.is_field_goal and a.value == 3:
            _bump(out, a.period, a.team, "fg3a")
            if a.made:
                _bump(out, a.period, a.team, "fg3m")
    return out


def _period_resolver(aligned: Sequence[AlignedAction], timeline: ClockTimeline | None) -> Callable[[Event], int | None]:
    spans: dict[int, list[float]] = {}
    for al in aligned:
        if al.t is not None:
            s = spans.setdefault(al.action.period, [al.t, al.t])
            s[0], s[1] = min(s[0], al.t), max(s[1], al.t)

    def resolve(e: Event) -> int | None:
        if timeline is not None:
            cp = timeline.clock_at(e.t)
            if cp is not None:
                return cp.period
        if e.attrs.get("period") is not None:
            return int(e.attrs["period"])
        if not spans:
            return None
        return min(spans, key=lambda p: 0.0 if spans[p][0] <= e.t <= spans[p][1] else min(abs(e.t - spans[p][0]), abs(e.t - spans[p][1])))
    return resolve


def _engine_counts(events: Sequence[Event], resolve: Callable[[Event], int | None]) -> dict[tuple[int, str, str], int]:
    out: dict[tuple[int, str, str], int] = {}
    attempts = {e.shot_id: e for e in events if e.kind is EventKind.SHOT_ATTEMPT and e.shot_id}
    for e in events:
        if e.kind not in (EventKind.SHOT_ATTEMPT, EventKind.SHOT_MADE):
            continue
        period = resolve(e)
        if period is None:
            continue
        att = e if e.kind is EventKind.SHOT_ATTEMPT else attempts.get(e.shot_id or "")
        value = att.value if att is not None else e.value
        ft = value == 1 or (att is not None and att.attrs.get("shot_type") == "free_throw")
        suffix = "a" if e.kind is EventKind.SHOT_ATTEMPT else "m"
        _bump(out, period, e.team, ("ft" if ft else "fg") + suffix)
        if not ft and value == 3:
            _bump(out, period, e.team, "fg3" + suffix)
    return out


def evaluate_against_pbp(engine_events: Iterable[Event], aligned: AlignmentResult | Sequence[AlignedAction],
                         timeline: ClockTimeline | None = None) -> PBPAgreement:
    """Per-period FGA / FGM / 3PA / 3PM / FTA / FTM of the engine against the aligned PBP.

    Engine events are put in a period by the clock timeline (default: the alignment's), else by
    ``attrs["period"]``, else by the video-time span of each period's aligned actions.
    """
    al_list = aligned.aligned if isinstance(aligned, AlignmentResult) else list(aligned)
    tl = timeline or (aligned.timeline if isinstance(aligned, AlignmentResult) else None)
    pbp = _pbp_counts(al_list)
    eng = _engine_counts(_latest(engine_events), _period_resolver(al_list, tl))
    periods = sorted({k[0] for k in pbp} | {k[0] for k in eng})
    teams = sorted({k[1] for k in pbp} | {k[1] for k in eng}) or ["home", "away"]
    rows: list[dict[str, Any]] = []
    totals = {s: {"engine": 0, "pbp": 0, "diff": 0} for s in _STATS}
    for p in periods:
        for team in teams:
            for s in _STATS:
                ev, pv = eng.get((p, team, s), 0), pbp.get((p, team, s), 0)
                rows.append({"period": p, "team": team, "stat": s, "engine": ev, "pbp": pv, "diff": ev - pv})
                totals[s]["engine"] += ev
                totals[s]["pbp"] += pv
                totals[s]["diff"] += ev - pv
    agree = sum(r["diff"] == 0 for r in rows) / len(rows) if rows else 1.0
    return PBPAgreement(rows, totals, round(agree, 4), sum(abs(r["diff"]) for r in rows), periods)
