"""Roster files: load / save the team and player list a video or camera game is played with.

The documented format (``docs/examples/roster.yaml``) nests the players under their team::

    teams:
      home:
        name: Riverside Hawks
        players:
          - {jersey: "23", name: Alex Moreno, position: G}
          - {jersey: "00", name: Sam Okafor, position: C}
      away:
        name: Lakeview Comets
        players:
          - {jersey: "7", name: Jordan Reyes}

``player_id`` is optional and defaults to ``"<team>-<jersey>"`` (the simulator's id scheme), so
event files from a simulator render and from a roster file use the same ids. The older flat form
``{"teams": {"home": "Name", "away": "Name"}, "players": [{"team": "home", "jersey": ...}]}`` is
still accepted, in YAML or JSON.

Jersey numbers are strings: ``"0"`` and ``"00"`` are different numbers (``types.RosterEntry``). YAML
would read an unquoted ``00`` as the integer 0 and ``07`` as 7, so roster YAML is parsed with every
plain scalar kept as a string (only ``null`` / ``~`` stay null). Duplicate ids and duplicate
(team, jersey) pairs are errors, so identity voting can never map one number to two players.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from courtvision.types import Roster, RosterEntry, Team

__all__ = ["RosterError", "load_roster", "parse_roster", "roster_to_dict", "roster_to_yaml", "save_roster"]

log = logging.getLogger(__name__)


class RosterError(ValueError):
    """A malformed roster file (the message names the file and the entry)."""


class _StringLoader(yaml.SafeLoader):
    """SafeLoader whose plain scalars stay strings (jersey ``00`` != ``0``), except null."""


_StringLoader.yaml_implicit_resolvers = {
    first: [(tag, rx) for tag, rx in resolvers if tag == "tag:yaml.org,2002:null"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _team(value: Any, where: str) -> Team:
    try:
        return Team(str(value).strip().lower())
    except ValueError as exc:
        raise RosterError(f"{where}: team must be 'home' or 'away', got {value!r}") from exc


def _jersey(value: Any, where: str) -> str:
    if value is None or isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise RosterError(f"{where}: jersey must be a number like \"23\" or \"00\", got {value!r}")
    text = str(int(value)) if isinstance(value, (int, float)) else str(value).strip()
    if not text:
        raise RosterError(f"{where}: empty jersey")
    if not text.isdigit():
        log.warning("%s: jersey %r is not all digits (jersey readers only read digits)", where, text)
    return text


def _entry(row: Any, team: Team | None, where: str) -> RosterEntry:
    if not isinstance(row, Mapping):
        raise RosterError(f"{where}: expected a mapping with jersey and name, got {type(row).__name__}")
    known = {"player_id", "team", "jersey", "name", "position"}
    unknown = sorted(str(k) for k in row if k not in known)
    if unknown:
        raise RosterError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(known)}")
    if team is None:
        if row.get("team") is None:
            raise RosterError(f"{where}: a flat player list needs 'team' on every entry")
        team = _team(row["team"], where)
    elif row.get("team") is not None and _team(row["team"], where) is not team:
        raise RosterError(f"{where}: listed under {team.value} but says team {row['team']!r}")
    if "jersey" not in row:
        raise RosterError(f"{where}: missing jersey")
    jersey = _jersey(row["jersey"], where)
    name = row.get("name")
    name = str(name).strip() if name is not None and str(name).strip() else f"#{jersey}"
    pid = row.get("player_id")
    pid = str(pid).strip() if pid is not None and str(pid).strip() else f"{team.value}-{jersey}"
    position = row.get("position")
    position = str(position).strip() if position is not None and str(position).strip() else None
    return RosterEntry(player_id=pid, team=team, jersey=jersey, name=name, position=position)


def parse_roster(data: Any, source: str = "roster") -> Roster:
    """Build a :class:`Roster` from parsed YAML / JSON (nested or flat form, see module docstring)."""
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise RosterError(f"{source}: a roster must be a mapping with 'teams', got {type(data).__name__}")
    unknown = sorted(str(k) for k in data if k not in ("teams", "players"))
    if unknown:
        raise RosterError(f"{source}: unknown top-level key(s) {unknown}; expected 'teams' (and 'players')")
    names = {Team.HOME: "Home", Team.AWAY: "Away"}
    entries: list[RosterEntry] = []
    teams = data.get("teams") or {}
    if not isinstance(teams, Mapping):
        raise RosterError(f"{source}: 'teams' must map home / away to a team")
    for side, spec in teams.items():
        team = _team(side, f"{source}: teams")
        where = f"{source}: teams.{team.value}"
        if isinstance(spec, Mapping):
            bad = sorted(str(k) for k in spec if k not in ("name", "players"))
            if bad:
                raise RosterError(f"{where}: unknown key(s) {bad}; expected 'name' and 'players'")
            if spec.get("name") is not None and str(spec["name"]).strip():
                names[team] = str(spec["name"]).strip()
            players = spec.get("players") or []
            if not isinstance(players, list):
                raise RosterError(f"{where}.players must be a list")
            entries += [_entry(row, team, f"{where}.players[{i}]") for i, row in enumerate(players)]
        elif spec is not None:  # flat form: teams: {home: "Name"}
            names[team] = str(spec).strip() or names[team]
    flat = data.get("players") or []
    if not isinstance(flat, list):
        raise RosterError(f"{source}: 'players' must be a list")
    entries += [_entry(row, None, f"{source}: players[{i}]") for i, row in enumerate(flat)]
    seen_ids: dict[str, str] = {}
    seen_numbers: dict[tuple[Team, str], str] = {}
    for e in entries:
        key = (e.team, e.jersey)
        if key in seen_numbers:
            raise RosterError(f"{source}: {e.team.value} lists jersey {e.jersey!r} twice "
                              f"({seen_numbers[key]!r} and {e.name!r})")
        seen_numbers[key] = e.name
        if e.player_id in seen_ids:
            raise RosterError(f"{source}: duplicate player_id {e.player_id!r}")
        seen_ids[e.player_id] = e.name
    return Roster(entries=entries, team_names=names)


def load_roster(path: str | Path | None, home_name: str | None = None, away_name: str | None = None) -> Roster:
    """Roster from a YAML / JSON file (an empty roster when ``path`` is None); team display names
    given here win over the file's."""
    if path is None:
        roster = Roster(entries=[], team_names={Team.HOME: "Home", Team.AWAY: "Away"})
    else:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"roster file not found: {p}")
        text = p.read_text(encoding="utf-8")
        try:
            data = json.loads(text) if p.suffix.lower() == ".json" else yaml.load(text, Loader=_StringLoader)  # noqa: S506 - SafeLoader subclass
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            raise RosterError(f"{p}: cannot parse roster: {exc}") from exc
        roster = parse_roster(data, str(p))
    if home_name:
        roster.team_names[Team.HOME] = home_name
    if away_name:
        roster.team_names[Team.AWAY] = away_name
    return roster


def roster_to_dict(roster: Roster) -> dict[str, Any]:
    """The nested file form (``player_id`` written only when it is not ``"<team>-<jersey>"``)."""
    teams: dict[str, Any] = {}
    for team in (Team.HOME, Team.AWAY):
        players = []
        for e in roster.entries:
            if e.team is not team:
                continue
            row: dict[str, Any] = {"jersey": e.jersey, "name": e.name}
            if e.position:
                row["position"] = e.position
            if e.player_id != f"{team.value}-{e.jersey}":
                row["player_id"] = e.player_id
            players.append(row)
        teams[team.value] = {"name": roster.team_names.get(team, team.value.title()), "players": players}
    return {"teams": teams}


def roster_to_yaml(roster: Roster) -> str:
    return yaml.safe_dump(roster_to_dict(roster), sort_keys=False, allow_unicode=True, default_flow_style=False)


def save_roster(roster: Roster, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".json":
        p.write_text(json.dumps(roster_to_dict(roster), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        p.write_text(roster_to_yaml(roster), encoding="utf-8")
    return p
