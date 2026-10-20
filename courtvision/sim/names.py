"""Fictional team and player names for the simulator (never real people).

Surnames are invented words and team names are invented city + nickname pairs, so a generated
"first + last" name cannot name a real player. Rosters are drawn deterministically from a
``numpy.random.Generator``.
"""

from __future__ import annotations

import numpy as np

from courtvision.types import Roster, RosterEntry, Team

__all__ = ["FIRST_NAMES", "LAST_NAMES", "ROLES", "TEAM_NAMES", "make_roster"]

TEAM_NAMES: tuple[str, ...] = (
    "Ashgrove Herons",
    "Port Calder Lanterns",
    "Marrowind Kestrels",
    "Silvermere Owls",
    "Thornbury Stags",
    "Quillfield Otters",
    "Brackwater Moths",
    "Elderpine Badgers",
    "Galloway Reach Tides",
    "Vellmont Foxgloves",
    "Corran Bay Gulls",
    "Duskhollow Lynx",
)

FIRST_NAMES: tuple[str, ...] = (
    "Ade", "Bram", "Cael", "Dario", "Emeka", "Finnick", "Gideon", "Hollis", "Idris", "Jarek",
    "Kofi", "Lennox", "Matteo", "Nico", "Oren", "Pax", "Quentin", "Rafe", "Soren", "Tobin",
    "Ulric", "Vance", "Wilder", "Xavi", "Yusuf", "Zeke", "Anders", "Bastien", "Cyrus", "Dorian",
    "Ezra", "Florian", "Gus", "Hugo", "Ilan", "Jonah", "Kian", "Lucan", "Milo", "Nash",
)

# Invented surnames (not common family names), so generated full names stay fictional.
LAST_NAMES: tuple[str, ...] = (
    "Ardenfell", "Brindlecombe", "Corvaine", "Dellacourt", "Elsmoor", "Farrowby", "Glimmerton",
    "Halvorsk", "Iverlund", "Jessaby", "Kettleridge", "Larkspurne", "Morrowgate", "Nettlefold",
    "Oakhallow", "Pembervane", "Quarrington", "Rookwell", "Sallowmere", "Thistlewick",
    "Umberline", "Vantreese", "Wexcombe", "Yarrowfield", "Zellerby", "Ashcombe-Rae", "Blackwyn",
    "Coldharbour", "Dunmarrow", "Embervale", "Fenwhistle", "Graymantle", "Hollowell", "Inglethorn",
    "Juniperry", "Kestwood", "Lowmarsh", "Merriwether-Kai", "Northvane", "Oxenbridge",
    "Pellucid", "Quillane", "Ravensmere", "Stonebarrow", "Tidewell", "Underbrook", "Vellichor",
    "Windermark", "Yewdale", "Zephyrine",
)

# Role per roster index: starters are indices 0-4 (PG, SG, SF, PF, C order), then the bench.
ROLES: tuple[str, ...] = ("G", "G", "F", "F", "C", "G", "F", "C", "G", "F", "C", "F", "G", "F", "C")

_JERSEY_POOL: tuple[str, ...] = tuple(["0", "00"] + [str(n) for n in range(1, 56)])


def make_roster(
    rng: np.random.Generator,
    size: int = 12,
    home_name: str | None = None,
    away_name: str | None = None,
) -> Roster:
    """Two fictional rosters of ``size`` players each (10..15), ids ``"<team>-<jersey>"``.

    Entries are ordered home then away; within a team the first five are the starters in
    PG, SG, SF, PF, C order (``position`` holds the role letter G/F/C).
    """
    if not 10 <= size <= len(ROLES):
        raise ValueError(f"roster size must be in 10..{len(ROLES)}, got {size}")
    picks = rng.choice(len(TEAM_NAMES), size=2, replace=False)
    names = {
        Team.HOME: home_name or TEAM_NAMES[int(picks[0])],
        Team.AWAY: away_name or TEAM_NAMES[int(picks[1])],
    }
    used: set[str] = set()
    entries: list[RosterEntry] = []
    for team in (Team.HOME, Team.AWAY):
        jerseys = rng.choice(len(_JERSEY_POOL), size=size, replace=False)
        for i in range(size):
            jersey = _JERSEY_POOL[int(jerseys[i])]
            entries.append(
                RosterEntry(
                    player_id=f"{team.value}-{jersey}",
                    team=team,
                    jersey=jersey,
                    name=_unique_name(rng, used),
                    position=ROLES[i],
                )
            )
    return Roster(entries=entries, team_names=names)


def _unique_name(rng: np.random.Generator, used: set[str]) -> str:
    for _ in range(1000):
        name = f"{FIRST_NAMES[int(rng.integers(len(FIRST_NAMES)))]} {LAST_NAMES[int(rng.integers(len(LAST_NAMES)))]}"
        if name not in used:
            used.add(name)
            return name
    raise RuntimeError("could not draw a unique fictional name")
