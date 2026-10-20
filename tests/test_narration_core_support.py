"""Shared fakes for the narration-core tests (no tests here).

GameStats is faked locally (duck-typed by attribute) so these tests do not depend on the stats
module, which is built in parallel. Names are fictional.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from courtvision.geometry.court import court_for
from courtvision.types import (Event, EventKind, EventStatus, FrameState, Roster, RosterEntry, ScoreboardReading,
                               Team)

HOME_NAMES = ["Avery Lindqvist", "Jordan Okafor", "Casey Marchetti", "Riley Thorsen", "Morgan Vasquez",
              "Quinn Halloran"]
AWAY_NAMES = ["Devon Achterberg", "Skyler Nakamura", "Emerson Duquesne", "Rowan Castellano", "Parker Ilunga",
              "Hayden Sorensen"]
JERSEYS = ["0", "3", "11", "23", "5", "00"]

COURT = court_for("nba")


def make_roster(home: str = "Riverside", away: str = "Lakeport") -> Roster:
    entries = [RosterEntry(f"home-{j}", Team.HOME, j, n) for j, n in zip(JERSEYS, HOME_NAMES)]
    entries += [RosterEntry(f"away-{j}", Team.AWAY, j, n) for j, n in zip(JERSEYS, AWAY_NAMES)]
    return Roster(entries, {Team.HOME: home, Team.AWAY: away})


@dataclass
class FakeLine:
    player_id: str
    team: Team | None
    pts: int = 0
    fgm: int = 0
    fga: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    oreb: int = 0
    dreb: int = 0
    ast: int = 0
    stl: int = 0
    blk: int = 0
    tov: int = 0
    pf: int = 0

    @property
    def reb(self) -> int:
        return self.oreb + self.dreb


@dataclass
class FakeTeamLine:
    team: Team
    pts: int = 0
    fgm: int = 0
    fga: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    oreb: int = 0
    dreb: int = 0
    ast: int = 0
    stl: int = 0
    blk: int = 0
    tov: int = 0
    pf: int = 0

    @property
    def reb(self) -> int:
        return self.oreb + self.dreb


@dataclass
class FakeShot:
    shot_id: str
    player_id: str
    team: Team
    t: float
    value: int
    made: bool | None


@dataclass
class FakeStats:
    t: float = 300.0
    period: int | None = 2
    game_clock: float | None = 272.4
    score: dict = field(default_factory=lambda: {Team.HOME: 42, Team.AWAY: 40})
    players: dict = field(default_factory=dict)
    teams: dict = field(default_factory=dict)
    shots: list = field(default_factory=list)
    run: dict = field(default_factory=lambda: {"team": None, "points": 0, "opp_points": 0, "since_t": None})
    score_source: str = "scoreboard"
    score_mismatch: bool = False
    shot_clock: float | None = 14.0


def make_stats(**kw: Any) -> FakeStats:
    st = FakeStats(**kw)
    if not st.players:
        st.players = {
            "home-23": FakeLine("home-23", Team.HOME, pts=12, fgm=5, fga=8, fg3m=2, fg3a=4, oreb=1, dreb=3, ast=2,
                                pf=3, tov=1),
            "home-3": FakeLine("home-3", Team.HOME, pts=6, fgm=3, fga=5, ast=4, stl=1),
            "away-11": FakeLine("away-11", Team.AWAY, pts=9, fgm=4, fga=10, dreb=5, blk=2, tov=2, pf=4),
            "away-0": FakeLine("away-0", Team.AWAY, pts=4, fgm=2, fga=6, stl=2, tov=1),
        }
    if not st.teams:
        st.teams = {Team.HOME: FakeTeamLine(Team.HOME, pts=st.score[Team.HOME], fgm=17, fga=36, tov=6, pf=8, dreb=14),
                    Team.AWAY: FakeTeamLine(Team.AWAY, pts=st.score[Team.AWAY], fgm=16, fga=38, tov=9, pf=10, dreb=12)}
    return st


def make_frame(stats: FakeStats, source: str = "sim") -> FrameState:
    sb = ScoreboardReading(t=stats.t, home_score=stats.score[Team.HOME], away_score=stats.score[Team.AWAY],
                           period=stats.period, game_clock=stats.game_clock, shot_clock=stats.shot_clock, source=source)
    return FrameState(frame_idx=int(stats.t * 30), t=stats.t, players=[], ball=None, scoreboard=sb)


_ids = itertools.count(1)


def ev(kind: EventKind, *, status: EventStatus = EventStatus.CONFIRMED, source: str = "cv", **kw: Any) -> Event:
    kw.setdefault("event_id", f"e{next(_ids):06d}")
    kw.setdefault("t", 299.0)
    kw.setdefault("frame_idx", 8970)
    return Event(kind=kind, status=status, source=source, **kw)


def shot_made(value: int = 3, pid: str = "home-23", team: Team = Team.HOME, shot_type: str = "jumper",
              x: float = 30.0, y: float = 8.0, **kw: Any) -> Event:
    attrs = {"shot_type": shot_type, "basket": "left", "distance_ft": 24.6}
    attrs.update(kw.pop("attrs", {}))
    return ev(EventKind.SHOT_MADE, team=team, player_id=pid, value=value, shot_id="s0042", x=x, y=y, attrs=attrs, **kw)


def scenario_events() -> dict[str, list[Event]]:
    """One event batch per template family the play-by-play planner can choose."""
    H, A = Team.HOME, Team.AWAY
    return {
        "made3": [shot_made(3)],
        "made2": [shot_made(2, x=15.0, y=20.0)],
        "layup": [shot_made(2, shot_type="layup", x=7.0, y=25.0)],
        "dunk": [shot_made(2, shot_type="dunk", x=5.5, y=25.0)],
        "ft_made": [shot_made(1, shot_type="free_throw", x=19.0, y=25.0)],
        "made_assisted": [shot_made(3), ev(EventKind.ASSIST, team=H, player_id="home-3", other_player_id="home-23",
                                           shot_id="s0042")],
        "miss": [ev(EventKind.SHOT_MISSED, team=A, player_id="away-11", x=60.0, y=40.0,
                    attrs={"shot_type": "jumper", "basket": "right"})],
        "ft_miss": [ev(EventKind.SHOT_MISSED, team=A, player_id="away-11", value=1, attrs={"shot_type": "free_throw"})],
        "block": [ev(EventKind.BLOCK, team=A, player_id="away-11", other_player_id="home-23", x=8.0, y=24.0,
                     attrs={"basket": "left"})],
        "steal": [ev(EventKind.STEAL, team=A, player_id="away-0", other_player_id="home-3", x=40.0, y=30.0),
                  ev(EventKind.TURNOVER, team=H, player_id="home-3", attrs={"type": "steal"})],
        "turnover": [ev(EventKind.TURNOVER, team=H, player_id="home-23", attrs={"type": "travel"})],
        "dreb": [ev(EventKind.REBOUND, team=A, player_id="away-11", attrs={"offensive": False})],
        "oreb": [ev(EventKind.REBOUND, team=H, player_id="home-23", attrs={"offensive": True})],
        "team_reb": [ev(EventKind.REBOUND, team=A, attrs={"offensive": False})],
        "release3": [ev(EventKind.SHOT_ATTEMPT, team=H, player_id="home-23", value=3, x=30.0, y=8.0,
                        attrs={"shot_type": "jumper", "basket": "left"})],
        "release": [ev(EventKind.SHOT_ATTEMPT, team=H, player_id="home-23", value=2, x=15.0, y=20.0,
                       attrs={"shot_type": "jumper", "basket": "left"})],
        "ft_attempt": [ev(EventKind.SHOT_ATTEMPT, team=H, player_id="home-23", value=1,
                          attrs={"shot_type": "free_throw"})],
        "possession": [ev(EventKind.POSSESSION, team=A, player_id="away-0", attrs={"cause": "steal"})],
        "bring_up": [ev(EventKind.POSSESSION, team=A, player_id="away-0", attrs={"cause": "made_shot"})],
        "foul": [ev(EventKind.FOUL, team=H, player_id="home-23", other_player_id="away-11", source="manual")],
        "timeout": [ev(EventKind.TIMEOUT, team=A, source="manual")],
        "substitution": [ev(EventKind.SUBSTITUTION, team=H, player_id="home-5", other_player_id="home-3",
                            source="manual")],
        "period_start": [ev(EventKind.PERIOD_START, source="scoreboard")],
        "period_end": [ev(EventKind.PERIOD_END, source="scoreboard")],
    }
