"""Free-throw trips, out-of-bounds inference, periods, shot clock, foul-out and other RuleSpec rules."""

from __future__ import annotations

import pytest

from courtvision.events.engine import EventEngine
from courtvision.geometry.court import court_for
from courtvision.geometry.rules import rules_for
from courtvision.types import Event, EventKind, EventStatus, Team

from test_events_kit import MODES, Scene, kinds, live, one, run_engine, two_team_scene

TOL = 0.35


def ft_scene(**kw: object) -> Scene:
    """home-1 at the left FT line, four players set on the lane lines (NBA lane: |y - 25| = 9)."""
    s = Scene(**kw)  # type: ignore[arg-type]
    s.add("home-1", 20.0, 25.0)
    s.add("away-2", 8.0, 34.0).add("home-2", 11.0, 34.0).add("away-3", 8.0, 16.0).add("home-3", 11.0, 16.0)
    s.add("away-1", 26.0, 30.0)
    return s


def free_throw(s: Scene, made: bool, wait: float = 1.2) -> None:
    s.hold("home-1", wait)
    s.mark(f"ft{len([k for k in s.marks if k.startswith('ft')]) + 1}")
    s.shoot("left", made=made, apex_z=13.5, release_z=7.0)
    if made:
        s.drop_through("left")
        s.set_score(s.score[0] + 1, s.score[1])  # the (sim) scoreboard shows the point


def foul(t: float, fouler: str = "away-1", fouled: str = "home-1", n: int = 2, ftype: str = "shooting") -> Event:
    return Event("", EventKind.FOUL, t, int(t * 30), team=Team(fouler.split("-")[0]), player_id=fouler,
                 other_player_id=fouled, attrs={"type": ftype, "ft_awarded": n})


def run_with_foul(s: Scene, mode: str, foul_ev: Event | None, league: str = "nba", rules_overrides: list[str] | None = None) -> EventEngine:
    eng = EventEngine(court_for(league), rules=rules_for(league, rules_overrides) if rules_overrides else None)
    frames = s.frames(mode)
    if foul_ev is not None:
        eng.process(frames[0])
        eng.add_manual(foul_ev)
        frames = frames[1:]
    for fr in frames:
        eng.process(fr)
    eng.flush()
    return eng


@pytest.mark.parametrize("mode", MODES)
def test_ft_trip_made_then_missed_rebound_after_last(mode: str) -> None:
    s = ft_scene().start("home-1")
    free_throw(s, made=True)
    s.rest(0.8, visible=False)  # the referee returns the ball
    s.run(0.05, lambda a, pos: s.hands("home-1", pos))
    free_throw(s, made=False)
    s.fly(lambda pos: s.hands("away-2", pos), 0.6, apex=2.5, moves={"away-2": (7.0, 31.0)})
    s.mark("rebound")
    s.hold("away-2", 0.8)
    eng = run_with_foul(s, mode, foul(0.0))
    ev = eng.events
    atts = live(ev, EventKind.SHOT_ATTEMPT)
    assert [a.value for a in atts] == [1, 1] and all(a.attrs["shot_type"] == "free_throw" for a in atts)
    assert [a.attrs["ft_index"] for a in atts] == [1, 2] and atts[0].attrs["ft_total"] == 2
    assert abs(atts[0].t - s.marks["ft1"]) < TOL and abs(atts[1].t - s.marks["ft2"]) < TOL
    assert one(ev, EventKind.SHOT_MADE).value == 1 and one(ev, EventKind.SHOT_MADE).shot_id == atts[0].shot_id
    assert one(ev, EventKind.SHOT_MISSED).shot_id == atts[1].shot_id
    reb = one(ev, EventKind.REBOUND)
    assert reb.player_id == "away-2" and reb.attrs["offensive"] is False and reb.t > atts[1].t
    assert abs(reb.t - s.marks["rebound"]) < TOL
    away_poss = [e for e in live(ev, EventKind.POSSESSION) if e.team is Team.AWAY]
    assert len(away_poss) == 1 and away_poss[0].attrs["cause"] == "defensive_rebound"


@pytest.mark.parametrize("mode", MODES)
def test_ft_trip_missed_then_made_no_rebound_between(mode: str) -> None:
    s = ft_scene().start("home-1")
    s.add("away-4", -1.5, 30.0)  # inbounder after the last make
    free_throw(s, made=False)
    s.fly(lambda pos: s.hands("away-2", pos), 0.5, apex=1.5)  # dead ball: lane player catches it
    s.hold("away-2", 0.6)
    s.rest(0.6, visible=False)
    s.run(0.05, lambda a, pos: s.hands("home-1", pos))
    free_throw(s, made=True)
    s.fly(lambda pos: s.hands("away-4", pos), 0.4, apex=0.3).hold("away-4", 1.0)
    s.pass_to("away-1").hold("away-1", 0.5)
    eng = run_with_foul(s, mode, foul(0.0))
    ev = eng.events
    assert not live(ev, EventKind.REBOUND)
    assert not live(ev, EventKind.STEAL) and not live(ev, EventKind.TURNOVER)
    assert [e.kind for e in live(ev) if e.kind in (EventKind.SHOT_MADE, EventKind.SHOT_MISSED)] == \
        [EventKind.SHOT_MISSED, EventKind.SHOT_MADE]
    away_poss = [e for e in live(ev, EventKind.POSSESSION) if e.team is Team.AWAY]
    if mode == "3d":  # the baseline inbounder is only located reliably with true positions
        assert len(away_poss) == 1 and away_poss[0].attrs["cause"] == "free_throw"


def test_ft_unknown_trip_size_dead_retrieval_is_not_a_rebound() -> None:
    s = ft_scene().start("home-1")
    free_throw(s, made=False, wait=1.6)  # no FOUL / stoppage seen: the formation must hold 1.5 s (§7A.5 rule 2)
    s.fly(lambda pos: s.hands("away-2", pos), 0.5, apex=1.5).hold("away-2", 0.6)  # players stay set
    eng = run_engine(s.frames_3d())
    assert one(eng.events, EventKind.SHOT_ATTEMPT).attrs["shot_type"] == "free_throw"
    assert not live(eng.events, EventKind.REBOUND)


@pytest.mark.parametrize("mode", MODES)
def test_technical_ft_no_rebound_possession_retained(mode: str) -> None:
    s = ft_scene(league="nfhs").start("home-1")
    s.add("away-1", 30.0, 30.0)
    free_throw(s, made=True)
    s.rest(0.6, visible=False)
    s.run(0.05, lambda a, pos: s.hands("home-1", pos))
    free_throw(s, made=False)  # missed last technical FT: ball is dead
    s.fly(lambda pos: s.hands("away-2", pos), 0.5, apex=1.5).hold("away-2", 0.6)
    eng = run_with_foul(s, mode, foul(0.0, n=2, ftype="technical"), league="nfhs")
    ev = eng.events
    assert len(live(ev, EventKind.SHOT_ATTEMPT)) == 2
    assert all(a.attrs.get("technical") for a in live(ev, EventKind.SHOT_ATTEMPT))
    assert not live(ev, EventKind.REBOUND)
    assert not [e for e in live(ev, EventKind.POSSESSION) if e.team is Team.AWAY]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("other_team_inbounds", [True, False])
def test_out_of_bounds_inferred_from_next_inbound(mode: str, other_team_inbounds: bool) -> None:
    s = two_team_scene()
    inbounder = "away-3" if other_team_inbounds else "home-3"
    s.add(inbounder, 24.0, -2.0)  # beyond the near sideline
    s.start("home-1").hold("home-1", 0.8)
    s.fly((24.0, -3.0, 0.39), 0.6, apex=2.0)  # errant pass out over the sideline
    s.mark("out")
    s.rest(1.0, visible=False)
    s.run(0.05, lambda a, pos: s.hands(inbounder, pos))
    s.hold(inbounder, 1.5)
    receiver = "away-1" if other_team_inbounds else "home-2"
    s.pass_to(receiver).hold(receiver, 0.6)
    eng = run_engine(s.frames(mode))
    ev = eng.events
    tos = live(ev, EventKind.TURNOVER)
    if other_team_inbounds:
        assert len(tos) == 1
        to = tos[0]
        assert to.attrs["type"] == "out_of_bounds" and to.player_id == "home-1" and to.team is Team.HOME
        assert to.confidence < 1.0 and to.attrs["needs_review"] is True
        assert abs(to.t - s.marks["out"]) < 0.8
        poss = [e for e in live(ev, EventKind.POSSESSION) if e.team is Team.AWAY]
        assert len(poss) == 1 and poss[0].attrs["cause"] == "turnover"
        assert not live(ev, EventKind.STEAL)
    else:
        assert not tos
        assert eng.possession_team is Team.HOME
    p = [e for e in live(ev, EventKind.PASS) if e.player_id == inbounder]
    assert len(p) == 1 and p[0].other_player_id == receiver


def test_period_start_end_and_jump_ball() -> None:
    s = two_team_scene(clock=720.0)
    s.start(ball=(47.0, 25.0, 9.0))
    s.fly(lambda pos: s.hands("away-1", pos), 0.5, apex=1.0, moves={"away-1": (45.0, 25.0)})
    s.hold("away-1", 1.0)
    s.clock = 0.1
    s.hold("away-1", 0.1)  # the clock reaches 0.0 on the last of these frames
    s.mark("end")
    s.hold("away-1", 0.5)
    s.period, s.clock = 2, 720.0
    s.hold("away-1", 0.5)
    eng = run_engine(s.frames_3d())
    ev = live(eng.events)
    starts, ends = live(ev, EventKind.PERIOD_START), live(ev, EventKind.PERIOD_END)
    assert [e.value for e in starts] == [1, 2] and [e.value for e in ends] == [1]
    assert starts[0].t == 0.0 and abs(ends[0].t - s.marks["end"]) < 0.02 and ends[0].attrs["period"] == 1
    poss = live(ev, EventKind.POSSESSION)
    assert poss[0].attrs["cause"] == "jump_ball" and poss[0].team is Team.AWAY
    assert all(e.status is EventStatus.CONFIRMED for e in starts + ends)


def test_shot_clock_violation_is_team_turnover() -> None:
    s = two_team_scene(shot_clock=1.0)
    s.add("home-3", 18.0, -2.0)
    s.start("away-1").dribble("away-1", 1.6, 3)
    s.mark("expiry")
    s.hold("away-1", 0.5)
    s.rest(0.5, visible=False)
    s.shot_clock = 24.0
    s.run(0.05, lambda a, pos: s.hands("home-3", pos)).hold("home-3", 1.0)
    s.pass_to("home-1").hold("home-1", 0.5)
    eng = run_engine(s.frames_3d())
    to = one(eng.events, EventKind.TURNOVER)
    assert to.attrs["type"] == "shot_clock" and to.player_id == "away-1" and to.team is Team.AWAY  # last holder
    assert abs(to.t - 1.0) < 0.1
    home = [e for e in live(eng.events, EventKind.POSSESSION) if e.team is Team.HOME]
    assert len(home) == 1 and home[0].attrs["cause"] == "turnover"


def test_no_shot_clock_ruleset_never_calls_violations() -> None:
    s = two_team_scene(league="nfhs", shot_clock=1.0)
    s.start("away-1").dribble("away-1", 1.6, 3).hold("away-1", 0.5)
    eng = EventEngine(court_for("nfhs"), rules=rules_for("nfhs", ["no_shot_clock"]))
    for fr in s.frames_3d():
        eng.process(fr)
    assert not live(eng.events, EventKind.TURNOVER)


@pytest.mark.parametrize(("league", "fouls", "technicals", "out"), [
    ("nba", 5, 0, False), ("nba", 6, 0, True), ("nba", 5, 1, False),  # NBA: 6 personals; technicals not counted
    ("nfhs", 4, 1, True), ("fiba", 5, 0, True),
])
def test_foul_out_per_ruleset(league: str, fouls: int, technicals: int, out: bool) -> None:
    eng = EventEngine(court_for(league))
    for i in range(fouls):
        eng.add_manual(Event("", EventKind.FOUL, float(i), i, team=Team.AWAY, player_id="away-1",
                             other_player_id="home-1", attrs={"type": "personal"}))
    for i in range(technicals):
        eng.add_manual(Event("", EventKind.FOUL, 10.0 + i, 300 + i, team=Team.AWAY, player_id="away-1",
                             attrs={"type": "technical", "ft_awarded": 0}))
    assert ("away-1" in eng.fouled_out) is out


@pytest.mark.parametrize(("league", "period", "clock", "stops"), [
    ("nba", 1, 50.0, True), ("nba", 1, 90.0, False), ("nba", 4, 110.0, True), ("nfhs", 4, 10.0, False),
])
def test_made_fg_carries_clock_stop_rule(league: str, period: int, clock: float, stops: bool) -> None:
    s = two_team_scene(league=league, period=period, clock=clock)
    s.add("home-1", 18.0, 30.0)
    s.start("home-1").hold("home-1", 0.5).shoot("left").drop_through("left").rest(0.3)
    eng = run_engine(s.frames_3d(), league=league)
    assert one(eng.events, EventKind.SHOT_MADE).attrs["clock_stops"] is stops


def test_kinds_helper_ignores_dribbles() -> None:
    s = two_team_scene().start("home-1").dribble("home-1", 1.0, 2)
    assert kinds(run_engine(s.frames_3d()).events) == ["possession"]
