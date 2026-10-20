"""Shots, make / miss, value (2 vs 3), rebounds, blocks and assists in both observation modes."""

from __future__ import annotations

import math

import pytest

from courtvision.events.engine import EventEngineConfig
from courtvision.types import EventKind, EventStatus, Team

from test_events_kit import MODES, Scene, kinds, live, one, run_engine, two_team_scene

TOL = 0.35


def shooter_scene(x: float, y: float, **kw: object) -> Scene:
    s = two_team_scene(**kw)
    s.add("home-1", x, y).add("away-1", x + 5.0, y + 6.0)
    return s


def made_shot(s: Scene, side: str = "left", shooter: str = "home-1") -> Scene:
    s.start(shooter).hold(shooter, 0.8)
    s.mark("release")
    s.shoot(side, made=True)
    s.mark("made")
    return s.drop_through(side).rest(0.6)


@pytest.mark.parametrize("mode", MODES)
def test_made_two(mode: str) -> None:
    s = made_shot(shooter_scene(18.0, 30.0))
    eng = run_engine(s.frames(mode))
    att, made = one(eng.events, EventKind.SHOT_ATTEMPT), one(eng.events, EventKind.SHOT_MADE)
    assert att.value == 2 and made.value == 2 and att.shot_id == made.shot_id
    assert att.player_id == made.player_id == "home-1" and att.team is Team.HOME
    assert abs(att.t - s.marks["release"]) < TOL and abs(made.t - s.marks["made"]) < TOL
    assert att.attrs["basket"] == "left" and att.attrs["shot_type"] == "jumper"
    assert att.attrs["p_status"] == "uncalibrated" and 0.0 < att.confidence < 1.0
    for key in ("crossed_inside", "drop_in_band", "drop_is_freefall", "occluded_frames", "rim_contact"):
        assert key in made.attrs["cues"]
    assert made.attrs["cues"]["crossed_inside"] is True
    assert not live(eng.events, EventKind.SHOT_MISSED) and not live(eng.events, EventKind.REBOUND)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("name", "dx", "lateral", "value"), [
    ("corner", 3.0, 22.6, 3),  # straight corner segment, beyond 22 ft
    ("arc", 24.8, 0.0, 3),  # top of the key, beyond 23.75 ft
    ("on_the_line_arc", 23.75, 0.0, 2),  # exactly on the arc: the line is 2 (D8)
    ("on_the_line_corner", 2.0, 22.0, 2),  # exactly on the corner line
])
def test_shot_value_from_gather_position(mode: str, name: str, dx: float, lateral: float, value: int) -> None:
    s = Scene()
    bx, by = s.court.basket_xy("left")
    s.add("home-1", bx + dx, by - lateral).add("away-1", bx + dx + 6.0, by)
    made_shot(s)
    eng = run_engine(s.frames(mode))
    att = one(eng.events, EventKind.SHOT_ATTEMPT)
    if mode == "3d":
        assert att.value == value, name
        assert one(eng.events, EventKind.SHOT_MADE).value == value
    else:
        # Monocular players are located by the bbox foot point, which this kit's cylinder model puts
        # PLAYER_R (0.9 ft) in front of the true centre; only cases clear of that bias are asserted.
        corner = dx <= s.court.three_junction_along
        margin = abs(lateral) - 22.0 if corner else math.hypot(dx, lateral) - 23.75
        if abs(margin) > 1.0:
            assert att.value == value, name
    assert 0.0 <= att.attrs["p_three"] <= 1.0
    assert att.attrs["zone"] in ("corner_three", "above_break_three", "midrange")


@pytest.mark.parametrize("mode", MODES)
def test_miss_then_defensive_rebound(mode: str) -> None:
    s = shooter_scene(20.0, 25.0)
    s.add("away-3", 9.0, 25.0)
    s.start("home-1").hold("home-1", 0.8).shoot("left", made=False)
    s.mark("rim")
    s.fly(lambda pos: s.hands("away-3", pos), 0.6, apex=3.0)
    s.mark("rebound")
    s.hold("away-3", 0.8)
    eng = run_engine(s.frames(mode))
    ev = eng.events
    miss, reb = one(ev, EventKind.SHOT_MISSED), one(ev, EventKind.REBOUND)
    assert miss.player_id == "home-1" and miss.shot_id == one(ev, EventKind.SHOT_ATTEMPT).shot_id
    assert s.marks["rim"] - 0.1 <= miss.t <= s.marks["rebound"] + TOL
    assert reb.player_id == "away-3" and reb.attrs["offensive"] is False and reb.team is Team.AWAY
    assert abs(reb.t - s.marks["rebound"]) < TOL
    pos = [e for e in live(ev, EventKind.POSSESSION) if e.team is Team.AWAY]
    assert len(pos) == 1 and pos[0].attrs["cause"] == "defensive_rebound" and abs(pos[0].t - reb.t) < 1e-9
    assert eng.possession_team is Team.AWAY


@pytest.mark.parametrize("mode", MODES)
def test_miss_then_offensive_rebound(mode: str) -> None:
    s = shooter_scene(20.0, 25.0)
    s.add("home-3", 9.0, 27.0)
    s.start("home-1").hold("home-1", 0.8).shoot("left", made=False)
    s.fly(lambda pos: s.hands("home-3", pos), 0.6, apex=3.0).hold("home-3", 0.8)
    eng = run_engine(s.frames(mode))
    reb = one(eng.events, EventKind.REBOUND)
    assert reb.player_id == "home-3" and reb.attrs["offensive"] is True
    assert kinds(eng.events) == ["possession", "shot_attempt", "shot_missed", "rebound"]  # no POSSESSION for an OREB
    assert eng.possession_team is Team.HOME


@pytest.mark.parametrize("mode", MODES)
def test_team_rebound_when_ball_goes_out_before_control(mode: str) -> None:
    s = shooter_scene(20.0, 25.0)
    s.add("away-3", 6.0, 52.5)  # will inbound from beyond the far sideline
    s.start("home-1").hold("home-1", 0.8).shoot("left", made=False, rim_offset=-0.95)
    s.fly((6.0, 47.0, 3.0), 0.6, apex=2.0)
    s.fly((6.0, 52.0, 0.39), 0.4, apex=0.5)  # out over the sideline
    s.mark("out")
    s.rest(1.0, visible=False)
    s.run(0.1, lambda a, pos: s.hands("away-3", pos))
    s.hold("away-3", 1.5)
    s.pass_to("away-1", moves={"away-1": (14.0, 40.0)}).hold("away-1", 0.8)
    eng = run_engine(s.frames(mode))
    reb = one(eng.events, EventKind.REBOUND)
    assert reb.player_id is None and reb.team is Team.AWAY and reb.attrs["offensive"] is False
    assert reb.attrs["team_rebound"] is True and reb.confidence < 1.0
    assert reb.t <= s.marks["out"] + 0.5
    assert not live(eng.events, EventKind.TURNOVER)
    pos = [e for e in live(eng.events, EventKind.POSSESSION) if e.team is Team.AWAY]
    assert len(pos) == 1 and pos[0].attrs["cause"] == "defensive_rebound"


@pytest.mark.parametrize("mode", MODES)
def test_block(mode: str) -> None:
    s = Scene()
    s.add("home-1", 14.0, 25.0).add("away-1", 12.6, 24.3).add("away-3", 18.0, 30.0)
    s.start("home-1").hold("home-1", 0.8)
    s.mark("release")
    s.fly((12.9, 24.3, 10.0), 0.3, apex=0.4)  # rising fast toward the rim
    s.mark("block")
    s.fly((18.0, 29.3, 4.0), 0.5, apex=0.5)  # swatted back out
    s.hold("away-3", 0.8)
    eng = run_engine(s.frames(mode))
    ev = eng.events
    blk, att, miss = one(ev, EventKind.BLOCK), one(ev, EventKind.SHOT_ATTEMPT), one(ev, EventKind.SHOT_MISSED)
    assert blk.player_id == "away-1" and blk.other_player_id == "home-1" and blk.shot_id == att.shot_id == miss.shot_id
    assert abs(blk.t - s.marks["block"]) < TOL and abs(att.t - s.marks["release"]) < TOL
    reb = one(ev, EventKind.REBOUND)
    assert reb.player_id == "away-3" and reb.attrs["offensive"] is False
    assert not live(ev, EventKind.STEAL)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("hold_s", "bounces", "assisted"), [
    (1.0, 0, True),
    (1.5, 2, True),  # 2 dribbles allowed
    (1.5, 3, False),  # too many dribbles
    (5.4, 0, False),  # catch -> release > 5.0 s
])
def test_assist_rule(mode: str, hold_s: float, bounces: int, assisted: bool) -> None:
    s = shooter_scene(32.0, 25.0)
    s.add("home-2", 18.0, 32.0)
    s.start("home-1").hold("home-1", 0.6).pass_to("home-2")
    if bounces:
        s.dribble("home-2", hold_s, bounces)
    else:
        s.hold("home-2", hold_s)
    s.hold("home-2", 0.2)
    s.shoot("left", made=True).drop_through("left").rest(0.5)
    eng = run_engine(s.frames(mode))
    made = one(eng.events, EventKind.SHOT_MADE)
    ast = live(eng.events, EventKind.ASSIST)
    if assisted:
        assert len(ast) == 1
        a = ast[0]
        assert (a.player_id, a.other_player_id, a.shot_id, a.t) == ("home-1", "home-2", made.shot_id, made.t)
    else:
        assert not ast
    att = one(eng.events, EventKind.SHOT_ATTEMPT)
    assert att.attrs["dribbles_before"] == bounces
    assert att.attrs["catch_and_shoot"] is (bounces == 0)


@pytest.mark.parametrize("mode", MODES)
def test_unassisted_after_own_rebound(mode: str) -> None:
    s = shooter_scene(20.0, 25.0)
    s.add("home-3", 9.0, 27.0)
    s.start("home-1").hold("home-1", 0.8).shoot("left", made=False)
    s.fly(lambda pos: s.hands("home-3", pos), 0.6, apex=3.0).hold("home-3", 0.6)
    s.shoot("left", made=True, apex_z=12.0).drop_through("left").rest(0.5)
    eng = run_engine(s.frames(mode))
    assert len(live(eng.events, EventKind.SHOT_ATTEMPT)) == 2
    assert not live(eng.events, EventKind.ASSIST)
    second = live(eng.events, EventKind.SHOT_ATTEMPT)[1]
    assert second.player_id == "home-3" and second.attrs["shot_type"] in ("layup", "jumper", "tip")


def test_made_shot_dead_ball_then_inbound_possession() -> None:
    s = shooter_scene(18.0, 30.0)
    s.add("away-3", -1.5, 28.0)  # inbounder behind the baseline
    made_shot(s)
    s.set_score(2, 0)
    s.fly(lambda pos: s.hands("away-3", pos), 0.4, apex=0.3).hold("away-3", 1.5)
    s.mark("inbound")
    s.pass_to("away-1").hold("away-1", 0.6)
    eng = run_engine(s.frames_3d())
    ev = eng.events
    pos = [e for e in live(ev, EventKind.POSSESSION) if e.team is Team.AWAY]
    assert len(pos) == 1 and pos[0].attrs["cause"] == "made_shot" and pos[0].player_id == "away-3"
    assert not live(ev, EventKind.REBOUND) and not live(ev, EventKind.STEAL)
    p = one(ev, EventKind.PASS)
    assert p.player_id == "away-3" and abs(p.t - s.marks["inbound"]) < TOL
    made = one(ev, EventKind.SHOT_MADE)
    assert made.status is EventStatus.CONFIRMED  # sim scoreboard delta matched


def test_attacking_side_inferred_and_switches_at_halftime() -> None:
    cfg = EventEngineConfig(default_attacking={"home": "right", "away": "left"})
    s = shooter_scene(18.0, 30.0, period=1)
    made_shot(s)
    eng = run_engine(s.frames_3d(), cfg)
    assert eng.attacking_side == {Team.HOME: "left", Team.AWAY: "right"}
    eng._period = 3  # second half, no evidence there yet: flipped first-half evidence
    assert eng.attacking_side == {Team.HOME: "right", Team.AWAY: "left"}
