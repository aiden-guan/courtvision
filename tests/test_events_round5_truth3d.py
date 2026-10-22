"""Regression tests for the round-5 truth3d event-engine fixes.

3-D scenes (the kit in ``test_events_kit.py``) for each rule: a block whose touch falls early in a frame step, a
block at the ring told from a ring bounce by the ball's paths into and out of the contact, dead balls a dribble does
not restart (after a violation, between free throws), a throw-in in the air treated as a pass, a high catch of a loose
ball, a tip while a shot that touched the ring is still unresolved, a try at the shooter's own basket re-credited, a
rebound secured by the period-end buzzer, and a pass thrown out of bounds as an incomplete pass. Where a scene could
pass for an unrelated reason, a contrast run with the rule switched off through its config field shows that the rule
is what decides.
"""

from __future__ import annotations

import math

import pytest

from courtvision.events.engine import EventEngineConfig
from courtvision.events.shots import FlightSample, _reversal
from courtvision.types import EventKind, Team

from test_events_kit import BALL_R, Scene, live, one, run_engine, two_team_scene
from test_events_rules import foul, free_throw, ft_scene, run_with_foul

G = 32.174
TOL = 0.35
Vec3 = tuple[float, float, float]


def _ballistic(p: Vec3, v: Vec3, dt: float) -> Vec3:
    return (p[0] + v[0] * dt, p[1] + v[1] * dt, p[2] + v[2] * dt - 0.5 * G * dt * dt)


def _touched_flight(s: Scene, v_in: Vec3, t_touch: float, v_out: Vec3, duration: float) -> Vec3:
    """Free flight from the current ball position with ``v_in``; a touch ``t_touch`` s later (between frames) sends it
    on with ``v_out``. Returns the contact point."""
    p0 = s.ball
    assert p0 is not None
    c = _ballistic(p0, v_in, t_touch)
    s.marks["touch"] = s.t + t_touch

    def ball(a: float, pos: dict[str, tuple[float, float]]) -> Vec3:
        dt = a * duration
        return _ballistic(p0, v_in, dt) if dt <= t_touch else _ballistic(c, v_out, dt - t_touch)

    s.run(duration, ball)
    return c


def _to_hands(s: Scene, pid: str, duration: float) -> Scene:
    e = s.ball
    assert e is not None
    return s.run(duration, lambda a, pos: tuple(e[i] + a * (s.hands(pid, pos)[i] - e[i]) for i in range(3)))  # type: ignore[return-value]


def _shot_in(s: Scene, side: str, T: float) -> Scene:
    """Free flight (gravity) from the current ball position into ``side``'s basket centre at rim height in ``T`` s,
    then down through the net."""
    p0 = s.ball
    assert p0 is not None
    bx, by, bz = s.basket(side)
    v = ((bx - p0[0]) / T, (by - p0[1]) / T, (bz - p0[2] + 0.5 * G * T * T) / T)
    s.run(T, lambda a, pos: _ballistic(p0, v, a * T))
    return s.drop_through(side)


def _raise(s: Scene, top: Vec3, duration: float) -> Scene:
    st = s.ball
    assert st is not None
    return s.run(duration, lambda a, pos: tuple(st[i] + a * (top[i] - st[i]) for i in range(3)))  # type: ignore[return-value]


# ---------------------------------------------------------------------------- block: touch early in a frame step


def _early_touch_block() -> Scene:
    """home-1's jumper (released at 8 ft, rising at 14 ft/s toward the left basket) is turned back by away-1's hand
    0.35 of the way through its fourth frame step: that step's chord barely moves toward the rim (+1.7 ft/s) and the
    two steps across it still average away from it by less than the block test asks (-1.7 ft/s)."""
    s = Scene()
    s.add("home-1", 16.0, 25.0).add("away-1", 14.5, 24.4).add("away-3", 24.0, 30.0).add("home-2", 30.0, 40.0)
    s.start("home-1").dribble("home-1", 1.0, 2)
    _raise(s, (16.0, 24.6, 8.0), 0.3)
    s.mark("release")
    _touched_flight(s, (-14.0, 0.0, 14.0), 3.35 / 30.0, (5.0, 1.0, 1.0), 0.6)
    _to_hands(s, "away-3", 0.5).hold("away-3", 1.0)
    return s


def test_block_touched_early_in_a_frame_step_is_a_block_3d() -> None:
    s = _early_touch_block()
    ev = live(run_engine(s.frames_3d()).events)
    blk = one(ev, EventKind.BLOCK)
    assert (blk.player_id, blk.other_player_id) == ("away-1", "home-1") and abs(blk.t - s.marks["touch"]) <= 0.1
    att = one(ev, EventKind.SHOT_ATTEMPT)
    assert att.player_id == "home-1" and abs(att.t - s.marks["release"]) <= 0.1
    assert one(ev, EventKind.SHOT_MISSED).attrs.get("blocked") is True
    assert one(ev, EventKind.REBOUND).player_id == "away-3"
    assert not live(ev, EventKind.PASS) and not live(ev, EventKind.STEAL)


def test_reversal_leaving_the_contact_step_out() -> None:
    """The velocity tests on the samples around that touch: neither the consecutive steps nor the two steps across
    the contact show the reversal; the clean steps either side of the contact step do."""
    s = _early_touch_block()
    tc = s.marks["touch"]
    k = next(i for i, r in enumerate(s.recs) if r.t > tc)  # first sample after the touch

    def smp(i: int) -> FlightSample:
        r = s.recs[i]
        assert r.ball is not None
        return FlightSample(r.t, i, None, (r.ball[0], r.ball[1]), r.ball[2], None)

    b = s.court.basket_xy("left")
    pm, p0, q, p2, p3 = (smp(i) for i in (k - 2, k - 1, k, k + 1, k + 2))  # contact between p0 and q
    assert _reversal(p0, q, p2, 10.0, b, None, 3.0, p3) is None  # consecutive, vertex before the touch
    assert _reversal(pm, p0, q, 10.0, b, None, 3.0, p2) is None  # consecutive, vertex after the touch
    assert _reversal(pm, p0, p2, 10.0, b, None, 3.0, p3) is None  # across the contact step
    dv = _reversal(pm, p0, p2, 10.0, b, None, 3.0, p3, after_from=q)
    assert dv is not None and dv >= EventEngineConfig().block_reverse_dv_fts


# ---------------------------------------------------------------------------- block at the ring vs ring bounce


def _ring_block() -> tuple[Scene, Vec3]:
    """home-1's layup drifts over the front of the left ring, its centre 0.63 ft from the ring (a ball radius plus
    0.24 ft: no contact), when away-1 swats it away hard (23 ft/s) half-way through a frame step. Returns the scene
    and the contact point."""
    s = Scene()
    bx, by = s.court.basket_xy("left")
    s.add("home-1", bx + 3.5, by).add("away-1", bx + 1.5, by + 2.0).add("away-3", bx + 14.0, by + 10.0)
    s.add("home-2", 30.0, 40.0)
    s.start("home-1").hold("home-1", 0.6)
    _raise(s, (bx + 2.0, by, 9.2), 0.3)
    s.mark("release")
    t_touch = 9.5 / 30.0
    v_in = ((0.95 - 2.0) / t_touch, 0.0, (10.6 - 9.2 + 0.5 * G * t_touch ** 2) / t_touch)  # to (bx + 0.95, by, 10.6)
    c = _touched_flight(s, v_in, t_touch, (20.0, 12.0, 2.0), 0.7)
    _to_hands(s, "away-3", 0.5).hold("away-3", 1.0)
    return s, c


def test_swat_at_the_ring_is_a_block_not_a_ring_bounce_3d() -> None:
    s, c = _ring_block()
    cfg = EventEngineConfig()
    bx, by = s.court.basket_xy("left")

    def ring_dist(p: Vec3) -> float:
        return math.hypot(math.hypot(p[0] - bx, p[1] - by) - s.court.rim_radius, p[2] - s.court.rim_height)

    contact = cfg.ball_radius_ft + cfg.rim_contact_dist_ft
    tc = s.marks["touch"]
    before = [r for r in s.recs if r.t < tc][-1]
    after = next(r for r in s.recs if r.t > tc)
    nxt = s.recs[s.recs.index(after) + 1]
    assert contact < ring_dist(c) < contact + 0.15  # close to the ring, not touching it
    # the old test: a ball-sized slack of half the longer step around the samples (here the fast step after the
    # swat) reached the ring
    slack = 0.5 * max(math.dist(before.ball, after.ball), math.dist(after.ball, nxt.ball))  # type: ignore[arg-type]
    assert min(ring_dist(before.ball), ring_dist(after.ball)) <= contact + slack  # type: ignore[arg-type]
    ev = live(run_engine(s.frames_3d(), cfg).events)
    blk = one(ev, EventKind.BLOCK)
    assert (blk.player_id, blk.other_player_id) == ("away-1", "home-1") and abs(blk.t - tc) <= 0.1
    assert one(ev, EventKind.SHOT_MISSED).attrs.get("blocked") is True
    assert one(ev, EventKind.REBOUND).player_id == "away-3"


# ---------------------------------------------------------------------------- dead balls a dribble does not restart


@pytest.mark.parametrize("board_after", [True, False])
def test_dribbling_after_a_shot_clock_violation_does_not_restart_play(board_after: bool) -> None:
    """home-1's shot clock runs out while he dribbles (game clock stopped from then on). He keeps bouncing the dead
    ball 1.2 s later, then hands it to away-2 by the sideline. No restart: no steal of that hand-off, one turnover.
    Without scoreboard reads after the violation (``board_after`` False) the team that lost the ball still never
    restarts play itself."""
    s = two_team_scene(shot_clock=1.5)
    s.add("away-2", 26.0, 1.5)
    s.start("home-1").dribble("home-1", 1.2, 2)
    s.hold("home-1", 0.5)
    s.mark("expiry")
    s.clock_running = False
    s.hold("home-1", 1.2)
    s.dribble("home-1", 1.2, 2)
    s.hold("home-1", 0.3)
    s.pass_to("away-2", duration=0.4)
    s.hold("away-2", 1.5)
    frames = s.frames_3d()
    if not board_after:
        for fr in frames:
            if fr.t > s.marks["expiry"] + 0.3:
                fr.scoreboard = None
    eng = run_engine(frames)
    ev = live(eng.events)
    to = one(ev, EventKind.TURNOVER)
    assert to.attrs["type"] == "shot_clock" and to.player_id == "home-1"
    assert not live(ev, EventKind.STEAL) and not live(ev, EventKind.PASS)
    assert eng.phase == "dead"


def test_lane_player_bouncing_a_missed_first_free_throw_does_not_restart_play() -> None:
    """Two free throws: the first misses off the front rim, away-2 on the lane line catches it, bounces it twice
    (after ``dead_min_s``) and passes it back to the shooter, who makes the second. The ball is dead between the two:
    no steal, no turnover, no rebound of the first."""
    s = ft_scene().start("home-1")
    free_throw(s, made=False)
    s.fly(lambda pos: s.hands("away-2", pos), 0.6, apex=2.5)
    s.hold("away-2", 1.3)
    s.dribble("away-2", 1.0, 2)
    s.hold("away-2", 0.3)
    s.pass_to("home-1", duration=0.5)
    free_throw(s, made=True)
    s.rest(0.6)
    ev = live(run_with_foul(s, "3d", foul(0.0)).events)
    atts = live(ev, EventKind.SHOT_ATTEMPT)
    assert [a.attrs.get("shot_type") for a in atts] == ["free_throw", "free_throw"]
    assert not live(ev, EventKind.STEAL) and not live(ev, EventKind.TURNOVER) and not live(ev, EventKind.REBOUND)
    assert one(ev, EventKind.SHOT_MADE).shot_id == atts[1].shot_id


# ---------------------------------------------------------------------------- throw-in in the air: a pass


def test_throw_in_caught_next_to_a_defender_is_the_receivers_ball() -> None:
    """home-1's pass goes out over the near sideline; away-2 picks the dead ball up out of bounds and throws it in to
    away-1, who holds it on the side of home-1 guarding him: 0.82 ft from away-1, 0.5 ft from home-1. As in live play
    the thrower's team is preferred and an opponent needs the steal confirmation: no steal."""
    s = two_team_scene()
    s.add("away-2", 24.0, -2.0).add("away-1", 20.0, 3.0).add("home-1", 30.0, 25.0)
    s.start("home-1").hold("home-1", 0.8)
    s.fly((24.0, -3.0, BALL_R), 0.6, apex=2.0)
    s.rest(1.0, visible=False)
    s.run(0.05, lambda a, pos: s.hands("away-2", pos))
    s.hold("away-2", 1.5, move=(24.0, -1.5))
    s.add("home-1", 21.2, 3.5)
    s.mark("throw")
    s.fly((20.4, 3.1, 3.8), 0.3, apex=0.6)
    s.mark("catch")
    s.fly((20.8, 3.2, 3.5), 0.15, apex=0.0)
    s.rest(0.6)
    s.dribble("away-1", 1.2, 2)
    ev = live(run_engine(s.frames_3d()).events)
    assert not live(ev, EventKind.STEAL)
    assert [(p.player_id, p.other_player_id) for p in live(ev, EventKind.PASS) if p.t > s.marks["throw"] - 0.1] \
        == [("away-2", "away-1")]
    poss = [e for e in live(ev, EventKind.POSSESSION) if e.t > 0.5]
    assert [(e.team, e.attrs["cause"]) for e in poss] == [(Team.AWAY, "turnover")]  # (dated at away-2's pickup)


# ---------------------------------------------------------------------------- high catch of a loose ball


def _high_catch() -> Scene:
    """home-1's jumper comes off the front of the left ring; home-2, 12 ft out, catches the rebound overhead at
    9.4 ft (1.5 ft from his centre), pulls it in for 0.27 s without bringing it down and shoots again."""
    s = two_team_scene()
    bx, by = s.court.basket_xy("left")
    s.add("home-1", 20.0, 25.0).add("home-2", bx + 12.0, by + 6.0).add("away-1", bx + 4.0, by - 4.0)
    s.start("home-1").hold("home-1", 0.8)
    s.mark("release1")
    s.shoot("left", made=False)
    r = s.ball
    assert r is not None
    catch = (bx + 11.0, by + 5.0, 9.4)
    T = 0.6
    v = ((catch[0] - r[0]) / T, (catch[1] - r[1]) / T, (catch[2] - r[2] + 0.5 * G * T * T) / T)
    s.run(T, lambda a, pos: _ballistic(r, v, a * T))
    s.mark("catch")
    hold_end = (bx + 11.9, by + 5.6, 9.4)
    s.run(0.27, lambda a, pos: tuple(catch[i] + a * (hold_end[i] - catch[i]) for i in range(3)))  # type: ignore[arg-type,return-value]
    s.mark("release2")
    s.shoot("left", made=True, release_z=9.4)
    s.drop_through("left")
    s.set_score(2, 0)
    s.rest(0.8)
    return s


def test_rebound_caught_overhead_and_shot_again_is_a_rebound_and_a_second_try_3d() -> None:
    s = _high_catch()
    ev = live(run_engine(s.frames_3d()).events)
    atts = live(ev, EventKind.SHOT_ATTEMPT)
    assert [a.player_id for a in atts] == ["home-1", "home-2"]
    assert abs(atts[1].t - s.marks["release2"]) < TOL
    reb = one(ev, EventKind.REBOUND)
    assert reb.player_id == "home-2" and reb.attrs["offensive"] and abs(reb.t - s.marks["catch"]) < TOL
    assert one(ev, EventKind.SHOT_MADE).player_id == "home-2"
    assert "inferred" not in reb.attrs
    # the high catch is what decides: without it nobody gains the raised ball and the second try is never seen in
    # the ball track (only the scoreboard's +2 infers a putback, late)
    off = live(run_engine(s.frames_3d(), EventEngineConfig(holder_high_catch=False)).events)
    assert one(off, EventKind.REBOUND).attrs.get("inferred") == "scoreboard"
    assert abs(live(off, EventKind.SHOT_ATTEMPT)[1].t - s.marks["release2"]) > TOL


# ---------------------------------------------------------------------------- tip while the shot is unresolved


def _tip_in_zone() -> Scene:
    """home-1's jumper hits the front of the left ring and pops up and out; still in the 3-D rim zone (above rim
    height - 1 ft, 2.9 ft from the basket) home-2 taps it back up and in."""
    s = two_team_scene()
    bx, by = s.court.basket_xy("left")
    s.add("home-1", 20.0, 25.0).add("home-2", bx + 3.6, by + 1.5).add("away-1", bx + 5.0, by - 5.0)
    s.start("home-1").hold("home-1", 0.8)
    s.mark("release1")
    s.shoot("left", made=False)
    r = s.ball
    assert r is not None
    out_v = (5.0, 1.5, 4.0)
    t_tip = 0.38
    tip = _ballistic(r, out_v, t_tip)
    s.marks["tip"] = s.t + t_tip
    T = 0.75
    tip_v = ((bx - tip[0]) / T, (by - tip[1]) / T, (s.court.rim_height - tip[2] + 0.5 * G * T * T) / T)

    def ball(a: float, pos: dict[str, tuple[float, float]]) -> Vec3:
        dt = a * (t_tip + T)
        return _ballistic(r, out_v, dt) if dt <= t_tip else _ballistic(tip, tip_v, dt - t_tip)

    s.run(t_tip + T, ball)
    s.drop_through("left")
    s.set_score(2, 0)
    s.rest(0.8)
    return s


def test_tip_in_while_the_missed_shot_is_still_in_the_rim_zone_is_a_rebound_and_a_putback_3d() -> None:
    s = _tip_in_zone()
    ev = live(run_engine(s.frames_3d()).events)
    atts = live(ev, EventKind.SHOT_ATTEMPT)
    assert [a.player_id for a in atts] == ["home-1", "home-2"]
    assert abs(atts[1].t - s.marks["tip"]) < TOL
    assert one(ev, EventKind.SHOT_MISSED).shot_id == atts[0].shot_id
    reb = one(ev, EventKind.REBOUND)
    assert reb.player_id == "home-2" and reb.attrs["offensive"] and abs(reb.t - s.marks["tip"]) < TOL
    made = one(ev, EventKind.SHOT_MADE)
    assert made.player_id == "home-2" and made.shot_id == atts[1].shot_id
    # without tips inside the unresolved shot the putback plays on as home-1's make
    off = live(run_engine(s.frames_3d(), EventEngineConfig(tip_in_shot=False)).events)
    assert one(off, EventKind.SHOT_MADE).player_id == "home-1"


# ---------------------------------------------------------------------------- a try at one's own basket


def _scramble() -> Scene:
    """home-1 misses at the left basket (home attacks left). The rebound drops between away-1 (0.9 ft from it) and
    home-2 (1.7 ft): home-2 takes it, pulls it to his side (the ball stays nearer away-1 than the holder switch asks)
    and lays it in. The holder vote gives the rebound to away-1, whose "try" goes at the basket away defends."""
    s = two_team_scene()
    bx, by = s.court.basket_xy("left")
    s.add("home-1", 20.0, 25.0).add("away-1", bx + 4.35, by + 1.3).add("home-2", bx + 5.15, by + 3.0)
    s.add("home-3", 30.0, 40.0).add("away-2", 30.0, 10.0).add("away-3", 26.0, 30.0)
    s.start("home-1").hold("home-1", 0.8)
    s.shoot("left", made=False)
    r = s.ball
    assert r is not None
    p = (bx + 3.75, by + 2.0, 5.8)
    s.fly(p, 0.5, apex=1.0)
    s.mark("rebound")
    lift = (bx + 4.1, by + 2.4, 6.6)
    s.run(0.2, lambda a, pos: tuple(p[i] + a * (lift[i] - p[i]) for i in range(3)))  # type: ignore[arg-type,return-value]
    top = (bx + 4.3, by + 2.6, 8.2)
    s.run(0.25, lambda a, pos: tuple(lift[i] + a * (top[i] - lift[i]) for i in range(3)))  # type: ignore[arg-type,return-value]
    s.mark("release")
    _shot_in(s, "left", 0.6)
    s.set_score(2, 0)
    s.rest(0.8)
    return s


def test_try_at_the_basket_his_team_defends_goes_to_the_attacker_who_held_the_ball_3d() -> None:
    s = _scramble()
    off = live(run_engine(s.frames_3d(), EventEngineConfig(own_basket_recredit=False)).events)
    # the scene does what it says: without the rule the rebound and the try are away-1's
    assert one(off, EventKind.REBOUND).player_id == "away-1"
    assert [a.player_id for a in live(off, EventKind.SHOT_ATTEMPT)][:2] == ["home-1", "away-1"]
    ev = live(run_engine(s.frames_3d()).events)
    reb = one(ev, EventKind.REBOUND)
    assert reb.player_id == "home-2" and reb.team is Team.HOME and reb.attrs["offensive"]
    atts = live(ev, EventKind.SHOT_ATTEMPT)
    assert [a.player_id for a in atts] == ["home-1", "home-2"] and atts[1].team is Team.HOME
    assert one(ev, EventKind.SHOT_MADE).player_id == "home-2"
    assert not [e for e in live(ev, EventKind.POSSESSION) if e.team is Team.AWAY and e.t < s.marks["release"] + 0.5]


# ---------------------------------------------------------------------------- rebound at the period-end buzzer


def test_rebound_secured_at_the_buzzer_is_a_rebound() -> None:
    """home-1's jumper misses; away-3 gains control of the rebound at the last frame of the period (the game clock
    reads 0 one frame later): its gain, confirmed after the buzzer, is dated by it."""
    s = two_team_scene()
    bx, by = s.court.basket_xy("left")
    s.add("home-1", 20.0, 25.0).add("away-3", bx + 6.0, by + 4.0)
    s.start("home-1").hold("home-1", 0.8)
    s.shoot("left", made=False)
    land = s.hands("away-3")
    flight = 0.7
    s.clock = flight + 1.0 / 30.0  # the clock reaches 0 one frame after the ball reaches away-3's hands
    s.fly(land, flight, apex=1.5)
    s.mark("catch")
    s.hold("away-3", 1.0)
    ev = live(run_engine(s.frames_3d()).events)
    end = one(ev, EventKind.PERIOD_END)
    reb = one(ev, EventKind.REBOUND)
    assert reb.player_id == "away-3" and not reb.attrs["offensive"] and reb.t <= end.t + 1e-6
    assert abs(reb.t - s.marks["catch"]) < TOL


# ---------------------------------------------------------------------------- a pass thrown out of bounds


def _thrown_out(low: bool) -> Scene:
    """home-1 throws a pass over the near sideline from his hands (``low``: the ball is knocked away from his dribble
    near the floor instead); away-2 inbounds it to away-1."""
    s = two_team_scene()
    s.add("away-2", 24.0, -2.0)
    s.start("home-1").dribble("home-1", 1.0, 2)
    if low:
        b = s.ball
        assert b is not None
        s.run(0.1, lambda a, pos: (b[0], b[1], b[2] + a * (0.8 - b[2])))  # down to the floor
    else:
        s.hold("home-1", 0.6)
    s.mark("release")
    s.fly((24.0, -3.0, BALL_R), 0.7, apex=2.0 if not low else 0.5)
    s.rest(1.0, visible=False)
    s.run(0.05, lambda a, pos: s.hands("away-2", pos))
    s.hold("away-2", 1.0)
    s.pass_to("away-1").hold("away-1", 0.8)
    return s


def test_pass_thrown_out_of_bounds_is_an_incomplete_pass_3d() -> None:
    s = _thrown_out(low=False)
    ev = live(run_engine(s.frames_3d()).events)
    passes = live(ev, EventKind.PASS)
    lost = [p for p in passes if p.player_id == "home-1"]
    assert len(lost) == 1 and lost[0].attrs["completed"] is False and abs(lost[0].t - s.marks["release"]) < TOL
    assert one(ev, EventKind.TURNOVER).player_id == "home-1"
    off = live(run_engine(s.frames_3d(), EventEngineConfig(incomplete_pass_out=False)).events)
    assert not [p for p in live(off, EventKind.PASS) if p.player_id == "home-1"]


def test_dribble_knocked_out_of_bounds_near_the_floor_is_no_pass_3d() -> None:
    s = _thrown_out(low=True)
    ev = live(run_engine(s.frames_3d()).events)
    assert not [p for p in live(ev, EventKind.PASS) if p.player_id == "home-1"]


def test_catch_fumbled_out_of_bounds_is_no_pass_by_the_receiver_3d() -> None:
    """home-1 passes to home-2, who has the ball for 0.2 s before it is knocked off his hands (12 ft/s) over the
    sideline: a fumble, not a pass of his (``incomplete_pass_min_hold_s``)."""
    s = two_team_scene()
    s.start("home-1").hold("home-1", 0.8)
    s.pass_to("home-2")
    s.hold("home-2", 0.2)
    s.mark("fumble")
    s.fly((22.0, 51.5, BALL_R), 1.0, apex=1.0)
    s.rest(1.0, visible=False)
    ev = live(run_engine(s.frames_3d()).events)
    assert [p.player_id for p in live(ev, EventKind.PASS)] == ["home-1"]
    # the hold is what decides: with no minimum the fumble is an incomplete pass of home-2's
    off = live(run_engine(s.frames_3d(), EventEngineConfig(incomplete_pass_min_hold_s=0.0)).events)
    assert [p.player_id for p in live(off, EventKind.PASS)] == ["home-1", "home-2"]
