"""Regression tests for the round-4 event-engine changes.

Commits 7b1e0c6 (throw-ins restart on the court, dribble restart gate, shot-clock violation after a miss, 3-D shot
apex gate, ring-contact step slack, reconciler flip hold), dc30c32 (monocular loose-ball block gate), 38d1655
(monocular long-flight gains, firm dribbles) and f3e5e81 (monocular pass transit). Scenes reuse the kit in
``test_events_kit.py``. Where a scene could pass for an unrelated reason, a contrast run with the new rule switched off
through its config field shows that the rule is what decides.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from courtvision.events.engine import EventEngine, EventEngineConfig
from courtvision.events.reconcile import Action, Reconciler, ShotEntry
from courtvision.events.scoreboard import ScoreChange
from courtvision.events.shots import Flight, Release
from courtvision.geometry.court import court_for
from courtvision.types import BallState, BBox, EventKind, FrameState, PlayerState, Team

from test_events_kit import BALL_R, HANDS_Z, Scene, kinds, live, one, run_engine, two_team_scene

G = 32.174
TOL = 0.35


def _run_tracking(frames: list[FrameState], config: EventEngineConfig | None = None) -> tuple[EventEngine, list[tuple[float, str]]]:
    """Run the engine frame by frame, recording its phase after each frame."""
    eng = EventEngine(court_for("nba"), config=config)
    phases = []
    for fr in frames:
        eng.process(fr)
        phases.append((fr.t, eng.phase))
    eng.flush()
    return eng, phases


def _flight(s: Scene, v: tuple[float, float, float], duration: float) -> Scene:
    """Free flight from the current ball position with initial velocity ``v`` (ft/s)."""
    p0 = s.ball
    assert p0 is not None
    return s.run(duration, lambda a, pos: (p0[0] + v[0] * a * duration, p0[1] + v[1] * a * duration,
                                           p0[2] + v[2] * a * duration - 0.5 * G * (a * duration) ** 2))


def _to_hands(s: Scene, pid: str, duration: float) -> Scene:
    """Straight from the current ball position into ``pid``'s hands."""
    e = s.ball
    assert e is not None
    return s.run(duration, lambda a, pos: tuple(e[i] + a * (s.hands(pid, pos)[i] - e[i]) for i in range(3)))  # type: ignore[return-value]


def _passes(events: list) -> list[tuple[str | None, str | None]]:
    return [(p.player_id, p.other_player_id) for p in live(events, EventKind.PASS)]


# ---------------------------------------------------------------------------- 7b1e0c6: throw-ins restart on the court


def _throw_in_scene(handoff: bool) -> Scene:
    """home-1's pass goes out over the near sideline (3-D: a dead ball, cause "oob"). The dead ball is picked up
    out of bounds by away-3, who hands it to the inbounder away-2 (``handoff``), or by away-2 himself; away-2 throws
    it in to away-1 on the court."""
    s = two_team_scene()
    s.add("away-3", 20.0, -4.0).add("away-2", 24.0, -2.0)
    s.start("home-1").hold("home-1", 0.8)
    s.fly((24.0, -3.0, BALL_R), 0.6, apex=2.0)
    s.mark("out")
    s.rest(1.0, visible=False)
    s.run(0.05, lambda a, pos: s.hands("away-3" if handoff else "away-2", pos))
    s.mark("pickup")
    if handoff:
        s.hold("away-3", 1.0)
        s.mark("handoff")
        s.pass_to("away-2", duration=0.4)
        s.hold("away-2", 1.0)
    else:
        s.hold("away-2", 1.5)
    s.mark("release")
    s.pass_to("away-1")
    s.mark("catch")
    s.hold("away-1", 0.8)
    return s


def test_dead_ball_handed_to_the_inbounder_restarts_when_a_court_player_gains_the_throw_in() -> None:
    """A dead ball handed from one player out of bounds to another is no throw-in: play restarts only when
    away-1 on the court gains away-2's throw-in, dated at away-2's release; the inbounding team's possession starts
    with away-2's own control, not with away-3's pickup."""
    s = _throw_in_scene(handoff=True)
    eng, phases = _run_tracking(s.frames_3d())
    ev = live(eng.events)
    assert all(ph == "dead" for t, ph in phases if s.marks["out"] + 0.1 <= t < s.marks["catch"])
    assert eng.phase == "live"
    p = one(ev, EventKind.PASS)
    assert (p.player_id, p.other_player_id) == ("away-2", "away-1") and abs(p.t - s.marks["release"]) < TOL
    assert eng._live_since == pytest.approx(p.t)  # the restart is dated at the throw-in's release
    poss = [e for e in live(ev, EventKind.POSSESSION) if e.team is Team.AWAY]
    assert len(poss) == 1 and poss[0].player_id == "away-2" and poss[0].attrs["cause"] == "turnover"
    assert s.marks["handoff"] <= poss[0].t < s.marks["release"]
    to = one(ev, EventKind.TURNOVER)
    assert (to.player_id, to.attrs["type"]) == ("home-1", "out_of_bounds")


def test_inbounder_who_picked_up_the_dead_ball_restarts_play_when_the_throw_in_is_gained() -> None:
    """The player who picked the dead ball up out of bounds throws it in himself: the ball stays dead while the
    throw-in is in the air; when away-1 on the court gains it, play restarts dated at the release."""
    s = _throw_in_scene(handoff=False)
    eng, phases = _run_tracking(s.frames_3d())
    ev = live(eng.events)
    assert all(ph == "dead" for t, ph in phases if s.marks["out"] + 0.1 <= t < s.marks["catch"])
    assert eng.phase == "live"
    p = one(ev, EventKind.PASS)
    assert (p.player_id, p.other_player_id) == ("away-2", "away-1") and abs(p.t - s.marks["release"]) < TOL
    assert eng._live_since == pytest.approx(p.t)
    poss = [e for e in live(ev, EventKind.POSSESSION) if e.team is Team.AWAY]
    assert len(poss) == 1 and poss[0].player_id == "away-2" and poss[0].attrs["cause"] == "turnover"
    assert abs(poss[0].t - s.marks["pickup"]) < TOL


@pytest.mark.parametrize("court_receiver", [True, False])
def test_live_holder_who_steps_out_restarts_play_at_his_release(court_receiver: bool) -> None:
    """Guard: a holder who had live control when the ball went dead (he walked out over the sideline with it)
    restarts play at once when he passes it in, as before (and as with ``inbound_needs_court_receiver`` off)."""
    s = two_team_scene()
    s.add("home-1", 24.0, 4.0).add("home-2", 22.0, 14.0)
    s.start("home-1").dribble("home-1", 1.0, 2)
    s.hold("home-1", 0.5, move=(24.0, -1.8))
    s.mark("out")
    s.hold("home-1", 0.6)
    s.mark("release")
    s.pass_to("home-2")
    s.mark("catch")
    s.hold("home-2", 0.8)
    eng, phases = _run_tracking(s.frames_3d(), EventEngineConfig(inbound_needs_court_receiver=court_receiver))
    ev = live(eng.events)
    assert any(ph == "dead" for t, ph in phases if s.marks["out"] <= t < s.marks["release"])  # he stepped out
    first_live = min(t for t, ph in phases if t > s.marks["release"] and ph == "live")
    assert first_live < s.marks["catch"]  # live before home-2 has the ball
    p = one(ev, EventKind.PASS)
    assert (p.player_id, p.other_player_id) == ("home-1", "home-2") and eng._live_since == pytest.approx(p.t)
    assert not live(ev, EventKind.TURNOVER)


# ---------------------------------------------------------------------------- 7b1e0c6: dribble restart gate


def _dead_ball_dribble_scene() -> Scene:
    """home-1's pass goes out of bounds and the game clock stops. away-3 picks the dead ball up in bounds and
    bounces it around for 2.4 s (4 bounces, well after dead_min_s), then shoots."""
    s = two_team_scene(clock=600.0)
    s.add("away-3", 21.0, 3.0)
    s.start("home-1").hold("home-1", 0.8)
    s.fly((24.0, -3.0, BALL_R), 0.6, apex=2.0)
    s.mark("out")
    s.clock_running = False
    s.rest(1.0, visible=False)
    s.run(0.05, lambda a, pos: s.hands("away-3", pos))
    s.mark("pickup")
    s.hold("away-3", 0.4)
    s.mark("dribble")
    s.dribble("away-3", 2.4, 4, move=(16.0, 14.0))
    s.hold("away-3", 0.3)
    s.mark("release")
    s.shoot("left", made=False)
    s.rest(0.5)
    return s


def test_dribbling_the_dead_ball_while_the_read_clock_stays_stopped_does_not_restart_play() -> None:
    s = _dead_ball_dribble_scene()
    eng, phases = _run_tracking(s.frames_3d())
    assert all(ph == "dead" for t, ph in phases if t >= s.marks["out"] + 0.1)
    assert kinds(eng.events) == ["possession"]  # no inferred turnover, no away possession, no shot
    assert not live(eng.events, EventKind.DRIBBLE)
    # the stopped clock is what holds it: without clock-stop dead balls the same bounces restart play
    eng2 = run_engine(s.frames_3d(), EventEngineConfig(dead_from_clock_stop=False))
    assert one(eng2.events, EventKind.SHOT_ATTEMPT).player_id == "away-3"


def test_dribble_restarts_play_when_the_scoreboard_reads_stop() -> None:
    """With no fresh scoreboard reads (not streaming) a clock last seen stopped proves nothing: the dribble
    restarts play as before, and the shot after it is a field-goal attempt."""
    s = _dead_ball_dribble_scene()
    frames = s.frames_3d()
    for fr in frames:
        if fr.t > s.marks["out"] + 0.3:
            fr.scoreboard = None
    eng, _ = _run_tracking(frames)
    ev = live(eng.events)
    assert eng.phase == "live" and s.marks["dribble"] <= eng._live_since < s.marks["release"]
    assert any(abs(e.t - eng._live_since) < 1e-6 for e in live(ev, EventKind.DRIBBLE))  # restarted by a bounce
    to = one(ev, EventKind.TURNOVER)
    assert (to.player_id, to.attrs["type"]) == ("home-1", "out_of_bounds")
    poss = [e for e in live(ev, EventKind.POSSESSION) if e.team is Team.AWAY]
    assert len(poss) == 1 and poss[0].player_id == "away-3" and poss[0].attrs["cause"] == "turnover"
    att = one(ev, EventKind.SHOT_ATTEMPT)
    assert att.player_id == "away-3" and abs(att.t - s.marks["release"]) < TOL
    assert one(ev, EventKind.SHOT_MISSED).shot_id == att.shot_id


# ---------------------------------------------------------------------------- 7b1e0c6: shot-clock violation after a miss


def _miss_then_shot_clock_scene(players: dict[str, tuple[float, float]]) -> Scene:
    """home-1's jumper comes off the front rim and rolls loose; nobody has secured it when the shot clock (read
    2.2 s at the release) runs out."""
    s = two_team_scene()
    s.add("home-1", 20.0, 25.0)
    for pid, (x, y) in players.items():
        s.add(pid, x, y)
    s.start("home-1").hold("home-1", 0.8)
    s.shot_clock = 2.2
    s.mark("release")
    s.marks["expiry"] = s.t + 2.2
    s.shoot("left", made=False)
    s.fly((11.0, 30.0, BALL_R), 0.5, apex=1.0)
    b = s.ball
    assert b is not None
    s.run(1.0, lambda a, pos: (b[0] + 3.0 * a, b[1] + 1.0 * a, BALL_R))  # rolling, nobody near
    s.rest(0.6, visible=False)
    s.shot_clock = 24.0
    return s


@pytest.mark.parametrize("court_receiver", [True, False])
def test_shot_clock_violation_after_a_miss_leaves_no_rebound_for_the_inbound(court_receiver: bool) -> None:
    """The violation kills the ball: the miss before it gets no rebound. away-3 retrieves the ball out of bounds
    and hands it to the inbounder away-2, who throws it in. (With the pre-round-4 throw-in,
    ``inbound_needs_court_receiver`` off, the hand-off restarted play and the next gain out of bounds ended it
    again, carrying the pending miss into a team REBOUND.)"""
    s = _miss_then_shot_clock_scene({"away-3": (14.0, -4.0), "away-2": (18.0, -2.0), "away-1": (16.0, 12.0)})
    s.run(0.05, lambda a, pos: s.hands("away-3", pos)).hold("away-3", 1.0)
    s.pass_to("away-2", duration=0.4).hold("away-2", 1.0)
    s.mark("inbound")
    s.pass_to("away-1").hold("away-1", 0.8)
    eng = run_engine(s.frames_3d(), EventEngineConfig(inbound_needs_court_receiver=court_receiver))
    ev = live(eng.events)
    assert one(ev, EventKind.SHOT_MISSED).player_id == "home-1"
    to = one(ev, EventKind.TURNOVER)
    assert to.attrs["type"] == "shot_clock" and to.team is Team.HOME and abs(to.t - s.marks["expiry"]) < 0.1
    assert not live(ev, EventKind.REBOUND)
    poss = [e for e in live(ev, EventKind.POSSESSION) if e.team is Team.AWAY]
    assert len(poss) == 1 and poss[0].attrs["cause"] == "turnover"
    assert _passes(ev) == [("away-2", "away-1")]


def test_shot_clock_violation_after_a_miss_does_not_turn_the_next_dead_ball_into_a_team_rebound() -> None:
    """Default config: after the violation away-1 picks the ball up in bounds and play restarts on his dribble;
    he then dribbles out over the sideline. That out of bounds is away-1's turnover (inferred at home-3's inbound),
    not a team rebound of home-1's old miss."""
    s = _miss_then_shot_clock_scene({"away-1": (16.0, 6.0), "home-3": (26.0, -2.0), "home-2": (24.0, 12.0)})
    s.run(0.05, lambda a, pos: s.hands("away-1", pos)).hold("away-1", 0.5)
    s.dribble("away-1", 1.6, 3)
    s.dribble("away-1", 0.8, 2, move=(18.0, -2.0))
    s.hold("away-1", 0.3)
    s.rest(0.8, visible=False)
    s.run(0.05, lambda a, pos: s.hands("home-3", pos)).hold("home-3", 1.0)
    s.pass_to("home-2").hold("home-2", 0.8)
    eng = run_engine(s.frames_3d())
    ev = live(eng.events)
    assert not live(ev, EventKind.REBOUND)
    assert [(e.team, e.attrs["type"]) for e in live(ev, EventKind.TURNOVER)] == \
        [(Team.HOME, "shot_clock"), (Team.AWAY, "out_of_bounds")]
    assert live(ev, EventKind.TURNOVER)[1].player_id == "away-1"
    poss = [e for e in live(ev, EventKind.POSSESSION) if e.team is Team.HOME and e.t > s.marks["expiry"]]
    assert len(poss) == 1 and poss[0].player_id == "home-3" and poss[0].attrs["cause"] == "turnover"


# ---------------------------------------------------------------------------- 7b1e0c6: 3-D shot apex gate


def _chest_pass_turned_back() -> Scene:
    """home-1, 11 ft from the left basket, passes off his dribble from chest height, rising at 15 ft/s (free-flight
    apex ~7.5 ft, well below the rim); away-1 knocks it down at shoulder height (6.6 ft) and away-3 picks it up."""
    s = Scene()
    s.add("home-1", 16.0, 25.0).add("away-1", 10.5, 24.4).add("away-3", 8.0, 32.0).add("home-2", 30.0, 40.0)
    s.start("home-1").dribble("home-1", 1.0, 2)
    s.mark("release")
    _flight(s, (-20.0, 0.0, 15.0), 7 / 30)
    s.mark("touch")
    _flight(s, (-3.0, 6.0, -8.0), 0.2)
    e = s.ball
    assert e is not None
    s.run(0.25, lambda a, pos: (e[0] - 1.0 * a, e[1] + 2.0 * a, e[2] + a * (BALL_R - e[2])))
    _to_hands(s, "away-3", 0.5).hold("away-3", 1.0)
    return s


def test_fast_rising_pass_below_rim_height_turned_back_is_no_block_or_attempt_3d() -> None:
    s = _chest_pass_turned_back()
    ev = live(run_engine(s.frames_3d()).events)
    assert not live(ev, EventKind.BLOCK)
    assert not live(ev, EventKind.SHOT_ATTEMPT) and not live(ev, EventKind.SHOT_MISSED)
    # the apex gate is what rejects it: the same flight without it is a blocked shot
    loose = live(run_engine(s.frames_3d(), EventEngineConfig(block_min_apex_below_rim_ft=1e9)).events)
    assert one(loose, EventKind.BLOCK).player_id == "away-1"


def test_jump_shot_turned_back_on_its_way_up_is_still_a_block_3d() -> None:
    """Guard: a jumper released at 8 ft rising at 12 ft/s (apex ~10.2 ft) swatted 0.1 s after the release by
    away-1's raised hand (9 ft) is a block."""
    s = Scene()
    s.add("home-1", 16.0, 25.0).add("away-1", 14.2, 24.5).add("away-3", 24.0, 30.0).add("home-2", 30.0, 40.0)
    s.start("home-1").dribble("home-1", 1.0, 2)
    st, top = s.ball, (16.0, 24.6, 8.0)
    assert st is not None
    s.run(0.3, lambda a, pos: tuple(st[i] + a * (top[i] - st[i]) for i in range(3)))  # type: ignore[arg-type,return-value]
    s.mark("release")
    _flight(s, (-10.0, 0.0, 12.0), 3 / 30)
    s.mark("touch")
    _flight(s, (9.0, 2.0, -4.0), 0.4)
    _to_hands(s, "away-3", 0.5).hold("away-3", 1.0)
    ev = live(run_engine(s.frames_3d()).events)
    blk = one(ev, EventKind.BLOCK)
    assert (blk.player_id, blk.other_player_id) == ("away-1", "home-1") and abs(blk.t - s.marks["touch"]) <= 0.1
    att = one(ev, EventKind.SHOT_ATTEMPT)
    assert att.player_id == "home-1" and abs(att.t - s.marks["release"]) <= 0.1
    assert one(ev, EventKind.SHOT_MISSED).attrs.get("blocked") is True


# ---------------------------------------------------------------------------- 7b1e0c6: ring / board contact slack


def _ring_bounce_between_frames() -> Scene:
    """home-1's jumper hits the front of the ring half-way between two frames and bounces back out; away-1
    stands under the basket, ~2 ft from the ball at the bounce; away-3 rebounds."""
    s = two_team_scene()
    s.add("home-1", 20.0, 25.0).add("away-1", 8.0, 26.5).add("away-3", 12.0, 31.0)
    s.start("home-1").dribble("home-1", 1.0, 2)
    st, top = s.ball, (20.0, 24.3, 8.0)
    assert st is not None
    s.run(0.3, lambda a, pos: tuple(st[i] + a * (top[i] - st[i]) for i in range(3)))  # type: ignore[arg-type,return-value]
    s.mark("release")
    bx, by = s.court.basket_xy("left")
    k = 0.39 / math.sqrt(2.0)
    c = (bx + s.court.rim_radius + k, by, s.court.rim_height + k)  # ball centre touching the front of the ring
    T = 27.5 / 30.0  # the contact falls half-way between two frames
    v0 = ((c[0] - top[0]) / T, (c[1] - top[1]) / T, (c[2] - top[2]) / T + 0.5 * G * T)
    vin = (v0[0], v0[1], v0[2] - G * T)
    n = (1 / math.sqrt(2.0), 0.0, 1 / math.sqrt(2.0))  # contact normal (out and up)
    vn = sum(vin[i] * n[i] for i in range(3))
    vout = tuple(vin[i] - 1.6 * vn * n[i] for i in range(3))  # restitution 0.6 along the normal
    t0 = s.t
    s.marks["contact"] = t0 + T

    def traj(t: float) -> tuple[float, float, float]:
        dt = t - t0
        if dt <= T:
            return (top[0] + v0[0] * dt, top[1] + v0[1] * dt, top[2] + v0[2] * dt - 0.5 * G * dt * dt)
        u = dt - T
        return (c[0] + vout[0] * u, c[1] + vout[1] * u, c[2] + vout[2] * u - 0.5 * G * u * u)

    dur = T + 0.75
    s.run(dur, lambda a, pos: traj(t0 + a * dur))
    _to_hands(s, "away-3", 0.5).hold("away-3", 1.0)
    return s


def test_ring_bounce_between_frames_next_to_a_defender_is_a_miss_not_a_block_3d() -> None:
    s = _ring_bounce_between_frames()
    cfg = EventEngineConfig()
    bx, by = s.court.basket_xy("left")

    def ring_dist(p: tuple[float, float, float]) -> float:
        return math.hypot(math.hypot(p[0] - bx, p[1] - by) - s.court.rim_radius, p[2] - s.court.rim_height)

    tc = s.marks["contact"]
    before = [r for r in s.recs if r.t < tc][-1]
    after = next(r for r in s.recs if r.t > tc)
    prev = [r for r in s.recs if r.t < before.t][-1]
    contact = cfg.ball_radius_ft + cfg.rim_contact_dist_ft
    half_step = 0.5 * math.dist(prev.ball, before.ball)  # type: ignore[arg-type]
    for r in (before, after):  # neither sample touches the ring by itself, both are within half a step of it
        assert contact < ring_dist(r.ball) <= contact + half_step  # type: ignore[arg-type]
        assert math.dist(r.ball[:2], r.pos["away-1"]) < cfg.block_defender_radius_ft  # type: ignore[index]
    ev = live(run_engine(s.frames_3d(), cfg).events)
    assert not live(ev, EventKind.BLOCK)
    miss = one(ev, EventKind.SHOT_MISSED)
    assert miss.player_id == "home-1" and not miss.attrs.get("blocked")
    assert one(ev, EventKind.REBOUND).player_id == "away-3"
    # the step slack is what decides: without it the bounce is taken for away-1's block
    no_slack = live(run_engine(s.frames_3d(), EventEngineConfig(rim_contact_step_frac=0.0)).events)
    assert one(no_slack, EventKind.BLOCK).player_id == "away-1"


# ---------------------------------------------------------------------------- 7b1e0c6: reconciler flip hold


def _delta(t: float, team: Team = Team.HOME, delta: int = 2) -> ScoreChange:
    return ScoreChange(t=t, team=team, delta=delta, home=delta, away=0, period=1, game_clock=590.0, source="sim",
                       confirmed_t=t)


def _shot(shot_id: str, made: bool, result_t: float, shooter: str = "home-1", value: int = 2) -> ShotEntry:
    return ShotEntry(shot_id, Team.HOME, shooter, value, f"a-{shot_id}", result_t - 0.8, result_id=f"r-{shot_id}",
                     made=made, result_t=result_t)


def test_reconciler_flip_hold_lets_a_make_resolved_after_the_delta_confirm_it() -> None:
    """A +2 confirmed a few frames before the engine resolves home-3's layup as made confirms that make; it does
    not flip home-1's miss of 4 s earlier."""
    cfg = EventEngineConfig()
    rec = Reconciler(cfg)
    miss = _shot("s0001", made=False, result_t=6.0)
    ch = _delta(10.0)
    rec.add(ch)
    assert rec.step(10.0, [miss], streaming=True) == []
    assert rec.step(10.0 + 0.5 * cfg.reconcile_flip_hold_s, [miss], streaming=True) == []
    make = _shot("s0002", made=True, result_t=10.2, shooter="home-3")
    assert rec.step(10.0 + 0.6 * cfg.reconcile_flip_hold_s, [miss, make], streaming=True) == \
        [Action("confirm", "s0002", ch)]
    assert rec.pending == []


def test_reconciler_flips_the_recent_miss_once_the_flip_hold_has_passed() -> None:
    """Guard: with no make to match, the delta flips the team's recent miss after ``reconcile_flip_hold_s``."""
    cfg = EventEngineConfig()
    rec = Reconciler(cfg)
    miss = _shot("s0001", made=False, result_t=6.0)
    ch = _delta(10.0)
    rec.add(ch)
    assert rec.step(10.0 + 0.9 * cfg.reconcile_flip_hold_s, [miss], streaming=True) == []
    assert rec.step(10.0 + cfg.reconcile_flip_hold_s, [miss], streaming=True) == \
        [Action("flip_to_made", "s0001", ch, new_value=2)]
    assert cfg.reconcile_flip_hold_s < cfg.reconcile_hold_s  # it flips before a SCORE_CORRECTION would be due


# ---------------------------------------------------------------------------- dc30c32: monocular loose-ball block gate


def _mono_jumper(recover: str, moved: dict[str, tuple[float, float]] | None = None) -> tuple[Scene, list[FrameState]]:
    """The kit's monocular blocked jumper (as in test_events_mono_regressions): home-1 rises into a jumper and
    away-1, a step in front with his arms up, turns it back 0.1 s after the release. Then:

    * ``quick_mate``: knocked down into the hands of home-4 (the shooter's teammate) ~0.25 s after the touch;
    * ``over_blocker_team`` / ``over_mate``: knocked back high over home-1's head, it lands behind him and away-4
      (the blocker's teammate) / home-4 picks it up ~1 s after the touch;
    * ``down_blocker_team``: swatted down and back past home-1's head (the ball in his image box ~0.27 s after
      the touch), away-4 picks it up ~1 s after the touch.

    The shooter's box grows to his raised arms and the ball (as perception boxes a shooter). ``moved`` places other
    players elsewhere."""
    who, (wx, wy) = {"quick_mate": ("home-4", (24.0, 22.0)), "over_blocker_team": ("away-4", (28.5, 22.0)),
                     "over_mate": ("home-4", (28.5, 22.0)), "down_blocker_team": ("away-4", (25.5, 21.0))}[recover]
    s = two_team_scene()
    s.add("home-1", 19.0, 25.0).add("away-1", 17.6, 24.4).add("home-2", 22.0, 40.0).add(who, wx, wy)
    for pid, (x, y) in (moved or {}).items():
        s.add(pid, x, y)
    s.start("home-1").dribble("home-1", 1.0, 2)
    st, top = s.ball, (19.0, 24.6, 8.6)
    assert st is not None
    s.run(0.3, lambda a, pos: tuple(st[i] + a * (top[i] - st[i]) for i in range(3)))  # type: ignore[arg-type,return-value]
    s.mark("release")
    s.run(0.1, lambda a, pos: (top[0] - 1.4 * a, top[1], top[2] + 1.5 * a - 0.5 * G * (0.1 * a) ** 2))
    s.mark("block")
    b = s.ball
    assert b is not None
    if recover == "quick_mate":
        _to_hands(s, who, 0.25)
    elif recover.startswith("over"):
        s.run(0.9, lambda a, pos: (b[0] + 8.1 * a, b[1] - 0.9 * a, max(BALL_R, b[2] + 2.7 * a - 0.5 * G * (0.9 * a) ** 2)))
        _to_hands(s, who, 0.4)
    else:
        s.run(0.6, lambda a, pos: (b[0] + 5.4 * a, b[1] - 1.8 * a, max(BALL_R, b[2] - 4.8 * a - 0.5 * G * (0.6 * a) ** 2)))
        _to_hands(s, who, 0.4)
    s.mark("gain")
    s.dribble(who, 1.5, 3, move=(wx + 4.0, wy))
    return s, _raised_ball_in_box(s.frames_mono(), "home-1", s.marks["release"] - 0.3, s.marks["block"] + 0.05)


def _raised_ball_in_box(frames: list[FrameState], pid: str, t0: float, t1: float) -> list[FrameState]:
    """As perception boxes a shooter: his box grows to his raised arms and a ball just above his head."""
    out = []
    for fr in frames:
        if t0 <= fr.t <= t1 and fr.ball is not None and fr.ball.visible and fr.ball.image_xy is not None:
            u, v = fr.ball.image_xy
            ps = []
            for p in fr.players:
                bb = p.bbox
                if p.player_id == pid and bb is not None and v < bb.y1 + 0.3 * bb.h:
                    p = dataclasses.replace(p, bbox=BBox(min(bb.x1, u - 3.0), min(bb.y1, v - 4.0), max(bb.x2, u + 3.0),
                                                         bb.y2))
                ps.append(p)
            fr = dataclasses.replace(fr, players=ps)
        out.append(fr)
    return out


def test_mono_jumper_turned_back_then_caught_at_once_by_a_teammate_is_no_block() -> None:
    """A ball the shooter's teammate holds within retro_block_min_loose_s of the touch was a pass past raised
    arms: no BLOCK (nor attempt), and the dropped touch stays dropped to the end of the stream."""
    s, frames = _mono_jumper("quick_mate")
    eng = run_engine(frames)
    ev = live(eng.events)
    assert not live(ev, EventKind.BLOCK) and not live(ev, EventKind.SHOT_ATTEMPT)
    assert abs(eng._retro_dropped_t - s.marks["block"]) <= 0.1
    assert eng.possession_team is Team.HOME
    # the gate is what drops it: without it the same play is away-1's block
    ungated = live(run_engine(frames, EventEngineConfig(retro_block_min_loose_s=0.0)).events)
    assert one(ungated, EventKind.BLOCK).player_id == "away-1"


@pytest.mark.parametrize(("recover", "recoverer", "offensive"), [
    ("over_blocker_team", "away-4", False),
    ("over_mate", "home-4", True),  # the shooter's team, but only after the ball was loose > 0.4 s
])
def test_mono_block_loose_then_recovered_is_a_block(recover: str, recoverer: str, offensive: bool) -> None:
    s, frames = _mono_jumper(recover)
    eng = run_engine(frames)
    ev = live(eng.events)
    blk = one(ev, EventKind.BLOCK)
    assert (blk.player_id, blk.other_player_id) == ("away-1", "home-1") and abs(blk.t - s.marks["block"]) <= 0.2
    assert one(ev, EventKind.SHOT_ATTEMPT).player_id == "home-1"
    assert one(ev, EventKind.SHOT_MISSED).attrs.get("blocked") is True
    reb = one(ev, EventKind.REBOUND)
    assert reb.player_id == recoverer and reb.attrs["offensive"] is offensive
    assert reb.t - blk.t > EventEngineConfig().retro_block_min_loose_s
    assert not live(ev, EventKind.STEAL) and not live(ev, EventKind.TURNOVER)
    assert eng._retro_dropped_t == -math.inf


def _ball_in_far_box(frames: list[FrameState], pid: str, t0: float, t1: float) -> list[FrameState]:
    """Monocular depth ambiguity: between ``t0`` and ``t1`` the ball high over the shooter projects into the image box
    of ``pid``, who stands far behind it along the camera ray (his floor position is kept: only the image overlaps)."""
    out = []
    for fr in frames:
        if t0 <= fr.t <= t1 and fr.ball is not None and fr.ball.visible and fr.ball.image_xy is not None:
            u, v = fr.ball.image_xy
            ps = []
            for p in fr.players:
                if p.player_id == pid and p.bbox is not None:
                    w, h = p.bbox.w, p.bbox.h  # his box, slid so the ball sits at his chest
                    p = dataclasses.replace(p, bbox=BBox(u - w / 2.0, v - 0.4 * h, u + w / 2.0, v + 0.6 * h))
                ps.append(p)
            fr = dataclasses.replace(fr, players=ps)
        out.append(fr)
    return out


@pytest.mark.parametrize("hold_s", [None, 0.0])
def test_mono_block_seen_in_a_far_teammates_box_is_a_block(hold_s: float | None) -> None:
    """Video seed-4 pattern: a jumper turned back high over the shooter lies, for ~0.3 s after the touch, in the
    image box of home-2, the shooter's teammate 15 ft behind along the camera ray, and the engine sees home-2 gain it.
    The ball could not have reached him from the release by then (steal_reach_ft + steal_max_ball_speed_fts), so the
    loose-ball gate holds the block candidate instead of dropping it; home-2's dribble-less 0.3 s stint ends when the
    ball leaves his box, and is spliced out: home-1's attempt, away-1's BLOCK, away-4's rebound, no pass to home-2.
    With retro_block_phantom_hold_s = 0 the quick "catch" drops the block (the round-4 gate as before)."""
    s, frames = _mono_jumper("over_blocker_team", moved={"away-2": (40.0, 8.0)})  # (no other body on the ball's path)
    s2 = s.marks["block"]
    frames = _ball_in_far_box(frames, "home-2", s2 + 0.02, s2 + 0.3)
    cfg = EventEngineConfig() if hold_s is None else EventEngineConfig(retro_block_phantom_hold_s=hold_s)
    eng = run_engine(frames, cfg)
    ev = live(eng.events)
    gained = [p for p in eng.events if p.kind is EventKind.PASS and p.other_player_id == "home-2"]
    assert gained, "the scene must make the engine see home-2 gain the ball"
    if hold_s == 0.0:
        assert not live(ev, EventKind.BLOCK)
        assert ("home-1", "home-2") in _passes(ev)
        return
    blk = one(ev, EventKind.BLOCK)
    assert (blk.player_id, blk.other_player_id) == ("away-1", "home-1") and abs(blk.t - s2) <= 0.2
    assert one(ev, EventKind.SHOT_ATTEMPT).player_id == "home-1"
    assert one(ev, EventKind.SHOT_MISSED).attrs.get("blocked") is True
    assert one(ev, EventKind.REBOUND).player_id == "away-4"
    assert ("home-1", "home-2") not in _passes(ev)
    assert not live(ev, EventKind.STEAL) and not live(ev, EventKind.TURNOVER)
    assert eng._held_block is None


def test_mono_far_teammate_who_keeps_the_ball_after_a_block_candidate_has_it() -> None:
    """Guard for the phantom-block splice: the same far-teammate "gain" after the touch, but home-2 then really keeps
    the ball (a dribble in his hands) past retro_block_phantom_hold_s: the pass to him stands, no BLOCK, and the
    touch stays dropped."""
    s, frames = _mono_jumper("over_blocker_team", moved={"away-2": (40.0, 8.0)})
    s2 = s.marks["block"]
    frames = _ball_in_far_box(frames, "home-2", s2 + 0.02, s2 + 2.5)  # the ball stays in his box: he has it
    eng = run_engine(frames)
    ev = live(eng.events)
    assert not live(ev, EventKind.BLOCK)
    assert ("home-1", "home-2") in _passes(ev)
    assert eng._held_block is None and abs(eng._retro_dropped_t - s2) <= 0.1


@pytest.mark.xfail(strict=True, reason="monocular depth ambiguity: the ball swatted down past home-1 lies in the "
                   "image box of away-4, who stands behind it along the camera ray, ~0.23 s after the touch; that quick "
                   "gain by a defender other than the blocker makes the loose-ball gate drop the block (STEAL away-4 "
                   "instead). The shooter's own box no longer wins it (engine: the releaser is left out of the vote "
                   "during retro_block_min_loose_s).")
def test_mono_block_swatted_down_past_the_shooter_then_recovered_by_the_blockers_team_is_a_block() -> None:
    s, frames = _mono_jumper("down_blocker_team")
    ev = live(run_engine(frames).events)
    blk = one(ev, EventKind.BLOCK)
    assert (blk.player_id, blk.other_player_id) == ("away-1", "home-1")
    assert one(ev, EventKind.REBOUND).player_id == "away-4"
    assert not live(ev, EventKind.STEAL)


def _engine_with_players(players: dict[str, tuple[float, float]], n: int = 12) -> EventEngine:
    eng = EventEngine(court_for("nba"))
    for k in range(n):
        ps = [PlayerState(pid, Team(pid.split("-")[0]), x, y) for pid, (x, y) in players.items()]
        eng.process(FrameState(k, k / 30.0, ps, BallState(visible=False)))
    return eng


@pytest.mark.parametrize(("gainer", "after_s", "kept"), [
    ("home-2", 0.2, False),  # the shooter's teammate at once: a pass past raised arms
    ("home-1", 0.2, False),  # the shooter himself at once
    ("away-3", 0.2, False),  # another defender, standing elsewhere: a deflection, not a block
    ("away-5", 0.2, True),  # another id where the blocker stood: one player's renamed track
    ("away-1", 0.1, True),  # the blocker himself
    ("home-2", 0.5, True),  # the shooter's team, but the ball was loose > retro_block_min_loose_s
    (None, 0.0, True),  # no gain: the ball went dead (or the stream ended) first
])
def test_commit_retro_block_loose_ball_gate(gainer: str | None, after_s: float, kept: bool) -> None:
    eng = _engine_with_players({"home-1": (19.0, 25.0), "home-2": (24.0, 22.0), "away-1": (17.6, 24.4),
                                "away-3": (10.0, 35.0), "away-5": (17.9, 24.7)})
    cfg = eng.config
    r = Release("home-1", Team.HOME, 0.2, 6, (19.0, 25.0), (19.0, 25.0))
    flight = Flight(r, eng.court, cfg)
    t_touch = 0.3
    flight.retro_block = (t_touch, 9, "away-1", "left", {"reversal_dv": 20.0, "retro": True, "rise_t": 0.2,
                                                          "rise_frame": 6})
    eng._commit_retro_block(flight, gain=(t_touch + after_s, gainer) if gainer is not None else None)
    blocks = live(eng.events, EventKind.BLOCK)
    if kept:
        assert [(b.player_id, b.other_player_id) for b in blocks] == [("away-1", "home-1")]
        assert eng._retro_dropped_t == -math.inf
        return
    assert not blocks and flight.retro_block is None
    assert eng._retro_dropped_t == pytest.approx(t_touch)  # later look-backs skip touches up to here
    eng._commit_retro_block(flight)  # the dead ball / end of stream that follows does not revive it
    assert not live(eng.events, EventKind.BLOCK)


# ---------------------------------------------------------------------------- 38d1655: monocular long-flight gains


def _lob_to_opponent(duration: float, apex: float) -> Scene:
    """home-1 dribbles (his control is corroborated), then loses the ball to away-2 ~19 ft away after a flight
    of ``duration``; away-2 dribbles away."""
    s = Scene()
    s.add("home-1", 40.0, 20.0).add("home-2", 62.0, 12.0).add("away-1", 44.0, 24.0).add("away-2", 56.0, 30.0)
    s.start("home-1").dribble("home-1", 1.5, 3)
    s.mark("release")
    s.pass_to("away-2", duration=duration, apex=apex)
    s.mark("gain")
    s.dribble("away-2", 1.5, 3, move=(60.0, 30.0))
    s.hold("away-2", 0.5)
    return s


def test_mono_opponent_gain_long_after_the_release_changes_possession_without_a_steal() -> None:
    s = _lob_to_opponent(2.2, 18.0)
    ev = live(run_engine(s.frames_mono()).events)
    assert not live(ev, EventKind.STEAL) and not live(ev, EventKind.TURNOVER)
    poss = [e for e in live(ev, EventKind.POSSESSION) if e.team is Team.AWAY]
    assert len(poss) == 1 and poss[0].player_id == "away-2" and poss[0].attrs["cause"] == "other"
    assert poss[0].t - s.marks["release"] > EventEngineConfig().steal_max_flight_s
    # steal_max_flight_s is what decides: without it the same gain is a steal
    ev2 = live(run_engine(s.frames_mono(), EventEngineConfig(steal_max_flight_s=100.0)).events)
    assert one(ev2, EventKind.STEAL).player_id == "away-2"


def test_mono_quick_interception_is_still_a_steal() -> None:
    s = _lob_to_opponent(0.6, 1.5)
    ev = live(run_engine(s.frames_mono()).events)
    st, to = one(ev, EventKind.STEAL), one(ev, EventKind.TURNOVER)
    assert (st.player_id, st.other_player_id) == ("away-2", "home-1") and abs(st.t - s.marks["gain"]) < TOL
    assert (to.player_id, to.attrs["type"]) == ("home-1", "bad_pass")
    poss = [e for e in live(ev, EventKind.POSSESSION) if e.team is Team.AWAY]
    assert len(poss) == 1 and poss[0].attrs["cause"] == "steal"


# ---------------------------------------------------------------------------- 38d1655: firm dribbles


def test_bounce_after_a_release_is_a_dribble_but_not_a_firm_one() -> None:
    """3-D dribble excursion: home-1 pushes the ball out ahead (past his 3 ft keep radius, so nobody holds it),
    it bounces 3.8 ft from him and he takes it again (the same stint). That bounce counts as one of his dribbles but
    not as a firm one; the two bounces he then makes while holding the ball are firm."""
    s = two_team_scene()
    s.add("home-1", 30.0, 25.0).add("away-1", 34.0, 30.0)
    s.start("home-1").hold("home-1", 1.0)
    st, mid, low = s.ball, (26.6, 24.3, 3.0), (26.3, 24.3, BALL_R)
    assert st is not None
    s.run(0.2, lambda a, pos: tuple(st[i] + a * (mid[i] - st[i]) for i in range(3)))  # type: ignore[arg-type,return-value]
    s.run(0.15, lambda a, pos: tuple(mid[i] + a * (low[i] - mid[i]) for i in range(3)))  # type: ignore[arg-type,return-value]
    s.mark("bounce")
    s.run(0.3, lambda a, pos: (low[0] + 0.1 * a, low[1], BALL_R + a * (HANDS_Z - BALL_R)), {"home-1": (26.5, 25.0)})
    s.mark("regain")
    s.hold("home-1", 0.6)
    s.dribble("home-1", 1.2, 2)
    s.hold("home-1", 0.5)
    eng = EventEngine(court_for("nba"))
    loose_bounce = regained = None
    for fr in s.frames_3d():
        eng.process(fr)
        pc = eng._prev_control
        if loose_bounce is None and eng._control is None and pc is not None and pc.dribbles == 1:
            loose_bounce = (eng.holder_id, pc.player_id, pc.dribbles, pc.firm_dribbles)
        if regained is None and fr.t >= s.marks["regain"] + 0.2:
            c = eng._control
            regained = c and (c.player_id, c.onset_t, c.dribbles, c.firm_dribbles)
    eng.flush()
    assert loose_bounce == (None, "home-1", 1, 0)  # counted for the releaser while nobody held the ball ...
    assert regained == ("home-1", 0.0, 1, 0)  # ... in the stint he resumed, still not firm
    c = eng._control
    assert c is not None and (c.player_id, c.dribbles, c.firm_dribbles) == ("home-1", 3, 2)
    dr = live(eng.events, EventKind.DRIBBLE)
    assert len(dr) == 3 and abs(dr[0].t - s.marks["bounce"]) <= 0.07


# ---------------------------------------------------------------------------- f3e5e81: monocular pass transit


def _transit_players(far_passer: bool = False) -> Scene:
    s = Scene()
    s.add("home-1", 62.0 if far_passer else 34.0, 22.0).add("home-2", 28.0, 22.0).add("home-3", 20.0, 22.0)
    s.add("away-1", 30.0, 33.0).add("away-2", 14.0, 36.0).add("away-3", 75.0 if far_passer else 40.0, 10.0)
    s.start("home-1").dribble("home-1", 1.0, 2).hold("home-1", 0.3)
    s.mark("release")
    return s


def _made_by(s: Scene, shooter: str) -> Scene:
    s.hold(shooter, 0.6)
    s.mark("shot")
    s.shoot("left", made=True)
    s.drop_through("left")
    s.set_score(2, 0)
    return s.rest(0.8)


def _crossing_pass() -> Scene:
    """home-1 (A) passes to home-3 (C) straight through home-2's (B) hands line: in the image the ball crosses
    B's box for a few frames in flight."""
    s = _transit_players()
    s.pass_to("home-3", duration=0.9, apex=1.5)
    return _made_by(s, "home-3")


def _touch_pass(hold_s: float = 0.1, dribble: bool = False, back: bool = False, far_passer: bool = False) -> Scene:
    """A passes to B, who has the ball for ``hold_s`` (or bounces it once) and sends it on to C (or back to A)."""
    s = _transit_players(far_passer)
    s.pass_to("home-2", duration=1.3 if far_passer else 0.35)
    if dribble:
        s.dribble("home-2", 0.4, 1)
    else:
        s.hold("home-2", hold_s)
    s.mark("release2")
    receiver = "home-1" if back else "home-3"
    s.pass_to(receiver, duration=0.35 if back else 0.4)
    return _made_by(s, receiver)


@pytest.mark.parametrize("scene", ["crossing", "touch"])
def test_mono_pass_crossing_a_teammates_box_is_one_pass_to_the_next_holder(scene: str) -> None:
    s = _crossing_pass() if scene == "crossing" else _touch_pass()
    eng = run_engine(s.frames_mono())
    ev = live(eng.events)
    p = one(ev, EventKind.PASS)
    assert (p.player_id, p.other_player_id) == ("home-1", "home-3") and abs(p.t - s.marks["release"]) < TOL
    assert p.revision >= 1 and p.attrs["transit"] == ["home-2"] and p.attrs["completed"] is True
    assert not [e for e in eng.events if e.kind is EventKind.PASS and e.player_id == "home-2"]
    made = one(ev, EventKind.SHOT_MADE)
    ast = one(ev, EventKind.ASSIST)
    assert (ast.player_id, ast.other_player_id, ast.shot_id) == ("home-1", "home-3", made.shot_id)
    assert all(r.player_id != "home-2" for r, _ in eng._release_log)  # B's release is no later look-back origin


def _assert_two_passes(eng: EventEngine, second: tuple[str, str] = ("home-2", "home-3")) -> None:
    ev = live(eng.events)
    assert _passes(ev) == [("home-1", "home-2"), second]
    assert not any(p.attrs.get("transit") for p in live(ev, EventKind.PASS))
    assert one(ev, EventKind.ASSIST).player_id == "home-2"


def test_mono_no_transit_when_the_teammate_dribbles() -> None:
    """B bounces the ball once: he had it (even with the stint bound relaxed, so only the dribble decides)."""
    relaxed = EventEngineConfig(pass_transit_max_s=1.0)
    _assert_two_passes(run_engine(_touch_pass(dribble=True).frames_mono(), relaxed))
    assert one(run_engine(_touch_pass(hold_s=0.6).frames_mono(), relaxed).events, EventKind.PASS).attrs["transit"] == \
        ["home-2"]  # (the same config turns a dribble-less stint of that length into a transit)


def test_mono_no_transit_when_the_teammate_holds_the_ball_pass_transit_max_s_or_longer() -> None:
    s = _touch_pass(hold_s=0.6)
    _assert_two_passes(run_engine(s.frames_mono()))
    relaxed = live(run_engine(s.frames_mono(), EventEngineConfig(pass_transit_max_s=1.0)).events)
    assert _passes(relaxed) == [("home-1", "home-3")]


def test_mono_no_transit_when_the_receiver_gains_more_than_pass_transit_flight_s_after_the_first_release() -> None:
    """A long flat pass (34 ft, 1.3 s) to B, a 0.1 s touch, on to C: C gains ~1.6 s after A's release."""
    s = _touch_pass(far_passer=True)
    _assert_two_passes(run_engine(s.frames_mono()))
    relaxed = live(run_engine(s.frames_mono(), EventEngineConfig(pass_transit_flight_s=100.0)).events)
    assert _passes(relaxed) == [("home-1", "home-3")]


def test_mono_ball_straight_back_to_the_first_passer_stays_two_passes() -> None:
    eng = run_engine(_touch_pass(back=True).frames_mono())
    _assert_two_passes(eng, second=("home-2", "home-1"))


def test_transit_never_applies_in_3d() -> None:
    """The same quick touch pass with exact positions is a catch-and-pass: two passes and B's assist, although
    B's stint is under pass_transit_max_s and C gains within pass_transit_flight_s of A's release."""
    s = _touch_pass()
    cfg = EventEngineConfig()
    eng = EventEngine(court_for("nba"), config=cfg)
    b_onset = c_onset = None
    for fr in s.frames_3d():
        eng.process(fr)
        c = eng._control
        if c is not None and c.player_id == "home-2" and b_onset is None:
            b_onset = c.onset_t
        if c is not None and c.player_id == "home-3" and c_onset is None:
            c_onset = c.onset_t
    eng.flush()
    _assert_two_passes(eng)
    first, second = live(eng.events, EventKind.PASS)
    assert b_onset is not None and c_onset is not None
    assert second.t - b_onset < cfg.pass_transit_max_s and c_onset - first.t <= cfg.pass_transit_flight_s
