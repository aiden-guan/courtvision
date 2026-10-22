"""Regression tests for the round-4 shot / turnover changes (monocular).

* ``shot_max_air_s``: a released ball the camera lost, still "in the air" long after any shot would have reached the
  rim, no longer holds back the dead ball of a game-clock stop (a pass thrown out of bounds).
* ``flip_revisit_s``: a score change does not flip a miss when the ball was back at that rim well after the miss (a
  putback the engine did not see go in): the putback is inferred instead, with the offensive rebound it implies.

Scenes reuse the kit in ``test_events_kit.py``; each has a contrast run with the rule switched off.
"""

from __future__ import annotations

from courtvision.events.engine import EventEngineConfig
from courtvision.events.reconcile import Action, Reconciler, ShotEntry
from courtvision.events.scoreboard import ScoreChange
from courtvision.types import BallState, EventKind, FrameState, Team

from test_events_kit import BALL_R, Scene, live, one, run_engine, two_team_scene

# ---------------------------------------------------------------------------- shot_max_air_s


def _lost_pass_out_of_bounds() -> Scene:
    """home-1's pass goes out over the near sideline; the camera loses the ball 0.4 s after the release and the
    game clock stops when it goes out. 4 s later away-2, read in bounds near the line (monocular foot points), has
    the dead ball and passes it in to away-1."""
    s = two_team_scene(clock=600.0)
    s.start("home-1").hold("home-1", 1.0)
    s.mark("release")
    s.fly((26.0, 10.0, 3.0), 0.4, apex=1.0)
    s.run(0.3, lambda a, pos: (24.0, -3.0, BALL_R), visible=False)
    s.mark("out")
    s.clock_running = False
    s.run(4.0, lambda a, pos: (24.0, -3.0, BALL_R), moves={"away-2": (24.0, 2.5)}, visible=False)
    s.run(0.05, lambda a, pos: s.hands("away-2", pos))
    s.mark("pickup")
    s.hold("away-2", 1.2)
    s.pass_to("away-1")
    s.mark("inbound")
    s.hold("away-1", 1.5)
    return s


def test_clock_stop_with_a_lost_ball_in_the_air_is_the_dead_ball() -> None:
    s = _lost_pass_out_of_bounds()
    frames = s.frames_mono()
    ev = live(run_engine(frames).events)
    to = one(ev, EventKind.TURNOVER)
    assert to.team is Team.HOME and to.player_id == "home-1"
    assert abs(to.t - s.marks["out"]) <= 0.5  # dated at the clock stop, not at the inbound
    poss = [e for e in live(ev, EventKind.POSSESSION) if e.team is Team.AWAY]
    assert len(poss) == 1 and poss[0].attrs["cause"] == "turnover"
    assert not live(ev, EventKind.STEAL)
    # the air-time bound is what decides: a flight that may stay "in the air" forever holds the stop back, and the
    # away gain is a plain change of possession
    old = live(run_engine(frames, EventEngineConfig(shot_max_air_s=1e9)).events)
    assert not live(old, EventKind.TURNOVER)


# ---------------------------------------------------------------------------- flip_revisit_s: reconciler


def _delta(t: float, team: Team = Team.HOME, delta: int = 2) -> ScoreChange:
    return ScoreChange(t=t, team=team, delta=delta, home=delta, away=0, period=1, game_clock=590.0, source="sim",
                       confirmed_t=t)


def _miss(shot_id: str, result_t: float) -> ShotEntry:
    return ShotEntry(shot_id, Team.HOME, "home-1", 2, f"a-{shot_id}", result_t - 0.8, result_id=f"r-{shot_id}",
                     made=False, result_t=result_t)


def test_reconciler_does_not_flip_a_miss_the_ball_went_back_to_the_rim_after() -> None:
    cfg = EventEngineConfig()
    ch = _delta(10.0)
    miss = _miss("s0001", 7.0)
    rec = Reconciler(cfg)
    rec.add(ch)
    assert rec.step(10.0 + cfg.reconcile_flip_hold_s, [miss], streaming=True, revisited=lambda s, c: True) == []
    assert rec.step(10.0 + cfg.reconcile_hold_s, [miss], streaming=True, revisited=lambda s, c: True) == \
        [Action("correction", None, ch)]
    # guard: without a revisit the same miss is flipped
    rec2 = Reconciler(cfg)
    rec2.add(ch)
    assert rec2.step(10.0 + cfg.reconcile_flip_hold_s, [miss], streaming=True, revisited=lambda s, c: False) == \
        [Action("flip_to_made", "s0001", ch, new_value=2)]


# ---------------------------------------------------------------------------- flip_revisit_s: engine


def _unseen_putback() -> tuple[Scene, list[FrameState]]:
    """home-1 misses a jumper off the front rim; home-2 rebounds under the basket and puts it back up. The camera
    sees the ball only at the rim as it drops through (the putback release is hidden in the crowd under the
    basket), so the engine never sees home-2 release it; the board shows +2 a second after the make."""
    s = two_team_scene(clock=600.0)
    s.add("home-2", 9.0, 21.0)
    s.start("home-1").hold("home-1", 0.8)
    s.mark("shot")
    s.shoot("left", made=False)
    s.mark("rim")
    s.pass_to("home-2", duration=0.6, apex=1.0)
    s.hold("home-2", 1.0)
    top = s.basket("left", s.court.rim_height + 1.0)
    start = s.ball
    assert start is not None
    s.run(0.5, lambda a, pos: tuple(start[i] + a * (top[i] - start[i]) for i in range(3)), visible=False)  # type: ignore[arg-type]
    s.mark("putback")
    s.run(2.0 / s.fps, lambda a, pos: (top[0], top[1], top[2] - 0.8 * a))  # two frames at the rim, then hidden again
    s.mark("made")
    s.run(0.6, lambda a, pos: (top[0], top[1], BALL_R), visible=False)
    s.set_score(2, 0)
    s.mark("board")
    s.rest(4.0, visible=False)
    frames = s.frames_mono()
    return s, frames


def test_putback_the_engine_did_not_see_is_inferred_instead_of_flipping_the_miss() -> None:
    s, frames = _unseen_putback()
    ev = live(run_engine(frames).events)
    miss = one(ev, EventKind.SHOT_MISSED)
    assert miss.player_id == "home-1"
    made = one(ev, EventKind.SHOT_MADE)
    assert made.player_id == "home-2" and "putback" in made.attrs["reason"]
    assert s.marks["putback"] - 0.2 <= made.t <= s.marks["made"] + 0.2
    atts = sorted(live(ev, EventKind.SHOT_ATTEMPT), key=lambda e: e.t)
    assert [a.player_id for a in atts] == ["home-1", "home-2"]
    assert made.t - 1.0 <= atts[1].t <= made.t
    reb = one(ev, EventKind.REBOUND)
    assert reb.team is Team.HOME and reb.attrs["offensive"] and reb.shot_id == miss.shot_id
    # the revisit test is what decides: without it the +2 flips home-1's miss
    old = live(run_engine(frames, EventEngineConfig(flip_revisit_s=0.0)).events)
    assert one(old, EventKind.SHOT_MADE).player_id == "home-1"
    assert not live(old, EventKind.SHOT_MISSED)


def test_without_a_sighting_back_at_the_rim_the_miss_is_flipped_as_before() -> None:
    """Guard: the same play with the ball never seen back at the rim after the miss (a make the image test called
    missed looks like this: it drops through the net and is taken out below the rim): the +2 flips home-1's miss."""
    s, frames = _unseen_putback()
    for fr in frames:
        if s.marks["putback"] - 0.05 <= fr.t <= s.marks["made"] + 0.05:
            fr.ball = BallState(visible=False)
    ev = live(run_engine(frames).events)
    made = one(ev, EventKind.SHOT_MADE)
    assert made.player_id == "home-1" and made.attrs.get("flipped_by") == "scoreboard"
    assert len(live(ev, EventKind.SHOT_ATTEMPT)) == 1 and not live(ev, EventKind.SHOT_MISSED)


def test_a_free_throw_change_is_never_vetoed_by_a_rim_revisit() -> None:
    """A +1 is a free throw: an unseen make would never infer it (only +2 / +3), so a revisit must not stop the
    reconciler from flipping a missed free throw (the veto would end in a SCORE_CORRECTION and keep a false miss).
    A +2 after the same sighting is vetoed (the guard)."""
    from courtvision.geometry.court import court_for
    from courtvision.events.engine import EventEngine
    from courtvision.types import Event

    eng = EventEngine(court_for("nba"))
    eng._store["a-s0001"] = Event("a-s0001", EventKind.SHOT_ATTEMPT, 6.2, 186, team=Team.HOME, player_id="home-1",
                                  attrs={"basket": "left"})
    eng._rim_seen["left"].append(9.5)  # the ball back at that rim 2.5 s after the miss
    miss = _miss("s0001", 7.0)
    assert eng._rim_revisit(miss, _delta(10.0, delta=2)) is True
    assert eng._rim_revisit(miss, _delta(10.0, delta=1)) is False
