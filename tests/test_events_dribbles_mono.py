"""Monocular dribble counting and the scoreboard flip window (round 4, assists family).

* The image bounce detector measures the ball's height above the dribbler's feet in box heights: a player driving away
  from the camera moves his floor point up the image by about a box height per second, which in raw image rows
  cancels the fall into each bounce (the bounces after the first were lost, so a 4-dribble drive read as one dribble
  and its made layup as assisted).
* A low point needs a fall into it (``EventEngine._came_down``): a ball lifted off the floor by the player who
  just gained it rises from the floor like a bounce, but nothing came down into it.
* An assist candidate never names the scorer himself.
"""

from __future__ import annotations

from courtvision.events.engine import EventEngineConfig
from courtvision.events.reconcile import Action, Reconciler, ShotEntry
from courtvision.events.scoreboard import ScoreChange
from courtvision.types import BallState, EventKind, FrameState, Team

from test_events_kit import BALL_R, Scene, live, run_engine, two_team_scene


def _hide_top(s: Scene, frames: list[FrameState], above_z: float) -> list[FrameState]:
    """The torso hides a dribble's top on the far side of the body: the ball is unseen above ``above_z`` ft."""
    for rec, fr in zip(s.recs, frames, strict=True):
        if rec.ball is not None and rec.ball[2] > above_z:
            fr.ball = BallState(visible=False)
    return frames


def _drive(away_from_camera: bool) -> Scene:
    s = two_team_scene()
    y0, y1 = (12.0, 40.0) if away_from_camera else (40.0, 12.0)
    s.add("home-1", 30.0, y0).add("away-1", 34.0, 25.0)
    s.start("home-1").hold("home-1", 0.5)
    s.mark("drive")
    s.dribble("home-1", 2.0, 4, move=(30.0, y1))  # 14 ft/s towards / away from the camera (it sits at y = -56 ft)
    s.hold("home-1", 0.5)
    return s


def test_mono_drive_away_from_the_camera_counts_every_bounce() -> None:
    """home-1 drives 28 ft straight away from the camera with 4 dribbles; the top of each dribble (above hip height)
    is hidden by his body. Every bounce is counted, on time."""
    s = _drive(away_from_camera=True)
    eng = run_engine(_hide_top(s, s.frames_mono(), 2.3))
    got = [e.t for e in live(eng.events, EventKind.DRIBBLE) if e.player_id == "home-1"]
    assert len(got) == 4, got
    assert all(abs(a - b) <= 0.1 for a, b in zip(got, s.dribble_times, strict=True)), (got, s.dribble_times)


def test_mono_drive_towards_the_camera_counts_every_bounce() -> None:
    """Guard, the mirror case: driving towards the camera (floor point moving down the image)."""
    s = _drive(away_from_camera=False)
    eng = run_engine(_hide_top(s, s.frames_mono(), 2.3))
    got = [e.t for e in live(eng.events, EventKind.DRIBBLE) if e.player_id == "home-1"]
    assert len(got) == 4, got


def test_mono_ball_lifted_off_the_floor_by_its_new_holder_is_not_a_dribble() -> None:
    """A loose ball rolls to home-1's feet and he lifts it straight to his hands: it rises from the floor as after a
    bounce, but it never came down into that low point, so it is no dribble (whoever is credited with the gain)."""
    s = two_team_scene()
    s.add("home-1", 30.0, 20.0)
    s.start(None, (36.0, 18.0, BALL_R))
    s.run(0.6, lambda a, pos: (36.0 - 6.0 * a, 18.0 + 1.0 * a, BALL_R))  # rolls in front of his feet
    s.run(0.3, lambda a, pos: (30.0, 19.0, BALL_R + (4.0 - BALL_R) * a))  # picked up to the hands
    s.hold("home-1", 1.0)
    eng = run_engine(s.frames_mono())
    assert not live(eng.events, EventKind.DRIBBLE), [(e.t, e.player_id) for e in live(eng.events, EventKind.DRIBBLE)]


def test_mono_standing_dribble_still_counts() -> None:
    """Guard: a stationary dribbler's bounces (full arc visible) are all counted."""
    s = two_team_scene()
    s.start("home-1").hold("home-1", 0.5).dribble("home-1", 1.5, 3).hold("home-1", 0.5)
    eng = run_engine(s.frames_mono())
    assert len([e for e in live(eng.events, EventKind.DRIBBLE) if e.player_id == "home-1"]) == 3


# ---------------------------------------------------------------------------- assist candidate


def test_assist_candidate_never_names_the_scorer() -> None:
    """A stint whose recorded pass-in is the scorer's own (a pass re-addressed back to its passer) gives no assist;
    a teammate's pass within the window does."""
    from courtvision.events.engine import EventEngine, _Control, _PassRec
    from courtvision.events.shots import Release
    from courtvision.geometry.court import court_for

    eng = EventEngine(court_for("nba"))
    rel = Release("away-42", Team.AWAY, 10.0, 300, (10.0, 25.0), (10.0, 25.0), None, 8.0)
    own = _Control("away-42", Team.AWAY, 8.0, 240, origin="pass", via_pass=_PassRec("away-42", Team.AWAY, 7.5, "e1"))
    assert eng._assist_candidate(own, rel) is None
    mate = _Control("away-42", Team.AWAY, 8.0, 240, origin="pass", via_pass=_PassRec("away-44", Team.AWAY, 7.5, "e1"))
    assert eng._assist_candidate(mate, rel) == ("away-44", Team.AWAY)


# ---------------------------------------------------------------------------- scoreboard flip (guard)


def _miss(result_t: float) -> ShotEntry:
    return ShotEntry("s0001", Team.HOME, "home-1", 2, "e1", result_t - 1.0, result_id="e2", made=False,
                     result_t=result_t)


def _delta(t: float) -> ScoreChange:
    return ScoreChange(t=t, team=Team.HOME, delta=2, home=2, away=0, period=1, game_clock=590.0, source="sim",
                       confirmed_t=t)


def test_reconciler_flips_a_miss_within_the_board_lag() -> None:
    """Guard: the same delta 1.5 s after the miss (an operator's lag) flips it."""
    cfg = EventEngineConfig()
    rec = Reconciler(cfg)
    ch = _delta(10.0)
    rec.add(ch)
    assert rec.step(10.0 + cfg.reconcile_flip_hold_s, [_miss(8.5)], streaming=True) == \
        [Action("flip_to_made", "s0001", ch, new_value=2)]


# ---------------------------------------------------------------------------- noise robustness


def _hip_hold(s: Scene, pid: str, duration: float, z: float = 3.0) -> Scene:
    """A ball held still at hip height, a foot in front of the holder (a triple-threat stance)."""
    return s.run(duration, lambda a, pos: (pos[pid][0], pos[pid][1] - 1.0, z))


def test_mono_ball_held_still_at_hip_height_is_no_dribble_under_noise() -> None:
    """A ball held still at hip height for 3 s under box and ball jitter (the bounce signal adds the box bottom's
    jitter to the ball's) is no dribble, on several noise seeds."""
    for seed in range(6):
        s = two_team_scene()
        s.start("home-1").hold("home-1", 0.5)
        _hip_hold(s, "home-1", 3.0)
        eng = run_engine(s.frames_mono(seed=seed, px_noise=2.0))
        got = [round(e.t, 2) for e in live(eng.events, EventKind.DRIBBLE) if e.player_id == "home-1"]
        assert not got, (seed, got)


def test_mono_catch_hold_at_the_hip_then_score_keeps_the_assist_under_noise() -> None:
    """A pass, a 2 s hold at the hip without a dribble, then a made jumper: assisted (no false dribbles), on several
    noise seeds."""
    kept = 0
    for seed in range(6):
        s = two_team_scene(clock=600.0)
        s.add("home-1", 18.0, 30.0).add("home-2", 30.0, 38.0)  # (test_events_regressions' assisted make)
        s.start("home-2").hold("home-2", 0.5).pass_to("home-1")
        _hip_hold(s, "home-1", 2.0)
        s.hold("home-1", 0.3).shoot("left", made=True).drop_through("left")
        s.set_score(2, 0)
        s.rest(2.0, visible=False)
        ev = live(run_engine(s.frames_mono(seed=seed, px_noise=2.0)).events)
        kept += any(e.kind is EventKind.ASSIST and e.player_id == "home-2" for e in ev)
    assert kept == 6, kept
