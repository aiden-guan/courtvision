"""Regression tests for defects found in the adversarial review of the events module.

Each test names the defect it pins down. Scenes reuse the kit in ``test_events_kit.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from courtvision.events.calibration import _isotonic
from courtvision.events.engine import EventEngine, EventEngineConfig
from courtvision.events.scoreboard import ClockZeroWatch, ScoreboardConfig
from courtvision.geometry.court import court_for
from courtvision.types import BallState, Event, EventKind, EventStatus, FrameState, ScoreboardReading, Team

from test_events_kit import BALL_R, HANDS_Z, MODES, Scene, kinds, live, one, run_engine, two_team_scene

G = 32.174


def ocr(t: float, clock: float | None, shot_clock: float | None = None, period: int = 1,
        score: tuple[int, int] = (0, 0)) -> ScoreboardReading:
    return ScoreboardReading(t, score[0], score[1], period, clock, shot_clock, source="ocr", confidence=0.95)


def ocr_frames(frames: list[FrameState], read: callable) -> list[FrameState]:  # type: ignore[valid-type]
    """5 Hz OCR reads (every 6th frame at 30 fps) from ``read(t)``."""
    for fr in frames:
        fr.scoreboard = read(fr.t) if fr.frame_idx % 6 == 0 else None
    return frames


def passes_scene(**kw: object) -> Scene:
    s = two_team_scene(**kw).start("home-1").hold("home-1", 1.0).pass_to("home-2").hold("home-2", 0.6)
    s.mark("glitch")
    s.hold("home-2", 0.6).pass_to("home-1").hold("home-1", 0.6).pass_to("home-3").hold("home-3", 0.6)
    return s


# ---------------------------------------------------------------------------- clocks (D14 for clocks)


@pytest.mark.parametrize("reset_first", [True, False])
def test_period_number_before_clock_reset_does_not_end_the_new_period(reset_first: bool) -> None:
    """Defect: a scoreboard showing the new period while the clock still reads 0:00 ended the new
    period at once (spurious PERIOD_END) and the engine ignored the whole period."""
    s = two_team_scene(clock=720.0)
    s.add("home-3", 47.0, -2.0)
    s.start("away-1").hold("away-1", 1.0)
    s.clock = 0.1
    s.hold("away-1", 0.2)
    if reset_first:
        s.clock, s.clock_running = 720.0, False
        s.hold("away-1", 0.3)
        s.period = 2
    else:
        s.period = 2
        s.hold("away-1", 0.3)
        s.clock, s.clock_running = 720.0, False
    s.rest(0.5, visible=False)
    s.run(0.05, lambda a, pos: s.hands("home-3", pos)).hold("home-3", 1.0)
    s.clock_running = True
    s.pass_to("home-1").hold("home-1", 0.6).pass_to("home-2").hold("home-2", 0.6)
    eng = run_engine(s.frames_3d())
    ev = live(eng.events)
    assert [e.value for e in live(ev, EventKind.PERIOD_END)] == [1]
    assert [e.value for e in live(ev, EventKind.PERIOD_START)] == [1, 2]
    assert len(live(ev, EventKind.PASS)) == 2 and eng.phase == "live"


def test_single_ocr_game_clock_misread_does_not_end_the_period() -> None:
    """Defect: one OCR read of 0:00 mid-period emitted PERIOD_END and blinded the engine."""
    s = passes_scene()
    glitch = s.marks["glitch"]
    seen: list[float] = []

    def read(t: float) -> ScoreboardReading:
        if not seen and t >= glitch:
            seen.append(t)
            return ocr(t, 0.0)
        return ocr(t, 600.0 - t)

    eng = run_engine(ocr_frames(s.frames_3d(), read))
    assert not live(eng.events, EventKind.PERIOD_END)
    assert len(live(eng.events, EventKind.PASS)) == 3 and eng.phase == "live"


def test_ocr_game_clock_running_out_ends_the_period_at_the_first_zero_read() -> None:
    s = two_team_scene(scoreboard=False).start("away-1").hold("away-1", 2.0)
    eng = run_engine(ocr_frames(s.frames_3d(), lambda t: ocr(t, max(0.0, 1.0 - t))))
    end = one(eng.events, EventKind.PERIOD_END)
    assert end.value == 1 and end.t == pytest.approx(1.0)  # 2nd zero read at 1.2 s confirms it


def test_single_ocr_shot_clock_misread_is_not_a_violation() -> None:
    """Defect: one OCR read of the shot clock as 0 gave a shot-clock TURNOVER."""
    s = passes_scene(shot_clock=20.0)
    glitch = s.marks["glitch"]
    seen: list[float] = []

    def read(t: float) -> ScoreboardReading:
        sc = 20.0 - t
        if not seen and t >= glitch:
            seen.append(t)
            sc = 0.0
        return ocr(t, 600.0 - t, sc)

    eng = run_engine(ocr_frames(s.frames_3d(), read))
    assert not live(eng.events, EventKind.TURNOVER)


def test_ocr_shot_clock_expiry_still_detected() -> None:
    s = two_team_scene(scoreboard=False).start("away-1").dribble("away-1", 1.6, 3).hold("away-1", 0.5)
    eng = run_engine(ocr_frames(s.frames_3d(), lambda t: ocr(t, 600.0 - t, max(0.0, 1.0 - t))))
    to = one(eng.events, EventKind.TURNOVER)
    assert to.attrs["type"] == "shot_clock" and to.t == pytest.approx(1.0) and to.player_id == "away-1"


def test_clock_zero_watch_rules() -> None:
    cfg = ScoreboardConfig()
    w = ClockZeroWatch(cfg)
    assert w.update(0.0, 0.0, trusted=False) is None  # never seen running (new period still at 0:00)
    assert w.update(0.2, 3.0, trusted=False) is None
    assert w.update(3.2, 0.0, trusted=False) is None  # one read
    assert w.update(3.4, 0.0, trusted=False) == pytest.approx(3.2)
    assert w.update(3.6, 0.0, trusted=False) is None  # fires once
    w.reset()
    assert w.update(4.0, 300.0, trusted=True) is None
    assert w.update(4.2, 0.0, trusted=True) is None  # 300 s cannot run out in 0.2 s: a misread / reset
    assert w.update(4.4, float("nan"), trusted=True) is None
    w2 = ClockZeroWatch(cfg)
    w2.update(0.0, 0.03, trusted=True)
    assert w2.update(1 / 30, 0.0, trusted=True) == pytest.approx(1 / 30)  # trusted: one read


# ---------------------------------------------------------------------------- non-finite inputs


def test_non_finite_player_and_ball_positions_are_ignored() -> None:
    """Defect: a nan player position crashed the engine (geometry raises on non-finite input)."""
    s = two_team_scene()
    s.add("home-1", 18.0, 30.0)
    s.start("home-1").hold("home-1", 0.8).shoot("left").drop_through("left").rest(0.5)
    frames = s.frames_3d()
    for fr in frames:
        for p in fr.players:
            if p.player_id == "home-1" and 10 <= fr.frame_idx <= 26:
                p.x = math.nan
            if p.player_id == "away-2":
                p.vx = math.inf
    nan = math.nan
    frames[0].ball = BallState(True, (nan, nan), (nan, nan), nan, 4.0, 0.9)
    eng = run_engine(frames)
    assert kinds(eng.events) == ["possession", "shot_attempt", "shot_made"]
    assert one(eng.events, EventKind.SHOT_ATTEMPT).value == 2


# ---------------------------------------------------------------------------- holder detection


def test_strip_steal_by_a_dribbling_defender_is_timely() -> None:
    """Defect: a challenger's vertical (dribble) ball speed excluded him while the trailing holder's
    score ignored it, so a stealer who dribbled at once was never credited (3-D truth frames)."""
    s = two_team_scene()
    s.add("home-1", 30.0, 25.0).add("away-1", 30.5, 22.6)
    s.start("home-1").dribble("home-1", 1.0, 2)
    s.mark("strip")
    bounces = 3

    def ball(a: float, pos: dict[str, tuple[float, float]]) -> tuple[float, float, float]:
        x, y = pos["away-1"]
        return (x, y + 1.0, BALL_R + (HANDS_Z - BALL_R) * abs(math.cos(math.pi * bounces * a)))

    s.run(1.5, ball, {"away-1": (36.5, 22.6), "home-1": (36.0, 25.6)})  # home-1 trails ~2 ft from the ball
    for weight, credited in ((1.0, False), (0.0, True)):  # 1.0 = the old behaviour
        eng = run_engine(s.frames_3d(), EventEngineConfig(holder_rel_vertical_weight_held=weight))
        steals = live(eng.events, EventKind.STEAL)
        if not credited:
            assert not steals
            continue
        st = one(eng.events, EventKind.STEAL)
        assert (st.player_id, st.other_player_id) == ("away-1", "home-1") and 0.0 <= st.t - s.marks["strip"] < 0.8
        assert one(eng.events, EventKind.TURNOVER).attrs["type"] == "steal"


def test_loose_ball_in_free_flight_is_not_controlled_by_a_bystander() -> None:
    """Defect: a bounce near its apex (free fall, chest height) drifting past a player counted as
    control, giving a false rebound and a cascade of passes / steals."""
    s = two_team_scene()
    s.add("home-1", 20.0, 25.0).add("away-2", 12.1, 18.4).add("home-3", 17.5, 13.5)
    s.start("home-1").hold("home-1", 0.8).shoot("left", made=False)
    s.fly((7.0, 24.0, BALL_R), 0.45, apex=0.3)
    p0, v = (7.0, 24.0, BALL_R), (8.0, -8.0, 19.5)
    T = 2 * v[2] / G

    def ball(a: float, pos: dict[str, tuple[float, float]]) -> tuple[float, float, float]:
        t = a * T
        return (p0[0] + v[0] * t, p0[1] + v[1] * t, p0[2] + v[2] * t - 0.5 * G * t * t)

    s.run(T, ball)  # apex 6.3 ft about 0.8 ft from away-2
    s.fly(lambda pos: s.hands("home-3", pos), 0.25, apex=0.5)
    s.mark("gather")
    s.hold("home-3", 0.8)
    eng = run_engine(s.frames_3d())
    reb = one(eng.events, EventKind.REBOUND)
    assert reb.player_id == "home-3" and reb.attrs["offensive"] is True and abs(reb.t - s.marks["gather"]) < 0.35
    assert kinds(eng.events) == ["possession", "shot_attempt", "shot_missed", "rebound"]


# ---------------------------------------------------------------------------- blocks


@pytest.mark.parametrize("mode", MODES)
def test_release_of_a_ball_carried_above_holder_height_is_not_a_block(mode: str) -> None:
    """Defect: a layup / set-point ball above holder_max_z opens the flight in the shooter's hands;
    a one-frame position jump at the real release looked like a reversal by the nearby defender."""
    s = Scene()
    s.add("home-1", 14.0, 25.0).add("away-1", 12.4, 23.2).add("away-3", 20.0, 34.0)
    s.start("home-1").hold("home-1", 0.8)
    s.run(0.3, lambda a, pos: (14.0 - 1.5 * a, 24.3 + 0.2 * a, 4.0 + 4.8 * a))  # lifted above 7.5 ft
    s.run(1 / 30, lambda a, pos: (13.1, 24.1, 8.8))  # the ball jumps back at the real release
    s.shoot("left", made=True, release_z=8.8, apex_z=12.5).drop_through("left").rest(0.5)
    eng = run_engine(s.frames(mode))
    assert not live(eng.events, EventKind.BLOCK)
    assert kinds(eng.events) == ["possession", "shot_attempt", "shot_made"]


@pytest.mark.parametrize("mode", MODES)
def test_catch_next_to_a_defender_is_not_a_block(mode: str) -> None:
    """Defect: a rising pass caught (the ball knocked back into the receiver's hands and dribbled out)
    next to a defender was called shot attempt + block + rebound."""
    s = Scene()
    s.add("home-1", 19.5, 25.0).add("home-2", 12.0, 25.0).add("away-1", 10.8, 23.0).add("away-3", 30.0, 34.0)
    s.start("home-1").hold("home-1", 0.8)
    s.mark("release")
    s.fly((12.0, 24.3, 6.0), 0.35, apex=1.0)
    s.run(1 / 30, lambda a, pos: (12.6, 24.3, 5.6))
    bounces = 2

    def ball(a: float, pos: dict[str, tuple[float, float]]) -> tuple[float, float, float]:
        x, y = pos["home-2"]
        return (x + 0.6, y - 0.7, BALL_R + (5.4 - BALL_R) * abs(math.cos(math.pi * bounces * a)))

    s.run(1.0, ball, {"home-2": (18.0, 25.0)})
    s.hold("home-2", 0.5)
    eng = run_engine(s.frames(mode))
    assert not live(eng.events, EventKind.BLOCK) and not live(eng.events, EventKind.SHOT_ATTEMPT)
    p = one(eng.events, EventKind.PASS)
    assert (p.player_id, p.other_player_id) == ("home-1", "home-2")


# ---------------------------------------------------------------------------- dead balls and free throws


def test_dead_ball_bounce_next_to_a_bystander_does_not_restart_play() -> None:
    """Defect: after a make, a loose bounce next to whoever last 'held' it counted as his dribble,
    restarted play, and the ball rolling to the inbounder became an out-of-bounds TURNOVER."""
    s = two_team_scene()
    s.add("home-1", 18.0, 30.0).add("home-3", 8.0, 22.0).add("away-4", 14.0, -1.5)
    s.start("home-1").hold("home-1", 0.8).shoot("left", made=True).drop_through("left")
    s.set_score(2, 0)
    s.fly((7.6, 22.6, 3.6), 0.4, apex=1.5)
    s.run(0.2, lambda a, pos: (7.6 + 0.4 * a, 22.6 - 0.3 * a, 3.6 - 0.3 * a))  # drifts slowly past home-3
    s.fly((10.5, 19.3, BALL_R), 0.35, apex=0.2)  # bounces 3.7 ft from him ...
    s.fly((12.0, 17.5, 2.0), 0.3, apex=0.5)
    s.fly(lambda pos: s.hands("away-4", pos), 1.0, apex=1.0)  # ... and rolls out to the inbounder
    s.hold("away-4", 1.0)
    s.pass_to("away-1").hold("away-1", 0.6)
    eng = run_engine(s.frames_3d())
    assert not live(eng.events, EventKind.TURNOVER)
    poss = [e for e in live(eng.events, EventKind.POSSESSION) if e.team is Team.AWAY]
    assert len(poss) == 1 and poss[0].attrs["cause"] == "made_shot" and poss[0].player_id == "away-4"


@pytest.mark.parametrize("mode", MODES)
def test_live_play_jumper_from_the_ft_line_is_a_field_goal(mode: str) -> None:
    """Defect: an elbow / FT-line jumper after a pass, with players standing around the lane, was
    recognised as a FREE THROW (value 1) and put the engine in a dead ball."""
    s = Scene()
    s.add("home-2", 30.0, 25.0).add("home-1", 20.0, 25.0)
    s.add("away-2", 8.0, 34.0).add("home-3", 11.0, 34.0).add("away-3", 8.0, 16.0).add("home-4", 11.0, 16.0)
    s.add("away-1", 24.0, 28.0)
    s.start("home-2").hold("home-2", 0.6).pass_to("home-1").hold("home-1", 0.8)
    s.shoot("left", made=True).drop_through("left").rest(0.4)
    eng = run_engine(s.frames(mode))
    att = one(eng.events, EventKind.SHOT_ATTEMPT)
    assert att.value == 2 and att.attrs["shot_type"] != "free_throw"
    assert one(eng.events, EventKind.ASSIST).player_id == "home-2"


def test_offensive_foul_is_a_turnover_and_the_inbound_a_turnover_possession() -> None:
    """Missing semantics: an operator offensive foul gave an inferred 'other' turnover (confidence
    0.6, last holder) at the next inbound instead of an offensive-foul turnover by the fouler."""
    s = two_team_scene()
    s.add("away-3", 24.0, -2.0)
    s.start("home-1").hold("home-1", 1.0)
    s.mark("foul")
    s.rest(1.0, visible=False)
    s.run(0.05, lambda a, pos: s.hands("away-3", pos)).hold("away-3", 1.0)
    s.pass_to("away-1").hold("away-1", 0.6)
    eng = EventEngine(court_for("nba"))
    for fr in s.frames_3d():
        eng.process(fr)
        if fr.t == s.marks["foul"]:
            eng.add_manual(Event("", EventKind.FOUL, fr.t, fr.frame_idx, team=Team.HOME, player_id="home-2",
                                 other_player_id="away-1", attrs={"type": "offensive"}))
    eng.flush()
    to = one(eng.events, EventKind.TURNOVER)
    assert (to.player_id, to.team, to.attrs["type"]) == ("home-2", Team.HOME, "offensive_foul")
    assert to.status is EventStatus.CONFIRMED and to.t == s.marks["foul"]
    poss = [e for e in live(eng.events, EventKind.POSSESSION) if e.team is Team.AWAY]
    assert len(poss) == 1 and poss[0].attrs["cause"] == "turnover" and poss[0].player_id == "away-3"


# ---------------------------------------------------------------------------- manual corrections


def _flip_scene() -> Scene:
    s = two_team_scene()
    s.add("home-1", 18.0, 30.0).add("home-2", 30.0, 38.0).add("away-3", 9.0, 25.0)
    s.start("home-2").hold("home-2", 0.5).pass_to("home-1").hold("home-1", 0.6)
    s.shoot("left", made=False)
    s.fly(lambda pos: s.hands("away-3", pos), 0.6, apex=3.0).hold("away-3", 0.6)
    s.mark("op")
    s.hold("away-3", 0.6)
    s.set_score(2, 0)  # the scoreboard catches up after the operator's correction
    s.hold("away-3", 3.0)
    return s


def test_manual_miss_to_make_has_the_flip_consequences_and_absorbs_the_delta() -> None:
    """Defect: an operator miss -> make left the defensive REBOUND / POSSESSION, derived no ASSIST,
    and the later scoreboard delta became a SCORE_CORRECTION (team points counted twice)."""
    s = _flip_scene()
    eng = EventEngine(court_for("nba"))
    for fr in s.frames_3d():
        eng.process(fr)
        if fr.t == s.marks["op"]:
            miss = one(eng.events, EventKind.SHOT_MISSED)
            eng.add_manual(Event(miss.event_id, EventKind.SHOT_MADE, miss.t, miss.frame_idx, team=miss.team,
                                 player_id=miss.player_id, value=2, shot_id=miss.shot_id))
    eng.flush()
    ev = eng.events
    made = one(ev, EventKind.SHOT_MADE)
    assert made.source == "manual" and made.attrs.get("confirmed_by") == "scoreboard"
    assert not live(ev, EventKind.SCORE_CORRECTION) and not live(ev, EventKind.REBOUND)
    assert [e.status for e in ev if e.kind is EventKind.REBOUND] == [EventStatus.RETRACTED]
    ast = one(ev, EventKind.ASSIST)
    assert (ast.player_id, ast.other_player_id, ast.shot_id) == ("home-2", "home-1", made.shot_id)
    poss = [e for e in live(ev, EventKind.POSSESSION) if e.team is Team.AWAY]
    assert len(poss) == 1 and poss[0].attrs["cause"] == "made_shot"


def test_manual_make_to_miss_retracts_the_assist() -> None:
    s = two_team_scene(scoreboard=False)
    s.add("home-1", 18.0, 30.0).add("home-2", 30.0, 38.0)
    s.start("home-2").hold("home-2", 0.5).pass_to("home-1").hold("home-1", 0.6)
    s.shoot("left", made=True).drop_through("left").rest(0.5)
    eng = run_engine(s.frames_3d())
    made = one(eng.events, EventKind.SHOT_MADE)
    assert one(eng.events, EventKind.ASSIST).player_id == "home-2"
    eng.add_manual(Event(made.event_id, EventKind.SHOT_MISSED, made.t, made.frame_idx, team=made.team,
                         player_id=made.player_id, value=2, shot_id=made.shot_id))
    assert one(eng.events, EventKind.SHOT_MISSED).source == "manual"
    assert not live(eng.events, EventKind.ASSIST) and not live(eng.events, EventKind.SHOT_MADE)


def test_operator_inserted_basket_absorbs_its_scoreboard_delta() -> None:
    """Defect: a basket the engine never saw, inserted by the operator, was unknown to the
    reconciler, so its scoreboard delta also became a SCORE_CORRECTION."""
    s = two_team_scene().start("home-1").hold("home-1", 1.0)
    s.mark("op")
    s.hold("home-1", 0.5)
    s.set_score(0, 2).hold("home-1", 3.0)
    eng = EventEngine(court_for("nba"))
    for fr in s.frames_3d():
        eng.process(fr)
        if fr.t == s.marks["op"]:
            eng.add_manual(Event("", EventKind.SHOT_MADE, fr.t, fr.frame_idx, team=Team.AWAY, player_id="away-2",
                                 value=2, shot_id="op-1"))
    eng.flush()
    assert not live(eng.events, EventKind.SCORE_CORRECTION)
    made = one(eng.events, EventKind.SHOT_MADE)
    assert made.source == "manual" and made.attrs.get("confirmed_by") == "scoreboard"


def test_engine_ids_never_overwrite_an_operator_event() -> None:
    """Defect: an operator / feed event carrying an id in the engine's own range was overwritten
    when the engine later emitted that id."""
    eng = EventEngine(court_for("nba"))
    eng.add_manual(Event("e000002", EventKind.TIMEOUT, 0.0, 0, team=Team.AWAY))
    for fr in passes_scene(scoreboard=False).frames_3d():
        eng.process(fr)
    by_id = {e.event_id: e for e in eng.events}
    assert by_id["e000002"].kind is EventKind.TIMEOUT
    assert len(by_id) == len(eng.events) and len(live(eng.events, EventKind.PASS)) == 3


def test_make_revised_to_miss_does_not_keep_the_make_confidence() -> None:
    s = two_team_scene()
    s.add("home-1", 18.0, 30.0).add("away-3", 4.0, 34.0)
    s.start("home-1").hold("home-1", 0.8).shoot("left", made=True).drop_through("left")
    s.fly(lambda pos: s.hands("away-3", pos), 0.4, apex=0.3).hold("away-3", 4.0)  # board stays 0-0
    cfg = EventEngineConfig()
    eng = run_engine(s.frames_3d(), cfg)
    miss = one(eng.events, EventKind.SHOT_MISSED)
    assert miss.attrs["needs_review"] is True and miss.confidence <= cfg.make_to_miss_conf


# ---------------------------------------------------------------------------- calibration


def test_isotonic_pools_tied_scores() -> None:
    """Defect: PAV split tied raw scores into blocks with different means; np.interp then got a
    repeated knot and calibrated a tied score of 0.7 to 1.0 instead of its mean 0.67."""
    p = np.array([0.3] * 4 + [0.7] * 6)
    y = np.array([0, 1, 0, 1, 1, 0, 1, 1, 0, 1], dtype=float)
    xs, ys = _isotonic(p, y)
    assert all(b > a for a, b in zip(xs, xs[1:], strict=False))
    assert np.interp(0.3, xs, ys) == pytest.approx(0.5)
    assert np.interp(0.7, xs, ys) == pytest.approx(4 / 6)


def test_clock_only_scoreboard_stream_does_not_revise_makes() -> None:
    """Defect: OCR reads without score digits (clock-only ROI, unreadable cells) counted as a
    streaming scoreboard, so every make was revised to a miss after make_confirm_wait_s."""
    s = two_team_scene()
    s.add("home-1", 18.0, 30.0).add("away-3", 4.0, 34.0)
    s.start("home-1").hold("home-1", 0.8).shoot("left", made=True).drop_through("left")
    s.fly(lambda pos: s.hands("away-3", pos), 0.4, apex=0.3).hold("away-3", 4.0)
    frames = ocr_frames(s.frames_3d(), lambda t: ScoreboardReading(t, None, None, 1, 600.0 - t, None, source="ocr",
                                                                    confidence=0.9))
    eng = run_engine(frames)
    assert one(eng.events, EventKind.SHOT_MADE).status is EventStatus.PROVISIONAL
    assert not live(eng.events, EventKind.SHOT_MISSED)


def test_flipped_last_free_throw_relabels_the_possession_free_throw() -> None:
    """Defect: a missed last FT flipped to made by the scoreboard relabelled the defensive-rebound
    POSSESSION 'made_shot' instead of 'free_throw'."""
    from test_events_rules import foul, free_throw, ft_scene, run_with_foul

    s = ft_scene().start("home-1")
    free_throw(s, made=True)
    s.rest(0.8, visible=False)
    s.run(0.05, lambda a, pos: s.hands("home-1", pos))
    free_throw(s, made=False)  # the engine sees a miss ...
    s.fly(lambda pos: s.hands("away-2", pos), 0.6, apex=2.5, moves={"away-2": (7.0, 31.0)}).hold("away-2", 0.6)
    s.set_score(s.score[0] + 1, s.score[1])  # ... but the scoreboard counts it
    s.hold("away-2", 2.0)
    eng = run_with_foul(s, "3d", foul(0.0))
    made = live(eng.events, EventKind.SHOT_MADE)
    assert len(made) == 2 and made[1].attrs.get("flipped_by") == "scoreboard"
    assert not live(eng.events, EventKind.REBOUND)
    poss = [e for e in live(eng.events, EventKind.POSSESSION) if e.team is Team.AWAY]
    assert len(poss) == 1 and poss[0].attrs["cause"] == "free_throw"


def test_ocr_period_change_is_timestamped_at_the_first_read() -> None:
    """PERIOD_START / PERIOD_END used the confirming (second) OCR read instead of the first."""
    s = two_team_scene(scoreboard=False).start("away-1").hold("away-1", 2.0)
    eng = run_engine(ocr_frames(s.frames_3d(), lambda t: ocr(t, 720.0 if t >= 1.0 else 300.0 - t,
                                                             period=2 if t >= 1.0 else 1)))
    (end,) = live(eng.events, EventKind.PERIOD_END)
    starts = live(eng.events, EventKind.PERIOD_START)
    assert end.value == 1 and end.t < 1.0 and [e.value for e in starts] == [2] and starts[0].t == pytest.approx(1.0)


def test_foul_on_a_loose_ball_after_a_miss_gives_a_team_rebound() -> None:
    """Missing semantics: a foul after a missed FGA and before control (here leading to free
    throws) must give a team rebound to the team entitled to the ball [15]; none was emitted."""
    from test_events_rules import free_throw, ft_scene

    s = ft_scene()
    s.add("home-4", 30.0, 20.0)
    s.start("home-4").hold("home-4", 0.8).shoot("left", made=False)
    s.fly((14.0, 28.0, BALL_R), 0.6, apex=2.0)  # loose on the floor, nobody near
    s.mark("foul")
    s.rest(0.6, visible=False)
    s.run(0.05, lambda a, pos: s.hands("home-1", pos))
    free_throw(s, made=True)
    s.rest(0.6, visible=False)
    s.run(0.05, lambda a, pos: s.hands("home-1", pos))
    free_throw(s, made=True)
    eng = EventEngine(court_for("nba"))
    for fr in s.frames_3d():
        eng.process(fr)
        if fr.t == s.marks["foul"]:
            eng.add_manual(Event("", EventKind.FOUL, fr.t, fr.frame_idx, team=Team.AWAY, player_id="away-1",
                                 other_player_id="home-1", attrs={"type": "personal", "ft_awarded": 2}))
    eng.flush()
    reb = one(eng.events, EventKind.REBOUND)
    assert reb.player_id is None and reb.team is Team.HOME and reb.attrs["offensive"] is True
    assert reb.attrs["team_rebound"] is True and reb.t == pytest.approx(s.marks["foul"])
    assert reb.shot_id == live(eng.events, EventKind.SHOT_ATTEMPT)[0].shot_id
    assert [a.value for a in live(eng.events, EventKind.SHOT_ATTEMPT)] == [3, 1, 1]


def test_player_history_memory_is_bounded_under_track_id_churn() -> None:
    """Defect: per-id position histories were never dropped; provisional track ids churn all game."""
    from courtvision.types import PlayerState

    eng = EventEngine(court_for("nba"))
    for k in range(6000):
        t = k / 30.0
        players = [PlayerState(f"home-t{k // 15 * 5 + i}", Team.HOME, 30.0 + i, 20.0) for i in range(5)]
        eng.process(FrameState(k, t, players, BallState(False)))
    assert len(eng._hist._h) < 5 * 2 * 60 * 30 // 15 + 50  # ids of the last ~minute only
    assert len(eng._holder._centres) < 200


def test_offensive_foul_on_a_rebound_is_a_loose_ball_foul_not_a_turnover() -> None:
    s = two_team_scene()
    s.add("home-1", 20.0, 25.0).add("away-3", 6.0, -2.0)
    s.start("home-1").hold("home-1", 0.8).shoot("left", made=False)
    s.fly((9.0, 20.0, BALL_R), 0.6, apex=2.0)  # loose, nobody has it
    s.mark("foul")
    s.rest(1.0, visible=False)
    s.run(0.05, lambda a, pos: s.hands("away-3", pos)).hold("away-3", 1.0)
    s.pass_to("away-1").hold("away-1", 0.6)
    eng = EventEngine(court_for("nba"))
    for fr in s.frames_3d():
        eng.process(fr)
        if fr.t == s.marks["foul"]:
            eng.add_manual(Event("", EventKind.FOUL, fr.t, fr.frame_idx, team=Team.HOME, player_id="home-2",
                                 other_player_id="away-2", attrs={"type": "offensive"}))
    eng.flush()
    assert not live(eng.events, EventKind.TURNOVER)
    reb = one(eng.events, EventKind.REBOUND)
    assert reb.player_id is None and reb.team is Team.AWAY and reb.attrs["offensive"] is False
