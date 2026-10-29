"""Scorebug events (nba/scorebug_events.py) and their fusion with the event engine's events (nba/fusion.py)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np

from courtvision.geometry.court import court_for
from courtvision.nba.fusion import FusionConfig, PlayDirection, attack_sides, fuse
from courtvision.nba.scorebug_events import BugEvent, scorebug_events
from courtvision.perception.ball3d import G_FTS2, Flight3D
from courtvision.types import Event, EventKind, EventStatus, ScoreboardReading, Team

HOME, AWAY = Team.HOME, Team.AWAY
NBA = court_for("nba")
HZ = 3.0  # scorebug reads per second

Field = Callable[[int], float | None]


def steps(*changes: tuple[int, float | None]) -> Field:
    """A scorebug field that shows ``v`` from read ``k`` on, for each (k, v)."""
    return lambda k: next((v for k0, v in reversed(changes) if k >= k0), None)


def bug_reads(n: int, period: int = 2, home: Field = steps(), away: Field = steps(), shot: Field = steps(),
              game: Field = steps()) -> list[ScoreboardReading]:
    return [ScoreboardReading(t=k / HZ, home_score=home(k), away_score=away(k), period=period, game_clock=game(k),
                              shot_clock=shot(k)) for k in range(n)]


def aim_track(*blocks: tuple[float, float, float], hz: int = 10) -> PlayDirection:
    """The camera aiming at court x = ``x`` over [t0, t1) for each (t0, t1, x), sampled at ``hz``."""
    t = np.concatenate([np.arange(round(t0 * hz), round(t1 * hz)) / hz for t0, t1, _ in blocks])
    x = np.concatenate([np.full(round((t1 - t0) * hz), float(x)) for t0, t1, x in blocks])
    return PlayDirection(t, x)


# 10 s blocks with the camera on the right half (even blocks) or the left half (odd blocks)
ALTERNATING = aim_track(*[(10 * i, 10 * i + 10, 80.0 if i % 2 == 0 else 15.0) for i in range(10)])


def basket(block: int, team: Team, period: int, points: int = 2) -> BugEvent:
    """A score change whose basket (2.5 s before the bug update) falls in the middle of ``block``."""
    return BugEvent(10 * block + 7.5, "score", team, points, period=period)


def ft(t: float, team: Team) -> BugEvent:
    return BugEvent(t, "score", team, 1)


def only(*passes: str) -> FusionConfig:
    """A fusion config with only the named passes switched on."""
    flags = [f.name for f in dataclasses.fields(FusionConfig) if f.type in (bool, "bool")]
    assert set(passes) <= set(flags), set(passes) - set(flags)
    return FusionConfig(**{name: name in passes for name in flags})


def shot(sid: str, t_release: float, t_result: float, team: Team | None, made: bool = False, value: int = 2,
         player: str | None = None) -> list[Event]:
    """An engine shot: its attempt and result (a free throw when ``value`` is 1)."""
    attrs = {"shot_type": "free_throw"} if value == 1 else {}
    kind = EventKind.SHOT_MADE if made else EventKind.SHOT_MISSED
    return [Event(f"{sid}a", EventKind.SHOT_ATTEMPT, t_release, 0, team=team, player_id=player, value=value,
                  shot_id=sid, attrs=dict(attrs)),
            Event(f"{sid}r", kind, t_result, 0, team=team, player_id=player, value=value, shot_id=sid,
                  attrs=dict(attrs))]


def ids(events: list[Event]) -> dict[str, Event]:
    return {e.event_id: e for e in events}


def of(events: list[Event], kind: EventKind) -> list[Event]:
    return [e for e in events if e.kind is kind]


def flight_to(p0: tuple[float, float, float], target: tuple[float, float, float], t0: float,
              flight_s: float = 1.2) -> Flight3D:
    """A ballistic flight released at ``p0`` that comes down through ``target`` ``flight_s`` later."""
    p = np.array(p0, float)
    g = np.array([0.0, 0.0, -G_FTS2])
    v0 = (np.array(target, float) - p - 0.5 * g * flight_s**2) / flight_s
    return Flight3D(t0, t0 + flight_s - 0.2, p, v0, rms_px=1.0, n_obs=30)


# --------------------------------------------------------------------------- scorebug events


def test_confirmed_score_increases_of_one_two_and_three_are_events_at_their_first_read() -> None:
    rs = bug_reads(60, home=steps((0, 10), (15, 11), (36, 14)), away=steps((0, 8), (27, 10)))
    assert scorebug_events(rs) == [BugEvent(5.0, "score", HOME, 1, period=2),
                                   BugEvent(9.0, "score", AWAY, 2, period=2),
                                   BugEvent(12.0, "score", HOME, 3, period=2)]


def test_a_one_read_score_glitch_is_no_basket() -> None:
    home = steps((0, 30), (10, 33), (11, 30), (40, 32))  # read 10 shows 33 once, then 30 again; 32 for real later
    assert scorebug_events(bug_reads(60, home=home, away=steps((0, 25)))) == [
        BugEvent(40 / HZ, "score", HOME, 2, period=2)]


def test_a_score_misread_low_twice_does_not_make_the_recovery_a_basket() -> None:
    home = steps((0, 30), (10, 20), (12, 30))  # reads 10 and 11 agree on a wrong 20
    assert scorebug_events(bug_reads(30, home=home)) == []


def test_first_basket_of_a_game_read_from_period_one_counts_from_zero() -> None:
    # ESPN draws a lone 0 as a ring the recogniser skips: the score is unreadable until the team scores
    home, away = steps((0, None), (12, 2)), steps((0, None), (30, 3))
    assert scorebug_events(bug_reads(45, period=1, home=home, away=away)) == [
        BugEvent(4.0, "score", HOME, 2, period=1), BugEvent(10.0, "score", AWAY, 3, period=1)]
    assert scorebug_events(bug_reads(45, period=2, home=home, away=away)) == []  # mid-game: the first read is the base
    high = bug_reads(45, period=1, home=steps((0, 40), (30, 42)))  # a high first score is no first basket
    assert scorebug_events(high) == [BugEvent(10.0, "score", HOME, 2, period=1)]


def test_shot_clock_reset_to_24_from_a_low_value_is_reset24() -> None:
    # the shot clock runs 8 .. 3 (a value a second) and jumps back to 24 while the game clock runs
    rs = bug_reads(30, shot=lambda k: 8 - k // 3 if k < 18 else 24 - (k - 18) // 3, game=lambda k: 400.0 - k // 3)
    assert scorebug_events(rs) == [BugEvent(6.0, "reset24", shot_clock_before=3.0, game_clock=394.0, period=2),
                                   BugEvent(7.0, "run24", shot_clock_before=24.0, game_clock=393.0, period=2)]


def test_count_from_a_fresh_24_is_run24_but_not_after_a_long_hold() -> None:
    # a rim touch (24 held 2 s, then the defence's count) and later a timeout (24 held 30 s before the inbound)
    def shot_clock(k: int) -> float:
        if k < 12:
            return 9 - k // 3
        if k < 18:
            return 24.0
        if k < 60:
            return 23 - (k - 18) // 3
        if k < 150:
            return 24.0
        return 23 - (k - 150) // 3

    ev = [(e.kind, round(e.t, 2)) for e in scorebug_events(bug_reads(160, shot=shot_clock))]
    assert ev == [("reset24", 4.0), ("run24", 6.0), ("reset24", 20.0)]


def test_a_jump_from_a_low_shot_clock_straight_to_23_is_a_reset_and_a_run() -> None:
    rs = bug_reads(30, shot=lambda k: 7 - k // 3 if k < 12 else 23 - (k - 12) // 3)  # the 24 never read twice
    assert [(e.kind, round(e.t, 2)) for e in scorebug_events(rs)] == [("reset24", 3.0), ("run24", 4.0)]


def test_shot_clock_drop_from_24_to_14_is_reset14_and_counting_down_through_14_is_not() -> None:
    def shot_clock(k: int) -> float:
        if k < 24:
            return 17 - k // 3  # 17 .. 10, passing 14 on the way
        if k < 36:
            return 24.0  # rim touch: back to 24, held
        return 14 - (k - 36) // 3  # the offence kept the ball: 14, 13, ...

    ev = scorebug_events(bug_reads(45, shot=shot_clock))
    assert [(e.kind, round(e.t, 2), e.shot_clock_before) for e in ev] == [("reset24", 8.0, 10.0),
                                                                         ("reset14", 12.0, 24.0)]


def test_game_clock_stop_and_restart_above_a_minute() -> None:
    def game(k: int) -> float:
        if k < 30:
            return 125 - k // 3  # whole seconds, about three reads each
        if k < 63:
            return 115.0  # a dead ball from 10 s; the clock restarts at 20 s and shows 115 for one more second
        return 114 - (k - 63) // 3

    ev = scorebug_events(bug_reads(75, game=game))
    assert [(e.kind, round(e.t, 2), e.game_clock) for e in ev] == [("clock_stop", 11.0, 115.0),
                                                                  ("clock_start", 21.0, 114.0)]


def test_game_clock_stop_and_restart_under_a_minute_with_tenths() -> None:
    def game(k: int) -> float:
        if k < 30:
            return round(40.0 - k / HZ, 1)  # tenths: a running clock changes on every read
        if k < 60:
            return 30.0
        return round(30.0 - (k - 59) / HZ, 1)

    ev = scorebug_events(bug_reads(75, game=game))
    assert [(e.kind, round(e.t, 2), e.game_clock) for e in ev] == [("clock_stop", 10.1, 30.0),
                                                                  ("clock_start", 20.0, 29.7)]


def test_a_misread_digit_during_a_stoppage_is_no_restart() -> None:
    def game(k: int) -> float:
        if k < 30:
            return 125 - k // 3
        if k == 45:
            return 113.0  # one misread of the held 115
        if k < 63:
            return 115.0
        return 114 - (k - 63) // 3

    ev = scorebug_events(bug_reads(75, game=game))
    assert [(e.kind, round(e.t, 2)) for e in ev] == [("clock_stop", 11.0), ("clock_start", 21.0)]


# --------------------------------------------------------------------------- play direction and attack sides


def test_play_direction_aim_and_side_use_the_nearest_sample() -> None:
    d = aim_track((0, 5, 70.0), (5, 6, 47.0), (6, 10, 20.0))
    assert d.aim(2.0) == 70.0 and d.aim(5.96) == 20.0  # the 6.0 s sample is nearer than the 5.9 s one
    assert d.aim(10.5) == 20.0 and d.aim(11.5) is None  # more than a second past the last sample
    assert d.side(2.0) == "right" and d.side(5.5) is None and d.side(8.0) == "left"
    assert d.side(2.0, margin_ft=30.0) is None  # 70 ft is not 30 ft into the right half
    assert d.side(11.5) is None


def test_play_direction_leaves_a_half_only_when_the_aim_stays_in_the_other_half() -> None:
    d = aim_track((0, 3, 70.0), (3, 4, 20.0), (4, 6, 70.0), (6, 10, 20.0))  # a 1 s look across, then the play goes
    assert d.leaves("right", 0.0, 10.0) == 6.0
    assert d.leaves("right", 0.0, 10.0, hold_s=0.5) == 3.0
    assert d.leaves("right", 0.0, 7.5) is None  # under 2 s over there before the window ends
    assert d.leaves("left", 6.0, 10.0) is None


def test_attack_sides_follow_the_majority_of_made_baskets_in_each_half() -> None:
    first = [basket(0, HOME, 1), basket(2, HOME, 1), basket(4, HOME, 1), basket(1, HOME, 1),  # 3 right, 1 left
             basket(3, AWAY, 1, points=3), basket(5, AWAY, 2)]  # left
    second = [basket(7, HOME, 3), basket(9, HOME, 4), basket(6, AWAY, 3)]  # the teams switch baskets at half time
    assert attack_sides(first + second, ALTERNATING, lag_s=2.0) == {(0, HOME): "right", (0, AWAY): "left",
                                                                   (1, HOME): "left", (1, AWAY): "right"}


def test_attack_sides_ignore_free_throws_and_leave_a_tied_half_open() -> None:
    scores = [basket(0, HOME, 1)] + [basket(b, HOME, 1, points=1) for b in (1, 3, 5)]  # three free throws on the left
    scores += [basket(6, HOME, 3), basket(7, HOME, 3)]  # one basket at each end after half time: a tie
    # one home basket on the right is enough: the away team attacks the other side
    assert attack_sides(scores, ALTERNATING, lag_s=2.0) == {(0, HOME): "right", (0, AWAY): "left"}


# --------------------------------------------------------------------------- fusion: makes


def test_score_change_turns_the_nearest_engine_result_into_that_teams_make() -> None:
    ev = shot("s1", 2.0, 3.0, HOME) + shot("s2", 4.5, 5.5, HOME)  # both misses of the home team for the engine
    out = ids(fuse(ev, [BugEvent(7.0, "score", AWAY, 3)], only("makes")))  # the ball went in 2 s before the update
    made = out["s2r"]
    assert made.kind is EventKind.SHOT_MADE and made.team is AWAY and made.value == 3
    assert made.status is EventStatus.CONFIRMED and made.attrs["fused"] == "score change" and made.revision == 1
    assert out["s2a"].team is AWAY and out["s2a"].value == 3
    assert out["s1r"].kind is EventKind.SHOT_MISSED and out["s1r"].team is HOME and out["s1r"].value == 2


def test_score_change_prefers_the_scoring_teams_engine_make_over_a_nearer_miss() -> None:
    ev = shot("s1", 1.5, 2.5, AWAY, made=True) + shot("s2", 4.0, 5.0, HOME)
    out = ids(fuse(ev, [BugEvent(7.0, "score", AWAY, 2)], only("makes")))
    assert out["s1r"].kind is EventKind.SHOT_MADE and out["s1r"].attrs["fused"] == "score change"
    assert out["s2r"].kind is EventKind.SHOT_MISSED and "fused" not in out["s2r"].attrs


def test_engine_make_without_a_score_change_becomes_a_miss() -> None:
    out = ids(fuse(shot("s1", 9.0, 10.0, HOME, made=True), [], only("makes")))
    res = out["s1r"]
    assert res.kind is EventKind.SHOT_MISSED and res.team is HOME and res.value == 2
    assert res.attrs["fused"] == "no score change" and out["s1a"].attrs["fused"] == "no score change"


def test_score_change_with_no_engine_shot_infers_an_attempt_and_a_make() -> None:
    ev = [Event("p1", EventKind.POSSESSION, 20.0, 0, team=HOME)]
    out = fuse(ev, [BugEvent(30.0, "score", HOME, 2), BugEvent(50.0, "score", AWAY, 1)], only("makes"))
    made, att = of(out, EventKind.SHOT_MADE), of(out, EventKind.SHOT_ATTEMPT)
    assert [(e.t, e.team, e.value) for e in made] == [(28.0, HOME, 2), (48.0, AWAY, 1)]
    assert [(e.t, e.team, e.value) for e in att] == [(27.0, HOME, 2), (47.0, AWAY, 1)]
    assert [m.shot_id for m in made] == [a.shot_id for a in att]
    assert all(e.source == "scoreboard" and e.attrs["inferred"] for e in made + att)
    assert all(m.status is EventStatus.CONFIRMED for m in made)
    assert made[1].attrs["shot_type"] == "free_throw" and "shot_type" not in made[0].attrs


def guess(sid: str, t_release: float, t_result: float, team: Team, value: int = 2) -> list[Event]:
    """The engine's own make for a score change it matched to no seen shot (``inferred: scoreboard``)."""
    return [dataclasses.replace(e, attrs={**e.attrs, "inferred": "scoreboard"})
            for e in shot(sid, t_release, t_result, team, made=True, value=value)]


def test_score_change_goes_to_the_seen_shot_and_the_engines_guess_at_it_is_dropped() -> None:
    # the camera saw the try (called a miss of the wrong team) 3.7 s before the bug update; the engine, matching no
    # seen shot to the +3, added its own make just before the update
    ev = shot("s1", 444.3, 445.3, AWAY, value=3) + guess("s2", 447.8, 448.2, HOME, value=3)
    out = ids(fuse(ev, [BugEvent(449.0, "score", HOME, 3)], only("makes")))
    assert out["s1r"].kind is EventKind.SHOT_MADE and out["s1r"].team is HOME and out["s1r"].value == 3
    assert out["s1a"].team is HOME and "s2r" not in out and "s2a" not in out


def test_engines_guess_left_over_by_the_makes_pass_is_dropped_not_made_a_miss() -> None:
    ev = shot("s1", 84.5, 85.3, AWAY) + guess("s2", 88.2, 88.6, AWAY)  # the guess falls after the make window
    out = ids(fuse(ev, [BugEvent(89.0, "score", AWAY, 2)], only("makes")))
    assert out["s1r"].kind is EventKind.SHOT_MADE and "s2r" not in out and "s2a" not in out
    alone = ids(fuse(guess("s3", 47.0, 47.5, HOME), [BugEvent(49.5, "score", HOME, 2)], only("makes")))
    assert alone["s3r"].kind is EventKind.SHOT_MADE and alone["s3r"].attrs["fused"] == "score change"


# --------------------------------------------------------------------------- fusion: free-throw trips


def test_free_throw_trips_of_two_or_three_makes_have_no_misses() -> None:
    bug = [ft(100.0, HOME), ft(104.0, HOME), BugEvent(110.0, "clock_start"),
           ft(200.0, AWAY), ft(204.0, AWAY), ft(208.0, AWAY)]  # a fouled three
    assert fuse([], bug, only("ft_trips")) == []


def test_free_throw_trip_with_one_make_infers_the_missed_one() -> None:
    out = fuse([], [ft(200.0, AWAY)], only("ft_trips"))
    miss = of(out, EventKind.SHOT_MISSED)
    assert [(e.t, e.team, e.value) for e in miss] == [(206.0, AWAY, 1)]
    assert miss[0].attrs["shot_type"] == "free_throw" and miss[0].attrs["inferred"]
    assert miss[0].attrs["fused"] == "free-throw trip short of makes" and miss[0].status is EventStatus.PROVISIONAL
    assert [(e.t, e.team, e.shot_id) for e in of(out, EventKind.SHOT_ATTEMPT)] == [(205.0, AWAY, miss[0].shot_id)]


def test_two_lone_free_throws_with_a_restart_between_are_two_trips() -> None:
    bug = [ft(100.0, HOME), BugEvent(120.0, "clock_start"), ft(140.0, HOME)]
    assert [(e.t, e.team) for e in of(fuse([], bug, only("ft_trips")), EventKind.SHOT_MISSED)] == [(106.0, HOME),
                                                                                                 (146.0, HOME)]


def test_lone_free_throw_after_the_teams_basket_with_no_restart_is_an_and_one() -> None:
    # the whistle stops the clock on the shot; the bug shows the basket about 2 s after it went in
    and_one = [BugEvent(48.5, "clock_stop"), BugEvent(50.0, "score", HOME, 2), ft(80.0, HOME)]
    assert fuse([], and_one, only("ft_trips")) == []
    restarted = [BugEvent(48.5, "clock_stop"), BugEvent(50.0, "score", HOME, 2), BugEvent(60.0, "clock_start"),
                 ft(80.0, HOME)]
    assert len(of(fuse([], restarted, only("ft_trips")), EventKind.SHOT_MISSED)) == 1
    other_team = [BugEvent(48.5, "clock_stop"), BugEvent(50.0, "score", AWAY, 2), ft(80.0, HOME)]
    assert len(of(fuse([], other_team, only("ft_trips")), EventKind.SHOT_MISSED)) == 1


def test_lone_free_throw_is_no_and_one_when_the_clock_ran_on_after_the_basket() -> None:
    # home scores with the clock running, gets the ball back and is fouled at 70 s: a two-shot trip, one made
    bug = [BugEvent(50.0, "score", HOME, 2), BugEvent(71.0, "clock_stop"), ft(90.0, HOME)]
    miss = of(fuse([], bug, only("ft_trips")), EventKind.SHOT_MISSED)
    assert [(e.team, e.value) for e in miss] == [(HOME, 1)]


def test_engine_free_throw_misses_beyond_the_trip_size_are_dropped() -> None:
    ev = shot("f1", 102.0, 102.5, HOME, value=1)  # the engine called one of two made free throws a miss
    ev += shot("f2", 196.0, 196.5, AWAY, value=1) + shot("f3", 201.0, 201.5, AWAY, value=1)
    bug = [ft(100.0, HOME), ft(104.0, HOME), BugEvent(150.0, "clock_start"), ft(200.0, AWAY)]
    out = ids(fuse(ev, bug, only("ft_trips")))
    assert "f1r" not in out and "f1a" not in out  # two makes fill a two-shot trip
    assert "f2r" in out and "f2a" in out and "f3r" not in out and "f3a" not in out  # one make: room for one miss
    assert not any(e.attrs.get("inferred") for e in out.values())


# --------------------------------------------------------------------------- fusion: rim touches


def test_shot_clock_reset_with_no_shot_seen_infers_a_miss_for_the_team_with_the_ball() -> None:
    ev = [Event("p1", EventKind.POSSESSION, 0.0, 0, team=HOME), Event("p2", EventKind.PASS, 20.0, 0, team=AWAY,
                                                                      player_id="a1")]
    out = fuse(ev, [BugEvent(30.0, "reset24", shot_clock_before=9.0)], only("rim_misses"))
    miss = of(out, EventKind.SHOT_MISSED)
    assert [(e.t, e.team, e.value) for e in miss] == [(30.0, AWAY, 2)]
    assert miss[0].attrs["inferred"] and miss[0].attrs["fused"] == "shot clock reset (rim touch)"
    assert miss[0].status is EventStatus.PROVISIONAL
    assert [(e.t, e.team) for e in of(out, EventKind.SHOT_ATTEMPT)] == [(29.0, AWAY)]


def test_shot_clock_reset_followed_by_a_score_change_is_the_make_not_a_miss() -> None:
    ev = [Event("p2", EventKind.PASS, 20.0, 0, team=AWAY, player_id="a1")]
    bug = [BugEvent(30.0, "reset24", shot_clock_before=9.0), BugEvent(32.0, "score", AWAY, 2)]
    assert of(fuse(ev, bug, only("rim_misses")), EventKind.SHOT_MISSED) == []


def test_shot_clock_reset_next_to_a_turnover_or_an_engine_shot_adds_no_miss() -> None:
    ev = [Event("p1", EventKind.POSSESSION, 0.0, 0, team=HOME), Event("to", EventKind.TURNOVER, 31.0, 0, team=HOME)]
    ev += shot("s1", 57.0, 58.5, HOME)  # the engine's own miss just before the reset is the rim touch
    ev += shot("s2", 90.0, 91.0, AWAY, made=True)
    bug = [BugEvent(t, "reset24", shot_clock_before=10.0) for t in (30.0, 60.0, 90.5)]
    out = fuse(ev, bug, only("rim_misses"))
    assert sorted(e.event_id for e in out) == sorted(e.event_id for e in ev)


def test_shot_clock_reset_while_the_game_clock_is_stopped_is_no_rim_touch() -> None:
    # out of bounds at 25 s: the clock stops, the shot clock is reset for the other team at 30 s, play resumes at 45 s
    ev = [Event("p2", EventKind.PASS, 20.0, 0, team=AWAY, player_id="a1")]
    bug = [BugEvent(25.0, "clock_stop", game_clock=300.0),
           BugEvent(30.0, "reset24", shot_clock_before=9.0, game_clock=300.0),
           BugEvent(45.0, "clock_start", game_clock=299.0)]
    assert of(fuse(ev, bug, only("rim_misses")), EventKind.SHOT_MISSED) == []


# --------------------------------------------------------------------------- fusion: turnovers from the shot clock


def steal(k: str, t: float) -> list[Event]:
    """An engine steal of the away team and the home turnover that goes with it."""
    return [Event(f"st{k}", EventKind.STEAL, t, 0, team=AWAY, player_id="a1"),
            Event(f"to{k}", EventKind.TURNOVER, t, 0, team=HOME, player_id="h1", attrs={"type": "steal"})]


def test_engine_steal_and_its_turnover_need_a_reset_to_24_that_no_shot_explains() -> None:
    ev = steal("1", 40.0) + steal("2", 70.0) + steal("3", 101.0) + steal("4", 131.0)
    ev += shot("s1", 98.5, 99.5, HOME)
    bug = [BugEvent(41.0, "reset24", shot_clock_before=9.0),  # the steal's own fresh 24
           # 70 s: no reset, the shot clock ran on (the same team kept the ball)
           BugEvent(100.0, "reset24", shot_clock_before=12.0),  # the rim touch of the miss at 99.5 s
           BugEvent(130.0, "reset24", shot_clock_before=6.0), BugEvent(132.0, "score", AWAY, 2)]  # a make
    out = fuse(ev, bug, only("turnover_resets"))
    assert sorted(e.event_id for e in out if e.kind in (EventKind.STEAL, EventKind.TURNOVER)) == ["st1", "to1"]
    assert "s1r" in ids(out)


def test_turnover_at_a_whistle_takes_the_reset_made_in_the_stoppage_and_with_the_shot_clock_off_stays() -> None:
    ev = [Event("to1", EventKind.TURNOVER, 24.0, 0, team=HOME, attrs={"type": "out_of_bounds"}),
          Event("to2", EventKind.TURNOVER, 80.0, 0, team=AWAY)]
    bug = [BugEvent(25.0, "clock_stop", game_clock=300.0),
           BugEvent(31.0, "reset24", shot_clock_before=9.0, game_clock=300.0),  # 7 s after it, in the stoppage
           BugEvent(45.0, "clock_start", game_clock=299.0),
           BugEvent(60.0, "run24", shot_clock_before=24.0, game_clock=22.0)]  # under 24 s left: the shot clock is off
    out = fuse(ev, bug, only("turnover_resets"))
    assert sorted(e.event_id for e in of(out, EventKind.TURNOVER)) == ["to1", "to2"]


def test_reset_to_24_at_a_stoppage_with_no_shot_before_it_is_a_dead_ball_turnover_of_the_team_with_the_ball() -> None:
    ev = [Event("d1", EventKind.DRIBBLE, 20.0, 0, team=AWAY), Event("p1", EventKind.PASS, 23.0, 0, team=AWAY,
                                                                   player_id="a1")]
    ev += shot("s1", 66.0, 67.0, HOME)  # the ball goes out after it: a team rebound, no turnover
    bug = [BugEvent(25.0, "clock_stop", game_clock=300.0),
           BugEvent(26.0, "reset24", shot_clock_before=9.0, game_clock=300.0),
           BugEvent(45.0, "clock_start", game_clock=299.0),
           BugEvent(71.0, "clock_stop", game_clock=274.0),
           BugEvent(72.0, "reset24", shot_clock_before=10.0, game_clock=274.0),
           BugEvent(90.0, "clock_start", game_clock=273.0),
           BugEvent(120.0, "clock_stop", game_clock=245.0),  # a foul: free throws, the last one in, the other team's 24
           BugEvent(140.0, "score", HOME, 1), BugEvent(141.0, "reset24", shot_clock_before=12.0, game_clock=245.0),
           BugEvent(160.0, "clock_start", game_clock=244.0)]
    to = of(fuse(ev, bug, only("deadball_turnovers")), EventKind.TURNOVER)
    assert [(e.t, e.team, e.attrs["fused"]) for e in to] == [(25.0, AWAY, "shot clock reset at a stoppage")]
    assert to[0].attrs["inferred"] and to[0].source == "scoreboard"


# --------------------------------------------------------------------------- fusion: assists


def test_assist_of_a_shot_that_ended_up_missed_is_dropped() -> None:
    ev = shot("s1", 9.0, 10.0, HOME, made=True, player="h1")
    ev += [Event("as1", EventKind.ASSIST, 8.0, 0, team=HOME, player_id="h2", other_player_id="h1", shot_id="s1")]
    assert "as1" in ids(fuse(ev, [], only("makes")))  # no score change: the make became a miss ...
    assert of(fuse(ev, [], only("makes", "assists")), EventKind.ASSIST) == []  # ... and its assist goes


def test_make_takes_the_last_teammate_pass_within_the_window_as_its_assist() -> None:
    ev = shot("s1", 9.0, 10.0, HOME, made=True, player="h1")  # released at 9 s
    ev += [Event("p1", EventKind.PASS, 5.0, 0, team=HOME, player_id="h2"),  # 4 s before the release: too early
           Event("p2", EventKind.PASS, 7.0, 0, team=HOME, player_id="h3"),
           Event("p3", EventKind.PASS, 8.0, 0, team=HOME, player_id="h4"),
           Event("p4", EventKind.PASS, 8.5, 0, team=AWAY, player_id="a1"),  # not a teammate
           Event("p5", EventKind.PASS, 8.8, 0, team=HOME, player_id="h1")]  # the shooter's own pass
    ast = of(fuse(ev, [], only("assists")), EventKind.ASSIST)
    assert [(a.t, a.team, a.player_id, a.other_player_id, a.shot_id) for a in ast] == [(8.0, HOME, "h4", "h1", "s1")]
    assert ast[0].attrs["fused"] == "last pass before a make"


# --------------------------------------------------------------------------- fusion: 3-D flight values


def test_missed_shot_takes_two_or_three_points_from_its_3d_flight_release_point() -> None:
    flights = [flight_to((30.0, 25.0, 7.5), (5.25, 26.0, 10.0), 10.0),  # 24.75 ft from the left rim: a three
               flight_to((20.0, 25.0, 7.5), (5.25, 24.0, 10.0), 30.0),  # 14.75 ft: a two
               flight_to((10.0, 47.6, 7.5), (5.25, 26.0, 10.0), 50.0),  # left corner, 0.6 ft beyond the line: a three
               flight_to((29.25, 25.0, 7.5), (5.25, 26.0, 10.0), 70.0),  # 0.25 ft beyond the arc: within the margin
               flight_to((62.75, 25.0, 7.5), (88.75, 24.0, 10.0), 90.0),  # 26 ft from the right rim: a three
               flight_to((19.0, 25.0, 7.5), (5.25, 26.0, 10.0), 130.0)]  # from the free-throw line
    ev = (shot("s1", 10.0, 11.5, HOME) + shot("s2", 30.0, 31.0, HOME, value=3) + shot("s3", 50.0, 51.0, AWAY)
          + shot("s4", 70.0, 71.0, AWAY, value=3) + shot("s5", 90.0, 91.0, HOME)
          + shot("s6", 150.0, 151.0, HOME)  # no flight near it
          + shot("f1", 130.0, 131.0, AWAY, value=1))  # a missed free throw keeps its point
    out = ids(fuse(ev, [], only("flight_values"), flights=flights, court=NBA))
    assert [out[f"s{i}r"].value for i in range(1, 7)] == [3, 2, 3, 2, 3, 2]
    assert [out[f"s{i}a"].value for i in range(1, 7)] == [3, 2, 3, 2, 3, 2]
    assert (out["s1r"].x, out["s1r"].y) == (30.0, 25.0) and (out["s1a"].x, out["s1a"].y) == (30.0, 25.0)
    assert out["s1r"].attrs["value_from"] == "3-D flight" and out["s4r"].attrs["value_from"] == "3-D flight"
    assert out["s6r"].x is None and "value_from" not in out["s6r"].attrs
    assert out["f1r"].value == 1 and out["f1r"].x is None


def test_flight_value_follows_the_flight_back_to_the_release_when_the_ball_is_first_seen_high() -> None:
    # released 1.5 ft beyond the arc of the right basket; the detector first sees the ball 0.3 s later, 6 ft on
    full = flight_to((63.5, 25.0, 8.5), (88.75, 25.0, 10.0), 10.0)
    t0 = 10.3
    seen = Flight3D(t0, full.t1, full.at(t0), full.velocity(t0), rms_px=1.0, n_obs=25)
    assert seen.p0[0] > 65.0 + 3.0 and seen.p0[2] > 12.0  # inside the arc, well above the release height
    out = ids(fuse(shot("s1", 10.0, 11.2, HOME), [], only("flight_values"), flights=[seen], court=NBA))
    assert out["s1r"].value == 3 and abs(out["s1r"].x - 63.5) < 0.1 and abs(out["s1r"].y - 25.0) < 0.1


def test_flight_value_keeps_the_makes_pass_changes_to_the_attempt() -> None:
    ev = shot("s1", 10.0, 11.5, HOME, made=True)  # the engine called it in; the scorebug shows no score change
    three = flight_to((30.0, 25.0, 7.5), (5.25, 26.0, 10.0), 10.0)
    out = ids(fuse(ev, [], only("makes", "flight_values"), flights=[three], court=NBA))
    assert out["s1r"].attrs["fused"] == "no score change" and out["s1r"].value == 3  # the result has both passes
    assert out["s1a"].value == 3 and out["s1a"].attrs.get("fused") == "no score change"


# --------------------------------------------------------------------------- fusion: camera half (teams, rebounds)

SIDES = [basket(0, HOME, 1), basket(2, HOME, 1), basket(1, AWAY, 1)]  # home attacks the right half, away the left
PERIODS = [(0.0, 1), (200.0, 1)]


def test_shot_takes_the_team_attacking_the_half_the_camera_aims_at() -> None:
    ev = shot("s1", 35.0, 36.0, HOME) + shot("s2", 42.0, 43.0, HOME)  # released in block 3 (left), block 4 (right)
    out = ids(fuse(ev, SIDES, only("shot_teams"), direction=ALTERNATING, periods=PERIODS))
    assert out["s1a"].team is AWAY and out["s1r"].team is AWAY and out["s1r"].attrs["team_from"] == "camera half"
    assert out["s2a"].team is HOME and out["s2r"].team is HOME and "team_from" not in out["s2r"].attrs


def test_scorebug_make_keeps_the_bugs_team_whatever_half_the_camera_is_on() -> None:
    bug = SIDES + [BugEvent(37.5, "score", HOME, 2, period=1)]  # a home basket with the camera on away's half
    ev = shot("s1", 34.5, 35.5, AWAY)
    out = ids(fuse(ev, bug, only("makes", "shot_teams"), direction=ALTERNATING, periods=PERIODS))
    assert out["s1r"].kind is EventKind.SHOT_MADE and out["s1r"].team is HOME and out["s1a"].team is HOME


# home misses at 50 s with the camera on the right; the camera crosses to the left half at 53 s and stays
DREB = aim_track((0, 10, 80.0), (10, 20, 15.0), (20, 53, 80.0), (53, 90, 15.0))


def test_camera_leaving_the_half_after_a_miss_makes_a_defensive_rebound_of_the_other_team() -> None:
    ev = shot("s1", 49.0, 50.0, HOME) + [Event("rb", EventKind.REBOUND, 51.5, 0, team=HOME, player_id="h2",
                                               attrs={"offensive": True})]
    rb = ids(fuse(ev, SIDES, only("rebounds"), direction=DREB, periods=PERIODS))["rb"]
    assert rb.team is AWAY and rb.attrs["offensive"] is False and rb.attrs["fused"] == "play direction"


def test_another_attempt_from_the_same_half_before_the_camera_leaves_is_an_offensive_rebound() -> None:
    d = aim_track((0, 10, 80.0), (10, 20, 15.0), (20, 60, 80.0), (60, 90, 15.0))  # the play stays right until 60 s
    ev = shot("s1", 49.0, 50.0, HOME) + [Event("rb", EventKind.REBOUND, 51.2, 0, team=AWAY, player_id="a2",
                                               attrs={"offensive": False})]
    ev += shot("s2", 54.0, 55.0, HOME, made=True)  # the put-back, with the camera deep in the same half
    rb = ids(fuse(ev, SIDES, only("rebounds"), direction=d, periods=PERIODS))["rb"]
    assert rb.team is HOME and rb.attrs["offensive"] is True and rb.attrs["fused"] == "play direction"


def test_miss_with_no_rebound_seen_gets_an_inferred_one() -> None:
    out = fuse(shot("s1", 49.0, 50.0, HOME), SIDES, only("rebounds"), direction=DREB, periods=PERIODS)
    reb = of(out, EventKind.REBOUND)
    assert [(r.t, r.team, r.player_id, r.attrs["offensive"]) for r in reb] == [(51.5, AWAY, "away-unknown", False)]
    assert reb[0].attrs["inferred"] and reb[0].attrs["fused"] == "play direction"


# --------------------------------------------------------------------------- fusion: rebounds from the shot clock


def test_reset_to_14_after_a_miss_is_an_offensive_rebound_of_the_shooting_team() -> None:
    ev = shot("s1", 49.0, 50.0, HOME) + [Event("rb", EventKind.REBOUND, 51.0, 0, team=AWAY, player_id="a2",
                                               attrs={"offensive": False})]
    bug = [BugEvent(50.2, "reset24", shot_clock_before=9.0), BugEvent(52.3, "reset14", shot_clock_before=24.0)]
    rb = ids(fuse(ev, bug, only("bug_rebounds")))["rb"]
    assert rb.team is HOME and rb.attrs["offensive"] is True and rb.attrs["fused"] == "shot clock"


def test_count_from_a_fresh_24_after_a_miss_is_a_defensive_rebound_even_against_the_camera() -> None:
    # the camera stays on the right half (where home attacks) long after the miss: the camera pass alone would
    # call an offensive rebound; the shot clock's count from 24 says the defence got the ball
    d = aim_track((0, 10, 80.0), (10, 20, 15.0), (20, 90, 80.0))
    ev = shot("s1", 49.0, 50.0, HOME) + shot("s2", 54.0, 55.0, HOME)
    bug = SIDES + [BugEvent(50.2, "reset24", shot_clock_before=9.0), BugEvent(52.4, "run24", shot_clock_before=24.0)]
    out = fuse(ev, bug, only("bug_rebounds", "rebounds"), direction=d, periods=PERIODS)
    reb = sorted(of(out, EventKind.REBOUND), key=lambda e: e.t)
    first = reb[0]
    assert (first.team, first.attrs["offensive"], first.attrs["fused"]) == (AWAY, False, "shot clock")
    assert first.attrs["inferred"] and 50.5 <= first.t <= 51.5


def test_a_put_back_basket_is_an_offensive_rebound_and_a_clock_stop_leaves_the_miss_open() -> None:
    ev = shot("s1", 49.0, 50.0, HOME) + shot("s2", 79.0, 80.0, AWAY)
    bug = [BugEvent(53.0, "score", HOME, 2), BugEvent(81.0, "clock_stop", game_clock=300.0),
           BugEvent(83.0, "run24", shot_clock_before=24.0)]
    reb = of(fuse(ev, bug, only("bug_rebounds")), EventKind.REBOUND)
    assert [(r.team, r.attrs["offensive"]) for r in reb] == [(HOME, True)]  # s2 (foul on the rebound): no call


def test_each_miss_takes_only_the_signs_before_the_next_miss() -> None:
    # a miss, the offensive rebound (14), a missed put-back, then the defence's count from 24
    ev = shot("s1", 49.0, 50.0, HOME) + shot("s2", 52.0, 53.0, HOME)
    bug = [BugEvent(51.8, "reset14", shot_clock_before=24.0), BugEvent(55.5, "run24", shot_clock_before=24.0)]
    reb = sorted(of(fuse(ev, bug, only("bug_rebounds")), EventKind.REBOUND), key=lambda e: e.t)
    assert [(r.team, r.attrs["offensive"]) for r in reb] == [(HOME, True), (AWAY, False)]


def test_clock_stop_before_any_possession_sign_after_a_miss_is_a_team_rebound() -> None:
    # the game clock stops (the ball out of bounds) before the shot clock shows a count from 24 or a 14: nobody
    # had the ball, so the engine's rebound goes and the camera pass infers none
    ev = shot("s1", 49.0, 50.0, HOME) + [Event("rb", EventKind.REBOUND, 51.0, 0, team=HOME, player_id="h2",
                                               attrs={"offensive": True})]
    bug = SIDES + [BugEvent(50.2, "reset24", shot_clock_before=9.0), BugEvent(52.0, "clock_stop", game_clock=300.0),
                   BugEvent(60.0, "clock_start", game_clock=300.0), BugEvent(62.0, "run24", shot_clock_before=24.0)]
    out = fuse(ev, bug, only("bug_rebounds", "team_rebounds", "rebounds"), direction=DREB, periods=PERIODS)
    assert of(out, EventKind.REBOUND) == []
    # without the rule the miss is left to the camera, which calls the defence's rebound
    out = fuse(ev, bug, only("bug_rebounds", "rebounds"), direction=DREB, periods=PERIODS)
    assert [(r.event_id, r.team, r.attrs["offensive"]) for r in of(out, EventKind.REBOUND)] == [("rb", AWAY, False)]


def test_engine_rebound_after_a_make_or_with_no_miss_just_before_it_is_dropped() -> None:
    def reb(eid: str, t: float, team: Team) -> Event:
        return Event(eid, EventKind.REBOUND, t, 0, team=team, player_id=f"{team.value}-1", attrs={"offensive": False})

    ev = [reb("rb0", 20.0, HOME)]  # no shot before it
    ev += shot("s1", 49.0, 50.0, HOME, made=True) + [reb("rb1", 51.0, AWAY)]  # after a make: the inbound
    ev += shot("s2", 69.0, 70.0, AWAY) + [reb("rb2", 71.0, HOME), reb("rb3", 78.0, HOME)]  # rb3: 8 s after the miss
    out = fuse(ev, [], only("orphan_rebounds"))
    assert [e.event_id for e in of(out, EventKind.REBOUND)] == ["rb2"]


def test_reset_to_14_with_no_shot_seen_infers_a_miss_and_its_offensive_rebound() -> None:
    ev = [Event("p1", EventKind.POSSESSION, 40.0, 0, team=AWAY), Event("p2", EventKind.DRIBBLE, 45.0, 0, team=AWAY)]
    bug = [BugEvent(48.0, "reset14", shot_clock_before=6.0)]
    out = fuse(ev, bug, only("oreb_misses", "bug_rebounds"))
    miss = of(out, EventKind.SHOT_MISSED)
    assert [(e.t, e.team, e.attrs["fused"]) for e in miss] == [(46.5, AWAY, "shot clock reset to 14 (offensive rebound)")]
    reb = of(out, EventKind.REBOUND)
    assert [(r.team, r.attrs["offensive"]) for r in reb] == [(AWAY, True)]


def test_reset_to_14_at_a_whistle_or_right_after_a_known_miss_infers_nothing() -> None:
    ev = [Event("p1", EventKind.POSSESSION, 40.0, 0, team=AWAY)] + shot("s1", 69.0, 70.0, AWAY)
    bug = [BugEvent(47.5, "clock_stop", game_clock=300.0), BugEvent(48.0, "reset14", shot_clock_before=6.0,
                                                                     game_clock=300.0),
           BugEvent(60.0, "clock_start", game_clock=299.0), BugEvent(72.0, "reset14", shot_clock_before=24.0)]
    out = fuse(ev, bug, only("oreb_misses"))
    assert sorted(e.event_id for e in out) == sorted(e.event_id for e in ev)


def test_inferred_miss_takes_its_value_from_the_last_dribble_but_not_after_a_pass() -> None:
    cfg = only("rim_misses", "handler_values")
    ev = [Event("d1", EventKind.DRIBBLE, 27.5, 0, team=AWAY, player_id="a1", x=70.0, y=2.0)]  # right corner, 3
    out = fuse(ev, [BugEvent(30.0, "reset24", shot_clock_before=9.0)], cfg, court=NBA)
    miss = of(out, EventKind.SHOT_MISSED)[0]
    assert (miss.value, miss.attrs["value_from"]) == (3, "ball handler")
    assert of(out, EventKind.SHOT_ATTEMPT)[0].value == 3
    ev2 = ev + [Event("p1", EventKind.PASS, 28.3, 0, team=AWAY, player_id="a1", other_player_id="a2", x=70.0, y=2.0)]
    miss2 = of(fuse(ev2, [BugEvent(30.0, "reset24", shot_clock_before=9.0)], cfg, court=NBA), EventKind.SHOT_MISSED)[0]
    assert miss2.value == 2 and "value_from" not in miss2.attrs
    near = [Event("d1", EventKind.DRIBBLE, 27.5, 0, team=AWAY, player_id="a1", x=80.0, y=25.0)]  # in the lane: 2
    miss3 = of(fuse(near, [BugEvent(30.0, "reset24", shot_clock_before=9.0)], cfg, court=NBA), EventKind.SHOT_MISSED)[0]
    assert miss3.value == 2 and miss3.attrs["value_from"] == "ball handler"
