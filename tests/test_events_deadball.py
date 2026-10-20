"""DeadBallDetector rules 1-4 (RESEARCH §7A.5) and expected windows from RuleSpec."""

from __future__ import annotations

import pytest

from courtvision.events.deadball import DeadBallDetector, ft_formation
from courtvision.geometry.court import court_for
from courtvision.geometry.rules import rules_for
from courtvision.types import BallState, EventKind, FrameState, PlayerState, ScoreboardReading, Team

from test_events_kit import live, run_engine, two_team_scene


def frame(t: float, players: list[PlayerState], clock: float | None = None, ball: bool = True, period: int = 1) -> FrameState:
    sb = ScoreboardReading(t, 0, 0, period, clock, source="ocr") if clock is not None else None
    b = BallState(True, (600.0, 300.0), (40.0, 25.0), None, 4.0, 0.9) if ball else BallState(False)
    return FrameState(int(round(t * 30)), t, players, b, scoreboard=sb)


def moving(speed: float) -> list[PlayerState]:
    return [PlayerState(f"home-{i}", Team.HOME, 30.0 + 3 * i, 20.0, vx=speed) for i in range(5)] + \
        [PlayerState(f"away-{i}", Team.AWAY, 30.0 + 3 * i, 30.0, vx=speed) for i in range(5)]


def ft_players() -> list[PlayerState]:
    lane = [("away-1", 8.0, 34.0), ("home-2", 11.0, 34.0), ("away-2", 8.0, 16.0), ("home-3", 11.0, 16.0)]
    return [PlayerState("home-1", Team.HOME, 20.0, 25.0)] + [PlayerState(p, Team(p[:4]), x, y) for p, x, y in lane] + \
        [PlayerState("away-3", Team.AWAY, 25.0, 30.0, vx=1.0)]


def test_rule1_clock_stopped_while_it_should_run() -> None:
    det = DeadBallDetector(court_for("nba"), rules_for("nba"))
    for i in range(10):  # 5 Hz reads, clock running
        det.update(frame(0.2 * i, moving(8.0), clock=600.0 - 0.2 * i))
    assert not det.is_dead()
    for i in range(10, 20):  # clock frozen at 598.0
        st = det.update(frame(0.2 * i, moving(8.0), clock=598.0))
    assert st.dead and "clock" in st.reasons and st.since_t == pytest.approx(2.0)
    det.update(frame(4.2, moving(8.0), clock=597.8))  # clock runs again
    assert not det.is_dead() and det.is_dead(3.0) and not det.is_dead(1.0)


def test_rule2_free_throw_formation_hold() -> None:
    court = court_for("nba")
    det = DeadBallDetector(court, rules_for("nba"))
    f = ft_formation(court, ft_players(), det.config)
    assert f is not None and f.side == "left" and f.shooter_id == "home-1" and len(f.lane_ids) == 4
    det.update(frame(0.0, ft_players()))
    det.update(frame(1.0, ft_players()))
    assert not det.is_dead() and det.formation is not None
    st = det.update(frame(1.6, ft_players()))
    assert st.dead and st.stoppage == "free_throw"
    assert det.expected_window_s(1.6) == pytest.approx(det.config.window_free_throw_s - 1.6)
    assert ft_formation(court, moving(0.0), det.config) is None


def test_rule3_global_stillness_with_ball_not_held() -> None:
    det = DeadBallDetector(court_for("nba"))
    for i in range(12):
        st = det.update(frame(0.2 * i, moving(0.5), ball=False))
    assert st.dead and "stillness" in st.reasons
    for i in range(12, 20):
        det.update(frame(0.2 * i, moving(9.0)))
    assert not det.is_dead()
    held = DeadBallDetector(court_for("nba"))
    for i in range(12):
        st = held.update(frame(0.2 * i, moving(0.5)), holder_id="home-1")  # a player holds it: not dead
    assert not st.dead


def test_rule4_hotkey_and_note_live() -> None:
    det = DeadBallDetector(court_for("nba"), rules_for("nba"))
    det.mark_dead_ball(5.0, "timeout")
    st = det.update(frame(5.5, moving(6.0)))
    assert st.dead and st.stoppage == "timeout"
    assert det.expected_window_s(5.5) == pytest.approx(75.0 - 0.5)  # NBA timeout 1:15 (RuleSpec)
    det.note_live(9.0)
    assert not det.is_dead() and det.expected_window_s(9.0) == 0.0
    with pytest.raises(ValueError):
        det.mark_dead_ball(1.0, "nap")


@pytest.mark.parametrize(("league", "ended", "expected"), [
    ("nba", 1, 150.0), ("nba", 2, 900.0), ("nba", 4, 150.0), ("nfhs", 2, 600.0), ("ncaa_m", 1, None),
])
def test_period_break_windows_from_rules(league: str, ended: int, expected: float | None) -> None:
    rules = rules_for(league)
    det = DeadBallDetector(court_for(league), rules)
    det.update(frame(0.0, moving(5.0), clock=1.0, period=ended))
    det.update(frame(1.0, moving(5.0), clock=0.0, period=ended))
    assert det.state.stoppage is None  # one read of 0:00 may be a misread
    det.update(frame(1.2, moving(5.0), clock=0.0, period=ended))
    assert det.state.stoppage == "period_break"
    want = expected if expected is not None else float(rules.intervals["halftime_s"])
    assert det.expected_total_s("period_break") == pytest.approx(want)


def test_engine_feeds_made_basket_and_timeout_stoppages() -> None:
    s = two_team_scene()
    s.add("home-1", 18.0, 30.0)
    s.start("home-1").hold("home-1", 0.5).shoot("left").drop_through("left").set_score(2, 0).rest(0.5)
    eng = run_engine(s.frames_3d())
    assert eng.deadball.is_dead() and eng.deadball.stoppage() == "made_basket"
    assert live(eng.events, EventKind.SHOT_MADE)
