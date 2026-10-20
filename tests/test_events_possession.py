"""Holder hysteresis, passes, steals, dribbles, ball gaps and false detections (both modes)."""

from __future__ import annotations

import pytest

from courtvision.events.engine import EventEngine, EventEngineConfig
from courtvision.events.possession import BallGate, BallObs
from courtvision.geometry.court import court_for
from courtvision.types import EventKind, Team

from test_events_kit import MODES, kinds, live, one, run_engine, two_team_scene

TOL = 0.35  # s; event-time tolerance for hand-built scenes (eval uses 1.0 s)


@pytest.mark.parametrize("mode", MODES)
def test_holder_hysteresis_no_flicker(mode: str) -> None:
    s = two_team_scene().start("home-1")
    s.hold("home-1", 1.0)
    # the ball brushes past the nearby defender's position for 1 frame, then comes back
    s.run(1 / 30, lambda a, pos: (pos["away-1"][0] + 0.3, pos["away-1"][1] - 0.7, 4.0))
    s.hold("home-1", 1.0)
    eng = run_engine(s.frames(mode))
    assert kinds(eng.events) == ["possession"]
    assert eng.holder_id == "home-1" and eng.possession_team is Team.HOME


@pytest.mark.parametrize("mode", MODES)
def test_completed_pass(mode: str) -> None:
    s = two_team_scene().start("home-1").hold("home-1", 0.6)
    s.mark("release")
    s.pass_to("home-2").hold("home-2", 0.8)
    eng = run_engine(s.frames(mode))
    p = one(eng.events, EventKind.PASS)
    assert (p.player_id, p.other_player_id, p.team) == ("home-1", "home-2", Team.HOME)
    assert p.attrs["completed"] is True and abs(p.t - s.marks["release"]) < TOL
    assert kinds(eng.events) == ["possession", "pass"]
    assert eng.holder_id == "home-2"


@pytest.mark.parametrize("mode", MODES)
def test_intercepted_pass_is_steal_and_bad_pass_turnover(mode: str) -> None:
    s = two_team_scene().start("home-1").hold("home-1", 0.6)
    s.mark("release")
    s.pass_to("away-2", duration=0.5)  # thrown toward home-2's side, picked off by away-2
    s.mark("caught")
    s.hold("away-2", 0.8)
    eng = run_engine(s.frames(mode))
    ev = eng.events
    st, to = one(ev, EventKind.STEAL), one(ev, EventKind.TURNOVER)
    assert (st.player_id, st.other_player_id, st.team) == ("away-2", "home-1", Team.AWAY)
    assert (to.player_id, to.team, to.attrs["type"]) == ("home-1", Team.HOME, "bad_pass")
    p = one(ev, EventKind.PASS)
    assert p.attrs["completed"] is False and p.player_id == "home-1"
    poss = [e for e in live(ev, EventKind.POSSESSION) if e.team is Team.AWAY]
    assert len(poss) == 1 and poss[0].attrs["cause"] == "steal"
    assert abs(st.t - s.marks["caught"]) < TOL
    assert eng.possession_team is Team.AWAY


@pytest.mark.parametrize("mode", MODES)
def test_strip_steal(mode: str) -> None:
    s = two_team_scene()
    s.add("away-1", 29.0, 23.5)  # right next to the dribbler
    s.start("home-1").dribble("home-1", 1.2, 2)
    s.mark("strip")
    s.fly(lambda pos: s.hands("away-1", pos), 0.2, apex=0.2)
    s.hold("away-1", 0.8)
    eng = run_engine(s.frames(mode))
    to = one(eng.events, EventKind.TURNOVER)
    assert to.attrs["type"] == "steal" and to.player_id == "home-1"
    st = one(eng.events, EventKind.STEAL)
    assert st.player_id == "away-1" and abs(st.t - s.marks["strip"]) < 0.5
    assert not live(eng.events, EventKind.PASS)


@pytest.mark.parametrize("mode", MODES)
def test_dribble_counting(mode: str) -> None:
    s = two_team_scene().start("home-1").hold("home-1", 0.3)
    s.dribble("home-1", 2.5, 5, move=(34.0, 22.0)).hold("home-1", 0.5)
    eng = run_engine(s.frames(mode))
    dr = live(eng.events, EventKind.DRIBBLE)
    assert len(dr) == 5
    for e, t in zip(dr, s.dribble_times, strict=True):
        assert e.player_id == "home-1" and abs(e.t - t) <= 0.07
    assert kinds(eng.events) == ["possession"]


@pytest.mark.parametrize("mode", MODES)
def test_holder_kept_through_ball_gaps(mode: str) -> None:
    s = two_team_scene().start("home-1").hold("home-1", 0.5)
    for _ in range(4):
        s.hold("home-1", 0.4, visible=False).hold("home-1", 0.2)
    eng = run_engine(s.frames(mode))
    assert kinds(eng.events) == ["possession"]
    assert eng.holder_id == "home-1"


def test_false_ball_detections_are_rejected() -> None:
    s = two_team_scene().start("home-1").hold("home-1", 1.0).pass_to("home-2").hold("home-2", 1.5)
    s.dribble("home-2", 1.0, 2).hold("home-2", 1.0)
    frames = s.frames_mono(seed=3, false_rate=0.12, dropout=0.1, px_noise=1.0)
    eng = run_engine(frames)
    assert kinds(eng.events) == ["possession", "pass"]
    assert eng.holder_id == "home-2"


def test_ball_gate_rejects_jump_then_reacquires() -> None:
    gate = BallGate(EventEngineConfig())

    def obs(t: float, x: float) -> BallObs:
        return BallObs(t, int(t * 30), None, (x, 25.0), 4.0, None, 1.0)

    assert gate.accept(obs(0.0, 30.0), None) is not None
    assert gate.accept(obs(1 / 30, 30.5), None) is not None
    assert gate.accept(obs(2 / 30, 80.0), None) is None  # 50 ft in a frame
    assert gate.accept(obs(3 / 30, 31.0), None) is not None
    # the real ball really is elsewhere now: consistent detections re-acquire it
    got = [gate.accept(obs((4 + i) / 30, 70.0 + 0.3 * i), None) for i in range(3)]
    assert got[0] is None and got[1] is None and got[2] is not None


def test_possession_at_stream_start_without_scoreboard() -> None:
    s = two_team_scene(scoreboard=False).start("away-1").hold("away-1", 0.5)
    eng = run_engine(s.frames_3d())
    p = one(eng.events, EventKind.POSSESSION)
    assert p.team is Team.AWAY and p.player_id == "away-1" and p.t < 0.1


def test_engine_contract_properties() -> None:
    eng = EventEngine(court_for("nba"))
    assert eng.holder_id is None and eng.possession_team is None and eng.events == []
    assert set(eng.attacking_side) == {Team.HOME, Team.AWAY}
