"""add_manual (operator inserts / corrections), flush, and the per-frame performance budget."""

from __future__ import annotations

import time

from courtvision.events.engine import EventEngine
from courtvision.geometry.court import court_for
from courtvision.types import Event, EventKind, EventStatus, FrameState, Team

from test_events_kit import live, one, run_engine, two_team_scene


def test_manual_insert_is_confirmed_with_m_id() -> None:
    eng = EventEngine(court_for("nba"))
    out = eng.add_manual(Event("", EventKind.SUBSTITUTION, 12.0, 360, team=Team.HOME, player_id="home-6",
                               other_player_id="home-1"))
    assert len(out) == 1
    ev = out[0]
    assert ev.event_id == "m000001" and ev.source == "manual" and ev.status is EventStatus.CONFIRMED
    assert eng.events == [ev]


def test_manual_correction_of_engine_event_is_a_revision() -> None:
    s = two_team_scene(scoreboard=False)
    s.add("home-1", 18.0, 30.0).add("away-3", 9.0, 25.0)
    s.start("home-1").hold("home-1", 0.6).shoot("left", made=False)
    s.fly(lambda pos: s.hands("away-3", pos), 0.6, apex=3.0).hold("away-3", 0.8)
    eng = run_engine(s.frames_3d())
    miss = one(eng.events, EventKind.SHOT_MISSED)
    fixed = Event(miss.event_id, EventKind.SHOT_MADE, miss.t, miss.frame_idx, team=miss.team, player_id=miss.player_id,
                  value=2, shot_id=miss.shot_id)
    out = eng.add_manual(fixed)
    assert out[-1].event_id == miss.event_id and out[-1].revision == miss.revision + 1
    latest = one(eng.events, EventKind.SHOT_MADE)
    assert latest.source == "manual" and latest.status is EventStatus.CONFIRMED
    assert not live(eng.events, EventKind.SHOT_MISSED)
    assert eng._shots[miss.shot_id].made is True and eng._shots[miss.shot_id].confirmed  # reconciler sees it


def test_manual_retraction_and_timeout_stoppage() -> None:
    eng = EventEngine(court_for("nba"))
    ev = eng.add_manual(Event("", EventKind.TIMEOUT, 30.0, 900, team=Team.AWAY))[0]
    assert eng.deadball.stoppage() == "timeout" and eng.deadball.expected_window_s(30.0) == 75.0
    gone = eng.add_manual(Event(ev.event_id, EventKind.TIMEOUT, 30.0, 900, team=Team.AWAY, status=EventStatus.RETRACTED))
    assert gone[0].status is EventStatus.RETRACTED and gone[0].revision == 1


def test_flush_resolves_shot_in_flight() -> None:
    s = two_team_scene(scoreboard=False)
    s.add("home-1", 18.0, 30.0)
    s.start("home-1").hold("home-1", 0.6).shoot("left", made=True, duration=1.0)
    eng = EventEngine(court_for("nba"))
    for fr in s.frames_3d()[:-3]:  # stream ends with the ball in the rim zone
        eng.process(fr)
    out = eng.flush()
    miss = [e for e in out if e.kind is EventKind.SHOT_MISSED]
    assert len(miss) == 1 and miss[0].attrs["needs_review"] is True


def _retimed(frames: list[FrameState], n: int) -> list[FrameState]:
    """Repeat ``frames`` with continuous frame indices / times until ``n`` frames."""
    return [FrameState(k, k / 30.0, fr.players, fr.ball, fr.rims, None, fr.image_size)
            for k, fr in ((k, frames[k % len(frames)]) for k in range(n))]


def test_performance_18000_frames_under_budget() -> None:
    s = two_team_scene(scoreboard=False)
    s.add("home-1", 30.0, 25.0).add("away-3", 9.0, 25.0)
    s.start("home-1").dribble("home-1", 2.0, 4, move=(22.0, 28.0)).pass_to("home-2").hold("home-2", 0.5)
    s.pass_to("home-1").hold("home-1", 0.5).shoot("left", made=False)
    s.fly(lambda pos: s.hands("away-3", pos), 0.6, apex=3.0).hold("away-3", 0.5)
    s.pass_to("away-1").dribble("away-1", 2.0, 4, move=(30.0, 25.0)).hold("away-1", 0.5)
    s.pass_to("home-1", duration=0.6).hold("home-1", 1.0)
    loop = s.frames_3d()
    frames = _retimed(loop, 18000)
    eng = EventEngine(court_for("nba"))
    t0 = time.perf_counter()
    for fr in frames:
        eng.process(fr)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"{elapsed:.1f} s for 18,000 frames"
    assert len(live(eng.events, EventKind.SHOT_ATTEMPT)) > 50
