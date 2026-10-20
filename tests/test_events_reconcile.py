"""ScoreboardConfirmer (D14) and scoreboard reconciliation of shot results."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from courtvision.events.scoreboard import PeriodChange, ScoreboardConfirmer, ScoreChange
from courtvision.types import EventKind, EventStatus, FrameState, ScoreboardReading, Team

from test_events_kit import MODES, Scene, live, one, run_engine, two_team_scene


def ocr(t: float, home: int, away: int, clock: float | None = None, period: int = 1, conf: float = 0.95) -> ScoreboardReading:
    return ScoreboardReading(t, home, away, period, clock, None, source="ocr", confidence=conf)


# ---------------------------------------------------------------------------- confirmer


def test_ocr_change_needs_two_identical_consecutive_reads() -> None:
    sc = ScoreboardConfirmer()
    assert sc.update(ocr(0.0, 10, 8, 300.0)) == []  # baseline (and period) need a second read
    out = sc.update(ocr(0.2, 10, 8, 299.8))
    assert out == [PeriodChange(0.0, None, 1, 299.8, "ocr")]
    assert sc.score == {Team.HOME: 10, Team.AWAY: 8}
    assert [u for u in sc.update(ocr(0.4, 12, 8, 299.6)) if isinstance(u, ScoreChange)] == []
    assert sc.update(ocr(0.6, 10, 8, 299.4)) == []  # glitch read breaks the run
    assert sc.update(ocr(0.8, 12, 8, 299.2)) == []
    out = sc.update(ocr(1.0, 12, 8, 299.0))
    assert len(out) == 1 and isinstance(out[0], ScoreChange)
    ch = out[0]
    assert (ch.team, ch.delta, ch.home, ch.away, ch.t, ch.confirmed_t) == (Team.HOME, 2, 12, 8, 0.8, 1.0)
    assert not ch.needs_review


def test_inconsistent_clock_blocks_confirmation() -> None:
    sc = ScoreboardConfirmer()
    sc.update(ocr(0.0, 0, 0, 300.0))
    sc.update(ocr(0.2, 0, 0, 299.8))
    assert sc.update(ocr(0.4, 0, 3, 299.6)) == []
    assert sc.update(ocr(0.6, 0, 3, 250.0)) == []  # clock fell 49.6 s in 0.2 s: misread frame
    out = sc.update(ocr(0.8, 0, 3, 299.2))
    assert out == []  # 250 -> 299.2 rises: still inconsistent
    out = sc.update(ocr(1.0, 0, 3, 299.0))
    assert [(c.team, c.delta) for c in out] == [(Team.AWAY, 3)]


def test_jump_and_negative_deltas_need_more_reads_and_review() -> None:
    sc = ScoreboardConfirmer()
    for i in range(2):
        sc.update(ocr(0.2 * i, 20, 20))
    got = []
    for i in range(2, 7):
        got += sc.update(ocr(0.2 * i, 25, 20))
    assert len(got) == 1 and got[0].delta == 5 and got[0].needs_review
    got = []
    for i in range(7, 12):
        got += sc.update(ocr(0.2 * i, 23, 20))
    assert len(got) == 1 and got[0].delta == -2 and got[0].needs_review


def test_trusted_sources_confirm_immediately_and_streaming() -> None:
    sc = ScoreboardConfirmer()
    sc.update(ScoreboardReading(0.0, 0, 0, 1, 720.0, source="manual"))
    out = sc.update(ScoreboardReading(1.0, 2, 0, 1, 719.0, source="manual"))
    assert [(c.team, c.delta) for c in out if isinstance(c, ScoreChange)] == [(Team.HOME, 2)]
    assert not sc.is_streaming(1.0)  # two operator entries are not a stream
    feed = ScoreboardConfirmer()
    for i in range(10):
        feed.update(ScoreboardReading(i / 5, 0, 0, 1, 720.0 - i / 5, source="feed"))
    assert feed.is_streaming(1.9) and not feed.is_streaming(5.0)


def test_period_change_and_clock_stopped() -> None:
    sc = ScoreboardConfirmer()
    out = sc.update(ScoreboardReading(0.0, 0, 0, 1, 1.0, source="sim"))
    assert [(u.old, u.new) for u in out if isinstance(u, PeriodChange)] == [(None, 1)]
    out = sc.update(ScoreboardReading(0.1, 0, 0, 2, 720.0, source="sim"))
    assert [(u.old, u.new) for u in out if isinstance(u, PeriodChange)] == [(1, 2)]
    for i in range(8):
        sc.update(ScoreboardReading(0.2 + 0.2 * i, 0, 0, 2, 715.0, source="sim"))
    assert sc.clock_stopped(1.6, 2, 1.2) is True
    assert ScoreboardConfirmer().clock_stopped(0.0) is None


# ---------------------------------------------------------------------------- engine reconciliation


def with_ocr(frames: list[FrameState], score_at: Callable[[float], tuple[int, int]], every: int = 6) -> list[FrameState]:
    """Replace the scoreboard with 5 Hz OCR reads (every 6th frame at 30 fps)."""
    for fr in frames:
        if fr.frame_idx % every == 0:
            h, a = score_at(fr.t)
            fr.scoreboard = ocr(fr.t, h, a, 600.0 - fr.t)
        else:
            fr.scoreboard = None
    return frames


def made_scene(dx: float = 18.0, pass_first: bool = False) -> Scene:
    s = two_team_scene()
    s.add("home-1", dx, 30.0).add("home-2", 30.0, 38.0).add("away-3", 4.0, 34.0)
    if pass_first:
        s.start("home-2").hold("home-2", 0.5).pass_to("home-1").hold("home-1", 0.6)
    else:
        s.start("home-1").hold("home-1", 0.8)
    s.shoot("left", made=True)
    s.mark("made")
    s.drop_through("left")
    s.fly(lambda pos: s.hands("away-3", pos), 0.4, apex=0.3).hold("away-3", 1.2)
    s.mark("pickup")
    s.hold("away-3", 3.0)
    return s


@pytest.mark.parametrize("mode", MODES)
def test_confirm_make_from_ocr_delta(mode: str) -> None:
    s = made_scene()
    t_made = s.marks["made"]
    frames = with_ocr(s.frames(mode), lambda t: (2, 0) if t >= t_made + 0.3 else (0, 0))
    eng = run_engine(frames)
    made = one(eng.events, EventKind.SHOT_MADE)
    assert made.status is EventStatus.CONFIRMED and made.attrs["confirmed_by"] == "scoreboard"
    assert one(eng.events, EventKind.SHOT_ATTEMPT).status is EventStatus.CONFIRMED
    assert not live(eng.events, EventKind.SCORE_CORRECTION)


@pytest.mark.parametrize("mode", MODES)
def test_miss_flipped_to_make_retracts_rebound_and_derives_assist(mode: str) -> None:
    s = two_team_scene()
    s.add("home-1", 18.0, 30.0).add("home-2", 30.0, 38.0).add("away-3", 9.0, 25.0)
    s.start("home-2").hold("home-2", 0.5).pass_to("home-1").hold("home-1", 0.6)
    s.shoot("left", made=False)  # the engine sees a front-rim miss ...
    s.fly(lambda pos: s.hands("away-3", pos), 0.6, apex=3.0).hold("away-3", 0.6)
    s.set_score(2, 0)  # ... but the (lagging) scoreboard says it went in
    s.hold("away-3", 2.0)
    eng = run_engine(s.frames(mode))
    ev = eng.events
    made = one(ev, EventKind.SHOT_MADE)
    assert made.status is EventStatus.CONFIRMED and made.attrs["flipped_by"] == "scoreboard" and made.revision >= 1
    assert made.value == 2 and not live(ev, EventKind.SHOT_MISSED)
    reb = [e for e in ev if e.kind is EventKind.REBOUND]
    assert len(reb) == 1 and reb[0].status is EventStatus.RETRACTED
    ast = one(ev, EventKind.ASSIST)
    assert (ast.player_id, ast.other_player_id, ast.shot_id) == ("home-2", "home-1", made.shot_id)
    poss = [e for e in live(ev, EventKind.POSSESSION) if e.team is Team.AWAY]
    assert len(poss) == 1 and poss[0].attrs["cause"] == "made_shot" and poss[0].revision == 1


def test_flip_before_any_rebound_starts_the_post_make_dead_ball() -> None:
    s = two_team_scene()
    s.add("home-1", 18.0, 30.0).add("away-3", 2.0, 30.0).add("away-4", -1.5, 28.0)
    s.start("home-1").hold("home-1", 0.6).shoot("left", made=False)
    s.set_score(2, 0)  # delta arrives while the ball is still loose
    s.fly((3.0, 27.0, 0.39), 0.5, apex=1.0)
    s.fly(lambda pos: s.hands("away-4", pos), 0.5, apex=0.5).hold("away-4", 1.0)
    s.pass_to("away-3").hold("away-3", 0.5)
    eng = run_engine(s.frames_3d())
    ev = eng.events
    assert one(ev, EventKind.SHOT_MADE).attrs["flipped_by"] == "scoreboard"
    assert not live(ev, EventKind.REBOUND) and not live(ev, EventKind.STEAL)
    poss = [e for e in live(ev, EventKind.POSSESSION) if e.team is Team.AWAY]
    assert len(poss) == 1 and poss[0].attrs["cause"] == "made_shot" and poss[0].player_id == "away-4"


@pytest.mark.parametrize("mode", MODES)
def test_value_fixed_by_scoreboard(mode: str) -> None:
    s = made_scene(dx=18.0)  # a 2 for the engine
    eng_frames = s.frames(mode)
    t_made = s.marks["made"]
    for fr in eng_frames:
        assert fr.scoreboard is not None
        h = 3 if fr.t >= t_made + 0.2 else 0
        fr.scoreboard = type(fr.scoreboard)(fr.t, h, 0, 1, fr.scoreboard.game_clock, None, source="sim")
    eng = run_engine(eng_frames)
    att, made = one(eng.events, EventKind.SHOT_ATTEMPT), one(eng.events, EventKind.SHOT_MADE)
    assert att.value == 3 and made.value == 3 and made.status is EventStatus.CONFIRMED
    assert made.attrs["value_fixed_by"] == "scoreboard"


def test_unmatched_delta_becomes_score_correction() -> None:
    s = two_team_scene().start("home-1").hold("home-1", 1.0)
    s.set_score(0, 1).hold("home-1", 3.0)  # an away free throw the camera never saw
    eng = run_engine(s.frames_3d())
    sc = one(eng.events, EventKind.SCORE_CORRECTION)
    assert sc.team is Team.AWAY and sc.value == 1 and sc.attrs["needs_review"] is True
    assert sc.event_id.startswith("r") and sc.source == "scoreboard" and abs(sc.t - 1.0) < 0.1


@pytest.mark.parametrize("mode", MODES)
def test_make_without_delta_revised_to_miss_and_pickup_becomes_rebound(mode: str) -> None:
    s = made_scene(pass_first=True)  # scoreboard streams 0-0 throughout
    eng = run_engine(s.frames(mode))
    ev = eng.events
    assert not live(ev, EventKind.SHOT_MADE)
    miss = one(ev, EventKind.SHOT_MISSED)
    assert miss.attrs["needs_review"] is True and miss.revision >= 1
    assert all(e.status is EventStatus.RETRACTED for e in ev if e.kind is EventKind.ASSIST)
    reb = one(ev, EventKind.REBOUND)
    assert reb.player_id == "away-3" and reb.attrs["offensive"] is False
    assert abs(reb.t - s.marks["pickup"]) < 1.5


def test_make_kept_without_a_streaming_scoreboard() -> None:
    s = made_scene()
    s.scoreboard = False
    eng = run_engine(s.frames_3d())
    assert one(eng.events, EventKind.SHOT_MADE).status is EventStatus.PROVISIONAL
