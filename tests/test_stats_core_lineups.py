"""Elapsed-game-time timeline, operator lineup stints and lineup aggregation (stats/lineups.py)."""

from __future__ import annotations

import logging

import pytest

from courtvision.stats.boxscore import Possession, ScoringPlay
from courtvision.stats.lineups import GameClockTimeline, LineupTracker, compute_lineups, lineup_id
from courtvision.types import ScoreboardReading, Team

H, A = Team.HOME, Team.AWAY


def nba_length(period: int) -> float:
    return 720.0 if period <= 4 else 300.0


def reading(t: float, period: int, clock: float) -> ScoreboardReading:
    return ScoreboardReading(t=t, period=period, game_clock=clock, source="sim")


# --------------------------------------------------------------------------- GameClockTimeline


def test_clock_elapsed_interpolates_and_spans_periods() -> None:
    tl = GameClockTimeline(nba_length)
    tl.update(0.0, reading(0.0, 1, 720.0))
    tl.update(20.0, reading(20.0, 1, 700.0))
    assert tl.elapsed_at(10.0) == pytest.approx(10.0)
    assert tl.elapsed_at(-5.0) == pytest.approx(0.0) and tl.elapsed_at(99.0) == pytest.approx(20.0)
    tl.update(30.0, reading(30.0, 2, 720.0))
    assert tl.total == pytest.approx(720.0)
    tl.update(40.0, reading(40.0, 5, 290.0))  # first overtime
    assert tl.total == pytest.approx(4 * 720 + 10)
    assert tl.basis == "game_clock"


def test_clock_never_runs_backwards() -> None:
    tl = GameClockTimeline(nba_length)
    tl.update(0.0, reading(0.0, 1, 700.0))
    tl.update(1.0, reading(1.0, 1, 719.0))  # OCR glitch
    assert tl.total == pytest.approx(20.0)


def test_stale_clock_falls_back_to_video_time() -> None:
    tl = GameClockTimeline(nba_length, clock_stale_s=3.0)
    tl.update(20.0, reading(20.0, 1, 700.0))
    for i in range(201, 301):  # 10 Hz frames with no reading until t = 30
        tl.update(i / 10.0, None)
    assert tl.total == pytest.approx(20.0 + 7.0)  # held for 3 s, then video dt
    assert tl.basis == "mixed"


def test_video_time_caps_jumps_and_respects_live() -> None:
    tl = GameClockTimeline(nba_length, max_dt_s=0.5)
    for i in range(0, 101):
        tl.update(i / 10.0, None)
    tl.update(20.0, None)  # 10 s pause in the stream counts 0.5 s
    assert tl.total == pytest.approx(10.5)
    tl.update(21.0, None, live=False)
    assert tl.total == pytest.approx(10.5)
    assert tl.basis == "video" and GameClockTimeline(nba_length).elapsed_at(5.0) is None


# --------------------------------------------------------------------------- LineupTracker


def test_stints_follow_entries() -> None:
    lt = LineupTracker()
    lt.set_lineup(H, ["h1", "h2", "h3", "h4", "h5"], 0.0)
    lt.substitute(H, "h5", "h6", 40.0)
    lt.substitute(H, "h1", "h7", 40.0)  # two subs at one dead ball: one stint boundary
    lt.set_lineup(A, ["a1", "a2", "a3", "a4", "a5"], 0.0)
    st = lt.stints()
    home = [s for s in st if s.team is H]
    assert [(s.players, s.start_t, s.end_t) for s in home] == [
        (("h1", "h2", "h3", "h4", "h5"), 0.0, 40.0),
        (("h2", "h3", "h4", "h6", "h7"), 40.0, None),
    ]
    assert lt.on_court(H) == ["h2", "h3", "h4", "h6", "h7"]
    assert lt.on_court(H, t=10.0) == ["h1", "h2", "h3", "h4", "h5"]
    assert lt.has_entries(A) and lt.has_entries()


def test_entries_arriving_out_of_order_are_sorted() -> None:
    lt = LineupTracker()
    lt.substitute(H, "h5", "h6", 40.0)  # entered late, before the starting five
    lt.set_lineup(H, ["h1", "h2", "h3", "h4", "h5"], 0.0)
    assert [(s.start_t, s.end_t) for s in lt.stints()] == [(0.0, 40.0), (40.0, None)]


def test_bad_substitutions_warn(caplog: pytest.LogCaptureFixture) -> None:
    lt = LineupTracker()
    lt.set_lineup(H, ["h1", "h2", "h1"], 0.0)
    with caplog.at_level(logging.WARNING):
        lt.substitute(H, "h9", "h3", 5.0)  # h9 is not on the floor: h3 still comes in
        lt.substitute(H, "h1", "h2", 6.0)  # h2 already on the floor
    assert lt.on_court(H) == ["h2", "h3"]
    assert sum("not on the floor" in r.message or "already on the floor" in r.message for r in caplog.records) == 2
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        lt.stints()  # rebuilding stints (every snapshot) does not repeat the warnings
    assert not caplog.records


# --------------------------------------------------------------------------- compute_lineups


def test_compute_lineups_with_linear_clock() -> None:
    lt = LineupTracker()
    lt.set_lineup(H, ["h1", "h2"], 0.0)
    lt.set_lineup(A, ["a1", "a2"], 0.0)
    lt.substitute(H, "h2", "h3", 50.0)
    scoring = [ScoringPlay(10.0, H, 2, "e1"), ScoringPlay(50.0, A, 3, "e2"), ScoringPlay(60.0, H, 1, "e3")]
    poss = [Possession(0, H, 0.0, "jump_ball"), Possession(1, A, 11.0, "made_shot"), Possession(2, H, 51.0, "made_shot")]
    res = compute_lineups(lt.stints(), lambda t: t, 100.0, scoring, poss, min_possessions=1)
    first = res.lineups[lineup_id(H, ["h1", "h2"])]
    second = res.lineups[lineup_id(H, ["h3", "h1"])]
    assert (first["seconds"], first["pts_for"], first["pts_against"], first["plus_minus"]) == (50.0, 2, 0, 2)
    assert (second["seconds"], second["pts_for"], second["pts_against"]) == (50.0, 1, 3)  # the 3 at t=50 is theirs
    assert (first["off_possessions"], first["def_possessions"], first["small_sample"]) == (1, 1, False)
    assert second["small_sample"] and second["net_rtg"] is None
    assert second["current"] and not first["current"]
    assert res.player_seconds == {"h1": 100.0, "h2": 50.0, "h3": 50.0, "a1": 100.0, "a2": 100.0}
    assert res.player_plus_minus == {"h1": 0, "h2": 2, "h3": -2, "a1": 0, "a2": 0}


# --------------------------------------------------------------------------- review regressions


def ocr(t: float, period: int, clock: float, read_t: float | None = None) -> ScoreboardReading:
    return ScoreboardReading(t=t if read_t is None else read_t, period=period, game_clock=clock, source="ocr")


def test_ocr_clock_jump_needs_a_second_consistent_read() -> None:
    tl = GameClockTimeline(nba_length)
    tl.update(0.0, ocr(0.0, 1, 720.0))
    tl.update(1.0, ocr(1.0, 1, 719.0))
    tl.update(1.2, ocr(1.2, 3, 300.0))  # misread: 1860 s of game time in 0.2 s of video
    assert tl.total == pytest.approx(1.0)
    tl.update(1.4, ocr(1.4, 1, 718.6))  # back to normal: the glitch is dropped
    assert tl.total == pytest.approx(1.4)
    # A real jump (the scoreboard came into view mid-game) is confirmed by the next distinct read.
    tl.update(2.0, ocr(2.0, 3, 300.0))
    assert tl.total == pytest.approx(1.4)
    tl.update(2.2, ocr(2.2, 3, 299.8))
    assert tl.total == pytest.approx(2 * 720 + 420.2)


def test_one_ocr_read_on_several_frames_does_not_confirm_a_jump() -> None:
    tl = GameClockTimeline(nba_length)
    tl.update(0.0, ocr(0.0, 1, 720.0))
    glitch = ocr(1.0, 4, 100.0)
    tl.update(1.0, glitch)
    tl.update(1.033, glitch)  # the same read attached to the next frame
    assert tl.total == pytest.approx(0.0)


def test_trusted_clock_jumps_apply_at_once() -> None:
    tl = GameClockTimeline(nba_length)
    tl.update(0.0, reading(0.0, 1, 720.0))
    tl.update(1.0, reading(1.0, 2, 600.0))  # operator / feed correction
    assert tl.total == pytest.approx(840.0)


def test_non_finite_clock_reading_is_ignored() -> None:
    tl = GameClockTimeline(nba_length)
    tl.update(0.0, reading(0.0, 1, 700.0))
    tl.update(1.0, reading(1.0, 1, float("nan")))
    tl.update(2.0, reading(2.0, 1, 698.0))
    assert tl.total == pytest.approx(22.0)
    assert all(v == v for v in (tl.elapsed_at(0.5), tl.elapsed_at(1.5)))  # no nan anywhere


def test_lineup_possessions_are_per_side_not_both_ends() -> None:
    """10 trips each way is a 10-possession sample (B-Ref-style mean of both ends), so the
    20-possession small-sample gate keeps hiding it."""
    lt = LineupTracker()
    lt.set_lineup(H, ["h1"], 0.0)
    poss = [Possession(i, H if i % 2 == 0 else A, 10.0 * i, "made_shot") for i in range(20)]
    rec = compute_lineups(lt.stints(), lambda t: t, 300.0, [], poss, min_possessions=20).lineups[lineup_id(H, ["h1"])]
    assert (rec["off_possessions"], rec["def_possessions"], rec["possessions"]) == (10, 10, 10.0)
    assert rec["small_sample"]
