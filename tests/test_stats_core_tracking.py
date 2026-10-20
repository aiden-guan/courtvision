"""Tracking stats on synthetic paths (stats/tracking.py)."""

from __future__ import annotations

import numpy as np
import pytest

from courtvision.geometry.court import court_for
from courtvision.stats.tracking import TrackingAccumulator, event_counts
from courtvision.types import BallState, Event, EventKind, EventStatus, FrameState, PlayerState, Team

H, A = Team.HOME, Team.AWAY
FPS = 30.0


def frame(i: int, t: float, players: list[PlayerState]) -> FrameState:
    return FrameState(frame_idx=i, t=t, players=players, ball=BallState(visible=False))


def run_path(xs: list[float], ys: list[float], fps: float = FPS, acc: TrackingAccumulator | None = None,
             skip: set[int] | None = None) -> TrackingAccumulator:
    acc = acc or TrackingAccumulator(court_for("nba"))
    for i, (x, y) in enumerate(zip(xs, ys)):
        players = [] if skip and i in skip else [PlayerState("home-1", H, x, y)]
        acc.update(frame(i, i / fps, players), None)
    return acc


def test_straight_line_distance_and_speeds() -> None:
    n = 91  # 3 s at 30 fps, 10 ft/s
    xs = [20.0 + 10.0 * i / FPS for i in range(n)]
    line = run_path(xs, [25.0] * n).lines()["home-1"]
    assert line.distance_ft == pytest.approx(30.0)
    assert line.seconds_tracked == pytest.approx(3.0)
    assert line.avg_speed_fts == pytest.approx(10.0)
    assert line.top_speed_fts == pytest.approx(10.0)
    assert line.clamped_steps == 0


def test_identity_jump_is_clamped() -> None:
    n = 91
    xs = [20.0 + 10.0 * i / FPS for i in range(n)]
    ys = [25.0 if i < 45 else 45.0 for i in range(n)]  # 20 ft sideways in one frame = 600 ft/s
    line = run_path(xs, ys).lines()["home-1"]
    assert line.clamped_steps == 1
    assert line.distance_ft == pytest.approx(30.0 - 10.0 / FPS)  # every step but the jump
    assert line.seconds_tracked == pytest.approx(3.0 - 1.0 / FPS)
    assert line.top_speed_fts == pytest.approx(10.0)


def test_speed_clamp_threshold_is_configurable() -> None:
    n = 31
    xs = [10.0 + 35.0 * i / FPS for i in range(n)]  # 35 ft/s: above the default 30 ft/s clamp
    assert run_path(xs, [25.0] * n).lines()["home-1"].distance_ft == pytest.approx(0.0)
    fast = TrackingAccumulator(court_for("nba"), max_speed_fts=40.0)
    assert run_path(xs, [25.0] * n, acc=fast).lines()["home-1"].distance_ft == pytest.approx(35.0)


def test_top_speed_is_smoothed_against_jitter() -> None:
    rng = np.random.default_rng(7)
    n = 300
    xs = list(50.0 + rng.normal(0.0, 0.1, n))
    ys = list(25.0 + rng.normal(0.0, 0.1, n))
    line = run_path(xs, ys).lines()["home-1"]
    raw = np.hypot(np.diff(xs), np.diff(ys)) * FPS
    assert line.top_speed_fts < 0.15 * raw.max()
    assert line.top_speed_fts < 1.5  # per-step speeds here average ~5 ft/s with peaks of 12+


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_jitter_top_speed_is_robust_across_seeds(seed: int) -> None:
    rng = np.random.default_rng(seed)
    xs = list(50.0 + rng.normal(0.0, 0.1, 300))
    ys = list(25.0 + rng.normal(0.0, 0.1, 300))
    assert run_path(xs, ys).lines()["home-1"].top_speed_fts < 1.5


def test_optional_position_smoothing_suppresses_jitter_distance() -> None:
    rng = np.random.default_rng(11)
    n = 300  # 10 s standing still with 0.1 ft jitter
    xs = list(50.0 + rng.normal(0.0, 0.1, n))
    ys = list(25.0 + rng.normal(0.0, 0.1, n))
    raw = run_path(xs, ys).lines()["home-1"].distance_ft
    smooth_acc = TrackingAccumulator(court_for("nba"), position_smoothing_tau_s=0.3)
    smooth = run_path(xs, ys, acc=smooth_acc).lines()["home-1"].distance_ft
    assert smooth < 0.2 * raw
    m = 91  # a 10 ft/s straight run loses little to the smoothing lag
    run = run_path([20.0 + 10.0 * i / FPS for i in range(m)], [25.0] * m,
                   acc=TrackingAccumulator(court_for("nba"), position_smoothing_tau_s=0.3))
    assert run.lines()["home-1"].distance_ft == pytest.approx(30.0, abs=3.5)


def test_sprint_reaches_its_top_speed() -> None:
    v = [min(20.0, 10.0 * i / FPS) for i in range(121)]  # 0 -> 20 ft/s in 2 s, then 2 s flat
    xs = list(np.concatenate([[10.0], 10.0 + np.cumsum(np.array(v[1:]) / FPS)]))
    line = run_path(xs, [25.0] * len(xs)).lines()["home-1"]
    assert line.top_speed_fts == pytest.approx(20.0, rel=0.02)
    assert line.distance_ft == pytest.approx(sum(v[1:]) / FPS)


def test_gap_restarts_without_a_step() -> None:
    n = 91
    xs = [20.0 + 10.0 * i / FPS for i in range(n)]
    xs[60:] = [x + 15.0 for x in xs[60:]]  # reappears 15 ft further after a 1 s gap
    line = run_path(xs, [25.0] * n, skip=set(range(30, 60))).lines()["home-1"]
    assert line.distance_ft == pytest.approx(10.0 * 29 / FPS + 10.0 * 30 / FPS)
    assert line.seconds_tracked == pytest.approx(59 / FPS)


def test_time_of_possession_and_touches() -> None:
    acc = TrackingAccumulator(court_for("nba"))
    holders = ["home-1"] * 10 + [None] * 2 + ["home-1"] * 5 + ["home-2"] * 10 + [None]
    for i, h in enumerate(holders):  # 10 Hz
        acc.update(frame(i, i / 10.0, []), h)
    lines = acc.lines()
    assert lines["home-1"].time_of_possession_s == pytest.approx(1.5)
    assert lines["home-2"].time_of_possession_s == pytest.approx(1.0)
    assert lines["home-1"].touches == 1  # the 0.3 s flicker is not a new touch
    assert lines["home-2"].touches == 1

    acc2 = TrackingAccumulator(court_for("nba"))
    holders = ["home-1"] * 5 + [None] * 8 + ["home-1"] * 5  # 0.9 s loose ball: a new touch
    for i, h in enumerate(holders):
        acc2.update(frame(i, i / 10.0, []), h)
    assert acc2.lines()["home-1"].touches == 2


def test_paint_touches_use_the_attacking_basket() -> None:
    acc = TrackingAccumulator(court_for("nba"))
    attack = {H: "right", A: "left"}
    players = [
        PlayerState("home-1", H, 60.0, 25.0),  # outside the paint
        PlayerState("home-2", H, 85.0, 25.0),  # home's attacking (right) paint
        PlayerState("away-1", A, 86.0, 24.0),  # away's own defensive paint
    ]
    for i, h in enumerate(["home-1"] * 3 + ["home-2"] * 3 + ["away-1"] * 3):
        acc.update(frame(i, i / 10.0, players), h, attack.get)
    lines = acc.lines()
    assert [lines[p].touches for p in ("home-1", "home-2", "away-1")] == [1, 1, 1]
    assert [lines[p].paint_touches for p in ("home-1", "home-2", "away-1")] == [0, 1, 0]


def test_stale_frames_are_ignored() -> None:
    acc = TrackingAccumulator(court_for("nba"))
    acc.update(frame(0, 0.0, [PlayerState("home-1", H, 10.0, 25.0)]), "home-1")
    acc.update(frame(1, 0.1, [PlayerState("home-1", H, 11.0, 25.0)]), "home-1")
    acc.update(frame(1, 0.1, [PlayerState("home-1", H, 11.0, 25.0)]), "home-1")  # duplicate
    acc.update(frame(0, 0.0, [PlayerState("home-1", H, 10.0, 25.0)]), "home-1")  # out of order
    line = acc.lines()["home-1"]
    assert line.distance_ft == pytest.approx(1.0) and line.time_of_possession_s == pytest.approx(0.1)


def test_event_counts_passes_and_dribbles() -> None:
    def ev(i: int, kind: EventKind, pid: str, other: str | None = None, **attrs: object) -> Event:
        return Event(f"e{i}", kind, float(i), i, team=H, player_id=pid, other_player_id=other, attrs=dict(attrs))

    events = [
        ev(1, EventKind.DRIBBLE, "home-1"),
        ev(2, EventKind.DRIBBLE, "home-1"),
        ev(3, EventKind.PASS, "home-1", "home-2", completed=True),
        ev(4, EventKind.PASS, "home-2", "home-3", completed=False),
        ev(5, EventKind.PASS, "home-3", "home-1"),  # completed defaults to True
    ]
    events.append(Event("e6", EventKind.PASS, 6.0, 6, player_id="home-1", other_player_id="home-2",
                        status=EventStatus.RETRACTED))
    c = event_counts(events)
    assert c["home-1"] == {"dribbles": 2, "passes_made": 1, "passes_received": 1}
    assert c["home-2"] == {"passes_received": 1, "passes_made": 1}
    assert c["home-3"] == {"passes_made": 1}
