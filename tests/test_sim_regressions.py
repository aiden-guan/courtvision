"""Regression tests for defects found in the adversarial review of the sim module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from courtvision.geometry.camera import PinholeCamera
from courtvision.geometry.court import court_for
from courtvision.sim import palette
from courtvision.sim.noise import NoiseConfig, NoiseModel, floor_homography_from_frames
from courtvision.sim.render import STANDS_BGR, RenderConfig, SyntheticRenderer
from courtvision.sim.simulator import GameSimulator, SimConfig, SimResult, _Game, _Shot
from courtvision.types import EventKind as K
from courtvision.types import FrameState, Team


def _game(**kw: object) -> _Game:
    return _Game(SimConfig(**{"seconds": 10.0, "fps": 5.0, "seed": 1, **kw}))  # type: ignore[arg-type]


def _shot(g: _Game, made: bool, rim_touched: bool = False, blocked_by: int | None = None) -> _Shot:
    return _Shot(shot_id="s9999", slot=0, pid=g._pid(0), team=Team.HOME, side=g.attack[Team.HOME], value=2,
                 shot_type="layup", made=made, blocked_by=blocked_by, is_ft=False, release_t=g.t, x=0.0, y=0.0,
                 rim_touched=rim_touched)


# ---------------------------------------------------------------------------- shots


def test_flight_time_safe_when_target_is_straight_below() -> None:
    """``_flight_time`` raised a math domain error for a target (almost) straight below the release."""
    g = _game()
    T = g._flight_time(np.array([10.0, 25.0, 10.5]), np.array([10.01, 25.0, 10.0]), math.radians(60.0))
    assert math.isfinite(T) and T > 0.0


def test_close_made_shot_resolves_at_low_fps() -> None:
    """A make from just under the rim used to be detected only by a frame-sampled rim crossing: at
    5 fps a short arc crossed between frames, the shot never resolved and the period could not end."""
    g = _game(fps=5.0)
    g.period, g.game_clock = 1, 100.0
    bx, by = g.court.basket_xy(g.attack[Team.HOME])
    sign = 1.0 if g.attack[Team.HOME] == "left" else -1.0
    g.pos[0] = (bx + sign * 0.3, by)  # practically under the rim
    g._hold(0)
    shot = _shot(g, made=True)
    g._launch_shot(shot, g.pos[0].copy(), np.array([bx, by]), 9.9)  # asks for a release below the rim
    assert g.ball.pos[2] >= g.rim_h + g.ball.r  # inside the rim cylinder the ball starts above the rim
    flight = g._shot_flight(shot)
    result = None
    for _ in range(int(3 * g.cfg.fps)):
        g.t += g.dt
        g._advance_ball(g.dt)
        try:
            next(flight)
        except StopIteration as stop:
            result = stop.value
            break
    assert result == "made"
    assert [e.kind for e in g.events] == [K.SHOT_MADE]


@pytest.mark.parametrize("fps", [4.0, 7.0])
def test_every_attempt_resolves_at_low_fps(fps: float) -> None:
    r = GameSimulator(SimConfig(seconds=400.0, fps=fps, seed=40)).run()
    results = {e.shot_id for e in r.events if e.kind in (K.SHOT_MADE, K.SHOT_MISSED)}
    end = r.frames[-1].t
    for a in (e for e in r.events if e.kind is K.SHOT_ATTEMPT and e.t < end - 4.0):
        assert a.shot_id in results, a


def test_tip_putbacks_are_reachable() -> None:
    """``_phase_putback`` read the ball height after ``_hold`` had already snapped it to 3.75 ft, so a
    tip (shot_type "tip") could never happen."""
    r = GameSimulator(SimConfig(seconds=600.0, fps=10.0, seed=2, putback_rate=1.0, oreb_rate=0.7)).run()
    tips = [e for e in r.events if e.kind is K.SHOT_ATTEMPT and e.attrs["shot_type"] == "tip"]
    assert tips
    for a in tips:
        assert a.attrs["sim_play"] == "putback" and r.frames[a.frame_idx].ball.z >= 9.0


# ---------------------------------------------------------------------------- clocks & possessions


def test_shot_clock_expiry_on_a_loose_blocked_ball_is_a_violation() -> None:
    """After a block (no rim contact) the shot clock keeps running; expiry before control is a
    shot-clock turnover with no rebound (note §2a). It used to credit an OREB, then the violation."""
    g = _game()
    g.possession_team, g.ball_live, g.ball.mode = Team.HOME, True, "flight"
    g.last_holder_pid = g._pid(0)
    g.loose_after, g.sc_expired = _shot(g, made=False, blocked_by=5), True
    assert g._interrupt() is not None
    assert g.loose_after is None
    kinds = [e.kind for e in g.events]
    assert kinds == [K.TURNOVER] and g.events[0].attrs["type"] == "shot_clock"
    assert g.events[0].player_id == g._pid(0)
    # a miss that touched the rim stops the shot clock: no violation, the rebound continues
    g2 = _game()
    g2.possession_team, g2.ball_live, g2.ball.mode = Team.HOME, True, "flight"
    g2.loose_after, g2.sc_expired = _shot(g2, made=False, rim_touched=True), True
    assert g2._interrupt() is None and g2.loose_after is not None and not g2.events


def test_baseline_throw_in_uses_the_baseline_the_ball_crossed() -> None:
    g = _game()
    L = g.court.length
    for team in (Team.HOME, Team.AWAY):
        assert g._inbound_spot(team, ("baseline", np.array([-2.0, 12.0])))[0] == pytest.approx(-1.5)
        assert g._inbound_spot(team, ("baseline", np.array([L + 2.0, 12.0])))[0] == pytest.approx(L + 1.5)
        defended = g.court.other_side(g.attack[team])  # after a made basket: under the defended basket
        assert g._inbound_spot(team, ("baseline", None))[0] == pytest.approx(-1.5 if defended == "left" else L + 1.5)


def test_loose_ball_recovery_keeps_its_cause_and_starts_the_clock() -> None:
    g = _game()
    g.period, g.game_clock, g.clock_running, g.ball_live = 1, 100.0, False, True
    g.possession_team = None
    g.ball.mode, g.ball.vel[:] = "flight", 0.0
    g.ball.pos[:] = (*g.pos[1], 3.0)
    phase = g._phase_loose_ball(Team.HOME, cause="jump_ball")
    with pytest.raises(StopIteration):
        next(phase)
    poss = [e for e in g.events if e.kind is K.POSSESSION]
    assert len(poss) == 1 and poss[0].attrs["cause"] == "jump_ball"
    assert g.clock_running  # a missed throw-in touched inbounds starts the game clock


def test_dead_balls_stay_near_the_court() -> None:
    """A ball thrown out of bounds used to roll ~100 ft away; the throw-in toss then flew 50 ft high."""
    cfg = SimConfig(seconds=240.0, fps=10.0, seed=3, turnover_rate=0.6,
                    turnover_mix={"intercept": 0.0, "strip": 0.0, "oob": 1.0, "travel": 0.0, "shot_clock": 0.0})
    r = GameSimulator(cfg).run()
    assert sum(e.kind is K.TURNOVER and e.attrs["type"] == "out_of_bounds" for e in r.events) >= 3
    L, W = r.court.length, r.court.width
    worst = max(math.hypot(f.ball.court_xy[0] - min(max(f.ball.court_xy[0], 0.0), L),
                           f.ball.court_xy[1] - min(max(f.ball.court_xy[1], 0.0), W)) for f in r.frames)
    assert worst <= 12.0, worst  # the stop distance (6 ft) plus an overthrown lob's landing; was 65-100 ft
    assert max(f.ball.z for f in r.frames) < 25.0  # throw-in tosses from ~100 ft away peaked at 30-70 ft


def test_overtime_jump_ball_does_not_teleport_the_ball() -> None:
    """The jump ball put the ball at centre court in one frame (16 ft jumps at an NBA overtime).
    The seed is the first from 8 whose short-period game reaches overtime (re-pinned when the simulator's
    game flow changes)."""
    r = GameSimulator(SimConfig(seconds=200.0, fps=30.0, seed=8, period_s=15.0, ot_s=10.0)).run()
    ot = [e for e in r.events if e.kind is K.PERIOD_START and e.attrs["period"] > r.rules.n_periods]
    assert ot and r.rules.possession == "jump"
    ball = np.array([[*f.ball.court_xy, f.ball.z] for f in r.frames])
    speed = np.linalg.norm(np.diff(ball, axis=0), axis=1) * r.config.fps
    assert speed.max() < 60.0, (int(speed.argmax()) + 1, float(speed.max()))
    assert any(e.kind is K.POSSESSION and e.attrs["cause"] == "jump_ball" and e.t > ot[0].t for e in r.events)


def test_timeout_allowance_from_rulespec() -> None:
    """FIBA / NFHS YAMLs have no plain "count"; the allowance fell back to 6 per team."""
    for league, n in (("nba", 7), ("fiba", 5), ("nfhs", 5)):
        assert _game(league=league).timeouts_left == {Team.HOME: n, Team.AWAY: n}


# ---------------------------------------------------------------------------- noise & render


@pytest.fixture(scope="module")
def short_sim() -> SimResult:
    return GameSimulator(SimConfig(seconds=20.0, fps=30.0, seed=9)).run()


def test_noise_without_camera_recovers_the_floor_projection(short_sim: SimResult) -> None:
    """``NoiseModel(NoiseConfig()).apply(frames, seed)`` (the ARCHITECTURE call) gave court_xy None in
    monocular mode; the floor homography is now recovered from the 3-D truth in the frames."""
    cfg = NoiseConfig(ball_fp_rate=0.0, ball_px_sigma=0.0)
    blind = NoiseModel(cfg).apply(short_sim.frames, seed=5)
    known = NoiseModel(cfg, camera=short_sim.camera).apply(short_sim.frames, seed=5)
    vis = [(a.ball, b.ball) for a, b in zip(blind, known, strict=True) if a.ball.visible]
    assert len(vis) > 100
    for a, b in vis:
        assert a.z is None
        assert np.allclose(a.court_xy, b.court_xy, atol=1e-6)
    H = floor_homography_from_frames(short_sim.frames)
    uv = np.array([[640.0, 360.0], [100.0, 650.0], [640.0, 5.0]])
    assert np.allclose(H.image_to_court(uv, front_only=True), short_sim.camera.floor_homography().image_to_court(
        uv, front_only=True), atol=1e-6, equal_nan=True)
    flat = [FrameState(f.frame_idx, f.t, [], f.ball, scoreboard=None) for f in NoiseModel(cfg).apply(
        short_sim.frames[:5], seed=1)]  # no 3-D truth left (z is None): nothing to fit
    assert floor_homography_from_frames(flat) is None


def test_renderer_clips_geometry_behind_the_camera() -> None:
    """Floor points behind the camera projected to nan -> INT_MIN and drew garbage polygons."""
    court = court_for("nba")
    cam = PinholeCamera.look_at((60.0, 25.0, 20.0), (90.0, 25.0, 0.0), (640, 360), 72.0)
    r = SyntheticRenderer(cam, court, RenderConfig(draw_scoreboard=False))
    img = r.render(FrameState(0, 0.0, [], None))
    H = cam.floor_homography()
    a = r.config.apron_ft
    board = np.vstack([p for _, polys, ring in r._hoops for p in polys + ring])
    bx0, by0 = board.min(axis=0) - 3
    bx1, by1 = board.max(axis=0) + 3
    checked = 0
    for v in range(4, 360, 16):
        for u in range(4, 640, 16):
            if bx0 <= u <= bx1 and by0 <= v <= by1:
                continue
            x, y = H.image_to_court(np.array([u + 0.5, v + 0.5]), front_only=True)
            hit = bool(np.isfinite(x))
            edge = hit and min(abs(x + a), abs(x - court.length - a), abs(y + a), abs(y - court.width - a)) < 1.5
            if edge:
                continue
            px = tuple(int(c) for c in img[v, u])
            if not (hit and -a < x < court.length + a and -a < y < court.width + a):
                assert px == STANDS_BGR, (u, v, x, y)  # beyond the horizon / off the floor
                checked += 1
            elif 0.5 < x < court.length - 0.5 and 0.5 < y < court.width - 0.5:
                assert px in (palette.FLOOR_BGR, palette.PAINT_BGR, palette.LINE_BGR), (u, v, x, y)
    assert checked > 50
    for _, polys, ring in r._hoops:
        for p in polys + ring:
            assert np.abs(p).max() < 1e6


def test_low_fps_catches_happen_where_the_hands_meet_the_ball() -> None:
    """At 10 fps a caught pass used to be held where it had flown on to by the end of the frame (up to
    ~4 ft past the catch point, out of bounds near a sideline): the catch is at its exact flight time."""
    r = GameSimulator(SimConfig(seconds=600.0, fps=10.0, seed=5, period_s=240.0)).run()
    checked = 0
    for p in (e for e in r.events if e.kind is K.PASS and e.attrs["completed"]):
        i = int(round(p.attrs["catch_t"] * r.config.fps))
        f = r.frames[i]
        recv = f.player(p.other_player_id)
        if recv is None or not r.court.in_bounds(recv.x, recv.y):
            continue
        bx, by = f.ball.court_xy
        assert r.court.in_bounds(bx, by), (p.t, bx, by)
        assert math.hypot(bx - recv.x, by - recv.y) <= 4.0
        checked += 1
    assert checked > 50


def test_tip_putbacks_go_straight_back_up() -> None:
    """A tip is released from the catch, not carried overhead to a release spot (it was held at ~9.5 ft
    for up to 0.7 s)."""
    r = GameSimulator(SimConfig(seconds=600.0, fps=30.0, seed=12, putback_rate=1.0, oreb_rate=0.7)).run()
    tips = [a for a in r.events if a.kind is K.SHOT_ATTEMPT and a.attrs.get("shot_type") == "tip"]
    assert tips
    for a in tips:
        reb = max((e for e in r.events if e.kind is K.REBOUND and e.t <= a.t), key=lambda e: e.t)
        assert reb.player_id == a.player_id and a.t - reb.t <= 0.35, (reb.t, a.t)
