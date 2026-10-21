"""Broadcast perception on simulator broadcast renders: court tracking of a panning / zooming camera, replay and
cutaway gating, scorebug OCR, and the sim broadcast plan itself."""

from __future__ import annotations

import math

import numpy as np
import pytest

from courtvision.perception.court_tracker import CourtTracker
from courtvision.perception.scorebug import GlyphBank, ScorebugLayout, ScorebugReader, parse_clock, parse_period
from courtvision.sim.broadcast import BroadcastConfig, BroadcastRenderer, plan_broadcast, scorebug_layout, scorebug_text
from courtvision.sim.simulator import GameSimulator, SimConfig
from courtvision.types import ScoreboardReading

FPS = 30.0


@pytest.fixture(scope="module")
def game():
    r = GameSimulator(SimConfig(seconds=120.0, fps=FPS, seed=3)).run()
    plan = plan_broadcast(r.frames, FPS, r.court.length, BroadcastConfig(seed=1, min_dead_s=3.0))
    return r, plan, BroadcastRenderer(r.frames, r.court, plan, r.feet_z, r.referees)


def test_plan_puts_replays_and_cutaways_in_dead_balls_only(game) -> None:
    r, plan, _ = game
    assert plan.segments, "the 120 s game has stoppages long enough for a replay or a cutaway"
    for kind, a, b in plan.segments:
        assert kind in ("replay", "cutaway")
        clocks = {r.frames[i].scoreboard.game_clock for i in range(a, b + 1)}
        assert len(clocks) == 1, (kind, a, b)  # the game clock is stopped throughout: no live action hidden
        if kind == "replay":
            assert all(plan.frames[i].src < a for i in range(a, b + 1)) and not plan.frames[a].scorebug
    live = [f for f in plan.frames if f.kind == "live"]
    pans = [math.degrees(math.atan2(f.camera.R[2][0], f.camera.R[2][1])) for f in live]
    assert max(pans) - min(pans) > 10.0  # the game camera follows the play


def test_court_tracker_follows_the_camera(game) -> None:
    """Started from the first frame's homography (an operator calibration), the tracker keeps the court mapping
    of the panning / zooming camera to ~0.1 ft, loses the court during a cutaway and re-acquires after it."""
    r, plan, br = game
    cam0 = plan.frames[0].camera
    trk = CourtTracker(r.court, cam0.image_size, cam0.floor_homography())
    grid = np.array([[u, v] for u in np.linspace(100, 1180, 6) for v in np.linspace(250, 650, 4)])
    errs, statuses = [], {}
    end = max(b for _, _, b in plan.segments) + 60
    for i in range(0, min(end, len(plan.frames))):
        res = trk.update(br.render(i))
        bf = plan.frames[i]
        statuses.setdefault(bf.kind, []).append(res.status)
        if bf.kind == "live" and res.homography is not None:
            true = bf.camera.floor_homography().image_to_court(grid, front_only=True)
            est = res.homography.image_to_court(grid, front_only=True)
            ok = np.all(np.isfinite(true), 1) & np.all(np.isfinite(est), 1) & (true[:, 1] > -5) & (true[:, 1] < 55)
            errs.append(float(np.median(np.linalg.norm(true[ok] - est[ok], axis=1))))
    live = statuses["live"]
    assert sum(s in ("tracking", "reacquired") for s in live) >= 0.97 * len(live)
    assert np.percentile(errs, 95) < 0.3 and max(errs) < 0.6
    for kind in ("replay", "cutaway"):
        if kind in statuses:  # the court of another camera / no court: never mistaken for the game camera
            assert sum(s == "lost" for s in statuses[kind]) >= 0.9 * len(statuses[kind])


def test_scorebug_reader_reads_the_board_and_sees_replays(game) -> None:
    r, plan, br = game
    lay = scorebug_layout((1280, 720))
    bank = GlyphBank.from_font(lay["fields"]["game_clock"][3], scale=0.9 * lay["scale"], thickness=2)
    reader = ScorebugReader(ScorebugLayout(lay["box"], lay["fields"]), bank)
    ok = n = 0
    for i in range(0, len(plan.frames), 29):
        img, bf = br.render(i), plan.frames[i]
        got = reader.read(img, r.frames[i].t)
        if bf.kind == "replay":
            assert got is None and not reader.present(img)
            continue
        tb = r.frames[i].scoreboard
        exp_clock = math.floor(tb.game_clock * 10 + 1e-9) / 10 if tb.game_clock < 60 else math.ceil(tb.game_clock - 1e-9)
        n += 1
        ok += got is not None and (got.home_score, got.away_score, got.period, got.game_clock) == \
            (tb.home_score, tb.away_score, tb.period, exp_clock)
    assert n > 50 and ok == n


def test_glyph_bank_learns_from_labelled_crops() -> None:
    """A bank with no templates learns them from crops whose text is known, then reads new values."""
    import cv2

    from courtvision.sim.broadcast import draw_scorebug

    lay = scorebug_layout((1280, 720))
    x, y, w, h = lay["fields"]["game_clock"]
    bank = GlyphBank()

    def crop(reading: ScoreboardReading):
        img = np.zeros((720, 1280, 3), np.uint8)
        draw_scorebug(img, reading)
        return img[y:y + h, x:x + w]

    for c in (754.0, 618.0, 323.0, 59.9, 27.4):  # 12:34, 10:18, 5:23, 59.9, 27.4: every digit, ":" and "."
        assert bank.learn(crop(ScoreboardReading(t=0, game_clock=c)), scorebug_text(ScoreboardReading(t=0, game_clock=c))["game_clock"])
    got = bank.read(crop(ScoreboardReading(t=0, game_clock=489.0)), 0.75)
    assert got is not None and got[0] == "8:09"
    assert cv2 is not None


@pytest.mark.parametrize(("text", "period"), [("Q1", 1), ("Q4", 4), ("OT", 5), ("2OT", 6), ("3RD", 3), ("X", None)])
def test_parse_period(text: str, period: int | None) -> None:
    assert parse_period(text) == period


def test_parse_clock() -> None:
    assert parse_clock("11:39") == 699.0 and parse_clock("59.3") == pytest.approx(59.3) and parse_clock("1:2") is None


def test_broadcast_mode_gates_cutaways_in_the_pipeline() -> None:
    """sim_mode=broadcast through LiveGame: the cutaway at 1-3 s yields no players or ball (the engine sees
    nothing), live frames are perceived with the tracked court and carry the camera centre."""
    from courtvision.cli import simulate_config
    from courtvision.perception.broadcast import BroadcastPerception
    from courtvision.pipeline import LiveGame, make_source

    cfg = simulate_config("broadcast", seed=3, seconds=20.0)
    src = make_source(cfg, realtime=False)
    game = LiveGame(cfg, src, realtime=False, env={})
    plan = src.broadcast()[0]
    cut = [i for i, f in enumerate(plan.frames[:150]) if f.kind == "cutaway"]
    assert cut, "seed 3 cuts away during the first stoppage"
    for i in range(0, 150):
        state, _ = game._perceive(src.render(i), i, src.result.frames[i].t)
        gate = game.perception.last_gate
        if i in cut[3:]:  # (the tracker holds the last homography for a couple of frames after the cut)
            assert not state.players and state.ball is None and not gate.live
        elif plan.frames[i].kind == "live" and i > 5:
            assert gate.live and len(state.players) >= 8 and state.camera_xyz is not None
    assert isinstance(game.perception, BroadcastPerception)
    assert np.allclose(game.perception.tracker.center, plan.frames[0].camera.center, atol=0.5)
