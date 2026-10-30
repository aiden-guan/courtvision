"""The shot judge (courtvision/nba/live_judge.py), the engine's verdict path (EventEngine.judge_shot) and its
switch off the scoreboard (EventEngineConfig.score_reconcile), and the rim evidence they read (nba/rim_clips.py,
makes.zone_features)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from courtvision.events.engine import EventEngineConfig
from courtvision.nba import makes as mk
from courtvision.nba.live_judge import JUDGE_FEATURES, JudgeConfig, ShotJudge, judge_from_config
from courtvision.nba.rim_clips import ZONES, crop_rim, zone_scores
from courtvision.types import BallState, BBox, Event, EventKind, EventStatus, FrameState, RimObservation, Team

from test_events_kit import MODES, live, one, run_engine, two_team_scene

ZI = {name: i for i, (name, _, _) in enumerate(ZONES)}
RIM = BBox(600.0, 300.0, 640.0, 308.0)  # a rim 40 px wide, its centre at (620, 304)


def at(dx: float, dy: float, score: float = 0.9) -> list[float]:
    """A ball candidate ``dx``, ``dy`` rim widths from the rim's centre."""
    return [620.0 + 40.0 * dx, 304.0 + 40.0 * dy, score]


# ---------------------------------------------------------------------------- rim evidence


def test_zone_scores_put_each_candidate_in_its_zone_and_ignore_weak_ones():
    balls = np.array([at(0.0, -1.0, 0.7), at(0.1, 1.0, 0.8), at(1.5, -0.5, 0.6), at(0.0, 1.0, 0.05)], np.float32)
    z = zone_scores(balls, RIM, min_score=0.08)
    assert z[ZI["above"]] == pytest.approx(0.7) and z[ZI["net"]] == pytest.approx(0.8)
    assert z[ZI["side"]] == pytest.approx(0.6) and z[ZI["ring"]] == 0.0 and z[ZI["high"]] == 0.0
    assert not zone_scores(None, RIM).any() and not zone_scores(np.zeros((0, 3), np.float32), RIM).any()


def test_crop_rim_keeps_the_rim_at_the_same_place_and_size():
    img = np.zeros((720, 1280, 3), np.uint8)
    img[300:308, 600:640] = 255  # the rim
    a = crop_rim(img, RIM)
    big = np.zeros((1080, 1920, 3), np.uint8)
    big[450:462, 900:960] = 255  # the same rim seen 1.5 times as large
    b = crop_rim(big, BBox(900.0, 450.0, 960.0, 462.0))
    assert a.shape == b.shape == (96, 96, 3)
    ya, xa = np.nonzero(a[..., 0] > 128)
    yb, xb = np.nonzero(b[..., 0] > 128)
    assert abs(xa.mean() - xb.mean()) < 1.0 and abs(ya.mean() - yb.mean()) < 1.0 and abs(len(xa) - len(xb)) <= 12


def series(*spots: tuple[int, str, float]) -> np.ndarray:
    s = np.zeros((73, len(ZONES)), np.float32)
    for k, zone, score in spots:
        s[k, ZI[zone]] = score
    return s


def test_zone_features_tell_a_ball_through_the_net_from_one_off_the_rim():
    made = mk.zone_features(series((26, "above", 0.9), (30, "ring", 0.8), (33, "net", 0.9), (37, "under", 0.7)))
    missed = mk.zone_features(series((26, "above", 0.9), (30, "ring", 0.8), (34, "side", 0.8), (40, "high", 0.6)))
    assert made["zone_net"] == pytest.approx(0.9) and made["zone_top_then_net"] == pytest.approx(0.9)
    assert made["zone_net_then_under"] == pytest.approx(0.7) and made["zone_top_then_out"] == 0.0
    assert missed["zone_net"] == 0.0 and missed["zone_top_then_out"] == pytest.approx(0.8)
    assert missed["zone_post_side"] == pytest.approx(0.8)
    assert set(mk.zone_features(None)) == set(mk.ZONE_FEATURES) and not any(mk.zone_features(None).values())
    assert mk.zone_features(np.full((73, len(ZONES)), -1.0, np.float32))["zone_valid"] == 0.0


# ---------------------------------------------------------------------------- the engine's verdict path


def missed_scene():
    s = two_team_scene()
    s.add("home-1", 18.0, 30.0).add("home-2", 30.0, 38.0).add("away-3", 9.0, 25.0)
    s.start("home-2").hold("home-2", 0.5).pass_to("home-1").hold("home-1", 0.6)
    s.shoot("left", made=False)
    s.fly(lambda pos: s.hands("away-3", pos), 0.6, apex=3.0).hold("away-3", 0.6)
    return s


@pytest.mark.parametrize("mode", MODES)
def test_a_judged_make_flips_the_miss_and_retracts_its_rebound(mode):
    s = missed_scene()
    s.hold("away-3", 2.0)
    eng = run_engine(s.frames(mode))
    miss = one(eng.events, EventKind.SHOT_MISSED)
    out = eng.judge_shot(miss.shot_id, True, 0.93, miss.t + 2.5)
    made = one(eng.events, EventKind.SHOT_MADE)
    assert made.shot_id == miss.shot_id and made.attrs["flipped_by"] == "vision" and made.attrs["vision_p_made"] == 0.93
    assert made.attrs["points"] == 2 and not live(eng.events, EventKind.SHOT_MISSED)
    assert all(e.status is EventStatus.RETRACTED for e in eng.events if e.kind is EventKind.REBOUND)
    ast = one(eng.events, EventKind.ASSIST)
    assert (ast.player_id, ast.other_player_id) == ("home-2", "home-1")
    assert {e.kind for e in out} >= {EventKind.SHOT_MADE, EventKind.REBOUND}
    assert eng.judge_shot(miss.shot_id, False, 0.1, miss.t + 3.0) == []  # judged once


def test_an_agreeing_verdict_keeps_the_call_and_notes_its_probability():
    s = missed_scene()
    s.hold("away-3", 2.0)
    eng = run_engine(s.frames("3d"))
    miss = one(eng.events, EventKind.SHOT_MISSED)
    eng.judge_shot(miss.shot_id, False, 0.12, miss.t + 2.5)
    again = one(eng.events, EventKind.SHOT_MISSED)
    assert again.attrs["judged_by"] == "vision" and again.attrs["vision_p_made"] == 0.12
    assert one(eng.events, EventKind.REBOUND).status is not EventStatus.RETRACTED
    assert eng.judge_shot("no-such-shot", True, 0.9, 10.0) == []


@pytest.mark.parametrize("reconcile", [True, False])
def test_score_reconcile_off_leaves_the_shots_to_the_camera(reconcile):
    s = missed_scene()
    s.set_score(2, 0)  # the scoreboard says it went in
    s.hold("away-3", 2.0)
    eng = run_engine(s.frames("3d"), EventEngineConfig(score_reconcile=reconcile))
    if reconcile:
        assert one(eng.events, EventKind.SHOT_MADE).attrs["flipped_by"] == "scoreboard"
    else:
        assert one(eng.events, EventKind.SHOT_MISSED).attrs.get("flipped_by") is None
        assert not live(eng.events, EventKind.SHOT_MADE) and not live(eng.events, EventKind.SCORE_CORRECTION)


# ---------------------------------------------------------------------------- the judge


def frame(t: float, balls: list[list[float]], ball_xy: tuple[float, float] | None = None) -> FrameState:
    ball = BallState(True, image_xy=ball_xy) if ball_xy is not None else None
    return FrameState(round(t * 30), t, [], ball, [RimObservation("left", RIM)], None, (1280, 720),
                      balls=np.array(balls, np.float32).reshape(-1, 3))


def result(kind: EventKind, t: float, crossed: bool) -> Event:
    return Event("e1", kind, t, round(t * 30), team=Team.HOME, value=2, shot_id="s1", source="cv",
                 attrs={"cues": {"crossed_inside": crossed, "rim_contact": False, "crossing_margin_px": 5.0}})


def toy_model() -> mk.Logistic:
    """P(made) rises with the ball seen in the net after the arrival and falls with it beside the ring."""
    w = {n: 0.0 for n in JUDGE_FEATURES}
    w.update(zone_net=6.0, zone_post_side=-6.0)
    return mk.Logistic(JUDGE_FEATURES, tuple(w[n] for n in JUDGE_FEATURES), -2.0)


def drive(judge: ShotJudge, path: dict[int, tuple[float, float]], t_result: float, kind: EventKind) -> list:
    out = []
    for k in range(150):
        t = k / 30.0
        xy = path.get(k)
        judge.observe(frame(t, [[xy[0], xy[1], 0.9]] if xy is not None else [], xy))
        if abs(t - t_result) < 1e-6:
            judge.on_events([result(kind, t_result, crossed=False)])
        out += judge.due(t)
    return out


def test_the_judge_overturns_a_miss_the_detector_saw_drop_through_the_net():
    path = {k: tuple(at(0.0, -2.0 + 0.1 * (k - 40))[:2]) for k in range(40, 76)}  # down through the ring and net
    verdicts = drive(ShotJudge(toy_model()), path, 60 / 30.0, EventKind.SHOT_MISSED)
    assert len(verdicts) == 1 and verdicts[0][0] == "s1" and verdicts[0][1] is True and verdicts[0][2] > 0.9


def test_the_judge_keeps_a_miss_that_bounced_off_to_the_side():
    path = {k: tuple(at(0.0, -2.0 + 0.1 * (k - 40))[:2]) for k in range(40, 60)}  # down to the ring ...
    path.update({k: tuple(at(0.9 + 0.05 * (k - 60), -0.2)[:2]) for k in range(60, 80)})  # ... and off to the side
    verdicts = drive(ShotJudge(toy_model()), path, 60 / 30.0, EventKind.SHOT_MISSED)
    assert len(verdicts) == 1 and verdicts[0][1] is False and verdicts[0][2] < 0.2


def test_the_judge_waits_for_its_delay_and_judges_each_shot_once():
    judge = ShotJudge(toy_model(), JudgeConfig(delay_s=2.5))
    for k in range(70):
        judge.observe(frame(k / 30.0, []))
    judge.on_events([result(EventKind.SHOT_MADE, 1.0, crossed=True)])
    assert judge.due(3.4) == []
    assert len(judge.due(3.6)) == 1 and judge.due(10.0) == []
    judge.on_events([result(EventKind.SHOT_MADE, 1.0, crossed=True)])  # a later revision of the same result
    assert judge.due(20.0) == []


def test_judge_from_config_reads_the_fitted_model(tmp_path):
    path = tmp_path / "judge.json"
    path.write_text(json.dumps({"made": toy_model().to_dict(), "min_score": 0.2, "fitted_on": ["g"]}))
    judge = judge_from_config(path, 59.94)
    assert judge is not None and judge.config.min_score == 0.2 and judge.fps == pytest.approx(59.94)
    assert judge_from_config(None, 30.0) is None and judge_from_config(tmp_path / "missing.json", 30.0) is None


# ---------------------------------------------------------------------------- the possession analysis of an analysis ahead


def test_the_recorded_camera_reads_back_as_a_camera_track(tmp_path):
    from courtvision.nba.analyze import CameraTrack
    from courtvision.nba.auto_analysis import Recorder
    from courtvision.perception.court_search import CameraParams

    rec = Recorder(tmp_path / "possession")
    center = np.array([47.0, -90.0, 35.0])
    for k in range(12):
        p = CameraParams(center, 1.0 + 0.1 * k, 20.0, 0.0, 1500.0) if k != 5 else None
        rec.frame(k, k / 29.97, center, (1280, 720), 29.97, 2, "tracking" if p is not None else "lost", p,
                  np.array([[600.0, 300.0, 0.9]], np.float32) if k % 3 == 0 else None)
    rec.close()
    cam = CameraTrack.load(sorted((tmp_path / "possession" / "camera_parts").glob("part*.jsonl")))
    assert cam.image_size == (1280, 720) and cam.fps == pytest.approx(29.97) and np.allclose(cam.center, center)
    assert cam.get(5) is not None  # a one-frame gap is filled, as for a tracked game
    assert sorted(rec.balls) == [0, 3, 6, 9] and rec.balls[3].shape == (1, 3) and rec.usable


def test_build_without_a_tracked_camera_gives_no_analysis(tmp_path):
    from courtvision.nba.auto_analysis import Recorder, build

    rec = Recorder(tmp_path / "p")
    rec.close()
    assert not rec.usable and build(tmp_path / "p", [], {}, "video.mp4") is None


def test_the_judge_runs_its_rim_model_on_the_rim_crops_about_the_arrival(monkeypatch):
    from courtvision.nba import rim_model

    clips = []

    def fake_predict(net, frames, cfg=None, flip=True, batch=16):
        clips.append(frames)
        return np.array([[0.6, 0.2, 0.2]], np.float32)

    monkeypatch.setattr(rim_model, "predict", fake_predict)
    judge = ShotJudge(toy_model(), rim_model=("net", "cfg"))
    crop = np.full((96, 96, 3), 7, np.uint8)
    for k in range(150):
        f = frame(k / 30.0, [])
        f.rim_crops = {"left": crop} if k >= 45 else None  # the rim cut out from 1.5 s on
        judge.observe(f)
    feats = judge.features(result(EventKind.SHOT_MISSED, 2.0, crossed=False))
    assert feats["cnn_made"] == pytest.approx(0.75) and feats["cnn_missing"] == 0.0
    assert clips[0].shape == (1, 37, 96, 96, 3) and clips[0][0, 0].max() == 0 and clips[0][0, -1].max() == 7
    plain = ShotJudge(toy_model())  # no rim model: the features say so
    for k in range(150):
        plain.observe(frame(k / 30.0, []))
    assert plain.features(result(EventKind.SHOT_MISSED, 2.0, crossed=False))["cnn_missing"] == 1.0
