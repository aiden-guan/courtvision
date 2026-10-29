"""NBA broadcast analysis passes (nba/analyze.py) and the box-score comparison (nba/evaluate.py)."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from courtvision.nba import analyze
from courtvision.nba.analyze import (
    REF_FPS,
    BallLinkConfig,
    BallTrack,
    CameraTrack,
    DetectionCache,
    _segments,
    _smooth_runs,
    _warp,
    build_ball_track,
    flight_config_for,
    is_referee,
    iter_frames,
    link_ball,
    perceive,
    perception_rate_overrides,
    probe_video,
    rate_ratio,
    rectify_box,
    scale_frames,
)
from courtvision.nba.espn import TeamBox
from courtvision.nba.evaluate import BOX_STATS, box_error
from courtvision.perception import student
from courtvision.perception.ball3d import FlightSearchConfig
from courtvision.perception.court_search import CameraParams
from courtvision.types import Team

C = np.array([47.0, -94.0, 36.0])
SIZE = (1920, 1080)


def track(params: dict[int, tuple[float, float, float, float]]) -> CameraTrack:
    return CameraTrack(center=C, image_size=SIZE, fps=30.0, params=params)


def jersey(bgr: tuple[int, int, int], stripes: bool = False) -> np.ndarray:
    """A frame filled with one shirt colour, optionally with black vertical stripes over a third of it."""
    img = np.full((300, 200, 3), bgr, np.uint8)
    if stripes:
        img[:, np.arange(200) % 9 < 3] = 0
    return img


# --------------------------------------------------------------------------- camera track


def test_smooth_runs_fill_gaps_of_up_to_max_fill_frames_and_keep_longer_gaps_empty() -> None:
    frames = [*range(0, 10), *range(13, 21), *range(25, 30)]  # 3 frames missing at 10-12, 4 at 21-24
    raw = {f: (0.5 * f, 15.0, 0.0, 2000.0 if f < 10 else 4000.0) for f in frames}
    out = _smooth_runs(raw, k=1, max_fill=3)
    assert sorted(out) == [*range(0, 21), *range(25, 30)]
    pan, tilt, roll, focal = out[11]
    assert pan == pytest.approx(5.5) and tilt == pytest.approx(15.0) and roll == 0.0
    assert focal == pytest.approx(math.sqrt(2000.0 * 4000.0))  # the zoom is interpolated in log focal length
    assert sorted(_smooth_runs(raw, k=1, max_fill=2)) == frames


def test_smooth_runs_drop_a_bad_fit_within_a_run_but_do_not_smooth_across_a_long_gap() -> None:
    raw = {f: (5.0 if f == 5 else 0.0, 15.0, 0.0, 3000.0) for f in range(11)}  # one bad fit
    raw.update({f: (10.0, 15.0, 0.0, 3000.0) for f in range(20, 31)})  # the next run, after a cut
    out = _smooth_runs(raw, k=5, max_fill=3)
    assert all(out[f][0] == pytest.approx(0.0) for f in range(11))  # the median drops it; a mean would smear it
    assert all(out[f][0] == pytest.approx(10.0) for f in range(20, 31))
    assert out[0][3] == pytest.approx(3000.0)


def test_smooth_runs_keep_a_steady_pan_and_average_its_jitter() -> None:
    rng = np.random.default_rng(0)
    raw = {f: (0.4 * f + rng.normal(0.0, 0.05), 15.0, 0.0, 3000.0) for f in range(40)}
    out = _smooth_runs(raw, k=5, max_fill=3)
    err = [out[f][0] - 0.4 * f for f in range(4, 36)]  # away from the ends (held flat by the padding)
    raw_err = [raw[f][0] - 0.4 * f for f in range(4, 36)]
    assert abs(float(np.mean(err))) < 0.02  # no lag or bias on a steady turn
    assert float(np.std(err)) < 0.7 * float(np.std(raw_err))


def test_frame_to_frame_maps_a_point_to_the_same_world_direction_in_the_other_frame() -> None:
    cam = track({0: (5.0, 16.0, 0.4, 2600.0), 1: (9.0, 18.5, -0.3, 3400.0)})  # a turn, a tilt, a roll and a zoom
    H = cam.frame_to_frame(0, 1)
    assert H is not None
    for X in ((60.0, 20.0, 0.0), (52.0, 30.0, 10.0), (40.0, 10.0, 6.5)):  # on the floor, at rim height, a head
        a, b = (cam.get(f).camera(SIZE).project(np.array(X)) for f in (0, 1))  # type: ignore[union-attr]
        h = H @ np.array([a[0], a[1], 1.0])
        assert np.allclose(h[:2] / h[2], b, atol=1e-6)
    back = cam.frame_to_frame(1, 0)
    assert back is not None and np.allclose(back @ H / (back @ H)[2, 2], np.eye(3), atol=1e-9)
    assert cam.frame_to_frame(0, 7) is None  # no camera in frame 7


# --------------------------------------------------------------------------- detections and the ball path


def test_detection_cache_save_and_load_round_trip(tmp_path: Path) -> None:
    balls = {3: np.array([[100.0, 200.0, 0.9], [50.0, 60.0, 0.2]], np.float32), 4: np.zeros((0, 3), np.float32),
             9: np.array([[1.5, 2.5, 0.5]], np.float32)}
    persons = {3: np.array([[10.0, 20.0, 30.0, 80.0, 0.8]], np.float32),
               4: np.array([[1.0, 2.0, 3.0, 4.0, 0.5], [5.0, 6.0, 7.0, 8.0, 0.4]], np.float32)}  # none in frame 9
    path = tmp_path / "detections.npz"
    DetectionCache(balls, persons).save(path)
    back = DetectionCache.load(path)
    assert sorted(back.balls) == sorted(back.persons) == [3, 4, 9]
    for f in (3, 4, 9):
        np.testing.assert_array_equal(back.balls[f], balls[f])
    for f in (3, 4):
        np.testing.assert_array_equal(back.persons[f], persons[f])
    assert back.balls[4].shape == (0, 3) and back.persons[9].shape == (0, 5)


def test_detection_cache_load_keeps_one_copy_of_each_array(tmp_path: Path) -> None:
    # frames are views of one loaded array; a copy per frame took 46 GB on the first quarter of a game
    balls = {f: np.full((2, 3), f, np.float32) for f in range(20)}
    persons = {f: np.full((3, 5), f, np.float32) for f in range(20)}
    path = tmp_path / "detections.npz"
    DetectionCache(balls, persons).save(path)
    back = DetectionCache.load(path)
    assert len({id(a.base) for a in back.balls.values()}) == 1
    assert len({id(a.base) for a in back.persons.values()}) == 1
    np.testing.assert_array_equal(back.balls[19], balls[19])


def test_link_ball_follows_a_moving_ball_rather_than_a_static_distractor() -> None:
    cam = track({f: (0.0, 15.0, 0.0, 3000.0) for f in range(100, 115)})  # a still camera
    ball: dict[int, tuple[float, float, float]] = {}
    cache = DetectionCache()
    for f in range(100, 115):
        k = f - 100
        ball[f] = (400.0 + 40.0 * k, 600.0 + 8.0 * k, 0.3 if k in (6, 7) else 0.6)  # its score dips in two frames
        # a static ball-like thing scoring 0.5, and a candidate below the minimum score
        cache.balls[f] = np.array([[1500.0, 200.0, 0.5], ball[f], [900.0, 900.0, 0.05]], np.float32)
    path = link_ball(cache, cam)
    assert sorted(path) == list(range(100, 115))
    assert all(np.allclose(path[f], ball[f]) for f in path)


def test_link_ball_keeps_the_chain_through_a_fast_camera_pan() -> None:
    params = {f: (-12.0 + 2.5 * f, 18.0, 0.3, 3000.0) for f in range(12)}  # the camera turns 2.5 degrees a frame
    cam = track(params)
    X = np.array([50.0, 22.0, 4.0])  # the ball, held still
    uv = {f: cam.get(f).camera(SIZE).project(X) for f in params}  # type: ignore[union-attr]
    # in the image the ball jumps further per frame than a ball may move
    assert min(np.linalg.norm(uv[f + 1] - uv[f]) for f in range(11)) > BallLinkConfig().max_step_px
    logo = (300.0, 950.0)  # a ball-like graphic fixed in the image, scoring higher than the ball
    cache = DetectionCache({f: np.array([[*logo, 0.55], [*uv[f], 0.5]], np.float32) for f in params})
    path = link_ball(cache, cam)
    assert sorted(path) == list(range(12))
    assert all(np.allclose(path[f][:2], uv[f], atol=0.01) for f in path)
    # told the camera stood still, the same candidates link the logo: removing the pan is what keeps the ball
    still = link_ball(cache, track({f: params[0] for f in params}))
    assert sorted(still) == list(range(12)) and all(np.allclose(still[f][:2], logo) for f in still)


# --------------------------------------------------------------------------- person boxes


def test_rectify_box_is_none_when_the_box_centre_ray_does_not_come_down() -> None:
    cam = CameraParams(C, 0.0, 3.0, 0.0, 3000.0)  # nearly level: the horizon is at v = 540 - 3000 tan 3 deg = 383
    assert rectify_box((900.0, 150.0, 960.0, 250.0), cam, SIZE) is None  # someone in the stands, above the horizon
    assert rectify_box((900.0, 760.0, 960.0, 840.0), cam, SIZE) is not None


def test_is_referee_on_synthetic_torsos() -> None:
    box = (40.0, 20.0, 160.0, 280.0)
    assert is_referee(jersey((128, 128, 128), stripes=True), box)  # grey with black stripes
    assert not is_referee(jersey((255, 255, 255)), box)  # a white jersey
    assert not is_referee(jersey((150, 30, 10), stripes=True), box)  # saturated blue, even with dark stripes
    assert not is_referee(jersey((128, 128, 128)), box)  # plain grey: no dark stripes


# --------------------------------------------------------------------------- box score


def test_box_error_totals_and_exact_share_on_a_tiny_box() -> None:
    truth = {Team.HOME: TeamBox(Team.HOME, "HOM", {"pts": 10, "fgm": 4, "fga": 8, "ftm": 2, "fta": 2}),
             Team.AWAY: TeamBox(Team.AWAY, "AWY", {"pts": 6, "fgm": 3, "fga": 7})}  # 42 official counts in all
    pred = {Team.HOME: {"pts": 12, "fgm": 5, "fga": 8, "ftm": 2, "fta": 2},
            Team.AWAY: {"pts": 6, "fgm": 3, "fga": 5, "oreb": 1}}  # off by +2, +1, -2 and +1
    cmp = box_error(pred, truth)
    assert cmp.error == round(6 / 42, 4)
    assert len(cmp.rows) == 2 * len(BOX_STATS) and cmp.exact_share == round(22 / 26, 4)
    rows = {(r["team"], r["stat"]): r for r in cmp.rows}
    assert rows[("home", "pts")] == {"team": "home", "stat": "pts", "pred": 12, "truth": 10, "diff": 2}
    assert rows[("away", "fga")]["diff"] == -2 and rows[("away", "oreb")]["diff"] == 1
    assert rows[("away", "blk")] == {"team": "away", "stat": "blk", "pred": 0, "truth": 0, "diff": 0}


# --------------------------------------------------------------------------- other frame rates and frame sizes
#
# The settings were tuned at 29.97 fps on 1920x1080 frames: there the code must give exactly what it gave before
# (``_link_ball_063e638``: link_ball as of commit 063e638, before rates and sizes were converted); at other rates and
# sizes the same physical scene must come out the same.

NTSC = 30000 / 1001  # what OpenCV reports for a 29.97 fps file


def _link_ball_063e638(cache: DetectionCache, camera: CameraTrack, config: BallLinkConfig | None = None) -> dict:
    """``analyze.link_ball`` of commit 063e638, verbatim (settings in frames and px whatever the video)."""
    cfg = config or BallLinkConfig()
    path: dict[int, tuple[float, float, float]] = {}
    for seg in _segments(sorted(f for f in cache.balls if camera.get(f) is not None)):
        T = len(seg)
        cands = []
        for f in seg:
            b = cache.balls[f]
            b = b[b[:, 2] >= cfg.min_score][: cfg.max_candidates]
            cands.append(b)
        V: list[np.ndarray] = []
        back: list[list[tuple[str, tuple[int, int] | None]]] = []
        prefix: list[tuple[float, tuple[int, int] | None]] = [(0.0, None)]
        for t in range(T):
            un = cands[t][:, 2] - cfg.bias
            base, base_node = prefix[t]
            best = base - cfg.restart_cost + un
            ptr: list[tuple[str, tuple[int, int] | None]] = [("restart", base_node)] * len(un)
            for g in range(1, cfg.max_gap + 1):
                tp = t - g
                if tp < 0 or len(un) == 0:
                    break
                if len(cands[tp]) == 0:
                    continue
                M = camera.frame_to_frame(seg[tp], seg[t])
                prev = _warp(M, cands[tp][:, :2].astype(float)) if M is not None else cands[tp][:, :2]
                D = np.linalg.norm(cands[t][:, None, :2] - prev[None, :, :], axis=2) / g
                val = V[tp][None, :] + un[:, None] - cfg.gap_cost * (g - 1) - cfg.speed_cost * np.maximum(0.0, D - cfg.free_step_px)
                val[D > cfg.max_step_px] = -np.inf
                k = np.argmax(val, axis=1)
                v = val[np.arange(len(un)), k]
                for i in np.flatnonzero(v > best):
                    best[i] = v[i]
                    ptr[i] = ("chain", (tp, int(k[i])))
            V.append(best)
            back.append(ptr)
            bv, bn = prefix[t]
            if len(best):
                i = int(np.argmax(best))
                if best[i] > bv:
                    bv, bn = float(best[i]), (t, i)
            prefix.append((bv, bn))
        node = prefix[T][1]
        while node is not None:
            t, k = node
            x, y, s = cands[t][k]
            path[seg[t]] = (float(x), float(y), float(s))
            node = back[t][k][1]
    return path


def random_detections(seed: int, fps: float, n: int = 110) -> tuple[CameraTrack, DetectionCache]:
    """A turning, tilting, zooming camera with cuts, a wandering ball missed now and then and up to four other
    candidates a frame: random input for the linker."""
    rng = np.random.default_rng(seed)
    params, pan = {}, -12.0
    for f in range(n):
        pan += rng.normal(0.25, 0.3)
        if 60 <= f < 64 or rng.random() < 0.03:
            continue
        params[f] = (pan, 17.0 + rng.normal(0, 0.05), rng.normal(0, 0.02), 3000.0 * math.exp(rng.normal(0, 0.01)))
    cache = DetectionCache()
    xy, v = np.array([800.0, 600.0]), np.array([20.0, -5.0])
    for f in range(n):
        v = np.clip(v + rng.normal(0, 6, 2), -80, 80)
        xy = np.clip(xy + v, 0, [1919, 1079])
        rows = [(*xy, rng.uniform(0.05, 0.95))] if rng.random() < 0.8 else []
        rows += [(rng.uniform(0, 1920), rng.uniform(0, 1080), rng.uniform(0.0, 0.9)) for _ in range(rng.integers(0, 5))]
        rows.sort(key=lambda r: -r[2])
        cache.balls[f] = np.array(rows, np.float32).reshape(-1, 3)
    return CameraTrack(C, SIZE, fps, params), cache


def test_rate_ratio_is_exactly_one_at_29_97_and_30_fps_and_scales_frame_counts_elsewhere() -> None:
    for fps in (29.97, NTSC, 30.0, 29.9, 30.1):
        assert rate_ratio(fps) == 1.0
    for fps in (0.0, -1.0, float("nan"), float("inf")):
        assert rate_ratio(fps) == 1.0  # unknown: the tuned settings
    assert rate_ratio(59.94) == 2.0 and rate_ratio(50.0) == pytest.approx(1.668, abs=1e-3)
    assert rate_ratio(25.0) == pytest.approx(0.834, abs=1e-3)
    assert [scale_frames(6, rate_ratio(f)) for f in (23.976, 25.0, 29.97, 50.0, 59.94, 60.0, 119.88)] == \
        [5, 5, 6, 10, 12, 12, 24]
    assert scale_frames(2, 0.2) == 1 and scale_frames(2, 0.2, minimum=0) == 0


def test_at_29_97_fps_and_1080_lines_the_settings_are_the_tuned_ones() -> None:
    cfg = BallLinkConfig()
    for fps in (29.97, NTSC, 30.0):
        assert cfg.for_video(fps, 1080) is cfg
        assert flight_config_for(fps) is None  # find_flights' own defaults, as before
        assert perception_rate_overrides(fps) == {}
    custom = FlightSearchConfig(min_obs=5)
    assert flight_config_for(NTSC, custom) is custom
    assert student.history_step(NTSC) == 1 and student.history_step(30.0) == 1


def test_link_ball_at_29_97_fps_and_1080_lines_is_the_linker_of_commit_063e638() -> None:
    for seed, fps in ((0, 29.97), (1, NTSC), (2, NTSC), (3, 30.0)):
        cam, cache = random_detections(seed, fps)
        for cfg in (None, BallLinkConfig(max_gap=4, restart_cost=0.3, gap_cost=0.2)):
            new, old = link_ball(cache, cam, cfg), _link_ball_063e638(cache, cam, cfg)
            assert len(old) > 40 and new == old  # the same frames and the same float picks


def test_the_settings_follow_the_frame_rate_and_the_frame_height() -> None:
    fast = BallLinkConfig().for_video(59.94, 1080)
    assert (fast.max_gap, fast.max_step_px, fast.free_step_px, fast.restart_cost) == (12, 45.0, 15.0, 1.0)
    assert fast.speed_cost == pytest.approx(0.008) and fast.gap_cost == 0.1  # gap: per frame past one 29.97 frame
    for height in (720, 2160):
        s = height / 1080
        c = BallLinkConfig().for_video(29.97, height)
        assert (c.max_gap, c.max_step_px, c.free_step_px) == (6, pytest.approx(90.0 * s), pytest.approx(30.0 * s))
        assert c.speed_cost == pytest.approx(0.004 / s) and c.restart_cost == 0.5
    fl = flight_config_for(59.94)
    assert fl is not None and (fl.min_obs, fl.seed_obs, fl.max_skip) == (16, (20, 14), 4)
    assert fl.max_gap_s == FlightSearchConfig().max_gap_s  # already seconds


def broadcast_scene(seed: int, fps: float, height: int = 1080) -> tuple[CameraTrack, DetectionCache, dict]:
    """4 s of a panning, zooming broadcast camera: a dribble, a fast pass (2.0-2.6 s) and a hold; the ball is
    missed now and then and for 0.17 s from 1.2 s, the camera is lost from 3.2 s to 3.35 s; a floor logo is
    detected in every frame (lower scores) and false detections pop up. What is detected changes every 1/29.97 s,
    the same at any frame rate; ``objs[frame]`` names each candidate."""
    s = height / 1080
    size = (round(height * 16 / 9), height)
    rng = np.random.default_rng(seed)
    n_ref = 122
    vis, score = rng.random(n_ref) < 0.85, rng.uniform(0.45, 0.9, n_ref)
    fp, fp_sc = rng.random(n_ref) < 0.35, rng.uniform(0.05, 0.6, n_ref)
    fp_xy = rng.uniform([0, 0], [1920, 1080], (n_ref, 2))
    logo_sc = rng.uniform(0.15, 0.45, n_ref)
    noise = rng.normal(0, 1, (800, 2))  # px of a 1080-line frame, by time (200 Hz)

    def ball(t: float) -> np.ndarray:
        if t < 2.0:
            return np.array([25.0 + 4.0 * t, 18.0 + 1.0 * t, 1.0 + 2.0 * abs(math.sin(4.0 * t))])
        if t < 2.6:
            u = (t - 2.0) / 0.6
            return np.array([33.0 + 40.0 * u, 20.0 + 12.0 * u, 3.0 + 4.0 * u * (1 - u)])
        return np.array([73.0, 32.0, 4.0 + 0.3 * math.sin(5.0 * t)])

    params = {k: (-15.0 + 5.0 * (k / fps) + 2.0 * math.sin(2.0 * k / fps), 17.0 + 0.5 * math.sin(k / fps), 0.2,
                  3000.0 * s * (1 + 0.1 * math.sin(0.7 * k / fps))) for k in range(int(4.0 * fps))
              if not 3.2 <= k / fps < 3.35}
    track = CameraTrack(C, size, fps, params)
    cache, objs = DetectionCache(), {}
    for k in params:
        t = k / fps
        j = math.floor(t * REF_FPS + 1e-9)
        cam = track.get(k).camera(size)  # type: ignore[union-attr]
        rows: list[tuple[float, float, float]] = []
        ids: list[str] = []
        if vis[j] and not 1.2 <= t < 1.37:
            u, v = cam.project(ball(t))
            e = noise[int(t * 200)] * s
            rows.append((u + e[0], v + e[1], score[j]))
            ids.append("ball")
        u, v = cam.project(np.array([60.0, 40.0, 0.0]))
        rows.append((u, v, logo_sc[j]))
        ids.append("logo")
        if fp[j]:
            rows.append((fp_xy[j, 0] * s, fp_xy[j, 1] * s, fp_sc[j]))
            ids.append(f"fp{j}")
        order = np.argsort([-r[2] for r in rows], kind="stable")
        cache.balls[k] = np.array([rows[i] for i in order], np.float32).reshape(-1, 3)
        objs[k] = [ids[i] for i in order]
    return track, cache, objs


def on_the_ball(path: dict, cache: DetectionCache, objs: dict) -> set[int]:
    """Frames whose linked point is the ball."""
    out = set()
    for f, (x, y, _) in path.items():
        b = cache.balls[f]
        if objs[f][int(np.argmin(np.hypot(b[:, 0] - x, b[:, 1] - y)))] == "ball":
            out.add(f)
    return out


def test_the_same_ball_is_linked_at_29_97_and_59_94_fps_and_at_720_1080_and_2160_lines() -> None:
    for seed in range(3):
        paths = {}
        for fps in (REF_FPS, 59.94):
            cam, cache, objs = broadcast_scene(seed, fps)
            paths[fps] = link_ball(cache, cam)
            seen = {f for f in cache.balls if "ball" in objs[f]}
            assert on_the_ball(paths[fps], cache, objs) == seen  # wherever it was detected, nowhere else
            if fps == REF_FPS:
                assert paths[fps] == _link_ball_063e638(cache, cam)
        base = paths[REF_FPS]
        for height in (720, 2160) if seed < 2 else ():
            cam_s, cache_s, objs_s = broadcast_scene(seed, REF_FPS, height)
            path = link_ball(cache_s, cam_s)
            assert sorted(path) == sorted(base)
            assert all(np.allclose(path[f][:2], np.array(base[f][:2]) * height / 1080, atol=0.01) for f in path)
            if height == 2160:  # 90 px a frame was 45 px of a 1080-line frame: the old linker lost the fast pass
                old = on_the_ball(_link_ball_063e638(cache_s, cam_s), cache_s, objs_s)
                pass_frames = {f for f in cache_s.balls if "ball" in objs_s[f] and 2.0 <= f / REF_FPS < 2.6}
                assert len(pass_frames - old) >= 5


def hidden_ball(fps: float) -> tuple[CameraTrack, DetectionCache]:
    """A still camera; the ball (scored 0.6) crosses it at 300 px/s, hidden for 0.1 s from 0.667 s; in the middle
    1/29.97 s of that something ball-like scores 0.55 900 px away."""
    n = int(1.5 * fps)
    cache = DetectionCache()
    for f in range(n):
        t = f / fps
        rows = [] if 0.667 <= t < 0.767 else [(400.0 + 300.0 * t, 600.0, 0.6)]
        if 0.7 <= t < 0.7334:
            rows.append((1300.0, 300.0, 0.55))
        cache.balls[f] = np.array(rows, np.float32).reshape(-1, 3)
    return CameraTrack(C, SIZE, fps, {f: (0.0, 15.0, 0.0, 3000.0) for f in range(n)}), cache


def test_a_false_detection_while_the_ball_is_hidden_stays_out_at_any_frame_rate() -> None:
    for fps in (REF_FPS, 59.94):
        cam, cache = hidden_ball(fps)
        path = link_ball(cache, cam)
        assert all(x < 1000.0 for x, _, _ in path.values())  # the ball's chain bridges the gap
        assert len(path) == sum(1 for b in cache.balls.values() if len(b) and b[:, 0].min() < 1000.0)
    # with frame counts and costs of 29.97 fps at 59.94 the gap is 7 frames, too long to bridge, and the cheap
    # restart takes the false detection
    cam, cache = hidden_ball(59.94)
    assert any(x > 1000.0 for x, _, _ in _link_ball_063e638(cache, cam).values())


def test_the_runs_fitted_in_3d_span_the_same_time_at_both_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[np.ndarray, Any]] = []

    def record(t: np.ndarray, uv: np.ndarray, cams: list, size: tuple, frames: Any = None, config: Any = None,
               **kw: Any) -> list:
        calls.append((t, config))
        return []

    monkeypatch.setattr(analyze, "find_flights", record)
    for fps in (REF_FPS, 59.94):
        n = int(2.0 * fps)
        # the ball is missed for 0.15 s from 1.0 s: 5 frames at 29.97 fps, 9 at 59.94 (more than 6)
        balls = {f: np.array([[500.0 + 200.0 * f / fps, 500.0, 0.7]] if not 1.0 <= f / fps < 1.15 else
                             np.zeros((0, 3)), np.float32) for f in range(n)}
        cam = CameraTrack(C, SIZE, fps, {f: (0.0, 15.0, 0.0, 3000.0) for f in range(n)})
        calls.clear()
        build_ball_track(DetectionCache(balls), cam)
        assert len(calls) == 1  # one run across the gap
        t, config = calls[0]
        assert t[0] == 0.0 and t[-1] == pytest.approx((n - 1) / fps)
        assert config == flight_config_for(fps)


def camera_parts(path: Path, fps: float, rows: dict[int, tuple[float, float, float, float]],
                 times: dict[int, float] | None = None, vfr: bool = False) -> list[Path]:
    """A camera part as tools/nba_track_court.py writes it."""
    head = {"header": True, "center": C.tolist(), "image_size": list(SIZE), "fps": fps, "stride": 1}
    if vfr:
        head["vfr"] = True
    lines = [head] + [{"frame": f, "t": (times or {}).get(f, round(f / fps, 4)), "status": "tracking",
                       "params": {"pan": q[0], "tilt": q[1], "roll": q[2], "f": q[3]}} for f, q in sorted(rows.items())]
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    return [path]


def pan_fits(fps: float, burst: tuple[float, float] = (1.0, 1.067), gap: tuple[float, float] = (2.0, 2.1),
             seed: int = 0) -> dict[int, tuple[float, float, float, float]]:
    """3 s of a turning, zooming camera's per-frame fits: a little jitter, two bad fits at 29.97 fps (67 ms of them)
    and 0.1 s without a fit."""
    rng = np.random.default_rng(seed)
    out = {}
    for f in range(int(3.0 * fps)):
        t = f / fps
        if gap[0] <= t < gap[1]:
            continue
        pan = -10.0 + 8.0 * t + 3.0 * math.sin(3.0 * t) + rng.normal(0, 0.02)
        pan += 3.0 if burst[0] <= t < burst[1] else 0.0
        out[f] = (pan, 17.0 + 0.3 * math.sin(2.0 * t), 0.1, 3000.0 * (1 + 0.2 * math.sin(t)))
    return out


def test_camera_load_at_29_97_fps_smooths_as_before(tmp_path: Path) -> None:
    for fps in (29.97, NTSC, 30.0):
        raw = pan_fits(fps)
        track = CameraTrack.load(camera_parts(tmp_path / "part0.jsonl", fps, raw))
        assert track.params == _smooth_runs(raw, 5, 3) and track.times is None
        assert all(track.time(f) == f / fps for f in (0, 1, 77, 89))


def test_camera_smoothing_and_gap_filling_keep_their_times_at_59_94_fps(tmp_path: Path) -> None:
    slow = CameraTrack.load(camera_parts(tmp_path / "a.jsonl", REF_FPS, pan_fits(REF_FPS)))
    fast = CameraTrack.load(camera_parts(tmp_path / "b.jsonl", 59.94, pan_fits(59.94, seed=1)))
    old = CameraTrack.load(camera_parts(tmp_path / "c.jsonl", 59.94, pan_fits(59.94, seed=1)), smooth=5, max_fill=3)

    def truth(t: float) -> float:
        return -10.0 + 8.0 * t + 3.0 * math.sin(3.0 * t)

    for f in range(5, 85):  # away from the ends, which the padding holds flat
        assert slow.params[f][0] == pytest.approx(truth(f / REF_FPS), abs=0.06)
        assert fast.params[2 * f][0] == pytest.approx(slow.params[f][0], abs=0.06)  # the bad fits are dropped,
    gap = [f for f in range(int(3.0 * 59.94)) if 2.0 <= f / 59.94 < 2.1]  # the gap is filled at both rates
    assert all(f in fast.params for f in gap) and all(f in slow.params for f in range(60, 63))
    # 5 frames and 3 frames at 59.94 fps: four bad fits in five survive the median, six missing frames cut the run
    assert max(abs(old.params[f][0] - truth(f / 59.94)) for f in range(60, 64)) > 1.0
    assert not any(f in old.params for f in gap)


def test_a_variable_frame_rate_track_takes_its_times_from_the_decoder(tmp_path: Path) -> None:
    # 0.2 s at 60 fps, then 30 fps: the tracker writes each frame's decoder time, every other frame (--stride 2)
    times = {f: f / 60.0 if f < 12 else 0.2 + (f - 12) / 30.0 for f in range(0, 40, 2)}
    raw = {f: (0.0, 15.0, 0.0, 3000.0) for f in times}
    track = CameraTrack.load(camera_parts(tmp_path / "p.jsonl", 40.0, raw, times, vfr=True))
    assert track.vfr and all(track.time(f) == pytest.approx(t) for f, t in times.items())
    assert track.time(13) == pytest.approx(0.2 + 1 / 30.0) and track.time(41) == pytest.approx(times[38] + 3 / 40.0)
    assert [track.frame_at(times[f]) for f in times] == list(times)
    assert track.frame_at(0.2 + 3.1 / 30.0) == 15 and track.frame_at(-1.0) == -40
    cfr = CameraTrack.load(camera_parts(tmp_path / "q.jsonl", 40.0, raw, times))  # no "vfr" in the header
    assert not cfr.vfr and cfr.time(13) == 13 / 40.0 and cfr.frame_at(0.5) == 20


def numbered_video(path: Path, n: int, fps: float, size: tuple[int, int] = (320, 180)) -> Path:
    """Frames whose top row of 10 blocks spells their index (MJPG)."""
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n):
        img = np.full((size[1], size[0], 3), 90, np.uint8)
        for k in range(10):
            img[0:20, 20 * k:20 * k + 20] = 255 if (i >> k) & 1 else 0
        img[60:120, (7 * i) % (size[0] - 40):(7 * i) % (size[0] - 40) + 40] = (40, 90, 200)
        wr.write(img)
    wr.release()
    return path


def frame_number(img: np.ndarray) -> int:
    return sum(1 << k for k in range(10) if img[10, 20 * k + 10].mean() > 128)


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="needs ffmpeg and ffprobe")


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


@needs_ffmpeg
def test_a_variable_frame_rate_file_is_flagged_and_read_without_seeking(tmp_path: Path) -> None:
    src = numbered_video(tmp_path / "src.avi", 300, 60.0)
    # a screen capture's frames: every other one to 1 s, all to 2 s, then every other one with 0.5 s missing
    keep = "if(lt(n,60),not(mod(n,2)),if(lt(n,120),1,if(between(n,200,229),0,not(mod(n,2)))))"
    vfr = tmp_path / "vfr.mp4"
    ffmpeg("-i", str(src), "-vf", f"select='{keep}'", "-fps_mode", "vfr", "-c:v", "libx264", "-qp", "0", "-g", "30",
           "-pix_fmt", "yuv420p", str(vfr))
    info = probe_video(vfr)
    assert info.vfr and any("variable frame rate" in w for w in info.warnings)
    assert info.times is not None and len(info.times) == info.n_frames == 165
    decoded = [frame_number(img) for _, img in iter_frames(vfr)]
    assert len(decoded) == 165 and decoded[:3] == [0, 2, 4] and decoded[100:102] == [140, 142]
    assert [frame_number(img) for _, img in iter_frames(vfr, 100, 103, exact=True)] == decoded[100:103]
    assert info.time(100) == pytest.approx(140 / 60.0) and info.frame_at(140 / 60.0 + 0.004) == 100
    cfr = tmp_path / "cfr.mp4"
    ffmpeg("-i", str(src), "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(cfr))
    plain = probe_video(cfr)
    assert not plain.vfr and plain.warnings == [] and plain.frame_at(2.0) == 120
    assert [frame_number(img) for _, img in iter_frames(cfr, 150, 153)] == [150, 151, 152]  # seeking is exact


@needs_ffmpeg
def test_letterboxed_and_interlaced_files_are_flagged(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30", "-t", "1", "-vf", "pad=320:240:0:30:black",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src))
    info = probe_video(src)
    assert any(w.startswith("black bars (top 30,") for w in info.warnings) and not info.vfr
    inter = tmp_path / "inter.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30000/1001", "-t", "1", "-vf", "setfield=tff",
           "-flags", "+ilme+ildct", "-c:v", "libx264", "-x264opts", "tff=1", "-pix_fmt", "yuv420p", str(inter))
    info = probe_video(inter)
    assert info.interlaced and any(w.startswith("interlaced (tt)") for w in info.warnings)
    flat = tmp_path / "flat.mp4"  # what OpenCV 5.0.0 gave for interlaced files: frames of one colour
    ffmpeg("-f", "lavfi", "-i", "color=c=gray:size=320x180:rate=30", "-t", "1", "-c:v", "libx264", "-pix_fmt",
           "yuv420p", str(flat))
    assert any("decode as flat images" in w for w in probe_video(flat).warnings)
    # a stream-copied trim: the frames before the cut stay in the file (an edit list hides them), and OpenCV
    # counts them (here 120 counted, 111 decoded)
    full, cut = tmp_path / "full.mp4", tmp_path / "cut.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30000/1001", "-t", "6", "-c:v", "libx264", "-g", "60",
           "-pix_fmt", "yuv420p", str(full))
    ffmpeg("-ss", "2.3", "-i", str(full), "-c", "copy", str(cut))
    info = probe_video(cut)
    decoded = sum(1 for _ in iter_frames(cut))
    assert info.times is not None and len(info.times) == decoded  # the hidden frames are not counted
    assert (decoded != info.n_frames) == any("disagree with the duration" in w for w in info.warnings)


def test_detect_video_stacks_the_frames_at_the_training_spacing(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[int, list[int], tuple[int, int]]] = []

    class FakeStudent:
        def __init__(self, config: student.StudentConfig) -> None:
            self.config = config

        def infer_arrays(self, x: np.ndarray, sizes: list) -> list:
            for sample, wh in zip(x, sizes, strict=True):
                seen.append((int(sample[-1, 0, 0]), [int(v) for v in sample[:, 0, 0]], wh))
            return [([(1.0, 2.0, 0.5)], [])] * len(x)

    monkeypatch.setattr(student, "StudentDetector", FakeStudent)
    monkeypatch.setattr(student, "preprocess_frames", lambda fs: np.full((1, 2, 2), frame_number(fs[0]), np.float32))
    for fps, step in ((REF_FPS, 1), (59.94, 2)):
        video = numbered_video(tmp_path / f"v{step}.avi", 30, fps)
        cam = CameraTrack(C, (320, 180), fps, {f: (0.0, 15.0, 0.0, 1000.0) for f in range(3, 30) if f != 20})
        seen.clear()
        cache = analyze.detect_video(video, cam, Path("w.pt"), start=2, end=26, batch=4, progress=False)
        assert sorted(cache.balls) == [f for f in range(3, 26) if f != 20]  # frames with the camera only
        for f, inputs, wh in seen:
            assert inputs == [max(2, f - 2 * step), max(2, f - step), f] and wh == (320, 180)
    other = CameraTrack(C, (640, 360), REF_FPS, {0: (0.0, 15.0, 0.0, 1000.0)})
    with pytest.raises(ValueError, match="detections would be misplaced"):
        analyze.detect_video(video, other, Path("w.pt"), progress=False)


def test_perceive_converts_the_pipelines_frame_counts_and_takes_the_cameras_times(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from courtvision.perception import pipeline

    made: list[Any] = []

    class Recording(pipeline.PerceptionPipeline):
        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, **kw)
            made.append(self)

    monkeypatch.setattr(pipeline, "PerceptionPipeline", Recording)
    for fps in (NTSC, 59.94):
        video = numbered_video(tmp_path / "v.avi", 12, fps)
        params = {f: (0.0, 16.8, 0.0, 1000.0) for f in range(12) if f != 5}
        cam = CameraTrack(C, (320, 180), fps, params)
        persons = {f: np.array([[150.0, 80.0, 170.0, 140.0, 0.8]], np.float32) for f in params}
        reads = [analyze.ScoreboardReading(t=7 / fps, home_score=1, away_score=2, period=1, game_clock=600.0,
                                           shot_clock=24.0)]
        states = list(perceive(video, cam, DetectionCache({f: np.zeros((0, 3), np.float32) for f in params}, persons),
                               BallTrack(), reads))
        assert [s.frame_idx for s in states] == list(range(12)) and all(s.t == f / fps for f, s in enumerate(states))
        assert [s.frame_idx for s in states if s.scoreboard is not None] == [7]
        cfg = made[-1].config
        if fps == NTSC:  # the tuned pipeline, as before
            assert cfg == pipeline.PerceptionConfig(team={}, foot_direct=True, court_margin_ft=4.0,
                                                    compute_embeddings=True, tracker=analyze.PerceiveConfig().tracker)
            assert made[-1].tracker._buffer == 30
        else:
            assert (cfg.duplicate_new_frames, cfg.track_memory_frames, cfg.team_observe_every) == (60, 600, 6)
            assert cfg.coast_confidence_decay ** 2 == pytest.approx(0.8)
            assert cfg.tracker["high_thresh"] == 0.3 and cfg.tracker["emit_lost_frames"] == 10
            assert made[-1].tracker._buffer == 60 and cfg.identity["relink_new_track_frames"] == 60


def legacy_preprocess(frames_bgr: list[np.ndarray]) -> np.ndarray:
    """``student.preprocess_frames`` of commit 063e638 (960 px wide whatever the frame's shape)."""
    h_in, w_in = student.INPUT_HW
    chans = []
    for f in frames_bgr:
        s = w_in / f.shape[1]
        img = cv2.resize(f, (w_in, int(round(f.shape[0] * s))), interpolation=cv2.INTER_AREA) if s != 1.0 else f  # noqa: RUF046
        if img.shape[0] < h_in:
            img = cv2.copyMakeBorder(img, 0, h_in - img.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        img = img[:h_in]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chans.append(((rgb - student.MEAN) / student.STD).transpose(2, 0, 1))
    return np.concatenate(chans, axis=0)


def test_student_input_of_a_16_9_frame_is_unchanged_and_a_taller_frame_keeps_its_bottom() -> None:
    rng = np.random.default_rng(0)
    for w, h in ((960, 540), (1280, 720), (1920, 1080), (3840, 2160)):
        img = np.tile(rng.integers(0, 255, (h // 4, w // 4, 3), dtype=np.uint8), (4, 4, 1))
        assert student.input_scale(w, h) == 960 / w
        np.testing.assert_array_equal(student.preprocess_frames([img]), legacy_preprocess([img]))
    sd = np.full((480, 640, 3), 90, np.uint8)  # a 4:3 SD capture of a letterboxed 16:9 picture
    sd[400:420] = 255  # the bottom of the picture
    x = student.preprocess_frames([sd])
    s = student.input_scale(640, 480)
    assert s == pytest.approx(544 / 480) and x.shape == (3, 544, 960)
    assert x[0, int(410 * s), 100] > 1.5 and x[0, 300, 900] < -1.5  # the bright rows are in; the right is padding
    assert legacy_preprocess([sd])[0, :, 100].max() < 1.5  # 960 px wide, the old input lost them


def test_student_history_steps_and_the_live_detectors_input_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    assert [student.history_step(f) for f in (15.0, 23.976, 25.0, 29.97, 44.0, 50.0, 59.94, 60.0, 119.88, 0.0)] == \
        [1, 1, 1, 1, 1, 2, 2, 2, 4, 1]
    for step, expect in ((1, [[0, 0, 0], [0, 0, 1], [0, 1, 2], [3, 4, 5]]), (2, [[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                                                               [1, 3, 5]])):
        det = student.StudentDetector(student.StudentConfig(frame_step=step))
        got: list[list[int]] = []
        monkeypatch.setattr(det, "infer", lambda batch, got=got: got.append([int(f[0, 0, 0]) for f in batch[0]])
                            or [([], [])])
        for i in range(6):
            det.detect(np.full((4, 4, 3), i, np.uint8), i, i / 60.0)
        assert [got[i] for i in (0, 1, 2, 5)] == expect
