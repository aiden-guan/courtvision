"""Video paths: the sim "video" mode through PerceptionPipeline, VideoFileSource (lossless FFV1 file of
sim renders) with operator calibration + scoreboard ROI, and CameraSource on a stream URL.

Measured (seed 7, 8 s window around two made free throws, this machine): MOTA 0.99, perception
~19 ms/frame, both makes confirmed by the OCR scoreboard and named in the transcript. Re-pinned to seed 13
when the simulator's flow changed (round 2; seed 7's first make is now a free throw whose operator-lagged
board change falls after the window): an 8 s window ending 1.5 s after a made 2, MOTA 0.93, the make
confirmed by the OCR scoreboard. (Seeds 8-12 fail the same window rule; 13 is the first that fits it.)
The window now ends 3.0 s after the make (the make wait, RESEARCH §7A.5): the operator's score lag is drawn
from up to 2.0 s plus two 5 Hz OCR reads, so a window ending 1.5 s after the make could close before the
board changed (round 3: seed 13's make was left provisional).
"""

from __future__ import annotations

import asyncio
import json

import cv2
import numpy as np
import pytest

from courtvision.config import load_config
from courtvision.pipeline import CameraSource, LiveGame, VideoFileSource, make_source, run_offline
from courtvision.sim import render
from courtvision.sim.simulator import GameSimulator
from courtvision.types import EventKind, EventStatus

SEED, SIM_S, WINDOW_S = 13, 40.0, 8.0
NO_KEYS: dict[str, str] = {}


def _cfg(*extra: str):
    return load_config(None, [f"sim.seconds={SIM_S}", f"sim.seed={SEED}", "narration.tts=none", *extra])


@pytest.fixture(scope="module")
def sim_result():
    return GameSimulator(_cfg().sim).run()


@pytest.fixture(scope="module")
def video_run(sim_result, tmp_path_factory):
    makes = [e.t for e in sim_result.events if e.kind is EventKind.SHOT_MADE and e.t >= WINDOW_S]
    start = max(0.0, makes[0] + 3.0 - WINDOW_S) if makes else 16.0
    cfg = _cfg("source.sim_mode=video", f"source.start_s={start}", f"source.duration_s={WINDOW_S}")
    out = tmp_path_factory.mktemp("video")
    return run_offline(cfg, make_source(cfg, sim_result=sim_result, realtime=False), out, env=NO_KEYS)


def test_video_mode_runs_through_perception(video_run) -> None:
    lat = video_run.latency
    n = int(round(WINDOW_S * 30))
    assert video_run.frames == n
    assert lat["perception_ms_per_frame"] is not None and lat["perception_ms_per_frame"]["n"] == n
    trk = video_run.track_report
    assert trk is not None and trk.n_frames == n
    print(f"\nvideo: MOTA {trk.mota:.3f} IDF1 {trk.idf1:.3f} RMSE {trk.rmse_ft:.2f} ft; perception "
          f"{lat['perception_ms_per_frame']['mean']:.1f} ms/frame; events "
          f"{ {k: r.f1 for k, r in video_run.event_report.per_kind.items()} }")
    assert trk.mota > 0.9 and trk.rmse_ft < 1.5
    rep = video_run.event_report
    assert rep is not None and rep.per_kind["shot_made"].f1 >= 0.99


def test_video_mode_scoreboard_identity_and_narration(video_run) -> None:
    makes = [e for e in video_run.events if e.kind is EventKind.SHOT_MADE and e.status is not EventStatus.RETRACTED]
    assert makes and all(e.status is EventStatus.CONFIRMED for e in makes)  # OCR score deltas confirmed them
    # identity confidences come from perception (not the 1.0 used for sim truth / manual events)
    confs = [(e.attrs["id_conf"], e.attrs["jersey_conf"]) for e in video_run.events if e.player_id]
    assert confs and all(0.0 <= a <= 1.0 and 0.0 <= b <= 1.0 for a, b in confs)
    assert any(a < 1.0 or b < 1.0 for a, b in confs)
    assert video_run.transcript
    assert video_run.stats.score_source == "scoreboard"


def _write_ffv1(path, images, fps=30.0) -> None:
    h, w = images[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
    assert writer.isOpened()
    for img in images:
        writer.write(img)
    writer.release()


@pytest.fixture(scope="module")
def footage(sim_result, tmp_path_factory):
    """2 s of rendered sim frames as a lossless video file + a roster file."""
    d = tmp_path_factory.mktemp("footage")
    renderer = render.SyntheticRenderer(sim_result.camera, sim_result.court)
    i0 = int(16.0 * 30)
    images = [renderer.render(sim_result.frames[i], referees=sim_result.referees[i]) for i in range(i0, i0 + 60)]
    path = d / "game.avi"
    _write_ffv1(path, images)
    roster = {"teams": {t.value: n for t, n in sim_result.roster.team_names.items()},
              "players": [{"player_id": e.player_id, "team": e.team.value, "jersey": e.jersey, "name": e.name}
                          for e in sim_result.roster.entries]}
    (d / "roster.json").write_text(json.dumps(roster))
    return path, d / "roster.json", images


def _landmark_points(sim_result) -> tuple[list[list[float]], list[str]]:
    cam, court = sim_result.camera, sim_result.court
    names, pts = [], []
    for name, (x, y) in court.landmarks().items():
        uv = cam.project(np.array([x, y, 0.0]))
        if np.all(np.isfinite(uv)) and cam.in_image(uv, 5.0)[0]:
            names.append(name)
            pts.append([float(uv[0]), float(uv[1])])
    return pts, names


def _operator_roi() -> dict:
    """render.SCOREBOARD_ROI in the dashboard's image-pixel form ({"<group>_<i>": [x1, y1, x2, y2]})."""
    roi = render.SCOREBOARD_ROI
    x0, y0 = roi["corners"][0]
    cells = {f"{group}_{i}": [x0 + x, y0 + y, x0 + x + w, y0 + y + h]
             for group, boxes in roi["cells"].items() for i, (x, y, w, h) in enumerate(boxes)}
    return {"corners": roi["corners"], "cells": cells, "space": "image"}


def test_video_file_source_reads_pts(footage) -> None:
    path, _, images = footage

    async def read():
        src = VideoFileSource(path, realtime=False)
        out = [item async for item in src]
        src.close()
        return src, out

    src, items = asyncio.run(read())
    assert len(items) == len(images) and src.image_size == (1280, 720) and src.fps == pytest.approx(30.0)
    ts = [it.t for it in items]
    assert ts[0] == pytest.approx(0.0) and np.allclose(np.diff(ts), 1.0 / 30.0, atol=1e-3)
    assert np.array_equal(items[5].image, images[5])  # FFV1 is lossless
    with pytest.raises(FileNotFoundError):
        VideoFileSource(path.parent / "missing.avi")


def test_uncalibrated_video_waits_for_operator_calibration(footage, sim_result) -> None:
    path, roster_path, _ = footage
    cfg = _cfg("source.kind=video", f"source.path={path}", f"game.roster_path={roster_path}",
               "perception.detector=color_blob", "perception.jersey_reader=none")
    pts, names = _landmark_points(sim_result)
    assert len(pts) >= 8

    async def go():
        src = make_source(cfg, realtime=False)
        game = LiveGame(cfg, src, realtime=False, env=NO_KEYS)
        await game.start()
        seen: dict[str, object] = {}
        async for item in src:
            if item.frame_idx == 10:
                seen["status"] = game.status()
                seen["jpeg"] = await game.controller.latest_frame_jpeg()
                seen["roi"] = await game.controller.set_scoreboard_roi(_operator_roi())
                # clicks on a half-size JPEG are rescaled to the source resolution
                half = [[u / 2.0, v / 2.0] for u, v in pts]
                seen["cal"] = await game.controller.calibrate({"image_points": half, "landmarks": names,
                                                               "image_size": [640, 360]})
            await game.process(item)
        await game.finish()
        return game, seen

    game, seen = asyncio.run(go())
    assert seen["status"]["needs_calibration"] is True and game.perception is not None
    assert isinstance(seen["jpeg"], bytes) and seen["jpeg"][:2] == b"\xff\xd8"
    assert seen["roi"]["applied"] is False  # stored until perception starts
    cal = seen["cal"]
    assert cal["rmse_ft"] < 0.05 and cal["inliers"] == len(pts) and cal["image_size"] == [1280, 720]
    assert game.status()["needs_calibration"] is False
    assert game._frames_done == 60 - 10  # frames before the calibration only fed the JPEG endpoint
    assert game.latest_board is not None and game.latest_board.source == "ocr"
    assert game.snapshot().tracking  # perceived players reached the stats engine


def test_realtime_video_perceives_in_a_worker_thread(sim_result) -> None:
    """Real time: perception runs in the worker thread (the loop stays free), calibration hot-swaps
    while frames flow, and drop_late_s sheds frames when perception cannot keep up."""
    import threading

    from courtvision.pipeline import SimSource

    pts, names = _landmark_points(sim_result)

    async def go(speed: float, drop: str | None):
        c = _cfg("pipeline.tail_s=0", *([f"source.drop_late_s={drop}"] if drop else []))
        src = SimSource(sim_result, "video", realtime=True, speed=speed, start_s=16.0, duration_s=1.0)
        game = LiveGame(c, src, env=NO_KEYS)
        threads: set[str] = set()
        original = game._perceive

        def spy(*args):
            threads.add(threading.current_thread().name)
            return original(*args)

        game._perceive = spy  # type: ignore[method-assign]
        task = asyncio.create_task(game.run())
        ticks = 0
        cal = None
        while not task.done():
            await asyncio.sleep(0.005)
            ticks += 1  # the loop keeps running while frames are perceived
            if cal is None and game.perception is not None:
                cal = await game.controller.calibrate({"image_points": pts, "landmarks": names})
        await task
        return game, threads, ticks, cal

    game, threads, ticks, cal = asyncio.run(go(speed=4.0, drop=None))
    assert game.realtime and game._frames_done == 30
    assert threads and all(name.startswith("perception") for name in threads)
    assert ticks > 10
    assert cal is not None and cal["applied"] is True and cal["rmse_ft"] < 0.05
    assert game.perception.calibration.homography is game._calibration.homography
    late, _, _, _ = asyncio.run(go(speed=200.0, drop="0.05"))
    assert late._dropped > 0 and late._frames_done + late._dropped == 30


def test_camera_source_on_a_stream_url(footage) -> None:
    path, _, images = footage

    async def read():
        src = CameraSource(str(path), duration_s=None)
        items = []
        async for item in src:
            items.append(item)
            if len(items) == 5:
                break
        src.close()
        return src, items

    src, items = asyncio.run(read())
    assert src.live and src.realtime and len(items) == 5
    assert items[0].t == 0.0 and all(b.t >= a.t for a, b in zip(items, items[1:]))
    assert items[0].image.shape == images[0].shape
