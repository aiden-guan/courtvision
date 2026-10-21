"""GameController (server.app.Controller): manual revisions reach stats, scoreboard override,
substitutions change lineup seconds, calibration from landmarks, control, frame JPEG, FastAPI wiring."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

import numpy as np
import pytest
from fastapi.testclient import TestClient

from courtvision.config import load_config
from courtvision.pipeline import LiveGame, SimSource
from courtvision.server.app import Controller, create_app
from courtvision.server.hub import Hub, HubConfig
from courtvision.sim.simulator import GameSimulator
from courtvision.types import EventKind, EventStatus, Team

NO_KEYS: dict[str, str] = {}


def _cfg(*extra: str):
    return load_config(None, ["sim.seconds=120", "sim.seed=7", "narration.tts=none", *extra])


@pytest.fixture(scope="module")
def result():
    return GameSimulator(_cfg().sim).run()


Hook = Callable[[LiveGame], Awaitable[None]]


async def _drive(game: LiveGame, hooks: dict[int, Hook]) -> None:
    await game.start()
    async for item in game.source:
        hook = hooks.get(item.frame_idx)
        if hook is not None:
            await hook(game)
        await game.process(item)
    await game.finish()


def _landmarks(result) -> tuple[list[list[float]], list[str], list[list[float]]]:
    cam, court = result.camera, result.court
    pts, names, court_pts = [], [], []
    for name, (x, y) in court.landmarks().items():
        uv = cam.project(np.array([x, y, 0.0]))
        if np.all(np.isfinite(uv)) and cam.in_image(uv, 5.0)[0]:
            pts.append([float(uv[0]), float(uv[1])])
            names.append(name)
            court_pts.append([x, y])
    return pts, names, court_pts


async def test_manual_revisions_reach_stats(result) -> None:
    game = LiveGame(_cfg(), SimSource(result, "truth3d", duration_s=60.0), hub=Hub(), realtime=False, env=NO_KEYS)
    record: dict = {}

    async def correct(g: LiveGame) -> None:
        makes = [e for e in g.events if e.kind is EventKind.SHOT_MADE and e.status is not EventStatus.RETRACTED]
        made = makes[0]  # seed 7: made free throws at t = 30.0 s and 33.8 s
        shooter, value = made.player_id, made.value
        teammates = [e.player_id for e in result.roster.entries if e.team is made.team and e.player_id != shooter]
        attempt = next(e for e in g.events if e.kind is EventKind.SHOT_ATTEMPT and e.shot_id == made.shot_id)
        before = g.snapshot()
        # 1) the operator credits the shot to a teammate: revisions of the engine's attempt and make
        #    (stats key a shot by shot_id and credit the attempt's shooter)
        await g.controller.add_manual({**attempt.to_dict(), "player_id": teammates[0], "source": "manual"})
        out = await g.controller.add_manual({**made.to_dict(), "player_id": teammates[0], "source": "manual"})
        revised = next(e for e in out["events"] if e["event_id"] == made.event_id)
        after = g.snapshot()
        record["moved"] = (before.players[shooter].pts - after.players[shooter].pts,
                           after.players[teammates[0]].pts - before.players[teammates[0]].pts, value)
        record["revision"] = (made.revision, revised["revision"])
        # 2) ... then retracts it altogether
        await g.controller.add_manual({"event_id": made.event_id, "kind": "shot_made", "t": made.t,
                                       "status": "retracted", "source": "manual"})
        final = g.snapshot()
        record["retracted"] = (after.event_score[made.team] - final.event_score[made.team], value)
        record["event"] = g._events[made.event_id]
        record["hub"] = [m for m in g.hub.events() if m["event_id"] == made.event_id]

    await _drive(game, {1500: correct})  # t = 50 s: two baskets have been scored by then
    shooter_lost, teammate_gained, value = record["moved"]
    assert shooter_lost == value and teammate_gained == value
    assert record["revision"][1] == record["revision"][0] + 1
    assert record["retracted"][0] == record["retracted"][1]
    assert record["event"].status is EventStatus.RETRACTED and record["event"].source == "manual"
    assert record["hub"] and record["hub"][-1]["status"] == "retracted"  # the dashboard sees the revision
    assert game.final_stats is not None and game.final_stats.score_mismatch  # board still counts the basket


async def test_scoreboard_override(result) -> None:
    hub = Hub(HubConfig(max_rate_hz={}))
    game = LiveGame(_cfg(), SimSource(result, "truth3d", scoreboard="none", duration_s=20.0), hub=hub,
                    realtime=False, env=NO_KEYS)
    seen: dict = {}

    async def override(g: LiveGame) -> None:
        assert g.snapshot().score_source == "events"
        out = await g.controller.set_scoreboard({"home_score": 21, "away_score": 19, "period": 2,
                                                 "game_clock": 300.0, "shot_clock": 14.0, "source": "manual"})
        seen["result"] = out
        snap = g.snapshot()
        seen["snap"] = (snap.score, snap.score_source, snap.period, snap.game_clock)

    async def later(g: LiveGame) -> None:
        sb = g.engine.scoreboard
        seen["engine"] = (sb.score, sb.period)
        await g.controller.set_scoreboard({"game_clock": 250.0, "source": "manual"})  # partial: scores kept
        seen["partial"] = g.latest_board

    await _drive(game, {300: override, 330: later})
    assert seen["result"]["home_score"] == 21
    assert seen["snap"] == ({Team.HOME: 21, Team.AWAY: 19}, "scoreboard", 2, 300.0)
    assert seen["engine"] == ({Team.HOME: 21, Team.AWAY: 19}, 2)  # the engine received the manual reading
    board = seen["partial"]
    assert (board.home_score, board.away_score, board.period, board.game_clock) == (21, 19, 2, 250.0)
    frame = hub.latest("frame")
    assert frame["clock"]["period"] == 2 and frame["clock"]["game"] == 250.0


async def test_scoreboard_override_holds_over_source_readings(result) -> None:
    """A manual score / period used to be undone by the next sim (or OCR) reading: each overridden field is held
    until the source itself shows something else for it, then the source takes over again."""
    game = LiveGame(_cfg(), SimSource(result, "truth3d", scoreboard="truth", duration_s=40.0), realtime=False,
                    env=NO_KEYS)
    truth = {f.frame_idx: f.scoreboard for f in result.frames[:1200]}
    seen: dict = {}

    async def override(g: LiveGame) -> None:
        seen["before"] = (truth[300].home_score, truth[300].away_score, truth[300].period)
        await g.controller.set_scoreboard({"home_score": 5, "away_score": 3, "period": 2, "source": "manual"})

    async def check(name: str) -> Hook:
        async def hook(g: LiveGame) -> None:
            b = g.latest_board
            seen[name] = (b.home_score, b.away_score, b.period, g.engine.scoreboard.score)
        return hook

    first_change = next(i for i in range(301, 1200) if (truth[i].home_score, truth[i].away_score)
                        != (truth[300].home_score, truth[300].away_score))
    await _drive(game, {300: override, 600: await check("held"), 1199: await check("released")})
    assert seen["before"] == (0, 0, 1) and first_change < 1100  # seed 7: the board changes at ~30 s
    assert seen["held"] == (5, 3, 2, {Team.HOME: 5, Team.AWAY: 3})  # 10 s of truth readings later
    end = truth[1199]
    home, away, period, _ = seen["released"]
    assert (home, away) == (end.home_score, end.away_score)  # the source scored: it leads again
    assert period == 2  # the source never showed another period: still held


async def test_substitution_changes_lineup_seconds(result) -> None:
    game = LiveGame(_cfg("operator.replay_kinds=[]"), SimSource(result, "truth3d", duration_s=60.0),
                    realtime=False, env=NO_KEYS)
    seen: dict = {}

    async def sub(g: LiveGame) -> None:
        on = g.stats.on_court(Team.HOME)
        bench = [e.player_id for e in result.roster.entries if e.team is Team.HOME and e.player_id not in on]
        out_id, in_id = on[0], bench[0]
        snap = g.snapshot()
        seen.update(out_id=out_id, in_id=in_id, out_before=snap.players[out_id].seconds,
                    in_before=snap.players[in_id].seconds)
        res = await g.controller.set_lineup({"team": "home", "out": out_id, "in": in_id, "t": g.current_t})
        seen["res"] = res

    await _drive(game, {600: sub})  # at t = 20 s
    final = game.final_stats
    assert final is not None
    out_id, in_id = seen["out_id"], seen["in_id"]
    assert seen["in_before"] == 0.0 and seen["out_before"] > 0.0
    assert in_id in seen["res"]["on_court"] and out_id not in seen["res"]["on_court"]
    assert final.players[in_id].seconds > 5.0  # on the floor for the last 40 s of video
    assert final.players[out_id].seconds == pytest.approx(seen["out_before"], abs=0.5)  # frozen on the bench
    subs = [e for e in game.events if e.kind is EventKind.SUBSTITUTION]
    assert subs and subs[-1].player_id == in_id and subs[-1].other_player_id == out_id
    assert subs[-1].source == "manual" and subs[-1].status is EventStatus.CONFIRMED


async def test_calibration_from_landmarks_on_sim_projected_points(result) -> None:
    pts, names, court_pts = _landmarks(result)
    assert len(pts) >= 8
    game = LiveGame(_cfg(), SimSource(result, "truth3d", duration_s=1.0), realtime=False, env=NO_KEYS)
    await game.start()
    res = await game.controller.calibrate({"image_points": pts, "landmarks": names})
    assert res["rmse_ft"] < 0.01 and res["inliers"] == len(pts) and res["applied"] is False  # no perception here
    by_points = await game.controller.calibrate({"image_points": pts, "court_points": court_points(court_pts)})
    assert by_points["rmse_ft"] < 0.01
    # rescaled clicks from a smaller JPEG
    half = [[u / 2, v / 2] for u, v in pts]
    scaled = await game.controller.calibrate({"image_points": half, "landmarks": names, "image_size": [640, 360]})
    assert scaled["rmse_ft"] < 0.01
    H = game._calibration.homography
    xy = H.image_to_court(np.array(pts[0]))
    assert np.allclose(xy, court_pts[0], atol=0.01)
    with pytest.raises(ValueError):
        await game.controller.calibrate({"image_points": pts[:3], "landmarks": names[:3]})
    with pytest.raises(ValueError):
        await game.controller.calibrate({"image_points": pts[:4], "landmarks": ["nowhere", *names[1:4]]})
    await game.finish()


def court_points(rows: list[list[float]]) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in rows]


async def test_control_and_frame_jpeg(result) -> None:
    game = LiveGame(_cfg(), SimSource(result, "truth3d", duration_s=2.0), realtime=False, env=NO_KEYS)
    await game.start()
    assert await game.controller.latest_frame_jpeg() is None  # nothing processed yet
    seen: dict = {}

    async def act(g: LiveGame) -> None:
        seen["jpeg"] = await g.controller.latest_frame_jpeg()
        seen["mute"] = (await g.controller.control({"action": "mute"}))["muted"]
        seen["unmute"] = (await g.controller.control({"action": "unmute"}))["muted"]
        seen["speed"] = (await g.controller.control({"action": "speed", "value": 2.0}))["speed"]
        await g.controller.control({"action": "dead_ball"})
        seen["dead"] = g.engine.deadball.is_dead(g.current_t)
        with pytest.raises(ValueError):
            await g.controller.control({"action": "rewind"})

    await _drive(game, {30: act})
    assert seen["jpeg"][:2] == b"\xff\xd8"  # rendered on demand from the sim truth frame
    assert seen["mute"] is True and seen["unmute"] is False and seen["speed"] == 2.0 and seen["dead"] is True


async def test_controller_through_the_fastapi_app(result) -> None:
    hub = Hub()
    game = LiveGame(_cfg(), SimSource(result, "truth3d", duration_s=30.0), hub=hub, realtime=False, env=NO_KEYS)
    assert isinstance(game.controller, Controller)
    await game.run()
    pts, names, _ = _landmarks(result)
    starters = result.starters[Team.HOME]
    bench = [e.player_id for e in result.roster.entries if e.team is Team.HOME and e.player_id not in starters]
    event = next(e for e in game.events if e.kind is EventKind.PASS)
    with TestClient(create_app(hub, game.controller)) as client:  # the app runs on its own loop / thread
        r = client.post("/api/calibration", json={"image_points": pts, "landmarks": names})
        assert r.status_code == 200 and r.json()["result"]["rmse_ft"] < 0.01
        r = client.get("/api/frame.jpg")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        r = client.post("/api/lineup", json={"team": "home", "out": starters[0], "in": bench[0], "t": 29.0})
        assert r.status_code == 200 and bench[0] in r.json()["result"]["on_court"]
        r = client.post("/api/events", json={"event_id": event.event_id, "kind": "pass", "t": event.t,
                                             "status": "retracted"})
        assert r.status_code == 200
        r = client.post("/api/scoreboard", json={"home_score": 30, "away_score": 31})
        assert r.status_code == 200
        r = client.post("/api/control", json={"action": "pause"})
        assert r.status_code == 200 and r.json()["result"]["paused"] is True
        r = client.post("/api/events", json={"kind": "foul", "t": 29.5, "player_id": starters[1],
                                             "other_player_id": result.starters[Team.AWAY][0]})
        assert r.status_code == 200
        events = client.get("/api/events?kind=pass").json()["events"]
        assert next(e for e in events if e["event_id"] == event.event_id)["status"] == "retracted"
    assert game._events[event.event_id].status is EventStatus.RETRACTED
    snap = game.snapshot()
    assert snap.score == {Team.HOME: 30, Team.AWAY: 31}
    assert any(e.kind is EventKind.FOUL and e.player_id == starters[1] for e in game.events)
    assert json.loads(json.dumps(hub.latest("stats")))["stats"]["score"] == {"home": 30, "away": 31}
