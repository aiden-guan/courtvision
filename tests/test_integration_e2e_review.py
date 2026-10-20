"""Regression tests for defects found by the adversarial end-to-end review (running simulate, serve,
render + run against the whole stack). Each test names the defect it pins down."""

from __future__ import annotations

import asyncio
import dataclasses
import json

import numpy as np
import pytest

from courtvision.cli import build_parser, load_cli_config, serve_flags, unused_serve_flags
from courtvision.config import load_config
from courtvision.pipeline import (
    EventEnricher,
    LiveGame,
    SimSource,
    TieredPseudonyms,
    director_overrides,
    hub_stats_msg,
)
from courtvision.server.hub import Hub, HubConfig, dumps
from courtvision.sim.simulator import GameSimulator
from courtvision.types import Event, EventKind, EventStatus, FrameState, PlayerState, Team

NO_KEYS: dict[str, str] = {}


def _cfg(*extra: str):
    return load_config(None, ["sim.seconds=60", "sim.seed=7", "narration.tts=none", *extra])


@pytest.fixture(scope="module")
def result():
    return GameSimulator(_cfg().sim).run()


async def _drive(game: LiveGame, hooks: dict) -> None:
    await game.start()
    async for item in game.source:
        hook = hooks.get(item.frame_idx)
        if hook is not None:
            await hook(game)
        await game.process(item)
    await game.finish()


def _mates(result, pid: str) -> list[str]:
    team = result.roster.team_of(pid)
    return [e.player_id for e in result.roster.entries if e.team is team and e.player_id != pid]


# --------------------------------------------------------------------------- operator corrections


async def test_dashboard_scorer_correction_moves_the_points(result) -> None:
    """Defect: the dashboard revises ONE event. Correcting the scorer on the make made the narrator
    say "Correction: the basket belongs to <new>" while the box score kept every point on the old
    shooter (stats credit the attempt's shooter, which was never revised)."""
    hub = Hub(HubConfig(max_rate_hz={}))
    game = LiveGame(_cfg(), SimSource(result, "truth3d"), hub=hub, realtime=False, env=NO_KEYS)
    seen: dict = {}

    async def correct(g: LiveGame) -> None:
        made = next(e for e in g.events if e.kind is EventKind.SHOT_MADE and e.status is not EventStatus.RETRACTED)
        new = _mates(result, made.player_id)[0]
        before = g.snapshot()
        body = next(e for e in hub.events(kinds=["shot_made"]) if e["event_id"] == made.event_id)
        # exactly what static/app.js eventBody() sends for a correction
        body = {**body, "player_id": new, "source": "manual", "revision": body["revision"] + 1,
                "attrs": {**body["attrs"], "corrected_from_revision": body["revision"]}}
        await g.controller.add_manual(body)
        after = g.snapshot()
        attempt = next(e for e in g.events if e.kind is EventKind.SHOT_ATTEMPT and e.shot_id == made.shot_id)
        seen.update(old=made.player_id, new=new, value=made.value, attempt=attempt,
                    pts=(before.players[made.player_id].pts - after.players[made.player_id].pts,
                         after.players[new].pts - before.players.get(new, after.players[new]).pts))
        seen["new_name"] = result.roster.display_name(new)

    await _drive(game, {1300: correct})  # t = 43.3 s: the made free throws at 30.0 s / 33.8 s are in
    assert seen["pts"] == (seen["value"], seen["value"])  # the points moved with the correction
    assert seen["attempt"].player_id == seen["new"] and seen["attempt"].source == "manual"
    assert seen["attempt"].attrs.get("synced_from")
    corrections = [u for u in game.transcript if u.channel == "correction"]
    assert len(corrections) == 1 and seen["new_name"] in corrections[0].text
    final = game.final_stats
    assert final is not None and final.players[seen["new"]].pts >= seen["value"]


async def test_scorer_correction_carries_the_shots_assist_and_drops_impossible_ones(result) -> None:
    """The shot's ASSIST names the scorer as ``other_player_id``: it follows a teammate correction
    and is retracted when the basket moves to the other team (or to the assister himself)."""
    game = LiveGame(_cfg(), SimSource(result, "truth3d", duration_s=1.0), realtime=False, env=NO_KEYS)
    home = [e.player_id for e in result.roster.entries if e.team is Team.HOME]
    away = [e.player_id for e in result.roster.entries if e.team is Team.AWAY]
    c = game.controller
    for sid, kind_args in (("s9001", {}), ("s9002", {})):
        await c.add_manual({"kind": "shot_attempt", "t": 10.0, "team": "home", "player_id": home[0], "value": 2,
                            "shot_id": sid, "attrs": {"shot_type": "jumper", **kind_args}})
        await c.add_manual({"kind": "shot_made", "t": 11.0, "team": "home", "player_id": home[0], "value": 2,
                            "shot_id": sid})
        await c.add_manual({"kind": "assist", "t": 11.0, "team": "home", "player_id": home[1],
                            "other_player_id": home[0], "shot_id": sid})

    def latest(sid: str, kind: EventKind) -> Event:
        return next(e for e in game.events if e.shot_id == sid and e.kind is kind)

    made = latest("s9001", EventKind.SHOT_MADE)
    await c.add_manual({**made.to_dict(), "player_id": home[2], "source": "manual"})  # teammate
    assert latest("s9001", EventKind.SHOT_ATTEMPT).player_id == home[2]
    ast = latest("s9001", EventKind.ASSIST)
    assert ast.status is not EventStatus.RETRACTED and ast.other_player_id == home[2] and ast.player_id == home[1]

    made = latest("s9002", EventKind.SHOT_MADE)
    await c.add_manual({**made.to_dict(), "player_id": away[0], "team": "away", "source": "manual"})  # other team
    att = latest("s9002", EventKind.SHOT_ATTEMPT)
    assert (att.player_id, att.team) == (away[0], Team.AWAY)
    assert latest("s9002", EventKind.ASSIST).status is EventStatus.RETRACTED
    snap = game.snapshot()
    assert snap.players[away[0]].pts == 2 and snap.players[home[2]].pts == 2
    assert snap.players[home[1]].ast == 1  # one assist left (s9001)
    await game.finish()


def test_operator_events_get_full_identity_confidence() -> None:
    """Defect: an operator correction copies the CV event's attrs (the dashboard sends them back),
    including that event's ``id_conf`` / ``jersey_conf``; the enricher kept them, so a player the
    operator had just named was rendered as the team name (and the copied value belonged to the old
    actor)."""
    enr = EventEnricher(trusted_identity=False)
    enr.observe(FrameState(1, 1.0, [PlayerState("home-23", Team.HOME, 10.0, 10.0, confidence=0.6)], None))
    cv = enr.enrich(Event("e1", EventKind.FOUL, 1.0, 1, team=Team.HOME, player_id="home-23", source="cv"))
    assert cv.attrs["id_conf"] == 0.6
    manual = Event("e1", EventKind.FOUL, 1.0, 1, team=Team.HOME, player_id="home-5", source="manual", revision=1,
                   attrs={**cv.attrs, "corrected_from_revision": 0})
    out = enr.enrich(manual)
    assert (out.attrs["id_conf"], out.attrs["jersey_conf"]) == (1.0, 1.0)
    feed = enr.enrich(Event("f1", EventKind.FOUL, 1.0, 1, player_id="home-5", source="feed",
                            attrs={"id_conf": 0.1}))
    assert feed.attrs["id_conf"] == 1.0


async def test_operator_entry_with_copied_confidence_is_spoken_by_name(result) -> None:
    game = LiveGame(_cfg(), SimSource(result, "truth3d"), realtime=False, env=NO_KEYS)
    seen: dict = {}

    async def foul(g: LiveGame) -> None:
        pid = [e.player_id for e in result.roster.entries if e.team is Team.HOME][3]
        out = await g.controller.add_manual({"kind": "foul", "t": g.current_t, "team": "home", "player_id": pid,
                                             "attrs": {"id_conf": 0.5, "jersey_conf": 0.0}})
        seen["id"], seen["name"] = out["events"][0]["event_id"], result.roster.display_name(pid)

    await _drive(game, {1300: foul})
    lines = [u.text for u in game.transcript if seen["id"] in u.event_ids]
    assert lines and seen["name"] in lines[0], lines


async def test_substitution_from_the_event_dialog_changes_the_lineup(result) -> None:
    """Defect: the dashboard's event dialog offers "substitution"; POST /api/events stored and
    narrated it, but lineups / minutes / +/- come only from set_lineup / substitute, so the player
    "subbed in" never got on the floor."""
    game = LiveGame(_cfg(), SimSource(result, "truth3d"), realtime=False, env=NO_KEYS)
    seen: dict = {}

    async def sub(g: LiveGame) -> None:
        floor = g.stats.on_court(Team.HOME)
        bench = next(e.player_id for e in result.roster.entries if e.team is Team.HOME and e.player_id not in floor)
        out = floor[0]
        res = await g.controller.add_manual({"kind": "substitution", "t": g.current_t, "team": "home",
                                             "player_id": bench, "other_player_id": out})
        seen.update(bench=bench, out=out, floor=g.stats.on_court(Team.HOME), eid=res["events"][0]["event_id"])
        # a correction of that event must not enter the substitution a second time
        ev = next(e for e in g.events if e.event_id == seen["eid"])
        await g.controller.add_manual({**ev.to_dict(), "source": "manual", "attrs": {"reason": "typo"}})
        seen["floor_after_revision"] = g.stats.on_court(Team.HOME)

    await _drive(game, {900: sub})  # t = 30 s; the simulated operator entered the starting fives at t = 0
    assert seen["bench"] in seen["floor"] and seen["out"] not in seen["floor"] and len(seen["floor"]) == 5
    assert seen["floor_after_revision"] == seen["floor"]
    final = game.final_stats
    assert final is not None and final.players[seen["bench"]].seconds > 0


# --------------------------------------------------------------------------- identity tiers on every line


def test_lines_without_events_apply_the_identity_tiers(result) -> None:
    """Defect: colour lines ("{P07} is 4-for-6" from ``derived`` phrases, Claude candidates) and coach
    cards carry no events, so ``Narrator._render`` passed no confidences and the pseudonym map counted
    the missing value as 1.0: on video / camera sources they named players below the 0.95 tier."""
    game = LiveGame(_cfg(), SimSource(result, "video", duration_s=0.5), realtime=False, env=NO_KEYS)
    assert isinstance(game.narrator.pseudo, TieredPseudonyms) and game.narrator.pseudo.shared is game.pseudo
    roster = result.roster
    low, high, unseen = (e.player_id for e in roster.entries[:3])
    game.enricher.observe(FrameState(1, 1.0, [PlayerState(low, roster.team_of(low), 10, 10, confidence=0.6),
                                              PlayerState(high, roster.team_of(high), 20, 20, confidence=0.99)], None),
                          {high: 1.0, low: 0.0})
    pm = game.narrator.pseudo
    assert pm.token(low) == game.pseudo.token(low)  # same tokens as every other packet builder

    def say(pid: str, **conf) -> str:
        return pm.render("{" + pm.token(pid) + "} is 4-for-6.", roster, **conf)

    team = roster.team_names[roster.team_of(low)]
    assert say(low).startswith(team) and roster.display_name(low) not in say(low)
    assert say(high).startswith(roster.display_name(high))
    assert roster.display_name(unseen) not in say(unseen)  # never identified by perception: team name
    assert say(low, id_conf={low: 0.99}).startswith(roster.display_name(low))  # an event's own value wins
    # the narrator's own rendering path (no events) goes through the tiers
    assert roster.display_name(low) not in game.narrator._render("{" + pm.token(low) + "} scores.", [])

    truth = LiveGame(_cfg(), SimSource(result, "truth3d", duration_s=0.5), realtime=False, env=NO_KEYS)
    tp = truth.narrator.pseudo
    assert tp.render("{" + tp.token(unseen) + "} scores.", roster).startswith(roster.display_name(unseen))


# --------------------------------------------------------------------------- calibration wait


def _landmarks(result) -> tuple[list[list[float]], list[str]]:
    cam, court = result.camera, result.court
    pts, names = [], []
    for name, (x, y) in court.landmarks().items():
        uv = cam.project(np.array([x, y, 0.0]))
        if np.all(np.isfinite(uv)) and cam.in_image(uv, 5.0)[0]:
            pts.append([float(uv[0]), float(uv[1])])
            names.append(name)
    return pts, names


async def test_file_playback_waits_for_the_dashboard_calibration(result) -> None:
    """Defect: with no calibration, ``serve --source video`` kept playing the file unseen (the CLI
    says perception "waits until you calibrate"), and no status message was published while it
    waited, so the dashboard showed ``needs_calibration: false`` until the video had run out."""
    cfg = _cfg("perception.use_source_hints=false")  # a rendered file with no calibration supplied
    hub = Hub(HubConfig(max_rate_hz={}))
    game = LiveGame(cfg, SimSource(result, "video", realtime=True, speed=20.0, duration_s=1.0), hub=hub, env=NO_KEYS)
    task = asyncio.create_task(game.run())
    try:
        for _ in range(400):
            if game._calib_hold:
                break
            await asyncio.sleep(0.01)
        assert game._calib_hold and game.pacer.paused
        status = hub.latest("status")
        assert status is not None and status["needs_calibration"] is True and status["paused"] is True
        idx = game._frame_idx
        await asyncio.sleep(0.3)
        assert game._frame_idx == idx and game._frames_done == 0  # held: the file is not consumed unseen
        pts, names = _landmarks(result)
        out = await game.controller.calibrate({"image_points": pts, "landmarks": names})
        assert out["rmse_ft"] < 0.1
        await asyncio.wait_for(task, timeout=60.0)
    finally:
        if not task.done():
            game.request_stop(drain=False)
            await asyncio.gather(task, return_exceptions=True)
    assert game._frames_done >= 25  # every frame after the held one was perceived
    assert hub.latest("status")["needs_calibration"] is False


# --------------------------------------------------------------------------- hub stats size


async def test_hub_stats_message_stays_small(result) -> None:
    """Defect: ``GameStats.decisions`` holds every graded pass / shot with all options (~5 KB each),
    and the whole list went into the 1 Hz hub ``stats`` message: ~0.8 MB at 10 min, 3.7 MB at 48 min,
    encoded on the event loop (frame steps of 100-200 ms every second, single steps over 1 s)."""
    hub = Hub(HubConfig(max_rate_hz={}, queue_size=1_000_000, disconnect_on_overflow=False))
    hub.bind_loop()
    client = hub.subscribe()
    game = LiveGame(_cfg(), SimSource(result, "truth3d"), hub=hub, realtime=False, env=NO_KEYS)
    await game.run()
    stats_msgs = []
    while (text := client.get_nowait()) is not None:
        msg = json.loads(text)
        if msg["type"] == "stats":
            stats_msgs.append(msg["stats"])
    assert len(stats_msgs) > 30  # ~1 Hz over 60 s, every one of them bounded and slim
    assert all("decisions_total" in m and len(m["decisions"]) <= game.config.pipeline.hub_decisions
               and all("options" not in d for d in m["decisions"]) for m in stats_msgs)
    full = game.final_stats
    assert full is not None and full.decisions and "options" in full.decisions[0]
    sent = hub.latest("stats")["stats"]
    assert sent["decisions_total"] == len(full.decisions)
    assert 0 < len(sent["decisions"]) <= game.config.pipeline.hub_decisions
    d = sent["decisions"][-1]
    assert "options" not in d and "detail" not in d["chosen"] and "detail" not in d["best"]
    assert {"t", "player_id", "label", "delta"} <= set(d) and {"kind", "target_id", "value"} <= set(d["chosen"])
    assert sent["decisions"][-1]["t"] == full.decisions[-1]["t"]  # the newest grades are the ones kept

    many = dataclasses.replace(full, decisions=list(full.decisions) * 40)  # a long game's worth
    msg = hub_stats_msg(many, 40)
    assert len(msg["stats"]["decisions"]) == 40 and msg["stats"]["decisions_total"] == len(many.decisions)
    assert len(dumps(msg)) < 60_000 < len(dumps(many.to_dict()))
    assert hub_stats_msg(many, 0)["stats"]["decisions"] == []
    assert len(full.decisions) == len(game.final_stats.decisions)  # the snapshot itself is untouched


# --------------------------------------------------------------------------- director config


def test_director_follows_narration_settings_and_mode(result) -> None:
    """Defect: the pipeline always passed ``cfg.director`` to the narrator, replacing the director the
    narrator derives from ``narration.cooldown_s`` / ``narration.min_gap_s`` and its mode: those
    settings had no effect, and accessibility mode lost its read-every-event policy."""
    ad = LiveGame(_cfg("narration.mode=accessibility", "narration.min_gap_s=1.25", "narration.ad_ttl_s=5.0"),
                  SimSource(result, "truth3d", duration_s=0.5), realtime=False, env=NO_KEYS)
    dc = ad.narrator.director.config
    assert (dc.p1_latest_wins, dc.p1_flushes_lower, dc.ttl_p1_s, dc.min_gap_s) == (False, False, 5.0, 1.25)
    assert dc.max_queued[1] > 1
    tuned = LiveGame(_cfg("narration.mode=accessibility", "director.ttl_p0_s=4.5", "narration.cooldown_s.color=2.5"),
                     SimSource(result, "truth3d", duration_s=0.5), realtime=False, env=NO_KEYS)
    tc = tuned.narrator.director.config
    assert tc.ttl_p0_s == 4.5 and tc.p1_latest_wins is False and tc.cooldown_s["color"] == 2.5
    assert director_overrides(load_config().director) == {}
    assert director_overrides(tuned.config.director) == {"ttl_p0_s": 4.5}


# --------------------------------------------------------------------------- CLI


def test_serve_warns_about_flags_the_source_never_reads(tmp_path, capsys) -> None:
    roster, calib = tmp_path / "r.yaml", tmp_path / "c.json"
    roster.write_text("teams: {}\n")
    calib.write_text(json.dumps({}))

    def notes(*argv: str) -> list[str]:
        args = build_parser().parse_args(["serve", *argv])
        return unused_serve_flags(args, load_cli_config(args, serve_flags(args)))

    sim = notes("--roster", str(roster), "--calibration", str(calib), "--detector", "colorblob")
    assert len(sim) == 3 and sim[0].startswith("--roster") and all("--mode video" in n for n in sim[1:])
    assert notes("--mode", "video", "--calibration", str(calib)) == []
    video = notes("--source", "video", "--video", str(calib), "--seconds", "30", "--roster", str(roster))
    assert video == ["--seconds (only the simulator source uses it)"]
    assert notes() == []


async def test_speed_control_is_refused_for_a_live_camera(result) -> None:
    """Defect: ``POST /api/control {"action": "speed"}`` on a camera returned ok and showed the new
    speed in the status while a live source is never paced (nothing changed)."""
    src = SimSource(result, "truth3d", duration_s=0.5)
    src.live = True  # stands in for CameraSource (no device in tests)
    game = LiveGame(_cfg(), src, realtime=False, env=NO_KEYS)
    with pytest.raises(ValueError, match="live camera"):
        await game.controller.control({"action": "speed", "value": 2.0})
    assert game.pacer.speed == 1.0
    status = await game.controller.control({"action": "pause"})  # pause still applies
    assert status["paused"] is True
    file_game = LiveGame(_cfg(), SimSource(result, "truth3d", duration_s=0.5), realtime=False, env=NO_KEYS)
    assert (await file_game.controller.control({"action": "speed", "value": 2.0}))["speed"] == 2.0


async def test_real_time_runs_do_not_hoard_perceived_frames(result) -> None:
    """Defect: every perceived FrameState of a sim video source was kept for the tracking report
    (up to 36 000 frames, ~5 KB each: ~180 MB) also in real-time ``serve``, which never builds one."""
    live = LiveGame(_cfg(), SimSource(result, "video", realtime=True, speed=20.0, duration_s=0.5), env=NO_KEYS)
    await live.run()
    assert live._frames_done > 10 and live.kept_frames == []
    offline = LiveGame(_cfg(), SimSource(result, "video", duration_s=0.5), realtime=False, env=NO_KEYS)
    await offline.run()
    assert len(offline.kept_frames) == offline._frames_done > 10


def test_cli_serve_keeps_the_pre_simulated_game_out_of_gc(monkeypatch, capsys) -> None:
    """Defect: with a simulator source the process holds the whole pre-simulated game (2.3 M
    objects for 48 min); each full GC took ~0.5 s and stalled the event loop that runs the game, the
    hub and the WebSockets. ``serve`` now freezes the objects that exist once the app is built."""
    import gc

    from courtvision import cli

    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "TYPESAFE_API_KEY", "COURTVISION_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    seen: dict = {}

    async def fake_serve(server, open_url):
        seen["frozen"] = gc.get_freeze_count()
        return True

    monkeypatch.setattr(cli, "_serve", fake_serve)
    assert cli.main(["serve", "--seconds", "5", "--tts", "none", "--no-llm", "--jev", "off"]) == 0, capsys.readouterr()
    assert seen["frozen"] > 10_000  # the simulated game is frozen while serving
    assert gc.get_freeze_count() == 0  # ...and released when serving ends
