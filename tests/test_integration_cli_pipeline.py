"""Pipeline hooks the CLI relies on: an injected detector (``--detector rnd-yolo:...``), a non-draining
stop for server shutdown, and ``pipeline.load_roster`` (now ``courtvision.roster``) staying
backward compatible with the flat roster form."""

from __future__ import annotations

import asyncio
import json
import time

from courtvision.config import load_config
from courtvision.narration.llm import ClaudeNarrationLLM
from courtvision.perception.detector import ColorBlobDetector
from courtvision.pipeline import LiveGame, SimSource, build_llm, load_roster, make_source, run_offline
from courtvision.sim.simulator import GameSimulator
from courtvision.types import Team

NO_KEYS: dict[str, str] = {}
SECRET = "sk-ant-test-secret-value-0123456789"


def _cfg(*extra: str):
    return load_config(None, ["sim.seconds=8", "sim.seed=7", "narration.tts=none", "jev.mode=off", *extra])


class CountingDetector:
    """A Detector built outside the pipeline (as the CLI builds the R&D adapter)."""

    def __init__(self) -> None:
        self.inner = ColorBlobDetector()
        self.calls = 0

    def detect(self, frame_bgr, frame_idx, t):
        self.calls += 1
        return self.inner.detect(frame_bgr, frame_idx, t)


def test_an_injected_detector_replaces_the_configured_one() -> None:
    cfg = _cfg("source.sim_mode=video", "source.duration_s=1", "perception.detector=rfdetr")
    det = CountingDetector()
    result = run_offline(cfg, make_source(cfg, realtime=False), write=False, env=NO_KEYS, detector=det)
    assert det.calls == 30 and result.frames == 30  # rfdetr (configured) was never built
    assert result.latency["perception_ms_per_frame"]["n"] == 30


def test_request_stop_without_drain_skips_the_narration_tail() -> None:
    sim = GameSimulator(_cfg().sim).run()

    async def scenario(drain: bool) -> tuple[list[float], float]:
        cfg = _cfg("pipeline.tail_s=30")
        game = LiveGame(cfg, SimSource(sim, "truth3d", realtime=True, speed=8.0), realtime=True, env=NO_KEYS)
        drains: list[float] = []
        original = game.narrator.drain

        async def spy(timeout: float = 10.0) -> bool:
            drains.append(timeout)
            return await original(timeout=timeout)

        game.narrator.drain = spy  # type: ignore[method-assign]
        task = asyncio.create_task(game.run())
        await asyncio.sleep(0.3)
        t0 = time.perf_counter()
        game.request_stop(drain=drain)
        await asyncio.wait_for(task, 40)
        assert game._finished and game._frames_done < len(sim.frames)
        return drains, time.perf_counter() - t0

    drains, took = asyncio.run(scenario(drain=False))
    assert drains == [] and took < 2.0  # server shutdown: no 30 s narration tail
    drains, _ = asyncio.run(scenario(drain=True))
    assert drains == [30.0]  # the default stop still lets narration finish


def test_pipeline_load_roster_keeps_the_flat_form(tmp_path) -> None:
    flat = {"teams": {"home": "Hawks", "away": "Owls"},
            "players": [{"player_id": "home-23", "team": "home", "jersey": "23", "name": "A"},
                        {"team": "away", "jersey": "00", "name": "B"}]}
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(flat))
    roster = load_roster(path, away_name="Visitors")
    assert [(e.player_id, e.jersey) for e in roster.entries] == [("home-23", "23"), ("away-00", "00")]
    assert roster.team_names == {Team.HOME: "Hawks", Team.AWAY: "Visitors"}
    empty = load_roster(None)
    assert empty.entries == [] and empty.team_names[Team.HOME] == "Home"


def test_claude_is_not_built_when_the_jev_line_guard_cannot_run(tmp_path, caplog) -> None:
    """D4: Claude lines are spoken only after a passing Jev guard. With an Anthropic key but no Jev,
    building Claude (and pre-warming its cache every 55 min) would only cost money."""
    env = {"HOME": str(tmp_path), "ANTHROPIC_API_KEY": SECRET}
    sim = GameSimulator(_cfg("sim.seconds=2").sim).run()
    cfg = _cfg("jev.mode=live")  # live without TYPESAFE_API_KEY -> off
    game = LiveGame(cfg, SimSource(sim, "truth3d", realtime=True), realtime=True, env=env)
    assert game.jev.available is False and game.llm is None and game.narrator.pool is None
    assert "need the Jev line guard" in caplog.text and SECRET not in caplog.text
    stub = LiveGame(_cfg("jev.mode=stub"), SimSource(sim, "truth3d", realtime=True), realtime=True, env=env)
    assert isinstance(stub.llm, ClaudeNarrationLLM) and stub.narrator.pool is not None
    # the guard can be switched off explicitly (against D4); then Claude is built without Jev
    assert isinstance(build_llm(_cfg("narration.use_jev_validator=false"), env, True, jev_available=False),
                      ClaudeNarrationLLM)
    assert build_llm(cfg, env, True) is not None  # callers that do not say keep the old behaviour
