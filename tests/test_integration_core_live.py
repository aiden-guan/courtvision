"""LiveGame: hub messages to a subscribed client, held scoreboard, real-time pacing with pause / resume /
speed, credentials (names only), Claude / Jev gating and the prompt-cache pre-warm."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter

import pytest

from courtvision.config import load_config
from courtvision.narration.llm import ClaudeNarrationLLM
from courtvision.pipeline import (
    LiveGame,
    SimSource,
    _prewarm_interval,
    anthropic_credential_source,
    build_jev,
    build_llm,
    typesafe_credential_source,
)
from courtvision.server.hub import Hub, HubConfig
from courtvision.sim.simulator import GameSimulator

SEED = 7
NO_KEYS: dict[str, str] = {}
SECRET = "sk-ant-test-secret-value-0123456789"


def _cfg(*extra: str):
    return load_config(None, ["sim.seconds=60", f"sim.seed={SEED}", "narration.tts=none", *extra])


@pytest.fixture(scope="module")
def result():
    return GameSimulator(_cfg().sim).run()


def _drain(client) -> list[dict]:
    out = []
    while (text := client.get_nowait()) is not None:
        out.append(json.loads(text))
    return out


def _hub() -> Hub:
    return Hub(HubConfig(queue_size=1_000_000, disconnect_on_overflow=False, max_rate_hz={}))


async def test_live_game_feeds_a_hub_client(result) -> None:
    hub = _hub()
    hub.bind_loop()
    client = hub.subscribe()
    game = LiveGame(_cfg(), SimSource(result, "truth3d"), hub=hub, realtime=False, env=NO_KEYS)
    await game.run()
    msgs = _drain(client)
    kinds = Counter(m["type"] for m in msgs)
    print(f"\nhub messages: {dict(kinds)}")
    for needed in ("meta", "status", "frame", "event", "utterance", "stats", "caption"):
        assert kinds[needed] > 0, (needed, kinds)
    assert msgs[0]["type"] == "meta"
    meta = msgs[0]
    assert len(meta["roster"]) == len(result.roster.entries) and meta["landmarks"] and meta["rules"]["ruleset"] == "nba"
    # frames are throttled to pipeline.frame_hz of source time (15 Hz over 60 s)
    assert 850 <= kinds["frame"] <= 905
    frame = next(m for m in msgs if m["type"] == "frame" and m["players"])
    assert len(frame["players"]) == 10 and all(p["name"] for p in frame["players"])
    assert {"period", "game", "shot"} <= set(frame["clock"])
    # every event revision is published at once; utterances carry text and event ids
    assert kinds["event"] >= len(game.events)
    utt = [m["utterance"] for m in msgs if m["type"] == "utterance"]
    assert len(utt) == len(game.transcript) and all(u["text"] for u in utt)
    assert any(u["event_ids"] for u in utt)
    caps = [m for m in msgs if m["type"] == "caption"]
    assert all(c["text"] and c["channel"] == "caption" and c["pts"] is not None for c in caps)
    last_stats = [m for m in msgs if m["type"] == "stats"][-1]
    assert last_stats["stats"]["score"] == {"home": result.truth_score[result.roster.entries[0].team],
                                            "away": result.truth_score[result.roster.entries[-1].team]}
    assert msgs[-1]["type"] == "status" and msgs[-1]["finished"] is True


async def test_frame_messages_keep_the_last_confirmed_scoreboard(result) -> None:
    hub = _hub()
    hub.bind_loop()
    client = hub.subscribe()
    src = SimSource(result, "truth3d", scoreboard="ocr_5hz", duration_s=20.0)
    seen: list[bool] = []
    game = LiveGame(_cfg("pipeline.frame_hz=30"), src, hub=hub, realtime=False, env=NO_KEYS)
    original = game.engine.process

    def spy(frame):
        seen.append(frame.scoreboard is not None)
        return original(frame)

    game.engine.process = spy  # type: ignore[method-assign]
    await game.run()
    frames = [m for m in _drain(client) if m["type"] == "frame"]
    assert sum(seen) == 100 and len(seen) == 600  # the engine only ever sees the fresh 5 Hz reads
    assert all(f["clock"]["period"] == 1 and f["clock"]["game"] is not None for f in frames)
    assert len(frames) >= 590


async def test_realtime_pacing_pause_resume_and_speed(result) -> None:
    cfg = _cfg("pipeline.tail_s=0")
    src = SimSource(result, "truth3d", realtime=True, speed=20.0, duration_s=3.0)
    game = LiveGame(cfg, src, realtime=None, env=NO_KEYS)
    assert game.realtime and game.pacer.enabled
    t0 = time.perf_counter()
    task = asyncio.create_task(game.run())
    await asyncio.sleep(0.04)
    status = await game.controller.control({"action": "pause"})
    assert status["paused"] is True
    await asyncio.sleep(0.05)
    n0 = game._frames_done
    await asyncio.sleep(0.15)
    assert game._frames_done - n0 <= 1  # nothing advances while paused
    await game.controller.control({"action": "resume"})
    status = await game.controller.control({"action": "speed", "value": 40.0})
    assert status["speed"] == 40.0 and status["paused"] is False
    await task
    wall = time.perf_counter() - t0
    assert game._frames_done == 90
    assert wall >= 0.25  # >= 3 s / 40x plus the 0.2 s pause, i.e. frames were really paced
    assert 0.0 < n0 < 90


def test_anthropic_credentials_are_detected_by_name_only(tmp_path) -> None:
    empty = tmp_path / "none"
    assert anthropic_credential_source({}, config_dir=empty) is None
    assert anthropic_credential_source({"ANTHROPIC_API_KEY": SECRET}, config_dir=empty) == "ANTHROPIC_API_KEY"
    assert anthropic_credential_source({"ANTHROPIC_API_KEY": "  "}, config_dir=empty) is None
    assert anthropic_credential_source({"ANTHROPIC_AUTH_TOKEN": SECRET}, config_dir=empty) == "ANTHROPIC_AUTH_TOKEN"
    profiles = tmp_path / "anthropic"
    (profiles / "credentials").mkdir(parents=True)
    (profiles / "credentials" / "work.json").write_text("{}")
    assert anthropic_credential_source({}, config_dir=profiles) == "profile"
    assert anthropic_credential_source({"ANTHROPIC_PROFILE": "work"}, config_dir=profiles) == "profile:work"
    assert anthropic_credential_source({"ANTHROPIC_PROFILE": "other"}, config_dir=profiles) is None
    assert anthropic_credential_source({"ANTHROPIC_CONFIG_DIR": str(profiles)}) == "profile"
    wif = {"ANTHROPIC_FEDERATION_RULE_ID": "r", "ANTHROPIC_ORGANIZATION_ID": "o",
           "ANTHROPIC_SERVICE_ACCOUNT_ID": "s", "ANTHROPIC_IDENTITY_TOKEN_FILE": "/tmp/token"}
    assert anthropic_credential_source(wif, config_dir=empty) == "workload_identity"
    assert typesafe_credential_source({}) is None
    assert typesafe_credential_source({"TYPESAFE_API_KEY": SECRET}) == "TYPESAFE_API_KEY"


def test_claude_only_with_credentials_and_never_logs_them(tmp_path, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    env_none = {"HOME": str(tmp_path)}
    env_key = {"HOME": str(tmp_path), "ANTHROPIC_API_KEY": SECRET}
    cfg = _cfg()
    assert build_llm(cfg, env_none, realtime=True) is None  # no keys => template-only
    llm = build_llm(cfg, env_key, realtime=True)
    assert isinstance(llm, ClaudeNarrationLLM)
    assert llm.config.cache_ttl == "1h" and llm.config.model == cfg.narration.model
    assert llm.config.effort == cfg.narration.effort and llm.config.max_tokens == cfg.narration.max_tokens
    assert build_llm(cfg, env_key, realtime=False) is None  # offline runs race ahead of wall time
    assert isinstance(build_llm(_cfg("llm.offline=true"), env_key, realtime=False), ClaudeNarrationLLM)
    assert build_llm(_cfg("narration.llm_enabled=false"), env_key, realtime=True) is None
    assert build_llm(_cfg("narration.mode=coach"), env_key, realtime=True) is None
    assert SECRET not in caplog.text


def test_jev_falls_back_without_a_key(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    off = build_jev(_cfg().jev, {})
    assert off.client.mode == "off" and off.available is False
    stub = build_jev(_cfg("jev.mode=stub").jev, {})
    assert stub.client.mode == "stub" and stub.available is True
    live = build_jev(_cfg().jev, {"TYPESAFE_API_KEY": SECRET})
    assert live.client.mode == "live"
    assert build_jev(_cfg("jev.enabled=false").jev, {"TYPESAFE_API_KEY": SECRET}).client.mode == "off"
    assert SECRET not in caplog.text


async def test_no_keys_game_is_template_only(result) -> None:
    game = LiveGame(_cfg(), SimSource(result, "truth3d", duration_s=10.0), realtime=False, env=NO_KEYS)
    assert game.llm is None and game.narrator.pool is None
    assert game.jev.client.mode == "off"
    await game.run()
    assert game.status()["llm"] is False and game.status()["jev"] == "off"


class FakeLLM:
    """NarrationLLM + prewarm, no network."""

    def __init__(self) -> None:
        self.prewarms: list[str] = []
        self.streams = 0

    async def prewarm(self, system: str | None = None) -> None:
        self.prewarms.append(system or "")

    def stream(self, system: str, prompt: str, max_tokens: int | None = None):
        self.streams += 1

        async def gen():
            yield "SILENCE"

        return gen()


async def test_claude_prompt_cache_is_prewarmed_at_start(result) -> None:
    fake = FakeLLM()
    cfg = _cfg("pipeline.tail_s=0", "jev.mode=stub")
    game = LiveGame(cfg, SimSource(result, "truth3d", realtime=True, speed=50.0, duration_s=1.0), llm=fake,
                    env=NO_KEYS)
    assert game.narrator.pool is not None
    await game.run()
    assert fake.prewarms and "colour commentator" in fake.prewarms[0]
    assert fake.prewarms[0] == game.narrator.pool.system_prompt  # same byte-stable prefix the candidates use
    assert _prewarm_interval(cfg) == 55 * 60.0
    assert _prewarm_interval(_cfg("llm.cache_ttl=null")) == 4 * 60.0
    assert _prewarm_interval(_cfg("llm.prewarm_interval_s=600")) == 600.0
