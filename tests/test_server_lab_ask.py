"""Model lab and Ask endpoints, and the dashboard scripts that use them."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from courtvision.config import load_config
from courtvision.server.studio import create_studio_app

STATIC = Path(__file__).resolve().parents[1] / "courtvision" / "server" / "static"
AUTH = {"Authorization": "Bearer local-test"}


@pytest.fixture
def no_claude(tmp_path, monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_FEDERATION_RULE_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "no-anthropic"))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)


@pytest.fixture
def app(tmp_path, no_claude):
    cfg = load_config(None, ["narration.llm_enabled=false", "narration.tts=none", "jev.mode=off"])
    return create_studio_app(cfg, root=tmp_path / "workspace", auth_token="local-test", lab_runs=tmp_path / "runs")


class FakeMessages:
    def __init__(self, text: str, stop: str = "end_turn"):
        self.text, self.stop, self.calls = text, stop, []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(stop_reason=self.stop, model="claude-opus-5-5",
                               content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=self.text)])


def fake_client(text: str, stop: str = "end_turn"):
    messages = FakeMessages(text, stop)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


SPEC = {"title": "Away vs Home", "summary": "Away lead by two.", "icon": "chart", "tone": "neutral",
        "blocks": [{"type": "duel", "text": None, "tone": None, "stat": None, "stats": ["pts", "reb"], "team": None,
                    "player": None, "id": None, "filter": None, "limit": None, "items": None},
                   {"type": "not-a-block"}]}


# --------------------------------------------------------------------------- lab


def test_lab_reads_an_empty_runs_folder_without_writing(app, tmp_path):
    with TestClient(app) as client:
        r = client.get("/api/lab", headers=AUTH)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["available"] is True
        assert body["status"]["games"] == [] and body["runner"]["running"] is False
        assert set(body["machine"]) >= {"free_pct", "min_free", "watch_free", "jobs"}
        assert client.get("/api/lab/log", headers=AUTH).json()["text"] == ""
    assert not (tmp_path / "runs" / "models" / "registry.json").exists()  # a status read never writes the registry


def test_lab_writes_are_guarded(app):
    with TestClient(app) as client:
        assert client.post("/api/lab/stop", headers=AUTH).status_code == 409  # nothing started here
        assert client.post("/api/lab/promote", json={"version": "../x"}, headers=AUTH).status_code == 400
        assert client.post("/api/lab/promote", json={"version": "s002"}, headers=AUTH).status_code == 404
        assert client.post("/api/lab/run", json={"game": "no-such-game"}, headers=AUTH).status_code == 404
        assert client.post("/api/lab/run", json={"max_steps": 0}, headers=AUTH).status_code == 400
        assert client.post("/api/lab/run", json={}).status_code == 401  # writes need the token


# --------------------------------------------------------------------------- ask


def test_ask_without_credentials_is_unavailable(app):
    with TestClient(app) as client:
        assert client.get("/api/ask", headers=AUTH).json() == {"available": False, "model": None}
        r = client.post("/api/ask", json={"question": "Who is winning?"}, headers=AUTH)
        assert r.status_code == 503 and "not configured" in r.json()["detail"]
        assert client.post("/api/ask", json={"question": ""}, headers=AUTH).status_code == 400


def test_ask_sends_grounded_structured_request_and_returns_clean_card(app):
    client_obj, messages = fake_client(json.dumps(SPEC))
    app.state.asker._client = client_obj
    hub = app.state.workspace.hub
    hub.publish_threadsafe({"type": "meta", "t": 0.0, "teams": {"home": "Warriors", "away": "Lakers"}, "roster": [
        {"player_id": "away-23", "name": "LeBron James", "jersey": "23", "team": "away"}]})
    hub.publish_threadsafe({"type": "stats", "t": 5.0, "stats": {"score": {"home": 2, "away": 4}, "teams": {"away": {"pts": 4, "fga": 3}},
                                                                 "players": {"away-23": {"team": "away", "pts": 4}}}})
    with TestClient(app) as client:
        assert client.get("/api/ask", headers=AUTH).json()["available"] is True
        r = client.post("/api/ask", json={"question": "Compare the teams"}, headers=AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["spec"]["title"] == "Away vs Home" and [b["type"] for b in body["spec"]["blocks"]] == ["duel"]
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5-5" and "tool_choice" not in call and "tools" not in call
    assert call["output_config"]["effort"] == "low" and call["output_config"]["format"]["type"] == "json_schema"
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    prompt = call["messages"][0]["content"]
    ctx = json.loads(prompt.split("CONTEXT\n", 1)[1].split("\n\nQUESTION", 1)[0])
    assert ctx["score"] == {"home": 2, "away": 4} and ctx["player_lines"][0]["name"] == "LeBron James"
    assert prompt.endswith("QUESTION\nCompare the teams")


@pytest.mark.parametrize(("text", "stop", "status"), [("{}", "refusal", 422), ("not json", "end_turn", 502),
                                                       (json.dumps(SPEC), "max_tokens", 502)])
def test_ask_failures_become_plain_errors(app, text, stop, status):
    app.state.asker._client = fake_client(text, stop)[0]
    with TestClient(app) as client:
        assert client.post("/api/ask", json={"question": "Who leads?"}, headers=AUTH).status_code == status


# --------------------------------------------------------------------------- dashboard scripts


SCRIPTS = ("lens.js", "app.js", "studio.js", "genui.js", "lab.js", "shell.js")


def test_every_dashboard_script_is_loaded_self_contained_and_finds_its_elements():
    html = (STATIC / "index.html").read_text()
    ids = set(re.findall(r'id="([^"]+)"', html))
    for name in SCRIPTS:
        assert f'src="static/{name}?v=' in html, name
        text = (STATIC / name).read_text()
        urls = re.findall(r"https?://[^\s'\"<>)]+", text)
        assert all("www.w3.org/2000/svg" in u for u in urls), (name, urls)
        assert ".innerHTML" not in text, name  # server and model text go through textContent only
        used = set(re.findall(r'(?:byId|\$|getElementById)\("([^"]+)"\)', text))
        assert used <= ids, (name, used - ids)


def test_card_commands_match_the_ask_schema():
    from courtvision.server.ask import BLOCK_TYPES, COMMANDS, STAT_KEYS

    js = (STATIC / "genui.js").read_text()
    pattern = re.search(r"const COMMAND = /(.+)/;", js).group(1)
    assert all(re.fullmatch(pattern, c) for c in COMMANDS)
    kinds = set(re.findall(r'"([a-z]+)"', re.search(r"const KINDS = new Set\(\[([^\]]+)\]\)", js).group(1)))
    assert set(BLOCK_TYPES) <= kinds
    stats_block = js[js.index("const STATS = {"):js.index("const GAME = {")]
    assert all(re.search(rf"\b{k}: {{", stats_block) for k in STAT_KEYS)
