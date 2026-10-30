"""Ask: questions about the live game answered as a small dashboard card composed by Claude.

The dashboard answers most questions on the device (static/genui.js). This endpoint takes the ones it cannot place
when Anthropic credentials are configured. Claude gets a compact snapshot of the hub's game state and returns a view
spec through structured output (``output_config.format``; forced tool choice is a 400 on Claude Opus 5.5). Data
blocks name live sources (stat keys, teams, player ids) rather than numbers, so the card the dashboard renders shows
the game data, never figures the model wrote; free text is labelled as composed by Claude. One request runs at a time.

Request shape (anthropic 1.15.0, as in narration/llm.py)::

    client.beta.messages.create(model="claude-opus-5-5", max_tokens=8000, system=..., messages=[...],
                                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SPEC}},
                                betas=["server-side-fallback-2026-07-01"], fallbacks="default")
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from courtvision.pipeline import anthropic_credential_source
from courtvision.server.hub import Hub, dumps

log = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
BLOCK_TYPES = ["text", "note", "metrics", "duel", "leaders", "spark", "shots", "plays", "player", "kv", "steps",
               "actions", "tracking", "decisions", "insights", "narration", "lab"]
STAT_KEYS = ["pts", "fg_pct", "fg3_pct", "fg3m", "ft_pct", "efg", "ts", "reb", "oreb", "ast", "stl", "blk", "tov", "pf",
             "poss", "ppp", "pace", "xpts", "avg_xfg", "plus_minus"]
GAME_KEYS = ["score_home", "score_away", "period", "game_clock", "margin", "events", "review"]
COMMANDS = ["nav:live", "nav:stats", "nav:board", "nav:operator", "nav:calibration", "nav:roi", "nav:lab",
            "footage:load", "footage:demo", "footage:export", "court:map", "plays:review", "plays:add",
            "playback:pause", "playback:resume", "lab:refresh", "lab:run", "lab:copy-next"]
ICONS = ["spark", "alert", "check", "info", "court", "ball", "people", "chart", "flame", "review", "clock", "lab", "film",
         "trophy", "bolt"]
TONES = ["neutral", "good", "warn", "bad", "ember", "info"]


def _opt(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


_ITEM = {
    "type": "object", "additionalProperties": False,
    "required": ["label", "value", "sub", "stat", "team", "player", "game", "title", "text", "state", "command", "primary"],
    "properties": {
        "label": _opt({"type": "string"}), "value": _opt({"type": "string"}), "sub": _opt({"type": "string"}),
        "stat": _opt({"type": "string", "enum": STAT_KEYS}), "team": _opt({"type": "string", "enum": ["home", "away"]}),
        "player": _opt({"type": "string"}), "game": _opt({"type": "string", "enum": GAME_KEYS}),
        "title": _opt({"type": "string"}), "text": _opt({"type": "string"}),
        "state": _opt({"type": "string", "enum": ["done", "now", "todo"]}),
        "command": _opt({"type": "string", "enum": COMMANDS}), "primary": _opt({"type": "boolean"}),
    },
}
_BLOCK = {
    "type": "object", "additionalProperties": False,
    "required": ["type", "text", "tone", "stat", "stats", "team", "player", "id", "filter", "limit", "items"],
    "properties": {
        "type": {"type": "string", "enum": BLOCK_TYPES},
        "text": _opt({"type": "string"}), "tone": _opt({"type": "string", "enum": TONES}),
        "stat": _opt({"type": "string", "enum": STAT_KEYS}),
        "stats": _opt({"type": "array", "items": {"type": "string", "enum": STAT_KEYS}}),
        "team": _opt({"type": "string", "enum": ["home", "away"]}), "player": _opt({"type": "string"}),
        "id": _opt({"type": "string"}), "filter": _opt({"type": "string", "enum": ["key", "all", "review", "scoring"]}),
        "limit": _opt({"type": "integer"}), "items": _opt({"type": "array", "items": _ITEM}),
    },
}
SPEC_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["title", "summary", "icon", "tone", "blocks"],
    "properties": {
        "title": {"type": "string"}, "summary": {"type": "string"},
        "icon": {"type": "string", "enum": ICONS}, "tone": {"type": "string", "enum": TONES},
        "blocks": {"type": "array", "items": _BLOCK},
    },
}

SYSTEM_PROMPT = """You compose answer cards for CourtVision Studio, a local basketball video-analysis dashboard. The user asks a question about the game being analysed; you answer with one card: a short title, a one- or two-sentence summary, and two to five blocks that the dashboard renders from live data.

Ground every word in the CONTEXT JSON of the request. It holds the team names, the score and clock, team and player box-score lines, recent plays, tracking status and counts. Do not state a number, name, play or circumstance that the context does not contain. If the context cannot answer the question, say so plainly in the summary and offer the closest blocks that help (for example the plays list or tracking status). The numbers in this data are computer-vision estimates; when the question is about accuracy or official statistics, say they need review.

Prefer live data blocks over writing numbers in text: the dashboard fills them from the newest game state, so they stay correct as the game runs. Write numbers in the summary only when they appear in the context.

Blocks (fields not listed for a block are null):
- metrics: items, each one of {stat + team} (a team stat), {stat + player} (a player stat), {game} (a game value), or {label + value + sub} (only for values copied from the context).
- duel: stats, a list of stat keys compared between the teams.
- leaders: stat (a player stat key), optional team, limit (players to list).
- spark: the score margin over the analysed video.
- shots: the shot chart; optional team or player.
- plays: the play list; filter (key, all, review, scoring), optional team or player, limit.
- player: id, a player id from the context; a live card for that player.
- kv: items of {label, value} copied from the context.
- steps: items of {title, text, state} for a how-to; state is done, now or todo.
- actions: items of {label, command, primary} buttons; command must be one of the listed commands.
- note: text and tone, a one-sentence caveat or callout.
- text: text, a short paragraph.
- tracking: detector and tracker status. decisions, insights, narration: recent items, with limit. lab: the model-training loop summary.

Stat keys: pts, fg_pct, fg3_pct, fg3m, ft_pct, efg, ts, reb, oreb, ast, stl, blk, tov, pf, poss, ppp, pace, xpts, avg_xfg, plus_minus (players only). Game keys: score_home, score_away, period, game_clock, margin, events, review.

Use player ids exactly as they appear in the context. Keep the title under eight words and in sentence case. Write plainly, without hype."""


class AskIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=300)


def _r(v: Any, nd: int = 3) -> Any:
    return round(v, nd) if isinstance(v, float) else v


def game_context(hub: Hub, studio: dict[str, Any] | None = None) -> dict[str, Any]:
    """A compact, JSON-safe snapshot of what the dashboard shows (team and player lines, recent plays, status)."""
    meta = hub.latest("meta") or {}
    stats = (hub.latest("stats") or {}).get("stats") or {}
    roster = {e.get("player_id"): e for e in meta.get("roster") or [] if isinstance(e, dict)}
    team_keys = ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf",
                 "efg", "ts", "pace", "ppp", "possessions", "possessions_est", "xpts", "avg_xfg")
    teams = {t: {k: _r(v) for k, v in (line or {}).items() if k in team_keys}
             for t, line in (stats.get("teams") or {}).items() if t in ("home", "away")}
    players = []
    for pid, line in (stats.get("players") or {}).items():
        if not isinstance(line, dict):
            continue
        r = roster.get(pid) or {}
        players.append({"id": pid, "name": r.get("name"), "jersey": r.get("jersey"), "team": line.get("team") or r.get("team"),
                        **{k: _r(line.get(k)) for k in ("pts", "fgm", "fga", "fg3m", "fg3a", "reb", "oreb", "dreb", "ast",
                                                         "stl", "blk", "tov", "pf", "xpts", "plus_minus") if line.get(k) is not None}})
    players.sort(key=lambda p: -(p.get("pts") or 0))
    events = [e for e in hub.events() if isinstance(e, dict)]
    review = sum(1 for e in events if (e.get("attrs") or {}).get("needs_review") and e.get("status") != "retracted")
    recent = [{"t": _r(e.get("t"), 1), "kind": e.get("kind"), "team": e.get("team"), "player": e.get("player_id"),
               "other": e.get("other_player_id"), "value": e.get("value"), "status": e.get("status"),
               "confidence": _r(e.get("confidence"), 2), "needs_review": bool((e.get("attrs") or {}).get("needs_review"))}
              for e in sorted(events, key=lambda e: e.get("t") or 0)[-30:] if e.get("kind") not in ("dribble", "pass")]
    ctx: dict[str, Any] = {
        "teams": meta.get("teams") or {"home": "Home", "away": "Away"},
        "score": stats.get("score"), "score_source": stats.get("score_source"), "period": stats.get("period"),
        "game_clock": stats.get("game_clock"), "run": stats.get("run"), "team_lines": teams, "player_lines": players[:16],
        "roster": [{"id": pid, "name": r.get("name"), "jersey": r.get("jersey"), "team": r.get("team")} for pid, r in roster.items()][:40],
        "plays_logged": len(events), "plays_needing_review": review, "recent_plays": recent,
        "shots_logged": len(stats.get("shots") or []),
    }
    if studio:
        vision = studio.get("vision") or {}
        status = studio.get("status") or {}
        active = studio.get("active") or {}
        ctx["session"] = {"footage": active.get("name"), "demo": bool(active.get("demo")), "duration_s": active.get("duration_s"),
                          "people_tracked": vision.get("people"), "ball_visible": vision.get("ball_visible"),
                          "court_mapped": vision.get("calibrated"), "fps": vision.get("fps"), "paused": status.get("paused"),
                          "finished": status.get("finished"), "needs_court_mapping": status.get("needs_calibration")}
    return json.loads(dumps(ctx))


def _clip(spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict) or not isinstance(spec.get("blocks"), list):
        raise ValueError("no card in the response")
    out = {k: str(spec.get(k) or "")[:400] for k in ("title", "summary", "icon", "tone")}
    out["blocks"] = [b for b in spec["blocks"][:8] if isinstance(b, dict) and b.get("type") in BLOCK_TYPES]
    return out


class Asker:
    def __init__(self, hub: Hub, studio_state: Any = None, client: Any = None) -> None:
        self.hub = hub
        self.studio_state = studio_state
        self._client = client
        self.lock = asyncio.Lock()

    def available(self) -> bool:
        return self._client is not None or anthropic_credential_source() is not None

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic  # lazy: tests and on-device answers never import the SDK client

            self._client = anthropic.AsyncAnthropic(max_retries=1, timeout=90.0)
        return self._client

    async def ask(self, question: str) -> dict[str, Any]:
        if not self.available():
            raise HTTPException(503, "Claude is not configured. Add an Anthropic key (scripts/mac/courtvision.sh keys) and restart the Studio.")
        if self.lock.locked():
            raise HTTPException(429, "Another answer is being composed. Try again in a moment.")
        async with self.lock:
            studio = self.studio_state() if callable(self.studio_state) else None
            ctx = game_context(self.hub, studio)
            started = time.perf_counter()
            try:
                resp = await self.client.beta.messages.create(
                    model=MODEL, max_tokens=8000, system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": f"CONTEXT\n{json.dumps(ctx, separators=(',', ':'))}\n\nQUESTION\n{question.strip()}"}],
                    output_config={"effort": "low", "format": {"type": "json_schema", "schema": SPEC_SCHEMA}},
                    betas=[FALLBACK_BETA], fallbacks="default",
                )
            except Exception as err:  # noqa: BLE001 - every SDK failure becomes one plain message for the dashboard
                log.warning("ask: Claude request failed: %s", err)
                raise HTTPException(502, "Claude could not be reached. The on-device answer is shown instead.") from err
            if getattr(resp, "stop_reason", None) == "refusal":
                raise HTTPException(422, "Claude declined this question.")
            if getattr(resp, "stop_reason", None) == "max_tokens":
                raise HTTPException(502, "Claude's answer was cut short. Try a narrower question.")
            text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
            try:
                spec = _clip(json.loads(text))
            except (ValueError, TypeError) as err:
                raise HTTPException(502, "Claude returned no usable card.") from err
            ms = round((time.perf_counter() - started) * 1000)
            log.info("ask: model=%s blocks=%d %d ms", getattr(resp, "model", MODEL), len(spec["blocks"]), ms)
            return {"spec": spec, "model": getattr(resp, "model", MODEL), "ms": ms}


def register_ask(app: FastAPI, asker: Asker, read: Any, write: Any) -> None:
    @app.get("/api/ask", dependencies=[read])
    async def ask_state() -> Any:
        return {"available": asker.available(), "model": MODEL if asker.available() else None}

    @app.post("/api/ask", dependencies=[write])
    async def ask(body: AskIn) -> Any:
        return await asker.ask(body.question)
