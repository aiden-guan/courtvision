"""CourtVision cloud latency harness (RESEARCH.md §7A.3; narration.md §11.3).

Measures the configurations that set every cloud deadline:

=====  ==================  ===========================================================
C1     claude-opus-5-5     effort low, fallbacks="default", max_tokens=1536
C2     claude-opus-5-5     C1 plus the system line "Answer directly without deliberating."
C3     claude-haiku-5-5    effort low, thinking={"type": "disabled"}, no fallbacks
C4     claude-haiku-5-5    effort low, adaptive thinking
J1-J5  jev-1.13.0          Jev bundles B1-B5 with real-shaped states
=====  ==================  ===========================================================

Each Claude call sends a real-shaped FactPacket behind the byte-stable cached system prefix
(pre-warmed with a non-streaming ``max_tokens=0`` request) and records time to ``message_start``,
time to the first text delta, total time, ``usage.output_tokens`` (includes thinking),
``usage.cache_read_input_tokens``, ``stop_reason`` and ``response.model``. Each Jev call records
wall time, ``resp.model``, ``usage.input_tokens`` and the question count. Output: a JSON file with
every record plus a markdown p50/p95 table per configuration and network label. Every deadline is
then set from the measured p95 (``candidate_deadline_s`` and each Jev bundle's deadline).

The harness refuses to run without credentials (Anthropic: API key, auth token or ``ant auth``
profile; TypeSafe: ``TYPESAFE_API_KEY``). It makes ~50 real, billed calls per configuration.

    .venv/bin/python tools/latency_harness.py --configs C1,C2,C3,C4,J1,J2,J3,J4,J5 --n 50 \\
        --network dev-mac --out runs/latency
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # run as a script from the repo root
    sys.path.insert(0, str(ROOT))

from courtvision.decision import questions as Q  # noqa: E402
from courtvision.decision import states as S  # noqa: E402
from courtvision.decision.client import JEV_MODEL_TEXT, sanitize  # noqa: E402
from courtvision.geometry.court import court_for  # noqa: E402
from courtvision.narration.facts import build_facts  # noqa: E402
from courtvision.narration.llm import (ClaudeConfig, ClaudeNarrationLLM, NarrationLLMError,  # noqa: E402
                                       build_user_prompt)
from courtvision.narration.pseudonym import PseudonymMap  # noqa: E402
from courtvision.types import Event, EventKind, EventStatus, Roster, RosterEntry, Team  # noqa: E402

log = logging.getLogger("latency_harness")

CLAUDE_CONFIGS: dict[str, ClaudeConfig] = {
    "C1": ClaudeConfig(model="claude-opus-5-5", effort="low"),
    "C2": ClaudeConfig(model="claude-opus-5-5", effort="low", answer_directly=True),
    "C3": ClaudeConfig(model="claude-haiku-5-5", effort="low", disable_thinking=True),
    "C4": ClaudeConfig(model="claude-haiku-5-5", effort="low"),
}
JEV_CONFIGS: dict[str, str] = {"J1": "B1", "J2": "B2", "J3": "B3", "J4": "B4", "J5": "B5"}
ALL_CONFIGS = (*CLAUDE_CONFIGS, *JEV_CONFIGS)
# Output price per million tokens (thinking is billed as output): Opus 5.5 $20, Haiku 5.5 $0.50 (§7A.3); sourced
OUTPUT_USD_PER_MTOK = {"claude-opus-5-5": 20.0, "claude-haiku-5-5": 0.50}


class CredentialsMissing(RuntimeError):
    """The harness makes real calls and will not run without credentials."""


@dataclass
class HarnessConfig:
    configs: list[str] = field(default_factory=lambda: list(ALL_CONFIGS))
    n_calls: int = 50  # ~50 calls per configuration (§7A.3); sourced
    network: str = "dev-mac"  # label: dev-mac | gym-wifi | lte-hotspot ...
    out_dir: Path = Path("runs/latency")
    packet_path: Path | None = None  # a real FactPacket JSON; default: the built-in real-shaped packet
    pause_s: float = 0.5  # spacing between calls (avoids measuring our own rate limits); tuned
    jev_model: str = JEV_MODEL_TEXT  # pinned (D2)
    jev_timeout_s: float = 30.0  # measure the tail, do not cut it
    prewarm: bool = True


# --------------------------------------------------------------------------- real-shaped inputs

_HOME = ["Avery Lindqvist", "Jordan Okafor", "Casey Marchetti", "Riley Thorsen", "Morgan Vasquez"]
_AWAY = ["Devon Achterberg", "Skyler Nakamura", "Emerson Duquesne", "Rowan Castellano", "Parker Ilunga"]
_JERSEYS = ["0", "3", "11", "23", "5"]


def sample_roster() -> Roster:
    """Fictional roster (never real people)."""
    entries = [RosterEntry(f"home-{j}", Team.HOME, j, n) for j, n in zip(_JERSEYS, _HOME)]
    entries += [RosterEntry(f"away-{j}", Team.AWAY, j, n) for j, n in zip(_JERSEYS, _AWAY)]
    return Roster(entries, {Team.HOME: "Riverside", Team.AWAY: "Lakeport"})


@dataclass
class _Line:
    player_id: str
    team: Team
    pts: int = 0
    fgm: int = 0
    fga: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    oreb: int = 0
    dreb: int = 0
    ast: int = 0
    stl: int = 0
    blk: int = 0
    tov: int = 0
    pf: int = 0


@dataclass
class _Stats:
    t: float = 1890.0
    period: int = 3
    game_clock: float = 312.0
    shot_clock: float = 24.0
    score: dict = field(default_factory=lambda: {Team.HOME: 61, Team.AWAY: 55})
    score_source: str = "scoreboard"
    players: dict = field(default_factory=dict)
    teams: dict = field(default_factory=dict)
    shots: list = field(default_factory=list)
    run: dict = field(default_factory=lambda: {"team": Team.HOME, "points": 9, "opp_points": 0, "since_t": 1790.0})


def sample_facts() -> dict[str, Any]:
    """A FactPacket shaped like a real timeout packet (several events, players, run, storylines)."""
    roster = sample_roster()
    lines = {}
    for i, e in enumerate(roster.entries):
        lines[e.player_id] = _Line(e.player_id, e.team, pts=4 + 3 * i, fgm=2 + i, fga=5 + i, fg3m=i % 3, fg3a=2 + i % 4,
                                   ftm=i % 4, fta=i % 5, oreb=i % 3, dreb=2 + i % 5, ast=1 + i % 4, stl=i % 2,
                                   blk=i % 2, tov=1 + i % 3, pf=1 + i % 4)
    stats = _Stats(players=lines, teams={Team.HOME: _Line("home", Team.HOME, pts=61, fga=48, fgm=24, tov=8, pf=12),
                                         Team.AWAY: _Line("away", Team.AWAY, pts=55, fga=50, fgm=21, tov=11, pf=14)})
    events = [
        Event("e000096", EventKind.STEAL, 1852.0, 55560, Team.HOME, "home-0", "away-3", x=60.0, y=30.0,
              status=EventStatus.CONFIRMED),
        Event("e000097", EventKind.TURNOVER, 1852.0, 55560, Team.AWAY, "away-3", attrs={"type": "bad_pass"},
              status=EventStatus.CONFIRMED),
        Event("e000098", EventKind.SHOT_MADE, 1855.0, 55650, Team.HOME, "home-0", value=2, shot_id="s0058",
              x=6.0, y=25.0, status=EventStatus.CONFIRMED, attrs={"shot_type": "layup", "basket": "left"}),
        Event("e000099", EventKind.SHOT_MISSED, 1866.0, 55980, Team.AWAY, "away-11", value=3, shot_id="s0059",
              x=70.0, y=45.0, status=EventStatus.CONFIRMED, attrs={"shot_type": "jumper", "basket": "right"}),
        Event("e000100", EventKind.REBOUND, 1867.0, 56010, Team.HOME, "home-5", attrs={"offensive": False},
              status=EventStatus.CONFIRMED),
        Event("e000101", EventKind.SHOT_MADE, 1880.0, 56400, Team.HOME, "home-23", value=3, shot_id="s0060",
              x=24.0, y=6.0, status=EventStatus.CONFIRMED, source="cv",
              attrs={"shot_type": "jumper", "basket": "left", "distance_ft": 23.4}),
        Event("e000102", EventKind.ASSIST, 1880.0, 56400, Team.HOME, "home-3", "home-23", shot_id="s0060",
              status=EventStatus.CONFIRMED),
        Event("e000103", EventKind.TIMEOUT, 1884.0, 56520, Team.AWAY, status=EventStatus.CONFIRMED, source="manual"),
    ]
    return build_facts(events, stats, roster, court_for("nba"), None, PseudonymMap(roster), state_version=4242,
                       recent_lines=["{P07} buries the three!", "Timeout, Lakeport.",
                                     "Riverside has scored nine straight."],
                       storylines=["Avery Lindqvist reaches 20 points for the third time this season"],
                       labels={"last_play": "spot up"})


def jev_cases(facts: Mapping[str, Any]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Real-shaped (state, questions) per bundle, built with the shipped state builders."""
    roster = sample_roster()
    names = [e.name for e in roster.entries]
    sit = S.GameSituation(home="Riverside", away="Lakeport", home_score=61, away_score=55, period=3, clock_s=312.0,
                          run_team="Riverside", run_points=9, run_opp_points=0)
    evs = [Event("e1", EventKind.POSSESSION, 1860.0, 55800, Team.HOME, "home-3", attrs={"cause": "defensive_rebound"}),
           Event("e2", EventKind.PASS, 1866.0, 55980, Team.HOME, "home-3", "home-23", attrs={"completed": True}),
           Event("e3", EventKind.SHOT_ATTEMPT, 1879.0, 56370, Team.HOME, "home-23", value=3, shot_id="s0060",
                 attrs={"shot_type": "jumper", "distance_ft": 23.4, "catch_and_shoot": True, "dribbles_before": 0}),
           Event("e4", EventKind.SHOT_MADE, 1880.0, 56400, Team.HOME, "home-23", value=3, shot_id="s0060")]
    poss = S.summarize_possession(evs, Team.HOME, "Riverside", game_clock_at_end_s=312.0)
    b1 = S.b1_possession_end(sit, poss, names=names)
    stories = ["{P07} reaches 20 points for the third time this season"]
    b2 = S.b2_dead_ball_fan(sit, stoppage=S.stoppage_text("timeout", "Lakeport"),
                            material={"stat_note": "{P07} is 7-for-11 from the field"}, storylines=stories,
                            dead_ball_id="w1", names=names)
    cards = ["{P12}: 4 fouls, Q3 (Q+1).", "Riverside 9-0 run, 2:41.", "Lakeport eFG 41% on 50 shots.",
             "Riverside 8 offensive rebounds, n = 26 misses."]
    b3, _keys = S.b3_dead_ball_coach(sit, cards, seed="w1", staff_team="Lakeport", names=names)
    lines = ["{P07} has 19 points tonight.", "Riverside is on a 9-0 run."]
    b4 = S.b4_line_guard(facts, lines, names=names)
    b5 = S.b5_event_ambiguity("shot_outcome", {"drop_speed_frac": 0.55, "net_motion_z": 2.1, "ball_in_basket_frames": 1,
                                               "rim_contact": True, "view": "rim partly blocked by a player"})
    return {
        "B1": (b1, Q.B1_POSSESSION_END),
        "B2": (b2, Q.b2_dead_ball_fan({"stat_note", "storyline"}, len(stories))),
        "B3": (b3, Q.b3_dead_ball_coach(b3["cards"])),
        "B4": (b4, Q.b4_line_guard(len(lines))),
        "B5": (b5, Q.B5_KINDS["shot_outcome"][1]),
    }


# --------------------------------------------------------------------------- credentials


def claude_credentials_available(client: Any = None) -> bool:
    """True when the Anthropic client resolved an API key, auth token or credentials provider."""
    if client is None:
        import anthropic

        client = anthropic.AsyncAnthropic()
    return bool(getattr(client, "api_key", None) or getattr(client, "auth_token", None)
                or getattr(client, "credentials", None))


def jev_credentials_available() -> bool:
    return bool(os.environ.get("TYPESAFE_API_KEY"))


# --------------------------------------------------------------------------- measurement


async def measure_claude(name: str, cfg: ClaudeConfig, client: Any, facts: Mapping[str, Any], n: int, *,
                         pause_s: float = 0.0, prewarm: bool = True) -> dict[str, Any]:
    """``n`` streamed calls of one Claude configuration (after an optional cache pre-warm)."""
    llm = ClaudeNarrationLLM(cfg, client)
    system = llm.system_prompt
    prompt = build_user_prompt(facts, kind="stat_note", max_words=35, max_sentences=2)
    out: dict[str, Any] = {"config": name, "family": "claude", "model": cfg.model, "effort": cfg.effort,
                           "settings": {"answer_directly": cfg.answer_directly, "disable_thinking": cfg.disable_thinking,
                                        "fallbacks": "betas" in _kwargs_for(cfg)},
                           "prompt_chars": len(prompt), "system_chars": len(system)}
    out["system_tokens"] = await _count_system_tokens(client, cfg, system)
    if prewarm:
        out["prewarm"] = (await llm.prewarm(system)).to_dict()
    records = []
    for i in range(n):
        try:
            async for _ in llm.stream(system, prompt, cfg.max_tokens):
                pass
        except NarrationLLMError as err:
            log.info("%s call %d: %s", name, i, err.reason)
        call = llm.last_call
        rec = call.to_dict() if call is not None else {"error": "no call"}
        rec.pop("text", None)
        records.append(rec)
        if pause_s and i + 1 < n:
            await asyncio.sleep(pause_s)
    out["records"] = records
    return out


def _kwargs_for(cfg: ClaudeConfig) -> dict[str, Any]:
    from courtvision.narration.llm import build_stream_kwargs

    return build_stream_kwargs(cfg, "s", "p")


async def _count_system_tokens(client: Any, cfg: ClaudeConfig, system: str) -> int | None:
    """Verify the cached prefix clears the 512-token minimum (None when the client cannot count)."""
    counter = getattr(getattr(client, "messages", None), "count_tokens", None)
    if counter is None:
        return None
    try:
        res = await counter(model=cfg.model, system=system, messages=[{"role": "user", "content": "x"}])
        return int(res.input_tokens)
    except Exception as err:  # noqa: BLE001 - optional check
        log.warning("count_tokens failed for %s: %r", cfg.model, err)
        return None


async def measure_jev(name: str, bundle: str, client: Any, state: Mapping[str, Any], questions: Mapping[str, Any],
                      n: int, *, model: str = JEV_MODEL_TEXT, pause_s: float = 0.0,
                      clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """``n`` raw ``system_one`` calls of one bundle (no deadline, so the tail is measured)."""
    wire = sanitize(state)
    records = []
    for i in range(n):
        t0 = clock()
        rec: dict[str, Any] = {"n_questions": len(questions)}
        try:
            resp = await client.system_one(wire, questions, model=model)
            rec.update(wall_s=clock() - t0, model=getattr(resp, "model", None),
                       input_tokens=getattr(getattr(resp, "usage", None), "input_tokens", None),
                       output_tokens=getattr(getattr(resp, "usage", None), "output_tokens", None))
        except Exception as err:  # noqa: BLE001 - record and continue
            rec.update(wall_s=clock() - t0, error=type(err).__name__)
        records.append(rec)
        if pause_s and i + 1 < n:
            await asyncio.sleep(pause_s)
    return {"config": name, "family": "jev", "bundle": bundle, "model": model,
            "state_bytes": len(json.dumps(wire, separators=(",", ":"))), "records": records}


# --------------------------------------------------------------------------- summaries


def _pct(values: Sequence[float | None], q: float) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return float(np.percentile(vals, q)) if vals else None


def summarize(result: Mapping[str, Any]) -> dict[str, Any]:
    """p50/p95 per metric for one configuration's records."""
    recs = result["records"]
    ok = [r for r in recs if not r.get("error")]
    out: dict[str, Any] = {"config": result["config"], "family": result["family"], "model": result.get("model"),
                           "n": len(recs), "errors": dict(Counter(str(r.get("error")) for r in recs if r.get("error")))}
    if result["family"] == "claude":
        metrics = ("t_message_start_s", "t_first_text_s", "total_s", "output_tokens", "cache_read_input_tokens")
        for m in metrics:
            out[m] = {"p50": _pct([r.get(m) for r in ok], 50), "p95": _pct([r.get(m) for r in ok], 95)}
        out["stop_reasons"] = dict(Counter(str(r.get("stop_reason")) for r in recs))
        out["models_seen"] = sorted({str(r.get("model")) for r in recs if r.get("model")})
        price = OUTPUT_USD_PER_MTOK.get(str(result.get("model")), 0.0)
        out["output_cost_usd_per_call"] = {k: (v * price / 1e6 if v is not None else None)
                                           for k, v in out["output_tokens"].items()}
        out["system_tokens"] = result.get("system_tokens")
    else:
        out["wall_s"] = {"p50": _pct([r.get("wall_s") for r in ok], 50), "p95": _pct([r.get("wall_s") for r in ok], 95)}
        out["input_tokens"] = {"p50": _pct([r.get("input_tokens") for r in ok], 50),
                               "p95": _pct([r.get("input_tokens") for r in ok], 95)}
        out["n_questions"] = recs[0].get("n_questions") if recs else None
        out["models_seen"] = sorted({str(r.get("model")) for r in recs if r.get("model")})
        out["bundle"] = result.get("bundle")
    return out


def recommended_deadlines(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every deadline comes from the measured p95 (§7A.3)."""
    rec: dict[str, Any] = {"claude_candidate_deadline_s": {}, "jev_deadlines_s": {}}
    for s in summaries:
        if s["family"] == "claude" and s["total_s"]["p95"] is not None:
            rec["claude_candidate_deadline_s"][s["config"]] = round(s["total_s"]["p95"], 2)
        elif s["family"] == "jev" and s["wall_s"]["p95"] is not None:
            rec["jev_deadlines_s"][s["bundle"]] = round(s["wall_s"]["p95"], 3)
    return rec


def _fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def markdown_table(summaries: Sequence[Mapping[str, Any]], network: str) -> str:
    """Markdown p50/p95 tables (Claude and Jev) for one network label."""
    lines = [f"# Latency harness: {network}", "",
             "| Config | Model | n | errors | message_start p50/p95 (s) | first text p50/p95 (s) | total p50/p95 (s) "
             "| output tokens p50/p95 | cache read p50 | stop reasons | models |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        if s["family"] != "claude":
            continue
        pp = lambda m, nd=2: f"{_fmt(s[m]['p50'], nd)} / {_fmt(s[m]['p95'], nd)}"  # noqa: E731
        lines.append(f"| {s['config']} | {s['model']} | {s['n']} | {sum(s['errors'].values())} | "
                     f"{pp('t_message_start_s')} | {pp('t_first_text_s')} | {pp('total_s')} | "
                     f"{pp('output_tokens', 0)} | {_fmt(s['cache_read_input_tokens']['p50'], 0)} | "
                     f"{', '.join(f'{k}: {v}' for k, v in s['stop_reasons'].items())} | {', '.join(s['models_seen'])} |")
    lines += ["", "| Config | Bundle | Model | n | errors | wall p50/p95 (s) | input tokens p50 | questions | models |",
              "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        if s["family"] != "jev":
            continue
        lines.append(f"| {s['config']} | {s['bundle']} | {s['model']} | {s['n']} | {sum(s['errors'].values())} | "
                     f"{_fmt(s['wall_s']['p50'], 3)} / {_fmt(s['wall_s']['p95'], 3)} | "
                     f"{_fmt(s['input_tokens']['p50'], 0)} | {s['n_questions']} | {', '.join(s['models_seen'])} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- runner


def _make_jev_client(cfg: HarnessConfig) -> Any:
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

    return AsyncTypeSafeClient(model=cfg.jev_model, retry=RetryPolicy(max_retries=0), timeout=cfg.jev_timeout_s)


def _make_claude_client() -> Any:
    import anthropic

    return anthropic.AsyncAnthropic(max_retries=0, timeout=60.0)


async def run(cfg: HarnessConfig, *, claude_client: Any = None, jev_client: Any = None) -> dict[str, Any]:
    """Run the selected configurations; injected clients skip the credential check (tests)."""
    unknown = [c for c in cfg.configs if c not in ALL_CONFIGS]
    if unknown:
        raise ValueError(f"unknown configs {unknown}; choose from {ALL_CONFIGS}")
    want_claude = [c for c in cfg.configs if c in CLAUDE_CONFIGS]
    want_jev = [c for c in cfg.configs if c in JEV_CONFIGS]
    own_claude = own_jev = False
    if want_claude and claude_client is None:
        claude_client, own_claude = _make_claude_client(), True
        if not claude_credentials_available(claude_client):
            raise CredentialsMissing("no Anthropic credentials (set ANTHROPIC_API_KEY or run `ant auth login`)")
    if want_jev and jev_client is None:
        if not jev_credentials_available():
            raise CredentialsMissing("no TypeSafe credentials (set TYPESAFE_API_KEY)")
        jev_client, own_jev = _make_jev_client(cfg), True
    facts = json.loads(cfg.packet_path.read_text()) if cfg.packet_path else sample_facts()
    results: list[dict[str, Any]] = []
    try:
        for name in want_claude:
            log.info("running %s (%d calls)", name, cfg.n_calls)
            results.append(await measure_claude(name, CLAUDE_CONFIGS[name], claude_client, facts, cfg.n_calls,
                                                pause_s=cfg.pause_s, prewarm=cfg.prewarm))
        cases = jev_cases(facts) if want_jev else {}
        for name in want_jev:
            bundle = JEV_CONFIGS[name]
            state, qs = cases[bundle]
            log.info("running %s / %s (%d calls)", name, bundle, cfg.n_calls)
            results.append(await measure_jev(name, bundle, jev_client, state, qs, cfg.n_calls, model=cfg.jev_model,
                                             pause_s=cfg.pause_s))
    finally:
        for client, own in ((claude_client, own_claude), (jev_client, own_jev)):
            closer = getattr(client, "close", None) or getattr(client, "aclose", None)
            if own and closer is not None:
                await closer()
    summaries = [summarize(r) for r in results]
    return {"network": cfg.network, "started_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "n_calls": cfg.n_calls, "packet_chars": len(json.dumps(facts)), "summaries": summaries,
            "recommended": recommended_deadlines(summaries), "results": results}


def write_outputs(report: Mapping[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(report["started_utc"]).replace(":", "").replace("-", "").replace("+0000", "Z")
    base = out_dir / f"latency_{report['network']}_{stamp}"
    json_path, md_path = base.with_suffix(".json"), base.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2, default=str), "utf-8")
    md = markdown_table(report["summaries"], str(report["network"]))
    md += "\nRecommended deadlines (measured p95):\n\n```json\n" + json.dumps(report["recommended"], indent=2) + "\n```\n"
    md_path.write_text(md, "utf-8")
    return json_path, md_path


def parse_args(argv: Sequence[str] | None = None) -> HarnessConfig:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--configs", default=",".join(ALL_CONFIGS), help="comma list of C1-C4, J1-J5")
    p.add_argument("--n", type=int, default=50, help="calls per configuration")
    p.add_argument("--network", default="dev-mac", help="network label (dev-mac, gym-wifi, lte-hotspot, ...)")
    p.add_argument("--out", default="runs/latency", help="output directory")
    p.add_argument("--packet", default=None, help="FactPacket JSON to send instead of the built-in one")
    p.add_argument("--pause", type=float, default=0.5, help="seconds between calls")
    p.add_argument("--no-prewarm", action="store_true", help="skip the max_tokens=0 cache pre-warm")
    a = p.parse_args(argv)
    return HarnessConfig(configs=[c.strip() for c in a.configs.split(",") if c.strip()], n_calls=a.n,
                         network=a.network, out_dir=Path(a.out), packet_path=Path(a.packet) if a.packet else None,
                         pause_s=a.pause, prewarm=not a.no_prewarm)


def main(argv: Sequence[str] | None = None, *, claude_client: Any = None, jev_client: Any = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse_args(argv)
    try:
        report = asyncio.run(run(cfg, claude_client=claude_client, jev_client=jev_client))
    except CredentialsMissing as err:
        log.error("refusing to run: %s", err)
        return 2
    except ValueError as err:
        log.error("%s", err)
        return 2
    json_path, md_path = write_outputs(report, cfg.out_dir)
    sys.stdout.write(markdown_table(report["summaries"], cfg.network))
    log.info("wrote %s and %s", json_path, md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
