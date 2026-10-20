"""Shared fakes for the narration-io tests (no tests here). Names are fictional; nothing touches the network."""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from courtvision.decision.decoders import Judgement, LineVerdict, PossessionAssessment
from courtvision.geometry.court import court_for
from courtvision.narration.narrator import NarrationConfig, Narrator
from courtvision.narration.tts import NullTTS
from courtvision.types import Event, EventKind, EventStatus, Insight, Roster, RosterEntry, Team, Utterance

HOME_NAMES = ["Avery Lindqvist", "Jordan Okafor", "Casey Marchetti", "Riley Thorsen", "Morgan Vasquez"]
AWAY_NAMES = ["Devon Achterberg", "Skyler Nakamura", "Emerson Duquesne", "Rowan Castellano", "Parker Ilunga"]
JERSEYS = ["0", "3", "11", "23", "5"]
COURT = court_for("nba")


def make_roster() -> Roster:
    entries = [RosterEntry(f"home-{j}", Team.HOME, j, n) for j, n in zip(JERSEYS, HOME_NAMES)]
    entries += [RosterEntry(f"away-{j}", Team.AWAY, j, n) for j, n in zip(JERSEYS, AWAY_NAMES)]
    return Roster(entries, {Team.HOME: "Riverside", Team.AWAY: "Lakeport"})


@dataclass
class FakeLine:
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
class FakeStats:
    """Duck-typed GameStats (read by attribute, as facts.py does)."""

    t: float = 300.0
    period: int | None = 2
    game_clock: float | None = 272.4
    score: dict = field(default_factory=lambda: {Team.HOME: 42, Team.AWAY: 40})
    players: dict = field(default_factory=lambda: {
        "home-23": FakeLine("home-23", Team.HOME, pts=12, fgm=5, fga=8, fg3m=2, fg3a=4, ast=2, pf=2),
        "home-3": FakeLine("home-3", Team.HOME, pts=6, fgm=3, fga=5, ast=4, stl=1),
        "away-11": FakeLine("away-11", Team.AWAY, pts=9, fgm=4, fga=10, dreb=5, blk=2, pf=3),
        "away-0": FakeLine("away-0", Team.AWAY, pts=4, fgm=2, fga=6, stl=2, tov=1)})
    teams: dict = field(default_factory=dict)
    shots: list = field(default_factory=list)
    run: dict = field(default_factory=lambda: {"team": None, "points": 0, "opp_points": 0, "since_t": None})
    score_source: str = "scoreboard"
    score_mismatch: bool = False
    shot_clock: float | None = 14.0


_ids = itertools.count(1)


def ev(kind: EventKind, *, status: EventStatus = EventStatus.CONFIRMED, source: str = "sim", **kw: Any) -> Event:
    kw.setdefault("event_id", f"e{next(_ids):06d}")
    kw.setdefault("t", 299.0)
    kw.setdefault("frame_idx", int(kw["t"] * 30))
    return Event(kind=kind, status=status, source=source, **kw)


def made(value: int = 3, pid: str = "home-23", team: Team = Team.HOME, **kw: Any) -> Event:
    attrs = {"shot_type": "jumper", "basket": "left", "distance_ft": 24.6}
    attrs.update(kw.pop("attrs", {}))
    kw.setdefault("shot_id", f"s{next(_ids):04d}")
    kw.setdefault("x", 30.0)
    kw.setdefault("y", 8.0)
    return ev(EventKind.SHOT_MADE, team=team, player_id=pid, value=value, attrs=attrs, **kw)


def revise(e: Event, **changes: Any) -> Event:
    d = e.to_dict()
    d.update(changes)
    d["revision"] = e.revision + 1
    out = Event.from_dict(d)
    if "status" in changes:
        out.status = changes["status"]
    return out


# --------------------------------------------------------------------------- fake LLM


class FakeLLM:
    """``NarrationLLM`` whose reply is a string, an exception, or a callable(prompt) -> str."""

    def __init__(self, reply: Any = "SILENCE", delay_s: float = 0.0) -> None:
        self.reply = reply
        self.delay_s = delay_s
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.max_tokens: list[int] = []

    def stream(self, system: str, prompt: str, max_tokens: int) -> AsyncIterator[str]:
        self.systems.append(system)
        self.prompts.append(prompt)
        self.max_tokens.append(max_tokens)
        return self._gen(prompt)

    async def _gen(self, prompt: str) -> AsyncIterator[str]:
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        reply = self.reply(prompt) if callable(self.reply) and not isinstance(self.reply, BaseException) else self.reply
        if isinstance(reply, BaseException):
            raise reply
        words = str(reply).split(" ")
        for i in range(0, len(words), 2):
            yield " ".join(words[i:i + 2]) + (" " if i + 2 < len(words) else "")


# --------------------------------------------------------------------------- fake Jev


def _j(label: str | None, value: float | None = None, source: str = "jev") -> Judgement:
    return Judgement(label=label, value=value, confidence=0.9, probabilities={}, source=source)


class FakeJev:
    """Duck-typed JevDecider: controllable guard verdicts, stakes, line kind and card pick."""

    def __init__(self, *, guard_pass: bool = True, stakes: str = "routine", highlight: str = "no",
                 line_kind: str = "stat_note", pick: Callable[[list[Insight]], Insight | None] | None = None,
                 available: bool = True) -> None:
        self.guard_pass = guard_pass
        self.stakes = stakes
        self.highlight = highlight
        self.line_kind = line_kind
        self.pick = pick
        self._available = available
        self.guard_calls: list[tuple[dict, list[str]]] = []
        self.b1_states: list[dict] = []
        self.b2_calls: list[tuple[dict, set[str], list[str]]] = []
        self.pick_calls: list[tuple[list[Insight], dict]] = []

    @property
    def available(self) -> bool:
        return self._available

    async def guard_lines(self, facts: dict, sentences: list[str]) -> list[LineVerdict]:
        self.guard_calls.append((facts, list(sentences)))
        p = 0.99 if self.guard_pass else 0.2
        return [LineVerdict(s, p, 0.01, 0.01, self.guard_pass, "jev") for s in sentences]

    async def possession_end(self, state: dict) -> PossessionAssessment:
        self.b1_states.append(state)
        return PossessionAssessment(play_type=_j("unclear", source="fallback"), summary_clear=_j("yes", 0.9),
                                    stakes=_j(self.stakes, 0.5), highlight=_j(self.highlight, 0.1))

    async def next_line_kind(self, state: dict, available_kinds: set[str],
                             storylines: list[str]) -> tuple[Judgement, list[Judgement]]:
        self.b2_calls.append((state, set(available_kinds), list(storylines)))
        return _j(self.line_kind), [_j("no", 0.1) for _ in storylines]

    async def pick_insight(self, candidates: list[Insight], ctx: dict) -> Insight | None:
        self.pick_calls.append((list(candidates), ctx))
        if self.pick is not None:
            return self.pick(candidates)
        return min(candidates, key=lambda i: i.priority)  # deliberately not the highest priority


# --------------------------------------------------------------------------- fake AsyncAnthropic


def text_events(text: str, model: str = "claude-opus-5-5") -> list[Any]:
    evs: list[Any] = [SimpleNamespace(type="message_start", message=SimpleNamespace(model=model)),
                      SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="thinking_delta",
                                                                                         thinking=""))]
    for part in text.split(" "):
        evs.append(SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=part + " ")))
        evs.append(SimpleNamespace(type="text", text=part + " "))  # the SDK also yields parsed text events
    evs.append(SimpleNamespace(type="message_stop"))
    return evs


class _FakeStream:
    def __init__(self, owner: FakeAnthropic) -> None:
        self.owner = owner
        self.closed = False

    async def __aenter__(self) -> _FakeStream:
        if self.owner.raise_on_enter is not None:
            raise self.owner.raise_on_enter
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.closed = True

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._gen()

    async def _gen(self) -> AsyncIterator[Any]:
        for e in text_events(self.owner.text, self.owner.served_model):
            if self.owner.delay_s:
                await asyncio.sleep(self.owner.delay_s)
            yield e

    async def get_final_message(self) -> Any:
        o = self.owner
        return SimpleNamespace(model=o.served_model, stop_reason=o.stop_reason, content=[],
                               usage=SimpleNamespace(input_tokens=420, output_tokens=o.output_tokens,
                                                     cache_read_input_tokens=880, cache_creation_input_tokens=0))


class FakeAnthropic:
    """Captures ``beta.messages.stream`` / ``messages.create`` / ``count_tokens`` kwargs."""

    def __init__(self, text: str = "Riverside leads.", stop_reason: str = "end_turn", served_model: str = "claude-opus-5-5",
                 output_tokens: int = 210, delay_s: float = 0.0, raise_on_enter: BaseException | None = None) -> None:
        self.text, self.stop_reason, self.served_model = text, stop_reason, served_model
        self.output_tokens, self.delay_s, self.raise_on_enter = output_tokens, delay_s, raise_on_enter
        self.stream_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.count_calls: list[dict[str, Any]] = []
        owner = self

        class _Beta:
            class messages:  # noqa: N801 - mimics the SDK attribute path
                @staticmethod
                def stream(**kwargs: Any) -> _FakeStream:
                    owner.stream_calls.append(kwargs)
                    return _FakeStream(owner)

        class _Messages:
            @staticmethod
            async def create(**kwargs: Any) -> Any:
                owner.create_calls.append(kwargs)
                return SimpleNamespace(model=owner.served_model, stop_reason="max_tokens", content=[],
                                       usage=SimpleNamespace(input_tokens=0, output_tokens=0,
                                                             cache_creation_input_tokens=880, cache_read_input_tokens=0))

            @staticmethod
            async def count_tokens(**kwargs: Any) -> Any:
                owner.count_calls.append(kwargs)
                return SimpleNamespace(input_tokens=870)

        self.beta = _Beta()
        self.messages = _Messages()
        self.api_key = "test-key"


class FakeTypeSafe:
    """Async ``system_one`` stand-in for the latency harness."""

    def __init__(self, model: str = "jev-1.13.0") -> None:
        self.model = model
        self.calls: list[tuple[Any, dict]] = []

    async def system_one(self, state: Any, questions: dict, *, model: str | None = None) -> Any:
        self.calls.append((state, questions))
        return SimpleNamespace(model=self.model, usage=SimpleNamespace(input_tokens=300 + len(questions), output_tokens=9))


# --------------------------------------------------------------------------- narrator factory


@dataclass
class Rig:
    narrator: Narrator
    tts: NullTTS
    utterances: list[Utterance]
    stats: FakeStats


def make_rig(mode: str = "broadcast", *, llm: Any = None, jev: Any = None, time_scale: float = 0.0,
             captions: Any = None, dead_ball: Any = None, **cfg: Any) -> Rig:
    cfg.setdefault("min_gap_s", 0.0)
    cfg.setdefault("cooldown_s", {k: 0.0 for k in ("play_by_play", "color", "correction", "accessibility", "coach")})
    cfg.setdefault("colour_start_delay_s", 0.0)
    cfg.setdefault("tts", "none")
    config = NarrationConfig(mode=mode, **cfg)
    tts = NullTTS(time_scale=time_scale)
    utts: list[Utterance] = []
    n = Narrator(make_roster(), COURT, config, llm=llm, jev=jev, on_utterance=utts.append, tts=tts,
                 captions=captions, dead_ball=dead_ball)
    return Rig(n, tts, utts, FakeStats())

