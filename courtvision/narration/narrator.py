"""Narrator: events -> FactPacket -> template cues / speculative Claude colour -> Director -> TTS + captions.

Contract (ARCHITECTURE narration-io; RESEARCH.md §2.6, §4.1, §7A.2-§7A.3, §7A.7-§7A.8; narration.md §3-§9):

* ``on_events``: builds facts and template cues (hedged when the packet is ``hedge_only``). Outcome
  calls are held per RESEARCH §4.1: release calls at once; a made FG after a 150-300 ms hold when its
  gate is met, a 0.80-0.95 calibrated make waits up to ``make_wait_s`` for confirmation (else
  silence), an uncalibrated make gets a hedged line now and the outcome call only once confirmed;
  misses/blocks/rebounds 300 ms, steals/turnovers 500 ms. Cues go to the deterministic Director
  (D3). A possession end triggers ``JevDecider.possession_end`` in the background; its stakes become a
  prosody tag on the next cue (never blocking).
* Revisions: queued cues about a revised event are dropped silently (a basket call is re-rendered
  from the new version); a wrong *material* fact already stated, or being stated now, gets one P0
  "Correction: ..." line (never twice; never for a line that only hedged: a hedged shot later
  confirmed as a basket gets its outcome call instead, within ``make_wait_s``).
* ``tick``: during a dead ball (``DeadBallDetector`` or any dead-ball source the caller supplies) runs
  colour: B2 ``next_line_kind`` (fallback when Jev is absent), speculative Claude candidates when the
  window is long enough (:class:`CandidatePool`), else a deterministic template colour line. A window
  relabelled longer (e.g. stillness, then a timeout) is decided again; storylines are >= 5 min apart.
* ``on_insights`` (coach mode): code-made cards ranked with ``JevDecider.pick_insight`` (fallback:
  priority), <= 12 words, at most 3 visible.
* Modes: ``broadcast`` (fan audio), ``coach`` (visual cards), ``accessibility`` (AD line for every
  possession event, score read after each possession, fixed lexicon).
* A speaking loop pops ``Director.next()`` into ``TTS.speak`` (``prosody`` is passed only to engines
  whose ``speak`` accepts it; the contract is ``speak(text, priority)``) and cuts the line when the
  Director asks for preemption. Every spoken line goes to ``on_utterance`` and to the caption emitter keyed to the
  source PTS of ``t_occ``. Nothing blocks the event loop; ``aclose`` cancels every task.

D16: cloud payloads carry ``{Pnn}`` tokens; names are rendered locally (identity tiers) only after
validation. The LLM path is off unless ``cloud_narration_ok`` and ``pseudonymise`` hold (§7A.7 gate).
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import itertools
import logging
import random
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from courtvision.decision import states as S
from courtvision.geometry.court import CourtSpec
from courtvision.geometry.rules import RuleSpec
from courtvision.narration.candidates import STATE_INDEPENDENT, CandidateConfig, CandidatePool
from courtvision.narration.captions import CaptionEmitter
from courtvision.narration.director import Cue, Director, DirectorConfig, fact_changed, is_material_change, priority_for
from courtvision.narration.facts import OUTCOME_KINDS, FactsConfig, build_facts, event_trusted
from courtvision.narration.llm import ClaudeConfig, ClaudeNarrationLLM, NarrationLLM
from courtvision.narration.pseudonym import PseudonymMap
from courtvision.narration.templates import TemplateConfig, TemplateRenderer, render_line
from courtvision.narration.tts import TTS, NullTTS, make_tts
from courtvision.narration.validator import GroundingValidator, ValidatorConfig
from courtvision.types import Event, EventKind, EventStatus, FrameState, Insight, Roster, Team, Utterance

if TYPE_CHECKING:
    from courtvision.decision.advisor import JevDecider
    from courtvision.stats.engine import GameStats

log = logging.getLogger(__name__)

MODES = ("broadcast", "coach", "accessibility")
_NO_VERSION_BUMP = frozenset({EventKind.DRIBBLE, EventKind.PASS})
_NOT_NARRATED = frozenset({EventKind.DRIBBLE, EventKind.PASS, EventKind.SCORE_CORRECTION})
_SHOT_UNIT = frozenset({EventKind.SHOT_ATTEMPT, EventKind.SHOT_MADE, EventKind.SHOT_MISSED, EventKind.BLOCK,
                        EventKind.ASSIST})
_PRIMARY_ORDER = (EventKind.SHOT_MADE, EventKind.BLOCK, EventKind.SHOT_MISSED, EventKind.SHOT_ATTEMPT,
                  EventKind.STEAL, EventKind.TURNOVER, EventKind.REBOUND, EventKind.FOUL, EventKind.ASSIST,
                  EventKind.TIMEOUT, EventKind.SUBSTITUTION, EventKind.PERIOD_END, EventKind.PERIOD_START,
                  EventKind.POSSESSION)
# Possession changes already voiced by the rebound / steal call (broadcast and AD).
_REDUNDANT_CAUSES = frozenset({"defensive_rebound", "offensive_rebound", "steal"})
# Families whose line states no outcome or point value (a revision of it needs no correction).
_NO_OUTCOME_FAMILIES = frozenset({"release", "ft_attempt", "hedge_shot", "hedge_ft", "hedge_loose", "hedge_rebound",
                                  "whistle", "possession", "bring_up"})
# AD lines all render as family "ad"; these primaries get a line that states no outcome or value
# ("{P07} shot, left wing." for an attempt; "Riverside ball.").
_AD_NO_OUTCOME_KINDS = frozenset({EventKind.SHOT_ATTEMPT, EventKind.POSSESSION})
_SI_CUE_VERSION = 1 << 62  # Director state_version of state-independent cues: never invalidated


@dataclass
class NarrationConfig:
    """Narrator settings (ARCHITECTURE NarrationConfig plus the hold / colour policy)."""

    mode: str = "broadcast"  # "broadcast" | "coach" | "accessibility"
    llm_enabled: bool = True
    model: str = "claude-opus-5-5"  # RESEARCH §2.6; sourced
    effort: str = "low"  # §7A.3; sourced
    max_tokens: int = 1536  # D13; sourced
    candidate_deadline_s: float = 12.0  # D6 placeholder until the latency harness measures p95; unverified
    p95_total_s: float = 12.0  # measured Claude p95 total; placeholder (§7A.3); unverified
    max_words: dict[str, int] = field(default_factory=lambda: {
        "play_by_play": 12, "color": 35, "coach": 12, "accessibility": 6, "correction": 12})  # narration.md §9.2; proposal
    cooldown_s: dict[str, float] = field(default_factory=lambda: {
        "play_by_play": 0.6, "color": 1.0, "correction": 0.0, "accessibility": 0.0, "coach": 0.0})  # §9.5; sourced
    min_gap_s: float = 0.6  # min silence between lines, live ball (narration.md §9.5; human 1.7 s [2]); sourced
    use_jev_validator: bool = True  # D4: LLM lines need a passing B4 verdict; sourced
    pseudonymise: bool = True  # D16: False disables the cloud path (names never leave the device); sourced
    cloud_narration_ok: bool = True  # §7A.7 consent gate (DPA, consent, under-13, region); False = template-only
    tts: str = "avspeech"  # "avspeech" | "none" | "say" (dev-only, D5)
    seed: int = 0  # pseudonym shuffle and template variety
    # Confirmation holds before an outcome call (RESEARCH §4.1 / narration.md §9.3; proposals); unverified
    hold_s: dict[str, float] = field(default_factory=lambda: {
        "shot_attempt": 0.0, "shot_made": 0.2, "shot_missed": 0.3, "block": 0.3, "rebound": 0.3,
        "steal": 0.5, "turnover": 0.5, "foul": 0.5})
    make_wait_s: float = 3.0  # wait for confirmation (§7A.5 / narration §11.2: 3.0 s until lag is measured); sourced (proposal)
    make_wait_band: float = 0.80  # calibrated makes in [0.80, gate) wait; below: silence (RESEARCH §4.1); sourced (proposal)
    colour_lines_per_window: int = 2  # 1-2 colour lines per dead ball (narration.md §4.2); sourced
    two_line_window_s: float = 20.0  # shorter stoppages (free throws) get 1 line (narration.md §9.2); sourced (proposal)
    colour_start_delay_s: float = 1.0  # first colour line 1.0-2.0 s after the stoppage call (§4.2); sourced
    min_colour_window_s: float = 4.0  # a stoppage >= 4 s opens P3 colour (§4.2); sourced
    default_dead_window_s: float = 8.0  # window assumed when the dead-ball source gives none; tuned
    colour_events: int = 6  # recent events put in a dead-ball colour packet (their players' stats come along); tuned
    storylines: list[str] = field(default_factory=list)  # state-independent topics (season storylines, program facts)
    storyline_refresh_s: float = 30.0  # min seconds between storyline candidate requests; tuned
    # Storyline lines at least this far apart in video time: same colour topic >= 5 min, same fact
    # never within 5 min (narration.md §9.5; video seconds stand in for game clock); sourced (proposal)
    storyline_cooldown_s: float = 300.0
    coach_max_visible: int = 3  # at most 3 cards visible (narration.md §7.3); sourced (proposal)
    coach_max_words: int = 12  # <= 12 words per card (§7.3); sourced (proposal)
    coach_immediate_kinds: tuple[str, ...] = ("foul_trouble", "team_fouls")  # Q+1 alerts post immediately (§7.3)
    coach_event_kinds: tuple[str, ...] = ("foul", "period_end")  # event cards in coach mode; tuned
    staff_team: str | None = None  # "home" | "away": the staff the coach channel serves (B3 context)
    ad_ttl_s: float = 4.0  # AD line TTL (AD p95 <= 2 s target, RESEARCH §7A.8); tuned
    recent_lines: int = 10  # spoken lines kept for repetition checks (narration.md §3.3); sourced
    poll_s: float = 0.02  # speaking-loop tick (narration.md §8: 50 Hz director tick); sourced
    stop_grace_s: float = 1.0  # TTS stop() must end the line within this, else the task is cancelled; tuned
    prosody_from_stakes: dict[str, str] = field(default_factory=lambda: {
        "routine": "calm", "notable": "up", "high_stakes": "big", "decisive": "big"})  # narration.md §3.4; proposal


@dataclass(frozen=True)
class DeadBallInfo:
    """What :meth:`Narrator.tick` needs from a dead-ball source."""

    dead: bool
    since_t: float | None = None
    expected_window_s: float = 0.0
    stoppage: str | None = None


DeadBallSource = Any  # DeadBallDetector | DeadBallInfo | DeadBallState | Mapping | bool | Callable[[float], ...]


@dataclass
class _Held:
    event_ids: list[str]
    primary: str
    due: float
    deadline: float | None  # None: plain hold; else wait for confirmation until the deadline
    received: float


@dataclass
class _CueMeta:
    source: str  # "template" | "llm"
    received: float
    ready: float
    prosody: str | None = None
    stated: frozenset[str] = frozenset()  # outcome events whose outcome / value the line states


@dataclass
class _Window:
    wid: int
    since_t: float | None
    stoppage: str | None
    opened: float
    ends: float
    kind: str | None = None  # B2 line kind; None while the decision is pending
    offered: int = 0
    max_lines: int = 1
    llm_requested: bool = False
    llm_at: float | None = None  # when Claude candidates were requested for this window
    tier: tuple[bool, bool, bool] = (False, False, False)  # (colour, Claude, two lines) allowed at open


@dataclass
class _NoStats:
    """Empty GameStats stand-in (before the first snapshot; state-independent packets)."""

    t: float = 0.0
    period: int | None = None
    game_clock: float | None = None
    score: dict = field(default_factory=dict)
    players: dict = field(default_factory=dict)
    teams: dict = field(default_factory=dict)
    shots: list = field(default_factory=list)
    run: dict = field(default_factory=dict)


def make_claude_llm(config: NarrationConfig | None = None, client: Any = None) -> ClaudeNarrationLLM:
    """The production ``NarrationLLM`` for a :class:`NarrationConfig` (model, effort, max_tokens)."""
    cfg = config or NarrationConfig()
    return ClaudeNarrationLLM(ClaudeConfig(model=cfg.model, effort=cfg.effort, max_tokens=cfg.max_tokens), client)


class Narrator:
    """Fan / coach / AD narration for one game (see the module docstring)."""

    def __init__(self, roster: Roster, court: CourtSpec, config: NarrationConfig | None = None,
                 llm: NarrationLLM | None = None, jev: JevDecider | None = None,
                 on_utterance: Callable[[Utterance], Awaitable[None] | None] | None = None,
                 tts: TTS | None = None, *, rules: RuleSpec | None = None, dead_ball: DeadBallSource = None,
                 captions: CaptionEmitter | None = None, clock: Callable[[], float] = time.monotonic,
                 director_config: DirectorConfig | None = None, candidate_config: CandidateConfig | None = None,
                 facts_config: FactsConfig | None = None, template_config: TemplateConfig | None = None,
                 pseudo: PseudonymMap | None = None) -> None:
        cfg = self.config = config or NarrationConfig()
        if cfg.mode not in MODES:
            raise ValueError(f"NarrationConfig.mode must be one of {MODES}, got {cfg.mode!r}")
        self.roster, self.court, self.rules = roster, court, rules
        self.jev = jev
        self.on_utterance = on_utterance
        self.captions = captions
        self._clock = clock
        self._dead_ball = dead_ball
        self.facts_config = facts_config or FactsConfig()
        # One map per game shared by facts, candidates and rendering (D16). The integration layer passes
        # ``narration.facts.pseudonyms_for(roster, seed)`` so every other packet builder sees the same
        # tokens; standalone narrators keep a private map.
        self.pseudo = pseudo if pseudo is not None else PseudonymMap(roster, seed=cfg.seed)
        self.validator = GroundingValidator(ValidatorConfig(max_words=dict(cfg.max_words)))
        self.templates = TemplateRenderer(template_config, seed=cfg.seed)
        self.director = Director(director_config or self._director_config())
        self._own_tts = tts is None
        self.tts: TTS = tts if tts is not None else self._default_tts()
        self._tts_prosody = _accepts_prosody(self.tts)  # ARCHITECTURE's TTS is speak(text, priority)
        self.llm = llm
        self.pool = self._make_pool(llm, candidate_config)
        self.transcript: list[Utterance] = []
        self.coach_cards: list[Utterance] = []
        self._pinned_cards: list[Utterance] = []  # Q+1 alerts and corrections stay on top
        self._ranked_cards: list[Utterance] = []
        self._ad_fifo: deque[tuple[Cue, _CueMeta]] = deque()  # AD lines are read in event order
        self._names = [e.name for e in roster.entries if e.name]
        self._team_names = {t: roster.team_names.get(t, t.value.capitalize()) for t in Team}
        self._stats: Any = None
        self._frame: FrameState | None = None
        self._last_t = 0.0
        self._version = 0
        self._events: dict[str, Event] = {}
        self._received: dict[str, float] = {}
        self._history: deque[Event] = deque(maxlen=400)
        self._held: dict[str, _Held] = {}
        self._announced: set[str] = set()
        self._unhedged: set[str] = set()  # events whose outcome / value a finished line stated
        self._speaking: tuple[Cue, _CueMeta] | None = None  # the line now playing
        self._carded: dict[str, Event] = {}
        self._card_corrected: set[str] = set()
        self._meta: OrderedDict[str, _CueMeta] = OrderedDict()
        self._recent: deque[str] = deque(maxlen=max(1, cfg.recent_lines))
        self._labels: dict[str, str] = {}
        self._prosody_next: str | None = None
        self._offense: Team | None = None
        self._poss_seq = 0
        self._window: _Window | None = None
        self._wids = itertools.count(1)
        self._cue_ids = itertools.count(1)
        self._utt_ids = itertools.count(1)
        self._story_requested_at: float | None = None
        self._story_spoken_t: float | None = None  # video time of the last storyline line
        self._loop_task: asyncio.Task[None] | None = None
        self._bg: set[asyncio.Task[Any]] = set()
        self._closed = False

    # ------------------------------------------------------------------ construction helpers

    def _director_config(self) -> DirectorConfig:
        cfg = self.config
        if cfg.mode == "accessibility":  # every possession event is read: no "latest wins", no flush
            return DirectorConfig(cooldown_s=dict(cfg.cooldown_s), min_gap_s=cfg.min_gap_s, p1_latest_wins=False,
                                  p1_flushes_lower=False, ttl_p1_s=cfg.ad_ttl_s, max_queued={0: 4, 1: 6, 2: 3, 3: 2})
        return DirectorConfig(cooldown_s=dict(cfg.cooldown_s), min_gap_s=cfg.min_gap_s)

    def _default_tts(self) -> TTS:
        if self.config.mode == "coach":
            return NullTTS(time_scale=0.0)  # coach cards are visual
        try:
            return make_tts(self.config.tts)
        except Exception as err:  # noqa: BLE001 - narration must keep running without audio
            log.error("TTS %r unavailable (%s); narrating silently with NullTTS", self.config.tts, err)
            return NullTTS()

    def _make_pool(self, llm: NarrationLLM | None, candidate_config: CandidateConfig | None) -> CandidatePool | None:
        cfg = self.config
        reasons = [why for ok, why in (
            (cfg.llm_enabled, "llm_enabled=False"), (llm is not None, "no llm"),
            (cfg.mode == "broadcast", f"mode={cfg.mode}"), (cfg.cloud_narration_ok, "consent gate closed (§7A.7)"),
            (cfg.pseudonymise, "pseudonymise=False (names must not leave the device, D16)")) if not ok]
        if reasons:
            log.info("Claude colour off: %s; templates only", ", ".join(reasons))
            return None
        assert llm is not None
        ccfg = candidate_config or CandidateConfig(
            p95_total_s=cfg.p95_total_s, candidate_deadline_s=cfg.candidate_deadline_s, max_tokens=cfg.max_tokens,
            max_words=int(cfg.max_words.get("color", 35)), use_jev_guard=cfg.use_jev_validator)
        return CandidatePool(llm, self.roster, self.pseudo, validator=self.validator, jev=self.jev, config=ccfg,
                             clock=self._clock)

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Start the speaking loop (done automatically by on_events / tick / on_insights)."""
        if self._closed:
            raise RuntimeError("narrator is closed")
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(self._run(), name="narrator-speaker")

    async def prewarm(self) -> None:
        """Pre-warm the Claude prompt cache (non-streaming max_tokens=0, same system blocks)."""
        if self.pool is not None and hasattr(self.llm, "prewarm"):
            await self.llm.prewarm(self.pool.system_prompt)  # type: ignore[union-attr]

    async def drain(self, timeout: float = 10.0) -> bool:
        """Wait until holds, queued cues, the current line, background work and the open dead-ball
        window's colour lines are finished."""
        end = self._clock() + timeout
        while self._clock() < end:
            now = self._clock()
            busy = (self._held or self._ad_fifo or self.director.current is not None
                    or self.director.pending(now) or self._colour_due(now)
                    or any(not t.done() for t in self._bg) or (self.pool is not None and self.pool.pending()))
            if not busy:
                return True
            await asyncio.sleep(self.config.poll_s)
        return False

    def _colour_due(self, now: float) -> bool:
        """The open window still owes a colour line (its line kind may still be being decided)."""
        w = self._window
        return (w is not None and not self._closed and now < w.ends and w.kind != "silence"
                and w.offered < w.max_lines)

    async def aclose(self) -> None:
        """Cancel every task, stop the voice and drop queued work."""
        self._closed = True
        tasks = [t for t in (self._loop_task, *self._bg) if t is not None]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._bg.clear()
        self._held.clear()
        if self.pool is not None:
            await self.pool.aclose()
        try:
            await self.tts.stop()
            if self._own_tts and hasattr(self.tts, "aclose"):
                await self.tts.aclose()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - shutdown must finish
            log.exception("TTS stop failed during aclose")

    # ------------------------------------------------------------------ public inputs

    async def on_events(self, events: list[Event], stats: GameStats, frame: FrameState | None) -> None:
        """New events and revisions (latest revision wins) plus the stats snapshot after them."""
        if self._closed:
            return
        self._ensure_started()
        now = self._clock()
        self._observe(stats, frame)
        fresh, revised = self._ingest(events, now)
        if any(e.kind not in _NO_VERSION_BUMP for e in [*fresh, *(n for n, _ in revised)]):
            self._bump_version()
        for new, old in revised:
            self._on_revised(new, old, now)
        if any(e.kind is EventKind.PERIOD_START for e in fresh):
            self.director.on_period_start()
        for unit in _units(fresh):
            self._schedule(unit, now)
        self._track_possessions(fresh)
        self._process_holds(now)

    async def on_insights(self, insights: list[Insight], stats: GameStats) -> None:
        """Coach mode: rank and post code-made cards (other modes ignore insights)."""
        if self._closed:
            return
        self._ensure_started()
        self._observe(stats, None)
        if self.config.mode != "coach":
            return
        cards = [i for i in insights if i.text]
        immediate = sorted((i for i in cards if i.kind in self.config.coach_immediate_kinds), key=lambda i: -i.priority)
        rest = [i for i in cards if i not in immediate]
        if immediate:
            self._post_cards(immediate, pinned=True)
        if not rest:
            return
        if self.jev is None:
            self._post_cards(_ranked(rest, max(rest, key=lambda i: i.priority)))
        else:
            self._spawn(self._rank_and_post(rest))

    async def tick(self, t: float, stats: GameStats, dead_ball: DeadBallSource = None) -> None:
        """Dead-ball colour scheduling (call regularly, e.g. at 5-10 Hz)."""
        if self._closed:
            return
        self._ensure_started()
        self._observe(stats, None, t)
        now = self._clock()
        self._process_holds(now)
        if self.config.mode != "broadcast":
            return
        self._stock_storylines(now)
        info = self._dead_ball_info(t, dead_ball)
        if not info.dead:
            self._window = None
            return
        w = self._window
        if w is None or (info.since_t is not None and info.since_t != w.since_t):
            self._open_window(info, now)
        elif self._upgraded(w, info):
            # Same stoppage, now known to be longer (e.g. "unknown" stillness relabelled a timeout):
            # decide again so a long window gets colour and Claude candidates.
            log.info("dead-ball window w%d upgraded to %s (%.0f s)", w.wid, info.stoppage, info.expected_window_s)
            self._open_window(info, now, prev=w)
        self._colour_step(now)

    # ------------------------------------------------------------------ ingest

    def _ensure_started(self) -> None:
        if not self._closed and (self._loop_task is None or self._loop_task.done()):
            self.start()

    def _observe(self, stats: Any, frame: FrameState | None, t: float | None = None) -> None:
        if stats is not None:
            self._stats = stats
        if frame is not None:
            self._frame = frame
        ts = [x for x in (getattr(stats, "t", None), frame.t if frame is not None else None, t) if x is not None]
        if ts:
            self._last_t = max(self._last_t, float(max(ts)))

    def _ingest(self, events: Iterable[Event], now: float) -> tuple[list[Event], list[tuple[Event, Event]]]:
        fresh: list[Event] = []
        revised: list[tuple[Event, Event]] = []
        for e in events:
            prev = self._events.get(e.event_id)
            if prev is not None and e.revision <= prev.revision:
                continue  # duplicate or stale revision
            self._events[e.event_id] = e
            self._received.setdefault(e.event_id, now)
            self._history.append(e)
            if prev is None:
                if e.status is not EventStatus.RETRACTED:
                    fresh.append(e)
            else:
                revised.append((e, prev))
        return fresh, revised

    def _bump_version(self) -> None:
        self._version += 1
        if self.pool is not None:
            self.pool.set_version(self._version)
        self.director.invalidate(state_version=self._version, min_priority=3)

    # ------------------------------------------------------------------ scheduling live calls

    def _schedule(self, unit: list[Event], now: float) -> None:
        cfg = self.config
        primary = _primary(unit)
        if primary is None or primary.kind in _NOT_NARRATED:
            return
        if cfg.mode == "coach" and primary.kind.value not in cfg.coach_event_kinds:
            return
        if primary.kind is EventKind.POSSESSION and primary.attrs.get("cause") in _REDUNDANT_CAUSES:
            return
        ids = [e.event_id for e in unit]
        held = self._held.get(_unit_key(primary))
        if held is not None:  # a late part of a unit still on hold (e.g. the assist of a held make)
            held.event_ids += [i for i in ids if i not in held.event_ids]
            merged = _primary([e for e in (self._events.get(i) for i in held.event_ids) if e is not None])
            held.primary = merged.event_id if merged is not None else held.primary
            return
        received = min(self._received.get(i, now) for i in ids)
        hold = cfg.hold_s.get(primary.kind.value, 0.0)
        if primary.kind in OUTCOME_KINDS and hold > 0:
            hold += self.director.hold_boost_s
        if primary.kind is EventKind.SHOT_ATTEMPT:
            self._schedule_attempt(primary, ids, now, received)
        elif primary.kind is EventKind.SHOT_MADE:
            self._schedule_make(primary, ids, now, received, hold)
        elif hold > 0:
            self._held[_unit_key(primary)] = _Held(ids, primary.event_id, now + hold, None, received)
        else:
            self._finalize(ids, primary.event_id, now, received)

    def _schedule_attempt(self, att: Event, ids: list[str], now: float, received: float) -> None:
        if self._calibrated(att) and not self._trusted(att):
            log.debug("release call skipped: %s below its gate", att.event_id)
            return  # RESEARCH §4.1: P(shot) < 0.90 -> no call
        if self._finalize(ids, att.event_id, now, received):
            self._announced.add(_shot_key(att))

    def _schedule_make(self, made: Event, ids: list[str], now: float, received: float, hold: float) -> None:
        cfg = self.config
        key = _unit_key(made)
        if self._trusted(made):
            self._held[key] = _Held(ids, made.event_id, now + hold, None, received)
            return
        if self._calibrated(made) and made.confidence < cfg.make_wait_band + self.director.gate_boost:
            log.debug("made call dropped: %s confidence %.2f below the wait band", made.event_id, made.confidence)
            return  # "still unsure: silence"
        if not self._calibrated(made) and _shot_key(made) not in self._announced:
            if self._finalize(ids, made.event_id, now, received, force_hedge=True):
                self._announced.add(_shot_key(made))
        self._held[key] = _Held(ids, made.event_id, now + hold, now + cfg.make_wait_s + self.director.hold_boost_s,
                                received)

    def _process_holds(self, now: float) -> None:
        for key, h in list(self._held.items()):
            live = [e for e in (self._events.get(i) for i in h.event_ids)
                    if e is not None and e.status is not EventStatus.RETRACTED]
            primary = self._events.get(h.primary)
            if not live or primary is None or primary.status is EventStatus.RETRACTED:
                self._held.pop(key, None)
                continue
            if h.deadline is None:
                if now >= h.due:
                    self._held.pop(key, None)
                    self._finalize(h.event_ids, h.primary, now, h.received)
            elif self._trusted(primary) and now >= h.due:
                self._held.pop(key, None)
                self._finalize(h.event_ids, h.primary, now, h.received)
            elif now >= h.deadline:
                self._held.pop(key, None)
                log.info("no confirmation for %s within the wait: silence", h.primary)

    def _finalize(self, ids: list[str], primary_id: str, now: float, received: float, *,
                  force_hedge: bool = False) -> bool:
        """Render, validate and offer (or post) the template line for one unit; True if offered."""
        live = self._unit_events(ids, primary_id)
        if not live:
            return False
        channel = self._live_channel()
        facts = self._facts(live)
        line = self.templates.render_line(live, facts, channel, hedge=force_hedge)
        if not line.text:
            return False
        primary = self._events[primary_id]
        if not force_hedge and self._repeats_hedge(primary, facts):
            return False
        if not self._valid_template(line.text, facts, channel):
            return False
        hedged = bool(line.hedged or force_hedge)  # hedge wording (counts toward the 1-in-3 cap)
        if channel == "coach":
            self._post_event_card(line.text, live)
            return True
        priority = 1 if channel == "accessibility" else priority_for(line.family, facts)
        cue = Cue(self._next_cue_id(), priority, line.text, max(e.t for e in live), 0.0, [e.event_id for e in live],
                  channel, hedged, self._version, line.family, f"poss{self._poss_seq}", list(live))
        stated = _stated_ids(line.family, live, primary, facts, hedged)
        self._offer(cue, _CueMeta("template", received, now, self._take_prosody(), stated), now)
        return True

    def _unit_events(self, ids: Sequence[str], primary_id: str) -> list[Event]:
        evs = [e for e in (self._events.get(i) for i in ids) if e is not None and e.status is not EventStatus.RETRACTED]
        primary = self._events.get(primary_id)
        if primary is not None and self._trusted(primary):
            # A confirmed call must not turn hedge-only because a secondary credit (assist) is unconfirmed.
            evs = [e for e in evs if e is primary or e.kind not in OUTCOME_KINDS or self._trusted(e)]
        return evs

    def _repeats_hedge(self, primary: Event, facts: Mapping[str, Any]) -> bool:
        """A hedged shot line after the release call adds nothing (the attempt was already voiced)."""
        if primary.kind not in (EventKind.SHOT_MADE, EventKind.SHOT_MISSED, EventKind.BLOCK):
            return False
        ev = next((f for f in facts.get("events") or () if f.get("event_id") == primary.event_id), {})
        hedged = facts.get("hedge_only") or not ev.get("outcome_ok", False)
        return bool(hedged) and _shot_key(primary) in self._announced

    def _valid_template(self, text: str, facts: Mapping[str, Any], channel: str) -> bool:
        problems = [p for p in self.validator.validate(text, facts, channel).problems
                    if not p.startswith("repeats recent line")]  # templates already avoid repeats when they can
        if problems:
            log.warning("template line %r rejected by the validator: %s", text, problems)
        return not problems

    def _live_channel(self) -> str:
        return {"broadcast": "play_by_play", "accessibility": "accessibility", "coach": "coach"}[self.config.mode]

    # ------------------------------------------------------------------ revisions and corrections

    def _on_revised(self, new: Event, old: Event, now: float) -> None:
        if self.config.mode == "coach":
            self._coach_revision(new, now)
            return
        outcome = self.director.on_revision(new, now, old=old)
        if outcome.action == "invalidated":
            # A queued line was rendered from the old version and dropped unheard (narration.md
            # §3.6: re-render or drop). A basket is re-rendered from the new version.
            self._late_outcome(new, now)
            return
        if outcome.action != "correct":
            return
        if not self._outcome_voiced(new.event_id):
            # Only a hedge (or a line stating no outcome) was heard, possibly still playing: nothing
            # wrong was said. A result now confirmed as a basket gets its outcome call instead.
            log.info("revision of %s: its outcome was never stated, no correction", new.event_id)
            self._late_outcome(new, now)
            return
        self._offer_correction(new, outcome.spoken or old, now)

    def _outcome_voiced(self, event_id: str) -> bool:
        """Did a finished line, or the line now playing, state this event's outcome or value?"""
        if event_id in self._unhedged:
            return True
        return self._speaking is not None and event_id in self._speaking[1].stated

    def _late_outcome(self, new: Event, now: float) -> None:
        """Outcome call for a basket the audience has not heard stated: only a hedge was voiced, or
        the queued call was dropped by this revision (the scoreboard flipped a hedged miss to a make;
        the scorer changed before speech). Same window as a fresh unconfirmed make: within
        ``make_wait_s`` of the event, else silence (RESEARCH §4.1, §7A.2)."""
        if new.kind is not EventKind.SHOT_MADE or new.status is EventStatus.RETRACTED or not self._trusted(new):
            return
        key = _unit_key(new)
        if key in self._held:
            return  # already waiting for confirmation; the hold makes the call
        received = self._received.get(new.event_id, now)
        if now - received > self.config.make_wait_s + self.director.hold_boost_s:
            log.info("confirmed basket %s arrived after the make wait: silence", new.event_id)
            return
        ids = [e.event_id for e in self._events.values()
               if new.shot_id and e.shot_id == new.shot_id and e.kind in (EventKind.SHOT_MADE, EventKind.ASSIST)]
        ids = ids if new.event_id in ids else [new.event_id, *ids]
        hold = self.config.hold_s.get(EventKind.SHOT_MADE.value, 0.0) + self.director.hold_boost_s
        self._held[key] = _Held(ids, new.event_id, now + hold, None, received)

    def _correction_line(self, new: Event, spoken: Event | None) -> tuple[str, dict[str, Any]] | None:
        facts = self._facts([new], previous=[spoken] if spoken is not None else None)
        line = self.templates.render_line([new], facts, "correction")
        if not line.text or not self._valid_template(line.text, facts, "correction"):
            return None
        return line.text, facts

    def _offer_correction(self, new: Event, spoken: Event | None, now: float) -> None:
        made = self._correction_line(new, spoken)
        if made is None:
            return
        cue = Cue(self._next_cue_id(), 0, made[0], new.t, 0.0, [new.event_id], "correction", False, self._version,
                  "correction", None, [new])
        self._offer(cue, _CueMeta("template", now, now, stated=frozenset({new.event_id})), now)

    def _coach_revision(self, new: Event, now: float) -> None:
        old = self._carded.get(new.event_id)
        if old is None or new.event_id in self._card_corrected:
            return
        if fact_changed(old, new) and is_material_change(old, new):
            made = self._correction_line(new, old)
            if made is not None:
                self._card_corrected.add(new.event_id)
                self._post_card(made[0], new.t, [new.event_id], priority=100, channel="correction",
                                events=[new], pinned=True)

    # ------------------------------------------------------------------ possessions (B1)

    def _track_possessions(self, fresh: list[Event]) -> None:
        for e in sorted(fresh, key=lambda e: e.t):
            if e.kind is EventKind.POSSESSION and e.team is not None and e.team is not self._offense:
                if self._offense is not None:
                    self._possession_end(self._offense)
                self._offense = e.team
                self._poss_seq += 1
            elif e.kind is EventKind.PERIOD_END and self._offense is not None:
                self._possession_end(self._offense)
                self._offense = None

    def _possession_end(self, offense: Team) -> None:
        if self.jev is None or self.config.mode == "coach":
            return
        try:
            poss = S.summarize_possession(list(self._history), offense, self._team_names[offense],
                                          game_clock_at_end_s=getattr(self._stats, "game_clock", None))
            state = S.b1_possession_end(self._situation(), poss, names=self._names)
        except Exception:  # noqa: BLE001 - flavour only; never break narration
            log.exception("B1 state build failed")
            return
        self._spawn(self._assess(state))

    async def _assess(self, state: dict) -> None:
        assert self.jev is not None
        a = await self.jev.possession_end(state)
        tag = self.config.prosody_from_stakes.get(str(a.stakes.label)) if a.stakes.label else None
        if a.highlight.label == "yes":
            tag = "big"
        if tag:
            self._prosody_next = tag
        if a.play_type.source != "fallback" and a.play_type.label and a.play_type.label != "unclear":
            self._labels = {"last_play": str(a.play_type.label).replace("_", " ")}

    def _take_prosody(self) -> str | None:
        tag, self._prosody_next = self._prosody_next, None
        return tag

    def _situation(self) -> S.GameSituation:
        st = self._stats or _NoStats()
        score = getattr(st, "score", None) or {}
        run = getattr(st, "run", None) or {}
        run_team = run.get("team") if isinstance(run, Mapping) else None
        run_team = Team(run_team) if isinstance(run_team, str) and run_team in ("home", "away") else run_team
        n_periods, halves = ((self.rules.n_periods, self.rules.halves) if self.rules is not None
                             else self.facts_config.period_format.get(self.court.league, (4, False)))
        return S.GameSituation(
            home=self._team_names[Team.HOME], away=self._team_names[Team.AWAY],
            home_score=int(_by_team(score, Team.HOME) or 0), away_score=int(_by_team(score, Team.AWAY) or 0),
            period=getattr(st, "period", None), clock_s=getattr(st, "game_clock", None), league=self.court.league,
            n_periods=int(n_periods), halves=bool(halves),
            shot_clock_full_s=(self.rules.shot_clock_s if self.rules is not None else 24),
            run_team=self._team_names[run_team] if isinstance(run_team, Team) else None,
            run_points=int(run.get("points", 0) or 0) if isinstance(run, Mapping) else 0,
            run_opp_points=int(run.get("opp_points", 0) or 0) if isinstance(run, Mapping) else 0)

    # ------------------------------------------------------------------ dead-ball colour

    def _dead_ball_info(self, t: float, src: DeadBallSource, depth: int = 0) -> DeadBallInfo:
        src = self._dead_ball if src is None and depth == 0 else src
        if src is None:
            return DeadBallInfo(False)
        if isinstance(src, DeadBallInfo):
            return src
        if isinstance(src, bool):
            return DeadBallInfo(src, None, self.config.default_dead_window_s if src else 0.0)
        if hasattr(src, "is_dead") and hasattr(src, "expected_window_s"):  # DeadBallDetector
            st = getattr(src, "state", None)
            stoppage = src.stoppage() if callable(getattr(src, "stoppage", None)) else getattr(st, "stoppage", None)
            return DeadBallInfo(bool(src.is_dead(t)), getattr(st, "since_t", None), float(src.expected_window_s(t)),
                                stoppage)
        if isinstance(src, Mapping):
            return DeadBallInfo(bool(src.get("dead")), src.get("since_t"),
                                float(src.get("expected_window_s", self.config.default_dead_window_s) or 0.0),
                                src.get("stoppage"))
        if hasattr(src, "dead"):  # DeadBallState
            window = getattr(src, "expected_window_s", None)
            return DeadBallInfo(bool(src.dead), getattr(src, "since_t", None),
                                float(window if window is not None else self.config.default_dead_window_s),
                                getattr(src, "stoppage", None))
        if callable(src) and depth < 2:
            return self._dead_ball_info(t, src(t), depth + 1)
        log.warning("unknown dead-ball source %r; treating the ball as live", type(src).__name__)
        return DeadBallInfo(False)

    def _window_tier(self, expected_window_s: float) -> tuple[bool, bool, bool]:
        """(colour allowed, Claude allowed, two lines allowed) for a window this long."""
        cfg = self.config
        return (expected_window_s >= cfg.min_colour_window_s,
                self.pool is not None and self.pool.window_ok(expected_window_s),
                expected_window_s >= cfg.two_line_window_s)

    def _upgraded(self, w: _Window, info: DeadBallInfo) -> bool:
        """The open window now qualifies for something it did not at open (each step happens once)."""
        return any(new and not old for new, old in zip(self._window_tier(info.expected_window_s), w.tier))

    def _open_window(self, info: DeadBallInfo, now: float, prev: _Window | None = None) -> None:
        cfg = self.config
        w = _Window(next(self._wids), info.since_t, info.stoppage, prev.opened if prev else now,
                    now + max(0.0, info.expected_window_s))
        w.tier = self._window_tier(info.expected_window_s)
        if prev is not None:  # an upgrade of the same stoppage keeps what was already said
            w.offered, w.llm_requested, w.llm_at = prev.offered, prev.llm_requested, prev.llm_at
            w.tier = tuple(a or b for a, b in zip(w.tier, prev.tier))  # type: ignore[assignment]
        w.max_lines = cfg.colour_lines_per_window if w.tier[2] else 1
        self._window = w
        if not w.tier[0]:
            w.kind = "silence"
            return
        facts = self._facts(self._colour_events(), colour=True)
        if self.pool is not None and w.tier[1] and not w.llm_requested:
            n = self.pool.request(facts, state_version=self._version, t_occ=self._last_t,
                                  expected_window_s=info.expected_window_s)
            w.llm_requested = n > 0 or self.pool.pending(state_independent=False) > 0
            w.llm_at = now if w.llm_requested else None
        material = render_line(self._colour_events(), facts, "color", random.Random(w.wid)).text
        stories = [self.pseudo.pseudonymise(s, self.roster) for s in cfg.storylines][: S.MAX_STORYLINES]
        if self.jev is None:
            w.kind = "stat_note" if material or w.llm_requested else ("storyline" if stories else "silence")
            return
        kinds = ({"stat_note"} if material or w.llm_requested else set()) | ({"storyline"} if stories else set())
        state = S.b2_dead_ball_fan(self._situation(), stoppage=S.stoppage_text(info.stoppage or "unknown"),
                                   material={"stat_note": material or "a number-backed note from tonight's stats"},
                                   storylines=stories, dead_ball_id=f"w{w.wid}", names=self._names)
        self._spawn(self._decide_kind(w, state, kinds, stories))

    async def _decide_kind(self, w: _Window, state: dict, kinds: set[str], stories: list[str]) -> None:
        assert self.jev is not None
        kind, due = await self.jev.next_line_kind(state, kinds, stories)
        label = kind.label or "silence"
        if label != "storyline" and any(j.label == "yes" for j in due) and "storyline" in kinds:
            label = "storyline"
        w.kind = label

    def _colour_step(self, now: float) -> None:
        cfg, w = self.config, self._window
        if w is None or w.kind in (None, "silence") or w.offered >= w.max_lines:
            return
        if now - w.opened < cfg.colour_start_delay_s or now >= w.ends or self._p3_busy(now):
            return
        cand = None
        if self.pool is not None:
            story_ok = (self._story_spoken_t is None
                        or self._last_t - self._story_spoken_t >= cfg.storyline_cooldown_s)
            cand = self.pool.take(now, kinds=[w.kind], recent_lines=list(self._recent),
                                  allow_dependent=w.kind != "storyline" or w.llm_requested,
                                  allow_independent=story_ok)
            if (cand is None and w.llm_requested and self.pool.pending(state_independent=False)
                    and now - (w.llm_at if w.llm_at is not None else w.opened) < cfg.p95_total_s):
                return  # Claude is still writing; the window is long enough to wait
        ttl = max(1.0, w.ends - now)
        if cand is not None:
            version = _SI_CUE_VERSION if cand.state_version == STATE_INDEPENDENT else cand.state_version
            cue = Cue(self._next_cue_id(), 3, cand.text_tokens, cand.t_occ, ttl, list(cand.event_ids), "color", False,
                      version, f"color:llm:{cand.kind}", None, [])
            self._offer(cue, _CueMeta("llm", cand.created_at, cand.ready_at, self._take_prosody()), now)
        elif not self._offer_template_colour(now, ttl):
            w.kind = "silence"  # nothing left to say in this window
            return
        w.offered += 1

    def _offer_template_colour(self, now: float, ttl: float) -> bool:
        events = self._colour_events()
        facts = self._facts(events, colour=True)
        line = self.templates.render_line(events, facts, "color")
        if not line.text:
            return False
        problems = self.validator.validate(line.text, facts, "color").problems
        if problems:
            log.debug("template colour %r rejected: %s", line.text, problems)
            return False
        cue = Cue(self._next_cue_id(), 3, line.text, self._last_t, ttl, [], "color", False, self._version,
                  line.family, None, [])
        self._offer(cue, _CueMeta("template", now, now, self._take_prosody()), now)
        return True

    def _colour_events(self) -> list[Event]:
        """Latest versions of the last few narrated events (the dead ball's context)."""
        out: list[Event] = []
        seen: set[str] = set()
        for e in reversed(self._history):
            if e.event_id in seen or e.kind in _NOT_NARRATED:
                continue
            seen.add(e.event_id)
            latest = self._events.get(e.event_id, e)
            if latest.status is not EventStatus.RETRACTED:
                out.append(latest)
            if len(out) >= self.config.colour_events:
                break
        return sorted(out, key=lambda e: e.t)

    def _p3_busy(self, now: float) -> bool:
        cur = self.director.current
        return (cur is not None and cur.priority == 3) or any(c.priority == 3 for c in self.director.pending(now))

    def _stock_storylines(self, now: float) -> None:
        """Keep up to ``n_parallel`` state-independent storyline candidates in stock."""
        cfg, pool = self.config, self.pool
        if pool is None or not cfg.storylines or not pool.guard_available:
            return
        if self._story_requested_at is not None and now - self._story_requested_at < cfg.storyline_refresh_s:
            return
        have = pool.pending(state_independent=True) + sum(1 for c in pool.ready(now) if c.state_independent)
        if have >= pool.config.n_parallel:
            return
        facts = build_facts([], _NoStats(), self.roster, self.court, None, self.pseudo, self.rules,
                            recent_lines=list(self._recent), state_version=STATE_INDEPENDENT,
                            storylines=cfg.storylines, config=self.facts_config)
        if pool.request(facts, state_version=STATE_INDEPENDENT, t_occ=self._last_t, state_independent=True,
                        kinds=("storyline",)):
            self._story_requested_at = now

    # ------------------------------------------------------------------ coach cards

    async def _rank_and_post(self, rest: list[Insight]) -> None:
        assert self.jev is not None
        wire = [dataclasses.replace(i, text=self._card_tokens(i)) for i in rest]
        ctx: dict[str, Any] = {"situation": S.situation_block(self._situation())}
        if self.config.staff_team in ("home", "away"):
            ctx["staff_team"] = self._team_names[Team(self.config.staff_team)]
        chosen = await self.jev.pick_insight(wire, ctx)
        if chosen is None:
            log.debug("Jev: no card worth pushing at this stoppage")
            return
        original = next((i for i in rest if i.insight_id == chosen.insight_id), None)
        self._post_cards(_ranked(rest, original or max(rest, key=lambda i: i.priority)))

    def _card_tokens(self, ins: Insight) -> str:
        player = "{" + self.pseudo.token(ins.player_id) + "}" if ins.player_id else "a player"
        team = self._team_names.get(ins.team, "the team") if ins.team is not None else "the team"
        return ins.text.replace("{player}", player).replace("{team}", team)

    def _post_cards(self, ranked: list[Insight], *, pinned: bool = False) -> None:
        posted: list[Utterance] = []
        for ins in ranked[: self.config.coach_max_visible]:
            text = self._card_tokens(ins)
            words = len(self._render(text, []).split())  # the cap is on what staff read: names can be 3 words
            if words > self.config.coach_max_words:
                log.warning("coach card %s dropped: %d words > %d", ins.insight_id, words, self.config.coach_max_words)
                continue
            posted.append(self._card_utterance(text, ins.t, [], ins.priority, "coach", []))
        self._board(posted, pinned)

    def _post_event_card(self, text: str, live: list[Event]) -> None:
        for e in live:
            self._carded[e.event_id] = e
        self._post_card(text, max(e.t for e in live), [e.event_id for e in live], priority=50, channel="coach",
                        events=live)

    def _post_card(self, text: str, t: float, event_ids: list[str], *, priority: int, channel: str,
                   events: list[Event], pinned: bool = False) -> None:
        self._board([self._card_utterance(text, t, event_ids, priority, channel, events)], pinned)

    def _card_utterance(self, text: str, t: float, event_ids: list[str], priority: int, channel: str,
                        events: list[Event]) -> Utterance:
        rendered = self._render(text, events)
        utt = Utterance(self._next_utt_id(), rendered, channel, t, list(event_ids), "template", priority)  # type: ignore[arg-type]
        self._publish(utt, caption=False)
        return utt

    def _board(self, posted: list[Utterance], pinned: bool) -> None:
        """Newest first; pinned cards (Q+1 alerts, corrections) keep at least one slot above the rest."""
        if not posted:
            return
        n = self.config.coach_max_visible
        if pinned:
            self._pinned_cards = (posted + self._pinned_cards)[:n]
        else:
            self._ranked_cards = (posted + self._ranked_cards)[:n]
        pins = self._pinned_cards[: max(1, n - 1)] if self._ranked_cards else self._pinned_cards
        self.coach_cards = (pins + self._ranked_cards)[:n]

    # ------------------------------------------------------------------ speaking loop

    def _offer(self, cue: Cue, meta: _CueMeta, now: float) -> None:
        self._meta[cue.cue_id] = meta
        while len(self._meta) > 512:
            self._meta.popitem(last=False)
        if self.config.mode == "accessibility" and cue.priority > 0:
            self._ad_fifo.append((cue, meta))  # the Director picks freshest first; AD must keep event order
            self._feed_ad(now)
            return
        self.director.offer(cue, now)

    def _feed_ad(self, now: float) -> None:
        """Hand the oldest AD line to the Director once nothing else is queued (stale lines are dropped)."""
        while self._ad_fifo and self.director.current is None and not self.director.pending(now):
            cue, meta = self._ad_fifo.popleft()
            if now - meta.ready > self.config.ad_ttl_s:
                log.debug("AD line %s dropped: older than %.1f s", cue.cue_id, self.config.ad_ttl_s)
                continue
            self.director.offer(cue, now)
            return

    async def _run(self) -> None:
        while True:
            try:
                await self._step()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - keep narrating after a bad line
                log.exception("narrator speaking loop error")
                await asyncio.sleep(self.config.poll_s)

    async def _step(self) -> None:
        now = self._clock()
        self._process_holds(now)
        self._colour_step(now)
        self._feed_ad(now)
        cue = self.director.next(now) if self.config.mode != "coach" else None
        if cue is None:
            await asyncio.sleep(self.config.poll_s)
            return
        await self._speak(cue)

    async def _speak(self, cue: Cue) -> None:
        start = self._clock()
        meta = self._meta.pop(cue.cue_id, None) or _CueMeta("template", start, start)
        text = self._render(cue.text_template, cue.events)
        utt = Utterance(self._next_utt_id(), text, cue.channel, cue.t_occ, list(cue.event_ids),  # type: ignore[arg-type]
                        meta.source, 3 - cue.priority, round((meta.ready - meta.received) * 1000.0, 1), True)
        try:
            task = asyncio.ensure_future(self.tts.speak(text, utt.priority, prosody=meta.prosody) if self._tts_prosody
                                         else self.tts.speak(text, utt.priority))
        except Exception:  # noqa: BLE001 - a TTS that cannot start must not stall the director
            log.exception("TTS speak() failed to start; line %s dropped", cue.cue_id)
            self.director.on_interrupted(cue, self._clock())
            return
        self._publish(utt, caption=True)
        self._speaking = (cue, meta)
        try:
            interrupted = await self._watch(task)
        except BaseException:  # cancelled (aclose) or a failing stop(): never leave the voice task behind
            if not task.done():
                task.cancel()
                await asyncio.wait({task}, timeout=self.config.stop_grace_s)
            self.director.on_interrupted(cue, self._clock())
            raise
        finally:
            self._speaking = None
        now = self._clock()
        if interrupted:
            self.director.on_interrupted(cue, now)
            log.debug("line %s cut by preemption", cue.cue_id)
            return
        self.director.on_spoken(cue, now, game_time=getattr(self._stats, "elapsed_game_s", None))
        self._recent.append(cue.text_template)
        self._note_spoken(cue, meta)
        if cue.priority == 3 and cue.state_version == _SI_CUE_VERSION:
            self._story_spoken_t = self._last_t  # storyline topic cooldown starts

    async def _watch(self, task: asyncio.Future[None]) -> bool:
        """Wait for the line; cut it when the Director requests preemption. True if interrupted."""
        cfg = self.config
        while not task.done():
            await asyncio.wait({task}, timeout=cfg.poll_s)
            if task.done():
                break
            now = self._clock()
            self._process_holds(now)
            if self.director.preempt_requested(now):
                try:
                    await self.tts.stop()
                except Exception:  # noqa: BLE001 - fall through to cancelling the line
                    log.exception("TTS stop() failed")
                done, _ = await asyncio.wait({task}, timeout=cfg.stop_grace_s)
                if not done:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                return True
        if task.cancelled():
            return True
        err = task.exception()
        if err is not None:
            log.error("TTS failed: %r", err)
            return True
        return False

    def _note_spoken(self, cue: Cue, meta: _CueMeta) -> None:
        """Remember which outcomes the audience has now heard stated (corrections only fix those)."""
        self._unhedged.update(meta.stated)

    def _publish(self, utt: Utterance, *, caption: bool) -> None:
        self.transcript.append(utt)
        if self.on_utterance is not None:
            try:
                res = self.on_utterance(utt)
                if inspect.isawaitable(res):
                    self._spawn(res)
            except Exception:  # noqa: BLE001 - a bad consumer must not stop narration
                log.exception("on_utterance failed")
        if caption and self.captions is not None and utt.channel != "coach":
            channel = "ad" if self.config.mode == "accessibility" else "caption"
            self._spawn(self.captions.emit(utt.text, utt.t, channel=channel, t_confirm=self._last_t,
                                           event_ids=utt.event_ids))

    # ------------------------------------------------------------------ helpers

    def _facts(self, events: list[Event], *, previous: list[Event] | None = None, colour: bool = False) -> dict[str, Any]:
        return build_facts(list(events), self._stats if self._stats is not None else _NoStats(), self.roster,
                           self.court, self._frame, self.pseudo, self.rules, recent_lines=list(self._recent),
                           state_version=self._version, storylines=self.config.storylines if colour else None,
                           labels=self._labels if colour else None, previous=previous,
                           gate_boost=self.director.gate_boost, config=self.facts_config)

    def _render(self, text: str, events: Sequence[Event]) -> str:
        """Tokens -> names under the identity tiers (event attrs ``id_conf`` / ``jersey_conf``)."""
        id_conf: dict[str, float] = {}
        jersey_conf: dict[str, float] = {}
        for e in events:
            for pid in (e.player_id, e.other_player_id):
                if pid and "id_conf" in e.attrs:
                    id_conf[pid] = float(e.attrs["id_conf"])
                if pid and "jersey_conf" in e.attrs:
                    jersey_conf[pid] = float(e.attrs["jersey_conf"])
        return self.pseudo.render(text, self.roster, id_conf or None, jersey_conf or None)

    def _trusted(self, e: Event) -> bool:
        return event_trusted(e, self.facts_config, self.director.gate_boost)

    def _calibrated(self, e: Event) -> bool:
        cfg = self.facts_config
        return (e.status is EventStatus.CONFIRMED or e.source in cfg.trusted_sources
                or str(e.attrs.get("p_status", "uncalibrated")) in cfg.calibrated_statuses)

    def _spawn(self, aw: Awaitable[Any]) -> None:
        if self._closed:  # aclose() already cancelled everything; start nothing new
            if inspect.iscoroutine(aw):
                aw.close()
            return
        task = asyncio.ensure_future(aw)
        self._bg.add(task)
        task.add_done_callback(self._bg_done)

    def _bg_done(self, task: asyncio.Future[Any]) -> None:
        self._bg.discard(task)  # type: ignore[arg-type]
        if not task.cancelled() and task.exception() is not None:
            log.error("narrator background task failed: %r", task.exception())

    def _next_cue_id(self) -> str:
        return f"q{next(self._cue_ids):05d}"

    def _next_utt_id(self) -> str:
        return f"u{next(self._utt_ids):04d}"

    def stats(self) -> dict[str, Any]:
        """Counters for dashboards and tests."""
        by_source: dict[str, int] = {}
        for u in self.transcript:
            by_source[u.source] = by_source.get(u.source, 0) + 1
        return {"utterances": len(self.transcript), "by_source": by_source, "state_version": self._version,
                "held": len(self._held), "candidates": self.pool.stats() if self.pool is not None else None}


# --------------------------------------------------------------------------- module helpers


def _by_team(d: Any, team: Team) -> Any:
    if not isinstance(d, Mapping):
        return None
    return d.get(team, d.get(team.value))


def _accepts_prosody(tts: Any) -> bool:
    """True when ``tts.speak`` takes the ``prosody`` keyword (our adapters' extension of the
    ARCHITECTURE contract ``speak(text, priority)``)."""
    try:
        params = inspect.signature(tts.speak).parameters
    except (AttributeError, TypeError, ValueError):
        return False
    return "prosody" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _stated_ids(family: str, live: Sequence[Event], primary: Event, facts: Mapping[str, Any],
                hedged: bool) -> frozenset[str]:
    """Outcome events whose outcome or point value this template line states. Hedged lines and
    families that state no outcome (release call, possession, AD shot call) state none."""
    if hedged or facts.get("hedge_only") or family in _NO_OUTCOME_FAMILIES:
        return frozenset()
    if family == "ad" and primary.kind in _AD_NO_OUTCOME_KINDS:
        return frozenset()
    ok = {f.get("event_id") for f in facts.get("events") or () if isinstance(f, Mapping) and f.get("outcome_ok")}
    return frozenset(e.event_id for e in live if e.kind in OUTCOME_KINDS and e.event_id in ok)


def _shot_key(e: Event) -> str:
    return f"shot:{e.shot_id or e.event_id}"


def _unit_key(e: Event) -> str:
    return _shot_key(e) if e.kind in _SHOT_UNIT else f"ev:{e.event_id}"


def _units(events: Sequence[Event]) -> list[list[Event]]:
    """Group one batch into narration units: one per shot (attempt/result/block/assist), steal +
    turnover at the same frame together, everything else on its own."""
    units: dict[str, list[Event]] = {}
    for e in sorted(events, key=lambda e: (e.t, e.frame_idx)):
        if e.kind in _SHOT_UNIT and (e.shot_id or e.kind is not EventKind.ASSIST):
            key = _shot_key(e)
        elif e.kind in (EventKind.STEAL, EventKind.TURNOVER):
            key = f"loose:{e.frame_idx}"
        else:
            key = f"ev:{e.event_id}"
        units.setdefault(key, []).append(e)
    return list(units.values())


def _primary(unit: Sequence[Event]) -> Event | None:
    rank = {k: i for i, k in enumerate(_PRIMARY_ORDER)}
    return min(unit, key=lambda e: (rank.get(e.kind, len(rank)), -e.t)) if unit else None


def _ranked(cards: list[Insight], chosen: Insight) -> list[Insight]:
    return [chosen] + sorted((c for c in cards if c is not chosen), key=lambda i: -i.priority)
