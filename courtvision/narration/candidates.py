"""Speculative LLM colour candidates (RESEARCH.md §2.6, §7A.3; narration.md §5.4, §9.4; D4, D6, D16).

* On a dead ball whose ``expected_window_s >= p95_total_s + tts_s + margin_s`` (12 + 10 + 1 s until the
  latency harness measures p95; ``tts_s`` is the speaking time of a line at the word cap), or for
  state-independent topics, 2 candidates are requested in parallel with different line-kind hints.
* Each candidate must pass the deterministic :class:`GroundingValidator` (every sentence) and then
  ``JevDecider.guard_lines`` (B4). Without a passing Jev verdict the line is not spoken (D4).
* Candidates are bound to a ``state_version`` and dropped (in flight: cancelled) when it changes.
  State-independent candidates use :data:`STATE_INDEPENDENT` and survive version changes.
* The prompt carries ``{Pnn}`` tokens only; a roster-name check blocks any request that would leak a
  name (D16). Tokens are rendered to names locally with :class:`PseudonymMap` only after validation.
* No first-token abort: one total deadline per candidate (``candidate_deadline_s``, D6).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from courtvision.narration.llm import SILENCE, NarrationLLM, NarrationLLMError, build_system_prompt, build_user_prompt
from courtvision.narration.pseudonym import PseudonymMap
from courtvision.narration.validator import GroundingValidator, count_words, jaccard, split_sentences
from courtvision.types import Roster

if TYPE_CHECKING:
    from courtvision.decision.advisor import JevDecider
    from courtvision.decision.decoders import LineVerdict

log = logging.getLogger(__name__)

STATE_INDEPENDENT = -1  # state_version of topics no fact change can invalidate (season storylines)


@dataclass
class CandidateConfig:
    """Candidate-pool policy (RESEARCH.md §7A.3 until the harness has measured p95)."""

    n_parallel: int = 2  # 2 speculative candidates per dead ball (RESEARCH §2.6, Kawamatsu [74]); sourced
    p95_total_s: float = 12.0  # Claude p95 total latency; D6 placeholder until §7A.3 measures it; unverified
    # Time to speak the line, "tts_s" in the §7A.3 window policy. None: the longest line the cap allows,
    # max_words / words_per_s (35 / 3.5 = 10 s), so an accepted line always ends inside the window.
    tts_s: float | None = None
    words_per_s: float = 3.5  # macOS voice measured 3.5-3.7 words/s (narration.md §4.1 [M]); sourced
    margin_s: float = 1.0  # "+ 1 s" in the §7A.3 window policy; sourced
    candidate_deadline_s: float = 12.0  # D6: total deadline, no first-token abort; reset from measured p95; unverified
    max_tokens: int = 1536  # D13: thinking counts toward max_tokens; sourced
    max_words: int = 35  # colour line cap (narration.md §9.2: 35 words per timeout line); sourced (proposal)
    max_sentences: int = 2  # colour is 1-2 sentences (narration.md §5.3); B4 checks <= 2 per call; sourced
    kinds: tuple[str, ...] = ("stat_note", "tendency")  # hints for the parallel pair (narration.md §5.4); sourced
    use_jev_guard: bool = True  # D4: LLM lines need a passing B4 verdict; sourced
    ready_ttl_s: float = 20.0  # state-dependent candidates live <= one dead-ball window (narration.md §9.1); sourced (proposal)
    independent_ttl_s: float = 900.0  # state-independent candidates (storylines); tuned
    max_ready: int = 4  # ready buffer depth (P3 <= 2 per window, narration.md §9.4, plus storylines); tuned
    repetition_jaccard: float = 0.8  # re-checked against the newest recent lines at take time (validator rule); tuned
    channel: str = "color"
    answer_directly: bool = False  # C2 system line (only after the harness shows it helps); unverified


@dataclass
class Candidate:
    """A validated, guarded LLM line ready to be offered as a P3 cue."""

    candidate_id: str
    kind: str
    state_version: int
    text_tokens: str  # token form ({Pnn}); what the validator and the guard saw
    text: str  # rendered locally after both checks
    sentences: list[str]
    facts: dict[str, Any]
    t_occ: float
    created_at: float
    ready_at: float
    latency_ms: float
    event_ids: list[str] = field(default_factory=list)
    verdicts: list[Any] = field(default_factory=list)

    @property
    def state_independent(self) -> bool:
        return self.state_version == STATE_INDEPENDENT


@dataclass
class _Pending:
    task: asyncio.Task[Candidate | None]
    state_version: int
    kind: str


class CandidatePool:
    """Generates, checks, stores and hands out speculative colour candidates. Never raises."""

    def __init__(self, llm: NarrationLLM, roster: Roster, pseudo: PseudonymMap, *,
                 validator: GroundingValidator | None = None, jev: JevDecider | None = None,
                 config: CandidateConfig | None = None, system_prompt: str | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.config = config or CandidateConfig()
        self.llm = llm
        self.roster = roster
        self.pseudo = pseudo
        self.validator = validator or GroundingValidator()
        self.jev = jev
        self.system_prompt = system_prompt or build_system_prompt(self.config.answer_directly)
        self._clock = clock
        self._ids = itertools.count(1)
        self._version = 0
        self._pending: dict[str, _Pending] = {}
        self._ready: list[Candidate] = []
        self.counts: Counter[str] = Counter()
        self._name_res = _name_patterns(roster)
        if not self.config.use_jev_guard:
            log.warning("candidate pool: Jev line guard disabled (D4 requires it for spoken LLM lines)")

    # ------------------------------------------------------------------ policy

    @property
    def tts_s(self) -> float:
        """Speaking time budgeted for one candidate line (see :class:`CandidateConfig`)."""
        cfg = self.config
        return cfg.tts_s if cfg.tts_s is not None else cfg.max_words / max(cfg.words_per_s, 1e-6)

    @property
    def threshold_s(self) -> float:
        cfg = self.config
        return cfg.p95_total_s + self.tts_s + cfg.margin_s

    def window_ok(self, expected_window_s: float) -> bool:
        """§7A.3: Claude only when the stoppage outlasts p95 + TTS + 1 s."""
        return float(expected_window_s) >= self.threshold_s

    @property
    def guard_available(self) -> bool:
        if not self.config.use_jev_guard:
            return True
        return self.jev is not None and bool(getattr(self.jev, "available", False))

    # ------------------------------------------------------------------ requests

    def request(self, facts: Mapping[str, Any], *, state_version: int, t_occ: float,
                expected_window_s: float = 0.0, state_independent: bool = False,
                kinds: Sequence[str] | None = None, event_ids: Sequence[str] = ()) -> int:
        """Start up to ``n_parallel`` candidates; returns how many were started (0 when the window is
        too short, the guard cannot run, or enough are already pending or ready)."""
        if not self.guard_available:
            self.counts["skipped:no_guard"] += 1
            return 0
        version = STATE_INDEPENDENT if state_independent else int(state_version)
        if not state_independent and not self.window_ok(expected_window_s):
            self.counts["skipped:short_window"] += 1
            return 0
        have = (sum(1 for p in self._pending.values() if p.state_version == version)
                + sum(1 for c in self._ready if c.state_version == version))
        want = max(0, self.config.n_parallel - have)
        hints = list(kinds or self.config.kinds) or ["stat_note"]
        packet = dict(facts)
        started = 0
        for i in range(want):
            kind = hints[i % len(hints)]
            cid = f"c{next(self._ids):04d}"
            task = asyncio.create_task(self._generate(cid, kind, packet, version, float(t_occ), list(event_ids)),
                                       name=f"candidate-{cid}")
            self._pending[cid] = _Pending(task, version, kind)
            task.add_done_callback(lambda _t, c=cid: self._pending.pop(c, None))
            started += 1
        self.counts["requested"] += started
        return started

    def set_version(self, version: int) -> int:
        """The fact store moved to ``version``: drop stale ready candidates and cancel stale requests."""
        self._version = int(version)
        stale = [c for c in self._ready if not c.state_independent and c.state_version != self._version]
        for c in stale:
            self._ready.remove(c)
        cancelled = 0
        for p in list(self._pending.values()):
            if p.state_version != STATE_INDEPENDENT and p.state_version != self._version and not p.task.done():
                p.task.cancel()
                cancelled += 1
        if stale or cancelled:
            self.counts["dropped:stale"] += len(stale) + cancelled
            log.debug("candidates: version %d dropped %d ready, cancelled %d pending", version, len(stale), cancelled)
        return len(stale) + cancelled

    # ------------------------------------------------------------------ hand-out

    def ready(self, now: float | None = None) -> list[Candidate]:
        self._expire(self._clock() if now is None else now)
        return list(self._ready)

    def pending(self, state_independent: bool | None = None) -> int:
        return sum(1 for p in self._pending.values()
                   if state_independent is None or (p.state_version == STATE_INDEPENDENT) == state_independent)

    def take(self, now: float | None = None, *, kinds: Sequence[str] = (), recent_lines: Sequence[str] = (),
             allow_dependent: bool = True, allow_independent: bool = True) -> Candidate | None:
        """Pop the best fresh candidate: preferred ``kinds`` first, then the freshest; a candidate
        too similar to a line spoken since it was checked is discarded."""
        self._expire(self._clock() if now is None else now)
        pool = [c for c in self._ready
                if (allow_dependent or c.state_independent) and (allow_independent or not c.state_independent)
                and (c.state_independent or c.state_version == self._version)]
        rank = {k: i for i, k in enumerate(kinds)}
        for c in sorted(pool, key=lambda c: (rank.get(c.kind, len(rank)), c.state_independent, -c.ready_at)):
            if any(jaccard(c.text_tokens, str(r)) >= self.config.repetition_jaccard for r in recent_lines):
                self._ready.remove(c)
                self.counts["dropped:repeat"] += 1
                continue
            self._ready.remove(c)
            self.counts["taken"] += 1
            return c
        return None

    async def wait(self, timeout: float | None = None) -> None:
        """Wait for every pending request (tests and shutdown)."""
        tasks = [p.task for p in self._pending.values()]
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)

    async def aclose(self) -> None:
        tasks = [p.task for p in self._pending.values()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._pending.clear()
        self._ready.clear()

    def stats(self) -> dict[str, Any]:
        return {"counts": dict(self.counts), "ready": len(self._ready), "pending": len(self._pending),
                "threshold_s": self.threshold_s}

    # ------------------------------------------------------------------ generation

    async def _generate(self, cid: str, kind: str, facts: dict[str, Any], version: int, t_occ: float,
                        event_ids: list[str]) -> Candidate | None:
        t0 = self._clock()
        try:
            return await self._generate_inner(cid, kind, facts, version, t_occ, event_ids, t0)
        except asyncio.CancelledError:
            self.counts["cancelled"] += 1
            raise
        except Exception:  # noqa: BLE001 - a broken candidate must never break narration
            log.exception("candidate %s failed", cid)
            self.counts["failed:internal"] += 1
            return None

    async def _generate_inner(self, cid: str, kind: str, facts: dict[str, Any], version: int, t_occ: float,
                              event_ids: list[str], t0: float) -> Candidate | None:
        cfg = self.config
        prompt = build_user_prompt(facts, kind=kind, max_words=cfg.max_words, max_sentences=cfg.max_sentences)
        leaked = self._leaked_names(prompt)
        if leaked:
            log.error("candidate %s blocked: prompt would carry roster names %s (D16)", cid, leaked)
            self.counts["blocked:names"] += 1
            return None
        try:
            text = await asyncio.wait_for(self._collect(prompt), timeout=cfg.candidate_deadline_s)
        except TimeoutError:
            self.counts["failed:deadline"] += 1
            log.info("candidate %s missed its %.1f s deadline", cid, cfg.candidate_deadline_s)
            return None
        except NarrationLLMError as err:
            self.counts[f"failed:{err.reason}"] += 1
            return None
        if self._stale(version):
            self.counts["dropped:stale"] += 1
            return None
        text = text.strip()
        if not text or text.strip(" .").upper() == SILENCE:
            self.counts["silence"] += 1
            return None
        sentences = self._sentences(text)
        problems = self._validate(sentences, facts)
        if problems:
            self.counts["rejected:validator"] += 1
            log.info("candidate %s rejected by the validator: %s", cid, problems)
            return None
        verdicts = await self._guard(facts, sentences)
        if verdicts is None:
            self.counts["rejected:guard"] += 1
            log.info("candidate %s rejected by the Jev guard", cid)
            return None
        if self._stale(version):
            self.counts["dropped:stale"] += 1
            return None
        tokens = " ".join(sentences)
        rendered = self.pseudo.render(tokens, self.roster)
        now = self._clock()
        cand = Candidate(cid, kind, version, tokens, rendered, sentences, facts, t_occ, t0, now,
                         (now - t0) * 1000.0, event_ids, list(verdicts))
        self._ready.append(cand)
        self._trim()
        self.counts["ready"] += 1
        return cand

    async def _collect(self, prompt: str) -> str:
        parts: list[str] = []
        async for chunk in self.llm.stream(self.system_prompt, prompt, self.config.max_tokens):
            parts.append(chunk)
        return "".join(parts)

    def _sentences(self, text: str) -> list[str]:
        out: list[str] = []
        for line in text.splitlines():
            out.extend(split_sentences(line))
        return out

    def _validate(self, sentences: list[str], facts: Mapping[str, Any]) -> list[str]:
        cfg = self.config
        if not sentences:
            return ["no sentence"]
        problems: list[str] = []
        if len(sentences) > cfg.max_sentences:
            problems.append(f"too many sentences: {len(sentences)} > {cfg.max_sentences}")
        total = sum(count_words(s) for s in sentences)
        if total > cfg.max_words:
            problems.append(f"too long: {total} words > {cfg.max_words}")
        for s in sentences:
            res = self.validator.validate(s, facts, cfg.channel)
            problems += [f"{s!r}: {p}" for p in res.problems]
        return problems

    async def _guard(self, facts: Mapping[str, Any], sentences: list[str]) -> list[LineVerdict] | list[Any] | None:
        if not self.config.use_jev_guard:
            return []
        if self.jev is None:
            return None
        verdicts = await self.jev.guard_lines(dict(facts), list(sentences))
        if len(verdicts) != len(sentences) or not all(getattr(v, "passed", False) for v in verdicts):
            return None
        return verdicts

    def _stale(self, version: int) -> bool:
        return version != STATE_INDEPENDENT and version != self._version

    def _expire(self, now: float) -> None:
        cfg = self.config
        keep = []
        for c in self._ready:
            ttl = cfg.independent_ttl_s if c.state_independent else cfg.ready_ttl_s
            if now - c.ready_at <= ttl:
                keep.append(c)
            else:
                self.counts["dropped:expired"] += 1
        self._ready = keep

    def _trim(self) -> None:
        while len(self._ready) > self.config.max_ready:
            self._ready.pop(0)
            self.counts["dropped:buffer_full"] += 1

    def _leaked_names(self, text: str) -> list[str]:
        return sorted({m.group(0) for rx in self._name_res for m in rx.finditer(text)})


def _name_patterns(roster: Roster) -> list[re.Pattern[str]]:
    """Full names and unique surnames (>= 3 letters) of the roster, as whole words."""
    pats: list[re.Pattern[str]] = []
    surnames: Counter[str] = Counter(e.name.split()[-1] for e in roster.entries if len(e.name.split()) >= 2)
    for e in roster.entries:
        if not e.name.strip():
            continue
        pats.append(re.compile(rf"(?<!\w){re.escape(e.name)}(?!\w)"))
        parts = e.name.split()
        if len(parts) >= 2 and surnames[parts[-1]] == 1 and len(parts[-1]) >= 3:
            pats.append(re.compile(rf"(?<!\w){re.escape(parts[-1])}(?!\w)"))
    return pats
