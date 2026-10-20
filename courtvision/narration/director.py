"""Deterministic narration director: speak-now, priority, TTL, cooldowns, preemption (D3; narration.md §9).

Priorities: P0 critical (corrections, clutch makes) > P1 live calls > P2 context > P3 colour.

* TTL (if not started, from the moment it was offered): release call 0.8 s, other P1 2.0 s, P2 3.0 s,
  P0 3.0 s (corrections 10 s), P3 from the cue itself (end of its dead-ball window).
* A newer P1 replaces any queued P1 of the same ``group`` (possession) on the same channel (latest
  wins) and, by default, flushes queued P2/P3 (live play has resumed).
* A minimum human-like gap after each line; per-channel and per-kind cooldowns. P0 skips them.
* Preemption is a flag (:meth:`Director.preempt_requested`) that the TTS layer polls to cut the
  current line at the next word boundary: P0 cuts P1-P3, P1 cuts P2/P3, P2 cuts P3; P1 never cuts P1.
* Hedged lines: at most 1 in any 3 consecutive utterances.
* Corrections: a revision before speech is silent (the queued cue is invalidated); a wrong
  *material* fact already spoken asks for one P0 correction (never twice per event). More than 2
  spoken corrections within 5 game minutes raise ``gate_boost`` (+0.03) and ``hold_boost_s``
  (+0.2 s) for the rest of the period.

Use one Director per output stream (fan audio: play_by_play + color + correction; AD and coach
channels get their own instances).
"""

from __future__ import annotations

import itertools
import logging
from collections import deque
from dataclasses import dataclass, field

from courtvision.types import Event, EventKind, EventStatus

log = logging.getLogger(__name__)

# Kinds whose already-spoken facts get an audio correction when revised (narration.md §3.6).
MATERIAL_KINDS = frozenset({EventKind.SHOT_MADE, EventKind.SHOT_MISSED, EventKind.SHOT_ATTEMPT, EventKind.FOUL,
                            EventKind.TIMEOUT, EventKind.SCORE_CORRECTION})
# Retracting one of these removes a spoken material fact (points, a foul, a timeout). A retracted
# miss or attempt removes no points: the engine flips a miss to a make in place instead.
_RETRACT_MATERIAL = frozenset({EventKind.SHOT_MADE, EventKind.FOUL, EventKind.TIMEOUT, EventKind.SCORE_CORRECTION})
_RESULT_KINDS = frozenset({EventKind.SHOT_MADE, EventKind.SHOT_MISSED})
RELEASE_KINDS = frozenset({"release", "release3"})  # template families of the anticipatory release call


@dataclass
class Cue:
    """One candidate line. ``text_template`` holds ``{Pnn}`` tokens (rendered after validation)."""

    cue_id: str
    priority: int  # 0..3; 0 = highest
    text_template: str
    t_occ: float  # video time of the event it describes
    ttl_s: float = 0.0  # only P3 uses it (end of the dead-ball window); P0-P2 TTLs come from config
    event_ids: list[str] = field(default_factory=list)
    channel: str = "play_by_play"
    hedge: bool = False
    state_version: int = 0
    kind: str = ""  # extension: template family ("release", "made3", "correction", "score_update", ...)
    group: str | None = None  # extension: possession key for "latest P1 wins"
    events: list[Event] = field(default_factory=list)  # extension: event versions the text was rendered from


@dataclass
class DirectorConfig:
    """Director timing (narration.md §9.1/§9.5; design proposals unless noted, tune on labelled games)."""

    ttl_release_s: float = 0.8  # release call (narration.md §9.1); unverified
    ttl_p0_s: float = 3.0  # P0 critical (narration.md §9.1); unverified
    ttl_correction_s: float = 10.0  # corrections (narration.md §3.6: <= 10 s TTL); unverified
    ttl_p1_s: float = 2.0  # live calls, after confirmation (narration.md §9.1); unverified
    ttl_p2_s: float = 3.0  # context lines (spec; narration.md proposes 8 s or until the next P1); unverified
    ttl_p3_default_s: float = 20.0  # P3 cue without its own TTL: <= 20 s dead-ball window (§9.1); unverified
    min_gap_s: float = 0.6  # min silence between lines, live ball (narration.md §9.5; human 1.7 s [2]); sourced
    cooldown_s: dict[str, float] = field(default_factory=lambda: {
        "play_by_play": 0.6, "color": 1.0, "correction": 0.0, "accessibility": 0.0, "coach": 0.0,
    })  # per-channel min spacing (narration.md §9.5: 0.6 s live, 1.0 s dead ball); sourced (proposal)
    kind_cooldown_s: dict[str, float] = field(default_factory=lambda: {
        "score_update": 60.0, "run": 60.0,
    })  # score update <= 1 per 60 s (narration.md §9.5; video seconds stand in for game clock); unverified
    hedge_window: int = 3  # hedged phrasing <= 1 in 3 consecutive utterances (narration.md §3.5); sourced
    max_hedges_in_window: int = 1
    hedge_cap_channels: tuple[str, ...] = ("play_by_play", "color")  # AD / coach state hedges plainly
    max_queued: dict[int, int] = field(default_factory=lambda: {0: 4, 1: 1, 2: 3, 3: 2})  # §9.4 buffer depth
    p1_latest_wins: bool = True  # newer P1 replaces queued P1 of the same group (narration.md §4.2); sourced
    p1_flushes_lower: bool = True  # a P1 offer drops queued P2/P3 (P2 lives "until the next P1", §9.1)
    p0_ignores_gap: bool = True  # P0 skips gap and cooldowns (it cuts in at the next word boundary)
    speaking_timeout_s: float = 30.0  # forget a current line never reported spoken after this long
    correction_storm_count: int = 2  # more than this many spoken corrections... (narration.md §3.6); sourced
    correction_storm_window_s: float = 300.0  # ...within 5 game minutes (same); sourced (proposal)
    gate_boost_step: float = 0.03  # then raise live gates by +0.03 (same); sourced (proposal)
    hold_boost_step_s: float = 0.2  # and widen holds by +200 ms (same); sourced (proposal)


@dataclass
class RevisionOutcome:
    """What the director did with an event revision (see :meth:`Director.on_revision`)."""

    action: str  # "none" | "invalidated" (silent) | "correct" (offer a P0 correction) | "suppressed"
    material: bool = False
    dropped: int = 0  # queued cues invalidated
    spoken: Event | None = None  # the version that was spoken (for the correction line)
    preempt: bool = False  # the current line refers to the revised event (cut it)


@dataclass
class _Entry:
    cue: Cue
    offered_at: float
    expires_at: float
    seq: int


_MISSING = object()


class Director:
    """Owns the queue of candidate cues; never blocks, never calls the network."""

    def __init__(self, config: DirectorConfig | None = None) -> None:
        self.config = config or DirectorConfig()
        self._queue: list[_Entry] = []
        self._seq = itertools.count()
        self._current: _Entry | None = None
        self._current_since = 0.0
        self._current_invalid = False
        self._last_end: float | None = None
        self._last_channel: dict[str, float] = {}
        self._last_kind: dict[str, float] = {}
        self._hedges: deque[bool] = deque(maxlen=max(0, self.config.hedge_window - 1))
        self._spoken: dict[str, Event | None] = {}
        self._corrected: set[str] = set()
        self._correction_times: deque[float] = deque()
        self._escalated = False

    # ------------------------------------------------------------------ queue

    def effective_ttl(self, cue: Cue) -> float:
        """TTL the director applies to ``cue`` (P3 from the cue, P0-P2 from config)."""
        cfg = self.config
        if cue.priority >= 3:
            return cue.ttl_s if cue.ttl_s > 0 else cfg.ttl_p3_default_s
        if _is_correction(cue):
            return cfg.ttl_correction_s
        if cue.kind in RELEASE_KINDS:
            return cfg.ttl_release_s
        return {0: cfg.ttl_p0_s, 1: cfg.ttl_p1_s}.get(cue.priority, cfg.ttl_p2_s)

    def offer(self, cue: Cue, now: float) -> None:
        """Enqueue ``cue``. A higher priority than the current line raises the preemption flag."""
        if not 0 <= cue.priority <= 3:
            raise ValueError(f"cue priority must be 0..3, got {cue.priority}")
        if not cue.text_template.strip():
            return
        self._expire(now)
        cfg = self.config
        drop = [e for e in self._queue if e.cue.channel == cue.channel and e.cue.text_template == cue.text_template]
        if cue.priority == 1 and cfg.p1_latest_wins:
            drop += [e for e in self._queue
                     if e.cue.priority == 1 and e.cue.channel == cue.channel and e.cue.group == cue.group]
        if cue.priority == 1 and cfg.p1_flushes_lower:
            drop += [e for e in self._queue if e.cue.priority >= 2]
        for e in drop:
            if e in self._queue:
                self._queue.remove(e)
                log.debug("cue %s dropped by newer cue %s", e.cue.cue_id, cue.cue_id)
        self._queue.append(_Entry(cue, now, now + self.effective_ttl(cue), next(self._seq)))
        same = sorted((e for e in self._queue if e.cue.priority == cue.priority), key=lambda e: e.seq)
        for e in same[:max(0, len(same) - cfg.max_queued.get(cue.priority, 3))]:
            self._queue.remove(e)
            log.debug("cue %s dropped: P%d buffer full", e.cue.cue_id, cue.priority)

    def next(self, now: float) -> Cue | None:
        """Pop the best non-expired cue that respects gap, cooldowns and the hedge cap.

        While a line is speaking, only a cue that preempts it is returned (the current line is then
        treated as interrupted)."""
        self._expire(now)
        if self._current is not None:
            pre = self._best_preemptor()
            if pre is None:
                return None
            log.debug("cue %s preempts %s", pre.cue.cue_id, self._current.cue.cue_id)
            self._current = None
            return self._start(pre, now)
        for e in self._ordered():
            if self._eligible(e, now):
                return self._start(e, now)
        return None

    def on_spoken(self, cue: Cue, now: float, game_time: float | None = None) -> None:
        """``cue`` finished playing at ``now``. ``game_time`` (elapsed game seconds) scopes the
        correction-storm window; video time is used when it is None."""
        if self._current is not None and self._current.cue.cue_id == cue.cue_id:
            self._current = None
            self._current_invalid = False
        self._queue = [e for e in self._queue if e.cue.cue_id != cue.cue_id]
        self._last_end = now
        self._last_channel[cue.channel] = now
        if cue.kind:
            self._last_kind[cue.kind] = now
        if cue.channel in self.config.hedge_cap_channels and self._hedges.maxlen:
            self._hedges.append(cue.hedge)
        snapshots = {e.event_id: e for e in cue.events}
        for eid in cue.event_ids:
            self._spoken[eid] = snapshots.get(eid, self._spoken.get(eid))
        if _is_correction(cue):
            self._corrected.update(cue.event_ids)
            self._note_correction(now if game_time is None else game_time)

    def on_interrupted(self, cue: Cue, now: float) -> None:
        """The current line was cut before it finished (nothing is recorded as spoken)."""
        if self._current is not None and self._current.cue.cue_id == cue.cue_id:
            self._current = None
            self._current_invalid = False

    def preempt_requested(self, now: float) -> bool:
        """True when the current line should be cut at the next word boundary."""
        if self._current is None:
            return False
        if self._current_invalid:
            return True
        self._expire(now)
        return self._best_preemptor() is not None

    def invalidate(self, state_version: int | None = None, event_ids: list[str] | None = None,
                   min_priority: int = 0) -> int:
        """Drop queued cues older than ``state_version`` or referring to ``event_ids`` (with
        priority >= ``min_priority``); returns how many. A matching current line raises the
        preemption flag."""
        ids = set(event_ids or ())

        def stale(c: Cue) -> bool:
            if c.priority < min_priority:
                return False
            old = state_version is not None and c.state_version < state_version
            return old or bool(ids and ids.intersection(c.event_ids))

        before = len(self._queue)
        self._queue = [e for e in self._queue if not stale(e.cue)]
        if self._current is not None and stale(self._current.cue):
            self._current_invalid = True
        dropped = before - len(self._queue)
        if dropped:
            log.debug("invalidated %d queued cue(s) (version<%s, events=%s)", dropped, state_version, sorted(ids))
        return dropped

    # ------------------------------------------------------------------ corrections

    def on_revision(self, new: Event, now: float, old: Event | None = None) -> RevisionOutcome:
        """Apply the correction policy to a revised event (narration.md §3.6, RESEARCH.md §4.2).

        Compares ``new`` with the version that was spoken (or ``old``). Queued cues about it are
        dropped silently; if a wrong material fact was already spoken, returns ``action="correct"``
        and the caller offers one P0 correction cue (``kind="correction"``, ``events=[new]``)."""
        eid = new.event_id
        spoken = self._spoken.get(eid, _MISSING)
        current_refs = self._current is not None and eid in self._current.cue.event_ids
        base = self._snapshot(eid, spoken, current_refs) or old
        if base is not None and not fact_changed(base, new):
            # Back to what was heard: queued lines rendered from an in-between version (e.g. a
            # correction not yet spoken) are now wrong and go silently.
            dropped = self._drop_stale(eid, new)
            return RevisionOutcome("invalidated" if dropped else "none", dropped=dropped,
                                   spoken=base if spoken is not _MISSING else None)
        dropped = self.invalidate(event_ids=[eid])
        if current_refs:
            self._current_invalid = True
        material = is_material_change(base, new)
        was_spoken = spoken is not _MISSING or current_refs
        if not was_spoken:
            return RevisionOutcome("invalidated" if dropped else "none", material, dropped, None, False)
        if eid in self._corrected or not material:
            return RevisionOutcome("suppressed", material, dropped, base, current_refs)
        return RevisionOutcome("correct", True, dropped, base, current_refs)

    def _drop_stale(self, eid: str, new: Event) -> int:
        """Drop queued cues whose snapshot of ``eid`` differs from ``new``; flag a stale current line."""
        def stale(c: Cue) -> bool:
            return any(e.event_id == eid and fact_changed(e, new) for e in c.events)

        before = len(self._queue)
        self._queue = [e for e in self._queue if not stale(e.cue)]
        if self._current is not None and stale(self._current.cue):
            self._current_invalid = True
        return before - len(self._queue)

    def _snapshot(self, eid: str, spoken: object, current_refs: bool) -> Event | None:
        """Newest version of ``eid`` the audience heard or is about to hear."""
        if isinstance(spoken, Event):
            return spoken
        cues = ([self._current.cue] if current_refs and self._current is not None else []) + \
            [e.cue for e in self._ordered()]
        for c in cues:
            for e in c.events:
                if e.event_id == eid:
                    return e
        return None

    def _note_correction(self, t: float) -> None:
        cfg = self.config
        self._correction_times.append(t)
        while self._correction_times and t - self._correction_times[0] > cfg.correction_storm_window_s:
            self._correction_times.popleft()
        if len(self._correction_times) > cfg.correction_storm_count and not self._escalated:
            self._escalated = True
            log.warning("correction storm: %d corrections in %.0f s; gates +%.2f, holds +%.1f s",
                        len(self._correction_times), cfg.correction_storm_window_s, cfg.gate_boost_step,
                        cfg.hold_boost_step_s)

    def on_period_start(self) -> None:
        """New period: the correction-storm escalation ends."""
        self._escalated = False
        self._correction_times.clear()

    @property
    def gate_boost(self) -> float:
        """Added to every live confidence gate while a correction storm is active."""
        return self.config.gate_boost_step if self._escalated else 0.0

    @property
    def hold_boost_s(self) -> float:
        """Added to every confirmation hold while a correction storm is active."""
        return self.config.hold_boost_step_s if self._escalated else 0.0

    @property
    def hold_boost(self) -> float:
        """Alias of :attr:`hold_boost_s` (seconds)."""
        return self.hold_boost_s

    @property
    def escalated(self) -> bool:
        return self._escalated

    # ------------------------------------------------------------------ inspection

    @property
    def current(self) -> Cue | None:
        return self._current.cue if self._current is not None else None

    def pending(self, now: float | None = None) -> list[Cue]:
        """Queued cues in selection order (expired ones removed when ``now`` is given)."""
        if now is not None:
            self._expire(now)
        return [e.cue for e in self._ordered()]

    # ------------------------------------------------------------------ internals

    def _ordered(self) -> list[_Entry]:
        # Highest priority first; within a priority the freshest (latest event, then latest offer).
        return sorted(self._queue, key=lambda e: (e.cue.priority, -e.cue.t_occ, -e.seq))

    def _expire(self, now: float) -> None:
        for e in [e for e in self._queue if now > e.expires_at]:
            self._queue.remove(e)
            log.debug("cue %s expired (P%d, ttl %.1f s)", e.cue.cue_id, e.cue.priority, e.expires_at - e.offered_at)
        if self._current is not None and now - self._current_since > self.config.speaking_timeout_s:
            log.warning("cue %s never reported spoken; releasing it", self._current.cue.cue_id)
            self._current = None
            self._current_invalid = False

    def _hedge_blocked(self, cue: Cue) -> bool:
        if not cue.hedge or cue.channel not in self.config.hedge_cap_channels:
            return False
        return sum(self._hedges) >= self.config.max_hedges_in_window

    def _eligible(self, e: _Entry, now: float) -> bool:
        cue, cfg = e.cue, self.config
        if self._hedge_blocked(cue):
            return False
        if cue.priority == 0 and cfg.p0_ignores_gap:
            return True
        if self._last_end is not None and now - self._last_end < cfg.min_gap_s:
            return False
        last = self._last_channel.get(cue.channel)
        if last is not None and now - last < cfg.cooldown_s.get(cue.channel, 0.0):
            return False
        last_kind = self._last_kind.get(cue.kind) if cue.kind else None
        return last_kind is None or now - last_kind >= cfg.kind_cooldown_s.get(cue.kind, 0.0)

    def _best_preemptor(self) -> _Entry | None:
        if self._current is None:
            return None
        cur = self._current.cue.priority
        for e in self._ordered():
            if e.cue.priority < cur and not self._hedge_blocked(e.cue):
                return e
        return None

    def _start(self, e: _Entry, now: float) -> Cue:
        self._queue.remove(e)
        self._current = e
        self._current_since = now
        self._current_invalid = False
        return e.cue


# --------------------------------------------------------------------------- revision helpers


def _is_correction(cue: Cue) -> bool:
    """A correction cue: the correction channel, or kind "correction" / a "correction:<key>" family."""
    return cue.channel == "correction" or cue.kind.split(":", 1)[0] == "correction"


def _fact_tuple(e: Event) -> tuple:
    return (e.kind, e.team, e.player_id, e.other_player_id, int(e.value or 0),
            e.status is EventStatus.RETRACTED, bool(e.attrs.get("offensive", False)), e.attrs.get("type"))


def fact_changed(old: Event, new: Event) -> bool:
    """True when a narrated fact differs (status-only confirmation or confidence changes are not)."""
    return _fact_tuple(old) != _fact_tuple(new)


def is_material_change(old: Event | None, new: Event) -> bool:
    """Material = scorer, points value, score, who fouled, timeout team (narration.md §3.6).
    Assist / rebound / steal / block credit and shot distance are not."""
    if new.kind not in MATERIAL_KINDS:
        return False
    if new.status is EventStatus.RETRACTED:
        return (old.kind if old is not None else new.kind) in _RETRACT_MATERIAL and (
            old is None or old.status is not EventStatus.RETRACTED)
    if old is None:
        return False
    if old.kind is not new.kind and {old.kind, new.kind} <= _RESULT_KINDS:
        return True  # result flipped in place (make <-> miss): the points and the score changed
    if old.status is EventStatus.RETRACTED:
        return new.kind in _RETRACT_MATERIAL  # a withdrawn basket / foul / timeout came back
    if new.kind is EventKind.SHOT_MADE:
        return (old.player_id, old.team, old.value) != (new.player_id, new.team, new.value)
    if new.kind is EventKind.SHOT_ATTEMPT:
        return old.value != new.value and 3 in (old.value, new.value)
    if new.kind is EventKind.FOUL:
        return (old.player_id, old.team) != (new.player_id, new.team)
    if new.kind is EventKind.TIMEOUT:
        return old.team != new.team
    if new.kind is EventKind.SCORE_CORRECTION:
        return (old.team, old.value) != (new.team, new.value)
    return False


def priority_for(family: str, facts: dict | None = None) -> int:
    """Default priority of a template family. P0 (narration.md §9.1): corrections, a made shot in the
    last 2:00 of the final period or OT with a margin <= 5 before the play, and a foul-out."""
    if family.startswith("correction"):
        return 0
    if family in ("made2", "made3", "layup", "dunk") and facts and _clutch(facts):
        return 0
    if family == "foul" and facts and any(str(d).endswith("has fouled out") for d in facts.get("derived") or ()):
        return 0
    if family in ("release", "release3", "made2", "made3", "layup", "dunk", "miss", "block", "steal", "turnover",
                  "oreb", "dreb", "team_reb", "foul", "whistle", "hedge_shot", "hedge_loose", "hedge_stop",
                  "hedge_rebound"):
        return 1
    if family.startswith("color"):
        return 3
    return 2


def _clutch(facts: dict, window_s: float = 120.0, margin: int = 5) -> bool:
    clock, score, ev = facts.get("clock") or {}, facts.get("score") or {}, facts.get("event") or {}
    secs = clock.get("seconds_left")
    if not clock.get("is_last_period") or secs is None or secs > window_s or score.get("home") is None:
        return False
    sign = {"home": 1, "away": -1}.get(str(ev.get("team_side")), 0)
    before = int(score["home"]) - int(score["away"]) - sign * int(ev.get("value") or 0)
    return abs(before) <= margin
