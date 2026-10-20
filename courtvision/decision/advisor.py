"""JevDecider: the TypeSafe Jev decision-layer façade (ARCHITECTURE "decision/").

Every method computes its deterministic fallback FIRST and returns it (``source="fallback"``)
when Jev is off, unavailable, late, failing, breaker-open, answered by an unpinned model, or below
threshold, so a missing Jev answer only makes output plainer, never wrong. Methods never raise.

State dicts may carry a top-level ``"_local"`` block (built by ``decision.states``): numbers for the
fallbacks, ``redact_names``, ``dead_ball_id``, ``storylines_due_now``, ``forbidden``. It never leaves
the device.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from courtvision.decision import questions as Q
from courtvision.decision import states as S
from courtvision.decision.client import JevClient, JevConfig, JevResult
from courtvision.decision.decoders import (
    ABSTAIN,
    FallbackRules,
    Judgement,
    LineVerdict,
    PossessionAssessment,
    Thresholds,
    ambiguity_judgement,
    as_fallback,
    decide_ambiguity,
    decode_cards,
    decode_guard,
    decode_line_kind,
    decode_noul_yes,
    decode_play_type,
    decode_stakes,
    fallback,
    fallback_highlight,
    fallback_play_type,
    fallback_stakes,
    guard_fallback,
    unusable_reason,
)
from courtvision.types import Insight

__all__ = ["JevConfig", "JevDecider", "Judgement", "LineVerdict", "PossessionAssessment"]

log = logging.getLogger(__name__)


class JevDecider:
    """Async façade over ``JevClient`` (a.k.a. JevAdvisor in the research notes)."""

    def __init__(self, config: JevConfig | None = None, client: JevClient | None = None) -> None:
        self.config = config or (client.config if client is not None else JevConfig())
        self._client = client or JevClient(self.config)
        self.thresholds = Thresholds.from_overrides(self.config.thresholds)
        self.rules = FallbackRules.from_overrides(self.config.fallback_rules)
        self.state_config = S.StateConfig()
        self._methods: dict[str, Counter] = defaultdict(Counter)
        self._kind_last_window: dict[str, int] = {}  # line kind -> dead-ball window it was last used
        self._window = 0

    # ---------------------------------------------------------------- properties

    @property
    def client(self) -> JevClient:
        return self._client

    @property
    def available(self) -> bool:
        """False when off, no key, or the session was disabled (401/403)."""
        return self._client.available

    # ---------------------------------------------------------------- B1 possession_end

    async def possession_end(self, state: dict) -> PossessionAssessment:
        """B1: play type (order-ensembled), summary clarity, moment stakes and highlight."""
        try:
            return await self._possession_end(state)
        except Exception:  # noqa: BLE001 - last resort: never raise into the caller
            log.exception("jev possession_end failed; plain fallback")
            self._count("possession_end", "fallback:internal_error")
            return PossessionAssessment(play_type=fallback(ABSTAIN), summary_clear=fallback(None),
                                        stakes=fallback("routine", value=0.0), highlight=fallback("no", value=0.0))

    async def _possession_end(self, state: dict) -> PossessionAssessment:
        wire, local = self._prepare(state, "B1", droppable=[("possession", "actions")])
        fb = PossessionAssessment(
            play_type=fallback_play_type(local, self.rules),
            summary_clear=fallback(None),
            stakes=fallback_stakes(wire.get("situation") or {}, local),
            highlight=fallback_highlight(wire.get("last_play") or {}, local),
        )
        r, ms = await self._ask("possession_end", "B1", wire, Q.B1_POSSESSION_END)
        if r is None:
            return self._stamp(fb, ms)
        try:
            play = decode_play_type(r, self.thresholds)
            rule_wins = fb.play_type.label in self.rules.crisp_labels and fb.play_type.confidence == 1.0
            if rule_wins:  # "when a rule fires, it wins"; Jev's label is only logged for evaluation
                log.debug("jev play type %s, rule %s wins", play.label if play else None, fb.play_type.label)
            parts = {
                "play_type": (None if rule_wins else play, fb.play_type),
                "summary_clear": (decode_noul_yes(r, "summary_is_clear", self.thresholds.summary_clear_p),
                                  fb.summary_clear),
                "stakes": (decode_stakes(r, self.thresholds), fb.stakes),
                "highlight": (decode_noul_yes(r, "highlight_play", self.thresholds.highlight_p), fb.highlight),
            }
        except Exception:  # noqa: BLE001 - a malformed answer must not break the caller
            log.exception("jev B1 decode failed")
            self._count("possession_end", "fallback:decode_error")
            return self._stamp(fb, ms)
        if rule_wins:
            self._count("possession_end", "rule:play_type")
            out = {"play_type": as_fallback(fb.play_type, ms)}
            parts.pop("play_type")
        else:
            out = {}
        out.update({name: self._pick("possession_end", name, jev, fbj, ms) for name, (jev, fbj) in parts.items()})
        return PossessionAssessment(**out)

    # ---------------------------------------------------------------- B2 dead_ball_fan

    async def next_line_kind(self, state: dict, available_kinds: set[str],
                             storylines: list[str]) -> tuple[Judgement, list[Judgement]]:
        """B2: the kind of the next colour line (or silence) plus one ``storyline_due_i`` per storyline."""
        try:
            return await self._next_line_kind(state, available_kinds, storylines)
        except Exception:  # noqa: BLE001 - last resort: never raise into the caller
            log.exception("jev next_line_kind failed; silence")
            self._count("next_line_kind", "fallback:internal_error")
            n = min(len(storylines or ()), Q.MAX_STORYLINES)
            return fallback("silence"), [fallback("no") for _ in range(n)]

    async def _next_line_kind(self, state: dict, available_kinds: set[str],
                              storylines: list[str]) -> tuple[Judgement, list[Judgement]]:
        wire, local = self._prepare(state, None)
        stories = [str(s) for s in storylines][: Q.MAX_STORYLINES]
        kinds = [k for k in self.rules.line_kind_rotation if k in available_kinds and k in Q.LINE_KINDS
                 and k != "silence" and (k != "storyline" or stories)]
        due = [bool(x) for x in (local.get("storylines_due_now") or [])][: len(stories)]
        due += [False] * (len(stories) - len(due))
        window = self._window
        self._window += 1
        fb_kind = self._line_kind_fallback(kinds, due, window)
        fb_stories = [fallback("yes" if d else "no", fired=d) for d in due]
        # Design §6.2: a kind whose material is missing is absent from both `material` and the options
        # (the question's criteria point at `material.<kind>`); the fallback still uses the caller's kinds.
        material = wire.get("material") if isinstance(wire.get("material"), Mapping) else {}
        jev_kinds = [k for k in kinds if k == "storyline" or material.get(k)]
        if not jev_kinds:  # only "silence" could be offered: no question worth asking
            self._count("next_line_kind", "calls")
            self._count("next_line_kind", "fallback:nothing_to_say")
            return self._note_kind(fb_kind, window), fb_stories
        wire = self._b2_wire(wire, jev_kinds, stories, local)
        r, ms = await self._ask("next_line_kind", "B2", wire, Q.b2_dead_ball_fan(set(jev_kinds), len(stories)))
        if r is None:
            return self._note_kind(as_fallback(fb_kind, ms), window), [as_fallback(j, ms) for j in fb_stories]
        try:
            jev_kind = decode_line_kind(r, [*jev_kinds, "silence"], self.thresholds)
            jev_stories = [decode_noul_yes(r, f"storyline_due_{i}", self.thresholds.storyline_p)
                           for i in range(len(stories))]
        except Exception:  # noqa: BLE001
            log.exception("jev B2 decode failed")
            self._count("next_line_kind", "fallback:decode_error")
            return self._note_kind(as_fallback(fb_kind, ms), window), [as_fallback(j, ms) for j in fb_stories]
        kind = self._pick("next_line_kind", "kind", jev_kind, fb_kind, ms)
        out_stories = [self._pick("next_line_kind", "storyline", j, f, ms) for j, f in zip(jev_stories, fb_stories)]
        return self._note_kind(kind, window), out_stories

    def _b2_wire(self, wire: dict, kinds: list[str], stories: list[str], local: Mapping[str, Any]) -> dict:
        material = {k: v for k, v in (wire.get("material") or {}).items() if k in kinds}
        wire = {k: v for k, v in wire.items() if k not in ("material", "storylines")}
        if material:
            wire["material"] = material
        if stories:
            wire["storylines"] = stories
        wire = S.redact_state(wire, local.get("redact_names") or ())
        return S.fit_state(wire, self.state_config.max_state_bytes.get("B2", 1024), droppable=[("recent_topics",)])

    def _line_kind_fallback(self, kinds: list[str], due: list[bool], window: int) -> Judgement:
        """Due storyline first; otherwise the least-recently-used kind on every ``speak_every``-th
        dead ball (deterministic rotation), else silence."""
        if "storyline" in kinds and any(due):
            return fallback("storyline", fired=True)
        if not kinds or window % max(1, self.rules.speak_every) != 0:
            return fallback("silence")
        order = self.rules.line_kind_rotation
        label = min(kinds, key=lambda k: (self._kind_last_window.get(k, -1), order.index(k)))
        return fallback(label)

    def _note_kind(self, j: Judgement, window: int) -> Judgement:
        if j.label and j.label != "silence":
            self._kind_last_window[j.label] = window
        return j

    # ---------------------------------------------------------------- B3 dead_ball_coach

    async def pick_insight(self, candidates: list[Insight], ctx: dict) -> Insight | None:
        """B3: the single code-made card worth pushing now (labels ``card_a``..``card_h``).

        Fallback: the highest-priority candidate (first on ties). Returns None when Jev confidently
        answers "none".
        """
        if not candidates:
            return None
        ranked = sorted(range(len(candidates)), key=lambda i: -(candidates[i].priority or 0))
        fb = candidates[ranked[0]]
        try:
            return await self._pick_insight(candidates, ranked, fb, ctx)
        except Exception:  # noqa: BLE001 - last resort: never raise into the caller
            log.exception("jev pick_insight failed; highest-priority card")
            self._count("pick_insight", "fallback:internal_error")
            return fb

    async def _pick_insight(self, candidates: list[Insight], ranked: list[int], fb: Insight,
                            ctx: dict) -> Insight | None:
        if not self._client.available:  # skip state building when no call can happen
            self._count("pick_insight", "calls")
            self._count("pick_insight", f"fallback:{self._client.disabled_reason or 'unavailable'}")
            return fb
        chosen = ranked[: self.state_config.max_cards]
        state, key_map = self._b3_state([candidates[i].text for i in chosen], ctx)
        r, _ms = await self._ask("pick_insight", "B3", state, Q.b3_dead_ball_coach(state["cards"]))
        if r is None:
            return fb
        try:
            dec = decode_cards(r, list(key_map), self.thresholds)
        except Exception:  # noqa: BLE001
            log.exception("jev B3 decode failed")
            self._count("pick_insight", "fallback:decode_error")
            return fb
        if dec is None or dec.top is None:
            self._count("pick_insight", "below:top_card")
            return fb
        self._count("pick_insight", "jev:top_card")
        if dec.top == "none":
            return None
        return candidates[chosen[key_map[dec.top]]]

    def _b3_state(self, texts: list[str], ctx: Mapping[str, Any]) -> tuple[dict, dict[str, int]]:
        """``ctx``: {"situation": block or GameSituation, "staff_team", "timeouts", "bonus", "_local"},
        or the situation block itself. Cards are shuffled with the dead-ball id (else a text hash)."""
        wire_ctx, local = S.split_local(ctx or {})
        situation = wire_ctx.get("situation", wire_ctx)
        extra_keys = ("staff_team", "timeouts", "bonus")
        sit_map = situation if isinstance(situation, Mapping) else {}
        extras = {k: wire_ctx.get(k) or sit_map.get(k) for k in extra_keys}
        if isinstance(situation, Mapping):
            situation = {k: v for k, v in situation.items() if k not in extra_keys and k != "situation"}
        seed = local.get("dead_ball_id") or hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]
        return S.b3_dead_ball_coach(situation, texts, seed=seed, names=local.get("redact_names") or (),
                                    cfg=self.state_config, **extras)

    # ---------------------------------------------------------------- B4 line_guard

    async def guard_lines(self, facts: dict, sentences: list[str]) -> list[LineVerdict]:
        """B4: per-sentence grounding verdicts (<= 2 sentences per request; longer lists are split
        into concurrent requests). Fallback ``passed=False`` (D4)."""
        if not sentences:
            return []
        try:
            wire_facts, local = S.split_local(facts or {})
            names = local.get("redact_names") or ()
            chunks = [sentences[i:i + Q.MAX_GUARD_SENTENCES]
                      for i in range(0, len(sentences), Q.MAX_GUARD_SENTENCES)]
            results = await asyncio.gather(*(self._guard_chunk(wire_facts, c, names) for c in chunks))
            return [v for chunk in results for v in chunk]
        except Exception:  # noqa: BLE001 - last resort: never raise into the caller
            log.exception("jev guard_lines failed; nothing passes")
            self._count("guard_lines", "fallback:internal_error")
            return guard_fallback(list(sentences))

    async def _guard_chunk(self, facts: Mapping[str, Any], chunk: list[str], names: Sequence[str]) -> list[LineVerdict]:
        try:
            return await self._guard_chunk_inner(facts, chunk, names)
        except Exception:  # noqa: BLE001 - one bad chunk must not fail its siblings in the gather
            log.exception("jev B4 chunk failed; not passed")
            self._count("guard_lines", "fallback:internal_error")
            return guard_fallback(chunk)

    async def _guard_chunk_inner(self, facts: Mapping[str, Any], chunk: list[str],
                                 names: Sequence[str]) -> list[LineVerdict]:
        fb = guard_fallback(chunk)
        if not self._client.available:
            self._count("guard_lines", "calls")
            self._count("guard_lines", f"fallback:{self._client.disabled_reason or 'unavailable'}")
            return fb
        state = S.b4_line_guard(facts, chunk, names=names, cfg=self.state_config)
        r, _ms = await self._ask("guard_lines", "B4", state, Q.b4_line_guard(len(chunk)))
        if r is None:
            return fb
        try:
            verdicts = decode_guard(r, chunk, self.thresholds)
        except Exception:  # noqa: BLE001
            log.exception("jev B4 decode failed")
            self._count("guard_lines", "fallback:decode_error")
            return fb
        for v in verdicts:
            self._count("guard_lines", "jev:passed" if v.passed else "jev:rejected")
        return verdicts

    # ---------------------------------------------------------------- B5 event_ambiguity

    async def ambiguity(self, kind: str, cues: dict, rule_lean: str) -> Judgement:
        """B5 ``shot_outcome`` | ``release_intent``: a provisional label only when Jev's ensembled
        answer agrees across orders, reaches ``event_pmax`` and matches the rule engine's lean;
        otherwise ``unclear``. The caller uses B5 only when its cue model is calibrated."""
        try:
            return await self._ambiguity(kind, cues, rule_lean)
        except Exception:  # noqa: BLE001 - last resort: never raise into the caller
            log.exception("jev ambiguity failed; unclear")
            self._count("ambiguity", "fallback:internal_error")
            return fallback(ABSTAIN)

    async def _ambiguity(self, kind: str, cues: dict, rule_lean: str) -> Judgement:
        fb = fallback(ABSTAIN)
        spec = Q.B5_KINDS.get(kind)
        if spec is None or rule_lean not in spec[4]:
            log.warning("jev B5 refused: kind=%r rule_lean=%r", kind, rule_lean)
            self._count("ambiguity", "calls")
            self._count("ambiguity", "fallback:bad_arguments")
            return fb
        _root, qs, _fwd, _rev, _committable = spec
        wire_cues, local = S.split_local(cues)
        forbidden = frozenset(local.get("forbidden") or ())
        state = S.b5_event_ambiguity(kind, wire_cues, cfg=self.state_config)
        r, ms = await self._ask("ambiguity", "B5", state, qs)
        if r is None:
            return as_fallback(fb, ms)
        try:
            ens = decide_ambiguity(r, kind, rule_lean, forbidden, self.thresholds, self._pinned("B5"))
        except Exception:  # noqa: BLE001
            log.exception("jev B5 decode failed")
            ens = None
        if ens is None:
            self._count("ambiguity", "below:commit")
            return as_fallback(fb, r.latency_ms)
        self._count("ambiguity", "jev:commit")
        return ambiguity_judgement(r, ens)

    # ---------------------------------------------------------------- stats and shutdown

    def stats(self) -> dict:
        """Per-bundle calls / fallbacks / p50 / p95 / tokens / models (client) plus per-method
        outcome counts (jev parts used, below-threshold parts, fallback reasons)."""
        out = self._client.stats()
        methods: dict[str, Any] = {}
        for name, counter in self._methods.items():
            fallbacks = {k.split(":", 1)[1]: v for k, v in counter.items() if k.startswith("fallback:")}
            methods[name] = {
                "calls": counter.get("calls", 0),
                "fallbacks": fallbacks,
                "fallback_total": sum(fallbacks.values()),
                "jev_parts": {k.split(":", 1)[1]: v for k, v in counter.items() if k.startswith("jev:")},
                "below_threshold": {k.split(":", 1)[1]: v for k, v in counter.items() if k.startswith("below:")},
                "rule_wins": {k.split(":", 1)[1]: v for k, v in counter.items() if k.startswith("rule:")},
            }
        out["methods"] = methods
        return out

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---------------------------------------------------------------- internals

    def _pinned(self, key: str) -> str:
        return self._client.bundle(key).model

    def _count(self, method: str, key: str) -> None:
        self._methods[method][key] += 1

    def _prepare(self, state: Mapping[str, Any], bundle: str | None,
                 droppable: Sequence[Sequence[str]] = ()) -> tuple[dict, dict]:
        """Deep-copied wire part (redacted, size-fitted) and the ``_local`` part of a state."""
        wire, local = S.split_local(state or {})
        wire = S.redact_state(copy.deepcopy(wire), local.get("redact_names") or ())
        if bundle is not None:
            S.fit_state(wire, self.state_config.max_state_bytes.get(bundle, 1024), droppable)
        return wire, local

    async def _ask(self, method: str, key: str, wire: Mapping[str, Any],
                   questions: Mapping[str, Any]) -> tuple[JevResult | None, float]:
        """One bundle request -> (usable result or None, ms spent). The fallback reason is counted."""
        self._count(method, "calls")
        if not self._client.available:
            self._count(method, f"fallback:{self._client.disabled_reason or 'unavailable'}")
            return None, 0.0
        r = await self._client.aask(key, wire, questions)
        reason = unusable_reason(r, self._pinned(key))
        if reason is not None:
            self._count(method, f"fallback:{reason}")
            return None, r.latency_ms
        return r, r.latency_ms

    def _pick(self, method: str, part: str, jev: Judgement | None, fb: Judgement, ms: float) -> Judgement:
        if jev is not None:
            self._count(method, f"jev:{part}")
            return jev
        self._count(method, f"below:{part}")
        return as_fallback(fb, ms)

    @staticmethod
    def _stamp(a: PossessionAssessment, ms: float) -> PossessionAssessment:
        return PossessionAssessment(play_type=as_fallback(a.play_type, ms), summary_clear=as_fallback(a.summary_clear, ms),
                                    stakes=as_fallback(a.stakes, ms), highlight=as_fallback(a.highlight, ms))
