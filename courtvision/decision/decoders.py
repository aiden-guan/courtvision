"""Decoders, thresholds and deterministic fallbacks for Jev answers (jev_integration_design.md §7.3-§7.4).

Every decoder takes a ``JevResult`` and returns ``None`` (or a not-passed verdict) when the answer
is unusable or below threshold; the caller then keeps the deterministic fallback, which is always
computed first. Fallback judgements carry ``confidence=1.0`` when a crisp rule fired and ``0.0``
when they are the table's default.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from typesafe_sdk import SystemOneResponse

from courtvision.decision.client import JEV_MODEL_TEXT, JevResult, choice_confidence
from courtvision.decision.questions import B5_KINDS, LINE_KINDS, STAKES_LEVELS
from courtvision.decision.states import (
    CLUTCH_YES,
    FLAG_PHRASES,
    HIGHLIGHT_FLAGS,
    MARGIN_TEMPLATES,
    TIME_TEMPLATES,
    code_from_text,
    finite,
)

log = logging.getLogger(__name__)

ABSTAIN = "unclear"


# --------------------------------------------------------------------------- result types


@dataclass
class Judgement:
    """One decoded answer (or its deterministic fallback)."""

    label: str | None  # Choice label, Score level name, "yes"/"no" for a Noul, or None
    value: float | None  # Noul P(yes), or Score normalised to [0,1] (score / (levels-1))
    confidence: float  # Choice/Score confidence; for Noul: |2p-1|
    probabilities: dict[str, float] = field(default_factory=dict)
    source: str = "fallback"  # "jev" | "cache" | "stub" | "replay" | "fallback"
    latency_ms: float = 0.0
    model: str | None = None  # resp.model that answered (None for fallback)


@dataclass
class PossessionAssessment:
    """B1 possession_end: flavour and play type (tiers A/D)."""

    play_type: Judgement  # Synergy-style label, order-ensembled (play_type + play_type_rev)
    summary_clear: Judgement  # Noul summary_is_clear
    stakes: Judgement  # 4-level Score moment_stakes, value normalised to [0,1]
    highlight: Judgement  # Noul highlight_play


@dataclass
class LineVerdict:
    """B4 line_guard verdict for one sentence."""

    sentence: str
    supported: float | None
    unlisted: float | None
    speculation: float | None
    passed: bool  # supported >= 0.90, unlisted <= 0.20, speculation <= 0.30
    source: str


# --------------------------------------------------------------------------- thresholds and rules


@dataclass(frozen=True)
class Thresholds:
    """Decode thresholds, tuned for jev-1.13.0 (design proposals; re-tune on any model change)."""

    play_type_narr_conf: float = 0.60  # ensembled Choice confidence (~ p_max 0.63 with 12 options)
    play_type_coach_conf: float = 0.70  # coach PPP tables (callers compare Judgement.confidence)
    summary_clear_p: float = 0.60
    stakes_conf: float = 0.50  # Score confidence
    highlight_p: float = 0.80  # Noul review band 0.2/0.8
    line_kind_conf: float = 0.50  # ~ p_max 0.625 with 4 options
    storyline_p: float = 0.75  # narration.md
    card_relevant_p: float = 0.80  # narration.md
    top_card_conf: float = 0.50
    guard_supported_p: float = 0.90  # a false "supported" is costly: high bar
    guard_unlisted_max_p: float = 0.20
    guard_speculation_max_p: float = 0.30
    event_pmax: float = 0.85  # Tier C: jev_api's >= 0.85-and-agree policy

    @classmethod
    def from_overrides(cls, overrides: Mapping[str, float] | None) -> Thresholds:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: float(v) for k, v in (overrides or {}).items() if k in known})


@dataclass(frozen=True)
class FallbackRules:
    """Deterministic rule constants used when Jev is off, late, unusable or below threshold."""

    putback_window_s: float = 2.0  # OffRebound: shot <= 2 s after an offensive rebound (design §3 site 3; proposal)
    putback_max_dribbles: int = 0  # ... "with no dribble" (design §3 site 3; proposal)
    putback_max_passes: int = 0  # ... by the rebounder: a kick-out after the rebound is not a putback (rules note §5; unverified)
    transition_window_s: float = 7.0  # task spec; design crisp <= 5 s, ambiguous 5-8; FIBA fast break <= 8 s (tuned)
    transition_causes: tuple[str, ...] = ("defensive_rebound", "steal")  # live-ball changes of possession
    isolation_max_passes: int = 0  # isolation: no pass in the segment (task spec; unverified)
    isolation_min_dribbles: int = 4  # ... and >= 4 dribbles by the terminal player (task spec; unverified)
    heave_min_distance_ft: float = 40.0  # end-of-period heave > 40 ft (basketball_rules_stats.md; sourced)
    heave_max_clock_s: float = 2.0  # ... with < 2 s left (basketball_rules_stats.md; sourced)
    crisp_labels: tuple[str, ...] = ("OffRebound", "Transition", "Misc")  # "when a rule fires, it wins" (design §3)
    line_kind_rotation: tuple[str, ...] = ("stat_note", "tendency", "storyline")  # LRU tie order
    speak_every: int = 2  # speak at every 2nd dead ball: Mori et al. 0.5 colour rate (sourced via narration.md)

    @classmethod
    def from_overrides(cls, overrides: Mapping[str, Any] | None) -> FallbackRules:
        known = {f.name: f for f in dataclasses.fields(cls)}
        vals = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (overrides or {}).items() if k in known}
        return cls(**vals)


# --------------------------------------------------------------------------- usability and statistics


def usable(r: JevResult, pinned: str = JEV_MODEL_TEXT) -> bool:
    """True only for an answered call from the pinned model (any other version is untuned)."""
    return r.ok and r.response is not None and r.model == pinned


def unusable_reason(r: JevResult, pinned: str = JEV_MODEL_TEXT) -> str | None:
    """None when usable, "unpinned_model" for an answer from another version, else the client error."""
    if usable(r, pinned):
        return None
    if r.ok and r.response is not None:
        return "unpinned_model"
    return r.error or "error"


def noul_confidence(p: float) -> float:
    """Noul has no confidence field; |2p-1| puts it on the Choice scale."""
    return abs(2.0 * p - 1.0)


@dataclass(frozen=True)
class Ensembled:
    """The same Choice asked in two option orders, averaged."""

    label: str  # averaged top label, or the abstain label when the two orders disagree
    p_max: float  # averaged probability of the averaged top label
    confidence: float  # Choice confidence of the averaged distribution
    agree: bool  # both orders picked the same label
    probabilities: dict[str, float]
    raw_top: str  # averaged top label regardless of agreement


def ensembled_choice(resp: SystemOneResponse, a: str, b: str, abstain: str = ABSTAIN) -> Ensembled:
    """Order-ensembling (first-option bias): average both orders; ``abstain`` when they disagree."""
    ca, cb = resp.choices[a], resp.choices[b]
    labels = list(dict.fromkeys([*ca.probabilities, *cb.probabilities]))
    avg = {k: (ca.probabilities.get(k, 0.0) + cb.probabilities.get(k, 0.0)) / 2.0 for k in labels}
    top = max(avg, key=lambda k: avg[k])
    agree = ca.choice == cb.choice
    return Ensembled(label=top if agree else abstain, p_max=avg[top], confidence=choice_confidence(avg),
                     agree=agree, probabilities=avg, raw_top=top)


def _jev(label: str | None, value: float | None, confidence: float, probs: Mapping[str, float],
         r: JevResult) -> Judgement:
    return Judgement(label=label, value=value, confidence=float(confidence), probabilities=dict(probs),
                     source=r.source, latency_ms=r.latency_ms, model=r.model)


def fallback(label: str | None, *, value: float | None = None, fired: bool = False,
             probabilities: Mapping[str, float] | None = None, latency_ms: float = 0.0) -> Judgement:
    """A deterministic judgement (confidence 1.0 when a rule fired, else 0.0)."""
    probs = dict(probabilities) if probabilities is not None else ({label: 1.0} if fired and label else {})
    return Judgement(label=label, value=value, confidence=1.0 if fired else 0.0, probabilities=probs,
                     source="fallback", latency_ms=latency_ms)


def as_fallback(j: Judgement, latency_ms: float) -> Judgement:
    """The same fallback judgement stamped with the time spent before falling back."""
    return dataclasses.replace(j, latency_ms=latency_ms)


# --------------------------------------------------------------------------- B1 fallbacks


def fallback_play_type(local: Mapping[str, Any], rules: FallbackRules = FallbackRules()) -> Judgement:
    """Crisp play-type rules on the possession's local numbers; ``unclear`` when none fires.

    ``passes`` counts completed passes since the possession began or, after an offensive rebound,
    since that rebound; ``terminal_dribbles`` are the shooter's dribbles before the shot.
    """
    end = local.get("end")
    shot = end in ("made_shot", "missed_shot")
    since_start = finite(local.get("seconds_since_start"))
    since_oreb = finite(local.get("seconds_since_oreb"))
    dist, clock = finite(local.get("shot_distance_ft")), finite(local.get("game_clock_at_end_s"))
    passes, dribbles = finite(local.get("passes")), finite(local.get("terminal_dribbles"))
    if shot and dist is not None and clock is not None and dist > rules.heave_min_distance_ft \
            and clock < rules.heave_max_clock_s:
        return fallback("Misc", fired=True)
    if (shot and since_oreb is not None and since_oreb <= rules.putback_window_s
            and (passes is None or passes <= rules.putback_max_passes)
            and (dribbles is None or dribbles <= rules.putback_max_dribbles)):
        return fallback("OffRebound", fired=True)
    if (local.get("start_cause") in rules.transition_causes and since_start is not None
            and since_start <= rules.transition_window_s and local.get("seconds_since_oreb") is None
            and end != "other"):
        return fallback("Transition", fired=True)
    if shot and passes is not None and dribbles is not None and passes <= rules.isolation_max_passes \
            and dribbles >= rules.isolation_min_dribbles:
        return fallback("Isolation", fired=True)
    cns = local.get("catch_and_shoot")
    if shot and cns is not None and bool(cns):  # numpy bools from CV code count too
        return fallback("Spotup", fired=True)
    return fallback(ABSTAIN)


def situation_codes(situation: Mapping[str, Any], local: Mapping[str, Any]) -> dict[str, Any]:
    """Bucket codes for the stakes rule: from ``_local`` when present, else read back from the
    situation phrases (the same vocabulary ``states`` writes)."""
    margin = local.get("margin_code") or code_from_text(situation.get("margin"), MARGIN_TEMPLATES)
    clutch = local.get("clutch")
    if clutch is None:
        clutch = str(situation.get("clutch", "")).strip().lower() == CLUTCH_YES
    run = local.get("run_points")
    return {
        "clutch": bool(clutch),
        "margin": margin,
        "time": local.get("time_code") or code_from_text(situation.get("time_bucket"), TIME_TEMPLATES),
        "run": bool(run) if run is not None else bool(situation.get("run")),
        "lead_change": bool(local.get("lead_change") or situation.get("lead_change")),
        "milestone": bool(local.get("milestone")),
    }


def stakes_level(codes: Mapping[str, Any]) -> tuple[int, bool]:
    """Design §3 site 2 rule table -> (level, rule fired). Proposal: 3 clutch + one possession + last
    minute; 2 clutch; 1 run >= 8-0 / lead change / tie / milestone; 0 otherwise."""
    if codes["clutch"] and codes["margin"] in ("tied", "one_possession") \
            and codes["time"] in ("under_shot_clock", "final_minute"):
        return 3, True
    if codes["clutch"]:
        return 2, True
    if codes["run"] or codes["lead_change"] or codes["margin"] == "tied" or codes["milestone"]:
        return 1, True
    return 0, False


def fallback_stakes(situation: Mapping[str, Any], local: Mapping[str, Any]) -> Judgement:
    """Stakes from the clutch / margin / time / run buckets via the rule table."""
    level, fired = stakes_level(situation_codes(situation, local))
    return fallback(STAKES_LEVELS[level], value=level / (len(STAKES_LEVELS) - 1), fired=fired)


def fallback_highlight(last_play: Mapping[str, Any], local: Mapping[str, Any]) -> Judgement:
    """Highlight from event-engine flags (dunk, block, and-one, buzzer, go-ahead three late)."""
    flags = set(local.get("flags") or ())
    phrase_to_code = {v: k for k, v in FLAG_PHRASES.items()}
    flags |= {phrase_to_code[f] for f in last_play.get("flags", []) if f in phrase_to_code}
    hit = bool(flags & HIGHLIGHT_FLAGS)
    return fallback("yes" if hit else "no", value=1.0 if hit else 0.0, fired=hit)


# --------------------------------------------------------------------------- B1 decoders


def decode_play_type(r: JevResult, th: Thresholds, *, coach: bool = False) -> Judgement | None:
    """Ensembled play type, accepted when both orders agree, confidence >= threshold and
    ``summary_is_clear`` >= its threshold; an abstaining answer is not accepted."""
    resp = r.response
    if resp is None or "play_type" not in resp.choices or "play_type_rev" not in resp.choices:
        return None
    ens = ensembled_choice(resp, "play_type", "play_type_rev")
    clear = resp.nouls.get("summary_is_clear")
    bar = th.play_type_coach_conf if coach else th.play_type_narr_conf
    if not ens.agree or ens.label == ABSTAIN or ens.confidence < bar:
        return None
    if clear is None or clear.noul < th.summary_clear_p:
        return None
    return _jev(ens.label, None, ens.confidence, ens.probabilities, r)


def decode_noul_yes(r: JevResult, name: str, threshold: float) -> Judgement | None:
    """A Noul accepted as "yes" when P(yes) >= threshold; None otherwise."""
    ans = r.response.nouls.get(name) if r.response is not None else None
    if ans is None or ans.noul < threshold:
        return None
    return _jev("yes", ans.noul, noul_confidence(ans.noul), {"yes": ans.noul, "no": 1.0 - ans.noul}, r)


def decode_stakes(r: JevResult, th: Thresholds) -> Judgement | None:
    """``moment_stakes``: level = round(score) when the Score confidence >= threshold; value = score/3."""
    ans = r.response.scores.get("moment_stakes") if r.response is not None else None
    if ans is None or ans.confidence < th.stakes_conf:
        return None
    top = len(STAKES_LEVELS) - 1
    level = min(top, max(0, math.floor(ans.score + 0.5)))  # half-up; round() is half-to-even (2.5 -> 2)
    probs = {STAKES_LEVELS[int(k)]: v for k, v in ans.probabilities.items() if 0 <= int(k) <= top}
    return _jev(STAKES_LEVELS[level], min(1.0, max(0.0, ans.score / top)), ans.confidence, probs, r)


# --------------------------------------------------------------------------- B2 / B3 / B4 / B5 decoders


def decode_line_kind(r: JevResult, offered: Sequence[str], th: Thresholds) -> Judgement | None:
    """``next_line_kind`` accepted when it is an offered kind and its confidence >= threshold."""
    ans = r.response.choices.get("next_line_kind") if r.response is not None else None
    if ans is None or ans.choice not in offered or ans.choice not in LINE_KINDS or ans.confidence < th.line_kind_conf:
        return None
    return _jev(ans.choice, None, ans.confidence, ans.probabilities, r)


@dataclass(frozen=True)
class CardDecision:
    """Decoded B3 answer."""

    top: str | None  # card key, "none", or None when below threshold
    relevant: dict[str, float]  # card key -> P(relevant)


def decode_cards(r: JevResult, keys: Sequence[str], th: Thresholds) -> CardDecision | None:
    """``top_card`` accepted when confidence >= threshold and (for a card) its ``relevant`` >= threshold."""
    resp = r.response
    if resp is None:
        return None
    relevant = {k: resp.nouls[f"relevant__{k}"].noul for k in keys if f"relevant__{k}" in resp.nouls}
    top = resp.choices.get("top_card")
    if top is None or top.confidence < th.top_card_conf:
        return CardDecision(None, relevant)
    if top.choice == "none":
        return CardDecision("none", relevant)
    if top.choice in keys and relevant.get(top.choice, 0.0) >= th.card_relevant_p:
        return CardDecision(top.choice, relevant)
    return CardDecision(None, relevant)


def guard_fallback(sentences: Sequence[str]) -> list[LineVerdict]:
    """D4: no Jev verdict => the LLM line is not spoken."""
    return [LineVerdict(s, None, None, None, passed=False, source="fallback") for s in sentences]


def decode_guard(r: JevResult, sentences: Sequence[str], th: Thresholds) -> list[LineVerdict]:
    """Per-sentence verdicts; a sentence missing any of its three answers is not passed."""
    out: list[LineVerdict] = []
    nouls = r.response.nouls if r.response is not None else {}
    for i, s in enumerate(sentences, 1):
        sup, unl, spe = (nouls.get(f"{q}_s{i}") for q in ("supported", "unlisted", "speculation"))
        if sup is None or unl is None or spe is None:
            out.append(LineVerdict(s, None, None, None, passed=False, source="fallback"))
            continue
        passed = (sup.noul >= th.guard_supported_p and unl.noul <= th.guard_unlisted_max_p
                  and spe.noul <= th.guard_speculation_max_p)
        out.append(LineVerdict(s, sup.noul, unl.noul, spe.noul, passed=passed, source=r.source))
    return out


def decide_ambiguity(r: JevResult, kind: str, rule_lean: str, forbidden: set[str] | frozenset[str],
                     th: Thresholds, pinned: str = JEV_MODEL_TEXT) -> Ensembled | None:
    """Tier C commit rule (design §3 site 5): the ensembled label is committable, both orders agree,
    averaged p_max >= ``event_pmax``, it equals the rule engine's lean and no hard constraint forbids
    it. ``None`` means pending (deterministic reconciliation decides)."""
    if kind not in B5_KINDS or not usable(r, pinned):
        return None
    _root, _qs, fwd, rev, committable = B5_KINDS[kind]
    if fwd not in r.response.choices or rev not in r.response.choices:
        return None
    ens = ensembled_choice(r.response, fwd, rev)
    if (ens.label in committable and ens.agree and ens.p_max >= th.event_pmax and ens.label == rule_lean
            and ens.label not in forbidden):
        return ens
    return None


def decide_shot_outcome(r: JevResult, rule_lean: Literal["make", "miss"], forbidden: set[str], th: Thresholds,
                        pinned: str = JEV_MODEL_TEXT) -> Literal["provisional_make", "provisional_miss", "pending"]:
    ens = decide_ambiguity(r, "shot_outcome", rule_lean, forbidden, th, pinned)
    if ens is None:
        return "pending"
    return "provisional_make" if ens.label == "make" else "provisional_miss"


def decide_release_intent(r: JevResult, rule_lean: Literal["shot_attempt", "pass"], forbidden: set[str],
                          th: Thresholds, pinned: str = JEV_MODEL_TEXT
                          ) -> Literal["provisional_shot_attempt", "provisional_pass", "pending"]:
    ens = decide_ambiguity(r, "release_intent", rule_lean, forbidden, th, pinned)
    if ens is None:
        return "pending"
    return "provisional_shot_attempt" if ens.label == "shot_attempt" else "provisional_pass"


def ambiguity_judgement(r: JevResult, ens: Ensembled) -> Judgement:
    """A committed B5 label (Choice: ``value`` stays None; ``probabilities`` holds the averaged set)."""
    return _jev(ens.label, None, ens.confidence, ens.probabilities, r)
