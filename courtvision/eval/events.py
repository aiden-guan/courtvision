"""Event-level evaluation: one-to-one temporal matching of predicted vs truth events.

Semantics follow ARCHITECTURE "Shared event semantics": consumers keep only the latest
revision per ``event_id`` and drop RETRACTED events; evaluation tolerance is +/-1.0 s.

Per kind, predicted and truth events are paired one-to-one by an optimal assignment
(``scipy.optimize.linear_sum_assignment``) with lexicographic priorities:

1. as many pairs with ``|dt| <= tol`` as possible (detection precision/recall),
2. among those, as many pairs whose actor agrees as possible (only when
   ``match_player=True``),
3. among those, the smallest total ``|dt|``.

The actor is ``player_id``; for team rebounds / team turnovers (``player_id is None``)
it is the team, since those are first-class team entities (basketball_rules_stats.md
§0.5). A pair whose actor differs still counts for detection but is wrong for player
accuracy. Attributed F1 (class + time + actor; training_data.md §4.1 "who-did-it") uses
its own one-to-one matching restricted to same-actor pairs, so a detection pairing can
never hide a correct attribution.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from courtvision.types import Event, EventKind, EventStatus

logger = logging.getLogger(__name__)

EventLike = Event | Mapping[str, Any]

# Headline kinds (spec). POSSESSION/FOUL/SUBSTITUTION/... are still reported per kind.
CORE_KINDS: tuple[str, ...] = (
    "shot_attempt",
    "shot_made",
    "shot_missed",
    "rebound",
    "assist",
    "steal",
    "turnover",
    "block",
    "pass",
)

_SHOT_RESULTS = (EventKind.SHOT_MADE, EventKind.SHOT_MISSED)

# Slack on the inclusive |dt| <= tol bound. Timestamps are frame_idx / fps, so an exact
# 1.0 s offset can come out as 1.0000000000000002 s (e.g. 33/30 - 3/30).
_TOL_EPS_S = 1e-9


@dataclass
class EventEvalConfig:
    """Tunables for :func:`match_events`."""

    # Kinds averaged into the headline macro-F1 (spec: core basketball events).
    core_kinds: tuple[str, ...] = CORE_KINDS
    # High-frequency kinds reported apart from ``per_kind`` and never in the headline
    # (spec: DRIBBLE is "not narrated").
    separate_kinds: tuple[str, ...] = ("dribble",)
    # Meta-events no truth source emits (reconciliation output); skipped when kinds=None.
    ignore_kinds: tuple[str, ...] = ("score_correction",)
    # Per-kind tolerance overrides (s). Dribble: bounces come every ~0.5-0.6 s, so the
    # global 1.0 s window would let spurious bounces match neighbours (unverified; tuned).
    kind_tol_s: dict[str, float] = field(default_factory=lambda: {"dribble": 0.25})
    # Kinds a human operator enters (ARCHITECTURE: SUBSTITUTION is operator / sim truth): they are
    # matched within the kind's tolerance plus the modelled entry lag (s), since an entry cannot come
    # before the operator sees the event. The pipeline sets the lag from ``operator.entry_delay_s``.
    operator_lag_s: dict[str, float] = field(default_factory=dict)
    # Stream boundaries: when a run is cut mid-game (``match_events(window=...)``), unmatched predictions
    # and truths within this many seconds of a cut edge are not scored (their evidence lies outside the
    # stream: a shot still in the air at the end, an event too close to the end to confirm). None = the
    # kind's matching tolerance.
    boundary_guard_s: float | None = None
    # Event.attrs keys whose agreement is reported per kind (types.py well-known keys).
    attr_keys: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "rebound": ("offensive",),
            "pass": ("completed",),
            "turnover": ("type",),
            "shot_attempt": ("shot_type",),
            "possession": ("cause",),
        }
    )


# --------------------------------------------------------------------------- report types


@dataclass
class KindReport:
    """Detection and attribution metrics for one event kind."""

    kind: str
    tol_s: float
    n_pred: int
    n_truth: int
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None
    player_accuracy: float | None  # actor agreement over matched pairs; None when match_player=False
    other_player_accuracy: float | None  # pairs where either side names other_player_id
    team_accuracy: float | None  # pairs whose truth team is set
    attributed_precision: float | None  # TP from a same-actor one-to-one matching
    attributed_recall: float | None
    attributed_f1: float | None
    mean_abs_dt: float | None
    mean_dt: float | None  # signed pred - truth (positive = engine late)
    attr_accuracy: dict[str, float | None] = field(default_factory=dict)
    attr_n: dict[str, int] = field(default_factory=dict)  # pairs judged per attr key (truth carries the key)
    fp_ids: list[str] = field(default_factory=list)
    fn_ids: list[str] = field(default_factory=list)
    wrong_player: list[tuple[str, str]] = field(default_factory=list)  # (pred_id, truth_id)

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["wrong_player"] = [list(p) for p in self.wrong_player]
        d["attr_accuracy"] = dict(self.attr_accuracy)
        d["attr_n"] = dict(self.attr_n)
        d["fp_ids"] = list(self.fp_ids)
        d["fn_ids"] = list(self.fn_ids)
        return d


@dataclass
class ShotReport:
    """Shot-level join of attempts, results and assists through ``shot_id``."""

    n_pred_shots: int = 0
    n_truth_shots: int = 0
    n_paired: int = 0
    n_paired_by_result: int = 0  # pairs found from result timing because an attempt was unmatched/missing
    fg_outcome_accuracy: float | None = None  # make/miss on paired field goals (unresolved = wrong)
    fg_outcome_n: int = 0
    ft_outcome_accuracy: float | None = None
    ft_outcome_n: int = 0
    outcome_confusion: dict[str, dict[str, int]] = field(default_factory=dict)  # FG only: truth -> pred -> n
    made_fg_tp: int = 0
    n_pred_made_fg: int = 0
    n_truth_made_fg: int = 0
    made_fg_precision: float | None = None
    made_fg_recall: float | None = None
    made_fg_f1: float | None = None
    value_accuracy: float | None = None  # 2 vs 3 on paired field goals
    value_n: int = 0
    value_confusion: dict[str, dict[str, int]] = field(default_factory=dict)  # truth value -> pred value -> n
    assist_accuracy: float | None = None  # on pairs both sides call made FG; "unassisted" is a label
    assist_n: int = 0
    shooter_accuracy: float | None = None
    shooter_n: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["outcome_confusion"] = {k: dict(v) for k, v in self.outcome_confusion.items()}
        d["value_confusion"] = {k: dict(v) for k, v in self.value_confusion.items()}
        return d


@dataclass
class EventReport:
    """Result of :func:`match_events`. ``to_dict()`` is JSON-serialisable."""

    tol_s: float
    match_player: bool
    per_kind: dict[str, KindReport]
    separate: dict[str, KindReport]  # e.g. dribble; never in the headline
    shots: ShotReport
    headline_macro_f1: float | None  # mean F1 over core kinds that occur in pred or truth
    headline_attributed_f1: float | None
    headline_kinds: list[str]
    input_counts: dict[str, dict[str, int]]
    # Companions to the (unchanged, strict) headline: the macro over core kinds with truth support,
    # the micro-F1 over core kinds (2 sum TP / (sum pred + sum truth)), and the false positives of core
    # kinds with no truth (each such kind enters the headline as F1 0).
    supported_macro_f1: float | None = None
    supported_kinds: list[str] = field(default_factory=list)
    micro_f1: float | None = None
    unsupported_fp: dict[str, int] = field(default_factory=dict)
    boundary_excluded: dict[str, dict[str, int]] = field(default_factory=dict)  # kind -> {"pred", "truth"}
    # The protocol this report was scored with (so ``courtvision eval`` can re-score a run the same way).
    window: tuple[float | None, float | None] | None = None
    operator_lag_s: dict[str, float] = field(default_factory=dict)

    @property
    def dribble(self) -> KindReport | None:
        return self.separate.get(EventKind.DRIBBLE.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tol_s": self.tol_s,
            "match_player": self.match_player,
            "headline": {
                "macro_f1": self.headline_macro_f1,
                "attributed_macro_f1": self.headline_attributed_f1,
                "kinds": list(self.headline_kinds),
                "supported_macro_f1": self.supported_macro_f1,
                "supported_kinds": list(self.supported_kinds),
                "micro_f1": self.micro_f1,
                "unsupported_fp": dict(self.unsupported_fp),
            },
            "boundary_excluded": {k: dict(v) for k, v in self.boundary_excluded.items()},
            "window": list(self.window) if self.window is not None else None,
            "operator_lag_s": dict(self.operator_lag_s),
            "per_kind": {k: r.to_dict() for k, r in self.per_kind.items()},
            "separate": {k: r.to_dict() for k, r in self.separate.items()},
            "shots": self.shots.to_dict(),
            "input_counts": {k: dict(v) for k, v in self.input_counts.items()},
        }


# --------------------------------------------------------------------------- revisions


def _as_event(e: EventLike) -> Event:
    if isinstance(e, Event):
        return e
    d = dict(e)
    if d.get("attrs") is None:  # JSON from manual/feed sources may carry "attrs": null
        d["attrs"] = {}
    return Event.from_dict(d)


def _dedupe(events: Iterable[EventLike]) -> tuple[list[Event], dict[str, int]]:
    latest: dict[str, Event] = {}
    n_seen: dict[str, int] = {}
    n_input = 0
    for raw in events:
        e = _as_event(raw)
        n_input += 1
        n_seen[e.event_id] = n_seen.get(e.event_id, 0) + 1
        cur = latest.get(e.event_id)
        if cur is None or e.revision >= cur.revision:  # ties: later in the input wins
            latest[e.event_id] = e
    kept = [e for e in latest.values() if e.status is not EventStatus.RETRACTED]
    kept.sort(key=lambda e: (e.t, e.event_id))
    counts = {
        "input": n_input,
        "unique_ids": len(latest),
        "retracted": len(latest) - len(kept),
        "revised_ids": sum(1 for n in n_seen.values() if n > 1),
        "kept": len(kept),
    }
    return kept, counts


def latest_revisions(events: Iterable[EventLike]) -> list[Event]:
    """Latest revision per ``event_id`` (ties: later input wins), RETRACTED dropped, sorted by t."""
    return _dedupe(events)[0]


# --------------------------------------------------------------------------- assignment


def gated_assignment(cost: np.ndarray, feasible: np.ndarray) -> list[tuple[int, int]]:
    """One-to-one assignment using only ``feasible`` pairs: max cardinality, then min cost.

    ``cost`` must be non-negative. Infeasible entries get a cost larger than any complete
    feasible assignment, so the solver never trades a feasible pair for a cheaper one.
    """
    cost = np.asarray(cost, dtype=float)
    feasible = np.asarray(feasible, dtype=bool)
    if cost.size == 0 or not feasible.any():
        return []
    rows = np.flatnonzero(feasible.any(axis=1))
    cols = np.flatnonzero(feasible.any(axis=0))
    sub_cost = cost[np.ix_(rows, cols)]
    sub_ok = feasible[np.ix_(rows, cols)]
    k = min(len(rows), len(cols))
    big = (k + 1) * (float(sub_cost[sub_ok].max()) + 1.0)
    r, c = linear_sum_assignment(np.where(sub_ok, sub_cost, big))
    return [(int(rows[i]), int(cols[j])) for i, j in zip(r, c, strict=True) if sub_ok[i, j]]


def _time_blocks(pred_t: np.ndarray, truth_t: np.ndarray, tol: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split into independent blocks: no pair within ``tol`` can straddle a gap > tol."""
    if len(pred_t) == 0 or len(truth_t) == 0:
        return []
    times = np.concatenate([pred_t, truth_t])
    is_pred = np.concatenate([np.ones(len(pred_t), bool), np.zeros(len(truth_t), bool)])
    idx = np.concatenate([np.arange(len(pred_t)), np.arange(len(truth_t))])
    order = np.argsort(times, kind="stable")
    block = np.concatenate([[0], np.cumsum(np.diff(times[order]) > tol + _TOL_EPS_S)])
    out = []
    for b in np.unique(block):
        members = order[block == b]
        p = idx[members[is_pred[members]]]
        t = idx[members[~is_pred[members]]]
        if len(p) and len(t):
            out.append((p, t))
    return out


def _key_codes(
    pred_keys: Sequence[Hashable] | None, truth_keys: Sequence[Hashable] | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Hashable keys -> integer codes shared by both sides (None when keys are not used)."""
    if pred_keys is None or truth_keys is None:
        return None, None
    codes: dict[Hashable, int] = {}
    pk = np.array([codes.setdefault(k, len(codes)) for k in pred_keys], dtype=np.int64)
    tk = np.array([codes.setdefault(k, len(codes)) for k in truth_keys], dtype=np.int64)
    return pk, tk


def match_times(
    pred_t: Sequence[float],
    truth_t: Sequence[float],
    tol_s: float,
    pred_keys: Sequence[Hashable] | None = None,
    truth_keys: Sequence[Hashable] | None = None,
) -> list[tuple[int, int]]:
    """Pair indices one-to-one with ``|pred_t - truth_t| <= tol_s`` (inclusive, with 1e-9 s
    float slack so frame-quantised timestamps exactly ``tol_s`` apart still pair).

    Priorities: most pairs, then (if keys are given) most pairs with equal keys, then the
    smallest total ``|dt|``. Returns ``(pred_index, truth_index)`` sorted by truth index.
    """
    pt = np.asarray(pred_t, dtype=float)
    tt = np.asarray(truth_t, dtype=float)
    pk, tk = _key_codes(pred_keys, truth_keys)
    pairs: list[tuple[int, int]] = []
    for p_idx, t_idx in _time_blocks(pt, tt, tol_s):
        dt = np.abs(pt[p_idx][:, None] - tt[t_idx][None, :])
        feasible = dt <= tol_s + _TOL_EPS_S
        if not feasible.any():
            continue
        cost = dt
        if pk is not None and tk is not None:
            mismatch = pk[p_idx][:, None] != tk[t_idx][None, :]
            # Any one key agreement outweighs the total |dt| of a whole block. Built from
            # the largest feasible |dt| (not tol_s) so it stays finite for tol_s=inf.
            weight = (min(len(p_idx), len(t_idx)) + 1) * float(dt[feasible].max()) + 1.0
            cost = dt + weight * mismatch
        for i, j in gated_assignment(cost, feasible):
            pairs.append((int(p_idx[i]), int(t_idx[j])))
    pairs.sort(key=lambda ij: ij[1])
    return pairs


def _actor(e: Event) -> Hashable:
    """Who the event is credited to: the player, or the team when ``player_id`` is None."""
    if e.player_id is not None:
        return e.player_id
    return ("team", e.team.value if e.team is not None else None)


def _match_events(
    pred: list[Event], truth: list[Event], tol: float, use_actor: bool
) -> list[tuple[Event, Event]]:
    keys_p = [_actor(e) for e in pred] if use_actor else None
    keys_t = [_actor(e) for e in truth] if use_actor else None
    ij = match_times([e.t for e in pred], [e.t for e in truth], tol, keys_p, keys_t)
    return [(pred[i], truth[j]) for i, j in ij]


def _attributed_tp(pred: list[Event], truth: list[Event], tol: float) -> int:
    """Max one-to-one pairs with ``|dt| <= tol`` AND the same actor (class + time + actor)."""
    by_p: dict[Hashable, list[float]] = {}
    by_t: dict[Hashable, list[float]] = {}
    for e in pred:
        by_p.setdefault(_actor(e), []).append(e.t)
    for e in truth:
        by_t.setdefault(_actor(e), []).append(e.t)
    return sum(len(match_times(by_p[a], by_t[a], tol)) for a in by_p.keys() & by_t.keys())


# --------------------------------------------------------------------------- metrics


def _ratio(num: int, den: int) -> float | None:
    return num / den if den > 0 else None


def _f1(tp: int, n_pred: int, n_truth: int) -> float | None:
    if n_pred + n_truth == 0:
        return None
    return 2.0 * tp / (n_pred + n_truth)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _kind_report(
    kind: str,
    pred: list[Event],
    truth: list[Event],
    pairs: list[tuple[Event, Event]],
    tol: float,
    match_player: bool,
    attr_keys: tuple[str, ...],
) -> KindReport:
    tp = len(pairs)
    matched_p = {p.event_id for p, _ in pairs}
    matched_t = {t.event_id for _, t in pairs}
    right_player = [_actor(p) == _actor(t) for p, t in pairs]
    other = [(p, t) for p, t in pairs if p.other_player_id is not None or t.other_player_id is not None]
    teamed = [(p, t) for p, t in pairs if t.team is not None]
    dts = [p.t - t.t for p, t in pairs]
    attr_acc: dict[str, float | None] = {}
    attr_n: dict[str, int] = {}
    for key in attr_keys:
        judged = [p.attrs.get(key) == t.attrs[key] for p, t in pairs if key in t.attrs]
        attr_acc[key] = _ratio(sum(judged), len(judged))
        attr_n[key] = len(judged)
    atp = _attributed_tp(pred, truth, tol) if match_player else 0
    return KindReport(
        kind=kind,
        tol_s=tol,
        n_pred=len(pred),
        n_truth=len(truth),
        tp=tp,
        fp=len(pred) - tp,
        fn=len(truth) - tp,
        precision=_ratio(tp, len(pred)),
        recall=_ratio(tp, len(truth)),
        f1=_f1(tp, len(pred), len(truth)),
        player_accuracy=_ratio(sum(right_player), tp) if match_player else None,
        other_player_accuracy=(
            _ratio(sum(p.other_player_id == t.other_player_id for p, t in other), len(other))
            if match_player
            else None
        ),
        team_accuracy=_ratio(sum(p.team == t.team for p, t in teamed), len(teamed)),
        attributed_precision=_ratio(atp, len(pred)) if match_player else None,
        attributed_recall=_ratio(atp, len(truth)) if match_player else None,
        attributed_f1=_f1(atp, len(pred), len(truth)) if match_player else None,
        mean_abs_dt=_mean([abs(d) for d in dts]),
        mean_dt=_mean(dts),
        attr_accuracy=attr_acc,
        attr_n=attr_n,
        fp_ids=[e.event_id for e in pred if e.event_id not in matched_p],
        fn_ids=[e.event_id for e in truth if e.event_id not in matched_t],
        wrong_player=(
            [(p.event_id, t.event_id) for (p, t), ok in zip(pairs, right_player, strict=True) if not ok]
            if match_player
            else []
        ),
    )


# --------------------------------------------------------------------------- shots


@dataclass
class _Shot:
    key: str
    attempt: Event | None = None
    result: Event | None = None
    assist: Event | None = None

    @property
    def outcome(self) -> str | None:
        if self.result is None:
            return None
        return "made" if self.result.kind is EventKind.SHOT_MADE else "missed"

    @property
    def shooter(self) -> str | None:
        src = self.attempt or self.result
        return src.player_id if src is not None else None

    @property
    def value(self) -> int | None:
        if self.attempt is not None and self.attempt.value > 0:
            return self.attempt.value
        if self.result is not None and self.result.kind is EventKind.SHOT_MADE and self.result.value > 0:
            return self.result.value
        return None

    @property
    def is_ft(self) -> bool:
        for e in (self.attempt, self.result):
            if e is not None and e.attrs.get("shot_type") == "free_throw":
                return True
        return self.value == 1

    @property
    def assister(self) -> str | None:
        return self.assist.player_id if self.assist is not None else None


def _shot_key(e: Event) -> str:
    return e.shot_id if e.shot_id is not None else f"~{e.event_id}"


def _build_shots(events: list[Event]) -> dict[str, _Shot]:
    shots: dict[str, _Shot] = {}
    for e in events:  # already sorted by t
        if e.kind is EventKind.ASSIST:
            if e.shot_id is None:
                logger.debug("assist %s has no shot_id; ignored for shot analysis", e.event_id)
                continue
            shots.setdefault(e.shot_id, _Shot(e.shot_id)).assist = e
        elif e.kind is EventKind.SHOT_ATTEMPT:
            s = shots.setdefault(_shot_key(e), _Shot(_shot_key(e)))
            if s.attempt is None:
                s.attempt = e
            else:
                logger.warning("shot %s has two attempts; keeping %s over %s", s.key, s.attempt.event_id, e.event_id)
        elif e.kind in _SHOT_RESULTS:
            s = shots.setdefault(_shot_key(e), _Shot(_shot_key(e)))
            if s.result is not None and s.result.kind is not e.kind:
                logger.warning("shot %s has conflicting results; keeping the later %s", s.key, e.event_id)
            s.result = e  # latest result by time wins
    # An assist whose shot never appears is not a shot.
    return {k: s for k, s in shots.items() if s.attempt is not None or s.result is not None}


def _attempt_t(s: _Shot) -> float:
    assert s.attempt is not None
    return s.attempt.t


def _result_t(s: _Shot) -> float:
    assert s.result is not None
    return s.result.t


def _pair_shots(
    pred: dict[str, _Shot], truth: dict[str, _Shot], tol: float, use_actor: bool
) -> tuple[list[tuple[_Shot, _Shot]], int]:
    p_att = [s for s in pred.values() if s.attempt is not None]
    t_att = [s for s in truth.values() if s.attempt is not None]
    ij = match_times(
        [_attempt_t(s) for s in p_att],
        [_attempt_t(s) for s in t_att],
        tol,
        [s.shooter for s in p_att] if use_actor else None,
        [s.shooter for s in t_att] if use_actor else None,
    )
    pairs = [(p_att[i], t_att[j]) for i, j in ij]
    used_p = {p.key for p, _ in pairs}
    used_t = {t.key for _, t in pairs}
    p_rest = [s for s in pred.values() if s.key not in used_p and s.result is not None]
    t_rest = [s for s in truth.values() if s.key not in used_t and s.result is not None]
    ij = match_times(
        [_result_t(s) for s in p_rest],
        [_result_t(s) for s in t_rest],
        tol,
        [s.shooter for s in p_rest] if use_actor else None,
        [s.shooter for s in t_rest] if use_actor else None,
    )
    by_result = [(p_rest[i], t_rest[j]) for i, j in ij]
    return pairs + by_result, len(by_result)


def _bump(table: dict[str, dict[str, int]], row: str, col: str) -> None:
    table.setdefault(row, {})
    table[row][col] = table[row].get(col, 0) + 1


def _shot_report(pred_events: list[Event], truth_events: list[Event], tol: float, use_actor: bool) -> ShotReport:
    pred = _build_shots(pred_events)
    truth = _build_shots(truth_events)
    pairs, n_by_result = _pair_shots(pred, truth, tol, use_actor)
    rep = ShotReport(
        n_pred_shots=len(pred), n_truth_shots=len(truth), n_paired=len(pairs), n_paired_by_result=n_by_result
    )

    fg_ok: list[bool] = []
    ft_ok: list[bool] = []
    value_ok: list[bool] = []
    assist_ok: list[bool] = []
    shooter_ok: list[bool] = []
    for p, t in pairs:
        shooter_ok.append(p.shooter == t.shooter)
        if t.outcome is not None:
            if not t.is_ft:
                _bump(rep.outcome_confusion, t.outcome, p.outcome or "unresolved")
            (ft_ok if t.is_ft else fg_ok).append(p.outcome == t.outcome)
        if not t.is_ft and t.value in (2, 3) and p.value is not None:
            _bump(rep.value_confusion, str(t.value), str(p.value))
            value_ok.append(p.value == t.value)
        if t.outcome == "made" and p.outcome == "made" and not t.is_ft and not p.is_ft:
            rep.made_fg_tp += 1
            assist_ok.append(p.assister == t.assister)

    rep.n_pred_made_fg = sum(1 for s in pred.values() if s.outcome == "made" and not s.is_ft)
    rep.n_truth_made_fg = sum(1 for s in truth.values() if s.outcome == "made" and not s.is_ft)
    rep.made_fg_precision = _ratio(rep.made_fg_tp, rep.n_pred_made_fg)
    rep.made_fg_recall = _ratio(rep.made_fg_tp, rep.n_truth_made_fg)
    rep.made_fg_f1 = _f1(rep.made_fg_tp, rep.n_pred_made_fg, rep.n_truth_made_fg)
    rep.fg_outcome_accuracy, rep.fg_outcome_n = _ratio(sum(fg_ok), len(fg_ok)), len(fg_ok)
    rep.ft_outcome_accuracy, rep.ft_outcome_n = _ratio(sum(ft_ok), len(ft_ok)), len(ft_ok)
    rep.value_accuracy, rep.value_n = _ratio(sum(value_ok), len(value_ok)), len(value_ok)
    rep.assist_accuracy, rep.assist_n = _ratio(sum(assist_ok), len(assist_ok)), len(assist_ok)
    rep.shooter_accuracy, rep.shooter_n = _ratio(sum(shooter_ok), len(shooter_ok)), len(shooter_ok)
    return rep


# --------------------------------------------------------------------------- entry point


def _kind_values(kinds: Iterable[EventKind | str] | EventKind | str | None) -> list[str] | None:
    if kinds is None:
        return None
    if isinstance(kinds, str):  # one kind (EventKind is a str subclass), not its characters
        kinds = [kinds]
    return [EventKind(k).value for k in kinds]


def _group(events: list[Event]) -> dict[str, list[Event]]:
    out: dict[str, list[Event]] = {}
    for e in events:
        out.setdefault(e.kind.value, []).append(e)
    return out


def _outside_guard(e: Event, window: tuple[float | None, float | None] | None, guard: float) -> bool:
    if window is None:
        return True
    lo, hi = window
    return not ((hi is not None and e.t > hi - guard) or (lo is not None and e.t < lo + guard))


def match_events(
    pred: Iterable[EventLike],
    truth: Iterable[EventLike],
    kinds: Iterable[EventKind | str] | EventKind | str | None = None,
    tol_s: float = 1.0,
    match_player: bool = True,
    config: EventEvalConfig | None = None,
    window: tuple[float | None, float | None] | None = None,
) -> EventReport:
    """Compare predicted events with truth events (``Event`` objects or ``to_dict()`` dicts).

    ``kinds=None`` evaluates every kind present on either side (except
    ``config.ignore_kinds``). With ``match_player=False`` actors are ignored when pairing
    and player metrics are ``None``. The shot-level analysis always runs on the full
    (deduplicated) streams.

    ``window=(start, end)`` gives the edges where the stream was cut mid-game (None = not cut): an
    event within ``config.boundary_guard_s`` (default: the kind's tolerance) of a cut edge that finds no
    partner is left out of that kind's counts instead of being scored FP / FN (see
    ``boundary_excluded``); matched pairs always count.
    """
    cfg = config or EventEvalConfig()
    pred_ev, pred_counts = _dedupe(pred)
    truth_ev, truth_counts = _dedupe(truth)
    by_p, by_t = _group(pred_ev), _group(truth_ev)

    selected = _kind_values(kinds)
    if selected is None:
        present = set(by_p) | set(by_t)
        order = [k.value for k in EventKind]
        selected = [k for k in order if k in present and k not in cfg.ignore_kinds]

    per_kind: dict[str, KindReport] = {}
    separate: dict[str, KindReport] = {}
    excluded: dict[str, dict[str, int]] = {}
    for kind in selected:
        tol = cfg.kind_tol_s.get(kind, tol_s) + cfg.operator_lag_s.get(kind, 0.0)
        p, t = by_p.get(kind, []), by_t.get(kind, [])
        pairs = _match_events(p, t, tol, match_player)
        if window is not None:
            guard = cfg.boundary_guard_s if cfg.boundary_guard_s is not None else tol
            mp, mt = {a.event_id for a, _ in pairs}, {b.event_id for _, b in pairs}
            p2 = [e for e in p if e.event_id in mp or _outside_guard(e, window, guard)]
            t2 = [e for e in t if e.event_id in mt or _outside_guard(e, window, guard)]
            if len(p2) < len(p) or len(t2) < len(t):
                excluded[kind] = {"pred": len(p) - len(p2), "truth": len(t) - len(t2)}
            p, t = p2, t2
        rep = _kind_report(kind, p, t, pairs, tol, match_player, cfg.attr_keys.get(kind, ()))
        (separate if kind in cfg.separate_kinds else per_kind)[kind] = rep

    head = [k for k in cfg.core_kinds if k in per_kind and per_kind[k].f1 is not None]
    macro = _mean([per_kind[k].f1 or 0.0 for k in head])
    attributed = _mean([per_kind[k].attributed_f1 or 0.0 for k in head]) if match_player else None
    supported = [k for k in head if per_kind[k].n_truth > 0]
    supported_macro = _mean([per_kind[k].f1 or 0.0 for k in supported])
    micro = _f1(sum(per_kind[k].tp for k in head), sum(per_kind[k].n_pred for k in head),
                sum(per_kind[k].n_truth for k in head))
    unsupported_fp = {k: per_kind[k].fp for k in head if per_kind[k].n_truth == 0}

    shot_tol = cfg.kind_tol_s.get(EventKind.SHOT_ATTEMPT.value, tol_s)
    shots = _shot_report(pred_ev, truth_ev, shot_tol, match_player)
    report = EventReport(
        tol_s=tol_s,
        match_player=match_player,
        per_kind=per_kind,
        separate=separate,
        shots=shots,
        headline_macro_f1=macro,
        headline_attributed_f1=attributed,
        headline_kinds=head,
        input_counts={"pred": pred_counts, "truth": truth_counts},
        supported_macro_f1=supported_macro,
        supported_kinds=supported,
        micro_f1=micro,
        unsupported_fp=unsupported_fp,
        boundary_excluded=excluded,
        window=tuple(window) if window is not None else None,
        operator_lag_s=dict(cfg.operator_lag_s),
    )
    logger.debug("match_events: macro-F1 %s over %s", macro, head)
    return report
