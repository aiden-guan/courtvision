"""Shots from the video alone: which of the engine's shot results, and of the ball's visits to a rim, are shots, free
throws or field goals, made or missed.

Design: docs/research/possession_decision_model.md §3.3, §7 and §8. The event engine's own made / missed label leans
on the scorebug (a make it sees waits for a score change and becomes a miss without one; a score change turns a miss
it saw into a make), it reports some free throws as field goals, and it never reports about one field goal in six
(most put-backs). Here only what the camera saw counts.

The candidates (``candidates``) are the shot results the engine reports from the camera (field goals and free
throws), and the visits of the linked ball to a rim (``rim_visits``: within a rim width over a few frames) that no
engine result is near. Their evidence (``candidate_features``):

* the engine's rim cues, for its results (``Event.attrs["cues"]``: the ball crossed the rim plane inside the ring,
  touched the rim, dropped at net speed, the crossing margin, frames hidden by the net);
* the ball's path at the rim (``RimVisit``): how close it came, whether it came down from above and left below;
* what the possession timeline (``timeline.Timeline``) shows happens next: did the ball change hands, how soon, and
  how long the other team stayed in its back court (an inbound after a make takes seconds);
* a free throw's setting: the camera off the court in the seconds before (a stoppage: replays and close-ups) and
  almost no dribbling or passing around it;
* optionally, Jev's read of the rim crops (``rim_jev.RimJudge``: the in-net probability per crop and made / missed).

Three small logistic models (``ShotModels``), fitted on aligned play-by-play (``fit_shot_models``), decide a real shot
or not, a free throw or a field goal (for the real ones), and made or missed. ``decide`` rebuilds the shots: the
engine's results relabelled or dropped (with their attempts, blocks and assists), shots added at the rim visits it
missed, an assist kept for a made field goal only, and a made field goal without one given the last pass of a
teammate (the timeline's team) just before the release. Results the engine inferred from the scorebug go too.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from courtvision.events.calibration import _fit_logistic
from courtvision.eval.events import latest_revisions
from courtvision.types import Event, EventKind, EventStatus

from .analyze import rim_boxes
from .timeline import BACK, Timeline, from_scorebug

__all__ = ["CUE_FEATURES", "FT_FEATURES", "JEV_FEATURES", "RIM_FEATURES", "TIMELINE_FEATURES", "VISIT_FEATURES",
           "ZONE_FEATURES", "Candidate", "zone_features",
           "DecideConfig", "Logistic", "RimVisit", "ShotModels", "candidate_features", "candidate_rows", "candidates",
           "decide", "fit_shot_models", "label_candidates", "rim_visits"]

K = EventKind
RESULTS = (K.SHOT_MADE, K.SHOT_MISSED)
CUE_FEATURES = ("crossed_inside", "rim_contact", "drop_in_band", "drop_is_freefall", "margin", "margin_missing",
                "occluded")
VISIT_FEATURES = ("engine", "visit", "visit_dmin", "visit_from_above", "visit_below", "visit_frames")
TIMELINE_FEATURES = ("changed", "dt_change", "back_s")
FT_FEATURES = ("cam_before", "activity", "engine_ft")
JEV_FEATURES = ("jev_in_net", "jev_made", "jev_missing")
RIM_FEATURES = ("rim_made", "rim_shot", "rim_missing")
ZONE_FEATURES = ("zone_net", "zone_net_n", "zone_top_then_net", "zone_net_then_under", "zone_post_above",
                 "zone_post_side", "zone_top_then_out", "zone_above", "zone_ring", "zone_valid")


def _is_ft(e: Event) -> bool:
    return e.value == 1 or e.attrs.get("shot_type") == "free_throw"


# --------------------------------------------------------------------------- candidates


@dataclass(frozen=True)
class RimVisit:
    """A run of frames with the linked ball near a rim: when, its closest approach (rim widths) and the time of it,
    and where it entered and left the run (height against the rim centre in rim widths, + below)."""

    side: str
    t0: float
    t1: float
    t_near: float
    frames: int
    dmin: float
    dy_first: float
    dy_last: float


def rim_visits(path: Mapping[int, Sequence[float]], camera: Any, court: Any, max_rims: float = 2.5,
               max_gap: int = 3) -> list[RimVisit]:
    """The runs of frames in which the linked ball (frame -> image x, y, ...) lies within ``max_rims`` rim widths of a
    rim the camera projects (``rim_boxes``), gaps of up to ``max_gap`` frames bridged."""
    w, h = camera.image_size
    runs: list[dict[str, Any]] = []
    for f in sorted(path):
        cam = camera.get(f)
        if cam is None:
            continue
        x, y = float(path[f][0]), float(path[f][1])
        for side, b in rim_boxes(cam, court, (w, h)).items():
            rw = max(b.x2 - b.x1, 1.0)
            cy = (b.y1 + b.y2) / 2
            d = math.hypot(x - (b.x1 + b.x2) / 2, y - cy) / rw
            if d > max_rims:
                continue
            run = runs[-1] if runs else None
            if run is not None and run["side"] == side and f - run["f1"] <= max_gap:
                run.update(f1=f, n=run["n"] + 1, dy_last=(y - cy) / rw)
                if d < run["dmin"]:
                    run.update(dmin=d, f_near=f)
            else:
                runs.append({"side": side, "f0": f, "f1": f, "n": 1, "dmin": d, "f_near": f, "dy_first": (y - cy) / rw,
                             "dy_last": (y - cy) / rw})
    return [RimVisit(r["side"], camera.time(r["f0"]), camera.time(r["f1"]), camera.time(r["f_near"]), r["n"],
                     round(r["dmin"], 4), round(r["dy_first"], 4), round(r["dy_last"], 4)) for r in runs]


@dataclass
class Candidate:
    """A possible shot: an engine result (``result``), with the rim visit nearest it, or a rim visit the engine
    reported nothing at (``result`` None). ``t``: when it resolves at the rim; ``key``: the result's event id, or
    "rim-<t>"."""

    key: str
    t: float
    release_t: float
    result: Event | None = None
    visit: RimVisit | None = None

    @property
    def side(self) -> str | None:
        return self.visit.side if self.visit is not None else None


def candidates(events: Iterable[Event], visits: Sequence[RimVisit] = (), visit_max_rims: float = 1.0,
               visit_min_frames: int = 3, near_s: float = 2.0) -> list[Candidate]:
    """The engine's shot results the camera saw, each with the nearest rim visit (coming within ``visit_max_rims`` rim
    widths over at least ``visit_min_frames`` frames) within ``near_s``, and every such visit with no result within
    ``near_s``, in time order."""
    evs = [e for e in latest_revisions(events) if e.status is not EventStatus.RETRACTED]
    attempts = {e.shot_id: e for e in evs if e.kind is K.SHOT_ATTEMPT and e.shot_id}
    results = sorted((e for e in evs if e.kind in RESULTS and not from_scorebug(e)), key=lambda e: e.t)
    close = [v for v in visits if v.dmin <= visit_max_rims and v.frames >= visit_min_frames]
    out = []
    for r in results:
        att = attempts.get(r.shot_id or "")
        v = min((v for v in close if v.t0 - near_s <= r.t <= v.t1 + near_s), key=lambda v: abs(v.t_near - r.t),
                default=None)
        out.append(Candidate(r.event_id, r.t, att.t if att is not None else r.t - 1.0, r, v))
    rt = np.array([r.t for r in results])
    for v in close:
        if not len(rt) or np.min(np.abs(rt - v.t_near)) > near_s:
            out.append(Candidate(f"rim-{v.t_near:.2f}", v.t_near, v.t_near - 1.0, None, v))
    return sorted(out, key=lambda c: c.t)


def candidate_features(c: Candidate, timeline: Timeline, activity: np.ndarray, jev: Mapping[str, Any] | None = None,
                       change_window_s: tuple[float, float] = (-1.0, 8.0), max_back_s: float = 12.0,
                       rim: Mapping[str, float] | None = None, zones: np.ndarray | None = None) -> dict[str, float]:
    """Named evidence about one candidate (see the module docstring); every value is finite. ``activity``: the sorted
    times of the engine's dribbles and passes; ``rim``: the rim model's probabilities for the candidate's rim clip
    (``p_made``, ``p_missed``, ``p_none``; ``rim_model.py``), or None; ``zones``: the detector's ball about the rim
    (``rim_clips.ball_zone_series``), or None."""
    r = c.result
    cues = (r.attrs.get("cues") if r is not None else None) or {}
    margin = cues.get("crossing_margin_px")
    f = {"crossed_inside": float(bool(cues.get("crossed_inside"))), "rim_contact": float(bool(cues.get("rim_contact"))),
         "drop_in_band": float(bool(cues.get("drop_in_band"))),
         "drop_is_freefall": float(bool(cues.get("drop_is_freefall"))),
         "margin": 0.0 if margin is None else float(np.clip(margin, -60.0, 40.0)) / 20.0,
         "margin_missing": float(margin is None), "occluded": min(float(cues.get("occluded_frames") or 0), 5.0),
         "engine": float(r is not None), "engine_ft": float(r is not None and _is_ft(r))}
    v = c.visit
    f.update(visit=float(v is not None), visit_dmin=v.dmin if v is not None else 2.5,
             visit_from_above=float(np.clip(-v.dy_first, -3.0, 3.0)) if v is not None else 0.0,
             visit_below=float(np.clip(v.dy_last, -3.0, 3.0)) if v is not None else 0.0,
             visit_frames=math.log1p(v.frames) / 4.0 if v is not None else 0.0)
    shooter = timeline.team_at(c.release_t - 0.25)
    change = next(((t, gained) for t, _, gained in timeline.changes()
                   if c.t + change_window_s[0] <= t <= c.t + change_window_s[1]), None)
    f["changed"] = float(change is not None and change[1] is not shooter)
    f["dt_change"] = float(np.clip(change[0] - c.t, -1.0, 8.0)) if f["changed"] else 0.0  # type: ignore[index]
    f["back_s"] = _back_seconds(timeline, change[0], max_back_s) if f["changed"] else 0.0  # type: ignore[index]
    f["cam_before"] = timeline.tracked_share(c.t - 20.0, c.t - 5.0)
    n_act = int(np.searchsorted(activity, c.t + 6.0) - np.searchsorted(activity, c.t - 6.0)) if len(activity) else 0
    f["activity"] = min(n_act, 10) / 10.0
    f["is_ft"] = 0.0  # set by the free-throw decision (decide) or from the truth (fitting)
    if jev is not None:
        probs = jev.get("probs") or {}
        made, missed = float(probs.get("made", 0.0)), float(probs.get("missed", 0.0))
        f.update(jev_in_net=max(jev.get("in_net") or [0.0]), jev_made=made / (made + missed) if made + missed > 0 else 0.5,
                 jev_missing=0.0)
    else:
        f.update(jev_in_net=0.0, jev_made=0.5, jev_missing=1.0)
    if rim is not None:
        made, missed = float(rim.get("p_made", 0.0)), float(rim.get("p_missed", 0.0))
        f.update(rim_made=made / (made + missed) if made + missed > 0 else 0.5,
                 rim_shot=1.0 - float(rim.get("p_none", 0.0)), rim_missing=0.0)
    else:
        f.update(rim_made=0.5, rim_shot=0.5, rim_missing=1.0)
    f.update(zone_features(zones))
    return f


def zone_features(series: np.ndarray | None, centre: int = 30, gap: int = 24, seen: float = 0.3) -> dict[str, float]:
    """``ZONE_FEATURES`` from a ball-zone series (``rim_clips.ball_zone_series``: per frame about the rim's arrival,
    the detector's best ball score in each zone of ``rim_clips.ZONES``; -1 on frames without the rim): the ball in the
    net after the arrival (best score, frames seen), above or at the ring and then in the net within ``gap`` frames,
    in the net and then below it, above or beside the ring after the arrival (a miss), at the ring and then beside or
    high (off the rim), and the share of frames with the rim in view. All 0 without a series."""
    if series is None or not len(series):
        return {n: 0.0 for n in ZONE_FEATURES}
    from .rim_clips import ZONES

    z = {name: i for i, (name, _, _) in enumerate(ZONES)}
    v = np.clip(np.asarray(series, float), 0.0, None)
    c = min(max(0, centre), len(v) - 1)

    def then(a: np.ndarray, b: np.ndarray) -> float:  # max over t1 < t2 <= t1 + gap of min(a[t1], b[t2])
        best = 0.0
        for t1 in range(len(a) - 1):
            if a[t1] > best:
                best = max(best, min(float(a[t1]), float(b[t1 + 1:t1 + 1 + gap].max())))
        return best

    top = np.maximum(v[:, z["above"]], v[:, z["ring"]])
    out = np.maximum(v[:, z["side"]], v[:, z["high"]])
    return {"zone_net": float(v[c:, z["net"]].max()), "zone_net_n": float((v[:, z["net"]] > seen).sum()) / 10.0,
            "zone_top_then_net": then(top, v[:, z["net"]]),
            "zone_net_then_under": then(v[:, z["net"]], v[:, z["under"]]),
            "zone_post_above": float(v[c:, z["above"]].max()), "zone_post_side": float(v[c:, z["side"]].max()),
            "zone_top_then_out": then(top, out), "zone_above": float(v[:, z["above"]].max()),
            "zone_ring": float(v[:, z["ring"]].max()), "zone_valid": float((np.asarray(series)[:, 0] >= 0).mean())}


def candidate_rows(events: Iterable[Event], timeline: Timeline, visits: Sequence[RimVisit] = (),
                   jev: Callable[[Candidate], Mapping[str, Any] | None] | None = None,
                   rim: Callable[[Candidate], Mapping[str, float] | None] | None = None,
                   zones: Callable[[Candidate], np.ndarray | None] | None = None,
                   ) -> list[tuple[Candidate, dict[str, float]]]:
    """Each candidate with its features: what ``decide`` and fitting both read. ``jev(candidate)``: Jev's read of its
    rim crops, or None; ``rim(candidate)``: the rim model's probabilities, or None; ``zones(candidate)``: its ball-zone
    series (``rim_clips.ball_zone_series``), or None."""
    evs = [e for e in latest_revisions(events) if e.status is not EventStatus.RETRACTED]
    activity = np.array(sorted(e.t for e in evs if e.kind in (K.DRIBBLE, K.PASS) and not from_scorebug(e)), float)
    return [(c, candidate_features(c, timeline, activity, jev(c) if jev is not None else None,
                                   rim=rim(c) if rim is not None else None,
                                   zones=zones(c) if zones is not None else None))
            for c in candidates(evs, visits)]


def _back_seconds(timeline: Timeline, t: float, cap: float) -> float:
    """Seconds the team that gained the ball at ``t`` spent in its back court (the timeline's phase), at most ``cap``."""
    if timeline.phase is None or len(timeline.t) < 2:
        return 0.0
    i = int(np.searchsorted(timeline.t, t))
    j, n = i, min(len(timeline.t), i + int(cap / max(timeline.t[1] - timeline.t[0], 1e-9)) + 1)
    while j < n and timeline.phase[j] == BACK and timeline.state[j] == timeline.state[min(i, len(timeline.t) - 1)]:
        j += 1
    return float(min(cap, max(0.0, (j - i) * (timeline.t[1] - timeline.t[0]))))


# --------------------------------------------------------------------------- models


@dataclass
class Logistic:
    """p = sigmoid(sum_i w_i * x[name_i] + b) over named features."""

    names: tuple[str, ...]
    weights: tuple[float, ...]
    bias: float

    def p(self, features: Mapping[str, float]) -> float:
        z = self.bias + sum(w * float(features.get(n, 0.0)) for n, w in zip(self.names, self.weights, strict=True))
        return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z))))

    def to_dict(self) -> dict[str, Any]:
        return {"names": list(self.names), "weights": [round(w, 6) for w in self.weights], "bias": round(self.bias, 6)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Logistic:
        return cls(tuple(d["names"]), tuple(float(w) for w in d["weights"]), float(d["bias"]))

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, float]], labels: Sequence[bool], names: Sequence[str],
            l2: float = 1.0) -> Logistic:
        """L2 logistic regression (``events.calibration._fit_logistic``); a constant model when one class is missing."""
        y = np.array(labels, float)
        if not len(y) or y.min() == y.max():
            p = float(np.clip(y.mean() if len(y) else 0.5, 0.02, 0.98))
            return cls(tuple(names), tuple(0.0 for _ in names), math.log(p / (1.0 - p)))
        X = np.array([[float(r.get(n, 0.0)) for n in names] for r in rows])
        w, b = _fit_logistic(X, y, l2, 100)
        return cls(tuple(names), tuple(float(v) for v in w), float(b))


@dataclass
class ShotModels:
    real: Logistic  # P(a candidate is a real shot)
    ft: Logistic  # P(a free throw | a real shot)
    made: Logistic  # P(made | a real shot, and whether it is a free throw: feature "is_ft")

    def to_dict(self) -> dict[str, Any]:
        return {"real": self.real.to_dict(), "ft": self.ft.to_dict(), "made": self.made.to_dict()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ShotModels:
        return cls(Logistic.from_dict(d["real"]), Logistic.from_dict(d["ft"]), Logistic.from_dict(d["made"]))

    @property
    def uses_jev(self) -> bool:
        return any(n in JEV_FEATURES for n in self.real.names + self.made.names)

    @property
    def uses_rim(self) -> bool:
        return any(n in RIM_FEATURES for n in self.real.names + self.made.names)

    @property
    def uses_zones(self) -> bool:
        return any(n in ZONE_FEATURES for n in self.real.names + self.made.names)


def label_candidates(cands: Sequence[Candidate], shots: Sequence[tuple[float, bool, bool]],
                     tol_s: float = 2.5) -> dict[str, str]:
    """Per candidate key: "fg_made", "fg_missed", "ft_made" or "ft_missed" (paired with a play-by-play shot, given as
    (time, free throw, made), one to one, nearest first, within ``tol_s``) or "false" (no shot there)."""
    pairs = sorted((abs(c.t - t), i, j) for i, c in enumerate(cands) for j, (t, _, _) in enumerate(shots)
                   if abs(c.t - t) <= tol_s)
    used_c: set[int] = set()
    used_s: set[int] = set()
    out = {c.key: "false" for c in cands}
    for _, i, j in pairs:
        if i not in used_c and j not in used_s:
            used_c.add(i)
            used_s.add(j)
            _, ft, made = shots[j]
            out[cands[i].key] = f"{'ft' if ft else 'fg'}_{'made' if made else 'missed'}"
    return out


def fit_shot_models(rows: Sequence[Mapping[str, float]], labels: Sequence[str], jev: bool = False,
                    l2: float = 1.0, rim: bool = False, zones: bool = False) -> ShotModels:
    """The three models from candidates' features and labels (``label_candidates``); ``jev``, ``rim``, ``zones``: the
    real-shot and made models read Jev's features, the rim model's, the ball-zone features too."""
    extra = JEV_FEATURES if jev else ()
    real = Logistic.fit(rows, [lab != "false" for lab in labels],
                        CUE_FEATURES + VISIT_FEATURES + TIMELINE_FEATURES + FT_FEATURES[:2] + extra, l2)
    # made or missed only: the zones lowered the held-out real-shot accuracy, and a rim model may have been trained
    # on real shots alone (its p_none then means nothing)
    extra += (RIM_FEATURES[:1] + RIM_FEATURES[2:] if rim else ()) + (ZONE_FEATURES if zones else ())
    shots = [({**r, "is_ft": float(lab.startswith("ft"))}, lab) for r, lab in zip(rows, labels, strict=True)
             if lab != "false"]
    ft = Logistic.fit([r for r, _ in shots], [lab.startswith("ft") for _, lab in shots], FT_FEATURES, l2)
    made = Logistic.fit([r for r, _ in shots], [lab.endswith("made") for _, lab in shots],
                        CUE_FEATURES + VISIT_FEATURES + TIMELINE_FEATURES + ("is_ft",) + extra, l2)
    return ShotModels(real, ft, made)


# --------------------------------------------------------------------------- decisions


@dataclass
class DecideConfig:
    real_min: float = 0.5  # a candidate less likely than this to be a real shot is no shot
    ft_min: float = 0.5  # a real shot at least this likely a free throw is one
    made_min: float = 0.5  # a real shot at least this likely made is made
    assist_window_s: float = 3.5  # a make's assist: the last pass of a teammate this long before the release
    # (FusionConfig.assist_window_s)


def decide(events: Iterable[Event], timeline: Timeline, models: ShotModels | Callable[[Candidate], ShotModels],
           visits: Sequence[RimVisit] = (), config: DecideConfig | None = None,
           jev: Callable[[Candidate], Mapping[str, Any] | None] | None = None,
           rim: Callable[[Candidate], Mapping[str, float] | None] | None = None,
           zones: Callable[[Candidate], np.ndarray | None] | None = None) -> list[Event]:
    """The engine's events with every shot decided from the video (``attrs["makes"]``: the probabilities). An engine
    result that is no shot goes with its attempt, block and assist; the others become free throws (value 1) or field
    goals (the engine's value, else 2), made or missed, their attempts following. A rim visit that is a shot becomes
    an attempt and a result of the timeline's team (``attrs["inferred"] == "rim_visit"``). An assist stays with a
    made field goal only, and a made field goal with none takes the last pass of a teammate (the timeline's team at
    the pass) within ``assist_window_s`` before the release. Results the engine inferred from the scorebug are dropped
    with their attempts. ``models``: the models, or the models for a candidate (e.g. fitted on the other half of the
    game); ``visits``, ``jev``, ``rim``, ``zones``: as ``candidate_rows``."""
    cfg = config or DecideConfig()
    evs = [e for e in latest_revisions(events) if e.status is not EventStatus.RETRACTED]
    attempts = {e.shot_id: e for e in evs if e.kind is K.SHOT_ATTEMPT and e.shot_id}
    changed: dict[str, Event] = {}  # engine results and attempts, relabelled
    added: list[Event] = []  # the shots at rim visits
    dropped: set[str] = set()
    no_assist: set[str] = set()
    made_fg: list[tuple[Event, float]] = []
    for i, (c, f) in enumerate(candidate_rows(evs, timeline, visits, jev, rim, zones)):
        m = models(c) if callable(models) else models
        p_real = m.real.p(f)
        if p_real < cfg.real_min:
            if c.result is not None:
                dropped.add(c.result.shot_id or c.result.event_id)
            continue
        p_ft = m.ft.p(f)
        ft = p_ft >= cfg.ft_min
        p_made = m.made.p({**f, "is_ft": float(ft)})
        kind = K.SHOT_MADE if p_made >= cfg.made_min else K.SHOT_MISSED
        probs = {"p_real": round(p_real, 4), "p_ft": round(p_ft, 4), "p_made": round(p_made, 4)}
        if c.result is not None:
            r = c.result
            value = 1 if ft else (r.value if r.value in (2, 3) else 2)
            res = changed[r.event_id] = dataclasses.replace(r, kind=kind, value=value, revision=r.revision + 1,
                                                            attrs=_shot_type({**r.attrs, "makes": probs}, ft))
            att = attempts.get(r.shot_id or "")
            if att is not None and (att.value != value or _is_ft(att) != ft):
                changed[att.event_id] = dataclasses.replace(att, value=value, revision=att.revision + 1,
                                                            attrs=_shot_type(dict(att.attrs), ft))
            sid = r.shot_id or r.event_id
        else:
            sid = f"rv{i + 1}"
            team = timeline.team_at(c.release_t - 0.25)
            value = 1 if ft else 2
            attrs = _shot_type({"inferred": "rim_visit"}, ft)
            added.append(Event(f"{sid}a", K.SHOT_ATTEMPT, c.release_t, 0, team=team, value=value, shot_id=sid,
                               confidence=0.5, attrs=attrs))
            res = Event(f"{sid}r", kind, c.t, 0, team=team, value=value, shot_id=sid, confidence=0.5,
                        attrs={**attrs, "makes": probs})
            added.append(res)
        if ft or kind is K.SHOT_MISSED:
            no_assist.add(sid)
        else:
            made_fg.append((res, c.release_t))
    out = []
    for e in evs:
        if e.event_id in changed:
            out.append(changed[e.event_id])
        elif e.kind in RESULTS or (from_scorebug(e) and e.kind is K.SHOT_ATTEMPT):
            continue  # no shot, or inferred from the scorebug
        elif e.shot_id and e.shot_id in dropped and e.kind in (K.SHOT_ATTEMPT, K.BLOCK, K.ASSIST):
            continue
        elif e.kind is K.ASSIST and e.shot_id in no_assist:
            continue
        else:
            out.append(e)
    out += added
    assisted = {e.shot_id for e in out if e.kind is K.ASSIST}
    passes = sorted((e for e in evs if e.kind is K.PASS and not from_scorebug(e)), key=lambda e: e.t)
    for i, (r, rel) in enumerate(made_fg):
        if r.shot_id in assisted:
            continue
        team = timeline.team_at(rel - 0.25)
        att = attempts.get(r.shot_id or "")
        scorer = att.player_id if att is not None else r.player_id
        last = [p for p in passes if rel - cfg.assist_window_s <= p.t <= rel and timeline.team_at(p.t) is team
                and (scorer is None or p.player_id != scorer)]
        if last:
            p = last[-1]
            out.append(Event(f"mk-a{i + 1}", K.ASSIST, p.t, p.frame_idx, team=team, player_id=p.player_id,
                             other_player_id=scorer, shot_id=r.shot_id, confidence=0.6,
                             attrs={"makes": "last pass of a teammate before a make", "inferred": True}))
    return sorted(out, key=lambda e: (e.t, e.event_id))


def _shot_type(attrs: dict[str, Any], ft: bool) -> dict[str, Any]:
    """``attrs`` with ``shot_type`` "free_throw" for a free throw, and none left over for a field goal."""
    if ft:
        attrs["shot_type"] = "free_throw"
    elif attrs.get("shot_type") == "free_throw":
        del attrs["shot_type"]
    return attrs
