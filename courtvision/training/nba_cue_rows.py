"""Cue rows from real NBA games: the event engine's cues labelled by the aligned play-by-play (research use).

    python -m courtvision.training.nba_cue_rows runs/nba/<game> --tag <tag> [--file events.jsonl] [--out PATH] [--tol 2.5]

**Inputs** (the game folder of docs/NBA.md): ``runs/<tag>/run.json`` (the analysed window), ``runs/<tag>/events.jsonl``
(the engine's own events: ``attrs["cues"]`` / ``attrs["three_cues"]`` are what its cue models scored; not
``fused.jsonl``, whose outcomes and teams the scorebug fusion rewrote), the ground truth of ``tools/nba_analyze.py``
``truth()`` (ESPN play-by-play on video time through the scorebug clock, the truth ``eval`` scores against), the
scorebug reads, and ``nba_pbp.csv`` (NBA.com shots with locations) when the folder has it.

**Output**: ``runs/<tag>/cue_rows.jsonl`` in the ``fit_cue_models`` schema (``CUE_ROW_KEYS``; ``calibration_source``
"nba_pbp", ``game_id`` the folder name, ``event_id`` / ``t`` the engine event's; extra keys ``truth_event_id`` and
``dt_s`` (engine minus truth time) of the paired true event (the attempt, for shot, make and three rows; None when
unpaired), ``label_source``, ``run``, ``clean`` / ``scorebug`` on make rows, ``loc_err_ft`` on three rows) and
``cue_rows.summary.json`` next to it (rows, labels and skips per kind and reason, the alignment summary). Fit on every game's rows: ``fit_cue_models --jsonl <game 1> --jsonl
<game 2> ...``; it holds out whole games, so each analysed game adds training rows and held-out evidence.

**Pairing.** Engine events of one kind are paired one-to-one with true events of that kind within ``tol_s`` (most
pairs, then the least total |dt|: the evaluation's assignment, ``eval.events.gated_assignment``) under the team rule:
the same team, where an engine event without a team is compatible with either team. Shots also pair FG with FG and
FT with FT. Only events with cues take part, the population the cue models scored: an event without cues (a
scoreboard-inferred make, an engine team rebound) neither gives a row nor takes a partner from one that has cues.

**Labels.** The play-by-play sits on video time with an error of about 1-2 s and the engine gets teams wrong, so a row
is written only when its label does not hinge on either; the rest is skipped and counted, never labelled negative:

* **shot** (``cues`` of a SHOT_ATTEMPT): 1 when paired. 0 only when nothing explains the attempt within
  ``tol_s + guard_s``: no unpaired true attempt of its team and type (an alignment error beyond the tolerance), no
  true attempt of the other team (a team error: the shot was real), no true attempt of the other type (FG vs FT) and
  no shooting foul (a fouled miss has no FGA row). Free throws share a stopped clock and are placed by their order
  between score anchors (or by the running clock when their clock misses the stoppage: tens of seconds off), so near a
  true FT the reach is ``tol_s + ft_guard_s``, and an attempt while the scorebug's clock is stopped at the clock of a
  true free throw (``stoppage_clock_tol_s``) is never a negative, whatever times the FTs were given. A second engine
  attempt for one true attempt (a duplicate) is 0.
* **make** (``cues`` of a SHOT_MADE / SHOT_MISSED, linked to its attempt by ``shot_id``), from two sources:
  play-by-play: the outcome of the paired true shot, kept only when every true attempt of a compatible team within
  ``tol_s`` of the engine attempt has that outcome (a put-back after a miss otherwise flips the label with the
  alignment error) or the NBA.com locations single one out; scorebug (ESPN shows a basket ``score_lag_s`` after it
  goes in): "made" when exactly one score change of any team lies in the quiet window after the result, and it lies
  in the make window, has the right size (+1 for a FT, +2 / +3 for a FG), is of the engine's team (or the engine has
  none), lies in no other engine result's make window and follows no missed FG of that team within ``putback_s``
  (it could be the put-back); "missed" when no score change of any team lies in the quiet window and the bug was read
  across it with the same score at both ends. Both agree: the label, ``clean``. They disagree: skipped. Play-by-play
  only: its label. Scorebug only (the attempt has no unambiguous true partner): "missed" only, and only when no true
  make of any team lies within ``tol_s + guard_s`` of the result (a late bug update or a basket given to the wrong
  team would otherwise read as a miss). No basket at all is certain whichever flight the engine saw, so a phantom
  shot's result is a miss and misses the clock placed too far from their engine shot are not lost; a basket on the
  bug does not say which flight went in (the engine may not have seen the one that did), so it never labels alone.
  ``clean`` marks the rows the scorebug confirms: a made basket is certain on the bug, and so is its absence.
* **three** (``three_cues`` of a paired FG attempt): the value of the paired true shot, NBA.com's (from its
  location) when the shot is linked to an NBA.com row (same period, team and result, clock within
  ``nba_clock_tol_s``), else ESPN's. Candidates with different values: resolved by the NBA.com locations, else
  skipped.
* **rebound / steal / block**: 1 when paired with a true event of the same kind and team (the rebounder's, the
  stealer's, the blocker's); 0 under the shot rule (nothing of that kind within ``tol_s + guard_s``), where a team
  rebound (feed bookkeeping, not in the truth) also explains an engine rebound.

**NBA.com locations** (``nba_pbp.csv``; its team codes are inferred by matching its shots to ESPN's): when every
candidate of an ambiguous shot has one, the candidate nearest the engine's shot point (gather point, else release)
is chosen if it lies within ``loc_max_ft`` and all others are at least ``loc_margin_ft`` farther. **Window and
coverage**: engine events outside the run window, or with no scorebug clock read within ``coverage_s`` (no truth
there), give no row; the truth is taken from the window padded by the reach, so an event at a window edge still
finds its partner.

**Limits.** Time decides the pairing: a phantom attempt within ``tol_s`` of a true shot the engine did not see takes
that shot's labels, and of an attempt and its duplicate the one nearer in time gets the 1. The ESPN bug lag (2 s) is
one game's median; the guards and windows here are hand-set and were checked on synthetic games only.

**Fitting** (``fit_cue_models``). The fitter holds out round(0.3 x games) games (at least one, ``split_by_game``).
Rows of a single game leave no held-out game: the fitter then refits in sample (status ``platt`` once >= 300 rows
of a kind with >= 50 in the ambiguous band), reports ``split: "in_sample"``, ``n_holdout: 0``, and never certifies
a gate (reason "no held-out games: ..."); it does not fail. Held-out calibration and certificates need rows of
>= 2 games. Use one run per game: two runs of one game share its ``game_id`` (one side of the split) but count twice.

NBA footage and statistics: research use only (docs/TRAINING.md §1).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from courtvision.eval.events import gated_assignment, latest_revisions
from courtvision.events.scoreboard import ScoreChange
from courtvision.nba.evaluate import truth_events
from courtvision.nba.nba_stats import NBAShot, load_shots
from courtvision.training.align_pbp import AlignedAction, ClockTimeline, PBPAction, score_changes
from courtvision.training.fit_cue_models import CUE_ROW_KEYS, write_cue_rows
from courtvision.types import Event, EventKind, ScoreboardReading, Team

__all__ = [
    "CUE_ROW_KEYS",
    "KINDS",
    "NBA_SOURCE",
    "GameRun",
    "NBACueConfig",
    "infer_nba_codes",
    "label_game",
    "link_nba_shots",
    "load_game_run",
    "main",
    "run_window",
    "summary_path",
]

log = logging.getLogger(__name__)

NBA_SOURCE = "nba_pbp"
KINDS: tuple[str, ...] = ("shot", "make", "three", "rebound", "steal", "block")
_RESULTS = (EventKind.SHOT_MADE, EventKind.SHOT_MISSED)
_EPS = 1e-9
_TRUTH_ID = re.compile(r"pbp(\d+)[ar]?")  # nba.evaluate.truth_events: "pbp<aligned index>" (+ "a" / "r" for shots)


@dataclass
class NBACueConfig:
    """Labelling settings (module docstring). Times are video seconds."""

    tol_s: float = 2.5  # pairing tolerance: the evaluation's (nba_analyze.py eval --tol)
    guard_s: float = 1.5  # an unpaired event is 0 only with nothing within tol_s + guard_s (alignment error 1-2 s); tuned
    ft_guard_s: float = 12.0  # FTs in a stoppage are placed by order between score anchors (seconds off); tuned
    stoppage_clock_tol_s: float = 1.0  # a true FT belongs to the stoppage whose stopped clock is this close; tuned
    coverage_s: float = 5.0  # a scorebug clock read this close to an event: the truth covers it; tuned
    score_lag_s: float = 2.0  # ESPN bug update after a basket (ESPN_ALIGN; FusionConfig.score_lag_s): one game's median
    make_window_s: tuple[float, float] = (-1.5, 3.5)  # FG: score change - (result t + lag); FusionConfig.make_window_s
    ft_make_window_s: tuple[float, float] = (-2.0, 4.0)  # FT: FusionConfig.ft_window_s
    quiet_window_s: tuple[float, float] = (-3.0, 6.0)  # no score change in here (bug read) -> missed; tuned
    ft_quiet_window_s: tuple[float, float] = (-3.0, 6.0)  # tuned
    read_gap_s: float = 2.0  # the bug counts as read while score reads are at most this far apart; tuned
    putback_s: float = 6.0  # a basket this soon after a missed FG of the same team may be its put-back; tuned
    loc_max_ft: float = 8.0  # NBA.com location choice: the nearest candidate within this of the engine's point ...
    loc_margin_ft: float = 8.0  # ... and every other candidate at least this much farther; tuned
    nba_clock_tol_s: float = 1.0  # ESPN <-> NBA.com shot link (ESPN clocks are whole seconds); tuned
    kinds: tuple[str, ...] = KINDS


@dataclass
class GameRun:
    """One analysed run of one game, loaded from its folder (``load_game_run``)."""

    game_id: str
    events: list[Event]
    aligned: list[AlignedAction]
    window: tuple[float, float] | None = None
    reads: list[ScoreboardReading] = field(default_factory=list)
    timeline: ClockTimeline | None = None
    nba_shots: list[NBAShot] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------- helpers


def _team_ok(a: Team | None, b: Team | None) -> bool:
    return a is None or b is None or a is b


def _is_ft(e: Event) -> bool:
    return e.value == 1 or e.attrs.get("shot_type") == "free_throw"


def _is_shooting_foul(a: PBPAction) -> bool:
    text = " ".join(str(v) for v in (a.attrs.get("type", ""), a.sub_type, a.description)).lower()
    return a.kind is EventKind.FOUL and "shoot" in text


def _shot_point(e: Event) -> tuple[float, float] | None:
    gx, gy = e.attrs.get("gather_x"), e.attrs.get("gather_y")
    if isinstance(gx, (int, float)) and isinstance(gy, (int, float)):
        return float(gx), float(gy)
    return (float(e.x), float(e.y)) if e.x is not None and e.y is not None else None


@dataclass
class _Pairing:
    pair: dict[int, int]  # engine index -> truth index
    cand: list[list[int]]  # engine index -> compatible truth indices within tol
    taken: set[int]


def _pairing(pred: Sequence[Event], truth: Sequence[Event], tol: float,
             compat: Callable[[Event, Event], bool]) -> _Pairing:
    cand: list[list[int]] = [[] for _ in pred]
    pair: dict[int, int] = {}
    if pred and truth:
        dt = np.abs(np.array([e.t for e in pred])[:, None] - np.array([e.t for e in truth])[None, :])
        ok = dt <= tol + _EPS
        for i, j in zip(*np.nonzero(ok), strict=True):
            if compat(pred[i], truth[j]):
                cand[int(i)].append(int(j))
            else:
                ok[i, j] = False
        pair = dict(gated_assignment(dt, ok))
    return _Pairing(pair, cand, set(pair.values()))


def run_window(run: Mapping[str, Any]) -> tuple[float, float]:
    """(start, end) of a run from its run.json: 0 / None mean the video's start / end (as ``nba_analyze eval``)."""
    lo = float(run.get("start") or 0.0)
    hi = float(run.get("end") or 0.0)
    return lo, (hi if hi > 0.0 else math.inf)


# ---------------------------------------------------------------------------------------------- NBA.com shots


def link_nba_shots(aligned: Sequence[AlignedAction], shots: Sequence[NBAShot],
                   clock_tol_s: float = 1.0) -> dict[int, NBAShot]:
    """ESPN field-goal actions (index into ``aligned``) -> the same shot in NBA.com's play-by-play: same period, team
    and result, clocks within ``clock_tol_s``; one to one, nearest clock first, equal values preferred."""
    pairs = []
    for i, al in enumerate(aligned):
        a = al.action
        if not a.is_field_goal or a.team is None:
            continue
        for k, s in enumerate(shots):
            d = abs(s.clock - a.clock)
            if s.period == a.period and s.team is a.team and s.made == bool(a.made) and d <= clock_tol_s + _EPS:
                pairs.append((d + (0.0 if s.value == a.value else 0.5), i, k))
    out: dict[int, NBAShot] = {}
    used: set[int] = set()
    for _, i, k in sorted(pairs):
        if i not in out and k not in used:
            out[i] = shots[k]
            used.add(k)
    return out


def infer_nba_codes(path: str | Path, actions: Sequence[PBPAction], clock_tol_s: float = 1.0,
                    min_votes: int = 2) -> dict[str, Team]:
    """Team code of ``nba_pbp.csv`` -> Team, by matching its shots to ESPN's (same period, result and value, clocks
    within ``clock_tol_s``, one team only). {} unless two codes each get ``min_votes`` votes, 80% of them for
    one team, the two teams different."""
    fg = [a for a in actions if a.is_field_goal and a.team is not None]
    votes: dict[str, Counter[Team]] = {}
    with Path(path).open(newline="") as fh:
        for r in csv.DictReader(fh):
            code = (r.get("team") or "").strip()
            if r.get("type") not in ("2", "3") or not code:
                continue
            try:
                period, clock = int(r["period"]), float(r["clock"])
            except (KeyError, TypeError, ValueError):
                continue
            value, made = int(r["type"]), r.get("res") == "M"
            teams = {a.team for a in fg if a.period == period and bool(a.made) == made and a.value == value
                     and abs(a.clock - clock) <= clock_tol_s + _EPS}
            if len(teams) == 1:
                votes.setdefault(code, Counter())[teams.pop()] += 1  # type: ignore[arg-type]
    out: dict[str, Team] = {}
    for code, c in votes.items():
        team, n = c.most_common(1)[0]
        if n >= min_votes and n >= 0.8 * sum(c.values()):
            out[code] = team
    return out if len(out) == 2 and len(set(out.values())) == 2 else {}


# ---------------------------------------------------------------------------------------------- labelling


class _Labeller:
    def __init__(self, events: Iterable[Event], aligned: Sequence[AlignedAction], game_id: str,
                 window: tuple[float, float] | None, reads: Sequence[ScoreboardReading] | None,
                 timeline: ClockTimeline | None, nba_shots: Sequence[NBAShot] | None, cfg: NBACueConfig) -> None:
        self.cfg, self.game_id = cfg, game_id
        self.lo, self.hi = (float(window[0]), float(window[1])) if window is not None else (-math.inf, math.inf)
        reach = cfg.tol_s + max(cfg.guard_s, cfg.ft_guard_s)
        self.aligned = list(aligned)
        self.truth = truth_events(self.aligned, (self.lo - reach, self.hi + reach))
        self.src: dict[str, int] = {}
        for ev in self.truth:
            m = _TRUTH_ID.fullmatch(ev.event_id)
            if m is None:
                raise ValueError(f"unexpected truth event id {ev.event_id!r} (nba.evaluate.truth_events changed?)")
            self.src[ev.event_id] = int(m.group(1))
        placed = [al for al in self.aligned if al.t is not None and self.lo - reach <= al.t <= self.hi + reach]
        self.fouls = [float(al.t) for al in placed if _is_shooting_foul(al.action)]  # type: ignore[arg-type]
        self.team_rebounds = [float(al.t) for al in placed  # type: ignore[arg-type]
                              if al.action.kind is EventKind.REBOUND and al.action.player_id is None
                              and not al.action.synthetic]
        self.ft_clocks = [(al.action.period, al.action.clock) for al in self.aligned if al.action.is_free_throw]
        self._ft_stops: list[tuple[float, float]] | None = None  # (t0, t1) of stoppages with a true FT (lazy)
        self.eng = latest_revisions(events)
        self.timeline = timeline
        reads = list(reads or [])
        clock_t = sorted(r.t for r in reads if r.game_clock is not None)
        self.clock_t = np.array(clock_t) if reads else None
        score = sorted((r.t, (r.home_score, r.away_score)) for r in reads
                       if r.home_score is not None and r.away_score is not None)
        self.score_t = np.array([t for t, _ in score])
        self.score_v = [v for _, v in score]
        self.changes: list[ScoreChange] | None = sorted(score_changes(reads), key=lambda c: c.t) if reads else None
        self.nba = link_nba_shots(self.aligned, nba_shots or [], cfg.nba_clock_tol_s)
        # linked shots whose value NBA.com and ESPN disagree on (NBA.com's is used)
        self.value_conflicts = sum(s.value != self.aligned[i].action.value for i, s in self.nba.items())
        self.rows: list[dict[str, Any]] = []
        self.stats: dict[str, Counter[str]] = {k: Counter() for k in KINDS}

    # ------------------------------------------------------------------ bookkeeping

    def _in_window(self, e: Event) -> bool:
        return self.lo <= e.t <= self.hi

    def _covered(self, t: float) -> bool:
        if self.clock_t is not None:
            k = int(np.searchsorted(self.clock_t, t))
            near = [abs(self.clock_t[j] - t) for j in (k - 1, k) if 0 <= j < len(self.clock_t)]
            if not near or min(near) > self.cfg.coverage_s:
                return False
        if self.timeline is not None:
            cp = self.timeline.clock_at(t)
            if cp is None or cp.extrapolated:
                return False
        return True

    def _skip(self, kind: str, reason: str) -> None:
        self.stats[kind][f"skip:{reason}"] += 1

    def _row(self, kind: str, e: Event, feats: Mapping[str, Any], label: bool, source: str,
             truth: Event | None = None, **extra: Any) -> None:
        row: dict[str, Any] = {"kind": kind, "features": dict(feats), "label": bool(label), "game_id": self.game_id,
                               "calibration_source": NBA_SOURCE, "event_id": e.event_id, "t": round(e.t, 3),
                               "truth_event_id": truth.event_id if truth is not None else None,
                               "dt_s": round(e.t - truth.t, 3) if truth is not None else None, "label_source": source}
        row.update(extra)
        self.rows.append(row)
        st = self.stats[kind]
        st["rows"] += 1
        st["label_1" if label else "label_0"] += 1
        st[f"source:{source}"] += 1

    def _action(self, ev: Event) -> PBPAction:
        return self.aligned[self.src[ev.event_id]].action

    def _value(self, ev: Event) -> tuple[int, str]:
        """(value, "nba" | "espn") of a true FG attempt: NBA.com's when the shot is linked."""
        s = self.nba.get(self.src[ev.event_id])
        return (s.value, "nba") if s is not None else (ev.value, "espn")

    def _location(self, ev: Event) -> tuple[float, float] | None:
        s = self.nba.get(self.src[ev.event_id])
        return (s.x, s.y) if s is not None else None

    def _ft_stoppage(self, t: float) -> bool:
        """The scorebug's clock is stopped at ``t`` (a timeline plateau) at the clock of a true free throw."""
        if self._ft_stops is None:
            self._ft_stops = []
            tol = self.cfg.stoppage_clock_tol_s
            for p in self.timeline.periods if self.timeline is not None else []:
                clocks = [c for per, c in self.ft_clocks if per == p]
                self._ft_stops += [(t0 - tol, t1 + tol) for t0, t1, v in self.timeline.plateaus(p)  # type: ignore[union-attr]
                                   if any(abs(c - v) <= tol + _EPS for c in clocks)]
        return any(a <= t <= b for a, b in self._ft_stops)

    def _why_unpaired(self, e: Event, P: _Pairing, truth: Sequence[Event], shots: bool = False,
                      rebounds: bool = False) -> str | None:
        """Why an unpaired engine event may still be real (skip), or None (a clean negative)."""
        cfg, found = self.cfg, set()
        ft = shots and _is_ft(e)
        if shots and self._ft_stoppage(e.t):
            found.add("ft_stoppage")  # a free throw of this stoppage, wherever the alignment placed it
        for j, t in enumerate(truth):
            other_type = shots and _is_ft(t) != ft
            reach = cfg.tol_s + (cfg.ft_guard_s if shots and _is_ft(t) else cfg.guard_s)  # the true event's placement
            if abs(e.t - t.t) > reach + _EPS:
                continue
            if other_type:
                found.add("shot_type")
            elif not _team_ok(e.team, t.team):
                found.add("other_team")
            elif j not in P.taken:
                found.add("near_tolerance")
        near = cfg.tol_s + cfg.guard_s + _EPS
        if shots and not ft and any(abs(e.t - t) <= near for t in self.fouls):
            found.add("shooting_foul")
        if rebounds and any(abs(e.t - t) <= near for t in self.team_rebounds):
            found.add("team_rebound")
        for reason in ("other_team", "shot_type", "ft_stoppage", "shooting_foul", "team_rebound", "near_tolerance"):
            if reason in found:
                return reason
        return None

    def _resolve(self, e: Event, i: int, P: _Pairing, truth: Sequence[Event],
                 key: Callable[[Event], Any]) -> tuple[int | None, str]:
        """The true partner whose label is used: the paired one when every candidate within tol gives the same
        label, else the one the NBA.com locations single out; (None, "ambiguous") otherwise."""
        js = P.cand[i]
        if len({key(truth[j]) for j in js}) == 1:
            return P.pair[i], "time"
        p = _shot_point(e)
        locs = [self._location(truth[j]) for j in js]
        if p is None or any(q is None for q in locs):
            return None, "ambiguous"
        ds = sorted((math.hypot(p[0] - q[0], p[1] - q[1]), j) for q, j in zip(locs, js, strict=True))  # type: ignore[index]
        if ds[0][0] > self.cfg.loc_max_ft or ds[1][0] - ds[0][0] < self.cfg.loc_margin_ft:
            return None, "ambiguous"
        return ds[0][1], "location"

    # ------------------------------------------------------------------ scorebug

    def _bug_read(self, a: float, b: float) -> bool:
        """Score reads cover [a, b] (gaps <= read_gap_s) and show the same score at both ends."""
        gap = self.cfg.read_gap_s
        i, j = int(np.searchsorted(self.score_t, a, "left")), int(np.searchsorted(self.score_t, b, "right"))
        if j - i < 2:
            return False
        ts = self.score_t[i:j]
        if ts[0] - a > gap or b - ts[-1] > gap or float(np.max(np.diff(ts))) > gap:
            return False
        first = Counter(self.score_v[k] for k in range(i, j) if self.score_t[k] <= a + gap).most_common(1)[0][0]
        last = Counter(self.score_v[k] for k in range(i, j) if self.score_t[k] >= b - gap).most_common(1)[0][0]
        return first == last

    def _bug_verdict(self, r: Event, team: Team | None, ft: bool, results: Sequence[tuple[Event, bool]],
                     truth_att: Sequence[Event]) -> tuple[bool | None, str]:
        """(made / missed / None, why not) from the scorebug for the engine result ``r``. ``results``: every engine
        result (also those without cues: an inferred make is the engine's own claim on a basket)."""
        cfg = self.cfg
        if self.changes is None:
            return None, "no_scorebug"
        lag = cfg.score_lag_s
        q0, q1 = cfg.ft_quiet_window_s if ft else cfg.quiet_window_s
        m0, m1 = cfg.ft_make_window_s if ft else cfg.make_window_s
        base = r.t + lag
        quiet = [c for c in self.changes if base + q0 <= c.t <= base + q1]
        if not quiet:
            return (False, "") if self._bug_read(base + q0, base + q1) else (None, "bug_unread")
        c = quiet[0]
        if len(quiet) > 1 or not base + m0 <= c.t <= base + m1 or (c.delta == 1) != ft:
            return None, "bug_busy"
        if not _team_ok(team, c.team):
            return None, "bug_other_team"
        for o, o_ft in results:
            if o.event_id != r.event_id and o_ft == ft and _team_ok(o.team, c.team) \
                    and o.t + lag + m0 <= c.t <= o.t + lag + m1:
                return None, "bug_shared"
        if not ft and any(not self._action(t).made and t.team is c.team and c.t - lag - cfg.putback_s <= t.t <= c.t - lag
                          for t in truth_att if not _is_ft(t)):
            return None, "bug_putback"
        return True, ""

    def _true_make_near(self, r: Event, ft: bool) -> bool:
        """A true make (FT for a FT) of any team within tol + guard of the result: a scorebug-only miss is not trusted."""
        reach = self.cfg.tol_s + self.cfg.guard_s + _EPS
        return any(t.kind is EventKind.SHOT_MADE and _is_ft(t) == ft and abs(t.t - r.t) <= reach for t in self.truth)

    # ------------------------------------------------------------------ kinds

    def run(self) -> list[dict[str, Any]]:
        want = set(self.cfg.kinds)
        att = [e for e in self.eng if e.kind is EventKind.SHOT_ATTEMPT and isinstance(e.attrs.get("cues"), Mapping)]
        truth_att = [t for t in self.truth if t.kind is EventKind.SHOT_ATTEMPT]
        P = _pairing(att, truth_att, self.cfg.tol_s, lambda p, t: _is_ft(p) == _is_ft(t) and _team_ok(p.team, t.team))
        if "shot" in want:
            self._shots(att, truth_att, P)
        if "make" in want:
            self._makes(att, truth_att, P)
        if "three" in want:
            self._threes(att, truth_att, P)
        for kind, ek in (("rebound", EventKind.REBOUND), ("steal", EventKind.STEAL), ("block", EventKind.BLOCK)):
            if kind in want:
                self._simple(kind, ek)
        return self.rows

    def _shots(self, att: Sequence[Event], truth_att: Sequence[Event], P: _Pairing) -> None:
        for i, e in enumerate(att):
            cues = e.attrs["cues"]
            if not self._in_window(e):
                continue
            if not self._covered(e.t):
                self._skip("shot", "no_truth_coverage")
            elif i in P.pair:
                self._row("shot", e, cues, True, "pbp", truth_att[P.pair[i]])
            elif (why := self._why_unpaired(e, P, truth_att, shots=True)) is not None:
                self._skip("shot", why)
            else:
                self._row("shot", e, cues, False, "pbp")

    def _makes(self, att: Sequence[Event], truth_att: Sequence[Event], P: _Pairing) -> None:
        index = {e.shot_id: i for i, e in enumerate(att) if e.shot_id}
        results = []
        for r in (e for e in self.eng if e.kind in _RESULTS):
            i = index.get(r.shot_id or "")
            results.append((r, _is_ft(att[i]) if i is not None else _is_ft(r)))
        for r, ft in results:
            cues = r.attrs.get("cues")
            if not isinstance(cues, Mapping) or not self._in_window(r):
                continue
            if not self._covered(r.t):
                self._skip("make", "no_truth_coverage")
                continue
            i = index.get(r.shot_id or "")
            a = att[i] if i is not None else None
            pbp: bool | None = None
            pbp_src, why, truth = "pbp", "unpaired" if a is not None else "no_attempt", None
            if a is not None and i in P.pair:
                j, how = self._resolve(a, i, P, truth_att, lambda t: bool(self._action(t).made))
                if j is None:
                    why = "ambiguous_outcome"
                else:
                    pbp, truth = bool(self._action(truth_att[j]).made), truth_att[j]
                    pbp_src = "pbp" if how == "time" else "nba_location"
            bug, bug_why = self._bug_verdict(r, a.team if a is not None else r.team, ft, results, truth_att)
            verdict = None if bug is None else ("made" if bug else "missed")
            subject = a if a is not None else r
            if pbp is not None and bug is not None and pbp != bug:
                self._skip("make", "scorebug_conflict")
            elif pbp is not None:
                self._row("make", subject, cues, pbp, pbp_src + ("+scorebug" if bug is not None else ""), truth,
                          event_id=r.event_id, t=round(r.t, 3), clean=bug is not None, scorebug=verdict)
            elif bug is True:
                self._skip("make", "scorebug_unattributed")  # a basket, but not necessarily this flight's
            elif bug is False and self._true_make_near(r, ft):
                self._skip("make", "scorebug_unsupported")
            elif bug is False:
                self._row("make", subject, cues, False, "scorebug", None, event_id=r.event_id, t=round(r.t, 3),
                          clean=True, scorebug=verdict)
            else:
                self._skip("make", why if why == "ambiguous_outcome" else f"{why}+{bug_why}")

    def _threes(self, att: Sequence[Event], truth_att: Sequence[Event], P: _Pairing) -> None:
        for i, e in enumerate(att):
            cues = e.attrs.get("three_cues")
            if not isinstance(cues, Mapping) or _is_ft(e) or not self._in_window(e):
                continue
            if not self._covered(e.t):
                self._skip("three", "no_truth_coverage")
                continue
            if i not in P.pair:
                self._skip("three", "unpaired")
                continue
            j, how = self._resolve(e, i, P, truth_att, lambda t: self._value(t)[0])
            if j is None:
                self._skip("three", "ambiguous_value")
                continue
            value, src = self._value(truth_att[j])
            p, q = _shot_point(e), self._location(truth_att[j])
            loc_err = round(math.hypot(p[0] - q[0], p[1] - q[1]), 2) if p is not None and q is not None else None
            self._row("three", e, cues, value == 3, "nba_location" if how == "location" else src, truth_att[j],
                      loc_err_ft=loc_err)

    def _simple(self, kind: str, ek: EventKind) -> None:
        pred = [e for e in self.eng if e.kind is ek and isinstance(e.attrs.get("cues"), Mapping)]
        truth = [t for t in self.truth if t.kind is ek]
        P = _pairing(pred, truth, self.cfg.tol_s, lambda p, t: _team_ok(p.team, t.team))
        for i, e in enumerate(pred):
            cues = e.attrs["cues"]
            if not self._in_window(e):
                continue
            if not self._covered(e.t):
                self._skip(kind, "no_truth_coverage")
            elif i in P.pair:
                self._row(kind, e, cues, True, "pbp", truth[P.pair[i]])
            elif (why := self._why_unpaired(e, P, truth, rebounds=ek is EventKind.REBOUND)) is not None:
                self._skip(kind, why)
            else:
                self._row(kind, e, cues, False, "pbp")

    def summary(self) -> dict[str, Any]:
        kinds = {}
        for k in self.cfg.kinds:
            st = self.stats.get(k, Counter())
            kinds[k] = {"rows": st["rows"], "positive": st["label_1"], "negative": st["label_0"],
                        "skipped": {key[5:]: n for key, n in sorted(st.items()) if key.startswith("skip:")},
                        "sources": {key[7:]: n for key, n in sorted(st.items()) if key.startswith("source:")}}
        truth = Counter(t.kind.value for t in self.truth if self.lo <= t.t <= self.hi)
        return {"game_id": self.game_id, "calibration_source": NBA_SOURCE,
                "window": [self.lo, None if math.isinf(self.hi) else self.hi], "kinds": kinds,
                "n_engine_events": sum(1 for e in self.eng if self._in_window(e)), "n_truth": dict(sorted(truth.items())),
                "nba_shots_linked": len(self.nba), "nba_espn_value_conflicts": self.value_conflicts,
                "scorebug": self.changes is not None, "config": {k: v for k, v in vars(self.cfg).items()}}


def label_game(events: Iterable[Event], aligned: Any, game_id: str, window: tuple[float, float] | None = None, *,
               reads: Sequence[ScoreboardReading] | None = None, timeline: ClockTimeline | None = None,
               nba_shots: Sequence[NBAShot] | None = None,
               config: NBACueConfig | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cue rows of one game (module docstring) and a summary of what was labelled and skipped.

    ``aligned``: the ``AlignedAction`` list of ``align_pbp`` (or its ``AlignmentResult``, whose timeline is used
    when ``timeline`` is None). ``reads``: the scorebug reads (coverage and the scorebug make labels; None skips
    both). ``nba_shots``: NBA.com shots with their teams (``load_shots``)."""
    if hasattr(aligned, "aligned"):
        timeline = timeline if timeline is not None else getattr(aligned, "timeline", None)
        aligned = aligned.aligned
    lab = _Labeller(events, list(aligned), game_id, window, reads, timeline, nba_shots, config or NBACueConfig())
    rows = lab.run()
    summary = lab.summary()
    log.info("cue rows %s: %s", game_id, {k: v["rows"] for k, v in summary["kinds"].items()})
    return rows, summary


# ---------------------------------------------------------------------------------------------- loading + CLI


def _nba_analyze() -> Any:
    """``tools/nba_analyze.py``, whose ``truth()`` is the ground truth ``eval`` scores against (imported by path, as
    tools/nba_diagnose.py imports it by name)."""
    mod = sys.modules.get("nba_analyze")
    if mod is not None and hasattr(mod, "truth"):
        return mod
    path = Path(__file__).resolve().parents[2] / "tools" / "nba_analyze.py"
    spec = importlib.util.spec_from_file_location("nba_analyze", path) if path.exists() else None
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"{path} not found: the ground truth comes from tools/nba_analyze.py truth()")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nba_analyze"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_game_run(root: str | Path, tag: str, file: str = "events.jsonl") -> GameRun:
    """The engine events of ``runs/<tag>/<file>``, the run window, the aligned truth (``nba_analyze.truth``), the
    scorebug reads and the NBA.com shots of a game folder."""
    from courtvision.nba.analyze import load_scorebug  # (imports cv2: kept out of the labelling core)

    root = Path(root)
    run_dir = root / "runs" / tag
    run = json.loads((run_dir / "run.json").read_text())
    events = [Event.from_dict(json.loads(line)) for line in (run_dir / file).read_text().splitlines() if line.strip()]
    _, result = _nba_analyze().truth(root)
    reads = load_scorebug(root / "scorebug.jsonl")
    info: dict[str, Any] = {"run": tag, "events_file": file, "alignment": result.summary()}
    shots: list[NBAShot] = []
    csv_path = root / "nba_pbp.csv"
    if csv_path.exists():
        codes = infer_nba_codes(csv_path, [al.action for al in result.aligned])
        if codes:
            home = next(c for c, t in codes.items() if t is Team.HOME)
            away = next(c for c, t in codes.items() if t is Team.AWAY)
            shots = load_shots(csv_path, home_code=home, away_code=away)
        else:
            log.warning("%s: its team codes match no ESPN shots; NBA.com locations not used", csv_path)
        info["nba_pbp"] = {"codes": {c: t.value for c, t in codes.items()}, "shots": len(shots)}
    return GameRun(root.resolve().name, events, list(result.aligned), run_window(run), reads, result.timeline, shots,
                   info)


def summary_path(out: str | Path) -> Path:
    """``<name>.summary.json`` next to the rows file ``<name>.jsonl``; a trailing ``.partial`` (a file renamed into
    place when the step succeeds, as the NBA loop writes) is ignored, so the summary has its final name."""
    out = Path(out)
    name = out.name.removesuffix(".partial").removesuffix(".jsonl")
    return out.with_name(name + ".summary.json")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; exit codes: 0 ok (also with 0 rows), 2 bad input (missing / unreadable files)."""
    ap = argparse.ArgumentParser(prog="python -m courtvision.training.nba_cue_rows",
                                 description=__doc__.split("\n\n")[0] if __doc__ else None)
    ap.add_argument("root", help="game folder (runs/nba/<game>)")
    ap.add_argument("--tag", required=True, help="the analysed run: <root>/runs/<tag>/")
    ap.add_argument("--file", default="events.jsonl", help="the run's engine events (not fused.jsonl)")
    ap.add_argument("--out", help="output JSONL (default: <root>/runs/<tag>/cue_rows.jsonl); the summary goes next to "
                                  "it as <name>.summary.json")
    ap.add_argument("--tol", type=float, default=2.5, help="pairing tolerance in s (the evaluation's)")
    args = ap.parse_args(argv)
    root = Path(args.root)
    try:
        run = load_game_run(root, args.tag, args.file)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.error("nba_cue_rows: %s", exc)
        return 2
    if any(e.attrs.get("fused") for e in run.events):
        log.warning("%s holds fused events (scorebug outcomes and teams); the engine's events.jsonl is meant", args.file)
    if not run.info["alignment"].get("accepted"):
        log.warning("alignment gate not passed (%s); rows are written, check report.json before fitting",
                    run.info["alignment"])
    rows, summary = label_game(run.events, run.aligned, run.game_id, run.window, reads=run.reads,
                               timeline=run.timeline, nba_shots=run.nba_shots, config=NBACueConfig(tol_s=args.tol))
    for r in rows:
        r["run"] = args.tag
    out = Path(args.out) if args.out else root / "runs" / args.tag / "cue_rows.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = write_cue_rows(rows, out)
    summary.update(run.info)
    side = summary_path(out)
    side.write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    per_kind = ", ".join(f"{k} {v['rows']} ({v['positive']} pos)" for k, v in summary["kinds"].items())
    print(f"{run.game_id} {args.tag}: {n} cue rows [{per_kind}] -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)  # (the alignment and the scorebug confirmer log every basket at INFO)
    sys.exit(main())
