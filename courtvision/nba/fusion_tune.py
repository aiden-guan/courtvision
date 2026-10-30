"""Calibrate the scorebug-fusion settings (``nba/fusion.py`` ``FusionConfig``) on games whose truth is known.

Research use (docs/NBA.md). ``fuse`` is a black box here: a setting is scored by running ``fuse`` on each game's
engine events and scorebug events (loaded once, ``GameCase``) and ``evaluate.compare`` against the game's aligned
play-by-play, as ``tools/nba_analyze.py eval`` and ``tools/nba_fusion_ab.py`` do. Nothing here knows the game
folder layout (``tools/nba_calibrate_fusion.py`` loads the games), so the search runs on synthetic games in tests.

**Objective** (lower is better), pooled over the games::

    box error + f1_weight * (1 - event F1)

* box error: sum |pred - truth| / sum truth over ``BOX_STATS`` and both teams, each game in its run's window, the
  sums taken over all games, so each game weighs by its truth total (for one game: the number ``nba_fusion_ab.py``
  reports, and ``nba_analyze.py eval`` for a whole game). It nets out within a team's totals: a missing and an
  extra field-goal attempt of the same team cancel in its FGA;
* event F1: 2 TP / (pred + truth) summed over the event kinds of ``compare`` and the games (``compare`` matches
  events in time only, not by team). ``f1_weight`` (``F1_WEIGHT``) is small, a tiebreak: with e.g. 200 box counts
  and 300 events (pred + truth) one box count weighs as much as 15 more matched events. The truth places a made shot
  at its bug update minus a fixed 2.0 s (``nba_analyze.ESPN_ALIGN``), so this term leans ``score_lag_s`` to 2.0.

**Search** (``search``), deterministic: coordinate descent from the defaults over ``space()`` (a grid per numeric
parameter in the order of the passes in ``fuse``, then the pass switches), at most ``max_rounds`` rounds or
``budget`` settings. Against overfitting, the defaults are preferred: a parameter moves only when the best value on
its grid beats its current value by ``margin`` (``MARGIN``, 0.5 % of the pooled box total: e.g. one count of 200,
four of 840); when several values tie for the best, the run of them nearest the default is taken and its middle
value (on an even run the middle one nearer the default). After the descent each change whose undoing costs less
than ``margin`` goes back to its default (the cheapest first), so every change left pays for itself. With two or
more games ``calibrate`` also holds each game out in turn: the search runs on the other games and its setting is
scored on the held-out one (``loo_guard``: keep the defaults when that does worse in total than the defaults). With
one game every number is in-sample.

**Search space** (``NUMERIC``; ``.lo`` / ``.hi``: the ends of a window field; the default is always on the grid):

======================  =======  ============  =====  ========================================================
parameter               default  range         step   why this range
======================  =======  ============  =====  ========================================================
score_lag_s             2.0 s    1.0 .. 3.5    0.25   the bug shows a basket after it goes in (ESPN about 2 s,
                                                      median of one game); the bug is read at 3 Hz
make_window_s.lo        -3.5 s   -6.0 .. -1.5  0.5    how early an engine result may be and still be the make
make_window_s.hi        1.5 s    0.5 .. 3.5    0.5    how late (the bug lag varies from basket to basket)
ft_window_s.lo / .hi    -4 / 2   -6..-1.5 /    0.5    the same for a free throw; the shots of a trip are several
                                 0.5..3.5             seconds apart (fuse spaces inferred ones 8 s apart)
and_one_window_s        90 s     30 .. 150     15     basket to the lone free throw, clock stopped (a review or
                                                      a timeout can come in between)
reset_window_s.lo       -3.0 s   -5.0 .. -1.0  0.5    engine miss before a shot-clock reset to 24 (rim touch)
reset_window_s.hi       1.0 s    0.0 .. 2.5    0.5    after it (a reset is timed up to one 3 Hz read late)
reset_turnover_s        2.5 s    1.0 .. 5.0    0.5    an engine turnover / steal this near explains the reset
reset_make_after_s      4.0 s    2.0 .. 6.0    0.5    a score this soon after the reset: the reset was the make
                                                      (at least the score lag, less than the next shot)
inferred_release_s      1.0 s    0.5 .. 2.0    0.25   release to rim of an unseen shot (~1 s jump-shot flight)
flight_match_s          1.5 s    0.5 .. 3.0    0.25   3-D flight rim pass vs the engine's missed-shot time
three_margin_ft         0.5 ft   -1.0 .. 2.0   0.25   release beyond the arc to count a three: the ball is ahead
                                                      of the feet; about a foot either way of the default
bug_rebound_window_s    8.0 s    4 .. 12       1.0    the shot clock answers a miss (14 / count from 24)
rebound_window_s        6.0 s    3 .. 10       1.0    the engine's rebound follows its miss within this
oreb_hold_s             2.0 s    1.0 .. 4.0    0.5    camera in the other half after a defensive rebound (a
                                                      transition shot can come 5 s later)
shot_deep_ft            17 ft    9 .. 29       2.0    camera aim past half court for a put-back: at least the
                                                      9 ft side margin, short of the rim (41.75 ft)
handler_window_s        2.5 s    1.0 .. 5.0    0.5    last ball-handler position before an unseen release
assist_window_s         3.5 s    1.5 .. 6.0    0.5    pass to shot: catch-and-shoot ~1 s, a few dribbles ~4 s
pass switches                    off / on             every boolean field of ``FusionConfig``
======================  =======  ============  =====  ========================================================
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from courtvision.types import Event, Team

from .evaluate import BOX_STATS, compare
from .fusion import FusionConfig, PlayDirection, fuse
from .scorebug_events import BugEvent

__all__ = ["BUDGET", "F1_WEIGHT", "MARGIN", "MAX_ROUNDS", "NUMERIC", "Evaluator", "GameCase", "GameScore", "Param",
           "Pooled", "SearchResult", "calibrate", "config_key", "describe_space", "fusion_config", "overrides",
           "resolve", "score_game", "search", "space"]

F1_WEIGHT = 0.05  # objective = box error + F1_WEIGHT * (1 - event F1)
MARGIN = 0.005  # a change must lower the pooled objective by this much (0.5 % of the pooled box total)
BUDGET = 600  # distinct settings one search may score (each: fuse + compare on every game of the search)
MAX_ROUNDS = 4
EPS = 1e-9  # objectives this close are equal


@dataclass(frozen=True)
class Param:
    """One searched coordinate: a numeric ``FusionConfig`` field (``<field>.lo`` / ``<field>.hi``: one end of a
    (lo, hi) window) on the grid lo, lo + step, ..., hi plus its default, or a pass switch (no ``step``)."""

    name: str
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    why: str = ""

    @property
    def field(self) -> str:
        return self.name.split(".", 1)[0]

    @property
    def end(self) -> int | None:
        """0 / 1 for the lo / hi end of a window field, None for a plain field."""
        return None if "." not in self.name else ("lo", "hi").index(self.name.split(".", 1)[1])

    @property
    def switch(self) -> bool:
        return self.step is None

    def default(self) -> Any:
        v = getattr(FusionConfig(), self.field)
        if self.end is not None:
            v = v[self.end]
        return bool(v) if self.switch else _r(v)

    def grid(self) -> list[Any]:
        if self.switch:
            return [False, True]
        assert self.lo is not None and self.hi is not None and self.step is not None
        n = round((self.hi - self.lo) / self.step)
        return sorted({_r(self.lo + k * self.step) for k in range(n + 1)} | {self.default()})


NUMERIC: tuple[Param, ...] = (
    # pass 1: makes (free throws included)
    Param("score_lag_s", 1.0, 3.5, 0.25, "the bug shows a basket after the ball goes in (ESPN: about 2 s, median of "
                                         "one game); the bug is read at 3 Hz"),
    Param("make_window_s.lo", -6.0, -1.5, 0.5, "how early an engine result may be (s before the bug time minus the "
                                               "lag) and still be the make"),
    Param("make_window_s.hi", 0.5, 3.5, 0.5, "how late an engine result may be and still be the make; the bug lag "
                                             "varies from basket to basket"),
    Param("ft_window_s.lo", -6.0, -1.5, 0.5, "the same for a made free throw; the shots of a trip are several seconds "
                                             "apart (fuse spaces inferred ones 8 s apart), a wide window reaches the "
                                             "next one"),
    Param("ft_window_s.hi", 0.5, 3.5, 0.5, "late end of the free-throw window"),
    Param("and_one_window_s", 30.0, 150.0, 15.0, "basket to the lone free throw of an and-one with the clock stopped: "
                                                 "half a minute to two (a review or a timeout in between)"),
    # pass 2: rim touches
    Param("reset_window_s.lo", -5.0, -1.0, 0.5, "an engine miss this long before a shot-clock reset to 24 is its rim "
                                                "touch; more than 5 s before it is another play"),
    Param("reset_window_s.hi", 0.0, 2.5, 0.5, "an engine miss this long after the reset (a reset is timed up to one "
                                              "3 Hz read late; the engine may call the miss at the rebound)"),
    Param("reset_turnover_s", 1.0, 5.0, 0.5, "an engine turnover / steal this near a reset explains it (a change of "
                                             "possession resets the shot clock too)"),
    Param("reset_make_after_s", 2.0, 6.0, 0.5, "a score change this soon after a reset means the reset was the make: "
                                               "at least the score lag, less than the time to the next shot"),
    Param("inferred_release_s", 0.5, 2.0, 0.25, "release to rim of a shot nobody saw: about 1 s for a jump shot, "
                                                "less for a layup, more for a heave"),
    # 3-D flight values
    Param("flight_match_s", 0.5, 3.0, 0.25, "a 3-D flight's rim pass this near an engine miss is that shot"),
    Param("three_margin_ft", -1.0, 2.0, 0.25, "a 3-D release point this far beyond the arc counts as a three (the "
                                              "ball is ahead of the feet); about a foot either way of the default"),
    # passes 4-5: rebounds
    Param("bug_rebound_window_s", 4.0, 12.0, 1.0, "the shot clock answers a miss (reset to 14, count from 24) within "
                                                  "this; a long scramble takes up to ~10 s"),
    Param("rebound_window_s", 3.0, 10.0, 1.0, "the engine's rebound follows its miss within this"),
    Param("oreb_hold_s", 1.0, 4.0, 0.5, "the camera stays in the other half this long after a defensive rebound "
                                        "(under 1 s: a look across; a transition shot can come 5 s later)"),
    Param("shot_deep_ft", 9.0, 29.0, 2.0, "camera aim past half court for a put-back attempt: at least the 9 ft side "
                                          "margin of PlayDirection, short of the rim (41.75 ft past half court)"),
    # pass 6: handler values
    Param("handler_window_s", 1.0, 5.0, 0.5, "the ball handler's last position at most this long before an unseen "
                                             "release (matters with handler_values on)"),
    # pass 3: assists
    Param("assist_window_s", 1.5, 6.0, 0.5, "pass to shot release: roughly 1 s for a catch-and-shoot, 4 s with a "
                                            "few dribbles"),
)


def _r(v: Any) -> Any:
    return v if isinstance(v, bool) else round(float(v), 6)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b) \
            and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a is b
    return _r(a) == _r(b)


def space() -> list[Param]:
    """The searchable coordinates of the current ``FusionConfig``: the ``NUMERIC`` entries whose field exists (a
    number, or a (lo, hi) pair for a ``.lo`` / ``.hi`` entry), then one switch per boolean field."""
    cfg = FusionConfig()
    out = []
    for p in NUMERIC:
        v = getattr(cfg, p.field, None)
        number = isinstance(v, (int, float)) and not isinstance(v, bool)
        if (p.end is None and number) or (p.end is not None and isinstance(v, tuple) and len(v) == 2):
            out.append(p)
    for f in dataclasses.fields(FusionConfig):
        if isinstance(getattr(cfg, f.name), bool):
            out.append(Param(f.name, why="pass switch (nba/fusion.py)"))
    return out


def resolve(names: Iterable[str] | None = None) -> list[Param]:
    """``space()`` restricted to ``names`` (a coordinate, or a window field for both of its ends), in its order."""
    full = space()
    wanted = [n.strip() for n in (names or []) if n.strip()]
    if not wanted:
        return full
    picked: set[str] = set()
    for n in wanted:
        hit = [p.name for p in full if n in (p.name, p.field)]
        if not hit:
            raise ValueError(f"unknown fusion parameter {n!r}; searchable: {', '.join(p.name for p in full)}")
        picked.update(hit)
    return [p for p in full if p.name in picked]


def describe_space(params: Sequence[Param]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in params:
        d: dict[str, Any] = {"default": p.default()}
        if p.switch:
            d["values"] = [False, True]
        else:
            d.update(range=[p.lo, p.hi], step=p.step)
        d["why"] = p.why
        out[p.name] = d
    return out


def overrides(point: Mapping[str, Any]) -> dict[str, Any]:
    """Coordinate values -> ``FusionConfig`` overrides: the fields that differ from their defaults (a window field
    rebuilt from its two ends)."""
    cfg = FusionConfig()
    fields: dict[str, Any] = {}
    for name, v in point.items():
        f, _, end = name.partition(".")
        if not hasattr(cfg, f):
            raise ValueError(f"FusionConfig has no field {f!r}")
        if end:
            cur = list(fields.get(f, getattr(cfg, f)))
            cur[("lo", "hi").index(end)] = _r(v)
            fields[f] = tuple(_r(x) for x in cur)
        else:
            fields[f] = _r(v)
    return {f: v for f, v in fields.items() if not _same(getattr(cfg, f), v)}


def fusion_config(over: Mapping[str, Any] | None = None) -> FusionConfig:
    """``FusionConfig`` with ``over`` applied: unknown names fail, JSON lists become the tuples of window fields."""
    names = {f.name for f in dataclasses.fields(FusionConfig)}
    unknown = sorted(set(over or {}) - names)
    if unknown:
        raise ValueError(f"unknown FusionConfig field(s): {', '.join(unknown)}")
    base = FusionConfig()
    kw = {k: tuple(v) if isinstance(getattr(base, k), tuple) and isinstance(v, (list, tuple)) else v
          for k, v in (over or {}).items()}
    return dataclasses.replace(base, **kw)


Key = tuple[tuple[str, Any], ...]


def config_key(over: Mapping[str, Any]) -> Key:
    """A hashable key of a setting (``overrides``, or the JSON ``config`` of a calibration)."""
    return tuple(sorted((k, tuple(_r(x) for x in v) if isinstance(v, (list, tuple)) else _r(v))
                        for k, v in over.items()))


def _jsonable(over: Mapping[str, Any]) -> dict[str, Any]:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in over.items()}


# --------------------------------------------------------------------------- scoring


@dataclass
class GameCase:
    """One game's inputs to ``fuse`` and its truth, loaded once (``tools/nba_calibrate_fusion.py`` ``load_case``)."""

    name: str
    events: list[Event]  # the event engine's events of the run (runs/<tag>/events.jsonl)
    bug: list[BugEvent]  # scorebug events within the run's window
    aligned: Sequence[Any]  # truth: the play-by-play on video time (align_pbp ``AlignedAction``)
    truth_box: Mapping[Team, Any] | None = None  # the official box (espn ``TeamBox``), scored when window is None
    window: tuple[float, float] | None = None  # video seconds of the run; None: the whole game
    direction: PlayDirection | None = None
    periods: list[tuple[float, int]] | None = None
    flights: list[Any] | None = None
    court: Any | None = None
    tol_s: float = 2.5  # event matching tolerance of compare (nba_analyze.py eval --tol)


@dataclass(frozen=True)
class GameScore:
    abs_error: int  # sum |pred - truth| over BOX_STATS and both teams
    truth_total: int  # sum truth over the same cells
    tp: float  # matched events, summed over the event kinds
    n_events: int  # predicted + truth events, summed over the event kinds
    diff: dict[str, int] = field(default_factory=dict)  # "home.fga" -> pred - truth
    f1: dict[str, float | None] = field(default_factory=dict)  # per event kind

    @property
    def box_error(self) -> float:
        return self.abs_error / max(1, self.truth_total)

    @property
    def event_f1(self) -> float | None:
        return 2.0 * self.tp / self.n_events if self.n_events else None

    @classmethod
    def from_report(cls, rep: Mapping[str, Any]) -> GameScore:
        """From ``evaluate.compare``'s report (a whole game's ``box``, or a window's ``box_window``)."""
        if "box" in rep:
            cells = {(r["team"], r["stat"]): (int(r["pred"]), int(r["truth"])) for r in rep["box"]["rows"]}
        else:
            w = rep["box_window"]
            cells = {(t, s): (int(w["pred"][t].get(s, 0)), int(w["truth"][t].get(s, 0)))
                     for t in (Team.HOME.value, Team.AWAY.value) for s in BOX_STATS}
        tp, n = 0.0, 0
        for r in rep["events"].values():
            m = int(r["n_pred"]) + int(r["n_truth"])
            n += m
            tp += (r["f1"] or 0.0) * m / 2.0  # f1 = 2 tp / (n_pred + n_truth)
        return cls(sum(abs(p - q) for p, q in cells.values()), sum(q for _, q in cells.values()), round(tp, 6), n,
                   {f"{t}.{s}": p - q for (t, s), (p, q) in sorted(cells.items())},
                   {k: r["f1"] for k, r in rep["events"].items()})


def score_game(case: GameCase, config: FusionConfig) -> GameScore:
    """``fuse`` with ``config``, then ``compare`` against the truth (the comparison of ``nba_analyze.py eval``)."""
    fused = fuse(case.events, case.bug, config, case.direction, case.periods, case.flights, case.court)
    rep = compare(fused, case.aligned, case.truth_box or {}, tol_s=case.tol_s, window=case.window)
    return GameScore.from_report(rep)


@dataclass(frozen=True)
class Pooled:
    abs_error: int
    truth_total: int
    tp: float
    n_events: int

    @classmethod
    def of(cls, scores: Iterable[GameScore | Pooled]) -> Pooled:
        s = list(scores)
        return cls(sum(x.abs_error for x in s), sum(x.truth_total for x in s), sum(x.tp for x in s),
                   sum(x.n_events for x in s))

    @property
    def box_error(self) -> float:
        return self.abs_error / max(1, self.truth_total)

    @property
    def event_f1(self) -> float | None:
        return 2.0 * self.tp / self.n_events if self.n_events else None

    def objective(self, f1_weight: float) -> float:
        return self.box_error + f1_weight * (1.0 - (self.event_f1 or 0.0))


_WORKER: dict[str, Any] = {}  # a pool worker's games and scorer (``Evaluator`` with workers > 1)


def _init_worker(cases: list[GameCase], scorer: Callable[[GameCase, FusionConfig], GameScore]) -> None:
    _WORKER.update(cases={c.name: c for c in cases}, scorer=scorer)


def _work(task: tuple[str, dict[str, Any]]) -> GameScore:
    name, over = task
    return _WORKER["scorer"](_WORKER["cases"][name], fusion_config(over))


class Evaluator:
    """Scores settings on the games; each (game, setting) runs ``scorer`` (``fuse`` + ``compare``) once (cached).
    With ``workers`` > 1, ``prefetch`` scores a batch in that many processes (each gets the games once; ``scorer``
    must then be a module-level function); the scores are the same as with one process."""

    def __init__(self, cases: Sequence[GameCase], f1_weight: float = F1_WEIGHT,
                 scorer: Callable[[GameCase, FusionConfig], GameScore] = score_game, workers: int = 1) -> None:
        self.cases = {c.name: c for c in cases}
        if len(self.cases) != len(cases):
            raise ValueError("game names must be unique")
        self.f1_weight = f1_weight
        self.scorer = scorer
        self.workers = max(1, int(workers))
        self.runs = 0  # scorer calls (cache misses)
        self.seconds = 0.0  # wall time spent in them
        self._cache: dict[tuple[str, Key], GameScore] = {}
        self._pool: Any = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def game(self, name: str, over: Mapping[str, Any]) -> GameScore:
        key = (name, config_key(over))
        s = self._cache.get(key)
        if s is None:
            t0 = time.perf_counter()
            s = self.scorer(self.cases[name], fusion_config(over))
            self.seconds += time.perf_counter() - t0
            self.runs += 1
            self._cache[key] = s
        return s

    def prefetch(self, names: Sequence[str], overs: Iterable[Mapping[str, Any]]) -> None:
        """Score the (game, setting) pairs not cached yet: in the worker processes when there are several."""
        todo: dict[tuple[str, Key], dict[str, Any]] = {}
        for over in overs:
            k = config_key(over)
            for n in names:
                if (n, k) not in self._cache:
                    todo.setdefault((n, k), dict(over))
        if self.workers == 1 or len(todo) < 2:
            for (n, _), over in todo.items():
                self.game(n, over)
            return
        if self._pool is None:
            import multiprocessing as mp

            self._pool = mp.get_context("spawn").Pool(self.workers, _init_worker,
                                                      (list(self.cases.values()), self.scorer))
        t0 = time.perf_counter()
        scores = self._pool.map(_work, [(n, over) for (n, _), over in todo.items()], chunksize=1)
        self.seconds += time.perf_counter() - t0
        self.runs += len(todo)
        self._cache.update(zip(todo, scores, strict=True))

    def pooled(self, names: Sequence[str], over: Mapping[str, Any]) -> Pooled:
        return Pooled.of(self.game(n, over) for n in names)

    def objective(self, names: Sequence[str], over: Mapping[str, Any]) -> float:
        return self.pooled(names, over).objective(self.f1_weight)


# --------------------------------------------------------------------------- search


@dataclass
class SearchResult:
    games: list[str]
    point: dict[str, Any]  # searched coordinate -> value
    overrides: dict[str, Any]  # the FusionConfig fields off their defaults
    objective: float
    default_objective: float
    evaluations: int  # distinct settings scored (each on every game of the search)
    rounds: int
    converged: bool  # the last round moved nothing
    budget_exhausted: bool
    moves: list[dict[str, Any]] = field(default_factory=list)
    reverted: list[dict[str, Any]] = field(default_factory=list)


def _pick(results: Mapping[Any, float], grid: Sequence[Any], default: Any) -> Any:
    """Among the grid values tied for the best objective: the run of consecutive ones nearest the default (then the
    longer run, then the lower), and its middle value (on an even run the middle one nearer the default)."""
    best = min(results.values())
    runs: list[list[Any]] = [[]]
    for v in grid:
        if v in results and results[v] <= best + EPS:
            runs[-1].append(v)
        elif runs[-1]:
            runs.append([])
    runs = [r for r in runs if r]
    d = float(default)
    run = min(runs, key=lambda r: (min(abs(float(v) - d) for v in r), -len(r), float(r[0])))
    middle = [run[(len(run) - 1) // 2], run[len(run) // 2]]
    return min(middle, key=lambda v: (abs(float(v) - d), float(v)))


def search(ev: Evaluator, games: Sequence[str], params: Sequence[Param], budget: int = BUDGET,
           margin: float = MARGIN, max_rounds: int = MAX_ROUNDS,
           log: Callable[[str], None] | None = None) -> SearchResult:
    """Coordinate descent from the defaults on the pooled objective of ``games``, then the undoing of changes worth
    less than ``margin`` (module docstring). ``budget`` caps the distinct settings of the descent (the undoing step
    may go a few beyond it)."""
    games = list(games)
    say = log or (lambda _msg: None)
    defaults = {p.name: p.default() for p in params}
    point = dict(defaults)
    seen: set[Key] = set()

    def admit(pt: Mapping[str, Any], free: bool = False) -> dict[str, Any] | None:
        """The overrides of ``pt``, counted against the budget when new (None: over the budget)."""
        over = overrides(pt)
        key = config_key(over)
        if key not in seen:
            if len(seen) >= budget and not free:
                return None
            seen.add(key)
        return over

    def free(pt: Mapping[str, Any]) -> dict[str, Any]:
        """The overrides of ``pt`` for the start and the undoing step, which the budget does not stop."""
        over = admit(pt, free=True)
        assert over is not None
        return over

    def objectives(overs: Sequence[Mapping[str, Any]]) -> list[float]:
        ev.prefetch(games, overs)  # one batch: in parallel with several workers
        return [ev.objective(games, o) for o in overs]

    start = objectives([free(point)])[0]
    current = start
    moves: list[dict[str, Any]] = []
    rounds, converged, exhausted = 0, False, False
    while rounds < max_rounds and not exhausted and not converged:
        rounds += 1
        moved = False
        for p in params:
            cur = point[p.name]
            grid = p.grid()
            batch: list[tuple[Any, dict[str, Any]]] = []
            for v in sorted(grid, key=lambda v: (abs(float(v) - float(cur)), float(v))):  # nearest first
                if v == cur:
                    continue
                over = admit({**point, p.name: v})
                if over is None:
                    exhausted = True
                    break
                batch.append((v, over))
            results: dict[Any, float] = {cur: current}
            results.update(zip([v for v, _ in batch], objectives([o for _, o in batch]), strict=True))
            gain = current - min(results.values())
            if gain > EPS and gain >= margin - EPS:  # the tied best values exclude ``cur``: a real move
                choice = _pick(results, grid, defaults[p.name])
                moves.append({"round": rounds, "param": p.name, "from": cur, "to": choice,
                              "objective": round(results[choice], 6), "gain": round(current - results[choice], 6)})
                say(f"  round {rounds}: {p.name} {cur} -> {choice} (objective {current:.4f} -> {results[choice]:.4f})")
                point[p.name] = choice
                current = results[choice]
                moved = True
            if exhausted:
                break
        converged = not moved and not exhausted
        say(f"  round {rounds} done: objective {current:.4f}, {len(seen)} settings scored"
            + (" (budget reached)" if exhausted else " (converged)" if converged else ""))
    # undo the changes that no longer pay for themselves, the cheapest first
    reverted: list[dict[str, Any]] = []
    while True:
        changed = [(i, p) for i, p in enumerate(params) if point[p.name] != defaults[p.name]]
        if not changed:
            break
        undone = objectives([free({**point, p.name: defaults[p.name]}) for _, p in changed])
        costs = [(o - current, i, p, o) for (i, p), o in zip(changed, undone, strict=True)]
        cost, _, p, o = min(costs, key=lambda c: (c[0], c[1]))
        if cost >= margin - EPS:
            break
        reverted.append({"param": p.name, "from": point[p.name], "to": defaults[p.name], "cost": round(cost, 6)})
        say(f"  undo {p.name} {point[p.name]} -> {defaults[p.name]} (costs {cost:+.4f}, under the margin {margin})")
        point[p.name] = defaults[p.name]
        current = o
    if point != defaults and start - current < margin - EPS:  # together the changes are not worth the margin
        reverted += [{"param": p.name, "from": point[p.name], "to": defaults[p.name], "cost": None,
                      "why": "all the changes together gain less than the margin"}
                     for p in params if point[p.name] != defaults[p.name]]
        point, current = dict(defaults), start
    return SearchResult(games, point, overrides(point), current, start, len(seen), rounds, converged, exhausted,
                        moves, reverted)


# --------------------------------------------------------------------------- calibration report


def _score_dict(s: GameScore | Pooled, f1_weight: float) -> dict[str, Any]:
    f1 = s.event_f1
    out: dict[str, Any] = {"box_error": round(s.box_error, 4), "abs_error": s.abs_error,
                           "truth_total": s.truth_total, "event_f1": None if f1 is None else round(f1, 4),
                           "objective": round(Pooled.of([s]).objective(f1_weight), 5)}
    if isinstance(s, GameScore):
        out["diff"] = s.diff
    return out


def _block(ev: Evaluator, names: Sequence[str], over: Mapping[str, Any]) -> dict[str, Any]:
    return {"per_game": {n: _score_dict(ev.game(n, over), ev.f1_weight) for n in names},
            "summed": _score_dict(ev.pooled(names, over), ev.f1_weight)}


def calibrate(cases: Sequence[GameCase], params: Iterable[str] | None = None, budget: int = BUDGET,
              margin: float = MARGIN, f1_weight: float = F1_WEIGHT, max_rounds: int = MAX_ROUNDS,
              loo_guard: bool = False, workers: int = 1,
              scorer: Callable[[GameCase, FusionConfig], GameScore] = score_game,
              log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Search on all the games, then (two or more games) hold each game out in turn. Returns the report of
    ``tools/nba_calibrate_fusion.py`` without its ``tag``: ``config`` (FusionConfig overrides), ``games``,
    ``objective``, ``before`` / ``after`` (the defaults / ``config``: per game and summed), ``leave_one_out`` (game ->
    its box error with the setting tuned on the other games; empty with one game), ``search`` and ``metrics`` (the
    headline numbers, flat).

    ``loo_guard``: keep the defaults when the held-out games together do worse with the settings tuned without them
    than with the defaults. ``workers``: processes that score settings (``Evaluator``)."""
    with Evaluator(cases, f1_weight, scorer, workers) as ev:
        return _calibrate(ev, resolve(params), budget, margin, max_rounds, loo_guard, log or (lambda _msg: None))


def _calibrate(ev: Evaluator, sp: list[Param], budget: int, margin: float, max_rounds: int, loo_guard: bool,
               say: Callable[[str], None]) -> dict[str, Any]:
    t0 = time.perf_counter()
    f1_weight = ev.f1_weight
    names = list(ev.cases)
    say(f"search on {len(names)} game(s): {', '.join(names)}")
    main = search(ev, names, sp, budget, margin, max_rounds, say)
    config = dict(main.overrides)
    loo: dict[str, Any] | None = None
    guard: dict[str, Any] = {"enabled": loo_guard, "applied": False}
    if len(names) >= 2:
        tuned: dict[str, dict[str, Any]] = {}
        per: dict[str, Any] = {}
        for g in names:
            rest = [n for n in names if n != g]
            say(f"hold out {g}: search on {', '.join(rest)}")
            res = search(ev, rest, sp, budget, margin, max_rounds, say)
            tuned[g] = res.overrides
            ev.prefetch([g], [res.overrides, {}])
            per[g] = {"config": _jsonable(res.overrides), "held_out": _score_dict(ev.game(g, res.overrides), f1_weight),
                      "default": _score_dict(ev.game(g, {}), f1_weight), "evaluations": res.evaluations,
                      "converged": res.converged, "budget_exhausted": res.budget_exhausted}
        held = Pooled.of(ev.game(g, tuned[g]) for g in names)
        base = ev.pooled(names, {})
        h, b = held.objective(f1_weight), base.objective(f1_weight)
        verdict = "better" if h < b - EPS else "worse" if h > b + EPS else "the same"
        loo = {"per_game": per,
               "summed": {"held_out": _score_dict(held, f1_weight), "default": _score_dict(base, f1_weight)},
               "verdict": f"with the settings tuned without them the held-out games did {verdict} "
                          f"{'as' if verdict == 'the same' else 'than'} with the defaults"}
        if loo_guard and verdict == "worse" and config:
            guard.update(applied=True, why="the held-out games did worse with the settings tuned without them")
            config = {}
    in_sample = len(names) == 1
    note = ("one game: the setting was tuned and scored on the same game, so 'after' is in-sample and says nothing "
            "about other games; no held-out check is possible" if in_sample else
            "'after' is in-sample (tuned on these games); 'leave_one_out' gives each game's box error with the "
            "setting tuned on the other games")
    ev.prefetch(names, [{}, config])
    before, after = ev.pooled(names, {}), ev.pooled(names, config)
    held_out = None if loo is None else loo["summed"]["held_out"]
    metrics = {"games": len(names), "in_sample": in_sample, "changed": sorted(config),
               "box_error_before": round(before.box_error, 4), "box_error_after": round(after.box_error, 4),
               "box_error_held_out": None if held_out is None else held_out["box_error"],
               "event_f1_before": _score_dict(before, f1_weight)["event_f1"],
               "event_f1_after": _score_dict(after, f1_weight)["event_f1"]}
    return {
        "config": _jsonable(config),
        "games": names,
        "objective": {"formula": "pooled box error + f1_weight * (1 - pooled event F1)",
                      "box_error": "sum |pred - truth| / sum truth over BOX_STATS, both teams and all games, each "
                                   "game in its run's window",
                      "event_f1": "2 TP / (pred + truth) summed over the event kinds of evaluate.compare and the games",
                      "f1_weight": f1_weight, "margin": margin,
                      "before": round(before.objective(f1_weight), 5), "after": round(after.objective(f1_weight), 5)},
        "before": _block(ev, names, {}),
        "after": _block(ev, names, config),
        "leave_one_out": {} if loo is None else {g: loo["per_game"][g]["held_out"]["box_error"] for g in names},
        "search": {
            "method": "coordinate descent over a grid per parameter, then the pass switches; a move needs a gain of "
                      "at least the margin; tied values take the middle of the tied run nearest the default; "
                      "changes worth less than the margin are undone",
            "in_sample": in_sample, "note": note,
            "params": [p.name for p in sp], "space": describe_space(sp),
            "unsearched": sorted({f.name for f in dataclasses.fields(FusionConfig)} - {p.field for p in space()}),
            "budget": budget, "max_rounds": max_rounds,
            "evaluations": main.evaluations, "rounds": main.rounds, "converged": main.converged,
            "budget_exhausted": main.budget_exhausted, "moves": main.moves, "reverted": main.reverted,
            "tuned_config": _jsonable(main.overrides), "loo_guard": guard, "leave_one_out": loo,
            "defaults": _jsonable(dataclasses.asdict(FusionConfig())),
            "workers": ev.workers, "scored": ev.runs, "scoring_seconds": round(ev.seconds, 2),
            "seconds": round(time.perf_counter() - t0, 2),
        },
        "metrics": metrics,  # the headline numbers in one flat place (before / after summed, held out)
    }
