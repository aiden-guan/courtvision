"""Team possession timeline: one decision over the whole game instead of a vote per frame.

Design: docs/research/possession_decision_model.md §3.1. The event engine decides who holds the ball frame by frame
(``events/possession.py``), and every wrong flip of team becomes a steal, a turnover or a rebound. Here the team
with the ball is a hidden state on a fixed time grid, decided once for the whole game by Viterbi (the most likely
sequence of states given all the evidence).

The state is the team with the ball and the half the ball is in: the team's front court (the half it attacks) or its
back court, with the time spent there. A team that gains the ball usually does so in its back court (after a basket
or a defensive rebound at the other end), then crosses half court: its stay there is gamma distributed
(``back_mean_s``, ``back_sd_s``) and ends within 8 seconds of live play by rule (``max_back_s`` allows for the
inbound too; the count pauses while the camera is off the court, at a dead ball). The evidence:

* **The camera's aim** (``PlayDirection``): the broadcast camera follows the ball, so the half it aims into (``lead_s``
  later: it follows a little behind) is the half the ball is in. Home in its front court and away in its back court
  both put the ball in the half home attacks; the transitions and the back-court time limit tell them apart. Each
  team attacks one basket per half of the game; which one, and when they swap, is voted from the engine's own team
  cues against the camera (``attack_map``), so decoding reads nothing from the scorebug.
* **The engine's team cues** (dribbles, passes, possession gains), weighted by ``engine_weight``.
* **Moments where the ball can change hands** (``candidate_times``: the engine's shot results, rebounds, steals and
  turnovers, and 3-D flights that come down at a rim): the ball changes hands with probability ``switch[kind]`` at a
  step near one of that kind and ``switch_base`` at any other step. Crossing half court needs no such moment.

So after a made basket the ball changes hands at the basket, and the other team stays in its back court, where the
camera still is, until it brings the ball up. ``fit`` estimates the probabilities on aligned play-by-play
(``truth_marks``) and picks the rest there; ``derive_events`` rebuilds the possession-dependent events from a
decoded ``Timeline``.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import gamma as gamma_dist

from courtvision.eval.events import latest_revisions
from courtvision.types import Event, EventKind, EventStatus, Team

__all__ = ["BACK", "FRONT", "AttackMap", "DeriveConfig", "Timeline", "TimelineConfig", "attack_map", "candidate_times",
           "decode", "derive_events", "fit", "from_scorebug", "grid_for", "rim_arrivals", "rim_shots", "score",
           "settled_mask", "truth_marks", "truth_teams", "viterbi"]

K = EventKind
TEAMS = (Team.HOME, Team.AWAY)  # state 0, state 1
OTHER = {Team.HOME: Team.AWAY, Team.AWAY: Team.HOME}
G_FTS2 = 32.174
TEAM_CUES = (K.DRIBBLE, K.PASS, K.POSSESSION)
RESULTS = (K.SHOT_MADE, K.SHOT_MISSED)
# candidate moments by kind: the engine events of each (and "rim", 3-D flights reaching a rim)
CANDIDATES: dict[str, tuple[EventKind, ...]] = {"result": RESULTS, "rebound": (K.REBOUND,),
                                                "loss": (K.STEAL, K.TURNOVER), "gain": (K.POSSESSION,)}


@dataclass
class TimelineConfig:
    """Decoder settings. ``fit`` sets the probabilities, ``lead_s``, ``cam_weight``, ``engine_weight`` and
    ``back_mean_s`` from aligned play-by-play; the defaults are its fit on the whole of Lakers @ Warriors 2024-12-25,
    for games without play-by-play."""

    dt_s: float = 0.25  # grid step
    lead_s: float = 1.0  # the camera evidence for time t is its aim at t + lead_s (it follows the ball a little behind)
    cam_p: float = 0.98  # P(the aim is in the half the ball is in)
    cam_weight: float = 0.3  # weight of one step's camera term: consecutive steps are far from independent
    margin_ft: float = 9.0  # the aim must be this far into a half to count (as PlayDirection.side)
    engine_p: float = 0.73  # P(an engine team cue names the team with the ball)
    engine_weight: float = 0.0  # weight of one engine team cue (fitted 0: the camera says it better)
    switch: dict[str, float] = dataclasses.field(default_factory=lambda: {
        "result": 0.03, "rebound": 0.04, "rim": 0.016, "loss": 0.03})  # P(the ball changes hands at a grid step near
    # a candidate moment of that kind; the most likely kind near the step counts). Kinds left out are not used: the
    # engine's possession gains ("gain") come when its holder logic sees the new team, seconds after the basket or
    # rebound, and moved changes there (measured: fewer changes matched)
    switch_base: float = 0.0022  # P(... at any other grid step)
    back_mean_s: float = 2.0  # a stay in the back court lasts this long on average (gamma distributed) ...
    back_sd_s: float = 1.6  # ... with this spread
    max_back_s: float = 12.0  # ... and at most this long in live play: 8 s by rule (NBA, FIBA) after the inbound
    paused_cross: float = 0.01  # P(crossing half court at a step with the camera off the court: the count pauses)
    retreat: float = 1e-4  # P(a team in its front court goes back over half court and keeps the ball)
    jump: float = 0.05  # a change of hands that puts the ball at once in the other half: this share of the change
    cand_window_s: float = 0.75  # a grid step is near a candidate moment within this
    rim_tol_ft: float = 3.0  # a 3-D flight coming down through rim height this near a rim centre reached the rim

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- where each team attacks


def _flip(side: str) -> str:
    return "right" if side == "left" else "left"


@dataclass(frozen=True)
class AttackMap:
    """The side each team attacks: home attacks ``home_first`` before ``switch_t`` (the teams swap baskets once, at
    half time) and the other side after it; no swap seen: ``switch_t`` is +inf."""

    home_first: str
    switch_t: float = math.inf

    def side(self, team: Team, t: float) -> str:
        home = self.home_first if t < self.switch_t else _flip(self.home_first)
        return home if team is Team.HOME else _flip(home)

    def to_dict(self) -> dict[str, Any]:
        return {"home_first": self.home_first, "switch_t": None if math.isinf(self.switch_t) else self.switch_t}


def _nearest(sorted_ts: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Index of the nearest of ``sorted_ts`` (not empty) for each of ``ts``."""
    if len(sorted_ts) == 1:
        return np.zeros(len(ts), int)
    j = np.clip(np.searchsorted(sorted_ts, ts), 1, len(sorted_ts) - 1)
    return np.where(np.abs(sorted_ts[j - 1] - ts) <= np.abs(sorted_ts[j] - ts), j - 1, j)


def _aim_sides(direction: Any, ts: np.ndarray, margin_ft: float, max_gap_s: float = 1.0) -> np.ndarray:
    """Per time: -1 the camera aims into the left half, +1 the right, 0 the middle or no camera (vectorised
    ``PlayDirection.side``)."""
    dt, dx = np.asarray(direction.t, float), np.asarray(direction.x, float)
    out = np.zeros(len(ts), np.int8)
    if not len(dt):
        return out
    j = _nearest(dt, ts)
    ok = np.abs(dt[j] - ts) <= max_gap_s
    x = dx[j]
    out[ok & (x > direction.mid + margin_ft)] = 1
    out[ok & (x < direction.mid - margin_ft)] = -1
    return out


def attack_map(events: Iterable[Event], direction: Any, lead_s: float = 3.0, margin_ft: float = 9.0) -> AttackMap:
    """Vote from the engine's team cues against the camera ``lead_s`` later (by then a team that just gained the ball
    has mostly crossed half court): a cue of team X with the aim then in side S says X attacks S (the engine's team
    labels are often wrong, the majority is not). The one swap (half
    time) is put where the votes before and after it agree best with opposite maps, and kept only when that
    explains clearly more votes than no swap."""
    cues = [e for e in _seen(events) if e.kind in TEAM_CUES and e.team is not None]
    ts = np.array([e.t for e in cues], float)
    sides = _aim_sides(direction, ts + lead_s, margin_ft) if len(ts) else np.zeros(0, np.int8)
    keep = sides != 0
    if not keep.any():
        return AttackMap("left")
    ts, sides = ts[keep], sides[keep]
    home = np.array([e.team is Team.HOME for e, k in zip(cues, keep, strict=True) if k])
    votes = np.where(home == (sides < 0), 1, -1)  # +1: "home attacks left"
    cs = np.concatenate([[0], np.cumsum(votes)])
    total = int(cs[-1])
    split = np.abs(2 * cs - total)
    k = int(np.argmax(split))
    if 0 < k < len(votes) and split[k] > abs(total) + max(10.0, 0.1 * len(votes)):
        return AttackMap("left" if 2 * cs[k] - total > 0 else "right", float(ts[k - 1] + ts[k]) / 2.0)
    return AttackMap("left" if total >= 0 else "right")


# --------------------------------------------------------------------------- candidate moments


def rim_shots(flights: Iterable[Any], court: Any, tol_ft: float = 3.0) -> list[tuple[float, str]]:
    """(time, rim side) at which a 3-D flight comes down through rim height within ``tol_ft`` of a rim centre (shots
    that reached the rim, made or missed), at most a second past the flight's last observation."""
    out = []
    for f in flights:
        p0, v0 = np.asarray(f.p0, float), np.asarray(f.v0, float)
        for side in ("left", "right"):
            rx, ry = court.basket_xy(side)
            disc = v0[2] ** 2 + 2.0 * G_FTS2 * (p0[2] - court.rim_height)
            if disc < 0:
                continue
            tc = (v0[2] + math.sqrt(disc)) / G_FTS2  # the descending crossing of rim height
            if tc > 0 and f.t0 + tc <= f.t1 + 1.0 and math.hypot(p0[0] + v0[0] * tc - rx, p0[1] + v0[1] * tc - ry) <= tol_ft:
                out.append((float(f.t0 + tc), side))
                break
    return sorted(out)


def rim_arrivals(flights: Iterable[Any], court: Any, tol_ft: float = 3.0) -> list[float]:
    """The times of ``rim_shots``."""
    return [t for t, _ in rim_shots(flights, court, tol_ft)]


def candidate_times(events: Iterable[Event], flights: Iterable[Any] = (), court: Any = None,
                    rim_tol_ft: float = 3.0) -> dict[str, np.ndarray]:
    """Moments where the ball can change hands, by kind (``CANDIDATES``): the engine's shot results, rebounds,
    steals and turnovers ("loss") and possession gains, and ("rim") 3-D flights reaching a rim (with ``court``)."""
    evs = list(events)
    out = {k: np.array(sorted(e.t for e in evs if e.kind in kinds), float) for k, kinds in CANDIDATES.items()}
    out["rim"] = np.array(rim_arrivals(flights, court, rim_tol_ft) if court is not None else [], float)
    return out


# --------------------------------------------------------------------------- decoding


FRONT, BACK = 0, 1


@dataclass
class Timeline:
    """The decoded team with the ball on a time grid: ``state`` indexes ``TEAMS``; ``phase`` is ``FRONT`` (the ball
    in the half that team attacks) or ``BACK`` (its own half); ``tracked``: the camera was on the court at the step
    (None: unknown)."""

    t: np.ndarray
    state: np.ndarray
    attack: AttackMap
    tracked: np.ndarray | None = None
    phase: np.ndarray | None = None

    def teams(self) -> list[Team]:
        return [TEAMS[int(k)] for k in self.state]

    def tracked_share(self, t0: float, t1: float) -> float:
        """Share of the steps in [t0, t1] with the camera on the court (1.0 when unknown or empty)."""
        if self.tracked is None:
            return 1.0
        sel = (self.t >= t0) & (self.t <= t1)
        return float(self.tracked[sel].mean()) if sel.any() else 1.0

    def team_at(self, t: float) -> Team | None:
        if not len(self.t) or t < self.t[0] - 1e-9:
            return None
        i = min(int(np.searchsorted(self.t, t, side="right")) - 1, len(self.t) - 1)
        return TEAMS[int(self.state[i])]

    def changes(self) -> list[tuple[float, Team, Team]]:
        """(time, team that had the ball, team that has it): each time is the first grid step of the new team."""
        idx = np.flatnonzero(np.diff(self.state)) + 1
        return [(float(self.t[i]), TEAMS[int(self.state[i - 1])], TEAMS[int(self.state[i])]) for i in idx]

    def segments(self) -> list[tuple[float, float, Team]]:
        if not len(self.t):
            return []
        cuts = [0, *(np.flatnonzero(np.diff(self.state)) + 1).tolist(), len(self.t)]
        return [(float(self.t[a]), float(self.t[b - 1]), TEAMS[int(self.state[a])]) for a, b in itertools.pairwise(cuts)]

    def back_court(self) -> list[tuple[float, float]]:
        """The stretches in which the team with the ball is still in its own back court (bringing it up)."""
        if self.phase is None or not len(self.t):
            return []
        back = np.asarray(self.phase) == BACK
        edges = np.flatnonzero(np.diff(np.concatenate([[0], back.astype(np.int8), [0]])))
        return [(float(self.t[a]), float(self.t[b - 1])) for a, b in zip(edges[::2], edges[1::2], strict=True)]

    def to_dict(self) -> dict[str, Any]:
        return {"attack": self.attack.to_dict(), "dt_s": float(self.t[1] - self.t[0]) if len(self.t) > 1 else None,
                "segments": [[round(a, 2), round(b, 2), team.value] for a, b, team in self.segments()],
                "back_court": [[round(a, 2), round(b, 2)] for a, b in self.back_court()]}


def _back_steps(cfg: TimelineConfig) -> int:
    return max(1, round(cfg.max_back_s / cfg.dt_s))


def _cross_hazards(cfg: TimelineConfig) -> np.ndarray:
    """Per back-court step k = 1 .. ``_back_steps``: P(crossing half court at step k | still in the back court) for a
    gamma-distributed stay cut at ``max_back_s`` (the last step must cross)."""
    m = _back_steps(cfg)
    shape, scale = (cfg.back_mean_s / cfg.back_sd_s) ** 2, cfg.back_sd_s ** 2 / cfg.back_mean_s
    cdf = gamma_dist.cdf(np.arange(m + 1) * cfg.dt_s, shape, scale=scale)
    hazard = np.clip(np.diff(cdf) / np.maximum(1.0 - cdf[:-1], 1e-12), 1e-6, 1.0)
    hazard[-1] = 1.0
    return hazard


def _log_transitions(cfg: TimelineConfig, change: float, live: bool = True) -> np.ndarray:
    """(S, S) log transition probabilities from row to column, ``change`` being the probability that the ball
    changes hands at the step. Each team's block of states is its front court, then its back court after 1, 2, ...
    ``_back_steps`` steps there. ``live`` False (the camera off the court, a dead ball): the back-court count stands
    still."""
    m = _back_steps(cfg)
    hazard = _cross_hazards(cfg)
    lose = change * (1.0 + cfg.jump)
    tr = np.zeros((2 * (m + 1), 2 * (m + 1)))
    for team in (0, 1):
        front, o_front = team * (m + 1), (1 - team) * (m + 1)
        tr[front, o_front + 1] = change  # lost in its front court: the other team has it in its own back court
        tr[front, o_front] = change * cfg.jump
        tr[front, front + 1] = cfg.retreat
        tr[front, front] = 1.0 - lose - cfg.retreat
        for k in range(1, m + 1):
            back = front + k
            tr[back, o_front] = change  # lost in its back court: the other team has it in its front court
            tr[back, o_front + 1] = change * cfg.jump
            cross = (1.0 - lose) * (hazard[k - 1] if live else cfg.paused_cross)
            tr[back, front] = cross
            tr[back, back + 1 if live and k < m else back] = 1.0 - lose - cross
    with np.errstate(divide="ignore"):
        return np.log(tr)


def viterbi(loglik: np.ndarray, log_trans: Sequence[np.ndarray], which: np.ndarray) -> np.ndarray:
    """The most likely state path: ``loglik`` (n, S) per step and state; the move into step i uses the (S, S) log
    transition matrix ``log_trans[which[i]]`` (from row to column)."""
    n, n_states = loglik.shape
    if n == 0:
        return np.zeros(0, np.int8)
    back = np.zeros((n, n_states), np.int16)
    cols = np.arange(n_states)
    v = loglik[0].astype(float)
    for i in range(1, n):
        m = v[:, None] + log_trans[which[i]]
        best = np.argmax(m, axis=0)
        back[i] = best
        v = m[best, cols] + loglik[i]
    path = np.zeros(n, np.int16)
    path[-1] = int(np.argmax(v))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def grid_for(events: Sequence[Event], direction: Any, dt_s: float) -> np.ndarray:
    """The decoding grid: every ``dt_s`` from the first to the last camera frame or event."""
    ts = [float(direction.t[0]), float(direction.t[-1])] if len(direction.t) else []
    ts += [e.t for e in events]
    if not ts:
        return np.zeros(0)
    return np.arange(min(ts), max(ts) + dt_s, dt_s)


def _home_sides(attack: AttackMap, grid: np.ndarray) -> np.ndarray:
    """Per step: +1 when home attacks the right half, -1 the left."""
    return np.where(grid < attack.switch_t, 1, -1) * (1 if attack.home_first == "right" else -1)


def _evidence(grid: np.ndarray, events: Sequence[Event], direction: Any, attack: AttackMap,
              cfg: TimelineConfig) -> np.ndarray:
    """(n, S) log evidence: the camera's half against the half the ball is in for each state (home in its front court
    and away in its back court both put it in the half home attacks), and the engine's team cues against each
    state's team."""
    m = _back_steps(cfg)
    sides = _aim_sides(direction, grid + cfg.lead_s, cfg.margin_ft)
    sign = np.concatenate([[1], -np.ones(m), [-1], np.ones(m)])  # the ball's half per state, as a multiple of home's
    ball = _home_sides(attack, grid)[:, None] * sign[None, :]
    yes, no = cfg.cam_weight * math.log(cfg.cam_p), cfg.cam_weight * math.log(1.0 - cfg.cam_p)
    ll = np.where((sides != 0)[:, None], np.where(ball == sides[:, None], yes, no), 0.0)
    if cfg.engine_weight > 0 and len(grid):
        y, n = cfg.engine_weight * math.log(cfg.engine_p), cfg.engine_weight * math.log(1.0 - cfg.engine_p)
        for e in events:
            if e.kind in TEAM_CUES and e.team is not None:
                i = int(round((e.t - grid[0]) / cfg.dt_s))
                if 0 <= i < len(grid):
                    k = TEAMS.index(e.team)
                    ll[i, k * (m + 1):(k + 1) * (m + 1)] += y
                    ll[i, (1 - k) * (m + 1):(2 - k) * (m + 1)] += n
    return ll


def _near(grid: np.ndarray, times: np.ndarray, window_s: float) -> np.ndarray:
    """Per grid step: is one of ``times`` within ``window_s``?"""
    if not len(times) or not len(grid):
        return np.zeros(len(grid), bool)
    return np.abs(times[_nearest(times, grid)] - grid) <= window_s


def _change_probability(grid: np.ndarray, cands: dict[str, np.ndarray], cfg: TimelineConfig) -> np.ndarray:
    """Per grid step: the probability that the ball changes hands there (the most likely candidate kind within
    ``cand_window_s``, else ``switch_base``)."""
    out = np.full(len(grid), cfg.switch_base)
    for kind, times in cands.items():
        p = cfg.switch.get(kind)
        if p is not None:
            out = np.where(_near(grid, times, cfg.cand_window_s), np.maximum(out, p), out)
    return out


def _live(events: Iterable[Event]) -> list[Event]:
    return sorted((e for e in latest_revisions(events) if e.status is not EventStatus.RETRACTED), key=lambda e: e.t)


def from_scorebug(e: Event) -> bool:
    """An event the engine took from the scorebug (score corrections, period edges) or inferred from it (makes and
    their attempts: ``attrs["inferred"] == "scoreboard"``)."""
    return e.source == "scoreboard" or e.attrs.get("inferred") == "scoreboard"


def _seen(events: Iterable[Event]) -> list[Event]:
    """The events the camera saw: nothing the engine took from the scorebug (``from_scorebug``)."""
    return [e for e in _live(events) if not from_scorebug(e)]


def decode(events: Iterable[Event], direction: Any, config: TimelineConfig | None = None, flights: Iterable[Any] = (),
           court: Any = None, attack: AttackMap | None = None, grid: np.ndarray | None = None) -> Timeline:
    """The team with the ball over the game from the engine's events, the camera's aim (``PlayDirection``) and the
    3-D flights (with ``court``). ``attack``: where each team attacks (default: ``attack_map`` of the events)."""
    cfg = config or TimelineConfig()
    evs = _seen(events)
    attack = attack or attack_map(evs, direction, margin_ft=cfg.margin_ft)
    grid = grid_for(evs, direction, cfg.dt_s) if grid is None else grid
    ll = _evidence(grid, evs, direction, attack, cfg)
    change = _change_probability(grid, candidate_times(evs, flights, court, cfg.rim_tol_ft), cfg)
    tracked = _near(grid, np.asarray(direction.t, float), 1.0)
    pairs, which = np.unique(np.stack([change, tracked.astype(float)], axis=1), axis=0, return_inverse=True)
    trans = [_log_transitions(cfg, float(c), bool(live)) for c, live in pairs]
    path = viterbi(ll, trans, which.ravel()).astype(int)
    block = _back_steps(cfg) + 1
    return Timeline(grid, (path // block).astype(np.int8), attack, tracked,
                    np.where(path % block == 0, FRONT, BACK).astype(np.int8))


# --------------------------------------------------------------------------- truth, scoring and fitting


def truth_marks(aligned: Iterable[Any]) -> list[tuple[float, Team | None]]:
    """The team with the ball from aligned play-by-play (``AlignedAction``): (time, team from then on), None while a
    missed last shot is loose. A shot's team had the ball up to it; a made basket (or made last free throw) gives it
    to the other team, a rebound to the rebounder's team, a turnover to the other team, a steal to the stealer's.
    Technical free throws change nothing."""
    marks: list[tuple[float, Team | None]] = []
    for al in sorted((al for al in aligned if al.t is not None and al.action.kind is not None
                      and not al.action.synthetic and al.action.team is not None), key=lambda al: al.t):
        a, t = al.action, float(al.t)
        if a.kind in RESULTS:
            if a.is_free_throw and a.attrs.get("technical"):
                continue
            last = not a.is_free_throw or int(a.attrs.get("ft_index") or 1) >= int(a.attrs.get("ft_total") or 1)
            marks.append((t - 1e-3, a.team))
            if last:
                marks.append((t, OTHER[a.team] if a.made else None))
        elif a.kind in (K.REBOUND, K.STEAL, K.POSSESSION):
            marks.append((t, a.team))
        elif a.kind is K.TURNOVER:
            marks.append((t, OTHER[a.team]))
    return marks


def reference(aligned: Iterable[Any]) -> dict[str, list[list[Any]]]:
    """The play-by-play to show beside a timeline (JSON-ready; ``timeline.json``'s ``reference``): ``marks``, the team
    with the ball from each time on (``truth_marks``; None while loose), and ``plays``, each action's time and text
    (once: a shot and its assist share one)."""
    aligned = list(aligned)
    return {"marks": [[round(t, 3), team.value if team is not None else None] for t, team in truth_marks(aligned)],
            "plays": [list(p) for p in sorted({(round(float(al.t), 3), al.action.description) for al in aligned
                                               if al.t is not None and al.action.description and not al.action.synthetic})]}


def truth_teams(marks: Sequence[tuple[float, Team | None]], ts: np.ndarray) -> list[Team | None]:
    mt = np.array([m[0] for m in marks], float)
    idx = np.searchsorted(mt, ts, side="right") - 1
    return [marks[i][1] if i >= 0 else None for i in idx]


def _truth_changes(marks: Sequence[tuple[float, Team | None]]) -> list[tuple[float, Team]]:
    out: list[tuple[float, Team]] = []
    cur: Team | None = None
    for t, team in marks:
        if team is not None and team is not cur:
            if cur is not None:
                out.append((t, team))
            cur = team
    return out


def settled_mask(ts: np.ndarray, marks: Sequence[tuple[float, Team | None]], radius_s: float = 2.0) -> np.ndarray:
    """Grid steps whose true team is known and the same ``radius_s`` either side (away from changes)."""
    base = truth_teams(marks, ts)
    out = np.array([b is not None for b in base])
    for d in (-radius_s, -radius_s / 2, radius_s / 2, radius_s):
        other = truth_teams(marks, ts + d)
        out &= np.array([a is b for a, b in zip(base, other, strict=True)])
    return out


def _match(pred: Sequence[float], truth: Sequence[float], tol_s: float) -> int:
    """Greedy one-to-one pairs within ``tol_s``, nearest first."""
    pairs = sorted((abs(p - q), i, j) for i, p in enumerate(pred) for j, q in enumerate(truth) if abs(p - q) <= tol_s)
    used_p: set[int] = set()
    used_t: set[int] = set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
    return len(used_p)


def score(teams: Sequence[Team | None], ts: np.ndarray, marks: Sequence[tuple[float, Team | None]], select: np.ndarray,
          tol_s: float = 2.5) -> dict[str, Any]:
    """A team-per-step sequence (a decoded timeline, or the engine's) against the play-by-play on the ``select``ed
    steps: the share of settled steps with the right team, and its changes of team matched to the true ones
    (same new team, within ``tol_s``)."""
    truth = truth_teams(marks, ts)
    settled = settled_mask(ts, marks) & select
    right = sum(teams[i] is truth[i] for i in np.flatnonzero(settled))
    pred_ch: dict[Team, list[float]] = {t: [] for t in TEAMS}
    for i in range(1, len(ts)):
        if select[i] and teams[i] is not None and teams[i - 1] is not None and teams[i] is not teams[i - 1]:
            pred_ch[teams[i]].append(float(ts[i]))  # type: ignore[index]
    lo, hi = (float(ts[np.flatnonzero(select)[0]]), float(ts[np.flatnonzero(select)[-1]])) if select.any() else (0, -1)
    true_ch: dict[Team, list[float]] = {t: [] for t in TEAMS}
    for t, team in _truth_changes(marks):
        if lo <= t <= hi:
            true_ch[team].append(t)
    tp = sum(_match(pred_ch[t], true_ch[t], tol_s) for t in TEAMS)
    n_pred, n_true = sum(map(len, pred_ch.values())), sum(map(len, true_ch.values()))
    return {"settled_steps": int(settled.sum()), "accuracy": round(right / max(1, int(settled.sum())), 4),
            "changes_pred": n_pred, "changes_true": n_true, "changes_matched": tp,
            "precision": round(tp / max(1, n_pred), 4), "recall": round(tp / max(1, n_true), 4),
            "f1": round(2 * tp / max(1, n_pred + n_true), 4)}


def fit(events: Iterable[Event], direction: Any, marks: Sequence[tuple[float, Team | None]], grid: np.ndarray,
        select: np.ndarray, live: np.ndarray, config: TimelineConfig | None = None, flights: Iterable[Any] = (),
        court: Any = None, leads: Sequence[float] = (1.0, 1.5), back_means: Sequence[float] = (2.0, 3.5, 5.0),
        cam_weights: Sequence[float] = (0.15, 0.3), base_scales: Sequence[float] = (1.0, 0.3),
        engine_weights: Sequence[float] = (0.0, 0.1), attack: AttackMap | None = None,
        front_after_s: float = 8.0) -> TimelineConfig:
    """Fit on the ``select``ed steps of ``grid`` (``grid_for``; ``live``: the steps in live play, the clock running).
    Counted: ``cam_p`` on live steps at least ``front_after_s`` into a possession (the ball in that team's front
    court), ``engine_p`` on the engine's team cues, the switch rates as true changes per step near / not near a
    candidate moment. Picked: ``lead_s``, ``back_mean_s`` (``back_sd_s`` 0.8 of it), ``cam_weight``,
    ``engine_weight`` and a scale of ``switch_base`` (the play-by-play's times are a second or two off, so the count
    overstates changes away from a candidate), by decoding each combination and keeping the best change F1 plus
    settled accuracy on the selected live steps."""
    base = config or TimelineConfig()
    evs = _seen(events)
    truth = truth_teams(marks, grid)
    changes = np.array([t for t, _ in _truth_changes(marks)], float)
    since = grid - changes[np.clip(np.searchsorted(changes, grid, side="right") - 1, 0, None)] if len(changes) \
        else np.full(len(grid), np.inf)
    front = settled_mask(grid, marks) & select & live & (since >= front_after_s)
    cands = candidate_times(evs, flights, court, base.rim_tol_ft)
    ch_steps = np.zeros(len(grid), bool)
    if len(changes):
        ch_steps[np.clip(np.round((changes - grid[0]) / base.dt_s).astype(int), 0, len(grid) - 1)] = True
    ch_steps &= select
    any_near = np.zeros(len(grid), bool)
    switch: dict[str, float] = {}
    for kind, times in cands.items():
        if kind not in base.switch:
            continue
        near = _near(grid, times, base.cand_window_s) & select
        any_near |= near
        switch[kind] = float(np.clip(ch_steps[near].sum() / max(1, int(near.sum())), 1e-4, 0.5))
    switch_base = float(np.clip((ch_steps & ~any_near).sum() / max(1, int((~any_near & select).sum())), 1e-5, 0.1))
    cues = [e for e in evs if e.kind in TEAM_CUES and e.team is not None]
    cue_truth = truth_teams(marks, np.array([e.t for e in cues], float)) if cues else []
    cue_sel = [select[min(len(grid) - 1, max(0, int(round((e.t - grid[0]) / base.dt_s))))] for e in cues]
    hits = [e.team is t for e, t, s in zip(cues, cue_truth, cue_sel, strict=True) if s and t is not None]
    engine_p = float(np.clip(np.mean(hits), 0.5, 0.99)) if hits else base.engine_p
    att = attack or attack_map(evs, direction, margin_ft=base.margin_ft)
    best: tuple[float, TimelineConfig] | None = None
    for lead in leads:
        sides = _aim_sides(direction, grid + lead, base.margin_ft)
        home = _home_sides(att, grid)
        team_side = np.array([0 if t is None else (h if t is Team.HOME else -h) for t, h in zip(truth, home, strict=True)])
        seen = front & (sides != 0)
        cam_p = float(np.clip(np.mean(sides[seen] == team_side[seen]), 0.55, 0.995)) if seen.any() else base.cam_p
        for mean, cw, scale, ew in itertools.product(back_means, cam_weights, base_scales, engine_weights):
            c = dataclasses.replace(base, lead_s=lead, cam_p=cam_p, cam_weight=cw, back_mean_s=mean,
                                    back_sd_s=0.8 * mean, engine_p=engine_p, engine_weight=ew, switch=dict(switch),
                                    switch_base=switch_base * scale)
            tl = decode(evs, direction, c, flights, court, att, grid)
            s = score(tl.teams(), grid, marks, select & live)
            objective = s["f1"] + s["accuracy"]
            if best is None or objective > best[0]:
                best = (objective, c)
    assert best is not None
    return best[1]


# --------------------------------------------------------------------------- events from the timeline


@dataclass
class DeriveConfig:
    shot_before_s: float = 10.0  # a change explained by a shot result of the team that lost the ball up to this long
    # before it: after a basket the timeline may change only when the other team brings the ball up
    shot_after_s: float = 1.5  # ... or this long after it (the change placed a little early)
    rebound_after_s: float = 2.0  # the engine's rebound of a defensive change: up to this after the change
    turnover_window_s: float = 3.0  # the engine's turnover / steal of a change: this near it
    attempt_window_s: float = 8.0  # a change with the team's engine attempt this long before it: a shot, result unseen
    oreb_window_s: float = 4.0  # an offensive rebound: the engine's rebound this long after a miss with no change
    putback_s: float = 5.0  # a miss followed this soon by the same team's next attempt, no change between: a put-back
    infer_turnovers: bool = True  # a change nothing explains is a turnover of the team that lost the ball
    break_s: float = 15.0  # a change after this long with the camera mostly off the court is a restart (time-out,
    break_share: float = 0.25  # free throws, a new period): no turnover is inferred when the share on it is below this
    steal_after_s: float = 8.0  # a turnover after which the camera stays on the court this long (no cut to a
    steal_share: float = 0.9  # replay or a stoppage) on at least this share of it is a live-ball one: a steal
    release_s: float = 1.0  # a result with no attempt was released this long before it


def _replace(e: Event, **kw: Any) -> Event:
    attrs = dict(e.attrs)
    attrs.update(kw.pop("attrs", {}))
    return dataclasses.replace(e, attrs=attrs, revision=e.revision + 1, **kw)


def derive_events(events: Iterable[Event], timeline: Timeline, config: DeriveConfig | None = None,
                  flights: Iterable[Any] = (), court: Any = None, rim_tol_ft: float = 3.0) -> list[Event]:
    """The engine's events with every possession-dependent one rebuilt from the timeline (``attrs["timeline"]`` says
    what changed):

    * shots (attempt and result), assists, passes and dribbles take the team with the ball at the release or the
      pass; blocks take the other team;
    * possession gains: one per change of team, its ``cause`` from what explains the change;
    * a change after a shot result of the team that lost the ball: a made basket (nothing to add) or a defensive
      rebound (the engine's rebound there, re-teamed, else one inferred at the change);
    * a change with no shot result: a turnover of the team that lost the ball (the engine's turnover there, else one
      inferred, unless the team had an engine shot attempt just before or a 3-D flight came down at the rim it
      attacks (with ``flights`` and ``court``: a shot whose result the engine missed), or the camera was mostly off
      the court before it, a restart after a time-out, free throws or a break), and a
      steal of the other team when the camera stays on the court after it (play went on: a live-ball turnover;
      a dead ball usually cuts to a replay or a stoppage);
    * a miss followed by no change: an offensive rebound where the engine saw a rebound, or where the same team
      shoots again within ``putback_s`` (a put-back);
    * every other engine rebound, steal, turnover and possession gain is dropped.

    Only what the camera saw explains a change (the engine's scorebug-sourced events pass through, re-teamed).
    Players are the engine's (their ids keep the engine's team prefix)."""
    cfg = config or DeriveConfig()
    evs = _live(events)
    attempts = {e.shot_id: e for e in evs if e.kind is K.SHOT_ATTEMPT and e.shot_id}
    out: dict[str, Event] = {}
    n_new = itertools.count(1)

    def release(e: Event) -> float:
        att = attempts.get(e.shot_id or "")
        return att.t if att is not None else e.t - cfg.release_s

    def team_at(t: float) -> Team | None:
        return timeline.team_at(t - 0.25)

    def keep(e: Event, team: Team | None, why: str, attrs: dict[str, Any] | None = None) -> None:
        """``e`` as it is, or a revision with the timeline's team and ``attrs``."""
        if (team is None or team is e.team) and all(e.attrs.get(k) == v for k, v in (attrs or {}).items()):
            out[e.event_id] = e
        else:
            out[e.event_id] = _replace(e, team=team or e.team, attrs={"timeline": why, **(attrs or {})})

    for e in evs:  # team labels
        if e.kind in (K.SHOT_ATTEMPT, *RESULTS, K.ASSIST):
            keep(e, team_at(release(e) if e.kind is not K.SHOT_ATTEMPT else e.t), "shot_team")
        elif e.kind in (K.PASS, K.DRIBBLE):
            keep(e, team_at(e.t), "team")
        elif e.kind is K.BLOCK:
            att = attempts.get(e.shot_id or "")
            shooter = team_at(att.t if att is not None else e.t)
            keep(e, OTHER[shooter] if shooter is not None else None, "team")
        elif e.kind not in (K.REBOUND, K.STEAL, K.TURNOVER, K.POSSESSION):
            out[e.event_id] = e

    seen = [e for e in evs if not from_scorebug(e)]  # changes are explained by what the camera saw
    results = [out[e.event_id] for e in seen if e.kind in RESULTS]
    rebounds = [e for e in seen if e.kind is K.REBOUND]
    losses = [e for e in seen if e.kind in (K.STEAL, K.TURNOVER)]
    used: set[str] = set()

    def new(kind: EventKind, t: float, team: Team, cause: str, attrs: dict[str, Any] | None = None) -> None:
        eid = f"tl-{kind.value}{next(n_new)}"
        out[eid] = Event(eid, kind, t, 0, team=team, confidence=0.6,
                         attrs={"timeline": cause, "inferred": True, **(attrs or {})})

    rims = rim_shots(flights, court, rim_tol_ft) if court is not None else []
    changes = timeline.changes()
    for c, lost, gained in changes:
        shot = max((r for r in results if r.team is lost and c - cfg.shot_before_s <= r.t <= c + cfg.shot_after_s),
                   key=lambda r: r.t, default=None)
        cause = "other"
        if shot is not None and shot.kind is K.SHOT_MADE:
            cause = "made_shot"
        elif shot is not None:
            cause = "defensive_rebound"
            reb = next((r for r in rebounds if r.event_id not in used and shot.t <= r.t <= c + cfg.rebound_after_s), None)
            if reb is not None:
                used.add(reb.event_id)
                keep(reb, gained, "defensive_rebound", attrs={"offensive": False})
            else:
                new(K.REBOUND, min(max(c, shot.t + 0.5), shot.t + 3.0), gained, "defensive_rebound",
                    attrs={"offensive": False})
        else:
            near = [x for x in losses if x.event_id not in used and abs(x.t - c) <= cfg.turnover_window_s]
            if near:
                cause = "turnover"
            elif any(e.kind is K.SHOT_ATTEMPT and team_at(e.t) is lost and c - cfg.attempt_window_s <= e.t <= c
                     for e in seen) or any(c - cfg.shot_before_s <= t <= c + cfg.shot_after_s
                                           and side == timeline.attack.side(lost, t) for t, side in rims):
                cause = "shot_unseen"
            elif timeline.tracked_share(c - cfg.break_s, c - 1.0) < cfg.break_share:
                cause = "restart"
            elif cfg.infer_turnovers:
                cause = "turnover"
            if cause == "turnover":
                used.update(x.event_id for x in near)
                tov = next((x for x in near if x.kind is K.TURNOVER), None)
                stl = next((x for x in near if x.kind is K.STEAL), None)
                t_loss = (tov or stl).t if (tov or stl) is not None else c  # type: ignore[union-attr]
                if tov is not None:
                    keep(tov, lost, "turnover")
                else:
                    new(K.TURNOVER, t_loss, lost, "turnover")
                if timeline.tracked_share(t_loss, t_loss + cfg.steal_after_s) >= cfg.steal_share:
                    cause = "steal"
                    if stl is not None:
                        keep(stl, gained, "steal")
                    else:
                        new(K.STEAL, t_loss, gained, "steal")
        new(K.POSSESSION, c, gained, cause, attrs={"cause": cause})

    for r in results:  # offensive rebounds: a miss with no change after it, where the engine saw a rebound
        if r.kind is not K.SHOT_MISSED or r.team is None or r.value == 1 or r.attrs.get("shot_type") == "free_throw":
            continue
        if any(r.t - cfg.shot_after_s <= c <= r.t + cfg.oreb_window_s for c, _, _ in changes):
            continue
        reb = next((x for x in rebounds if x.event_id not in used and r.t <= x.t <= r.t + cfg.oreb_window_s), None)
        if reb is not None:
            used.add(reb.event_id)
            keep(reb, r.team, "offensive_rebound", attrs={"offensive": True})
    # put-backs: a missed field goal, then the same team's next attempt within putback_s, the ball never changing hands
    # and no rebound credited between them: its own offensive rebound came in between
    tries = sorted((e for e in out.values() if e.kind is K.SHOT_ATTEMPT and not from_scorebug(e)), key=lambda e: e.t)
    credited = sorted(e.t for e in out.values() if e.kind is K.REBOUND)
    for r in results:
        if r.kind is not K.SHOT_MISSED or r.team is None or r.value == 1 or r.attrs.get("shot_type") == "free_throw":
            continue
        nxt = next((a for a in tries if r.t < a.t <= r.t + cfg.putback_s), None)
        if nxt is None or team_at(nxt.t) is not r.team or any(r.t <= c <= nxt.t for c, _, _ in changes) \
                or any(r.t <= t <= nxt.t for t in credited):
            continue
        new(K.REBOUND, (r.t + nxt.t) / 2, r.team, "putback", attrs={"offensive": True})
    return sorted(out.values(), key=lambda e: (e.t, e.event_id))
