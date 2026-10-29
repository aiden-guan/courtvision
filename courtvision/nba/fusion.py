"""Fuse the event engine's camera events with the scorebug's account of the game (``scorebug_events``).

The camera sees who had the ball and where it went; the scorebug knows for certain when a team scored and how
much, and its shot clock resets when the ball touches the rim. Each pass below is switchable (``FusionConfig``)
so its effect on the box score can be measured on its own:

0. ``team_map``: the engine's HOME / AWAY are jersey clusters, bound to the real teams only when the game's colours
   are configured (``perception/teams.py``: otherwise the order is arbitrary). The scorebug names the team of
   every basket: when the engine's own makes at the score changes are mostly of the other team, the engine's
   labels are swapped throughout.
1. ``makes``: every confirmed score change is a made basket of that team and value: the nearest engine shot
   result (made or missed, any team) within the window before the bug update is turned into that make (team,
   value; its attempt follows), else an attempt and a make are inferred (no shooter). Engine makes with no score
   change are turned into misses (a ball that did not go in). A make the engine itself inferred from a score change
   (``inferred: scoreboard``) comes after the shots the camera saw, and is dropped when its score change went to
   another shot (it was that shot, seen late, not a miss of its own).
2. ``rim_misses``: a shot-clock reset to 24 while the game clock runs, with no score change right after it and no
   engine turnover / steal near it, is a rim touch: the engine's missed shot just before it (or released in the
   ``reset_window_s`` before it, its result seen late) is kept; without one a missed attempt is inferred for the
   team in possession (``possession_at``). ``held_resets``: the shot clock starts when a player gains the ball, so
   after a rim touch the 24 holds while the ball is loose; a reset whose count starts at once (the 24 not read
   twice: a ``run24`` from the value before the reset) is a change of possession (a steal), not a rim touch.
   ``rimless_misses``: the other way round, an engine miss the shot clock shows no sign of (no reset, count or
   score near its result, the game clock running on) did not reach the rim, and is dropped (``_rimless_misses``).
3. ``assists``: assists of shots that ended up missed are retracted; a make without one takes the last completed
   pass of a teammate within ``assist_window_s`` (NBA scorers credit a pass leading directly to the basket).
   ``assist_possession``: the pass only has to fall in the scorer's possession, with no shot-clock reset (a rim
   touch, a change of possession, a whistle) between it and the release: the scorebug names the team that scored,
   which then had the ball all along, whatever team the engine gave the pass.
4. ``shot_teams`` (needs the camera, ``PlayDirection``): the broadcast camera follows the ball, so the floor point
   it aims at shows which half the play is in. Each team attacks one basket per half of the game; the made
   baskets (scorebug team + the half the camera was on) tell which. A shot takes the team attacking the half the
   camera aims at on its release, or with ``teams_at_rim`` at its result (the ball at the rim: the basket it went
   to, also when the engine's release time is off or the shot is a heave from the other half). ``rebounds``: after
   a missed field goal the play either leaves that half (the defence rebounded: the camera crosses half court and
   stays) or the attack goes on there (an offensive rebound); the engine's rebound after the miss is re-typed and
   re-teamed accordingly, and a miss without one gets one.
5. ``bug_rebounds``: the shot clock says the same more directly (NBA rule 7): after a missed field goal the first of
   a reset to 14 (the offence kept the ball), a count down from a fresh 24 (``run24``: the defence got it) or a
   put-back basket of the shooting team decides an offensive or a defensive rebound; such misses are settled before
   the camera's pass, which handles the rest. ``team_rebounds`` (off by default): the game clock stopping first
   (the ball out of bounds, a loose-ball foul) means no player gained the ball: a team rebound, which the box score
   credits to no player, so the engine's rebounds up to the stop go and the miss is settled with none. Off: on
   Q2-Q4 of Lakers @ Warriors (not tuned on) the play-by-play had a team rebound after 7 of the 15 such misses, a
   player's rebound (then a foul) after 7 (``tools/nba_bug_study.py --split``). ``oreb_misses``: a reset to
   14 while the game clock runs, with no shot just before it, is an offensive rebound of a miss nobody saw: a missed
   attempt is inferred for the team that kept the ball. ``orphan_rebounds`` (first): a rebound recovers a missed
   shot, so an engine rebound whose latest shot result before it is a make (the other team inbounds the ball), or
   that has no shot result within ``rebound_window_s`` before it, is dropped.
   Resets right at a restart of the game clock (``restart_guard``) are neither: play resumes there after free
   throws (a missed last one is rebounded as the clock starts) or an inbound.
6. ``no_rebound_at_whistle``: a miss followed at once by a clock stop (the ball out of bounds, a foul on the
   rebound) gets no inferred rebound: a team rebound is not in the box score. ``ft_rebounds`` (off by default: to be
   measured): a reset as play resumes after a team's free throws marks a missed last free throw and its rebound
   (14: the shooters kept the ball, 24: the defence got it).
7. ``handler_values`` (off by default: to be measured): an inferred miss (no shot seen, no 3-D flight) takes 2 or 3
   points from where the shooting team last had the ball just before the release (the engine's possession /
   dribble / pass positions), at the basket that team attacks.
8. ``turnover_resets``: a steal or a turnover hands the ball to the other team, which gets a fresh 24 (NBA rule 7),
   so the shot clock resets to 24 with it, and no shot or score explains that reset (a possession that ends in a shot
   is no turnover). An engine steal or turnover with no such reset near it is dropped. ``deadball_turnovers``: a
   reset to 24 at a stoppage with no shot before it and no free throw in the stoppage (the 24 is often set before
   the free throws of a foul) is a dead-ball turnover (out of bounds, a violation, an offensive foul) of the team
   that had the ball. ``reset_steals`` (run before ``rim_misses``): a
   reset to 24 in play with no shot attempt before it, after which the camera leaves the half the play was in, is a
   steal the engine missed (the ball changed hands in play), not a rim touch.

Times: the ESPN bug shows a basket ``score_lag_s`` after it; a shot result is a little after its release.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from courtvision.eval.events import latest_revisions
from courtvision.types import Event, EventKind, EventStatus, Team

from .scorebug_events import BugEvent

__all__ = ["FusionConfig", "PlayDirection", "attack_sides", "fuse", "possession_at"]


@dataclass
class PlayDirection:
    """Where the game camera aims on the floor over time: ``aim_x`` (court ft along the length) per tracked frame."""

    t: np.ndarray
    x: np.ndarray
    mid: float = 47.0
    margin_ft: float = 9.0  # the aim must be this far into a half to call the side

    @classmethod
    def from_camera(cls, camera: Any, smooth_s: float = 0.5) -> PlayDirection:
        frames = sorted(camera.params)
        C = np.asarray(camera.center, float)
        ts, xs = [], []
        for f in frames:
            pan, tilt, _, _ = camera.params[f]
            if tilt <= 0.5:
                continue
            d = C[2] / math.tan(math.radians(tilt))
            ts.append(f / camera.fps)
            xs.append(C[0] + d * math.sin(math.radians(pan)))
        t, x = np.asarray(ts), np.asarray(xs)
        k = max(1, int(round(smooth_s * camera.fps)))
        if len(x) > k:
            x = np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
        return cls(t, x)

    def aim(self, t: float, max_gap_s: float = 1.0) -> float | None:
        i = int(np.searchsorted(self.t, t))
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.t) and abs(self.t[j] - t) <= max_gap_s and (best is None or abs(self.t[j] - t) < abs(self.t[best] - t)):
                best = j
        return None if best is None else float(self.x[best])

    def side(self, t: float, margin_ft: float | None = None) -> str | None:
        a = self.aim(t)
        m = self.margin_ft if margin_ft is None else margin_ft
        if a is None:
            return None
        if a > self.mid + m:
            return "right"
        if a < self.mid - m:
            return "left"
        return None

    def leaves(self, side: str, t0: float, t1: float, hold_s: float = 2.0) -> float | None:
        """First time in [t0, t1] after which the aim stays in the other half for ``hold_s``."""
        other = "left" if side == "right" else "right"
        i0, i1 = int(np.searchsorted(self.t, t0)), int(np.searchsorted(self.t, t1))
        start = None
        for j in range(i0, min(i1, len(self.t))):
            x = self.x[j]
            in_other = x < self.mid - self.margin_ft if other == "left" else x > self.mid + self.margin_ft
            if in_other:
                if start is None:
                    start = self.t[j]
                if self.t[j] - start >= hold_s:
                    return float(start)
            else:
                start = None
        return None


def _half(period: int | None) -> int | None:
    """0: first half, 1: second half and overtime (teams switch baskets at half time)."""
    return None if period is None else (0 if period <= 2 else 1)


def attack_sides(scores: list[BugEvent], direction: PlayDirection, lag_s: float) -> dict[tuple[int, Team], str]:
    """(half, team) -> the side that team attacks, by majority over its made field goals."""
    votes: dict[tuple[int, Team], dict[str, int]] = {}
    for s in scores:
        h = _half(s.period)
        side = direction.side(s.t - lag_s - 0.5)
        if h is None or side is None or s.team is None or s.points == 1:
            continue
        v = votes.setdefault((h, s.team), {"left": 0, "right": 0})
        v[side] += 1
    out: dict[tuple[int, Team], str] = {}
    for h in (0, 1):
        home = votes.get((h, Team.HOME), {"left": 0, "right": 0})
        away = votes.get((h, Team.AWAY), {"left": 0, "right": 0})
        # the two teams attack opposite sides: pick the assignment the made baskets support most
        a = home["left"] + away["right"]
        b = home["right"] + away["left"]
        if a == b:
            continue
        out[(h, Team.HOME)] = "left" if a > b else "right"
        out[(h, Team.AWAY)] = "right" if a > b else "left"
    return out


@dataclass
class FusionConfig:
    team_map: bool = True
    team_map_min_votes: int = 4  # baskets matched to an engine shot of a known team before the labels can swap
    team_map_share: float = 0.75  # the share of them on the other team that swaps the labels
    makes: bool = True
    ft_trips: bool = True
    rim_misses: bool = True
    held_resets: bool = True  # rim_misses: only a reset whose 24 holds (the ball loose) is a rim touch
    rimless_misses: bool = True  # engine misses the shot clock shows no sign of (no rim touch, no stop) are dropped
    assists: bool = True
    flight_values: bool = True  # misses take 2 / 3 from a 3-D flight that reaches the rim near them
    handler_values: bool = False  # inferred misses take 2 / 3 from the ball handler's last position (pass 7)
    shot_teams: bool = True
    teams_at_rim: bool = True  # shot_teams: the camera half when the ball reaches the rim (the result), not the release
    rebounds: bool = True
    bug_rebounds: bool = True
    oreb_misses: bool = True
    orphan_rebounds: bool = True  # engine rebounds with no miss just before them are dropped (pass 5)
    reset_steals: bool = True  # a reset to 24 in play with no shot before it and the ball changing hands: a steal
    turnover_resets: bool = True  # engine steals / turnovers need a reset to 24 that no shot explains (pass 8)
    deadball_turnovers: bool = True  # a reset to 24 at a stoppage with no shot or free throw: a turnover (pass 8)
    team_rebounds: bool = False  # bug_rebounds: a clock stop before any possession sign is a team rebound (no player's)
    rebound_window_s: float = 6.0  # the rebound of a miss happens within this
    bug_rebound_window_s: float = 8.0  # the shot clock shows who got the ball within this after the miss
    restart_guard: bool = True  # resets at a restart of the clock infer no field-goal miss (pass 2)
    restart_window_s: tuple[float, float] = (-1.5, 3.0)  # reset time - clock start: "at the restart"
    no_rebound_at_whistle: bool = True  # a miss with a whistle right after it gets no inferred rebound (pass 6)
    whistle_after_miss_s: float = 2.5  # a clock stop this soon after a miss is that whistle
    ft_rebounds: bool = False  # a reset as play resumes after free throws: the last one missed, its rebound (pass 6)
    oreb_hold_s: float = 2.0  # the camera must stay in the other half this long to call a defensive rebound
    and_one_window_s: float = 90.0  # a team's basket this soon before its lone free throw (clock stopped): an and-one
    shot_deep_ft: float = 17.0  # a put-back attempt is taken with the camera this deep in the half (not mid-pan)
    flight_match_s: float = 1.5  # a flight's rim pass this close to a missed shot's result is that shot
    release_z_ft: float = 8.5  # the flight is followed back to this height for the release point (ball3d.release_point)
    handler_window_s: float = 2.5  # pass 6: the ball handler's position this long before the release at most
    three_margin_ft: float = 0.5  # release point beyond the arc by this much (ft; may be negative) counts as a three
    score_lag_s: float = 2.0  # ESPN: bug update after the ball goes in (median, Lakers @ Warriors)
    make_window_s: tuple[float, float] = (-3.5, 1.5)  # engine result time - (bug time - lag)
    ft_window_s: tuple[float, float] = (-4.0, 2.0)
    reset_window_s: tuple[float, float] = (-3.0, 1.0)  # engine miss time - reset time
    reset_turnover_s: float = 2.5
    reset_make_after_s: float = 4.0  # a score change this soon after a reset: the reset was the make
    inferred_release_s: float = 1.0  # an inferred attempt is released this long before its rim touch / make
    assist_window_s: float = 3.5
    assist_possession: bool = True  # pass 3: the assist is the last pass in the scorer's possession, any team label


def _team_swap(events: list[Event], scores: list[BugEvent], cfg: FusionConfig) -> bool:
    """Whether the engine's HOME / AWAY are the other way round (pass 0): a score change with exactly one engine make
    (of a known team) in its make window votes with that make's team against the scorebug's. Engine misses do not
    vote: next to a basket they are as often the other team's shots."""
    same = other = 0
    makes = [e for e in _results(events) if e.kind is EventKind.SHOT_MADE and e.team is not None
             and e.source != "scoreboard"]
    for s in scores:
        if s.team is None:
            continue
        tm = s.t - cfg.score_lag_s
        ft = s.points == 1
        lo, hi = cfg.ft_window_s if ft else cfg.make_window_s
        cands = [e for e in makes if _is_ft(e) == ft and lo <= e.t - tm <= hi]
        if len(cands) != 1:
            continue
        c = cands[0]
        if c.team is s.team:
            same += 1
        else:
            other += 1
    n = same + other
    return n >= cfg.team_map_min_votes and other >= cfg.team_map_share * n


def _results(events: list[Event]) -> list[Event]:
    return [e for e in events if e.kind in (EventKind.SHOT_MADE, EventKind.SHOT_MISSED)]


def _is_ft(e: Event) -> bool:
    return e.value == 1 or e.attrs.get("shot_type") == "free_throw"


def _engine_guess(e: Event) -> bool:
    """A make the engine inferred from a score change it matched to no seen shot (``inferred: scoreboard``): the
    engine's own reading of a score change the makes pass reads again, not a shot the camera saw."""
    return e.attrs.get("inferred") == "scoreboard"


def possession_at(events: list[Event], t: float) -> Team | None:
    """The team that last gained or used the ball before ``t`` (possession, pass, rebound, steal, shot)."""
    best: tuple[float, Team] | None = None
    for e in events:
        if e.t > t or e.team is None:
            continue
        if e.kind in (EventKind.POSSESSION, EventKind.PASS, EventKind.REBOUND, EventKind.STEAL,
                      EventKind.SHOT_ATTEMPT, EventKind.DRIBBLE):
            if best is None or e.t >= best[0]:
                best = (e.t, e.team)
    return None if best is None else best[1]


def _replace(e: Event, **kw: object) -> Event:
    attrs = dict(e.attrs)
    attrs.update(kw.pop("attrs", {}))  # type: ignore[arg-type]
    return dataclasses.replace(e, attrs=attrs, revision=e.revision + 1, **kw)  # type: ignore[arg-type]


def fuse(engine_events: Iterable[Event], bug: list[BugEvent], config: FusionConfig | None = None,
         direction: PlayDirection | None = None, periods: list[tuple[float, int]] | None = None,
         flights: list[Any] | None = None, court: Any | None = None) -> list[Event]:
    """``direction``: the camera's aim over time (passes 4-5); ``periods``: (video t, period) reads, for the half of
    the game at a time."""
    cfg = config or FusionConfig()
    events = [e for e in latest_revisions(engine_events) if e.status is not EventStatus.RETRACTED]
    events.sort(key=lambda e: e.t)
    by_id = {e.event_id: e for e in events}
    attempts = {e.shot_id: e for e in events if e.kind is EventKind.SHOT_ATTEMPT and e.shot_id}
    n_new = 0

    def new_id(prefix: str) -> str:
        nonlocal n_new
        n_new += 1
        return f"fuse-{prefix}{n_new}"

    def set_shot(res: Event, made: bool, team: Team | None, value: int, why: str) -> None:
        kind = EventKind.SHOT_MADE if made else EventKind.SHOT_MISSED
        by_id[res.event_id] = _replace(res, kind=kind, team=team or res.team, value=value,
                                       status=EventStatus.CONFIRMED, attrs={"fused": why})
        att = attempts.get(res.shot_id or "")
        if att is not None:
            cur = by_id.get(att.event_id, att)
            by_id[att.event_id] = attempts[res.shot_id or ""] = _replace(cur, team=team or cur.team, value=value,
                                                                         attrs={"fused": why})

    def add_shot(t_res: float, team: Team | None, value: int, made: bool, why: str) -> None:
        sid = new_id("s")
        att = Event(sid + "a", EventKind.SHOT_ATTEMPT, t_res - cfg.inferred_release_s, 0, team=team, value=value,
                    shot_id=sid, confidence=0.6, source="scoreboard", attrs={"fused": why, "inferred": True,
                                                                               **({"shot_type": "free_throw"} if value == 1 else {})})
        res = Event(sid + "r", EventKind.SHOT_MADE if made else EventKind.SHOT_MISSED, t_res, 0, team=team,
                    value=value, shot_id=sid, confidence=0.6, source="scoreboard",
                    status=EventStatus.CONFIRMED if made else EventStatus.PROVISIONAL,
                    attrs={"fused": why, "inferred": True, **({"shot_type": "free_throw"} if value == 1 else {})})
        by_id[att.event_id] = att
        by_id[res.event_id] = res
        attempts[sid] = att

    scores = [b for b in bug if b.kind == "score"]
    if cfg.team_map:
        swap = _team_swap(list(by_id.values()), scores, cfg)
        if swap:
            for e in list(by_id.values()):
                if e.team is not None:
                    by_id[e.event_id] = _replace(e, team=e.team.other(), attrs={"team_from": "scorebug team map"})
            attempts = {sid: by_id.get(a.event_id, a) for sid, a in attempts.items()}
    if cfg.makes:
        used: set[str] = set()
        for s in scores:
            tm = s.t - cfg.score_lag_s
            ft = s.points == 1
            lo, hi = cfg.ft_window_s if ft else cfg.make_window_s
            cands = [e for e in _results(list(by_id.values())) if e.event_id not in used and _is_ft(e) == ft
                     and lo <= e.t - tm <= hi]
            if cands:
                # a shot the camera saw goes before the engine's own guess at this score change
                c = min(cands, key=lambda e: (_engine_guess(e),
                                              e.kind is not EventKind.SHOT_MADE or e.team is not s.team, abs(e.t - tm)))
                used.add(c.event_id)
                set_shot(c, True, s.team, s.points, "score change")
            else:
                add_shot(tm, s.team, s.points, True, "score change with no seen shot")
                used.add(f"fuse-s{n_new}r")
        for e in _results(list(by_id.values())):
            if e.kind is EventKind.SHOT_MADE and e.event_id not in used and e.source != "scoreboard":
                if _engine_guess(e):  # its score change is another shot's make: the guess was no shot of its own
                    by_id.pop(e.event_id, None)
                    att = attempts.pop(e.shot_id or "", None)
                    if att is not None:
                        by_id.pop(att.event_id, None)
                    continue
                set_shot(e, False, e.team, e.value, "no score change")
    if cfg.ft_trips:
        _ft_trips(by_id, bug, cfg, add_shot)
    clock = sorted((b.t, b.kind == "clock_start") for b in bug if b.kind in ("clock_start", "clock_stop"))
    clock_t = np.array([c[0] for c in clock])

    def running(t: float) -> bool:
        i = int(np.searchsorted(clock_t, t, side="right")) - 1
        return i < 0 or clock[i][1]

    starts = [b.t for b in bug if b.kind == "clock_start"]
    stops = [b.t for b in bug if b.kind == "clock_stop"]

    def at_restart(t: float) -> bool:
        """Play resuming: after free throws (a missed last one rebounded) or an inbound, not a field goal's rim."""
        lo, hi = cfg.restart_window_s
        return cfg.restart_guard and any(lo <= t - st <= hi for st in starts)

    def in_play(t: float) -> bool:
        return running(t) and not at_restart(t)

    if cfg.rimless_misses:
        _rimless_misses(by_id, attempts, bug, cfg, running)
    sides = attack_sides(scores, direction, cfg.score_lag_s) if direction is not None else {}

    def period_at(t: float) -> int | None:
        if not periods:
            return None
        i = int(np.searchsorted(np.array([p[0] for p in periods]), t, side="right")) - 1
        return periods[max(0, min(i, len(periods) - 1))][1]

    def attacker(t: float) -> tuple[Team | None, str | None]:
        if direction is None:
            return None, None
        side = direction.side(t)
        h = _half(period_at(t))
        if side is None or h is None:
            return None, side
        for team in (Team.HOME, Team.AWAY):
            if sides.get((h, team)) == side:
                return team, side
        return None, side

    if cfg.reset_steals and direction is not None and sides:
        _reset_steals(by_id, bug, cfg, in_play, direction, attacker)  # play resuming at a restart is no steal
    if cfg.rim_misses:
        cur = sorted(by_id.values(), key=lambda e: e.t)
        for r in (b for b in bug if b.kind == "reset24"):
            if not running(r.t) or at_restart(r.t):
                continue  # a dead-ball reset (foul, violation, out of bounds) or play resuming, not a rim touch
            if any(0.0 <= s.t - r.t <= cfg.reset_make_after_s for s in scores):
                continue  # the reset was a make
            if any(e.kind in (EventKind.TURNOVER, EventKind.STEAL) and abs(e.t - r.t) <= cfg.reset_turnover_s for e in cur):
                continue
            lo, hi = cfg.reset_window_s
            if any(e.kind is EventKind.SHOT_MISSED and lo <= e.t - r.t <= hi for e in cur):
                continue
            if any(e.kind is EventKind.SHOT_MADE and abs(e.t - r.t) <= 2.0 for e in cur):
                continue
            if any(e.kind is EventKind.SHOT_ATTEMPT and not _is_ft(e) and lo <= e.t - r.t <= 0.0 for e in cur):
                continue  # a shot in the air at the reset (its result, seen late, can come well after the rim)
            if cfg.held_resets and _counts_at_once(r, bug):
                continue  # the count started with the reset (the 24 never held): a team had the ball, no rim touch
            add_shot(r.t, possession_at(cur, r.t - cfg.inferred_release_s), 2, False, "shot clock reset (rim touch)")
    if cfg.oreb_misses:
        cur = sorted(by_id.values(), key=lambda e: e.t)
        for r in (b for b in bug if b.kind == "reset14"):
            # a foul or a kicked ball also sets 14, but it stops the game clock; a missed last free throw's
            # offensive rebound sets it as the clock starts
            if not (running(r.t) and running(r.t + 1.5)) or any(r.t - 0.5 <= t <= r.t + 1.5 for t in stops) \
                    or at_restart(r.t):
                continue
            if any(e.kind in (EventKind.SHOT_MISSED, EventKind.SHOT_MADE) and r.t - 6.0 <= e.t <= r.t + 0.5
                   for e in cur):
                continue  # the miss it follows is known
            t_miss = r.t - 1.5  # the rim touch, the rebound and the 14 on the bug
            add_shot(t_miss, possession_at(cur, t_miss - cfg.inferred_release_s), 2, False,
                     "shot clock reset to 14 (offensive rebound)")
    if cfg.flight_values and flights and court is not None:
        _flight_values(by_id, attempts, flights, court, cfg)
    out = sorted(by_id.values(), key=lambda e: e.t)
    if cfg.shot_teams and sides:
        res_by_shot = {e.shot_id: e for e in out if e.kind in (EventKind.SHOT_MADE, EventKind.SHOT_MISSED) and e.shot_id}
        fixed = []
        for e in out:
            if e.kind in (EventKind.SHOT_ATTEMPT, EventKind.SHOT_MADE, EventKind.SHOT_MISSED) and not _is_ft(e) \
                    and not (e.kind is EventKind.SHOT_MADE and e.attrs.get("fused") == "score change") and e.shot_id:
                att = attempts.get(e.shot_id)
                res = res_by_shot.get(e.shot_id)
                if res is not None and res.kind is EventKind.SHOT_MADE and res.attrs.get("fused", "").startswith("score"):
                    fixed.append(e)  # the scorebug named the team
                    continue
                # the half the ball went to: at its result (the ball at the rim) with teams_at_rim, else the release
                t_cam = res.t if cfg.teams_at_rim and res is not None else (att.t if att is not None else e.t)
                team, _ = attacker(t_cam)
                if team is not None and team is not e.team:
                    e = _replace(e, team=team, attrs={"team_from": "camera half"})
            fixed.append(e)
        out = fixed
    if cfg.orphan_rebounds:
        out = _orphan_rebounds(out, cfg)
    settled: set[str] = set()
    if cfg.bug_rebounds:
        out, settled = _bug_rebounds(out, bug, cfg, running)
    if cfg.rebounds and sides:
        out = _fix_rebounds(out, cfg, direction, attacker, settled, stops)  # type: ignore[arg-type]
    if cfg.ft_rebounds:
        out = _ft_rebounds(out, bug, cfg)
    if cfg.handler_values and court is not None:
        out = _handler_values(out, cfg, court, attacker)
    if cfg.turnover_resets or cfg.deadball_turnovers:
        out = _bug_turnovers(out, bug, cfg, running)
    if cfg.assists:
        made_ids = {e.shot_id for e in out if e.kind is EventKind.SHOT_MADE and not _is_ft(e)}
        final = []
        for e in out:
            if e.kind is EventKind.ASSIST and e.shot_id is not None and e.shot_id not in made_ids:
                continue  # its shot was not made
            final.append(e)
        out = final
        assisted = {e.shot_id for e in out if e.kind is EventKind.ASSIST}
        resets = np.array(sorted(b.t for b in bug if b.kind in ("reset24", "reset14", "run24")))
        game_clock = sorted((b.t, b.game_clock) for b in bug if b.game_clock is not None)
        all_passes = sorted((p for p in out if p.kind is EventKind.PASS and p.team is not None), key=lambda p: p.t)
        pass_t = np.array([p.t for p in all_passes])

        def n_in(ts: np.ndarray, t0: float, t1: float) -> int:  # how many of the sorted ``ts`` lie in (t0, t1]
            return int(np.searchsorted(ts, t1, side="right") - np.searchsorted(ts, t0, side="right"))

        for e in [e for e in out if e.kind is EventKind.SHOT_MADE and not _is_ft(e) and e.shot_id not in assisted]:
            att = attempts.get(e.shot_id or "")
            t_rel = att.t if att is not None else e.t - cfg.inferred_release_s
            # the shot clock tells the possession only where it is on and read: a shot-clock event in the 24 s before
            # the release, or just after it (the make's own reset; a long possession counts down with no event)
            gc = [g for tb, g in game_clock if tb <= t_rel]
            told = cfg.assist_possession and not (gc and gc[-1] < 24.0) and (
                n_in(resets, t_rel - 24.0, t_rel) > 0 or n_in(resets, t_rel, t_rel + 6.0) > 0)

            def teammate(p: Event) -> bool:
                if not told:
                    return p.team is e.team
                # no shot-clock reset since the pass (a steal resets it too): the scoring team (the bug's) had the
                # ball all along, whatever team the engine gave the pass; the engine's own steals are too often false
                return n_in(resets, p.t, t_rel) == 0

            i0, i1 = np.searchsorted(pass_t, t_rel - cfg.assist_window_s), np.searchsorted(pass_t, t_rel, side="right")
            passes = [p for p in all_passes[i0:i1] if teammate(p) and (e.player_id is None or p.player_id != e.player_id)]
            if passes:
                p = max(passes, key=lambda p: p.t)
                out.append(Event(new_id("a"), EventKind.ASSIST, p.t, p.frame_idx, team=e.team, player_id=p.player_id,
                                 other_player_id=e.player_id, shot_id=e.shot_id, confidence=0.6, source="cv",
                                 attrs={"fused": "last pass before a make"}))
        out.sort(key=lambda e: e.t)
    return out


def _set_rebound(out: list[Event], m: Event, nxt_t: float, window_s: float, team: Team, offensive: bool,
                 t_reb: float, why: str, new_id: str) -> None:
    """The rebound of miss ``m`` is ``team``'s (``offensive``): the engine's rebound after it (before the next miss
    at ``nxt_t``) re-teamed / re-typed, else an inferred one at ``t_reb``."""
    rebs = [e for e in out if e.kind is EventKind.REBOUND and 0.0 <= e.t - m.t <= window_s and e.t < nxt_t]
    if rebs:
        r = min(rebs, key=lambda e: abs(e.t - t_reb))
        if r.team is not team or bool(r.attrs.get("offensive")) != offensive:
            out[out.index(r)] = _replace(r, team=team, attrs={"offensive": offensive, "fused": why})
        return
    out.append(Event(new_id, EventKind.REBOUND, t_reb, m.frame_idx, team=team, player_id=f"{team.value}-unknown",
                     confidence=0.5, source="cv", attrs={"offensive": offensive, "fused": why, "inferred": True}))


def _orphan_rebounds(events: list[Event], cfg: FusionConfig) -> list[Event]:
    """Engine rebounds with nothing to rebound (pass 5): a rebound is the recovery of a missed shot, so the latest
    shot result at or before it (within ``rebound_window_s``) must be a miss. A rebound right after a make (the
    other team inbounds the ball) or with no shot before it is dropped."""
    results = sorted(_results(events), key=lambda e: e.t)
    rt = np.array([e.t for e in results])
    out = []
    for e in events:
        if e.kind is EventKind.REBOUND:
            i = int(np.searchsorted(rt, e.t, side="right")) - 1
            if i < 0 or results[i].kind is not EventKind.SHOT_MISSED or e.t - results[i].t > cfg.rebound_window_s:
                continue
        out.append(e)
    return out


def _bug_rebounds(events: list[Event], bug: list[BugEvent], cfg: FusionConfig,
                  running: Any) -> tuple[list[Event], set[str]]:
    """Offensive / defensive rebounds of missed field goals from the shot clock (pass 5) -> (events, the misses
    settled). The first sign within ``bug_rebound_window_s`` after the miss (and before the next one) decides: a
    reset to 14 with the clock running or a put-back basket of the shooting team (offensive), a count from a
    fresh 24 (defensive). A clock stop first (a foul, the ball out of bounds) is a team rebound with
    ``team_rebounds`` (no player's: the engine's rebounds up to the stop are dropped), else it leaves the miss
    open, as no sign does."""
    out = list(events)
    marks = sorted((b for b in bug if b.kind in ("reset14", "run24", "score", "clock_stop")), key=lambda b: b.t)
    misses = sorted((e for e in out if e.kind is EventKind.SHOT_MISSED and not _is_ft(e) and e.team is not None
                     and not e.attrs.get("blocked_oob")), key=lambda e: e.t)
    settled: set[str] = set()
    for k, m in enumerate(misses):
        nxt_t = misses[k + 1].t if k + 1 < len(misses) else math.inf
        hi = min(m.t + cfg.bug_rebound_window_s, nxt_t)
        mark = next((b for b in marks if m.t - 0.5 < b.t <= hi), None)
        if mark is None:
            continue
        if mark.kind == "clock_stop":
            if cfg.team_rebounds:
                # the whistle before the shot clock showed anyone with the ball (a count from 24, or 14): the ball
                # went out of bounds or a loose-ball foul, a team rebound the box score credits to no player
                out = [e for e in out if not (e.kind is EventKind.REBOUND and 0.0 <= e.t - m.t and e.t <= mark.t)]
                settled.add(m.event_id)
            continue
        assert m.team is not None
        if mark.kind == "reset14" and running(mark.t):
            offensive, t_reb = True, mark.t - 1.0
        elif mark.kind == "run24":
            offensive, t_reb = False, mark.t - 1.5
        elif mark.kind == "score" and mark.team is m.team and mark.points >= 2:
            offensive, t_reb = True, m.t + 0.5  # a tip-in: the put-back is the rebound
        else:
            continue
        team = m.team if offensive else m.team.other()
        _set_rebound(out, m, nxt_t, cfg.bug_rebound_window_s, team, offensive, max(m.t + 0.5, t_reb),
                     "shot clock", f"fuse-breb{len(settled) + 1}")
        settled.add(m.event_id)
    out.sort(key=lambda e: e.t)
    return out, settled


def _counts_at_once(r: BugEvent, bug: list[BugEvent]) -> bool:
    """The shot clock counts straight on from a reset to 24 (the 24 not read twice: scorebug_events' reset24 + run24
    pair from the value before the reset): a team had the ball at the reset. After a rim touch the 24 holds while the
    ball is loose and starts when a player gains it."""
    return any(b.kind == "run24" and b.shot_clock_before == r.shot_clock_before and 0.0 < b.t - r.t <= 2.0 for b in bug)


def _reset_steals(by_id: dict[str, Event], bug: list[BugEvent], cfg: FusionConfig, running: Any,
                  direction: PlayDirection, attacker: Any) -> None:
    """Steals the engine missed, from the shot clock (pass 2a, before the rim touches). A reset to 24 in play is a rim
    touch or a change of possession; with no engine shot attempt in the ``rebound_window_s`` before it (no shot that
    could have reached the rim or been blocked), no shot result within ``reset_window_s`` and no score change within
    ``reset_make_after_s`` after it, it is a change of possession in play: the ball was stolen. The camera names the
    teams (the engine's team labels flip too easily around a steal): the team attacking the half it aims at on the
    reset lost the ball, and the camera has to leave that half (as for a defensive rebound) within
    ``rebound_window_s``: a steal of the other team and a turnover of that one. A rim touch looks the same when the
    engine missed the shot, so the shot clock must also count at once (``_counts_at_once``: the thief has the ball,
    while after a rim touch the 24 holds as long as the ball is loose)."""
    cur = sorted(by_id.values(), key=lambda e: e.t)
    scores = [b for b in bug if b.kind == "score"]
    lo, hi = cfg.reset_window_s
    n_new = 0
    for r in (b for b in bug if b.kind == "reset24"):
        if not running(r.t):
            continue  # a dead-ball reset: ``deadball_turnovers`` (pass 8)
        if any(0.0 <= s.t - r.t <= cfg.reset_make_after_s for s in scores):
            continue
        if any(e.kind in (EventKind.TURNOVER, EventKind.STEAL) and abs(e.t - r.t) <= cfg.reset_turnover_s for e in cur):
            continue  # the engine has it
        if any(e.kind is EventKind.SHOT_ATTEMPT and r.t - cfg.rebound_window_s <= e.t <= r.t for e in cur) \
                or any(e.kind in (EventKind.SHOT_MADE, EventKind.SHOT_MISSED) and lo <= e.t - r.t <= hi for e in cur):
            continue  # a shot's rim touch, rebound or block
        if not _counts_at_once(r, bug):
            continue  # the 24 held: a loose ball after a rim touch (a shot the engine did not see), not a steal
        lost, side = attacker(r.t)
        if lost is None or side is None or direction.leaves(side, r.t, r.t + cfg.rebound_window_s,
                                                             cfg.oreb_hold_s) is None:
            continue
        got = lost.other()
        n_new += 1
        for kind, team in ((EventKind.STEAL, got), (EventKind.TURNOVER, lost)):
            e = Event(f"fuse-st{n_new}{kind.value[0]}", kind, r.t, 0, team=team, player_id=f"{team.value}-unknown",
                      confidence=0.5, source="scoreboard",
                      attrs={"fused": "shot clock reset in play with no shot", "inferred": True})
            by_id[e.event_id] = e


def _bug_turnovers(events: list[Event], bug: list[BugEvent], cfg: FusionConfig, running: Any) -> list[Event]:
    """Steals and turnovers from the shot clock (pass 8). The team that gains the ball gets a fresh 24, so a turnover
    shows as a reset to 24 that no shot explains: no shot result within ``reset_window_s`` of it (the rim touch), no
    score change within ``reset_make_after_s`` after it and, at a stoppage, none since the whistle (the free throws).
    ``turnover_resets``: an engine steal or turnover with no such reset within ``reset_turnover_s`` (of the whistle,
    at a stoppage) is dropped; a steal is also the other team's turnover at the same time, so the pair goes together.
    With under 24 s on the game clock the shot clock is off and shows nothing: those are kept. ``deadball_turnovers``:
    such a reset at a stoppage, with no shot result in the ``rebound_window_s`` before the whistle (a team rebound,
    a foul on the shot), no free throw before the game clock restarts (a foul: the 24 is often set before its free
    throws) and no engine turnover near, is a dead-ball turnover of the team that had the ball."""
    out = list(events)
    resets = [b for b in bug if b.kind == "reset24"]
    if not resets:
        return out  # no shot clock read: nothing to check against
    scores = [b for b in bug if b.kind == "score"]
    stops = sorted(b.t for b in bug if b.kind == "clock_stop")
    starts = sorted(b.t for b in bug if b.kind == "clock_start")
    clocks = sorted((b.t, b.game_clock) for b in bug if b.game_clock is not None)
    shots = [e for e in out if e.kind in (EventKind.SHOT_MADE, EventKind.SHOT_MISSED) and not _is_ft(e)]
    lo, hi = cfg.reset_window_s

    def whistle(r: BugEvent) -> float:
        """When the possession changed: the clock stop before a reset made at a stoppage, else the reset."""
        return r.t if running(r.t) else next((t for t in reversed(stops) if t <= r.t), r.t)

    def explained(r: BugEvent) -> bool:
        if any(lo <= e.t - r.t <= hi for e in shots) or any(0.0 <= s.t - r.t <= cfg.reset_make_after_s for s in scores):
            return True
        w = whistle(r)
        return not running(r.t) and any(w - cfg.score_lag_s <= s.t <= r.t for s in scores)  # the free throws

    def clock_off(t: float) -> bool:
        before = [g for tb, g in clocks if tb <= t]
        return bool(before) and before[-1] < 24.0

    changes = [(whistle(r), r) for r in resets if not explained(r)]
    if cfg.turnover_resets:
        out = [e for e in out if e.kind not in (EventKind.STEAL, EventKind.TURNOVER) or clock_off(e.t)
               or any(w - cfg.reset_turnover_s <= e.t <= r.t + cfg.reset_turnover_s for w, r in changes)]
    if cfg.deadball_turnovers:
        n_new = 0
        for w, r in changes:
            if running(r.t):
                continue  # a reset in play: a steal (``reset_steals``) or a rim touch
            restart = next((t for t in starts if t > r.t), math.inf)
            if any(s.points == 1 and r.t <= s.t <= restart for s in scores):
                continue  # free throws in the stoppage: a foul (the 24 was set before them), not a turnover
            if any(w - cfg.rebound_window_s <= e.t <= r.t for e in shots):
                continue  # the ball went out after that shot (a team rebound) or the shooter was fouled
            if any(e.kind is EventKind.TURNOVER and w - cfg.reset_turnover_s <= e.t <= r.t + cfg.reset_turnover_s
                   for e in out):
                continue  # the engine has it
            team = possession_at(out, w)
            if team is None:
                continue
            n_new += 1
            out.append(Event(f"fuse-to{n_new}", EventKind.TURNOVER, w, 0, team=team, player_id=f"{team.value}-unknown",
                             confidence=0.5, source="scoreboard",
                             attrs={"fused": "shot clock reset at a stoppage", "inferred": True}))
    out.sort(key=lambda e: e.t)
    return out


def _fix_rebounds(events: list[Event], cfg: FusionConfig, direction: PlayDirection,
                  attacker: Any, settled: set[str] | None = None, stops: list[float] | None = None) -> list[Event]:
    """Re-type / re-team the rebound after each missed field goal from where the play goes next (pass 5); misses in
    ``settled`` (the shot clock decided them) are left alone. A miss with a clock stop right after it (``stops``;
    the ball out of bounds, a foul) gets no inferred rebound (``no_rebound_at_whistle``)."""
    out = list(events)
    misses = [e for e in out if e.kind is EventKind.SHOT_MISSED and not _is_ft(e) and not e.attrs.get("blocked_oob")
              and e.event_id not in (settled or set())]
    shots = sorted((e for e in out if e.kind is EventKind.SHOT_ATTEMPT), key=lambda e: e.t)
    n_new = 0
    for m in misses:
        team, side = attacker(m.t - 0.5)
        if team is None or side is None:
            continue
        # the next attempt at the same basket (the camera on the same half at its release)
        nxt_shot = next((s for s in shots if m.t + 0.5 < s.t <= m.t + 30.0
                         and direction.side(s.t, cfg.shot_deep_ft) == side), None)
        left = direction.leaves(side, m.t, m.t + 30.0, cfg.oreb_hold_s)
        if left is not None and (nxt_shot is None or left < nxt_shot.t + 1.0):
            reb_team, offensive, t_reb = team.other(), False, left - 1.0
        elif nxt_shot is not None and (left is None or nxt_shot.t + 1.0 <= left):
            reb_team, offensive, t_reb = team, True, m.t + 1.0
        else:
            continue
        rebs = [e for e in out if e.kind is EventKind.REBOUND and 0.0 <= e.t - m.t <= cfg.rebound_window_s
                and (offensive or e.t <= (left or 1e9) + 0.5)]
        if rebs:
            r = min(rebs, key=lambda e: abs(e.t - t_reb))
            if r.team is not reb_team or bool(r.attrs.get("offensive")) != offensive:
                out[out.index(r)] = _replace(r, team=reb_team, attrs={"offensive": offensive, "fused": "play direction"})
        elif cfg.no_rebound_at_whistle and any(m.t - 0.5 <= t <= m.t + cfg.whistle_after_miss_s for t in stops or ()):
            continue  # out of bounds or a foul: a team rebound, not in the box score
        else:
            n_new += 1
            out.append(Event(f"fuse-reb{n_new}", EventKind.REBOUND, min(m.t + 1.5, max(m.t + 0.5, t_reb)), m.frame_idx,
                             team=reb_team,
                             player_id=f"{reb_team.value}-unknown", confidence=0.5, source="cv",
                             attrs={"offensive": offensive, "fused": "play direction", "inferred": True}))
    out.sort(key=lambda e: e.t)
    return out


def _ft_rebounds(events: list[Event], bug: list[BugEvent], cfg: FusionConfig) -> list[Event]:
    """The rebound of a missed last free throw (pass 6): a stoppage in which one team made free throws, ended by a
    restart of the clock with a shot-clock reset at it (``restart_window_s``). After a made last free throw the
    reset to 24 comes during the stoppage, before the inbound; after a missed one, at the rebound, which starts the
    clock: 14 when the shooters kept the ball (offensive), 24 when the defence got it."""
    out = list(events)
    made_ft = [b for b in bug if b.kind == "score" and b.points == 1 and b.team is not None]
    stops = sorted(b.t for b in bug if b.kind == "clock_stop")
    resets = sorted((b for b in bug if b.kind in ("reset24", "reset14")), key=lambda b: b.t)
    lo, hi = cfg.restart_window_s
    n_new = 0
    for st in sorted(b.t for b in bug if b.kind == "clock_start"):
        stop = max((t for t in stops if t < st), default=None)
        if stop is None:
            continue
        fts = [b for b in made_ft if stop - 1.0 < b.t <= st + 1.0]
        teams = {b.team for b in fts}
        r = next((b for b in resets if lo <= b.t - st <= hi), None)
        if len(teams) != 1 or r is None:
            continue  # no free throws, both teams shot (technicals), or no sign of a rebound
        team = fts[0].team
        assert team is not None
        offensive = r.kind == "reset14"
        reb_team = team if offensive else team.other()
        rebs = [e for e in out if e.kind is EventKind.REBOUND and -2.0 <= e.t - st <= 2.0]
        if rebs:
            rb = min(rebs, key=lambda e: abs(e.t - st))
            if rb.team is not reb_team or bool(rb.attrs.get("offensive")) != offensive:
                out[out.index(rb)] = _replace(rb, team=reb_team, attrs={"offensive": offensive,
                                                                       "fused": "free-throw rebound"})
            continue
        n_new += 1
        out.append(Event(f"fuse-ftreb{n_new}", EventKind.REBOUND, st, 0, team=reb_team,
                         player_id=f"{reb_team.value}-unknown", confidence=0.5, source="cv",
                         attrs={"offensive": offensive, "fused": "free-throw rebound", "inferred": True}))
    out.sort(key=lambda e: e.t)
    return out


def _ft_trips(by_id: dict[str, Event], bug: list[BugEvent], cfg: FusionConfig, add_shot: Any) -> None:
    """Missed free throws from the trip each group of made ones belongs to (pass 2b).

    Made free throws (+1 score changes) of a team within one clock stoppage (no clock start between them) are one
    trip. A lone make right after the same team's basket with the clock stopped since is an and-one (one shot);
    otherwise a trip has at least two shots (three when three went in: a fouled three, or a technical before a
    two-shot trip). The shortfall are misses: the engine's own free-throw misses of that team in the stoppage count
    first, inferred ones make up the rest, extra ones are dropped. A trip with no make (both missed) is not seen."""
    starts = [b.t for b in bug if b.kind == "clock_start"]
    stops = [b.t for b in bug if b.kind == "clock_stop"]
    scores = sorted((b for b in bug if b.kind == "score"), key=lambda b: b.t)

    def started_between(a: float, b: float) -> bool:
        return any(a < t < b for t in starts)

    def stopped_since(a: float, b: float) -> bool:
        """The clock stopped right after ``a`` (the foul on the basket) and did not run again before ``b``."""
        stop = next((t for t in stops if a - 3.0 <= t <= a + 6.0), None)
        return stop is not None and not started_between(stop, b)

    groups: list[list[BugEvent]] = []
    for s in (s for s in scores if s.points == 1):
        g = groups[-1] if groups else None
        if g is not None and g[-1].team is s.team and not started_between(g[-1].t, s.t) and s.t - g[-1].t < 90.0:
            g.append(s)
        else:
            groups.append([s])
    for g in groups:
        team, k = g[0].team, len(g)
        basket = [s for s in scores if s.points >= 2 and s.team is team and 0.0 < g[0].t - s.t <= cfg.and_one_window_s]
        and_one = k == 1 and bool(basket) and stopped_since(basket[-1].t - cfg.score_lag_s, g[0].t)
        size = 1 if and_one else max(2, k)
        lo = g[0].t - cfg.score_lag_s - 20.0
        hi = g[-1].t + 25.0
        misses = [e for e in by_id.values() if e.kind is EventKind.SHOT_MISSED and _is_ft(e) and e.team is team
                  and lo <= e.t <= hi]
        want = size - k
        for e in sorted(misses, key=lambda e: e.t)[want:]:  # more misses than the trip has room for
            by_id.pop(e.event_id, None)
            if e.shot_id:
                for a in [a for a in by_id.values() if a.kind is EventKind.SHOT_ATTEMPT and a.shot_id == e.shot_id]:
                    by_id.pop(a.event_id, None)
        for j in range(max(0, want - len(misses))):
            add_shot(g[-1].t - cfg.score_lag_s + 8.0 * (j + 1), team, 1, False, "free-throw trip short of makes")


def _rimless_misses(by_id: dict[str, Event], attempts: dict[str, Event], bug: list[BugEvent], cfg: FusionConfig,
                    running: Any) -> None:
    """Drop the engine's missed field goals the scorebug shows no sign of (pass 2c): a miss that reaches the rim
    resets the shot clock, the next possession shows on it (a count from 24, a reset to 14), a put-back scores, and a
    foul on the shot or the ball out of bounds stops the game clock. With the game clock running and the shot clock
    read around the miss (a shot-clock event within 24 s on either side: a possession cannot run longer; not with
    under 24 s on the game clock, when the shot clock is off), none of these within ``reset_window_s`` of its result
    leaves the engine's call (a pass, a dribble taken for a release) unsupported: the miss and what hangs on it (its
    attempt, the engine's rebound, block and assist of it) are dropped. A blocked shot stops before the rim and
    shows nothing: it is kept."""
    reads = sorted(b.t for b in bug if b.kind in ("reset24", "reset14", "run24"))
    signs = sorted(b.t for b in bug if b.kind in ("reset24", "reset14", "run24", "score", "clock_stop"))
    clocks = sorted((b.t, b.game_clock) for b in bug if b.game_clock is not None)
    lo, hi = cfg.reset_window_s

    def clock_off(t: float) -> bool:
        before = [g for tb, g in clocks if tb <= t]
        return bool(before) and before[-1] < 24.0

    for e in [e for e in by_id.values() if e.kind is EventKind.SHOT_MISSED and not _is_ft(e)
              and e.source != "scoreboard" and not e.attrs.get("inferred")
              and not e.attrs.get("blocked") and not e.attrs.get("blocked_oob")]:
        if not running(e.t) or clock_off(e.t) or not any(e.t - 24.0 <= t < e.t for t in reads) \
                or not any(e.t < t <= e.t + 24.0 for t in reads):
            continue  # a dead ball, or the shot clock off or not read here
        if any(-hi <= t - e.t <= -lo for t in signs):
            continue
        by_id.pop(e.event_id, None)
        if e.shot_id:
            for x in [x for x in by_id.values() if x.shot_id == e.shot_id and x.kind in
                      (EventKind.SHOT_ATTEMPT, EventKind.REBOUND, EventKind.BLOCK, EventKind.ASSIST)]:
                by_id.pop(x.event_id, None)
            attempts.pop(e.shot_id, None)


def _flight_values(by_id: dict[str, Event], attempts: dict[str, Event], flights: list[Any], court: Any,
                   cfg: FusionConfig) -> None:
    """2 or 3 points for missed field goals from the release point of the 3-D flight that reaches a rim near them
    (the flight followed back to ``release_z_ft``: the detector loses the ball at the release, so its first
    observation is already on the way to the rim)."""
    from courtvision.perception.ball3d import release_point, rim_pass

    baskets = {side: court.basket_xy(side) for side in ("left", "right")}
    passes = []
    for f in flights:
        rp = rim_pass(f, baskets, court.rim_height, court.rim_radius)
        if rp is not None and rp.dist_ft <= 3.0:
            passes.append((rp.t, rp.side, f))
    for e in list(by_id.values()):
        if e.kind is not EventKind.SHOT_MISSED or _is_ft(e):
            continue
        near = [p for p in passes if abs(p[0] - e.t) <= cfg.flight_match_s]
        if not near:
            continue
        _, side, f = min(near, key=lambda p: abs(p[0] - e.t))
        x, y = release_point(f, cfg.release_z_ft)
        value = 3 if _arc_margin(court, x, y, side) >= cfg.three_margin_ft else 2
        if value != e.value or e.x is None:
            by_id[e.event_id] = _replace(e, value=value, x=x, y=y, attrs={"value_from": "3-D flight"})
            att = attempts.get(e.shot_id or "")
            if att is not None:
                cur = by_id.get(att.event_id, att)
                by_id[att.event_id] = _replace(cur, value=value, x=x, y=y, attrs={"value_from": "3-D flight"})


def _handler_values(events: list[Event], cfg: FusionConfig, court: Any, attacker: Any) -> list[Event]:
    """2 or 3 points for inferred misses from the shooting team's last ball position before the release (pass 7):
    the dribbler's or the new possessor's position (not after a pass: its event carries the passer's); the basket
    is the one the camera says that team attacks, else the nearer one."""
    out = list(events)
    held = sorted((e for e in out if e.kind in (EventKind.POSSESSION, EventKind.DRIBBLE, EventKind.PASS)
                   and e.team is not None and e.x is not None and e.y is not None
                   and not e.attrs.get("inferred")), key=lambda e: e.t)
    atts = {e.shot_id: e for e in out if e.kind is EventKind.SHOT_ATTEMPT and e.shot_id}
    for i, e in enumerate(out):
        if e.kind is not EventKind.SHOT_MISSED or _is_ft(e) or not e.attrs.get("inferred") or e.team is None \
                or e.attrs.get("value_from"):
            continue
        att = atts.get(e.shot_id or "")
        t_rel = att.t if att is not None else e.t - cfg.inferred_release_s
        h = [p for p in held if p.team is e.team and 0.0 <= t_rel - p.t <= cfg.handler_window_s]
        if not h or h[-1].kind is EventKind.PASS:
            continue  # after a pass the shooter is its receiver, whose position the pass does not carry
        x, y = float(h[-1].x), float(h[-1].y)  # type: ignore[arg-type]
        team, side = attacker(t_rel)
        if side is None or team is not e.team:
            side = min(("left", "right"), key=lambda sd: math.hypot(x - court.basket_xy(sd)[0], y - court.basket_xy(sd)[1]))
        value = 3 if court.is_three(x, y, side) else 2
        out[i] = _replace(e, value=value, x=x, y=y, attrs={"value_from": "ball handler"})
        if att is not None:
            j = out.index(att)
            out[j] = _replace(att, value=value, x=x, y=y, attrs={"value_from": "ball handler"})
    return out


def _arc_margin(court: Any, x: float, y: float, side: str) -> float:
    """How far (ft) the point lies beyond the 3-point line (negative inside)."""
    bx, by = court.basket_xy(side)
    lateral = abs(y - by)
    if lateral >= court.corner_three_dist and abs(x - bx) <= court.three_junction_along:
        return lateral - court.corner_three_dist
    return math.hypot(x - bx, y - by) - court.three_pt_radius
