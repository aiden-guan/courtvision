"""Box score from an event log: counting stats, shot records, possessions and context points.

``build_box`` is a pure function of the *active* events (latest revision per ``event_id``,
RETRACTED removed). ``StatsEngine`` calls it on every snapshot, so a revision or retraction is
reflected immediately and nothing is ever accumulated blindly.

Crediting follows ARCHITECTURE "Shared event semantics" and the FIBA Statisticians' Manual grammar
(``docs/research/basketball_rules_stats.md`` §2a):

* A shot is keyed by ``shot_id`` (attempt + result). Free throws (``shot_type == "free_throw"`` or
  value 1) count as FTA/FTM, never FGA. A made shot without an attempt still counts as an
  attempt (an FGM is always an FGA); a BLOCK without an attempt creates one (no FGA, no block).
* Team rebounds and team turnovers (``player_id is None``) are credited to the team only
  (``team_oreb`` / ``team_dreb`` / ``team_tov``). Team counting stats are team *totals*: they
  include these team-level entries, so ``TeamLine.oreb`` is the OREB the possession estimate needs.
* SCORE_CORRECTION adds unattributed team points (``pts_unattributed``).

Possessions are counted from events (rules note §2b): a possession ends on a made FG or last made
FT (resolved lazily so an and-one keeps the possession), a defensive rebound, a turnover, or the
period end; an offensive rebound continues it. Each team possession is counted when it starts.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any

from courtvision.stats import advanced
from courtvision.types import Event, EventKind, EventStatus, Roster, Team

if TYPE_CHECKING:
    from courtvision.geometry.court import CourtSpec
    from courtvision.stats.engine import StatsConfig

__all__ = [
    "PlayerLine",
    "TeamLine",
    "ShotRecord",
    "Possession",
    "ScoringPlay",
    "BoxScore",
    "TeamResolver",
    "build_box",
    "refresh_team_rates",
    "order_events",
    "is_free_throw_attempt",
]

log = logging.getLogger(__name__)

XfgFn = Callable[["ShotRecord"], "float | None"]

# Same-t ordering: what must happen first when two events share a timestamp.
_KIND_ORDER: dict[EventKind, int] = {
    EventKind.PERIOD_START: 0,
    EventKind.SUBSTITUTION: 1,
    EventKind.TIMEOUT: 1,
    EventKind.POSSESSION: 2,
    EventKind.PASS: 3,
    EventKind.DRIBBLE: 3,
    EventKind.FOUL: 4,
    EventKind.SHOT_ATTEMPT: 5,
    EventKind.BLOCK: 6,
    EventKind.SHOT_MADE: 7,
    EventKind.SHOT_MISSED: 7,
    EventKind.ASSIST: 8,
    EventKind.SCORE_CORRECTION: 8,
    EventKind.REBOUND: 9,
    EventKind.TURNOVER: 10,
    EventKind.STEAL: 11,
    EventKind.PERIOD_END: 12,
}

_TECHNICAL_FT_TYPES = frozenset({"technical", "bench_technical", "coach_technical"})


# --------------------------------------------------------------------------- data model


def _clean(v: Any) -> Any:
    """JSON-safe scalar: enums -> values, non-finite floats -> None."""
    if isinstance(v, Team):
        return v.value
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _from_fields(cls: type, d: dict[str, Any]) -> dict[str, Any]:
    names = {f.name for f in fields(cls)}
    out = {k: v for k, v in d.items() if k in names}
    if "team" in out and out["team"] is not None and not isinstance(out["team"], Team):
        out["team"] = Team(out["team"])
    return out


@dataclass
class PlayerLine:
    """Counting stats for one player (derived rates are properties)."""

    player_id: str
    team: Team | None
    pts: int = 0
    fgm: int = 0
    fga: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    oreb: int = 0
    dreb: int = 0
    ast: int = 0
    stl: int = 0
    blk: int = 0
    tov: int = 0
    pf: int = 0
    plus_minus: int = 0  # operator lineup entries only (0 without them)
    seconds: float = 0.0  # operator lineup entries only (0 without them)
    xpts: float = 0.0  # sum of expected points of attempts
    tf: int = 0  # technical fouls (kept out of pf; NFHS foul-out counts them, RuleSpec)
    provisional: bool = False  # True while any contributing event is not CONFIRMED

    @property
    def reb(self) -> int:
        return self.oreb + self.dreb

    @property
    def fg_pct(self) -> float | None:
        return advanced.pct(self.fgm, self.fga)

    @property
    def fg3_pct(self) -> float | None:
        return advanced.pct(self.fg3m, self.fg3a)

    @property
    def ft_pct(self) -> float | None:
        return advanced.pct(self.ftm, self.fta)

    @property
    def efg(self) -> float | None:
        return advanced.efg(self.fgm, self.fg3m, self.fga)

    @property
    def ts(self) -> float | None:
        return advanced.ts(self.pts, self.fga, self.fta)

    def to_dict(self) -> dict[str, Any]:
        d = {k: _clean(v) for k, v in asdict(self).items()}
        for name in ("reb", "fg_pct", "fg3_pct", "ft_pct", "efg", "ts"):
            d[name] = _clean(getattr(self, name))
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlayerLine:
        return cls(**_from_fields(cls, d))


@dataclass
class TeamLine:
    """Team totals (team-level rebounds / turnovers / corrections included) plus team rates."""

    team: Team
    pts: int = 0
    fgm: int = 0
    fga: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    oreb: int = 0  # includes team_oreb
    dreb: int = 0  # includes team_dreb
    ast: int = 0
    stl: int = 0
    blk: int = 0
    tov: int = 0  # includes team_tov
    pf: int = 0
    plus_minus: int = 0  # game margin (pts - opponent pts, both from events)
    seconds: float = 0.0  # elapsed game seconds used for pace
    xpts: float = 0.0
    tf: int = 0
    team_reb: int = 0  # unattributed (player None) rebounds = team_oreb + team_dreb
    team_oreb: int = 0
    team_dreb: int = 0
    team_tov: int = 0
    pts_unattributed: int = 0  # SCORE_CORRECTION points and made shots with no known shooter
    possessions: int = 0  # counted from events
    possessions_est: float = 0.0  # FGA - OREB + TOV + c * FTA
    ppp: float | None = None
    pace: float | None = None
    efg: float | None = None
    ts: float | None = None
    ortg: float | None = None
    drtg: float | None = None
    tov_pct: float | None = None
    oreb_pct: float | None = None
    ft_rate: float | None = None
    paint_pts: int = 0
    second_chance_pts: int = 0
    fast_break_pts: int = 0
    pts_off_turnovers: int = 0
    avg_xfg: float | None = None  # mean xFG over field-goal attempts with an xFG
    n_xfg: int = 0  # sample size of avg_xfg
    provisional: bool = False

    @property
    def reb(self) -> int:
        return self.oreb + self.dreb

    def to_dict(self) -> dict[str, Any]:
        d = {k: _clean(v) for k, v in asdict(self).items()}
        d["reb"] = self.reb
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TeamLine:
        return cls(**_from_fields(cls, d))


@dataclass
class ShotRecord:
    """One field-goal or free-throw attempt with its result (``made is None`` while pending)."""

    shot_id: str
    player_id: str | None
    team: Team | None
    t: float  # release time (result time when no attempt event exists)
    x: float | None
    y: float | None
    value: int  # 1 (FT), 2 or 3
    made: bool | None
    distance_ft: float | None
    zone: str | None
    shot_type: str
    xfg: float | None
    closest_defender_ft: float | None
    assisted_by: str | None
    blocked_by: str | None = None
    basket: str | None = None
    catch_and_shoot: bool | None = None
    points: int = 0
    t_result: float | None = None
    provisional: bool = True

    @property
    def is_ft(self) -> bool:
        return self.shot_type == "free_throw"

    def to_dict(self) -> dict[str, Any]:
        return {k: _clean(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ShotRecord:
        return cls(**_from_fields(cls, d))


@dataclass
class Possession:
    """One team possession inferred from the event log."""

    index: int
    team: Team
    start_t: float
    cause: str  # types.py POSSESSION causes, plus "other" when inferred
    period: int | None = None
    end_t: float | None = None
    end_reason: str | None = None
    points: int = 0
    oreb: int = 0


@dataclass
class ScoringPlay:
    t: float
    team: Team
    points: int
    event_id: str
    player_id: str | None = None


@dataclass
class BoxScore:
    """Everything ``build_box`` derives from one event log."""

    players: dict[str, PlayerLine]
    teams: dict[Team, TeamLine]
    shots: list[ShotRecord]
    possessions: list[Possession]
    scoring: list[ScoringPlay]
    period: int | None = None
    last_t: float | None = None


# --------------------------------------------------------------------------- helpers


class TeamResolver:
    """Team of a player id: roster first, then the ``home-`` / ``away-`` id prefix (types.py)."""

    def __init__(self, roster: Roster | None) -> None:
        self._roster = roster

    def of(self, player_id: str | None, fallback: Team | None = None) -> Team | None:
        if player_id is None:
            return fallback
        if self._roster is not None and (team := self._roster.team_of(player_id)) is not None:
            return team
        prefix = player_id.split("-", 1)[0].lower()
        if prefix in (Team.HOME.value, Team.AWAY.value):
            return Team(prefix)
        return fallback


def order_events(events: Iterable[Event]) -> list[Event]:
    """Sort by time, then same-t kind order, then input order (stable)."""
    indexed = list(enumerate(events))
    indexed.sort(key=lambda p: (p[1].t, _KIND_ORDER.get(p[1].kind, 5), p[0]))
    return [e for _, e in indexed]


def is_free_throw_attempt(e: Event) -> bool:
    return e.attrs.get("shot_type") == "free_throw" or e.value == 1


def _is_technical_ft(e: Event) -> bool:
    return bool(e.attrs.get("technical")) or e.attrs.get("ft_type") in _TECHNICAL_FT_TYPES


def _confirmed(e: Event) -> bool:
    return e.status is EventStatus.CONFIRMED


def _opt_float(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _finite_xy(x: Any, y: Any) -> tuple[float | None, float | None]:
    """Both coordinates as finite floats, else (None, None): CourtSpec raises on nan / inf."""
    fx, fy = _opt_float(x), _opt_float(y)
    return (fx, fy) if fx is not None and fy is not None else (None, None)


def _keeps_possession_ft(e: Event, kept_types: Iterable[str]) -> bool:
    """FT of a flagrant / disruptive trip: the shooter's team keeps the ball (rules note §8b)."""
    return e.attrs.get("ft_type") in tuple(kept_types)


@dataclass
class _ShotBuild:
    record: ShotRecord
    events: list[Event] = field(default_factory=list)
    technical: bool = False
    keeps_possession: bool = False  # flagrant / disruptive FT: no possession change on the make
    has_attempt: bool = False
    attempt_event: Event | None = None
    result_event: Event | None = None
    result_rank: tuple[int, float, int] = (-1, -math.inf, -1)


def _contributing(build: _ShotBuild) -> list[Event]:
    """Events a shot's line credit rests on: its attempt and winning result (a superseded result
    or a duplicate attempt does not make a line provisional); a block-only shot rests on the block."""
    evs = [e for e in (build.attempt_event, build.result_event) if e is not None]
    return evs or build.events


# --------------------------------------------------------------------------- shot collection


class _ShotCollector:
    """Pass 1: join attempts, results and blocks into shot records."""

    def __init__(self, court: CourtSpec | None, teams: TeamResolver, kept_ft_types: Iterable[str] = ()) -> None:
        self.court = court
        self.teams = teams
        self.kept_ft_types = tuple(kept_ft_types)
        self.shots: dict[str, _ShotBuild] = {}

    def collect(self, ordered: list[Event]) -> dict[str, _ShotBuild]:
        for i, e in enumerate(ordered):
            if e.kind is EventKind.SHOT_ATTEMPT:
                self._attempt(e)
            elif e.kind in (EventKind.SHOT_MADE, EventKind.SHOT_MISSED):
                self._result(e, i)
            elif e.kind is EventKind.BLOCK:
                self._block(e)
        return self.shots

    def _geometry(self, x: float | None, y: float | None, attrs: dict[str, Any], is_ft: bool) -> tuple:
        basket = attrs.get("basket")
        distance = _opt_float(attrs.get("distance_ft"))
        zone = attrs.get("zone")
        if self.court is not None and x is not None and y is not None:
            if basket not in ("left", "right"):
                basket = self.court.nearest_basket(x, y)
            if distance is None:
                distance = self.court.distance_to_basket(x, y, basket)
            if zone is None and not is_ft:
                zone = self.court.zone(x, y, basket)
        return (basket if basket in ("left", "right") else None), distance, zone

    def _value(self, e: Event, is_ft: bool, x: float | None, y: float | None, basket: str | None) -> int:
        if is_ft:
            return 1
        if e.value in (2, 3):
            return e.value
        # Fallback only: D8 evaluates is_three at the last grounded foot before the gather
        # (attrs gather_x / gather_y from the event engine), else at the release point.
        gx, gy = _finite_xy(e.attrs.get("gather_x"), e.attrs.get("gather_y"))
        if gx is not None:
            x, y = gx, gy
        if self.court is not None and x is not None and y is not None and basket is not None:
            return 3 if self.court.is_three(x, y, basket) else 2  # type: ignore[arg-type]
        return 2

    def _new(self, key: str, e: Event, player_id: str | None, team: Team | None, value_hint: int) -> _ShotBuild:
        is_ft = e.attrs.get("shot_type") == "free_throw" or value_hint == 1
        x, y = _finite_xy(e.x, e.y)
        basket, distance, zone = self._geometry(x, y, e.attrs, is_ft)
        rec = ShotRecord(
            shot_id=key,
            player_id=player_id,
            team=team,
            t=e.t,
            x=x,
            y=y,
            value=1 if is_ft else (value_hint if value_hint in (2, 3) else self._value(e, False, x, y, basket)),
            made=None,
            distance_ft=distance,
            zone=zone,
            shot_type="free_throw" if is_ft else str(e.attrs.get("shot_type") or "jumper"),
            xfg=_opt_float(e.attrs.get("xfg")),
            closest_defender_ft=_opt_float(e.attrs.get("closest_defender_ft")),
            assisted_by=None,
            basket=basket,
            catch_and_shoot=cas if isinstance(cas := e.attrs.get("catch_and_shoot"), bool) else None,
        )
        build = _ShotBuild(record=rec, technical=is_ft and _is_technical_ft(e),
                           keeps_possession=is_ft and _keeps_possession_ft(e, self.kept_ft_types))
        self.shots[key] = build
        return build

    def _attempt(self, e: Event) -> None:
        key = e.shot_id or f"evt:{e.event_id}"
        existing = self.shots.get(key)
        if existing is not None and existing.has_attempt:
            log.debug("duplicate SHOT_ATTEMPT for %s (%s); keeping the first", key, e.event_id)
            existing.events.append(e)
            return
        is_ft = is_free_throw_attempt(e)
        team = self.teams.of(e.player_id, e.team)
        x, y = _finite_xy(e.x, e.y)
        basket, _, _ = self._geometry(x, y, e.attrs, is_ft)
        value = self._value(e, is_ft, x, y, basket)
        build = self._new(key, e, e.player_id, team, value)
        if existing is not None:  # a result or block arrived first (same t, implicit shot)
            build.events = existing.events
            build.result_event, build.result_rank = existing.result_event, existing.result_rank
            build.record.made, build.record.points = existing.record.made, existing.record.points
            build.record.t_result, build.record.blocked_by = existing.record.t_result, existing.record.blocked_by
        build.has_attempt = True
        build.attempt_event = e
        build.events.append(e)

    def _result(self, e: Event, order: int) -> None:
        key = e.shot_id or f"evt:{e.event_id}"
        build = self.shots.get(key)
        made = e.kind is EventKind.SHOT_MADE
        if build is None:
            hint = e.value if e.value in (1, 2, 3) else 0  # 0: value from the court geometry, else 2
            build = self._new(key, e, e.player_id, self.teams.of(e.player_id, e.team), hint)
        build.events.append(e)
        rank = (1 if _confirmed(e) else 0, e.t, order)
        if build.result_event is not None:
            log.debug("shot %s has several results (%s, %s); keeping the stronger",
                      key, build.result_event.event_id, e.event_id)
        if rank <= build.result_rank:
            return
        build.result_event, build.result_rank = e, rank
        rec = build.record
        rec.made = made
        rec.t_result = e.t
        if made:
            rec.points = e.value if e.value > 0 else rec.value
            if not rec.is_ft and rec.points in (2, 3):
                rec.value = rec.points  # the result (e.g. reconciled against the scoreboard) wins
        else:
            rec.points = 0

    def _block(self, e: Event) -> None:
        build = self.shots.get(e.shot_id) if e.shot_id else self._recent_attempt(e)
        if build is None:
            key = e.shot_id or f"blk:{e.event_id}"
            blocker_team = self.teams.of(e.player_id, e.team)
            shooter_team = self.teams.of(e.other_player_id, blocker_team.other() if blocker_team else None)
            build = self._new(key, e, e.other_player_id, shooter_team, e.value if e.value in (2, 3) else 2)
            build.record.x = build.record.y = None
            build.has_attempt = True
        build.record.blocked_by = e.player_id
        build.events.append(e)

    def _recent_attempt(self, e: Event, window_s: float = 3.0) -> _ShotBuild | None:
        best: _ShotBuild | None = None
        for b in self.shots.values():
            r = b.record
            if b.has_attempt and r.player_id == e.other_player_id and not r.is_ft and 0 <= e.t - r.t <= window_s:
                if best is None or r.t > best.record.t:
                    best = b
        return best


# --------------------------------------------------------------------------- the walk


def _assist_winners(ordered: list[Event]) -> dict[str, str]:
    """One assist per score (FIBA [15]): per shot_id, a CONFIRMED assist beats a provisional one,
    then the later one wins (a re-derived assist supersedes the first guess)."""
    best: dict[str, tuple[tuple[int, float, int], str]] = {}
    for i, e in enumerate(ordered):
        if e.kind is EventKind.ASSIST and e.shot_id is not None:
            rank = (1 if _confirmed(e) else 0, e.t, i)
            if e.shot_id not in best or rank > best[e.shot_id][0]:
                best[e.shot_id] = (rank, e.event_id)
    return {sid: eid for sid, (_, eid) in best.items()}


class _BoxWalker:
    """Pass 2: credit counting stats and run the possession state machine in time order."""

    def __init__(
        self,
        shots: dict[str, _ShotBuild],
        teams: TeamResolver,
        config: StatsConfig,
        roster: Roster | None,
    ) -> None:
        self.cfg = config
        self.teams = teams
        self.shots = shots
        self.result_ids = {b.result_event.event_id: b for b in shots.values() if b.result_event is not None}
        self.attempt_ids = {b.attempt_event.event_id: b for b in shots.values() if b.attempt_event is not None}
        self.players: dict[str, PlayerLine] = {}
        self.team_lines = {t: TeamLine(team=t) for t in Team}
        self.possessions: list[Possession] = []
        self.scoring: list[ScoringPlay] = []
        self.offense: Team | None = None
        self.current: Possession | None = None
        self.pending: tuple[Team, float, str] | None = None  # (scoring team, t, next cause)
        self.last_miss_team: Team | None = None
        self.period: int | None = None
        self.assist_winner: dict[str, str] = {}  # shot_id -> the one ASSIST event credited
        if roster is not None and config.include_roster_players:
            for entry in roster.entries:
                self.players[entry.player_id] = PlayerLine(player_id=entry.player_id, team=entry.team)

    # ---------------------------------------------------------------- crediting

    def _line(self, player_id: str, team: Team | None) -> PlayerLine:
        line = self.players.get(player_id)
        if line is None:
            line = self.players[player_id] = PlayerLine(player_id=player_id, team=team)
        elif line.team is None and team is not None:
            line.team = team
        return line

    def _credit(self, e: Event, player_id: str | None, team: Team | None, stat: str, n: int = 1) -> None:
        provisional = not _confirmed(e)
        if team is not None:
            tl = self.team_lines[team]
            setattr(tl, stat, getattr(tl, stat) + n)
            tl.provisional |= provisional
        if player_id is not None:
            line = self._line(player_id, team)
            setattr(line, stat, getattr(line, stat) + n)
            line.provisional |= provisional

    # ---------------------------------------------------------------- possessions

    def _end(self, t: float, reason: str) -> None:
        if self.current is not None and self.current.end_t is None:
            self.current.end_t = t
            self.current.end_reason = reason
        self.current = None
        self.offense = None

    def _start(self, team: Team, t: float, cause: str) -> Possession:
        self._end(t, f"change:{cause}")
        poss = Possession(index=len(self.possessions), team=team, start_t=t, cause=cause, period=self.period)
        self.possessions.append(poss)
        self.current, self.offense = poss, team
        self.team_lines[team].possessions += 1
        return poss

    def _resolve_pending(self, start_next: bool = True) -> None:
        """Close the possession a made FG / made FT ended (no and-one FT followed)."""
        if self.pending is None:
            return
        team, t, cause = self.pending
        self.pending = None
        self._end(t, cause)
        if start_next:
            self._start(team.other(), t, cause)

    def _ensure(self, team: Team, t: float, cause: str = "other") -> None:
        if self.offense is not team:
            self._resolve_pending()
            if self.offense is not team:
                self._start(team, t, cause)

    # ---------------------------------------------------------------- handlers

    def walk(self, ordered: list[Event]) -> None:
        handlers: dict[EventKind, Callable[[Event], None]] = {
            EventKind.SHOT_ATTEMPT: self._on_attempt,
            EventKind.SHOT_MADE: self._on_result,
            EventKind.SHOT_MISSED: self._on_result,
            EventKind.BLOCK: self._on_block,
            EventKind.REBOUND: self._on_rebound,
            EventKind.ASSIST: self._on_assist,
            EventKind.STEAL: self._on_steal,
            EventKind.TURNOVER: self._on_turnover,
            EventKind.FOUL: self._on_foul,
            EventKind.POSSESSION: self._on_possession,
            EventKind.PASS: self._on_ball_action,
            EventKind.DRIBBLE: self._on_ball_action,
            EventKind.SCORE_CORRECTION: self._on_correction,
            EventKind.PERIOD_START: self._on_period_start,
            EventKind.PERIOD_END: self._on_period_end,
        }
        self.assist_winner = _assist_winners(ordered)
        for e in ordered:
            handler = handlers.get(e.kind)
            if handler is not None:
                handler(e)

    def _on_attempt(self, e: Event) -> None:
        build = self.attempt_ids.get(e.event_id)
        if build is None:
            return  # duplicate attempt event for an already-seen shot
        if build.technical or build.record.team is None:
            return
        team = build.record.team
        if build.record.is_ft and self.pending is not None and self.pending[0] is team:
            self.pending = None  # and-one, or the next FT of the trip: same possession
            return
        self._ensure(team, e.t, "free_throw" if build.record.is_ft else "other")

    def _on_result(self, e: Event) -> None:
        build = self.result_ids.get(e.event_id)
        if build is None:
            return  # superseded result for a shot with a stronger result event
        rec = build.record
        if not rec.made:
            if not (build.technical or build.keeps_possession):  # those trips have no rebound
                self.last_miss_team = rec.team
            return
        self.last_miss_team = None
        if rec.team is None:
            log.debug("made shot %s has no team; points dropped", rec.shot_id)
            return
        if rec.player_id is None:
            self.team_lines[rec.team].pts_unattributed += rec.points
        self.scoring.append(ScoringPlay(e.t, rec.team, rec.points, e.event_id, rec.player_id))
        if build.technical:
            return
        self._ensure(rec.team, rec.t)
        self._context_points(rec)
        if not build.keeps_possession:  # a flagrant / disruptive trip is followed by own throw-in
            self.pending = (rec.team, e.t, "free_throw" if rec.is_ft else "made_shot")

    def _context_points(self, rec: ShotRecord) -> None:
        assert rec.team is not None
        tl = self.team_lines[rec.team]
        pts = rec.points
        if not rec.is_ft and rec.zone in self.cfg.paint_zones:
            tl.paint_pts += pts
        poss = self.current
        if poss is None or poss.team is not rec.team:
            return
        poss.points += pts
        if poss.oreb > 0:
            tl.second_chance_pts += pts
        if poss.cause in self.cfg.pts_off_turnover_causes:
            tl.pts_off_turnovers += pts
        if (
            not rec.is_ft
            and poss.cause in self.cfg.fast_break_causes
            and rec.t - poss.start_t <= self.cfg.fast_break_window_s
        ):
            tl.fast_break_pts += pts

    def _on_block(self, e: Event) -> None:
        self._credit(e, e.player_id, self.teams.of(e.player_id, e.team), "blk")

    def _on_rebound(self, e: Event) -> None:
        team = self.teams.of(e.player_id, e.team)
        if team is None:
            log.debug("rebound %s has no team; ignored", e.event_id)  # debug: runs every snapshot
            return
        flag = e.attrs.get("offensive")
        offensive = flag if isinstance(flag, bool) else (self.last_miss_team is team)
        self._credit(e, e.player_id, team, "oreb" if offensive else "dreb")
        if e.player_id is None:
            self._credit(e, None, team, "team_oreb" if offensive else "team_dreb")
            self.team_lines[team].team_reb += 1
        self.last_miss_team = None
        self._resolve_pending()
        if self.offense is not team:
            self._start(team, e.t, "offensive_rebound" if offensive else "defensive_rebound")
        if offensive and self.current is not None:
            self.current.oreb += 1

    def _on_assist(self, e: Event) -> None:
        if e.shot_id is not None and (build := self.shots.get(e.shot_id)) is not None:
            if self.cfg.assist_requires_made_shot and build.record.made is not True:
                log.debug("assist %s on a shot that is not made; not credited", e.event_id)
                return
            if self.assist_winner.get(e.shot_id) != e.event_id:
                log.debug("assist %s: shot %s already has an assist; not credited", e.event_id, e.shot_id)
                return
            build.record.assisted_by = e.player_id
        self._credit(e, e.player_id, self.teams.of(e.player_id, e.team), "ast")

    def _on_steal(self, e: Event) -> None:
        team = self.teams.of(e.player_id, e.team)
        self._credit(e, e.player_id, team, "stl")
        if team is None:
            return
        self._resolve_pending()
        cur = self.current
        if self.offense is team and cur is not None and e.t - cur.start_t > self.cfg.steal_link_s:
            self._start(team.other(), e.t, "other")  # the victim's team had a possession we never saw
            cur = self.current
        if self.offense is not team:
            self._start(team, e.t, "steal")
        elif cur is not None and cur.cause == "turnover" and e.t - cur.start_t <= self.cfg.steal_link_s:
            cur.cause = "steal"

    def _on_turnover(self, e: Event) -> None:
        team = self.teams.of(e.player_id, e.team)
        self._credit(e, e.player_id, team, "tov")
        if e.player_id is None and team is not None:
            self._credit(e, None, team, "team_tov")
        self.last_miss_team = None
        if team is None:
            return
        self._resolve_pending()
        cur = self.current
        linked = cur is not None and self.offense is team.other() and e.t - cur.start_t <= self.cfg.steal_link_s
        if self.offense is None or (self.offense is team.other() and not linked):
            self._start(team, e.t, "other")  # the committing team had the ball (unseen possession)
        if self.offense is team:
            self._start(team.other(), e.t, "turnover")

    def _on_foul(self, e: Event) -> None:
        kind = str(e.attrs.get("type") or e.attrs.get("foul_type") or "")
        stat = "tf" if kind in self.cfg.technical_foul_types else "pf"
        self._credit(e, e.player_id, self.teams.of(e.player_id, e.team), stat)

    def _on_possession(self, e: Event) -> None:
        team = e.team or self.teams.of(e.player_id)
        if team is None:
            return
        self._resolve_pending()
        if self.offense is not team:
            self._start(team, e.t, str(e.attrs.get("cause") or "other"))

    def _on_ball_action(self, e: Event) -> None:
        team = self.teams.of(e.player_id, e.team)
        if team is None:
            return
        if self.pending is not None and self.pending[0] is not team:
            self._resolve_pending()  # the other team has the ball: the scoring possession is over
        if self.offense is None and self.pending is None:
            self._start(team, e.t, "other")

    def _on_correction(self, e: Event) -> None:
        team = e.team or self.teams.of(e.player_id)
        if team is None or e.value == 0:
            return
        tl = self.team_lines[team]
        tl.pts += e.value
        tl.pts_unattributed += e.value
        tl.provisional |= not _confirmed(e)
        self.scoring.append(ScoringPlay(t=e.t, team=team, points=e.value, event_id=e.event_id))

    def _on_period_start(self, e: Event) -> None:
        self._resolve_pending(start_next=False)
        self._end(e.t, "period_start")
        period = e.attrs.get("period", e.value)
        self.period = period if isinstance(period, int) and period > 0 else (self.period or 0) + 1
        self.last_miss_team = None

    def _on_period_end(self, e: Event) -> None:
        self._resolve_pending(start_next=False)
        self._end(e.t, "period_end")
        self.last_miss_team = None

    # ---------------------------------------------------------------- shots -> lines

    def credit_shots(self, xfg_fn: XfgFn | None) -> list[ShotRecord]:
        records: list[ShotRecord] = []
        for build in self.shots.values():
            rec = build.record
            rec.provisional = any(not _confirmed(ev) for ev in _contributing(build))
            if rec.xfg is None and xfg_fn is not None:
                rec.xfg = xfg_fn(rec)
            self._credit_shot(build)
            records.append(rec)
        records.sort(key=lambda r: (r.t, r.shot_id))
        return records

    def _credit_shot(self, build: _ShotBuild) -> None:
        rec = build.record
        ev = build.result_event or build.events[0]
        pid, team = rec.player_id, rec.team
        made = rec.made is True
        if rec.is_ft:
            self._credit(ev, pid, team, "fta")
            if made:
                self._credit(ev, pid, team, "ftm")
        else:
            self._credit(ev, pid, team, "fga")
            if rec.value == 3:
                self._credit(ev, pid, team, "fg3a")
            if made:
                self._credit(ev, pid, team, "fgm")
                if rec.value == 3:
                    self._credit(ev, pid, team, "fg3m")
        if made:
            self._credit(ev, pid, team, "pts", rec.points)
        if any(not _confirmed(e) for e in _contributing(build)):
            if pid is not None:
                self._line(pid, team).provisional = True
            if team is not None:
                self.team_lines[team].provisional = True
        if rec.xfg is not None and (self.cfg.xpts_include_ft or not rec.is_ft):
            xp = rec.xfg * rec.value
            if pid is not None:
                self._line(pid, team).xpts += xp
            if team is not None:
                self.team_lines[team].xpts += xp


# --------------------------------------------------------------------------- team rates


def _finish_teams(
    lines: dict[Team, TeamLine],
    shots: list[ShotRecord],
    config: StatsConfig,
    elapsed_s: float | None,
    game_minutes: float,
) -> None:
    def poss_basis(tl: TeamLine) -> float:
        return float(tl.possessions) if tl.possessions > 0 else max(tl.possessions_est, 0.0)

    for tl in lines.values():
        tl.possessions_est = advanced.possessions_est(tl.fga, tl.oreb, tl.tov, tl.fta, config.ft_poss_coef)
        tl.efg = advanced.efg(tl.fgm, tl.fg3m, tl.fga)
        tl.ts = advanced.ts(tl.pts, tl.fga, tl.fta, config.ts_ft_coef)
        tl.tov_pct = advanced.tov_pct(tl.tov, tl.fga, tl.fta, config.ts_ft_coef)
        tl.ft_rate = advanced.ft_rate(tl.fta, tl.fga)
        tl.seconds = float(elapsed_s or 0.0)
        xfgs = [s.xfg for s in shots if s.team is tl.team and not s.is_ft and s.xfg is not None]
        tl.n_xfg = len(xfgs)
        tl.avg_xfg = sum(xfgs) / len(xfgs) if xfgs else None
    for tl in lines.values():
        opp = lines[tl.team.other()]
        own_p, opp_p = poss_basis(tl), poss_basis(opp)
        tl.ppp = advanced.ppp(tl.pts, own_p)
        tl.ortg = advanced.ortg(tl.pts, own_p)
        tl.drtg = advanced.ortg(opp.pts, opp_p)
        tl.oreb_pct = advanced.oreb_pct(tl.oreb, opp.dreb)
        tl.plus_minus = tl.pts - opp.pts
        if elapsed_s is not None and elapsed_s >= config.pace_min_elapsed_s:
            tl.pace = advanced.pace(own_p, opp_p, elapsed_s / 60.0, game_minutes)
        else:
            tl.pace = None


def refresh_team_rates(
    box: BoxScore, config: StatsConfig, elapsed_s: float | None, game_minutes: float = 48.0
) -> None:
    """Recompute the time-dependent team rates (pace, seconds) of ``box`` in place.

    Counting stats only change with the event log; elapsed time changes every frame, so the
    engine rebuilds the box per event change and only refreshes these per frame.
    """
    _finish_teams(box.teams, box.shots, config, elapsed_s, game_minutes)


# --------------------------------------------------------------------------- entry point


def build_box(
    events: Iterable[Event],
    *,
    court: CourtSpec | None = None,
    roster: Roster | None = None,
    config: StatsConfig | None = None,
    xfg_fn: XfgFn | None = None,
    elapsed_s: float | None = None,
    game_minutes: float = 48.0,
) -> BoxScore:
    """Box score of the given *active* events (caller drops RETRACTED / superseded revisions).

    ``xfg_fn`` supplies an xFG for shots whose attempt carries no ``attrs["xfg"]``.
    ``elapsed_s`` (elapsed game seconds) and ``game_minutes`` (regulation length) feed pace.
    """
    if config is None:
        from courtvision.stats.engine import StatsConfig

        config = StatsConfig()
    ordered = order_events(e for e in events if e.status is not EventStatus.RETRACTED)
    teams = TeamResolver(roster)
    shots = _ShotCollector(court, teams, config.possession_kept_ft_types).collect(ordered)
    walker = _BoxWalker(shots, teams, config, roster)
    walker.walk(ordered)
    records = walker.credit_shots(xfg_fn)
    _finish_teams(walker.team_lines, records, config, elapsed_s, game_minutes)
    return BoxScore(
        players=walker.players,
        teams=walker.team_lines,
        shots=records,
        possessions=walker.possessions,
        scoring=walker.scoring,
        period=walker.period,
        last_t=ordered[-1].t if ordered else None,
    )
