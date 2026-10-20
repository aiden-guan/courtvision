"""StatsEngine: live box score, tracking, advanced stats, lineups and decision grades.

Contract: ARCHITECTURE.md "stats/". Data flow:

* ``on_events`` keeps the latest revision per ``event_id`` (RETRACTED kept in the log so a stale
  lower revision cannot resurrect it, but ignored by every stat). Counting stats, shots,
  possessions, runs and context points are recomputed from the log on ``snapshot``
  (``boxscore.build_box``), so they always agree with the latest revisions.
* ``on_frame`` folds frames into tracking stats, the elapsed-game-time timeline (pace and lineup
  seconds), the confirmed scoreboard and a ~10 s frame ring buffer used to grade PASS /
  SHOT_ATTEMPT decisions with the optional ``DecisionModel``.
* ``set_lineup`` / ``substitute`` are the ONLY source of lineups, player seconds and plus-minus.

``GameStats.score`` is the confirmed scoreboard when one exists (manual / feed / sim readings at
once; OCR after ``ocr_confirm_reads`` identical consecutive reads, D14), else the event-derived
team points; ``score_mismatch`` flags disagreement between the two.
"""

from __future__ import annotations

import enum
import logging
import math
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any

from courtvision.stats import advanced
from courtvision.stats.boxscore import (
    BoxScore,
    PlayerLine,
    ScoringPlay,
    ShotRecord,
    TeamLine,
    TeamResolver,
    build_box,
    is_free_throw_attempt,
    order_events,
    refresh_team_rates,
)
from courtvision.stats.lineups import GameClockTimeline, LineupTracker, compute_lineups
from courtvision.stats.tracking import TrackingAccumulator, TrackingLine, event_counts
from courtvision.types import Event, EventKind, EventStatus, FrameState, Roster, ScoreboardReading, Side, Team

if TYPE_CHECKING:
    from courtvision.geometry.court import CourtSpec
    from courtvision.geometry.rules import RuleSpec

__all__ = [
    "StatsConfig",
    "StatsEngine",
    "GameStats",
    "PlayerLine",
    "TeamLine",
    "TrackingLine",
    "ShotRecord",
    "scoring_run",
]

log = logging.getLogger(__name__)


@dataclass
class StatsConfig:
    """Stats tunables. Each constant notes its source and status (sourced / unverified / tuned)."""

    # ---- possessions and efficiency (rules note §2b)
    ft_poss_coef: float = advanced.FT_POSS_COEF  # 0.44, Kubatko et al. 2007 [40]; NBA PBP 0.42-0.43; sourced
    ts_ft_coef: float = advanced.TS_FT_COEF  # 0.44 in TS% [30][31]; sourced
    pace_min_elapsed_s: float = 60.0  # no pace before 1 game minute (pure noise); tuned
    # ---- context points
    paint_zones: tuple[str, ...] = ("restricted_area", "paint")  # CourtSpec.zone names (PITP [30]); sourced
    fast_break_window_s: float = 7.0  # release within 7 s of the possession start; FIBA allows max 8 s [15]; tuned
    fast_break_causes: tuple[str, ...] = ("defensive_rebound", "steal")  # transition starters (spec); tuned
    pts_off_turnover_causes: tuple[str, ...] = ("turnover", "steal")  # possession after a TO [30]; sourced
    steal_link_s: float = 1.0  # STEAL and TURNOVER this close describe one play; tuned
    # ---- crediting
    assist_requires_made_shot: bool = True  # an assist on a known missed/pending shot is not credited; sourced [15]
    # FOUL attrs["type"] values kept out of PF and counted as TF [15]; sourced
    technical_foul_types: tuple[str, ...] = ("technical", "bench_technical", "coach_technical")
    # SHOT_ATTEMPT attrs["ft_type"] of trips after which the shooting team keeps the ball (flagrant /
    # disruptive; EventEngine tags them "flagrant"): a made FT does not end the possession. Rules
    # note §8b ("no rebound, possession retained"); sourced
    possession_kept_ft_types: tuple[str, ...] = ("flagrant", "disruptive", "unsportsmanlike")
    xpts_include_ft: bool = True  # xPTS includes FT attempts so it compares with PTS; design choice
    include_roster_players: bool = True  # list every roster player (DNP lines are zeros); design choice
    # ---- scoring runs
    run_window_s: float = 300.0  # "last N minutes" differential window (game seconds when known); tuned
    # ---- tracking
    max_speed_fts: float = 30.0  # step speed above this is an identity jump (ARCHITECTURE perception clamp); sourced
    speed_window_s: float = 0.5  # top speed = displacement over this window / window; tuned
    position_smoothing_tau_s: float = 0.0  # >0: distance on EMA-smoothed positions (off: perception smooths); tuned
    max_track_gap_s: float = 0.5  # a player unseen longer than this restarts (no step, no time); tuned
    touch_regain_gap_s: float = 0.5  # holder flicker shorter than this is the same touch; tuned
    # ---- decision grades
    decision_buffer_s: float = 10.0  # frame ring buffer length (spec ~10 s); tuned
    decision_frame_tol_s: float = 0.5  # nearest buffered frame must be this close to the event; tuned
    grade_kinds: tuple[str, ...] = ("pass", "shot_attempt")  # ARCHITECTURE DecisionModel.grade; sourced
    # ---- scoreboard / game clock
    ocr_confirm_reads: int = 2  # identical consecutive OCR reads to confirm a score (D14); sourced
    trusted_scoreboard_sources: tuple[str, ...] = ("manual", "feed", "sim")  # confirm at once (ARCHITECTURE); sourced
    clock_stale_s: float = 3.0  # no clock reading for this long -> fall back to live video dt; tuned
    max_frame_dt_s: float = 0.5  # cap on one frame's video dt (pauses / dropped frames); tuned
    # An untrusted (OCR) clock reading that puts elapsed game time this far ahead of the video time
    # since the last accepted sample (the game clock cannot run faster than real time) waits for a
    # second consistent read before it is used (D14 clock consistency); tuned
    clock_jump_tol_s: float = 2.0
    # ---- lineups
    lineup_min_possessions: int = 20  # below this a lineup is small_sample (narration.md §7.2 proposal); unverified
    # ---- attacking direction before any shot evidence: home attacks this basket in the first half
    default_first_half_attack: dict[str, str] = field(
        default_factory=lambda: {"home": "right", "away": "left"}
    )  # convention; unverified
    # ---- period lengths when no RuleSpec is given (config/rules/*.yaml; sourced)
    period_s_by_league: dict[str, int] = field(
        default_factory=lambda: {"nba": 720, "fiba": 600, "ncaa_m": 1200, "ncaa_w": 600, "nfhs": 480}
    )
    ot_s_by_league: dict[str, int] = field(
        default_factory=lambda: {"nba": 300, "fiba": 300, "ncaa_m": 300, "ncaa_w": 300, "nfhs": 240}
    )
    n_periods_by_league: dict[str, int] = field(
        default_factory=lambda: {"nba": 4, "fiba": 4, "ncaa_m": 2, "ncaa_w": 4, "nfhs": 4}
    )


# --------------------------------------------------------------------------- JSON helpers


def _json_safe(obj: Any) -> Any:
    """Recursively convert to JSON-safe values (enum -> value, nan/inf -> None, tuple -> list)."""
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, dict):
        return {(k.value if isinstance(k, enum.Enum) else str(k)): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return _json_safe(obj.to_dict())
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _json_safe(getattr(obj, f.name)) for f in fields(obj)}
    if hasattr(obj, "tolist"):  # numpy scalar or array (array.item() raises for size > 1)
        return _json_safe(obj.tolist())
    return str(obj)


def _team_keyed(d: dict[Any, Any] | None) -> dict[Team, Any]:
    return {Team(k) if not isinstance(k, Team) else k: v for k, v in (d or {}).items()}


def _opt_team(v: Any) -> Team | None:
    return None if v is None else (v if isinstance(v, Team) else Team(v))


# --------------------------------------------------------------------------- GameStats


@dataclass
class GameStats:
    """Snapshot of every stat at time ``t``. ``to_dict()`` is JSON-safe."""

    t: float
    period: int | None
    game_clock: float | None
    score: dict[Team, int]
    players: dict[str, PlayerLine]
    teams: dict[Team, TeamLine]
    tracking: dict[str, TrackingLine]
    shots: list[ShotRecord]
    run: dict  # {"team","points","opp_points","since_t", + window keys}; see scoring_run
    lineups: dict  # lineup id -> record (operator entries only; empty otherwise)
    decisions: list[dict]  # DecisionGrade.to_dict() entries (+ "event_id")
    # Additions beyond the ARCHITECTURE field list (all defaulted).
    score_source: str = "events"  # "scoreboard" (confirmed reading) | "events"
    score_mismatch: bool = False  # confirmed scoreboard != event-derived team points
    event_score: dict[Team, int] = field(default_factory=dict)  # TeamLine.pts per team
    shot_clock: float | None = None
    elapsed_game_s: float | None = None  # elapsed game seconds (pace / lineup seconds basis)
    clock_basis: str = "none"  # "game_clock" | "video" | "mixed" | "none"
    possession_team: Team | None = None  # from on_frame
    n_events: int = 0  # active (non-retracted) events

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": _json_safe(self.t),
            "period": self.period,
            "game_clock": _json_safe(self.game_clock),
            "score": _json_safe(self.score),
            "players": {pid: line.to_dict() for pid, line in self.players.items()},
            "teams": {team.value: line.to_dict() for team, line in self.teams.items()},
            "tracking": {pid: line.to_dict() for pid, line in self.tracking.items()},
            "shots": [s.to_dict() for s in self.shots],
            "run": _json_safe(self.run),
            "lineups": _json_safe(self.lineups),
            "decisions": _json_safe(self.decisions),
            "score_source": self.score_source,
            "score_mismatch": self.score_mismatch,
            "event_score": _json_safe(self.event_score),
            "shot_clock": _json_safe(self.shot_clock),
            "elapsed_game_s": _json_safe(self.elapsed_game_s),
            "clock_basis": self.clock_basis,
            "possession_team": _json_safe(self.possession_team),
            "n_events": self.n_events,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GameStats:
        """Inverse of ``to_dict`` (derived keys such as ``reb`` / ``efg`` are recomputed)."""
        run = dict(d.get("run") or {})
        if "team" in run:
            run["team"] = _opt_team(run["team"])
        if isinstance(run.get("window"), dict):
            run["window"] = _team_keyed(run["window"])
        if "window_leader" in run:
            run["window_leader"] = _opt_team(run["window_leader"])
        lineups = {k: {**v, "team": _opt_team(v.get("team"))} for k, v in (d.get("lineups") or {}).items()}
        return cls(
            t=d["t"],
            period=d.get("period"),
            game_clock=d.get("game_clock"),
            score=_team_keyed(d.get("score")),
            players={pid: PlayerLine.from_dict(v) for pid, v in (d.get("players") or {}).items()},
            teams={Team(k): TeamLine.from_dict(v) for k, v in (d.get("teams") or {}).items()},
            tracking={pid: TrackingLine.from_dict(v) for pid, v in (d.get("tracking") or {}).items()},
            shots=[ShotRecord.from_dict(s) for s in d.get("shots") or []],
            run=run,
            lineups=lineups,
            decisions=list(d.get("decisions") or []),
            score_source=d.get("score_source", "events"),
            score_mismatch=bool(d.get("score_mismatch", False)),
            event_score=_team_keyed(d.get("event_score")),
            shot_clock=d.get("shot_clock"),
            elapsed_game_s=d.get("elapsed_game_s"),
            clock_basis=d.get("clock_basis", "none"),
            possession_team=_opt_team(d.get("possession_team")),
            n_events=int(d.get("n_events", 0)),
        )


# --------------------------------------------------------------------------- runs


def scoring_run(
    scoring: list[ScoringPlay],
    now_t: float,
    window_s: float,
    game_time: Callable[[float], float] | None = None,
) -> dict[str, Any]:
    """Current unanswered run plus the points of each team in the last ``window_s`` seconds.

    ``{"team", "points", "opp_points" (0: the run is unanswered), "since_t" (first basket of the
    run), "plays", "window_s", "window" {team: pts}, "window_leader", "window_diff"}``.
    ``game_time`` maps video t to elapsed game seconds; with it the window is "the last N game
    minutes" (timeouts and breaks do not eat it), else it is measured in video seconds.
    """
    clock = game_time or (lambda t: t)
    plays = sorted((s for s in scoring if s.points != 0), key=lambda s: s.t)
    baskets = [s for s in plays if s.points > 0]
    run: dict[str, Any] = {"team": None, "points": 0, "opp_points": 0, "since_t": None, "plays": 0}
    if baskets:
        team = baskets[-1].team
        for s in reversed(plays):
            if s.team is not team:
                if s.points > 0:
                    break  # the opponent scored: the run starts after this
                continue  # a negative SCORE_CORRECTION to the opponent does not answer the run
            run["points"] += s.points  # a negative correction to the run team shrinks it
            if s.points > 0:
                run["since_t"] = s.t
                run["plays"] += 1
        run["points"] = max(run["points"], 0)
        run["team"] = team
    window = {t: 0 for t in Team}
    start = clock(now_t) - window_s
    for s in plays:
        if clock(s.t) > start:
            window[s.team] += s.points
    diff = window[Team.HOME] - window[Team.AWAY]
    run.update(
        window_s=window_s,
        window=window,
        window_leader=None if diff == 0 else (Team.HOME if diff > 0 else Team.AWAY),
        window_diff=abs(diff),
    )
    return run


# --------------------------------------------------------------------------- shot features


@dataclass
class _FallbackShotFeatures:
    """Same fields as ``stats.shot_quality.ShotFeatures`` (used if that module is unavailable)."""

    distance_ft: float
    value: int
    shot_type: str = "jumper"
    closest_defender_ft: float | None = None
    catch_and_shoot: bool | None = None
    zone: str | None = None


def _shot_features_cls() -> type:
    try:
        from courtvision.stats.shot_quality import ShotFeatures
    except ImportError:  # stats-models module not present
        return _FallbackShotFeatures
    return ShotFeatures


# --------------------------------------------------------------------------- engine

# Kinds that only happen with the ball live inside a period.
_LIVE_PLAY_KINDS = frozenset({EventKind.POSSESSION, EventKind.PASS, EventKind.DRIBBLE, EventKind.SHOT_ATTEMPT})


class StatsEngine:
    """Live stats over the event stream and frames (ARCHITECTURE "stats/")."""

    def __init__(
        self,
        court: CourtSpec,
        roster: Roster | None,
        config: StatsConfig | None = None,
        shot_model: Any | None = None,
        decision_model: Any | None = None,
        rules: RuleSpec | None = None,
    ) -> None:
        self.court = court
        self.roster = roster
        self.config = config or StatsConfig()
        self.shot_model = shot_model
        self.decision_model = decision_model
        self.rules = rules
        cfg = self.config
        self._teams = TeamResolver(roster)
        self._log: dict[str, Event] = {}
        self._order: dict[str, int] = {}
        self._events_version = 0
        self._box_cache: tuple[int, float | None, BoxScore] | None = None  # (events version, elapsed, box)
        self._xfg_cache: dict[tuple, float | None] = {}
        self._features_cls: type | None = None
        self._tracking = TrackingAccumulator(
            court,
            max_speed_fts=cfg.max_speed_fts,
            speed_window_s=cfg.speed_window_s,
            max_gap_s=cfg.max_track_gap_s,
            touch_regain_gap_s=cfg.touch_regain_gap_s,
            position_smoothing_tau_s=cfg.position_smoothing_tau_s,
        )
        self._clock = GameClockTimeline(
            self._period_length_s,
            clock_stale_s=cfg.clock_stale_s,
            max_dt_s=cfg.max_frame_dt_s,
            jump_tol_s=cfg.clock_jump_tol_s,
            trusted_sources=cfg.trusted_scoreboard_sources,
        )
        self._lineups = LineupTracker()
        self._frames: deque[FrameState] = deque()
        self._pending_grades: list[Event] = []
        self._grades: dict[str, dict[str, Any]] = {}
        self._graded_rev: dict[str, int] = {}
        self._scoreboard: ScoreboardReading | None = None  # latest confirmed score
        self._clock_reading: ScoreboardReading | None = None  # latest period / clock reading
        self._ocr_key: tuple[int, int] | None = None
        self._ocr_count = 0
        self._ocr_last_t: float | None = None
        # SHOT_ATTEMPT id -> ((team, half), basket): one vote per shot; the majority per (team, half)
        # is the attacked basket, so one stray heave or mislabelled shot cannot flip it.
        self._attack_shots: dict[str, tuple[tuple[Team, int], str]] = {}
        self._attack_cache: dict[tuple[Team, int], str] | None = None
        self._attack_override: dict[Team, Side] = {}
        # Active PERIOD_START / PERIOD_END events by id: a revision replaces, a retraction removes,
        # so revised period events are never counted twice.
        self._period_events: dict[str, Event] = {}
        self._last_live_play_t = -math.inf  # latest live-play event (play resumed without PERIOD_START)
        self._latest_t = 0.0
        self._latest_frame_t: float | None = None
        self._possession_team: Team | None = None

    # ------------------------------------------------------------------ rules helpers

    def _league_value(self, table: dict[str, int], default: int) -> int:
        return int(table.get(self.court.league, default))

    def _period_length_s(self, period: int) -> float:
        if self.rules is not None:
            return float(self.rules.period_length_s(period))
        n = self._league_value(self.config.n_periods_by_league, 4)
        if period > n:
            return float(self._league_value(self.config.ot_s_by_league, 300))
        return float(self._league_value(self.config.period_s_by_league, 720))

    def _half_index(self, period: int) -> int:
        n = self.rules.n_periods if self.rules is not None else self._league_value(self.config.n_periods_by_league, 4)
        return 0 if period <= max(n // 2, 1) else 1

    def _period_at(self, t: float) -> int | None:
        """Period in play at video time ``t`` from the active PERIOD_START events (None: none yet)."""
        starts = [e for e in self._period_events.values() if e.kind is EventKind.PERIOD_START and e.t <= t]
        if not starts:
            return None
        last = max(starts, key=lambda e: e.t)
        p = last.attrs.get("period", last.value)
        if isinstance(p, int) and not isinstance(p, bool) and p > 0:
            return p
        return len(starts)

    def _current_period(self) -> int:
        if self._clock_reading is not None and self._clock_reading.period:
            return int(self._clock_reading.period)
        return self._period_at(self._latest_t) or 1

    def _is_live(self, t: float) -> bool:
        """Inside a period at ``t``: video dt counts toward elapsed game time (fallback basis)."""
        ends = [e.t for e in self._period_events.values() if e.kind is EventKind.PERIOD_END and e.t <= t]
        if not ends:
            return True
        end = max(ends)
        starts = [e.t for e in self._period_events.values() if e.kind is EventKind.PERIOD_START and e.t <= t]
        if starts and max(starts) >= end:
            return True
        return self._last_live_play_t > end  # play resumed although no PERIOD_START was seen

    def _attack_majority(self) -> dict[tuple[Team, int], str]:
        if self._attack_cache is None:
            votes: dict[tuple[Team, int], Counter[str]] = {}
            for key, side in self._attack_shots.values():
                votes.setdefault(key, Counter())[side] += 1
            self._attack_cache = {key: c.most_common(1)[0][0] for key, c in votes.items()}
        return self._attack_cache

    def attacking_side(self, team: Team) -> Side | None:
        """Basket ``team`` attacks now: operator/engine override, shot evidence, then config default."""
        if team in self._attack_override:
            return self._attack_override[team]
        h = self._half_index(self._current_period())
        flip: dict[str, Side] = {"left": "right", "right": "left"}
        opp = team.other()
        evidence = self._attack_majority()
        for key, same in (((team, h), True), ((team, 1 - h), False), ((opp, h), False), ((opp, 1 - h), True)):
            side = evidence.get(key)
            if side is not None:
                return side if same else flip[side]  # type: ignore[return-value]
        default = self.config.default_first_half_attack.get(team.value)
        if default not in flip:
            return None
        return default if h == 0 else flip[default]  # type: ignore[return-value]

    # ------------------------------------------------------------------ inputs

    def on_events(self, events: list[Event]) -> None:
        """Ingest new events and revisions; the highest ``revision`` per ``event_id`` wins."""
        for e in events:
            self._ingest(e)

    def _ingest(self, e: Event) -> None:
        cur = self._log.get(e.event_id)
        if cur is not None and e.revision < cur.revision:
            log.debug("stale revision %d of %s ignored (have %d)", e.revision, e.event_id, cur.revision)
            return
        stored = replace(e, attrs=dict(e.attrs))
        self._log[e.event_id] = stored
        self._order.setdefault(e.event_id, len(self._order))
        self._events_version += 1
        self._latest_t = max(self._latest_t, e.t)
        self._period_events.pop(e.event_id, None)
        if self._attack_shots.pop(e.event_id, None) is not None:
            self._attack_cache = None
        if stored.status is EventStatus.RETRACTED:
            return
        if e.kind in (EventKind.PERIOD_START, EventKind.PERIOD_END):
            self._period_events[e.event_id] = stored
        elif e.kind in _LIVE_PLAY_KINDS:
            self._last_live_play_t = max(self._last_live_play_t, e.t)
        if e.kind is EventKind.SHOT_ATTEMPT:
            self._note_attack(stored)
        if self.decision_model is not None and e.kind.value in self.config.grade_kinds:
            # grade once per revision; a failed regrade keeps the previous grade
            if self._graded_rev.get(e.event_id, -1) < e.revision or e.event_id not in self._grades:
                if not (e.kind is EventKind.SHOT_ATTEMPT and is_free_throw_attempt(e)):
                    self._pending_grades = [x for x in self._pending_grades if x.event_id != e.event_id]
                    self._queue_grade(stored)

    def _note_attack(self, e: Event) -> None:
        basket = e.attrs.get("basket")
        team = self._teams.of(e.player_id, e.team)
        if basket in ("left", "right") and team is not None:
            period = self._period_at(e.t) or self._current_period()  # the shot's own period
            self._attack_shots[e.event_id] = ((team, self._half_index(period)), basket)
            self._attack_cache = None

    def on_frame(
        self,
        frame: FrameState,
        holder_id: str | None,
        possession_team: Team | None,
        attacking_side: dict[Team, Side] | None = None,
    ) -> None:
        """Fold one frame in. ``attacking_side`` (optional, e.g. ``EventEngine.attacking_side``)
        overrides the shot-evidence inference used for paint touches and decision grading."""
        if attacking_side:
            self._attack_override = dict(attacking_side)
        accepted = self._clock.update(frame.t, frame.scoreboard, self._is_live(frame.t))
        if frame.scoreboard is not None:
            self._ingest_scoreboard(frame.scoreboard, clock_accepted=accepted)
        self._tracking.update(frame, holder_id, self.attacking_side)
        self._possession_team = possession_team
        self._latest_frame_t = frame.t
        self._latest_t = max(self._latest_t, frame.t)
        self._frames.append(frame)
        while self._frames and self._frames[0].t < frame.t - self.config.decision_buffer_s:
            self._frames.popleft()
        self._grade_pending()

    def on_scoreboard(self, reading: ScoreboardReading, confirmed: bool | None = None) -> None:
        """Push a scoreboard / feed / operator reading outside the frame stream.

        ``confirmed=None`` applies the source rule (trusted sources at once, OCR after
        ``ocr_confirm_reads`` identical reads). Its game clock also feeds the elapsed timeline.
        """
        accepted = False
        if reading.game_clock is not None and reading.period is not None:
            accepted = self._clock.update(reading.t, reading, self._is_live(reading.t))
        self._ingest_scoreboard(reading, confirmed, clock_accepted=accepted)
        self._latest_t = max(self._latest_t, reading.t)

    def _ingest_scoreboard(
        self, reading: ScoreboardReading, confirmed: bool | None = None, clock_accepted: bool = True
    ) -> None:
        if self._clock_usable(reading, clock_accepted):
            if self._clock_reading is None or reading.t >= self._clock_reading.t:
                self._clock_reading = reading
        if reading.home_score is None or reading.away_score is None:
            return
        if confirmed is None:
            confirmed = reading.source in self.config.trusted_scoreboard_sources or self._ocr_vote(reading)
        if confirmed:
            self._scoreboard = reading

    def _clock_usable(self, reading: ScoreboardReading, clock_accepted: bool) -> bool:
        """Period / clock of this reading may become ``GameStats.period`` / ``game_clock``."""
        if reading.period is None and reading.game_clock is None:
            return False
        if reading.game_clock is not None and not math.isfinite(float(reading.game_clock)):
            return False  # unusable clock digits: keep the previous period / clock
        held = reading.game_clock is not None and reading.period is not None and not clock_accepted
        # an untrusted (OCR) clock jump the timeline is holding for a second read (D14)
        return not (held and reading.source not in self.config.trusted_scoreboard_sources)

    def _ocr_vote(self, reading: ScoreboardReading) -> bool:
        key = (int(reading.home_score or 0), int(reading.away_score or 0))
        if self._scoreboard is not None and key == (self._scoreboard.home_score, self._scoreboard.away_score):
            self._ocr_key, self._ocr_count = key, self.config.ocr_confirm_reads
            return True
        if self._ocr_last_t is not None and reading.t == self._ocr_last_t:
            return False  # the same read attached to several frames is one read
        self._ocr_last_t = reading.t
        if key == self._ocr_key:
            self._ocr_count += 1
        else:
            self._ocr_key, self._ocr_count = key, 1
        return self._ocr_count >= self.config.ocr_confirm_reads

    # ------------------------------------------------------------------ operator lineups

    def set_lineup(self, team: Team, player_ids: list[str], t: float) -> None:
        """Operator: starting five / full reset of ``team``'s floor lineup at video time ``t``."""
        self._lineups.set_lineup(team, list(player_ids), t)

    def substitute(self, team: Team, out_id: str, in_id: str, t: float) -> None:
        """Operator tap-to-sub at video time ``t``."""
        self._lineups.substitute(team, out_id, in_id, t)

    def on_court(self, team: Team) -> list[str]:
        """Current operator lineup of ``team`` (empty without entries)."""
        return self._lineups.on_court(team)

    # ------------------------------------------------------------------ decisions

    def _queue_grade(self, e: Event) -> None:
        if self._latest_frame_t is not None and e.t <= self._latest_frame_t:
            self._grade(e)
        else:
            self._pending_grades.append(e)

    def _grade_pending(self) -> None:
        if not self._pending_grades or self._latest_frame_t is None:
            return
        ready = [e for e in self._pending_grades if e.t <= self._latest_frame_t]
        self._pending_grades = [e for e in self._pending_grades if e.t > self._latest_frame_t]
        for e in ready:
            self._grade(e)

    def _frame_at(self, e: Event) -> FrameState | None:
        best: FrameState | None = None
        for f in self._frames:
            if f.frame_idx == e.frame_idx:
                return f
            if best is None or abs(f.t - e.t) < abs(best.t - e.t):
                best = f
        if best is not None and abs(best.t - e.t) <= self.config.decision_frame_tol_s:
            return best
        return None

    def _grade(self, e: Event) -> None:
        frame = self._frame_at(e)
        offense = self._teams.of(e.player_id, e.team)
        if frame is None or offense is None or e.player_id is None:
            log.debug("decision grade skipped for %s (frame=%s, offense=%s)", e.event_id, frame is not None, offense)
            return
        basket = e.attrs.get("basket")
        if basket not in ("left", "right"):
            basket = self.attacking_side(offense)
        if basket is None:
            return
        try:
            grade = self.decision_model.grade(frame, e, offense, basket)  # type: ignore[union-attr]
        except Exception:  # a model failure must never break the stats
            log.exception("decision_model.grade failed for %s", e.event_id)
            return
        if grade is None:
            return
        d = dict(grade.to_dict()) if hasattr(grade, "to_dict") else dict(grade)
        fill = {"event_id": e.event_id, "event_kind": e.kind.value, "t": e.t, "player_id": e.player_id}
        for key, value in fill.items():
            if d.get(key) is None:
                d[key] = value
        self._grades[e.event_id] = _json_safe(d)
        self._graded_rev[e.event_id] = e.revision

    # ------------------------------------------------------------------ xFG

    def _xfg(self, shot: ShotRecord) -> float | None:
        if self.shot_model is None:
            return None
        distance = shot.distance_ft
        if shot.is_ft and distance is None:
            distance = self.court.free_throw_line_from_baseline - self.court.basket_offset
        if distance is None:
            return None
        key = (shot.shot_id, distance, shot.value, shot.shot_type, shot.closest_defender_ft,
               shot.catch_and_shoot, shot.zone)
        if key not in self._xfg_cache:
            if self._features_cls is None:
                self._features_cls = _shot_features_cls()
            feats = self._features_cls(
                distance_ft=float(distance),
                value=shot.value,
                shot_type=shot.shot_type,
                closest_defender_ft=shot.closest_defender_ft,
                catch_and_shoot=shot.catch_and_shoot,
                zone=shot.zone,
            )
            try:
                self._xfg_cache[key] = float(self.shot_model.xfg(feats))
            except Exception:
                log.exception("shot_model.xfg failed for %s", shot.shot_id)
                self._xfg_cache[key] = None
        return self._xfg_cache[key]

    # ------------------------------------------------------------------ output

    @property
    def events(self) -> list[Event]:
        """Latest revision of each non-retracted event, in game order."""
        return order_events(self._active())

    def _active(self) -> list[Event]:
        ids = sorted(self._log, key=self._order.__getitem__)
        return [self._log[i] for i in ids if self._log[i].status is not EventStatus.RETRACTED]

    def _box(self, active: list[Event]) -> BoxScore:
        elapsed = self._clock.total
        game_minutes = advanced.regulation_minutes(self.rules, self.court.league)
        cache = self._box_cache
        if cache is not None and cache[0] == self._events_version:
            if cache[1] != elapsed:  # same events, later frame: only pace / seconds move
                refresh_team_rates(cache[2], self.config, elapsed, game_minutes)
                self._box_cache = (cache[0], elapsed, cache[2])
            return cache[2]
        box = build_box(
            active,
            court=self.court,
            roster=self.roster,
            config=self.config,
            xfg_fn=self._xfg if self.shot_model is not None else None,
            elapsed_s=elapsed,
            game_minutes=game_minutes,
        )
        self._box_cache = (self._events_version, elapsed, box)
        return box

    def _elapsed_at(self, t: float) -> float:
        v = self._clock.elapsed_at(t)
        return t if v is None else v

    def snapshot(self) -> GameStats:
        """Recompute every stat from the event log, frames and operator entries."""
        active = self._active()
        box = self._box(active)
        players = {pid: replace(line) for pid, line in box.players.items()}
        teams = {t: replace(line) for t, line in box.teams.items()}
        lineups: dict[str, dict[str, Any]] = {}
        if self._lineups.has_entries():
            res = compute_lineups(
                self._lineups.stints(),
                self._elapsed_at,
                self._latest_t,
                box.scoring,
                box.possessions,
                self.config.lineup_min_possessions,
            )
            lineups = res.lineups
            for pid, secs in res.player_seconds.items():
                line = players.get(pid)
                if line is None:
                    line = players[pid] = PlayerLine(player_id=pid, team=self._teams.of(pid))
                line.seconds = secs
                line.plus_minus = res.player_plus_minus.get(pid, 0)
        event_score = {t: teams[t].pts for t in Team}
        score, source, mismatch = dict(event_score), "events", False
        if self._scoreboard is not None:
            score = {Team.HOME: int(self._scoreboard.home_score or 0), Team.AWAY: int(self._scoreboard.away_score or 0)}
            source = "scoreboard"
            mismatch = score != event_score
        clock = self._clock_reading
        decisions = [d for eid, d in self._grades.items() if self._log[eid].status is not EventStatus.RETRACTED]
        decisions.sort(key=lambda d: (d.get("t") or 0.0))
        return GameStats(
            t=self._latest_t,
            period=int(clock.period) if clock is not None and clock.period else box.period,
            game_clock=clock.game_clock if clock is not None else None,
            score=score,
            players=players,
            teams=teams,
            tracking=self._tracking.lines(event_counts(active)),
            shots=[replace(s) for s in box.shots],
            run=scoring_run(box.scoring, self._latest_t, self.config.run_window_s,
                            self._elapsed_at if self._clock.total is not None else None),
            lineups=lineups,
            decisions=decisions,
            score_source=source,
            score_mismatch=mismatch,
            event_score=event_score,
            shot_clock=clock.shot_clock if clock is not None else None,
            elapsed_game_s=self._clock.total,
            clock_basis=self._clock.basis,
            possession_team=self._possession_team,
            n_events=len(active),
        )

