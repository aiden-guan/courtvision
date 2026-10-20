"""FactPacket builder: the only facts a narration line may use (narration.md §3.2, RESEARCH.md §2.6).

The packet is pseudonymised (D16): players appear only as ``P07``-style tokens, never names. All
arithmetic (margins, runs, streaks, shooting lines, milestones) happens here in Python; the
``derived`` phrases are code-made and may be quoted verbatim. ``allowed_numbers`` lists every
number that appears anywhere in the packet, as digits and words, plus ordinals for the period, so
the validator can reject any other number.

Keys: ``state_version``, ``event`` (primary triggering event), ``events``, ``score``, ``clock``,
``game_clock``, ``players`` (involved players' counting stats), ``teams``, ``run``, ``derived``,
``storylines``, ``recent_events``, ``labels``, ``allowed_numbers``, ``allowed_tokens``,
``recent_lines``, ``hedge_only``, ``confirmed`` (+ ``correction`` when ``previous`` is given).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from courtvision.geometry.court import CourtSpec
from courtvision.geometry.rules import RuleSpec
from courtvision.narration.pseudonym import TOKEN_RE, PseudonymMap, pseudonyms_for, team_from_player_id
from courtvision.narration.validator import extract_numbers, int_to_words, number_forms, ordinal_words
from courtvision.types import Event, EventKind, EventStatus, FrameState, Roster, Side, Team

if TYPE_CHECKING:  # read by attribute only, so the stats module is not imported at run time
    from courtvision.stats.engine import GameStats

log = logging.getLogger(__name__)

# Events whose narration states an outcome or a point value (hedge-only unless trusted).
OUTCOME_KINDS = frozenset({
    EventKind.SHOT_ATTEMPT, EventKind.SHOT_MADE, EventKind.SHOT_MISSED, EventKind.BLOCK, EventKind.REBOUND,
    EventKind.STEAL, EventKind.TURNOVER, EventKind.ASSIST, EventKind.FOUL, EventKind.SCORE_CORRECTION,
})
SHOT_KINDS = frozenset({EventKind.SHOT_ATTEMPT, EventKind.SHOT_MADE, EventKind.SHOT_MISSED, EventKind.BLOCK})

# Salience order for the packet's primary ``event`` (lower = more salient).
_KIND_RANK = {k: i for i, k in enumerate([
    EventKind.SCORE_CORRECTION, EventKind.SHOT_MADE, EventKind.BLOCK, EventKind.STEAL, EventKind.TURNOVER,
    EventKind.SHOT_MISSED, EventKind.REBOUND, EventKind.FOUL, EventKind.SHOT_ATTEMPT, EventKind.ASSIST,
    EventKind.TIMEOUT, EventKind.SUBSTITUTION, EventKind.PERIOD_END, EventKind.PERIOD_START,
    EventKind.POSSESSION, EventKind.PASS, EventKind.DRIBBLE,
])}

# Keys whose values are not speakable numbers (ids, probabilities, coordinates, raw seconds).
_SKIP_KEYS = frozenset({
    "event_id", "shot_id", "state_version", "confidence", "t", "since_t", "x", "y", "seconds_left",
    "recent_lines", "allowed_numbers", "allowed_tokens", "status", "kind", "old_kind", "team_side", "actor",
    "other",
    "cause", "shot_type", "turnover_type", "outcome_ok", "retracted", "league", "is_last_period",
    "confirmed", "speakable", "hedge_only", "offensive", "basket",
})

_PLAYER_STATS = ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl",
                 "blk", "tov", "pf")
_TEAM_STATS = ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl",
               "blk", "tov", "pf")
_DOUBLE_CATS = ("pts", "reb", "ast", "stl", "blk")


@dataclass
class FactsConfig:
    """FactPacket settings. Gates and callout thresholds are design proposals until tuned."""

    max_recent_lines: int = 10  # repetition is checked against the last 10 lines (narration.md §3.3); sourced
    trusted_sources: tuple[str, ...] = ("sim", "manual")  # never hedge-only (ARCHITECTURE events/); sourced
    calibrated_statuses: tuple[str, ...] = ("platt", "isotonic", "calibrated", "assumed_sim")  # §7A.2; sourced
    apply_confidence_gates: bool = True  # calibrated events must also pass their live gate (RESEARCH.md §4.1)
    # Live gates on Event.confidence for calibrated, unconfirmed events (RESEARCH.md §4.1 / narration.md
    # §9.3; design proposals, unverified until certified on held-out games).
    gates: dict[str, float] = field(default_factory=lambda: {
        "shot_attempt": 0.90, "shot_made": 0.95, "shot_missed": 0.90, "rebound": 0.90, "steal": 0.90,
        "turnover": 0.90, "block": 0.92, "foul": 0.97, "assist": 0.90, "score_correction": 0.95})
    three_gate: float = 0.97  # say "three" only if P(3) >= 0.97 (attrs "p_three" when present; §4.1); unverified
    ocr_min_conf: float = 0.90  # a frame scoreboard OCR read below this does not confirm the score; unverified
    streak_min: int = 3  # "third straight make" from 3 consecutive made FGs (design proposal); tuned
    run_min_margin: int = 8  # run callout at 8-0 or 10-2 (narration.md §9.5); sourced (proposal)
    run_max_opp: int = 2  # ...with the opponent held to <= 2 in that span (same); sourced (proposal)
    milestones: tuple[int, ...] = (10, 20, 30, 40)  # player point milestones (narration.md §9.5); sourced (proposal)
    shooting_line_min_fga: int = 4  # "{P07} is 4-for-6" derived only from 4 attempts (design proposal); tuned
    top_half_angle_deg: float = 22.5  # AD zone "top": within this angle of the basket axis; tuned
    # Period naming when no RuleSpec is given: (n_periods, halves) per league (rules note §3).
    period_format: dict[str, tuple[int, bool]] = field(default_factory=lambda: {
        "nba": (4, False), "fiba": (4, False), "ncaa_m": (2, True), "ncaa_w": (4, False), "nfhs": (4, False)})


# --------------------------------------------------------------------------- public helpers


def event_trusted(e: Event, config: FactsConfig | None = None, gate_boost: float = 0.0,
                  p_three: float | None = None) -> bool:
    """May this event's outcome / point value be stated? (RESEARCH.md §7A.2 hedge-only rule).

    ``p_three`` is P(3) from the shot's attempt when the event itself does not carry it (the
    engine puts ``p_three`` on the SHOT_ATTEMPT only); a value-3 attempt or make below the 0.97
    three gate is not trusted (§4.1: "three" only if P(3) >= 0.97)."""
    cfg = config or FactsConfig()
    if e.status is EventStatus.CONFIRMED or e.source in cfg.trusted_sources:
        return True
    if str(e.attrs.get("p_status", "uncalibrated")) not in cfg.calibrated_statuses:
        return False
    if not cfg.apply_confidence_gates:
        return True
    gate = cfg.gates.get(e.kind.value)
    if gate is not None and e.confidence < min(1.0, gate + gate_boost):
        return False
    if e.kind in (EventKind.SHOT_ATTEMPT, EventKind.SHOT_MADE) and e.value == 3:
        p3 = e.attrs.get("p_three", p_three)
        if p3 is not None and float(p3) < min(1.0, cfg.three_gate + gate_boost):
            return False
    return True


def trust_map(events: Iterable[Event], config: FactsConfig | None = None,
              gate_boost: float = 0.0) -> dict[str, bool]:
    """event_id -> may its outcome be stated. Non-outcome kinds are always True. P(3) is taken from
    the shot's SHOT_ATTEMPT in the same batch; an ASSIST without its own ``p_status`` (it has no cue
    model: the engine derives it from the pass and the make) inherits the trust of the make it
    credits, so an assist never turns a trusted make hedge-only."""
    cfg = config or FactsConfig()
    evs = list(events)
    p3 = {e.shot_id: e.attrs["p_three"] for e in evs
          if e.kind is EventKind.SHOT_ATTEMPT and e.shot_id and e.attrs.get("p_three") is not None}
    trusted = {e.event_id: e.kind not in OUTCOME_KINDS or event_trusted(e, cfg, gate_boost, p3.get(e.shot_id))
               for e in evs}
    makes = {e.shot_id: trusted[e.event_id] for e in evs
             if e.kind is EventKind.SHOT_MADE and e.shot_id and e.status is not EventStatus.RETRACTED}
    for e in evs:
        own = e.status is EventStatus.CONFIRMED or e.source in cfg.trusted_sources or "p_status" in e.attrs
        if e.kind is EventKind.ASSIST and not own:
            trusted[e.event_id] = makes.get(e.shot_id, False)
    return trusted


def primary_event(events: Iterable[Event]) -> Event | None:
    """Most salient live (non-retracted) event; ties go to the latest."""
    live = [e for e in events if e.status is not EventStatus.RETRACTED] or list(events)
    if not live:
        return None
    return min(live, key=lambda e: (_KIND_RANK.get(e.kind, 99), -e.t))


def format_game_clock(seconds: float | None) -> str | None:
    """Game clock as shown: "4:32" from one minute up, tenths ("24.3") below (None if unknown/NaN)."""
    if _finite(seconds) is None:
        return None
    s = max(0.0, float(seconds))
    if s >= 60.0:
        m, sec = divmod(int(s), 60)
        return f"{m}:{sec:02d}"
    return f"{math.floor(s * 10) / 10:.1f}"


def period_text(period: int | None, n_periods: int = 4, halves: bool = False) -> str | None:
    """"fourth quarter", "second half", "overtime", "second overtime"."""
    if period is None or period < 1:
        return None
    if period > n_periods:
        k = period - n_periods
        return "overtime" if k == 1 else f"{ordinal_words(k)} overtime"
    return f"{ordinal_words(period)} {'half' if halves else 'quarter'}"


def ad_zone(court: CourtSpec, x: float | None, y: float | None, basket: Side | None = None,
            config: FactsConfig | None = None) -> str | None:
    """Audio-description zone relative to the attacking basket, from the offense's view facing it:
    left corner, left wing, top, right wing, right corner, paint, backcourt (RESEARCH.md §7A.8)."""
    if _finite(x) is None or _finite(y) is None:
        return None  # unknown or non-finite position (CourtSpec raises on NaN): no zone, never a crash
    cfg = config or FactsConfig()
    side: Side = basket if basket in ("left", "right") else court.nearest_basket(x, y)
    zone = court.zone(x, y, side)
    if zone == "backcourt":
        return "backcourt"
    if zone in ("paint", "restricted_area"):
        return "paint"
    bx, by = court.basket_xy(side)
    along = (x - bx) * (1.0 if side == "left" else -1.0)
    lateral = y - by
    # Facing the left basket the offense looks toward -x, so its left hand points to -y.
    hand = "left" if (lateral < 0) == (side == "left") else "right"
    if along <= court.three_junction_along:
        return f"{hand} corner"
    if math.degrees(math.atan2(abs(lateral), along)) <= cfg.top_half_angle_deg:
        return "top"
    return f"{hand} wing"


# --------------------------------------------------------------------------- builder


def build_facts(events: list[Event], stats: GameStats, roster: Roster, court: CourtSpec,
                frame: FrameState | None = None, pseudo: PseudonymMap | None = None,
                rules: RuleSpec | None = None, recent_lines: list[str] | None = None, *,
                state_version: int | None = None, storylines: list[str] | None = None,
                labels: Mapping[str, str] | None = None, previous: list[Event] | None = None,
                gate_boost: float = 0.0, config: FactsConfig | None = None) -> dict[str, Any]:
    """Build the pseudonymised FactPacket for ``events`` (the events a line is about).

    ``stats`` is a ``GameStats`` snapshot taken after ``events`` were applied (read by attribute).
    ``previous`` holds the versions of revised events that were already spoken (correction lines).
    ``gate_boost`` raises every live gate (director escalation after a correction storm).
    """
    cfg = config or FactsConfig()
    pseudo = pseudo or pseudonyms_for(roster)
    names = {t: roster.team_names.get(t, t.value.capitalize()) for t in Team}
    live = [e for e in events if e.status is not EventStatus.RETRACTED]
    outcome = [e for e in live if e.kind in OUTCOME_KINDS]
    trusted = trust_map(events, cfg, gate_boost)
    hedge_only = any(not trusted[e.event_id] for e in outcome)

    score = _score_fact(stats, frame, names, cfg)
    # An unconfirmed (event-derived) score is speakable only when no counted basket is still
    # provisional: it is a sum of outcomes (RESEARCH.md §7A.2; AD: confirmed values only, §7A.8).
    score["speakable"] = bool(score.get("confirmed") or (not hedge_only and not _provisional_points(stats)))
    n_periods, halves = _period_format(court, rules, cfg)
    clock = _clock_fact(stats, frame, n_periods, halves)
    ev_facts = [_event_fact(e, roster, court, pseudo, names, trusted[e.event_id], cfg) for e in events]
    primary = primary_event(events)
    involved = _involved_players(events, previous)
    players = {pseudo.token(pid): _player_fact(stats, pid, roster, names) for pid in involved}
    teams = _teams_fact(stats, names)
    _mask_untrusted_stats(live, trusted, roster, pseudo, players, teams)
    run = _run_fact(stats, names, cfg) if score["speakable"] else {}
    derived = _derived(live, trusted, stats, roster, pseudo, names, score, run, rules, cfg)

    packet: dict[str, Any] = {
        "state_version": int(state_version) if state_version is not None else _fallback_version(stats),
        "league": court.league,
        "event": next((f for f in ev_facts if primary and f["event_id"] == primary.event_id), None),
        "events": ev_facts,
        "score": score,
        "clock": clock,
        "game_clock": clock.get("text"),
        "players": players,
        "teams": teams,
        "run": run,
        "derived": derived,
        "storylines": [pseudo.pseudonymise(s, roster) for s in storylines or ()],
        "recent_events": [_describe(f) for f in ev_facts],
        "labels": dict(labels or {}),
        "recent_lines": [pseudo.pseudonymise(s, roster) for s in (recent_lines or [])[-cfg.max_recent_lines:]],
        "hedge_only": hedge_only,
        "confirmed": bool(live) and all(e.status is EventStatus.CONFIRMED for e in live),
    }
    correction = _correction_fact(events, previous, pseudo, names)
    if correction:
        packet["correction"] = correction
    packet["allowed_tokens"] = _collect_tokens(packet)
    packet["allowed_numbers"] = _collect_numbers(packet)
    return packet


# --------------------------------------------------------------------------- packet parts


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _by_team(d: Any, team: Team) -> Any:
    if not isinstance(d, Mapping):
        return None
    if team in d:
        return d[team]
    return d.get(team.value)


def _as_team(v: Any) -> Team | None:
    if isinstance(v, Team):
        return v
    if isinstance(v, str) and v in (Team.HOME.value, Team.AWAY.value):
        return Team(v)
    return None


def _finite(v: Any) -> float | None:
    """``v`` as a finite float, else None (None, NaN, inf, non-numbers)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _provisional_points(stats: Any) -> bool:
    """True when a made shot counted in the (event-derived) score is still provisional."""
    return any(_get(s, "made") and _get(s, "provisional", False) for s in _get(stats, "shots", []) or [])


def _fallback_version(stats: Any) -> int:
    # No fact-store counter was passed: video time in ms is monotonic across snapshots.
    return int(round(float(_get(stats, "t", 0.0) or 0.0) * 1000))


def _period_format(court: CourtSpec, rules: RuleSpec | None, cfg: FactsConfig) -> tuple[int, bool]:
    if rules is not None:
        return int(rules.n_periods), bool(rules.halves)
    return cfg.period_format.get(court.league, (4, False))


def _score_fact(stats: Any, frame: FrameState | None, names: dict[Team, str], cfg: FactsConfig) -> dict[str, Any]:
    raw = _get(stats, "score", {}) or {}
    h, a = _by_team(raw, Team.HOME), _by_team(raw, Team.AWAY)
    if h is None or a is None:
        return {"confirmed": False}
    h, a = int(h), int(a)
    leader = None if h == a else names[Team.HOME if h > a else Team.AWAY]
    return {
        "home": h, "away": a, "margin": abs(h - a), "leader": leader,
        "text": f"{names[Team.HOME]} {h}, {names[Team.AWAY]} {a}",
        "confirmed": _score_confirmed(stats, frame, h, a, cfg),
    }


def _score_confirmed(stats: Any, frame: FrameState | None, h: int, a: int, cfg: FactsConfig) -> bool:
    if _get(stats, "score_mismatch", False):
        # Board and events disagree: the board may lag a basket just called (or CV missed one).
        # A frame read of that same board must not confirm it ("Tied at 40" right after a make).
        return False
    if _get(stats, "score_source") == "scoreboard":
        return True
    sb = frame.scoreboard if frame is not None else None
    if sb is None or sb.home_score is None or sb.away_score is None:
        return False
    if sb.source == "ocr" and sb.confidence < cfg.ocr_min_conf:
        return False
    return (int(sb.home_score), int(sb.away_score)) == (h, a)


def _clock_fact(stats: Any, frame: FrameState | None, n_periods: int, halves: bool) -> dict[str, Any]:
    sb = frame.scoreboard if frame is not None else None
    period = _get(stats, "period")
    if period is None and sb is not None:
        period = sb.period
    secs = _finite(_get(stats, "game_clock"))
    if secs is None and sb is not None:
        secs = _finite(sb.game_clock)
    shot = _finite(_get(stats, "shot_clock"))
    if shot is None and sb is not None:
        shot = _finite(sb.shot_clock)
    out: dict[str, Any] = {}
    ptext = period_text(period, n_periods, halves)
    if period is not None:
        out.update(period=int(period), period_text=ptext, is_last_period=int(period) >= n_periods)
    game = format_game_clock(secs)
    if game is not None:
        out.update(game=game, seconds_left=float(secs))
        if float(secs) < 60.0:
            out["whole_seconds"] = int(float(secs))
    if shot is not None:
        shot = max(0.0, shot)  # tenths below 5 s, truncated like the display (4.97 -> "4.9", not "5.0")
        out["shot"] = str(int(shot)) if shot >= 5 else f"{math.floor(shot * 10) / 10:.1f}"
    if ptext and game:
        out["text"] = f"{ptext}, {game} left"
    elif ptext:
        out["text"] = ptext
    return out


def _event_fact(e: Event, roster: Roster, court: CourtSpec, pseudo: PseudonymMap, names: dict[Team, str],
                trusted: bool, cfg: FactsConfig) -> dict[str, Any]:
    team = e.team or roster.team_of(e.player_id)
    d: dict[str, Any] = {"event_id": e.event_id, "kind": e.kind.value, "status": e.status.value,
                         "outcome_ok": trusted, "retracted": e.status is EventStatus.RETRACTED}
    if team is not None:
        d.update(team=names[team], team_side=team.value)
    if e.player_id:
        d["actor"] = pseudo.token(e.player_id)
    if e.other_player_id:
        d["other"] = pseudo.token(e.other_player_id)
    if e.kind in SHOT_KINDS:
        shot_type = e.attrs.get("shot_type")
        if shot_type:
            d["shot_type"] = str(shot_type)
        if trusted and e.value:
            d["value"] = int(e.value)
        dist = e.attrs.get("distance_ft")
        if trusted and dist is not None:
            d["distance_ft"] = int(round(float(dist)))
        basket = e.attrs.get("basket")
        d["basket"] = basket if basket in ("left", "right") else None
        zone = ad_zone(court, e.x, e.y, d["basket"], cfg)
        if zone:
            d["zone"] = zone
    elif e.x is not None and e.y is not None:
        basket = e.attrs.get("basket")  # attacking basket when the producer knows it, else the nearest
        zone = ad_zone(court, e.x, e.y, basket if basket in ("left", "right") else None, cfg)
        if zone:
            d["zone"] = zone
    if e.kind is EventKind.REBOUND:
        d["offensive"] = bool(e.attrs.get("offensive", False))
    if e.kind is EventKind.TURNOVER and e.attrs.get("type"):
        d["turnover_type"] = str(e.attrs["type"])
    if e.kind is EventKind.POSSESSION and e.attrs.get("cause"):
        d["cause"] = str(e.attrs["cause"])
    if e.kind is EventKind.SCORE_CORRECTION and trusted and e.value:
        d["value"] = int(e.value)
    return d


def _involved_players(events: list[Event], previous: list[Event] | None) -> list[str]:
    out: list[str] = []
    for e in [*events, *(previous or [])]:
        for pid in (e.player_id, e.other_player_id):
            if pid and pid not in out:
                out.append(pid)
    return out


def _player_fact(stats: Any, pid: str, roster: Roster, names: dict[Team, str]) -> dict[str, Any]:
    line = (_get(stats, "players", {}) or {}).get(pid)
    team = roster.team_of(pid) or _as_team(_get(line, "team"))
    out: dict[str, Any] = {"team": names[team]} if team is not None else {}
    if line is None:
        return out
    for k in _PLAYER_STATS:
        v = _get(line, k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = int(v)
    if "reb" not in out and "oreb" in out and "dreb" in out:
        out["reb"] = out["oreb"] + out["dreb"]
    for key, made, att in (("fg", "fgm", "fga"), ("fg3", "fg3m", "fg3a"), ("ft", "ftm", "fta")):
        if out.get(att):
            out[key] = f"{out.get(made, 0)}-for-{out[att]}"
    return out


def _teams_fact(stats: Any, names: dict[Team, str]) -> dict[str, Any]:
    teams = _get(stats, "teams", {}) or {}
    out: dict[str, Any] = {}
    for t in Team:
        d: dict[str, Any] = {"name": names[t]}
        line = _by_team(teams, t)
        for k in _TEAM_STATS:
            v = _get(line, k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                d[k] = int(v)
        out[t.value] = d
    return out


_SHOT_STATS = ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "fg", "fg3", "ft")
# Counting stats an outcome event changes, for its actor and for ``other_player_id``.
_STATS_BY_KIND: dict[EventKind, tuple[tuple[str, ...], tuple[str, ...]]] = {
    EventKind.SHOT_ATTEMPT: (_SHOT_STATS, ()), EventKind.SHOT_MADE: (_SHOT_STATS, ()),
    EventKind.SHOT_MISSED: (_SHOT_STATS, ()), EventKind.BLOCK: (("blk",), _SHOT_STATS),
    EventKind.REBOUND: (("reb", "oreb", "dreb"), ()), EventKind.STEAL: (("stl",), ("tov",)),
    EventKind.TURNOVER: (("tov",), ()), EventKind.ASSIST: (("ast",), _SHOT_STATS), EventKind.FOUL: (("pf",), ()),
}


def _mask_untrusted_stats(live: list[Event], trusted: Mapping[str, bool], roster: Roster, pseudo: PseudonymMap,
                          players: dict[str, dict[str, Any]], teams: dict[str, dict[str, Any]]) -> None:
    """Drop the counting stats an untrusted event already changed: the snapshot counts it, so
    "5-for-8" or a team total would state its outcome without any outcome word (§7A.2)."""
    for e in live:
        if trusted.get(e.event_id, True) or e.kind not in _STATS_BY_KIND:
            continue
        mine, theirs = _STATS_BY_KIND[e.kind]
        for pid, keys, is_actor in ((e.player_id, mine, True), (e.other_player_id, theirs, False)):
            if not pid or not keys:
                continue
            for k in keys:
                players.get(pseudo.token(pid), {}).pop(k, None)
            team = roster.team_of(pid) or team_from_player_id(pid)
            if team is None and e.team is not None:  # ASSIST credits a teammate; BLOCK / STEAL an opponent
                team = e.team if is_actor or e.kind is EventKind.ASSIST else e.team.other()
            for k in keys:
                if team is not None:
                    teams.get(team.value, {}).pop(k, None)
        if not e.player_id and e.team is not None:  # team rebound / team turnover
            for k in mine:
                teams.get(e.team.value, {}).pop(k, None)


def _run_fact(stats: Any, names: dict[Team, str], cfg: FactsConfig) -> dict[str, Any]:
    run = _get(stats, "run", {}) or {}
    team = _as_team(_get(run, "team"))
    pts, opp = int(_get(run, "points", 0) or 0), int(_get(run, "opp_points", 0) or 0)
    window, w_leader = _get(run, "window"), _as_team(_get(run, "window_leader"))
    if team is not None and _is_run(pts, opp, cfg):
        return {"team": names[team], "points": pts, "opp_points": opp, "text": f"{pts}-{opp} {names[team]} run"}
    if isinstance(window, Mapping) and w_leader is not None:
        w, o = int(_by_team(window, w_leader) or 0), int(_by_team(window, w_leader.other()) or 0)
        if _is_run(w, o, cfg):
            return {"team": names[w_leader], "points": w, "opp_points": o, "text": f"{w}-{o} {names[w_leader]} run"}
    if team is not None and pts > 0:
        return {"team": names[team], "points": pts, "opp_points": opp}
    return {}


def _is_run(pts: int, opp: int, cfg: FactsConfig) -> bool:
    return pts - opp >= cfg.run_min_margin and opp <= cfg.run_max_opp


def _derived(live: list[Event], trusted: dict[str, bool], stats: Any, roster: Roster, pseudo: PseudonymMap,
             names: dict[Team, str], score: dict[str, Any], run: dict[str, Any], rules: RuleSpec | None,
             cfg: FactsConfig) -> list[str]:
    out: list[str] = []
    for e in live:
        if not trusted.get(e.event_id):
            continue  # nothing outcome-dependent from a hedge-only event
        if e.kind is EventKind.SHOT_MADE:
            out += _made_phrases(e, stats, pseudo, names, score, cfg)
        if e.kind is EventKind.FOUL and e.player_id and rules is not None:
            out += _foul_phrases(e, stats, pseudo, rules)
    for pid in _involved_players([e for e in live if trusted.get(e.event_id)], None):
        line = (_get(stats, "players", {}) or {}).get(pid)
        if line is not None and sum(1 for k in _DOUBLE_CATS if (_get(line, k, 0) or 0) >= 10) >= 2:
            out.append(f"double-double for {{{pseudo.token(pid)}}}")
    if run.get("text"):
        out.append(run["text"])
    return list(dict.fromkeys(out))


def _made_phrases(e: Event, stats: Any, pseudo: PseudonymMap, names: dict[Team, str],
                  score: dict[str, Any], cfg: FactsConfig) -> list[str]:
    out: list[str] = []
    line = (_get(stats, "players", {}) or {}).get(e.player_id) if e.player_id else None
    if e.player_id and e.value in (2, 3):
        tok = pseudo.token(e.player_id)
        streak = _make_streak(_get(stats, "shots", []) or [], e.player_id, e.shot_id)
        if streak >= cfg.streak_min:
            out.append(f"{ordinal_words(streak)} straight make for {{{tok}}}")
        pts = int(_get(line, "pts", 0) or 0)
        for m in cfg.milestones:
            if pts >= m > pts - e.value:
                out.append(f"{{{tok}}} reaches {m} points")
        fgm, fga = int(_get(line, "fgm", 0) or 0), int(_get(line, "fga", 0) or 0)
        if fga >= cfg.shooting_line_min_fga:
            out.append(f"{{{tok}}} is {fgm}-for-{fga} from the field")
    # A confirmed scoreboard that disagrees with the events may not include this basket yet, so the
    # "before" score (score minus this basket) would be wrong: no lead phrase then.
    lead = None if _get(stats, "score_mismatch", False) else _lead_phrase(e, score, names)
    if lead:
        out.append(lead)
    return out


def _make_streak(shots: list[Any], pid: str, shot_id: str | None) -> int:
    """Trailing run of made field goals by ``pid``; 0 unless the current shot is the latest one."""
    mine = sorted((s for s in shots if _get(s, "player_id") == pid and int(_get(s, "value", 0) or 0) in (2, 3)
                   and _get(s, "made") is not None), key=lambda s: float(_get(s, "t", 0.0) or 0.0))
    if not mine or (shot_id is not None and _get(mine[-1], "shot_id") != shot_id):
        return 0
    n = 0
    for s in reversed(mine):
        if not _get(s, "made"):
            break
        n += 1
    return n


def _lead_phrase(e: Event, score: dict[str, Any], names: dict[Team, str]) -> str | None:
    if e.team is None or score.get("home") is None or not e.value:
        return None
    h, a, v = score["home"], score["away"], int(e.value)
    bh, ba = (h - v, a) if e.team is Team.HOME else (h, a - v)
    me = 1 if e.team is Team.HOME else -1
    before, now = _sign(bh - ba), _sign(h - a)
    if now == 0 and before != 0:
        return f"{names[e.team]} ties it"
    if now == me and before != me:
        return f"{names[e.team]} takes the lead"
    return None


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _foul_phrases(e: Event, stats: Any, pseudo: PseudonymMap, rules: RuleSpec) -> list[str]:
    line = (_get(stats, "players", {}) or {}).get(e.player_id)
    pf = int(_get(line, "pf", 0) or 0)
    tok = pseudo.token(e.player_id or "")
    if pf >= rules.foul_out:
        return [f"{{{tok}}} has fouled out"]
    if pf == rules.foul_out - 1:
        return [f"{{{tok}}} is one foul from fouling out"]
    return []


def _correction_fact(events: list[Event], previous: list[Event] | None, pseudo: PseudonymMap,
                     names: dict[Team, str]) -> dict[str, Any] | None:
    if not previous:
        return None
    prev = {p.event_id: p for p in previous}
    for e in sorted(events, key=lambda ev: _KIND_RANK.get(ev.kind, 99)):
        old = prev.get(e.event_id)
        if old is None:
            continue
        d: dict[str, Any] = {"event_id": e.event_id, "kind": e.kind.value, "old_kind": old.kind.value,
                             "retracted": e.status is EventStatus.RETRACTED}
        if old.player_id:
            d["old_actor"] = pseudo.token(old.player_id)
        if e.player_id:
            d["new_actor"] = pseudo.token(e.player_id)
        if old.value:
            d["old_value"] = int(old.value)
        if e.value:
            d["new_value"] = int(e.value)
        if old.team is not None:
            d["old_team"] = names[old.team]
        if e.team is not None:
            d["new_team"] = names[e.team]
        return d
    return None


_VALUE_NOUN = {1: "free throw", 2: "two", 3: "three"}


def _describe(f: Mapping[str, Any]) -> str:
    """Short, token-only description of one event fact (for ``recent_events``)."""
    who = "{" + f["actor"] + "}" if f.get("actor") else (f.get("team") or "a player")
    other = "{" + f["other"] + "}" if f.get("other") else None
    kind, ok = f["kind"], f.get("outcome_ok", False)
    if f.get("retracted"):
        return f"{kind.replace('_', ' ')} by {who} was withdrawn"
    if kind in ("shot_made", "shot_missed", "block") and not ok:
        return f"{who} shot attempt, result unconfirmed"
    value = f.get("value")
    phrases = {
        "shot_attempt": f"{who} shot attempt" + (f" for {int_to_words(value)}" if value and ok else ""),
        "shot_made": f"{who} made a {_VALUE_NOUN.get(value or 0, 'shot')}",
        "shot_missed": f"{who} missed a shot",
        "block": f"{who} blocked {other or 'a shot'}",
        "steal": f"{who} stole the ball" + (f" from {other}" if other else ""),
        "turnover": f"{who} turnover",
        "rebound": f"{who} {'offensive' if f.get('offensive') else 'defensive'} rebound",
        "assist": f"{who} assist" + (f" to {other}" if other else ""),
        "foul": f"foul on {who}",
        "possession": f"{f.get('team') or who} possession",
        "timeout": f"timeout {f.get('team') or ''}".strip(),
        "substitution": f"{who} in" + (f" for {other}" if other else ""),
        "period_start": "period start",
        "period_end": "period end",
        "score_correction": "score corrected",
    }
    return phrases.get(kind, kind.replace("_", " "))


# --------------------------------------------------------------------------- allowed sets


def _collect_tokens(packet: Mapping[str, Any]) -> list[str]:
    toks: set[str] = set(packet.get("players", {}).keys())

    def walk(obj: Any, key: str | None = None) -> None:
        if key in ("recent_lines", "allowed_tokens", "allowed_numbers"):
            return
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                if k in ("actor", "other", "old_actor", "new_actor") and isinstance(v, str):
                    toks.add(v)
                walk(v, k)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v, key)
        elif isinstance(obj, str):
            toks.update(TOKEN_RE.findall(obj))

    walk(packet)
    return sorted(toks)


def _collect_numbers(packet: Mapping[str, Any]) -> list[str]:
    forms: set[str] = set()

    def add_mentions(s: str) -> None:
        for m in extract_numbers(s):
            kind, _, body = m.key.partition(":")
            if kind == "card":
                forms.update(number_forms(float(body) if "." in body else int(body)))
            elif kind == "ord":
                n = int(body)
                forms.update(number_forms(n, ordinal=True)[2:])
            elif kind in ("clock", "pct"):
                forms.add(m.text if kind == "clock" else f"{body}%")
            elif kind in ("pair", "dash"):
                x, y = body.replace("/", "-").split("-")
                forms.add(f"{x}-for-{y}" if kind == "pair" else f"{x}-{y}")
                forms.update(number_forms(int(x)) + number_forms(int(y)))

    def walk(obj: Any, key: str | None = None) -> None:
        if key in _SKIP_KEYS:
            return
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                walk(v, k)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v, key)
        elif isinstance(obj, bool) or obj is None:
            return
        elif isinstance(obj, (int, float)):
            forms.update(number_forms(obj, ordinal=key == "period"))
        elif isinstance(obj, str):
            add_mentions(obj)

    walk(packet)
    return sorted(forms)
