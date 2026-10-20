"""EPV-lite decision grading for the ball handler.

**This is a hand-built heuristic, not the EPV model of Cervone et al. (JASA 2016).** There are no
fitted transition hazards, no spatial terms and no player effects. At the moment of a PASS or a
SHOT_ATTEMPT it prices a few options in expected points (offense points from this action on):

* ``shoot``: the holder's shot from where they stand, priced by ``ShotQualityModel`` with the
  current closest defender, plus the value of a miss (offensive-rebound rate x putback PPP).
* ``pass`` to each teammate: P(complete) from the pass length and the defender closest to the
  passing lane (point-to-segment distance), times the receiver's catch-and-shoot shot value, with
  the receiver's closest-defender distance reduced by closeout speed x pass flight time. A
  completed pass is worth at least the reset value (``receiver_can_reset``): a receiver with a bad
  shot keeps the possession going, so a safe ball reversal is not graded as a forced bad shot. A
  failed pass is worth ``turnover_value`` (0: the possession is lost).
* ``hold``: reset the action. League points per possession, scaled down as the shot clock (or the
  game clock, whichever is shorter) runs out, when the clock is known.

``grade()`` compares the chosen option with the best one: ``delta = chosen - best`` (<= 0), labelled
good / neutral / poor by thresholds in ``DecisionConfig``. Grades are coach-facing and must stay
reproducible and explainable (jev_integration_design.md site 4: Jev never scores decisions).
Every constant records its source and status; most pass and closeout numbers are unverified.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from courtvision.geometry.court import CourtSpec
from courtvision.stats.shot_quality import ShotFeatures, ShotQualityModel
from courtvision.types import Event, EventKind, FrameState, PlayerState, Side, Team

__all__ = ["DecisionConfig", "DecisionGrade", "DecisionModel", "DecisionOption", "point_segment_distance"]

log = logging.getLogger(__name__)

Point = tuple[float, float]


@dataclass
class DecisionOption:
    """One priced option. ``kind`` is "shoot" | "pass" | "hold"; ``value`` is in expected points."""

    kind: str
    target_id: str | None
    success_prob: float  # shoot: xFG; pass: P(complete); hold: 1
    xpts_if_success: float  # shoot: points if made; pass: max(receiver's shot value, reset value); hold: the hold value
    value: float
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def same_as(self, other: DecisionOption) -> bool:
        return self.kind == other.kind and self.target_id == other.target_id


@dataclass
class DecisionGrade:
    """Chosen vs best option for one PASS / SHOT_ATTEMPT."""

    t: float
    player_id: str
    chosen: DecisionOption
    best: DecisionOption
    options: list[DecisionOption]
    delta: float  # chosen.value - best.value (<= 0)
    label: str  # "good" | "neutral" | "poor"
    event_id: str | None = None  # addition: lets consumers replace a grade when the event is revised
    event_kind: str | None = None  # addition: "pass" | "shot_attempt"

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "player_id": self.player_id,
            "event_id": self.event_id,
            "event_kind": self.event_kind,
            "chosen": self.chosen.to_dict(),
            "best": self.best.to_dict(),
            "options": [o.to_dict() for o in self.options],
            "delta": self.delta,
            "label": self.label,
        }


@dataclass
class DecisionConfig:
    """EPV-lite constants. Plain fields for YAML loading."""

    # Passes
    pass_speed_fts: float = 35.0  # average ball speed of a pass (~24 mph); unverified, tune on tracking
    pass_release_s: float = 0.15  # wind-up before the ball leaves the hands; unverified
    lead_receiver: bool = True  # price the catch where the receiver will be (pos + v * flight time)
    # The lane starts this far from the passer, so the on-ball defender (who guards from 3-6 ft and is
    # passed around / over, not through) is not "in the lane". 6 ft = NCAA "closely guarded" (rules
    # note §3). At 3 ft the on-ball defender (median 4.4 ft from the passer on simulated passes) was
    # the lane defender on most passes; placement sourced, effect unverified
    lane_ignore_near_passer_ft: float = 6.0
    lane_ignore_near_receiver_ft: float = 2.0  # ... and ends this far short of the catch point, so the
    # receiver's own defender beside / behind them (already priced by the closeout term) is not "in
    # the lane"; a fronting defender still is; unverified
    complete_base_logit: float = 4.2  # logit P(complete) for a short, open pass (~.985); unverified.
    # NBA bad-pass turnovers are a small share of passes (TOV/poss .146, 58% live-ball, several
    # passes per possession; rules note §4c), so open passes complete almost always (inference)
    complete_per_10ft_logit: float = -0.35  # longer passes fail more (40 ft ~ .94 open); unverified
    lane_defender_logit: float = -3.0  # extra log-odds when a defender sits on the lane; unverified
    lane_defender_scale_ft: float = 2.0  # e-folding distance of that penalty; unverified
    closeout_speed_fts: float = 10.0  # receiver's defender closes this fast while the ball travels; unverified
    turnover_value: float = 0.0  # offense points after a failed pass (possession lost); EPV-lite convention
    # A completed pass is worth max(receiver's shot, reset value): the receiver may also hold. Off,
    # every pass is priced as a forced catch-and-shoot and routine reversals grade "poor" (on a
    # simulated 10-min NBA game: 90 of 154 passes, median delta -0.33); design choice
    receiver_can_reset: bool = True
    self_match_ft: float = 1.0  # a teammate this close to a holder found only by the event x/y is
    # treated as the holder under another track id, not a pass target; tuned
    # Shots
    include_miss_value: bool = True  # add (1 - xFG) x OREB rate x putback PPP so shots compare with "hold"
    # OREB% after a missed FGA by zone, pbpstats 2025-26 (rules note §4c); sourced (NBA)
    oreb_at_rim: float = 0.387  # distance < 4 ft
    oreb_short_mid: float = 0.320  # 2s, 4-14 ft
    oreb_long_mid: float = 0.243  # 2s, >= 14 ft
    oreb_three: float = 0.273  # corner and above-the-break 3s (both .273 in 2025-26)
    putback_ppp: float = 1.10  # Synergy "Putbacks" placeholder PPP ~1.08-1.15 (§5); unverified
    # Hold / reset: league points per possession. NBA 1.160 (pbpstats 2025-26, §4c); sourced.
    # Other leagues: unverified placeholders (no league PPP in the notes).
    league_ppp: dict[str, float] = field(default_factory=lambda: {
        "nba": 1.160,
        "fiba": 1.05,
        "ncaa_m": 1.03,
        "ncaa_w": 0.92,
        "nfhs": 0.85,
    })
    hold_full_s: float = 12.0  # shot clock >= this: a reset is worth a full possession; unverified
    hold_zero_s: float = 1.0  # shot clock <= this: a reset is worth nothing; unverified
    # The reset value also runs down with the game clock when it is shorter than the shot clock
    # (end of period, or the shot clock is off: rules note §3 "Shot clock off"); design choice
    hold_uses_game_clock: bool = True
    # Grade labels on delta = chosen - best (expected points)
    good_delta: float = -0.05  # >= this is "good" (jev_integration_design site 4 "best_option"; proposal)
    poor_delta: float = -0.20  # < this is "poor" (spec; proposal)


def point_segment_distance(p: Point, a: Point, b: Point) -> tuple[float, float]:
    """Distance from ``p`` to segment ``a``-``b`` and the clamped segment parameter s in [0, 1]."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    seg2 = dx * dx + dy * dy
    s = 0.0 if seg2 <= 0 else min(max(((p[0] - ax) * dx + (p[1] - ay) * dy) / seg2, 0.0), 1.0)
    return math.hypot(p[0] - (ax + s * dx), p[1] - (ay + s * dy)), s


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass(frozen=True)
class _ShotHint:
    """Attempt details known from a SHOT_ATTEMPT event (override the frame-derived defaults)."""

    value: int | None = None
    shot_type: str | None = None
    catch_and_shoot: bool | None = None
    closest_defender_ft: float | None = None


class DecisionModel:
    """Prices shoot / pass / hold options and grades the chosen one (EPV-lite heuristic)."""

    def __init__(self, court: CourtSpec, shot_model: ShotQualityModel, config: DecisionConfig | None = None) -> None:
        self.court = court
        self.shot_model = shot_model
        self.config = config or DecisionConfig()
        if shot_model.league not in self.config.league_ppp:
            raise ValueError(f"no league_ppp for league {shot_model.league!r}")

    # ------------------------------------------------------------------ public API

    def options(self, frame: FrameState, holder_id: str, offense: Team, basket: Side) -> list[DecisionOption]:
        """shoot, pass to each located teammate, and hold; empty when the holder is not in the frame."""
        holder = frame.player(holder_id)
        if holder is None or not _located(holder):
            return []
        return self._options(frame, (holder.x, holder.y), holder_id, offense, basket, _ShotHint())

    def grade(self, frame: FrameState, event: Event, offense: Team, basket: Side) -> DecisionGrade | None:
        """Grade a PASS (chosen = pass to ``other_player_id``) or a SHOT_ATTEMPT (chosen = shoot).

        Returns None for other kinds, free throws, a missing passer / shooter, or a receiver who is
        not a located teammate. Uses the frame's positions; for a shot the event's attrs (value,
        shot_type, catch_and_shoot, closest_defender_ft) refine the shoot option.
        """
        if event.kind not in (EventKind.PASS, EventKind.SHOT_ATTEMPT) or event.player_id is None:
            return None
        hint = _ShotHint()
        if event.kind is EventKind.SHOT_ATTEMPT:
            if event.value == 1 or event.attrs.get("shot_type") == "free_throw":
                return None
            hint = _hint_from_event(event)
        holder = frame.player(event.player_id)
        located = holder is not None and _located(holder)
        if located:
            pos: Point = (holder.x, holder.y)
        elif event.x is not None and event.y is not None and math.isfinite(event.x) and math.isfinite(event.y):
            pos = (float(event.x), float(event.y))
        else:
            return None
        opts = self._options(frame, pos, event.player_id, offense, basket, hint, holder_located=located)
        if event.kind is EventKind.SHOT_ATTEMPT:
            chosen = next((o for o in opts if o.kind == "shoot"), None)
        else:
            chosen = next((o for o in opts if o.kind == "pass" and o.target_id == event.other_player_id), None)
        if chosen is None:
            log.debug("cannot grade %s: chosen option not available", event.event_id)
            return None
        best = max(opts, key=lambda o: o.value)
        delta = chosen.value - best.value
        return DecisionGrade(t=event.t, player_id=event.player_id, chosen=chosen, best=best, options=opts,
                             delta=delta, label=self.label_for(delta), event_id=event.event_id,
                             event_kind=event.kind.value)

    def label_for(self, delta: float) -> str:
        if delta >= self.config.good_delta:
            return "good"
        if delta < self.config.poor_delta:
            return "poor"
        return "neutral"

    # ------------------------------------------------------------------ options

    def _options(self, frame: FrameState, pos: Point, holder_id: str, offense: Team, basket: Side,
                 hint: _ShotHint, holder_located: bool = True) -> list[DecisionOption]:
        located = [p for p in frame.players if _located(p)]  # NaN positions (bad homography) are dropped
        teammates = [p for p in located if p.team == offense and p.player_id != holder_id]
        if not holder_located:  # the holder may be in the frame under a provisional track id
            teammates = [p for p in teammates
                         if math.hypot(p.x - pos[0], p.y - pos[1]) > self.config.self_match_ft]
        defenders = [p for p in located if p.team == offense.other()]
        hold = self._hold_option(frame)
        opts = [self._shoot_option(pos, defenders, basket, hint)]
        opts += [self._pass_option(pos, r, defenders, basket, frame) for r in teammates]
        opts.append(hold)
        return opts

    def _shoot_option(self, pos: Point, defenders: list[PlayerState], basket: Side, hint: _ShotHint) -> DecisionOption:
        d_def = hint.closest_defender_ft if hint.closest_defender_ft is not None else _closest(pos, defenders)
        f = self._features(pos, basket, d_def, hint.catch_and_shoot, hint.shot_type or "jumper", hint.value)
        xfg, shot_value, detail = self._shot_value(f)
        detail["closest_defender_ft"] = d_def
        return DecisionOption("shoot", None, xfg, float(f.value), shot_value, detail)

    def _pass_option(self, pos: Point, receiver: PlayerState, defenders: list[PlayerState], basket: Side,
                     frame: FrameState) -> DecisionOption:
        cfg = self.config
        catch, flight_s = self._catch_point(pos, receiver)
        length = math.hypot(catch[0] - pos[0], catch[1] - pos[1])
        lane_d, lane_id = self._lane_defender(pos, catch, length, defenders)
        z = cfg.complete_base_logit + cfg.complete_per_10ft_logit * length / 10.0
        if lane_d is not None:
            z += cfg.lane_defender_logit * math.exp(-lane_d / cfg.lane_defender_scale_ft)
        p_complete = _sigmoid(z)
        d_now = _closest(catch, defenders)
        d_catch = None if d_now is None else max(0.0, d_now - cfg.closeout_speed_fts * flight_s)
        f = self._features(catch, basket, d_catch, True, "jumper", None)
        xfg, shot_value, detail = self._shot_value(f)
        reset_value = self._hold_value(frame, elapsed_s=flight_s)[0] if cfg.receiver_can_reset else 0.0
        if_complete = max(shot_value, reset_value)
        detail.update({
            "pass_ft": length,
            "flight_s": flight_s,
            "catch_xy": [catch[0], catch[1]],
            "lane_defender_ft": lane_d,
            "lane_defender_id": lane_id,
            "receiver_defender_now_ft": d_now,
            "receiver_defender_at_catch_ft": d_catch,
            "receiver_xfg": xfg,
            "receiver_shot_value": shot_value,
            "receiver_reset_value": reset_value,
            "receiver_action": "shoot" if shot_value >= reset_value else "reset",
        })
        value = p_complete * if_complete + (1.0 - p_complete) * cfg.turnover_value
        return DecisionOption("pass", receiver.player_id, p_complete, if_complete, value, detail)

    def _hold_option(self, frame: FrameState) -> DecisionOption:
        value, detail = self._hold_value(frame)
        return DecisionOption("hold", None, 1.0, value, value, detail)

    def _hold_value(self, frame: FrameState, elapsed_s: float = 0.0) -> tuple[float, dict[str, Any]]:
        """League PPP x a factor that falls from 1 to 0 as the effective clock runs out."""
        cfg = self.config
        ppp = cfg.league_ppp[self.shot_model.league]
        sb = frame.scoreboard
        sc = _finite(sb.shot_clock) if sb is not None else None
        gc = _finite(sb.game_clock) if sb is not None and cfg.hold_uses_game_clock else None
        known = [c for c in (sc, gc) if c is not None]
        clock = min(known) - elapsed_s if known else None
        factor = 1.0
        if clock is not None:
            span = max(cfg.hold_full_s - cfg.hold_zero_s, 1e-9)
            factor = min(max((clock - cfg.hold_zero_s) / span, 0.0), 1.0)
        detail = {"league_ppp": ppp, "shot_clock": sc, "game_clock": gc, "clock_used": clock,
                  "shot_clock_factor": factor}
        return ppp * factor, detail

    # ------------------------------------------------------------------ pieces

    def _features(self, pos: Point, basket: Side, d_def: float | None, catch_and_shoot: bool | None,
                  shot_type: str, value: int | None) -> ShotFeatures:
        x, y = pos
        v = value if value in (2, 3) else (3 if self.court.is_three(x, y, basket) else 2)
        return ShotFeatures(distance_ft=self.court.distance_to_basket(x, y, basket), value=int(v),
                            shot_type=shot_type, closest_defender_ft=d_def, catch_and_shoot=catch_and_shoot,
                            zone=self.court.zone(x, y, basket))

    def _shot_value(self, f: ShotFeatures) -> tuple[float, float, dict[str, Any]]:
        """(xFG, expected points incl. the miss continuation, detail)."""
        xfg = self.shot_model.xfg(f)
        made_pts = xfg * f.value
        miss = (1.0 - xfg) * self._oreb_rate(f) * self.config.putback_ppp if self.config.include_miss_value else 0.0
        detail = {"distance_ft": f.distance_ft, "points": f.value, "zone": f.zone, "xfg": xfg,
                  "made_xpts": made_pts, "miss_value": miss, "catch_and_shoot": f.catch_and_shoot,
                  "shot_type": f.shot_type}
        return xfg, made_pts + miss, detail

    def _oreb_rate(self, f: ShotFeatures) -> float:
        cfg = self.config
        if f.value == 3:
            return cfg.oreb_three
        if f.distance_ft < 4.0:  # pbpstats AT_RIM_CUTOFF = 4 ft [37]
            return cfg.oreb_at_rim
        return cfg.oreb_short_mid if f.distance_ft < 14.0 else cfg.oreb_long_mid  # SHORT_MID_RANGE_CUTOFF [37]

    def _catch_point(self, pos: Point, receiver: PlayerState) -> tuple[Point, float]:
        cfg = self.config
        speed = max(cfg.pass_speed_fts, 1e-6)
        flight = cfg.pass_release_s + math.hypot(receiver.x - pos[0], receiver.y - pos[1]) / speed
        if not cfg.lead_receiver:
            return (receiver.x, receiver.y), flight
        catch = (receiver.x + receiver.vx * flight, receiver.y + receiver.vy * flight)
        flight = cfg.pass_release_s + math.hypot(catch[0] - pos[0], catch[1] - pos[1]) / speed
        return catch, flight

    def _lane_defender(self, pos: Point, catch: Point, length: float,
                       defenders: list[PlayerState]) -> tuple[float | None, str | None]:
        """Closest defender to the lane: the pass segment minus ``lane_ignore_near_passer_ft`` at the
        passer end and ``lane_ignore_near_receiver_ft`` at the catch end (each at most half the pass)."""
        if not defenders:
            return None, None
        if length > 0:
            cfg = self.config
            skip_a = min(max(cfg.lane_ignore_near_passer_ft, 0.0), length / 2.0)
            skip_b = min(max(cfg.lane_ignore_near_receiver_ft, 0.0), length / 2.0)
            ux, uy = (catch[0] - pos[0]) / length, (catch[1] - pos[1]) / length
            start = (pos[0] + ux * skip_a, pos[1] + uy * skip_a)
            end = (catch[0] - ux * skip_b, catch[1] - uy * skip_b)
        else:
            start = end = pos
        best_d, best_id = math.inf, None
        for d in defenders:
            dist, _ = point_segment_distance((d.x, d.y), start, end)
            if dist < best_d:
                best_d, best_id = dist, d.player_id
        return best_d, best_id


def _located(p: PlayerState) -> bool:
    return math.isfinite(p.x) and math.isfinite(p.y) and math.isfinite(p.vx) and math.isfinite(p.vy)


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _closest(pos: Point, players: list[PlayerState]) -> float | None:
    if not players:
        return None
    return min(math.hypot(p.x - pos[0], p.y - pos[1]) for p in players)


def _hint_from_event(event: Event) -> _ShotHint:
    a = event.attrs
    cs = a.get("catch_and_shoot")
    cd = a.get("closest_defender_ft")
    return _ShotHint(
        value=int(event.value) if event.value in (2, 3) else None,
        shot_type=a.get("shot_type") or None,
        catch_and_shoot=None if cs is None else bool(cs),
        closest_defender_ft=None if cd is None else float(cd),
    )
