"""Deterministic streaming basketball event engine (ARCHITECTURE "events/").

``EventEngine.process(frame)`` turns one ``FrameState`` into the new events and revisions it
produces. It never looks past its own small buffers (the last ~3 s of ball samples).

Pipeline per frame
------------------
1. Scoreboard: ``ScoreboardConfirmer`` confirms score / period changes (D14); period changes and a
   confirmed game-clock expiry (``ClockZeroWatch``) give PERIOD_START / PERIOD_END; score changes
   go to the ``Reconciler``.
2. Ball: ``BallGate`` rejects physically impossible jumps (false detections).
3. Holder: ``HolderTracker`` (hysteresis) reports releases and gains of control.
4. Flights: a release opens a ``Flight`` (``shots.py``) classified as it flies: SHOT_ATTEMPT when
   it enters a rim zone (timestamped at release), SHOT_MADE / SHOT_MISSED, BLOCK; a gain closes it
   as PASS, STEAL + TURNOVER, REBOUND, ...
5. Phases: LIVE play; DEAD after a made basket, an out-of-bounds ball, a clock stop, a period
   break or between free throws. A DEAD phase ends at the inbound (a release by a holder standing
   out of bounds), a jump ball, a free throw, or live-play evidence; the inbounding team decides
   POSSESSION, inferred out-of-bounds TURNOVERs (confidence < 1) and team REBOUNDs.
6. Dead balls for narration: ``DeadBallDetector`` (§7A.5).
7. Reconciliation against the confirmed scoreboard (``reconcile.py``).

Event semantics follow ARCHITECTURE "Shared event semantics" and the FIBA Statisticians' Manual
rules in docs/research/basketball_rules_stats.md §2a. Every tunable constant is a field of
``EventEngineConfig`` with its source and status (sourced / unverified / tuned).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import statistics
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from courtvision.geometry.court import CourtSpec
from courtvision.geometry.rules import RuleSpec, rules_for
from courtvision.types import (
    BBox,
    Event,
    EventKind,
    EventStatus,
    FrameState,
    PlayerState,
    Roster,
    Side,
    Team,
)

from .calibration import CueModelSet
from .deadball import DeadBallConfig, DeadBallDetector, ft_formation
from .possession import BallGate, BallObs, ControlChange, HolderTracker, PlayerHistory, RayModel, ball_obs, keep_box
from .reconcile import Action, Reconciler, ShotEntry
from .scoreboard import ClockZeroWatch, PeriodChange, ScoreboardConfig, ScoreboardConfirmer, ScoreChange
from .shots import Flight, FlightSample, FlightSignal, Release, shot_type_for, three_margin_ft

__all__ = ["EventEngine", "EventEngineConfig"]

log = logging.getLogger(__name__)

LIVE, DEAD = "live", "dead"


@dataclass
class EventEngineConfig:
    """All event-engine thresholds. Each carries its source and status (RESEARCH §7A.2 registry).

    ``scoreboard`` and ``deadball`` are field overrides for ``ScoreboardConfig`` /
    ``DeadBallConfig`` (plain dicts so the integration layer can load them from YAML).
    """

    # --- ball observations and gating
    ball_min_conf: float = 0.2  # ignore ball detections below this confidence; tuned
    use_interpolated_ball: bool = False  # tracker-filled positions are not evidence of a crossing; tuned
    ball_max_speed_fts: float = 60.0  # fastest plausible ball (hard passes ~40-50 ft/s) + margin; unverified
    ball_max_speed_px_s: float = 1500.0  # image gate speed when no px/ft scale is known; tuned
    ball_gate_base_ft: float = 3.0  # 3-D gate radius at dt = 0; tuned
    ball_gate_base_px: float = 25.0  # image gate radius at dt = 0; tuned
    ball_reacquire_frames: int = 3  # consecutive agreeing off-gate detections to re-acquire; tuned
    ball_gate_reset_s: float = 1.0  # gate opens after this long without an accepted ball; tuned
    player_height_ft: float = 6.5  # nominal player height for px/ft from bbox heights; unverified
    nominal_px_per_ft: float = 10.0  # 3-D residuals -> "px" cue units; tuned (sim camera ~8-11 px/ft)
    hands_height_ft: float = 5.0  # nominal hands height for release separation; unverified
    # --- holder detection (possession.py)
    holder_bbox_expand: float = 0.25  # bbox growth to acquire the ball (spec "~25%"); unverified
    holder_keep_expand: float = 0.6  # wider box to keep it (spatial hysteresis, dribbles); tuned
    holder_keep_top_frac: float = 0.05  # keep box reaches this fraction of bbox height above the head; tuned
    holder_keep_bottom_frac: float = 0.15  # ... and this far below the feet (bounces); tuned
    holder_head_frac: float = 0.15  # top fraction of the bbox treated as the head; unverified
    holder_above_head_factor: float = 0.4  # score multiplier for a ball above the head; tuned
    holder_min_score: float = 0.1  # minimum candidate score; tuned
    holder_switch_margin: float = 0.25  # current holder's score bonus (no flicker); tuned
    holder_radius_ft: float = 1.6  # 0.5 m holder radius (RESEARCH §7A.2); unverified
    holder_keep_radius_ft: float = 3.0  # keep radius for dribble excursions; tuned
    holder_max_z_ft: float = 7.5  # a held ball is below this height (spec); unverified
    holder_freefall_min_z_ft: float = 4.5  # 3-D: above dribble height (catches and dribbles near the floor are partly ballistic) ... ; tuned
    holder_freefall_tol_fts2: float = 8.0  # ... a ball with |a_z + g| below this is in free flight, nobody gains it; tuned
    holder_acquire_max_speed_fts: float = 15.0  # a catch slows the ball to the catcher's speed; tuned
    holder_rel_speed_scale_fts: float = 10.0  # 3-D: score -= relative ball-player speed / this; tuned
    holder_rel_vertical_weight: float = 0.5  # weight of vertical motion in that speed during a pass (catch-then-dribble); tuned
    holder_rel_vertical_weight_loose: float = 1.0  # ... for a loose ball (rebounds, fumbles); tuned
    holder_rel_vertical_weight_held: float = 0.0  # ... for challengers of a current holder (a dribble's vertical motion belongs to whoever controls it; the holder's own score ignores it); tuned
    holder_rel_speed_scale_img_fts: float = 25.0  # image: same, noisier velocities; tuned
    holder_rel_speed_slack_img_fts: float = 10.0  # image: extra slack on the hard acquire gate (noise); tuned
    holder_team_prior: float = 0.15  # score bonus for the passer's team while a pass is in the air; tuned
    holder_switch_margin_opp: float = 0.25  # image: extra margin an opponent needs to take the ball from the holder; tuned
    holder_switch_margin_opp_court: float = 0.0  # 3-D / court: positions are exact, relative motion decides; tuned
    use_ray_model: bool = True  # monocular: estimate the camera from rims + ball floor projections; tuned
    holder_ray_max_lateral_ft: float = 2.5  # image: the ball's viewing ray passes this close to the holder; tuned
    holder_ray_z_slack_ft: float = 1.5  # image: implied height may leave [0, holder_max_z] by this; tuned
    holder_ray_keep_slack_ft: float = 1.5  # extra slack for the current holder; tuned
    holder_ray_z_ref_ft: float = 2.5  # typical held / dribbled ball height; unverified
    holder_ray_z_band_ft: float = 2.0  # no penalty within ref +- band; tuned
    holder_ray_z_weight: float = 0.1  # score penalty per ft outside the band; tuned
    steal_confirm_s: float = 0.3  # a change of team control needs this long (narration §9.3 holds 500 ms); tuned
    holder_contest_margin: float = 0.02  # opponents within this score of each other: contested, no holder; tuned
    holder_confirm_s: float = 0.1  # new holder after ~3 frames at 30 fps; tuned
    holder_release_s: float = 0.1  # release after ~3 frames without the ball; tuned
    holder_min_frames: int = 2  # floor on hysteresis frame counts at low fps; tuned
    holder_gap_keep_s: float = 10.0  # keep the holder through ball gaps up to this (cv_pipeline §6.2); tuned
    id_swap_radius_ft: float = 2.5  # a vanished holder replaced in place by a teammate id = id swap; tuned
    self_regain_s: float = 1.5  # releaser regains within this: same control stint (dribble excursion); tuned
    # --- flights, shots and make / miss (shots.py)
    rim_zone_radius_ft: float = 4.5  # 3-D shot zone radius around the basket (spec); unverified
    rim_zone_below_ft: float = 1.0  # zone floor = rim height - this (spec); unverified
    rim_zone_exit_margin_ft: float = 0.5  # sideways exit beyond the zone radius + this; tuned
    shot_aim_radius_ft: float = 3.5  # 3-D: a shot's arc comes down within this of the basket (off-glass incl.); tuned
    ball_radius_ft: float = 0.39  # 9.39 in diameter / 2 (rules note §1a); sourced
    make_tol_ft: float = 0.1  # made radius = rim_r - 0.5 ball_r + this (spec); tuned
    rim_contact_dist_ft: float = 0.15  # ball surface within this of the ring = rim contact; tuned
    rim_zone_side_k: float = 1.25  # image entry zone half-width beyond the rim box, in rim widths; tuned
    rim_zone_up_k: float = 4.0  # image entry zone height above the rim top, in rim widths; tuned
    rim_zone_down_k: float = 0.25  # image entry zone extension below the rim box; tuned
    rim_zone_min_samples_img: int = 2  # image samples inside the zone before a shot is called; tuned
    rim_exit_side_k: float = 2.5  # image miss: ball this many rim widths beside the rim; tuned
    rim_exit_down_k: float = 1.0  # image miss: ball this many rim widths below the rim box; tuned
    make_span_tol_px: float = 1.0  # tolerance on the shrunk rim span (px); tuned
    made_within_k: float = 0.5  # "seen below the rim" must lie within the rim box +- this * width; tuned
    made_below_k: float = 0.25  # "below the rim" = below the rim box by this * width; tuned
    made_bounce_k: float = 0.25  # back above the top line by this * width cancels a crossing; tuned
    drop_band: tuple[float, float] = (0.1, 0.9)  # make drop band, fraction of free fall [16]; sourced
    freefall_min_frac: float = 0.9  # >= this = air-ball free-fall signature [16]; sourced
    shot_resolve_timeout_s: float = 4.0  # unresolved shot -> missed + needs_review; tuned
    pass_max_flight_s: float = 4.0  # a "pass" flight longer than this is a loose ball; tuned
    gather_lead_s: float = 0.35  # last grounded foot position ~ this before release (D8 proxy); unverified
    tip_max_control_s: float = 0.35  # shorter control near the rim = tip; tuned
    layup_max_dist_ft: float = 6.0  # tuned
    dunk_min_release_z_ft: float = 9.5  # tuned
    dunk_max_dist_ft: float = 3.0  # tuned
    catch_shoot_min_dist_ft: float = 10.0  # NBA catch-and-shoot uses jumpers > 10 ft (unverified)
    block_min_up_speed_fts: float = 6.0  # shot-like release rises at least this fast; tuned
    block_min_toward_fts: float = 3.0  # before a block the ball moves toward the rim at least this fast; tuned
    block_reverse_dv_fts: float = 8.0  # velocity change beyond ballistic for a block; tuned
    block_reverse_dv_img_fts: float = 18.0  # image mode: same, above the velocity noise of ~1.5 px jitter; tuned
    block_image_span: int = 2  # image mode: frames per velocity baseline around the contact; tuned
    block_defender_radius_ft: float = 4.0  # defender this close to the ball at the reversal; tuned
    block_min_z_ft: float = 5.0  # above shoulder height = block, below = steal [15]; unverified height
    block_bbox_expand: float = 0.4  # image: defender bbox horizontal growth for the contact test; tuned
    block_reach_frac: float = 0.4  # image: raised-arm reach above the head, fraction of bbox height; unverified
    block_shoulder_frac: float = 0.35  # image: contact in the top fraction of the defender bbox; tuned
    block_max_after_release_s: float = 1.2  # tuned
    block_max_from_release_ft: float = 10.0  # tuned
    block_max_basket_dist_ft: float = 35.0  # shot-like releases only within this of a basket; tuned
    airball_radius_ft: float = 7.0  # 3-D: descending past rim height within this of the basket; tuned
    airball_min_apex_above_ft: float = 1.0  # tuned
    airball_min_release_dist_ft: float = 6.0  # tuned
    strip_max_travel_ft: float = 6.0  # opponent gains within this of the release: steal, not bad pass; tuned
    strip_max_flight_s: float = 0.6  # tuned
    # --- dribbles
    dribble_floor_z_ft: float = 1.2  # bounce minimum below this (ball centre 0.39 ft at contact); tuned
    dribble_min_amp_ft: float = 0.8  # 3-D reversal amplitude of a bounce; tuned
    dribble_radius_ft: float = 4.0  # bounce within this of the dribbler; tuned
    dribble_foot_band: float = 0.25  # image: bounce within this fraction of bbox height of the feet; tuned
    dribble_min_amp_h: float = 0.15  # image: reversal amplitude as a fraction of bbox height; tuned
    dribble_min_interval_s: float = 0.2  # tuned
    # --- assists (ARCHITECTURE shared semantics)
    assist_window_s: float = 5.0  # catch -> release; sourced (project rule)
    assist_max_dribbles: int = 2  # sourced (project rule)
    assist_on_ft_trips: bool = False  # NBA flag; FIBA counts FT trips [15]; sourced
    # --- free throws, dead balls and possession
    ft_min_hold_s: float = 0.5  # formation held this long before a release counts as an FT; tuned
    oob_margin_ft: float = 1.0  # ball / holder beyond the boundary by this = out of bounds; tuned
    oob_max_z_ft: float = 4.0  # 3-D: the ball must be this low to count as out; tuned
    inbound_holder_margin_ft: float = 0.5  # holder this far outside the lines = inbounder; tuned
    holder_oob_frames: int = 3  # live holder out of bounds this many frames = stepped out; tuned
    oob_turnover_conf: float = 0.6  # inferred out-of-bounds turnovers (< 1 per semantics); tuned
    team_rebound_conf: float = 0.7  # inferred team rebounds; tuned
    dead_min_s: float = 1.0  # fallback restarts (dribble / pass in bounds) only after this; tuned
    dead_from_clock_stop: bool = True  # enter a dead ball when the trusted game clock stops; tuned
    clock_stop_min_s: float = 1.2  # clock unchanged this long (MM:SS display) = stopped; tuned
    detect_shot_clock_violations: bool = True  # shot clock read 0 while live -> team turnover; tuned
    period_start_tol_s: float = 1.0  # first read within this of a full period = period start; tuned
    toss_min_speed_fts: float = 5.0  # an unheld ball moving this fast at a period start = the jump-ball toss; tuned
    possession_conf: float = 0.9  # confidence of POSSESSION events from control; tuned
    pass_conf: float = 0.9  # tuned
    dribble_conf: float = 0.8  # tuned
    # --- attacking side
    default_attacking: dict[str, str] = field(default_factory=lambda: {"home": "right", "away": "left"})  # assumption until shots are seen
    # --- reconciliation (reconcile.py)
    make_confirm_wait_s: float = 3.0  # make-wait band (§7A.5 / narration §11.2); sourced (placeholder)
    make_to_miss_conf: float = 0.7  # max confidence of a make revised to a miss for lack of a delta (needs_review); tuned
    reconcile_match_back_s: float = 10.0  # delta may lag the make by up to this; unverified
    reconcile_match_fwd_s: float = 3.0  # ... or lead it (engine decides late) by this; tuned
    reconcile_flip_window_s: float = 15.0  # misses this old can be flipped to makes; tuned
    reconcile_hold_s: float = 2.0  # unmatched delta waits this long before a SCORE_CORRECTION; tuned
    reconcile_inflight_max_s: float = 5.0  # max wait for an in-flight shot of the scoring team; tuned
    # --- calibration
    assume_calibrated: bool = False  # simulator evaluation only: p_status "assumed_sim"
    cue_model_path: str | None = None  # CueModelSet JSON; None = hand-set cold start
    # --- sub-module overrides
    scoreboard: dict[str, Any] = field(default_factory=dict)  # ScoreboardConfig field overrides
    deadball: dict[str, Any] = field(default_factory=dict)  # DeadBallConfig field overrides


# ---------------------------------------------------------------------------- internal records


@dataclass(slots=True)
class _PassRec:
    passer: str
    team: Team | None
    release_t: float
    event_id: str


@dataclass(slots=True)
class _Control:
    """One holding stint of one player."""

    player_id: str
    team: Team | None
    onset_t: float
    onset_frame: int
    origin: str = "other"  # "pass" | "rebound" | "steal" | "inbound" | "self" | "other"
    via_pass: _PassRec | None = None
    dribbles: int = 0


@dataclass(slots=True)
class _Dead:
    cause: str  # "made_fg" | "made_ft" | "oob" | "unknown" | "shot_clock" | "ft_trip" | "period" | "break"
    since_t: float
    since_frame: int
    scoring_team: Team | None = None
    shot_id: str | None = None
    miss: ShotEntry | None = None
    last_team: Team | None = None
    last_holder: str | None = None
    out_t: float | None = None
    out_frame: int | None = None
    first_control: dict[Team, tuple[float, int, str]] = field(default_factory=dict)
    first_pickup: tuple[float, int, str] | None = None
    toss_seen: bool = False  # period start: the ball moved while nobody held it (jump-ball toss)


@dataclass(slots=True)
class _FtTrip:
    shooter: str | None
    team: Team | None
    total: int | None  # None = unknown (no FOUL event)
    taken: int = 0
    live_last: bool = True  # False for technical / flagrant trips (no rebound, possession kept)
    ft_type: str = "shooting"
    assist_candidate: tuple[str, Team | None] | None = None
    assisted: bool = False


class _Extremum:
    """Local-minimum detector with hysteresis amplitude (bounce detection)."""

    def __init__(self) -> None:
        self.down = True
        self.best: tuple[float, float, int, float] | None = None  # (value, t, frame, aux)

    def reset(self) -> None:
        self.down, self.best = True, None

    def push(self, val: float, t: float, frame: int, amp: float, aux: float) -> tuple[float, float, int, float] | None:
        b = self.best
        if b is None:
            self.best = (val, t, frame, aux)
            return None
        if self.down:
            if val < b[0]:
                self.best = (val, t, frame, aux)
            elif val > b[0] + amp:
                self.down, self.best = False, (val, t, frame, aux)
                return b
        else:
            if val > b[0]:
                self.best = (val, t, frame, aux)
            elif val < b[0] - amp:
                self.down, self.best = True, (val, t, frame, aux)
        return None


# ---------------------------------------------------------------------------- engine


class EventEngine:
    """Streaming event engine; see the module docstring."""

    def __init__(self, court: CourtSpec, roster: Roster | None = None, config: EventEngineConfig | None = None,
                 rules: RuleSpec | None = None, cue_models: CueModelSet | None = None) -> None:
        self.court = court
        self.roster = roster
        self.config = cfg = config or EventEngineConfig()
        self.rules = rules if rules is not None else _default_rules(court.league)
        if cue_models is None:
            cue_models = (CueModelSet.load(cfg.cue_model_path, cfg.assume_calibrated) if cfg.cue_model_path
                          else CueModelSet(assume_calibrated=cfg.assume_calibrated))
        self.cue_models = cue_models
        self.scoreboard = ScoreboardConfirmer(ScoreboardConfig(**cfg.scoreboard))
        self.deadball = DeadBallDetector(court, self.rules, DeadBallConfig(**cfg.deadball))
        self.reconciler = Reconciler(cfg)
        self._gate = BallGate(cfg)
        self._holder = HolderTracker(cfg)
        self._hist = PlayerHistory()
        self._store: dict[str, Event] = {}
        self._order: dict[str, int] = {}
        self._out: list[Event] = []
        self._seq = {"e": 0, "m": 0, "r": 0}
        self._shot_seq = 0
        self._phase = LIVE
        self._live_since = -math.inf
        self._poss_team: Team | None = None
        self._flight: Flight | None = None
        self._flight_ctrl: _Control | None = None
        self._pending_miss: ShotEntry | None = None
        self._control: _Control | None = None
        self._prev_control: _Control | None = None
        self._dead: _Dead | None = None
        self._ft: _FtTrip | None = None
        self._shots: dict[str, ShotEntry] = {}
        self._recent: deque[ShotEntry] = deque(maxlen=48)
        self._rims: dict[Side, BBox] = {}
        self._players: dict[str, PlayerState] = {}
        self._buffer: deque[tuple[float, int, BallObs | None]] = deque(maxlen=120)
        self._dt = 1.0 / 30.0
        self._last_t: float | None = None
        self._last_frame: int | None = None
        self._period: int | None = None
        self._period_ended = False
        self._attack_votes: dict[tuple[Team, int], dict[Side, int]] = {}
        self._bounce = _Extremum()
        self._bounce_pid: str | None = None
        self._last_dribble_t: dict[str, float] = {}
        self._live_ball: deque[tuple[float, int]] = deque(maxlen=240)  # in-bounds live ball sightings
        self._holder_oob_n = 0
        self._clock_stop_used: float | None = None
        self._gc_watch = ClockZeroWatch(self.scoreboard.config)  # game clock expiry (PERIOD_END)
        self._sc_watch = ClockZeroWatch(self.scoreboard.config)  # shot clock expiry (violation)
        self._fouls: dict[str, int] = {}
        self.fouled_out: set[str] = set()
        self._ppf: float | None = None
        self.ray_model = RayModel(court.rim_height, {s: court.basket_xy(s) for s in ("left", "right")})

    # ------------------------------------------------------------------ public API

    def process(self, frame: FrameState) -> list[Event]:
        """New events and revisions produced at this frame."""
        self._out = []
        frame = _sanitized(frame)
        self._tick(frame)
        self._hist.update(frame)
        self._players = {p.player_id: p for p in frame.players}
        for rim in frame.rims:
            self._rims[rim.side] = rim.bbox
        self._ppf = _px_per_ft(frame.players, self.config.player_height_ft)
        if frame.scoreboard is not None:
            self._on_scoreboard(frame)
        obs = self._gate.accept(ball_obs(frame, self.config), self._ppf)
        self._buffer.append((frame.t, frame.frame_idx, obs))
        if obs is not None and self.config.use_ray_model:
            self.ray_model.observe(obs, self._rims)  # type: ignore[arg-type]
        self._note_live_ball(obs)
        if self._flight is not None:
            self._step_flight(self._flight, obs, frame.t, frame.frame_idx)
        confirm_n, release_n = self._hysteresis_frames()
        f = self._flight
        prefer = f.release.team if f is not None and f.state == "air" and f.context == "live" else None
        opp_n = max(confirm_n, round(self.config.steal_confirm_s / max(self._dt, 1e-3))) if self._phase == LIVE else None
        changes = self._holder.update(frame, obs, self._gate.velocity(), self._ppf, confirm_n, release_n, prefer, opp_n,
                                      self.ray_model.ray, self._gate.vertical_accel())
        self._note_toss(obs)
        for ch in changes:
            if ch.kind == "release":
                self._on_release(ch)
            else:
                self._on_gain(ch)
        self._detect_dribble(obs)
        self._check_holder_oob(frame)
        self._update_deadball(frame, obs)
        self._reconcile(frame.t)
        self._last_t, self._last_frame = frame.t, frame.frame_idx
        return self._out

    def flush(self) -> list[Event]:
        """End of stream: resolve pending shots and scoreboard changes."""
        self._out = []
        t = self._last_t if self._last_t is not None else 0.0
        f = self._last_frame if self._last_frame is not None else 0
        if self._flight is not None:
            for sig in self._flight.finish(t, f):
                self._on_signal(self._flight, sig)
        self._reconcile(t, force=True)
        return self._out

    def add_manual(self, event: Event) -> list[Event]:
        """Operator insert / correction: source "manual", CONFIRMED; a known event_id is a revision."""
        self._out = []
        ev = dataclasses.replace(event, attrs=dict(event.attrs))
        if ev.source not in ("feed",):
            ev.source = "manual"
        if ev.status is not EventStatus.RETRACTED:
            ev.status = EventStatus.CONFIRMED
        old = self._store.get(ev.event_id) if ev.event_id else None
        if old is not None:
            ev.revision = old.revision + 1
        elif not ev.event_id:
            ev.event_id = self._next_id("m")
        self._manual_side_effects(ev, old)
        self._put(ev)
        return self._out

    @property
    def events(self) -> list[Event]:
        """Latest revision of each event (RETRACTED included), sorted by t."""
        return sorted(self._store.values(), key=lambda e: (e.t, self._order[e.event_id]))

    @property
    def holder_id(self) -> str | None:
        return self._holder.holder

    @property
    def possession_team(self) -> Team | None:
        return self._poss_team

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def attacking_side(self) -> dict[Team, Side]:
        """Basket each team attacks now: shot evidence in this half, else the other half's
        evidence flipped, else the config default (flipped in the second half)."""
        half = self._half_index(self._period)
        out: dict[Team, Side] = {}
        for team in (Team.HOME, Team.AWAY):
            side = self._voted_side(team, half)
            if side is None:
                other = self._voted_side(team, 1 - half)
                side = self.court.other_side(other) if other is not None else None
            if side is None:
                default = self.config.default_attacking.get(team.value, "right" if team is Team.HOME else "left")
                side = default if half == 0 else self.court.other_side(default)  # type: ignore[arg-type]
            out[team] = side  # type: ignore[assignment]
        return out

    def fouls(self, player_id: str) -> int:
        return self._fouls.get(player_id, 0)

    # ------------------------------------------------------------------ event store

    def _next_id(self, prefix: str) -> str:
        while True:  # skip ids an operator / feed event already uses (its own id is kept)
            self._seq[prefix] += 1
            eid = f"{prefix}{self._seq[prefix]:06d}"
            if eid not in self._store:
                return eid

    def _put(self, ev: Event) -> Event:
        if ev.event_id not in self._order:
            self._order[ev.event_id] = len(self._order)
        self._store[ev.event_id] = ev
        self._out.append(ev)
        return ev

    def _emit(self, kind: EventKind, t: float, frame_idx: int, *, team: Team | None = None, player_id: str | None = None,
              other: str | None = None, xy: tuple[float, float] | None = None, value: int = 0, shot_id: str | None = None,
              confidence: float = 1.0, status: EventStatus = EventStatus.PROVISIONAL, source: str = "cv",
              attrs: dict[str, Any] | None = None, prefix: str = "e") -> Event:
        ev = Event(
            event_id=self._next_id(prefix), kind=kind, t=t, frame_idx=frame_idx, team=team, player_id=player_id,
            other_player_id=other, x=xy[0] if xy else None, y=xy[1] if xy else None, value=value, shot_id=shot_id,
            confidence=float(min(max(confidence, 0.0), 1.0)), status=status, source=source, attrs=attrs or {},
        )
        log.debug("%s %s t=%.2f %s", ev.event_id, kind.value, t, player_id)
        return self._put(ev)

    def _revise(self, event_id: str | None, **changes: Any) -> Event | None:
        if event_id is None or event_id not in self._store:
            return None
        old = self._store[event_id]
        attrs = dict(old.attrs)
        attrs.update(changes.pop("attrs", {}))
        new = dataclasses.replace(old, revision=old.revision + 1, attrs=attrs, **changes)
        return self._put(new)

    def _retract(self, event_id: str | None, reason: str) -> None:
        if event_id and event_id in self._store and self._store[event_id].status is not EventStatus.RETRACTED:
            self._revise(event_id, status=EventStatus.RETRACTED, attrs={"retract_reason": reason})

    # ------------------------------------------------------------------ helpers

    def _tick(self, frame: FrameState) -> None:
        if self._last_t is not None:
            dt = frame.t - self._last_t
            if 1e-4 < dt < 0.5:
                self._dt = 0.9 * self._dt + 0.1 * dt

    def _hysteresis_frames(self) -> tuple[int, int]:
        cfg = self.config
        dt = max(self._dt, 1e-3)
        return (max(cfg.holder_min_frames, round(cfg.holder_confirm_s / dt)),
                max(cfg.holder_min_frames, round(cfg.holder_release_s / dt)))

    def _team_of(self, pid: str | None) -> Team | None:
        if pid is None:
            return None
        p = self._players.get(pid)
        if p is not None and p.team is not None:
            return p.team
        team = self._hist.team.get(pid)
        if team is not None:
            return team
        if self.roster is not None:
            team = self.roster.team_of(pid)
            if team is not None:
                return team
        prefix = pid.split("-", 1)[0]
        return Team(prefix) if prefix in ("home", "away") else None

    def _pos(self, pid: str, t: float | None = None) -> tuple[float, float] | None:
        if t is not None:
            pos = self._hist.position_at(pid, t)
            if pos is not None:
                return pos
        p = self._players.get(pid)
        return (p.x, p.y) if p is not None else self._hist.last(pid)

    def _player_conf(self, pid: str) -> float:
        p = self._players.get(pid)
        return p.confidence if p is not None else 0.8

    def _p(self, kind: str, cues: dict[str, Any]) -> tuple[float, str]:
        return self.cue_models.predict(kind, cues)

    def _note_live_ball(self, obs: BallObs | None) -> None:
        if obs is None or obs.xy is None or self._phase != LIVE:
            return
        if obs.z is not None and obs.z > self.config.oob_max_z_ft + 6.0:
            return
        if self.court.in_bounds(obs.xy[0], obs.xy[1], margin=0.5):
            self._live_ball.append((obs.t, obs.frame_idx))

    def _last_live_ball(self, before: float) -> tuple[float, int] | None:
        for t, f in reversed(self._live_ball):
            if t < before:
                return t, f
        return None

    def _note_toss(self, obs: BallObs | None) -> None:
        d = self._dead
        if d is None or d.cause != "period" or d.toss_seen or obs is None or self._holder.holder is not None:
            return
        vel = self._gate.velocity()
        if vel is None:
            return
        speed = math.sqrt(sum(v * v for v in vel))
        if not obs.is3d and obs.uv is not None:
            speed /= self._ppf or self.config.nominal_px_per_ft
        d.toss_seen = speed >= self.config.toss_min_speed_fts

    def _sample(self, obs: BallObs | None) -> FlightSample | None:
        if obs is None:
            return None
        return FlightSample(obs.t, obs.frame_idx, obs.uv, obs.xy, obs.z, obs.r_px)

    # ------------------------------------------------------------------ scoreboard and periods

    def _on_scoreboard(self, frame: FrameState) -> None:
        sb = frame.scoreboard
        assert sb is not None
        for u in self.scoreboard.update(sb):
            if isinstance(u, PeriodChange):
                self._on_period_change(u, frame)
            elif isinstance(u, ScoreChange):
                self.reconciler.add(u)
        sbc = self.scoreboard.config
        trusted = sb.source in sbc.immediate_sources
        if not trusted and sb.confidence < sbc.min_ocr_conf:
            return
        # Clock expiry is confirmed like a score change (D14): a single misread "0:00", or a period
        # number that advances before the clock is reset, must not end the period.
        t0 = self._gc_watch.update(frame.t, sb.game_clock, trusted)
        if t0 is not None and self._period is not None and not self._period_ended:
            self._end_period(t0, self._frame_at(t0, frame), "game clock reached 0")
        self._check_shot_clock(frame, trusted)

    def _frame_at(self, t: float, frame: FrameState) -> int:
        return frame.frame_idx if t >= frame.t else self._frame_near(t)

    def _on_period_change(self, u: PeriodChange, frame: FrameState) -> None:
        sb = frame.scoreboard
        gc = sb.game_clock if sb is not None else None
        # u.t is the first read that showed the new period (OCR confirms one read later)
        start_t, start_f = min(u.t, frame.t), self._frame_at(u.t, frame)
        if u.old is not None and not self._period_ended and self._period is not None:
            prev = self._frame_before(start_t)
            t, f = prev if prev is not None else (start_t, start_f)
            self._end_period(t, f, "scoreboard period changed")
        self._period = u.new
        self._period_ended = False
        self._gc_watch.reset()  # the new period's clock must be seen running before it can expire
        self._sc_watch.reset()
        full = self.rules.period_length_s(u.new) if u.new >= 1 else None
        if u.old is not None or gc is None or (full is not None and gc >= full - self.config.period_start_tol_s):
            self._emit(EventKind.PERIOD_START, start_t, start_f, value=u.new, source="scoreboard",
                       status=EventStatus.CONFIRMED, attrs={"period": u.new, "reason": "scoreboard period"})
            self._start_period(start_t, start_f)

    def _end_period(self, t: float, f: int, reason: str) -> None:
        if self._period is None:
            return
        self._period_ended = True
        self._emit(EventKind.PERIOD_END, t, f, value=self._period, source="scoreboard", status=EventStatus.CONFIRMED,
                   attrs={"period": self._period, "reason": reason})
        self.deadball.note_stoppage(t, "period_break")
        self._pending_miss = None  # no rebound after the period-end buzzer [15]
        self._ft = None
        self._enter_dead("break", t, f)  # a ball already on its way may still become an attempt / make

    def _start_period(self, t: float, f: int) -> None:
        self._poss_team = None
        self._flight = None
        self._pending_miss = None
        self._control = self._prev_control = None
        self._ft = None
        self._dead = None
        self._enter_dead("period", t, f)

    def _half_index(self, period: int | None) -> int:
        if period is None:
            return 0
        if self.rules.halves:
            return 0 if period <= 1 else 1
        return 0 if period <= self.rules.n_periods // 2 else 1

    def _voted_side(self, team: Team, half: int) -> Side | None:
        votes = dict(self._attack_votes.get((team, half), {}))
        for side, n in self._attack_votes.get((team.other(), half), {}).items():
            other = self.court.other_side(side)
            votes[other] = votes.get(other, 0) + n
        if not votes:
            return None
        best = max(votes.values())
        winners = [s for s, n in votes.items() if n == best]
        return winners[0] if len(winners) == 1 else None

    # ------------------------------------------------------------------ phases

    def _enter_dead(self, cause: str, t: float, f: int, **kw: Any) -> _Dead:
        self._phase = DEAD
        self._dead = _Dead(cause=cause, since_t=t, since_frame=f, **kw)
        return self._dead

    def _go_live(self, t: float) -> None:
        self._phase = LIVE
        self._dead = None
        self._live_since = t
        self.deadball.note_live(t)

    def _restart(self, team: Team | None, t: float, f: int, pid: str, how: str) -> None:
        """End a dead ball: ``team`` puts the ball in play (inbound, jump ball, live evidence)."""
        d = self._dead
        if d is None or d.cause == "break":
            return
        first = d.first_control.get(team) if team is not None else None
        ctrl = self._control
        if how == "inbound" and ctrl is not None and ctrl.player_id == pid:
            first = (ctrl.onset_t, ctrl.onset_frame, pid)  # the inbounder's own control, not a stray touch
        ft, ff, fp = first if first is not None else (t, f, pid)
        if d.cause == "period":
            cause = "jump_ball" if how == "jump" else "period_start"
            self._set_possession(team, ft, ff, fp, cause, force=True)
        elif d.cause in ("made_fg", "made_ft"):
            if team is not None and d.scoring_team is not None and team is not d.scoring_team:
                self._set_possession(team, ft, ff, fp, "made_shot" if d.cause == "made_fg" else "free_throw")
        elif d.miss is not None:
            self._team_rebound(d, team, ft, ff, fp)
        elif d.cause in ("oob", "unknown"):
            self._inferred_turnover(d, team, ft, ff, fp)
        elif d.cause in ("shot_clock", "turnover"):
            self._set_possession(team, ft, ff, fp, "turnover")
        elif d.cause == "ft_trip":
            self._set_possession(team, ft, ff, fp, "free_throw")
        if self._ft is not None and self._ft.taken > 0:  # a trip that has not started (manual FOUL) survives
            self._ft = None
        self._go_live(t)

    def _team_rebound(self, d: _Dead, team: Team | None, ft: float, ff: int, fp: str,
                      reason: str = "ball dead before control; credited to the inbounding team") -> None:
        miss = d.miss
        assert miss is not None
        if team is None:
            return
        offensive = team is miss.team
        t = d.out_t if d.out_t is not None else d.since_t
        f = d.out_frame if d.out_frame is not None else d.since_frame
        p, status = self._p("rebound", {"control_duration_s": 0.0, "track_conf": 1.0, "team_margin": 1.0})
        ev = self._emit(EventKind.REBOUND, t, f, team=team, value=0, shot_id=miss.shot_id,
                        confidence=min(p, self.config.team_rebound_conf),
                        attrs={"offensive": offensive, "team_rebound": True, "p_status": status, "needs_review": True,
                               "reason": reason})
        miss.rebound_id = ev.event_id
        self._pending_miss = None
        if not offensive:
            pev = self._set_possession(team, ft, ff, fp, "defensive_rebound")
            if pev is not None:
                miss.possession_id = pev.event_id

    def _inferred_turnover(self, d: _Dead, team: Team | None, ft: float, ff: int, fp: str) -> None:
        last = d.last_team
        if team is None or last is None or team is last:
            if team is not None and self._poss_team is None:
                self._set_possession(team, ft, ff, fp, "inbound")
            return
        t = d.out_t if d.out_t is not None else d.since_t
        f = d.out_frame if d.out_frame is not None else d.since_frame
        kind = "out_of_bounds" if d.cause == "oob" or d.out_t is not None else "other"
        self._emit(EventKind.TURNOVER, t, f, team=last, player_id=d.last_holder, confidence=self.config.oob_turnover_conf,
                   attrs={"type": kind, "needs_review": True,
                          "reason": "inferred from the next inbound by the other team (last touch not seen)"})
        self._set_possession(team, ft, ff, fp, "turnover")

    def _set_possession(self, team: Team | None, t: float, f: int, pid: str | None, cause: str,
                        conf: float | None = None, force: bool = False) -> Event | None:
        if team is None or (team is self._poss_team and not force):
            return None
        self._poss_team = team
        return self._emit(EventKind.POSSESSION, t, f, team=team, player_id=pid, xy=self._pos(pid, t) if pid else None,
                          confidence=self.config.possession_conf if conf is None else conf,
                          attrs={"cause": cause, "reason": f"{team.value} gains control ({cause})"})

    # ------------------------------------------------------------------ releases and gains

    def _on_release(self, ch: ControlChange) -> None:
        cfg = self.config
        pid, team = ch.player_id, self._team_of(ch.player_id)
        ctrl = self._control if self._control is not None and self._control.player_id == pid else None
        if self._dead is not None and self._dead.cause == "break":
            self._control = None
            return
        pos = self._pos(pid, ch.t)
        gather = self._hist.position_at(pid, ch.t - cfg.gather_lead_s, not_before=ctrl.onset_t if ctrl else None) or pos
        p = self._players.get(pid)
        bbox = p.bbox if p is not None else self._hist.bbox.get(pid)
        rel = Release(pid, team, ch.t, ch.frame_idx, pos, gather, bbox, ctrl.onset_t if ctrl else None)
        ft_side = self._ft_side(pid, ch.t)
        if self._flight is not None and not self._flight.resolved and self._flight.is_shot:
            for sig in self._flight.finish(ch.t, ch.frame_idx):
                self._on_signal(self._flight, sig)
        if ft_side is not None:
            flight = Flight(rel, self.court, cfg, free_throw_side=ft_side)
        else:
            if self._phase == DEAD and pos is not None and not self.court.in_bounds(*pos, margin=cfg.inbound_holder_margin_ft):
                self._restart(team, ch.t, ch.frame_idx, pid, "inbound")
            flight = Flight(rel, self.court, cfg, context="live" if self._phase == LIVE else "dead")
        self._flight, self._flight_ctrl = flight, ctrl
        self._prev_control, self._control = ctrl, None
        if ft_side is not None:
            self._start_free_throw(flight)
        for t, fi, o in list(self._buffer):
            if fi >= ch.frame_idx and self._flight is flight:
                self._step_flight(flight, o, t, fi)

    def _on_gain(self, ch: ControlChange) -> None:
        pid, team = ch.player_id, self._team_of(ch.player_id)
        pos = self._pos(pid, ch.t)
        flight = self._flight
        if self._is_id_swap(pid, team, pos):
            prev = self._prev_control
            assert prev is not None
            self._control = dataclasses.replace(prev, player_id=pid)
            self._flight = None
            log.info("holder id %s -> %s treated as an identity swap", prev.player_id, pid)
            return
        if flight is not None:
            for sig in flight.on_control(ch.t, ch.frame_idx):
                self._on_signal(flight, sig)
        oob = pos is not None and not self.court.in_bounds(*pos, margin=self.config.inbound_holder_margin_ft)
        if self._phase == LIVE and oob:
            self._enter_dead_from_live("oob", ch.t, ch.frame_idx, flight, before=ch.t)
        if self._phase == DEAD:
            self._gain_dead(pid, team, ch, oob)
        else:
            self._gain_live(pid, team, ch, flight)

    def _enter_dead_from_live(self, cause: str, t: float, f: int, flight: Flight | None, out: bool = True,
                              before: float | None = None) -> None:
        last_holder = flight.release.player_id if flight is not None else (self._holder.holder or None)
        last_team = flight.release.team if flight is not None else self._poss_team
        if flight is None and self._control is not None:
            last_holder, last_team = self._control.player_id, self._control.team
        seen = self._last_live_ball(t if before is None else before) if out else None
        since = seen if seen is not None else (t, f)
        d = self._enter_dead(cause, since[0], since[1], last_team=last_team, last_holder=last_holder,
                             miss=self._pending_miss)
        if out:
            d.out_t, d.out_frame = since
            self.deadball.note_stoppage(since[0], "out_of_bounds")
        self._pending_miss = None
        self._flight = None

    def _gain_dead(self, pid: str, team: Team | None, ch: ControlChange, oob: bool) -> None:
        d = self._dead
        assert d is not None
        flight, self._flight = self._flight, None
        self._control = _Control(pid, team, ch.t, ch.frame_idx, origin="inbound" if oob else "other")
        if d.cause == "break":
            return
        if team is not None and team not in d.first_control:
            d.first_control[team] = (ch.t, ch.frame_idx, pid)
        if d.first_pickup is None:
            d.first_pickup = (ch.t, ch.frame_idx, pid)
        if d.cause == "period" and not oob:
            if d.toss_seen or flight is not None:  # control after the toss, not a ball resting by the jumpers
                self._restart(team, ch.t, ch.frame_idx, pid, "jump")
            return
        if self._in_bounds_pass_restart(d, flight, pid, team, ch, oob):
            r = flight.release  # type: ignore[union-attr]
            self._restart(team, r.t, r.frame_idx, r.player_id, "live")
            ev = self._pass(flight, pid, ch, completed=True)  # type: ignore[arg-type]
            self._control = _Control(pid, team, ch.t, ch.frame_idx, origin="pass",
                                     via_pass=_PassRec(r.player_id, r.team, r.t, ev.event_id))

    def _in_bounds_pass_restart(self, d: _Dead, flight: Flight | None, pid: str, team: Team | None,
                                ch: ControlChange, oob: bool) -> bool:
        """Fallback restart: an in-bounds pass between teammates well into a dead ball."""
        if flight is None or oob or team is None or d.cause in ("break", "period", "ft_trip"):
            return False
        r = flight.release
        if r.team is not team or r.player_id == pid or r.xy is None or not self.court.in_bounds(*r.xy):
            return False
        return r.t - d.since_t >= self.config.dead_min_s

    def _gain_live(self, pid: str, team: Team | None, ch: ControlChange, flight: Flight | None) -> None:
        cfg = self.config
        self._flight = None
        miss = self._pending_miss
        if miss is not None and (flight is None or flight.is_shot or flight.state == "loose"):
            if miss.is_ft and miss.extra.get("ft_total_unknown") and self._ft_retrieval(miss):
                self._enter_dead("ft_trip", ch.t, ch.frame_idx)
                self._pending_miss = None
                self._control = _Control(pid, team, ch.t, ch.frame_idx)
                return
            self._rebound(pid, team, ch, miss)
            if miss.is_ft:
                self._ft = None
            self._control = _Control(pid, team, ch.t, ch.frame_idx, origin="rebound")
            return
        if flight is None or flight.is_shot or flight.state == "loose":
            self._set_possession(team, ch.t, ch.frame_idx, pid, "other")
            self._control = _Control(pid, team, ch.t, ch.frame_idx)
            return
        r = flight.release
        if pid == r.player_id:
            prev = self._prev_control
            if prev is not None and prev.player_id == pid and ch.t - r.t <= cfg.self_regain_s:
                self._control = prev
            else:
                self._control = _Control(pid, team, ch.t, ch.frame_idx, origin="self")
            return
        if ch.t - r.t > cfg.pass_max_flight_s:
            # Long loose ball: a recovery, not a pass. No steal is credited without a defender's
            # action on the ball [15]; the possession change is still recorded.
            self._set_possession(team, ch.t, ch.frame_idx, pid, "other")
            self._control = _Control(pid, team, ch.t, ch.frame_idx)
            return
        if team is not None and team is r.team:
            ev = self._pass(flight, pid, ch, completed=True)
            self._control = _Control(pid, team, ch.t, ch.frame_idx, origin="pass",
                                     via_pass=_PassRec(r.player_id, r.team, r.t, ev.event_id))
            self._set_possession(team, ch.t, ch.frame_idx, pid, "other")
            return
        if team is not None and r.team is not None:
            travel = math.dist(r.xy, self._pos(pid, ch.t) or r.xy) if r.xy is not None else 0.0
            strip = travel <= cfg.strip_max_travel_ft and ch.t - r.t <= cfg.strip_max_flight_s
            self._steal(pid, team, ch, flight, "steal" if strip else "bad_pass")
            return
        self._control = _Control(pid, team, ch.t, ch.frame_idx)

    def _is_id_swap(self, pid: str, team: Team | None, pos: tuple[float, float] | None) -> bool:
        prev = self._prev_control
        flight = self._flight
        if prev is None or flight is None or flight.is_shot or prev.player_id == pid or pos is None:
            return False
        if prev.player_id in self._players or team is None or team is not prev.team:
            return False
        last = self._hist.last(prev.player_id)
        return last is not None and math.dist(last, pos) <= self.config.id_swap_radius_ft

    # ------------------------------------------------------------------ event builders

    def _pass(self, flight: Flight, receiver: str | None, ch: ControlChange, completed: bool) -> Event:
        r = flight.release
        rpos = self._pos(receiver, ch.t) if receiver else None
        dist = math.dist(r.xy, rpos) if r.xy is not None and rpos is not None else None
        return self._emit(EventKind.PASS, r.t, r.frame_idx, team=r.team, player_id=r.player_id, other=receiver,
                          xy=r.xy, confidence=self.config.pass_conf,
                          attrs={"completed": completed, "distance_ft": dist,
                                 "reason": "teammate gained control" if completed else "intercepted"})

    def _steal(self, pid: str, team: Team, ch: ControlChange, flight: Flight, kind: str) -> None:
        r = flight.release
        if kind == "bad_pass":
            self._pass(flight, self._intended_receiver(r, ch.t, pid), ch, completed=False)
        cues = self._control_cues(pid, ch)
        p, status = self._p("steal", cues)
        pos = self._pos(pid, ch.t)
        self._emit(EventKind.STEAL, ch.t, ch.frame_idx, team=team, player_id=pid, other=r.player_id, xy=pos,
                   confidence=p, attrs={"p_status": status, "cues": cues, "reason": f"opponent gained control ({kind})"})
        self._emit(EventKind.TURNOVER, ch.t, ch.frame_idx, team=r.team, player_id=r.player_id, xy=pos, confidence=p,
                   attrs={"type": kind, "p_status": status, "reason": "lost the ball to the stealer"})
        self._set_possession(team, ch.t, ch.frame_idx, pid, "steal", conf=p)
        self._control = _Control(pid, team, ch.t, ch.frame_idx, origin="steal")

    def _intended_receiver(self, r: Release, t: float, stealer: str) -> str | None:
        spos = self._pos(stealer, t)
        if spos is None:
            return None
        best, best_d = None, math.inf
        for p in self._players.values():
            if p.player_id == r.player_id or self._team_of(p.player_id) is not r.team:
                continue
            d = math.hypot(p.x - spos[0], p.y - spos[1])
            if d < best_d:
                best, best_d = p.player_id, d
        return best

    def _control_cues(self, pid: str, ch: ControlChange) -> dict[str, Any]:
        cur = self._buffer[-1][2] if self._buffer else None
        pos = self._pos(pid)
        dist = math.dist(cur.xy, pos) if cur is not None and cur.xy is not None and pos is not None and cur.is3d else None
        return {"control_duration_s": max(0.0, (self._buffer[-1][0] if self._buffer else ch.t) - ch.t),
                "holder_dist_ft": dist, "track_conf": self._player_conf(pid), "team_margin": 1.0}

    def _rebound(self, pid: str, team: Team | None, ch: ControlChange, miss: ShotEntry) -> None:
        offensive = team is not None and team is miss.team
        cues = self._control_cues(pid, ch)
        p, status = self._p("rebound", cues)
        ev = self._emit(EventKind.REBOUND, ch.t, ch.frame_idx, team=team, player_id=pid, xy=self._pos(pid, ch.t),
                        shot_id=miss.shot_id, confidence=p,
                        attrs={"offensive": offensive, "p_status": status, "cues": cues,
                               "reason": "first control after a missed shot"})
        miss.rebound_id = ev.event_id
        self._pending_miss = None
        if not offensive:
            pev = self._set_possession(team, ch.t, ch.frame_idx, pid, "defensive_rebound", conf=p)
            if pev is not None:
                miss.possession_id = pev.event_id

    # ------------------------------------------------------------------ flights

    def _step_flight(self, flight: Flight, obs: BallObs | None, t: float, frame_idx: int) -> None:
        sigs = flight.step(self._sample(obs), t, frame_idx, self._rims, list(self._players.values()), self._team_of)
        for sig in sigs:
            self._on_signal(flight, sig)

    def _on_signal(self, flight: Flight, sig: FlightSignal) -> None:
        if sig.kind == "shot":
            if not flight.is_ft:
                self._emit_attempt(flight)
        elif sig.kind == "blocked":
            self._on_block(flight, sig)
        elif sig.kind in ("made", "missed"):
            self._on_result(flight, sig)
        elif sig.kind == "out":
            self._on_out(flight, sig)

    def _emit_attempt(self, flight: Flight) -> ShotEntry:
        cfg, court = self.config, self.court
        r, side = flight.release, flight.target
        assert side is not None
        self._shot_seq += 1
        shot_id = f"s{self._shot_seq:04d}"
        gather = r.gather_xy or r.xy
        ctrl = self._flight_ctrl
        cues = flight.shot_cues()
        p_shot, status = self._p("shot", cues)
        attrs: dict[str, Any] = {"basket": side, "p_status": status, "cues": cues, "release_t": r.t}
        if flight.is_ft:
            value, distance = 1, court.distance_to_basket(*court.free_throw_xy(side), side)
            attrs.update(shot_type="free_throw", distance_ft=distance)
        else:
            gx, gy = gather if gather is not None else court.basket_xy(side)
            margin = three_margin_ft(court, gx, gy, side)
            along = (gx - court.basket_xy(side)[0]) * (1.0 if side == "left" else -1.0)
            three_cues = {"foot_margin_ft": margin, "foot_obs_quality": self._player_conf(r.player_id),
                          "in_corner": along <= court.three_junction_along}
            p3, _ = self._p("three", three_cues)
            value = 3 if court.is_three(gx, gy, side) else 2
            distance = court.distance_to_basket(gx, gy, side)
            control_s = r.t - ctrl.onset_t if ctrl is not None else None
            first = flight.samples[0] if flight.samples else None
            dribbles = ctrl.dribbles if ctrl is not None else 0
            attrs.update(
                shot_type=shot_type_for(distance, first.z if first is not None else None, control_s, False, cfg),
                distance_ft=distance, zone=court.zone(gx, gy, side), gather_x=gx, gather_y=gy,
                closest_defender_ft=self._closest_defender(r), dribbles_before=dribbles,
                catch_and_shoot=bool(ctrl is not None and ctrl.origin == "pass" and dribbles == 0
                                     and distance >= cfg.catch_shoot_min_dist_ft),
                p_three=p3, three_cues=three_cues,
            )
            self._vote_attack(r.team, side)
        if self._ft is not None and flight.is_ft:
            attrs.update(ft_index=self._ft.taken + 1, ft_total=self._ft.total, ft_type=self._ft.ft_type,
                         technical=self._ft.ft_type == "technical")
        ev = self._emit(EventKind.SHOT_ATTEMPT, r.t, r.frame_idx, team=r.team, player_id=r.player_id, xy=r.xy,
                        value=value, shot_id=shot_id, confidence=p_shot, attrs=attrs)
        entry = ShotEntry(shot_id, r.team, r.player_id, value, ev.event_id, r.t, is_ft=flight.is_ft,
                          assist_candidate=self._assist_candidate(ctrl, r))
        flight.attempt_id, flight.shot_id = ev.event_id, shot_id
        self._shots[shot_id] = entry
        self._recent.append(entry)
        return entry

    def _closest_defender(self, r: Release) -> float | None:
        if r.xy is None or r.team is None:
            return None
        best = None
        for p in self._players.values():
            if self._team_of(p.player_id) is r.team.other():
                pos = self._hist.position_at(p.player_id, r.t) or (p.x, p.y)
                d = math.dist(pos, r.xy)
                best = d if best is None or d < best else best
        return best

    def _assist_candidate(self, ctrl: _Control | None, r: Release) -> tuple[str, Team | None] | None:
        cfg = self.config
        if ctrl is None or ctrl.via_pass is None or ctrl.via_pass.team is not r.team or r.team is None:
            return None
        if r.t - ctrl.onset_t > cfg.assist_window_s or ctrl.dribbles > cfg.assist_max_dribbles:
            return None
        return ctrl.via_pass.passer, ctrl.via_pass.team

    def _vote_attack(self, team: Team | None, side: Side) -> None:
        if team is None:
            return
        votes = self._attack_votes.setdefault((team, self._half_index(self._period)), {})
        votes[side] = votes.get(side, 0) + 1

    def _entry(self, flight: Flight) -> ShotEntry | None:
        return self._shots.get(flight.shot_id) if flight.shot_id else None

    def _on_block(self, flight: Flight, sig: FlightSignal) -> None:
        entry = self._emit_attempt(flight)
        mc = flight.make_cues()
        p_make, mstatus = self._p("make", mc)
        miss = self._emit(EventKind.SHOT_MISSED, sig.t, sig.frame_idx, team=entry.team, player_id=entry.shooter_id,
                          value=entry.value, shot_id=entry.shot_id, confidence=1.0 - p_make,
                          attrs={"blocked": True, "points": 0, "p_status": mstatus, "reason": sig.reason})
        bcues = dict(flight.block_data)
        bcues["track_conf"] = self._player_conf(sig.player_id or "")
        p, status = self._p("block", bcues)
        blocker = sig.player_id
        bev = self._emit(EventKind.BLOCK, sig.t, sig.frame_idx, team=self._team_of(blocker), player_id=blocker,
                         other=entry.shooter_id, xy=self._pos(blocker, sig.t) if blocker else None,
                         shot_id=entry.shot_id, confidence=p, attrs={"p_status": status, "cues": bcues, "reason": sig.reason})
        entry.made, entry.result_t, entry.result_id, entry.block_id = False, sig.t, miss.event_id, bev.event_id
        self._pending_miss = entry

    def _on_result(self, flight: Flight, sig: FlightSignal) -> None:
        entry = self._entry(flight)
        if entry is None:
            if flight.is_ft or flight.context != "live":
                return
            entry = self._emit_attempt(flight)
        made = sig.kind == "made"
        mc = flight.make_cues()
        p_make, status = self._p("make", mc)
        attrs: dict[str, Any] = {"p_status": status, "cues": mc, "reason": sig.reason}
        if sig.needs_review:
            attrs["needs_review"] = True
        if made:
            if self._period is not None and self.scoreboard.game_clock is not None and not flight.is_ft:
                attrs["clock_stops"] = self.rules.clock_stops_after_fg_at(self._period, self.scoreboard.game_clock)
            ev = self._emit(EventKind.SHOT_MADE, sig.t, sig.frame_idx, team=entry.team, player_id=entry.shooter_id,
                            value=entry.value, shot_id=entry.shot_id, confidence=p_make, attrs=attrs)
        else:
            attrs["points"] = 0
            ev = self._emit(EventKind.SHOT_MISSED, sig.t, sig.frame_idx, team=entry.team, player_id=entry.shooter_id,
                            value=entry.value, shot_id=entry.shot_id, confidence=1.0 - p_make, attrs=attrs)
        entry.made, entry.result_t, entry.result_id = made, sig.t, ev.event_id
        if flight.is_ft:
            self._after_free_throw(entry, made, sig)
            return
        after_buzzer = self._dead is not None and self._dead.cause == "break"
        if made:
            self._emit_assist(entry, sig.t, sig.frame_idx)
            self._flight = None if self._flight is flight else self._flight
            if not after_buzzer:
                self._enter_dead("made_fg", sig.t, sig.frame_idx, scoring_team=entry.team, shot_id=entry.shot_id)
                self.deadball.note_stoppage(sig.t, "made_basket")
        elif not after_buzzer:
            self._pending_miss = entry

    def _emit_assist(self, entry: ShotEntry, t: float, f: int) -> None:
        cand = entry.assist_candidate
        if cand is None or (entry.is_ft and not self.config.assist_on_ft_trips):
            return
        passer, team = cand
        ev = self._emit(EventKind.ASSIST, t, f, team=team, player_id=passer, other=entry.shooter_id,
                        shot_id=entry.shot_id, confidence=self.config.pass_conf,
                        attrs={"reason": "last pass to the scorer within the assist window"})
        entry.assist_id = ev.event_id

    def _on_out(self, flight: Flight, sig: FlightSignal) -> None:
        if self._phase != LIVE or self._flight is not flight:
            return
        d = self._enter_dead("oob", sig.t, sig.frame_idx, last_team=flight.release.team,
                             last_holder=flight.release.player_id, miss=self._pending_miss)
        d.out_t, d.out_frame = sig.t, sig.frame_idx
        self.deadball.note_stoppage(sig.t, "out_of_bounds")
        self._pending_miss = None
        self._flight = None

    # ------------------------------------------------------------------ free throws

    def _ft_side(self, pid: str, t: float) -> Side | None:
        f = self.deadball.formation
        if f is None or f.shooter_id != pid or t - f.since_t < self.config.ft_min_hold_s:
            return None
        if self._phase == DEAD or self._ft is not None:
            return f.side
        # Live play with no foul or stoppage seen: an elbow jumper with players set around the lane
        # looks like a formation. A free throw then needs the full rule-2 hold (§7A.5) and a shooter
        # who did not just catch a live pass from a teammate (FT shooters get the ball from the referee).
        ctrl = self._control if self._control is not None and self._control.player_id == pid else None
        if ctrl is not None and ctrl.origin == "pass":
            return None
        if t - f.since_t < self.deadball.config.ft_formation_hold_s:
            return None
        return f.side

    def _start_free_throw(self, flight: Flight) -> None:
        r = flight.release
        trip = self._ft
        if trip is None or (trip.shooter is not None and trip.shooter != r.player_id and trip.ft_type == "shooting"):
            trip = self._ft = _FtTrip(r.player_id, r.team, None)
        if trip.team is None:
            trip.team = r.team
        if self._phase != DEAD or self._dead is None:
            self._enter_dead("ft_trip", r.t, r.frame_idx, last_team=r.team, last_holder=r.player_id)
        else:
            d = self._dead
            if d.miss is not None:  # a foul after a miss, before control: team rebound to the fouled team [15]
                self._team_rebound(d, trip.team, d.since_t, d.since_frame, r.player_id,
                                   reason="foul before control; credited to the team entitled to the ball")
                d.miss = None
            d.cause = "ft_trip"
        self._pending_miss = None
        self._emit_attempt(flight)

    def _after_free_throw(self, entry: ShotEntry, made: bool, sig: FlightSignal) -> None:
        trip = self._ft
        self._flight = None
        if trip is None:
            trip = self._ft = _FtTrip(entry.shooter_id, entry.team, None)
        trip.taken += 1
        last = trip.total is not None and trip.taken >= trip.total
        unknown = trip.total is None
        if made:
            if self.config.assist_on_ft_trips and trip.assist_candidate and not trip.assisted:
                entry.assist_candidate = trip.assist_candidate
                self._emit_assist(entry, sig.t, sig.frame_idx)
                trip.assisted = True
            self._enter_dead("made_ft" if trip.live_last else "ft_trip", sig.t, sig.frame_idx,
                             scoring_team=entry.team, shot_id=entry.shot_id)
            if last and trip.live_last:
                self._ft = None
            return
        if (last or unknown) and trip.live_last:
            self._go_live(sig.t)
            self._pending_miss = entry
            entry.extra["ft_total_unknown"] = unknown
            if last:
                self._ft = None
            return
        self._enter_dead("ft_trip", sig.t, sig.frame_idx)

    def _ft_retrieval(self, miss: ShotEntry) -> bool:
        """After a missed FT of unknown trip size: lane players still set = dead-ball retrieval."""
        cfg = self.deadball.config
        relaxed = dataclasses.replace(cfg, ft_lane_min_players=max(1, cfg.ft_lane_min_players - 1),
                                      ft_spot_radius_ft=cfg.ft_spot_radius_ft * 3, ft_shooter_max_speed_fts=1e9)
        return ft_formation(self.court, list(self._players.values()), relaxed) is not None

    # ------------------------------------------------------------------ dribbles, out of bounds, dead balls

    def _detect_dribble(self, obs: BallObs | None) -> None:
        cfg = self.config
        pid = self._holder.holder
        if pid is None and self._flight is not None and self._flight.state == "air":
            pid = self._flight.release.player_id
        if pid != self._bounce_pid:
            self._bounce.reset()
            self._bounce_pid = pid
        if pid is None or obs is None:
            return
        p = self._players.get(pid)
        if obs.is3d:
            if p is None:
                return
            aux = math.hypot(obs.xy[0] - p.x, obs.xy[1] - p.y)  # type: ignore[index]
            hit = self._bounce.push(obs.z, obs.t, obs.frame_idx, cfg.dribble_min_amp_ft, aux)  # type: ignore[arg-type]
            ok = hit is not None and hit[0] <= cfg.dribble_floor_z_ft and hit[3] <= cfg.dribble_radius_ft
        elif obs.uv is not None and p is not None and p.bbox is not None:
            h = max(p.bbox.h, 1.0)
            box = keep_box(p.bbox, cfg)
            aux = abs(obs.uv[1] - p.bbox.y2) / h if box.x1 <= obs.uv[0] <= box.x2 else math.inf
            hit = self._bounce.push(-obs.uv[1], obs.t, obs.frame_idx, cfg.dribble_min_amp_h * h, aux)
            ok = hit is not None and hit[3] <= cfg.dribble_foot_band
        else:
            return
        if not ok or hit is None:
            return
        _, t, f, _ = hit
        if t - self._last_dribble_t.get(pid, -math.inf) < cfg.dribble_min_interval_s:
            return
        self._last_dribble_t[pid] = t
        self._on_dribble(pid, t, f)

    def _on_dribble(self, pid: str, t: float, f: int) -> None:
        ctrl = self._control if self._control is not None and self._control.player_id == pid else None
        if ctrl is None and self._prev_control is not None and self._prev_control.player_id == pid:
            ctrl = self._prev_control
        if self._phase == DEAD:
            d = self._dead
            pos = self._pos(pid, t)
            # Only the confirmed holder's dribble restarts play: a dead ball bouncing loose next to
            # whoever last touched it (e.g. under the basket after a make) is not live evidence.
            if d is not None and d.cause not in ("break",) and t - d.since_t >= self.config.dead_min_s \
                    and self._holder.holder == pid \
                    and pos is not None and self.court.in_bounds(*pos) and (self._ft is None or self._ft.shooter != pid):
                self._restart(self._team_of(pid), t, f, pid, "live")
            else:
                return
        if ctrl is not None:
            ctrl.dribbles += 1
        self._emit(EventKind.DRIBBLE, t, f, team=self._team_of(pid), player_id=pid, xy=self._pos(pid, t),
                   confidence=self.config.dribble_conf, attrs={"reason": "ball bounced next to the holder"})

    def _check_holder_oob(self, frame: FrameState) -> None:
        pid = self._holder.holder
        if self._phase != LIVE or pid is None:
            self._holder_oob_n = 0
            return
        p = self._players.get(pid)
        if p is None or self.court.in_bounds(p.x, p.y, margin=self.config.oob_margin_ft):
            self._holder_oob_n = 0
            return
        self._holder_oob_n += 1
        if self._holder_oob_n >= self.config.holder_oob_frames:
            self._holder_oob_n = 0
            d = self._enter_dead("oob", frame.t, frame.frame_idx, last_team=self._team_of(pid), last_holder=pid,
                                 miss=self._pending_miss)
            d.out_t, d.out_frame = frame.t, frame.frame_idx
            self._pending_miss = None
            self.deadball.note_stoppage(frame.t, "out_of_bounds")

    def _update_deadball(self, frame: FrameState, obs: BallObs | None) -> None:
        cfg = self.config
        ball_out = bool(obs is not None and obs.is3d and obs.z <= cfg.oob_max_z_ft  # type: ignore[operator]
                        and not self.court.in_bounds(*obs.xy, margin=cfg.oob_margin_ft))  # type: ignore[misc]
        self.deadball.update(frame, self._holder.holder, ball_out=ball_out, ball_visible=obs is not None)
        if not cfg.dead_from_clock_stop or self._phase != LIVE or frame.scoreboard is None:
            return
        since = self.scoreboard.clock_stopped_since
        stopped = self.scoreboard.clock_stopped(frame.t, 2, cfg.clock_stop_min_s)
        if not stopped or since is None or since == self._clock_stop_used or since < self._live_since:
            return
        if (self.scoreboard.game_clock or 0.0) <= 0.05:
            return
        if self._flight is not None and self._flight.state in ("air", "shot"):
            return
        self._clock_stop_used = since
        f = self._last_frame if self._last_frame is not None else frame.frame_idx
        self._enter_dead_from_live("unknown", since, f, self._flight, out=False)

    def _check_shot_clock(self, frame: FrameState, trusted: bool) -> None:
        sb = frame.scoreboard
        cfg = self.config
        assert sb is not None
        t0 = self._sc_watch.update(frame.t, sb.shot_clock, trusted)
        if t0 is None or not cfg.detect_shot_clock_violations or self.rules.shot_clock_s is None:
            return
        if self._phase != LIVE or self._poss_team is None:
            return
        gc = sb.game_clock if sb.game_clock is not None else self.scoreboard.game_clock
        if (gc or 0.0) <= 0.05 or (self._flight is not None and self._flight.state in ("air", "shot")):
            return
        team = self._poss_team
        holder = self._holder.holder or (self._control.player_id if self._control is not None else None)
        if holder is not None and self._team_of(holder) is not team:
            holder = None
        f = self._frame_at(t0, frame)
        self._emit(EventKind.TURNOVER, t0, f, team=team, player_id=holder, confidence=0.8,
                   attrs={"type": "shot_clock", "reason": "shot clock expired with the ball held"})
        self._enter_dead("shot_clock", t0, f, last_team=team, last_holder=holder)

    # ------------------------------------------------------------------ reconciliation

    def _reconcile(self, now: float, force: bool = False) -> None:
        inflight: set[Team | None] = set()
        f = self._flight
        if f is not None and f.context == "live" and (f.state == "shot" or (f.state == "air" and f.samples)):
            inflight.add(f.release.team)
        actions = self.reconciler.step(now, list(self._recent), self.scoreboard.is_streaming(now), inflight, force=force)
        for a in actions:
            self._apply(a, now)

    def _apply(self, a: Action, now: float) -> None:
        entry = self._shots.get(a.shot_id) if a.shot_id else None
        ch = a.change
        conf_attrs = {"confirmed_by": "scoreboard", "needs_review": False}
        if a.kind == "confirm" and entry is not None:
            entry.confirmed = entry.score_matched = True
            self._revise(entry.result_id, status=EventStatus.CONFIRMED, attrs=conf_attrs)
            self._revise(entry.attempt_id, status=EventStatus.CONFIRMED, attrs={"confirmed_by": "scoreboard"})
        elif a.kind == "fix_value" and entry is not None and a.new_value is not None:
            entry.confirmed = entry.score_matched = True
            entry.value = a.new_value
            self._revise(entry.attempt_id, value=a.new_value, status=EventStatus.CONFIRMED,
                         attrs={"confirmed_by": "scoreboard", "value_fixed_by": "scoreboard"})
            self._revise(entry.result_id, value=a.new_value, status=EventStatus.CONFIRMED,
                         attrs=dict(conf_attrs, value_fixed_by="scoreboard"))
        elif a.kind == "flip_to_made" and entry is not None and a.new_value is not None:
            self._flip_to_made(entry, a.new_value)
        elif a.kind == "make_to_miss" and entry is not None:
            self._make_to_miss(entry, now)
        elif a.kind == "correction" and ch is not None:
            self._emit(EventKind.SCORE_CORRECTION, ch.t, self._frame_near(ch.t), team=ch.team, value=ch.delta,
                       source="scoreboard", status=EventStatus.CONFIRMED, prefix="r",
                       attrs={"needs_review": True, "home": ch.home, "away": ch.away,
                              "reason": ch.reason or "scoreboard change with no matching shot"})

    def _frame_before(self, t: float) -> tuple[float, int] | None:
        """(t, frame_idx) of the last processed frame strictly before ``t``, if still buffered."""
        for bt, bf, _ in reversed(self._buffer):
            if bt < t:
                return bt, bf
        return None

    def _frame_near(self, t: float) -> int:
        for bt, bf, _ in reversed(self._buffer):
            if bt <= t:
                return bf
        return self._buffer[0][1] if self._buffer else 0

    def _flip_to_made(self, entry: ShotEntry, value: int, by: str = "scoreboard") -> None:
        """A shot called missed (or still unresolved) was made: revise it (``by`` "scoreboard") and
        undo the miss's consequences (rebound, block, possession cause, missing assist). For
        ``by="operator"`` the operator's own event is the new result revision."""
        unresolved = entry.made is None
        entry.made, entry.confirmed = True, True
        if by == "scoreboard":
            entry.score_matched = True
        if value != entry.value:
            entry.value = value
            self._revise(entry.attempt_id, value=value, attrs={"value_fixed_by": by})
        self._revise(entry.attempt_id, status=EventStatus.CONFIRMED, attrs={"confirmed_by": by})
        if by == "scoreboard":
            self._revise(entry.result_id, kind=EventKind.SHOT_MADE, value=value, status=EventStatus.CONFIRMED,
                         attrs={"flipped_by": "scoreboard", "confirmed_by": "scoreboard", "points": value,
                                "needs_review": False, "reason": "scoreboard delta matched a shot called missed"})
        self._retract(entry.rebound_id, f"the shot was made ({by})")
        self._retract(entry.block_id, f"the shot was made ({by})")
        entry.rebound_id = entry.block_id = None
        if entry.possession_id is not None:
            self._revise(entry.possession_id, attrs={"cause": "free_throw" if entry.is_ft else "made_shot"})
        if (self._pending_miss is entry or unresolved) and not entry.is_ft:  # no rebound yet: dead ball after a make
            self._pending_miss = None
            self._flight = None
            res_t = entry.result_t if entry.result_t is not None else entry.t_release
            res = self._store.get(entry.result_id or "")
            self._enter_dead("made_fg", res_t, res.frame_idx if res else self._frame_near(res_t), scoring_team=entry.team,
                             shot_id=entry.shot_id)
            self.deadball.note_stoppage(res_t, "made_basket")
        elif self._pending_miss is entry:
            self._pending_miss = None
        if entry.assist_id is None and not entry.is_ft:
            res = self._store.get(entry.result_id or "")
            if res is not None:
                self._emit_assist(entry, res.t, res.frame_idx)

    def _make_to_miss(self, entry: ShotEntry, now: float, by: str = "scoreboard") -> None:
        """A shot called made was missed: revise it (``by`` "scoreboard": no delta while the board
        streamed), retract its assist and turn a pickup during the post-make dead ball into the
        rebound. For ``by="operator"`` the operator's own event is the new result revision."""
        entry.made = False
        if by == "scoreboard":
            old = self._store.get(entry.result_id or "")
            conf = min(max(1.0 - old.confidence, 0.5), self.config.make_to_miss_conf) if old is not None else None
            self._revise(entry.result_id, kind=EventKind.SHOT_MISSED,
                         **({"confidence": conf} if conf is not None else {}),
                         attrs={"needs_review": True, "points": 0,
                                "reason": f"no scoreboard change within {self.config.make_confirm_wait_s:.1f} s"})
        self._retract(entry.assist_id, f"the shot was not made ({by})")
        entry.assist_id = None
        d = self._dead
        if d is None or d.shot_id != entry.shot_id or entry.is_ft:
            return
        if d.first_pickup is not None:
            t, f, pid = d.first_pickup
            ch = ControlChange("gain", pid, t, f)
            self._go_live(t)
            self._rebound(pid, self._team_of(pid), ch, entry)
            ctrl = self._control
            if ctrl is not None and ctrl.player_id == pid:
                ctrl.origin = "rebound"
        else:
            self._go_live(now)
            self._pending_miss = entry

    # ------------------------------------------------------------------ manual events

    def _manual_side_effects(self, ev: Event, old: Event | None) -> None:
        t, f = ev.t, ev.frame_idx
        if ev.kind is EventKind.FOUL and ev.status is not EventStatus.RETRACTED and old is None:
            self._on_manual_foul(ev)
        elif ev.kind is EventKind.TIMEOUT and old is None:
            self.deadball.note_stoppage(t, "timeout")
            if self._phase == LIVE:
                self._enter_dead_from_live("unknown", t, f, self._flight, out=False)
        if ev.kind in (EventKind.SHOT_ATTEMPT, EventKind.SHOT_MADE, EventKind.SHOT_MISSED):
            entry = self._shots.get(ev.shot_id) if ev.shot_id else None
            if entry is None:
                if old is None and ev.status is not EventStatus.RETRACTED:
                    self._register_manual_shot(ev)
            elif ev.kind is not EventKind.SHOT_ATTEMPT:
                self._manual_result(entry, ev)

    def _register_manual_shot(self, ev: Event) -> None:
        """A shot the engine never saw, inserted by the operator: ledger it so that its scoreboard
        delta confirms it (instead of becoming a SCORE_CORRECTION counted on top of it)."""
        is_ft = ev.attrs.get("shot_type") == "free_throw" or ev.value == 1
        value = ev.value if ev.value in (1, 2, 3) else (1 if is_ft else 2)
        sid = ev.shot_id or f"m:{ev.event_id}"
        entry = ShotEntry(sid, ev.team or self._team_of(ev.player_id), ev.player_id, value,
                          ev.event_id if ev.kind is EventKind.SHOT_ATTEMPT else "", ev.t, is_ft=is_ft)
        if ev.kind is not EventKind.SHOT_ATTEMPT:
            entry.made, entry.result_id, entry.result_t = ev.kind is EventKind.SHOT_MADE, ev.event_id, ev.t
        entry.confirmed = True
        self._shots[sid] = entry
        self._recent.append(entry)

    def _manual_result(self, entry: ShotEntry, ev: Event) -> None:
        """Operator made / missed call for a known shot: the same consequences as a scoreboard
        flip (rebound / block retraction, possession cause, assist), and the shot still absorbs its
        scoreboard delta later (``score_matched`` stays False) instead of becoming a correction."""
        made = ev.kind is EventKind.SHOT_MADE and ev.status is not EventStatus.RETRACTED
        fl = self._flight
        if fl is not None and fl.shot_id == entry.shot_id and not fl.resolved:
            fl.state = "made" if made else "missed"  # the operator resolved it; the flight emits no result
            if made:
                self._flight = None
        if entry.result_t is None:
            entry.result_t = ev.t
        if made and ev.value in (2, 3) and not entry.is_ft:
            value = ev.value
        else:
            value = entry.value
        if made and entry.made is not True:
            self._flip_to_made(entry, value, by="operator")
        elif not made and entry.made is not False:
            if entry.made is None:
                entry.made = False
                self._pending_miss = None if entry.is_ft and self._ft is not None else entry
            else:
                self._make_to_miss(entry, ev.t, by="operator")
        else:
            entry.value = value
        entry.confirmed = True
        if entry.result_id is not None and entry.result_id != ev.event_id:
            self._retract(entry.result_id, "replaced by an operator entry")
        entry.result_id = ev.event_id

    def _offensive_foul_turnover(self, ev: Event) -> None:
        """An offensive (team-control) foul is a turnover by the fouler [15]; the other team's
        inbound then starts its possession (no inferred out-of-bounds turnover). On a loose ball
        after a missed try nobody has team control: that is a loose-ball foul, the inbounding team
        gets a team rebound and there is no turnover."""
        d = self._dead
        if d is not None and d.miss is not None:
            return
        team = ev.team if ev.team is not None else self._team_of(ev.player_id)
        self._emit(EventKind.TURNOVER, ev.t, ev.frame_idx, team=team, player_id=ev.player_id, source="manual",
                   status=EventStatus.CONFIRMED, attrs={"type": "offensive_foul", "foul_id": ev.event_id,
                                                        "reason": "offensive foul entered by the operator"})
        if d is not None and d.cause in ("unknown", "oob"):
            d.cause = "turnover"

    def _on_manual_foul(self, ev: Event) -> None:
        cfg_rules = self.rules
        ftype = str(ev.attrs.get("type") or ev.attrs.get("foul_type") or "personal")
        fouler = ev.player_id
        technical = ftype in ("technical", "bench_technical", "coach_technical")
        if fouler is not None and (not technical or cfg_rules.foul_out_counts_technicals):
            self._fouls[fouler] = self._fouls.get(fouler, 0) + 1
            ev.attrs["personal_fouls"] = self._fouls[fouler]
            if self._fouls[fouler] >= cfg_rules.foul_out:
                self.fouled_out.add(fouler)
                ev.attrs["fouled_out"] = True
        awarded = ev.attrs.get("ft_awarded", ev.attrs.get("n_ft"))  # "n_ft": simulator / feed spelling
        if awarded is None and technical:
            awarded = int((cfg_rules.tech_ft or {}).get("n_ft", 1))
        if awarded is None and ftype in ("flagrant", "disruptive", "unsportsmanlike"):
            awarded = int((cfg_rules.tech_ft or {}).get("flagrant_n_ft", 2))
        self.deadball.note_stoppage(ev.t, "foul")
        if self._phase == LIVE:
            self._enter_dead_from_live("unknown", ev.t, ev.frame_idx, self._flight, out=False)
        if ftype in _OFFENSIVE_FOULS:
            self._offensive_foul_turnover(ev)
        if awarded:
            shooter = None if technical else ev.other_player_id
            team = self._team_of(shooter) if shooter else (ev.team.other() if ev.team is not None else None)
            live_last = ftype not in ("technical", "bench_technical", "coach_technical", "flagrant", "disruptive",
                                      "unsportsmanlike")
            trip = _FtTrip(shooter, team, int(awarded), live_last=live_last,
                           ft_type="technical" if technical else ("flagrant" if not live_last else "shooting"))
            ctrl = self._control
            if shooter is not None and ctrl is not None and ctrl.player_id == shooter:
                trip.assist_candidate = self._assist_candidate(
                    ctrl, Release(shooter, team, ev.t, ev.frame_idx, None, None, None, ctrl.onset_t))
            self._ft = trip


# ---------------------------------------------------------------------------- module helpers


_OFFENSIVE_FOULS = frozenset({"offensive", "offensive_foul", "charge", "charging"})  # FOUL attrs["type"] spellings


def _default_rules(league: str) -> RuleSpec:
    try:
        return rules_for(league)
    except (ValueError, FileNotFoundError):
        log.warning("no RuleSpec for league %r; using the NBA rules", league)
        return rules_for("nba")


def _sanitized(frame: FrameState) -> FrameState:
    """Drop players with non-finite floor positions (nan / inf from a degenerate homography) and
    zero non-finite velocities; the geometry helpers raise on non-finite coordinates."""
    ok = [p for p in frame.players if math.isfinite(p.x) and math.isfinite(p.y)]
    if len(ok) == len(frame.players) and all(math.isfinite(p.vx) and math.isfinite(p.vy) for p in ok):
        return frame
    players = [p if math.isfinite(p.vx) and math.isfinite(p.vy) else dataclasses.replace(p, vx=0.0, vy=0.0) for p in ok]
    log.debug("frame %d: dropped %d players with non-finite positions", frame.frame_idx, len(frame.players) - len(ok))
    return dataclasses.replace(frame, players=players)


def _px_per_ft(players: Sequence[PlayerState], player_height_ft: float) -> float | None:
    hs = [p.bbox.h for p in players if p.bbox is not None and p.bbox.h > 1.0]
    if not hs:
        return None
    return statistics.median(hs) / player_height_ft

