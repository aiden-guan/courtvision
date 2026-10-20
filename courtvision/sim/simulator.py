"""Deterministic (seeded) basketball game simulator: ground truth for evaluating the stack.

The simulator steps a 10-player game at ``SimConfig.fps``:

* players and referees are points with speed and acceleration limits;
* the ball is 3-D (feet, g = 32.17 ft/s^2): held ~1 ft ahead of the holder, dribbled with real
  floor bounces, passed and shot as true projectiles, deflected off the rim, bouncing and rolling;
* game and shot clocks follow ``RuleSpec`` (D15);
* a scripted-but-stochastic possession flow (bring-up, 1-6 passes, dribbles, drives, cuts, shots,
  rebounds, turnovers, steals, blocks, pre-shot fouls with free throws, substitutions, timeouts,
  periods, halftime side switch) draws its outcome mix from the NBA priors in
  ``docs/research/basketball_rules_stats.md`` (the "note").

Truth events follow ARCHITECTURE "Shared event semantics" (CONFIRMED, source "sim"); the box score
is tallied by the simulation logic itself, separately from the event list.

Flow control uses one generator per game phase (period start, jump ball, inbound, bring-up, half
court, shot, rebound, dead ball, free throws, timeout, period end). Each frame the main loop
advances physics, checks interrupts (period buzzer, shot clock, ball out of bounds), steps the
active phase, updates AI targets and records the frame.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from courtvision.geometry.camera import PinholeCamera
from courtvision.geometry.court import CourtSpec, court_for
from courtvision.geometry.rules import RuleSpec, rules_for
from courtvision.sim.names import make_roster
from courtvision.types import (
    BallState,
    BBox,
    Event,
    EventKind,
    EventStatus,
    FrameState,
    PlayerState,
    RimObservation,
    Roster,
    ScoreboardReading,
    Side,
    Team,
)

__all__ = ["GRAVITY_FTS2", "BOX_STATS", "GameSimulator", "SimConfig", "SimResult", "cylinder_boxes", "sphere_boxes"]

log = logging.getLogger(__name__)

GRAVITY_FTS2 = 32.17  # standard gravity in ft/s^2; sourced
BOX_STATS: tuple[str, ...] = ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk",
                              "tov", "pf")
_TEAMS = (Team.HOME, Team.AWAY)
# Descent angle at the rim per shot type (deg); typical arcs, unverified (tuned for visual realism).
_ENTRY_ANGLE_DEG: dict[str, tuple[float, float]] = {
    "jumper": (40.0, 50.0), "hook": (45.0, 55.0), "dunk": (55.0, 72.0), "layup": (45.0, 60.0), "tip": (45.0, 60.0),
    "free_throw": (43.0, 50.0)}
_EPS = 1e-9

Phase = Generator[None, None, Any]  # a phase returns the next phase generator (or None)


def _cfg_dict(d: dict[str, float]) -> Any:
    return field(default_factory=lambda: dict(d))


@dataclass
class SimConfig:
    """Simulator settings. "note" = docs/research/basketball_rules_stats.md."""

    league: str = "nba"
    seconds: float = 600.0  # video seconds simulated (frames = seconds * fps)
    fps: float = 30.0
    seed: int = 7
    image_size: tuple[int, int] = (1280, 720)
    # Camera: the geometry agent's verified framing (geometry.camera.RECOMMENDED_SIM_CAMERA). The
    # ARCHITECTURE default position (47, -38, 26) puts both near court corners off-image.
    camera_position: tuple[float, float, float] = (47.0, -56.0, 48.0)
    camera_target: tuple[float, float, float] = (47.0, 25.0, 0.0)
    hfov_deg: float = 72.0
    ruleset: str | None = None  # RuleSpec id; None = league
    rule_overrides: list[str] = field(default_factory=list)
    period_s: float | None = None  # game-clock length of a regulation period (None = RuleSpec)
    ot_s: float | None = None  # overtime length (None = RuleSpec)
    timeout_video_s: float = 8.0  # video seconds a timeout lasts (keeps sims short); design choice
    period_break_video_s: float = 6.0  # video seconds between periods; design choice
    halftime_video_s: float = 10.0  # video seconds of halftime; design choice
    roster_size: int = 12  # players per team (10-12 typical)
    home_name: str | None = None
    away_name: str | None = None

    # --- kinematics
    max_speed_fts: float = 22.0  # hard player speed cap (~15 mph sprint); spec, unverified
    sprint_speed_fts: float = 19.0  # tuned
    run_speed_fts: float = 15.0  # tuned
    jog_speed_fts: float = 11.0  # tuned
    walk_speed_fts: float = 6.0  # tuned
    accel_fts2: float = 16.0  # player acceleration limit; unverified
    decel_fts2: float = 12.0  # arrival braking used to slow down on targets; tuned
    player_height_ft: float = 6.5  # bbox cylinder height; spec
    player_radius_ft: float = 1.0  # bbox cylinder radius; spec
    referee_height_ft: float = 6.2  # unverified
    referee_speed_fts: float = 16.0  # tuned
    separation_ft: float = 2.2  # players closer than this push apart; tuned

    # --- ball
    ball_radius_ft: float = 0.39  # size 7: 29.5 in circumference -> 9.39 in diameter (note §1a); sourced
    ball_restitution: float = 0.77  # FIBA 1.8 m drop rebounds 1.035-1.085 m -> sqrt(1.06/1.8); derived
    floor_friction: float = 0.85  # horizontal speed kept per floor bounce; tuned
    roll_decel_fts2: float = 4.0  # rolling deceleration; tuned
    roll_below_fts: float = 2.0  # bounces slower than this become rolling; tuned
    # A dead ball moving away from the court is stopped (bench, table, fans) this far beyond the
    # boundary, so throw-in tosses start near the line instead of from 100 ft away; design choice.
    dead_ball_stop_ft: float = 6.0
    held_z_ft: float = 3.75  # held ball centre height (spec 3.5-4 ft)
    hand_offset_ft: float = 1.0  # held ball ahead of the holder (spec)
    # A caught / rebounded ball is carried from the catch point to the hand at this relative speed
    # instead of snapping there (keeps the truth trajectory continuous); tuned.
    carry_speed_fts: float = 30.0
    ft_gather_s: float = 0.35  # the FT shooter raises the ball to the release height over this time; tuned
    dribble_period_s: tuple[float, float] = (0.45, 0.6)  # hand-floor-hand cycle, ~2 Hz (spec); tuned
    dribble_hand_z_ft: float = 3.6  # dribble hand height; tuned
    pass_speed_fts: tuple[float, float] = (30.0, 45.0)  # horizontal pass speed (spec)
    lazy_pass_speed_fts: tuple[float, float] = (20.0, 26.0)  # intercepted passes; tuned
    jumper_release_z_ft: tuple[float, float] = (7.5, 9.0)  # spec
    layup_release_z_ft: tuple[float, float] = (9.0, 10.5)  # layup/dunk release (spec)
    hook_release_z_ft: tuple[float, float] = (8.5, 9.5)  # unverified
    ft_release_z_ft: tuple[float, float] = (7.3, 8.0)  # unverified

    # --- possession flow
    pass_count_weights: tuple[float, ...] = (0.18, 0.24, 0.22, 0.17, 0.11, 0.08)  # P(1..6 passes); tuned
    hold_s: tuple[float, float] = (0.6, 2.0)  # holding / probing time between passes; tuned
    urgency_s: float = 3.0  # shot clock / game clock left that forces a shot; tuned
    relocate_s: tuple[float, float] = (2.5, 4.5)  # off-ball relocation period; tuned

    # --- outcome priors (per half-court segment; NBA)
    turnover_rate: float = 0.135  # per segment; realises ~12-14% of possessions (spec; note §4c 0.144); tuned
    turnover_mix: dict[str, float] = _cfg_dict({  # steals ~54% (note §4c: 57% live-ball); tuned
        "intercept": 0.27, "strip": 0.27, "oob": 0.22, "travel": 0.14, "shot_clock": 0.10})
    after_oreb_scale: float = 0.5  # turnover / foul rates after an offensive rebound are scaled; tuned
    foul_rate: float = 0.10  # pre-shot fouls -> 2 FTs per segment (spec 8-10% of possessions); tuned
    shot_mix: dict[str, float] = _cfg_dict({  # share of FGA by distance, NBA 2024-25 (note §4b); sourced
        "0_3": 0.226, "3_10": 0.214, "10_16": 0.091, "16_3p": 0.048, "3p": 0.421})
    corner3_share: float = 0.243  # corner 3s / 3PA, 2024-25 (note §4b); sourced
    putback_share: float = 0.02  # 0-3 ft FGA expected from putbacks, removed from the scripted mix; tuned
    fg_pct: dict[str, float] = _cfg_dict({  # FG% by bin, 2024-25 (note §4b, corner/arc 3 from §4c); sourced
        "0_3": 0.696, "3_10": 0.461, "10_16": 0.443, "16_3p": 0.400, "c3": 0.388, "ab3": 0.351,
        "heave": 0.027})  # heave = 16/584 made (note §4c); sourced
    # Defender-distance adjustment: FG% += slope * (closest_defender_ft - ref_ft), clipped to +-0.1.
    # Note §4d: placeholder priors only (open shots better within a zone); slopes unverified. Each
    # ref_ft is the simulator's mean closest-defender distance in that bin (tuned), so the bin's
    # average FG% stays at the sourced prior.
    defender_ref_ft: dict[str, float] = _cfg_dict({
        "0_3": 1.8, "3_10": 2.5, "10_16": 3.3, "16_3p": 4.0, "c3": 8.0, "ab3": 5.6, "heave": 6.0})
    defender_slope: dict[str, float] = _cfg_dict({
        "0_3": 0.03, "3_10": 0.02, "10_16": 0.015, "16_3p": 0.015, "c3": 0.012, "ab3": 0.012, "heave": 0.0})
    block_rate: dict[str, float] = _cfg_dict({  # P(block) by bin; spec ~5-6% of 2PA overall; tuned
        "0_3": 0.045, "3_10": 0.045, "10_16": 0.02, "16_3p": 0.01, "c3": 0.008, "ab3": 0.008, "heave": 0.0})
    block_max_defender_ft: float = 6.0  # no block when the closest defender is farther; tuned
    dunk_share_0_3: float = 0.27  # dunks 6.1% FGA / 0-3 ft 22.6% FGA (note §4b); derived
    ft_pct: float = 0.780  # NBA 2024-25 league FT% (note §4a); sourced
    ft_pct_sd: float = 0.06  # per-player spread; unverified
    oreb_rate: float = 0.25  # offensive rebound rate on FG misses (spec ~25%; note ORB% 25.2); sourced
    oreb_rate_ft: float = 0.131  # on missed final FTs, 2024-25 (note §4c); sourced
    oreb_rate_block: float = 0.40  # after blocks (note: 58% of blocks recovered by the defense); unverified
    putback_rate: float = 0.40  # immediate putback after an OREB within 6 ft of the rim; tuned
    team_rebound_rate: float = 0.03  # misses deflected out of bounds before control; unverified
    assist_window_s: float = 5.0  # ARCHITECTURE assist rule (catch -> release)
    assist_max_dribbles: int = 2  # ARCHITECTURE assist rule
    ft_set_s: float = 1.5  # shooter stationary at the FT line before each FT (RESEARCH §7A.5 rule 2)

    # --- rotation & stoppages
    sub_prob: float = 0.3  # P(a team substitutes) at an eligible dead ball; tuned
    min_stint_s: float = 120.0  # game seconds on court before a fatigue substitution; tuned
    min_rest_s: float = 60.0  # game seconds on the bench before re-entry; tuned
    timeout_prob: float = 0.05  # P(timeout) at a whistle dead ball; tuned


@dataclass
class SimResult:
    """Simulation output: truth frames, truth events and an independently tallied box score."""

    config: SimConfig
    court: CourtSpec
    roster: Roster
    camera: PinholeCamera
    frames: list[FrameState]
    events: list[Event]
    truth_box: dict[str, dict[str, int]]  # player_id -> counting stats (BOX_STATS)
    truth_score: dict[Team, int]
    referees: list[list[tuple[float, float]]] = field(default_factory=list)  # per frame, 3 refs
    substitutions: list[dict[str, Any]] = field(default_factory=list)  # {"t","frame_idx","team","in","out"}
    rules: RuleSpec | None = None
    # Team-only entities (not in truth_box): team_reb / team_oreb / team_dreb / team_tov.
    team_box: dict[Team, dict[str, int]] = field(default_factory=dict)
    # Ball holder per frame; None while the ball is in the air, loose or dead (a pass release frame
    # already shows the ball in flight, still at the passer's hand).
    holders: list[str | None] = field(default_factory=list)
    starters: dict[Team, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------- camera helpers


def _conic_boxes(cam: PinholeCamera, centers: np.ndarray, A: np.ndarray, radius: float) -> np.ndarray:
    """Vectorised exact outline bboxes (H&Z §8.3 image dual conic), as in ``PinholeCamera``."""
    P = cam.P
    ch = centers @ P[:, :3].T + P[:, 3]
    w = ch[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        ou, ov = ch[:, 0] / w, ch[:, 1] / w
        a2 = A[2][None, :]
        row0 = (A[0][None, :] - ou[:, None] * a2) / w[:, None]
        row1 = (A[1][None, :] - ov[:, None] * a2) / w[:, None]
        row2 = a2 / w[:, None]
        r2 = radius * radius
        d00, d11 = -r2 * (row0 * row0).sum(1), -r2 * (row1 * row1).sum(1)
        d02, d12 = -r2 * (row0 * row2).sum(1), -r2 * (row1 * row2).sum(1)
        d22 = 1.0 - r2 * (row2 * row2).sum(1)
        su, sv = np.sqrt(d02 * d02 - d22 * d00), np.sqrt(d12 * d12 - d22 * d11)
        out = np.stack([(d02 - su) / d22 + ou, (d12 - sv) / d22 + ov, (d02 + su) / d22 + ou, (d12 + sv) / d22 + ov], 1)
    out[~(d22 > 0)] = np.nan
    return out


def _circle_boxes(cam: PinholeCamera, centers: np.ndarray, radius: float) -> np.ndarray:
    out = _conic_boxes(cam, centers, cam.K @ cam.R[:, :2], radius)
    near = cam.depth(centers) - radius * math.hypot(cam.R[2, 0], cam.R[2, 1])
    out[~(near > _EPS)] = np.nan
    return out


def cylinder_boxes(cam: PinholeCamera, xy: np.ndarray, height: float, radius: float) -> np.ndarray:
    """(N, 2) floor points -> (N, 4) exact bboxes of upright cylinders (= ``bbox_of_cylinder``)."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    bottom = np.column_stack([xy, np.zeros(len(xy))])
    top = np.column_stack([xy, np.full(len(xy), float(height))])
    a, b = _circle_boxes(cam, bottom, radius), _circle_boxes(cam, top, radius)
    out = np.column_stack([np.minimum(a[:, 0], b[:, 0]), np.minimum(a[:, 1], b[:, 1]),
                           np.maximum(a[:, 2], b[:, 2]), np.maximum(a[:, 3], b[:, 3])])
    out[np.isnan(a).any(1) | np.isnan(b).any(1)] = np.nan
    return out


def sphere_boxes(cam: PinholeCamera, xyz: np.ndarray, radius: float) -> np.ndarray:
    """(N, 3) centres -> (N, 4) exact sphere bboxes (= ``bbox_of_sphere``)."""
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    out = _conic_boxes(cam, xyz, cam.K @ cam.R, radius)
    out[~(cam.depth(xyz) > radius + _EPS)] = np.nan
    return out


# ---------------------------------------------------------------------------- small helpers


def _unit(v: np.ndarray) -> np.ndarray:
    n = math.hypot(float(v[0]), float(v[1]))
    return v / n if n > 1e-9 else np.zeros(2)


def _rot(v: np.ndarray, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


def _ballistic(p0: np.ndarray, p1: np.ndarray, T: float) -> np.ndarray:
    """Launch velocity that reaches ``p1`` after ``T`` seconds under gravity."""
    v = (p1 - p0) / T
    v[2] += 0.5 * GRAVITY_FTS2 * T
    return v


# ---------------------------------------------------------------------------- state records


@dataclass
class _Catch:
    receiver: int  # slot that may catch at time T (-1 = nobody)
    T: float  # flight time of the catch
    kind: str  # "pass" | "toss" | "tap" | "strip" | "inbound"
    interceptor: int | None = None
    Ti: float = math.inf
    reach: float = 3.5  # ft, horizontal hand reach
    p0: np.ndarray = field(default_factory=lambda: np.zeros(3))  # launch state (catch checks are exact
    v0: np.ndarray = field(default_factory=lambda: np.zeros(3))  # in time, independent of the fps)

    def pos_at(self, tau: float) -> np.ndarray:
        """Ball position ``tau`` s after launch (catch flights never bounce before the catch)."""
        p = self.p0 + self.v0 * tau
        p[2] -= 0.5 * GRAVITY_FTS2 * tau * tau
        return p


@dataclass
class _Shot:
    shot_id: str
    slot: int
    pid: str
    team: Team
    side: Side
    value: int
    shot_type: str
    made: bool
    blocked_by: int | None
    is_ft: bool
    release_t: float
    x: float
    y: float
    team_rebound_oob: bool = False
    rim_touched: bool = False


@dataclass
class _PassInfo:
    passer: str
    receiver: str
    team: Team
    t: float
    frame: int
    x: float
    y: float
    dist: float


@dataclass
class _SubTask:
    team: Team
    slot: int
    out_idx: int
    in_idx: int
    spot: np.ndarray
    start_t: float


@dataclass
class _Plan:
    outcome: str  # "shot" | "foul" | "intercept" | "strip" | "oob" | "travel" | "shot_clock"
    bin: str | None = None
    creation: str | None = None  # "catch" | "pullup" | "drive" | "cut" | "post"
    actor: int = -1
    passes: list[int] = field(default_factory=list)
    spot: np.ndarray | None = None  # release spot (court xy)
    corner: bool = False


class _Ball:
    """3-D ball state plus per-step physics flags."""

    def __init__(self, radius: float) -> None:
        self.r = radius
        self.pos = np.array([47.0, 25.0, 4.5])
        self.vel = np.zeros(3)
        self.mode = "rest"  # "held" | "dribble" | "flight" | "rest"
        self.holder: int | None = None
        self.held_z = 3.75
        self.flight_t = 0.0
        self.impulses: list[tuple[float, np.ndarray, str]] = []
        self.rolling = False
        self.net_left = 0.0
        self.dribble_phase = 0.0
        self.dribble_period = 0.5
        self.catch: _Catch | None = None
        self.no_oob = False
        self.carry = np.zeros(3)  # held/dribbled ball offset from its nominal hand position (decays to 0)
        self.aim = np.zeros(2)  # shooting motion: the held ball moves toward this release point ...
        self.aim_w = 0.0  # ... with this weight (0 = in the hand, 1 = at the release point)
        self.fired: list[str] = []
        self.contact = False
        self.bounced = False

    def clear_flags(self) -> None:
        self.fired = []
        self.contact = False
        self.bounced = False


# ---------------------------------------------------------------------------- simulator


class GameSimulator:
    """Seeded basketball simulator; ``run()`` returns a :class:`SimResult`."""

    def __init__(self, config: SimConfig | None = None) -> None:
        self.config = config or SimConfig()

    def run(self) -> SimResult:
        return _Game(self.config).run()


class _Game:
    """One simulated game (internal; see module docstring for the flow)."""

    # ------------------------------------------------------------------ setup

    def __init__(self, cfg: SimConfig) -> None:
        if cfg.fps <= 0 or cfg.seconds <= 0:
            raise ValueError("fps and seconds must be > 0")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.court = court_for(cfg.league)
        self.rules = rules_for(cfg.ruleset or cfg.league, list(cfg.rule_overrides) or None)
        self.camera = PinholeCamera.look_at(cfg.camera_position, cfg.camera_target, tuple(cfg.image_size), cfg.hfov_deg)
        self.roster = make_roster(self.rng, cfg.roster_size, cfg.home_name, cfg.away_name)
        self.dt = 1.0 / cfg.fps
        self.rim_h = self.court.rim_height
        self._setup_people()
        self._setup_state()

    def _setup_people(self) -> None:
        entries = self.roster.entries
        self.pid = [e.player_id for e in entries]
        self.jersey = [e.jersey for e in entries]
        self.role = [e.position or "F" for e in entries]
        self.team_of_idx = [e.team for e in entries]
        self.team_idx = {t: [i for i, e in enumerate(entries) if e.team is t] for t in _TEAMS}
        self.slot_team = [Team.HOME] * 5 + [Team.AWAY] * 5
        self.occ = np.array(self.team_idx[Team.HOME][:5] + self.team_idx[Team.AWAY][:5])
        n = len(entries)
        self.stint = np.zeros(n)
        self.rest = np.full(n, 1e6)
        self.fouls = np.zeros(n, dtype=int)
        self.ft_skill = np.clip(self.rng.normal(self.cfg.ft_pct, self.cfg.ft_pct_sd, n), 0.45, 0.95)
        for i in range(n):
            if self.role[i] == "C":
                self.ft_skill[i] = max(0.45, self.ft_skill[i] - 0.06)
        self.pos = np.zeros((10, 2))
        self.vel = np.zeros((10, 2))
        self.target = np.zeros((10, 2))
        self.vmax = np.full(10, self.cfg.jog_speed_fts)
        self.face = np.tile([1.0, 0.0], (10, 1))
        self.override: list[np.ndarray | None] = [None] * 10
        self.auto = np.zeros(10, dtype=bool)
        self.locked = np.zeros(10, dtype=bool)
        self.still_since = np.zeros(10)
        self.ref_pos = np.array([[47.0, -2.5], [47.0, self.court.width + 2.5], [-2.0, 25.0]])
        self.ref_vel = np.zeros((3, 2))
        self.ref_target = self.ref_pos.copy()
        mid, by = self.court.length / 2.0, self.court.basket_y
        for s in range(10):
            sign = -1.0 if s < 5 else 1.0
            self.pos[s] = (mid + sign * (6.0 + 3.0 * (s % 5)), by + (s % 5 - 2) * 6.0)
        self.target[:] = self.pos

    def _setup_state(self) -> None:
        c = self.cfg
        self.ball = _Ball(c.ball_radius_ft)
        self.ball.pos[:] = (self.court.length / 2.0, self.court.basket_y - 1.0, 4.5)  # referee's hand at tip-off
        self.t = 0.0
        self.frame = 0
        self.period = 0
        self.game_clock = 0.0
        self.clock_running = False
        self.sc: float | None = None
        self.sc_running = False
        self.score = {Team.HOME: 0, Team.AWAY: 0}
        self.attack: dict[Team, Side] = {Team.HOME: "right", Team.AWAY: "left"}
        self.possession_team: Team | None = None
        self.last_touch: Team | None = None
        self.holder: int | None = None
        self.last_holder_pid: str | None = None
        self.control_t = 0.0  # time the current holder gained the ball
        self.last_ft = True  # False while a non-final free throw is being shot
        self.ball_live = False
        self.shot_pending: _Shot | None = None
        self.loose_after: _Shot | None = None
        self.pass_info: _PassInfo | None = None
        self.period_expired = False
        self.sc_expired = False
        self.ball_out: np.ndarray | None = None
        self.def_auto_on = False
        self.ai_offense: Team = Team.HOME
        self.ai_ball_xy = np.array([47.0, 25.0])
        self.motion_on = False
        self.next_relocate = 0.0
        self.jump_winner: Team = Team.HOME
        self.arrow: Team = Team.AWAY
        self.timeouts_left = {t: self._timeout_allowance() for t in _TEAMS}
        self.sub_tasks: list[_SubTask] = []
        self.spell_start = 0.0
        self.spell_from: str | None = None
        self.spell_dribbles = 0
        self.events: list[Event] = []
        self.n_shots = 0
        self.box = {p: {k: 0 for k in BOX_STATS} for p in self.pid}
        self.team_box = {t: {"team_reb": 0, "team_oreb": 0, "team_dreb": 0, "team_tov": 0} for t in _TEAMS}
        self.subs_log: list[dict[str, Any]] = []
        self.starters = {t: [self.pid[i] for i in self.team_idx[t][:5]] for t in _TEAMS}
        self.game_over = False

    def _timeout_allowance(self) -> int:
        """Regulation timeouts per team from RuleSpec (per-game count, or the two halves' counts)."""
        to = self.rules.timeouts
        n = int(to.get("count") or 0) + int(to.get("short_count") or 0)
        if n == 0:
            n = int(to.get("first_half") or 0) + int(to.get("second_half") or 0)
        return n if n > 0 else 6  # unknown format: a generic allowance; design choice

    # ------------------------------------------------------------------ main loop

    def run(self) -> SimResult:
        n = max(1, int(round(self.cfg.seconds * self.cfg.fps)))
        self._alloc(n)
        self.gen: Phase = self._start_period(1)
        for i in range(n):
            self.frame, self.t = i, i / self.cfg.fps
            if i > 0:
                self._physics(self.dt)
            self._drive_phase()
            self._ai_targets()
            self._record(i)
        log.debug("simulated %d frames: %d events, score %s, period %d", n, len(self.events),
                  {t.value: v for t, v in self.score.items()}, self.period)
        return self._result(n)

    def _drive_phase(self) -> None:
        nxt = self._interrupt()
        if nxt is not None:
            self.gen.close()
            self.gen = nxt
        for _ in range(16):
            try:
                next(self.gen)
                return
            except StopIteration as stop:
                self.gen = stop.value if stop.value is not None else self._phase_idle()
        raise RuntimeError("simulator phase chain did not yield")

    def _interrupt(self) -> Phase | None:
        """Period buzzer, shot-clock violation or ball out of bounds pre-empt the current phase."""
        if self.period_expired and self.shot_pending is None and not self.game_over:
            self.period_expired = False
            return self._phase_period_end()
        if self.sc_expired and self.shot_pending is None:
            self.sc_expired = False
            loose = self.loose_after
            # A loose ball after a shot that never touched the rim (a block) does not stop the shot
            # clock: expiry before control is a violation, and no rebound is credited (note §2a).
            if loose is not None and not loose.rim_touched and not loose.is_ft:
                self.loose_after = loose = None
            if self.ball_live and loose is None and self.possession_team is not None:
                return self._shot_clock_violation()
        if self.ball_out is not None:
            point, self.ball_out = self.ball_out, None
            if self.ball_live and self.shot_pending is None:
                return self._out_of_bounds(point)
        return None

    # ------------------------------------------------------------------ physics

    def _physics(self, dt: float) -> None:
        self._move_players(dt)
        self._move_refs(dt)
        self._advance_ball(dt)
        self._advance_clocks(dt)
        self._run_subs()
        if self.ball.contact and self.ball.mode == "dribble" and self.holder is not None:
            self._on_dribble_contact()

    def _move_players(self, dt: float) -> None:
        c = self.cfg
        tgt = self.target.copy()
        for s, o in enumerate(self.override):
            if o is not None:
                tgt[s] = o
        d = tgt - self.pos
        dist = np.hypot(d[:, 0], d[:, 1])
        speed = np.minimum(self.vmax, np.sqrt(2.0 * c.decel_fts2 * dist))
        speed[dist < 0.05] = 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            v_des = np.where(dist[:, None] > 1e-9, d / dist[:, None], 0.0) * speed[:, None]
        v_des += self._separation()
        dv = v_des - self.vel
        mag = np.hypot(dv[:, 0], dv[:, 1])
        lim = c.accel_fts2 * dt
        scale = np.where(mag > lim, lim / np.maximum(mag, 1e-12), 1.0)
        self.vel += dv * scale[:, None]
        sp = np.hypot(self.vel[:, 0], self.vel[:, 1])
        cap = np.where(sp > c.max_speed_fts, c.max_speed_fts / np.maximum(sp, 1e-12), 1.0)
        self.vel *= cap[:, None]
        self.pos += self.vel * dt
        moving = np.hypot(self.vel[:, 0], self.vel[:, 1]) > 0.25
        self.still_since[moving] = self.t
        self._update_facing(dt)

    def _separation(self) -> np.ndarray:
        """Repulsion between players closer than ``separation_ft`` (not applied to locked players)."""
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        dist = np.hypot(diff[..., 0], diff[..., 1])
        np.fill_diagonal(dist, 1e6)
        sep = self.cfg.separation_ft
        close = dist < sep
        if not close.any():
            return np.zeros((10, 2))
        w = np.where(close, (sep - dist) / sep * 6.0 / np.maximum(dist, 0.1), 0.0)
        push = (diff * w[..., None]).sum(1)
        push[self.locked] = 0.0
        return push

    def _update_facing(self, dt: float) -> None:
        sp = np.hypot(self.vel[:, 0], self.vel[:, 1])
        for s in range(10):
            if sp[s] > 2.0:
                want = self.vel[s] / sp[s]
            else:
                want = _unit(self._face_point(s) - self.pos[s])
                if not want.any():
                    continue
            f = self.face[s] + (want - self.face[s]) * min(1.0, 8.0 * dt)
            n = math.hypot(f[0], f[1])
            if n > 1e-6:
                self.face[s] = f / n

    def _face_point(self, s: int) -> np.ndarray:
        team = self.slot_team[s]
        if self.holder == s and not self.court.in_bounds(*self.pos[s]):
            return np.array([self.court.length / 2.0, self.court.basket_y])
        return np.array(self.court.basket_xy(self.attack[team]))

    def _move_refs(self, dt: float) -> None:
        d = self.ref_target - self.ref_pos
        dist = np.hypot(d[:, 0], d[:, 1])
        speed = np.minimum(self.cfg.referee_speed_fts, np.sqrt(2.0 * self.cfg.decel_fts2 * dist))
        speed[dist < 0.05] = 0.0
        v_des = np.where(dist[:, None] > 1e-9, d / np.maximum(dist, 1e-9)[:, None], 0.0) * speed[:, None]
        dv = v_des - self.ref_vel
        mag = np.hypot(dv[:, 0], dv[:, 1])
        lim = self.cfg.accel_fts2 * dt
        self.ref_vel += dv * np.where(mag > lim, lim / np.maximum(mag, 1e-12), 1.0)[:, None]
        self.ref_pos += self.ref_vel * dt

    def _hand_xy(self, s: int, lateral: float = 0.0) -> np.ndarray:
        f = self.face[s]
        p = self.pos[s] + f * self.cfg.hand_offset_ft
        if lateral:
            p = p + np.array([f[1], -f[0]]) * lateral
        return p

    def _advance_ball(self, dt: float) -> None:
        b = self.ball
        b.clear_flags()
        if b.mode == "held" and b.holder is not None:
            self._decay_carry(self.cfg.carry_speed_fts * dt)
            b.pos[:] = self._held_nominal(b.holder) + b.carry
            b.vel[:2] = self.vel[b.holder]
            b.vel[2] = 0.0
        elif b.mode == "dribble" and b.holder is not None:
            self._advance_dribble(dt)
        elif b.mode == "flight":
            self._advance_flight(dt)
            self._detect_out()
            if not self.ball_live:
                self._stop_dead_ball_outside()

    def _stop_dead_ball_outside(self) -> None:
        b, L, W = self.ball, self.court.length, self.court.width
        out = np.array([b.pos[0] - min(max(b.pos[0], 0.0), L), b.pos[1] - min(max(b.pos[1], 0.0), W)])
        if math.hypot(*out) > self.cfg.dead_ball_stop_ft and float(out @ b.vel[:2]) > 0.0:
            b.vel[:2] = 0.0

    def _held_nominal(self, s: int) -> np.ndarray:
        """Where a held ball sits: ~1 ft ahead of the holder at ``held_z`` (moving to the release
        point during a shooting motion)."""
        b = self.ball
        xy = self._hand_xy(s)
        if b.aim_w > 0.0:
            xy = xy + b.aim_w * (b.aim - xy)
        return np.append(xy, b.held_z)

    def _dribble_nominal(self, s: int) -> np.ndarray:
        b = self.ball
        return np.append(self._hand_xy(s, lateral=0.5), self._dribble_z(b.dribble_phase, b.dribble_period))

    def _decay_carry(self, step: float) -> None:
        b = self.ball
        n = float(np.linalg.norm(b.carry))
        b.carry *= max(0.0, n - step) / n if n > 1e-12 else 0.0

    def _advance_dribble(self, dt: float) -> None:
        b = self.ball
        prev = b.dribble_phase
        b.dribble_phase += dt / b.dribble_period
        if prev < 0.5 <= b.dribble_phase:
            b.contact = True
        if b.dribble_phase >= 1.0:
            b.dribble_phase -= 1.0
            lo, hi = self.cfg.dribble_period_s
            fast = math.hypot(*self.vel[b.holder]) > 8.0
            b.dribble_period = lo if fast else float(self.rng.uniform(lo, hi))
        # the carry offset is gone by the floor contact, so DRIBBLE truth is always a real bounce
        if b.contact:
            b.carry[:] = 0.0
        else:
            to_contact = (0.5 - b.dribble_phase) * b.dribble_period if b.dribble_phase < 0.5 else math.inf
            n = float(np.linalg.norm(b.carry))
            self._decay_carry(max(self.cfg.carry_speed_fts * dt, n * dt / (to_contact + dt)))
        b.pos[:] = self._dribble_nominal(b.holder) + b.carry
        b.pos[2] = max(b.pos[2], b.r)
        b.vel[:2] = self.vel[b.holder]

    def _dribble_z(self, phase: float, period: float) -> float:
        """Hand -> floor -> hand: gravity plus a hand push, floor contact at phase 0.5."""
        h, r, g = self.cfg.dribble_hand_z_ft, self.ball.r, GRAVITY_FTS2
        tc = period / 2.0
        v0 = (h - r - 0.5 * g * tc * tc) / tc
        tau = phase * period
        if tau > tc:
            tau = period - tau
        return max(r, h - v0 * tau - 0.5 * g * tau * tau)

    def _advance_flight(self, dt: float) -> None:
        b = self.ball
        remaining = dt
        for _ in range(8):
            if remaining <= 1e-12:
                break
            step, hit = remaining, None
            if b.impulses and b.impulses[0][0] <= b.flight_t + step:
                step = max(0.0, b.impulses[0][0] - b.flight_t)
                hit = b.impulses.pop(0)
            self._ballistic(step)
            b.flight_t += step
            remaining -= step
            if hit is not None:
                b.vel = hit[1].copy()
                b.rolling = False
                b.fired.append(hit[2])
                if hit[2] == "make":  # the centre is in the rim plane inside the cylinder: the net takes it
                    b.net_left = 0.3

    def _ballistic(self, dt: float) -> None:
        """Exact projectile sub-step with floor bounces, rolling and net drag."""
        b, c = self.ball, self.cfg
        for _ in range(6):
            if dt <= 1e-12:
                return
            if b.rolling:
                sp = math.hypot(b.vel[0], b.vel[1])
                if sp > 1e-6:
                    new_sp = max(0.0, sp - c.roll_decel_fts2 * dt)
                    b.pos[:2] += b.vel[:2] * dt * (sp + new_sp) / (2.0 * sp)
                    b.vel[:2] *= new_sp / sp
                b.pos[2], b.vel[2] = b.r, 0.0
                return
            if b.net_left > 0.0:
                damp = math.exp(-14.0 * dt)
                b.vel[:2] *= damp
                b.vel[2] = max(b.vel[2], -7.0)
                b.net_left -= dt
            z0, vz = float(b.pos[2]), float(b.vel[2])
            z1 = z0 + vz * dt - 0.5 * GRAVITY_FTS2 * dt * dt
            if z1 >= b.r:
                b.pos[:2] += b.vel[:2] * dt
                b.pos[2] = z1
                b.vel[2] = vz - GRAVITY_FTS2 * dt
                return
            disc = vz * vz + 2.0 * GRAVITY_FTS2 * (z0 - b.r)
            tau = (vz + math.sqrt(max(disc, 0.0))) / GRAVITY_FTS2
            tau = min(max(tau, 0.0), dt)
            b.pos[:2] += b.vel[:2] * tau
            b.pos[2] = b.r
            vz_hit = vz - GRAVITY_FTS2 * tau
            b.vel[2] = -c.ball_restitution * vz_hit
            b.vel[:2] *= c.floor_friction
            b.bounced = True
            if b.vel[2] < c.roll_below_fts:
                b.rolling = True
                b.vel[2] = 0.0
            dt -= tau

    def _detect_out(self) -> None:
        b = self.ball
        if not self.ball_live or self.ball_out is not None:
            return
        x, y, z = b.pos
        if self.court.in_bounds(x, y):
            b.no_oob = False  # a throw-in becomes subject to the boundary once it is inside
            return
        if b.no_oob or self._catch_due(b):
            return
        if z < 4.5 or b.bounced or b.rolling:
            self.ball_out = b.pos.copy()

    def _catch_due(self, b: _Ball) -> bool:
        """A planned catch falls inside this step: it happened before the ball crossed the line."""
        c = b.catch
        if c is None:
            return False
        if c.interceptor is not None and b.flight_t >= c.Ti and self._reach(c.interceptor, 3.5, 8.5, at=c.pos_at(c.Ti)):
            return True
        return c.receiver >= 0 and b.flight_t >= c.T and self._reach(c.receiver, c.reach, at=c.pos_at(c.T))

    def _advance_clocks(self, dt: float) -> None:
        live_players = self.occ
        if self.clock_running:
            self.stint[live_players] += dt
            bench = np.ones(len(self.pid), dtype=bool)
            bench[live_players] = False
            self.rest[bench] += dt
            self.game_clock -= dt
            if self.game_clock <= 1e-9:
                self.game_clock = 0.0
                self.clock_running = False
                self.sc_running = False
                self.period_expired = True
        if self.sc_running and self.sc is not None:
            self.sc -= dt
            if self.sc <= 1e-9:
                self.sc = 0.0
                self.sc_running = False
                self.sc_expired = True

    def _on_dribble_contact(self) -> None:
        s = self.holder
        assert s is not None
        self.spell_dribbles += 1
        team = self.slot_team[s]
        x, y = float(self.ball.pos[0]), float(self.ball.pos[1])
        self._emit(EventKind.DRIBBLE, team=team, player=self._pid(s), x=x, y=y)

    # ------------------------------------------------------------------ ball control helpers

    def _pid(self, s: int) -> str:
        return self.pid[int(self.occ[s])]

    def _slots(self, team: Team) -> range:
        return range(0, 5) if team is Team.HOME else range(5, 10)

    @staticmethod
    def _opp(s: int) -> int:
        return (s + 5) % 10

    def _closest(self, team: Team, xy: np.ndarray) -> int:
        """Slot of ``team``'s player nearest to the floor point ``xy``."""
        return min(self._slots(team), key=lambda q: _dist(self.pos[q], xy))

    def _basket(self, team: Team) -> np.ndarray:
        return np.array(self.court.basket_xy(self.attack[team]))

    def _spot(self, side: Side, along: float, lateral: float) -> np.ndarray:
        bx, by = self.court.basket_xy(side)
        sign = 1.0 if side == "left" else -1.0
        return np.array([bx + sign * along, by + lateral])

    def _along_lat(self, side: Side, xy: np.ndarray) -> tuple[float, float]:
        bx, by = self.court.basket_xy(side)
        sign = 1.0 if side == "left" else -1.0
        return (float(xy[0]) - bx) * sign, float(xy[1]) - by

    def _hold(self, s: int, z: float | None = None) -> None:
        """Slot ``s`` holds the ball. It stays where the hands met it this frame and is then carried
        to the hand (``carry_speed_fts``), so catches and rebounds never teleport the ball."""
        b = self.ball
        if self.holder != s:
            self.control_t = self.t
        b.mode, b.holder, b.catch = "held", s, None
        b.held_z = self.cfg.held_z_ft if z is None else z
        b.aim_w = 0.0
        b.carry = b.pos - self._held_nominal(s)
        self.holder = s
        self.last_touch = self.slot_team[s]
        self.last_holder_pid = self._pid(s)

    def _pickup(self) -> Phase:
        """End a dribble when the ball is back in the hand (keeps the ball path continuous)."""
        b = self.ball
        end = self.t + 0.6
        while b.mode == "dribble" and 0.12 < b.dribble_phase < 0.88 and self.t < end - 1e-9:
            yield
        if b.mode == "dribble" and b.holder is not None:
            self._hold(b.holder)
        return None

    def _dribble(self, s: int) -> None:
        b = self.ball
        if b.mode != "dribble" or b.holder != s:
            self._hold(s)
            b.mode = "dribble"
            b.dribble_phase = 0.0
            b.dribble_period = float(self.rng.uniform(*self.cfg.dribble_period_s))
            b.carry = b.pos - self._dribble_nominal(s)

    def _launch(self, p1: np.ndarray, T: float, catch: _Catch | None, no_oob: bool = False) -> None:
        b = self.ball
        p0 = b.pos.copy()
        b.mode, b.holder, b.catch = "flight", None, catch
        b.carry[:] = 0.0
        b.aim_w = 0.0
        b.vel = _ballistic(p0, p1, max(T, 0.05))
        if catch is not None:
            catch.p0, catch.v0 = p0.copy(), b.vel.copy()
        b.flight_t, b.impulses, b.rolling, b.net_left, b.no_oob = 0.0, [], False, 0.0, no_oob
        self.holder = None

    def _reach(self, s: int, radius: float, zmax: float = 9.0, at: np.ndarray | None = None) -> bool:
        p = self.ball.pos if at is None else at
        return _dist(p, self.pos[s]) <= radius and p[2] <= zmax

    def _start_spell(self, from_pass: str | None) -> None:
        self.spell_start, self.spell_from, self.spell_dribbles = self.t, from_pass, 0

    def _gain(self, s: int, cause: str) -> None:
        """Slot ``s`` takes control; emits POSSESSION when the team changes."""
        team = self.slot_team[s]
        self._hold(s)
        self._start_spell(None)
        if team is not self.possession_team:
            self.possession_team = team
            self._emit(EventKind.POSSESSION, team=team, player=self._pid(s), x=float(self.pos[s, 0]),
                       y=float(self.pos[s, 1]), attrs={"cause": cause})
            self._reset_shot_clock(self.rules.shot_clock_s)
            self.sc_running = self.clock_running and self.ball_live

    def _reset_shot_clock(self, value: int | None) -> None:
        if value is None or self.rules.shot_clock_s is None:
            self.sc = None
        elif self.game_clock < value:
            self.sc = None  # shot clock off: less game time left than the reset (note §3)
        else:
            self.sc = float(value)
        self.sc_expired = False

    def _toss_to(self, s: int, speed: float = 25.0) -> None:
        """Dead-ball toss (referee / pick-up) to slot ``s``; no event."""
        p1 = np.append(self._hand_xy(s), self.cfg.held_z_ft)
        d = _dist(self.ball.pos, p1)
        self._launch(p1, max(0.35, d / speed), _Catch(s, max(0.35, d / speed), "toss", reach=4.0), no_oob=True)

    # ------------------------------------------------------------------ events & box

    def _emit(self, kind: EventKind, team: Team | None = None, player: str | None = None, other: str | None = None,
              x: float | None = None, y: float | None = None, value: int = 0, shot_id: str | None = None,
              attrs: dict[str, Any] | None = None, t: float | None = None, frame: int | None = None) -> Event:
        ev = Event(event_id="", kind=kind, t=self.t if t is None else t,
                   frame_idx=self.frame if frame is None else frame, team=team, player_id=player,
                   other_player_id=other, x=x, y=y, value=value, shot_id=shot_id, confidence=1.0,
                   status=EventStatus.CONFIRMED, source="sim", attrs=attrs or {})
        self.events.append(ev)
        return ev

    def _tally(self, pid: str, **counts: int) -> None:
        line = self.box[pid]
        for k, v in counts.items():
            line[k] += v

    def _emit_pass(self, completed: bool, catch_t: float | None = None) -> None:
        p = self.pass_info
        if p is None:
            return
        self.pass_info = None
        attrs: dict[str, Any] = {"distance_ft": round(p.dist, 3), "completed": completed}
        if catch_t is not None:
            attrs["catch_t"] = catch_t
        self._emit(EventKind.PASS, team=p.team, player=p.passer, other=p.receiver, x=p.x, y=p.y, attrs=attrs,
                   t=p.t, frame=p.frame)

    # ------------------------------------------------------------------ generic waits

    def _wait(self, seconds: float) -> Phase:
        end = self.t + seconds
        while self.t < end - 1e-9:
            yield

    def _wait_until(self, cond: Callable[[], bool], max_s: float) -> Phase:
        end = self.t + max_s
        while not cond() and self.t < end - 1e-9:
            yield

    def _phase_idle(self) -> Phase:
        while True:
            yield

    def _raise_ball(self, s: int, z_rel: float, seconds: float) -> Phase:
        """Shooting motion of slot ``s``: the held ball rises to ``z_rel`` and moves to the release
        point over ``seconds`` (reached on the frame the generator returns)."""
        b = self.ball
        z0, t0 = b.held_z, self.t
        while self.t - t0 < seconds - 1e-9:
            self._shooting_motion(s, z0, z_rel, min(1.0, (self.t + self.dt - t0) / max(seconds, 1e-9)))
            yield
        b.held_z = z_rel

    def _shooting_motion(self, s: int, z0: float, z_rel: float, w: float) -> None:
        """Set the held ball for the next frame's physics (which runs before the phase step)."""
        b = self.ball
        b.held_z = z0 + (z_rel - z0) * w
        b.aim[:] = self._release_xy(self.pos[s], self.attack[self.slot_team[s]])
        b.aim_w = w

    def _set_speed(self, slots: Any, v: float) -> None:
        for s in slots:
            self.vmax[s] = v

    def _arrived(self, slots: Any, tol: float = 0.8) -> bool:
        return all(_dist(self.pos[s], self.target[s]) <= tol for s in slots)

    # ------------------------------------------------------------------ periods

    def _period_length(self, period: int) -> float:
        if self.rules.is_overtime(period):
            return float(self.cfg.ot_s if self.cfg.ot_s is not None else self.rules.ot_s)
        return float(self.cfg.period_s if self.cfg.period_s is not None else self.rules.period_s)

    def _start_period(self, period: int) -> Phase:
        """PERIOD_START now; returns the opening phase (jump ball or throw-in)."""
        self.period = period
        self.game_clock = self._period_length(period)
        self.clock_running = self.sc_running = False
        self.sc = None
        self.period_expired = self.sc_expired = False
        self.possession_team = None
        self._emit(EventKind.PERIOD_START, attrs={"period": period})
        jump = period == 1 or (self.rules.is_overtime(period) and self.rules.possession == "jump")
        if jump:
            return self._phase_jump_ball()
        team = self._period_start_team(period)
        return self._phase_inbound(team, ("division", None), "period_start")

    def _period_start_team(self, period: int) -> Team:
        winner, loser = self.jump_winner, self.jump_winner.other()
        if self.rules.possession == "jump" and not self.rules.is_overtime(period):
            return winner if period == self.rules.n_periods and period > 2 else loser
        team = self.arrow
        self.arrow = team.other()
        return team

    def _phase_jump_ball(self) -> Phase:
        c, mid, by = self.cfg, self.court.length / 2.0, self.court.basket_y
        self.ball_live = False
        centers = {t: max(self._slots(t), key=lambda s: self.role[self.occ[s]] == "C") for t in _TEAMS}
        receivers: dict[Team, int] = {}
        for team in _TEAMS:
            sign = 1.0 if self.attack[team] == "right" else -1.0  # attack direction in x
            others = [s for s in self._slots(team) if s != centers[team]]
            self.target[centers[team]] = (mid - sign * 1.0, by)
            receivers[team] = others[0]
            spots = [(mid + sign * 8.0, by + 4.0 * (1 if team is Team.HOME else -1)), (mid - sign * 9.0, by - 9.0),
                     (mid - sign * 9.0, by + 9.0), (mid - sign * 14.0, by)]
            for s, p in zip(others, spots, strict=True):
                self.target[s] = p
        self._set_speed(range(10), c.walk_speed_fts)
        self.auto[:] = False
        self.ref_target[0] = (mid, by - 2.0)
        ref_hand = np.array([mid, by - 1.0, 4.5])
        d = float(np.linalg.norm(self.ball.pos - ref_hand))
        if d > 0.5:  # e.g. an overtime jump: the ball is tossed to the referee (it used to teleport)
            T = max(0.5, d / 20.0)
            self._launch(ref_hand, T, None, no_oob=True)
            yield from self._wait(T)
        self.ball.mode, self.ball.holder = "rest", None
        self.ball.pos[:] = ref_hand
        self.ball.vel[:] = 0.0
        yield from self._wait_until(lambda: self._arrived(range(10), 0.6), 6.0)
        yield from self._wait(0.6)
        self.ball.pos[:] = (mid, by, 5.0)
        up_down = 2.0 * math.sqrt(2.0 * GRAVITY_FTS2 * 7.0) / GRAVITY_FTS2  # toss peaks 7 ft above the hand
        self._launch(np.array([mid, by, 5.0]), up_down, None, no_oob=True)
        yield from self._wait_until(lambda: self.ball.vel[2] < 0.0 and self.ball.pos[2] <= 10.6, 3.0)
        winner = Team.HOME if self.rng.random() < 0.5 else Team.AWAY
        self.jump_winner, self.arrow = winner, winner.other()
        recv = receivers[winner]
        self.last_touch = winner
        self.clock_running = True  # the clock starts when the tossed ball is legally tapped
        self.ball_live = True
        p1 = np.append(self.pos[recv], c.held_z_ft)
        T = max(0.4, _dist(self.ball.pos, p1) / 22.0)
        self.target[recv] = self.pos[recv]
        self._launch(p1, T, _Catch(recv, T, "tap", reach=4.0), no_oob=True)
        result = yield from self._await_catch()
        if result != "caught":
            return self._phase_loose_ball(winner, cause="jump_ball")
        self._gain(recv, "jump_ball")
        return self._continue_with_ball(winner)

    def _phase_period_end(self) -> Phase:
        self.clock_running = self.sc_running = False
        self.ball_live = False
        self.shot_pending = self.loose_after = None
        self._emit_pass(False)
        self.motion_on = self.def_auto_on = False
        self.auto[:] = False
        self._emit(EventKind.PERIOD_END, attrs={"period": self.period})
        last_reg = self.period >= self.rules.n_periods
        if last_reg and self.score[Team.HOME] != self.score[Team.AWAY]:
            self.game_over = True
            self._walk_to_benches()
            return self._phase_idle()
        half = self.rules.n_periods // 2
        halftime = self.period == half and self.rules.n_periods % 2 == 0
        self._walk_to_benches()
        if self.ball.mode in ("held", "dribble") and self.ball.holder is not None:
            self._hold(self.ball.holder)
        yield from self._wait(0.5)
        self._maybe_subs(0.9 if halftime else 0.6, exclude=())
        yield from self._wait(self.cfg.halftime_video_s if halftime else self.cfg.period_break_video_s)
        yield from self._wait_until(lambda: not self.sub_tasks, 6.0)
        if halftime:
            self.attack = {t: self.court.other_side(s) for t, s in self.attack.items()}
        return self._start_period(self.period + 1)

    def _walk_to_benches(self) -> None:
        mid = self.court.length / 2.0
        for team in _TEAMS:
            bx = mid - 16.0 if team is Team.HOME else mid + 16.0
            for k, s in enumerate(self._slots(team)):
                self.target[s] = (bx + (k - 2) * 2.5, -3.0)
        self._set_speed(range(10), self.cfg.walk_speed_fts)

    # ------------------------------------------------------------------ inbound & bring-up

    def _inbound_spot(self, team: Team, spot: tuple[str, Any]) -> np.ndarray:
        kind, data = spot
        L, W, by = self.court.length, self.court.width, self.court.basket_y
        if kind == "baseline":
            # after a made basket: under the basket the opponent attacks; otherwise the baseline the ball crossed
            own = self.court.other_side(self.attack[team]) if data is None else self.court.half(float(data[0]))
            x = -1.5 if own == "left" else L + 1.5
            y = by + float(self.rng.choice([-1.0, 1.0])) * float(self.rng.uniform(4.0, 8.0))
            if data is not None:
                y = float(np.clip(data[1], 3.0, W - 3.0))
                if abs(y - by) < 4.0:
                    y = by + math.copysign(4.0, y - by if y != by else 1.0)
            return np.array([x, y])
        if kind == "division":
            return np.array([L / 2.0, -1.5])
        x = float(np.clip(data[0], 4.0, L - 4.0))
        y = -1.5 if data[1] < W / 2.0 else W + 1.5
        return np.array([x, y])

    def _phase_inbound(self, team: Team, spot: tuple[str, Any], cause: str) -> Phase:
        c = self.cfg
        self.ball_live = False
        self.clock_running = self.clock_running and cause == "made_shot"
        self.sc_running = False
        self.motion_on = False
        p_spot = self._inbound_spot(team, spot)
        inb = self._pick_inbounder(team, p_spot)
        recv = self._pick_receiver(team, inb)
        inside = np.array([float(np.clip(p_spot[0], 2.0, self.court.length - 2.0)),
                           float(np.clip(p_spot[1], 2.0, self.court.width - 2.0))])
        toward = _unit(self._basket(team) - inside) if spot[0] != "baseline" else _unit(
            np.array([self.court.length / 2.0, self.court.basket_y]) - inside)
        recv_spot = inside + toward * 7.0 + _rot(toward, math.pi / 2) * float(self.rng.uniform(-4.0, 4.0))
        self.locked[:] = False
        self.target[inb], self.target[recv] = p_spot, recv_spot
        self.locked[inb] = self.locked[recv] = True
        self._set_formation(team, exclude=(inb, recv), backcourt=spot[0] == "baseline")
        self._set_speed(range(10), c.run_speed_fts if cause == "made_shot" else c.jog_speed_fts)
        self._enable_defense(team)
        self.auto[self._opp(inb)] = False
        self.target[self._opp(inb)] = inside + toward * 3.0
        if self.ball.mode not in ("held", "dribble") or self.ball.holder != inb:
            if self.ball.mode in ("held", "dribble"):  # the official places the ball near the spot
                land = np.append(inside, self.ball.r)
                self._launch(land, max(0.6, _dist(self.ball.pos, land) / 20.0), None, no_oob=True)
            yield from self._wait_until(lambda: _dist(self.pos[inb], p_spot) < 1.0 and self._ball_settled(), 8.0)
            self._toss_to(inb)
            result = yield from self._await_catch()
            if result != "caught":
                self._hold(inb)
        self._hold(inb)
        if team is not self.possession_team:
            self._gain(inb, cause)
        else:
            self._start_spell(None)
        self.sc_running = False
        yield from self._wait_until(lambda: _dist(self.pos[inb], p_spot) < 0.8, 4.0)
        yield from self._wait(float(self.rng.uniform(0.8, 1.6)))
        yield from self._wait_until(lambda: not self.sub_tasks and _dist(self.pos[recv], recv_spot) < 2.0, 5.0)
        result = yield from self._act_pass(recv, "inbound")
        if isinstance(result, Generator):
            return result
        self.clock_running = True
        self.sc_running = self.sc is not None
        self.locked[inb] = False
        return self._continue_with_ball(team)

    def _ball_settled(self) -> bool:
        b = self.ball
        return b.mode == "rest" or (b.mode == "flight" and (b.rolling or b.flight_t > 0.9 or b.bounced))

    def _pick_inbounder(self, team: Team, spot: np.ndarray) -> int:
        slots = [s for s in self._slots(team) if not any(t.slot == s for t in self.sub_tasks)]
        weights = {"C": 0.6, "F": 1.0, "G": 0.5}
        return min(slots, key=lambda s: _dist(self.pos[s], spot) / weights[self.role[self.occ[s]]])

    def _pick_receiver(self, team: Team, inb: int) -> int:
        slots = [s for s in self._slots(team) if s != inb]
        guards = [s for s in slots if self.role[self.occ[s]] == "G"]
        return guards[0] if guards else slots[0]

    def _set_formation(self, team: Team, exclude: tuple[int, ...] = (), backcourt: bool = False) -> None:
        """Offensive spacing slots in the frontcourt (or a shallow bring-up shape)."""
        side = self.attack[team]
        flip = float(self.rng.choice([-1.0, 1.0]))
        spots = {0: (25.5, 0.0), 1: (17.5, 20.5 * flip), 2: (3.0, -23.2 * flip), 3: (17.5, -20.5 * flip),
                 4: (4.0, 8.5 * flip)}
        base = 0 if team is Team.HOME else 5
        for k in range(5):
            s = base + k
            if s in exclude:
                continue
            along, lat = spots[self._role_slot(s)]
            if backcourt:
                along += 12.0
            self.target[s] = self._spot(side, along, lat)
            self.locked[s] = False
        self.motion_on = not backcourt

    def _role_slot(self, s: int) -> int:
        return s % 5

    def _enable_defense(self, offense: Team) -> None:
        self.def_auto_on = True
        self.ai_offense = offense
        self.auto[:] = False
        for s in self._slots(offense.other()):
            self.auto[s] = True
            self.vmax[s] = self.cfg.sprint_speed_fts

    def _continue_with_ball(self, team: Team) -> Phase:
        """After control: bring the ball up or start the half-court set."""
        s = self.holder
        if s is None:
            return self._phase_loose_ball(team)
        along, _ = self._along_lat(self.attack[team], self.pos[s])
        if along > 31.0:
            return self._phase_bring_up(team)
        return self._phase_half_court(team)

    def _phase_bring_up(self, team: Team) -> Phase:
        c = self.cfg
        self.ball_live = True
        self._enable_defense(team)
        s = self.holder
        if s is None:
            return self._phase_loose_ball(team)
        guards = [g for g in self._slots(team) if g != s and self.role[self.occ[g]] == "G"]
        if self.role[self.occ[s]] != "G" and guards and self.rng.random() < 0.75:
            g = min(guards, key=lambda q: _dist(self.pos[q], self.pos[s]))
            self.target[g] = self.pos[s] + _unit(self._basket(team) - self.pos[s]) * 12.0
            self.vmax[g] = c.run_speed_fts
            yield from self._wait_until(lambda: _dist(self.pos[g], self.pos[s]) >= 9.0, 1.5)
            result = yield from self._act_pass(g, "normal")
            if isinstance(result, Generator):
                return result
            s = self.holder
            if s is None:
                return self._phase_loose_ball(team)
        self._set_formation(team, exclude=(s,))
        self._set_speed([q for q in self._slots(team) if q != s], c.sprint_speed_fts)
        side = self.attack[team]
        entry = self._spot(side, 27.0, float(self.rng.uniform(-9.0, 9.0)))
        self.target[s] = entry
        self.vmax[s] = float(self.rng.uniform(13.0, 17.0))
        self.locked[s] = True
        self._dribble(s)
        yield from self._wait_until(lambda: self._along_lat(side, self.pos[s])[0] < 30.0 or self._urgent(), 10.0)
        self.locked[s] = False
        return self._phase_half_court(team)

    # ------------------------------------------------------------------ half court

    def _urgent(self) -> bool:
        sc = self.sc if (self.sc is not None and self.sc_running) else math.inf
        gc = self.game_clock if self.clock_running else math.inf
        return min(sc, gc) < self.cfg.urgency_s

    def _phase_half_court(self, team: Team, after_oreb: bool = False, force_shot: bool = False) -> Phase:
        c = self.cfg
        self.ball_live = True
        self._enable_defense(team)
        self._set_speed(self._slots(team), c.jog_speed_fts)
        if self.holder is None:
            return self._phase_loose_ball(team)
        plan = self._plan_segment(team, after_oreb, force_shot)
        self._apply_plan_spots(team, plan)
        self.motion_on = True
        self.next_relocate = self.t + float(self.rng.uniform(*c.relocate_s))
        n = len(plan.passes)
        for k, recv in enumerate(plan.passes):
            last = k == n - 1
            nxt = yield from self._pre_pass(team, plan, recv, last)
            if nxt is not None:
                return nxt
            kind = "normal"
            if last and plan.outcome in ("intercept", "oob"):
                kind = plan.outcome
            nxt = yield from self._act_pass(recv, kind)
            if nxt is not None:
                return nxt
            if self.holder is None:
                return self._phase_loose_ball(team)
            if self.holder != plan.actor:
                self._apply_plan_spots(team, plan)
        if plan.outcome == "intercept":  # the interception failed: keep playing
            return self._phase_half_court(team, force_shot=True)
        return (yield from self._act_final(team, plan))

    def _pre_pass(self, team: Team, plan: _Plan, recv: int, last: bool) -> Phase:
        """Hold / probe before a pass, then make sure passer and receiver are apart."""
        nxt = yield from self._prepare_pass(team, plan, recv, last)
        if nxt is not None:
            return nxt
        return (yield from self._spread(recv, plan))

    def _spread(self, recv: int, plan: _Plan) -> Phase:
        """The holder dribbles away from a receiver closer than 8 ft (no hand-off passes)."""
        s = self.holder
        if s is None or _dist(self.pos[s], self.pos[recv]) >= 8.0:
            return None
        away = _unit(self.pos[s] - self.pos[recv])
        if not away.any():
            away = np.array([0.0, 1.0])
        dest = self.pos[s] + away * 10.0
        self.target[s] = (float(np.clip(dest[0], 2.0, self.court.length - 2.0)),
                          float(np.clip(dest[1], 2.0, self.court.width - 2.0)))
        self._dribble(s)
        return (yield from self._probe_until(lambda: _dist(self.pos[s], self.pos[recv]) >= 9.0, 1.5, plan))

    def _prepare_pass(self, team: Team, plan: _Plan, recv: int, last: bool) -> Phase:
        """Hold / probe; prepares the receiver (spot, cut) or the interceptor (passing lane)."""
        c = self.cfg
        dur = float(self.rng.uniform(*c.hold_s))
        s = self.holder
        assert s is not None
        if self.rng.random() < 0.55 and plan.outcome != "travel":
            self._dribble(s)
        else:
            yield from self._pickup()
        self.target[s] = self.pos[s] + _unit(self._basket(team) - self.pos[s]) * float(self.rng.uniform(0.0, 3.0))
        self.locked[s] = True
        if last and plan.creation == "cut":
            nxt = yield from self._probe(dur * 0.4, plan)
            if nxt is not None:
                return nxt
            self.vmax[recv] = c.sprint_speed_fts
            self.target[recv] = self._cut_catch_point(team, recv)
            self.locked[recv] = True
            nxt = yield from self._probe_until(lambda: _dist(self.pos[recv], self.target[recv]) < 7.0, 3.0, plan)
            return nxt
        if last and plan.creation in ("catch", "post") and plan.spot is not None:
            nxt = yield from self._probe(dur, plan)
            if nxt is not None:
                return nxt
            spot = plan.spot if plan.creation == "catch" else self._post_spot(team, recv)
            nxt = yield from self._probe_until(lambda: _dist(self.pos[recv], spot) < 0.5 and math.hypot(
                *self.vel[recv]) < 2.0, 4.0, plan)
            return nxt
        if last and plan.outcome == "intercept":
            nxt = yield from self._probe(max(0.2, dur - 0.9), plan)
            if nxt is not None:
                return nxt
            d = self._opp(recv)
            self.auto[d] = False
            self.vmax[d] = c.sprint_speed_fts
            self.target[d] = self.pos[s] + (self.pos[recv] - self.pos[s]) * 0.55  # jumps the passing lane
            return (yield from self._probe(0.9, plan))
        return (yield from self._probe(dur, plan))

    def _probe(self, seconds: float, plan: _Plan) -> Phase:
        return (yield from self._probe_until(lambda: False, seconds, plan))

    def _probe_until(self, cond: Callable[[], bool], max_s: float, plan: _Plan) -> Phase:
        end = self.t + max_s
        while not cond() and self.t < end - 1e-9:
            if self._urgent() and plan.outcome != "shot_clock":
                return self._phase_forced_shot()
            yield
        return None

    def _plan_segment(self, team: Team, after_oreb: bool, force_shot: bool) -> _Plan:
        c, rng = self.cfg, self.rng
        scale = 0.0 if force_shot else (c.after_oreb_scale if after_oreb else 1.0)
        to_rate, foul_rate = c.turnover_rate * scale, c.foul_rate * scale
        r = rng.random()
        if r < to_rate:
            kinds = list(c.turnover_mix)
            w = np.array([c.turnover_mix[k] for k in kinds], dtype=float)
            outcome = kinds[int(rng.choice(len(kinds), p=w / w.sum()))]
            if outcome == "shot_clock" and (self.sc is None or not self.rules.has_shot_clock):
                outcome = "travel"
            plan = _Plan(outcome=outcome)
            plan.actor = self._choose(team, {"G": 1.0, "F": 0.8, "C": 0.6})
        elif r < to_rate + foul_rate:
            plan = _Plan(outcome="foul", creation="drive", bin="0_3")
            plan.actor = self._choose(team, {"G": 1.0, "F": 0.9, "C": 0.7})
        else:
            plan = self._plan_shot(team)
        if plan.outcome == "oob":
            plan.actor = max((s for s in self._slots(team) if s != self.holder),
                             key=lambda s: abs(self.pos[s, 1] - self.court.basket_y))
        plan.passes = self._pass_chain(team, plan)
        return plan

    def _choose(self, team: Team, role_w: dict[str, float], exclude: tuple[int, ...] = ()) -> int:
        slots = [s for s in self._slots(team) if s not in exclude]
        w = np.array([role_w[self.role[self.occ[s]]] for s in slots], dtype=float)
        return slots[int(self.rng.choice(len(slots), p=w / w.sum()))]

    def _plan_shot(self, team: Team) -> _Plan:
        c, rng = self.cfg, self.rng
        mix = dict(c.shot_mix)
        mix["0_3"] = max(0.02, mix["0_3"] - c.putback_share)
        bins = list(mix)
        w = np.array([mix[b] for b in bins], dtype=float)
        b = bins[int(rng.choice(len(bins), p=w / w.sum()))]
        plan = _Plan(outcome="shot", bin=b)
        if b == "3p":
            plan.corner = bool(rng.random() < c.corner3_share)
            plan.creation = "catch" if rng.random() < (0.95 if plan.corner else 0.65) else "pullup"
            plan.actor = self._choose(team, {"G": 1.0, "F": 0.8, "C": 0.15})
        elif b in ("10_16", "16_3p"):
            plan.creation = "pullup" if rng.random() < 0.55 else "catch"
            plan.actor = self._choose(team, {"G": 1.0, "F": 0.9, "C": 0.4})
        elif b == "3_10":
            u = rng.random()
            plan.creation = "drive" if u < 0.5 else ("post" if u < 0.75 else "pullup")
            plan.actor = self._choose(team, {"G": 0.8, "F": 0.9, "C": 1.0} if plan.creation != "post"
                                      else {"G": 0.1, "F": 0.6, "C": 1.0})
        else:
            plan.creation = "drive" if rng.random() < 0.6 else "cut"
            plan.actor = self._choose(team, {"G": 0.7, "F": 0.9, "C": 1.3})
        if plan.creation in ("catch", "pullup"):
            plan.spot = self._shot_spot(team, b, plan.corner, plan.actor)
        return plan

    def _shot_spot(self, team: Team, b: str, corner: bool, actor: int) -> np.ndarray:
        """Release spot for a jumper of distance bin ``b`` on the actor's side of the floor."""
        rng, court = self.rng, self.court
        side = self.attack[team]
        _, lat_now = self._along_lat(side, self.pos[actor])
        sgn = 1.0 if lat_now >= 0 else -1.0
        if b == "3p" and corner:
            lat = court.corner_three_dist + float(rng.uniform(0.6, 1.6))
            lat = min(lat, court.width / 2.0 - 0.6)
            along = float(rng.uniform(-2.5, min(7.5, court.three_junction_along - 0.8)))
            return self._spot(side, along, sgn * lat)
        if b == "3p":
            d = court.three_pt_radius + float(rng.uniform(0.5, 3.5))
            jang = math.atan2(court.corner_three_dist, court.three_junction_along)
            ang = float(rng.uniform(-1.0, 1.0)) * min(math.radians(62.0), jang - 0.06)
        else:
            lo, hi = {"3_10": (3.4, 9.6), "10_16": (10.4, 15.6),
                      "16_3p": (16.4, min(21.3, court.three_pt_radius - 1.2))}[b]
            d = float(rng.uniform(lo, hi))
            ang = float(rng.uniform(-1.0, 1.0)) * math.radians(65.0)
        ang = abs(ang) * sgn
        spot = self._spot(side, d * math.cos(ang), d * math.sin(ang))
        spot[1] = float(np.clip(spot[1], 0.8, court.width - 0.8))
        return spot

    def _pass_chain(self, team: Team, plan: _Plan) -> list[int]:
        rng = self.rng
        k = int(rng.choice(len(self.cfg.pass_count_weights), p=np.array(self.cfg.pass_count_weights) /
                           sum(self.cfg.pass_count_weights))) + 1
        actor = plan.actor
        if actor == self.holder and k == 1:
            k = 2
        chain: list[int] = []
        prev = self.holder
        for _ in range(k - 1):
            pool = [s for s in self._slots(team) if s not in (prev, actor)]
            prev = pool[int(rng.integers(len(pool)))]
            chain.append(prev)
        chain.append(actor)
        return chain

    def _apply_plan_spots(self, team: Team, plan: _Plan) -> None:
        self.locked[:] = False
        if plan.creation == "catch" and plan.spot is not None:
            self.target[plan.actor] = plan.spot
            self.locked[plan.actor] = True
        elif plan.creation == "post":
            self.target[plan.actor] = self._post_spot(team, plan.actor)
            self.locked[plan.actor] = True

    def _post_spot(self, team: Team, s: int) -> np.ndarray:
        side = self.attack[team]
        _, lat = self._along_lat(side, self.pos[s])
        return self._spot(side, 5.5, 7.5 if lat >= 0 else -7.5)

    def _cut_catch_point(self, team: Team, s: int) -> np.ndarray:
        """Catch point of a basket cut: 6 ft from the rim on the cutter's side (design choice)."""
        side = self.attack[team]
        _, lat = self._along_lat(side, self.pos[s])
        ang = math.radians(float(self.rng.uniform(15.0, 55.0))) * (1.0 if lat >= 0 else -1.0)
        return self._spot(side, 6.0 * math.cos(ang), 6.0 * math.sin(ang))

    # ------------------------------------------------------------------ passes

    def _act_pass(self, recv: int, kind: str) -> Phase:
        """Throw a pass; returns None when caught (holder = receiver) or the next phase."""
        c, rng = self.cfg, self.rng
        yield from self._wait_until(lambda: self.t - self.control_t >= 0.3, 0.3)  # secure the ball first
        yield from self._pickup()
        s = self.holder
        assert s is not None
        team = self.slot_team[s]
        catch_xy = self.pos[recv] + self.vel[recv] * 0.3
        if kind == "oob":
            catch_xy = self._oob_point(recv)
        p1 = np.append(catch_xy, 4.0)
        dist = _dist(self.ball.pos, p1)
        speed = float(rng.uniform(*(c.lazy_pass_speed_fts if kind == "intercept" else c.pass_speed_fts)))
        if dist > 40.0:
            speed = min(speed, 32.0)
        T = max(0.2, dist / speed)
        if kind == "oob":  # overthrown lob: sails over the receiver's reach and lands out of bounds
            T = max(T, 1.0, dist / 24.0)
        else:
            self.target[recv] = catch_xy
            self.locked[recv] = True
        catch = _Catch(recv if kind != "oob" else -1, T, "inbound" if kind == "inbound" else "pass")
        if kind == "intercept":
            self._plan_interception(catch, s, recv, catch_xy, T)
        self.pass_info = _PassInfo(self._pid(s), self._pid(recv), team, self.t, self.frame, float(self.pos[s, 0]),
                                   float(self.pos[s, 1]), _dist(self.pos[s], catch_xy))
        self.locked[s] = False
        self._launch(p1, T, catch, no_oob=kind == "inbound")
        self.ball_live = True
        passer_pid = self._pid(s)
        result = yield from self._await_catch()
        if result == "caught":
            self._emit_pass(True, catch_t=self.t)
            self._start_spell(passer_pid)
            self.locked[recv] = False
            return None
        if result == "intercepted":
            return self._on_interception(catch.interceptor, passer_pid)
        # not caught: wait for the ball to go out of bounds (interrupt) or recover it
        yield from self._wait_until(lambda: False, 2.5)
        self._emit_pass(False)
        return self._phase_loose_ball(team)

    def _oob_point(self, recv: int) -> np.ndarray:
        W = self.court.width
        y = self.pos[recv, 1]
        out_y = -float(self.rng.uniform(4.0, 8.0)) if y < W / 2.0 else W + float(self.rng.uniform(4.0, 8.0))
        return np.array([self.pos[recv, 0] + float(self.rng.uniform(-4.0, 4.0)), out_y])

    def _plan_interception(self, catch: _Catch, s: int, recv: int, catch_xy: np.ndarray, T: float) -> None:
        d = self._opp(recv)
        a, b = self.pos[s], catch_xy
        ab = b - a
        L2 = float(ab @ ab)
        frac = float(np.clip(((self.pos[d] - a) @ ab) / max(L2, 1e-9), 0.35, 0.85))
        point = a + ab * frac
        catch.interceptor, catch.Ti = d, frac * T
        self.auto[d] = False
        self.target[d] = point
        self.vmax[d] = self.cfg.max_speed_fts

    def _await_catch(self) -> Generator[None, None, str]:
        """Wait for the pending catch: 'caught' | 'intercepted' | 'missed'."""
        b = self.ball
        c = b.catch
        if c is None:
            return "missed"
        while True:
            if b.mode != "flight" or b.catch is not c:
                return "missed"
            if c.interceptor is not None and b.flight_t >= c.Ti:
                if self._reach(c.interceptor, 3.5, 8.5, at=c.pos_at(c.Ti)):
                    self._hold(c.interceptor)
                    return "intercepted"
                c.interceptor = None
            if b.flight_t >= c.T:
                if c.receiver >= 0 and self._reach(c.receiver, c.reach, at=c.pos_at(c.T)):
                    self._hold(c.receiver)
                    return "caught"
                b.catch = None
                return "missed"
            yield

    def _on_interception(self, d: int | None, passer_pid: str) -> Phase:
        assert d is not None
        self._emit_pass(False)
        self._steal(d, passer_pid, "bad_pass")
        return self._continue_with_ball(self.slot_team[d])

    def _steal(self, d: int, victim_pid: str, to_type: str) -> None:
        team = self.slot_team[d]
        x, y = float(self.pos[d, 0]), float(self.pos[d, 1])
        self._emit(EventKind.STEAL, team=team, player=self._pid(d), other=victim_pid, x=x, y=y)
        self._emit(EventKind.TURNOVER, team=team.other(), player=victim_pid, x=x, y=y, attrs={"type": to_type})
        self._tally(self._pid(d), stl=1)
        self._tally(victim_pid, tov=1)
        self._gain(d, "steal")

    # ------------------------------------------------------------------ final actions

    def _act_final(self, team: Team, plan: _Plan) -> Phase:
        s = self.holder
        assert s is not None
        if plan.outcome == "strip":
            return (yield from self._act_strip(team, s))
        if plan.outcome == "travel":
            return (yield from self._act_travel(team, s))
        if plan.outcome == "shot_clock":
            return (yield from self._act_stall(team))
        if plan.outcome == "oob":
            return self._phase_half_court(team, force_shot=True)
        if plan.creation == "catch":
            nxt = yield from self._probe(float(self.rng.uniform(0.1, 0.35)), plan)
            if nxt is not None:
                return nxt
            return (yield from self._act_shoot(s, "jumper", tag="catch"))
        if plan.creation == "pullup":
            return (yield from self._act_pullup(s, plan))
        if plan.creation == "post":
            return (yield from self._act_post(team, s, plan))
        if plan.creation == "cut":
            return (yield from self._act_cut_finish(team, s))
        return (yield from self._act_drive(team, s, plan))

    def _act_pullup(self, s: int, plan: _Plan) -> Phase:
        assert plan.spot is not None
        spot = plan.spot
        self._dribble(s)
        self.target[s] = spot
        self.vmax[s] = float(self.rng.uniform(9.0, 13.0))
        self.locked[s] = True
        nxt = yield from self._probe_until(lambda: _dist(self.pos[s], spot) < 0.35 and math.hypot(
            *self.vel[s]) < 2.5, 4.5, plan)
        if nxt is not None:
            return nxt
        return (yield from self._act_shoot(s, "jumper", tag="pullup"))

    def _release_spot(self, team: Team, s: int, b: str) -> np.ndarray:
        side = self.attack[team]
        basket = self._basket(team)
        along, lat = self._along_lat(side, self.pos[s])
        u = _unit(self._spot(side, max(along, 0.8), lat) - basket)  # never behind the backboard plane
        lo, hi = {"0_3": (0.5, 2.8), "3_10": (3.4, 9.5)}[b]
        return basket + u * float(self.rng.uniform(lo, hi))

    def _act_drive(self, team: Team, s: int, plan: _Plan) -> Phase:
        c, rng = self.cfg, self.rng
        b = plan.bin if plan.bin in ("0_3", "3_10") else "0_3"
        release = self._release_spot(team, s, b)
        basket = self._basket(team)
        gather = release + _unit(release - basket) * (2.5 if b == "0_3" else 1.6)
        self._dribble(s)
        self.target[s] = gather
        self.vmax[s] = float(rng.uniform(14.0, 18.0))
        self.locked[s] = True
        nxt = yield from self._probe_until(lambda: _dist(self.pos[s], gather) < 1.0, 4.0, plan)
        if nxt is not None:
            return nxt
        if plan.outcome == "foul":
            yield from self._pickup()
            return self._foul_on(s)
        if b == "0_3":
            dunk = rng.random() < c.dunk_share_0_3 * (1.5 if self.role[self.occ[s]] != "G" else 0.5)
            kind = "dunk" if dunk else "layup"
            if dunk:
                release = basket + _unit(release - basket) * float(rng.uniform(0.5, 1.4))
        else:
            kind = "layup" if rng.random() < 0.5 else "jumper"
        self.vmax[s] = 8.0
        return (yield from self._act_shoot(s, kind, gather_xy=self.pos[s].copy(), release_target=release,
                                           gather_s=0.35, tag="drive"))

    def _act_post(self, team: Team, s: int, plan: _Plan) -> Phase:
        release = self._release_spot(team, s, "3_10")
        self._dribble(s)
        self.target[s] = release
        self.vmax[s] = 5.0
        self.locked[s] = True
        nxt = yield from self._probe_until(lambda: _dist(self.pos[s], release) < 0.4, 2.5, plan)
        if nxt is not None:
            return nxt
        return (yield from self._act_shoot(s, "hook", tag="post"))

    def _act_cut_finish(self, team: Team, s: int) -> Phase:
        release = self._release_spot(team, s, "0_3")
        dunk = self.rng.random() < self.cfg.dunk_share_0_3
        self.vmax[s] = 8.0
        return (yield from self._act_shoot(s, "dunk" if dunk else "layup", gather_xy=self.pos[s].copy(),
                                           release_target=release, gather_s=0.3, tag="cut"))

    def _phase_forced_shot(self) -> Phase:
        s = self.holder
        if s is None:
            return self._phase_loose_ball(self.possession_team or Team.HOME)
        team = self.slot_team[s]
        d = _dist(self.pos[s], self._basket(team))
        return (yield from self._act_shoot(s, "layup" if d < 4.0 else "jumper", gather_s=0.25, tag="forced"))

    def _act_strip(self, team: Team, s: int) -> Phase:
        d = self._opp(s)
        self._dribble(s)
        self.auto[d] = False
        self.vmax[d] = self.cfg.sprint_speed_fts
        t_end = self.t + float(self.rng.uniform(0.6, 1.2))
        while self.t < t_end or not self.ball.contact:
            self.target[d] = self.pos[s] + _unit(self._basket(team) - self.pos[s]) * 1.6
            if self.t > t_end + 2.0:
                break
            yield
        victim = self._pid(s)
        p1 = np.append(self._hand_xy(d), self.cfg.held_z_ft)
        T = 0.3
        self.target[d] = self.pos[d]
        self._launch(p1, T, _Catch(d, T, "strip", reach=4.0))
        result = yield from self._await_catch()
        if result != "caught":
            return self._phase_loose_ball(team.other(), steal_by=d, victim=victim)
        self._steal(d, victim, "steal")
        return self._continue_with_ball(team.other())

    def _act_travel(self, team: Team, s: int) -> Phase:
        yield from self._pickup()
        yield from self._wait(float(self.rng.uniform(0.6, 1.4)))
        pid = self._pid(s)
        self._emit(EventKind.TURNOVER, team=team, player=pid, x=float(self.pos[s, 0]), y=float(self.pos[s, 1]),
                   attrs={"type": "travel"})
        self._tally(pid, tov=1)
        return self._phase_dead_ball(team.other(), ("sideline", self.pos[s].copy()), "turnover")

    def _act_stall(self, team: Team) -> Phase:
        plan = _Plan(outcome="shot_clock")
        for _ in range(20):
            s = self.holder
            if s is None:
                return self._phase_loose_ball(team)
            self._dribble(s)
            yield from self._probe(float(self.rng.uniform(1.5, 2.5)), plan)
            recv = self._choose(team, {"G": 1.0, "F": 1.0, "C": 0.5}, exclude=(s,))
            nxt = yield from self._act_pass(recv, "normal")
            if nxt is not None:
                return nxt
        return self._phase_forced_shot()

    def _foul_on(self, s: int) -> Phase:
        team = self.slot_team[s]
        d = min(self._slots(team.other()), key=lambda q: _dist(self.pos[q], self.pos[s]))
        self._hold(s)
        fouler = self._pid(d)
        self._emit(EventKind.FOUL, team=team.other(), player=fouler, other=self._pid(s), x=float(self.pos[s, 0]),
                   y=float(self.pos[s, 1]), attrs={"type": "shooting", "n_ft": 2})
        self._tally(fouler, pf=1)
        self.fouls[int(self.occ[d])] += 1
        return self._phase_dead_ball(team, None, "other", ft_shooter=s)

    # ------------------------------------------------------------------ shots

    def _act_shoot(self, s: int, shot_type: str, gather_xy: np.ndarray | None = None,
                   release_target: np.ndarray | None = None, gather_s: float = 0.3, tag: str = "other") -> Phase:
        """Gather (ball rises to the release height), release a projectile, play the shot out."""
        c = self.cfg
        yield from self._pickup()
        b = self.ball
        self._hold(s, z=b.held_z if b.mode == "held" and b.holder == s else None)  # a tip keeps its catch height
        self.locked[s] = True
        gxy = self.pos[s].copy() if gather_xy is None else gather_xy
        self.target[s] = self.pos[s].copy() if release_target is None else release_target
        if release_target is None:
            self.vmax[s] = 3.0
        lo, hi = {"jumper": c.jumper_release_z_ft, "hook": c.hook_release_z_ft}.get(shot_type, c.layup_release_z_ft)
        if shot_type == "dunk":
            lo = max(lo, 10.0)
        z_rel = float(self.rng.uniform(lo, max(lo, hi)))
        side = self.attack[self.slot_team[s]]
        z_rel = max(z_rel, self._min_release_z(self.target[s], side))
        z0, t0 = self.ball.held_z, self.t

        def gathering() -> bool:  # gather time, extended (<= 0.5 s) until a layup reaches its release spot
            if self.t - t0 < gather_s - 1e-9:
                return True
            late = self.t - t0 < gather_s + 0.5
            return release_target is not None and late and _dist(self.pos[s], release_target) > 0.4

        while gathering():  # the ball reaches z_rel and the release point on the release frame
            self._shooting_motion(s, z0, z_rel, min(1.0, (self.t + self.dt - t0) / gather_s))
            yield
        self.ball.held_z = z_rel
        shot = self._release(s, shot_type, gxy, z_rel, tag=tag)
        result = yield from self._shot_flight(shot)
        self.locked[s] = False
        return self._after_shot(shot, result)

    def _shot_bin(self, dist: float, three: bool, zone: str) -> str:
        if dist > 40.0:
            return "heave"
        if three:
            return "c3" if zone == "corner_three" else "ab3"
        if dist < 3.0:
            return "0_3"
        if dist < 10.0:
            return "3_10"
        return "10_16" if dist < 16.0 else "16_3p"

    def _release(self, s: int, shot_type: str, gather_xy: np.ndarray, z_rel: float, is_ft: bool = False,
                 tag: str = "free_throw") -> _Shot:
        c, rng, court = self.cfg, self.rng, self.court
        team = self.slot_team[s]
        side = self.attack[team]
        pid = self._pid(s)
        pos = self.pos[s].copy()
        basket = self._basket(team)
        dist = court.distance_to_basket(float(pos[0]), float(pos[1]), side)
        gx, gy = float(gather_xy[0]), float(gather_xy[1])
        value = 1 if is_ft else (3 if court.is_three(gx, gy, side) else 2)
        zone = court.zone(gx, gy, side)
        defenders = self.pos[list(self._slots(team.other()))]
        closest = float(np.min(np.hypot(*(defenders - pos).T)))
        bkey = self._shot_bin(dist, value == 3, zone)
        if is_ft:
            p_make, p_block = float(self.ft_skill[int(self.occ[s])]), 0.0
        else:
            adj = c.defender_slope[bkey] * (closest - c.defender_ref_ft[bkey])
            p_make = float(np.clip(c.fg_pct[bkey] + np.clip(adj, -0.1, 0.1), 0.01, 0.97))
            p_block = c.block_rate[bkey] * (1.3 if closest < 3.0 else 0.6) if closest < c.block_max_defender_ft else 0.0
        blocked = (not is_ft) and rng.random() < p_block
        made = (not blocked) and rng.random() < min(0.98, p_make / max(1e-6, 1.0 - p_block))
        self.n_shots += 1
        shot = _Shot(shot_id=f"s{self.n_shots:04d}", slot=s, pid=pid, team=team, side=side, value=value,
                     shot_type="free_throw" if is_ft else shot_type, made=made,
                     blocked_by=self._closest(team.other(), pos) if blocked else None,
                     is_ft=is_ft, release_t=self.t, x=float(pos[0]), y=float(pos[1]))
        shot.team_rebound_oob = (not made) and (not blocked) and (not is_ft) and rng.random() < c.team_rebound_rate
        catch_shoot = (self.spell_from is not None and self.spell_dribbles == 0 and shot_type == "jumper")
        attrs = {"shot_type": shot.shot_type, "distance_ft": round(dist, 3), "closest_defender_ft": round(closest, 3),
                 "catch_and_shoot": bool(catch_shoot), "dribbles_before": int(self.spell_dribbles),
                 "basket": side, "zone": zone, "xfg": round(p_make, 4), "gather_x": gx, "gather_y": gy,
                 "sim_play": tag}
        self._emit(EventKind.SHOT_ATTEMPT, team=team, player=pid, x=shot.x, y=shot.y, value=value,
                   shot_id=shot.shot_id, attrs=attrs)
        if is_ft:
            self._tally(pid, fta=1)
        else:
            self._tally(pid, fga=1, fg3a=1 if value == 3 else 0)
        self._launch_shot(shot, pos, basket, z_rel)
        self.shot_pending = shot
        self.ball_live = not is_ft or self.last_ft
        return shot

    def _launch_shot(self, shot: _Shot, pos: np.ndarray, basket: np.ndarray, z_rel: float) -> None:
        c, rng = self.cfg, self.rng
        u = self._to_basket_dir(pos, shot.side)
        p0 = np.append(self._release_xy(pos, shot.side), max(z_rel, self._min_release_z(pos, shot.side)))
        rim = np.append(basket, self.rim_h)
        if shot.made:
            jitter = _rot(np.array([1.0, 0.0]), float(rng.uniform(0, 2 * math.pi))) * float(
                rng.uniform(0.0, 0.6)) * (self.court.rim_radius - self.ball.r)
            p1 = rim + np.append(jitter, 0.0)
        else:
            p1 = self._rim_contact_point(shot, u)
        lo, hi = _ENTRY_ANGLE_DEG.get(shot.shot_type, (40.0, 50.0))
        theta = math.radians(float(rng.uniform(lo, hi)))
        T = self._flight_time(p0, p1, theta)
        self.ball.pos[:] = p0
        self._launch(p1, T, None)
        if shot.made:  # exact in time (like rim contacts): never missed between two frames at low fps
            v_in = self.ball.vel - np.array([0.0, 0.0, GRAVITY_FTS2 * T])
            net = np.array([0.15 * v_in[0], 0.15 * v_in[1], max(0.5 * v_in[2], -8.0)])  # the net catches it
            self.ball.impulses.append((T, net, "make"))
        if shot.blocked_by is not None:
            tb = float(rng.uniform(0.07, 0.14))
            tb = min(tb, 0.5 * T)
            back = _rot(-u, math.radians(float(rng.uniform(-60, 60)))) * float(rng.uniform(8.0, 15.0))
            self.ball.impulses.append((tb, np.append(back, float(rng.uniform(-4.0, 6.0))), "block"))
            self.target[shot.blocked_by] = pos + u * 1.5
            self.auto[shot.blocked_by] = False
        elif not shot.made:
            self.ball.impulses.append((T, self._deflection(shot, p1), "rim"))

    def _flight_time(self, p0: np.ndarray, p1: np.ndarray, theta: float) -> float:
        """Flight time that reaches ``p1`` descending at ``theta`` (at least 0.05 ft of drop, so the ball
        always arrives from above; also safe when ``p1`` is straight below ``p0``)."""
        d = math.hypot(*(p1[:2] - p0[:2]))
        dz = float(p1[2] - p0[2])
        return math.sqrt(2.0 * max(dz + d * math.tan(theta), 0.05) / GRAVITY_FTS2)

    def _to_basket_dir(self, pos: np.ndarray, side: Side) -> np.ndarray:
        basket = np.array(self.court.basket_xy(side))
        to_b = basket - pos
        if math.hypot(*to_b) > 1e-6:
            return _unit(to_b)
        return -_unit(self._spot(side, 1.0, 0.0) - basket)

    def _release_xy(self, pos: np.ndarray, side: Side) -> np.ndarray:
        """Release point: just in front of the shooter's floor position, toward the rim."""
        d_floor = self.court.distance_to_basket(float(pos[0]), float(pos[1]), side)
        return pos + self._to_basket_dir(pos, side) * min(0.5, 0.4 * d_floor)

    def _min_release_z(self, pos: np.ndarray, side: Side) -> float:
        """A ball released inside the rim cylinder must start above the rim (it cannot go up through
        the net); elsewhere no constraint."""
        d = _dist(self._release_xy(pos, side), np.array(self.court.basket_xy(side)))
        return self.rim_h + self.ball.r + 0.05 if d < self.court.rim_radius + self.ball.r else 0.0

    def _rim_contact_point(self, shot: _Shot, u: np.ndarray) -> np.ndarray:
        r = self.rng.random()
        ang = float(self.rng.uniform(-0.7, 0.7))
        if r < 0.5:
            direction = _rot(-u, ang)  # front rim (toward the shooter)
        elif r < 0.8:
            direction = _rot(-u, math.copysign(math.pi / 2, ang) + ang * 0.3)  # side rim
        else:
            direction = _rot(u, ang * 0.6)  # back rim
        basket = np.array(self.court.basket_xy(shot.side))
        # radius rim_r + 0.1 keeps a back-rim contact clear of the board face (1.25 ft behind the centre)
        xy = basket + direction * (self.court.rim_radius + 0.1)
        return np.append(xy, self.rim_h + 0.25)

    def _deflection(self, shot: _Shot, contact: np.ndarray) -> np.ndarray:
        """Rebound velocity off the rim that never drops back through the hoop or into the board."""
        rng, g = self.rng, GRAVITY_FTS2
        basket = np.array(self.court.basket_xy(shot.side))
        out = _unit(self._spot(shot.side, 1.0, 0.0) - basket)
        if shot.team_rebound_oob:
            lat = math.copysign(1.0, float(contact[1] - basket[1]) or float(rng.choice([-1.0, 1.0])))
            return np.array([out[0] * 2.0, lat * float(rng.uniform(17.0, 21.0)), float(rng.uniform(3.0, 6.0))])
        if shot.is_ft:
            sp_rng, vz_rng = (3.0, 7.0), (3.0, 8.0)
        elif shot.value == 3:
            sp_rng, vz_rng = (5.0, 11.0), (4.0, 10.0)
        else:
            sp_rng, vz_rng = (3.5, 8.5), (4.0, 10.0)
        dz0 = float(contact[2]) - self.rim_h
        for _ in range(40):
            direction = _rot(out, math.radians(float(rng.uniform(-80.0, 80.0))))
            sp, vz = float(rng.uniform(*sp_rng)), float(rng.uniform(*vz_rng))
            tau = (vz + math.sqrt(vz * vz + 2.0 * g * dz0)) / g
            land = contact[:2] + direction * sp * tau
            if _dist(land, basket) > self.court.rim_radius + self.ball.r + 0.3:
                return np.append(direction * sp, vz)
        return np.append(out * 7.0, 6.0)

    def _left_rim_zone(self, shot: _Shot) -> bool:
        b = self.ball
        basket = np.array(self.court.basket_xy(shot.side))
        return _dist(b.pos, basket) > self.court.rim_radius + 1.5 or b.pos[2] < self.rim_h - 1.0

    def _shot_flight(self, shot: _Shot) -> Generator[None, None, str]:
        """Play the shot out: returns 'made' | 'missed' | 'blocked' (events emitted here)."""
        b = self.ball
        while True:
            if "block" in b.fired and shot.blocked_by is not None:
                d = shot.blocked_by
                bx, by_ = float(b.pos[0]), float(b.pos[1])
                self._emit(EventKind.BLOCK, team=shot.team.other(), player=self._pid(d), other=shot.pid, x=bx, y=by_,
                           shot_id=shot.shot_id)
                self._tally(self._pid(d), blk=1)
                self._emit_result(shot, made=False)
                self.last_touch = shot.team.other()
                return "blocked"
            if "make" in b.fired and shot.made:
                self._emit_result(shot, made=True)
                return "made"
            if "rim" in b.fired:
                shot.rim_touched = True
                self.sc_running = False
                self.sc_expired = False
            if shot.rim_touched and self._left_rim_zone(shot):
                self._emit_result(shot, made=False)
                return "missed"
            if b.mode != "flight":  # safety: should not happen
                self._emit_result(shot, made=False)
                return "missed"
            yield

    def _emit_result(self, shot: _Shot, made: bool) -> None:
        self.shot_pending = None
        kind = EventKind.SHOT_MADE if made else EventKind.SHOT_MISSED
        self._emit(kind, team=shot.team, player=shot.pid, x=shot.x, y=shot.y, value=shot.value if made else 0,
                   shot_id=shot.shot_id, attrs={"basket": shot.side})
        if not made:
            if not shot.is_ft or self.last_ft:
                self.loose_after = shot
            return
        self.score[shot.team] += shot.value
        if shot.is_ft:
            self._tally(shot.pid, ftm=1, pts=1)
            return
        self._tally(shot.pid, fgm=1, fg3m=1 if shot.value == 3 else 0, pts=shot.value)
        self._maybe_assist(shot)
        if self.clock_running and self.rules.clock_stops_after_fg_at(self.period, self.game_clock):
            self.clock_running = False
        self.sc_running = False

    def _maybe_assist(self, shot: _Shot) -> None:
        c = self.cfg
        passer = self.spell_from
        if passer is None:
            return
        if shot.release_t - self.spell_start > c.assist_window_s + 1e-9 or self.spell_dribbles > c.assist_max_dribbles:
            return
        self._emit(EventKind.ASSIST, team=shot.team, player=passer, other=shot.pid, shot_id=shot.shot_id)
        self._tally(passer, ast=1)

    def _after_shot(self, shot: _Shot, result: str) -> Phase | None:
        if self.period_expired:
            self.loose_after = None
            self.period_expired = False
            return self._phase_period_end()
        if result == "made":
            self.ball_live = False
            return self._phase_inbound(shot.team.other(), ("baseline", None), "made_shot")
        if result == "blocked" and self.sc_expired:
            self.loose_after = None
            return self._shot_clock_violation()
        return self._phase_rebound(shot, after_block=result == "blocked")

    # ------------------------------------------------------------------ rebounds & loose balls

    def _landing_point(self) -> np.ndarray:
        b = self.ball
        z0, vz = float(b.pos[2]), float(b.vel[2])
        target = 7.5 if z0 > 7.5 or vz > 0 else b.r
        disc = vz * vz + 2.0 * GRAVITY_FTS2 * (z0 - target)
        tau = (vz + math.sqrt(max(disc, 0.0))) / GRAVITY_FTS2
        return b.pos[:2] + b.vel[:2] * tau

    def _phase_rebound(self, shot: _Shot, after_block: bool = False) -> Phase:
        c, rng = self.cfg, self.rng
        self.loose_after = shot
        self.ball_live = True
        self.motion_on = False
        self.def_auto_on = False
        self.auto[:] = False
        self.locked[:] = False
        p_off = c.oreb_rate_ft if shot.is_ft else (c.oreb_rate_block if after_block else c.oreb_rate)
        land = self._landing_point()
        rebounder: int | None = None
        if not shot.team_rebound_oob:
            team_r = shot.team if rng.random() < p_off else shot.team.other()
            rebounder = self._choose_rebounder(team_r, land)
        self._crash_boards(shot, land, rebounder)
        start = self.t
        while True:
            if rebounder is not None:
                if self._reach(rebounder, 2.5, 9.5):
                    break
                if self.ball.pos[2] < 8.0 or self.t - start > 0.8:
                    lead = min(1.2, _dist(self.pos[rebounder], self.ball.pos) / self.cfg.sprint_speed_fts)
                    self.target[rebounder] = self.ball.pos[:2] + self.ball.vel[:2] * lead
            if self.t - start > 5.0:
                near = min(range(10), key=lambda q: _dist(self.pos[q], self.ball.pos))
                rebounder = near
                for q in range(10):
                    self.target[q] = self.ball.pos[:2]
                if self._reach(near, 2.0, 9.0):
                    break
            yield
        return self._rebound_control(shot, rebounder)

    def _choose_rebounder(self, team: Team, land: np.ndarray) -> int:
        w = []
        slots = list(self._slots(team))
        for s in slots:
            role_w = {"C": 1.6, "F": 1.2, "G": 0.7}[self.role[self.occ[s]]]
            w.append(role_w * math.exp(-_dist(self.pos[s], land) / 6.0))
        w_arr = np.array(w)
        return slots[int(self.rng.choice(len(slots), p=w_arr / w_arr.sum()))]

    def _crash_boards(self, shot: _Shot, land: np.ndarray, rebounder: int | None) -> None:
        c = self.cfg
        basket = np.array(self.court.basket_xy(shot.side))
        for s in range(10):
            if s == rebounder:
                self.target[s] = land
                self.vmax[s] = c.sprint_speed_fts
                continue
            off = self.slot_team[s] is shot.team
            if off and self.role[self.occ[s]] == "G":
                self.target[s] = self._spot(shot.side, 28.0, float(self.rng.uniform(-10, 10)))  # get back
            elif off:
                self.target[s] = land + _rot(np.array([4.0, 0.0]), float(self.rng.uniform(0, 2 * math.pi)))
            else:
                self.target[s] = basket + _unit(self.pos[s] - basket) * float(self.rng.uniform(5.0, 9.0))
            self.vmax[s] = c.run_speed_fts

    def _rebound_control(self, shot: _Shot, s: int) -> Phase:
        """REBOUND by slot ``s`` now; returns the next phase."""
        team = self.slot_team[s]
        offensive = team is shot.team
        pid = self._pid(s)
        self.loose_after = None
        self._hold(s)
        self._start_spell(None)
        self._emit(EventKind.REBOUND, team=team, player=pid, x=float(self.pos[s, 0]), y=float(self.pos[s, 1]),
                   attrs={"offensive": offensive})
        self._tally(pid, oreb=1 if offensive else 0, dreb=0 if offensive else 1)
        if not self.clock_running and self.game_clock > 0:
            self.clock_running = True  # clock starts on the touch after a missed final FT
        if offensive:
            if shot.rim_touched:
                self._reset_shot_clock(self.rules.shot_clock_reset_s)
            self.sc_running = self.sc is not None
            d = _dist(self.pos[s], self._basket(team))
            if d < 8.0 and self.rng.random() < self.cfg.putback_rate:
                self.vmax[s] = 6.0
                return self._phase_putback(s)
            return self._phase_half_court(team, after_oreb=True)
        self._gain_after_rebound(s)
        return self._continue_with_ball(team)

    def _gain_after_rebound(self, s: int) -> None:
        team = self.slot_team[s]
        self.possession_team = team
        self._emit(EventKind.POSSESSION, team=team, player=self._pid(s), x=float(self.pos[s, 0]),
                   y=float(self.pos[s, 1]), attrs={"cause": "defensive_rebound"})
        self._reset_shot_clock(self.rules.shot_clock_s)
        self.sc_running = self.sc is not None

    def _phase_putback(self, s: int) -> Phase:
        team = self.slot_team[s]
        release = self._release_spot(team, s, "0_3")
        quick = self.ball.pos[2] > 7.0  # controlled high (the ball is still where it was caught): tip it
        if quick:
            self._hold(s, z=float(self.ball.pos[2]))
        return (yield from self._act_shoot(s, "tip" if quick else "layup", gather_xy=self.pos[s].copy(),
                                           release_target=release, gather_s=0.2 if quick else 0.4, tag="putback"))

    def _phase_loose_ball(self, favored: Team, steal_by: int | None = None, victim: str | None = None,
                          cause: str = "other") -> Phase:
        """A loose ball outside the scripted flow: the nearest ``favored`` player recovers it."""
        self.ball_live = True
        self.auto[:] = False
        s = steal_by if steal_by is not None else self._closest(favored, self.ball.pos[:2])
        start = self.t
        while not (self._reach(s, 2.0, 7.0) and self.ball.mode in ("flight", "rest")):
            self.target[s] = np.clip(self.ball.pos[:2], -2.0, [self.court.length + 2.0, self.court.width + 2.0])
            self.vmax[s] = self.cfg.sprint_speed_fts
            if self.ball.mode in ("held", "dribble") and self.holder is not None:
                s = self.holder
                break
            if self.t - start > 6.0:
                break
            yield
        team = self.slot_team[s]
        if not self.clock_running and self.game_clock > 0.0:  # e.g. a missed throw-in: touched inbounds now
            self.clock_running = True
            self.sc_running = self.sc is not None
        if victim is not None and team is not self.possession_team:
            self._steal(s, victim, "steal")
        else:
            self._gain(s, cause)
        return self._continue_with_ball(team)

    # ------------------------------------------------------------------ dead balls

    def _out_of_bounds(self, point: np.ndarray) -> Phase:
        """Ball out: team rebound or turnover now; returns the dead-ball phase."""
        self.clock_running = self.sc_running = False
        self.ball_live = False
        L = self.court.length
        spot: tuple[str, Any] = ("baseline", point) if (point[0] < 0.0 or point[0] > L) else ("sideline", point)
        shot = self.loose_after
        if shot is not None:
            self.loose_after = None
            inbound = (self.last_touch or shot.team).other()
            offensive = inbound is shot.team
            self._emit(EventKind.REBOUND, team=inbound, player=None, x=float(point[0]), y=float(point[1]),
                       attrs={"offensive": offensive})
            tb = self.team_box[inbound]
            tb["team_reb"] += 1
            tb["team_oreb" if offensive else "team_dreb"] += 1
            if offensive and shot.rim_touched:
                self._reset_shot_clock(self.rules.shot_clock_reset_s)
            if spot[0] == "baseline" and offensive:
                spot = ("sideline", np.array([point[0], point[1]]))
            return self._phase_dead_ball(inbound, spot, "defensive_rebound" if not offensive else "other")
        last = self.pass_info.passer if self.pass_info is not None else (
            self._pid(self.holder) if self.holder is not None else None)
        team = self.possession_team or self.last_touch or Team.HOME
        self._emit_pass(False)
        if last is None:
            last = self.last_holder_pid
        self._emit(EventKind.TURNOVER, team=team, player=last, x=float(point[0]), y=float(point[1]),
                   attrs={"type": "out_of_bounds"})
        if last is not None:
            self._tally(last, tov=1)
        else:
            self.team_box[team]["team_tov"] += 1
        return self._phase_dead_ball(team.other(), spot, "turnover")

    def _shot_clock_violation(self) -> Phase:
        """Shot-clock TURNOVER (last holder, per ARCHITECTURE) now; returns the dead-ball phase."""
        team = self.possession_team or Team.HOME
        pid = self._pid(self.holder) if self.holder is not None else self.last_holder_pid
        x, y = float(self.ball.pos[0]), float(self.ball.pos[1])
        self._emit_pass(False)
        self._emit(EventKind.TURNOVER, team=team, player=pid, x=x, y=y, attrs={"type": "shot_clock"})
        if pid is not None:
            self._tally(pid, tov=1)
        else:
            self.team_box[team]["team_tov"] += 1
        return self._phase_dead_ball(team.other(), ("sideline", np.array([x, y])), "turnover")

    def _phase_dead_ball(self, next_team: Team, spot: tuple[str, Any] | None, cause: str,
                         ft_shooter: int | None = None) -> Phase:
        """Whistle: clock stops; maybe a timeout and substitutions; then FTs or the inbound."""
        self.clock_running = self.sc_running = False
        self.ball_live = False
        self.motion_on = False
        for s in range(10):
            self.vmax[s] = self.cfg.walk_speed_fts
            self.target[s] = self.pos[s].copy()
        self.auto[:] = False
        yield from self._pickup()
        if self.ball.mode == "held" and self.ball.holder is not None:
            self._hold(self.ball.holder)  # ends a shooting motion cut short by the whistle
        yield from self._wait(float(self.rng.uniform(0.4, 0.8)))
        if self.rng.random() < self.cfg.timeout_prob:
            team = next_team if self.rng.random() < 0.6 else next_team.other()
            if self.timeouts_left[team] > 0:
                yield from self._timeout(team, exclude=(ft_shooter,) if ft_shooter is not None else ())
        exclude = (ft_shooter,) if ft_shooter is not None else ()
        if ft_shooter is None and self.holder is not None:
            exclude = (self.holder,)
        self._maybe_subs(self.cfg.sub_prob, exclude)
        if ft_shooter is not None:
            return self._phase_free_throws(ft_shooter, 2)
        assert spot is not None
        return self._phase_inbound(next_team, spot, cause)

    def _timeout(self, team: Team, exclude: tuple[int, ...]) -> Phase:
        self.timeouts_left[team] -= 1
        self._emit(EventKind.TIMEOUT, team=team, attrs={"length_video_s": self.cfg.timeout_video_s})
        self._walk_to_benches()
        yield from self._wait(0.5)
        self._maybe_subs(0.7, exclude)
        yield from self._wait(max(0.0, self.cfg.timeout_video_s - 0.5))

    # ------------------------------------------------------------------ free throws

    def _ft_formation(self, shooter: int) -> None:
        team = self.slot_team[shooter]
        side = self.attack[team]
        hw = self.court.lane_width / 2.0 + 0.9
        ft_along = self.court.free_throw_line_from_baseline - self.court.basket_offset
        self.target[shooter] = self._spot(side, ft_along + 0.6, 0.0)
        offense = [s for s in self._slots(team) if s != shooter]
        defense = list(self._slots(team.other()))
        defense.sort(key=lambda s: {"C": 0, "F": 1, "G": 2}[self.role[self.occ[s]]])
        offense.sort(key=lambda s: {"C": 0, "F": 1, "G": 2}[self.role[self.occ[s]]])
        spots_d = [(2.5, -hw), (2.5, hw), (8.8, -hw), (25.0, 12.0), (25.0, -12.0)]
        spots_o = [(5.7, -hw), (5.7, hw), (26.0, 6.0), (26.0, -6.0)]
        for s, (a, lat) in zip(defense, spots_d, strict=True):
            self.target[s] = self._spot(side, a, lat)
        for s, (a, lat) in zip(offense, spots_o, strict=True):
            self.target[s] = self._spot(side, a, lat)
        self._set_speed(range(10), self.cfg.jog_speed_fts)
        self.locked[:] = True

    def _phase_free_throws(self, shooter: int, n: int) -> Phase:
        c = self.cfg
        team = self.slot_team[shooter]
        self._ft_formation(shooter)
        self.auto[:] = False
        self.motion_on = self.def_auto_on = False
        spot = self.target[shooter].copy()
        if self.ball.mode not in ("held", "dribble") or self.ball.holder != shooter:
            yield from self._wait_until(lambda: _dist(self.pos[shooter], spot) < 4.0, 6.0)
            self._toss_to(shooter)
            res = yield from self._await_catch()
            if res != "caught":
                self._hold(shooter)
        self._hold(shooter)
        for k in range(1, n + 1):
            self.last_ft = k == n
            yield from self._wait_until(lambda: not self.sub_tasks and self._arrived(range(10), 0.6), 15.0)
            have_ball = self.t
            yield from self._wait_until(lambda: self.t - self.still_since[shooter] >= c.ft_set_s and
                                        self.t - have_ball >= 0.6, 6.0)
            yield from self._wait(float(self.rng.uniform(0.2, 0.6)))
            self._hold(shooter)
            self._start_spell(None)
            z = float(self.rng.uniform(*c.ft_release_z_ft))
            yield from self._raise_ball(shooter, z, c.ft_gather_s)  # shooting motion: no jump at the release
            shot = self._release(shooter, "free_throw", self.pos[shooter].copy(), z, is_ft=True)
            result = yield from self._shot_flight(shot)
            if k < n:
                b = self.ball
                yield from self._wait_until(lambda: b.bounced or b.rolling or b.pos[2] < 1.0, 2.0)
                self.ball_live = False
                self.loose_after = None
                self._toss_to(shooter)
                res = yield from self._await_catch()
                if res != "caught":
                    self._hold(shooter)
                continue
            self.last_ft = True
            self.locked[:] = False
            if result == "made":
                self.ball_live = False
                return self._phase_inbound(team.other(), ("baseline", None), "free_throw")
            self.ball_live = True
            return self._phase_rebound(shot)
        return None

    # ------------------------------------------------------------------ substitutions

    def _maybe_subs(self, prob: float, exclude: tuple[int, ...]) -> None:
        c = self.cfg
        for team in _TEAMS:
            forced = [s for s in self._slots(team) if self.fouls[int(self.occ[s])] >= self.rules.foul_out]
            if not forced and self.rng.random() > prob:
                continue
            busy = {t.slot for t in self.sub_tasks}
            tired = sorted((s for s in self._slots(team) if s not in exclude and s not in busy and
                            self.stint[int(self.occ[s])] >= c.min_stint_s), key=lambda s: -self.stint[int(self.occ[s])])
            outs = [s for s in forced if s not in exclude and s not in busy] + [s for s in tired if s not in forced]
            for s in outs[: 1 + int(self.rng.random() < 0.4)]:
                inn = self._bench_pick(team, int(self.occ[s]))
                if inn is None:
                    continue
                mid = self.court.length / 2.0
                spot = np.array([mid + (-3.0 if team is Team.HOME else 3.0), -1.5])
                self.sub_tasks.append(_SubTask(team, s, int(self.occ[s]), inn, spot, self.t))
                self.override[s] = spot

    def _bench_pick(self, team: Team, out_idx: int) -> int | None:
        on = set(int(i) for i in self.occ)
        pending = {t.in_idx for t in self.sub_tasks}
        bench = [i for i in self.team_idx[team] if i not in on and i not in pending and
                 self.fouls[i] < self.rules.foul_out and self.rest[i] >= self.cfg.min_rest_s]
        if not bench:
            return None
        same = [i for i in bench if self.role[i] == self.role[out_idx]]
        pool = same or bench
        return max(pool, key=lambda i: (self.rest[i], -i))

    def _run_subs(self) -> None:
        for task in list(self.sub_tasks):
            s = task.slot
            if _dist(self.pos[s], task.spot) > 1.5 and self.t - task.start_t < 4.0:
                self.vmax[s] = max(self.vmax[s], self.cfg.jog_speed_fts)
                continue
            out_pid, in_pid = self.pid[task.out_idx], self.pid[task.in_idx]
            self.occ[s] = task.in_idx
            self.stint[task.in_idx] = 0.0
            self.rest[task.out_idx] = 0.0
            self.override[s] = None
            self.pos[s] = task.spot + np.array([0.0, -1.0])
            self.vel[s] = 0.0
            if self.ball.holder == s and self.ball.mode in ("held", "dribble"):
                self._hold(s)  # the outgoing player hands the ball over (carried, no jump)
            self.sub_tasks.remove(task)
            self._emit(EventKind.SUBSTITUTION, team=task.team, player=in_pid, other=out_pid,
                       x=float(self.pos[s, 0]), y=float(self.pos[s, 1]))
            self.subs_log.append({"t": self.t, "frame_idx": self.frame, "team": task.team.value, "in": in_pid,
                                  "out": out_pid})

    # ------------------------------------------------------------------ AI targets

    def _ai_targets(self) -> None:
        off = self.possession_team
        if self.def_auto_on:
            self._defense_targets(self.ai_offense)
        if off is not None and self.motion_on and self.t >= self.next_relocate:
            self._relocate(off)
        self._ref_targets()

    def _defense_targets(self, off: Team) -> None:
        B = self._basket(off)
        if self.ball.mode != "flight":  # defenders react to where the ball is held, not to passes in the air
            self.ai_ball_xy = self.ball.pos[:2].copy()
        ball_xy = self.ai_ball_xy
        L, W = self.court.length, self.court.width
        for s in self._slots(off):
            d = self._opp(s)
            if not self.auto[d]:
                continue
            m = self.pos[s]
            to_b = B - m
            dist_b = math.hypot(*to_b)
            u = to_b / dist_b if dist_b > 1e-6 else np.zeros(2)
            dball = _dist(m, ball_xy)
            gap = 3.2 if s == self.holder else float(np.clip(4.0 + 0.25 * dball, 4.0, 11.0))
            gap = min(gap, 0.6 * dist_b)
            tgt = m + u * gap
            if s != self.holder:  # sag toward the ball (help side)
                tgt = tgt + _unit(ball_xy - m) * min(3.0, 0.1 * dball)
            self.target[d] = (float(np.clip(tgt[0], 0.5, L - 0.5)), float(np.clip(tgt[1], 0.5, W - 0.5)))

    def _relocate(self, off: Team) -> None:
        self.next_relocate = self.t + float(self.rng.uniform(*self.cfg.relocate_s))
        free = [s for s in self._slots(off) if s != self.holder and not self.locked[s]]
        if not free:
            return
        s = free[int(self.rng.integers(len(free)))]
        side = self.attack[off]
        options = [(25.5, 0.0), (24.5, 11.0), (24.5, -11.0), (17.5, 20.5), (17.5, -20.5), (3.0, 23.2), (3.0, -23.2),
                   (13.5, 8.5), (13.5, -8.5), (4.0, 8.5), (4.0, -8.5)]
        taken = [self.target[q] for q in self._slots(off) if q != s]
        ok = [self._spot(side, a, lat) for a, lat in options]
        ok = [p for p in ok if all(_dist(p, q) > 7.0 for q in taken)]
        if ok:
            self.target[s] = ok[int(self.rng.integers(len(ok)))]

    def _ref_targets(self) -> None:
        team = self.possession_team or Team.HOME
        side = self.attack[team]
        sign = 1.0 if side == "right" else -1.0
        L, W = self.court.length, self.court.width
        bx, by = float(self.ball.pos[0]), float(self.ball.pos[1])
        self.ref_target[0] = (float(np.clip(bx - sign * 12.0, 4.0, L - 4.0)), -2.5)
        self.ref_target[1] = (float(np.clip(bx - sign * 3.0, 4.0, L - 4.0)), W + 2.5)
        self.ref_target[2] = (L + 2.0 if side == "right" else -2.0, float(np.clip(by, 12.0, W - 12.0)))

    # ------------------------------------------------------------------ recording & output

    def _alloc(self, n: int) -> None:
        self.r_pos = np.zeros((n, 10, 2))
        self.r_vel = np.zeros((n, 10, 2))
        self.r_occ = np.zeros((n, 10), dtype=np.int32)
        self.r_ball = np.zeros((n, 3))
        self.r_refs = np.zeros((n, 3, 2))
        self.r_score = np.zeros((n, 2), dtype=np.int32)
        self.r_period = np.zeros(n, dtype=np.int32)
        self.r_gc = np.zeros(n)
        self.r_sc = np.full(n, np.nan)
        self.r_holder = np.full(n, -1, dtype=np.int32)

    def _record(self, i: int) -> None:
        self.r_pos[i] = self.pos
        self.r_vel[i] = self.vel
        self.r_occ[i] = self.occ
        self.r_ball[i] = self.ball.pos
        self.r_refs[i] = self.ref_pos
        self.r_score[i] = (self.score[Team.HOME], self.score[Team.AWAY])
        self.r_period[i] = self.period
        self.r_gc[i] = self.game_clock
        if self.sc is not None:
            self.r_sc[i] = self.sc
        if self.ball.mode in ("held", "dribble") and self.ball.holder is not None:
            self.r_holder[i] = self.occ[self.ball.holder]

    def _result(self, n: int) -> SimResult:
        frames = self._build_frames(n)
        events = sorted(self.events, key=lambda e: e.frame_idx)  # stable: emission order within a frame
        for k, ev in enumerate(events, 1):
            ev.event_id = f"t{k:06d}"
        refs = [[(float(a), float(b)) for a, b in fr] for fr in self.r_refs.tolist()]
        holders = [self.pid[h] if h >= 0 else None for h in self.r_holder.tolist()]
        return SimResult(config=self.cfg, court=self.court, roster=self.roster, camera=self.camera, frames=frames,
                         events=events, truth_box={p: dict(v) for p, v in self.box.items()},
                         truth_score=dict(self.score), referees=refs, substitutions=list(self.subs_log),
                         rules=self.rules, team_box={t: dict(v) for t, v in self.team_box.items()}, holders=holders,
                         starters={t: list(v) for t, v in self.starters.items()})

    def _build_frames(self, n: int) -> list[FrameState]:
        c, cam = self.cfg, self.camera
        boxes = cylinder_boxes(cam, self.r_pos.reshape(-1, 2), c.player_height_ft, c.player_radius_ft).reshape(n, 10, 4)
        ball_uv = cam.project(self.r_ball)
        ball_box = sphere_boxes(cam, self.r_ball, self.ball.r)
        in_img = cam.in_image(ball_uv)
        rims = self._rim_observations()
        pos, vel, occ, boxes_l = self.r_pos.tolist(), self.r_vel.tolist(), self.r_occ.tolist(), boxes.tolist()
        bx, buv, bbox = self.r_ball.tolist(), ball_uv.tolist(), ball_box.tolist()
        teams = [self.team_of_idx[i] for i in range(len(self.pid))]
        frames: list[FrameState] = []
        size = (int(c.image_size[0]), int(c.image_size[1]))
        for i in range(n):
            players = []
            for s in range(10):
                idx = occ[i][s]
                b = boxes_l[i][s]
                players.append(PlayerState(
                    player_id=self.pid[idx], team=teams[idx], x=pos[i][s][0], y=pos[i][s][1], vx=vel[i][s][0],
                    vy=vel[i][s][1], jersey=self.jersey[idx], track_id=idx + 1,
                    bbox=BBox(b[0], b[1], b[2], b[3]), confidence=1.0))
            vis = bool(in_img[i])
            r_px = (bbox[i][2] - bbox[i][0]) / 2.0 if vis else None
            ball = BallState(visible=vis, image_xy=(buv[i][0], buv[i][1]) if vis else None,
                             court_xy=(bx[i][0], bx[i][1]), z=bx[i][2], radius_px=r_px, confidence=1.0)
            sc = None if math.isnan(self.r_sc[i]) else float(self.r_sc[i])
            board = ScoreboardReading(t=i / c.fps, home_score=int(self.r_score[i, 0]),
                                      away_score=int(self.r_score[i, 1]), period=int(self.r_period[i]),
                                      game_clock=float(self.r_gc[i]), shot_clock=sc, source="sim", confidence=1.0)
            frames.append(FrameState(frame_idx=i, t=i / c.fps, players=players, ball=ball, rims=list(rims),
                                     scoreboard=board, image_size=size))
        return frames

    def _rim_observations(self) -> list[RimObservation]:
        out = []
        for side in ("left", "right"):
            bx, by = self.court.basket_xy(side)
            box = self.camera.bbox_of_circle(bx, by, self.rim_h, self.court.rim_radius)
            out.append(RimObservation(side=side, bbox=box, confidence=1.0))
        return out
