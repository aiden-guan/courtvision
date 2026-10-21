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
    separation_ft: float = 2.2  # players closer than this push apart (soft steering); tuned
    # Hard non-penetration (FIBA Art. 33.1 cylinder principle: a player may not enter an opponent's
    # cylinder). Centre-to-centre contact distance of two bodies: adult-male torso depth ~0.85-1.0 ft and
    # shoulder breadth ~1.6-1.7 ft (anthropometric survey means), ~1.5 ft averaged over orientations.
    # Applied to every pair after each motion step, locked players included; anthropometric estimate,
    # unverified.
    body_contact_ft: float = 1.5
    # No other player's body centre closer than this to a held / dribbled ball: torso half-depth ~0.5 ft +
    # ball radius 0.39 ft + a forearm between them ~0.9 ft (a defender reaches with the hand; his torso never
    # rests on the ball: pressed onto it for > 1 s it would be a held ball or a foul). Was 1.2 ft, which let a
    # defender's centre sit within 1.5 ft of the ball on 14% of guarded control frames, for up to 5 s. Only
    # the defender making a planned strip may come to strip_clear_ft (his reach). Anthropometric estimate,
    # unverified.
    ball_clear_ft: float = 1.8
    strip_clear_ft: float = 1.2
    # Ball protection (coaching fundamental: the handler keeps his body between the ball and the defender and
    # dribbles with the hand away from him; FIBA Art. 33 cylinder): when an opponent is within protect_ft, the
    # held / dribbled ball moves to the side away from him, protect_lat_ft out and protect_fwd_ft ahead at full
    # protection (from protect_full_ft in), switching hands at most every hand_switch_s and moving at most
    # hand_move_fts. Unguarded, a held ball sits hand_offset_ft ahead and a dribble is dribble_lat_ft out on the
    # hand he has (his strong hand at a catch; left_hand_share of players are left-handed: general-population
    # ~10%, unverified for basketball players). The ball used to sit 1.0 ft ahead toward the basket, where the
    # on-ball defender stands, and was always dribbled with the right hand. Design choices, unverified.
    protect_ft: float = 5.0
    protect_full_ft: float = 2.5
    protect_lat_ft: float = 1.2
    protect_fwd_ft: float = 0.3
    dribble_lat_ft: float = 0.5
    hand_switch_s: float = 0.5
    hand_move_fts: float = 8.0
    left_hand_share: float = 0.1
    # On-ball defenders shade the handler's strong hand by force_weak_ft ("force him to his weak hand", a
    # defensive principle) on force_weak_share of possessions, so guarded handlers use both hands. Design
    # choices, unverified.
    force_weak_share: float = 0.5
    force_weak_ft: float = 1.0
    # On-ball defender stays about arm's length from the ball handler's centre (arm length ~0.44 x
    # stature ~2.9 ft at 6.5 ft, plus torso half-depth); near the rim he slides laterally instead of
    # standing on the handler. Design choice (anthropometric), unverified.
    on_ball_gap_ft: float = 3.2
    # Off-ball defenders sit help_gap_ft = clip(a + b * distance to the ball, lo, hi) toward the basket from
    # their man (strong side tighter than weak side), sagging up to 3 ft toward the ball. Was clip(4 + 0.25 d,
    # 4, 11), which left catch-and-shoot threes wide open (3PA closest-defender buckets 0-2/2-4/4-6/6+ ft:
    # sim 1/32/18/50% vs NBA.com tracking 1.7/17/41.7/39.4% via CBS Sports, season unstated, unverified).
    help_gap_ft: tuple[float, float, float, float] = (3.0, 0.2, 3.0, 8.0)
    # A pull-up shooter creates separation (step-back / side-step, pullup_step_ft) that the on-ball defender
    # reads late (pullup_react_s); design choices, calibrated with the closeout below to the 3PA buckets.
    pullup_step_ft: tuple[float, float] = (1.5, 3.0)
    pullup_react_s: tuple[float, float] = (0.25, 0.5)
    # Closeout: at a pass release the receiver's defender reads it (closeout_read_s) and, on closeout_share
    # of passes (else a late / no rotation), sprints to a spot closeout_gap_ft short of the catch point on the
    # basket side, holds it closeout_hold_s after the catch (a catch-and-shoot), hands up (closeout_arms_s).
    # With the pull-up separation, 18 full train-seed games (sim only) give 3PA buckets 1.5/19.7/43.9/34.9%
    # (target bands +-8 pts) and corner 3s 51% wide open (> 6 ft; was 78-87%); tuned on sim train seeds,
    # design, unverified.
    closeout_share: float = 0.7
    closeout_read_s: tuple[float, float] = (0.15, 0.3)
    closeout_gap_ft: tuple[float, float] = (4.5, 7.5)
    closeout_hold_s: float = 0.6
    closeout_arms_s: float = 1.0
    strip_reach_ft: float = 2.3  # stripping defender's centre distance from the dribbler (hand reach); unverified
    # Share of strips made by a help / trailing defender (a dig from the side or a poke from behind) instead
    # of the on-ball defender from the basket side (they all used to come from him); design, unverified.
    strip_help_share: float = 0.3
    # After a live-ball steal the robbed player reacts this long (s) before defending (choice reaction
    # ~0.35-0.5 s plus turning round to chase); the stealer pushes the ball in transition at once.
    # Design choice, unverified.
    steal_reaction_s: tuple[float, float] = (0.4, 0.7)
    # Share of strips that are poke-aways: the ball skips low off the poke (first bounce strip_poke_ft away
    # after strip_poke_s), rolls on a few feet and is a loose ball until the first player within reach
    # controls it: the deflector, a teammate (STEAL still to the first deflector, note §2a) or the offence
    # (no steal, no turnover). It used to be a 1-frame holder swap on the dribble spot. Design choices,
    # unverified (no sourced split of strip types or recoveries).
    strip_poke_share: float = 0.5
    strip_poke_ft: tuple[float, float] = (1.5, 3.0)
    strip_poke_s: tuple[float, float] = (0.25, 0.35)
    # A reaching hand is in the person's image box (a detector's person box includes the extended arm): while a
    # player gets a hand to the ball (catching a pass, an interception, a rebound, a block, a strip) his box
    # extends to the hand (the ball, at most arm_length_ratio x stature from the shoulder; a hand_radius_ft
    # sphere) over the last hand_reach_s before the touch (blocks: through the touch). Boxes used to be
    # fixed-radius cylinders, so a blocking / rebounding hand 2-3 ft out fell outside its own box (at-rim
    # blocks: the ball inside the blocker's box on 47% of block frames). Anthropometric (arm ~0.44 x
    # stature), unverified.
    arm_length_ratio: float = 0.44
    hand_radius_ft: float = 0.35
    hand_reach_s: float = 0.15
    # Standing reach / stature ~1.33 (anthropometric ratio); used for raised-ball bbox tops; unverified.
    standing_reach_ratio: float = 1.33
    shoulder_ratio: float = 0.82  # shoulder height / stature (anthropometric); a ball above it is in raised hands
    # Vertical jump heights (ft) by action: NBA combine standing verticals ~2.0-2.8 ft, max ~2.8-3.5 ft;
    # jumpers rise less than layups / dunks. Ranges are design choices bounded by those figures; unverified.
    jump_ft: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "jumper": (0.8, 1.6), "hook": (0.6, 1.2), "layup": (1.4, 2.4), "dunk": (2.2, 3.0), "tip": (1.6, 2.6),
        "rebound": (1.2, 2.6), "block": (1.6, 2.8)})

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
    # A caught / rebounded ball is carried from the catch point to the hand instead of snapping there (keeps
    # the truth trajectory continuous). The held ball moves with the holder's body (a ball caught in the air
    # comes down with him) and the arms bring it to the hand at this speed relative to the body: an arm
    # speed of ~10 ft/s (chinning a rebound: ~3 ft in ~0.3 s). It used to be 30 ft/s with the body ignored,
    # so high rebound catches came down at a median 27.5 ft/s against ~3 ft/s for a body falling from its
    # apex. Unverified.
    carry_speed_fts: float = 10.0
    ft_gather_s: float = 0.35  # the FT shooter raises the ball to the release height over this time; tuned
    dribble_period_s: tuple[float, float] = (0.45, 0.6)  # hand-floor-hand cycle, ~2 Hz (spec); tuned
    dribble_hand_z_ft: float = 3.6  # dribble hand height; tuned
    pass_speed_fts: tuple[float, float] = (30.0, 45.0)  # horizontal pass speed (spec)
    # Pass arcs (design notes, unverified): chest / skip passes peak at most ~10 ft; an overthrown lob
    # at most ~14 ft; no pass crosses the zone above the rim (within 4.5 ft of a basket, above
    # rim height - 1 ft, i.e. rim and backboard): the passer throws harder / flatter, up to 55 ft/s
    # (a hard overhead / baseball pass).
    pass_apex_ft: float = 10.0
    oob_lob_apex_ft: float = 14.0
    pass_rim_zone_ft: float = 4.5
    pass_rim_clear_ft: float = 1.0
    pass_max_speed_fts: float = 55.0
    lazy_pass_speed_fts: tuple[float, float] = (20.0, 26.0)  # intercepted passes; tuned
    # Interceptions are reads after the release, not a defender parked in the lane before the throw (he
    # used to leave his man 0.9 s early and stand 55% along it, a median 1.1 ft off the line at the
    # release): the receiver's defender reacts after intercept_read_s (choice reaction ~0.15-0.3 s) and
    # must reach the lane in time, hand reach intercept_hand_ft, the ball at most intercept_max_z_ft high
    # (a jump above head height). Design choices, unverified.
    intercept_read_s: tuple[float, float] = (0.15, 0.3)
    intercept_hand_ft: float = 3.0
    intercept_max_z_ft: float = 8.5
    # --- make / miss paths (ring contacts, the backboard, air balls)
    # Backboard (rules note §1a, NBA / FIBA, sourced): 6 ft wide (72 in), 3.5 ft tall, its bottom edge 0.5 ft
    # below the ring top (FIBA: the inner rectangle's base is level with the ring top and 150 mm above the
    # board's bottom edge); the face is CourtSpec.backboard_offset from the baseline.
    board_width_ft: float = 6.0
    board_height_ft: float = 3.5
    board_bottom_below_rim_ft: float = 0.5
    # Off the glass: normal restitution and the share of the tangential speed kept; design, unverified.
    board_restitution: tuple[float, float] = (0.45, 0.7)
    board_tangent_keep: tuple[float, float] = (0.75, 0.95)
    # How an unblocked try plays out, as shares of the makes / misses of each shot type ("jumper3" = 3-pt
    # jumpers): swish = through without touching the ring; rim_in = off the ring (1-2 contacts) and in;
    # bank = off the backboard and in; rim_out = one ring contact and out; rim_multi = 2-3 ring contacts and
    # out; in_out = into the cylinder above the ring and back out; board_rim = backboard, ring, out; air = no
    # ring or board contact (an air ball: no 14 s reset, rules note §3). Every make used to be a swish within
    # 0.22 ft of the ring centre and every unblocked miss a single ring contact. A bank needs an angle (the
    # shooter 20-75 deg off the basket axis, 1-18 ft out); otherwise that draw is a rim_in make / rim_out miss.
    # No sourced shares (RESEARCH §2.4, cv_pipeline §6.4 / §11 #10 name rattles, banks, the net and air balls
    # as the make / miss failure modes): design, unverified.
    make_paths: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "dunk": {"swish": 1.0},
        "layup": {"swish": 0.35, "rim_in": 0.2, "bank": 0.45},
        "tip": {"swish": 0.6, "rim_in": 0.4},
        "hook": {"swish": 0.55, "rim_in": 0.3, "bank": 0.15},
        "jumper": {"swish": 0.6, "rim_in": 0.33, "bank": 0.07},
        "jumper3": {"swish": 0.62, "rim_in": 0.36, "bank": 0.02},
        "free_throw": {"swish": 0.6, "rim_in": 0.4}})
    miss_paths: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "dunk": {"rim_out": 1.0},
        "layup": {"rim_out": 0.4, "rim_multi": 0.15, "in_out": 0.1, "board_rim": 0.35},
        "tip": {"rim_out": 0.75, "rim_multi": 0.25},
        "hook": {"rim_out": 0.56, "rim_multi": 0.2, "in_out": 0.1, "board_rim": 0.12, "air": 0.02},
        "jumper": {"rim_out": 0.595, "rim_multi": 0.25, "in_out": 0.07, "board_rim": 0.06, "air": 0.025},
        "jumper3": {"rim_out": 0.61, "rim_multi": 0.25, "in_out": 0.06, "board_rim": 0.04, "air": 0.04},
        "free_throw": {"rim_out": 0.6, "rim_multi": 0.3, "in_out": 0.1}})
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

    # Turnovers per half-court segment. Round 2 (interceptions read after the release, poke-aways loose, bad passes
    # thrown to a covered teammate, closeouts / tighter help): 18 full train-seed games (seeds 1-6 x 10 / 15 /
    # 30 fps, sim only) realise 0.140 turnovers per possession (note §4c .144; possessions = FGA - OREB + TOV +
    # 0.44 FTA, team rebounds / turnovers included) and a live-ball share of 0.58 (.574); tuned on sim train
    # seeds.
    turnover_rate: float = 0.127
    # Turnover types: live-ball (intercept + strip) 0.57 of turnovers (note §4c .574; sourced). The
    # dead-ball split is unverified (no sourced turnover-type shares): overthrown pass out of bounds,
    # a catch fumbled out of bounds, a dribbler stepping out, travels, offensive fouls (charge),
    # shot-clock violations.
    # Planned shares; a bad pass that no defender can read in time is completed and a poke-away the offence
    # recovers is no turnover, so the planned live-ball share (0.63) sits near the realised one (~0.57-0.60
    # on 6 full train-seed games, sim only); tuned on sim train seeds.
    turnover_mix: dict[str, float] = _cfg_dict({
        "intercept": 0.33, "strip": 0.29, "oob": 0.09, "fumble_out": 0.03, "step_out": 0.05, "travel": 0.07,
        "offensive_foul": 0.10, "shot_clock": 0.03})
    after_oreb_scale: float = 0.5  # turnover / foul rates after an offensive rebound are scaled; tuned
    # 2-pt shooting fouls with no make (whistle at the gather of a drive, no FGA) -> 2 FTs, per segment:
    # 17,711 such trips (23,048 2-pt shooting fouls - 5,337 and-ones) / (2,460 team-games x 99.1
    # possessions) = 0.073 per possession (note §4c); derived, then calibrated per segment on the sim.
    foul_rate: float = 0.07
    # 3-pt shooting fouls with no make -> 3 FTs, per 3-pt shot: (0.011 x ~93,300 3-pt actions - 218
    # and-ones) / ~93,300 (note §4a/§4c: P(shooting foul | 3PT) .011, 3PA 37.6 per team-game); derived.
    three_foul_rate: float = 0.0086
    # And-ones per made FG by value: 5,337 / ~69,300 2PM and 218 / ~33,300 3PM (note §4a/§4c: 2P% .545,
    # 3P% .360, 3PAr .421, 3PA 37.6 per team-game); derived.
    and_one_rate: dict[str, float] = _cfg_dict({"2": 0.077, "3": 0.0065})
    # Non-shooting defensive fouls per segment (reach-ins, holds, illegal screens defended): FTs only in
    # the team-foul penalty (RuleSpec.bonus), otherwise a throw-in for the offense. Calibrated on the sim
    # so the share of offensive possessions played in the penalty approaches the note's .225 (§4c);
    # tuned.
    nonshoot_foul_rate: float = 0.08
    # Passes deflected out of bounds by the defence (offence keeps the ball, no turnover, the shot clock
    # keeps its unexpired time) per segment; unverified (no sourced rate).
    deflect_out_rate: float = 0.03
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
    # Re-measured after the body-contact constraint (train seeds 1-6, full games, sim only).
    # Re-measured in round 2 after the closeouts / help gaps / pull-up separation (18 full train-seed games).
    defender_ref_ft: dict[str, float] = _cfg_dict({
        "0_3": 2.35, "3_10": 2.6, "10_16": 4.3, "16_3p": 4.0, "c3": 6.2, "ab3": 5.25, "heave": 3.4})
    defender_slope: dict[str, float] = _cfg_dict({
        "0_3": 0.03, "3_10": 0.02, "10_16": 0.015, "16_3p": 0.015, "c3": 0.012, "ab3": 0.012, "heave": 0.0})
    # Realised P(block | FGA) by bin from the note's pbpstats zone "% blocked" 2024-25 (§4c: at rim
    # .095, short mid 4-14 ft .102, long mid .017, corner 3 .011, arc 3 .008; ~5.5% of FGA), mapped onto
    # the B-Ref distance bins by their FGA shares (3-10 ft: 3-4 ft at-rim + 4-10 ft short mid; 10-16 ft:
    # 10-14 short mid + 14-16 long mid); sourced (zone rates), derived (bin mapping).
    block_rate: dict[str, float] = _cfg_dict({
        "0_3": 0.095, "3_10": 0.100, "10_16": 0.079, "16_3p": 0.017, "c3": 0.011, "ab3": 0.008, "heave": 0.0})
    block_max_defender_ft: float = 6.0  # no block when the closest defender is farther; tuned
    # Closest-defender weighting of the block chance (x1.3 under 3 ft, x0.6 at 3-6 ft, 0 beyond, and 0 when no
    # defender can get a hand to the ball in the block window: _block_reachable), divided
    # by its mean in the bin so the realised block share stays at ``block_rate``: these are the sim's
    # mean weights per bin (train seeds 1-6, full games, sim only); tuned. Re-measured in round 2 (closeouts,
    # blocks need the blocker within reach, help blockers on at-rim tries; 18 full train-seed games; the
    # at-rim bins x the reachable share, 0.98 / 0.94): realised 4.8-5.5% of FGA blocked across runs.
    block_defender_norm: dict[str, float] = _cfg_dict({
        "0_3": 1.2, "3_10": 1.0, "10_16": 0.58, "16_3p": 0.71, "c3": 0.27, "ab3": 0.43, "heave": 0.79})
    # When / by whom (_plan_block): any touch before rim contact while the ball still rises (touching
    # it on its downward flight above the ring is goaltending: FIBA OBR / NBA rule, not yet in the rules note;
    # to verify) and within reach (standing reach + best block jump). Jumpers are contested at the release
    # (block_jumper_s after it); at-rim tries anywhere up to the apex, and a rotating help defender is the
    # blocker in block_help_share of those when he gets there in time from the release (hand reach
    # block_hand_ft). Blocks used to come 0.07-0.14 s after the release, always by the closest
    # defender. Design choices, unverified (rules note §2a: most blocks on rim / short-mid tries; no sourced
    # timing or help share).
    block_jumper_s: tuple[float, float] = (0.07, 0.14)
    block_close_s: float = 0.25  # at-rim tries: the block window used by the reachability pre-check
    block_help_share: float = 0.3
    # Help at the rim (_rotate_help): on rim_help_share of drives (once the driver is within
    # rim_help_trigger_ft of the basket) and cuts, the help defender nearest the basket (within rim_help_ft)
    # rotates to rim_help_spot_ft in front of the rim. Design choices, unverified (no sourced help rate).
    rim_help_share: float = 0.5
    rim_help_trigger_ft: float = 14.0
    rim_help_ft: float = 16.0
    rim_help_spot_ft: tuple[float, float] = (2.0, 3.5)
    block_hand_ft: float = 3.0  # arm length ~0.44 x stature (anthropometric) + a lean; unverified
    dunk_share_0_3: float = 0.27  # dunks 6.1% FGA / 0-3 ft 22.6% FGA (note §4b); derived
    ft_pct: float = 0.780  # NBA 2024-25 league FT% (note §4a); sourced
    ft_pct_sd: float = 0.06  # per-player spread; unverified
    oreb_rate: float = 0.25  # offensive rebound rate on FG misses (spec ~25%; note ORB% 25.2); sourced
    # ... by the miss's pbpstats zone (note §4c 2024-25 OREB% after a miss: at rim .382, short mid 4-14 ft .312,
    # long mid .220, corner 3 .264, arc 3 .242; .285 overall, team rebounds included), scaled by
    # oreb_rate / oreb_zone_ref so the overall rate stays at oreb_rate. It used to be oreb_rate for every
    # zone. Zone shape sourced, level derived.
    oreb_zone: dict[str, float] = _cfg_dict({"rim": 0.382, "short_mid": 0.312, "long_mid": 0.220, "c3": 0.264,
                                             "ab3": 0.242})
    oreb_zone_ref: float = 0.285
    oreb_rate_ft: float = 0.131  # on missed final FTs, 2024-25 (note §4c); sourced
    oreb_rate_block: float = 0.40  # after blocks (note: 58% of blocks recovered by the defense); unverified
    putback_rate: float = 0.40  # immediate putback after an OREB within 6 ft of the rim; tuned
    # Players (besides the rebounder) contesting the landing spot of a missed shot; design choice,
    # unverified (no sourced contested-rebound share; NBA.com's 3.5 ft "contested" threshold is for shots).
    rebound_contest_n: tuple[int, int] = (1, 3)
    # A blocked ball leaves the blocker's hand unexpectedly at 8-15 ft/s: nobody controls it before this
    # reaction time has passed (choice reaction to an unexpected deflection ~0.15-0.25 s; it used to be
    # "caught" on the block frame itself, then dropped as a dribble). Design choice, unverified.
    block_loose_s: float = 0.2
    # Rebound positioning starts at the release of a field-goal try, not at the miss (players react to
    # the shot): defenders box out the man they guard when he crashes, else sink toward the rim; offensive
    # players crash by role (P(crash) C / F / G) and the rest get back in transition defence; the shooter
    # follows his shot only from close range, after landing. Shares are design choices, unverified (no
    # sourced crash / get-back split); the rebound control rule and the OREB prior are unchanged.
    crash_share: dict[str, float] = _cfg_dict({"C": 0.9, "F": 0.6, "G": 0.15})
    shooter_follow_ft: float = 12.0  # the shooter crashes his own miss only from within this distance
    crash_spot_ft: tuple[float, float] = (4.0, 8.0)  # crashers' spots, from the rim
    sink_ft: tuple[float, float] = (5.0, 9.0)  # defenders (F / C) without a man to box out sink this far from the rim
    outlet_ft: tuple[float, float] = (13.0, 19.0)  # ... defensive guards hold the long-rebound / outlet area
    get_back_ft: tuple[float, float] = (30.0, 42.0)  # get-back spot: this far from the shot's basket
    team_rebound_rate: float = 0.03  # misses deflected out of bounds before control; unverified
    assist_window_s: float = 5.0  # ARCHITECTURE assist rule (catch -> release)
    assist_max_dribbles: int = 2  # ARCHITECTURE assist rule
    ft_set_s: float = 1.5  # shooter stationary at the FT line before each FT (RESEARCH §7A.5 rule 2)
    # Free-throw cadence is not machine-regular (the FT-to-FT gap used to be 3.6-4.0 s, p5-p95): after a
    # first FT the official retrieves the ball before handing it back (ft_return_s), the shooter's routine
    # before the release varies (ft_routine_s; well inside the FT time limit once he has the ball, FIBA 5 s /
    # NBA 10 s, a limit not yet in the rules note: to verify), and lane players shift a little between
    # attempts (ft_lane_shuffle_ft). Ranges are design choices, unverified (no sourced FT-interval
    # distribution).
    ft_return_s: tuple[float, float] = (0.2, 1.6)
    ft_routine_s: tuple[float, float] = (0.2, 2.2)
    ft_lane_shuffle_ft: float = 0.35
    # After a made basket the inbounder picks the ball up himself, off the bounce once it is below this
    # height (a hand-height catch; the official's toss is kept for other dead balls); design, unverified.
    made_pickup_z_ft: float = 5.0
    # A rebounder secures the ball, lands and pivots before he outlets it ("rebound - chin - pivot - outlet"
    # fundamentals; a jump pass is a fundamentals error): no pass leaves while the passer's feet are off the
    # floor, and after a rebound the first pass waits rebound_secure_s after the landing (after the control
    # when he caught it on the floor). Outlets used to leave 0.3 s after the catch (p10 = p25 = 0.30 s), 45% of
    # them with the rebounder still in the air (a 1.5-2.5 ft jump has 0.6-0.8 s hang time). Grounded catches
    # keep the 0.3 s catch-and-pass. Design, unverified.
    rebound_secure_s: tuple[float, float] = (0.5, 1.0)
    # At-rim tries: a drive that does not get to its gather spot (cut off) is finished from where it stops,
    # and the label follows the release distance: a dunk only within dunk_max_ft of the basket, a layup
    # within layup_max_ft, else a floater / runner (a "jumper"). 12% of layups and 11% of dunks used to be
    # released > 6 ft out (max 9.0 / 10.1 ft). Design, unverified.
    dunk_max_ft: float = 3.5
    layup_max_ft: float = 6.0

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
    # Per frame, substitutes waiting in the substitution box (player_id, team value, x, y): visible
    # people off the court (rendered in video mode), not on-court players.
    waiting: list[list[tuple[str, str, float, float]]] = field(default_factory=list)
    # Per frame, each on-court player's feet height above the floor (ft; frame.players order, 0 when
    # grounded): jumps lift the image box, so a monocular foot point is off the floor (NoiseModel).
    feet_z: list[list[float]] = field(default_factory=list)
    # Per frame, players reaching for the ball with a hand (player_id, hand x, y, z): their image box includes
    # the arm (SimConfig.arm_length_ratio); video renders draw the arm.
    hands: list[list[tuple[str, float, float, float]]] = field(default_factory=list)


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


def cylinder_boxes(cam: PinholeCamera, xy: np.ndarray, height: Any, radius: float, base: Any = 0.0) -> np.ndarray:
    """(N, 2) floor points -> (N, 4) exact bboxes of upright cylinders (= ``bbox_of_cylinder``) spanning
    heights ``base`` .. ``height`` (scalars or (N,) arrays: a jumping player's feet leave the floor and
    raised arms / a raised ball extend the top)."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    z0 = np.broadcast_to(np.asarray(base, dtype=float), (len(xy),))
    z1 = np.broadcast_to(np.asarray(height, dtype=float), (len(xy),))
    bottom = np.column_stack([xy, z0])
    top = np.column_stack([xy, z1])
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
    deflect: bool = False  # the interceptor only deflects the ball (out of bounds), no control
    fumble: bool = False  # the receiver touches but cannot hold it (off his hands, out of bounds)
    read_t: float = math.inf  # flight time at which the interceptor has read the pass and jumps the lane
    lane_xy: np.ndarray | None = None  # where he goes to meet the ball
    closeout: tuple[int, float, np.ndarray] | None = None  # (receiver's defender, flight time he reacts, spot)
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
    and_one: bool = False
    p_make_open: float = 0.0  # P(make | not blocked): a planned block nobody can reach plays out with it
    path: str = "swish"  # how it plays out (SimConfig.make_paths / miss_paths, or "blocked")
    zone: str = ""  # CourtSpec.zone at the gather
    board_touched: bool = False
    decide_t: float | None = None  # air ball: flight time its miss is decided (it has come down past the ring)


@dataclass
class _ShotPath:
    """A planned try: first target ``p1`` reached after ``T`` s, then the contacts (impulses)."""

    kind: str
    p1: np.ndarray
    T: float
    impulses: list[tuple[float, np.ndarray, str]] = field(default_factory=list)
    decide_t: float | None = None
    touch: bool = False  # a make that brushes the ring without a planned contact (counts as rim contact)


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
    spot: np.ndarray  # the outgoing player's bench seat (off the court)
    start_t: float
    box: np.ndarray = field(default_factory=lambda: np.zeros(2))  # substitution box: the substitute waits here


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
        self.dribble_after: tuple[int, float, float] | None = None  # (holder, landing time, period): pending dribble
        self.bounce_xy: np.ndarray | None = None  # floor contact point of the last bounce (this step)

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
        # Separate stream for appearance-only draws (jump heights): they never change the game flow.
        self.vis_rng = np.random.default_rng([cfg.seed, 17])
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
        self.team_fouls = {t: 0 for t in _TEAMS}  # team fouls in the current bonus scope (period / half)
        self.late_fouls = {t: 0 for t in _TEAMS}  # team fouls inside the period's late window (RuleSpec.bonus)
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
        # Vertical jumps (feet off the floor) and raised arms, per slot: takeoff time, takeoff speed,
        # arms raised until. Only the image boxes use them; court positions stay on the floor.
        self.jump_t0 = np.full(10, -1e9)
        self.jump_v0 = np.zeros(10)
        self.arms_until = np.full(10, -1e9)
        self.react_until = np.full(10, -1e9)  # auto-defense resumes after this time (post-steal reaction)
        # Ball hand per slot (ball protection, SimConfig.protect_*): the held / dribbled ball's (ahead, lateral)
        # offset from the holder (lateral > 0 = his right), the hand it is in, and when he last switched hands.
        # Handedness per player from its own stream (it never shifts the game's other draws).
        self.hand_cur = np.tile([self.cfg.hand_offset_ft, 0.0], (10, 1))
        self.hand_side = np.ones(10)
        self.hand_switch_t = np.full(10, -1e9)
        hand_rng = np.random.default_rng([self.cfg.seed, 29])
        self.strong = np.where(hand_rng.random(n) < self.cfg.left_hand_share, -1.0, 1.0)
        self.stripper: int | None = None  # the defender making a planned strip (may reach the ball)
        self.scheme_rng = np.random.default_rng([self.cfg.seed, 31])  # per-possession defensive scheme draws
        self.force_weak = False  # this possession's on-ball defenders shade the handler's strong hand
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
        # (slot, control_t, time) after a rebound: no pass before ``time`` (land + secure, rebound_secure_s)
        self.secure: tuple[int, float, float] | None = None
        self.last_ft = True  # False while a non-final free throw is being shot
        self.ball_live = False
        self.shot_pending: _Shot | None = None
        self.loose_after: _Shot | None = None
        # rebound positioning during a shot's flight: box-out pairs (defender, man) followed every frame,
        # and the shooter's own move once he has landed (slot, target, from time)
        self.boxouts: list[tuple[int, int]] = []
        self.shooter_next: tuple[int, np.ndarray, float] | None = None
        self.pass_info: _PassInfo | None = None
        self.period_expired = False
        self.horn_period = 0  # period whose PERIOD_END was emitted
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
            self.boxouts, self.shooter_next = [], None
        for _ in range(16):
            try:
                next(self.gen)
                return
            except StopIteration as stop:
                self.gen = stop.value if stop.value is not None else self._phase_idle()
        raise RuntimeError("simulator phase chain did not yield")

    def _interrupt(self) -> Phase | None:
        """Period buzzer, shot-clock violation or ball out of bounds pre-empt the current phase."""
        if self.period_expired and not self.game_over:
            self._horn()  # PERIOD_END on the frame the game clock reaches 0 (a shot in the air still counts)
            if self.shot_pending is None:
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
        self._resolve_contacts()
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

    def _ball_hand_xy(self) -> np.ndarray | None:
        """Where a held / dribbled ball is this step (its nominal hand point), else None."""
        b = self.ball
        if b.holder is None or b.mode not in ("held", "dribble"):
            return None
        if b.mode == "held":
            return self._held_nominal(b.holder)[:2]
        return self._hand_xy(b.holder, lateral=0.5)

    def _resolve_contacts(self) -> None:
        """Hard non-penetration (``body_contact_ft``), locked players included, and no other body on a
        held / dribbled ball (``ball_clear_ft``). Overlaps are projected apart, weighted by mobility
        (the ball handler and locked players yield least), and the approaching normal velocity of a
        contact is removed (an inelastic body contact)."""
        c = self.cfg
        dmin = c.body_contact_ft
        mob = np.where(self.locked, 0.35, 1.0)
        for d, _ in self.boxouts:  # a box-out holds its ground: the crashing man is sealed, not pushed through
            mob[d] = 0.35
        holder = self.ball.holder if self.ball.mode in ("held", "dribble") else None
        if holder is not None:
            mob[holder] = 0.1
        hand = self._ball_hand_xy()
        others = [q for q in range(10) if q != holder] if holder is not None else []
        pos, vel = self.pos, self.vel
        for _ in range(4):
            moved = False
            if hand is not None:
                for q in others:
                    off = pos[q] - hand
                    d = float(np.hypot(*off))
                    clear = c.strip_clear_ft if q == self.stripper else c.ball_clear_ft
                    if d < clear - 1e-9:
                        n = off / d if d > 1e-6 else _unit(pos[q] - pos[holder])
                        if not n.any():
                            n = np.array([0.0, 1.0])
                        pos[q] = hand + n * clear
                        vn = float(vel[q] @ n)
                        if vn < 0.0:
                            vel[q] -= n * vn
                        moved = True
            diff = pos[:, None, :] - pos[None, :, :]
            dist = np.hypot(diff[..., 0], diff[..., 1])
            ii, jj = np.nonzero(np.triu(dist < dmin - 1e-9, 1))
            for a, b in zip(ii.tolist(), jj.tolist(), strict=True):
                d = float(np.hypot(*(pos[a] - pos[b])))
                over = dmin - d
                if over <= 0.0:
                    continue
                n = (pos[a] - pos[b]) / d if d > 1e-6 else _rot(np.array([1.0, 0.0]), 2.399 * (a + 3 * b))
                wa, wb = mob[a], mob[b]
                tot = wa + wb
                pos[a] += n * over * wa / tot
                pos[b] -= n * over * wb / tot
                vrel = float((vel[a] - vel[b]) @ n)
                if vrel < 0.0:  # approaching: cancel the normal component, mobility-weighted
                    vel[a] -= n * vrel * wa / tot
                    vel[b] += n * vrel * wb / tot
                moved = True
            if not moved:
                break

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

    def _hand_xy(self, s: int, lateral: float = 0.0, ahead: float | None = None) -> np.ndarray:
        """A hand point of slot ``s``: ``ahead`` (default ``hand_offset_ft``) along his facing, ``lateral`` to
        his right (negative: left)."""
        f = self.face[s]
        p = self.pos[s] + f * (self.cfg.hand_offset_ft if ahead is None else ahead)
        if lateral:
            p = p + np.array([f[1], -f[0]]) * lateral
        return p

    def _ball_xy(self, s: int) -> np.ndarray:
        """Where the holder ``s`` has the ball: his current ball-hand offset (ball protection)."""
        ahead, lateral = self.hand_cur[s]
        return self._hand_xy(s, lateral=float(lateral), ahead=float(ahead))

    def _reset_hand(self, s: int) -> None:
        """A new holder has the ball in his strong hand, ahead of him (unguarded offset)."""
        self.hand_side[s] = self.strong[int(self.occ[s])]
        self.hand_cur[s] = (self.cfg.hand_offset_ft, 0.0)
        self.hand_switch_t[s] = self.t

    def _update_hand(self, s: int, dribble: bool, dt: float) -> None:
        """Ball protection (``SimConfig.protect_*``): with an opponent within ``protect_ft`` the ball goes to
        the side away from him (hysteresis: a defender straight ahead or behind keeps the hand; hands switch
        at most every ``hand_switch_s``), and the offset moves at most ``hand_move_fts``."""
        c = self.cfg
        opp = list(self._slots(self.slot_team[s].other()))
        rel = self.pos[opp] - self.pos[s]
        dist = np.hypot(rel[:, 0], rel[:, 1])
        k = int(np.argmin(dist))
        w = float(np.clip((c.protect_ft - dist[k]) / max(c.protect_ft - c.protect_full_ft, 1e-6), 0.0, 1.0))
        side = float(self.hand_side[s])
        if w > 0.0:
            f = self.face[s]
            cross = float(f[0] * rel[k, 1] - f[1] * rel[k, 0])  # > 0: the defender is on his left
            want = 1.0 if cross > 0.3 else (-1.0 if cross < -0.3 else side)
            if want != side and self.t - self.hand_switch_t[s] >= c.hand_switch_s - 1e-9:
                self.hand_side[s] = side = want
                self.hand_switch_t[s] = self.t
        base_lat = c.dribble_lat_ft if dribble else 0.0
        tgt = np.array([c.hand_offset_ft + (c.protect_fwd_ft - c.hand_offset_ft) * w,
                        side * (base_lat + (c.protect_lat_ft - base_lat) * w)])
        d = tgt - self.hand_cur[s]
        n = float(np.hypot(*d))
        step = c.hand_move_fts * dt
        self.hand_cur[s] = tgt if n <= step else self.hand_cur[s] + d * (step / n)

    def _jump_now(self, s: int) -> float:
        """Slot ``s``'s feet height above the floor now (his scheduled jump)."""
        tau = self.t - self.jump_t0[s]
        if tau < 0.0:
            return 0.0
        return max(0.0, float(self.jump_v0[s] * tau - 0.5 * GRAVITY_FTS2 * tau * tau))

    def _land_t(self, s: int) -> float:
        """When slot ``s`` lands from his current jump (-inf if he is not jumping)."""
        if self.jump_t0[s] < -1e8:
            return -math.inf
        return float(self.jump_t0[s] + 2.0 * self.jump_v0[s] / GRAVITY_FTS2)

    def _advance_ball(self, dt: float) -> None:
        b = self.ball
        b.clear_flags()
        pend = b.dribble_after
        if pend is not None:
            if b.mode != "held" or b.holder != pend[0] or b.aim_w > 0.0:
                b.dribble_after = None
            elif self.t >= pend[1] - 1e-9:
                self._start_dribble(pend[0], pend[2])
        if b.mode in ("held", "dribble") and b.holder is not None and b.aim_w <= 0.0:
            self._update_hand(b.holder, b.mode == "dribble", dt)
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
        """Where a held ball sits: in the holder's ball hand (ahead of him, or to the side away from a close
        defender) at ``held_z`` above his feet, so a ball caught in the air comes down with him (moving to the
        release point, at absolute heights, during a shooting motion)."""
        b = self.ball
        xy = self._ball_xy(s)
        if b.aim_w > 0.0:
            return np.append(xy + b.aim_w * (b.aim - xy), b.held_z)
        return np.append(xy, b.held_z + self._jump_now(s))

    def _dribble_nominal(self, s: int) -> np.ndarray:
        b = self.ball
        return np.append(self._ball_xy(s), self._dribble_z(b.dribble_phase, b.dribble_period))

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
            b.bounce_xy = b.pos[:2].copy()
            if b.vel[2] < c.roll_below_fts:
                b.rolling = True
                b.vel[2] = 0.0
            dt -= tau

    def _detect_out(self) -> None:
        b = self.ball
        if not self.ball_live or self.ball_out is not None:
            return
        x, y, z = b.pos
        court = self.court
        bounce_out = b.bounced and b.bounce_xy is not None and not court.in_bounds(*b.bounce_xy)
        if court.in_bounds(x, y) and not bounce_out:
            b.no_oob = False  # a throw-in becomes subject to the boundary once it is inside
            return
        if b.no_oob or self._catch_due(b):
            return
        # Out of bounds is a touch: the floor (the line itself is out), a person or an object out of bounds
        # (rules note §2a / NBA Rule 8; to verify), not a ball still in the air beyond the line (it used to be
        # out the moment its centre crossed the line below 4.5 ft, ~0.2-0.45 s early).
        L, W = court.length, court.width
        beyond = math.hypot(x - min(max(x, 0.0), L), y - min(max(y, 0.0), W))
        floor = bounce_out or (b.rolling and not court.in_bounds(x, y))
        thing = beyond >= self.cfg.dead_ball_stop_ft - 0.05 and z < 8.0  # bench, table, fans
        person = z < 7.5 and self._person_out_of_bounds_at(b.pos[:2])
        if floor or thing or person:
            self.ball_out = b.pos.copy()
            self._stop_dead_ball_outside()  # dead from this frame: the bench / fans stop it

    def _person_out_of_bounds_at(self, xy: np.ndarray) -> bool:
        """A referee or player standing out of bounds within touching distance of the ball (horizontal)."""
        reach = self.cfg.player_radius_ft * 0.8 + self.ball.r
        for p in list(self.ref_pos) + [self.pos[q] for q in range(10)] + [t.box for t in self.sub_tasks]:
            if not self.court.in_bounds(float(p[0]), float(p[1])) and _dist(p, xy) <= reach:
                return True
        return False

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
        if b.holder != s or b.mode not in ("held", "dribble"):
            self._reset_hand(s)
        self.boxouts = []  # control ends any box-out
        b.mode, b.holder, b.catch = "held", s, None
        # (``z`` is an absolute height; the held height is kept relative to his feet, see _held_nominal)
        b.held_z = self.cfg.held_z_ft if z is None else z - self._jump_now(s)
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
        b.dribble_after = None  # (a dribble still waiting for the landing never starts)
        return None

    def _dribble(self, s: int) -> None:
        b = self.ball
        if b.mode != "dribble" or b.holder != s:
            period = float(self.rng.uniform(*self.cfg.dribble_period_s))
            if self._land_t(s) > self.t + 1e-9:  # caught in the air: the dribble starts once he has landed
                if b.mode != "held" or b.holder != s:
                    self._hold(s)
                b.dribble_after = (s, self._land_t(s), period)
                return
            self._start_dribble(s, period)

    def _start_dribble(self, s: int, period: float) -> None:
        b = self.ball
        self._hold(s)
        b.mode = "dribble"
        b.dribble_phase = 0.0
        b.dribble_period = period
        b.carry = b.pos - self._dribble_nominal(s)
        b.dribble_after = None

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
            self.force_weak = bool(self.scheme_rng.random() < self.cfg.force_weak_share)
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

    def _await_subs(self, max_s: float = 25.0) -> Phase:
        """A dead ball is not put back in play while a substitution is pending (the outgoing player is
        still walking off; ``max_s`` is only a safety bound)."""
        yield from self._wait_until(lambda: not self.sub_tasks, max_s)

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

    def _jump_heights(self, t: float) -> np.ndarray:
        tau = t - self.jump_t0
        z = self.jump_v0 * tau - 0.5 * GRAVITY_FTS2 * tau * tau
        return np.where((tau >= 0.0) & (z > 0.0), z, 0.0)

    def _schedule_jump(self, s: int, h: float, t_apex: float, arms: bool = True) -> None:
        """Slot ``s`` jumps ``h`` ft with the apex at ``t_apex`` (arms raised while airborne). A takeoff
        before the current frame is written back into the frames already recorded."""
        if h <= 0.0:
            return
        v0 = math.sqrt(2.0 * GRAVITY_FTS2 * h)
        t_up = v0 / GRAVITY_FTS2
        t0 = t_apex - t_up
        self.jump_t0[s], self.jump_v0[s] = t0, v0
        if arms:
            self.arms_until[s] = max(self.arms_until[s], t0 + 2.0 * t_up)
        fps = self.cfg.fps
        for i in range(max(0, int(math.ceil(t0 * fps - 1e-9))), min(self.frame, len(self.r_jump))):
            tau = i / fps - t0
            self.r_jump[i, s] = max(self.r_jump[i, s], max(0.0, v0 * tau - 0.5 * GRAVITY_FTS2 * tau * tau))
            if arms:
                self.r_arms[i, s] = True

    def _jump_for(self, s: int, kind: str, need_ft: float, t_apex: float) -> None:
        """A ``kind`` jump (``SimConfig.jump_ft``) high enough to reach ``need_ft`` above the standing reach."""
        lo, hi = self.cfg.jump_ft.get(kind, (0.0, 0.0))
        h = max(float(self.vis_rng.uniform(lo, hi)) if hi > 0 else 0.0, need_ft)
        self._schedule_jump(s, min(h, 3.6), t_apex)

    def _reach_ft(self) -> float:
        return self.cfg.standing_reach_ratio * self.cfg.player_height_ft

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
        self._reset_team_fouls(period)
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
        for team in _TEAMS:  # both jumpers rise to the tossed ball (apex at the tap)
            self._jump_for(centers[team], "rebound", float(self.ball.pos[2]) + self.ball.r - self._reach_ft() + 0.3,
                           self.t)
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

    def _horn(self) -> None:
        """The period's horn: PERIOD_END now (ARCHITECTURE: the last frame of the period), once. A shot
        released before it still plays out and counts (note §2a); no rebound follows a miss."""
        if self.horn_period == self.period:
            return
        self.horn_period = self.period
        self.clock_running = self.sc_running = False
        self._emit(EventKind.PERIOD_END, attrs={"period": self.period})

    def _phase_period_end(self) -> Phase:
        self._horn()
        self.clock_running = self.sc_running = False
        self.ball_live = False
        self.shot_pending = self.loose_after = None
        self._emit_pass(False)
        self.motion_on = self.def_auto_on = False
        self.auto[:] = False
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
        yield from self._await_subs()
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
        if cause in ("made_shot", "free_throw") and self.ball.mode in ("flight", "rest"):
            # after a made basket a player of the scored-on team retrieves the ball himself from under the
            # basket (off the bounce or off the floor), then steps out of bounds behind the end line: no
            # official handles it. POSSESSION is the first frame of that control (ARCHITECTURE).
            nxt = yield from self._retrieve_made_ball(team, inb, p_spot, cause)
            if nxt is not None:
                return nxt
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
        yield from self._await_subs()
        yield from self._wait_until(lambda: _dist(self.pos[recv], recv_spot) < 2.0, 5.0)
        result = yield from self._act_pass(recv, "inbound")
        if isinstance(result, Generator):
            return result
        self.clock_running = True
        self.sc_running = self.sc is not None
        self.locked[inb] = False
        return self._continue_with_ball(team)

    def _retrieve_made_ball(self, team: Team, inb: int, p_spot: np.ndarray, cause: str) -> Phase:
        """After a made basket: the inbounder runs to the ball under the basket and picks it up (off the
        bounce, below ``made_pickup_z_ft``), which is the new team's first control; he then heads for the
        throw-in spot behind the end line. Falls back to the dead-ball toss if he cannot get to it."""
        c, b = self.cfg, self.ball
        L, W = self.court.length, self.court.width
        self.locked[inb] = True
        self.vmax[inb] = c.run_speed_fts
        end = self.t + 8.0
        while not (b.mode in ("flight", "rest") and self._reach(inb, 1.8, c.made_pickup_z_ft)):
            if self.t >= end - 1e-9 or b.mode not in ("flight", "rest"):
                return None
            lead = 0.2 if b.mode == "flight" and not b.rolling else 0.0
            tgt = b.pos[:2] + b.vel[:2] * lead
            self.target[inb] = np.clip(tgt, -8.0, [L + 8.0, W + 8.0])
            yield
        if team is not self.possession_team:
            self._gain(inb, cause)
        else:
            self._hold(inb)
        self.target[inb] = p_spot
        self.vmax[inb] = c.jog_speed_fts
        return None

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

    def _continue_with_ball(self, team: Team, transition: bool = False) -> Phase:
        """After control: bring the ball up (pushing it in transition after a live-ball steal) or start
        the half-court set."""
        s = self.holder
        if s is None:
            return self._phase_loose_ball(team)
        along, _ = self._along_lat(self.attack[team], self.pos[s])
        if along > 31.0:
            return self._phase_bring_up(team, push=transition)
        return self._phase_half_court(team)

    def _phase_bring_up(self, team: Team, push: bool = False) -> Phase:
        """Bring the ball into the frontcourt. ``push``: after a steal the stealer attacks at once
        (dribbling at run / sprint speed, no outlet pass to a guard)."""
        c = self.cfg
        self.ball_live = True
        self._enable_defense(team)
        s = self.holder
        if s is None:
            return self._phase_loose_ball(team)
        guards = [g for g in self._slots(team) if g != s and self.role[self.occ[g]] == "G"]
        if not push and self.role[self.occ[s]] != "G" and guards and self.rng.random() < 0.75:
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
        along = float(self.rng.uniform(16.0, 22.0)) if push else 27.0
        entry = self._spot(side, along, float(self.rng.uniform(-9.0, 9.0)))
        self.target[s] = entry
        self.vmax[s] = float(self.rng.uniform(c.run_speed_fts, c.sprint_speed_fts)) if push else float(
            self.rng.uniform(13.0, 17.0))
        self.locked[s] = True
        self._dribble(s)
        if push:  # first get past a defender standing in the way (the robbed player is right there)
            fwd = _unit(entry - self.pos[s])
            perp = np.array([-fwd[1], fwd[0]])
            q = min(self._slots(team.other()), key=lambda o: _dist(self.pos[o], self.pos[s]))
            rel = self.pos[q] - self.pos[s]
            if float(rel @ fwd) > 0.0 and math.hypot(*rel) < 5.0:
                sign = -1.0 if float(rel @ perp) >= 0.0 else 1.0
                way = self.pos[s] + perp * sign * 4.0 + fwd * 2.5
                self.target[s] = way
                yield from self._wait_until(lambda: _dist(self.pos[s], way) < 1.2 or self._urgent(), 0.8)
                self.target[s] = entry
        stop_at = 24.0 if push else 30.0
        yield from self._wait_until(lambda: self._along_lat(side, self.pos[s])[0] < stop_at or self._urgent(), 10.0)
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
            if last and plan.outcome in ("intercept", "oob", "fumble_out", "deflect_out"):
                kind = plan.outcome
            nxt = yield from self._act_pass(recv, kind)
            if nxt is not None:
                return nxt
            if self.holder is None:
                return self._phase_loose_ball(team)
            if self.holder != plan.actor:
                self._apply_plan_spots(team, plan)
        if plan.outcome in ("intercept", "fumble_out", "deflect_out"):  # it did not happen: keep playing
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
        # (an interception is a read after the release, _intercept_read: no defender leaves his man
        # and parks in the lane before the throw)
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
        nsf_rate, defl_rate = c.nonshoot_foul_rate * scale, c.deflect_out_rate * scale
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
        elif r < to_rate + foul_rate + nsf_rate:
            plan = _Plan(outcome="nonshoot_foul")
            plan.actor = self._choose(team, {"G": 1.0, "F": 0.9, "C": 0.8})
        elif r < to_rate + foul_rate + nsf_rate + defl_rate:
            plan = _Plan(outcome="deflect_out")
            plan.actor = self._choose(team, {"G": 1.0, "F": 1.0, "C": 0.6})
        else:
            plan = self._plan_shot(team)
        if plan.outcome in ("oob", "fumble_out", "deflect_out"):  # a receiver near a sideline
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
        ready = self._pass_ready_t()
        while self.t < ready - 1e-9:  # secure the ball first (and land: no jump passes; after a rebound, pivot)
            yield
        yield from self._pickup()
        s = self.holder
        assert s is not None
        team = self.slot_team[s]
        speed = float(rng.uniform(*(c.lazy_pass_speed_fts if kind in ("intercept", "deflect_out")
                                    else c.pass_speed_fts)))
        read = None
        if kind in ("intercept", "deflect_out"):  # a bad pass: thrown to a covered teammate (the planned one
            # if he is covered) whose defender reads it (intercepts it, or gets a hand on it and knocks it out)
            others = [q for q in self._slots(team) if q not in (s, recv)]
            for q in [recv] + [others[k] for k in rng.permutation(len(others))]:
                xy_q, T_q = self._pass_geometry(q, kind, speed)
                read = self._intercept_read(s, q, xy_q, T_q, float(rng.uniform(*c.intercept_read_s)))
                if read is not None:
                    recv = q
                    break
        catch_xy, T = self._pass_geometry(recv, kind, speed)
        if kind != "oob":
            self.target[recv] = catch_xy
            self.locked[recv] = True
        catch = _Catch(recv if kind != "oob" else -1, T, "inbound" if kind == "inbound" else "pass")
        if kind in ("intercept", "deflect_out") and read is not None:  # (none covered: the pass goes through)
            catch.Ti, catch.interceptor, catch.lane_xy, catch.read_t = read
            catch.deflect = kind == "deflect_out"  # ... a hand on it in the lane: off to the sideline
        elif kind == "fumble_out":  # ... or the ball comes off the receiver's hands
            catch.fumble = True
        closer = self._opp(recv)
        if kind == "normal" and catch.interceptor is None and self.def_auto_on and self.auto[closer] and \
                rng.random() < c.closeout_share:
            basket = self._basket(team)
            spot = catch_xy + _unit(basket - catch_xy) * float(rng.uniform(*c.closeout_gap_ft))
            catch.closeout = (closer, float(rng.uniform(*c.closeout_read_s)), spot)
        self.pass_info = _PassInfo(self._pid(s), self._pid(recv), team, self.t, self.frame, float(self.pos[s, 0]),
                                   float(self.pos[s, 1]), _dist(self.pos[s], catch_xy))
        self.locked[s] = False
        self._launch(np.append(catch_xy, 4.0), T, catch, no_oob=kind == "inbound")
        self.ball_live = True
        passer_pid = self._pid(s)
        result = yield from self._await_catch()
        if catch.closeout is not None:  # back to man-to-man (he is now the on-ball defender), hands up
            d = catch.closeout[0]
            self.auto[d] = self.def_auto_on and self.slot_team[d] is not team
            if result == "caught":
                self.arms_until[d] = max(self.arms_until[d], self.t + c.closeout_arms_s)
                self.react_until[d] = max(self.react_until[d], self.t + c.closeout_hold_s)  # set at his spot
        if result == "caught":
            self._emit_pass(True, catch_t=self.t)
            self._start_spell(passer_pid)
            self.locked[recv] = False
            return None
        if result == "intercepted":
            return self._on_interception(catch.interceptor, passer_pid)
        # not caught (missed, deflected or fumbled): wait for the ball to go out of bounds (interrupt)
        # or recover it
        yield from self._wait_until(lambda: False, 2.5)
        self._emit_pass(False)
        return self._phase_loose_ball(team)

    def _pass_ready_t(self) -> float:
        """Earliest release of the holder's pass: 0.3 s after his control (a grounded catch-and-pass), not
        before he has landed, and after a rebound not before his secure / pivot time (``rebound_secure_s``)."""
        s = self.holder
        ready = self.control_t + 0.3
        if s is None:
            return ready
        ready = max(ready, self._land_t(s))
        sec = self.secure
        if sec is not None and sec[0] == s and abs(sec[1] - self.control_t) < 1e-9:
            ready = max(ready, sec[2])
        return ready

    def _pass_time(self, p0: np.ndarray, p1: np.ndarray, T: float, apex_max: float) -> float:
        """Flight time of a pass: the planned ``T`` shortened (a harder, flatter pass, at most
        ``pass_max_speed_fts``) until the arc peaks at or below ``apex_max`` and stays clear of both
        rims and backboards (below ``rim_height - pass_rim_clear_ft`` within ``pass_rim_zone_ft`` of a
        basket): passers throw around or under the basket, never through the cylinder above the rim."""
        c = self.cfg
        d = math.hypot(*(p1[:2] - p0[:2]))
        t_min = max(0.12, d / c.pass_max_speed_fts)
        baskets = [np.array(self.court.basket_xy(side)) for side in ("left", "right")]
        ceiling = self.rim_h - c.pass_rim_clear_ft
        taus = np.linspace(0.0, 1.0, 25)
        while True:
            v = _ballistic(p0, p1, T)
            pts = p0[None, :] + v[None, :] * (taus[:, None] * T)
            pts[:, 2] -= 0.5 * GRAVITY_FTS2 * (taus * T) ** 2
            ok = float(pts[:, 2].max()) <= apex_max + 1e-9
            if ok:
                for bxy in baskets:
                    near = np.hypot(pts[:, 0] - bxy[0], pts[:, 1] - bxy[1]) <= c.pass_rim_zone_ft
                    if np.any(near & (pts[:, 2] > ceiling)):
                        ok = False
                        break
            if ok or T <= t_min + 1e-9:
                return max(T, t_min)
            T = max(t_min, T * 0.93)

    def _oob_point(self, recv: int) -> np.ndarray:
        W = self.court.width
        y = self.pos[recv, 1]
        out_y = -float(self.rng.uniform(4.0, 8.0)) if y < W / 2.0 else W + float(self.rng.uniform(4.0, 8.0))
        return np.array([self.pos[recv, 0] + float(self.rng.uniform(-4.0, 4.0)), out_y])

    def _pass_geometry(self, recv: int, kind: str, speed: float) -> tuple[np.ndarray, float]:
        """Catch point (the receiver's position led 0.3 s, kept inbounds; an overthrown lob lands out of
        bounds) and flight time of a pass to ``recv`` thrown at ``speed``."""
        c = self.cfg
        if kind == "oob":
            catch_xy = self._oob_point(recv)
        else:  # receivers catch inbounds (a lead toward the line never puts the catch over it)
            catch_xy = np.clip(self.pos[recv] + self.vel[recv] * 0.3, 0.8,
                               [self.court.length - 0.8, self.court.width - 0.8])
        p1 = np.append(catch_xy, 4.0)
        T = max(0.2, _dist(self.ball.pos, p1) / speed)
        if kind == "oob":  # overthrown lob: sails over the receiver's reach and lands out of bounds
            T = max(T, 1.0, _dist(self.ball.pos, p1) / 24.0)
        T = self._pass_time(self.ball.pos.copy(), p1, T, c.oob_lob_apex_ft if kind == "oob" else c.pass_apex_ft)
        return catch_xy, T

    def _intercept_read(self, s: int, recv: int, catch_xy: np.ndarray, T: float, react: float
                        ) -> tuple[float, int, np.ndarray, float] | None:
        """An interception is a read: the receiver's defender reacts ``react`` s after the release and
        sprints for the earliest point of the passing lane he can reach in time (acceleration and speed
        limits, hand reach ``intercept_hand_ft``, the ball at most ``intercept_max_z_ft`` high, a jump above
        head height). Returns (flight time, defender, lane point, react) or None when he cannot get there
        (the pass goes through)."""
        c = self.cfg
        d = self._opp(recv)
        p0, p1 = self.ball.pos.copy(), np.append(catch_xy, 4.0)
        v = _ballistic(p0, p1, max(T, 0.05))
        taus = np.linspace(0.15, 0.92, 40) * T
        pts = p0[None, :] + v[None, :] * taus[:, None]
        pts[:, 2] -= 0.5 * GRAVITY_FTS2 * taus ** 2
        for tau, pt in zip(taus.tolist(), pts, strict=True):
            if pt[2] > c.intercept_max_z_ft or tau <= react:
                continue
            if _dist(self.pos[d], pt[:2]) - c.intercept_hand_ft <= self._run_reach(tau - react):
                return tau, d, pt[:2].copy(), react
        return None

    def _run_reach(self, t: float) -> float:
        """Distance a player covers from rest in ``t`` s (acceleration then top speed), with a margin for
        the arrival braking of the motion model."""
        a, vmax = self.cfg.accel_fts2, self.cfg.max_speed_fts
        t_acc = vmax / a
        d = 0.5 * a * t * t if t <= t_acc else vmax * t - 0.5 * vmax * t_acc
        return 0.85 * d

    def _await_catch(self) -> Generator[None, None, str]:
        """Wait for the pending catch: 'caught' | 'intercepted' | 'missed'."""
        b = self.ball
        c = b.catch
        if c is None:
            return "missed"
        while True:
            if b.mode != "flight" or b.catch is not c:
                return "missed"
            if c.closeout is not None and b.flight_t >= c.closeout[1] - 1e-9:
                d, _, spot = c.closeout  # the receiver's defender has read the pass: he closes out
                self.auto[d] = False
                self.target[d] = spot
                self.vmax[d] = self.cfg.sprint_speed_fts
                c.closeout = (d, math.inf, spot)
            if c.interceptor is not None and c.lane_xy is not None and b.flight_t >= c.read_t - 1e-9:
                d = c.interceptor  # he has read the pass: he leaves his man and jumps the lane
                self.auto[d] = False
                self.target[d] = c.lane_xy
                self.vmax[d] = self.cfg.max_speed_fts
                c.lane_xy = None
            # (a touch happens at its exact flight time: the ball is put back where the hands met it, not
            # where it flew on to by the end of the frame, which at low fps can be feet away / out of bounds)
            if c.interceptor is not None and b.flight_t >= c.Ti:
                if self._reach(c.interceptor, 3.5, 8.5, at=c.pos_at(c.Ti)):
                    b.pos[:] = c.pos_at(c.Ti)
                    self._catch_jump(c.interceptor)  # above head height: he goes up for it
                    self._reach_hand(c.interceptor)
                    if c.deflect:
                        self._knock_out(self.slot_team[c.interceptor], speed=(14.0, 20.0))
                        return "deflected"
                    self._hold(c.interceptor)
                    return "intercepted"
                c.interceptor = None
            if b.flight_t >= c.T:
                if c.receiver >= 0 and self._reach(c.receiver, c.reach, at=c.pos_at(c.T)):
                    b.pos[:] = c.pos_at(c.T)
                    if c.kind in ("pass", "inbound", "strip"):
                        self._reach_hand(c.receiver)
                    if c.fumble:
                        self._knock_out(self.slot_team[c.receiver], speed=(9.0, 14.0))
                        return "caught_fumble"
                    self._hold(c.receiver)
                    return "caught"
                b.catch = None
                return "missed"
            yield

    def _knock_out(self, toucher: Team, speed: tuple[float, float]) -> None:
        """A touch that sends the flying ball toward the nearest sideline (deflection / fumble)."""
        b, rng = self.ball, self.rng
        out = -1.0 if b.pos[1] < self.court.width / 2.0 else 1.0
        along = float(rng.uniform(-0.5, 0.5))
        d = _unit(np.array([along, out]))
        b.vel = np.append(d * float(rng.uniform(*speed)), float(rng.uniform(2.0, 5.0)))
        b.catch = None
        b.impulses = []
        self.last_touch = toucher

    def _on_interception(self, d: int | None, passer_pid: str) -> Phase:
        assert d is not None
        self._emit_pass(False)
        self._steal(d, passer_pid, "bad_pass")
        return self._continue_with_ball(self.slot_team[d], transition=True)

    def _steal(self, d: int, victim_pid: str, to_type: str, recoverer: int | None = None) -> None:
        """STEAL (credited to ``d``, the first deflector, note §2a) + TURNOVER now; ``recoverer`` (default
        ``d``) is the defender who gains control (a teammate may pick up a ball ``d`` poked loose)."""
        team = self.slot_team[d]
        rec = d if recoverer is None else recoverer
        x, y = float(self.pos[rec, 0]), float(self.pos[rec, 1])
        victim = next((q for q in self._slots(team.other()) if self._pid(q) == victim_pid), None)
        if victim is not None:  # the robbed player needs a moment before he turns and defends
            lo, hi = self.cfg.steal_reaction_s
            self.react_until[victim] = self.t + float(self.rng.uniform(lo, hi))
            self.target[victim] = self.pos[victim] + self.vel[victim] * 0.25
            self.locked[victim] = False
        self._emit(EventKind.STEAL, team=team, player=self._pid(d), other=victim_pid, x=x, y=y)
        self._emit(EventKind.TURNOVER, team=team.other(), player=victim_pid, x=x, y=y, attrs={"type": to_type})
        self._tally(self._pid(d), stl=1)
        self._tally(victim_pid, tov=1)
        self._gain(rec, "steal")

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
        if plan.outcome == "step_out":
            return (yield from self._act_step_out(team, s))
        if plan.outcome == "offensive_foul":
            return (yield from self._act_offensive_foul(team, s))
        if plan.outcome == "nonshoot_foul":
            return (yield from self._act_nonshoot_foul(team, s))
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
        # he creates separation for the pull-up: a step-back / side-step the on-ball defender reads late
        c, d = self.cfg, self._opp(s)
        away = _unit(self.pos[s] - self.pos[d])
        step = self.pos[s] + (away if away.any() else _unit(self.pos[s] - self._basket(self.slot_team[s]))) * float(
            self.rng.uniform(*c.pullup_step_ft))
        step = np.clip(step, 1.0, [self.court.length - 1.0, self.court.width - 1.0])
        self.react_until[d] = max(self.react_until[d], self.t + float(self.rng.uniform(*c.pullup_react_s)))
        self.vmax[s] = 7.0
        return (yield from self._act_shoot(s, "jumper", release_target=step, tag="pullup"))

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
        rotated = False

        def gathered() -> bool:  # (help rotates to the rim once the driver gets near it)
            nonlocal rotated
            if not rotated and _dist(self.pos[s], basket) < c.rim_help_trigger_ft:
                self._rotate_help(team, s, basket)
                rotated = True
            return _dist(self.pos[s], gather) < 1.0

        nxt = yield from self._probe_until(gathered, 4.0, plan)
        if nxt is not None:
            return nxt
        if plan.outcome == "foul":
            yield from self._pickup()
            return self._foul_on(s)
        if _dist(self.pos[s], gather) >= 1.0:  # cut off before his gather spot: he finishes from here
            d = _dist(self.pos[s], basket)
            self.vmax[s] = 6.0
            if d <= c.layup_max_ft:
                return (yield from self._act_shoot(s, "layup", gather_xy=self.pos[s].copy(),
                                                   release_target=self._release_spot(team, s, "0_3"),
                                                   gather_s=0.35, tag="drive"))
            return (yield from self._act_shoot(s, "jumper", gather_s=0.3, tag="drive"))  # a floater / runner
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

    def _rotate_help(self, team: Team, s: int, basket: np.ndarray) -> None:
        """Help at the rim: on ``rim_help_share`` of drives / cuts the help defender nearest the basket
        (within ``rim_help_ft``, not the man on the ball) rotates to a spot ``rim_help_spot_ft`` in front of the
        rim on the driver's side, so at-rim tries can be contested and blocked from the help side."""
        c = self.cfg
        if self.rng.random() >= c.rim_help_share:
            return
        on_ball = self._opp(s)
        cands = [q for q in self._slots(team.other()) if q != on_ball and _dist(self.pos[q], basket) < c.rim_help_ft]
        if not cands:
            return
        q = min(cands, key=lambda q: _dist(self.pos[q], basket))
        u = _unit(self.pos[s] - basket)
        self.auto[q] = False
        self.target[q] = basket + (u if u.any() else np.array([1.0, 0.0])) * float(self.rng.uniform(*c.rim_help_spot_ft))
        self.vmax[q] = c.sprint_speed_fts

    def _act_cut_finish(self, team: Team, s: int) -> Phase:
        self._rotate_help(team, s, self._basket(team))
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
        from_side = False
        if self.rng.random() < self.cfg.strip_help_share:  # a help defender digs / a trailer pokes from behind
            helpers = [q for q in self._slots(team.other()) if q != d and _dist(self.pos[q], self.pos[s]) < 14.0]
            if helpers:
                d, from_side = min(helpers, key=lambda q: _dist(self.pos[q], self.pos[s])), True
        self._dribble(s)
        self.auto[d] = False
        self.vmax[d] = self.cfg.sprint_speed_fts
        t_end = self.t + float(self.rng.uniform(0.6, 1.2))
        reach = self.cfg.strip_reach_ft + 0.6
        self.stripper = d  # he may reach the ball (strip_clear_ft)
        # the strip comes at a floor contact once the defender is within hand reach of the dribbler
        try:
            while self.t < t_end or not self.ball.contact or _dist(self.pos[d], self.pos[s]) > reach:
                way = _unit(self.pos[d] - self.pos[s]) if from_side else _unit(self._basket(team) - self.pos[s])
                self.target[d] = self.pos[s] + way * self.cfg.strip_reach_ft
                if self.t > t_end + 2.0:
                    break
                yield
        finally:
            self.stripper = None
        victim = self._pid(s)
        c = self.cfg
        if self.rng.random() < c.strip_poke_share:
            # poke-away: the ball is knocked a few feet past the dribbler toward the stealer's basket and
            # is loose until someone wins the race to it (see _phase_poke_loose)
            fwd = _unit(self._basket(team.other()) - self.pos[s])
            side = 1.0 if float((self.pos[d] - self.pos[s]) @ np.array([-fwd[1], fwd[0]])) >= 0.0 else -1.0
            way = _rot(fwd, side * math.radians(float(self.rng.uniform(35.0, 70.0))))
            land = self.pos[s] + way * float(self.rng.uniform(*c.strip_poke_ft))
            land = np.clip(land, 1.5, [self.court.length - 1.5, self.court.width - 1.5])
            self.target[s] = self.pos[s].copy()  # the robbed dribbler is caught flat-footed
            self.locked[s] = False
            lo, hi = c.steal_reaction_s
            self.react_until[s] = self.t + float(self.rng.uniform(lo, hi))
            T = float(self.rng.uniform(*c.strip_poke_s))
            self._reach_hand(d)  # his hand on the ball
            self._launch(np.append(land, self.ball.r), T, None)
            self.last_touch = team.other()
            return self._phase_poke_loose(team, d, s, T)
        p1 = np.append(self._hand_xy(d), c.held_z_ft)
        T = 0.3
        self.target[d] = self.pos[d]
        self._launch(p1, T, _Catch(d, T, "strip", reach=4.0))
        result = yield from self._await_catch()
        if result != "caught":
            return self._phase_loose_ball(team.other(), steal_by=d, victim=victim)
        self._steal(d, victim, "steal")
        return self._continue_with_ball(team.other(), transition=True)

    def _phase_poke_loose(self, off: Team, d: int, victim: int, T: float) -> Phase:
        """A poked-away dribble is a loose ball: nobody can control it until it has left the strip area
        (its flight time ``T`` or ``strip_reach_ft`` of travel). The deflector, the robbed dribbler (after
        his reaction time) and the nearest other player of each team race for it; the first within reach
        controls it. A defender's recovery is a STEAL for the first deflector (note §2a) + TURNOVER; an
        offensive recovery is no steal and no turnover (play on with the unexpired shot clock); out of
        bounds off the deflector it stays with the offence (``_out_of_bounds``)."""
        c, b = self.cfg, self.ball
        self.ball_live = True
        self.auto[:] = False
        start_xy, start = b.pos[:2].copy(), self.t
        victim_pid = self._pid(victim)
        defense = off.other()
        near = lambda q: _dist(self.pos[q], b.pos[:2])  # noqa: E731
        chasers = [d, victim]
        chasers += sorted((q for q in self._slots(off) if q != victim), key=near)[:1]
        chasers += sorted((q for q in self._slots(defense) if q != d), key=near)[:1]
        L, W = self.court.length, self.court.width
        winner: int | None = None
        while winner is None:
            free = b.mode == "flight" and (b.flight_t >= T - 1e-9 or _dist(b.pos[:2], start_xy) >= c.strip_reach_ft)
            if free:
                reach = [q for q in chasers if (q != victim or self.t >= self.react_until[q]) and
                         self._reach(q, 2.0, 7.0)]
                if reach:
                    winner = min(reach, key=near)
                    break
            if self.t - start > 6.0:
                winner = min(range(10), key=near)
                break
            for q in chasers:
                if q == victim and self.t < self.react_until[q]:
                    continue
                lead = min(0.5, near(q) / c.sprint_speed_fts) if not b.rolling else 0.3
                self.target[q] = np.clip(b.pos[:2] + b.vel[:2] * lead, -2.0, [L + 2.0, W + 2.0])
                self.vmax[q] = c.sprint_speed_fts
            yield
        if self.slot_team[winner] is defense:
            self._steal(d, victim_pid, "steal", recoverer=winner)
            return self._continue_with_ball(defense, transition=True)
        self._gain(winner, "loose_ball")  # the offence keeps it: no steal, no turnover
        self._start_spell(None)
        return self._continue_with_ball(off)

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

    def _foul_on(self, s: int, n_ft: int = 2) -> Phase:
        """Shooting foul before the release (no FGA): ``n_ft`` FTs (2 or 3) for slot ``s``."""
        team = self.slot_team[s]
        d = min(self._slots(team.other()), key=lambda q: _dist(self.pos[q], self.pos[s]))
        self._hold(s)
        self._whistle_foul(d, s, "shooting", n_ft)
        return self._phase_dead_ball(team, None, "other", ft_shooter=s, n_ft=n_ft)

    # ------------------------------------------------------------------ fouls

    def _reset_team_fouls(self, period: int) -> None:
        """Team fouls reset per period, or per half when RuleSpec.bonus.scope is "half"; the late-window
        count resets every period."""
        b = self.rules.bonus
        half_start = period == 1 or period == self.rules.n_periods // 2 + 1
        if b.get("scope") != "half" or half_start or (self.rules.is_overtime(period) and
                                                       not b.get("ot_counts_toward_last_period")):
            self.team_fouls = {t: 0 for t in _TEAMS}
        self.late_fouls = {t: 0 for t in _TEAMS}

    def _penalty(self, defense: Team) -> str:
        """FTs the next team foul by ``defense`` awards: "two" | "one_and_one" | "none" (RuleSpec.bonus)."""
        b = self.rules.bonus
        n = self.team_fouls[defense] + 1  # this foul's number in the scope
        two = b.get("ot_two_shots_from") if self.rules.is_overtime(self.period) and b.get("ot_two_shots_from") \
            else b.get("two_shots_from")
        if two is not None and n >= int(two):
            return "two"
        late = b.get("late_period_s")
        if late is not None and b.get("late_two_shots_from") and self.game_clock <= float(late) + 1e-9:
            if self.late_fouls[defense] + 1 >= int(b["late_two_shots_from"]):
                return "two"
        one = b.get("one_and_one_from")
        if one is not None and n >= int(one):
            return "one_and_one"
        return "none"

    def _whistle_foul(self, fouler: int, fouled: int, kind: str, n_ft: int, team_foul: bool = True,
                      **attrs: Any) -> None:
        """FOUL now (whistle): the clocks stop; personal and (defensive) team fouls are counted."""
        self.clock_running = self.sc_running = False
        fpid = self._pid(fouler)
        x, y = float(self.pos[fouled, 0]), float(self.pos[fouled, 1])
        self._emit(EventKind.FOUL, team=self.slot_team[fouler], player=fpid, other=self._pid(fouled), x=x, y=y,
                   attrs={"type": kind, "n_ft": int(n_ft), **attrs})
        self._tally(fpid, pf=1)
        self.fouls[int(self.occ[fouler])] += 1
        if team_foul:
            team = self.slot_team[fouler]
            self.team_fouls[team] += 1
            late = self.rules.bonus.get("late_period_s")
            if late is not None and self.game_clock <= float(late) + 1e-9:
                self.late_fouls[team] += 1

    def _act_nonshoot_foul(self, team: Team, s: int) -> Phase:
        """A non-shooting defensive foul (reach-in / hold): FTs in the penalty, else a throw-in for the
        offence with the shot clock at max(remaining, short reset) (note §3: NBA / FIBA frontcourt rule)."""
        self._dribble(s)
        nxt = yield from self._probe(float(self.rng.uniform(0.4, 1.4)), _Plan(outcome="nonshoot_foul"))
        if nxt is not None:
            return nxt
        s = self.holder if self.holder is not None else s
        if self.rng.random() < 0.7:  # on the ball handler
            fouler, fouled = self._opp(s), s
        else:  # off the ball (holding / illegal contact on a cutter)
            fouled = self._choose(team, {"G": 1.0, "F": 1.0, "C": 1.0}, exclude=(s,))
            fouler = self._opp(fouled)
        pen = self._penalty(team.other())
        n_ft = 2 if pen in ("two", "one_and_one") else 0
        self._whistle_foul(fouler, fouled, "personal", n_ft, **({"one_and_one": True} if pen == "one_and_one" else {}))
        if n_ft:
            return self._phase_dead_ball(team, None, "foul", ft_shooter=fouled, n_ft=2,
                                         one_and_one=pen == "one_and_one")
        reset = self.rules.shot_clock_reset_s
        if self.sc is not None and reset is not None:
            self.sc = max(self.sc, float(reset))
        return self._phase_dead_ball(team, ("sideline", self.pos[fouled].copy()), "foul")

    def _act_offensive_foul(self, team: Team, s: int) -> Phase:
        """The ball handler drives into a set defender (charge): FOUL on him + TURNOVER "offensive_foul";
        not a team foul (NBA convention; unverified for other rulesets)."""
        d = self._opp(s)
        self._dribble(s)
        fwd = _unit(self._basket(team) - self.pos[s])
        self.auto[d] = False  # the defender sets his feet in the driving lane
        self.target[d] = self.pos[d].copy()
        self.target[s] = self.pos[s] + fwd * 8.0
        self.vmax[s] = float(self.rng.uniform(12.0, 16.0))
        self.locked[s] = True
        t_end = self.t + float(self.rng.uniform(0.5, 1.0))
        while self.t < t_end or _dist(self.pos[s], self.pos[d]) > self.cfg.body_contact_ft + 0.4:
            if self.t > t_end + 1.5:
                break
            yield
        pid = self._pid(s)
        self._whistle_foul(s, d, "offensive", 0, team_foul=False)
        self._emit(EventKind.TURNOVER, team=team, player=pid, x=float(self.pos[s, 0]), y=float(self.pos[s, 1]),
                   attrs={"type": "offensive_foul"})
        self._tally(pid, tov=1)
        self.locked[s] = False
        return self._phase_dead_ball(team.other(), ("sideline", self.pos[s].copy()), "turnover")

    def _act_step_out(self, team: Team, s: int) -> Phase:
        """The dribbler steps on / over the nearest sideline: TURNOVER "out_of_bounds" at that frame."""
        W = self.court.width
        out_y = -1.0 if self.pos[s, 1] < W / 2.0 else W + 1.0
        x = float(np.clip(self.pos[s, 0] + float(self.rng.uniform(-6.0, 6.0)), 3.0, self.court.length - 3.0))
        self._dribble(s)
        self.target[s] = np.array([x, out_y])
        self.vmax[s] = float(self.rng.uniform(9.0, 13.0))
        self.locked[s] = True
        yield from self._wait_until(lambda: not self.court.in_bounds(*self.pos[s]), 4.0)
        self.locked[s] = False
        if self.court.in_bounds(*self.pos[s]):  # never got there: play on
            return self._phase_half_court(team, force_shot=True)
        pid = self._pid(s)
        self.clock_running = self.sc_running = False
        self._emit(EventKind.TURNOVER, team=team, player=pid, x=float(self.pos[s, 0]), y=float(self.pos[s, 1]),
                   attrs={"type": "out_of_bounds"})
        self._tally(pid, tov=1)
        return self._phase_dead_ball(team.other(), ("sideline", self.pos[s].copy()), "turnover")

    # ------------------------------------------------------------------ shots

    def _act_shoot(self, s: int, shot_type: str, gather_xy: np.ndarray | None = None,
                   release_target: np.ndarray | None = None, gather_s: float = 0.3, tag: str = "other") -> Phase:
        """Gather (ball rises to the release height), release a projectile, play the shot out."""
        c = self.cfg
        yield from self._pickup()
        b = self.ball
        keep = b.mode == "held" and b.holder == s and b.aim_w <= 0.0  # a tip keeps its catch height
        self._hold(s, z=b.held_z + self._jump_now(s) if keep else None)
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
        z0, t0 = self.ball.held_z + self._jump_now(s), self.t  # (absolute: the shooting motion's heights)
        if shot_type == "jumper" and self.court.is_three(float(gxy[0]), float(gxy[1]), side) and \
                self.rng.random() < c.three_foul_rate:  # fouled shooting a three, no make: 3 FTs
            yield from self._wait(0.5 * gather_s)
            self.locked[s] = False
            return self._foul_on(s, n_ft=3)

        def gathering() -> bool:  # gather time, extended (<= 0.5 s) until a layup reaches its release spot
            if self.t - t0 < gather_s - 1e-9:
                return True
            late = self.t - t0 < gather_s + 0.5
            return release_target is not None and late and _dist(self.pos[s], release_target) > 0.4

        while gathering():  # the ball reaches z_rel and the release point on the release frame
            self._shooting_motion(s, z0, z_rel, min(1.0, (self.t + self.dt - t0) / gather_s))
            yield
        self.ball.held_z = z_rel
        shot_type = self._release_label(s, shot_type)
        # the shooter jumped (release at the apex; a jumper slightly before it): written back from the
        # takeoff, since a layup's gather lasts until he reaches his release spot
        need = z_rel + self.ball.r - self._reach_ft() + 0.1
        self._jump_for(s, shot_type, need, self.t + (0.03 if shot_type == "jumper" else 0.0))
        self._cap_gather(s, t0)
        shot = self._release(s, shot_type, gxy, z_rel, tag=tag)
        self._position_for_rebound(shot)
        result = yield from self._shot_flight(shot)
        self.locked[s] = False
        self.boxouts, self.shooter_next = [], None
        return self._after_shot(shot, result)

    def _release_label(self, s: int, shot_type: str) -> str:
        """The try's label from where it is released (``dunk_max_ft`` / ``layup_max_ft``): a planned dunk or
        layup that did not get close enough is a layup or a floater / runner ("jumper")."""
        if shot_type not in ("dunk", "layup"):
            return shot_type
        c = self.cfg
        d = self.court.distance_to_basket(float(self.pos[s, 0]), float(self.pos[s, 1]), self.attack[self.slot_team[s]])
        if d > c.layup_max_ft:
            return "jumper"
        if shot_type == "dunk" and d > c.dunk_max_ft:
            return "layup"
        return shot_type

    def _cap_gather(self, s: int, t0: float) -> None:
        """The ball rises with the shooter's jump: before the takeoff a held ball is never above his standing
        reach (its top at most 0.1 ft below the hand), in the air at most his reach plus the jump. The gather
        raises the ball before the (written-back) takeoff is known, so its recorded frames since ``t0`` are
        capped here (37-39% of layups / dunks used to hold the ball above standing reach + 0.4 ft with the
        feet on the floor)."""
        fps = self.cfg.fps
        lim = self._reach_ft() - self.ball.r - 0.1
        lo = max(0, int(math.ceil(t0 * fps - 1e-9)))
        for i in range(lo, min(self.frame, len(self.r_ball))):
            if self.r_holder[i] != self.occ[s]:
                continue
            cap = lim + self.r_jump[i, s]
            if self.r_ball[i, 2] > cap:
                self.r_ball[i, 2] = cap
                self.r_balltop[i, s] = min(self.r_balltop[i, s], cap + self.ball.r)

    def _position_for_rebound(self, shot: _Shot) -> None:
        """Rebound positioning from the release of a field-goal try (players react to the shot, not to
        the miss; see ``SimConfig.crash_share``): offensive players crash or get back by role, defenders
        box out a crashing man (else sink toward the rim). Followed every frame by
        :meth:`_track_rebound_positions`; the miss itself re-plans the race (:meth:`_crash_boards`)."""
        c, rng = self.cfg, self.rng
        if shot.is_ft:
            return
        basket = np.array(self.court.basket_xy(shot.side))
        self.motion_on = self.def_auto_on = False
        self.auto[:] = False
        crashing: set[int] = set()
        for s in self._slots(shot.team):
            role = self.role[self.occ[s]]
            if s == shot.slot:
                close = _dist(self.pos[s], basket) <= c.shooter_follow_ft
                follow = close and rng.random() < c.crash_share[role]
                tgt = self._crash_spot(basket, s) if follow else self._get_back_spot(shot)
                if follow:
                    crashing.add(s)
                self.shooter_next = (s, tgt, self.t + 0.4)  # after landing from the shot
                continue
            self.locked[s] = False
            self.vmax[s] = c.run_speed_fts
            if rng.random() < c.crash_share[role]:
                crashing.add(s)
                self.target[s] = self._crash_spot(basket, s)
            else:
                self.target[s] = self._get_back_spot(shot)
        self.boxouts = []
        for d in self._slots(shot.team.other()):
            self.locked[d] = False
            self.vmax[d] = c.run_speed_fts
            man = self._opp(d)
            if man in crashing and _dist(self.pos[man], basket) < 24.0:
                self.boxouts.append((d, man))
            else:
                self.target[d] = self._sink_spot(basket, d)
        self._track_rebound_positions(shot)

    def _sink_spot(self, basket: np.ndarray, d: int) -> np.ndarray:
        """A defender with no crashing man to box out: bigs and forwards sink toward the rim (``sink_ft``),
        guards hold the long-rebound / outlet area (``outlet_ft``) instead of crowding the paint."""
        lo, hi = self.cfg.outlet_ft if self.role[self.occ[d]] == "G" else self.cfg.sink_ft
        u = _unit(self.pos[d] - basket)
        if not u.any():
            u = np.array([1.0, 0.0])
        spot = basket + u * float(self.rng.uniform(lo, hi))
        return np.clip(spot, 1.0, [self.court.length - 1.0, self.court.width - 1.0])

    def _crash_spot(self, basket: np.ndarray, s: int) -> np.ndarray:
        """A crashing player's spot: 4-8 ft from the rim, from his side of the floor."""
        u = _unit(self.pos[s] - basket)
        if not u.any():
            u = np.array([1.0, 0.0])
        ang = float(self.rng.uniform(-0.6, 0.6))
        spot = basket + _rot(u, ang) * float(self.rng.uniform(*self.cfg.crash_spot_ft))
        return np.clip(spot, 1.0, [self.court.length - 1.0, self.court.width - 1.0])

    def _get_back_spot(self, shot: _Shot) -> np.ndarray:
        """Transition defence: back toward the other end, ``get_back_ft`` from the shot's basket."""
        lo, hi = self.cfg.get_back_ft
        return self._spot(shot.side, float(self.rng.uniform(lo, hi)), float(self.rng.uniform(-12.0, 12.0)))

    def _track_rebound_positions(self, shot: _Shot) -> None:
        """Per frame while the shot is in the air: box-outs stay between the man and the basket in body
        contact; the shooter moves on once he has landed."""
        basket = np.array(self.court.basket_xy(shot.side))
        for d, man in self.boxouts:
            self.target[d] = self.pos[man] + _unit(basket - self.pos[man]) * (self.cfg.body_contact_ft + 0.2)
        nxt = self.shooter_next
        if nxt is not None and self.t >= nxt[2] - 1e-9:
            s, tgt, _ = nxt
            self.locked[s] = False
            self.target[s] = tgt
            self.vmax[s] = self.cfg.run_speed_fts
            self.shooter_next = None

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
            w = (1.3 if closest < 3.0 else 0.6) if closest < c.block_max_defender_ft else 0.0
            if w > 0.0 and not self._block_reachable(team.other(), pos, side, shot_type):
                w = 0.0  # nobody can get a hand to the ball in the legal window (see _plan_block)
            p_block = min(0.5, c.block_rate[bkey] * w / max(c.block_defender_norm.get(bkey, 1.0), 1e-6))
        blocked = (not is_ft) and rng.random() < p_block
        p_open = min(0.98, p_make / max(1e-6, 1.0 - p_block))
        made = (not blocked) and rng.random() < p_open
        and_one = made and not is_ft and rng.random() < c.and_one_rate.get(str(value), 0.0)
        self.n_shots += 1
        shot = _Shot(shot_id=f"s{self.n_shots:04d}", slot=s, pid=pid, team=team, side=side, value=value,
                     shot_type="free_throw" if is_ft else shot_type, made=made,
                     blocked_by=self._closest(team.other(), pos) if blocked else None,
                     is_ft=is_ft, release_t=self.t, x=float(pos[0]), y=float(pos[1]), p_make_open=p_open, zone=zone)
        shot.team_rebound_oob = (not made) and (not blocked) and (not is_ft) and rng.random() < c.team_rebound_rate
        catch_shoot = (self.spell_from is not None and self.spell_dribbles == 0 and shot_type == "jumper")
        attrs = {"shot_type": shot.shot_type, "distance_ft": round(dist, 3), "closest_defender_ft": round(closest, 3),
                 "catch_and_shoot": bool(catch_shoot), "dribbles_before": int(self.spell_dribbles),
                 "basket": side, "zone": zone, "xfg": round(p_make, 4), "gather_x": gx, "gather_y": gy,
                 "sim_play": tag}
        self._emit(EventKind.SHOT_ATTEMPT, team=team, player=pid, x=shot.x, y=shot.y, value=value,
                   shot_id=shot.shot_id, attrs=attrs)
        if and_one:  # contact in the act of shooting; the whistle sounds, the basket counts + 1 FT
            shot.and_one = True
            self._whistle_foul(self._closest(team.other(), pos), s, "shooting", 1, and_one=True)
        if is_ft:
            self._tally(pid, fta=1)
        else:
            self._tally(pid, fga=1, fg3a=1 if value == 3 else 0)
        self._launch_shot(shot, pos, basket, z_rel)
        self.shot_pending = shot
        self.ball_live = not is_ft or self.last_ft
        return shot

    def _launch_shot(self, shot: _Shot, pos: np.ndarray, basket: np.ndarray, z_rel: float) -> None:
        rng = self.rng
        u = self._to_basket_dir(pos, shot.side)
        p0 = np.append(self._release_xy(pos, shot.side), max(z_rel, self._min_release_z(pos, shot.side)))
        block = None
        if shot.blocked_by is not None:  # (planned on the plain one-contact path the block rates were set on)
            path = self._plan_path(shot, u, p0, force="rim_out")
            block = self._plan_block(shot, p0, _ballistic(p0, path.p1, max(path.T, 0.05)), path.T)
            if block is None:  # nobody can get a hand on it in time: it plays out as an ordinary try
                shot.blocked_by = None
                shot.made = rng.random() < shot.p_make_open
        if block is None:
            path = self._plan_path(shot, u, p0)
        self.ball.pos[:] = p0
        self._launch(path.p1, path.T, None)
        if block is not None:
            tb, blocker = block
            shot.blocked_by, shot.path = blocker, "blocked"
            back = _rot(-u, math.radians(float(rng.uniform(-60, 60)))) * float(rng.uniform(8.0, 15.0))
            self.ball.impulses.append((tb, np.append(back, float(rng.uniform(-4.0, 6.0))), "block"))
            z_hit = float(p0[2] + self.ball.vel[2] * tb - 0.5 * GRAVITY_FTS2 * tb * tb)
            self._jump_for(blocker, "block", z_hit - self._reach_ft() + 0.2, self.t + tb)
            hit_xy = p0[:2] + self.ball.vel[:2] * tb
            self.target[blocker] = hit_xy if _dist(self.pos[blocker], hit_xy) > 1.6 else pos + u * 1.5
            self.vmax[blocker] = self.cfg.sprint_speed_fts
            self.auto[blocker] = False
            return
        # (contacts are exact in time, like the make: never missed between two frames at low fps)
        shot.path, shot.decide_t = path.kind, path.decide_t
        shot.rim_touched = path.touch
        self.ball.impulses.extend(path.impulses)

    # ------------------------------------------------------------------ shot paths (ring, board, air)

    def _rim_c(self, side: Side) -> np.ndarray:
        bx, by = self.court.basket_xy(side)
        return np.array([bx, by, self.rim_h])

    def _board_frame(self, side: Side) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(a point of the backboard face on the floor below the rim, the face normal into the court, the
        face's horizontal axis), all horizontal."""
        basket = np.array(self.court.basket_xy(side))
        n = _unit(self._spot(side, 1.0, 0.0) - basket)
        return basket - n * (self.court.basket_offset - self.court.backboard_offset), n, np.array([-n[1], n[0]])

    @staticmethod
    def _traj(p0: np.ndarray, v0: np.ndarray, ts: np.ndarray) -> np.ndarray:
        pts = p0[None, :] + v0[None, :] * ts[:, None]
        pts[:, 2] -= 0.5 * GRAVITY_FTS2 * ts * ts
        return pts

    @staticmethod
    def _vel_at(v0: np.ndarray, t: float) -> np.ndarray:
        return v0 - np.array([0.0, 0.0, GRAVITY_FTS2 * t])

    @staticmethod
    def _down_cross(z0: float, vz: float, z: float) -> float | None:
        """Flight time at which a ballistic height ``z0 + vz t - g t^2 / 2`` comes down through ``z``."""
        disc = vz * vz + 2.0 * GRAVITY_FTS2 * (z0 - z)
        if disc < 0.0:
            return None
        t = (vz + math.sqrt(disc)) / GRAVITY_FTS2
        return t if t > 1e-6 else None

    def _ring_gap(self, pts: np.ndarray, side: Side) -> np.ndarray:
        """Distance of ball centres to the ring (the rim circle) minus the ball radius (< 0: touching)."""
        C = self._rim_c(side)
        rho = np.hypot(pts[:, 0] - C[0], pts[:, 1] - C[1])
        return np.hypot(rho - self.court.rim_radius, pts[:, 2] - C[2]) - self.ball.r

    def _board_gap(self, pts: np.ndarray, side: Side) -> np.ndarray:
        """Ball centres' distance from the backboard face minus the ball radius, where the ball is within the
        board's extent (+inf elsewhere; < 0: touching or behind the face)."""
        c = self.cfg
        b0, n, t = self._board_frame(side)
        rel = pts[:, :2] - b0[None, :]
        dn, dt = rel @ n, rel @ t
        z_lo = self.rim_h - c.board_bottom_below_rim_ft
        inside = (np.abs(dt) <= c.board_width_ft / 2.0 + self.ball.r) & (pts[:, 2] >= z_lo - self.ball.r) & (
            pts[:, 2] <= z_lo + c.board_height_ft + self.ball.r) & (dn > -1.5)
        return np.where(inside, dn - self.ball.r, np.inf)

    def _seg_clear(self, p0: np.ndarray, v0: np.ndarray, t_end: float, side: Side, skip0: float = 0.0,
                   skip1: float = 0.0, plane: bool = True, n: int = 48) -> bool:
        """The ballistic segment touches neither the ring nor the board (outside ``skip0`` / ``skip1`` s at its
        ends: its own contacts) and, with ``plane``, never comes down through the rim plane inside the ring."""
        if t_end <= skip0 + skip1 + 1e-6:
            return True
        ts = np.linspace(skip0, t_end - skip1, n)
        pts = self._traj(p0, v0, ts)
        if float(self._ring_gap(pts, side).min()) < -0.02 or float(self._board_gap(pts, side).min()) < -0.02:
            return False
        if plane:
            C = self._rim_c(side)
            above = pts[:, 2] > C[2]
            cross = np.nonzero(above[:-1] & ~above[1:])[0]
            for k in cross:
                if math.hypot(pts[k + 1, 0] - C[0], pts[k + 1, 1] - C[1]) < self.court.rim_radius:
                    return False
        return True

    def _ring_point(self, side: Side, direction: np.ndarray, inside: bool = False) -> np.ndarray:
        """A ball-centre position touching the top of the ring in ``direction`` from its centre (outside the
        ring, or inside it for a ball coming down on the back of the ring)."""
        r, R = self.ball.r, self.court.rim_radius
        a = math.radians(float(self.rng.uniform(55.0, 80.0)))
        rho = R - r * math.cos(a) if inside else R + r * math.cos(a)
        C = self._rim_c(side)
        return np.array([C[0] + direction[0] * rho, C[1] + direction[1] * rho, C[2] + r * math.sin(a)])

    def _contact_dir(self, u: np.ndarray) -> tuple[np.ndarray, bool]:
        """Where a try meets the ring first: front rim (toward the shooter), side rim or back rim (the back
        of the ring, from inside); returns (direction from the centre, inside)."""
        r, ang = self.rng.random(), float(self.rng.uniform(-0.7, 0.7))
        if r < 0.5:
            return _rot(-u, ang), False
        if r < 0.8:
            return _rot(-u, math.copysign(math.pi / 2, ang) + ang * 0.3), False
        return _rot(u, ang * 0.6), True

    def _path_kind(self, shot: _Shot, u_pos: np.ndarray) -> str:
        c = self.cfg
        key = "jumper3" if shot.shot_type == "jumper" and shot.value == 3 else shot.shot_type
        table = (c.make_paths if shot.made else c.miss_paths).get(key) or {"swish" if shot.made else "rim_out": 1.0}
        kinds = list(table)
        w = np.array([table[k] for k in kinds], dtype=float)
        kind = kinds[int(self.rng.choice(len(kinds), p=w / w.sum()))]
        if kind in ("bank", "board_rim"):  # the glass needs an angle (and a sane distance)
            along, lat = self._along_lat(shot.side, u_pos)
            ang = math.degrees(math.atan2(abs(lat), along))
            if not (20.0 <= ang <= 75.0 and 1.0 <= math.hypot(along, lat) <= 18.0):
                kind = "rim_in" if shot.made else "rim_out"
        return kind

    def _plan_path(self, shot: _Shot, u: np.ndarray, p0: np.ndarray, force: str | None = None) -> _ShotPath:
        """How the try plays out (``SimConfig.make_paths`` / ``miss_paths``): the first target and flight time
        plus every later contact as an exact-time impulse. A drawn path that cannot be realised from this
        release (geometry checks: no unplanned ring / board touch, no pass down through the ring) is redrawn,
        falling back to a swish / a single ring contact."""
        kind = force or self._path_kind(shot, np.array([shot.x, shot.y]))
        lo, hi = _ENTRY_ANGLE_DEG.get(shot.shot_type, (40.0, 50.0))
        build = {"swish": self._path_swish, "rim_in": self._path_rim_in, "bank": self._path_bank,
                 "rim_out": self._path_rim_out, "rim_multi": self._path_rim_multi, "in_out": self._path_in_out,
                 "board_rim": self._path_board_rim, "air": self._path_air}[kind]
        for _ in range(12):
            theta = math.radians(float(self.rng.uniform(lo, hi)))
            path = build(shot, u, p0, theta)
            if path is not None:
                return path
        theta = math.radians(0.5 * (lo + hi))
        if shot.made:  # e.g. a dunk released right above the ring: through it, touching the ring on the way
            path = self._path_swish(shot, u, p0, theta, check=False)
            assert path is not None
            v0 = _ballistic(p0, path.p1, max(path.T, 0.05))
            if float(self._ring_gap(self._traj(p0, v0, np.linspace(0.0, path.T, 48)), shot.side).min()) < 0.0:
                path.kind, path.touch = "rim_in", True
            return path
        return self._path_rim_out(shot, u, p0, theta, check=False)  # type: ignore[return-value]

    def _make_impulse(self, v_in: np.ndarray) -> np.ndarray:
        return np.array([0.15 * v_in[0], 0.15 * v_in[1], max(0.5 * v_in[2], -8.0)])  # the net catches it

    def _path_swish(self, shot: _Shot, u: np.ndarray, p0: np.ndarray, theta: float,
                    check: bool = True) -> _ShotPath | None:
        R, r = self.court.rim_radius, self.ball.r
        jitter = _rot(np.array([1.0, 0.0]), float(self.rng.uniform(0, 2 * math.pi))) * float(
            self.rng.uniform(0.0, 0.85 if check else 0.6)) * (R - r)
        p1 = self._rim_c(shot.side) + np.append(jitter, 0.0)
        T = self._flight_time(p0, p1, theta)
        v0 = _ballistic(p0, p1, max(T, 0.05))
        if check:
            ts = np.linspace(0.0, T, 48)
            if float(self._ring_gap(self._traj(p0, v0, ts), shot.side).min()) < 0.005:
                return None  # it would touch the ring: not a swish
            if not self._seg_clear(p0, v0, T, shot.side, plane=False):
                return None
        return _ShotPath("swish", p1, T, [(T, self._make_impulse(self._vel_at(v0, T)), "make")])

    def _first_contact(self, shot: _Shot, u: np.ndarray, p0: np.ndarray, theta: float
                       ) -> tuple[np.ndarray, float, np.ndarray] | None:
        """First segment to a ring contact: (contact point, flight time, arrival velocity) or None."""
        d, inside = self._contact_dir(u)
        pc = self._ring_point(shot.side, d, inside)
        T = self._flight_time(p0, pc, theta)
        v0 = _ballistic(p0, pc, max(T, 0.05))
        if not self._seg_clear(p0, v0, T, shot.side, skip1=0.06):
            return None
        return pc, T, self._vel_at(v0, T)

    def _ring_hop(self, shot: _Shot, pc: np.ndarray, v_in: np.ndarray, across: bool = False
                  ) -> tuple[np.ndarray, float, np.ndarray] | None:
        """A bounce off the ring onto another ring point (``across``: to the far side, low through the
        cylinder above the ring: an in-and-out). Returns (bounce velocity, time, the next contact point)."""
        C = self._rim_c(shot.side)
        phi0 = math.atan2(pc[1] - C[1], pc[0] - C[0])
        dphi = math.radians(float(self.rng.uniform(150.0, 210.0) if across else self.rng.uniform(60.0, 180.0)))
        dphi *= float(self.rng.choice([-1.0, 1.0]))
        d2 = np.array([math.cos(phi0 + dphi), math.sin(phi0 + dphi)])
        back = float(d2 @ _unit(self._board_frame(shot.side)[1])) < -0.5  # toward the board: from inside
        pc2 = self._ring_point(shot.side, d2, inside=back or across)
        tau = float(self.rng.uniform(0.15, 0.3) if across else self.rng.uniform(0.12, 0.4))
        v = _ballistic(pc, pc2, tau)
        if across:  # low over the inside of the ring: the centre stays above the rim plane, apex <= 0.35 ft up
            if v[2] > math.sqrt(2.0 * GRAVITY_FTS2 * 0.35):
                return None
        if float(np.linalg.norm(v)) > 0.75 * float(np.linalg.norm(v_in)) + 1.0:
            return None  # the ring takes energy out of the ball
        if not self._seg_clear(pc, v, tau, shot.side, skip0=0.05, skip1=0.05):
            return None
        return v, tau, pc2

    def _drop_in(self, shot: _Shot, pc: np.ndarray, v_in: np.ndarray) -> tuple[np.ndarray, float] | None:
        """Off a ring contact down through the ring: (bounce velocity, time to the rim-plane crossing)."""
        C = self._rim_c(shot.side)
        R, r = self.court.rim_radius, self.ball.r
        toward = _unit(C[:2] - pc[:2])
        e = _rot(toward, float(self.rng.uniform(-0.6, 0.6)))
        x = np.append(C[:2] + e * float(self.rng.uniform(0.0, R - r - 0.1)), C[2])
        tau = float(self.rng.uniform(0.08, 0.35))
        v = _ballistic(pc, x, tau)
        if float(v[2]) - GRAVITY_FTS2 * tau > -1.0 or float(np.linalg.norm(v)) > 0.75 * float(np.linalg.norm(v_in)) + 1.0:
            return None
        if not self._seg_clear(pc, v, tau, shot.side, skip0=0.05, plane=False):
            return None
        return v, tau

    def _off_ring(self, shot: _Shot, pc: np.ndarray) -> np.ndarray | None:
        """Last contact -> out: a deflection that never comes back down through the ring or into the board."""
        for _ in range(8):
            v = self._deflection(shot, pc)
            t_end = self._down_cross(float(pc[2]), float(v[2]), self.rim_h - 1.0) or 0.6
            if self._seg_clear(pc, v, t_end, shot.side, skip0=0.05):
                return v
        return None

    def _path_rim_in(self, shot: _Shot, u: np.ndarray, p0: np.ndarray, theta: float) -> _ShotPath | None:
        first = self._first_contact(shot, u, p0, theta)
        if first is None:
            return None
        pc, T, v_in = first
        imps: list[tuple[float, np.ndarray, str]] = []
        t = T
        if self.rng.random() < 0.35:  # a rattle: a second contact before it drops
            hop = self._ring_hop(shot, pc, v_in)
            if hop is None:
                return None
            v, tau, pc2 = hop
            imps.append((t, v, "rim"))
            t, v_in, pc = t + tau, self._vel_at(v, tau), pc2
        drop = self._drop_in(shot, pc, v_in)
        if drop is None:
            return None
        v, tau = drop
        imps += [(t, v, "rim"), (t + tau, self._make_impulse(self._vel_at(v, tau)), "make")]
        return _ShotPath("rim_in", first[0], T, imps)

    def _path_rim_out(self, shot: _Shot, u: np.ndarray, p0: np.ndarray, theta: float,
                      check: bool = True) -> _ShotPath | None:
        if not check:  # the old single contact (front / side / back rim), always realisable
            d, inside = self._contact_dir(u)
            pc = self._ring_point(shot.side, d, inside)
            T = self._flight_time(p0, pc, theta)
            return _ShotPath("rim_out", pc, T, [(T, self._deflection(shot, pc), "rim")])
        first = self._first_contact(shot, u, p0, theta)
        if first is None:
            return None
        pc, T, _ = first
        v = self._off_ring(shot, pc)
        if v is None:
            return None
        return _ShotPath("rim_out", pc, T, [(T, v, "rim")])

    def _path_rim_multi(self, shot: _Shot, u: np.ndarray, p0: np.ndarray, theta: float) -> _ShotPath | None:
        first = self._first_contact(shot, u, p0, theta)
        if first is None:
            return None
        pc, t, v_in = first
        imps: list[tuple[float, np.ndarray, str]] = []
        for _ in range(1 + int(self.rng.random() < 0.35)):  # 2-3 contacts
            hop = self._ring_hop(shot, pc, v_in)
            if hop is None:
                return None
            v, tau, pc2 = hop
            imps.append((t, v, "rim"))
            t, v_in, pc = t + tau, self._vel_at(v, tau), pc2
        v = self._off_ring(shot, pc)
        if v is None:
            return None
        imps.append((t, v, "rim"))
        return _ShotPath("rim_multi", first[0], first[1], imps)

    def _path_in_out(self, shot: _Shot, u: np.ndarray, p0: np.ndarray, theta: float) -> _ShotPath | None:
        d, _ = self._contact_dir(u)
        if float(d @ u) > 0.3:  # it goes in from the front or side rim
            d = -d
        pc = self._ring_point(shot.side, d)
        T = self._flight_time(p0, pc, theta)
        v0 = _ballistic(p0, pc, max(T, 0.05))
        if not self._seg_clear(p0, v0, T, shot.side, skip1=0.06):
            return None
        hop = self._ring_hop(shot, pc, self._vel_at(v0, T), across=True)
        if hop is None:
            return None
        v, tau, pc2 = hop
        out = self._off_ring(shot, pc2)
        if out is None:
            return None
        return _ShotPath("in_out", pc, T, [(T, v, "rim"), (T + tau, out, "rim")])

    def _glass(self, shot: _Shot, p0: np.ndarray, theta: float) -> tuple[np.ndarray, float, np.ndarray] | None:
        """A first segment onto the backboard (on the shooter's side of the square): (board point, flight
        time, velocity off the glass) or None."""
        c, rng = self.cfg, self.rng
        b0, n, tb = self._board_frame(shot.side)
        lat = float((p0[:2] - b0) @ tb)
        yb = math.copysign(float(rng.uniform(0.2, 1.3)), lat)
        zb = self.rim_h + float(rng.uniform(0.4, 1.8))
        pb = np.append(b0 + n * self.ball.r + tb * yb, zb)
        T = self._flight_time(p0, pb, theta)
        v0 = _ballistic(p0, pb, max(T, 0.05))
        v_in = self._vel_at(v0, T)
        vn = float(v_in[:2] @ n)
        if vn >= -1.0 or not self._seg_clear(p0, v0, T, shot.side, skip1=0.06):
            return None
        e_n, e_t = float(rng.uniform(*c.board_restitution)), float(rng.uniform(*c.board_tangent_keep))
        tang = v_in - np.append(n * vn, 0.0)
        return pb, T, np.append(-n * vn * e_n, 0.0) + tang * e_t

    def _path_bank(self, shot: _Shot, u: np.ndarray, p0: np.ndarray, theta: float) -> _ShotPath | None:
        """Off the glass and in: a board point whose bounce comes down through the ring without touching it
        (searched over board points and entry angles)."""
        R, r = self.court.rim_radius, self.ball.r
        C = self._rim_c(shot.side)
        for _ in range(16):
            glass = self._glass(shot, p0, theta)
            if glass is None:
                theta = math.radians(float(self.rng.uniform(30.0, 60.0)))
                continue
            pb, T, v = glass
            tc = self._down_cross(float(pb[2]), float(v[2]), C[2])
            if tc is None:
                continue
            x = pb + v * tc
            if math.hypot(x[0] - C[0], x[1] - C[1]) > R - r - 0.03:
                continue
            ts = np.linspace(0.05, tc, 40)
            if float(self._ring_gap(self._traj(pb, v, ts), shot.side).min()) < 0.005 or float(
                    self._board_gap(self._traj(pb, v, ts), shot.side).min()) < -0.02:
                continue
            return _ShotPath("bank", pb, T, [(T, v, "board"), (T + tc, self._make_impulse(self._vel_at(v, tc)), "make")])
        return None

    def _path_board_rim(self, shot: _Shot, u: np.ndarray, p0: np.ndarray, theta: float) -> _ShotPath | None:
        """Off the glass onto the ring and out (the bounce's first ring touch is the contact)."""
        for _ in range(16):
            glass = self._glass(shot, p0, theta)
            if glass is None:
                theta = math.radians(float(self.rng.uniform(30.0, 60.0)))
                continue
            pb, T, v = glass
            ts = np.linspace(0.04, 0.7, 120)
            pts = self._traj(pb, v, ts)
            gap = self._ring_gap(pts, shot.side)
            hit = np.nonzero(gap <= 0.0)[0]
            if not len(hit):
                continue
            k = int(hit[0])
            if not self._seg_clear(pb, v, float(ts[k]), shot.side, skip0=0.04, skip1=0.04):
                continue  # through the ring before it touches it: that is a bank make, not this
            pc = pts[k]
            out = self._off_ring(shot, pc)
            if out is None:
                continue
            return _ShotPath("board_rim", pb, T, [(T, v, "board"), (T + float(ts[k]), out, "rim")])
        return None

    def _path_air(self, shot: _Shot, u: np.ndarray, p0: np.ndarray, theta: float) -> _ShotPath | None:
        """An air ball: short of the ring or wide of it, touching neither the ring nor the board."""
        R, r, rng = self.court.rim_radius, self.ball.r, self.rng
        C = self._rim_c(shot.side)
        perp = np.array([-u[1], u[0]])
        if rng.random() < 0.5:  # short
            xy = C[:2] - u * (R + r + float(rng.uniform(0.3, 2.0))) + perp * float(rng.uniform(-0.6, 0.6))
        else:  # wide
            xy = C[:2] + perp * float(rng.choice([-1.0, 1.0])) * (R + r + float(rng.uniform(0.2, 1.2))) + \
                u * float(rng.uniform(-0.6, 0.3))
        p1 = np.append(xy, C[2])
        T = self._flight_time(p0, p1, theta)
        v0 = _ballistic(p0, p1, max(T, 0.05))
        t_dec = self._down_cross(float(p0[2]), float(v0[2]), C[2] - 0.3)  # (it has passed the ring: decided)
        if t_dec is None or t_dec <= T:
            return None
        t_floor = self._down_cross(float(p0[2]), float(v0[2]), r) or t_dec
        ts = np.linspace(0.0, t_floor, 80)
        pts = self._traj(p0, v0, ts)
        if float(self._ring_gap(pts, shot.side).min()) < 0.08 or float(self._board_gap(pts, shot.side).min()) < 0.02:
            return None
        return _ShotPath("air", p1, T, [], decide_t=t_dec)

    def _block_reachable(self, defense: Team, pos: np.ndarray, side: Side, shot_type: str) -> bool:
        """At-rim tries: can any defender get a hand to the ball near its release point within
        ``block_close_s``? A cheap pre-check of :meth:`_plan_block` (jumpers are not checked)."""
        c = self.cfg
        if shot_type not in ("layup", "dunk", "hook", "tip"):
            return True  # jumpers: the round-1 contest model (see _plan_block)
        t = c.block_close_s
        rel = self._release_xy(pos, side)
        reach = c.block_hand_ft + self._run_reach(t)
        return any(_dist(self.pos[q], rel) <= reach for q in self._slots(defense))

    def _plan_block(self, shot: _Shot, p0: np.ndarray, v0: np.ndarray, T: float) -> tuple[float, int] | None:
        """When and by whom a blocked try is blocked: any time before rim contact while the ball is still
        rising (on its downward flight above the ring it would be goaltending), within the blocker's reach
        (standing reach + his best block jump). Jumpers are contested at the release (``block_jumper_s``, the
        closest defender, reach not resolved: see below);
        layups, dunks, hooks and tips can be blocked anywhere up to the apex, near the rim too, and on those a
        rotating help defender is the blocker in ``block_help_share`` of blocks when he can get there in time
        from the release (hand reach ``block_hand_ft``). Returns (flight time of the touch, blocker slot), or
        None when no defender can get a hand on it in the window (the try then plays out unblocked; the
        per-bin block rate is calibrated for that, ``block_defender_norm``)."""
        c, rng, g = self.cfg, self.rng, GRAVITY_FTS2
        t_apex = float(v0[2]) / g if v0[2] > 0 else 0.0
        top = self._reach_ft() + c.jump_ft["block"][1] - 0.2
        def z_at(t: float) -> float:
            return float(p0[2] + v0[2] * t - 0.5 * g * t * t)
        hi = min(t_apex, T - 0.05)
        while hi > 0.07 and z_at(hi) > top:  # higher than any block jump: earlier
            hi -= 0.01
        close = shot.shot_type in ("layup", "dunk", "hook", "tip")
        # (a try released inside the rim cylinder, above the ring, starts down at once: its only legal block is
        # at the hand, so hi stays at the earliest touch, 0.07 s; later touches on the way down would be
        # goaltending)
        hi = max(hi, 0.07)
        if not close:
            # A jumper keeps the round-1 contest model: blocked at the release by the closest defender
            # (within block_max_defender_ft). Known approximation: the sim has no flying closeouts, so a 4-6 ft
            # contest-block is not physically resolved; requiring reach here would remove 3-pt blocks entirely
            # and break the sourced per-bin block rates.
            lo_j, hi_j = c.block_jumper_s
            return float(rng.uniform(lo_j, max(lo_j, min(hi_j, t_apex - 0.01)))), int(shot.blocked_by)  # type: ignore[arg-type]
        grid = np.linspace(0.07, max(0.07, hi), 16).tolist()
        hit = lambda t: p0[:2] + v0[:2] * t  # noqa: E731
        short = lambda q, t: _dist(self.pos[q], hit(t)) - c.block_hand_ft - self._run_reach(t)  # noqa: E731
        blocker = shot.blocked_by
        assert blocker is not None
        if close and rng.random() < c.block_help_share:
            helpers = [(q, t) for q in self._slots(shot.team.other()) if q != blocker for t in grid if short(q, t) <= 0]
            if helpers:
                blocker = helpers[int(rng.integers(len(helpers)))][0]
        ok = [t for t in grid if short(blocker, t) <= 0]
        if not ok:
            return None
        tb = ok[int(rng.integers(len(ok)))]  # a reachable moment of the legal window
        return tb, blocker

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
                self._reach_hand(d, after=1)
                self._emit(EventKind.BLOCK, team=shot.team.other(), player=self._pid(d), other=shot.pid, x=bx, y=by_,
                           shot_id=shot.shot_id)
                self._tally(self._pid(d), blk=1)
                self._emit_result(shot, made=False)
                self.last_touch = shot.team.other()
                return "blocked"
            if "make" in b.fired and shot.made:
                self._emit_result(shot, made=True)
                return "made"
            if "board" in b.fired:
                shot.board_touched = True
            if "rim" in b.fired:
                shot.rim_touched = True
                self.sc_running = False
                self.sc_expired = False
            if shot.rim_touched and not b.impulses and self._left_rim_zone(shot):  # (after its last contact)
                self._emit_result(shot, made=False)
                return "missed"
            if shot.decide_t is not None and b.flight_t >= shot.decide_t - 1e-9:  # an air ball: past the ring
                self._emit_result(shot, made=False)
                return "missed"
            if b.mode != "flight":  # safety: should not happen
                self._emit_result(shot, made=False)
                return "missed"
            self._track_rebound_positions(shot)
            yield

    def _emit_result(self, shot: _Shot, made: bool) -> None:
        self.shot_pending = None
        kind = EventKind.SHOT_MADE if made else EventKind.SHOT_MISSED
        self._emit(kind, team=shot.team, player=shot.pid, x=shot.x, y=shot.y, value=shot.value if made else 0,
                   shot_id=shot.shot_id, attrs={"basket": shot.side, "path": shot.path,
                                                "rim_contact": bool(shot.rim_touched),
                                                "backboard": bool(shot.board_touched)})
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
            if shot.and_one:
                return self._phase_dead_ball(shot.team, None, "foul", ft_shooter=shot.slot, n_ft=1)
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
        p_off = c.oreb_rate_ft if shot.is_ft else (c.oreb_rate_block if after_block else self._oreb_rate(shot))
        land = self._landing_point()
        rebounder: int | None = None
        if not shot.team_rebound_oob:
            team_r = shot.team if rng.random() < p_off else shot.team.other()
            rebounder = self._choose_rebounder(team_r, land)
        boxers, chasers = self._crash_boards(shot, land, rebounder)
        self.boxouts = boxers
        start = self.t
        while True:
            for d, man in boxers:  # box-out: stay between the man and the basket, in body contact
                basket = np.array(self.court.basket_xy(shot.side))
                self.target[d] = self.pos[man] + _unit(basket - self.pos[man]) * (self.cfg.body_contact_ft + 0.2)
            # contest the ball: crowd the landing spot, a step behind the rebounder who wins the race (control
            # goes to the first to arrive; the OREB prior picked him), never closer to the ball than him
            lead = _dist(self.pos[rebounder], self.ball.pos) if rebounder is not None else 0.0
            ring = float(np.clip(lead + 0.3, 1.8, 3.0))
            for q in chasers:
                self.target[q] = self.ball.pos[:2] + _unit(self.pos[q] - self.ball.pos[:2]) * ring
            if rebounder is not None and self.ball.pos[2] < 9.0:  # the winner has the inside position
                inside = min(lead, 2.5)
                for q in range(10):
                    if q != rebounder and q not in chasers and _dist(self.pos[q], self.ball.pos) < inside:
                        self.target[q] = self.ball.pos[:2] + _unit(self.pos[q] - self.ball.pos[:2]) * (inside + 0.3)
            reacted = not after_block or self.t - start >= c.block_loose_s - 1e-9
            if rebounder is not None:
                if reacted and self._reach(rebounder, 2.5, 9.5):
                    self._catch_jump(rebounder)
                    self._reach_hand(rebounder)
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

    def _oreb_rate(self, shot: _Shot) -> float:
        """P(offensive rebound) after a missed field goal, by the shot's pbpstats zone (``oreb_zone``)."""
        c = self.cfg
        d = self.court.distance_to_basket(shot.x, shot.y, shot.side)
        if shot.value == 3:
            zone = "c3" if shot.zone == "corner_three" else "ab3"
        else:
            zone = "rim" if d < 4.0 else ("short_mid" if d < 14.0 else "long_mid")
        return float(np.clip(c.oreb_rate * c.oreb_zone.get(zone, c.oreb_zone_ref) / c.oreb_zone_ref, 0.0, 1.0))

    def _reach_hand(self, s: int, after: int = 0) -> None:
        """Slot ``s`` gets a hand to the ball on this frame: his box includes the reaching arm over the last
        ``hand_reach_s`` (and ``after`` more frames)."""
        k = max(1, int(round(self.cfg.hand_reach_s * self.cfg.fps)))
        self.hand_marks.append((max(0, self.frame - k), self.frame + after, s))

    def _catch_jump(self, s: int) -> None:
        """A ball controlled above the head is caught in the air (jump apex at the catch)."""
        z_top = float(self.ball.pos[2]) + self.ball.r
        if z_top > self.cfg.player_height_ft:
            self._jump_for(s, "rebound", z_top - self._reach_ft() + 0.2, self.t)

    def _choose_rebounder(self, team: Team, land: np.ndarray) -> int:
        w = []
        slots = list(self._slots(team))
        for s in slots:
            role_w = {"C": 1.6, "F": 1.2, "G": 0.7}[self.role[self.occ[s]]]
            w.append(role_w * math.exp(-_dist(self.pos[s], land) / 6.0))
        w_arr = np.array(w)
        return slots[int(self.rng.choice(len(slots), p=w_arr / w_arr.sum()))]

    def _crash_boards(self, shot: _Shot, land: np.ndarray, rebounder: int | None
                      ) -> tuple[list[tuple[int, int]], list[int]]:
        """Rebound positioning: the rebounder goes for the ball; 1-3 of the nearest other players from
        both teams contest the landing spot (they lose the race but crowd the ball); defenders box out
        crashing offensive players; the other offensive bigs crash a ring round the landing point and
        offensive guards get back. Returns (box-out pairs (defender, man), contesting slots)."""
        c, rng = self.cfg, self.rng
        basket = np.array(self.court.basket_xy(shot.side))
        others = sorted((q for q in range(10) if q != rebounder), key=lambda q: _dist(self.pos[q], land))
        lo, hi = c.rebound_contest_n
        chasers = [q for q in others[: int(rng.integers(lo, hi + 1))] if _dist(self.pos[q], land) < 15.0]
        boxers: list[tuple[int, int]] = []
        for s in range(10):
            self.vmax[s] = c.run_speed_fts
            if s == rebounder:
                self.target[s] = land
                self.vmax[s] = c.sprint_speed_fts
                continue
            if s in chasers:
                self.vmax[s] = c.sprint_speed_fts
                continue
            off = self.slot_team[s] is shot.team
            if off and self.role[self.occ[s]] == "G":
                self.target[s] = self._spot(shot.side, 28.0, float(rng.uniform(-10, 10)))  # get back
            elif off:
                self.target[s] = land + _rot(np.array([4.0, 0.0]), float(rng.uniform(0, 2 * math.pi)))
            else:
                man = self._opp(s)
                crashing = self.role[self.occ[man]] != "G" and man != rebounder and man not in chasers
                if crashing and _dist(self.pos[man], basket) < 20.0:
                    boxers.append((s, man))
                else:
                    self.target[s] = self._sink_spot(basket, s)
        return boxers, chasers

    def _rebound_control(self, shot: _Shot, s: int) -> Phase:
        """REBOUND by slot ``s`` now; returns the next phase."""
        team = self.slot_team[s]
        offensive = team is shot.team
        pid = self._pid(s)
        self.loose_after = None
        self._hold(s)
        self._start_spell(None)
        # he secures the ball and lands before an outlet (rebound - chin - pivot - outlet)
        self.secure = (s, self.control_t, max(self.t, self._land_t(s)) + float(self.rng.uniform(*self.cfg.rebound_secure_s)))
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
        self.force_weak = bool(self.scheme_rng.random() < self.cfg.force_weak_share)
        self._emit(EventKind.POSSESSION, team=team, player=self._pid(s), x=float(self.pos[s, 0]),
                   y=float(self.pos[s, 1]), attrs={"cause": "defensive_rebound"})
        self._reset_shot_clock(self.rules.shot_clock_s)
        self.sc_running = self.sc is not None

    def _phase_putback(self, s: int) -> Phase:
        team = self.slot_team[s]
        release = self._release_spot(team, s, "0_3")
        quick = self.ball.pos[2] > 7.0  # controlled high (the ball is still where it was caught): tip it
        if quick:  # a tip goes straight back up from the catch (no carrying the ball overhead to a spot)
            self._hold(s, z=float(self.ball.pos[2]))
            return (yield from self._act_shoot(s, "tip", gather_xy=self.pos[s].copy(), release_target=None,
                                               gather_s=0.12, tag="putback"))
        return (yield from self._act_shoot(s, "layup", gather_xy=self.pos[s].copy(), release_target=release,
                                           gather_s=0.4, tag="putback"))

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
        stolen = victim is not None and team is not self.possession_team
        if stolen:
            self._steal(s, victim, "steal")
        else:
            self._gain(s, cause)
        return self._continue_with_ball(team, transition=stolen)

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
        team = self.possession_team or self.last_touch or Team.HOME
        if self.last_touch is not None and self.last_touch is not team:
            # the defence knocked it out: no turnover, no steal (note §2a); the offence throws in with
            # the unexpired shot clock
            self._emit_pass(False)
            return self._phase_dead_ball(team, spot, "out_of_bounds")
        last = self.pass_info.passer if self.pass_info is not None else (
            self._pid(self.holder) if self.holder is not None else None)
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
                         ft_shooter: int | None = None, n_ft: int = 2, one_and_one: bool = False) -> Phase:
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
            return self._phase_free_throws(ft_shooter, n_ft, one_and_one=one_and_one)
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

    def _shuffle_lane(self, shooter: int) -> None:
        """Between free throws the lane players are not statues: each steps up to ``ft_lane_shuffle_ft``
        within his lane spot (the shooter stays on the line)."""
        r = self.cfg.ft_lane_shuffle_ft
        for q in range(10):
            if q != shooter and r > 0.0:
                self.target[q] = self.target[q] + self.vis_rng.uniform(-r, r, 2)

    def _ft_lane_entry(self, shooter: int) -> None:
        """Players in the free-throw lane spaces may enter the lane once the ball leaves the shooter's hands
        (FIBA OBR Art. 43 / NBA Rule 9; not yet in the rules note: to verify): lane defenders box out the
        offensive lane players, who step in toward the rim (they used to stand still through the whole flight
        of the last free throw). The shooter stays behind the line."""
        team = self.slot_team[shooter]
        basket = self._basket(team)
        lane = [q for q in range(10) if q != shooter and _dist(self.pos[q], basket) < 13.5]
        off = [q for q in lane if self.slot_team[q] is team]
        dfn = [q for q in lane if self.slot_team[q] is not team]
        self.boxouts = []
        for o in sorted(off, key=lambda q: _dist(self.pos[q], basket)):
            free = [d for d in dfn if all(d != b for b, _ in self.boxouts)]
            if free:
                self.boxouts.append((min(free, key=lambda d: _dist(self.pos[d], self.pos[o])), o))
            self.target[o] = basket + _unit(self.pos[o] - basket) * float(self.rng.uniform(3.0, 5.0))
        for d in dfn:
            if all(d != b for b, _ in self.boxouts):
                self.target[d] = basket + _unit(self.pos[d] - basket) * float(self.rng.uniform(2.5, 4.0))
        for q in lane:
            self.locked[q] = False
            self.vmax[q] = self.cfg.run_speed_fts

    def _phase_free_throws(self, shooter: int, n: int, one_and_one: bool = False) -> Phase:
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
            yield from self._await_subs()
            yield from self._wait_until(lambda: self._arrived(range(10), 0.6), 15.0)
            have_ball = self.t
            yield from self._wait_until(lambda: self.t - self.still_since[shooter] >= c.ft_set_s and
                                        self.t - have_ball >= 0.6, 6.0)
            yield from self._wait(float(self.rng.uniform(*c.ft_routine_s)))  # his routine (dribbles, breath)
            self._hold(shooter)
            self._start_spell(None)
            z = float(self.rng.uniform(*c.ft_release_z_ft))
            yield from self._raise_ball(shooter, z, c.ft_gather_s)  # shooting motion: no jump at the release
            shot = self._release(shooter, "free_throw", self.pos[shooter].copy(), z, is_ft=True)
            if self.last_ft or one_and_one:  # a miss is live: the lane players enter at the release
                self._ft_lane_entry(shooter)
            result = yield from self._shot_flight(shot)
            if k < n and one_and_one and result != "made":  # a missed front end of a one-and-one is live
                self.last_ft = True
                self.locked[:] = False
                self.loose_after = shot
                self.ball_live = True
                return self._phase_rebound(shot)
            if k < n:
                b = self.ball
                yield from self._wait_until(lambda: b.bounced or b.rolling or b.pos[2] < 1.0, 2.0)
                self.ball_live = False
                self.loose_after = None
                self._shuffle_lane(shooter)  # lane players shift their weight / spots a little
                yield from self._wait(float(self.rng.uniform(*c.ft_return_s)))  # the official retrieves it
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
                # The substitute waits in the substitution box by the scorer's table (a visible person)
                # while the outgoing player walks off to his bench; they are two people, never one body
                # changing jersey (design: NBA / FIBA substitution procedure, positions unverified).
                k = sum(1 for t in self.sub_tasks if t.team is team)
                sign = -1.0 if team is Team.HOME else 1.0
                mid = self.court.length / 2.0
                box = np.array([mid + sign * (4.0 + 2.5 * k), -2.5])
                seat = np.array([mid + sign * (14.0 + 2.5 * k), -5.0])
                self.sub_tasks.append(_SubTask(team, s, int(self.occ[s]), inn, seat, self.t, box))
                self.override[s] = seat

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
            # The substitute enters once the outgoing player has left the floor near his seat (a safety:
            # anywhere off the court after 20 s). He is never taken off from inside the court lines (a 5 s
            # cutoff used to teleport him to the table, 11% of substitutions): restarts wait for pending
            # substitutions instead (_await_subs).
            out = not self.court.in_bounds(*self.pos[s])
            off_floor = out and (_dist(self.pos[s], task.spot) <= 3.0 or self.t - task.start_t >= 20.0)
            # ... and his entry spot is clear (another outgoing player may be walking past the table)
            clear = all(_dist(task.box, self.pos[q]) >= self.cfg.body_contact_ft + 0.1 for q in range(10) if q != s)
            if not (off_floor and clear):
                self.vmax[s] = max(self.vmax[s], self.cfg.jog_speed_fts)
                continue
            out_pid, in_pid = self.pid[task.out_idx], self.pid[task.in_idx]
            self.occ[s] = task.in_idx
            self.jump_t0[s], self.jump_v0[s], self.arms_until[s] = -1e9, 0.0, -1e9
            self.stint[task.in_idx] = 0.0
            self.rest[task.out_idx] = 0.0
            self.override[s] = None
            self.pos[s] = task.box.copy()
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
        c = self.cfg
        B = self._basket(off)
        if self.ball.mode != "flight":  # defenders react to where the ball is held, not to passes in the air
            self.ai_ball_xy = self.ball.pos[:2].copy()
        ball_xy = self.ai_ball_xy
        L, W = self.court.length, self.court.width
        for s in self._slots(off):
            d = self._opp(s)
            if not self.auto[d] or self.t < self.react_until[d]:
                continue
            m = self.pos[s]
            to_b = B - m
            dist_b = math.hypot(*to_b)
            u = to_b / dist_b if dist_b > 1e-6 else np.zeros(2)
            dball = _dist(m, ball_xy)
            on_ball = s == self.holder
            a, b, lo, hi = c.help_gap_ft
            gap = c.on_ball_gap_ft if on_ball else float(np.clip(a + b * dball, lo, hi))
            along = min(gap, 0.6 * dist_b)  # never behind the basket
            least = c.on_ball_gap_ft if on_ball else 2.0  # never on top of his man (body contact is 1.5 ft)
            tgt = m + u * along
            if on_ball and self.force_weak:  # shade his strong hand (force him to his weak hand)
                tgt = tgt + np.array([u[1], -u[0]]) * self.strong[int(self.occ[s])] * c.force_weak_ft
            if along < least:  # under the rim: slide to the side he is on instead of standing on the man
                perp = np.array([-u[1], u[0]])
                sign = 1.0 if float((self.pos[d] - m) @ perp) >= 0.0 else -1.0
                tgt = tgt + perp * sign * math.sqrt(least * least - along * along)
            if not on_ball:  # sag toward the ball (help side)
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
        self.r_jump = np.zeros((n, 10))  # feet height above the floor (ft)
        self.r_arms = np.zeros((n, 10), dtype=bool)  # arms raised (shot follow-through, block, rebound)
        self.r_balltop = np.zeros((n, 10))  # top of a ball held by this slot (raised-ball bbox top)
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
        self.r_waiting: list[list[tuple[str, str, float, float]]] = [[] for _ in range(n)]
        self.hand_marks: list[tuple[int, int, int]] = []  # (first frame, last frame, slot) of a reaching hand

    def _record(self, i: int) -> None:
        self.r_jump[i] = np.maximum(self.r_jump[i], self._jump_heights(self.t))
        self.r_arms[i] |= self.t <= self.arms_until + 1e-9
        if self.ball.mode in ("held", "dribble") and self.ball.holder is not None:  # (a caught ball being
            self.r_balltop[i, self.ball.holder] = float(self.ball.pos[2]) + self.ball.r  # carried down counts)
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
        self.r_waiting[i] = [(self.pid[t.in_idx], t.team.value, float(t.box[0]), float(t.box[1]))
                             for t in self.sub_tasks]

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
                         starters={t: list(v) for t, v in self.starters.items()}, waiting=self.r_waiting,
                         feet_z=np.round(self.r_jump, 4).tolist(), hands=self.r_hands)

    def _figure_tops(self) -> np.ndarray:
        """(n, 10) top of each player's image figure: head height plus the jump, raised arms (standing
        reach plus the jump) and a ball held above the head (its top)."""
        c = self.cfg
        top = c.player_height_ft + self.r_jump
        reach = c.standing_reach_ratio * c.player_height_ft
        top = np.where(self.r_arms, np.maximum(top, reach + self.r_jump), top)
        return np.maximum(top, self.r_balltop)

    def _build_frames(self, n: int) -> list[FrameState]:
        c, cam = self.cfg, self.camera
        boxes = cylinder_boxes(cam, self.r_pos.reshape(-1, 2), self._figure_tops().reshape(-1), c.player_radius_ft,
                               base=self.r_jump.reshape(-1)).reshape(n, 10, 4)
        ball_uv = cam.project(self.r_ball)
        ball_box = sphere_boxes(cam, self.r_ball, self.ball.r)
        # a ball held above the shoulders is in the raised hands: the person box includes it
        raised = self.r_balltop > c.shoulder_ratio * c.player_height_ft
        if raised.any():
            ii, ss = np.nonzero(raised)
            bb = ball_box[ii]
            ok = np.all(np.isfinite(bb), axis=1)
            ii, ss, bb = ii[ok], ss[ok], bb[ok]
            boxes[ii, ss, 0] = np.minimum(boxes[ii, ss, 0], bb[:, 0])
            boxes[ii, ss, 1] = np.minimum(boxes[ii, ss, 1], bb[:, 1])
            boxes[ii, ss, 2] = np.maximum(boxes[ii, ss, 2], bb[:, 2])
            boxes[ii, ss, 3] = np.maximum(boxes[ii, ss, 3], bb[:, 3])
        hands = self._hand_points(n)
        for (i, s_), (hx, hy, hz) in hands.items():
            hb = sphere_boxes(cam, np.array([[hx, hy, hz]]), c.hand_radius_ft)[0]
            if np.all(np.isfinite(hb)):
                boxes[i, s_, 0] = min(boxes[i, s_, 0], hb[0])
                boxes[i, s_, 1] = min(boxes[i, s_, 1], hb[1])
                boxes[i, s_, 2] = max(boxes[i, s_, 2], hb[2])
                boxes[i, s_, 3] = max(boxes[i, s_, 3], hb[3])
        self.r_hands: list[list[tuple[str, float, float, float]]] = [[] for _ in range(n)]
        for (i, s_), (hx, hy, hz) in sorted(hands.items()):
            self.r_hands[i].append((self.pid[int(self.r_occ[i, s_])], float(hx), float(hy), float(hz)))
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

    def _hand_points(self, n: int) -> dict[tuple[int, int], tuple[float, float, float]]:
        """(frame, slot) -> the reaching hand: at the ball, at most ``arm_length_ratio`` x stature from the
        shoulder (``shoulder_ratio`` x stature above the feet)."""
        c = self.cfg
        arm, h = c.arm_length_ratio * c.player_height_ft, c.player_height_ft
        out: dict[tuple[int, int], tuple[float, float, float]] = {}
        for lo, hi, s in self.hand_marks:
            for i in range(lo, min(hi, n - 1) + 1):
                if self.r_occ[i, s] != self.r_occ[min(hi, n - 1), s]:
                    continue  # (a substitution inside the window)
                sh = np.array([self.r_pos[i, s, 0], self.r_pos[i, s, 1], c.shoulder_ratio * h + self.r_jump[i, s]])
                d = self.r_ball[i] - sh
                dn = float(np.linalg.norm(d))
                hand = self.r_ball[i] if dn <= arm else sh + d * (arm / dn)
                out[(i, s)] = (float(hand[0]), float(hand[1]), float(hand[2]))
        return out

    def _rim_observations(self) -> list[RimObservation]:
        out = []
        for side in ("left", "right"):
            bx, by = self.court.basket_xy(side)
            box = self.camera.bbox_of_circle(bx, by, self.rim_h, self.court.rim_radius)
            out.append(RimObservation(side=side, bbox=box, confidence=1.0))
        return out
