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

Monocular evidence rules (one camera: boxes overlap, the ball hides in the hands): a change of team
control is a STEAL only once the new team's control is corroborated (``_PendingSteal``) and the releaser
had shown control himself (``_corroborated``); a releaser who never did is a wrong body, spliced out
(``_correct_phantom``). A track whose label falls back to a provisional id keeps its roster identity
(``_canonical_ids``). A blocked shot recognised late is found in the buffered image track
(``_retro_block``), and a score delta matching no seen shot becomes an inferred attempt + make
(``_unseen_make``). Dead balls recognised from a clock stop are dated at the stop.

Event semantics follow ARCHITECTURE "Shared event semantics" and the FIBA Statisticians' Manual
rules in docs/research/basketball_rules_stats.md §2a. Every tunable constant is a field of
``EventEngineConfig`` with its source and status (sourced / unverified / tuned).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import re
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
from .possession import G_FTS2, BallGate, BallObs, ControlChange, HolderTracker, PlayerHistory, RayModel, ball_obs, keep_box
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
    holder_img_raised_keep: bool = True  # image: the holder keeps a ball raised overhead while the ray model puts it within holder_reach_z_ft at him (a rising shooter's ball used to go to a farther player on the same viewing ray); tuned on sim train seeds
    holder_min_score: float = 0.1  # minimum candidate score; tuned
    holder_switch_margin: float = 0.25  # current holder's score bonus (no flicker); tuned
    holder_radius_ft: float = 1.6  # 0.5 m holder radius (RESEARCH §7A.2); unverified
    holder_keep_radius_ft: float = 3.0  # keep radius for dribble excursions; tuned
    holder_max_z_ft: float = 7.5  # a held ball is below this height (spec); unverified
    holder_reach_z_ft: float = 12.0  # 3-D: above holder_max_z_ft the holder keeps a raised ball up to this height until it is first in free flight (standing reach ~1.33 x stature + a jump, sim anthropometrics); unverified
    holder_freefall_min_z_ft: float = 4.5  # 3-D: above dribble height (catches and dribbles near the floor are partly ballistic) ... ; tuned
    holder_loose_free_min_z_ft: float = 3.5  # 3-D, loose ball after a miss / block (nobody dribbles it): free flight above about waist height (~0.53 x stature) is nobody's yet; tuned on sim train seeds
    holder_freefall_tol_fts2: float = 8.0  # ... a ball with |a_z + g| below this is in free flight, nobody gains it; tuned
    holder_acquire_max_speed_fts: float = 15.0  # a catch slows the ball to the catcher's speed; tuned
    holder_lift_min_accel_fts2: float = -16.0  # 3-D, loose ball: a ball whose vertical acceleration is above this (supported, not falling at -g) is in someone's hands, so its vertical speed (a rebounder ripping a low catch up overhead, ~15 ft/s) does not count against the catch; half of g; tuned on sim train seeds
    holder_rel_speed_scale_fts: float = 10.0  # 3-D: score -= relative ball-player speed / this; tuned
    holder_rel_vertical_weight: float = 0.5  # weight of vertical motion in that speed during a pass (catch-then-dribble); tuned
    holder_rel_vertical_weight_loose: float = 1.0  # ... for a loose ball (rebounds, fumbles); tuned
    holder_rel_vertical_weight_held: float = 0.0  # ... for challengers of a current holder (a dribble's vertical motion belongs to whoever controls it; the holder's own score ignores it); tuned
    holder_rel_speed_scale_img_fts: float = 25.0  # image: same, noisier velocities; tuned
    holder_rel_speed_slack_img_fts: float = 10.0  # image: extra slack on the hard acquire gate (noise); tuned
    holder_team_prior: float = 0.15  # score bonus for the passer's team while a pass is in the air; tuned
    holder_switch_margin_opp: float = 0.1  # image: extra margin an opponent needs to take the ball from the holder; tuned
    holder_switch_margin_opp_court: float = 0.0  # 3-D / court: positions are exact, relative motion decides; tuned
    use_ray_model: bool = True  # monocular: estimate the camera from rims + ball floor projections; tuned
    holder_ray_max_lateral_ft: float = 2.5  # image: the ball's viewing ray passes this close to the holder; tuned
    holder_ray_z_slack_ft: float = 1.5  # image: implied height may leave [0, holder_max_z] by this; tuned
    holder_ray_keep_slack_ft: float = 1.5  # extra slack for the current holder; tuned
    holder_ray_z_ref_ft: float = 2.5  # typical held / dribbled ball height; unverified
    holder_ray_z_band_ft: float = 2.0  # no penalty within ref +- band; tuned
    holder_ray_z_weight: float = 0.1  # score penalty per ft outside the band; tuned
    holder_ray_lateral_free_ft: float = 1.0  # image: no penalty while the viewing ray passes within this of the player ...; tuned on sim train seeds
    holder_ray_lateral_weight: float = 0.15  # ... then this score penalty per ft (depth disambiguation of overlapping boxes); tuned on sim train seeds
    # Monocular body-span holder score (possession._body_score); 3-D / court mode does not use these.
    holder_img_reach_ft: float = 0.4  # image acquire: ball centre at most this far beyond the body's image span (hands); tuned on sim train seeds
    holder_img_keep_reach_ft: float = 0.6  # image: the holder keeps it this far beyond the span (crossovers, side dribbles); tuned on sim train seeds
    holder_img_bottom_frac: float = 0.12  # image acquire: down to this fraction of bbox height below the feet (floor contact in front of the body); tuned on sim train seeds
    holder_img_centre_pref: float = 0.3  # image: score drop from the centre line to the span edge (tie-break between overlapping boxes); tuned on sim train seeds
    # Monocular leaky k-of-n confirmation (possession.HolderTracker._vote_gain); 3-D keeps consecutive frames.
    holder_vote_slack: int = 1  # image: a new holder wins confirm_n of the last confirm_n + this observed frames; tuned on sim train seeds
    holder_acquire_loose_img_fts: float = 16.0  # image: a loose ball after a miss is acquired only below this ball-player relative speed (vertical included); tuned on sim train seeds
    holder_vote_loose_k: int = 5  # image, loose ball after a miss: the rebounder wins this many ...; tuned on sim train seeds
    rebound_backdate_max_s: float = 1.0  # image, loose ball: the rebounder's gain is dated back over his valid candidacy (during the loose ball) at most this before his first counted win; tuned on sim train seeds
    holder_hidden_confirm_s: float = 0.4  # image, loose ball: a clear winner whose ball is then hidden this long (a rebound secured at the chest in traffic, covered by bodies) while he stays in view holds it; 0 disables; tuned on sim train seeds
    holder_hidden_share: float = 0.6  # ... the ball hidden on at least this share of the frames since his first win ...; tuned on sim train seeds
    holder_hidden_window_s: float = 1.0  # ... looking back this far for other winners; tuned on sim train seeds
    holder_vote_loose_n: int = 6  # ... of the last this many observed frames (RESEARCH §2.4 holds a rebound 300 ms); tuned on sim train seeds
    steal_confirm_img_s: float = 0.5  # image: a change of team control must last this long (steal hold 500 ms, RESEARCH §2.4 / §4.1); sourced
    steal_vote_window_s: float = 0.8  # image: votes for a change of team control are counted over this window ...; tuned on sim train seeds
    steal_vote_frac: float = 0.6  # ... the other team must win this share of the observed frames since its first win ...; tuned on sim train seeds
    steal_min_votes: int = 5  # ... and at least this many of them; tuned on sim train seeds
    steal_backdate_slack: int = 2  # ... observed frames in a row without a candidate of that team that do not end its contest; tuned on sim train seeds
    steal_backdate_max_s: float = 1.0  # image: such a change is dated back to the start of the new team's contest for the ball (its valid candidacy), at most this before its first win (possession._contest_onset); tuned on sim train seeds
    steal_confirm_s: float = 0.3  # 3-D / court: a change of team control needs this long (exact positions); tuned
    # Monocular steal corroboration (engine._PendingSteal): STEAL / TURNOVER / POSSESSION of an opponent's gain
    # are emitted (dated at the gain) once the new team has held the ball steal_sustain_s and dribbled, shot or
    # been fouled within steal_corroborate_s of the gain; the old team holding it again first means no change.
    # (Unambiguous holding is no evidence here: on sim train seeds wrong-body gains in the open, e.g. a pass
    # crossing a lone defender's box, are as unambiguous.)
    steal_sustain_s: float = 0.9  # 0 disables (immediate steals); tuned on sim train seeds
    steal_corroborate_s: float = 1.5  # ... counted from the confirmation of the gain; tuned on sim train seeds
    # Monocular physical check on a change of team control (engine._reachable): the ball goes from the releaser
    # to the gainer either within a strip's reach or in flight, at most steal_max_ball_speed_fts. A gain dated
    # (backdated) before the release or farther than the ball can fly in the time is a wrong-body holder.
    steal_reach_ft: float = 6.0  # stripping hand reach (~2.3 ft, sim) + both players' monocular position error; design choice, unverified
    steal_max_ball_speed_fts: float = 50.0  # a hard pass; design choice, unverified
    holder_clear_margin: float = 0.3  # image: a frame is unambiguous when no other candidate scores within this of the pick (keep bonus left out); tuned on sim train seeds
    holder_clear_frames: int = 6  # image: unambiguous observed frames that corroborate a releaser's control (victim_min_hold_s); tuned on sim train seeds
    victim_min_hold_s: float = 1.5  # monocular: a releaser without a dribble held the ball at least this long, or his "loss" to an opponent is a wrong-body correction, not a steal; tuned on sim train seeds
    holder_contest_margin: float = 0.02  # opponents within this score of each other: contested, no holder; tuned
    holder_confirm_s: float = 0.1  # new holder after ~3 frames at 30 fps; tuned
    holder_release_s: float = 0.1  # release after ~3 frames without the ball; tuned
    release_gap_backdate_s: float = 0.3  # image: a ball reappearing in flight after a gap this long was released when its flight left the holder's box (back-extrapolated), not at the gap start; tuned on sim train seeds
    holder_min_frames: int = 2  # floor on hysteresis frame counts at low fps; tuned
    holder_gap_keep_s: float = 10.0  # keep the holder through ball gaps up to this (cv_pipeline §6.2); tuned
    id_swap_radius_ft: float = 2.5  # a vanished holder replaced in place by a teammate id = id swap; tuned
    track_identity_memory_s: float = 3.0  # a track shown with a provisional id within this of its last roster id keeps that id (team-vote flips last ~0.3-1.5 s); design, unverified
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
    pass_min_flight_s: float = 0.2  # monocular: a same-team change faster than this ... (engine._flicker); tuned on sim train seeds
    pass_min_dist_ft: float = 6.0  # ... between players this close, the ball moving less than a body width in the image, is no pass (2% of sim passes are under 6 ft); tuned on sim train seeds
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
    # Monocular retro-block (engine._retro_block): the buffered image track around a release
    retro_block_lookback_s: float = 1.5  # look back this far before the (often late) release, not before his control began; tuned on sim train seeds
    retro_block_stand_lead_s: float = 0.6  # the shooter's standing box (head line) this long before the release; tuned on sim train seeds
    retro_block_min_base_s: float = 0.06  # ... over baselines of at least this (+-1.5 px jitter); tuned on sim train seeds
    retro_block_max_gap_s: float = 0.35  # velocities only between observations this close (the ball is often hidden at the touch); tuned on sim train seeds
    retro_block_turn_rise_frac: float = 1.0  # a ball turned back from the rim has at least this much less rise after the touch than before (a swat from the side or behind keeps some of its upward speed: the ball was turned, not stopped); tuned on sim train seeds
    # Blocks of shots already inside the rim zone (layups, dunks: they enter it at the release; shots.Flight.
    # _check_rim_block). 3-D: a reversal clear of the ring (rim_contact_dist_ft) and the board is a defender's.
    backboard_half_width_ft: float = 3.0  # 72 in board (NBA rule 1 sec. II); sourced
    backboard_above_rim_ft: float = 3.5  # board top ~ 42 in board, rim ~ 6 in above its bottom edge; sourced (approx.)
    # 3-D tip (engine._detect_tip): a loose ball after a miss redirected back up by a hand above holder height,
    # next to a player and clear of the ring / board, is that player's touch (OREB + a release at the touch:
    # a tip-in is a rebound and a shot attempt, NBA stat rules), as he never holds it long enough to confirm.
    tip_min_z_ft: float = 7.0  # contact at least this high (lower catches go through the holder vote); tuned on sim train seeds
    tip_reach_ft: float = 2.5  # a player's centre this close (horizontal) to the contact; tuned on sim train seeds
    tip_min_dv_fts: float = 10.0  # upward velocity change beyond gravity's across the contact step; tuned on sim train seeds
    tip_min_up_fts: float = 3.0  # ... and the ball rises at least this fast after it; tuned on sim train seeds
    tip_max_basket_ft: float = 8.0  # contact within this of the basket (horizontal); tuned on sim train seeds
    airball_radius_ft: float = 7.0  # 3-D: descending past rim height within this of the basket; tuned
    airball_min_apex_above_ft: float = 1.0  # tuned
    airball_min_release_dist_ft: float = 6.0  # tuned
    # Monocular: a steal or rebound followed this quickly by the other team holding the ball, with no
    # completed pass or shot in between, was a wrong-body holder: the steal is retracted (the pass it
    # intercepted is completed) or the rebound re-credited, instead of a second change of possession.
    team_change_revert_s: float = 2.5  # tuned on sim train seeds
    rebound_retime_ft: float = 4.0  # 3-D: a re-credited rebound moves to the new holder's gain when the wrong body stood farther than this from him (monocular: kept, positions too noisy); tuned on sim train seeds
    rebound_retime_mono_ft: float = 6.0  # monocular: the same, from the (noisier) floor positions; tuned on sim train seeds
    strip_max_travel_ft: float = 6.0  # opponent gains within this of the release: steal, not bad pass; tuned
    strip_max_flight_s: float = 0.6  # tuned
    # --- dribbles
    dribble_floor_z_ft: float = 1.2  # bounce minimum below this (ball centre 0.39 ft at contact); tuned
    dribble_min_amp_ft: float = 0.8  # 3-D reversal amplitude of a bounce; tuned
    dribble_radius_ft: float = 4.0  # bounce within this of the dribbler; tuned
    dribble_foot_band: float = 0.35  # image: lowest observed point of a bounce within this fraction of bbox height of the feet (the floor-contact frames are often missed: the ball is hidden by the legs); tuned on sim train seeds
    dribble_min_amp_h: float = 0.15  # image: reversal amplitude as a fraction of bbox height; tuned
    dribble_min_interval_s: float = 0.2  # tuned
    dribble_max_rise_s: float = 0.6  # a bounce is back up by the reversal amplitude within this of its low point (floor to hand ~0.2-0.3 s); unverified
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
    dead_min_s: float = 1.0  # fallback restarts (dribble / pass in bounds) only this long after the dead ball was recognised; tuned
    dead_from_clock_stop: bool = True  # enter a dead ball when the trusted game clock stops; tuned
    clock_stop_min_s: float = 1.2  # clock unchanged this long (MM:SS display) = stopped; tuned
    detect_shot_clock_violations: bool = True  # shot clock read 0 while live -> team turnover; tuned
    period_start_tol_s: float = 1.0  # first read within this of a full period = period start; tuned
    continuation_s: float = 0.8  # a shooter fouled in the act releases (or released) within this of the whistle: the shot plays out (and-one if made); tuned on sim train seeds
    continuation_miss_wait_s: float = 5.0  # such a shot called missed is retracted (no FGA) unless the scoreboard shows it made by then (make_confirm_wait_s + reconcile_hold_s); tuned
    toss_min_speed_fts: float = 5.0  # 3-D: an unheld ball moving this fast at a period start = the jump-ball toss; tuned
    # Monocular jump ball (period 1 / jump-ball overtimes per RuleSpec): the referee holds the ball among the
    # jumpers, and referees are not in the player list, so nobody has control until the toss. The toss is
    # the ball rising steadily in the image (over a long baseline: +-1.5 px jitter must not trigger it).
    toss_min_rise_ft: float = 2.0  # image: rise (in px at the players' px/ft) over toss_window_s that marks the toss (it peaks ~7 ft above the hand); tuned on sim train seeds
    toss_window_s: float = 0.6  # tuned on sim train seeds
    toss_min_frames: int = 4  # observed frames in that window, the last three rising; tuned on sim train seeds
    toss_fallback_move_ft: float = 8.0  # image: the ball carried this far from the referee's spot = play began (no toss seen); tuned
    toss_wait_max_s: float = 12.0  # image: no toss seen this long after the period start -> normal holder logic; unverified
    jump_ball_radius_ft: float = 12.0  # image: the ball's floor projection this close to centre court = the referee's jump-ball spot (6 ft circle + the projection offset of a hand-held ball); unverified
    jump_control_s: float = 0.45  # image: a gain after the toss counts once held this long (the jumpers' tap is a touch, not control); tuned on sim train seeds
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
    # A confirmed +2 / +3 matching no seen shot becomes an inferred attempt + make (engine._unseen_make)
    infer_unseen_makes: bool = True  # design
    unseen_make_window_s: float = 6.0  # the shooter's release / shooting foul at most this before the delta's first read (make -> board lag 0.4-2 s + OCR + the reconcile hold); tuned on sim train seeds
    unseen_make_max_dist_ft: float = 10.0  # a releaser this close to his basket (layups, dunks, close shots: the ones the image misses); tuned on sim train seeds
    unseen_make_release_lag_s: float = 1.0  # ... or released at most this after the delta (a ball carried through the rim leaves the shooter's box late); tuned on sim train seeds
    unseen_make_flight_s: float = 0.4  # release -> rim when the ball is not seen at the rim; unverified
    unseen_make_conf: float = 0.5  # confidence of inferred attempts / makes; design
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
class _TeamChange:
    """A change of team control (steal, rebound) that the next moments may still contradict: with one
    camera the first holder after a loose ball or a catch in traffic can be the wrong body, and the
    true holder taking the ball a moment later is not a second change of possession."""

    kind: str  # "steal" | "rebound"
    t: float
    team: Team | None  # the team that gained control
    poss_before: Team | None  # possession team before the change
    ids: list[str] = field(default_factory=list)  # steal: the events a reversal retracts
    miss: ShotEntry | None = None  # rebound: the missed shot
    release: Release | None = None  # bad pass: the intercepted pass
    chain: list[str] = field(default_factory=list)  # completed passes inside the gaining team since (retracted too)


@dataclass(slots=True)
class _PendingSteal:
    """Monocular: an opponent of the releaser gained the ball. STEAL / TURNOVER / POSSESSION wait until his
    team's control is corroborated: it lasts ``steal_sustain_s`` and that team dribbles, shoots or is fouled
    within ``steal_corroborate_s``. With one camera a defender's box over the ball handler, or a pass crossing
    a defender's box, wins the image votes for a moment far more often than a real steal happens."""

    stealer: str
    team: Team
    ch: ControlChange  # the stealer's gain (onset)
    flight: Flight  # the flight he took the ball from (its release is the victim's)
    kind: str  # "steal" (strip) | "bad_pass"
    victim_ctrl: _Control | None  # the victim's stint (resumed if he gets the ball straight back)
    created_t: float = 0.0  # when the gain was confirmed (its onset can be dated back up to ~1.5 s)
    chain: list[str] = field(default_factory=list)  # completed passes inside the new team since
    evidence: bool = False  # a dribble / shot / foul on the new team since the onset
    implausible: bool = False  # the ball could not have reached him in time (``_reachable``): never a steal


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
    change: _TeamChange | None = None  # monocular: provisional change of team control that started this stint
    via_flight: Flight | None = None  # the pass flight that started it (monocular: spliced out if a wrong body)
    clear: int = 0  # monocular: observed frames on which the ball was unambiguously his (HolderTracker.clear_n)
    doubtful: bool = False  # monocular: began with a change of team control that was never corroborated
    miss: ShotEntry | None = None  # the missed shot a rebound stint followed
    prev_ctrl: _Control | None = None  # the passer's stint (one level only)


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
    toss_uv: list[tuple[float, float, float]] = field(default_factory=list)  # image: (t, u, v) before the toss
    jump_pending: tuple[str, Team | None, float, int] | None = None  # image: first control after the toss, not yet held
    entered_t: float = -math.inf  # stream time the engine entered this dead ball (since_t can be back-dated)


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
        self._continuation: tuple[str, float] | None = None  # (shooter, foul t): fouled in the act, not released yet
        self._cont_misses: list[ShotEntry] = []  # continuation shots called missed, awaiting the scoreboard
        self._post_make: tuple[str, tuple[float, int, str], str | None] | None = None  # (shot, pickup, POSSESSION id)
        self.fouled_out: set[str] = set()
        self._ppf: float | None = None
        self._mono = False  # the last accepted ball observation was image-only (monocular)
        self._now = -math.inf  # time of the frame being processed
        self._track_bound: dict[int, tuple[str, Team | None, float]] = {}  # track -> (roster id, team, last seen)
        self._pend: _PendingSteal | None = None  # monocular change of team control awaiting corroboration
        self._last_poss: tuple[str, Team | None, float] | None = None  # (POSSESSION id, team before, t)
        self._release_log: deque[tuple[Release, _Control | None]] = deque(maxlen=24)  # live releases
        self._boxes: deque[tuple[float, dict[str, tuple[BBox, Team | None]]]] = deque(maxlen=45)  # image boxes
        self._shoot_fouls: deque[tuple[float, int, str]] = deque(maxlen=8)  # (t, frame, fouled shooter)
        self.ray_model = RayModel(court.rim_height, {s: court.basket_xy(s) for s in ("left", "right")})

    # ------------------------------------------------------------------ public API

    def process(self, frame: FrameState) -> list[Event]:
        """New events and revisions produced at this frame."""
        self._out = []
        frame = self._canonical_ids(_sanitized(frame))
        self._now = frame.t
        self._tick(frame)
        self._hist.update(frame)
        self._players = {p.player_id: p for p in frame.players}
        if any(p.bbox is not None for p in frame.players):
            self._boxes.append((frame.t, {p.player_id: (p.bbox, p.team) for p in frame.players if p.bbox is not None}))
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
            fl = self._flight
            if fl is not None and self._mono and fl.state == "air" and fl.context == "live" and fl.retro_block is None \
                    and not fl.is_ft and frame.t - fl.release.t <= self.config.block_max_after_release_s:
                self._retro_block(fl)  # a release recognised early: the touch after it is buffered only now
        confirm_n, release_n = self._hysteresis_frames()
        f = self._flight
        prefer = f.release.team if f is not None and f.state == "air" and f.context == "live" else None
        opp_n = max(confirm_n, round(self.config.steal_confirm_s / max(self._dt, 1e-3))) if self._phase == LIVE else None
        if obs is not None:
            self._mono = not obs.is3d and obs.uv is not None
        if self._await_toss(obs, frame.t):
            changes: list[ControlChange] = []
            self._holder.reset()  # the referee holds the ball between the jumpers: nobody has control yet
        else:
            changes = self._holder.update(frame, obs, self._gate.velocity(), self._ppf, confirm_n, release_n, prefer,
                                          opp_n, self.ray_model.ray, self._gate.vertical_accel(),
                                          loose=self._pending_miss is not None)
            if not changes and obs is not None and obs.is3d and self._pending_miss is not None:
                changes = self._detect_tip()
        self._note_toss(obs)
        for ch in changes:
            if ch.kind == "release":
                self._on_release(ch)
            else:
                self._on_gain(ch)
        self._check_jump_pending(frame.t)
        self._detect_dribble(obs)
        ctrl = self._control
        if ctrl is not None and ctrl.player_id == self._holder.holder:
            ctrl.clear = self._holder.clear_n
        self._check_pending(frame.t)
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
        self._resolve_pending(end=True)
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

    def _canonical_ids(self, frame: FrameState) -> FrameState:
        """Identity continuity by track: a track whose label falls back to a provisional id
        ("<team>-t<track>", ARCHITECTURE PlayerState: e.g. a short team-vote flip) is still the roster
        player it was bound to moments ago, unless that player is seen elsewhere in the frame. Without
        this a holder whose team label flips for a second becomes a steal and a steal back."""
        mem = self.config.track_identity_memory_s
        if mem <= 0.0:
            return frame
        bound = self._track_bound
        present = {p.player_id for p in frame.players}
        changed = False
        players = []
        for p in frame.players:
            if p.track_id is not None and not _PROVISIONAL.match(p.player_id):
                bound[p.track_id] = (p.player_id, p.team, frame.t)
            players.append(p)
        for i, p in enumerate(players):
            if p.track_id is None or not _PROVISIONAL.match(p.player_id):
                continue
            hit = bound.get(p.track_id)
            if hit is not None and frame.t - hit[2] <= mem and hit[0] not in present:
                players[i] = dataclasses.replace(p, player_id=hit[0], team=hit[1])
                present.add(hit[0])
                changed = True
        if len(bound) > 256:  # forget stale tracks
            self._track_bound = {k: v for k, v in bound.items() if frame.t - v[2] <= mem}
        return dataclasses.replace(frame, players=players) if changed else frame

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

    def _jump_expected(self) -> bool:
        """The period opens with a jump ball (period 1; overtimes in jump-ball leagues), not a throw-in."""
        p = self._period
        return p is None or p == 1 or (self.rules.is_overtime(p) and self.rules.possession == "jump")

    def _await_toss(self, obs: BallObs | None, t: float) -> bool:
        """Monocular jump ball: True while the referee still holds the ball (no toss seen yet)."""
        d = self._dead
        if d is None or d.cause != "period" or d.toss_seen or not self._mono or not self._jump_expected():
            return False
        cfg = self.config
        if t - d.since_t > cfg.toss_wait_max_s:
            d.toss_seen = True
            return False
        if obs is None or obs.uv is None:
            return True
        if obs.xy is not None and math.hypot(obs.xy[0] - self.court.length / 2.0,
                                             obs.xy[1] - self.court.basket_y) > cfg.jump_ball_radius_ft:
            d.toss_seen = True  # the ball is not at centre court: no jump ball to wait for (e.g. a free throw)
            return False
        ppf = self._ppf or cfg.nominal_px_per_ft
        hist = d.toss_uv
        hist.append((obs.t, obs.uv[0], obs.uv[1]))
        if math.hypot(obs.uv[0] - hist[0][1], obs.uv[1] - hist[0][2]) >= cfg.toss_fallback_move_ft * ppf:
            d.toss_seen = True  # the ball was carried away from the referee's spot: play began without a toss seen
            return False
        recent = [s for s in hist if obs.t - s[0] <= cfg.toss_window_s]
        del hist[1:max(1, len(hist) - 64)]  # keep the first (the referee's spot) and the recent samples
        if len(recent) >= cfg.toss_min_frames and recent[0][2] - recent[-1][2] >= cfg.toss_min_rise_ft * ppf \
                and recent[-3][2] > recent[-2][2] > recent[-1][2]:
            d.toss_seen = True  # image v decreases: the ball rises steadily (the toss)
            return False
        return True

    def _check_jump_pending(self, t: float) -> None:
        """Monocular jump ball: the first player still holding the ball ``jump_control_s`` after gaining
        it after the toss starts the period (a jumper's tap is a touch: the ball leaves him at once)."""
        d = self._dead
        if d is None or d.cause != "period" or d.jump_pending is None:
            return
        pid, team, t0, f0 = d.jump_pending
        if self._holder.holder != pid:
            d.jump_pending = None
            return
        if t - t0 >= self.config.jump_control_s:
            if team is not None:
                d.first_control[team] = (t0, f0, pid)
            self._restart(team, t0, f0, pid, "jump")

    def _note_toss(self, obs: BallObs | None) -> None:
        d = self._dead
        if d is None or d.cause != "period" or d.toss_seen or obs is None or self._holder.holder is not None:
            return
        if self._mono and self._jump_expected():
            return  # image mode: see _await_toss
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
        self._resolve_pending()
        self._phase = DEAD
        self._dead = _Dead(cause=cause, since_t=t, since_frame=f, entered_t=max(t, self._now), **kw)
        return self._dead

    def _dead_for(self, d: _Dead, t: float) -> float:
        """How long the dead ball has lasted at ``t`` for the fallback restarts: counted from when it was
        recognised, as one dated back to the clock stop is often recognised over a second later."""
        return t - max(d.since_t, d.entered_t)

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
            pev = None
            if team is not None and d.scoring_team is not None and team is not d.scoring_team:
                pev = self._set_possession(team, ft, ff, fp, "made_shot" if d.cause == "made_fg" else "free_throw")
            if d.cause == "made_fg" and d.shot_id is not None:
                # kept until the make is confirmed: a make revised to a miss turns this pickup into the rebound
                self._post_make = (d.shot_id, d.first_pickup or (ft, ff, fp), pev.event_id if pev is not None else None)
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
        if self._pend is not None and team is self._pend.team and cause != "steal":
            return None  # decided when the pending change of control is corroborated or settled
        before, self._poss_team = self._poss_team, team
        ev = self._emit(EventKind.POSSESSION, t, f, team=team, player_id=pid, xy=self._pos(pid, t) if pid else None,
                        confidence=self.config.possession_conf if conf is None else conf,
                        attrs={"cause": cause, "reason": f"{team.value} gains control ({cause})"})
        self._last_poss = (ev.event_id, before, t)
        return ev

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
            cont = self._continuation
            continuation = cont is not None and cont[0] == pid and ch.t - cont[1] <= cfg.continuation_s
            flight = Flight(rel, self.court, cfg, context="live" if self._phase == LIVE or continuation else "dead")
            flight.continuation = continuation
            if continuation:
                self._continuation = None
        self._flight, self._flight_ctrl = flight, ctrl
        self._prev_control, self._control = ctrl, None
        if flight.context == "live" and not flight.is_ft:
            self._release_log.append((rel, ctrl))
        if ft_side is not None:
            self._start_free_throw(flight)
        elif flight.context == "live" and self._mono:
            self._retro_block(flight)
        for t, fi, o in list(self._buffer):
            if fi >= ch.frame_idx and self._flight is flight and not flight.resolved:
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
        self._commit_retro_block(flight)  # the ball never reached the rim
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

    def _commit_retro_block(self, flight: Flight | None) -> None:
        """A monocular block candidate (``_retro_block``) becomes the block once the ball's next owner is known:
        someone gains it, or it goes dead (a blocked ball knocked out of bounds) before reaching a rim zone."""
        if flight is None or flight.retro_block is None or flight.state != "air" or flight.context != "live" \
                or self._phase != LIVE:
            return
        data = flight.retro_block[4]
        r = flight.release
        if data["rise_t"] < r.t:  # the attempt is the rise the release was recognised after
            r.t, r.frame_idx = data["rise_t"], data["rise_frame"]
        self._on_block(flight, flight.mark_blocked(*flight.retro_block))

    def _enter_dead_from_live(self, cause: str, t: float, f: int, flight: Flight | None, out: bool = True,
                              before: float | None = None) -> None:
        self._commit_retro_block(flight)
        last_holder = flight.release.player_id if flight is not None else (self._holder.holder or None)
        last_team = flight.release.team if flight is not None else self._poss_team
        if flight is None and self._control is not None:
            last_holder, last_team = self._control.player_id, self._control.team
        elif flight is None and last_holder is not None:
            last_team = self._team_of(last_holder) or last_team  # the TURNOVER's player and team must agree
        seen = self._last_live_ball(t if before is None else before) if out else None
        stop = self._clock_stop_before(t) if out else None
        if stop is not None and (seen is None or stop < seen[0]):
            seen = (stop, self._frame_near(stop))  # the whistle (clock stop) came before the last in-bounds sighting
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
            if self._mono and self._jump_expected():
                if d.toss_seen:
                    d.jump_pending = (pid, team, ch.t, ch.frame_idx)  # confirmed once held (_check_jump_pending)
                return
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
        return self._dead_for(d, r.t) >= self.config.dead_min_s

    def _gain_live(self, pid: str, team: Team | None, ch: ControlChange, flight: Flight | None) -> None:
        cfg = self.config
        self._flight = None
        pend = self._pend
        if pend is not None and team is not None and team is not pend.team:
            self._drop_pending(pid, team, ch)  # the other team holds the ball again: it never changed hands
            return
        miss = self._pending_miss
        if miss is not None and (flight is None or flight.is_shot or flight.state == "loose"):
            if miss.is_ft and miss.extra.get("ft_total_unknown") and self._ft_retrieval(miss):
                self._enter_dead("ft_trip", ch.t, ch.frame_idx)
                self._pending_miss = None
                self._control = _Control(pid, team, ch.t, ch.frame_idx)
                return
            poss_before = self._poss_team
            self._rebound(pid, team, ch, miss)
            if miss.is_ft:
                self._ft = None
            self._control = _Control(pid, team, ch.t, ch.frame_idx, origin="rebound",
                                     change=self._provisional("rebound", ch.t, team, poss_before, miss=miss), miss=miss)
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
        if team is not None and team is r.team and self._mono and self._flicker(r, pid, ch):
            # the ball never left the area: a neighbouring teammate's box won it for a moment (no pass)
            prev = self._prev_control
            self._control = dataclasses.replace(prev, player_id=pid) if prev is not None and prev.player_id == r.player_id \
                else _Control(pid, team, ch.t, ch.frame_idx)
            return
        if team is not None and team is r.team:
            ev = self._pass(flight, pid, ch, completed=True)
            chg = self._carried_change(r.player_id, ch.t)
            if chg is not None:
                chg.chain.append(ev.event_id)  # a pass inside a team whose control is still provisional
            if pend is not None:
                pend.chain.append(ev.event_id)
            pc = self._prev_control if self._prev_control is not None and self._prev_control.player_id == r.player_id \
                else None
            if pc is not None:
                pc.prev_ctrl = None  # keep one level
            self._control = _Control(pid, team, ch.t, ch.frame_idx, origin="pass",
                                     via_pass=_PassRec(r.player_id, r.team, r.t, ev.event_id), change=chg,
                                     via_flight=flight, prev_ctrl=pc)
            self._set_possession(team, ch.t, ch.frame_idx, pid, "other")
            return
        if team is not None and r.team is not None:
            chg = self._carried_change(r.player_id, ch.t)
            if chg is not None and team is not chg.team:
                self._revert_change(chg, pid, team, ch)
                return
            vc = self._prev_control if self._prev_control is not None and self._prev_control.player_id == r.player_id \
                else None
            # (3-D too for a rebound stint: a loose ball bouncing past a player is not always free flight)
            if (self._mono or (vc is not None and vc.origin == "rebound")) and not self._corroborated(vc, r.t):
                self._correct_phantom(vc, pid, team, ch)
                return
            travel = math.dist(r.xy, self._pos(pid, ch.t) or r.xy) if r.xy is not None else 0.0
            strip = travel <= cfg.strip_max_travel_ft and ch.t - r.t <= cfg.strip_max_flight_s
            kind = "steal" if strip else "bad_pass"
            if self._mono and cfg.steal_sustain_s > 0.0:
                self._pend = _PendingSteal(pid, team, ch, flight, kind, self._prev_control, created_t=self._now,
                                           implausible=not self._reachable(r, pid, ch))
                self._control = _Control(pid, team, ch.t, ch.frame_idx, origin="steal")
                return
            self._steal(pid, team, ch, flight, kind)
            return
        self._control = _Control(pid, team, ch.t, ch.frame_idx)

    def _reachable(self, r: Release, pid: str, ch: ControlChange) -> bool:
        """Could the ball have gone from the releaser to the gainer ``pid`` by his gain? A gain dated before the
        release (a strip): the two players within ``steal_reach_ft`` at its onset. Otherwise from the release
        point: within that reach, or beyond it in a flight of at most ``steal_max_ball_speed_fts``. Monocular
        labels are noisy (a duplicate or swapped box can carry an id), so the distance is the nearest over every
        position recorded for them around the onset: only a change far beyond reach on every sample is
        rejected, and fewer than three positions is no evidence."""
        cfg = self.config
        near = self._hist.positions_near
        flight_s = ch.t - r.t
        dst = near(pid, ch.t)
        src = near(r.player_id, ch.t) if flight_s < 0.0 else ([r.xy] if r.xy is not None else [])
        if len(dst) < 3 or not src or (flight_s < 0.0 and len(src) < 3):  # a stray box or two: unknown
            return True
        extra = min(math.dist(a, b) for a in src for b in dst) - cfg.steal_reach_ft
        return extra <= 0.0 or max(flight_s, 0.0) * cfg.steal_max_ball_speed_fts >= extra

    def _flicker(self, r: Release, pid: str, ch: ControlChange) -> bool:
        """Monocular: a same-team "pass" whose flight was shorter than ``pass_min_flight_s`` between players
        closer than ``pass_min_dist_ft`` while the ball moved less than one body width in the image is the
        holder flickering between overlapping teammates' boxes, not a pass (real hand-offs exist, but a pass
        moves the ball). Kept to clear cases: every observed ball sample of the flight must be close."""
        cfg = self.config
        if ch.t - r.t >= cfg.pass_min_flight_s or r.xy is None or r.bbox is None:
            return False
        pos = self._pos(pid, ch.t)
        if pos is None or math.dist(r.xy, pos) >= cfg.pass_min_dist_ft:
            return False
        pts = [o.uv for t, _, o in self._buffer if o is not None and o.uv is not None and r.t - 0.05 <= t <= ch.t + 0.05]
        if len(pts) < 2:
            return False
        return max(math.dist(pts[0], q) for q in pts[1:]) < r.bbox.w

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
        return self._pass_from(flight.release, receiver, ch, completed)

    def _pass_from(self, r: Release, receiver: str | None, ch: ControlChange, completed: bool) -> Event:
        rpos = self._pos(receiver, ch.t) if receiver else None
        dist = math.dist(r.xy, rpos) if r.xy is not None and rpos is not None else None
        return self._emit(EventKind.PASS, r.t, r.frame_idx, team=r.team, player_id=r.player_id, other=receiver,
                          xy=r.xy, confidence=self.config.pass_conf,
                          attrs={"completed": completed, "distance_ft": dist,
                                 "reason": "teammate gained control" if completed else "intercepted"})

    def _steal(self, pid: str, team: Team, ch: ControlChange, flight: Flight, kind: str,
               control: bool = True) -> _TeamChange | None:
        r = flight.release
        poss_before = self._poss_team
        ids = []
        if kind == "bad_pass":
            ids.append(self._pass(flight, self._intended_receiver(r, ch.t, pid), ch, completed=False).event_id)
        cues = self._control_cues(pid, ch)
        p, status = self._p("steal", cues)
        pos = self._pos(pid, ch.t)
        travel = math.dist(r.xy, pos) if r.xy is not None and pos is not None else None
        ids.append(self._emit(EventKind.STEAL, ch.t, ch.frame_idx, team=team, player_id=pid, other=r.player_id, xy=pos,
                              confidence=p, attrs={"p_status": status, "cues": cues, "travel_ft": travel,
                                                   "flight_s": ch.t - r.t,
                                                   "reason": f"opponent gained control ({kind})"}).event_id)
        ids.append(self._emit(EventKind.TURNOVER, ch.t, ch.frame_idx, team=r.team, player_id=r.player_id, xy=pos,
                              confidence=p, attrs={"type": kind, "p_status": status,
                                                   "reason": "lost the ball to the stealer"}).event_id)
        pev = self._set_possession(team, ch.t, ch.frame_idx, pid, "steal", conf=p)
        if pev is not None:
            ids.append(pev.event_id)
        chg = self._provisional("steal", ch.t, team, poss_before, ids=ids, release=r if kind == "bad_pass" else None)
        if control:
            self._control = _Control(pid, team, ch.t, ch.frame_idx, origin="steal", change=chg)
        return chg

    # ------------------------------------------------------------------ monocular wrong-body stints

    def _corroborated(self, ctrl: _Control | None, release_t: float) -> bool:
        """Monocular: did this stint show control: a dribble, ``holder_clear_frames`` unambiguous frames (the
        ball in his box only), or ``victim_min_hold_s`` of holding? A short, ambiguous, dribble-less stint is
        often the ball crossing a bystander's box (a pass in flight, a loose ball, a defender's box over the
        handler)."""
        if ctrl is None:
            return False
        if ctrl.dribbles > 0:
            return True
        cfg = self.config
        if ctrl.doubtful:
            return False  # his team's control was never corroborated: only a dribble shows it
        return ctrl.clear >= cfg.holder_clear_frames or release_t - ctrl.onset_t >= cfg.victim_min_hold_s

    def _correct_phantom(self, vc: _Control | None, pid: str, team: Team, ch: ControlChange) -> None:
        """An opponent gained the ball from a releaser who never showed control: not a steal, the releaser's
        stint was a wrong body. Splice it out: a pass to him is retracted and the gain is judged against
        the passer's release; a rebound / steal that started it is re-credited / retracted; a change of
        possession it made is undone."""
        reason = "the releaser never showed control (no dribble, short stint): wrong-body holder (monocular)"
        if vc is not None and vc.origin == "pass" and vc.via_flight is not None and vc.via_pass is not None \
                and vc.via_pass.team is not team:
            self._retract(vc.via_pass.event_id, reason)
            self._prev_control = vc.prev_ctrl
            vc.via_flight.state = "air"
            self._gain_live(pid, team, ch, vc.via_flight)
            return
        if vc is not None and vc.change is not None and vc.change.team is not team:
            self._revert_change(vc.change, pid, team, ch)
            return
        if vc is not None and vc.change is None and vc.origin == "rebound" and vc.miss is not None \
                and vc.miss.rebound_id is not None:
            poss_before = self._last_poss[1] if self._last_poss is not None else None
            self._revert_change(_TeamChange("rebound", vc.onset_t, vc.team, poss_before, miss=vc.miss), pid, team, ch)
            return
        lp = self._last_poss
        if vc is not None and lp is not None and lp[2] >= vc.onset_t - 0.05:
            ev = self._store.get(lp[0])
            if ev is not None and ev.team is vc.team and ev.status is not EventStatus.RETRACTED:
                self._retract(lp[0], reason)
                self._poss_team = lp[1]
        self._control = _Control(pid, team, ch.t, ch.frame_idx)
        self._set_possession(team, ch.t, ch.frame_idx, pid, "other")

    # ------------------------------------------------------------------ monocular pending steals

    def _check_pending(self, t: float) -> None:
        """Corroborate (dribble / shot / foul + sustained control) or settle a pending change of control."""
        pend, cfg = self._pend, self.config
        if pend is None:
            return
        held = t - pend.ch.t
        holder = self._holder.holder
        mine = holder is None or self._team_of(holder) is pend.team
        if pend.evidence and held >= cfg.steal_sustain_s and mine and not pend.implausible:
            self._confirm_pending()
        elif t - pend.created_t > cfg.steal_corroborate_s:
            self._settle_pending()

    def _note_pending_evidence(self, team: Team | None, t: float) -> None:
        pend = self._pend
        if pend is not None and team is pend.team and t >= pend.ch.t - 1e-6:
            pend.evidence = True

    def _resolve_pending(self, end: bool = False) -> None:
        """A dead ball while a change of control is pending: corroborated -> steal, else no steal (the inbound
        decides possession). At the end of the stream no contradiction can come any more: a change the new
        team has held ``steal_confirm_img_s`` (the steal hold) is confirmed too."""
        pend = self._pend
        if pend is None:
            return
        holder = self._holder.holder
        held = self._now - pend.ch.t >= self.config.steal_confirm_img_s and holder is not None \
            and self._team_of(holder) is pend.team
        if (pend.evidence or (end and held)) and not pend.implausible:
            self._confirm_pending()
            return
        self._pend = None
        if pend.kind == "bad_pass":
            self._pass(pend.flight, self._intended_receiver(pend.flight.release, pend.ch.t, pend.stealer), pend.ch,
                       completed=False)

    def _confirm_pending(self) -> None:
        pend, self._pend = self._pend, None
        assert pend is not None
        chg = self._steal(pend.stealer, pend.team, pend.ch, pend.flight, pend.kind, control=False)
        ctrl = self._control
        if chg is not None:
            chg.chain = list(pend.chain)
        if ctrl is not None and ctrl.team is pend.team:
            ctrl.change = chg
            if ctrl.player_id == pend.stealer and ctrl.onset_t == pend.ch.t:
                ctrl.origin = "steal"

    def _settle_pending(self) -> None:
        """No corroboration within ``steal_corroborate_s``: no STEAL / TURNOVER. If the new team still has the
        ball, its possession starts at the gain (an unexplained change, needs review)."""
        pend, self._pend = self._pend, None
        assert pend is not None
        if pend.kind == "bad_pass":  # the release was a pass all the same (not known to be completed)
            self._pass(pend.flight, self._intended_receiver(pend.flight.release, pend.ch.t, pend.stealer), pend.ch,
                       completed=False)
        holder = self._holder.holder or (self._control.player_id if self._control is not None else None)
        if holder is not None and self._team_of(holder) is pend.team:
            ev = self._set_possession(pend.team, pend.ch.t, pend.ch.frame_idx, pend.stealer, "other",
                                      conf=0.5 * self.config.possession_conf)
            if ev is not None:
                ev.attrs["needs_review"] = True
            ctrl = self._control
            if ctrl is not None and ctrl.team is pend.team:
                ctrl.doubtful = True  # the old team taking it back is a correction, not a steal

    def _drop_pending(self, pid: str, team: Team, ch: ControlChange) -> None:
        """The releaser's team holds the ball again before the change of control was corroborated: the
        "stealer" was a wrong-body holder. His team's passes are retracted and the original release resumes
        (an intercepted-looking pass completed to this teammate, or the releaser's own stint)."""
        pend, self._pend = self._pend, None
        assert pend is not None
        reason = "the other team held the ball again before the change of control was corroborated (monocular)"
        for eid in pend.chain:
            self._retract(eid, reason)
        r0 = pend.flight.release
        if pid == r0.player_id:
            vc = pend.victim_ctrl
            self._control = vc if vc is not None and vc.player_id == pid else \
                _Control(pid, team, ch.t, ch.frame_idx, origin="self")
        elif r0.team is team:
            ev = self._pass_from(r0, pid, ch, completed=True)
            self._control = _Control(pid, team, ch.t, ch.frame_idx, origin="pass",
                                     via_pass=_PassRec(r0.player_id, r0.team, r0.t, ev.event_id))
        else:
            self._control = _Control(pid, team, ch.t, ch.frame_idx)
        self._set_possession(team, ch.t, ch.frame_idx, pid, "other")

    def _provisional(self, kind: str, t: float, team: Team | None, poss_before: Team | None, *,
                     ids: list[str] | None = None, miss: ShotEntry | None = None,
                     release: Release | None = None) -> _TeamChange | None:
        """Monocular only: the change of team control can still be reversed (``_revert_change``); with
        exact 3-D positions the first holder is right and a quick reversal is a real play."""
        if not self._mono:
            return None
        return _TeamChange(kind, t, team, poss_before, ids or [], miss, release)

    def _carried_change(self, releaser: str, t: float) -> _TeamChange | None:
        """The provisional change of team control behind the releaser's stint, if still revertible at ``t``."""
        prev = self._prev_control
        chg = prev.change if prev is not None and prev.player_id == releaser else None
        if chg is None or t - chg.t > self.config.team_change_revert_s:
            return None
        return chg

    def _revert_change(self, chg: _TeamChange, pid: str, team: Team, ch: ControlChange) -> None:
        """The other team holds the ball ``team_change_revert_s`` or less after a provisional change of
        team control: the earlier holder was the wrong body. A steal is retracted (its intercepted pass
        becomes a completed pass when a teammate of the passer now holds the ball); a rebound is
        re-credited to the new holder (offensive / defensive and the POSSESSION follow)."""
        reason = "the other team held the ball at once: wrong-body holder (monocular)"
        for eid in chg.chain:
            self._retract(eid, reason)
        if chg.kind == "steal":
            for eid in chg.ids:
                self._retract(eid, reason)
            self._poss_team = chg.poss_before
            r = chg.release
            if r is not None and r.team is team and pid != r.player_id:
                ev = self._pass_from(r, pid, ch, completed=True)
                self._control = _Control(pid, team, ch.t, ch.frame_idx, origin="pass",
                                         via_pass=_PassRec(r.player_id, r.team, r.t, ev.event_id))
            else:
                self._control = _Control(pid, team, ch.t, ch.frame_idx)
            self._set_possession(team, ch.t, ch.frame_idx, pid, "other")
            return
        miss = chg.miss
        assert miss is not None
        offensive = team is miss.team
        # the first holder was the wrong body; the rebound keeps its time when he was next to the new holder
        # (the right moment, the wrong body), else the new holder's gain is the rebound
        old = self._store.get(miss.rebound_id or "")
        retime = {}
        if old is not None and old.player_id is not None:
            p_old, p_new = self._pos(old.player_id, old.t), self._pos(pid, ch.t)
            far = self.config.rebound_retime_mono_ft if self._mono else self.config.rebound_retime_ft
            if p_old is not None and p_new is not None and math.dist(p_old, p_new) > far:
                retime = {"t": ch.t, "frame_idx": ch.frame_idx}
        self._revise(miss.rebound_id, team=team, player_id=pid,
                     x=(self._pos(pid, ch.t) or (None, None))[0], y=(self._pos(pid, ch.t) or (None, None))[1],
                     attrs={"offensive": offensive, "revised_by": "reversal", "reason": "first control after a "
                            "missed shot (re-credited: the earlier holder was the wrong body)"}, **retime)
        self._retract(miss.possession_id, reason)
        miss.possession_id = None
        self._poss_team = chg.poss_before
        reb = self._store.get(miss.rebound_id or "")
        rt, rf = (reb.t, reb.frame_idx) if reb is not None else (ch.t, ch.frame_idx)
        if not offensive:
            pev = self._set_possession(team, rt, rf, pid, "defensive_rebound")
            if pev is not None:
                miss.possession_id = pev.event_id
        self._control = _Control(pid, team, ch.t, ch.frame_idx, origin="rebound",
                                 change=self._provisional("rebound", ch.t, team, chg.poss_before, miss=miss))

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
        if self._pend is not None and r.team is self._pend.team and not flight.is_ft:
            self._confirm_pending()  # the new team shoots: its control is corroborated (the steal comes first)
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

    def _box_at(self, pid: str, t: float) -> BBox | None:
        """``pid``'s image box at (or just before) ``t`` from the short box history."""
        best = None
        for bt, boxes in self._boxes:
            if bt > t + 1e-6:
                break
            if pid in boxes:
                best = boxes[pid][0]
        return best

    def _first_box(self, pid: str, t0: float, t1: float) -> BBox | None:
        """``pid``'s first image box in ``[t0, t1]`` from the short box history."""
        for bt, boxes in self._boxes:
            if bt > t1 + 1e-6:
                break
            if bt >= t0 - 1e-6 and pid in boxes:
                return boxes[pid][0]
        return None

    def _detect_tip(self) -> list[ControlChange]:
        """3-D: a loose ball after a miss tipped back up (``tip_*``). The contact is inside the step between the
        two middle of the last four observations (the ball was falling into the hand, then rises): compare the
        velocities either side of it. Ring and board bounces are left to the flight."""
        cfg, court = self.config, self.court
        if self._holder.holder is not None:
            return []
        pts = [(t, fi, o) for t, fi, o in list(self._buffer)[-4:] if o is not None and o.is3d]
        if len(pts) < 4 or pts[-1][0] - pts[0][0] > 0.25:
            return []
        (t0, _, a), (t1, f1, b), (t2, _, c), (t3, _, d) = pts
        if min(t1 - t0, t3 - t2) <= 1e-6 or b.z is None or c.z is None or max(b.z, c.z) < cfg.tip_min_z_ft:
            return []
        vz1, vz3 = (b.z - a.z) / (t1 - t0), (d.z - c.z) / (t3 - t2)  # type: ignore[operator]
        if vz3 < cfg.tip_min_up_fts or vz3 - (vz1 - G_FTS2 * (t3 - t1)) < cfg.tip_min_dv_fts:
            return []
        contact = c if c.z >= b.z else b  # type: ignore[operator]
        cx, cy = contact.xy  # type: ignore[misc]
        side = court.nearest_basket(cx, cy)
        bx, by = court.basket_xy(side)
        dh = math.hypot(cx - bx, cy - by)
        if dh > cfg.tip_max_basket_ft:
            return []
        near_ring = math.hypot(dh - court.rim_radius, contact.z - court.rim_height) \
            <= cfg.ball_radius_ft + cfg.rim_contact_dist_ft + 0.3  # type: ignore[operator]
        board_x = court.backboard_offset if side == "left" else court.length - court.backboard_offset
        near_board = abs(cx - board_x) <= cfg.ball_radius_ft + cfg.rim_contact_dist_ft + 0.3 \
            and abs(cy - by) <= cfg.backboard_half_width_ft + cfg.ball_radius_ft
        if near_ring or near_board:
            return []
        best, best_d = None, cfg.tip_reach_ft
        for pid in self._players:
            pos = self._pos(pid, t1)
            if pos is None:
                continue
            dd = math.hypot(pos[0] - cx, pos[1] - cy)
            if dd <= best_d:
                best, best_d = pid, dd
        if best is None:
            return []
        log.debug("tip by %s at %.2f s (z %.1f ft)", best, t1, contact.z)
        return self._holder.touch(best, t1, f1)

    def _retro_block(self, flight: Flight) -> bool:
        """Monocular blocked shot seen only in hindsight. With the shooter's box grown to his raised arms and
        the ball, a release is often recognised only after a block 0.07-0.15 s later has already turned the
        ball back, so the flight never shows the shot-like rise. Look back over the buffered image track
        around the release: the ball rising above the shooter's standing head line at
        ``block_min_up_speed_fts`` or more, then turned down or back abruptly (``block_reverse_dv_img_fts``)
        inside an opponent's raised-arm box, away from the rim. A legal block touches the ball on its way
        up (goaltending, NBA rule 11 / FIBA art. 31), so a ball turned back after its peak is not one."""
        cfg, court = self.config, self.court
        r = flight.release
        if r.xy is None or r.team is None or flight.blocker is not None:
            return False
        side = court.nearest_basket(*r.xy)
        if court.distance_to_basket(r.xy[0], r.xy[1], side) > cfg.block_max_basket_dist_ft:
            return False
        stand = self._box_at(r.player_id, r.t - cfg.retro_block_stand_lead_s) or self._first_box(
            r.player_id, r.t - cfg.retro_block_stand_lead_s, r.t) or r.bbox  # (a track renamed since: his first box)
        if stand is None:
            return False
        ppf = max(stand.h / cfg.player_height_ft, 1e-6)
        head_v = stand.y1 + cfg.holder_head_frac * stand.h
        lo = r.t - cfg.retro_block_lookback_s
        if r.control_onset_t is not None:
            lo = max(lo, r.control_onset_t)  # the shot rises from his hands: not the pass that reached him
        pts = [(t, f, o.uv) for t, f, o in self._buffer if o is not None and o.uv is not None and lo <= t]
        rim = self._rims.get(side)
        span, base = cfg.block_image_span, cfg.retro_block_min_base_s
        g_px = G_FTS2 * ppf
        for k in range(span, len(pts) - 2 * span):
            (t0, _, a), (t1, f1, b) = pts[k - span], pts[k]
            (t2, _, c), (t3, _, d) = pts[k + span], pts[k + 2 * span]
            if b[1] > head_v or not (base <= t1 - t0 <= cfg.retro_block_max_gap_s) \
                    or not (base <= t2 - t1 <= cfg.retro_block_max_gap_s) or t3 - t2 < base:
                continue
            if t1 - r.t > cfg.block_max_after_release_s:
                break
            v1 = ((b[0] - a[0]) / (t1 - t0), (b[1] - a[1]) / (t1 - t0))
            v2 = ((c[0] - b[0]) / (t2 - t1), (c[1] - b[1]) / (t2 - t1))
            v3 = ((d[0] - c[0]) / (t3 - t2), (d[1] - c[1]) / (t3 - t2))
            up1, up2, up3 = -v1[1] / ppf, -v2[1] / ppf, -v3[1] / ppf
            back = cfg.block_min_toward_fts
            if up1 < cfg.block_min_up_speed_fts:
                continue  # a touch on the way up only
            knocked_down = up2 <= -back and up3 <= -back
            sgn = 1.0 if rim is None or rim.center[0] >= b[0] else -1.0  # image direction of the rim
            turned_back = (rim is not None and sgn * v1[0] / ppf >= -back and sgn * v2[0] / ppf <= -back
                           and sgn * v3[0] / ppf <= -back and up2 <= cfg.retro_block_turn_rise_frac * up1)
            if not (knocked_down or turned_back):
                continue
            # velocity change beyond gravity's (image v grows downward)
            dv = math.hypot(v2[0] - v1[0], v2[1] - v1[1] - g_px * 0.5 * (t2 - t0)) / ppf
            if dv < cfg.block_reverse_dv_img_fts:
                continue
            if rim is not None:
                w = max(rim.w, 1.0)
                if rim.x1 - w <= b[0] <= rim.x2 + w and rim.y1 - 2.0 * w <= b[1] <= rim.y2 + w:
                    continue  # at the rim: the rim-zone logic decides (ring bounces look the same here)
            blocker = self._raised_arm_opponent(r, b, t1)
            if blocker is None:
                continue
            # a candidate: committed when someone gains the ball before it reached a rim zone (a ball that
            # then goes to the rim was a shot after all; the flight drops the candidate on zone entry)
            t_rise, f_rise = pts[k - span][0], pts[k - span][1]
            flight.retro_block = (t1, f1, blocker, side, {"reversal_dv": dv, "retro": True, "rise_t": t_rise,
                                                          "rise_frame": f_rise})
            return True
        return False

    def _raised_arm_opponent(self, r: Release, uv: tuple[float, float], t: float) -> str | None:
        """An opponent of the releaser whose raised-arm box (``block_reach_frac`` above his head, the top
        ``block_shoulder_frac`` of his body) holds the ball at ``t``; the one nearest its centre line."""
        cfg = self.config
        boxes = None
        for bt, bx in self._boxes:
            if bt > t + 1e-6:
                break
            boxes = bx
        if boxes is None or r.team is None:
            return None
        best, best_d = None, math.inf
        for pid, (bb, team) in boxes.items():
            if (team or self._team_of(pid)) is not r.team.other():
                continue
            dx = bb.w * cfg.block_bbox_expand / 2.0
            if not (bb.x1 - dx <= uv[0] <= bb.x2 + dx and bb.y1 - cfg.block_reach_frac * bb.h <= uv[1]
                    <= bb.y1 + cfg.block_shoulder_frac * bb.h):
                continue
            d = abs(uv[0] - (bb.x1 + bb.x2) / 2.0)
            if d < best_d:
                best, best_d = pid, d
        return best

    def _on_block(self, flight: Flight, sig: FlightSignal) -> None:
        entry = self._entry(flight) or self._emit_attempt(flight)  # a block in the rim zone: attempt already out
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
        if flight.continuation and not made:
            # fouled in the act and missed: no field-goal attempt, the free throws follow (no rebound). The
            # image make test can be wrong, so the scoreboard may still flip it to an and-one make first.
            self._cont_misses.append(entry)
            self._flight = None if self._flight is flight else self._flight
            return
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
        f = self.deadball.formation_at(t)  # as of the release frame, not the (later) decision frame
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
        if obs.t - t > cfg.dribble_max_rise_s:
            return  # back up long after the low point: a ball rolled and then picked up, not a bounce
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
            jump = d is not None and d.cause == "period" and self._mono and self._jump_expected()
            if d is not None and d.cause not in ("break",) and not jump and self._dead_for(d, t) >= self.config.dead_min_s \
                    and self._holder.holder == pid \
                    and pos is not None and self.court.in_bounds(*pos) and (self._ft is None or self._ft.shooter != pid):
                self._restart(self._team_of(pid), t, f, pid, "live")
            else:
                return
        if ctrl is not None:
            ctrl.dribbles += 1
        self._note_pending_evidence(self._team_of(pid), t)
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
        fl = self._flight
        if fl is not None and (fl.state == "shot" or (fl.state == "air" and fl.release.t < since)):
            return  # a ball released before the stoppage may still be a shot: wait for it to resolve
        # (a ball tossed to the official after the whistle is dead from the stoppage on: a travel or a
        # step out of bounds is dated by the clock stop, not by the inbound seconds later)
        self._clock_stop_used = since
        self._enter_dead_from_live("unknown", since, self._frame_near(since), fl, out=False)

    def _clock_stop_before(self, t: float) -> float | None:
        """Start of a game-clock stoppage still in effect at ``t`` (live play since before it), or None."""
        sb, cfg = self.scoreboard, self.config
        since = sb.clock_stopped_since
        if not cfg.dead_from_clock_stop or since is None or since < self._live_since or since > t:
            return None
        if not sb.clock_stopped(t, 2, cfg.clock_stop_min_s) or (sb.game_clock or 0.0) <= 0.05:
            return None
        return since


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
        for entry in list(self._cont_misses):
            if entry.made is not False:
                self._cont_misses.remove(entry)  # flipped to a make by the scoreboard: an and-one
            elif force or now - (entry.result_t or entry.t_release) > self.config.continuation_miss_wait_s:
                self._cont_misses.remove(entry)
                entry.confirmed = True
                for eid in (entry.attempt_id, entry.result_id):
                    self._retract(eid, "fouled in the act of shooting and the shot missed: no FGA")
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
            if self._unseen_make(ch):
                return
            self._emit(EventKind.SCORE_CORRECTION, ch.t, self._frame_near(ch.t), team=ch.team, value=ch.delta,
                       source="scoreboard", status=EventStatus.CONFIRMED, prefix="r",
                       attrs={"needs_review": True, "home": ch.home, "away": ch.away,
                              "reason": ch.reason or "scoreboard change with no matching shot"})

    def _unseen_make(self, ch: ScoreChange) -> bool:
        """A confirmed +2 / +3 that matches no shot the engine saw (monocular: a layup or dunk carried through
        the rim inside the shooter's box, an and-one whose shooter was never the holder) is a made field
        goal all the same: SHOT_ATTEMPT + SHOT_MADE (needs_review, low confidence) credited to the fouled
        shooter, else to the team's last live releaser near its basket, dated at that release and at the
        ball's next sighting at the rim. What followed the make is re-read as a dead ball after a score: an
        out-of-bounds turnover inferred for the scoring team is retracted, and the other team's possession
        starts after a made shot, not a turnover or a steal."""
        cfg = self.config
        team, value = ch.team, ch.delta
        if not cfg.infer_unseen_makes or team is None or value not in (2, 3):
            return False
        half = self._half_index(self._period)
        known = self._voted_side(team, half) is not None or self._voted_side(team, 1 - half) is not None
        attack = self.attacking_side.get(team) if known else None  # else: the basket nearest the release
        lo = ch.t - cfg.unseen_make_window_s
        cand: tuple[str, float, int, str] | None = None
        for ft, ff, pid in reversed(self._shoot_fouls):
            if lo <= ft <= ch.t and self._team_of(pid) is team:
                cand = (pid, ft, ff, "shooter fouled in the act")
                break
        ctrl: _Control | None = None
        rel: Release | None = None
        for r, c in reversed(self._release_log):
            if r.t < lo:
                break
            if r.t > ch.t + cfg.unseen_make_release_lag_s:
                continue
            if cand is not None and r.player_id == cand[0] and abs(r.t - cand[1]) <= cfg.continuation_s:
                rel, ctrl = r, c
                cand = (r.player_id, r.t, r.frame_idx, cand[3])  # his release, if it was seen
                break
            if cand is None and r.team is team and r.t <= ch.t + cfg.unseen_make_release_lag_s and r.xy is not None:
                near = attack or self.court.nearest_basket(r.xy[0], r.xy[1])
                if self.court.distance_to_basket(r.xy[0], r.xy[1], near) <= cfg.unseen_make_max_dist_ft:
                    rel, ctrl = r, c
                    cand = (r.player_id, r.t, r.frame_idx, "last release near its basket")
                    break
        if cand is None:
            return False
        pid, rt, rf, why = cand
        pos = rel.xy if rel is not None else self._pos(pid, rt)
        side = attack or (self.court.nearest_basket(pos[0], pos[1]) if pos is not None else None)
        if side is None:
            return False
        # the ball at the rim: after the release, or before a release seen late (carried through the rim)
        made_t = self._rim_sighting(side, rt - cfg.unseen_make_release_lag_s, ch.t)
        if made_t is None:
            made_t = rt + cfg.unseen_make_flight_s
        elif made_t < rt + cfg.unseen_make_flight_s:
            rt = min(rt, made_t - cfg.unseen_make_flight_s)
            rf = self._frame_near(rt)
        mf = self._frame_near(made_t)
        dist = self.court.distance_to_basket(pos[0], pos[1], side) if pos is not None else None
        self._shot_seq += 1
        shot_id = f"s{self._shot_seq:04d}"
        reason = f"scoreboard +{value} matched no seen shot ({why})"
        att = self._emit(EventKind.SHOT_ATTEMPT, rt, rf, team=team, player_id=pid, xy=pos, value=value, shot_id=shot_id,
                         confidence=cfg.unseen_make_conf,
                         attrs={"basket": side, "inferred": "scoreboard", "needs_review": True, "reason": reason,
                                "distance_ft": dist, "release_t": rt,
                                "shot_type": "layup" if dist is not None and dist <= cfg.layup_max_dist_ft else "jumper"})
        made = self._emit(EventKind.SHOT_MADE, made_t, mf, team=team, player_id=pid, value=value, shot_id=shot_id,
                          confidence=cfg.unseen_make_conf, status=EventStatus.CONFIRMED,
                          attrs={"inferred": "scoreboard", "confirmed_by": "scoreboard", "needs_review": True,
                                 "reason": reason})
        entry = ShotEntry(shot_id, team, pid, value, att.event_id, rt, result_id=made.event_id, made=True,
                          result_t=made_t, confirmed=True, score_matched=True)
        if ctrl is not None and rel is not None:
            entry.assist_candidate = self._assist_candidate(ctrl, rel)
        self._shots[shot_id] = entry
        self._recent.append(entry)
        self._emit_assist(entry, made_t, mf)
        self._after_unseen_make(team, made_t, entry)
        return True

    def _rim_sighting(self, side: Side, t0: float, t1: float) -> float | None:
        """Last buffered ball observation at ``side``'s rim (image box, a rim width around it) in (t0, t1]:
        a made ball leaves that region downward through the net, so it is about the crossing."""
        rim = self._rims.get(side)
        if rim is None:
            return None
        w = max(rim.w, 1.0)
        last = None
        for t, _, o in self._buffer:
            if o is None or o.uv is None or not t0 < t <= t1:
                continue
            if rim.x1 - w <= o.uv[0] <= rim.x2 + w and rim.y1 - 2.0 * w <= o.uv[1] <= rim.y2 + w:
                last = t
        return last

    def _after_unseen_make(self, team: Team, made_t: float, entry: ShotEntry) -> None:
        """Re-read what followed an inferred make as the dead ball after a score."""
        reason = "the scoreboard shows a make here: dead ball after a score"
        other = team.other()
        for ev in list(self._store.values()):
            if ev.status is EventStatus.RETRACTED or ev.t < made_t - 0.2 or ev.source not in ("cv",):
                continue
            if ev.kind is EventKind.TURNOVER and ev.team is team and "inferred" in str(ev.attrs.get("reason", "")):
                self._retract(ev.event_id, reason)
            elif ev.kind is EventKind.STEAL and ev.team is other and ev.t <= made_t + 3.0:
                self._retract(ev.event_id, reason)
                for tv in list(self._store.values()):
                    if tv.kind is EventKind.TURNOVER and tv.team is team and abs(tv.t - ev.t) < 1e-6:
                        self._retract(tv.event_id, reason)
            elif ev.kind is EventKind.POSSESSION and ev.team is other and ev.attrs.get("cause") in ("turnover", "steal",
                                                                                                    "other"):
                self._revise(ev.event_id, attrs={"cause": "made_shot", "reason": reason})
                entry.possession_id = ev.event_id
        d = self._dead
        if d is not None and d.since_t >= made_t - 0.5 and d.cause in ("oob", "unknown"):
            d.cause, d.scoring_team, d.shot_id = "made_fg", team, entry.shot_id
            d.last_team = d.last_holder = None

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
        d = self._dead
        cause = "made_ft" if entry.is_ft else "made_fg"
        if self._pending_miss is entry or (unresolved and not entry.is_ft):
            # no rebound yet: the dead ball after a make (a last free throw called missed had sent play live)
            self._pending_miss = None
            self._flight = None
            res_t = entry.result_t if entry.result_t is not None else entry.t_release
            res = self._store.get(entry.result_id or "")
            self._enter_dead(cause, res_t, res.frame_idx if res else self._frame_near(res_t), scoring_team=entry.team,
                             shot_id=entry.shot_id)
            self.deadball.note_stoppage(res_t, "made_basket")
        elif d is not None and d.miss is entry:
            # a dead ball (clock stop, out of bounds) entered while the miss was still pending: it follows the make,
            # so the other team's inbound is the possession after a score, not a team rebound / turnover
            d.miss, d.cause, d.scoring_team, d.shot_id = None, cause, entry.team, entry.shot_id
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
        if (d is None or d.shot_id != entry.shot_id) and not entry.is_ft:
            self._late_make_to_miss(entry)
            return
        if d is None or d.shot_id != entry.shot_id or entry.is_ft:
            return
        if d.first_pickup is not None:
            t, f, pid = d.first_pickup
            ch = ControlChange("gain", pid, t, f)
            self._go_live(t)
            poss_before, team = self._poss_team, self._team_of(pid)
            self._rebound(pid, team, ch, entry)
            ctrl = self._control
            if ctrl is not None and ctrl.player_id == pid:
                ctrl.origin = "rebound"
                ctrl.change = self._provisional("rebound", t, team, poss_before, miss=entry)
        else:
            self._go_live(now)
            self._pending_miss = entry

    def _late_make_to_miss(self, entry: ShotEntry) -> None:
        """A make revised to a miss after play restarted: the first pickup of the post-make dead ball was
        the rebound (REBOUND at its time; its POSSESSION is a defensive rebound, not "made_shot")."""
        pm = self._post_make
        if pm is None or pm[0] != entry.shot_id:
            return
        self._post_make = None
        (t, f, pid), poss_id = pm[1], pm[2]
        team = self._team_of(pid)
        self._retract(poss_id, "the shot was not made: the pickup after it was the rebound")
        saved = self._poss_team
        self._poss_team = entry.team  # the shooting team had the ball until the rebound
        self._rebound(pid, team, ControlChange("gain", pid, t, f), entry)
        if saved is not None:
            self._poss_team = saved
        ctrl = self._control
        if ctrl is not None and ctrl.player_id == pid:
            ctrl.origin = "rebound"
            ctrl.change = self._provisional("rebound", t, team, entry.team, miss=entry)

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

    def _in_the_act(self, ev: Event) -> Flight | None:
        """A shooting foul on a player in the act of shooting: the shot plays out (continuation, FIBA / NBA
        rules; an and-one when it goes in). Returns his live flight when it is already in the air; when he
        still holds the ball, his next release within ``continuation_s`` is a live shot."""
        shooter = ev.other_player_id
        if shooter is None:
            return None
        fl = self._flight
        if fl is not None and fl.context == "live" and not fl.resolved and fl.release.player_id == shooter \
                and ev.t - fl.release.t <= self.config.continuation_s:
            fl.continuation = True
            return fl
        ctrl = self._control
        if self._holder.holder == shooter or (ctrl is not None and ctrl.player_id == shooter):
            self._continuation = (shooter, ev.t)
        return None

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
        self._note_pending_evidence(self._team_of(ev.other_player_id), ev.t)  # fouled while holding: control
        if ftype == "shooting" and ev.other_player_id:
            self._shoot_fouls.append((ev.t, ev.frame_idx, ev.other_player_id))
        keep = self._in_the_act(ev) if ftype == "shooting" else None
        if self._phase == LIVE:
            self._enter_dead_from_live("unknown", ev.t, ev.frame_idx, self._flight, out=False)
        if keep is not None:
            self._flight = keep  # the shot released in the act of shooting plays out (continuation)
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


_PROVISIONAL = re.compile(r"^(home|away|unk)-t\d+$")  # perception's provisional ids (types.PlayerState)
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

