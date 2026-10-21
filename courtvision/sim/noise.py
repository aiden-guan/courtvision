"""Observation noise for simulator truth frames (what a perception stack would plausibly output).

``NoiseModel(config, camera).apply(frames, seed)`` returns an observed copy of truth frames, built to
fail the way a single fixed camera + detector + tracker fails (RESEARCH §2.1-§2.3, cv_pipeline §2-§6):

* **players**: court positions from one error source, the foot point (bbox bottom centre) off by an
  AR(1)-correlated pixel error (tracker errors persist over frames and do not average out) projected to
  the floor, so the error grows with depth and lies mostly along it, and the box moves with it; the foot
  point of a jumping player lifted off the floor (``feet_z``: the floor projection of a lifted foot lands
  behind the player, cv_pipeline §11 #8), bbox jitter, short team-label flip episodes (a provisional id, no
  jersey), detection confidences below 1 (lower when covered or doubtful), velocities
  estimated from the noisy positions by the same alpha-beta smoothing the video pipeline uses (never the
  simulator's own velocities), i.i.d. drops plus clustered drops while a nearer body covers the player
  (screens merge bodies for 10-30 frames, §3.4), and identity swaps between same-team players after an
  image occlusion episode (plus the legacy random ``id_swap_rate``);
* **ball**: detector misses from a 2-state burst process (Gilbert-Elliott: misses come in runs), a
  held ball hidden in the hands, the ball hidden behind a nearer body (depth order: a near-certain state
  while it lasts, as in video) and in the net
  after a downward rim-plane crossing (§6.2, §6.4); persistent false positives (spare / rack balls
  that stay put, head-like blobs that follow a player) whose confidence overlaps true detections, and
  the single-ball output keeps the more confident candidate; image-centre jitter;
* rim-box jitter; in ``monocular`` mode no ball height and a ball ``court_xy`` equal to the floor-
  homography projection of the noisy image point (only right when the ball is on the floor, exactly
  like monocular video).

Scoreboard readings are kept (``ScoreboardModel`` below makes the operator-run, OCR-read version).
The input frames are never modified.

The floor homography (and the depth order) comes from the camera when one is given; otherwise (the
ARCHITECTURE call ``NoiseModel(NoiseConfig()).apply(frames, seed)``) the 3x4 projection is recovered from
the truth frames themselves: a normalised-DLT fit to the true ball centres (court_xy, z) and their image
points, which the simulator computes with that same camera.

Every rate below is a design choice (status unverified) unless its comment cites a source.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from courtvision.geometry.camera import PinholeCamera
from courtvision.geometry.homography import Homography
from courtvision.types import BallState, BBox, FrameState, PlayerState, RimObservation, ScoreboardReading, Team

__all__ = ["NoiseConfig", "NoiseModel", "ScoreboardConfig", "ScoreboardModel", "floor_homography_from_frames",
           "projection_from_frames"]

log = logging.getLogger(__name__)

_RIM_HEIGHT_FT = 10.0  # every supported league (geometry.court); the net window only needs the plane


@dataclass
class NoiseConfig:
    """Noise levels (design choices, unverified, unless a comment cites a source)."""

    # Player court-position error, from one source: the foot point (bbox bottom centre) is off by a slow
    # AR(1) pixel error, and the reported court point is its floor projection, so the error grows with depth
    # (this camera's floor scale is ~0.125 / 0.21 / 0.33 ft per px, near / mid / far sideline) and mostly
    # lies along the depth direction; the same pixel error moves the box's bottom edge and centre (it used to
    # be an isotropic court error independent of depth and of the box). pos_sigma_ft is the error scale at the
    # mid-court reference depth (sqrt of the floor-scale area there x the pixel sd); 0 = exact. Without a
    # camera (no floor homography) it falls back to an isotropic court error of that sd. Design, unverified.
    pos_sigma_ft: float = 0.4
    # AR(1) frame-to-frame correlation of that error. Tracked positions err mostly by slowly varying
    # biases (foot point, pose, homography); the tracker's filter removes most frame jitter. 0.995 at
    # 30 fps (correlation time ~7 s) gives a velocity error of ~0.8 ft/s median, the video pipeline's
    # measured ~0.7 ft/s (sim train seed 1); tuned.
    pos_rho: float = 0.995
    bbox_sigma_px: float = 2.0  # per-edge bbox jitter (marginal sd)
    bbox_rho: float = 0.7  # AR(1) correlation of the bbox jitter
    foot_lift: bool = True  # a jumping player's court point follows his lifted foot point (needs feet_z)
    # Velocities from the noisy positions: alpha-beta gains, speed clamp and reset jump of the video
    # pipeline's court-space smoothing (perception.pipeline.PerceptionConfig smoothing_alpha / _beta /
    # max_speed_ft_s / reset_jump_ft), so mono and video see the same kind of speed estimate.
    vel_alpha: float = 0.6
    vel_beta: float = 0.15
    vel_max_fts: float = 30.0
    vel_reset_ft: float = 8.0
    # Detector miss rate (burst-process mean) of an unoccluded ball in flight / loose, and of an
    # unoccluded held ball (detector + hands). Body occlusion and the net come on top: with them the
    # realised rates on train seeds 1-6 (300 s) stay at ~0.19 not held and ~0.35 held, their levels
    # before body occlusion became a near-certain state (round 2: base 0.08 -> 0.06, held 0.17 ->
    # 0.09); tuned on sim train seeds (noise only).
    ball_miss_rate: float = 0.06
    ball_miss_rate_held: float = 0.09
    ball_burst_frames: float = 4.0  # mean length of a "bad" detector state (misses come in runs)
    ball_bad_miss: float = 0.85  # miss probability in the bad state
    ball_good_miss: float = 0.03  # miss probability in the good state
    # Occlusion is a state, not an i.i.d. per-frame coin: while the ball's image point lies behind a
    # nearer upper body (the holder's own when the ball is on his far side included) it is missed on
    # nearly every frame, as video renders it hidden on every such frame (RESEARCH §2.1, cv_pipeline
    # §6.2: a held / occluded ball vanishes). Was 0.75 per frame, which still "detected" a fully hidden
    # ball on ~21% of its frames (realised miss 0.79 -> 0.97). Value unverified (a residual for the ball
    # edge peeking out).
    ball_occluded_miss: float = 0.97
    ball_behind_ft: float = 0.4  # ... i.e. the ball is this much farther from the camera than the body axis
    ball_hide_frac: float = 0.55  # ... and behind the upper body (top share of the box); legs leave gaps
    ball_net_frames: int = 3  # frames hidden in the net after a downward crossing (cv_pipeline §6.4: ~2)
    ball_net_miss: float = 0.8
    ball_fp_rate: float = 0.01  # share of frames with a false-positive candidate (persistent episodes)
    ball_fp_episode_s: tuple[float, float] = (0.2, 1.5)  # how long one false-positive source persists
    ball_fp_head_share: float = 0.5  # share of episodes that follow a player's head (else a static blob)
    ball_fp_present: float = 0.85  # P(the source is detected on a frame of its episode)
    ball_fp_conf: tuple[float, float] = (0.35, 0.85)  # overlaps true detections (cv_pipeline §2.1)
    ball_conf: tuple[float, float] = (0.5, 0.95)  # true detections
    ball_px_sigma: float = 1.5  # ball image-centre jitter
    ball_radius_sigma_px: float = 0.3
    drop_player_rate: float = 0.02  # player missing from a frame (i.i.d.)
    occluded_drop_rate: float = 0.15  # extra per-frame drop while >= half the box is behind a nearer body
    occluded_cover: float = 0.5  # covered box share that counts as occluded
    id_swap_rate: float = 0.0  # legacy: P(start a random nearby same-team identity swap) per frame
    id_swap_frames: int = 15  # legacy swap duration
    id_swap_max_ft: float = 5.0  # legacy: only nearby same-team players swap
    # Occlusion swaps: two same-team players whose image boxes overlap (IoU >= occlusion_iou) for >=
    # occlusion_frames frames come out of it with swapped identities with this probability, kept for
    # occlusion_swap_s until a jersey re-read fixes it (cv_pipeline §3.2/§3.4: ID switches cluster in
    # occlusions; Basketball-SORT still 9.3 ID switches per sequence); calibrated so the mono identity
    # accuracy stays near the video pipeline's measured ~0.92 (train seeds; tuned).
    occlusion_swap_prob: float = 0.35
    occlusion_iou: float = 0.3
    occlusion_frames: int = 8
    occlusion_swap_s: tuple[float, float] = (1.0, 3.0)
    # Team-label errors: a track's team vote flips for a short episode (team_flip_s) in team_flip_share of
    # frames; the flipped detection cannot be mapped to the roster, so it carries the shown team's provisional
    # id "<team>-t<track>" and no jersey (ARCHITECTURE PlayerState). Calibrated to the video pipeline's
    # measured team accuracy 0.990-0.993 (train seeds, eval.tracking); tuned. Mono used to be 1.0 always.
    team_flip_share: float = 0.008
    team_flip_s: tuple[float, float] = (0.3, 1.5)
    # Detection confidence: uniform in player_conf, scaled down by the covered share of the box and for an
    # identity in doubt, judged only from what a tracker can see: a provisional (unidentified) label, or a
    # same-team overlap episode of >= occlusion_frames that is under way or ended within the last
    # occlusion_swap_s[1] (whether or not it actually swapped the identities). It used to be lowered exactly
    # when the shown identity was wrong (a truth signal). Design, unverified (was 1.0 always).
    player_conf: tuple[float, float] = (0.75, 0.98)
    conf_covered_drop: float = 0.4
    conf_doubt: float = 0.85
    # Track fragmentation: a real tracker starts new track ids (after a player is lost, and re-initialising on
    # a jumpy box without a gap); the identity layer re-links them to the roster, at once or after a jersey
    # re-read (reads every 5 frames, 3 agreeing reads; RESEARCH §2.2). A reappearing detection gets a new
    # track id with frag_after_drop_1 after a 1-frame gap, rising by frag_per_frame per further missed frame (up
    # to 1); a visible one with frag_hazard per frame; a new track carries a provisional identity ("<team>-t
    # <track>", no jersey) with frag_provisional for frag_relink_s. Mono track ids used to be the player index
    # for the whole game. Calibrated to the video pipeline on train seeds 1-3 (180 s sim renders: 7.5-9.0
    # track segments per player-minute, provisional share 1.1-2.8% of matched detections; mono drops are
    # mostly 1-frame i.i.d. gaps that a tracker coasts through, so the per-gap rates differ); tuned.
    frag_after_drop_1: float = 0.015
    frag_per_frame: float = 0.25
    frag_hazard: float = 0.0003
    frag_provisional: float = 0.3
    frag_relink_s: tuple[float, float] = (0.2, 1.2)
    # Phantom people (calibrated to the video pipeline on train seeds 1-3, 180 s renders; tuned): a second
    # detection of a real player (a split / duplicate box dup_offset_ft away, 0.12-0.16 per frame in video,
    # median 3 frames: dup_hazard per visible player per frame, episodes of dup_frames mean frames) labelled
    # provisional / with another roster id of his team / with his own id (video: ~35 / 45 / 20%); a referee
    # detected as a player (0.3-0.8% of frames: ref_hazard per referee per frame, ref_frames long); a
    # substitute waiting at the table detected (98.5% of his frames in video; waiting_detect), identified with
    # waiting_id. Needs ``referees`` / ``waiting`` (SimResult) for the last two. Mono used to emit only the ten
    # on-court players (PlayerState is contracted as never a referee, yet perception lets some through).
    dup_hazard: float = 0.003
    dup_frames: float = 4.7
    dup_offset_ft: tuple[float, float] = (1.0, 2.5)
    dup_labels: tuple[float, float, float] = (0.35, 0.45, 0.2)  # provisional / other roster id / own id
    dup_conf: tuple[float, float] = (0.3, 0.8)
    ref_hazard: float = 0.00013
    ref_frames: tuple[int, int] = (3, 30)
    ref_height_ft: float = 6.2  # SimConfig.referee_height_ft
    waiting_detect: float = 0.95
    waiting_id: float = 0.7
    person_height_ft: float = 6.5  # cylinder of an off-court person (SimConfig.player_height_ft)
    person_radius_ft: float = 1.0
    rim_px_sigma: float = 0.5
    monocular: bool = True  # ball z -> None, court_xy -> floor projection of the image point
    held_radius_ft: float = 2.5  # "held" heuristic: ball within this of a player ...
    held_max_z_ft: float = 6.0  # ... and below this height
    fps: float = 30.0  # frame rate of the input (correlations and durations are per frame at this rate)


@dataclass
class _FPEpisode:
    frames_left: int
    kind: str  # "head" | "static"
    conf: float
    uv: tuple[float, float] = (0.0, 0.0)  # static blob position
    pid: str | None = None  # head: the player it follows


@dataclass
class _State:
    """Per-``apply`` noise state (deterministic for a seed)."""

    rng: np.random.Generator
    pos_err: dict[str, np.ndarray] = field(default_factory=dict)  # foot-point pixel error (or court ft, no camera)
    box_err: dict[str, np.ndarray] = field(default_factory=dict)
    vel: dict[str, tuple[np.ndarray, np.ndarray, float]] = field(default_factory=dict)  # pid -> (x, v, t)
    bad: bool = False  # ball detector burst state
    net_left: int = 0
    fp: _FPEpisode | None = None
    overlap: dict[tuple[str, str], int] = field(default_factory=dict)  # same-team pair -> frames overlapping
    swaps: dict[str, tuple[str, int]] = field(default_factory=dict)  # pid -> (shown identity, frames left)
    team_flips: dict[str, int] = field(default_factory=dict)  # pid -> frames left of a team-label error
    legacy: tuple[str, str, int] | None = None
    prev_frame: FrameState | None = None
    overlap_end: dict[str, int] = field(default_factory=dict)  # pid -> frame a long same-team overlap ended
    frame_i: int = 0
    # track fragmentation: truth pid -> [track id, provisional frames left, frames missing]
    tracks: dict[str, list[int]] = field(default_factory=dict)
    next_track: int = 1000
    dups: dict[str, list] = field(default_factory=dict)  # truth pid -> [frames left, dx, dy, label, track id, conf]
    refs: dict[int, list] = field(default_factory=dict)  # referee index -> [frames left, team, track id, conf]
    waiting_tracks: dict[str, tuple[int, bool]] = field(default_factory=dict)  # sub pid -> (track id, identified)
    shown: dict[str, PlayerState] = field(default_factory=dict)  # this frame: truth pid -> its observed detection


class NoiseModel:
    """Applies :class:`NoiseConfig` noise to truth frames. ``camera`` gives the monocular floor
    projection and the depth order; without it both are recovered from the frames' 3-D ball truth."""

    def __init__(self, config: NoiseConfig | None = None, camera: PinholeCamera | None = None) -> None:
        self.config = config or NoiseConfig()
        self.camera = camera
        self._floor: Homography | None = camera.floor_homography() if camera is not None else None
        self._P: np.ndarray | None = np.asarray(camera.P, dtype=float) if camera is not None else None
        self._px_sigma: float | None = None  # foot-point pixel sd (per camera; _foot_sigma_px)

    def apply(self, frames: list[FrameState], seed: int = 0, camera: PinholeCamera | None = None,
              feet_z: Sequence[Sequence[float]] | None = None,
              referees: Sequence[Sequence[tuple[float, float]]] | None = None,
              waiting: Sequence[Sequence[tuple[str, str, float, float]]] | None = None) -> list[FrameState]:
        """Observed copies of ``frames`` (deterministic for a given ``seed``). ``feet_z`` (optional, per
        frame in ``frame.players`` order) is the simulator's feet height of each player; ``referees`` /
        ``waiting`` (optional, per frame: ``SimResult.referees`` / ``.waiting``) are the people a detector
        can mistake for players."""
        if camera is not None and camera is not self.camera:
            self.camera, self._floor, self._P = camera, camera.floor_homography(), np.asarray(camera.P, dtype=float)
            self._px_sigma = None
        fitted = self._P is None
        if fitted:
            self._P = projection_from_frames(frames)
            if self._floor is None and self.config.monocular:
                self._floor = _floor_from_projection(self._P)
                if self._floor is None:
                    log.warning("monocular noise without a camera or 3-D ball truth: ball court_xy will be None")
        try:
            return self._apply(frames, seed, feet_z, referees, waiting)
        finally:
            if fitted:  # fitted per call: never reused for other frames
                self._P, self._px_sigma = None, None
                if self.camera is None:
                    self._floor = None

    def _apply(self, frames: list[FrameState], seed: int, feet_z: Sequence[Sequence[float]] | None,
               referees: Sequence[Sequence[tuple[float, float]]] | None,
               waiting: Sequence[Sequence[tuple[str, str, float, float]]] | None) -> list[FrameState]:
        st = _State(rng=np.random.default_rng(seed))
        ph_rng = np.random.default_rng([seed, 47])  # phantom people: their own stream
        out: list[FrameState] = []
        for i, f in enumerate(frames):
            st.frame_i = i
            lifts = feet_z[i] if feet_z is not None and i < len(feet_z) else None
            st.legacy = self._update_legacy_swap(f, st.legacy, st.rng)
            self._update_occlusion_swaps(f, st)
            players = self._players(f, st, lifts)
            players += self._phantoms(f, st, players, ph_rng, referees[i] if referees is not None and i < len(referees)
                                      else None, waiting[i] if waiting is not None and i < len(waiting) else None)
            out.append(FrameState(
                frame_idx=f.frame_idx, t=f.t, players=players,
                ball=self._ball(f, st), rims=[self._rim(r, st.rng) for r in f.rims],
                scoreboard=f.scoreboard, image_size=f.image_size))
        return out

    # ------------------------------------------------------------------ geometry helpers

    def _depth(self, xyz: Sequence[float]) -> float | None:
        if self._P is None:
            return None
        return float(self._P[2] @ np.array([xyz[0], xyz[1], xyz[2], 1.0]))

    def _lift_shift(self, x: float, y: float, h: float) -> tuple[float, float]:
        """Floor displacement of the foot point of a player whose feet are ``h`` ft off the floor: the
        camera ray through the lifted foot meets the floor behind him."""
        if h <= 1e-6 or self._P is None or self._floor is None:
            return 0.0, 0.0
        q = self._P @ np.array([x, y, h, 1.0])
        if not q[2] > 0:
            return 0.0, 0.0
        xy = self._floor.image_to_court(q[:2] / q[2], front_only=True)
        if not np.all(np.isfinite(xy)):
            return 0.0, 0.0
        return float(xy[0]) - x, float(xy[1]) - y

    # ------------------------------------------------------------------ players

    def _update_legacy_swap(self, f: FrameState, swap: tuple[str, str, int] | None,
                            rng: np.random.Generator) -> tuple[str, str, int] | None:
        c = self.config
        if swap is not None:
            return (swap[0], swap[1], swap[2] - 1) if swap[2] > 1 else None
        if c.id_swap_rate <= 0 or rng.random() >= c.id_swap_rate:
            return None
        pairs = [(a, b) for i, a in enumerate(f.players) for b in f.players[i + 1:]
                 if a.team is b.team and math.hypot(a.x - b.x, a.y - b.y) <= c.id_swap_max_ft]
        if not pairs:
            return None
        a, b = pairs[int(rng.integers(len(pairs)))]
        return (a.player_id, b.player_id, c.id_swap_frames)

    def _update_occlusion_swaps(self, f: FrameState, st: _State) -> None:
        """Same-team image occlusion episodes end in an identity swap with ``occlusion_swap_prob``."""
        c = self.config
        for pid in list(st.swaps):
            other, left = st.swaps[pid]
            if left <= 1:
                del st.swaps[pid]
            else:
                st.swaps[pid] = (other, left - 1)
        live: set[tuple[str, str]] = set()
        ps = [p for p in f.players if p.bbox is not None]
        for i, a in enumerate(ps):
            for b in ps[i + 1:]:
                if a.team is not b.team or _iou(a.bbox, b.bbox) < c.occlusion_iou:
                    continue
                key = (a.player_id, b.player_id) if a.player_id < b.player_id else (b.player_id, a.player_id)
                live.add(key)
                st.overlap[key] = st.overlap.get(key, 0) + 1
        for key in [k for k in st.overlap if k not in live]:
            n = st.overlap.pop(key)
            a, b = key
            if n >= c.occlusion_frames:  # (observable: a long overlap of two same-team boxes ended)
                st.overlap_end[a] = st.overlap_end[b] = st.frame_i
            if n >= c.occlusion_frames and a not in st.swaps and b not in st.swaps and \
                    st.rng.random() < c.occlusion_swap_prob:
                dur = max(1, int(round(float(st.rng.uniform(*c.occlusion_swap_s)) * c.fps)))
                st.swaps[a], st.swaps[b] = (b, dur), (a, dur)

    def _foot_sigma_px(self) -> float | None:
        """Pixel sd of the foot-point error that gives ``pos_sigma_ft`` at the mid-court reference depth
        (None without a floor homography: the isotropic court-error fallback)."""
        if self._floor is None or self._P is None:
            return None
        if self._px_sigma is None:
            J = self._floor_jacobian(47.0, 25.0)
            k = math.sqrt(abs(float(np.linalg.det(J)))) if J is not None else 0.0
            self._px_sigma = self.config.pos_sigma_ft / k if k > 1e-9 else 0.0
        return self._px_sigma

    def _floor_jacobian(self, x: float, y: float, h: float = 0.0) -> np.ndarray | None:
        """d(court floor point)/d(image point) at the image of (x, y, h) (2x2, ft per px; analytic for the
        floor homography court = (H[:2] m) / (H[2] m), m = (u, v, 1))."""
        assert self._P is not None and self._floor is not None
        q = self._P @ np.array([x, y, h, 1.0])
        if not q[2] > 0:
            return None
        H = self._floor.H
        m = np.array([q[0] / q[2], q[1] / q[2], 1.0])
        num, w = H[:2] @ m, float(H[2] @ m)
        if abs(w) < 1e-12:
            return None
        return (H[:2, :2] * w - np.outer(num, H[2, :2])) / (w * w)

    def _players(self, f: FrameState, st: _State, lifts: Sequence[float] | None) -> list[PlayerState]:
        c, rng = self.config, st.rng
        players = f.players
        by_id = {p.player_id: p for p in players}
        covered = self._covered(players)
        s_px = self._foot_sigma_px()
        out = []
        st.shown = {}
        for k, p in enumerate(players):
            pos_e = _ar1(st.pos_err, p.player_id, 2, c.pos_rho, c.pos_sigma_ft if s_px is None else s_px, rng)
            box_e = _ar1(st.box_err, p.player_id, 4, c.bbox_rho, c.bbox_sigma_px, rng)
            drop = rng.random() < c.drop_player_rate
            occl = covered[k] >= c.occluded_cover and rng.random() < c.occluded_drop_rate
            trk = st.tracks.setdefault(p.player_id, [p.track_id if p.track_id is not None else k + 1, 0, 0])
            if drop or occl:
                trk[2] += 1  # missing: the tracker may lose the track
                continue
            self._fragment(st, trk)
            ident = p
            shown = st.swaps.get(p.player_id)
            if shown is not None and shown[0] in by_id:
                ident = by_id[shown[0]]
            elif st.legacy is not None and p.player_id in st.legacy[:2]:
                ident = by_id.get(st.legacy[1] if p.player_id == st.legacy[0] else st.legacy[0]) or p
            flipped = self._team_flip(st, p.player_id)
            pid, jersey, team = ident.player_id, ident.jersey, ident.team
            # the track following this body: a fragment's own id; the (possibly swapped) label's id otherwise
            track_id = trk[0] if trk[0] != p.track_id else ident.track_id
            provisional = trk[1] > 0
            if provisional:
                trk[1] -= 1
            if (flipped or provisional) and team is not None:  # no roster identity: a provisional id
                team = team.other() if flipped else team
                pid, jersey = f"{team.value}-t{track_id}", None
            doubt = flipped or provisional or self._overlap_doubt(st, p.player_id)
            conf = float(rng.uniform(*c.player_conf)) * (1.0 - c.conf_covered_drop * min(1.0, covered[k]))
            conf *= c.conf_doubt if doubt else 1.0
            h = float(lifts[k]) if c.foot_lift and lifts is not None and k < len(lifts) else 0.0
            lx, ly = self._lift_shift(p.x, p.y, h) if h > 0.0 else (0.0, 0.0)
            if s_px is None:  # no floor homography: isotropic court error
                ex, ey = float(pos_e[0]), float(pos_e[1])
            else:  # the floor projection of the foot point's pixel error (depth-dependent)
                J = self._floor_jacobian(p.x, p.y, h) if s_px > 0.0 else None
                ex, ey = (float(v) for v in (J @ pos_e)) if J is not None else (0.0, 0.0)
                box_e = box_e + np.array([pos_e[0], 0.0, pos_e[0], pos_e[1]])  # the same foot point moves the box
            x, y = p.x + lx + ex, p.y + ly + ey
            vx, vy = self._velocity(st, p.player_id, x, y, f.t)
            out.append(dataclasses.replace(
                p, player_id=pid, jersey=jersey, track_id=track_id, team=team,
                x=x, y=y, vx=vx, vy=vy, bbox=_shift_box(p.bbox, box_e), confidence=conf))
            st.shown[p.player_id] = out[-1]
        return out

    def _fragment(self, st: _State, trk: list[int]) -> None:
        """A visible player: maybe a new track id (``frag_*``: after ``trk[2]`` missing frames, or a re-init
        without a gap), sometimes with a provisional identity until the re-link."""
        c = self.config
        gap, trk[2] = trk[2], 0
        p_new = c.frag_hazard if gap <= 0 else min(1.0, c.frag_after_drop_1 + c.frag_per_frame * (gap - 1))
        if p_new > 0.0 and st.rng.random() < p_new:
            st.next_track += 1
            trk[0] = st.next_track
            if st.rng.random() < c.frag_provisional:
                trk[1] = max(1, int(round(float(st.rng.uniform(*c.frag_relink_s)) * c.fps)))

    # ------------------------------------------------------------------ phantom people

    def _person_box(self, x: float, y: float, height: float) -> BBox | None:
        if self.camera is None:
            return None
        box = self.camera.bbox_of_cylinder(x, y, height, self.config.person_radius_ft)
        return box if _finite(box) else None

    def _phantoms(self, f: FrameState, st: _State, seen: list[PlayerState], rng: np.random.Generator,
                  referees: Sequence[tuple[float, float]] | None,
                  waiting: Sequence[tuple[str, str, float, float]] | None) -> list[PlayerState]:
        """Detections that are not one of the ten on-court players (``dup_*``, ``ref_*``, ``waiting_*``)."""
        c = self.config
        out: list[PlayerState] = []
        # duplicates / splits of a visible player
        for q in f.players:
            real = st.shown.get(q.player_id)
            ep = st.dups.get(q.player_id)
            if ep is None:
                if real is None or c.dup_hazard <= 0.0 or rng.random() >= c.dup_hazard:
                    continue
                ang, dist = float(rng.uniform(0.0, 2.0 * math.pi)), float(rng.uniform(*c.dup_offset_ft))
                u = rng.random()
                mates = [p for p in f.players if p.team is q.team and p.player_id != q.player_id]
                if u < c.dup_labels[0] or q.team is None:
                    label = None  # provisional
                elif u < c.dup_labels[0] + c.dup_labels[1] and mates:
                    label = mates[int(rng.integers(len(mates)))].player_id
                else:
                    label = real.player_id
                st.next_track += 1
                n = 1 + int(rng.geometric(1.0 / max(1.0, c.dup_frames))) - 1
                ep = st.dups[q.player_id] = [n, dist * math.cos(ang), dist * math.sin(ang), label, st.next_track,
                                             float(rng.uniform(*c.dup_conf))]
            ep[0] -= 1
            if ep[0] <= 0:
                del st.dups[q.player_id]
            if real is None:
                continue
            x, y = real.x + ep[1], real.y + ep[2]
            box = real.bbox
            if box is not None and self._P is not None:
                a, b = self._P @ np.array([real.x, real.y, 0.0, 1.0]), self._P @ np.array([x, y, 0.0, 1.0])
                if a[2] > 0 and b[2] > 0:
                    du, dv = b[0] / b[2] - a[0] / a[2], b[1] / b[2] - a[1] / a[2]
                    box = _shift_box(box, np.array([du, dv, du, dv]))
            label = ep[3]
            jersey = next((p.jersey for p in f.players if p.player_id == label), None) if label else None
            pid = label if label else f"{q.team.value if q.team else 'unk'}-t{ep[4]}"
            out.append(dataclasses.replace(real, player_id=pid, jersey=jersey, track_id=ep[4], x=x, y=y, bbox=box,
                                           confidence=float(np.clip(ep[5] + rng.normal(0.0, 0.03), 0.05, 0.99))))
        # referees mistaken for players
        for k, (x, y) in enumerate(referees or []):
            ep = st.refs.get(k)
            if ep is None:
                if c.ref_hazard <= 0.0 or rng.random() >= c.ref_hazard:
                    continue
                st.next_track += 1
                team = (Team.HOME, Team.AWAY)[int(rng.integers(2))]
                ep = st.refs[k] = [int(rng.integers(c.ref_frames[0], c.ref_frames[1] + 1)), team, st.next_track,
                                   float(rng.uniform(0.3, 0.7))]
            ep[0] -= 1
            if ep[0] <= 0:
                del st.refs[k]
            box = self._person_box(x, y, c.ref_height_ft)
            team = ep[1]
            out.append(PlayerState(player_id=f"{team.value}-t{ep[2]}", team=team, x=float(x), y=float(y),
                                   track_id=ep[2], bbox=_jitter_box(box, c.bbox_sigma_px, rng), confidence=ep[3]))
        # substitutes waiting at the table
        live = set()
        for pid, team_v, x, y in waiting or []:
            live.add(pid)
            if rng.random() >= c.waiting_detect:
                continue
            if pid not in st.waiting_tracks:
                st.next_track += 1
                st.waiting_tracks[pid] = (st.next_track, bool(rng.random() < c.waiting_id))
            tid, known = st.waiting_tracks[pid]
            team = Team(team_v)
            box = self._person_box(x, y, c.person_height_ft)
            out.append(PlayerState(player_id=pid if known else f"{team.value}-t{tid}", team=team, x=float(x),
                                   y=float(y), track_id=tid, bbox=_jitter_box(box, c.bbox_sigma_px, rng),
                                   confidence=float(rng.uniform(0.6, 0.95))))
        for pid in [p for p in st.waiting_tracks if p not in live]:
            del st.waiting_tracks[pid]
        return out

    def _overlap_doubt(self, st: _State, pid: str) -> bool:
        """Identity in doubt from what is observable: a long same-team overlap under way or recently over."""
        c = self.config
        if any(n >= c.occlusion_frames and pid in key for key, n in st.overlap.items()):
            return True
        end = st.overlap_end.get(pid)
        return end is not None and st.frame_i - end <= int(round(c.occlusion_swap_s[1] * c.fps))

    def _team_flip(self, st: _State, pid: str) -> bool:
        """Is this player's team label wrong on this frame (a 2-state episode process, ``team_flip_*``)?"""
        c = self.config
        left = st.team_flips.get(pid, 0)
        if left > 0:
            st.team_flips[pid] = left - 1
            return True
        if c.team_flip_share <= 0.0:
            return False
        lo, hi = c.team_flip_s
        mean = max(1.0, 0.5 * (lo + hi) * c.fps)
        share = min(c.team_flip_share, 0.5)
        if st.rng.random() < share / (mean * (1.0 - share)):
            st.team_flips[pid] = max(1, int(round(float(st.rng.uniform(lo, hi)) * c.fps))) - 1
            return True
        return False

    def _velocity(self, st: _State, pid: str, x: float, y: float, t: float) -> tuple[float, float]:
        """Alpha-beta velocity from the noisy positions (the video pipeline's court-space smoothing)."""
        c = self.config
        z = np.array([x, y])
        prev = st.vel.get(pid)
        if prev is None:
            st.vel[pid] = (z, np.zeros(2), t)
            return 0.0, 0.0
        xf, vf, t0 = prev
        dt = t - t0 if t > t0 else 1.0 / c.fps
        pred = xf + vf * dt
        r = z - pred
        if float(np.hypot(*r)) > c.vel_reset_ft:
            xf, vf = z, np.zeros(2)
        else:
            xf = pred + c.vel_alpha * r
            vf = vf + (c.vel_beta / dt) * r
        sp = float(np.hypot(*vf))
        if sp > c.vel_max_fts:
            vf = vf * (c.vel_max_fts / sp)
        st.vel[pid] = (xf, vf, t)
        return float(vf[0]), float(vf[1])

    def _covered(self, players: list[PlayerState]) -> list[float]:
        """Share of each player's box covered by nearer players' boxes (union approximated by the max)."""
        out = [0.0] * len(players)
        depth = [self._depth((p.x, p.y, 3.0)) for p in players]
        for i, p in enumerate(players):
            if p.bbox is None or not _finite(p.bbox) or p.bbox.w <= 0 or p.bbox.h <= 0:
                continue
            for j, q in enumerate(players):
                if i == j or q.bbox is None or not _finite(q.bbox):
                    continue
                if depth[i] is not None and depth[j] is not None:
                    nearer = depth[j] < depth[i]
                else:  # no camera: the lower foot point is nearer
                    nearer = q.bbox.y2 > p.bbox.y2
                if nearer:
                    out[i] = max(out[i], _inter(p.bbox, q.bbox) / (p.bbox.w * p.bbox.h))
        return out

    def _rim(self, rim: RimObservation, rng: np.random.Generator) -> RimObservation:
        box = _jitter_box(rim.bbox, self.config.rim_px_sigma, rng)
        assert box is not None
        return RimObservation(side=rim.side, bbox=box, confidence=rim.confidence)

    # ------------------------------------------------------------------ ball

    def _held(self, f: FrameState) -> bool:
        b, c = f.ball, self.config
        if b is None or b.court_xy is None or b.z is None or b.z > c.held_max_z_ft:
            return False
        bx, by = b.court_xy
        return any(math.hypot(p.x - bx, p.y - by) <= c.held_radius_ft for p in f.players)

    def _behind_body(self, f: FrameState) -> bool:
        """The ball's image point lies inside the upper body of a player nearer to the camera than the ball."""
        b = f.ball
        if b is None or b.image_xy is None or b.court_xy is None or b.z is None:
            return False
        db = self._depth((b.court_xy[0], b.court_xy[1], b.z))
        u, v = b.image_xy
        for p in f.players:
            box = p.bbox
            if box is None:
                continue
            upper = box.y1 + self.config.ball_hide_frac * box.h  # legs leave gaps: only the upper body hides it
            if not (box.x1 <= u <= box.x2 and box.y1 <= v <= upper):
                continue
            dp = self._depth((p.x, p.y, min(max(b.z, 0.5), 6.0)))
            if db is not None and dp is not None and dp < db - self.config.ball_behind_ft:
                return True
        return False

    def _in_net(self, f: FrameState, prev: FrameState | None) -> bool:
        b = f.ball
        if prev is None or prev.ball is None or b is None or b.z is None or prev.ball.z is None or b.image_xy is None:
            return False
        if not (prev.ball.z > _RIM_HEIGHT_FT >= b.z):
            return False
        u = b.image_xy[0]
        return any(r.bbox.x1 <= u <= r.bbox.x2 for r in f.rims)

    def _ball(self, f: FrameState, st: _State) -> BallState | None:
        b, c, rng = f.ball, self.config, st.rng
        if b is None:
            return None
        fp = self._fp_candidate(f, st)
        # Detector misses (Gilbert-Elliott burst process): "bad" runs last ball_burst_frames on average and
        # the stationary miss rate is ball_miss_rate. A held ball adds the hands (the extra chance that
        # brings its mean to ball_miss_rate_held); a nearer body and the net add their own on top.
        leave = 1.0 / max(1.0, c.ball_burst_frames)
        enter = _enter_rate(c.ball_miss_rate, leave, c.ball_good_miss, c.ball_bad_miss)
        st.bad = (rng.random() >= leave) if st.bad else (rng.random() < enter)
        p_miss = c.ball_bad_miss if st.bad else c.ball_good_miss
        if self._held(f):
            p_miss = _combine(p_miss, _extra(c.ball_miss_rate_held, c.ball_miss_rate))
        if self._behind_body(f):
            p_miss = _combine(p_miss, c.ball_occluded_miss)
        if self._in_net(f, st.prev_frame):
            st.net_left = c.ball_net_frames
        if st.net_left > 0:
            st.net_left -= 1
            p_miss = _combine(p_miss, c.ball_net_miss)
        st.prev_frame = f
        true_obs: BallState | None = None
        if b.visible and b.image_xy is not None and rng.random() >= p_miss:
            du, dv = rng.normal(0.0, c.ball_px_sigma, 2)
            uv = (b.image_xy[0] + float(du), b.image_xy[1] + float(dv))
            radius = None if b.radius_px is None else max(
                1.0, b.radius_px + float(rng.normal(0.0, c.ball_radius_sigma_px)))
            conf = float(rng.uniform(*c.ball_conf))
            true_obs = self._ball_state(b, uv, radius, conf, rng)
        if fp is not None and (true_obs is None or fp.confidence > true_obs.confidence):
            return fp  # a single-ball output keeps the more confident candidate
        return true_obs if true_obs is not None else BallState(visible=False)

    def _ball_state(self, b: BallState, uv: tuple[float, float], radius: float | None, conf: float,
                    rng: np.random.Generator) -> BallState:
        c = self.config
        if c.monocular:
            return BallState(visible=True, image_xy=uv, court_xy=self._floor_xy(uv), z=None, radius_px=radius,
                             confidence=conf)
        court_xy = None
        if b.court_xy is not None:
            ex, ey = rng.normal(0.0, c.pos_sigma_ft, 2)
            court_xy = (b.court_xy[0] + float(ex), b.court_xy[1] + float(ey))
        z = None if b.z is None else max(0.0, b.z + float(rng.normal(0.0, 0.2)))
        return BallState(visible=True, image_xy=uv, court_xy=court_xy, z=z, radius_px=radius, confidence=conf)

    def _fp_candidate(self, f: FrameState, st: _State) -> BallState | None:
        """The false-positive source of the current episode (if any) on this frame."""
        c, rng = self.config, st.rng
        if st.fp is None and c.ball_fp_rate > 0.0:
            lo, hi = c.ball_fp_episode_s
            mean_frames = max(1.0, 0.5 * (lo + hi) * c.fps)
            rate = min(c.ball_fp_rate, 0.999)
            p_start = 1.0 if c.ball_fp_rate >= 1.0 else rate / (mean_frames * (1.0 - rate))
            if rng.random() < p_start:
                st.fp = self._new_episode(f, rng)
        ep = st.fp
        if ep is None:
            return None
        ep.frames_left -= 1
        if ep.frames_left <= 0:
            st.fp = None
        if rng.random() >= c.ball_fp_present:
            return None
        if ep.kind == "head":
            p = next((q for q in f.players if q.player_id == ep.pid and q.bbox is not None), None)
            if p is None:
                return None
            box = p.bbox
            uv = ((box.x1 + box.x2) / 2.0 + float(rng.normal(0.0, 1.0)),
                  box.y1 + 0.06 * box.h + float(rng.normal(0.0, 1.0)))
            radius = max(2.0, 0.06 * box.h)
        else:
            uv = (ep.uv[0] + float(rng.normal(0.0, 0.5)), ep.uv[1] + float(rng.normal(0.0, 0.5)))
            radius = float(rng.uniform(2.5, 5.0))
        conf = float(np.clip(ep.conf + rng.normal(0.0, 0.03), 0.05, 0.99))
        court_xy = self._floor_xy(uv) if self.config.monocular else None
        return BallState(visible=True, image_xy=uv, court_xy=court_xy, z=None, radius_px=radius, confidence=conf)

    def _new_episode(self, f: FrameState, rng: np.random.Generator) -> _FPEpisode:
        c = self.config
        frames = max(1, int(round(float(rng.uniform(*c.ball_fp_episode_s)) * c.fps)))
        conf = float(rng.uniform(*c.ball_fp_conf))
        heads = [p for p in f.players if p.bbox is not None]
        if heads and rng.random() < c.ball_fp_head_share:
            return _FPEpisode(frames, "head", conf, pid=heads[int(rng.integers(len(heads)))].player_id)
        w, h = f.image_size or (self.camera.image_size if self.camera else (1280, 720))
        return _FPEpisode(frames, "static", conf, uv=(float(rng.uniform(0, w)), float(rng.uniform(0.3 * h, h))))

    def _floor_xy(self, uv: tuple[float, float]) -> tuple[float, float] | None:
        if self._floor is None:
            return None
        xy = self._floor.image_to_court(np.array(uv, dtype=float), front_only=True)
        if not np.all(np.isfinite(xy)):
            return None
        return float(xy[0]), float(xy[1])


def _enter_rate(mean_miss: float, leave: float, good: float, bad: float) -> float:
    """P(good -> bad) per frame so the burst process's stationary miss rate is ``mean_miss``."""
    span = bad - good
    share = float(np.clip((mean_miss - good) / span, 0.0, 0.999)) if span > 0 else 0.0
    return leave * share / (1.0 - share)


def _combine(p: float, q: float) -> float:
    """Miss probability of two independent miss causes."""
    return 1.0 - (1.0 - p) * (1.0 - q)


def _extra(target: float, base: float) -> float:
    """Independent extra miss chance that lifts a ``base`` mean miss rate to ``target``."""
    if target <= base:
        return 0.0
    return float(np.clip(1.0 - (1.0 - target) / (1.0 - base), 0.0, 1.0))


def _ar1(store: dict[str, np.ndarray], key: str, n: int, rho: float, sigma: float,
         rng: np.random.Generator) -> np.ndarray:
    """AR(1) noise with marginal standard deviation ``sigma`` (stationary from the first draw)."""
    prev = store.get(key)
    if prev is None:
        e = rng.normal(0.0, sigma, n)
    else:
        e = rho * prev + math.sqrt(max(0.0, 1.0 - rho * rho)) * rng.normal(0.0, sigma, n)
    store[key] = e
    return e


def _shift_box(box: BBox | None, d: np.ndarray) -> BBox | None:
    if box is None:
        return None
    x1, y1, x2, y2 = box.x1 + d[0], box.y1 + d[1], box.x2 + d[2], box.y2 + d[3]
    return BBox(float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2)))


def _jitter_box(box: BBox | None, sigma: float, rng: np.random.Generator) -> BBox | None:
    if box is None:
        return None
    return _shift_box(box, rng.normal(0.0, sigma, 4))


def _finite(box: BBox) -> bool:
    return all(math.isfinite(v) for v in (box.x1, box.y1, box.x2, box.y2))


def _inter(a: BBox, b: BBox) -> float:
    w = min(a.x2, b.x2) - max(a.x1, b.x1)
    h = min(a.y2, b.y2) - max(a.y1, b.y1)
    return w * h if w > 0 and h > 0 else 0.0


def _iou(a: BBox, b: BBox) -> float:
    if not (_finite(a) and _finite(b)):
        return 0.0
    i = _inter(a, b)
    u = a.w * a.h + b.w * b.h - i
    return i / u if u > 0 else 0.0


# ---------------------------------------------------------------------------- scoreboard


@dataclass
class ScoreboardConfig:
    """The venue scoreboard as an operator runs it and a 5 Hz OCR reads it (shared by mono and video).

    Display (rules note §3): the game clock shows the ceiling of the remaining seconds, tenths only in
    the last minute (``tenths_below_s``); the shot clock shows whole seconds (ceiling). The synthetic
    video panel has MM:SS cells only (render.py), so video shows ceiling seconds throughout.
    Sampling: OCR at ``ocr_hz`` (D14: 5 Hz); readings carry ``source`` "ocr", so the engine's 2-read
    confirmation applies. Operator lag (RESEARCH §7A.5: no source gives amateur operator lag; the make
    wait is 3.0 s until it is measured): the score changes, the clock stops / starts and the shot clock
    resets late, each lag drawn uniformly from its range; when the displayed clock has drifted more than
    ``clock_fix_s`` from the official time at a stop, the officials put it right. Misreads: one OCR read
    in ``misread_rate`` garbles a digit. All operator / OCR values are design choices, unverified.
    """

    ocr_hz: float = 5.0  # D14
    source: str = "ocr"
    tenths_below_s: float = 60.0  # rules note §3: tenths in the last minute (NBA, NCAA)
    score_lag_s: tuple[float, float] = (0.4, 2.0)  # unverified; bounded by the 3.0 s make wait (§7A.5)
    clock_lag_s: tuple[float, float] = (0.15, 0.6)  # stop / start reaction to whistle / touch; unverified
    shot_reset_lag_s: tuple[float, float] = (0.2, 0.9)  # unverified
    clock_fix_s: float = 1.0  # unverified
    misread_rate: float = 0.01  # unverified
    confidence: float = 0.9


class _Clock:
    """A displayed countdown the operator starts and stops late (``lag`` per transition).

    Expiry is automatic, not an operator action: the venue game clock and the shot clock are the
    official timers, and their signal (horn) sounds by itself when the display reaches zero; a shot
    counts only if released before that signal (rules note §3 "legal shot vs. shot clock" and the
    period-end buzzer of §2a; sources [3] NBA Rule 5, [4] NBA Rule 7, [9] Comments on the Rules §L
    "Expiration of time", [11] FIBA OBR 2026; exact clauses to verify). Nobody presses stop at the horn,
    so the operator's accumulated start / stop lag and drift never apply to the timer's own expiry: on
    the frame the official (truth) clock runs out, the display shows 0.0 (whatever residual lag it
    carried is absorbed there, because a board can never show time left after its own horn). Lags
    still apply to starts, stops and resets after whistles and touches. Status unverified.
    """

    def __init__(self) -> None:
        self.value: float | None = None
        self.running = False
        self.pending: list[tuple[float, bool]] = []  # (time, running) operator actions due
        self.truth_running = False

    def step(self, t: float, dt: float, truth: float | None, prev_truth: float | None, lag: float,
             fix_s: float) -> float | None:
        if truth is None:
            self.value, self.running, self.pending, self.truth_running = None, False, [], False
            return None
        if self.value is None or prev_truth is None or truth > prev_truth + 1.0:
            # first reading, a clock (re)set for a new period / possession: the operator sets it at once
            self.value, self.running, self.pending = truth, False, []
            self.truth_running = False
            return self.value
        if truth <= _EXPIRED_S and prev_truth > _EXPIRED_S:
            # the timer ran out: its own signal, no operator lag (see the class docstring)
            self.value, self.running, self.pending, self.truth_running = 0.0, False, [], False
            return self.value
        running = truth < prev_truth - 1e-9
        if running != self.truth_running:
            self.truth_running = running
            self.pending.append((max([t + lag] + [p[0] for p in self.pending]), running))
        if self.running:
            self.value = max(0.0, self.value - dt)
        while self.pending and self.pending[0][0] <= t + 1e-9:
            self.running = self.pending.pop(0)[1]
            if not self.running and abs(self.value - truth) > fix_s:
                self.value = truth  # the officials correct a clock that has drifted
        if not self.running and not self.truth_running and not self.pending and self.value < truth:
            self.value = truth  # never shows less time than the officials stopped it at
        return self.value


class _ShotClock(_Clock):
    """The shot clock: runs / stops like the game clock; a reset (the official clock jumping up) is
    pressed ``reset_lag`` late, the display then shows the official value."""

    def __init__(self) -> None:
        super().__init__()
        self.reset_due: float | None = None

    def step(self, t: float, dt: float, truth: float | None, prev_truth: float | None,  # type: ignore[override]
             lag: float, reset_lag: float = 0.0, fix_s: float = 1.0) -> float | None:
        if truth is None or self.value is None or prev_truth is None:
            self.reset_due = None
            return super().step(t, dt, truth, prev_truth, lag, fix_s)
        if truth > prev_truth + 0.5 and self.reset_due is None:
            self.reset_due = t + reset_lag
        if self.reset_due is not None:
            if t + 1e-9 >= self.reset_due:
                self.reset_due, self.pending = None, []
                self.value, self.running = truth, truth < prev_truth - 1e-9
                self.truth_running = self.running
                return self.value
            if self.running:  # still showing the old count
                self.value = max(0.0, self.value - dt)
            return self.value
        return super().step(t, dt, truth, prev_truth, lag, fix_s)


class ScoreboardModel:
    """Truth scoreboard readings (one per frame) -> the operator-run board -> 5 Hz OCR readings."""

    def __init__(self, config: ScoreboardConfig | None = None) -> None:
        self.config = config or ScoreboardConfig()

    def displayed(self, boards: Sequence[ScoreboardReading | None], fps: float, seed: int = 0,
                  tenths: bool = True) -> list[ScoreboardReading | None]:
        """Per frame, the board as displayed: operator lags applied; clocks quantised for display
        (``tenths=False``: ceiling seconds throughout, the synthetic panel's MM:SS cells)."""
        c, rng = self.config, np.random.default_rng([seed, 41])
        dt = 1.0 / fps
        out: list[ScoreboardReading | None] = []
        shown: tuple[int, int] | None = None
        pending: list[tuple[float, tuple[int, int]]] = []
        game, shot = _Clock(), _ShotClock()
        prev: ScoreboardReading | None = None
        for i, b in enumerate(boards):
            t = i * dt
            if b is None:
                out.append(None)
                prev = None
                continue
            score = (int(b.home_score), int(b.away_score))
            if shown is None:
                shown = score
            if prev is not None and score != (prev.home_score, prev.away_score):
                due = max([t + float(rng.uniform(*c.score_lag_s))] + [p[0] for p in pending])
                pending.append((due, score))
            while pending and pending[0][0] <= t + 1e-9:
                shown = pending.pop(0)[1]
            gc = game.step(t, dt, b.game_clock, None if prev is None else prev.game_clock,
                           float(rng.uniform(*c.clock_lag_s)), c.clock_fix_s)
            sc = shot.step(t, dt, b.shot_clock, None if prev is None else prev.shot_clock,
                           float(rng.uniform(*c.clock_lag_s)), float(rng.uniform(*c.shot_reset_lag_s)), c.clock_fix_s)
            prev = b
            out.append(ScoreboardReading(
                t=b.t, home_score=shown[0], away_score=shown[1], period=b.period,
                game_clock=_display_clock(gc, c.tenths_below_s if tenths else 0.0),
                shot_clock=None if sc is None else float(_ceil_s(sc)), source=c.source, confidence=c.confidence))
        return out

    def readings(self, boards: Sequence[ScoreboardReading | None], fps: float, seed: int = 0
                 ) -> list[ScoreboardReading | None]:
        """Per frame: an OCR reading of the displayed board on the ``ocr_hz`` frames (with rare
        misreads), None on the others."""
        c = self.config
        shown = self.displayed(boards, fps, seed)
        rng = np.random.default_rng([seed, 43])
        every = max(1, int(round(fps / c.ocr_hz)))
        out: list[ScoreboardReading | None] = []
        for i, r in enumerate(shown):
            if r is None or i % every != 0:
                out.append(None)
                continue
            out.append(_misread(r, rng) if rng.random() < c.misread_rate else r)
        return out


_EXPIRED_S = 1e-9  # an official clock at or below this has run out


def _ceil_s(v: float) -> int:
    return max(0, int(math.ceil(v - 1e-6)))


def _display_clock(v: float | None, tenths_below: float) -> float | None:
    """The game clock as the board shows it: ceiling seconds, tenths (ceiling) under ``tenths_below``."""
    if v is None:
        return None
    if v < tenths_below - 1e-9:
        return max(0.0, math.ceil(v * 10.0 - 1e-6) / 10.0)
    return float(_ceil_s(v))


def _misread(r: ScoreboardReading, rng: np.random.Generator) -> ScoreboardReading:
    """One garbled digit (a segment read wrong): the home or away score's last digit, or the clock."""
    which = int(rng.integers(3))
    if which == 0 and r.home_score is not None:
        return dataclasses.replace(r, home_score=(r.home_score // 10) * 10 + int(rng.integers(10)))
    if which == 1 and r.away_score is not None:
        return dataclasses.replace(r, away_score=(r.away_score // 10) * 10 + int(rng.integers(10)))
    if r.game_clock is not None:
        return dataclasses.replace(r, game_clock=max(0.0, r.game_clock + float(rng.choice([-10.0, 10.0]))))
    return r


# ---------------------------------------------------------------------------- camera recovery


def projection_from_frames(frames: list[FrameState], max_points: int = 500,
                           max_rms_px: float = 0.5) -> np.ndarray | None:
    """3x4 projection recovered from 3-D ball truth in ``frames`` (None if unavailable), oriented so that
    points in front of the camera have ``w > 0``.

    Fits ``P`` by normalised DLT (Hartley & Zisserman §7.1) to visible balls with known ``court_xy`` and
    ``z``. Needs >= 6 points that are not coplanar (ball heights must vary) and a reprojection RMS <=
    ``max_rms_px``.
    """
    pts = [(*f.ball.court_xy, f.ball.z, *f.ball.image_xy) for f in frames
           if f.ball is not None and f.ball.visible and f.ball.image_xy is not None
           and f.ball.court_xy is not None and f.ball.z is not None]
    if len(pts) < 6:
        return None
    data = np.asarray(pts, dtype=float)
    data = data[np.all(np.isfinite(data), axis=1)]
    if len(data) > max_points:
        data = data[np.linspace(0, len(data) - 1, max_points).astype(int)]
    if len(data) < 6 or np.ptp(data[:, 2]) < 0.5:
        return None
    P = _dlt_projection(data[:, :3], data[:, 3:])
    if P is None:
        return None
    X = np.hstack([data[:, :3], np.ones((len(data), 1))])
    proj = X @ P.T
    if np.median(proj[:, 2]) < 0:  # the observed balls are in front of the camera
        P, proj = -P, -proj
    k = float(np.linalg.norm(P[2, :3]))
    if k > 1e-12:  # metric depth: the third row's direction part is a unit vector for a real camera
        P, proj = P / k, proj / k
    with np.errstate(divide="ignore", invalid="ignore"):
        rms = float(np.sqrt(np.mean(np.sum((proj[:, :2] / proj[:, 2:3] - data[:, 3:]) ** 2, axis=1))))
    if not rms <= max_rms_px:
        log.warning("camera recovery from frames failed: reprojection RMS %.3f px", rms)
        return None
    return P


def _floor_from_projection(P: np.ndarray | None) -> Homography | None:
    if P is None:
        return None
    G = P[:, [0, 1, 3]]  # court floor (x, y, 1) -> image
    try:
        H = np.linalg.inv(G)
        return Homography(H / np.linalg.norm(H))
    except (np.linalg.LinAlgError, ValueError):
        return None


def floor_homography_from_frames(frames: list[FrameState], max_points: int = 500,
                                 max_rms_px: float = 0.5) -> Homography | None:
    """Image -> floor homography recovered from 3-D ball truth in ``frames`` (None if unavailable):
    ``inv(P[:, [0, 1, 3]])`` of :func:`projection_from_frames`."""
    return _floor_from_projection(projection_from_frames(frames, max_points, max_rms_px))


def _dlt_projection(world: np.ndarray, image: np.ndarray) -> np.ndarray | None:
    def normaliser(pts: np.ndarray) -> np.ndarray:
        c = pts.mean(axis=0)
        d = float(np.mean(np.linalg.norm(pts - c, axis=1)))
        k = math.sqrt(pts.shape[1]) / d if d > 1e-12 else 1.0
        T = np.eye(pts.shape[1] + 1)
        T[:-1, :-1] *= k
        T[:-1, -1] = -k * c
        return T

    Tw, Ti = normaliser(world), normaliser(image)
    Xw = np.hstack([world, np.ones((len(world), 1))]) @ Tw.T
    xi = np.hstack([image, np.ones((len(image), 1))]) @ Ti.T
    n = len(world)
    A = np.zeros((2 * n, 12))
    A[0::2, 0:4] = Xw
    A[0::2, 8:12] = -xi[:, 0:1] * Xw
    A[1::2, 4:8] = Xw
    A[1::2, 8:12] = -xi[:, 1:2] * Xw
    _, sv, vt = np.linalg.svd(A)
    if sv[-2] <= 1e-12 * sv[0]:  # rank-deficient (e.g. coplanar points)
        return None
    Pn = vt[-1].reshape(3, 4)
    return np.linalg.inv(Ti) @ Pn @ Tw
