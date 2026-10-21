"""Ball gating, ball-holder detection with hysteresis, and per-player position history.

Holder detection runs in one of two ways, chosen per frame from the data:

* **3-D / court mode** (``ball.z`` known, or no player bboxes): a player can take the ball when
  ``ball.court_xy`` is within ``holder_radius_ft`` of the player's floor position and the ball is
  no higher than ``holder_max_z_ft``; the current holder keeps it out to ``holder_keep_radius_ft``.
* **Monocular mode** (``ball.z is None``, bboxes present): candidates are players whose bbox,
  grown by ``holder_bbox_expand`` (the wider, barely-above-the-head ``keep_box`` for the current
  holder), contains the ball centre, scored by horizontal closeness to the box centre line and by
  the ball being in the hands region. Once ``RayModel`` has located the camera, a candidate must
  also be at a plausible depth: the ball's viewing ray passes close to the player at a plausible
  ball height.

Scores fall with the ball's speed relative to the candidate (a held ball moves with its holder;
vertical motion counts half during a pass, since a catch is often followed by a dribble or a shot,
and fully for a loose ball). Two opponents scoring alike is contested (nobody yet).

Hysteresis: a new holder needs ``confirm_n`` consecutive observed frames, and a change of team
control (an opponent of the holder, or of the passer while a pass is in the air) needs
``confirm_opp_n``; the holder is released after ``release_n`` observed frames without the ball.
Frames with no accepted ball observation (occlusion by the body) neither confirm nor release, so
the holder is kept through short gaps (cv_pipeline §6.2).

Timestamps: a monocular change of team control is dated back over the new team's contest for the
ball, a loose-ball gain over the rebounder's candidacy (``_walk_onset``); a release after a hidden hold
is back-extrapolated from the first in-flight observations (``_release_in_gap``); in 3-D a ball raised
above ``holder_max_z_ft`` stays held until it is first in free flight, whose start is the release.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from courtvision.types import BBox, FrameState, PlayerState, Team

if TYPE_CHECKING:
    from .engine import EventEngineConfig

__all__ = ["BallGate", "BallObs", "ControlChange", "HolderTracker", "PlayerHistory", "Ray", "RayModel", "ball_obs",
           "keep_box"]

log = logging.getLogger(__name__)

G_FTS2 = 32.174  # gravity, ft/s^2


@dataclass(frozen=True, slots=True)
class BallObs:
    """One accepted ball observation (image and/or court coordinates)."""

    t: float
    frame_idx: int
    uv: tuple[float, float] | None
    xy: tuple[float, float] | None
    z: float | None
    r_px: float | None
    conf: float

    @property
    def is3d(self) -> bool:
        return self.z is not None and self.xy is not None


def ball_obs(frame: FrameState, cfg: EventEngineConfig) -> BallObs | None:
    """Usable ball observation of ``frame`` or None (invisible, low confidence, interpolated)."""
    b = frame.ball
    if b is None or not b.visible or b.confidence < cfg.ball_min_conf:
        return None
    if b.interpolated and not cfg.use_interpolated_ball:
        return None
    uv, xy, z = _finite_pair(b.image_xy), _finite_pair(b.court_xy), b.z
    if z is not None and not math.isfinite(z):
        return None  # a nan height would poison the gate, flights and holder distances
    if uv is None and xy is None:
        return None
    r_px = b.radius_px if b.radius_px is not None and math.isfinite(b.radius_px) else None
    return BallObs(frame.t, frame.frame_idx, uv, xy, z, r_px, b.confidence)


def _finite_pair(p: tuple[float, float] | None) -> tuple[float, float] | None:
    return p if p is not None and math.isfinite(p[0]) and math.isfinite(p[1]) else None


class BallGate:
    """Rejects physically impossible ball jumps (false detections) and estimates ball velocity.

    An observation farther from the last accepted one than ``base + v_max * dt`` is rejected
    unless ``ball_reacquire_frames`` consecutive observations agree with each other (the real ball
    was lost and found elsewhere). After ``ball_gate_reset_s`` without an accepted observation the
    gate opens.
    """

    def __init__(self, cfg: EventEngineConfig) -> None:
        self.cfg = cfg
        self._last: BallObs | None = None
        self._cand: list[BallObs] = []
        self._hist: deque[BallObs] = deque(maxlen=5)
        self.rejected = 0

    def reset(self) -> None:
        self._last, self._cand = None, []
        self._hist.clear()

    def accept(self, obs: BallObs | None, px_per_ft: float | None) -> BallObs | None:
        if obs is None:
            return None
        cfg = self.cfg
        last = self._last
        if last is None or obs.t - last.t > cfg.ball_gate_reset_s or self._dist(obs, last) <= self._gate(obs, last, px_per_ft):
            return self._take(obs)
        if self._cand and self._dist(obs, self._cand[-1]) <= self._gate(obs, self._cand[-1], px_per_ft):
            self._cand.append(obs)
        else:
            self._cand = [obs]
        if len(self._cand) >= cfg.ball_reacquire_frames:
            log.debug("ball re-acquired at frame %d after %d off-gate frames", obs.frame_idx, len(self._cand))
            self._hist.clear()
            return self._take(obs)
        self.rejected += 1
        return None

    def velocity(self) -> tuple[float, ...] | None:
        """(vx, vy, vz) ft/s in 3-D, (du, dv) px/s in image mode, (vx, vy) ft/s in court mode."""
        if len(self._hist) < 2:
            return None
        a, b = self._hist[0] if len(self._hist) < 3 else self._hist[-3], self._hist[-1]
        dt = b.t - a.t
        if dt <= 1e-6:
            return None
        pa, pb = _vec(a), _vec(b)
        if pa is None or pb is None or len(pa) != len(pb):
            return None
        return tuple((y - x) / dt for x, y in zip(pa, pb, strict=True))

    def vertical_accel(self) -> float | None:
        """3-D vertical acceleration (ft/s^2) over the last three accepted observations, or None."""
        if len(self._hist) < 3:
            return None
        a, b, c = self._hist[-3], self._hist[-2], self._hist[-1]
        if not (a.is3d and b.is3d and c.is3d):
            return None
        d1, d2 = b.t - a.t, c.t - b.t
        if d1 <= 1e-6 or d2 <= 1e-6 or d1 + d2 > 0.25:
            return None
        return 2.0 * ((c.z - b.z) / d2 - (b.z - a.z) / d1) / (d1 + d2)  # type: ignore[operator]

    def _take(self, obs: BallObs) -> BallObs:
        self._last, self._cand = obs, []
        self._hist.append(obs)
        return obs

    def _gate(self, obs: BallObs, ref: BallObs, px_per_ft: float | None) -> float:
        cfg = self.cfg
        dt = max(obs.t - ref.t, 0.0)
        if obs.uv is not None and ref.uv is not None and not (obs.is3d and ref.is3d):
            speed_px = cfg.ball_max_speed_fts * px_per_ft if px_per_ft else cfg.ball_max_speed_px_s
            return cfg.ball_gate_base_px + speed_px * dt
        return cfg.ball_gate_base_ft + cfg.ball_max_speed_fts * dt

    @staticmethod
    def _dist(a: BallObs, b: BallObs) -> float:
        if a.is3d and b.is3d:
            return math.dist((*a.xy, a.z), (*b.xy, b.z))  # type: ignore[misc]
        if a.uv is not None and b.uv is not None:
            return math.dist(a.uv, b.uv)
        if a.xy is not None and b.xy is not None:
            return math.dist(a.xy, b.xy)
        return 0.0


def _vec(o: BallObs) -> tuple[float, ...] | None:
    if o.is3d:
        return (o.xy[0], o.xy[1], o.z)  # type: ignore[index, return-value]
    if o.uv is not None:
        return o.uv
    return o.xy


@dataclass(frozen=True, slots=True)
class ControlChange:
    """Holder change: ``gain`` is timestamped at the onset of the confirmed run, ``release`` at the
    first frame after the ball was last observed with the holder."""

    kind: Literal["gain", "release"]
    player_id: str
    t: float
    frame_idx: int


class HolderTracker:
    """Confirmed ball holder with confirm / release hysteresis."""

    def __init__(self, cfg: EventEngineConfig) -> None:
        self.cfg = cfg
        self.holder: str | None = None
        self.holder_since: float | None = None
        self._seen_t: float | None = None
        self._after_seen: tuple[float, int] | None = None
        self._away = 0
        self._cand: str | None = None
        self._cand_n = 0
        self._cand_onset: tuple[float, int] = (0.0, 0)
        self._centres: dict[str, deque[tuple[float, tuple[float, float]]]] = {}
        self._teams: dict[str, Team | None] = {}
        self._holder_team: Team | None = None
        # image mode, per recent observed frame (leaky k-of-n confirmation, gain onsets): (t, frame, winner,
        # winner's team, teams with a valid candidate, valid candidates, loose ball, unambiguous winner)
        self._votes: deque[tuple] = deque(maxlen=64)
        self._loose = False  # image: this frame's ball is loose after a missed shot, nobody holds it
        self._scored_teams: frozenset[Team] = frozenset()  # teams with a valid candidate this frame
        self._scored_pids: frozenset[str] = frozenset()  # ... and those candidates
        self._clear_pick = False  # this frame's pick is unambiguous (``_is_clear``)
        self.clear_n = 0  # image: unambiguous observed frames of the current holder since his gain's onset
        self._lift_free = False  # 3-D: the ball raised above holder_max_z_ft has been in free flight since
        self._free_from: tuple[float, int] | None = None  # ... starting at this observation (the release)
        self._obs_tf: deque[tuple[float, int]] = deque(maxlen=3)  # (t, frame) of the last accepted observations
        self._seen_f: int | None = None  # frame of the holder's last observation with the ball
        self._seen_box: BBox | None = None  # ... and his image box then
        self._post: list[tuple[float, int, tuple[float, float]]] = []  # image: ball away from the holder since

    def reset(self) -> None:
        self.holder = self.holder_since = self._seen_t = None
        self._after_seen, self._away = None, 0
        self._cand, self._cand_n = None, 0
        self._votes.clear()
        self._lift_free, self._free_from = False, None

    def force_release(self, t: float, frame_idx: int) -> ControlChange | None:
        if self.holder is None:
            return None
        if self._after_seen is None:
            self._after_seen = (t, frame_idx)
        return self._release()

    def update(self, frame: FrameState, obs: BallObs | None, vel: tuple[float, ...] | None,
               px_per_ft: float | None, confirm_n: int, release_n: int,
               prefer_team: Team | None = None, confirm_opp_n: int | None = None,
               ray: Ray | None = None, accel_z: float | None = None, loose: bool = False) -> list[ControlChange]:
        """Advance one frame; returns releases before gains.

        ``loose``: a loose ball after a missed shot (image mode: a rebounder needs a longer vote).

        ``prefer_team``: the passer's team while a pass is in the air. A candidate of the other team
        than the holder (or than ``prefer_team``) is a change of team control and needs
        ``confirm_opp_n`` frames instead of ``confirm_n``.
        """
        changes: list[ControlChange] = []
        t, fi = frame.t, frame.frame_idx
        if obs is None:
            if self.holder is not None:
                if self._after_seen is None:
                    self._after_seen = (t, fi)
                if self._seen_t is not None and t - self._seen_t > self.cfg.holder_gap_keep_s:
                    changes.append(self._release())
            elif loose:
                won = self._hidden_gain(frame, t)
                if won is not None:
                    changes.append(won)
            self._remember_centres(frame)
            return changes
        image = _image_mode(frame.players, obs)
        self._loose = loose and self.holder is None
        self._obs_tf.append((t, fi))
        raw = self.candidate(frame.players, obs, vel, prefer_team, t, ray, accel_z)
        self._remember_centres(frame)
        ref_team = self._holder_team if self.holder is not None else prefer_team  # before any release below
        if image:
            self._votes.append((t, fi, raw, self._teams.get(raw) if raw is not None else None, self._scored_teams,
                                self._scored_pids, self._loose, self._clear_pick))
        else:
            self._votes.clear()
        if self.holder is not None:
            if raw == self.holder:
                self._seen_t, self._after_seen, self._away = t, None, 0
                self._seen_f, self._post = fi, []
                if image:
                    self._seen_box = next((p.bbox for p in frame.players if p.player_id == self.holder), None)
                if image and self._clear_pick:
                    self.clear_n += 1
            else:
                if image and obs.uv is not None:
                    self._post.append((t, fi, obs.uv))
                if self._after_seen is None:
                    self._after_seen = (t, fi)
                    ff = self._free_from
                    if ff is not None and self._seen_t is not None and ff[0] > self._seen_t - 0.2:
                        # 3-D raised ball: free flight is recognised over three observations; the
                        # release is the first of them
                        self._after_seen = min(self._after_seen, ff)
                self._away += 1
                if self._away >= release_n:
                    changes.append(self._release())
        if raw is not None and raw != self.holder and image:
            won = self._vote_gain(raw, ref_team if confirm_opp_n is not None else None, t, confirm_n,
                                  loose and self.holder is None)
            if won is not None:
                pid, self._cand_onset = won
                if self.holder is not None:
                    changes.append(self._release())
                changes.append(self._gain(pid, t))
                self.clear_n = sum(1 for v in self._votes if v[2] == pid and v[7] and v[0] >= self._cand_onset[0])
                self._votes.clear()
        elif raw is not None and raw != self.holder:
            if raw == self._cand:
                self._cand_n += 1
            else:
                self._cand, self._cand_n, self._cand_onset = raw, 1, (t, fi)
            team = self._teams.get(raw)
            need = confirm_n
            if confirm_opp_n is not None and ref_team is not None and team is not None and team is not ref_team:
                need = max(confirm_n, confirm_opp_n)
            if self._cand_n >= need:
                if self.holder is not None:
                    changes.append(self._release())
                changes.append(self._gain(raw, t))
        else:
            self._cand, self._cand_n = None, 0
        return changes

    def _hidden_gain(self, frame: FrameState, t: float) -> ControlChange | None:
        """Image, loose ball: since his first win, one player has been the only winner, for at least
        ``holder_hidden_confirm_s``, and the ball was hidden on at least ``holder_hidden_share`` of those frames
        while he stays in view: he secured it (a rebound held at the chest in traffic is covered by bodies, so
        the k-of-n vote never completes). Dated back over his candidacy."""
        cfg = self.cfg
        if cfg.holder_hidden_confirm_s <= 0.0 or not self._votes:
            return None
        votes = [v for v in self._votes if t - v[0] <= cfg.holder_hidden_window_s]
        winners = [v for v in votes if v[2] is not None]
        if not winners or len({v[2] for v in winners}) != 1:
            return None
        first = winners[0]
        pid = first[2]
        since = [v for v in votes if v[1] >= first[1]]
        frames = frame.frame_idx - first[1] + 1
        if t - first[0] < cfg.holder_hidden_confirm_s or not all(v[6] for v in since) \
                or 1.0 - len(since) / max(frames, 1) < cfg.holder_hidden_share:
            return None
        if not any(p.player_id == pid for p in frame.players):
            return None
        allv = list(self._votes)
        self._cand_onset = self._walk_onset(allv, first, lambda v: pid in v[5] and v[6], cfg.rebound_backdate_max_s)
        last = winners[-1]
        self._votes.clear()
        change = self._gain(pid, t)
        self._seen_t, self._seen_f = last[0], last[1]
        return change

    def _vote_gain(self, raw: str, ref_team: Team | None, t: float, confirm_n: int,
                   loose: bool = False) -> tuple[str, tuple[float, int]] | None:
        """Image mode: leaky k-of-n confirmation over the recent observed frames (one sub-threshold
        frame, or the ball briefly in a neighbour's box, does not reset it).

        Same team (or no reference team): ``raw`` won ``confirm_n`` of the last
        ``confirm_n + holder_vote_slack`` observed frames (a loose ball after a miss: ``holder_vote_loose_k``
        of ``holder_vote_loose_n``, as a bouncing ball near its apex moves slowly past bystanders). A change of team control (``raw`` is an
        opponent of ``ref_team``): the opposing team, counted over all its players, won at least
        ``steal_vote_frac`` of the observed frames since its first win in the last
        ``steal_vote_window_s`` and at least ``steal_min_votes`` of them, over at least
        ``steal_confirm_img_s``; the new holder is the team's most frequent winner. The gain is
        timestamped at the first winning frame of the window."""
        cfg = self.cfg
        team = self._teams.get(raw)
        votes = list(self._votes)
        if ref_team is None or team is None or team is ref_team:
            k, n = (cfg.holder_vote_loose_k, cfg.holder_vote_loose_n) if loose and ref_team is None \
                else (confirm_n, confirm_n + cfg.holder_vote_slack)
            hv = [v for v in votes[-n:] if v[2] == raw]
            if len(hv) < k:
                return None
            if loose and ref_team is None:
                # a loose ball: the rebounder's control began where he became a valid candidate (a held ball is
                # often hidden by his own body, so k observed wins can take a second)
                return raw, self._walk_onset(votes, hv[0], lambda v: raw in v[5] and v[6],
                                             cfg.rebound_backdate_max_s)
            return raw, (hv[0][0], hv[0][1])
        win = [v for v in votes if t - v[0] <= cfg.steal_vote_window_s]
        first = next((i for i, v in enumerate(win) if v[3] is team), None)
        if first is None:
            return None
        win = win[first:]
        hits = [v for v in win if v[3] is team and v[2] is not None]
        if len(hits) < cfg.steal_min_votes or len(hits) < cfg.steal_vote_frac * len(win) \
                or t - hits[0][0] < cfg.steal_confirm_img_s - 1e-6:
            return None
        counts: dict[str, int] = {}
        for v in hits:
            counts[v[2]] = counts.get(v[2], 0) + 1  # type: ignore[index]
        order = {v[2]: i for i, v in enumerate(hits)}  # ties: the most recent winner
        best = max(counts, key=lambda p: (counts[p], order[p]))
        return best, self._walk_onset(votes, hits[0], lambda v: team in v[4], cfg.steal_backdate_max_s)

    def _walk_onset(self, votes: list[tuple], first: tuple, valid: Callable[[tuple], bool],
                    cap: float) -> tuple[float, int]:
        """Image: onset of a confirmed gain. A change of team control starts where the new team's contest for
        the ball began, not at its first outright win: while a defender strips a dribbler their boxes overlap,
        the holder's keep margins win every frame and the ball is often hidden, so the first win comes ~1-2 s
        after the strip. Walk back over observed frames on which the new team had a valid candidate (hidden frames, and
        up to ``steal_backdate_slack`` observed frames in a row without one, do not break the run), at most
        ``steal_backdate_max_s`` before the first win. A loose-ball gain walks back over the rebounder's own
        valid candidacy instead."""
        k = next((i for i, v in enumerate(votes) if v is first), None)
        if cap <= 0.0 or k is None:
            return first[0], first[1]
        onset, miss = k, 0
        for j in range(k - 1, -1, -1):
            v = votes[j]
            if first[0] - v[0] > cap:
                break
            if valid(v):
                onset, miss = j, 0
            else:
                miss += 1  # a frame or two with the ball in a third player's box does not end the contest
                if miss > self.cfg.steal_backdate_slack:
                    break
        return votes[onset][0], votes[onset][1]

    # ------------------------------------------------------------------ candidates

    def candidate(self, players: Sequence[PlayerState], obs: BallObs, vel: tuple[float, ...] | None,
                  prefer_team: Team | None = None, t: float | None = None, ray: Ray | None = None,
                  accel_z: float | None = None) -> str | None:
        """Best holder candidate for this observation, or None (nobody, or contested by opponents)."""
        if not obs.is3d and obs.uv is not None and any(p.bbox is not None for p in players):
            scored = self._scores_image(players, obs, vel, prefer_team, t, ray)
            opp_margin = self.cfg.holder_switch_margin_opp
        elif obs.xy is not None:
            scored = self._scores_court(players, obs, vel, prefer_team, accel_z)
            opp_margin = self.cfg.holder_switch_margin_opp_court
        else:
            return None
        return self._pick(scored, opp_margin)

    def _pick(self, scored: list[tuple[float, str, Team | None]], opp_margin: float) -> str | None:
        self._scored_teams = frozenset(team for _, _, team in scored if team is not None)
        self._scored_pids = frozenset(pid for _, pid, _ in scored)
        pick = self._pick_inner(scored, opp_margin)
        self._clear_pick = pick is not None and self._is_clear(scored, pick)
        return pick

    def _is_clear(self, scored: list[tuple[float, str, Team | None]], pick: str) -> bool:
        """Is ``pick`` unambiguous on this frame: no other candidate within ``holder_clear_margin`` of its
        score (the holder's keep bonus left out)? A ball inside two overlapping boxes is not."""
        bonus = self.cfg.holder_switch_margin
        base = {pid: score - (bonus if pid == self.holder else 0.0) for score, pid, _ in scored}
        mine = base.get(pick)
        if mine is None:
            return False
        others = [b for pid, b in base.items() if pid != pick]
        return not others or mine - max(others) >= self.cfg.holder_clear_margin

    def _pick_inner(self, scored: list[tuple[float, str, Team | None]], opp_margin: float) -> str | None:
        if not scored:
            return None
        for _, pid, team in scored:
            self._teams[pid] = team
        scored.sort(key=lambda c: -c[0])
        best_score, best, best_team = scored[0]
        if best == self.holder:
            return best
        held = next((c for c in scored if c[1] == self.holder), None)
        if held is not None and best_team is not None and held[2] is not None and best_team is not held[2] \
                and best_score < held[0] + opp_margin:
            return self.holder  # an opponent must clearly beat the holder to take the ball
        for score, pid, team in scored[1:]:
            if team is not None and best_team is not None and team is not best_team:
                if best_score - score < self.cfg.holder_contest_margin:
                    return None  # two opponents equally on the ball (e.g. a jump ball): nobody yet
                break
        return best

    def _scores_court(self, players: Sequence[PlayerState], obs: BallObs, vel: tuple[float, ...] | None,
                      prefer_team: Team | None, accel_z: float | None = None) -> list[tuple[float, str, Team | None]]:
        cfg = self.cfg
        # 3-D: a ball in free flight above dribble height (vertical acceleration ~ -g) is not in
        # anyone's hands yet, e.g. a loose bounce drifting past a player at chest height
        free_min = cfg.holder_loose_free_min_z_ft if self._loose else cfg.holder_freefall_min_z_ft  # loose: no dribbler
        free = accel_z is not None and obs.z is not None and obs.z >= free_min \
            and abs(accel_z + G_FTS2) <= cfg.holder_freefall_tol_fts2
        # a loose ball that is not falling (vertical acceleration well above -g) is being lifted by hands
        lifted = self.holder is None and accel_z is not None and accel_z > cfg.holder_lift_min_accel_fts2
        if obs.z is not None and obs.z > cfg.holder_max_z_ft:
            # A ball raised above the head (layup / dunk gather, an overhead hold) stays in its holder's
            # hands until it is first seen in free flight: the release is the start of free flight, not
            # the moment it passed holder_max_z_ft (that dated layup releases ~0.5 s early, so the carry
            # over the ring looked like rim contact and the block after the release fell in the "shot"
            # phase). Once in flight up there, a later deflection (a block) does not hand it back.
            if not self._lift_free and free and self.holder is not None and self._obs_tf:
                self._free_from = self._obs_tf[0]
            self._lift_free = self._lift_free or free or self.holder is None
            if self._lift_free or obs.z > cfg.holder_reach_z_ft:
                return []
            players = [p for p in players if p.player_id == self.holder]
        else:
            self._lift_free, self._free_from = False, None
        bx, by = obs.xy  # type: ignore[misc]
        if not (obs.is3d or obs.uv is None):
            vel = None  # the gate's velocity is in px/s here, not ft/s
        out = []
        for p in players:
            d = math.hypot(bx - p.x, by - p.y)
            if p.player_id == self.holder:
                if d <= cfg.holder_keep_radius_ft:
                    score = 1.0 - d / cfg.holder_keep_radius_ft + cfg.holder_switch_margin
                    if vel is not None:  # the ball drifting away from its holder (horizontally) is losing it
                        score -= _rel_speed_court(vel, p, 0.0) / cfg.holder_rel_speed_scale_fts
                    out.append((score, p.player_id, p.team))
                continue
            if d > cfg.holder_radius_ft or free:
                continue
            score = 1.0 - d / cfg.holder_keep_radius_ft
            if vel is not None:
                rel = _rel_speed_court(vel, p, cfg.holder_rel_vertical_weight_held if lifted
                                       else self._vertical_weight(prefer_team))
                if rel > cfg.holder_acquire_max_speed_fts:
                    continue
                score -= rel / cfg.holder_rel_speed_scale_fts
            if prefer_team is not None and p.team is prefer_team:
                score += cfg.holder_team_prior
            out.append((score, p.player_id, p.team))
        return out

    def _scores_image(self, players: Sequence[PlayerState], obs: BallObs, vel: tuple[float, ...] | None,
                      prefer_team: Team | None, t: float | None, ray: Ray | None) -> list[tuple[float, str, Team | None]]:
        cfg = self.cfg
        u, v = obs.uv  # type: ignore[misc]
        out = []
        for p in players:
            if p.bbox is None:
                continue
            is_h = p.player_id == self.holder
            lateral = z = None
            if ray is not None and obs.xy is not None:
                lateral, z = ray.at_player(obs.xy, p.x, p.y)
            # the holder raising the ball overhead (a shot / pass gather) keeps it while the ray puts it within
            # his reach (3-D: holder_reach_z_ft): no above-head discount, no height penalty above the band
            raised = is_h and cfg.holder_img_raised_keep and z is not None and z <= cfg.holder_reach_z_ft
            score = _body_score(p.bbox, u, v, cfg, keep=is_h, raised=raised)
            if score is None:
                continue
            if lateral is not None and z is not None:
                slack = cfg.holder_ray_keep_slack_ft if is_h else 0.0
                z_hi = cfg.holder_reach_z_ft if raised else cfg.holder_max_z_ft + cfg.holder_ray_z_slack_ft + slack
                if abs(lateral) > cfg.holder_ray_max_lateral_ft + slack or not \
                        (-cfg.holder_ray_z_slack_ft - slack <= z <= z_hi):
                    continue  # the ball is at another depth than this player
                dz = cfg.holder_ray_z_ref_ft - z if raised else abs(z - cfg.holder_ray_z_ref_ft)
                score -= cfg.holder_ray_z_weight * max(0.0, dz - cfg.holder_ray_z_band_ft)
                score -= cfg.holder_ray_lateral_weight * max(0.0, abs(lateral) - cfg.holder_ray_lateral_free_ft)
            if is_h:
                out.append((score + cfg.holder_switch_margin, p.player_id, p.team))
                continue
            if vel is not None and len(vel) == 2:
                px_per_ft = max(p.bbox.h / cfg.player_height_ft, 1e-6)
                pv = self._centre_velocity(p, t)
                # vertical image motion counts partly: a dribble is vertical, but so is a ball falling past
                rel = math.hypot(vel[0] - pv[0], self._vertical_weight(prefer_team) * (vel[1] - pv[1])) / px_per_ft
                gate = cfg.holder_acquire_loose_img_fts if self._loose and prefer_team is None else \
                    cfg.holder_acquire_max_speed_fts + cfg.holder_rel_speed_slack_img_fts
                if rel > gate:
                    continue  # (a loose ball falling or bouncing past a player is not in his hands yet)
                score -= rel / cfg.holder_rel_speed_scale_img_fts
            if prefer_team is not None and p.team is prefer_team:
                score += cfg.holder_team_prior
            if score >= cfg.holder_min_score:
                out.append((score, p.player_id, p.team))
        return out

    def _vertical_weight(self, prefer_team: Team | None) -> float:
        """Weight of vertical ball motion in a challenger's relative speed. While someone holds
        (dribbles) the ball, its vertical motion says nothing about who controls it, so challengers
        are scored like the holder (both modes); otherwise see ``_vertical_weight``."""
        if self.holder is not None:
            return self.cfg.holder_rel_vertical_weight_held
        return _vertical_weight(self.cfg, prefer_team)

    def _remember_centres(self, frame: FrameState) -> None:
        if len(self._centres) > 4 * max(len(frame.players), 1) + 64:  # drop ids gone for > 10 s
            self._centres = {pid: h for pid, h in self._centres.items() if h and frame.t - h[-1][0] <= 10.0}
        for p in frame.players:
            if p.bbox is not None:
                h = self._centres.setdefault(p.player_id, deque(maxlen=3))
                h.append((frame.t, p.bbox.center))

    def _centre_velocity(self, p: PlayerState, t: float | None) -> tuple[float, float]:
        """Image velocity (px/s) of a player's bbox centre over the last couple of frames."""
        h = self._centres.get(p.player_id)
        if not h or t is None or p.bbox is None:
            return 0.0, 0.0
        t0, (u0, v0) = h[0]
        if t - t0 <= 1e-6:
            return 0.0, 0.0
        u1, v1 = p.bbox.center
        return (u1 - u0) / (t - t0), (v1 - v0) / (t - t0)

    # ------------------------------------------------------------------ transitions

    def touch(self, pid: str, t: float, frame_idx: int) -> list[ControlChange]:
        """A touch too brief to confirm (a tip): ``pid`` gains the ball and releases it at ``t``."""
        self._cand_onset = (t, frame_idx)
        gain = self._gain(pid, t)
        self._after_seen = (t, frame_idx)
        release = self._release()
        self._votes.clear()
        self._lift_free, self._free_from = True, None  # the tipped ball is in free flight
        return [gain, release]

    def _release(self) -> ControlChange:
        assert self.holder is not None
        t, fi = self._after_seen if self._after_seen is not None else (self._seen_t or 0.0, 0)
        est = self._release_in_gap()
        if est is not None and est[0] > t:
            t, fi = est
        change = ControlChange("release", self.holder, t, fi)
        self.holder = self.holder_since = self._seen_t = None
        self._after_seen, self._away = None, 0
        self._post = []
        return change

    def _release_in_gap(self) -> tuple[float, int] | None:
        """Image: the ball was last seen with the holder, then hidden (his own body covers a held ball, RESEARCH
        §2.1), and reappears in flight. The release is not the start of that gap but about when the flight left
        his box: back-extrapolate the first in-flight observations to his box at their image speed."""
        post, seen_t, seen_f, box = self._post, self._seen_t, self._seen_f, self._seen_box
        if len(post) < 2 or seen_t is None or seen_f is None or box is None:
            return None
        (t0, f0, a), (t1, _, b) = post[0], post[-1]
        if t0 - seen_t < self.cfg.release_gap_backdate_s or t1 - t0 < 0.05 or f0 <= seen_f:
            return None
        speed = math.dist(a, b) / (t1 - t0)
        if speed <= 1e-6:
            return None
        d = math.hypot(max(0.0, box.x1 - a[0], a[0] - box.x2), max(0.0, box.y1 - a[1], a[1] - box.y2))
        te = min(max(t0 - d / speed, seen_t), t0)
        rate = (f0 - seen_f) / (t0 - seen_t)
        return te, f0 - int(round((t0 - te) * rate))

    def _gain(self, pid: str, now: float) -> ControlChange:
        onset_t, onset_f = self._cand_onset
        self.holder, self.holder_since = pid, onset_t
        self._free_from = None
        self.clear_n = 0
        self._holder_team = self._teams.get(pid)
        self._seen_t, self._after_seen, self._away = now, None, 0
        self._cand, self._cand_n = None, 0
        return ControlChange("gain", pid, onset_t, onset_f)


def keep_box(bbox: BBox, cfg: EventEngineConfig) -> BBox:
    """Region where the current holder keeps the ball: wide (dribbles), barely above the head."""
    dx = bbox.w * cfg.holder_keep_expand / 2.0
    return BBox(bbox.x1 - dx, bbox.y1 - cfg.holder_keep_top_frac * bbox.h, bbox.x2 + dx,
                bbox.y2 + cfg.holder_keep_bottom_frac * bbox.h)


def _image_mode(players: Sequence[PlayerState], obs: BallObs) -> bool:
    return not obs.is3d and obs.uv is not None and any(p.bbox is not None for p in players)


def _body_score(bbox: BBox, u: float, v: float, cfg: EventEngineConfig, keep: bool,
                raised: bool = False) -> float | None:
    """Image control score of a ball at (u, v) for the player in ``bbox``, or None (out of reach).

    A held or dribbled ball sits anywhere across the body's image span (the hands reach the sides,
    a dribble bounces beside the feet), so a ball inside the span scores near 1 whatever its distance
    from the centre line (a slight centre preference breaks ties between overlapping boxes), and the
    score falls to 0 at ``holder_img_reach_ft`` (``holder_img_keep_reach_ft`` for the current holder)
    beyond the span. Vertically the ball must lie between just above the box top (the person box
    includes raised hands) and the dribble band below the feet; above the head line it is scaled by
    ``holder_above_head_factor`` (not for ``raised``: the holder's ball raised within his reach)."""
    h = max(bbox.h, 1.0)
    ppf = h / cfg.player_height_ft
    half = max(bbox.w / 2.0, 1e-6)
    off = abs(u - (bbox.x1 + bbox.x2) / 2.0)
    reach = cfg.holder_img_keep_reach_ft if keep else cfg.holder_img_reach_ft
    excess = max(0.0, off - half) / ppf
    if excess > reach:
        return None
    bottom = cfg.holder_keep_bottom_frac if keep else cfg.holder_img_bottom_frac
    if not bbox.y1 - cfg.holder_keep_top_frac * h <= v <= bbox.y2 + bottom * h:
        return None
    if excess > 0.0:
        score = (1.0 - cfg.holder_img_centre_pref) * (1.0 - (excess / reach) ** 2)
    else:
        score = 1.0 - cfg.holder_img_centre_pref * (off / half) ** 2
    return score * (1.0 if raised or v >= bbox.y1 + cfg.holder_head_frac * h else cfg.holder_above_head_factor)


def _vertical_weight(cfg: EventEngineConfig, prefer_team: Team | None) -> float:
    """Weight of vertical motion in the relative speed: a caught pass is often raised or dribbled at
    once (low weight); a loose ball falling past a player is not control (full weight)."""
    return cfg.holder_rel_vertical_weight if prefer_team is not None else cfg.holder_rel_vertical_weight_loose


def _rel_speed_court(vel: tuple[float, ...], p: PlayerState, vertical_weight: float) -> float:
    """Ball speed relative to the player; vertical speed is down-weighted (a dribble starts at once)."""
    vz = vel[2] if len(vel) == 3 else 0.0
    return math.sqrt((vel[0] - p.vx) ** 2 + (vel[1] - p.vy) ** 2 + (vertical_weight * vz) ** 2)


@dataclass(frozen=True, slots=True)
class Ray:
    """Camera ground point (cx, cy) and height ``h`` (court ft), for monocular depth checks."""

    cx: float
    cy: float
    h: float

    def at_player(self, ball_floor_xy: tuple[float, float], x: float, y: float) -> tuple[float, float]:
        """(signed lateral offset ft, implied ball height ft) where the ball's viewing ray passes the
        player standing at (x, y). The ball's floor projection lies beyond the ball on that ray."""
        dx, dy = ball_floor_xy[0] - self.cx, ball_floor_xy[1] - self.cy
        n2 = dx * dx + dy * dy
        if n2 < 1e-9:
            return math.inf, math.inf
        qx, qy = x - self.cx, y - self.cy
        s = (qx * dx + qy * dy) / n2
        return (dx * qy - dy * qx) / math.sqrt(n2), self.h * (1.0 - s)


class RayModel:
    """Estimates the camera's ground point and height from monocular observations.

    Ball observations pair an image point with its floor projection, so they determine the floor
    homography exactly. Each rim centre (known court position, known height) projected to the
    floor lies on the line from the camera's ground point through the basket; two rims give the
    ground point (line intersection) and the height (similar triangles).
    """

    def __init__(self, rim_height: float, basket_xy: dict[str, tuple[float, float]], min_pairs: int = 40,
                 min_spread_px: float = 200.0) -> None:
        self.rim_height = rim_height
        self.basket_xy = basket_xy
        self.min_pairs = min_pairs
        self.min_spread_px = min_spread_px
        self._uv: list[tuple[float, float]] = []
        self._xy: list[tuple[float, float]] = []
        self.ray: Ray | None = None
        self._tried_at = 0

    def observe(self, obs: BallObs, rims: dict[str, BBox]) -> None:
        if self.ray is not None or obs.is3d or obs.uv is None or obs.xy is None or len(rims) < 2:
            return
        if self._uv and math.dist(self._uv[-1], obs.uv) < 15.0:
            return  # keep the sample spread out
        self._uv.append(obs.uv)
        self._xy.append(obs.xy)
        n = len(self._uv)
        if n >= self.min_pairs and n >= self._tried_at + 20:
            self._tried_at = n
            self.ray = self._fit(rims)

    def _fit(self, rims: dict[str, BBox]) -> Ray | None:
        import numpy as np

        from courtvision.geometry.homography import Homography

        uv = np.array(self._uv)
        if np.ptp(uv[:, 0]) < self.min_spread_px or np.ptp(uv[:, 1]) < 0.25 * self.min_spread_px:
            return None
        try:
            hom, _, rmse = Homography.fit(uv, np.array(self._xy), ransac_thresh_ft=1.0)
        except ValueError:
            return None
        if rmse > 1.0:
            return None
        lines = []
        for side in ("left", "right"):
            rim = rims.get(side)
            if rim is None:
                return None
            fx, fy = (float(v) for v in hom.image_to_court(np.array(rim.center)))
            lines.append((self.basket_xy[side], (fx, fy)))
        (b1, f1), (b2, f2) = lines
        cg = _intersect(b1, f1, b2, f2)
        if cg is None:
            return None
        d_b, d_f = math.dist(cg, b1), math.dist(cg, f1)
        if d_f <= d_b + 1e-6:
            return None
        h = self.rim_height * d_f / (d_f - d_b)
        if not 10.5 < h < 500.0:
            return None
        log.info("monocular ray model: camera ground (%.1f, %.1f) ft, height %.1f ft", cg[0], cg[1], h)
        return Ray(cg[0], cg[1], h)


def _intersect(a1: tuple[float, float], a2: tuple[float, float], b1: tuple[float, float],
               b2: tuple[float, float]) -> tuple[float, float] | None:
    """Intersection of line a1-a2 with line b1-b2 (None when nearly parallel)."""
    d1 = (a2[0] - a1[0], a2[1] - a1[1])
    d2 = (b2[0] - b1[0], b2[1] - b1[1])
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9 * (math.hypot(*d1) * math.hypot(*d2) + 1e-12):
        return None
    t = ((b1[0] - a1[0]) * d2[1] - (b1[1] - a1[1]) * d2[0]) / den
    return a1[0] + t * d1[0], a1[1] + t * d1[1]


@dataclass(frozen=True, slots=True)
class _Pos:
    t: float
    x: float
    y: float
    speed: float


class PlayerHistory:
    """Short per-player history of floor positions (gather spot, stationarity, id swaps)."""

    def __init__(self, max_len: int = 240, forget_after_s: float = 60.0, prune_every: int = 300) -> None:
        self._h: dict[str, deque[_Pos]] = {}
        self.team: dict[str, Team | None] = {}
        self.bbox: dict[str, BBox | None] = {}
        self.last_seen: dict[str, float] = {}
        self._max_len = max_len
        self._forget_after_s = forget_after_s  # provisional track ids churn all game: bound the memory
        self._prune_every = prune_every
        self._updates = 0

    def update(self, frame: FrameState) -> None:
        self._updates += 1
        if self._updates % self._prune_every == 0:
            self._prune(frame.t)
        for p in frame.players:
            h = self._h.get(p.player_id)
            if h is None:
                h = self._h[p.player_id] = deque(maxlen=self._max_len)
            h.append(_Pos(frame.t, p.x, p.y, p.speed))
            if p.team is not None:
                self.team[p.player_id] = p.team
            self.bbox[p.player_id] = p.bbox
            self.last_seen[p.player_id] = frame.t

    def _prune(self, now: float) -> None:
        """Forget positions / boxes of ids unseen for ``forget_after_s`` (team labels are kept)."""
        stale = [pid for pid, seen in self.last_seen.items() if now - seen > self._forget_after_s]
        for pid in stale:
            self._h.pop(pid, None)
            self.bbox.pop(pid, None)
            del self.last_seen[pid]

    def position_at(self, pid: str, t: float, not_before: float | None = None) -> tuple[float, float] | None:
        """Floor position at (or just before) ``t``; never earlier than ``not_before`` when given."""
        h = self._h.get(pid)
        if not h:
            return None
        lo = -math.inf if not_before is None else not_before
        chosen: _Pos | None = None
        for s in reversed(h):
            if s.t <= t and s.t >= lo:
                chosen = s
                break
            if s.t < lo:
                break
        if chosen is None:
            chosen = next((s for s in h if s.t >= lo), h[-1])
        return chosen.x, chosen.y

    def last(self, pid: str) -> tuple[float, float] | None:
        h = self._h.get(pid)
        return (h[-1].x, h[-1].y) if h else None

    def mean_speed(self, pid: str, t0: float, t1: float) -> float | None:
        h = self._h.get(pid)
        if not h:
            return None
        vals = [s.speed for s in h if t0 <= s.t <= t1]
        return sum(vals) / len(vals) if vals else None
