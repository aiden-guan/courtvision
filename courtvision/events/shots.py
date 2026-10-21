"""Ball flights after a release: shot vs pass, make / miss, blocks and the cue features.

A ``Flight`` starts when the holder loses the ball and ends when someone gains control, the ball
goes dead, or the engine drops it. It is stepped with one ``FlightSample`` per frame (``None`` when
the ball is not observed) and emits ``FlightSignal``s:

* ``shot``: the ball entered a rim zone. 3-D: within ``rim_zone_radius_ft`` horizontally of a
  basket and ``z >= rim_height - rim_zone_below_ft``. Monocular: inside an image box around/above
  the rim bbox scaled by the rim width. The signal carries the release time (attempts are
  timestamped at release).
* ``made``: 3-D: the ball centre crosses the rim plane downward within
  ``rim_radius - 0.5 * ball_radius + make_tol_ft`` of the basket centre. Monocular: the image
  trajectory crosses the rim bbox top line downward with the interpolated crossing inside the rim
  span shrunk by the ball radius, and the ball is next seen below the rim inside the span (a ball
  seen back above the top line is a rim bounce and cancels the crossing).
* ``missed``: the ball leaves the rim zone downward or sideways, a player gains control first, or
  nothing resolves within ``shot_resolve_timeout_s`` (then ``needs_review``).
* ``blocked``: a shot-like release (rising fast) reversed abruptly next to a defender before the
  rim zone (the engine then emits attempt + SHOT_MISSED + BLOCK).
* ``out``: 3-D only, the ball is low and beyond the boundary.

Cue features for the cue models (RESEARCH §7A.2) are collected along the way: ``shot_cues()``
and ``make_cues()``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from courtvision.geometry.court import CourtSpec
from courtvision.types import BBox, PlayerState, Side, Team

if TYPE_CHECKING:
    from .engine import EventEngineConfig

__all__ = ["Flight", "FlightSample", "FlightSignal", "Release", "shot_type_for", "three_margin_ft"]

log = logging.getLogger(__name__)

G_FTS2 = 32.174  # gravity, ft/s^2
SIDES: tuple[Side, Side] = ("left", "right")


@dataclass(frozen=True, slots=True)
class FlightSample:
    t: float
    frame_idx: int
    uv: tuple[float, float] | None
    xy: tuple[float, float] | None
    z: float | None
    r_px: float | None

    @property
    def is3d(self) -> bool:
        return self.z is not None and self.xy is not None


@dataclass(frozen=True, slots=True)
class FlightSignal:
    kind: Literal["shot", "made", "missed", "blocked", "out"]
    t: float
    frame_idx: int
    side: Side | None = None
    player_id: str | None = None
    reason: str = ""
    needs_review: bool = False
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Release:
    """Who released the ball, when and where (court ft)."""

    player_id: str
    team: Team | None
    t: float
    frame_idx: int
    xy: tuple[float, float] | None  # releaser's floor position at release
    gather_xy: tuple[float, float] | None  # last grounded position before the gather (D8)
    bbox: BBox | None = None  # releaser's image bbox at release (monocular scale)
    control_onset_t: float | None = None


def three_margin_ft(court: CourtSpec, x: float, y: float, side: Side) -> float:
    """Signed distance (ft) beyond ``side``'s 3-point line; <= 0 is a 2 (the line is 2)."""
    bx, by = court.basket_xy(side)
    along = (x - bx) * (1.0 if side == "left" else -1.0)
    lateral = y - by
    if along <= court.three_junction_along:
        return abs(lateral) - court.corner_three_dist
    return math.hypot(along, lateral) - court.three_pt_radius


def shot_type_for(distance_ft: float, release_z: float | None, control_s: float | None, is_ft: bool,
                  cfg: EventEngineConfig) -> str:
    """Heuristic shot type: free_throw | tip | dunk | layup | jumper (no hook detection)."""
    if is_ft:
        return "free_throw"
    if control_s is not None and control_s <= cfg.tip_max_control_s and distance_ft <= cfg.layup_max_dist_ft:
        return "tip"
    if release_z is not None and release_z >= cfg.dunk_min_release_z_ft and distance_ft <= cfg.dunk_max_dist_ft:
        return "dunk"
    if distance_ft <= cfg.layup_max_dist_ft:
        return "layup"
    return "jumper"


class Flight:
    """One released ball, classified as it flies."""

    def __init__(self, release: Release, court: CourtSpec, cfg: EventEngineConfig, *,
                 free_throw_side: Side | None = None, context: str = "live") -> None:
        self.release = release
        self.court = court
        self.cfg = cfg
        self.context = context  # "live" | "dead" (inbound / pickup tosses)
        self.is_ft = free_throw_side is not None
        self.state: str = "shot" if self.is_ft else "air"  # air -> shot -> made | missed ; air -> loose (blocked)
        self.target: Side | None = free_throw_side
        self.zone_t: float | None = None  # first sample inside the target's rim zone
        self.samples: list[FlightSample] = []
        self.result_t: float | None = None
        self.result_frame: int | None = None
        self.result_reason = ""
        self.blocker: str | None = None
        self.block_data: dict[str, Any] = {}
        self.oob_t: float | None = None
        self.oob_frame: int | None = None
        self.max_z = -math.inf
        self.rim_contact = False
        self.occluded_frames = 0
        self._entry_samples = 0
        self._best_margin: float | None = None
        self._cross: tuple[FlightSample, FlightSample | None, float] | None = None  # (at, before, margin)
        self._drop_frac: float | None = None
        self._crossed_inside = False
        self._crossed_inside_3d = False
        self._ring_bounce = False  # 3-D: the ball was deflected at the ring / board (no block after that)
        self._last_dv: float | None = None
        self._view_quality: float | None = None
        self._shot_cues: dict[str, Any] | None = None
        self.attempt_id: str | None = None
        self.shot_id: str | None = None
        self.continuation = False  # released in the act of shooting while fouled (the whistle does not kill it)
        # a block candidate the engine saw in hindsight (engine._retro_block): (t, frame, blocker, side, data)
        self.retro_block: tuple[float, int, str, Side, dict[str, Any]] | None = None

    # ------------------------------------------------------------------ status

    @property
    def is_shot(self) -> bool:
        return self.state in ("shot", "made", "missed") or (self.state == "loose" and self.blocker is not None)

    @property
    def resolved(self) -> bool:
        return self.state in ("made", "missed", "loose")

    @property
    def made(self) -> bool:
        return self.state == "made"

    def last_sample(self) -> FlightSample | None:
        return self.samples[-1] if self.samples else None

    # ------------------------------------------------------------------ stepping

    def step(self, s: FlightSample | None, t: float, frame_idx: int, rims: Mapping[Side, BBox],
             players: Sequence[PlayerState], team_of: Callable[[str], Team | None]) -> list[FlightSignal]:
        out: list[FlightSignal] = []
        if s is None:
            if self.state == "shot":
                self.occluded_frames += 1
            out += self._check_timeout(t, frame_idx)
            return out
        prev = self.samples[-1] if self.samples else None
        self.samples.append(s)
        if s.z is not None:
            self.max_z = max(self.max_z, s.z)
        if self.state == "air":
            out += self._check_entry(prev, s, rims, players, team_of)
        if self.state == "shot" and self.zone_t is None:  # free throw on its way to the rim
            if (self._zone3d(s) if s.is3d else self._zone_image(s, rims)) == self.target:
                self.zone_t = s.t
        if self.state == "shot" and self.zone_t is not None and not self.is_ft:
            blocked = self._check_rim_block(players, team_of, rims)
            if blocked:
                return out + blocked
        if self.state == "shot" and self.zone_t is not None:
            out += self._step_shot3d(prev, s) if s.is3d else self._step_shot_image(prev, s, rims)
        if self.state != "made":
            out += self._check_out(s)
        out += self._check_timeout(t, frame_idx)
        return out

    def on_control(self, t: float, frame_idx: int) -> list[FlightSignal]:
        """A player gained control (at ``t``): an unresolved shot is a miss."""
        if self.state != "shot":
            return []
        return [self._resolve("missed", t, frame_idx, "player gained control before the shot resolved")]

    def finish(self, t: float, frame_idx: int) -> list[FlightSignal]:
        """End of stream / dead ball: an unresolved shot becomes a miss that needs review."""
        if self.state != "shot":
            return []
        return [self._resolve("missed", t, frame_idx, "unresolved at end of tracking", needs_review=True)]

    # ------------------------------------------------------------------ shot entry

    def _check_entry(self, prev: FlightSample | None, s: FlightSample, rims: Mapping[Side, BBox],
                     players: Sequence[PlayerState], team_of: Callable[[str], Team | None]) -> list[FlightSignal]:
        if self.context != "live":
            return []
        side = self._zone3d(s) if s.is3d else self._zone_image(s, rims)
        if side is not None and s.is3d and prev is not None and prev.is3d and not self._aimed_at(prev, s, side):
            side = None  # e.g. a high pass crossing the zone: its arc comes down far from the rim
        if side is not None:
            self._entry_samples += 1
            if s.is3d or self._entry_samples >= self.cfg.rim_zone_min_samples_img:
                return self._enter(side, s)
            return []
        self._entry_samples = 0
        blocked = self._check_block(s, players, team_of, rims)
        if blocked:
            return blocked
        if s.is3d and prev is not None and prev.is3d:
            return self._check_airball(prev, s)
        return []

    def _enter(self, side: Side, s: FlightSample) -> list[FlightSignal]:
        self.retro_block = None  # it reached the rim zone: a shot that went on, not a block
        self.state, self.target, self.zone_t = "shot", side, s.t
        self._shot_cues = self._compute_shot_cues(side)
        r = self.release
        return [FlightSignal("shot", r.t, r.frame_idx, side=side, reason="ball entered the rim zone")]

    def _zone3d(self, s: FlightSample) -> Side | None:
        cfg, court = self.cfg, self.court
        if s.z is None or s.xy is None or s.z < court.rim_height - cfg.rim_zone_below_ft:
            return None
        best, best_d = None, cfg.rim_zone_radius_ft
        for side in SIDES:
            d = court.distance_to_basket(s.xy[0], s.xy[1], side)
            if d <= best_d:
                best, best_d = side, d
        return best

    def _aimed_at(self, prev: FlightSample, s: FlightSample, side: Side) -> bool:
        """3-D: does the ballistic extrapolation descend through rim height near ``side``'s basket?"""
        dt = s.t - prev.t
        if dt <= 1e-6:
            return True
        vx = (s.xy[0] - prev.xy[0]) / dt  # type: ignore[index]
        vy = (s.xy[1] - prev.xy[1]) / dt  # type: ignore[index]
        vz = (s.z - prev.z) / dt  # type: ignore[operator]
        dz = s.z - self.court.rim_height  # type: ignore[operator]
        disc = vz * vz + 2.0 * G_FTS2 * dz
        if disc < 0.0:
            return True  # never reaches rim height again: let the zone logic decide
        tau = (vz + math.sqrt(disc)) / G_FTS2
        lx, ly = s.xy[0] + vx * tau, s.xy[1] + vy * tau  # type: ignore[index]
        return self.court.distance_to_basket(lx, ly, side) <= self.cfg.shot_aim_radius_ft

    def _zone_image(self, s: FlightSample, rims: Mapping[Side, BBox]) -> Side | None:
        if s.uv is None:
            return None
        cfg = self.cfg
        u, v = s.uv
        for side, rim in rims.items():
            w = max(rim.w, 1.0)
            if rim.x1 - cfg.rim_zone_side_k * w <= u <= rim.x2 + cfg.rim_zone_side_k * w \
                    and rim.y1 - cfg.rim_zone_up_k * w <= v <= rim.y2 + cfg.rim_zone_down_k * w:
                if self._view_quality is None:
                    self._view_quality = min(1.0, rim.h / w) if w > 0 else None
                return side
        return None

    # ------------------------------------------------------------------ 3-D make / miss

    def _step_shot3d(self, prev: FlightSample | None, s: FlightSample) -> list[FlightSignal]:
        cfg, court = self.cfg, self.court
        assert self.target is not None and s.xy is not None and s.z is not None
        bx, by = court.basket_xy(self.target)
        rim_h, rim_r, ball_r = court.rim_height, court.rim_radius, cfg.ball_radius_ft
        d = math.hypot(s.xy[0] - bx, s.xy[1] - by)
        if math.hypot(d - rim_r, s.z - rim_h) <= ball_r + cfg.rim_contact_dist_ft:
            self.rim_contact = True
        if prev is not None and prev.is3d and prev.z > rim_h >= s.z:  # type: ignore[operator]
            f = (prev.z - rim_h) / (prev.z - s.z)  # type: ignore[operator]
            cx = prev.xy[0] + f * (s.xy[0] - prev.xy[0])  # type: ignore[index]
            cy = prev.xy[1] + f * (s.xy[1] - prev.xy[1])  # type: ignore[index]
            dc = math.hypot(cx - bx, cy - by)
            margin = rim_r - 0.5 * ball_r + cfg.make_tol_ft - dc
            self._best_margin = margin if self._best_margin is None else max(self._best_margin, margin)
            if margin >= 0.0:
                self._crossed_inside = self._crossed_inside_3d = True
                return [self._resolve("made", s.t, s.frame_idx, "ball centre crossed the rim plane inside the rim")]
        if d > cfg.rim_zone_radius_ft + cfg.rim_zone_exit_margin_ft:
            return [self._resolve("missed", s.t, s.frame_idx, "ball left the rim zone sideways")]
        if s.z < rim_h - cfg.rim_zone_below_ft:
            return [self._resolve("missed", s.t, s.frame_idx, "ball left the rim zone downward")]
        return []

    # ------------------------------------------------------------------ monocular make / miss

    def _step_shot_image(self, prev: FlightSample | None, s: FlightSample, rims: Mapping[Side, BBox]) -> list[FlightSignal]:
        rim = rims.get(self.target) if self.target is not None else None
        if rim is None or s.uv is None:
            return []
        cfg = self.cfg
        u, v = s.uv
        w = max(rim.w, 1.0)
        r_px = s.r_px if s.r_px is not None else w * cfg.ball_radius_ft / (2.0 * self.court.rim_radius)
        self._note_image_rim_contact(prev, s, rim, w)
        if self._cross is not None:
            return self._confirm_crossing(s, rim, w, r_px)
        if prev is not None and prev.uv is not None and prev.uv[1] < rim.y1 <= v:
            f = (rim.y1 - prev.uv[1]) / (v - prev.uv[1])
            uc = prev.uv[0] + f * (u - prev.uv[0])
            lo, hi = rim.x1 + r_px - cfg.make_span_tol_px, rim.x2 - r_px + cfg.make_span_tol_px
            margin = min(uc - lo, hi - uc)
            self._best_margin = margin if self._best_margin is None else max(self._best_margin, margin)
            if margin >= 0.0:
                before = self.samples[-2] if len(self.samples) >= 2 else None
                self._cross = (s, before, margin)
                return self._confirm_crossing(s, rim, w, r_px)
        if u < rim.x1 - cfg.rim_exit_side_k * w or u > rim.x2 + cfg.rim_exit_side_k * w:
            return [self._resolve("missed", s.t, s.frame_idx, "ball left the rim zone sideways")]
        if v > rim.y2 + cfg.rim_exit_down_k * w:
            return [self._resolve("missed", s.t, s.frame_idx, "ball dropped below the rim outside it")]
        return []

    def _confirm_crossing(self, s: FlightSample, rim: BBox, w: float, r_px: float) -> list[FlightSignal]:
        assert self._cross is not None and s.uv is not None
        cfg = self.cfg
        u, v = s.uv
        at, before, _ = self._cross
        within = rim.x1 - cfg.made_within_k * w <= u <= rim.x2 + cfg.made_within_k * w
        if within and v > rim.y2 + cfg.made_below_k * w:
            self._crossed_inside = True
            self._drop_frac = self._drop_fraction(before, at, s, rim, w)
            return [self._resolve("made", at.t, at.frame_idx, "crossed the rim top line inside the rim, then seen below it")]
        if v < rim.y1 - cfg.made_bounce_k * w or not within:
            self._cross = None  # rim bounce: back above the rim or out to the side
            self.rim_contact = True
        return []

    def _note_image_rim_contact(self, prev: FlightSample | None, s: FlightSample, rim: BBox, w: float) -> None:
        if prev is None or prev.uv is None or s.uv is None or s.t <= prev.t:
            return
        dv = (s.uv[1] - prev.uv[1]) / (s.t - prev.t)
        near = rim.x1 - w <= s.uv[0] <= rim.x2 + w and rim.y1 - 1.5 * w <= s.uv[1] <= rim.y2 + 0.5 * w
        if near and self._last_dv is not None and self._last_dv > 0.0 > dv:
            self.rim_contact = True  # falling ball turned upward next to the rim
        self._last_dv = dv

    def _drop_fraction(self, before: FlightSample | None, at: FlightSample, below: FlightSample, rim: BBox,
                       w: float) -> float | None:
        """Post-crossing drop speed as a fraction of free fall (net drag gives 10-90% [16])."""
        if before is None or before.uv is None or at.uv is None or below.uv is None:
            return None
        if below.t <= at.t or at.t <= before.t:
            return None
        sin_e = min(max(rim.h / w, 0.05), 1.0)
        px_per_ft_v = (w / (2.0 * self.court.rim_radius)) * math.sqrt(1.0 - sin_e * sin_e)
        if px_per_ft_v <= 0:
            return None
        v_pre = max((at.uv[1] - before.uv[1]) / (at.t - before.t), 0.0) / px_per_ft_v
        dh = max(below.uv[1] - at.uv[1], 0.0) / px_per_ft_v
        v_post = dh / (below.t - at.t)
        v_end = math.sqrt(v_pre * v_pre + 2.0 * G_FTS2 * dh)
        expected = 0.5 * (v_pre + v_end)
        return v_post / expected if expected > 1e-6 else None

    # ------------------------------------------------------------------ blocks, air balls, out

    def _shot_like(self) -> tuple[bool, float, Side | None]:
        """(rising fast toward a basket?, upward speed ft/s, candidate basket)."""
        r = self.release
        if len(self.samples) < 2 or r.xy is None:
            return False, 0.0, None
        a, b = self.samples[0], self.samples[1]
        dt = b.t - a.t
        if dt <= 0:
            return False, 0.0, None
        side = self.court.nearest_basket(*r.xy)
        if self.court.distance_to_basket(r.xy[0], r.xy[1], side) > self.cfg.block_max_basket_dist_ft:
            return False, 0.0, None
        if a.is3d and b.is3d:
            up = (b.z - a.z) / dt  # type: ignore[operator]
        elif a.uv is not None and b.uv is not None:
            up = -(b.uv[1] - a.uv[1]) / dt / self._px_per_ft()
        else:
            return False, 0.0, None
        return up >= self.cfg.block_min_up_speed_fts, up, side

    def _px_per_ft(self) -> float:
        bb = self.release.bbox
        return bb.h / self.cfg.player_height_ft if bb is not None and bb.h > 0 else self.cfg.nominal_px_per_ft

    def _check_block(self, s: FlightSample, players: Sequence[PlayerState],
                     team_of: Callable[[str], Team | None], rims: Mapping[Side, BBox]) -> list[FlightSignal]:
        cfg = self.cfg
        if self.is_ft or len(self.samples) < 3 or s.t - self.release.t > cfg.block_max_after_release_s:
            return []
        shot_like, _, side = self._shot_like()
        if not shot_like or side is None:
            return []
        image = not s.is3d
        span = cfg.block_image_span if image else 1  # image velocities are noisy: longer baselines
        if len(self.samples) < 3 * span + 1:
            return []
        # p1 is the contact; p3 checks that the reversal lasts (a one-frame jump at the real release
        # of a ball carried above holder height is not a block: the ball goes on toward the rim)
        p0, p1, p2, p3 = (self.samples[-1 - 3 * span], self.samples[-1 - 2 * span], self.samples[-1 - span],
                          self.samples[-1])
        rim = rims.get(side)
        dv = _reversal(p0, p1, p2, self._px_per_ft(), self.court.basket_xy(side),
                       rim.center[0] if rim is not None else None, cfg.block_min_toward_fts, p3)
        if dv is None or dv < (cfg.block_reverse_dv_img_fts if image else cfg.block_reverse_dv_fts):
            return []
        if p1.is3d and self.release.xy is not None \
                and math.dist(p1.xy, self.release.xy) > cfg.block_max_from_release_ft:  # type: ignore[arg-type]
            return []
        # contact at the reversal vertex; a teammate of the passer nearer the ball is a catch (e.g. into a
        # dribble), not a block
        return self._blocked_at(p1, dv, side, players, team_of)

    def _check_rim_block(self, players: Sequence[PlayerState], team_of: Callable[[str], Team | None],
                         rims: Mapping[Side, BBox]) -> list[FlightSignal]:
        """A shot already inside the rim zone (layups, dunks, close shots: they enter it at the release)
        turned back or knocked down by a defender before it touched the ring or the board. That is a
        block too: any contact before the ball touches the rim, at the rim included (FIBA; rules note
        §2a). Ring and board bounces are told apart by where the reversal happens (3-D)."""
        cfg, court = self.cfg, self.court
        if self._ring_bounce or self._crossed_inside or self._cross is not None or self.target is None \
                or len(self.samples) < 4:
            return []
        p0, p1, p2, p3 = self.samples[-4:]
        if p1.t - self.release.t > cfg.block_max_after_release_s:
            return []
        if not all(p.is3d for p in (p0, p1, p2, p3)):
            return []
        dv = _deflection3d(p0, p1, p2, p3, court.basket_xy(self.target), cfg.block_min_toward_fts)
        if dv is None or dv < cfg.block_reverse_dv_fts:
            return []
        if self._at_ring_or_board(p1) or self._at_ring_or_board(p2):
            # the ball came off the ring or the board: no block after that. (Mere proximity is not contact:
            # a layup released over the ring passes within a ball radius of it without bouncing.)
            self._ring_bounce = True
            return []
        # no teammate test: a shooter's teammate under the rim touching the ball in flight is rare (basket
        # interference), while teammates crashing the boards stand next to every layup
        return self._blocked_at(p1, dv, self.target, players, team_of, mates=False)

    def _at_ring_or_board(self, s: FlightSample) -> bool:
        """3-D: the ball touches the ring or the backboard at ``s`` (the rim-contact distance of ``_step_shot3d``)."""
        cfg, court = self.cfg, self.court
        assert self.target is not None and s.xy is not None and s.z is not None
        bx, by = court.basket_xy(self.target)
        d = math.hypot(s.xy[0] - bx, s.xy[1] - by)
        rim_h = court.rim_height
        if math.hypot(d - court.rim_radius, s.z - rim_h) <= cfg.ball_radius_ft + cfg.rim_contact_dist_ft:
            return True
        board_x = court.backboard_offset if self.target == "left" else court.length - court.backboard_offset
        return abs(s.xy[0] - board_x) <= cfg.ball_radius_ft + cfg.rim_contact_dist_ft \
            and abs(s.xy[1] - by) <= cfg.backboard_half_width_ft + cfg.ball_radius_ft \
            and rim_h - cfg.rim_zone_below_ft <= s.z <= rim_h + cfg.backboard_above_rim_ft

    def _blocked_at(self, p1: FlightSample, dv: float, side: Side, players: Sequence[PlayerState],
                    team_of: Callable[[str], Team | None], mates: bool = True) -> list[FlightSignal]:
        """The block decision at the reversal vertex ``p1``: an opponent at the ball (not a teammate of the
        shooter nearer it: that is a catch / tip), then the flight turns loose with its blocker."""
        cfg = self.cfg
        blocker, dist, above = self._defender_at(p1, players, team_of)
        if blocker is None:
            return []
        mate = self._teammate_dist(p1, players, team_of) if mates else None
        if mate is not None and dist is not None and mate <= dist:
            return []
        self.blocker = blocker
        self.block_data = {"reversal_dv": dv, "defender_dist_ft": dist, "above_shoulder": above, "basket": side}
        self.target = side
        self.state = "loose"
        if self._shot_cues is None:  # a block in the rim zone keeps the cues computed at the zone entry
            self._shot_cues = self._compute_shot_cues(side)
        self.result_t, self.result_frame = p1.t, p1.frame_idx
        self.result_reason = "shot reversed by a defender before reaching the rim"
        return [FlightSignal("blocked", p1.t, p1.frame_idx, side=side, player_id=blocker, reason=self.result_reason,
                             data=dict(self.block_data))]

    def mark_blocked(self, t: float, frame_idx: int, blocker: str, side: Side, data: dict[str, Any]) -> FlightSignal:
        """The engine found the block in hindsight (monocular retro-block): the flight is a blocked shot."""
        self.blocker, self.target, self.state = blocker, side, "loose"
        self.block_data = {"basket": side, **data}
        if self._shot_cues is None:
            self._shot_cues = self._compute_shot_cues(side)
        self.result_t, self.result_frame = t, frame_idx
        self.result_reason = "shot turned back on its way up by a defender (seen in hindsight)"
        return FlightSignal("blocked", t, frame_idx, side=side, player_id=blocker, reason=self.result_reason,
                            data=dict(self.block_data))

    def _defender_at(self, s: FlightSample, players: Sequence[PlayerState],
                     team_of: Callable[[str], Team | None]) -> tuple[str | None, float | None, bool]:
        cfg = self.cfg
        team = self.release.team
        if team is None:
            return None, None, False
        best, best_d, above = None, math.inf, False
        for p in players:
            if team_of(p.player_id) is not team.other():
                continue
            if s.is3d:
                d = math.hypot(p.x - s.xy[0], p.y - s.xy[1])  # type: ignore[index]
                if d <= cfg.block_defender_radius_ft and s.z >= cfg.block_min_z_ft and d < best_d:  # type: ignore[operator]
                    best, best_d, above = p.player_id, d, True
            elif s.uv is not None and p.bbox is not None:
                bb = p.bbox
                dx = bb.w * cfg.block_bbox_expand / 2.0
                box = BBox(bb.x1 - dx, bb.y1 - cfg.block_reach_frac * bb.h, bb.x2 + dx, bb.y2)  # raised arms
                if box.contains(*s.uv):
                    d = abs(s.uv[0] - p.bbox.center[0]) / max(p.bbox.h / cfg.player_height_ft, 1e-6)
                    hi = s.uv[1] <= p.bbox.y1 + cfg.block_shoulder_frac * p.bbox.h
                    if hi and d < best_d:
                        best, best_d, above = p.player_id, d, True
        return best, (best_d if best is not None else None), above

    def _teammate_dist(self, s: FlightSample, players: Sequence[PlayerState],
                       team_of: Callable[[str], Team | None]) -> float | None:
        """Distance (ft, same metric as ``_defender_at``) from the ball to the nearest teammate of the
        releaser (the releaser excluded) who could have caught it, or None."""
        cfg, r = self.cfg, self.release
        if r.team is None:
            return None
        best: float | None = None
        for p in players:
            if p.player_id == r.player_id or team_of(p.player_id) is not r.team:
                continue
            if s.is3d:
                d = math.hypot(p.x - s.xy[0], p.y - s.xy[1])  # type: ignore[index]
                if d > cfg.block_defender_radius_ft:
                    continue
            elif s.uv is not None and p.bbox is not None:
                bb = p.bbox
                dx = bb.w * cfg.block_bbox_expand / 2.0
                if not BBox(bb.x1 - dx, bb.y1 - cfg.block_reach_frac * bb.h, bb.x2 + dx, bb.y2).contains(*s.uv):
                    continue
                d = abs(s.uv[0] - bb.center[0]) / max(bb.h / cfg.player_height_ft, 1e-6)
            else:
                continue
            best = d if best is None else min(best, d)
        return best

    def _check_airball(self, prev: FlightSample, s: FlightSample) -> list[FlightSignal]:
        cfg, court = self.cfg, self.court
        rim_h = court.rim_height
        if not (prev.z >= rim_h > s.z) or self.max_z < rim_h + cfg.airball_min_apex_above_ft:  # type: ignore[operator]
            return []
        shot_like, _, side = self._shot_like()
        r = self.release
        if not shot_like or side is None or r.xy is None:
            return []
        if court.distance_to_basket(s.xy[0], s.xy[1], side) > cfg.airball_radius_ft:  # type: ignore[index]
            return []
        if court.distance_to_basket(r.xy[0], r.xy[1], side) < cfg.airball_min_release_dist_ft:
            return []
        out = self._enter(side, s)
        out.append(self._resolve("missed", s.t, s.frame_idx, "air ball: fell past rim height without reaching the rim"))
        return out

    def _check_out(self, s: FlightSample) -> list[FlightSignal]:
        if self.oob_t is not None or not s.is3d:
            return []
        cfg = self.cfg
        x, y = s.xy  # type: ignore[misc]
        if s.z <= cfg.oob_max_z_ft and not self.court.in_bounds(x, y, margin=cfg.oob_margin_ft):  # type: ignore[operator]
            self.oob_t, self.oob_frame = s.t, s.frame_idx
            return [FlightSignal("out", s.t, s.frame_idx, reason="ball out of bounds")]
        return []

    def _check_timeout(self, t: float, frame_idx: int) -> list[FlightSignal]:
        if self.state != "shot":
            return []
        start = self.zone_t if self.zone_t is not None else self.release.t
        if t - start > self.cfg.shot_resolve_timeout_s:
            return [self._resolve("missed", t, frame_idx, "shot unresolved after the timeout", needs_review=True)]
        return []

    def _resolve(self, kind: Literal["made", "missed"], t: float, frame_idx: int, reason: str,
                 needs_review: bool = False) -> FlightSignal:
        self.state = kind
        self.result_t, self.result_frame, self.result_reason = t, frame_idx, reason
        return FlightSignal(kind, t, frame_idx, side=self.target, reason=reason, needs_review=needs_review)

    # ------------------------------------------------------------------ cues

    def shot_cues(self) -> dict[str, Any]:
        if self._shot_cues is None:
            self._shot_cues = self._compute_shot_cues(self.target) if self.target is not None else {}
        return dict(self._shot_cues)

    def make_cues(self) -> dict[str, Any]:
        cfg = self.cfg
        three_d = bool(self.samples and self.samples[-1].is3d)
        frac = self._drop_frac
        cues: dict[str, Any] = {
            "crossed_inside": self._crossed_inside,
            "crossed_inside_3d": self._crossed_inside_3d,
            "rim_contact": self.rim_contact,
            "occluded_frames": self.occluded_frames,
            "drop_speed_frac": frac,
            "drop_in_band": frac is not None and cfg.drop_band[0] <= frac < cfg.drop_band[1],
            "drop_is_freefall": frac is not None and frac >= cfg.freefall_min_frac,
            "net_motion_z": None,
            "bib_conf_max": None,
            "view_quality": 1.0 if three_d else self._view_quality,
        }
        key = "crossing_margin_ft" if three_d else "crossing_margin_px"
        cues[key] = self._best_margin
        return cues

    def _compute_shot_cues(self, side: Side) -> dict[str, Any]:
        r, court = self.release, self.court
        pre = [s for s in self.samples if self.zone_t is None or s.t <= self.zone_t] or list(self.samples)
        shot_like, up, _ = self._shot_like()
        rmse, n = _parabola_fit(pre, self.cfg.nominal_px_per_ft)
        return {
            "entered_rim_zone": True,
            "upward_speed": up,
            "toward_rim_cos": self._toward_rim_cos(side),
            "in_frontcourt": bool(r.xy is not None and court.half(r.xy[0]) == side),
            "parabola_rmse_px": rmse,
            "parabola_n_pts": n,
            "release_sep_ft": self._release_sep_ft(),
        }

    def _toward_rim_cos(self, side: Side) -> float | None:
        r = self.release
        if len(self.samples) < 2:
            return None
        a, b = self.samples[0], self.samples[min(len(self.samples) - 1, 3)]
        if a.is3d and b.is3d and r.xy is not None:
            bx, by = self.court.basket_xy(side)
            mv = (b.xy[0] - a.xy[0], b.xy[1] - a.xy[1])  # type: ignore[index]
            to = (bx - r.xy[0], by - r.xy[1])
            n1, n2 = math.hypot(*mv), math.hypot(*to)
            return (mv[0] * to[0] + mv[1] * to[1]) / (n1 * n2) if n1 > 1e-6 and n2 > 1e-6 else None
        return None

    def _release_sep_ft(self) -> float | None:
        r = self.release
        if not self.samples:
            return None
        s = next((x for x in self.samples if x.t - r.t >= 0.1), self.samples[-1])
        if s.is3d and r.xy is not None:
            return math.hypot(s.xy[0] - r.xy[0], s.xy[1] - r.xy[1], max(s.z - self.cfg.hands_height_ft, 0.0))  # type: ignore[index, operator]
        if s.uv is not None and r.bbox is not None:
            top = (r.bbox.center[0], r.bbox.y1)
            return math.dist(s.uv, top) / self._px_per_ft()
        return None


def _reversal(p0: FlightSample, p1: FlightSample, p2: FlightSample, px_per_ft: float,
              target_xy: tuple[float, float], target_u: float | None, min_toward_fts: float,
              p3: FlightSample | None = None) -> float | None:
    """Velocity change (ft/s) at p1 when the ball was turned back: it moved toward the target
    basket (3-D) / rim (image) and then away from it, or rose and then dropped sharply. With
    ``p3`` the reversal must last: from p2 to p3 the ball must not head back toward the target at
    shot speed (turned back) or rise again (knocked down), and it must still be in flight: moving
    away (turned back) or falling (knocked down) at >= ``min_toward_fts``, as a ball caught and
    lowered into a receiver's hands next to a defender is not."""
    dt1, dt2 = p1.t - p0.t, p2.t - p1.t
    dt3 = p3.t - p2.t if p3 is not None else 1.0
    if dt1 <= 0 or dt2 <= 0 or dt3 <= 0:
        return None
    if p0.is3d and p1.is3d and p2.is3d and (p3 is None or p3.is3d):
        v1 = [(b - a) / dt1 for a, b in zip((*p0.xy, p0.z), (*p1.xy, p1.z), strict=True)]  # type: ignore[misc]
        v2 = [(b - a) / dt2 for a, b in zip((*p1.xy, p1.z), (*p2.xy, p2.z), strict=True)]  # type: ignore[misc]
        tx, ty = target_xy[0] - p1.xy[0], target_xy[1] - p1.xy[1]  # type: ignore[index]
        n = math.hypot(tx, ty)
        if n < 1e-6:
            return None
        h1 = (v1[0] * tx + v1[1] * ty) / n
        h2 = (v2[0] * tx + v2[1] * ty) / n
        turned_back = h1 >= min_toward_fts and h2 <= -min_toward_fts  # a catch only stops the ball
        knocked_down = v1[2] > min_toward_fts and v2[2] < -max(0.5 * abs(v1[2]), min_toward_fts)
        if p3 is not None:
            v3 = [(b - a) / dt3 for a, b in zip((*p2.xy, p2.z), (*p3.xy, p3.z), strict=True)]  # type: ignore[misc]
            turned_back = turned_back and (v3[0] * tx + v3[1] * ty) / n <= -min_toward_fts
            knocked_down = knocked_down and v3[2] <= -min_toward_fts
        if not (turned_back or knocked_down):
            return None
        return math.dist(v2, [v1[0], v1[1], v1[2] - G_FTS2 * dt2])
    if p0.uv is None or p1.uv is None or p2.uv is None or (p3 is not None and p3.uv is None):
        return None
    v1 = ((p1.uv[0] - p0.uv[0]) / dt1, (p1.uv[1] - p0.uv[1]) / dt1)
    v2 = ((p2.uv[0] - p1.uv[0]) / dt2, (p2.uv[1] - p1.uv[1]) / dt2)
    v3 = ((p3.uv[0] - p2.uv[0]) / dt3, (p3.uv[1] - p2.uv[1]) / dt3) if p3 is not None and p3.uv is not None else None
    min_px = min_toward_fts * px_per_ft
    turned_back = False
    if target_u is not None:
        sgn = 1.0 if target_u >= p1.uv[0] else -1.0
        turned_back = v1[0] * sgn >= min_px and v2[0] * sgn <= -min_px  # a catch only stops the ball
        turned_back = turned_back and (v3 is None or v3[0] * sgn <= -min_px)
    knocked_down = v1[1] < -min_px and v2[1] > max(0.5 * abs(v1[1]), min_px)  # image v grows downward
    knocked_down = knocked_down and (v3 is None or v3[1] >= min_px)
    if not (turned_back or knocked_down):
        return None
    return math.dist(v1, v2) / px_per_ft


def _deflection3d(p0: FlightSample, p1: FlightSample, p2: FlightSample, p3: FlightSample,
                  target_xy: tuple[float, float], min_away_fts: float) -> float | None:
    """3-D velocity change (ft/s) beyond ballistic at ``p1`` when the ball, still going toward the basket or
    rising, was sent away from it or down, and kept going (p2 -> p3). Near the rim a shot moves slowly (a
    layup is released next to it), so unlike ``_reversal`` no approach speed is required."""
    dt1, dt2, dt3 = p1.t - p0.t, p2.t - p1.t, p3.t - p2.t
    if dt1 <= 0 or dt2 <= 0 or dt3 <= 0:
        return None
    pts = [(*p.xy, p.z) for p in (p0, p1, p2, p3)]  # type: ignore[misc]
    v1 = [(b - a) / dt1 for a, b in zip(pts[0], pts[1], strict=True)]
    v2 = [(b - a) / dt2 for a, b in zip(pts[1], pts[2], strict=True)]
    v3 = [(b - a) / dt3 for a, b in zip(pts[2], pts[3], strict=True)]
    tx, ty = target_xy[0] - pts[1][0], target_xy[1] - pts[1][1]
    n = math.hypot(tx, ty)
    if n < 1e-6:
        return None

    def toward(v: list[float]) -> float:
        return (v[0] * tx + v[1] * ty) / n

    before = toward(v1) > -min_away_fts or v1[2] > 0.0
    sent = toward(v2) <= -min_away_fts or v2[2] <= -min_away_fts
    kept = toward(v3) <= -min_away_fts or v3[2] <= -min_away_fts
    if not (before and sent and kept):
        return None
    return math.dist(v2, [v1[0], v1[1], v1[2] - G_FTS2 * dt2])


def _parabola_fit(samples: Sequence[FlightSample], nominal_px_per_ft: float) -> tuple[float | None, int]:
    """RMSE (px; 3-D residuals converted at a nominal scale) of a quadratic fit of height vs time."""
    if not samples:
        return None, 0
    if samples[0].is3d:
        pts = [(s.t, s.z * nominal_px_per_ft) for s in samples if s.z is not None]
    else:
        pts = [(s.t, -s.uv[1]) for s in samples if s.uv is not None]
    n = len(pts)
    if n < 3:
        return None, n
    t = np.array([p[0] for p in pts]) - pts[0][0]
    y = np.array([p[1] for p in pts])
    coef = np.polyfit(t, y, 2)
    resid = y - np.polyval(coef, t)
    return float(np.sqrt(np.mean(resid * resid))), n
