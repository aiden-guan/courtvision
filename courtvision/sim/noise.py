"""Observation noise for simulator truth frames (what a perception stack would plausibly output).

``NoiseModel(config, camera).apply(frames, seed)`` returns an observed copy of truth frames:
position and bbox jitter, dropped players, optional brief identity swaps, ball dropouts (separate
rates for a held ball and a ball in flight), ball false positives, rim-box jitter, and, in
``monocular`` mode, no ball height and a ball ``court_xy`` equal to the floor-homography projection
of the noisy image point (only right when the ball is on the floor, exactly like monocular video).
Scoreboard readings are kept. The input frames are never modified.

The floor homography comes from the camera when one is given; otherwise (the ARCHITECTURE call
``NoiseModel(NoiseConfig()).apply(frames, seed)``) it is recovered from the truth frames themselves:
a normalised-DLT fit of the 3x4 projection to the true ball centres (court_xy, z) and their image
points, which the simulator computes with that same camera.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass

import numpy as np

from courtvision.geometry.camera import PinholeCamera
from courtvision.geometry.homography import Homography
from courtvision.types import BallState, BBox, FrameState, PlayerState, RimObservation

__all__ = ["NoiseConfig", "NoiseModel", "floor_homography_from_frames"]

log = logging.getLogger(__name__)


@dataclass
class NoiseConfig:
    """Noise levels (defaults per ARCHITECTURE sim/; all are design choices, unverified)."""

    pos_sigma_ft: float = 0.4  # player court-position jitter
    bbox_sigma_px: float = 2.0  # per-edge bbox jitter
    ball_miss_rate: float = 0.12  # ball not detected while in flight / loose
    ball_miss_rate_held: float = 0.35  # ball not detected while held (occluded by hands/body)
    ball_fp_rate: float = 0.01  # a spurious ball detection replaces the observation
    ball_px_sigma: float = 1.5  # ball image-centre jitter
    ball_radius_sigma_px: float = 0.3
    drop_player_rate: float = 0.02  # player missing from a frame
    id_swap_rate: float = 0.0  # P(start an identity swap) per frame
    id_swap_frames: int = 15  # swap duration
    id_swap_max_ft: float = 5.0  # only nearby same-team players swap
    rim_px_sigma: float = 0.5
    monocular: bool = True  # ball z -> None, court_xy -> floor projection of the image point
    held_radius_ft: float = 2.5  # "held" heuristic: ball within this of a player ...
    held_max_z_ft: float = 6.0  # ... and below this height


class NoiseModel:
    """Applies :class:`NoiseConfig` noise to truth frames. ``camera`` gives the monocular floor
    projection; without it the projection is recovered from the frames' 3-D ball truth."""

    def __init__(self, config: NoiseConfig | None = None, camera: PinholeCamera | None = None) -> None:
        self.config = config or NoiseConfig()
        self.camera = camera
        self._floor: Homography | None = camera.floor_homography() if camera is not None else None

    def apply(self, frames: list[FrameState], seed: int = 0, camera: PinholeCamera | None = None) -> list[FrameState]:
        """Observed copies of ``frames`` (deterministic for a given ``seed``)."""
        if camera is not None and camera is not self.camera:
            self.camera, self._floor = camera, camera.floor_homography()
        fitted = self._floor is None and self.config.monocular
        if fitted:
            self._floor = floor_homography_from_frames(frames)
            if self._floor is None:
                log.warning("monocular noise without a camera or 3-D ball truth: ball court_xy will be None")
        try:
            return self._apply(frames, seed)
        finally:
            if fitted:
                self._floor = None  # fitted per call: never reused for other frames

    def _apply(self, frames: list[FrameState], seed: int) -> list[FrameState]:
        rng = np.random.default_rng(seed)
        swap: tuple[str, str, int] | None = None  # (id_a, id_b, frames left)
        out: list[FrameState] = []
        for f in frames:
            swap = self._update_swap(f, swap, rng)
            out.append(FrameState(
                frame_idx=f.frame_idx, t=f.t, players=self._players(f.players, rng, swap),
                ball=self._ball(f, rng), rims=[self._rim(r, rng) for r in f.rims],
                scoreboard=f.scoreboard, image_size=f.image_size))
        return out

    # ------------------------------------------------------------------ players

    def _update_swap(self, f: FrameState, swap: tuple[str, str, int] | None,
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

    def _players(self, players: list[PlayerState], rng: np.random.Generator,
                 swap: tuple[str, str, int] | None) -> list[PlayerState]:
        c = self.config
        by_id = {p.player_id: p for p in players}
        out = []
        for p in players:
            if rng.random() < c.drop_player_rate:
                continue
            ident = p
            if swap is not None and p.player_id in swap[:2]:
                other = by_id.get(swap[1] if p.player_id == swap[0] else swap[0])
                ident = other or p
            dx, dy = rng.normal(0.0, c.pos_sigma_ft, 2)
            out.append(dataclasses.replace(
                p, player_id=ident.player_id, jersey=ident.jersey, track_id=ident.track_id,
                x=p.x + float(dx), y=p.y + float(dy), bbox=self._jitter_box(p.bbox, c.bbox_sigma_px, rng)))
        return out

    @staticmethod
    def _jitter_box(box: BBox | None, sigma: float, rng: np.random.Generator) -> BBox | None:
        if box is None:
            return None
        d = rng.normal(0.0, sigma, 4)
        x1, y1, x2, y2 = box.x1 + d[0], box.y1 + d[1], box.x2 + d[2], box.y2 + d[3]
        return BBox(float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2)))

    def _rim(self, rim: RimObservation, rng: np.random.Generator) -> RimObservation:
        box = self._jitter_box(rim.bbox, self.config.rim_px_sigma, rng)
        assert box is not None
        return RimObservation(side=rim.side, bbox=box, confidence=rim.confidence)

    # ------------------------------------------------------------------ ball

    def _held(self, f: FrameState) -> bool:
        b, c = f.ball, self.config
        if b is None or b.court_xy is None or b.z is None or b.z > c.held_max_z_ft:
            return False
        bx, by = b.court_xy
        return any(math.hypot(p.x - bx, p.y - by) <= c.held_radius_ft for p in f.players)

    def _ball(self, f: FrameState, rng: np.random.Generator) -> BallState | None:
        b, c = f.ball, self.config
        if b is None:
            return None
        if rng.random() < c.ball_fp_rate:
            return self._false_positive(f, rng)
        if not b.visible or b.image_xy is None:
            return BallState(visible=False)
        miss = c.ball_miss_rate_held if self._held(f) else c.ball_miss_rate
        if rng.random() < miss:
            return BallState(visible=False)
        du, dv = rng.normal(0.0, c.ball_px_sigma, 2)
        uv = (b.image_xy[0] + float(du), b.image_xy[1] + float(dv))
        radius = None if b.radius_px is None else max(1.0, b.radius_px + float(rng.normal(0.0, c.ball_radius_sigma_px)))
        conf = float(rng.uniform(0.5, 0.95))
        if c.monocular:
            return BallState(visible=True, image_xy=uv, court_xy=self._floor_xy(uv), z=None, radius_px=radius,
                             confidence=conf)
        court_xy = None
        if b.court_xy is not None:
            ex, ey = rng.normal(0.0, c.pos_sigma_ft, 2)
            court_xy = (b.court_xy[0] + float(ex), b.court_xy[1] + float(ey))
        z = None if b.z is None else max(0.0, b.z + float(rng.normal(0.0, 0.2)))
        return BallState(visible=True, image_xy=uv, court_xy=court_xy, z=z, radius_px=radius, confidence=conf)

    def _false_positive(self, f: FrameState, rng: np.random.Generator) -> BallState:
        w, h = f.image_size or (self.camera.image_size if self.camera else (1280, 720))
        uv = (float(rng.uniform(0, w)), float(rng.uniform(0.3 * h, h)))
        court_xy = self._floor_xy(uv) if self.config.monocular else None
        return BallState(visible=True, image_xy=uv, court_xy=court_xy, z=None,
                         radius_px=float(rng.uniform(2.5, 5.0)), confidence=float(rng.uniform(0.3, 0.6)))

    def _floor_xy(self, uv: tuple[float, float]) -> tuple[float, float] | None:
        if self._floor is None:
            return None
        xy = self._floor.image_to_court(np.array(uv, dtype=float), front_only=True)
        if not np.all(np.isfinite(xy)):
            return None
        return float(xy[0]), float(xy[1])


# ---------------------------------------------------------------------------- camera recovery


def floor_homography_from_frames(frames: list[FrameState], max_points: int = 500,
                                 max_rms_px: float = 0.5) -> Homography | None:
    """Image -> floor homography recovered from 3-D ball truth in ``frames`` (None if unavailable).

    Fits the 3x4 projection ``P`` by normalised DLT (Hartley & Zisserman §7.1) to visible balls with
    known ``court_xy`` and ``z``; the floor homography is ``inv(P[:, [0, 1, 3]])``, oriented so that
    points in front of the camera have ``w > 0`` (needed by ``front_only``). Needs >= 6 points that
    are not coplanar (ball heights must vary) and a reprojection RMS <= ``max_rms_px``.
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
    with np.errstate(divide="ignore", invalid="ignore"):
        rms = float(np.sqrt(np.mean(np.sum((proj[:, :2] / proj[:, 2:3] - data[:, 3:]) ** 2, axis=1))))
    if not rms <= max_rms_px:
        log.warning("camera recovery from frames failed: reprojection RMS %.3f px", rms)
        return None
    G = P[:, [0, 1, 3]]  # court floor (x, y, 1) -> image
    try:
        H = np.linalg.inv(G)
        return Homography(H / np.linalg.norm(H))
    except (np.linalg.LinAlgError, ValueError):
        return None


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
