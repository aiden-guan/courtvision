"""Scenario kit for the event-engine tests.

``Scene`` builds hand-made 3-D plays (players on the floor, a ball trajectory) frame by frame and
renders them two ways:

* ``frames_3d()``: truth-like frames (ball ``court_xy`` + ``z``; image data via the camera too),
* ``frames_mono()``: monocular frames, ``PinholeCamera``-projected bboxes / ball / rims, no ball
  ``z``, ball ``court_xy`` = floor-homography projection of the image point, players located by
  the foot-point homography; optional seeded dropouts, false detections and pixel noise.

The tests in this file only check the kit itself.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from courtvision.events.engine import EventEngine, EventEngineConfig
from courtvision.geometry.camera import RECOMMENDED_SIM_CAMERA, PinholeCamera
from courtvision.geometry.court import court_for
from courtvision.types import (
    BallState,
    Event,
    EventKind,
    EventStatus,
    FrameState,
    PlayerState,
    RimObservation,
    ScoreboardReading,
    Team,
)

HANDS_Z = 4.0
PLAYER_H = 6.7
PLAYER_R = 0.9
BALL_R = 0.39
MODES = ("3d", "mono")

Vec3 = tuple[float, float, float]
BallFn = Callable[[float, dict[str, tuple[float, float]]], Vec3 | None]


def camera() -> PinholeCamera:
    c = RECOMMENDED_SIM_CAMERA
    return PinholeCamera.look_at(c["camera_position"], c["camera_target"], c["image_size"], c["hfov_deg"])


_CAM = camera()
_H = _CAM.floor_homography()


@dataclass
class Rec:
    t: float
    pos: dict[str, tuple[float, float]]
    ball: Vec3 | None
    visible: bool
    score: tuple[int, int]
    clock: float | None
    shot_clock: float | None
    period: int


def _lerp(a: Vec3, b: Vec3, s: float) -> Vec3:
    return (a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]), a[2] + s * (b[2] - a[2]))


class Scene:
    """Frame-by-frame play builder (court feet, z up)."""

    def __init__(self, league: str = "nba", fps: float = 30.0, period: int = 1, clock: float | None = 600.0,
                 shot_clock: float | None = None, scoreboard: bool = True, score: tuple[int, int] = (0, 0)) -> None:
        self.court = court_for(league)
        self.fps = fps
        self.pos: dict[str, tuple[float, float]] = {}
        self.recs: list[Rec] = []
        self.period = period
        self.clock = clock
        self.clock_running = True
        self.shot_clock = shot_clock
        self.scoreboard = scoreboard
        self.score = score
        self.ball: Vec3 | None = None
        self.marks: dict[str, float] = {}
        self.dribble_times: list[float] = []

    # ------------------------------------------------------------------ building

    @property
    def t(self) -> float:
        return (len(self.recs) - 1) / self.fps if self.recs else 0.0

    def add(self, pid: str, x: float, y: float) -> Scene:
        self.pos[pid] = (x, y)
        return self

    def mark(self, name: str) -> float:
        self.marks[name] = self.t
        return self.t

    def hands(self, pid: str, pos: dict[str, tuple[float, float]] | None = None) -> Vec3:
        x, y = (pos or self.pos)[pid]
        return (x, y - 0.7, HANDS_Z)

    def run(self, duration: float, ball: BallFn | None, moves: dict[str, tuple[float, float]] | None = None,
            visible: bool = True) -> Scene:
        """Advance ``duration`` s; ``ball(a, pos)`` gives the ball at fraction ``a`` in (0, 1]."""
        n = max(1, round(duration * self.fps))
        start = dict(self.pos)
        for i in range(1, n + 1):
            a = i / n
            for pid, (tx, ty) in (moves or {}).items():
                sx, sy = start[pid]
                self.pos[pid] = (sx + a * (tx - sx), sy + a * (ty - sy))
            self.ball = ball(a, self.pos) if ball is not None else self.ball
            self._record(visible)
        return self

    def _record(self, visible: bool) -> None:
        if self.recs and self.clock is not None and self.clock_running:
            self.clock = max(0.0, self.clock - 1.0 / self.fps)
        if self.recs and self.shot_clock is not None and self.clock_running:
            self.shot_clock = max(0.0, self.shot_clock - 1.0 / self.fps)
        self.recs.append(Rec(len(self.recs) / self.fps, dict(self.pos), self.ball, visible, self.score,
                             self.clock, self.shot_clock, self.period))

    def start(self, holder: str | None = None, ball: Vec3 | None = None) -> Scene:
        self.ball = self.hands(holder) if holder else ball
        self._record(True)
        return self

    def hold(self, pid: str, duration: float, move: tuple[float, float] | None = None, visible: bool = True) -> Scene:
        return self.run(duration, lambda a, pos: self.hands(pid, pos), {pid: move} if move else None, visible)

    def dribble(self, pid: str, duration: float, bounces: int, move: tuple[float, float] | None = None) -> Scene:
        t0 = self.t
        self.dribble_times += [t0 + (k + 0.5) / bounces * duration for k in range(bounces)]

        def ball(a: float, pos: dict[str, tuple[float, float]]) -> Vec3:
            x, y = pos[pid]
            return (x, y - 1.0, BALL_R + (HANDS_Z - BALL_R) * abs(math.cos(math.pi * bounces * a)))

        return self.run(duration, ball, {pid: move} if move else None)

    def fly(self, end: Vec3 | Callable[[dict[str, tuple[float, float]]], Vec3], duration: float, apex: float = 1.5,
            moves: dict[str, tuple[float, float]] | None = None) -> Scene:
        """Parabolic flight from the current ball position to ``end`` (a point or f(pos))."""
        start = self.ball
        assert start is not None

        def ball(a: float, pos: dict[str, tuple[float, float]]) -> Vec3:
            e = end(pos) if callable(end) else end
            p = _lerp(start, e, a)
            return (p[0], p[1], p[2] + 4.0 * apex * a * (1.0 - a))

        return self.run(duration, ball, moves)

    def pass_to(self, receiver: str, duration: float | None = None, apex: float = 1.5,
                moves: dict[str, tuple[float, float]] | None = None) -> Scene:
        assert self.ball is not None
        if duration is None:
            rx, ry, _ = self.hands(receiver)
            duration = max(0.3, math.dist(self.ball[:2], (rx, ry)) / 30.0)
        return self.fly(lambda pos: self.hands(receiver, pos), duration, apex, moves)

    def basket(self, side: str, z: float | None = None) -> Vec3:
        bx, by = self.court.basket_xy(side)  # type: ignore[arg-type]
        return (bx, by, self.court.rim_height if z is None else z)

    def shoot(self, side: str, made: bool = True, duration: float | None = None, apex_z: float = 15.0,
              release_z: float = 7.5, rim_offset: float = 0.95) -> Scene:
        """Jump shot from the current ball position to the rim (made) or the front rim (miss)."""
        assert self.ball is not None
        sx, sy, _ = self.ball
        start = (sx, sy, release_z)
        target = self.basket(side)
        if not made:
            dx, dy = sx - target[0], sy - target[1]
            n = math.hypot(dx, dy)
            target = (target[0] + rim_offset * dx / n, target[1] + rim_offset * dy / n, target[2] + 0.25)
        if duration is None:
            duration = 0.5 + math.dist(start[:2], target[:2]) / 25.0
        h = apex_z - (start[2] + target[2]) / 2.0
        self.ball = start

        def ball(a: float, pos: dict[str, tuple[float, float]]) -> Vec3:
            p = _lerp(start, target, a)
            return (p[0], p[1], p[2] + 4.0 * h * a * (1.0 - a))

        return self.run(duration, ball)

    def drop_through(self, side: str, duration: float = 0.4) -> Scene:
        top, floor = self.basket(side), self.basket(side, BALL_R)
        return self.run(duration, lambda a, pos: _lerp(top, floor, a * a))

    def rest(self, duration: float, moves: dict[str, tuple[float, float]] | None = None, visible: bool = True) -> Scene:
        b = self.ball
        return self.run(duration, lambda a, pos: b, moves, visible)

    def set_score(self, home: int, away: int) -> Scene:
        self.score = (home, away)
        return self

    # ------------------------------------------------------------------ rendering

    def _players(self, k: int, mono: bool, rng: random.Random | None, px_noise: float) -> list[PlayerState]:
        rec = self.recs[k]
        prev = self.recs[k - 1] if k > 0 else rec
        out = []
        for pid, (x, y) in rec.pos.items():
            px, py = prev.pos.get(pid, (x, y))
            vx, vy = (x - px) * self.fps, (y - py) * self.fps
            bb = _CAM.bbox_of_cylinder(x, y, PLAYER_H, PLAYER_R)
            if mono and rng is not None and px_noise > 0:
                j = [rng.gauss(0.0, px_noise) for _ in range(4)]
                bb = type(bb)(bb.x1 + j[0], bb.y1 + j[1], bb.x2 + j[2], bb.y2 + j[3])
            if mono:
                fx, fy = _H.image_to_court(np.array(bb.foot))
                x, y = float(fx), float(fy)
            team = Team(pid.split("-", 1)[0])
            out.append(PlayerState(pid, team, x, y, vx, vy, bbox=bb, track_id=None))
        return out

    def _ball(self, rec: Rec, mono: bool) -> BallState:
        if rec.ball is None or not rec.visible:
            return BallState(visible=False)
        x, y, z = rec.ball
        uv = _CAM.project(np.array([x, y, z]))
        sb = _CAM.bbox_of_sphere(x, y, z, BALL_R)
        r_px = sb.w / 2.0
        if mono:
            cxy = _H.image_to_court(uv)
            return BallState(True, (float(uv[0]), float(uv[1])), (float(cxy[0]), float(cxy[1])), None, r_px, 0.9)
        return BallState(True, (float(uv[0]), float(uv[1])), (x, y), z, r_px, 1.0)

    def _rims(self) -> list[RimObservation]:
        out = []
        for side in ("left", "right"):
            bx, by = self.court.basket_xy(side)  # type: ignore[arg-type]
            out.append(RimObservation(side, _CAM.bbox_of_circle(bx, by, self.court.rim_height, self.court.rim_radius)))  # type: ignore[arg-type]
        return out

    def frames(self, mode: str, **kw: float) -> list[FrameState]:
        return self.frames_3d() if mode == "3d" else self.frames_mono(**kw)  # type: ignore[arg-type]

    def frames_3d(self) -> list[FrameState]:
        rims = self._rims()
        return [self._frame(k, False, rims, None, 0.0) for k in range(len(self.recs))]

    def frames_mono(self, seed: int = 0, dropout: float = 0.0, false_rate: float = 0.0, px_noise: float = 0.0) -> list[FrameState]:
        rng = random.Random(seed)
        rims = self._rims()
        out = []
        for k in range(len(self.recs)):
            fr = self._frame(k, True, rims, rng, px_noise)
            b = fr.ball
            if b is not None and b.visible and rng.random() < dropout:
                fr.ball = BallState(visible=False)
            elif rng.random() < false_rate:
                u, v = rng.uniform(50, 1230), rng.uniform(50, 670)
                cxy = _H.image_to_court(np.array([u, v]))
                fr.ball = BallState(True, (u, v), (float(cxy[0]), float(cxy[1])), None, 4.0, 0.6)
            elif b is not None and b.visible and px_noise > 0 and b.image_xy is not None:
                u, v = b.image_xy[0] + rng.gauss(0, px_noise * 0.5), b.image_xy[1] + rng.gauss(0, px_noise * 0.5)
                cxy = _H.image_to_court(np.array([u, v]))
                fr.ball = BallState(True, (u, v), (float(cxy[0]), float(cxy[1])), None, b.radius_px, b.confidence)
            out.append(fr)
        return out

    def _frame(self, k: int, mono: bool, rims: list[RimObservation], rng: random.Random | None, px_noise: float) -> FrameState:
        rec = self.recs[k]
        sb = None
        if self.scoreboard:
            sb = ScoreboardReading(rec.t, rec.score[0], rec.score[1], rec.period, rec.clock, rec.shot_clock, source="sim")
        return FrameState(k, rec.t, self._players(k, mono, rng, px_noise), self._ball(rec, mono), rims=list(rims),
                          scoreboard=sb, image_size=_CAM.image_size)


# ---------------------------------------------------------------------------- engine helpers


def run_engine(frames: list[FrameState], config: EventEngineConfig | None = None, league: str = "nba",
               flush: bool = True, **kw: object) -> EventEngine:
    eng = EventEngine(court_for(league), config=config, **kw)  # type: ignore[arg-type]
    for fr in frames:
        eng.process(fr)
    if flush:
        eng.flush()
    return eng


def live(events: list[Event], kind: EventKind | None = None) -> list[Event]:
    """Latest, non-retracted events (optionally of one kind)."""
    return [e for e in events if e.status is not EventStatus.RETRACTED and (kind is None or e.kind is kind)]


def one(events: list[Event], kind: EventKind) -> Event:
    found = live(events, kind)
    assert len(found) == 1, f"expected one {kind.value}, got {[(e.t, e.player_id) for e in found]}"
    return found[0]


def kinds(events: list[Event]) -> list[str]:
    return [e.kind.value for e in live(events) if e.kind is not EventKind.DRIBBLE]


def two_team_scene(**kw: object) -> Scene:
    """Five-on-five-lite: 3 home and 3 away players spread over the left half."""
    s = Scene(**kw)  # type: ignore[arg-type]
    s.add("home-1", 30.0, 25.0).add("home-2", 22.0, 38.0).add("home-3", 18.0, 10.0)
    s.add("away-1", 26.0, 25.0).add("away-2", 18.0, 34.0).add("away-3", 12.0, 14.0)
    return s


# ---------------------------------------------------------------------------- kit self-tests


def test_scene_hold_projects_ball_inside_holder_bbox() -> None:
    s = two_team_scene().start("home-1").hold("home-1", 0.5)
    fr = s.frames_mono()[-1]
    holder = next(p for p in fr.players if p.player_id == "home-1")
    assert fr.ball is not None and fr.ball.image_xy is not None and holder.bbox is not None
    assert holder.bbox.contains(*fr.ball.image_xy)
    assert fr.ball.z is None and fr.frame_idx == len(s.recs) - 1


def test_scene_made_shot_passes_through_rim_centre() -> None:
    s = two_team_scene().start("home-1").hold("home-1", 0.3).shoot("left", made=True)
    assert s.ball is not None
    bx, by, bz = s.basket("left")
    assert math.dist(s.ball, (bx, by, bz)) < 1e-9
