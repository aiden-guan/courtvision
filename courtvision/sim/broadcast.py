"""Broadcast-style renders of a simulation: a pan / tilt / zoom game camera, replays, cutaways and a scorebug.

A TV broadcast is not the fixed venue camera the rest of the simulator draws (``render.py``):

* **Game camera.** One camera at a fixed position high on the sideline pans and zooms to follow the play
  (``BroadcastConfig``: the aim point follows the ball through a first-order lag with a speed cap; the field
  of view narrows when the play is in a half court). Every frame has its own camera, so the court
  homography changes frame to frame (``perception/court_tracker.py`` tracks it).
* **Replays and cutaways** are scheduled only inside dead-ball windows (the game clock stopped for at least
  ``min_dead_s``), so no live action is hidden by them: a replay re-shows the last ``replay_s`` of the play
  before the stoppage from a baseline camera, without the scorebug; a cutaway is a close-up (no court).
  Broadcast perception must not emit events from either (``perception/segments.py`` gates them).
* **Scorebug.** A graphic at the bottom left on game-camera frames: team abbreviations, scores, period,
  game clock (M:SS, tenths in the last minute as SS.t) and shot clock, white Hershey text on a dark box
  (``scorebug_layout`` gives the field rectangles, ``perception/scorebug.py`` reads them).

``BroadcastPlan`` holds the per-frame decisions (kind, source frame, camera); ``BroadcastRenderer`` draws
them. Camera motion, replay and cutaway values are design choices, not measurements of a real broadcast.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from courtvision.geometry.camera import PinholeCamera
from courtvision.sim.render import STANDS_BGR, RenderConfig, SyntheticRenderer
from courtvision.types import BallState, FrameState, ScoreboardReading

__all__ = [
    "BUG_BG_BGR",
    "BUG_TEXT_BGR",
    "BroadcastConfig",
    "BroadcastFrame",
    "BroadcastPlan",
    "BroadcastRenderer",
    "plan_broadcast",
    "reproject_frame",
    "scorebug_layout",
    "scorebug_text",
]

BUG_BG_BGR = (60, 30, 20)
BUG_TEXT_BGR = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class BroadcastConfig:
    """Broadcast direction for a simulated game (all design choices, unverified)."""

    camera_position: tuple[float, float, float] = (47.0, -52.0, 42.0)  # high on the sideline at half court
    aim_y_ft: float = 22.0  # the aim point's court y (a little toward the near sideline of the centre)
    aim_tau_s: float = 0.5  # first-order lag of the aim point behind the ball
    aim_max_speed_fts: float = 30.0  # pan speed cap (aim point, court ft / s)
    hfov_wide_deg: float = 60.0  # field of view with the play near half court ...
    hfov_tight_deg: float = 50.0  # ... and with the play in a half court
    zoom_tau_s: float = 1.0
    image_size: tuple[int, int] = (1280, 720)
    # replays / cutaways inside dead-ball windows
    min_dead_s: float = 4.0  # a stoppage at least this long can hold one
    replay_share: float = 0.5  # of such stoppages show a replay ...
    cutaway_share: float = 0.3  # ... or a cutaway (the rest stay on the game camera)
    replay_s: float = 3.0  # the replay shows the last 3 s of play before the stoppage
    lead_in_s: float = 1.0  # starting this long into the stoppage
    cutaway_s: float = 2.0
    replay_camera_position: tuple[float, float, float] = (-14.0, 25.0, 22.0)  # behind a baseline
    replay_hfov_deg: float = 50.0
    scorebug: bool = True
    seed: int = 0


@dataclass(frozen=True)
class BroadcastFrame:
    """One output frame: what is shown (``kind`` "live" | "replay" | "cutaway"), which simulation frame is
    drawn (``src``; for a replay an earlier one), the camera (None for a cutaway) and the scorebug."""

    kind: str
    src: int
    camera: PinholeCamera | None
    scorebug: bool


@dataclass
class BroadcastPlan:
    frames: list[BroadcastFrame]
    segments: list[tuple[str, int, int]] = field(default_factory=list)  # (kind, first, last) output frames

    def kind(self, i: int) -> str:
        return self.frames[i].kind

    def live_mask(self) -> list[bool]:
        return [f.kind == "live" for f in self.frames]


def _stopped_windows(frames: list[FrameState], fps: float, min_s: float) -> list[tuple[int, int]]:
    """Frame ranges where the truth game clock does not move for at least ``min_s``."""
    out, start, last = [], None, None
    for i, f in enumerate(frames):
        gc = f.scoreboard.game_clock if f.scoreboard is not None else None
        if gc is not None and last is not None and abs(gc - last) < 1e-9:
            if start is None:
                start = i - 1
        else:
            if start is not None and (i - start) / fps >= min_s:
                out.append((start, i - 1))
            start = None
        last = gc
    if start is not None and (len(frames) - start) / fps >= min_s:
        out.append((start, len(frames) - 1))
    return out


def _game_cameras(frames: list[FrameState], fps: float, cfg: BroadcastConfig, length: float) -> list[PinholeCamera]:
    dt = 1.0 / fps
    aim_x, hfov = length / 2.0, cfg.hfov_wide_deg
    cams = []
    for f in frames:
        b = f.ball
        bx = b.court_xy[0] if b is not None and b.court_xy is not None else aim_x
        step = (bx - aim_x) * (1.0 - math.exp(-dt / cfg.aim_tau_s))
        cap = cfg.aim_max_speed_fts * dt
        aim_x += max(-cap, min(cap, step))
        off = min(1.0, abs(aim_x - length / 2.0) / (length / 4.0))  # 0 at half court, 1 a quarter away
        target = cfg.hfov_wide_deg + (cfg.hfov_tight_deg - cfg.hfov_wide_deg) * off
        hfov += (target - hfov) * (1.0 - math.exp(-dt / cfg.zoom_tau_s))
        cams.append(PinholeCamera.look_at(cfg.camera_position, (aim_x, cfg.aim_y_ft, 0.0), cfg.image_size, hfov))
    return cams


def plan_broadcast(frames: list[FrameState], fps: float, length: float, config: BroadcastConfig | None = None) -> BroadcastPlan:
    """Per-frame broadcast decisions for simulation ``frames`` (court ``length`` ft)."""
    cfg = config or BroadcastConfig()
    rng = np.random.default_rng(cfg.seed)
    cams = _game_cameras(frames, fps, cfg, length)
    out = [BroadcastFrame("live", i, cams[i], cfg.scorebug) for i in range(len(frames))]
    segments: list[tuple[str, int, int]] = []
    for a, b in _stopped_windows(frames, fps, cfg.min_dead_s):
        u = rng.random()
        start = a + int(round(cfg.lead_in_s * fps))
        if u < cfg.replay_share:
            n = int(round(cfg.replay_s * fps))
            src0 = max(0, a - n)
            n = min(n, a - src0, b - start - int(round(0.5 * fps)))
            if n < int(fps):
                continue
            # the replay camera looks at the end of the court the play was at
            bx = np.mean([f.ball.court_xy[0] for f in frames[src0:a] if f.ball is not None and f.ball.court_xy])
            right = bool(bx > length / 2.0)
            pos = list(cfg.replay_camera_position)
            if right:
                pos[0] = length - pos[0]
            target = (length - 20.0 if right else 20.0, 25.0, 0.0)
            cam = PinholeCamera.look_at(pos, target, cfg.image_size, cfg.replay_hfov_deg)
            for k in range(n):
                out[start + k] = BroadcastFrame("replay", src0 + k, cam, False)
            segments.append(("replay", start, start + n - 1))
        elif u < cfg.replay_share + cfg.cutaway_share:
            n = min(int(round(cfg.cutaway_s * fps)), b - start - int(round(0.5 * fps)))
            if n < int(0.5 * fps):
                continue
            for k in range(n):
                out[start + k] = BroadcastFrame("cutaway", start + k, None, cfg.scorebug)
            segments.append(("cutaway", start, start + n - 1))
    return BroadcastPlan(out, segments)


def reproject_frame(frame: FrameState, camera: PinholeCamera, feet_z: list[float] | None = None,
                    player_height_ft: float = 6.5, player_radius_ft: float = 1.0) -> FrameState:
    """``frame`` seen through ``camera``: player boxes (upright cylinders, lifted by ``feet_z``) and the ball's
    image point / radius are recomputed; positions are unchanged (truth)."""
    players = []
    for k, p in enumerate(frame.players):
        lift = feet_z[k] if feet_z is not None and k < len(feet_z) else 0.0
        box = camera.bbox_of_cylinder(p.x, p.y, player_height_ft, player_radius_ft)
        if lift > 0.0 and all(math.isfinite(v) for v in (box.x1, box.y1, box.x2, box.y2)):
            top = camera.bbox_of_cylinder(p.x, p.y, player_height_ft + lift, player_radius_ft)
            box = type(box)(box.x1, top.y1, box.x2, top.y1 + (box.y2 - box.y1))
        players.append(dataclasses.replace(p, bbox=box))
    ball = frame.ball
    if ball is not None and ball.court_xy is not None and ball.z is not None:
        xyz = np.array([ball.court_xy[0], ball.court_xy[1], ball.z], dtype=float)
        uv = camera.project(xyz)
        sb = camera.bbox_of_sphere(xyz[0], xyz[1], xyz[2], 0.39)
        seen = bool(np.all(np.isfinite(uv)) and camera.in_image(uv)[0])
        ball = BallState(visible=seen and ball.visible, image_xy=(float(uv[0]), float(uv[1])) if seen else None,
                         court_xy=ball.court_xy, z=ball.z, radius_px=max(1.0, sb.w / 2.0) if seen else None,
                         confidence=ball.confidence)
    return FrameState(frame.frame_idx, frame.t, players, ball, [], frame.scoreboard, camera.image_size)


# --------------------------------------------------------------------------- scorebug


def scorebug_layout(image_size: tuple[int, int] = (1280, 720)) -> dict[str, Any]:
    """The scorebug box and its text fields (image px ``[x, y, w, h]``), JSON-serialisable."""
    w, h = image_size
    s = h / 720.0
    bx, by, bw, bh = int(round(24 * s)), int(round(h - 76 * s)), int(round(560 * s)), int(round(52 * s))
    cell = int(round(36 * s))
    y = by + (bh - cell) // 2

    def rect(x: float, width: float) -> list[int]:
        return [bx + int(round(x * s)), y, int(round(width * s)), cell]

    return {"box": [bx, by, bw, bh], "scale": s,
            "fields": {"away_abbr": rect(10, 70), "away_score": rect(84, 70), "home_abbr": rect(166, 70),
                       "home_score": rect(240, 70), "period": rect(330, 46), "game_clock": rect(382, 98),
                       "shot_clock": rect(494, 56)}}


def _fmt_clock(s: float | None) -> str:
    if s is None:
        return ""
    if s < 60.0:
        tenths = math.floor(max(0.0, s) * 10.0 + 1e-9) / 10.0
        return f"{tenths:.1f}"
    total = int(math.ceil(s - 1e-9))
    return f"{total // 60}:{total % 60:02d}"


def scorebug_text(reading: ScoreboardReading | None, abbr: tuple[str, str] = ("AWY", "HOM")) -> dict[str, str]:
    """Text shown in each scorebug field (periods past the 4th: "OT", "2OT", ...)."""
    r = reading
    period = ""
    if r is not None and r.period is not None:
        period = f"Q{r.period}" if r.period <= 4 else ("OT" if r.period == 5 else f"{r.period - 4}OT")
    shot = "" if r is None or r.shot_clock is None else str(int(math.ceil(r.shot_clock - 1e-9)))
    return {"away_abbr": abbr[0], "away_score": "" if r is None or r.away_score is None else str(r.away_score),
            "home_abbr": abbr[1], "home_score": "" if r is None or r.home_score is None else str(r.home_score),
            "period": period, "game_clock": _fmt_clock(None if r is None else r.game_clock), "shot_clock": shot}


def draw_scorebug(img: np.ndarray, reading: ScoreboardReading | None, abbr: tuple[str, str] = ("AWY", "HOM")) -> None:
    lay = scorebug_layout((img.shape[1], img.shape[0]))
    x, y, w, h = lay["box"]
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), BUG_BG_BGR, -1)
    scale = 0.9 * lay["scale"]
    thick = max(1, int(round(2 * lay["scale"])))
    for name, text in scorebug_text(reading, abbr).items():
        if not text:
            continue
        fx, fy, fw, fh = lay["fields"][name]
        (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thick)
        org = (fx + max(0, (fw - tw) // 2), fy + (fh + th) // 2)
        cv2.putText(img, text, org, _FONT, scale, BUG_TEXT_BGR, thick, lineType=cv2.LINE_AA)


# --------------------------------------------------------------------------- renderer


class BroadcastRenderer:
    """Draws a ``BroadcastPlan`` of simulation ``frames`` (``court``; ``feet_z`` / ``referees`` per frame)."""

    def __init__(self, frames: list[FrameState], court: Any, plan: BroadcastPlan,
                 feet_z: list[list[float]] | None = None, referees: list[list[tuple[float, float]]] | None = None,
                 render_config: RenderConfig | None = None, abbr: tuple[str, str] = ("AWY", "HOM"), seed: int = 0) -> None:
        self.frames, self.court, self.plan = frames, court, plan
        self.feet_z, self.referees, self.abbr = feet_z, referees, abbr
        base = render_config or RenderConfig()
        self.render_config = RenderConfig(**{**base.__dict__, "draw_scoreboard": False})
        self._rng = np.random.default_rng(seed)
        self._cut_tint: dict[int, tuple[int, int, int]] = {}

    def render(self, i: int) -> np.ndarray:
        bf = self.plan.frames[i]
        f = self.frames[bf.src]
        live = self.frames[i]  # the scorebug always shows the live board
        if bf.camera is None:
            img = self._cutaway(i, bf.camera)
        else:
            seen = reproject_frame(f, bf.camera, self.feet_z[bf.src] if self.feet_z else None)
            r = SyntheticRenderer(bf.camera, self.court, self.render_config)
            img = r.render(seen, referees=self.referees[bf.src] if self.referees else None)
        if bf.scorebug:
            draw_scorebug(img, live.scoreboard, self.abbr)
        return img

    def _cutaway(self, i: int, _cam: Any) -> np.ndarray:
        w, h = self.plan.frames[0].camera.image_size if self.plan.frames[0].camera else (1280, 720)
        seg = next((s for s in self.plan.segments if s[1] <= i <= s[2]), ("cutaway", i, i))
        tint = self._cut_tint.setdefault(seg[1], tuple(int(v) for v in self._rng.integers(40, 200, 3)))
        img = np.empty((h, w, 3), dtype=np.uint8)
        img[:] = STANDS_BGR
        cv2.ellipse(img, (w // 2, h), (w // 4, int(h * 0.75)), 0, 180, 360, tint, -1)  # a player in close-up
        cv2.circle(img, (w // 2, int(h * 0.3)), h // 7, (150, 170, 200), -1)
        return img
