"""SyntheticRenderer: palette colours, figure geometry, scoreboard digits, speed."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np
import pytest

from courtvision.sim import palette
from courtvision.sim.render import (
    PANEL_BG_BGR,
    SCOREBOARD_ROI,
    SEGMENT_ON_BGR,
    SEGMENTS,
    SyntheticRenderer,
    scoreboard_digits,
    scoreboard_roi,
    segment_rects,
)
from courtvision.sim.simulator import GameSimulator, SimConfig, SimResult, cylinder_boxes
from courtvision.types import BBox, FrameState, PlayerState, ScoreboardReading, Team


@pytest.fixture(scope="module")
def sim() -> SimResult:
    return GameSimulator(SimConfig(seconds=30.0, fps=30.0, seed=21)).run()


@pytest.fixture(scope="module")
def renderer(sim: SimResult) -> SyntheticRenderer:
    return SyntheticRenderer(sim.camera, sim.court)


def _px(img: np.ndarray, u: float, v: float) -> tuple[int, ...]:
    return tuple(int(c) for c in img[int(round(v)), int(round(u))])


def _isolated(frame: FrameState, sim: SimResult | None = None, idx: int | None = None) -> list[PlayerState]:
    """Players whose box overlaps no other player's box (no occlusion), nor a referee's figure when the
    simulation is given (referees are rendered too)."""
    refs: list[BBox] = []
    if sim is not None and idx is not None and sim.referees:
        xy = np.array(sim.referees[idx], dtype=float)
        boxes = cylinder_boxes(sim.camera, xy, sim.config.referee_height_ft, sim.config.player_radius_ft)
        refs = [BBox(*map(float, b)) for b in boxes]
    out = []
    for p in frame.players:
        if all(p.bbox.iou(q.bbox) == 0.0 for q in frame.players if q is not p) and \
                all(p.bbox.iou(r) == 0.0 for r in refs):
            out.append(p)
    return out


def test_jersey_colours_and_heads(sim: SimResult, renderer: SyntheticRenderer) -> None:
    checked = 0
    for idx in (0, 50, 100, 200, 450, 650, 800):
        f = sim.frames[idx]
        img = renderer.render(f, sim.referees[idx])
        for p in _isolated(f, sim, idx):
            b = p.bbox
            cx = (b.x1 + b.x2) / 2.0
            want = palette.HOME_BGR if p.team is Team.HOME else palette.AWAY_BGR
            assert _px(img, cx, b.y1 + 0.85 * b.h) == want  # shorts, below the number
            assert _px(img, b.x1 + 1, b.y1 + 0.85 * b.h) == want
            assert _px(img, cx, b.y1 + 0.04 * b.h + 1) == palette.SKIN_BGR
            checked += 1
    assert checked > 10


def test_figure_fills_its_bbox_exactly(sim: SimResult) -> None:
    r = SyntheticRenderer(sim.camera, sim.court)
    f = sim.frames[0]
    p = f.players[0]
    lone = FrameState(frame_idx=0, t=0.0, players=[p], ball=None, rims=[], scoreboard=None)
    img = r.render(lone)
    bg = r.render(FrameState(frame_idx=0, t=0.0, players=[], ball=None, rims=[], scoreboard=None))
    figure = np.all(img == palette.HOME_BGR, axis=2) | np.all(img == palette.SKIN_BGR, axis=2)
    ys, xs = np.nonzero(figure & np.any(img != bg, axis=2))
    b = p.bbox
    assert p.team is Team.HOME
    assert abs(xs.min() - round(b.x1)) <= 1 and abs(xs.max() + 1 - round(b.x2)) <= 1
    assert abs(ys.min() - round(b.y1)) <= 1 and abs(ys.max() + 1 - round(b.y2)) <= 1


def test_jersey_number_drawn_in_white(sim: SimResult) -> None:
    r = SyntheticRenderer(sim.camera, sim.court)
    p = sim.frames[0].players[0]
    img = r.render(FrameState(frame_idx=0, t=0.0, players=[p], ball=None, rims=[], scoreboard=None))
    b = p.bbox
    crop = img[int(b.y1):int(b.y2), int(b.x1):int(b.x2)]
    white = np.all(crop == (255, 255, 255), axis=2)
    assert white.sum() > 5
    ys = np.nonzero(white)[0] + int(b.y1)
    assert abs(ys.mean() - (b.y1 + 0.4 * b.h)) < 0.15 * b.h


def test_referees_ball_and_rims(sim: SimResult, renderer: SyntheticRenderer) -> None:
    idx = next(i for i, f in enumerate(sim.frames) if f.ball.visible and sim.holders[i] is None)
    f = sim.frames[idx]
    img = renderer.render(f, sim.referees[idx])
    u, v = f.ball.image_xy
    assert _px(img, u, v) == palette.BALL_BGR
    refs = sim.referees[idx]
    x, y = refs[0]
    box = sim.camera.bbox_of_cylinder(x, y, 6.2, 1.0)
    assert _px(img, (box.x1 + box.x2) / 2, box.y1 + 0.7 * box.h) == palette.REF_BGR
    for rim in f.rims:  # the ring is drawn in red on the rim box outline
        crop = img[int(rim.bbox.y1) - 1:int(rim.bbox.y2) + 2, int(rim.bbox.x1) - 1:int(rim.bbox.x2) + 2]
        assert np.all(crop == palette.RIM_BGR, axis=2).sum() > 10


def test_court_lines_and_floor(sim: SimResult, renderer: SyntheticRenderer) -> None:
    img = renderer.render(FrameState(frame_idx=0, t=0.0, players=[], ball=None, rims=[], scoreboard=None))
    court = sim.court
    mid = sim.camera.project(np.array([court.length / 2.0, court.width / 2.0 - 3.0, 0.0]))
    assert _px(img, *mid) == palette.LINE_BGR  # division line
    paint = sim.camera.project(np.array([court.basket_offset + 6.0, court.width / 2.0 + 2.0, 0.0]))
    assert _px(img, *paint) == palette.PAINT_BGR
    floor = sim.camera.project(np.array([30.0, 10.0, 0.0]))
    assert _px(img, *floor) == palette.FLOOR_BGR


def _read_panel(img: np.ndarray, roi: dict) -> dict[str, list[str]]:
    """Independent reader: rectify per the convention and test each segment's centre pixel."""
    (w, h) = roi["panel_size"]
    src = np.array(roi["corners"], dtype=np.float32)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    panel = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), flags=cv2.INTER_NEAREST)
    lookup = {frozenset(v): k for k, v in SEGMENTS.items()}
    out: dict[str, list[str]] = {}
    for name, cells in roi["cells"].items():
        out[name] = []
        for cell in cells:
            lit = set()
            for seg, (x1, y1, x2, y2) in segment_rects(cell).items():
                px = panel[(y1 + y2) // 2, (x1 + x2) // 2]
                if tuple(int(c) for c in px) == SEGMENT_ON_BGR:
                    lit.add(seg)
            out[name].append(lookup[frozenset(lit)])
    return out


def test_scoreboard_digits_round_trip(sim: SimResult, renderer: SyntheticRenderer) -> None:
    readings = [
        ScoreboardReading(t=0, home_score=7, away_score=102, period=3, game_clock=605.2, shot_clock=8.1, source="sim"),
        ScoreboardReading(t=0, home_score=0, away_score=58, period=1, game_clock=59.0, shot_clock=None, source="sim"),
        ScoreboardReading(t=0, home_score=99, away_score=9, period=4, game_clock=0.0, shot_clock=24.0, source="sim"),
    ]
    for rd in readings:
        f = FrameState(frame_idx=0, t=0.0, players=[], ball=None, rims=[], scoreboard=rd)
        got = _read_panel(renderer.render(f), renderer.roi)
        assert got == scoreboard_digits(rd)
    assert scoreboard_digits(readings[0]) == {"home": ["", "", "7"], "away": ["1", "0", "2"], "period": ["3"],
                                              "clock": ["1", "0", "0", "6"], "shot": ["", "9"]}
    assert scoreboard_digits(readings[1])["clock"] == ["", "0", "5", "9"]
    assert scoreboard_digits(readings[1])["shot"] == ["", ""]
    sim_frame = sim.frames[300]
    assert _read_panel(renderer.render(sim_frame), renderer.roi) == scoreboard_digits(sim_frame.scoreboard)


def test_scoreboard_panel_convention(renderer: SyntheticRenderer) -> None:
    assert SCOREBOARD_ROI == scoreboard_roi((1280, 720))
    roi = SCOREBOARD_ROI
    assert len(roi["corners"]) == 4 and [len(roi["cells"][k]) for k in ("home", "away", "period", "clock", "shot")] == \
        [3, 3, 1, 4, 2]
    img = renderer.render(FrameState(frame_idx=0, t=0.0, players=[], ball=None, rims=[], scoreboard=None))
    (x0, y0), _, (x1, y1), _ = roi["corners"]
    assert np.all(img[y0:y1, x0:x1] == PANEL_BG_BGR)  # blank board: background only, unlit segments not drawn
    r = segment_rects([0, 0, 25, 44])
    t, m = round(0.16 * 25), round(0.12 * 25)
    assert r["a"] == (m, m, 25 - m, m + t) and r["d"][3] == 44 - m
    assert r["b"][3] == r["c"][1] and r["f"][3] == r["e"][1]  # vertical bars split the inner height


def test_render_speed(sim: SimResult, renderer: SyntheticRenderer) -> None:
    frames = sim.frames[:60]
    t0 = time.perf_counter()
    for i, f in enumerate(frames):
        img = renderer.render(f, sim.referees[i])
    fps = len(frames) / (time.perf_counter() - t0)
    assert img.shape == (720, 1280, 3) and img.dtype == np.uint8
    assert fps >= 40.0, fps


def test_noisy_frames_render(sim: SimResult, renderer: SyntheticRenderer) -> None:
    f = sim.frames[10]
    odd = FrameState(frame_idx=0, t=0.0, players=[PlayerState("x", None, 1.0, 1.0, bbox=BBox(math.nan, 0, 1, 1))],
                     ball=None, rims=[], scoreboard=f.scoreboard)
    assert renderer.render(odd).shape == (720, 1280, 3)


# ---------------------------------------------------------------------------- ball occlusion & blur


def _scene(sim: SimResult, player_xy: tuple[float, float], ball_xyz: tuple[float, float, float],
           idx: int = 0) -> FrameState:
    """One player and the ball, at chosen court points (the rest of a real frame dropped)."""
    import dataclasses

    from courtvision.types import BallState

    cam = sim.camera
    x, y = player_xy
    box = cam.bbox_of_cylinder(x, y, 6.5, 1.0)
    p = PlayerState(player_id="home-1", team=Team.HOME, x=x, y=y, bbox=box, jersey="1")
    uv = cam.project(np.array(ball_xyz, dtype=float))
    bb = cam.bbox_of_sphere(*ball_xyz, 0.39)
    ball = BallState(visible=True, image_xy=(float(uv[0]), float(uv[1])), court_xy=ball_xyz[:2], z=ball_xyz[2],
                     radius_px=bb.w / 2.0)
    return dataclasses.replace(sim.frames[idx], frame_idx=10_000 + idx, players=[p], ball=ball)


def test_a_nearer_body_hides_the_ball(sim: SimResult) -> None:
    """RESEARCH §2.1: the ball drawn in depth order, so a player between it and the camera covers it;
    the same ball in front of him is drawn (it used to be drawn on top of everything)."""
    r = SyntheticRenderer(sim.camera, sim.court)
    behind = r.render(_scene(sim, (47.0, 20.0), (47.0, 23.0, 3.5)))  # the camera looks along +y
    front = r.render(_scene(sim, (47.0, 26.0), (47.0, 23.0, 3.5)))
    u, v = sim.camera.project(np.array([47.0, 23.0, 3.5]))
    assert _px(behind, u, v) != palette.BALL_BGR
    assert _px(front, u, v) == palette.BALL_BGR
    from courtvision.sim.render import RenderConfig

    old = SyntheticRenderer(sim.camera, sim.court, RenderConfig(ball_depth_order=False))
    assert _px(old.render(_scene(sim, (47.0, 20.0), (47.0, 23.0, 3.5))), u, v) == palette.BALL_BGR


def test_a_held_ball_is_partly_covered_by_the_hand(sim: SimResult) -> None:
    from courtvision.sim.render import RenderConfig

    f = _scene(sim, (47.0, 23.0), (47.0, 21.6, 3.75))  # held 1.4 ft in front of the holder
    plain = SyntheticRenderer(sim.camera, sim.court).render(f)
    hand = SyntheticRenderer(sim.camera, sim.court, RenderConfig(ball_hand_cover=0.5)).render(f)
    n_plain = int(np.all(plain == palette.BALL_BGR, axis=2).sum())
    n_hand = int(np.all(hand == palette.BALL_BGR, axis=2).sum())
    assert n_plain > 10 and 0 < n_hand < 0.8 * n_plain


def test_a_fast_ball_is_motion_blurred(sim: SimResult) -> None:
    """cv_pipeline §6.1: at pass / shot speeds the ball smears along its image motion (shutter 1/60 s)."""
    r = SyntheticRenderer(sim.camera, sim.court)
    a = _scene(sim, (20.0, 10.0), (40.0, 25.0, 6.0), idx=0)
    b = _scene(sim, (20.0, 10.0), (41.5, 25.0, 6.0), idx=1)  # 1.5 ft in 1/30 s = 45 ft/s
    r.render(a)
    img = r.render(b)
    mask = np.all(img == palette.BALL_BGR, axis=2)
    ys, xs = np.nonzero(mask)
    still = SyntheticRenderer(sim.camera, sim.court).render(b)  # no previous frame: no blur
    ys0, xs0 = np.nonzero(np.all(still == palette.BALL_BGR, axis=2))
    assert (xs.max() - xs.min()) > (xs0.max() - xs0.min()) + 4  # longer along the motion (+x in the image)
    assert abs(xs.mean() - xs0.mean()) < 1.5  # centred on the ball: the centroid stays put


def test_waiting_substitutes_are_drawn_off_the_court(sim: SimResult, renderer: SyntheticRenderer) -> None:
    f = sim.frames[0]
    img = renderer.render(f, extras=[(Team.AWAY, 51.0, -2.5, "7")])
    box = sim.camera.bbox_of_cylinder(51.0, -2.5, 6.5, 1.0)
    assert _px(img, (box.x1 + box.x2) / 2, box.y1 + 0.85 * box.h) == palette.AWAY_BGR
