"""Synthetic scene drawing shared by the perception tests (plus sanity tests of the drawing).

It mimics the ARCHITECTURE render description: palette floor/paint/lines, players as a jersey
rectangle with a skin head circle on top and a white Hershey number (sim/render.py convention),
black referees, an orange ball, red rim rings and a seven-segment scoreboard panel.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from courtvision.geometry.camera import RECOMMENDED_SIM_CAMERA, PinholeCamera
from courtvision.geometry.court import CourtSpec, court_for
from courtvision.perception.scoreboard_reader import PANEL_BGR, draw_digit
from courtvision.sim import palette
from courtvision.types import BBox, Team

PLAYER_HEIGHT_FT = 6.5
PLAYER_RADIUS_FT = 0.9
BALL_RADIUS_FT = 0.39


@dataclass
class SynthPlayer:
    x: float
    y: float
    team: Team | None  # None = referee
    number: str = ""


def make_camera() -> PinholeCamera:
    c = RECOMMENDED_SIM_CAMERA
    return PinholeCamera.look_at(c["camera_position"], c["camera_target"], c["image_size"], c["hfov_deg"])


def make_court() -> CourtSpec:
    return court_for("nba")


def draw_court(camera: PinholeCamera, court: CourtSpec) -> np.ndarray:
    w, h = camera.image_size
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (90, 90, 90)  # stands
    floor = np.array([[0, 0, 0], [court.length, 0, 0], [court.length, court.width, 0], [0, court.width, 0]], float)
    cv2.fillPoly(img, [np.round(camera.project(floor)).astype(np.int32)], palette.FLOOR_BGR)
    for side in ("left", "right"):
        bx, by = court.basket_xy(side)
        x0 = 0.0 if side == "left" else court.length
        x1 = court.free_throw_line_from_baseline if side == "left" else court.length - court.free_throw_line_from_baseline
        lane = np.array([[x0, by - court.lane_width / 2, 0], [x1, by - court.lane_width / 2, 0],
                         [x1, by + court.lane_width / 2, 0], [x0, by + court.lane_width / 2, 0]])
        cv2.fillPoly(img, [np.round(camera.project(lane)).astype(np.int32)], palette.PAINT_BGR)
    for seg in court.line_segments():
        pts = np.array([[x, y, 0.0] for x, y in seg])
        cv2.polylines(img, [np.round(camera.project(pts)).astype(np.int32)], False, palette.LINE_BGR, 2)
    return img


def player_bbox(camera: PinholeCamera, x: float, y: float) -> BBox:
    return camera.bbox_of_cylinder(x, y, PLAYER_HEIGHT_FT, PLAYER_RADIUS_FT)


def draw_figure(img: np.ndarray, box: BBox, shirt_bgr: tuple[int, int, int], number: str = "") -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box.as_tuple())
    w, h = x2 - x1, y2 - y1
    r = max(2, int(round(min(w / 2.0, 0.11 * h))))
    cv2.rectangle(img, (x1, y1 + 2 * r), (x2 - 1, y2 - 1), shirt_bgr, -1)
    cv2.circle(img, ((x1 + x2) // 2, y1 + r), r, palette.SKIN_BGR, -1)
    if number:
        scale = max(0.35, box.h / 70.0)
        (tw, th), _ = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cx = (box.x1 + box.x2) / 2.0
        org = (int(round(cx - tw / 2.0)), int(round(box.y1 + 0.4 * box.h + th / 2.0)))
        cv2.putText(img, number, org, cv2.FONT_HERSHEY_SIMPLEX, scale, palette.NUMBER_BGR, 2)


def team_bgr(team: Team | None) -> tuple[int, int, int]:
    if team is Team.HOME:
        return palette.HOME_BGR
    if team is Team.AWAY:
        return palette.AWAY_BGR
    return palette.REF_BGR


def rim_bbox(camera: PinholeCamera, court: CourtSpec, side: str) -> BBox:
    bx, by = court.basket_xy(side)  # type: ignore[arg-type]
    return camera.bbox_of_circle(bx, by, court.rim_height, court.rim_radius)


def draw_rims(img: np.ndarray, camera: PinholeCamera, court: CourtSpec) -> None:
    for side in ("left", "right"):
        bx, by = court.basket_xy(side)  # type: ignore[arg-type]
        ring = np.array([[bx + court.rim_radius * np.cos(a), by + court.rim_radius * np.sin(a), court.rim_height]
                         for a in np.linspace(0, 2 * np.pi, 48)])
        cv2.polylines(img, [np.round(camera.project(ring)).astype(np.int32)], True, palette.RIM_BGR, 2)


def ball_image(camera: PinholeCamera, x: float, y: float, z: float) -> tuple[float, float, float]:
    b = camera.bbox_of_sphere(x, y, z, BALL_RADIUS_FT)
    return (b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0, (b.w + b.h) / 4.0


def draw_ball(img: np.ndarray, u: float, v: float, r: float) -> None:
    # Sub-pixel centre: cv2.circle with shift=4 (1/16 px).
    cv2.circle(img, (int(round(u * 16)), int(round(v * 16))), int(round(r * 16)), palette.BALL_BGR, -1, cv2.LINE_8, 4)


def render(
    camera: PinholeCamera,
    court: CourtSpec,
    people: list[SynthPlayer],
    ball_xyz: tuple[float, float, float] | None = None,
    base: np.ndarray | None = None,
) -> tuple[np.ndarray, list[BBox], tuple[float, float, float] | None]:
    """Draw a frame; returns (image, truth bboxes in ``people`` order, ball (u, v, r) or None)."""
    img = (base if base is not None else draw_court(camera, court)).copy()
    draw_rims(img, camera, court)
    boxes = [player_bbox(camera, p.x, p.y) for p in people]
    order = sorted(range(len(people)), key=lambda i: -float(camera.depth([people[i].x, people[i].y, 0.0])))
    for i in order:
        draw_figure(img, boxes[i], team_bgr(people[i].team), people[i].number)
    ball = None
    if ball_xyz is not None:
        ball = ball_image(camera, *ball_xyz)
        draw_ball(img, *ball)
    return img, boxes, ball


SCOREBOARD_ROI = {
    "corners": [[540.0, 20.0], [740.0, 20.0], [740.0, 100.0], [540.0, 100.0]],
    "panel_size": [200, 80],
    "cells": {
        "home": [[6.0, 8.0, 16.0, 28.0], [24.0, 8.0, 16.0, 28.0], [42.0, 8.0, 16.0, 28.0]],
        "away": [[142.0, 8.0, 16.0, 28.0], [160.0, 8.0, 16.0, 28.0], [178.0, 8.0, 16.0, 28.0]],
        "period": [[92.0, 8.0, 16.0, 28.0]],
        "clock": [[60.0, 44.0, 16.0, 28.0], [78.0, 44.0, 16.0, 28.0], [106.0, 44.0, 16.0, 28.0], [124.0, 44.0, 16.0, 28.0]],
        "shot": [[160.0, 44.0, 16.0, 28.0], [178.0, 44.0, 16.0, 28.0]],
    },
}


def scoreboard_text(home: int, away: int, period: int, clock_s: int, shot: int | None) -> dict[str, list[str]]:
    mm, ss = divmod(clock_s, 60)
    return {
        "home": list(f"{home:>3d}".replace(" ", "\0")),
        "away": list(f"{away:>3d}".replace(" ", "\0")),
        "period": [str(period)],
        "clock": list(f"{mm:>2d}{ss:02d}".replace(" ", "\0")),
        "shot": list(f"{shot:>2d}".replace(" ", "\0")) if shot is not None else ["\0", "\0"],
    }


def draw_scoreboard(img: np.ndarray, roi: dict, text: dict[str, list[str]]) -> None:
    """Draw an axis-aligned panel at the ROI corners (TL..BR) per the seven-segment convention."""
    pw, ph = roi["panel_size"]
    panel = np.zeros((ph, pw, 3), np.uint8)
    panel[:] = PANEL_BGR
    for name, cells in roi["cells"].items():
        for cell, ch in zip(cells, text[name]):
            draw_digit(panel, tuple(cell), "" if ch == "\0" else ch)
    x0, y0 = (int(v) for v in roi["corners"][0])
    img[y0:y0 + ph, x0:x0 + pw] = panel


# ---------------------------------------------------------------------------- sanity tests


def test_recommended_camera_frames_players_and_rims() -> None:
    cam, court = make_camera(), make_court()
    img, boxes, ball = render(cam, court, [SynthPlayer(10, 10, Team.HOME, "23"), SynthPlayer(80, 40, Team.AWAY, "7")],
                              ball_xyz=(47, 25, 4))
    h, w = img.shape[:2]
    for b in boxes:
        assert 0 <= b.x1 < b.x2 <= w and 0 <= b.y1 < b.y2 <= h
        assert b.h > 40
    assert ball is not None and 2.0 < ball[2] < 12.0
    for side in ("left", "right"):
        rb = rim_bbox(cam, court, side)
        assert 0 < rb.x1 < rb.x2 < w


def test_scoreboard_text_layout() -> None:
    t = scoreboard_text(7, 102, 4, 9 * 60 + 5, 14)
    assert t["home"] == ["\0", "\0", "7"] and t["away"] == ["1", "0", "2"]
    assert t["clock"] == ["\0", "9", "0", "5"] and t["shot"] == ["1", "4"]
