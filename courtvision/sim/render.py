"""Synthetic BGR renderer for simulator frames.

Draws the shared palette (``courtvision.sim.palette``): floor, paint, every court line projected
through the camera, backboards and red rims, players as jersey-coloured figures that exactly fill
``PlayerState.bbox`` with a skin head on top and a white jersey number, referees in black, an orange
ball of the projected radius, and a venue scoreboard panel with seven-segment digits.

Seven-segment convention (shared with ``perception/scoreboard_reader.py``, which reads it):

* panel background BGR (20, 20, 20); lit segments BGR (0, 0, 255); unlit segments are not drawn;
* a digit cell is an integer rectangle ``(x, y, w, h)`` in the RECTIFIED panel frame (size
  ``SCOREBOARD_ROI["panel_size"]``); the reader warps the image quadrilateral
  ``SCOREBOARD_ROI["corners"]`` (TL, TR, BR, BL) to that size with ``cv2.getPerspectiveTransform``;
* segments are filled axis-aligned bars of thickness ``t = round(0.16 * w)`` inset by
  ``m = round(0.12 * w)`` from the cell edges; a=top, b=upper-right, c=lower-right, d=bottom,
  e=lower-left, f=upper-left, g=middle; vertical bars span half the inner height each
  (:func:`segment_rects` gives the exact pixel rectangles);
* digits use ``SEGMENTS``; a blank cell has no lit segment. Scores are right-aligned with leading
  blanks, the period is one cell, the game clock is ``MM:SS`` in 4 cells (minute tens blank below
  10:00, colon drawn but not a cell), the shot clock is 2 cells (tens blank below 10, both blank
  when the shot clock is off). Clock values show ``ceil`` of the remaining seconds.

The panel is axis-aligned in the image, so the rectification is a pure translation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from courtvision.geometry.camera import PinholeCamera
from courtvision.geometry.court import CourtSpec
from courtvision.sim import palette
from courtvision.types import BBox, FrameState, ScoreboardReading, Team

__all__ = [
    "PANEL_BG_BGR",
    "RenderConfig",
    "SCOREBOARD_ROI",
    "SEGMENTS",
    "SEGMENT_ON_BGR",
    "SyntheticRenderer",
    "scoreboard_digits",
    "scoreboard_roi",
    "segment_rects",
]

log = logging.getLogger(__name__)

PANEL_BG_BGR = (20, 20, 20)
SEGMENT_ON_BGR = (0, 0, 255)
SEGMENTS: dict[str, str] = {
    "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc",
    "7": "abc", "8": "abcdefg", "9": "abcdfg", "": "",
}
STANDS_BGR = (55, 50, 50)
_TEAM_BGR = {Team.HOME: palette.HOME_BGR, Team.AWAY: palette.AWAY_BGR}
TEXT_BGR = (255, 255, 255)

# Panel layout in panel pixels (rectified frame); the panel sits at the top centre of the image.
_PANEL_W, _PANEL_H = 460, 64
_CELL_W, _CELL_H, _CELL_Y = 24, 44, 10
_CELL_X: dict[str, list[int]] = {
    "home": [12, 38, 64],
    "period": [104],
    "clock": [144, 170, 214, 240],
    "shot": [290, 316],
    "away": [370, 396, 422],
}
_COLON_X = 201
_NEAR_FT = 0.5  # static scene geometry is clipped at this camera depth (points behind would project to nan)


def scoreboard_roi(image_size: tuple[int, int] = (1280, 720)) -> dict[str, Any]:
    """ROI of the venue scoreboard panel for an image of ``image_size`` (JSON-serialisable).

    ``corners`` are image pixels (TL, TR, BR, BL); ``cells`` are ``[x, y, w, h]`` in the rectified
    panel frame of size ``panel_size``.
    """
    w = int(image_size[0])
    x0, y0 = (w - _PANEL_W) // 2, 10
    return {
        "corners": [[x0, y0], [x0 + _PANEL_W, y0], [x0 + _PANEL_W, y0 + _PANEL_H], [x0, y0 + _PANEL_H]],
        "panel_size": [_PANEL_W, _PANEL_H],
        "cells": {name: [[x, _CELL_Y, _CELL_W, _CELL_H] for x in xs] for name, xs in _CELL_X.items()},
    }


SCOREBOARD_ROI: dict[str, Any] = scoreboard_roi((1280, 720))


def segment_rects(cell: Any) -> dict[str, tuple[int, int, int, int]]:
    """Segment -> half-open pixel rectangle ``(x1, y1, x2, y2)`` inside a cell ``(x, y, w, h)``."""
    x, y, w, h = (int(v) for v in cell)
    t, m = int(round(0.16 * w)), int(round(0.12 * w))
    ix0, ix1, iy0, iy1 = x + m, x + w - m, y + m, y + h - m
    ymid = iy0 + (iy1 - iy0) // 2
    g0 = ymid - t // 2
    return {
        "a": (ix0, iy0, ix1, iy0 + t),
        "b": (ix1 - t, iy0, ix1, ymid),
        "c": (ix1 - t, ymid, ix1, iy1),
        "d": (ix0, iy1 - t, ix1, iy1),
        "e": (ix0, ymid, ix0 + t, iy1),
        "f": (ix0, iy0, ix0 + t, ymid),
        "g": (ix0, g0, ix1, g0 + t),
    }


def _ceil_s(v: float) -> int:
    return max(0, int(math.ceil(v - 1e-6)))


def _right_aligned(value: int | None, n: int) -> list[str]:
    if value is None:
        return [""] * n
    text = str(int(value) % (10 ** n))
    return [""] * (n - len(text)) + list(text)


def scoreboard_digits(reading: ScoreboardReading | None) -> dict[str, list[str]]:
    """Characters shown in each cell group ("" = blank) for a scoreboard reading."""
    if reading is None:
        return {name: [""] * len(xs) for name, xs in _CELL_X.items()}
    clock = ["", "", "", ""]
    if reading.game_clock is not None:
        mins, secs = divmod(_ceil_s(reading.game_clock), 60)
        mins = min(mins, 99)
        clock = [str(mins // 10) if mins >= 10 else "", str(mins % 10), str(secs // 10), str(secs % 10)]
    shot = _right_aligned(None if reading.shot_clock is None else min(99, _ceil_s(reading.shot_clock)), 2)
    period = [] if reading.period is None else [str(min(9, max(0, reading.period)))]
    return {
        "home": _right_aligned(reading.home_score, 3),
        "away": _right_aligned(reading.away_score, 3),
        "period": period or [""],
        "clock": clock,
        "shot": shot,
    }


@dataclass
class RenderConfig:
    """Renderer settings."""

    player_height_ft: float = 6.5  # must match SimConfig.player_height_ft for referee boxes
    referee_height_ft: float = 6.2
    figure_radius_ft: float = 1.0
    line_thickness_px: int = 2
    rim_thickness_px: int = 2
    apron_ft: float = 6.0  # floor drawn beyond the boundary lines
    head_frac: float = 0.07  # head radius / figure height
    draw_scoreboard: bool = True


class SyntheticRenderer:
    """Renders a ``FrameState`` from the simulator (or a noisy copy) to a BGR image."""

    def __init__(self, camera: PinholeCamera, court: CourtSpec, config: RenderConfig | None = None) -> None:
        self.camera = camera
        self.court = court
        self.config = config or RenderConfig()
        self.width, self.height = camera.image_size
        self.roi = scoreboard_roi(camera.image_size)
        self._background = self._draw_background()
        self._hoops = self._hoop_items()

    # ------------------------------------------------------------------ static scene

    def _project_int(self, pts: Any) -> np.ndarray:
        uv = self.camera.project(np.asarray(pts, dtype=float))
        return np.round(uv).astype(np.int32)

    def _polygon(self, pts: Any) -> list[np.ndarray]:
        """Projected polygon, clipped to the part in front of the camera ([] if none)."""
        P = np.asarray(pts, dtype=float)
        d = self.camera.depth(P) - _NEAR_FT
        if not np.all(d > 0):  # Sutherland-Hodgman against the near plane
            out = []
            for i in range(len(P)):
                j = (i + 1) % len(P)
                if d[i] > 0:
                    out.append(P[i])
                if (d[i] > 0) != (d[j] > 0):
                    out.append(P[i] + (P[j] - P[i]) * (d[i] / (d[i] - d[j])))
            if len(out) < 3:
                return []
            P = np.asarray(out)
        return [self._project_int(P)]

    def _polylines(self, pts: Any) -> list[np.ndarray]:
        """Projected pieces of a polyline in front of the camera."""
        P = np.asarray(pts, dtype=float)
        d = self.camera.depth(P) - _NEAR_FT
        pieces: list[list[np.ndarray]] = []
        cur: list[np.ndarray] = []
        for i in range(len(P)):
            crossing = i > 0 and (d[i] > 0) != (d[i - 1] > 0)
            if crossing:
                cur.append(P[i - 1] + (P[i] - P[i - 1]) * (d[i - 1] / (d[i - 1] - d[i])))
            if d[i] > 0:
                cur.append(P[i])
            elif cur:
                pieces.append(cur)
                cur = []
        if cur:
            pieces.append(cur)
        return [self._project_int(np.asarray(c)) for c in pieces if len(c) >= 2]

    @staticmethod
    def _on_floor(pts: list[tuple[float, float]]) -> list[tuple[float, float, float]]:
        return [(x, y, 0.0) for x, y in pts]

    def _draw_background(self) -> np.ndarray:
        img = np.empty((self.height, self.width, 3), dtype=np.uint8)
        img[:] = STANDS_BGR
        L, W, a = self.court.length, self.court.width, self.config.apron_ft
        floor = self._on_floor([(-a, -a), (L + a, -a), (L + a, W + a), (-a, W + a)])
        cv2.fillPoly(img, self._polygon(floor), palette.FLOOR_BGR)
        half, ft = self.court.lane_width / 2.0, self.court.free_throw_line_from_baseline
        by = self.court.basket_y
        for x0, x1 in ((0.0, ft), (L, L - ft)):
            lane = [(x0, by - half), (x1, by - half), (x1, by + half), (x0, by + half)]
            cv2.fillPoly(img, self._polygon(self._on_floor(lane)), palette.PAINT_BGR)
        for line in self.court.line_segments():
            cv2.polylines(img, self._polylines(self._on_floor(line)), False, palette.LINE_BGR,
                          self.config.line_thickness_px)
        return img

    def _hoop_items(self) -> list[tuple[float, list[np.ndarray], list[np.ndarray]]]:
        """(depth, backboard polygon, rim polyline) per basket, drawn in depth order with figures."""
        items = []
        by, rim_h = self.court.basket_y, self.court.rim_height
        for side in ("left", "right"):
            bx, _ = self.court.basket_xy(side)
            off = self.court.backboard_offset
            board_x = off if side == "left" else self.court.length - off
            z0, z1 = rim_h - 0.5, rim_h + 3.0  # 3.5-ft board, bottom ~6 in below the ring (note §1a)
            board = self._polygon([(board_x, by - 3.0, z0), (board_x, by + 3.0, z0), (board_x, by + 3.0, z1),
                                   (board_x, by - 3.0, z1)])
            ang = np.linspace(0.0, 2.0 * math.pi, 33)
            r = self.court.rim_radius
            ring = self._polylines([(bx + r * math.cos(t), by + r * math.sin(t), rim_h) for t in ang])
            items.append((float(self.camera.depth([bx, by, rim_h])), board, ring))
        return items

    # ------------------------------------------------------------------ frame

    def render(self, frame: FrameState, referees: list[tuple[float, float]] | None = None) -> np.ndarray:
        """BGR image (H, W, 3) uint8 of one frame."""
        img = self._background.copy()
        drawables: list[tuple[float, int, Any]] = []
        for depth, board, ring in self._hoops:
            drawables.append((depth, 0, ("hoop", board, ring)))
        for p in frame.players:
            if p.bbox is None or not _finite(p.bbox):
                continue
            color = _TEAM_BGR.get(p.team, (128, 128, 128))
            depth = float(self.camera.depth([p.x, p.y, self.config.player_height_ft / 2.0]))
            drawables.append((depth, 1, ("figure", p.bbox, color, p.jersey)))
        for x, y in referees or []:
            box = self.camera.bbox_of_cylinder(x, y, self.config.referee_height_ft, self.config.figure_radius_ft)
            if _finite(box):
                depth = float(self.camera.depth([x, y, self.config.referee_height_ft / 2.0]))
                drawables.append((depth, 1, ("figure", box, palette.REF_BGR, None)))
        drawables.sort(key=lambda d: (-d[0], d[1]))
        for _, _, item in drawables:
            if item[0] == "hoop":
                self._draw_hoop(img, item[1], item[2])
            else:
                self._draw_figure(img, item[1], item[2], item[3])
        self._draw_ball(img, frame)
        if self.config.draw_scoreboard:
            self._draw_scoreboard(img, frame.scoreboard)
        return img

    def _draw_hoop(self, img: np.ndarray, board: list[np.ndarray], ring: list[np.ndarray]) -> None:
        for poly in board:
            cv2.fillConvexPoly(img, poly, palette.BACKBOARD_BGR)
        cv2.polylines(img, ring, False, palette.RIM_BGR, self.config.rim_thickness_px)  # ring points repeat the start

    def _draw_figure(self, img: np.ndarray, box: BBox, color: tuple[int, int, int], number: str | None) -> None:
        """Body rectangle + head circle that together span exactly ``box`` (pixel-rounded)."""
        x1, y1, x2, y2 = int(round(box.x1)), int(round(box.y1)), int(round(box.x2)), int(round(box.y2))
        if x2 <= x1 or y2 <= y1:
            return
        w, h = x2 - x1, y2 - y1
        r = max(1, min(int(round(self.config.head_frac * h)), w // 2))
        cx = (x1 + x2 - 1) // 2
        cv2.rectangle(img, (x1, y1 + 2 * r - 1), (x2 - 1, y2 - 1), color, -1)
        cv2.circle(img, (cx, y1 + r), r, palette.SKIN_BGR, -1)
        if number:  # exact shared call (Hershey glyphs are anti-aliased; wide numbers overflow the figure)
            scale = max(0.35, box.h / 70.0)
            (tw, th), _ = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            org = (int(round((box.x1 + box.x2) / 2.0 - tw / 2.0)), int(round(box.y1 + 0.4 * box.h + th / 2.0)))
            cv2.putText(img, number, org, cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT_BGR, thickness=2)

    def _draw_ball(self, img: np.ndarray, frame: FrameState) -> None:
        b = frame.ball
        if b is None or not b.visible or b.image_xy is None:
            return
        u, v = b.image_xy
        if not (math.isfinite(u) and math.isfinite(v)):
            return
        r = max(2, int(round(b.radius_px or 3.0)))
        cv2.circle(img, (int(round(u)), int(round(v))), r, palette.BALL_BGR, -1)

    def _draw_scoreboard(self, img: np.ndarray, reading: ScoreboardReading | None) -> None:
        (x0, y0), _, (x1, y1), _ = self.roi["corners"]
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), PANEL_BG_BGR, -1)
        digits = scoreboard_digits(reading)
        for name, cells in self.roi["cells"].items():
            for cell, ch in zip(cells, digits[name], strict=True):
                for seg in SEGMENTS[ch]:
                    sx1, sy1, sx2, sy2 = segment_rects(cell)[seg]
                    cv2.rectangle(img, (x0 + sx1, y0 + sy1), (x0 + sx2 - 1, y0 + sy2 - 1), SEGMENT_ON_BGR, -1)
        if any(digits["clock"]):  # colon (not a cell; the reader ignores it)
            for cy in (y0 + _CELL_Y + _CELL_H // 3, y0 + _CELL_Y + 2 * _CELL_H // 3):
                cv2.rectangle(img, (x0 + _COLON_X, cy - 2), (x0 + _COLON_X + 3, cy + 1), SEGMENT_ON_BGR, -1)


def _finite(box: BBox) -> bool:
    return all(math.isfinite(v) for v in (box.x1, box.y1, box.x2, box.y2))
