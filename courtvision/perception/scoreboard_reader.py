"""Fixed-camera venue scoreboard reader (RESEARCH.md §7A.5, cv_pipeline.md §13.5).

The operator clicks the panel's 4 corners (TL, TR, BR, BL) and marks digit cells in the rectified
panel frame (``ScoreboardROI``, JSON via ``POST /api/scoreboard_roi``). ``rectify`` warps the panel
to ``panel_size`` with ``cv2.getPerspectiveTransform``; a ``DigitClassifier`` reads each cell.

Seven-segment convention shared with ``sim/render.py`` (which draws) and this module (which reads):
panel background BGR (20, 20, 20); lit segments BGR (0, 0, 255); unlit segments are not drawn.
In a cell (x, y, w, h) segments are filled axis-aligned bars of thickness ``t = round(0.16 w)``
inset by ``m = round(0.12 w)`` from the cell edges; the vertical bars each span half the inner
height. a = top, b = upper right, c = lower right, d = bottom, e = lower left, f = upper left,
g = middle. Blank = no segment lit. The clock is "MM:SS" in 4 cells (the colon is not read).

``SevenSegmentClassifier`` samples the centre of every segment, so it is exact on the synthetic
renderer and tolerant of a pixel of rectification error. A trained 11-class CNN (0-9, blank) is the
production plan for real boards; it plugs in through ``DigitClassifier``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np

from courtvision.types import ScoreboardReading

__all__ = [
    "DIGIT_SEGMENTS",
    "DigitClassifier",
    "ScoreboardROI",
    "ScoreboardReader",
    "ScoreboardReaderConfig",
    "SevenSegmentClassifier",
    "SevenSegmentConfig",
    "draw_digit",
    "rectify",
    "segment_rects",
]

log = logging.getLogger(__name__)

PANEL_BGR = (20, 20, 20)
LIT_BGR = (0, 0, 255)
SEGMENT_NAMES = "abcdefg"
DIGIT_SEGMENTS: dict[str, str] = {
    "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg",
    "5": "acdfg", "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg", "": "",
}
_PATTERN_TO_DIGIT = {frozenset(v): k for k, v in DIGIT_SEGMENTS.items()}
Cell = tuple[float, float, float, float]  # x, y, w, h in the rectified panel frame


@dataclass
class ScoreboardROI:
    """Panel corners in the image and digit cells in the rectified panel frame."""

    corners: list[list[float]]  # [[x, y]] x 4: TL, TR, BR, BL (image px)
    cells: dict[str, list[Cell]]  # "home"/"away" (3), "period" (1), "clock" (4), "shot" (2)
    panel_size: tuple[int, int] | None = None  # rectified (w, h); None => from corner distances

    def __post_init__(self) -> None:
        c = np.asarray(self.corners, dtype=float)
        if c.shape != (4, 2):
            raise ValueError(f"scoreboard ROI needs 4 corners [[x, y]] x 4, got shape {c.shape}")
        self.corners = c.tolist()
        self.cells = {k: [_cell(c) for c in cells] for k, cells in self.cells.items()}
        if self.panel_size is None:
            w = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2.0
            h = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2.0
            self.panel_size = (int(round(w)), int(round(h)))
        self.panel_size = (int(self.panel_size[0]), int(self.panel_size[1]))
        if min(self.panel_size) <= 0:
            raise ValueError(f"scoreboard panel size must be positive, got {self.panel_size}")

    def to_dict(self) -> dict[str, Any]:
        assert self.panel_size is not None
        return {"corners": [list(p) for p in self.corners], "panel_size": list(self.panel_size),
                "cells": {k: [list(cell) for cell in v] for k, v in self.cells.items()}}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScoreboardROI:
        """Parse ``to_dict`` output, or the operator form sent by ``POST /api/scoreboard_roi``.

        Native form: ``cells`` maps a group ("home", "away", "period", "clock", "shot") to a list of
        ``[x, y, w, h]`` cells in the rectified panel frame. Operator form (``"space": "image"``, or
        every cell a flat ``[x1, y1, x2, y2]``): one image-pixel box per named digit cell, e.g.
        ``home_tens``, ``home_ones``, ``period``, ``clock_min_tens`` .. ``clock_sec_ones``,
        ``shot_tens`` (or ``<group>_<index>``); boxes are mapped into the rectified frame and grouped
        by significance. Unknown names are ignored. Raises ``ValueError`` on malformed input.
        """
        try:
            size = d.get("panel_size")
            panel_size = (int(size[0]), int(size[1])) if size is not None else None
            cells = d.get("cells") or {}
            if d.get("space") == "image" or (cells and all(_is_flat_box(v) for v in cells.values())):
                return cls._from_image_cells(d["corners"], cells, panel_size)
            return cls(corners=d["corners"], cells=cells, panel_size=panel_size)
        except (KeyError, TypeError, IndexError, AttributeError) as exc:
            raise ValueError(f"malformed scoreboard ROI: {exc!r}") from exc

    @classmethod
    def _from_image_cells(
        cls, corners: Any, cells: dict[str, Any], panel_size: tuple[int, int] | None
    ) -> ScoreboardROI:
        base = cls(corners=corners, cells={}, panel_size=panel_size)
        assert base.panel_size is not None
        M = _panel_transform(base)
        slots: dict[str, dict[float, Cell]] = {}
        for name, box in cells.items():
            slot = _cell_slot(str(name))
            if slot is None:
                log.warning("scoreboard ROI: ignoring unrecognised cell name %r", name)
                continue
            if not _is_flat_box(box):
                raise ValueError(f"cell {name!r} must be [x1, y1, x2, y2] in image pixels")
            group, rank = slot
            if rank in slots.setdefault(group, {}):
                raise ValueError(f"scoreboard ROI: two cells for the same digit of {group!r} ({name!r})")
            slots[group][rank] = _image_box_to_cell(M, [float(v) for v in box])
        if not slots:
            raise ValueError("scoreboard ROI has no recognised digit cells (e.g. home_tens, clock_sec_ones)")
        grouped = {g: [s[r] for r in sorted(s)] for g, s in slots.items()}
        return cls(corners=base.corners, cells=grouped, panel_size=base.panel_size)


def _cell(c: Any) -> Cell:
    x, y, w, h = (float(v) for v in c)
    return (x, y, w, h)


_GROUPS = ("home", "away", "period", "clock", "shot")
_PLACE = {"hundreds": 0.0, "tens": 1.0, "ones": 2.0}  # most significant digit first
_CLOCK_PLACE = {"min_tens": 0.0, "min_ones": 1.0, "sec_tens": 2.0, "sec_ones": 3.0}


def _is_flat_box(v: Any) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v)


def _cell_slot(name: str) -> tuple[str, float] | None:
    """(group, rank) of an operator cell name; rank orders digits most significant first."""
    group, _, rest = name.strip().lower().partition("_")
    if group not in _GROUPS:
        return None
    if not rest:
        return group, 0.0
    places = _CLOCK_PLACE if group == "clock" else _PLACE
    if rest in places:
        return group, places[rest]
    return (group, float(rest)) if rest.isdigit() else None


def _panel_transform(roi: ScoreboardROI) -> np.ndarray:
    """Image -> rectified-panel perspective transform (the same one ``rectify`` warps with)."""
    assert roi.panel_size is not None
    w, h = roi.panel_size
    src = np.asarray(roi.corners, dtype=np.float32)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def _image_box_to_cell(M: np.ndarray, box: list[float]) -> Cell:
    x1, y1, x2, y2 = box
    if not (x2 > x1 and y2 > y1):
        raise ValueError(f"cell box {box} needs x2 > x1 and y2 > y1")
    pts = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.float32)
    q = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
    if not np.all(np.isfinite(q)):
        raise ValueError(f"cell box {box} does not map into the scoreboard panel")
    lo, hi = q.min(axis=0), q.max(axis=0)
    return (float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1]))


def rectify(frame: np.ndarray, roi: ScoreboardROI) -> np.ndarray:
    """Warp the panel quadrilateral to a ``panel_size`` image (TL at the origin)."""
    assert roi.panel_size is not None
    w, h = roi.panel_size
    return cv2.warpPerspective(frame, _panel_transform(roi), (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)


def segment_rects(w: int, h: int) -> dict[str, tuple[int, int, int, int]]:
    """Segment bars (x1, y1, x2, y2 exclusive) of a w x h cell per the shared convention."""
    t = int(round(0.16 * w))
    m = int(round(0.12 * w))
    mid = int(round(h / 2.0))
    g1 = int(round(h / 2.0 - t / 2.0))
    return {
        "a": (m, m, w - m, m + t),
        "b": (w - m - t, m, w - m, mid),
        "c": (w - m - t, mid, w - m, h - m),
        "d": (m, h - m - t, w - m, h - m),
        "e": (m, mid, m + t, h - m),
        "f": (m, m, m + t, mid),
        "g": (m, g1, w - m, g1 + t),
    }


def draw_digit(panel: np.ndarray, cell: Cell, digit: str, lit_bgr: tuple[int, int, int] = LIT_BGR) -> None:
    """Reference renderer of the convention: draw ``digit`` ("" = blank) into ``cell`` of ``panel``."""
    x, y, w, h = (int(round(v)) for v in cell)
    for name, (x1, y1, x2, y2) in segment_rects(w, h).items():
        if name in DIGIT_SEGMENTS[digit]:
            cv2.rectangle(panel, (x + x1, y + y1), (x + x2 - 1, y + y2 - 1), lit_bgr, thickness=-1)


@runtime_checkable
class DigitClassifier(Protocol):
    def classify(self, cell_bgr: np.ndarray) -> tuple[str, float]: ...


@dataclass
class SevenSegmentConfig:
    """Segment sampling settings (synthetic convention)."""

    lit_bgr: tuple[int, int, int] = LIT_BGR  # sourced (render convention)
    lit_tol: float = 100.0  # BGR distance to the lit colour; background (20,20,20) is ~254 away; tuned
    shrink_long: float = 0.25  # trim this fraction from each end along a bar; tuned
    shrink_short: float = 0.25  # and across it; tuned
    lit_fraction: float = 0.5  # sampled pixels lit => segment on; tuned


class SevenSegmentClassifier:
    """Deterministic segment sampling; confidence = decisiveness of the weakest segment."""

    def __init__(self, config: SevenSegmentConfig | None = None) -> None:
        self.config = config or SevenSegmentConfig()

    def classify(self, cell_bgr: np.ndarray) -> tuple[str, float]:
        h, w = cell_bgr.shape[:2]
        if w < 5 or h < 5:
            return "", 0.0
        lit = np.linalg.norm(cell_bgr.astype(np.float32) - np.asarray(self.config.lit_bgr, np.float32), axis=2)
        lit = lit < self.config.lit_tol
        fracs = {name: self._fraction(lit, rect) for name, rect in segment_rects(w, h).items()}
        on = frozenset(n for n, f in fracs.items() if f >= self.config.lit_fraction)
        decisiveness = min(abs(2.0 * f - 1.0) for f in fracs.values())
        digit = _PATTERN_TO_DIGIT.get(on)
        if digit is not None:
            return digit, float(decisiveness)
        # Not a valid pattern: nearest digit by Hamming distance, heavily discounted.
        best = min(DIGIT_SEGMENTS, key=lambda d: (len(on ^ frozenset(DIGIT_SEGMENTS[d])), d))
        dist = len(on ^ frozenset(DIGIT_SEGMENTS[best]))
        return best, float(decisiveness * 0.25**dist)

    def _fraction(self, lit: np.ndarray, rect: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = rect
        horizontal = (x2 - x1) >= (y2 - y1)
        sx = self.config.shrink_long if horizontal else self.config.shrink_short
        sy = self.config.shrink_short if horizontal else self.config.shrink_long
        dx, dy = (x2 - x1) * sx, (y2 - y1) * sy
        a1, a2 = int(np.floor(x1 + dx)), int(np.ceil(x2 - dx))
        b1, b2 = int(np.floor(y1 + dy)), int(np.ceil(y2 - dy))
        a2, b2 = max(a2, a1 + 1), max(b2, b1 + 1)
        region = lit[max(0, b1):b2, max(0, a1):a2]
        return float(region.mean()) if region.size else 0.0


@dataclass
class ScoreboardReaderConfig:
    """Reader settings."""

    min_confidence: float = 0.0  # readings below this are dropped (None); tuned
    max_clock_minutes: int = 20  # longer periods are misreads; sourced (no ruleset exceeds 20 min)


@dataclass
class _Field:
    value: int | None
    conf: float
    digits: str = field(default="")


class ScoreboardReader:
    """``read(frame, t) -> ScoreboardReading | None`` (source="ocr", confidence = min cell confidence)."""

    def __init__(
        self,
        roi: ScoreboardROI,
        classifier: DigitClassifier | None = None,
        config: ScoreboardReaderConfig | None = None,
    ) -> None:
        self.roi = roi
        self.classifier = classifier or SevenSegmentClassifier()
        self.config = config or ScoreboardReaderConfig()

    def read(self, frame: np.ndarray, t: float) -> ScoreboardReading | None:
        panel = rectify(frame, self.roi)
        fields: dict[str, _Field] = {}
        for name, cells in self.roi.cells.items():
            digits = [self._classify(panel, c) for c in cells]
            fields[name] = self._parse_clock(digits) if name == "clock" else self._parse_number(digits)
        if not fields or all(f.value is None for f in fields.values()):
            return None
        conf = min(f.conf for f in fields.values())
        if conf < self.config.min_confidence:
            return None
        val = {k: f.value for k, f in fields.items()}
        clock = val.get("clock")
        shot = val.get("shot")
        return ScoreboardReading(
            t=t,
            home_score=val.get("home"),
            away_score=val.get("away"),
            period=val.get("period"),
            game_clock=float(clock) if clock is not None else None,
            shot_clock=float(shot) if shot is not None else None,
            source="ocr",
            confidence=float(conf),
        )

    def _classify(self, panel: np.ndarray, cell: Cell) -> tuple[str, float]:
        x, y, w, h = (int(round(v)) for v in cell)
        crop = panel[max(0, y):y + h, max(0, x):x + w]
        if crop.shape[0] != h or crop.shape[1] != w:
            return "", 0.0  # cell outside the panel
        return self.classifier.classify(crop)

    @staticmethod
    def _parse_number(digits: list[tuple[str, float]]) -> _Field:
        conf = min((c for _, c in digits), default=0.0)
        text = "".join(d if d else " " for d, _ in digits)
        stripped = text.strip()
        if not stripped:
            return _Field(None, conf, text)
        if " " in stripped or text.rstrip() != text:  # interior or trailing blank: misread
            return _Field(None, 0.0, text)
        return _Field(int(stripped), conf, text)

    def _parse_clock(self, digits: list[tuple[str, float]]) -> _Field:
        """"MM:SS" from 4 cells, or "M:SS" from 3 (boards without a minute-tens digit)."""
        conf = min((c for _, c in digits), default=0.0)
        if len(digits) not in (3, 4):
            return _Field(None, 0.0)
        text = "".join(d if d else " " for d, _ in digits)
        if not text.strip():
            return _Field(None, conf, text)
        mm, ss = text[:-2], text[-2:]
        if " " in ss or (mm.strip() and mm.rstrip() != mm):  # blank seconds, or "5 :.." misread
            return _Field(None, 0.0, text)
        minutes = int(mm.strip() or "0")
        seconds = int(ss)
        if seconds > 59 or minutes > self.config.max_clock_minutes:
            return _Field(None, 0.0, text)
        return _Field(minutes * 60 + seconds, conf, text)
