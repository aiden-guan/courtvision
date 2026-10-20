"""Seven-segment scoreboard reading on cells drawn per the shared convention."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from test_perception_synth import SCOREBOARD_ROI, draw_scoreboard, scoreboard_text

from courtvision.perception.scoreboard_reader import (
    DIGIT_SEGMENTS,
    PANEL_BGR,
    DigitClassifier,
    ScoreboardReader,
    ScoreboardROI,
    SevenSegmentClassifier,
    draw_digit,
    rectify,
    segment_rects,
)


def cell_image(digit: str, w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = PANEL_BGR
    draw_digit(img, (0, 0, w, h), digit)
    return img


@pytest.mark.parametrize("w,h", [(10, 18), (16, 28), (23, 40), (40, 70)])
def test_every_digit_and_blank_exact(w: int, h: int) -> None:
    clf = SevenSegmentClassifier()
    assert isinstance(clf, DigitClassifier)
    for d in DIGIT_SEGMENTS:
        got, conf = clf.classify(cell_image(d, w, h))
        assert (got, conf) == (d, 1.0), (d, w, h)


def test_segment_geometry_follows_convention() -> None:
    r = segment_rects(25, 44)  # t = round(4.0) = 4, m = round(3.0) = 3
    assert r["a"] == (3, 3, 22, 7) and r["d"] == (3, 37, 22, 41)
    assert r["f"] == (3, 3, 7, 22) and r["e"] == (3, 22, 7, 41)
    assert r["b"] == (18, 3, 22, 22) and r["c"] == (18, 22, 22, 41)


def test_invalid_pattern_has_low_confidence() -> None:
    img = cell_image("", 20, 36)
    for name in ("a", "e"):  # not a digit
        x1, y1, x2, y2 = segment_rects(20, 36)[name]
        img[y1:y2, x1:x2] = (0, 0, 255)
    _, conf = SevenSegmentClassifier().classify(img)
    assert conf <= 0.25


def test_reader_on_axis_aligned_panel() -> None:
    frame = np.full((720, 1280, 3), 90, np.uint8)
    draw_scoreboard(frame, SCOREBOARD_ROI, scoreboard_text(45, 38, 2, 7 * 60 + 3, 12))
    roi = ScoreboardROI.from_dict(SCOREBOARD_ROI)
    reading = ScoreboardReader(roi).read(frame, 12.5)
    assert reading is not None
    assert (reading.home_score, reading.away_score, reading.period) == (45, 38, 2)
    assert reading.game_clock == 423.0 and reading.shot_clock == 12.0
    assert reading.source == "ocr" and reading.confidence == 1.0 and reading.t == 12.5


@pytest.mark.parametrize("home,away,period,clock,shot", [(0, 0, 1, 720, 24), (101, 9, 4, 59, None), (7, 120, 5, 300, 3)])
def test_reader_values_blank_cells_and_shot_clock_off(home, away, period, clock, shot) -> None:
    frame = np.full((720, 1280, 3), 90, np.uint8)
    draw_scoreboard(frame, SCOREBOARD_ROI, scoreboard_text(home, away, period, clock, shot))
    r = ScoreboardReader(ScoreboardROI.from_dict(SCOREBOARD_ROI)).read(frame, 0.0)
    assert r is not None
    assert (r.home_score, r.away_score, r.period, r.game_clock, r.shot_clock) == (
        home, away, period, float(clock), None if shot is None else float(shot))


def test_reader_rectifies_a_perspective_panel() -> None:
    scale = 2  # 400 x 160 panel, cells 32 px wide
    roi_d = {"panel_size": [200 * scale, 80 * scale], "corners": [[0, 0]] * 4,
             "cells": {k: [[v * scale for v in c] for c in cells] for k, cells in SCOREBOARD_ROI["cells"].items()}}
    panel = np.zeros((80 * scale, 200 * scale, 3), np.uint8)
    panel[:] = PANEL_BGR
    for name, cells in roi_d["cells"].items():
        for cell, ch in zip(cells, scoreboard_text(88, 91, 3, 2 * 60 + 41, 7)[name]):
            draw_digit(panel, tuple(cell), "" if ch == "\0" else ch)
    corners = np.array([[420, 30], [830, 50], [820, 205], [430, 190]], np.float32)  # keystoned
    src = np.array([[0, 0], [400, 0], [400, 160], [0, 160]], np.float32)
    frame = np.full((720, 1280, 3), 90, np.uint8)
    warped = cv2.warpPerspective(panel, cv2.getPerspectiveTransform(src, corners), (1280, 720),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT, dst=frame)
    roi_d["corners"] = corners.tolist()
    roi = ScoreboardROI.from_dict(roi_d)
    assert rectify(warped, roi).shape == (160, 400, 3)
    r = ScoreboardReader(roi).read(warped, 1.0)
    assert r is not None
    assert (r.home_score, r.away_score, r.period, r.game_clock, r.shot_clock) == (88, 91, 3, 161.0, 7.0)
    assert r.confidence > 0.5


def test_roi_serialisation_and_no_panel() -> None:
    roi = ScoreboardROI.from_dict(SCOREBOARD_ROI)
    again = ScoreboardROI.from_dict(roi.to_dict())
    assert again == roi and again.panel_size == (200, 80)
    derived = ScoreboardROI(corners=SCOREBOARD_ROI["corners"], cells={})
    assert derived.panel_size == (200, 80)
    with pytest.raises(ValueError):
        ScoreboardROI(corners=[[0, 0], [1, 1]], cells={})
    dark = np.full((720, 1280, 3), 20, np.uint8)
    assert ScoreboardReader(roi).read(dark, 0.0) is None  # every cell blank


_PLACES = {3: ["hundreds", "tens", "ones"], 2: ["tens", "ones"], 1: [""]}
_CLOCK_NAMES = ["min_tens", "min_ones", "sec_tens", "sec_ones"]


def operator_roi(native: dict) -> dict:
    """The POST /api/scoreboard_roi body for a native ROI whose panel sits axis-aligned at its TL
    corner: one image-pixel [x1, y1, x2, y2] box per named digit cell (dashboard cell names)."""
    x0, y0 = native["corners"][0]
    cells = {}
    for group, boxes in native["cells"].items():
        names = _CLOCK_NAMES if group == "clock" else _PLACES[len(boxes)]
        for suffix, (x, y, w, h) in zip(names, boxes):
            cells[f"{group}_{suffix}" if suffix else group] = [x0 + x, y0 + y, x0 + x + w, y0 + y + h]
    return {"corners": native["corners"], "cells": cells, "space": "image"}


def test_operator_roi_form_from_the_server_is_read_exactly() -> None:
    """Regression: the server posts image-pixel boxes per named cell; ScoreboardROI.from_dict raised
    TypeError on that body (an HTTP 500), so the dashboard could never set up the reader."""
    frame = np.full((720, 1280, 3), 90, np.uint8)
    draw_scoreboard(frame, SCOREBOARD_ROI, scoreboard_text(104, 9, 3, 4 * 60 + 27, 18))
    body = operator_roi(SCOREBOARD_ROI)
    roi = ScoreboardROI.from_dict(body)
    assert roi.panel_size == (200, 80)
    assert roi.cells["home"] == [tuple(c) for c in SCOREBOARD_ROI["cells"]["home"]]
    r = ScoreboardReader(roi).read(frame, 2.0)
    assert r is not None and r.confidence == 1.0
    assert (r.home_score, r.away_score, r.period, r.game_clock, r.shot_clock) == (104, 9, 3, 267.0, 18.0)
    shuffled = dict(reversed(list(body["cells"].items())))  # dict order must not matter
    assert ScoreboardROI.from_dict({**body, "cells": shuffled}).cells == roi.cells
    again = ScoreboardROI.from_dict(roi.to_dict())  # native round trip
    assert again.cells == roi.cells and again.corners == roi.corners


def test_operator_roi_three_cell_clock_unknown_names_and_errors() -> None:
    frame = np.full((720, 1280, 3), 90, np.uint8)
    draw_scoreboard(frame, SCOREBOARD_ROI, scoreboard_text(12, 30, 2, 7 * 60 + 5, None))
    body = operator_roi(SCOREBOARD_ROI)
    del body["cells"]["clock_min_tens"]  # a board without a minute-tens digit
    body["cells"]["fouls_home"] = [545.0, 25.0, 550.0, 30.0]  # not a digit we read
    r = ScoreboardReader(ScoreboardROI.from_dict(body)).read(frame, 0.0)
    assert r is not None and (r.home_score, r.away_score, r.game_clock, r.shot_clock) == (12, 30, 425.0, None)
    with pytest.raises(ValueError, match="no recognised"):
        ScoreboardROI.from_dict({"corners": SCOREBOARD_ROI["corners"], "cells": {"a": [545, 25, 560, 50]}})
    with pytest.raises(ValueError):
        ScoreboardROI.from_dict({"cells": {"home_ones": [545, 25, 560, 50]}})  # no corners
    with pytest.raises(ValueError):
        ScoreboardROI.from_dict({**body, "cells": {"home_ones": [560, 25, 545, 50]}})  # x2 < x1
    with pytest.raises(ValueError):
        ScoreboardROI.from_dict({"corners": SCOREBOARD_ROI["corners"], "cells": {"home": [1, 2]}})
