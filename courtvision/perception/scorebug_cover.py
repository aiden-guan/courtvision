"""Where a broadcast's scorebug covers the picture, and the ball candidates there.

The bug is an opaque overlay, so no ball candidate centred in it is the game ball, but a team logo in it can outscore
the real ball. The court tracker keeps the bug out of its line evidence (tools/nba_track_court.py; a profile's boxes:
``scorebug_ocr.bug_boxes``, x, y, w, h of a 1920x1080 frame); the ball filter drops the candidates centred in the same
boxes: offline before the ball is linked (``nba/analyze.without_balls_in``, ``nba_analyze.py ball --scorebug-filter``)
and live while the reader sees its bug (``perception/broadcast.py``, ``drop_scorebug_balls``). Light on purpose:
the browser runtime imports ``perception/broadcast.py``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

__all__ = ["DROP_BALLS_IN_BUG", "Box", "frame_boxes", "in_boxes"]

Box = tuple[int, int, int, int]  # x, y, w, h

# Drop ball candidates centred in the scorebug before the ball is linked or tracked: the default of
# ``nba_analyze.py ball --scorebug-filter`` and of ``BroadcastPerceptionConfig.drop_scorebug_balls`` (the Studio).
# Off until its effect is measured on the research games (docs/NBA.md, "The scorebug filter").
DROP_BALLS_IN_BUG = False


def frame_boxes(boxes: Iterable[Box], size: tuple[int, int]) -> list[Box]:
    """Boxes of a 1920x1080 frame (x, y, w, h) in the pixels of a ``size`` (width, height) frame, rounded as the
    court evidence rounds its masks (``court_search.court_evidence``)."""
    sx, sy = size[0] / 1920.0, size[1] / 1080.0
    return [(round(x * sx), round(y * sy), round(w * sx), round(h * sy)) for x, y, w, h in boxes]


def in_boxes(points: Any, boxes: Iterable[Box]) -> np.ndarray:
    """Whether each point (rows x, y, ...) lies in one of ``boxes`` (x, y, w, h in the points' pixels). A box covers
    [x, x + w) x [y, y + h): the pixels the court evidence masks for it."""
    pts = np.asarray(points, float)
    inside = np.zeros(len(pts), bool)
    if not len(pts):
        return inside
    pts = pts.reshape(len(pts), -1)
    x, y = pts[:, 0], pts[:, 1]
    for bx, by, bw, bh in boxes:
        inside |= (x >= bx) & (x < bx + bw) & (y >= by) & (y < by + bh)
    return inside
