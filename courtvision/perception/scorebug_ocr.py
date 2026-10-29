"""TV scorebug reader through text recognition (``vision_ocr.py``), for real broadcasts.

``scorebug.py`` reads a scorebug with glyph templates that must be learned per broadcast font. This reader needs no
templates: Apple's Vision recognises the text of the bug's crop and ``parse_lines`` assigns each recognised line to
a field by its pattern and position. Score panels are fixed regions of the bug; the period, the game clock and the
shot clock share one panel whose layout shifts (ESPN moves the clock right when the shot clock is off), so they are
told apart by pattern and order rather than by fixed boxes:

* period: ``1ST`` / ``2ND`` / ``3RD`` / ``4TH`` / ``OT`` / ``2OT`` (``Q3`` style too);
* game clock: ``M:SS``, or ``SS.t`` in the last minute;
* shot clock: an integer 0-24 right of the game clock.

A profile (``ScorebugProfile``) holds a broadcaster's regions in pixels of a 1920x1080 frame (scaled to the
frame). ``ESPN_2024`` is measured on ESPN / ABC NBA broadcasts of the 2024-25 season (gold frames of
Lakers @ Warriors, 2024-12-25).

**Presence** (the broadcast gate's "the bug is shown", a replay or close-up hides it) is a cheap check: the
normalised correlation of the profile's static logo region with the logo learned from the first good reads.
Scores read as None until seen: ESPN draws a lone 0 as a ring that the recogniser skips; ``zero_until_seen`` reads
a missing score as 0 while that team has not scored yet.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from courtvision.types import ScoreboardReading

from .vision_ocr import OCRLine, VisionOCR

__all__ = ["ESPN_2024", "OCRScorebugReader", "ScorebugProfile", "parse_lines"]

log = logging.getLogger(__name__)

_PERIOD = re.compile(r"^(?:Q?([1-4])(?:ST|ND|RD|TH)?|([2-9])?OT)$")
_CLOCK = re.compile(r"^(\d{1,2}):(\d{2})$|^(\d{1,2})\.(\d)$")


@dataclass(frozen=True)
class ScorebugProfile:
    """A broadcaster's scorebug in pixels of a ``ref_size`` frame: the crop sent to the recogniser, the x-ranges
    of the away score, the home score and the period / clock / shot-clock panel, and a static logo region."""

    name: str
    ocr_box: tuple[int, int, int, int]  # x, y, w, h
    away_score_x: tuple[int, int]
    home_score_x: tuple[int, int]
    clock_panel_x: tuple[int, int]
    logo_box: tuple[int, int, int, int]
    away_first: bool = True  # the away team is drawn left of the home team
    ref_size: tuple[int, int] = (1920, 1080)

    def scaled(self, size: tuple[int, int]) -> ScorebugProfile:
        sx, sy = size[0] / self.ref_size[0], size[1] / self.ref_size[1]

        def box(b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
            return (int(round(b[0] * sx)), int(round(b[1] * sy)), int(round(b[2] * sx)), int(round(b[3] * sy)))

        def xs(r: tuple[int, int]) -> tuple[int, int]:
            return (int(round(r[0] * sx)), int(round(r[1] * sx)))

        return ScorebugProfile(self.name, box(self.ocr_box), xs(self.away_score_x), xs(self.home_score_x),
                               xs(self.clock_panel_x), box(self.logo_box), self.away_first, size)


ESPN_2024 = ScorebugProfile(
    name="espn-2024",
    ocr_box=(590, 950, 870, 60),
    away_score_x=(700, 860),
    home_score_x=(960, 1125),
    clock_panel_x=(1128, 1460),
    logo_box=(478, 962, 104, 34),
)


def _period(text: str) -> int | None:
    m = _PERIOD.match(text.upper().replace(" ", "").replace("I", "1"))
    if m is None:
        return None
    if m.group(1):
        return int(m.group(1))
    return 4 + (int(m.group(2)) if m.group(2) else 1)


def _clock(text: str) -> float | None:
    m = _CLOCK.match(text.replace(" ", ""))
    if m is None:
        return None
    if m.group(1) is not None:
        mins, secs = int(m.group(1)), int(m.group(2))
        return 60.0 * mins + secs if mins <= 12 and secs < 60 else None
    return int(m.group(3)) + int(m.group(4)) / 10.0


def parse_lines(lines: list[OCRLine], profile: ScorebugProfile, origin_x: float = 0.0) -> dict[str, Any]:
    """Fields of one bug read. ``lines`` are boxes in crop pixels; ``origin_x`` is the crop's x in the frame."""
    out: dict[str, Any] = {"away_score": None, "home_score": None, "period": None, "game_clock": None,
                           "shot_clock": None}
    tokens: list[tuple[str, float, float]] = []  # text, centre x (frame), right x (frame)
    for ln in lines:
        x, _, w, _ = ln.box
        parts = ln.text.split()
        if len(parts) > 1:  # a merged line ("2ND 5"): split its box evenly by characters
            n = sum(len(p) for p in parts) + len(parts) - 1
            pos = 0
            for p in parts:
                a = x + w * pos / n
                b = x + w * (pos + len(p)) / n
                tokens.append((p, origin_x + (a + b) / 2.0, origin_x + b))
                pos += len(p) + 1
        else:
            tokens.append((ln.text.strip(), origin_x + x + w / 2.0, origin_x + x + w))
    clock_right = None
    for text, cx, right in tokens:
        lo, hi = profile.clock_panel_x
        if lo <= cx <= hi:
            c = _clock(text)
            if c is not None and out["game_clock"] is None:
                out["game_clock"], clock_right = c, right
                continue
            p = _period(text)
            if p is not None and out["period"] is None and (out["game_clock"] is None or cx < clock_right):
                out["period"] = p
    for text, cx, _ in tokens:
        if not text.isdigit():
            continue
        v = int(text)
        if profile.away_score_x[0] <= cx <= profile.away_score_x[1]:
            out["away_score"] = v
        elif profile.home_score_x[0] <= cx <= profile.home_score_x[1]:
            out["home_score"] = v
        elif clock_right is not None and cx > clock_right and profile.clock_panel_x[0] <= cx <= profile.clock_panel_x[1] \
                and v <= 24:
            out["shot_clock"] = v
    return out


@dataclass
class OCRScorebugConfig:
    min_logo_corr: float = 0.75  # presence: logo correlation with the learned logo; tuned on ESPN gold frames
    confidence: float = 0.95
    zero_until_seen: bool = True


@dataclass
class OCRScorebugReader:
    """``present(frame)`` and ``read(frame, t) -> ScoreboardReading | None`` (see the module docstring)."""

    profile: ScorebugProfile = ESPN_2024
    config: OCRScorebugConfig = field(default_factory=OCRScorebugConfig)
    ocr: VisionOCR | None = None
    _logo: np.ndarray | None = None
    _scaled: ScorebugProfile | None = None
    _seen: dict[str, bool] = field(default_factory=dict)

    def _prof(self, frame: np.ndarray) -> ScorebugProfile:
        size = (frame.shape[1], frame.shape[0])
        if self._scaled is None or self._scaled.ref_size != size:
            self._scaled = self.profile.scaled(size)
        return self._scaled

    @staticmethod
    def _norm(patch: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g -= g.mean()
        n = float(np.linalg.norm(g))
        return g / n if n > 1e-6 else g

    def _logo_patch(self, frame: np.ndarray) -> np.ndarray:
        x, y, w, h = self._prof(frame).logo_box
        return frame[y:y + h, x:x + w]

    def presence(self, frame: np.ndarray) -> float:
        """Logo correlation in [-1, 1]; 1.0 before a logo is learned (no evidence against the bug)."""
        if self._logo is None:
            return 1.0
        patch = self._logo_patch(frame)
        if patch.shape[:2] != self._logo.shape:
            return 0.0
        return float((self._norm(patch) * self._logo).sum())

    def present(self, frame: np.ndarray) -> bool:
        return self.presence(frame) >= self.config.min_logo_corr

    def read_fields(self, frame: np.ndarray) -> dict[str, Any]:
        if self.ocr is None:
            self.ocr = VisionOCR()
        prof = self._prof(frame)
        x, y, w, h = prof.ocr_box
        crop = frame[max(0, y):y + h, max(0, x):x + w]
        return parse_lines(self.ocr.read(crop), prof, origin_x=float(max(0, x)))

    def read(self, frame: np.ndarray, t: float) -> ScoreboardReading | None:
        if self._logo is not None and not self.present(frame):
            return None
        f = self.read_fields(frame)
        if f["game_clock"] is None and f["period"] is None:
            return None
        if self._logo is None and f["game_clock"] is not None and f["period"] is not None:
            self._logo = self._norm(self._logo_patch(frame))  # a confident read: this frame shows the bug
        home, away = f["home_score"], f["away_score"]
        if self.config.zero_until_seen:
            for k in ("home_score", "away_score"):
                if f[k] is not None and f[k] > 0:
                    self._seen[k] = True
            if home is None and not self._seen.get("home_score"):
                home = 0
            if away is None and not self._seen.get("away_score"):
                away = 0
        return ScoreboardReading(t=t, home_score=home, away_score=away, period=f["period"], game_clock=f["game_clock"],
                                 shot_clock=None if f["shot_clock"] is None else float(f["shot_clock"]),
                                 source="ocr", confidence=self.config.confidence)

    def close(self) -> None:
        if self.ocr is not None:
            self.ocr.close()
            self.ocr = None
