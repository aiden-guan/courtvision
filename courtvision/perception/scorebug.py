"""Broadcast scorebug reader: the score / clock graphic a TV broadcast overlays (RESEARCH §7A.5, D14).

Unlike the venue board (``scoreboard_reader.py``: seven-segment digits through a perspective warp), a
scorebug is a flat graphic in fixed image pixels with font glyphs. Each broadcaster has its own layout and
font, so the reader is configured per broadcast and **learns its glyphs**:

* ``ScorebugLayout``: the bug box and one rectangle per field (``home_score``, ``away_score``, ``period``,
  ``game_clock``, ``shot_clock``; others are ignored), image pixels, JSON. ``sim/broadcast.scorebug_layout``
  gives the simulator's.
* ``segment_glyphs``: a field crop is binarised (text brighter than the bug: Otsu on grey), connected
  components become glyphs, left to right (stacked components merge, so ":" is one glyph; touching
  digits are cut apart at their emptiest columns).
* ``GlyphBank``: per character a few normalised binary templates (``glyph_size``); a glyph is the
  character of the best normalised correlation. ``learn(crop, text)`` adds the glyphs of a crop whose text
  is known (the operator's first reads, PBP-aligned scores, or a confirmed earlier read) when the glyph
  count matches; ``from_font`` bootstraps it from an OpenCV Hershey font (exact on simulator renders).
* **Presence.** A replay or a cutaway hides the bug: the box must still look like the bug (``presence``:
  share of its pixels within ``bg_tol`` of the bug's background colour, learned from the first confident
  reads or given), else ``read`` returns None. Callers use ``present`` for the broadcast gate.
* Parsing: scores are integers; the period "Q3" / "3RD" / "OT" / "2OT"; the clock "M:SS" or "SS.t"; the
  shot clock an integer. A field whose glyphs are not all recognised (best correlation < ``min_corr``) is
  None. Readings carry ``source="ocr"`` (the engine's 2-read confirmation applies).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from courtvision.types import ScoreboardReading

__all__ = ["GlyphBank", "ScorebugLayout", "ScorebugReader", "ScorebugReaderConfig", "segment_glyphs"]

log = logging.getLogger(__name__)

FIELDS = ("home_score", "away_score", "period", "game_clock", "shot_clock")
_CHARSET = "0123456789:.QOT"


@dataclass
class ScorebugLayout:
    box: list[int]  # [x, y, w, h] image px
    fields: dict[str, list[int]]  # name -> [x, y, w, h]
    bg_bgr: list[int] | None = None  # the bug's background colour (None: learned)

    def to_dict(self) -> dict[str, Any]:
        return {"box": list(self.box), "fields": {k: list(v) for k, v in self.fields.items()}, "bg_bgr": self.bg_bgr}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScorebugLayout:
        return cls([int(v) for v in d["box"]], {k: [int(x) for x in v] for k, v in d["fields"].items()},
                   None if d.get("bg_bgr") is None else [int(v) for v in d["bg_bgr"]])


def segment_glyphs(crop_bgr: np.ndarray, min_area: int = 3, max_digit_aspect: float = 0.8) -> list[np.ndarray]:
    """Binary glyph images (uint8 0/255) left to right. Components stacked one above the other are one glyph
    (":"); a component wider than 1.5 glyphs (``max_digit_aspect`` x the glyph height) is cut into glyphs."""
    if crop_bgr.size == 0:
        return []
    grey = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if int(grey.max()) - int(grey.min()) < 40:
        return []  # empty field (no text)
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = [tuple(int(v) for v in stats[i, :4]) for i in range(1, n) if stats[i, 4] >= min_area]
    boxes.sort()
    merged: list[list[int]] = []
    for x, y, w, h in boxes:
        m = merged[-1] if merged else None
        stacked = m is not None and x <= m[0] + m[2] - 1 and (y >= m[1] + m[3] or y + h <= m[1])
        if stacked:  # one above the other and overlapping in x (the dots of ":"): one glyph
            x2, y2 = max(m[0] + m[2], x + w), max(m[1] + m[3], y + h)
            m[0], m[1] = min(m[0], x), min(m[1], y)
            m[2], m[3] = x2 - m[0], y2 - m[1]
        else:
            merged.append([x, y, w, h])
    tall = [h for _, _, _, h in merged if h >= 0.6 * max(hh for _, _, _, hh in merged)]
    ref_h = max(float(np.median(tall)) if tall else 0.0, 0.4 * binary.shape[0])  # (a lone "." is not a digit)
    digit_w = max_digit_aspect * ref_h
    out = []
    for x, y, w, h in merged:
        g = binary[y:y + h, x:x + w]
        n = int(round(w / digit_w)) if w > 1.5 * digit_w else 1
        if n > 1:  # touching glyphs (kerning, anti-aliasing): cut at the emptiest columns near even spacing
            ink = (g > 0).sum(axis=0)
            cuts = [0]
            for k in range(1, n):
                c = int(round(k * w / n))
                lo, hi = max(cuts[-1] + 1, c - w // (3 * n)), min(w - 1, c + w // (3 * n))
                cuts.append(lo + int(np.argmin(ink[lo:hi + 1])) if hi >= lo else c)
            cuts.append(w)
            out.extend(g[:, a:b] for a, b in zip(cuts[:-1], cuts[1:], strict=True) if b > a)
        else:
            out.append(g)
    return out


@dataclass
class GlyphBank:
    """Character templates: ``char -> [normalised glyph]`` (see the module docstring)."""

    glyph_size: tuple[int, int] = (12, 18)  # (w, h) of the normalised template
    templates: dict[str, list[np.ndarray]] = field(default_factory=dict)
    max_per_char: int = 8

    def _norm(self, g: np.ndarray, field_h: int) -> np.ndarray:
        """Glyphs keep their height relative to the field (":" and "." are small) inside the template."""
        w, h = self.glyph_size
        gh, gw = g.shape
        s = h / max(field_h, gh, 1)
        nw, nh = max(1, int(round(gw * s))), max(1, int(round(gh * s)))
        small = cv2.resize(g, (min(w, nw), min(h, nh)), interpolation=cv2.INTER_AREA)
        out = np.zeros((h, w), np.float32)
        y0, x0 = h - small.shape[0], (w - small.shape[1]) // 2  # bottom-aligned (baseline)
        out[y0:, x0:x0 + small.shape[1]] = small / 255.0
        v = out - out.mean()
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else v

    def add(self, char: str, glyph: np.ndarray, field_h: int) -> None:
        bank = self.templates.setdefault(char, [])
        if len(bank) < self.max_per_char:
            bank.append(self._norm(glyph, field_h))

    def learn(self, crop_bgr: np.ndarray, text: str) -> bool:
        """Add the glyphs of ``crop_bgr`` showing ``text`` (spaces ignored); False when the counts differ."""
        chars = [c for c in text if not c.isspace()]
        glyphs = segment_glyphs(crop_bgr)
        if not chars or len(glyphs) != len(chars):
            return False
        for c, g in zip(chars, glyphs, strict=True):
            self.add(c, g, crop_bgr.shape[0])
        return True

    def classify(self, glyph: np.ndarray, field_h: int) -> tuple[str, float]:
        v = self._norm(glyph, field_h)
        best, best_c = "", -1.0
        for ch, ts in self.templates.items():
            for t in ts:
                c = float((v * t).sum())
                if c > best_c:
                    best, best_c = ch, c
        return best, best_c

    def read(self, crop_bgr: np.ndarray, min_corr: float) -> tuple[str, float] | None:
        """Text of a field crop and its weakest glyph correlation; "" for an empty field; None if unsure."""
        glyphs = segment_glyphs(crop_bgr)
        if not glyphs:
            return "", 1.0
        out, worst = [], 1.0
        for g in glyphs:
            ch, c = self.classify(g, crop_bgr.shape[0])
            if c < min_corr:
                return None
            out.append(ch)
            worst = min(worst, c)
        return "".join(out), worst

    @classmethod
    def from_font(cls, field_h: int, font: int = cv2.FONT_HERSHEY_SIMPLEX, scale: float = 0.9, thickness: int = 2,
                  charset: str = _CHARSET, **kw: Any) -> GlyphBank:
        """Templates rendered with an OpenCV font in a field ``field_h`` px high (white on black)."""
        bank = cls(**kw)
        for ch in charset:
            img = np.zeros((field_h, field_h * 2, 3), np.uint8)
            (tw, th), _ = cv2.getTextSize(ch, font, scale, thickness)
            cv2.putText(img, ch, ((img.shape[1] - tw) // 2, (field_h + th) // 2), font, scale, (255, 255, 255),
                        thickness, lineType=cv2.LINE_AA)
            glyphs = segment_glyphs(img)
            if len(glyphs) == 1:
                bank.add(ch, glyphs[0], field_h)
        return bank

    def to_dict(self) -> dict[str, Any]:
        return {"glyph_size": list(self.glyph_size), "max_per_char": self.max_per_char,
                "templates": {k: [t.round(5).tolist() for t in v] for k, v in self.templates.items()}}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GlyphBank:
        bank = cls(tuple(d.get("glyph_size", (12, 18))), max_per_char=int(d.get("max_per_char", 8)))  # type: ignore[arg-type]
        bank.templates = {k: [np.asarray(t, np.float32) for t in v] for k, v in d.get("templates", {}).items()}
        return bank


@dataclass
class ScorebugReaderConfig:
    min_corr: float = 0.75  # weakest glyph correlation for a field to count; tuned on sim renders
    bg_tol: float = 30.0  # BGR distance from the bug background
    min_presence: float = 0.45  # share of the bug box near its background colour: the bug is shown
    confidence: float = 0.9


def parse_period(text: str) -> int | None:
    t = text.upper()
    m = re.fullmatch(r"Q?([1-4])(ST|ND|RD|TH)?", t)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"(\d)?OT", t)
    if m:
        return 4 + (int(m.group(1)) if m.group(1) else 1)
    return None


def parse_clock(text: str) -> float | None:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if m:
        return 60.0 * int(m.group(1)) + int(m.group(2))
    m = re.fullmatch(r"(\d{1,2})\.(\d)", text)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 10.0
    return None


class ScorebugReader:
    """``read(frame_bgr, t) -> ScoreboardReading | None`` for a ``ScorebugLayout`` and a ``GlyphBank``."""

    def __init__(self, layout: ScorebugLayout, bank: GlyphBank, config: ScorebugReaderConfig | None = None) -> None:
        self.layout = layout
        self.bank = bank
        self.config = config or ScorebugReaderConfig()
        self._bg = None if layout.bg_bgr is None else np.asarray(layout.bg_bgr, float)

    def _crop(self, img: np.ndarray, r: list[int]) -> np.ndarray:
        x, y, w, h = r
        return img[max(0, y):y + h, max(0, x):x + w]

    def presence(self, img: np.ndarray) -> float:
        """Share of the bug box at its background colour (1 = clearly shown); learned on first use."""
        box = self._crop(img, self.layout.box).reshape(-1, 3).astype(float)
        if box.size == 0:
            return 0.0
        if self._bg is None:
            self._bg = np.median(box, axis=0)  # the first frame read is assumed to show the bug
        return float((np.linalg.norm(box - self._bg, axis=1) <= self.config.bg_tol).mean())

    def present(self, img: np.ndarray) -> bool:
        return self.presence(img) >= self.config.min_presence

    def read_fields(self, img: np.ndarray) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for name in FIELDS:
            r = self.layout.fields.get(name)
            if r is None:
                continue
            got = self.bank.read(self._crop(img, r), self.config.min_corr)
            out[name] = None if got is None else got[0]
        return out

    def read(self, img: np.ndarray, t: float) -> ScoreboardReading | None:
        if not self.present(img):
            return None
        f = self.read_fields(img)

        def num(k: str) -> int | None:
            v = f.get(k)
            return int(v) if v and v.isdigit() else None

        period = parse_period(f["period"]) if f.get("period") else None
        clock = parse_clock(f["game_clock"]) if f.get("game_clock") else None
        shot = num("shot_clock")
        reading = ScoreboardReading(t=t, home_score=num("home_score"), away_score=num("away_score"), period=period,
                                    game_clock=clock, shot_clock=float(shot) if shot is not None else None,
                                    source="ocr", confidence=self.config.confidence)
        if all(getattr(reading, k) is None for k in FIELDS):
            return None
        return reading
