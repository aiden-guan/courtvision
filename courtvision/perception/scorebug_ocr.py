"""TV scorebug reader through text recognition (``vision_ocr.py``), for real broadcasts.

``scorebug.py`` reads a scorebug with glyph templates that must be learned per broadcast font. This reader needs no
templates: Apple's Vision recognises the text of the bug's crop and ``parse_lines`` assigns each recognised token to
a field by its pattern and position:

* period: ``1ST`` / ``2ND`` / ``3RD`` / ``4TH`` / ``OT`` / ``2OT`` (``Q3``, ``3Q`` and ``OT2`` styles too);
* game clock: ``M:SS``, or ``SS.t`` in the last minute (``:SS.t`` and ``M:SS.t`` too);
* shot clock: an integer 0-24;
* scores: integers in the score regions.

A profile (``ScorebugProfile``) holds a broadcaster's regions in pixels of a ``ref_size`` frame (scaled to the
frame). It comes in two kinds:

* **x-ranges** (``ESPN_2024``, measured by hand on ESPN / ABC NBA broadcasts of the 2024-25 season, gold frames of
  Lakers @ Warriors, 2024-12-25): the score panels are x-ranges of a one-row bug; the period, the game clock and the
  shot clock share one panel whose layout shifts (ESPN moves the clock right when the shot clock is off), so they
  are told apart by pattern and order (the period, the clock, then the shot clock right of it).
* **boxes** (found for any broadcaster by ``scorebug_layout.discover``, saved as JSON): every field has a 2-D box,
  so stacked team rows, a bug at the top and a shot clock in its own box work. A token belongs to the box holding
  its centre. The words of a longer recognised line ("2ND 5", "LAL 109 GS 105") only have boxes estimated by
  characters, so within such a line the integers are matched in order to the boxes the line crosses (one that two
  boxes could take is dropped rather than guessed) and the clock and period words to theirs. The pattern still
  decides the kind of a token (a clock is never a score); the boxes decide which score, and the game clock from a
  shot clock shown with tenths.

**Presence** (the broadcast gate's "the bug is shown", a replay or close-up hides it) is a cheap check: the
normalised correlation of the profile's static region (``logo_box``: a logo or the team tricodes) with the same
region learned from the first good read (not learned while that region is flat: no texture to match). Scores read
as None until seen: ESPN draws a lone 0 as a ring that the recogniser skips; ``zero_until_seen`` reads a missing
score as 0 while that team has not scored yet.
"""

from __future__ import annotations

import itertools
import logging
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np

from courtvision.types import ScoreboardReading

from .vision_ocr import OCRLine, VisionOCR

__all__ = ["ESPN_2024", "FIELDS", "Box", "OCRScorebugReader", "ScorebugProfile", "Token", "join_ordinals",
           "line_tokens", "parse_lines"]

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]  # x, y, w, h

FIELDS = ("away_score", "home_score", "period", "game_clock", "shot_clock")
PROFILE_FORMAT = "courtvision.scorebug_profile/1"
_BOXES = ("away_score_box", "home_score_box", "clock_box", "period_box", "shot_clock_box")

# a period has its ordinal suffix or a Q: a bare "1"-"4" is far more often a shot clock or a score fragment
_PERIOD = re.compile(r"^(?:Q([1-4])|([1-4])(?:ST|ND|RD|TH|Q)|([2-9])?OT|OT([2-9]))$")
_CLOCK = re.compile(r"^(\d{1,2}):(\d{2})$|^(\d{1,2})\.(\d)$|^(\d{1,2})?:(\d{2})\.(\d)$")
_ORDINAL = ("ST", "ND", "RD", "TH")


def _scale_box(b: Box, sx: float, sy: float) -> Box:
    return (round(b[0] * sx), round(b[1] * sy), round(b[2] * sx), round(b[3] * sy))


@dataclass(frozen=True)
class ScorebugProfile:
    """A broadcaster's scorebug in pixels of a ``ref_size`` frame: the crop sent to the recogniser, the x-ranges
    of the away score, the home score and the period / clock / shot-clock panel, and a static region for presence
    (``logo_box``). Discovered profiles also carry a 2-D box per field (``boxed``), which ``parse_lines`` then uses
    instead of the x-ranges, and ``notes``: how the layout was found and what was assumed."""

    name: str
    ocr_box: Box
    away_score_x: tuple[int, int]
    home_score_x: tuple[int, int]
    clock_panel_x: tuple[int, int]
    logo_box: Box
    away_first: bool = True  # the away team is drawn left of (or above) the home team
    ref_size: tuple[int, int] = (1920, 1080)
    away_score_box: Box | None = None
    home_score_box: Box | None = None
    clock_box: Box | None = None  # where the game clock is drawn (all its positions)
    period_box: Box | None = None  # None: anywhere in the clock box
    shot_clock_box: Box | None = None  # None: no shot clock is read
    notes: tuple[str, ...] = ()

    @property
    def boxed(self) -> bool:
        return self.away_score_box is not None and self.home_score_box is not None and self.clock_box is not None

    def scaled(self, size: tuple[int, int]) -> ScorebugProfile:
        sx, sy = size[0] / self.ref_size[0], size[1] / self.ref_size[1]

        def box(b: Box | None) -> Box | None:
            return None if b is None else _scale_box(b, sx, sy)

        def xs(r: tuple[int, int]) -> tuple[int, int]:
            return (round(r[0] * sx), round(r[1] * sx))

        return replace(self, ocr_box=_scale_box(self.ocr_box, sx, sy), away_score_x=xs(self.away_score_x),
                       home_score_x=xs(self.home_score_x), clock_panel_x=xs(self.clock_panel_x),
                       logo_box=_scale_box(self.logo_box, sx, sy), ref_size=size,
                       away_score_box=box(self.away_score_box), home_score_box=box(self.home_score_box),
                       clock_box=box(self.clock_box), period_box=box(self.period_box),
                       shot_clock_box=box(self.shot_clock_box))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"format": PROFILE_FORMAT, "name": self.name, "ref_size": list(self.ref_size),
                             "ocr_box": list(self.ocr_box), "away_score_x": list(self.away_score_x),
                             "home_score_x": list(self.home_score_x), "clock_panel_x": list(self.clock_panel_x),
                             "logo_box": list(self.logo_box), "away_first": self.away_first}
        for k in _BOXES:
            v = getattr(self, k)
            d[k] = None if v is None else list(v)
        d["notes"] = list(self.notes)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScorebugProfile:
        def ints(v: Any, n: int) -> Any:
            out = tuple(round(float(x)) for x in v)
            if len(out) != n:
                raise ValueError(f"expected {n} numbers, got {v!r}")
            return out

        fmt = d.get("format", PROFILE_FORMAT)
        if fmt != PROFILE_FORMAT:
            raise ValueError(f"unknown scorebug profile format {fmt!r}")
        return cls(name=str(d["name"]), ocr_box=ints(d["ocr_box"], 4), away_score_x=ints(d["away_score_x"], 2),
                   home_score_x=ints(d["home_score_x"], 2), clock_panel_x=ints(d["clock_panel_x"], 2),
                   logo_box=ints(d["logo_box"], 4), away_first=bool(d.get("away_first", True)),
                   ref_size=ints(d.get("ref_size", (1920, 1080)), 2),
                   **{k: None if d.get(k) is None else ints(d[k], 4) for k in _BOXES},
                   notes=tuple(str(n) for n in d.get("notes", ())))


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
    if m.group(1) or m.group(2):
        return int(m.group(1) or m.group(2))
    n = m.group(3) or m.group(4)
    return 4 + (int(n) if n else 1)


def _clock(text: str) -> float | None:
    m = _CLOCK.match(text.replace(" ", ""))
    if m is None:
        return None
    if m.group(1) is not None:
        mins, secs = int(m.group(1)), int(m.group(2))
        return 60.0 * mins + secs if mins <= 12 and secs < 60 else None
    if m.group(3) is not None:
        return int(m.group(3)) + int(m.group(4)) / 10.0
    mins, secs = int(m.group(5) or 0), int(m.group(6))  # ":SS.t" / "M:SS.t"
    return 60.0 * mins + secs + int(m.group(7)) / 10.0 if mins <= 12 and secs < 60 else None


@dataclass(frozen=True)
class Token:
    """One word of a recognised line, its box in frame pixels; ``merged``: split out of a longer line, so its box
    is an estimate (the line's box shared out by characters); ``line``: which line."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 1.0
    merged: bool = False
    line: int = -1

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0


def line_tokens(lines: list[OCRLine], origin_x: float = 0.0, origin_y: float = 0.0) -> list[Token]:
    """Words of recognised lines, boxes moved by the origin (the crop's position in the frame). A line of several
    words ("2ND 5") has its box split evenly by characters."""
    out: list[Token] = []
    for i, ln in enumerate(lines):
        x, y, w, h = ln.box
        y0, y1 = origin_y + y, origin_y + y + h
        parts = ln.text.split()
        if len(parts) > 1:
            n = sum(len(p) for p in parts) + len(parts) - 1
            pos = 0
            for p in parts:
                a = x + w * pos / n
                b = x + w * (pos + len(p)) / n
                out.append(Token(p, origin_x + a, y0, origin_x + b, y1, ln.conf, True, i))
                pos += len(p) + 1
        elif ln.text.strip():
            out.append(Token(ln.text.strip(), origin_x + x, y0, origin_x + x + w, y1, ln.conf, False, i))
    return out


def join_ordinals(tokens: list[Token]) -> list[Token]:
    """A period digit and its ordinal read apart ("4" and a raised "TH") become one token ("4TH")."""
    out: list[Token | None] = list(tokens)
    for i, t in enumerate(tokens):
        if out[i] is None or t.text not in ("1", "2", "3", "4"):
            continue
        for j, u in enumerate(tokens):
            if j == i or out[j] is None or u.text.upper() not in _ORDINAL:
                continue
            if t.x1 - 0.3 * t.h <= u.x0 <= t.x1 + 0.6 * t.h and t.y0 - 0.3 * t.h <= u.cy <= t.y1:
                out[i] = replace(t, text=t.text + u.text.upper(), y0=min(t.y0, u.y0), x1=u.x1, y1=max(t.y1, u.y1),
                                 conf=min(t.conf, u.conf), merged=t.merged or u.merged)
                out[j] = None
                break
    return [t for t in out if t is not None]


_MIN_SHARE = 0.3  # a token split out of a longer line belongs to a box covering this share of its estimated width


def _share(t: Token, box: Box | None) -> float:
    """How much of ``t`` lies in ``box``: 0 or 1 by its centre; for an estimated box, the share of its width."""
    if box is None:
        return 0.0
    bx, by, bw, bh = box
    if not by <= t.cy <= by + bh:
        return 0.0
    if not t.merged:
        return 1.0 if bx <= t.cx <= bx + bw else 0.0
    return max(0.0, min(t.x1, bx + bw) - max(t.x0, bx)) / max(t.w, 1e-6)


def _union(a: Box | None, b: Box | None) -> Box | None:
    if a is None or b is None:
        return a or b
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    return (x0, y0, max(a[0] + a[2], b[0] + b[2]) - x0, max(a[1] + a[3], b[1] + b[3]) - y0)


def _snap_merged(tokens: list[Token], p: ScorebugProfile) -> list[Token]:
    """Words split out of one long line ("LAL 109 GS 105 4TH 11:28 24") get boxes shared out by characters, which
    drift along the line. Within each such line the integers are matched, keeping their order, to the score and
    shot-clock boxes the line crosses (the order-keeping match nearest the estimates), and a clock or period word
    to its box; a matched word takes the box it is matched to. An integer whose box another match nearly as close
    (within three characters' width) would change is dropped, as are integers matched to no box: a missing read costs
    less than a wrong score."""
    out: list[Token | None] = list(tokens)
    lines: dict[int, list[int]] = {}
    for i, t in enumerate(tokens):
        if t.merged:
            lines.setdefault(t.line, []).append(i)
    for idx in lines.values():
        lx0, lx1 = min(tokens[i].x0 for i in idx), max(tokens[i].x1 for i in idx)
        cy = tokens[idx[0]].cy
        char_w = (lx1 - lx0) / max(1, sum(len(tokens[i].text) + 1 for i in idx) - 1)

        def crossed(b: Box | None, cy: float = cy, lx0: float = lx0, lx1: float = lx1) -> bool:
            return b is not None and b[1] <= cy <= b[1] + b[3] and b[0] < lx1 and b[0] + b[2] > lx0

        def snap(i: int, b: Box) -> None:
            out[i] = replace(tokens[i], x0=float(b[0]), x1=float(b[0] + b[2]), merged=False)

        fields = sorted((b[0] + b[2] / 2.0, name, b) for name, b in (
            ("away", p.away_score_box), ("home", p.home_score_box), ("shot", p.shot_clock_box))
            if b is not None and crossed(b))
        ints = [i for i in idx if tokens[i].text.isdigit() and len(tokens[i].text) <= 3]
        if fields and ints:
            n = min(len(ints), len(fields))
            matches: list[tuple[float, frozenset[tuple[int, int]]]] = []
            for ti in itertools.combinations(range(len(ints)), n):
                for fi in itertools.combinations(range(len(fields)), n):
                    cost = 0.0
                    for a, b in zip(ti, fi, strict=True):
                        t = tokens[ints[a]]
                        if fields[b][1] == "shot" and (len(t.text) > 2 or int(t.text) > 24):
                            cost = math.inf
                            break
                        cost += abs(t.cx - fields[b][0])
                    if math.isfinite(cost):
                        matches.append((cost, frozenset(zip(ti, fi, strict=True))))
            agreed: frozenset[tuple[int, int]] = frozenset()
            if matches:
                low = min(c for c, _ in matches)
                agreed = frozenset.intersection(*(m for c, m in matches if c - low < 3.0 * char_w))
            for a, b in agreed:
                snap(ints[a], fields[b][2])
            for a in set(range(len(ints))) - {a for a, _ in agreed}:
                out[ints[a]] = None
        for match, box in ((_clock, p.clock_box), (_period, p.period_box or p.clock_box)):
            words = [i for i in idx if match(tokens[i].text) is not None]
            if words and box is not None and crossed(box):
                centre = box[0] + box[2] / 2.0
                snap(min(words, key=lambda i: abs(tokens[i].cx - centre)), box)
    return [t for t in out if t is not None]


def _parse_boxes(tokens: list[Token], p: ScorebugProfile) -> dict[str, Any]:
    out: dict[str, Any] = dict.fromkeys(FIELDS)
    tokens = _snap_merged(join_ordinals(tokens), p)
    clock_tok: Token | None = None
    best: tuple[float, float] | None = None
    for t in tokens:
        c = _clock(t.text)
        s = _share(t, p.clock_box)
        if c is None or s < _MIN_SHARE:
            continue
        key = (-_share(t, p.shot_clock_box), s)  # a clock-like token in the shot clock's box: its tenths
        if best is None or key > best:
            best, clock_tok, out["game_clock"] = key, t, c
    panel = _union(p.period_box, p.clock_box)
    best_p = 0.0
    for t in tokens:
        v = _period(t.text)
        s = _share(t, panel)
        if v is not None and s >= _MIN_SHARE and s > best_p:
            best_p, out["period"] = s, v
    picks: dict[str, tuple[float, int, int]] = {}
    for t in tokens:
        if t is clock_tok or not t.text.isdigit() or len(t.text) > 3:
            continue
        v = int(t.text)
        cands = [("away_score", _share(t, p.away_score_box)), ("home_score", _share(t, p.home_score_box))]
        if v <= 24 and len(t.text) <= 2:
            cands.append(("shot_clock", _share(t, p.shot_clock_box)))
        name, s = max(cands, key=lambda c: c[1])
        if s < _MIN_SHARE:
            continue
        pick = (s, len(t.text), v)
        if name not in picks or pick[:2] > picks[name][:2]:
            picks[name] = pick
    for name, (_, _, v) in picks.items():
        out[name] = v
    return out


def parse_lines(lines: list[OCRLine], profile: ScorebugProfile, origin_x: float = 0.0,
                origin_y: float = 0.0) -> dict[str, Any]:
    """Fields of one bug read. ``lines`` are boxes in crop pixels; ``origin_x`` / ``origin_y`` is the crop's
    position in the frame (an x-range profile such as ``ESPN_2024`` only uses x)."""
    if profile.boxed:
        return _parse_boxes(line_tokens(lines, origin_x, origin_y), profile)
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
    clock_right: float | None = None
    for text, cx, right in tokens:
        lo, hi = profile.clock_panel_x
        if lo <= cx <= hi:
            c = _clock(text)
            if c is not None and out["game_clock"] is None:
                out["game_clock"], clock_right = c, right
                continue
            per = _period(text)
            if per is not None and out["period"] is None and (clock_right is None or cx < clock_right):
                out["period"] = per
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
    min_logo_std: float = 4.0  # a presence region this flat (grey level std) is not learned: no texture to match
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
        return frame[max(0, y):y + h, max(0, x):x + w]

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
        x0, y0 = max(0, x), max(0, y)
        crop = frame[y0:y + h, x0:x + w]
        return parse_lines(self.ocr.read(crop), prof, origin_x=float(x0), origin_y=float(y0))

    def _learn_logo(self, frame: np.ndarray) -> None:
        patch = self._logo_patch(frame)
        if patch.size and float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).std()) >= self.config.min_logo_std:
            self._logo = self._norm(patch)

    def read(self, frame: np.ndarray, t: float) -> ScoreboardReading | None:
        if self._logo is not None and not self.present(frame):
            return None
        f = self.read_fields(frame)
        if f["game_clock"] is None and f["period"] is None:
            return None
        if self._logo is None and f["game_clock"] is not None and f["period"] is not None:
            self._learn_logo(frame)  # a confident read: this frame shows the bug
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
