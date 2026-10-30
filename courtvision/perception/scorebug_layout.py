"""Scorebug layout discovery: where a broadcast's bug draws the clocks and the scores, from OCR of sample frames.

``scorebug_ocr.ESPN_2024`` was measured by hand on one ESPN game. Other broadcasters draw their bug elsewhere and
in other shapes (top or bottom, one row or two stacked team rows, tricodes or logos only, the shot clock in its own
box, other fonts and resolutions), so ``discover`` finds the layout from Vision reads of whole frames at ~20-40
moments of a game (``ocr_frame`` reads a frame in two overlapping bands):

1. Every recognised line is split into tokens, classed by pattern: clock-like (``M:SS``, ``SS.t``), period
   (``1ST``, ``Q3``, ``OT``), integer, team (any spelling of the given teams: ``GS`` / ``GSW`` / ``WARRIORS``) and
   other words.
2. Tokens of one class are grouped by place across samples. A bug field stays put while its number changes width,
   so two reads are one field when they share a row and height and their left edges, centres or right edges line up
   (a right-aligned score keeps its right edge, a centred one its centre).
3. Fields are told by how they behave over the game, not by where they are:
   * game clock: the clock-like group seen in the most moments that counts down within each period and, in frames
     a second apart, counts down about a second or stands still. A group beside it that never shows in the same
     frame is the same clock drawn elsewhere (scorebug_ocr.py notes that ESPN moves its clock while the shot clock
     is off);
   * period: period tokens beside the clock that never go back;
   * scores: two integer groups near the clock that never decrease (a few misreads allowed), take many values,
     share a font size, show together and stay within an NBA margin of each other;
   * shot clock: an integer group 0-24 next to the game clock, not a score, not monotone over the game, that in
     frames a second apart moves with the game clock (down while it runs, still while it is stopped, or a reset);
   * home and away: by the team tricodes next to the scores when ``teams`` are given and read; else the left (or
     top) score is *assumed* to be the away team (the "away @ home" reading order; a convention, not checked here)
     and the profile says so. ``side_votes`` can then check the sides against play-by-play scores.
4. The crop sent to the recogniser covers every field with a margin. The presence region is a static token of the
   bug (a tricode, else other text that never changes); with none it is guessed next to the away score and flagged
   (``presence_box`` picks one from pixels when frames are at hand).
5. A layout that does not hold up raises ``LayoutError`` naming the check that failed: too few moments, no
   clock-like group that counts down, no pair of never-decreasing scores beside it, fields that overlap or spread
   over more of the frame than a bug does.

Profiles save to and load from JSON (``save_profile`` / ``load_profile``, ``<game>/scorebug_profile.json``);
``find_saved_profile`` finds the one saved for a video.
"""

from __future__ import annotations

import bisect
import itertools
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .scorebug_ocr import (
    ESPN_2024,
    Box,
    ScorebugProfile,
    Token,
    _clock,
    _period,
    join_ordinals,
    line_tokens,
    parse_lines,
)
from .vision_ocr import OCRLine

__all__ = ["PROFILES", "PROFILE_FILE", "Discovery", "DiscoveryConfig", "LayoutError", "PresenceChoice", "discover",
           "discover_profile", "find_saved_profile", "load_profile", "load_saved_profile", "ocr_frame",
           "pbp_score_lookup", "presence_box", "presence_score", "resolve_profile", "save_profile", "side_votes",
           "swap_sides", "team_aliases"]

log = logging.getLogger(__name__)

PROFILE_FILE = "scorebug_profile.json"
PROFILES: dict[str, ScorebugProfile] = {"espn-2024": ESPN_2024}

FBox = tuple[float, float, float, float]  # x0, y0, x1, y1 in frame pixels

# Spellings of each team a bug may show, keyed by NBA.com tricode: the tricode, ESPN's abbreviation where it differs
# (GS, NY, NO, SA, UTAH, WSH), other sources' codes (BRK, CHO, PHO) and the nickname.
_TEAM_NAMES: dict[str, tuple[str, ...]] = {
    "ATL": ("ATL", "HAWKS"), "BOS": ("BOS", "CELTICS"), "BKN": ("BKN", "BRK", "NETS"),
    "CHA": ("CHA", "CHO", "HORNETS"), "CHI": ("CHI", "BULLS"), "CLE": ("CLE", "CAVS", "CAVALIERS"),
    "DAL": ("DAL", "MAVS", "MAVERICKS"), "DEN": ("DEN", "NUGGETS"), "DET": ("DET", "PISTONS"),
    "GSW": ("GSW", "GS", "WARRIORS"), "HOU": ("HOU", "ROCKETS"), "IND": ("IND", "PACERS"),
    "LAC": ("LAC", "CLIPPERS"), "LAL": ("LAL", "LAKERS"), "MEM": ("MEM", "GRIZZLIES"), "MIA": ("MIA", "HEAT"),
    "MIL": ("MIL", "BUCKS"), "MIN": ("MIN", "WOLVES", "TIMBERWOLVES"), "NOP": ("NOP", "NO", "PELICANS"),
    "NYK": ("NYK", "NY", "KNICKS"), "OKC": ("OKC", "THUNDER"), "ORL": ("ORL", "MAGIC"),
    "PHI": ("PHI", "76ERS", "SIXERS"), "PHX": ("PHX", "PHO", "SUNS"), "POR": ("POR", "BLAZERS"),
    "SAC": ("SAC", "KINGS"), "SAS": ("SAS", "SA", "SPURS"), "TOR": ("TOR", "RAPTORS"),
    "UTA": ("UTA", "UTAH", "JAZZ"), "WAS": ("WAS", "WSH", "WIZARDS"),
}


class LayoutError(ValueError):
    """No scorebug layout holds up over the samples; the message names the check that failed."""


@dataclass
class DiscoveryConfig:
    burst_gap_s: float = 3.0  # samples closer than this are frames of one moment (a pair a second apart)
    min_moments: int = 6  # fewer sampled moments: no discovery
    min_support: float = 0.15  # the game clock is read in at least this share of the moments (and 3)
    clock_countdown: float = 0.75  # share of steps within a period in which the game clock does not go up
    pair_consistent: float = 0.7  # share of frame pairs consistent with a running or stopped clock
    score_monotone: float = 0.85  # share of a score's reads on its longest never-decreasing run
    score_support: float = 0.4  # a score is read in at least this share of the game clock's moments
    max_margin: int = 60  # two scores further apart than this are no NBA score pair
    reach_x: float = 0.45  # the scores are within this share of the frame width of the game clock
    reach_y: float = 0.15  # and this share of its height
    max_bug_w: float = 0.75  # a bug's crop is at most this share of the frame width
    max_bug_h: float = 0.35  # and of its height
    static_support: float = 0.6  # presence text is read in at least this share of the game clock's moments


@dataclass
class Discovery:
    profile: ScorebugProfile
    report: list[str]  # what was found, one line per field
    reads: list[dict[str, Any] | None]  # per sample: its fields parsed with the profile (None: no field read)
    moments: list[int]  # per sample: its moment (frames a second apart share one)
    sides_by: str  # "tricode" | "position" (assumed)
    presence_by: str  # "text" | "guess"


class _Reader(Protocol):
    def read(self, image_bgr: np.ndarray) -> list[OCRLine]: ...


# --------------------------------------------------------------------------- geometry


def _fbox(t: Token) -> FBox:
    return (t.x0, t.y0, t.x1, t.y1)


def _funion(boxes: Iterable[FBox]) -> FBox:
    bs = list(boxes)
    return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))


def _gaps(a: FBox, b: FBox) -> tuple[float, float]:
    """Horizontal and vertical gaps between two boxes (0 where they overlap)."""
    return max(0.0, max(a[0], b[0]) - min(a[2], b[2])), max(0.0, max(a[1], b[1]) - min(a[3], b[3]))


def _overlaps(a: FBox, b: FBox) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _pad(b: FBox, px: float, py: float) -> FBox:
    return (b[0] - px, b[1] - py, b[2] + px, b[3] + py)


def _to_box(b: FBox, size: tuple[int, int]) -> Box:
    x0, y0 = max(0, math.floor(b[0])), max(0, math.floor(b[1]))
    x1, y1 = min(size[0], math.ceil(b[2])), min(size[1], math.ceil(b[3]))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _monotone_share(values: Sequence[float], increasing: bool = True) -> float:
    """Share of ``values`` on their longest non-decreasing (non-increasing) subsequence."""
    tails: list[float] = []
    for v in values:
        k = v if increasing else -v
        i = bisect.bisect_right(tails, k)
        if i == len(tails):
            tails.append(k)
        else:
            tails[i] = k
    return len(tails) / len(values) if values else 0.0


# --------------------------------------------------------------------------- tokens and groups


def team_aliases(code: str) -> frozenset[str]:
    """Every spelling of a team a bug may show (NBA tricode, ESPN abbreviation, nickname), from any one of them."""
    key = re.sub(r"[^0-9A-Z]", "", code.upper())
    for names in _TEAM_NAMES.values():
        if key in names:
            return frozenset(names)
    return frozenset({key})


@dataclass(frozen=True, eq=False)
class _Obs:
    s: int  # sample
    m: int  # moment
    t: float
    tok: Token
    kind: str  # "clock" | "period" | "int" | "team" | "word"
    value: Any  # seconds, period number, integer, side ("away" / "home") or normalised text


class _Group:
    """The reads of one place across samples (at most one per sample)."""

    def __init__(self, obs: Iterable[_Obs]) -> None:
        self.obs = sorted(obs, key=lambda o: (o.t, o.s))
        self.moments = {o.m for o in self.obs}
        self.samples = {o.s for o in self.obs}
        exact = [o for o in self.obs if not o.tok.merged]
        self.core = exact if len(exact) >= 2 else self.obs  # boxes of tokens split out of a line are estimates
        self.box = _funion(_fbox(o.tok) for o in self.core)
        self.h = float(np.median([o.tok.h for o in self.core]))
        self.cx = float(np.median([o.tok.cx for o in self.core]))
        self.cy = float(np.median([o.tok.cy for o in self.core]))

    @property
    def support(self) -> int:
        return len(self.moments)

    def values(self) -> list[Any]:
        return [o.value for o in self.obs]

    def at(self) -> dict[int, _Obs]:
        return {o.s: o for o in self.obs}

    def by_moment(self) -> dict[int, Any]:
        out: dict[int, Any] = {}
        for o in self.obs:
            out.setdefault(o.m, o.value)
        return out

    def text(self) -> str:
        return Counter(o.tok.text for o in self.obs).most_common(1)[0][0]

    def where(self) -> str:
        x0, y0, x1, y1 = self.box
        return f"x {x0:.0f}-{x1:.0f}, y {y0:.0f}-{y1:.0f}"


def _classify(tok: Token, sides: dict[str, frozenset[str]]) -> tuple[str, Any] | None:
    text = tok.text
    c = _clock(text)
    if c is not None:
        return "clock", c
    p = _period(text)
    if p is not None:
        return "period", p
    if text.isdigit() and len(text) <= 3:
        return "int", int(text)
    norm = re.sub(r"[^0-9A-Z]", "", text.upper())
    for side, names in sides.items():
        if norm in names:
            return "team", side
    if len(norm) >= 2 and re.search(r"[A-Z]", norm):
        return "word", norm
    return None


def _moments(times: Sequence[float], gap: float) -> list[int]:
    out: list[int] = []
    m, prev = -1, None
    for t in times:
        if prev is None or t - prev > gap:
            m += 1
        out.append(m)
        prev = t
    return out


def _observe(samples: Sequence[Sequence[OCRLine]], times: Sequence[float], moments: Sequence[int],
             sides: dict[str, frozenset[str]]) -> list[_Obs]:
    obs: list[_Obs] = []
    for s, lines in enumerate(samples):
        kept: list[Token] = []
        for tok in join_ordinals(line_tokens(list(lines))):
            kv = _classify(tok, sides)
            if kv is None:
                continue
            if any(u.text == tok.text and _iou(_fbox(u), _fbox(tok)) >= 0.5 for u in kept):
                continue  # the same word from overlapping reads of one frame
            kept.append(tok)
            obs.append(_Obs(s, moments[s], times[s], tok, *kv))
    return obs


def _iou(a: FBox, b: FBox) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _groups(obs: Sequence[_Obs]) -> list[_Group]:
    """Group reads of one class by place: the read with reads in the most other moments seeds a group of its
    neighbours (one per sample, the nearest), until no read has neighbours in two moments. Neighbours share a row
    (centres within 0.45 heights) and a height (within 0.6x) and line up at the left edge, centre or right edge
    (within 0.45 heights; 0.8 for a token split out of a longer line, whose box is an estimate)."""
    n = len(obs)
    if n < 2:
        return []
    x0 = np.array([o.tok.x0 for o in obs])
    x1 = np.array([o.tok.x1 for o in obs])
    y0 = np.array([o.tok.y0 for o in obs])
    y1 = np.array([o.tok.y1 for o in obs])
    cx, cy, h = (x0 + x1) / 2.0, (y0 + y1) / 2.0, np.maximum(y1 - y0, 1.0)
    merged = np.array([o.tok.merged for o in obs])
    hmax = np.maximum(h[:, None], h[None, :])
    link = np.minimum(h[:, None], h[None, :]) >= 0.6 * hmax
    link &= np.abs(cy[:, None] - cy[None, :]) <= 0.45 * hmax
    dx = np.minimum(np.minimum(np.abs(x0[:, None] - x0[None, :]), np.abs(x1[:, None] - x1[None, :])),
                    np.abs(cx[:, None] - cx[None, :]))
    link &= dx <= np.where(merged[:, None] | merged[None, :], 0.8, 0.45) * hmax
    samples = np.array([o.s for o in obs])
    _, minv = np.unique(np.array([o.m for o in obs]), return_inverse=True)
    onehot = np.zeros((n, int(minv.max()) + 1), np.int32)
    onehot[np.arange(n), minv] = 1
    alive = np.ones(n, bool)
    out: list[_Group] = []
    while True:
        lk = link & alive[None, :]
        dens = ((lk.astype(np.int32) @ onehot) > 0).sum(1) * 2 + (~merged).astype(np.int64)
        dens[~alive] = -1
        i = int(np.argmax(dens))
        if dens[i] < 4:  # no read with neighbours in two moments
            break
        members = []
        for s in np.unique(samples[lk[i]]):
            idx = np.flatnonzero(lk[i] & (samples == s))
            members.append(int(idx[np.argmin(np.abs(cx[idx] - cx[i]) + np.abs(cy[idx] - cy[i]))]))
        out.append(_Group(obs[j] for j in members))
        alive[members] = False
    return out


def _groups_by_value(obs: Sequence[_Obs]) -> list[_Group]:
    by: dict[Any, list[_Obs]] = {}
    for o in obs:
        by.setdefault(o.value, []).append(o)
    return [g for v in by.values() for g in _groups(v)]


def _absorb(main: _Group, others: Iterable[_Group], ok: Callable[[_Group], bool], max_gap: float) -> tuple[_Group, int]:
    """``main`` joined by the groups beside it (same row and height, within ``max_gap`` heights) that are never read
    in the same frame as it and keep ``ok``: one field drawn in two places. Returns it and how many places."""
    places = 1
    for o in sorted(others, key=lambda g: -g.support):
        if o is main or o.support < 2 or len(o.samples & main.samples) > 0.1 * len(o.samples):
            continue
        if abs(o.cy - main.cy) > 0.5 * main.h or not 0.7 <= o.h / main.h <= 1.43:
            continue
        if _gaps(o.box, main.box)[0] > max_gap * main.h:
            continue
        cand = _Group([*main.obs, *o.obs])
        if ok(cand):
            main, places = cand, places + 1
    return main, places


# --------------------------------------------------------------------------- fields


def _clock_stats(g: _Group, period_of: dict[int, int]) -> dict[str, int]:
    st = {"support": g.support, "distinct": len({round(float(o.value), 1) for o in g.obs}), "steps": 0, "down": 0,
          "unknown": 0, "ups": 0, "pairs": 0, "pairs_ok": 0}
    for a, b in itertools.pairwise(g.obs):
        d = float(b.value) - float(a.value)
        if a.m == b.m:  # frames of one moment, about a second apart: counting down by that, or stopped
            st["pairs"] += 1
            st["pairs_ok"] += -(b.t - a.t) - 1.05 <= d <= 0.05
            continue
        pa, pb = period_of.get(a.m), period_of.get(b.m)
        if pa is not None and pb is not None:
            if pa == pb:
                st["steps"] += 1
                st["down"] += d <= 0.05
            elif pb < pa:
                st["steps"] += 1  # the period went back: counts against
        else:
            st["unknown"] += 1
            st["ups"] += d > 0.05
    return st


def _clock_problem(st: dict[str, int], cfg: DiscoveryConfig, min_support: int) -> str | None:
    if st["support"] < min_support:
        return f"read in {st['support']} moments (needs {min_support})"
    if st["distinct"] < min(3, st["support"]):
        return "its value hardly changes (static text)"
    if st["steps"] >= 3 and st["down"] < cfg.clock_countdown * st["steps"]:
        return f"goes up within a period in {st['steps'] - st['down']} of {st['steps']} steps"
    if st["steps"] < 3 and st["unknown"] >= 3 and st["ups"] > max(2.0, 0.34 * st["unknown"]):
        return f"goes up in {st['ups']} of {st['unknown']} steps, more than period changes explain"
    if st["pairs"] >= 2 and st["pairs_ok"] < cfg.pair_consistent * st["pairs"]:
        return f"frames a second apart count down or stand still in only {st['pairs_ok']} of {st['pairs']} pairs"
    if st["steps"] + st["unknown"] + st["pairs"] < 2:
        return "too few reads to see it count down"
    return None


def _clock_quality(st: dict[str, int]) -> float:
    q = [st["down"] / st["steps"]] if st["steps"] else []
    q += [1.0 - st["ups"] / st["unknown"]] if st["unknown"] else []
    q += [st["pairs_ok"] / st["pairs"]] if st["pairs"] else []
    return float(np.mean(q)) if q else 0.0


def _near_period(clock: _Group, periods: Sequence[_Group]) -> _Group | None:
    best, key = None, None
    for p in periods:
        gx, gy = _gaps(p.box, clock.box)
        if gx > 12 * clock.h or gy > 3 * clock.h:
            continue
        co = len(p.moments & clock.moments)
        if co < 2 or _monotone_share(p.values()) < 0.8:
            continue
        k = (co, -(gx + gy))
        if key is None or k > key:
            best, key = p, k
    return best


def _find_clock(obs: Sequence[_Obs], cfg: DiscoveryConfig, min_support: int
                ) -> tuple[_Group, _Group | None, dict[str, int], int, list[str]]:
    clocks = sorted(_groups([o for o in obs if o.kind == "clock"]), key=lambda g: -g.support)
    periods = _groups([o for o in obs if o.kind == "period"])
    tried: list[str] = []
    passed: list[_Group] = []
    best: tuple[_Group, _Group | None, dict[str, int], float] | None = None
    for g in clocks:
        pg = _near_period(g, periods)
        st = _clock_stats(g, pg.by_moment() if pg else {})
        problem = _clock_problem(st, cfg, min_support)
        if problem is not None:
            tried.append(f"'{g.text()}' at {g.where()}: {problem}")
            continue
        passed.append(g)
        q = _clock_quality(st)
        if best is None or (g.support, q) > (best[0].support, best[3]):
            best = (g, pg, st, q)
    if best is None:
        why = "; ".join(tried[:3]) if tried else "no clock-like text (M:SS or SS.t) read in two moments"
        raise LayoutError(f"no token behaves like a game clock counting down: {why}")
    g, pg, st, _ = best
    period_of = pg.by_moment() if pg else {}
    g, places = _absorb(g, clocks, lambda c: _clock_problem(_clock_stats(c, period_of), cfg, min_support) is None,
                        max_gap=3.0)
    st = _clock_stats(g, period_of)
    # another place that also behaves like the game clock: a second graphic (worth a look, not used)
    others = [f"'{c.text()}' at {c.where()} ({c.support} moments)" for c in passed
              if c.support >= 0.5 * g.support and not c.samples <= g.samples]
    if pg is not None:
        pg, _ = _absorb(pg, periods, lambda c: _monotone_share(c.values()) >= 0.8, max_gap=2.0)
    return g, pg, st, places, others


def _find_scores(ints: Sequence[_Group], clock: _Group, cfg: DiscoveryConfig) -> tuple[_Group, _Group]:
    cands: list[tuple[_Group, float]] = []
    seen: list[str] = []
    for g in ints:
        vals = g.values()
        mono = _monotone_share(vals)
        if g.support >= 3:
            seen.append(f"'{g.text()}' at {g.where()}: {g.support} moments, {100 * mono:.0f}% never decreasing")
        if g.support < max(3.0, cfg.score_support * clock.support) or mono < cfg.score_monotone:
            continue
        if len(set(vals)) < min(3, g.support // 2) or len(g.moments & clock.moments) < 0.5 * g.support:
            continue
        cands.append((g, mono))
    best: tuple[_Group, _Group] | None = None
    key: tuple[float, float] | None = None
    for (a, ma), (b, mb) in itertools.combinations(cands, 2):
        if _overlaps(a.box, b.box) or not 0.7 <= a.h / b.h <= 1.43:
            continue
        both = a.samples & b.samples
        if len(both) < 0.5 * min(len(a.samples), len(b.samples)):
            continue
        va, vb = a.at(), b.at()
        if sum(abs(va[s].value - vb[s].value) <= cfg.max_margin for s in both) < 0.9 * len(both):
            continue
        spread = sum(math.log1p(max(g.values()) - min(g.values())) for g in (a, b))
        dist = (sum(_gaps(a.box, clock.box)) + sum(_gaps(b.box, clock.box))) / clock.h
        k = (len(both) * (1.0 + spread) * (ma + mb), -dist)
        if key is None or k > key:
            best, key = (a, b), k
    if best is None:
        why = "; ".join(seen[:4]) if seen else "no integer read in three moments near the clock"
        raise LayoutError(f"no pair of never-decreasing scores next to the game clock: {why}")
    return best


def _shot_pairs(g: _Group, clock: _Group) -> tuple[int, int]:
    """Frame pairs of one moment, and how many move with the game clock: down while it runs, still while it is
    stopped (a second's tick allowed), or a reset (up to 12 or more)."""
    at = clock.at()
    pairs = ok = 0
    for a, b in itertools.pairwise(g.obs):
        if a.m != b.m:
            continue
        pairs += 1
        dt, d = b.t - a.t, float(b.value - a.value)
        reset = d > 0 and b.value >= 12
        ca, cb = at.get(a.s), at.get(b.s)
        dc = None if ca is None or cb is None else float(cb.value) - float(ca.value)
        if dc is not None and dc <= -0.5:
            good = -dt - 1.05 <= d <= -0.5
        elif dc is not None:
            good = -1.05 <= d <= 0.05
        else:
            good = -dt - 1.05 <= d <= 0.05
        ok += good or reset
    return pairs, ok


def _find_shot(ints: Sequence[_Group], scores: tuple[_Group, _Group], clock: _Group,
               cfg: DiscoveryConfig) -> tuple[_Group | None, tuple[int, int]]:
    best: _Group | None = None
    key: tuple[float, float, float] | None = None
    stats = (0, 0)
    for g in ints:
        if g in scores or g.support < max(2.0, 0.15 * clock.support):
            continue
        gx, gy = _gaps(g.box, clock.box)
        if gx > 6 * clock.h or gy > 3 * clock.h:
            continue
        vals = g.values()
        if sum(v <= 24 for v in vals) < 0.9 * len(vals) or sum(len(o.tok.text) <= 2 for o in g.obs) < 0.9 * len(vals):
            continue
        if len(g.moments & clock.moments) < 0.8 * g.support or len(set(vals)) < min(3, g.support):
            continue
        if g.support >= 6 and max(_monotone_share(vals), _monotone_share(vals, increasing=False)) >= 0.95:
            continue  # a count that only goes one way over the game (timeouts, fouls) is no shot clock
        pairs, ok = _shot_pairs(g, clock)
        if pairs >= 2 and ok < cfg.pair_consistent * pairs:
            continue
        k = (float(g.support), ok / pairs if pairs else 0.5, -(gx + gy))
        if key is None or k > key:
            best, key, stats = g, k, (pairs, ok)
    if best is not None:
        best, _ = _absorb(best, [g for g in ints if g not in scores], lambda c: True, max_gap=1.5)
    return best, stats


def _sides(a: _Group, b: _Group, teams: dict[str, _Group]) -> tuple[_Group, _Group, str]:
    """(away, home, how): by the tricodes next to the scores, else the left (top) score is assumed away. A tricode is
    near a score when little separates them across and their centres share a row (stacked rows are close together,
    so a row apart weighs more than the same gap across)."""
    h = (a.h + b.h) / 2.0

    def d(s: _Group, t: _Group) -> float:
        return (_gaps(s.box, t.box)[0] + 2.0 * abs(s.cy - t.cy)) / h

    ta, th = teams.get("away"), teams.get("home")
    if ta is not None and th is not None:
        straight, crossed = d(a, ta) + d(b, th), d(a, th) + d(b, ta)
        if straight + 0.5 <= 0.8 * crossed:
            return a, b, "tricode"
        if crossed + 0.5 <= 0.8 * straight:
            return b, a, "tricode"
    elif ta is not None or th is not None:
        t = ta if ta is not None else th
        assert t is not None
        da, db = d(a, t), d(b, t)
        if da + 0.5 <= 0.6 * db or db + 0.5 <= 0.6 * da:
            near, far = (a, b) if da < db else (b, a)
            return (near, far, "tricode") if ta is not None else (far, near, "tricode")
    if abs(a.cy - b.cy) <= 0.5 * h:
        return (a, b, "position") if a.cx <= b.cx else (b, a, "position")
    return (a, b, "position") if a.cy <= b.cy else (b, a, "position")


def _field_box(g: _Group, max_chars: int, pad: float = 0.25) -> FBox:
    """The place of a field for numbers up to ``max_chars`` characters: the reads' extent, widened on the side
    away from its alignment edge (left, centre or right: whichever is steadiest over reads of different widths)."""
    core = g.core
    x0 = np.array([o.tok.x0 for o in core])
    x1 = np.array([o.tok.x1 for o in core])
    n = np.array([max(1, len(o.tok.text)) for o in core])
    cw = float(np.median((x1 - x0) / n))
    extra = max(0, max_chars - int(n.max())) * cw
    align = int(np.argmin([np.std(x0), np.std((x0 + x1) / 2.0), np.std(x1)])) if n.max() > n.min() else 1
    lo, hi = float(x0.min()), float(x1.max())
    if align == 0:
        hi += extra
    elif align == 2:
        lo -= extra
    else:
        lo, hi = lo - extra / 2.0, hi + extra / 2.0
    y0 = min(o.tok.y0 for o in core)
    y1 = max(o.tok.y1 for o in core)
    return (lo - pad * g.h, y0 - pad * g.h, hi + pad * g.h, y1 + pad * g.h)


def _separate(a: FBox, b: FBox, ca: FBox, cb: FBox) -> tuple[FBox, FBox]:
    """Two padded field boxes cut apart between their reads' extents ``ca`` / ``cb`` where the padding overlaps."""
    if not _overlaps(a, b):
        return a, b
    gx, gy = _gaps(ca, cb)
    if gx > 0 and gx >= gy:
        if ca[2] <= cb[0]:
            mid = (ca[2] + cb[0]) / 2.0
            return (a[0], a[1], min(a[2], mid), a[3]), (max(b[0], mid), b[1], b[2], b[3])
        mid = (cb[2] + ca[0]) / 2.0
        return (max(a[0], mid), a[1], a[2], a[3]), (b[0], b[1], min(b[2], mid), b[3])
    if gy > 0:
        if ca[3] <= cb[1]:
            mid = (ca[3] + cb[1]) / 2.0
            return (a[0], a[1], a[2], min(a[3], mid)), (b[0], max(b[1], mid), b[2], b[3])
        mid = (cb[3] + ca[1]) / 2.0
        return (a[0], max(a[1], mid), a[2], a[3]), (b[0], b[1], b[2], min(b[3], mid))
    raise LayoutError("two integer fields overlap: the reads do not separate into scores and a shot clock")


def _find_static(obs: Sequence[_Obs], teams: Iterable[_Group], core: FBox, h: float, dynamic: Sequence[FBox],
                 clock_support: int, cfg: DiscoveryConfig) -> _Group | None:
    """A token that never changes on the bug: a tricode beside a score first, else the widest other text within
    six text heights across and two down of the fields."""
    cands = [(True, g) for g in teams]
    for g in _groups_by_value([o for o in obs if o.kind == "word"]):
        gx, gy = _gaps(g.box, core)
        if gx <= 6 * h and gy <= 2 * h:
            cands.append((False, g))
    best, key = None, None
    for is_team, g in cands:
        if g.support < cfg.static_support * clock_support or any(_overlaps(g.box, f) for f in dynamic):
            continue
        k = (is_team, g.support, g.box[2] - g.box[0])
        if key is None or k > key:
            best, key = g, k
    return best


def _guess_logo(away: FBox, clock: FBox, h: float) -> FBox:
    """A box beside the away score on the side away from the clock, where bugs without tricodes tend to draw the
    team's logo: a guess, flagged as one."""
    w = 1.6 * h
    if (away[0] + away[2]) / 2.0 <= (clock[0] + clock[2]) / 2.0:
        return (away[0] - 0.2 * h - w, away[1], away[0] - 0.2 * h, away[3])
    return (away[2] + 0.2 * h, away[1], away[2] + 0.2 * h + w, away[3])


# --------------------------------------------------------------------------- discovery


def discover(samples: Sequence[Sequence[OCRLine]], frame_size: tuple[int, int],
             teams: tuple[str, str] | None = None, *, times: Sequence[float] | None = None,
             config: DiscoveryConfig | None = None) -> Discovery:
    """The scorebug layout behind OCR reads of whole frames (``samples``: lines with boxes in frame pixels, one list
    per frame, in time order) of a ``frame_size`` video; see the module docstring. ``teams``: the (away, home) team
    codes in any spelling (``("LAL", "GS")`` or ``("LAL", "GSW")``). ``times``: each frame's video time; frames
    within ``burst_gap_s`` are one moment, and their changes are checked second by second. Raises ``LayoutError``."""
    cfg = config or DiscoveryConfig()
    W, H = int(frame_size[0]), int(frame_size[1])
    n = len(samples)
    ts = [float(t) for t in times] if times is not None else [1e4 * i for i in range(n)]
    if len(ts) != n:
        raise ValueError(f"{len(ts)} times for {n} samples")
    if any(b < a for a, b in itertools.pairwise(ts)):
        raise ValueError("samples must be in time order")
    moments = _moments(ts, cfg.burst_gap_s)
    n_moments = moments[-1] + 1 if moments else 0
    if n_moments < cfg.min_moments:
        raise LayoutError(f"{n_moments} moments sampled; discovery needs at least {cfg.min_moments}")
    sides = {} if teams is None else {"away": team_aliases(teams[0]), "home": team_aliases(teams[1])}
    obs = _observe(samples, ts, moments, sides)
    min_support = max(3, math.ceil(cfg.min_support * n_moments))

    clock, period, cst, places, rivals = _find_clock(obs, cfg, min_support)
    cb = clock.box

    def near(o: _Obs) -> bool:
        gx, gy = _gaps(_fbox(o.tok), cb)
        return gx <= cfg.reach_x * W and gy <= cfg.reach_y * H

    ints = _groups([o for o in obs if o.kind == "int" and near(o)])
    s1, s2 = _find_scores(ints, clock, cfg)
    h_scores = (s1.h + s2.h) / 2.0
    core = _funion([s1.box, s2.box, clock.box])
    team_groups: dict[str, _Group] = {}
    for side in ("away", "home"):
        gs = [g for g in _groups_by_value([o for o in obs if o.kind == "team" and o.value == side])
              if _gaps(g.box, core)[0] <= 6 * h_scores and _gaps(g.box, core)[1] <= 3 * h_scores]
        if gs:
            team_groups[side] = max(gs, key=lambda g: g.support)
    away, home, sides_by = _sides(s1, s2, team_groups)
    shot, (pairs, pairs_ok) = _find_shot(ints, (s1, s2), clock, cfg)

    # each field's extent (for numbers up to their widest), its box a quarter height around it; the crop is the
    # extents and the tricodes beside them with a margin
    fields: dict[str, tuple[_Group, int]] = {"away": (away, 3), "home": (home, 3), "clock": (clock, 5)}
    if shot is not None:
        fields["shot"] = (shot, 2)
    if period is not None:
        fields["period"] = (period, 3)
    extents = {k: _field_box(g, chars, pad=0.0) for k, (g, chars) in fields.items()}
    boxes = {k: _pad(extents[k], 0.25 * g.h, 0.25 * g.h) for k, (g, _) in fields.items()}
    for k1, k2 in itertools.combinations([k for k in ("away", "home", "shot") if k in boxes], 2):
        boxes[k1], boxes[k2] = _separate(boxes[k1], boxes[k2], fields[k1][0].box, fields[k2][0].box)
    h_text = float(np.median([g.h for g, _ in fields.values()]))
    parts = list(extents.values()) + [g.box for g in team_groups.values()
                                      if _gaps(g.box, core)[0] <= 2 * h_scores and _gaps(g.box, core)[1] <= h_scores]
    crop = _pad(_funion(parts), 0.6 * h_text, 0.35 * h_text)
    if crop[2] - crop[0] > cfg.max_bug_w * W or crop[3] - crop[1] > cfg.max_bug_h * H:
        raise LayoutError(f"the fields spread over {(crop[2] - crop[0]) / W:.0%} x {(crop[3] - crop[1]) / H:.0%} of "
                          "the frame: more than one bug")
    static = _find_static(obs, team_groups.values(), _funion(boxes.values()), h_text, list(boxes.values()),
                          clock.support, cfg)
    presence_by = "text" if static is not None else "guess"
    if static is not None:
        logo = _pad(static.box, 0.15 * static.h, 0.15 * static.h)
    else:
        logo = _guess_logo(boxes["away"], boxes["clock"], h_scores)

    notes: list[str] = []
    if sides_by == "position":
        where = "left" if abs(away.cy - home.cy) <= 0.5 * h_scores else "top"
        why = "no team names were given" if not sides else "no tricode was read clearly beside one score each"
        notes.append(f"home/away ASSUMED from position: the {where} score is the away team ({why})")
    if presence_by == "guess":
        notes.append("presence region GUESSED beside the away score (no static text on the bug)")
    if period is None:
        notes.append("no period found: periods are read by pattern in the clock box")
    if shot is None:
        notes.append("no shot clock found")
    if places > 1:
        notes.append(f"the game clock is drawn in {places} places (its box covers them all)")
    size = (W, H)
    ib = {k: _to_box(b, size) for k, b in boxes.items()}
    xr = {k: (b[0], b[0] + b[2]) for k, b in ib.items()}
    panel = _to_box(_funion([boxes[k] for k in ("clock", "period", "shot") if k in boxes]), size)
    profile = ScorebugProfile(
        name="discovered", ocr_box=_to_box(crop, size), away_score_x=xr["away"], home_score_x=xr["home"],
        clock_panel_x=(panel[0], panel[0] + panel[2]), logo_box=_to_box(logo, size),
        away_first=bool(away.cx < home.cx if abs(away.cy - home.cy) <= 0.5 * h_scores else away.cy < home.cy),
        ref_size=size, away_score_box=ib["away"], home_score_box=ib["home"], clock_box=ib["clock"],
        period_box=ib.get("period"), shot_clock_box=ib.get("shot"), notes=tuple(notes))

    reads = [_read_frame(lines, profile) for lines in samples]
    full = sum(1 for r in reads if r is not None and None not in (r["game_clock"], r["away_score"], r["home_score"]))

    def label(side: str) -> str:
        t = team_groups.get(side)
        return f"{t.text()} " if t is not None else ""

    vals = {k: g.values() for k, g in (("away", away), ("home", home))}
    report = [f"{n} frames at {n_moments} moments ({W}x{H}); the bug's clock and scores read in {full} frames"]
    steps = f"{cst['down']}/{cst['steps']} steps within a period" if cst["steps"] else \
        f"up {cst['ups']} times in {cst['unknown']} steps"
    report.append(f"game clock: {clock.where()}, {clock.support} moments; counts down in {steps}; "
                  f"{cst['pairs_ok']}/{cst['pairs']} frame pairs a second apart consistent"
                  + (f"; drawn in {places} places" if places > 1 else ""))
    report.append(f"period: {period.where()}, {period.support} moments, values {sorted(set(period.values()))}"
                  if period is not None else "period: not found")
    for side, g in (("away", away), ("home", home)):
        report.append(f"{side} score: {label(side)}{g.where()}, {g.support} moments, values "
                      f"{min(vals[side])}-{max(vals[side])}, never decreasing in "
                      f"{100 * _monotone_share(vals[side]):.0f}% of reads")
    report.append(f"shot clock: {shot.where()}, {shot.support} moments, {pairs_ok}/{pairs} frame pairs move with "
                  "the game clock" if shot is not None else "shot clock: not found")
    report.append("home/away: by tricode" if sides_by == "tricode" else f"home/away: {notes[0]}")
    report.append(f"presence: the '{static.text()}' text, {static.where()}" if static is not None
                  else "presence: guessed beside the away score (no static text on the bug)")
    report.append(f"crop: x {profile.ocr_box[0]}-{profile.ocr_box[0] + profile.ocr_box[2]}, "
                  f"y {profile.ocr_box[1]}-{profile.ocr_box[1] + profile.ocr_box[3]}")
    report += [f"another clock-like place seen often: {r}" for r in rivals]
    return Discovery(profile, report, reads, moments, sides_by, presence_by)


def discover_profile(samples: Sequence[Sequence[OCRLine]], frame_size: tuple[int, int],
                     teams: tuple[str, str] | None = None, *, times: Sequence[float] | None = None,
                     config: DiscoveryConfig | None = None) -> ScorebugProfile:
    """``discover(...).profile``."""
    return discover(samples, frame_size, teams, times=times, config=config).profile


def _read_frame(lines: Sequence[OCRLine], profile: ScorebugProfile) -> dict[str, Any] | None:
    """The profile's read of whole-frame lines: those centred in its crop, parsed (None when no field is read)."""
    x, y, w, h = profile.ocr_box
    inside = [ln for ln in lines if x <= ln.center[0] <= x + w and y <= ln.center[1] <= y + h]
    f = parse_lines(inside, profile)
    return f if any(v is not None for v in f.values()) else None


# --------------------------------------------------------------------------- frames


def ocr_frame(ocr: _Reader, frame_bgr: np.ndarray, bands: Sequence[tuple[float, float]] = ((0.0, 0.55), (0.45, 1.0)),
              max_width: int = 1920) -> list[OCRLine]:
    """Recognised lines of a whole frame, boxes in frame pixels. The frame is read in horizontal bands (overlapping:
    a line cut by a band's inner edge is dropped there and read whole in the other band), each scaled down to at
    most ``max_width`` pixels wide; the same text read twice is kept once."""
    H, W = frame_bgr.shape[:2]
    scale = min(1.0, max_width / float(W))
    lines: list[OCRLine] = []
    for a, b in bands:
        y0, y1 = round(a * H), round(b * H)
        img = frame_bgr[y0:y1]
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for ln in ocr.read(img):
            x, y, w, h = (v / scale for v in ln.box)
            y += y0
            if (y0 > 0 and y <= y0 + 2) or (y1 < H and y + h >= y1 - 2):
                continue
            lines.append(OCRLine(ln.text, ln.conf, (x, y, w, h)))
    kept: list[OCRLine] = []
    for ln in sorted(lines, key=lambda ln: -ln.conf):
        lb = (ln.box[0], ln.box[1], ln.box[0] + ln.box[2], ln.box[1] + ln.box[3])
        if any(k.text.strip() == ln.text.strip() and
               _iou(lb, (k.box[0], k.box[1], k.box[0] + k.box[2], k.box[1] + k.box[3])) >= 0.5 for k in kept):
            continue
        kept.append(ln)
    return sorted(kept, key=lambda ln: (ln.box[1], ln.box[0]))


@dataclass(frozen=True)
class PresenceChoice:
    box: Box  # frame pixels
    shown: float  # 10th percentile of the correlation with the bug shown
    hidden: float | None  # 90th percentile with the bug hidden (None: no such frames)

    @property
    def margin(self) -> float:
        return self.shown - (0.5 if self.hidden is None else self.hidden)


def _box_mean(img: np.ndarray, ww: int, wh: int) -> np.ndarray:
    return cv2.boxFilter(img, cv2.CV_32F, (ww, wh), normalize=True, borderType=cv2.BORDER_REPLICATE)


def _ncc_maps(stack: np.ndarray, ref: np.ndarray, ww: int, wh: int) -> np.ndarray:
    """Per image of ``stack``, the normalised correlation with ``ref`` of the ``ww`` x ``wh`` window centred at each
    pixel (box-filter moments)."""
    mR = _box_mean(ref, ww, wh)
    varR = _box_mean(ref * ref, ww, wh) - mR * mR
    out = []
    for img in stack:
        mP, mPP, mPR = _box_mean(img, ww, wh), _box_mean(img * img, ww, wh), _box_mean(img * ref, ww, wh)
        out.append((mPR - mP * mR) / np.sqrt(np.maximum((mPP - mP * mP) * varR, 1e-6)))
    return np.stack(out)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / n) if n > 1e-6 else 0.0


def presence_score(shown: Sequence[np.ndarray], hidden: Sequence[np.ndarray], origin: tuple[int, int],
                   box: Box) -> PresenceChoice | None:
    """How well ``box`` (frame pixels, inside the crops at ``origin``) tells the bug shown from hidden: the 10th
    percentile of its patches' correlation with their median over ``shown``, the 90th over ``hidden``. None when
    the box leaves the crops or there are fewer than 3 shown crops."""
    x, y = box[0] - origin[0], box[1] - origin[1]
    if len(shown) < 3 or x < 0 or y < 0 or y + box[3] > shown[0].shape[0] or x + box[2] > shown[0].shape[1]:
        return None

    def patches(crops: Sequence[np.ndarray]) -> list[np.ndarray]:
        return [cv2.cvtColor(c[y:y + box[3], x:x + box[2]], cv2.COLOR_BGR2GRAY).astype(np.float32) for c in crops]

    ps = patches(shown)
    ref = np.median(np.stack(ps), axis=0)
    q_shown = float(np.quantile([_ncc(p, ref) for p in ps], 0.1))
    q_hidden = float(np.quantile([_ncc(p, ref) for p in patches(hidden)], 0.9)) if hidden else None
    return PresenceChoice(tuple(box), q_shown, q_hidden)  # type: ignore[arg-type]


def presence_box(shown: Sequence[np.ndarray], hidden: Sequence[np.ndarray], origin: tuple[int, int],
                 profile: ScorebugProfile, text_h: float | None = None) -> PresenceChoice | None:
    """The presence region that best tells frames showing the bug from frames without it. ``shown`` / ``hidden``:
    crops (BGR, equal sizes) of one region at frame position ``origin``, from frames where the bug was read and
    where nothing was. Each window (a few text heights wide) is scored by the 10th percentile of its normalised
    correlation with the median shown crop over ``shown`` minus the 90th percentile over ``hidden``; windows on the
    profile's number fields, or with too little texture to correlate, are skipped. None: no usable window."""
    if len(shown) < 3:
        return None
    if text_h is None:
        text_h = (profile.away_score_box or profile.ocr_box)[3] / 1.5
    s = min(1.0, 16.0 / max(text_h, 1.0))  # windows of a 16-pixel text height

    def prep(c: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else g
    P = np.stack([prep(c) for c in shown])
    A = np.stack([prep(c) for c in hidden]) if hidden else None
    M = np.median(P, axis=0)
    h, w = M.shape
    dyn = np.zeros((h, w), np.float32)
    for fb in (profile.away_score_box, profile.home_score_box, profile.clock_box, profile.period_box,
               profile.shot_clock_box):
        if fb is None:
            continue
        i0, j0 = math.floor((fb[0] - origin[0]) * s), math.floor((fb[1] - origin[1]) * s)
        i1, j1 = math.ceil((fb[0] + fb[2] - origin[0]) * s), math.ceil((fb[1] + fb[3] - origin[1]) * s)
        dyn[max(0, j0):max(0, j1), max(0, i0):max(0, i1)] = 1.0
    th = text_h * s  # text height in the scaled crops
    edges = cv2.magnitude(cv2.Sobel(M, cv2.CV_32F, 1, 0), cv2.Sobel(M, cv2.CV_32F, 0, 1))
    best: PresenceChoice | None = None
    for fw, fh in ((1.6, 1.0), (2.5, 1.0), (4.0, 1.2), (1.6, 1.6)):
        ww, wh = max(4, round(fw * th)), max(4, round(fh * th))
        if ww >= w or wh >= h:
            continue
        mM, mMM = _box_mean(M, ww, wh), _box_mean(M * M, ww, wh)
        varM = mMM - mM * mM
        ok = (varM >= 36.0) & (_box_mean(dyn, ww, wh) <= 0.05)  # texture: a grey level std of 6 or more
        ax, ay = ww // 2, wh // 2
        valid = np.zeros_like(ok)
        valid[ay:h - (wh - ay) + 1, ax:w - (ww - ax) + 1] = True
        ok &= valid
        if not ok.any():
            continue
        q_shown = np.quantile(_ncc_maps(P, M, ww, wh), 0.1, axis=0)
        q_hidden = np.quantile(_ncc_maps(A, M, ww, wh), 0.9, axis=0) if A is not None else None
        score = np.where(ok, q_shown - (0.5 if q_hidden is None else q_hidden), -np.inf)
        top = float(score.max())
        if not np.isfinite(top):
            continue
        # of the windows within 0.02 of the best margin, the most detailed (the most logo, the least plain panel)
        i = np.unravel_index(int(np.argmax(np.where(score >= top - 0.02, _box_mean(edges, ww, wh), -np.inf))),
                             score.shape)
        cyy, cxx = int(i[0]), int(i[1])
        x0, y0 = (cxx - ax) / s + origin[0], (cyy - ay) / s + origin[1]
        choice = PresenceChoice((round(x0), round(y0), round(ww / s), round(wh / s)), float(q_shown[i]),
                                None if q_hidden is None else float(q_hidden[i]))
        if best is None or choice.margin > best.margin:
            best = choice
    return best


# --------------------------------------------------------------------------- sides from play-by-play scores


def pbp_score_lookup(rows: Iterable[tuple[int, float, int, int]]) -> Callable[[int, float], tuple[int, int] | None]:
    """``rows``: (period, game clock seconds left, away score, home score) after each play -> the score at a
    (period, clock) of the game (None before the first row)."""
    keys = sorted((int(p), -float(c), int(a), int(h)) for p, c, a, h in rows)

    def at(period: int, clock: float) -> tuple[int, int] | None:
        i = bisect.bisect_right(keys, (int(period), -float(clock), math.inf, math.inf)) - 1
        return (keys[i][2], keys[i][3]) if i >= 0 else None

    return at


def side_votes(reads: Iterable[dict[str, Any] | None], score_at: Callable[[int, float], tuple[int, int] | None]
               ) -> tuple[int, int]:
    """(agree, swapped): reads whose (away, home) scores equal the true score at their period and clock, and reads
    where they equal it swapped. Reads of a tied score say nothing."""
    agree = swapped = 0
    for r in reads:
        if r is None or None in (r["period"], r["game_clock"], r["away_score"], r["home_score"]):
            continue
        truth = score_at(r["period"], r["game_clock"])
        if truth is None or truth[0] == truth[1]:
            continue
        got = (r["away_score"], r["home_score"])
        agree += got == truth
        swapped += got == (truth[1], truth[0])
    return agree, swapped


def swap_sides(profile: ScorebugProfile, note: str) -> ScorebugProfile:
    """The profile with its away and home scores exchanged (and ``note`` in place of a position assumption)."""
    notes = tuple(n for n in profile.notes if not n.startswith("home/away ASSUMED")) + (note,)
    return replace(profile, away_score_x=profile.home_score_x, home_score_x=profile.away_score_x,
                   away_score_box=profile.home_score_box, home_score_box=profile.away_score_box,
                   away_first=not profile.away_first, notes=notes)


# --------------------------------------------------------------------------- files


def save_profile(profile: ScorebugProfile, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
    """Write the profile as JSON (atomically); ``extra`` keys (a discovery report, its inputs) are stored beside it
    for people to read and ignored on loading."""
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps({**profile.to_dict(), **(extra or {})}, indent=1) + "\n")
    tmp.replace(p)
    return p


def load_profile(path: str | Path) -> ScorebugProfile:
    return ScorebugProfile.from_dict(json.loads(Path(path).read_text()))


def resolve_profile(spec: str) -> ScorebugProfile:
    """A built-in profile by name (``espn-2024``) or a saved one by its JSON path."""
    return PROFILES[spec] if spec in PROFILES else load_profile(spec)


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.exists() and b.exists() and a.resolve() == b.resolve()
    except OSError:
        return False


def find_saved_profile(*anchors: str | Path | None) -> Path | None:
    """The ``scorebug_profile.json`` saved for a video: next to one of ``anchors`` (files such as the video or its
    calibration), else in the NBA game folder (``runs/nba/<game>/``, where tools/nba_scan_scorebug.py saves it)
    whose ``game.json`` names one of them as its video."""
    files = [Path(a) for a in anchors if a]
    for f in files:
        if (f.parent / PROFILE_FILE).is_file():
            return f.parent / PROFILE_FILE
    for f in files:
        try:
            parents = list(f.resolve().parents)[:4]
        except OSError:
            continue
        for root in parents:
            for gj in sorted((root / "nba").glob("*/game.json")) if (root / "nba").is_dir() else []:
                try:
                    v = json.loads(gj.read_text()).get("video")
                except (OSError, ValueError, AttributeError):
                    continue
                if not v or not (gj.parent / PROFILE_FILE).is_file():
                    continue
                vp = Path(str(v))
                cands = [vp] if vp.is_absolute() else [root.parent / vp, root / vp, gj.parent / vp]
                if any(_same_file(c, f) for c in cands):
                    return gj.parent / PROFILE_FILE
    return None


def load_saved_profile(*anchors: str | Path | None) -> ScorebugProfile | None:
    """``find_saved_profile`` loaded (None when there is none, or it cannot be read: logged)."""
    path = find_saved_profile(*anchors)
    if path is None:
        return None
    try:
        prof = load_profile(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("could not read the scorebug profile %s: %s", path, exc)
        return None
    log.info("scorebug profile %s from %s", prof.name, path)
    return prof
