"""Scorebug layout discovery (perception/scorebug_layout.py) on synthetic Vision reads of several bug styles, the
boxed profiles of scorebug_ocr.py, and tools/nba_scan_scorebug.py end to end on a synthetic video (no Vision: a
stand-in recogniser reads block-coded text)."""

from __future__ import annotations

import importlib.util
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from courtvision.perception.scorebug_layout import (
    PROFILE_FILE,
    DiscoveryConfig,
    LayoutError,
    discover,
    discover_profile,
    find_saved_profile,
    load_profile,
    load_saved_profile,
    ocr_frame,
    pbp_score_lookup,
    presence_box,
    resolve_profile,
    save_profile,
    side_votes,
    swap_sides,
    team_aliases,
)
from courtvision.perception.scorebug_ocr import ESPN_2024, FIELDS, ScorebugProfile, parse_lines
from courtvision.perception.vision_ocr import OCRLine

TOOLS = Path(__file__).resolve().parents[1] / "tools"


# --------------------------------------------------------------------------- synthetic bugs and games


@dataclass(frozen=True)
class F:
    """A field drawn at a fixed place of a 1920x1080 frame: anchor x (its left edge, centre or right edge by
    ``align``), centre y, text height and character width."""

    x: float
    cy: float
    h: float
    cw: float
    align: str = "center"

    def box(self, text: str, s: float) -> tuple[float, float, float, float]:
        w = self.cw * len(text)
        x0 = {"left": self.x, "center": self.x - w / 2.0, "right": self.x - w}[self.align]
        return (x0 * s, (self.cy - self.h / 2.0) * s, w * s, self.h * s)


@dataclass(frozen=True)
class Style:
    away_score: F
    home_score: F
    period: F
    clock: F
    shot: F | None = None
    clock_off: F | None = None  # where the clock is drawn while the shot clock is off (ESPN moves it)
    away_tri: F | None = None
    home_tri: F | None = None
    quarter: str = "ord"  # "ord": 1ST, "q": Q1
    statics: tuple[tuple[str, F], ...] = ()
    merges: tuple[tuple[str, ...], ...] = ()  # fields Vision may read as one line
    merge: float = 0.0  # chance per frame of one such line
    ring_zero: bool = False  # a lone 0 is drawn as a ring the recogniser skips


# The ESPN-like bottom bar at the ESPN_2024 panels: positions and sizes of the Vision boxes of
# tests/test_nba_truth.py (crop of ESPN_2024.ocr_box moved to frame pixels). Where the clock goes while the shot
# clock is off is not measured: here it moves 57 px right, inside ESPN_2024's clock panel.
ESPN = Style(away_tri=F(612, 977, 28, 76 / 3, "left"), away_score=F(785, 977, 36, 38),
             home_tri=F(873, 979, 30, 30.5, "left"), home_score=F(1052, 977, 40, 40),
             period=F(1141, 980, 28, 50 / 3, "left"), clock=F(1288, 983, 36, 28.5), clock_off=F(1345, 983, 36, 28.5),
             shot=F(1434, 982, 32, 22.5, "right"),
             merges=(("period", "shot_clock"), ("away_tri", "away_score"), ("away_score", "home_tri", "home_score")),
             merge=0.2, ring_zero=True)
# two stacked team rows in the top-left corner, the period over the clock, the shot clock in its own box
STACKED = Style(away_tri=F(48, 60, 30, 21, "left"), away_score=F(236, 60, 34, 22, "right"),
                home_tri=F(48, 104, 30, 21, "left"), home_score=F(236, 104, 34, 22, "right"),
                period=F(300, 60, 26, 17), clock=F(300, 104, 30, 19), shot=F(392, 82, 30, 20), quarter="q",
                merges=(("away_tri", "away_score"),), merge=0.15)
# one row across the top, the home team on the left
TOP_HOME_LEFT = Style(home_tri=F(560, 50, 28, 24, "left"), home_score=F(700, 50, 34, 26, "right"),
                      period=F(760, 50, 26, 17, "left"), clock=F(905, 50, 32, 24), shot=F(1030, 50, 28, 20, "right"),
                      away_score=F(1060, 50, 34, 26, "left"), away_tri=F(1170, 50, 28, 24, "left"))


@dataclass(frozen=True)
class State:
    t: float
    period: int
    clock: float
    shot: int | None
    away: int
    home: int
    shown: bool


def game(n: int = 30, burst: int = 2, seed: int = 0, shown: float = 0.75, shot_off: float = 0.2,
         count_up: bool = False, random_scores: bool = False) -> list[State]:
    """``n`` moments of a game (``burst`` frames a second apart each), in time order."""
    rng = random.Random(seed)
    per = math.ceil(n / 4)
    out: list[State] = []
    away = home = 0
    for k in range(n):
        period = min(4, 1 + k // per)
        i = k - (period - 1) * per
        cnt = per if period < 4 else n - 3 * per
        frac = (i + rng.uniform(0.15, 0.85)) / cnt
        clock = 720.0 * (frac if count_up else 1.0 - frac)
        clock = round(clock, 1) if clock < 60 else float(int(clock))
        if k:
            away, home = away + rng.randint(0, 9), home + rng.randint(0, 9)
        if random_scores:
            away, home = rng.randint(0, 120), rng.randint(0, 120)
        shot = None if rng.random() < shot_off else rng.randint(1, 24)
        vis = rng.random() < shown
        t = 300.0 + 280.0 * k
        for j in range(burst):
            if j and rng.random() < 0.65:  # the clocks ran for the second
                clock = round(clock + 1.0, 1) if count_up else max(0.0, round(clock - 1.0, 1))
                shot = None if shot is None else max(0, shot - 1)
            out.append(State(t + j, period, clock, shot, away, home, vis))
    return out


def clock_text(c: float) -> str:
    return f"{int(c) // 60}:{int(c) % 60:02d}" if c >= 60 else f"{c:.1f}"


def period_text(p: int, style: str) -> str:
    return f"Q{p}" if style == "q" else ("1ST", "2ND", "3RD", "4TH")[p - 1]


def render(style: Style, st: State, rng: random.Random, s: float = 1.0, teams: tuple[str, str] = ("LAL", "GS"),
           drop: float = 0.03) -> tuple[list[OCRLine], dict[str, Any]]:
    """Vision's lines of the bug in one frame (frame pixels, scale ``s`` of 1920x1080) and the fields they show."""
    if not st.shown:
        return [], {}
    put: dict[str, tuple[str, tuple[float, float, float, float]]] = {}
    expected: dict[str, Any] = {}

    def add(name: str, text: str | None, f: F | None, value: Any = None) -> None:
        if text is None or f is None or rng.random() < drop:
            return
        x, y, w, h = f.box(text, s)
        j = rng.uniform(-1.0, 1.0) * s
        put[name] = (text, (x + j, y + 0.5 * j, w, h))
        if name in FIELDS:
            expected[name] = value

    add("away_score", None if style.ring_zero and st.away == 0 else str(st.away), style.away_score, st.away)
    add("home_score", None if style.ring_zero and st.home == 0 else str(st.home), style.home_score, st.home)
    add("game_clock", clock_text(st.clock), style.clock_off if st.shot is None and style.clock_off else style.clock,
        st.clock)
    add("period", period_text(st.period, style.quarter), style.period, st.period)
    add("shot_clock", None if st.shot is None else str(st.shot), style.shot, st.shot)
    add("away_tri", teams[0], style.away_tri)
    add("home_tri", teams[1], style.home_tri)
    for i, (text, f) in enumerate(style.statics):
        add(f"static{i}", text, f)
    groups = [m for m in style.merges if all(k in put for k in m)]
    if groups and rng.random() < style.merge:  # Vision reads neighbouring fields as one line
        names = sorted(rng.choice(groups), key=lambda k: put[k][1][0])
        bs = [put[k][1] for k in names]
        x0, y0 = min(b[0] for b in bs), min(b[1] for b in bs)
        x1, y1 = max(b[0] + b[2] for b in bs), max(b[1] + b[3] for b in bs)
        text = " ".join(put.pop(k)[0] for k in names)
        put["merged"] = (text, (x0, y0, x1 - x0, y1 - y0))
    lines = [OCRLine(text, 0.9, box) for text, box in put.values()]
    rng.shuffle(lines)
    return lines, expected


def noise(st: State, rng: random.Random, s: float = 1.0) -> list[OCRLine]:
    """Other text of a broadcast frame: a sponsor, BONUS, timeouts, a player graphic, a replay timer, a promo time,
    jersey numbers, the backboard's shot clock unit (same clocks, moving with the camera), a ticker of other games."""
    out: list[OCRLine] = []

    def add(text: str, x: float, cy: float, h: float, cw: float) -> None:
        out.append(OCRLine(text, 0.8, (x * s, (cy - h / 2.0) * s, cw * len(text) * s, h * s)))

    if st.shown:
        add("STATE FARM", 260, 977, 22, 14)
        if rng.random() < 0.3:
            add("BONUS", 873, 1008, 14, 10)
        add(str(max(0, 7 - int(st.t // 1500))), 600, 1003, 14, 10)  # timeouts left
    if rng.random() < 0.2:
        add("STEPHEN CURRY", 300, 880, 26, 15)
        add(f"{rng.randint(10, 40)} PTS", 520, 880, 26, 15)
    if not st.shown and rng.random() < 0.6:
        add(f"0:{int(st.t) % 9:02d}", 1700, 80, 30, 18)  # a replay's timer, counting up
    if rng.random() < 0.4:
        add("7:30", 1600, 1040, 24, 16)  # a promo's start time
    for _ in range(3):
        add(str(rng.randint(0, 99)), rng.uniform(200, 1700), rng.uniform(300, 800), rng.uniform(14, 30), 12)
    if st.shown and rng.random() < 0.3:  # the backboard's shot clock unit
        x, y = rng.uniform(400, 1500), rng.uniform(150, 400)
        add(clock_text(st.clock), x, y, 14, 9)
        if st.shot is not None:
            add(str(st.shot), x + 10, y - 30, 30, 18)
    if rng.random() < 0.3:
        add("BOS 98 NYK 95", rng.uniform(0, 1500), 1062, 18, 11)
    return out


def frames(style: Style, states: list[State], seed: int = 1, s: float = 1.0, teams: tuple[str, str] = ("LAL", "GS"),
           with_noise: bool = False) -> tuple[list[list[OCRLine]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    samples, truth = [], []
    for st in states:
        lines, expected = render(style, st, rng, s, teams)
        if with_noise:
            lines += noise(st, rng, s)
        samples.append(lines)
        truth.append(expected)
    return samples, truth


def crop_read(lines: list[OCRLine], profile: ScorebugProfile) -> dict[str, Any]:
    """What the scan reads with ``profile``: the lines inside its crop, in crop pixels, parsed."""
    x, y, w, h = profile.ocr_box
    inside = [OCRLine(ln.text, ln.conf, (ln.box[0] - x, ln.box[1] - y, ln.box[2], ln.box[3])) for ln in lines
              if x <= ln.center[0] <= x + w and y <= ln.center[1] <= y + h]
    return parse_lines(inside, profile, origin_x=x, origin_y=y)


def round_trip_errors(samples: list[list[OCRLine]], truth: list[dict[str, Any]], profile: ScorebugProfile
                      ) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    out = []
    for i, (lines, expected) in enumerate(zip(samples, truth, strict=True)):
        got = crop_read(lines, profile)
        want = {k: expected.get(k) for k in FIELDS}
        if got != want:
            out.append((i, got, want))
    return out


def inside(b: tuple[int, int, int, int], x: float, y: float) -> bool:
    return b[0] <= x <= b[0] + b[2] and b[1] <= y <= b[1] + b[3]


def edges(b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (b[0], b[1], b[0] + b[2], b[1] + b[3])


def overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = edges(a), edges(b)
    return min(ax1, bx1) > max(ax0, bx0) and min(ay1, by1) > max(ay0, by0)


def times_of(states: list[State]) -> list[float]:
    return [s.t for s in states]


BOXES = ("ocr_box", "logo_box", "away_score_box", "home_score_box", "clock_box", "period_box", "shot_clock_box")

# Vision's reads of ESPN_2024's crop on the gold frames (tests/test_nba_truth.py), crop pixels
ESPN_READS = [
    [OCRLine("LAL", 1.0, (22, 13, 76, 28)), OCRLine("109", 1.0, (138, 9, 114, 36)),
     OCRLine("GS", 1.0, (283, 14, 61, 30)), OCRLine("105", 1.0, (402, 7, 120, 40)),
     OCRLine("4TH", 1.0, (551, 16, 50, 28)), OCRLine("48.4", 1.0, (641, 15, 114, 36)),
     OCRLine("24", 1.0, (799, 16, 45, 32))],
    [OCRLine("2ND 5", 1.0, (552, 15, 290, 30)), OCRLine("11:28", 1.0, (634, 15, 150, 29))],
    [OCRLine("4", 1.0, (560, 16, 30, 28)), OCRLine("11:28", 1.0, (634, 15, 150, 29))],
]


# --------------------------------------------------------------------------- discovery


def test_espn_like_bar_gives_a_profile_equivalent_to_espn_2024() -> None:
    states = game(30, 2, seed=3)
    samples, truth = frames(ESPN, states, seed=4)
    d = discover(samples, (1920, 1080), ("LAL", "GS"), times=times_of(states))
    p, e = d.profile, ESPN_2024
    assert p.boxed and p.ref_size == (1920, 1080) and d.sides_by == "tricode" and d.presence_by == "text"
    assert all(abs(a - b) <= 10 for a, b in zip(edges(p.ocr_box), edges(e.ocr_box), strict=True))
    assert e.away_score_x[0] <= p.away_score_x[0] < p.away_score_x[1] <= e.away_score_x[1]
    assert e.home_score_x[0] <= p.home_score_x[0] < p.home_score_x[1] <= e.home_score_x[1]
    assert e.clock_panel_x[0] <= p.clock_panel_x[0] < p.clock_panel_x[1] <= e.clock_panel_x[1]
    assert p.away_first and inside(p.logo_box, 650, 977)  # presence: the away tricode
    # the synthetic ESPN bar moves its clock while the shot clock is off: one clock field drawn in two places
    assert p.clock_box is not None and inside(p.clock_box, 1288, 983) and inside(p.clock_box, 1345, 983)
    assert any("2 places" in n for n in p.notes)
    # the gold-frame reads (one of them a merged line "2ND 5") give the fields ESPN_2024 gives
    for lines in ESPN_READS:
        moved = [OCRLine(ln.text, ln.conf, (ln.box[0] + e.ocr_box[0], ln.box[1] + e.ocr_box[1], *ln.box[2:]))
                 for ln in lines]
        assert crop_read(moved, p) == parse_lines(lines, e, origin_x=e.ocr_box[0])
    assert round_trip_errors(samples, truth, p) == []
    assert [r is not None for r in d.reads] == [st.shown for st in states] and d.moments[:4] == [0, 0, 1, 1]


def test_stacked_corner_bug_with_its_own_shot_clock_box() -> None:
    states = game(30, 2, seed=5)
    samples, truth = frames(STACKED, states, seed=6, teams=("LAL", "GSW"))  # the bug shows NBA tricodes
    d = discover(samples, (1920, 1080), ("LAL", "GS"), times=times_of(states))  # given in ESPN's style
    p = d.profile
    assert d.sides_by == "tricode" and p.notes == ()
    a, h = p.away_score_box, p.home_score_box
    assert a is not None and h is not None and p.shot_clock_box is not None and p.clock_box is not None
    assert abs(a[0] - h[0]) <= 2 and a[1] + a[3] <= h[1] + 1  # one column, the away row on top
    assert not overlap(p.shot_clock_box, p.clock_box) and inside(p.shot_clock_box, 392, 82)
    assert p.ocr_box[0] < 48 and p.ocr_box[1] < 45 and p.ocr_box[0] + p.ocr_box[2] > 412
    assert round_trip_errors(samples, truth, p) == []


def test_top_bar_with_the_home_team_left_by_tricode_and_assumed_without_names() -> None:
    states = game(30, 2, seed=7)
    samples, truth = frames(TOP_HOME_LEFT, states, seed=8)
    d = discover(samples, (1920, 1080), ("LAL", "GSW"), times=times_of(states))
    p = d.profile
    assert p.home_score_box is not None and p.away_score_box is not None
    assert d.sides_by == "tricode" and not p.away_first and p.home_score_box[0] < p.away_score_box[0]
    assert p.ocr_box[1] < 30 and round_trip_errors(samples, truth, p) == []
    # without team names the left score is assumed to be the away team: wrong here, and the profile says so
    d2 = discover(samples, (1920, 1080), None, times=times_of(states))
    assert d2.sides_by == "position" and any(n.startswith("home/away ASSUMED") for n in d2.profile.notes)
    errs = round_trip_errors(samples, truth, d2.profile)
    assert errs and all(got["away_score"] == want["home_score"] for _, got, want in errs if want["home_score"])


def test_logos_only_bug_assumes_the_away_team_left_and_says_so() -> None:
    logos = replace(ESPN, away_tri=None, home_tri=None, merges=(("period", "shot_clock"),))
    states = game(30, 2, seed=9)
    samples, truth = frames(logos, states, seed=10)
    for teams in (("LAL", "GS"), None):
        d = discover(samples, (1920, 1080), teams, times=times_of(states))
        assert d.sides_by == "position" and d.presence_by == "guess"
        assert any(n.startswith("home/away ASSUMED from position: the left score") for n in d.profile.notes)
        assert any(n.startswith("presence region GUESSED") for n in d.profile.notes)
        assert round_trip_errors(samples, truth, d.profile) == []  # ESPN draws the away team left
    # text that never changes on the bug (a sponsor's wordmark) becomes the presence region
    sponsor = replace(logos, statics=(("STATE FARM", F(1000, 1018, 18, 11)),))
    samples, truth = frames(sponsor, states, seed=10)
    d = discover(samples, (1920, 1080), ("LAL", "GS"), times=times_of(states))
    assert d.presence_by == "text" and inside(d.profile.logo_box, 1000, 1018)
    assert round_trip_errors(samples, truth, d.profile) == []


@pytest.mark.parametrize("style,shown", [(ESPN, ("LAL", "GS")), (STACKED, ("LAL", "GSW")),
                                         (TOP_HOME_LEFT, ("LAL", "GS"))])
def test_noise_takes_no_field(style: Style, shown: tuple[str, str]) -> None:
    """Sponsor text, BONUS, timeouts, a player graphic, a replay timer counting up, a promo's "7:30", jersey
    numbers, the backboard's clocks moving with the camera and a ticker of other games."""
    states = game(30, 2, seed=11)
    clean, _ = frames(style, states, seed=12, teams=shown)
    noisy, truth = frames(style, states, seed=12, teams=shown, with_noise=True)
    p0 = discover_profile(clean, (1920, 1080), ("LAL", "GS"), times=times_of(states))
    d1 = discover(noisy, (1920, 1080), ("LAL", "GS"), times=times_of(states))
    p1 = d1.profile
    for k in BOXES[:1] + BOXES[2:]:
        assert all(abs(a - b) <= 4 for a, b in zip(getattr(p0, k), getattr(p1, k), strict=True)), k
    tricodes = [f for f in (style.away_tri, style.home_tri) if f is not None]
    assert d1.presence_by == "text" and any(inside(p1.logo_box, f.x + f.cw, f.cy) for f in tricodes)
    assert round_trip_errors(noisy, truth, p1) == []


@pytest.mark.parametrize("s,size", [(2 / 3, (1280, 720)), (2.0, (3840, 2160))])
def test_720p_and_4k_give_the_1080p_profile_scaled(s: float, size: tuple[int, int]) -> None:
    states = game(30, 2, seed=13)
    ref = discover_profile(frames(ESPN, states, seed=14)[0], (1920, 1080), ("LAL", "GS"), times=times_of(states))
    samples, truth = frames(ESPN, states, seed=14, s=s, with_noise=True)
    p = discover_profile(samples, size, ("LAL", "GS"), times=times_of(states))
    assert p.ref_size == size and round_trip_errors(samples, truth, p) == []
    q = p.scaled((1920, 1080))
    for k in BOXES:
        assert all(abs(a - b) <= 4 for a, b in zip(getattr(q, k), getattr(ref, k), strict=True)), k


def test_frames_without_pairs_and_fewer_moments() -> None:
    states = game(12, 1, seed=15, shown=0.8)
    samples, truth = frames(ESPN, states, seed=16, with_noise=True)
    p = discover_profile(samples, (1920, 1080), ("LAL", "GS"), times=times_of(states))
    assert round_trip_errors(samples, truth, p) == []
    p2 = discover_profile(samples, (1920, 1080), ("LAL", "GS"))  # no times: every sample its own moment
    assert p2.ocr_box == p.ocr_box


@pytest.mark.parametrize("kw,why", [
    ({"shown": 0.0}, "no token behaves like a game clock"),  # only the noise: replay timer, a promo's 7:30, ...
    ({"count_up": True}, "goes up within a period"),
    ({"random_scores": True}, "no pair of never-decreasing scores"),
    ({"n": 4}, "4 moments sampled"),
])
def test_layouts_that_do_not_hold_up_are_rejected(kw: dict[str, Any], why: str) -> None:
    states = game(**{"n": 30, **kw}, seed=17)
    samples, _ = frames(ESPN, states, seed=18, with_noise=True)
    with pytest.raises(LayoutError, match=why):
        discover(samples, (1920, 1080), ("LAL", "GS"), times=times_of(states))


def test_scores_and_clock_of_different_graphics_are_rejected() -> None:
    wide = Style(away_score=F(120, 1000, 36, 30), home_score=F(1800, 1000, 36, 30), period=F(900, 1000, 28, 17),
                 clock=F(1000, 1000, 36, 28))
    states = game(30, 2, seed=19)
    samples, _ = frames(wide, states, seed=20)
    with pytest.raises(LayoutError, match="more than one bug"):
        discover(samples, (1920, 1080), times=times_of(states))
    with pytest.raises(ValueError, match="time order"):
        discover(samples, (1920, 1080), times=times_of(states)[::-1])
    with pytest.raises(LayoutError):  # stricter checks can reject what the defaults accept
        discover(frames(ESPN, states, seed=20)[0], (1920, 1080), times=times_of(states),
                 config=DiscoveryConfig(min_support=0.95))


# --------------------------------------------------------------------------- boxed parsing and the reader


BOXED = ScorebugProfile("test", (0, 0, 500, 200), (150, 250), (150, 250), (250, 420), (0, 40, 30, 80),
                        away_score_box=(150, 40, 100, 40), home_score_box=(150, 84, 100, 40),
                        clock_box=(250, 84, 110, 40), period_box=(270, 40, 60, 40), shot_clock_box=(360, 60, 60, 44))


def test_boxed_parse_tells_fields_by_pattern_and_box() -> None:
    lines = [OCRLine("LAL 109", 0.9, (40, 44, 200, 32)), OCRLine("98", 0.9, (200, 88, 44, 32)),
             OCRLine("4", 0.9, (285, 46, 16, 30)), OCRLine("TH", 0.9, (303, 44, 18, 14)),  # a raised ordinal
             OCRLine("48.4", 0.9, (262, 88, 80, 32)), OCRLine("4.3", 0.9, (370, 66, 40, 32)),  # shot clock tenths
             OCRLine("3", 0.9, (20, 150, 10, 14))]  # timeouts: in no box
    assert parse_lines(lines, BOXED) == {"away_score": 109, "home_score": 98, "period": 4, "game_clock": 48.4,
                                         "shot_clock": None}
    shifted = [OCRLine(ln.text, ln.conf, (ln.box[0] - 10, ln.box[1] - 20, *ln.box[2:])) for ln in lines]
    assert parse_lines(shifted, BOXED, origin_x=10, origin_y=20) == parse_lines(lines, BOXED)
    two = [OCRLine("Q2 5:07", 0.9, (275, 50, 80, 30)), OCRLine("14", 0.9, (372, 66, 36, 32))]
    assert parse_lines(two, BOXED)["period"] == 2 and parse_lines(two, BOXED)["shot_clock"] == 14
    # an x-range profile ignores y, as before
    assert parse_lines(ESPN_READS[0], ESPN_2024, origin_x=590, origin_y=950) == \
        parse_lines(ESPN_READS[0], ESPN_2024, origin_x=590)


def test_a_merged_line_is_matched_to_the_boxes_it_crosses() -> None:
    """A whole bar read as one line: its words' boxes are only shared out by characters, so the integers are
    matched in order to the boxes the line crosses; one that could belong to two neighbouring boxes is dropped."""
    states = game(30, 2, seed=27)
    p = discover_profile(frames(TOP_HOME_LEFT, states, seed=28)[0], (1920, 1080), ("LAL", "GS"),
                         times=times_of(states))
    box = (560.0, 36.0, 682.0, 28.0)  # the home tricode's left edge to the away tricode's right edge
    full = parse_lines([OCRLine("GS 17 1ST 3:54 14 52 LAL", 0.9, box)], p)
    assert full == {"away_score": 52, "home_score": 17, "period": 1, "game_clock": 234.0, "shot_clock": 14}
    # the away score not read: is "1" the shot clock or the away score? Not a guess: dropped
    gap = parse_lines([OCRLine("GS 17 1ST 3:54 1 LAL", 0.9, box)], p)
    assert gap == {"away_score": None, "home_score": 17, "period": 1, "game_clock": 234.0, "shot_clock": None}
    # a number too big for a shot clock can only be the away score
    big = parse_lines([OCRLine("GS 17 1ST 3:54 31 LAL", 0.9, box)], p)
    assert big["away_score"] == 31 and big["shot_clock"] is None


def test_period_and_clock_spellings() -> None:
    from courtvision.perception.scorebug_ocr import _clock, _period

    assert [_period(t) for t in ("Q3", "3Q", "3rd", "OT", "2OT", "OT2", "OT3")] == [3, 3, 3, 5, 6, 6, 7]
    assert [_period(t) for t in ("1", "5TH", "Q5", "OT1", "QT")] == [None] * 5
    assert [_clock(t) for t in ("12:00", "0:48", "48.4", ":48.4", "0:48.4", "1:05.3")] == \
        [720.0, 48.0, 48.4, 48.4, 48.4, 65.3]
    assert [_clock(t) for t in ("13:00", "5:60", ":48", "1:2")] == [None] * 4


class CropOCR:
    """Stands in for Vision on crops of a frame whose lines are known in frame pixels: it returns the lines inside
    the crop it is sent (located by the ``origin`` the test expects), in crop pixels."""

    def __init__(self, lines: list[OCRLine], origin: tuple[int, int]) -> None:
        self.lines, self.origin = lines, origin

    def read(self, img: np.ndarray) -> list[OCRLine]:
        x0, y0 = self.origin
        h, w = img.shape[:2]
        return [OCRLine(ln.text, ln.conf, (ln.box[0] - x0, ln.box[1] - y0, *ln.box[2:])) for ln in self.lines
                if x0 <= ln.center[0] <= x0 + w and y0 <= ln.center[1] <= y0 + h]

    def close(self) -> None:
        pass


def test_reader_reads_a_discovered_profile_at_another_resolution() -> None:
    from courtvision.perception.scorebug_ocr import OCRScorebugReader

    states = game(30, 2, seed=21)
    p = discover_profile(frames(STACKED, states, seed=22, teams=("LAL", "GSW"))[0], (1920, 1080), ("LAL", "GS"),
                         times=times_of(states))
    st = State(0.0, 3, 431.0, 17, 77, 81, True)
    lines, expected = render(STACKED, st, random.Random(0), s=2 / 3, teams=("LAL", "GSW"), drop=0.0)
    scaled = p.scaled((1280, 720))
    reader = OCRScorebugReader(p, ocr=CropOCR(lines, scaled.ocr_box[:2]))  # type: ignore[arg-type]
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert reader.read_fields(frame) == {k: expected[k] for k in FIELDS}
    reading = reader.read(frame, 5.0)
    assert reading is not None and (reading.away_score, reading.home_score, reading.period) == (77, 81, 3)
    assert reader._logo is None  # a flat presence region (this black frame) is not learned


# --------------------------------------------------------------------------- profiles on disk


def test_profiles_save_and_load(tmp_path: Path) -> None:
    states = game(30, 2, seed=23)
    p = discover_profile(frames(ESPN, states, seed=24)[0], (1920, 1080), ("LAL", "GS"), times=times_of(states))
    path = save_profile(p, tmp_path / PROFILE_FILE)
    assert load_profile(path) == p and resolve_profile(str(path)) == p
    assert json.loads(path.read_text())["format"] == "courtvision.scorebug_profile/1"
    assert resolve_profile("espn-2024") is ESPN_2024
    assert ScorebugProfile.from_dict(ESPN_2024.to_dict()) == ESPN_2024 and not ESPN_2024.boxed
    assert ESPN_2024.scaled((1280, 720)).ocr_box == (393, 633, 580, 40)
    with pytest.raises(ValueError, match="format"):
        ScorebugProfile.from_dict({**p.to_dict(), "format": "something/9"})
    with pytest.raises(ValueError):
        ScorebugProfile.from_dict({**p.to_dict(), "ocr_box": [1, 2, 3]})


def test_the_studio_finds_the_profile_saved_for_its_video(tmp_path: Path) -> None:
    video = tmp_path / "runs" / "studio" / "abc123" / "source.video"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"\0")
    calib = video.with_name("calibration.json")
    assert find_saved_profile(video, calib) is None and load_saved_profile(video, None) is None
    game_dir = tmp_path / "runs" / "nba" / "2025-01-01-LAL-GSW"
    game_dir.mkdir(parents=True)
    (game_dir / "game.json").write_text(json.dumps({"video": "runs/studio/abc123/source.video", "espn_event": "1"}))
    in_game = replace(ESPN_2024, name="from-the-game-folder")
    save_profile(in_game, game_dir / PROFILE_FILE)
    assert find_saved_profile(None, str(calib), video) == game_dir / PROFILE_FILE
    assert load_saved_profile(str(video), str(calib)) == in_game
    other = tmp_path / "runs" / "studio" / "zzz" / "source.video"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"\0")
    assert find_saved_profile(other) is None  # another video: not this game's profile
    beside = replace(ESPN_2024, name="next-to-the-video")
    save_profile(beside, video.with_name(PROFILE_FILE))
    assert load_saved_profile(video, calib) == beside
    video.with_name(PROFILE_FILE).write_text("{")
    assert load_saved_profile(video) is None  # unreadable: logged, the studio keeps espn-2024


# --------------------------------------------------------------------------- frames, presence, sides


class BandOCR:
    """Stands in for Vision on bands of a known frame: finds which band (and scale) it was sent and returns the
    known lines visible in it (a line crossing the band's edge comes back cut, as a recogniser would read it)."""

    def __init__(self, frame: np.ndarray, lines: list[OCRLine], bands: list[tuple[float, float]], scale: float) -> None:
        self.frame, self.lines, self.bands, self.scale = frame, lines, bands, scale
        self.calls = 0

    def read(self, img: np.ndarray) -> list[OCRLine]:
        import cv2

        self.calls += 1
        H = self.frame.shape[0]
        for a, b in self.bands:
            y0, y1 = round(a * H), round(b * H)
            ref = self.frame[y0:y1]
            if self.scale < 1.0:
                ref = cv2.resize(ref, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            if ref.shape == img.shape and np.array_equal(ref, img):
                break
        else:
            raise AssertionError("not a band of the frame")
        out = []
        for ln in self.lines:
            x, y, w, h = ln.box
            top, bot = max(y, y0), min(y + h, y1)
            if bot <= top:
                continue
            out.append(OCRLine(ln.text if (top, bot) == (y, y + h) else ln.text[:2], ln.conf,
                               tuple(v * self.scale for v in (x, top - y0, w, bot - top))))  # type: ignore[arg-type]
        return out


@pytest.mark.parametrize("size", [(1920, 1080), (3840, 2160)])
def test_ocr_frame_reads_bands_in_frame_pixels(size: tuple[int, int]) -> None:
    W, H = size
    k = W / 1920
    frame = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    lines = [OCRLine("LAL", 0.9, (612 * k, 963 * k, 76 * k, 28 * k)),  # bottom band only
             OCRLine("11:28", 0.9, (100 * k, 40 * k, 140 * k, 30 * k)),  # top band only
             OCRLine("BONUS", 0.8, (900 * k, 530 * k, 120 * k, 20 * k)),  # in both bands: kept once
             OCRLine("STATE FARM", 0.8, (300 * k, 580 * k, 200 * k, 40 * k))]  # crosses the top band's edge
    bands = [(0.0, 0.55), (0.45, 1.0)]
    ocr = BandOCR(frame, lines, bands, min(1.0, 1920 / W))
    got = ocr_frame(ocr, frame, bands)
    assert ocr.calls == 2 and sorted(ln.text for ln in got) == ["11:28", "BONUS", "LAL", "STATE FARM"]
    for ln in got:
        want = next(w for w in lines if w.text == ln.text)
        assert np.allclose(ln.box, want.box, atol=1.0)


def test_presence_box_finds_the_static_part_of_the_bug() -> None:
    import cv2

    rng = np.random.default_rng(1)
    logo = rng.integers(0, 256, (30, 40, 3), dtype=np.uint8)  # a textured logo at x 20-60, y 15-45 of the crop

    def bug(score: int) -> np.ndarray:
        img = np.full((60, 400, 3), 40, np.uint8)
        img[15:45, 20:60] = logo
        cv2.putText(img, str(score), (200, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        noise = rng.integers(-4, 5, img.shape)
        return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

    shown = [bug(v) for v in (10, 23, 47, 88, 102, 115)]
    hidden = [rng.integers(0, 256, (60, 400, 3), dtype=np.uint8) for _ in range(4)]
    origin = (1000, 900)
    prof = replace(BOXED, away_score_box=(1190, 905, 90, 50), home_score_box=(1300, 905, 8, 8),
                   clock_box=(1310, 905, 8, 8), period_box=None, shot_clock_box=None)
    choice = presence_box(shown, hidden, origin, prof, text_h=30)
    assert choice is not None and choice.shown > 0.9 and choice.hidden is not None and choice.hidden < 0.5
    x, y, w, h = choice.box  # mostly on the logo (frame x 1020-1060, y 915-945)
    on_logo = max(0, min(x + w, 1060) - max(x, 1020)) * max(0, min(y + h, 945) - max(y, 915))
    assert on_logo >= 0.6 * w * h
    from courtvision.perception.scorebug_layout import presence_score

    flat = presence_score(shown, hidden, origin, (1000 + 100, 900 + 15, 40, 30))  # the plain panel
    assert flat is not None and flat.margin < choice.margin
    assert presence_score(shown, hidden, origin, (990, 900, 40, 30)) is None  # outside the crops
    assert presence_box(shown[:2], hidden, origin, prof) is None


def test_play_by_play_scores_check_home_and_away() -> None:
    at = pbp_score_lookup([(1, 720.0, 0, 0), (1, 700.0, 2, 0), (1, 650.0, 2, 3), (2, 720.0, 30, 28),
                           (2, 500.0, 35, 28)])
    assert [at(1, 710.0), at(1, 700.0), at(1, 600.0), at(2, 400.0)] == [(0, 0), (2, 0), (2, 3), (35, 28)]

    def r(period: int, clock: float, away: int, home: int) -> dict[str, Any]:
        return {"period": period, "game_clock": clock, "away_score": away, "home_score": home, "shot_clock": None}

    reads = [r(1, 690.0, 2, 0), r(2, 600.0, 28, 30), r(2, 450.0, 28, 35), None, r(1, 715.0, 0, 0),
             {**r(1, 690.0, 2, 0), "home_score": None}]
    assert side_votes(reads, at) == (1, 2)
    states = game(30, 2, seed=25)
    p = discover_profile(frames(ESPN, states, seed=26)[0], (1920, 1080), None, times=times_of(states))
    assert any(n.startswith("home/away ASSUMED") for n in p.notes)
    q = swap_sides(p, "home/away swapped by a test")
    assert (q.away_score_box, q.home_score_box, q.away_first) == (p.home_score_box, p.away_score_box, False)
    assert q.notes[-1] == "home/away swapped by a test" and not any("ASSUMED" in n for n in q.notes)


def test_team_aliases_cover_both_styles() -> None:
    assert team_aliases("GS") == team_aliases("gsw") >= {"GS", "GSW", "WARRIORS"}
    assert "NY" in team_aliases("NYK") and "UTAH" in team_aliases("UTA") and "WSH" in team_aliases("WAS")
    assert "SA" in team_aliases("SAS") and "NO" in team_aliases("NOP") and "BRK" in team_aliases("BKN")
    assert team_aliases("XYZ") == {"XYZ"}


# --------------------------------------------------------------------------- the scan tool on a synthetic video

ALPHA = "0123456789:.QLAGSW"  # the characters of the block code


def block_text(img: np.ndarray, text: str, x: int, y: int, align: str = "left", h: int = 20, cw: int = 9,
               gap: int = 4) -> None:
    """``text`` drawn as grey blocks, one per character, its grey level the character's code (``BlockOCR``)."""
    if align == "right":
        x -= len(text) * (cw + gap) - gap
    for i, ch in enumerate(text):
        x0 = x + i * (cw + gap)
        img[y:y + h, x0:x0 + cw] = 80 + 9 * ALPHA.index(ch)


class BlockOCR:
    """Stands in for Vision on ``block_text``: grey blocks are characters, neighbouring blocks one line."""

    def read(self, img: np.ndarray) -> list[OCRLine]:
        import cv2

        b, g, r = (img[..., k].astype(np.int16) for k in range(3))
        mask = ((np.abs(r - g) < 20) & (np.abs(g - b) < 20) & (g > 60)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        blocks = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if w < 5 or h < 12 or area < 0.6 * w * h:
                continue
            k = round((float(np.median(g[labels == i])) - 80) / 9)
            if 0 <= k < len(ALPHA):
                blocks.append((x, y, w, h, ALPHA[k]))
        words: list[list[tuple[int, int, int, int, str]]] = []
        for bl in sorted(blocks):
            for wd in words:
                last = wd[-1]
                if abs(last[1] + last[3] / 2 - bl[1] - bl[3] / 2) < bl[3] / 2 and 0 <= bl[0] - last[0] - last[2] <= 7:
                    wd.append(bl)
                    break
            else:
                words.append([bl])
        out = []
        for wd in words:
            x0, x1 = wd[0][0], wd[-1][0] + wd[-1][2]
            y0, y1 = min(v[1] for v in wd), max(v[1] + v[3] for v in wd)
            out.append(OCRLine("".join(v[4] for v in wd), 0.99, (float(x0), float(y0), float(x1 - x0), float(y1 - y0))))
        return out

    def close(self) -> None:
        pass


VW, VH, VDUR = 960, 540, 200  # 1 frame a second
REPLAYS = ((60, 75), (140, 160))  # no bug


def vstate(t: float) -> dict[str, Any]:
    t = float(int(t))
    return {"period": 1 + int(t // 50), "game_clock": float(720 - int(t % 50)), "shot_clock": 24 - int(t % 24),
            "away_score": int(t // 3), "home_score": int(t // 3.3), "shown": not any(a <= t < b for a, b in REPLAYS)}


def vframe(t: int, rng: np.random.Generator) -> np.ndarray:
    """A green floor; the bug in the bottom-left: a checkered logo, stacked LAL / GSW rows, the period over the
    clock, the shot clock beside them. A replay shows colour noise there instead."""
    img = np.zeros((VH, VW, 3), np.uint8)
    img[:] = (0, 90, 0)
    st = vstate(t)
    if not st["shown"]:
        img[400:540, 0:480] = rng.integers(0, 2, (140, 480, 1)) * np.array([200, 0, 200], np.uint8)
        return img
    img[440:520, 0:380] = 0
    cells = (np.indices((66, 24)).sum(0) // 6) % 2
    img[446:512, 6:30] = np.where(cells[..., None] == 1, np.array([0, 0, 220], np.uint8), np.array([220, 0, 0], np.uint8))
    block_text(img, "LAL", 40, 450)
    block_text(img, str(st["away_score"]), 200, 450, "right")
    block_text(img, "GSW", 40, 490)
    block_text(img, str(st["home_score"]), 200, 490, "right")
    block_text(img, f"Q{st['period']}", 230, 450)
    c = int(st["game_clock"])
    block_text(img, f"{c // 60}:{c % 60:02d}", 230, 490)
    block_text(img, str(st["shot_clock"]), 360, 470, "right")
    return img


def make_video(path: Path) -> None:
    rng = np.random.default_rng(0)
    proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                             f"{VW}x{VH}", "-r", "1", "-i", "pipe:0", "-c:v", "libx264", "-preset", "ultrafast", "-crf",
                             "6", "-g", "10", "-pix_fmt", "yuv420p", "-f", "mp4", str(path)], stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for t in range(VDUR):
        proc.stdin.write(vframe(t, rng).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg")
def test_scan_tool_discovers_a_new_layout_and_reads_the_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    video = tmp_path / "source.video"
    make_video(video)
    g = tmp_path / "runs" / "nba" / "2025-01-01-LAL-GSW"
    g.mkdir(parents=True)
    plays = [{"period": {"number": st["period"]},
              "clock": {"displayValue": f"{int(st['game_clock']) // 60}:{int(st['game_clock']) % 60:02d}"},
              "awayScore": st["away_score"], "homeScore": st["home_score"]} for st in map(vstate, range(VDUR))]
    (g / "espn_summary.json").write_text(json.dumps({"header": {"competitions": [{"competitors": [
        {"homeAway": "away", "team": {"abbreviation": "LAL"}}, {"homeAway": "home", "team": {"abbreviation": "GS"}}]}]},
        "plays": plays}))
    scan = _load("nba_scan_scorebug")
    monkeypatch.setattr(scan, "make_ocr", BlockOCR)
    out = g / "scorebug.jsonl.partial"  # where tools/nba_loop.py has it write
    assert scan.main([str(video), "--out", str(out), "--hz", "1", "--samples", "12"]) == 0
    printed = capsys.readouterr().out
    assert "teams LAL (away) @ GS (home)" in printed and "home/away: by tricode" in printed
    prof = load_profile(g / PROFILE_FILE)
    assert prof.boxed and prof.ref_size == (VW, VH) and str(g / PROFILE_FILE) in printed
    assert any(n.startswith("home/away confirmed by the ESPN play-by-play scores") for n in prof.notes)
    saved = json.loads((g / PROFILE_FILE).read_text())["discovery"]
    assert saved["teams"] == ["LAL", "GS"] and saved["samples"] == 12 and saved["report"][1].startswith("game clock")
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == VDUR and all(set(r) == {"t", "logo", *FIELDS} for r in rows)
    for r in rows:
        st = vstate(r["t"])
        if st["shown"]:
            assert {k: r[k] for k in FIELDS} == {k: st[k] for k in FIELDS}, r
            assert r["logo"] >= 0.9, r
        else:
            assert all(r[k] is None for k in FIELDS) and r["logo"] < 0.75, r
    assert inside(prof.logo_box, 57, 460)  # presence: the LAL tricode, kept after the check on pixels
    # a guessed presence box on plain floor is replaced by one the pixels show tells the bug from replays
    disc, times = scan.discover_video(str(video), (VW, VH), 0.0, float(VDUR), BlockOCR(), ("LAL", "GS"), 12, 2)
    guessed = replace(disc.profile, logo_box=(250, 424, 40, 12), notes=("presence region GUESSED beside the away score",))
    fixed = scan.refine_presence(str(video), guessed, disc, times, (VW, VH))
    assert fixed.logo_box != guessed.logo_box and fixed.notes[-1].startswith("presence box chosen on pixels")
    assert not any("GUESSED" in n for n in fixed.notes)
    # the saved profile reads the same; the hand-measured ESPN profile reads nothing of this bug
    again = tmp_path / "again.jsonl"
    assert scan.main([str(video), "--out", str(again), "--hz", "1", "--duration", "20", "--profile",
                      str(g / PROFILE_FILE)]) == 0
    assert again.read_text().splitlines() == out.read_text().splitlines()[:20]
    espn = tmp_path / "espn.jsonl"
    assert scan.main([str(video), "--out", str(espn), "--hz", "1", "--duration", "5", "--profile", "espn-2024"]) == 0
    assert all(json.loads(line)["game_clock"] is None for line in espn.read_text().splitlines())
    # too few moments to discover: espn-2024 with a warning, no profile saved
    capsys.readouterr()
    fallback = tmp_path / "fallback" / "scorebug.jsonl"
    assert scan.main([str(video), "--out", str(fallback), "--hz", "1", "--duration", "60", "--samples", "4"]) == 0
    printed = capsys.readouterr().out
    assert "WARNING: scorebug discovery failed: 4 moments sampled" in printed and "falling back to espn-2024" in printed
    assert not (fallback.parent / PROFILE_FILE).exists() and len(fallback.read_text().splitlines()) == 60


def test_scan_tool_helpers() -> None:
    scan = _load("nba_scan_scorebug")
    moments = scan.sample_times(100.0, 1100.0, 10, 2)
    assert len(moments) == 10 and all(len(m) == 2 and m[1] - m[0] == 1.0 for m in moments)
    assert 100.0 < moments[0][0] < 200.0 and 1000.0 < moments[-1][1] < 1100.0
    assert scan.parse_teams("LAL, GS") == ("LAL", "GS") and scan.parse_teams(None) is None
    with pytest.raises(SystemExit):
        scan.parse_teams("LAL")
    summary = {"header": {"competitions": [{"competitors": [{"homeAway": "home", "team": {"abbreviation": "GS"}},
                                                            {"homeAway": "away", "team": {"abbreviation": "LAL"}}]}]},
               "plays": [{"period": {"number": 4}, "clock": {"displayValue": "48.4"}, "awayScore": "109",
                          "homeScore": "105"}, {"period": {"number": 1}, "clock": {"displayValue": "11:40"}},
                         {"period": {"number": 1}, "clock": {"displayValue": "11:30"}, "awayScore": 0, "homeScore": 2}]}
    assert scan.summary_teams(summary) == ("LAL", "GS") and scan.summary_teams({}) is None
    assert scan.summary_scores(summary) == [(4, 48.4, 109, 105), (1, 690.0, 0, 2)] and scan.summary_scores(None) == []
    p = replace(ESPN_2024, away_score_box=(700, 950, 160, 60), home_score_box=(960, 950, 165, 60),
                clock_box=(1128, 950, 332, 60))
    x, y, w, h = scan.presence_region(p, (1920, 1080))
    assert x % 2 == y % 2 == w % 2 == h % 2 == 0 and x <= 478 and x + w >= 1460 and y <= 950 and y + h >= 1010
