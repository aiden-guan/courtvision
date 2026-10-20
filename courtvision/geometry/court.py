"""Court geometry per league, in feet (RESEARCH.md D1).

Axes follow ``courtvision.types``: origin where the left baseline meets the bottom sideline,
``x`` along the length (0 = left baseline), ``y`` along the width (0 = bottom sideline). Every
distance is measured from the inside edge of the baseline / sideline. FIBA metres are converted
once, here, with ``ft = m / 0.3048``.

Source values are transcribed from ``docs/research/basketball_rules_stats.md`` (the "rules note")
§1a/§1b. Bracketed numbers in comments are that note's source list ([1] NBA Rule 1, [10] NBA
2026-27 PDF diagram, [11] FIBA OBR 2026, [13] FIBA Equipment 2026, [16] NCAA men, [20] NCAA women,
[23] NFHS Rule 1 reproduction).

Shot frame
----------
Per basket, ``along`` is the distance from the basket centre along the court length, positive
toward midcourt, and ``lateral`` is ``y - width / 2``. The 3-point line is two straight corner
segments at ``|lateral| = corner_three_dist`` running from the baseline to the junction at
``along = sqrt(R^2 - c^2)``, joined by an arc of radius ``R`` beyond the junction. The line itself
is 2-point territory (D8, [3][11]).

Landmark naming scheme (``CourtSpec.landmarks``)
------------------------------------------------
* Side-specific points start with ``L_`` (basket at ``x = basket_offset``) or ``R_`` (basket at
  ``x = length - basket_offset``). ``_bottom`` is the ``y < width / 2`` side (the ``y = 0``
  sideline), ``_top`` the ``y > width / 2`` side.
* ``{S}_corner_{bottom|top}``: baseline/sideline corner.
* ``{S}_lane_baseline_{bottom|top}``: lane line meets the baseline.
* ``{S}_lane_ft_{bottom|top}``: lane line meets the free-throw line (extended, for FIBA).
* ``{S}_ft_center``: centre of the free-throw line.
* ``{S}_three_baseline_{bottom|top}``: straight 3-point segment meets the baseline.
* ``{S}_three_junction_{bottom|top}``: straight segment meets the arc.
* ``{S}_arc_apex``: top of the 3-point arc on the court's long axis.
* Shared: ``mid_bottom`` / ``mid_top`` (division line meets the sidelines), ``center``,
  ``center_circle_bottom`` / ``center_circle_top``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from courtvision.types import Side

__all__ = ["CourtSpec", "LEAGUES", "ZONES", "court_for"]

LEAGUES: tuple[str, ...] = ("nba", "fiba", "ncaa_m", "ncaa_w", "nfhs")

_FT_PER_M = 1.0 / 0.3048  # D1: ft = m / 0.3048
_FT_PER_IN = 1.0 / 12.0
_LINE_TOL = 1e-9  # ft; float slack so points computed on a line stay on it (on the line = 2)
ZONES: tuple[str, ...] = ("restricted_area", "paint", "midrange", "corner_three", "above_break_three", "backcourt")


def _m(metres: float) -> float:
    return metres * _FT_PER_M


@dataclass(frozen=True)
class CourtSpec:
    """Floor geometry of one league's court (feet). Rules live in ``RuleSpec`` (D15)."""

    league: str  # "nba" | "fiba" | "ncaa_m" | "ncaa_w" | "nfhs"
    length: float  # ft, inside edges
    width: float  # ft, inside edges
    rim_height: float  # ft, floor to top of ring
    rim_radius: float  # inner radius ft (18 in diameter -> 0.75)
    basket_offset: float  # baseline -> rim centre (ft)
    backboard_offset: float  # baseline -> backboard face (ft)
    three_pt_radius: float  # arc radius from rim centre to the line's outer edge (ft)
    corner_three_dist: float  # |y - basket_y| of the straight corner segments, outer edge (ft)
    lane_width: float  # ft, outside edges
    free_throw_dist: float  # backboard face -> free-throw line (ft)
    restricted_radius: float  # ft from rim centre, line included (0 = league has no arc)
    center_circle_radius: float  # ft, outside radius
    # Optional markings, drawn by line_segments only (0 = not drawn: not sourced for the league).
    inner_circle_radius: float = 0.0  # ft, the small circle inside the centre circle
    hash_mark_from_baseline: float = 0.0  # ft, sideline hash marks' distance from each baseline
    hash_mark_length: float = 0.0  # ft, how far the hash marks extend onto the court

    def __post_init__(self) -> None:
        positive = ("length", "width", "rim_height", "rim_radius", "basket_offset", "three_pt_radius",
                    "corner_three_dist", "lane_width", "free_throw_dist", "center_circle_radius")
        for name in positive:
            value = getattr(self, name)
            if not (value > 0 and math.isfinite(value)):
                raise ValueError(f"CourtSpec.{name} must be finite and > 0, got {value!r}")
        for name in ("restricted_radius", "inner_circle_radius", "hash_mark_from_baseline", "hash_mark_length"):
            value = getattr(self, name)
            if not (value >= 0 and math.isfinite(value)):
                raise ValueError(f"CourtSpec.{name} must be finite and >= 0, got {value!r}")
        if self.hash_mark_from_baseline >= self.length / 2 or self.hash_mark_length >= self.width / 2:
            raise ValueError("hash marks must lie inside one half of the court")
        if not 0 <= self.backboard_offset < self.basket_offset:
            raise ValueError("backboard_offset must lie between the baseline and the basket centre")
        if self.corner_three_dist > self.three_pt_radius:
            raise ValueError("corner_three_dist cannot exceed three_pt_radius")
        if self.corner_three_dist >= self.width / 2:
            raise ValueError("corner 3-point segments must lie inside the sidelines")

    # ------------------------------------------------------------------ derived quantities

    @property
    def basket_y(self) -> float:
        return self.width / 2.0

    @property
    def three_junction_along(self) -> float:
        """Junction of corner segment and arc, measured from the basket centre: sqrt(R^2 - c^2)."""
        return math.sqrt(max(self.three_pt_radius**2 - self.corner_three_dist**2, 0.0))

    @property
    def three_junction_from_baseline(self) -> float:
        return self.basket_offset + self.three_junction_along

    @property
    def free_throw_line_from_baseline(self) -> float:
        return self.backboard_offset + self.free_throw_dist

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------------ frames

    def _sign(self, side: Side) -> float:
        if side == "left":
            return 1.0
        if side == "right":
            return -1.0
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    def _baseline_x(self, side: Side) -> float:
        return 0.0 if self._sign(side) > 0 else self.length

    def _to_shot_frame(self, x: float, y: float, side: Side) -> tuple[float, float]:
        """Court (x, y) -> (along, lateral) relative to ``side``'s basket."""
        bx, by = self.basket_xy(side)
        return (x - bx) * self._sign(side), y - by

    def _from_shot_frame(self, along: float, lateral: float, side: Side) -> tuple[float, float]:
        bx, by = self.basket_xy(side)
        return bx + self._sign(side) * along, by + lateral

    def _from_baseline(self, dist: float, y: float, side: Side) -> tuple[float, float]:
        """Point ``dist`` ft in from ``side``'s baseline at width coordinate ``y``."""
        return self._baseline_x(side) + self._sign(side) * dist, y

    # ------------------------------------------------------------------ queries

    def basket_xy(self, side: Side) -> tuple[float, float]:
        """Floor point under the rim centre."""
        sign = self._sign(side)
        x = self.basket_offset if sign > 0 else self.length - self.basket_offset
        return x, self.basket_y

    def other_side(self, side: Side) -> Side:
        self._sign(side)
        return "right" if side == "left" else "left"

    def nearest_basket(self, x: float, y: float) -> Side:
        """Closer basket; ties (the division line) go to ``"left"``. ValueError on non-finite input."""
        _require_finite(x, y)
        left = self.distance_to_basket(x, y, "left")
        right = self.distance_to_basket(x, y, "right")
        return "left" if left <= right else "right"

    def distance_to_basket(self, x: float, y: float, side: Side) -> float:
        bx, by = self.basket_xy(side)
        return math.hypot(x - bx, y - by)

    def half(self, x: float) -> Side:
        """Half containing ``x``; the division line (x = length / 2) counts as ``"right"``."""
        _require_finite(x, 0.0)
        return "left" if x < self.length / 2.0 else "right"

    def in_bounds(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Inside the boundary inside edges, grown by ``margin`` ft (inclusive)."""
        m = margin + _LINE_TOL
        return -m <= x <= self.length + m and -m <= y <= self.width + m

    def in_paint(self, x: float, y: float, side: Side) -> bool:
        """Inside the lane rectangle (lane lines included), baseline to free-throw line."""
        dist = (x - self._baseline_x(side)) * self._sign(side)
        in_length = -_LINE_TOL <= dist <= self.free_throw_line_from_baseline + _LINE_TOL
        return in_length and abs(y - self.basket_y) <= self.lane_width / 2.0 + _LINE_TOL

    def free_throw_xy(self, side: Side) -> tuple[float, float]:
        """Centre of the free-throw line."""
        return self._from_baseline(self.free_throw_line_from_baseline, self.basket_y, side)

    def is_three(self, x: float, y: float, side: Side) -> bool:
        """True when (x, y) is strictly beyond ``side``'s 3-point line (the line is 2).

        Evaluate it at the shooter's last grounded foot position before the gather (D8). Below the
        junction only the straight corner line applies, including behind the backboard plane;
        beyond it only the arc applies (rules note §1b, verifier fix). ValueError on non-finite
        input (a nan position has no point value; silently answering 2 would corrupt stats).
        """
        _require_finite(x, y)
        along, lateral = self._to_shot_frame(x, y, side)
        if along <= self.three_junction_along:
            return abs(lateral) > self.corner_three_dist + _LINE_TOL
        return math.hypot(along, lateral) > self.three_pt_radius + _LINE_TOL

    def zone(self, x: float, y: float, side: Side) -> str:
        """Shot zone relative to ``side``'s basket.

        One of "restricted_area" | "paint" | "midrange" | "corner_three" | "above_break_three" |
        "backcourt". "backcourt" is the half that does not contain ``side``'s basket (the
        division line included). Leagues without a restricted arc (``restricted_radius == 0``:
        NCAA women, NFHS) never return "restricted_area". ValueError on non-finite input.
        """
        _require_finite(x, y)
        if (x - self._baseline_x(side)) * self._sign(side) >= self.length / 2.0:
            return "backcourt"
        if self.restricted_radius > 0 and self.distance_to_basket(x, y, side) <= self.restricted_radius + _LINE_TOL:
            return "restricted_area"
        if self.in_paint(x, y, side):
            return "paint"
        if not self.is_three(x, y, side):
            return "midrange"
        along, _ = self._to_shot_frame(x, y, side)
        return "corner_three" if along <= self.three_junction_along else "above_break_three"

    # ------------------------------------------------------------------ drawing / calibration

    def landmarks(self) -> dict[str, tuple[float, float]]:
        """Named floor keypoints for calibration (naming scheme in the module docstring)."""
        mid = self.length / 2.0
        pts: dict[str, tuple[float, float]] = {
            "mid_bottom": (mid, 0.0),
            "mid_top": (mid, self.width),
            "center": (mid, self.basket_y),
            "center_circle_bottom": (mid, self.basket_y - self.center_circle_radius),
            "center_circle_top": (mid, self.basket_y + self.center_circle_radius),
        }
        for side, prefix in (("left", "L"), ("right", "R")):
            pts.update(self._side_landmarks(side, prefix))
        return pts

    def _side_landmarks(self, side: Side, prefix: str) -> dict[str, tuple[float, float]]:
        by, half_lane, c = self.basket_y, self.lane_width / 2.0, self.corner_three_dist
        ft = self.free_throw_line_from_baseline
        out: dict[str, tuple[float, float]] = {}
        for tag, sgn in (("bottom", -1.0), ("top", 1.0)):
            out[f"{prefix}_corner_{tag}"] = self._from_baseline(0.0, 0.0 if sgn < 0 else self.width, side)
            out[f"{prefix}_lane_baseline_{tag}"] = self._from_baseline(0.0, by + sgn * half_lane, side)
            out[f"{prefix}_lane_ft_{tag}"] = self._from_baseline(ft, by + sgn * half_lane, side)
            out[f"{prefix}_three_baseline_{tag}"] = self._from_baseline(0.0, by + sgn * c, side)
            out[f"{prefix}_three_junction_{tag}"] = self._from_shot_frame(self.three_junction_along, sgn * c, side)
        out[f"{prefix}_ft_center"] = self.free_throw_xy(side)
        out[f"{prefix}_arc_apex"] = self._from_shot_frame(self.three_pt_radius, 0.0, side)
        return out

    def line_segments(self, arc_steps: int = 48) -> list[list[tuple[float, float]]]:
        """All floor markings as polylines (court ft); ``arc_steps`` segments per semicircle.

        Order: boundary, division line, centre circle, then per side (left, right) lane, FT
        semicircle, 3-point line and (if any) restricted arc; then the optional inner centre
        circle and sideline hash marks when the league's spec sets them.
        """
        if arc_steps < 2:
            raise ValueError("arc_steps must be >= 2")
        L, W, mid = self.length, self.width, self.length / 2.0
        lines: list[list[tuple[float, float]]] = [
            [(0.0, 0.0), (L, 0.0), (L, W), (0.0, W), (0.0, 0.0)],  # boundary
            [(mid, 0.0), (mid, W)],  # division line
            _circle((mid, self.basket_y), self.center_circle_radius, 2 * arc_steps),
        ]
        for side in ("left", "right"):
            lines.extend(self._side_lines(side, arc_steps))
        if self.inner_circle_radius > 0:
            lines.append(_circle((mid, self.basket_y), self.inner_circle_radius, arc_steps))
        if self.hash_mark_from_baseline > 0 and self.hash_mark_length > 0:
            d, h = self.hash_mark_from_baseline, self.hash_mark_length
            for x in (d, L - d):
                lines.extend([[(x, 0.0), (x, h)], [(x, W), (x, W - h)]])
        return lines

    def _side_lines(self, side: Side, arc_steps: int) -> list[list[tuple[float, float]]]:
        by, half_lane = self.basket_y, self.lane_width / 2.0
        ft = self.free_throw_line_from_baseline
        lane = [self._from_baseline(d, by + s * half_lane, side)
                for d, s in ((0.0, -1.0), (ft, -1.0), (ft, 1.0), (0.0, 1.0))]
        r = self.center_circle_radius  # FT and centre circles share a radius (NBA, FIBA, NCAA men per the note)
        ft_circle = [self._from_baseline(ft + r * math.sin(a), by + r * math.cos(a), side)
                     for a in _linspace(0.0, math.pi, arc_steps)]  # half facing midcourt
        lines = [lane, ft_circle, self._three_point_line(side, arc_steps)]
        if self.restricted_radius > 0:
            lines.append(self._restricted_arc(side, arc_steps))
        return lines

    def _three_point_line(self, side: Side, arc_steps: int) -> list[tuple[float, float]]:
        c, R, aj, b = self.corner_three_dist, self.three_pt_radius, self.three_junction_along, self.basket_offset
        phi = math.atan2(c, aj)  # junction angle from the long axis
        arc = [(R * math.cos(a), R * math.sin(a)) for a in _linspace(-phi, phi, arc_steps)]
        local = [(-b, -c), *arc, (-b, c)]
        return [self._from_shot_frame(a, lat, side) for a, lat in local]

    def _restricted_arc(self, side: Side, arc_steps: int) -> list[tuple[float, float]]:
        r, back = self.restricted_radius, -(self.basket_offset - self.backboard_offset)
        arc = [(r * math.cos(a), r * math.sin(a)) for a in _linspace(-math.pi / 2, math.pi / 2, arc_steps)]
        local = [(back, -r), *arc, (back, r)]  # straight ends run back to the backboard plane
        return [self._from_shot_frame(a, lat, side) for a, lat in local]


def _require_finite(x: float, y: float) -> None:
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"court coordinates must be finite, got ({x!r}, {y!r})")


def _linspace(a: float, b: float, steps: int) -> list[float]:
    return [a + (b - a) * i / steps for i in range(steps + 1)]


def _circle(center: tuple[float, float], r: float, steps: int) -> list[tuple[float, float]]:
    cx, cy = center
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in _linspace(0.0, 2 * math.pi, steps)]


# ---------------------------------------------------------------------------- league table
# Status per value: sourced = primary rule text / diagram in the rules note; derived = computed in
# the note from sourced values; unverified = flagged unverified by the note.

_NBA = CourtSpec(
    league="nba",
    length=94.0,  # "LENGTH 94 FEET (inside)" [10]; sourced
    width=50.0,  # "WIDTH 50 FEET (inside)" [10]; sourced
    rim_height=10.0,  # [1]; sourced
    rim_radius=9.0 * _FT_PER_IN,  # 18 in inner diameter [1]; sourced
    basket_offset=5.25,  # "5 FEET 3 IN." baseline -> basket centre [10]; sourced
    backboard_offset=4.0,  # "4 FEET" [10]; sourced
    three_pt_radius=23.75,  # 23 ft 9 in (outside) [1][10]; sourced
    corner_three_dist=22.0,  # 22 ft, lines 3 ft from the sideline [1]; sourced
    lane_width=16.0,  # "16 FEET (OUTSIDE)" [10]; sourced
    free_throw_dist=15.0,  # FT line 15 ft from the board face [1]; sourced
    restricted_radius=4.0,  # "4 FEET (INSIDE)" [1][10]; sourced (line inclusion unverified)
    center_circle_radius=6.0,  # 6 ft outside radius [10]; sourced
    inner_circle_radius=2.0,  # inner centre circle, 2 ft inside radius [10]; sourced
    hash_mark_from_baseline=28.0,  # sideline hash marks 28 ft from the baseline [1]; sourced
    hash_mark_length=3.0,  # extending 3 ft (onto the court) [1]; sourced (direction unverified in the note)
)

_FIBA = CourtSpec(
    league="fiba",
    length=_m(28.0),  # 28 m to the inner edge of the boundary [11]; sourced
    width=_m(15.0),  # 15 m [11]; sourced
    rim_height=_m(3.050),  # 3.050 m +/- 6 mm [13]; sourced
    rim_radius=_m(0.454) / 2.0,  # 450-459 mm inner diameter, 454 mm used (rules note §6) [13]; sourced
    basket_offset=_m(1.575),  # basket point 1.575 m from the endline [11]; sourced
    backboard_offset=_m(1.200),  # board face 1.200 m from the endline [13]; sourced
    three_pt_radius=_m(6.75),  # 6.75 m to the outer edge [11]; sourced
    corner_three_dist=_m(6.60),  # 7.5 m half width - 0.90 m from the sideline [11]; sourced
    lane_width=_m(4.90),  # lines 2.45 m from the endline midpoint [11]; sourced
    free_throw_dist=_m(4.60),  # 5.80 m from the endline - 1.20 m board offset [11][13]; derived
    restricted_radius=_m(1.30),  # no-charge semicircle, 1.30 m to the outer edge, line included [11]; sourced
    center_circle_radius=_m(1.80),  # 1.80 m outer edge [11]; sourced
)

_NCAA_M = CourtSpec(
    league="ncaa_m",
    length=94.0,  # inside edges [16]; sourced
    width=50.0,  # [16]; sourced
    rim_height=10.0,  # [16]; sourced
    rim_radius=9.0 * _FT_PER_IN,  # 18 in [16]; sourced
    basket_offset=5.25,  # 4 ft + 6 in + 9 in [16]; derived
    backboard_offset=4.0,  # [16]; sourced
    three_pt_radius=22.0 + 1.75 * _FT_PER_IN,  # 22 ft 1 3/4 in to the outside edge [16]; sourced
    corner_three_dist=21.0 + 7.875 * _FT_PER_IN,  # 21 ft 7 7/8 in [16]; sourced
    lane_width=12.0,  # [16]; sourced
    free_throw_dist=15.0,  # [16]; sourced
    # 4 ft to the inside edge + the 2-in line: the RA is bounded by the arc's outer edge (Rule 4-31)
    restricted_radius=4.0 + 2.0 * _FT_PER_IN,  # [16]; sourced
    center_circle_radius=6.0,  # 6 ft outside radius [16]; sourced
)

_NCAA_W = CourtSpec(
    league="ncaa_w",
    length=94.0,  # [20]; sourced
    width=50.0,  # [20]; sourced
    rim_height=10.0,  # [20]; sourced
    rim_radius=9.0 * _FT_PER_IN,  # 18 in [20]; sourced
    basket_offset=5.25,  # 4 ft + 6 in + 9 in [20]; derived
    backboard_offset=4.0,  # [20]; sourced
    three_pt_radius=22.0 + 1.75 * _FT_PER_IN,  # 22 ft 1 3/4 in [20]; sourced
    corner_three_dist=21.0 + 7.875 * _FT_PER_IN,  # 21 ft 7 7/8 in [20]; sourced
    lane_width=12.0,  # [20]; sourced
    free_throw_dist=15.0,  # [20]; sourced
    restricted_radius=0.0,  # no arc since 2023-24: RA is "directly underneath" the basket [20][21]; sourced
    center_circle_radius=6.0,  # not in the note's table; men's 6 ft assumed; unverified
)

_NFHS = CourtSpec(
    league="nfhs",
    length=84.0,  # 84 ft recommended, 94 ft max; unverified (use dataclasses.replace for a 94-ft floor)
    width=50.0,  # 50 ft max; unverified
    rim_height=10.0,  # [23]; sourced
    rim_radius=9.0 * _FT_PER_IN,  # 18 in; unverified
    basket_offset=5.25,  # 63 in, matches the 63-in straight 3-pt segment [23]; derived
    backboard_offset=4.0,  # [23]; sourced
    three_pt_radius=19.75,  # 19 ft 9 in semicircle [23]; sourced
    corner_three_dist=19.75,  # straight 63-in extensions at the arc radius [23]; sourced
    lane_width=12.0,  # [23]; sourced
    free_throw_dist=15.0,  # [23]; sourced
    restricted_radius=0.0,  # none described [23]; sourced
    center_circle_radius=6.0,  # not in the note's table; 6 ft assumed; unverified
)

_COURTS: dict[str, CourtSpec] = {c.league: c for c in (_NBA, _FIBA, _NCAA_M, _NCAA_W, _NFHS)}


def court_for(league: str) -> CourtSpec:
    """CourtSpec for ``league`` (case-insensitive); ValueError on an unknown league."""
    key = str(league).strip().lower()
    try:
        return _COURTS[key]
    except KeyError:
        raise ValueError(f"unknown league {league!r}; expected one of {', '.join(LEAGUES)}") from None
