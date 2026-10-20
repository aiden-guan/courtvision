"""CourtSpec: league constants vs docs/research/basketball_rules_stats.md, 2-vs-3, zones, landmarks."""

from __future__ import annotations

import dataclasses
import math

import pytest

from courtvision.geometry.court import LEAGUES, ZONES, CourtSpec, court_for

M = 1.0 / 0.3048  # ft per metre

# Rules note Table 1a (feet; FIBA converted from metres).
EXPECTED = {
    "nba": dict(length=94.0, width=50.0, rim_height=10.0, rim_radius=0.75, basket_offset=5.25,
                backboard_offset=4.0, three_pt_radius=23.75, corner_three_dist=22.0, lane_width=16.0,
                free_throw_dist=15.0, restricted_radius=4.0, center_circle_radius=6.0),
    "fiba": dict(length=28.0 * M, width=15.0 * M, rim_height=3.05 * M, rim_radius=0.227 * M,
                 basket_offset=1.575 * M, backboard_offset=1.2 * M, three_pt_radius=6.75 * M,
                 corner_three_dist=6.60 * M, lane_width=4.90 * M, free_throw_dist=4.60 * M,
                 restricted_radius=1.30 * M, center_circle_radius=1.80 * M),
    "ncaa_m": dict(length=94.0, width=50.0, rim_height=10.0, rim_radius=0.75, basket_offset=5.25,
                   backboard_offset=4.0, three_pt_radius=22.0 + 1.75 / 12, corner_three_dist=21.0 + 7.875 / 12,
                   lane_width=12.0, free_throw_dist=15.0, restricted_radius=4.0 + 2.0 / 12,
                   center_circle_radius=6.0),
    "ncaa_w": dict(length=94.0, width=50.0, rim_height=10.0, rim_radius=0.75, basket_offset=5.25,
                   backboard_offset=4.0, three_pt_radius=22.0 + 1.75 / 12, corner_three_dist=21.0 + 7.875 / 12,
                   lane_width=12.0, free_throw_dist=15.0, restricted_radius=0.0, center_circle_radius=6.0),
    "nfhs": dict(length=84.0, width=50.0, rim_height=10.0, rim_radius=0.75, basket_offset=5.25,
                 backboard_offset=4.0, three_pt_radius=19.75, corner_three_dist=19.75, lane_width=12.0,
                 free_throw_dist=15.0, restricted_radius=0.0, center_circle_radius=6.0),
}


@pytest.mark.parametrize("league", LEAGUES)
def test_league_constants_match_rules_note(league: str) -> None:
    court = court_for(league)
    assert court.league == league
    for name, value in EXPECTED[league].items():
        assert getattr(court, name) == pytest.approx(value, abs=1e-9), name


# Rules note Table 1b (junction from the baseline) and Table 1a (corner line -> sideline, FT line).
@pytest.mark.parametrize(
    ("league", "junction_ft", "corner_to_sideline_ft", "ft_from_baseline_ft", "tol"),
    [
        ("nba", 14.198, 3.0, 19.0, 1e-3),
        ("fiba", 2.990 * M, 0.90 * M, 5.80 * M, 1e-3 * M),
        ("ncaa_m", 9.881, 3.344, 19.0, 1e-3),
        ("ncaa_w", 9.881, 3.344, 19.0, 1e-3),
        ("nfhs", 5.25, 5.25, 19.0, 1e-9),
    ],
)
def test_derived_geometry_matches_rules_note(league, junction_ft, corner_to_sideline_ft, ft_from_baseline_ft, tol) -> None:
    court = court_for(league)
    assert court.three_junction_from_baseline == pytest.approx(junction_ft, abs=tol)
    assert court.width / 2 - court.corner_three_dist == pytest.approx(corner_to_sideline_ft, abs=tol)
    assert court.free_throw_line_from_baseline == pytest.approx(ft_from_baseline_ft, abs=tol)


def test_free_throw_line_to_basket_centre() -> None:
    assert court_for("nba").free_throw_line_from_baseline - court_for("nba").basket_offset == pytest.approx(13.75)
    fiba = court_for("fiba")
    assert (fiba.free_throw_line_from_baseline - fiba.basket_offset) / M == pytest.approx(4.225, abs=1e-9)


def test_court_for_is_case_insensitive_and_rejects_unknown() -> None:
    assert court_for(" NBA ") is court_for("nba")
    with pytest.raises(ValueError, match="unknown league"):
        court_for("wnba")


def test_courtspec_validation() -> None:
    nba = court_for("nba")
    with pytest.raises(ValueError):
        dataclasses.replace(nba, corner_three_dist=24.0)
    with pytest.raises(ValueError):
        dataclasses.replace(nba, backboard_offset=6.0)
    with pytest.raises(ValueError):
        dataclasses.replace(nba, length=0.0)
    assert dataclasses.replace(court_for("nfhs"), length=94.0).basket_xy("right") == (88.75, 25.0)


# ---------------------------------------------------------------------------- is_three

NBA = court_for("nba")
AJ = NBA.three_junction_along  # 8.948 ft from the basket centre


def _mirror(x: float) -> float:
    return NBA.length - x


# (x, y) for the LEFT basket, expected value; each case is also mirrored onto the right basket.
NBA_CASES = [
    ((29.0, 25.0), False),  # on the arc at the top of the key: the line is 2
    ((29.01, 25.0), True),
    ((28.99, 25.0), False),
    ((3.0, 3.0), False),  # on the bottom corner line
    ((3.0, 2.99), True),
    ((3.0, 47.0), False),  # on the top corner line
    ((3.0, 47.01), True),
    ((5.25 + AJ, 3.0), False),  # the junction itself
    ((5.25 + AJ, 2.99), True),
    ((5.25 + AJ - 0.5, 2.95), True),  # below the junction only the straight line applies (dist 23.61 < R)
    ((20.25, 5.0), True),  # beyond the junction only the arc applies (|lat| 20 < 22, dist 25 > R)
    ((20.25, 7.0), False),  # dist 23.43 < R
    ((1.0, 2.5), True),  # behind the backboard plane, outside the corner line
    ((1.0, 4.0), False),  # behind the backboard plane, inside the corner line
    ((-0.5, 2.0), True),  # behind the baseline (out of bounds) still uses the corner line
    ((5.25, 25.0), False),  # under the basket
    ((60.0, 25.0), True),  # backcourt heave
]


@pytest.mark.parametrize(("pt", "expected"), NBA_CASES)
def test_is_three_nba_left(pt, expected) -> None:
    assert NBA.is_three(*pt, "left") is expected


@pytest.mark.parametrize(("pt", "expected"), NBA_CASES)
def test_is_three_nba_right_mirrors_left(pt, expected) -> None:
    x, y = pt
    assert NBA.is_three(_mirror(x), y, "right") is expected


def test_is_three_on_and_off_the_arc_at_the_junction_angle() -> None:
    bx, by = NBA.basket_xy("left")
    phi = math.atan2(NBA.corner_three_dist, AJ) * 0.5  # halfway between the long axis and the junction
    for r, expected in ((NBA.three_pt_radius - 0.01, False), (NBA.three_pt_radius + 0.01, True)):
        x, y = bx + r * math.cos(phi), by - r * math.sin(phi)
        assert NBA.is_three(x, y, "left") is expected


def test_is_three_nfhs_sliver_from_rules_note() -> None:
    nfhs = court_for("nfhs")
    # Foot at lateral 19.5 ft, 5 ft behind the basket centre: dist 20.1 > 19.75 but inside the 63-in line.
    x, y = nfhs.basket_offset - 5.0, nfhs.basket_y + 19.5
    assert math.hypot(5.0, 19.5) > nfhs.three_pt_radius
    assert nfhs.is_three(x, y, "left") is False
    assert nfhs.is_three(x, nfhs.basket_y + 19.76, "left") is True
    assert nfhs.is_three(nfhs.length - x, y, "right") is False


def test_is_three_fiba_corner_line_in_metres() -> None:
    fiba = court_for("fiba")
    y_line = fiba.width - 0.90 * M  # top corner line, 0.90 m from the sideline
    assert fiba.is_three(2.0, y_line, "left") is False
    assert fiba.is_three(2.0, y_line + 0.01, "left") is True
    assert fiba.is_three(fiba.length - 2.0, y_line + 0.01, "right") is True


def test_is_three_rejects_bad_side() -> None:
    with pytest.raises(ValueError):
        NBA.is_three(10.0, 10.0, "middle")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------- zones

NBA_ZONES = [
    ((5.25, 25.0), "restricted_area"),
    ((5.25 + 3.9, 25.0), "restricted_area"),
    ((5.25 + 4.0, 25.0), "restricted_area"),  # radius inclusive
    ((5.25 + 4.1, 25.0), "paint"),
    ((15.0, 32.5), "paint"),  # 16-ft lane: |lat| 7.5 <= 8
    ((19.0, 25.0), "paint"),  # on the free-throw line
    ((19.5, 25.0), "midrange"),
    ((20.25, 10.0), "midrange"),
    ((29.0, 25.0), "midrange"),  # on the arc
    ((29.5, 25.0), "above_break_three"),
    ((3.0, 2.5), "corner_three"),
    ((3.0, 47.5), "corner_three"),
    ((5.25 + AJ - 0.5, 2.95), "corner_three"),
    ((20.25, 5.0), "above_break_three"),
    ((46.9, 25.0), "above_break_three"),
    ((47.0, 25.0), "backcourt"),  # the division line belongs to the backcourt
    ((80.0, 10.0), "backcourt"),
]


@pytest.mark.parametrize(("pt", "zone"), NBA_ZONES)
def test_zone_nba_both_baskets(pt, zone) -> None:
    x, y = pt
    assert NBA.zone(x, y, "left") == zone
    assert NBA.zone(_mirror(x), y, "right") == zone


def test_zone_other_leagues() -> None:
    ncaa_m, ncaa_w, nfhs, fiba = (court_for(k) for k in ("ncaa_m", "ncaa_w", "nfhs", "fiba"))
    assert ncaa_m.zone(15.0, 32.5, "left") == "midrange"  # 12-ft lane
    assert ncaa_m.zone(5.25 + 4.15, 25.0, "left") == "restricted_area"  # 4 ft 2 in, line included
    assert ncaa_w.zone(5.25, 25.0, "left") == "paint"  # no restricted arc
    assert nfhs.zone(5.25, 25.0, "left") == "paint"
    assert nfhs.zone(2.0, 4.0, "left") == "corner_three"
    assert nfhs.zone(42.0, 25.0, "left") == "backcourt"  # 84-ft court
    fb, fy = fiba.basket_xy("left")
    assert fiba.zone(fb + 4.2, fy, "left") == "restricted_area"  # 1.30 m = 4.265 ft
    assert fiba.zone(fb + 4.3, fy, "left") == "paint"


@pytest.mark.parametrize("league", LEAGUES)
def test_zone_values_are_known(league: str) -> None:
    court = court_for(league)
    for x in range(0, int(court.length) + 1, 3):
        for y in range(0, int(court.width) + 1, 3):
            assert court.zone(float(x), float(y), "left") in ZONES


# ---------------------------------------------------------------------------- simple queries


def test_basic_queries() -> None:
    assert NBA.basket_xy("left") == (5.25, 25.0)
    assert NBA.basket_xy("right") == (88.75, 25.0)
    assert NBA.other_side("left") == "right" and NBA.other_side("right") == "left"
    assert NBA.nearest_basket(20.0, 40.0) == "left" and NBA.nearest_basket(80.0, 3.0) == "right"
    assert NBA.nearest_basket(47.0, 25.0) == "left"  # tie
    assert NBA.distance_to_basket(5.25 + 3.0, 25.0 + 4.0, "left") == pytest.approx(5.0)
    assert NBA.half(46.99) == "left" and NBA.half(47.0) == "right"
    assert NBA.in_bounds(0.0, 0.0) and NBA.in_bounds(94.0, 50.0)
    assert not NBA.in_bounds(-0.1, 10.0) and NBA.in_bounds(-0.1, 10.0, margin=6.0)
    assert not NBA.in_bounds(50.0, 56.5, margin=6.0)
    assert NBA.free_throw_xy("left") == (19.0, 25.0) and NBA.free_throw_xy("right") == (75.0, 25.0)
    assert NBA.in_paint(1.0, 18.0, "left") and not NBA.in_paint(1.0, 16.9, "left")
    assert NBA.in_paint(93.0, 25.0, "right") and not NBA.in_paint(93.0, 25.0, "left")
    assert not NBA.in_paint(-0.5, 25.0, "left")
    assert NBA.to_dict()["three_pt_radius"] == 23.75


# ---------------------------------------------------------------------------- landmarks / lines

SHARED_NAMES = {"mid_bottom", "mid_top", "center", "center_circle_bottom", "center_circle_top"}
REQUIRED_SIDE_NAMES = [
    "corner_bottom", "corner_top", "lane_baseline_bottom", "lane_baseline_top", "lane_ft_bottom", "lane_ft_top",
    "ft_center", "three_baseline_bottom", "three_baseline_top", "three_junction_bottom", "three_junction_top",
    "arc_apex",
]


@pytest.mark.parametrize("league", LEAGUES)
def test_landmarks_sanity(league: str) -> None:
    court = court_for(league)
    lm = court.landmarks()
    assert len(lm) >= 24
    for name in ("mid_bottom", "mid_top", "center"):
        assert name in lm
    for suffix in REQUIRED_SIDE_NAMES:
        left, right = lm[f"L_{suffix}"], lm[f"R_{suffix}"]
        assert right[0] == pytest.approx(court.length - left[0]) and right[1] == pytest.approx(left[1])
    for name, (x, y) in lm.items():
        assert name.startswith(("L_", "R_")) or name in SHARED_NAMES, name
        assert court.in_bounds(x, y, margin=1e-9), name
    coords = {(round(x, 6), round(y, 6)) for x, y in lm.values()}
    assert len(coords) == len(lm), "landmarks must be distinct points"
    # Junctions and apex sit on the 3-point line: neither strictly beyond nor at distance < R off-arc.
    for side, p in (("left", "L"), ("right", "R")):
        for name in (f"{p}_three_junction_bottom", f"{p}_three_junction_top", f"{p}_arc_apex"):
            assert court.distance_to_basket(*lm[name], side) == pytest.approx(court.three_pt_radius)
            assert court.is_three(*lm[name], side) is False


def test_landmark_values_nba() -> None:
    lm = NBA.landmarks()
    assert lm["L_three_junction_bottom"] == pytest.approx((14.198, 3.0), abs=1e-3)
    assert lm["R_three_baseline_top"] == pytest.approx((94.0, 47.0))
    assert lm["L_arc_apex"] == pytest.approx((29.0, 25.0))
    assert lm["L_ft_center"] == pytest.approx((19.0, 25.0))
    assert lm["L_lane_ft_top"] == pytest.approx((19.0, 33.0))
    assert lm["L_lane_baseline_bottom"] == pytest.approx((0.0, 17.0))
    assert lm["R_corner_bottom"] == pytest.approx((94.0, 0.0))
    assert lm["center"] == pytest.approx((47.0, 25.0))
    assert lm["center_circle_top"] == pytest.approx((47.0, 31.0))


@pytest.mark.parametrize("league", LEAGUES)
def test_line_segments(league: str) -> None:
    court = court_for(league)
    lines = court.line_segments(arc_steps=32)
    extra = (1 if court.inner_circle_radius > 0 else 0) + (4 if court.hash_mark_length > 0 else 0)
    assert len(lines) == (11 if court.restricted_radius > 0 else 9) + extra
    for poly in lines:
        assert len(poly) >= 2
        for x, y in poly:
            assert court.in_bounds(x, y, margin=1e-9)
    # The 3-point polyline: ends on the baseline, arc points at radius R beyond the junction.
    three = lines[5]
    assert three[0][0] == pytest.approx(0.0) and three[-1][0] == pytest.approx(0.0)
    for x, y in three[1:-1]:
        assert court.distance_to_basket(x, y, "left") == pytest.approx(court.three_pt_radius)
        assert x - court.basket_offset >= court.three_junction_along - 1e-9
    with pytest.raises(ValueError):
        court.line_segments(arc_steps=1)


def test_courtspec_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        NBA.length = 90.0  # type: ignore[misc]
    assert isinstance(NBA, CourtSpec)


# ---------------------------------------------------------------------------- review regressions


@pytest.mark.parametrize("bad", [(math.nan, 25.0), (20.0, math.nan), (math.inf, 3.0), (-math.inf, -math.inf)])
def test_non_finite_positions_raise_instead_of_answering(bad) -> None:
    # Before: is_three(nan, nan) -> False (silently 2 points), zone -> "midrange",
    # nearest_basket / half -> "right".
    x, y = bad
    for call in (lambda: NBA.is_three(x, y, "left"), lambda: NBA.zone(x, y, "right"),
                 lambda: NBA.nearest_basket(x, y)):
        with pytest.raises(ValueError, match="finite"):
            call()
    if not math.isfinite(x):
        with pytest.raises(ValueError, match="finite"):
            NBA.half(x)
    assert not NBA.in_bounds(x, y) and not NBA.in_paint(x, y, "left")  # natural False, no raise


def test_courtspec_rejects_non_finite_dimensions() -> None:
    with pytest.raises(ValueError, match="finite"):
        dataclasses.replace(NBA, length=math.inf)
    with pytest.raises(ValueError):
        dataclasses.replace(NBA, hash_mark_from_baseline=50.0)
    with pytest.raises(ValueError):
        dataclasses.replace(NBA, inner_circle_radius=-1.0)


def test_nba_sourced_extra_markings_are_drawn() -> None:
    # Rules note Table 1a: inner centre circle 2 ft [10]; hash marks 28 ft from the baseline, 3 ft [1].
    lines = NBA.line_segments(arc_steps=16)
    inner = lines[11]
    assert all(math.hypot(x - 47.0, y - 25.0) == pytest.approx(2.0) for x, y in inner)
    hashes = lines[12:]
    assert {round(p[0][0], 6) for p in hashes} == {28.0, 66.0}
    for poly in hashes:
        (x0, y0), (x1, y1) = poly
        assert x0 == x1 and abs(y1 - y0) == pytest.approx(3.0) and y0 in (0.0, 50.0)
    # Leagues where the note does not source these markings draw none.
    for league in ("fiba", "ncaa_m", "ncaa_w", "nfhs"):
        c = court_for(league)
        assert c.inner_circle_radius == 0.0 and c.hash_mark_length == 0.0
