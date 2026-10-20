"""Advanced-stat formulas against hand calculation (stats/advanced.py)."""

from __future__ import annotations

import pytest

from courtvision.geometry.rules import rules_for
from courtvision.stats import advanced as adv


def test_shooting_efficiency() -> None:
    assert adv.efg(5, 2, 10) == pytest.approx(0.6)  # (5 + 0.5 * 2) / 10
    assert adv.ts(20, 15, 10) == pytest.approx(20 / (2 * (15 + 4.4)))
    assert adv.tsa(15, 10) == pytest.approx(19.4)
    assert adv.pct(3, 4) == pytest.approx(0.75)


def test_possession_estimate_and_coefficients() -> None:
    assert adv.possessions_est(80, 10, 12, 20) == pytest.approx(80 - 10 + 12 + 0.44 * 20)
    assert adv.possessions_est(80, 10, 12, 20, coef=adv.FT_POSS_COEF_NBA_PBP) == pytest.approx(90.6)
    assert (adv.FT_POSS_COEF, adv.FT_POSS_COEF_NBA_PBP, adv.TS_FT_COEF) == (0.44, 0.43, 0.44)


def test_ratings_and_pace() -> None:
    assert adv.ortg(110, 100) == pytest.approx(110.0)
    assert adv.ppp(110, 100) == pytest.approx(1.1)
    assert adv.pace(95, 97, 48.0, 48.0) == pytest.approx(96.0)  # full NBA game: the average
    assert adv.pace(40, 42, 24.0, 48.0) == pytest.approx(82.0)  # half an NBA game, doubled
    assert adv.pace(40, 40, 20.0, 40.0) == pytest.approx(80.0)  # half a FIBA game
    # B-Ref: 48 * ((Tm Poss + Opp Poss) / (2 * (Tm MP / 5))) with Tm MP = 5 * elapsed minutes
    tm, opp, minutes = 51.0, 49.0, 26.0
    assert adv.pace(tm, opp, minutes, 48.0) == pytest.approx(48 * ((tm + opp) / (2 * (5 * minutes / 5))))


def test_four_factor_helpers() -> None:
    assert adv.tov_pct(12, 80, 20) == pytest.approx(12 / (80 + 8.8 + 12))
    assert adv.oreb_pct(10, 30) == pytest.approx(0.25)
    assert adv.ft_rate(20, 80) == pytest.approx(0.25)


def test_zero_denominators_are_none() -> None:
    assert adv.efg(0, 0, 0) is None
    assert adv.ts(0, 0, 0) is None
    assert adv.ortg(10, 0) is None
    assert adv.ppp(1, 0) is None
    assert adv.pace(10, 10, 0.0) is None
    assert adv.oreb_pct(0, 0) is None


@pytest.mark.parametrize(("ruleset", "minutes"), [("nba", 48), ("fiba", 40), ("ncaa_m", 40), ("ncaa_w", 40), ("nfhs", 32)])
def test_regulation_minutes_from_rulespec_and_league(ruleset: str, minutes: float) -> None:
    assert adv.regulation_minutes(rules_for(ruleset)) == pytest.approx(minutes)
    assert adv.regulation_minutes(None, ruleset) == pytest.approx(minutes)


def test_regulation_minutes_default() -> None:
    assert adv.regulation_minutes(None, "aau_custom") == 48.0
    assert adv.regulation_minutes(None, None, default=32.0) == 32.0
