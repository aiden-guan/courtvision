"""Pure advanced-stat formulas (no state, no I/O).

Formulas and coefficients come from ``docs/research/basketball_rules_stats.md`` §2b (the "rules
note"), which cites NBA.com [30], Basketball-Reference [31] and Kubatko et al. 2007 via [40].
Every function returns ``None`` when its denominator is zero, so callers never divide by zero and
JSON consumers see ``null`` rather than ``NaN``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FT_POSS_COEF",
    "FT_POSS_COEF_NBA_PBP",
    "FT_POSS_COEF_KENPOM",
    "TS_FT_COEF",
    "REGULATION_MINUTES",
    "efg",
    "ts",
    "tsa",
    "possessions_est",
    "pace",
    "ortg",
    "ppp",
    "pct",
    "tov_pct",
    "oreb_pct",
    "ft_rate",
    "regulation_minutes",
]

# Box-score possession estimate FT coefficient: Kubatko et al. 2007 (0.44) via [40]; sourced.
FT_POSS_COEF = 0.44
# NBA play-by-play implies 0.42 (2024-25) / 0.43 (2025-26): rules note §2b, our computation from
# pbpstats totals [35], corroborated by [40] (mu ~ 0.425); sourced.
FT_POSS_COEF_NBA_PBP = 0.43
# KenPom college coefficient; unverified (search-snippet text only, rules note §2b).
FT_POSS_COEF_KENPOM = 0.475
# True-shooting FT weight: TS% = PTS / (2 * (FGA + 0.44 * FTA)) [30][31]; sourced.
TS_FT_COEF = 0.44

# Regulation game length in minutes, used to normalise pace (RuleSpec.regulation_s / 60 when a
# RuleSpec is available). NBA 4x12, FIBA 4x10, NCAA men 2x20, NCAA women 4x10, NFHS 4x8;
# rules note §3 / config/rules/*.yaml; sourced.
REGULATION_MINUTES: dict[str, float] = {
    "nba": 48.0,
    "fiba": 40.0,
    "ncaa_m": 40.0,
    "ncaa_w": 40.0,
    "nfhs": 32.0,
}


def _ratio(num: float, den: float) -> float | None:
    return float(num) / float(den) if den > 0 else None


def pct(made: float, att: float) -> float | None:
    """made / attempts, or None with no attempts."""
    return _ratio(made, att)


def efg(fgm: float, fg3m: float, fga: float) -> float | None:
    """Effective FG%: (FGM + 0.5 * 3PM) / FGA [30][31]."""
    return _ratio(fgm + 0.5 * fg3m, fga)


def tsa(fga: float, fta: float, coef: float = TS_FT_COEF) -> float:
    """True-shooting attempts: FGA + 0.44 * FTA [30][31]."""
    return float(fga) + coef * float(fta)


def ts(pts: float, fga: float, fta: float, coef: float = TS_FT_COEF) -> float | None:
    """True shooting %: PTS / (2 * (FGA + 0.44 * FTA)) [30][31]."""
    return _ratio(pts, 2.0 * tsa(fga, fta, coef))


def possessions_est(fga: float, oreb: float, tov: float, fta: float, coef: float = FT_POSS_COEF) -> float:
    """Box-score possession estimate FGA - OREB + TOV + coef * FTA (rules note §2b).

    ``oreb`` and ``tov`` must include team offensive rebounds and team turnovers [40]. Use
    ``coef=FT_POSS_COEF_NBA_PBP`` (0.43) to match NBA play-by-play possession counts.
    """
    return float(fga) - float(oreb) + float(tov) + coef * float(fta)


def ppp(pts: float, poss: float) -> float | None:
    """Points per possession."""
    return _ratio(pts, poss)


def ortg(pts: float, poss: float) -> float | None:
    """Offensive (or, with opponent values, defensive) rating: 100 * PTS / POSS [30]."""
    r = _ratio(pts, poss)
    return None if r is None else 100.0 * r


def pace(team_poss: float, opp_poss: float, elapsed_minutes: float, game_minutes: float = 48.0) -> float | None:
    """Possessions per regulation game, B-Ref style [31].

    ``game_minutes * ((Tm Poss + Opp Poss) / 2) / elapsed_minutes``; B-Ref's ``Tm MP / 5`` is the
    elapsed game minutes. ``game_minutes`` is the league's regulation length (48 NBA, 40 FIBA /
    NCAA, 32 NFHS).
    """
    if elapsed_minutes <= 0:
        return None
    return game_minutes * ((float(team_poss) + float(opp_poss)) / 2.0) / float(elapsed_minutes)


def tov_pct(tov: float, fga: float, fta: float, coef: float = TS_FT_COEF) -> float | None:
    """Turnover %: TOV / (FGA + 0.44 * FTA + TOV) [31] (as a fraction, not x100)."""
    return _ratio(tov, float(fga) + coef * float(fta) + float(tov))


def oreb_pct(oreb: float, opp_dreb: float) -> float | None:
    """Offensive rebound %: ORB / (ORB + Opp DRB) (team form of [31], as a fraction)."""
    return _ratio(oreb, float(oreb) + float(opp_dreb))


def ft_rate(fta: float, fga: float) -> float | None:
    """Free-throw rate FTA / FGA (Oliver's fourth factor, attempt form)."""
    return _ratio(fta, fga)


def regulation_minutes(rules: Any | None = None, league: str | None = None, default: float = 48.0) -> float:
    """Regulation game minutes from a ``RuleSpec`` (preferred) or a league id."""
    if rules is not None:
        reg = getattr(rules, "regulation_s", None)
        if reg:
            return float(reg) / 60.0
    if league is not None and league in REGULATION_MINUTES:
        return REGULATION_MINUTES[league]
    return default
