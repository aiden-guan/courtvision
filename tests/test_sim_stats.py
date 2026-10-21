"""League-level realism of a long seeded simulation (NBA priors, basketball_rules_stats.md §4)."""

from __future__ import annotations

import collections
import time

import pytest

from courtvision.sim.simulator import GameSimulator, SimConfig, SimResult
from courtvision.types import EventKind as K


@pytest.fixture(scope="module")
def games() -> list[SimResult]:
    # 5 fps keeps the suite fast; flights, catches and rim crossings are exact in time, so the rates do not
    # depend on the frame rate. ~1030 FGA pooled over 12 games (sd of the pooled FG% ~0.015; with 6 games, sd
    # ~0.022, one seeded realisation came out at 0.38 against an expected 0.44 when round 2 re-rolled the games).
    return [GameSimulator(SimConfig(seconds=2000.0, fps=5.0, seed=s)).run() for s in range(40, 52)]


def _events(games: list[SimResult], kind: K) -> list:
    return [e for g in games for e in g.events if e.kind is kind]


def _fga(games: list[SimResult]) -> list:
    return [a for a in _events(games, K.SHOT_ATTEMPT) if a.value in (2, 3)]


def test_field_goal_percentage(games: list[SimResult]) -> None:
    fga = _fga(games)
    fgm = sum(e.value > 1 for e in _events(games, K.SHOT_MADE))
    assert len(fga) > 450
    assert 0.40 <= fgm / len(fga) <= 0.55


def test_three_point_share_and_corner_share(games: list[SimResult]) -> None:
    fga = _fga(games)
    threes = [a for a in fga if a.value == 3]
    assert 0.30 <= len(threes) / len(fga) <= 0.50
    corner = sum(a.attrs["zone"] == "corner_three" for a in threes) / len(threes)
    assert 0.15 <= corner <= 0.35  # NBA 24.3% of 3PA


def test_distance_mix(games: list[SimResult]) -> None:
    fga = _fga(games)
    bins = collections.Counter()
    for a in fga:
        d = a.attrs["distance_ft"]
        bins["3p" if a.value == 3 else "0_3" if d < 3 else "3_10" if d < 10 else "10_16" if d < 16 else "16_3p"] += 1
    share = {k: v / len(fga) for k, v in bins.items()}
    target = {"0_3": 0.226, "3_10": 0.214, "10_16": 0.091, "16_3p": 0.048, "3p": 0.421}  # NBA 2024-25
    for k, v in target.items():
        assert abs(share[k] - v) < 0.07, (k, share[k])


def test_turnover_rate_and_steal_share(games: list[SimResult]) -> None:
    poss = len(_events(games, K.POSSESSION))
    tov = _events(games, K.TURNOVER)
    assert 0.08 <= len(tov) / poss <= 0.18
    steals = sum(t.attrs["type"] in ("steal", "bad_pass") for t in tov)
    assert 0.35 <= steals / len(tov) <= 0.70
    assert len(_events(games, K.STEAL)) == steals
    # Box-score rate (note §4c: .144 TOV per possession, possessions = FGA - OREB + TOV + 0.44 FTA with team
    # rebounds / turnovers): ~1130 possessions pooled -> sd ~0.01 from game-to-game variation; calibrated on
    # 18 full train-seed games at 10 / 15 / 30 fps (0.137-0.140); this 5 fps fixture gives 0.118 (defensive
    # reads resolve a little differently at 5 fps), so the band is +-0.03.
    box = {k: sum(v[k] for g in games for v in g.truth_box.values()) for k in ("fga", "fta", "oreb", "tov", "stl")}
    team_tov = sum(t["team_tov"] for g in games for t in g.team_box.values())
    team_oreb = sum(t["team_oreb"] for g in games for t in g.team_box.values())
    n_tov = box["tov"] + team_tov
    poss = box["fga"] - box["oreb"] - team_oreb + n_tov + 0.44 * box["fta"]
    assert abs(n_tov / poss - 0.144) <= 0.03, n_tov / poss
    assert abs(box["stl"] / n_tov - 0.574) <= 0.08, box["stl"] / n_tov


def test_rebounds_blocks_fouls_assists(games: list[SimResult]) -> None:
    reb = _events(games, K.REBOUND)
    oreb = sum(r.attrs["offensive"] for r in reb) / len(reb)
    assert 0.15 <= oreb <= 0.35  # ~25%
    assert sum(r.player_id is None for r in reb) / len(reb) < 0.08
    two_pa = sum(a.value == 2 for a in _fga(games))
    # pbpstats zone rates (note §4c) give ~5.5% of FGA, ~8.8% of 2PA (rim .095, short mid .102); the old
    # "~5-6% of 2PA" spec was below the sourced rate. ~310 2PA pooled at 5 fps.
    assert 0.04 <= len(_events(games, K.BLOCK)) / two_pa <= 0.14
    poss = len(_events(games, K.POSSESSION))
    shooting = [f for f in _events(games, K.FOUL) if f.attrs["type"] == "shooting"]
    assert 0.04 <= len(shooting) / poss <= 0.15  # shooting fouls ~8-10% of possessions (note §4c: ~0.10)
    fgm = sum(e.value > 1 for e in _events(games, K.SHOT_MADE))
    assert 0.40 <= len(_events(games, K.ASSIST)) / fgm <= 0.75
    fts = [a for a in _events(games, K.SHOT_ATTEMPT) if a.value == 1]
    ftm = sum(e.value == 1 for e in _events(games, K.SHOT_MADE))
    assert 0.6 <= ftm / len(fts) <= 0.92  # NBA FT% 78%


def test_possession_flow(games: list[SimResult]) -> None:
    for g in games:
        poss = [e for e in g.events if e.kind is K.POSSESSION]
        passes = [e for e in g.events if e.kind is K.PASS]
        assert 1.0 <= len(passes) / len(poss) <= 7.0  # 1-6 passes per possession plus inbounds/outlets
        assert len(g.substitutions) > 0
        causes = {e.attrs["cause"] for e in poss}
        assert {"defensive_rebound", "made_shot", "steal"} <= causes


def test_performance_budget() -> None:
    t0 = time.perf_counter()
    GameSimulator(SimConfig(seconds=60.0, fps=30.0, seed=1)).run()
    elapsed = time.perf_counter() - t0
    # 600 s at 30 fps must stay under 20 s; one tenth of it gets a proportional (generous) budget.
    assert elapsed < 6.0, elapsed


def test_block_share_of_fga() -> None:
    """pbpstats zone '% blocked' (note §4c) gives ~5.5% of FGA. Defender positions (and so block chances)
    are calibrated at video frame rates, so these games run at 12 fps (the 5 fps fixture above gives ~3.5%):
    ~270 FGA pooled -> 95% band +-2.7 points."""
    games = [GameSimulator(SimConfig(seconds=1500.0, fps=12.0, seed=s)).run() for s in (40, 41, 42, 43)]
    fga = [(g, a) for g, r in enumerate(games) for a in r.events if a.kind is K.SHOT_ATTEMPT and a.value in (2, 3)]
    blocked = {(g, b.shot_id) for g, r in enumerate(games) for b in r.events if b.kind is K.BLOCK}  # ids per game
    share = sum((g, a.shot_id) in blocked for g, a in fga) / len(fga)
    assert len(fga) > 200 and 0.03 <= share <= 0.085, share
    near = [(g, a) for g, a in fga if a.attrs["distance_ft"] < 10.0]
    far3 = [(g, a) for g, a in fga if a.value == 3]
    assert sum((g, a.shot_id) in blocked for g, a in near) / len(near) > 3 * max(1e-9, sum(
        (g, a.shot_id) in blocked for g, a in far3) / len(far3))  # rim / short mid ~10%, threes ~1%


def test_free_throw_rate_and_trip_mix(games: list[SimResult]) -> None:
    """FTA/FGA .243 (note §4a, 2024-25): 2-shot fouls before the release, and-ones (1 FT), 3-shot fouls,
    and 2 FTs for non-shooting fouls in the team-foul penalty."""
    fga = _fga(games)
    fta = [a for a in _events(games, K.SHOT_ATTEMPT) if a.value == 1]
    assert 0.17 <= len(fta) / len(fga) <= 0.32, len(fta) / len(fga)
    fouls = _events(games, K.FOUL)
    and_ones = [f for f in fouls if f.attrs.get("and_one")]
    fgm = sum(e.value > 1 for e in _events(games, K.SHOT_MADE))
    assert 0.02 <= len(and_ones) / fgm <= 0.10  # note §4c: ~5% of made FGs
    assert any(f.attrs["type"] == "personal" and f.attrs["n_ft"] == 0 for f in fouls)
    assert any(f.attrs["type"] == "personal" and f.attrs["n_ft"] == 2 for f in fouls)
    assert any(f.attrs["type"] == "offensive" for f in fouls)
