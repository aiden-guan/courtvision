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
    # depend on the frame rate. ~540 FGA pooled.
    return [GameSimulator(SimConfig(seconds=2000.0, fps=5.0, seed=s)).run() for s in range(40, 46)]


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


def test_rebounds_blocks_fouls_assists(games: list[SimResult]) -> None:
    reb = _events(games, K.REBOUND)
    oreb = sum(r.attrs["offensive"] for r in reb) / len(reb)
    assert 0.15 <= oreb <= 0.35  # ~25%
    assert sum(r.player_id is None for r in reb) / len(reb) < 0.08
    two_pa = sum(a.value == 2 for a in _fga(games))
    assert 0.02 <= len(_events(games, K.BLOCK)) / two_pa <= 0.10  # ~5-6% of 2PA
    poss = len(_events(games, K.POSSESSION))
    assert 0.04 <= len(_events(games, K.FOUL)) / poss <= 0.15  # ~8-10% of possessions
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
