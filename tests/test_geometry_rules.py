"""RuleSpec YAMLs vs the rules note (§3, §8a) and override deep-merge / validation."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from courtvision.geometry.rules import (
    OVERRIDES_DIR,
    RULES_DIR,
    RuleSpec,
    available_overrides,
    available_rulesets,
    rules_for,
)

LEAGUES = ("nba", "fiba", "ncaa_m", "ncaa_w", "nfhs")


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_every_league_has_a_yaml() -> None:
    assert set(LEAGUES) <= set(available_rulesets())
    assert "no_shot_clock" in available_overrides()
    assert RULES_DIR.is_dir() and OVERRIDES_DIR.is_dir()


# Rules note §3 "Periods" / "Overtime" / "Shot clock" / "Personal foul limit" rows.
@pytest.mark.parametrize(
    ("ruleset", "period_s", "n_periods", "ot_s", "halves", "shot", "reset", "foul_out"),
    [
        ("nba", 720, 4, 300, False, 24, 14, 6),
        ("fiba", 600, 4, 300, False, 24, 14, 5),
        ("ncaa_m", 1200, 2, 300, True, 30, 20, 5),
        ("ncaa_w", 600, 4, 300, False, 30, 20, 5),
        ("nfhs", 480, 4, 240, False, 35, 35, 5),
    ],
)
def test_league_rules_match_note(ruleset, period_s, n_periods, ot_s, halves, shot, reset, foul_out) -> None:
    r = rules_for(ruleset)
    assert r.ruleset == ruleset
    assert (r.period_s, r.n_periods, r.ot_s, r.halves) == (period_s, n_periods, ot_s, halves)
    assert (r.shot_clock_s, r.shot_clock_reset_s, r.foul_out) == (shot, reset, foul_out)
    assert r.has_shot_clock


def test_possession_bonus_and_technicals() -> None:
    nba, fiba, ncaa_m, ncaa_w, nfhs = (rules_for(k) for k in LEAGUES)
    assert nba.possession == "jump"
    assert {fiba.possession, ncaa_m.possession, ncaa_w.possession, nfhs.possession} == {"arrow"}
    assert nfhs.tech_ft == {"n_ft": 2, "shooter": "any", "resume": "division_line", "flagrant_disqualifies": True}
    assert fiba.tech_ft["n_ft"] == 1 and fiba.tech_ft["flagrant_resume"] == "frontcourt"
    assert nfhs.foul_out_counts_technicals is True and nba.foul_out_counts_technicals is False
    assert ncaa_m.bonus["one_and_one_from"] == 7 and ncaa_m.bonus["two_shots_from"] == 10
    assert ncaa_m.bonus["scope"] == "half"
    assert nfhs.bonus["one_and_one_from"] is None and nfhs.bonus["two_shots_from"] == 5
    assert fiba.bonus["two_shots_from"] == 5 and fiba.bonus["ot_counts_toward_last_period"] is True
    assert nba.timeouts["count"] == 7 and nba.timeouts["per_ot"] == 2
    assert fiba.timeouts["first_half"] == 2 and fiba.timeouts["second_half"] == 3
    assert nfhs.timeouts["count"] == 3 and nfhs.timeouts["short_count"] == 2 and nfhs.timeouts["carry_over"]
    assert ncaa_m.timeouts["media"]["short_count"] == 3
    assert (nba.backcourt_s, fiba.backcourt_s, nfhs.backcourt_s) == (8, 8, 10)
    assert nfhs.intervals["halftime_s"] == 600 and fiba.intervals["between_periods_s"] == 120


def test_clock_stops_after_made_fg() -> None:
    nba, fiba, ncaa_m, ncaa_w, nfhs = (rules_for(k) for k in LEAGUES)
    assert nba.clock_stops_after_fg == {"q1_3_last_s": 60, "q4_ot_last_s": 120}
    assert nfhs.clock_stops_after_fg is None
    assert nba.clock_stops_after_fg_at(1, 59.0) and not nba.clock_stops_after_fg_at(1, 61.0)
    assert nba.clock_stops_after_fg_at(4, 119.0) and not nba.clock_stops_after_fg_at(4, 121.0)
    assert nba.clock_stops_after_fg_at(5, 100.0)  # overtime
    assert not fiba.clock_stops_after_fg_at(3, 10.0) and fiba.clock_stops_after_fg_at(4, 120.0)
    assert ncaa_m.clock_stops_after_fg_at(2, 59.9) and not ncaa_m.clock_stops_after_fg_at(2, 60.0)
    assert not ncaa_m.clock_stops_after_fg_at(1, 10.0)  # first half: never
    assert ncaa_w.clock_stops_after_fg_at(4, 30.0) and not ncaa_w.clock_stops_after_fg_at(3, 30.0)
    assert not nfhs.clock_stops_after_fg_at(4, 1.0)


def test_period_helpers() -> None:
    nfhs = rules_for("nfhs")
    assert nfhs.regulation_s == 32 * 60
    assert nfhs.period_length_s(4) == 480 and nfhs.period_length_s(5) == 240
    assert nfhs.is_overtime(5) and not nfhs.is_overtime(4)
    with pytest.raises(ValueError):
        nfhs.period_length_s(0)


def test_no_shot_clock_override() -> None:
    r = rules_for("nfhs", ["no_shot_clock"])
    assert r.ruleset == "nfhs"
    assert r.shot_clock_s is None and r.shot_clock_reset_s is None and not r.has_shot_clock
    assert r.period_s == 480  # everything else untouched


def test_deep_merge_from_explicit_path(tmp_path: Path) -> None:
    aau = _write(tmp_path, "aau.yaml", """
ruleset: aau_test
period_s: 960
n_periods: 2
halves: true
running_clock: {stop_last_s: 120, stop_scope: game, mercy_margin: 20}
timeouts: {count: 2}
bonus: {scope: half, two_shots_from: 10}
""")
    r = rules_for("nfhs", [str(aau)])
    assert r.ruleset == "aau_test"
    assert (r.period_s, r.n_periods, r.halves) == (960, 2, True)
    assert r.running_clock == {"stop_last_s": 120, "stop_scope": "game", "mercy_margin": 20}
    assert r.timeouts["count"] == 2 and r.timeouts["length_s"] == 60 and r.timeouts["short_count"] == 2
    assert r.bonus == {"scope": "half", "one_and_one_from": None, "two_shots_from": 10, "ot_counts_toward_last_period": True}
    assert r.shot_clock_s == 35  # base value kept


def test_overrides_apply_in_order_and_null_clears(tmp_path: Path) -> None:
    first = _write(tmp_path, "a.yaml", "running_clock: {mercy_margin: 35}\n")
    second = _write(tmp_path, "b.yaml", "running_clock: null\nfoul_out: 6\n")
    assert rules_for("nfhs", [str(first)]).running_clock == {"mercy_margin": 35}
    r = rules_for("nfhs", [str(first), str(second)])
    assert r.running_clock is None and r.foul_out == 6
    r2 = rules_for("nfhs", [str(first), "no_shot_clock"])
    assert r2.running_clock == {"mercy_margin": 35} and r2.shot_clock_s is None


def test_loading_returns_independent_dicts() -> None:
    a = rules_for("nba")
    a.timeouts["count"] = 99
    assert rules_for("nba").timeouts["count"] == 7


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("shot_clok_s: 30\n", "unknown rule key"),
        ("running_clock: {mercy_margn: 30}\n", "running_clock: unknown key"),
        ("timeouts: {media: {cuont: 1}}\n", "timeouts.media: unknown key"),
        ("halves: true\n", "halves=true needs n_periods=2"),
        ("period_s: 0\n", "period_s must be a positive integer"),
        ("foul_out: true\n", "foul_out must be a positive integer"),
        ("possession: coin_flip\n", "possession must be"),
        ("shot_clock_s: null\n", "shot_clock_reset_s must be null"),
        ("shot_clock_reset_s: 40\n", "cannot exceed"),
        ("tech_ft: {shooter: captain}\n", "tech_ft.shooter must be one of"),
        ("timeouts: null\n", "timeouts must be a mapping"),
        ("- 1\n- 2\n", "top level must be a mapping"),
    ],
)
def test_validation_errors(tmp_path: Path, text: str, match: str) -> None:
    bad = _write(tmp_path, "bad.yaml", text)
    with pytest.raises(ValueError, match=match):
        rules_for("nfhs", [str(bad)])


def test_unknown_ruleset_and_missing_override(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown ruleset"):
        rules_for("aau")
    with pytest.raises(ValueError, match="unknown ruleset"):
        rules_for("../nba")
    with pytest.raises(FileNotFoundError):
        rules_for("nfhs", ["does_not_exist"])
    with pytest.raises(FileNotFoundError):
        rules_for("nfhs", [str(tmp_path / "missing.yaml")])


def test_empty_override_is_a_no_op(tmp_path: Path) -> None:
    empty = _write(tmp_path, "empty.yaml", "# nothing\n")
    assert rules_for("fiba", [str(empty)]) == rules_for("fiba")


def test_rulespec_contract_fields_and_dict_round_trip() -> None:
    names = [f.name for f in dataclasses.fields(RuleSpec)]
    assert names[:14] == [
        "ruleset", "period_s", "n_periods", "ot_s", "halves", "running_clock", "timeouts", "possession",
        "bonus", "foul_out", "tech_ft", "shot_clock_s", "shot_clock_reset_s", "clock_stops_after_fg",
    ]
    for league in LEAGUES:
        r = rules_for(league)
        assert RuleSpec(**r.to_dict()) == r
    with pytest.raises(dataclasses.FrozenInstanceError):
        rules_for("nba").foul_out = 5  # type: ignore[misc]


def test_override_given_as_a_bare_string_or_file_name() -> None:
    # Before: rules_for("nfhs", "no_shot_clock") iterated the characters ("n", "o", ...) and
    # rules_for("nfhs", ["no_shot_clock.yaml"]) looked for the file in the working directory.
    for overrides in ("no_shot_clock", ["no_shot_clock.yaml"], [OVERRIDES_DIR / "no_shot_clock.yaml"]):
        r = rules_for("nfhs", overrides)  # type: ignore[arg-type]
        assert r.shot_clock_s is None and r.shot_clock_reset_s is None
    with pytest.raises(FileNotFoundError):
        rules_for("nfhs", ["sub/no_shot_clock.yaml"])
