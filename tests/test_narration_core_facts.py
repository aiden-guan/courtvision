"""build_facts: pseudonymised FactPackets with complete allowed_numbers and the hedge-only rule."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from test_narration_core_support import (COURT, AWAY_NAMES, HOME_NAMES, FakeShot, ev, make_frame, make_roster,
                                         make_stats, scenario_events, shot_made)

from courtvision.geometry.court import court_for
from courtvision.geometry.rules import rules_for
from courtvision.narration.facts import (FactsConfig, ad_zone, build_facts, event_trusted, format_game_clock,
                                         period_text, primary_event)
from courtvision.narration.pseudonym import PseudonymMap
from courtvision.narration.validator import extract_numbers, number_keys
from courtvision.types import EventKind, EventStatus, Team

ROSTER = make_roster()


def _facts(events, stats=None, frame="sim", **kw):
    stats = stats or make_stats()
    fr = make_frame(stats, frame) if frame else None
    return build_facts(events, stats, ROSTER, COURT, fr, PseudonymMap(ROSTER, seed=7), **kw)


def _uncal(e, **attrs):
    e.status = EventStatus.PROVISIONAL
    e.attrs.update(attrs)
    return e


# --------------------------------------------------------------------------- names and tokens


@pytest.mark.parametrize("name", list(scenario_events()))
def test_packets_contain_no_roster_names(name: str) -> None:
    facts = _facts(scenario_events()[name], recent_lines=["Riley Thorsen hits again", "Nakamura with the board"],
                   storylines=["Jordan Okafor two points from twenty"])
    blob = json.dumps(facts, default=str)
    for full in HOME_NAMES + AWAY_NAMES:
        assert full not in blob
        assert full.split()[-1] not in blob


def test_allowed_tokens_are_the_involved_players() -> None:
    pm = PseudonymMap(ROSTER, seed=7)
    evs = scenario_events()["made_assisted"]
    facts = build_facts(evs, make_stats(), ROSTER, COURT, None, pm)
    assert set(facts["allowed_tokens"]) == {pm.token("home-23"), pm.token("home-3")}
    assert set(facts["players"]) == set(facts["allowed_tokens"])
    assert facts["event"]["actor"] == pm.token("home-23")
    # stale tokens in recent lines are not allowed
    facts2 = build_facts(evs, make_stats(), ROSTER, COURT, None, pm, recent_lines=["Devon Achterberg scores."])
    assert pm.token("away-0") not in facts2["allowed_tokens"]


def test_default_pseudonym_map_is_deterministic() -> None:
    evs = scenario_events()["made3"]
    a = build_facts(evs, make_stats(), ROSTER, COURT, None)
    b = build_facts(evs, make_stats(), ROSTER, COURT, None)
    assert a["allowed_tokens"] == b["allowed_tokens"]


# --------------------------------------------------------------------------- allowed numbers


def _packet_strings_and_ints(obj, key=None, out=None):
    skip = {"event_id", "shot_id", "state_version", "confidence", "t", "since_t", "x", "y", "seconds_left",
            "recent_lines", "allowed_numbers", "allowed_tokens"}
    out = out if out is not None else []
    if key in skip:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _packet_strings_and_ints(v, k, out)
    elif isinstance(obj, list):
        for v in obj:
            _packet_strings_and_ints(v, key, out)
    elif isinstance(obj, bool) or obj is None:
        pass
    elif isinstance(obj, (int, float, str)):
        out.append((key, obj))
    return out


@pytest.mark.parametrize("name", list(scenario_events()))
def test_allowed_numbers_cover_every_number_in_the_packet(name: str) -> None:
    stats = make_stats(run={"team": Team.HOME, "points": 10, "opp_points": 2, "since_t": 250.0})
    facts = _facts(scenario_events()[name], stats, storylines=["{P01} needs 4 more"])
    allowed = number_keys(facts["allowed_numbers"])
    for key, value in _packet_strings_and_ints(facts):
        if isinstance(value, str):
            for m in extract_numbers(value):
                assert m.key in allowed, (key, value, m)
        else:
            assert f"card:{int(value)}" in allowed or f"card:{value:g}" in allowed, (key, value)


def test_allowed_numbers_have_digits_words_and_period_ordinals() -> None:
    facts = _facts(scenario_events()["made3"])
    nums = set(facts["allowed_numbers"])
    assert {"42", "forty-two", "40", "forty", "2", "two", "3", "three"} <= nums  # score, margin, value
    assert {"2nd", "second"} <= nums  # period ordinals
    assert "4:32" in nums  # pre-formatted clock
    assert "5-for-8" in nums  # shooting line of the scorer
    assert "fifth" not in nums and "4:33" not in nums


def test_clock_strings_are_preformatted() -> None:
    assert format_game_clock(272.4) == "4:32"
    assert format_game_clock(60.0) == "1:00"
    assert format_game_clock(24.36) == "24.3"
    assert format_game_clock(None) is None
    facts = _facts(scenario_events()["made3"], make_stats(game_clock=24.36, period=4))
    assert facts["clock"]["game"] == "24.3"
    assert {"24.3", "24", "4th", "fourth"} <= set(facts["allowed_numbers"])
    assert facts["clock"]["text"] == "fourth quarter, 24.3 left"
    assert facts["clock"]["is_last_period"] is True


def test_period_text_quarters_halves_and_overtime() -> None:
    assert period_text(1) == "first quarter"
    assert period_text(5) == "overtime"
    assert period_text(6) == "second overtime"
    assert period_text(2, 2, True) == "second half"
    ncaa = build_facts(scenario_events()["made3"], make_stats(period=2), ROSTER, court_for("ncaa_m"), None)
    assert ncaa["clock"]["period_text"] == "second half"
    nba_rules = build_facts(scenario_events()["made3"], make_stats(period=5), ROSTER, COURT, None,
                            rules=rules_for("nba"))
    assert nba_rules["clock"]["period_text"] == "overtime"


# --------------------------------------------------------------------------- hedge-only and confirmed


def test_confirmed_events_are_not_hedge_only() -> None:
    facts = _facts(scenario_events()["made3"])
    assert facts["hedge_only"] is False and facts["confirmed"] is True
    assert facts["event"]["outcome_ok"] is True and facts["event"]["value"] == 3


def test_uncalibrated_provisional_cv_event_is_hedge_only_without_value_or_derived() -> None:
    e = _uncal(shot_made(3, confidence=0.99), p_status="uncalibrated")
    stats = make_stats(shots=[FakeShot("s0040", "home-23", Team.HOME, 1, 2, True),
                              FakeShot("s0041", "home-23", Team.HOME, 2, 2, True),
                              FakeShot("s0042", "home-23", Team.HOME, 3, 3, True)])
    facts = _facts([e], stats)
    assert facts["hedge_only"] is True and facts["confirmed"] is False
    assert "value" not in facts["event"] and "distance_ft" not in facts["event"]
    assert facts["derived"] == []
    assert "three" not in " ".join(facts["recent_events"])


@pytest.mark.parametrize("p_status,conf,hedge", [
    ("platt", 0.97, False), ("isotonic", 0.96, False), ("assumed_sim", 0.99, False), ("calibrated", 0.99, False),
    ("platt", 0.90, True),  # calibrated but below the 0.95 make gate
    ("uncalibrated", 0.999, True), (None, 0.999, True),
])
def test_hedge_only_follows_p_status_and_gate(p_status, conf, hedge) -> None:
    e = shot_made(2, confidence=conf, status=EventStatus.PROVISIONAL)
    if p_status is not None:
        e.attrs["p_status"] = p_status
    assert _facts([e])["hedge_only"] is hedge


@pytest.mark.parametrize("source", ["sim", "manual"])
def test_sim_and_manual_sources_are_never_hedge_only(source: str) -> None:
    e = shot_made(2, status=EventStatus.PROVISIONAL, source=source, confidence=0.3)
    assert _facts([e])["hedge_only"] is False


def test_gate_boost_and_three_gate() -> None:
    e = shot_made(2, confidence=0.96, status=EventStatus.PROVISIONAL, attrs={"p_status": "platt"})
    assert event_trusted(e) is True
    assert event_trusted(e, gate_boost=0.03) is False
    a = ev(EventKind.SHOT_ATTEMPT, status=EventStatus.PROVISIONAL, player_id="home-23", team=Team.HOME, value=3,
           confidence=0.95, attrs={"p_status": "platt", "p_three": 0.93})
    assert event_trusted(a) is False  # P(3) below 0.97: no "for three"
    a.attrs["p_three"] = 0.98
    assert event_trusted(a) is True


def test_one_untrusted_outcome_event_makes_the_packet_hedge_only() -> None:
    made = shot_made(3)
    reb = _uncal(ev(EventKind.REBOUND, team=Team.AWAY, player_id="away-11", attrs={"offensive": False}),
                 p_status="uncalibrated")
    facts = _facts([made, reb])
    assert facts["hedge_only"] is True and facts["confirmed"] is False


def test_non_outcome_events_do_not_hedge() -> None:
    poss = _uncal(ev(EventKind.POSSESSION, team=Team.AWAY, player_id="away-0", attrs={"cause": "steal"}))
    facts = _facts([poss])
    assert facts["hedge_only"] is False and facts["events"][0]["outcome_ok"] is True


def test_retracted_events_are_ignored_for_hedging() -> None:
    gone = _uncal(shot_made(2))
    gone.status = EventStatus.RETRACTED
    assert _facts([gone, shot_made(3)])["hedge_only"] is False


# --------------------------------------------------------------------------- score and run


def test_score_confirmation_sources() -> None:
    evs = scenario_events()["made3"]
    assert _facts(evs)["score"]["confirmed"] is True  # stats.score_source == "scoreboard"
    events_only = make_stats(score_source="events")
    assert _facts(evs, events_only, frame=None)["score"]["confirmed"] is False
    assert _facts(evs, events_only, frame="sim")["score"]["confirmed"] is True  # sim scoreboard matches
    mismatch = make_stats(score_source="scoreboard", score_mismatch=True)
    assert _facts(evs, mismatch, frame=None)["score"]["confirmed"] is False
    stats = make_stats(score_source="events")
    fr = make_frame(stats, "ocr")
    fr.scoreboard = replace(fr.scoreboard, confidence=0.5)
    assert build_facts(evs, stats, ROSTER, COURT, fr)["score"]["confirmed"] is False


def test_score_fields() -> None:
    s = _facts(scenario_events()["made3"])["score"]
    assert (s["home"], s["away"], s["margin"], s["leader"]) == (42, 40, 2, "Riverside")
    assert s["text"] == "Riverside 42, Lakeport 40"
    tie = _facts(scenario_events()["made3"], make_stats(score={Team.HOME: 40, Team.AWAY: 40}))["score"]
    assert tie["leader"] is None and tie["margin"] == 0


def test_run_phrase_threshold_and_window_form() -> None:
    hot = make_stats(run={"team": Team.AWAY, "points": 8, "opp_points": 0, "since_t": 100.0})
    f = _facts(scenario_events()["dreb"], hot)
    assert f["run"]["text"] == "8-0 Lakeport run" and "8-0 Lakeport run" in f["derived"]
    small = make_stats(run={"team": Team.AWAY, "points": 5, "opp_points": 0, "since_t": 100.0})
    f2 = _facts(scenario_events()["dreb"], small)
    assert "text" not in f2["run"] and f2["run"]["points"] == 5
    window = make_stats(run={"team": Team.HOME, "points": 2, "opp_points": 0, "since_t": 280.0,
                             "window": {Team.HOME: 10, Team.AWAY: 2}, "window_leader": Team.HOME})
    assert _facts(scenario_events()["dreb"], window)["run"]["text"] == "10-2 Riverside run"


# --------------------------------------------------------------------------- derived phrases


def _shots(*made: bool) -> list[FakeShot]:
    return [FakeShot(f"s00{40 + i}", "home-23", Team.HOME, float(i), 2, m) for i, m in enumerate(made)]


def test_streak_derived_only_when_current_shot_is_latest() -> None:
    pm = PseudonymMap(ROSTER, seed=7)
    tok = pm.token("home-23")
    stats = make_stats(shots=_shots(False, True, True, True))  # last id s0043
    e = shot_made(2)
    e.shot_id = "s0043"
    f = build_facts([e], stats, ROSTER, COURT, None, pm)
    assert f"third straight make for {{{tok}}}" in f["derived"]
    e.shot_id = "s9999"  # snapshot does not contain this shot yet: never guess
    f = build_facts([e], stats, ROSTER, COURT, None, pm)
    assert not any("straight" in d for d in f["derived"])


def test_milestone_lead_change_tie_and_shooting_line() -> None:
    pm = PseudonymMap(ROSTER, seed=7)
    tok = pm.token("home-23")
    f = build_facts([shot_made(3)], make_stats(), ROSTER, COURT, None, pm)  # 12 pts after a 3: crosses 10
    assert f"{{{tok}}} reaches 10 points" in f["derived"]
    assert "Riverside takes the lead" in f["derived"]  # 39-40 before, 42-40 after
    assert f"{{{tok}}} is 5-for-8 from the field" in f["derived"]
    tie = make_stats(score={Team.HOME: 40, Team.AWAY: 40})
    f = build_facts([shot_made(2)], tie, ROSTER, COURT, None, pm)
    assert "Riverside ties it" in f["derived"]
    same = make_stats(score={Team.HOME: 50, Team.AWAY: 40})
    f = build_facts([shot_made(2)], same, ROSTER, COURT, None, pm)
    assert not any("lead" in d or "ties" in d for d in f["derived"])


def test_double_double_and_foul_trouble() -> None:
    pm = PseudonymMap(ROSTER, seed=7)
    stats = make_stats()
    stats.players["home-23"].pts, stats.players["home-23"].dreb = 14, 10
    stats.players["home-23"].pf = 5
    foul = ev(EventKind.FOUL, team=Team.HOME, player_id="home-23", source="manual")
    f = build_facts([foul], stats, ROSTER, COURT, None, pm, rules=rules_for("nba"))
    tok = pm.token("home-23")
    assert f"double-double for {{{tok}}}" in f["derived"]
    assert f"{{{tok}}} is one foul from fouling out" in f["derived"]
    stats.players["home-23"].pf = 6
    f = build_facts([foul], stats, ROSTER, COURT, None, pm, rules=rules_for("nba"))
    assert f"{{{tok}}} has fouled out" in f["derived"]


# --------------------------------------------------------------------------- AD zones


@pytest.mark.parametrize("x,y,zone", [
    (2.0, 2.0, "left corner"),  # left basket: offense faces -x, its left hand is -y
    (2.0, 48.0, "right corner"),
    (25.0, 8.0, "left wing"),
    (25.0, 42.0, "right wing"),
    (30.0, 25.0, "top"),
    (6.0, 25.0, "paint"),
    (60.0, 25.0, "backcourt"),
])
def test_ad_zones_relative_to_attacking_basket(x: float, y: float, zone: str) -> None:
    assert ad_zone(COURT, x, y, "left") == zone
    # mirrored point attacking the right basket gives the same zone
    assert ad_zone(COURT, COURT.length - x, COURT.width - y, "right") == zone


def test_ad_zone_without_coordinates() -> None:
    assert ad_zone(COURT, None, 3.0) is None
    assert ad_zone(COURT, 4.0, 25.0) == "paint"  # nearest basket when none is given


# --------------------------------------------------------------------------- misc


def test_primary_event_and_events_list() -> None:
    evs = scenario_events()["steal"]
    assert primary_event(evs).kind is EventKind.STEAL
    f = _facts(evs)
    assert f["event"]["kind"] == "steal" and len(f["events"]) == 2


def test_state_version_and_recent_lines_window() -> None:
    lines = [f"line {i}." for i in range(15)]
    f = _facts(scenario_events()["made3"], state_version=17, recent_lines=lines)
    assert f["state_version"] == 17
    assert f["recent_lines"] == lines[-FactsConfig().max_recent_lines:]
    assert _facts(scenario_events()["made3"])["state_version"] == 300000  # fallback: video ms


def test_correction_fact_from_previous_version() -> None:
    pm = PseudonymMap(ROSTER, seed=7)
    old = shot_made(3, pid="home-23")
    new = replace(old, player_id="home-3", revision=1, attrs=dict(old.attrs))
    f = build_facts([new], make_stats(), ROSTER, COURT, None, pm, previous=[old])
    c = f["correction"]
    assert c["old_actor"] == pm.token("home-23") and c["new_actor"] == pm.token("home-3")
    assert {pm.token("home-23"), pm.token("home-3")} <= set(f["allowed_tokens"])
