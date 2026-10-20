"""State builders: buckets, possession summaries, <= 1 KiB budgets and pseudonymisation (design §6, D16)."""

from __future__ import annotations

import json
import re

import pytest

from courtvision.decision import states as S
from courtvision.decision.client import sanitize, state_bytes
from courtvision.types import Event, EventKind, EventStatus, Roster, RosterEntry, Team

ROSTER = Roster([
    RosterEntry("home-3", Team.HOME, "3", "Jalen Brooks"),
    RosterEntry("home-7", Team.HOME, "7", "Marcus Hill"),
    RosterEntry("away-11", Team.AWAY, "11", "Theo Okafor"),
], team_names={Team.HOME: "Harbor", Team.AWAY: "Summit"})
NAMES = [e.name for e in ROSTER.entries]


def _wire_json(state: dict) -> str:
    return json.dumps(sanitize(state), ensure_ascii=False)


def _assert_no_names(text: str) -> None:
    for name in NAMES:
        for part in [name, *name.split()]:
            assert not re.search(rf"(?i)\b{re.escape(part)}\b", text), f"{part!r} leaked"
    for e in ROSTER.entries:
        assert e.player_id not in text


def ev(eid: str, kind: EventKind, t: float, team: Team | None = Team.HOME, pid: str | None = None,
       other: str | None = None, value: int = 0, shot_id: str | None = None, **attrs) -> Event:
    return Event(eid, kind, t, int(t * 30), team=team, player_id=pid, other_player_id=other, value=value,
                 shot_id=shot_id, attrs=attrs)


# --------------------------------------------------------------------------- buckets


@pytest.mark.parametrize(("diff", "expected"), [
    (0, "the game is tied"), (3, "Harbor leads; a one-possession game"), (-5, "Harbor leads; a two-possession game"),
    (9, "Harbor leads; a three-possession game"), (15, "Harbor leads by double digits"),
    (25, "Harbor leads comfortably"),
])
def test_margin_text(diff: int, expected: str) -> None:
    assert S.margin_text("Harbor", diff) == expected
    assert S.code_from_text(expected, S.MARGIN_TEMPLATES) == S.margin_code(diff)


@pytest.mark.parametrize(("last", "clock", "expected"), [
    (True, 20.0, "less than one shot clock left in the game"), (True, 50.0, "final minute of the game"),
    (False, 100.0, "final two minutes of the period"), (True, 250.0, "final five minutes of the game"),
    (False, 500.0, "earlier in the period"), (True, None, "game clock unknown"),
])
def test_time_bucket(last: bool, clock: float | None, expected: str) -> None:
    assert S.time_bucket(last, clock, 24) == expected
    assert S.code_from_text(expected, S.TIME_TEMPLATES) == S.time_code(last, clock, 24)


def test_time_bucket_without_shot_clock() -> None:
    assert S.time_bucket(True, 20.0, None) == "final minute of the game"


def test_clutch_text_uses_last_period_window_and_margin() -> None:
    assert S.clutch_text(True, 300.0, 5) == S.CLUTCH_YES
    assert S.clutch_text(True, 301.0, 5) == "no"
    assert S.clutch_text(True, 100.0, 6) == "no"
    assert S.clutch_text(False, 100.0, 2) == "no"


def test_period_text_and_number_words() -> None:
    assert S.period_text(4) == "fourth quarter"
    assert S.period_text(5) == "overtime"
    assert S.period_text(6) == "second overtime"
    assert S.period_text(2, n_periods=2, halves=True) == "second half"
    assert S.number_words(8) == "eight" and S.number_words(42) == "forty-two"


def test_run_text_needs_a_net_eight_point_run() -> None:
    assert S.run_text("Summit", 8, 0) == "Summit has scored the last eight points"
    assert S.run_text("Summit", 10, 2) == "Summit is on a ten to two run"
    assert S.run_text("Summit", 9, 3) is None
    assert S.run_text("Summit", 6, 0) is None


# --------------------------------------------------------------------------- possession summaries


def test_putback_summary() -> None:
    evs = [ev("e1", EventKind.POSSESSION, 10.0, pid="home-3", cause="inbound"),
           ev("e2", EventKind.SHOT_ATTEMPT, 18.0, pid="home-3", value=2, shot_id="s1", shot_type="jumper"),
           ev("e3", EventKind.SHOT_MISSED, 19.0, pid="home-3", shot_id="s1"),
           ev("e4", EventKind.REBOUND, 19.5, pid="home-7", offensive=True),
           ev("e5", EventKind.SHOT_ATTEMPT, 20.7, pid="home-7", value=2, shot_id="s2", shot_type="tip"),
           ev("e6", EventKind.SHOT_MADE, 21.0, pid="home-7", value=2, shot_id="s2")]
    p = S.summarize_possession(evs, Team.HOME, "Harbor")
    assert p.end == "made_shot" and p.shot_type == "tip"
    assert p.seconds_since_oreb == pytest.approx(1.2)
    assert p.seconds_since_start == pytest.approx(10.7)
    assert "a teammate grabbed an offensive rebound" in p.actions


def test_transition_and_spotup_summary() -> None:
    evs = [ev("e1", EventKind.POSSESSION, 10.0, pid="home-3", cause="steal"),
           ev("e2", EventKind.DRIBBLE, 10.4, pid="home-3"),
           ev("e3", EventKind.PASS, 12.0, pid="home-3", other="home-7", completed=True),
           ev("e4", EventKind.SHOT_ATTEMPT, 14.5, pid="home-7", value=3, shot_id="s1", shot_type="jumper",
              catch_and_shoot=True, distance_ft=24.0)]
    p = S.summarize_possession(evs, Team.HOME, "Harbor")
    assert (p.start_cause, p.end, p.shot_value, p.passes) == ("steal", "missed_shot", 3, 1)
    assert p.seconds_since_start == pytest.approx(4.5) and p.seconds_since_oreb is None
    assert p.catch_and_shoot is True and p.terminal_dribbles == 0


def test_isolation_summary_counts_terminal_player_dribbles() -> None:
    evs = [ev("e0", EventKind.POSSESSION, 0.0, pid="home-3", cause="inbound"),
           *[ev(f"d{i}", EventKind.DRIBBLE, 1.0 + i, pid="home-3") for i in range(5)],
           ev("e9", EventKind.SHOT_ATTEMPT, 12.0, pid="home-3", value=2, shot_id="s1", shot_type="jumper",
              catch_and_shoot=False)]
    p = S.summarize_possession(evs, Team.HOME, "Harbor")
    assert p.passes == 0 and p.terminal_dribbles == 5
    assert "the ball handler dribbled many times" in p.actions


def test_retracted_make_and_turnover_and_free_throws() -> None:
    made = ev("e5", EventKind.SHOT_MADE, 15.0, pid="home-7", value=2, shot_id="s1")
    retracted = ev("e5", EventKind.SHOT_MADE, 15.0, pid="home-7", value=2, shot_id="s1")
    retracted.revision, retracted.status = 1, EventStatus.RETRACTED
    base = [ev("e1", EventKind.POSSESSION, 10.0, pid="home-3", cause="inbound"),
            ev("e4", EventKind.SHOT_ATTEMPT, 14.0, pid="home-7", value=2, shot_id="s1", shot_type="layup")]
    assert S.summarize_possession([*base, made], Team.HOME, "Harbor").end == "made_shot"
    assert S.summarize_possession([*base, made, retracted], Team.HOME, "Harbor").end == "missed_shot"

    tov = [ev("e1", EventKind.POSSESSION, 10.0, pid="home-3", cause="inbound"),
           ev("e2", EventKind.STEAL, 13.0, team=Team.AWAY, pid="away-11", other="home-3"),
           ev("e3", EventKind.TURNOVER, 13.0, pid="home-3", type="steal")]
    p = S.summarize_possession(tov, Team.HOME, "Harbor")
    assert p.end == "turnover" and p.turnover_type == "steal" and "steal" in p.flags

    fts = [ev("e1", EventKind.POSSESSION, 10.0, pid="home-3", cause="inbound"),
           ev("e2", EventKind.FOUL, 16.0, team=Team.AWAY, pid="away-11", other="home-3"),
           ev("e3", EventKind.SHOT_ATTEMPT, 30.0, pid="home-3", value=1, shot_id="s2", shot_type="free_throw")]
    p = S.summarize_possession(fts, Team.HOME, "Harbor")
    assert p.end == "free_throws" and p.seconds_since_start == pytest.approx(6.0)


def test_and_one_ends_with_the_made_shot_but_a_fouled_miss_ends_with_free_throws() -> None:
    and_one = [ev("e1", EventKind.POSSESSION, 10.0, pid="home-3", cause="inbound"),
               ev("e2", EventKind.SHOT_ATTEMPT, 15.0, pid="home-7", value=2, shot_id="s1", shot_type="layup"),
               ev("e3", EventKind.FOUL, 15.3, team=Team.AWAY, pid="away-11", other="home-7"),
               ev("e4", EventKind.SHOT_MADE, 15.6, pid="home-7", value=2, shot_id="s1"),
               ev("e5", EventKind.SHOT_ATTEMPT, 40.0, pid="home-7", value=1, shot_id="s2", shot_type="free_throw")]
    p = S.summarize_possession(and_one, Team.HOME, "Harbor")
    assert p.end == "made_shot" and p.seconds_since_start == pytest.approx(5.0) and "and_one" in p.flags
    fouled_miss = [and_one[0], and_one[2], and_one[4]]  # no FGA for a fouled shooter who misses
    p = S.summarize_possession(fouled_miss, Team.HOME, "Harbor")
    assert p.end == "free_throws" and p.seconds_since_start == pytest.approx(5.3)


def test_flags_from_events_dunk_and_one_and_block() -> None:
    evs = [ev("a", EventKind.SHOT_ATTEMPT, 5.0, pid="home-7", value=2, shot_id="s1", shot_type="dunk"),
           ev("b", EventKind.SHOT_MADE, 5.5, pid="home-7", value=2, shot_id="s1"),
           ev("c", EventKind.FOUL, 5.4, team=Team.AWAY, pid="away-11", other="home-7"),
           ev("d", EventKind.BLOCK, 9.0, team=Team.AWAY, pid="away-11", other="home-3", shot_id="s2")]
    assert S.flags_from_events(evs) == ["dunk", "and_one", "block"]


# --------------------------------------------------------------------------- budgets and names


def _situation() -> S.GameSituation:
    return S.GameSituation(home="Harbor", away="Summit", home_score=98, away_score=95, period=4, clock_s=95.0,
                           run_team="Summit", run_points=8, lead_change=True,
                           foul_note="Theo Okafor is one foul from fouling out")


def _long_possession() -> S.PossessionSummary:
    return S.PossessionSummary(
        offense="Harbor", start_cause="defensive_rebound", end="made_shot", shot_value=2, shot_type="layup",
        passes=3, seconds_since_start=14.0, shot_clock_at_end_s=3.0,
        actions=["Jalen Brooks brought the ball up and Marcus Hill set a screen at the top, "
                 "then the defense rotated and recovered"] * 11 + ["the shooter took a layup in traffic"])


def test_b1_state_fits_budget_keeps_terminal_action_and_has_no_names_or_digits() -> None:
    st = S.b1_possession_end(_situation(), _long_possession(), names=NAMES)
    wire = _wire_json(st)
    assert state_bytes(st) <= 1024
    _assert_no_names(wire)
    assert not re.search(r"\d", wire), wire
    assert "_local" not in wire and "redact_names" not in wire
    assert st["possession"]["actions"][-1] == "the shooter took a layup in traffic"  # terminal action kept
    assert st["situation"]["clutch"] == S.CLUTCH_YES and st["situation"]["run"]
    assert st["_local"]["margin_code"] == "one_possession" and st["_local"]["seconds_since_start"] == 14.0


def test_b2_state_budget_and_names() -> None:
    st = S.b2_dead_ball_fan(
        _situation(), stoppage=S.stoppage_text("timeout", "Summit"),
        material={"stat_note": "Jalen Brooks has 28 points tonight, the most on either team " * 3,
                  "tendency": "Summit keeps sending help at Brooks from the left corner", "junk": "x"},
        storylines=["Marcus Hill is two points from a season high"] * 5,
        recent_topics=[f"topic about Theo Okafor number {i}" for i in range(9)],
        storylines_due_now=[True, False, False, True], dead_ball_id="db-12", names=NAMES)
    assert state_bytes(st) <= 1024
    _assert_no_names(_wire_json(st))
    assert len(st["storylines"]) == 3 and st["_local"]["storylines_due_now"] == [True, False, False]
    assert set(st["material"]) == {"stat_note", "tendency"}


def test_b3_state_shuffles_deterministically_with_neutral_keys() -> None:
    cards = [f"card {i}: Marcus Hill 3 fouls in the 2nd quarter, long text " * 3 for i in range(10)]
    st1, map1 = S.b3_dead_ball_coach(_situation(), cards, seed="db-7", staff_team="Harbor",
                                     timeouts="Harbor has two timeouts left", names=NAMES)
    st2, map2 = S.b3_dead_ball_coach(_situation(), cards, seed="db-7", staff_team="Harbor",
                                     timeouts="Harbor has two timeouts left", names=NAMES)
    st3, map3 = S.b3_dead_ball_coach(_situation(), cards, seed="db-8", names=NAMES)
    assert map1 == map2 and map1 != map3
    assert list(st1["cards"]) == list(S.CARD_KEYS) and sorted(map1.values()) == list(range(8))
    assert state_bytes(st1) <= 1024
    _assert_no_names(_wire_json(st1))
    assert st1["situation"]["staff_team"] == "Harbor"


def test_b4_state_projects_facts_and_never_alters_sentences() -> None:
    facts = {"state_version": 41, "clock": "fourth quarter, under two minutes", "score": "Harbor 98, Summit 95",
             "recent_events": [f"{{P0{i % 9}}} made a layup over Theo Okafor" for i in range(12)],
             "players": {f"P{i:02d}": {"pts": i, "reb": 2, "ast": 1.5} for i in range(30)},
             "derived": ["Harbor answered an eight-nothing run"] * 6, "allowed_numbers": ["98", "95"],
             "allowed_tokens": ["P01"], "recent_lines": ["x"], "hedge_only": False,
             "labels": {"last_play_type": "Spotup"}}
    sentences = ["{P07} stops the bleeding with a strong drive that silences the arena tonight.",
                 "Summit needs a timeout answer here."]
    st = S.b4_line_guard(facts, sentences, names=NAMES)
    assert state_bytes(st) <= 1024
    assert st["lines"] == {"s1": sentences[0], "s2": sentences[1]}
    for k in ("state_version", "allowed_numbers", "allowed_tokens", "recent_lines", "hedge_only"):
        assert k not in st["facts"]
    _assert_no_names(_wire_json(st))
    assert not any(isinstance(v, float) for v in json.loads(_wire_json(st))["facts"].values())


def test_b5_state_buckets_numbers_and_drops_raw_measurements() -> None:
    st = S.b5_event_ambiguity("shot_outcome", {"drop_speed_frac": 0.4, "net_motion_z": 3.2,
                                               "ball_in_basket_frames": 1, "rim_contact": True,
                                               "crossing_margin_ft": 0.31, "view_quality": "clear view of the rim",
                                               "_local": {"forbidden": ["make"]}})
    shot = st["shot"]
    assert shot["ball_after_rim"].startswith("reappeared below the net")
    assert shot["net_motion"].startswith("strong") and shot["ball_in_basket_detections"] == "detected in one frame"
    assert shot["rim_contact"] == "yes" and "crossing_margin_ft" not in shot
    assert st["_local"] == {"forbidden": ["make"]} and state_bytes(st) <= 614
    with pytest.raises(ValueError):
        S.b5_event_ambiguity("foul", {})


def test_redact_handles_full_names_and_surnames_case_insensitively() -> None:
    text = "Jalen Brooks fed HILL, then brooks again; Hillside stays"
    out = S.redact(text, NAMES)
    assert "Brooks" not in out and "brooks" not in out and "HILL" not in out
    assert "Hillside" in out  # word boundaries respected


def test_sanitize_strips_private_keys_and_non_finite_numbers() -> None:
    st = {"a": float("nan"), "b": [1.0, float("inf")], "_local": {"x": 1}, "c": {"_y": 2, "z": Team.HOME}}
    assert sanitize(st) == {"a": None, "b": [1.0, None], "c": {"z": "home"}}
