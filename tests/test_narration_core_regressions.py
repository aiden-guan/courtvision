"""Regression tests for defects found in the adversarial review of narration-core."""

from __future__ import annotations

import json
import random
import re
from dataclasses import replace

import pytest
from test_narration_core_support import (COURT, FakeShot, ev, make_frame, make_roster, make_stats, scenario_events,
                                         shot_made)

from courtvision.geometry.rules import rules_for
from courtvision.narration import templates as T
from courtvision.narration.director import Cue, Director, is_material_change, priority_for
from courtvision.narration.facts import build_facts, trust_map
from courtvision.narration.pseudonym import PseudonymMap, pseudonyms_for
from courtvision.narration.validator import OUTCOME_RE, GroundingValidator
from courtvision.types import EventKind, EventStatus, Team

ROSTER = make_roster()
PM = PseudonymMap(ROSTER, seed=7)
V = GroundingValidator()


def _facts(events, stats=None, frame=True, **kw):
    stats = stats or make_stats()
    return build_facts(events, stats, ROSTER, COURT, make_frame(stats) if frame else None, PM, **kw)


def _calibrated(e, conf: float = 0.99, **attrs):
    e.status, e.confidence = EventStatus.PROVISIONAL, conf
    e.attrs.update({"p_status": "platt", **attrs})
    return e


# --------------------------------------------------------------------------- validator: score phrases


@pytest.mark.parametrize("text", [
    "Riverside 40, Lakeport 42.",  # swapped (score is Riverside 42, Lakeport 40)
    "Lakeport 42, Riverside 40.",  # wrong team for each number
    "Riverside forty, Lakeport forty-two.",
    "It's 42 to 41.",
    "Lakeport has the lead.",
    "Lakeport with the lead.",
    "Riverside down two.",
    "Riverside behind by two.",
    "Lakeport is winning.",
    "Riverside is losing by two.",
    "A five-point game.",  # 5 is in the packet (a field-goal count) but the margin is 2
    "Riverside holds a 3-point lead.",
])
def test_wrong_score_phrases_fail(text: str) -> None:
    res = V.validate(text, _facts([shot_made(3)]), "color")
    assert not res.ok, text
    assert any(p.startswith(("score inconsistent", "leader", "trailer", "lead inconsistent", "margin"))
               for p in res.problems), res.problems


@pytest.mark.parametrize("text", [
    "Riverside 42, Lakeport 40.",
    "Riverside forty-two, Lakeport forty.",
    "Riverside has the lead.",
    "Lakeport trails by two.",
    "A two-point game.",
    "Lakeport is winning the rebound battle.",  # not about the score
    "Riverside 10, Lakeport 2 in this stretch.",  # the run split from the packet
    "It's 42 to 40.",
])
def test_grounded_score_phrases_pass(text: str) -> None:
    stats = make_stats(run={"team": Team.HOME, "points": 10, "opp_points": 2, "since_t": 250.0})
    res = V.validate(text, _facts([shot_made(3)], stats), "color")
    assert res.ok, (text, res.problems)


# --------------------------------------------------------------------------- facts: trust rules


def _assist(shot_id: str = "s0042"):
    return ev(EventKind.ASSIST, status=EventStatus.PROVISIONAL, team=Team.HOME, player_id="home-3",
              other_player_id="home-23", shot_id=shot_id, confidence=0.9)


def test_assist_without_cue_model_inherits_the_make() -> None:
    made = _calibrated(shot_made(3))
    a = _assist()
    facts = _facts([made, a])
    assert facts["hedge_only"] is False  # was True: the assist's missing p_status hedged every assisted make
    text = T.render([made, a], facts, "play_by_play", random.Random(0))
    assert not text.startswith("Shot") and V.validate(text, facts, "play_by_play").ok
    uncal = shot_made(3, status=EventStatus.PROVISIONAL, confidence=0.99, attrs={"p_status": "uncalibrated"})
    assert _facts([uncal, _assist()])["hedge_only"] is True
    assert trust_map([_assist()])  # an assist alone (no make in the batch) is not trusted
    assert not next(iter(trust_map([_assist()]).values()))


def test_made_three_needs_the_three_gate_from_its_attempt() -> None:
    att = _calibrated(ev(EventKind.SHOT_ATTEMPT, team=Team.HOME, player_id="home-23", value=3, shot_id="s0042"),
                      p_three=0.80)
    made = _calibrated(shot_made(3))
    trust = trust_map([att, made])
    assert trust[made.event_id] is False  # "buries the three" needs P(3) >= 0.97 (RESEARCH §4.1)
    att.attrs["p_three"] = 0.99
    assert trust_map([att, made])[made.event_id] is True
    own = _calibrated(shot_made(3), p_three=0.5)
    assert trust_map([own])[own.event_id] is False
    two = _calibrated(shot_made(2))
    assert trust_map([replace(att, value=2, attrs={**att.attrs, "p_three": 0.1}), two])[two.event_id] is True


def test_unconfirmed_score_from_provisional_baskets_is_not_speakable() -> None:
    stats = make_stats(score_source="events",
                       shots=[FakeShot("s0042", "home-23", Team.HOME, 3, 3, True)])
    stats.shots[0].provisional = True  # type: ignore[attr-defined]
    timeout = ev(EventKind.TIMEOUT, team=Team.AWAY, source="manual")
    facts = _facts([timeout], stats, frame=False)
    assert facts["hedge_only"] is False and facts["score"]["speakable"] is False
    assert not V.validate("Riverside leads 42-40.", facts, "color").ok
    assert not V.validate("Riverside 42, Lakeport 40.", facts, "color").ok
    stats.shots[0].provisional = False  # type: ignore[attr-defined]
    assert _facts([timeout], stats, frame=False)["score"]["speakable"] is True


def test_no_lead_phrase_when_the_scoreboard_disagrees() -> None:
    stale = make_stats(score_source="scoreboard", score_mismatch=True)  # board may not include this basket
    derived = _facts([shot_made(3)], stale, frame=False)["derived"]
    assert not any("lead" in d or "ties" in d for d in derived)


# --------------------------------------------------------------------------- facts: robustness


def test_nan_coordinates_and_clocks_do_not_crash() -> None:
    bad = ev(EventKind.REBOUND, team=Team.AWAY, player_id="away-11", x=float("nan"), y=3.0,
             attrs={"offensive": False})
    shot = shot_made(3, x=float("nan"), y=float("inf"))
    stats = make_stats(game_clock=float("nan"), shot_clock=float("nan"))
    facts = _facts([bad, shot], stats, frame=False)
    assert all("zone" not in f for f in facts["events"])
    assert "game" not in facts["clock"] and "shot" not in facts["clock"]
    assert not any("nan" in n for n in facts["allowed_numbers"])
    json.dumps(facts)  # still a plain JSON packet


def test_shot_clock_tenths_are_truncated() -> None:
    assert _facts([], make_stats(shot_clock=4.97), frame=False)["clock"]["shot"] == "4.9"


def test_non_shot_zone_uses_the_attacking_basket_when_known() -> None:
    poss = ev(EventKind.POSSESSION, team=Team.AWAY, player_id="away-0", x=30.0, y=25.0,
              attrs={"cause": "inbound", "basket": "right"})
    assert _facts([poss])["events"][0]["zone"] == "backcourt"  # nearest basket would say "top"
    del poss.attrs["basket"]
    assert _facts([poss])["events"][0]["zone"] == "top"


def test_default_pseudonym_map_is_shared_per_roster() -> None:
    roster = make_roster()
    assert pseudonyms_for(roster) is pseudonyms_for(roster)
    a = ev(EventKind.REBOUND, team=Team.HOME, player_id="home-t12", attrs={"offensive": True}, source="manual")
    b = ev(EventKind.REBOUND, team=Team.AWAY, player_id="away-t3", attrs={"offensive": True}, source="manual")
    f1 = build_facts([b], make_stats(), roster, COURT, None)
    f2 = build_facts([a], make_stats(), roster, COURT, None)
    f3 = build_facts([b, a], make_stats(), roster, COURT, None)
    # provisional tokens stay the same across packets (a fresh map per call reissued them)
    assert f1["events"][0]["actor"] == f3["events"][0]["actor"]
    assert f2["events"][0]["actor"] == f3["events"][1]["actor"]
    shown = pseudonyms_for(roster).render("{" + f2["events"][0]["actor"] + "}.", roster)
    assert shown == "Riverside."


# --------------------------------------------------------------------------- templates


def test_colour_in_hedge_only_packets_states_no_outcome() -> None:
    stats = make_stats(shots=[FakeShot(f"s004{i}", "home-23", Team.HOME, float(i), 2 if i < 2 else 3, True)
                              for i in range(3)],
                       run={"team": Team.HOME, "points": 10, "opp_points": 2, "since_t": 250.0})
    made = shot_made(3)
    reb = ev(EventKind.REBOUND, status=EventStatus.PROVISIONAL, team=Team.AWAY, player_id="away-11",
             attrs={"offensive": False, "p_status": "uncalibrated"})
    facts = _facts([made, reb], stats)
    assert facts["hedge_only"] and any("points" in d or "make" in d for d in facts["derived"])
    for s in range(40):
        text = T.render([made, reb], facts, "color", random.Random(s))
        assert text and not OUTCOME_RE.search(re.sub(r"\{P\d+\}", "", text)), text
        assert V.validate(text, facts, "color").ok, text


@pytest.mark.parametrize("hedge_kind", ["shot_attempt", "rebound"])
def test_every_channel_validates_in_hedge_only_packets(hedge_kind: str) -> None:
    stats = make_stats(run={"team": Team.HOME, "points": 10, "opp_points": 2, "since_t": 250.0})
    extra = ev(EventKind(hedge_kind), status=EventStatus.PROVISIONAL, team=Team.AWAY, player_id="away-0",
               value=2 if hedge_kind == "shot_attempt" else 0, attrs={"p_status": "uncalibrated", "offensive": False})
    for name, events in scenario_events().items():
        combo = [*events, extra]
        facts = _facts(combo, stats)
        assert facts["hedge_only"]
        for ch in ("play_by_play", "color", "coach", "accessibility"):
            for s in range(6):
                text = T.render(combo, facts, ch, random.Random(s))
                if text:
                    res = V.validate(text, facts, ch)
                    assert res.ok, (name, ch, text, res.problems)


# --------------------------------------------------------------------------- director


def test_release3_gets_the_release_ttl() -> None:
    d = Director()
    assert d.effective_ttl(Cue("a", 1, "x...", 0.0, kind="release3")) == pytest.approx(0.8)
    assert d.effective_ttl(Cue("b", 1, "x...", 0.0, kind="release")) == pytest.approx(0.8)


def test_correction_family_kind_is_a_correction() -> None:
    d = Director()
    c = Cue("c", 0, "Correction: x.", 0.0, event_ids=["e1"], channel="play_by_play", kind="correction:scorer")
    assert d.effective_ttl(c) == pytest.approx(10.0)
    d.offer(c, 0.0)
    assert d.next(0.0) is c
    d.on_spoken(c, 1.0)
    later = replace(shot_made(3), event_id="e1")
    d.on_spoken(Cue("m", 1, "made.", 2.0, event_ids=["e1"], events=[later], kind="made3"), 3.0)
    assert d.on_revision(replace(later, player_id="home-3", revision=1), 4.0).action == "suppressed"


def test_result_flips_are_material_and_retracted_misses_are_not() -> None:
    miss = ev(EventKind.SHOT_MISSED, team=Team.HOME, player_id="home-23", value=3, shot_id="s1")
    made = replace(miss, kind=EventKind.SHOT_MADE, revision=1)  # engine flips the kind in place
    assert is_material_change(miss, made)
    assert is_material_change(made, replace(made, kind=EventKind.SHOT_MISSED, revision=2))
    assert not is_material_change(miss, replace(miss, status=EventStatus.RETRACTED))
    assert is_material_change(made, replace(made, status=EventStatus.RETRACTED))
    d = Director()
    c = Cue("u1", 1, "{P01} misses.", 0.0, event_ids=[miss.event_id], events=[miss], kind="miss")
    d.offer(c, 0.0)
    assert d.next(0.0) is c
    d.on_spoken(c, 1.0)
    assert d.on_revision(made, 3.0).action == "correct"  # a spoken miss that went in needs a correction


def test_revert_drops_a_queued_correction() -> None:
    d = Director()
    e0 = shot_made(3)
    c = Cue("u1", 1, "{P01} buries it.", 0.0, event_ids=[e0.event_id], events=[e0], kind="made3")
    d.offer(c, 0.0)
    assert d.next(0.0) is c
    d.on_spoken(c, 1.0)
    e1 = replace(e0, player_id="home-3", revision=1)
    assert d.on_revision(e1, 2.0).action == "correct"
    corr = Cue("u2", 0, "Correction: {P02} scored.", 2.0, event_ids=[e0.event_id], events=[e1],
               channel="correction", kind="correction")
    d.offer(corr, 2.0)
    e2 = replace(e0, revision=2)  # back to what the audience heard
    out = d.on_revision(e2, 2.1)
    assert out.action == "invalidated" and out.dropped == 1
    assert d.pending(2.1) == []  # the now-wrong correction is never spoken


def test_foul_out_is_p0() -> None:
    stats = make_stats()
    stats.players["home-23"].pf = 6
    foul = ev(EventKind.FOUL, team=Team.HOME, player_id="home-23", source="manual")
    facts = _facts([foul], stats, rules=rules_for("nba"))
    assert priority_for("foul", facts) == 0
    stats.players["home-23"].pf = 3
    assert priority_for("foul", _facts([foul], stats, rules=rules_for("nba"))) == 1


def test_shot_type_wording_only_for_jump_shots() -> None:
    for shot_type in ("tip", "hook", None):
        made = shot_made(2, x=8.0, y=22.0, attrs={"shot_type": shot_type})
        att = ev(EventKind.SHOT_ATTEMPT, team=Team.HOME, player_id="home-23", value=2, x=8.0, y=22.0,
                 attrs={"shot_type": shot_type, "basket": "left"})
        for evs in ([made], [att]):
            facts = _facts(evs)
            texts = {T.render(evs, facts, "play_by_play", random.Random(s)) for s in range(200)}
            assert not any("jumper" in t or "pulls up" in t for t in texts), (shot_type, texts)
    jumper = [shot_made(2, x=15.0, y=20.0)]
    texts = {T.render(jumper, _facts(jumper), "play_by_play", random.Random(s)) for s in range(200)}
    assert any("jumper" in t for t in texts)


def test_miss_calls_do_not_claim_rim_contact() -> None:
    for fam in ("miss", "ft_miss"):
        assert len(T.PBP[fam]) >= 8
        assert not any("rim" in v.lower() for v in T.PBP[fam]), fam  # an air ball never touches the rim


def test_hedge_only_allows_a_confirmed_score_line_but_no_point_values() -> None:
    stats = make_stats(score={Team.HOME: 63, Team.AWAY: 40})
    reb = ev(EventKind.REBOUND, status=EventStatus.PROVISIONAL, team=Team.AWAY, player_id="away-11",
             attrs={"offensive": False, "p_status": "uncalibrated"})
    facts = _facts([reb], stats)
    assert facts["hedge_only"] and facts["score"]["confirmed"]
    for text in ("The score: 63-40, Riverside in front.", "Score check: Riverside 63, Lakeport 40.",
                 "Riverside ahead by twenty-three.", "Riverside up twenty-three, 63-40."):
        res = V.validate(text, facts, "color")
        assert res.ok, (text, res.problems)
    tok = "{" + PM.token("away-11") + "}"
    for text in (f"{tok} for three.", f"A three from {tok}.", f"{tok} scores.", f"{tok} can score here.",
                 f"{tok} hits a three-pointer.", "Two more for Lakeport."):
        assert any(p.startswith("outcome word") for p in V.validate(text, facts, "color").problems), text


def test_margins_above_ninety_nine_in_words() -> None:
    stats = make_stats(score={Team.HOME: 4, Team.AWAY: 110})
    facts = _facts([shot_made(2, team=Team.AWAY, pid="away-11")], stats)
    assert V.validate("Lakeport up one hundred six, 110-4.", facts, "color").ok
    assert not V.validate("Lakeport up one hundred five.", facts, "color").ok


def test_hedge_only_packets_hide_stats_the_unconfirmed_event_changed() -> None:
    made = shot_made(3, status=EventStatus.PROVISIONAL, confidence=0.99, attrs={"p_status": "uncalibrated"})
    blk = ev(EventKind.BLOCK, status=EventStatus.PROVISIONAL, team=Team.AWAY, player_id="away-11",
             other_player_id="home-3", attrs={"p_status": "uncalibrated"})
    facts = _facts([made, blk])
    scorer, shooter, blocker = (facts["players"][PM.token(p)] for p in ("home-23", "home-3", "away-11"))
    for line in (scorer, shooter):
        assert not {"pts", "fgm", "fga", "fg", "fg3", "fg3m", "fg3a"} & set(line), line  # "5-for-8" counts the make
    assert "blk" not in blocker and "dreb" in blocker  # unrelated stats stay
    assert "pts" not in facts["teams"]["home"] and "fgm" not in facts["teams"]["home"]
    assert "blk" not in facts["teams"]["away"] and "tov" in facts["teams"]["away"]
    tok = "{" + PM.token("home-23") + "}"
    assert not V.validate(f"{tok} is 5-for-8 from the field.", facts, "color").ok
    trusted = _facts([shot_made(3)])
    assert trusted["players"][PM.token("home-23")]["fg"] == "5-for-8"


@pytest.mark.parametrize("to_type,family", [("out_of_bounds", "hedge_stop"), ("shot_clock", "hedge_stop"),
                                            ("travel", "hedge_stop"), ("bad_pass", "hedge_loose")])
def test_unconfirmed_dead_ball_turnover_is_not_a_loose_ball(to_type: str, family: str) -> None:
    to = ev(EventKind.TURNOVER, status=EventStatus.PROVISIONAL, team=Team.HOME, player_id="home-23",
            confidence=0.8, attrs={"type": to_type, "needs_review": True})
    facts = _facts([to])
    assert facts["hedge_only"]
    for s in range(10):
        line = T.render_line([to], facts, "play_by_play", random.Random(s))
        assert line.family == family and line.hedged, line
        assert V.validate(line.text, facts, "play_by_play").ok, line.text
    for ch in ("coach", "accessibility"):
        text = T.render([to], facts, ch, random.Random(0))
        assert text and V.validate(text, facts, ch).ok and "loose" not in text.lower() or family == "hedge_loose"
    assert priority_for(family) == 1


def test_lagging_scoreboard_is_not_confirmed_by_its_own_frame_read() -> None:
    stats = make_stats(score={Team.HOME: 40, Team.AWAY: 40}, score_source="scoreboard", score_mismatch=True)
    made = shot_made(2, x=15.0, y=20.0)  # home basket just called; the board still shows 40-40
    facts = build_facts([made], stats, ROSTER, COURT, make_frame(stats, "ocr"), PM)
    assert facts["score"]["confirmed"] is False
    for s in range(30):
        for ch in ("play_by_play", "accessibility"):
            text = T.render([made], facts, ch, random.Random(s))
            assert "40" not in text and "tied" not in text.lower(), text
