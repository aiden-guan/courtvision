"""Templates: every family renders, validates against its own facts, respects word caps; hedge lines
never state an outcome or a point value; variety without immediate repeats."""

from __future__ import annotations

import random
import re
from dataclasses import replace

import pytest
from test_narration_core_support import (COURT, FakeShot, ev, make_frame, make_roster, make_stats,
                                         scenario_events, shot_made)

from courtvision.narration import templates as T
from courtvision.narration.facts import build_facts
from courtvision.narration.pseudonym import PseudonymMap
from courtvision.narration.templates import TemplateConfig, TemplateRenderer, family_cap, render, render_line
from courtvision.narration.validator import (OUTCOME_RE, GroundingValidator, count_words, extract_numbers,
                                             jaccard, split_sentences)
from courtvision.types import EventKind, EventStatus, Team

V = GroundingValidator()
CFG = TemplateConfig()
OUTCOME_SCENARIOS = ["made3", "made2", "layup", "dunk", "ft_made", "made_assisted", "miss", "ft_miss", "block",
                     "steal", "turnover", "dreb", "oreb", "team_reb", "release3", "foul"]


def _facts(events, roster=None, stats=None, **kw):
    roster = roster or make_roster()
    stats = stats or make_stats(shots=[FakeShot("s0040", "home-23", Team.HOME, 1, 2, True),
                                       FakeShot("s0041", "home-23", Team.HOME, 2, 2, True),
                                       FakeShot("s0042", "home-23", Team.HOME, 3, 3, True)],
                                run={"team": Team.HOME, "points": 10, "opp_points": 2, "since_t": 250.0})
    return build_facts(events, stats, roster, COURT, make_frame(stats), PseudonymMap(roster, seed=7),
                       state_version=1, **kw)


# --------------------------------------------------------------------------- bank shape


def test_every_play_by_play_family_has_at_least_eight_variants() -> None:
    for fam, variants in T.PBP.items():
        assert len(variants) >= 8, fam
        assert len(set(variants)) == len(variants), fam
    for fam in ("assist", "tail_lead", "tail_tie"):
        assert len(T.ADDONS[fam]) >= 8, fam
    for fam in ("no_basket", "value", "scorer"):
        assert len(T.CORRECTION[fam]) >= 8, fam


def _full_ctx() -> dict[str, str]:
    return {"actor": "{P01}", "actor_tok": "{P01}", "other_tok": "{P02}", "other": "{P02}", "team": "Riverside",
            "opp": "Lakeport", "zone": "left wing", "value_word": "three", "pts": "12", "fg": "5-for-8",
            "reb": "4", "blk": "2", "stl": "1", "tov": "1", "pf": "3", "pf_word": "three", "team_tov": "6",
            "team_pf": "8", "opp_tov": "9", "how": "shot-clock violation", "reb_type": "offensive",
            "home": "Riverside", "away": "Lakeport", "h": "42", "a": "40", "leader": "Riverside",
            "margin_word": "two", "lead_score": "42-40", "period": "second quarter", "run_team": "Riverside",
            "run_score": "10-2", "assister": "{P02}", "streak_ord": "third", "jumper": "jumper",
            "pull_up": "pulls up"}


def test_every_variant_fits_its_family_cap() -> None:
    ctx = _full_ctx()
    for fam, variants in T.PBP.items():
        for v in variants:
            text = v.format_map(ctx)
            assert count_words(text) <= family_cap(fam), (fam, text)
    for fam, variants in T.COACH.items():
        for v in variants:
            assert count_words(v.format_map(ctx)) <= 12, (fam, v)
    for fam, variants in T.CORRECTION.items():
        for v in variants:
            text = v.format_map({"new": "{P01}", "old": "{P02}", "new_word": "three", "old_word": "two",
                                 "new_team": "Riverside"})
            assert count_words(text) <= 12 and text.startswith("Correction"), (fam, text)


def test_hedge_variants_never_state_outcome_or_point_value() -> None:
    ctx = _full_ctx()
    hedge_fams = [f for f in T.PBP if f.startswith("hedge_")] + ["release", "whistle", "ft_attempt"]
    for fam in hedge_fams:
        for v in T.PBP[fam]:
            text = v.format_map(ctx)
            assert not OUTCOME_RE.search(text), (fam, text)
            assert not extract_numbers(text), (fam, text)
    for fam in ("hedge", "hedge_rebound", "hedge_loose"):
        for v in T.COACH[fam]:
            text = v.format_map(ctx)
            assert not OUTCOME_RE.search(text) and not extract_numbers(text), (fam, text)


# --------------------------------------------------------------------------- render + validate


@pytest.mark.parametrize("name", list(scenario_events()))
@pytest.mark.parametrize("channel", ["play_by_play", "coach", "accessibility"])
@pytest.mark.parametrize("hedge", [False, True])
def test_every_kind_renders_and_validates_against_its_own_facts(name: str, channel: str, hedge: bool) -> None:
    events = scenario_events()[name]
    facts = _facts(events)
    for seed in range(12):
        line = render_line(events, facts, channel, random.Random(seed), hedge)
        if channel == "play_by_play":
            assert line.text, name  # every kind has a live call
        if not line.text:
            continue
        res = V.validate(line.text, facts, channel)
        assert res.ok, (name, channel, line.text, res.problems)
        if channel == "accessibility":
            core = [s for s in split_sentences(line.text) if not re.fullmatch(r"\w+ \d+, \w+ \d+\.", s)]
            assert sum(count_words(s) for s in core) <= 6, line.text
        else:
            assert count_words(line.text) <= CFG.line_cap[channel], line.text
        if channel == "play_by_play" and line.family not in T._MADE_FAMILIES:
            assert count_words(line.text) <= family_cap(line.family), (line.family, line.text)


@pytest.mark.parametrize("name", OUTCOME_SCENARIOS)
@pytest.mark.parametrize("channel", ["play_by_play", "coach", "accessibility"])
def test_hedge_lines_have_no_outcome_words_or_point_values(name: str, channel: str) -> None:
    events = scenario_events()[name]
    facts = _facts(events)
    for seed in range(20):
        text = render(events, facts, channel, random.Random(seed), hedge=True)
        assert not OUTCOME_RE.search(re.sub(r"\{P\d+\}", "", text)), (name, text)
        assert not extract_numbers(text), (name, text)


def test_hedge_only_packet_forces_hedge_wording() -> None:
    e = shot_made(3, status=EventStatus.PROVISIONAL, confidence=0.99)
    e.attrs["p_status"] = "uncalibrated"
    facts = _facts([e])
    assert facts["hedge_only"]
    for seed in range(20):
        line = render_line([e], facts, "play_by_play", random.Random(seed), hedge=False)
        assert line.family == "hedge_shot"
        assert V.validate(line.text, facts, "play_by_play").ok


@pytest.mark.parametrize("name", ["made3", "made2", "layup", "dunk", "block", "steal", "turnover", "dreb", "oreb",
                                  "miss", "release3", "possession", "bring_up"])
def test_at_least_eight_distinct_live_calls(name: str) -> None:
    events = scenario_events()[name]
    facts = _facts(events)
    seen = {render(events, facts, "play_by_play", random.Random(s)) for s in range(300)}
    assert len(seen) >= 8, (name, seen)


def test_rendered_line_reports_hedge_wording() -> None:
    events = scenario_events()["made3"]
    facts = _facts(events)
    assert render_line(events, facts, "play_by_play", random.Random(0), hedge=True).hedged
    assert not render_line(events, facts, "play_by_play", random.Random(0)).hedged
    assert render_line(events, facts, "coach", random.Random(0), hedge=True).hedged


def test_made_call_with_assist_streak_and_score_tail() -> None:
    events = scenario_events()["made_assisted"]
    facts = _facts(events)
    pm = PseudonymMap(make_roster(), seed=7)
    assister = "{" + pm.token("home-3") + "}"
    texts = [render(events, facts, "play_by_play", random.Random(s)) for s in range(60)]
    assert any(assister in t for t in texts)  # assist add-on
    assert any("third" in t.lower() for t in texts)  # streak add-on from the derived phrase
    assert any("Riverside" in t for t in texts)  # lead-change tail (confirmed score)
    for t in texts:
        assert count_words(t) <= 12
        assert V.validate(t, facts, "play_by_play").ok, t


def test_score_tail_only_from_confirmed_score() -> None:
    stats = make_stats(score_source="events")
    events = [shot_made(3)]
    facts = build_facts(events, stats, make_roster(), COURT, None, PseudonymMap(make_roster(), seed=7))
    assert not facts["score"]["confirmed"]
    for s in range(40):
        for ch in ("play_by_play", "accessibility"):
            t = render(events, facts, ch, random.Random(s))
            assert not re.search(r"\d", re.sub(r"\{P\d+\}", "", t)), t


def test_accessibility_lexicon_and_score_after_possession() -> None:
    sc = scenario_events()
    facts = _facts(sc["made3"])
    assert render(sc["made3"], facts, "accessibility", random.Random(0)).endswith("Riverside 42, Lakeport 40.")
    assert "left wing, good." in render(sc["made3"], facts, "accessibility", random.Random(0))
    miss = render(sc["miss"], _facts(sc["miss"]), "accessibility", random.Random(0))
    assert miss.endswith("left wing, no good.")
    assert render(sc["block"], _facts(sc["block"]), "accessibility", random.Random(0)) == "Shot, paint, blocked."
    assert render(sc["dreb"], _facts(sc["dreb"]), "accessibility", random.Random(0)).startswith("Rebound, Lakeport.")
    assert render(sc["steal"], _facts(sc["steal"]), "accessibility", random.Random(0)).startswith("Turnover, Riverside.")
    assert render(sc["possession"], _facts(sc["possession"]), "accessibility", random.Random(0)) == "Lakeport ball."
    hedged = render(sc["made3"], facts, "accessibility", random.Random(0), hedge=True)
    assert "good" not in hedged and not re.search(r"\d", re.sub(r"\{P\d+\}", "", hedged))


def test_coach_cards_are_flat_and_numeric() -> None:
    sc = scenario_events()
    facts = _facts(sc["made3"])
    cards = {render(sc["made3"], facts, "coach", random.Random(s)) for s in range(40)}
    assert any("12" in c and "5-for-8" in c for c in cards)
    assert all(count_words(c) <= 12 for c in cards)
    assert render(sc["release3"], _facts(sc["release3"]), "coach", random.Random(0)) == ""  # no card for a release


@pytest.mark.parametrize("change", ["scorer", "value", "retract_make", "retract_miss", "flip_to_made", "flip_to_miss",
                                    "foul", "timeout", "rebound"])
def test_correction_lines(change: str) -> None:
    roster = make_roster()
    pm = PseudonymMap(roster, seed=7)
    if change in ("scorer", "value", "retract_make", "flip_to_miss"):
        old = shot_made(3)
    elif change in ("retract_miss", "flip_to_made"):
        old = ev(EventKind.SHOT_MISSED, team=Team.HOME, player_id="home-23")
    elif change == "foul":
        old = ev(EventKind.FOUL, team=Team.HOME, player_id="home-23", source="manual")
    elif change == "timeout":
        old = ev(EventKind.TIMEOUT, team=Team.HOME, source="manual")
    else:
        old = ev(EventKind.REBOUND, team=Team.HOME, player_id="home-23", attrs={"offensive": True})
    new = replace(old, revision=1, attrs=dict(old.attrs))
    if change in ("scorer", "foul", "rebound"):
        new.player_id = "home-3"
    elif change == "value":
        new.value = 2
    elif change == "timeout":
        new.team = Team.AWAY
    elif change == "flip_to_made":  # the engine revises the result kind in place (same event_id)
        new.kind, new.value = EventKind.SHOT_MADE, 2
    elif change == "flip_to_miss":
        new.kind = EventKind.SHOT_MISSED
    else:
        new.status = EventStatus.RETRACTED
    stats = make_stats()
    facts = build_facts([new], stats, roster, COURT, make_frame(stats), pm, previous=[old])
    # A retracted miss means no shot happened, never "that shot was good" (regression).
    expect = {"scorer": "scorer", "value": "value", "retract_make": "no_basket", "retract_miss": "generic",
              "flip_to_made": "was_good", "flip_to_miss": "no_basket", "foul": "foul", "timeout": "timeout",
              "rebound": "credit"}[change]
    for s in range(10):
        line = render_line([new], facts, "correction", random.Random(s))
        assert line.family == f"correction:{expect}", line
        assert line.text.startswith("Correction") and count_words(line.text) <= 12
        res = V.validate(line.text, facts, "correction")
        assert res.ok, (line.text, res.problems)
    if change == "scorer":
        text = render([new], facts, "correction", random.Random(0))
        assert "{" + pm.token("home-3") + "}" in text


def test_correction_without_details_is_generic() -> None:
    facts = _facts([shot_made(3)])
    line = render_line([shot_made(3)], facts, "correction", random.Random(0))
    assert line.family == "correction:generic" and V.validate(line.text, facts, "correction").ok


def test_period_start_and_end_lines() -> None:
    sc = scenario_events()
    for name in ("period_start", "period_end"):
        facts = _facts(sc[name])
        texts = {render(sc[name], facts, "play_by_play", random.Random(s)) for s in range(80)}
        assert len(texts) >= 8
        assert any("second quarter" in t for t in texts)
        assert all(V.validate(t, facts, "play_by_play").ok for t in texts)


def test_color_channel_uses_derived_phrases_or_run() -> None:
    events = [shot_made(3)]
    facts = _facts(events)
    texts = {render(events, facts, "color", random.Random(s)) for s in range(30)}
    assert texts and all(V.validate(t, facts, "color").ok for t in texts)
    assert any(t.startswith("Third straight make") for t in texts)
    empty = _facts([], stats=make_stats())
    t = render([], empty, "color", random.Random(0))
    assert t and V.validate(t, empty, "color").ok  # score update from the confirmed score


def test_score_update_with_no_events() -> None:
    facts = _facts([], stats=make_stats(score={Team.HOME: 40, Team.AWAY: 40}))
    t = render([], facts, "play_by_play", random.Random(3))
    assert "40" in t and V.validate(t, facts, "play_by_play").ok


# --------------------------------------------------------------------------- variety


def test_no_variant_repeats_within_three_uses() -> None:
    events = scenario_events()["made3"]
    facts = _facts(events)
    r = TemplateRenderer(seed=5)
    picks = [r.render_line(events, facts, "play_by_play") for _ in range(30)]
    idx = [p.variant for p in picks]
    for i in range(len(idx)):
        assert idx[i] not in idx[max(0, i - 3):i], idx
    texts = [p.text for p in picks]
    assert all(a != b for a, b in zip(texts, texts[1:]))


def test_stateless_render_avoids_recent_lines() -> None:
    events = scenario_events()["dreb"]
    recent: list[str] = []
    for s in range(15):  # 9 dreb variants vs the last 6 utterances (narration.md §3.2 recent_utterances)
        facts = _facts(events, recent_lines=recent[-6:])
        text = render(events, facts, "play_by_play", random.Random(s))
        assert all(jaccard(text, r) < 0.8 for r in recent[-6:]), (text, recent)
        assert V.validate(text, facts, "play_by_play").ok
        recent.append(text)


def test_rendering_is_deterministic_per_seed() -> None:
    events = scenario_events()["steal"]
    facts = _facts(events)
    a = [render(events, facts, "play_by_play", random.Random(9)) for _ in range(3)]
    assert len(set(a)) == 1


def test_two_word_team_names_still_fit_caps() -> None:
    roster = make_roster("Riverside Hawks", "Lakeport Owls")
    for name, events in scenario_events().items():
        facts = _facts(events, roster=roster)
        for ch in ("play_by_play", "coach", "accessibility"):
            for s in range(6):
                text = render(events, facts, ch, random.Random(s))
                if text:
                    res = V.validate(text, facts, ch)
                    assert res.ok, (name, ch, text, res.problems)


def test_unknown_channel_raises() -> None:
    with pytest.raises(ValueError):
        render([], {}, "radio", random.Random(0))


def test_end_to_end_render_names_after_validation() -> None:
    roster = make_roster()
    pm = PseudonymMap(roster, seed=7)
    events = scenario_events()["made3"]
    stats = make_stats()
    facts = build_facts(events, stats, roster, COURT, make_frame(stats), pm)
    text = render(events, facts, "play_by_play", random.Random(2))
    assert V.validate(text, facts, "play_by_play").ok
    spoken = pm.render(text, roster)
    assert "{" not in spoken
    assert "Riley Thorsen" in spoken or "Riverside" in spoken
