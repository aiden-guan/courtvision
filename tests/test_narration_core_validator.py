"""GroundingValidator: tokens, numbers, claims, score consistency, tense, length, repetition."""

from __future__ import annotations

import random
import time

import pytest
from test_narration_core_support import COURT, FakeShot, make_frame, make_roster, make_stats, shot_made

from courtvision.narration.facts import build_facts
from courtvision.narration.pseudonym import PseudonymMap
from courtvision.narration.validator import (GroundingValidator, ValidatorConfig, extract_numbers, int_to_words,
                                             jaccard, number_keys, ordinal_words)
from courtvision.types import Team

ROSTER = make_roster()
PM = PseudonymMap(ROSTER, seed=7)
TOK = PM.token("home-23")  # scorer: 12 pts, 5-for-8, 2-for-4 from three
V = GroundingValidator()


def _facts(**kw):
    stats = make_stats(shots=[FakeShot("s0040", "home-23", Team.HOME, 1, 2, True),
                              FakeShot("s0041", "home-23", Team.HOME, 2, 2, True),
                              FakeShot("s0042", "home-23", Team.HOME, 3, 3, True)],
                       run={"team": Team.HOME, "points": 10, "opp_points": 2, "since_t": 250.0})
    stats.players["home-23"].fta, stats.players["home-23"].ftm = 4, 3
    return build_facts([shot_made(3)], stats, ROSTER, COURT, make_frame(stats), PM, **kw)


FACTS = _facts()


def ok(text: str, channel: str = "color", facts=None) -> bool:
    res = V.validate(text, facts or FACTS, channel)
    return res.ok


def problems(text: str, channel: str = "color", facts=None) -> list[str]:
    return V.validate(text, facts or FACTS, channel).problems


# --------------------------------------------------------------------------- happy path


@pytest.mark.parametrize("text", [
    f"{{{TOK}}} buries the three, and Riverside leads 42-40.",
    f"{{{TOK}}} has twelve points on 5-for-8 shooting.",
    f"{{{TOK}}} is 5 of 8 from the field with 12 points.",
    "Riverside is on a 10-2 run in the second quarter.",
    f"That is the third straight make for {{{TOK}}}.",
    "Lakeport trails by two with 4:32 left.",
    "Riverside by two, forty-two to forty.",
    f"No one could stop {{{TOK}}} on that one.",
    "Riverside gets a second chance, and that one's good.",
    "It's 40-42, Lakeport still behind.",
    f"{{{TOK}}} is 3-for-4 at the line.",
])
def test_grounded_lines_pass(text: str) -> None:
    assert ok(text), problems(text)


# --------------------------------------------------------------------------- injected errors


@pytest.mark.parametrize("text,needle", [
    (f"{{{TOK}}} has 31 points.", "31"),
    (f"{{{TOK}}} has thirty-one points.", "thirty-one"),
    (f"{{{TOK}}} is 6-for-9 tonight.", "6-for-9"),
    (f"{{{TOK}}} is 7 of 8 from the field.", "7 of 8"),
    (f"{{{TOK}}} shoots 62% from deep.", "62%"),
    ("We are in the fifth quarter.", "fifth"),
    ("Only 4:33 to go.", "4:33"),
    (f"{{{TOK}}} has a hundred points.", "a hundred"),
    (f"{{{TOK}}} with his 11th assist.", "11th"),
])
def test_wrong_numbers_fail(text: str, needle: str) -> None:
    probs = problems(text)
    assert any(needle in p and p.startswith("number not in facts") for p in probs), probs


def test_unknown_malformed_and_placeholder_tokens_fail() -> None:
    assert any("unknown token {P99}" in p for p in problems("{P99} scores."))
    assert any("malformed token" in p for p in problems(f"{TOK} scores."))
    assert any("unresolved placeholder" in p for p in problems("{actor} scores."))


@pytest.mark.parametrize("text,claim", [
    (f"{{{TOK}}} is having a career night.", "career"),
    (f"{{{TOK}}} matches his season best.", "season"),
    ("Riverside sets a record here.", "record"),
    (f"{{{TOK}}} is the all-time leader.", "all-time"),
    (f"{{{TOK}}} looks hurt after that ankle roll.", "injury"),
    (f"For the first time {{{TOK}}} leads them.", "first time"),
    (f"{{{TOK}}} with tonight's high.", "tonight's high"),
    ("Riverside has never lost here.", "never/ever"),
    (f"{{{TOK}}} is averaging twelve.", "average"),
    ("A pick-and-roll gets Riverside going.", "play type"),
])
def test_claims_lexicon_rejects_unsupported_claims(text: str, claim: str) -> None:
    assert any(f"unsupported claim ({claim})" in p for p in problems(text)), problems(text)


def test_team_names_do_not_trigger_the_claims_lexicon() -> None:
    roster = make_roster("Central Record Academy", "Lakeport")
    stats = make_stats()
    facts = build_facts([shot_made(3)], stats, roster, COURT, make_frame(stats), PseudonymMap(roster, seed=7))
    assert V.validate("Central Record Academy takes the lead.", facts, "color").ok
    assert not V.validate("Lakeport sets a record.", facts, "color").ok


def test_streak_needs_a_derived_streak() -> None:
    text = f"Third straight make for {{{TOK}}}."
    assert ok(text)
    no_streak = dict(FACTS, derived=[d for d in FACTS["derived"] if "straight" not in d])
    assert any("unsupported claim (streak)" in p for p in problems(text, facts=no_streak))
    assert any("streak" in p for p in problems(f"{{{TOK}}} has hit three in a row.", facts=no_streak))


def test_claims_allowed_by_storylines_and_labels() -> None:
    facts = _facts(storylines=["Riley Thorsen is near his season high"], labels={"last_play_type": "pick-and-roll"})
    assert ok(f"{{{TOK}}} is near his season high.", facts=facts)
    assert ok("That was a pick-and-roll from Riverside.", facts=facts)


@pytest.mark.parametrize("text,kind", [
    ("Riverside leads 45-40.", "score inconsistent"),
    ("It is 42-41.", "score inconsistent"),
    ("Riverside leads by five.", "margin inconsistent"),
    ("Riverside by five.", "lead inconsistent"),
    ("Lakeport leads.", "leader inconsistent"),
    ("Riverside trails here.", "trailer inconsistent"),
    ("We are tied at the moment.", "tie claimed"),
])
def test_wrong_scores_fail(text: str, kind: str) -> None:
    assert any(p.startswith(kind) for p in problems(text)), problems(text)


def test_leader_checked_with_up_by_phrasing() -> None:
    assert ok("Riverside up by two.") and ok("Riverside is up two.") and ok("Riverside leads 42-40.")
    assert any(p.startswith(("lead inconsistent", "leader inconsistent")) for p in problems("Lakeport up by two."))
    assert any(p.startswith("leader inconsistent") for p in problems("Lakeport is ahead now."))


def test_unconfirmed_score_is_not_speakable_in_hedge_only_packets() -> None:
    hedged = dict(FACTS, hedge_only=True, score=dict(FACTS["score"], confirmed=False, speakable=False))
    assert any("score not speakable" in p for p in problems("It is 42-40 here.", facts=hedged))
    assert any("score not speakable" in p for p in problems("Riverside 42, Lakeport 40.", facts=hedged))
    assert ok("Shot from the left wing by Riverside.", facts=hedged)


def test_name_right_after_a_token_is_caught() -> None:
    assert any("unknown name 'Smith'" in p for p in problems(f"Great pass. {{{TOK}}} Smith scores."))


def test_future_tense_predictions_fail() -> None:
    for text in ("Riverside will win this one.", "They're going to pull away.", "Lakeport is sure to lose."):
        assert any(p.startswith("prediction") for p in problems(text)), text
    assert ok("They're going to take a look at that one.")


def test_word_caps_per_channel() -> None:
    long = "Riverside keeps moving the ball around the perimeter and looks for a good shot now."
    assert any(p.startswith("too long") for p in problems(long, "play_by_play"))
    assert ok(long, "color")
    assert any(p.startswith("too long") for p in problems("Shot from the left wing by the shooter.", "accessibility"))
    assert ok(f"{{{TOK}}} shot, left wing, good. Riverside 42, Lakeport 40.", "accessibility")
    tight = GroundingValidator(ValidatorConfig(max_words={"color": 3}))
    assert not tight.validate("Riverside moves it around.", FACTS, "color").ok


def test_repetition_against_recent_lines() -> None:
    facts = _facts(recent_lines=["Riley Thorsen buries the three!"])
    assert any(p.startswith("repeats recent line") for p in problems(f"{{{TOK}}} buries the three!", facts=facts))
    assert ok(f"{{{TOK}}} knocks it down from the wing.", facts=facts)
    # AD lines are a fixed lexicon: repetition is not a problem there
    ad = _facts(recent_lines=["Lakeport ball."])
    assert V.validate("Lakeport ball.", ad, "accessibility").ok


def test_form_checks() -> None:
    assert "no terminal punctuation" in problems("Riverside moves it")
    assert ok("Riverside moves it...")
    assert any("Correction" in p for p in problems("That bucket goes elsewhere.", "correction"))
    assert ok("Correction: that basket does not count.", "correction")
    assert problems("") == ["empty line"]


def test_unknown_names_fail_but_team_names_pass() -> None:
    assert any("unknown name 'LeBron'" in p for p in problems("What a pass to LeBron there."))
    assert ok("What a pass from Riverside there.")
    assert ok("Correction: Riverside ball.", "correction")


def test_hedge_only_packets_reject_outcome_words() -> None:
    hedged = dict(FACTS, hedge_only=True)
    for text in (f"{{{TOK}}} buries it.", "That is good for two.", f"{{{TOK}}} hits the three.",
                 "Blocked at the rim.", "Turnover on Riverside."):
        assert any(p.startswith("outcome word while hedge_only") for p in problems(text, facts=hedged)), text
    assert ok("Shot from the left wing by Riverside.", facts=hedged)


def test_all_problems_are_returned() -> None:
    probs = problems("{P99} will win with 31 points in his career, leading 45-40")
    kinds = {p.split(":")[0].split(" (")[0] for p in probs}
    assert {"unknown token {P99}", "number not in facts", "unsupported claim", "score inconsistent",
            "prediction", "no terminal punctuation"} <= kinds, probs


# --------------------------------------------------------------------------- utilities


def test_number_words_round_trip() -> None:
    for n in range(0, 200):
        assert [m.key for m in extract_numbers(int_to_words(n))] == [f"card:{n}"], n
    for n in range(1, 100):
        assert [m.key for m in extract_numbers(ordinal_words(n))] == [f"ord:{n}"], n


def test_extraction_forms() -> None:
    keys = [m.key for m in extract_numbers("4:32 left, 3-for-5, 3 of 5, 60%, sixty percent, 24.3, 4th, 42-40")]
    assert keys[:6] == ["clock:4:32", "pct:60", "pair:3/5", "pair:3/5", "dash:42-40", "ord:4"]
    assert "card:24.3" in keys and "card:60" in keys
    assert not extract_numbers("{P07} and P12 with a one-handed one-on-one and-one")
    assert number_keys(["forty-two", "4th"]) == frozenset({"card:42", "ord:4"})


def test_jaccard() -> None:
    assert jaccard("Rebound, Riverside.", "rebound riverside") == 1.0
    assert jaccard("a b c d", "a b c e") == pytest.approx(3 / 5)


def test_validation_is_fast() -> None:
    rng = random.Random(0)
    pieces = [f"{{{TOK}}} buries the three", "Riverside leads 42-40", "with 4:32 left in the second quarter",
              "on 5-for-8 shooting", "for a 10-2 run", "and twelve points", "third straight make"]
    lines = [" ".join(rng.sample(pieces, 3)) + "." for _ in range(300)]
    t0 = time.perf_counter()
    for line in lines:
        V.validate(line, FACTS, "color")
    per_ms = (time.perf_counter() - t0) * 1000 / len(lines)
    assert per_ms < 5.0, per_ms
