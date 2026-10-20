"""Question hygiene for every Jev question CourtVision ships (design §5 constraints, §8.5, §11.1)."""

from __future__ import annotations

import re

import pytest
from typesafe_sdk import Box, Choice, Noul, Score

from courtvision.decision import questions as Q
from courtvision.decision.client import wire_questions

BACKTICKED = re.compile(r"`[^`]*`")
SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")

ALL = Q.all_shipped_bundles()


def _texts_outside_backticks(qs: dict) -> list[str]:
    return [BACKTICKED.sub("", t) for t in Q.question_texts(qs)]


@pytest.mark.parametrize("bundle", sorted(ALL))
def test_no_digit_in_any_instruction_or_criterion(bundle: str) -> None:
    for text in _texts_outside_backticks(ALL[bundle]):
        assert not re.search(r"\d", text), f"{bundle}: digit in {text!r}"


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_dynamic_b2_has_no_digits_at_every_size(n: int) -> None:
    for kinds in ({"stat_note"}, {"tendency", "storyline"}, set()):
        for text in _texts_outside_backticks(Q.b2_dead_ball_fan(kinds, n)):
            assert not re.search(r"\d", text)


@pytest.mark.parametrize("n", range(1, len(Q.CARD_KEYS) + 1))
def test_dynamic_b3_has_no_digits_names_or_card_text(n: int) -> None:
    cards = {k: f"{{P0{i}}} Brooks has three fouls" for i, k in enumerate(Q.CARD_KEYS[:n])}
    qs = Q.b3_dead_ball_coach(cards)
    texts = list(Q.question_texts(qs))
    for text in texts:
        assert not re.search(r"\d", BACKTICKED.sub("", text))
        assert "Brooks" not in text and "fouls" not in text  # card text stays in the state only
    assert set(qs["top_card"].criteria) == set(cards) | {"none"}


@pytest.mark.parametrize("bundle", sorted(ALL))
def test_every_question_has_instructions_and_valid_sizes(bundle: str) -> None:
    for name, q in ALL[bundle].items():
        assert isinstance(q, (Choice, Score, Noul, Box)), name
        assert q.instructions, name
        if isinstance(q, Choice):
            assert 2 <= len(q.criteria) <= 255, name
        if isinstance(q, Score):
            assert 2 <= len(q.criteria) <= 10, name


@pytest.mark.parametrize("bundle", sorted(ALL))
def test_wire_dict_equals_model_dump(bundle: str) -> None:
    qs = ALL[bundle]
    wire = wire_questions(qs)
    for name, q in qs.items():
        assert wire[name] == q.model_dump(mode="json", exclude_none=True)
        assert wire[name]["type"] in {"choice", "score", "noul", "box"}


def test_card_and_guard_names_are_path_safe_snake_case() -> None:
    for name in [*ALL["B3"], *ALL["B4"], *ALL["B2"]]:
        assert SNAKE.match(name), name
    for label in ALL["B3"]["top_card"].criteria:
        assert SNAKE.match(label), label


def test_reversed_questions_mirror_the_forward_order() -> None:
    for fwd, rev in (("play_type", "play_type_rev"),):
        assert list(Q.B1_POSSESSION_END[rev].criteria) == list(reversed(list(Q.B1_POSSESSION_END[fwd].criteria)))
    for spec in Q.B5_KINDS.values():
        _root, qs, fwd, rev, committable = spec
        assert list(qs[rev].criteria) == list(reversed(list(qs[fwd].criteria)))
        assert "unclear" in qs[fwd].criteria and set(committable) <= set(qs[fwd].criteria)


def test_every_choice_offers_an_abstain_label() -> None:
    abstain = {"unclear", "none", "silence"}
    for bundle, qs in ALL.items():
        for name, q in qs.items():
            if isinstance(q, Choice):
                assert abstain & set(q.criteria), f"{bundle}.{name}"


def test_b2_always_offers_silence_and_only_available_kinds() -> None:
    qs = Q.b2_dead_ball_fan({"tendency"}, 5)
    assert list(qs["next_line_kind"].criteria) == ["tendency", "silence"]
    assert [k for k in qs if k.startswith("storyline_due_")] == [f"storyline_due_{i}" for i in range(3)]


def test_builders_reject_out_of_range_sizes() -> None:
    with pytest.raises(ValueError):
        Q.b4_line_guard(3)
    with pytest.raises(ValueError):
        Q.b3_dead_ball_coach({"card_z": "x"})
    with pytest.raises(ValueError):
        Q.b3_dead_ball_coach({})
