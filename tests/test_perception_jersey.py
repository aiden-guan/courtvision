"""Jersey reading on synthetic renders and confidence-weighted voting."""

from __future__ import annotations

import numpy as np
import pytest
from test_perception_synth import SynthPlayer, draw_court, draw_figure, make_camera, make_court, player_bbox, render

from courtvision.perception.jersey import (
    ALL_JERSEY_NUMBERS,
    JerseyReader,
    JerseyVoter,
    NullJerseyReader,
    TemplateJerseyReader,
)
from courtvision.sim import palette
from courtvision.types import BBox, Team


def figure(number: str, h: float, w: float | None = None) -> tuple[np.ndarray, BBox]:
    img = np.zeros((200, 200, 3), np.uint8)
    img[:] = palette.FLOOR_BGR
    w = w if w is not None else 0.33 * h
    box = BBox(100 - w / 2, 180 - h, 100 + w / 2, 180)
    draw_figure(img, box, palette.HOME_BGR, number)
    return img, box


@pytest.mark.parametrize("h", [45.0, 70.0, 110.0])
def test_reads_every_roster_number_including_lookalikes(h: float) -> None:
    roster = ["0", "00", "1", "11", "7", "17", "71", "3", "8", "23", "32", "88", "6", "9", "99"]
    reader = TemplateJerseyReader(candidates=roster)
    assert isinstance(reader, JerseyReader)
    for n in roster:
        img, box = figure(n, h)
        got, conf = reader.read(img, box)
        assert got == n, (n, got, h)
        assert conf > 0.3


def test_reads_with_detection_box_error_and_full_candidate_set() -> None:
    reader = TemplateJerseyReader()  # all 101 numbers
    rng = np.random.default_rng(4)
    numbers = rng.choice(np.array(ALL_JERSEY_NUMBERS), size=12, replace=False)
    for n in numbers:
        img, box = figure(str(n), 80.0)
        jitter = BBox(box.x1 + rng.uniform(-1.5, 1.5), box.y1 + rng.uniform(-1.5, 1.5),
                      box.x2 + rng.uniform(-1.5, 1.5), box.y2 + rng.uniform(-1.5, 1.5))
        got, _ = reader.read(img, jitter)
        assert got == str(n)


def test_reads_on_the_court_render() -> None:
    cam, court = make_camera(), make_court()
    people = [SynthPlayer(15, 8, Team.HOME, "12"), SynthPlayer(70, 30, Team.AWAY, "5"), SynthPlayer(47, 40, Team.HOME, "34")]
    img, boxes, _ = render(cam, court, people, base=draw_court(cam, court))
    reader = TemplateJerseyReader(candidates=["12", "5", "34", "2", "1", "3", "4", "15"])
    for p, b in zip(people, boxes):
        assert reader.read(img, b)[0] == p.number
    assert player_bbox(cam, 47, 40).h > 40


def test_legibility_gate_and_no_number() -> None:
    reader = TemplateJerseyReader(candidates=["1", "2"])
    img, box = figure("2", 15.0)
    assert reader.read(img, box) == (None, 0.0)
    blank, bbox = figure("", 80.0)
    assert reader.read(blank, bbox) == (None, 0.0)
    assert NullJerseyReader().read(blank, bbox) == (None, 0.0)


def test_voter_confirms_after_three_agreeing_reads() -> None:
    v = JerseyVoter(min_votes=3, min_share=0.6)
    v.add(1, "23", 0.9)
    v.add(1, "23", 0.8)
    assert v.result(1) == (None, 1.0)
    v.add(1, "28", 0.5)
    v.add(1, None, 0.9)  # no read
    v.add(1, "23", 0.01)  # too weak to count
    assert v.result(1)[0] is None
    v.add(1, "23", 0.7)
    number, share = v.result(1)
    assert number == "23" and share == pytest.approx(2.4 / 2.9)
    assert v.best(1)[2] == 3


def test_voter_share_threshold_and_roster_constraint() -> None:
    v = JerseyVoter()
    for n in ["5", "8", "5", "8", "5", "8"]:
        v.add(7, n, 0.9)
    assert v.result(7)[0] is None  # 50 % share, no streak
    assert v.result(7, allowed={"5", "12"})[0] == "5"  # 8 is not on this team's roster
    assert v.result(7, allowed={"12"}) == (None, 0.0)


def test_voter_streak_recovers_after_swap_and_window_forgets() -> None:
    v = JerseyVoter(window=12, streak=True)
    for _ in range(10):
        v.add(1, "23", 0.9)
    assert v.result(1)[0] == "23"
    for _ in range(2):  # the track now follows #5 (tracker swap)
        v.add(1, "5", 0.9)
    assert v.result(1)[0] == "23"
    v.add(1, "5", 0.9)  # third identical read in a row
    number, share = v.result(1)
    assert number == "5" and share == pytest.approx(3 / 12)
    assert v.result(1, allowed={"23"})[0] == "23"  # the streak must be an allowed number
    for _ in range(9):
        v.add(1, "5", 0.9)
    assert v.weight(1, "23") == 0.0  # window of 12 reads
    no_streak = JerseyVoter(streak=False)
    for n in ["23"] * 5 + ["5"] * 3:
        no_streak.add(2, n, 0.9)
    assert no_streak.result(2)[0] == "23"


def test_voter_default_needs_share_not_just_a_streak() -> None:
    """Regression: the default voter follows the spec (>= 3 reads AND share >= 0.6); a run of three
    reads alone used to confirm a number holding a 23 % or 10 % share."""
    v = JerseyVoter()  # spec signature JerseyVoter(min_votes=3, min_share=0.6)
    for _ in range(5):
        v.add(1, "23", 0.9)
    for _ in range(3):
        v.add(1, "2", 0.9)  # partial reads of 23 (the 3 is covered)
    assert v.result(1)[0] == "23"
    assert v.result(1, allowed={"2", "23"})[0] == "23"
    for _ in range(9):
        v.add(2, "23", 0.9)
    for _ in range(3):
        v.add(2, "28", 0.3)  # three weak misreads in a row
    assert v.result(2)[0] == "23"


def test_opt_in_streak_ignores_weak_and_partial_runs() -> None:
    """Regression: with streak=True, weak reads or a one-digit partial of a voted two-digit number
    no longer flip a confirmed number; three confident full reads still do."""
    v = JerseyVoter(window=12, streak=True)
    for _ in range(5):
        v.add(1, "23", 0.9)
    for _ in range(3):
        v.add(1, "2", 0.9)
    assert v.result(1)[0] == "23"
    for _ in range(3):
        v.add(1, "28", 0.3)
    assert v.result(1)[0] == "23"
    for _ in range(3):
        v.add(1, "28", 0.9)
    assert v.result(1)[0] == "28"


def test_voter_downweights_single_digit_inside_two_digit() -> None:
    v = JerseyVoter(single_digit_weight=0.5)
    for _ in range(4):
        v.add(3, "2", 0.9)  # partial reads of "23"
    for _ in range(3):
        v.add(3, "23", 0.9)
    assert v.result(3)[0] == "23"
    w = JerseyVoter()
    for _ in range(3):
        w.add(4, "2", 0.9)
    assert w.result(4)[0] == "2"  # no competing two-digit read: no penalty


def test_voter_merge_and_forget() -> None:
    v = JerseyVoter()
    v.add(1, "9", 0.9)
    v.add(1, "9", 0.9)
    v.add(2, "9", 0.9)
    v.merge(1, 2)
    assert v.result(2)[0] == "9" and v.result(1) == (None, 0.0)
    assert v.weight(2, "9") == pytest.approx(2.7)
    v.forget(2)
    assert v.weight(2, "9") == 0.0
