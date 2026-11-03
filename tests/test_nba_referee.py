"""nba/referee.py (the learned referee check) and nba.analyze.is_referee's two methods."""

from __future__ import annotations

import types

import cv2
import numpy as np
import pytest

from courtvision.nba import referee_model
from courtvision.nba.analyze import is_referee, is_referee_shirt
from courtvision.nba.referee import FEATURE_NAMES, fit_logistic, referee_features, referee_prob


def _person(h: int, w: int, shirt: tuple[int, int, int], striped: bool, trousers: tuple[int, int, int]) -> np.ndarray:
    """A person box drawn h x w: a shirt (rows 15-52 %), trousers (52-90 %), stripes every eighth of the width."""
    img = np.full((h, w, 3), (40, 150, 70), np.uint8)
    img[int(0.15 * h):int(0.52 * h)] = shirt
    if striped:
        for k in range(0, 8, 2):
            img[int(0.15 * h):int(0.52 * h), int(k * w / 8):int((k + 1) * w / 8)] = (25, 25, 25)
    img[int(0.52 * h):int(0.9 * h)] = trousers
    return img


def test_features_need_a_readable_box_and_do_not_depend_on_resolution() -> None:
    small = _person(96, 40, (128, 128, 128), True, (20, 20, 20))
    big = cv2.resize(small, (120, 288), interpolation=cv2.INTER_NEAREST)  # the same person at three times the size
    f_small = referee_features(small, (0, 0, 40, 96))
    f_big = referee_features(big, (0, 0, 120, 288))
    assert f_small is not None and len(f_small) == len(FEATURE_NAMES)
    np.testing.assert_allclose(f_small, f_big, atol=0.03)
    assert referee_features(small, (0, 0, 3, 96)) is None  # narrower than 4 px
    assert referee_features(small, (-50, -50, -10, -10)) is None  # outside the frame


def test_fit_separates_referees_from_players_and_prob_reads_a_model() -> None:
    rng = np.random.default_rng(0)
    rows, labels = [], []
    for i in range(80):
        ref = i % 4 == 0  # a quarter are referees: grey and striped, dark trousers
        g = int(rng.integers(110, 150))
        colour = (g, g, g) if ref else tuple(int(c) for c in rng.integers(0, 256, 3))
        img = _person(96, 40, colour, ref, (20, 20, 20) if ref else colour)
        rows.append(referee_features(img, (0, 0, 40, 96)))
        labels.append(ref)
    mu, sd, coef, bias = fit_logistic(np.array(rows), np.array(labels))
    model = types.SimpleNamespace(MEAN=mu, STD=sd, COEF=coef, BIAS=bias)
    ref = _person(96, 40, (130, 130, 130), True, (20, 20, 20))
    player = _person(96, 40, (30, 40, 200), False, (30, 40, 200))
    assert referee_prob(ref, (0, 0, 40, 96), model) > 0.9
    assert referee_prob(player, (0, 0, 40, 96), model) < 0.1
    assert referee_prob(ref, (0, 0, 2, 2), model) is None


def test_is_referee_methods() -> None:
    assert len(referee_model.FEATURES) == len(FEATURE_NAMES) == len(referee_model.COEF) == len(referee_model.MEAN)
    assert tuple(referee_model.FEATURES) == FEATURE_NAMES
    img = _person(96, 40, (128, 128, 128), True, (20, 20, 20))
    box = (0, 0, 40, 96)
    assert is_referee(img, box, method="shirt") == is_referee_shirt(img, box)
    p = referee_prob(img, box)
    assert p is not None and is_referee(img, box) == (p >= referee_model.THRESHOLD)
    assert is_referee(img, (0, 0, 2, 2)) is False  # too small to read
    with pytest.raises(ValueError):
        is_referee(img, box, method="jersey")
