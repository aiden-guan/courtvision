"""nba/referee.py (the learned referee check and the in-game check) and nba.analyze's referee methods."""

from __future__ import annotations

import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from courtvision.nba import referee_model
from courtvision.nba.analyze import is_referee, is_referee_shirt
from courtvision.nba.referee import (
    FEATURE_NAMES,
    GAME_FEATURE_NAMES,
    HIST_BINS,
    GameConfig,
    colour_histograms,
    fit_game,
    fit_logistic,
    game_features_batch,
    game_referee_calls,
    referee_features,
    referee_features_batch,
    referee_prob,
    track_votes,
)


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
    assert p is not None and is_referee(img, box, method="learned") == (p >= referee_model.THRESHOLD)
    assert is_referee(img, box) == is_referee_shirt(img, box)  # the shirt rule is the default (research note §10.11)
    assert is_referee(img, (0, 0, 2, 2), method="learned") is False  # too small to read
    with pytest.raises(ValueError):
        is_referee(img, box, method="jersey")


def test_referee_check_guard_falls_back_to_the_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    from courtvision.nba import analyze

    monkeypatch.setattr(analyze, "are_referees",
                        lambda img, boxes, method="shirt", model=None: [method == "learned"] * len(boxes))
    img = np.zeros((10, 10, 3), np.uint8)
    four, one = [(0, 0, 1, 1)] * 4, [(0, 0, 1, 1)]
    guard = analyze.RefereeCheck("guarded", max_mean=3.5, min_frames=3, min_people=2)
    assert guard(img, four, 1) == [True] * 4 and guard(img, one, 2) == [True]  # too few people: not counted
    assert guard(img, four, 3) == [True] * 4  # 2 counted frames: not yet judged
    assert guard(img, four, 4) == [False] * 4 and guard.fell_back_at == 4  # 4 calls a frame > 3.5: the rule
    assert guard(img, one, 5) == [False]  # for the rest of the game
    calm = analyze.RefereeCheck("guarded", max_mean=3.5, min_frames=3, min_people=2)
    for k in range(5):
        assert calm(img, [(0, 0, 1, 1)] * 3, k) == [True] * 3  # 3 a frame: the learned check stays
    assert calm.fell_back_at is None
    assert analyze.RefereeCheck("learned")(img, four) == [True] * 4
    assert analyze.RefereeCheck()(img, four) == [False] * 4  # the default is the rule
    with pytest.raises(ValueError):
        analyze.RefereeCheck("jersey")


def test_load_referee_model_reads_another_fit(tmp_path) -> None:  # noqa: ANN001
    from courtvision.nba import analyze

    text = Path(referee_model.__file__).read_text().replace("THRESHOLD = ", "THRESHOLD = 2 * ")
    path = tmp_path / "refit.py"
    path.write_text(text)
    model = analyze.load_referee_model(path)
    assert model.THRESHOLD == 2 * referee_model.THRESHOLD and model.COEF == referee_model.COEF
    img = _person(96, 40, (128, 128, 128), True, (20, 20, 20))
    box = (0.0, 0.0, 40.0, 96.0)
    p = referee_prob(img, box)
    assert analyze.are_referees(img, [box], "learned", model) == [bool(p >= 2 * referee_model.THRESHOLD)]
    assert analyze.RefereeCheck("learned", model=model)(img, [box]) == [bool(p >= 2 * referee_model.THRESHOLD)]


# --------------------------------------------------------------------------- the in-game check

RED, WHITE, BLACK, BLUE = (40, 40, 200), (235, 235, 235), (28, 28, 28), (190, 90, 30)  # BGR uniforms


def _game_frame(rng: np.random.Generator, teams: tuple = (RED, WHITE), n_team: int = 5, n_ref: int = 3,
                noise: bool = True) -> tuple[np.ndarray, list[tuple[float, float, float, float]], list[bool]]:
    """One frame of a synthetic game on a wooden floor: ``n_team`` players of each team (shirt and shorts in the team's
    colour, a white number on the shirt, skin below the knee) and ``n_ref`` referees (grey shirt with black stripes,
    black trousers), side by side; each person's colours jittered, and (``noise``) the camera's pixel noise and
    softness. (Frame, boxes, which are referees.)"""
    people = [(c, False) for c in teams for _ in range(n_team)] + [(None, True)] * n_ref
    rng.shuffle(people)
    h, w = 300, 80 * len(people)
    img = np.full((h, w, 3), (95, 145, 195), np.uint8)

    def jit(c: tuple[int, int, int]) -> tuple[int, ...]:
        return tuple(int(np.clip(v + rng.integers(-12, 13), 0, 255)) for v in c)

    boxes, refs = [], []
    for k, (colour, ref) in enumerate(people):
        bh = int(rng.integers(150, 190))
        x, y = 80 * k + 20, int(rng.integers(5, h - bh - 5))
        rows = lambda a, b: slice(y + int(a * bh), y + int(b * bh))  # noqa: E731
        img[rows(0.0, 0.15), x + 12:x + 28] = jit((80, 110, 160))  # head
        if ref:
            img[rows(0.15, 0.52), x:x + 40] = jit((128, 128, 128))
            for c in range(x, x + 40, 10):
                img[rows(0.15, 0.52), c:c + 4] = jit(BLACK)
            img[rows(0.52, 1.0), x + 4:x + 36] = jit(BLACK)
        else:
            img[rows(0.15, 0.52), x:x + 40] = jit(colour)
            img[rows(0.25, 0.38), x + 14:x + 26] = jit(WHITE)  # the number
            img[rows(0.52, 0.72), x + 2:x + 38] = jit(colour)  # shorts
            img[rows(0.72, 1.0), x + 6:x + 34] = jit((90, 125, 175))  # legs
        boxes.append((float(x), float(y), float(x + 40), float(y + bh)))
        refs.append(ref)
    if noise:
        img = cv2.GaussianBlur(np.clip(img + rng.normal(0, 10, img.shape), 0, 255).astype(np.uint8), (3, 3), 0)
    return img, boxes, refs


def _game(seed: int, frames: int, **kw) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """``frames`` synthetic frames: each box's GAME_FEATURE_NAMES row, frame, truth, and the (frame, boxes) pairs."""
    rng = np.random.default_rng(seed)
    X, F, T, shown = [], [], [], []
    for i in range(frames):
        img, boxes, refs = _game_frame(rng, **kw)
        f, keep = game_features_batch(img, boxes)
        assert len(keep) == len(boxes)
        X.append(f)
        F += [i] * len(boxes)
        T += refs
        shown.append((img, boxes, refs))
    return np.concatenate(X), np.array(F), np.array(T), shown


def test_colour_histograms_bin_black_greys_and_hues() -> None:
    patches = np.array([[[[0, 0, 20]]], [[[0, 0, 100]]], [[[0, 0, 250]]], [[[0, 255, 200]]], [[[60, 100, 200]]]],
                       np.uint8)  # HSV: black, mid grey, white, saturated red, a paler green
    h = colour_histograms(patches)
    assert h.shape == (5, HIST_BINS) and np.allclose(h.sum(1), 1.0)
    assert [int(np.argmax(r)) for r in h] == [0, 1 + 100 // 64, 4, 5 + 1, 5 + (60 * 12 // 180) * 2]


def test_game_features_extend_the_learned_checks_features() -> None:
    img, boxes, _ = _game_frame(np.random.default_rng(1))
    f, keep = game_features_batch(img, boxes + [(0.0, 0.0, 2.0, 2.0)])  # the last box is too small to read
    ref, ref_keep = referee_features_batch(img, boxes)
    assert f.shape == (len(boxes), len(GAME_FEATURE_NAMES)) and list(keep) == list(ref_keep) == list(range(len(boxes)))
    np.testing.assert_array_equal(f[:, :len(FEATURE_NAMES)], ref)
    n = len(FEATURE_NAMES)
    np.testing.assert_allclose(f[:, n:n + HIST_BINS].sum(1), 1.0)
    np.testing.assert_allclose(f[:, n + HIST_BINS:].sum(1), 1.0)


@pytest.mark.parametrize("teams", [(RED, WHITE), (BLACK, WHITE), (BLACK, BLUE)])
def test_fit_game_takes_the_small_third_group_for_the_referees(teams: tuple) -> None:
    X, frames, truth, shown = _game(0, 30, teams=teams)
    X[5] = np.nan  # a box too small to read: never a referee
    fit = fit_game(X, frames)
    assert fit.accepted, fit.reason
    assert sorted(fit.sizes) == ([89, 150, 150] if truth[5] else [90, 149, 150])  # 13 people a frame: 3 referees
    assert fit.per_frame == pytest.approx((truth.sum() - truth[5]) / 30)
    assert fit.silhouette >= GameConfig().min_silhouette
    expect = truth.copy()
    expect[5] = False
    np.testing.assert_array_equal(fit.calls, expect)
    img, boxes, refs = _game_frame(np.random.default_rng(99), teams=teams)  # a frame the fit has not seen
    np.testing.assert_array_equal(fit.predict(game_features_batch(img, boxes)[0]), refs)
    calls, _ = game_referee_calls(X, frames, np.zeros(len(X), bool))
    np.testing.assert_array_equal(calls, expect)


def test_fit_game_falls_back_when_the_third_group_does_not_fit_the_referees() -> None:
    X, frames, truth, _ = _game(0, 30, n_ref=5)  # 5 "referees" a frame: as many as a team
    fit = fit_game(X, frames)
    assert not fit.accepted and "referees a frame" in fit.reason
    fallback = np.arange(len(X)) % 7 == 0
    calls, _ = game_referee_calls(X, frames, fallback)
    np.testing.assert_array_equal(calls, fallback)  # the caller's rule, unchanged
    assert not fit.predict(X).any()
    few = fit_game(X[:26], frames[:26])  # two frames: too few boxes to cluster
    assert not few.accepted and "readable boxes" in few.reason
    X, frames, _, _ = _game(0, 30, n_team=3, n_ref=2)  # 8 people a frame: with min_people 9 no frame counts,
    unjudged = fit_game(X, frames, GameConfig(min_people=9))  # so the size cannot be judged
    assert not unjudged.accepted and "frames with 9" in unjudged.reason
    rng = np.random.default_rng(0)  # no structure: the size can pass, the silhouette does not
    noise = rng.random((330, len(GAME_FEATURE_NAMES)))
    assert not fit_game(noise, np.repeat(np.arange(30), 11)).accepted
    with pytest.raises(ValueError):
        fit_game(noise, np.arange(10))
    X, frames, _, _ = _game(0, 30, noise=False)  # flat colours: near-copies of each uniform, apart from each other
    flat = fit_game(X, frames)
    assert not flat.accepted and flat.parts > 3 and "parts" in flat.reason


def test_fit_game_clusters_a_sample_and_places_the_other_boxes() -> None:
    X, frames, truth, _ = _game(2, 60)
    fit = fit_game(X, frames, GameConfig(max_fit=300))
    assert fit.accepted and sum(fit.sizes) == 300
    np.testing.assert_array_equal(fit.calls, truth)


def test_track_votes_take_each_tracks_majority() -> None:
    calls = [True, True, False, False, True, False, True]
    tracks = [1, 1, 1, 2, 2, None, -1]
    assert track_votes(calls, tracks).tolist() == [True, True, True, False, False, False, True]  # 2-2: not a referee
    X, frames, truth, _ = _game(0, 30)
    flipped = X.copy()
    first_ref = int(np.flatnonzero(truth)[0])
    flipped[first_ref] = X[np.flatnonzero(~truth)[0]]  # one box of a referee reads as a player's
    ids = np.arange(len(X))
    ids[first_ref] = int(np.flatnonzero(truth)[1])  # ... and shares a track with another referee box
    ids[np.flatnonzero(truth)[2]] = ids[first_ref]
    calls, fit = game_referee_calls(flipped, frames, np.zeros(len(X), bool), track_ids=list(ids))
    assert fit.accepted and calls[first_ref]  # the track's two other boxes outvote it
    with pytest.raises(ValueError):
        track_votes([True], [1, 2])


def test_referee_check_game_method_refits_and_votes_per_track(monkeypatch: pytest.MonkeyPatch) -> None:
    from courtvision.nba import analyze

    monkeypatch.setattr(analyze, "are_referees", lambda img, boxes, method="shirt", model=None: [False] * len(boxes))
    check = analyze.RefereeCheck("game", game=GameConfig(observe_every=2, refit_every=15, window=20))
    _, _, _, shown = _game(3, 60)
    for i, (img, boxes, refs) in enumerate(shown):
        calls = check(img, boxes, i)
        if i < 28:  # 15 observed frames (every second one) before the first fit: the shirt rule (patched: no one)
            assert calls == [False] * len(boxes)
        else:
            assert calls == refs
        if i == 28:
            assert check.fits == 1 and sum(check.fit.sizes) == 15 * 13
    assert check.fits == 2 and check.rejected == 0 and check.fit is not None and check.fit.accepted
    assert sum(check.fit.sizes) == 20 * 13  # the second fit, at frame 58: the newest 20 observed frames
    img, boxes, refs = shown[-1]
    ids = list(range(len(boxes)))
    assert check(img, boxes, 60, track_ids=ids) == refs
    r = refs.index(True)
    assert check(img, [], 61) == []
    swapped = boxes.copy()
    swapped[r] = boxes[refs.index(False)]  # the referee's track now shows a player's box ...
    assert check(img, swapped, 62, track_ids=ids)[r] is False  # 1 of 2 calls: a tie is not a referee
    assert check(img, boxes, 63, track_ids=ids)[r] is True  # 2 of 3
    assert check(img, swapped, 64, track_ids=[None] * len(boxes))[r] is False  # no track: the box's own call
    with pytest.raises(ValueError):
        analyze.RefereeCheck("jersey")
