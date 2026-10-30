"""TeamClassifier (numpy k-means++), TeamVoter and the lazily-imported SigLIP embedder."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from courtvision.perception.appearance import torso_feature
from courtvision.perception.teams import SiglipEmbedder, TeamClassifier, TeamConfig, TeamVoter, kmeans
from courtvision.sim import palette
from courtvision.types import BBox, Team


def patch_feature(bgr: tuple[int, int, int], rng: np.random.Generator, noise: float = 6.0) -> np.ndarray:
    img = np.clip(np.asarray(bgr, float) + rng.normal(0, noise, (40, 20, 3)), 0, 255).astype(np.uint8)
    return torso_feature(img, BBox(0, 0, 20, 40))


def test_kmeans_separates_blobs_deterministically() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (50, 3)), rng.normal(3, 0.1, (50, 3))])
    C1, l1, i1 = kmeans(X, 2, np.random.default_rng(5))
    C2, l2, i2 = kmeans(X, 2, np.random.default_rng(5))
    assert np.array_equal(l1, l2) and np.allclose(C1, C2) and i1 == i2
    assert len(set(l1[:50])) == 1 and len(set(l1[50:])) == 1 and l1[0] != l1[-1]
    with pytest.raises(ValueError):
        kmeans(X[:1], 2, np.random.default_rng(0))


def test_classifier_fits_and_anchors_bind_home_and_away() -> None:
    rng = np.random.default_rng(1)
    cfg = TeamConfig(home_color_bgr=palette.HOME_BGR, away_color_bgr=palette.AWAY_BGR, min_fit_samples=20)
    clf = TeamClassifier(cfg)
    for _ in range(15):  # away first, so k-means cluster order alone would not give HOME = 0
        clf.observe(patch_feature(palette.AWAY_BGR, rng))
    assert not clf.fit()  # too few samples
    for _ in range(15):
        clf.observe(patch_feature(palette.HOME_BGR, rng))
    clf.observe(None)
    clf.observe(np.zeros(58))
    assert clf.n_samples == 30
    assert clf.fit() and clf.is_fitted
    for bgr, team in ((palette.HOME_BGR, Team.HOME), (palette.AWAY_BGR, Team.AWAY)):
        for _ in range(10):
            got, conf = clf.predict(patch_feature(bgr, rng))
            assert got is team and conf > 0.5
    assert clf.predict(None) == (None, 0.0)


def test_predict_before_fit_uses_anchors_or_abstains() -> None:
    rng = np.random.default_rng(2)
    anchored = TeamClassifier(TeamConfig(home_color_bgr=palette.HOME_BGR, away_color_bgr=palette.AWAY_BGR))
    assert anchored.predict(patch_feature(palette.AWAY_BGR, rng))[0] is Team.AWAY
    bare = TeamClassifier(TeamConfig(min_fit_samples=10))
    assert bare.predict(patch_feature(palette.AWAY_BGR, rng)) == (None, 0.0)
    for k in range(20):
        bare.observe(patch_feature(palette.HOME_BGR if k % 2 else palette.AWAY_BGR, rng))
    assert bare.fit()
    a = bare.predict(patch_feature(palette.AWAY_BGR, rng))[0]
    h = bare.predict(patch_feature(palette.HOME_BGR, rng))[0]
    assert {a, h} == {Team.HOME, Team.AWAY}  # permutation is arbitrary without anchors


def test_max_distance_rejects_outliers_and_pca_reduction() -> None:
    rng = np.random.default_rng(3)
    cfg = TeamConfig(min_fit_samples=20, reduce_dims=3, max_distance=0.5)
    clf = TeamClassifier(cfg, anchors={Team.HOME: np.r_[np.ones(8), np.zeros(56)], Team.AWAY: np.r_[np.zeros(56), np.ones(8)]})
    for k in range(40):
        base = np.r_[np.ones(8), np.zeros(56)] if k % 2 else np.r_[np.zeros(56), np.ones(8)]
        clf.observe(base + rng.normal(0, 0.02, 64))
    assert clf.fit()
    assert clf.predict(np.r_[np.ones(8), np.zeros(56)])[0] is Team.HOME
    assert clf.predict(np.r_[np.zeros(56), np.ones(8)])[0] is Team.AWAY
    odd = np.zeros(64)
    odd[30] = 1.0  # neither team (a coach in a third colour)
    assert clf.predict(odd) == (None, 0.0)


def test_team_voter_hysteresis() -> None:
    v = TeamVoter(momentum=0.9)
    assert v.get(1) == (None, 0.0)
    for _ in range(10):
        v.add(1, Team.HOME, 0.9)
    v.add(1, Team.AWAY, 0.9)  # one flicker
    team, conf = v.get(1)
    assert team is Team.HOME and 0.5 < conf < 0.9
    v.add(2, None, 0.0)
    assert v.get(2) == (None, 0.0)
    v.forget(1)
    assert v.get(1) == (None, 0.0) and v.track_ids() == [2]


def test_teams_module_imports_no_heavy_libraries() -> None:
    code = ("import sys, courtvision.perception.teams, courtvision.perception.pipeline;"
            "print([m for m in ('torch', 'transformers', 'rfdetr', 'ultralytics', 'supervision', 'trackers') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_siglip_embedder_with_injected_fakes() -> None:
    torch = pytest.importorskip("torch")

    class FakeProcessor:
        def __call__(self, images, return_tensors):
            assert return_tensors == "pt"
            arr = np.stack([np.resize(im, (8, 8, 3)) for im in images]).astype(np.float32)
            return {"pixel_values": torch.from_numpy(arr).permute(0, 3, 1, 2)}

    class FakeModel:
        def __call__(self, pixel_values):
            b = pixel_values.shape[0]
            hidden = pixel_values.reshape(b, 3, -1).permute(0, 2, 1)  # (B, tokens, 3)

            class Out:
                last_hidden_state = hidden

            return Out()

    emb = SiglipEmbedder(model=FakeModel(), processor=FakeProcessor(), batch_size=2)
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[:, :50] = (255, 0, 0)  # blue (BGR) on the left
    frame[:, 50:] = (0, 0, 255)  # red on the right
    out = emb.embed(frame, [BBox(10, 10, 40, 90), BBox(60, 10, 90, 90), BBox(200, 200, 210, 210)])
    assert out.shape == (3, 3)
    assert out[0, 2] > 200 and out[1, 0] > 200  # RGB order: blue patch -> channel 2
    assert not out[2].any()  # crop outside the frame -> zero row


def test_a_refit_mends_a_first_fit_on_one_team_and_keeps_the_order() -> None:
    """The first fit sees one team only (two shades of it); refits as both teams come into view separate them, and
    the clusters keep their names from one fit to the next (no colour anchors)."""
    rng = np.random.default_rng(3)
    white, white_shade, black = (235, 235, 235), (200, 205, 215), (35, 35, 35)
    clf = TeamClassifier(TeamConfig(min_fit_samples=30, refit_at=(150, 600)))
    for k in range(30):
        clf.observe(patch_feature(white if k % 2 else white_shade, rng))
    assert clf.fit() and not clf.refit_due
    first = clf.predict(patch_feature(white, rng))[0]
    for k in range(120):
        clf.observe(patch_feature(white if k % 2 else black, rng))
    assert clf.refit_due and clf.fit()
    got = {name: {clf.predict(patch_feature(bgr, rng))[0] for _ in range(10)}
           for name, bgr in (("white", white), ("shade", white_shade), ("black", black))}
    assert got["white"] == got["shade"] == {first} and got["black"] == {Team.AWAY if first is Team.HOME else Team.HOME}
    for k in range(450):  # the next refit names them as before
        clf.observe(patch_feature(white if k % 2 else black, rng))
    assert clf.refit_due and clf.fit() and clf.predict(patch_feature(white, rng))[0] is first


def test_no_refit_by_default() -> None:
    rng = np.random.default_rng(4)
    clf = TeamClassifier(TeamConfig(min_fit_samples=10))
    for k in range(200):
        clf.observe(patch_feature((235, 235, 235) if k % 2 else (35, 35, 35), rng))
        if clf.n_samples >= 10 and not clf.is_fitted:
            clf.fit()
        assert not clf.refit_due
