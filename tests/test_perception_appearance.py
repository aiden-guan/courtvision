"""The torso colour feature (perception/appearance.py): its bin table gives the per-pixel maths' histogram exactly."""

from __future__ import annotations

import numpy as np
import pytest

from courtvision.perception import appearance
from courtvision.perception.appearance import AppearanceConfig, color_feature, torso_feature
from courtvision.sim.palette import FLOOR_BGR, LINE_BGR
from courtvision.types import BBox


@pytest.mark.parametrize("cfg", [AppearanceConfig(), AppearanceConfig(h_bins=12, s_bins=4, v_bins=3, achromatic_s=40,
                                                                      suppress_tol=12.5),
                                 AppearanceConfig(suppress_bgr=())])
def test_bin_table_histograms_equal_the_direct_ones(cfg: AppearanceConfig) -> None:
    rng = np.random.default_rng(0)
    near = np.array(FLOOR_BGR) + rng.integers(-40, 41, (3000, 3))  # around a suppressed colour: both sides of tol
    crops = [rng.integers(0, 256, (n, 3), dtype=np.uint8) for n in (0, 1, 3, 4, 5, 64, 1000, 20000)]
    crops += [np.clip(near, 0, 255).astype(np.uint8), np.tile(np.array(LINE_BGR, np.uint8), (50, 1))]
    for px in crops:
        got, want = appearance._histogram(px, cfg), appearance._histogram_direct(px, cfg)
        assert got.dtype == want.dtype and np.array_equal(got, want)


def test_torso_feature_and_anchor_features_unchanged() -> None:
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    frame[20:100, 40:80] = (200, 60, 30)  # a blue jersey
    box = BBox(30.0, 10.0, 90.0, 110.0)
    cfg = AppearanceConfig()
    x1, x2, y1, y2 = 42, 78, 35, 75  # the torso window of ``box``
    want = appearance._histogram_direct(frame[y1:y2, x1:x2].reshape(-1, 3), cfg)
    assert np.array_equal(torso_feature(frame, box), want)
    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[40:60, 45:70] = 1
    masked = frame[y1:y2, x1:x2].reshape(-1, 3)[mask[y1:y2, x1:x2].reshape(-1) > 0]
    assert np.array_equal(torso_feature(frame, box, mask=mask), appearance._histogram_direct(masked, cfg))
    assert color_feature((200, 60, 30)).sum() == pytest.approx(1.0)  # anchors: no suppression, direct path


def test_settings_the_table_cannot_hold_take_the_direct_path() -> None:
    assert appearance._bin_table(AppearanceConfig(suppress_bgr=((10.5, 20, 30),))) is None
    assert appearance._bin_table(AppearanceConfig(h_bins=90, s_bins=3)) is None  # 274 bins
    px = np.random.default_rng(2).integers(0, 256, (500, 3), dtype=np.uint8)
    cfg = AppearanceConfig(suppress_bgr=((10.5, 20, 30),))
    assert np.array_equal(appearance._histogram(px, cfg), appearance._histogram_direct(px, cfg))
