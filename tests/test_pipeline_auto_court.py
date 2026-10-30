"""Automatic court registration in the pipeline: probes at a saved camera centre are rate-limited and counted."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from courtvision.config import PerceptionSettings
from courtvision.geometry.court import court_for
from courtvision.perception import broadcast_court
from courtvision.perception.broadcast_court import CourtFix
from courtvision.pipeline import LiveGame


def test_failed_probes_at_a_saved_centre_use_up_the_tries(monkeypatch) -> None:  # noqa: ANN001
    probes: list[float] = []

    class NoCourt:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            pass

        def update(self, frame, cut=False):  # noqa: ANN001, ANN202
            probes.append(1.0)
            return CourtFix("lost", None)

    monkeypatch.setattr(broadcast_court, "BroadcastCourtTracker", NoCourt)
    pc = PerceptionSettings(broadcast="on", auto_court=True, auto_court_frames=3, auto_court_every_s=1.0,
                            auto_court_tries=2, calibration_path=None)
    game = SimpleNamespace(config=SimpleNamespace(perception=pc), court=court_for("nba"),
                           source=SimpleNamespace(synthetic=False), _auto_center=np.array([47.0, -94.0, 36.0]),
                           _auto_params=None, _auto_frames=[], _auto_tries=0, _auto_probes=0, _auto_probe_t=None)
    img = np.zeros((540, 960, 3), np.uint8)
    t = 0.0
    while LiveGame._auto_court_wanted(game):  # type: ignore[arg-type]
        assert LiveGame._auto_court(game, img, t) is None  # type: ignore[arg-type]
        t += 0.25
        assert t < 60.0, "the retries never ran out"
    assert len(probes) == 6  # one probe a second; every 3 failed probes are one of the 2 tries
    assert 5.0 <= t <= 6.5
