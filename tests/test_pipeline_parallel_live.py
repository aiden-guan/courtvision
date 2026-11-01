"""Live chooses a file worker for playback and a causal worker for compatible camera detectors."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from courtvision.config import AppConfig
from courtvision.geometry.homography import Calibration, Homography
from courtvision.perception.broadcast_court import BroadcastCourtTracker
from courtvision.perception.court_worker import CourtTrackerProcess
from courtvision.perception.live_court_worker import LiveCourtTrackerProcess
from courtvision.pipeline import LiveGame, VideoFileSource
from test_perception_broadcast_auto import CAM, CENTER, COURT, SIZE, FakeInner


@pytest.mark.parametrize("file,compatible,enabled,expected", [
    (True, True, True, CourtTrackerProcess),
    (True, False, True, CourtTrackerProcess),
    (False, True, True, LiveCourtTrackerProcess),
    (False, False, True, BroadcastCourtTracker),
    (True, True, False, BroadcastCourtTracker),
])
def test_parallel_live_worker_selection(file, compatible, enabled, expected):
    cfg = AppConfig()
    cfg.pipeline.parallel_live = enabled
    cfg.perception.scorebug_ocr = False
    source = object.__new__(VideoFileSource) if file else SimpleNamespace()
    source.live, source.path, source.start_s = not file, Path("unused.mp4") if file else None, 0.0
    game = object.__new__(LiveGame)
    game.config, game.source, game.court, game.realtime = cfg, source, COURT, True
    game._auto_center, game._auto_params = CENTER, CAM
    calib = Calibration(Homography(np.linalg.inv(CAM.G(SIZE))), SIZE)
    inner = FakeInner(calib)
    inner.detector.parallel_court = compatible
    pipe = game._broadcast_perception(inner, calib, False)
    try:
        assert type(pipe.auto) is expected
    finally:
        pipe.close()
