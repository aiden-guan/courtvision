"""Direction evidence and CLI clip bounds must follow a variable-rate video's timestamps."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from courtvision.nba.analyze import CameraTrack
from courtvision.nba.fusion import PlayDirection


def camera():
    return CameraTrack(np.array([47., -94., 36.]), (1920, 1080), 30.,
                       params={f: (5., 16., 0., 4400.) for f in range(10)},
                       times={f: float(f) for f in range(10)})


def test_play_direction_uses_the_decoder_timestamps():
    direction = PlayDirection.from_camera(camera(), smooth_s=0)
    assert direction.t.tolist() == list(range(10))
    assert direction.aim(8.0) is not None


def test_detect_and_events_clip_bounds_use_the_decoder_timestamps(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("nba_analyze_timing", Path(__file__).parents[1] / "tools/nba_analyze.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    monkeypatch.setattr(tool, "load_game", lambda root: ({"video": "unused"}, camera()))
    args = argparse.Namespace(start=2., end=5., workers=1, chunks=1, det_dir="detections", weights="unused",
                              backend="torch", use_3d=True, engine=None, tag="clip")
    detected = []
    monkeypatch.setattr(tool, "_detect_part", lambda *args: detected.append(args[3:5]))
    tool.cmd_detect(tmp_path, args)
    assert detected == [(2, 5)]

    monkeypatch.setattr(tool, "load_cache", lambda *args: None)
    monkeypatch.setattr(tool.BallTrack, "load", lambda path: None)
    monkeypatch.setattr(tool, "load_scorebug", lambda path: [])
    perceived = []
    monkeypatch.setattr(tool, "perceive", lambda *args: perceived.append(args[-2:]) or [])
    monkeypatch.setattr(tool, "run_engine", lambda *args, **kwargs: [])
    tool.cmd_events(tmp_path, args)
    assert perceived == [(2, 5)]
    assert json.loads((tmp_path / "runs/clip/run.json").read_text())["start"] == 2.
