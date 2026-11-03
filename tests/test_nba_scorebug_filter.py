"""The scorebug filter in the research tools: ``nba_analyze.py ball --scorebug-filter on`` (the game's scorebug as the
court tracker masks it, in its frames' pixels, recorded in the ball track; ``--ball-track`` leaves ball_track.json
alone), the run's ball track for the later stages, and ``nba_ball_bench.py``'s comparison on the same detections."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from courtvision.nba.analyze import BallTrack, DetectionCache
from courtvision.perception.scorebug_layout import save_profile
from courtvision.perception.scorebug_cover import DROP_BALLS_IN_BUG
from test_nba_track_court_tool import TOP_RIGHT_720, TOP_RIGHT_MASK

TOOLS = Path(__file__).resolve().parents[1] / "tools"
ESPN = [(468, 944, 1000, 125)]  # the ESPN 2024 bug and its lower row at 1080 lines
LOGO, BALL = (530.0, 979.0, 0.7), (960.0, 600.0, 0.45)  # a still team logo in the ESPN bug; the still ball


def load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"{name}_scorebug_filter", TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def game(tmp_path: Path, balls: list[tuple[float, float, float]], size: tuple[int, int] = (1920, 1080)) -> Path:
    """A game folder whose camera holds still for 2 s with the same ball candidates in every frame."""
    root = tmp_path / "g"
    (root / "camera_parts").mkdir(parents=True)
    (root / "detections").mkdir()
    (root / "game.json").write_text(json.dumps({"video": str(tmp_path / "game.mp4")}))
    rows = [{"header": True, "center": [47.0, -94.0, 36.0], "image_size": list(size), "fps": 29.97}]
    rows += [{"frame": f, "t": f / 29.97, "status": "tracking",
              "params": {"pan": 0.0, "tilt": 15.0, "roll": 0.0, "f": 3000.0 * size[1] / 1080.0}} for f in range(60)]
    (root / "camera_parts" / "part0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    DetectionCache({f: np.array(balls, np.float32) for f in range(60)}).save(root / "detections" / "0-60.npz")
    return root


def run(tool: ModuleType, monkeypatch: pytest.MonkeyPatch, *argv: object) -> None:
    monkeypatch.setattr(sys, "argv", ["nba_analyze.py", *map(str, argv)])
    assert tool.main() == 0


def linked_at(track: BallTrack, xy: tuple[float, float, float]) -> bool:
    return len(track.path) == 60 and all(np.allclose(p[:2], xy[:2]) for p in track.path.values())


def test_ball_with_the_filter_links_the_ball_into_its_own_file_and_leaves_ball_track_json(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    tool = load_tool("nba_analyze")
    assert tool.SCOREBUG_FILTER == ("on" if DROP_BALLS_IN_BUG else "off")  # the switch's shared default
    root = game(tmp_path, [LOGO, BALL])
    run(tool, monkeypatch, root, "ball", "--scorebug-filter", "off")
    plain = (root / "ball_track.json").read_text()
    assert "scorebug_boxes" not in json.loads(plain) and linked_at(BallTrack.load(root / "ball_track.json"), LOGO)
    capsys.readouterr()
    run(tool, monkeypatch, root, "ball", "--scorebug-filter", "on", "--ball-track", "ball_track_bug.json")
    out = capsys.readouterr().out
    assert "60 of 120 ball candidates dropped, 60 of them scored 0.08 or more" in out
    assert "best candidate on 60 of 60 frames" in out
    track = BallTrack.load(root / "ball_track_bug.json")
    assert track.scorebug_boxes == ESPN and linked_at(track, BALL)  # no saved profile: the ESPN 2024 bug
    assert (root / "ball_track.json").read_text() == plain  # left alone


def test_the_filter_takes_the_games_saved_scorebug_in_its_frames_pixels(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    tool = load_tool("nba_analyze")
    logo = (1100.0, 43.0, 0.7)  # in the top-right bug of a 720p game
    root = game(tmp_path, [logo, (640.0, 400.0, 0.45)], size=(1280, 720))
    save_profile(TOP_RIGHT_720, root / "scorebug_profile.json")  # as nba_scan_scorebug.py saves it, beside the reads
    run(tool, monkeypatch, root, "ball", "--scorebug-filter", "on")
    track = BallTrack.load(root / "ball_track.json")
    assert track.scorebug_boxes == [(993, 9, 253, 68), (939, 17, 61, 52)]  # TOP_RIGHT_MASK at 720 lines
    assert linked_at(track, (640.0, 400.0, 0.45))
    run(tool, monkeypatch, root, "ball", "--scorebug-filter", "on", "--scorebug-profile", "espn-2024",
        "--ball-track", "espn.json")  # another bug named: ESPN's, which this logo is not in
    track = BallTrack.load(root / "espn.json")
    assert track.scorebug_boxes == [(312, 629, 667, 83)] and linked_at(track, logo)


def test_the_later_stages_read_the_ball_track_the_runs_events_were_made_with(tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    tool = load_tool("nba_analyze")
    loaded: list[Path] = []
    monkeypatch.setattr(tool, "load_game", lambda root: ({"video": "unused"}, None))
    monkeypatch.setattr(tool, "load_cache", lambda *args: None)
    monkeypatch.setattr(tool.BallTrack, "load", lambda path: loaded.append(path))
    monkeypatch.setattr(tool, "load_scorebug", lambda path: [])
    monkeypatch.setattr(tool, "perceive", lambda *args: [])
    monkeypatch.setattr(tool, "run_engine", lambda *args, **kwargs: [])
    events = dict(start=0.0, end=0.0, use_3d=False, engine=None, det_dir="detections", team_refit=False)
    tool.cmd_events(tmp_path, argparse.Namespace(**events, tag="off"))  # a namespace without the option (as before)
    tool.cmd_events(tmp_path, argparse.Namespace(**events, tag="on", ball_track="ball_track_bug.json"))
    assert loaded == [tmp_path / "ball_track.json", tmp_path / "ball_track_bug.json"]
    runs = tmp_path / "runs"
    assert json.loads((runs / "on" / "run.json").read_text())["ball_track"] == "ball_track_bug.json"
    assert tool.run_ball_track(argparse.Namespace(ball_track=None), runs / "on") == "ball_track_bug.json"
    assert tool.run_ball_track(argparse.Namespace(), runs / "off") == "ball_track.json"
    assert tool.run_ball_track(argparse.Namespace(ball_track="x.json"), runs / "on") == "x.json"  # asked for
    (runs / "old").mkdir()
    (runs / "old" / "run.json").write_text(json.dumps({"start": 0.0, "end": 0.0}))  # written before the option
    assert tool.run_ball_track(argparse.Namespace(ball_track=None), runs / "old") == "ball_track.json"
    assert tool.run_ball_track(argparse.Namespace(ball_track=None), runs / "none") == "ball_track.json"


def test_the_rim_zones_read_the_candidates_the_ball_was_linked_from(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    tool = load_tool("nba_analyze")
    cache = DetectionCache({5: np.array([LOGO, BALL], np.float32)})
    monkeypatch.setattr(tool, "load_cache", lambda root, det_dir: cache)
    monkeypatch.setattr("courtvision.nba.rim_jev.rim_centre", lambda *args, **kwargs: (5, (0.0, 0.0)))
    seen: list[np.ndarray] = []
    monkeypatch.setattr("courtvision.nba.rim_clips.ball_zone_series", lambda balls, *args: seen.append(balls[5]))
    shot = SimpleNamespace(t=1.0, side="left")
    for boxes in ([], ESPN):
        tool.zone_reader(tmp_path, None, None, BallTrack({5: BALL}, [], boxes), "detections")(shot)
    assert [len(b) for b in seen] == [2, 1] and np.allclose(seen[1][0], BALL)


def test_the_bench_compares_the_filter_on_the_same_detections_and_counts_what_lies_in_the_bug(tmp_path: Path) -> None:
    bench = load_tool("nba_ball_bench")
    teacher = tmp_path / "g" / "teacher"
    teacher.mkdir(parents=True)
    assert bench.game_scorebug(teacher) == ESPN  # nothing saved for the game: the ESPN 2024 bug
    save_profile(TOP_RIGHT_720, tmp_path / "g" / "scorebug_profile.json")
    assert bench.game_scorebug(teacher) == TOP_RIGHT_MASK and bench.game_scorebug(teacher, "espn-2024") == ESPN
    labels = {(5, 1): {"w": 1.0, "xy": [960.0, 600.0]}, (5, 2): {"w": 1.0, "xy": [530.0, 979.0]},
              (5, 3): {"w": 0.2, "xy": [10.0, 10.0]}, (5, 4): {"w": 1.0, "xy": None}, (15, 1): {"w": 1.0, "xy": [1, 1]}}
    results = {"student": {(5, 1): (530.0, 979.0), (5, 2): (530.0, 979.0), (5, 3): None, (5, 4): None},
               bench.FILTERED: {(5, 1): (961.0, 601.0), (5, 2): None, (5, 3): None, (5, 4): None}}
    report = bench.score(results, labels, 15.0)
    assert report["student"] == {"labelled_frames": 2, "hit": 0.5, "wrong": 0.5, "miss": 0.0}
    assert report[bench.FILTERED] == {"labelled_frames": 2, "hit": 0.5, "wrong": 0.0, "miss": 0.5}
    assert bench.labels_inside(labels, results["student"], ESPN, {}) == 1  # a label in the bug only the filter loses
    check = {"frames": 0, "best_candidate_inside": 0, "linked_inside": 0}
    balls = {1: np.array([LOGO, BALL], np.float32), 2: np.array([BALL, (530.0, 979.0, 0.05)], np.float32),
             3: np.zeros((0, 3), np.float32)}
    bench.tally(check, balls, {1: LOGO, 2: BALL}, ESPN, [1, 2, 3], 0.08)
    assert check == {"frames": 3, "best_candidate_inside": 1, "linked_inside": 1}
