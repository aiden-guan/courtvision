"""tools/nba_overlay.py without a video: the per-frame lookups, the labels, the busiest stretch, drawing a frame and
reading a run (its play-by-play reference included)."""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from courtvision.types import Event, EventKind, Team

TOOLS = Path(__file__).resolve().parents[1] / "tools"
K = EventKind
HOME, AWAY = Team.HOME, Team.AWAY
_ids = itertools.count()


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


overlay = _load("nba_overlay")


def ev(kind: EventKind, t: float, team: Team | None = HOME, **kw: Any) -> Event:
    return Event(f"e{next(_ids)}", kind, t, int(t * 30), team=team, **kw)


def view(**kw: Any) -> Any:
    events = [ev(K.SHOT_MADE, 10.0, HOME, value=3, attrs={"makes": {"p_made": 0.91}}),
              ev(K.POSSESSION, 10.0, AWAY, attrs={"cause": "made_shot"}),
              ev(K.SHOT_MISSED, 30.0, AWAY, value=2)]
    return overlay.Overlay({HOME: "GS", AWAY: "LAL"}, {HOME: (150, 75, 25), AWAY: (230, 230, 230)},
                           [(0.0, 10.0, HOME), (10.25, 40.0, AWAY)], [(10.25, 14.0)], events, **kw)


def test_the_lookups_follow_the_timeline_and_the_play_by_play() -> None:
    ov = view(marks=[(9.999, HOME), (10.0, AWAY), (30.0, None)], pbp=[(10.0, "Curry makes a three")])
    assert ov.teams_at(np.array([5.0, 12.0, 41.0])) == [HOME, AWAY, None]
    assert ov.truths_at(np.array([1.0, 12.0, 31.0])) == [None, AWAY, None]
    assert ov.bringing_up(12.0) and not ov.bringing_up(20.0)
    assert [e.kind for e in ov.recent(11.0, 2.5)] == [K.SHOT_MADE, K.POSSESSION]
    assert ov.pbp_line(12.0) == "Curry makes a three" and ov.pbp_line(20.0) is None


def test_labels_say_what_was_decided() -> None:
    names = {HOME: "GS", AWAY: "LAL"}
    assert overlay.label(ev(K.SHOT_MADE, 1.0, HOME, value=3, attrs={"makes": {"p_made": 0.91}}), names)[:2] == (
        "GS 3-POINTER MADE", "P(made) 0.91")
    assert overlay.label(ev(K.SHOT_MISSED, 1.0, AWAY, value=1, attrs={"shot_type": "free_throw"}), names)[0] == (
        "LAL FREE THROW MISSED")
    added = overlay.label(ev(K.SHOT_MADE, 1.0, AWAY, value=2, attrs={"inferred": "rim_visit"}), names)
    assert added[1].startswith("the engine had missed this one")
    assert overlay.label(ev(K.POSSESSION, 1.0, AWAY, attrs={"cause": "defensive_rebound"}), names)[:2] == (
        "LAL BALL", "defensive rebound")
    assert overlay.label(ev(K.PASS, 1.0), names) is None


def test_the_busiest_stretch_in_live_play_is_picked() -> None:
    events = [ev(K.SHOT_MADE, t, value=2) for t in (20.0, 25.0, 300.0)] + [
        ev(K.SHOT_MADE, 310.0, value=1), ev(K.STEAL, 315.0), ev(K.SHOT_MADE, 600.0, value=2)]
    tracked = np.arange(0.0, 700.0, 1 / 30)
    start, end = overlay.pick_window(events, 60.0, tracked)
    assert end - start == 60.0 and start <= 300.0 and end > 315.0  # a free throw and a steal outweigh two shots
    replay = tracked[(tracked < 270.0) | (tracked > 340.0)]  # the camera off the court there
    start, end = overlay.pick_window(events, 60.0, replay)
    assert not any(start <= t < end for t in (300.0, 310.0, 315.0))


def test_a_frame_is_drawn_with_the_bar_the_rims_the_ball_and_the_strip() -> None:
    img = np.zeros((360, 640, 3), np.uint8)
    ov = view(marks=[(9.999, HOME), (10.0, AWAY)], pbp=[(10.0, "Curry makes a three")])
    overlay.draw(img, 11.0, ov, {"left": (100.0, 200.0, 130.0, 220.0)}, (400.0, 300.0), 0.5)
    assert img[:20].any() and img[-20:].any()  # the bar along the top and the strip along the bottom
    assert (img[90:120, 40:80] == overlay.GREEN).all(axis=-1).any()  # the rim lit green for the made three


def test_a_run_is_read_with_its_play_by_play_reference(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "full"
    run.mkdir(parents=True)
    (run / "timeline.json").write_text(json.dumps({
        "segments": [[0.0, 10.0, "home"], [10.25, 40.0, "away"]], "back_court": [[10.25, 14.0]],
        "reference": {"marks": [[9.999, "home"], [10.0, None]], "plays": [[10.0, "Curry makes a three"]]}}))
    (run / "timeline.jsonl").write_text("".join(json.dumps(e.to_dict()) + "\n" for e in view().events))
    ov = overlay.load_overlay(tmp_path, "full", {"home_color_bgr": [1, 2, 3]})
    assert ov.segments == [(0.0, 10.0, HOME), (10.25, 40.0, AWAY)] and ov.back == [(10.25, 14.0)]
    assert ov.marks == [(9.999, HOME), (10.0, None)] and ov.pbp == [(10.0, "Curry makes a three")]
    assert ov.colours[HOME] == (1, 2, 3) and ov.names == {HOME: "HOME", AWAY: "AWAY"}
    assert [e.t for e in ov.events] == sorted(e.t for e in ov.events)
