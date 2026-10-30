"""A run's analysis found from its video and shaped for the Studio's feed (courtvision/nba/results.py)."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
from typing import Any

from courtvision.nba import results
from courtvision.types import Event, EventKind, Team

K = EventKind
_ids = itertools.count()


def ev(kind: EventKind, t: float, team: Team | None = None, **kw: Any) -> Event:
    return Event(f"e{next(_ids)}", kind, t, int(t * 30), team=team, **kw)


def write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def game(nba: Path, name: str, video: str, tags: tuple[str, ...] = ("full",), events: tuple[Event, ...] = (),
         doc: dict[str, Any] | None = None) -> Path:
    root = nba / name
    root.mkdir(parents=True)
    (root / "game.json").write_text(json.dumps({"video": video}))
    for tag in tags:
        run = root / "runs" / tag
        run.mkdir(parents=True)
        (run / "timeline.json").write_text(json.dumps(doc or {"dt_s": 0.25, "segments": [[0.0, 10.0, "away"]]}))
        (run / "timeline.jsonl").write_text("".join(json.dumps(e.to_dict()) + "\n" for e in events))
    return root


def test_a_run_is_found_from_its_video_or_a_copy_of_it(tmp_path: Path) -> None:
    data = os.urandom(3 * results.PROBE)
    original = write(tmp_path / "studio" / "a" / "source.video", data)
    copy = write(tmp_path / "studio" / "b" / "source.video", data)  # the same file uploaded again
    other = write(tmp_path / "studio" / "c" / "source.video", data[:-1] + bytes([data[-1] ^ 1]))
    nba = tmp_path / "nba"
    g = game(nba, "g", "studio/a/source.video")  # relative: from the repo
    assert results.find_run(original, nba, tmp_path)[:2] == (g, "full")  # type: ignore[index]
    assert results.find_run(copy, nba, tmp_path)[:2] == (g, "full")  # type: ignore[index]
    assert results.find_run(other, nba, tmp_path) is None  # same size, different bytes
    assert results.find_run(original, tmp_path / "no games", tmp_path) is None


def test_the_newest_finished_run_of_any_game_on_the_video_wins(tmp_path: Path) -> None:
    video = write(tmp_path / "v.mp4", b"x" * 1000)
    nba = tmp_path / "nba"
    old = game(nba, "a", str(video), tags=("q1",))
    new = game(nba, "b", str(video), tags=("full", "partial"))
    (new / "runs" / "partial" / "timeline.jsonl").unlink()  # still being written: not a run yet
    (nba / "c").mkdir()
    (nba / "c" / "game.json").write_text(json.dumps({"video": str(video)}))  # never analysed
    for path, mtime in ((old / "runs/q1/timeline.json", 1000), (new / "runs/full/timeline.json", 2000),
                        (new / "runs/partial/timeline.json", 3000)):
        os.utime(path, (mtime, mtime))
    assert results.find_run(video, nba, tmp_path) == (new, "full", 2000)


def test_load_shapes_the_events_the_feed_shows_from_the_video(tmp_path: Path) -> None:
    video = write(tmp_path / "v.mp4", b"x")
    events = (ev(K.SHOT_MADE, 5.0, Team.HOME, value=3, attrs={"makes": {"p_real": 0.9, "p_ft": 0.1, "p_made": 0.8712}}),
              ev(K.POSSESSION, 5.0, Team.AWAY, attrs={"cause": "made_shot"}),
              ev(K.PASS, 4.0, Team.AWAY),  # not shown
              ev(K.SHOT_MISSED, 6.0, Team.AWAY, value=1, attrs={"shot_type": "free_throw"}),
              ev(K.REBOUND, 6.5, Team.HOME, attrs={"offensive": True, "inferred": True}),
              ev(K.SHOT_MADE, 7.0, Team.AWAY, value=2, attrs={"inferred": "rim_visit", "makes": {"p_made": 0.6}}),
              ev(K.SHOT_MADE, 8.0, Team.HOME, value=2, source="scoreboard"),  # the scorebug's: never shown
              ev(K.TURNOVER, 9.0, Team.HOME, attrs={"inferred": "scoreboard"}),
              ev(K.STEAL, 9.5))  # no team
    doc = {"dt_s": 0.25, "segments": [[0.0, 5.0, "home"], [5.25, 9.0, "away"]], "back_court": [[5.25, 6.0]],
           "timeline": {"accuracy": 0.95}, "shots": {"made_accuracy": 0.85, "jev": True},
           "reference": {"marks": [[4.999, "home"], [5.0, "away"]], "plays": [[5.0, "Home makes a three"]]}}
    g = game(tmp_path / "nba", "2024-12-25-LAL-GSW", str(video), events=events, doc=doc)
    (g / "espn_summary.json").write_text(json.dumps({"header": {"competitions": [{"competitors": [
        {"homeAway": "home", "team": {"id": "9", "abbreviation": "GS", "displayName": "Golden State Warriors"}},
        {"homeAway": "away", "team": {"id": "13", "abbreviation": "LAL", "displayName": "Los Angeles Lakers"}}]}]}}))
    got = results.load(g, "full")
    assert got["game"] == "2024-12-25-LAL-GSW" and got["tag"] == "full" and got["jev"] is True
    assert got["teams"] == {"home": {"abbr": "GS", "name": "Golden State Warriors"},
                            "away": {"abbr": "LAL", "name": "Los Angeles Lakers"}}
    assert got["segments"] == doc["segments"] and got["back_court"] == doc["back_court"]
    assert got["reference"] == doc["reference"] and got["measures"] == {"possession": 0.95, "made": 0.85}
    assert got["events"] == [
        {"t": 5.0, "kind": "shot_made", "team": "home", "value": 3, "free_throw": False, "found_at_rim": False,
         "p_made": 0.871},
        {"t": 5.0, "kind": "possession", "team": "away", "cause": "made_shot"},
        {"t": 6.0, "kind": "shot_missed", "team": "away", "value": 1, "free_throw": True, "found_at_rim": False,
         "p_made": None},
        {"t": 6.5, "kind": "rebound", "team": "home", "offensive": True},
        {"t": 7.0, "kind": "shot_made", "team": "away", "value": 2, "free_throw": False, "found_at_rim": True,
         "p_made": 0.6}]


def test_a_game_without_the_play_by_play_has_no_teams_reference_or_measures(tmp_path: Path) -> None:
    video = write(tmp_path / "v.mp4", b"x")
    g = game(tmp_path / "nba", "new", str(video), events=(ev(K.SHOT_MADE, 1.0, Team.AWAY, value=2),))
    got = results.load(g, "full")
    assert got["teams"] is None and got["reference"] is None and got["measures"] == {} and got["back_court"] == []
    assert got["events"][0]["p_made"] is None
