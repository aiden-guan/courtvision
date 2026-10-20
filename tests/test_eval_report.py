"""Tests for courtvision.eval.report (markdown + JSON rendering)."""

from __future__ import annotations

import copy
import json

from courtvision.eval.boxscore import compare_box
from courtvision.eval.events import match_events
from courtvision.eval.report import format_report, report_dict
from courtvision.eval.tracking import tracking_report
from courtvision.types import BallState, Event, EventKind, FrameState, PlayerState, Team


def _events() -> tuple[list[Event], list[Event]]:
    truth = [
        Event("t1", EventKind.SHOT_ATTEMPT, 1.0, 30, Team.HOME, "home-1", value=3, shot_id="s1"),
        Event("t2", EventKind.SHOT_MADE, 2.0, 60, Team.HOME, "home-1", value=3, shot_id="s1"),
        Event("t3", EventKind.ASSIST, 2.0, 60, Team.HOME, "home-2", "home-1", shot_id="s1"),
        Event("t4", EventKind.DRIBBLE, 5.0, 150, Team.AWAY, "away-1"),
        Event("t5", EventKind.FOUL, 7.0, 210, Team.AWAY, "away-1", "home-2"),
    ]
    pred = [
        Event("e1", EventKind.SHOT_ATTEMPT, 1.1, 33, Team.HOME, "home-1", value=2, shot_id="s9"),
        Event("e2", EventKind.SHOT_MISSED, 2.0, 60, Team.HOME, "home-1", shot_id="s9"),
        Event("e3", EventKind.DRIBBLE, 5.1, 153, Team.AWAY, "away-1"),
    ]
    return pred, truth


def _reports():
    pred, truth = _events()
    ev = match_events(pred, truth)
    box = compare_box(
        {"players": {"home-1": {"pts": 0, "team": "home"}, "home-t4": {"pts": 2, "team": "home", "provisional": True}},
         "score": {"home": 2, "away": 0}},
        {"home-1": {"pts": 3}, "home-2": {"pts": 0}},
        {Team.HOME: 3, Team.AWAY: 0},
    )
    frames = [FrameState(k, k / 30, [PlayerState("home-1", Team.HOME, 10.0 + k, 20.0)],
                         BallState(True, image_xy=(5.0, 5.0))) for k in range(5)]
    tr = tracking_report(copy.deepcopy(frames), frames)
    return ev, box, tr


def test_format_report_has_all_sections() -> None:
    ev, box, tr = _reports()
    md = format_report(ev, box, tr)
    assert md.startswith("# CourtVision evaluation")
    for heading in ("## Events", "### Reported separately", "### Shots", "## Box score", "## Tracking", "### Ball"):
        assert heading in md
    assert "| kind | truth | pred |" in md
    assert "| dribble |" in md and "| foul |" in md
    assert "FG outcome confusion" in md and "| truth made | 0 | 1 | 0 |" in md
    assert "Score match: **no**" in md and "home-t4" in md and "home-2" in md
    assert "| MOTA | 1.000 |" in md
    # Every row of a table has the header's column count.
    tables: list[list[str]] = []
    prev_is_row = False
    for line in md.splitlines():
        is_row = line.startswith("|")
        if is_row:
            assert line.endswith("|")
            if not prev_is_row:
                tables.append([])
            tables[-1].append(line)
        prev_is_row = is_row
    assert len(tables) >= 6
    for rows in tables:
        widths = {r.count("|") for r in rows}
        assert len(widths) == 1, rows


def test_format_report_accepts_dicts_and_optional_sections() -> None:
    ev, box, tr = _reports()
    from_dicts = format_report(json.loads(json.dumps(ev.to_dict())), box.to_dict(), tr.to_dict())
    assert from_dicts == format_report(ev, box, tr)
    only_events = format_report(ev)
    assert "## Box score" not in only_events and "## Tracking" not in only_events
    assert format_report(None, box).count("##") == 1


def test_report_dict_is_json_serialisable() -> None:
    ev, box, tr = _reports()
    d = report_dict(ev, box, tr)
    assert set(d) == {"events", "boxscore", "tracking"}
    assert d["events"]["headline"]["kinds"] == ["shot_attempt", "shot_made", "shot_missed", "assist"]
    assert d["boxscore"]["score_match"] is False and d["tracking"]["mota"] == 1.0
    json.dumps(d)
    assert report_dict(ev)["tracking"] is None
