"""tools/nba_track_court.py: the scorebug region kept out of the court evidence."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from courtvision.geometry.court import court_for
from courtvision.perception.broadcast_court import BroadcastCourtTracker, BroadcastTrackerConfig
from courtvision.perception.court_search import EvidenceConfig
from courtvision.perception.scorebug_layout import save_profile
from courtvision.perception.scorebug_ocr import ESPN_2024

TOOLS = Path(__file__).resolve().parents[1] / "tools"
# a bug in the top-right corner, found on a 1080p and on a 720p copy of a game (the same place: 720p x 1.5)
TOP_RIGHT = dataclasses.replace(ESPN_2024, name="top-right", ocr_box=(1500, 39, 360, 51), logo_box=(1419, 45, 72, 39))
TOP_RIGHT_720 = dataclasses.replace(TOP_RIGHT, ocr_box=(1000, 26, 240, 34), logo_box=(946, 30, 48, 26),
                                    ref_size=(1280, 720))
TOP_RIGHT_MASK = [(1490, 14, 380, 102), (1409, 26, 92, 78)]  # x +-10, y +-h/2 at 1080 lines


def load_tool():  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("nba_track_court", TOOLS / "nba_track_court.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nba_track_court"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_discovered_scorebug_is_masked_and_the_espn_bug_without_one(tmp_path) -> None:  # noqa: ANN001
    tool = load_tool()
    reads = tmp_path / "scorebug.jsonl"
    reads.write_text("")
    assert tool.bug_exclude(str(reads)) == tool.ESPN_EXCLUDE  # no profile: the ESPN 2024 bug
    save_profile(ESPN_2024, tmp_path / "scorebug_profile.json")
    assert tool.bug_exclude(str(reads)) == tool.ESPN_EXCLUDE  # the ESPN profile itself: unchanged
    top = dataclasses.replace(ESPN_2024, name="discovered", ocr_box=(40, 20, 400, 60), logo_box=(40, 20, 80, 30),
                              ref_size=(1280, 720))
    save_profile(top, tmp_path / "scorebug_profile.json")
    boxes = tool.bug_exclude(str(reads))
    assert boxes[0] == (50, -15, 620, 180)  # 40*1.5-10, (20-30)*1.5, 400*1.5+20, (60+60)*1.5 at 1080 lines
    assert len(boxes) == 2


def test_the_same_bug_found_at_1080p_or_720p_gives_one_mask(tmp_path) -> None:  # noqa: ANN001
    tool = load_tool()
    reads = str(tmp_path / "scorebug.jsonl")
    for prof in (TOP_RIGHT, TOP_RIGHT_720):
        path = save_profile(prof, tmp_path / f"profile{prof.ref_size[1]}.json")
        assert tool.bug_exclude(reads, profile=str(path)) == TOP_RIGHT_MASK


@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720)])
def test_the_bug_leaves_the_line_evidence_of_a_1080p_and_a_720p_video(size, tmp_path) -> None:  # noqa: ANN001
    tool = load_tool()
    W, H = size
    s = H / 1080.0
    bug = TOP_RIGHT.scaled(size)
    (ox, oy, ow, oh), (lx, ly, lw, lh) = bug.ocr_box, bug.logo_box
    pad = round(4 * s)  # the bug's panel edge, a few px outside its text crop
    x0, y0 = min(ox, lx) - pad, min(oy, ly) - pad
    x1, y1 = max(ox + ow, lx + lw) + pad, max(oy + oh, ly + lh) + pad
    img = np.full((H, W, 3), (120, 170, 215), np.uint8)  # bare floor (light wood, BGR)
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), max(2, round(3 * s)))

    def lines(exclude: list[tuple[int, int, int, int]]) -> np.ndarray:
        cfg = BroadcastTrackerConfig(evidence=EvidenceConfig(exclude=exclude))
        return BroadcastCourtTracker(court_for("nba"), size, np.array([47.0, -94.0, 36.0]), cfg).evidence(img).lines

    assert lines(tool.ESPN_EXCLUDE).any()  # the ESPN mask is elsewhere: this bug's edges pass as court lines
    reads = str(tmp_path / "scorebug.jsonl")
    for prof in (TOP_RIGHT, TOP_RIGHT_720):
        path = save_profile(prof, tmp_path / f"profile{prof.ref_size[1]}.json")
        assert not lines(tool.bug_exclude(reads, profile=str(path))).any()


def test_the_profile_option_and_the_profile_saved_for_the_video(tmp_path) -> None:  # noqa: ANN001
    tool = load_tool()
    video = tmp_path / "runs" / "studio" / "v1" / "source.video"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")
    game = tmp_path / "runs" / "nba" / "g1"
    game.mkdir(parents=True)
    (game / "game.json").write_text(json.dumps({"video": str(video)}))
    reads = str(tmp_path / "parts" / "scorebug.jsonl")  # read elsewhere: no profile beside them
    assert tool.bug_exclude(reads, str(video)) == tool.ESPN_EXCLUDE  # nothing saved yet
    save_profile(TOP_RIGHT_720, game / "scorebug_profile.json")
    assert tool.bug_exclude(reads, str(video)) == TOP_RIGHT_MASK  # the game folder whose game.json names the video
    assert tool.bug_exclude(reads, str(video), "espn-2024") == tool.ESPN_EXCLUDE  # --scorebug-profile wins
    with pytest.raises(OSError):  # a profile asked for but unreadable is an error, not the ESPN bug
        tool.bug_exclude(reads, str(video), str(tmp_path / "missing.json"))
