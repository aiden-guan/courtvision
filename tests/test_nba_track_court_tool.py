"""tools/nba_track_court.py: the scorebug region kept out of the court evidence."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

from courtvision.perception.scorebug_layout import save_profile
from courtvision.perception.scorebug_ocr import ESPN_2024

TOOLS = Path(__file__).resolve().parents[1] / "tools"


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
    assert boxes[0] == (45, -15, 630, 180)  # (40-10)*1.5, (20-30)*1.5, (400+20)*1.5, (60+60)*1.5 at 1080 lines
    assert len(boxes) == 2
