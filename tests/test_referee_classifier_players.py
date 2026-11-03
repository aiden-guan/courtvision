"""tools/referee_classifier.py: player-only folders as samples that are not referees."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import referee_classifier as rc  # noqa: E402


def test_load_players_takes_every_nth_labelled_frame_as_players(tmp_path: Path) -> None:
    (tmp_path / "frames" / "0000").mkdir(parents=True)
    rows = []
    for f in range(1, 7):
        img = np.full((270, 480, 3), 40, np.uint8)
        cv2.rectangle(img, (100, 60), (140, 200), (20, 20, 20), -1)  # a dark jersey
        cv2.imwrite(str(tmp_path / "frames" / "0000" / f"{f:06d}.jpg"), img)
        row = {"clip": 0, "frame": f, "size": [960, 540], "persons": [[200, 120, 280, 400, 1.0]], "balls": []}
        if f % 2:
            row["context"] = True
        rows.append(row)
    (tmp_path / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "meta.json").write_text(json.dumps({"clips": [{"clip": 0, "sequence": "v_abc_c007"}]}))
    X, y, games, rule = rc.load_players(tmp_path, every=2, jitter=1, seed=0)
    assert len(X) == 2 and not y.any()  # labelled frames 2, 4, 6: every 2nd is frame 4; one box and one jittered copy
    assert games == ["sportsmot:v_abc", "sportsmot:v_abc"]
    assert rule[0] is not None and rule[1] is None  # the rule's call on the labelled box only
