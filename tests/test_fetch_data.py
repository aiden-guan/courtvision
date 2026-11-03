"""tools/fetch_data.py: which SportsMOT tar members are written and where, and the BARD clip lists."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import fetch_data as fd  # noqa: E402


def test_target_keeps_listed_sequences_inside_the_folder(tmp_path: Path) -> None:
    keep = {"v_abc_c001"}
    assert fd.target("val/v_abc_c001/img1/000001.jpg", "val", keep, tmp_path) == \
        tmp_path / "val" / "v_abc_c001" / "img1" / "000001.jpg"
    assert fd.target("val/v_abc_c001/gt/gt.txt", "val", keep, tmp_path) is not None
    assert fd.target("val/v_other_c001/img1/000001.jpg", "val", keep, tmp_path) is None  # not basketball
    assert fd.target("train/v_abc_c001/img1/000001.jpg", "val", keep, tmp_path) is None  # another split
    assert fd.target("val/._v_abc_c001", "val", keep, tmp_path) is None  # macOS resource fork
    assert fd.target("val/v_abc_c001/img1/._000001.jpg", "val", keep, tmp_path) is None
    assert fd.target("val/v_abc_c001/../../etc/passwd", "val", keep, tmp_path) is None


def test_bard_clips_reads_a_list_and_refuses_odd_names(tmp_path: Path) -> None:
    assert len(fd.bard_clips("check_clips")) == 90  # the repo's splits file
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"bard": {"ok": ["g/1.mp4"], "bad": ["../x.mp4"]}}))
    assert fd.bard_clips("ok", splits) == ["g/1.mp4"]
    with pytest.raises(SystemExit):
        fd.bard_clips("bad", splits)
    with pytest.raises(SystemExit):
        fd.bard_clips("missing", splits)


def test_roboflow_split_takes_exact_clips_from_the_named_folders_and_renumbers() -> None:
    cats = [{"id": 1, "name": "player"}]
    anns = {"train": {"categories": cats,
                      "images": [{"id": 1, "file_name": "game-a-q1_mp4-0001_jpg.rf.x.jpg"},
                                 {"id": 2, "file_name": "game-a-q1-long_mp4-0001_jpg.rf.y.jpg"}],
                      "annotations": [{"id": 1, "image_id": 1}, {"id": 2, "image_id": 2}]},
            "valid": {"categories": cats,
                      "images": [{"id": 1, "file_name": "game-a-q1_mov-0002_jpg.rf.z.jpg"}],
                      "annotations": [{"id": 1, "image_id": 1}]}}
    assert fd.rf_clip("game-a-q1_mov-0002_jpg.rf.z.jpg") == "game-a-q1"
    part = fd.roboflow_split(anns, ["game-a-q1"], ("train", "valid"))
    assert [(im["id"], im["folder"]) for im in part["images"]] == [(1, "train"), (2, "valid")]  # not "game-a-q1-long"
    assert [(a["id"], a["image_id"]) for a in part["annotations"]] == [(1, 1), (2, 2)]
    assert part["categories"] == cats
    assert len(fd.roboflow_split(anns, ["game-a-q1"], ("valid",))["images"]) == 1
