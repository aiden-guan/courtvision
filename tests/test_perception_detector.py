"""ColorBlobDetector and appearance features on synthetic drawn frames."""

from __future__ import annotations

import numpy as np
import pytest
from test_perception_synth import (
    SCOREBOARD_ROI,
    SynthPlayer,
    draw_court,
    draw_scoreboard,
    make_camera,
    make_court,
    render,
    rim_bbox,
    scoreboard_text,
)

from courtvision.perception.appearance import color_feature, feature_distance, feature_size, torso_feature
from courtvision.perception.detector import ColorBlobDetector, Detector, DetectorConfig
from courtvision.sim import palette
from courtvision.types import BBox, ObjClass, Team


@pytest.fixture(scope="module")
def scene():
    cam, court = make_camera(), make_court()
    people = [
        SynthPlayer(10, 10, Team.HOME, "23"),
        SynthPlayer(80, 40, Team.AWAY, "7"),
        SynthPlayer(47, 25, None),
        SynthPlayer(60, 12, Team.HOME, "4"),
        SynthPlayer(25, 38, Team.AWAY, "11"),
    ]
    img, boxes, ball = render(cam, court, people, ball_xyz=(35.0, 20.0, 5.0))
    draw_scoreboard(img, SCOREBOARD_ROI, scoreboard_text(45, 38, 2, 7 * 60 + 3, 12))
    return cam, court, people, img, boxes, ball


def _match(dets, box: BBox):
    return max(dets, key=lambda d: d.bbox.iou(box))


def test_detects_players_referee_ball_and_rims(scene) -> None:
    cam, court, people, img, boxes, ball = scene
    det = ColorBlobDetector()
    det.add_mask_polygon(SCOREBOARD_ROI["corners"])
    assert isinstance(det, Detector)
    fd = det.detect(img, 7, 0.25)
    assert fd.frame_idx == 7 and fd.image_size == (1280, 720)
    players, refs = fd.of(ObjClass.PLAYER), fd.of(ObjClass.REFEREE)
    assert len(players) == 4 and len(refs) == 1
    for p, box in zip(people, boxes):
        d = _match(refs if p.team is None else players, box)
        assert d.bbox.iou(box) > 0.85, (p, d.bbox, box)
        assert abs(d.bbox.y2 - box.y2) < 1.5  # foot point within a pixel or so
        assert abs(d.bbox.center[0] - box.center[0]) < 1.5
    (b,) = fd.of(ObjClass.BALL)
    u, v, r = ball
    assert np.hypot(b.bbox.center[0] - u, b.bbox.center[1] - v) < 0.75
    assert abs((b.bbox.w + b.bbox.h) / 4.0 - r) < 1.0
    rims = sorted(fd.of(ObjClass.RIM), key=lambda d: d.bbox.x1)
    assert len(rims) == 2
    for d, side in zip(rims, ("left", "right")):
        assert d.bbox.iou(rim_bbox(cam, court, side).expand(0.2)) > 0.5


def test_player_embeddings_separate_teams(scene) -> None:
    _, _, people, img, boxes, _ = scene
    fd = ColorBlobDetector().detect(img, 0, 0.0)
    feats = {p.number: _match(fd.of(ObjClass.PLAYER), b).embedding for p, b in zip(people, boxes) if p.team}
    assert all(f is not None and f.shape == (feature_size(),) and abs(f.sum() - 1) < 1e-5 for f in feats.values())
    assert feature_distance(feats["23"], feats["4"]) < 0.1  # same team
    assert feature_distance(feats["23"], feats["7"]) > 0.9  # different teams
    assert feature_distance(feats["7"], color_feature(palette.AWAY_BGR)) < 0.1


def test_empty_court_and_scoreboard_give_no_people_or_ball() -> None:
    cam, court = make_camera(), make_court()
    img = draw_court(cam, court)
    draw_scoreboard(img, SCOREBOARD_ROI, scoreboard_text(188, 8, 4, 0, 0))
    fd = ColorBlobDetector().detect(img, 0, 0.0)  # no mask: shape filters must reject the panel
    assert fd.detections == []


def test_mask_polygon_hides_region(scene) -> None:
    _, _, people, img, boxes, _ = scene
    box = boxes[1]
    poly = [[box.x1 - 2, box.y1 - 2], [box.x2 + 2, box.y1 - 2], [box.x2 + 2, box.y2 + 2], [box.x1 - 2, box.y2 + 2]]
    det = ColorBlobDetector(DetectorConfig(mask_polygons=[poly]))
    fd = det.detect(img, 0, 0.0)
    assert all(d.bbox.iou(box) < 0.1 for d in fd.detections)
    assert len(fd.of(ObjClass.PLAYER)) == 3


def test_head_fallback_when_no_skin() -> None:
    img = np.zeros((200, 200, 3), np.uint8)
    img[:] = palette.FLOOR_BGR
    img[100:160, 90:110] = palette.HOME_BGR  # headless jersey blob
    (d,) = ColorBlobDetector().detect(img, 0, 0.0).detections
    assert d.bbox.y2 == pytest.approx(159.5)
    assert d.bbox.y1 == pytest.approx(99.5 - 0.25 * 60, abs=1.0)


def test_torso_feature_ignores_court_and_number() -> None:
    img = np.zeros((100, 60, 3), np.uint8)
    img[:] = palette.FLOOR_BGR
    img[20:100, 10:50] = palette.AWAY_BGR
    img[40:50, 15:45] = palette.NUMBER_BGR
    f = torso_feature(img, BBox(10, 0, 50, 100))
    assert feature_distance(f, color_feature(palette.AWAY_BGR)) < 1e-6
    assert not torso_feature(img, BBox(500, 500, 510, 510)).any()


def test_overlapping_same_team_players_are_split_by_heads() -> None:
    cam, court = make_camera(), make_court()
    det = ColorBlobDetector()
    # Warm the online body/head proportions on single figures.
    warm, _, _ = render(cam, court, [SynthPlayer(15 + 12 * k, 10 + 7 * k, Team.HOME, str(k)) for k in range(5)])
    det.detect(warm, 0, 0.0)
    pair = [SynthPlayer(40.0, 20.0, Team.HOME, "3"), SynthPlayer(41.2, 24.0, Team.HOME, "8")]
    img, boxes, _ = render(cam, court, pair)
    assert boxes[0].iou(boxes[1]) > 0.1  # the two figures really overlap
    players = det.detect(img, 1, 0.0).of(ObjClass.PLAYER)
    assert len(players) == 2
    for box in boxes:
        d = _match(players, box)
        assert d.bbox.iou(box) > 0.7
        assert abs(d.bbox.y2 - box.y2) < 4.0  # feet: observed for the near player, predicted for the far one
    no_split = ColorBlobDetector(DetectorConfig(split_merged=False)).detect(img, 1, 0.0).of(ObjClass.PLAYER)
    assert len(no_split) == 1
