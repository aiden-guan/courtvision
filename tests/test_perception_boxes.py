"""Player boxes whose feet are hidden or cut, on the simulator's own renderer.

Regressions for the round-2 court-position root causes (sim train seeds 1-3, video mode): 4 % of
perceived player boxes ended more than 5 px from the true feet (1.9 % more than 10 px, ~3 ft):

* the skin "leg" search below a shirt blob took a nearer player's head just under it for legs
  (46 % of the boxes it extended were 5-15 px too low; seed 3);
* a ball in front of a body cut its visible shirt in two, so the head's box ended at the ball and a
  headless phantom box stood below it (the ball holder, hidden behind a defender);
* before the figure model has its spread, hidden feet came from head-width proportions (a 1-px
  head-width error is ~6 px of feet): a player right behind a teammate sat ~1.5 ft off or on the
  teammate's feet;
* two heads touching at an angle were one skin component, so the merged shirt was not split;
* a nearer shirt took a neighbour's head, leaving the neighbour headless and splitting a narrow
  duplicate box off the nearer player (a phantom player track).
"""

from __future__ import annotations

import numpy as np
import pytest

render_mod = pytest.importorskip("courtvision.sim.render")

from courtvision.geometry.camera import RECOMMENDED_SIM_CAMERA, PinholeCamera  # noqa: E402
from courtvision.geometry.court import court_for  # noqa: E402
from courtvision.perception.detector import ColorBlobDetector, DetectorConfig, FigureModel  # noqa: E402
from courtvision.types import BallState, FrameState, ObjClass, PlayerState, Team  # noqa: E402

FPS = 30.0
HEIGHT_FT, RADIUS_FT = 6.5, 1.0  # SimConfig.player_height_ft / player_radius_ft


@pytest.fixture(scope="module")
def scene():
    c = RECOMMENDED_SIM_CAMERA
    cam = PinholeCamera.look_at(c["camera_position"], c["camera_target"], c["image_size"], c["hfov_deg"])
    return cam, render_mod.SyntheticRenderer(cam, court_for("nba"))


def _frame(cam, people, ball=None) -> FrameState:
    players = [PlayerState(pid, team, x, y, jersey=n, bbox=cam.bbox_of_cylinder(x, y, HEIGHT_FT, RADIUS_FT))
               for pid, team, n, x, y in people]
    return FrameState(frame_idx=0, t=0.0, players=players, ball=ball or BallState(visible=False))


def _ball(cam, x, y, z) -> BallState:
    b = cam.bbox_of_sphere(x, y, z, 0.39)
    return BallState(visible=True, image_xy=((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2), court_xy=(x, y), z=z,
                     radius_px=(b.w + b.h) / 4)


def _warm(det: ColorBlobDetector, cam, renderer, frames: int = 12) -> None:
    """Isolated figures all over the court, so the detector learns the camera's figure geometry."""
    rng = np.random.default_rng(5)
    for i in range(frames):
        people: list = []
        while len(people) < 8:
            x, y = rng.uniform(6, 88), rng.uniform(3, 47)
            if all(abs(x - p[3]) > 7 or abs(y - p[4]) > 7 for p in people):
                people.append((f"w{len(people)}", Team.HOME if len(people) % 2 else Team.AWAY, str(len(people)), x, y))
        det.detect(renderer.render(_frame(cam, people)), i, i / FPS)


def _box_of(dets, truth_box):
    return max(dets, key=lambda d: d.bbox.iou(truth_box))


def test_leg_search_ignores_a_nearer_players_head(scene) -> None:
    """Seed 3 frame 789: #00 stands behind #14 near the far sideline and a nearer teammate's head sits
    just under #00's feet. That head was taken for #00's legs: box 13 px too low, ~3.7 ft nearer."""
    cam, renderer = scene
    people = [("a00", Team.AWAY, "00", 7.82, 48.43), ("h14", Team.HOME, "14", 7.53, 44.33),
              ("h3", Team.HOME, "3", 12.12, 31.36)]
    f = _frame(cam, people)
    img = renderer.render(f)
    far = f.players[0].bbox
    for cfg, ok in ((DetectorConfig(), True), (DetectorConfig(legs_skip_heads=False), False)):
        det = ColorBlobDetector(cfg)
        _warm(det, cam, renderer)
        d = _box_of(det.detect(img, 99, 3.3).of(ObjClass.PLAYER), far)
        assert d.bbox.iou(far) > 0.6
        assert (abs(d.bbox.y2 - far.y2) < 1.5) is ok, (cfg.legs_skip_heads, d.bbox, far)


def test_a_ball_cutting_a_shirt_leaves_one_box_with_its_feet(scene) -> None:
    """Seed 3 frame 2814: #3 (home) behind #20 (away) with the ball in front of #3's visible strip.
    The strip was cut in two: the head's box ended 32 px above the feet and a headless phantom box
    stood under the ball."""
    cam, renderer = scene
    people = [("h3", Team.HOME, "3", 61.46, 20.31), ("a20", Team.AWAY, "20", 60.55, 19.12)]
    f = _frame(cam, people, _ball(cam, 62.53, 19.97, 2.99))
    img = renderer.render(f)
    hidden = f.players[0].bbox
    for cfg, ok in ((DetectorConfig(), True), (DetectorConfig(bridge_ball=False), False)):
        det = ColorBlobDetector(cfg)
        _warm(det, cam, renderer)
        homes = [d for d in det.detect(img, 99, 3.3).of(ObjClass.PLAYER) if d.bbox.iou(hidden) > 0.05
                 and d.bbox.iou(f.players[1].bbox) < 0.5]
        d = _box_of(homes, hidden)
        assert (len(homes) == 1 and abs(d.bbox.y2 - hidden.y2) < 1.5 and abs(d.bbox.y1 - hidden.y1) < 1.5) is ok, (
            cfg.bridge_ball, [h.bbox for h in homes], hidden)


def test_feet_row_fallback_before_the_figure_model_has_its_spread(scene) -> None:
    """Head top -> feet row is affine for a fixed camera without roll: a few clean figures spanning
    image rows give it to well under a pixel long before the homography model trusts its spread."""
    cam, renderer = scene

    def sample(x: float, y: float):
        b = cam.bbox_of_cylinder(x, y, HEIGHT_FT, RADIUS_FT)
        return ((b.x1 + b.x2) / 2.0, b.y1), ((b.x1 + b.x2) / 2.0, b.y2), b.x2 - b.x1

    model = FigureModel(min_samples=40, refit_every=10, row_min_samples=10, row_min_span_px=40.0)
    for k in range(9):  # one row band only: no fallback either
        model.add(*sample(10.0 + 9 * k, 25.0))
    assert model.predict_feet_row((640.0, 300.0)) is None
    spots = [(20, 10), (70, 12), (25, 40), (65, 38), (45, 25)]
    for k in range(20):
        model.add(*sample(*spots[k % len(spots)]))
    assert not model.ready  # the homography still lacks spread ...
    for x, y in [(30, 25), (60, 8), (85, 45), (10, 30)]:  # ... the row fallback does not
        top, feet, _ = sample(x, y)
        row = model.predict_feet_row(top)
        assert row is not None and abs(row - feet[1]) < 1.0

    # In the detector: a player right behind a teammate on a fresh detector (a few frames of a stream)
    people = [("h33", Team.HOME, "33", 36.0, 22.2), ("h3", Team.HOME, "3", 36.3, 24.0),
              ("a7", Team.AWAY, "7", 60.0, 30.0), ("a15", Team.AWAY, "15", 78.0, 12.0),
              ("a24", Team.AWAY, "24", 15.0, 42.0)]
    f = _frame(cam, people)
    img = renderer.render(f)
    for cfg, ok in ((DetectorConfig(), True), (DetectorConfig(figure_row_min_samples=10**6), False)):
        det = ColorBlobDetector(cfg)
        for i in range(4):
            det.detect(img, i, i / FPS)
        assert det.figure is not None and not det.figure.ready
        dets = det.detect(img, 4, 4 / FPS).of(ObjClass.PLAYER)
        far = f.players[1].bbox
        d = _box_of(dets, far)
        assert (abs(d.bbox.y2 - far.y2) < 2.0) is ok, (d.bbox, far)


def _matches(dets, players):
    """(truth ids matched at IoU >= 0.5 one-to-one, unmatched boxes overlapping a player at IoU >= 0.3)."""
    from scipy.optimize import linear_sum_assignment

    iou = np.array([[p.bbox.iou(d.bbox) for d in dets] for p in players])
    rows, cols = linear_sum_assignment(-iou)
    pairs = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= 0.5]
    used = {c for _, c in pairs}
    dups = [c for c in range(len(dets)) if c not in used and iou[:, c].max() >= 0.3]
    return {players[r].player_id for r, _ in pairs}, dups


def test_heads_touching_at_an_angle_are_two_heads(scene) -> None:
    """Seed 2 frame 3440 (two teammates overlapping diagonally): touching head discs formed one skin
    component; the merged shirt kept one head, was not split, and the tracker swapped the two."""
    cam, renderer = scene
    f = _frame(cam, [("a40", Team.AWAY, "40", 40.0, 20.0), ("a42", Team.AWAY, "42", 40.6, 21.5)])
    img = renderer.render(f)
    for cfg, ok in ((DetectorConfig(), True), (DetectorConfig(head_disc_fill=None), False)):
        det = ColorBlobDetector(cfg)
        _warm(det, cam, renderer)
        dets = det.detect(img, 99, 3.3).of(ObjClass.PLAYER)
        good = all(max(d.bbox.iou(p.bbox) for d in dets) > 0.8 for p in f.players)
        assert (len(dets) == 2 and good) is ok, (cfg.head_disc_fill, [d.bbox for d in dets])


def test_a_contested_head_goes_to_the_headless_body(scene) -> None:
    """Seed 1 frame 639 (a crowd of six near the right block): a nearer shirt covering most of a
    neighbour's head bottom took that head, so the neighbour had none and the nearer blob was split
    into a narrow duplicate box. Over sim train seeds 1-3 the rule removed ~45 % of such duplicates
    for 0.03 % of recall."""
    cam, renderer = scene
    people = [("home-54", Team.HOME, "54", 68.84, 12.25), ("home-14", Team.HOME, "14", 77.31, 29.31),
              ("away-22", Team.AWAY, "22", 77.08, 41.01), ("away-3", Team.AWAY, "3", 75.72, 13.23),
              ("away-36", Team.AWAY, "36", 76.83, 33.43), ("away-7", Team.AWAY, "7", 78.56, 30.23)]
    f = _frame(cam, people)
    img = renderer.render(f, referees=[(65.21, -2.5), (74.21, 52.5), (96.0, 36.83)])
    found = {}
    for cfg in (DetectorConfig(), DetectorConfig(headless_takes_head=False)):
        det = ColorBlobDetector(cfg)
        _warm(det, cam, renderer)
        found[cfg.headless_takes_head] = _matches(det.detect(img, 99, 3.3).of(ObjClass.PLAYER), f.players)
    assert found[False][1] and not found[True][1], found  # the duplicate is gone ...
    assert found[True][0] >= found[False][0]  # ... and no player lost his box
