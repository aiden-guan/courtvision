"""Same-team merges on the simulator's own renderer: figure model, stacked heads, ids through a merge.

Regressions for the perception identity root cause (sim train seeds 1-3, video mode): 37 % of player
boxes came from split merged blobs whose head-width proportions put the feet up to ~10 px off; two
teammates one right behind the other had their touching heads merged into one, so one box carried
both players and its jersey reads flipped its id; and a hidden teammate's established track was
deleted as a "stale duplicate", handing his detection to the merged track (identity swap).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

render_mod = pytest.importorskip("courtvision.sim.render")

from courtvision.events.engine import EventEngine, EventEngineConfig  # noqa: E402
from courtvision.geometry.camera import RECOMMENDED_SIM_CAMERA, PinholeCamera  # noqa: E402
from courtvision.geometry.court import court_for  # noqa: E402
from courtvision.geometry.homography import Calibration  # noqa: E402
from courtvision.perception.detector import ColorBlobDetector, DetectorConfig, FigureModel  # noqa: E402
from courtvision.perception.jersey import TemplateJerseyReader  # noqa: E402
from courtvision.perception.pipeline import PerceptionConfig, PerceptionPipeline  # noqa: E402
from courtvision.types import (  # noqa: E402
    BallState,
    EventKind,
    FrameState,
    ObjClass,
    PlayerState,
    Roster,
    RosterEntry,
    Team,
)

FPS = 30.0
HEIGHT_FT, RADIUS_FT = 6.5, 1.0  # SimConfig.player_height_ft / player_radius_ft


@pytest.fixture(scope="module")
def scene():
    c = RECOMMENDED_SIM_CAMERA
    cam = PinholeCamera.look_at(c["camera_position"], c["camera_target"], c["image_size"], c["hfov_deg"])
    court = court_for("nba")
    return cam, court, render_mod.SyntheticRenderer(cam, court)


def _frame(cam, people, i=0, ball=None) -> FrameState:
    """people: (player_id, team, jersey, x, y) -> a FrameState the sim renderer draws."""
    players = [PlayerState(pid, team, x, y, jersey=n, bbox=cam.bbox_of_cylinder(x, y, HEIGHT_FT, RADIUS_FT))
               for pid, team, n, x, y in people]
    return FrameState(frame_idx=i, t=i / FPS, players=players, ball=ball or BallState(visible=False))


def _warm(det: ColorBlobDetector, cam, renderer, frames: int = 12) -> None:
    """Isolated figures all over the court, so the detector learns the camera's figure geometry."""
    rng = np.random.default_rng(5)
    for i in range(frames):
        people: list = []
        while len(people) < 8:
            x, y = rng.uniform(6, 88), rng.uniform(3, 47)
            if all(abs(x - p[3]) > 7 or abs(y - p[4]) > 7 for p in people):
                team = Team.HOME if len(people) % 2 else Team.AWAY
                people.append((f"w{len(people)}", team, str(len(people)), x, y))
        det.detect(renderer.render(_frame(cam, people, i)), i, i / FPS)


def test_figure_model_learns_head_to_feet_geometry_and_needs_spread(scene) -> None:
    cam, _, _ = scene

    def sample(x: float, y: float):
        b = cam.bbox_of_cylinder(x, y, HEIGHT_FT, RADIUS_FT)
        return ((b.x1 + b.x2) / 2.0, b.y1), ((b.x1 + b.x2) / 2.0, b.y2), b.x2 - b.x1

    model = FigureModel(min_samples=40, refit_every=10)
    for k in range(60):  # four spots only: an exact but untrustworthy fit
        model.add(*sample(*[(20, 10), (70, 10), (20, 40), (70, 40)][k % 4]))
    assert not model.ready and model.predict((640.0, 300.0)) is None
    rng = np.random.default_rng(3)
    for i in range(150):
        top, feet, width = sample(rng.uniform(5, 89), rng.uniform(2, 48))
        if i % 10 == 0:  # a jumping player: feet off the floor, must not bend the fit
            feet = (feet[0], feet[1] - 15.0)
        model.add(top, feet, width)
    assert model.ready and model.residual_px is not None and model.residual_px < 1.0
    for x, y in [(30, 25), (60, 8), (85, 45), (10, 30)]:
        top, feet, width = sample(x, y)
        pred = model.predict(top)
        assert pred is not None
        assert abs(pred[0] - feet[1]) < 2.0 and abs(pred[1] - width) < 2.5


def test_split_boxes_take_their_feet_from_the_figure_model(scene) -> None:
    cam, _, renderer = scene
    det = ColorBlobDetector()
    _warm(det, cam, renderer)
    assert det.figure is not None and det.figure.ready
    pairs = [(40.0, 20.0, 41.2, 24.0), (60.0, 15.0, 61.0, 19.5), (30.0, 30.0, 31.5, 34.0)]
    for x0, y0, x1, y1 in pairs:
        f = _frame(cam, [("a", Team.HOME, "3", x0, y0), ("b", Team.HOME, "8", x1, y1)])
        assert f.players[0].bbox.iou(f.players[1].bbox) > 0.05  # really one merged blob
        dets = det.detect(renderer.render(f), 99, 3.3).of(ObjClass.PLAYER)
        assert len(dets) == 2
        for p in f.players:
            d = max(dets, key=lambda d: d.bbox.iou(p.bbox))
            assert d.bbox.iou(p.bbox) > 0.85
            # the far player's feet are hidden: head-width proportions put them 3.5-7 px off here
            assert abs(d.bbox.y2 - p.bbox.y2) < 1.5


def test_player_right_behind_a_teammate_is_split_at_the_stacked_heads(scene) -> None:
    cam, _, renderer = scene
    det = ColorBlobDetector()
    _warm(det, cam, renderer)
    for dx in (0.0, 0.15, 0.3, 0.6):  # touching heads: one tall skin component, or two offset discs
        f = _frame(cam, [("near", Team.HOME, "33", 36.0, 22.2), ("far", Team.HOME, "3", 36.0 + dx, 24.0)])
        dets = det.detect(renderer.render(f), 99, 3.3).of(ObjClass.PLAYER)
        assert len(dets) == 2, dx
        for p in f.players:
            assert max(d.bbox.iou(p.bbox) for d in dets) > 0.8, dx
    off = ColorBlobDetector(DetectorConfig(stacked_head_ratio=None))
    f = _frame(cam, [("near", Team.HOME, "33", 36.0, 22.2), ("far", Team.HOME, "3", 36.3, 24.0)])
    assert len(off.detect(renderer.render(f), 0, 0.0).of(ObjClass.PLAYER)) == 1  # the old merge


ROSTER = Roster(entries=[RosterEntry("h3", Team.HOME, "3", "Avery"), RosterEntry("h33", Team.HOME, "33", "Blake"),
                         RosterEntry("h8", Team.HOME, "8", "Casey"), RosterEntry("a7", Team.AWAY, "7", "Drew"),
                         RosterEntry("a15", Team.AWAY, "15", "Emery"), RosterEntry("a24", Team.AWAY, "24", "Finley")])


def _walk_together(t: float) -> list[tuple[str, Team, str, float, float]]:
    """h33 dribbles; h3 walks right behind him (heads touching in the image) for 2 s, then they part."""
    x = 26.0 + 7.0 * t
    sep = 0.0 if t < 2.0 else min(1.0, t - 2.0)
    return [("h33", Team.HOME, "33", x, 22.2 - 2.0 * sep), ("h3", Team.HOME, "3", x + 0.3, 24.0 + 4.0 * sep),
            ("h8", Team.HOME, "8", 70.0, 12.0 + 2.0 * math.sin(t)), ("a7", Team.AWAY, "7", 60.0, 30.0 + 2.0 * math.cos(t)),
            ("a15", Team.AWAY, "15", 78.0, 20.0), ("a24", Team.AWAY, "24", 15.0, 40.0)]


def test_teammates_one_behind_the_other_keep_their_ids_and_no_phantom_pass(scene) -> None:
    cam, court, renderer = scene
    calib = Calibration(homography=cam.floor_homography(), image_size=cam.image_size, league="nba")
    pipe = PerceptionPipeline(ColorBlobDetector(), calib, roster=ROSTER, config=PerceptionConfig(foot_radius_ft=RADIUS_FT),
                              jersey_reader=TemplateJerseyReader(candidates=sorted({e.jersey for e in ROSTER.entries})))
    engine = EventEngine(court, ROSTER, EventEngineConfig(assume_calibrated=True))
    n, events = int(4.5 * FPS), []
    matched = {"h33": 0, "h3": 0}
    swapped, final = 0, {}
    for i in range(n):
        t = i / FPS
        people = _walk_together(t)
        hx, hy = people[0][3], people[0][4]
        z = 1.0 + 2.0 * abs(math.sin(math.pi * t / 0.5))  # dribble on h33's open side
        b = cam.bbox_of_sphere(hx - 0.9, hy - 1.2, z, 0.39)
        ball = BallState(visible=True, image_xy=((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2), court_xy=(hx - 0.9, hy - 1.2),
                         z=z, radius_px=(b.w + b.h) / 4)
        truth = _frame(cam, people, i, ball)
        state = pipe.process(renderer.render(truth), i, t)
        events += engine.process(state)
        pair = truth.players[:2]
        if not state.players:
            continue
        cost = np.array([[math.hypot(p.x - q.x, p.y - q.y) for p in state.players] for q in pair])
        for r, c in zip(*linear_sum_assignment(np.minimum(cost, 1e3))):
            if cost[r, c] > 3.0:
                continue
            q, p = pair[r], state.players[c]
            matched[q.player_id] += 1
            swapped += int(p.player_id in ("h3", "h33") and p.player_id != q.player_id)
            if t > 4.0:
                final.setdefault(q.player_id, set()).add(p.player_id)
    # Both teammates are tracked through the walk (the old detector merged them into one box for 2 s,
    # and that box's reads flipped its id), never under each other's id, and identified at the end.
    assert min(matched.values()) >= 0.95 * n, matched
    assert swapped == 0
    assert final == {"h33": {"h33"}, "h3": {"h3"}}
    assert not [e for e in events if e.kind is EventKind.PASS]
