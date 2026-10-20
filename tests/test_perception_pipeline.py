"""PerceptionPipeline end to end on locally drawn synthetic frames (independent of sim/)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from test_perception_synth import (
    PLAYER_RADIUS_FT,
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

from courtvision.geometry.homography import Calibration, Homography
from courtvision.perception.detector import ColorBlobDetector
from courtvision.perception.jersey import TemplateJerseyReader
from courtvision.perception.pipeline import PerceptionConfig, PerceptionPipeline
from courtvision.types import FrameState, ObjClass, Roster, RosterEntry, Team

FPS = 30.0
N_FRAMES = 90
NUMBERS = {Team.HOME: ["3", "11", "23", "8", "40"], Team.AWAY: ["0", "7", "15", "32", "24"]}
ROSTER = Roster(entries=[RosterEntry(f"{t.value[0]}{n}", t, n, f"{t.value} {n}") for t, ns in NUMBERS.items() for n in ns])


def player_paths(t: float) -> list[tuple[str, Team, str, float, float]]:
    """(player_id, team, number, x, y) at time t: smooth, seeded paths with team crossings."""
    rng = np.random.default_rng(11)
    out = []
    for team, numbers in NUMBERS.items():
        for k, n in enumerate(numbers):
            x0 = 20 + 13 * k + (4 if team is Team.AWAY else 0)
            y0 = 12 + 26 * rng.random()
            ax, ay, w, ph = 6 + 4 * rng.random(), 4 + 4 * rng.random(), 1.2 + rng.random(), 6.28 * rng.random()
            sgn = 1 if team is Team.HOME else -1
            out.append((f"{team.value[0]}{n}", team, n, x0 + sgn * ax * math.sin(w * t + ph), y0 + ay * math.cos(w * t + ph)))
    return out


def ball_xyz(t: float, players: list[tuple[str, Team, str, float, float]]) -> tuple[float, float, float]:
    """Held by h3 (dribbling) until 1.2 s, an arc pass to a7 until 2.0 s, then held by a7."""
    pos = {pid: (x, y) for pid, _, _, x, y in players}
    if t < 1.2:
        x, y = pos["h3"]
        bounce = abs(math.sin(math.pi * t / 0.4))  # dribble: floor contact every 0.4 s
        return x + 1.2, y - 1.0, 0.4 + 2.6 * bounce
    if t < 2.0:
        s = (t - 1.2) / 0.8
        (x0, y0), (x1, y1) = (pos["h3"][0] + 1.2, pos["h3"][1] - 1.0), (pos["a7"][0] + 1.2, pos["a7"][1] - 1.0)
        return x0 + s * (x1 - x0), y0 + s * (y1 - y0), 3.0 + 10.0 * s * (1 - s)
    x, y = pos["a7"]
    return x + 1.2, y - 1.0, 3.5


@pytest.fixture(scope="module")
def run_pipeline():
    cam, court = make_camera(), make_court()
    base = draw_court(cam, court)
    calib = Calibration(homography=cam.floor_homography(), image_size=cam.image_size, league="nba",
                        rims={s: rim_bbox(cam, court, s) for s in ("left", "right")})
    cfg = PerceptionConfig(scoreboard_roi=SCOREBOARD_ROI, foot_radius_ft=PLAYER_RADIUS_FT)
    roster_numbers = sorted({e.jersey for e in ROSTER.entries})
    pipe = PerceptionPipeline(ColorBlobDetector(), calib, roster=ROSTER, config=cfg,
                              jersey_reader=TemplateJerseyReader(candidates=roster_numbers))
    frames: list[tuple[FrameState, list, tuple | None, dict]] = []
    for f in range(N_FRAMES):
        t = f / FPS
        players = player_paths(t)
        refs = [SynthPlayer(30 + 8 * t, -2.5, None), SynthPlayer(70 - 6 * t, 52.0, None)]
        people = [SynthPlayer(x, y, team, n) for _, team, n, x, y in players] + refs
        bxyz = ball_xyz(t, players)
        img, _, ball_uv = render(cam, court, people, ball_xyz=bxyz, base=base)
        clock = 9 * 60 + 41 - int(t)
        board = {"home": 52, "away": 49 + (2 if t >= 2.0 else 0), "period": 3, "clock": clock, "shot": 21 - int(t)}
        draw_scoreboard(img, SCOREBOARD_ROI, scoreboard_text(board["home"], board["away"], 3, clock, board["shot"]))
        frames.append((pipe.process(img, f, t), players, ball_uv, board))
    return pipe, frames, court


def _match(state: FrameState, players) -> list[tuple[float, bool, bool]]:
    """(distance ft, team correct, id correct) for each truth player matched to the nearest state."""
    out = []
    for pid, team, _, x, y in players:
        if not state.players:
            continue
        p = min(state.players, key=lambda s: math.hypot(s.x - x, s.y - y))
        d = math.hypot(p.x - x, p.y - y)
        if d < 4.0:
            out.append((d, p.team is team, p.player_id == pid))
    return out


def test_players_located_teams_and_identities(run_pipeline) -> None:
    _, frames, court = run_pipeline
    rows = [m for state, players, _, _ in frames for m in _match(state, players)]
    dists = np.array([r[0] for r in rows])
    team_acc = np.mean([r[1] for r in rows])
    late = [m for state, players, _, _ in frames[45:] for m in _match(state, players)]
    id_acc = np.mean([r[2] for r in late])
    coverage = len(rows) / (10 * N_FRAMES)
    print(f"\nlocal synthetic e2e: median {np.median(dists):.2f} ft, p90 {np.percentile(dists, 90):.2f} ft, "
          f"coverage {coverage:.3f}, team acc {team_acc:.3f}, roster-id acc (last 1.5 s) {id_acc:.3f}")
    assert np.median(dists) < 1.0
    assert coverage > 0.97
    assert team_acc > 0.97
    assert id_acc > 0.95
    for state, *_ in frames:
        assert all(p.team is not None or p.player_id.startswith("unk") for p in state.players)
        assert all(court.in_bounds(p.x, p.y, margin=6.0) for p in state.players)
        assert all(p.speed <= 30.0 + 1e-9 for p in state.players)
        ids = [p.player_id for p in state.players]
        assert len(ids) == len(set(ids))  # never one id on two players


def test_ball_rims_and_scoreboard(run_pipeline) -> None:
    pipe, frames, _ = run_pipeline
    errs = [math.hypot(s.ball.image_xy[0] - uv[0], s.ball.image_xy[1] - uv[1])
            for s, _, uv, _ in frames if s.ball is not None and s.ball.visible and not s.ball.interpolated]
    print(f"ball: {len(errs)}/{N_FRAMES} observed frames, median {np.median(errs):.2f} px, max {max(errs):.2f} px")
    assert len(errs) > 0.9 * N_FRAMES and max(errs) < 3.0
    assert all(s.ball.court_xy is not None for s, *_ in frames if s.ball and s.ball.visible)
    sides = [[r.side for r in s.rims] for s, *_ in frames]
    assert all(sd == ["left", "right"] for sd in sides)
    reads = [(s, board) for s, _, _, board in frames if s.scoreboard is not None]
    assert len(reads) == 15  # 5 Hz over 3 s
    for s, board in reads:
        r = s.scoreboard
        assert (r.home_score, r.away_score, r.period, r.game_clock, r.shot_clock) == (
            board["home"], board["away"], 3, float(board["clock"]), float(board["shot"]))
        assert r.source == "ocr" and r.t == s.t
    assert pipe.teams.is_fitted


def test_detector_rims_side_assignment_and_config_validation() -> None:
    cam, court = make_camera(), make_court()
    calib = Calibration(homography=cam.floor_homography(), image_size=cam.image_size)
    pipe = PerceptionPipeline(ColorBlobDetector(), calib, config=PerceptionConfig(rim_source="detector"))
    img, _, _ = render(cam, court, [SynthPlayer(47, 25, Team.HOME, "1")])
    state = pipe.process(img, 0, 0.0)
    assert [r.side for r in state.rims] == ["left", "right"]
    for r, side in zip(state.rims, ("left", "right")):
        assert r.bbox.iou(rim_bbox(cam, court, side).expand(0.2)) > 0.5
    assert len(state.players) == 1 and state.players[0].player_id == "home-t1"
    with pytest.raises(ValueError, match="unknown TrackerConfig"):
        PerceptionPipeline(ColorBlobDetector(), calib, config=PerceptionConfig(tracker={"bogus": 1}))
    with pytest.raises(ValueError):
        PerceptionPipeline(ColorBlobDetector(), calib, config=PerceptionConfig(tracker_backend="nope"))


def test_players_far_outside_the_court_are_dropped() -> None:
    cam, court = make_camera(), make_court()
    calib = Calibration(homography=cam.floor_homography(), image_size=cam.image_size)
    pipe = PerceptionPipeline(ColorBlobDetector(), calib)
    people = [SynthPlayer(47, 25, Team.HOME, "1"), SynthPlayer(47, -8.0, Team.AWAY, "2"), SynthPlayer(-4.0, 30, Team.AWAY, "3")]
    img, _, _ = render(cam, court, people)
    state = pipe.process(img, 0, 0.0)
    xs = sorted(p.x for p in state.players)
    assert len(xs) == 2  # 4 ft beyond the baseline is kept; 8 ft beyond the sideline is not
    assert abs(xs[0] + 4.0) < 1.5 and abs(xs[1] - 47.0) < 1.5
    assert pipe.detector.detect(img, 0, 0.0).of(ObjClass.PLAYER).__len__() == 3


def test_negated_homography_still_locates_players_and_ball() -> None:
    """Regression: H and -H are the same mapping (the contract allows any scale), but front-only
    mapping assumed w > 0 on the floor, so a negated calibration produced no players at all."""
    cam, court = make_camera(), make_court()
    people = [SynthPlayer(47, 25, Team.HOME, "1"), SynthPlayer(20, 10, Team.AWAY, "2")]
    img, _, _ = render(cam, court, people, ball_xyz=(30.0, 20.0, 0.4))
    H = cam.floor_homography()
    states = []
    for hom in (H, Homography(-H.H)):
        pipe = PerceptionPipeline(ColorBlobDetector(), Calibration(homography=hom, image_size=cam.image_size))
        states.append(pipe.process(img, 0, 0.0))
    ref, neg = states
    assert len(neg.players) == len(ref.players) == 2
    for a, b in zip(ref.players, neg.players):
        assert (a.x, a.y) == pytest.approx((b.x, b.y))
    assert neg.ball.court_xy == pytest.approx(ref.ball.court_xy)


def test_scoreboard_roi_from_the_server_body_and_replacement_unmasks() -> None:
    """Regression: (1) the POST /api/scoreboard_roi body (image-pixel boxes per named cell) used to
    raise TypeError; (2) each new ROI was added to the detector mask and the old region stayed
    hidden, so a player standing where a mistaken ROI had been was never detected again."""
    cam, court = make_camera(), make_court()
    calib = Calibration(homography=cam.floor_homography(), image_size=cam.image_size)
    pipe = PerceptionPipeline(ColorBlobDetector(), calib)
    player = SynthPlayer(47.0, 25.0, Team.HOME, "1")
    img, (box,), _ = render(cam, court, [player])
    draw_scoreboard(img, SCOREBOARD_ROI, scoreboard_text(61, 58, 4, 75, 9))
    x0, y0 = SCOREBOARD_ROI["corners"][0]
    names = {"home": ["home_hundreds", "home_tens", "home_ones"], "away": ["away_hundreds", "away_tens", "away_ones"],
             "period": ["period"], "clock": ["clock_min_tens", "clock_min_ones", "clock_sec_tens", "clock_sec_ones"],
             "shot": ["shot_tens", "shot_ones"]}
    cells = {n: [x0 + x, y0 + y, x0 + x + w, y0 + y + h]
             for g, boxes in SCOREBOARD_ROI["cells"].items() for n, (x, y, w, h) in zip(names[g], boxes)}
    wrong = [[box.x1 - 5, box.y1 - 5], [box.x2 + 5, box.y1 - 5], [box.x2 + 5, box.y2 + 5], [box.x1 - 5, box.y2 + 5]]
    pipe.set_scoreboard_roi({"corners": wrong, "cells": {"home_ones": [box.x1, box.y1, box.x2, box.y2]},
                             "space": "image"})  # operator first boxes the wrong region
    assert pipe.process(img, 0, 0.0).players == []  # masked
    pipe.set_scoreboard_roi({"corners": SCOREBOARD_ROI["corners"], "cells": cells, "space": "image"})
    state = pipe.process(img, 1, 1 / FPS)
    assert len(pipe.process(img, 2, 2 / FPS).players) == 1  # old region unmasked (tentative -> confirmed)
    r = state.scoreboard
    assert r is not None and (r.home_score, r.away_score, r.period, r.game_clock, r.shot_clock) == (61, 58, 4, 75.0, 9.0)
    with pytest.raises(ValueError):
        pipe.set_scoreboard_roi({"corners": SCOREBOARD_ROI["corners"], "cells": {"bogus": [0, 0, 1, 1]}})
    assert pipe.process(img, 7, 7 / FPS).scoreboard is not None  # a rejected ROI leaves the reader as it was
    pipe.set_scoreboard_roi(None)
    assert pipe.process(img, 13, 13 / FPS).scoreboard is None
    assert pipe.detector._polygons == []  # removing the ROI removes its mask
