"""End to end on the simulator's own renders: sim -> SyntheticRenderer -> PerceptionPipeline.

Renders a 3 s window of live play ending 0.6 s after the first made shot (so the shot, the make and
the score change are in it), runs the pipeline with ColorBlobDetector and a Calibration built from
the sim camera (floor homography + rims), and checks player location, team labels, the ball and the
scoreboard against the sim truth.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

render_mod = pytest.importorskip("courtvision.sim.render")
sim_mod = pytest.importorskip("courtvision.sim.simulator")

from courtvision.geometry.homography import Calibration  # noqa: E402
from courtvision.perception.detector import ColorBlobDetector  # noqa: E402
from courtvision.perception.jersey import TemplateJerseyReader  # noqa: E402
from courtvision.perception.pipeline import PerceptionConfig, PerceptionPipeline  # noqa: E402
from courtvision.types import EventKind, FrameState, ScoreboardReading  # noqa: E402

SIM_S, SECONDS, AFTER_MAKE_S = 60.0, 3.0, 0.6
FALLBACK_START_S = 16.0  # live play in seed 7 if no make happens (sim tuning changes)
MATCH_FT = 4.0


@pytest.fixture(scope="module")
def e2e():
    cfg = sim_mod.SimConfig(seconds=SIM_S)
    result = sim_mod.GameSimulator(cfg).run()
    makes = [e.t for e in result.events if e.kind is EventKind.SHOT_MADE and e.t >= SECONDS + 3.0]
    start = makes[0] + AFTER_MAKE_S - SECONDS if makes else FALLBACK_START_S
    cam, court = result.camera, result.court
    renderer = render_mod.SyntheticRenderer(cam, court)
    rims = {r.side: r.bbox for r in result.frames[0].rims}
    calib = Calibration(homography=cam.floor_homography(), image_size=cam.image_size, league=cfg.league, rims=rims)
    pcfg = PerceptionConfig(scoreboard_roi=dict(render_mod.SCOREBOARD_ROI), foot_radius_ft=cfg.player_radius_ft)
    reader = TemplateJerseyReader(candidates=sorted({e.jersey for e in result.roster.entries}))
    pipe = PerceptionPipeline(ColorBlobDetector(), calib, roster=result.roster, config=pcfg, jersey_reader=reader)
    i0 = int(round(start * cfg.fps))
    out: list[tuple[FrameState, FrameState]] = []
    for i in range(i0, i0 + int(round(SECONDS * cfg.fps))):
        truth = result.frames[i]
        img = renderer.render(truth, referees=result.referees[i])
        out.append((pipe.process(img, truth.frame_idx, truth.t), truth))
    return out


def _pairs(pred: FrameState, truth: FrameState) -> list[tuple[float, bool, bool]]:
    """Hungarian match truth players to predictions (court ft, gated): (dist, team ok, id ok)."""
    if not pred.players:
        return []
    cost = np.array([[math.hypot(p.x - q.x, p.y - q.y) for p in pred.players] for q in truth.players])
    rows, cols = linear_sum_assignment(np.minimum(cost, 1e3))
    out = []
    for r, c in zip(rows, cols):
        if cost[r, c] <= MATCH_FT:
            q, p = truth.players[r], pred.players[c]
            out.append((float(cost[r, c]), p.team is q.team, p.player_id == q.player_id))
    return out


def _displayed(truth: ScoreboardReading) -> tuple:
    """What the panel shows for a truth reading, from the renderer's own cell characters."""
    cells = render_mod.scoreboard_digits(truth)

    def num(chars: list[str]) -> int | None:
        text = "".join(chars)
        return int(text) if text else None

    clock = cells["clock"]
    game = None if not "".join(clock) else 60 * int("".join(clock[:2]) or "0") + int("".join(clock[2:]))
    shot = num(cells["shot"])
    return (num(cells["home"]), num(cells["away"]), num(cells["period"]),
            None if game is None else float(game), None if shot is None else float(shot))


def test_sim_players_teams_ball_and_scoreboard(e2e) -> None:
    rows = [m for pred, truth in e2e for m in _pairs(pred, truth)]
    n_truth = sum(len(t.players) for _, t in e2e)
    dists = np.array([r[0] for r in rows])
    team_acc = float(np.mean([r[1] for r in rows]))
    late = [m for pred, truth in e2e[len(e2e) // 2:] for m in _pairs(pred, truth)]
    id_acc = float(np.mean([r[2] for r in late]))

    ball_err, ball_all = [], []
    for pred, truth in e2e:
        tb, pb = truth.ball, pred.ball
        if tb is None or not tb.visible or pb is None or not pb.visible or pb.image_xy is None:
            continue
        err = math.hypot(pb.image_xy[0] - tb.image_xy[0], pb.image_xy[1] - tb.image_xy[1])
        ball_all.append(err)
        if not pb.interpolated:
            ball_err.append(err)
    n_ball_truth = sum(1 for _, t in e2e if t.ball is not None and t.ball.visible)

    reads = [(pred.scoreboard, truth.scoreboard) for pred, truth in e2e if pred.scoreboard is not None]
    scores_seen = {(r.home_score, r.away_score) for r, _ in reads}
    exact = [
        (r.home_score, r.away_score, r.period, r.game_clock, r.shot_clock) == _displayed(tr) for r, tr in reads
    ]
    print(
        f"\nsim e2e ({len(e2e)} frames): players median {np.median(dists):.2f} ft, p90 {np.percentile(dists, 90):.2f} ft, "
        f"matched {len(rows)}/{n_truth}; team acc {team_acc:.3f}; roster-id acc (2nd half) {id_acc:.3f}; "
        f"ball observed {len(ball_err)}/{n_ball_truth} frames, median {np.median(ball_err):.2f} px, "
        f"p95 {np.percentile(ball_err, 95):.2f} px, max {max(ball_err):.2f} px "
        f"(incl. interpolated: max {max(ball_all):.2f} px); scoreboard {sum(exact)}/{len(reads)} exact, "
        f"scores read {sorted(scores_seen)}; window t={e2e[0][1].t:.2f}-{e2e[-1][1].t:.2f} s"
    )
    assert np.median(dists) < 1.5
    assert len(rows) > 0.9 * n_truth
    assert team_acc > 0.95
    # Spec: ball image error < 3 px when visible -> every observed frame, and the ball must actually
    # be observed in most frames where it is visible (otherwise the bound would hold vacuously).
    assert len(ball_err) >= 0.9 * n_ball_truth
    assert max(ball_err) < 3.0
    assert len(reads) == 15 and all(exact)
    for pred, _ in e2e:
        ids = [p.player_id for p in pred.players]
        assert len(ids) == len(set(ids))  # never one id on two players
