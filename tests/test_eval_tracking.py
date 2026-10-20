"""Tests for courtvision.eval.tracking (synthetic, deterministic)."""

from __future__ import annotations

import copy
import json
import random

import numpy as np
import pytest

from courtvision.eval.tracking import TrackEvalConfig, tracking_report
from courtvision.types import BallState, FrameState, PlayerState, Team

N = 20  # frames


def truth_frames(n: int = N) -> list[FrameState]:
    """Three players moving right; far apart (> 2 gates) so matching is unambiguous."""
    frames = []
    for k in range(n):
        players = [
            PlayerState("home-1", Team.HOME, 10.0 + 0.5 * k, 10.0, track_id=1),
            PlayerState("home-2", Team.HOME, 10.0 + 0.5 * k, 30.0, track_id=2),
            PlayerState("away-5", Team.AWAY, 60.0, 25.0 + 0.2 * k, track_id=5),
        ]
        ball = BallState(visible=True, image_xy=(100.0 + k, 200.0), court_xy=(20.0, 20.0), z=5.0)
        frames.append(FrameState(frame_idx=k, t=k / 30.0, players=players, ball=ball))
    return frames


def test_perfect_tracking() -> None:
    truth = truth_frames()
    rep = tracking_report(copy.deepcopy(truth), truth)
    assert rep.gt == 3 * N and rep.tp == 3 * N and rep.fp == rep.fn == rep.idsw == 0
    assert rep.mota == 1.0 and rep.idf1 == 1.0 and rep.rmse_ft == 0.0
    assert rep.identity_accuracy == 1.0 and rep.team_accuracy == 1.0
    assert rep.id_map == {"home-1": "home-1", "home-2": "home-2", "away-5": "away-5"}
    assert rep.ball.detection_rate == 1.0 and rep.ball.px_mean == 0.0 and rep.ball.px_hit_rate == 1.0
    assert rep.ball.false_visible_rate is None
    json.dumps(rep.to_dict())


def test_id_switch_between_two_players() -> None:
    truth = truth_frames()
    pred = copy.deepcopy(truth)
    for f in pred[N // 2:]:  # identities swap halfway; positions stay right
        a, b = f.players[0], f.players[1]
        a.player_id, b.player_id = b.player_id, a.player_id
    rep = tracking_report(pred, truth)
    assert rep.idsw == 2
    assert rep.mota == pytest.approx(1 - 2 / (3 * N))
    # Each swapped truth id agrees with its mapped id for half the frames.
    assert rep.idtp == N + N // 2 + N // 2
    assert rep.idf1 == pytest.approx(2 * rep.idtp / (2 * rep.idtp + rep.idfp + rep.idfn))
    assert rep.idf1 == pytest.approx(2 / 3)
    assert rep.identity_accuracy == pytest.approx((N + N) / (3 * N))
    assert rep.rmse_ft == 0.0


def test_track_id_field_ignores_identity_relabels() -> None:
    truth = truth_frames()
    pred = copy.deepcopy(truth)
    for f in pred[N // 2:]:
        f.players[0].player_id, f.players[1].player_id = "home-2", "home-1"
    rep = tracking_report(pred, truth, config=TrackEvalConfig(pred_id_field="track_id"))
    assert rep.idsw == 0 and rep.idf1 == 1.0
    assert rep.identity_accuracy == pytest.approx(2 / 3)  # still judged on player_id


def test_missing_extra_and_out_of_gate() -> None:
    truth = truth_frames()
    pred = copy.deepcopy(truth)
    for f in pred:
        f.players = [p for p in f.players if p.player_id != "away-5"]  # never tracked: FN
        f.players.append(PlayerState("unk-t9", None, 90.0, 5.0))  # ghost: FP
        f.players[0].x += 3.5  # beyond the 3 ft gate: FN + FP
    rep = tracking_report(pred, truth)
    assert rep.gt == 3 * N and rep.tp == N
    assert rep.fn == 2 * N and rep.fp == 2 * N
    assert rep.mota == pytest.approx(1 - 4 * N / (3 * N))
    loose = tracking_report(pred, truth, match_radius_ft=4.0)
    assert loose.tp == 2 * N and loose.rmse_ft == pytest.approx(np.sqrt(3.5**2 / 2))


def test_position_rmse_and_team_accuracy() -> None:
    truth = truth_frames()
    pred = copy.deepcopy(truth)
    for f in pred:
        for p in f.players:
            p.x += 0.3
            p.y -= 0.4
        f.players[2].team = Team.HOME  # wrong team for away-5
    rep = tracking_report(pred, truth)
    assert rep.rmse_ft == pytest.approx(0.5) and rep.motp_ft == pytest.approx(0.5)
    assert rep.team_accuracy == pytest.approx(2 / 3) and rep.team_n == 3 * N


def test_previous_matches_are_kept_inside_the_gate() -> None:
    # Two truth players 2 ft apart; predictions jitter so that, in frame 1, the distance-
    # optimal assignment would swap ids. CLEAR-MOT keeps last frame's pairs: no switch.
    def frame(k: int, xa: float, xb: float, pa: float, pb: float) -> tuple[FrameState, FrameState]:
        t = FrameState(k, k / 30, [PlayerState("A", Team.HOME, xa, 10.0), PlayerState("B", Team.HOME, xb, 10.0)], None)
        p = FrameState(k, k / 30, [PlayerState("a", Team.HOME, pa, 10.0), PlayerState("b", Team.HOME, pb, 10.0)], None)
        return t, p

    pairs = [frame(0, 10.0, 12.0, 10.0, 12.0), frame(1, 10.0, 12.0, 11.6, 10.5)]
    truth, pred = [t for t, _ in pairs], [p for _, p in pairs]
    kept = tracking_report(pred, truth)
    assert kept.idsw == 0 and kept.tp == 4
    pure = tracking_report(pred, truth, config=TrackEvalConfig(keep_previous_matches=False))
    assert pure.idsw == 2


def test_frame_alignment_and_ball_metrics() -> None:
    truth = truth_frames()
    pred = copy.deepcopy(truth[::2])  # perception ran at half rate
    for k, f in enumerate(pred):
        if k % 5 == 0:
            f.ball = BallState(visible=False)
        elif f.ball is not None and f.ball.image_xy is not None:
            f.ball.image_xy = (f.ball.image_xy[0] + 3.0, f.ball.image_xy[1] + 4.0)
            f.ball.interpolated = k % 5 == 1
    truth[4].ball = BallState(visible=False)  # pred still reports a ball there
    rep = tracking_report(pred, truth)
    assert rep.n_frames == N // 2 and rep.n_unaligned_truth_frames == N // 2 and rep.mota == 1.0
    b = rep.ball
    assert b.truth_visible == N // 2 - 1 and b.false_visible_rate == 1.0
    assert b.detection_rate == pytest.approx(7 / 9)
    assert b.px_n == 7 and b.px_mean == pytest.approx(5.0) and b.px_hit_rate == 1.0
    assert b.interpolated_share == pytest.approx(2 / 8)
    by_t = tracking_report(pred, truth, config=TrackEvalConfig(align_by="t"))
    assert by_t.n_frames == N // 2 and by_t.tp == rep.tp


def test_nan_ball_point_does_not_poison_metrics_or_json() -> None:
    truth = truth_frames(4)
    pred = copy.deepcopy(truth)
    assert pred[1].ball is not None
    pred[1].ball.image_xy = (float("nan"), 200.0)
    pred[2].players[0].x = float("nan")  # failed projection: unmatched, not a crash
    rep = tracking_report(pred, truth)
    assert rep.ball.px_n == 3 and rep.ball.px_mean == 0.0
    assert rep.tp == 3 * 4 - 1 and rep.fn == 1 and rep.fp == 1
    json.dumps(rep.to_dict(), allow_nan=False)


def test_truth_frame_order_does_not_change_id_switches() -> None:
    truth = truth_frames()
    pred = copy.deepcopy(truth)
    for f in pred[N // 2:]:
        a, b = f.players[0], f.players[1]
        a.player_id, b.player_id = b.player_id, a.player_id
    ordered = tracking_report(pred, truth)
    order = list(range(N))
    random.Random(11).shuffle(order)  # e.g. frames merged from several workers
    shuffled = tracking_report([pred[k] for k in order], [truth[k] for k in order])
    assert ordered.to_dict() == shuffled.to_dict()
    assert shuffled.idsw == 2


def test_empty_string_pred_id_is_not_treated_as_a_previous_match() -> None:
    # Truth A has no previous match; a predicted id "" must not be "kept" for it.
    t0 = FrameState(0, 0.0, [PlayerState("A", Team.HOME, 10.0, 10.0), PlayerState("B", Team.HOME, 12.0, 10.0)], None)
    p0 = FrameState(0, 0.0, [PlayerState("", Team.HOME, 12.1, 10.0), PlayerState("x", Team.HOME, 10.0, 10.0)], None)
    rep = tracking_report([p0], [t0])
    assert rep.tp == 2 and rep.motp_ft == pytest.approx(0.05)


def test_empty_inputs() -> None:
    rep = tracking_report([], truth_frames(3))
    assert rep.n_frames == 0 and rep.mota is None and rep.idf1 is None and rep.ball.detection_rate is None
    json.dumps(rep.to_dict())
