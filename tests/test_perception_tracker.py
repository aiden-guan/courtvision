"""ByteTracker: id persistence through crossings and short occlusions, two-stage association."""

from __future__ import annotations

import numpy as np

from courtvision.perception.appearance import color_feature
from courtvision.perception.tracker import ByteTracker, KalmanBoxFilter, TrackerConfig, iou_matrix
from courtvision.sim import palette
from courtvision.types import BBox, Detection, ObjClass

HOME_F = color_feature(palette.HOME_BGR)
AWAY_F = color_feature(palette.AWAY_BGR)


def box(cx: float, cy: float, w: float = 24.0, h: float = 70.0) -> BBox:
    return BBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def det(cx: float, cy: float, score: float = 0.9, emb: np.ndarray | None = None,
        cls: ObjClass = ObjClass.PLAYER, h: float = 70.0) -> Detection:
    return Detection(bbox=box(cx, cy, h=h, w=h * 0.34), cls=cls, score=score, embedding=emb)


def ids_by_side(tracks, x_mid: float) -> tuple[int, int]:
    left = min(tracks, key=lambda t: t.bbox.center[0])
    right = max(tracks, key=lambda t: t.bbox.center[0])
    return left.track_id, right.track_id


def test_ids_persist_through_crossing_with_appearance() -> None:
    rng = np.random.default_rng(0)
    trk = ByteTracker()
    first: dict[int, str] = {}
    for f in range(60):
        a = 300 + 6 * f + rng.normal(0, 0.5)  # moves right
        b = 660 - 6 * f + rng.normal(0, 0.5)  # moves left; they overlap around f = 30
        tracks = trk.update([det(a, 400, emb=HOME_F), det(b, 402, emb=AWAY_F)], f)
        for t in tracks:
            if t.time_since_update:
                continue
            side = "home" if np.allclose(t.embedding, HOME_F, atol=1e-3) else "away"
            first.setdefault(t.track_id, side)
            assert first[t.track_id] == side, f"id {t.track_id} switched at frame {f}"
    assert sorted(first) == [1, 2]


def test_ids_persist_through_crossing_by_motion_only() -> None:
    trk = ByteTracker(TrackerConfig(appearance_weight=0.0))
    traj: dict[int, list[float]] = {}
    for f in range(60):
        near = det(300 + 6 * f, 450, h=90)  # nearer player (bigger, lower)
        far = det(660 - 6 * f, 380, h=60)
        for t in trk.update([near, far], f):
            if t.time_since_update == 0:
                traj.setdefault(t.track_id, []).append(t.bbox.h)
    assert len(traj) == 2
    for heights in traj.values():
        assert max(heights) - min(heights) < 1.0  # each id kept its own (constant-height) target


def test_short_occlusion_keeps_id_and_long_gap_starts_new_id() -> None:
    trk = ByteTracker(TrackerConfig(track_buffer=30, emit_lost_frames=5))
    seen_lost = []
    for f in range(80):
        x = 200 + 4 * f
        dets = [] if 20 <= f < 28 else [det(x, 300)]  # 8-frame occlusion
        out = trk.update(dets, f)
        if 20 <= f < 28:
            seen_lost += [t.time_since_update for t in out]
        else:
            assert [t.track_id for t in out] == [1], f
    assert seen_lost == [1, 2, 3, 4, 5]  # coasting tracks are reported for 5 frames only
    for f in range(80, 130):  # gone for 50 frames > buffer
        trk.update([], f)
    for f in range(130, 133):
        out = trk.update([det(200 + 4 * f, 300)], f)
    assert [t.track_id for t in out] == [2]


def test_tentative_tracks_need_min_hits() -> None:
    trk = ByteTracker(TrackerConfig(min_hits=2))
    assert [t.track_id for t in trk.update([det(50, 300)], 0)] == [1]  # first frame: confirmed at once
    assert [t.track_id for t in trk.update([det(52, 300), det(100, 100)], 1)] == [1]  # new one is tentative
    assert [t.track_id for t in trk.update([det(54, 300)], 2)] == [1]  # one-frame blip dropped
    assert [t.track_id for t in trk.update([det(56, 300), det(400, 100)], 3)] == [1]
    out = trk.update([det(58, 300), det(401, 100)], 4)
    assert [t.track_id for t in out] == [1, 2] and out[1].hits == 2
    late = ByteTracker(TrackerConfig(instant_first_frame_activation=False))
    assert late.update([det(50, 300)], 0) == []


def test_low_score_detections_keep_tracked_track_alive() -> None:
    cfg = TrackerConfig(emit_lost_frames=0)
    trk = ByteTracker(cfg)
    for f in range(5):
        trk.update([det(100 + 3 * f, 200)], f)
    for f in range(5, 10):  # occluded: weak detections only (stage 2)
        out = trk.update([det(100 + 3 * f, 200, score=0.3)], f)
        assert [t.track_id for t in out] == [1] and out[0].time_since_update == 0
    # Low-score detections never start tracks.
    trk2 = ByteTracker()
    for f in range(5):
        assert trk2.update([det(100, 200, score=0.3)], f) == []


def test_classes_do_not_match_each_other() -> None:
    trk = ByteTracker()
    for f in range(4):
        trk.update([det(100, 200)], f)
    out = trk.update([det(100, 200, cls=ObjClass.REFEREE)], 4)
    player = [t for t in out if t.cls is ObjClass.PLAYER]
    assert player and player[0].time_since_update == 1
    for f in range(5, 8):  # balls are not tracked here
        out = trk.update([det(5, 5, cls=ObjClass.BALL)], f)
    assert all(t.cls is not ObjClass.BALL for t in out)


def test_frame_skipping_and_determinism() -> None:
    def run() -> list[tuple[int, int, tuple[float, ...]]]:
        trk = ByteTracker()
        rows = []
        for f in range(0, 60, 2):  # 15 Hz updates on a 30 Hz index
            for t in trk.update([det(100 + 5 * f, 300), det(900 - 5 * f, 320, h=60)], f):
                rows.append((f, t.track_id, tuple(round(v, 6) for v in t.bbox.as_tuple())))
        return rows

    a, b = run(), run()
    assert a == b
    assert {r[1] for r in a} == {1, 2}


def test_kalman_predicts_constant_velocity_and_iou_matrix() -> None:
    kf = KalmanBoxFilter()
    mean, cov = kf.initiate(np.array([100.0, 100.0, 0.5, 50.0]))
    for k in range(1, 10):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, np.array([100.0 + 4 * k, 100.0, 0.5, 50.0]))
    pred, _ = kf.predict(mean, cov, dt=2.0)
    assert abs(pred[0] - (100 + 4 * 11)) < 1.5
    m = iou_matrix(np.array([[0, 0, 10, 10], [5, 5, 15, 15]]), np.array([[0, 0, 10, 10]]))
    assert m.shape == (2, 1) and m[0, 0] == 1.0 and abs(m[1, 0] - 25 / 175) < 1e-9


def test_remove_deletes_a_confirmed_track() -> None:
    trk = ByteTracker()
    for f in range(3):
        trk.update([det(100 + 2 * f, 200), det(400, 200)], f)
    assert trk.remove(1) and not trk.remove(99)
    out = trk.update([det(400, 200)], 3)
    assert [t.track_id for t in out] == [2]
