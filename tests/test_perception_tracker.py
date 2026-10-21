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



# Seed 2 frames 3790-3821 (video mode, detector boxes, team colour H/A): away #27 stands behind home #49
# and is hidden for ~20 frames while away #42 walks out from behind them (a too-tall merged box for 3
# frames); then everyone reappears. First/last-frame truth boxes of #27 and #42.
_CROWD = [
    [('H', (966, 284, 992, 344)), ('A', (982, 272, 1014, 332)), ('A', (980, 286, 1010, 348))],
    [('H', (966, 284, 992, 346)), ('A', (984, 274, 1014, 332)), ('A', (980, 286, 1012, 348))],
    [('H', (966, 286, 994, 346)), ('A', (984, 274, 1016, 332)), ('A', (982, 288, 1012, 350))],
    [('H', (968, 264, 1004, 346)), ('A', (982, 274, 1016, 350))],
    [('H', (970, 262, 1006, 344)), ('A', (982, 278, 1018, 350))],
    [('H', (970, 260, 1006, 342)), ('A', (984, 266, 1018, 352)), ('A', (932, 284, 962, 346))],
    [('H', (972, 258, 1008, 340)), ('A', (984, 266, 1020, 352)), ('A', (934, 286, 964, 346))],
    [('H', (972, 258, 1008, 338)), ('A', (984, 266, 1020, 352)), ('A', (938, 286, 968, 348))],
    [('H', (974, 256, 1010, 338)), ('A', (986, 266, 1020, 354)), ('A', (940, 286, 970, 348))],
    [('H', (974, 256, 1012, 336)), ('A', (986, 266, 1020, 354)), ('A', (942, 286, 972, 348))],
    [('H', (976, 254, 1012, 336)), ('A', (988, 266, 1020, 354)), ('A', (944, 286, 976, 348))],
    [('H', (976, 254, 1012, 336)), ('A', (988, 268, 1022, 354)), ('A', (946, 288, 978, 350))],
    [('H', (976, 254, 1014, 336)), ('A', (988, 266, 1022, 356)), ('A', (948, 288, 980, 350))],
    [('H', (978, 254, 1028, 336)), ('A', (950, 288, 982, 350)), ('A', (988, 294, 1022, 356))],
    [('H', (978, 254, 1026, 336)), ('A', (952, 288, 984, 350)), ('A', (990, 294, 1022, 356))],
    [('H', (980, 256, 1022, 338)), ('A', (954, 288, 984, 350)), ('A', (988, 294, 1024, 356))],
    [('H', (980, 256, 1020, 338)), ('A', (956, 288, 986, 350)), ('A', (994, 294, 1026, 356))],
    [('H', (982, 258, 1020, 340)), ('A', (956, 288, 988, 350)), ('A', (996, 294, 1028, 356))],
    [('H', (984, 260, 1020, 342)), ('A', (958, 290, 988, 352)), ('A', (998, 294, 1030, 354))],
    [('H', (984, 262, 1022, 344)), ('A', (958, 290, 990, 352)), ('A', (998, 294, 1032, 354))],
    [('H', (986, 264, 1022, 346)), ('A', (960, 290, 992, 352)), ('A', (1000, 294, 1034, 356))],
    [('H', (988, 266, 1020, 324)), ('A', (962, 290, 994, 352)), ('A', (1006, 272, 1060, 416))],
    [('H', (988, 270, 1020, 328)), ('A', (964, 290, 986, 352)), ('A', (1018, 272, 1058, 416))],
    [('H', (986, 272, 1018, 332)), ('A', (964, 290, 986, 354)), ('A', (1014, 296, 1058, 414))],
    [('H', (930, 286, 960, 348)), ('H', (986, 294, 1018, 356)), ('A', (980, 280, 1010, 340)),
     ('A', (966, 292, 998, 350)), ('A', (1004, 292, 1038, 354))],
    [('H', (932, 288, 962, 350)), ('H', (986, 294, 1020, 356)), ('A', (978, 282, 1010, 342)),
     ('A', (968, 292, 998, 352)), ('A', (1006, 292, 1038, 354))],
    [('H', (934, 288, 964, 350)), ('H', (986, 294, 1020, 356)), ('A', (976, 282, 1008, 340)),
     ('A', (1006, 292, 1040, 352)), ('A', (970, 292, 1000, 354))],
    [('H', (936, 290, 968, 352)), ('H', (986, 294, 1020, 356)), ('A', (974, 282, 1006, 342)),
     ('A', (970, 292, 1002, 354)), ('A', (1008, 292, 1040, 352))],
    [('H', (938, 290, 970, 352)), ('A', (972, 292, 1004, 354)), ('H', (986, 294, 1020, 356)),
     ('A', (972, 282, 1004, 348)), ('A', (1008, 292, 1042, 352))],
    [('H', (940, 292, 972, 354)), ('A', (972, 294, 1004, 356)), ('H', (986, 294, 1020, 356)),
     ('A', (970, 284, 1002, 348)), ('A', (1008, 290, 1042, 352))],
    [('H', (942, 292, 974, 354)), ('A', (974, 294, 1004, 356)), ('H', (986, 294, 1020, 356)),
     ('A', (968, 284, 1000, 346)), ('A', (1010, 290, 1042, 352))],
    [('H', (944, 292, 976, 354)), ('H', (1006, 286, 1020, 356)), ('A', (966, 284, 998, 344)),
     ('A', (1010, 290, 1042, 350)), ('H', (986, 294, 1020, 356)), ('A', (974, 294, 1004, 356))],
]
_CROWD_FIRST = {"27": (983, 273, 1014, 333), "42": (980, 287, 1012, 348), "49": (966, 284, 992, 344)}
_CROWD_LAST = {"27": (967, 285, 998, 347), "42": (1010, 290, 1043, 353)}


def _crowd_owners(config: TrackerConfig) -> tuple[dict[str, int], dict[str, int | None]]:
    """Track ids on the first-frame and last-frame boxes of #27, #42 (and #49 first). No embeddings:
    teammates' shirts look alike, so association rests on geometry."""
    trk = ByteTracker(config)
    owners = []
    for f, row in enumerate(_CROWD):
        out = [t for t in trk.update([Detection(bbox=BBox(*b), cls=ObjClass.PLAYER, score=0.95) for _, b in row], f)
               if t.time_since_update == 0]
        boxes = _CROWD_FIRST if f == 0 else _CROWD_LAST
        o: dict[str, int | None] = {}
        for name, b in boxes.items():
            t = max(out, key=lambda t: t.bbox.iou(BBox(*b)), default=None)
            o[name] = t.track_id if t is not None and t.bbox.iou(BBox(*b)) > 0.7 else None
        owners.append(o)
    return owners[0], owners[-1]  # type: ignore[return-value]


def test_foot_points_keep_a_reappearing_player_off_other_players_tracks() -> None:
    """Regression (round 2, perception; sim train seed 2): after an occlusion, IoU alone matched the
    reappearing #27's box to the home player's track and #42's to #27's (wrong same-team ids for ~3 s).
    The foot-point term refuses boxes whose feet are far from a track's predicted feet: #42 stays on
    his track and #27 restarts (the identity layer re-links or re-reads him) instead of swapping."""
    first, last = _crowd_owners(TrackerConfig())
    assert len(set(first.values())) == 3
    assert last["42"] == first["42"]
    assert last["27"] is not None and last["27"] not in (first["42"], first["49"])
    first, last = _crowd_owners(TrackerConfig(feet_weight=0.0))  # the old tracker: #27 lands on #49's track
    assert last["27"] == first["49"]
