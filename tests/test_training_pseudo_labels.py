"""Self-training data for the student detector: pseudo-label frame selection and folders
(training/pseudo_labels.py), how train_student reads them, and teacher ball labels (training/ball_labels.py)."""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from courtvision.nba.analyze import BallTrack, CameraTrack, DetectionCache
from courtvision.perception.ball3d import Flight3D
from courtvision.perception.student import STRIDE
from courtvision.training import ball_labels, pseudo_labels
from courtvision.training.pseudo_labels import (
    Candidate,
    FrameGate,
    PseudoConfig,
    flight_candidates,
    path_candidates,
    scorebug_live,
    select_targets,
)
from courtvision.training.train_student import (
    PSEUDO_MAX_WEIGHT,
    ClipDataset,
    folder_weights,
    parse_data,
    plan_data,
)

C = np.array([47.0, -94.0, 36.0])
SIZE = (640, 360)  # a small "full-resolution" video: thresholds in 1080-line px scale by 1/3
FPS = 30.0
FOCAL, TILT = 1000.0, 16.8


def cam_track(frames: range, pan: Callable[[int], float] = lambda f: -5.0) -> CameraTrack:
    return CameraTrack(center=C, image_size=SIZE, fps=FPS, params={f: (pan(f), TILT, 0.0, FOCAL) for f in frames})


def project(track: CameraTrack, f: int, X: np.ndarray) -> tuple[float, float]:
    cam = track.get(f)
    assert cam is not None
    pc = cam.R() @ (np.asarray(X, float) - C)
    return (float(FOCAL * pc[0] / pc[2] + SIZE[0] / 2), float(FOCAL * pc[1] / pc[2] + SIZE[1] / 2))


def shot(f0: int = 30, f1: int = 60, rms: float = 0.3) -> Flight3D:
    return Flight3D(f0 / FPS, f1 / FPS, np.array([40.0, 25.0, 8.0]), np.array([12.0, 0.0, 14.0]), rms, f1 - f0 + 1,
                    (f0, f1))


def everywhere(f: int) -> bool:
    return True


# --------------------------------------------------------------------------- which frames may carry a label


def test_frame_gate_keeps_away_from_cuts_and_camera_run_ends_and_needs_the_scorebug(tmp_path: Path) -> None:
    frames = [*range(200), *range(230, 400)]  # the camera is lost for 30 frames
    gate = FrameGate(frames, cuts=[100], margin=10)
    ok = {f for f in range(420) if gate(f)}
    assert ok == {*range(10, 90), *range(111, 190), *range(240, 390)}
    bug = tmp_path / "scorebug.jsonl"  # 3 Hz reads; the bug is hidden (a replay) from 5 s to 7 s
    bug.write_text("".join(json.dumps({"t": k / 3.0, "game_clock": None if 15 <= k <= 21 else 600.0 - k}) + "\n"
                           for k in range(150)))
    live = scorebug_live(bug, tol_s=0.5)
    assert live(4.0) and not live(6.0) and live(7.6)
    gated = FrameGate(frames, cuts=[100], margin=10, live=live, fps=FPS)
    assert gated(140) and not gated(180)  # 4.7 s: the bug is shown; 6 s: it is not


# --------------------------------------------------------------------------- flights


def flight_scene(gap: range, wrong: int, weak: int) -> tuple[CameraTrack, BallTrack, DetectionCache, Flight3D]:
    """A shot seen for frames 30-60 by a still camera; the student misses ``gap``, picks a wrong point 12 px off at
    ``wrong`` and sees the ball only weakly (0.15, unlinked) at ``weak``."""
    track = cam_track(range(100))
    fl = shot()
    path: dict[int, tuple[float, float, float]] = {}
    cache = DetectionCache()
    for f in range(30, 61):
        u, v = project(track, f, fl.at(f / FPS))
        if f in gap:
            cache.balls[f] = np.array([[u, v, 0.15]] if f == weak else np.zeros((0, 3)), np.float32)
            continue
        if f == wrong:
            u += 12.0
        path[f] = (u, v, 0.6)
        cache.balls[f] = np.array([[u, v, 0.6]], np.float32)
    return track, BallTrack(path, [fl]), cache, fl


def test_flight_frames_are_labelled_at_the_flights_projection_where_the_student_missed_or_erred() -> None:
    track, bt, cache, fl = flight_scene(gap=range(50, 56), wrong=57, weak=52)
    cands, spans = flight_candidates(bt, track, cache, everywhere, PseudoConfig())
    by = {c.frame: c for c in cands}
    assert spans == set(range(30, 61))
    # the 6-frame gap: a label needs an agreeing student point at most 4 frames before and after
    assert {52, 53} <= set(by) and not {50, 51, 54, 55} & set(by)
    assert set(by) == set(range(30, 61)) - {50, 51, 54, 55}
    for f, c in by.items():
        assert np.allclose(c.xy, project(track, f, fl.at(f / FPS)), atol=1e-6)  # the flight's point, not the pick
    cfg = PseudoConfig()
    assert by[40].kind == "confirmed" and by[40].weight == cfg.max_weight and by[40].hardness == pytest.approx(0.4)
    assert by[52].kind == "filled" and by[52].weight == pytest.approx(cfg.max_weight * cfg.filled_share)
    assert by[52].hardness == pytest.approx(0.85)  # the student's weak 0.15 there
    assert by[57].kind == "filled" and by[57].hardness == pytest.approx(1.6)  # missed it, and preferred a wrong point
    assert max(c.weight for c in cands) <= PSEUDO_MAX_WEIGHT


def test_flights_that_fit_badly_or_lie_next_to_a_cut_give_no_labels() -> None:
    track, bt, cache, _ = flight_scene(gap=range(0), wrong=-1, weak=-1)
    bt.flights = [shot(rms=4.0)]  # 4 px rms in a 360-line frame: 12 px at 1080 lines
    assert flight_candidates(bt, track, cache, everywhere, PseudoConfig()) == ([], set())
    bt.flights = [shot()]
    gate = FrameGate(track.params.keys(), cuts=[45], margin=3)
    cands, _ = flight_candidates(bt, track, cache, gate, PseudoConfig())
    assert {c.frame for c in cands} == set(range(30, 61)) - set(range(42, 49))


# --------------------------------------------------------------------------- the linked path


def pass_scene(n: int = 81) -> tuple[CameraTrack, dict[int, tuple[float, float, float]], DetectionCache]:
    """A pass at 4 ft, 12 ft/s, seen by a camera turning 0.25 degrees a frame; the student scores it 0.7."""
    track = cam_track(range(n), pan=lambda f: -8.0 + 0.25 * f)
    path = {}
    for f in range(n):
        u, v = project(track, f, np.array([30.0 + 0.4 * f, 20.0 + 0.1 * f, 4.0]))
        path[f] = (u, v, 0.7)
    cache = DetectionCache({f: np.array([p], np.float32) for f, p in path.items()})
    return track, path, cache


def test_path_points_need_smooth_motion_and_a_score_of_their_own_or_on_both_sides() -> None:
    track, path, cache = pass_scene()
    path[30] = (*path[30][:2], 0.2)  # unsure, between confident neighbours
    path[40] = (path[40][0] + 12.0, path[40][1], 0.7)  # a wrong pick off the line
    for f in range(70, 74):
        path[f] = (*path[f][:2], 0.3)  # unsure for four frames
    far = (path[55][0] + 60.0, path[55][1], 0.65)  # something else scores nearly as high as the ball
    cache.balls[55] = np.array([path[55], far], np.float32)
    cache.balls[60] = np.array([path[60], (path[60][0] - 60.0, path[60][1], 0.5)], np.float32)
    cache.balls.update({f: np.array([path[f]], np.float32) for f in (30, 40, *range(70, 74))})
    gate = FrameGate(track.params.keys(), cuts=[], margin=5)
    cands = path_candidates(BallTrack(path, []), track, cache, gate, PseudoConfig())
    by = {c.frame: c for c in cands}
    # the wrong pick, and every frame that has it among its neighbours, is out; so is the ambiguous frame and the
    # unsure frames with no confident neighbour on one side
    expected = set(range(5, 76)) - set(range(37, 44)) - {55, 70, 73}
    assert set(by) == expected
    assert {f for f, c in by.items() if c.kind == "anchored"} == {30, 71, 72}
    cfg = PseudoConfig()
    assert by[20].kind == "scored" and by[20].weight == pytest.approx(cfg.max_weight * cfg.scored_share)
    assert by[30].weight == pytest.approx(cfg.max_weight * cfg.anchored_share) and by[30].hardness == pytest.approx(0.8)
    assert by[60].hardness == pytest.approx(0.3 + 0.5)  # a lesser distractor raises the priority
    assert by[20].xy == pytest.approx(path[20][:2])


def test_a_still_point_is_no_ball_even_when_the_camera_pans() -> None:
    track = cam_track(range(60), pan=lambda f: -8.0 + 0.4 * f)
    logo = np.array([60.0, 40.0, 0.0])  # fixed on the floor: it moves 7 px a frame in the image
    path = {f: (*project(track, f, logo), 0.9) for f in range(60)}
    assert abs(path[1][0] - path[0][0]) > 5.0
    cache = DetectionCache({f: np.array([p], np.float32) for f, p in path.items()})
    assert path_candidates(BallTrack(path, []), track, cache, everywhere, PseudoConfig()) == []


# --------------------------------------------------------------------------- sampling


def test_selection_spreads_over_minutes_keeps_spacing_and_caps_and_takes_hard_frames_first() -> None:
    fpm = int(60 * FPS)  # frames per minute
    cands = [Candidate(f, (0.0, 0.0), 0.4, "path", "scored", 0.3 + 0.01 * (k % 7)) for k, f in
             enumerate(range(0, fpm, 5))]  # minute 0: 360 easy-ish frames
    cands += [Candidate(fpm + 10 * k, (0.0, 0.0), 0.35, "flight", "filled", 1.0, flight=k % 2) for k in range(8)]
    cands += [Candidate(2 * fpm + 3 * k, (0.0, 0.0), 0.4, "path", "scored", 0.2) for k in range(fpm // 3)]
    cfg = PseudoConfig(max_frames=20, per_minute=8, min_spacing=10, per_flight=2)
    got = select_targets(cands, cfg, FPS)
    frames = [c.frame for c in got]
    by_min = {m: [c for c in got if c.frame // fpm == m] for m in range(3)}
    # minute 1 can give only 2 + 2 (two flights, 2 each); minutes 0 and 2 stop at 8: 20 in all
    assert len(got) == cfg.max_frames and [len(by_min[m]) for m in range(3)] == [8, 4, 8]
    assert sorted(c.flight for c in by_min[1]) == [0, 0, 1, 1]
    assert all(b - a >= cfg.min_spacing for a, b in itertools.pairwise(frames))
    assert all(c.hardness > 0.355 for c in by_min[0])  # minute 0 offers its hardest frames first
    late = [c.frame - 2 * fpm for c in by_min[2]]  # equally hard frames all over minute 2: not just its start
    assert max(late) - min(late) > fpm / 2
    few = select_targets(cands, PseudoConfig(max_frames=3, per_minute=8, min_spacing=10), FPS)
    assert sorted(c.frame // fpm for c in few) == [0, 1, 2]  # turns: one per minute before a second anywhere


def test_one_candidate_per_frame_keeps_the_higher_weight() -> None:
    a = Candidate(100, (1.0, 1.0), 0.5, "flight", "confirmed", 0.4, flight=0)
    b = Candidate(100, (2.0, 2.0), 0.4, "path", "scored", 0.9)
    assert select_targets([b, a], PseudoConfig(), FPS) == [a]


# --------------------------------------------------------------------------- the folder and train_student


def encode(idx: int) -> np.ndarray:
    """A 640x360 frame whose top-left 8 blocks of 40 px spell ``idx`` in binary (JPEG-proof)."""
    img = np.full((SIZE[1], SIZE[0], 3), 90, np.uint8)
    for k in range(8):
        img[0:40, 40 * k:40 * k + 40] = 255 if (idx >> k) & 1 else 0
    return img


def decode(img: np.ndarray, scale: float) -> int:
    return sum(1 << k for k in range(8) if img[int(20 * scale), int((40 * k + 20) * scale)].mean() > 128)


def make_game(root: Path, n: int = 120, cut: int = 100) -> tuple[CameraTrack, Flight3D, dict[int, tuple]]:
    """A game folder with a 120-frame video, a still camera (a flagged cut at frame 100), a held ball, a shot
    (frames 30-60, missed by the student at 40-42) and a pass; persons in every frame."""
    root.mkdir(parents=True, exist_ok=True)
    video = root / "source.avi"
    wr = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, SIZE)
    for i in range(n):
        wr.write(encode(i))
    wr.release()
    (root / "game.json").write_text(json.dumps({"video": str(video)}))
    (root / "camera_parts").mkdir()
    rows = [{"header": True, "center": C.tolist(), "image_size": list(SIZE), "fps": FPS}]
    rows += [{"frame": f, "t": f / FPS, "status": "tracking", "cut": f == cut,
              "params": {"pan": -5.0, "tilt": TILT, "roll": 0.0, "f": FOCAL}} for f in range(n)]
    (root / "camera_parts" / "part0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    track = cam_track(range(n))
    fl = shot()
    path: dict[int, tuple] = {}
    for f in range(10, 91):
        if 30 <= f <= 60:
            X = fl.at(f / FPS)
        elif f < 30:
            X = np.array([36.0 + 0.2 * (f - 10), 25.0, 4.0])
        else:
            X = np.array([52.0 + 0.3 * (f - 61), 25.0 - 0.2 * (f - 61), 3.0])
        if f not in (40, 41, 42):
            path[f] = (*project(track, f, X), 0.6 if 30 <= f <= 60 else 0.7)
    cache = DetectionCache()
    for f in range(n):
        balls = [path[f]] if f in path else []
        cache.balls[f] = np.array(balls + [(600.0, 30.0, 0.3)], np.float32).reshape(-1, 3)
        cache.persons[f] = np.array([[200.0, 150.0, 230.0, 230.0, 0.8]], np.float32)
    (root / "detections").mkdir()
    cache.save(root / "detections" / f"{0:07d}-{n:07d}.npz")
    BallTrack(path, [fl]).save(root / "ball_track.json")
    return track, fl, path


def test_pseudo_folder_round_trip_through_the_students_dataset(tmp_path: Path) -> None:
    root, out = tmp_path / "game", tmp_path / "pseudo"
    make_game(root)
    assert pseudo_labels.main([str(root), "--out", str(out), "--max-frames", "6", "--per-minute", "6",
                               "--save-width", "320"]) == 0
    meta = json.loads((out / "meta.json").read_text())
    assert meta["kind"] == "pseudo" and meta["complete"] and meta["max_weight"] == PSEUDO_MAX_WEIGHT
    assert meta["person_weight"] == 0.0 and meta["counts"]["labelled"] == 6
    dets = [json.loads(line) for line in (out / "detections.jsonl").open()]
    labs = {(r["clip"], r["frame"]): r for r in map(json.loads, (out / "ball_labels.jsonl").open())}
    targets = [r for r in dets if not r.get("context")]
    assert len(dets) == 18 and len(targets) == 6
    assert all(r["size"] == list(SIZE) and r["persons"] == [[200.0, 150.0, 230.0, 230.0, 0.8]] for r in dets)
    frames = sorted(r["frame"] for r in targets)
    assert all(15 <= f <= 84 for f in frames)  # not near the start, the cut at 100 or the end of the camera run
    assert all(b - a >= 10 for a, b in itertools.pairwise(frames))
    assert any(labs[(r["clip"], r["frame"])]["source"] == "flight:filled" for r in targets)  # a missed ball
    for r in dets:  # every stored image is the video frame its row names
        img = cv2.imread(str(out / "frames" / f"{r['clip']:04d}" / f"{r['frame']:06d}.jpg"))
        assert img.shape[:2] == (180, 320) and decode(img, 0.5) == r["frame"]
        lab = labs[(r["clip"], r["frame"])]
        assert (lab["w"] == 0.0) == bool(r.get("context"))
        assert 0.0 <= lab["w"] <= PSEUDO_MAX_WEIGHT

    ds = ClipDataset([out], None, train=False)
    assert len(ds) == 6  # the context rows are input frames, not samples
    for i, (_, clip, fr, det, lab) in enumerate(ds.items):
        assert fr == [det["frame"] - 2, det["frame"] - 1, det["frame"]]
        x, t = ds[i]
        assert x.shape == (9, 544, 960)
        assert t["ball_w"][0] == pytest.approx(lab["w"]) and 0.0 < t["ball_w"][0] <= PSEUDO_MAX_WEIGHT
        r, c = np.unravel_index(np.argmax(t["ball_hm"][0]), t["ball_hm"][0].shape)
        assert (c, r) == (int(lab["xy"][0] * 0.5 / STRIDE), int(lab["xy"][1] * 0.5 / STRIDE))
        assert t["person_w"][0] == 0.0 and t["person_mask"].max() == 0.0 and t["person_hm"].max() == 0.0
    half = ClipDataset([out], None, train=False, weights=[0.5])
    assert all(half[i][1]["ball_w"][0] == pytest.approx(0.5 * ds.items[i][4]["w"]) for i in range(len(half)))
    x, t = ClipDataset([out], None, train=True)[0]
    assert x.shape == (9, 512, 512) and t["ball_hm"].shape == (1, 128, 128)

    again = pseudo_labels.main([str(root), "--out", str(out), "--max-frames", "3", "--save-width", "320"])
    assert again == 0 and len(list((out / "frames").iterdir())) == 3  # an earlier pseudo folder is replaced


def test_pseudo_labels_refuse_a_cache_the_ball_track_was_not_linked_from_and_a_teacher_folder(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "game"
    make_game(root)
    other = DetectionCache({f: np.array([[1.0, 2.0, 0.9]], np.float32) for f in range(120)})
    (root / "detections-new").mkdir()
    other.save(root / "detections-new" / "0000000-0000120.npz")
    assert pseudo_labels.main([str(root), "--out", str(tmp_path / "p"), "--det-dir", "detections-new"]) == 2
    assert "was not linked from" in capsys.readouterr().err
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    (teacher / "detections.jsonl").write_text("{}\n")
    assert pseudo_labels.main([str(root), "--out", str(teacher)]) == 2
    assert "not a pseudo-label folder" in capsys.readouterr().err
    assert (teacher / "detections.jsonl").read_text() == "{}\n"  # untouched
    with pytest.raises(SystemExit):
        pseudo_labels.main([str(root), "--out", str(tmp_path / "p"), "--max-weight", "0.8"])
    assert not (tmp_path / "p").exists()
    small = tmp_path / "small.avi"  # a video of another size than the camera track's
    wr = cv2.VideoWriter(str(small), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (320, 180))
    for i in range(120):
        wr.write(cv2.resize(encode(i), (320, 180)))
    wr.release()
    (root / "game.json").write_text(json.dumps({"video": str(small)}))
    assert pseudo_labels.main([str(root), "--out", str(tmp_path / "p")]) == 2
    assert "labels would be misplaced" in capsys.readouterr().err
    assert json.loads((tmp_path / "p" / "meta.json").read_text())["complete"] is False
    with pytest.raises(ValueError, match="incomplete"):  # and the trainer will not read it
        plan_data([str(make_folder(tmp_path / "teacher2", {0: range(3)})), str(tmp_path / "p")])


def test_the_detection_folder_the_ball_track_was_linked_from_is_found(tmp_path: Path) -> None:
    root = tmp_path / "game"
    make_game(root)
    (root / "detections").rename(root / "detections-s002")  # the loop keeps one folder per student
    (root / "detections-s003").mkdir()
    for p in (root / "detections-s002").glob("*.npz"):
        (root / "detections-s003" / p.name).write_bytes(p.read_bytes())
    decoy = DetectionCache({f: np.array([[1.0, 2.0, 0.9]], np.float32) for f in range(120)})
    (root / "detections").mkdir()
    decoy.save(root / "detections" / "0000000-0000120.npz")
    track = BallTrack.load(root / "ball_track.json")
    assert pseudo_labels.resolve_detections(root, track)[0] == "detections-s002"  # detections/ is another cache
    (root / "ball_track.loop.json").write_text(json.dumps({"student": "s003", "det_dir": "detections-s003"}))
    assert pseudo_labels.resolve_detections(root, track)[0] == "detections-s003"  # the loop's stamp comes first
    with pytest.raises(pseudo_labels.InputError, match="was not linked from detections: 0%"):
        pseudo_labels.resolve_detections(root, track, "detections")
    with pytest.raises(pseudo_labels.InputError, match="no detection folder"):
        pseudo_labels.resolve_detections(root, track, "detections-x")
    out = tmp_path / "p"
    assert pseudo_labels.main([str(root), "--out", str(out), "--max-frames", "2", "--save-width", "320"]) == 0
    assert json.loads((out / "meta.json").read_text())["det_dir"] == "detections-s003"


def make_folder(path: Path, clips: dict[int, range], meta: dict | None = None) -> Path:
    path.mkdir(parents=True)
    rows = [{"clip": c, "frame": f, "size": [1920, 1080], "persons": [], "balls": []}
            for c, fr in clips.items() for f in fr]
    (path / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path / "ball_labels.jsonl").write_text("")
    if meta is not None:
        (path / "meta.json").write_text(json.dumps(meta))
    return path


def test_plan_holds_out_teacher_clips_only_and_leaves_out_training_clips_next_to_them(tmp_path: Path) -> None:
    def teacher_clips(base: int) -> dict[int, range]:
        return {k: range(base + 1000 * k, base + 1000 * k + 36) for k in range(12)}

    old = make_folder(tmp_path / "old", teacher_clips(0))  # the first format: no meta.json
    new = make_folder(tmp_path / "new", teacher_clips(500), {"kind": "teacher", "video": "/v/b.mp4", "fps": 30.0})
    pv = {"kind": "pseudo", "video": "/v/a.mp4", "fps": 30.0, "complete": True}
    # pseudo clips of video a: next to old's held-out clip 5 (frames 5000-5035), inside new's held-out clip 5
    # (5500-5535, but new shows video b), and far from both
    pa = make_folder(tmp_path / "pa", {0: range(4950, 4953), 1: range(5530, 5533), 2: range(9000, 9003)}, pv)
    pb = make_folder(tmp_path / "pb", {0: range(5590, 5593), 1: range(7000, 7003)}, {**pv, "video": "/v/b.mp4"})
    plan = plan_data([str(old), str(new), f"{pa}:0.5", str(pb)])
    assert plan.val == {(0, 5), (1, 5)}  # pseudo folders are never held out
    assert plan.weights == [1.0, 1.0, 0.5, 1.0]
    assert (2, 0) in plan.guarded  # old has no meta.json (its video is unknown): 2 s (60 frames) from its clip 5
    assert (2, 1) in plan.train  # another video than new's
    assert (2, 2) in plan.train and (3, 1) in plan.train
    assert (3, 0) in plan.guarded  # pb shows new's video: 5590 is 55 frames after its clip 5
    assert plan.guarded == {(2, 0), (3, 0)}
    rows = {r["path"]: r for r in plan.summary()}
    assert rows[str(pa)] == {"path": str(pa), "kind": "pseudo", "weight": 0.5, "train_clips": 2, "val_clips": 0,
                             "guarded_clips": 1}
    # two folders of the first format keep the split they always had, even where their frames overlap
    twin = make_folder(tmp_path / "twin", {k: range(1000 * k + 10, 1000 * k + 46) for k in range(12)})
    plan2 = plan_data([str(old), str(twin)])
    assert plan2.guarded == set() and plan2.val == {(0, 5), (1, 5)} and len(plan2.train) == 22
    with pytest.raises(ValueError, match="no teacher folder"):
        plan_data([str(pa)])
    make_folder(tmp_path / "partial", {0: range(3)}, {**pv, "complete": False})
    with pytest.raises(ValueError, match="incomplete"):
        plan_data([str(old), str(tmp_path / "partial")])


def test_data_specs_and_folder_weights() -> None:
    assert parse_data("runs/nba/g/pseudo:0.25") == (Path("runs/nba/g/pseudo"), 0.25)
    assert parse_data("runs/nba/g/teacher") == (Path("runs/nba/g/teacher"), None)
    assert parse_data("odd:name") == (Path("odd:name"), None)
    teacher = folder_weights({}, 0.7)
    assert (teacher.ball, teacher.person, teacher.pseudo) == (0.7, 0.7, False) and teacher.ball_cap == float("inf")
    pseudo = folder_weights({"kind": "pseudo", "max_weight": 0.9, "person_weight": 0.3}, 2.0)
    assert pseudo.ball_cap == PSEUDO_MAX_WEIGHT and pseudo.person == PSEUDO_MAX_WEIGHT  # never above the cap
    assert folder_weights({"kind": "pseudo"}).person == 0.0


# --------------------------------------------------------------------------- teacher ball labels


def test_teacher_folder_ball_labels_link_each_clip(tmp_path: Path) -> None:
    folder = tmp_path / "teacher"
    rows = []
    for clip in (0, 1):
        d = folder / "frames" / f"{clip:04d}"
        d.mkdir(parents=True)
        for k in range(8):
            f = 100 * clip + k
            cv2.imwrite(str(d / f"{f:06d}.jpg"), np.full((540, 960, 3), (40, 90, 200), np.uint8))  # orange-ish
            x = 400.0 + 30.0 * k
            balls = [] if k in (3, 4) else [[x - 12, 300.0 - 12, x + 12, 300.0 + 12, 0.6]]
            rows.append({"clip": clip, "frame": f, "size": [1920, 1080], "persons": [], "balls": balls})
    (folder / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    labs = ball_labels.label_folder(folder)
    assert [(r["clip"], r["frame"]) for r in labs] == [(r["clip"], r["frame"]) for r in rows]
    assert [r["source"] for r in labs[:8]] == ["detected"] * 3 + ["filled"] * 2 + ["detected"] * 3
    assert labs[3]["xy"] == [490.0, 300.0]
    assert ball_labels.main([str(folder)]) == 0
    assert len((folder / "ball_labels.jsonl").read_text().splitlines()) == 16
    with pytest.raises(SystemExit):
        ball_labels.main([str(folder)])  # it exists: --force replaces it
    (folder / "meta.json").write_text(json.dumps({"kind": "pseudo"}))
    with pytest.raises(SystemExit):
        ball_labels.main([str(folder), "--force"])
