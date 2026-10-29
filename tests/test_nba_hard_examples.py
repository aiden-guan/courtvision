"""Active learning for the student detector: tools/nba_hard_examples.py (teacher clips where the analysis went wrong)
and the ``--times`` option of tools/nba_teacher_labels.py (planning only: no RF-DETR)."""

from __future__ import annotations

import ast
import importlib.util
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from courtvision.types import Event, EventKind, Team

TOOLS = Path(__file__).resolve().parents[1] / "tools"
K = EventKind
_ids = itertools.count()


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


hard = _load("nba_hard_examples")
teacher = _load("nba_teacher_labels")
FPS = 30.0


def ev(kind: EventKind, t: float, team: Team | None = Team.HOME, value: int = 0) -> Event:
    return Event(f"e{next(_ids)}", kind, t, int(t * FPS), team=team, value=value, player_id="p1")


def scenario() -> tuple[list[Event], list[Event]]:
    """Truth and predictions with every outcome of the diagnosis, and its greedy tie (300 / 302 s)."""
    truth = [ev(K.SHOT_ATTEMPT, 9.0, value=2), ev(K.SHOT_MADE, 10.0, value=2),  # attempts are not matched
             ev(K.SHOT_MISSED, 30.0, Team.AWAY, 3), ev(K.SHOT_MADE, 50.0, value=3), ev(K.SHOT_MADE, 70.0, Team.AWAY, 2),
             ev(K.SHOT_MISSED, 90.0, value=2), ev(K.SHOT_MISSED, 300.0, value=2), ev(K.SHOT_MISSED, 302.0, value=2),
             ev(K.REBOUND, 31.5), ev(K.TURNOVER, 150.0, Team.AWAY)]
    pred = [ev(K.SHOT_MADE, 10.5, value=2), ev(K.SHOT_MISSED, 30.4, Team.HOME, 3), ev(K.SHOT_MISSED, 51.0, value=3),
            ev(K.SHOT_MADE, 69.0, Team.AWAY, 3), ev(K.SHOT_MADE, 120.0, Team.AWAY, 2), ev(K.SHOT_MISSED, 301.5, value=2),
            ev(K.REBOUND, 33.5), ev(K.REBOUND, 200.0, Team.AWAY), ev(K.TURNOVER, 160.0, Team.AWAY)]
    return truth, pred


def test_matching_is_the_diagnosis_matching(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    truth, pred = scenario()
    run = tmp_path / "game" / "runs" / "t"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"start": 0.0, "end": 0.0}))
    (run / "events.jsonl").write_text("".join(json.dumps(e.to_dict()) + "\n" for e in pred))
    diagnose = sys.modules["nba_diagnose"]  # loaded by nba_hard_examples
    monkeypatch.setattr(diagnose, "truth", lambda root: (None, SimpleNamespace(aligned=truth)))
    monkeypatch.setattr(diagnose, "truth_events", lambda aligned: list(aligned))
    monkeypatch.setattr(sys, "argv", ["nba_diagnose.py", str(tmp_path / "game"), "--tag", "t"])
    assert diagnose.main() == 0
    lines = capsys.readouterr().out.splitlines()
    theirs = sorted((float(x.split()[0]), x.split()[1], x.split()[2]) for x in lines if "|" in x)
    rows = [o for k in ("shot", "rebound", "turnover") for o in hard.match(truth, pred, k, 2.5)]
    ours = sorted((round(o.t, 2), o.name, o.outcome) for o in rows)
    assert ours == theirs
    assert dict(Counter(f"{o.name}:{o.outcome}" for o in rows)) == ast.literal_eval(lines[-1])
    out = {(o.t, o.outcome) for o in rows if o.name == "shot"}
    assert {(10.0, "ok"), (30.0, "team"), (50.0, "result"), (70.0, "value"), (90.0, "miss"), (120.0, "extra"),
            (300.0, "ok"), (302.0, "miss")} == out  # 300 s takes the prediction first, though 302 s is nearer


def err(name: str, outcome: str, t: float) -> object:
    return hard.Outcome(name, outcome, t, None, None)


def test_windows_merge_into_back_to_back_clips_and_the_cap_keeps_shots_first() -> None:
    errors = [err("turnover", "miss", 500.0), err("rebound", "team", 300.0), err("shot", "miss", 100.0),
              err("shot", "extra", 101.0)]
    kinds = ["shot", "rebound", "turnover"]
    clips, chosen, skipped = hard.plan_clips(errors, FPS, 36, 100, kinds)
    # shots: 97.5-101 s and 98.5-102 s merge into frames 2925-3060 (4 clips); rebound 8955-9045 and turnover
    # 14955-15045 take 3 each
    assert [c["frame"] for c in clips] == [2925, 2961, 2997, 3033, 8955, 8991, 9027, 14955, 14991, 15027]
    assert all(b["frame"] - a["frame"] >= 36 for a, b in itertools.pairwise(clips))  # no frame labelled twice
    assert clips[0]["t"] == pytest.approx(97.5) and [e["outcome"] for e in clips[1]["events"]] == ["miss", "extra"]
    assert skipped == {"off_camera": 0, "over_cap": 0} and len(chosen) == 4
    capped, chosen, skipped = hard.plan_clips(errors, FPS, 36, 5, kinds)
    assert [c["frame"] for c in capped] == [2925, 2961, 2997, 3033]  # the rebound would need 3 more
    assert [e.name for e in chosen] == ["shot", "shot"] and skipped["over_cap"] == 2
    # rebounds first: their 3 clips leave room for no 3-clip shot or turnover window
    assert [c["frame"] for c in hard.plan_clips(errors, FPS, 36, 5, ["rebound", "shot", "turnover"])[0]] == \
        [8955, 8991, 9027]


def test_the_cap_prefers_errors_where_the_ball_path_was_lost_and_skips_errors_off_camera() -> None:
    errors = [err("shot", "miss", 100.0), err("shot", "miss", 200.0), err("shot", "team", 300.0)]
    ball = set(range(2900, 3100))  # the ball path covers the first shot's window, not the second's
    camera = set(range(8000))  # the third is where the camera is lost (frames 8925-9030)
    clips, chosen, skipped = hard.plan_clips(errors, FPS, 36, 3, ["shot"], camera, ball)
    assert [e.t for e in chosen] == [200.0] and skipped == {"off_camera": 1, "over_cap": 1}
    assert [c["frame"] for c in clips] == [5925, 5961, 5997] and all(c["ball_share"] == 0.0 for c in clips)
    assert all(c["camera_share"] == 1.0 for c in clips)
    edge = hard.plan_clips([err("shot", "miss", 266.0)], FPS, 36, 10, ["shot"], camera)[0]
    assert [c["frame"] for c in edge] == [7905, 7941, 7977]  # 7905-8010: the clip from 7977 is 23/36 on camera
    edge = hard.plan_clips([err("shot", "miss", 267.0)], FPS, 36, 10, ["shot"], camera)[0]
    assert [c["frame"] for c in edge] == [7935, 7971]  # the clip from 8007 has no camera


def test_main_lists_clips_that_the_teacher_labels_exactly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from courtvision.nba.analyze import BallTrack

    root = tmp_path / "game"
    run = root / "runs" / "t"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"start": 0.0, "end": 0.0}))
    truth, pred = scenario()
    (run / "events.jsonl").write_text("")  # the fused events are read when the run has them
    (run / "fused.jsonl").write_text("".join(json.dumps(e.to_dict()) + "\n" for e in pred))
    rows = [{"header": True, "center": [47.0, -94.0, 36.0], "image_size": [1920, 1080], "fps": FPS}]
    rows += [{"frame": f, "status": "tracking", "params": {"pan": 0.0, "tilt": 15.0, "roll": 0.0, "f": 3000.0}}
             for f in range(9500) if not 4400 <= f < 4600]  # the camera is lost around the turnover (150 s)
    (root / "camera_parts").mkdir()
    (root / "camera_parts" / "part0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    BallTrack({f: (900.0, 500.0, 0.6) for f in range(2900, 3100)}, []).save(root / "ball_track.json")
    (root / "game.json").write_text(json.dumps({"video": "source.video"}))
    monkeypatch.setattr(hard, "load_truth", lambda root, lo, hi: [e for e in truth if lo <= e.t <= hi])
    out = root / "hard.json"
    assert hard.main([str(root), "--tag", "t", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["file"] == "fused.jsonl" and doc["fps"] == FPS and doc["clip_frames"] == 36
    assert doc["summary"] == {"rebound:extra": 1, "rebound:ok": 1, "shot:extra": 1, "shot:miss": 2, "shot:ok": 2,
                              "shot:result": 1, "shot:team": 1, "shot:value": 1, "turnover:extra": 1,
                              "turnover:miss": 1}
    # miss / extra / team: shots at 30 (team), 90 and 302 (miss), 120 (extra); the extra rebound at 200; the
    # turnover miss at 150 is off camera, the extra at 160 is not
    assert doc["selected"] == {"rebound:extra": 1, "shot:extra": 1, "shot:miss": 2, "shot:team": 1,
                               "turnover:extra": 1}
    assert doc["skipped"] == {"off_camera": 1, "over_cap": 0}
    times = teacher.load_times(out)
    assert teacher.starts_from_times(times, FPS, 36, 20000) == [c["frame"] for c in doc["clips"]]
    args = teacher.build_parser().parse_args(["video.mp4", "--times", str(out), "--out", str(tmp_path / "th")])
    planned = teacher.plan(args, FPS, 20000)
    assert planned == [(c["frame"], "times") for c in doc["clips"]]  # exactly those; no random clips, no scorebug


# --------------------------------------------------------------------------- tools/nba_teacher_labels.py --times


def test_teacher_times_files_in_every_accepted_form(tmp_path: Path) -> None:
    forms = {"list.json": [12.5, 3.0], "objects.json": [{"t": 12.5, "kind": "shot"}, {"t": 3.0}],
             "clips.json": {"clips": [{"t": 12.5}, {"t": 3.0}], "fps": 30.0}, "times.json": {"times": [12.5, 3]}}
    for name, data in forms.items():
        (tmp_path / name).write_text(json.dumps(data))
        assert teacher.load_times(tmp_path / name) == [12.5, 3.0], name
    for bad in ({"start": [1.0]}, ["1.0"], [True], [{"frame": 30}], 5.0):
        (tmp_path / "bad.json").write_text(json.dumps(bad))
        with pytest.raises(TypeError):
            teacher.load_times(tmp_path / "bad.json")


def test_teacher_starts_from_times_and_merge_with_random_clips() -> None:
    assert teacher.starts_from_times([3.0, 12.5, 12.51, 0.0, -1.0, 400.0], FPS, 36, 9000) == [0, 90, 375, 8963]
    # a random clip goes when it overlaps a listed one (120: frames 120-155 against 100-135)
    assert teacher.merge_starts([100, 500], [0, 60, 120, 450, 536, 900], 36) == [
        (0, "random"), (60, "random"), (100, "times"), (450, "random"), (500, "times"), (536, "random"),
        (900, "random")]


class _Tensor:  # what the teacher does with a frame tensor before RF-DETR
    def permute(self, *a: int) -> _Tensor:
        return self

    def float(self) -> _Tensor:
        return self

    def div(self, v: float) -> _Tensor:
        return self

    def to(self, device: str) -> _Tensor:
        return self


class _RFDETR:
    """Stands in for RF-DETR Small: a person, and a ball moving 10 px a frame."""

    def __init__(self, pretrain_weights: str, device: str) -> None:
        self.model, self.k = SimpleNamespace(device=device), 0

    def predict(self, x: _Tensor, threshold: float, shape: tuple[int, int], include_source_image: bool) -> object:
        import numpy as np

        u = 100.0 + 10.0 * (self.k % 12)  # each 12-frame clip: 100 to 210 px
        self.k += 1
        return SimpleNamespace(xyxy=np.array([[50.0, 60.0, 90.0, 200.0], [u - 6, 94.0, u + 6, 106.0]]),
                               confidence=np.array([0.9, 0.6]), data={"class_name": np.array(["person", "sports ball"])})


def test_a_teacher_run_with_times_labels_exactly_those_clips_and_links_their_balls(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2
    import numpy as np

    from courtvision.training.train_student import ClipDataset, plan_data

    video = tmp_path / "v.avi"
    wr = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (640, 360))
    for i in range(200):
        wr.write(np.full((360, 640, 3), (40, 90, 200), np.uint8))
    wr.release()
    times = tmp_path / "hard.json"
    times.write_text(json.dumps({"clips": [{"t": 4.0}, {"t": 1.0}], "clip_frames": 12}))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        from_numpy=lambda a: _Tensor(), backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))))
    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(RFDETRSmall=_RFDETR))
    out = tmp_path / "teacher-hard"
    assert teacher.main([str(video), "--times", str(times), "--out", str(out), "--clip-frames", "12",
                         "--save-width", "320"]) == 0
    meta = json.loads((out / "meta.json").read_text())
    assert meta["kind"] == "teacher" and meta["video"] == str(video.resolve())
    assert meta["clips"] == [{"clip": 0, "start": 30, "origin": "times"}, {"clip": 1, "start": 120, "origin": "times"}]
    rows = [json.loads(line) for line in (out / "detections.jsonl").open()]
    assert [r["frame"] for r in rows] == [*range(30, 42), *range(120, 132)]  # exactly those clips, nothing random
    labs = [json.loads(line) for line in (out / "ball_labels.jsonl").open()]
    assert len(labs) == 24 and all(lab["source"] == "detected" for lab in labs)
    assert labs[0]["xy"] == [100.0, 100.0] and labs[11]["xy"] == [210.0, 100.0] and labs[0]["w"] >= 0.75
    plan = plan_data([str(out)])  # a teacher folder the trainer reads, clip 5 held out as before (2 clips: none)
    assert plan.val == set() and plan.train == {(0, 0), (0, 1)}
    assert len(ClipDataset([out], plan.train, train=False)) == 24


def test_teacher_plan_adds_random_clips_only_when_asked(tmp_path: Path) -> None:
    times = tmp_path / "times.json"
    times.write_text(json.dumps([10.0, 20.0]))
    bug = tmp_path / "scorebug.jsonl"
    bug.write_text("".join(json.dumps({"t": k / 3.0, "game_clock": 700.0 - k / 3.0}) + "\n" for k in range(900)))
    out = str(tmp_path / "t")
    p = teacher.build_parser()
    assert teacher.plan(p.parse_args(["v.mp4", "--times", str(times), "--out", out]), FPS, 9000) == \
        [(300, "times"), (600, "times")]
    mixed = teacher.plan(p.parse_args(["v.mp4", "--times", str(times), "--clips", "4", "--scorebug", str(bug),
                                       "--bugless-share", "0", "--out", out]), FPS, 9000)
    assert [s for s, o in mixed if o == "times"] == [300, 600] and sum(o == "random" for _, o in mixed) >= 2
    assert all(b - a >= 36 for (a, _), (b, _) in itertools.pairwise(mixed))
    with pytest.raises(ValueError, match="--scorebug"):
        teacher.plan(p.parse_args(["v.mp4", "--times", str(times), "--clips", "3", "--out", out]), FPS, 9000)
    plain = teacher.plan(p.parse_args(["v.mp4", "--scorebug", str(bug), "--bugless-share", "0", "--clips", "5",
                                       "--out", out]), FPS, 9000)
    assert len(plain) == 5 and all(o == "random" for _, o in plain)  # without --times: as before
    assert p.parse_args(["v.mp4", "--scorebug", str(bug), "--out", out]).clips is None  # 150 random clips
