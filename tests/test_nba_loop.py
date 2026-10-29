"""The continuous NBA loop (nba/loop.py) on a synthetic runs tree with fake tools: stage order, resuming, the model
registry, promotion, the leaderboard and the memory / process guard. No video, footage or torch is needed."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from courtvision.nba import loop as nl

G1 = "2024-12-25-LAL-GSW"
G2 = "2025-01-10-BOS-NYK"
TAG = "loop-s001-c000-f000"


def opt(cmd: list[str], name: str, default: str | None = None) -> str | None:
    return cmd[cmd.index(name) + 1] if name in cmd else default


class FakeTools:
    """Stands in for every tool the loop starts: writes what the real one writes, per its CLI contract. ``fail``:
    step names that write their outputs, then exit 1 (once each). ``box_error(game, tag)`` sets the reports."""

    def __init__(self, repo: Path, duration: float = 2500.0, fail: set[str] | None = None, box_error=None,
                 hard_clips: int = 1, cue_rows: int = 1, fail_early: set[str] | None = None,
                 unreadable: set[str] = frozenset()) -> None:
        self.repo, self.duration, self.hard_clips, self.cue_rows = repo, duration, hard_clips, cue_rows
        self.unreadable = unreadable  # nba_analyze.py sub-commands that see no frame (exit 0, empty outputs)
        self.fail = set(fail or ())
        self.fail_early = set(fail_early or ())  # step names that exit 1 before writing anything (once each)
        self.box_error = box_error or (lambda game, tag: 0.5)
        self.calls: list[nl.Step] = []

    def p(self, s: str) -> Path:
        return Path(s) if Path(s).is_absolute() else self.repo / s

    def __call__(self, step: nl.Step, loop: nl.Loop) -> nl.StepResult:
        self.calls.append(step)
        if step.name in self.fail_early:
            self.fail_early.discard(step.name)
            return nl.StepResult(1)
        self.write(step.cmd)
        if step.name in self.fail:
            self.fail.discard(step.name)
            return nl.StepResult(1)
        return nl.StepResult(0)

    def write(self, cmd: list[str]) -> None:  # one branch per tool
        tool = cmd[2] if cmd[1] in ("-m", "-c") else cmd[1]
        out = self.p(opt(cmd, "--out")) if "--out" in cmd else None
        if cmd[1] == "-c":  # ESPN summary
            self.p(cmd[4]).write_text(json.dumps({"plays": []}))
        elif tool == "tools/nba_scan_scorebug.py":
            out.write_text("".join(json.dumps({"t": float(t), "game_clock": 700.0, "period": 1}) + "\n"
                                   for t in range(0, int(self.duration) + 1, 50)))
        elif tool == "tools/nba_track_court.py":
            center = [float(v) for v in opt(cmd, "--center", "47.25,-94.5,36.125").split(",")]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"header": True, "center": center, "image_size": [1920, 1080], "fps": 29.97})
                           + "\n" + json.dumps({"frame": 0, "status": "tracking"}) + "\n")
        elif tool == "tools/nba_analyze.py":
            root, sub = self.p(cmd[2]), cmd[3]
            tdir = root / "runs" / str(opt(cmd, "--tag", "baseline"))
            if sub == "detect":
                import numpy as np

                (root / opt(cmd, "--det-dir")).mkdir(exist_ok=True)
                n = 0 if "detect" in self.unreadable else 3
                np.savez_compressed(root / opt(cmd, "--det-dir") / "0000000-0001000.npz", frames=np.arange(n),
                                    weights=np.array(opt(cmd, "--weights")))
            elif sub == "ball":
                (root / "ball_track.json").write_text(json.dumps({"path": {}, "flights": [], "from": opt(cmd, "--det-dir")}))
            elif sub == "events":
                tdir.mkdir(parents=True, exist_ok=True)
                (tdir / "events.jsonl").write_text("" if "events" in self.unreadable else
                                                   json.dumps({"engine": opt(cmd, "--engine")}) + "\n")
                (tdir / "run.json").write_text(json.dumps({"start": 0.0, "end": 0.0, "engine": opt(cmd, "--engine")}))
            elif sub == "fuse":
                (tdir / "fused.jsonl").write_text(json.dumps({"fusion": opt(cmd, "--fusion-file")}) + "\n")
            elif sub == "eval":
                stem = Path(str(opt(cmd, "--file", "events.jsonl"))).stem
                err = self.box_error(root.name, tdir.name)
                (tdir / f"report_{stem}.json").write_text(json.dumps({
                    "box": {"error": err, "exact_share": 0.25,
                            "rows": [{"team": "home", "stat": "pts", "pred": 110, "truth": 111, "diff": -1},
                                     {"team": "away", "stat": "ast", "pred": 30, "truth": 27, "diff": 3}]},
                    "events": {"shot_attempt": {"f1": 0.71, "precision": 0.7, "recall": 0.72, "n_pred": 90, "n_truth": 88}}}))
        elif tool in ("tools/nba_teacher_labels.py", "courtvision.training.pseudo_labels"):
            pseudo = tool == "courtvision.training.pseudo_labels"
            assert not pseudo or opt(cmd, "--det-dir")  # the folder its ball track was linked from
            (out / "frames" / "0000").mkdir(parents=True, exist_ok=True)
            (out / "meta.json").write_text(json.dumps({"kind": "pseudo", "complete": True} if pseudo else {"kind": "teacher"}))
            (out / "detections.jsonl").write_text(json.dumps({"clip": 0, "frame": 1, "persons": [], "balls": []}) + "\n")
            if "--no-ball-labels" not in cmd:  # the teacher tool links them at the end of its run
                self.write(["py", "-m", "courtvision.training.ball_labels", str(out)])
        elif tool == "courtvision.training.ball_labels":
            folder = self.p(cmd[3])
            assert not (folder / "ball_labels.jsonl").exists() or "--force" in cmd
            (folder / "ball_labels.jsonl").write_text(json.dumps({"clip": 0, "frame": 1, "xy": [5.0, 6.0], "w": 1.0}) + "\n")
        elif tool == "tools/nba_hard_examples.py":
            out.write_text(json.dumps({"tag": opt(cmd, "--tag"), "clips": [{"t": 12.0 + k} for k in range(self.hard_clips)]}))
        elif tool == "courtvision.training.nba_cue_rows":
            out.write_text("".join(json.dumps({"kind": "make", "features": {"x": 1}, "label": 1, "game_id": cmd[3]}) + "\n"
                                   for _ in range(self.cue_rows)))
        elif tool == "courtvision.training.fit_cue_models":
            out.write_text(json.dumps({"models": {}, "for_live": True, "inputs": cmd[cmd.index("--jsonl"):]}))
            out.with_suffix(".report.json").write_text(json.dumps({"make": {
                "status": "uncalibrated", "n_rows": 1, "n_holdout": 0, "split": "in_sample",
                "calibration": {"ece": 0.1}, "gate": {"certified": False}}}))
        elif tool == "tools/nba_calibrate_fusion.py":
            out.write_text(json.dumps({"config": {"rebound_window_s": 5.0}, "box_error": 0.4}))
        elif tool == "courtvision.training.train_student":
            out.mkdir(parents=True, exist_ok=True)
            (out / "last.pt").write_bytes(b"last")
            (out / "best.pt").write_bytes(b"best " + " ".join(cmd).encode())
            (out / "train_summary.json").write_text(json.dumps({"best_ball_f1": 0.66, "epochs": int(opt(cmd, "--epochs")),
                                                                "data": [cmd[i + 1] for i, a in enumerate(cmd) if a == "--data"]}))
        else:
            raise AssertionError(f"unexpected command {cmd}")


def make_game(tmp: Path, name: str = G1, espn: str = "401705034", **meta) -> Path:
    root = tmp / "runs" / "nba" / name
    root.mkdir(parents=True)
    video = tmp / "videos" / f"{name}.mp4"
    video.parent.mkdir(exist_ok=True)
    video.write_bytes(b"")
    (root / "game.json").write_text(json.dumps({"video": str(video), "espn_event": espn, **meta}))
    return root


def legacy_student(tmp: Path, data: tuple[str, ...] = (f"runs/nba/{G1}/teacher",)) -> Path:
    d = tmp / "runs" / "models" / "student"
    d.mkdir(parents=True)
    (d / "best.pt").write_bytes(b"weights before the loop")
    (d / "train_summary.json").write_text(json.dumps({"best_ball_f1": 0.61, "epochs": 12, "data": list(data)}))
    return d / "best.pt"


def make_loop(tmp: Path, **kw) -> nl.Loop:
    cfg = nl.LoopConfig(python="py", poll_s=0.02, grace_s=2.0, camera_chunk_s=1000.0, **kw)
    return nl.Loop(tmp / "runs", tmp, cfg)


def short(step: nl.Step) -> str:
    return step.name.split(":", 1)[1] if step.game else step.name


@pytest.fixture
def calm(monkeypatch):
    """Plenty of free memory and no other CourtVision job."""
    monkeypatch.setattr(nl, "free_memory_pct", lambda: 60.0)
    monkeypatch.setattr(nl, "list_processes", lambda: [])


def run(loop: nl.Loop, tools: FakeTools, **kw) -> tuple[int, list[str]]:
    out: list[str] = []
    rc = nl.run_loop(loop, executor=tools, out=out.append, **kw)
    return rc, out


# --------------------------------------------------------------------------- games


def test_a_second_folder_of_the_same_game_is_a_variant_and_loop_false_leaves_a_folder_out(tmp_path: Path) -> None:
    g = make_game(tmp_path)
    v2 = tmp_path / "runs" / "nba" / f"{G1}-v2"  # as nba_mac_jobs.sh link_v2 makes it
    v2.mkdir()
    (v2 / "game.json").symlink_to(g / "game.json")
    make_game(tmp_path, G2, "401705999", loop=False)
    games = {x.name: x for x in nl.discover_games(tmp_path / "runs" / "nba")}
    assert games[G1].primary and games[G1].game_id == "401705034"
    assert not games[f"{G1}-v2"].primary and games[f"{G1}-v2"].variant_of == G1
    assert games[G2].excluded and not games[G2].primary
    assert [x.name for x in make_loop(tmp_path).primaries()] == [G1]


def test_status_and_next_only_read(tmp_path: Path) -> None:
    make_game(tmp_path)
    legacy_student(tmp_path)
    loop = make_loop(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    text = nl.format_next(loop)
    assert "first, run imports runs/models/student/best.pt as student s001" in text
    assert "next: 2024-12-25-LAL-GSW espn (light)" in text
    shown = nl.format_status(loop)
    assert "student  current s001 (imported)" in shown and f"current set s001-c000-f000: tag {TAG}" in shown
    stages = nl.status(loop)["games"][0]["stages"]
    assert stages["espn"] == stages["scorebug"] == "ready" and stages["camera"] == stages["detect"] == "waiting"
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before  # nothing written


# --------------------------------------------------------------------------- planning and resuming


def test_a_new_game_runs_its_stages_in_order_then_the_fits_then_each_candidate(tmp_path: Path, calm) -> None:
    make_game(tmp_path)
    legacy = legacy_student(tmp_path)
    loop = make_loop(tmp_path)
    tools = FakeTools(tmp_path)
    rc, out = run(loop, tools)
    assert rc == 0, out
    names = [short(s) for s in tools.calls]
    assert names[:16] == ["espn", "scorebug", "camera:part00", "camera:part01", "camera:part02", "detect:s001",
                          "ball:s001", f"events:{TAG}", f"fuse:{TAG}", f"eval_fused:{TAG}", f"eval_events:{TAG}",
                          f"cue_rows:{TAG}", f"hard_times:{TAG}", "hard_teacher:s001", "pseudo:s001", "teacher"]
    f1, c1, s2 = "loop-s001-c000-f001", "loop-s001-c001-f000", "loop-s002-c000-f000"
    assert names[16:] == ["train-cues:c001", "calibrate-fusion:f001", "train-student:s002",
                          # the fusion candidate re-uses the current events: no events step
                          f"fuse:{f1}", f"eval_fused:{f1}", f"eval_events:{f1}",
                          f"events:{c1}", f"fuse:{c1}", f"eval_fused:{c1}", f"eval_events:{c1}",
                          "detect:s002", "ball:s002", f"events:{s2}", f"fuse:{s2}", f"eval_fused:{s2}", f"eval_events:{s2}"]
    calls = {short(s): s.cmd for s in tools.calls}
    # camera chunks: the first estimates the centre, the others get it; the last runs to the end of the video
    assert opt(calls["camera:part00"], "--duration") == "1000" and "--center" not in calls["camera:part00"]
    assert opt(calls["camera:part01"], "--start") == "1000" and opt(calls["camera:part01"], "--center") == "47.25,-94.5,36.125"
    assert "--duration" not in calls["camera:part02"]
    assert calls["detect:s001"][3:] == ["detect", "--weights", "runs/models/student/s001/best.pt", "--det-dir",
                                         "detections-s001", "--workers", "1", "--chunks", "16"]
    assert json.loads(opt(calls[f"events:{c1}"], "--engine")) == {"cue_model_path": "runs/models/cue/c001/cue_models.json"}
    assert opt(calls[f"fuse:{f1}"], "--fusion-file") == "runs/models/fusion/f001/fusion.json"
    assert "--fusion-file" not in calls[f"fuse:{TAG}"]
    student = calls["train-student:s002"]
    assert [student[i + 1] for i, a in enumerate(student) if a == "--data"] == [
        f"runs/nba/{G1}/teacher", f"runs/nba/{G1}/teacher-hard-s001", f"runs/nba/{G1}/pseudo-s001"]
    assert opt(student, "--init") == "runs/models/student/s001/best.pt" and opt(student, "--epochs") == "12"
    assert calls["calibrate-fusion:f001"][1:] == ["tools/nba_calibrate_fusion.py", f"runs/nba/{G1}", "--tag", TAG,
                                                   "--out", "runs/models/fusion/f001.partial/fusion.json", "--workers", "1"]
    assert opt(calls["pseudo:s001"], "--det-dir") == "detections-s001"
    assert calls["hard_teacher:s001"][1:] == ["tools/nba_teacher_labels.py", str(tmp_path / "videos" / f"{G1}.mp4"),
                                               "--scorebug", f"runs/nba/{G1}/scorebug.jsonl", "--out",
                                               f"runs/nba/{G1}/teacher-hard-s001.partial", "--times",
                                               f"runs/nba/{G1}/runs/{TAG}/hard_examples.json"]
    # the registry: the imported student is a copy; the fitted versions are candidates; nothing was promoted
    reg = json.loads((tmp_path / "runs" / "models" / "registry.json").read_text())
    assert reg["current"] == {"student": "s001", "cue": "c000", "fusion": "f000"}
    s1 = reg["versions"]["student"]["s001"]
    assert s1["imported_from"] == "runs/models/student/best.pt" and s1["games_seen"] == ["401705034"]
    assert (tmp_path / "runs/models/student/s001/best.pt").read_bytes() == legacy.read_bytes()
    c001 = reg["versions"]["cue"]["c001"]
    assert c001["fit_games"] == ["401705034"] and c001["holdout_games"] == [] and c001["tags"] == [TAG]
    assert c001["command"][:3] == ["py", "-m", "courtvision.training.fit_cue_models"] and c001["created"]
    assert c001["metrics"]["make"]["status"] == "uncalibrated"
    assert reg["versions"]["fusion"]["f001"]["metrics"]["config"] == {"rebound_window_s": 5.0}
    assert reg["versions"]["student"]["s002"]["metrics"] == {"best_ball_f1": 0.66, "epochs": 12}
    assert not list((tmp_path / "runs" / "models").glob("*/*.partial"))
    receipt = json.loads((tmp_path / f"runs/nba/{G1}/logs/detect-s001.json").read_text())
    assert receipt["status"] == "ok" and receipt["returncode"] == 0
    # resumable: a second run finds nothing to do
    tools.calls.clear()
    rc, out = run(loop, tools)
    assert rc == 0 and tools.calls == [] and "nothing to do" in out
    assert "  train-student: s002 was trained on these inputs already (3 folder(s); no game held out)" in out
    assert nl.next_step(loop)[0] is None


def test_with_two_games_the_fits_hold_out_the_newest_and_promotion_decides_on_it(tmp_path: Path, calm) -> None:
    make_game(tmp_path, G1, "401")
    make_game(tmp_path, G2, "402")
    legacy_student(tmp_path)  # trained on G1's teacher folder
    loop = make_loop(tmp_path)
    # c001 does better than the current models on the game it was not fitted on, worse on the one it was
    tools = FakeTools(tmp_path, box_error=lambda game, tag: {(G1, "c001"): 0.6, (G2, "c001"): 0.3}.get(
        (game, tag.split("-")[2]), 0.5))
    rc, out = run(loop, tools)
    assert rc == 0, out
    reg = loop.registry
    for vid in ("c001", "f001", "s002"):
        v = reg.get(vid)
        assert v["fit_games"] == ["401"] and v["holdout_games"] == ["402"], vid
    student = reg.get("s002")["command"]
    assert [student[i + 1] for i, a in enumerate(student) if a == "--data"] == [
        f"runs/nba/{G1}/teacher", f"runs/nba/{G1}/teacher-hard-s001", f"runs/nba/{G1}/pseudo-s001"]
    assert reg.get("c001")["command"].count("--jsonl") == 1
    names = [s.name for s in tools.calls]
    assert f"{G2}:events:loop-s001-c001-f000" in names and f"{G2}:detect:s002" in names  # every game, each candidate
    d = nl.promote(loop, "c001")
    assert d["outcome"] == "promoted" and d["games"] == ["402"]
    assert (d["candidate_box_error"], d["current_box_error"]) == (0.3, 0.5)
    # the next round: with c001 current, the cue rows of the new tag are the next inputs
    tools.calls.clear()
    assert run(loop, tools)[0] == 0
    assert [s.name for s in tools.calls][:2] == [f"{G1}:cue_rows:loop-s001-c001-f000", f"{G2}:cue_rows:loop-s001-c001-f000"]
    c2 = loop.registry.get("c002")
    assert (c2["fit_games"], c2["holdout_games"], c2["context"]) == (["401"], ["402"], "s001-c001-f000")
    assert loop.registry.get("s003") is None  # the student's inputs did not change
    rows = loop.latest_rows()
    assert {("401", "s002-c001-f000"), ("402", "s002-c001-f000")} <= set(rows)  # re-evaluated with the new current


def test_a_failed_step_leaves_no_output_and_the_rerun_starts_there(tmp_path: Path, calm) -> None:
    make_game(tmp_path)
    legacy_student(tmp_path)
    loop = make_loop(tmp_path)
    tools = FakeTools(tmp_path, fail={f"{G1}:camera:part01"})
    rc, out = run(loop, tools, game=G1)
    assert rc == 1
    first = [short(s) for s in tools.calls]
    assert first[:4] == ["espn", "scorebug", "camera:part00", "camera:part01"]
    assert "detect:s001" not in first and "teacher" in first  # the teacher needs only the scorebug
    parts = tmp_path / "runs" / "nba" / G1 / "camera_parts"
    assert (parts / "part01.jsonl.partial").exists() and not (parts / "part01.jsonl").exists()  # never renamed
    assert json.loads((tmp_path / f"runs/nba/{G1}/logs/camera-part01.json").read_text())["status"] == "failed"
    assert "FAIL" in nl.format_status(loop)
    tools.calls.clear()
    rc, out = run(loop, tools, game=G1)
    assert rc == 0, out
    second = [short(s) for s in tools.calls]
    assert second[:3] == ["camera:part01", "camera:part02", "detect:s001"]
    assert not {"espn", "scorebug", "camera:part00", "teacher"} & set(second)


def test_an_unreadable_video_gives_no_detections_or_events_that_count(tmp_path: Path, calm) -> None:
    make_game(tmp_path)
    legacy_student(tmp_path)
    loop = make_loop(tmp_path)
    root = tmp_path / "runs" / "nba" / G1
    rc, out = run(loop, FakeTools(tmp_path, unreadable={"detect"}), game=G1)  # the tool exits 0 with empty chunks
    assert rc == 1 and any("no frame was detected" in line for line in out)
    det = root / "detections-s001"
    assert not list(det.glob("*.npz")) and not (det / "loop.json").exists()
    assert json.loads((root / "logs" / "detect-s001.json").read_text())["status"].startswith("failed after it ran")
    assert nl.status(loop)["games"][0]["stages"]["detect"].startswith("failed")
    rc, out = run(loop, FakeTools(tmp_path, unreadable={"events"}), game=G1)  # detected now; the events come back empty
    assert rc == 1 and any("has no event" in line for line in out)
    tdir = root / "runs" / TAG
    assert (tdir / "run.json.empty").is_file() and not (tdir / "run.json").exists() and not (tdir / "fused.jsonl").exists()
    tools = FakeTools(tmp_path)
    assert run(loop, tools, game=G1)[0] == 0
    assert [short(s) for s in tools.calls][0] == f"events:{TAG}" and (tdir / "report_fused.json").is_file()


def test_max_steps_counts_steps_with_a_command(tmp_path: Path, calm) -> None:
    make_game(tmp_path)
    legacy_student(tmp_path)
    loop = make_loop(tmp_path)
    tools = FakeTools(tmp_path)
    rc, out = run(loop, tools, max_steps=2)
    assert rc == 0 and [short(s) for s in tools.calls] == ["espn", "scorebug"]
    assert "stopped after 2 step(s); next: 2024-12-25-LAL-GSW camera part00" in out[-1]


def test_events_are_reused_and_the_ball_track_is_rebuilt_for_the_student_a_step_needs(tmp_path: Path, calm) -> None:
    make_game(tmp_path)
    legacy_student(tmp_path)
    loop = make_loop(tmp_path)
    tools = FakeTools(tmp_path)
    assert run(loop, tools)[0] == 0  # ends with the s002 candidate: ball_track.json is now s002's
    root = tmp_path / "runs" / "nba" / G1
    assert json.loads((root / "ball_track.loop.json").read_text())["student"] == "s002"
    # a new fusion candidate: fusion reads the ball track's flights, so the current student's track comes back first
    vdir = tmp_path / "runs" / "models" / "fusion" / "f002"
    vdir.mkdir()
    (vdir / "fusion.json").write_text(json.dumps({"config": {"assists": False}}))
    (vdir / "loop.json").write_text(json.dumps({"kind": "fusion", "id": "f002", "games_seen": ["401705034"]}))
    loop.refresh()
    steps = nl.plan(loop, scope="eval", candidate="f002")
    assert [short(s) for s in steps] == ["register:f002"]  # a finished folder not in the registry yet
    tools.calls.clear()
    rc, out = run(loop, tools, scope="eval", candidate="f002")
    assert rc == 0, out
    assert [short(s) for s in tools.calls] == ["ball:s001", "fuse:loop-s001-c000-f002", "eval_fused:loop-s001-c000-f002",
                                               "eval_events:loop-s001-c000-f002"]
    copied = root / "runs" / "loop-s001-c000-f002"
    assert (copied / "events.jsonl").read_text() == (root / "runs" / TAG / "events.jsonl").read_text()
    assert json.loads((copied / "loop.json").read_text())["events_from"] == TAG


def test_an_interrupted_student_training_resumes_with_the_same_command(tmp_path: Path, calm) -> None:
    make_game(tmp_path)
    legacy_student(tmp_path)
    loop = make_loop(tmp_path)
    tools = FakeTools(tmp_path, fail={"train-student:s002"})
    assert run(loop, tools, game=G1)[0] == 0
    rc, _ = run(loop, tools, scope="student")
    assert rc == 1
    part = tmp_path / "runs" / "models" / "student" / "s002.partial"
    first = json.loads((part / "loop.json").read_text())["command"]
    assert (part / "last.pt").is_file() and nl.Registry(tmp_path / "runs" / "models").get("s002") is None
    loop.refresh()
    step, _ = nl.next_step(loop, scope="student")
    assert step is not None and step.cmd == [*first, "--resume"]
    tools.calls.clear()
    assert run(loop, tools, scope="student")[0] == 0
    assert [s.cmd for s in tools.calls] == [[*first, "--resume"]]
    assert not part.exists() and (tmp_path / "runs/models/student/s002/best.pt").is_file()
    assert nl.Registry(tmp_path / "runs" / "models").get("s002")["fit_games"] == ["401705034"]


def legacy_game(tmp: Path) -> Path:
    """A game analysed before the loop (nba_mac_jobs.sh): every early stage done, detections of unknown weights."""
    root = make_game(tmp)
    (root / "scorebug.jsonl").write_text(json.dumps({"t": 0.0}) + "\n" + json.dumps({"t": 9000.0}) + "\n")
    (root / "camera_parts").mkdir()
    (root / "camera_parts" / "part0.jsonl").write_text(json.dumps({"header": True, "center": [1, 2, 3]}) + "\n")
    (root / "espn_summary.json").write_text("{}")
    FakeTools(tmp).write(["py", "tools/nba_teacher_labels.py", "v", "--out", f"runs/nba/{G1}/teacher"])
    for d in ("detections", "detections-full"):
        (root / d).mkdir()
        (root / d / "0000000-0026000.npz").write_bytes(b"PK")
    (root / "ball_track.json").write_text("{}")
    return root


def test_adopt_declares_a_detection_folder_of_a_student_so_detect_is_not_run_again(tmp_path: Path, calm) -> None:
    root = legacy_game(tmp_path)
    legacy_student(tmp_path)
    loop = make_loop(tmp_path)
    step, _ = nl.next_step(loop)
    assert short(step) == "detect:s001"  # nothing says which weights made detections-full
    stamp = nl.adopt_detections(loop, G1, "detections-full")
    assert stamp["student"] == "s001" and stamp["adopted"]
    loop.refresh()
    step, _ = nl.next_step(loop)
    assert short(step) == "ball:s001" and opt(step.cmd, "--det-dir") == "detections-full"
    assert "before the loop" in step.why
    with pytest.raises(ValueError, match="already holds"):
        nl.adopt_detections(loop, G1, "detections")
    (root / "detections-empty").mkdir()
    with pytest.raises(ValueError, match="no detection files"):
        nl.adopt_detections(loop, G1, "detections-empty")


def test_a_teacher_folder_made_without_ball_labels_gets_them_before_training(tmp_path: Path, calm) -> None:
    root = legacy_game(tmp_path)
    legacy_student(tmp_path)
    (root / "teacher" / "ball_labels.jsonl").unlink()  # a teacher folder of the first format
    loop = make_loop(tmp_path)
    game = nl.status(loop)["games"][0]
    assert game["stages"]["ball_labels"] == "ready" and game["inventory"]["no_ball_labels"] == ["teacher"]
    steps = {short(s): s for s in nl.plan(loop)}
    step = steps["ball_labels:teacher"]
    assert step.cmd == ["py", "-m", "courtvision.training.ball_labels", f"runs/nba/{G1}/teacher"] and not step.heavy
    assert not any(k.startswith("train-student") for k in steps)  # nothing to train on yet
    assert loop.fit_notes["student"] == "no game has a teacher-format folder with ball_labels.jsonl"
    assert FakeTools(tmp_path)(step, loop).returncode == 0
    loop.refresh()
    assert nl.training_folders(loop.game(G1)) == [root / "teacher"]
    assert "train-student:s002" in [s.name for s in nl.plan(loop)]


def test_a_student_is_not_trained_on_pseudo_labels_alone(tmp_path: Path, calm) -> None:
    root = legacy_game(tmp_path)
    legacy_student(tmp_path)
    (root / "teacher" / "ball_labels.jsonl").unlink()
    FakeTools(tmp_path).write(["py", "-m", "courtvision.training.pseudo_labels", f"runs/nba/{G1}", "--out",
                               f"runs/nba/{G1}/pseudo-s001", "--det-dir", "detections"])
    loop = make_loop(tmp_path)
    assert nl.training_folders(loop.game(G1)) == [root / "pseudo-s001"]
    assert nl.plan(loop, scope="student") == []
    assert "train_student needs a teacher folder" in loop.fit_notes["student"]
    (root / "pseudo-s001" / "meta.json").write_text(json.dumps({"kind": "pseudo", "complete": False}))
    assert nl.training_folders(loop.game(G1)) == []  # train_student refuses an incomplete pseudo folder


def test_no_hard_examples_need_no_teacher_run_and_empty_cue_rows_no_fit(tmp_path: Path, calm) -> None:
    make_game(tmp_path)
    legacy_student(tmp_path)
    loop = make_loop(tmp_path)
    tools = FakeTools(tmp_path, hard_clips=0, cue_rows=0)
    assert run(loop, tools, game=G1)[0] == 0
    names = [short(s) for s in tools.calls]
    assert f"hard_times:{TAG}" in names and "hard_teacher:s001" not in names
    assert nl.status(loop)["games"][0]["stages"]["hard_teacher"] == "done"
    tools.calls.clear()
    rc, out = run(loop, tools, scope="cue")  # nba_cue_rows exits 0 with no rows: nothing to fit
    assert rc == 0 and tools.calls == []
    assert out[-1] == f"  train-cues: the cue rows of {G1} are empty: nothing to fit"


def test_detect_drops_chunk_files_cut_short_before_it_resumes(tmp_path: Path, calm) -> None:
    import numpy as np

    root = legacy_game(tmp_path)
    legacy_student(tmp_path)
    det = root / "detections-s001"
    det.mkdir()
    np.savez_compressed(det / "0000000-0001000.npz", frames=np.arange(3))
    (det / "0001000-0002000.npz").write_bytes((det / "0000000-0001000.npz").read_bytes()[:40])  # killed mid-write
    loop = make_loop(tmp_path)
    step, _ = nl.next_step(loop)
    assert short(step) == "detect:s001"
    nl.apply_actions(loop, step.prepare)
    assert sorted(p.name for p in det.glob("*.npz")) == ["0000000-0001000.npz"]


def test_an_unreadable_registry_is_never_replaced(tmp_path: Path) -> None:
    make_game(tmp_path)
    reg = tmp_path / "runs" / "models" / "registry.json"
    reg.parent.mkdir(parents=True)
    reg.write_text('{"versions": {"cue": {"c001"')  # cut short
    with pytest.raises(ValueError, match="unreadable"):
        make_loop(tmp_path)
    assert reg.read_text() == '{"versions": {"cue": {"c001"'


def test_a_student_training_that_failed_before_its_first_epoch_starts_over_on_the_current_data(tmp_path: Path,
                                                                                             calm) -> None:
    make_game(tmp_path)
    legacy_student(tmp_path)
    loop = make_loop(tmp_path)
    tools = FakeTools(tmp_path, fail_early={"train-student:s002"})
    assert run(loop, tools, game=G1)[0] == 0
    assert run(loop, tools, scope="student")[0] == 1
    part = tmp_path / "runs" / "models" / "student" / "s002.partial"
    assert (part / "loop.json").is_file() and not (part / "last.pt").exists()
    # new training data meanwhile: nothing to resume, so the next attempt takes it
    FakeTools(tmp_path).write(["py", "tools/nba_teacher_labels.py", "v", "--out", f"runs/nba/{G1}/teacher-hard-s009"])
    loop.refresh()
    step, _ = nl.next_step(loop, scope="student")
    assert step.name == "train-student:s002" and "--resume" not in step.cmd
    assert f"runs/nba/{G1}/teacher-hard-s009" in step.cmd and ("clean", part) in step.prepare


# --------------------------------------------------------------------------- registry


def test_the_registry_never_overwrites_a_version(tmp_path: Path, calm) -> None:
    reg = nl.Registry(tmp_path / "models")
    reg.add({"kind": "cue", "id": "c001", "path": "a"})
    with pytest.raises(ValueError, match="never overwritten"):
        reg.add({"kind": "cue", "id": "c001", "path": "b"})
    (tmp_path / "models" / "cue" / "c005").mkdir(parents=True)  # a stray finished folder keeps its id too
    assert reg.next_id("cue") == "c006"
    dst = tmp_path / "models" / "cue" / "c005"
    src = tmp_path / "models" / "cue" / "c007.partial"
    src.mkdir()
    with pytest.raises(ValueError, match="not overwritten"):
        nl.apply_actions(None, [("rename", src, dst)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not a .partial"):
        nl.apply_actions(None, [("clean", dst)])  # type: ignore[arg-type]
    assert dst.is_dir()

    make_game(tmp_path)
    legacy_student(tmp_path)
    loop = make_loop(tmp_path)
    tools = FakeTools(tmp_path)
    assert run(loop, tools, game=G1)[0] == 0
    assert run(loop, tools, scope="cue")[0] == 0
    c1 = tmp_path / "runs" / "models" / "cue" / "c001" / "cue_models.json"
    sha = nl.file_sha256(c1)
    rc, out = run(loop, tools, scope="cue")  # the same inputs: nothing to fit
    assert rc == 0 and out[-1] == "  train-cues: c001 was fitted on these inputs already (1 game(s); no game held out)"
    assert sorted(loop.registry.versions("cue")) == ["c000", "c001"]
    rows = tmp_path / "runs" / "nba" / G1 / "runs" / TAG / "cue_rows.jsonl"
    rows.write_text(rows.read_text() + rows.read_text())  # new inputs (the game was re-analysed)
    assert run(loop, tools, scope="cue")[0] == 0
    assert sorted(loop.registry.versions("cue")) == ["c000", "c001", "c002"]
    assert nl.file_sha256(c1) == sha and loop.registry.get("c001")["sha256"] == sha
    assert loop.registry.get("c002")["inputs_key"] != loop.registry.get("c001")["inputs_key"]


# --------------------------------------------------------------------------- promotion and leaderboard


def evaluated(loop: nl.Loop, game: str, mset: nl.ModelSet, err: float) -> None:
    """A report_fused.json for ``mset`` on ``game`` and its leaderboard row."""
    tdir = loop.game(game).root / "runs" / mset.tag
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "report_fused.json").write_text(json.dumps({"box": {"error": err, "exact_share": 0.5, "rows": []}, "events": {}}))
    loop.record(game, mset)


def candidate(loop: nl.Loop, vid: str, seen: list[str]) -> None:
    kind = nl.kind_of(vid)
    with loop.registry.locked() as reg:
        reg.bootstrap(loop, persist=True)
        reg.add({"kind": kind, "id": vid, "path": None, "games_seen": seen, "fit_games": seen})
        reg.save()
    loop.refresh()


def test_promotion_counts_only_the_games_the_candidate_was_not_fitted_on(tmp_path: Path) -> None:
    make_game(tmp_path, G1, "401")
    make_game(tmp_path, G2, "402")
    legacy_student(tmp_path)
    loop = make_loop(tmp_path)
    candidate(loop, "c001", ["401"])
    cur = loop.current_set()
    evaluated(loop, G1, cur, 0.30)
    evaluated(loop, G2, cur, 0.30)
    evaluated(loop, G1, cur.swap("c001"), 0.05)  # much better on the game it was fitted on ...
    d = nl.promote(loop, "c001")
    assert d["outcome"] == "refused" and "eval --candidate c001" in d["reason"]  # ... not yet measured on G2
    evaluated(loop, G2, cur.swap("c001"), 0.35)  # ... and worse on the held-out game
    d = nl.promote(loop, "c001")
    assert d["outcome"] == "rejected" and not d["promoted"] and d["games"] == ["402"]
    assert d["candidate_box_error"] == 0.35 and d["current_box_error"] == 0.30
    assert loop.registry.current_id("cue") == "c000" and nl.open_candidates(loop, "cue") == []
    candidate(loop, "c002", ["401"])
    evaluated(loop, G2, cur.swap("c002"), 0.25)
    d = nl.promote(loop, "c002")
    assert d["outcome"] == "promoted" and d["promoted"]
    assert loop.registry.current_id("cue") == "c002" and loop.current_set().key == "s001-c002-f000"
    reg = json.loads((tmp_path / "runs" / "models" / "registry.json").read_text())
    assert [(x["candidate"], x["outcome"]) for x in reg["decisions"]] == [("c001", "refused"), ("c001", "rejected"),
                                                                           ("c002", "promoted")]
    assert reg["current"]["cue"] == "c002"


def test_with_a_single_game_promotion_is_refused_unless_forced(tmp_path: Path) -> None:
    make_game(tmp_path, G1, "401")
    legacy_student(tmp_path)
    loop = make_loop(tmp_path)
    candidate(loop, "f001", ["401"])
    cur = loop.current_set()
    evaluated(loop, G1, cur, 0.30)
    evaluated(loop, G1, cur.swap("f001"), 0.20)
    d = nl.promote(loop, "f001")
    assert d["outcome"] == "refused" and "no held-out game" in d["reason"] and "--force" in d["reason"]
    assert d["in_sample"]["candidate_box_error"] == 0.20 and loop.registry.current_id("fusion") == "f000"
    d = nl.promote(loop, "f001", force=True)
    assert d["outcome"] == "promoted" and d["forced"] and loop.registry.current_id("fusion") == "f001"
    reg = json.loads((tmp_path / "runs" / "models" / "registry.json").read_text())
    assert [(x["candidate"], x["outcome"], x["forced"]) for x in reg["decisions"]] == [("f001", "refused", False),
                                                                                      ("f001", "promoted", True)]
    with pytest.raises(ValueError, match="already the current"):
        nl.promote(loop, "f001")


def test_fits_hold_out_the_newest_game_a_model_has_not_seen(tmp_path: Path) -> None:
    games = [nl.Game(n, tmp_path / n, {}, i) for n, i in (("2025-02-01-A-B", "3"), ("2024-12-25-C-D", "1"),
                                                           ("2025-01-10-E-F", "2"))]
    hold, fit = nl.split_holdout(games, 1)
    assert [g.game_id for g in hold] == ["3"] and [g.game_id for g in fit] == ["1", "2"]
    hold, fit = nl.split_holdout(games, 1, seen={"3"})  # the init already saw game 3: hold out game 2
    assert [g.game_id for g in hold] == ["2"]
    assert nl.split_holdout(games[:1], 1) == ([], games[:1])  # one game: nothing held out
    assert nl.split_holdout(games, 5)[1] != []  # always one game to fit on


def test_the_leaderboard_gets_one_row_per_report(tmp_path: Path, calm) -> None:
    make_game(tmp_path)
    legacy_student(tmp_path)
    loop = make_loop(tmp_path)
    tools = FakeTools(tmp_path)
    assert run(loop, tools, game=G1)[0] == 0
    board = tmp_path / "runs" / "nba" / "leaderboard.jsonl"
    rows = [json.loads(x) for x in board.read_text().splitlines()]
    assert len(rows) == 1
    r = rows[0]
    assert (r["game"], r["game_id"], r["tag"], r["set"]) == (G1, "401705034", TAG, "s001-c000-f000")
    assert r["models"] == {"student": "s001", "cue": "c000", "fusion": "f000"}
    assert r["box_error"] == 0.5 and r["exact_share"] == 0.25
    assert r["diffs"] == {"home.pts": -1, "away.ast": 3} and r["event_f1"] == {"shot_attempt": 0.71}
    assert run(loop, tools, game=G1)[0] == 0 and len(board.read_text().splitlines()) == 1  # not again
    rep = tmp_path / "runs" / "nba" / G1 / "runs" / TAG / "report_fused.json"
    doc = json.loads(rep.read_text())
    doc["box"]["error"] = 0.2  # the report was redone
    rep.write_text(json.dumps(doc))
    loop.refresh()
    steps = nl.plan(loop, game=G1)
    assert [short(s) for s in steps][:1] == [f"record:{TAG}"] and steps[0].cmd is None
    tools.calls.clear()
    assert run(loop, tools, game=G1)[0] == 0 and tools.calls == []
    assert len(board.read_text().splitlines()) == 2
    assert loop.latest_rows()[("401705034", "s001-c000-f000")]["box_error"] == 0.2


# --------------------------------------------------------------------------- memory and process guard


def test_memory_readers() -> None:
    assert nl.parse_memory_pressure("The system has 19327352832 (4718592 pages ...)\n"
                                    "System-wide memory free percentage: 23%\n") == 23.0
    assert nl.parse_memory_pressure("no such line") is None
    meminfo = "MemTotal:       16000000 kB\nMemFree:  1000000 kB\nMemAvailable:    4000000 kB\n"
    assert nl.parse_meminfo(meminfo) == 25.0
    assert nl.parse_meminfo("MemTotal: 100 kB\n") is None


def test_the_runner_refuses_to_start_a_step_below_the_free_memory_floor(tmp_path: Path, monkeypatch) -> None:
    make_game(tmp_path)
    legacy_student(tmp_path)
    monkeypatch.setattr(nl, "list_processes", lambda: [])
    monkeypatch.setattr(nl, "free_memory_pct", lambda: 20.0)
    tools = FakeTools(tmp_path)
    rc, out = run(make_loop(tmp_path), tools)
    assert rc == 3 and tools.calls == [] and "only 20% of memory free (need 25%)" in out[-1]
    # memory runs low once the first step has run: the second one does not start
    monkeypatch.setattr(nl, "free_memory_pct", lambda: 24.0 if tools.calls else 60.0)
    rc, out = run(make_loop(tmp_path), tools)
    assert rc == 3 and [short(s) for s in tools.calls] == ["espn"] and "only 24% of memory free" in out[-1]


def test_the_runner_refuses_while_another_courtvision_job_runs(tmp_path: Path, monkeypatch) -> None:
    make_game(tmp_path)
    legacy_student(tmp_path)
    monkeypatch.setattr(nl, "free_memory_pct", lambda: 60.0)
    monkeypatch.setattr(nl, "list_processes", lambda: [
        (os.getpid(), 1, "python tools/nba_loop.py run"),  # this process
        (99991, 1, "/Users/x/courtvision/.venv/bin/python tools/nba_analyze.py runs/nba/g detect --workers 1")])
    tools = FakeTools(tmp_path)
    rc, out = run(make_loop(tmp_path), tools)
    assert rc == 3 and tools.calls == [] and "pid 99991" in out[0] and str(os.getpid()) not in out[0]


def test_other_jobs_leaves_out_this_process_its_wrapper_its_children_and_read_only_loop_commands() -> None:
    procs = [(1, 0, "/sbin/launchd"),
             (10, 1, "caffeinate -i .venv/bin/python tools/nba_loop.py run"),  # the wrapper of this process
             (11, 10, ".venv/bin/python tools/nba_loop.py run"),  # this process
             (12, 11, ".venv/bin/python tools/nba_analyze.py runs/nba/g detect"),  # its step
             (20, 1, ".venv/bin/python tools/nba_loop.py status"),
             (21, 1, ".venv/bin/python -m courtvision.training.train_student --data x --out y"),
             (22, 1, "vim notes.txt"),
             (23, 1, "/bin/bash tools/nba_mac_jobs.sh full"),
             (24, 1, ".venv/bin/python tools/nba_loop.py --runs runs eval")]
    assert [p for p, _ in nl.other_jobs(procs, me=11)] == [21, 23, 24]


def tree_script(pids: Path) -> list[str]:
    """A step that starts two sleepers, one of them in a session of its own (out of the step's process group),
    writes their pids and sleeps."""
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    return [sys.executable, "-c", ("import os, subprocess, time\n"
                                   f"a = subprocess.Popen({sleeper!r})\n"
                                   f"b = subprocess.Popen({sleeper!r}, start_new_session=True)\n"
                                   f"open({str(pids)!r} + '.tmp', 'w').write(f'{{a.pid}} {{b.pid}}')\n"
                                   f"os.replace({str(pids)!r} + '.tmp', {str(pids)!r})\n"
                                   "time.sleep(60)\n")]


def gone(pids: Path) -> bool:
    a, b = (int(v) for v in pids.read_text().split())
    deadline = time.time() + 30
    while time.time() < deadline and (nl._alive(a) or nl._alive(b)):
        time.sleep(0.05)
    return not nl._alive(a) and not nl._alive(b)


def test_the_watchdog_stops_the_step_and_everything_it_started(tmp_path: Path, monkeypatch) -> None:
    pids = tmp_path / "pids"
    # plenty of memory until the step has started its children, then less than the 8% floor
    monkeypatch.setattr(nl, "free_memory_pct", lambda: 5.0 if pids.exists() else 50.0)
    loop = make_loop(tmp_path)
    step = nl.Step("g:sleep", "sleep", "", "g", True, "test", tree_script(pids), tmp_path / "logs" / "sleep.log")
    t0 = time.time()
    res = nl.execute(step, loop)
    assert res.killed and res.returncode != 0 and res.free_min == 5.0
    assert time.time() - t0 < 30
    assert gone(pids)
    assert "watchdog: only 5% of memory free" in step.log.read_text()


@pytest.mark.parametrize("inherited", ["default", "ignored"])  # some harnesses start processes with SIGTERM ignored
def test_stopping_the_loop_stops_the_running_step_too(tmp_path: Path, monkeypatch, inherited: str) -> None:
    import dataclasses
    import signal

    make_game(tmp_path)
    legacy_student(tmp_path)
    pids = tmp_path / "pids"

    def free() -> float:  # once the step runs its children, the loop gets a SIGTERM (as from `kill`)
        if pids.exists():
            os.kill(os.getpid(), signal.SIGTERM)
        return 60.0

    monkeypatch.setattr(nl, "free_memory_pct", free)
    monkeypatch.setattr(nl, "other_jobs", lambda: [])  # the real process list: kill_tree needs it
    loop = make_loop(tmp_path)

    def executor(step: nl.Step, lp: nl.Loop) -> nl.StepResult:  # the real runner, on a long step
        return nl.execute(dataclasses.replace(step, cmd=tree_script(pids)), lp)

    original = signal.signal(signal.SIGTERM, signal.SIG_IGN if inherited == "ignored" else signal.SIG_DFL)
    try:
        before = signal.getsignal(signal.SIGTERM)
        t0 = time.time()
        rc = nl.run_loop(loop, executor=executor, out=lambda s: None)
        seconds, after = time.time() - t0, signal.getsignal(signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, original)
    assert rc == 130 and seconds < 55 and gone(pids)  # well inside the 60 s the step would sleep
    assert json.loads((tmp_path / f"runs/nba/{G1}/logs/espn.json").read_text())["status"] == "interrupted"
    assert not (tmp_path / f"runs/nba/{G1}/espn_summary.json").exists()
    assert after == before  # the loop put the disposition back
