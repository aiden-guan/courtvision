"""CLI commands end to end (offline, no network): simulate writes every output, eval reads them back,
config prints loadable YAML, calibrate (--from-sim and the example points file), render writes a
readable mp4 plus its sidecars, run processes that video, and the roster file loader."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from courtvision.cli import load_points_file, main, read_events_jsonl
from courtvision.config import AppConfig, load_config
from courtvision.geometry.homography import Calibration
from courtvision.roster import RosterError, load_roster, parse_roster, roster_to_dict, save_roster
from courtvision.types import EventKind, Team

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs" / "examples"
RUN_FILES = ("events.jsonl", "truth_events.jsonl", "transcript.jsonl", "insights.jsonl", "stats.json",
             "report.json", "report.md", "config.yaml")


CREDENTIAL_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "TYPESAFE_API_KEY", "ANTHROPIC_PROFILE",
                   "COURTVISION_TOKEN")


@contextlib.contextmanager
def no_credentials(tmp: Path):
    """Commands read os.environ: never pick up a developer's real keys or `ant auth` profile."""
    with pytest.MonkeyPatch.context() as mp:
        for name in CREDENTIAL_VARS:
            mp.delenv(name, raising=False)
        mp.setenv("ANTHROPIC_CONFIG_DIR", str(tmp / "no-anthropic-profile"))
        yield


@pytest.fixture(autouse=True)
def _no_credentials(tmp_path):
    with no_credentials(tmp_path):
        yield


@pytest.fixture(scope="module")
def sim_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    out = d / "sim30"
    buf = io.StringIO()
    with no_credentials(d), contextlib.redirect_stdout(buf):
        code = main(["simulate", "--seconds", "30", "--seed", "7", "--out", str(out), "--set", "narration.tts=none"])
    return code, out, buf.getvalue()


def test_simulate_writes_every_output(sim_run) -> None:
    code, out, text = sim_run
    assert code == 0
    for name in RUN_FILES:
        assert (out / name).is_file(), name
        assert name == "insights.jsonl" or (out / name).stat().st_size > 0, name  # 30 s may raise no coach card
    assert text.startswith("# CourtVision simulate: sim/truth3d")
    assert "Headline macro-F1" in text and "| pass |" in text and "## Box score vs truth" in text
    assert "templates only" in text and "Jev: off" in text  # no keys: template-only, Jev fallbacks
    report = json.loads((out / "report.json").read_text())
    assert report["summary"]["frames"] == 900 and report["summary"]["headline_f1"] >= 0.6
    assert report["jev"]["mode"] == "off"
    cfg = load_config(out / "config.yaml")  # the run's config reloads into the same settings
    assert cfg.sim.seconds == 30.0 and cfg.sim.seed == 7 and cfg.jev.mode == "off"
    events = read_events_jsonl(out / "events.jsonl")
    assert any(e["kind"] == "pass" for e in events) and all("event_id" in e for e in events)
    assert json.loads((out / "stats.json").read_text())["score"]


def test_simulate_json_summary(tmp_path, capsys) -> None:
    assert main(["simulate", "--seconds", "10", "--mode", "mono", "--out", str(tmp_path / "m"), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["source"] == "sim/mono" and summary["frames"] == 300 and summary["run_dir"] == str(tmp_path / "m")


def test_eval_on_the_simulate_outputs(sim_run, tmp_path, capsys) -> None:
    _, out, _ = sim_run
    report_json, report_md = tmp_path / "eval.json", tmp_path / "eval.md"
    assert main(["eval", str(out), "--json", str(report_json), "--out", str(report_md)]) == 0  # run dir form
    text = capsys.readouterr().out
    assert "# CourtVision evaluation" in text and "Headline macro-F1" in text
    rep = json.loads(report_json.read_text())
    saved = json.loads((out / "report.json").read_text())["eval"]["events"]
    assert rep["headline"]["macro_f1"] == pytest.approx(saved["headline"]["macro_f1"])  # same as the run's own eval
    assert report_md.read_text().startswith("# CourtVision evaluation")
    # explicit files, a kind filter, and a perfect self-match
    assert main(["eval", str(out / "truth_events.jsonl"), "--truth", str(out / "truth_events.jsonl"),
                 "--kinds", "pass", "--json", str(report_json)]) == 0
    capsys.readouterr()
    self_rep = json.loads(report_json.read_text())
    assert list(self_rep["per_kind"]) == ["pass"] and self_rep["per_kind"]["pass"]["f1"] == 1.0


def test_eval_errors(tmp_path, capsys) -> None:
    assert main(["eval", str(tmp_path / "missing.jsonl"), "--truth", str(tmp_path / "t.jsonl")]) == 2
    bad = tmp_path / "events.jsonl"
    bad.write_text('{"kind": "pass", "t": 1.0}\nnot json\n')
    assert main(["eval", str(bad), "--truth", str(bad)]) == 2
    assert f"{bad}:2" in capsys.readouterr().err
    assert main(["eval", str(bad)]) == 2  # no truth file next to it
    assert "pass --truth" in capsys.readouterr().err


def test_config_prints_parseable_yaml(tmp_path, capsys) -> None:
    assert main(["config"]) == 0
    data = yaml.safe_load(capsys.readouterr().out)
    path = tmp_path / "effective.yaml"
    path.write_text(yaml.safe_dump(data))
    assert load_config(path) == AppConfig()
    assert main(["config", "--set", "serve.port=9000", "--set", "jev.mode=stub"]) == 0
    changed = yaml.safe_load(capsys.readouterr().out)
    assert changed["serve"]["port"] == 9000 and changed["jev"]["mode"] == "stub"
    assert main(["config", "--section", "events"]) == 0
    assert list(yaml.safe_load(capsys.readouterr().out)) == ["events"]
    assert main(["config", "--default"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("# CourtVision default configuration") and yaml.safe_load(text)["serve"]["port"] == 8765
    assert main(["config", "--section", "nope"]) == 2


def test_calibrate_from_sim(tmp_path, capsys) -> None:
    out, pts, roi = tmp_path / "cal.json", tmp_path / "points.json", tmp_path / "roi.json"
    assert main(["calibrate", "--from-sim", "--out", str(out), "--points-out", str(pts), "--roi-out", str(roi)]) == 0
    line = capsys.readouterr().out.splitlines()[0]
    assert line.startswith("rmse 0.000 ft over") and "rims: left, right" in line
    calib = Calibration.load(out)
    assert calib.image_size == (1280, 720) and set(calib.rims) == {"left", "right"} and calib.league == "nba"
    # the fitted homography maps the sim camera's projected landmarks back onto the court
    from courtvision.geometry.camera import PinholeCamera
    from courtvision.geometry.court import court_for

    sim = AppConfig().sim
    cam = PinholeCamera.look_at(sim.camera_position, sim.camera_target, tuple(sim.image_size), sim.hfov_deg)
    xy = court_for("nba").landmarks()["L_lane_ft_bottom"]
    uv = cam.project(np.array([*xy, 0.0]))
    assert np.allclose(calib.homography.image_to_court(uv), xy, atol=0.01)
    # the written points file (rounded to 0.1 px) calibrates too
    assert main(["calibrate", str(pts), "--out", str(tmp_path / "cal2.json")]) == 0
    rmse = float(capsys.readouterr().out.split()[1])
    assert rmse < 0.05
    assert json.loads(roi.read_text())["cells"]["home"]


def test_calibrate_the_example_points_file(tmp_path, capsys) -> None:
    out = tmp_path / "cal.json"
    assert main(["calibrate", str(EXAMPLES / "calibration_points.json"), "--out", str(out)]) == 0
    rmse = float(capsys.readouterr().out.split()[1])
    assert 0.0 < rmse < 0.25  # integer-pixel clicks on the sim camera
    assert set(Calibration.load(out).rims) == {"left", "right"}


def test_points_file_forms_and_errors(tmp_path, capsys) -> None:
    ex = json.loads((EXAMPLES / "calibration_points.json").read_text())
    from courtvision.geometry.court import court_for

    lm = court_for("nba").landmarks()
    arrays = {"image_points": [p["image"] for p in ex["points"]],
              "court_points": [list(lm[p["landmark"]]) for p in ex["points"]]}
    f = tmp_path / "arrays.yaml"
    f.write_text(yaml.safe_dump(arrays))
    parsed = load_points_file(f)
    assert len(parsed["court_points"]) == len(ex["points"]) and "image_size" not in parsed
    assert main(["calibrate", str(f), "--out", str(tmp_path / "c.json")]) == 2  # no image size anywhere
    assert "--image-size" in capsys.readouterr().err
    assert main(["calibrate", str(f), "--image-size", "1280", "720", "--out", str(tmp_path / "c.json")]) == 0
    capsys.readouterr()
    mixed = {"points": [{"image": [1, 2], "landmark": "center"}, {"image": [3, 4], "court": [1, 1]}]}
    f.write_text(json.dumps(mixed))
    assert main(["calibrate", str(f), "--image-size", "1280", "720"]) == 2
    assert main(["calibrate"]) == 2  # neither a file nor --from-sim
    unknown = {"image_size": [1280, 720], "points": [{"image": [i, i], "landmark": "nowhere"} for i in range(4)]}
    f.write_text(json.dumps(unknown))
    assert main(["calibrate", str(f), "--out", str(tmp_path / "c.json")]) == 1  # ValueError from fit_landmarks
    assert "unknown landmark" in capsys.readouterr().err


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    d = tmp_path_factory.mktemp("render")
    out = d / "demo.mp4"
    with no_credentials(d), contextlib.redirect_stdout(io.StringIO()):
        # a 3 s window (frames 660-749) of a 25 s seed-7 game: possessions, passes and dribbles
        assert main(["render", "--seconds", "25", "--start", "22", "--seed", "7", "--out", str(out)]) == 0
    return out


def test_render_writes_a_readable_mp4_and_sidecars(rendered) -> None:
    cap = cv2.VideoCapture(str(rendered))
    assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(30.0)
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    assert len(frames) == 90 and frames[0].shape == (720, 1280, 3)
    assert frames[0].std() > 10  # an actual picture, not a blank frame
    stem = rendered.with_suffix("")
    calib = Calibration.load(f"{stem}.calibration.json")
    assert calib.image_size == (1280, 720) and set(calib.rims) == {"left", "right"}
    roster = load_roster(f"{stem}.roster.yaml")
    assert len(roster.entries) == 24 and all(e.player_id == f"{e.team.value}-{e.jersey}" for e in roster.entries)
    truth = read_events_jsonl(f"{stem}.truth_events.jsonl")
    assert truth and min(e["t"] for e in truth) >= 0.0  # re-timed to the rendered file's clock
    assert json.loads(Path(f"{stem}.scoreboard_roi.json").read_text())["corners"]


def test_render_without_sidecars_and_bad_start(tmp_path) -> None:
    out = tmp_path / "clip.avi"  # lossless FFV1 for .avi
    assert main(["render", "--seconds", "1", "--out", str(out), "--no-sidecars"]) == 0
    assert out.is_file() and not (tmp_path / "clip.calibration.json").exists()
    assert main(["render", "--seconds", "1", "--start", "5", "--out", str(out)]) == 2


def test_run_processes_the_rendered_video(rendered, tmp_path, capsys) -> None:
    stem = rendered.with_suffix("")
    out = tmp_path / "run"
    code = main(["run", "--video", str(rendered), "--calibration", f"{stem}.calibration.json",
                 "--roster", f"{stem}.roster.yaml", "--scoreboard-roi", f"{stem}.scoreboard_roi.json",
                 "--detector", "colorblob", "--out", str(out)])
    text = capsys.readouterr().out
    assert code == 0, text
    assert text.startswith("# CourtVision run: video/video") and "perception (ms/frame) | 90 |" in text
    assert "courtvision eval" in text  # hint: the render's truth sits next to the video
    for name in ("events.jsonl", "transcript.jsonl", "stats.json", "report.md", "config.yaml"):
        assert (out / name).is_file(), name
    cfg = load_config(out / "config.yaml")
    assert cfg.perception.detector == "color_blob" and cfg.jev.mode == "off" and cfg.source.kind == "video"
    events = read_events_jsonl(out / "events.jsonl")
    kinds = {e["kind"] for e in events}
    assert EventKind.PASS.value in kinds or EventKind.POSSESSION.value in kinds
    named = [e for e in events if e.get("player_id")]
    assert named and all(e["player_id"].startswith(("home-", "away-")) for e in named)
    assert main(["eval", str(out), "--truth", f"{stem}.truth_events.jsonl"]) == 0
    assert "Headline macro-F1" in capsys.readouterr().out


def test_run_needs_a_calibration_and_existing_files(rendered, tmp_path, capsys) -> None:
    assert main(["run", "--video", str(rendered), "--detector", "colorblob", "--out", str(tmp_path / "r")]) == 2
    assert "needs a calibration" in capsys.readouterr().err
    assert main(["run", "--video", str(tmp_path / "missing.mp4")]) == 2
    stem = rendered.with_suffix("")
    assert main(["run", "--video", str(rendered), "--calibration", f"{stem}.calibration.json",
                 "--roster", str(tmp_path / "missing.yaml")]) == 2
    assert "roster file not found" in capsys.readouterr().err


# --------------------------------------------------------------------------- roster files


def test_example_roster_loads() -> None:
    roster = load_roster(EXAMPLES / "roster.yaml")
    assert roster.team_names == {Team.HOME: "Riverside Herons", Team.AWAY: "Lakeview Comets"}
    assert len(roster.entries) == 18
    zero, double = roster.by_jersey(Team.HOME, "0"), roster.by_jersey(Team.HOME, "00")
    assert zero is not None and double is not None and zero.player_id == "home-0" and double.player_id == "home-00"
    assert roster.get("away-7").name == "Lyra Fenwick" and roster.get("away-7").position == "G"


def test_roster_yaml_keeps_jersey_text(tmp_path) -> None:
    f = tmp_path / "r.yaml"
    f.write_text("teams:\n  home:\n    name: A\n    players:\n      - {jersey: 00, name: Zed}\n      - {jersey: 0, name: Yan}\n"
                 "      - {jersey: 07, name: Xia, position: Yes}\n  away:\n    name: B\n    players: []\n")
    roster = load_roster(f, away_name="Visitors")
    assert [e.jersey for e in roster.entries] == ["00", "0", "07"]
    assert roster.entries[2].position == "Yes"  # not the boolean True
    assert roster.team_names[Team.AWAY] == "Visitors"


def test_roster_round_trip_and_flat_form(tmp_path) -> None:
    roster = load_roster(EXAMPLES / "roster.yaml")
    for name in ("r.yaml", "r.json"):
        path = save_roster(roster, tmp_path / name)
        again = load_roster(path)
        assert again.entries == roster.entries and again.team_names == roster.team_names
    flat = {"teams": {"home": "H", "away": "A"},
            "players": [{"team": "home", "jersey": "5", "name": "P"}, {"team": "away", "jersey": 9, "name": "Q",
                                                                          "player_id": "q9"}]}
    r = parse_roster(flat)
    assert [(e.player_id, e.team, e.jersey) for e in r.entries] == [("home-5", Team.HOME, "5"), ("q9", Team.AWAY, "9")]
    assert roster_to_dict(r)["teams"]["away"]["players"][0]["player_id"] == "q9"


@pytest.mark.parametrize(("data", "fragment"), [
    ({"teams": {"home": {"players": [{"jersey": "1"}, {"jersey": "1"}]}}}, "jersey '1' twice"),
    ({"teams": {"home": {"players": [{"jersey": "1", "player_id": "x"}]},
                "away": {"players": [{"jersey": "2", "player_id": "x"}]}}}, "duplicate player_id"),
    ({"teams": {"home": {"players": [{"name": "No Number"}]}}}, "missing jersey"),
    ({"teams": {"visitors": {"players": []}}}, "team must be 'home' or 'away'"),
    ({"teams": {"home": {"players": [{"jersey": "1", "nmae": "typo"}]}}}, "unknown key"),
    ({"players": [{"jersey": "1"}]}, "needs 'team'"),
    ({"roster": []}, "unknown top-level key"),
    ({"teams": {"home": {"players": [{"jersey": "1", "team": "away"}]}}}, "listed under home"),
])
def test_roster_errors_name_the_entry(data, fragment) -> None:
    with pytest.raises(RosterError, match=fragment):
        parse_roster(data, "r.yaml")


def test_run_reports_a_bad_roster(rendered, tmp_path, capsys) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("teams:\n  home:\n    players:\n      - {jersey: '1'}\n      - {jersey: '1'}\n")
    stem = rendered.with_suffix("")
    assert main(["run", "--video", str(rendered), "--calibration", f"{stem}.calibration.json", "--roster", str(bad),
                 "--detector", "colorblob"]) == 2
    assert "twice" in capsys.readouterr().err
