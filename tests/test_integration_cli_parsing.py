"""CLI argument parsing for every subcommand, detector specs, flag -> config mapping, the latency
wrapper's credential refusal, and the override regression (``jev.mode=off`` stays a string)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from courtvision.cli import (
    AGPL_WARNING,
    DetectorSpec,
    build_parser,
    cmd_latency,
    latency_credentials_missing,
    load_cli_config,
    main,
    parse_detector,
    serve_flags,
)
from courtvision.config import ConfigError, load_config

SECRET = "sk-ant-test-secret-value-0123456789"


def parse(*argv: str) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def test_simulate_arguments() -> None:
    a = parse("simulate", "--seconds", "30", "--seed", "11", "--league", "fiba", "--mode", "mono", "--out", "runs/x",
              "--config", "my.yaml", "--set", "events.assist_window_s=4", "--set", "sim.fps=25", "--jev", "stub",
              "--json")
    assert (a.command, a.seconds, a.seed, a.league, a.mode, a.out) == ("simulate", 30.0, 11, "fiba", "mono", "runs/x")
    assert a.config == "my.yaml" and a.set == ["events.assist_window_s=4", "sim.fps=25"]
    assert a.jev == "stub" and a.json is True and a.func.__name__ == "cmd_simulate"
    d = parse("simulate")
    assert (d.seconds, d.seed, d.league, d.mode, d.out, d.jev, d.set) == (120.0, None, None, "truth3d", None, None, [])


def test_serve_arguments() -> None:
    a = parse("serve", "--source", "video", "--mode", "mono", "--video", "game.mp4", "--camera", "rtsp://cam/1",
              "--speed", "4", "--host", "0.0.0.0", "--port", "9001", "--jev", "off", "--no-llm", "--tts", "say",
              "--auth-token", "tok", "--open", "--calibration", "c.json", "--roster", "r.yaml",
              "--scoreboard-roi", "roi.json", "--detector", "coreml:model.mlpackage", "--seconds", "60")
    assert (a.source, a.mode, a.video, a.camera, a.speed) == ("video", "mono", "game.mp4", "rtsp://cam/1", 4.0)
    assert (a.host, a.port, a.jev, a.no_llm, a.tts, a.auth_token, a.open) == (
        "0.0.0.0", 9001, "off", True, "say", "tok", True)
    assert (a.calibration, a.roster, a.scoreboard_roi, a.seconds) == ("c.json", "r.yaml", "roi.json", 60.0)
    assert a.detector == DetectorSpec("coreml", "model.mlpackage") and a.func.__name__ == "cmd_serve"
    d = parse("serve")
    assert (d.source, d.host, d.port, d.speed, d.jev, d.no_llm, d.tts, d.open, d.insecure_no_auth) == (
        None, None, None, None, None, False, None, False, False)


def test_run_arguments() -> None:
    a = parse("run", "--video", "g.mp4", "--calibration", "c.json", "--detector", "rfdetr:small",
              "--roster", "r.yaml", "--out", "runs/g", "--start", "5", "--duration", "60", "--league", "ncaa_m")
    assert (a.video, a.calibration, a.roster, a.out, a.start, a.duration, a.league) == (
        "g.mp4", "c.json", "r.yaml", "runs/g", 5.0, 60.0, "ncaa_m")
    assert a.detector == DetectorSpec("rfdetr", "small") and a.func.__name__ == "cmd_run"
    with pytest.raises(SystemExit):
        parse("run", "--calibration", "c.json")  # --video is required


def test_calibrate_render_eval_config_latency_arguments() -> None:
    c = parse("calibrate", "points.json", "--out", "cal.json", "--league", "nfhs", "--image-size", "1920", "1080",
              "--ransac-thresh", "0.5")
    assert (c.points, c.out, c.league, c.image_size, c.ransac_thresh, c.from_sim) == (
        "points.json", "cal.json", "nfhs", [1920, 1080], 0.5, False)
    s = parse("calibrate", "--from-sim", "--points-out", "p.json", "--roi-out", "roi.json")
    assert s.from_sim and s.points is None and s.points_out == "p.json" and s.roi_out == "roi.json"
    r = parse("render", "--seconds", "12", "--seed", "3", "--out", "demo.mp4", "--start", "2", "--no-sidecars")
    assert (r.seconds, r.seed, r.out, r.start, r.no_sidecars) == (12.0, 3, "demo.mp4", 2.0, True)
    assert parse("render").out == "runs/demo.mp4" and parse("render").seconds == 30.0
    e = parse("eval", "runs/x/events.jsonl", "--truth", "t.jsonl", "--tol", "2", "--kinds", "pass,rebound",
              "--no-player", "--json", "r.json", "--out", "r.md")
    assert (e.pred, e.truth, e.tol, e.kinds, e.no_player, e.json, e.out) == (
        "runs/x/events.jsonl", "t.jsonl", 2.0, "pass,rebound", True, "r.json", "r.md")
    g = parse("config", "--default")
    assert g.default and g.func.__name__ == "cmd_config"
    assert parse("config", "--section", "serve", "--set", "serve.port=9000").section == "serve"
    lat = parse("latency", "--configs", "C1,J4", "--n", "5", "--network", "gym-wifi", "--out", "o", "--pause", "0",
                "--no-prewarm", "--packet", "p.json")
    assert (lat.configs, lat.n, lat.network, lat.out, lat.pause, lat.no_prewarm, lat.packet) == (
        "C1,J4", 5, "gym-wifi", "o", 0.0, True, "p.json")


@pytest.mark.parametrize("argv", [
    ["simulate", "--mode", "stereo"],
    ["simulate", "--seconds", "-3"],
    ["simulate", "--set", "no-equals-sign"],
    ["serve", "--port", "70000"],
    ["serve", "--jev", "maybe"],
    ["serve", "--tts", "festival"],
    ["serve", "--detector", "yolo9000:x"],
    ["serve", "--detector", "yolo:w.pt"],  # the AGPL adapter needs the explicit rnd-yolo prefix
    ["serve", "--detector", "coreml"],
    ["run", "--video", "g.mp4", "--detector", "rnd-yolo"],
    ["eval", "events.jsonl", "--tol", "0"],
])
def test_bad_arguments_exit_with_usage_errors(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        parse(*argv)
    assert exc.value.code == 2


def test_no_command_prints_help_and_returns_usage(capsys) -> None:
    assert main([]) == 2
    assert "simulate" in capsys.readouterr().err


@pytest.mark.parametrize(("text", "spec"), [
    ("colorblob", DetectorSpec("colorblob")),
    ("color_blob", DetectorSpec("colorblob")),
    ("rfdetr", DetectorSpec("rfdetr")),
    ("rfdetr:medium", DetectorSpec("rfdetr", "medium")),
    ("rfdetr:weights/ft.pth", DetectorSpec("rfdetr", "weights/ft.pth")),
    ("coreml:models/rfdetr_s.mlpackage", DetectorSpec("coreml", "models/rfdetr_s.mlpackage")),
    ("rnd-yolo:yolo26s.pt", DetectorSpec("rnd-yolo", "yolo26s.pt")),
    ("auto", DetectorSpec("auto")),
])
def test_parse_detector(text: str, spec: DetectorSpec) -> None:
    assert parse_detector(text) == spec


def test_detector_specs_map_to_perception_config() -> None:
    def cfg(spec: str):
        return load_config(None, parse_detector(spec).overrides())

    blob = cfg("colorblob").perception
    assert blob.detector == "color_blob" and blob.jersey_reader == "template"
    small = cfg("rfdetr:small").perception
    assert small.detector == "rfdetr" and small.rfdetr.variant == "small" and small.rfdetr.backend == "torch"
    weights = cfg("rfdetr:/models/a: b.pth").perception  # ':' in a path stays a string (JSON-quoted override)
    assert weights.rfdetr.weights == "/models/a: b.pth"
    coreml = cfg("coreml:m.mlpackage").perception
    assert coreml.rfdetr.backend == "coreml" and coreml.rfdetr.coreml_path == "m.mlpackage"
    assert parse_detector("rnd-yolo:w.pt").overrides() == []  # built by the CLI, injected into LiveGame


def test_rnd_yolo_prints_the_agpl_warning_and_builds_lazily(monkeypatch, capsys) -> None:
    import courtvision.perception.rnd.ultralytics_adapter as adapter

    built: list[str] = []

    class FakeYolo:
        def __init__(self, weights: str) -> None:
            built.append(weights)

    monkeypatch.setattr(adapter, "UltralyticsDetector", FakeYolo)
    det = parse_detector("rnd-yolo:weights.pt").build()
    assert isinstance(det, FakeYolo) and built == ["weights.pt"]
    err = capsys.readouterr().err
    assert AGPL_WARNING in err and "AGPL" in err and "R&D" in err
    assert parse_detector("colorblob").build() is None  # the pipeline builds its own detectors


def test_shipped_cli_never_imports_the_agpl_adapter(monkeypatch, tmp_path) -> None:
    """D9: the bridge is tools/rnd_detectors.py (source checkouts only), never the package itself."""
    import courtvision.cli as cli
    from courtvision.cli import CLIError

    assert "perception.rnd" not in Path(cli.__file__).read_text()
    assert "courtvision.perception.rnd" in (cli.TOOLS_DIR / "rnd_detectors.py").read_text()
    monkeypatch.setattr(cli, "TOOLS_DIR", tmp_path)  # an installed wheel has no tools/ directory
    monkeypatch.delitem(sys.modules, "courtvision_tools_rnd_detectors", raising=False)
    with pytest.raises(CLIError, match="source checkout"):
        parse_detector("rnd-yolo:weights.pt").build()


def test_detector_availability_checks(tmp_path) -> None:
    from courtvision.cli import CLIError

    with pytest.raises(CLIError, match="CoreML package not found"):
        DetectorSpec("coreml", str(tmp_path / "missing.mlpackage")).check_available()
    with pytest.raises(CLIError, match="weights not found"):
        DetectorSpec("rfdetr", str(tmp_path / "missing.pth")).check_available()
    DetectorSpec("colorblob").check_available()  # always available (core dependencies only)


def test_serve_flags_build_the_expected_config(tmp_path) -> None:
    video = tmp_path / "game.mp4"
    video.write_bytes(b"")
    args = parse("serve", "--video", str(video), "--speed", "2", "--port", "9100", "--host", "0.0.0.0",
                 "--no-llm", "--tts", "none", "--jev", "stub", "--seed", "5", "--league", "fiba",
                 "--set", "source.speed=8")
    cfg = load_cli_config(args, serve_flags(args))
    assert cfg.source.kind == "video" and cfg.source.path == str(video) and cfg.source.realtime is True
    assert cfg.source.speed == 2.0  # explicit flags win over --set
    assert (cfg.serve.host, cfg.serve.port) == ("0.0.0.0", 9100)
    assert cfg.narration.llm_enabled is False and cfg.narration.tts == "none" and cfg.jev.mode == "stub"
    assert cfg.sim.seed == 5 and cfg.game.league == "fiba"
    cam = parse("serve", "--camera", "1")
    assert load_cli_config(cam, serve_flags(cam)).source.kind == "camera"
    defaults = parse("serve")
    d = load_cli_config(defaults, serve_flags(defaults))
    assert (d.source.kind, d.serve.host, d.serve.port) == ("sim", "127.0.0.1", 8765)


def test_override_boolean_words_stay_strings_for_string_fields(tmp_path) -> None:
    # regression: YAML 1.1 reads `off` as False, so `--set jev.mode=off` used to fail on a str field
    cfg = load_config(None, ["jev.mode=off", "narration.llm_enabled=off", "narration.tts=none",
                             "pipeline.captions_sidecar=no", "llm.prewarm=yes", "source.realtime=false"])
    assert cfg.jev.mode == "off" and cfg.narration.llm_enabled is False
    assert cfg.pipeline.captions_sidecar is False and cfg.llm.prewarm is True and cfg.source.realtime is False
    # a YAML file cannot be fixed up (the spelling is gone), but the error says what to do
    path = tmp_path / "cfg.yaml"
    path.write_text("jev:\n  mode: off\n")
    with pytest.raises(ConfigError, match="quote the value"):
        load_config(path)
    path.write_text("jev:\n  mode: 'off'\n")
    assert load_config(path).jev.mode == "off"


def test_cli_reports_config_errors_with_the_dotted_path(capsys) -> None:
    assert main(["config", "--set", "events.ball_min_conff=1"]) == 2
    assert "events.ball_min_conff" in capsys.readouterr().err


def test_latency_refuses_without_credentials(tmp_path, capsys) -> None:
    env = {"HOME": str(tmp_path)}  # no keys, no `ant auth` profile
    assert latency_credentials_missing(["C1", "J1"], env) == [
        "Anthropic (ANTHROPIC_API_KEY or `ant auth login`)", "TypeSafe (TYPESAFE_API_KEY)"]
    assert latency_credentials_missing(["J2"], {**env, "TYPESAFE_API_KEY": SECRET}) == []
    called: list[list[str]] = []
    from courtvision.cli import CLIError

    with pytest.raises(CLIError, match="refusing to run") as exc:
        cmd_latency(parse("latency", "--configs", "C1"), env=env, harness_main=called.append)
    assert called == [] and SECRET not in str(exc.value)
    # through main(): the real environment of the test run may hold keys, so only check the parse error path
    assert main(["latency", "--configs", "C9"]) == 2
    assert "unknown latency configs" in capsys.readouterr().err


def test_latency_forwards_to_the_harness_with_credentials(tmp_path) -> None:
    env = {"HOME": str(tmp_path), "ANTHROPIC_API_KEY": SECRET, "TYPESAFE_API_KEY": SECRET}
    seen: list[list[str]] = []

    def fake_main(argv: list[str]) -> int:
        seen.append(argv)
        return 0

    args = parse("latency", "--configs", "c1, j4", "--n", "3", "--network", "lte", "--out", str(tmp_path),
                 "--no-prewarm")
    assert cmd_latency(args, env=env, harness_main=fake_main) == 0
    assert seen == [["--configs", "C1,J4", "--n", "3", "--network", "lte", "--out", str(tmp_path), "--pause", "0.5",
                     "--no-prewarm"]]
    assert SECRET not in json.dumps(seen)


def test_harness_parses_the_forwarded_arguments(tmp_path) -> None:
    from courtvision.cli import _load_harness

    harness = _load_harness()
    cfg = harness.parse_args(["--configs", "C1,J4", "--n", "3", "--network", "lte", "--out", str(tmp_path),
                              "--pause", "0.5", "--no-prewarm"])
    assert cfg.configs == ["C1", "J4"] and cfg.n_calls == 3 and cfg.prewarm is False
