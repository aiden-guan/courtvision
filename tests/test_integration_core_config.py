"""Integration config: default.yaml round trip, unknown-key errors, overrides and type coercion."""

from __future__ import annotations

import dataclasses

import pytest
import yaml

from courtvision.config import (
    DEFAULT_CONFIG_PATH,
    AppConfig,
    ConfigError,
    build_dataclass,
    config_to_yaml,
    dump_config,
    load_config,
    render_default_yaml,
)
from courtvision.geometry.camera import RECOMMENDED_SIM_CAMERA, PinholeCamera, sees_full_court
from courtvision.geometry.court import court_for
from courtvision.narration.candidates import CandidateConfig


def test_default_yaml_is_generated_from_the_dataclasses() -> None:
    # Drift guard: regenerate with `python -m courtvision.config` after changing a default.
    assert DEFAULT_CONFIG_PATH.read_text() == render_default_yaml()


def test_default_yaml_round_trips_to_the_defaults() -> None:
    assert load_config(DEFAULT_CONFIG_PATH) == AppConfig()
    assert load_config(None) == AppConfig()
    # every section is listed with a comment
    text = DEFAULT_CONFIG_PATH.read_text()
    for f in dataclasses.fields(AppConfig):
        assert f"\n{f.name}:" in text and f"# {f.name}: " in text, f.name


def test_dump_and_reload_a_modified_config(tmp_path) -> None:
    cfg = load_config(None, ["sim.seconds=90", "sim.image_size=[640, 360]", "narration.mode=coach",
                             "jev.deadlines_s.B1=1.2", "candidates.n_parallel=3", "game.rule_overrides=[no_shot_clock]"])
    path = tmp_path / "cfg.yaml"
    path.write_text(config_to_yaml(cfg))
    again = load_config(path)
    assert again == cfg
    assert yaml.safe_load(path.read_text()) == dump_config(cfg)


def test_overrides_are_coerced_to_field_types() -> None:
    cfg = load_config(None, [
        "sim.seconds=90", "sim.image_size=[640,360]", "source.realtime=false", "source.camera=1",
        "jev.deadlines_s.B1=1.2", "narration.max_words.color=40", "game.ruleset=null",
        "candidates.n_parallel=3", "events.default_attacking.home=left", "director.max_queued.0=5",
    ])
    assert cfg.sim.seconds == 90.0 and isinstance(cfg.sim.seconds, float)
    assert cfg.sim.image_size == (640, 360) and isinstance(cfg.sim.image_size, tuple)
    assert cfg.source.realtime is False
    assert cfg.source.camera == "1"
    # dict fields merge key by key onto their defaults
    assert cfg.jev.deadlines_s["B1"] == 1.2 and cfg.jev.deadlines_s["B4"] == AppConfig().jev.deadlines_s["B4"]
    assert cfg.narration.max_words["color"] == 40 and cfg.narration.max_words["play_by_play"] == 12
    assert cfg.events.default_attacking == {"home": "left", "away": "left"}
    assert cfg.director.max_queued[0] == 5 and 1 in cfg.director.max_queued
    assert cfg.game.ruleset is None
    # an Optional nested dataclass is built from defaults + the override
    assert isinstance(cfg.candidates, CandidateConfig) and cfg.candidates.n_parallel == 3
    assert cfg.candidates.max_words == CandidateConfig().max_words


def test_yaml_sections_are_partial(tmp_path) -> None:
    path = tmp_path / "partial.yaml"
    path.write_text("events:\n  ball_min_conf: 0.3\nperception:\n  pipeline:\n    jersey_every: 3\n")
    cfg = load_config(path, ["perception.pipeline.smoothing_alpha=0.5"])
    defaults = AppConfig()
    assert cfg.events.ball_min_conf == 0.3
    assert cfg.events.make_tol_ft == defaults.events.make_tol_ft
    assert cfg.perception.pipeline.jersey_every == 3 and cfg.perception.pipeline.smoothing_alpha == 0.5
    assert cfg.perception.color_blob == defaults.perception.color_blob


@pytest.mark.parametrize("override, path", [
    ("events.ball_min_conff=0.3", "events.ball_min_conff"),
    ("perception.pipeline.bogus=1", "perception.pipeline.bogus"),
    ("nosection.key=1", "nosection"),
    ("sim.seconds.deeper=1", "sim.seconds"),
])
def test_unknown_override_keys_name_the_dotted_path(override: str, path: str) -> None:
    with pytest.raises(ConfigError, match=path.replace(".", r"\.")):
        load_config(None, [override])


def test_unknown_yaml_keys_name_the_dotted_path(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("stats:\n  run_window_s: 120\n  run_windwo_s: 3\n")
    with pytest.raises(ConfigError, match=r"stats\.run_windwo_s"):
        load_config(path)
    path.write_text("narration:\n  mode: broadcast\nnarattion: {}\n")
    with pytest.raises(ConfigError, match="narattion"):
        load_config(path)


@pytest.mark.parametrize("override, fragment", [
    ("sim.fps=fast", "sim.fps"),
    ("sim.image_size=[1, 2, 3]", r"sim\.image_size"),
    ("source.realtime=maybe", "source.realtime"),
    ("sim.seed=1.5", "sim.seed"),
    ("events.ball_min_conf=null", "events.ball_min_conf"),
    ("source.kind=tv", "source.kind"),
    ("game.league=wnba", "game"),
    ("narration.mode=radio", "narration.mode"),
    ("operator.replay_kinds=[foul, slam]", "operator.replay_kinds"),
])
def test_bad_values_raise_config_errors(override: str, fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment):
        load_config(None, [override])


def test_override_syntax_errors() -> None:
    with pytest.raises(ConfigError):
        load_config(None, ["sim.seconds"])
    with pytest.raises(ConfigError):
        load_config(None, ["sim..seconds=3"])
    with pytest.raises(ConfigError, match="not found"):
        load_config("/nonexistent/courtvision.yaml")


def test_build_dataclass_rejects_non_mappings() -> None:
    with pytest.raises(ConfigError, match="expected a mapping"):
        build_dataclass(AppConfig, {"sim": [1, 2]})


def test_sim_default_camera_is_the_recommended_full_court_framing() -> None:
    sim = AppConfig().sim
    assert sim.camera_position == RECOMMENDED_SIM_CAMERA["camera_position"]
    assert sim.camera_target == RECOMMENDED_SIM_CAMERA["camera_target"]
    assert sim.hfov_deg == RECOMMENDED_SIM_CAMERA["hfov_deg"]
    assert tuple(sim.image_size) == RECOMMENDED_SIM_CAMERA["image_size"]
    cam = PinholeCamera.look_at(sim.camera_position, sim.camera_target, tuple(sim.image_size), sim.hfov_deg)
    assert sees_full_court(cam, court_for(AppConfig().game.league), margin_px=10.0)


def test_integration_defaults() -> None:
    cfg = AppConfig()
    assert cfg.llm.cache_ttl == "1h"  # Claude runs rarely: keep the cached prefix for an hour
    assert cfg.jev.mode == "live" and cfg.narration.use_jev_validator  # D4 default, fallback without a key
    assert cfg.events.assume_calibrated is False  # only forced for simulator sources (pipeline.sim_assume_calibrated)
