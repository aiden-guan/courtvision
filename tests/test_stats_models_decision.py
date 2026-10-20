"""DecisionModel (EPV-lite): options on hand-built frames, lane and closeout effects, grade labels."""

from __future__ import annotations

import json
import math

import pytest

from courtvision.geometry.court import court_for
from courtvision.stats.decision import DecisionConfig, DecisionModel, point_segment_distance
from courtvision.stats.shot_quality import ShotQualityModel
from courtvision.types import Event, EventKind, FrameState, PlayerState, ScoreboardReading, Team

H, A = Team.HOME, Team.AWAY
COURT = court_for("nba")
BX, BY = COURT.basket_xy("left")


@pytest.fixture(scope="module")
def model() -> DecisionModel:
    return DecisionModel(COURT, ShotQualityModel("nba"))


def frame(players: list[PlayerState], shot_clock: float | None = None) -> FrameState:
    sb = ScoreboardReading(t=1.0, shot_clock=shot_clock) if shot_clock is not None else None
    return FrameState(frame_idx=30, t=1.0, players=players, ball=None, scoreboard=sb)


def p(pid: str, team: Team, along: float, lateral: float, vx: float = 0.0, vy: float = 0.0) -> PlayerState:
    """Player at (along, lateral) ft from the left basket centre."""
    return PlayerState(pid, team, BX + along, BY + lateral, vx, vy)


def contested_top_vs_open_corner() -> FrameState:
    return frame([
        p("h1", H, 25.0, 0.0), p("a1", A, 24.0, 0.0),        # holder at the top, defender 1 ft away
        p("h2", H, 2.0, -22.5), p("a2", A, 2.0, 2.5),       # corner shooter, nearest defender 25 ft away
        p("h3", H, 15.0, 10.0), p("a3", A, 14.0, 10.5),     # wing, tightly guarded
    ])


def by_kind(opts, kind, target=None):
    return next(o for o in opts if o.kind == kind and o.target_id == target)


# --------------------------------------------------------------------------- geometry helper


def test_point_segment_distance() -> None:
    assert point_segment_distance((5, 3), (0, 0), (10, 0)) == pytest.approx((3.0, 0.5))
    assert point_segment_distance((-4, 3), (0, 0), (10, 0)) == pytest.approx((5.0, 0.0))
    assert point_segment_distance((13, 4), (0, 0), (10, 0)) == pytest.approx((5.0, 1.0))
    assert point_segment_distance((1, 1), (2, 2), (2, 2)) == pytest.approx((math.sqrt(2), 0.0))


# --------------------------------------------------------------------------- options


def test_options_cover_shoot_each_teammate_and_hold(model: DecisionModel) -> None:
    opts = model.options(contested_top_vs_open_corner(), "h1", H, "left")
    assert [(o.kind, o.target_id) for o in opts] == [("shoot", None), ("pass", "h2"), ("pass", "h3"), ("hold", None)]
    assert all(math.isfinite(o.value) and 0.0 <= o.success_prob <= 1.0 for o in opts)
    assert model.options(contested_top_vs_open_corner(), "nobody", H, "left") == []


def test_open_teammate_beats_contested_shot(model: DecisionModel) -> None:
    fr = contested_top_vs_open_corner()
    opts = model.options(fr, "h1", H, "left")
    shoot, corner = by_kind(opts, "shoot"), by_kind(opts, "pass", "h2")
    assert shoot.xpts_if_success == 3.0 and shoot.detail["closest_defender_ft"] == pytest.approx(1.0)
    assert corner.detail["zone"] == "corner_three" and corner.detail["receiver_defender_at_catch_ft"] > 6.0
    assert corner.value > shoot.value
    assert max(opts, key=lambda o: o.value) is corner

    shot = Event("e10", EventKind.SHOT_ATTEMPT, 1.0, 30, team=H, player_id="h1", value=3)
    g = model.grade(fr, shot, H, "left")
    assert g is not None and g.chosen.kind == "shoot" and g.best.target_id == "h2"
    assert g.delta == pytest.approx(shoot.value - corner.value) and g.label == "poor"

    pass_ = Event("e11", EventKind.PASS, 1.0, 30, team=H, player_id="h1", other_player_id="h2")
    g2 = model.grade(fr, pass_, H, "left")
    assert g2 is not None and g2.label == "good" and g2.delta == pytest.approx(0.0)


def test_intercept_prone_lane_is_penalised(model: DecisionModel) -> None:
    # Two mirror-image wings, equally open; a defender sits on the lane to h2 only.
    fr = frame([p("h1", H, 22.0, 0.0), p("h2", H, 12.0, -16.0), p("h3", H, 12.0, 16.0),
                p("a2", A, 17.0, -8.0), p("a9", A, 40.0, 40.0)])
    opts = model.options(fr, "h1", H, "left")
    risky, safe = by_kind(opts, "pass", "h2"), by_kind(opts, "pass", "h3")
    assert risky.detail["lane_defender_ft"] < 0.5 and risky.detail["lane_defender_id"] == "a2"
    assert risky.success_prob < safe.success_prob - 0.1
    assert risky.value < safe.value


def test_longer_passes_complete_less(model: DecisionModel) -> None:
    fr = frame([p("h1", H, 22.0, 0.0), p("h2", H, 22.0, 10.0), p("h3", H, 22.0, -24.0), p("a1", A, 60.0, 30.0)])
    opts = model.options(fr, "h1", H, "left")
    assert by_kind(opts, "pass", "h2").success_prob > by_kind(opts, "pass", "h3").success_prob


def test_closeout_shrinks_receiver_space_by_flight_time(model: DecisionModel) -> None:
    cfg = model.config
    fr = frame([p("h1", H, 24.0, 0.0), p("h2", H, 24.0, -20.0), p("a2", A, 24.0, -12.0)])
    o = by_kind(model.options(fr, "h1", H, "left"), "pass", "h2")
    flight = cfg.pass_release_s + 20.0 / cfg.pass_speed_fts
    assert o.detail["flight_s"] == pytest.approx(flight)
    assert o.detail["receiver_defender_now_ft"] == pytest.approx(8.0)
    assert o.detail["receiver_defender_at_catch_ft"] == pytest.approx(8.0 - cfg.closeout_speed_fts * flight)
    assert o.detail["catch_and_shoot"] is True


def test_moving_receiver_is_led(model: DecisionModel) -> None:
    fr = frame([p("h1", H, 24.0, 0.0), p("h2", H, 10.0, -10.0, vx=-12.0)])
    o = by_kind(model.options(fr, "h1", H, "left"), "pass", "h2")
    cx, cy = o.detail["catch_xy"]
    assert cx < BX + 10.0 and cy == pytest.approx(BY - 10.0)
    assert o.detail["pass_ft"] == pytest.approx(math.hypot(cx - (BX + 24.0), cy - BY))


def test_hold_scales_with_shot_clock(model: DecisionModel) -> None:
    players = [p("h1", H, 20.0, 0.0)]
    hold = lambda sc: by_kind(model.options(frame(players, sc), "h1", H, "left"), "hold")  # noqa: E731
    assert hold(None).value == pytest.approx(1.160)
    assert hold(20.0).value == pytest.approx(1.160)
    assert hold(6.5).value == pytest.approx(1.160 * 0.5)
    assert hold(0.5).value == pytest.approx(0.0)
    fiba = DecisionModel(court_for("fiba"), ShotQualityModel("fiba"))
    assert by_kind(fiba.options(frame(players), "h1", H, "left"), "hold").value == pytest.approx(1.05)


def test_shot_attempt_attrs_refine_the_shoot_option(model: DecisionModel) -> None:
    fr = contested_top_vs_open_corner()
    base = model.grade(fr, Event("e1", EventKind.SHOT_ATTEMPT, 1.0, 30, player_id="h1", value=3), H, "left")
    open_ = model.grade(fr, Event("e2", EventKind.SHOT_ATTEMPT, 1.0, 30, player_id="h1", value=3,
                                  attrs={"closest_defender_ft": 9.0, "catch_and_shoot": True}), H, "left")
    assert open_.chosen.success_prob > base.chosen.success_prob
    assert open_.chosen.detail["closest_defender_ft"] == 9.0


def test_grade_returns_none_when_not_gradable(model: DecisionModel) -> None:
    fr = contested_top_vs_open_corner()
    ft = Event("e1", EventKind.SHOT_ATTEMPT, 1.0, 30, player_id="h1", value=1, attrs={"shot_type": "free_throw"})
    reb = Event("e2", EventKind.REBOUND, 1.0, 30, player_id="h1")
    to_defender = Event("e3", EventKind.PASS, 1.0, 30, player_id="h1", other_player_id="a2")
    ghost = Event("e4", EventKind.PASS, 1.0, 30, player_id="h9", other_player_id="h2")
    for e in (ft, reb, to_defender, ghost):
        assert model.grade(fr, e, H, "left") is None
    # holder missing from the frame but the event carries the release point
    shot = Event("e5", EventKind.SHOT_ATTEMPT, 1.0, 30, player_id="h9", value=2, x=BX + 3.0, y=BY)
    g = model.grade(fr, shot, H, "left")
    assert g is not None and g.chosen.detail["distance_ft"] == pytest.approx(3.0)


def test_label_thresholds_come_from_config() -> None:
    m = DecisionModel(COURT, ShotQualityModel("nba"))
    assert [m.label_for(d) for d in (0.0, -0.05, -0.051, -0.20, -0.2001)] == ["good", "good", "neutral", "neutral", "poor"]
    strict = DecisionModel(COURT, ShotQualityModel("nba"), DecisionConfig(good_delta=-0.01, poor_delta=-0.05))
    assert [strict.label_for(d) for d in (-0.005, -0.03, -0.06)] == ["good", "neutral", "poor"]


def test_grade_to_dict_is_json_safe(model: DecisionModel) -> None:
    fr = contested_top_vs_open_corner()
    g = model.grade(fr, Event("e7", EventKind.PASS, 1.0, 30, player_id="h1", other_player_id="h3"), H, "left")
    d = json.loads(json.dumps(g.to_dict()))
    assert {"t", "player_id", "chosen", "best", "options", "delta", "label", "event_id"} <= set(d)
    assert d["chosen"]["kind"] == "pass" and d["chosen"]["target_id"] == "h3" and d["event_id"] == "e7"
    assert len(d["options"]) == 4 and d["label"] in {"good", "neutral", "poor"}
