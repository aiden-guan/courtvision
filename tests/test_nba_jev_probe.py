"""tools/nba_jev_probe.py without the network or a video: sampling, crops, the engine's answers, request shapes,
decoding and scoring."""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from courtvision.nba import rim_jev
from courtvision.training.align_pbp import AlignedAction, PBPAction
from courtvision.types import BBox, Event, EventKind, EventStatus, Team

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


probe = _load("nba_jev_probe")


def act(kind: EventKind, t: float, team: Team = Team.HOME, **attrs: object) -> AlignedAction:
    made = {K.SHOT_MADE: True, K.SHOT_MISSED: False}.get(kind)
    value = int(attrs.pop("value", 2))  # type: ignore[call-overload]
    a = PBPAction(next(_ids), 1, 600.0, kind, team=team, made=made, value=value, attrs=dict(attrs))
    return AlignedAction(a, t, t, "clock")


def ev(kind: EventKind, t: float, team: Team = Team.HOME, value: int = 2, **kw: object) -> Event:
    return Event(f"e{next(_ids)}", kind, t, int(t * 30), team=team, value=value, **kw)  # type: ignore[arg-type]


def answer(choices: dict[str, dict[str, float]], nouls: dict[str, float] | None = None) -> SimpleNamespace:
    """A stand-in for the SDK response: ``choices[q].probabilities`` and ``nouls[q].noul``."""
    return SimpleNamespace(choices={k: SimpleNamespace(probabilities=v) for k, v in choices.items()},
                           nouls={k: SimpleNamespace(noul=v) for k, v in (nouls or {}).items()})


def test_ensemble_averages_the_option_orders_and_pick_prefers_an_answer_to_the_abstain_on_a_tie() -> None:
    p = rim_jev.ensemble({"made": 0.6, "missed": 0.3, "unclear": 0.1}, {"made": 0.4, "missed": 0.5, "unclear": 0.1})
    assert p == pytest.approx({"made": 0.5, "missed": 0.4, "unclear": 0.1})
    assert rim_jev.pick(p, "unclear") == "made"
    assert rim_jev.pick({"blue": 0.4, "unclear": 0.4, "white": 0.2}, "unclear") == "blue"
    assert rim_jev.pick({"blue": 0.2, "unclear": 0.6, "white": 0.2}, "unclear") == "unclear"


def test_crops_are_squares_inside_the_frame_with_the_rim_in_their_upper_half() -> None:
    assert rim_jev.square_window(10, 10, 400, 1920, 1080) == (0, 0, 400, 400)
    assert rim_jev.square_window(1900, 1070, 400, 1920, 1080) == (1520, 680, 1920, 1080)
    assert rim_jev.square_window(960, 540, 5000, 1920, 1080) == (420, 0, 1500, 1080)  # never larger than the frame
    rim = BBox(900, 300, 960, 312)  # a 60 px rim: a 420 px crop
    x0, y0, x1, y1 = rim_jev.rim_window(rim, 1920, 1080)
    assert (x1 - x0, y1 - y0) == (420, 420)
    assert x0 < 900 and 960 < x1 and y0 < 306 < (y0 + y1) / 2  # the net below the rim is inside too


def test_balanced_order_alternates_the_classes_so_any_prefix_is_balanced() -> None:
    made = [act(K.SHOT_MADE, float(i)) for i in range(5)]
    missed = [act(K.SHOT_MISSED, 100.0 + i) for i in range(3)]
    order = probe.balanced(made + missed, lambda al: bool(al.action.made), 8, seed=1)
    assert sorted(id(al) for al in order) == sorted(id(al) for al in made + missed)
    assert [bool(al.action.made) for al in order[:6]] == [False, True] * 3
    assert [id(al) for al in probe.balanced(made + missed, lambda al: bool(al.action.made), 8, seed=1)] \
        == [id(al) for al in order]  # seeded


def test_field_goals_leave_out_free_throws_and_unplaced_rows() -> None:
    fg, ft = act(K.SHOT_MADE, 10.0), act(K.SHOT_MADE, 20.0, value=1, shot_type="free_throw")
    unplaced = act(K.SHOT_MISSED, 30.0)
    unplaced.t = None
    assert probe.field_goals([fg, ft, unplaced, act(K.REBOUND, 31.0)]) == [fg]


def test_what_changes_possession() -> None:
    assert probe.changes_possession(act(K.TURNOVER, 1.0))
    assert probe.changes_possession(act(K.STEAL, 1.0))
    assert probe.changes_possession(act(K.REBOUND, 1.0, offensive=False))
    assert not probe.changes_possession(act(K.REBOUND, 1.0, offensive=True))
    assert probe.changes_possession(act(K.SHOT_MADE, 1.0, value=3))
    assert not probe.changes_possession(act(K.SHOT_MISSED, 1.0))
    ft = {"value": 1, "shot_type": "free_throw"}
    assert not probe.changes_possession(act(K.SHOT_MADE, 1.0, ft_index=1, ft_total=2, **ft))
    assert probe.changes_possession(act(K.SHOT_MADE, 1.0, ft_index=2, ft_total=2, **ft))


def test_the_engines_answers() -> None:
    events = [ev(K.SHOT_MISSED, 9.0), ev(K.SHOT_MADE, 11.0), ev(K.SHOT_MADE, 10.2, value=1),  # a free throw
              ev(K.POSSESSION, 5.0, Team.AWAY), ev(K.POSSESSION, 8.0, Team.HOME)]
    assert probe.engine_result(events, 10.4, tol=2.5) == "made"  # the nearest field goal, not the free throw
    assert probe.engine_result(events, 8.7, tol=2.5) == "missed"
    assert probe.engine_result(events, 20.0, tol=2.5) is None
    assert probe.engine_team(events, 7.9) is Team.AWAY
    assert probe.engine_team(events, 8.0) is Team.HOME
    assert probe.engine_team(events, 4.0) is None


def test_engine_events_keep_the_latest_revision_and_drop_retractions(tmp_path: Path) -> None:
    a = ev(K.TURNOVER, 3.0)
    b = Event(a.event_id, K.TURNOVER, 3.5, 105, team=Team.HOME, revision=1)
    gone = ev(K.STEAL, 9.0)
    gone_r = Event(gone.event_id, K.STEAL, 9.0, 270, team=Team.HOME, revision=1, status=EventStatus.RETRACTED)
    path = tmp_path / "events.jsonl"
    path.write_text("".join(__import__("json").dumps(e.to_dict()) + "\n" for e in (a, b, gone, gone_r)))
    assert [(e.event_id, e.t) for e in probe.engine_events(path)] == [(a.event_id, 3.5)]


def test_make_request_asks_both_option_orders_and_one_in_net_question_per_crop() -> None:
    state, q = rim_jev.make_request([b"\xff\xd8 jpeg 1", b"\xff\xd8 jpeg 2", b"\xff\xd8 jpeg 3"])
    assert [f["moment"] for f in state["frames"]] == ["1 of 3 (earliest)", "2 of 3", "3 of 3 (latest)"]
    assert list(q["outcome"]["criteria"]) == ["made", "missed", "unclear"]
    assert list(q["outcome_rev"]["criteria"]) == ["unclear", "missed", "made"]
    assert [k for k in q if k.startswith("in_net_")] == ["in_net_0", "in_net_1", "in_net_2"]
    assert "`frames[2].image`" in q["in_net_2"]["instructions"]


def test_turnover_request_names_each_frame_by_side() -> None:
    state, q = probe.turnover_request([b"b1", b"b2"], [b"a1"], {Team.AWAY: "white", Team.HOME: "blue"})
    assert list(state) == ["before_1", "before_2", "after_1"]
    assert set(q) == {f"team_{k}{r}" for k in state for r in ("", "_rev")}
    assert list(q["team_after_1"]["criteria"]) == ["white", "blue", "unclear"]
    assert "`after_1`" in q["team_after_1"]["instructions"]


def test_decode_makes_reads_the_crops_in_order() -> None:
    nouls = {f"in_net_{i}": 0.1 for i in range(12)} | {"in_net_10": 0.9, "hoop_visible": 0.97}
    resp = answer({"outcome": {"made": 0.3, "missed": 0.6, "unclear": 0.1},
                   "outcome_rev": {"made": 0.5, "missed": 0.4, "unclear": 0.1}}, nouls)
    out = probe.decode("makes", resp, {})
    assert out["outcome"] == "missed" and out["p"] == pytest.approx(0.5)
    assert out["in_net"].index(0.9) == 10 and out["in_net_pick"] == "made"


def test_decode_possession_names_teams_not_colours() -> None:
    jerseys = {Team.AWAY: "white", Team.HOME: "blue"}
    resp = answer({"holder": {"white": 0.2, "blue": 0.7, "nobody": 0.1},
                   "holder_rev": {"white": 0.2, "blue": 0.6, "nobody": 0.2},
                   "team": {"white": 0.1, "blue": 0.1, "unclear": 0.8},
                   "team_rev": {"white": 0.1, "blue": 0.2, "unclear": 0.7}}, {"ball_visible": 0.4})
    out = probe.decode("possession", resp, jerseys)
    assert out["holder"] == "home" and out["team"] == "unclear"
    assert set(out["probs_holder"]) == {"home", "away", "nobody"}


def test_decode_turnovers_takes_the_mean_over_each_sides_frames() -> None:
    jerseys = {Team.AWAY: "white", Team.HOME: "blue"}
    sure_home = {"white": 0.05, "blue": 0.9, "unclear": 0.05}
    weak_away = {"white": 0.45, "blue": 0.35, "unclear": 0.2}
    resp = answer({"team_before_1": sure_home, "team_before_1_rev": sure_home,
                   "team_before_2": weak_away, "team_before_2_rev": weak_away,
                   "team_after_1": weak_away, "team_after_1_rev": weak_away})
    out = probe.decode("turnovers", resp, jerseys)
    assert out["before"] == "home" and out["frames_before"] == ["home", "away"]  # 0.625 blue over two frames
    assert out["after"] == "away" and out["p_after"] == pytest.approx(0.45)


def test_score_turnovers_counts_vetoes_by_truth() -> None:
    def row(kind: str, before: str, after: str, p: float = 0.9) -> dict:
        return {"pbp_turnover": kind == "turnover", "pbp_change": kind != "none",
                "jev": {"before": before, "after": after, "p_before": p, "p_after": p}}

    rows = [row("turnover", "home", "away"), row("none", "home", "home"), row("none", "away", "away", 0.6),
            row("change", "home", "home"), row("none", "home", "unclear"), {"pbp_turnover": False}]
    out = probe.score_turnovers(rows)
    assert out["truth"] == {"turnover": 1, "no_change": 3, "other_change": 1}
    assert out["verdicts_any"]["no_change"] == {"changed": 0, "same": 2, "unsure": 1}
    assert out["verdicts_p>=0.85"]["no_change"] == {"changed": 0, "same": 1, "unsure": 2}
    assert out["verdicts_any"]["other_change"]["same"] == 1


def test_score_makes_compares_jev_and_the_engine_on_the_same_shots() -> None:
    def row(made: bool, jev: str, in_net: str, engine: str | None, p: float = 0.8) -> dict:
        return {"made": made, "engine": engine, "centred_on": "ball",
                "jev": {"outcome": jev, "in_net_pick": in_net, "p": p}}

    rows = [row(True, "made", "made", "made"), row(True, "missed", "made", None),
            row(False, "missed", "made", "missed"), row(False, "made", "missed", "missed", 0.6),
            row(True, "unclear", "missed", "made")]
    out = probe.score_makes(rows)
    assert out["jev_choice"]["answered"] == 4 and out["jev_choice"]["right"] == 2
    assert out["engine"]["coverage"] == 0.8
    assert out["jev_and_engine"] == {"both_answered": 3, "agree": 2, "accuracy_when_agree": 1.0,
                                     "jev_right_when_disagree": 0, "engine_right_when_disagree": 1}


def test_possession_and_turnovers_need_two_jersey_colours() -> None:
    with pytest.raises(SystemExit, match="--jerseys"):
        probe.main(["nowhere", "possession"])
    with pytest.raises(SystemExit, match="--jerseys"):
        probe.main(["nowhere", "turnovers", "--jerseys", "white,white"])
