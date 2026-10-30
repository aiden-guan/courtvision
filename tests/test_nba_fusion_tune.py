"""Calibration of the scorebug-fusion settings (nba/fusion_tune.py, tools/nba_calibrate_fusion.py) and
``nba_analyze.py fuse --fusion-file``, on synthetic games."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from courtvision.nba import fusion as fusion_mod
from courtvision.nba.espn import TeamBox
from courtvision.nba.evaluate import box_error, team_box
from courtvision.nba.fusion import FusionConfig, PlayDirection
from courtvision.nba.fusion_tune import (
    Evaluator,
    GameCase,
    GameScore,
    _pick,
    calibrate,
    config_key,
    fusion_config,
    overrides,
    resolve,
    score_game,
    search,
    space,
)
from courtvision.nba.scorebug_events import BugEvent
from courtvision.training.align_pbp import AlignedAction, PBPAction
from courtvision.types import Event, EventKind, Team

HOME, AWAY = Team.HOME, Team.AWAY
TOOLS = Path(__file__).resolve().parents[1] / "tools"


def load_tool(name: str):  # noqa: ANN201
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shot(sid: str, t_release: float, t_result: float, team: Team, made: bool = False, value: int = 2) -> list[Event]:
    """An engine shot: its attempt and result."""
    kind = EventKind.SHOT_MADE if made else EventKind.SHOT_MISSED
    return [Event(f"{sid}a", EventKind.SHOT_ATTEMPT, t_release, 0, team=team, value=value, shot_id=sid),
            Event(f"{sid}r", kind, t_result, 0, team=team, value=value, shot_id=sid)]


def truth_shot(i: int, t: float, team: Team, made: bool, value: int = 2) -> AlignedAction:
    """A play-by-play shot placed on video time ``t`` (its result; compare puts a make's release 1 s earlier)."""
    a = PBPAction(i, 1, 0.0, EventKind.SHOT_MADE if made else EventKind.SHOT_MISSED, team, f"{team.value}1",
                  value=value, made=made, attrs={"shot_type": "three" if value == 3 else "two"})
    return AlignedAction(a, t, t, "clock")


def truth_play(i: int, t: float, kind: EventKind, team: Team, **attrs: object) -> AlignedAction:
    """A play-by-play rebound, steal, turnover, ... of a player of ``team`` placed on video time ``t``."""
    return AlignedAction(PBPAction(i, 1, 0.0, kind, team, f"{team.value}2", attrs=dict(attrs)), t, t, "clock")


def aim_track(*blocks: tuple[float, float, float], hz: int = 10) -> PlayDirection:
    """The camera aiming at court x = ``x`` over [t0, t1) for each (t0, t1, x), sampled at ``hz``."""
    t = np.concatenate([np.arange(round(t0 * hz), round(t1 * hz)) / hz for t0, t1, _ in blocks])
    x = np.concatenate([np.full(round((t1 - t0) * hz), float(x)) for t0, t1, x in blocks])
    return PlayDirection(t, x)


def lag_game(name: str, lags: list[float], gap_s: float = 1.2) -> GameCase:
    """Made twos of the home team 30 s apart, each shown on the bug ``lags[i]`` after the ball went in. The engine
    calls every shot a miss: the make, and two real misses of the away team ``gap_s`` before and after it. Only a
    ``score_lag_s`` within ``gap_s / 2`` of a basket's lag makes the right shot that basket (``fuse`` takes the
    engine result nearest the bug time minus the lag)."""
    events: list[Event] = []
    bug: list[BugEvent] = []
    aligned: list[AlignedAction] = []
    for i, lag in enumerate(lags):
        t = 20.0 + 30.0 * i
        events += shot(f"m{i}", t - 1.0, t, HOME) + shot(f"b{i}", t - gap_s - 1.0, t - gap_s, AWAY)
        events += shot(f"a{i}", t + gap_s - 1.0, t + gap_s, AWAY)
        bug.append(BugEvent(t + lag, "score", HOME, 2, period=1))
        aligned += [truth_shot(3 * i, t, HOME, True), truth_shot(3 * i + 1, t - gap_s, AWAY, False),
                    truth_shot(3 * i + 2, t + gap_s, AWAY, False)]
    return GameCase(name, events, bug, aligned, window=(0.0, 30.0 * len(lags) + 20.0))


def camera_game(name: str, stay_s: float = 1.6, n: int = 4) -> GameCase:
    """Home attacks the right half (three baskets with the camera there tell ``fuse`` so). Then ``n`` times: a home
    miss, the away team's defensive rebound and the camera across half court for only ``stay_s`` (the ball is stolen
    straight back), and a home basket 10 s after the miss. Only an ``oreb_hold_s`` up to ``stay_s`` sees the crossing;
    a longer one takes the next home shot at the same basket for an offensive rebound."""
    right, left = 80.0, 15.0
    events: list[Event] = []
    bug: list[BugEvent] = []
    aligned: list[AlignedAction] = []
    blocks: list[tuple[float, float, float]] = []
    for i in range(3):
        t = 10.0 + 30.0 * i
        blocks += [(t - 10.0, t + 1.5, right), (t + 1.5, t + 20.0, left)]
        events += shot(f"h{i}", t - 1.0, t, HOME, made=True)
        bug.append(BugEvent(t + 2.0, "score", HOME, 2, period=1))
        aligned.append(truth_shot(len(aligned), t, HOME, True))
    for i in range(n):
        t = 100.0 + 40.0 * i
        blocks += [(t - 10.0, t + 1.5, right), (t + 1.5, t + 1.5 + stay_s, left), (t + 1.5 + stay_s, t + 11.5, right),
                   (t + 11.5, t + 30.0, left)]
        events += shot(f"m{i}", t - 1.0, t, HOME) + shot(f"p{i}", t + 9.0, t + 10.0, HOME, made=True)
        events += [Event(f"st{i}", EventKind.STEAL, t + 3.0, 0, team=HOME, player_id="home2"),
                   Event(f"to{i}", EventKind.TURNOVER, t + 3.0, 0, team=AWAY, player_id="away2")]
        bug.append(BugEvent(t + 12.0, "score", HOME, 2, period=1))
        aligned += [truth_shot(len(aligned), t, HOME, False),
                    truth_play(len(aligned) + 1, t + 1.0, EventKind.REBOUND, AWAY, offensive=False),
                    truth_play(len(aligned) + 2, t + 3.0, EventKind.STEAL, HOME),
                    truth_play(len(aligned) + 3, t + 3.0, EventKind.TURNOVER, AWAY),
                    truth_shot(len(aligned) + 4, t + 10.0, HOME, True)]
    return GameCase(name, events, bug, aligned, window=(0.0, 100.0 + 40.0 * n), direction=aim_track(*sorted(blocks)),
                    periods=[(0.0, 1)])


# --------------------------------------------------------------------------- scoring


def test_the_synthetic_game_needs_a_score_lag_within_half_the_gap_of_the_true_lag() -> None:
    game = lag_game("g", [3.0] * 10)
    err = {lag: score_game(game, FusionConfig(score_lag_s=lag)) for lag in (2.0, 2.25, 2.5, 3.0, 3.5)}
    assert [err[lag].abs_error for lag in (2.5, 3.0, 3.5)] == [0, 0, 0]
    assert err[2.0].abs_error == err[2.25].abs_error == 30  # per basket: home fga +1, away fga -1, a home oreb
    assert err[2.0].truth_total == 60 and err[2.0].diff["home.fga"] == 10 and err[2.0].diff["away.fga"] == -10
    assert err[3.0].event_f1 == pytest.approx(1.0) and err[2.0].event_f1 < 1.0


def test_game_score_of_a_whole_game_is_the_box_error_eval_reports() -> None:
    game = lag_game("g", [2.0] * 4)
    pred = fusion_mod.fuse(game.events, game.bug, FusionConfig())
    box = team_box(pred)
    truth = {HOME: TeamBox(HOME, "H", {**box[HOME], "fga": box[HOME]["fga"] + 2, "dreb": 5}),
             AWAY: TeamBox(AWAY, "A", {**box[AWAY], "pts": box[AWAY]["pts"] + 3})}
    whole = GameCase("g", game.events, game.bug, game.aligned, truth_box=truth, window=None)
    s = score_game(whole, FusionConfig())
    cmp = box_error(box, truth)
    assert (s.abs_error, s.truth_total) == (10, sum(r["truth"] for r in cmp.rows))
    assert round(s.box_error, 4) == cmp.error
    assert s.diff["home.fga"] == -2 and s.diff["home.dreb"] == -5 and s.diff["away.pts"] == -3


# --------------------------------------------------------------------------- the search


def test_search_recovers_a_score_lag_that_is_off() -> None:
    rep = calibrate([lag_game("g", [3.0] * 10)])  # the true lag is 3.0 s, the default 2.0 s
    # 2.5 .. 3.5 all give a perfect box: the middle of that run is taken; no other parameter moves
    assert rep["config"] == {"score_lag_s": 3.0}
    assert rep["before"]["summed"]["box_error"] == 0.5 and rep["after"]["summed"]["box_error"] == 0.0
    assert rep["after"]["per_game"]["g"]["event_f1"] == 1.0
    assert rep["objective"]["after"] < rep["objective"]["before"]
    s = rep["search"]
    assert s["in_sample"] is True and "in-sample" in s["note"] and rep["leave_one_out"] == {}
    assert s["converged"] and not s["budget_exhausted"] and s["moves"][0]["param"] == "score_lag_s"
    assert s["params"] == [p.name for p in space()]


def test_search_recovers_how_long_the_camera_must_stay_across_after_a_defensive_rebound() -> None:
    game = camera_game("cam")  # the camera crosses for 1.6 s; the default oreb_hold_s is 2.0 s
    before = score_game(game, FusionConfig())
    assert {k: v for k, v in before.diff.items() if v} == {"home.oreb": 4, "away.dreb": -4}
    assert before.event_f1 == pytest.approx(1.0)  # compare matches rebounds in time only: only the box sees it
    rep = calibrate([game])
    assert rep["config"] == {"oreb_hold_s": 1.5}  # 1.0 and 1.5 both see the crossing: the one nearer the default
    assert rep["after"]["summed"]["box_error"] == 0.0 and rep["before"]["summed"]["abs_error"] == 8
    assert [m["param"] for m in rep["search"]["moves"]] == ["oreb_hold_s"]


def test_a_game_the_defaults_fit_keeps_the_defaults() -> None:
    rep = calibrate([lag_game("g", [2.0] * 10)])
    assert rep["config"] == {} and rep["search"]["moves"] == []
    assert rep["before"] == rep["after"]


def test_a_change_must_beat_the_margin() -> None:
    game = lag_game("g", [2.0] * 19 + [3.0])  # one basket the bug shows late: 3 box counts of 120 at the default
    gain = score_game(game, FusionConfig()).box_error - score_game(game, FusionConfig(score_lag_s=2.5)).box_error
    assert gain == pytest.approx(3 / 120)
    assert calibrate([game], margin=0.03)["config"] == {}
    assert calibrate([game], margin=0.02)["config"] == {"score_lag_s": 2.5}  # the only lag right for all 20


def test_tied_values_take_the_middle_of_the_tied_run_nearest_the_default() -> None:
    grid = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]
    res = {v: (0.0 if v >= 2.5 else 1.0) for v in grid}
    assert _pick(res, grid, 2.0) == 3.0
    res = {v: (0.0 if v <= 1.75 else 1.0) for v in grid}  # an even run: the middle value nearer the default
    assert _pick(res, grid, 2.0) == 1.5
    res = {v: (0.0 if v in (1.0, 1.25, 3.0, 3.25, 3.5) else 1.0) for v in grid}  # two runs: the one nearer 2.0
    assert _pick(res, grid, 2.0) == 1.25
    res = {v: (0.0 if v in (1.0, 1.25, 2.75, 3.0, 3.25) else 1.0) for v in grid}  # as near: the longer run
    assert _pick(res, grid, 2.0) == 3.0
    assert _pick({False: 1.0, True: 0.5}, [False, True], False) is True


def fake_scorer(errors) -> object:  # noqa: ANN001
    """A scorer whose box error (of 100 truth counts) is ``errors(config)``."""
    return lambda case, cfg: GameScore(errors(cfg), 100, 0.0, 0)


def test_a_change_made_redundant_by_a_later_one_is_undone() -> None:
    # score_lag_s 3.0 halves the error, assist_window_s 5.0 removes it whatever the lag: the lag goes back
    errors = fake_scorer(lambda c: 0 if c.assist_window_s == 5.0 else (5 if c.score_lag_s == 3.0 else 10))
    rep = calibrate([lag_game("g", [2.0])], params=["score_lag_s", "assist_window_s"], scorer=errors)
    s = rep["search"]
    assert [m["param"] for m in s["moves"]] == ["score_lag_s", "assist_window_s"]
    assert [(r["param"], r["to"]) for r in s["reverted"]] == [("score_lag_s", 2.0)]
    assert rep["config"] == {"assist_window_s": 5.0} and rep["after"]["summed"]["box_error"] == 0.0


def test_the_budget_stops_the_search() -> None:
    ev = Evaluator([lag_game("g", [3.0] * 4)])
    res = search(ev, ["g"], resolve(["score_lag_s", "make_window_s"]), budget=4)
    assert res.budget_exhausted and not res.converged and res.evaluations == 4
    # the defaults, then the nearest lags first: 1.75, 2.25 and 1.5; the budget ends before 2.5, the first that works
    assert res.overrides == {} and res.moves == []


def test_params_restrict_the_search_and_name_both_ends_of_a_window() -> None:
    assert [p.name for p in resolve(["make_window_s", "rim_misses"])] == ["make_window_s.lo", "make_window_s.hi",
                                                                         "rim_misses"]
    assert [p.name for p in resolve(["reset_window_s.hi"])] == ["reset_window_s.hi"]
    with pytest.raises(ValueError, match="unknown fusion parameter"):
        resolve(["score_lag"])
    rep = calibrate([lag_game("g", [3.0] * 10)], params=["assist_window_s", "rim_misses"])
    assert rep["config"] == {} and rep["search"]["params"] == ["assist_window_s", "rim_misses"]


def test_every_grid_holds_its_default_within_a_range_and_switches_are_searched() -> None:
    names = {p.name for p in space()}
    assert {"score_lag_s", "make_window_s.lo", "make_window_s.hi", "makes", "handler_values"} <= names
    for p in space():
        g = p.grid()
        assert p.default() in g
        if not p.switch:
            assert g == sorted(g) and min(g) <= p.default() <= max(g) and len(g) >= 5
    lag = next(p for p in space() if p.name == "score_lag_s")
    assert (lag.grid()[0], lag.grid()[-1]) == (1.0, 3.5)


def test_overrides_keep_only_changed_fields_and_round_trip_through_json() -> None:
    point = {"score_lag_s": 2.0, "make_window_s.lo": -3.0, "make_window_s.hi": 1.5, "makes": True, "rim_misses": False}
    over = overrides(point)
    assert over == {"make_window_s": (-3.0, 1.5), "rim_misses": False}
    cfg = fusion_config(json.loads(json.dumps(over)))  # JSON makes the window a list
    assert cfg.make_window_s == (-3.0, 1.5) and cfg.rim_misses is False and cfg.score_lag_s == 2.0
    assert config_key(json.loads(json.dumps(over))) == config_key(over)
    with pytest.raises(ValueError, match="unknown FusionConfig"):
        fusion_config({"score_lag": 2.5})


def test_several_workers_give_the_same_calibration() -> None:
    games = [lag_game("a", [3.0] * 4), lag_game("b", [2.0] * 4)]
    one = calibrate(games, params=["score_lag_s", "make_window_s"])
    two = calibrate(games, params=["score_lag_s", "make_window_s"], workers=2)
    for key in ("config", "before", "after", "leave_one_out"):
        assert one[key] == two[key]
    assert two["search"]["workers"] == 2 and one["search"]["moves"] == two["search"]["moves"]


# --------------------------------------------------------------------------- leave one game out


def test_leave_one_out_scores_each_game_with_the_setting_tuned_on_the_others() -> None:
    games = [lag_game("late1", [3.0] * 10), lag_game("late2", [3.0] * 10), lag_game("ontime", [2.0] * 10)]
    rep = calibrate(games)
    assert rep["config"] == {"score_lag_s": 2.5}  # the one lag right for all three games
    assert rep["after"]["summed"]["box_error"] == 0.0
    loo = rep["search"]["leave_one_out"]["per_game"]
    assert loo["late1"]["config"] == {"score_lag_s": 2.5}  # tuned on late2 + ontime
    assert loo["ontime"]["config"] == {"score_lag_s": 3.0}  # tuned on the late games alone: wrong for ontime
    assert rep["leave_one_out"] == {"late1": 0.0, "late2": 0.0, "ontime": round(20 / 60, 4)}
    assert [loo[g]["default"]["box_error"] for g in ("late1", "late2", "ontime")] == [0.5, 0.5, 0.0]
    assert rep["search"]["leave_one_out"]["verdict"].startswith("with the settings tuned without them the held-out "
                                                                "games did better")
    assert rep["search"]["in_sample"] is False and not rep["search"]["loo_guard"]["applied"]


def test_the_leave_one_out_guard_keeps_the_defaults_when_held_out_games_do_worse() -> None:
    games = [lag_game("late", [3.0] * 10), lag_game("ontime", [2.0] * 10)]
    params = ["score_lag_s", "make_window_s", "bug_rebound_window_s", "bug_rebounds"]
    plain = calibrate(games, params)
    assert plain["config"] == {"score_lag_s": 2.5}
    # each game alone: "late" moves the lag to 3.0, which costs "ontime" 20 counts; "ontime" keeps the default
    assert plain["leave_one_out"] == {"late": 0.5, "ontime": round(20 / 60, 4)}
    assert "did worse" in plain["search"]["leave_one_out"]["verdict"]
    guarded = calibrate(games, params, loo_guard=True)
    assert guarded["config"] == {} and guarded["search"]["loo_guard"]["applied"]
    assert guarded["search"]["tuned_config"] == {"score_lag_s": 2.5} and guarded["after"] == guarded["before"]


# --------------------------------------------------------------------------- the CLI and nba_analyze.py fuse


def write_game(root: Path, lag_s: float, n: int = 6, tag: str = "q1") -> None:
    """A game folder as the pipeline leaves it (no video): the bug read at 3 Hz with a running game clock in period
    1, the home team's ``n`` baskets of ``lag_game`` (the bug shows each ``lag_s`` after it), the engine's events of
    the run, a camera aimed at half court and the ESPN summary with the shots."""
    game = lag_game("x", [lag_s] * n)
    end = 30.0 * n + 30.0
    (root / "camera_parts").mkdir(parents=True)
    (root / "runs" / tag).mkdir(parents=True)
    (root / "game.json").write_text(json.dumps({"video": "none.mp4", "espn_event": "0"}))
    cam = [{"header": True, "center": [47.0, -94.0, 36.0], "image_size": [1920, 1080], "fps": 30.0}]
    cam += [{"frame": f, "params": {"pan": 0.0, "tilt": 20.0, "roll": 0.0, "f": 3000.0}} for f in range(0, 30 * int(end), 15)]
    (root / "camera_parts" / "part0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cam))
    shown = [b.t for b in game.bug]
    reads = [{"t": k / 3, "home_score": 2 * sum(1 for s in shown if s <= k / 3 + 1e-9), "away_score": 0, "period": 1,
              "game_clock": float(720 - int(k / 3)), "shot_clock": None} for k in range(int(3 * end))]
    (root / "scorebug.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reads))
    (root / "runs" / tag / "run.json").write_text(json.dumps({"start": 0.0, "end": end}))
    (root / "runs" / tag / "events.jsonl").write_text("".join(json.dumps(e.to_dict()) + "\n" for e in game.events))

    def clock(t: float) -> str:
        c = int(720 - t)
        return f"{c // 60}:{c % 60:02d}"

    plays = []
    for al in game.aligned:
        a = al.action
        made = a.kind is EventKind.SHOT_MADE
        plays.append({"sequenceNumber": str(a.action_number + 1), "type": {"text": "Jump Shot"},
                      "text": f"{'B' if a.team is HOME else 'A'} One {'makes' if made else 'misses'} 15-foot jumper",
                      "period": {"number": 1}, "clock": {"displayValue": clock(float(al.t))}, "shootingPlay": True,
                      "scoringPlay": made, "pointsAttempted": 2, "team": {"id": "2" if a.team is HOME else "1"},
                      "participants": [{"athlete": {"id": "21" if a.team is HOME else "11"}}]})
    stats = [{"name": "fieldGoalsMade-fieldGoalsAttempted", "displayValue": "6-18"},
             {"name": "threePointFieldGoalsMade-threePointFieldGoalsAttempted", "displayValue": "0-0"},
             {"name": "freeThrowsMade-freeThrowsAttempted", "displayValue": "0-0"}]
    stats += [{"name": k, "displayValue": "0"} for k in ("offensiveRebounds", "defensiveRebounds", "totalRebounds",
                                                         "assists", "steals", "blocks", "turnovers", "fouls")]
    labels = ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO", "STL", "BLK", "OREB", "DREB", "PF", "+/-"]
    line = ["30", "0", "0-0", "0-0", "0-0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]

    def player(pid: str) -> dict:  # type: ignore[type-arg]
        return {"athlete": {"id": pid, "displayName": f"P{pid}", "jersey": pid}, "starter": True, "stats": line}

    summary = {"header": {"competitions": [{"competitors": [
        {"homeAway": "away", "team": {"id": "1", "abbreviation": "AAA"}},
        {"homeAway": "home", "team": {"id": "2", "abbreviation": "BBB"}}]}]},
        "plays": plays,
        "boxscore": {"teams": [{"team": {"id": "1"}, "statistics": stats}, {"team": {"id": "2"}, "statistics": stats}],
                     "players": [{"team": {"id": tid}, "statistics": [{"labels": labels, "athletes": [player(pid)]}]}
                                 for tid, pid in (("1", "11"), ("2", "21"))]}}
    (root / "espn_summary.json").write_text(json.dumps(summary))


def test_cli_writes_the_calibration_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for name in ("g1", "g2"):
        write_game(tmp_path / name, lag_s=3.0)
    out = tmp_path / "calib" / "fusion_q1.json"
    tool = load_tool("nba_calibrate_fusion")
    argv = [str(tmp_path / "g1"), str(tmp_path / "g2"), "--tag", "q1", "--out", str(out), "--params",
            "score_lag_s,make_window_s,rim_misses", "--budget", "60"]
    assert tool.main(argv) == 0  # default --workers: a pool of processes
    rep = json.loads(out.read_text())
    assert list(rep) == ["config", "games", "tag", "objective", "before", "after", "leave_one_out", "search",
                         "metrics"]
    assert rep["games"] == ["g1", "g2"] and rep["tag"] == "q1"
    assert rep["metrics"] == {"games": 2, "in_sample": False, "changed": ["score_lag_s"],
                              "box_error_before": rep["before"]["summed"]["box_error"], "box_error_after": 0.0,
                              "box_error_held_out": 0.0, "event_f1_before": rep["before"]["summed"]["event_f1"],
                              "event_f1_after": rep["after"]["summed"]["event_f1"]}
    assert rep["search"]["workers"] > 1 or (os.cpu_count() or 1) == 1
    assert rep["config"] == {"score_lag_s": 3.0}  # the bug of both games shows the baskets 3 s late
    for block in (rep["before"], rep["after"]):
        assert set(block) == {"per_game", "summed"} and set(block["per_game"]) == {"g1", "g2"}
        for s in (*block["per_game"].values(), block["summed"]):
            assert {"box_error", "abs_error", "truth_total", "event_f1", "objective"} <= set(s)
            assert isinstance(s["box_error"], float) and s["truth_total"] == (36 if s is not block["summed"] else 72)
    assert rep["after"]["summed"]["box_error"] < rep["before"]["summed"]["box_error"]
    assert set(rep["leave_one_out"]) == {"g1", "g2"}
    assert all(isinstance(v, float) for v in rep["leave_one_out"].values())
    assert {"f1_weight", "margin", "before", "after", "formula"} <= set(rep["objective"])
    s = rep["search"]
    assert s["params"] == ["score_lag_s", "make_window_s.lo", "make_window_s.hi", "rim_misses"]
    assert s["space"]["score_lag_s"]["range"] == [1.0, 3.5] and s["budget"] == 60
    assert s["roots"] == {"g1": str(tmp_path / "g1"), "g2": str(tmp_path / "g2")}
    assert s["in_sample"] is False and set(s["leave_one_out"]["per_game"]) == {"g1", "g2"}
    printed = capsys.readouterr().out
    assert "| g1 |" in printed and '"score_lag_s": 3.0' in printed

    # one game: in-sample, said plainly; no held-out errors
    one = tmp_path / "one.json"
    argv = [str(tmp_path / "g1"), "--tag", "q1", "--out", str(one), "--params", "score_lag_s", "--workers", "1"]
    assert tool.main(argv) == 0
    rep1 = json.loads(one.read_text())
    assert rep1["leave_one_out"] == {} and rep1["search"]["in_sample"] is True and rep1["search"]["workers"] == 1
    assert rep1["metrics"]["in_sample"] is True and rep1["metrics"]["box_error_held_out"] is None
    assert "in-sample" in capsys.readouterr().out


def run_fuse(tool, root: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> FusionConfig:  # noqa: ANN001
    """``nba_analyze.py <root> fuse --tag q1 <extra>``; the FusionConfig it fused with."""
    seen: list[FusionConfig] = []
    real = fusion_mod.fuse

    def spy(events, bug, config=None, *args, **kw):  # noqa: ANN001, ANN002, ANN003, ANN202
        seen.append(config)
        return real(events, bug, config, *args, **kw)

    monkeypatch.setattr(fusion_mod, "fuse", spy)
    monkeypatch.setattr(sys, "argv", ["nba_analyze.py", str(root), "fuse", "--tag", "q1", *extra])
    assert tool.main() == 0
    assert len(seen) == 1
    return seen[0]


def test_fuse_applies_the_config_of_a_fusion_file_with_fusion_overrides_on_top(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "g"
    write_game(root, lag_s=3.0)
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps({"config": {"score_lag_s": 3.0, "make_window_s": [-3.0, 2.0], "rim_misses": False},
                                 "games": ["g"], "tag": "q1"}))
    tool = load_tool("nba_analyze")
    cfg = run_fuse(tool, root, monkeypatch, "--fusion-file", str(calib))
    assert (cfg.score_lag_s, cfg.make_window_s, cfg.rim_misses) == (3.0, (-3.0, 2.0), False)
    assert cfg.assist_window_s == FusionConfig().assist_window_s
    fused = [json.loads(line) for line in (root / "runs" / "q1" / "fused.jsonl").open()]
    made = [e for e in fused if e["kind"] == "shot_made"]
    assert sorted(e["event_id"] for e in made) == [f"m{i}r" for i in range(6)]  # the right shots, with the 3 s lag

    cfg = run_fuse(tool, root, monkeypatch, "--fusion-file", str(calib), "--fusion",
                   '{"score_lag_s": 2.25, "assist_window_s": 2.0}')
    assert (cfg.score_lag_s, cfg.assist_window_s, cfg.make_window_s) == (2.25, 2.0, (-3.0, 2.0))
    assert run_fuse(tool, root, monkeypatch, "--fusion", '{"score_lag_s": 2.5}').score_lag_s == 2.5
    assert run_fuse(tool, root, monkeypatch) == FusionConfig()

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"score_lag_s": 3.0}))
    with pytest.raises(SystemExit, match="no \"config\" object"):
        run_fuse(tool, root, monkeypatch, "--fusion-file", str(bad))
