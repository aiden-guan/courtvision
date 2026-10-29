"""The team possession timeline (courtvision/nba/timeline.py) on synthetic games: where each team attacks, the
decoder, fitting on play-by-play, the events rebuilt from it, and ``nba_analyze.py timeline``."""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from courtvision.geometry.court import court_for
from courtvision.nba import timeline as tlm
from courtvision.nba.fusion import PlayDirection
from courtvision.perception.ball3d import G_FTS2, Flight3D
from courtvision.training.align_pbp import AlignedAction, PBPAction
from courtvision.types import Event, EventKind, Team

HOME, AWAY = Team.HOME, Team.AWAY
K = EventKind
NBA = court_for("nba")
TOOLS = Path(__file__).resolve().parents[1] / "tools"
_ids = itertools.count()


def aim_track(*blocks: tuple[float, float, float], hz: int = 10) -> PlayDirection:
    """The camera aiming at court x = ``x`` over [t0, t1) for each (t0, t1, x), sampled at ``hz``."""
    t = np.concatenate([np.arange(round(t0 * hz), round(t1 * hz)) / hz for t0, t1, _ in blocks])
    x = np.concatenate([np.full(round((t1 - t0) * hz), float(x)) for t0, t1, x in blocks])
    return PlayDirection(t, x)


def ev(kind: EventKind, t: float, team: Team | None = None, **kw: object) -> Event:
    return Event(f"e{next(_ids)}", kind, t, int(t * 30), team=team, **kw)  # type: ignore[arg-type]


def dribbles(team: Team, t0: float, t1: float) -> list[Event]:
    return [ev(K.DRIBBLE, float(t), team) for t in np.arange(t0, t1, 1.0)]


# Home attacks the left half (x < 47) all game. The camera follows the ball.
CAMERA = ((0, 20, 20.0),  # home in its front court
          (20, 26, 25.0),  # home scores at 20 s; away inbounds in its back court (the same half)
          (26, 28, 47.0),  # away brings the ball over half court
          (28, 51, 75.0),  # away in its front court; misses at 50 s, home rebounds at 51 s
          (51, 54, 70.0), (54, 56, 47.0), (56, 80, 20.0),  # home brings it up and attacks
          (80, 82, 47.0), (82, 100, 75.0))  # away steals at 80 s and goes at once
CHANGES = [(20.0, HOME, AWAY), (51.0, AWAY, HOME), (80.0, HOME, AWAY)]
MARKS: list[tuple[float, Team | None]] = [(0.0, HOME), (20.0, AWAY), (51.0, HOME), (80.0, AWAY)]


def game() -> tuple[PlayDirection, list[Event]]:
    """The camera above, and the engine's events: its team labels are wrong for the made shot and the rebound, and
    it reports a steal at 40 s that never happened."""
    events = [*dribbles(HOME, 1, 19), ev(K.SHOT_ATTEMPT, 19.0, HOME, value=2, shot_id="s1"),
              ev(K.SHOT_MADE, 20.0, AWAY, value=2, shot_id="s1"),
              *dribbles(AWAY, 21, 49), ev(K.STEAL, 40.0, HOME), ev(K.TURNOVER, 40.0, AWAY),
              ev(K.SHOT_ATTEMPT, 49.0, AWAY, value=3, shot_id="s2"), ev(K.SHOT_MISSED, 50.0, AWAY, value=3, shot_id="s2"),
              ev(K.REBOUND, 51.0, AWAY), *dribbles(HOME, 52, 79), ev(K.STEAL, 80.5, AWAY), *dribbles(AWAY, 82, 99)]
    return aim_track(*CAMERA), events


def manual(state: list[tuple[float, float, Team]], tracked: list[tuple[float, float]] | None = None,
           dt: float = 0.25) -> tlm.Timeline:
    """A timeline from (t0, t1, team) runs; ``tracked``: the (t0, t1) stretches with the camera on the court
    (default: all)."""
    t = np.arange(0.0, max(b for _, b, _ in state), dt)
    st = np.zeros(len(t), np.int8)
    for a, b, team in state:
        st[(t >= a) & (t < b)] = tlm.TEAMS.index(team)
    tr = np.ones(len(t), bool) if tracked is None else np.zeros(len(t), bool)
    for a, b in tracked or []:
        tr[(t >= a) & (t < b)] = True
    return tlm.Timeline(t, st, tlm.AttackMap("left"), tr)


# --------------------------------------------------------------------------- where each team attacks


def test_the_attack_map_is_voted_from_the_engines_noisy_team_cues_and_finds_the_half_time_swap() -> None:
    rng = np.random.default_rng(0)
    direction = aim_track((0, 600, 20.0), (600, 1200, 75.0), (1200, 1500, 20.0), (1500, 2100, 75.0), (2100, 2700, 20.0))
    cues = []
    for t in np.arange(1.0, 2699.0, 2.0):
        home_left = t < 1300  # home attacks left in the first half, right in the second
        aim_left = direction.side(t + 3.0) == "left"
        team = HOME if aim_left == home_left else AWAY
        if rng.random() < 0.25:
            team = AWAY if team is HOME else HOME  # the engine's team labels are often wrong
        cues.append(ev(K.DRIBBLE, float(t), team))
    attack = tlm.attack_map(cues, direction)
    assert attack.home_first == "left"
    assert 1199.0 <= attack.switch_t <= 1501.0  # in the break at half time (no cue between 1200 and 1500 s on court)
    assert attack.side(HOME, 100.0) == "left" and attack.side(AWAY, 100.0) == "right"
    assert attack.side(HOME, 2000.0) == "right"
    one_half = tlm.attack_map([e for e in cues if e.t < 1200], direction)
    assert one_half == tlm.AttackMap("left")  # no swap seen: none assumed


def test_the_attack_map_ignores_the_engines_scorebug_events() -> None:
    direction = aim_track((0, 100, 20.0))
    cues = [ev(K.POSSESSION, float(t), AWAY, source="scoreboard") for t in range(1, 90)] + \
        [ev(K.DRIBBLE, float(t), HOME) for t in range(1, 30)]
    assert tlm.attack_map(cues, direction).home_first == "left"


# --------------------------------------------------------------------------- candidates and decoding


def test_rim_arrivals_find_flights_that_come_down_at_a_rim() -> None:
    rx, ry = NBA.basket_xy("left")
    # from (20, 25, 8) ft, through the rim centre at rim height one second later, on the way down
    v0 = np.array([rx - 20.0, ry - 25.0, NBA.rim_height - 8.0 + 0.5 * G_FTS2])
    shot = Flight3D(100.0, 100.6, np.array([20.0, 25.0, 8.0]), v0, 1.0, 9, (3000, 3018))
    pass_ = Flight3D(200.0, 200.5, np.array([20.0, 25.0, 5.0]), np.array([20.0, 0.0, 4.0]), 1.0, 9, (6000, 6015))
    assert tlm.rim_arrivals([shot, pass_], NBA) == [pytest.approx(101.0)]
    cands = tlm.candidate_times([ev(K.SHOT_MISSED, 5.0), ev(K.REBOUND, 6.0), ev(K.STEAL, 7.0)], [shot], NBA)
    assert cands["result"].tolist() == [5.0] and cands["rebound"].tolist() == [6.0]
    assert cands["loss"].tolist() == [7.0] and cands["rim"].tolist() == [pytest.approx(101.0)]


def test_viterbi_changes_state_where_the_transition_allows_it() -> None:
    ll = np.array([[0.0, -1.0]] * 5 + [[-1.0, 0.0]] * 5)
    rigid = np.log([[1 - 1e-6, 1e-6], [1e-6, 1 - 1e-6]])
    loose = np.log([[0.9, 0.1], [0.1, 0.9]])
    which = np.zeros(10, int)
    which[7] = 1
    assert tlm.viterbi(ll, [rigid, loose], which).tolist() == [0] * 7 + [1] * 3


def test_decode_follows_the_camera_and_changes_hands_at_the_basket_not_at_the_crossing() -> None:
    direction, events = game()
    tl = tlm.decode(events, direction, flights=[], court=NBA)
    assert tl.attack == tlm.AttackMap("left")
    got = tl.changes()
    assert [(lost, gained) for _, lost, gained in got] == [(lost, gained) for _, lost, gained in CHANGES]
    for (t, _, _), (want, _, _) in zip(got, CHANGES, strict=True):
        assert abs(t - want) <= 1.0  # at the basket, rebound and steal (the camera crosses up to 8 s later)
    assert [tl.team_at(t) for t in (10.0, 23.0, 35.0, 53.0, 65.0, 90.0)] == [HOME, AWAY, AWAY, HOME, HOME, AWAY]
    assert tl.phase is not None and tl.phase[int(23.0 / 0.25) - int(tl.t[0] / 0.25)] == tlm.BACK  # away inbounding


def test_the_engines_scorebug_events_are_not_evidence() -> None:
    direction, events = game()
    events = [e for e in events if e.kind is not K.SHOT_MADE] + [ev(K.SHOT_MADE, 20.0, HOME, value=2, source="scoreboard")]
    tl = tlm.decode(events, direction, court=NBA)
    first = tl.changes()[0][0]
    assert first > 21.0  # with no basket the camera saw, the change waits for the next evidence


def test_truth_marks_from_play_by_play() -> None:
    def act(kind: EventKind, t: float, team: Team, made: bool | None = None, value: int = 2, **attrs: object) -> AlignedAction:
        return AlignedAction(PBPAction(next(_ids), 1, 600.0, kind, team=team, made=made, value=value, attrs=dict(attrs)),
                             t, t, "clock")

    ft = {"value": 1, "shot_type": "free_throw"}
    aligned = [act(K.SHOT_MADE, 10.0, HOME, True), act(K.SHOT_MISSED, 30.0, AWAY, False),
               act(K.REBOUND, 31.0, HOME, offensive=False), act(K.TURNOVER, 50.0, HOME),
               act(K.SHOT_MADE, 70.0, AWAY, True, ft_index=1, ft_total=2, **ft),
               act(K.SHOT_MADE, 71.0, AWAY, True, ft_index=2, ft_total=2, **ft),
               act(K.SHOT_MADE, 80.0, HOME, True, ft_index=1, ft_total=1, technical=True, **ft)]
    marks = tlm.truth_marks(aligned)
    teams = tlm.truth_teams(marks, np.array([5.0, 20.0, 30.5, 40.0, 60.0, 70.5, 75.0, 85.0]))
    assert teams == [None, AWAY, None, HOME, AWAY, AWAY, HOME, HOME]  # nothing known before the first action


def test_score_counts_settled_steps_and_matches_changes_of_the_same_team() -> None:
    t = np.arange(0.0, 100.0, 0.25)
    right = [HOME if x < 20.5 else AWAY if x < 51 else HOME if x < 80 else AWAY for x in t]  # changes 0.5 s late
    s = tlm.score(right, t, MARKS, np.ones(len(t), bool))
    assert s["accuracy"] == 1.0 and s["changes_matched"] == 3 and s["precision"] == s["recall"] == 1.0
    wrong = [HOME if x < 20.5 else AWAY if x < 60 else HOME if x < 70 else AWAY for x in t]  # a false return
    s = tlm.score(wrong, t, MARKS, np.ones(len(t), bool))
    assert s["changes_matched"] == 1 and s["changes_pred"] == 3 and s["accuracy"] < 0.85


def test_fit_counts_the_probabilities_and_the_fitted_decoder_finds_the_changes() -> None:
    direction, events = game()
    grid = tlm.grid_for(events, direction, 0.25)
    every = np.ones(len(grid), bool)
    cfg = tlm.fit(events, direction, MARKS, grid, every, every, court=NBA, leads=(1.0,), back_means=(3.5,),
                  cam_weights=(0.25,), base_scales=(1.0,), engine_weights=(0.0, 0.1), front_after_s=5.0)
    assert cfg.cam_p > 0.9
    assert cfg.switch["result"] > cfg.switch_base and cfg.switch["loss"] > cfg.switch_base
    tl = tlm.decode(events, direction, cfg, court=NBA, grid=grid)
    assert tlm.score(tl.teams(), grid, MARKS, every)["changes_matched"] == 3


# --------------------------------------------------------------------------- events from the timeline


def kinds(events: list[Event], kind: EventKind) -> list[Event]:
    return [e for e in events if e.kind is kind]


def test_derive_rebuilds_rebounds_turnovers_and_steals_from_the_changes() -> None:
    direction, events = game()
    out = tlm.derive_events(events, tlm.decode(events, direction, court=NBA))
    made = kinds(out, K.SHOT_MADE)[0]
    assert made.team is HOME and made.attrs["timeline"] == "shot_team"  # the engine said away
    assert kinds(out, K.SHOT_ATTEMPT)[0].team is HOME
    [reb] = kinds(out, K.REBOUND)
    assert (reb.t, reb.team, reb.attrs["offensive"]) == (51.0, HOME, False)  # the engine's rebound, re-teamed
    [tov] = kinds(out, K.TURNOVER)
    assert tov.team is HOME and abs(tov.t - 80.5) < 1e-9 and tov.attrs["inferred"]
    [stl] = kinds(out, K.STEAL)
    assert (stl.t, stl.team) == (80.5, AWAY)  # the engine's steal at the change; the one at 40 s is gone
    causes = [(round(e.t), e.team, e.attrs["cause"]) for e in kinds(out, K.POSSESSION)]
    assert [c[2] for c in causes] == ["made_shot", "defensive_rebound", "steal"]
    assert all(e.team is AWAY for e in kinds(out, K.DRIBBLE) if 30 < e.t < 45)


def test_a_turnover_with_a_cut_after_it_is_no_steal_and_a_change_after_a_break_is_no_turnover() -> None:
    events = [ev(K.DRIBBLE, 5.0, HOME)]
    cut = manual([(0, 30, HOME), (30, 60, AWAY)], tracked=[(0, 31), (45, 60)])  # a replay right after it
    out = tlm.derive_events(events, cut)
    assert [(e.team, e.t) for e in kinds(out, K.TURNOVER)] == [(HOME, 30.0)] and not kinds(out, K.STEAL)
    restart = manual([(0, 30, HOME), (30, 60, AWAY)], tracked=[(0, 10), (29.5, 60)])  # off the court 10-29.5 s
    out = tlm.derive_events(events, restart)
    assert not kinds(out, K.TURNOVER) and kinds(out, K.POSSESSION)[0].attrs["cause"] == "restart"


def test_a_change_after_the_teams_attempt_is_a_shot_whose_result_was_not_seen() -> None:
    events = [ev(K.SHOT_ATTEMPT, 25.0, HOME, value=2)]
    out = tlm.derive_events(events, manual([(0, 30, HOME), (30, 60, AWAY)]))
    assert not kinds(out, K.TURNOVER) and kinds(out, K.POSSESSION)[0].attrs["cause"] == "shot_unseen"


def test_a_change_after_a_flight_reaches_the_rim_the_team_attacks_is_a_shot_not_a_turnover() -> None:
    rx, ry = NBA.basket_xy("left")  # home attacks the left rim (AttackMap("left"))
    v0 = np.array([rx - 20.0, ry - 25.0, NBA.rim_height - 8.0 + 0.5 * G_FTS2])
    shot = Flight3D(26.0, 26.6, np.array([20.0, 25.0, 8.0]), v0, 1.0, 9, (780, 798))  # at the rim at 27 s
    timeline = manual([(0, 30, HOME), (30, 60, AWAY)])
    out = tlm.derive_events([ev(K.DRIBBLE, 5.0, HOME)], timeline, flights=[shot], court=NBA)
    assert not kinds(out, K.TURNOVER) and kinds(out, K.POSSESSION)[0].attrs["cause"] == "shot_unseen"
    away_rim = Flight3D(26.0, 26.6, np.array([74.0, 25.0, 8.0]), v0 * np.array([-1, 1, 1]), 1.0, 9, (780, 798))
    out = tlm.derive_events([ev(K.DRIBBLE, 5.0, HOME)], timeline, flights=[away_rim], court=NBA)
    assert kinds(out, K.TURNOVER)  # a flight at the other rim explains nothing


def test_a_miss_kept_by_the_shooting_team_is_an_offensive_rebound_only_where_the_engine_saw_one() -> None:
    events = [ev(K.SHOT_MISSED, 20.0, HOME, value=2), ev(K.REBOUND, 21.0, AWAY),
              ev(K.SHOT_MISSED, 40.0, HOME, value=2), ev(K.REBOUND, 90.0, AWAY)]
    out = tlm.derive_events(events, manual([(0, 60, HOME)]))
    [reb] = kinds(out, K.REBOUND)
    assert (reb.t, reb.team, reb.attrs["offensive"]) == (21.0, HOME, True)  # the one 50 s after nothing is dropped


def test_a_miss_then_the_same_teams_next_attempt_is_a_put_back_after_an_offensive_rebound() -> None:
    events = [ev(K.SHOT_MISSED, 20.0, HOME, value=2), ev(K.SHOT_ATTEMPT, 23.0, HOME, value=2),
              ev(K.SHOT_MISSED, 40.0, HOME, value=2), ev(K.SHOT_ATTEMPT, 48.0, HOME, value=2)]
    out = tlm.derive_events(events, manual([(0, 60, HOME)]))
    [reb] = kinds(out, K.REBOUND)  # the attempt 8 s after the second miss is no put-back
    assert (reb.t, reb.team, reb.attrs["offensive"], reb.attrs["timeline"]) == (21.5, HOME, True, "putback")
    out = tlm.derive_events(events[:2], manual([(0, 21, HOME), (21, 22, AWAY), (22, 60, HOME)]))
    assert all(r.attrs["offensive"] is False for r in kinds(out, K.REBOUND))  # the ball changed hands in between


def test_a_scorebug_make_does_not_explain_a_change() -> None:
    events = [ev(K.SHOT_MADE, 29.0, HOME, value=2, source="scoreboard")]
    out = tlm.derive_events(events, manual([(0, 30, HOME), (30, 60, AWAY)]))
    assert kinds(out, K.SHOT_MADE)[0].source == "scoreboard"  # passed through
    assert kinds(out, K.POSSESSION)[0].attrs["cause"] in ("turnover", "steal")


def test_derive_leaves_unrelated_events_as_they_are() -> None:
    timeout = ev(K.TIMEOUT, 10.0, HOME)
    pass_ = ev(K.PASS, 12.0, HOME)
    out = tlm.derive_events([timeout, pass_], manual([(0, 60, HOME)]))
    assert timeout in out and pass_ in out  # the same objects: no revision when nothing changes


# --------------------------------------------------------------------------- the command


def test_nba_analyze_timeline_writes_the_events_and_the_timeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "game"
    (root / "camera_parts").mkdir(parents=True)
    (root / "runs" / "t").mkdir(parents=True)
    (root / "game.json").write_text(json.dumps({"video": "none.mp4"}))
    center, tilt, fps = np.array([47.0, -94.0, 36.0]), 20.0, 29.97
    d = center[2] / math.tan(math.radians(tilt))
    head = {"header": True, "center": center.tolist(), "image_size": [1920, 1080], "fps": fps, "stride": 1}
    rows = [head]
    for t0, t1, x in CAMERA:
        pan = math.degrees(math.asin((x - center[0]) / d))
        for f in range(round(t0 * fps), round(t1 * fps)):
            rows.append({"frame": f, "t": round(f / fps, 4), "status": "tracking",
                         "params": {"pan": pan, "tilt": tilt, "roll": 0.0, "f": 2000.0}})
    (root / "camera_parts" / "part0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    _, events = game()
    (root / "runs" / "t" / "events.jsonl").write_text("".join(json.dumps(e.to_dict()) + "\n" for e in events))
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("nba_analyze", TOOLS / "nba_analyze.py")
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    monkeypatch.setattr(sys, "argv", ["nba_analyze.py", str(root), "timeline", "--tag", "t", "--fit", "none",
                                      "--shots", "off"])
    assert tool.main() == 0
    out = [Event.from_dict(json.loads(line)) for line in (root / "runs" / "t" / "timeline.jsonl").open()]
    assert [e.attrs["cause"] for e in kinds(out, K.POSSESSION)] == ["made_shot", "defensive_rebound", "steal"]
    doc = json.loads((root / "runs" / "t" / "timeline.json").read_text())
    assert doc["attack"] == {"home_first": "left", "switch_t": None}
    assert [s[2] for s in doc["segments"]] == ["home", "away", "home", "away"]
