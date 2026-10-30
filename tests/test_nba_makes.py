"""Shots from the video (courtvision/nba/makes.py): rim visits, candidates, the three models and the decisions; Jev's
rim reads (nba/rim_jev.py); and ``nba_analyze.py timeline --shots``, without the network or a video."""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from courtvision.nba import makes as mk
from courtvision.nba import rim_jev
from courtvision.nba import timeline as tlm
from courtvision.types import BBox, Event, EventKind, Team

HOME, AWAY = Team.HOME, Team.AWAY
K = EventKind
TOOLS = Path(__file__).resolve().parents[1] / "tools"
_ids = itertools.count()


def ev(kind: EventKind, t: float, team: Team | None = HOME, **kw: object) -> Event:
    return Event(f"e{next(_ids)}", kind, t, int(t * 30), team=team, **kw)  # type: ignore[arg-type]


def shot(sid: str, release: float, result: float, made: bool, cues: dict | None = None, team: Team = HOME,
         value: int = 2, **attrs: object) -> list[Event]:
    """An engine shot: its attempt and its result with rim ``cues``."""
    kind = K.SHOT_MADE if made else K.SHOT_MISSED
    return [ev(K.SHOT_ATTEMPT, release, team, value=value, shot_id=sid, player_id="home-t1", attrs=dict(attrs)),
            ev(kind, result, team, value=value, shot_id=sid, player_id="home-t1", attrs={"cues": cues or {}, **attrs})]


def manual(runs: list[tuple[float, float, Team, int]], dt: float = 0.25,
           off: tuple[float, float] | None = None) -> tlm.Timeline:
    """A timeline from (t0, t1, team, phase) runs; ``off``: a stretch with the camera off the court."""
    t = np.arange(0.0, max(b for _, b, _, _ in runs), dt)
    state, phase = np.zeros(len(t), np.int8), np.zeros(len(t), np.int8)
    for a, b, team, ph in runs:
        sel = (t >= a) & (t < b)
        state[sel], phase[sel] = tlm.TEAMS.index(team), ph
    tracked = np.ones(len(t), bool)
    if off is not None:
        tracked[(t >= off[0]) & (t < off[1])] = False
    return tlm.Timeline(t, state, tlm.AttackMap("left"), tracked, phase)


# home has the ball to 30 s; away gains it there, in its back court to 34 s, then attacks to 80 s; the camera is off
# the court 40-58 s (a stoppage before free throws)
TIMELINE = manual([(0, 30, HOME, tlm.FRONT), (30, 34, AWAY, tlm.BACK), (34, 80, AWAY, tlm.FRONT)], off=(40, 58))


def visit(t: float, side: str = "left", dmin: float = 0.2, frames: int = 12, dy_first: float = -1.6,
          dy_last: float = 2.1) -> mk.RimVisit:
    return mk.RimVisit(side, t - 0.4, t + 0.4, t, frames, dmin, dy_first, dy_last)


def test_rim_visits_are_runs_of_frames_with_the_ball_near_a_projected_rim(monkeypatch: pytest.MonkeyPatch) -> None:
    box = BBox(100.0, 100.0, 120.0, 104.0)  # a 20 px rim centred on (110, 102)
    monkeypatch.setattr(mk, "rim_boxes", lambda cam, court, size: {"left": box})
    camera = SimpleNamespace(image_size=(1920, 1080), get=lambda f: object(), time=lambda f: f / 30.0)
    path = {f: (110.0, 72.0 + 5.0 * f, 0.9) for f in range(13)}  # falls from 1.5 rim widths above through the rim
    path |= {f: (110.0, 102.0, 0.9) for f in (40, 41)}  # a second, short visit after a long gap
    path[200] = (900.0, 500.0, 0.9)  # far from the rim
    first, second = mk.rim_visits(path, camera, court=None)
    assert (first.frames, first.dmin, first.dy_first, first.dy_last) == (13, 0.0, -1.5, 1.5)
    assert first.t_near == pytest.approx(6 / 30.0) and first.side == "left"
    assert (second.frames, second.t0) == (2, pytest.approx(40 / 30.0))


def test_candidates_are_the_engines_results_with_their_visit_and_the_visits_it_missed() -> None:
    engine = shot("s1", 20.0, 21.0, True)
    bug = shot("s2", 40.0, 41.0, True, inferred="scoreboard")
    visits = [visit(21.3), visit(45.0), visit(60.0, dmin=1.6), visit(62.0, frames=2)]
    cands = mk.candidates([*engine, *bug], visits)
    assert [(c.key, c.visit) for c in cands] == [(engine[1].event_id, visits[0]), ("rim-45.00", visits[1])]
    assert cands[0].release_t == 20.0 and cands[1].release_t == pytest.approx(44.0) and cands[1].side == "left"


def test_candidate_features_read_cues_the_rim_visit_what_follows_and_the_setting() -> None:
    [_, made] = shot("s1", 28.0, 29.0, True, {"crossed_inside": True, "rim_contact": False, "crossing_margin_px": 15.0,
                                              "occluded_frames": 2})
    c = mk.Candidate(made.event_id, 29.0, 28.0, made, visit(29.0))
    activity = np.array([20.0, 25.0, 26.0, 33.0])
    f = mk.candidate_features(c, TIMELINE, activity)
    assert {k: f[k] for k in mk.CUE_FEATURES} == {"crossed_inside": 1.0, "rim_contact": 0.0, "drop_in_band": 0.0,
                                                  "drop_is_freefall": 0.0, "margin": 0.75, "margin_missing": 0.0,
                                                  "occluded": 2.0}
    assert (f["engine"], f["visit"], f["visit_dmin"], f["visit_from_above"], f["visit_below"]) == (1.0, 1.0, 0.2, 1.6, 2.1)
    assert (f["changed"], f["dt_change"], f["back_s"]) == (1.0, 1.0, 4.0)  # away inbounds, 4 s in its back court
    assert (f["cam_before"], f["activity"]) == (1.0, 0.3)
    assert (f["jev_in_net"], f["jev_made"], f["jev_missing"]) == (0.0, 0.5, 1.0)
    ft = mk.Candidate("rim-60.00", 60.0, 59.0, None, visit(60.0))
    f = mk.candidate_features(ft, TIMELINE, activity, {"probs": {"made": 0.6, "missed": 0.2}, "in_net": [0.1, 0.8]})
    assert (f["engine"], f["margin_missing"], f["activity"], f["cam_before"]) == (0.0, 1.0, 0.0, 0.0)  # a stoppage
    assert (f["jev_in_net"], f["jev_missing"]) == (0.8, 0.0) and f["jev_made"] == pytest.approx(0.75)


def test_logistic_learns_the_direction_and_is_a_constant_with_one_class() -> None:
    rng = np.random.default_rng(0)
    rows = [{"x": float(x), "noise": float(rng.normal())} for x in rng.integers(0, 2, 200)]
    labels = [r["x"] == 1.0 if rng.random() > 0.1 else r["x"] == 0.0 for r in rows]
    m = mk.Logistic.fit(rows, labels, ("x", "noise"))
    assert m.p({"x": 1.0}) > 0.75 > 0.25 > m.p({"x": 0.0})
    assert mk.Logistic.from_dict(json.loads(json.dumps(m.to_dict()))).p({"x": 1.0}) == pytest.approx(m.p({"x": 1.0}),
                                                                                                    abs=1e-5)
    always = mk.Logistic.fit(rows, [True] * len(rows), ("x",))
    assert always.p({"x": 0.0}) == always.p({"x": 1.0}) == pytest.approx(0.98)


def test_label_candidates_pairs_one_to_one_nearest_first_with_field_goals_and_free_throws() -> None:
    cands = [mk.Candidate(k, t, t - 1.0) for k, t in (("a", 10.0), ("b", 10.5), ("c", 30.0), ("d", 50.0))]
    labels = mk.label_candidates(cands, [(10.2, False, True), (31.0, False, False), (50.4, True, True)])
    assert labels == {"a": "fg_made", "b": "false", "c": "fg_missed", "d": "ft_made"}


def test_fit_shot_models_learns_free_throws_from_their_setting() -> None:
    rows, labels = [], []
    for i in range(60):
        ft = i % 3 == 0
        rows.append({"cam_before": 0.2 if ft else 1.0, "activity": 0.1 if ft else 0.6, "crossed_inside": float(i % 2),
                     "jev_in_net": 0.5})
        labels.append(("ft_" if ft else "fg_") + ("made" if i % 2 else "missed") if i % 5 else "false")
    plain, with_jev = mk.fit_shot_models(rows, labels), mk.fit_shot_models(rows, labels, jev=True)
    assert plain.ft.names == mk.FT_FEATURES and not plain.uses_jev and with_jev.uses_jev
    assert plain.ft.p({"cam_before": 0.2, "activity": 0.1}) > 0.5 > plain.ft.p({"cam_before": 1.0, "activity": 0.6})
    assert plain.made.p({"crossed_inside": 1.0}) > 0.5 > plain.made.p({"crossed_inside": 0.0})
    assert mk.ShotModels.from_dict(with_jev.to_dict()).made.names == with_jev.made.names


# a real shot when the engine measured a crossing margin or the ball visited the rim; a free throw when the camera was
# off the court before; made when the ball crossed inside the ring or left below the rim
MODELS = mk.ShotModels(real=mk.Logistic(("margin_missing", "visit"), (-10.0, 10.0), 5.0),
                       ft=mk.Logistic(("cam_before",), (-20.0,), 10.0),
                       made=mk.Logistic(("crossed_inside", "visit_below"), (10.0, 5.0), -5.0))


def test_decide_relabels_drops_adds_and_credits_assists_from_the_video_alone() -> None:
    vetoed = shot("s1", 20.0, 21.0, False, {"crossed_inside": True, "crossing_margin_px": 12.0})  # a make the bug vetoed
    pass_ = ev(K.PASS, 19.0, HOME, player_id="home-t2", other_player_id="home-t1")
    flipped = shot("s2", 24.0, 25.0, True, {"crossed_inside": False, "crossing_margin_px": -30.0})
    its_assist = ev(K.ASSIST, 25.0, HOME, shot_id="s2", player_id="home-t3")
    false = shot("s3", 26.0, 27.0, False, {"rim_contact": True})  # no margin, no visit: no shot
    false_block = ev(K.BLOCK, 26.5, AWAY, shot_id="s3")
    from_bug = shot("s4", 27.0, 28.0, True, {}, inferred="scoreboard")
    as_fg = shot("s5", 59.0, 60.0, True, {"crossed_inside": True, "crossing_margin_px": 8.0}, team=AWAY, value=2)
    out = mk.decide([*vetoed, pass_, *flipped, its_assist, *false, false_block, *from_bug, *as_fg], TIMELINE, MODELS,
                    visits=[visit(76.0, dy_last=2.4)])
    by_id = {e.event_id: e for e in out}
    assert by_id[vetoed[1].event_id].kind is K.SHOT_MADE and by_id[vetoed[1].event_id].attrs["makes"]["p_made"] > 0.9
    assert by_id[flipped[1].event_id].kind is K.SHOT_MISSED and its_assist.event_id not in by_id
    assert not {e.event_id for e in (*false, false_block, *from_bug)} & set(by_id)
    ft_att, ft = by_id[as_fg[0].event_id], by_id[as_fg[1].event_id]  # after the stoppage: a free throw, not a 2
    assert (ft.value, ft.attrs["shot_type"], ft_att.value, ft_att.attrs["shot_type"]) == (1, "free_throw", 1, "free_throw")
    [att, res] = [e for e in out if e.attrs.get("inferred") == "rim_visit"]
    assert (att.kind, att.t, res.kind, res.t, res.team, res.value) == (K.SHOT_ATTEMPT, 75.0, K.SHOT_MADE, 76.0, AWAY, 2)
    [assist] = [e for e in out if e.kind is K.ASSIST]
    assert (assist.shot_id, assist.player_id, assist.other_player_id, assist.team) == ("s1", "home-t2", "home-t1", HOME)


def test_decide_uses_the_models_for_each_candidate() -> None:
    early = shot("s1", 10.0, 11.0, True, {"crossed_inside": True, "crossing_margin_px": 5.0})
    late = shot("s2", 60.0, 61.0, True, {"crossed_inside": True, "crossing_margin_px": 5.0}, team=AWAY)
    never = dataclasses_replace(MODELS, made=mk.Logistic(("crossed_inside",), (0.0,), -5.0))
    out = mk.decide([*early, *late], TIMELINE, lambda c: MODELS if c.t < 30 else never)
    assert [e.kind for e in out if e.kind in (K.SHOT_MADE, K.SHOT_MISSED)] == [K.SHOT_MADE, K.SHOT_MISSED]


def dataclasses_replace(models: mk.ShotModels, **kw: mk.Logistic) -> mk.ShotModels:
    return mk.ShotModels(**{"real": models.real, "ft": models.ft, "made": models.made, **kw})


def test_the_rim_judge_answers_from_its_cache_without_a_client(tmp_path: Path) -> None:
    cache = tmp_path / "jev_rim.jsonl"
    cache.write_text(json.dumps({"key": "e7", "t": 10.0, "jev": {"outcome": "made", "in_net": [0.9]}}) + "\n")
    judge = rim_jev.RimJudge(tmp_path / "none.mp4", None, None, {}, client=None, cache=cache)
    assert judge.judge("e7", 10.0) == {"outcome": "made", "in_net": [0.9]}
    assert judge.judge("e8", 12.0) is None  # not asked before, and nobody to ask


def _tool():  # noqa: ANN202
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("nba_analyze", TOOLS / "nba_analyze.py")
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_nba_analyze_timeline_decides_shots_with_stored_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from test_nba_timeline import CAMERA, game  # the synthetic game: the camera above and the engine's events

    root = tmp_path / "game"
    (root / "camera_parts").mkdir(parents=True)
    (root / "runs" / "t").mkdir(parents=True)
    (root / "game.json").write_text(json.dumps({"video": "none.mp4"}))
    center, tilt = np.array([47.0, -94.0, 36.0]), 20.0
    d = center[2] / np.tan(np.radians(tilt))
    rows: list[dict] = [{"header": True, "center": center.tolist(), "image_size": [1920, 1080], "fps": 29.97, "stride": 1}]
    for t0, t1, x in CAMERA:
        pan = float(np.degrees(np.arcsin((x - center[0]) / d)))
        rows += [{"frame": f, "t": round(f / 29.97, 4), "status": "tracking",
                  "params": {"pan": pan, "tilt": tilt, "roll": 0.0, "f": 2000.0}}
                 for f in range(round(t0 * 29.97), round(t1 * 29.97))]
    (root / "camera_parts" / "part0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    _, events = game()
    for e in events:  # the made shot at 20 s: the ball crossed inside the ring
        if e.kind is K.SHOT_MADE:
            e.attrs["cues"] = {"crossed_inside": True, "crossing_margin_px": 10.0}
    (root / "runs" / "t" / "events.jsonl").write_text("".join(json.dumps(e.to_dict()) + "\n" for e in events))
    models = tmp_path / "shot_models.json"
    models.write_text(json.dumps(MODELS.to_dict()))
    tool = _tool()
    argv = ["nba_analyze.py", str(root), "timeline", "--tag", "t", "--fit", "none"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="--shot-models"):
        tool.main()
    monkeypatch.setattr(sys, "argv", [*argv, "--shot-models", str(models), "--jev"])
    with pytest.raises(SystemExit, match="without --jev"):
        tool.main()  # models fitted without Jev's answers: nothing is asked
    monkeypatch.setattr(sys, "argv", [*argv, "--shot-models", str(models)])
    assert tool.main() == 0
    out = [Event.from_dict(json.loads(line)) for line in (root / "runs" / "t" / "timeline.jsonl").open()]
    results = [e for e in out if e.kind in (K.SHOT_MADE, K.SHOT_MISSED)]
    assert [(e.kind, e.team, e.value) for e in results] == [(K.SHOT_MADE, HOME, 2)]  # the 50 s miss had no margin
    doc = json.loads((root / "runs" / "t" / "timeline.json").read_text())
    assert (doc["shots"]["candidates"], doc["shots"]["kept"], doc["shots"]["made"], doc["shots"]["added"]) == (2, 1, 1, 0)
