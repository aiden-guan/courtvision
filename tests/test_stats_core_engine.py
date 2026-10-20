"""StatsEngine: revisions, scoreboard, operator lineups, pace, decisions, xPTS and JSON (stats/engine.py)."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from courtvision.geometry.court import court_for
from courtvision.geometry.rules import rules_for
from courtvision.stats.engine import GameStats, StatsConfig, StatsEngine, scoring_run
from courtvision.stats.boxscore import ScoringPlay
from courtvision.types import (
    BallState,
    Event,
    EventKind,
    EventStatus,
    FrameState,
    PlayerState,
    Roster,
    RosterEntry,
    ScoreboardReading,
    Team,
)

K = EventKind
H, A = Team.HOME, Team.AWAY
HOME5 = [f"home-{i}" for i in range(1, 6)]
AWAY5 = [f"away-{i}" for i in range(1, 6)]


def make_roster() -> Roster:
    entries = [RosterEntry(f"home-{i}", H, str(i), f"Home {i}") for i in range(1, 9)]
    entries += [RosterEntry(f"away-{i}", A, str(i), f"Away {i}") for i in range(1, 9)]
    return Roster(entries)


class Log:
    def __init__(self) -> None:
        self._n = 0
        self._s = 0

    def ev(self, kind: EventKind, t: float, **kw: Any) -> Event:
        self._n += 1
        kw.setdefault("status", EventStatus.CONFIRMED)
        kw.setdefault("source", "sim")
        return Event(event_id=f"e{self._n:06d}", kind=kind, t=t, frame_idx=int(round(t * 30)), **kw)

    def shot(self, pid: str, team: Team, t: float, value: int = 2, made: bool = True, **attrs: Any) -> list[Event]:
        self._s += 1
        sid = f"s{self._s:04d}"
        xy = (80.0, 25.0) if team is H else (14.0, 25.0)
        att = self.ev(K.SHOT_ATTEMPT, t, team=team, player_id=pid, value=value, shot_id=sid, x=xy[0], y=xy[1], attrs=attrs)
        res = self.ev(K.SHOT_MADE if made else K.SHOT_MISSED, t + 1.0, team=team, player_id=pid, shot_id=sid,
                      value=value if made else 0)
        return [att, res]


def frame(i: int, t: float, players: list[PlayerState] | None = None, sb: ScoreboardReading | None = None) -> FrameState:
    return FrameState(frame_idx=i, t=t, players=players or [], ball=BallState(visible=False), scoreboard=sb)


def engine(**kw: Any) -> StatsEngine:
    court = kw.pop("court", court_for("nba"))
    return StatsEngine(court, make_roster(), **kw)


# --------------------------------------------------------------------------- revisions


def test_latest_revision_wins_and_retractions_undo() -> None:
    lg = Log()
    att, made = lg.shot("home-1", H, 5.0, value=3)
    eng = engine()
    eng.on_events([att, made])
    p = eng.snapshot().players["home-1"]
    assert (p.pts, p.fg3m, p.fg3a) == (3, 1, 1)

    eng.on_events([replace(made, value=2, revision=1)])  # reconciled to a 2
    p = eng.snapshot().players["home-1"]
    assert (p.pts, p.fgm, p.fg3m, p.fg3a) == (2, 1, 0, 0)

    eng.on_events([replace(made, value=3, revision=0)])  # stale revision ignored
    assert eng.snapshot().players["home-1"].pts == 2

    eng.on_events([replace(made, status=EventStatus.RETRACTED, revision=2)])
    p = eng.snapshot().players["home-1"]
    assert (p.pts, p.fgm, p.fga) == (0, 0, 1)  # the attempt still stands (result pending)

    eng.on_events([replace(att, status=EventStatus.RETRACTED, revision=1)])
    snap = eng.snapshot()
    assert snap.players["home-1"].fga == 0 and snap.shots == [] and snap.n_events == 0
    assert eng.events == []


def test_stored_events_are_copies() -> None:
    lg = Log()
    att, made = lg.shot("home-1", H, 5.0, value=2)
    eng = engine()
    eng.on_events([att, made])
    made.value = 3  # producer mutates its object without re-emitting
    made.attrs["x"] = 1
    assert eng.snapshot().players["home-1"].pts == 2


# --------------------------------------------------------------------------- scoreboard


def test_trusted_scoreboard_sets_score_and_flags_mismatch() -> None:
    lg = Log()
    eng = engine()
    eng.on_events(lg.shot("home-1", H, 5.0, value=2))
    snap = eng.snapshot()
    assert snap.score == {H: 2, A: 0} and snap.score_source == "events" and not snap.score_mismatch

    sb = ScoreboardReading(t=7.0, home_score=3, away_score=0, period=1, game_clock=700.0, shot_clock=20.0, source="sim")
    eng.on_frame(frame(210, 7.0, sb=sb), None, None)
    snap = eng.snapshot()
    assert snap.score == {H: 3, A: 0} and snap.score_source == "scoreboard"
    assert snap.score_mismatch and snap.event_score == {H: 2, A: 0}
    assert (snap.period, snap.game_clock, snap.shot_clock) == (1, 700.0, 20.0)

    eng.on_events([lg.ev(K.SCORE_CORRECTION, 8.0, team=H, value=1, source="scoreboard")])
    snap = eng.snapshot()
    assert not snap.score_mismatch and snap.teams[H].pts_unattributed == 1


def test_ocr_scores_need_two_identical_distinct_reads() -> None:
    eng = engine()
    r1 = ScoreboardReading(t=1.0, home_score=2, away_score=0, source="ocr")
    eng.on_frame(frame(30, 1.0, sb=r1), None, None)
    eng.on_frame(frame(31, 1.033, sb=r1), None, None)  # the same read attached to the next frame
    assert eng.snapshot().score_source == "events"
    eng.on_frame(frame(36, 1.2, sb=replace(r1, t=1.2)), None, None)
    snap = eng.snapshot()
    assert snap.score_source == "scoreboard" and snap.score == {H: 2, A: 0}
    eng.on_frame(frame(42, 1.4, sb=replace(r1, t=1.4, home_score=5)), None, None)  # one glitchy read
    assert eng.snapshot().score == {H: 2, A: 0}


def test_manual_scoreboard_push() -> None:
    eng = engine()
    eng.on_scoreboard(ScoreboardReading(t=3.0, home_score=10, away_score=8, period=2, game_clock=300.0, source="manual"))
    snap = eng.snapshot()
    assert snap.score == {H: 10, A: 8} and snap.period == 2
    assert snap.elapsed_game_s == pytest.approx(720 + 420)


# --------------------------------------------------------------------------- lineups


def _clock(t: float) -> float:
    """Game clock that runs, stops for a dead ball between t = 30 and 45, then runs again."""
    if t <= 30.0:
        return 720.0 - t
    if t <= 45.0:
        return 690.0
    return 720.0 - (t - 15.0)


def _lineup_game(with_clock: bool, entries: bool = True) -> StatsEngine:
    lg = Log()
    eng = engine()
    if entries:
        eng.set_lineup(H, HOME5, 0.0)
        eng.set_lineup(A, AWAY5, 0.0)
    events = [lg.ev(K.POSSESSION, 0.0, team=H, player_id="home-1", attrs={"cause": "jump_ball"})]
    events += lg.shot("home-1", H, 20.0, value=2)  # made at 21
    events += lg.shot("away-1", A, 50.0, value=3)  # made at 51
    events += lg.shot("home-6", H, 70.0, value=2)  # made at 71
    for i in range(0, 1001):  # 10 Hz, t = 0 .. 100
        t = i / 10.0
        sb = ScoreboardReading(t=t, period=1, game_clock=_clock(t), source="sim") if with_clock else None
        if entries and i == 400:
            eng.substitute(H, "home-5", "home-6", 40.0)
        eng.on_frame(frame(i * 3, t, sb=sb), None, None)
        eng.on_events([e for e in events if t - 0.05 < e.t <= t + 0.05])
    return eng


def test_lineup_seconds_and_plus_minus_use_the_game_clock() -> None:
    snap = _lineup_game(with_clock=True)
    p = snap.snapshot().players
    assert p["home-1"].seconds == pytest.approx(85.0)  # 100 s of video, 15 s of stopped clock
    assert p["home-5"].seconds == pytest.approx(30.0)
    assert p["home-6"].seconds == pytest.approx(55.0)
    assert (p["home-1"].plus_minus, p["home-5"].plus_minus, p["home-6"].plus_minus) == (1, 2, -1)
    assert all(p[a].plus_minus == -1 and p[a].seconds == pytest.approx(85.0) for a in AWAY5)
    assert p["home-8"].seconds == 0.0 and p["home-8"].plus_minus == 0
    s = snap.snapshot()
    assert s.clock_basis == "game_clock" and s.elapsed_game_s == pytest.approx(85.0)


def test_lineup_records_with_possessions() -> None:
    lineups = _lineup_game(with_clock=True).snapshot().lineups
    first = lineups["home:" + "+".join(sorted(HOME5))]
    second = lineups["home:" + "+".join(sorted(HOME5[:4] + ["home-6"]))]
    away = lineups["away:" + "+".join(sorted(AWAY5))]
    assert (first["seconds"], first["pts_for"], first["pts_against"], first["plus_minus"]) == (
        pytest.approx(30.0), 2, 0, 2)
    assert (second["seconds"], second["pts_for"], second["pts_against"]) == (pytest.approx(55.0), 2, 3)
    # possessions: home @0 (jump), away @21 (after the make), home @51 (after the make)
    assert (first["off_possessions"], first["def_possessions"], first["possessions"]) == (1, 1, 1.0)
    assert (second["off_possessions"], second["def_possessions"]) == (1, 0)
    assert (away["off_possessions"], away["def_possessions"], away["stints"]) == (1, 2, 1)
    assert first["small_sample"] and first["off_rtg"] == pytest.approx(200.0) and first["team"] is H


def test_lineups_fall_back_to_live_video_time() -> None:
    p = _lineup_game(with_clock=False).snapshot().players
    assert p["home-1"].seconds == pytest.approx(100.0)
    assert p["home-5"].seconds == pytest.approx(40.0) and p["home-6"].seconds == pytest.approx(60.0)


def test_no_operator_entries_means_no_minutes_or_lineups() -> None:
    snap = _lineup_game(with_clock=True, entries=False).snapshot()
    assert snap.lineups == {}
    assert all(line.seconds == 0.0 and line.plus_minus == 0 for line in snap.players.values())
    assert snap.teams[H].plus_minus == 1  # team margin is still known


def test_period_end_pauses_video_time() -> None:
    lg = Log()
    eng = engine()
    for i in range(0, 501):
        eng.on_frame(frame(i, i / 10.0), None, None)
    eng.on_events([lg.ev(K.PERIOD_END, 50.0)])
    for i in range(501, 1001):
        eng.on_frame(frame(i, i / 10.0), None, None)
    assert eng.snapshot().elapsed_game_s == pytest.approx(50.0)
    eng.on_events([lg.ev(K.PERIOD_START, 100.0)])
    for i in range(1001, 1101):
        eng.on_frame(frame(i, i / 10.0), None, None)
    assert eng.snapshot().elapsed_game_s == pytest.approx(60.0)


def test_play_after_period_end_resumes_video_time_without_period_start() -> None:
    lg = Log()
    eng = engine()
    eng.on_events([lg.ev(K.PERIOD_END, 0.0)])
    for i in range(0, 101):
        eng.on_frame(frame(i, i / 10.0), None, None)
    assert eng.snapshot().elapsed_game_s == pytest.approx(0.0)
    eng.on_events([lg.ev(K.POSSESSION, 10.0, team=H, attrs={"cause": "inbound"})])
    for i in range(101, 201):
        eng.on_frame(frame(i, i / 10.0), None, None)
    assert eng.snapshot().elapsed_game_s == pytest.approx(10.0)


# --------------------------------------------------------------------------- pace


def _pace_engine(court_league: str, rules: Any) -> StatsEngine:
    lg = Log()
    eng = engine(court=court_for(court_league), rules=rules)
    evs = [lg.ev(K.POSSESSION, 0.0, team=H, attrs={"cause": "jump_ball"})]
    for k in range(3):  # 3 possessions each
        evs += lg.shot("home-1", H, 10.0 + 30 * k, value=2)
        evs += lg.shot("away-1", A, 25.0 + 30 * k, value=2)
    eng.on_events(evs)
    period_s = rules.period_s if rules is not None else {"nba": 720, "nfhs": 480}[court_league]
    for i in range(0, 121):
        eng.on_frame(frame(i, float(i), sb=ScoreboardReading(t=float(i), period=1, game_clock=period_s - i, source="sim")),
                     None, None)
    return eng


@pytest.mark.parametrize(
    ("league", "rules", "game_minutes"),
    [("nba", None, 48.0), ("nfhs", "nfhs", 32.0), ("ncaa_m", "ncaa_m", 40.0), ("nba", "fiba", 40.0)],
)
def test_pace_is_normalised_to_the_league_game_length(league: str, rules: str | None, game_minutes: float) -> None:
    spec = rules_for(rules) if rules else None
    home = _pace_engine(league, spec).snapshot().teams[H]
    assert home.possessions == 3 and home.seconds == pytest.approx(120.0)
    assert home.pace == pytest.approx(game_minutes * 3 / 2.0)


def test_pace_uses_league_table_without_rules() -> None:
    home = _pace_engine("nfhs", None).snapshot().teams[H]
    assert home.pace == pytest.approx(32.0 * 3 / 2.0)


# --------------------------------------------------------------------------- xPTS


class FakeShotModel:
    def __init__(self) -> None:
        self.seen: list[Any] = []

    def xfg(self, f: Any) -> float:
        self.seen.append(f)
        return 0.75 if f.value == 1 else 0.4


def test_xpts_uses_shot_model_when_attrs_lack_xfg() -> None:
    lg = Log()
    model = FakeShotModel()
    eng = engine(shot_model=model)
    eng.on_events(lg.shot("home-1", H, 5.0, value=3, xfg=0.3) + lg.shot("home-1", H, 15.0, value=2))
    att = lg.ev(K.SHOT_ATTEMPT, 30.0, team=H, player_id="home-1", value=1, shot_id="ft1", attrs={"shot_type": "free_throw"})
    eng.on_events([att])
    snap = eng.snapshot()
    assert snap.players["home-1"].xpts == pytest.approx(0.3 * 3 + 0.4 * 2 + 0.75)
    assert snap.teams[H].avg_xfg == pytest.approx((0.3 + 0.4) / 2)
    assert {round(f.distance_ft, 2) for f in model.seen} == {8.75, 13.75}  # FG from x,y; FT line distance
    eng.snapshot()
    assert len(model.seen) == 2  # cached


def test_xpts_with_the_real_shot_quality_model() -> None:
    sq = pytest.importorskip("courtvision.stats.shot_quality")
    lg = Log()
    eng = engine(shot_model=sq.ShotQualityModel("nba"))
    eng.on_events(lg.shot("home-1", H, 5.0, value=2))
    shot = eng.snapshot().shots[0]
    assert shot.xfg is not None and 0.0 < shot.xfg < 1.0
    assert eng.snapshot().players["home-1"].xpts == pytest.approx(2 * shot.xfg)


# --------------------------------------------------------------------------- decisions


class FakeGrade:
    def __init__(self, e: Event) -> None:
        self.e = e

    def to_dict(self) -> dict[str, Any]:
        opt = {"kind": "pass", "target_id": self.e.other_player_id, "success_prob": 0.9,
               "xpts_if_success": 1.0, "value": 0.9, "detail": {}}
        return {"t": self.e.t, "player_id": self.e.player_id, "event_id": None, "chosen": opt, "best": opt,
                "options": [opt], "delta": 0.0, "label": "good"}


class FakeDecisionModel:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, Team, str]] = []

    def grade(self, frame: FrameState, event: Event, offense: Team, basket: str) -> FakeGrade:
        self.calls.append((frame.frame_idx, event.event_id, offense, basket))
        return FakeGrade(event)


def test_decision_grades_for_passes_and_shots() -> None:
    lg = Log()
    model = FakeDecisionModel()
    eng = engine(decision_model=model)
    for i in range(0, 61):
        eng.on_frame(frame(i, i / 30.0), "home-1", H)
    pass_e = lg.ev(K.PASS, 1.0, team=H, player_id="home-1", other_player_id="home-2", attrs={"completed": True})
    shot = lg.ev(K.SHOT_ATTEMPT, 1.5, team=A, player_id="away-1", value=2, shot_id="s1", attrs={"basket": "left"})
    ft = lg.ev(K.SHOT_ATTEMPT, 1.8, team=H, player_id="home-2", value=1, shot_id="s2", attrs={"shot_type": "free_throw"})
    later = lg.ev(K.PASS, 2.5, team=H, player_id="home-2", other_player_id="home-3")  # frame not seen yet
    eng.on_events([pass_e, shot, ft, later])
    assert [c[:4] for c in model.calls] == [(30, pass_e.event_id, H, "right"), (45, shot.event_id, A, "left")]
    for i in range(61, 80):
        eng.on_frame(frame(i, i / 30.0), "home-2", H)
    assert model.calls[-1][:2] == (75, later.event_id)
    decisions = eng.snapshot().decisions
    assert [d["event_id"] for d in decisions] == [pass_e.event_id, shot.event_id, later.event_id]
    assert decisions[0]["event_kind"] == "pass" and decisions[0]["label"] == "good"

    eng.on_events([replace(pass_e, status=EventStatus.RETRACTED, revision=1)])
    assert [d["event_id"] for d in eng.snapshot().decisions] == [shot.event_id, later.event_id]


def test_decision_model_failure_and_old_events_are_skipped() -> None:
    class Broken:
        def grade(self, *a: Any) -> None:
            raise RuntimeError("boom")

    lg = Log()
    eng = engine(decision_model=Broken())
    for i in range(0, 400):  # 13.3 s; the buffer keeps the last 10 s
        eng.on_frame(frame(i, i / 30.0), None, None)
    eng.on_events([lg.ev(K.PASS, 12.0, team=H, player_id="home-1", other_player_id="home-2")])
    assert eng.snapshot().decisions == []
    eng2 = engine(decision_model=FakeDecisionModel())
    for i in range(0, 400):
        eng2.on_frame(frame(i, i / 30.0), None, None)
    eng2.on_events([lg.ev(K.PASS, 1.0, team=H, player_id="home-1", other_player_id="home-2")])  # left the buffer
    assert eng2.snapshot().decisions == []


def test_decision_grades_with_the_real_decision_model() -> None:
    dec = pytest.importorskip("courtvision.stats.decision")
    sq = pytest.importorskip("courtvision.stats.shot_quality")
    court = court_for("nba")
    model = dec.DecisionModel(court, sq.ShotQualityModel("nba"))
    eng = StatsEngine(court, make_roster(), decision_model=model)
    players = [PlayerState(f"home-{i}", H, 70.0 + 3 * i, 10.0 + 6 * i) for i in range(1, 6)]
    players += [PlayerState(f"away-{i}", A, 74.0 + 3 * i, 12.0 + 6 * i) for i in range(1, 6)]
    eng.on_frame(frame(30, 1.0, players=players), "home-1", H)
    eng.on_events([Log().ev(K.PASS, 1.0, team=H, player_id="home-1", other_player_id="home-2")])
    decisions = eng.snapshot().decisions
    assert len(decisions) == 1 and decisions[0]["label"] in {"good", "neutral", "poor"}
    assert decisions[0]["event_id"] == "e000001"
    json.dumps(decisions, allow_nan=False)


# --------------------------------------------------------------------------- tracking + runs via the engine


def test_tracking_lines_merge_frames_and_events() -> None:
    lg = Log()
    eng = engine()
    for i in range(0, 31):  # home-1 runs 10 ft/s for 1 s while holding the ball
        p = PlayerState("home-1", H, 50.0 + 10.0 * i / 30.0, 25.0)
        eng.on_frame(frame(i, i / 30.0, players=[p]), "home-1", H)
    eng.on_events([
        lg.ev(K.DRIBBLE, 0.2, team=H, player_id="home-1"),
        lg.ev(K.PASS, 0.9, team=H, player_id="home-1", other_player_id="home-2", attrs={"completed": True}),
        lg.ev(K.PASS, 1.5, team=H, player_id="home-2", other_player_id="home-3", attrs={"completed": False}),
    ])
    tr = eng.snapshot().tracking
    assert tr["home-1"].distance_ft == pytest.approx(10.0)
    assert tr["home-1"].time_of_possession_s == pytest.approx(1.0)
    assert (tr["home-1"].touches, tr["home-1"].dribbles, tr["home-1"].passes_made) == (1, 1, 1)
    assert (tr["home-2"].passes_received, tr["home-2"].passes_made) == (1, 1)
    assert "home-3" not in tr  # intended receiver of an incomplete pass: nothing observed


def test_scoring_run_and_window() -> None:
    plays = [ScoringPlay(10.0, H, 2, "a"), ScoringPlay(20.0, A, 3, "b"), ScoringPlay(30.0, H, 2, "c"),
             ScoringPlay(40.0, H, 3, "d")]
    run = scoring_run(plays, now_t=45.0, window_s=300.0)
    assert (run["team"], run["points"], run["opp_points"], run["since_t"], run["plays"]) == (H, 5, 0, 30.0, 2)
    assert run["window"] == {H: 7, A: 3} and run["window_leader"] is H and run["window_diff"] == 4
    short = scoring_run(plays, now_t=45.0, window_s=20.0)
    assert short["window"] == {H: 5, A: 0}
    assert scoring_run([], 0.0, 300.0)["team"] is None


# --------------------------------------------------------------------------- JSON


def _full_engine() -> StatsEngine:
    lg = Log()
    eng = engine(shot_model=FakeShotModel(), decision_model=FakeDecisionModel())
    eng.set_lineup(H, HOME5, 0.0)
    eng.set_lineup(A, AWAY5, 0.0)
    for i in range(0, 91):
        t = i / 30.0
        players = [PlayerState("home-1", H, 60.0 + t, 25.0), PlayerState("away-1", A, 62.0, 25.0 + t)]
        eng.on_frame(frame(i, t, players=players, sb=ScoreboardReading(t=t, home_score=0, away_score=0, period=1,
                                                                      game_clock=720.0 - t, source="sim")),
                     "home-1", H)
    eng.on_events([lg.ev(K.PASS, 1.0, team=H, player_id="home-1", other_player_id="home-2")])
    eng.on_events(lg.shot("home-2", H, 2.0, value=3, closest_defender_ft=4.5))
    eng.on_events([lg.ev(K.REBOUND, 2.9, team=A, player_id=None, status=EventStatus.PROVISIONAL)])
    return eng


def test_to_dict_is_json_safe_and_round_trips() -> None:
    snap = _full_engine().snapshot()
    d = snap.to_dict()
    text = json.dumps(d, allow_nan=False)
    back = GameStats.from_dict(json.loads(text))
    assert back.to_dict() == d
    assert back.teams[H].pts == snap.teams[H].pts and back.players["home-2"].team is H
    assert set(d["teams"]) == {"home", "away"} and set(d["score"]) == {"home", "away"}
    assert d["players"]["home-2"]["provisional"] is False and "reb" in d["players"]["home-2"]
    assert d["teams"]["away"]["provisional"] is True
    assert d["lineups"] and all(rec["team"] in ("home", "away") for rec in d["lineups"].values())
    assert d["decisions"][0]["event_id"] == "e000001"
    assert d["shots"][0]["team"] == "home" and d["run"]["team"] == "home"
    assert d["possession_team"] == "home"


def test_nan_values_become_null() -> None:
    eng = engine()
    eng.on_frame(frame(0, 0.0, players=[PlayerState("home-1", H, float("nan"), 25.0)]), None, None)
    text = json.dumps(eng.snapshot().to_dict(), allow_nan=False)
    assert "NaN" not in text


def test_config_is_plain() -> None:
    cfg = StatsConfig()
    for v in vars(cfg).values():
        assert isinstance(v, (float, int, str, bool, tuple, list, dict))


# --------------------------------------------------------------------------- review regressions


def test_revised_period_start_is_not_counted_twice() -> None:
    """A re-emitted PERIOD_START (e.g. confirmed later) must not advance the period / half."""
    lg = Log()
    eng = engine()
    p1 = lg.ev(K.PERIOD_START, 0.0, status=EventStatus.PROVISIONAL)  # no attrs["period"]
    eng.on_events([p1, lg.ev(K.SHOT_ATTEMPT, 10.0, team=H, player_id="home-1", value=2, shot_id="s1",
                             attrs={"basket": "right"})])
    eng.on_events([replace(p1, status=EventStatus.CONFIRMED, revision=1)])
    eng.on_events([lg.ev(K.PERIOD_END, 700.0), lg.ev(K.PERIOD_START, 800.0)])  # Q2: still the first half
    assert eng.attacking_side(H) == "right" and eng.attacking_side(A) == "left"
    eng.on_events([lg.ev(K.PERIOD_END, 1500.0), lg.ev(K.PERIOD_START, 1600.0)])  # Q3: teams switch
    assert eng.attacking_side(H) == "left"


def test_retracted_period_start_is_forgotten() -> None:
    lg = Log()
    eng = engine()
    eng.on_events([lg.ev(K.PERIOD_START, 0.0),
                   lg.ev(K.SHOT_ATTEMPT, 10.0, team=H, player_id="home-1", value=2, shot_id="s1",
                         attrs={"basket": "right"})])
    bogus = [lg.ev(K.PERIOD_START, 20.0), lg.ev(K.PERIOD_START, 30.0)]  # two spurious starts -> "Q3"
    eng.on_events(bogus)
    assert eng.attacking_side(H) == "left"
    eng.on_events([replace(e, status=EventStatus.RETRACTED, revision=1) for e in bogus])
    assert eng.attacking_side(H) == "right"


def test_late_period_end_revision_does_not_stop_the_next_period() -> None:
    lg = Log()
    eng = engine()
    for i in range(0, 501):
        eng.on_frame(frame(i, i / 10.0), None, None)
    end = lg.ev(K.PERIOD_END, 50.0)
    eng.on_events([end])
    for i in range(501, 601):  # break: not counted
        eng.on_frame(frame(i, i / 10.0), None, None)
    eng.on_events([lg.ev(K.PERIOD_START, 60.0)])
    for i in range(601, 701):
        eng.on_frame(frame(i, i / 10.0), None, None)
    eng.on_events([replace(end, status=EventStatus.CONFIRMED, revision=1, confidence=0.99)])  # late re-emit
    for i in range(701, 801):
        eng.on_frame(frame(i, i / 10.0), None, None)
    assert eng.snapshot().elapsed_game_s == pytest.approx(70.0)


def test_attacking_side_is_a_majority_vote_not_the_last_shot() -> None:
    lg = Log()
    eng = engine()
    shots = [lg.ev(K.SHOT_ATTEMPT, 10.0 * k, team=H, player_id="home-1", value=2, shot_id=f"s{k}",
                   attrs={"basket": "right"}) for k in range(1, 4)]
    heave = lg.ev(K.SHOT_ATTEMPT, 50.0, team=H, player_id="home-2", value=3, shot_id="s9", attrs={"basket": "left"})
    eng.on_events(shots + [heave])
    assert eng.attacking_side(H) == "right"
    eng.on_events([replace(s, status=EventStatus.RETRACTED, revision=1) for s in shots])
    assert eng.attacking_side(H) == "left"  # retracted shots lose their vote


def test_ocr_clock_glitch_does_not_inflate_minutes_or_pace() -> None:
    eng = engine()
    eng.set_lineup(H, HOME5, 0.0)
    for i in range(0, 301):  # 5 Hz OCR, Q1 clock running
        t = i / 5.0
        clock = 720.0 - t
        period = 1
        if i == 150:
            period = 4  # one misread period digit
        sb = ScoreboardReading(t=t, period=period, game_clock=clock, source="ocr", confidence=0.9)
        eng.on_frame(frame(i, t, sb=sb), None, None)
    snap = eng.snapshot()
    assert snap.elapsed_game_s == pytest.approx(60.0)
    assert snap.players["home-1"].seconds == pytest.approx(60.0)


def test_scoring_run_with_negative_corrections() -> None:
    plays = [ScoringPlay(10.0, A, 3, "a"), ScoringPlay(20.0, H, 2, "b"), ScoringPlay(30.0, H, 2, "c")]
    run = scoring_run(plays + [ScoringPlay(35.0, A, -1, "r1")], now_t=40.0, window_s=300.0)
    assert (run["team"], run["points"], run["since_t"], run["plays"]) == (H, 4, 20.0, 2)  # not "A -1"
    assert run["window"] == {H: 4, A: 2}
    run = scoring_run(plays + [ScoringPlay(35.0, H, -2, "r2")], now_t=40.0, window_s=300.0)
    assert (run["team"], run["points"], run["plays"]) == (H, 2, 2)  # the correction shrinks H's run


def test_decision_detail_with_numpy_arrays_is_json_safe() -> None:
    np = pytest.importorskip("numpy")

    class ArrayGrade(FakeGrade):
        def to_dict(self) -> dict[str, Any]:
            d = super().to_dict()
            d["chosen"] = {**d["chosen"], "detail": {"xy": np.array([1.5, 2.5]), "n": np.int64(3)}}
            return d

    class ArrayModel(FakeDecisionModel):
        def grade(self, frame: FrameState, event: Event, offense: Team, basket: str) -> FakeGrade:
            return ArrayGrade(event)

    lg = Log()
    eng = engine(decision_model=ArrayModel())
    eng.on_frame(frame(30, 1.0), "home-1", H)
    eng.on_events([lg.ev(K.PASS, 1.0, team=H, player_id="home-1", other_player_id="home-2")])
    d = eng.snapshot().to_dict()
    json.dumps(d, allow_nan=False)
    assert d["decisions"][0]["chosen"]["detail"] == {"xy": [1.5, 2.5], "n": 3}


def test_held_ocr_clock_jump_does_not_change_period_or_attack_side() -> None:
    lg = Log()
    eng = engine()
    eng.on_events([lg.ev(K.SHOT_ATTEMPT, 0.5, team=H, player_id="home-1", value=2, shot_id="s1",
                         attrs={"basket": "right"})])
    eng.on_frame(frame(0, 0.0, sb=ScoreboardReading(t=0.0, period=1, game_clock=720.0, source="ocr")), None, None)
    eng.on_frame(frame(6, 0.2, sb=ScoreboardReading(t=0.2, period=3, game_clock=719.8, source="ocr")), None, None)
    snap = eng.snapshot()
    assert (snap.period, snap.game_clock) == (1, 720.0)  # the one-off "3" is not shown
    assert eng.attacking_side(H) == "right"
    eng.on_frame(frame(12, 0.4, sb=ScoreboardReading(t=0.4, period=1, game_clock=float("nan"), source="ocr")),
                 None, None)
    assert eng.snapshot().game_clock == 720.0


def test_pace_refreshes_per_frame_without_mutating_old_snapshots() -> None:
    lg = Log()
    eng = engine()
    eng.on_events([lg.ev(K.POSSESSION, 0.0, team=H, attrs={"cause": "jump_ball"})] + lg.shot("home-1", H, 5.0))
    for i in range(0, 61):
        eng.on_frame(frame(i, float(i), sb=ScoreboardReading(t=float(i), period=1, game_clock=720.0 - i, source="sim")),
                     None, None)
    first = eng.snapshot()
    for i in range(61, 121):
        eng.on_frame(frame(i, float(i), sb=ScoreboardReading(t=float(i), period=1, game_clock=720.0 - i, source="sim")),
                     None, None)
    second = eng.snapshot()
    # home 1 possession, away 0 (its possession opens on the next event): pace = 48 * 0.5 / minutes
    assert first.teams[H].pace == pytest.approx(24.0) and first.teams[H].seconds == pytest.approx(60.0)
    assert second.teams[H].pace == pytest.approx(12.0) and second.teams[H].seconds == pytest.approx(120.0)


def test_run_window_is_measured_in_game_minutes() -> None:
    """A 5-minute window must not shrink because 10 minutes of halftime video went by."""
    lg = Log()
    eng = engine(config=StatsConfig(run_window_s=120.0))
    eng.on_events(lg.shot("away-1", A, 10.0, value=3) + lg.shot("home-1", H, 50.0, value=2))
    for i in range(0, 101):  # 100 s of running clock
        t = float(i)
        eng.on_frame(frame(i, t, sb=ScoreboardReading(t=t, period=2, game_clock=100.0 - t, source="sim")), None, None)
    for i in range(101, 701):  # 600 s of halftime video with the clock at 0:00
        t = float(i)
        eng.on_frame(frame(i, t, sb=ScoreboardReading(t=t, period=2, game_clock=0.0, source="sim")), None, None)
    run = eng.snapshot().run
    assert run["window"] == {H: 2, A: 3} and run["window_leader"] is A
    assert scoring_run([ScoringPlay(11.0, A, 3, "x")], 700.0, 120.0)["window"][A] == 0  # video-time fallback
