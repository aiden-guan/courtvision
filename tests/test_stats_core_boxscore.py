"""Hand-built event logs -> exact box scores, possessions and context points (stats/boxscore.py)."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from courtvision.geometry.court import court_for
from courtvision.stats.boxscore import ShotRecord, build_box, order_events
from courtvision.stats.engine import StatsConfig
from courtvision.types import Event, EventKind, EventStatus, Roster, RosterEntry, Team

K = EventKind
H, A = Team.HOME, Team.AWAY
COURT = court_for("nba")
# Home attacks the right basket (88.75, 25); away attacks the left one (5.25, 25).
H_THREE, H_LAYUP, H_MID = (63.75, 25.0), (87.0, 25.0), (72.0, 25.0)
A_THREE, A_LAYUP = (30.25, 25.0), (7.0, 25.0)

COUNTING = ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf")


def make_roster() -> Roster:
    entries = [RosterEntry(f"home-{i}", H, str(i), f"Home {i}") for i in range(1, 9)]
    entries += [RosterEntry(f"away-{i}", A, str(i), f"Away {i}") for i in range(1, 9)]
    return Roster(entries)


class Log:
    """Tiny event-log builder (confirmed sim events unless told otherwise)."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self._n = 0
        self._s = 0

    def add(self, kind: EventKind, t: float, **kw: Any) -> Event:
        self._n += 1
        kw.setdefault("status", EventStatus.CONFIRMED)
        kw.setdefault("source", "sim")
        e = Event(event_id=f"e{self._n:06d}", kind=kind, t=t, frame_idx=int(round(t * 30)), **kw)
        self.events.append(e)
        return e

    def shot(self, pid: str, team: Team, t: float, value: int = 2, made: bool | None = True,
             xy: tuple[float, float] | None = None, dt: float = 1.0, **attrs: Any) -> str:
        self._s += 1
        sid = f"s{self._s:04d}"
        x, y = xy if xy else (None, None)
        self.add(K.SHOT_ATTEMPT, t, team=team, player_id=pid, value=value, shot_id=sid, x=x, y=y, attrs=attrs)
        if made is not None:
            kind = K.SHOT_MADE if made else K.SHOT_MISSED
            self.add(kind, t + dt, team=team, player_id=pid, shot_id=sid, value=value if made else 0)
        return sid

    def ft(self, pid: str, team: Team, t: float, made: bool = True, **attrs: Any) -> str:
        return self.shot(pid, team, t, value=1, made=made, dt=0.5, shot_type="free_throw", **attrs)

    def reb(self, pid: str | None, team: Team, t: float, offensive: bool) -> Event:
        return self.add(K.REBOUND, t, team=team, player_id=pid, attrs={"offensive": offensive})

    def poss(self, team: Team, t: float, cause: str, pid: str | None = None) -> Event:
        return self.add(K.POSSESSION, t, team=team, player_id=pid, attrs={"cause": cause})


def line(box: Any, pid: str) -> dict[str, int]:
    pl = box.players[pid]
    return {k: getattr(pl, k) for k in COUNTING if getattr(pl, k)}


def box_of(log: Log, **kw: Any) -> Any:
    return build_box(log.events, court=COURT, roster=make_roster(), config=StatsConfig(), **kw)


# --------------------------------------------------------------------------- the reference game


def reference_log() -> Log:
    """Jump ball, missed 3 + OREB putback, inbound, steal -> fast-break and-one, missed 3 + DREB,
    team shot-clock turnover, period end."""
    lg = Log()
    lg.add(K.PERIOD_START, 0.0)
    lg.poss(H, 0.0, "jump_ball", "home-1")
    lg.shot("home-1", H, 5.0, value=3, made=False, xy=H_THREE, shot_type="jumper")
    lg.reb("home-2", H, 7.0, offensive=True)
    lg.shot("home-2", H, 8.0, value=2, made=True, xy=H_LAYUP, shot_type="layup")
    lg.poss(A, 12.0, "inbound", "away-1")
    lg.add(K.TURNOVER, 15.0, team=A, player_id="away-1", attrs={"type": "bad_pass"})
    lg.add(K.STEAL, 15.0, team=H, player_id="home-3", other_player_id="away-1")
    lg.shot("home-3", H, 18.0, value=2, made=True, xy=H_LAYUP, shot_type="layup")
    lg.add(K.FOUL, 19.0, team=A, player_id="away-2", other_player_id="home-3")
    lg.ft("home-3", H, 30.0, made=True)
    lg.poss(A, 35.0, "inbound", "away-1")
    lg.shot("away-1", A, 40.0, value=3, made=False, xy=A_THREE)
    lg.reb("home-1", H, 42.0, offensive=False)
    lg.add(K.TURNOVER, 60.0, team=H, player_id=None, attrs={"type": "shot_clock"})
    lg.add(K.PERIOD_END, 70.0)
    return lg


def test_reference_player_lines_exact() -> None:
    box = box_of(reference_log())
    assert line(box, "home-1") == {"fga": 1, "fg3a": 1, "dreb": 1}
    assert line(box, "home-2") == {"pts": 2, "fgm": 1, "fga": 1, "oreb": 1}
    assert line(box, "home-3") == {"pts": 3, "fgm": 1, "fga": 1, "ftm": 1, "fta": 1, "stl": 1}
    assert line(box, "away-1") == {"fga": 1, "fg3a": 1, "tov": 1}
    assert line(box, "away-2") == {"pf": 1}
    assert line(box, "home-5") == {}  # roster player who did nothing is listed
    assert not box.players["home-3"].provisional


def test_reference_team_totals_and_team_level_entries() -> None:
    box = box_of(reference_log())
    home, away = box.teams[H], box.teams[A]
    assert (home.pts, home.fgm, home.fga, home.fg3m, home.fg3a, home.ftm, home.fta) == (5, 2, 3, 0, 1, 1, 1)
    assert (home.oreb, home.dreb, home.tov, home.team_tov, home.stl) == (1, 1, 1, 1, 1)
    assert sum(p.tov for p in box.players.values() if p.team is H) == 0  # the team TO is team-level only
    assert (away.fga, away.fg3a, away.tov, away.pf, away.pts) == (1, 1, 1, 1, 0)
    assert (home.plus_minus, away.plus_minus) == (5, -5)


def test_reference_possessions_counted_and_estimated() -> None:
    box = box_of(reference_log())
    home, away = box.teams[H], box.teams[A]
    assert (home.possessions, away.possessions) == (3, 3)
    causes = [(p.team, p.cause, p.start_t) for p in box.possessions]
    assert causes == [
        (H, "jump_ball", 0.0),
        (A, "made_shot", 9.0),  # the opponent's possession starts at the made basket
        (H, "steal", 15.0),
        (A, "free_throw", 30.5),  # the and-one FT kept the steal possession alive
        (H, "defensive_rebound", 42.0),
        (A, "turnover", 60.0),
    ]
    assert box.possessions[0].oreb == 1 and box.possessions[0].points == 2
    assert box.possessions[-1].end_reason == "period_end"
    # FGA - OREB + TOV + 0.44 FTA (team OREB / TOV included)
    assert home.possessions_est == pytest.approx(3 - 1 + 1 + 0.44 * 1)
    assert away.possessions_est == pytest.approx(1 - 0 + 1 + 0.0)


def test_reference_context_points() -> None:
    home = box_of(reference_log()).teams[H]
    assert home.second_chance_pts == 2  # putback after the OREB
    assert home.paint_pts == 4  # two restricted-area layups; the FT is not a paint point
    assert home.fast_break_pts == 2  # steal at 15 s, release at 18 s
    assert home.pts_off_turnovers == 3  # layup + and-one FT in the steal possession


def test_reference_advanced_vs_hand_calculation() -> None:
    home = box_of(reference_log(), elapsed_s=120.0, game_minutes=48.0).teams[H]
    assert home.efg == pytest.approx((2 + 0.5 * 0) / 3)
    assert home.ts == pytest.approx(5 / (2 * (3 + 0.44 * 1)))
    assert home.ppp == pytest.approx(5 / 3)
    assert home.ortg == pytest.approx(100 * 5 / 3)
    assert home.drtg == pytest.approx(0.0)
    assert home.tov_pct == pytest.approx(1 / (3 + 0.44 + 1))
    assert home.oreb_pct == pytest.approx(1 / (1 + 0))
    assert home.pace == pytest.approx(48 * ((3 + 3) / 2) / 2.0)  # 2 elapsed minutes
    assert box_of(reference_log(), elapsed_s=30.0).teams[H].pace is None  # < pace_min_elapsed_s


def test_scoring_plays_feed_runs() -> None:
    box = box_of(reference_log())
    assert [(s.team, s.points, s.t) for s in box.scoring] == [(H, 2, 9.0), (H, 2, 19.0), (H, 1, 30.5)]


# --------------------------------------------------------------------------- free throws


def test_free_throw_trips_and_possessions() -> None:
    lg = Log()
    lg.poss(A, 0.0, "inbound", "away-1")
    lg.add(K.FOUL, 3.0, team=H, player_id="home-1", other_player_id="away-2")  # fouled shooter missed: no FGA
    lg.ft("away-2", A, 10.0, made=True)
    lg.ft("away-2", A, 20.0, made=False)
    lg.reb("away-3", A, 21.0, offensive=True)  # OREB on the missed last FT continues the possession
    lg.shot("away-3", A, 22.0, value=2, made=True, xy=A_LAYUP)
    lg.poss(H, 26.0, "inbound", "home-1")
    lg.add(K.FOUL, 31.0, team=A, player_id="away-1", other_player_id="home-1")
    lg.ft("home-1", H, 35.0, made=True)
    lg.ft("home-1", H, 40.0, made=True)
    lg.add(K.DRIBBLE, 45.0, team=A, player_id="away-1")  # away has the ball: home's trip ended
    box = box_of(lg)
    assert line(box, "away-2") == {"pts": 1, "ftm": 1, "fta": 2}
    assert line(box, "away-3") == {"pts": 2, "fgm": 1, "fga": 1, "oreb": 1}
    assert line(box, "home-1") == {"pts": 2, "ftm": 2, "fta": 2, "pf": 1}
    assert box.teams[A].fga == 1 and box.teams[H].fga == 0  # FTs are never FGA
    assert [(p.team, p.cause) for p in box.possessions] == [
        (A, "inbound"), (H, "made_shot"), (A, "free_throw")]
    assert box.teams[A].second_chance_pts == 2
    shots = {s.shot_id: s for s in box.shots}
    assert shots["s0001"].is_ft and shots["s0001"].value == 1 and shots["s0001"].zone is None


def test_free_throw_detected_from_value_one_without_shot_type() -> None:
    lg = Log()
    lg.add(K.SHOT_ATTEMPT, 1.0, team=H, player_id="home-1", value=1, shot_id="s1")
    lg.add(K.SHOT_MADE, 1.5, team=H, player_id="home-1", value=1, shot_id="s1")
    box = box_of(lg)
    assert line(box, "home-1") == {"pts": 1, "ftm": 1, "fta": 1}


def test_technical_free_throw_does_not_change_possession() -> None:
    lg = Log()
    lg.poss(H, 0.0, "inbound", "home-1")
    lg.ft("away-1", A, 5.0, made=True, technical=True)
    lg.shot("home-2", H, 10.0, value=2, made=True, xy=H_MID)
    box = box_of(lg)
    assert box.teams[A].pts == 1 and box.teams[A].possessions == 0
    assert box.teams[H].possessions == 1


# --------------------------------------------------------------------------- shots


def test_blocked_shot_is_an_fga() -> None:
    lg = Log()
    sid = lg.shot("home-1", H, 5.0, value=2, made=False, xy=H_MID)
    lg.add(K.BLOCK, 5.4, team=A, player_id="away-4", other_player_id="home-1", shot_id=sid)
    lg.add(K.BLOCK, 9.0, team=A, player_id="away-5", other_player_id="home-2")  # no attempt in the log
    box = box_of(lg)
    assert line(box, "home-1") == {"fga": 1}
    assert line(box, "home-2") == {"fga": 1}
    assert line(box, "away-4") == {"blk": 1} and line(box, "away-5") == {"blk": 1}
    blocked = {s.player_id: s.blocked_by for s in box.shots}
    assert blocked == {"home-1": "away-4", "home-2": "away-5"}


def test_made_shot_without_attempt_counts_as_attempt() -> None:
    lg = Log()
    lg.add(K.SHOT_MADE, 4.0, team=H, player_id="home-4", value=3, shot_id="m1", source="manual")
    box = box_of(lg)
    assert line(box, "home-4") == {"pts": 3, "fgm": 1, "fga": 1, "fg3m": 1, "fg3a": 1}


def test_pending_shot_counts_attempt_only() -> None:
    lg = Log()
    lg.shot("home-1", H, 5.0, value=3, made=None, xy=H_THREE)
    box = box_of(lg)
    assert line(box, "home-1") == {"fga": 1, "fg3a": 1}
    assert box.shots[0].made is None


def test_shot_geometry_from_court_when_attrs_missing() -> None:
    lg = Log()
    lg.shot("home-1", H, 5.0, value=3, made=True, xy=H_THREE)
    lg.shot("home-2", H, 15.0, value=2, made=True, xy=H_LAYUP, zone="paint", distance_ft=1.0, basket="right")
    rec = {s.player_id: s for s in box_of(lg).shots}
    assert rec["home-1"].basket == "right" and rec["home-1"].zone == "above_break_three"
    assert rec["home-1"].distance_ft == pytest.approx(25.0)
    assert (rec["home-2"].zone, rec["home-2"].distance_ft) == ("paint", 1.0)  # attrs win


def test_assists_only_on_made_shots_and_recorded_on_shot() -> None:
    lg = Log()
    made = lg.shot("home-2", H, 5.0, value=2, made=True, xy=H_MID)
    lg.add(K.ASSIST, 6.0, team=H, player_id="home-1", other_player_id="home-2", shot_id=made)
    missed = lg.shot("home-2", H, 15.0, value=2, made=False, xy=H_MID)
    lg.add(K.ASSIST, 16.0, team=H, player_id="home-1", other_player_id="home-2", shot_id=missed)
    lg.add(K.ASSIST, 20.0, team=H, player_id="home-3", source="manual")  # operator entry, no shot id
    box = box_of(lg)
    assert box.players["home-1"].ast == 1 and box.players["home-3"].ast == 1
    assert {s.shot_id: s.assisted_by for s in box.shots} == {made: "home-1", missed: None}
    assert box.teams[H].ast == 2


def test_xpts_from_attrs_and_model_fallback() -> None:
    lg = Log()
    lg.shot("home-1", H, 5.0, value=3, made=False, xy=H_THREE, xfg=0.5)
    lg.shot("home-2", H, 15.0, value=2, made=True, xy=H_LAYUP)
    lg.ft("home-2", H, 25.0)

    def xfg_fn(s: ShotRecord) -> float:
        return 0.8 if s.is_ft else 0.6

    box = box_of(lg, xfg_fn=xfg_fn)
    assert box.players["home-1"].xpts == pytest.approx(1.5)
    assert box.players["home-2"].xpts == pytest.approx(0.6 * 2 + 0.8)
    assert box.teams[H].xpts == pytest.approx(1.5 + 1.2 + 0.8)
    assert box.teams[H].avg_xfg == pytest.approx((0.5 + 0.6) / 2)  # FTs excluded
    assert box.teams[H].n_xfg == 2


# --------------------------------------------------------------------------- team-level entries


def test_team_rebounds_and_team_turnovers() -> None:
    lg = Log()
    lg.poss(H, 0.0, "inbound", "home-1")
    lg.shot("home-1", H, 3.0, value=2, made=False, xy=H_MID)
    lg.reb(None, H, 5.0, offensive=True)  # out of bounds off the defense: home inbounds
    lg.shot("home-2", H, 9.0, value=2, made=False, xy=H_MID)
    lg.reb(None, A, 11.0, offensive=False)
    lg.add(K.TURNOVER, 20.0, team=A, player_id=None, attrs={"type": "shot_clock"})
    box = box_of(lg)
    home, away = box.teams[H], box.teams[A]
    assert (home.oreb, home.team_oreb, home.team_reb) == (1, 1, 1)
    assert (away.dreb, away.team_dreb, away.team_reb, away.tov, away.team_tov) == (1, 1, 1, 1, 1)
    assert all(p.oreb == p.dreb == p.tov == 0 for p in box.players.values())
    assert [(p.team, p.cause) for p in box.possessions] == [
        (H, "inbound"), (A, "defensive_rebound"), (H, "turnover")]
    assert home.possessions_est == pytest.approx(2 - 1 + 0)
    assert away.possessions_est == pytest.approx(0 - 0 + 1)


def test_rebound_type_inferred_when_flag_missing() -> None:
    lg = Log()
    lg.shot("home-1", H, 3.0, value=2, made=False, xy=H_MID)
    lg.add(K.REBOUND, 4.0, team=H, player_id="home-2")
    lg.shot("home-2", H, 6.0, value=2, made=False, xy=H_MID)
    lg.add(K.REBOUND, 7.0, team=A, player_id="away-2")
    box = box_of(lg)
    assert box.players["home-2"].oreb == 1 and box.players["away-2"].dreb == 1


def test_score_correction_adds_unattributed_team_points() -> None:
    lg = Log()
    lg.shot("home-1", H, 3.0, value=2, made=True, xy=H_MID)
    lg.add(K.SCORE_CORRECTION, 30.0, team=H, value=2, source="scoreboard", attrs={"needs_review": True})
    box = box_of(lg)
    assert box.teams[H].pts == 4 and box.teams[H].pts_unattributed == 2
    assert sum(p.pts for p in box.players.values()) == 2
    assert [s.points for s in box.scoring] == [2, 2]


def test_fouls_and_technicals() -> None:
    lg = Log()
    lg.add(K.FOUL, 1.0, team=H, player_id="home-1", other_player_id="away-1")
    lg.add(K.FOUL, 2.0, team=H, player_id="home-1", attrs={"type": "technical"})
    lg.add(K.FOUL, 3.0, team=H, player_id=None, attrs={"type": "bench_technical"})
    box = box_of(lg)
    assert (box.players["home-1"].pf, box.players["home-1"].tf) == (1, 1)
    assert (box.teams[H].pf, box.teams[H].tf) == (1, 2)


# --------------------------------------------------------------------------- possession edge cases


def test_fast_break_window_and_cause() -> None:
    lg = Log()
    lg.shot("away-1", A, 0.0, value=2, made=False, xy=A_LAYUP)
    lg.reb("home-1", H, 2.0, offensive=False)
    lg.shot("home-1", H, 8.5, value=2, made=True, xy=H_MID)  # 6.5 s after the DREB: fast break
    lg.add(K.TURNOVER, 20.0, team=A, player_id="away-2", attrs={"type": "travel"})
    lg.shot("home-2", H, 28.0, value=3, made=True, xy=H_THREE)  # TO (not a steal) start: not a fast break
    lg.poss(A, 31.0, "inbound", "away-1")
    lg.shot("away-1", A, 33.0, value=2, made=False, xy=A_LAYUP)
    lg.reb("home-3", H, 35.0, offensive=False)
    lg.add(K.STEAL, 40.0, team=A, player_id="away-3", other_player_id="home-3")
    lg.add(K.TURNOVER, 40.0, team=H, player_id="home-3", attrs={"type": "steal"})
    lg.shot("away-3", A, 48.0, value=2, made=True, xy=A_LAYUP)  # 8 s after the steal: too slow
    box = box_of(lg)
    assert box.teams[H].fast_break_pts == 2
    assert box.teams[H].pts_off_turnovers == 3
    assert box.teams[A].fast_break_pts == 0 and box.teams[A].pts_off_turnovers == 2
    assert [p.cause for p in box.possessions] == [
        "other", "defensive_rebound", "made_shot", "turnover", "made_shot", "defensive_rebound", "steal"]


def test_turnover_by_team_without_the_ball_infers_unseen_possession() -> None:
    lg = Log()
    lg.shot("home-1", H, 5.0, value=2, made=True, xy=H_MID)  # away now has the ball
    lg.add(K.TURNOVER, 40.0, team=H, player_id="home-3", attrs={"type": "steal"})  # ...yet home loses it
    lg.add(K.STEAL, 40.0, team=A, player_id="away-3", other_player_id="home-3")
    box = box_of(lg)
    assert [(p.team, p.cause) for p in box.possessions] == [
        (H, "other"), (A, "made_shot"), (H, "other"), (A, "steal")]
    assert box.teams[H].possessions == 2 and box.teams[A].possessions == 2


def test_steal_and_turnover_order_independent() -> None:
    def run(steal_first: bool) -> list[tuple[Team, str]]:
        lg = Log()
        lg.poss(A, 0.0, "inbound", "away-1")
        evs = [
            (K.STEAL, dict(team=H, player_id="home-1", other_player_id="away-1")),
            (K.TURNOVER, dict(team=A, player_id="away-1", attrs={"type": "steal"})),
        ]
        for kind, kw in (evs if steal_first else evs[::-1]):
            lg.add(kind, 5.0, **kw)
        return [(p.team, p.cause) for p in box_of(lg).possessions]

    assert run(True) == run(False) == [(A, "inbound"), (H, "steal")]


def test_period_end_closes_possession_and_period_start_resets() -> None:
    lg = Log()
    lg.add(K.PERIOD_START, 0.0, attrs={"period": 1})
    lg.poss(H, 1.0, "jump_ball", "home-1")
    lg.shot("home-1", H, 700.0, value=2, made=True, xy=H_MID)  # basket, then the buzzer
    lg.add(K.PERIOD_END, 720.0)
    lg.add(K.PERIOD_START, 900.0, attrs={"period": 2})
    lg.poss(A, 901.0, "period_start", "away-1")
    box = box_of(lg)
    assert [(p.team, p.cause, p.period) for p in box.possessions] == [(H, "jump_ball", 1), (A, "period_start", 2)]
    assert box.possessions[0].end_reason == "made_shot"
    assert box.period == 2


def test_retracted_and_provisional_events() -> None:
    lg = Log()
    lg.shot("home-1", H, 3.0, value=2, made=True, xy=H_MID)
    bad = lg.add(K.STEAL, 5.0, team=A, player_id="away-1", status=EventStatus.RETRACTED)
    lg.add(K.REBOUND, 9.0, team=A, player_id="away-2", status=EventStatus.PROVISIONAL, attrs={"offensive": False})
    box = box_of(lg)
    assert box.players["away-1"].stl == 0 and bad.event_id
    assert box.players["away-2"].provisional and not box.players["home-1"].provisional
    assert box.teams[A].provisional and not box.teams[H].provisional


def test_order_events_is_stable_by_time_and_kind() -> None:
    lg = Log()
    a = lg.add(K.PERIOD_END, 10.0)
    b = lg.add(K.SHOT_MADE, 10.0, team=H, player_id="home-1", value=2, shot_id="s9")
    c = lg.add(K.SHOT_ATTEMPT, 9.0, team=H, player_id="home-1", value=2, shot_id="s9")
    assert [e.event_id for e in order_events([a, b, c])] == [c.event_id, b.event_id, a.event_id]


# --------------------------------------------------------------------------- review regressions


def test_non_finite_coordinates_do_not_crash_the_box() -> None:
    """CourtSpec raises on nan / inf; a shot with an unprojectable release point must still count."""
    lg = Log()
    nan, inf = float("nan"), float("inf")
    lg.add(K.SHOT_ATTEMPT, 1.0, team=H, player_id="home-1", value=0, shot_id="s1", x=nan, y=25.0)
    lg.add(K.SHOT_MADE, 2.0, team=H, player_id="home-1", value=2, shot_id="s1")
    lg.add(K.SHOT_ATTEMPT, 5.0, team=H, player_id="home-2", value=3, shot_id="s2", x=inf, y=nan)
    lg.add(K.SHOT_MISSED, 6.0, team=H, player_id="home-2", shot_id="s2", x=nan, y=nan)
    lg.add(K.BLOCK, 9.0, team=A, player_id="away-1", other_player_id="home-3", x=nan, y=1.0)
    box = box_of(lg)
    assert line(box, "home-1") == {"pts": 2, "fgm": 1, "fga": 1}
    assert line(box, "home-2") == {"fga": 1, "fg3a": 1}
    assert line(box, "home-3") == {"fga": 1}
    assert all(s.x is None and s.y is None and s.zone is None for s in box.shots)


def test_flagrant_free_throws_keep_the_possession() -> None:
    """Rules note §8b: flagrant / disruptive FT trips have no rebound and the shooting team keeps
    the ball, so the made FTs must not hand the opponent a possession."""

    def run(second_shot_oreb: bool) -> list[tuple[Team, str, float]]:
        lg = Log()
        lg.poss(H, 0.0, "inbound", "home-1")
        lg.add(K.FOUL, 4.0, team=A, player_id="away-1", other_player_id="home-1", attrs={"type": "flagrant"})
        lg.ft("home-1", H, 10.0, made=True, ft_type="flagrant")
        lg.ft("home-1", H, 15.0, made=True, ft_type="flagrant")
        lg.add(K.PASS, 20.0, team=H, player_id="home-2", other_player_id="home-3")  # H throw-in, no POSSESSION
        lg.shot("home-3", H, 25.0, value=2, made=False, xy=H_MID)
        if second_shot_oreb:
            lg.reb("home-4", H, 27.0, offensive=True)
        else:
            lg.reb("away-2", A, 27.0, offensive=False)
        return [(p.team, p.cause, p.start_t) for p in box_of(lg).possessions]

    assert run(False) == [(H, "inbound", 0.0), (A, "defensive_rebound", 27.0)]
    assert run(True) == [(H, "inbound", 0.0)]  # the OREB continues the same possession


def test_flagrant_free_throws_by_the_defense_give_it_the_ball() -> None:
    lg = Log()
    lg.poss(H, 0.0, "inbound", "home-1")
    lg.ft("away-1", A, 10.0, made=True, ft_type="flagrant")  # offensive flagrant: A shoots and keeps it
    lg.ft("away-1", A, 15.0, made=True, ft_type="flagrant")
    lg.shot("away-2", A, 25.0, value=2, made=False, xy=A_LAYUP)
    lg.reb("home-2", H, 27.0, offensive=False)
    box = box_of(lg)
    assert [(p.team, p.cause, p.start_t) for p in box.possessions] == [
        (H, "inbound", 0.0), (A, "free_throw", 10.0), (H, "defensive_rebound", 27.0)]
    assert box.teams[A].pts == 2 and box.teams[A].possessions == 1


def test_only_one_assist_per_made_shot() -> None:
    """FIBA [15]: one assist per score. A confirmed (operator) assist beats the CV guess."""
    lg = Log()
    sid = lg.shot("home-2", H, 5.0, value=2, made=True, xy=H_MID)
    lg.add(K.ASSIST, 6.0, team=H, player_id="home-1", other_player_id="home-2", shot_id=sid,
           status=EventStatus.PROVISIONAL, source="cv")
    lg.add(K.ASSIST, 6.0, team=H, player_id="home-3", other_player_id="home-2", shot_id=sid, source="manual")
    box = box_of(lg)
    assert (box.players["home-1"].ast, box.players["home-3"].ast, box.teams[H].ast) == (0, 1, 1)
    assert box.shots[0].assisted_by == "home-3"


def test_value_fallback_uses_the_gather_position() -> None:
    """D8: without an event value, 2 vs 3 comes from the last grounded foot before the gather."""
    lg = Log()
    # Released 1 ft inside the right arc after gathering 1 ft beyond it (a step-in three).
    lg.add(K.SHOT_ATTEMPT, 1.0, team=H, player_id="home-1", value=0, shot_id="s1", x=65.75, y=25.0,
           attrs={"gather_x": 63.75, "gather_y": 25.0, "basket": "right"})
    lg.add(K.SHOT_MISSED, 2.0, team=H, player_id="home-1", shot_id="s1")
    lg.add(K.SHOT_ATTEMPT, 5.0, team=H, player_id="home-2", value=0, shot_id="s2", x=65.75, y=25.0,
           attrs={"basket": "right"})  # no gather point: the release point decides
    lg.add(K.SHOT_MISSED, 6.0, team=H, player_id="home-2", shot_id="s2")
    values = {s.shot_id: s.value for s in box_of(lg).shots}
    assert values == {"s1": 3, "s2": 2}


def test_rebound_without_team_does_not_warn_on_every_snapshot(caplog: pytest.LogCaptureFixture) -> None:
    lg = Log()
    lg.add(K.REBOUND, 1.0, player_id=None, team=None)
    with caplog.at_level(logging.WARNING, logger="courtvision.stats.boxscore"):
        for _ in range(3):
            box_of(lg)
    assert not caplog.records


def test_missed_shot_without_attempt_takes_its_value_from_the_court() -> None:
    lg = Log()
    lg.add(K.SHOT_MISSED, 3.0, team=H, player_id="home-1", value=0, shot_id="m1", x=H_THREE[0], y=H_THREE[1],
           source="manual")  # operator logs a missed three; no attempt event
    lg.add(K.SHOT_MISSED, 6.0, team=H, player_id="home-2", value=0, shot_id="m2", source="manual")
    box = box_of(lg)
    assert line(box, "home-1") == {"fga": 1, "fg3a": 1}
    assert line(box, "home-2") == {"fga": 1}  # no location: a 2


def test_superseded_provisional_result_does_not_keep_the_line_provisional() -> None:
    lg = Log()
    lg.add(K.SHOT_ATTEMPT, 1.0, team=H, player_id="home-1", value=2, shot_id="s1", x=H_MID[0], y=H_MID[1])
    lg.add(K.SHOT_MISSED, 2.0, team=H, player_id="home-1", shot_id="s1", status=EventStatus.PROVISIONAL, source="cv")
    lg.add(K.SHOT_MADE, 2.5, team=H, player_id="home-1", value=2, shot_id="s1", source="manual")  # operator fix
    box = box_of(lg)
    assert line(box, "home-1") == {"pts": 2, "fgm": 1, "fga": 1}
    assert not box.players["home-1"].provisional and not box.shots[0].provisional
    lg.add(K.SHOT_ATTEMPT, 5.0, team=H, player_id="home-2", value=2, shot_id="s2", status=EventStatus.PROVISIONAL)
    lg.add(K.SHOT_MADE, 6.0, team=H, player_id="home-2", value=2, shot_id="s2")
    box = box_of(lg)
    assert box.players["home-2"].provisional  # the attempt itself is still provisional
