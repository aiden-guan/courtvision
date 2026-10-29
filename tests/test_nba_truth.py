"""ESPN ground truth (nba/espn.py), the scorebug OCR parser (scorebug_ocr.py), ball labels and the box comparison."""

from __future__ import annotations

from courtvision.nba import espn
from courtvision.nba.evaluate import box_error, team_box
from courtvision.perception.scorebug_ocr import ESPN_2024, parse_lines
from courtvision.perception.vision_ocr import OCRLine
from courtvision.training.ball_labels import link_ball_path
from courtvision.types import Event, EventKind, Team


def summary() -> dict:  # type: ignore[type-arg]
    def ath(pid: str, name: str, jersey: str) -> dict:  # type: ignore[type-arg]
        return {"athlete": {"id": pid, "displayName": name, "jersey": jersey}, "starter": True, "didNotPlay": False,
                "stats": ["30", "2", "1-2", "0-1", "0-0", "1", "1", "0", "1", "0", "0", "1", "0", "+2"]}

    labels = ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO", "STL", "BLK", "OREB", "DREB", "PF", "+/-"]
    team_stats = [{"name": "fieldGoalsMade-fieldGoalsAttempted", "displayValue": "1-2"},
                  {"name": "threePointFieldGoalsMade-threePointFieldGoalsAttempted", "displayValue": "0-1"},
                  {"name": "freeThrowsMade-freeThrowsAttempted", "displayValue": "0-0"},
                  {"name": "offensiveRebounds", "displayValue": "0"}, {"name": "defensiveRebounds", "displayValue": "1"},
                  {"name": "totalRebounds", "displayValue": "1"}, {"name": "assists", "displayValue": "1"},
                  {"name": "steals", "displayValue": "1"}, {"name": "blocks", "displayValue": "0"},
                  {"name": "turnovers", "displayValue": "0"}, {"name": "fouls", "displayValue": "0"}]
    plays = [
        {"sequenceNumber": "4", "type": {"text": "Jump Shot"}, "text": "A One misses 26-foot three point jumper",
         "period": {"number": 1}, "clock": {"displayValue": "11:40"}, "shootingPlay": True, "scoringPlay": False,
         "pointsAttempted": 3, "team": {"id": "1"}, "participants": [{"athlete": {"id": "11"}}]},
        {"sequenceNumber": "5", "type": {"text": "Defensive Rebound"}, "text": "B One defensive rebound",
         "period": {"number": 1}, "clock": {"displayValue": "11:37"}, "team": {"id": "2"},
         "participants": [{"athlete": {"id": "21"}}]},
        {"sequenceNumber": "6", "type": {"text": "Layup Shot"}, "text": "B One makes 2-foot layup (B Two assists)",
         "period": {"number": 1}, "clock": {"displayValue": "11:30"}, "shootingPlay": True, "scoringPlay": True,
         "pointsAttempted": 2, "scoreValue": 2, "homeScore": 2, "awayScore": 0, "team": {"id": "2"},
         "participants": [{"athlete": {"id": "21"}}, {"athlete": {"id": "22"}}]},
        {"sequenceNumber": "7", "type": {"text": "Bad Pass\nTurnover"}, "text": "A One bad pass (B Two steals)",
         "period": {"number": 1}, "clock": {"displayValue": "11:20"}, "team": {"id": "1"},
         "participants": [{"athlete": {"id": "11"}}, {"athlete": {"id": "22"}}]},
    ]
    return {
        "header": {"competitions": [{"competitors": [
            {"homeAway": "away", "team": {"id": "1", "abbreviation": "AAA"}},
            {"homeAway": "home", "team": {"id": "2", "abbreviation": "BBB"}}]}]},
        "plays": plays,
        "boxscore": {"teams": [{"team": {"id": "1"}, "statistics": team_stats},
                               {"team": {"id": "2"}, "statistics": team_stats}],
                     "players": [{"team": {"id": "1"}, "statistics": [{"labels": labels, "athletes": [ath("11", "A One", "1")]}]},
                                 {"team": {"id": "2"}, "statistics": [{"labels": labels, "athletes": [
                                     ath("21", "B One", "2"), ath("22", "B Two", "3")]}]}]},
    }


def test_espn_plays_become_shots_rebounds_and_child_actions() -> None:
    acts = espn.pbp_actions(summary())
    kinds = [(a.kind, a.team, a.player_id) for a in acts]
    assert (EventKind.SHOT_MISSED, Team.AWAY, "11") in kinds
    assert (EventKind.REBOUND, Team.HOME, "21") in kinds
    assert (EventKind.SHOT_MADE, Team.HOME, "21") in kinds
    assert (EventKind.ASSIST, Team.HOME, "22") in kinds
    assert (EventKind.TURNOVER, Team.AWAY, "11") in kinds
    assert (EventKind.STEAL, Team.HOME, "22") in kinds
    three = next(a for a in acts if a.kind is EventKind.SHOT_MISSED)
    assert three.value == 3 and three.clock == 700.0


def test_espn_box_score_and_roster() -> None:
    box = espn.box_score(summary())
    assert box[Team.HOME].abbr == "BBB" and box[Team.HOME].stats["pts"] == 2
    assert {p.player_id for p in box[Team.HOME].players} == {"21", "22"}
    assert {r["jersey"] for r in espn.roster(summary())} == {"1", "2", "3"}


def test_scorebug_lines_parse_by_pattern_and_panel() -> None:
    x0 = ESPN_2024.ocr_box[0]
    lines = [OCRLine("LAL", 1.0, (22, 13, 76, 28)), OCRLine("109", 1.0, (138, 9, 114, 36)),
             OCRLine("GS", 1.0, (283, 14, 61, 30)), OCRLine("105", 1.0, (402, 7, 120, 40)),
             OCRLine("4TH", 1.0, (551, 16, 50, 28)), OCRLine("48.4", 1.0, (641, 15, 114, 36)),
             OCRLine("24", 1.0, (799, 16, 45, 32))]
    f = parse_lines(lines, ESPN_2024, origin_x=x0)
    assert f == {"away_score": 109, "home_score": 105, "period": 4, "game_clock": 48.4, "shot_clock": 24}
    merged = [OCRLine("2ND 5", 1.0, (552, 15, 290, 30)), OCRLine("11:28", 1.0, (634, 15, 150, 29))]
    g = parse_lines(merged, ESPN_2024, origin_x=x0)
    assert g["period"] == 2 and g["game_clock"] == 688.0 and g["shot_clock"] == 5


def test_ball_path_ignores_a_static_distractor_and_fills_a_flight_gap() -> None:
    cands = []
    for t in range(10):
        row = [(100.0 + 30.0 * t, 200.0, 12.0, 0.5)] if t not in (4, 5) else []
        row.append((800.0, 100.0, 10.0, 0.4))
        cands.append(row)
    colors = [[0.9, 0.1] if len(r) == 2 else [0.1] for r in cands]
    labels = link_ball_path(cands, colors)
    assert [lab.source for lab in labels] == ["detected"] * 4 + ["filled"] * 2 + ["detected"] * 4
    assert labels[4].xy == (220.0, 200.0)


def test_team_box_counts_shots_by_potential_value() -> None:
    ev = [Event("a", EventKind.SHOT_MADE, 1.0, 0, team=Team.HOME, value=3),
          Event("b", EventKind.SHOT_MISSED, 2.0, 0, team=Team.HOME, value=3),
          Event("c", EventKind.SHOT_MADE, 3.0, 0, team=Team.AWAY, value=1, attrs={"shot_type": "free_throw"}),
          Event("d", EventKind.REBOUND, 4.0, 0, team=Team.AWAY, player_id="x", attrs={"offensive": False}),
          Event("e", EventKind.REBOUND, 5.0, 0, team=Team.AWAY, player_id=None)]  # team rebound: not counted
    box = team_box(ev)
    assert box[Team.HOME]["fga"] == 2 and box[Team.HOME]["fg3m"] == 1 and box[Team.HOME]["pts"] == 3
    assert box[Team.AWAY]["ftm"] == 1 and box[Team.AWAY]["dreb"] == 1 and box[Team.AWAY]["pts"] == 1
    cmp = box_error(box, espn.box_score(summary()))
    assert 0.0 < cmp.error < 2.0


def test_scorebug_events_find_scores_resets_and_clock_stops() -> None:
    from courtvision.nba.scorebug_events import scorebug_events
    from courtvision.types import ScoreboardReading

    reads = []
    t, gc, sc, home = 0.0, 600.0, 20.0, 10
    for k in range(60):  # 20 s at 3 Hz: clock runs, the shot clock resets at 6 s (rim), home scores at 10 s
        t = k / 3.0
        if k % 3 == 0 and k < 45:
            gc -= 1.0
            sc = 24.0 if k == 18 else sc - 1.0
        if k == 30:
            home = 12
        reads.append(ScoreboardReading(t=t, home_score=home, away_score=8, period=1, game_clock=gc, shot_clock=sc))
    ev = scorebug_events(reads)
    kinds = [(e.kind, round(e.t, 2)) for e in ev]
    assert ("reset24", 6.0) in kinds
    score = [e for e in ev if e.kind == "score"]
    assert len(score) == 1 and score[0].team is Team.HOME and score[0].points == 2 and abs(score[0].t - 10.0) < 0.5
    assert any(k == "clock_stop" for k, _ in kinds)  # the game clock holds from 15 s


def test_fusion_turns_a_score_change_into_a_make_and_infers_a_rim_miss() -> None:
    from courtvision.nba.fusion import FusionConfig, fuse
    from courtvision.nba.scorebug_events import BugEvent

    ev = [Event("p1", EventKind.POSSESSION, 0.0, 0, team=Team.AWAY),
          Event("a1", EventKind.SHOT_ATTEMPT, 4.0, 0, team=Team.HOME, value=2, shot_id="s1"),
          Event("r1", EventKind.SHOT_MISSED, 5.0, 0, team=Team.HOME, value=2, shot_id="s1"),
          Event("p2", EventKind.PASS, 20.0, 0, team=Team.AWAY, player_id="x")]
    bug = [BugEvent(7.0, "score", Team.AWAY, 3),  # the shot at 5 s went in, and it was the away team's three
           BugEvent(30.0, "reset24")]  # a rim touch with no shot seen
    out = fuse(ev, bug, FusionConfig(assists=False))
    made = [e for e in out if e.kind is EventKind.SHOT_MADE]
    assert len(made) == 1 and made[0].team is Team.AWAY and made[0].value == 3
    missed = [e for e in out if e.kind is EventKind.SHOT_MISSED]
    assert len(missed) == 1 and missed[0].team is Team.AWAY and missed[0].attrs.get("inferred")
