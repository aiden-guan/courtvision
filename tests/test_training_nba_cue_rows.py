"""Tests for courtvision.training.nba_cue_rows: engine cues labelled by aligned NBA play-by-play, then fitted.

Synthetic data only: engine events with cues, aligned truth built as ``AlignedAction`` rows (and, for the CLI, an
ESPN-like summary with a scorebug scan in a game folder), NBA.com shots, scorebug reads.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from courtvision.events.calibration import CueModel, default_cue_models
from courtvision.nba.nba_stats import NBAShot
from courtvision.training import fit_cue_models as fcm
from courtvision.training.align_pbp import AlignedAction, ClockTimeline, PBPAction
from courtvision.training.nba_cue_rows import (
    CUE_ROW_KEYS,
    NBACueConfig,
    infer_nba_codes,
    label_game,
    link_nba_shots,
    main,
    run_window,
)
from courtvision.types import Event, EventKind, EventStatus, ScoreboardReading, Team

H, A = Team.HOME, Team.AWAY
K = EventKind
SHOT = {"entered_rim_zone": True, "upward_speed": 18.0, "toward_rim_cos": 0.9, "in_frontcourt": True,
        "parabola_n_pts": 9, "parabola_rmse_px": 1.0, "release_sep_ft": 2.0}
MAKE = {"crossed_inside": True, "drop_in_band": True, "drop_is_freefall": False, "rim_contact": False}
THREE = {"foot_margin_ft": 0.8, "foot_obs_quality": 0.9, "in_corner": False}
CTRL = {"control_duration_s": 0.8, "holder_dist_ft": 0.5, "track_conf": 0.9, "team_margin": 1.0}
BLOCK = {"reversal_dv": 12.0, "defender_dist_ft": 1.0, "above_shoulder": True}


class Game:
    """Aligned truth (``AlignedAction`` rows) and engine events of one synthetic game."""

    def __init__(self) -> None:
        self.aligned: list[AlignedAction] = []
        self.events: list[Event] = []

    def pbp(self, kind: EventKind, t: float, team: Team | None, *, player: str | None = "p", value: int = 0,
            made: bool | None = None, clock: float | None = None, **attrs: object) -> AlignedAction:
        a = PBPAction(len(self.aligned) + 1, 1, 720.0 - t if clock is None else clock, kind, team, player,
                      value=value, made=made, attrs=dict(attrs))
        al = AlignedAction(a, t, t, "clock")
        self.aligned.append(al)
        return al

    def true_shot(self, t_release: float, team: Team, made: bool, value: int = 2, **kw: object) -> AlignedAction:
        """A true shot whose attempt lies at ``t_release`` (truth_events puts a made FG's attempt 1 s before it)."""
        t = t_release + (1.0 if made and value != 1 else 0.0)
        attrs = {"shot_type": "free_throw"} if value == 1 else {}
        return self.pbp(K.SHOT_MADE if made else K.SHOT_MISSED, t, team, value=value, made=made, **attrs, **kw)

    def ev(self, kind: EventKind, t: float, team: Team | None, **kw: object) -> Event:
        e = Event(f"e{len(self.events):04d}", kind, t, int(t * 30), team=team, **kw)  # type: ignore[arg-type]
        self.events.append(e)
        return e

    def shot(self, t: float, team: Team | None, made: bool | None = None, value: int = 2, cues: dict | None = None,
             three: dict | None = None, make: dict | None = None, flight: float = 1.0,
             **attrs: object) -> tuple[str, str | None]:
        sid = f"s{len(self.events):04d}"
        at: dict[str, object] = {"cues": dict(cues or SHOT), **attrs}
        if three is not None:
            at["three_cues"] = dict(three)
        if value == 1:
            at["shot_type"] = "free_throw"
        att = self.ev(K.SHOT_ATTEMPT, t, team, value=value, shot_id=sid, attrs=at)
        if made is None:
            return att.event_id, None
        res = self.ev(K.SHOT_MADE if made else K.SHOT_MISSED, t + flight, team, value=value, shot_id=sid,
                      attrs={"cues": dict(make or MAKE)})
        return att.event_id, res.event_id

    def label(self, **kw: object) -> tuple[list[dict], dict]:
        return label_game(self.events, self.aligned, "g1", **kw)  # type: ignore[arg-type]


def labels(rows: list[dict], kind: str | None = None) -> dict[str, bool]:
    return {r["event_id"]: r["label"] for r in rows if kind is None or r["kind"] == kind}


def reads(t0: float, t1: float, score: Callable[[float], tuple[int, int]], hz: float = 3.0,
          clock0: float = 720.0) -> list[ScoreboardReading]:
    """Scorebug reads at ``hz``: a running clock (whole seconds) and ``score(t) -> (home, away)``."""
    out = []
    for k in range(int((t1 - t0) * hz) + 1):
        t = round(t0 + k / hz, 4)
        h, a = score(t)
        out.append(ScoreboardReading(t=t, home_score=h, away_score=a, period=1,
                                     game_clock=float(math.floor(clock0 - (t - t0))), source="ocr", confidence=0.95))
    return out


# ---------------------------------------------------------------------------------------------- pairing


def test_shots_pair_by_time_and_skip_what_the_alignment_cannot_decide() -> None:
    g = Game()
    for t, team, made in ((100.0, H, False), (130.0, A, True), (200.0, H, False)):
        g.true_shot(t, team, made)
    e1, _ = g.shot(100.8, H)
    e2, _ = g.shot(131.2, A)  # the made shot's attempt is at 130.0 (one flight before the basket)
    e3, _ = g.shot(160.0, H)  # nothing within tol + guard: a phantom shot
    e4, _ = g.shot(101.6, H)  # a second engine attempt for the shot at 100: a duplicate
    g.shot(203.2, H)  # 3.2 s from an unpaired true attempt: beyond tol, inside tol + guard
    old = g.ev(K.SHOT_ATTEMPT, 300.0, H, value=2, shot_id="old", attrs={"cues": SHOT})
    g.events.append(Event(old.event_id, K.SHOT_ATTEMPT, 300.0, 9000, team=H, value=2, shot_id="old", revision=1,
                          status=EventStatus.RETRACTED, attrs={"cues": SHOT}))  # withdrawn by the engine: no row
    rows, summary = g.label()
    assert labels(rows) == {e1: True, e2: True, e3: False, e4: False}
    assert summary["kinds"]["shot"]["skipped"] == {"near_tolerance": 1}
    by = {r["event_id"]: r for r in rows}
    assert by[e2]["dt_s"] == pytest.approx(1.2) and by[e2]["truth_event_id"].endswith("a")
    assert by[e3]["truth_event_id"] is None
    for r in rows:
        assert set(CUE_ROW_KEYS) <= set(r) and r["calibration_source"] == "nba_pbp" and r["game_id"] == "g1"
    # the same pairs with a tighter tolerance: e2 (1.2 s) is no longer paired, and its true shot is in reach -> skip
    rows, summary = g.label(config=NBACueConfig(tol_s=1.0))
    assert e2 not in labels(rows) and summary["kinds"]["shot"]["skipped"]["near_tolerance"] >= 1


def test_team_rules() -> None:
    g = Game()
    g.true_shot(100.0, H, False)
    g.true_shot(200.0, A, False)
    g.true_shot(300.0, H, False)
    g.true_shot(301.6, A, True)  # (two candidates of different teams for a team-less engine shot)
    wrong, _ = g.shot(100.4, A)  # only a home attempt there: a team error, the shot itself was real
    anon, _ = g.shot(200.5, None)  # no team: compatible with either team
    both, both_r = g.shot(300.7, None, made=False)
    g.pbp(K.REBOUND, 400.0, H, offensive=False)
    rb, _ = (g.ev(K.REBOUND, t, team, attrs={"cues": CTRL}).event_id for t, team in ((400.6, H), (401.0, A)))
    revised = g.ev(K.SHOT_ATTEMPT, 500.0, H, value=2, shot_id="r", attrs={"cues": SHOT})
    g.true_shot(500.0, A, False)
    rows, summary = g.label()
    assert labels(rows, "shot") == {anon: True, both: True}
    assert wrong not in labels(rows) and revised.event_id not in labels(rows)
    assert labels(rows, "rebound") == {rb: True}
    assert summary["kinds"]["shot"]["skipped"] == {"other_team": 2}
    assert summary["kinds"]["rebound"]["skipped"] == {"other_team": 1}
    # a miss and a make within tol of a team-less shot: the outcome depends on the pairing -> no make row
    assert both_r not in labels(rows, "make") and summary["kinds"]["make"]["skipped"] == {"ambiguous_outcome": 1}
    # the latest revision counts: re-teamed to the away team, the attempt pairs with the away shot at 500
    g.events.append(Event(revised.event_id, K.SHOT_ATTEMPT, 500.0, 15000, team=A, value=2, shot_id="r", revision=1,
                          attrs={"cues": SHOT}))
    assert labels(g.label()[0], "shot")[revised.event_id] is True


def test_window_and_its_edges() -> None:
    g = Game()
    for t, team in ((48.5, H), (100.0, H), (152.0, A)):
        g.true_shot(t, team, False)
    before, _ = g.shot(40.0, H)
    edge, _ = g.shot(50.5, H)  # its partner lies just outside the window (inside the padding)
    inside, _ = g.shot(100.3, H)
    phantom, _ = g.shot(125.0, H)
    g.shot(148.9, A)  # 3.1 s from the true shot just after the window
    after, _ = g.shot(160.0, A)
    rows, summary = g.label(window=(50.0, 150.0))
    assert labels(rows) == {edge: True, inside: True, phantom: False}
    assert before not in labels(rows) and after not in labels(rows)
    assert summary["kinds"]["shot"]["skipped"] == {"near_tolerance": 1} and summary["window"] == [50.0, 150.0]
    assert run_window({"start": 0.0, "end": 0.0}) == (0.0, math.inf)
    assert run_window({"start": 600, "end": 1200.5}) == (600.0, 1200.5)
    assert run_window({"start": None, "end": None}) == (0.0, math.inf)


def test_free_throws_pair_with_free_throws_and_fouled_misses_are_not_negatives() -> None:
    g = Game()
    g.true_shot(100.0, H, True, value=1)
    g.true_shot(160.0, H, False, value=1)  # placed by order inside the stoppage: may be seconds off
    ft, _ = g.shot(100.3, H, value=1)
    ft_far, _ = g.shot(167.0, H, value=1)  # 7 s from the true FT: unpaired, but inside the FT reach
    fg_at_ft, _ = g.shot(101.0, H)  # an FG attempt where a free throw was taken: a type error, not a phantom
    g.pbp(K.FOUL, 300.0, A, type="Shooting Foul")  # a missed shot on a shooting foul has no FGA row
    fouled, _ = g.shot(299.5, H)
    lone, _ = g.shot(400.0, H, value=1)  # a "free throw" with no free throw anywhere near
    rows, summary = g.label()
    assert labels(rows) == {ft: True, lone: False}
    assert summary["kinds"]["shot"]["skipped"] == {"near_tolerance": 1, "shot_type": 1, "shooting_foul": 1}
    assert {fg_at_ft, fouled, ft_far}.isdisjoint(labels(rows))


def test_attempts_while_the_clock_stops_for_free_throws_are_never_negatives() -> None:
    # The clock stops at 6:00 from 120 s to 200 s. The trip's free throws were given clock 6:00 but placed 40 s off
    # (as when their clock misses the stoppage and the running clock maps them): the stoppage still claims them.
    def clock(t: float) -> float:
        return 480.0 - t if t < 120.0 else 360.0 if t < 200.0 else 360.0 - (t - 200.0)
    rd = [ScoreboardReading(t=round(k / 3.0, 4), home_score=0, away_score=0, period=1,
                            game_clock=float(math.floor(clock(k / 3.0))), source="ocr", confidence=0.95)
          for k in range(3 * 400)]
    timeline = ClockTimeline.fit(rd)
    assert any(t0 <= 150.0 <= t1 and v == 360.0 for t0, t1, v in timeline.plateaus(1))
    g = Game()
    for t in (80.0, 81.0):  # FT 1 and 2 of the trip at the 6:00 stoppage, placed far too early
        g.true_shot(t, H, False, value=1, clock=360.0)
    ft, _ = g.shot(150.0, H, value=1)
    fg, _ = g.shot(170.0, H)  # an FG attempt in that dead ball: a free throw seen as a field goal
    lone_ft, _ = g.shot(300.0, H, value=1)  # the clock runs: no free throw is being taken
    rows, summary = g.label(reads=rd, timeline=timeline)
    assert labels(rows) == {lone_ft: False}
    assert summary["kinds"]["shot"]["skipped"] == {"ft_stoppage": 2}
    rows, summary = g.label(reads=rd)  # without the timeline the stoppage is unknown: both become negatives
    assert labels(rows) == {ft: False, fg: False, lone_ft: False}


# ---------------------------------------------------------------------------------------------- make


def _make_game() -> tuple[Game, dict[str, str | None]]:
    g = Game()
    g.true_shot(101.0, H, True)  # basket at 102 s, on the bug at 104 s
    g.true_shot(150.0, H, False)
    g.true_shot(250.0, H, True)  # a basket the bug never shows (conflict)
    g.true_shot(300.0, A, True, value=1)  # free throw, on the bug at 302 s
    g.true_shot(350.0, H, False)
    g.true_shot(352.0, H, True)  # the put-back of the miss at 350: on the bug at 355 s
    ids: dict[str, str | None] = {}
    _, ids["made"] = g.shot(101.3, H, made=True, flight=0.9)
    _, ids["missed"] = g.shot(150.4, H, made=False)
    ids["phantom_att"], ids["phantom"] = g.shot(200.0, H, made=False)
    _, ids["conflict"] = g.shot(250.3, H, made=True, flight=0.9)
    _, ids["ft"] = g.shot(300.2, A, made=True, value=1, flight=0.2)
    _, ids["first_of_putback"] = g.shot(350.3, H, made=False)  # the engine saw the first shot only
    ids["far"] = g.shot(500.0, H, made=False)[1]  # no scorebug read there: no truth
    return g, ids


def _make_score(t: float) -> tuple[int, int]:
    return (2 if t >= 104.0 else 0) + (2 if t >= 355.0 else 0), 1 if t >= 302.0 else 0


def test_make_labels_from_the_play_by_play_checked_by_the_scorebug() -> None:
    g, ids = _make_game()
    rows, summary = g.label(reads=reads(0.0, 420.0, _make_score))
    make = {r["event_id"]: r for r in rows if r["kind"] == "make"}
    got = {name: (make[eid]["label"], make[eid]["label_source"], make[eid]["clean"])
           for name, eid in ids.items() if eid in make}
    assert got == {"made": (True, "pbp+scorebug", True), "missed": (False, "pbp+scorebug", True),
                   "phantom": (False, "scorebug", True), "ft": (True, "pbp+scorebug", True)}
    assert make[ids["phantom"]]["scorebug"] == "missed" and make[ids["phantom"]]["truth_event_id"] is None
    skipped = summary["kinds"]["make"]["skipped"]
    # the bug alone would call the first shot of the put-back "made": the put-back guard stops it
    assert skipped == {"scorebug_conflict": 1, "ambiguous_outcome": 1, "no_truth_coverage": 1}
    assert labels(rows, "shot")[ids["phantom_att"]] is False
    assert summary["kinds"]["shot"]["skipped"] == {"no_truth_coverage": 1}
    # without the scorebug: play-by-play labels only (the conflict is no longer seen), nothing for the phantoms
    rows, summary = g.label()
    make = {r["event_id"]: r for r in rows if r["kind"] == "make"}
    assert make[ids["conflict"]]["label"] is True and make[ids["conflict"]]["label_source"] == "pbp"
    assert not make[ids["conflict"]]["clean"] and ids["phantom"] not in make and ids["far"] not in make
    assert summary["kinds"]["make"]["skipped"] == {"unpaired+no_scorebug": 2, "ambiguous_outcome": 1}


def test_scorebug_make_needs_one_basket_of_the_shooting_team() -> None:
    g = Game()
    g.true_shot(100.0, H, True)
    _, first = g.shot(100.2, H, made=True, flight=0.8)
    _, dup = g.shot(101.4, None, made=True, flight=0.5)  # a second engine shot for the same basket
    _, other = g.shot(200.0, A, made=False)  # no true shot: the bug shows a home basket after it
    _, gap = g.shot(300.0, H, made=False)  # the bug is not read after it

    def score(t: float) -> tuple[int, int]:
        return (2 if t >= 103.0 else 0) + (2 if t >= 203.0 else 0), 0
    rd = [r for r in reads(0.0, 400.0, score) if not 301.0 < r.t < 306.0]
    rows, summary = g.label(reads=rd)
    make = {r["event_id"]: r for r in rows if r["kind"] == "make"}
    assert make[first]["label"] is True and make[first]["label_source"] == "pbp"  # the basket is shared with dup
    assert make[first]["scorebug"] is None and not make[first]["clean"]
    assert dup not in make and other not in make and gap not in make
    assert summary["kinds"]["make"]["skipped"] == {"unpaired+bug_shared": 1, "unpaired+bug_other_team": 1,
                                                   "unpaired+bug_unread": 1}


def test_scorebug_alone_must_agree_with_the_play_by_play() -> None:
    g = Game()
    g.true_shot(400.0, H, True)  # the bug shows this basket late (410 s): outside the quiet window
    g.true_shot(500.0, H, True)  # on the bug at 503 s
    _, late = g.shot(400.5, A, made=True, flight=0.8)  # the home basket, given to the away team
    _, early = g.shot(494.0, H, made=True, flight=7.2)  # its release was dated far too early: no true partner
    _, ghost = g.shot(600.0, H, made=True)  # the bug shows +2 at 603 s, the play-by-play no basket

    def score(t: float) -> tuple[int, int]:
        return 2 * (t >= 410.0) + 2 * (t >= 503.0) + 2 * (t >= 603.0), 0
    rows, summary = g.label(reads=reads(0.0, 700.0, score))
    # "missed" on the bug alone, but a true make lies right there: no row. A basket on the bug alone does not say
    # which flight went in (the engine may have missed the one that did): no row either
    assert not {late, early, ghost} & set(labels(rows, "make"))
    assert summary["kinds"]["make"]["skipped"] == {"scorebug_unsupported": 1, "scorebug_unattributed": 2}


# ---------------------------------------------------------------------------------------------- three


def test_three_prefers_nba_values_and_locations() -> None:
    g = Game()
    g.true_shot(100.0, H, False, value=3, clock=620.0)  # a missed three ...
    g.true_shot(101.3, H, True, value=2, clock=618.0)  # ... and its put-back
    g.true_shot(200.0, A, False, value=2, clock=520.0)  # ESPN says 2, NBA.com says 3
    three, three_r = g.shot(100.5, H, made=False, value=3, three=THREE, gather_x=27.0, gather_y=29.0)
    putback, putback_r = g.shot(101.6, H, made=True, three={**THREE, "foot_margin_ft": -20.0}, gather_x=6.5,
                                gather_y=25.5)
    corner, _ = g.shot(200.4, A, three=THREE)
    nba = [NBAShot(1, 620.0, H, 3, False, 28.0, 30.0, 24.0), NBAShot(1, 618.0, H, 2, True, 6.0, 25.0, 1.0),
           NBAShot(1, 520.0, A, 3, False, 70.0, 47.0, 23.0)]
    # ESPN alone: both candidates (a three and a two) lie within tol of either engine shot -> skipped
    rows, summary = g.label()
    assert labels(rows, "three") == {corner: False}
    assert summary["kinds"]["three"]["skipped"] == {"ambiguous_value": 2}
    assert summary["kinds"]["make"]["skipped"] == {"ambiguous_outcome": 2}
    rows, summary = g.label(nba_shots=nba)
    th = {r["event_id"]: r for r in rows if r["kind"] == "three"}
    assert {k: (r["label"], r["label_source"]) for k, r in th.items()} == {
        three: (True, "nba_location"), putback: (False, "nba_location"), corner: (True, "nba")}
    assert th[three]["loc_err_ft"] == pytest.approx(math.hypot(1.0, 1.0), abs=0.01)
    assert summary["nba_shots_linked"] == 3 and summary["nba_espn_value_conflicts"] == 1
    mk = {r["event_id"]: (r["label"], r["label_source"]) for r in rows if r["kind"] == "make"}
    assert mk == {three_r: (False, "nba_location"), putback_r: (True, "nba_location")}
    # locations that do not single one candidate out (both within the margin) decide nothing
    near = [NBAShot(1, 620.0, H, 3, False, 27.0, 29.5, 24.0), NBAShot(1, 618.0, H, 2, True, 30.0, 29.0, 24.0)]
    assert three not in labels(g.label(nba_shots=near)[0], "three")


def test_nba_shot_link_and_team_codes(tmp_path: Path) -> None:
    g = Game()
    for t, team, made, value in ((100.0, H, False, 2), (130.0, A, True, 3), (160.0, H, True, 2), (190.0, A, False, 2),
                                 (220.0, H, False, 3)):
        g.true_shot(t, team, made, value)
    acts = [al.action for al in g.aligned]
    path = tmp_path / "nba_pbp.csv"
    cols = ["n", "period", "clock", "team", "person", "type", "sub", "res", "dist", "x", "y", "side", "poss", "sh",
            "sa", "qual"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for k, a in enumerate(acts):
            w.writerow({"n": k + 1, "period": 1, "clock": a.clock + 0.4, "team": "X" if a.team is H else "Y",
                        "type": a.value, "res": "M" if a.made else "X", "x": 50.0, "y": 50.0, "dist": 10})
    assert infer_nba_codes(path, acts) == {"X": H, "Y": A}
    assert infer_nba_codes(path, acts[:1]) == {}  # one team's code only: not enough to tell
    shots = [NBAShot(1, a.clock + 0.4, a.team, a.value, bool(a.made), 1.0, 2.0, 3.0) for a in acts]  # type: ignore[arg-type]
    assert link_nba_shots(g.aligned, shots) == dict(enumerate(shots))
    shots[0] = NBAShot(1, acts[0].clock, A, 2, False, 1.0, 2.0, 3.0)  # the other team: not the same shot
    assert 0 not in link_nba_shots(g.aligned, shots)


# ---------------------------------------------------------------------------------------------- rebounds, steals, blocks


def test_rebound_steal_block() -> None:
    g = Game()
    g.pbp(K.REBOUND, 300.0, H, offensive=False)
    g.pbp(K.REBOUND, 500.0, A, player=None)  # a team rebound: not in the truth, but it explains an engine rebound
    g.pbp(K.STEAL, 600.0, A)
    g.pbp(K.BLOCK, 800.0, H)
    ok = {kind: g.ev(ek, t, team, attrs={"cues": cues}).event_id
          for kind, ek, t, team, cues in (("rebound", K.REBOUND, 300.5, H, CTRL), ("steal", K.STEAL, 600.7, A, CTRL),
                                          ("block", K.BLOCK, 800.3, H, BLOCK))}
    neg = {kind: g.ev(ek, t, team, attrs={"cues": cues}).event_id
           for kind, ek, t, team, cues in (("rebound", K.REBOUND, 400.0, A, CTRL), ("steal", K.STEAL, 700.0, H, CTRL),
                                           ("block", K.BLOCK, 900.0, A, BLOCK))}
    g.ev(K.REBOUND, 500.4, A, attrs={"cues": CTRL})
    g.ev(K.BLOCK, 801.0, A, attrs={"cues": BLOCK})  # the block was the home team's
    g.ev(K.REBOUND, 300.2, H, attrs={"team_rebound": True})  # no cues: closer, but takes no partner and gives no row
    rows, summary = g.label()
    for kind in ("rebound", "steal", "block"):
        assert labels(rows, kind) == {ok[kind]: True, neg[kind]: False}
    assert summary["kinds"]["rebound"]["skipped"] == {"team_rebound": 1}
    assert summary["kinds"]["block"]["skipped"] == {"other_team": 1}


# ---------------------------------------------------------------------------------------------- CLI on a game folder


def _espn(plays: list[dict]) -> dict:
    def ath(pid: str, name: str) -> dict:
        return {"athlete": {"id": pid, "displayName": name, "jersey": pid}, "starter": True, "didNotPlay": False,
                "stats": []}

    return {"header": {"competitions": [{"competitors": [
        {"homeAway": "away", "team": {"id": "1", "abbreviation": "AAA"}},
        {"homeAway": "home", "team": {"id": "2", "abbreviation": "BBB"}}]}]},
        "plays": plays,
        "boxscore": {"teams": [], "players": [
            {"team": {"id": "1"}, "statistics": [{"labels": [], "athletes": [ath("11", "A One"), ath("12", "A Two")]}]},
            {"team": {"id": "2"}, "statistics": [{"labels": [], "athletes": [ath("21", "B One"), ath("22", "B Two")]}]}]}}


def _play(seq: int, clock: int, typ: str, text: str, team: str, parts: list[str], **kw: object) -> dict:
    return {"sequenceNumber": str(seq), "type": {"text": typ}, "text": text, "period": {"number": 1},
            "clock": {"displayValue": f"{clock // 60}:{clock % 60:02d}"}, "team": {"id": team},
            "participants": [{"athlete": {"id": p}} for p in parts], **kw}


def _shot_play(seq: int, clock: int, team: str, parts: list[str], made: bool, value: int, home: int, away: int,
               text: str = "") -> dict:
    words = f"{'makes' if made else 'misses'} {'three point jumper' if value == 3 else 'jumper'}"
    return _play(seq, clock, "Jump Shot", text or f"X {words}", team, parts, shootingPlay=True, scoringPlay=made,
                 pointsAttempted=value, homeScore=home, awayScore=away)


def _game_folder(root: Path, *, nba_csv: bool = True) -> dict[str, str]:
    """A synthetic game folder: ESPN summary, a 3 Hz scorebug scan (clock 12:00 at 10 s, running), one run."""
    plays = [
        _shot_play(1, 700, "2", ["21"], False, 2, 0, 0),  # video ~30 s
        _play(2, 697, "Defensive Rebound", "A One defensive rebound", "1", ["11"]),
        _shot_play(3, 690, "1", ["11"], True, 2, 0, 2),  # ~40 s, on the bug at 42 s
        _play(4, 680, "Bad Pass\nTurnover", "B One bad pass (A Two steals)", "2", ["21", "12"]),
        _shot_play(5, 660, "1", ["12"], True, 3, 0, 5),  # ~70 s, on the bug at 72 s
        _shot_play(6, 640, "2", ["21", "11"], False, 2, 0, 5, text="A One blocks B One 's layup"),
        _play(7, 637, "Defensive Rebound", "A Two defensive rebound", "1", ["12"]),
        _shot_play(8, 600, "2", ["21"], True, 2, 2, 5),  # ~130 s, on the bug at 132 s
        _shot_play(9, 570, "1", ["11"], False, 3, 2, 5),  # ~160 s
        _play(10, 567, "Defensive Rebound", "B Two defensive rebound", "2", ["22"]),
    ]
    (root / "espn_summary.json").write_text(json.dumps(_espn(plays)))
    with (root / "scorebug.jsonl").open("w") as fh:
        for k in range(300 * 3 + 1):
            t = round(10.0 + k / 3.0, 4)
            away = 5 if t >= 72.0 else 2 if t >= 42.0 else 0
            home = 2 if t >= 132.0 else 0
            fh.write(json.dumps({"t": t, "home_score": home, "away_score": away, "period": 1,
                                 "game_clock": float(math.floor(720.0 - (t - 10.0))), "shot_clock": None}) + "\n")
    (root / "game.json").write_text(json.dumps({"video": "none", "espn_event": "0"}))
    run = root / "runs" / "base"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"start": 0.0, "end": 0.0}))
    g = Game()
    ids = {}
    ids["a1"], ids["r1"] = g.shot(30.2, H, made=False)
    ids["reb"] = g.ev(K.REBOUND, 33.3, A, attrs={"cues": CTRL}).event_id
    ids["a2"], ids["r2"] = g.shot(39.3, A, made=True, flight=0.9)
    ids["steal"] = g.ev(K.STEAL, 50.4, A, attrs={"cues": CTRL}).event_id
    ids["a3"], ids["r3"] = g.shot(69.0, A, made=True, value=3, three=THREE, flight=1.1, gather_x=80.5, gather_y=30.5)
    ids["block"] = g.ev(K.BLOCK, 89.9, A, attrs={"cues": BLOCK}).event_id
    ids["a4"], ids["r4"] = g.shot(110.0, H, made=False)  # a phantom shot
    ids["reb2"] = g.ev(K.REBOUND, 112.0, H, attrs={"cues": CTRL}).event_id  # and its phantom rebound
    ids["a5"], ids["r5"] = g.shot(129.5, H, made=True, flight=0.8)
    ids["a6"], _ = g.shot(159.8, H)  # the away team's shot, given to the home team
    ids["a7"], _ = g.shot(400.0, H)  # after the scan: no truth there
    (run / "events.jsonl").write_text("".join(json.dumps(e.to_dict()) + "\n" for e in g.events))
    if nba_csv:
        cols = ["n", "period", "clock", "team", "person", "type", "sub", "res", "dist", "x", "y", "side", "poss",
                "sh", "sa", "qual"]
        shots = [(700, "B", 2, "X", 70.0, 50.0), (690, "A", 2, "M", 5.0, 50.0), (660, "A", 3, "M", 100 * (1 - 80 / 94), 40.0),
                 (640, "B", 2, "X", 95.0, 50.0), (600, "B", 2, "M", 80.0, 30.0), (570, "A", 3, "X", 20.0, 90.0)]
        with (root / "nba_pbp.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for k, (clock, team, value, res, x, y) in enumerate(shots):
                w.writerow({"n": k + 1, "period": 1, "clock": float(clock), "team": team, "type": value, "res": res,
                            "dist": 10, "x": round(x, 3), "y": y})
    return ids


def test_cli_labels_a_game_folder(tmp_path: Path) -> None:
    root = tmp_path / "2024-01-01-AAA-BBB"
    root.mkdir()
    ids = _game_folder(root)
    assert main([str(root), "--tag", "base"]) == 0
    out = root / "runs" / "base" / "cue_rows.jsonl"
    rows = fcm.load_cue_rows([out])  # the fitter's loader accepts them
    assert rows and all(r["calibration_source"] == "nba_pbp" and r["game_id"] == root.name and r["run"] == "base"
                        for r in rows)
    got = {(r["kind"], r["event_id"]): r["label"] for r in rows}
    assert {k[1]: v for k, v in got.items() if k[0] == "shot"} == {ids["a1"]: True, ids["a2"]: True, ids["a3"]: True,
                                                                     ids["a4"]: False, ids["a5"]: True}
    assert {k[1]: v for k, v in got.items() if k[0] == "make"} == {ids["r1"]: False, ids["r2"]: True,
                                                                     ids["r3"]: True, ids["r4"]: False,
                                                                     ids["r5"]: True}
    assert {k[1]: v for k, v in got.items() if k[0] == "three"} == {ids["a3"]: True}
    assert {k[1]: v for k, v in got.items() if k[0] == "rebound"} == {ids["reb"]: True, ids["reb2"]: False}
    assert got[("steal", ids["steal"])] is True and got[("block", ids["block"])] is True
    by = {(r["kind"], r["event_id"]): r for r in rows}
    assert by[("three", ids["a3"])]["label_source"] == "nba" and by[("three", ids["a3"])]["loc_err_ft"] < 1.0
    assert all(by[("make", ids[k])]["clean"] for k in ("r1", "r2", "r3", "r4", "r5"))
    summary = json.loads((root / "runs" / "base" / "cue_rows.summary.json").read_text())
    assert summary["kinds"]["shot"]["skipped"] == {"other_team": 1, "no_truth_coverage": 1}
    assert summary["nba_pbp"] == {"codes": {"A": "away", "B": "home"}, "shots": 6}
    assert summary["nba_shots_linked"] == 6 and summary["alignment"]["n_actions"] > 0
    # --out, --tol; a missing run is bad input
    other = tmp_path / "rows.jsonl"
    assert main([str(root), "--tag", "base", "--out", str(other), "--tol", "1.0"]) == 0 and other.exists()
    assert (tmp_path / "rows.summary.json").exists()
    assert main([str(root), "--tag", "nope"]) == 2
    # the NBA loop's call: --out <rows>.partial, renamed into place afterwards; the summary gets its final name
    part = root / "runs" / "base" / "cue_rows.jsonl.partial"
    (root / "runs" / "base" / "cue_rows.summary.json").unlink()
    assert main([str(root), "--tag", "base", "--out", str(part)]) == 0
    assert part.read_text() == out.read_text() and (root / "runs" / "base" / "cue_rows.summary.json").exists()


def test_cli_without_nba_csv(tmp_path: Path) -> None:
    root = tmp_path / "g"
    root.mkdir()
    ids = _game_folder(root, nba_csv=False)
    assert main([str(root), "--tag", "base"]) == 0
    rows = [json.loads(line) for line in (root / "runs" / "base" / "cue_rows.jsonl").open()]
    three = [r for r in rows if r["kind"] == "three"]
    assert [(r["event_id"], r["label"], r["label_source"]) for r in three] == [(ids["a3"], True, "espn")]


# ---------------------------------------------------------------------------------------------- fitting


def _synthetic_game(seed: int, n_shots: int = 150) -> Game:
    """Shots every 12 s; the engine sees 90% of them (shot-like cues) and adds phantoms (weaker cues) between them."""
    rng = np.random.default_rng(seed)
    g = Game()
    for k in range(n_shots):
        t = 20.0 + 12.0 * k
        team = H if k % 2 == 0 else A
        made, value = bool(rng.random() < 0.45), 3 if rng.random() < 0.35 else 2
        g.true_shot(t, team, made, value)
        if rng.random() < 0.9:
            cues = {"entered_rim_zone": True, "upward_speed": rng.uniform(4, 22), "toward_rim_cos": rng.uniform(0.2, 1.0),
                    "in_frontcourt": bool(rng.random() < 0.9), "parabola_n_pts": int(rng.integers(3, 14)),
                    "parabola_rmse_px": rng.uniform(0, 6), "release_sep_ft": rng.uniform(0.5, 3.0)}
            three = {"foot_margin_ft": float(rng.normal(3.0 if value == 3 else -6.0, 2.5)), "foot_obs_quality": 0.8}
            make = {"crossed_inside": bool(rng.random() < (0.85 if made else 0.15)), "drop_in_band": bool(rng.random() < 0.5)}
            g.shot(t + float(np.clip(rng.normal(0.0, 0.6), -2.0, 2.0)), team, made=made, value=value, cues=cues,
                   three=three, make=make)
        if rng.random() < 0.3:
            cues = {"entered_rim_zone": True, "upward_speed": rng.uniform(0, 10), "toward_rim_cos": rng.uniform(-0.6, 0.6),
                    "in_frontcourt": bool(rng.random() < 0.5), "parabola_n_pts": int(rng.integers(2, 8)),
                    "parabola_rmse_px": rng.uniform(2, 10), "release_sep_ft": rng.uniform(0.0, 1.5)}
            g.shot(t + 6.0, team, cues=cues)  # 6 s from either neighbour: beyond tol + guard
    return g


def test_fit_on_rows_of_several_games(tmp_path: Path) -> None:
    rows, paths = [], []
    for k in range(4):
        g = _synthetic_game(k)
        game_rows, _ = label_game(g.events, g.aligned, f"game{k}")
        rows += game_rows
        paths.append(tmp_path / f"game{k}.jsonl")
        fcm.write_cue_rows(game_rows, paths[-1])
    assert {r["game_id"] for r in rows} == {f"game{k}" for k in range(4)}
    cfg = fcm.CueFitConfig(n_boot=50)
    models, reports = fcm.fit_cue_models(fcm.load_cue_rows(paths), config=cfg)
    rep = reports["shot"]
    assert rep.status == "platt" == models.models["shot"].status and rep.calibration_source == "nba_pbp"
    assert rep.split == "by_game" and len(rep.games_holdout) == 1 and rep.n_holdout > 0  # round(0.3 * 4) games
    held = set(rep.games_holdout)
    test = [r for r in rows if r["kind"] == "shot" and r["game_id"] in held]
    assert rep.n_holdout == len(test) == rep.calibration["n"] and rep.n_train == rep.n_rows - rep.n_holdout >= 300
    assert len(rep.calibration["reliability"]) == 15 and rep.gate.n_required == 30

    def log_loss(model: CueModel) -> float:
        p = model.predict_many([r["features"] for r in test])
        y = np.array([r["label"] for r in test], dtype=float)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    assert log_loss(models.models["shot"]) < log_loss(default_cue_models()["shot"])
    assert {"make", "three"} <= set(reports)  # fewer rows than the platt minimum: kept, evaluated on the held-out game
    # the CLI on the per-game files (one --jsonl per game)
    out = tmp_path / "cue_models.json"
    argv = [a for p in paths for a in ("--jsonl", str(p))] + ["--out", str(out), "--n-boot", "50"]
    assert fcm.main(argv) == 0
    doc = json.loads(out.read_text())
    assert doc["for_live"] is True and doc["models"]["shot"]["calibration_source"] == "nba_pbp"
    assert doc["calibration"]["shot"]["split"] == "by_game"
    assert json.loads((tmp_path / "cue_models.report.json").read_text())["shot"]["status"] == "platt"


def test_fit_on_one_game_is_in_sample_and_never_certified(tmp_path: Path) -> None:
    g = _synthetic_game(7, n_shots=280)
    rows, _ = label_game(g.events, g.aligned, "only")
    assert sum(r["kind"] == "shot" for r in rows) >= 300
    _, reports = fcm.fit_cue_models(rows, config=fcm.CueFitConfig(n_boot=20))
    rep = reports["shot"]
    assert rep.split == "in_sample" and rep.n_holdout == 0 and rep.games_holdout == []
    assert rep.status == "platt"  # refit on its training rows ...
    assert not rep.gate.certified and rep.gate.reason.startswith("no held-out games")  # ... but never certified
    path, out = tmp_path / "rows.jsonl", tmp_path / "cue.json"
    fcm.write_cue_rows(rows, path)
    assert fcm.main(["--jsonl", str(path), "--out", str(out), "--n-boot", "20"]) == 0
    assert json.loads(out.read_text())["calibration"]["shot"]["split"] == "in_sample"
