"""Round 4, monocular rebounds: hidden holds and late credits (sim train seeds 1-6, 1200 s).

- HolderTracker._hidden_gain: a neighbour's earlier win in the window no longer voids the latest winner's hidden
  hold (the latest run of wins counts).
- HolderTracker._hidden_catch: a rebound caught out of view is dated inside the hidden stretch, not at the
  reappearance.
- EventEngine._loose_rebound: a short rebound stint followed by a long free ball is no control; the next gainer
  has the rebound.
"""

from __future__ import annotations

import dataclasses

import pytest

from courtvision.events.engine import EventEngineConfig
from courtvision.events.possession import HolderTracker
from courtvision.types import BBox, EventKind, FrameState, PlayerState, Team

from test_events_kit import BALL_R, live, one, run_engine, two_team_scene

FPS = 30.0


def _vote(k: int, winner: str | None, cands: tuple[str, ...] = (), loose: bool = True) -> tuple:
    """One observed frame of the image vote: (t, frame, winner, team, teams, candidates, loose, clear)."""
    team = Team(winner.split("-")[0]) if winner else None
    return (k / FPS, k, winner, team, frozenset({Team(c.split("-")[0]) for c in cands}), frozenset(cands), loose,
            winner is not None)


def _frame(k: int, *pids: str) -> FrameState:
    players = [PlayerState(p, Team(p.split("-")[0]), 10.0, 20.0, bbox=BBox(100.0, 100.0, 130.0, 170.0)) for p in pids]
    return FrameState(k, k / FPS, players, None)


def _tracker(**over: object) -> HolderTracker:
    return HolderTracker(dataclasses.replace(EventEngineConfig(), **over))  # type: ignore[arg-type]


def _hidden_hold_votes(tr: HolderTracker) -> None:
    """A loose ball after a miss: free (no candidate) frames 0-9, a neighbour (away-3) wins frame 10, the
    rebounder (away-2) wins 11, 12 and 14, then the ball is hidden at his chest (no observation from frame 15)."""
    for k in range(10):
        tr._votes.append(_vote(k, None))
    tr._votes.append(_vote(10, "away-3", ("away-3", "away-2")))
    tr._votes.append(_vote(11, "away-2", ("away-2",)))
    tr._votes.append(_vote(12, "away-2", ("away-2",)))
    tr._votes.append(_vote(13, None))
    tr._votes.append(_vote(14, "away-2", ("away-2",)))


def test_hidden_hold_after_a_neighbours_win_is_the_latest_runs() -> None:
    """Defect (mono 1200 s: s6 665.37, s5 240.97, s4 685.23): the rebounder's hold hidden at his chest was
    confirmed by _hidden_gain only for a single winner over the last second, so one frame of the ball in a
    neighbour's box before the hold voided it; the rebound then went to the outlet receiver 1.3-1.8 s late."""
    tr = _tracker()
    _hidden_hold_votes(tr)
    k = 14 + round(0.45 * FPS)  # hidden for 0.45 s (> holder_hidden_confirm_s) since his run began
    ch = tr._hidden_gain(_frame(k, "away-2", "away-3"), k / FPS)
    assert ch is not None and ch.kind == "gain" and ch.player_id == "away-2"
    assert ch.frame_idx == 10  # dated back over his candidacy (he was a valid candidate on the neighbour's frame)
    off = _tracker(holder_hidden_run_wins=0)  # the old rule: the only winner in the window
    _hidden_hold_votes(off)
    assert off._hidden_gain(_frame(k, "away-2", "away-3"), k / FPS) is None


def test_hidden_hold_needs_a_run_of_wins_after_a_neighbours_win() -> None:
    """A single latest win after another player's wins is not a hold (the ball crossing two boxes)."""
    tr = _tracker()
    for k in range(10):
        tr._votes.append(_vote(k, None))
    tr._votes.append(_vote(10, "away-3", ("away-3",)))
    tr._votes.append(_vote(11, "away-3", ("away-3",)))
    tr._votes.append(_vote(12, "away-2", ("away-2",)))
    k = 12 + round(0.5 * FPS)
    assert tr._hidden_gain(_frame(k, "away-2", "away-3"), k / FPS) is None


@pytest.mark.parametrize("enabled", [True, False])
def test_rebound_caught_out_of_view_is_dated_inside_the_hidden_stretch(enabled: bool) -> None:
    """Defect (mono 1200 s: s3 1127.50, ball hidden 1.1 s from the catch): a loose ball that fell into the
    bodies, was caught out of view and reappeared in the rebounder's hands was dated at the reappearance."""
    tr = _tracker(holder_hidden_catch=enabled)
    votes = [_vote(k, None) for k in range(10)]  # the ball falls free, in nobody's reach
    votes += [_vote(k, "away-2", ("away-2",)) for k in range(30, 36)]  # hidden frames 10-29, then in his hands
    onset = (votes[10][0], votes[10][1])
    t, f = tr._hidden_catch(votes, onset, votes[10])
    if enabled:
        assert f == (9 + 30) // 2 and t == pytest.approx(0.5 * (9 + 30) / FPS)
    else:
        assert (t, f) == onset
    # last seen in someone's reach before the gap: the gap says nothing about the catch
    votes2 = [_vote(k, None) for k in range(9)] + [_vote(9, None, ("away-3",))] + votes[10:]
    assert tr._hidden_catch(votes2, onset, votes2[10]) == onset


def _loose_ball_scene(gainer: str):
    """A missed jumper drops off the rim through away-2's box (the ball crosses his hands for 0.4 s), carries
    on to the floor and rolls free in the open for 2 s, then ``gainer`` walks up, takes it and dribbles."""
    s = two_team_scene()
    s.add("home-1", 22.0, 25.0).add("away-2", 9.0, 20.0).add("away-1", 16.0, 14.0).add("away-3", 36.0, 44.0)
    s.add("home-3", 30.0, 8.0)
    s.start("home-1").hold("home-1", 0.8).shoot("left", made=False)
    s.fly(lambda pos: s.hands("away-2", pos), 0.5, apex=2.0)
    s.mark("touch")
    s.hold("away-2", 0.4)
    s.fly((12.0, 44.0, BALL_R), 0.6, apex=0.5)
    gx, gy = s.pos[gainer]
    s.rest(2.4, moves={gainer: (12.0, 44.7)})
    s.fly(lambda pos: s.hands(gainer, pos), 0.2, apex=0.0)
    s.mark("gain")
    s.hold(gainer, 0.5).dribble(gainer, 1.5, 3, move=(gx, gy))
    return s


@pytest.mark.parametrize("gainer", ["home-3", "away-1"])
def test_loose_ball_after_a_short_touch_is_the_next_holders_rebound(gainer: str) -> None:
    """Defect (mono 1200 s: s1 789.17 / 791.80, s4 22.57 / 25.47, s3 525.90 / 528.17): a missed shot falling
    through a player's box won his vote for a moment; the ball then rolled free in view for ~2 s and someone
    else picked it up. That pickup was a recovery ("other" possession) or a PASS from the first player, and the
    rebound stayed with the first player 2-3 s early. A short dribble-less rebound stint followed by a ball free
    in view for longer than any pass flies was no control: the rebound is the next gainer's, at his gain."""
    s = _loose_ball_scene(gainer)
    eng = run_engine(s.frames_mono())
    ev = live(eng.events)
    reb = one(ev, EventKind.REBOUND)
    # at his gain (his box reaches the resting ball as he walks up, before the pick-up mark), not at the touch
    assert reb.player_id == gainer and s.marks["touch"] + 2.0 < reb.t <= s.marks["gain"] + 0.35
    assert reb.attrs["offensive"] is (gainer == "home-3")
    assert not live(ev, EventKind.STEAL) and not live(ev, EventKind.TURNOVER) and not live(ev, EventKind.PASS)
    off = live(run_engine(s.frames_mono(), config=dataclasses.replace(EventEngineConfig(), rebound_loose_s=0.0)).events)
    assert one(off, EventKind.REBOUND).player_id == "away-2"  # the defect


def test_long_pass_held_out_of_view_is_no_loose_ball() -> None:
    """A rebounder's long outlet caught and then held out of view (the ball hidden at the receiver's chest) stays a
    rebound and a pass: only a ball seen free on most frames since the release is a loose ball."""
    s2 = two_team_scene()
    s2.add("home-1", 22.0, 25.0).add("away-2", 9.0, 20.0).add("away-1", 16.0, 14.0).add("away-3", 36.0, 44.0)
    s2.add("home-3", 30.0, 8.0)
    s2.start("home-1").hold("home-1", 0.8).shoot("left", made=False)
    s2.fly(lambda pos: s2.hands("away-2", pos), 0.5, apex=2.0)
    s2.mark("touch")
    s2.hold("away-2", 0.4)
    s2.pass_to("away-1", 0.5)
    for rec in s2.recs[-8:]:  # the ball goes out of view behind bodies before it reaches his hands
        rec.visible = False
    s2.hold("away-1", 1.7, visible=False)
    s2.hold("away-1", 0.3).dribble("away-1", 1.5, 3, move=(20.0, 20.0))
    ev = live(run_engine(s2.frames_mono()).events)
    reb = one(ev, EventKind.REBOUND)
    assert reb.player_id == "away-2" and abs(reb.t - s2.marks["touch"]) <= 0.35
    assert [(e.player_id, e.other_player_id) for e in live(ev, EventKind.PASS)] == [("away-2", "away-1")]
    # without the in-view test the hidden hold would read as a loose ball (sim train seeds: s6 997.93)
    cfg = dataclasses.replace(EventEngineConfig(), rebound_loose_seen_share=0.0)
    assert one(live(run_engine(s2.frames_mono(), config=cfg).events), EventKind.REBOUND).player_id == "away-1"


def test_full_court_outlet_in_view_is_no_loose_ball() -> None:
    """A rebounder's full-court outlet (61 ft in 1.8 s, in view all the way) is longer in the air than any pass of
    the simulator, but it covers the ground at pass speed: it stays a rebound and a pass
    (rebound_loose_max_speed_fts), not a loose ball the receiver rebounds."""
    s = two_team_scene()
    s.add("home-1", 22.0, 25.0).add("away-2", 9.0, 20.0).add("away-1", 16.0, 14.0).add("away-4", 70.0, 30.0)
    s.start("home-1").hold("home-1", 0.8).shoot("left", made=False)
    s.fly(lambda pos: s.hands("away-2", pos), 0.5, apex=2.0)
    s.mark("touch")
    s.hold("away-2", 0.4)
    s.pass_to("away-4", duration=1.8, apex=6.0)
    s.hold("away-4", 0.5).dribble("away-4", 1.5, 3, move=(80.0, 30.0))
    frames = s.frames_mono()
    ev = live(run_engine(frames).events)
    reb = one(ev, EventKind.REBOUND)
    assert reb.player_id == "away-2" and abs(reb.t - s.marks["touch"]) <= 0.35
    assert [(e.player_id, e.other_player_id) for e in live(ev, EventKind.PASS)] == [("away-2", "away-4")]
    # the speed test is what decides: without it the outlet reads as a loose ball
    cfg = dataclasses.replace(EventEngineConfig(), rebound_loose_max_speed_fts=1e9)
    assert one(live(run_engine(frames, config=cfg).events), EventKind.REBOUND).player_id == "away-4"
