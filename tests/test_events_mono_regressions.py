"""Regression tests for monocular (single fixed camera) event-engine defects found on sim train seeds.

Each test names the defect it pins down. Hand-built scenes reuse the kit in ``test_events_kit.py``;
the monocular frames carry only what one camera + scoreboard give (bboxes, ball image point, rims,
floor-homography positions), never ball height or true ball court position.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from courtvision.events.deadball import DeadBallConfig, DeadBallDetector
from courtvision.events.engine import EventEngine, EventEngineConfig
from courtvision.geometry.court import court_for
from courtvision.types import BallState, BBox, Event, EventKind, FrameState, PlayerState, Team

from test_events_kit import MODES, Scene, live, one, run_engine, two_team_scene
from test_events_rules import foul, free_throw, ft_scene


def _drop_player(frames: list[FrameState], k: int, pid: str) -> None:
    fr = frames[k]
    frames[k] = dataclasses.replace(fr, players=[p for p in fr.players if p.player_id != pid])


# ---------------------------------------------------------------------------- free-throw formation


def test_formation_start_survives_a_one_frame_detector_dropout() -> None:
    """Defect: one frame without a lane player (detector drop, ~2% of boxes per frame) reset the FT
    formation's start, so the formation never reached its hold and the free throw became a jumper."""
    court = court_for("nba")
    det = DeadBallDetector(court, config=DeadBallConfig())
    lane = [("away-1", 8.0, 34.0), ("home-2", 11.0, 34.0), ("away-2", 8.0, 16.0), ("home-3", 11.0, 16.0)]
    players = [PlayerState("home-1", Team.HOME, 20.0, 25.0)] + \
        [PlayerState(p, Team(p.split("-")[0]), x, y) for p, x, y in lane]
    for k in range(60):
        t = k / 30.0
        shown = [p for p in players if not (k == 30 and p.player_id == "home-2")]
        det.update(FrameState(k, t, shown, None))
    f = det.formation
    assert f is not None and f.shooter_id == "home-1" and f.since_t == pytest.approx(0.0)
    assert det.formation_at(1.0) is not None and det.formation_at(1.0).since_t == pytest.approx(0.0)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("who", ["home-2", "home-1"])
def test_ft_attempt_survives_lane_or_shooter_dropout_just_before_release(mode: str, who: str) -> None:
    """Defect (sim train seeds: 9 of 16 FT attempts lost in mono): a lane player or the shooter
    missing from one frame ~3 frames before the release reset the formation, and the release was
    checked against the formation of the decision frame (a few frames later), so the FT was lost."""
    s = ft_scene().start("home-1")
    free_throw(s, made=True, wait=0.8)
    frames = s.frames(mode)
    release_k = round(s.marks["ft1"] * s.fps)
    _drop_player(frames, release_k - 3, who)
    _drop_player(frames, release_k + 2, "away-3")  # and one more right after the release
    eng = _run_with_foul_frames(frames, foul(0.0))
    att = live(eng.events, EventKind.SHOT_ATTEMPT)
    assert len(att) == 1 and att[0].attrs["shot_type"] == "free_throw" and att[0].value == 1
    assert abs(att[0].t - s.marks["ft1"]) < 0.35


def _run_with_foul_frames(frames: list[FrameState], foul_ev: Event) -> EventEngine:
    eng = EventEngine(court_for("nba"))
    eng.process(frames[0])
    eng.add_manual(foul_ev)
    for fr in frames[1:]:
        eng.process(fr)
    eng.flush()
    return eng


# ---------------------------------------------------------------------------- round 2: changes of team control
# (sim train seeds: steals found late, wrong-body stints, team-label flips)


def _tracks(frames: list[FrameState]) -> list[FrameState]:
    """Give every player a stable track id (the kit leaves track_id None)."""
    ids: dict[str, int] = {}
    out = []
    for fr in frames:
        ps = [dataclasses.replace(p, track_id=ids.setdefault(p.player_id, len(ids) + 1)) for p in fr.players]
        out.append(dataclasses.replace(fr, players=ps))
    return out


def test_holder_team_label_flip_is_not_a_steal() -> None:
    """Defect (mono 300 s: 5 of 25 steal FPs): the holder's team vote flips for under a second, so perception
    shows him as the other team's provisional id "<team>-t<track>"; the engine saw a steal and a steal back."""
    s = two_team_scene().start("home-1").dribble("home-1", 3.0, 5, move=(34.0, 25.0))
    frames = _tracks(s.frames_mono())
    tid = next(p.track_id for p in frames[0].players if p.player_id == "home-1")
    for k in range(30, 54):  # 0.8 s with the wrong team label
        fr = frames[k]
        frames[k] = dataclasses.replace(fr, players=[
            dataclasses.replace(p, player_id=f"away-t{tid}", team=Team.AWAY) if p.player_id == "home-1" else p
            for p in fr.players])
    eng = run_engine(frames)
    ev = live(eng.events)
    assert not live(ev, EventKind.STEAL) and not live(ev, EventKind.TURNOVER)
    assert [e.team for e in live(ev, EventKind.POSSESSION)] == [Team.HOME]
    assert {e.player_id for e in live(ev, EventKind.DRIBBLE)} == {"home-1"}


class Mono:
    """Monocular frames built observation by observation: players stand still (kit camera boxes, floor
    homography positions) and the ball's image point is placed explicitly, so a test controls exactly which
    boxes contain it (the depth ambiguity of one camera), when it is hidden, and when it bounces."""

    def __init__(self, players: dict[str, tuple[float, float]], fps: float = 30.0) -> None:
        import numpy as np

        from test_events_kit import _CAM, _H, PLAYER_H, PLAYER_R
        self.np, self.cam, self.H = np, _CAM, _H
        self.fps = fps
        self.players = []
        for i, (pid, (x, y)) in enumerate(players.items()):
            bb = _CAM.bbox_of_cylinder(x, y, PLAYER_H, PLAYER_R)
            fx, fy = _H.image_to_court(np.array(bb.foot))
            from courtvision.types import PlayerState
            self.players.append(PlayerState(pid, Team(pid.split("-")[0]), float(fx), float(fy), bbox=bb, track_id=i + 1,
                                            jersey=pid.split("-")[1]))
        self.box = {p.player_id: p.bbox for p in self.players}
        self.frames: list[FrameState] = []
        self.marks: dict[str, float] = {}

    @property
    def t(self) -> float:
        return len(self.frames) / self.fps

    def mark(self, name: str) -> None:
        self.marks[name] = self.t

    def _frame(self, uv: tuple[float, float] | None) -> None:
        from courtvision.types import BallState, RimObservation
        from test_events_kit import _CAM
        np = self.np
        if uv is None:
            ball = BallState(visible=False)
        else:
            cxy = self.H.image_to_court(np.array(uv))
            ball = BallState(True, (float(uv[0]), float(uv[1])), (float(cxy[0]), float(cxy[1])), None, 4.0, 0.9)
        from courtvision.geometry.court import court_for
        court = court_for("nba")
        rims = [RimObservation(side, _CAM.bbox_of_circle(*court.basket_xy(side), court.rim_height, court.rim_radius))
                for side in ("left", "right")]
        self.frames.append(FrameState(len(self.frames), self.t, list(self.players), ball, rims=rims,
                                      image_size=_CAM.image_size))

    def hold(self, pid: str, seconds: float, frac: float = 0.45, side: float = 0.0) -> Mono:
        """Ball in ``pid``'s box, ``frac`` of its height below the top (chest), ``side`` half-widths right
        of its centre line."""
        bb = self.box[pid]
        for _ in range(round(seconds * self.fps)):
            self._frame(((bb.x1 + bb.x2) / 2.0 + side * bb.w / 2.0, bb.y1 + frac * bb.h))
        return self

    def dribble(self, pid: str, seconds: float, bounces: int) -> Mono:
        bb = self.box[pid]
        n = round(seconds * self.fps)
        for k in range(n):
            a = abs(math.cos(math.pi * bounces * (k + 1) / n))  # 1 at the hands, 0 at the floor
            self._frame(((bb.x1 + bb.x2) / 2.0 + 0.3 * bb.w, bb.y2 + 0.05 * bb.h - a * 0.55 * bb.h))
        return self

    def hidden(self, seconds: float) -> Mono:
        for _ in range(round(seconds * self.fps)):
            self._frame(None)
        return self

    def move(self, a: str, b: str, seconds: float) -> Mono:
        """Ball travelling in the image from ``a``'s chest to ``b``'s chest."""
        ba, bb = self.box[a], self.box[b]
        pa = ((ba.x1 + ba.x2) / 2.0, ba.y1 + 0.45 * ba.h)
        pb = ((bb.x1 + bb.x2) / 2.0, bb.y1 + 0.45 * bb.h)
        n = round(seconds * self.fps)
        for k in range(1, n + 1):
            s = k / n
            self._frame((pa[0] + s * (pb[0] - pa[0]), pa[1] + s * (pb[1] - pa[1]) - 20.0 * s * (1 - s)))
        return self


def test_brief_uncorroborated_change_of_control_is_no_steal() -> None:
    """Defect (mono 1200 s: 70 steal FPs for 18 TPs): a defender's box held the ball for a moment (depth
    ambiguity, occlusion) and the engine emitted STEAL + TURNOVER at once; when the handler had it again
    more than team_change_revert_s later, that was a second steal. A change of team control now waits
    for corroboration (the new team dribbles, shoots or is fouled while holding it)."""
    m = Mono({"home-1": (30.0, 20.0), "home-2": (20.0, 35.0), "away-1": (45.0, 22.0), "away-2": (60.0, 30.0)})
    m.dribble("home-1", 1.5, 3).hold("home-1", 0.3)
    m.move("home-1", "away-1", 0.3).hold("away-1", 0.7).hidden(2.4)
    m.hold("home-1", 0.6).dribble("home-1", 1.5, 3)
    eng = run_engine(m.frames)
    ev = live(eng.events)
    assert not live(ev, EventKind.STEAL) and not live(ev, EventKind.TURNOVER)


def test_steal_from_a_wrong_body_receiver_is_credited_against_the_passer() -> None:
    """Defect (mono: wrong-body stints): a pass seen for a moment in a teammate's box (it flew in front of
    him) became a completed pass to him, and the interception that followed a steal from him. A releaser
    who never showed control (no dribble, ambiguous, short) is spliced out: the pass is the passer's
    intercepted pass and the steal is from the passer."""
    m = Mono({"home-1": (30.0, 20.0), "home-3": (40.0, 26.0), "away-1": (41.8, 26.0), "home-2": (55.0, 30.0)})
    m.dribble("home-1", 1.5, 3).hold("home-1", 0.3)
    m.mark("release")
    m.move("home-1", "home-3", 0.3).hold("home-3", 0.3, side=0.9)  # where his and away-1's boxes overlap
    m.hold("away-1", 0.6, side=0.3).dribble("away-1", 1.8, 3)
    eng = run_engine(m.frames)
    ev = live(eng.events)
    st, to = one(ev, EventKind.STEAL), one(ev, EventKind.TURNOVER)
    assert (st.player_id, st.other_player_id) == ("away-1", "home-1") and to.player_id == "home-1"
    p = one(ev, EventKind.PASS)
    assert p.player_id == "home-1" and p.attrs["completed"] is False and abs(p.t - m.marks["release"]) < 0.2


def _strip_in_traffic() -> Scene:
    """home-1 dribbles with away-1 at his shoulder; away-1 strips him and dribbles away with home-1 chasing a
    step behind, so their image boxes overlap for the whole next second."""
    s = two_team_scene()
    s.add("home-1", 30.0, 25.0).add("away-1", 30.8, 26.5)
    s.start("home-1").dribble("home-1", 1.5, 3)
    s.mark("strip")
    s.fly(lambda pos: s.hands("away-1", pos), 0.15, apex=0.1)
    k0 = len(s.recs)
    s.dribble("away-1", 1.6, 3, move=(36.8, 26.5))
    for k in range(k0, len(s.recs)):  # home-1 chases, 1.5 ft behind
        x, y = s.recs[k].pos["away-1"]
        s.recs[k].pos["home-1"] = (x - 1.4, y - 0.6)
    s.pos["home-1"] = s.recs[-1].pos["home-1"]
    s.dribble("away-1", 1.0, 2, move=(40.0, 26.5))
    return s


def test_strip_in_traffic_is_dated_at_the_strip_with_noise_model_observations() -> None:
    """Defect (mono 300 s: 4 of 6 truth steals found 1.1-2.4 s late, each an FP + FN on STEAL and TURNOVER):
    while the stealer and the chasing victim overlap in the image, the holder's keep margins win every frame
    and the ball is often hidden, so the change of control was stamped at the stealer's first outright win.
    It is now dated back over the new team's valid candidacy (at most steal_backdate_max_s). Observations come
    from the simulator's NoiseModel (monocular), 12 seeds."""
    noise = pytest.importorskip("courtvision.sim.noise")
    from test_events_kit import camera

    late = []
    for seed in range(1, 13):
        s = _strip_in_traffic()
        obs = noise.NoiseModel(noise.NoiseConfig(monocular=True), camera=camera()).apply(
            _tracks(s.frames_3d()), seed=seed)
        st = one(live(run_engine(obs).events), EventKind.STEAL)
        assert (st.player_id, st.other_player_id) == ("away-1", "home-1")
        if abs(st.t - s.marks["strip"]) > 1.0:
            late.append((seed, round(st.t - s.marks["strip"], 2)))
    assert len(late) <= 1, late


def _and_one_after_an_unseen_gather() -> Scene:
    """home-2 passes to home-1 under the left rim; home-1 gathers in traffic (the ball unseen) and lays it in
    while fouled; the board shows +2 ~0.4 s after the make (the free throw is left out)."""
    s = two_team_scene()
    s.add("home-1", 9.0, 22.0).add("home-2", 22.0, 30.0).add("away-1", 8.0, 24.0)
    s.start("home-2").dribble("home-2", 1.0, 2)
    s.pass_to("home-1", duration=0.5, moves={"home-1": (7.5, 23.5)})
    st, top = s.ball, (6.9, 24.0, 9.0)
    s.run(0.3, lambda a, pos: tuple(st[i] + a * (top[i] - st[i]) for i in range(3)), {"home-1": (7.0, 24.0)},
          visible=False)
    s.mark("release")
    s.shoot("left", made=True, duration=0.45, apex_z=11.5, release_z=9.0)
    s.mark("made")
    s.drop_through("left")
    s.set_score(2, 0)
    s.rest(3.0)
    return s


def test_and_one_whose_shooter_was_never_the_holder_is_an_attempt_and_a_make() -> None:
    """Defect (mono 1200 s: 19 SCORE_CORRECTIONs, most of them and-ones and layups): the shooting FOUL came
    while the engine had no holder (the gather was unseen), so the dead ball killed the flight and the
    make was only a SCORE_CORRECTION. The unmatched +2 is now the fouled shooter's attempt + make."""
    s = _and_one_after_an_unseen_gather()
    eng = EventEngine(court_for("nba"))
    k_foul = round(s.marks["release"] * s.fps)
    for k, fr in enumerate(s.frames_mono()):
        if k == k_foul:
            eng.add_manual(foul(fr.t, "away-1", "home-1", n=1))
        eng.process(fr)
    eng.flush()
    ev = live(eng.events)
    att, made = one(ev, EventKind.SHOT_ATTEMPT), one(ev, EventKind.SHOT_MADE)
    assert att.player_id == made.player_id == "home-1" and made.value == 2 and att.team is Team.HOME
    assert abs(att.t - s.marks["release"]) <= 1.0 and abs(made.t - s.marks["made"]) <= 1.0
    assert made.attrs["needs_review"] is True and made.attrs["inferred"] == "scoreboard"
    assert not live(ev, EventKind.SCORE_CORRECTION) and not live(ev, EventKind.TURNOVER)


def _blocked_jumper() -> Scene:
    """home-1 rises into a jumper; away-1, a step in front with his arms up, swats it 0.1 s after the
    release (the ball goes back and down at 8-12 ft/s) and recovers it."""
    g = 32.174
    s = two_team_scene()
    s.add("home-1", 19.0, 25.0).add("away-1", 17.6, 24.4).add("home-2", 22.0, 40.0)
    s.start("home-1").dribble("home-1", 1.0, 2)
    st, top = s.ball, (19.0, 24.6, 8.6)
    s.run(0.3, lambda a, pos: tuple(st[i] + a * (top[i] - st[i]) for i in range(3)))  # the shooting motion
    s.mark("release")
    s.run(0.1, lambda a, pos: (top[0] - 1.4 * a, top[1], top[2] + 1.5 * a - 0.5 * g * (0.1 * a) ** 2))
    s.mark("block")
    b = s.ball
    s.run(0.6, lambda a, pos: (b[0] + 5.4 * a, b[1] - 1.8 * a, max(0.39, b[2] - 4.8 * a - 0.5 * g * (0.6 * a) ** 2)))
    e = s.ball
    s.run(0.4, lambda a, pos: tuple(e[i] + a * (s.hands("away-1", pos)[i] - e[i]) for i in range(3)))
    s.dribble("away-1", 1.5, 3, move=(30.0, 26.0))
    return s


def _raised_ball_in_box(frames: list[FrameState], pid: str, t0: float, t1: float) -> list[FrameState]:
    """As perception boxes a shooter: his box grows to his raised arms and a ball held / just released
    above his head (sim/render.py merges a ball above the shoulders into the person box)."""
    out = []
    for fr in frames:
        if t0 <= fr.t <= t1 and fr.ball is not None and fr.ball.visible and fr.ball.image_xy is not None:
            u, v = fr.ball.image_xy
            ps = []
            for p in fr.players:
                bb = p.bbox
                if p.player_id == pid and bb is not None and v < bb.y1 + 0.3 * bb.h:
                    p = dataclasses.replace(p, bbox=BBox(min(bb.x1, u - 3.0), min(bb.y1, v - 4.0), max(bb.x2, u + 3.0),
                                                         bb.y2))
                ps.append(p)
            fr = dataclasses.replace(fr, players=ps)
        out.append(fr)
    return out


def test_jumper_blocked_inside_the_shooters_box_is_a_block() -> None:
    """Defect (mono 1200 s: block 0 of 25; 20 of 55 shot-attempt FNs are blocked shots): with the shooter's
    box grown to his raised arms and the ball, the release was recognised only after a block 0.1 s later had
    already turned the ball back, so the flight never looked like a shot (here the recovery became a steal).
    The buffered image track around a release is now checked in hindsight: a ball rising above the head,
    then knocked down / back inside an opponent's raised-arm box away from the rim, before anyone else had
    it, is a blocked shot."""
    s = _blocked_jumper()
    frames = _raised_ball_in_box(s.frames_mono(), "home-1", s.marks["release"] - 0.3, s.marks["block"] + 0.05)
    ev = live(run_engine(frames).events)
    blk = one(ev, EventKind.BLOCK)
    assert (blk.player_id, blk.other_player_id) == ("away-1", "home-1") and abs(blk.t - s.marks["block"]) <= 0.2
    att = one(ev, EventKind.SHOT_ATTEMPT)
    assert att.player_id == "home-1" and abs(att.t - s.marks["release"]) <= 0.2
    assert one(ev, EventKind.SHOT_MISSED).attrs.get("blocked") is True
    reb = one(ev, EventKind.REBOUND)
    assert reb.player_id == "away-1" and reb.attrs["offensive"] is False
    assert not live(ev, EventKind.STEAL) and not live(ev, EventKind.TURNOVER)


def _swatted_back_jumper() -> Scene:
    """home-1 rises into a jumper; away-1, a step in front, swats it 0.1 s after the release straight back down
    into home-1's raised hands (0.25 s later), and home-1 keeps it."""
    g = 32.174
    s = two_team_scene()
    s.add("home-1", 19.0, 25.0).add("away-1", 17.6, 24.4).add("home-2", 22.0, 40.0)
    s.start("home-1").dribble("home-1", 1.0, 2)
    st, top = s.ball, (19.0, 24.6, 8.6)
    s.run(0.3, lambda a, pos: tuple(st[i] + a * (top[i] - st[i]) for i in range(3)))  # the shooting motion
    s.mark("release")
    s.run(0.1, lambda a, pos: (top[0] - 1.4 * a, top[1], top[2] + 1.5 * a - 0.5 * g * (0.1 * a) ** 2))
    s.mark("block")
    b, back = s.ball, (19.0, 24.6, 7.4)
    s.run(0.25, lambda a, pos: tuple(b[i] + a * (back[i] - b[i]) for i in range(3)))  # knocked back into his hands
    s.mark("regain")
    e = s.ball
    s.run(0.4, lambda a, pos: tuple(e[i] + a * (s.hands("home-1", pos)[i] - e[i]) for i in range(3)))
    s.dribble("home-1", 1.5, 3)
    return s


def _jump(frames: list[FrameState], pid: str, t_apex: float, height_ft: float) -> list[FrameState]:
    """As a camera sees a jump: ``pid``'s box lifted off the floor by a ballistic jump peaking at ``t_apex``."""
    g, out = 32.174, []
    for fr in frames:
        h = height_ft - 0.5 * g * (fr.t - t_apex) ** 2
        if h > 0.0:
            ps = []
            for p in fr.players:
                bb = p.bbox
                if p.player_id == pid and bb is not None:
                    dy = h * bb.h / 6.7
                    p = dataclasses.replace(p, bbox=BBox(bb.x1, bb.y1 - dy, bb.x2, bb.y2 - dy))
                ps.append(p)
            fr = dataclasses.replace(fr, players=ps)
        out.append(fr)
    return out


@pytest.mark.parametrize("jump_ft", [2.0, 0.0])
def test_jumper_swatted_back_to_the_shooter_is_a_block(jump_ft: float) -> None:
    """Defect (mono 1200 s: 5 of 17 block FNs): a shot swatted straight back into the shooter's raised hands
    never left his box (it grows to his raised arms and the ball), so no release was seen and the look-back at
    a release never ran: one continuous hold, no attempt, no block, no rebound. While he holds the ball the
    newest image observations are now tested like the look-back, with the blocker's jump (his box lifted off
    the floor) as extra evidence. A defender who does not leave the floor (the ball raised and pulled back
    down next to his raised arms) blocks nothing."""
    s = _swatted_back_jumper()
    frames = _raised_ball_in_box(s.frames_mono(), "home-1", s.marks["release"] - 0.3, s.marks["regain"] + 0.4)
    frames = _jump(frames, "away-1", s.marks["block"], jump_ft)
    # the hold-time path is off by default (pump fakes, see test_pump_fake_drawing_a_jump_is_no_block)
    assert not live(run_engine(frames).events, EventKind.BLOCK)
    ev = live(run_engine(frames, EventEngineConfig(hold_block_confirm_s=0.5)).events)
    if jump_ft == 0.0:
        assert not live(ev, EventKind.BLOCK) and not live(ev, EventKind.SHOT_ATTEMPT)
        return
    blk = one(ev, EventKind.BLOCK)
    assert (blk.player_id, blk.other_player_id) == ("away-1", "home-1") and abs(blk.t - s.marks["block"]) <= 0.2
    att = one(ev, EventKind.SHOT_ATTEMPT)
    assert att.player_id == "home-1" and abs(att.t - s.marks["release"]) <= 0.2
    assert one(ev, EventKind.SHOT_MISSED).attrs.get("blocked") is True
    reb = one(ev, EventKind.REBOUND)
    assert reb.player_id == "home-1" and reb.attrs["offensive"] is True and abs(reb.t - s.marks["regain"]) <= 0.5
    assert not live(ev, EventKind.STEAL) and not live(ev, EventKind.TURNOVER)



def _pump_fake() -> Scene:
    """home-1 raises the ball above his head as if to shoot and pulls it back down into his hands without letting
    go (a pump fake); away-1, a step in front with his arms up, jumps at it."""
    s = two_team_scene()
    s.add("home-1", 19.0, 25.0).add("away-1", 17.6, 24.4).add("home-2", 22.0, 40.0)
    s.start("home-1").dribble("home-1", 1.0, 2)
    st, top = s.ball, (19.0, 24.6, 8.6)
    s.run(0.3, lambda a, pos: tuple(st[i] + a * (top[i] - st[i]) for i in range(3)))  # the fake shooting motion
    s.mark("top")
    back = (19.0, 24.6, 7.0)
    s.run(0.15, lambda a, pos: tuple(top[i] + a * (back[i] - top[i]) for i in range(3)))  # pulled back down
    s.mark("back")
    e = s.ball
    s.run(0.4, lambda a, pos: tuple(e[i] + a * (s.hands("home-1", pos)[i] - e[i]) for i in range(3)))
    s.dribble("home-1", 1.5, 3)
    return s


@pytest.mark.parametrize("enabled", [
    pytest.param(False, marks=pytest.mark.xfail(
        strict=True, reason="known monocular failure (pre-existing, f47ea37 too): the jumping defender's lifted box "
                            "takes the faked ball, so the release look-back reads the fake's rise and pull-back next "
                            "to his raised arms as a shot turned back (attempt + blocked miss + block)")),
    pytest.param(True, marks=pytest.mark.xfail(
        strict=True, reason="known failure mode of the hold-time block path too (one reason it is off by default): "
                            "a pump fake that draws a jump looks like a shot swatted back into the shooter's hands")),
])
def test_pump_fake_drawing_a_jump_is_no_block(enabled: bool) -> None:
    """A pump fake is not a shot: no attempt, no block. One camera does not tell it from a shot turned back by a
    jumping defender; the simulator has no pump fakes, so the bench does not measure this false-positive source."""
    s = _pump_fake()
    frames = _raised_ball_in_box(s.frames_mono(), "home-1", s.marks["top"] - 0.3, s.marks["back"] + 0.3)
    frames = _jump(frames, "away-1", s.marks["top"], 2.0)
    cfg = EventEngineConfig(hold_block_confirm_s=0.5 if enabled else 0.0)
    ev = live(run_engine(frames, cfg).events)
    assert not live(ev, EventKind.BLOCK) and not live(ev, EventKind.SHOT_ATTEMPT)

_STAND = BBox(100.0, 200.0, 130.0, 265.0)  # 65 px tall: 10 px/ft at the 6.5 ft nominal height; head line v 209.75


def _turn_track(after: tuple[float, float] = (0.0, 100.0),
                jump_px: float = 0.0) -> list[tuple[float, int, tuple[float, float]]]:
    """An image ball track at 30 fps (t, frame, uv): rising at 20 ft/s (200 px/s) to a touch at index 2, then moving
    at ``after`` px/s; ``jump_px`` moves every observation after the touch that much further down (a false
    detection)."""
    dt, out = 1.0 / 30.0, []
    for i in range(7):
        if i <= 2:
            uv = (150.0, 190.0 - 200.0 * dt * i)
        else:
            uv = (150.0 + after[0] * dt * (i - 2), 190.0 - 400.0 * dt + after[1] * dt * (i - 2) + jump_px)
        out.append((i * dt, i, uv))
    return out


def test_image_block_turn_ignores_a_jump_to_a_false_detection() -> None:
    """Defect (mono 1200 s: all 3 remaining block FPs): the image "ball" jumped between the real ball and a false
    detection (a static clutter blob or another ball-like track) 2-10 ft away, so the velocity baselines across the
    jump showed an 80-600 ft/s "turn" next to a defender's raised arms. No baseline of a real touch is faster than a
    shot or a swat (retro_block_max_img_speed_fts)."""
    eng = EventEngine(court_for("nba"))
    assert eng._image_block_turn(_turn_track(), 2, _STAND, None) == pytest.approx(27.86, abs=0.05)  # a swat down
    jumped = _turn_track(jump_px=60.0)  # every observation after the touch is a blob 6 ft lower
    assert eng._image_block_turn(jumped, 2, _STAND, None) is None
    eng.config.retro_block_max_img_speed_fts = 0.0  # (the gate is what decides)
    assert eng._image_block_turn(jumped, 2, _STAND, None) is not None


def test_image_block_turn_needs_the_ball_above_the_head_at_the_touch() -> None:
    """Defect (mono 1200 s: a hold-time block FP in a dribble): the head line came from the holder's box 0.6 s
    earlier; walking away from the camera since, his box had risen 2-3 ft in the image, so a dribble's top looked
    above his head. His head line at the touch is now also taken from his box bottom then (feet_v)."""
    eng = EventEngine(court_for("nba"))
    track = _turn_track()
    assert eng._image_block_turn(track, 2, _STAND, None, feet_v=_STAND.y2) is not None  # standing where he stood
    assert eng._image_block_turn(track, 2, _STAND, None, feet_v=_STAND.y2 + 20.0) is not None  # came closer
    assert eng._image_block_turn(track, 2, _STAND, None, feet_v=_STAND.y2 - 45.0) is None  # 4.5 ft up the image
    eng.config.retro_block_head_at_touch = False
    assert eng._image_block_turn(track, 2, _STAND, None, feet_v=_STAND.y2 - 45.0) is not None


@pytest.mark.parametrize(("u", "margin_r", "found"), [
    (156.0, 0.0, True),  # inside the raised-arm box (x 154-196)
    (151.0, 0.0, False),  # the ball's centre 3 px outside it
    (151.0, 1.0, True),  # ... but its surface (radius 3.9 px at his scale) touches it
    (149.0, 1.0, False),  # beyond a ball radius
])
def test_raised_arm_opponent_touches_the_balls_surface(u: float, margin_r: float, found: bool) -> None:
    """Defect (mono 1200 s: 2 block FNs): the blocker's raised-arm box had to hold the ball's centre, but a hand
    touches the ball's surface: a ball centre 1-6 px outside his box was "nobody's". The box now grows by
    retro_block_touch_margin_r ball radii at the defender's own scale."""
    eng = EventEngine(court_for("nba"))
    eng.config.retro_block_touch_margin_r = margin_r
    eng._boxes.append((0.0, {"away-1": (BBox(160.0, 200.0, 190.0, 265.0), Team.AWAY),
                             "home-2": (BBox(140.0, 200.0, 158.0, 265.0), Team.HOME)}))
    got = eng._raised_arm_opponent(Team.HOME, (u, 210.0), 0.0)
    assert got == ("away-1" if found else None)


def test_pass_after_a_hidden_hold_is_dated_at_the_release() -> None:
    """Defect (mono 1200 s: ~40 pass FN / FP timing twins, passes 1.0-1.7 s early): a held ball hidden by
    the holder's own body (RESEARCH §2.1: a held ball vanishes into the hands) and seen again in flight was
    released at the start of the gap. The release is now back-extrapolated from the first in-flight
    observations to the holder's box."""
    s = two_team_scene().start("home-1").dribble("home-1", 1.0, 2)
    s.hold("home-1", 0.3).hold("home-1", 1.5, visible=False)  # 45 frames unseen
    s.mark("release")
    s.pass_to("home-2", duration=0.6)
    s.mark("catch")
    s.hold("home-2", 1.0)
    frames = s.frames_mono()
    k0 = round(s.marks["release"] * s.fps)
    for k in range(k0, k0 + 6):  # first seen again in flight, clear of his box
        frames[k] = dataclasses.replace(frames[k], ball=BallState(visible=False))
    p = one(live(run_engine(frames).events), EventKind.PASS)
    assert (p.player_id, p.other_player_id) == ("home-1", "home-2") and abs(p.t - s.marks["release"]) <= 0.2


def test_holder_flicker_between_overlapping_teammates_is_no_pass() -> None:
    """Defect (mono 300 s: 41 of 60 pass FPs from a "passer" who never held the ball; FP passes have a median
    flight of 0.17 s and 3.8 ft vs 0.53 s and 11 ft for true ones): a teammate's overlapping box won the ball
    for a few frames and each switch was a PASS (and a PASS back). A same-team change with a very short
    flight between close players, the ball moving less than a body width, is a holder correction."""
    m = Mono({"home-1": (30.0, 25.0), "home-3": (31.6, 25.6), "away-1": (45.0, 30.0), "away-2": (60.0, 30.0)})
    m.dribble("home-1", 1.5, 3)
    m.hold("home-3", 0.2)  # the ball, still in home-1's hands, shows in the overlapping teammate's box
    m.dribble("home-1", 1.5, 3)
    ev = live(run_engine(m.frames).events)
    assert not live(ev, EventKind.PASS)
    assert [e.team for e in live(ev, EventKind.POSSESSION)] == [Team.HOME]
