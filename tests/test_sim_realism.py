"""Scenario-level regression tests for the simulator's physical realism (bodies, jumps, passes, steals,
fouls, out-of-bounds, rebounds, substitutions, the horn) and the operator-run scoreboard model.

Each test pins a realism fix: see the SimConfig / NoiseConfig / ScoreboardConfig comments for the sources
and statuses of the constants involved.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from courtvision.sim.noise import ScoreboardConfig, ScoreboardModel
from courtvision.sim.simulator import GameSimulator, SimConfig, SimResult
from courtvision.types import EventKind as K
from courtvision.types import Team


@pytest.fixture(scope="module")
def game() -> SimResult:
    return GameSimulator(SimConfig(seconds=300.0, fps=30.0, seed=4)).run()


def _kind(r: SimResult, kind: K) -> list:
    return [e for e in r.events if e.kind is kind]


# ---------------------------------------------------------------------------- bodies


def test_bodies_never_interpenetrate_or_stand_on_the_ball(game: SimResult) -> None:
    """FIBA Art. 33.1 cylinder principle: every pair keeps the body-contact distance (locked players too)
    and nobody else's body centre sits on a held / dribbled ball (it was 7.2% of frames < 1.0 ft)."""
    cfg = game.config
    pos = np.array([[(p.x, p.y) for p in f.players] for f in game.frames])
    d = np.hypot(pos[:, :, None, 0] - pos[:, None, :, 0], pos[:, :, None, 1] - pos[:, None, :, 1])
    iu = np.triu_indices(10, 1)
    pair = d[:, iu[0], iu[1]].min(axis=1)
    assert pair.min() >= cfg.body_contact_ft - 0.1, pair.min()
    assert float(np.mean(pair < 1.2)) == 0.0
    held = on_ball = 0
    for i, f in enumerate(game.frames):
        h = game.holders[i]
        if h is None:
            continue
        held += 1
        bx, by = f.ball.court_xy
        on_ball += min(math.hypot(p.x - bx, p.y - by) for p in f.players if p.player_id != h) < 1.0
    assert held > 3000 and on_ball / held < 0.005  # only the hand-over frames of a catch / strip


def test_closest_defender_at_release_is_never_inside_the_shooter(game: SimResult) -> None:
    fga = [a for a in _kind(game, K.SHOT_ATTEMPT) if a.value in (2, 3)]
    assert fga and min(a.attrs["closest_defender_ft"] for a in fga) >= game.config.body_contact_ft - 0.1


def test_raised_ball_stays_inside_the_holders_box_and_shooters_jump(game: SimResult) -> None:
    """Person boxes include raised arms and the jump (standing reach ~1.33 x stature, NBA verticals
    2-3.5 ft): a ball held above the head is inside its holder's box, and layups / dunks leave the floor."""
    raised = 0
    for i, f in enumerate(game.frames):
        h = game.holders[i]
        if h is None or f.ball.z < game.config.player_height_ft:
            continue
        b, (u, v) = f.player(h).bbox, f.ball.image_xy
        assert b.x1 - 2 <= u <= b.x2 + 2 and b.y1 - 2 <= v <= b.y2 + 2, (f.t, h)
        raised += 1
    assert raised > 10
    feet = np.array(game.feet_z)
    for a in _kind(game, K.SHOT_ATTEMPT):
        if a.attrs["shot_type"] in ("layup", "dunk"):
            k = [p.player_id for p in game.frames[a.frame_idx].players].index(a.player_id)
            assert feet[a.frame_idx - 6:a.frame_idx + 7, k].max() > 1.0, a
    assert feet.max() <= 3.7  # never above the vertical-leap bound


# ---------------------------------------------------------------------------- passes


def test_passes_go_around_the_rim_and_stay_low(game: SimResult) -> None:
    """No pass crosses the zone above the rim (within 4.5 ft of a basket, above rim height - 1 ft); arcs
    peak at most at ``pass_apex_ft`` (overthrown lobs at ``oob_lob_apex_ft``)."""
    cfg, court = game.config, game.court
    passes = _kind(game, K.PASS)
    assert len(passes) > 40
    for p in passes:
        end = int(round(p.attrs["catch_t"] * cfg.fps)) if "catch_t" in p.attrs else p.frame_idx + 30
        seg = game.frames[p.frame_idx:end + 1]
        assert max(f.ball.z for f in seg) <= cfg.pass_apex_ft + 0.05, p
        for f in seg:
            for side in ("left", "right"):
                bx, by = court.basket_xy(side)
                near = math.hypot(f.ball.court_xy[0] - bx, f.ball.court_xy[1] - by) <= cfg.pass_rim_zone_ft
                assert not (near and f.ball.z > court.rim_height - cfg.pass_rim_clear_ft), (p, f.t)
    lob = GameSimulator(SimConfig(seconds=240.0, fps=15.0, seed=3, turnover_rate=0.6,
                                  turnover_mix={"oob": 1.0})).run()
    outs = [t for t in _kind(lob, K.TURNOVER) if t.attrs["type"] == "out_of_bounds"]
    assert len(outs) >= 3
    for t in outs:  # the overthrown lob: from its release to the whistle
        p = max((p for p in _kind(lob, K.PASS) if p.frame_idx <= t.frame_idx), key=lambda p: p.frame_idx)
        assert max(f.ball.z for f in lob.frames[p.frame_idx:t.frame_idx + 1]) <= lob.config.oob_lob_apex_ft + 0.05


# ---------------------------------------------------------------------------- steals


def test_stealer_attacks_and_the_victim_reacts_late() -> None:
    """After a strip the stealer pushes toward his basket at once; the robbed player reacts 0.4-0.7 s
    later, so the two are apart one second after the steal (they stood ~1.9 ft apart for ~1 s before)."""
    gaps, toward = [], []
    for seed in (1, 2, 3):
        r = GameSimulator(SimConfig(seconds=200.0, fps=30.0, seed=seed, turnover_rate=0.5,
                                    turnover_mix={"strip": 1.0})).run()
        sides = {}
        for e in r.events:
            if e.kind is K.SHOT_ATTEMPT:
                sides[(e.team, r.frames[e.frame_idx].scoreboard.period)] = e.attrs["basket"]
        for st in _kind(r, K.STEAL):
            j = st.frame_idx + int(r.config.fps)
            if j >= len(r.frames):
                continue
            a, v = r.frames[j].player(st.player_id), r.frames[j].player(st.other_player_id)
            if a is None or v is None:
                continue
            gaps.append(math.hypot(a.x - v.x, a.y - v.y))
            side = sides.get((st.team, r.frames[j].scoreboard.period))
            if side is not None and a.speed > 2.0:
                bx, by = r.court.basket_xy(side)
                toward.append((bx - a.x) * a.vx + (by - a.y) * a.vy > 0.0)
    assert len(gaps) >= 15
    assert np.median(gaps) >= 4.0, np.median(gaps)
    assert toward and np.mean(toward) >= 0.7


def test_defenders_close_out_on_catch_and_shoot_threes() -> None:
    """At a pass release the receiver's defender closes out: the nearest defender to a catch-and-shoot
    3-point shooter is closer at the catch than at the pass release (it was not: 12.3 ft both, then 9.1 ft
    at the shot), and corner 3s are no longer almost always wide open (> 6 ft: 89% before). The league
    calibration (3PA buckets 1.3/20.4/42.2/36.0% vs NBA tracking 1.7/17/41.7/39.4%) runs on 18 full games."""
    rel, catch, shot, corner = [], [], [], []
    for seed in (1, 2, 3):
        r = GameSimulator(SimConfig(seconds=900.0, fps=15.0, seed=seed)).run()
        passes = [e for e in _kind(r, K.PASS) if e.attrs.get("completed")]
        for a in _kind(r, K.SHOT_ATTEMPT):
            if a.value != 3:
                continue
            if a.attrs["zone"] == "corner_three":
                corner.append(a.attrs["closest_defender_ft"])
            p = max((q for q in passes if q.frame_idx < a.frame_idx and q.other_player_id == a.player_id),
                    key=lambda q: q.frame_idx, default=None)
            if not a.attrs["catch_and_shoot"] or p is None or a.t - p.t > 3.0:
                continue

            def nearest(i: int, a=a) -> float:
                f = r.frames[i]
                me = f.player(a.player_id)
                return min(math.hypot(q.x - me.x, q.y - me.y) for q in f.players if q.team is not a.team)

            rel.append(nearest(p.frame_idx))
            catch.append(nearest(int(round(p.attrs["catch_t"] * r.config.fps))))
            shot.append(a.attrs["closest_defender_ft"])
    assert len(shot) >= 25 and len(corner) >= 8
    assert np.median(rel) - np.median(catch) >= 1.5, (np.median(rel), np.median(catch))
    assert np.median(shot) <= 7.5
    assert np.mean(np.array(corner) > 6.0) <= 0.75


def test_interceptions_are_reads_after_the_release() -> None:
    """The interceptor reads the pass after it is thrown: at the release he is usually still away from the
    passing lane (he used to leave his man 0.9 s early and stand in it, a median 0.9-1.1 ft off the line)."""
    lane = []
    for seed in (1, 2, 3, 4):
        r = GameSimulator(SimConfig(seconds=240.0, fps=30.0, seed=seed, turnover_rate=0.5,
                                    turnover_mix={"intercept": 1.0})).run()
        to = {e.frame_idx: e for e in _kind(r, K.TURNOVER)}
        passes = [p for p in _kind(r, K.PASS) if not p.attrs["completed"]]
        for st in _kind(r, K.STEAL):
            if to[st.frame_idx].attrs["type"] != "bad_pass":
                continue
            p = max((q for q in passes if q.frame_idx <= st.frame_idx), key=lambda q: q.frame_idx)
            a = np.array([p.x, p.y])
            recv = r.frames[st.frame_idx].player(p.other_player_id)
            ab = np.array([recv.x, recv.y]) - a
            d = r.frames[p.frame_idx].player(st.player_id)
            q = np.array([d.x, d.y])
            u = float(np.clip(((q - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0))
            lane.append(float(np.hypot(*(q - (a + u * ab)))))
    assert len(lane) >= 8
    assert np.median(lane) >= 2.5 and np.mean(np.array(lane) >= 3.0) >= 0.4, sorted(lane)


def test_poke_away_strips_are_loose_balls() -> None:
    """A strip is never a 1-frame holder swap on the dribble spot (58% were, with 0.0 ft of ball travel):
    the ball is out of every hand before the steal. Poke-aways are loose for a while and travel a few feet,
    and some are recovered by the offence (no steal, no turnover) or by a teammate of the deflector (the
    STEAL stays with the first deflector, note §2a)."""
    pokes, offence_kept, steals = [], 0, 0
    for seed in (1, 2, 3, 4):
        r = GameSimulator(SimConfig(seconds=200.0, fps=30.0, seed=seed, turnover_rate=0.5,
                                    turnover_mix={"strip": 1.0})).run()
        H = r.holders
        turnovers = {e.frame_idx: e for e in _kind(r, K.TURNOVER)}
        for st in _kind(r, K.STEAL):
            i = st.frame_idx
            if turnovers[i].attrs["type"] != "steal":
                continue
            steals += 1
            assert H[i - 1] is None, st  # the ball was off the victim's hand before anyone took it
            j = i - 1
            while j > 0 and H[j] is None:
                j -= 1
            loose_s = r.frames[i].t - r.frames[j + 1].t
            if loose_s > 0.31:  # longer than a hand-to-hand rip (0.3 s): a poke-away
                a, b = r.frames[j].ball.court_xy, r.frames[i].ball.court_xy
                pokes.append(math.hypot(a[0] - b[0], a[1] - b[1]))
        for k in range(1, len(H)):  # the offence recovered its own poked ball: same team, no pass, no steal
            if H[k - 1] is None and H[k] is not None:
                j = k - 1
                while j > 0 and H[j] is None:
                    j -= 1
                if H[j] is None or r.frames[k].t - r.frames[j + 1].t < 0.31:
                    continue
                same = r.frames[k].player(H[k]).team is r.frames[j].player(H[j]).team
                quiet = not [e for e in r.events if j <= e.frame_idx <= k and e.kind in
                             (K.PASS, K.SHOT_ATTEMPT, K.STEAL, K.REBOUND, K.FOUL, K.TURNOVER)]
                offence_kept += same and quiet
    assert steals >= 15 and len(pokes) >= 5
    assert 2.0 <= float(np.median(pokes)) <= 10.0, pokes
    assert offence_kept >= 1


# ---------------------------------------------------------------------------- out of bounds


def test_offence_keeps_a_ball_the_defence_knocks_out() -> None:
    """A pass deflected out of bounds by the defence: no TURNOVER, no STEAL, no POSSESSION change; the
    offence throws in (rules note §2a: no steal when the ball becomes dead)."""
    r = GameSimulator(SimConfig(seconds=300.0, fps=15.0, seed=2, deflect_out_rate=0.5, turnover_rate=0.0,
                                foul_rate=0.0, nonshoot_foul_rate=0.0)).run()
    assert not _kind(r, K.TURNOVER) and not _kind(r, K.STEAL)
    incomplete = [p for p in _kind(r, K.PASS) if not p.attrs["completed"]]
    assert len(incomplete) >= 3
    retained = 0
    for p in incomplete:
        nxt = [e for e in r.events if e.t > p.t and e.kind in (K.PASS, K.POSSESSION, K.SHOT_ATTEMPT)]
        if nxt and nxt[0].kind is K.PASS and nxt[0].team is p.team:
            holder = r.frames[nxt[0].frame_idx].player(nxt[0].player_id)
            retained += not r.court.in_bounds(holder.x, holder.y)  # a throw-in by the same team
    assert retained >= 2


def test_dead_ball_turnover_types() -> None:
    mix = {"step_out": 0.4, "fumble_out": 0.3, "offensive_foul": 0.3}
    r = GameSimulator(SimConfig(seconds=400.0, fps=15.0, seed=5, turnover_rate=0.5, turnover_mix=mix)).run()
    tos = _kind(r, K.TURNOVER)
    types = {t.attrs["type"] for t in tos}
    assert {"out_of_bounds", "offensive_foul"} <= types
    for t in tos:
        if t.attrs["type"] == "offensive_foul":  # a charge: FOUL on the ball handler at the same frame
            fouls = [f for f in _kind(r, K.FOUL) if f.frame_idx == t.frame_idx]
            assert fouls and fouls[0].player_id == t.player_id and fouls[0].attrs["type"] == "offensive"
            assert fouls[0].attrs["n_ft"] == 0
    steps = [t for t in tos if t.attrs["type"] == "out_of_bounds" and r.holders[t.frame_idx] == t.player_id]
    assert steps  # a dribbler stepping out: he still has the ball at the whistle
    for t in steps:
        p = r.frames[t.frame_idx].player(t.player_id)
        assert not r.court.in_bounds(p.x, p.y)


# ---------------------------------------------------------------------------- rebounds


def test_rebounds_are_contested_and_won_by_the_first_to_the_ball() -> None:
    """Players from both teams pursue the miss (contesting the landing spot, box-outs) but the rebounder
    has the inside position: he is the nearest to the ball at control (he used to grab balls two
    opponents stood on). Opponent within 3.5 ft at control (NBA.com's "contested" distance): band
    unverified (no sourced contested-rebound share)."""
    nearest, opp = [], []
    for seed in (1, 2, 3):
        r = GameSimulator(SimConfig(seconds=300.0, fps=15.0, seed=seed)).run()
        for e in _kind(r, K.REBOUND):
            if e.player_id is None:
                continue
            f = r.frames[e.frame_idx]
            p, (bx, by) = f.player(e.player_id), f.ball.court_xy
            mine = math.hypot(p.x - bx, p.y - by)
            others = [q for q in f.players if q.player_id != p.player_id]
            nearest.append(min(math.hypot(q.x - bx, q.y - by) for q in others) >= mine - 0.2)
            opp.append(min(math.hypot(q.x - p.x, q.y - p.y) for q in others if q.team is not p.team) < 3.5)
    assert len(nearest) >= 15
    assert np.mean(nearest) >= 0.75
    assert 0.15 <= np.mean(opp) <= 0.7


def test_players_crash_and_box_out_while_the_shot_is_in_the_air() -> None:
    """Rebound positioning starts at the release (players react to the shot, not to the miss): bodies move
    during the flight (they used to stand still for the ~1.2-1.5 s to the miss decision, mean speed
    ~1.1-1.6 ft/s and all ten exactly still in 5-16% of flight frames). When the miss is decided, offensive
    players within 15 ft of the rim are mostly boxed out: the nearest defender is on the basket side and in
    contact (< 3 ft); that share was ~0.09 (defenders stood in their help spots)."""
    speeds, still, n_frames, between, sealed, n_men = [], 0, 0, 0, 0, 0
    for seed in (1, 2, 3, 4, 5):
        r = GameSimulator(SimConfig(seconds=300.0, fps=15.0, seed=seed)).run()
        res = {e.shot_id: e for e in r.events if e.kind in (K.SHOT_MADE, K.SHOT_MISSED)}
        blocked = {e.shot_id for e in _kind(r, K.BLOCK)}
        for a in _kind(r, K.SHOT_ATTEMPT):
            e = res.get(a.shot_id)
            if a.value == 1 or e is None or a.shot_id in blocked:
                continue
            for f in r.frames[a.frame_idx + 1:e.frame_idx + 1]:
                v = [math.hypot(p.vx, p.vy) for p in f.players]
                speeds.append(float(np.mean(v)))
                still += all(x < 0.05 for x in v)
                n_frames += 1
            if e.kind is K.SHOT_MISSED and a.attrs["distance_ft"] > 10.0:
                f = r.frames[e.frame_idx]
                bx, by = r.court.basket_xy(a.attrs["basket"])
                defence = [q for q in f.players if q.team is not a.team]
                for p in f.players:
                    dm = math.hypot(p.x - bx, p.y - by)
                    if p.team is not a.team or dm > 15.0:
                        continue
                    q = min(defence, key=lambda q: math.hypot(q.x - p.x, q.y - p.y))
                    inside = math.hypot(q.x - bx, q.y - by) < dm
                    between += inside
                    sealed += inside and math.hypot(q.x - p.x, q.y - p.y) < 3.0
                    n_men += 1
    assert n_frames > 300 and n_men > 20
    assert np.mean(speeds) >= 3.0, np.mean(speeds)
    assert still == 0
    assert between / n_men >= 0.55 and sealed / n_men >= 0.35, (between / n_men, sealed / n_men)


def test_no_one_controls_a_blocked_ball_on_the_block_frame() -> None:
    """A blocked ball leaves at 8-15 ft/s: it is controlled only after ``block_loose_s`` (it used to be
    "caught" on the block frame itself, 3 of 20 blocks within 0.1 s)."""
    delays = []
    for seed in (5, 6, 7, 8, 9):
        cfg = SimConfig(seconds=300.0, fps=30.0, seed=seed, block_rate={k: 0.5 for k in SimConfig().block_rate})
        r = GameSimulator(cfg).run()
        blocks = {e.shot_id: e for e in _kind(r, K.BLOCK)}
        for i, e in enumerate(r.events):
            if e.kind is K.SHOT_MISSED and e.shot_id in blocks:
                nxt = next((x for x in r.events[i + 1:] if x.kind in (K.REBOUND, K.TURNOVER, K.PERIOD_END)), None)
                if nxt is not None and nxt.kind is K.REBOUND and nxt.player_id is not None:
                    delays.append(nxt.t - blocks[e.shot_id].t)
    assert len(delays) >= 8
    assert min(delays) >= cfg.block_loose_s - 1e-6, sorted(delays)[:3]


def test_blocks_are_legal_within_reach_and_some_come_from_help_defenders() -> None:
    """A block is a touch at the hand or on a still-rising ball (on its downward flight above the ring it would
    be goaltending). On at-rim tries the blocker is within reach of the ball and may be a rotating help defender
    (it used to be the closest defender every time); jumpers keep the contest model (the closest defender
    within block_max_defender_ft at the release: no flying closeouts in the sim, a documented approximation)."""
    reach_top = SimConfig().standing_reach_ratio * SimConfig().player_height_ft + SimConfig().jump_ft["block"][1]
    help_blocks = close_blocks = 0
    for seed in (5, 6, 7, 8, 9, 10):
        r = GameSimulator(SimConfig(seconds=400.0, fps=30.0, seed=seed,
                                    block_rate={k: 0.5 for k in SimConfig().block_rate})).run()
        att = {e.shot_id: e for e in _kind(r, K.SHOT_ATTEMPT)}
        for b in _kind(r, K.BLOCK):
            a, i = att[b.shot_id], b.frame_idx
            z = [f.ball.z for f in r.frames[a.frame_idx:i]]
            if b.t - a.t > 0.11:  # (a touch within ~0.1 s of the release is at the hand)
                assert all(z1 >= z0 - 1e-6 for z0, z1 in zip(z, z[1:], strict=False)), b  # rising up to the touch
            f = r.frames[i]
            blocker, (bx, by) = f.player(b.player_id), f.ball.court_xy
            close = a.attrs["shot_type"] in ("layup", "dunk", "hook", "tip")
            assert f.ball.z <= reach_top + 0.5, b
            if close:  # within reach of the ball
                assert math.hypot(blocker.x - bx, blocker.y - by) <= 4.0, b
            else:  # the contest model: the closest defender, within block_max_defender_ft at the release
                at = r.frames[a.frame_idx].player(b.player_id)
                assert math.hypot(at.x - a.x, at.y - a.y) <= r.config.block_max_defender_ft + 0.1, b
            if close:
                close_blocks += 1
                rel = r.frames[a.frame_idx]
                closest = min((p for p in rel.players if p.team is not a.team),
                              key=lambda p: math.hypot(p.x - a.x, p.y - a.y))
                help_blocks += closest.player_id != b.player_id
    assert close_blocks >= 10 and help_blocks >= 1, (close_blocks, help_blocks)


def test_scored_on_team_retrieves_a_made_basket_itself() -> None:
    """After a made basket a player of the scored-on team picks the ball up under the basket (off the
    bounce), steps out behind the end line and throws in; POSSESSION is that pickup (the first frame of
    confirmed control). The official used to toss the ball to an inbounder already out of bounds, so
    POSSESSION was stamped ~1 s after the player could be seen taking the ball."""
    n = 0
    for seed in (2, 3, 4, 5, 6):
        r = GameSimulator(SimConfig(seconds=300.0, fps=30.0, seed=seed)).run()
        ev = r.events
        for i, e in enumerate(ev):
            if e.kind is not K.SHOT_MADE or e.value == 1:
                continue
            nxt = [x for x in ev[i + 1:] if x.kind in (K.POSSESSION, K.PASS, K.FOUL, K.PERIOD_END, K.TIMEOUT,
                                                         K.SHOT_ATTEMPT)]  # (an and-one's FT comes first)
            if len(nxt) < 2 or nxt[0].kind is not K.POSSESSION or nxt[1].kind is not K.PASS:
                continue
            pos, inbound = nxt[0], nxt[1]
            assert pos.attrs["cause"] == "made_shot" and pos.team is not e.team
            f = r.frames[pos.frame_idx]
            p, (bx, by) = f.player(pos.player_id), r.court.basket_xy(e.attrs["basket"])
            assert math.hypot(f.ball.court_xy[0] - p.x, f.ball.court_xy[1] - p.y) <= 2.0, pos
            assert f.ball.z <= r.config.made_pickup_z_ft + 1e-6
            assert math.hypot(p.x - bx, p.y - by) <= 8.0, pos  # under the basket, not at the throw-in spot
            assert pos.t - e.t <= 2.5
            thrower = r.frames[inbound.frame_idx].player(inbound.player_id)
            assert inbound.player_id == pos.player_id and not r.court.in_bounds(thrower.x, thrower.y)
            n += 1
    assert n >= 8


# ---------------------------------------------------------------------------- substitutions


def test_substitutes_walk_on_as_separate_people() -> None:
    """The substitute waits at the table (a visible person, SimResult.waiting) and steps on where he
    waited once the outgoing player has left the floor: no body changes jersey in place."""
    r = GameSimulator(SimConfig(seconds=600.0, fps=15.0, seed=6, sub_prob=1.0, min_stint_s=30.0,
                                min_rest_s=20.0)).run()
    subs = _kind(r, K.SUBSTITUTION)
    assert len(subs) >= 4
    for s in subs:
        i = s.frame_idx
        waited = {pid: (x, y) for pid, _, x, y in r.waiting[i - 1]}
        assert s.player_id in waited  # visible at the table the frame before
        entered = r.frames[i].player(s.player_id)
        assert math.hypot(entered.x - waited[s.player_id][0], entered.y - waited[s.player_id][1]) < 1e-6
        out = r.frames[i - 1].player(s.other_player_id)
        assert not r.court.in_bounds(out.x, out.y)  # he walked off first
        assert math.hypot(out.x - entered.x, out.y - entered.y) > 4.0  # two people, apart (> 2 body widths)


def test_no_substitution_takes_a_player_off_from_inside_the_court() -> None:
    """The outgoing player walks all the way off before his substitute steps on: no SUBSTITUTION frame has
    him inside the court lines (a 5 s cutoff used to teleport him to the table in ~10% of substitutions),
    and play never restarts with one pending (none happens while the game clock runs)."""
    n = 0
    for seed in (1, 2, 4):
        r = GameSimulator(SimConfig(seconds=600.0, fps=10.0, seed=seed, sub_prob=0.8, min_stint_s=60.0,
                                    min_rest_s=30.0)).run()
        for e in _kind(r, K.SUBSTITUTION):
            out = r.frames[e.frame_idx - 1].player(e.other_player_id)
            assert not r.court.in_bounds(out.x, out.y), (seed, e)
            b0, b1 = r.frames[e.frame_idx - 1].scoreboard, r.frames[e.frame_idx].scoreboard
            assert b1.game_clock == b0.game_clock, (seed, e)
            n += 1
    assert n >= 20


# ---------------------------------------------------------------------------- fouls & free throws


@pytest.fixture(scope="module")
def foul_game() -> SimResult:
    # (a foul-dominated game: only ~15% of segments end in an FGA, so a long one is needed to see every trip
    # type, and-ones included)
    return GameSimulator(SimConfig(seconds=1200.0, fps=10.0, seed=8, foul_rate=0.3, nonshoot_foul_rate=0.4,
                                   and_one_rate={"2": 0.5, "3": 0.5}, three_foul_rate=0.2)).run()


def test_every_foul_awards_its_free_throws(foul_game: SimResult) -> None:
    """Shooting fouls: 2 (or 3 on a three) FTs with no FGA, or 1 FT after a made and-one; non-shooting
    fouls: none below the team-foul penalty, 2 in it (RuleSpec.bonus); offensive fouls: a turnover."""
    r = foul_game
    attempts = _kind(r, K.SHOT_ATTEMPT)
    kinds = set()
    for foul in _kind(r, K.FOUL):
        if foul.t > r.frames[-1].t - 30.0:
            continue
        n = foul.attrs["n_ft"]
        kinds.add((foul.attrs["type"], n, bool(foul.attrs.get("and_one"))))
        later = [a for a in attempts if a.frame_idx >= foul.frame_idx and a.player_id == foul.other_player_id]
        fts = [a for a in attempts if a.t > foul.t and a.value == 1][:n]
        if n:
            assert len(fts) == n and all(a.player_id == foul.other_player_id for a in fts), foul
        if foul.attrs.get("and_one"):  # the FGA (made) is at the whistle frame
            made = [e for e in _kind(r, K.SHOT_MADE) if e.player_id == foul.other_player_id and e.t >= foul.t]
            assert made and later and later[0].value in (2, 3)
        elif foul.attrs["type"] == "shooting":  # fouled before the release: no FGA until the FTs
            fga = [a for a in attempts if foul.t <= a.t < fts[0].t and a.value > 1]
            assert not fga, foul
        elif foul.attrs["type"] == "personal" and n == 0:  # a throw-in, not a free throw, comes next
            nxt = [e for e in r.events if e.t > foul.t and e.kind in (K.SHOT_ATTEMPT, K.POSSESSION, K.FOUL)]
            assert not nxt or nxt[0].kind is not K.SHOT_ATTEMPT or nxt[0].value != 1
    assert ("shooting", 2, False) in kinds and ("shooting", 1, True) in kinds
    assert ("personal", 0, False) in kinds and ("personal", 2, False) in kinds


def test_free_throw_cadence_is_not_machine_regular(foul_game: SimResult) -> None:
    """The official retrieves the ball between FTs and the shooter's routine varies: the FT-to-FT gap is
    not a constant (it was 3.6-4.0 s p5-p95, sd ~0.1 s); design ranges, unverified."""
    r = foul_game
    fts = [a for a in _kind(r, K.SHOT_ATTEMPT) if a.value == 1]
    gaps = [b.t - a.t for a, b in zip(fts, fts[1:], strict=False) if a.player_id == b.player_id and b.t - a.t < 12.0]
    assert len(gaps) >= 8
    assert float(np.std(gaps)) > 0.3, gaps


def test_fouled_three_point_shooters_get_three_free_throws() -> None:
    r = GameSimulator(SimConfig(seconds=300.0, fps=10.0, seed=8, three_foul_rate=1.0)).run()
    threes = [f for f in _kind(r, K.FOUL) if f.attrs["n_ft"] == 3 and f.t < r.frames[-1].t - 30.0]
    assert threes
    attempts = _kind(r, K.SHOT_ATTEMPT)
    for foul in threes:
        assert foul.attrs["type"] == "shooting"
        fts = [a for a in attempts if a.t > foul.t][:3]
        assert [a.value for a in fts] == [1, 1, 1] and all(a.player_id == foul.other_player_id for a in fts)
    jumpers3 = [a for a in attempts if a.value == 3 and a.attrs["shot_type"] == "jumper"]
    assert not jumpers3  # every three-point jumper was fouled before the release: no FGA


def test_penalty_follows_the_team_foul_count(foul_game: SimResult) -> None:
    r = foul_game
    two = r.rules.bonus["two_shots_from"]
    count: dict[tuple[int, Team], int] = {}
    for foul in _kind(r, K.FOUL):
        if foul.attrs["type"] == "offensive":
            continue  # not a team foul (NBA convention)
        sb = r.frames[foul.frame_idx].scoreboard
        key = (sb.period, foul.team)
        count[key] = count.get(key, 0) + 1
        if foul.attrs["type"] == "personal":
            late = sb.game_clock <= r.rules.bonus.get("late_period_s", -1)
            if not late:
                assert (foul.attrs["n_ft"] == 2) == (count[key] >= two), (foul, count[key])


# ---------------------------------------------------------------------------- the horn


def test_period_end_is_stamped_at_the_horn_even_with_a_shot_in_the_air() -> None:
    """PERIOD_END is the frame the game clock reaches 0; a shot released before the horn still plays out
    (rules note §2a) after it, and no rebound follows its miss."""
    pending = 0
    for seed in (1, 2, 3, 4):
        r = GameSimulator(SimConfig(seconds=300.0, fps=15.0, seed=seed, period_s=30.0)).run()
        for end in _kind(r, K.PERIOD_END):
            period = end.attrs["period"]
            first_zero = next(f for f in r.frames if f.scoreboard.period == period and f.scoreboard.game_clock <= 1e-9)
            assert end.frame_idx == first_zero.frame_idx, (seed, period)
            after = [e for e in r.events if e.frame_idx > end.frame_idx and e.kind is not K.DRIBBLE]
            before_next = [e for e in after if not (e.kind is K.PERIOD_START)]
            nxt_start = next((e for e in after if e.kind is K.PERIOD_START), None)
            if nxt_start is not None:
                before_next = [e for e in after if e.frame_idx < nxt_start.frame_idx]
            results = [e for e in before_next if e.kind in (K.SHOT_MADE, K.SHOT_MISSED)]
            pending += bool(results)
            assert not [e for e in before_next if e.kind is K.REBOUND]
            assert all(e.kind in (K.SHOT_MADE, K.SHOT_MISSED, K.BLOCK, K.ASSIST, K.SUBSTITUTION) for e in before_next)
    assert pending >= 1  # a buzzer shot was in the air at the horn


# ---------------------------------------------------------------------------- operator-run scoreboard


def test_operator_board_lags_and_is_read_by_5hz_ocr(game: SimResult) -> None:
    """D14 / §7A.5: the board is read at 5 Hz (source "ocr", so 2-read confirmation applies), the score
    changes after the make by the operator's lag (never on the make frame), the game clock shows whole
    seconds above a minute (tenths below), the shot clock whole seconds."""
    cfg = ScoreboardConfig()
    fps = game.config.fps
    boards = [f.scoreboard for f in game.frames]
    shown = ScoreboardModel(cfg).displayed(boards, fps, seed=1)
    reads = ScoreboardModel(cfg).readings(boards, fps, seed=1)
    every = int(round(fps / cfg.ocr_hz))
    assert all((r is None) == (i % every != 0) for i, r in enumerate(reads))
    assert all(r.source == "ocr" for r in reads if r is not None)
    for r in shown:
        if r.game_clock >= cfg.tenths_below_s:
            assert r.game_clock == int(r.game_clock)
        assert r.shot_clock is None or r.shot_clock == int(r.shot_clock)
    makes = _kind(game, K.SHOT_MADE)
    assert makes
    for m in makes:
        i = m.frame_idx
        truth = (boards[i].home_score, boards[i].away_score)
        lo = int(math.floor(cfg.score_lag_s[0] * fps))
        assert (shown[i].home_score, shown[i].away_score) != truth  # never on the make frame
        assert (shown[i + lo - 1].home_score, shown[i + lo - 1].away_score) != truth
        hi = min(len(shown) - 1, i + int(math.ceil(cfg.score_lag_s[1] * fps)) + 1)
        assert (shown[hi].home_score, shown[hi].away_score) == truth or any(
            boards[j].home_score + boards[j].away_score != boards[i].home_score + boards[i].away_score
            for j in range(i + 1, hi + 1))
    drift = [abs(s.game_clock - b.game_clock) for s, b in zip(shown, boards, strict=True)]
    assert max(drift) <= cfg.clock_fix_s + cfg.clock_lag_s[1] + 1.0  # the officials keep it right
    final = shown[-1]
    assert (final.home_score, final.away_score) == (boards[-1].home_score, boards[-1].away_score) or \
        game.frames[-1].t - max(m.t for m in makes) < cfg.score_lag_s[1]


def test_operator_clock_stops_late_after_a_whistle(game: SimResult) -> None:
    fps = game.config.fps
    cfg = ScoreboardConfig(clock_lag_s=(0.4, 0.4), clock_fix_s=10.0)
    boards = [f.scoreboard for f in game.frames]
    shown = ScoreboardModel(cfg).displayed(boards, fps, seed=1, tenths=True)
    checked = 0
    for foul in _kind(game, K.FOUL):
        i = foul.frame_idx
        if i + 30 >= len(boards) or boards[i].game_clock < 70.0:
            continue
        # the official clock is stopped at the whistle; the operator's keeps running ~0.4 s more
        assert boards[i + 6].game_clock == boards[i].game_clock
        assert shown[i + 6].game_clock < math.ceil(boards[i].game_clock - 1e-6) + 1e-9 or \
            shown[i + 6].game_clock <= shown[i - 15].game_clock
        checked += 1
    assert checked > 0


def test_operator_board_reaches_zero_at_the_horn() -> None:
    """The venue game clock and shot clock are the official timers: their signal sounds by itself at 0
    (rules note §3 [3][4][9][11]), so the operator's lag never applies to expiry. The displayed board and
    its 5 Hz OCR read 0.0 within one OCR read of every PERIOD_END and every shot-clock TURNOVER (the
    board used to freeze at 0.3-1.0 s for the whole break), and mono emits PERIOD_END on time from the
    board (it used to fall back to the next period's start, 6.5-8 s late)."""
    from courtvision.cli import simulate_config
    from courtvision.pipeline import make_source, run_offline

    cfg = ScoreboardConfig()
    fps = 30.0
    every = int(round(fps / cfg.ocr_hz))
    horns = violations = 0
    sims = [SimConfig(seconds=200.0, fps=fps, seed=seed, period_s=30.0) for seed in (1, 2, 5)]
    sims.append(SimConfig(seconds=240.0, fps=fps, seed=3, turnover_rate=0.6, turnover_mix={"shot_clock": 1.0}))
    for sc in sims:
        seed = sc.seed
        r = GameSimulator(sc).run()
        boards = [f.scoreboard for f in r.frames]
        shown = ScoreboardModel(cfg).displayed(boards, fps, seed=seed)
        reads = ScoreboardModel(ScoreboardConfig(misread_rate=0.0)).readings(boards, fps, seed=seed)
        for e in r.events:
            if e.kind is K.PERIOD_END:
                clock = [None if b is None else b.game_clock for b in shown]
                ocr = [None if b is None else b.game_clock for b in reads]
                horns += 1
            elif e.kind is K.TURNOVER and e.attrs["type"] == "shot_clock":
                clock = [None if b is None else b.shot_clock for b in shown]
                ocr = [None if b is None else b.shot_clock for b in reads]
                violations += 1
            else:
                continue
            i = e.frame_idx
            assert clock[i] == 0.0, (seed, e)
            first = next(j for j in range(i, i + every) if reads[j] is not None)
            assert ocr[first] == 0.0, (seed, e)
            if e.kind is K.PERIOD_END:  # it stays at 0.0 through the break
                nxt = next((x.frame_idx for x in r.events if x.kind is K.PERIOD_START and x.frame_idx > i),
                           len(shown))
                assert all(v == 0.0 for v in clock[i:nxt]), (seed, e)
    assert horns >= 9 and violations >= 3, (horns, violations)
    run_cfg = simulate_config("mono", seed=2, seconds=200.0, overrides=["sim.period_s=30", "narration.tts=none"])
    res = run_offline(run_cfg, make_source(run_cfg, realtime=False), write=False, env={})
    truth = [e for e in res.truth_events if e.kind is K.PERIOD_END]
    pred = [e for e in res.events if e.kind is K.PERIOD_END]
    assert len(truth) >= 3
    for e in truth:
        assert any(abs(p.t - e.t) <= run_cfg.eval.tol_s for p in pred), (e.t, [p.t for p in pred])
    assert res.event_report.per_kind["period_end"].f1 == 1.0


def test_sim_sources_pick_the_board_per_mode() -> None:
    """source.scoreboard "auto": truth3d keeps the exact board (source "sim", every frame); mono gets the
    operator-run board read by 5 Hz OCR (source "ocr", every 6th frame at 30 fps)."""
    import asyncio

    from courtvision.pipeline import SimSource

    r = GameSimulator(SimConfig(seconds=12.0, fps=30.0, seed=4)).run()

    async def boards(src: SimSource) -> list:
        return [item.state.scoreboard async for item in src]

    truth = asyncio.run(boards(SimSource(r, "truth3d")))
    assert all(b is not None and b.source == "sim" for b in truth)
    mono = asyncio.run(boards(SimSource(r, "mono")))
    assert all((b is None) == (i % 6 != 0) for i, b in enumerate(mono))
    assert all(b.source == "ocr" for b in mono if b is not None)
    exact = asyncio.run(boards(SimSource(r, "mono", scoreboard="truth")))
    assert all(b is not None and b.source == "sim" for b in exact)
