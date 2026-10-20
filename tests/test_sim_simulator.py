"""Invariants of the simulator's truth (ARCHITECTURE "Shared event semantics")."""

from __future__ import annotations

import collections
import math

import numpy as np
import pytest

from courtvision.geometry.camera import sees_full_court
from courtvision.sim.simulator import BOX_STATS, GameSimulator, SimConfig, SimResult, cylinder_boxes, sphere_boxes
from courtvision.types import EventKind as K
from courtvision.types import EventStatus, Team

BALL_R = 0.39


@pytest.fixture(scope="module")
def sim() -> SimResult:
    # Short periods so one run covers several periods, halftime and the late-clock rules.
    return GameSimulator(SimConfig(seconds=420.0, fps=30.0, seed=3, period_s=80.0)).run()


def _kind(r: SimResult, kind: K) -> list:
    return [e for e in r.events if e.kind is kind]


def _crossing(r: SimResult, i: int) -> tuple[float, float] | None:
    """Interpolated ball (x, y) where the centre crosses the rim plane downward between frames i-1, i."""
    f0, f1 = r.frames[i - 1].ball, r.frames[i].ball
    h = r.court.rim_height
    if not (f0.z > h >= f1.z):
        return None
    s = (f0.z - h) / (f0.z - f1.z)
    (x0, y0), (x1, y1) = f0.court_xy, f1.court_xy
    return (x0 + s * (x1 - x0), y0 + s * (y1 - y0))


# ---------------------------------------------------------------------------- structure


def test_frames_and_events_are_well_formed(sim: SimResult) -> None:
    cfg = sim.config
    assert len(sim.frames) == int(cfg.seconds * cfg.fps)
    assert len(sim.referees) == len(sim.frames) and all(len(r) == 3 for r in sim.referees)
    assert len(sim.holders) == len(sim.frames)
    ids = [e.event_id for e in sim.events]
    assert len(set(ids)) == len(ids)
    assert all(e.status is EventStatus.CONFIRMED and e.source == "sim" for e in sim.events)
    assert [e.frame_idx for e in sim.events] == sorted(e.frame_idx for e in sim.events)
    for e in sim.events:
        assert math.isclose(e.t, e.frame_idx / cfg.fps)
    for f in sim.frames:
        assert len(f.players) == 10
        assert len({p.player_id for p in f.players}) == 10
        assert collections.Counter(p.team for p in f.players) == {Team.HOME: 5, Team.AWAY: 5}
        assert f.scoreboard is not None and f.scoreboard.source == "sim"
        assert len(f.rims) == 2 and f.ball is not None


def test_roster_and_box_cover_every_player(sim: SimResult) -> None:
    ids = {e.player_id for e in sim.roster.entries}
    assert set(sim.truth_box) == ids
    assert all(set(line) == set(BOX_STATS) for line in sim.truth_box.values())
    for team in (Team.HOME, Team.AWAY):
        assert 10 <= sum(e.team is team for e in sim.roster.entries) <= 12


def test_all_event_kinds_appear(sim: SimResult) -> None:
    kinds = {e.kind for e in sim.events}
    for k in (K.PERIOD_START, K.PERIOD_END, K.POSSESSION, K.PASS, K.DRIBBLE, K.SHOT_ATTEMPT, K.SHOT_MADE,
              K.SHOT_MISSED, K.REBOUND, K.ASSIST, K.SUBSTITUTION):
        assert k in kinds, k


# ---------------------------------------------------------------------------- scoring


def test_score_equals_made_values_and_scoreboard(sim: SimResult) -> None:
    made = collections.Counter()
    for e in _kind(sim, K.SHOT_MADE):
        made[e.team] += e.value
    for team in (Team.HOME, Team.AWAY):
        assert made[team] == sim.truth_score[team]
        assert sum(sim.truth_box[p]["pts"] for p in sim.truth_box if sim.roster.team_of(p) is team) == made[team]
    last = sim.frames[-1].scoreboard
    assert (last.home_score, last.away_score) == (sim.truth_score[Team.HOME], sim.truth_score[Team.AWAY])
    scores = np.array([[f.scoreboard.home_score, f.scoreboard.away_score] for f in sim.frames])
    assert np.all(np.diff(scores, axis=0) >= 0)


def test_truth_box_equals_event_tallies(sim: SimResult) -> None:
    """truth_box is tallied by the simulation logic; it must agree with the truth events."""
    box = {p: dict.fromkeys(BOX_STATS, 0) for p in sim.truth_box}
    attempts = {a.shot_id: a for a in _kind(sim, K.SHOT_ATTEMPT)}
    simple = {K.ASSIST: "ast", K.STEAL: "stl", K.BLOCK: "blk", K.TURNOVER: "tov", K.FOUL: "pf"}
    for e in sim.events:
        if e.kind is K.SHOT_ATTEMPT:
            box[e.player_id]["fta" if e.value == 1 else "fga"] += 1
            box[e.player_id]["fg3a"] += e.value == 3
        elif e.kind is K.SHOT_MADE:
            value = attempts[e.shot_id].value
            box[e.player_id]["pts"] += e.value
            box[e.player_id]["ftm" if value == 1 else "fgm"] += 1
            box[e.player_id]["fg3m"] += value == 3
        elif e.kind is K.REBOUND and e.player_id is not None:
            box[e.player_id]["oreb" if e.attrs["offensive"] else "dreb"] += 1
        elif e.kind in simple and e.player_id is not None:
            box[e.player_id][simple[e.kind]] += 1
    assert box == sim.truth_box
    assert sum(line["fga"] for line in box.values()) > 20 and sum(line["oreb"] + line["dreb"] for line in box.values())


def test_ball_trajectory_is_continuous(sim: SimResult) -> None:
    """No teleports: catches, rebounds, steals and FT releases carry the ball (regression)."""
    ball = np.array([[*f.ball.court_xy, f.ball.z] for f in sim.frames])
    speed = np.linalg.norm(np.diff(ball, axis=0), axis=1) * sim.config.fps
    assert speed.max() < 60.0, (int(speed.argmax()) + 1, float(speed.max()))
    for a in _kind(sim, K.SHOT_ATTEMPT):  # the release frame continues the gather (FTs jumped ~4 ft)
        assert np.linalg.norm(ball[a.frame_idx] - ball[a.frame_idx - 1]) < 1.0, a


def test_frames_do_not_share_mutable_lists(sim: SimResult) -> None:
    assert sim.frames[0].rims == sim.frames[1].rims and sim.frames[0].rims is not sim.frames[1].rims


def test_each_attempt_has_exactly_one_result(sim: SimResult) -> None:
    results = collections.Counter(e.shot_id for e in sim.events if e.kind in (K.SHOT_MADE, K.SHOT_MISSED))
    end_t = sim.frames[-1].t
    for a in _kind(sim, K.SHOT_ATTEMPT):
        if a.t < end_t - 3.0:
            assert results[a.shot_id] == 1, a
    assert all(n == 1 for n in results.values())
    attempts = {a.shot_id: a for a in _kind(sim, K.SHOT_ATTEMPT)}
    for e in sim.events:
        if e.kind in (K.SHOT_MADE, K.SHOT_MISSED, K.BLOCK, K.ASSIST):
            att = attempts[e.shot_id]
            assert att.t <= e.t
            if e.kind is not K.BLOCK:
                assert e.team is att.team
            if e.kind in (K.SHOT_MADE, K.SHOT_MISSED):
                assert e.player_id == att.player_id
            if e.kind is K.SHOT_MADE:
                assert e.value == att.value


def test_point_value_from_gather_position(sim: SimResult) -> None:
    for a in _kind(sim, K.SHOT_ATTEMPT):
        if a.attrs["shot_type"] == "free_throw":
            assert a.value == 1
            continue
        three = sim.court.is_three(a.attrs["gather_x"], a.attrs["gather_y"], a.attrs["basket"])
        assert a.value == (3 if three else 2)
        assert a.attrs["zone"] == sim.court.zone(a.attrs["gather_x"], a.attrs["gather_y"], a.attrs["basket"])
        for key in ("shot_type", "distance_ft", "closest_defender_ft", "catch_and_shoot", "dribbles_before"):
            assert key in a.attrs


def test_made_balls_pass_through_rim_cylinder(sim: SimResult) -> None:
    lim = sim.court.rim_radius - BALL_R
    made = _kind(sim, K.SHOT_MADE)
    assert made
    for m in made:
        xy = _crossing(sim, m.frame_idx)
        assert xy is not None, m
        bx, by = sim.court.basket_xy(m.attrs["basket"])
        assert math.hypot(xy[0] - bx, xy[1] - by) <= lim + 1e-6
        later = sim.frames[min(m.frame_idx + 5, len(sim.frames) - 1)].ball  # drops through the net
        assert later.z < sim.court.rim_height and math.hypot(later.court_xy[0] - bx, later.court_xy[1] - by) < 1.5


def test_no_crossing_through_the_hoop_without_a_make(sim: SimResult) -> None:
    made_frames = {e.frame_idx for e in _kind(sim, K.SHOT_MADE)}
    for i in range(1, len(sim.frames)):
        xy = _crossing(sim, i)
        if xy is None:
            continue
        for side in ("left", "right"):
            bx, by = sim.court.basket_xy(side)
            if math.hypot(xy[0] - bx, xy[1] - by) <= sim.court.rim_radius:
                assert i in made_frames


def test_misses_touch_the_rim_or_are_blocked(sim: SimResult) -> None:
    blocked = {e.shot_id for e in _kind(sim, K.BLOCK)}
    attempts = {a.shot_id: a for a in _kind(sim, K.SHOT_ATTEMPT)}
    for m in _kind(sim, K.SHOT_MISSED):
        a = attempts[m.shot_id]
        if m.shot_id in blocked:
            assert m.t - a.t <= 0.25
            continue
        bx, by = sim.court.basket_xy(a.attrs["basket"])
        dmin = min(math.dist((f.ball.court_xy[0], f.ball.court_xy[1], f.ball.z), (bx, by, sim.court.rim_height))
                   for f in sim.frames[a.frame_idx:m.frame_idx + 1])
        assert dmin < sim.court.rim_radius + 0.6


def test_blocks_reverse_the_ball_near_the_defender(sim: SimResult) -> None:
    attempts = {a.shot_id: a for a in _kind(sim, K.SHOT_ATTEMPT)}
    for b in _kind(sim, K.BLOCK):
        a = attempts[b.shot_id]
        assert b.other_player_id == a.player_id and b.team is a.team.other()
        f = sim.frames[b.frame_idx]
        blocker = f.player(b.player_id)
        assert blocker is not None and math.hypot(blocker.x - f.ball.court_xy[0], blocker.y - f.ball.court_xy[1]) < 8.0


# ---------------------------------------------------------------------------- rebounds, assists, possessions


def test_rebounds_only_after_misses(sim: SimResult) -> None:
    pending = None
    for e in sim.events:
        if e.kind is K.SHOT_MISSED:
            pending = e
        elif e.kind is K.SHOT_MADE:
            pending = None
        elif e.kind is K.REBOUND:
            assert pending is not None, e
            if e.player_id is not None:
                assert sim.holders[e.frame_idx] == e.player_id
                assert e.attrs["offensive"] == (e.team is pending.team)
            pending = None


def test_assists_follow_the_rule(sim: SimResult) -> None:
    attempts = {a.shot_id: a for a in _kind(sim, K.SHOT_ATTEMPT)}
    passes = [p for p in _kind(sim, K.PASS) if p.attrs["completed"]]
    dribbles = _kind(sim, K.DRIBBLE)
    assists = _kind(sim, K.ASSIST)
    assert assists
    for a in assists:
        shot = attempts[a.shot_id]
        assert shot.value in (2, 3)
        made = [m for m in _kind(sim, K.SHOT_MADE) if m.shot_id == a.shot_id]
        assert made and made[0].frame_idx == a.frame_idx
        last = [p for p in passes if p.other_player_id == shot.player_id and p.attrs["catch_t"] <= shot.t][-1]
        assert last.player_id == a.player_id and a.other_player_id == shot.player_id
        assert sim.roster.team_of(a.player_id) is shot.team
        assert shot.t - last.attrs["catch_t"] <= 5.0 + 1e-9
        n = sum(d.player_id == shot.player_id and last.attrs["catch_t"] <= d.t <= shot.t for d in dribbles)
        assert n <= 2


def test_possessions_alternate_within_a_period(sim: SimResult) -> None:
    last = None
    for e in sim.events:
        if e.kind is K.PERIOD_START:
            last = None
        elif e.kind is K.POSSESSION:
            assert e.team is not last, e
            assert sim.holders[e.frame_idx] == e.player_id
            assert e.attrs["cause"]
            last = e.team


def test_steals_and_turnovers(sim: SimResult) -> None:
    turnovers = _kind(sim, K.TURNOVER)
    for st in _kind(sim, K.STEAL):
        tos = [t for t in turnovers if t.frame_idx == st.frame_idx]
        assert tos and tos[0].player_id == st.other_player_id and tos[0].attrs["type"] in ("steal", "bad_pass")
        assert sim.holders[st.frame_idx] == st.player_id
    for t in turnovers:
        assert t.attrs["type"] in ("steal", "bad_pass", "out_of_bounds", "shot_clock", "travel", "offensive_foul",
                                   "other")


def test_completed_passes_reach_the_receiver(sim: SimResult) -> None:
    fps = sim.config.fps
    for p in _kind(sim, K.PASS):
        # release frame: the ball is still at the passer's hand (it was held on the previous frame)
        assert sim.holders[p.frame_idx - 1] == p.player_id
        f = sim.frames[p.frame_idx]
        passer = f.player(p.player_id)
        assert math.hypot(f.ball.court_xy[0] - passer.x, f.ball.court_xy[1] - passer.y) < 2.0
        if p.attrs["completed"]:
            assert sim.holders[int(round(p.attrs["catch_t"] * fps))] == p.other_player_id


def test_dribble_events_at_floor_contact(sim: SimResult) -> None:
    for d in _kind(sim, K.DRIBBLE)[:400]:
        f = sim.frames[d.frame_idx]
        assert f.ball.z < 1.5
        # the dribbler holds the ball (a strip pops it loose at a floor contact: held one frame earlier)
        assert d.player_id in (sim.holders[d.frame_idx], sim.holders[d.frame_idx - 1])
        assert math.hypot(f.ball.court_xy[0] - d.x, f.ball.court_xy[1] - d.y) < 1e-6


# ---------------------------------------------------------------------------- physics & projection


def test_speeds_and_ball_height(sim: SimResult) -> None:
    speeds = [p.speed for f in sim.frames for p in f.players]
    assert max(speeds) <= 30.0
    assert min(f.ball.z for f in sim.frames) >= BALL_R - 1e-9


def test_projections_inside_the_image(sim: SimResult) -> None:
    assert sees_full_court(sim.camera, sim.court, margin_px=1.0)
    w, h = sim.config.image_size
    for f in sim.frames[::3]:
        for p in f.players:
            b = p.bbox
            assert 0 <= b.x1 < b.x2 <= w and 0 <= b.y1 < b.y2 <= h, (f.frame_idx, p)
        if f.ball.visible:
            u, v = f.ball.image_xy
            assert 0 <= u <= w and 0 <= v <= h and f.ball.radius_px > 0
        for rim in f.rims:
            assert 0 <= rim.bbox.x1 < rim.bbox.x2 <= w and 0 <= rim.bbox.y1 < rim.bbox.y2 <= h


def test_vectorised_boxes_match_camera(sim: SimResult) -> None:
    cam = sim.camera
    rng = np.random.default_rng(0)
    xy = rng.uniform([-3, -3], [97, 53], size=(20, 2))
    boxes = cylinder_boxes(cam, xy, 6.5, 1.0)
    for (x, y), b in zip(xy, boxes, strict=True):
        ref = cam.bbox_of_cylinder(float(x), float(y), 6.5, 1.0)
        assert np.allclose(b, ref.as_tuple(), atol=1e-6)
    xyz = np.column_stack([xy, rng.uniform(0.4, 15, 20)])
    for p, b in zip(xyz, sphere_boxes(cam, xyz, BALL_R), strict=True):
        assert np.allclose(b, cam.bbox_of_sphere(*map(float, p), BALL_R).as_tuple(), atol=1e-6)
    f = sim.frames[100]
    for p in f.players:
        assert np.allclose(p.bbox.as_tuple(), cam.bbox_of_cylinder(p.x, p.y, 6.5, 1.0).as_tuple(), atol=1e-6)


# ---------------------------------------------------------------------------- clocks & periods


def test_periods_and_halftime_side_switch(sim: SimResult) -> None:
    starts = [e.attrs["period"] for e in _kind(sim, K.PERIOD_START)]
    ends = [e.attrs["period"] for e in _kind(sim, K.PERIOD_END)]
    assert starts == list(range(1, len(starts) + 1)) and len(starts) >= 3
    assert ends == starts[: len(ends)]
    half = sim.rules.n_periods // 2
    sides: dict[tuple[Team, bool], set[str]] = collections.defaultdict(set)
    period_at = [f.scoreboard.period for f in sim.frames]
    for a in _kind(sim, K.SHOT_ATTEMPT):
        sides[(a.team, period_at[a.frame_idx] > half)].add(a.attrs["basket"])
    for team in (Team.HOME, Team.AWAY):
        first, second = sides[(team, False)], sides[(team, True)]
        assert len(first) == 1 and len(second) == 1 and first != second


def test_game_clock_and_shot_clock(sim: SimResult) -> None:
    rules = sim.rules
    for f0, f1 in zip(sim.frames, sim.frames[1:], strict=False):
        a, b = f0.scoreboard, f1.scoreboard
        if a.period == b.period:
            assert b.game_clock <= a.game_clock + 1e-9
        assert b.shot_clock is None or b.shot_clock <= rules.shot_clock_s + 1e-9
        if b.shot_clock is not None:
            assert b.shot_clock <= b.game_clock + rules.shot_clock_s
    for e in _kind(sim, K.POSSESSION):
        sc = sim.frames[e.frame_idx].scoreboard
        assert sc.shot_clock is None or math.isclose(sc.shot_clock, rules.shot_clock_s)
    blocked = {e.shot_id for e in _kind(sim, K.BLOCK)}
    misses = {e.frame_idx: e for e in _kind(sim, K.SHOT_MISSED)}
    for e in _kind(sim, K.REBOUND):
        prior = [m for i, m in misses.items() if i <= e.frame_idx]
        last_miss = max(prior, key=lambda m: m.frame_idx)
        if e.attrs["offensive"] and e.player_id and last_miss.shot_id not in blocked:
            sc = sim.frames[e.frame_idx].scoreboard.shot_clock
            assert sc is None or math.isclose(sc, rules.shot_clock_reset_s)


def test_clock_stops_after_late_made_field_goals(sim: SimResult) -> None:
    rules = sim.rules
    checked = 0
    for m in _kind(sim, K.SHOT_MADE):
        sb = sim.frames[m.frame_idx].scoreboard
        if m.value == 1 or not rules.clock_stops_after_fg_at(sb.period, sb.game_clock):
            continue
        nxt = [p for p in _kind(sim, K.PASS) if p.frame_idx > m.frame_idx]
        if not nxt:
            continue
        stop = sim.frames[nxt[0].frame_idx].scoreboard.game_clock
        assert math.isclose(stop, sb.game_clock, abs_tol=1e-6)
        checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------- free throws, subs, fouls


@pytest.fixture(scope="module")
def ft_sim() -> SimResult:
    """Foul-heavy game: many free-throw trips."""
    return GameSimulator(SimConfig(seconds=150.0, fps=30.0, seed=8, foul_rate=0.6, turnover_rate=0.0)).run()


def test_free_throw_formation(ft_sim: SimResult) -> None:
    sim = ft_sim
    fts = [a for a in _kind(sim, K.SHOT_ATTEMPT) if a.value == 1]
    assert len(fts) >= 6
    fps, court = sim.config.fps, sim.court
    lane = court.lane_width / 2.0
    for a in fts:
        n = int(math.ceil(1.5 * fps))
        for f in sim.frames[a.frame_idx - n:a.frame_idx + 1]:
            assert f.player(a.player_id).speed < 0.3
        f = sim.frames[a.frame_idx]
        bx, by = court.basket_xy(a.attrs["basket"])
        shooter = f.player(a.player_id)
        ft_dist = court.free_throw_line_from_baseline - court.basket_offset
        assert abs(math.hypot(shooter.x - bx, shooter.y - by) - ft_dist) < 1.2
        on_lane = [p for p in f.players if abs(abs(p.y - by) - lane) < 1.5 and abs(p.x - bx) < 10.0]
        assert len(on_lane) >= 4
    fouls = [f for f in _kind(sim, K.FOUL) if f.t < sim.frames[-1].t - 25.0]
    assert fouls
    for foul in fouls:  # fouls happen before a shot: no FGA, then 2 FTs by the fouled player
        later = [a for a in _kind(sim, K.SHOT_ATTEMPT) if a.t > foul.t][:2]
        assert [a.value for a in later] == [1, 1] and all(a.player_id == foul.other_player_id for a in later)


def test_substitutions_match_frames(sim: SimResult) -> None:
    subs = _kind(sim, K.SUBSTITUTION)
    assert len(subs) == len(sim.substitutions) > 0
    for ev, s in zip(subs, sim.substitutions, strict=True):
        assert (ev.player_id, ev.other_player_id, ev.frame_idx) == (s["in"], s["out"], s["frame_idx"])
        f = sim.frames[ev.frame_idx]
        on = {p.player_id for p in f.players}
        assert ev.player_id in on and ev.other_player_id not in on
        assert ev.other_player_id in {p.player_id for p in sim.frames[ev.frame_idx - 1].players}
    on_court = {p.player_id for p in sim.frames[0].players}
    assert on_court == set(sim.starters[Team.HOME]) | set(sim.starters[Team.AWAY])


def test_out_of_bounds_inbound_from_outside(sim: SimResult) -> None:
    court = sim.court
    for t in [e for e in _kind(sim, K.TURNOVER) if e.attrs["type"] in ("out_of_bounds", "travel", "shot_clock")]:
        nxt = [p for p in _kind(sim, K.POSSESSION) if p.frame_idx >= t.frame_idx]
        if not nxt:
            continue
        p = nxt[0]
        assert p.team is t.team.other()
        holder = sim.frames[p.frame_idx].player(p.player_id)
        assert not court.in_bounds(holder.x, holder.y)  # the thrower-in stands outside the line


# ---------------------------------------------------------------------------- configs & determinism


def test_determinism_by_seed() -> None:
    cfg = SimConfig(seconds=40.0, fps=15.0, seed=11)
    a, b = GameSimulator(cfg).run(), GameSimulator(cfg).run()
    assert [e.to_dict() for e in a.events] == [e.to_dict() for e in b.events]
    pa = np.array([[p.x, p.y] for f in a.frames for p in f.players])
    pb = np.array([[p.x, p.y] for f in b.frames for p in f.players])
    assert np.array_equal(pa, pb)
    c = GameSimulator(SimConfig(seconds=40.0, fps=15.0, seed=12)).run()
    assert [e.to_dict() for e in c.events] != [e.to_dict() for e in a.events]


@pytest.mark.parametrize("league", ["fiba", "ncaa_m", "ncaa_w", "nfhs"])
def test_other_leagues_run(league: str) -> None:
    r = GameSimulator(SimConfig(league=league, seconds=60.0, fps=15.0, seed=2, period_s=40.0)).run()
    assert sees_full_court(r.camera, r.court)
    made = sum(e.value for e in r.events if e.kind is K.SHOT_MADE)
    assert made == sum(r.truth_score.values())
    w, h = r.config.image_size
    assert all(0 <= p.bbox.x1 and p.bbox.x2 <= w and p.bbox.y2 <= h for f in r.frames[::5] for p in f.players)
    sc = [f.scoreboard.shot_clock for f in r.frames if f.scoreboard.shot_clock is not None]
    assert max(sc) <= r.rules.shot_clock_s


def test_no_shot_clock_override() -> None:
    r = GameSimulator(SimConfig(league="nfhs", seconds=40.0, fps=10.0, seed=4, rule_overrides=["no_shot_clock"])).run()
    assert r.rules.shot_clock_s is None
    assert all(f.scoreboard.shot_clock is None for f in r.frames)
    assert not any(e.kind is K.TURNOVER and e.attrs["type"] == "shot_clock" for e in r.events)


def test_timeouts_take_the_configured_video_time() -> None:
    r = GameSimulator(SimConfig(seconds=120.0, fps=10.0, seed=5, timeout_prob=1.0, timeout_video_s=4.0)).run()
    touts = [e for e in r.events if e.kind is K.TIMEOUT]
    assert touts
    for e in touts:
        sb0 = r.frames[e.frame_idx].scoreboard
        later = r.frames[min(e.frame_idx + int(4.0 * r.config.fps) - 1, len(r.frames) - 1)].scoreboard
        assert later.game_clock == sb0.game_clock  # clock stopped through the timeout
    per_team = collections.Counter(e.team for e in touts)
    assert all(n <= r.rules.timeouts["count"] for n in per_team.values())
    fiba = GameSimulator(SimConfig(league="fiba", seconds=200.0, fps=10.0, seed=5, timeout_prob=1.0)).run()
    per_team = collections.Counter(e.team for e in fiba.events if e.kind is K.TIMEOUT)
    assert per_team and max(per_team.values()) <= 5  # FIBA: 2 + 3 per game (no "count" key in the YAML)


def test_invalid_config() -> None:
    with pytest.raises(ValueError):
        GameSimulator(SimConfig(fps=0)).run()
    with pytest.raises(ValueError):
        GameSimulator(SimConfig(league="nope")).run()
