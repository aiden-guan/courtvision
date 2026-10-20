"""Director: priority ordering, TTLs, cooldowns, preemption, invalidation, hedge cap, corrections."""

from __future__ import annotations

import itertools
from dataclasses import replace

import pytest
from test_narration_core_support import ev, shot_made

from courtvision.narration.director import (Cue, Director, DirectorConfig, fact_changed, is_material_change,
                                            priority_for)
from courtvision.types import EventKind, EventStatus, Team

_n = itertools.count(1)


def cue(priority: int, text: str | None = None, *, t: float = 0.0, channel: str = "play_by_play",
        kind: str = "", hedge: bool = False, events=None, event_ids=None, ttl: float = 0.0,
        version: int = 0, group: str | None = None) -> Cue:
    i = next(_n)
    evs = list(events or [])
    ids = list(event_ids) if event_ids is not None else [e.event_id for e in evs]
    return Cue(f"u{i:04d}", priority, text or f"line {i}.", t, ttl, ids, channel, hedge, version, kind=kind,
               group=group, events=evs)


def speak(d: Director, c: Cue, start: float, dur: float = 1.0) -> None:
    got = d.next(start)
    assert got is c, (got, c)
    d.on_spoken(c, start + dur)


# --------------------------------------------------------------------------- ordering and TTL


def test_priority_ordering_and_freshest_first() -> None:
    d = Director()
    p3 = cue(3, channel="color", ttl=20.0)
    p2_old, p2_new = cue(2, t=1.0), cue(2, t=2.0)
    p0 = cue(0, channel="correction", kind="correction")
    for c in (p3, p2_old, p2_new, p0):
        d.offer(c, 0.0)
    assert [c.cue_id for c in d.pending()] == [p0.cue_id, p2_new.cue_id, p2_old.cue_id, p3.cue_id]
    order = []
    now = 0.0
    while (c := d.next(now)) is not None or d.pending(now):
        if c is not None:
            order.append(c.cue_id)
            d.on_spoken(c, now + 0.4)  # short lines so both P2s finish inside their 3 s TTL
        now += 0.1
    assert order == [p0.cue_id, p2_new.cue_id, p2_old.cue_id, p3.cue_id]


def test_newer_p1_replaces_queued_p1_and_flushes_p2_p3() -> None:
    d = Director()
    d.offer(cue(2), 0.0)
    d.offer(cue(3, channel="color", ttl=10.0), 0.0)
    miss = cue(1, "miss.", group="poss7")
    d.offer(miss, 0.0)
    assert [c.priority for c in d.pending()] == [1]
    reb = cue(1, "rebound.", group="poss7", t=0.5)
    d.offer(reb, 0.5)
    assert d.pending() == [reb]  # latest wins within the possession
    other = cue(1, "other possession.", group="poss8", t=0.6)
    d.offer(other, 0.6)
    assert {c.cue_id for c in d.pending()} == {other.cue_id}  # P1 depth is 1


@pytest.mark.parametrize("make,ttl", [
    (lambda: cue(1, kind="release"), 0.8),
    (lambda: cue(1, kind="made3"), 2.0),
    (lambda: cue(2, kind="possession"), 3.0),
    (lambda: cue(0, kind="made3"), 3.0),
    (lambda: cue(0, kind="correction", channel="correction"), 10.0),
    (lambda: cue(3, channel="color", ttl=7.5), 7.5),
    (lambda: cue(3, channel="color"), 20.0),
])
def test_ttl_expiry(make, ttl: float) -> None:
    d = Director()
    c = make()
    assert d.effective_ttl(c) == pytest.approx(ttl)
    d.offer(c, 100.0)
    assert d.pending(100.0 + ttl - 0.01) == [c]
    assert d.next(100.0 + ttl + 0.01) is None
    assert d.pending() == []


def test_p0_p2_ignore_cue_ttl_but_p3_uses_it() -> None:
    d = Director()
    assert d.effective_ttl(cue(1, ttl=60.0)) == pytest.approx(2.0)
    assert d.effective_ttl(cue(3, ttl=4.0)) == pytest.approx(4.0)


def test_buffer_depth_per_priority() -> None:
    d = Director()
    p2 = [cue(2, t=float(i)) for i in range(5)]
    for i, c in enumerate(p2):
        d.offer(c, float(i) * 0.1)
    assert [c.cue_id for c in d.pending()] == [c.cue_id for c in reversed(p2[2:])]  # P2 <= 3, oldest dropped


def test_duplicate_text_replaces_queued_copy() -> None:
    d = Director()
    a, b = cue(2, "Riverside ball."), cue(2, "Riverside ball.", t=1.0)
    d.offer(a, 0.0)
    d.offer(b, 0.5)
    assert d.pending() == [b]


# --------------------------------------------------------------------------- cooldowns and gaps


def test_min_gap_after_a_line() -> None:
    d = Director()
    first = cue(1)
    d.offer(first, 0.0)
    speak(d, first, 0.0, dur=1.0)  # ends at 1.0
    second = cue(1, t=1.1)
    d.offer(second, 1.1)
    assert d.next(1.3) is None  # 0.6 s gap
    assert d.next(1.65) is second


def test_p0_skips_gap_and_cooldowns() -> None:
    d = Director()
    first = cue(1)
    d.offer(first, 0.0)
    speak(d, first, 0.0)
    p0 = cue(0, channel="correction", kind="correction")
    d.offer(p0, 1.05)
    assert d.next(1.1) is p0


def test_per_channel_cooldown() -> None:
    d = Director(DirectorConfig(min_gap_s=0.0))
    c1 = cue(3, channel="color", ttl=20.0)
    d.offer(c1, 0.0)
    speak(d, c1, 0.0)  # ends 1.0
    c2 = cue(3, channel="color", ttl=20.0)
    pbp = cue(2, channel="play_by_play")
    d.offer(c2, 1.1)
    assert d.next(1.5) is None  # colour cooldown 1.0 s
    d.offer(pbp, 1.5)
    assert d.next(1.6) is pbp  # another channel is free
    d.on_spoken(pbp, 2.0)
    assert d.next(2.1) is c2


def test_kind_cooldown_for_score_updates() -> None:
    d = Director()
    s1 = cue(2, "Riverside leads 42-40.", kind="score_update")
    d.offer(s1, 0.0)
    speak(d, s1, 0.0)
    s2 = cue(2, "Riverside leads 44-40.", kind="score_update", t=10.0)
    d.offer(s2, 10.0)
    assert d.next(11.0) is None  # <= 1 score update per 60 s
    s3 = cue(2, "Riverside leads 46-40.", kind="score_update", t=70.0)
    d.offer(s3, 70.0)
    assert d.next(70.5) is s3


# --------------------------------------------------------------------------- preemption


def test_preemption_rules() -> None:
    d = Director()
    p2 = cue(2)
    d.offer(p2, 0.0)
    assert d.next(0.0) is p2 and d.current is p2
    assert not d.preempt_requested(0.1)
    p1 = cue(1, t=0.2)
    d.offer(p1, 0.2)
    assert d.preempt_requested(0.3)  # P1 cuts P2 at the next word boundary
    assert d.next(0.35) is p1  # the preemptor takes over
    assert d.current is p1
    p1b = cue(1, "another live call.", t=0.4, group="x")
    d.offer(p1b, 0.4)
    assert not d.preempt_requested(0.45)  # P1 never cuts P1
    assert d.next(0.5) is None
    p0 = cue(0, channel="correction", kind="correction")
    d.offer(p0, 0.6)
    assert d.preempt_requested(0.6)
    d.on_interrupted(p1, 0.7)
    assert d.next(0.7) is p0


def test_p2_preempts_p3_and_nothing_preempts_p0() -> None:
    d = Director()
    p3 = cue(3, channel="color", ttl=20.0)
    d.offer(p3, 0.0)
    assert d.next(0.0) is p3
    d.offer(cue(2), 0.1)
    assert d.preempt_requested(0.1)
    d2 = Director()
    p0 = cue(0, kind="correction", channel="correction")
    d2.offer(p0, 0.0)
    assert d2.next(0.0) is p0
    d2.offer(cue(1), 0.1)
    assert not d2.preempt_requested(0.2)


# --------------------------------------------------------------------------- invalidation


def test_invalidate_by_event_ids_and_state_version() -> None:
    d = Director()
    made = shot_made(3)
    a = cue(2, events=[made], version=5)
    b = cue(2, "other line.", event_ids=["e999"], version=5, t=1.0)
    c = cue(3, channel="color", ttl=20.0, version=3)
    for x in (a, b, c):
        d.offer(x, 0.0)
    assert d.invalidate(event_ids=[made.event_id]) == 1
    assert a not in d.pending()
    assert d.invalidate(state_version=5, min_priority=3) == 1  # only the stale colour line
    assert d.pending() == [b]
    assert d.invalidate(state_version=6) == 1
    assert d.pending() == []


def test_invalidating_the_current_line_requests_preemption() -> None:
    d = Director()
    made = shot_made(3)
    c = cue(1, events=[made])
    d.offer(c, 0.0)
    assert d.next(0.0) is c
    d.invalidate(event_ids=[made.event_id])
    assert d.preempt_requested(0.1)


# --------------------------------------------------------------------------- hedge cap


def test_at_most_one_hedge_in_three_utterances() -> None:
    d = Director(DirectorConfig(min_gap_s=0.0, cooldown_s={}))
    h1 = cue(1, "hedge one.", hedge=True)
    d.offer(h1, 0.0)
    speak(d, h1, 0.0)
    h2 = cue(2, "hedge two.", hedge=True, t=1.0)
    d.offer(h2, 1.0)
    assert d.next(1.1) is None  # blocked: a hedge within the last 2 utterances
    n1 = cue(2, "plain one.", t=1.2)
    d.offer(n1, 1.2)
    assert d.next(1.2) is n1
    d.on_spoken(n1, 1.5)
    h3 = cue(2, "hedge three.", hedge=True, t=1.6)
    d.offer(h3, 1.6)
    assert d.next(1.6) is None  # [hedge, plain]: still blocked
    n2 = cue(2, "plain two.", t=1.7)
    d.offer(n2, 1.7)
    assert d.next(1.7) is n2
    d.on_spoken(n2, 2.0)
    assert d.next(2.1) is h3  # [plain, plain]: allowed


def test_hedge_cap_does_not_apply_to_accessibility() -> None:
    d = Director(DirectorConfig(min_gap_s=0.0))
    for i in range(3):
        c = cue(1, f"Shot, left wing {i}.", hedge=True, channel="accessibility", t=float(i))
        d.offer(c, float(i))
        speak(d, c, float(i), dur=0.5)


# --------------------------------------------------------------------------- corrections


def test_revision_before_speech_is_silent() -> None:
    d = Director()
    made = shot_made(3)
    d.offer(cue(1, events=[made]), 0.0)
    new = replace(made, player_id="home-3", revision=1, attrs=dict(made.attrs))
    out = d.on_revision(new, 0.5, old=made)
    assert out.action == "invalidated" and out.dropped == 1
    assert d.pending() == []


def test_spoken_material_fact_needs_one_correction() -> None:
    d = Director()
    made = shot_made(3)
    c = cue(1, events=[made])
    d.offer(c, 0.0)
    speak(d, c, 0.0)
    new = replace(made, player_id="home-3", revision=1, attrs=dict(made.attrs))
    out = d.on_revision(new, 5.0)
    assert out.action == "correct" and out.material and out.spoken is made
    corr = cue(0, "Correction: that bucket goes to {P02}.", channel="correction", kind="correction", events=[new])
    d.offer(corr, 5.0)
    speak(d, corr, 5.0)
    again = replace(new, value=2, revision=2)
    assert d.on_revision(again, 9.0).action == "suppressed"  # never correct the same event twice


def test_spoken_non_material_fact_is_fixed_silently() -> None:
    d = Director()
    reb = ev(EventKind.REBOUND, team=Team.HOME, player_id="home-23", attrs={"offensive": False})
    c = cue(1, events=[reb])
    d.offer(c, 0.0)
    speak(d, c, 0.0)
    out = d.on_revision(replace(reb, player_id="home-3", revision=1, attrs=dict(reb.attrs)), 3.0)
    assert out.action == "suppressed" and not out.material


def test_confirmation_only_revision_and_revert_are_no_ops() -> None:
    d = Director()
    made = shot_made(3, status=EventStatus.PROVISIONAL)
    c = cue(1, events=[made])
    d.offer(c, 0.0)
    speak(d, c, 0.0)
    confirmed = replace(made, status=EventStatus.CONFIRMED, confidence=0.99, revision=1, attrs=dict(made.attrs))
    assert d.on_revision(confirmed, 2.0).action == "none"
    assert not fact_changed(made, confirmed)
    retracted = replace(made, status=EventStatus.RETRACTED, revision=2, attrs=dict(made.attrs))
    assert d.on_revision(retracted, 3.0).action == "correct"
    restored = replace(made, revision=3, attrs=dict(made.attrs))  # back to what was spoken
    assert d.on_revision(restored, 4.0).action == "none"


def test_revision_while_speaking_requests_preemption() -> None:
    d = Director()
    made = shot_made(3)
    c = cue(1, events=[made])
    d.offer(c, 0.0)
    assert d.next(0.0) is c
    out = d.on_revision(replace(made, value=2, revision=1, attrs=dict(made.attrs)), 0.3)
    assert out.preempt and out.action == "correct"
    assert d.preempt_requested(0.3)


def test_material_change_rules() -> None:
    made = shot_made(3)
    assert is_material_change(made, replace(made, player_id="home-3"))
    assert is_material_change(made, replace(made, value=2))
    assert is_material_change(made, replace(made, status=EventStatus.RETRACTED))
    assert not is_material_change(made, replace(made, x=1.0))
    ast = ev(EventKind.ASSIST, team=Team.HOME, player_id="home-3")
    assert not is_material_change(ast, replace(ast, player_id="home-0"))
    to = ev(EventKind.TIMEOUT, team=Team.HOME, source="manual")
    assert is_material_change(to, replace(to, team=Team.AWAY))


def _spoken_correction(d: Director, t: float, game_time: float | None = None) -> None:
    c = cue(0, f"Correction {t}.", channel="correction", kind="correction", event_ids=[f"e{t}"])
    d.offer(c, t)
    assert d.next(t) is c
    d.on_spoken(c, t + 1.0, game_time=game_time)


def test_correction_storm_escalation() -> None:
    d = Director()
    for t in (10.0, 100.0):
        _spoken_correction(d, t)
    assert d.gate_boost == 0.0 and d.hold_boost_s == 0.0
    _spoken_correction(d, 200.0)  # third within 5 minutes
    assert d.escalated
    assert d.gate_boost == pytest.approx(0.03) and d.hold_boost_s == pytest.approx(0.2) == d.hold_boost
    d.on_period_start()
    assert d.gate_boost == 0.0 and not d.escalated


def test_no_escalation_when_corrections_are_spread_out() -> None:
    d = Director()
    for t in (0.0, 200.0, 400.0, 600.0):
        _spoken_correction(d, t)
    assert not d.escalated


def test_storm_window_uses_game_time_when_given() -> None:
    d = Director()
    for i, t in enumerate((0.0, 200.0, 400.0)):  # 400 s of video but only 4 game minutes
        _spoken_correction(d, t, game_time=80.0 * i)
    assert d.escalated


# --------------------------------------------------------------------------- misc


def test_priority_for_families_and_clutch() -> None:
    assert priority_for("correction:scorer") == 0
    assert priority_for("made3") == 1 and priority_for("release") == 1
    assert priority_for("possession") == 2 and priority_for("color:derived") == 3
    clutch = {"clock": {"is_last_period": True, "seconds_left": 90.0}, "score": {"home": 100, "away": 99},
              "event": {"value": 3, "team_side": "home"}}
    assert priority_for("made3", clutch) == 0  # trailing by 2 before the play, last 2:00
    blowout = dict(clutch, score={"home": 120, "away": 99})
    assert priority_for("made3", blowout) == 1


def test_invalid_priority_rejected_and_blank_text_ignored() -> None:
    d = Director()
    with pytest.raises(ValueError):
        d.offer(cue(4), 0.0)
    d.offer(cue(1, "   "), 0.0)
    assert d.pending() == []


def test_speaking_timeout_releases_a_lost_line() -> None:
    d = Director(DirectorConfig(speaking_timeout_s=5.0))
    c = cue(2)
    d.offer(c, 0.0)
    assert d.next(0.0) is c
    nxt = cue(2, "next.", t=6.0)
    d.offer(nxt, 6.0)
    assert d.next(6.1) is nxt
