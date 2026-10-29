"""What the scorebug says about the play: score changes, shot-clock resets and clock stops (NBA rules).

The broadcast scorebug is read at a few Hz (``perception/scorebug_ocr.py``). Beyond the score, its two clocks
encode most possession-level events of an NBA game:

* **Score change** (+1 / +2 / +3 for a team, confirmed by two equal reads): a made free throw or field goal. The
  ESPN bug shows it about 2 s after the ball goes in.
* **Shot clock to 24** (from below 23, while the game clock runs or right after it stops): the ball hit the rim
  (NBA rule: the shot clock resets on a rim touch), a made basket, or a change of possession (a turnover, a
  defensive rebound of an airball).
* **Shot clock 24 -> 14** (or up to 14 from below): the offence kept the ball after a rim touch (an offensive
  rebound) or a foul / kicked ball in its front court (NBA rule 7).
* **Game clock stops / starts**: dead balls (fouls, violations, timeouts, out of bounds, made baskets in the last
  two minutes of a period) and the restart.

``scorebug_events`` turns confirmed reads into these ``BugEvent`` rows with times; ``nba/fusion.py`` uses them to
support, add or drop the event engine's shots and to decide offensive / defensive rebounds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from courtvision.types import ScoreboardReading, Team

__all__ = ["BugEvent", "scorebug_events"]


@dataclass(frozen=True)
class BugEvent:
    t: float  # first read showing the change (video seconds)
    kind: str  # "score" | "reset24" | "reset14" | "clock_stop" | "clock_start"
    team: Team | None = None  # score: the team that scored
    points: int = 0
    shot_clock_before: float | None = None
    game_clock: float | None = None
    period: int | None = None


def _confirmed(values: Sequence[tuple[float, float | None]]) -> list[tuple[float, float]]:
    """(t, v) where two consecutive reads agree (the second read's time is reported as the first's)."""
    out = []
    for (t0, v0), (_t1, v1) in zip(values[:-1], values[1:], strict=True):
        if v0 is not None and v0 == v1 and (not out or out[-1][1] != v0 or t0 - out[-1][0] > 1.5):
            out.append((t0, float(v0)))
    return out


def scorebug_events(reads: Sequence[ScoreboardReading], max_jump_gap_s: float = 3.0,
                    persist_s: float = 4.0) -> list[BugEvent]:
    """Score changes, shot-clock resets and game-clock stops / starts from scorebug reads (time order)."""
    reads = sorted(reads, key=lambda r: r.t)
    out: list[BugEvent] = []
    # score: per team, confirmed increases of 1-3
    first_period = next((r.period for r in reads if r.period is not None), None)
    for team, attr in ((Team.HOME, "home_score"), (Team.AWAY, "away_score")):
        conf = _confirmed([(r.t, getattr(r, attr)) for r in reads])
        # ESPN draws a lone 0 as a ring the recogniser skips: a game read from its start begins at 0
        cur = 0.0 if first_period == 1 and conf and conf[0][1] <= 3 else None
        for k, (t, v) in enumerate(conf):
            if cur is None:
                cur = v
                continue
            if v == cur:
                continue
            # a value is believed when it is a basket's step up, or (any other change: another graphic in the
            # score's place, a correction) when it holds for persist_s before the next change
            nxt_t = conf[k + 1][0] if k + 1 < len(conf) else t + 1e9
            holds = nxt_t - t >= persist_s
            if 1 <= v - cur <= 3:
                period = next((r.period for r in reads if r.t >= t), None)
                out.append(BugEvent(t, "score", team, int(v - cur), period=period))
                cur = v
            elif holds:
                cur = v  # a correction (or a jump over a hidden stretch): no basket is invented for it
    # shot clock: jumps up to 24 / 14, and drops from 24 to 14
    sc = _confirmed([(r.t, r.shot_clock) for r in reads])
    gc_at = {r.t: (r.game_clock, r.period) for r in reads}
    for (t0, v0), (t1, v1) in zip(sc[:-1], sc[1:], strict=True):
        if t1 - t0 > max_jump_gap_s:
            continue
        g, p = gc_at.get(t1, (None, None))
        if v1 == 24 and v0 < 23:
            out.append(BugEvent(t1, "reset24", shot_clock_before=v0, game_clock=g, period=p))
        elif v1 == 14 and (v0 >= 23 or v0 < 13):
            out.append(BugEvent(t1, "reset14", shot_clock_before=v0, game_clock=g, period=p))
    # game clock: a value held for >= 1.4 s (0.5 s under a minute, tenths) is stopped; it starts again only when the
    # display counts down twice in a row (a single misread digit during a stoppage is no restart)
    gcs = [(r.t, r.game_clock, r.period) for r in reads if r.game_clock is not None]
    runs: list[tuple[float, float, int | None, float]] = []  # (first t, value, period, last t) of equal reads
    for t, v, p in gcs:
        if runs and runs[-1][1] == v and runs[-1][2] == p:
            runs[-1] = (runs[-1][0], v, p, t)
        else:
            runs.append((t, v, p, t))
    running: bool | None = None
    for k, (t0, v, p, t1) in enumerate(runs):
        stopped = t1 - t0 >= (1.4 if v >= 60 else 0.5)
        if stopped and running is not False:
            out.append(BugEvent(t0 + (1.0 if v >= 60 else 0.1), "clock_stop", game_clock=v, period=p))
            running = False
        elif not stopped and running is False:
            nxt = runs[k + 1] if k + 1 < len(runs) else None
            ticking = nxt is not None and nxt[2] == p and 0 < v - nxt[1] <= 1.5 and v < runs[k - 1][1] \
                and nxt[0] - t0 <= 2.5
            if ticking:
                out.append(BugEvent(t0, "clock_start", game_clock=v, period=p))
                running = True
        elif running is None:
            running = not stopped
    out.sort(key=lambda e: e.t)
    return out
