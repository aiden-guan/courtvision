"""How the scorebug's shot clock lines up with the real play: next to each shot-clock event (``reset24``,
``reset14``, ``run24``; ``nba/scorebug_events.py``), what the aligned play-by-play says happened around it, so the
fusion rules built on them (``nba/fusion.py``: rim misses, shot-clock rebounds, offensive-rebound misses) can be
checked on a real game (research use; docs/NBA.md).

    .venv/bin/python tools/nba_bug_study.py runs/nba/<game> [--start S --end S] [--split S]

For each kind (``reset24`` split by whether the game clock runs, ``reset14`` likewise, ``run24``) it counts the true
action nearest to the event in a window: missed / made field goal, missed / made free throw, offensive / defensive /
team rebound, steal, turnover, block, foul, or none. It also gives the time from a running ``reset24`` to the next
``run24`` for each of the true causes (a rim touch then a defensive rebound waits for the rebound; a steal starts
at once).

``triggers``: the fusion rules' scorebug-only conditions (``FusionConfig`` defaults), each against the play-by-play:

* ``rim_miss`` (``rim_misses``): a ``reset24`` with the game clock running and no score change within
  ``reset_make_after_s`` after it. Precision: a true missed field goal in the ``reset24`` window; recall: the share of
  true missed field goals with such a reset near them.
* ``oreb_miss`` (``oreb_misses``): a ``reset14`` with the clock running at it and 1.5 s later and no clock stop
  around it. Precision: a true offensive rebound in the ``reset14`` window; recall: the share of true (player)
  offensive rebounds with such a reset near them.
* ``bug_oreb`` / ``bug_dreb`` (``bug_rebounds``, run on the true missed field goals): the first of a running
  ``reset14`` or a put-back basket (offensive), a ``run24`` (defensive) or a clock stop (open) within
  ``bug_rebound_window_s`` after the miss and before the next one. Precision: the true rebound of the miss is of that
  kind; recall: the share of the true misses with that rebound that it decides.
* ``rim_miss_held`` (``held_resets``): ``rim_miss`` without the resets whose count starts at once.
* ``team_reb`` (``team_rebounds``): a clock stop is the first sign after the true miss. Precision: the play-by-play
  has a team rebound for it (``rows``: what it has); recall: the share of the team rebounds of misses it finds.
* ``turnover_kept`` / ``steal_kept`` (``turnover_resets``): the share of the true turnovers / steals with a reset to
  24 that no field-goal result or score change explains near them (or the shot clock off), i.e. kept by the filter.
* ``deadball_to`` (``deadball_turnovers``): such resets at a stoppage with no field-goal result before the whistle
  and no free throw before the clock restarts, by the true action around them (``rows``); ``all_rows`` / ``with_ft``:
  every such reset, and those a free throw follows (a foul, which the rule skips).

``--split S`` reports [start, S) and [S, end] apart (e.g. the stretch the rules were tuned on and the rest of the
game). Writes ``bug_study.json`` in the game folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from courtvision.nba.analyze import load_scorebug  # noqa: E402
from courtvision.nba.fusion import FusionConfig  # noqa: E402
from courtvision.nba.scorebug_events import BugEvent, scorebug_events  # noqa: E402
from courtvision.types import EventKind  # noqa: E402
from nba_analyze import truth  # noqa: E402

# where to look for each kind's cause, relative to the bug event (s): the play-by-play times are about 1-2 s off
WINDOWS = {"reset24": (-4.0, 1.5), "reset14": (-4.0, 1.5), "run24": (-5.0, 1.0)}


def category(a) -> str | None:  # noqa: ANN001 (align_pbp.PBPAction)
    if a.kind in (EventKind.SHOT_MADE, EventKind.SHOT_MISSED):
        ft = a.is_free_throw
        return f"{'ft' if ft else 'fg'}_{'make' if a.made else 'miss'}"
    if a.kind is EventKind.REBOUND:
        if a.player_id is None:
            return "team_reb"
        return "oreb" if a.attrs.get("offensive") else "dreb"
    return {EventKind.STEAL: "steal", EventKind.TURNOVER: "turnover", EventKind.BLOCK: "block",
            EventKind.FOUL: "foul"}.get(a.kind)  # type: ignore[arg-type]


def running_at(bug: list[BugEvent]):  # noqa: ANN201
    clock = sorted((b.t, b.kind == "clock_start") for b in bug if b.kind in ("clock_start", "clock_stop"))
    ts = np.array([c[0] for c in clock])

    def running(t: float) -> bool:
        i = int(np.searchsorted(ts, t, side="right")) - 1
        return i < 0 or clock[i][1]

    return running


def is_oreb(a) -> bool:  # noqa: ANN001
    return a.kind is EventKind.REBOUND and bool(a.attrs.get("offensive"))


def is_fg_miss(a) -> bool:  # noqa: ANN001
    return a.kind is EventKind.SHOT_MISSED and not a.is_free_throw


def kinds(bug: list[BugEvent], acts: list, running) -> dict:  # noqa: ANN001
    """The true action next to each shot-clock event (``counts``) and the running reset24 -> run24 times."""
    counts: dict[str, Counter] = defaultdict(Counter)
    gaps: dict[str, list[float]] = defaultdict(list)
    runs = [b.t for b in bug if b.kind == "run24"]
    for b in bug:
        if b.kind not in WINDOWS:
            continue
        key = b.kind if b.kind == "run24" else f"{b.kind} ({'running' if running(b.t) else 'stopped'})"
        a, z = WINDOWS[b.kind]
        near = [(t, c) for t, c, _ in acts if a <= t - b.t <= z]
        if b.kind == "run24":
            # what gave the ball: the latest change of possession before the count (a rebound, a steal, a make)
            cause = max(near, key=lambda r: r[0])[1] if near else "none"
        else:
            cause = min(near, key=lambda r: abs(r[0] - b.t))[1] if near else "none"
        counts[key][cause] += 1
        if b.kind == "reset24" and running(b.t):
            nxt = next((t for t in runs if 0.0 < t - b.t <= 6.0), None)
            if nxt is not None:
                gaps[cause].append(round(nxt - b.t, 2))
    return {"counts": {k: dict(v.most_common()) for k, v in counts.items()},
            "reset24_to_run24_s": {k: {"n": len(v), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                                       "p90": float(np.percentile(v, 90))} for k, v in gaps.items() if v}}


def _pr(n: int, hits: int, n_true: int, found: int) -> dict:
    return {"n": n, "hits": hits, "precision": round(hits / n, 3) if n else None,
            "true": n_true, "found": found, "recall": round(found / n_true, 3) if n_true else None}


def triggers(bug: list[BugEvent], acts: list, running, lo: float, hi: float,  # noqa: ANN001
             cfg: FusionConfig | None = None) -> dict:
    """Precision / recall in [lo, hi) of each fusion trigger (the module docstring). ``bug`` and ``acts`` cover the
    whole game: a trigger or a miss near the section's edge sees what lies past it."""
    cfg = cfg or FusionConfig()
    scores = [b.t for b in bug if b.kind == "score"]
    stops = [b.t for b in bug if b.kind == "clock_stop"]
    inside = [b for b in bug if lo <= b.t < hi]
    misses = [(t, a) for t, _, a in acts if is_fg_miss(a)]
    orebs = [(t, a) for t, _, a in acts if is_oreb(a)]
    out = {}
    a, z = WINDOWS["reset24"]
    rim24 = [b for b in inside if b.kind == "reset24" and running(b.t)
             and not any(0.0 <= s - b.t <= cfg.reset_make_after_s for s in scores)]
    # held_resets: a reset whose count starts at once (a run24 from the value before it) is no rim touch
    held = [r for r in rim24 if not any(b.kind == "run24" and b.shot_clock_before == r.shot_clock_before
                                        and 0.0 < b.t - r.t <= 2.0 for b in bug)]
    for key, rs in (("rim_miss", rim24), ("rim_miss_held", held)):
        rim = [r.t for r in rs]
        out[key] = _pr(len(rim), sum(any(a <= t - r <= z for t, _ in misses) for r in rim),
                       sum(lo <= t < hi for t, _ in misses),
                       sum(any(a <= t - r <= z for r in rim) for t, _ in misses if lo <= t < hi))
    a, z = WINDOWS["reset14"]
    r14 = [b.t for b in inside if b.kind == "reset14" and running(b.t) and running(b.t + 1.5)
           and not any(b.t - 0.5 <= s <= b.t + 1.5 for s in stops)]
    live = [(t, x) for t, x in orebs if x.player_id is not None]  # a team rebound: mostly a dead ball
    out["oreb_miss"] = _pr(len(r14), sum(any(a <= t - r <= z for t, _ in orebs) for r in r14),
                           sum(lo <= t < hi for t, _ in live),
                           sum(any(a <= t - r <= z for r in r14) for t, _ in live if lo <= t < hi))
    # bug_rebounds on the true misses; the true rebound of a miss is the next rebound row before the next shot row
    marks = sorted((b for b in bug if b.kind in ("reset14", "run24", "score", "clock_stop")), key=lambda b: b.t)
    rows = [x for _, _, x in acts]
    tally = {"bug_oreb": [0, 0, 0, 0], "bug_dreb": [0, 0, 0, 0]}  # fired, hits, true, found
    team = [0, 0, 0, 0]  # team_rebounds: a clock stop first = a team rebound (no player's)
    team_rows: Counter = Counter()  # what the play-by-play has after the misses whose first sign is a clock stop
    for k, (t, m) in enumerate(misses):
        if not lo <= t < hi:
            continue
        j = next(i for i, x in enumerate(rows) if x is m)
        reb = next((x for x in rows[j + 1:] if x.kind in (EventKind.REBOUND, EventKind.SHOT_MADE,
                                                          EventKind.SHOT_MISSED)), None)
        true = None if reb is None or reb.kind is not EventKind.REBOUND else \
            ("bug_oreb" if is_oreb(reb) else "bug_dreb")
        nxt_t = misses[k + 1][0] if k + 1 < len(misses) else float("inf")
        mark = next((b for b in marks if t - 0.5 < b.t <= min(t + cfg.bug_rebound_window_s, nxt_t)), None)
        said = None
        if mark is not None and mark.kind == "reset14" and running(mark.t):
            said = "bug_oreb"
        elif mark is not None and mark.kind == "run24":
            said = "bug_dreb"
        elif mark is not None and mark.kind == "score" and mark.team is m.team and mark.points >= 2:
            said = "bug_oreb"
        if said is not None:
            tally[said][0] += 1
            tally[said][1] += said == true
        if true is not None:
            tally[true][2] += 1
            tally[true][3] += said == true
        is_team = true is not None and reb.player_id is None
        stop_first = mark is not None and mark.kind == "clock_stop"
        team[0] += stop_first
        team[1] += stop_first and is_team
        team[2] += is_team
        team[3] += stop_first and is_team
        if stop_first:
            team_rows["none" if reb is None or true is None else category(reb)] += 1
    for key, v in tally.items():
        out[key] = _pr(*v)
    out["team_reb"] = {**_pr(*team), "rows": dict(team_rows.most_common())}
    out.update(turnover_triggers(bug, acts, running, lo, hi, cfg))
    return out


def turnover_triggers(bug: list[BugEvent], acts: list, running, lo: float, hi: float,  # noqa: ANN001
                      cfg: FusionConfig) -> dict:
    """``turnover_resets`` / ``deadball_turnovers`` (fusion-integration, pass 7) with the play-by-play's field-goal
    results for the engine's shots: a reset to 24 that no shot result (``reset_window_s``) or score change explains
    is a change of possession; ``kept``: the true turnovers / steals the filter keeps (such a reset within
    ``reset_turnover_s`` of them or of the whistle, or the shot clock off); ``deadball``: such resets at a stoppage
    with no shot result in the ``rebound_window_s`` before the whistle, by the true action around the whistle."""
    a, z = cfg.reset_window_s
    fg = [t for t, _, x in acts if x.kind in (EventKind.SHOT_MADE, EventKind.SHOT_MISSED) and not x.is_free_throw]
    scores = [b.t for b in bug if b.kind == "score"]
    stops = sorted(b.t for b in bug if b.kind == "clock_stop")
    clocks = sorted((b.t, b.game_clock) for b in bug if b.game_clock is not None)

    def whistle(r: BugEvent) -> float:
        return r.t if running(r.t) else next((t for t in reversed(stops) if t <= r.t), r.t)

    def explained(r: BugEvent) -> bool:
        if any(a <= t - r.t <= z for t in fg) or any(0.0 <= s - r.t <= cfg.reset_make_after_s for s in scores):
            return True
        w = whistle(r)
        return not running(r.t) and any(w - cfg.score_lag_s <= s <= r.t for s in scores)  # the free throws

    def clock_off(t: float) -> bool:
        before = [g for tb, g in clocks if tb <= t]
        return bool(before) and before[-1] < 24.0

    changes = [(whistle(r), r) for r in bug if r.kind == "reset24" and not explained(r)]
    out = {}
    for kind, key in ((EventKind.TURNOVER, "turnover_kept"), (EventKind.STEAL, "steal_kept")):
        ts = [t for t, _, x in acts if x.kind is kind and lo <= t < hi]
        near = [any(w - cfg.reset_turnover_s <= t <= r.t + cfg.reset_turnover_s for w, r in changes) for t in ts]
        off = [clock_off(t) for t in ts]
        out[key] = {"true": len(ts), "kept": sum(n or o for n, o in zip(near, off, strict=True)),
                    "by_reset": sum(near), "clock_off": sum(o and not n for n, o in zip(near, off, strict=True)),
                    "share": round(sum(n or o for n, o in zip(near, off, strict=True)) / len(ts), 3) if ts else None}
    rows: Counter = Counter()
    ft_after: Counter = Counter()  # of those, a free throw (a 1-point score change) before the game clock restarts
    fired: Counter = Counter()  # the others: the resets deadball_turnovers turns into turnovers
    starts = sorted(b.t for b in bug if b.kind == "clock_start")
    ones = [b.t for b in bug if b.kind == "score" and b.points == 1]
    for w, r in changes:
        if running(r.t) or not lo <= r.t < hi or any(w - cfg.rebound_window_s <= t <= r.t for t in fg):
            continue
        near = {c for t, c, _ in acts if w - cfg.reset_turnover_s <= t <= r.t + cfg.reset_turnover_s}
        label = "turnover" if "turnover" in near else "foul" if "foul" in near else \
            (sorted(near)[0] if near else "none")
        rows[label] += 1
        restart = next((t for t in starts if t > r.t), float("inf"))
        ft = any(r.t <= t <= restart for t in ones)
        ft_after[label] += ft
        if not ft:  # what deadball_turnovers fires on: no free throw before the restart
            fired[label] += 1
    n, m = sum(rows.values()), sum(fired.values())
    out["deadball_to"] = {"n": m, "hits": fired["turnover"], "precision": round(fired["turnover"] / m, 3) if m else None,
                          "rows": dict(fired.most_common()), "with_ft": dict(ft_after.most_common()), "all": n,
                          "all_rows": dict(rows.most_common())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    ap.add_argument("--split", type=float, default=0.0, help="report [start, split) and [split, end] apart")
    args = ap.parse_args()
    root = Path(args.root)
    lo, hi = args.start, (args.end or 1e9)
    allbug = scorebug_events(load_scorebug(root / "scorebug.jsonl"))
    _, aligned = truth(root)
    acts = sorted(((al.t, category(al.action), al.action) for al in aligned.aligned
                   if al.t is not None and not al.action.synthetic and category(al.action) is not None),
                  key=lambda r: r[0])
    running = running_at(allbug)
    sections = {"all": (lo, hi)}
    if lo < args.split < hi:
        sections = {f"{lo:g}-{args.split:g}": (lo, args.split), f"{args.split:g}-end": (args.split, hi)}
    out = {}
    for name, (a, z) in sections.items():
        out[name] = kinds([b for b in allbug if a <= b.t < z], [r for r in acts if a <= r[0] < z], running)
        out[name]["triggers"] = triggers(allbug, acts, running, a, z)
    (root / "bug_study.json").write_text(json.dumps(out if len(out) > 1 else out["all"], indent=1))
    for name, doc in out.items():
        print(f"== {name}")
        for k, v in sorted(doc["counts"].items()):
            n = sum(v.values())
            print(f"{k} ({n}): " + ", ".join(f"{c} {m} ({m / n:.0%})" for c, m in v.items()))
        print("running reset24 -> next run24 (s), by the nearest true action:")
        for k, v in sorted(doc["reset24_to_run24_s"].items()):
            print(f"  {k}: n={v['n']} median {v['median']:.2f} (p10 {v['p10']:.2f}, p90 {v['p90']:.2f})")
        print("triggers: precision (hits / fired), recall (found / true)")
        for k, v in doc["triggers"].items():
            if "kept" in v:
                share = "-" if v["share"] is None else f"{v['share']:.0%}"
                print(f"  {k}: {share} ({v['kept']}/{v['true']}: {v['by_reset']} by a reset, {v['clock_off']} "
                      f"with the shot clock off)")
                continue
            prec = "-" if v["precision"] is None else f"{v['precision']:.0%}"
            line = f"  {k}: precision {prec} ({v['hits']}/{v['n']})"
            if "recall" in v:
                rec = "-" if v["recall"] is None else f"{v['recall']:.0%}"
                line += f", recall {rec} ({v['found']}/{v['true']})"
            print(line + (f"; play-by-play: {v['rows']}" if "rows" in v else "")
                  + (f"; skipped for a free throw before the restart: {v['with_ft']}" if "with_ft" in v else ""))
    print(f"\n-> {root / 'bug_study.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
