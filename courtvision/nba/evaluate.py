"""Score an analysis of a real NBA broadcast against the official box score and the aligned play-by-play.

Two views, both from ESPN (``espn.py``) placed on video time through the scorebug clock (``truth.py``):

* **Box score** (``team_box`` vs ``espn.box_score``): per team PTS, FGM/FGA, 3PM/3PA, FTM/FTA, OREB/DREB, AST,
  STL, BLK, TOV; the headline is the share of the official counting totals missed or added,
  ``sum |pred - truth| / sum truth`` over those stats (``box_error``), lower is better.
* **Events** (``truth_events`` + ``eval/events.match_events``): shots, rebounds, assists, steals, blocks and
  turnovers matched in time (``tol_s``, the play-by-play clock has 1 s resolution and lags or leads the action by
  about a second) and team; F1 per kind.

Team rebounds of the feed (no player) are not in the official box score and are left out of both sides.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from courtvision.eval.events import EventReport, latest_revisions, match_events
from courtvision.training.align_pbp import AlignedAction
from courtvision.types import Event, EventKind, EventStatus, Team

from .espn import TeamBox

__all__ = ["BOX_STATS", "BoxComparison", "box_error", "compare", "team_box", "truth_events"]

BOX_STATS = ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov")
EVENT_KINDS = (EventKind.SHOT_ATTEMPT, EventKind.SHOT_MADE, EventKind.SHOT_MISSED, EventKind.REBOUND,
               EventKind.ASSIST, EventKind.STEAL, EventKind.BLOCK, EventKind.TURNOVER)


def team_box(events: Iterable[Event]) -> dict[Team, dict[str, int]]:
    """Team counting stats from (latest-revision) events."""
    out = {t: dict.fromkeys(BOX_STATS, 0) for t in (Team.HOME, Team.AWAY)}
    for e in latest_revisions(events):
        if e.status is EventStatus.RETRACTED or e.team is None:
            continue
        b = out[e.team]
        if e.kind in (EventKind.SHOT_MADE, EventKind.SHOT_MISSED):
            ft = e.value == 1 or e.attrs.get("shot_type") == "free_throw"
            made = e.kind is EventKind.SHOT_MADE
            if ft:
                b["fta"] += 1
                b["ftm"] += int(made)
            else:
                three = e.value == 3  # the potential points of the attempt (engine and truth alike)
                b["fga"] += 1
                b["fgm"] += int(made)
                b["fg3a"] += int(three)
                b["fg3m"] += int(three and made)
            if made:
                b["pts"] += 1 if ft else (3 if e.value == 3 else 2)
        elif e.kind is EventKind.REBOUND and e.player_id is not None:
            b["oreb" if e.attrs.get("offensive") else "dreb"] += 1
        elif e.kind is EventKind.ASSIST:
            b["ast"] += 1
        elif e.kind is EventKind.STEAL:
            b["stl"] += 1
        elif e.kind is EventKind.BLOCK:
            b["blk"] += 1
        elif e.kind is EventKind.TURNOVER:
            b["tov"] += 1
    return out


@dataclass
class BoxComparison:
    rows: list[dict[str, Any]] = field(default_factory=list)  # team, stat, pred, truth, diff
    error: float = 0.0  # sum |diff| / sum truth over BOX_STATS and both teams
    exact_share: float = 0.0  # share of (team, stat) cells exactly right

    def table(self, abbr: Mapping[Team, str]) -> str:
        lines = ["| stat | " + " | ".join(f"{abbr[t]} pred | {abbr[t]} truth" for t in (Team.AWAY, Team.HOME)) + " |",
                 "|---|" + "---:|" * 4]
        by = {(r["team"], r["stat"]): r for r in self.rows}
        for s in BOX_STATS:
            cells = []
            for t in (Team.AWAY, Team.HOME):
                r = by[(t.value, s)]
                cells += [str(r["pred"]), str(r["truth"])]
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
        return "\n".join(lines)


def box_error(pred: Mapping[Team, Mapping[str, int]], truth: Mapping[Team, TeamBox]) -> BoxComparison:
    rows = []
    for t in (Team.HOME, Team.AWAY):
        for s in BOX_STATS:
            p, q = int(pred[t].get(s, 0)), int(truth[t].stats.get(s, 0))
            rows.append({"team": t.value, "stat": s, "pred": p, "truth": q, "diff": p - q})
    tot = sum(r["truth"] for r in rows)
    err = sum(abs(r["diff"]) for r in rows) / max(1, tot)
    exact = sum(r["diff"] == 0 for r in rows) / len(rows)
    return BoxComparison(rows, round(err, 4), round(exact, 4))


def truth_events(aligned: Iterable[AlignedAction], window: tuple[float, float] | None = None,
                 made_flight_s: float = 1.0) -> list[Event]:
    """Aligned play-by-play as ``Event`` truth (video time). A shot row gives SHOT_ATTEMPT + its result."""
    out: list[Event] = []
    for i, al in enumerate(aligned):
        a = al.action
        if al.t is None or a.kind is None or a.synthetic:
            continue
        if window is not None and not window[0] <= al.t <= window[1]:
            continue
        base = {"t": float(al.t), "frame_idx": 0, "team": a.team, "player_id": a.player_id, "source": "feed",
                "status": EventStatus.CONFIRMED}
        if a.is_shot:
            ft = a.is_free_throw
            attrs = {"shot_type": "free_throw" if ft else ("three" if a.value == 3 else "two")}
            sid = f"pbp{i}"
            # a made basket is placed at the ball going in; its release is about a flight earlier
            att = dict(base, t=base["t"] - (made_flight_s if a.made and not ft else 0.0))
            out.append(Event(f"{sid}a", EventKind.SHOT_ATTEMPT, value=a.value, shot_id=sid, attrs=dict(attrs), **att))
            out.append(Event(f"{sid}r", EventKind.SHOT_MADE if a.made else EventKind.SHOT_MISSED,
                             value=a.value, shot_id=sid, attrs=dict(attrs), **base))
        elif a.kind is EventKind.REBOUND:
            if a.player_id is None:
                continue  # team rebound: feed bookkeeping, not in the box score
            out.append(Event(f"pbp{i}", EventKind.REBOUND, attrs={"offensive": bool(a.attrs.get("offensive"))}, **base))
        elif a.kind in (EventKind.ASSIST, EventKind.STEAL, EventKind.BLOCK, EventKind.TURNOVER):
            out.append(Event(f"pbp{i}", a.kind, other_player_id=a.attrs.get("other_player_id"), **base))
    return out


def compare(pred_events: Iterable[Event], aligned: Iterable[AlignedAction], truth_box: Mapping[Team, TeamBox],
            tol_s: float = 2.5, window: tuple[float, float] | None = None) -> dict[str, Any]:
    """Box-score and event comparison of one analysis (``window``: video seconds analysed, None = the game)."""
    pred = list(pred_events)
    truth = truth_events(aligned, window)
    report: EventReport = match_events(pred, truth, kinds=list(EVENT_KINDS), tol_s=tol_s, match_player=False,
                                       window=window)
    out: dict[str, Any] = {"events": {k: {"f1": r.f1, "precision": r.precision, "recall": r.recall,
                                          "n_pred": r.n_pred, "n_truth": r.n_truth}
                                      for k, r in report.per_kind.items()}}
    if window is None:
        cmp = box_error(team_box(pred), truth_box)
        out["box"] = {"error": cmp.error, "exact_share": cmp.exact_share, "rows": cmp.rows}
    else:
        tb = team_box(truth)
        out["box_window"] = {"pred": {t.value: v for t, v in team_box(pred).items()},
                             "truth": {t.value: v for t, v in tb.items()}}
    return out
