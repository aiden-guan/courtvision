"""Box-score evaluation: per-stat MAE / exact-match rate against an independent tally.

Metrics follow training_data.md §4.3 (per-player per-stat MAE, exact stat-line match
rate, team totals, points error). Players are joined by ``player_id``; truth players the
prediction never names count as all-zero lines, and predicted players absent from truth
(provisional ``"home-t12"`` style ids) are reported with what was credited to them.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from courtvision.types import Roster, Team

logger = logging.getLogger(__name__)

TEAMS: tuple[str, str] = (Team.HOME.value, Team.AWAY.value)

# Counting stats of stats/engine.PlayerLine (ARCHITECTURE stats/); FIBA/NBA box-score
# columns per basketball_rules_stats.md §2a and training_data.md §4.3 (sourced).
DEFAULT_BOX_STATS: tuple[str, ...] = (
    "pts",
    "fgm",
    "fga",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "reb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
)


@dataclass
class BoxEvalConfig:
    """Tunables for :func:`compare_box`."""

    # Stats compared; a stat is evaluated only when truth reports it (or can derive it).
    stats: tuple[str, ...] = DEFAULT_BOX_STATS
    # Stats derivable from components when a side omits them (reb = oreb + dreb; definition).
    derived: dict[str, tuple[str, ...]] = field(default_factory=lambda: {"reb": ("oreb", "dreb")})


@dataclass
class StatError:
    """Error of one stat over the players present in truth."""

    stat: str
    n: int
    mae: float
    bias: float  # mean signed pred - truth
    exact_rate: float
    pred_total: float  # over truth players only
    truth_total: float

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class BoxReport:
    """Result of :func:`compare_box`. ``to_dict()`` is JSON-serialisable."""

    stats: list[str]
    n_truth_players: int
    n_matched_players: int
    per_stat: dict[str, StatError]
    mean_mae: float | None  # mean of per-stat MAE
    line_exact_rate: float | None  # truth players whose whole line matches
    missing_players: list[str]  # truth ids absent from the prediction (scored as zeros)
    unmatched_players: dict[str, dict[str, Any]]  # predicted ids absent from truth -> credited stats
    unattributed: dict[str, float]  # stat totals credited to unmatched predicted players
    player_errors: dict[str, dict[str, float]]  # truth id -> stat -> pred - truth (non-zero only)
    team_points: dict[str, dict[str, int | None]]  # team -> {"pred", "truth", "error"}
    team_points_abs_error: int | None
    score_match: bool
    points_source: str  # where predicted team points came from: "score" | "teams" | "players"
    team_totals: dict[str, dict[str, dict[str, float]]]  # team -> stat -> {"pred","truth","error"}
    team_totals_mae: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stats": list(self.stats),
            "n_truth_players": self.n_truth_players,
            "n_matched_players": self.n_matched_players,
            "per_stat": {k: v.to_dict() for k, v in self.per_stat.items()},
            "mean_mae": self.mean_mae,
            "line_exact_rate": self.line_exact_rate,
            "missing_players": list(self.missing_players),
            "unmatched_players": {k: dict(v) for k, v in self.unmatched_players.items()},
            "unattributed": dict(self.unattributed),
            "player_errors": {k: dict(v) for k, v in self.player_errors.items()},
            "team_points": {k: dict(v) for k, v in self.team_points.items()},
            "team_points_abs_error": self.team_points_abs_error,
            "score_match": self.score_match,
            "points_source": self.points_source,
            "team_totals": {t: {s: dict(v) for s, v in d.items()} for t, d in self.team_totals.items()},
            "team_totals_mae": self.team_totals_mae,
        }


# --------------------------------------------------------------------------- input adapters


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _team_key(team: Any) -> str | None:
    if team is None:
        return None
    if isinstance(team, Team):
        return team.value
    value = str(team).lower()
    return value if value in TEAMS else None


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _team_dict(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    items = raw.items() if isinstance(raw, Mapping) else []
    return {k: v for k, v in ((_team_key(t), v) for t, v in items) if k is not None}


def _player_lines(stats: Any) -> dict[str, Any]:
    players = _get(stats, "players") or {}
    if isinstance(players, Mapping):
        return {str(pid): line for pid, line in players.items()}
    return {str(_get(line, "player_id")): line for line in players}


def _stat_value(line: Any, stat: str, derived: Mapping[str, tuple[str, ...]]) -> float | None:
    value = _num(_get(line, stat))
    if value is None and stat in derived:
        parts = [_num(_get(line, s)) for s in derived[stat]]
        if all(p is not None for p in parts):
            value = float(sum(p for p in parts if p is not None))
    return value


def _evaluated_stats(truth_box: Mapping[str, Mapping[str, Any]], cfg: BoxEvalConfig) -> list[str]:
    return [
        s
        for s in cfg.stats
        if any(_stat_value(line, s, cfg.derived) is not None for line in truth_box.values())
    ]


def _id_team(pid: str, roster: Roster | None) -> str | None:
    if roster is not None and (team := roster.team_of(pid)) is not None:
        return team.value
    prefix = pid.split("-", 1)[0].lower()  # "home-23" / "home-t12" id convention (types.py)
    return prefix if prefix in TEAMS else None


def _pred_team(pid: str, pred_line: Any, roster: Roster | None) -> str | None:
    """A predicted line's own team wins; else roster / id prefix."""
    team = _team_key(_get(pred_line, "team")) if pred_line is not None else None
    return team if team is not None else _id_team(pid, roster)


def _truth_team(pid: str, pred_line: Any, roster: Roster | None) -> str | None:
    """Truth teams come from the roster / id prefix; the prediction is only a last resort."""
    team = _id_team(pid, roster)
    if team is None and pred_line is not None:
        team = _team_key(_get(pred_line, "team"))
    return team


# --------------------------------------------------------------------------- metrics


def _per_stat(
    stat_list: list[str],
    truth_ids: list[str],
    truth_vals: dict[str, dict[str, float]],
    pred_vals: dict[str, dict[str, float]],
) -> dict[str, StatError]:
    out: dict[str, StatError] = {}
    for s in stat_list:
        diffs = [pred_vals[pid][s] - truth_vals[pid][s] for pid in truth_ids]
        n = len(diffs)
        out[s] = StatError(
            stat=s,
            n=n,
            mae=sum(abs(d) for d in diffs) / n if n else 0.0,
            bias=sum(diffs) / n if n else 0.0,
            exact_rate=sum(1 for d in diffs if d == 0) / n if n else 0.0,
            pred_total=sum(pred_vals[pid][s] for pid in truth_ids),
            truth_total=sum(truth_vals[pid][s] for pid in truth_ids),
        )
    return out


def _team_points(
    stats: Any, truth_score: Mapping[Any, Any], pred_lines: dict[str, Any], roster: Roster | None
) -> tuple[dict, str]:
    truth = {k: (int(v) if v is not None else None) for k, v in _team_dict(truth_score).items()}
    score = _team_dict(_get(stats, "score"))
    teams = _team_dict(_get(stats, "teams"))
    if score:
        pred, source = {t: _num(score.get(t)) for t in TEAMS}, "score"
    elif teams:
        pred, source = {t: _num(_get(teams.get(t), "pts")) for t in TEAMS}, "teams"
    else:
        pred, source = {t: 0.0 for t in TEAMS}, "players"
        for pid, line in pred_lines.items():
            team = _pred_team(pid, line, roster)
            if team is not None:
                pred[team] = (pred[team] or 0.0) + (_num(_get(line, "pts")) or 0.0)
    rows: dict[str, dict[str, int | None]] = {}
    for t in TEAMS:
        p = int(pred[t]) if pred.get(t) is not None else None
        tr = truth.get(t)
        rows[t] = {"pred": p, "truth": tr, "error": (p - tr) if p is not None and tr is not None else None}
    return rows, source


def _team_totals(
    stat_list: list[str],
    truth_vals: dict[str, dict[str, float]],
    pred_vals: dict[str, dict[str, float]],
    truth_team: dict[str, str | None],
    pred_team: dict[str, str | None],
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for team in TEAMS:
        rows = {}
        for s in stat_list:
            tr = sum(v[s] for pid, v in truth_vals.items() if truth_team.get(pid) == team)
            pr = sum(v[s] for pid, v in pred_vals.items() if pred_team.get(pid) == team)
            rows[s] = {"pred": pr, "truth": tr, "error": pr - tr}
        out[team] = rows
    return out


# Counting stats each truth event adds (stat-line contributions), for the cut-edge alternative below.
def edge_contributions(events: Any) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Box-score lines and team points that ``events`` (truth ``Event`` objects) contribute."""
    from courtvision.types import EventKind as K

    box: dict[str, dict[str, int]] = {}
    score = {t: 0 for t in TEAMS}
    attempts = {e.shot_id: e for e in events if e.kind is K.SHOT_ATTEMPT and e.shot_id}

    def add(pid: str | None, **kw: int) -> None:
        if pid is None:
            return
        line = box.setdefault(pid, {})
        for k, v in kw.items():
            line[k] = line.get(k, 0) + v

    simple = {K.ASSIST: "ast", K.STEAL: "stl", K.BLOCK: "blk", K.TURNOVER: "tov", K.FOUL: "pf"}
    for e in events:
        if e.kind is K.SHOT_ATTEMPT:
            add(e.player_id, **({"fta": 1} if e.value == 1 else {"fga": 1, "fg3a": int(e.value == 3)}))
        elif e.kind is K.SHOT_MADE:
            value = attempts[e.shot_id].value if e.shot_id in attempts else e.value
            add(e.player_id, pts=e.value, **({"ftm": 1} if value == 1 else {"fgm": 1, "fg3m": int(value == 3)}))
            if e.team is not None:
                score[e.team.value] += e.value
        elif e.kind is K.REBOUND:
            add(e.player_id, **({"oreb": 1} if e.attrs.get("offensive") else {"dreb": 1}))
        elif e.kind in simple:
            add(e.player_id, **{simple[e.kind]: 1})
    return box, score


def compare_box(
    stats: Any,
    truth_box: Mapping[str, Mapping[str, Any]],
    truth_score: Mapping[Any, Any],
    roster: Roster | None = None,
    config: BoxEvalConfig | None = None,
    edge: tuple[Mapping[str, Mapping[str, Any]], Mapping[Any, Any]] | None = None,
) -> BoxReport:
    """Compare a ``GameStats`` (object or ``to_dict()`` form) with truth counting stats.

    ``truth_box`` maps player_id -> {stat: count}; ``truth_score`` maps Team (or "home" /
    "away") -> points. ``roster`` (optional) resolves player teams for team totals.

    ``edge`` = (box, score) contributed by truth events at the stream's cut end that the system could
    not yet confirm (``edge_contributions``; e.g. a basket the operator's board shows only after the
    stream stops): each truth value is then taken with or without its edge part, whichever is closer to
    the prediction ("don't care" at a cut edge, like ``events.match_events(window=...)``).
    """
    cfg = config or BoxEvalConfig()
    stat_list = _evaluated_stats(truth_box, cfg)
    pred_lines = _player_lines(stats)
    truth_ids = sorted(truth_box)
    pred_ids = sorted(pred_lines)

    def values(line: Any) -> dict[str, float]:
        return {s: (_stat_value(line, s, cfg.derived) or 0.0) if line is not None else 0.0 for s in stat_list}

    truth_vals = {pid: values(truth_box[pid]) for pid in truth_ids}
    pred_vals = {pid: values(pred_lines.get(pid)) for pid in sorted(set(truth_ids) | set(pred_ids))}
    if edge is not None:
        edge_box, edge_score = edge
        for pid in truth_ids:
            part = values(edge_box.get(pid)) if pid in edge_box else None
            if part is None:
                continue
            for st in stat_list:
                alt = truth_vals[pid][st] - part[st]
                if abs(pred_vals[pid][st] - alt) < abs(pred_vals[pid][st] - truth_vals[pid][st]):
                    truth_vals[pid][st] = alt
        truth_box = {pid: dict(truth_vals[pid]) for pid in truth_ids}
        pred_score = _team_dict(_get(stats, "score")) or {}
        full = {k: (int(v) if v is not None else None) for k, v in _team_dict(truth_score).items()}
        less = _team_dict(edge_score)
        adj: dict[str, Any] = {}
        for t in TEAMS:
            tr, p = full.get(t), _num(pred_score.get(t)) if pred_score else None
            if tr is not None and p is not None and less.get(t):
                alt = tr - int(less[t])
                adj[t] = alt if abs(p - alt) < abs(p - tr) else tr
            else:
                adj[t] = tr
        truth_score = adj
    missing = [pid for pid in truth_ids if pid not in pred_lines]
    extra = [pid for pid in pred_ids if pid not in truth_box]

    per_stat = _per_stat(stat_list, truth_ids, truth_vals, pred_vals)
    player_errors = {
        pid: {s: pred_vals[pid][s] - truth_vals[pid][s] for s in stat_list if pred_vals[pid][s] != truth_vals[pid][s]}
        for pid in truth_ids
    }
    player_errors = {pid: d for pid, d in player_errors.items() if d}
    unmatched: dict[str, dict[str, Any]] = {}
    for pid in extra:
        line = pred_lines[pid]
        rec: dict[str, Any] = {s: v for s, v in pred_vals[pid].items() if v != 0}
        rec["team"] = _team_key(_get(line, "team"))
        rec["provisional"] = _get(line, "provisional")
        unmatched[pid] = rec
    unattributed = {s: sum(pred_vals[pid][s] for pid in extra) for s in stat_list}

    team_rows, source = _team_points(stats, truth_score, pred_lines, roster)
    errors = [r["error"] for r in team_rows.values() if r["error"] is not None]
    known = [r for r in team_rows.values() if r["truth"] is not None]
    score_match = bool(known) and all(r["error"] == 0 for r in known)

    truth_team = {pid: _truth_team(pid, pred_lines.get(pid), roster) for pid in truth_ids}
    pred_team = {pid: _pred_team(pid, pred_lines.get(pid), roster) for pid in pred_vals}
    totals = _team_totals(stat_list, truth_vals, pred_vals, truth_team, pred_team)
    total_errs = [abs(v["error"]) for d in totals.values() for v in d.values()]

    if missing:
        logger.info("compare_box: %d truth players absent from prediction", len(missing))
    return BoxReport(
        stats=stat_list,
        n_truth_players=len(truth_ids),
        n_matched_players=len(truth_ids) - len(missing),
        per_stat=per_stat,
        mean_mae=(sum(e.mae for e in per_stat.values()) / len(per_stat)) if per_stat else None,
        line_exact_rate=(
            sum(1 for pid in truth_ids if pid not in player_errors) / len(truth_ids) if truth_ids else None
        ),
        missing_players=missing,
        unmatched_players=unmatched,
        unattributed=unattributed,
        player_errors=player_errors,
        team_points=team_rows,
        team_points_abs_error=sum(abs(e) for e in errors) if errors else None,
        score_match=score_match,
        points_source=source,
        team_totals=totals,
        team_totals_mae=(sum(total_errs) / len(total_errs)) if total_errs else None,
    )
