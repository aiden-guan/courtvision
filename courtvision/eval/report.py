"""Markdown and JSON rendering of evaluation reports.

Every formatter works from the ``to_dict()`` form, so reports loaded back from JSON
render the same as live report objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from courtvision.eval.events import CORE_KINDS

ReportLike = Any  # a report object with to_dict(), or its dict form

_MAX_LISTED = 15  # rows listed for missing / unmatched players before truncating


def _as_dict(report: ReportLike | None) -> dict[str, Any] | None:
    if report is None:
        return None
    if isinstance(report, Mapping):
        return dict(report)
    return report.to_dict()


def report_dict(
    event_report: ReportLike | None, box_report: ReportLike | None = None, track_report: ReportLike | None = None
) -> dict[str, Any]:
    """All reports as one JSON-serialisable dict (absent reports are ``None``)."""
    return {
        "events": _as_dict(event_report),
        "boxscore": _as_dict(box_report),
        "tracking": _as_dict(track_report),
    }


# --------------------------------------------------------------------------- cell formatting


def _f(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() and digits == 0 else f"{value:.{digits}f}"
    return str(value)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


# --------------------------------------------------------------------------- sections


def _kind_rows(kinds: Mapping[str, Mapping[str, Any]], order: Sequence[str]) -> list[list[str]]:
    rows = []
    for name in order:
        k = kinds[name]
        rows.append(
            [
                name,
                str(k["n_truth"]),
                str(k["n_pred"]),
                str(k["tp"]),
                str(k["fp"]),
                str(k["fn"]),
                _f(k["precision"]),
                _f(k["recall"]),
                _f(k["f1"]),
                _f(k["attributed_f1"]),
                _pct(k["player_accuracy"]),
                _f(k["mean_abs_dt"], 2),
            ]
        )
    return rows


_KIND_HEADERS = ("kind", "truth", "pred", "TP", "FP", "FN", "P", "R", "F1", "attr. F1", "player acc", "mean abs dt (s)")


def _events_section(ev: Mapping[str, Any]) -> list[str]:
    head = ev["headline"]
    lines = [
        f"## Events (tolerance +/-{_f(ev['tol_s'], 2)} s)",
        "",
        f"**Headline macro-F1: {_f(head['macro_f1'])}** "
        f"(attributed: {_f(head['attributed_macro_f1'])}) over {len(head['kinds'])} core kinds: "
        f"{', '.join(head['kinds']) or 'none'}.",
        "",
    ]
    per_kind = ev["per_kind"]
    order = [k for k in CORE_KINDS if k in per_kind] + sorted(k for k in per_kind if k not in CORE_KINDS)
    if order:
        lines += _table(_KIND_HEADERS, _kind_rows(per_kind, order)) + [""]
    separate = ev.get("separate") or {}
    if separate:
        lines += ["### Reported separately (not in the headline)", ""]
        lines += _table(_KIND_HEADERS, _kind_rows(separate, sorted(separate))) + [""]
    lines += _shots_section(ev["shots"])
    return lines


def _shots_section(s: Mapping[str, Any]) -> list[str]:
    rows = [
        ["shots paired (truth / pred)", f"{s['n_paired']} ({s['n_truth_shots']} / {s['n_pred_shots']})", ""],
        ["paired by result timing", str(s["n_paired_by_result"]), ""],
        ["FG make/miss accuracy", _pct(s["fg_outcome_accuracy"]), str(s["fg_outcome_n"])],
        ["FT make/miss accuracy", _pct(s["ft_outcome_accuracy"]), str(s["ft_outcome_n"])],
        ["made-FG precision", _f(s["made_fg_precision"]), str(s["n_pred_made_fg"])],
        ["made-FG recall", _f(s["made_fg_recall"]), str(s["n_truth_made_fg"])],
        ["2 vs 3 value accuracy", _pct(s["value_accuracy"]), str(s["value_n"])],
        ["assist attribution accuracy", _pct(s["assist_accuracy"]), str(s["assist_n"])],
        ["shooter accuracy", _pct(s["shooter_accuracy"]), str(s["shooter_n"])],
    ]
    lines = ["### Shots", ""] + _table(("metric", "value", "n"), rows) + [""]
    conf = s.get("outcome_confusion") or {}
    if conf:
        cols = ["made", "missed", "unresolved"]
        body = [[f"truth {r}"] + [str(conf.get(r, {}).get(c, 0)) for c in cols] for r in ("made", "missed")]
        lines += ["FG outcome confusion (rows truth, columns prediction):", ""]
        lines += _table(["", *cols], body) + [""]
    return lines


def _box_section(box: Mapping[str, Any]) -> list[str]:
    lines = [
        "## Box score",
        "",
        f"Players in truth: {box['n_truth_players']}, found in prediction: {box['n_matched_players']}. "
        f"Mean per-stat MAE {_f(box['mean_mae'])}; exact stat lines {_pct(box['line_exact_rate'])}.",
        "",
    ]
    rows = [
        [
            name,
            _f(e["mae"]),
            _f(e["bias"], 2),
            _pct(e["exact_rate"]),
            _f(e["pred_total"], 0),
            _f(e["truth_total"], 0),
        ]
        for name, e in box["per_stat"].items()
    ]
    if rows:
        lines += _table(("stat", "MAE", "bias", "exact", "pred total", "truth total"), rows) + [""]
    pts = box["team_points"]
    score_rows = [[team, _f(r["pred"]), _f(r["truth"]), _f(r["error"])] for team, r in pts.items()]
    lines += _table(("team", "pred pts", "truth pts", "error"), score_rows) + [""]
    lines += [f"Score match: **{_f(box['score_match'])}** (prediction from `{box['points_source']}`).", ""]
    missing = box.get("missing_players") or []
    if missing:
        shown = ", ".join(missing[:_MAX_LISTED]) + (" ..." if len(missing) > _MAX_LISTED else "")
        lines += [f"Truth players absent from the prediction ({len(missing)}): {shown}", ""]
    unmatched = box.get("unmatched_players") or {}
    if unmatched:
        stats = box["stats"]
        lines += [f"Predicted players not in truth ({len(unmatched)}):", ""]
        body = [
            [pid, _f(rec.get("team")), _f(rec.get("provisional"))] + [_f(rec.get(s, 0), 0) for s in stats]
            for pid, rec in list(unmatched.items())[:_MAX_LISTED]
        ]
        lines += _table(["player", "team", "provisional", *stats], body) + [""]
    return lines


def _track_section(tr: Mapping[str, Any]) -> list[str]:
    rows = [
        ["MOTA", _f(tr["mota"])],
        ["IDF1 (lite)", _f(tr["idf1"])],
        ["ID switches", str(tr["idsw"])],
        ["GT / TP / FP / FN", f"{tr['gt']} / {tr['tp']} / {tr['fp']} / {tr['fn']}"],
        ["precision / recall", f"{_f(tr['precision'])} / {_f(tr['recall'])}"],
        ["position RMSE (ft)", _f(tr["rmse_ft"])],
        ["MOTP, mean distance (ft)", _f(tr["motp_ft"])],
        ["identity accuracy", _pct(tr["identity_accuracy"])],
        ["team accuracy", _pct(tr["team_accuracy"])],
        ["frames evaluated / truth frames", f"{tr['n_frames']} / {tr['n_truth_frames']}"],
    ]
    ball = tr["ball"]
    n_hidden = ball["n_frames"] - ball["truth_visible"]
    ball_rows = [
        ["detection rate (truth visible)", _pct(ball["detection_rate"]), str(ball["truth_visible"])],
        ["false visible rate (truth hidden)", _pct(ball["false_visible_rate"]), str(n_hidden)],
        ["interpolated share", _pct(ball["interpolated_share"]), str(ball["pred_visible"])],
        ["image error mean / median (px)", f"{_f(ball['px_mean'], 1)} / {_f(ball['px_median'], 1)}", str(ball["px_n"])],
        ["image error RMSE (px)", _f(ball["px_rmse"], 1), str(ball["px_n"])],
        ["image hit rate", _pct(ball["px_hit_rate"]), str(ball["px_n"])],
    ]
    return (
        [f"## Tracking (court plane, gate {_f(tr['match_radius_ft'], 1)} ft)", ""]
        + _table(("metric", "value"), rows)
        + ["", "### Ball", ""]
        + _table(("metric", "value", "n"), ball_rows)
        + [""]
    )


def format_report(
    event_report: ReportLike | None, box_report: ReportLike | None = None, track_report: ReportLike | None = None
) -> str:
    """Render the given reports (objects or their ``to_dict()`` form) as markdown."""
    lines = ["# CourtVision evaluation", ""]
    ev, box, tr = _as_dict(event_report), _as_dict(box_report), _as_dict(track_report)
    if ev is not None:
        lines += _events_section(ev)
    if box is not None:
        lines += _box_section(box)
    if tr is not None:
        lines += _track_section(tr)
    return "\n".join(lines).rstrip() + "\n"
