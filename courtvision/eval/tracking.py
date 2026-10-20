"""Court-plane multi-object tracking evaluation (CLEAR-MOT style, IDF1-lite, ball).

Per aligned frame, truth players and predicted players are matched one-to-one on court
distance (feet) with a gate of ``match_radius_ft``; matches from the previous frame that
are still inside the gate are kept first (the CLEAR-MOT continuity rule), the rest go to
a Hungarian assignment. Formulas (training_data.md §4.2):

* MOTA = 1 - (FN + FP + IDSW) / GT
* IDF1 = IDTP / (IDTP + 0.5 IDFN + 0.5 IDFP), with a global one-to-one truth->pred id
  mapping chosen to maximise co-matched frames ("lite": co-occurrence comes from the
  per-frame matches, not from a separate distance-thresholded pass).
* identity accuracy: matched pairs whose predicted ``player_id`` equals the truth id
  (box-score attribution depends on it; training_data.md §4.2).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from courtvision.eval.events import gated_assignment
from courtvision.types import BallState, FrameState, PlayerState

logger = logging.getLogger(__name__)


@dataclass
class TrackEvalConfig:
    """Tunables for :func:`tracking_report`."""

    # Predicted identity used for IDSW / IDF1: "player_id" (what stats attribute to) or
    # "track_id" (raw tracker output; falls back to player_id when unset). Design choice.
    pred_id_field: str = "player_id"
    # Keep last frame's correspondences that are still inside the gate before Hungarian
    # (CLEAR-MOT, Bernardin & Stiefelhagen 2008; also py-motmetrics). Sourced.
    keep_previous_matches: bool = True
    # Frame alignment: "frame_idx" (exact) or "t" (nearest within align_tol_s).
    align_by: str = "frame_idx"
    # Max |dt| for t-alignment: just over half a 30 fps frame (1/60 s). Design choice.
    align_tol_s: float = 0.02
    # Ball image-space "hit" radius in px (~1-2 ball radii at 1280x720 wide view; unverified).
    ball_hit_px: float = 10.0


@dataclass
class BallReport:
    """Ball visibility and image-space error."""

    n_frames: int = 0
    truth_visible: int = 0
    pred_visible: int = 0
    detection_rate: float | None = None  # pred visible | truth visible
    false_visible_rate: float | None = None  # pred visible | truth not visible
    interpolated_share: float | None = None  # of pred-visible frames filled by the tracker
    px_n: int = 0
    px_mean: float | None = None
    px_median: float | None = None
    px_rmse: float | None = None
    px_hit_rate: float | None = None  # px error <= ball_hit_px, over frames with both image_xy

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class TrackReport:
    """Result of :func:`tracking_report`. ``to_dict()`` is JSON-serialisable."""

    match_radius_ft: float
    n_frames: int  # aligned frames evaluated
    n_truth_frames: int
    n_pred_frames: int
    n_unaligned_truth_frames: int
    gt: int
    tp: int
    fp: int
    fn: int
    idsw: int
    mota: float | None
    motp_ft: float | None  # mean matched distance
    rmse_ft: float | None
    precision: float | None
    recall: float | None
    idtp: int
    idfp: int
    idfn: int
    idf1: float | None
    id_map: dict[str, str]  # truth id -> predicted id (global one-to-one)
    identity_accuracy: float | None
    team_accuracy: float | None
    team_n: int
    ball: BallReport = field(default_factory=BallReport)

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "ball"}
        d["id_map"] = dict(self.id_map)
        d["ball"] = self.ball.to_dict()
        return d


# --------------------------------------------------------------------------- alignment


def _align(
    pred_frames: Sequence[FrameState], truth_frames: Sequence[FrameState], cfg: TrackEvalConfig
) -> list[tuple[FrameState, FrameState]]:
    """(truth, pred) pairs in temporal truth order (ID switches depend on the order)."""
    truth_frames = sorted(truth_frames, key=lambda f: (f.t, f.frame_idx))
    if cfg.align_by == "frame_idx":
        by_idx = {f.frame_idx: f for f in pred_frames}
        return [(tf, by_idx[tf.frame_idx]) for tf in truth_frames if tf.frame_idx in by_idx]
    if cfg.align_by != "t":
        raise ValueError(f"unknown align_by {cfg.align_by!r}")
    if not pred_frames:
        return []
    order = np.argsort([f.t for f in pred_frames], kind="stable")
    ts = np.array([pred_frames[i].t for i in order], dtype=float)
    out = []
    for tf in truth_frames:
        k = int(np.searchsorted(ts, tf.t))
        near = [c for c in (k - 1, k) if 0 <= c < len(ts)]
        best = min(near, key=lambda c: abs(ts[c] - tf.t))
        if abs(ts[best] - tf.t) <= cfg.align_tol_s:
            out.append((tf, pred_frames[int(order[best])]))
    return out


def _pred_id(p: PlayerState, field_name: str) -> str:
    value = getattr(p, field_name, None)
    return str(value) if value is not None else p.player_id


# --------------------------------------------------------------------------- per frame


def _match_frame(
    truth: list[PlayerState],
    pred: list[PlayerState],
    pred_ids: list[str],
    radius: float,
    previous: dict[str, str],
) -> list[tuple[int, int, float]]:
    """(truth_index, pred_index, distance_ft) matches for one frame."""
    if not truth or not pred:
        return []
    tx = np.array([[p.x, p.y] for p in truth], dtype=float)
    px = np.array([[p.x, p.y] for p in pred], dtype=float)
    dist = np.linalg.norm(tx[:, None, :] - px[None, :, :], axis=2)
    gate = dist <= radius
    matches: list[tuple[int, int]] = []
    index_of = {}
    for j, pid in enumerate(pred_ids):
        index_of.setdefault(pid, j)
    for i, t_state in enumerate(truth):
        prev_id = previous.get(t_state.player_id)
        j = index_of.get(prev_id, -1) if prev_id is not None else -1
        if j >= 0 and gate[i, j] and all(j != mj for _, mj in matches):
            matches.append((i, j))
    used_t = {i for i, _ in matches}
    used_p = {j for _, j in matches}
    rest_t = [i for i in range(len(truth)) if i not in used_t]
    rest_p = [j for j in range(len(pred)) if j not in used_p]
    if rest_t and rest_p:
        sub = np.ix_(rest_t, rest_p)
        for a, b in gated_assignment(dist[sub], gate[sub]):
            matches.append((rest_t[a], rest_p[b]))
    return [(i, j, float(dist[i, j])) for i, j in matches]


def _ball_report(aligned: list[tuple[FrameState, FrameState]], hit_px: float) -> BallReport:
    rep = BallReport(n_frames=len(aligned))
    false_vis = 0
    interp = 0
    errs: list[float] = []

    def visible(b: BallState | None) -> bool:
        return b is not None and b.visible

    for tf, pf in aligned:
        tb, pb = tf.ball, pf.ball
        if visible(pb):
            rep.pred_visible += 1
            interp += int(bool(pb and pb.interpolated))
        if visible(tb):
            rep.truth_visible += 1
            if visible(pb) and tb and pb and tb.image_xy is not None and pb.image_xy is not None:
                err = float(np.hypot(pb.image_xy[0] - tb.image_xy[0], pb.image_xy[1] - tb.image_xy[1]))
                if np.isfinite(err):  # a NaN point would poison every px stat and the JSON
                    errs.append(err)
        elif visible(pb):
            false_vis += 1
    hits = sum(1 for tf, pf in aligned if visible(tf.ball) and visible(pf.ball))
    n_hidden = rep.n_frames - rep.truth_visible
    rep.detection_rate = hits / rep.truth_visible if rep.truth_visible else None
    rep.false_visible_rate = false_vis / n_hidden if n_hidden else None
    rep.interpolated_share = interp / rep.pred_visible if rep.pred_visible else None
    if errs:
        e = np.asarray(errs)
        rep.px_n = len(errs)
        rep.px_mean = float(e.mean())
        rep.px_median = float(np.median(e))
        rep.px_rmse = float(np.sqrt(np.mean(e**2)))
        rep.px_hit_rate = float(np.mean(e <= hit_px))
    return rep


def _id_mapping(co: dict[tuple[str, str], int]) -> dict[str, str]:
    """Global one-to-one truth->pred mapping maximising co-matched frames."""
    if not co:
        return {}
    t_ids = sorted({t for t, _ in co})
    p_ids = sorted({p for _, p in co})
    t_pos = {t: i for i, t in enumerate(t_ids)}
    p_pos = {p: j for j, p in enumerate(p_ids)}
    mat = np.zeros((len(t_ids), len(p_ids)))
    for (t, p), n in co.items():
        mat[t_pos[t], p_pos[p]] = n
    rows, cols = linear_sum_assignment(mat, maximize=True)
    return {t_ids[i]: p_ids[j] for i, j in zip(rows, cols, strict=True) if mat[i, j] > 0}


# --------------------------------------------------------------------------- entry point


def tracking_report(
    pred_frames: Sequence[FrameState],
    truth_frames: Sequence[FrameState],
    match_radius_ft: float = 3.0,
    config: TrackEvalConfig | None = None,
) -> TrackReport:
    """Evaluate predicted player/ball states against truth frames (court feet)."""
    cfg = config or TrackEvalConfig()
    aligned = _align(pred_frames, truth_frames, cfg)
    gt = tp = fp = fn = idsw = 0
    n_pred_dets = 0
    dists: list[float] = []
    identity_ok = team_ok = team_n = 0
    last_match: dict[str, str] = {}  # truth id -> pred id at its most recent match
    previous: dict[str, str] = {}  # matches of the previous evaluated frame
    co: dict[tuple[str, str], int] = {}

    for tf, pf in aligned:
        truth, pred = list(tf.players), list(pf.players)
        pred_ids = [_pred_id(p, cfg.pred_id_field) for p in pred]
        matches = _match_frame(truth, pred, pred_ids, match_radius_ft, previous if cfg.keep_previous_matches else {})
        gt += len(truth)
        n_pred_dets += len(pred)
        tp += len(matches)
        fp += len(pred) - len(matches)
        fn += len(truth) - len(matches)
        previous = {}
        for i, j, d in matches:
            t_id, p_id = truth[i].player_id, pred_ids[j]
            if t_id in last_match and last_match[t_id] != p_id:
                idsw += 1
            last_match[t_id] = p_id
            previous[t_id] = p_id
            co[(t_id, p_id)] = co.get((t_id, p_id), 0) + 1
            dists.append(d)
            identity_ok += int(pred[j].player_id == t_id)
            if truth[i].team is not None:
                team_n += 1
                team_ok += int(pred[j].team == truth[i].team)

    id_map = _id_mapping(co)
    idtp = sum(co.get((t, p), 0) for t, p in id_map.items())
    idfn, idfp = gt - idtp, n_pred_dets - idtp
    d = np.asarray(dists)
    report = TrackReport(
        match_radius_ft=match_radius_ft,
        n_frames=len(aligned),
        n_truth_frames=len(truth_frames),
        n_pred_frames=len(pred_frames),
        n_unaligned_truth_frames=len(truth_frames) - len(aligned),
        gt=gt,
        tp=tp,
        fp=fp,
        fn=fn,
        idsw=idsw,
        mota=1.0 - (fn + fp + idsw) / gt if gt else None,
        motp_ft=float(d.mean()) if len(d) else None,
        rmse_ft=float(np.sqrt(np.mean(d**2))) if len(d) else None,
        precision=tp / n_pred_dets if n_pred_dets else None,
        recall=tp / gt if gt else None,
        idtp=idtp,
        idfp=idfp,
        idfn=idfn,
        idf1=(2 * idtp / (2 * idtp + idfp + idfn)) if (idtp + idfp + idfn) else None,
        id_map=id_map,
        identity_accuracy=identity_ok / tp if tp else None,
        team_accuracy=team_ok / team_n if team_n else None,
        team_n=team_n,
        ball=_ball_report(aligned, cfg.ball_hit_px),
    )
    if report.n_unaligned_truth_frames:
        logger.info("tracking_report: %d truth frames had no predicted frame", report.n_unaligned_truth_frames)
    return report
