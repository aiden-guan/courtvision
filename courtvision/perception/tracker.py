"""ByteTrack for players and referees (numpy + scipy).

Follows ByteTrack (Zhang et al., ECCV 2022) as in the reference ``BYTETracker``:

1. High-score detections are matched to every track (tracked + lost) by IoU cost, optionally
   blended with an appearance cost.
2. Low-score detections are matched to the still-unmatched *tracked* tracks by IoU only.
3. Remaining high-score detections are matched to tentative tracks; a tentative track that misses
   one frame is dropped, and it is confirmed after ``min_hits`` matches.
4. Unmatched high-score detections above ``new_track_thresh`` start tentative tracks.
5. Lost confirmed tracks are kept for ``track_buffer`` frames.

The Kalman filter is the DeepSORT/ByteTrack constant-velocity model on (cx, cy, aspect, h) with
noise proportional to box height. ``frame_idx`` gaps are predicted with ``dt`` = the gap, so the
tracker can run at 15 Hz on a 30 Hz stream. Classes never match each other. Track ids come from a
counter starting at 1, so identical inputs give identical ids.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from courtvision.types import BBox, Detection, ObjClass, Track

__all__ = ["ByteTracker", "KalmanBoxFilter", "TrackerConfig", "iou_matrix"]

log = logging.getLogger(__name__)

_INF = 1e6


@dataclass
class TrackerConfig:
    """ByteTrack thresholds. Defaults are the ByteTrack reference values unless noted."""

    high_thresh: float = 0.5  # ByteTrack track_thresh; sourced (ByteTrack repo default)
    low_thresh: float = 0.1  # second-stage lower bound; sourced (ByteTrack)
    new_track_thresh: float = 0.6  # track_thresh + 0.1; sourced (ByteTrack)
    match_thresh: float = 0.8  # max cost (1 - IoU) in stage 1; sourced (ByteTrack match_thresh)
    second_match_thresh: float = 0.5  # max cost in stage 2; sourced (ByteTrack)
    unconfirmed_match_thresh: float = 0.7  # max cost for tentative tracks; sourced (ByteTrack)
    track_buffer: int = 30  # frames a lost track survives at 30 fps; sourced (ByteTrack default)
    frame_rate: float = 30.0  # buffer is scaled by frame_rate / 30 (ByteTrack); sourced
    min_hits: int = 2  # matches before a track is confirmed; tuned (trackers' minimum_consecutive_frames=2)
    instant_first_frame_activation: bool = True  # tracks born on the first frame are confirmed at once;
    #   sourced (ByteTrack activates on frame 1; trackers' BoT-SORT default)
    appearance_weight: float = 0.3  # cost = (1-w)(1-IoU) + w * cosine distance; tuned
    appearance_gate: float | None = 0.6  # cosine distance above this forbids a match; tuned
    embedding_momentum: float = 0.9  # EMA of track embeddings; sourced (BoT-SORT alpha = 0.9)
    emit_lost_frames: int = 5  # also output confirmed tracks lost <= this many frames (predicted box); tuned
    output_detection_bbox: bool = True  # matched tracks report the detection box (not the KF state); design
    std_weight_position: float = 1.0 / 20  # sourced (DeepSORT/ByteTrack KalmanFilter)
    std_weight_velocity: float = 1.0 / 160  # sourced (DeepSORT/ByteTrack KalmanFilter)


class KalmanBoxFilter:
    """Constant-velocity Kalman filter on (cx, cy, a, h) with height-proportional noise."""

    def __init__(self, std_weight_position: float = 1.0 / 20, std_weight_velocity: float = 1.0 / 160) -> None:
        self._wp = std_weight_position
        self._wv = std_weight_velocity
        self._H = np.hstack([np.eye(4), np.zeros((4, 4))])

    def initiate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = z[3]
        mean = np.r_[z, np.zeros(4)]
        std = [2 * self._wp * h, 2 * self._wp * h, 1e-2, 2 * self._wp * h,
               10 * self._wv * h, 10 * self._wv * h, 1e-5, 10 * self._wv * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        F = np.eye(8)
        F[:4, 4:] = dt * np.eye(4)
        h = mean[3]
        std = [self._wp * h, self._wp * h, 1e-2, self._wp * h, self._wv * h, self._wv * h, 1e-5, self._wv * h]
        Q = np.diag(np.square(std)) * dt
        return F @ mean, F @ cov @ F.T + Q

    def update(self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        R = np.diag(np.square([self._wp * h, self._wp * h, 1e-1, self._wp * h]))
        S = self._H @ cov @ self._H.T + R
        K = np.linalg.solve(S, (cov @ self._H.T).T).T
        mean = mean + K @ (z - self._H @ mean)
        cov = cov - K @ S @ K.T
        return mean, cov


@dataclass
class _TrackState:
    track_id: int  # 0 while tentative
    cls: ObjClass
    mean: np.ndarray
    cov: np.ndarray
    score: float
    bbox: BBox
    hits: int
    first_frame: int
    last_frame: int  # frame of the last match
    confirmed: bool = False
    embedding: np.ndarray | None = None
    jersey: str | None = None
    jersey_conf: float = 0.0

    def predicted_bbox(self) -> BBox:
        return _xyah_to_bbox(self.mean[:4])


class ByteTracker:
    """ByteTrack with deterministic ids. ``update(dets, frame_idx) -> list[Track]``."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._kf = KalmanBoxFilter(self.config.std_weight_position, self.config.std_weight_velocity)
        self._tracks: list[_TrackState] = []  # confirmed tracks (tracked or lost)
        self._tentative: list[_TrackState] = []
        self._next_id = 1
        self._frame_idx: int | None = None
        self._first_frame = True
        self._buffer = max(1, int(round(self.config.frame_rate / 30.0 * self.config.track_buffer)))

    def reset(self) -> None:
        """Forget every track (e.g. at a broadcast cut). Ids keep increasing."""
        self._tracks, self._tentative, self._frame_idx = [], [], None
        self._first_frame = True

    def remove(self, track_id: int) -> bool:
        """Delete a confirmed track now (e.g. a stale duplicate of a restarted player)."""
        before = len(self._tracks)
        self._tracks = [t for t in self._tracks if t.track_id != track_id]
        return len(self._tracks) != before

    def update(self, dets: list[Detection], frame_idx: int) -> list[Track]:
        """Associate this frame's PLAYER/REFEREE detections; returns live confirmed tracks."""
        cfg = self.config
        dt = 1.0 if self._frame_idx is None else float(max(1, frame_idx - self._frame_idx))
        self._frame_idx = frame_idx
        dets = [d for d in dets if d.cls in (ObjClass.PLAYER, ObjClass.REFEREE) and d.score >= cfg.low_thresh]
        for tr in self._tracks + self._tentative:
            if tr.last_frame < frame_idx - dt:  # lost tracks: freeze the height velocity (ByteTrack)
                tr.mean[7] = 0.0
            tr.mean, tr.cov = self._kf.predict(tr.mean, tr.cov, dt)

        high = [i for i, d in enumerate(dets) if d.score >= cfg.high_thresh]
        low = [i for i, d in enumerate(dets) if d.score < cfg.high_thresh]

        # Stage 1: high-score detections vs all confirmed tracks (tracked and lost).
        m1, un_tr, un_high = self._associate(self._tracks, [dets[i] for i in high], cfg.match_thresh, True)
        for ti, di in m1:
            self._apply(self._tracks[ti], dets[high[di]], frame_idx)
        # Stage 2: low-score detections vs tracks that were tracked last frame, IoU only.
        tracked = [ti for ti in un_tr if self._tracks[ti].last_frame >= frame_idx - dt]
        m2, _, _ = self._associate([self._tracks[i] for i in tracked], [dets[i] for i in low],
                                   cfg.second_match_thresh, False)
        for ti, di in m2:
            self._apply(self._tracks[tracked[ti]], dets[low[di]], frame_idx)
        # Stage 3: remaining high-score detections vs tentative tracks.
        rest = [high[i] for i in un_high]
        m3, _, un_rest = self._associate(self._tentative, [dets[i] for i in rest], cfg.unconfirmed_match_thresh, True)
        for ti, di in m3:
            self._apply(self._tentative[ti], dets[rest[di]], frame_idx)
        self._promote_and_prune(frame_idx)
        for di in un_rest:
            if dets[rest[di]].score >= cfg.new_track_thresh:
                self._start(dets[rest[di]], frame_idx)
        self._first_frame = False
        return self._outputs(frame_idx)

    # ------------------------------------------------------------------ association

    def _associate(
        self, tracks: list[_TrackState], dets: list[Detection], thresh: float, use_appearance: bool
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        cost = self._cost(tracks, dets, use_appearance)
        cost[cost > thresh] = _INF
        rows, cols = linear_sum_assignment(cost)
        matches = [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] < _INF]
        mt = {r for r, _ in matches}
        md = {c for _, c in matches}
        return matches, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(dets)) if j not in md]

    def _cost(self, tracks: list[_TrackState], dets: list[Detection], use_appearance: bool) -> np.ndarray:
        cfg = self.config
        tb = np.array([t.predicted_bbox().as_tuple() for t in tracks])
        db = np.array([d.bbox.as_tuple() for d in dets])
        cost = 1.0 - iou_matrix(tb, db)
        no_overlap = cost >= 1.0
        if use_appearance and cfg.appearance_weight > 0:
            app = self._appearance_cost(tracks, dets)
            have = ~np.isnan(app)
            blended = (1.0 - cfg.appearance_weight) * cost + cfg.appearance_weight * np.where(have, app, cost)
            if cfg.appearance_gate is not None:
                blended[have & (app > cfg.appearance_gate)] = _INF
            cost = blended
        cost[no_overlap] = _INF
        cls_t = np.array([t.cls.value for t in tracks])[:, None]
        cls_d = np.array([d.cls.value for d in dets])[None, :]
        cost[cls_t != cls_d] = _INF
        return cost

    @staticmethod
    def _appearance_cost(tracks: list[_TrackState], dets: list[Detection]) -> np.ndarray:
        out = np.full((len(tracks), len(dets)), np.nan)
        for i, t in enumerate(tracks):
            if t.embedding is None:
                continue
            nt = float(np.linalg.norm(t.embedding))
            if nt <= 0:
                continue
            for j, d in enumerate(dets):
                if d.embedding is None:
                    continue
                nd = float(np.linalg.norm(d.embedding))
                if nd > 0:
                    out[i, j] = float(np.clip(1.0 - float(np.dot(t.embedding, d.embedding)) / (nt * nd), 0.0, 1.0))
        return out

    # ------------------------------------------------------------------ lifecycle

    def _apply(self, tr: _TrackState, det: Detection, frame_idx: int) -> None:
        tr.mean, tr.cov = self._kf.update(tr.mean, tr.cov, _bbox_to_xyah(det.bbox))
        tr.bbox = det.bbox
        tr.score = det.score
        tr.hits += 1
        tr.last_frame = frame_idx
        if det.embedding is not None and float(np.linalg.norm(det.embedding)) > 0:
            e = np.asarray(det.embedding, dtype=np.float32)
            m = self.config.embedding_momentum
            tr.embedding = e.copy() if tr.embedding is None else m * tr.embedding + (1.0 - m) * e
        tr.jersey, tr.jersey_conf = det.jersey, det.jersey_conf  # this frame's read only (votes count once)

    def _start(self, det: Detection, frame_idx: int) -> None:
        mean, cov = self._kf.initiate(_bbox_to_xyah(det.bbox))
        emb = None if det.embedding is None else np.asarray(det.embedding, dtype=np.float32).copy()
        st = _TrackState(0, det.cls, mean, cov, det.score, det.bbox, 1, frame_idx, frame_idx,
                         embedding=emb, jersey=det.jersey, jersey_conf=det.jersey_conf)
        if self.config.min_hits <= 1 or (self._first_frame and self.config.instant_first_frame_activation):
            self._confirm(st)
            self._tracks.append(st)
        else:
            self._tentative.append(st)

    def _confirm(self, st: _TrackState) -> None:
        st.confirmed = True
        st.track_id = self._next_id
        self._next_id += 1

    def _promote_and_prune(self, frame_idx: int) -> None:
        keep: list[_TrackState] = []
        for st in self._tentative:
            if st.last_frame != frame_idx:
                continue  # a tentative track that misses a frame is dropped (ByteTrack)
            if st.hits >= self.config.min_hits:
                self._confirm(st)
                self._tracks.append(st)
            else:
                keep.append(st)
        self._tentative = keep
        before = len(self._tracks)
        self._tracks = [t for t in self._tracks if frame_idx - t.last_frame <= self._buffer]
        if len(self._tracks) != before:
            log.debug("frame %d: removed %d expired tracks", frame_idx, before - len(self._tracks))

    def _outputs(self, frame_idx: int) -> list[Track]:
        out: list[Track] = []
        for st in sorted(self._tracks, key=lambda s: s.track_id):
            lost = frame_idx - st.last_frame
            if lost > self.config.emit_lost_frames:
                continue
            use_det = lost == 0 and self.config.output_detection_bbox
            out.append(
                Track(
                    track_id=st.track_id,
                    cls=st.cls,
                    bbox=st.bbox if use_det else st.predicted_bbox(),
                    score=st.score,
                    hits=st.hits,
                    age=frame_idx - st.first_frame + 1,
                    time_since_update=lost,
                    embedding=None if st.embedding is None else st.embedding.copy(),
                    jersey=st.jersey if lost == 0 else None,
                    jersey_conf=st.jersey_conf if lost == 0 else 0.0,
                )
            )
        return out


# ---------------------------------------------------------------------------- box helpers


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (N, 4) and (M, 4) xyxy boxes."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def _bbox_to_xyah(b: BBox) -> np.ndarray:
    h = max(b.h, 1e-6)
    return np.array([(b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0, b.w / h, h])


def _xyah_to_bbox(z: np.ndarray) -> BBox:
    cx, cy, a, h = (float(v) for v in z[:4])
    h = max(h, 1e-6)
    w = max(a, 0.0) * h
    return BBox(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
