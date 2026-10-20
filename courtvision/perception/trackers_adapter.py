"""Adapter over Roboflow ``trackers`` 2.6.1 (``ByteTrackTracker`` / ``BoTSORTTracker``).

Same contract as ``ByteTracker.update(dets, frame_idx) -> list[Track]``, plus the two inputs the
library needs (RESEARCH.md §7A.10, cv_pipeline.md §13.2; signatures verified by introspection):

* ``frame=``: BoT-SORT's camera-motion compensation silently does nothing without it (BGR,
  (H, W, 3)). ByteTrack ignores a frame and warns, so it is passed to BoT-SORT only.
* ``timestamp=``: the frame's absolute time in seconds; pass it when frames are skipped so the
  Kalman step and the lost-track budget use real elapsed time. ``frame_rate`` scales
  ``lost_track_buffer`` (counted in 30 fps frames): use 15.0 for 15 Hz updates.

The library is class-agnostic, so one tracker runs per ``ObjClass`` and its ids are remapped to a
single deterministic id space. Detections flow through ``supervision.Detections`` with their input
index in ``data["det_index"]`` (the library may reorder rows). ``tracker_id == -1`` rows (immature
tracks) are not emitted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from courtvision.types import BBox, Detection, ObjClass, Track

__all__ = ["TrackersAdapter", "TrackersAdapterConfig"]

log = logging.getLogger(__name__)


@dataclass
class TrackersAdapterConfig:
    """Library parameters; defaults are the library's own (trackers 2.6.1) except frame_rate."""

    kind: str = "bytetrack"  # "bytetrack" (fixed camera) | "botsort" (broadcast, CMC)
    frame_rate: float = 15.0  # detector update rate; RESEARCH §7A.10 (15 Hz full-frame detection)
    lost_track_buffer: int = 30  # in 30 fps frames; sourced (library default)
    track_activation_threshold: float = 0.7  # sourced (library default)
    minimum_consecutive_frames: int = 2  # sourced (library default)
    minimum_iou_threshold: float = 0.1  # ByteTrack; sourced (library default)
    high_conf_det_threshold: float = 0.6  # sourced (library default)
    enable_cmc: bool = True  # BoT-SORT; sourced (library default)
    cmc_method: str = "sparseOptFlow"  # sourced (library default)
    cmc_downscale: int = 2  # sourced (library default)
    use_timestamps: bool = True  # pass t as timestamp= (needed when frames are skipped); design
    embedding_momentum: float = 0.9  # EMA of our per-track embeddings; sourced (BoT-SORT alpha)


class TrackersAdapter:
    """``update(dets, frame_idx, frame=None, t=None) -> list[Track]`` over ``trackers``."""

    classes: tuple[ObjClass, ...] = (ObjClass.PLAYER, ObjClass.REFEREE)

    def __init__(self, config: TrackersAdapterConfig | None = None) -> None:
        self.config = config or TrackersAdapterConfig()
        if self.config.kind not in ("bytetrack", "botsort"):
            raise ValueError(f"unknown tracker kind {self.config.kind!r}")
        self._trackers: dict[ObjClass, Any] = {}
        self._ids: dict[tuple[ObjClass, int], int] = {}
        self._next_id = 1
        self._book: dict[int, dict[str, Any]] = {}

    def reset(self) -> None:
        """Reset every library tracker (e.g. at a cut); global ids keep increasing."""
        for tr in self._trackers.values():
            tr.reset()
        self._ids.clear()
        self._book.clear()

    def update(
        self, dets: list[Detection], frame_idx: int, frame: np.ndarray | None = None, t: float | None = None
    ) -> list[Track]:
        """Track PLAYER and REFEREE detections for one frame."""
        import supervision as sv  # lazy: extra `cv`

        out: list[Track] = []
        for cls in self.classes:
            group = [d for d in dets if d.cls is cls]
            sv_in = self._to_supervision(sv, group)
            kwargs: dict[str, Any] = {}
            if self.config.kind == "botsort" and frame is not None:
                kwargs["frame"] = frame
            if self.config.use_timestamps and t is not None:
                kwargs["timestamp"] = float(t)
            sv_out = self._tracker(cls).update(sv_in, **kwargs)
            out += self._from_supervision(sv_out, group, cls, frame_idx)
        return sorted(out, key=lambda tr: tr.track_id)

    # ------------------------------------------------------------------ internals

    def _tracker(self, cls: ObjClass) -> Any:
        tr = self._trackers.get(cls)
        if tr is None:
            import trackers  # lazy: extra `cv`

            cfg = self.config
            common = dict(
                lost_track_buffer=cfg.lost_track_buffer,
                frame_rate=cfg.frame_rate,
                track_activation_threshold=cfg.track_activation_threshold,
                minimum_consecutive_frames=cfg.minimum_consecutive_frames,
                high_conf_det_threshold=cfg.high_conf_det_threshold,
            )
            if cfg.kind == "bytetrack":
                tr = trackers.ByteTrackTracker(minimum_iou_threshold=cfg.minimum_iou_threshold, **common)
            else:
                tr = trackers.BoTSORTTracker(enable_cmc=cfg.enable_cmc, cmc_method=cfg.cmc_method,
                                             cmc_downscale=cfg.cmc_downscale, **common)
            self._trackers[cls] = tr
        return tr

    @staticmethod
    def _to_supervision(sv: Any, group: list[Detection]) -> Any:
        if not group:
            empty = sv.Detections.empty()
            empty.data["det_index"] = np.zeros(0, dtype=int)
            return empty
        return sv.Detections(
            xyxy=np.array([d.bbox.as_tuple() for d in group], dtype=np.float32),
            confidence=np.array([d.score for d in group], dtype=np.float32),
            class_id=np.zeros(len(group), dtype=int),
            data={"det_index": np.arange(len(group))},
        )

    def _from_supervision(self, sv_out: Any, group: list[Detection], cls: ObjClass, frame_idx: int) -> list[Track]:
        tids = sv_out.tracker_id
        if tids is None or len(sv_out) == 0:
            return []
        index = sv_out.data.get("det_index")
        out: list[Track] = []
        for row in range(len(sv_out)):
            lib_id = int(tids[row])
            if lib_id < 0:
                continue
            det = group[int(index[row])] if index is not None and int(index[row]) < len(group) else None
            gid = self._global_id(cls, lib_id)
            book = self._bookkeep(gid, det, frame_idx)
            x1, y1, x2, y2 = (float(v) for v in sv_out.xyxy[row])
            score = float(sv_out.confidence[row]) if sv_out.confidence is not None else (det.score if det else 1.0)
            out.append(Track(track_id=gid, cls=cls, bbox=BBox(x1, y1, x2, y2), score=score, hits=book["hits"],
                             age=frame_idx - book["first"] + 1, time_since_update=0,
                             embedding=None if book["emb"] is None else book["emb"].copy(),
                             jersey=book["jersey"], jersey_conf=book["jersey_conf"]))
        return out

    def _global_id(self, cls: ObjClass, lib_id: int) -> int:
        key = (cls, lib_id)
        gid = self._ids.get(key)
        if gid is None:
            gid = self._next_id
            self._next_id += 1
            self._ids[key] = gid
        return gid

    def _bookkeep(self, gid: int, det: Detection | None, frame_idx: int) -> dict[str, Any]:
        book = self._book.setdefault(
            gid, {"hits": 0, "first": frame_idx, "emb": None, "jersey": None, "jersey_conf": 0.0}
        )
        book["hits"] += 1
        book["jersey"], book["jersey_conf"] = None, 0.0  # this frame's read only (votes must count once)
        if det is not None:
            if det.embedding is not None:
                e = np.asarray(det.embedding, dtype=np.float32)
                m = self.config.embedding_momentum
                book["emb"] = e.copy() if book["emb"] is None else m * book["emb"] + (1.0 - m) * e
            book["jersey"], book["jersey_conf"] = det.jersey, det.jersey_conf
        return book
