"""The broadcast court tracker in a process of its own, working ahead of the rest of the analysis.

The court tracker (``broadcast_court.BroadcastCourtTracker``) is about half of the Studio's time per frame, and it
needs nothing but the frames: not the detections, not the players. ``CourtTrackerProcess`` runs it in a worker process
that opens the same video file, reads the same frames in the same order (the same seek and the same stride as the
Studio's ``VideoFileSource`` / ``preanalysis.StridedVideoSource``: decoding is a millisecond or two a frame) and tracks
up to ``lookahead`` frames ahead, so the court is ready by the time the detector has seen a frame.

It is a drop-in for the tracker inside ``BroadcastPerception``: ``update(frame, cut, frame_idx)`` returns the worker's
fix for that frame. The worker runs the same tracker on the same frames with its own cut detector (the same
``ShotBoundaryDetector`` fed from the same frame), so the fixes are those of the tracker in process. Each answer
carries a checksum of its frame and the worker's cut flag; when either differs from the frame and cut flag the
analysis has (or when the worker fails, or the operator recalibrates), the tracker falls back to running in process
from the last fix, and says so in the log.

Used for analyses ahead (``server/preanalysis.py``), where nothing moves the video under the analysis; Live keeps the
tracker in process.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
from collections.abc import Callable
from typing import Any

import numpy as np

from .broadcast_court import BroadcastCourtTracker, CourtFix

log = logging.getLogger(__name__)

__all__ = ["CourtTrackerProcess", "frame_checksum"]


def frame_checksum(img: np.ndarray) -> int:
    """A cheap fingerprint of a frame: the sum of a sparse grid of its pixels."""
    return int(img[::37, ::41].astype(np.int64).sum())


def _work(args: dict[str, Any], out: Any, stop: Any) -> None:
    """The worker: read the video from ``start_s`` every ``stride``-th frame, skip to ``first`` and track from there."""
    import cv2

    from courtvision.geometry.court import court_for

    from .broadcast import ShotBoundaryDetector

    try:
        cap = cv2.VideoCapture(args["video"])
        if args["start_s"] > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, args["start_s"] * 1000.0)
        tracker = BroadcastCourtTracker(court_for(args["league"]), args["image_size"], args["center"], args["config"])
        tracker.reset(args["params"])
        cuts = ShotBoundaryDetector(args["cut_threshold"])
        idx = -1
        while not stop.is_set():
            for _ in range(args["stride"] - 1):
                if not cap.grab():
                    break
            ok, img = cap.read()
            if not ok or img is None:
                out.put(None)
                return
            idx += 1
            if idx < args["first"]:
                continue
            cut = cuts.update(img)
            fix = tracker.update(img, cut=cut)
            while not stop.is_set():
                try:
                    out.put((idx, frame_checksum(img), cut, fix), timeout=0.5)
                    break
                except queue.Full:
                    continue
    except Exception as exc:  # reported to the analysis, which then tracks in process
        out.put(("error", repr(exc)))


class CourtTrackerProcess:
    """``BroadcastCourtTracker``'s interface (``update``, ``params``, ``center``, ``reset``, ``reset_to_homography``)
    with the tracking done ahead in a worker process (see the module docstring). ``video``, ``start_s`` and
    ``stride``: how the analysis reads the video; ``cut_threshold``: its ``BroadcastPerceptionConfig``'s."""

    wants_frame_idx = True  # BroadcastPerception passes each frame's number to ``update``

    def __init__(self, court: Any, image_size: tuple[int, int], center: np.ndarray, *, video: str, start_s: float = 0.0,
                 stride: int = 1, cut_threshold: float = 0.35, config: Any = None, lookahead: int = 48,
                 make_local: Callable[[], BroadcastCourtTracker] | None = None) -> None:
        self.court = court
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.center = np.asarray(center, float)
        self.config = config
        self._video, self._start_s, self._stride = str(video), float(start_s), max(1, int(stride))
        self._cut_threshold, self._lookahead = float(cut_threshold), int(lookahead)
        self._make_local = make_local or (lambda: BroadcastCourtTracker(court, image_size, center, config))
        self._params0: Any = None
        self._last: CourtFix | None = None
        self._proc: Any = None
        self._queue: Any = None
        self._stop: Any = None
        self._local: BroadcastCourtTracker | None = None
        self.fallback_reason: str | None = None
        self.status = "lost"

    # ------------------------------------------------------------------ the tracker's interface

    @property
    def params(self) -> Any:
        if self._local is not None:
            return self._local.params
        return self._last.params if self._last is not None else self._params0

    def reset(self, params: Any = None) -> None:
        """Start from ``params`` (before the first frame); later, the tracker continues in process from them."""
        if self._proc is None and self._local is None:
            self._params0 = params
            return
        self._go_local("reset during the analysis")
        assert self._local is not None
        self._local.reset(params)

    def reset_to_homography(self, hom: Any, size: tuple[int, int] | None = None) -> Any:
        self._go_local("an operator calibration")
        assert self._local is not None
        return self._local.reset_to_homography(hom, size)

    def update(self, frame_bgr: np.ndarray, cut: bool = False, people: Any = None, frame_idx: int | None = None) -> CourtFix:
        if self._local is None and frame_idx is None:
            self._go_local("no frame number")
        if self._local is not None:
            fix = self._local.update(frame_bgr, cut=cut, people=people)
            self._last, self.status = fix, self._local.status
            return fix
        assert frame_idx is not None
        if self._proc is None:
            self._start(frame_idx)
        got = self._next(frame_idx)
        if got is None:
            return self.update(frame_bgr, cut, people, None)  # gone local
        idx, check, wcut, fix = got
        if idx != frame_idx or check != frame_checksum(frame_bgr) or bool(wcut) != bool(cut):
            self._go_local(f"the worker's frame {idx} is not the analysis's frame {frame_idx}")
            return self.update(frame_bgr, cut, people, None)
        self._last, self.status = fix, fix.status
        return fix

    def close(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._proc is not None:
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
        self._proc = None

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ the worker

    def _start(self, first: int) -> None:
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue(maxsize=self._lookahead)
        self._stop = ctx.Event()
        args = {"video": self._video, "start_s": self._start_s, "stride": self._stride, "first": first,
                "league": self.court.league, "image_size": self.image_size, "center": self.center,
                "config": self.config, "params": self._params0, "cut_threshold": self._cut_threshold}
        self._proc = ctx.Process(target=_work, args=(args, self._queue, self._stop), daemon=True,
                                 name="courtvision-court")
        self._proc.start()

    def _next(self, frame_idx: int) -> tuple[int, int, bool, CourtFix] | None:
        while True:
            try:
                got = self._queue.get(timeout=60.0)
            except queue.Empty:
                self._go_local("the worker stopped answering")
                return None
            if got is None:
                self._go_local("the worker reached the end of the video first")
                return None
            if got[0] == "error":
                self._go_local(f"the worker failed: {got[1]}")
                return None
            if got[0] < frame_idx:
                continue  # a frame the analysis skipped (it never asked)
            return got

    def _go_local(self, reason: str) -> None:
        if self._local is not None:
            return
        self.fallback_reason = reason
        if self._proc is not None:
            log.warning("court tracking continues in process: %s", reason)
        self.close()
        self._local = self._make_local()
        last = self._last.params if self._last is not None else self._params0
        self._local.reset(last)
