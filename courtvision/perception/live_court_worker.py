"""Causal court tracking for a camera: one shared-memory frame, one answer, no video lookahead.

The parent may run the detector while this process fits the same frame's court, or submit one already-arrived
successor while that frame finishes. Results remain private until consumed in order; the shared frame cannot be
overwritten until its answer is drained. A skipped/replaced submission restores the complete prior tracker state.
Frame/cut checks and in-process fallback match the file worker. No camera is opened twice or frames dropped here.
"""

from __future__ import annotations

import contextlib
import copy
import multiprocessing as mp
import queue
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np

from .broadcast_court import BroadcastCourtTracker, CourtFix
from .court_worker import CourtTrackerProcess, frame_checksum


def _work_live(args: dict[str, Any], incoming: Any, outgoing: Any, stop: Any) -> None:
    import cv2

    from courtvision.geometry.court import court_for
    from .runtime import configure_live_thread

    memory = None
    try:
        cv2.setNumThreads(1)  # small images; avoid nested native thread pools alongside detection
        configure_live_thread()
        memory = SharedMemory(name=args["shared_name"])
        frame = np.ndarray(args["shape"], dtype=np.uint8, buffer=memory.buf)
        frame.setflags(write=False)
        tracker = BroadcastCourtTracker(court_for(args["league"]), args["image_size"], args["center"], args["config"])
        tracker.reset(args["params"])
        before = None
        while not stop.is_set():
            try:
                idx, cut, rollback = incoming.get(timeout=0.25)
            except queue.Empty:
                continue
            if rollback:
                assert before is not None
                tracker = before
            before = copy.deepcopy(tracker)  # bounded tracker state, no video/evidence arrays
            fix = tracker.update(frame, cut=cut)
            while not stop.is_set():
                try:
                    outgoing.put((idx, frame_checksum(frame), cut, fix), timeout=0.25)
                    break
                except queue.Full:
                    continue
    except Exception as exc:
        with contextlib.suppress(queue.Full):
            outgoing.put(("error", repr(exc)), timeout=0.25)
    finally:
        if memory is not None:
            memory.close()


class LiveCourtTrackerProcess(CourtTrackerProcess):
    """The file worker's interface, accepting the actual incoming frames instead of reading a file ahead."""

    parallel_detection = True

    def __init__(self, court: Any, image_size: tuple[int, int], center: np.ndarray, **kwargs: Any) -> None:
        super().__init__(court, image_size, center, video="", **kwargs)
        self._memory: SharedMemory | None = None
        self._shared: np.ndarray | None = None
        self._incoming: Any = None
        self._pending: tuple[np.ndarray, int, bool] | None = None

    @property
    def prefetch_available(self) -> bool:
        return self._proc is not None and self._local is None and self._pending is None

    def _submit(self, frame: np.ndarray, idx: int, cut: bool, rollback: bool = False) -> None:
        np.copyto(self._shared, frame)
        self._incoming.put((idx, cut, rollback), timeout=1.0)
        self._pending = (frame, idx, cut)

    def prefetch(self, frame: np.ndarray, frame_idx: int, *, cut: bool = False) -> bool:
        """Fit one already-arrived next frame while its predecessor finishes. Its camera stays private until
        update consumes this same image/index/cut; a mismatch restores the child's complete pre-fit state."""
        if not self.prefetch_available or frame.dtype != np.uint8 or frame.shape != self._shared.shape:
            return False
        try:
            self._submit(frame, frame_idx, cut)
            return True
        except (OSError, queue.Full, RuntimeError) as exc:
            self._go_local(f"could not prefetch live frame: {exc}")
            return False

    def _start(self, first: int) -> None:
        ctx = mp.get_context("spawn")
        shape = (self.image_size[1], self.image_size[0], 3)
        self._memory = SharedMemory(create=True, size=int(np.prod(shape)))
        self._shared = np.ndarray(shape, dtype=np.uint8, buffer=self._memory.buf)
        self._incoming = ctx.Queue(maxsize=1)
        self._queue = ctx.Queue(maxsize=1)
        self._stop = ctx.Event()
        args = {"shared_name": self._memory.name, "shape": shape, "league": self.court.league,
                "image_size": self.image_size, "center": self.center, "config": self.config, "params": self._params0}
        self._proc = ctx.Process(target=_work_live, args=(args, self._incoming, self._queue, self._stop),
                                 daemon=True, name="courtvision-live-court")
        self._proc.start()

    def update(self, frame_bgr: np.ndarray, cut: bool = False, people: Any = None,
               frame_idx: int | None = None) -> CourtFix:
        if self._local is not None:
            return super().update(frame_bgr, cut, people, frame_idx)
        expected_shape = (self.image_size[1], self.image_size[0], 3)
        if frame_idx is None or frame_bgr.dtype != np.uint8 or frame_bgr.shape != expected_shape or people is not None:
            self._go_local("live frame shape, frame number or people mask changed")
            return super().update(frame_bgr, cut, people, frame_idx)
        try:
            if self._proc is None:
                self._start(frame_idx)
            rollback = False
            if self._pending is not None:
                image, idx, pending_cut = self._pending
                if image is not frame_bgr or idx != frame_idx or bool(pending_cut) != bool(cut):
                    # The child may still read the shared image. Drain it before replacing pixels, and restore
                    # velocity, cut/loss counters and remembered views as well as the previous camera.
                    if self._next(idx) is None:
                        return super().update(frame_bgr, cut, people, frame_idx)
                    self._pending = None
                    rollback = True
            if self._pending is None:
                self._submit(frame_bgr, frame_idx, cut, rollback)
        except (OSError, queue.Full, RuntimeError) as exc:
            self._go_local(f"could not submit live frame: {exc}")
            return super().update(frame_bgr, cut, people, frame_idx)
        # The base validates the answer against the actual input frame and falls back on any mismatch.
        fix = super().update(frame_bgr, cut, people, frame_idx)
        self._pending = None
        return fix

    def _next(self, frame_idx: int) -> tuple[int, int, bool, CourtFix] | None:
        deadline = time.monotonic() + 30.0
        while True:
            try:
                got = self._queue.get(timeout=0.25)
            except queue.Empty:
                if self._proc is None or not self._proc.is_alive() or time.monotonic() >= deadline:
                    self._go_local("the live court worker stopped answering")
                    return None
                continue
            if got is None or got[0] == "error":
                self._go_local(f"the live court worker failed: {got}")
                return None
            return got

    def close(self) -> None:
        self._pending = None
        process = self._proc
        if process is not None and process.pid is None:
            if self._stop is not None:
                self._stop.set()
            self._proc = None  # start() failed before a child existed
        else:
            super().close()
        if process is not None and process.pid is not None:
            process.join(timeout=2.0)
        self._shared = None
        if self._memory is not None:
            self._memory.close()
            with contextlib.suppress(FileNotFoundError):
                self._memory.unlink()
            self._memory = None
        for channel in (self._incoming, self._queue):
            if channel is not None:
                channel.cancel_join_thread()
                channel.close()
        self._incoming, self._queue = None, None
