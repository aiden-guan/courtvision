"""Live decoding stays bounded, preserves order, and closes a blocked source on stop or failure."""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from courtvision.pipeline import FrameItem, LiveGame, VideoFileSource


class Source(VideoFileSource):
    def __init__(self, blocked=False, fail_at=None):
        self.produced = 0
        self.closed = False
        self.blocked = blocked
        self.fail_at = fail_at

    async def _iterate(self):
        try:
            for idx in range(8):
                await asyncio.sleep(0)  # chained tasks must not concurrently advance this generator
                if self.blocked and idx == 1:
                    await asyncio.Event().wait()
                if idx == self.fail_at:
                    raise ValueError("source failure")
                self.produced += 1
                yield FrameItem(idx, idx / 30, np.zeros((2, 2, 3), np.uint8), None)
        finally:
            self.closed = True


class Game:
    def __init__(self, blocked=False, depth=3, fail_at=None):
        self.source = Source(blocked, fail_at)
        self.depth = depth
        self.config = SimpleNamespace(pipeline=SimpleNamespace(parallel_live=True, live_lookahead_frames=depth))
        self.realtime, self._stop_requested = True, False
        self._ahead_items = ()
        self.frames = []
        self.finished = False

    async def start(self):
        pass

    async def process(self, item):
        await asyncio.sleep(0)
        assert self.source.produced <= len(self.frames) + 1 + self.depth
        self.frames.append(item.frame_idx)
        if self.source.blocked:
            self._stop_requested = True

    async def finish(self):
        self.finished = True


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [1, 3])
async def test_lookahead_preserves_order_and_is_bounded(depth):
    game = Game(depth=depth)
    await LiveGame.run(game)
    assert game.frames == list(range(8))
    assert game.source.closed and game.finished and not game._ahead_items


@pytest.mark.asyncio
async def test_stop_cancels_the_pending_read_before_closing_the_source():
    game = Game(blocked=True)
    await asyncio.wait_for(LiveGame.run(game), timeout=5)
    assert game.frames == [0]
    assert game.source.closed and game.finished and not game._ahead_items


@pytest.mark.asyncio
async def test_source_failure_cancels_queued_reads_and_finishes():
    game = Game(fail_at=2)
    with pytest.raises(ValueError, match="source failure"):
        await LiveGame.run(game)
    assert game.frames == [0, 1]
    assert game.source.closed and game.finished and not game._ahead_items


def test_prefetch_stops_when_the_ocr_wait_finishes():
    budget = Future()
    ready = tuple(Future() for _ in range(3))
    for i, future in enumerate(ready):
        future.set_result(FrameItem(i, i / 30, np.zeros((2, 2, 3), np.uint8), None))
    seen = []

    def prefetch(image, idx, t):
        seen.append(idx)
        budget.set_result(None)
        return True

    game = SimpleNamespace(_detector=SimpleNamespace(prefetch=prefetch), _ahead_items=ready)
    LiveGame._prefetch_detector_input(game, budget)
    assert seen == [0]


@pytest.mark.parametrize("first", ["waiting", "error", "failed_inference"])
def test_prefetch_cannot_cross_a_missing_frame(first):
    ready = tuple(Future() for _ in range(3))
    for i, future in enumerate(ready):
        if i == 0 and first == "waiting":
            continue
        if i == 0 and first == "error":
            future.set_exception(ValueError("source failure"))
        else:
            future.set_result(FrameItem(i, i / 30, np.zeros((2, 2, 3), np.uint8), None))
    seen = []

    def prefetch(image, idx, t):
        seen.append(idx)
        return False

    game = SimpleNamespace(_detector=SimpleNamespace(prefetch=prefetch), _ahead_items=ready)
    LiveGame._prefetch_detector_input(game, Future())
    assert seen == ([0] if first == "failed_inference" else [])
