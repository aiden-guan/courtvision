"""TrackersAdapter over trackers 2.6.1 on synthetic detections at 15 Hz with gaps."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("trackers")
pytest.importorskip("supervision")

from courtvision.perception.trackers_adapter import TrackersAdapter, TrackersAdapterConfig  # noqa: E402
from courtvision.types import BBox, Detection, ObjClass  # noqa: E402

HZ = 15.0


def targets(t: float) -> list[tuple[float, float]]:
    """Three players (cx, cy) moving at different velocities (px/s)."""
    return [(200 + 90 * t, 300), (700 - 60 * t, 420 + 10 * t), (1000, 250 + 30 * t)]


def dets_at(t: float, rng: np.random.Generator, drop: set[int] = frozenset(), cls: ObjClass = ObjClass.PLAYER) -> list[Detection]:
    out = []
    for i, (cx, cy) in enumerate(targets(t)):
        if i in drop:
            continue
        cx, cy = cx + rng.normal(0, 0.5), cy + rng.normal(0, 0.5)
        out.append(Detection(bbox=BBox(cx - 12, cy - 35, cx + 12, cy + 35), cls=cls, score=0.9,
                             embedding=np.eye(3, dtype=np.float32)[i], jersey=str(i) if i == 0 else None, jersey_conf=0.8))
    return out


def schedule() -> list[int]:
    """15 Hz update indices with 1-3 update gaps (skipped frames)."""
    skips = {12, 13, 30, 31, 32, 50}
    return [k for k in range(80) if k not in skips]


def run(adapter: TrackersAdapter, frames: bool) -> dict[int, set[int]]:
    rng = np.random.default_rng(0)
    owner: dict[int, set[int]] = {}
    # Static textured background: BoT-SORT's CMC needs scene features; on a flat background its
    # sparse optical flow locks onto the moving players and "compensates" a camera that never moved.
    img = np.random.default_rng(9).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    for k in schedule():
        t = k / HZ
        frame = None
        if frames:
            frame = img.copy()
            for cx, cy in targets(t):
                frame[int(cy) - 35:int(cy) + 35, int(cx) - 12:int(cx) + 12] = (40, 160, 40)
        for tr in adapter.update(dets_at(t, rng), frame_idx=2 * k, frame=frame, t=t):
            centres = [np.hypot(tr.bbox.center[0] - cx, tr.bbox.center[1] - cy) for cx, cy in targets(t)]
            owner.setdefault(tr.track_id, set()).add(int(np.argmin(centres)))
    return owner


def test_bytetrack_ids_persist_at_15hz_with_timestamp_gaps() -> None:
    adapter = TrackersAdapter(TrackersAdapterConfig(kind="bytetrack", frame_rate=HZ))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # ByteTrack must not be handed a frame
        owner = run(adapter, frames=True)
    assert len(owner) == 3 and all(len(v) == 1 for v in owner.values())
    assert sorted(next(iter(v)) for v in owner.values()) == [0, 1, 2]


def test_botsort_with_frames_and_cmc_keeps_ids() -> None:
    adapter = TrackersAdapter(TrackersAdapterConfig(kind="botsort", frame_rate=HZ))
    owner = run(adapter, frames=True)
    assert len(owner) == 3 and all(len(v) == 1 for v in owner.values())


def test_short_dropout_keeps_id_and_bookkeeping() -> None:
    rng = np.random.default_rng(1)
    adapter = TrackersAdapter(TrackersAdapterConfig(frame_rate=HZ, lost_track_buffer=30))
    ids_before, ids_after = set(), set()
    for k in range(40):
        t = k / HZ
        drop = {0} if 15 <= k < 20 else set()  # player 0 occluded for 5 updates (1/3 s)
        out = adapter.update(dets_at(t, rng, drop), 2 * k, t=t)
        p0 = [tr for tr in out if abs(tr.bbox.center[1] - 300) < 20 and tr.bbox.center[0] < 600]
        if 10 <= k < 15:
            ids_before |= {tr.track_id for tr in p0}
        if k >= 25:
            ids_after |= {tr.track_id for tr in p0}
            assert p0[0].jersey == "0" and p0[0].embedding is not None and p0[0].embedding.argmax() == 0
            assert p0[0].age > p0[0].hits  # age counts frame_idx (30 Hz index), hits count updates
    assert ids_before == ids_after and len(ids_after) == 1


def test_classes_get_separate_trackers_and_global_ids() -> None:
    rng = np.random.default_rng(2)
    adapter = TrackersAdapter(TrackersAdapterConfig(frame_rate=HZ))
    for k in range(6):
        t = k / HZ
        out = adapter.update(dets_at(t, rng) + dets_at(t, rng, cls=ObjClass.REFEREE)[:1], 2 * k, t=t)
    ids = [tr.track_id for tr in out]
    assert len(ids) == len(set(ids)) == 4
    assert sum(tr.cls is ObjClass.REFEREE for tr in out) == 1
    adapter.reset()
    assert adapter.update([], 100, t=100 / HZ) == []
    with pytest.raises(ValueError):
        TrackersAdapter(TrackersAdapterConfig(kind="sort"))


def test_jersey_read_is_not_repeated_when_the_detection_link_is_lost() -> None:
    """Regression: when an output row could not be traced back to its detection, the previous
    frame's jersey read was emitted again, so the identity layer counted one read many times."""
    import supervision as sv

    class LinkLosingTracker:
        """Library stand-in: tracks one box, but drops ``data`` after the first frame."""

        def __init__(self) -> None:
            self.calls = 0

        def update(self, detections, **_kwargs):
            self.calls += 1
            out = sv.Detections(xyxy=detections.xyxy.copy(), confidence=detections.confidence.copy(),
                                class_id=detections.class_id.copy(), tracker_id=np.zeros(len(detections), int),
                                data=dict(detections.data) if self.calls == 1 else {})
            return out

    adapter = TrackersAdapter(TrackersAdapterConfig(frame_rate=HZ))
    adapter._trackers[ObjClass.PLAYER] = LinkLosingTracker()
    adapter._trackers[ObjClass.REFEREE] = LinkLosingTracker()
    det = Detection(bbox=BBox(10, 10, 30, 70), cls=ObjClass.PLAYER, score=0.9, jersey="23", jersey_conf=0.8)
    (first,) = adapter.update([det], 0, t=0.0)
    assert first.jersey == "23"
    (second,) = adapter.update([det], 2, t=2 / HZ)
    assert second.jersey is None and second.jersey_conf == 0.0
