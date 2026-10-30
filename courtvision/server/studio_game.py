"""The Studio's game: the live pipeline plus what the video workspace shows of it.

Before a court mapping it keeps tracking people and the ball in image space (boxes for the monitor, no invented
floor positions or statistics); it publishes a ``vision`` message for the workspace, and draws the monitor's
frame with its boxes (:meth:`StudioGame.monitor_jpeg`). Used live (``courtvision/server/studio.py``) and ahead of
time (``courtvision/server/preanalysis.py``).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import cv2
import numpy as np

from courtvision.perception.broadcast import ShotBoundaryDetector
from courtvision.perception.tracker import ByteTracker
from courtvision.pipeline import FrameItem, LiveGame
from courtvision.server.hub import status_msg
from courtvision.types import ObjClass


class StudioGame(LiveGame):
    """Run useful image-space detection while awaiting a real court calibration."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.preview_tracker = ByteTracker()
        self.preview_cuts = ShotBoundaryDetector()
        self.visual: tuple[np.ndarray, list[dict[str, Any]], float] | None = None
        self.vision: dict[str, Any] = {}
        self.failure: str | None = None
        self._vision_t = -1.0
        self._consecutive_failures = 0
        self._jpeg_cache: dict[tuple[float, bool, str], bytes] = {}

    def _perceive(self, image: np.ndarray, frame_idx: int, t: float) -> Any:
        started = time.perf_counter()
        result = super()._perceive(image, frame_idx, t)
        boxes: list[dict[str, Any]] = []
        gate = getattr(self.perception, "last_gate", None)
        if result is None:
            # No invented homography: until setup, display tracks in pixels and emit no stats.
            fd = self._detector.detect(image, frame_idx, t)
            if self.preview_cuts.update(image):
                self.preview_tracker = ByteTracker()
            people = [d for d in fd.of(ObjClass.PLAYER) if d.bbox.h >= image.shape[0] * .045]
            for tr in self.preview_tracker.update(people, frame_idx):
                if tr.time_since_update == 0:
                    boxes.append({"box": tr.bbox.as_tuple(), "label": f"Track {tr.track_id}",
                                  "team": None, "confidence": tr.score})
            for ball in fd.of(ObjClass.BALL):
                boxes.append({"box": ball.bbox.as_tuple(), "label": "Ball", "team": "ball",
                              "confidence": ball.score})
        else:
            frame, _ = result
            for p in frame.players:
                if p.bbox is not None:
                    boxes.append({"box": p.bbox.as_tuple(), "label": p.player_id,
                                  "team": p.team.value if p.team else None, "confidence": p.confidence})
            if frame.ball is not None and frame.ball.visible and frame.ball.image_xy is not None:
                u, v = frame.ball.image_xy
                boxes.append({"box": (u - 9, v - 9, u + 9, v + 9), "label": "Ball", "team": "ball"})
        ms = (time.perf_counter() - started) * 1000
        if result is None:
            self._lat.perception_ms.append(ms)
        self._consecutive_failures = 0
        self.visual = (image, boxes, t)
        self.vision = {"type": "vision", "t": t, "people": sum(b["team"] != "ball" for b in boxes),
                       "ball_visible": any(b["team"] == "ball" for b in boxes),
                       "inference_ms": round(ms, 1), "fps": round(1000 / max(ms, 1), 1),
                       "calibrated": result is not None, "gate": gate.reason if gate else None,
                       "tracks": [{"id": b["label"], "team": b["team"],
                                   "confidence": b.get("confidence")}
                                  for b in boxes if b["team"] != "ball"]}
        return result

    async def _await_calibration(self, t: float) -> None:
        # Let pixel tracks continue; the setup UI pauses and freezes the chosen calibration frame.
        self._t = t
        self._frames_done += 1
        if self.hub is not None and t - self._vision_t >= .25:
            await self.hub.publish(status_msg(t, **self.status()))

    async def process(self, item: FrameItem) -> None:
        before = self._errors
        await super().process(item)
        if self._errors > before:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                self.failure = "Analysis failed on consecutive frames. Reopen the video or check the detector setup."
                self.request_stop(drain=False)
        if self.hub is not None and item.t - self._vision_t >= .25:
            self._vision_t = item.t
            if self.vision:
                await self.hub.publish(self.vision)
            await self.hub.publish(status_msg(self.current_t, **self.status(), error=self.failure))

    async def monitor_jpeg(self, overlay: bool, focus: str = "") -> bytes | None:
        visual = self.visual
        if visual is None:
            return await self.controller.latest_frame_jpeg()
        image, boxes, t = visual
        key = (t, overlay, focus)
        if key in self._jpeg_cache:
            return self._jpeg_cache[key]

        def encode() -> bytes | None:
            scale = min(1., 1440 / image.shape[1])
            frame = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image.copy()
            if overlay:
                for b in boxes:
                    x1, y1, x2, y2 = (round(v * scale) for v in b["box"])
                    h, w = frame.shape[:2]
                    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
                    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    color = {"home": (237, 187, 120), "away": (117, 191, 246),
                             "ball": (105, 224, 248)}.get(b["team"], (151, 235, 209))
                    muted = bool(focus and focus != b["label"] and b["team"] != "ball")
                    if muted:
                        color = (106, 112, 110)
                    if b["team"] == "ball":
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        cv2.circle(frame, (cx, cy), max(9, (x2-x1)//2+4), color, 2, cv2.LINE_AA)
                        cv2.line(frame, (cx-4, cy), (cx+4, cy), color, 1, cv2.LINE_AA)
                        cv2.line(frame, (cx, cy-4), (cx, cy+4), color, 1, cv2.LINE_AA)
                        continue
                    # Corner brackets keep bodies readable. The bottom marker is the box base,
                    # not a measured joint or calibrated foot position.
                    length = max(3, min(20, (x2-x1)//3, (y2-y1)//4))
                    thickness = 3 if focus == b["label"] else 2
                    for x, dx in ((x1, 1), (x2, -1)):
                        for y, dy in ((y1, 1), (y2, -1)):
                            cv2.line(frame, (x, y), (x+dx*length, y), color, thickness, cv2.LINE_AA)
                            cv2.line(frame, (x, y), (x, y+dy*length), color, thickness, cv2.LINE_AA)
                    if muted:
                        continue
                    cv2.ellipse(frame, ((x1+x2)//2, y2), (max(6, (x2-x1)//3), 4), 0, 0, 360, color, 1, cv2.LINE_AA)
                    cv2.circle(frame, ((x1+x2)//2, y2), 2, color, -1, cv2.LINE_AA)
                    label = str(b["label"])
                    confidence = b.get("confidence")
                    if confidence is not None:
                        label += f" / {confidence:.0%}"
                    font_scale = .42 if w < 800 else .5
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                    xx = max(0, min(x1, w - tw - 12))
                    yy = min(h - 1, max(th + 10, y1 - 5))
                    cv2.rectangle(frame, (xx, yy-th-9), (min(w-1, xx+tw+12), yy), (24, 30, 27), -1)
                    cv2.line(frame, (xx, yy-th-9), (xx, yy), color, 2)
                    cv2.putText(frame, label, (xx+6, yy-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                color, 1, cv2.LINE_AA)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buf.tobytes() if ok else None

        encoded = await asyncio.to_thread(encode)
        self._jpeg_cache = {key: encoded} if encoded else {}
        return encoded
