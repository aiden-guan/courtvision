"""Split-phase inference adapter: original OpenCV preprocessing and original Python perception.

Only the neural forward pass is performed by ONNX Runtime. Frame order, resizing, temporal
history, court tracking, box rectification, ByteTrack, teams and single-ball tracking use the
same modules as the local Studio. No JavaScript replacements of those algorithms.
"""
from collections import deque
import json

import cv2
import numpy as np

from courtvision.geometry.court import court_for
from courtvision.geometry.homography import Calibration
from courtvision.perception.ball import BallTracker
from courtvision.perception.broadcast import BroadcastPerception, BroadcastPerceptionConfig, ShotBoundaryDetector
from courtvision.perception.broadcast_court import BroadcastCourtTracker, estimate_center
from courtvision.perception.pipeline import PerceptionConfig, PerceptionPipeline
from courtvision.perception.student import OUTPUTS, decode, input_scale, preprocess_frames
from courtvision.perception.tracker import ByteTracker, TrackerConfig
from courtvision.types import BBox, Detection, FrameDetections, ObjClass


def array(value):
    return np.asarray(value.to_py() if hasattr(value, "to_py") else value)


def image_rgba(value, width, height):
    pixels = np.frombuffer(value, np.uint8) if isinstance(value, (bytes, bytearray, memoryview)) else array(value)
    return cv2.cvtColor(pixels.reshape(height, width, 4), cv2.COLOR_RGBA2BGR)


class _Decoded:
    parallel_court = False

    def __init__(self):
        self.current = None

    def detect(self, *_):
        return self.current

    def reset(self):
        pass  # BrowserPipeline.prepare already reset the temporal history before neural inference.


class _PreparedBroadcast(BroadcastPerception):
    def gate(self, image, frame_idx=None, before_wait=None):
        prepared = getattr(self, "prepared", None)
        if prepared is not None:
            self.prepared = None
            return prepared
        return super().gate(image, frame_idx, before_wait)


class BrowserPipeline:
    def __init__(self):
        self.court = court_for("nba")
        self.center = None
        self.initial = None
        self.pipeline = None
        self.detector = _Decoded()
        self.history = deque(maxlen=3)
        self.cuts = ShotBoundaryDetector()
        self.preview = ByteTracker(TrackerConfig(high_thresh=.3, low_thresh=.1, new_track_thresh=.35))
        self.ball = BallTracker()
        self.image = None
        self.frame_idx = -1
        self.t = 0.
        self.gate = None

    def calibrate(self, frames, width, height):
        images = [image_rgba(f, width, height) for f in frames]
        self.center, cameras = estimate_center(images, self.court)
        self.initial = next((cam for cam in cameras if cam is not None), None)
        return json.dumps({"calibrated": self.center is not None and self.initial is not None})

    def prepare(self, rgba, width, height, frame_idx, t):
        self.image = image_rgba(rgba, width, height)
        self.frame_idx, self.t = int(frame_idx), float(t)
        if self.pipeline is None and self.center is not None and self.initial is not None:
            auto = BroadcastCourtTracker(self.court, (width, height), self.center)
            auto.reset(self.initial)
            fix = auto.update(self.image, cut=True)
            hom = fix.homography((width, height))
            if hom is not None:
                calibration = Calibration(hom, (width, height))
                config = PerceptionConfig(tracker={"high_thresh": .3, "low_thresh": .1, "new_track_thresh": .35},
                                          team={"refit_at": (150, 600, 1500, 3000)})
                inner = PerceptionPipeline(self.detector, calibration, config=config)
                self.pipeline = _PreparedBroadcast(inner, calibration, config=BroadcastPerceptionConfig(
                    live_requires_scorebug=False, rectify_boxes=True), court_tracker=auto)
        if self.pipeline is not None:
            self.gate = self.pipeline.gate(self.image, self.frame_idx)
            self.pipeline.prepared = self.gate
            if self.gate.cut or not self.gate.live:
                self.history.clear()
        elif self.cuts.update(self.image):
            self.history.clear(); self.preview.reset(); self.ball.reset()
        if self.gate is not None and not self.gate.live:
            current = preprocess_frames([self.image])
            return np.ascontiguousarray(np.concatenate([current, current, current]))
        self.history.append(preprocess_frames([self.image]))
        frames = list(self.history)
        return np.ascontiguousarray(np.concatenate([frames[max(0, len(frames)-1-back)] for back in (2, 1, 0)]))

    def finish(self, outputs):
        raw = outputs.to_py() if hasattr(outputs, "to_py") else outputs
        out = {name: array(raw[name]).reshape((1 if name.endswith("hm") else 2, 136, 240)) for name in OUTPUTS}
        for name in ("ball_hm", "person_hm"):
            out[name] = 1 / (1 + np.exp(-out[name]))
        height, width = self.image.shape[:2]
        balls, people = decode(out, input_scale(width, height), ball_thresh=.08, person_thresh=.3)
        radius = 12 * height / 1080
        detections = [Detection(BBox(*box[:4]), ObjClass.PLAYER, box[4]) for box in people]
        detections += [Detection(BBox(x-radius, y-radius, x+radius, y+radius), ObjClass.BALL, score) for x,y,score in balls]
        self.detector.current = FrameDetections(self.frame_idx, self.t, (width, height), detections)
        boxes = []
        state = None
        if self.pipeline is not None:
            state = self.pipeline.process(self.image, self.frame_idx, self.t)
            for player in state.players:
                if player.bbox is not None:
                    boxes.append({"id": player.player_id, "class": "person", "bbox": list(player.bbox.as_tuple()),
                                  "confidence": player.confidence, "team": player.team.value if player.team else None})
            ball = state.ball
        else:
            observed = [d for d in detections if d.cls is ObjClass.PLAYER and d.bbox.h >= height * .045]
            tracks = self.preview.update(observed, self.frame_idx)
            for track in tracks:
                if track.time_since_update == 0:
                    boxes.append({"id": f"Track {track.track_id}", "class": "person", "bbox": list(track.bbox.as_tuple()),
                                  "confidence": track.score, "team": None})
            ball = self.ball.update([d for d in detections if d.cls is ObjClass.BALL], self.frame_idx, self.t,
                                    people=[d.bbox for d in observed])
        if ball is not None and ball.visible and ball.image_xy is not None:
            x,y = ball.image_xy
            boxes.append({"id": "Ball", "class": "sports ball", "bbox": [x-9,y-9,x+9,y+9],
                          "confidence": ball.confidence, "interpolated": ball.interpolated})
        for box in boxes:
            x1,y1,x2,y2 = box["bbox"]
            box["bbox"] = [float(v) for v in (x1,y1,x2-x1,y2-y1)]
            box["confidence"] = float(box["confidence"])
        return json.dumps({"t": self.t, "frame_idx": self.frame_idx, "detections": boxes,
                           "calibrated": self.pipeline is not None,
                           "gate": self.pipeline.last_gate.reason if self.pipeline and self.pipeline.last_gate else None})
