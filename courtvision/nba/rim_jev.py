"""Jev on rim crops: made or missed from the frames around a shot's arrival at the rim.

Research use: the frames go to TypeSafe's API, and Jev's image input (``jev-1.14.0``) is pre-production
(docs/research/jev_api.md §5). ``rim_centre`` finds the frame where the linked ball passes nearest the rim around a
time; ``rim_crops`` cuts ``MAKE_OFFSETS`` crops around that rim (``rim_window``); ``make_request`` asks, in one request,
made / missed / unclear in both option orders (averaged: jev_integration_design.md §1.5) and, for each crop, whether
the ball is in the net; ``decode_make`` reads the answer. ``RimJudge`` does all of it for one shot and keeps the
answers in a JSONL cache, so a re-run asks nothing it has asked before. Used by ``nba/makes.py`` and
``tools/nba_jev_probe.py``.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from courtvision.types import BBox

from .analyze import rim_boxes

__all__ = ["CROP_PX", "FATAL", "MAKE_OFFSETS", "MAKE_OPTIONS", "NEAR_RIM", "Frames", "RimJudge", "choice",
           "decode_make", "ensemble", "jpeg", "make_request", "pick", "reversed_options", "rim_centre", "rim_crops",
           "rim_window", "square_window"]

# frame offsets of the rim crops at 29.97 fps: 0.1 s apart around the ball's pass by the rim, wider at the ends
MAKE_OFFSETS = (-15, -9, -6, -3, 0, 3, 6, 9, 15, 24)
CROP_PX = 640  # side of a rim crop sent to Jev (280-1568 px allowed, never resized by the API)
NEAR_RIM = 3.0  # the ball passes "by the rim" within this many rim widths of its centre
FATAL = {"session_disabled", "bundle_disabled", "client_unavailable", "images_disabled", "bad_request"}

MAKE_OPTIONS = {
    "made": "The ball goes down through the rim into the net.",
    "missed": "The ball does not go through the rim: it bounces off the rim or the backboard, or misses the hoop.",
    "unclear": "The images do not show whether the ball went through the rim (the ball is hidden or out of view).",
}
MAKE_INSTRUCTIONS = ("`frames` are consecutive moments of one basketball shot, in time order, each cropped around "
                     "the hoop the ball was shot at. Did the ball go through the hoop?")


def choice(instructions: str, options: Mapping[str, str]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def reversed_options(options: Mapping[str, str]) -> dict[str, str]:
    return dict(reversed(list(options.items())))


def ensemble(*answers: Mapping[str, float]) -> dict[str, float]:
    """Mean probability per label over the same Choice asked in different option orders."""
    labels = sorted({k for a in answers for k in a})
    return {k: sum(a.get(k, 0.0) for a in answers) / len(answers) for k in labels}


def pick(probs: Mapping[str, float], abstain: str) -> str:
    """The most probable label; ``abstain`` when it is the most probable."""
    return max(probs, key=lambda k: (probs[k], k != abstain))


def square_window(cx: float, cy: float, side: float, w: int, h: int) -> tuple[int, int, int, int]:
    """An axis-aligned square of ``side`` px about (cx, cy), shifted (not shrunk) to lie inside a w x h image when
    it fits; (x0, y0, x1, y1)."""
    s = int(round(min(side, w, h)))
    x0 = int(round(min(max(cx - s / 2, 0), w - s)))
    y0 = int(round(min(max(cy - s / 2, 0), h - s)))
    return x0, y0, x0 + s, y0 + s


def rim_window(box: BBox, w: int, h: int) -> tuple[int, int, int, int]:
    """The rim crop: seven rim widths (at least 360 px) about a point just below the rim, so the ball's arrival
    above the rim and the net below it are both inside."""
    rw = box.x2 - box.x1
    return square_window((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2 + 0.6 * rw, max(360.0, 7.0 * rw), w, h)


def jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


class Frames:
    """Frames of a constant-rate video by number (OpenCV's seek is exact there; short skips are decoded through)."""

    def __init__(self, path: str | Path) -> None:
        self.cap = cv2.VideoCapture(str(path))
        self.pos = -1  # the frame the next read returns, minus one

    def get(self, f: int) -> np.ndarray | None:
        if not self.pos < f <= self.pos + 30:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            self.pos = f - 1
        while self.pos < f - 1 and self.cap.grab():
            self.pos += 1
        ok, img = self.cap.read()
        self.pos = f if ok else -1
        return img if ok else None


def rim_centre(t: float, camera: Any, court: Any, ball: Mapping[int, tuple[float, float]], search_s: float = 2.5,
               side: str | None = None) -> tuple[int, str, str] | None:
    """(centre frame, rim side, "ball" or "time") of a shot's crops around video time ``t``: the frame where the linked
    ball (frame -> image x, y) passes nearest the rim within ``search_s``, or ``t``'s frame when it does not come within
    ``NEAR_RIM`` rim widths. ``side``: the rim to use (default: the one in view most often). None: no rim in view."""
    w, h = camera.image_size
    f0 = camera.frame_at(t)
    span = round(search_s * camera.fps)
    rims: dict[int, dict[str, BBox]] = {}
    for f in range(f0 - span, f0 + span + 1):
        cam = camera.get(f)
        if cam is not None:
            rims[f] = rim_boxes(cam, court, (w, h))
    sides = Counter(s for boxes in rims.values() for s in boxes)
    if side is None:
        if not sides:
            return None
        side = max(sides, key=lambda s: (sides[s], s))
    elif not sides.get(side):
        return None
    best: tuple[float, int] | None = None
    for f, boxes in rims.items():
        p, box = ball.get(f), boxes.get(side)
        if p is None or box is None:
            continue
        d = float(np.hypot(p[0] - (box.x1 + box.x2) / 2, p[1] - (box.y1 + box.y2) / 2)) / max(box.x2 - box.x1, 1.0)
        if best is None or d < best[0]:
            best = (d, f)
    if best is not None and best[0] <= NEAR_RIM:
        return best[1], side, "ball"
    return f0, side, "time"


def rim_crops(frames: Frames, camera: Any, court: Any, centre: int, side: str) -> list[np.ndarray] | None:
    """The ``MAKE_OFFSETS`` crops around ``side``'s rim, each where the camera has the rim on that frame (else where it
    had it on the nearest frame of the set), ``CROP_PX`` square; None when a frame cannot be read or no rim is seen."""
    w, h = camera.image_size
    fs = [centre + round(o * camera.fps / 29.97) for o in MAKE_OFFSETS]
    boxes = []
    for f in fs:
        cam = camera.get(f)
        boxes.append(rim_boxes(cam, court, (w, h)).get(side) if cam is not None else None)
    if not any(b is not None for b in boxes):
        return None
    out = []
    for f, box in zip(fs, boxes, strict=True):
        if box is None:  # the rim box of the nearest frame of the set that has one
            box = next(b for g, b in sorted(zip(fs, boxes, strict=True), key=lambda gb: abs(gb[0] - f)) if b is not None)
        img = frames.get(f)
        if img is None:
            return None
        x0, y0, x1, y1 = rim_window(box, w, h)
        out.append(cv2.resize(img[y0:y1, x0:x1], (CROP_PX, CROP_PX), interpolation=cv2.INTER_CUBIC))
    return out


def make_request(crops: Sequence[bytes]) -> tuple[dict[str, Any], dict[str, Any]]:
    """State (the crops in order) and questions for one shot."""
    from typesafe_sdk import Image

    n = len(crops)
    state = {"frames": [{"moment": f"{i + 1} of {n}" + (" (earliest)" if i == 0 else " (latest)" if i == n - 1 else ""),
                         "image": Image(c)} for i, c in enumerate(crops)]}
    q: dict[str, Any] = {"outcome": choice(MAKE_INSTRUCTIONS, MAKE_OPTIONS),
                         "outcome_rev": choice(MAKE_INSTRUCTIONS, reversed_options(MAKE_OPTIONS))}
    for i in range(n):
        q[f"in_net_{i}"] = {"type": "noul",
                            "instructions": f"In `frames[{i}].image`, is the basketball inside the net, below the rim?"}
    q["hoop_visible"] = {"type": "noul",
                         "instructions": "Is a basketball hoop (rim and net) visible in `frames[0].image`?"}
    return state, q


def decode_make(resp: Any) -> dict[str, Any]:
    """The order-ensembled outcome and its probability, the in-net probability per crop (in order), the in-net read
    (made when any crop is >= 0.5) and whether a hoop is visible."""
    p = ensemble(*({k: float(v) for k, v in resp.choices[name].probabilities.items()}
                   for name in ("outcome", "outcome_rev")))
    in_net = [float(resp.nouls[k].noul) for k in sorted((k for k in resp.nouls if k.startswith("in_net_")),
                                                        key=lambda k: int(k.rsplit("_", 1)[1]))]
    out = pick(p, "unclear")
    return {"outcome": out, "p": round(p[out], 4), "probs": {k: round(v, 4) for k, v in p.items()},
            "in_net": [round(v, 3) for v in in_net], "in_net_pick": "made" if in_net and max(in_net) >= 0.5 else "missed",
            "hoop_visible": round(float(resp.nouls["hoop_visible"].noul), 3)}


class RimJudge:
    """Jev's read of one shot's rim crops, asked once: answers are kept in ``cache`` (JSONL, one per shot ``key``) and
    a later call with the same key reads it back. ``client``: a live ``JevClient`` with images enabled (None: answer
    from the cache only)."""

    def __init__(self, video: str | Path, camera: Any, court: Any, ball: Mapping[int, tuple[float, float]],
                 client: Any = None, cache: str | Path | None = None) -> None:
        self.camera, self.court, self.ball, self.client = camera, court, ball, client
        self.frames = Frames(video)
        self.cache_path = Path(cache) if cache is not None else None
        self.answers: dict[str, dict[str, Any]] = {}
        if self.cache_path is not None and self.cache_path.exists():
            for line in self.cache_path.open():
                row = json.loads(line)
                self.answers[row["key"]] = row
        self.bundle = None
        self.stopped: str | None = None  # why live asking stopped (a fatal request error), if it did
        if client is not None:
            from courtvision.decision.client import _retry

            # offline: generous HTTP timeout and retries; the model stays pinned to the B6 image model
            self.bundle = dataclasses.replace(client.bundle("B6"), http_timeout_s=60.0,
                                              retry=_retry(2, backoff_initial=0.5, backoff_max=5.0, timeout=180.0))

    def judge(self, key: str, t: float, side: str | None = None) -> dict[str, Any] | None:
        """Jev's decoded answer (``decode_make``, plus ``centred_on``) for the shot ``key`` resolving near video time
        ``t`` at the rim ``side``, or None (no rim in view, no frames, no client, or the request failed)."""
        if key in self.answers:
            return self.answers[key].get("jev")
        if self.client is None or self.bundle is None or self.stopped:
            return None
        found = rim_centre(t, self.camera, self.court, self.ball, side=side)
        if found is None:
            return None
        centre, side, how = found
        crops = rim_crops(self.frames, self.camera, self.court, centre, side)
        if crops is None:
            return None
        state, questions = make_request([jpeg(c) for c in crops])
        r = self.client.ask(self.bundle, state, questions, tag=key)
        if r.error in FATAL:
            self.stopped = r.error
        if not r.ok or r.response is None:
            return None
        row = {"key": key, "t": round(t, 3), "side": side, "centre": centre, "model": r.model,
               "input_tokens": r.input_tokens, "latency_ms": round(r.latency_ms),
               "jev": {**decode_make(r.response), "centred_on": how}}
        self.answers[key] = row
        if self.cache_path is not None:
            with self.cache_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        return row["jev"]
