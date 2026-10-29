"""Basketball student detector: a small multi-frame CenterNet trained on broadcast footage.

A frame-by-frame COCO detector misses the blurred ball of a pass or a shot and is slow at the resolution a
20-35 px ball needs. ``BallPersonNet`` sees the current frame and the two before it (motion makes a blurred ball
stand out; only past frames, so it runs live) at 960x544 and predicts, at stride 4:

* a ball centre heatmap and a sub-cell offset;
* a person centre heatmap, box size (log w, log h in output cells) and offset (CenterNet, Zhou et al. 2019).

The backbone is torchvision's ResNet-18 (ImageNet weights, first convolution widened to 3 frames), with an FPN
decoder to stride 4. ``StudentDetector`` is a ``Detector``: it keeps the last frames, runs the network and
returns PLAYER and BALL detections (ball boxes are drawn from the predicted point with the radius
``ball_radius_px``). Training: ``training/train_student.py`` on teacher labels (``training/ball_labels.py``).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from courtvision.types import BBox, Detection, FrameDetections, ObjClass

__all__ = ["INPUT_HW", "BallPersonNet", "StudentConfig", "StudentDetector", "decode", "preprocess_frames"]

log = logging.getLogger(__name__)

INPUT_HW = (544, 960)  # network input (h, w); 1080p frames are scaled by 0.5 and padded by 4 rows
STRIDE = 4
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _torch() -> Any:
    import torch  # lazy: extra `cv`

    return torch


def build_net(frames: int = 3, pretrained: bool = True) -> Any:
    """``BallPersonNet`` (constructed lazily so importing this module needs no torch)."""
    torch = _torch()
    nn = torch.nn
    import torchvision

    class _BallPersonNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            r = torchvision.models.resnet18(weights=weights)
            conv1 = nn.Conv2d(3 * frames, 64, 7, 2, 3, bias=False)
            with torch.no_grad():
                conv1.weight.copy_(r.conv1.weight.repeat(1, frames, 1, 1) / frames)
            self.stem = nn.Sequential(conv1, r.bn1, r.relu)  # stride 2
            self.pool = r.maxpool
            self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
            ch = 128
            self.lat = nn.ModuleList([nn.Conv2d(c, ch, 1) for c in (64, 128, 256, 512)])
            self.smooth = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))

            def head(n: int, bias: float = 0.0) -> Any:
                h = nn.Sequential(nn.Conv2d(ch, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, n, 1))
                nn.init.constant_(h[-1].bias, bias)
                return h

            self.ball_hm = head(1, -4.6)  # prior 0.01 (CenterNet)
            self.ball_off = head(2)
            self.person_hm = head(1, -4.6)
            self.person_wh = head(2)
            self.person_off = head(2)

        def forward(self, x: Any) -> dict[str, Any]:
            c1 = self.l1(self.pool(self.stem(x)))  # stride 4
            c2 = self.l2(c1)
            c3 = self.l3(c2)
            c4 = self.l4(c3)
            p = self.lat[3](c4)
            for lat, c in ((self.lat[2], c3), (self.lat[1], c2), (self.lat[0], c1)):
                p = lat(c) + nn.functional.interpolate(p, size=c.shape[-2:], mode="nearest")
            p = self.smooth(p)
            return {"ball_hm": self.ball_hm(p), "ball_off": self.ball_off(p), "person_hm": self.person_hm(p),
                    "person_wh": self.person_wh(p), "person_off": self.person_off(p)}

    return _BallPersonNet()


BallPersonNet = build_net  # the factory doubles as the public name


def preprocess_frames(frames_bgr: list[np.ndarray]) -> np.ndarray:
    """Frames (oldest first) -> (3 * n, H, W) float32 network input: scaled to 960 px wide, padded to 544 rows,
    RGB, ImageNet-normalised."""
    h_in, w_in = INPUT_HW
    chans = []
    for f in frames_bgr:
        s = w_in / f.shape[1]
        img = cv2.resize(f, (w_in, int(round(f.shape[0] * s))), interpolation=cv2.INTER_AREA) if s != 1.0 else f
        if img.shape[0] < h_in:
            img = cv2.copyMakeBorder(img, 0, h_in - img.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        img = img[:h_in]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chans.append(((rgb - MEAN) / STD).transpose(2, 0, 1))
    return np.concatenate(chans, axis=0)


def _peaks(hm: np.ndarray, k: int, thresh: float) -> list[tuple[int, int, float]]:
    """Top-k local maxima (3x3) of a heatmap above ``thresh``: (row, col, score)."""
    pooled = cv2.dilate(hm, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero((hm >= pooled) & (hm >= thresh))
    order = np.argsort(-hm[ys, xs])[:k]
    return [(int(ys[i]), int(xs[i]), float(hm[ys[i], xs[i]])) for i in order]


def decode(out: dict[str, np.ndarray], scale: float, ball_k: int = 8, ball_thresh: float = 0.05,
           person_k: int = 60, person_thresh: float = 0.25) -> tuple[list[tuple[float, float, float]],
                                                                     list[tuple[float, float, float, float, float]]]:
    """Network outputs of one frame (sigmoid applied to heatmaps) -> ball points (x, y, score) and person boxes
    (x1, y1, x2, y2, score) in original-frame pixels (``scale``: input px per frame px)."""
    balls = []
    for r, c, sc in _peaks(out["ball_hm"][0], ball_k, ball_thresh):
        ox, oy = out["ball_off"][:, r, c]
        balls.append(((c + ox) * STRIDE / scale, (r + oy) * STRIDE / scale, sc))
    persons = []
    for r, c, sc in _peaks(out["person_hm"][0], person_k, person_thresh):
        ox, oy = out["person_off"][:, r, c]
        lw, lh = out["person_wh"][:, r, c]
        cx, cy = (c + ox) * STRIDE / scale, (r + oy) * STRIDE / scale
        w, h = np.exp(lw) * STRIDE / scale, np.exp(lh) * STRIDE / scale
        persons.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, sc))
    return balls, persons


@dataclass
class StudentConfig:
    weights: str = "runs/models/student/best.pt"
    device: str | None = None  # None: mps when available, else cpu
    frames: int = 3
    ball_thresh: float = 0.08
    person_thresh: float = 0.3
    ball_radius_px: float = 12.0  # box half-size drawn around a ball point (1080p); radius of an NBA ball in view
    half: bool = True  # fp16 on mps


class StudentDetector:
    """``Detector`` running ``BallPersonNet`` on the current and previous frames."""

    def __init__(self, config: StudentConfig | None = None, net: Any | None = None) -> None:
        self.config = config or StudentConfig()
        self._net = net
        self._device: Any = None
        self._buf: deque[np.ndarray] = deque(maxlen=self.config.frames)
        self.last_raw: tuple[list[tuple[float, float, float]], list[tuple[float, float, float, float, float]]] | None = None

    def _model(self) -> Any:
        if self._net is None:
            torch = _torch()
            dev = self.config.device or ("mps" if torch.backends.mps.is_available() else "cpu")
            net = build_net(self.config.frames, pretrained=False)
            state = torch.load(Path(self.config.weights), map_location="cpu", weights_only=True)
            net.load_state_dict(state.get("model", state))
            net.eval().to(dev)
            if self.config.half and dev == "mps":
                net.half()
            self._net, self._device = net, dev
        return self._net

    def reset(self) -> None:
        self._buf.clear()

    def infer(self, frames_bgr: list[list[np.ndarray]]) -> list[tuple[list, list]]:
        """Batch inference: each item is a list of ``frames`` frames (oldest first)."""
        x = np.stack([preprocess_frames(fs) for fs in frames_bgr])
        return self.infer_arrays(x, [fs[-1].shape[1] for fs in frames_bgr])

    def infer_arrays(self, x: np.ndarray, frame_widths: list[int]) -> list[tuple[list, list]]:
        """Batch inference on preprocessed inputs (B, 3 * frames, H, W) (``preprocess_frames``; a caller that keeps
        each frame's own preprocessed slice avoids redoing it for every sample that contains the frame)."""
        torch = _torch()
        net = self._model()
        t = torch.from_numpy(np.ascontiguousarray(x)).to(self._device or "cpu")
        if self.config.half and self._device == "mps":
            t = t.half()
        with torch.no_grad():
            out = net(t)
        res = {k: v.float().cpu().numpy() for k, v in out.items()}
        decoded = []
        for i, width in enumerate(frame_widths):
            one = {k: v[i] for k, v in res.items()}
            for k in ("ball_hm", "person_hm"):
                one[k] = 1.0 / (1.0 + np.exp(-one[k]))
            decoded.append(decode(one, INPUT_HW[1] / width, ball_thresh=self.config.ball_thresh,
                                  person_thresh=self.config.person_thresh))
        return decoded

    def detect(self, frame_bgr: np.ndarray, frame_idx: int, t: float) -> FrameDetections:
        self._buf.append(frame_bgr)
        frames = list(self._buf)
        while len(frames) < self.config.frames:
            frames.insert(0, frames[0])
        balls, persons = self.infer([frames])[0]
        self.last_raw = (balls, persons)
        h, w = frame_bgr.shape[:2]
        r = self.config.ball_radius_px * h / 1080.0
        dets = [Detection(BBox(x1, y1, x2, y2), ObjClass.PLAYER, sc) for x1, y1, x2, y2, sc in persons]
        dets += [Detection(BBox(x - r, y - r, x + r, y + r), ObjClass.BALL, sc) for x, y, sc in balls]
        return FrameDetections(frame_idx=frame_idx, t=t, image_size=(w, h), detections=dets)
