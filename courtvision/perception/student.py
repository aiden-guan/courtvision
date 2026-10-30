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

Other videos. A frame is scaled to fit 960x544 whole (a 16:9 frame of any resolution becomes 960x540, as in
training; a taller one is padded at the right instead of losing its bottom rows). The training clips were 29.97 fps
(``TRAIN_FPS``), so the input frames are 33 ms apart; on a faster video the earlier input frames are taken
``history_step`` frames apart (every second frame at 50-60 fps) to keep that spacing, and with it the ball's blur
and motion between the frames the network learned. A 60 fps camera's shorter exposure blurs less than the training
frames did; that difference remains.

Speed. On Apple silicon the network runs on the Neural Engine through Core ML (``StudentConfig.backend``): the
weights are exported once to an fp16 ML program cached by their hash (``coreml_model``), 8 ms a frame on an M3 Pro
where MPS takes 28 (and the GPU stays free). Its fp16 outputs stay within 0.002 of the fp32 network, as close as MPS
fp16 is, and the held-out ball F1 and person recall of ``train_student.evaluate`` are unchanged (0.9608; 0.8039
against 0.8037). Elsewhere, or when the export fails, it runs on torch as before. Each frame is normalised once
(``detect`` keeps the normalised frames, not the raw ones) through per-channel lookup tables, bit-identical to the
float maths they replace and about 15 times faster.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from courtvision.types import BBox, Detection, FrameDetections, ObjClass

__all__ = ["INPUT_HW", "OUTPUTS", "TRAIN_FPS", "BallPersonNet", "StudentConfig", "StudentDetector", "coreml_model",
           "decode", "history_step", "input_scale", "preprocess_frames"]

log = logging.getLogger(__name__)

INPUT_HW = (544, 960)  # network input (h, w); 1080p frames are scaled by 0.5 and padded by 4 rows
STRIDE = 4
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
TRAIN_FPS = 29.97  # frame rate of the training clips: the input frames are 1 / TRAIN_FPS apart
OUTPUTS = ("ball_hm", "ball_off", "person_hm", "person_wh", "person_off")
# A pixel value's network input per channel (R, G, B): the float32 maths of before, ((v / 255) - mean) / std, done
# once for the 256 values, so a frame is normalised by table lookups (cv2.LUT) with the same results
_NORM = (np.arange(256, dtype=np.float32)[:, None] / 255.0 - MEAN) / STD
_NORM_LUTS = tuple(np.ascontiguousarray(_NORM[:, c]).reshape(1, 256) for c in range(3))


def history_step(fps: float, train_fps: float = TRAIN_FPS) -> int:
    """Frames between two input frames on a ``fps`` video so that they are about as far apart as in training: 1 up
    to 44 fps, 2 at 50 / 59.94 / 60 fps, 4 at 119.88 fps (1 for an unknown rate)."""
    if not (isinstance(fps, (int, float)) and np.isfinite(fps) and fps > 0):
        return 1
    return max(1, int(np.floor(fps / train_fps + 0.5)))


def input_scale(width: int, height: int) -> float:
    """Network input px per frame px: the frame scaled to fit ``INPUT_HW`` whole (960 / width for a 16:9 frame)."""
    s = INPUT_HW[1] / width
    return s if height * s <= INPUT_HW[0] + 0.5 else INPUT_HW[0] / height


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
    """Frames (oldest first) -> (3 * n, H, W) float32 network input (C-contiguous): scaled to fit 960x544
    (``input_scale``; 960 px wide for 16:9), padded at the bottom (and at the right for a frame taller than 16:9), RGB,
    ImageNet-normalised (through ``_NORM_LUTS``)."""
    h_in, w_in = INPUT_HW
    out = np.empty((3 * len(frames_bgr), h_in, w_in), np.float32)
    for i, f in enumerate(frames_bgr):
        s = input_scale(f.shape[1], f.shape[0])
        wide = s == w_in / f.shape[1]
        size = (w_in if wide else round(f.shape[1] * s), round(f.shape[0] * s))
        img = cv2.resize(f, size, interpolation=cv2.INTER_AREA) if s != 1.0 else f
        if img.shape[0] < h_in or img.shape[1] < w_in:
            img = cv2.copyMakeBorder(img, 0, max(0, h_in - img.shape[0]), 0, max(0, w_in - img.shape[1]),
                                     cv2.BORDER_CONSTANT)
        planes = cv2.split(np.ascontiguousarray(img[:h_in, :w_in]))
        for c in range(3):  # R, G, B: the BGR planes 2, 1, 0
            out[3 * i + c] = cv2.LUT(planes[2 - c], _NORM_LUTS[c])
    return out


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
    device: str | None = None  # torch backend: None = mps when available, else cpu
    frames: int = 3
    ball_thresh: float = 0.08
    person_thresh: float = 0.3
    ball_radius_px: float = 12.0  # box half-size drawn around a ball point (1080p); radius of an NBA ball in view
    half: bool = True  # fp16 on mps
    frame_step: int = 1  # stream frames between two input frames: ``history_step(fps)`` (2 on a 50-60 fps stream)
    # "coreml": the Neural Engine (macOS with coremltools; ``coreml_model``); "torch": PyTorch on ``device``;
    # "auto": coreml on Apple silicon where it loads, else torch
    backend: str = "auto"
    coreml_cache: str | None = None  # folder of the exported models; None: $COURTVISION_CACHE or ~/Library/Caches/courtvision


def _cache_dir(folder: str | None) -> Path:
    return Path(folder or os.environ.get("COURTVISION_CACHE") or Path.home() / "Library" / "Caches" / "courtvision")


def coreml_model(weights: str | Path, frames: int = 3, cache: str | Path | None = None) -> Any:
    """The network of ``weights`` as a Core ML ML program (fp16, one sample of ``INPUT_HW``) on the CPU and Neural
    Engine, exported on first use into ``cache`` (``student-<weights hash>-...mlpackage``; a few seconds) and loaded
    from there afterwards. Raises when coremltools is missing or the export fails."""
    import coremltools as ct

    digest = hashlib.sha256(Path(weights).read_bytes()).hexdigest()[:16]
    path = _cache_dir(str(cache) if cache else None) / (
        f"student-{digest}-f{frames}-{INPUT_HW[0]}x{INPUT_HW[1]}-ct{ct.__version__}.mlpackage")
    if not path.exists():
        torch = _torch()
        net = build_net(frames, pretrained=False)
        state = torch.load(Path(weights), map_location="cpu", weights_only=True)
        net.load_state_dict(state.get("model", state))
        net.eval()

        class _Outputs(torch.nn.Module):  # a tuple in ``OUTPUTS`` order: Core ML names them
            def __init__(self) -> None:
                super().__init__()
                self.net = net

            def forward(self, x: Any) -> tuple[Any, ...]:
                out = self.net(x)
                return tuple(out[k] for k in OUTPUTS)

        example = torch.zeros(1, 3 * frames, *INPUT_HW)
        with torch.no_grad():
            traced = torch.jit.trace(_Outputs().eval(), example)
        ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=tuple(example.shape))],
                        outputs=[ct.TensorType(name=k) for k in OUTPUTS], convert_to="mlprogram",
                        compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS13)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.stem}-tmp{os.getpid()}.mlpackage")  # Core ML saves only to .mlpackage
        ml.save(str(tmp))
        try:
            tmp.rename(path)
        except OSError:  # another process exported it meanwhile
            shutil.rmtree(tmp, ignore_errors=True)
        log.info("student detector exported to Core ML: %s", path)
    return ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)


class StudentDetector:
    """``Detector`` running ``BallPersonNet`` on the current and previous frames (``frame_step`` apart)."""

    def __init__(self, config: StudentConfig | None = None, net: Any | None = None) -> None:
        self.config = config or StudentConfig()
        self._net = net
        self._ml: Any = None  # the Core ML model (backend "coreml")
        self._device: Any = None
        # each frame's own preprocessed (3, H, W) input, so a frame is normalised once, not once per sample it is in
        self._buf: deque[np.ndarray] = deque(maxlen=(self.config.frames - 1) * max(1, self.config.frame_step) + 1)
        self.last_raw: tuple[list[tuple[float, float, float]], list[tuple[float, float, float, float, float]]] | None = None

    @property
    def backend(self) -> str:
        """The backend that runs the network: "coreml" or "torch" (loads it)."""
        self._model()
        return "coreml" if self._ml is not None else "torch"

    def _model(self) -> Any:
        if self._net is None and self._ml is None:
            want = self.config.backend
            if want not in ("auto", "coreml", "torch"):
                raise ValueError(f"unknown student backend {want!r}")
            if want == "coreml" or (want == "auto" and sys.platform == "darwin" and platform.machine() == "arm64"):
                try:
                    self._ml = coreml_model(self.config.weights, self.config.frames, self.config.coreml_cache)
                    return self._ml
                except Exception as exc:  # noqa: BLE001 - any failure (no coremltools, a failed export) -> torch
                    if want == "coreml":
                        raise
                    log.warning("student detector: Core ML unavailable (%s: %s); running on torch",
                                type(exc).__name__, exc)
            torch = _torch()
            dev = self.config.device or ("mps" if torch.backends.mps.is_available() else "cpu")
            net = build_net(self.config.frames, pretrained=False)
            state = torch.load(Path(self.config.weights), map_location="cpu", weights_only=True)
            net.load_state_dict(state.get("model", state))
            net.eval().to(dev)
            if self.config.half and dev == "mps":
                net.half()
            self._net, self._device = net, dev
        return self._ml if self._ml is not None else self._net

    def reset(self) -> None:
        self._buf.clear()

    def infer(self, frames_bgr: list[list[np.ndarray]]) -> list[tuple[list, list]]:
        """Batch inference: each item is a list of ``frames`` frames (oldest first)."""
        x = np.stack([preprocess_frames(fs) for fs in frames_bgr])
        return self.infer_arrays(x, [(fs[-1].shape[1], fs[-1].shape[0]) for fs in frames_bgr])

    def _forward(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Raw float32 outputs (logit heatmaps) for a batch of preprocessed inputs."""
        net = self._model()
        if self._ml is not None:
            outs = [net.predict({"x": np.ascontiguousarray(x[i:i + 1], np.float32)}) for i in range(len(x))]
            return {k: np.concatenate([np.asarray(o[k], np.float32) for o in outs]) for k in OUTPUTS}
        torch = _torch()
        t = torch.from_numpy(np.ascontiguousarray(x)).to(self._device or "cpu")
        if self.config.half and self._device == "mps":
            t = t.half()
        with torch.no_grad():
            out = net(t)
        return {k: v.float().cpu().numpy() for k, v in out.items()}

    def infer_arrays(self, x: np.ndarray, frame_sizes: list[int] | list[tuple[int, int]]) -> list[tuple[list, list]]:
        """Batch inference on preprocessed inputs (B, 3 * frames, H, W) (``preprocess_frames``; a caller that keeps
        each frame's own preprocessed slice avoids redoing it for every sample that contains the frame).
        ``frame_sizes``: each sample's frame (width, height), or its width alone for a 16:9 frame."""
        res = self._forward(x)
        decoded = []
        for i, wh in enumerate(frame_sizes):
            one = {k: v[i] for k, v in res.items()}
            for k in ("ball_hm", "person_hm"):
                one[k] = 1.0 / (1.0 + np.exp(-one[k]))
            scale = input_scale(*wh) if isinstance(wh, (tuple, list)) else INPUT_HW[1] / wh
            decoded.append(decode(one, scale, ball_thresh=self.config.ball_thresh,
                                  person_thresh=self.config.person_thresh))
        return decoded

    def detect(self, frame_bgr: np.ndarray, frame_idx: int, t: float) -> FrameDetections:
        self._buf.append(preprocess_frames([frame_bgr]))
        buf, step = list(self._buf), max(1, self.config.frame_step)
        x = np.concatenate([buf[max(0, len(buf) - 1 - j * step)] for j in range(self.config.frames - 1, -1, -1)])
        #   oldest first; the first frames of a stream repeat the earliest one
        h, w = frame_bgr.shape[:2]
        balls, persons = self.infer_arrays(x[None], [(w, h)])[0]
        self.last_raw = (balls, persons)
        r = self.config.ball_radius_px * h / 1080.0
        dets = [Detection(BBox(x1, y1, x2, y2), ObjClass.PLAYER, sc) for x1, y1, x2, y2, sc in persons]
        dets += [Detection(BBox(x - r, y - r, x + r, y + r), ObjClass.BALL, sc) for x, y, sc in balls]
        return FrameDetections(frame_idx=frame_idx, t=t, image_size=(w, h), detections=dets)
