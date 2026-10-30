"""The rim model: made, missed or no shot from a rim clip (``rim_clips.py``), learned from the play-by-play.

What it sees: each frame of a clip is the same square about the rim (the rim always the same size and place, the
camera's pans and zooms taken out by the camera track), so what changes from frame to frame is the ball and the net.
A made shot drops through the ring and the net; a miss bounces off the ring or the board, or never reaches it.

``RimNet``:

* a per-frame encoder, torchvision's ResNet-18 with its ImageNet weights up to ``layer3`` (stride 16: a 96 px frame
  becomes 6 x 6 cells), then a 1 x 1 convolution and a linear layer over the cells, so where the ball is (above the
  ring, in the net, beside it) is kept rather than averaged away;
* a temporal part, two 1-D convolutions over the frames' features and the maximum over time, so the moment the ball
  passes the rim can fall anywhere in the clip (the play-by-play places a shot only to a second or two);
* three outputs: made, missed, no shot (``CLASSES``).

With a few hundred shots a game, training leans on the ImageNet weights, on augmentation that keeps what matters
(mirror images, small shifts and zooms, colour, a shifted window in time) and on a short schedule. ``train`` fits one
model; ``predict`` gives each clip's probabilities.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["CLASSES", "RimConfig", "RimNet", "load_clips", "load_model", "predict", "save_model", "train"]

CLASSES = ("made", "missed", "none")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet, RGB
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class RimConfig:
    frames: int = 33  # frames of a clip the network sees (of rim_clips.CLIP_OFFSETS' 37: a window shifted in training)
    px: int = 96
    epochs: int = 24
    batch: int = 12
    lr: float = 3e-4
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-4
    shift_px: int = 8  # random translation in training
    zoom: float = 0.12  # random zoom in training, +-
    feat: int = 256
    temporal: int = 128
    dropout: float = 0.3
    seed: int = 0
    device: str | None = None
    class_weights: tuple[float, ...] | None = None  # None: inverse class frequency, normalised
    motion: bool = False  # frames as (grey, change from the frame before, change to the frame after) instead of RGB
    freeze: bool = False  # keep the ImageNet encoder as it is (train the rest only)
    extra: dict[str, Any] = field(default_factory=dict)


def _torch() -> Any:
    import torch

    return torch


def _device(cfg: RimConfig) -> Any:
    torch = _torch()
    if cfg.device:
        return torch.device(cfg.device)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _build(cfg: RimConfig) -> Any:
    torch = _torch()
    nn = torch.nn
    import torchvision

    class RimNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            try:
                r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception:  # no weights cached and no network: start from scratch
                log.warning("ResNet-18 ImageNet weights unavailable; the rim model starts from random weights")
                r = torchvision.models.resnet18(weights=None)
            self.encoder = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
            cells = math.ceil(cfg.px / 16) ** 2
            self.squeeze = nn.Sequential(nn.Conv2d(256, 32, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
            self.frame = nn.Sequential(nn.Flatten(), nn.Dropout(cfg.dropout), nn.Linear(32 * cells, cfg.feat),
                                       nn.ReLU(inplace=True))
            self.temporal = nn.Sequential(nn.Conv1d(cfg.feat, cfg.temporal, 5, padding=2), nn.ReLU(inplace=True),
                                          nn.Dropout(cfg.dropout), nn.Conv1d(cfg.temporal, cfg.temporal, 5, padding=2),
                                          nn.ReLU(inplace=True))
            self.head = nn.Linear(cfg.temporal, len(CLASSES))

        def forward(self, x: Any) -> Any:  # x: B, T, 3, H, W (normalised)
            b, t = x.shape[:2]
            f = self.frame(self.squeeze(self.encoder(x.flatten(0, 1))))
            g = self.temporal(f.view(b, t, -1).transpose(1, 2))
            return self.head(g.amax(dim=2))

    return RimNet()


RimNet = _build  # the network for a config (built lazily: torch is imported only when a model is made)


def load_clips(folders: Sequence[Path]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """The clips of ``rim_clips`` folders, concatenated: frames (N, T, px, px, 3) BGR, valid (N, T), meta."""
    frames, valid, meta = [], [], []
    for d in folders:
        z = np.load(Path(d) / "clips.npz")
        frames.append(z["frames"])
        valid.append(z["valid"])
        meta += [json.loads(line) for line in (Path(d) / "meta.jsonl").open()]
    return np.concatenate(frames), np.concatenate(valid), meta


def _label(m: dict[str, Any]) -> int:
    return CLASSES.index(m["label"])


def _batch(frames: np.ndarray, idx: Sequence[int], cfg: RimConfig, rng: np.random.Generator | None,
           device: Any = None) -> Any:
    """Network input for clips ``idx`` on ``device``: RGB, normalised, (B, T, 3, px, px). Augmented when ``rng`` is
    given (a window shifted in time, then per clip one zoom and shift, mirror image, gain, offset and saturation,
    applied on the device)."""
    torch = _torch()
    F = torch.nn.functional
    n_all = frames.shape[1]
    starts = [int(rng.integers(0, n_all - cfg.frames + 1)) if rng is not None else (n_all - cfg.frames) // 2
              for _ in idx]
    clips = np.stack([frames[i, s:s + cfg.frames] for i, s in zip(idx, starts, strict=True)])  # B, T, H, W, 3 BGR
    x = torch.from_numpy(clips).to(device)
    x = x.flip(-1).permute(0, 1, 4, 2, 3).float().div_(255.0)  # B, T, 3, H, W RGB in [0, 1]
    b, t = x.shape[:2]
    if x.shape[-1] != cfg.px:
        x = F.interpolate(x.flatten(0, 1), size=(cfg.px, cfg.px), mode="area").view(b, t, 3, cfg.px, cfg.px)
    if rng is not None:
        z = 1.0 + rng.uniform(-cfg.zoom, cfg.zoom, b)
        sh = rng.uniform(-cfg.shift_px, cfg.shift_px, (b, 2)) * 2.0 / cfg.px
        flip = np.where(rng.integers(0, 2, b) == 1, -1.0, 1.0)
        theta = np.zeros((b, 2, 3), np.float32)  # output -> input sampling: x_in = flip * x_out / z - shift
        theta[:, 0, 0], theta[:, 1, 1] = flip / z, 1.0 / z
        theta[:, 0, 2], theta[:, 1, 2] = -sh[:, 0], -sh[:, 1]
        th = torch.from_numpy(np.repeat(theta, t, axis=0)).to(x.device)
        grid = F.affine_grid(th, (b * t, 3, cfg.px, cfg.px), align_corners=False)
        x = F.grid_sample(x.flatten(0, 1), grid, mode="bilinear", padding_mode="reflection",
                          align_corners=False).view(b, t, 3, cfg.px, cfg.px)
        gain = torch.from_numpy(rng.uniform(0.8, 1.2, (b, 1, 1, 1, 1)).astype(np.float32)).to(x.device)
        bias = torch.from_numpy(rng.uniform(-0.08, 0.08, (b, 1, 1, 1, 1)).astype(np.float32)).to(x.device)
        sat = torch.from_numpy(rng.uniform(0.8, 1.2, (b, 1, 1, 1, 1)).astype(np.float32)).to(x.device)
        grey = x.mean(dim=2, keepdim=True)
        x = ((grey + (x - grey) * sat) * gain + bias).clamp_(0.0, 1.0)
    if cfg.motion:  # grey, and its change from the frame before and to the frame after (the ball and net moving)
        g = x.mean(dim=2)
        prev = torch.cat([g[:, :1], g[:, :-1]], dim=1)
        nxt = torch.cat([g[:, 1:], g[:, -1:]], dim=1)
        x = torch.stack([(g - 0.45) / 0.25, (g - prev) * 8.0, (nxt - g) * 8.0], dim=2)
        return x
    mean = torch.tensor(MEAN, device=x.device).view(1, 1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 1, 3, 1, 1)
    return (x - mean) / std


def train(frames: np.ndarray, labels: Sequence[int], cfg: RimConfig | None = None,
          val: tuple[np.ndarray, Sequence[int]] | None = None) -> Any:
    """Fit a ``RimNet`` on clips ``frames`` (N, T, px, px, 3 BGR) with class indices ``labels``. ``val``: clips and
    labels to report the loss on after each epoch (not used to choose anything)."""
    cfg = cfg or RimConfig()
    torch = _torch()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = _device(cfg)
    net = _build(cfg).to(dev)
    y = np.asarray(labels, np.int64)
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    w = cfg.class_weights or tuple((counts.sum() / np.maximum(counts, 1.0) / len(CLASSES)).tolist())
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev), label_smoothing=0.05)
    enc = list(net.encoder.parameters())
    rest = [p for n, p in net.named_parameters() if not n.startswith("encoder.")]
    if cfg.freeze:
        for p in enc:
            p.requires_grad_(False)
    groups = ([] if cfg.freeze else [{"params": enc, "lr": cfg.backbone_lr}]) + [{"params": rest, "lr": cfg.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * math.ceil(len(y) / cfg.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps,
                                                pct_start=0.15)
    for epoch in range(cfg.epochs):
        net.train()
        for mod in net.encoder.modules():  # the ImageNet batch statistics stay (batches are a few correlated clips)
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.eval()
        order = rng.permutation(len(y))
        t0, total = time.time(), 0.0
        for s in range(0, len(order), cfg.batch):
            idx = order[s:s + cfg.batch]
            x = _batch(frames, idx, cfg, rng, dev)
            loss = loss_fn(net(x), torch.from_numpy(y[idx]).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(idx)
        msg = f"epoch {epoch + 1}/{cfg.epochs} loss {total / len(y):.4f} ({time.time() - t0:.0f}s)"
        if val is not None and (epoch + 1) % 4 == 0:
            p = predict(net, val[0], cfg)
            acc = float(np.mean(np.argmax(p, 1) == np.asarray(val[1])))
            msg += f"; val accuracy {acc:.3f}"
        log.info(msg)
    net.eval()
    return net


def predict(net: Any, frames: np.ndarray, cfg: RimConfig | None = None, flip: bool = True,
            batch: int = 16) -> np.ndarray:
    """Class probabilities (N, len(CLASSES)) of clips ``frames``; ``flip``: averaged with the mirror image."""
    cfg = cfg or RimConfig()
    torch = _torch()
    dev = next(net.parameters()).device
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(frames), batch):
            x = _batch(frames, range(s, min(len(frames), s + batch)), cfg, None, dev)
            p = torch.softmax(net(x), dim=1)
            if flip:
                p = (p + torch.softmax(net(torch.flip(x, dims=(4,))), dim=1)) / 2
            out.append(p.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)), np.float32)


def save_model(net: Any, cfg: RimConfig, path: Path, info: dict[str, Any] | None = None) -> None:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "config": asdict(cfg), "classes": CLASSES, "info": info or {}}, path)


def load_model(path: Path, device: str | None = None) -> tuple[Any, RimConfig]:
    torch = _torch()
    doc = torch.load(path, map_location="cpu", weights_only=False)
    cfg = RimConfig(**{k: v for k, v in doc["config"].items() if k in RimConfig.__dataclass_fields__})
    cfg.device = device
    net = _build(cfg)
    net.load_state_dict(doc["state_dict"])
    return net.to(_device(cfg)).eval(), cfg
