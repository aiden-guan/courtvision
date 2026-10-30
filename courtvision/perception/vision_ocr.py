"""Text recognition through Apple's Vision framework (macOS only).

``VisionOCR`` keeps one helper process (``vision_ocr.swift``, compiled with ``swiftc -O`` on first use into
``~/Library/Caches/courtvision``) and sends it image crops; each call returns the recognised text lines with
their confidence and pixel boxes. Vision runs on the Neural Engine, so a scorebug crop takes a few ms to tens of
ms. Nothing is downloaded. On other platforms, or without the Swift toolchain, ``available()`` is False.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = ["OCRLine", "VisionOCR", "available"]

log = logging.getLogger(__name__)

_SOURCE = Path(__file__).with_name("vision_ocr.swift")


@dataclass(frozen=True)
class OCRLine:
    text: str
    conf: float
    box: tuple[float, float, float, float]  # x, y, w, h in pixels of the crop sent

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2.0, y + h / 2.0


def available() -> bool:
    return sys.platform == "darwin" and shutil.which("swiftc") is not None and _SOURCE.is_file()


def _binary() -> Path:
    """The compiled helper, rebuilt when the Swift source changes."""
    digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:12]
    cache = Path(os.environ.get("COURTVISION_CACHE", Path.home() / "Library" / "Caches" / "courtvision"))
    exe = cache / f"vision_ocr-{digest}"
    if not exe.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{exe.name}-", dir=cache)
        os.close(fd)
        tmp = Path(name)
        try:
            log.info("compiling %s", _SOURCE.name)
            subprocess.run(["swiftc", "-O", str(_SOURCE), "-o", str(tmp)], check=True, capture_output=True)
            tmp.replace(exe)
        finally:
            tmp.unlink(missing_ok=True)
    return exe


class VisionOCR:
    """``read(image_bgr) -> list[OCRLine]``. Thread-safe; ``close()`` ends the helper."""

    def __init__(self, fast: bool = False) -> None:
        if not available():
            raise RuntimeError("Vision OCR needs macOS with the Swift toolchain (xcode-select --install)")
        args = [str(_binary())] + (["--fast"] if fast else [])
        self._proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._lock = threading.Lock()

    def read(self, image_bgr: np.ndarray) -> list[OCRLine]:
        ok, buf = cv2.imencode(".png", image_bgr)
        if not ok:
            return []
        data = buf.tobytes()
        with self._lock:
            assert self._proc.stdin is not None and self._proc.stdout is not None
            self._proc.stdin.write(struct.pack(">I", len(data)) + data)
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("Vision OCR helper exited")
        return [OCRLine(str(r["text"]), float(r["conf"]), tuple(float(v) for v in r["box"]))  # type: ignore[arg-type]
                for r in json.loads(line)]

    def close(self) -> None:
        proc = self._proc
        if proc.poll() is None:
            try:
                assert proc.stdin is not None
                proc.stdin.write(struct.pack(">I", 0))
                proc.stdin.close()
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                proc.kill()

    def __enter__(self) -> VisionOCR:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
