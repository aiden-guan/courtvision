"""The shot judge: made or missed from what the camera saw at the rim, a moment after the event engine's call.

The engine calls a shot made or missed when the tracked ball crosses the rim's line or leaves its zone
(``events/shots.py``). One tracked ball is a thin thread at the rim, where the net, the board and the players' hands
hide it, so its call is often wrong; the scorebug used to confirm or veto it, which leaves the result to the
broadcast's graphic. The judge looks again from the video alone (docs/research/possession_decision_model.md §9):

* every frame it keeps the rims in view and **all** the detector's ball candidates (``FrameState.balls``), not only the
  one the ball tracker follows;
* ``delay_s`` after a shot result it finds the moment the tracked ball passed nearest that rim, reads where the
  candidates were about the rim from 1 s before to 1.4 s after (``rim_clips.zone_scores``: above, at the ring, in the
  net, under it, beside it, high), and turns that into ``makes.zone_features`` (the ball seen in the net after the
  arrival, above the ring and then in the net, at the ring and then off to the side, ...);
* with a rim model (``rim_model.py``), the frames of the rim itself around the arrival (``FrameState.rim_crops``, cut as
  ``rim_clips.clip_at`` cuts them) go through it once: its P(made) is ``cnn_made``;
* a logistic model over those and the engine's own rim cues (``makes.CUE_FEATURES``), fitted on the play-by-play of
  analysed games (``tools/nba_live_judge.py``), gives P(made); the engine then keeps or overturns its call
  (``EventEngine.judge_shot``).

Live it decides about 2.5 s after the engine (the scorebug's confirmation waited 3 s); ahead of time it is the same.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from courtvision.types import BBox, Event, EventKind, FrameState

from .makes import CUE_FEATURES, ZONE_FEATURES, Logistic, zone_features
from .rim_clips import CLIP_OFFSETS, CLIP_PX, SERIES_SPAN, ZONES, zone_scores

__all__ = ["CNN_FEATURES", "JUDGE_FEATURES", "JudgeConfig", "ShotJudge", "cue_features", "load_judge_model"]

CNN_FEATURES = ("cnn_made", "cnn_missing")
JUDGE_FEATURES = CUE_FEATURES + ZONE_FEATURES + ("is_ft",)
RESULTS = (EventKind.SHOT_MADE, EventKind.SHOT_MISSED)
log = logging.getLogger(__name__)


@dataclass
class JudgeConfig:
    delay_s: float = 2.5  # a result is judged this long after it (the series reaches 1.4 s past the rim arrival)
    search_s: tuple[float, float] = (-2.5, 1.0)  # the arrival: the tracked ball nearest the rim this near the result
    near_rims: float = 3.0  # ... within this many rim widths of its centre, else the result's own time
    hold_s: float = 0.5  # a frame without the rim takes the rim box of a frame at most this far away
    min_score: float = 0.08  # ball candidates below this are ignored (the live detector's ball threshold)
    keep_s: float = 8.0  # frames kept
    rim_model: str | None = None  # rim_model.py weights (relative: to the judge's model file); None: no cnn_made


def cue_features(e: Event) -> dict[str, float]:
    """``makes.CUE_FEATURES`` of an engine result (``attrs["cues"]``), as ``makes.candidate_features`` makes them."""
    cues = e.attrs.get("cues") or {}
    margin = cues.get("crossing_margin_px")
    return {"crossed_inside": float(bool(cues.get("crossed_inside"))), "rim_contact": float(bool(cues.get("rim_contact"))),
            "drop_in_band": float(bool(cues.get("drop_in_band"))),
            "drop_is_freefall": float(bool(cues.get("drop_is_freefall"))),
            "margin": 0.0 if margin is None else float(np.clip(margin, -60.0, 40.0)) / 20.0,
            "margin_missing": float(margin is None), "occluded": min(float(cues.get("occluded_frames") or 0), 5.0)}


def load_judge_model(path: str | Path) -> tuple[Logistic, dict[str, Any]]:
    """The judge's model (``tools/nba_live_judge.py``): the logistic and its settings (``min_score``, ``rim_model``)."""
    doc = json.loads(Path(path).read_text())
    return Logistic.from_dict(doc["made"]), {k: v for k, v in doc.items() if k != "made"}


@dataclass
class _Frame:
    t: float
    rims: dict[str, BBox]
    balls: np.ndarray | None
    ball: tuple[float, float] | None
    crops: dict[str, np.ndarray] | None = None


class ShotJudge:
    """Feed it every frame (``observe``) and the engine's events (``on_events``); ``due(now)`` gives the verdicts
    ready by ``now``: (shot id, made, P(made))."""

    def __init__(self, model: Logistic, config: JudgeConfig | None = None, fps: float = 29.97,
                 rim_model: Any = None) -> None:
        self.model = model
        self.config = config or JudgeConfig()
        self.fps = fps if fps and math.isfinite(fps) and fps > 0 else 29.97
        self.rim_model = rim_model  # (net, RimConfig) of rim_model.load_model, or None
        self._frames: deque[_Frame] = deque()
        self._waiting: dict[str, Event] = {}  # shot id -> the latest revision of its result
        self._judged: set[str] = set()

    @property
    def wants_crops(self) -> bool:
        """Whether the frames should carry their rim crops (``FrameState.rim_crops``)."""
        return self.rim_model is not None

    def observe(self, frame: FrameState) -> None:
        rims = {r.side: r.bbox for r in frame.rims}
        b = frame.ball
        ball = b.image_xy if b is not None and b.visible and b.image_xy is not None and not b.interpolated else None
        balls = None if frame.balls is None else np.asarray(frame.balls, np.float32).reshape(-1, 3)
        self._frames.append(_Frame(frame.t, rims, balls, ball, frame.rim_crops if self.wants_crops else None))
        while self._frames and frame.t - self._frames[0].t > self.config.keep_s:
            self._frames.popleft()

    def on_events(self, events: Iterable[Event]) -> None:
        for e in events:
            if e.kind in RESULTS and e.shot_id and e.source == "cv" and e.shot_id not in self._judged:
                self._waiting[e.shot_id] = e

    def due(self, now: float) -> list[tuple[str, bool, float]]:
        out = []
        for sid, e in list(self._waiting.items()):
            if now - e.t < self.config.delay_s:
                continue
            del self._waiting[sid]
            self._judged.add(sid)
            f = self.features(e)
            if f is not None:
                p = self.model.p(f)
                out.append((sid, p >= 0.5, p))
        return out

    # ------------------------------------------------------------------ evidence

    def _arrival(self, e: Event) -> tuple[int, str] | None:
        """(index in the kept frames, rim side) of the tracked ball's pass nearest a rim around result ``e``, else the
        frame of the result's time and the rim seen most around it; None when no rim is in view."""
        lo, hi = e.t + self.config.search_s[0], e.t + self.config.search_s[1]
        best: tuple[float, int, str] | None = None
        seen: dict[str, int] = {}
        for i, fr in enumerate(self._frames):
            if not lo <= fr.t <= hi:
                continue
            for side, box in fr.rims.items():
                seen[side] = seen.get(side, 0) + 1
                if fr.ball is None:
                    continue
                rw = max(box.x2 - box.x1, 1.0)
                d = math.hypot(fr.ball[0] - (box.x1 + box.x2) / 2, fr.ball[1] - (box.y1 + box.y2) / 2) / rw
                if best is None or d < best[0]:
                    best = (d, i, side)
        if best is not None and best[0] <= self.config.near_rims:
            return best[1], best[2]
        if not seen:
            return None
        side = max(seen, key=lambda s: (seen[s], s))
        i = min(range(len(self._frames)), key=lambda k: abs(self._frames[k].t - e.t))
        return i, side

    def series(self, centre: int, side: str) -> np.ndarray:
        """The ball-zone series about frame ``centre`` of the kept frames (as ``rim_clips.ball_zone_series``)."""
        frames = list(self._frames)
        t0 = frames[centre].t
        span = (SERIES_SPAN[0] / 29.97, SERIES_SPAN[1] / 29.97)
        n = SERIES_SPAN[1] - SERIES_SPAN[0] + 1
        out = np.full((n, len(ZONES)), -1.0, np.float32)
        times = np.array([fr.t for fr in frames])
        for k in range(n):
            t = t0 + span[0] + k / 29.97
            j = int(np.argmin(np.abs(times - t)))
            if abs(times[j] - t) > 0.6 / self.fps + 1e-6 and abs(times[j] - t) > 0.02:
                continue  # no frame at that moment (before the kept frames, or a gap)
            box = self._box_near(frames, j, side)
            if box is not None:
                out[k] = zone_scores(frames[j].balls, box, self.config.min_score)
        return out

    def _box_near(self, frames: list[_Frame], j: int, side: str) -> BBox | None:
        t = frames[j].t
        for d in range(len(frames)):
            for k in ((j,) if d == 0 else (j - d, j + d)):
                if 0 <= k < len(frames):
                    if abs(frames[k].t - t) > self.config.hold_s:
                        continue
                    box = frames[k].rims.get(side)
                    if box is not None:
                        return box
            if all(not (0 <= k < len(frames)) or abs(frames[k].t - t) > self.config.hold_s for k in (j - d, j + d)):
                break
        return None

    def clip(self, centre: int, side: str) -> np.ndarray | None:
        """The rim clip about frame ``centre`` of the kept frames (``rim_clips.CLIP_OFFSETS``, as ``rim_clips.clip_at``
        cuts it; a moment without a crop of that rim is black); None when no moment has one."""
        frames = list(self._frames)
        times = np.array([fr.t for fr in frames])
        t0 = frames[centre].t
        out = np.zeros((len(CLIP_OFFSETS), CLIP_PX, CLIP_PX, 3), np.uint8)
        seen = 0
        for k, o in enumerate(CLIP_OFFSETS):
            t = t0 + o / 29.97
            j = int(np.argmin(np.abs(times - t)))
            crops = frames[j].crops
            if abs(times[j] - t) <= 0.6 / self.fps + 0.02 and crops is not None and side in crops:
                out[k] = crops[side]
                seen += 1
        return out if seen >= len(CLIP_OFFSETS) // 2 else None

    def cnn(self, centre: int, side: str) -> dict[str, float]:
        """``CNN_FEATURES``: the rim model's P(made) on the rim clip about the arrival (missing: no model or clip)."""
        clip = self.clip(centre, side) if self.rim_model is not None else None
        if clip is None:
            return {"cnn_made": 0.5, "cnn_missing": 1.0}
        from .rim_model import predict

        net, cfg = self.rim_model
        p = predict(net, clip[None], cfg)[0]
        return {"cnn_made": float(p[0] / max(p[0] + p[1], 1e-9)), "cnn_missing": 0.0}

    def features(self, e: Event) -> dict[str, float] | None:
        """``JUDGE_FEATURES`` (and ``CNN_FEATURES``) of result ``e``; None when no rim was in view around it."""
        found = self._arrival(e)
        if found is None:
            return None
        series = self.series(*found)
        return {**cue_features(e), **zone_features(series), **self.cnn(*found),
                "is_ft": float(e.value == 1 or e.attrs.get("shot_type") == "free_throw")}


def judge_from_config(path: str | Path | None, fps: float) -> ShotJudge | None:
    """A judge with the model at ``path`` (None, or a missing file: no judge), and its rim model when it names one
    (a path relative to the model file's folder); a rim model that cannot be loaded is left out (``cnn_missing``)."""
    if not path or not Path(path).is_file():
        return None
    model, extra = load_judge_model(path)
    cfg = JudgeConfig(**{k: v for k, v in extra.items() if k in JudgeConfig.__dataclass_fields__})
    rim = None
    if cfg.rim_model:
        weights = Path(cfg.rim_model)
        weights = weights if weights.is_absolute() else Path(path).parent / weights
        try:
            from .rim_model import load_model

            rim = load_model(weights)
        except Exception:
            log.exception("Could not load the shot judge's rim model %s; judging without it", weights)
    return ShotJudge(model, cfg, fps, rim)


def wants_crops(path: str | Path | None) -> bool:
    """Whether the judge's model at ``path`` reads the rim model (so the frames must carry their rim crops)."""
    if not path or not Path(path).is_file():
        return False
    try:
        return bool(json.loads(Path(path).read_text()).get("rim_model"))
    except (OSError, ValueError):
        return False
