"""The possession analysis of a video the Studio analysed ahead, built from what that analysis already computed.

A research game gets its possession analysis from ``tools/nba_analyze.py <game> timeline`` (docs/NBA.md): the camera
track, the detector's ball candidates, the linked ball, the event engine's events, the possession timeline and every
shot decided from the video. The Studio's analysis ahead (``server/preanalysis.py``) tracks the court and detects the
ball on every frame it reads anyway, so it records both as it goes (``Recorder``), and when it ends ``build`` makes the
same analysis without any further pass over the video:

* the camera of each frame (``nba_track_court.py``'s format, so ``analyze.CameraTrack.load`` reads it), numbered by the
  frames analysed (a 60 fps video read at every 2nd frame is a 29.97 fps track);
* the linked ball and its 3-D flights (``analyze.build_ball_track``) from the detector's ball candidates;
* the possession timeline decoded with the stored settings (``timeline.decode``; no play-by-play to fit them);
* every shot decided by the shot models fitted on the research games (``makes.decide`` with ``shot_models.json`` of
  ``tools/nba_shot_models.py``, and the ball-zone evidence when they read it); without models the engine's results
  stay as they are;
* the events rebuilt from the timeline (``timeline.derive_events``).

It goes to ``<folder>/runs/auto/`` (``timeline.json``, ``timeline.jsonl``, as ``nba/results.py`` reads a run), with a
``game.json`` naming the video. Everything in it comes from the video.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from courtvision.types import Event

log = logging.getLogger(__name__)

__all__ = ["TAG", "Recorder", "build"]

TAG = "auto"


class Recorder:
    """Records an analysis's camera per frame (to ``<folder>/camera_parts/part0.jsonl``) and the detector's ball
    candidates per frame (kept, then saved by ``save``)."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        (folder / "camera_parts").mkdir(parents=True, exist_ok=True)
        self._fh: Any = None
        self.balls: dict[int, np.ndarray] = {}
        self.frames = 0

    def frame(self, frame_idx: int, t: float, center: Any, size: tuple[int, int], fps: float, stride: int,
              status: str, params: Any, balls: Any) -> None:
        if self._fh is None:
            self._fh = (self.folder / "camera_parts" / "part0.jsonl").open("w", encoding="utf-8")
            self._fh.write(json.dumps({"header": True, "center": [float(v) for v in center],
                                       "image_size": [int(size[0]), int(size[1])], "fps": float(fps),
                                       "stride": int(stride), "source": "studio analysis ahead"}) + "\n")
        row: dict[str, Any] = {"frame": int(frame_idx), "t": round(float(t), 4), "status": status}
        if params is not None:
            row["params"] = {"pan": float(params.pan), "tilt": float(params.tilt), "roll": float(params.roll),
                             "f": float(params.f)}
        self._fh.write(json.dumps(row) + "\n")
        if balls is not None and len(balls):
            self.balls[int(frame_idx)] = np.asarray(balls, np.float32).reshape(-1, 3)
        self.frames += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    @property
    def usable(self) -> bool:
        return self._fh is not None or (self.folder / "camera_parts" / "part0.jsonl").is_file()


def build(folder: Path, events: Iterable[Event], balls: Mapping[int, np.ndarray], video: str,
          shot_models: Path | None = None, title: str | None = None) -> Path | None:
    """The possession analysis (see the module docstring) in ``folder``; returns the run folder, or None when the
    camera was never tracked."""
    from courtvision.eval.events import latest_revisions
    from courtvision.geometry.court import court_for
    from courtvision.nba import makes as mk
    from courtvision.nba import timeline as tlm
    from courtvision.nba.analyze import BallTrack, CameraTrack, DetectionCache, build_ball_track
    from courtvision.nba.fusion import PlayDirection
    from courtvision.nba.rim_clips import ball_zone_series
    from courtvision.nba.rim_jev import rim_centre
    from courtvision.types import EventStatus

    t0 = time.time()
    parts = sorted((folder / "camera_parts").glob("part*.jsonl"))
    if not parts:
        return None
    try:
        camera = CameraTrack.load(parts)
    except ValueError:
        return None
    if not camera.params:
        return None
    court = court_for("nba")
    cache = DetectionCache(balls={int(f): b for f, b in balls.items()})
    ball = build_ball_track(cache, camera) if cache.balls else BallTrack({}, [])
    evs = [e for e in latest_revisions(list(events)) if e.status is not EventStatus.RETRACTED]
    direction = PlayDirection.from_camera(camera)
    cfg = tlm.TimelineConfig()
    grid = tlm.grid_for(evs, direction, cfg.dt_s)
    attack = tlm.attack_map(evs, direction, margin_ft=cfg.margin_ft)
    timeline = tlm.decode(evs, direction, cfg, ball.flights, court, attack, grid)
    shots: dict[str, Any] = {"models": str(shot_models) if shot_models else None}
    decided = evs
    if shot_models is not None and shot_models.is_file():
        models = mk.ShotModels.from_dict(json.loads(shot_models.read_text()))
        path = {f: (p[0], p[1]) for f, p in ball.path.items()}
        visits = mk.rim_visits(ball.path, camera, court)

        def zones(c: mk.Candidate) -> np.ndarray | None:
            found = rim_centre(c.t, camera, court, path, side=c.side)
            return None if found is None else ball_zone_series(cache.balls, camera, court, found[0], found[1])

        decided = mk.decide(evs, timeline, models, visits, zones=zones if models.uses_zones else None)
        kept = [e for e in decided if e.kind.value in ("shot_made", "shot_missed") and e.attrs.get("makes")]
        shots.update(candidates=len(mk.candidates(evs, visits)), kept=len(kept),
                     made=sum(e.kind.value == "shot_made" for e in kept), free_throws=sum(e.value == 1 for e in kept),
                     added=sum(e.attrs.get("inferred") == "rim_visit" for e in kept), zones=models.uses_zones)
    derived = tlm.derive_events(decided, timeline, tlm.DeriveConfig(), ball.flights, court)
    run = folder / "runs" / TAG
    run.mkdir(parents=True, exist_ok=True)
    with (run / "timeline.jsonl").open("w", encoding="utf-8") as fh:
        for e in derived:
            fh.write(json.dumps(e.to_dict(), default=_plain) + "\n")
    doc = {"attack": attack.to_dict(), "config": cfg.to_dict(), "shots": shots, "auto": True,
           "seconds": round(time.time() - t0, 1), **timeline.to_dict()}
    (run / "timeline.json").write_text(json.dumps(doc, default=_plain), encoding="utf-8")
    (folder / "game.json").write_text(json.dumps({"video": video, "title": title, "auto": True}, indent=1),
                                      encoding="utf-8")
    log.info("possession analysis of %s: %d events, %s (%.0f s)", video, len(derived), shots, time.time() - t0)
    return run


def _plain(o: Any) -> Any:
    return o.item() if hasattr(o, "item") else str(o)
