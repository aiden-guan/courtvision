"""Perception pipeline: one BGR frame -> ``FrameState``.

Per frame: detect -> track players/referees -> team (k-means + per-track votes) -> jersey reads
(every ``jersey_every`` frames per track, skipped while a nearer person covers the number) -> court
position of each box (``foot_model``: an upright cylinder seen through the camera recovered from the floor
homography, ``footpoint.py``; or the bottom centre pushed ``foot_radius_ft``) -> drop
coasting tracks that duplicate a freshly matched same-team track (a restart; two established tracks on
one spot are a merge, and the hidden player's track is kept) -> identity ->
per-track court-space alpha-beta smoothing with velocity (speed clamp 30 ft/s) -> ball tracker ->
rims (detector RIM class or ``calibration.rims``) -> drop players more than 6 ft outside the court
-> optional scoreboard read at ``scoreboard_hz`` (5 Hz, D14).

``FrameState.scoreboard`` is set only on frames with a fresh read, so the downstream
``ScoreboardConfirmer`` sees 5 Hz readings rather than one reading repeated at frame rate.
Sub-component settings are plain dicts of field overrides (YAML-friendly) turned into their
dataclasses here; unknown keys raise ``ValueError``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, fields
from typing import Any, TypeVar

import numpy as np

from courtvision.geometry.court import CourtSpec, court_for
from courtvision.geometry.homography import Calibration, Homography
from courtvision.types import (
    Detection,
    FrameState,
    ObjClass,
    PlayerState,
    RimObservation,
    Roster,
    ScoreboardReading,
    Side,
    Team,
    Track,
)

from .appearance import AppearanceConfig, torso_feature
from .ball import BallTracker, BallTrackerConfig
from .detector import Detector
from .footpoint import FootModel, FootModelConfig
from .identity import IdentityConfig, IdentityManager
from .jersey import JerseyReader, NullJerseyReader
from .scoreboard_reader import ScoreboardReader, ScoreboardROI
from .teams import TeamClassifier, TeamConfig, TeamVoter
from .tracker import ByteTracker, TrackerConfig

__all__ = ["PerceptionConfig", "PerceptionPipeline"]

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class PerceptionConfig:
    """Pipeline settings. Sub-configs are dicts of field overrides for their dataclasses."""

    tracker_backend: str = "bytetrack"  # "bytetrack" (ours, numpy) | "trackers" (TrackersAdapter)
    tracker: dict[str, Any] = field(default_factory=dict)  # TrackerConfig overrides
    trackers_adapter: dict[str, Any] = field(default_factory=dict)  # TrackersAdapterConfig overrides
    ball: dict[str, Any] = field(default_factory=dict)  # BallTrackerConfig overrides
    team: dict[str, Any] = field(default_factory=dict)  # TeamConfig overrides
    identity: dict[str, Any] = field(default_factory=dict)  # IdentityConfig overrides
    appearance: dict[str, Any] = field(default_factory=dict)  # AppearanceConfig overrides
    compute_embeddings: bool = True  # torso_feature for detections without an embedding; design
    team_observe_every: int = 3  # frames between warm-up samples per track; tuned
    jersey_every: int = 5  # read each track's number every 5 frames; sourced (Roboflow, cv_pipeline §1.1)
    jersey_min_height_px: float = 20.0  # legibility gate; tuned
    # Occlusion legibility gate on the number band (bbox rows 20-60 %, optionally widened each side
    # for number overflow): skip the read when a nearer person covers more than jersey_occluder_near
    # of the band, or anyone covers more than jersey_occluder_any of it (1.0 = off). Swept on sim
    # renders (10 windows, 3 seeds): roster-id accuracy 0.847 without the gate, 0.878 with these; tuned.
    jersey_occluder_near: float = 0.30
    jersey_occluder_any: float = 1.0
    jersey_band_widen: float = 0.0
    smoothing_alpha: float = 0.6  # alpha-beta position gain; tuned
    smoothing_beta: float = 0.15  # alpha-beta velocity gain; tuned
    max_speed_ft_s: float = 30.0  # speed clamp (fastest sprints are ~30 ft/s); spec
    reset_jump_ft: float = 8.0  # a measurement this far from the prediction restarts the filter; tuned
    court_margin_ft: float = 6.0  # players farther outside the court are dropped; spec
    foot_radius_ft: float = 0.0  # move the bbox-bottom point this far away from the camera (body
    #   centre vs front edge of the feet); 0 = raw bottom-centre; tuned per camera
    # "cylinder": a box is the image of an upright cylinder (foot_radius_ft, body_height_ft); the camera
    # behind the floor homography gives the perspective lean of off-centre figures (``footpoint.py``).
    # On sim train seeds the naive foot point ("push": bottom centre + foot_radius_ft towards image-up)
    # is biased by ~0.5 ft; median court error on perceived boxes 0.54 -> 0.12 ft; design (geometry).
    foot_model: str = "cylinder"  # "cylinder" (perspective-correct, footpoint.py) | "push" (naive); design (geometry)
    body_height_ft: float = 6.5  # cylinder height; NBA average ~6 ft 6 in; sourced (league average)
    foot_direct: bool = False  # evaluate the foot correction per box (FootModelConfig.direct; broadcast: a new homography each frame)
    coast_confidence_decay: float = 0.8  # per frame for tracks reported while lost; tuned
    duplicate_iou: float = 0.5  # a coasting track overlapping a matched same-team track this much ...
    duplicate_radius_ft: float = 1.5  # ... or this close on the court is a stale duplicate; tuned
    # ... but only when one of the two is new (a restart of the coasting track, or a short fragment).
    # Two established tracks are a same-team merge: one player went behind the other (one colour blob)
    # and must keep its track to come out under it again; deleting it handed the hidden player's
    # detection to the merged track (identity swap). Frames; tuned on sim train seeds.
    duplicate_new_frames: int = 30
    rim_source: str = "auto"  # "auto" (calibration if it has rims, else detector) | "detector" | "calibration"
    scoreboard_roi: dict[str, Any] | None = None  # ScoreboardROI.to_dict() form
    scoreboard_hz: float = 5.0  # OCR rate; sourced (RESEARCH.md D14)
    mask_scoreboard: bool = True  # tell the detector to ignore the panel; design
    fallback_dt_s: float = 1.0 / 30.0  # when t does not advance; design
    track_memory_frames: int = 300  # forget per-track team votes / filters after this; tuned


@dataclass
class _AlphaBeta:
    x: np.ndarray
    v: np.ndarray
    t: float


class PerceptionPipeline:
    """``process(frame_bgr, frame_idx, t) -> FrameState`` (see module docstring)."""

    def __init__(
        self,
        detector: Detector,
        calibration: Calibration,
        roster: Roster | None = None,
        config: PerceptionConfig | None = None,
        jersey_reader: JerseyReader | None = None,
        tracker: Any | None = None,
        team_classifier: TeamClassifier | None = None,
    ) -> None:
        self.config = config or PerceptionConfig()
        cfg = self.config
        self.detector = detector
        self.calibration = calibration
        self.roster = roster
        self.court: CourtSpec = court_for(calibration.league)
        self.appearance = _sub(AppearanceConfig, cfg.appearance)
        self.tracker = tracker if tracker is not None else self._make_tracker()
        self.ball_tracker = BallTracker(_sub(BallTrackerConfig, cfg.ball))
        self.teams = team_classifier if team_classifier is not None else self._make_team_classifier()
        self.team_votes = TeamVoter(self.teams.config.vote_momentum)
        self.identity = IdentityManager(roster, _sub(IdentityConfig, cfg.identity))
        self.jersey_reader: JerseyReader = jersey_reader or NullJerseyReader()
        self._filters: dict[int, _AlphaBeta] = {}
        self._last_seen: dict[int, int] = {}
        self._suppressed: set[int] = set()  # stale duplicate tracks, never reported again
        self._scoreboard: ScoreboardReader | None = None
        self._scoreboard_mask: list[list[float]] | None = None  # polygon this pipeline added to the detector
        self._last_read_t: float | None = None
        self._oriented: tuple[Homography, Homography] | None = None  # (source, oriented copy)
        self._foot: tuple[Homography, tuple[float, float], FootModel] | None = None
        if cfg.scoreboard_roi is not None:
            self.set_scoreboard_roi(ScoreboardROI.from_dict(cfg.scoreboard_roi))

    # ------------------------------------------------------------------ public

    def set_scoreboard_roi(self, roi: ScoreboardROI | dict[str, Any] | None) -> None:
        """Install, replace, or remove (None) the venue scoreboard ROI; masks it in the detector.

        A dict may be ``ScoreboardROI.to_dict()`` output or the ``POST /api/scoreboard_roi`` body
        (image-pixel cell boxes, ``ScoreboardROI.from_dict``). Invalid input raises ``ValueError``.
        """
        if isinstance(roi, dict):
            roi = ScoreboardROI.from_dict(roi)
        remove_mask = getattr(self.detector, "remove_mask_polygon", None)
        if self._scoreboard_mask is not None and callable(remove_mask):
            remove_mask(self._scoreboard_mask)  # a replaced ROI must not keep hiding its old region
        self._scoreboard_mask = None
        if roi is None:
            self._scoreboard = None
            return
        self._scoreboard = ScoreboardReader(roi)
        self._last_read_t = None
        add_mask = getattr(self.detector, "add_mask_polygon", None)
        if self.config.mask_scoreboard and callable(add_mask):
            add_mask(roi.corners)
            self._scoreboard_mask = [list(p) for p in roi.corners]

    def reset_tracks(self) -> None:
        """Forget image-space tracking state after a cut (a broadcast replay, a close-up): player tracks end
        (new ones get new ids; identities re-link by jersey and team), the ball track and the court-space
        smoothing restart. Team colours and jersey memory are kept."""
        reset = getattr(self.tracker, "reset", None)
        if callable(reset):
            reset()
        self.ball_tracker.reset()
        self._filters.clear()
        self._last_seen.clear()

    def warp_image_state(self, M: np.ndarray) -> None:
        """Camera-motion compensation before a frame: image-space track and ball states move by the image ->
        image homography ``M`` from the previous frame to this one (broadcast pan / zoom)."""
        warp = getattr(self.tracker, "warp", None)
        if callable(warp):
            warp(M)
        self.ball_tracker.warp(M)

    def process(self, frame_bgr: np.ndarray, frame_idx: int, t: float) -> FrameState:
        """Run perception on one BGR frame."""
        h, w = frame_bgr.shape[:2]
        fd = self.detector.detect(frame_bgr, frame_idx, t)
        people = [d for d in fd.detections if d.cls in (ObjClass.PLAYER, ObjClass.REFEREE)]
        self._embed(frame_bgr, people)
        tracks = [tr for tr in self._track(people, frame_idx, frame_bgr, t) if tr.track_id not in self._suppressed]
        players = [tr for tr in tracks if tr.cls is ObjClass.PLAYER]
        teams = self._update_teams(players, frame_idx)
        self._read_jerseys(frame_bgr, players, tracks, frame_idx)
        raw_xy = self._court_points(players)
        players = self._drop_stale_duplicates(players, teams, raw_xy)
        ids = self.identity.assign(players, teams, frame_idx, court_xy=raw_xy, t=t)
        states = self._player_states(players, ids, raw_xy, teams, t)
        ball = self.ball_tracker.update(fd.of(ObjClass.BALL), frame_idx, t, self._floor(),
                                        people=[tr.bbox for tr in tracks if tr.time_since_update == 0])
        self._forget_old(frame_idx, {tr.track_id for tr in tracks})
        balls = fd.of(ObjClass.BALL)
        return FrameState(
            frame_idx=frame_idx,
            t=t,
            players=states,
            ball=ball,
            rims=self._rims(fd.of(ObjClass.RIM)),
            scoreboard=self._read_scoreboard(frame_bgr, t),
            image_size=(w, h),
            balls=np.array([((b.bbox.x1 + b.bbox.x2) / 2.0, (b.bbox.y1 + b.bbox.y2) / 2.0, b.score) for b in balls],
                           np.float32).reshape(-1, 3),
        )

    # ------------------------------------------------------------------ construction

    def _make_tracker(self) -> Any:
        cfg = self.config
        if cfg.tracker_backend == "bytetrack":
            return ByteTracker(_sub(TrackerConfig, cfg.tracker))
        if cfg.tracker_backend == "trackers":
            from .trackers_adapter import TrackersAdapter, TrackersAdapterConfig  # lazy: extra `cv`

            return TrackersAdapter(_sub(TrackersAdapterConfig, cfg.trackers_adapter))
        raise ValueError(f"unknown tracker_backend {cfg.tracker_backend!r}")

    def _make_team_classifier(self) -> TeamClassifier:
        tc = _sub(TeamConfig, self.config.team)
        colours = getattr(self.detector, "team_colors", None)
        if isinstance(colours, dict):
            if tc.home_color_bgr is None and Team.HOME in colours:
                tc.home_color_bgr = tuple(colours[Team.HOME])  # type: ignore[assignment]
            if tc.away_color_bgr is None and Team.AWAY in colours:
                tc.away_color_bgr = tuple(colours[Team.AWAY])  # type: ignore[assignment]
        return TeamClassifier(tc, self.appearance)

    # ------------------------------------------------------------------ stages

    def _embed(self, frame: np.ndarray, people: list[Detection]) -> None:
        if not self.config.compute_embeddings:
            return
        for d in people:
            if d.embedding is None:
                d.embedding = torso_feature(frame, d.bbox, self.appearance)

    def _track(self, people: list[Detection], frame_idx: int, frame: np.ndarray, t: float) -> list[Track]:
        if isinstance(self.tracker, ByteTracker):
            return self.tracker.update(people, frame_idx)
        return self.tracker.update(people, frame_idx, frame=frame, t=t)

    def _update_teams(self, players: list[Track], frame_idx: int) -> dict[int, tuple[Team | None, float]]:
        cfg = self.config
        for tr in players:
            if tr.time_since_update != 0 or tr.embedding is None:
                continue
            if (frame_idx + tr.track_id) % max(1, cfg.team_observe_every) == 0:
                self.teams.observe(tr.embedding)
            team, conf = self.teams.predict(tr.embedding)
            self.team_votes.add(tr.track_id, team, conf)
        if (not self.teams.is_fitted and self.teams.n_samples >= self.teams.config.min_fit_samples) \
                or self.teams.refit_due:
            self.teams.fit()
        return {tr.track_id: self.team_votes.get(tr.track_id) for tr in players}

    def _read_jerseys(self, frame: np.ndarray, players: list[Track], people: list[Track], frame_idx: int) -> None:
        cfg = self.config
        if isinstance(self.jersey_reader, NullJerseyReader):
            return
        for tr in players:
            if tr.time_since_update != 0 or tr.bbox.h < cfg.jersey_min_height_px:
                continue
            if (frame_idx + tr.track_id) % max(1, cfg.jersey_every) != 0:
                continue
            if self._number_occluded(tr, people):
                continue
            number, conf = self.jersey_reader.read(frame, tr.bbox)
            if number is not None:
                tr.jersey, tr.jersey_conf = number, conf

    def _number_occluded(self, tr: Track, people: list[Track]) -> bool:
        """Legibility gate: another person covers the number band (nearer: a little; anyone: a lot)."""
        cfg = self.config
        b = tr.bbox
        widen = cfg.jersey_band_widen * b.w
        band = (b.x1 - widen, b.y1 + 0.2 * b.h, b.x2 + widen, b.y1 + 0.6 * b.h)
        area = (band[2] - band[0]) * (band[3] - band[1])
        if area <= 0:
            return True
        for other in people:
            if other.track_id == tr.track_id:
                continue
            o = other.bbox
            iw = min(band[2], o.x2) - max(band[0], o.x1)
            ih = min(band[3], o.y2) - max(band[1], o.y1)
            if iw <= 0 or ih <= 0:
                continue
            frac = iw * ih / area
            nearer = o.y2 > b.y2  # lower foot point = closer to an elevated camera
            if frac > cfg.jersey_occluder_any or (nearer and frac > cfg.jersey_occluder_near):
                return True
        return False

    def _floor(self) -> Homography:
        """``calibration.homography`` with its sign fixed so floor points in front of the camera have
        w > 0. ``front_only`` mapping needs that orientation, which ``Homography.fit`` and
        ``PinholeCamera.floor_homography`` produce but a hand-built or OpenCV-normalised matrix
        (any scale, same mapping) may lack: unoriented, every player would map to nan."""
        src = self.calibration.homography
        if self._oriented is None or self._oriented[0] is not src:
            w_img, h_img = self.calibration.image_size
            w = float(src.H[2] @ np.array([w_img / 2.0, h_img - 1.0, 1.0]))  # bottom centre: floor
            oriented = src if w > 0 else Homography(-src.H)
            if oriented is not src:
                log.warning("calibration homography maps the floor to w < 0; using its negation")
            self._oriented = (src, oriented)
        return self._oriented[1]

    def _court_points(self, players: list[Track]) -> dict[int, tuple[float, float]]:
        if not players:
            return {}
        cfg = self.config
        if cfg.foot_model == "cylinder":
            xy = self._foot_model().court_points(np.array([tr.bbox.as_tuple() for tr in players], dtype=float))
        elif cfg.foot_model == "push":
            H = self._floor()
            feet = np.array([tr.bbox.foot for tr in players], dtype=float)
            xy = H.image_to_court(feet, front_only=True)
            if cfg.foot_radius_ft > 0:
                xy = self._push_from_camera(feet, xy)
        else:
            raise ValueError(f"unknown foot_model {cfg.foot_model!r}")
        return {tr.track_id: (float(p[0]), float(p[1])) for tr, p in zip(players, xy) if np.all(np.isfinite(p))}

    def _foot_model(self) -> FootModel:
        """The cylinder foot model of the current calibration (rebuilt when it or the body changes)."""
        H = self._floor()
        body = (float(self.config.foot_radius_ft), float(self.config.body_height_ft))
        if self._foot is None or self._foot[0] is not H or self._foot[1] != body:
            fm = FootModel(H, self.calibration.image_size, (self.court.length, self.court.width),
                           FootModelConfig(radius_ft=body[0], height_ft=body[1], direct=self.config.foot_direct))
            self._foot = (H, body, fm)
        return self._foot[2]

    def _drop_stale_duplicates(
        self,
        players: list[Track],
        teams: dict[int, tuple[Team | None, float]],
        raw_xy: dict[int, tuple[float, float]],
    ) -> list[Track]:
        """Drop coasting tracks that sit on a matched track of the same team: the tracker lost the
        association and restarted the player, so the old track must hand its identity over. Only
        when one of the two tracks is new: two established tracks on one spot are a merge (one
        player hidden behind a teammate), and the hidden player's track is kept."""
        cfg = self.config
        matched = [tr for tr in players if tr.time_since_update == 0]
        new = cfg.duplicate_new_frames

        def duplicate(c: Track) -> bool:
            team = teams.get(c.track_id, (None, 0.0))[0]
            for m in matched:
                if team is None or teams.get(m.track_id, (None, 0.0))[0] is not team:
                    continue
                if c.hits > new and m.age > new:
                    continue  # merge, not a restart
                if c.bbox.iou(m.bbox) >= cfg.duplicate_iou:
                    return True
                a, b = raw_xy.get(c.track_id), raw_xy.get(m.track_id)
                if a is not None and b is not None and math.hypot(a[0] - b[0], a[1] - b[1]) <= cfg.duplicate_radius_ft:
                    return True
            return False

        keep = []
        for tr in players:
            if tr.time_since_update > 0 and duplicate(tr):
                self._suppressed.add(tr.track_id)
                remove = getattr(self.tracker, "remove", None)
                if callable(remove):
                    remove(tr.track_id)
                log.debug("dropped stale duplicate track %d", tr.track_id)
            else:
                keep.append(tr)
        return keep

    def _push_from_camera(self, feet: np.ndarray, xy: np.ndarray) -> np.ndarray:
        """Shift court points by ``foot_radius_ft`` away from the camera: the bbox bottom is the
        front edge of the feet. "Away" is the court direction of moving up the image."""
        up = self._floor().image_to_court(feet - np.array([0.0, 1.0]), front_only=True)
        d = up - xy
        n = np.linalg.norm(d, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            step = np.where(n > 1e-9, d / n, 0.0) * self.config.foot_radius_ft
        return xy + np.nan_to_num(step)

    def _player_states(
        self,
        players: list[Track],
        ids: dict[int, tuple[str, Team | None, str | None]],
        raw_xy: dict[int, tuple[float, float]],
        teams: dict[int, tuple[Team | None, float]],
        t: float,
    ) -> list[PlayerState]:
        cfg = self.config
        out: list[PlayerState] = []
        for tr in players:
            if tr.track_id not in raw_xy or tr.track_id not in ids:
                continue
            (x, y), (vx, vy) = self._smooth(tr, raw_xy[tr.track_id], t)
            if not self.court.in_bounds(x, y, margin=cfg.court_margin_ft):
                continue
            pid, team, jersey = ids[tr.track_id]
            conf = tr.score * cfg.coast_confidence_decay**tr.time_since_update
            out.append(PlayerState(player_id=pid, team=team, x=x, y=y, vx=vx, vy=vy, jersey=jersey,
                                   track_id=tr.track_id, bbox=tr.bbox, confidence=float(np.clip(conf, 0.0, 1.0))))
        return out

    def _smooth(
        self, tr: Track, z_xy: tuple[float, float], t: float
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        cfg = self.config
        z = np.asarray(z_xy, dtype=float)
        f = self._filters.get(tr.track_id)
        if f is None:
            f = _AlphaBeta(z.copy(), np.zeros(2), t)
            self._filters[tr.track_id] = f
            return (float(z[0]), float(z[1])), (0.0, 0.0)
        dt = t - f.t if t > f.t else cfg.fallback_dt_s
        pred = f.x + f.v * dt
        if tr.time_since_update > 0:
            f.x = pred
        else:
            r = z - pred
            if float(np.hypot(*r)) > cfg.reset_jump_ft:
                f.x, f.v = z.copy(), np.zeros(2)
            else:
                f.x = pred + cfg.smoothing_alpha * r
                f.v = f.v + (cfg.smoothing_beta / dt) * r
        speed = float(np.hypot(*f.v))
        if speed > cfg.max_speed_ft_s:
            f.v *= cfg.max_speed_ft_s / speed
        f.t = t
        return (float(f.x[0]), float(f.x[1])), (float(f.v[0]), float(f.v[1]))

    def _rims(self, rim_dets: list[Detection]) -> list[RimObservation]:
        src = self.config.rim_source
        calibrated = self.calibration.rim_observations()
        if src == "calibration" or (src == "auto" and calibrated):
            return calibrated
        if src not in ("auto", "detector", "calibration"):
            raise ValueError(f"unknown rim_source {src!r}")
        return self._detected_rims(rim_dets)

    def _detected_rims(self, rim_dets: list[Detection]) -> list[RimObservation]:
        if not rim_dets:
            return []
        mid = self.calibration.homography.court_to_image(np.array([self.court.length / 2.0, self.court.width / 2.0]))
        u_mid = float(mid[0]) if np.all(np.isfinite(mid)) else self.calibration.image_size[0] / 2.0
        best: dict[Side, Detection] = {}
        for d in rim_dets:
            side: Side = "left" if d.bbox.center[0] < u_mid else "right"
            if side not in best or d.score * d.bbox.area > best[side].score * best[side].bbox.area:
                best[side] = d
        return [RimObservation(side=s, bbox=best[s].bbox, confidence=best[s].score)
                for s in ("left", "right") if s in best]

    def _read_scoreboard(self, frame: np.ndarray, t: float) -> ScoreboardReading | None:
        if self._scoreboard is None:
            return None
        period = 1.0 / self.config.scoreboard_hz if self.config.scoreboard_hz > 0 else math.inf
        if self._last_read_t is not None and t - self._last_read_t < period - 1e-6:
            return None
        self._last_read_t = t
        return self._scoreboard.read(frame, t)

    def _forget_old(self, frame_idx: int, live: set[int]) -> None:
        for tid in live:
            self._last_seen[tid] = frame_idx
        stale = [tid for tid, seen in self._last_seen.items() if frame_idx - seen > self.config.track_memory_frames]
        for tid in stale:
            self._last_seen.pop(tid, None)
            self._filters.pop(tid, None)
            self.team_votes.forget(tid)


def _sub(cls: type[T], overrides: dict[str, Any] | None) -> T:
    """Build dataclass ``cls`` from a dict of field overrides; unknown keys raise ValueError."""
    overrides = dict(overrides or {})
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} field(s): {unknown}")
    for k, v in list(overrides.items()):
        if isinstance(v, list) and k.endswith(("_bgr", "_wh", "_x", "_y")):
            overrides[k] = tuple(v)  # YAML lists -> tuples for colour / range fields
    return cls(**overrides)
