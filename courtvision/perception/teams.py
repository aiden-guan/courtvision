"""Team assignment: numpy k-means (k=2) on appearance embeddings, per-track voting, optional SigLIP.

The core classifier clusters player embeddings (torso colour histograms by default) with k-means++
and a fixed seed, then binds the two clusters to HOME/AWAY with colour anchors when they are
configured; without anchors cluster 0 is HOME (the permutation is arbitrary, cv_pipeline.md §4.1).
Before ``fit`` the anchors alone classify (``predict_from_anchors``).

``SiglipEmbedder`` is the roboflow/sports recipe's embedding step (``google/siglip-base-patch16-224``,
mean-pooled ``last_hidden_state`` of a central crop; cv_pipeline.md §1.1, §4.1). It lives behind
the ``team`` extra, is imported lazily, is off by default and never downloads unless
``local_files_only=False``. ``TeamConfig.reduce_dims`` (numpy PCA) stands in for UMAP(3).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from courtvision.types import BBox, Team

from .appearance import AppearanceConfig, color_feature

__all__ = ["SiglipEmbedder", "TeamClassifier", "TeamConfig", "TeamVoter", "kmeans"]

log = logging.getLogger(__name__)

_EPS = 1e-12


@dataclass
class TeamConfig:
    """Team clustering settings."""

    home_color_bgr: tuple[int, int, int] | None = None  # jersey colour anchor; config / roster
    away_color_bgr: tuple[int, int, int] | None = None
    seed: int = 0  # k-means++ seed; design (determinism)
    n_init: int = 4  # k-means restarts, best inertia kept; tuned
    max_iter: int = 100  # tuned
    tol: float = 1e-6  # centroid shift to stop; tuned
    min_fit_samples: int = 30  # warm-up samples before fit() succeeds; tuned
    max_samples: int = 3000  # newest samples kept for fitting; tuned
    normalize: bool = True  # L2-normalise embeddings (k-means ~ cosine); design
    reduce_dims: int | None = None  # PCA dims before k-means (UMAP(3) stand-in for SigLIP); design
    max_distance: float | None = None  # farther than this from both centroids => no team (coaches); tuned
    predict_from_anchors: bool = True  # before fit(), classify by the nearest colour anchor; design
    vote_momentum: float = 0.9  # per-track vote decay in TeamVoter (~10-frame memory); tuned


def kmeans(
    X: np.ndarray, k: int, rng: np.random.Generator, n_init: int = 4, max_iter: int = 100, tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, float]:
    """Lloyd's k-means with k-means++ seeding; returns (centroids, labels, inertia) of the best run."""
    X = np.asarray(X, dtype=float)
    if len(X) < k:
        raise ValueError(f"k-means needs >= {k} samples, got {len(X)}")
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for _ in range(max(1, n_init)):
        C = _kmeans_pp(X, k, rng)
        for _it in range(max_iter):
            d = _sqdist(X, C)
            labels = np.argmin(d, axis=1)
            new_C = C.copy()
            for j in range(k):
                members = X[labels == j]
                if len(members):
                    new_C[j] = members.mean(axis=0)
                else:  # empty cluster: re-seed at the worst-fitted point
                    new_C[j] = X[int(np.argmax(d[np.arange(len(X)), labels]))]
            shift = float(np.linalg.norm(new_C - C))
            C = new_C
            if shift <= tol:
                break
        d = _sqdist(X, C)
        labels = np.argmin(d, axis=1)
        inertia = float(d[np.arange(len(X)), labels].sum())
        if best is None or inertia < best[2] - 1e-12:
            best = (C, labels, inertia)
    assert best is not None
    return best


def _kmeans_pp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    C = [X[int(rng.integers(len(X)))]]
    for _ in range(1, k):
        d = np.min(_sqdist(X, np.array(C)), axis=1)
        total = float(d.sum())
        idx = int(rng.integers(len(X))) if total <= 0 else int(rng.choice(len(X), p=d / total))
        C.append(X[idx])
    return np.array(C, dtype=float)


def _sqdist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    return np.maximum(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2), 0.0)


class TeamClassifier:
    """``observe(embedding)`` during warm-up, ``fit()``, then ``predict(embedding)``."""

    def __init__(
        self,
        config: TeamConfig | None = None,
        appearance: AppearanceConfig | None = None,
        anchors: dict[Team, np.ndarray] | None = None,
    ) -> None:
        self.config = config or TeamConfig()
        self._samples: deque[np.ndarray] = deque(maxlen=self.config.max_samples)
        self._centroids: np.ndarray | None = None  # rows ordered [HOME, AWAY] in the working space
        self._mean: np.ndarray | None = None
        self._basis: np.ndarray | None = None
        if anchors is not None:
            self._anchors = {team: np.asarray(v, dtype=float) for team, v in anchors.items()}
        else:
            self._anchors = {}
            if self.config.home_color_bgr is not None:
                self._anchors[Team.HOME] = color_feature(self.config.home_color_bgr, appearance).astype(float)
            if self.config.away_color_bgr is not None:
                self._anchors[Team.AWAY] = color_feature(self.config.away_color_bgr, appearance).astype(float)

    @property
    def is_fitted(self) -> bool:
        return self._centroids is not None

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    def observe(self, embedding: np.ndarray | None) -> None:
        """Add one player embedding to the warm-up buffer (zero / non-finite vectors are ignored)."""
        v = self._prep(embedding)
        if v is not None:
            self._samples.append(v)

    def fit(self) -> bool:
        """Cluster the buffered samples into two teams. Returns False when there are too few."""
        cfg = self.config
        if len(self._samples) < max(2, cfg.min_fit_samples):
            return False
        X = np.array(self._samples)
        self._mean, self._basis = None, None
        if cfg.reduce_dims is not None and cfg.reduce_dims < X.shape[1]:
            self._mean = X.mean(axis=0)
            _, _, vt = np.linalg.svd(X - self._mean, full_matrices=False)
            self._basis = vt[: cfg.reduce_dims].T
        Z = self._project(X)
        C, labels, inertia = kmeans(Z, 2, np.random.default_rng(cfg.seed), cfg.n_init, cfg.max_iter, cfg.tol)
        self._centroids = C[self._cluster_order(C)]
        sizes = np.bincount(labels, minlength=2).tolist()
        log.info("team k-means fitted on %d samples (inertia %.4f, sizes %s)", len(X), inertia, sizes)
        return True

    def predict(self, embedding: np.ndarray | None) -> tuple[Team | None, float]:
        """Team and confidence in [0, 1]; (None, 0.0) when unknown."""
        v = self._prep(embedding)
        if v is None:
            return None, 0.0
        if self._centroids is not None:
            C = self._centroids
            z = self._project(v[None, :])[0]
        elif self.config.predict_from_anchors and len(self._anchors) == 2:
            C = np.array([self._prep_anchor(Team.HOME), self._prep_anchor(Team.AWAY)])
            z = v
        else:
            return None, 0.0
        d = np.sqrt(_sqdist(z[None, :], C)[0])
        i = int(np.argmin(d))
        if self.config.max_distance is not None and d[i] > self.config.max_distance:
            return None, 0.0
        conf = float((d[1 - i] - d[i]) / (d[1 - i] + d[i] + _EPS))
        return (Team.HOME if i == 0 else Team.AWAY), float(np.clip(conf, 0.0, 1.0))

    # ------------------------------------------------------------------ internals

    def _prep(self, embedding: np.ndarray | None) -> np.ndarray | None:
        if embedding is None:
            return None
        v = np.asarray(embedding, dtype=float).ravel()
        n = float(np.linalg.norm(v))
        if not np.all(np.isfinite(v)) or n <= _EPS:
            return None
        return v / n if self.config.normalize else v

    def _prep_anchor(self, team: Team) -> np.ndarray:
        v = self._prep(self._anchors[team])
        assert v is not None
        return v

    def _project(self, X: np.ndarray) -> np.ndarray:
        if self._basis is None or self._mean is None:
            return X
        return (X - self._mean) @ self._basis

    def _cluster_order(self, C: np.ndarray) -> list[int]:
        """Row order [HOME, AWAY]: by colour anchors when both exist, else cluster index."""
        if len(self._anchors) != 2:
            if self._anchors:
                team = next(iter(self._anchors))
                za = self._project(self._prep_anchor(team)[None, :])[0]
                near = int(np.argmin(np.linalg.norm(C - za, axis=1)))
                return [near, 1 - near] if team is Team.HOME else [1 - near, near]
            return [0, 1]
        A = self._project(np.array([self._prep_anchor(Team.HOME), self._prep_anchor(Team.AWAY)]))
        straight = np.linalg.norm(C[0] - A[0]) + np.linalg.norm(C[1] - A[1])
        swapped = np.linalg.norm(C[1] - A[0]) + np.linalg.norm(C[0] - A[1])
        return [0, 1] if straight <= swapped else [1, 0]


class TeamVoter:
    """Per-track team hysteresis: decayed, confidence-weighted votes (a track attribute, not a
    per-frame label; cv_pipeline.md §4.1)."""

    def __init__(self, momentum: float = 0.9) -> None:
        self.momentum = momentum
        self._score: dict[int, np.ndarray] = {}  # track -> [home, away]
        self._weight: dict[int, float] = {}

    def add(self, track_id: int, team: Team | None, confidence: float) -> None:
        vote = np.zeros(2)
        if team is not None:
            vote[0 if team is Team.HOME else 1] = float(np.clip(confidence, 0.0, 1.0))
        m = self.momentum
        self._score[track_id] = m * self._score.get(track_id, np.zeros(2)) + vote
        self._weight[track_id] = m * self._weight.get(track_id, 0.0) + 1.0

    def get(self, track_id: int) -> tuple[Team | None, float]:
        s = self._score.get(track_id)
        if s is None or float(s.max()) <= 0.0:
            return None, 0.0
        i = int(np.argmax(s))
        conf = (s[i] - s[1 - i]) / max(self._weight[track_id], _EPS)
        return (Team.HOME if i == 0 else Team.AWAY), float(np.clip(conf, 0.0, 1.0))

    def forget(self, track_id: int) -> None:
        self._score.pop(track_id, None)
        self._weight.pop(track_id, None)

    def track_ids(self) -> list[int]:
        return list(self._score)


class SiglipEmbedder:
    """SigLIP image embeddings of central player crops (extra ``team``; lazy, default off).

    ``model`` / ``processor`` can be injected (tests, or a model loaded elsewhere). Otherwise the
    first ``embed`` call loads ``model_id`` with ``local_files_only`` (no download by default).
    """

    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        device: str | None = None,
        local_files_only: bool = True,
        batch_size: int = 32,  # roboflow/sports batch size; sourced (cv_pipeline.md §1.1)
        crop_frac: tuple[float, float] = (0.5, 0.5),  # central (w, h) fraction kept; tuned
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.local_files_only = local_files_only
        self.batch_size = batch_size
        self.crop_frac = crop_frac
        self._model = model
        self._processor = processor

    def embed(self, frame_bgr: np.ndarray, bboxes: list[BBox]) -> np.ndarray:
        """(N, D) float32 embeddings, one per box (zero rows for empty crops)."""
        import torch  # lazy: extra `team`

        self._ensure_loaded()
        crops, keep = [], []
        for i, b in enumerate(bboxes):
            crop = self._central_crop(frame_bgr, b)
            if crop is not None:
                crops.append(crop[:, :, ::-1].copy())  # BGR -> RGB
                keep.append(i)
        rows: list[np.ndarray] = []
        with torch.no_grad():
            for s in range(0, len(crops), self.batch_size):
                inputs = self._processor(images=crops[s : s + self.batch_size], return_tensors="pt")
                if self.device is not None:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                hidden = self._model(**inputs).last_hidden_state
                rows.append(hidden.mean(dim=1).float().cpu().numpy())
        dim = rows[0].shape[1] if rows else 0
        out = np.zeros((len(bboxes), dim), dtype=np.float32)
        if rows:
            out[keep] = np.concatenate(rows, axis=0)
        return out

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        from transformers import AutoProcessor, SiglipVisionModel  # lazy: extra `team`

        log.info("loading SigLIP %s (local_files_only=%s)", self.model_id, self.local_files_only)
        self._processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=self.local_files_only)
        model = SiglipVisionModel.from_pretrained(self.model_id, local_files_only=self.local_files_only)
        if self.device is not None:
            model = model.to(self.device)
        self._model = model.eval()

    def _central_crop(self, frame: np.ndarray, b: BBox) -> np.ndarray | None:
        cx, cy = b.center
        hw, hh = b.w * self.crop_frac[0] / 2.0, b.h * self.crop_frac[1] / 2.0
        h_img, w_img = frame.shape[:2]
        x1, x2 = max(0, int(cx - hw)), min(w_img, int(np.ceil(cx + hw)))
        y1, y2 = max(0, int(cy - hh)), min(h_img, int(np.ceil(cy + hh)))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]
