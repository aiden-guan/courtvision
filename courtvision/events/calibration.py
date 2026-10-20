"""Cue models: named CV cues -> event probabilities (RESEARCH.md §7A.2).

One small logistic model per gated event kind ("shot", "make", "three", "rebound", "steal",
"block", "player_id"). The rule engine computes named cues in Python; a model turns them into
``Event.confidence`` and the event carries ``attrs["p_status"]``:

* ``uncalibrated``: hand-set cold-start weights. Narration treats this as hedge-only.
* ``platt``: weights refit (L2 logistic) and Platt-scaled once >= 300 labelled events of the kind
  exist, at least 50 of them in the ambiguous band.
* ``isotonic``: from >= 1,000 labelled events (Platt is better below ~200-1,000 cases [480]).

Every cold-start weight is a placeholder, not a finding (§7A.2). Fitting is deterministic (fixed
fold assignment, Newton/IRLS), uses numpy only, and never touches the network.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "AMBIGUOUS_BAND",
    "GATED_KINDS",
    "STATUSES",
    "CalibrationConfig",
    "CueModel",
    "CueModelSet",
    "Feature",
    "default_cue_models",
    "ece",
]

log = logging.getLogger(__name__)

STATUSES: tuple[str, ...] = ("uncalibrated", "platt", "isotonic")
GATED_KINDS: tuple[str, ...] = ("shot", "make", "three", "rebound", "steal", "block", "player_id")
AMBIGUOUS_BAND: tuple[float, float] = (0.05, 0.95)  # B5 trigger band p in (0.05, 0.95) (§7A.2); sourced
_FORMAT_VERSION = 1
_TRANSFORMS = ("linear", "clip", "ge", "lt")
_Z_CLIP = 35.0  # logit clip so exp() never overflows


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -_Z_CLIP, _Z_CLIP)))


@dataclass
class CalibrationConfig:
    """Status-transition thresholds and fit settings (RESEARCH §7A.2)."""

    platt_min_labelled: int = 300  # labelled events before Platt (§7A.2); sourced (proposal)
    platt_min_ambiguous: int = 50  # of them in the ambiguous band (§7A.2); sourced (proposal)
    isotonic_min_labelled: int = 1000  # isotonic only above ~1000 samples [479]; sourced
    ambiguous_band: tuple[float, float] = AMBIGUOUS_BAND
    l2: float = 1.0  # ridge strength on standardised features; tuned
    cv_folds: int = 5  # out-of-fold scores for Platt / isotonic; tuned
    max_iter: int = 50  # Newton iterations; tuned


@dataclass
class Feature:
    """One named cue and how it enters the logit: ``weight * transform(cue)``.

    Transforms: ``linear`` (x / scale), ``clip`` (x / scale clipped to [lo, hi]), ``ge`` (1 when
    x >= threshold), ``lt`` (1 when x < threshold). A missing or non-finite cue uses ``default``
    (before the transform). Booleans count as 0/1.
    """

    name: str
    weight: float
    transform: str = "linear"
    scale: float = 1.0
    lo: float | None = None
    hi: float | None = None
    threshold: float = 0.0
    default: float = 0.0

    def __post_init__(self) -> None:
        if self.transform not in _TRANSFORMS:
            raise ValueError(f"feature {self.name}: unknown transform {self.transform!r}")
        if self.scale == 0:
            raise ValueError(f"feature {self.name}: scale must be non-zero")

    def value(self, cues: Mapping[str, Any]) -> float:
        raw = cues.get(self.name)
        try:
            x = self.default if raw is None else float(raw)
        except (TypeError, ValueError):
            x = self.default
        if not math.isfinite(x):
            x = self.default
        if self.transform == "ge":
            return 1.0 if x >= self.threshold else 0.0
        if self.transform == "lt":
            return 1.0 if x < self.threshold else 0.0
        x /= self.scale
        if self.transform == "clip":
            if self.lo is not None:
                x = max(x, self.lo)
            if self.hi is not None:
                x = min(x, self.hi)
        return x

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "weight": self.weight, "transform": self.transform, "scale": self.scale,
                "lo": self.lo, "hi": self.hi, "threshold": self.threshold, "default": self.default}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Feature:
        return cls(**{k: d[k] for k in ("name", "weight", "transform", "scale", "lo", "hi", "threshold", "default") if k in d})


class CueModel:
    """Logistic model over named cues for one gated event kind."""

    def __init__(
        self,
        kind: str,
        features: Sequence[Feature],
        bias: float,
        status: str = "uncalibrated",
        platt: tuple[float, float] | None = None,
        isotonic: tuple[list[float], list[float]] | None = None,
        n_labelled: int = 0,
        n_ambiguous: int = 0,
        config: CalibrationConfig | None = None,
    ) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}; expected one of {STATUSES}")
        names = [f.name for f in features]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate feature names in model {kind!r}")
        self.kind = kind
        self.features = list(features)
        self.bias = float(bias)
        self.status = status
        self.platt = platt  # (a, b): p = sigmoid(a * logit + b)
        self.isotonic = isotonic  # (x thresholds of raw p, calibrated y), non-decreasing
        self.n_labelled = int(n_labelled)
        self.n_ambiguous = int(n_ambiguous)
        self.config = config or CalibrationConfig()

    # ------------------------------------------------------------------ inference

    def vector(self, cues: Mapping[str, Any]) -> np.ndarray:
        return np.array([f.value(cues) for f in self.features], dtype=float)

    def logit(self, cues: Mapping[str, Any]) -> float:
        return self.bias + sum(f.weight * f.value(cues) for f in self.features)

    def _calibrate(self, z: np.ndarray) -> np.ndarray:
        if self.status == "platt" and self.platt is not None:
            a, b = self.platt
            return np.asarray(_sigmoid(a * z + b), dtype=float)
        p = np.asarray(_sigmoid(z), dtype=float)
        if self.status == "isotonic" and self.isotonic is not None:
            xs, ys = self.isotonic
            p = np.interp(p, xs, ys)
        return np.clip(p, 1e-6, 1.0 - 1e-6)

    def predict(self, cues: Mapping[str, Any]) -> float:
        """P(event is real / outcome is positive) for one set of cues."""
        return float(self._calibrate(np.array([self.logit(cues)]))[0])

    def predict_many(self, cue_list: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if not cue_list:
            return np.zeros(0)
        X = np.vstack([self.vector(c) for c in cue_list])
        w = np.array([f.weight for f in self.features])
        return self._calibrate(X @ w + self.bias)

    # ------------------------------------------------------------------ fitting

    def fit(self, cue_list: Sequence[Mapping[str, Any]], labels: Sequence[bool | int], l2: float | None = None) -> str:
        """Refit on labelled events and move the status forward when the data allows it.

        Below ``platt_min_labelled`` labelled events (or ``platt_min_ambiguous`` in the ambiguous
        band of the current model) nothing changes. Otherwise the weights are refit by L2 logistic
        regression, out-of-fold logits calibrate them (Platt, or isotonic from
        ``isotonic_min_labelled``), and the new status is returned.
        """
        cfg = self.config
        if len(cue_list) != len(labels):
            raise ValueError("cue_list and labels differ in length")
        y = np.array([1.0 if bool(v) else 0.0 for v in labels])
        n = len(y)
        if n == 0:
            return self.status
        lo, hi = cfg.ambiguous_band
        p_now = self.predict_many(cue_list)
        n_amb = int(np.sum((p_now > lo) & (p_now < hi)))
        if n < cfg.platt_min_labelled or n_amb < cfg.platt_min_ambiguous:
            log.info("cue model %s: %d labelled (%d ambiguous); stays %s", self.kind, n, n_amb, self.status)
            return self.status
        if y.min() == y.max():
            log.warning("cue model %s: all labels identical; not refit", self.kind)
            return self.status
        X = np.vstack([self.vector(c) for c in cue_list])
        lam = cfg.l2 if l2 is None else float(l2)
        w, b = _fit_logistic(X, y, lam, cfg.max_iter)
        oof = _out_of_fold_logits(X, y, lam, cfg.max_iter, cfg.cv_folds)
        self.bias = b
        for f, wi in zip(self.features, w, strict=True):
            f.weight = float(wi)
        self.n_labelled, self.n_ambiguous = n, n_amb
        if n >= cfg.isotonic_min_labelled:
            self.isotonic = _isotonic(np.asarray(_sigmoid(oof)), y)
            self.platt, self.status = None, "isotonic"
        else:
            self.platt = _fit_platt(oof, y, cfg.max_iter)
            self.isotonic, self.status = None, "platt"
        log.info("cue model %s refit on %d events -> %s", self.kind, n, self.status)
        return self.status

    # ------------------------------------------------------------------ serialisation

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _FORMAT_VERSION,
            "kind": self.kind,
            "bias": self.bias,
            "status": self.status,
            "features": [f.to_dict() for f in self.features],
            "platt": list(self.platt) if self.platt is not None else None,
            "isotonic": {"x": list(self.isotonic[0]), "y": list(self.isotonic[1])} if self.isotonic is not None else None,
            "n_labelled": self.n_labelled,
            "n_ambiguous": self.n_ambiguous,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], config: CalibrationConfig | None = None) -> CueModel:
        iso = d.get("isotonic")
        platt = d.get("platt")
        return cls(
            kind=str(d["kind"]),
            features=[Feature.from_dict(f) for f in d["features"]],
            bias=float(d["bias"]),
            status=str(d.get("status", "uncalibrated")),
            platt=(float(platt[0]), float(platt[1])) if platt else None,
            isotonic=([float(v) for v in iso["x"]], [float(v) for v in iso["y"]]) if iso else None,
            n_labelled=int(d.get("n_labelled", 0)),
            n_ambiguous=int(d.get("n_ambiguous", 0)),
            config=config,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path, config: CalibrationConfig | None = None) -> CueModel:
        return cls.from_dict(json.loads(Path(path).read_text()), config)


class CueModelSet:
    """The per-kind cue models used by the event engine."""

    def __init__(self, models: Mapping[str, CueModel] | None = None, assume_calibrated: bool = False) -> None:
        self.models: dict[str, CueModel] = dict(models) if models is not None else default_cue_models()
        self.assume_calibrated = assume_calibrated  # simulator evaluation only (ARCHITECTURE events/)

    def predict(self, kind: str, cues: Mapping[str, Any]) -> tuple[float, str]:
        """(probability, p_status); ``assume_calibrated`` reports ``"assumed_sim"``."""
        model = self.models.get(kind)
        if model is None:
            raise KeyError(f"no cue model for kind {kind!r}")
        p = model.predict(cues)
        return p, ("assumed_sim" if self.assume_calibrated else model.status)

    def status(self, kind: str) -> str:
        return "assumed_sim" if self.assume_calibrated else self.models[kind].status

    def to_dict(self) -> dict[str, Any]:
        return {"version": _FORMAT_VERSION, "models": {k: m.to_dict() for k, m in sorted(self.models.items())}}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], assume_calibrated: bool = False) -> CueModelSet:
        models = default_cue_models()
        for kind, md in (d.get("models") or {}).items():
            models[kind] = CueModel.from_dict(md)
        return cls(models, assume_calibrated=assume_calibrated)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path, assume_calibrated: bool = False) -> CueModelSet:
        return cls.from_dict(json.loads(Path(path).read_text()), assume_calibrated=assume_calibrated)


# ---------------------------------------------------------------------------- cold start


def default_cue_models() -> dict[str, CueModel]:
    """Hand-set cold-start models (placeholders; RESEARCH §7A.2). A textbook case scores high and
    any conflict between cues lands mid-range."""
    F = Feature
    specs: dict[str, tuple[float, list[Feature]]] = {
        # Shot attempt cues (§7A.2): release_sep_ft (0.35 m ≈ 1.15 ft [16]), upward_speed, parabola
        # fit quality and size, toward_rim_cos, in_frontcourt; plus entered_rim_zone (ours).
        "shot": (-2.0, [
            F("entered_rim_zone", 2.0),
            F("upward_speed", 0.8, "clip", scale=10.0, lo=0.0, hi=2.0),  # ft/s / 10
            F("toward_rim_cos", 1.5, "clip", lo=-1.0, hi=1.0),
            F("in_frontcourt", 0.5),
            F("parabola_n_pts", 0.5, "ge", threshold=5.0),  # >= 5 points (§7A.2, unverified)
            F("parabola_rmse_px", -0.5, "clip", scale=5.0, lo=0.0, hi=2.0),
            F("release_sep_ft", 0.3, "ge", threshold=1.15),
        ]),
        # Make (§7A.2 example, verbatim): −2.0 + 3.0·crossed_inside + 1.5·drop_in_band
        # − 2.5·drop_is_freefall + 0.8·min(net_motion_z, 3) + 1.0·bib_conf_max − 1.0·[occluded ≥ 3].
        # crossed_inside_3d (ours) is a measured 3-D crossing inside the cylinder.
        "make": (-2.0, [
            F("crossed_inside", 3.0),
            F("crossed_inside_3d", 4.0),
            F("drop_in_band", 1.5),
            F("drop_is_freefall", -2.5),
            F("net_motion_z", 0.8, "clip", hi=3.0),
            F("bib_conf_max", 1.0),
            F("occluded_frames", -1.0, "ge", threshold=3.0),
            F("rim_contact", 0.0),
            F("crossing_margin_px", 0.0),
            F("view_quality", 0.0),
        ]),
        # 3-point: signed foot margin beyond the line at the last grounded position (D8).
        "three": (0.0, [
            F("foot_margin_ft", 6.0, "clip", lo=-1.0, hi=1.0),
            F("foot_obs_quality", 0.0, default=1.0),
            F("homography_rmse_ft", 0.0),
            F("in_corner", 0.0),
        ]),
        # Rebound / steal / block: control duration, holder distance, track confidence, team margin.
        "rebound": (-0.5, [
            F("control_duration_s", 2.5, "clip", scale=0.5, lo=0.0, hi=1.0),
            F("holder_dist_ft", -0.3, "clip", lo=0.0, hi=3.0),
            F("track_conf", 1.0, default=1.0),
            F("team_margin", 1.0, "clip", lo=0.0, hi=1.0, default=1.0),
        ]),
        "steal": (-1.0, [
            F("control_duration_s", 2.5, "clip", scale=0.5, lo=0.0, hi=1.0),
            F("holder_dist_ft", -0.3, "clip", lo=0.0, hi=3.0),
            F("track_conf", 1.0, default=1.0),
            F("team_margin", 1.5, "clip", lo=0.0, hi=1.0, default=1.0),
        ]),
        "block": (-1.0, [
            F("reversal_dv", 1.5, "clip", scale=10.0, lo=0.0, hi=2.0),  # ft/s (or ft/s-equivalent) / 10
            F("defender_dist_ft", -0.5, "clip", lo=0.0, hi=4.0, default=4.0),
            F("above_shoulder", 1.0),
            F("track_conf", 1.0, default=1.0),
        ]),
        "player_id": (-3.0, [
            F("jersey_vote_share", 4.0, "clip", lo=0.0, hi=1.0),
            F("n_agreeing_reads", 0.3, "clip", lo=0.0, hi=5.0),
            F("roster_ok", 1.0),
            F("team_margin", 1.0, "clip", lo=0.0, hi=1.0),
        ]),
    }
    return {kind: CueModel(kind, feats, bias) for kind, (bias, feats) in specs.items()}


# ---------------------------------------------------------------------------- numerics


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float, max_iter: int) -> tuple[np.ndarray, float]:
    """L2 logistic regression by Newton/IRLS on standardised features; returns raw (w, b)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd
    n, d = Xs.shape
    Xb = np.hstack([Xs, np.ones((n, 1))])
    reg = np.eye(d + 1) * l2
    reg[-1, -1] = 0.0
    w = np.zeros(d + 1)
    for _ in range(max_iter):
        p = np.asarray(_sigmoid(Xb @ w))
        grad = Xb.T @ (p - y) + reg @ w
        hess = Xb.T @ (Xb * (p * (1.0 - p))[:, None]) + reg + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(hess, grad)
        w -= step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    ws, bs = w[:-1], float(w[-1])
    return ws / sd, bs - float(np.sum(ws * mu / sd))


def _out_of_fold_logits(X: np.ndarray, y: np.ndarray, l2: float, max_iter: int, folds: int) -> np.ndarray:
    n = len(y)
    k = max(2, min(folds, n))
    fold = np.arange(n) % k  # deterministic assignment
    oof = np.zeros(n)
    for f in range(k):
        test = fold == f
        train = ~test
        if y[train].min() == y[train].max():
            oof[test] = 0.0
            continue
        w, b = _fit_logistic(X[train], y[train], l2, max_iter)
        oof[test] = X[test] @ w + b
    return oof


def _fit_platt(z: np.ndarray, y: np.ndarray, max_iter: int) -> tuple[float, float]:
    """Platt scaling with Platt's smoothed targets; Newton on (a, b)."""
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    t = np.where(y > 0.5, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
    A = np.column_stack([z, np.ones_like(z)])
    ab = np.array([1.0, 0.0])
    for _ in range(max_iter):
        p = np.asarray(_sigmoid(A @ ab))
        grad = A.T @ (p - t)
        hess = A.T @ (A * (p * (1.0 - p))[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        ab -= step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return float(ab[0]), float(ab[1])


def _isotonic(p: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    """Pool-adjacent-violators on (p, y); returns block-mean knots for ``np.interp``.

    Tied scores are pooled first (one block per distinct p): cue models with binary features give
    many ties, and splitting a tie into blocks with different means would give ``np.interp`` a
    repeated knot x and an arbitrary calibrated value for that score.
    """
    order = np.argsort(p, kind="mergesort")
    xs, ys = p[order], y[order]
    blocks: list[list[float]] = []  # [sum_x, sum_y, count]
    uniq, start = np.unique(xs, return_index=True)
    ends = list(start[1:]) + [len(xs)]
    for xi, a, b in zip(uniq, start, ends, strict=True):
        n = float(b - a)
        blocks.append([float(xi) * n, float(ys[a:b].sum()), n])
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]:
            sx, sy, c = blocks.pop()
            blocks[-1][0] += sx
            blocks[-1][1] += sy
            blocks[-1][2] += c
    knots_x = [b[0] / b[2] for b in blocks]
    knots_y = [b[1] / b[2] for b in blocks]
    return knots_x, knots_y


def ece(p: Sequence[float], y: Sequence[bool | int], bins: int = 15) -> float:
    """Expected calibration error with equal-width bins (15 bins per §7A.2)."""
    pa = np.asarray(p, dtype=float)
    ya = np.asarray([1.0 if bool(v) else 0.0 for v in y])
    if len(pa) == 0:
        return 0.0
    idx = np.minimum((pa * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += float(m.sum()) * abs(float(pa[m].mean()) - float(ya[m].mean()))
    return total / len(pa)
