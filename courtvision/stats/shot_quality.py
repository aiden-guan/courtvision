"""Shot-quality (xFG) model: NBA priors as a log-odds offset, plus an optional refit on our shots.

Model
-----
For a field-goal attempt::

    logit P(make) = base(distance)            monotone piecewise-linear in log-odds through anchors
                  + corner                    corner-three shift (zone == "corner_three")
                  + defender(d)               closest-defender shift, calibrated to tracking splits
                  + catch_or_pull             catch-and-shoot / pull-up shift (jumpers, ramped 10-22 ft)
                  + shot_type                 dunk / layup / tip / hook shift
                  + league                    non-NBA placeholder scaling (unverified)
                  + coef . residual(f)        fitted by ``fit()``; zero until then

The first six terms are the *prior*; ``fit()`` is an L2-regularised logistic regression (Newton /
IRLS in numpy) on a few residual features with the prior as a fixed offset, so with little data the
model stays at the prior. Free throws are league FT%. Outputs are clipped to [0.01, 0.99].

Sources: ``docs/research/basketball_rules_stats.md`` (the "rules note") §4a-§4e and RESEARCH.md
§2.5. NBA zone and distance priors are sourced (2025-26 season). Defender-distance and
catch-and-shoot splits are the note's placeholder priors (§4d, unverified), and every non-NBA value
is an unverified placeholder: xeFG cards stay NBA-only (RESEARCH.md §7A.2). Each constant in
``ShotQualityConfig`` records its source and status (sourced / unverified / tuned).
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Sequence

import numpy as np

from courtvision.types import Event

__all__ = ["FEATURE_NAMES", "ShotFeatures", "ShotQualityConfig", "ShotQualityModel"]

log = logging.getLogger(__name__)

FEATURE_NAMES: tuple[str, ...] = (
    "intercept",     # 1
    "distance",      # (distance_ft - 15) / 10
    "three",         # 1 for a 3-point attempt
    "defender",      # log1p(min(d, 12)) - log1p(4); 0 when unknown (EPV uses log(1 + d) [45])
    "catch_shoot",   # +1 catch-and-shoot, -1 pull-up, 0 unknown; x the 10-22 ft ramp (jumpers only)
    "dunk",
    "layup",
    "tip",
    "hook",
    "corner_three",  # 1 for a corner three
)
_FT_TYPES = frozenset({"free_throw"})
_JSON_VERSION = 1


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic function (scalar or array)."""
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr)
    pos = z_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z_arr[pos]))
    ez = np.exp(z_arr[~pos])
    out[~pos] = ez / (1.0 + ez)
    return float(out) if out.ndim == 0 else out


# --------------------------------------------------------------------------- features


@dataclass
class ShotFeatures:
    """What the model needs to price one attempt (feet; ``value`` 1 = free throw)."""

    distance_ft: float
    value: int
    shot_type: str = "jumper"  # "jumper" is the reference: the league-average shot mix at that distance
    closest_defender_ft: float | None = None
    catch_and_shoot: bool | None = None
    zone: str | None = None  # CourtSpec.zone() name; only "corner_three" changes the prior

    @property
    def is_free_throw(self) -> bool:
        return self.value == 1 or self.shot_type in _FT_TYPES

    @property
    def defender_ft(self) -> float | None:
        """``closest_defender_ft`` when it is a finite number, else None (unknown)."""
        d = self.closest_defender_ft
        if d is None:
            return None
        try:
            d = float(d)
        except (TypeError, ValueError):
            return None
        return d if math.isfinite(d) else None

    @property
    def kind(self) -> str:
        """``shot_type`` with a missing value read as the reference "jumper"."""
        return self.shot_type or "jumper"

    @classmethod
    def from_event(cls, event: Event) -> ShotFeatures:
        """Features from a SHOT_ATTEMPT's well-known attrs (``types.py``); raises if distance is missing."""
        a = event.attrs
        if a.get("distance_ft") is None:
            raise ValueError(f"event {event.event_id} has no attrs['distance_ft']")
        cs = a.get("catch_and_shoot")
        return cls(
            distance_ft=float(a["distance_ft"]),
            value=int(event.value),
            shot_type=str(a.get("shot_type") or "jumper"),
            closest_defender_ft=None if a.get("closest_defender_ft") is None else float(a["closest_defender_ft"]),
            catch_and_shoot=None if cs is None else bool(cs),
            zone=a.get("zone"),
        )


# --------------------------------------------------------------------------- config


@dataclass
class ShotQualityConfig:
    """Priors and fitting settings. Plain fields so the integration layer can load them from YAML."""

    # (distance_ft, FG%) anchors for a generic NBA field-goal attempt. Interpolated linearly in
    # log-odds, flat beyond the first and last anchor, so the curve is monotone non-increasing.
    # Check (tests): with distance uniform inside each pbpstats zone the curve gives rim .668,
    # short mid .471 and long mid .411 against .673 / .444 / .412 (§4c); real attempts cluster
    # closer to the rim inside each bin, so bin-midpoint placement is a tuned compromise.
    distance_anchors: list[list[float]] = field(default_factory=lambda: [
        [1.5, 0.703],   # B-Ref 2025-26 "0-3 ft" FG% (rules note §4b); value sourced, placed at bin midpoint (tuned)
        [6.5, 0.463],   # B-Ref 2025-26 "3-10 ft" (§4b); sourced, bin midpoint (tuned)
        [13.0, 0.446],  # B-Ref 2025-26 "10-16 ft" (§4b); sourced, bin midpoint (tuned)
        [19.5, 0.401],  # B-Ref 2025-26 "16 ft - 3P" (§4b); sourced, midpoint of 16-23 ft (tuned)
        [26.0, 0.345],  # pbpstats 2025-26 above-the-break 3 (§4c; RESEARCH §2.5 sanity prior); sourced.
                        # Placed at ~26 ft: avg 3PA distance 25.56 ft with 24.6% corner threes at ~22.5 ft,
                        # less the heave tail (derived; unverified)
        [30.0, 0.300],  # deep three; unverified placeholder (no source in the notes)
        [40.0, 0.100],  # heave boundary (pbpstats heave = d > 40 ft, §4c); value unverified placeholder
        [50.0, 0.027],  # heaves 16/584 = 2.7% (pbpstats 2024-25, §4c); sourced value, placement unverified
    ])
    corner_three_fg: float = 0.387  # corner-3 FG% 2025-26, pbpstats and B-Ref agree (§4b/§4c); sourced
    corner_three_ref_ft: float = 22.5  # typical corner-3 distance (line at 22 ft [1]); unverified
    # Closest-defender buckets (NBA CloseDefDistRange 0-2/2-4/4-6/6+ ft; strings unverified, §4d).
    # The open-ended 6+ bucket is capped at 12 ft for calibration (unverified).
    defender_bucket_edges_ft: list[float] = field(default_factory=lambda: [0.0, 2.0, 4.0, 6.0, 12.0])
    # 3P% by closest defender, midpoints of the note's ranges .29-.31/.33-.35/.36-.37/.38-.39 (§4d);
    # unverified placeholders ("replace before shipping").
    defender_bucket_fg3: list[float] = field(default_factory=lambda: [0.30, 0.34, 0.365, 0.385])
    reference_fg3: float = 0.360  # NBA 3P% 2025-26, official/B-Ref headline value (§4a, §4c conflict note); sourced
    defender_scale_two: float = 1.0  # the 3P-calibrated shift applied to 2s at this scale; unverified
    defender_cap_ft: float = 12.0  # distances beyond this are treated as this (wide open); unverified
    catch_shoot_fg3: float = 0.375  # catch-and-shoot 3P% ~.37-.38 (§4d); unverified placeholder
    pull_up_fg3: float = 0.335  # pull-up 3P% ~.33-.34 (§4d); unverified placeholder
    # Catch/pull splits apply to jumpers from this distance: NBA GeneralRange puts "Less Than 10 ft"
    # in its own category apart from "Catch and Shoot" / "Pull Ups" (§4d; strings unverified).
    catch_shoot_min_ft: float = 10.0
    # ... and ramp linearly to full strength here (the splits are 3P%). The ramp is gentler than the
    # baseline's decline, so catch-and-shoot xFG stays monotone in distance; unverified.
    catch_shoot_full_ft: float = 22.0
    # Log-odds shift by shot type relative to the generic mix at that distance. No FG%-by-type
    # source exists in the notes; all unverified. dunk +1.5 puts an at-rim dunk near .91; layup is
    # set so a 22% dunk share (6.1% dunks of 27.7% at-rim FGA, §4b/§4c) keeps the <4 ft mix near .673.
    shot_type_logit: dict[str, float] = field(default_factory=lambda: {
        "jumper": 0.0,
        "layup": -0.25,
        "dunk": 1.5,
        "tip": -0.5,
        "hook": 0.1,
    })
    # League free-throw percentage. NBA .783 (2025-26, §4a; sourced). NCAA D1 men ~.70-.72 (§4e;
    # unverified). FIBA, NCAA women and NFHS: no source in the notes (unverified placeholders).
    ft_pct: dict[str, float] = field(default_factory=lambda: {
        "nba": 0.783,
        "fiba": 0.75,
        "ncaa_m": 0.71,
        "ncaa_w": 0.71,
        "nfhs": 0.62,
    })
    # Log-odds scaling of every FG prior per league. NBA 0 (the priors are NBA). NCAA men -0.10
    # ~ 3P% 34% vs NBA 36% (logit gap -0.088, §4e brief value). Others: unverified placeholders.
    league_logit_offset: dict[str, float] = field(default_factory=lambda: {
        "nba": 0.0,
        "fiba": -0.05,
        "ncaa_m": -0.10,
        "ncaa_w": -0.30,
        "nfhs": -0.40,
    })
    min_prob: float = 0.01  # output clip (spec)
    max_prob: float = 0.99
    fit_max_iter: int = 100  # Newton iterations
    fit_tol: float = 1e-8  # stop when the largest coefficient step is below this


# --------------------------------------------------------------------------- calibration helpers


def _calibrate_defender_shift(edges: Sequence[float], targets: Sequence[float], ref_p: float,
                              grid: int = 81, iters: int = 500, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Knots (bucket centres) and log-odds values of a piecewise-linear defender shift.

    Solved so that, for a reference shot with P = ``ref_p``, the mean P over each bucket (distance
    uniform within the bucket) equals that bucket's target. Jacobi iteration: each bucket mean is
    dominated by its own knot, so the update converges.
    """
    e = np.asarray(edges, dtype=float)
    tgt = np.asarray(targets, dtype=float)
    if e.ndim != 1 or len(e) != len(tgt) + 1 or np.any(np.diff(e) <= 0):
        raise ValueError("defender_bucket_edges_ft must be increasing with one more entry than the targets")
    centers = (e[:-1] + e[1:]) / 2.0
    grids = [np.linspace(lo, hi, grid) for lo, hi in zip(e[:-1], e[1:])]
    base = _logit(ref_p)
    tgt_logit = np.log(tgt / (1.0 - tgt))
    values = tgt_logit - base
    for _ in range(iters):
        means = np.array([np.mean(_sigmoid(base + np.interp(g, centers, values))) for g in grids])
        step = tgt_logit - np.log(means / (1.0 - means))
        values = values + step
        if np.max(np.abs(step)) < tol:
            break
    return centers, values


# --------------------------------------------------------------------------- model


class ShotQualityModel:
    """xFG prior from NBA league data, optionally refit (residual logistic regression) on our shots."""

    def __init__(self, league: str = "nba", config: ShotQualityConfig | None = None) -> None:
        self.config = config or ShotQualityConfig()
        cfg = self.config
        if league not in cfg.ft_pct or league not in cfg.league_logit_offset:
            raise ValueError(f"unknown league {league!r}; known: {sorted(cfg.ft_pct)}")
        self.league = league
        anchors = np.asarray(cfg.distance_anchors, dtype=float)
        if anchors.ndim != 2 or anchors.shape[1] != 2 or len(anchors) < 2:
            raise ValueError("distance_anchors must be a list of [distance_ft, fg_pct] pairs (>= 2)")
        if np.any(np.diff(anchors[:, 0]) <= 0):
            raise ValueError("distance_anchors distances must be strictly increasing")
        if np.any(np.diff(anchors[:, 1]) > 0) or np.any(anchors[:, 1] <= 0) or np.any(anchors[:, 1] >= 1):
            raise ValueError("distance_anchors FG% must lie in (0, 1) and be non-increasing with distance")
        self._anchor_d = anchors[:, 0]
        self._anchor_logit = np.log(anchors[:, 1] / (1.0 - anchors[:, 1]))
        self._corner_shift = _logit(cfg.corner_three_fg) - self._base_logit(cfg.corner_three_ref_ft)
        self._def_knots, self._def_values = _calibrate_defender_shift(
            cfg.defender_bucket_edges_ft, cfg.defender_bucket_fg3, cfg.reference_fg3)
        self._cs_shift = _logit(cfg.catch_shoot_fg3) - _logit(cfg.reference_fg3)
        self._pu_shift = _logit(cfg.pull_up_fg3) - _logit(cfg.reference_fg3)
        self.coef = np.zeros(len(FEATURE_NAMES))
        self.fit_info: dict[str, Any] = {}
        self._warned_types: set[str] = set()

    # ------------------------------------------------------------------ public API

    @property
    def prior_status(self) -> str:
        """``"sourced"`` for the NBA priors; every other league is a placeholder."""
        return "sourced" if self.league == "nba" else "unverified"

    def xfg(self, f: ShotFeatures) -> float:
        """P(make), clipped to [min_prob, max_prob]; free throws use the league FT%."""
        self._validate(f)
        if f.is_free_throw:
            p = self.config.ft_pct[self.league]
        else:
            p = float(_sigmoid(self.prior_logit(f) + float(self.coef @ self.residual_features(f))))
        return float(min(max(p, self.config.min_prob), self.config.max_prob))

    def expected_points(self, f: ShotFeatures) -> float:
        """xFG x points value (1, 2 or 3)."""
        return self.xfg(f) * int(f.value)

    def prior_logit(self, f: ShotFeatures) -> float:
        """Log-odds of the prior (the offset used by ``fit``); free throws -> logit(FT%)."""
        self._validate(f)
        cfg = self.config
        if f.is_free_throw:
            return _logit(cfg.ft_pct[self.league])
        d = max(float(f.distance_ft), 0.0)
        z = self._base_logit(d) + cfg.league_logit_offset[self.league]
        if f.value == 3 and f.zone == "corner_three":
            z += self._corner_shift
        d_def = f.defender_ft  # NaN / inf / garbage = unknown, never a NaN xFG
        if d_def is not None:
            scale = 1.0 if f.value == 3 else cfg.defender_scale_two
            z += scale * self.defender_shift(d_def)
        if f.kind == "jumper" and f.catch_and_shoot is not None:
            z += self._catch_ramp(d) * (self._cs_shift if f.catch_and_shoot else self._pu_shift)
        z += self._shot_type_shift(f.kind)
        return z

    def defender_shift(self, closest_defender_ft: float) -> float:
        """Calibrated log-odds shift for the closest-defender distance (monotone, flat outside the knots)."""
        d = min(max(float(closest_defender_ft), 0.0), self.config.defender_cap_ft)
        return float(np.interp(d, self._def_knots, self._def_values))

    def residual_features(self, f: ShotFeatures) -> np.ndarray:
        """Feature vector for the fitted residual (``FEATURE_NAMES``); zeros for free throws."""
        x = np.zeros(len(FEATURE_NAMES))
        if f.is_free_throw:
            return x
        cfg = self.config
        d = max(float(f.distance_ft), 0.0)
        x[0] = 1.0
        x[1] = (d - 15.0) / 10.0
        x[2] = 1.0 if f.value == 3 else 0.0
        d_def = f.defender_ft
        if d_def is not None:
            x[3] = math.log1p(min(max(d_def, 0.0), cfg.defender_cap_ft)) - math.log1p(4.0)
        if f.kind == "jumper" and f.catch_and_shoot is not None:
            x[4] = self._catch_ramp(d) * (1.0 if f.catch_and_shoot else -1.0)
        for i, name in enumerate(("dunk", "layup", "tip", "hook"), start=5):
            x[i] = 1.0 if f.kind == name else 0.0
        x[9] = 1.0 if (f.value == 3 and f.zone == "corner_three") else 0.0
        return x

    def fit(self, features: list[ShotFeatures], made: list[bool], l2: float = 1.0) -> None:
        """Refit the residual coefficients on labelled attempts (prior = fixed offset).

        L2-penalised logistic regression solved by Newton / IRLS with step halving. The penalty
        shrinks every coefficient (the intercept included) toward 0, i.e. toward the prior. Free
        throws and unlabelled attempts (``made is None``: a pending ``ShotRecord``) are skipped,
        never counted as misses. Replaces previous coefficients.
        """
        if len(features) != len(made):
            raise ValueError("features and made must have the same length")
        if l2 < 0:
            raise ValueError("l2 must be >= 0")
        n_unlabelled = sum(1 for m in made if m is None)
        rows = [(f, bool(m)) for f, m in zip(features, made) if m is not None and not f.is_free_throw]
        if not rows:
            raise ValueError("fit() needs at least one labelled field-goal attempt")
        X = np.array([self.residual_features(f) for f, _ in rows])
        offset = np.array([self.prior_logit(f) for f, _ in rows])
        y = np.array([m for _, m in rows], dtype=float)
        coef, iters, converged = self._newton(X, offset, y, float(l2))
        self.coef = coef
        self.fit_info = {
            "n": int(len(rows)),
            "n_free_throws_skipped": int(sum(1 for f, m in zip(features, made) if m is not None and f.is_free_throw)),
            "n_unlabelled_skipped": int(n_unlabelled),
            "l2": float(l2),
            "iterations": iters,
            "converged": converged,
            "logloss_prior": _logloss(offset, y),
            "logloss_fit": _logloss(offset + X @ coef, y),
        }
        if not converged:
            log.warning("shot model fit did not converge in %d iterations", iters)

    def to_json(self) -> dict[str, Any]:
        """JSON-safe dict (config, coefficients, fit info)."""
        return {
            "model": "ShotQualityModel",
            "version": _JSON_VERSION,
            "league": self.league,
            "config": asdict(self.config),
            "feature_names": list(FEATURE_NAMES),
            "coef": [float(c) for c in self.coef],
            "fit_info": dict(self.fit_info),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ShotQualityModel:
        if d.get("model", "ShotQualityModel") != "ShotQualityModel" or int(d.get("version", _JSON_VERSION)) != _JSON_VERSION:
            raise ValueError("not a ShotQualityModel v1 document")
        known = {f.name for f in fields(ShotQualityConfig)}
        raw = dict(d.get("config") or {})
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown ShotQualityConfig fields: {sorted(unknown)}")
        model = cls(str(d.get("league", "nba")), ShotQualityConfig(**raw))
        names = list(d.get("feature_names", FEATURE_NAMES))
        if names != list(FEATURE_NAMES):
            raise ValueError(f"feature_names mismatch: {names} != {list(FEATURE_NAMES)}")
        coef = np.asarray(d.get("coef", np.zeros(len(FEATURE_NAMES))), dtype=float)
        if coef.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(coef)):
            raise ValueError("coef must hold one finite number per feature")
        model.coef = coef
        model.fit_info = dict(d.get("fit_info") or {})
        return model

    # ------------------------------------------------------------------ internals

    def _catch_ramp(self, distance_ft: float) -> float:
        lo, hi = self.config.catch_shoot_min_ft, self.config.catch_shoot_full_ft
        if hi <= lo:
            return 1.0 if distance_ft >= lo else 0.0
        return min(max((distance_ft - lo) / (hi - lo), 0.0), 1.0)

    def _base_logit(self, distance_ft: float) -> float:
        return float(np.interp(distance_ft, self._anchor_d, self._anchor_logit))

    def _shot_type_shift(self, shot_type: str) -> float:
        table = self.config.shot_type_logit
        if shot_type in table:
            return float(table[shot_type])
        if shot_type not in self._warned_types:
            self._warned_types.add(shot_type)
            log.debug("unknown shot_type %r priced as the generic mix", shot_type)
        return 0.0

    @staticmethod
    def _validate(f: ShotFeatures) -> None:
        if f.value not in (1, 2, 3):  # None / 0 / 4 are caller bugs, reported as ValueError
            raise ValueError(f"shot value must be 1, 2 or 3, got {f.value!r}")
        try:
            finite = math.isfinite(float(f.distance_ft))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError(f"distance_ft must be a finite number, got {f.distance_ft!r}")

    def _newton(self, X: np.ndarray, offset: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, int, bool]:
        k = X.shape[1]
        b = np.zeros(k)
        ridge = l2 * np.eye(k) + 1e-9 * np.eye(k)  # tiny jitter keeps H invertible when l2 = 0
        loss = _penalised_nll(X, offset, y, b, l2)
        for it in range(1, self.config.fit_max_iter + 1):
            p = _sigmoid(offset + X @ b)
            grad = X.T @ (y - p) - l2 * b
            hess = (X * (p * (1.0 - p))[:, None]).T @ X + ridge
            step = np.linalg.solve(hess, grad)
            t = 1.0
            while True:  # step halving keeps the penalised loss non-increasing
                cand = b + t * step
                cand_loss = _penalised_nll(X, offset, y, cand, l2)
                if cand_loss <= loss + 1e-12 or t < 1e-6:
                    break
                t *= 0.5
            b, loss = cand, cand_loss
            if np.max(np.abs(t * step)) < self.config.fit_tol:
                return b, it, True
        return b, self.config.fit_max_iter, False


def _penalised_nll(X: np.ndarray, offset: np.ndarray, y: np.ndarray, b: np.ndarray, l2: float) -> float:
    z = offset + X @ b
    # log(1 + e^z) - y z, computed stably
    nll = np.sum(np.logaddexp(0.0, z) - y * z)
    return float(nll + 0.5 * l2 * float(b @ b))


def _logloss(z: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, z) - y * z))
