"""Referee or player, from a person box's colours: a logistic regression on shirt and trouser statistics.

The detector has one person class; the pipeline tells referees apart afterwards (``nba.analyze.is_referee``). Its
first rule (an achromatic, mid-bright shirt with dark stripes) misses about a quarter of the referees and calls
players in dark or grey jerseys referees. This classifier reads two parts of the box, the shirt (rows 20-50 %, the
middle half) and the trousers (rows 55-85 %, the middle 40 %), each resampled to a fixed size (``SHIRT_HW``,
``TROUSERS_HW``: the frame's resolution does not change the statistics): saturation and brightness statistics, the
share of dark, of bright grey and of coloured pixels, the coloured pixels' hues, and the brightness changes across
and down (stripes). All boxes of a frame are read at once (``referee_probs``).

Weights (``nba.referee_model``) come from ``tools/referee_classifier.py``, fitted on human labels of 8 NBA games
(Roboflow Universe "basketball-player-detection-2", CC BY 4.0; NBA footage, research use: docs/TRAINING.md §1).
Left out one game at a time it found 94.3 % of the referees against the rule's 73.9 % and called fewer players
referees (1,730 against 2,464 of 22,440): docs/research/cross_game_detector.md §11.

The in-game check (``fit_game``, ``game_referee_calls``; ``analyze.RefereeCheck("game")`` frame after frame) learns
nothing across games: it clusters the people of one game by these statistics and colour histograms of the same two
regions, and takes the smallest of three clusters for the referees (``tools/referee_game_check.py`` scores it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

SHIRT = (0.2, 0.5, 0.25, 0.75)  # rows from, to; columns from, to (shares of the box)
TROUSERS = (0.55, 0.85, 0.3, 0.7)
SHIRT_HW = (29, 20)  # each region resampled to this height and width before any statistic
TROUSERS_HW = (29, 16)
_STATS = ("s_median", "v_median", "v_p10", "v_p20", "v_p90", "dark", "bright_grey", "coloured", "grad_x", "grad_y",
          "hue_0", "hue_1", "hue_2", "hue_3", "hue_4", "hue_5")
FEATURE_NAMES = tuple(f"shirt_{n}" for n in _STATS) + tuple(f"trousers_{n}" for n in _STATS[:8])


def _region(frame_bgr: np.ndarray, box: tuple[float, float, float, float], part: tuple[float, float, float, float],
            hw: tuple[int, int]) -> np.ndarray | None:
    """``part`` of a box (shares of its height and width, the box clipped to the frame), resampled to ``hw``."""
    x1, y1, x2, y2 = box
    h, w = frame_bgr.shape[:2]
    xa, ya, xb, yb = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
    if xb - xa < 4 or yb - ya < 8:
        return None
    fy0, fy1, fx0, fx1 = part
    bw, bh = xb - xa, yb - ya
    r0, r1 = int(ya + fy0 * bh), max(int(ya + fy0 * bh) + 1, int(ya + fy1 * bh))
    c0, c1 = int(xa + fx0 * bw), max(int(xa + fx0 * bw) + 1, int(xa + fx1 * bw))
    return cv2.resize(frame_bgr[r0:r1, c0:c1], (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


def _quantiles(vals: np.ndarray, qs: tuple[float, ...]) -> np.ndarray:
    """``np.percentile(vals, q, axis=1)`` (the linear method) of N rows of 8-bit values for each ``q`` (100 * 0.5 is
    the median), from the rows' value counts: no sort. Returns (len(qs), N)."""
    n, px = vals.shape
    counts = np.bincount((vals.astype(np.int32) + 256 * np.arange(n, dtype=np.int32)[:, None]).ravel(),
                         minlength=256 * n).reshape(n, 256)
    cum = np.cumsum(counts, axis=1)

    def kth(k: int) -> np.ndarray:  # each row's k-th smallest value (from 0)
        return (cum <= k).sum(axis=1).astype(np.float64)

    out = []
    for q in qs:
        virtual = (px - 1) * (q / 100)
        if virtual >= px - 1:
            out.append(kth(px - 1))
            continue
        prev = math.floor(virtual)
        a, b, t = kth(prev), kth(prev + 1), virtual - prev
        out.append(b - (b - a) * (1 - t) if t >= 0.5 else a + (b - a) * t)
    return np.array(out)


def _stats(hsv: np.ndarray) -> np.ndarray:
    """The ``_STATS`` of one region of N boxes at once: ``hsv`` (N, rows, cols, 3), OpenCV's 8-bit HSV."""
    n = hsv.shape[0]
    flat = hsv.reshape(n, -1, 3)
    hh, s, v = flat[..., 0], flat[..., 1], flat[..., 2]
    px = flat.shape[1]
    col = s > 80
    hue = np.stack([np.count_nonzero(((hh // 30) == i) & col, axis=1) for i in range(6)], axis=1) / px
    sq = _quantiles(s, (50.0,))
    vq = _quantiles(v, (50.0, 10.0, 20.0, 90.0))
    vi = hsv[..., 2].astype(np.int16)
    gx = np.abs(np.diff(vi, axis=2)).mean(axis=(1, 2)) / 255
    gy = np.abs(np.diff(vi, axis=1)).mean(axis=(1, 2)) / 255
    return np.column_stack([sq[0] / 255, vq[0] / 255, vq[1] / 255, vq[2] / 255, vq[3] / 255,
                            np.count_nonzero(v < 60, axis=1) / px, np.count_nonzero((v > 160) & (s < 60), axis=1) / px,
                            np.count_nonzero(col, axis=1) / px, gx, gy, hue])


def _regions_hsv(frame_bgr: np.ndarray, boxes: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """The shirt and trouser regions of the readable boxes as OpenCV 8-bit HSV, (N, rows, cols, 3) each, and the
    boxes' indices; None when no box can be read (under 4 x 8 px inside the frame)."""
    shirts, trousers, keep = [], [], []
    for i, box in enumerate(boxes):
        sh = _region(frame_bgr, box, SHIRT, SHIRT_HW)
        if sh is None:
            continue
        shirts.append(sh)
        trousers.append(_region(frame_bgr, box, TROUSERS, TROUSERS_HW))
        keep.append(i)
    if not keep:
        return None

    def hsv(crops: list[np.ndarray], hw: tuple[int, int]) -> np.ndarray:
        stack = np.stack(crops)
        return cv2.cvtColor(stack.reshape(-1, hw[1], 3), cv2.COLOR_BGR2HSV).reshape(stack.shape)

    return hsv(shirts, SHIRT_HW), hsv(trousers, TROUSERS_HW), np.array(keep, int)


def referee_features_batch(frame_bgr: np.ndarray, boxes: Any) -> tuple[np.ndarray, np.ndarray]:
    """The ``FEATURE_NAMES`` of several person boxes of one frame (frame px) at once: (features of the readable
    boxes, their indices in ``boxes``); a box under 4 x 8 px inside the frame cannot be read."""
    regions = _regions_hsv(frame_bgr, boxes)
    if regions is None:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0, int)
    shirts, trousers, keep = regions
    return np.column_stack([_stats(shirts), _stats(trousers)[:, :8]]), keep


def referee_features(frame_bgr: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    """The ``FEATURE_NAMES`` of one person box (frame px), or None for a box under 4 x 8 px inside the frame."""
    feats, keep = referee_features_batch(frame_bgr, [box])
    return feats[0] if len(keep) else None


def referee_probs(frame_bgr: np.ndarray, boxes: Any, model: Any = None) -> np.ndarray:
    """``referee_prob`` of several boxes of one frame at once (NaN for a box too small to read)."""
    if model is None:
        from courtvision.nba import referee_model as model
    out = np.full(len(boxes), np.nan)
    feats, keep = referee_features_batch(frame_bgr, boxes)
    if len(keep):
        z = (feats - np.asarray(model.MEAN)) / np.asarray(model.STD)
        out[keep] = 1.0 / (1.0 + np.exp(-(z @ np.asarray(model.COEF) + float(model.BIAS))))
    return out


def referee_prob(frame_bgr: np.ndarray, box: tuple[float, float, float, float], model: Any = None) -> float | None:
    """The probability that a person box shows a referee (``model``: a module or object with ``MEAN``, ``STD``,
    ``COEF``, ``BIAS``; default ``nba.referee_model``), or None when the box is too small to tell."""
    p = float(referee_probs(frame_bgr, [box], model)[0])
    return None if np.isnan(p) else p


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 300
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """L2-regularised logistic regression on standardised features, the two classes weighted equally (so that 0.5 is
    the even point); Newton steps. Returns (mean, std, coefficients, bias)."""
    y = np.asarray(y, bool)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z1 = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.zeros(Z1.shape[1])
    sw = np.where(y, (len(y) - y.sum()) / max(1, y.sum()), 1.0)
    reg = np.diag(np.r_[np.ones(len(w) - 1), 0.0])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Z1 @ w))
        grad = Z1.T @ (sw * (p - y)) / len(y) + l2 * (reg @ w)
        hess = (Z1 * (sw * p * (1 - p))[:, None]).T @ Z1 / len(y) + l2 * reg
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return mu, sd, w[:-1], float(w[-1])


def referee_calls(frame_bgr: np.ndarray, boxes: Any, method: str = "shirt", model: Any = None) -> list[bool]:
    """Which person boxes of one frame show a referee: ``shirt`` (the default: ``is_referee_shirt``) or ``learned``
    (``referee_probs`` at the weights' threshold; ``model``: another fit, else the packaged ``nba.referee_model``; a
    box too small to read is not a referee). One entry point for the desktop pipeline (``analyze.are_referees``),
    the Studio and the browser app (``StudentConfig.referees``, ``browser.runtime``). The shirt rule is the default:
    the learned check reads the Cavaliers' black jerseys on the README's 2017 game as referees' shirts
    (``docs/research/cross_game_detector.md`` §10.10-10.12)."""
    if method == "shirt":
        return [is_referee_shirt(frame_bgr, b) for b in boxes]
    if method != "learned":
        raise ValueError(f"unknown referee method {method!r}")
    if model is None:
        from courtvision.nba import referee_model as model
    probs = referee_probs(frame_bgr, boxes, model) if len(boxes) else np.zeros(0)
    return [bool(p >= model.THRESHOLD) for p in probs]  # NaN (unreadable) compares False


# --------------------------------------------------------------------------- the in-game check

HIST_BINS = 29  # colour_histograms: black; four greys; 12 hues by 2 saturations
GAME_FEATURE_NAMES = FEATURE_NAMES + tuple(f"shirt_bin_{i}" for i in range(HIST_BINS)) \
    + tuple(f"trousers_bin_{i}" for i in range(HIST_BINS))


def colour_histograms(hsv: np.ndarray) -> np.ndarray:
    """The shares of each region's pixels in ``HIST_BINS`` colour bins, N regions at once (``hsv`` (N, rows, cols, 3),
    OpenCV's 8-bit HSV): black (V < 40); achromatic (S < 50) by brightness in four (V // 64); chromatic in 12 hues of
    30 degrees by two saturations (below 150 and from 150)."""
    n = hsv.shape[0]
    flat = hsv.reshape(n, -1, 3).astype(np.int32)
    h, s, v = flat[..., 0], flat[..., 1], flat[..., 2]
    b = np.where(v < 40, 0, np.where(s < 50, 1 + np.minimum(v // 64, 3), 5 + (h * 12 // 180) * 2 + (s >= 150)))
    counts = np.bincount((b + HIST_BINS * np.arange(n)[:, None]).ravel(), minlength=HIST_BINS * n)
    return counts.reshape(n, HIST_BINS) / flat.shape[1]


def game_features_batch(frame_bgr: np.ndarray, boxes: Any) -> tuple[np.ndarray, np.ndarray]:
    """The ``GAME_FEATURE_NAMES`` of several person boxes of one frame at once: the ``FEATURE_NAMES`` statistics and the
    ``colour_histograms`` of the same shirt and trouser regions. (Features of the readable boxes, their indices.)"""
    regions = _regions_hsv(frame_bgr, boxes)
    if regions is None:
        return np.zeros((0, len(GAME_FEATURE_NAMES))), np.zeros(0, int)
    shirts, trousers, keep = regions
    return np.column_stack([_stats(shirts), _stats(trousers)[:, :8], colour_histograms(shirts),
                            colour_histograms(trousers)]), keep


@dataclass(frozen=True)
class GameConfig:
    """Settings of the in-game check (``fit_game``, ``analyze.RefereeCheck("game")``), fixed on the 30 E-BARD
    training games before the held-out ones were scored (``tools/referee_game_check.py``)."""

    neighbours: int = 12  # the nearest-neighbour graph that is clustered; also the vote placing boxes outside a fit
    min_mean: float = 1.0  # accepted: the referee cluster's boxes a frame, over frames with min_people or more people,
    max_mean: float = 3.5  # in [min_mean, max_mean] (an NBA game has 3 referees; the guard's 3.5)
    min_people: int = 8
    min_frames: int = 10  # frames with min_people or more people needed to judge that size
    # and compact: its mean silhouette at least this (the E-BARD training games 0.16-0.39 on their 30 frames, 0.05 on
    # 10; boxes without structure 0.02 or less, which the size alone can pass)
    min_silhouette: float = 0.04
    max_fit: int = 1000  # at most this many boxes clustered (a seeded sample); the others placed by the vote
    seed: int = 0
    # RefereeCheck("game"), frame after frame
    observe_every: int = 30  # frames between two frames whose boxes join the context (1 s at 30 fps)
    window: int = 300  # the newest observed frames kept as the context (5 minutes)
    refit_every: int = 30  # observed frames between two fits; the first once this many are observed


@dataclass
class GameFit:
    """One fit of the in-game check: the clustered boxes, which of the three clusters holds the referees, and whether
    it passed (``reason`` says why not). ``predict`` places other boxes by their nearest clustered neighbours."""

    accepted: bool
    reason: str
    referee_cluster: int = -1
    sizes: tuple[int, ...] = ()  # boxes clustered in each of the three clusters
    per_frame: float = float("nan")  # the referee cluster's boxes a frame (frames with min_people or more)
    silhouette: float = float("nan")  # the referee cluster's mean silhouette among the clustered boxes
    parts: int = 0  # separate parts of the graph clustered (more than 3: not accepted)
    calls: np.ndarray = field(default_factory=lambda: np.zeros(0, bool), repr=False)  # each box in the referee cluster
    config: GameConfig = field(default_factory=GameConfig, repr=False)
    _scale: tuple[np.ndarray, ...] = field(default=(), repr=False)  # stats mean and std, histogram mean, block scales
    _points: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), repr=False)  # clustered boxes, standardised
    _labels: np.ndarray = field(default_factory=lambda: np.zeros(0, int), repr=False)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Whether each box (``GAME_FEATURE_NAMES`` rows; a row with NaN is unreadable) falls in the referee cluster:
        the cluster most of its ``neighbours`` nearest clustered boxes are in. All False for a fit not accepted."""
        features = np.asarray(features, float).reshape(-1, len(GAME_FEATURE_NAMES))
        out = np.zeros(len(features), bool)
        ok = np.isfinite(features).all(axis=1)
        if not self.accepted or not ok.any():
            return out
        out[ok] = _vote(self._points, self._labels, _standardise(features[ok], self._scale),
                        self.config.neighbours) == self.referee_cluster
        return out


def _standard_scale(X: np.ndarray) -> tuple[np.ndarray, ...]:
    """Per context: the statistics z-scored, the histograms square-rooted (Hellinger) and centred, then each block
    scaled to a mean squared norm of 1, so the two weigh the same."""
    n = len(FEATURE_NAMES)
    mu, sd = X[:, :n].mean(0), X[:, :n].std(0) + 1e-6
    hist = np.sqrt(np.clip(X[:, n:], 0.0, None))
    hmu = hist.mean(0)
    s1 = math.sqrt(float((((X[:, :n] - mu) / sd) ** 2).sum(1).mean())) + 1e-9
    s2 = math.sqrt(float(((hist - hmu) ** 2).sum(1).mean())) + 1e-9
    return mu, sd, hmu, np.array([s1, s2])


def _standardise(X: np.ndarray, scale: tuple[np.ndarray, ...]) -> np.ndarray:
    n = len(FEATURE_NAMES)
    mu, sd, hmu, s = scale
    return np.column_stack([(X[:, :n] - mu) / sd / s[0], (np.sqrt(np.clip(X[:, n:], 0.0, None)) - hmu) / s[1]])


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T, 0.0)


def _vote(points: np.ndarray, labels: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    """The label most of each row's ``k`` nearest ``points`` carry (the lowest label on a tie)."""
    k = min(k, len(points))
    near = np.argpartition(_sqdist(X, points), k - 1, axis=1)[:, :k]
    votes = np.stack([(labels[near] == j).sum(1) for j in range(int(labels.max()) + 1)], axis=1)
    return votes.argmax(1)


def _spectral3(D2: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, int]:
    """Three clusters of points given their squared distances ``D2``: spectral clustering (Ng, Jordan & Weiss 2002)
    of the symmetrised ``k``-nearest-neighbour graph with each point's own scale, its distance to the k-th neighbour
    (Zelnik-Manor & Perona 2004). A graph follows a team whose uniform looks different from the front and the back,
    where k-means splits that team and joins one half to the referees (a team in black). Also returns the graph's
    number of separate parts (eigenvalue 1's multiplicity): beyond three, the three eigenvectors are any of them."""
    from courtvision.perception.teams import kmeans

    n = len(D2)
    D2 = D2.copy()
    np.fill_diagonal(D2, np.inf)
    near = np.argpartition(D2, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    d2 = D2[rows, near.ravel()]
    sigma = np.sqrt(d2.reshape(n, k).max(1)) + 1e-9
    W = np.zeros((n, n))
    W[rows, near.ravel()] = np.exp(-d2 / (sigma[rows] * sigma[near.ravel()]))
    W = np.maximum(W, W.T)
    inv = 1.0 / np.sqrt(np.maximum(W.sum(1), 1e-12))
    vals, vecs = np.linalg.eigh(W * inv[:, None] * inv[None, :])
    U = vecs[:, -3:]
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    return kmeans(U, 3, np.random.default_rng(seed), n_init=8)[1], int((vals > 1.0 - 1e-9).sum())


def _silhouette(D2: np.ndarray, labels: np.ndarray, cluster: int) -> float:
    """The mean silhouette of one cluster's points (Rousseeuw 1987): (b - a) / max(a, b), with a the mean distance to
    the cluster's other points and b the mean distance to the nearest other cluster's points."""
    D = np.sqrt(D2)
    mine = labels == cluster
    if mine.sum() < 2 or mine.all():
        return float("nan")
    a = (D[np.ix_(mine, mine)].sum(1) - np.diag(D)[mine]) / (mine.sum() - 1)
    b = np.min([D[np.ix_(mine, labels == j)].mean(1) for j in np.unique(labels) if j != cluster], axis=0)
    return float(np.mean((b - a) / np.maximum(np.maximum(a, b), 1e-12)))


def fit_game(features: np.ndarray, frames: Any, config: GameConfig | None = None) -> GameFit:
    """The in-game referee check on the person boxes of one game, or of a window of it (``features``: one
    ``GAME_FEATURE_NAMES`` row per box, NaN for a box too small to read; ``frames``: each box's frame). In one game the
    players wear two uniforms and the 2-3 referees a third: the readable boxes (at most ``max_fit``, a seeded sample)
    are clustered in three (``_spectral3``), the two largest clusters are the teams and the smallest the referees. It
    is accepted only when that cluster holds ``min_mean`` to ``max_mean`` boxes a frame over the frames with
    ``min_people`` or more people (at least ``min_frames`` of them) and is compact (``min_silhouette``), and when the
    graph has at most three separate parts; else the caller keeps the shirt rule. The context decides, not one era's
    uniforms (research note ``cross_game_detector.md`` §10.10-10.12: the learned check takes the 2017 Cavaliers' black
    jerseys for referees' shirts)."""
    cfg = config or GameConfig()
    X = np.asarray(features, float).reshape(-1, len(GAME_FEATURE_NAMES))
    frames = np.asarray(frames)
    if len(frames) != len(X):
        raise ValueError(f"{len(X)} feature rows but {len(frames)} frames")
    ok = np.flatnonzero(np.isfinite(X).all(axis=1))
    if len(ok) < max(3 * cfg.neighbours, 3):
        return GameFit(False, f"{len(ok)} readable boxes", calls=np.zeros(len(X), bool), config=cfg)
    fit = ok
    if len(ok) > cfg.max_fit:
        fit = np.sort(np.random.default_rng(cfg.seed).choice(ok, cfg.max_fit, replace=False))
    scale = _standard_scale(X[fit])
    Z = _standardise(X[fit], scale)
    D2 = _sqdist(Z, Z)
    labels, parts = _spectral3(D2, min(cfg.neighbours, len(fit) - 1), cfg.seed)
    sizes = np.bincount(labels, minlength=3)
    ref = int(np.argmin(sizes))
    silhouette = _silhouette(D2, labels, ref)
    calls = np.zeros(len(X), bool)
    calls[fit] = labels == ref
    rest = np.setdiff1d(ok, fit)
    if len(rest):
        calls[rest] = _vote(Z, labels, _standardise(X[rest], scale), cfg.neighbours) == ref
    _, inverse, people = np.unique(frames, return_inverse=True, return_counts=True)
    busy = people[inverse] >= cfg.min_people
    n_busy = len(np.unique(frames[busy]))
    per_frame = float(calls[busy].sum() / n_busy) if n_busy else float("nan")
    if parts > 3:
        accepted, reason = False, f"the graph falls into {parts} parts"
    elif n_busy < cfg.min_frames:
        accepted, reason = False, f"{n_busy} frames with {cfg.min_people} or more people"
    elif not cfg.min_mean <= per_frame <= cfg.max_mean:
        accepted, reason = False, f"{per_frame:.2f} referees a frame"
    elif not silhouette >= cfg.min_silhouette:
        accepted, reason = False, f"silhouette {silhouette:.3f}"
    else:
        accepted, reason = True, "accepted"
    return GameFit(accepted, reason, ref, tuple(int(s) for s in sizes), per_frame, silhouette, parts, calls, cfg,
                   scale, Z, labels)


def track_votes(calls: Any, track_ids: Any) -> np.ndarray:
    """Per track, the majority of its boxes' calls (a tie is not a referee); a box without a track (None or below
    0) keeps its own call."""
    calls = np.asarray(calls, bool).copy()
    tracks = np.array([-1 if t is None else int(t) for t in track_ids], int)
    if len(tracks) != len(calls):
        raise ValueError(f"{len(calls)} calls but {len(tracks)} track ids")
    for t in np.unique(tracks[tracks >= 0]):
        mine = tracks == t
        calls[mine] = 2 * int(calls[mine].sum()) > int(mine.sum())
    return calls


def game_referee_calls(features: np.ndarray, frames: Any, fallback: Any, track_ids: Any = None,
                       config: GameConfig | None = None) -> tuple[np.ndarray, GameFit]:
    """Which boxes of a game (or window) show a referee, by the in-game check (``fit_game``): cluster membership when
    the fit is accepted, else ``fallback`` (the shirt rule's call of each box). With ``track_ids`` (one per box, None
    for a box without a track) a track takes the majority of its boxes' calls. Returns (calls, the fit)."""
    fit = fit_game(features, frames, config)
    calls = fit.calls.copy() if fit.accepted else np.asarray(fallback, bool).copy()
    if track_ids is not None:
        calls = track_votes(calls, track_ids)
    return calls, fit


def is_referee_shirt(frame_bgr: np.ndarray, box: tuple[float, float, float, float]) -> bool:
    """NBA referees wear grey shirts with black stripes: an achromatic, mid-bright torso with dark stripes (white
    and coloured jerseys fail it). Thresholds from torso statistics of on-court people on ESPN frames."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    crop = frame_bgr[int(y1 + 0.2 * bh):int(y1 + 0.5 * bh), int(x1 + 0.25 * bw):int(x2 - 0.25 * bw)]
    if crop.size < 30:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    s, v = _OrderStats(hsv[:, 1]), _OrderStats(hsv[:, 2])
    return bool(s.median() < 70 and v.median() < 175 and v.percentile(20) < 75)


class _OrderStats:
    """``np.median`` / ``np.percentile`` (linear) of 8-bit values from their counts: the same float64 results without
    sorting (``is_referee`` asks for three of them per person box)."""

    def __init__(self, values: np.ndarray) -> None:
        self.n = len(values)
        self.cum = np.cumsum(np.bincount(values, minlength=256))

    def kth(self, k: int) -> float:
        """The k-th smallest value (from 0)."""
        return float(np.searchsorted(self.cum, k, side="right"))

    def median(self) -> float:
        return (self.kth((self.n - 1) // 2) + self.kth(self.n // 2)) / 2.0  # numpy: the mean of the middle one or two

    def percentile(self, q: float) -> float:
        virtual = (self.n - 1) * (q / 100)  # numpy's "linear" method, step by step
        if virtual >= self.n - 1:
            return self.kth(self.n - 1)
        prev = math.floor(virtual)
        a, b, t = self.kth(prev), self.kth(prev + 1), virtual - prev
        return b - (b - a) * (1 - t) if t >= 0.5 else a + (b - a) * t
