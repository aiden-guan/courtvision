# CourtVision accuracy bench (heldout split)

`.venv/bin/python tools/accuracy_bench.py --modes truth3d,mono,video --split heldout --seconds 1200 --video-seconds 180 --workers 4 --json runs/bench/r4_heldout_ffe5129.json --md runs/bench/r4_heldout_ffe5129.md -q`

| mode | seeds | s/game | headline F1 (pooled) | seed mean ± sd | attr. F1 | FG make/miss | 2 vs 3 | made-FG F1 | rebound F1 / rebounder | assist F1 / label | box MAE | team pts err (mean / max) | score match | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| truth3d | 101,102,103,104 | 1200 | **0.970** | 0.976 ± 0.008 | 0.948 | 1.000 (n=215) | 1.000 (n=215) | 1.000 | 0.971 / 0.966 | 1.000 / 1.000 | 0.028 | 0.00 / 0 | 1.00 | 200 |
| mono | 101,102,103,104 | 1200 | **0.742** | 0.735 ± 0.049 | 0.640 | 0.986 (n=208) | 0.962 (n=208) | 0.975 | 0.727 / 0.728 | 0.814 / 0.776 | 0.260 | 0.00 / 0 | 1.00 | 375 |
| video | 101,102,103,104 | 180 | **0.650** | 0.762 ± 0.210 | 0.590 | 0.968 (n=31) | 1.000 (n=31) | 0.973 | 0.750 / 0.750 | 0.800 / 0.889 | 0.036 | 0.00 / 0 | 1.00 | 2753 |

| mode | headline 95% CI | supported-kind macro F1 [95% CI] | micro F1 [95% CI] | FP / min of core kinds without truth |
|---|---|---|---|---|
| truth3d | [0.960, 0.980] | 0.970 [0.960, 0.980] | 0.989 [0.983, 0.994] | none |
| mono | [0.696, 0.780] | 0.742 [0.696, 0.780] | 0.892 [0.881, 0.901] | none |
| video | [0.575, 0.934] | 0.650 [0.575, 0.934] | 0.912 [0.864, 0.961] | none |

Only pooled multi-seed numbers of >= 180-300 s games are comparable; a single-seed <= 60 s headline is not comparable (rare kinds with 0-15 truth events enter or leave its average).

Time trend (truth3d): pooled headline per non-cumulative 60 s window: 1.000, 1.000, 0.998, 0.999, 0.949, 0.825, 0.995, 0.997, 0.999, 0.935, 0.981, 0.998, 0.999, 0.999, 1.000, 1.000, 0.973, 0.998, 0.997, 1.000; spread 0.175 over 20 full windows (whole-run 95% CI [0.960, 0.980]).
Time trend (mono): pooled headline per non-cumulative 60 s window: 0.619, 0.830, 0.889, 0.691, 0.532, 0.639, 0.799, 0.730, 0.985, 0.814, 0.774, 0.972, 0.907, 0.880, 0.623, 0.512, 0.695, 0.600, 0.776, 0.841; spread 0.473 over 20 full windows (whole-run 95% CI [0.696, 0.780]).
Time trend (video): pooled headline per non-cumulative 60 s window: 0.660, 0.628, 0.912; spread 0.284 over 3 full windows (whole-run 95% CI [0.575, 0.934]).

## Per-kind F1 pooled over seeds (truth n in brackets, then the 95% seed-bootstrap interval; * = headline kind)

| kind | truth3d | mono | video |
|---|---|---|---|
| shot_attempt* | 0.991 [280] (0.99-0.99) | 0.918 [280] (0.88-0.94) | 0.935 [38] (0.86-0.99) |
| shot_made* | 1.000 [147] (1.00-1.00) | 0.969 [147] (0.95-0.99) | 0.978 [23] (0.92-1.00) |
| shot_missed* | 0.981 [133] (0.97-0.99) | 0.886 [133] (0.85-0.92) | 0.812 [15] (0.61-0.95) |
| rebound* | 0.971 [124] (0.95-0.98) | 0.727 [124] (0.66-0.79) | 0.750 [15] (0.52-0.88) |
| assist* | 1.000 [62] (1.00-1.00) | 0.814 [62] (0.72-0.90) | 0.800 [8] (0.80-0.80) |
| steal* | 0.963 [13] (0.89-1.00) | 0.500 [13] (0.34-0.70) | 0.000 [1] (0.00-0.00) |
| turnover* | 0.980 [25] (0.95-1.00) | 0.679 [25] (0.47-0.84) | 0.615 [5] (0.40-1.00) |
| block* | 0.857 [8] (0.80-1.00) | 0.267 [8] (0.00-0.40) | 0.000 [2] (0.00-0.00) |
| pass* | 0.991 [1095] (0.98-1.00) | 0.913 [1095] (0.91-0.92) | 0.957 [176] (0.93-0.98) |
| possession | 0.981 [243] (0.97-0.99) | 0.783 [243] (0.75-0.81) | 0.825 [38] (0.70-0.90) |
| foul | 1.000 [55] (1.00-1.00) | 1.000 [55] (1.00-1.00) | 1.000 [7] (1.00-1.00) |
| substitution | 1.000 [61] (1.00-1.00) | 1.000 [61] (1.00-1.00) | — |
| timeout | 1.000 [3] (1.00-1.00) | 1.000 [3] (1.00-1.00) | — |
| period_start | 1.000 [8] (1.00-1.00) | 1.000 [8] (1.00-1.00) | 1.000 [4] (1.00-1.00) |
| period_end | 1.000 [4] (1.00-1.00) | 1.000 [4] (1.00-1.00) | — |

Tracking (video, pooled): MOTA 0.971, IDF1 0.963, ID switches 593, identity accuracy 0.976 (seed mean), RMSE 0.30 ft.
