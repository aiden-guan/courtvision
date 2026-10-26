# CourtVision accuracy bench (heldout split)

`.venv/bin/python tools/accuracy_bench.py --modes truth3d,mono,video --split heldout --seconds 1200 --video-seconds 180 --workers 3 --json runs/bench/round5-heldout.json --md docs/bench/round5-heldout.md`

| mode | seeds | s/game | headline F1 (pooled) | seed mean ± sd | attr. F1 | FG make/miss | 2 vs 3 | made-FG F1 | rebound F1 / rebounder | assist F1 / label | box MAE | team pts err (mean / max) | score match | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| truth3d | 101,102,103,104 | 1200 | **0.991** | 0.991 ± 0.011 | 0.959 | 1.000 (n=227) | 1.000 (n=227) | 1.000 | 0.996 / 0.992 | 1.000 / 1.000 | 0.009 | 0.00 / 0 | 1.00 | 78 |
| mono | 101,102,103,104 | 1200 | **0.815** | 0.805 ± 0.060 | 0.678 | 0.982 (n=220) | 0.968 (n=220) | 0.957 | 0.750 / 0.719 | 0.895 / 0.859 | 0.247 | 0.00 / 0 | 1.00 | 152 |
| video | 101,102,103,104 | 180 | **0.792** | 0.832 ± 0.084 | 0.695 | 0.967 (n=30) | 0.933 (n=30) | 0.968 | 0.788 / 0.692 | 0.875 / 0.867 | 0.047 | 0.00 / 0 | 1.00 | 716 |

| mode | headline 95% CI | supported-kind macro F1 [95% CI] | micro F1 [95% CI] | FP / min of core kinds without truth |
|---|---|---|---|---|
| truth3d | [0.982, 1.000] | 0.991 [0.982, 1.000] | 0.996 [0.993, 0.999] | none |
| mono | [0.775, 0.873] | 0.815 [0.775, 0.873] | 0.911 [0.901, 0.920] | none |
| video | [0.733, 0.837] | 0.792 [0.733, 0.877] | 0.916 [0.885, 0.953] | none |

Only pooled multi-seed numbers of >= 180-300 s games are comparable; a single-seed <= 60 s headline is not comparable (rare kinds with 0-15 truth events enter or leave its average).

Time trend (truth3d): pooled headline per non-cumulative 60 s window: 1.000, 1.000, 0.889, 1.000, 1.000, 0.996, 0.955, 1.000, 1.000, 0.999, 1.000, 1.000, 0.999, 1.000, 0.999, 1.000, 0.964, 1.000, 1.000, 1.000; spread 0.111 over 20 full windows (whole-run 95% CI [0.982, 1.000]).
Time trend (mono): pooled headline per non-cumulative 60 s window: 0.466, 0.548, 0.821, 0.846, 0.827, 0.521, 0.761, 0.804, 0.994, 0.670, 0.857, 0.935, 0.922, 0.992, 0.855, 0.945, 0.965, 0.821, 0.930, 0.968; spread 0.528 over 20 full windows (whole-run 95% CI [0.775, 0.873]).
Time trend (video): pooled headline per non-cumulative 60 s window: 0.467, 0.821, 0.854; spread 0.387 over 3 full windows (whole-run 95% CI [0.733, 0.837]).

## Per-kind F1 pooled over seeds (truth n in brackets, then the 95% seed-bootstrap interval; * = headline kind)

| kind | truth3d | mono | video |
|---|---|---|---|
| shot_attempt* | 0.998 [272] (0.99-1.00) | 0.960 [272] (0.96-0.97) | 1.000 [37] (1.00-1.00) |
| shot_made* | 1.000 [138] (1.00-1.00) | 0.974 [138] (0.97-0.98) | 0.976 [21] (0.92-1.00) |
| shot_missed* | 0.996 [134] (0.99-1.00) | 0.938 [134] (0.90-0.98) | 0.970 [16] (0.91-1.00) |
| rebound* | 0.996 [127] (0.99-1.00) | 0.750 [127] (0.72-0.77) | 0.788 [16] (0.60-0.95) |
| assist* | 1.000 [57] (1.00-1.00) | 0.895 [57] (0.86-0.93) | 0.875 [8] (0.80-1.00) |
| steal* | 1.000 [15] (1.00-1.00) | 0.800 [15] (0.67-0.97) | 0.333 [3] (0.00-0.50) |
| turnover* | 0.982 [28] (0.94-1.00) | 0.857 [28] (0.65-0.97) | 0.600 [5] (0.50-0.67) |
| block* | 0.952 [11] (0.84-1.00) | 0.250 [11] (0.00-0.42) | 0.667 [2] (0.00-1.00) |
| pass* | 0.996 [1130] (0.99-1.00) | 0.915 [1130] (0.91-0.92) | 0.922 [161] (0.89-0.96) |
| possession | 0.992 [250] (0.98-1.00) | 0.806 [250] (0.79-0.83) | 0.850 [40] (0.70-0.97) |
| foul | 1.000 [46] (1.00-1.00) | 1.000 [46] (1.00-1.00) | 1.000 [7] (1.00-1.00) |
| substitution | 1.000 [62] (1.00-1.00) | 1.000 [62] (1.00-1.00) | — |
| timeout | 1.000 [3] (1.00-1.00) | 1.000 [3] (1.00-1.00) | 1.000 [1] (1.00-1.00) |
| period_start | 1.000 [8] (1.00-1.00) | 1.000 [8] (1.00-1.00) | 1.000 [4] (1.00-1.00) |
| period_end | 1.000 [4] (1.00-1.00) | 1.000 [4] (1.00-1.00) | — |

Tracking (video, pooled): MOTA 0.970, IDF1 0.964, ID switches 602, identity accuracy 0.977 (seed mean), RMSE 0.29 ft.
