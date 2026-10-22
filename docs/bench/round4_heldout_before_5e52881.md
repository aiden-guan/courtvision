# CourtVision accuracy bench (heldout split)

`.venv/bin/python tools/accuracy_bench.py --modes truth3d,mono,video --split heldout --seconds 1200 --video-seconds 180 --workers 4 --json /home/user/courtvision/runs/bench/r4_heldout_5e52881.json --md /home/user/courtvision/runs/bench/r4_heldout_5e52881.md -q`

| mode | seeds | s/game | headline F1 (pooled) | seed mean ± sd | attr. F1 | FG make/miss | 2 vs 3 | made-FG F1 | rebound F1 / rebounder | assist F1 / label | box MAE | team pts err (mean / max) | score match | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| truth3d | 101,102,103,104 | 1200 | **0.941** | 0.934 ± 0.052 | 0.934 | 1.000 (n=214) | 1.000 (n=214) | 1.000 | 0.975 / 0.958 | 1.000 / 1.000 | 0.031 | 0.00 / 0 | 1.00 | 166 |
| mono | 101,102,103,104 | 1200 | **0.708** | 0.695 ± 0.034 | 0.599 | 0.971 (n=206) | 0.951 (n=206) | 0.950 | 0.704 / 0.727 | 0.793 / 0.705 | 0.271 | 0.00 / 0 | 1.00 | 272 |
| video | 101,102,103,104 | 180 | **0.636** | 0.704 ± 0.164 | 0.556 | 0.933 (n=30) | 1.000 (n=30) | 0.919 | 0.710 / 0.727 | 0.706 / 0.706 | 0.041 | 0.00 / 0 | 1.00 | 2421 |

| mode | headline 95% CI | supported-kind macro F1 [95% CI] | micro F1 [95% CI] | FP / min of core kinds without truth |
|---|---|---|---|---|
| truth3d | [0.906, 0.965] | 0.941 [0.906, 0.965] | 0.987 [0.983, 0.991] | none |
| mono | [0.674, 0.729] | 0.708 [0.674, 0.729] | 0.879 [0.870, 0.890] | none |
| video | [0.565, 0.880] | 0.636 [0.565, 0.880] | 0.902 [0.873, 0.932] | none |

Only pooled multi-seed numbers of >= 180-300 s games are comparable; a single-seed <= 60 s headline is not comparable (rare kinds with 0-15 truth events enter or leave its average).

Time trend (truth3d): pooled headline per non-cumulative 60 s window: 1.000, 0.946, 0.998, 0.999, 0.949, 0.825, 0.995, 0.997, 0.999, 0.917, 0.981, 0.998, 0.999, 0.999, 1.000, 0.857, 0.973, 0.951, 0.997, 1.000; spread 0.175 over 20 full windows (whole-run 95% CI [0.906, 0.965]).
Time trend (mono): pooled headline per non-cumulative 60 s window: 0.618, 0.716, 0.755, 0.690, 0.747, 0.640, 0.799, 0.731, 0.958, 0.767, 0.756, 0.968, 0.826, 0.739, 0.622, 0.584, 0.667, 0.612, 0.704, 0.825; spread 0.385 over 20 full windows (whole-run 95% CI [0.674, 0.729]).
Time trend (video): pooled headline per non-cumulative 60 s window: 0.659, 0.595, 0.882; spread 0.287 over 3 full windows (whole-run 95% CI [0.565, 0.880]).

## Per-kind F1 pooled over seeds (truth n in brackets, then the 95% seed-bootstrap interval; * = headline kind)

| kind | truth3d | mono | video |
|---|---|---|---|
| shot_attempt* | 0.989 [280] (0.99-0.99) | 0.918 [280] (0.89-0.93) | 0.921 [38] (0.86-0.95) |
| shot_made* | 1.000 [147] (1.00-1.00) | 0.952 [147] (0.92-0.99) | 0.933 [23] (0.87-1.00) |
| shot_missed* | 0.977 [133] (0.97-0.98) | 0.881 [133] (0.84-0.93) | 0.774 [15] (0.55-0.89) |
| rebound* | 0.975 [124] (0.97-0.98) | 0.704 [124] (0.66-0.75) | 0.710 [15] (0.45-0.85) |
| assist* | 1.000 [62] (1.00-1.00) | 0.793 [62] (0.71-0.90) | 0.706 [8] (0.50-0.80) |
| steal* | 0.929 [13] (0.88-1.00) | 0.471 [13] (0.33-0.56) | 0.000 [1] (0.00-0.00) |
| turnover* | 0.943 [25] (0.90-0.98) | 0.567 [25] (0.38-0.71) | 0.727 [5] (0.67-1.00) |
| block* | 0.667 [8] (0.40-0.86) | 0.182 [8] (0.00-0.35) | 0.000 [2] (0.00-0.00) |
| pass* | 0.991 [1095] (0.98-1.00) | 0.902 [1095] (0.90-0.91) | 0.951 [176] (0.93-0.97) |
| possession | 0.973 [243] (0.96-0.98) | 0.768 [243] (0.73-0.80) | 0.846 [38] (0.70-0.97) |
| foul | 1.000 [55] (1.00-1.00) | 1.000 [55] (1.00-1.00) | 1.000 [7] (1.00-1.00) |
| substitution | 1.000 [61] (1.00-1.00) | 1.000 [61] (1.00-1.00) | — |
| timeout | 1.000 [3] (1.00-1.00) | 1.000 [3] (1.00-1.00) | — |
| period_start | 1.000 [8] (1.00-1.00) | 1.000 [8] (1.00-1.00) | 1.000 [4] (1.00-1.00) |
| period_end | 1.000 [4] (1.00-1.00) | 1.000 [4] (1.00-1.00) | — |

Tracking (video, pooled): MOTA 0.971, IDF1 0.963, ID switches 593, identity accuracy 0.976 (seed mean), RMSE 0.30 ft.
