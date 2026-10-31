# Benchmarks

Every accuracy and speed measurement the README summarises, in full. Simulator figures come from the seeded
simulator in `courtvision/sim` scored by `courtvision/eval`; NBA figures from two broadcasts scored against the ESPN
box score and play-by-play (research use, [NBA.md](NBA.md)). Newest first.

## NBA broadcast accuracy

Measured on two real games so far, Lakers @ Warriors (2024-12-25, ESPN, 1080p) and Cavaliers @ Warriors (2017-06-12,
NBA Finals Game 5, ABC, 720p at 59.94 fps), against the ESPN box score and play-by-play (research use; how it works:
[NBA.md](NBA.md)). Measurements of 2026-10-29 and 2026-10-30
([made shots at the rim](research/made_shots_at_the_rim.md)):

| What | Result | Measured on |
|---|---|---|
| Ball found within 15 px | **95.7%** (3.5% wrong, 0.8% missed); earlier student 88.3%, RF-DETR Nano starter 25.8% | 256 labelled frames of held-out teacher clips (balls the teacher found: a ball held in a player's hands is not labelled, so not measured) |
| Court registration, no clicks | of 26,103 frames: 22,160 tracking, 44 held, 173 re-acquired, 3,726 without the court (not in view or not found) | first quarter |
| Points, field goals made, threes made | exact for both teams (from the scorebug) | first quarter and the full scorebug scan |
| Box score, 13 stats x 2 teams | sum of absolute count errors / sum of true counts: 0.305 before the fusion work, **0.068** after (13 of 190 counts off) | first quarter: **in-sample**, the quarter the fusion rules were developed on |
| Shots, no scorebug | made or missed right on **84.8%** of real shots from the video (**87.6%** adding TypeSafe Jev on rim crops), free throw or field goal on 94.9%; points 119 / 95 for 115 / 113 (the engine alone 79 / 75), box-score error 0.203 with the timeline below (the engine alone 0.314) | whole game, fitted on one half and scored on the other ([details](research/possession_decision_model.md)) |
| Team with the ball, no scorebug (possession timeline) | **96.2%** of settled live play, the event engine alone 73.6%; turnovers F1 0.18 → 0.42, steals 0.06 → 0.29, rebounds 0.47 → 0.52 | whole game, fitted on one half and scored on the other ([details](research/possession_decision_model.md)) |
| Made or missed in the Studio (the shot judge, no scorebug) | made-shot F1 **0.32 → 0.81** (makes called 4 → 12, 11 right, of 15); made or missed right on 44% → **93%** of the results on real shots | Lakers @ Warriors 1589-2189 s, a stretch the judge was not fitted on ([details](research/made_shots_at_the_rim.md)) |
| Made or missed with the evidence at the rim (ball zones, rim model) | within a game: shot models 84.8% → **88.1%**, the Studio's judge 83.3% (the engine's own call) → **92.6%**; across games (fitted on the other game only): the judge **84.0% / 77.8%** against the engine's 83.3% / 68.9% | Lakers @ Warriors halves; each game by models fitted on the other |
| Second game, Cavaliers @ Warriors (720p, 2017) | court found on about half the frames (the crowd's yellow shirts fool the floor mask); team with the ball **75.8%** of settled play after team clusters are refitted (38.8% before); made or missed 78.1% within the game; the engine reports only 14 of 27 shots in a 10-minute stretch | whole game, halves ([details](research/made_shots_at_the_rim.md) §4-6) |

Held-out quarters (Q2-Q4, never used for tuning) and further games are the real test; `tools/nba_loop.py` records
every evaluation in `runs/nba/leaderboard.jsonl`.

## Analysis speed

The same game on an M3 Pro, before and after the speed-ups of 2026-10-29, on the same frames
([how, and how each was checked](research/analysis_speed.md)):

| Stage | Before | After | Same results |
|---|---:|---:|---|
| Studio analysis (Live, and the analysis ahead for Replay): 20 s of Q2 with replays; 60 s of live Q3 | 164; 118 ms a frame (0.20-0.28 x real time) | 47-55; 46 ms (0.6-0.72 x) | identical recording with the torch detector; with Core ML 99.8 % of boxes within 1 px, the same plays |
| Student detector, one live frame | 63 ms (MPS) | 10 ms (Neural Engine) | held-out ball F1 0.9608 both |
| `nba_analyze.py events` | 14.5 ms a frame | 5.1 ms | `events.jsonl` byte-identical |
| `nba_analyze.py detect` (one process) | 56.5 ms a frame | 20.6 ms | `--backend torch`: bit-identical |
| `nba_track_court.py` where the camera is often lost | 82.7 ms a frame | 48.0 ms | every frame identical |
| Studio analysis ahead (Replay), 2026-10-30: the court tracked in a worker process that works ahead | 56.2 / 55.0 ms a frame | **37.7 / 38.3 ms** | identical recordings (events, boxes, messages) |
| Studio analysis of a 60 fps video (read at every 2nd frame, as the models are set up for 29.97 fps) | 60 frames a second analysed | 30 | the detector's input frames 33 ms apart, as in training |


## Round 5 held-out verification

One aggregate check on seeds 101–104, after integrating `425a779`:

| Mode | Source seconds per game | Headline F1 | 95% seed-bootstrap CI |
|---|---:|---:|---:|
| `truth3d` | 1200 | **0.991** | 0.982–1.000 |
| `mono` | 1200 | **0.815** | 0.775–0.873 |
| `video` | 180 | **0.792** | 0.733–0.837 |

Truth3d exceeds the 0.98 simulator target; mono and video remain below 0.85. The round-5 simulator
adds pump fakes and changes the generated games, so these figures are not a controlled before/after
comparison with round 4. [Complete aggregate report](bench/round5-heldout.md).


## Round 4 Accuracy Benchmark (2026-10-22)

Error-driven event-engine fixes, tuned on train seeds 1–6 only, each checked by an independent review (rule compliance, overfitting, reproduced numbers), then scored once on the held-out seeds 101–104. "Round start" is commit `5e52881` measured on the same games.

| Mode (s/game) | Train 1–6: round start | Train 1–6: round 4 | Held-out 101–104: round start | Held-out 101–104: round 4 [95% CI] | Held-out verdict |
|---|---|---|---|---|---|
| `truth3d` (1200) | 0.967 | **0.996** | 0.941 | **0.970** [0.960, 0.980] | +0.029, outside the CI |
| `mono` (1200) | 0.751 | **0.829** | 0.708 | **0.742** [0.696, 0.780] | +0.034, outside the CI |
| `video` (180) | 0.806 | **0.892** | 0.636 | **0.650** [0.575, 0.934] | +0.014, noise (180 s games) |

- **What changed**:
  - `truth3d`: a shot rises to about rim height; a throw-in restarts play when a player on the court gains it; a ring bounce between two frames is not a block; the score reconciler waits a moment before flipping a miss.
  - `mono` / `video`:
    - blocks: committed only once the ball's next owner shows it was knocked loose, with image-speed, touch-margin and head-line gates;
    - steals: a gain long after the release is a loose-ball recovery, and only a dribble in the holder's hands corroborates control;
    - passes: a pass crossing a teammate's box in flight is one pass;
    - shots: a miss the ball went back to the rim after is not flipped by the score (an unseen putback is inferred instead);
    - dribbles: measured as the ball's height above the dribbler's feet, so drives towards or away from the camera keep their bounces;
    - rebounds: hidden holds, catches out of view and loose balls crossing a body.
- **Generalisation**: the `truth3d` gain carries over to the held-out games; the monocular gain only partly (+0.079 on train, +0.034 held-out). The six train games hold few rare events (23 blocks, 28 steals), so fixes to single plays fit them.
- **Held-out `mono` per kind**: steals 0.50, turnovers 0.68, blocks 0.27 (8 truth events), rebounds 0.73, assists 0.81, missed shots 0.89, passes 0.91.
- **Targets** (`mono` and `video` ≥ 0.85, `truth3d` ≥ 0.98) are **not met** on held-out games.
- Held-out reports (aggregate only): [`docs/bench/round4_heldout_before_5e52881.md`](bench/round4_heldout_before_5e52881.md), [`docs/bench/round4_heldout_after_ffe5129.md`](bench/round4_heldout_after_ffe5129.md).

## Round 3 Accuracy Benchmark (1200-Second Games)

Evaluated across train seeds (1–6) and held-out test seeds (101–104) with deterministic random seeds:

| Split (Seeds, Duration) | Mode | Session Baseline | Round 3 | Delta (95% Bootstrap CI) |
|---|---|---|---|---|
| **Train** (Seeds 1–6, 1200s) | `truth3d` | 0.949 | **0.967** | +0.018 |
| **Train** (Seeds 1–6, 1200s) | `mono` | 0.738 | **0.751** | +0.013 |
| **Held-Out** (Seeds 101–104, 1200s) | `truth3d` | 0.916 | **0.941** | +0.025 |
| **Held-Out** (Seeds 101–104, 1200s) | `mono` | 0.697 | **0.708** | +0.010 |

*Note: Video mode (rendering frames + perception stack, 180s) achieved **0.740** headline F1 on train seeds (seed mean 0.796) and **0.636** on held-out seeds (seed mean 0.704, MOTA 0.971, identity accuracy 0.976). Correction (round 4): the 0.740 train figure was measured before the last round-3 commits; the round-3 final commit `5e52881` scores 0.806 on the same games.*

## Realism Pass Baseline (300-Second Games, Seeds 1–6)

| Mode | Headline F1 (Pooled) | Micro F1 | FG Make/Miss | Rebound F1 / Attr | Assist F1 / Label | Box MAE |
|---|---|---|---|---|---|---|
| `truth3d` | **0.953** [0.869, 0.997] | 0.988 | 1.000 (n=83) | 1.000 / 0.945 | 0.976 / 1.000 | 0.015 |
| `mono` | **0.553** [0.536, 0.573] | 0.814 | 0.987 (n=78) | 0.642 / 0.600 | 0.800 / 0.655 | 0.113 |

## System Latency & Performance Profile

Re-measured on 2026-10-26 with `simulate --seconds 120 --seed 7 --mode truth3d`, Jev off,
template narration. This is an **offline synthetic run**, not RF-DETR throughput or live broadcast latency.

| Measurement | Result |
|---|---|
| Complete frame step | p50 0.1 ms, p95 0.5 ms (3,600 frames) |
| 120 source seconds | 0.9 s processing wall time (128.6×; excludes pre-simulation) |
| Event occurrence → engine emission | p50 66.7 ms, p95 833.3 ms (155 events) |
| Engine emission → narration | p50 233.3 ms, p95 931.7 ms (18 matched lines) |
| Event occurrence → narration | p50 650.0 ms, p95 3160.0 ms (18 matched lines) |
| Synthetic video perception (same seed/duration, 1280×720) | p50 28.5 ms, p95 41.5 ms (3,600 frames) |
| Complete synthetic video run | 120 source seconds in 114.5 s processing wall time; includes rendering and perception |

The studio displays vision-stage time and inference rate (excluding decoding and UI) for the loaded footage. It processes
every frame; slower hardware can fall behind source time. RF-DETR Nano starter detections were exercised
on the [Supervision basketball sample](https://supervision.roboflow.com/latest/assets/), without a labelled
event-accuracy evaluation. See [video workspace validation](STUDIO.md) for the exact scope.

