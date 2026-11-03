# Benchmarks

Every accuracy and speed measurement the README summarises, in full. Simulator figures come from the seeded
simulator in `courtvision/sim` scored by `courtvision/eval`; NBA figures from two broadcasts scored against the ESPN
box score and play-by-play (research use, [NBA.md](NBA.md)). Newest first.

## The detector on broadcasts it never trained on (2026-11-03)

The NBA student detector against human labels of other broadcasts and teacher labels of NBA games it never saw, its
CPU paths, and research fine-tunes on more varied footage. Measured in a Linux container (4 cores of an Intel Xeon
with AVX-512 VNNI and AMX, no GPU) from the published ONNX export; [research note](research/cross_game_detector.md).
Data: SportsMOT basketball (CC BY-NC 4.0) and BARD (NBA.com clips of 60 games of 2024-25, CC BY 4.0), research use
only; no NBA game video could be downloaded in the container.

**Players, human labels** (`tools/detector_bench.py`; every 10th labelled frame; IoU 0.5; recall and precision at the
pipeline's 0.3; SportsMOT boxes players only, so precision counts referees and the crowd as false positives):

| Model | Held-out broadcasts | AP50 | Recall | Precision |
|---|---|---:|---:|---:|
| Published student | all 30 sequences, 7 broadcasts (2,501 frames, 22,825 boxes) | 0.633 | 0.611 | 0.748 |
| Published student | NCAA + 3 FIBA (`val`, 15 sequences, 1,258 frames, 11,901 boxes) | 0.675 | 0.626 | 0.793 |
| Person heads retrained on 2 NBA + 1 FIBA broadcast | NCAA + 3 FIBA (`val`) | 0.796 | 0.709 | 0.830 |
| Full fine-tune on 2 NBA + 1 FIBA broadcast (+ NBA ball pseudo-labels) | NCAA + 3 FIBA (`val`) | **0.922** | **0.879** | 0.921 |
| Full fine-tune, int8 | NCAA + 3 FIBA (`val`) | 0.921 | 0.881 | 0.920 |
| Round 2, T1: round 1's data, bf16 training, folders sampled equally | NCAA + 3 FIBA (`val`) | 0.917 | 0.865 | 0.944 |
| Round 2, T2: ball pseudo-labels from round 1's model on 4.3 times the clips ([weights](../models/research/README.md)) | NCAA + 3 FIBA (`val`) | 0.918 | 0.870 | 0.939 |
| Published student | 2 NBA + 1 FIBA (`train`, 15 sequences) | 0.592 | 0.594 | 0.701 |
| Person heads retrained on NCAA + 3 FIBA | 2 NBA + 1 FIBA (`train`) | **0.714** | 0.615 | 0.766 |
| Published student | the 2 NBA broadcasts alone (11 sequences) | 0.599 | 0.606 | 0.693 |
| Person heads retrained on NCAA + 3 FIBA | the 2 NBA broadcasts alone | 0.721 | 0.622 | 0.760 |
| Round 2, T3: full fine-tune on NCAA + 3 FIBA players (+ NBA ball pseudo-labels, no NBA player label) | the 2 NBA broadcasts alone | **0.860** | **0.736** | **0.914** |

Recall by held-out broadcast (published, heads retrained, full fine-tune): NCAA, Kentucky v Wichita State 0.824,
0.847, 0.886; FIBA Asia Championship 0.730, 0.809, 0.931; FIBA Women's World Cup 2018 0.406, 0.546, 0.881; FIBA, USA v
Russia 0.517, 0.639, 0.839.

**Ball, 30 NBA games never trained on** (90 BARD clips, three per game, 1,153 frames labelled by the student's own
teacher recipe, RF-DETR Small at 1312x736 plus path linking; top ball candidate within 15 px of 1080p, threshold
0.08; paired exact McNemar test against the published student on the frames only one of the two hits):

| Model | Hit | Wrong | Missed | Median error | Only it hits / only published hits, p |
|---|---:|---:|---:|---:|---|
| Published student | 81.7 % | 18.0 % | 0.3 % | 2.18 px | |
| int8 | 81.3 % | 18.4 % | 0.3 % | 2.27 px | 3 / 8, p 0.23 |
| Full fine-tune (ball pseudo-labels from 30 *other* BARD games) | 86.7 % | 13.3 % | 0.0 % | 2.10 px | 70 / 12, p 4.1e-11 |
| Full fine-tune, int8 | 86.6 % | 13.4 % | 0.0 % | 2.15 px | 71 / 15, p 7.1e-10 |
| Round 2, T1 (round 1's data, bf16 training) | 86.9 % | 13.1 % | 0.0 % | 2.06 px | 73 / 13, p 2.7e-11 |
| Round 2, T2 (round-2 pseudo-labels) | **89.1 %** | 10.9 % | 0.0 % | 2.12 px | 96 / 11, p 4.3e-18 |
| Round 2, T2, int8 | **89.2 %** | 10.8 % | 0.0 % | 2.16 px | 98 / 11, p 1.3e-18 |

T2 against the full fine-tune: 37 frames only T2 hits, 10 only the full fine-tune (p 9.9e-5); T2's int8 against its
fp32: 2 and 0 (p 0.5). The person heads retrained alone leave the ball heads untouched (the published network's ball).
This is the top candidate of each frame; the README's 95.7 % on the two training games scores the linked ball path
(`tools/nba_ball_bench.py`), a different measure. On the first 30 clips (368 frames) the published student scored
86.1 %, the full fine-tune 90.8 %.

**CPU paths** (`tools/student_speed.py`; one detection, network and decoding, median of 40 on 12 BARD inputs, 4
threads, idle machine; `runs/bench/speed_4t.json`):

| Path | Median | p10-p90 | Outputs |
|---|---:|---:|---|
| PyTorch fp32 (`StudentDetector`'s CPU fallback until now) | 222.0 ms | 191-265 | reference |
| ONNX Runtime fp32, whole graph | 129.8 ms | 116-162 | within 5e-6 of PyTorch |
| ONNX Runtime fp32, offset and size heads at the peaks only (`backend="onnx"`) | **105.9 ms** | 95-137 | the whole graph's detections |
| ONNX Runtime int8, heads at the peaks only | **49.6 ms** | 47-62 | ball hits 81.3 % against fp32's 81.7 % (p 0.23, above) |

The sparse path (`perception/student_onnx.py`) gives the full network's detections (heatmaps identical, offsets
within 1.4e-6). int8 against fp32 on 120 frames of 30 held-out NBA games: person boxes 98.4 % matched, ball points
92.8 % (int8 mostly adds low-score candidates: on 60 of the frames it found all 112 fp32 balls scored 0.2 or more);
on the 1,153 labelled frames above, 81.3 % against 81.7 % (3 frames only int8 hits, 8 only fp32; p 0.23). File 47.3 MB
fp32, 12.7 MB int8. The fine-tunes have the same graph, so the same cost per frame.

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
