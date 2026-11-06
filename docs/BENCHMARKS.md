# Benchmarks

Every accuracy and speed measurement the README summarises, in full. Simulator figures come from the seeded
simulator in `courtvision/sim` scored by `courtvision/eval`; NBA figures from two broadcasts scored against the ESPN
box score and play-by-play (research use, [NBA.md](NBA.md)). Newest first.

## Main model v3 in the app (2026-11-05)

The app's detector (`courtvision/server/static/nba-student.onnx`, the Studio's `models/nba-student.pt`) is now main
model v3, candidate R8f of [research note §13](research/cross_game_detector.md): main model v2 fine-tuned 4,000 steps
on its own data with SportsMOT's unlabelled officials as people (`mark-ignore --promote`), E-BARD-detection's human
labels on the 30 BARD calibration games (players, referees, the ball) and v2's ball labels on 600 more BARD clips.
Every figure is on footage the model never trained on except where marked; paired tests against v2 (frames, people or
shots only v3 gets right / only v2).

| | Main model v2 | Main model v3 | Paired |
|---|---:|---:|---|
| Ball, 30 BARD NBA games (1,097 frames, Mac teacher labels), top / linked | 91.4 / 92.8 % | **92.7** / 93.2 % | 24 / 10 frames, p 0.024; 15 / 11, p 0.56 |
| Ball, E-BARD's 30 held-out games (746 frames, human labels) | 86.3 % | **89.5 %** | 33 / 9, p 0.00027 |
| Ball, 5 held-out Roboflow games (194 single frames) | 68.0 % | **73.2 %** | 14 / 4, p 0.031 |
| Ball, Cavaliers @ Warriors (606 frames), top / linked | 79.2 / 78.4 % | 78.1 / 77.7 % | 6 / 13, p 0.17; 10 / 14, p 0.54 |
| Ball, Lakers @ Warriors held-out clips, linked (README protocol, 256 frames; v2's training game) | 95.7 % | 95.7 % | the same rate |
| Referees found: E-BARD's 30 held-out games (1,918) / the 5 Roboflow games (506) | 62 / 84 % | **84** / 85 % | 438 / 20, p 1.3e-103; 19 / 15, p 0.61 |
| Players found, E-BARD's 30 held-out games (7,728) | 85.1 % | **87.0 %** | 284 / 140, p 2.4e-12 |
| Boxes on people nobody labelled, a frame: E-BARD / Roboflow (lower is better) | **0.16** / 0.19 | 0.24 / **0.12** | frames with more / fewer: 99 / 26, p 3.2e-11; 5 / 18, p 0.011 |
| Cavaliers @ Warriors: players / referees on the court | 84.8 / 23.3 % | **90.8 / 75.9 %** | +6.0 points (clips +4.1 to +7.7) |
| Cavaliers @ Warriors: benches and crowd boxed (lower is better) | **30 %** | 70 % | |
| Lakers @ Warriors held-out clips: players / referees on the court (v2's training game) | 97.9 / 98.7 % | 98.8 / 99.6 % | +0.9 points (clips -0.3 to +3.0) |
| Team with the ball, settled live play: Lakers @ Warriors / Cavaliers @ Warriors | **97.4** / 74.9 % | 97.1 / **75.6 %** | per half: 97.9 / 96.5 against 98.2 / 96.6; 71.3 / 79.9 against 71.3 / 78.3 |
| the same, the event engine alone (before the timeline) | **72.6** / 58.4 % | 55.8 / 59.9 % | |
| Made or missed, real shots (timeline) | 84.6 / 75.0 % | 87.6 / 74.6 % | 14 / 9 shots, p 0.41; 18 / 18, p 1 |
| Made or missed with the ball's zones at the rim, Lakers @ Warriors | 90.3 % | 88.8 % | |
| Studio shot judge across games (each game by a judge fitted on the other) | 84.8 / 72.0 % | 81.7 / 72.2 % | over the shots both see: 126 / 133 of 156, p 0.17; 66 / 68 of 92, p 0.80 |
| Studio rim model, made or missed on real shots (each game held out) | 67.1 % | 71.3 % (AUC 0.79) | |
| Box-score error from the timeline | 0.197 / 0.511 | **0.171 / 0.454** | |

The Studio's rim model, shot judge and shot models were refitted on v3's output (v2's are kept in
`models/previous-v2-1eee0dd1/` of its data folder). Court registration reads no detections and is unchanged. The
browser export reproduces the checkpoint within 8.4e-5 on the validation clip and the reference outputs within 1.9e-6 on
WASM.

**The in-game referee check on the two games** (v2's detections; research note §13.5, §13.11): on the 160 boxes labelled
by eye it is right on 64 and 74 of 80 (the shirt rule 44 and 66, the learned check 56 and 32) and calls 1 of 45
Cavaliers players a referee; in the pipeline possession and made-or-missed are within chance of the shirt rule
(97.4 to 96.9 %, 74.9 to 74.9 %; 3 / 2 and 7 / 9 shots) and the box-score error is lower (0.197 to 0.178, 0.511 to
0.448). The shirt rule stays the default.

## Main model v2 in the app (2026-11-03)

The app's detector (`courtvision/server/static/nba-student.onnx`, the Studio's `models/nba-student.pt`) is now main
model v2, candidate C of [research note §10.8-10.15](research/cross_game_detector.md): the first model fine-tuned,
through round 3, on SportsMOT's players (unlabelled people among them ignored), 3 Roboflow NBA games' players and
referees, ball pseudo-labels on 390 BARD clips and the Lakers @ Warriors teacher's people on the court. Every figure is
on footage the model never trained on except where marked; paired tests against the first model.

| | First model | Main model v2 | Paired |
|---|---:|---:|---|
| Ball, 30 BARD NBA games (1,097 frames, Mac teacher labels) | 84.8 % | **91.4 %** | 84 / 11 frames, p 4.5e-15 |
| Ball, Cavaliers @ Warriors (606 frames), top / linked | 75.7 / 72.4 % | **79.2 / 78.4 %** | 27 / 6, p 0.00032; 43 / 7, p 2.1e-7 |
| Ball, Lakers @ Warriors held-out clips, linked (README protocol, 256 frames; the first model's own game) | 95.7 % | 95.7 % | the same rate |
| 5 held-out Roboflow NBA games: players and referees AP50 / referees found | 0.505 / 37 % | **0.892 / 84 %** | |
| Cavaliers @ Warriors: players on the court / benches and crowd boxed | 80.2 / 76 % | **84.8 / 30 %** | +4.6 points (clips +1.2 to +8.7) |
| Cavaliers @ Warriors: referees on the court (light grey shirts, 2017) | **83.6 %** | 23.3 % | |
| Lakers @ Warriors held-out clips: players / referees on the court (the first model's own game) | **99.0 / 99.1 %** | 97.9 / 98.7 % | -1.1 points (per box p 0.014; over clips -3.9 to +0.4) |
| Team with the ball, settled live play: Lakers @ Warriors / Cavaliers @ Warriors | 96.4 / 75.8 % | **97.4** / 74.9 % | per half: 98.2 / 96.6 against 96.5 / 96.2; 71.3 / 78.3 against 70.1 / 81.4 |
| Made or missed, real shots (timeline) | 83.1 / 75.4 % | 84.6 / 75.0 % | 18 / 15 shots, p 0.73; 14 / 15, p 1 |
| Made or missed with the ball's zones at the rim, Lakers @ Warriors | 88.1 % (earlier code) | **90.3 %** | |
| Studio shot judge across games (each game by a judge fitted on the other) | 84.0 / 77.8 % | 84.8 / 72.0 % | over the shots both see: 130 / 129 of 154 and 56 / 57 of 75, p 1 and 1 |
| Box-score error from the timeline | 0.209 / 0.570 | **0.197 / 0.511** | |

The Studio's rim model, shot judge and shot models were refitted on v2's output (the first model's are kept in
`models/previous-8bdf229a/` of its data folder). Court registration reads no detections and is unchanged.

## The detector fine-tunes on the README's two games (2026-11-03, M3 Pro)

The published student (the app model) against the round-2 research fine-tune (T2, `models/research/`) where the app
is shown: Lakers @ Warriors, the game the app model was distilled on (its 15 held-out teacher clips), and Cavaliers @
Warriors, which neither trained on (60 new teacher clips). Ball and people on ONNX Runtime for both models; the
pipeline re-detected on Core ML for both, current code; [research note §10](research/cross_game_detector.md).

**Ball** (teacher labels; within 15 px of 1080p; paired exact McNemar test on the frames only one model hits; T5 is
the cloud's round 3, `student-ft-r3`):

| Game (frames) | Measure | App model | Round 2 | Round 2 int8 | T5 | Round 2 vs app (only it / only app, p) | T5 vs app |
|---|---|---:|---:|---:|---:|---|---|
| Lakers @ Warriors (243) | top candidate | **95.5 %** | 94.2 % | 93.8 % | 95.1 % | 0 / 3, p 0.25 | 0 / 1, p 1 |
| | linked path (README protocol) | 95.5 % | 95.5 % | 95.5 % | **95.9 %** | the same frames | 1 / 0, p 1 |
| Cavaliers @ Warriors (606) | top candidate | 75.7 % | **78.1 %** | 77.7 % | 77.9 % | 20 / 6, p 0.0094 | 25 / 12, p 0.047 |
| | linked path | 72.4 % | 77.9 % | 78.6 % | **78.7 %** | 36 / 3, p 3.6e-8 | 45 / 7, p 7.0e-8 |

**People** (the teacher's boxes, placed on the court by the game's camera track; referees by `analyze.is_referee`):
players on the court found, Lakers @ Warriors 99.0 % (app), 95.9 % (round 2), 95.2 % (T5) of 1,311 boxes; Cavaliers @
Warriors 80.2 %, 81.5 %, 80.1 % of 3,535; referees on the court 99.1 %, 27.0 %, 10.0 % and 83.6 %, 12.1 %, 12.1 %:
the fine-tunes, taught by player-only labels, leave referees and the crowd unboxed.

**The pipeline** (fitted on one half of each game, scored on the other; made or missed paired with the app model over
the shots both found):

| | LAL @ GSW: app | round 2 | T5 | CLE @ GSW: app | round 2 | T5 |
|---|---:|---:|---:|---:|---:|---:|
| Team with the ball, settled live play (timeline) | 96.4 % | 97.0 % | 96.8 % | 75.8 % | 74.0 % | 74.8 % |
| the same, engine alone | 72.8 % | 79.1 % | 69.1 % | 61.0 % | 58.4 % | 61.5 % |
| Real shot or not | 78.5 % | 73.7 % | 76.9 % | 52.4 % | 63.9 % | 57.8 % |
| Made or missed, real shots | 83.1 % | 85.4 % | 87.1 % | 75.4 % | 66.7 % | 78.8 % |
| paired with the app model (only it right / only app, p) | | 17 / 12, p 0.46 | 16 / 10, p 0.33 | | 11 / 18, p 0.27 | 14 / 12, p 0.85 |
| Box-score error | 0.209 | 0.225 | 0.208 | 0.570 | 0.581 | 0.574 |

Neither game's pipeline numbers move beyond chance, so the app keeps its detector. The app model's README run `full`
(PyTorch on MPS, older code: 96.2 %, 84.8 %) re-runs at 96.3 % and 83.6 % with the current code, and at 96.4 % and
83.1 % on Core ML. Re-detecting the whole game on the Neural Engine takes 12 ms a frame (20 minutes).

**Fine-tuning on the Mac** (`student_finetune.py train --device mps`, the same data and start; steps 20-60 of a short
run): MPS fp32 0.33 s a step, bf16 autocast 0.30 s, fp16 0.35 s, channels-last 0.33 s; the CPU (6 threads) 0.83 s
in fp32 and 11.4 s with bf16 autocast (no bf16 hardware path on Apple's CPU). A 3,000-step run takes about 16 minutes
on MPS; the losses agree with the CPU's to the fourth decimal.

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
| Round 2, T4 (all seven SportsMOT broadcasts) | 89.4 % | 10.6 % | 0.0 % | 2.12 px | 103 / 14, p 6.4e-18 |
| Round 3, T5 (+3 NBA games' human labels, temporal dropout) | **89.5 %** | 10.5 % | 0.0 % | 2.23 px | 104 / 14, p 3.6e-18 |
| Round 3, T5, int8 | 89.4 % | 10.6 % | 0.0 % | 2.24 px | 104 / 15, p 1.5e-17 |

T2 against the full fine-tune: 37 frames only T2 hits, 10 only the full fine-tune (p 9.9e-5); T2's int8 against its
fp32: 2 and 0 (p 0.5). T4 and T5 against T2: 14 / 10 (p 0.54) and 27 / 22 (p 0.57), no difference shown. The person heads retrained alone leave the ball heads untouched (the published network's ball).
This is the top candidate of each frame; the README's 95.7 % on the two training games scores the linked ball path
(`tools/nba_ball_bench.py`), a different measure. On the first 30 clips (368 frames) the published student scored
86.1 %, the full fine-tune 90.8 %.

**Players and the ball, 5 NBA games with human labels** (Roboflow Universe "basketball-player-detection-2", CC BY 4.0;
211 single frames of 7 clips, 1,961 player boxes, the ball on 194 frames; never trained on; `--coco`: IoU 0.5, recall
and precision at 0.3, ball within 15 px of 1080 lines; each frame fed as its own history, three times):

| Model | Players AP50 | Recall | Precision | Ball hit | Only it hits / only published hits, p |
|---|---:|---:|---:|---:|---|
| Published student | 0.396 | 0.396 | 0.691 | 22.7 % | |
| Round 1, full fine-tune | 0.743 | 0.710 | 0.796 | 33.5 % | 26 / 5, p 1.9e-4 |
| Round 2, T2 ([weights](../models/research/README.md)) | 0.761 | 0.733 | 0.820 | 34.5 % | 29 / 6, p 1.2e-4 |
| Round 2, T4 (all seven SportsMOT broadcasts) | 0.770 | 0.725 | 0.850 | 34.5 % | 31 / 8, p 2.9e-4 |
| Round 3, T5 (+3 other NBA games of this set, temporal dropout; [weights](../models/research/README.md)) | **0.870** | **0.821** | **0.913** | **68.0 %** | 90 / 2, p 1.7e-24 |
| Round 3, T5, int8 | 0.869 | 0.821 | 0.914 | 67.5 % | 90 / 3, p 2.7e-23 |

T5 trained on 3 other games of the same set (the same labelling) and, alone, on repeated frames (a quarter of its
samples), which favours it on these single frames, the ball most of all; in video its ball is T2's and T4's (above).

**Referees and the crowd, the same 5 games** (506 referees labelled; "on nobody": detections overlapping no labelled
person at IoU 0.1, i.e. the crowd, benches and staff; research note §11):

| Model | Referees found | Players and referees AP50 / recall | Detections on nobody, per frame | Ball (single frames) | Ball in video (1,153 BARD frames) |
|---|---:|---:|---:|---:|---:|
| Published student | 187 (37.0 %) | 0.505 / 0.391 | 0.09 | 22.7 % | 81.7 % |
| Round 2, T2 | 84 (16.6 %) | 0.687 / 0.617 | 0.02 | 34.5 % | 89.1 % |
| Round 3, T5 | 26 (5.1 %) | 0.768 / 0.663 | 0.06 | 68.0 % | 89.5 % |
| Round 4, R4a (+ Roboflow referees as people) | **424 (83.8 %)** | **0.881** / **0.831** | 0.19 | 63.9 % | 89.3 % |
| Round 4, R4b (+ SportsMOT ignore regions; [weights](../models/research/README.md)) | **424 (83.8 %)** | **0.881** / 0.828 | 0.17 | 66.5 % | 89.25 % |
| Round 4, R4b, int8 | 425 (84.0 %) | 0.882 / 0.830 | 0.15 | 67.5 % | 89.25 % |

**The pipeline's referee check** (`nba/referee.py`, the default of `analyze.is_referee`), each of the 8 Roboflow NBA
games left out in turn (28,875 labelled boxes; threshold 0.5, fixed): referees named 73.9 % (the shirt rule) to
**94.3 %**, players called referee 2,464 to 1,730 of 22,440, F1 0.690 to **0.847**, exact McNemar p 1.3e-188; worse
than the rule in the 2 games with Denver. 1.6 ms a frame of 12 people.

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
