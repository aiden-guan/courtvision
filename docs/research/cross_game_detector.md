# The student detector on broadcasts it never saw: a cross-game benchmark, a faster CPU path, and fine-tuning

Work of 2026-11-03 in a Linux container (4 cores of an Intel Xeon with AVX-512 VNNI and AMX, 15 GB, no GPU), research
use. Only the published ONNX export of the student (`courtvision/server/static/nba-student.onnx`, epoch 8 of the
checkpoint `8bdf229a…`) was available here; the two NBA games it was trained on, its teacher labels and its own
held-out frames were not, so nothing below re-measures the README's two-game numbers.

## 1. Data

No NBA game video could be downloaded in the container: `app.ytdown.to` answers with a Cloudflare bot challenge that
a headless browser never clears, and YouTube refuses video streams to the container's address (HTTP 403). Neither
was worked around. Two public research datasets stood in:

| Data | What | Licence | Used for |
|---|---|---|---|
| SportsMOT, basketball sequences ([Cui et al., ICCV 2023](https://github.com/MCG-NJU/SportsMOT); Hugging Face `MCG-NJU/SportsMOT`) | 720p 25 fps broadcast clips, every player on the court boxed by hand in every frame (referees, coaches and the crowd are not). 30 sequences with public labels from 7 broadcasts: 2 NBA, 1 NCAA, 4 FIBA (men's and women's) | CC BY-NC 4.0 | the person benchmark (§2); the 15 `train` sequences (2 NBA broadcasts and 1 FIBA) as fine-tuning labels, the 15 `val` sequences (1 NCAA, 3 FIBA) held out |
| BARD ([Giudici et al., CVIU 2026](https://github.com/GabrieleGiudic/BARD); Hugging Face `GabrieleGiudici/BARD`) | NBA.com event clips of 60 games of the 2024-25 season (720p 60 fps, ~8.5 s each) with play-by-play labels | CC BY 4.0 (the footage is the NBA's) | 3 clips per game (180): int8 calibration and checks, ball pseudo-labels; games split into 30 + 30 by alternation |

Both are research-only here (docs/TRAINING.md §1). No footage or label from them is committed (`data/` is
git-ignored). The round-2 fine-tune's weights (§8.2) are, for research use, in
[`models/research/`](../../models/research/README.md) with these sources and their terms. The exact splits (games, clips, sequences;
file names only) are in [`cross_game_detector_splits.json`](cross_game_detector_splits.json).

## 2. The benchmark: players on seven other broadcasts

`tools/detector_bench.py` runs the detector as the pipeline does (the frame and the two before it, 960x544 input,
person threshold 0.3) on every 10th labelled frame of each sequence and matches boxes greedily by score at IoU 0.5.
SportsMOT labels only players, so **recall** is the clean number; precision counts every referee, coach and spectator
the detector finds as a false positive and is only for comparing models.

The published student (2,501 frames, 22,825 player boxes):

| | AP50 | Recall at 0.3 | Precision at 0.3 |
|---|---:|---:|---:|
| All 30 sequences | 0.633 | 0.611 | 0.748 |
| The 15 `val` sequences (held out of the fine-tunes of §6.2; 1,258 frames, 11,901 boxes) | 0.675 | 0.626 | 0.793 |

By broadcast (recall at 0.3; the split in brackets):

| Broadcast (SportsMOT source video) | Sequences | Recall |
|---|---:|---:|
| NCAA tournament, Kentucky v Wichita State, CBS (`v_00HRwkvvjtQ`, val) | 6 | 0.824 |
| FIBA Asia Championship, Philippines v Palestine (`v_4r8QL_wglzQ`, val) | 1 | 0.730 |
| NBA, Warriors at Oracle Arena, CSN Bay Area (`v_2j7kLB-vEEk`, train) | 6 | 0.689 |
| FIBA Women's Asia Cup 2019, Australia v Philippines (`v_-6Os86HzwCs`, train) | 4 | 0.543 |
| FIBA, USA v Russia in Istanbul (`v_BgwzTUxJaeU`, val) | 3 | 0.517 |
| NBA, Lakers at New Orleans, a re-streamed local feed (`v_4LXTUim5anY`, train) | 5 | 0.467 |
| FIBA Women's World Cup 2018, Australia v USA (`v_5ekaksddqrc`, val) | 5 | 0.406 |

The `val` broadcasts are NCAA and FIBA games; the two NBA broadcasts are held out only in the swapped split of §6.1.
NBA footage held out of everything is the 30 BARD games of §5 (ball).

The misses are not small or occluded players: on two frames of the weakest broadcasts (`v_4LXTUim5anY_c002` and
`v_5ekaksddqrc_c004`, frame 300) whole, upright players went unboxed (the labelled boxes there have a median height of
175 px at 720p; 2 and 4 boxes scored 0.3 or more against 6 and 10 labelled players) while low-score boxes landed on
the crowd. A scale probe (`--zoom`:
the frame shrunk inside the network input) lowered recall (0.625 to 0.568 at 0.7x on four sequences), so the gap is
appearance, not size: a detector distilled on two NBA games has learnt those two games' players. "More, varied games"
(README) is measurably the lever.

## 3. Faster on a CPU, with the same detections

### 3.1 Offset and size heads only where they are read

The network's five heads each run a 3x3 convolution (128 to 64 channels) and a 1x1 convolution over the whole 136 x
240 stride-4 grid; the three offset and size heads are 7.23 of the network's 38.70 GMACs (18.7 %). The decoder reads
them only at the kept heatmap peaks (at most 8 balls and 60 people of 32,640 cells). `perception/student_onnx.py`
(`OnnxStudent`) cuts the ONNX graph after the shared feature map and the two heatmap heads, and evaluates the three
other heads in NumPy at the peaks, from the same weights. The detections are those of the full network: on SportsMOT
frames the heatmaps were identical, the offsets and sizes within 1.4e-6, the decoded boxes within 6e-5 px; tests
compare the two paths (`tests/test_perception_student_onnx.py`). `StudentDetector` runs it with `backend="onnx"` (or
`"auto"` given an `.onnx` file).

### 3.2 int8

`tools/quantize_student.py` writes a QDQ model with ONNX Runtime: int8 weights per output channel, 8-bit activations
calibrated on 64 inputs drawn from the SportsMOT `train` sequences and the clips of the 30 BARD calibration games
(percentile 99.999); the five 1x1 output convolutions and the three sparse heads stay fp32. Two ONNX Runtime 1.31
pitfalls cost four failed runs: its histogram calibrators keep every intermediate output of every input (the process
grew past 13.7 GB and was killed), and its `CalibMaxIntermediateOutputs` chunking clears each chunk without merging
it. Feeding one input per pass (`CalibStridedMinMax=1` with a reader that can `set_range`) bounds memory.

Checked against fp32 on 120 inputs of the 30 other BARD games (`runs/bench/int8_u8_pct.json`):

| | fp32 | int8 |
|---|---:|---:|
| File | 47.3 MB | 12.7 MB |
| Person boxes (IoU 0.5), both at 0.3 | 882 | 875 matched, 7 only fp32, 7 only int8 (98.4 %) |
| Ball points (15 px of 1080p), both at 0.08 | 475 | 465 matched, 10 only fp32, 26 only int8 (92.8 %); median 0.06 px apart |

On 60 of those inputs every fp32 ball scored 0.2 or more was found by int8 too (112 of 112; 177 of 178 between
0.08 and 0.2), and the top ball was the same in 56 of 59 frames: int8 mostly adds low-score candidates near the
threshold. Keeping the ball heatmap head's 3x3 convolution in fp32 as well changed little (93.6 %). On the 368
teacher-labelled NBA frames of §5 int8 gets the same hit, wrong and missed counts as fp32.

### 3.3 Speed

`tools/student_speed.py`, one detection (network and decoding) on 12 BARD inputs, median of 40, 4 threads, nothing
else running (`runs/bench/speed_4t.json`):

| Path | Median | p10-p90 |
|---|---:|---:|
| PyTorch fp32 (`StudentDetector`'s CPU fallback until now) | 222.0 ms | 191-265 |
| ONNX Runtime fp32, whole graph | 129.8 ms | 116-162 |
| ONNX Runtime fp32, sparse heads (`backend="onnx"`) | 105.9 ms | 95-137 |
| ONNX Runtime int8, sparse heads | 49.6 ms | 47-62 |

The same detections run 2.1 times as fast as on PyTorch, and int8 4.5 times. On a Mac the Core ML path stays the
faster one (the network 7.7 ms on the Neural Engine, a live detection 10 ms: `analysis_speed.md`); these paths are
for other machines.

## 4. Fine-tuning on more broadcasts

`tools/student_finetune.py` (TRAINING.md §7B) starts from an exact PyTorch copy of the export
(`training/student_from_onnx.py`: each folded batch norm becomes the identity plus its bias, outputs within 4.8e-6 of
ONNX Runtime) and trains with `train_student`'s data loading, augmentation and losses:

* **players** from the human boxes of the 15 SportsMOT `train` sequences (12,384 frames, every 3rd a sample: 4,125, of
  which 3,733 train and one sequence is `train_student`'s own held-out check). These label players only, so the person
  head also learns to leave referees, benches and the crowd alone (see §7 for what that changes downstream);
* **the ball** from the published network's own detections on the clips of the 30 BARD calibration games, linked into
  one path per clip (`training.ball_labels.link_ball_path`, which also fills fast flights the detector missed): 9,373
  of 10,800 frames labelled (141 of them filled), 3,132 samples, label weight at most 0.5. This anchors the ball head
  while the trunk changes (self-training with temporal consistency, after Noisy Student, Xie et al. 2020);
* batch norms frozen (they hold the folded statistics); the stem and first stage frozen; AdamW, cosine decay;
* domain randomisation on player frames: hue turned by up to ±30° (jerseys, floors and adverts change colour from
  game to game; frames with a ball label keep their colours) and scale 0.7-1.4 (training used 0.8-1.25).

Two variants:

* **heads only**: the three person heads retrained (1,500 steps, batch 4); the trunk, the feature pyramid and both
  ball heads untouched, so the ball detections are those of the published network bit for bit;
* **full**: everything after the first stage (3,000 steps, batch 4, both data sources).

## 5. The ball on 30 NBA games never trained on

The SportsMOT labels have no ball, so the ball is checked on NBA footage held out of everything: 90 clips of the 30
BARD check games (one per game, then two more per game), labelled by the student's own teacher recipe
(`tools/nba_teacher_labels.py`: RF-DETR Small, COCO, at 1312 x 736, threshold 0.04, linked by `link_ball_path`) on 24
frames at ~30 fps each: 1,153 frames carry a ball label of weight 0.5 or more with two frames before them to score
(368 of them in the first 30 clips). `tools/detector_bench.py --ball-folder` counts the network's top ball candidate
(threshold 0.08) within 15 px of 1080p of the label (hit), elsewhere (wrong) or nowhere (missed), and compares two
models with an exact McNemar test on the frames where only one of them hits (`runs/bench/ball_check12_r1.json`):

| Model | Hit | Wrong | Missed | Median error (px of 1080p) | vs published: frames only it hits / only published, p |
|---|---:|---:|---:|---:|---|
| Published student, fp32 | 81.7 % | 18.0 % | 0.3 % | 2.18 | |
| int8 | 81.3 % | 18.4 % | 0.3 % | 2.27 | 3 / 8, p 0.23 |
| Full fine-tune | **86.7 %** | 13.3 % | 0.0 % | 2.10 | 70 / 12, p 4.1e-11 |
| Full fine-tune, int8 | 86.6 % | 13.4 % | 0.0 % | 2.15 | 71 / 15, p 7.1e-10 |

The heads-only fine-tune leaves both ball heads untouched, so its ball detections are the published network's.

On games it never saw the published student puts its top ball candidate on the labelled ball on 81.7 % of the frames
(86.1 % on the first 30 clips). The README's 95.7 % on its own two games' held-out clips is a different measure: it
scores the ball path that `nba/analyze.link_ball` links through every frame's candidates (`tools/nba_ball_bench.py`),
not each frame's top candidate, so the two numbers are not comparable. An earlier version of this note said that
int8 gets the same hit, wrong and missed counts as fp32: that held on the first 368 frames only; on all 1,153 the
difference (3 frames against 8) is within chance. The full fine-tune's ball labels came only from the published
network's linked detections on 30 *other* NBA games, and it hits on 70 frames the published network misses against 12
the other way: self-training with the path linker's temporal consistency improved the ball on footage that played no
part in it. The fine-tuned network also reports more ball candidates above 0.08 (704 against 475 on 120 frames,
`runs/bench/int8_ft_full.json`); this benchmark scores the top candidate only.

## 6. Players on broadcasts held out of the fine-tuning

### 6.1 NBA broadcasts held out (heads only, trained on the four NCAA and FIBA broadcasts)

The person heads retrained on the SportsMOT `val` sequences instead (1,500 steps; the two NBA broadcasts and the
FIBA Women's Asia Cup never seen), scored on the `train` sequences (every 10th labelled frame;
`runs/bench/sportsmot_train_swap.json`):

| Model | AP50 | Recall at 0.3 | Precision at 0.3 |
|---|---:|---:|---:|
| Published student | 0.592 | 0.594 | 0.701 |
| Person heads retrained on NCAA and FIBA footage | **0.714** | 0.615 | 0.766 |

On the two NBA broadcasts alone (11 sequences, `runs/bench/sportsmot_nba_swap.json`): AP50 0.599 to **0.721**, recall
0.606 to 0.622, precision 0.693 to 0.760. Recall by broadcast: NBA, Warriors at Oracle 0.689 to 0.690; NBA, Lakers at
New Orleans 0.467 to 0.508; FIBA Women's Asia Cup 0.543 to 0.585. Footage of other leagues sharpens the person head on
NBA broadcasts, most of it in precision (fewer boxes scored 0.3 or more that match no labelled player), but three
retrained heads on a trunk distilled from two games recover little of the recall the NBA broadcasts lack.

### 6.2 NCAA and FIBA broadcasts held out (heads only, and full)

Trained on the SportsMOT `train` sequences (2 NBA broadcasts and the FIBA Women's Asia Cup) and, for the full
fine-tune, the BARD ball pseudo-labels; scored on the 15 `val` sequences (`runs/bench/sportsmot_val_all.json`):

| Model | AP50 | Recall at 0.3 | Precision at 0.3 | F1 |
|---|---:|---:|---:|---:|
| Published student | 0.675 | 0.626 | 0.793 | 0.700 |
| Person heads retrained | 0.796 | 0.709 | 0.830 | 0.765 |
| Full fine-tune | **0.922** | **0.879** | 0.921 | 0.900 |
| Full fine-tune, int8 | 0.921 | 0.881 | 0.920 | 0.900 |

| Held-out broadcast | Published | Heads | Full |
|---|---:|---:|---:|
| NCAA, Kentucky v Wichita State | 0.824 | 0.847 | 0.886 |
| FIBA Asia Championship, Philippines v Palestine | 0.730 | 0.809 | 0.931 |
| FIBA Women's World Cup 2018, Australia v USA | 0.406 | 0.546 | 0.881 |
| FIBA, USA v Russia (Istanbul) | 0.517 | 0.639 | 0.839 |

The full fine-tune finds 88 % of the players on four broadcasts it never saw, where the published network found 63 %;
the broadcast it handled worst goes from 41 % to 88 %. The model was taken at its last step, not chosen on these
sequences (a 90-frame sample of them was printed during training for monitoring only).

## 7. Limits, and what was tried and dropped

* **The two games of the README were not measured**: their footage is not in the container. The fine-tunes' effect
  there has to be measured on the Mac against `train_student.evaluate`'s held-out frames before anyone uses them in
  the pipeline. (NBA players held out of a full fine-tune: §8.3.)
* **Players only**: the fine-tuned person head was taught that referees, benches and the crowd are not people to box.
  The pipeline keeps referees out of the teams anyway (`analyze.is_referee`), but it also hands every tracked
  person's box, referees included, to the ball tracker as the people near the ball (`perception/pipeline.py`); what
  losing the referees' boxes does there was not measured.
* **Rights**: the fine-tuned weights learnt from CC BY-NC labels and NBA footage, so they are research-only. The
  round-2 fine-tune is published for research in `models/research/` (sources and terms in its README) and is not
  shipped: the app's browser export is unchanged.
* **Labels**: the ball references are the RF-DETR teacher's linked path, not human labels, as for the README's
  95.7 %; 30 games at three clips each is breadth, not depth.
* **The simulator's monocular blocks** (held-out F1 0.25) were studied and left alone. On the train seeds 1-6 the
  engine finds 3 of 19 blocks; a blocked shot it misses becomes a steal and a turnover, a miss called seconds later, or
  nothing. A learned check where a defender gains a released ball that is not a shot (the steal decision) was
  prototyped: over the six games that point saw 128 candidates holding only 3 of the 19 blocks, too few for any
  classifier there to matter, so nothing was changed.

## 8. Round 2: faster training, iterated self-training, NBA players held out

Every model in this section starts from the same exact copy of the published export (`student_from_onnx`), keeps
round 1's recipe (stem and first stage frozen, batch norms frozen, AdamW 2e-4 with cosine decay, batch 4, hue ±30°,
scale 0.7-1.4, ball loss weight 2) and is taken at its last step. Ball scores are on the 1,153 teacher-labelled frames
of §5 (per-frame top candidate unless marked linked); paired differences are exact McNemar tests on the frames where
exactly one of the two models hits.

### 8.1 Training 3.6 times faster on the same CPU

bf16 autocast for the forward pass plus channels-last tensors (oneDNN's preferred layout; the container's Xeon has
AMX): one training step went from 1.34 s to 0.37 s in a controlled A/B (fp32 channels-last alone 1.01 s, bf16 alone
0.61 s). The full 3,000-step runs took 115 min (round 1) and 30 min (T1). T1 is round 1's data and recipe plus bf16,
channels-last and balanced sampling (each data folder drawn equally often): players on the held-out NCAA and FIBA
broadcasts AP50 0.917 (round 1 0.922), F1 0.903 (0.900); ball 86.9 % (86.7 %; 24 frames only T1 hits, 22 only round
1, p 0.88). Same model quality, a quarter of the time.

### 8.2 Iterated self-training (round 2)

Round 2 re-labelled the ball on 390 clips of the 30 calibration games (35,100 frames, 4.3 times round 1's 90 clips)
with the round-1 network (int8), linked into one path per clip as before (32,053 frames labelled, 346 of them filled).
T2 = T1's recipe on these labels:

| Model | Players AP50 (val) | Players F1 | Ball hit | Wrong | vs round 1 (only it / only round 1, p) |
|---|---:|---:|---:|---:|---|
| Published student | 0.675 | 0.700 | 81.7 % | 18.0 % | |
| Round 1 (full fine-tune) | 0.922 | 0.900 | 86.7 % | 13.3 % | |
| T1 (bf16, balanced) | 0.917 | 0.903 | 86.9 % | 13.1 % | 24 / 22, p 0.88 |
| T2 (round-2 labels) | 0.918 | 0.903 | **89.1 %** | 10.9 % | 37 / 10, p 9.9e-5 |

Against the published network T2 hits on 96 frames it misses and misses 11 it hits (p 4.3e-18). Two things changed
at once (a better labeller and 4.3 times the clips), so this does not say which one helped. T2's int8 export
(`tools/quantize_student.py`, as in §3.2) hits 89.2 % (2 frames fp32 misses, none the other way, p 0.5); against fp32
on 120 held-out inputs 98.2 % of its person boxes and 91.6 % of its ball points match. Both files are in
[`models/research/`](../../models/research/README.md), research use only.

### 8.3 NBA players held out of a full fine-tune (T3)

The round-1 full fine-tune had seen the two NBA SportsMOT broadcasts' player boxes (§6.2 holds out NCAA and FIBA
only). T3 is T2's
recipe with the person labels swapped to the four NCAA and FIBA broadcasts (`val`) and the round-2 NBA ball labels kept
(those carry no person loss, so NBA video is seen but no NBA player is labelled). Scored on the 11 NBA sequences
(every 10th labelled frame, 1,014 frames; `runs/bench/sportsmot_nba_t3.json`):

| Model | AP50 | Recall at 0.3 | Precision at 0.3 | F1 |
|---|---:|---:|---:|---:|
| Published student | 0.599 | 0.606 | 0.693 | 0.647 |
| Person heads retrained on NCAA and FIBA (§6.1) | 0.721 | 0.622 | 0.760 | 0.684 |
| T3, full fine-tune, NBA players held out | **0.860** | **0.736** | **0.914** | **0.815** |

Recall by broadcast: Warriors at Oracle 0.689, 0.690, 0.706; Lakers at New Orleans (a re-streamed local feed) 0.467,
0.508, **0.787**. With the trunk free to change, footage of other leagues lifts NBA player detection far more than the
retrained heads did; the broadcast that was already handled best gains little.

## 9. Round 3: NBA human labels and temporal dropout

T4 is T2's recipe on all seven SportsMOT broadcasts (`train` and `val`) plus the round-2 ball labels, so no SportsMOT
broadcast is held out of it. T5 adds human labels of 3 NBA games of Roboflow Universe's "basketball-player-detection-2"
set (CC BY 4.0; Hugging Face `koppolusameer/basketball-coco-player-detection`; `prep-coco`: 1,071 single frames,
players boxed and the ball where labelled, referees left out) and *temporal dropout*: for a quarter of the samples the
two earlier input frames are replaced by the current one (`--static-p 0.25`), so the network also learns to work
without motion. 4,000 steps; the four data folders are sampled equally often.

**Held-out NBA games, human labels.** 5 other games of the same set (211 frames of 7 clips; 1,961 player boxes,
2,467 with the referees; the ball labelled on 194 frames), never trained on (`tools/detector_bench.py --coco`; IoU
0.5, recall and precision at 0.3; ball within 15 px of 1080 lines of frame height). The frames are sampled sparsely
from their clips, so each one is fed as its own history (the frame three times):

| Model | Players AP50 | Recall | Precision | Players and referees AP50 | Ball hit | vs published (only it / only published, p) |
|---|---:|---:|---:|---:|---:|---|
| Published student | 0.396 | 0.396 | 0.691 | 0.505 | 22.7 % | |
| Round 1, full fine-tune | 0.743 | 0.710 | 0.796 | 0.689 | 33.5 % | 26 / 5, p 1.9e-4 |
| T2 (round 2, [`student-ft-r2`](../../models/research/README.md)) | 0.761 | 0.733 | 0.820 | 0.687 | 34.5 % | 29 / 6, p 1.2e-4 |
| T3 (no NBA player label) | 0.748 | 0.691 | 0.846 | 0.632 | 34.5 % | 29 / 6, p 1.2e-4 |
| T4 (all seven SportsMOT broadcasts) | 0.770 | 0.725 | 0.850 | 0.674 | 34.5 % | 31 / 8, p 2.9e-4 |
| T5 (+3 NBA games' human labels, temporal dropout; [`student-ft-r3`](../../models/research/README.md)) | **0.870** | **0.821** | **0.913** | **0.768** | **68.0 %** | 90 / 2, p 1.7e-24 |
| T5, int8 | 0.869 | 0.821 | 0.914 | 0.767 | 67.5 % | 90 / 3, p 2.7e-23 |

T5's ball gain here is mostly about the missing motion: only T5 trained on repeated frames, and the earlier models'
ball heads lean on the difference between frames, which these inputs do not have. Its 3 human-labelled games also come
from the same set as the 5 held-out ones (other games, the same labelling), which favours it over models that never
saw this set's labels. The player gain over T4 (AP50 0.770 to 0.870, precision 0.850 to 0.913) is the clearer
evidence, though the shared labelling still favours T5 there. On the ball T5 hits 68 frames T4 misses against 3 (p
5.1e-17). The person head was taught to leave referees alone, so with them counted as people (2,467 boxes) recall stays
lower (T5 0.663).

**Video.** On the 1,153 BARD frames, with three real frames: T4 89.4 %, T5 89.5 % (T2 89.1 %; T5 against T2 27 frames
to 22, p 0.57; against T4 23 to 22, p 1.0; int8 89.4 %). Temporal dropout and the extra labels did not cost the
ball in video. T5 and its int8 export are in [`models/research/`](../../models/research/README.md) as
`student-ft-r3`; by the benchmarks here it is the best model so far, but no SportsMOT broadcast is held out of it.
