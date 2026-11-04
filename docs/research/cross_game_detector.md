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

**The README's protocol on these games** (the linked ball path, `--link`, round 4; `runs/bench/ball_check12_linked.json`):
the published student 81.9 % (wrong 11.4 %, missed 6.8 %; int8 81.5 %, p 0.34), the round-1 full fine-tune 88.7 %, T4
91.6 % and T5 91.6 % (wrong 3.0 %, missed 5.5 %; against the published student 114 frames to 2, p 1.6e-31). On the
published student's own game the same protocol gives 95.5 % (§10.2).

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

## 10. On the owner's Mac (M3 Pro): the README's two games

### 10.1 Protocol, fixed before any result of this round was read

The fine-tunes above never saw the README's two games, whose footage is only on the Mac. Before any number below was
looked at, the test sets, candidates and measures were fixed as follows.

* **Lakers @ Warriors (2024-12-25), ball and people**: the app model's own held-out clips, `train_student`'s every
  10th clip (`clip % 10 == 5`) of `runs/nba/2024-12-25-LAL-GSW/teacher` (15 clips of 36 frames; labels by the RF-DETR
  teacher recipe). The app model was distilled on the other clips of this game (clips within 2 s of a held-out clip
  left out), so for it these are in-game held-out frames; the fine-tunes never saw this game. Scored with
  `tools/detector_bench.py --ball-folder ... --held-out`: the top ball candidate (threshold 0.08) within 15 px of
  1080p, and with `--link` the clip's linked ball path (the protocol of `tools/nba_ball_bench.py` and the README's
  95.7 %). Also counted: the teacher's people (score >= 0.5) a box at 0.3 matches at IoU 0.5. The teacher boxes
  referees, benches and spectators too, which the fine-tunes were taught to leave alone, so this measure is biased
  against them.
* **Cavaliers @ Warriors (2017 Finals Game 5), ball and people**: no model of this repo has trained on it. New
  teacher labels on 60 clips (`tools/nba_teacher_labels.py --clips 60 --seed 0`, the tool's default mix of clips with
  and without the scorebug), every clip scored the same way. This folder is a test set only: it is listed in
  `cross_game_detector_splits.json` and is never trained or tuned on.
* **The pipeline on Lakers @ Warriors**: the whole game re-detected with a candidate (`nba_analyze.py detect
  --det-dir ...`), then `ball`, `events` and `timeline` as for the README's run `full` (the timeline and shot models
  fitted on one half and scored on the other): team with the ball in settled live play (README 96.2 %), made or missed
  on real shots (84.8 %), real shot or not, free throw or field goal, box-score error. The app model's run is
  repeated with the current code first. Court registration reads no detections (`nba_track_court.py`), so it does not
  change with the detector and is not re-measured.
* **Candidates**: the app model (`courtvision/server/static/nba-student.onnx`), `models/research/student-ft-r2.onnx`
  and its int8 file, and later models from the cloud rounds (T4, T5, the soups) or this round as they are made. Every
  candidate tried is reported. Paired ball comparisons are exact McNemar tests on the frames only one model hits.

### 10.2 The ball and people on the two games

Every file runs on ONNX Runtime (fp32 or int8, the sparse heads of §3.1) on the Mac's CPU, so the backend is the
same for every model (`runs/bench/local/*.json`, not in git). T5 (§9, `student-ft-r3`) was scored when it landed, on
the same frames, with the candidates named in §10.1.

**Lakers @ Warriors**, the app model's 15 held-out clips (243 frames with a ball label and two frames before them):

| Model | Top candidate: hit / wrong / missed | Linked path: hit / wrong / missed | Median error |
|---|---:|---:|---:|
| App model | **95.5 %** / 4.5 / 0.0 | **95.5 %** / 3.7 / 0.8 | 2.34 px |
| Round 2 (T2) | 94.2 % / 5.8 / 0.0 | **95.5 %** / 3.7 / 0.8 | 2.18 px |
| Round 2, int8 | 93.8 % / 6.2 / 0.0 | **95.5 %** / 3.7 / 0.8 | 2.19 px |
| T5 (round 3) | 95.1 % / 4.9 / 0.0 | **95.9 %** / 3.7 / 0.4 | 2.13 px |
| T5, int8 | **95.5 %** / 4.5 / 0.0 | 95.5 % / 3.7 / 0.8 | 2.13 px |

Top candidate, the app model against round 2: 3 frames only the app model hits, none the other way (p 0.25); against
T5 1 and 0. The linked paths hit the same 232 frames for the app model and round 2, T5 one frame more. (The README's 95.7 % is the same protocol on 256 frames: it also
scores each clip's first two frames, with their history padded.) On the game it was distilled on, the app model's ball
is matched, not beaten, by a model that never saw this game.

People against the teacher's boxes (score >= 0.5), each box placed on the court through the game's camera track
(`camera_parts`, its foot point within 2 ft of the floor) and tested with the pipeline's referee check
(`analyze.is_referee`):

| Teacher boxes | App model | Round 2 | T5 |
|---|---:|---:|---:|
| players on the court (1,311) | **99.0 %** | 95.9 % | 95.2 % |
| referees on the court (230) | 99.1 % | 27.0 % | 10.0 % |
| no court fit on the frame (close-ups, replays; 526) | 73.6 % | 52.5 % | 51.0 % |
| all (2,075) | 92.5 % | 77.2 % | 74.2 % |

Round 2 finds 41 fewer of the 1,311 players on the court than the app model on its own game, and has learnt from the
SportsMOT labels to leave the referees out (§7). What either costs the pipeline is §10.3.

**Cavaliers @ Warriors**, 60 new teacher clips of a game no model trained on (606 frames; the teacher labels the ball
on 642 of 2,160 frames, fewer than on the 1080p game: 720p, 2017):

| Model | Top candidate: hit / wrong / missed | vs app (only it / only app, p) | Linked: hit / wrong / missed | vs app |
|---|---:|---|---:|---|
| App model | 75.7 % / 22.4 / 1.8 | | 72.4 % / 6.4 / 21.1 | |
| Round 2 (T2) | **78.1 %** / 21.9 / 0.0 | 20 / 6, p 0.0094 | **77.9 %** / 15.0 / 7.1 | 36 / 3, p 3.6e-8 |
| Round 2, int8 | 77.7 % / 22.3 / 0.0 | 20 / 8, p 0.036 | 78.6 % / 16.0 / 5.4 | 41 / 4, p 9.3e-9 |
| T5 (round 3) | 77.9 % / 22.1 / 0.0 | 25 / 12, p 0.047 | **78.7 %** / 12.2 / 9.1 | 45 / 7, p 7.0e-8 |
| T5, int8 | 77.7 % / 22.3 / 0.0 | 25 / 13, p 0.073 | 78.6 % / 12.4 / 9.1 | 45 / 8, p 2.4e-7 |

T5 against round 2: top candidate 15 frames only T5 hits, 16 only round 2 (p 1.0); linked 19 and 14 (p 0.49).

On this game the app model's linked path loses the ball on 21 % of the frames (its candidates there are too weak to
link); round 2 loses it on 7 %. People against the teacher's boxes, split as above (the court is registered on fewer
frames of this game):

| Teacher boxes | App model | Round 2 | T5 |
|---|---:|---:|---:|
| players on the court (3,535) | 80.2 % | **81.5 %** | 80.1 % |
| referees on the court (116) | 83.6 % | 12.1 % | 12.1 % |
| off the court: benches, the crowd (226) | 76.1 % | 4.9 % | 7.5 % |
| no court fit on the frame (836) | 47.6 % | 52.0 % | 51.2 % |
| all (4,713) | 74.3 % | 70.9 % | 69.9 % |

T5's gains on the Roboflow set's single frames (§9) do not show on these two games' video: its ball and people are
round 2's, within chance.

### 10.3 The pipeline on Lakers @ Warriors

The whole game re-detected on the Neural Engine (Core ML, `nba_analyze.py detect --backend coreml`: 12 ms a frame,
20 minutes for the game) with each model, then `ball`, `events` and `timeline` with the current code; the timeline,
shot and free-throw models are fitted on one half of the game and scored on the other, both ways. The README's run
`full` (the app model on PyTorch/MPS, older code: 96.2 %, 84.8 %) re-run with the current code gives 96.3 % and
83.6 %, and on Core ML 96.4 % and 83.1 %: the backend changes little, the code a little more, so the two models are
compared on the same backend and code. Shots are paired through the play-by-play shot each candidate is matched to
(`label_candidates`); possession is reported per half, not tested per quarter second (neighbouring steps are not
independent).

| | App model | Round 2 (T2) | T5 (round 3) |
|---|---:|---:|---:|
| Team with the ball in settled live play, engine alone | 72.8 % | **79.1 %** | 69.1 % |
| the same, the possession timeline (first half / second half) | 96.4 % (96.5 / 96.2) | **97.0 %** (97.4 / 96.7) | 96.8 % (97.1 / 96.4) |
| Changes of possession found within 2.5 s: precision / recall (195 true) | 0.72 / 0.57 | 0.70 / 0.56 | **0.77 / 0.62** |
| Candidates (real shots among them) | 246 (177) | 262 (178) | 281 (178) |
| Real shot or not, all candidates | **78.5 %** | 73.7 % | 76.9 % |
| Free throw or field goal, real shots | 94.9 % | 94.9 % | 95.5 % |
| Made or missed, real shots | 83.1 % | 85.4 % | **87.1 %** |
| the same, paired with the app model over the shots both found: only it right / only the app, p | | 17 / 12 of 174, p 0.46 | 16 / 10 of 173, p 0.33 |
| False candidates kept as shots | 28 of 69 | 37 of 84 | 35 of 103 |
| Box-score error (timeline events) | 0.209 | 0.225 | **0.208** |
| Steals F1 (8 true) / turnovers F1 (22 true) | **0.40 / 0.48** | 0.14 / 0.33 | 0.37 / 0.38 |
| Ball path over the game | 81,654 frames, 960 flights | 89,648 frames, 995 flights | 90,124 frames, 992 flights |

T5 against round 2 on the shots both found: made or missed 9 and 8 (p 1.0). On the game the app model was
distilled on, neither fine-tune is a clear gain. Its ball path covers 10 % more frames and the
engine's own possession calls improve by 6 points, the timeline's by 0.6; made or missed is 2 points higher, within
chance (p 0.46). Against that, the longer ball path gives more rim visits and engine results that are not shots (84
false candidates against 69), so the real-or-not decision falls by 5 points, and steals and turnovers (8 and 22 in
the game) are found less well.

### 10.4 The pipeline on Cavaliers @ Warriors

The same comparison on the second game, which neither model trained on (`events --team-refit`, as for the README's
run `base2`; the app model's run with the current code reproduces `base2` exactly). Its detections were made on Core ML
for both models.

| | App model | Round 2 (T2) | T5 (round 3) |
|---|---:|---:|---:|
| Team with the ball in settled live play, engine alone | 61.0 % | 58.4 % | **61.5 %** |
| the same, the possession timeline (first half / second half) | **75.8 %** (70.1 / 81.4) | 74.0 % (70.4 / 77.5) | 74.8 % (71.4 / 78.1) |
| Candidates (real shots among them, of 229 in the play-by-play) | 254 (114) | 291 (120) | 284 (118) |
| Real shot or not, all candidates | 52.4 % | **63.9 %** | 57.8 % |
| Free throw or field goal, real shots | 91.2 % | 93.3 % | **94.1 %** |
| Made or missed, real shots | 75.4 % | 66.7 % | **78.8 %** |
| the same, paired with the app model over the shots both found: only it right / only the app, p | | 11 / 18 of 108, p 0.27 | 14 / 12 of 108, p 0.85 |
| False candidates kept as shots | 61 of 140 | 46 of 171 | 53 of 166 |
| Box-score error (timeline events) | **0.570** | 0.581 | 0.574 |

Round 2's and T5's better ball on this game (§10.2: linked path 72.4 % to 77.9 % and 78.7 %) does not reach the
pipeline's numbers: the
possession timeline and made-or-missed move within chance (made or missed 18 shots against 11, p 0.27), and the
real-or-not decision improves because fewer false candidates are kept. What limits this game is upstream of the
detector: the court is registered on about half of its frames (research note `made_shots_at_the_rim.md` §4), and the
engine reports a candidate for only 114-120 of its 229 shots.

### 10.5 What this round says about the fine-tunes where the app is shown

* On the app model's own game the fine-tune matches its ball (95.5 % linked, the same frames), finds 3 points fewer
  of the players on the court, drops the referees, and leaves the pipeline about where it was: possession slightly
  better, shot decisions within chance, fewer steals and turnovers found (8 and 22 in the game).
* On the second game, never trained on, the fine-tune's ball is clearly better (top candidate 75.7 % to 78.1 %, p
  0.0094; linked 72.4 % to 77.9 %, p 3.6e-8) and it finds slightly more players, but the pipeline's numbers do not
  move beyond chance.
* T5 (the cloud's round 3) behaves like round 2 on both games: its ball and people are round 2's within chance, and
  so are its pipeline numbers (made or missed 87.1 % and 78.8 % against the app model's 83.1 % and 75.4 %, p 0.33 and
  0.85).
* So the fine-tunes measured so far are not yet a reason to replace the app model: the detector gains on footage it
  never saw are real, but on these two games the pipeline's stages after the detector decide the results.

### 10.6 Training on the Mac's GPU

`student_finetune.py train --device mps` (PyTorch's Metal backend). One A/B on the same data (the Lakers @ Warriors
teacher folder) and start, 60 steps of the recipe each, timed from step 20 to step 60: MPS fp32 0.33 s a step, bf16
autocast 0.30 s, fp16 autocast 0.35 s, channels-last 0.33 s; the CPU (6 threads of the M3 Pro) 0.83 s in fp32 and
11.4 s with bf16 autocast, which Apple's CPU has no fast path for (on the container's Xeon it was the fast path,
§8.1). Four data-loader workers keep the GPU fed (8 workers: the same 0.32 s over 100 steps). The losses on MPS and
the CPU agree to the fourth decimal. A 3,000-step run takes about 16 minutes on MPS, half the container's 30 with
AMX. Getting there needed one fix: data-loader workers on macOS start by spawn and must unpickle the dataset, which
was a class defined inside a function.

### 10.7 Two training runs on the Mac: plan, fixed before training

Two runs, each changing one thing against the one before it, from the same start (the exact copy of the app model,
`student_from_onnx`) with round 2's recipe (§8: 3,000 steps, batch 4, AdamW 2e-4 cosine, stem and first stage
frozen, batch norms frozen, hue ±30°, scale 0.7-1.4, ball loss weight 2, folders sampled equally, seed 0, the last
step kept):

* **M1-ctrl**: players from the 15 SportsMOT `train` sequences, the ball from pseudo labels on the 390 BARD clips of
  the 30 calibration games. Round 2's labels came from the round-1 network, which is not on this machine, so these
  are made by round 2's int8 file (`models/research/student-ft-r2-int8.onnx`): a further round of self-training
  (not the cloud's round 3 of §9, which kept round 2's labels).
* **M1**: M1-ctrl plus the Lakers @ Warriors teacher folder the app model was distilled on (players, referees and
  the crowd as the teacher boxes them, and the ball), with `train_student`'s held-out clips left out (`plan_data`
  holds out every 10th clip of a teacher folder and the clips within 2 s of them).

Scored once each, after training, on the sets of §10.1 (the Lakers @ Warriors held-out clips, the 60 Cavaliers @
Warriors clips, which stay out of all training), the SportsMOT `val` players (§6.2) and the BARD check games' ball
(§5, labelled here again by the teacher recipe on the Mac's GPU, so the numbers are compared within this machine).
Nothing is tuned on these sets; the runs' own training logs print a few SportsMOT `val` frames for monitoring only.

## 11. Referees and the crowd (cloud): plan, fixed before training

The owner, watching the app: people in the crowd are often boxed as players, and referees could be found better. The
detector has one person class; the NBA pipeline tells referees apart afterwards by their shirts
(`nba/analyze.is_referee`: an achromatic torso with dark stripes) and the browser app not at all. §10.2 measured both
halves of the problem on the README's two games: the app model boxes 76 % of the people off the court (benches, the
crowd) and 99 % of the referees; the fine-tunes, taught by SportsMOT's player-only labels that everyone else is
background, box 5-8 % of the first and 10-27 % of the second. No new output is added; the aim is a person head that
boxes the players and the referees and leaves the crowd alone.

Two changes, each tried on T5's recipe (§9) with nothing else changed:

* **R4a**: the Roboflow games' referees count as people (`prep-coco --referees`; until now they were left out like
  SportsMOT's). The crowd, benches and coaches, never labelled there, stay background.
* **R4b**: R4a plus *ignore regions* on SportsMOT (`mark-ignore`): the app model's people on each SportsMOT sample
  that overlap no labelled player (IoU below 0.3) but stand among the players (foot point inside the hull of the
  labelled players' foot points, grown by 5 % of the frame height) are left out of the person heatmap loss
  (`ClipDataset`'s `ignore` boxes, `focal_loss(pixel_weight=...)`). These are mostly referees; people in the stands
  fall outside the hull and stay background.

  *Changed before any training*: the hull alone marked only 521 boxes on 494 of SportsMOT's 4,125 `train` samples. On
  40 random samples the app model found 79 people no player label covers; 69 of them stood more than 5 % of the frame
  height outside the hull, and the drawn frames showed them to be referees at the sidelines and baselines (the learned
  check of §11.1 read 61 of the 69 as referees). So a person is also ignored when that check reads them as a referee,
  wherever they stand (`mark-ignore --referees`, the default); everyone else stays background. No result of R4a or R4b
  existed when this was changed.

Measures, on the 5 held-out Roboflow games (§9; `detector_bench.py --coco`, detections at 0.3, IoU 0.5): the share of
the 506 labelled referees a box matches; the share of detections that match no labelled player or referee (the crowd,
benches and coaches; also per frame); players AP50, recall and precision; and the referee rule's calls on the matched
boxes. On the 1,153 BARD frames: the ball (§5), so that it is not paid for. Candidates: T5, R4a, R4b, with the app
model and T2 for reference; all reported. The referee rule is also scored on the labelled boxes themselves, per game,
so that any change to it would be fitted on the 3 training games and tested on the 5 held-out ones.

### 11.1 Telling referees from players: a learned check (done)

The shirt rule, scored on the human labels of the Roboflow set (every labelled player and referee box), misses about a
quarter of the referees and calls many players in dark or grey jerseys referees (in Lakers v Timberwolves a third of
the players). `nba/referee.py` replaces it as the pipeline's default (`analyze.is_referee`; `method="shirt"` keeps the
rule): a logistic regression on 24 statistics of two parts of the box, the shirt and the trousers, each resampled to
a fixed size so that the frame's resolution does not matter (saturation and brightness quantiles, the shares of dark,
bright grey and coloured pixels, the coloured pixels' hues, brightness changes across and down for stripes). Every
labelled box is a sample, with two copies moved and resized by about 6 % as a detector's boxes are; the two classes
are weighted equally and the threshold is fixed at 0.5 (`tools/referee_classifier.py`).

All 8 games of the set were left out one at a time (fitted on the other 7, scored on the boxes of the one left out;
`runs/bench/referee_logo.json`):

| Game left out | Rule: referees called | Rule: players called referee | Rule F1 | Learned: referees called | Learned: players called referee | Learned F1 |
|---|---:|---:|---:|---:|---:|---:|
| Celtics v Knicks, game 1 | 101 / 123 | 6 / 453 | 0.878 | 122 / 123 | 3 / 453 | 0.984 |
| Celtics v Knicks, game 2 | 713 / 858 | 216 / 3230 | 0.798 | 766 / 858 | 16 / 3230 | 0.934 |
| Celtics v Magic | 995 / 1315 | 568 / 4946 | 0.691 | 1218 / 1315 | 718 / 4946 | 0.749 |
| Warriors v Rockets, game 3 | 311 / 384 | 218 / 1410 | 0.681 | 365 / 384 | 84 / 1410 | 0.876 |
| Clippers v Nuggets | 782 / 833 | 176 / 2939 | 0.873 | 789 / 833 | 223 / 2939 | 0.855 |
| Lakers v Timberwolves | 573 / 891 | 1065 / 3143 | 0.453 | 854 / 891 | 168 / 3143 | 0.893 |
| Thunder v Nuggets, game 1 | 362 / 530 | 126 / 2251 | 0.711 | 492 / 530 | 453 / 2251 | 0.667 |
| Thunder v Nuggets, game 3 | 672 / 1167 | 89 / 4068 | 0.697 | 1147 / 1167 | 65 / 4068 | 0.964 |
| **All 8, pooled** | 4509 / 6101 | 2464 / 22440 | 0.690 | **5753 / 6101** | **1730 / 22440** | **0.847** |

On the same boxes the learned check is right where the rule is wrong 3,339 times and the other way
1,361 times (exact McNemar p 1.3e-188): referees found 73.9 % to 94.3 %, precision 0.647 to
0.769. It is not better everywhere: in the two games with Denver it calls more players referees than the rule
(Thunder v Nuggets game 1: 453 of 2,251), so some uniforms still pass for a referee's. The weights shipped
(`nba/referee_model.py`) are fitted on all 8 games, so its numbers on the 5 held-out games of §9 are in-sample;
the figures above are the honest estimate for a game it has not seen. It reads all boxes of a frame at once: 1.6 ms
for a frame of 12 people on one core (the rule 0.4 ms), about two extra minutes for a whole game.

The pipeline's numbers on the README's two games (§10.3, §10.4) were measured with the shirt rule; with the learned
check they have to be measured again before the README changes.

Where it runs: the NBA pipeline (`analyze.CachedDetector`, through `are_referees`), the browser app
(`browser/runtime.py`: a person read as a referee is tracked and still counts as a person near the ball, but is never
a player of either team, and is published as a grey "Referee" box), and the Studio's student detector
(`StudentConfig(referees=True)`; off by default elsewhere).

## 12. Round 5: weight averaging (no gain)

Three weight-space ensembles, named before T4 and T5 existed (`student_finetune.py soup`): the uniform average of T4
and T5, of T2, T4 and T5 (a "uniform soup", Wortsman et al. 2022, arXiv:2203.05482), and T5 pulled halfway back to
the shared start (WiSE-FT, alpha 0.5, arXiv:2109.01903). Each fine-tune lies 0.14 to 0.16 of the start's norm away
from it, but the averages only 0.08 to 0.10: the fine-tunes moved in nearly unrelated directions, so an average is
mostly a step back towards the start. Scored as T5 (`runs/bench/ball_check12_r5.json`, `..._linked_r5.json`,
`coco_heldout_r5.json`):

| Model | Ball per frame (1,153) | Ball, linked path | Players AP50, 5 Roboflow games (single frames) | Ball, those frames |
|---|---:|---:|---:|---:|
| T5 | 89.5 % | 91.6 % | **0.870** | **68.0 %** |
| T4 | 89.4 % | 91.6 % | 0.770 | 34.5 % |
| Average of T4 and T5 | 89.6 % (14 / 13, p 1) | 91.8 % (14 / 11, p 0.69) | 0.410 | 53.6 % |
| Average of T2, T4 and T5 | 90.1 % (22 / 15, p 0.32) | 91.9 % (16 / 12, p 0.57) | 0.271 | 48.5 % |
| T5 halfway back to the start | 88.2 % (14 / 29, p 0.032) | 89.4 % (4 / 29, p 1.1e-5) | 0.777 | 49.0 % |

(Paired against T5: frames only the candidate hits / only T5.) In video the averages match T5 on the ball (their
linked paths miss less and are wrong more often: 6.3 % against 3.0 %); on single frames they collapse, because only
T5 learnt to work without motion and averaging it with models that did not undoes that. No average is kept; T5
(`student-ft-r3`) stays the best model.

The new measures of §11 on these frames also show what the fine-tunes did to the referees: of the 506 labelled
referees, T5 boxes 26 and T4 33 (T5 halfway back: 103). The detections that match nobody labelled (the crowd, benches,
coaches) are 7.2 % of T5's (0.60 a frame) and 13.1 % of T4's.

### 11.2 Results: the referees back in the person head

`mark-ignore` (with the referee check) marked 6,920 ignore boxes on 3,506 of SportsMOT's 4,125 `train` samples and
4,277 on 2,935 of its 4,180 `val` samples, about two people a frame. Both runs are T5's recipe (§9) with the changes
of §11; each was scored once (`runs/bench/coco_heldout_r7.json`, `ball_check12_r7.json`).

**The 5 held-out Roboflow games** (211 single frames; 1,961 players and 506 referees labelled; detections at 0.3,
IoU 0.5). "On nobody" counts the detections that overlap no labelled person at IoU 0.1: the crowd, benches and staff
(most unmatched detections are loose boxes on a labelled person instead):

| Model | Referees found | Players and referees: AP50 / recall / precision | Detections on nobody (per frame) | Ball |
|---|---:|---:|---:|---:|
| Published student | 187 (37.0 %) | 0.505 / 0.391 / 0.857 | 18 (0.09) | 22.7 % |
| T2 | 84 (16.6 %) | 0.687 / 0.617 / 0.868 | 5 (0.02) | 34.5 % |
| T5 | 26 (5.1 %) | 0.768 / 0.663 / 0.928 | 12 (0.06) | 68.0 % |
| R4a: + Roboflow referees as people | **424 (83.8 %)** | **0.881** / **0.831** / 0.912 | 41 (0.19) | 63.9 % |
| R4b: + SportsMOT ignore regions | **424 (83.8 %)** | **0.881** / **0.828** / 0.909 | 35 (0.17) | 66.5 % |

On R4b's 424 matched referees the learned check of §11.1 calls 407 referees (the shirt rule 332) and calls 41 of its
1,618 matched players referees (2.5 %): with the pipeline's check, 80 % of the labelled referees are boxed and named.
Players-only AP50 falls (T5 0.870, R4a 0.752, R4b 0.732) because that measure counts every referee box as a false
player. The published student's low numbers here are those of single frames (it leans on motion: §9); on video it
boxes 99 % of the referees on the court and 76 % of the people off it (§10.2).

The price is a few more boxes on nobody: 0.17 a frame for R4b against T5's 0.06 (35 against 12 over 211 frames).
Single-frame ball: T5 against R4a 10 frames to 2 (p 0.039), against R4b 9 to 6 (p 0.61). **In video** (the 1,153
BARD frames) the ball does not move: T5 89.5 %, R4a 89.3 % (11 / 13, p 0.84), R4b 89.25 % (7 / 10, p 0.63).

R4a and R4b are alike on these NBA games, where the Roboflow referee labels already teach the referees; the ignore
regions keep SportsMOT's NCAA and FIBA referees from being taught as background, which no held-out set here can show
(both runs train on every SportsMOT broadcast). R4b is kept: the same referees, a few fewer boxes on nobody, and its
single-frame ball within chance of T5's. Not yet measured: what it boxes off the court on video, the owner's complaint,
which the README's two games can answer (§10.2's table).
