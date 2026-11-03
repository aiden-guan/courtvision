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

Both are research-only here (docs/TRAINING.md §1). Nothing from them is committed: footage stays under `data/`
(git-ignored), and the only artefacts are tools, tests and the numbers below.

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

The SportsMOT labels have no ball, so the ball is checked on NBA footage held out of everything: one clip from each
of the 30 BARD check games, labelled by the student's own teacher recipe (`tools/nba_teacher_labels.py`: RF-DETR
Small, COCO, at 1312 x 736, threshold 0.04, linked by `link_ball_path`) on 24 frames at ~30 fps each, 720 frames in
all: 402 carry a ball label of weight 0.5 or more (373 detected, 29 filled), 368 of them with two frames before them
to score. `tools/detector_bench.py --ball-folder` counts the network's top ball (threshold 0.08) within 15 px of 1080p
of the label (hit), elsewhere (wrong) or nowhere (missed), as the README's "ball found within 15 px" does on the two
training games' own held-out clips (95.7 %).

| Model | Hit | Wrong | Missed | Median error (px of 1080p) |
|---|---:|---:|---:|---:|
| Published student, fp32 | 86.1 % | 13.6 % | 0.3 % | 2.12 |
| int8 | 86.1 % | 13.6 % | 0.3 % | 2.18 |
| Heads-only fine-tune | 86.1 % | 13.6 % | 0.3 % | 2.12 |
| Full fine-tune | **90.8 %** | 9.2 % | 0.0 % | 2.05 |
| Full fine-tune, int8 | 90.5 % | 9.5 % | 0.0 % | 2.14 |

On games it never saw the published student puts its best ball candidate on the labelled ball 86.1 % of the time
against 95.7 % on its own games. The int8 model gets the same hit, wrong and missed counts. The full fine-tune's
ball labels came only from the published network's linked detections on 30 *other* NBA games, and it puts its top
ball on the teacher's label on 90.8 % of these frames (wrong 13.6 % to 9.2 %): self-training with the path linker's
temporal consistency improved the ball on footage that played no part in it. The fine-tuned network also reports more
ball candidates above 0.08 (704 against 475 on 120 frames, `runs/bench/int8_ft_full.json`); this benchmark scores the
top candidate only, so what the extra candidates do to the path linking (`nba/analyze.link_ball`) was not measured.

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

* **No NBA person labels are held out of the full fine-tune**: its person training includes the two NBA SportsMOT
  broadcasts. §6.1 shows the NBA direction for the person heads only. The full model's effect on NBA players, and on
  the two games of the README, has to be measured on the Mac against `train_student.evaluate`'s held-out frames
  before anyone uses it there.
* **Players only**: the fine-tuned person head was taught that referees, benches and the crowd are not people to box.
  The pipeline keeps referees out of the teams anyway (`analyze.is_referee`), but it also hands every tracked
  person's box, referees included, to the ball tracker as the people near the ball (`perception/pipeline.py`); what
  losing the referees' boxes does there was not measured.
* **Rights**: the fine-tuned weights learnt from CC BY-NC labels and NBA footage. They are research-only and are not
  committed or shipped; the published browser export is unchanged.
* **Labels**: the ball references are the RF-DETR teacher's linked path, not human labels, as for the README's
  95.7 %; 30 games at one clip each is breadth, not depth.
* **The simulator's monocular blocks** (held-out F1 0.25) were studied and left alone. On the train seeds 1-6 the
  engine finds 3 of 19 blocks; a blocked shot it misses becomes a steal and a turnover, a miss called seconds later, or
  nothing. A learned check where a defender gains a released ball that is not a shot (the steal decision) was
  prototyped: over the six games that point saw 128 candidates holding only 3 of the 19 blocks, too few for any
  classifier there to matter, so nothing was changed.

