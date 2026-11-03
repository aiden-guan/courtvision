# Research weights: the NBA student detector fine-tuned on more broadcasts

Fine-tuned copies of the NBA student detector (`BallPersonNet`, `courtvision/perception/student.py`), in the
browser/CPU ONNX interface of the app's `courtvision/server/static/nba-student.onnx`: input `frames`, float32
`[1, 9, 544, 960]` (three frames, the current one last, `preprocess_frames`); outputs `ball_hm`, `ball_off`,
`person_hm`, `person_wh`, `person_off`. Same graph and the same cost per frame as the app's model, which these files do
**not** replace. How they were made and measured: [docs/research/cross_game_detector.md](../../docs/research/cross_game_detector.md).

**Research and evaluation only** (see [Sources and terms](#sources-and-terms)).

| File | What | Bytes | sha256 |
|---|---|---:|---|
| `student-ft-r2.onnx` | Round 2 (T2), fp32 | 47,342,538 | `0025181de6bb7973317483a51bcf7adb0bd2be4e52629e611e66efb4cd4350f0` |
| `student-ft-r2-int8.onnx` | the same, int8 (QDQ) | 12,651,412 | `28a3e238177a810ae7ffea306ac26a65ecb9e597390a1a748362466595a18924` |

## Results (footage never trained on)

| Benchmark | App model | `student-ft-r2` |
|---|---:|---:|
| Players, 4 NCAA and FIBA broadcasts (SportsMOT `val`: 15 sequences, 1,258 frames, 11,901 human boxes), AP50 | 0.675 | **0.918** |
| the same, recall / precision at the pipeline's 0.3 | 0.626 / 0.793 | **0.870 / 0.939** |
| Ball, 30 NBA games of 2024-25 (90 BARD clips, 1,153 frames labelled by the RF-DETR teacher recipe): top candidate within 15 px of 1080p | 81.7 % | **89.1 %** |

On the ball frames `student-ft-r2` hits 96 that the app model misses and misses 11 that it hits (exact McNemar test,
p 4.3e-18). Not measured: the two games the app model was trained on (README), the full pipeline's numbers, and NBA
players held out (this model trained on the two NBA SportsMOT broadcasts; the research note's T3 holds them out).

The int8 file against fp32 on 120 held-out BARD inputs: 98.2 % of person boxes and 91.6 % of ball points matched
(`tools/quantize_student.py`); on the 1,153 ball frames it hits 89.2 % (2 frames fp32 misses, none the other way,
p 0.5).

## How it was trained

`tools/student_finetune.py` from the exact PyTorch copy of the app's export (`courtvision/training/student_from_onnx.py`):

* players: human boxes of the 15 SportsMOT basketball `train` sequences (2 NBA broadcasts and 1 FIBA; every third frame,
  3,733 samples), players on the court only;
* the ball: pseudo labels on 390 BARD clips of the 30 calibration games (90 frames each, 35,100 frames), the round-1
  fine-tune's (int8) detections linked into one path per clip (`training.ball_labels.link_ball_path`): 32,053 frames
  labelled (346 of them filled), weight at most 0.5, 10,664 samples;
* 3,000 steps, batch 4, 512 px crops, AdamW (lr 2e-4, weight decay 1e-4, 100 warm-up steps, cosine decay); stem and
  first stage frozen, batch norms frozen; hue turned by up to ±30° on frames without a ball label, scale 0.7-1.4; ball
  loss weight 2; both folders sampled equally often; bf16 autocast and channels-last; seed 0; the last step kept.

```bash
.venv/bin/python tools/student_finetune.py train --data data/ft/mot_train --data data/ft/bard_calib_r2 \
    --init runs/models/student-from-onnx.pt --steps 3000 --lr 2e-4 --freeze stem,l1 --hue 30 --zoom-min 0.7 \
    --zoom-max 1.4 --ball-weight 2.0 --threads 4 --workers 2 --bf16 --channels-last --balance \
    --eval-mot data/sportsmot/val --eval-every 1000 --eval-frames 6 --out runs/models/ft-t2
```

int8: static post-training quantization (`tools/quantize_student.py --method percentile --activation u8 --n-calib 64`),
weights int8 per output channel, activations 8-bit, the five 1x1 output convolutions left in fp32.

## Using them

```bash
.venv/bin/python tools/detector_bench.py --models courtvision/server/static/nba-student.onnx,models/research/student-ft-r2.onnx --split val --sparse
```

```bash
.venv/bin/python tools/nba_analyze.py runs/nba/<game> detect --backend onnx --weights models/research/student-ft-r2.onnx
```

```bash
.venv/bin/python -m courtvision.training.student_from_onnx models/research/student-ft-r2.onnx --out runs/models/student-ft-r2.pt
```

The last command rebuilds a trainable checkpoint for `student_finetune.py train --init`; rebuilt from this file, every
tensor equals the training checkpoint's (largest difference 0.0). The Studio uses a detector placed at
`models/nba-student.onnx` in its data folder.

## Sources and terms

The weights learnt from:

* **SportsMOT** (Cui et al., "SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes", ICCV 2023;
  <https://github.com/MCG-NJU/SportsMOT>): the player boxes. Licence **CC BY-NC 4.0**: non-commercial use, with credit.
* **BARD** (Giudici et al., CVIU 2026; <https://github.com/GabrieleGiudic/BARD>): the NBA.com clips the ball labels come
  from. Dataset licence CC BY 4.0; the footage is the NBA's.
* the app's NBA student, which these start from: distilled from an RF-DETR teacher (Apache-2.0) on two NBA broadcasts.

Under this repo's own rules ([docs/TRAINING.md](../../docs/TRAINING.md) §1) a model trained on CC BY-NC data or on
NBA footage is for research and evaluation, never a shipped product. Credit the datasets above when you use these
weights.
