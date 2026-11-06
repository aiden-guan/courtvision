# Research weights: the NBA student detector fine-tuned on more broadcasts

Fine-tuned copies of the NBA student detector (`BallPersonNet`, `courtvision/perception/student.py`), in the
browser/CPU ONNX interface of the app's `courtvision/server/static/nba-student.onnx`: input `frames`, float32
`[1, 9, 544, 960]` (three frames, the current one last, `preprocess_frames`); outputs `ball_hm`, `ball_off`,
`person_hm`, `person_wh`, `person_off`. Same graph and the same cost per frame as the app's model, which these files do
**not** replace. How they were made and measured: [docs/research/cross_game_detector.md](../../docs/research/cross_game_detector.md)
(§8 round 2, §9 round 3, §11 round 4).

**One main model** (the owner's decision, 2026-11-03). The detector has one main model: the app's own
`courtvision/server/static/nba-student.onnx`. Every round starts from the current main model and is checked on the
same held-out tests (research note §10.1: the README's two games, the 30 BARD check games, the 5 held-out Roboflow
games, and the full pipeline); a candidate replaces the main model only when it is better somewhere and worse nowhere
that matters, after the owner has seen the numbers.

**Main model v2 went into the app** (2026-11-03, research note §10.13-10.15): the first model fine-tuned, through round 3,
on everything the rounds below used, with round 4's referee and crowd changes and the Lakers @ Warriors game's people on
the court. What the three rounds below taught is in it: round 2's model made its ball labels, it started from round 3,
and round 4's two changes are in its data. So their files were **retired from the repo**; the table below keeps their
hashes and results, and the files are in git history (the last commit holding them: `b24f936`; e.g. `git show
b24f936:models/research/student-ft-r3.onnx > student-ft-r3.onnx`). New candidates live under `runs/`, not here.

**Main model v3 replaced v2** (2026-11-05, research note §13): v2 fine-tuned with SportsMOT's officials as people,
E-BARD-detection's human labels and ball labels on 600 more BARD clips. It finds far more referees (the 2017 game's
23 % to 76 %) and boxes more of the benches and crowd (the 2017 game's 30 % to 70 %); the owner chose it seeing both.

**Research and evaluation only** (see [Sources and terms](#sources-and-terms)).

| Retired file (in `b24f936`) | What | Bytes | sha256 |
|---|---|---:|---|
| `student-ft-r4.onnx` | Round 4 (R4b): players **and referees**, the crowd left out | 47,342,538 | `331947a27495999a9cf60cf5bbdd9433503654da13c28daead9a4f442ecb455b` |
| `student-ft-r4-int8.onnx` | the same, int8 (QDQ) | 12,651,412 | `f381ad2d8374392c7b7e9c92375c90f9015dd561a91de481a83671baf539f9e4` |
| `student-ft-r3.onnx` | Round 3 (T5): players only (referees left out like the crowd) | 47,342,538 | `2c647616b14486f8e4771bf350e42abd2f6a29d1cf2da47eb78f48f1f1a38832` |
| `student-ft-r3-int8.onnx` | the same, int8 (QDQ) | 12,651,414 | `1883651e2c6362e28787718570a3d135699f66ab9b680b0f88a96eb0fbefb2a7` |
| `student-ft-r2.onnx` | Round 2 (T2), fp32 | 47,342,538 | `0025181de6bb7973317483a51bcf7adb0bd2be4e52629e611e66efb4cd4350f0` |
| `student-ft-r2-int8.onnx` | the same, int8 (QDQ) | 12,651,412 | `28a3e238177a810ae7ffea306ac26a65ecb9e597390a1a748362466595a18924` |

## Results (footage never trained on)

| Benchmark | App model | `student-ft-r2` | `student-ft-r3` | `student-ft-r4` |
|---|---:|---:|---:|---:|
| 5 NBA games of the Roboflow set (211 single frames): referees found, of 506 labelled | 187 | 84 | 26 | **424** |
| the same: players and referees, AP50 / recall | 0.505 / 0.391 | 0.687 / 0.617 | 0.768 / 0.663 | **0.881 / 0.828** |
| the same: players only, AP50 (a referee box counts against it) | 0.396 | 0.761 | **0.870** | 0.732 |
| the same: boxes on nobody labelled (crowd, benches, staff), per frame | 0.09 | 0.02 | 0.06 | 0.17 |
| the same: ball within 15 px of 1080 lines (194 frames) | 22.7 % | 34.5 % | **68.0 %** | 66.5 % |
| Ball, 30 NBA games of 2024-25 in video (90 BARD clips, 1,153 frames labelled by the RF-DETR teacher recipe): top candidate within 15 px of 1080p | 81.7 % | 89.1 % | **89.5 %** | 89.25 % |
| the same, linked path (the README's protocol) | 81.9 % | | **91.6 %** | |
| Players, 4 NCAA and FIBA broadcasts (SportsMOT `val`, 1,258 frames), AP50 / recall / precision | 0.675 / 0.626 / 0.793 | **0.918 / 0.870 / 0.939** | (trained on them) | (trained on them) |

Paired exact McNemar tests on the ball (frames only one model hits): in video `student-ft-r3` against the app model 104
to 14 (p 3.6e-18), `student-ft-r4` against `student-ft-r3` 7 to 10 (p 0.63); in the single frames `student-ft-r3`
against `student-ft-r2` 68 to 3 (p 5.1e-17), `student-ft-r4` against `student-ft-r3` 6 to 9 (p 0.61).

Which to use: `student-ft-r4` boxes the referees (84 % of them here; with the pipeline's referee check,
`nba.referee`, 80 % are boxed and named referees) and leaves nearly all of the crowd alone; `student-ft-r3` boxes the
players a little more tightly but treats referees as background. The app model boxes many spectators on video: 76 %
of the people off the court (benches, the crowd) on Cavaliers @ Warriors, one of the README's two games (research note
§10.2).

What the numbers do not say:

* `student-ft-r3` and `student-ft-r4` trained on all seven SportsMOT broadcasts, so no SportsMOT figure for them is
  held out; and on 3 games of the same Roboflow set as the 5 held-out games (other games, the same labelling), which
  favours them there.
* The single frames are fed as their own history (the frame three times). Only `student-ft-r3` and `student-ft-r4`
  trained on such inputs (a quarter of their samples), so much of their single-frame ball gain is about missing
  motion, not video. The app model's single-frame numbers are low for the same reason.
* Not measured yet: what `student-ft-r4` boxes off the court on video, the full pipeline's numbers, and the two games
  the app model was trained on (research note §10 measured `student-ft-r2` and `student-ft-r3` there).

int8 against fp32: `student-ft-r4-int8` referees 425, players-and-referees AP50 0.882, video ball 89.25 % (2 frames to
2, p 1); `student-ft-r3-int8` players AP50 0.869, video ball 89.4 %; `student-ft-r2-int8` video ball 89.2 %. On 120
held-out BARD inputs 98.0 % (r4), 98.9 % (r3) and 98.2 % (r2) of person boxes and 92.2 %, 91.2 % and 91.6 % of ball
points match fp32 (`tools/quantize_student.py`).

## How they were trained

`tools/student_finetune.py` from the exact PyTorch copy of the app's export (`courtvision/training/student_from_onnx.py`):

* players: human boxes of the SportsMOT basketball sequences, players on the court only, every third frame: the 15
  `train` sequences (2 NBA broadcasts and 1 FIBA; 3,733 samples), and for r3 and r4 also the 15 `val` sequences
  (1 NCAA and 3 FIBA; 3,936 samples);
* the ball: pseudo labels on 390 BARD clips of the 30 calibration games (90 frames each, 35,100 frames), the round-1
  fine-tune's (int8) detections linked into one path per clip (`training.ball_labels.link_ball_path`): 32,053 frames
  labelled (346 of them filled), weight at most 0.5, 10,664 samples;
* r3 and r4: human labels of 3 NBA games of the Roboflow set (`prep-coco`: 1,071 single frames, players boxed and the
  ball where labelled; for r4 the referees too, `--referees`; the crowd, benches and coaches are never labelled) and
  temporal dropout (`--static-p 0.25`: for a quarter of the samples the two earlier input frames are the current one);
  4,000 steps instead of 3,000;
* r4: SportsMOT's unlabelled people who stand among the players or whom the referee check reads as referees are left
  out of the person loss (`mark-ignore`: 6,920 boxes on 3,506 of the `train` samples, 4,277 on 2,935 of `val`), so
  the referees its player-only labels skip are not taught as background; the people in the stands still are;
* batch 4, 512 px crops, AdamW (lr 2e-4, weight decay 1e-4, 100 warm-up steps, cosine decay); stem and first stage
  frozen, batch norms frozen; hue turned by up to ±30° on frames without a ball label, scale 0.7-1.4; ball loss weight
  2; every data folder sampled equally often; bf16 autocast and channels-last; seed 0; the last step kept.

```bash
.venv/bin/python tools/student_finetune.py prep-coco data/hfcoco/train3_ann.json data/hfcoco/train3 data/ft/coco3r --referees
```

```bash
.venv/bin/python tools/student_finetune.py mark-ignore data/ft/mot_train data/ft/mot_train_ign
```

```bash
.venv/bin/python tools/student_finetune.py mark-ignore data/ft/mot_val data/ft/mot_val_ign
```

```bash
.venv/bin/python tools/student_finetune.py train --data data/ft/mot_train_ign --data data/ft/mot_val_ign \
    --data data/ft/bard_calib_r2 --data data/ft/coco3r --init runs/models/student-from-onnx.pt --steps 4000 \
    --static-p 0.25 --lr 2e-4 --freeze stem,l1 --hue 30 --zoom-min 0.7 --zoom-max 1.4 --ball-weight 2.0 --threads 4 \
    --workers 2 --bf16 --channels-last --balance --eval-every 1000 --eval-frames 6 --out runs/models/ft-r4b
```

r3: the same with `data/ft/mot_train`, `data/ft/mot_val` and `data/ft/coco3` (players only). r2: the same as r3
without `data/ft/mot_val`, `data/ft/coco3` and `--static-p`, 3,000 steps, and `--eval-mot data/sportsmot/val`
(monitoring only). The held-out games, clips and sequences:
[docs/research/cross_game_detector_splits.json](../../docs/research/cross_game_detector_splits.json).

int8: static post-training quantization (`tools/quantize_student.py --method percentile --activation u8 --n-calib 64`),
weights int8 per output channel, activations 8-bit, the five 1x1 output convolutions left in fp32.

## Using them

```bash
.venv/bin/python tools/detector_bench.py --coco data/hfcoco/heldout_ann.json --models courtvision/server/static/nba-student.onnx,models/research/student-ft-r4.onnx --sparse
```

```bash
.venv/bin/python tools/nba_analyze.py runs/nba/<game> detect --backend onnx --weights models/research/student-ft-r4.onnx
```

```bash
.venv/bin/python -m courtvision.training.student_from_onnx models/research/student-ft-r4.onnx --out runs/models/student-ft-r4.pt
```

The last command rebuilds a trainable checkpoint for `student_finetune.py train --init` (checked on `student-ft-r2`:
every tensor equals the training checkpoint's, largest difference 0.0). The Studio uses a detector placed at
`models/nba-student.onnx` in its data folder.

## Sources and terms

The weights learnt from:

* **SportsMOT** (Cui et al., "SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes", ICCV 2023;
  <https://github.com/MCG-NJU/SportsMOT>): the player boxes. Licence **CC BY-NC 4.0**: non-commercial use, with credit.
* **BARD** (Giudici et al., CVIU 2026; <https://github.com/GabrieleGiudic/BARD>): the NBA.com clips the ball labels come
  from. Dataset licence CC BY 4.0; the footage is the NBA's.
* **E-BARD-detection** (Giudici et al.; <https://huggingface.co/datasets/GabrieleGiudici/E-BARD-detection>): human boxes
  of players, referees and the ball on 30 frames of each of BARD's 60 games; main model v3 trained on the 900 frames of
  the 30 calibration games, and the 900 of the 30 check games are a held-out test. Licence CC BY 4.0; the footage is
  the NBA's.
* **basketball-player-detection-2** (r3 and r4; a Roboflow Universe dataset provided by a Roboflow user,
  <https://universe.roboflow.com/sameer-prasad-koppolu/basketball-player-detection-2-othh3>, mirrored on Hugging Face as
  `koppolusameer/basketball-coco-player-detection`): player, referee and ball boxes on NBA frames. Licence CC BY 4.0;
  the footage is the NBA's.
* the app's NBA student, which these start from: distilled from an RF-DETR teacher (Apache-2.0) on two NBA broadcasts.

Under this repo's own rules ([docs/TRAINING.md](../../docs/TRAINING.md) §1) a model trained on CC BY-NC data or on
NBA footage is for research and evaluation, never a shipped product. Credit the datasets above when you use these
weights.
