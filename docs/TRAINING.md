# Training CourtVision on games it watches

This guide covers the training tools in `courtvision/training/`:

* the data flywheel they serve;
* the legal rules that decide which data may ship;
* aligning play-by-play (PBP) with video;
* fitting and certifying calibrated models;
* exporting detector data;
* fine-tuning RF-DETR on your own footage.

Sources are the research notes: `docs/research/training_data.md` (the "training note") and
`docs/RESEARCH.md` §5, §7A.2 and §7A.11. Values marked *proposal* or *unverified* come from those notes.
They are not measurements.

| Tool | Purpose |
|---|---|
| `align_pbp.py` | Parses PBP, fits a clock timeline from scoreboard reads, places PBP actions on video time, snaps them to engine anchors, runs the acceptance gate, and writes labelled clips (JSONL) |
| `fit_shot_model.py` | CLI: shots CSV → `ShotQualityModel.fit` → model JSON plus a calibration report |
| `fit_cue_models.py` | CLI: labelled cue JSONL → `events.calibration` cue models, status rules, ECE with CI, rule-of-three gate certificates |
| `export_yolo.py` | Frames + YOLO labels + `data.yaml` + a rights-aware manifest; shipping gate; uncertainty sampling |

---

## 1. Legal rules

These rules come first, because they decide what you may train on (RESEARCH §5.1, §5.4, §7A.11, §7A.12;
training note §7, §8).

1. **Shipped models train only on rights-cleared footage.** That means games we film ourselves, games
   from partner leagues, or licensed feeds. Every row carries one value from each of these:
   * `rights_status` ∈ {`own`, `licensed`, `research_only`}
   * `label_origin` ∈ {`pbp`, `ocr`, `cv_auto`, `pseudo`, `jev`, `human`}

   Every row also carries `label_conf`.
2. **Consent to process a game is not consent to train on it.** Each game has its own
   `training_allowed` flag (RESEARCH §5.1).
3. **Sources that are R&D-only or prohibited:**
   * **NBA video and statistics: R&D only, quarantined.** NBA.com's Terms of Use bar reproducing its
     content. Section 9(vii) forbids any product featuring a database of comprehensive, regularly
     updated statistics without consent. nba_api's MIT licence covers the code, not the data. PBP
     pulled with nba_api is therefore `research_only`. `AlignConfig.rights_status` defaults to
     `research_only` for this reason.
   * **NCAA championships:** prohibited. NCAA copyright, licensed through Veritone / TNT Sports.
   * **YouTube:** never. Its terms ban scraping and downloading.
   * **Datasets:** CC BY-NC(-ND) data is evaluation and pre-training only (DeepSportradar, SportsMOT,
     PL-NBA, NBA_Streaming). SportVU states no licence.
   * **Partner platforms:**
     * Hudl's terms (§3.2.8) bar training ML on Hudl content without written permission.
     * NFHS Network is personal, non-commercial use only.
     * Veo is conditional: customer-exported files only, where Veo permits.

     The default MVP source is a stream the customer controls (OBS, iPhone or a school camera over
     RTMP/SRT).
4. **Models and code:**
   * Ship only Apache/MIT code and weights: RF-DETR N/S/M/L, SAM 2.1, SAHI, CVAT Community, Label Studio.
   * Ultralytics YOLO26 is AGPL-3.0. It is R&D only, in extra `rnd-agpl` (D9). Never ship weights
     trained with it.
   * T-DEED is GPL-3.0. RF-DETR XL/2XL is PML 1.0. SAM 3 is under the SAM License. Each needs review
     before production use.
5. **Release gate** (§7A.11, training note §8.4):
   * CI refuses to package weights unless every training row is `own`/`licensed` and its game has
     `training_allowed=True`.
   * `export_yolo(..., for_shipping=True)` enforces this. It raises `RightsError` before writing
     anything.
   * The weights carry the sha256 of their training manifest. See §6.4.
6. **Privacy** (RESEARCH §5.4):
   * No faces.
   * Appearance embeddings stay in RAM for one game.
   * No keypoints of children under 13 without parental consent.
   * Audio capture is off by default.

---

## 2. The data flywheel

From training note §5 and RESEARCH §5.2. Each phase names the tool that implements it.

| Phase | What happens | Tool |
|---|---|---|
| 0 Bootstrap (R&D only) | SportVU / Roboflow / benchmark pre-training. Freeze the gold sets: 10 annotated games, 2,000 frames over ≥ 6 venues, 500 Jev questions, 300 narration lines | — |
| 1 Ingest | Video with frame PTS and capture UTC. PBP polled every 2-5 s, keeping every revision. Scorebug OCR at 5 Hz (D14) | `perception/scoreboard_reader.py` |
| 1 Align | Clock timeline, PBP placement, snapping, gate: median \|residual\| < 0.5 s on made baskets | `align_pbp.py` |
| 2 Auto-label | Teacher stack pseudo-labels (RF-DETR-L + SAHI, SAM 2.1 / SAM 3, TrackNet-style ball). PBP event clips. Scorebug-only labels (+1/+2/+3 deltas, clock stops) | `align_pbp.make_clips`, `export_yolo` |
| 3 Mine and QA | Review queue, in priority order: PBP ↔ engine disagreements, Jev low confidence, detector uncertainty, new venues | `evaluate_against_pbp`, `export_yolo.select_for_review` |
| 4 Train and distil | Noisy-Student RF-DETR-S/M; event engine on our tracks; recalibrated shot and cue models | §5, §6 below |
| 5 Gate | Frozen gold sets (table below) | the reports of the CLIs |
| 6 Deploy and monitor | Drift alarms send segments back to Phase 3. Weekly retrain | — |

Promotion gates (training note §5 Phase 5; RESEARCH §4.4; all proposals):

| Component | Promote if |
|---|---|
| Ball detector | Recall@IoU 0.5 on live-ball frames ≥ +2 pts, no player-AP regression |
| Shot / pass models | 15-bin ECE ≤ 0.03 |
| CV cue models | 15-bin ECE ≤ 0.03 with bootstrap CI; each gate certified by the rule of three |
| Box score | PTS error = 0 on ≥ 95% of games |

Jev cannot be trained: the same weights serve every account. Claude is improved through evaluation
only. Neither is covered here; see training note §3.7 and §4.4.

---

## 3. PBP ↔ video alignment (`align_pbp.py`)

### 3.1 Inputs

* **PBP.**
  * `parse_pbp_v3(data)` takes nba_api `PlayByPlayV3` or live-feed rows: a list, `{"game": {"actions": [...]}}`
    or `resultSets`. Fields used: `period`, `clock` (`"PT11M32.00S"`), `actionNumber`, `actionType`,
    `subType`, `personId` / `playerName`, `teamTricode`, `location`, `shotResult`, `scoreHome` /
    `scoreAway`, `xLegacy` / `yLegacy` or `x` / `y`, and `assistPersonId` / `blockPersonId` /
    `stealPersonId`.
  * `parse_pbp_generic(rows)` and `load_pbp(path)` read our own CSV/JSON schema, `GENERIC_COLUMNS`.
  * The field names were last validated in 2023 (V3) and 2020 (live). Check them against a live
    response before relying on them.
  * The semantics of `xLegacy`/`yLegacy` and live `x`/`y` are unverified. They are kept raw, with
    `attrs["xy_units"]`.
  * Assists, blocks and steals embedded in a description ("(Jokić 1 AST)", "Nurkić BLOCK (1 BLK)") are
    matched with Unicode names: the feeds keep diacritics.
* **Scoreboard reads.** These are `ScoreboardReading` values from OCR at 5 Hz, or from an exact feed.

### 3.2 Usage

```python
from courtvision.training.align_pbp import (
    ClockTimeline, align_pbp, evaluate_against_pbp, load_pbp, make_clips, score_changes, write_jsonl)

pbp = load_pbp("game_0022500123.json")                 # V3 / live rows or our CSV
tl = ClockTimeline.fit(ocr_reads)                       # noisy 5 Hz reads
result = align_pbp(pbp, tl, engine_events=engine.events, score_changes=score_changes(ocr_reads))
if result.accepted:                                     # median |residual| on made FGs < 0.5 s
    write_jsonl(make_clips(result, game_id="g123", video_id="v1"), "clips.jsonl")
report = evaluate_against_pbp(engine.events, result)    # per-period FGA/FGM/3PA/3PM/FTA/FTM agreement
```

### 3.3 How it works

1. **Fitting the timeline** (`ClockTimeline.fit`):
   * Drop reads below `min_conf`, period labels that disagree with their neighbours (majority filter),
     and reads that break monotonicity. The monotonicity filter is the longest non-increasing
     subsequence per period, which removes isolated digit misreads. A misread on a period's first or
     last reads is still monotone, so a short run there whose step to its neighbour is faster than real
     time is dropped as well (`stats["dropped_rate"]`).
   * Split each period into **plateaus** and **running segments**:
     * A plateau is a value held longer than one display quantum plus slack: the clock was stopped.
     * A running segment has slope −1 and a robust median intercept. Sustained intercept shifts split
       it (short stops, video cuts); transients are dropped.
   * Plateaus meet the adjacent running lines. Read gaps, such as replays, are bridged by linear
     interpolation.
   * Display resolution is whole seconds above 1:00 and tenths below. `clock_display="none"` keeps OCR
     and PBP on the same display scale, so the truncation bias cancels.
   * For exact feeds (simulator, official tracking), use `ClockTimeline.from_exact`.
2. **Placing live-ball actions** (shots, rebounds, turnovers, steals, blocks, assists, jump balls):
   * `t = video_t(period, clock) − lag`.
   * The lag is the PBP entry delay, 1-2 s according to BasketballBench. The default is 1.5 s per
     league (unverified).
   * The lag is re-estimated per game as the median over made FGs, using an engine make or score delta
     as the anchor.
3. **Placing dead-ball actions.** Actions that share a stopped-clock plateau (fouls, FTs, subs,
   timeouts) are ordered by `actionNumber`:
   * Leading live-ball actions get the stop time.
   * The first dead-ball action gets the stop time too when it is a foul, a timeout or the period start:
     it is the whistle that stopped the clock, also after a leading miss or rebound (a foul on the
     rebounder, an and-one).
   * A trailing rebound or jump ball gets the restart (the clock starts on the touch).
   * Made FTs anchor on +1 score deltas. A delta needs 2 identical reads (D14); `ScoreboardConfirmer`
     handles this.
   * Everything else is spaced between the anchors.
4. **Snapping.** Each placed action snaps, one-to-one, to the nearest compatible engine event within
   ±2 s: release, rim crossing, control, steal, block or assist. Made shots that do not snap anchor on a
   score delta. Both the PBP time (`t_pbp`) and the final time (`t`) are stored, with the residual.
5. **Gate** (training note §5 step 7): the game is usable only if median |residual| < 0.5 s over
   ≥ 5 anchored made field goals.
6. **Clips.** `make_clips` uses the windows of training note §2.3 (tune them on the gold set):

   | Clip | Window |
   |---|---|
   | Shot | [t−3.0, t+1.5] |
   | FT | [t−1.5, t+1.5] |
   | Rebound | [miss, t+2.0] |
   | Turnover / steal / foul | [t−3.0, t+1.0] |

   Labels are `fg2_made`, `fg3_missed`, `ft_made`, `rebound_off`, `rebound_def`, `turnover`, `steal`,
   `block`, `assist`, `foul` and `jump_ball`. Each clip has `label_origin="pbp"`, `label_conf =
   align_conf` and `rights_status`.

### 3.4 Basketball semantics enforced

These follow ARCHITECTURE "Shared event semantics" and `basketball_rules_stats.md` §2a.

* One PBP shot row covers the attempt and its result.
* A fouled shooter who misses has no FGA row.
* Team rebounds after a missed non-final FT, or at the period buzzer, are feed bookkeeping. They are
  flagged `synthetic` and never labelled.
* Offensive / defensive is inferred from the preceding miss when the feed does not say.

### 3.5 Testing on the simulator

1. Build the true clock with `pbp_from_events(sim.events, ClockTimeline.from_exact(frames), lag_s=1.5)`.
   This produces a feed-like pseudo-PBP: whole seconds above 1:00, and steal / block / assist rows that
   link to their parent.
2. Align it against a timeline fitted from noisy OCR reads.

`tests/test_training_align_pbp.py` does exactly this, with and without a real `EventEngine` run. It
gets a made-FG median error of about 0.1 s against truth.

### 3.6 Without PBP

Rec, high-school and our own games often have no PBP. The scorebug alone still gives labels:

* `score_changes()` deltas: +1/+2/+3 is a made FT/2/3 for a known team.
* `ClockTimeline.plateaus()`: whistles and stoppages.

---

## 4. Calibration workflow

### 4.1 Shot model (xFG)

```bash
.venv/bin/python -m courtvision.training.fit_shot_model --csv shots.csv --league nba --out shot_model.json
# -> shot_model.json (ShotQualityModel.to_json) + shot_model.report.json
```

* **CSV columns:** `distance_ft, value, made` (required), and optionally `shot_type,
  closest_defender_ft, catch_and_shoot, zone, game_id`.
* **What is fitted:** free throws are skipped, because they use the league FT%. The fit is a residual
  L2 logistic regression on top of the NBA prior, so small samples stay at the prior.
* **Splits:** rows are split **by game, never by shot**. The report is computed on held-out games for
  both the prior and the refit model.
* **Report contents:**
  * 15-bin ECE with a bias-corrected bootstrap 95% CI that resamples games (see §4.2 on the bias);
  * MCE, Brier and log-loss;
  * the reliability table, with bin counts.

  The target is ECE ≤ 0.03.
* **Non-NBA leagues:** xeFG cards stay NBA-only until we hold ≥ 2,000 labelled shots per league tier
  (§7A.2).

### 4.2 Cue models (`events/calibration.py`)

```bash
.venv/bin/python -m courtvision.training.fit_cue_models --jsonl make_rows.jsonl --out cue_models.json
# rows: {"kind","features","label","game_id","calibration_source","event_id","t"}
```

* **Status** (RESEARCH §7A.2), counted on the training games only:

  | Status | Requires | Meaning |
  |---|---|---|
  | `uncalibrated` | Fewer than the `platt` minimum | Hedge-only; gate never met |
  | `platt` | ≥ 300 labelled events, ≥ 50 of them in (0.05, 0.95) | Calibrated |
  | `isotonic` | ≥ 1,000 labelled events | Calibrated |

* **Gate certification by the rule of three.**
  * A gate at threshold τ with error budget ε is certified only if the held-out events with p ≥ τ show
    **0 errors in n ≥ 3/ε** cases. The 95% upper bound on the error rate is then 3/n.
  * Made FG (τ = 0.95, ε = 0.02, i.e. the 0.98 precision target) needs ≥ 150 accepted makes without
    an error. ε = 0.01 needs ≥ 300.
  * An `uncalibrated` model never certifies.
  * The make model also gets a negative-outcome certificate (`gate_low`): P(miss) = 1 − p ≥ 0.90 with
    ε = 0.10 (30 accepted misses without a make), because narration states "missed" at that gate
    (`FactsConfig.gates["shot_missed"]`). ε here is a proposal.
  * With errors, the report gives the Clopper-Pearson bound.
* **ECE has a finite-sample upward bias** on a few hundred events. Expect the 0.03 target to need
  ~1,000 held-out events. Resampling adds more of the same bias, so the plain percentile interval sits
  above the point estimate and, in our simulation, never covered the population ECE of a calibrated
  model at n = 300-1,000. The report's `ece_ci95` is therefore the bias-corrected ("basic") bootstrap
  interval [2·ECE − q97.5, 2·ECE − q2.5], clipped at 0. `ece_ci95_percentile`, `ece_bootstrap_bias`
  and `ece_debiased` (ECE − bias) are reported next to it. Near 0 no interval covers well; read the
  bias and the sample size together.
* **Output.** The model file loads with `CueModelSet.load` or `EventEngineConfig.cue_model_path`. Each
  fitted model records its `calibration_source`. The full reports sit under `"calibration"`.
* **Simulator-derived labels are for development only.**
  * `build_cue_dataset(engine_events, sim.events, game_id)` matches engine events to simulator truth.
  * It marks every row `calibration_source: "sim"`.
  * Such models are never `valid_for_live`.
  * The CLI writes them only with `--allow-sim`.
  * The mark travels with the weights. A model carried over from `--base` (a kind without rows, or too
    few rows to refit) keeps the base file's `calibration_source`, and `for_live` is false while any
    calibrated model in the file came from simulator rows. Only a refit on real rows clears it.
  * The engine reports their `p_status` as `platt`/`isotonic`, so **never deploy a sim-calibrated
    file**. Use `EventEngineConfig.assume_calibrated` for simulator evaluations instead.

---

## 5. Detector data (`export_yolo.py`)

```python
from courtvision.training.export_yolo import ExportItem, export_yolo, select_for_review

items = [ExportItem(image=frame_bgr, detections=dets, game_id="g12", video_id="cam1", frame_idx=i,
                    rights_status="own", label_origin="human", training_allowed=True) for ...]
queue = select_for_review(items, k=300)          # ~300 frames per new venue (training note §2.5)
export_yolo(items, "datasets/gym6", for_shipping=True)   # RightsError if any row cannot ship
```

* **Layout:** the Roboflow YOLO layout, `train/images`, `train/labels`, `valid/images`, `valid/labels`,
  plus `data.yaml` with names [player, referee, ball, rim]. rfdetr 1.11.2 detects it as a YOLO dataset;
  this is tested with `rfdetr.datasets.yolo.is_valid_yolo_dataset`. Ultralytics reads the same
  `data.yaml`.
* **Splits** are by game.
* **Pseudo-labels** are filtered per class by score; human labels keep every box. Boxes with a
  non-finite coordinate (a projection behind the camera) are never written.
* **Manifest:**
  * `manifest.jsonl` has one row per image: rights, origin, `label_conf`, game, frame and an
    uncertainty score.
  * `export_meta.json` holds its sha256.
* **Uncertainty sampling** ranks frames by detector scores near 0.5, plus a missing ball in live play,
  with a minimum frame gap per video.
* **`detections_from_frame_state`** turns `FrameState`s (sim truth, tracker output) into pseudo-label
  boxes.
* **First-party data sprint targets** (§7A.11; planning targets, not findings):
  * ≥ 6 gyms and ≥ 20 games with `training_allowed=True`;
  * ≈ 6,000 labelled frames, of which ≥ 3,000 are in-flight ball boxes;
  * ≥ 5,000 number crops;
  * ≥ 1,000 labelled shots.

---

## 6. Fine-tuning RF-DETR on your own footage (rfdetr 1.11.2)

The training note documents RF-DETR's `model.train(...)` API [50]. The details below were checked by
introspecting the installed `rfdetr==1.11.2` on 2026-10-20. Where 1.11.2 differs from the older docs,
the difference is noted.

### 6.1 Install

Training needs the training extras. The library's own error message says:
`pip install "rfdetr[train,loggers]"`. PyTorch Lightning is not in the repo `.venv`, so install the
extras in a separate training environment. Never add them to core.

### 6.2 Recipe

```python
from rfdetr import RFDETRSmall            # or RFDETRMedium; N/S/M/L are Apache-2.0

model = RFDETRSmall(gradient_checkpointing=True)   # a ModelConfig field in 1.11.2 (not a train() kwarg)
model.train(
    dataset_dir="datasets/gym6",          # export_yolo output (YOLO layout auto-detected)
    output_dir="runs/rfdetr_s_gym6",
    epochs=80, early_stopping=True,       # 50-100 epochs with early stopping (training note §3.1, proposal)
    batch_size=2, grad_accum_steps=8,     # effective batch 16 (T4: 4 x 4; A100: 16 x 1) [50]
    lr=1e-4, lr_encoder=1.5e-4,           # the TrainConfig defaults in 1.11.2
    resolution=768,                       # must be divisible by patch_size * num_windows
    device="mps",                         # mapped to Lightning accelerator="mps" (see §6.3)
    notes={"manifest_sha256": "<export_meta.json manifest_sha256>"},
)
```

API notes:

* **Resolution.** In 1.11.2, N/S/M/L use `patch_size=16` and `num_windows=2`, so `resolution` must be a
  multiple of **32**. The older training docs say "divisible by 56"; that applied to the earlier
  patch-14 backbones. Native resolutions are 384 (N), 512 (S), 576 (M) and 704 (L). Higher values help
  the small ball. The training note proposes 728-1008, which in 1.11.2 means multiples of 32, e.g. 768
  or 1024.
* **Unknown keyword arguments are rejected.** `TrainConfig` has `extra="forbid"`. `gradient_checkpointing`
  goes to the model constructor, not to `train()`.
* **Outputs** [50]: `checkpoint.pth`, `checkpoint_<n>.pth`, `checkpoint_best_ema.pth`,
  `checkpoint_best_regular.pth`, `checkpoint_best_total.pth`. Reload with
  `RFDETRSmall.from_checkpoint("…/checkpoint_best_total.pth")`. Select checkpoints on the validation mAP
  (`best_model_metric="map"`, the default).
* **Teacher-student** (Noisy Student, training note §2.4): pseudo-label with a large offline teacher,
  e.g. RF-DETR-L with SAHI tiles (`AutoDetectionModel.from_pretrained(model_type="roboflow", ...)`). Then
  train the S/M student on human labels plus the filtered pseudo-labels:
  * boxes persisting ≥ 3 frames;
  * a ballistic fit for the ball;
  * roster caps;
  * PBP agreement.

### 6.3 MPS caveats (unverified; RESEARCH §7A.11, §7A.13 item 10)

* `TrainConfig` has no `device` field. It uses `accelerator="auto"`. `train(device="mps")` is absorbed
  and mapped to `accelerator="mps"`; this was read from the 1.11.2 source and has not been run.
* `amp_dtype="fp16"` mentions "supported CUDA/MPS backends". The training docs list only `cpu` and
  `cuda`.
* No document confirms that RF-DETR trains on MPS. **Before relying on it, run a 1-hour smoke test:**
  the loss should fall and there should be no storm of CPU-fallback warnings. Then measure epoch time.
* M3 Pro proposal: S/M, `batch_size=2`, `grad_accum_steps=8`, `gradient_checkpointing=True`. Memory
  scales with `resolution`².
* **Fallback:** a rented CUDA GPU. Lambda on-demand prices: 1× H100 PCIe $3.29/h, A10 $1.29/h. A 24 h
  H100 run is ≈ $79 (arithmetic, not a measured run). Labelling time, not GPU time, dominates the cost
  of the first commercial model.
* Ultralytics YOLO26 documents `model.train(data="data.yaml", epochs=..., imgsz=..., device="mps")`.
  MPS forces NCHW. It can train on the same export for **R&D comparison only** (AGPL, D9).

### 6.4 Provenance and export

* **Pass the manifest hash with `notes=`.** rfdetr stores it in the best-model `.pth` and the Lightning
  `.ckpt` checkpoints, under `checkpoint["args"]["notes"]`. Passing the same `notes=` to `export()`
  also stores it in ONNX exports, as the `rfdetr_notes` metadata property.
* **CoreML exports ignore `notes`.** `model.export(format="coreml", ...)` is the production path
  (RESEARCH §7A.10). Write the manifest sha256 into a sidecar JSON next to the `.mlpackage`.
* The release gate checks that the manifest sha256 matches before packaging.
* Then run the PyTorch-vs-CoreML parity test (cv_pipeline §13, RESEARCH §7A.10).

---

## 7. The learning loop: `courtvision ingest` → `train` → `promote`

The tools above are the parts. `courtvision/learning/` wires them into a loop that gets better with every
game of footage ingested, and only ever ships an improvement that was measured on games it never trained on.

```bash
# 1. ingest each game: video + PBP -> a game dataset (perception + engine run once, tracks cached)
courtvision ingest --video game.mp4 --pbp game_pbp.json --calibration game_cam.json \
    --broadcast --scorebug-layout bug.json --rights research_only --out datasets/train
courtvision ingest --video holdout.mp4 --pbp holdout_pbp.json --calibration cam.json --gold --out datasets/gold
# 2. train a candidate bundle from the accepted, non-gold games
courtvision train datasets/train --base models/current --out models/candidates/c1
# 3. promote it only if it beats the current bundle on the frozen gold games
courtvision promote models/candidates/c1 --gold datasets/gold --models models
# 4. run with what was promoted
courtvision serve --set pipeline.model_bundle=models/current
```

**Ingest** (`learning/ingest.py`). Each game runs once through the normal offline pipeline. Every
`FrameState` the event engine consumed is cached in `tracks.jsonl.gz`, so training re-runs the engine
without the video or perception. Scoreboard reads (venue board or broadcast scorebug, 5 Hz) fit the clock
timeline. The PBP is aligned (§3) and the game gated. A rejected game is still written
(`manifest.accepted = false`) and never trains. The dataset holds:

* label events at their aligned times (snapped or anchored where possible; otherwise clock-mapped and
  shifted by the game's median clock-vs-anchor residual);
* clips;
* cue rows (`calibration_source` "pbp", or "sim" for simulated games);
* shot rows (engine shot location and nearest defender from the tracks, PBP make / miss);
* the roster;
* a manifest with rights, the gold flag, the alignment summary and the sha256 of every file.

**Train** (`learning/train.py`):

* **Engine settings.** A coordinate search over `SEARCH_SPACE` (the monocular / broadcast sensitivities in
  `events/engine.py`). Each candidate re-runs the engine on every training game's cached tracks. It is scored
  against the PBP labels by F1 per PBP-labelled kind (shots, makes, misses, rebounds, assists, steals,
  turnovers, blocks: a PBP has no passes), with counts pooled over games and events matched within 2 s
  without actors (PBP person ids are not roster ids). A value is kept only if it raises the pooled macro F1 by
  `min_gain`. Games run in parallel processes.
* **Cue and shot models.** The cue models are refit from the cue rows (§4.2) and the shot model from the
  shot rows (§4.1).
* **Glyph bank.** A scorebug glyph bank, if given, ships in the bundle.

The bundle (`learning/bundle.py`) records its training games, its parent bundle and its rights:
"shippable" only when every game is `own` / `licensed` with `training_allowed`.

**Promote** (`learning/promote.py`). Gold games are ingested with `--gold`. They never train, and they must
be unchanged since ingest (sha256). The candidate and the current bundle (or the defaults) both re-run on
every gold game. The candidate replaces `models/current` only if it:

* beats the current macro F1 by `--min-gain` (default 0.005);
* drops no kind with at least 5 gold labels by more than 0.05;
* for `--channel ship`, has shippable rights.

The previous bundle moves to `models/history/<id>`. Every decision, promoted or not, is appended to
`models/promotions.jsonl`.

**On the simulator.** `courtvision ingest --sim-seed N --mode mono|video|broadcast` ingests a simulated game.
Its PBP is a pseudo play-by-play built from the simulator's truth events: entered with a 1.5 s scorer lag, with
clocks on the board's display scale. That exercises the loop end to end without any NBA data
(`tests/test_learning_loop.py`). Simulated games are `own`, but their cue calibrations stay development-only
(§4.2).

**What does not improve yet.** The detector, the jersey reader and the court-keypoint stage have no training
step in the loop. §5 / §6 export the detector data and the RF-DETR recipe, but fine-tuning is a manual GPU
step outside `train`. A real 48-minute game passes the alignment gate easily: about 80 made field goals
against the 5 needed. Simulated games of a few minutes often do not, because with fewer than 5 anchored makes
the scorer's lag is not estimated.

### 7.1 Broadcast footage (`perception/broadcast.py`)

TV footage breaks two assumptions of the venue pipeline: the camera moves, and not every frame shows the game.
`--broadcast` (`perception.broadcast`) wraps the perception pipeline with:

* **Court tracking.** `court_tracker.py` handles the game camera's pan / tilt / zoom about its fixed centre.
  The centre comes from the one operator calibration. Each frame, court-line samples are searched along their
  normals in a white top-hat line response, then pan, tilt and focal length are fitted by robust least squares
  with a constant-velocity prediction and per-frame motion gates. Lost frames (cuts) re-acquire by a
  pan / zoom grid search about the known centre. On simulator broadcast renders (seed 3, 50 s) the court mapping
  error was 0.10 ft at the 95th percentile, and replay and close-up frames stayed "lost" rather than being
  mistaken for the game camera.
* **Cut / replay gating.** Histogram shot boundaries reset image-space tracks. A frame is live only while the
  court tracker holds the game camera and the scorebug is shown (replays hide it). Non-live frames give the
  engine no players and no ball.
* **Camera-motion compensation.** Track and ball filters move by the camera's frame-to-frame rotation
  homography.
* **Engine camera.** The known camera centre replaces the engine's fixed-camera ray estimate
  (`FrameState.camera_xyz`).
* **Scorebug OCR.** `scorebug.py` takes a per-broadcast `ScorebugLayout` of fields in image pixels. Glyphs are
  segmented (":" kept, touching digits cut) and read by normalised correlation against a `GlyphBank`. The bank
  is learned from crops with known text, or bootstrapped from a font, which is exact on simulator renders.

`source.sim_mode=broadcast` renders a simulated broadcast to test all of this (`sim/broadcast.py`). The game
camera pans and zooms with the play, replays and cutaways fall inside dead-ball windows, and the scorebug
shows the operator-run board.

A real broadcast still needs four things that are not in this repository:

* one calibration of the game camera;
* the scorebug layout;
* the scorebug's glyphs, learned from a few labelled frames;
* a person detector that works on broadcast footage (RF-DETR, §6).

---

## 8. Known gaps

* **The entry lag** of 1.5 s, and whether PBP clocks are truncated or rounded, are unverified per league.
  Estimate the lag per game, and measure it on the gold set.
* **Amateur scoreboard-operator lag** has no source (§7A.5). `AlignConfig.score_update_lag_s` defaults
  to 0.
* **Whether live-feed `timeActual` is the action time** or the data-entry time is unverified. Live-mode
  alignment with Δ = capture UTC − `timeActual` (training note §2.2 step 5) is not implemented. The
  clock timeline is used instead.
* **Cold-start cue weights, clip windows, uncertainty weights and pseudo-label thresholds** are
  placeholders until they are tuned on the gold sets.
