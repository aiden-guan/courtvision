# CourtVision: training "off of live games" (datasets, weak supervision, labeling, evaluation)

Research note, 2026-10-20. Scope: how CourtVision (CV tracking, then the event engine, then TypeSafe Jev typed decisions, then Claude `claude-opus-5-5` narration) gets training and evaluation data, mostly from games it watches, on Python 3.12 / Apple M3 Pro (18 GB, MPS, no CUDA).
Citations are `[n]` and map to the **Sources** list at the end. "(unverified)" marks claims I could not confirm from a primary source. "(proposal)" marks CourtVision design choices of mine, which have no source behind them.

The web-search budget ran out near the end of this research. A few items below are therefore marked "(unverified)" instead of being chased further.

---

## 0. TL;DR: decisions this note supports

1. **Licensing decides what can ship.** Nearly every public basketball dataset is non-commercial:
   - DeepSportradar data is CC BY-NC-ND 4.0 [3][7].
   - SportsMOT, NBA_Streaming annotations and PL-NBA are CC BY-NC 4.0 [27][24][21].
   - NBA.com's Terms of Use bar reproducing "Basketball Content" without written permission. They also limit NBA statistics to "legitimate news reporting or private, non-commercial purposes", and clause (vii) of Section 9 forbids using them in any product or service that features a database of comprehensive, regularly updated NBA/WNBA/G League statistics without the Operator's express prior consent (ToU last updated 2026-07-13) [42]. A live box-score product is squarely inside that clause.

   **Decision (proposal):** public and NBA-derived data is used only for R&D: pre-training, benchmarking and ablations. Any model that ships is fine-tuned on games CourtVision has rights to (own camera feeds, partner leagues, licensed feeds). Every record carries a `rights_status` column.
2. **The best free label source is the game's own bookkeeping.** This means official play-by-play (PBP) plus the on-screen scorebug, aligned to video. Three 2025-2026 datasets were built this way, from NBA PBP plus event-centred clips: BasketEvent (35k clips, 51k player-event samples; broadcast clips "from the Internet", source not named, located by game ID + game clock) [20], the rebound-anticipation set (100k clips scraped from the NBA Stats website) [19] and BARD (14,676 clips) [25]. NBA_Streaming (152 games, 307.5 h) and BasketballBench aligned PBP by OCR on the game clock [23][22]. CourtVision should industrialise the same pipeline.
3. **Two label sources stay available when no PBP exists** (rec, high-school or own games), both read from the scorebug (proposal):
   - A score delta of +1/+2/+3 at a video time is a made FT/2PT/3PT for a known team.
   - Game-clock stop/start and shot-clock resets mark whistles and possession changes.

   These are near-zero-cost, high-precision weak labels.
4. **Detector choice is driven by the license.**
   - RF-DETR N/S/M/L and all RF-DETR-Seg sizes are Apache-2.0. XL/2XL and the tiny A/F/P sizes are PML 1.0 [49].
   - Ultralytics YOLO26 is AGPL-3.0 or Enterprise [51].

   Use RF-DETR-S/M for production detection and keypoints. YOLO26 is fine for internal experiments. YOLO27 was announced 2026-09-13 and is "coming soon", not released [53].
5. **Train the event engine mainly on tracking trajectories, not pixels.** It is cheaper on MPS and is where public data is richest (SportVU, 631 games [15]). Add a small video-clip model only for what trajectories cannot see: make/miss confirmation, fouls, and who touched the ball last in a scrum. Two reference points: the video-based PlayNet reaches F1 0.682 on BasketEvent [20], and a trajectory transformer (CourtMotion) reaches AP 0.91 for shot-taker at 0.8 s before the shot [48].
6. **Jev cannot be trained.** "Jev is not fine-tuned or LoRA-adapted with customer data", and "the same weights serve every account"; proprietary content goes in `state`, domain rules and boundary cases in each question's `instructions` and `criteria` [74]. So "training Jev off live games" means three things:
   - building gold sets per question;
   - measuring reliability and ECE per question and pinned version;
   - adding an optional isotonic recalibration layer and confidence thresholds on top.

   Pin a versioned model ID and log the `model` field of every response [74]. As of 2026-10-20 the docs map `jev-latest` → `jev-1.13.0` and `jev-preview` → `jev-1.14.0` (Jev 1.14 "available from October 14, 2026"); aliases move when a release ships, so answers can change with no code change [74]. API specifics are in `docs/research/jev_api.md`.
7. **Claude narration is "trained" through evaluation.** The flywheel produces a frozen game-state log for each game. Every narrated claim is checked against that log with an atomic-claim precision metric (FActScore-style [71]) and NBA_Streaming-style Identity, Event and Action accuracy [23]. Regressions block release.

---

## 1. Public datasets: exact names, sizes, licenses, URLs, and how we use them

### 1.1 Inventory table

| Dataset | Task / content | Size | License | Where | CourtVision use |
|---|---|---|---|---|---|
| **DeepSportradar-v1** (MMSports @ ACM MM 2022) | 4 basketball tasks: ball 3D localization, camera calibration, instance segmentation, player re-ID [1] | see rows below | CC BY-NC-ND 4.0 [3][7] (the CC BY-NC-ND link on [1] is the arXiv paper's licence, not the data's) | github.com/DeepSportradar [2]; Kaggle `deepsportradar/basketball-instants-dataset` [3] | Calibration, segmentation and re-ID pre-training and benchmarks (non-commercial) |
| ↳ Instance segmentation challenge v2 (2023) | Humans (players, coaches, referees), 1 class (id 0), COCO JSON, occlusion-focused [3] | train 223 / val 37 / test 64 / challenge 84 (unlabelled), per the `tools/gen_annotations.py` output; the README prose says trainval 246 = train 211 + val 35, so the repo is internally inconsistent (trust the script output) [3] | Code Apache-2.0, data CC BY-NC-ND 4.0 [3] | [3] | Mask quality benchmark. Ranking metric "Occlusion Metric" = Occluded Instance Recall × Disconnected Pixel Recall; baseline Mask R-CNN R50-FPN (MMDet v3) [3] |
| ↳ Camera calibration challenge (2022) | Image to K, R, T (calib3d `Calib`), FIBA courts, Keemotion cameras [4] | 728 image/calibration pairs; default split 480/164/84; challenge images 960×540 [4] | (not stated on repo page; the underlying DeepSport data is CC BY-NC-ND 4.0 per [7]) | [4] | Benchmark for our court homography. Metric is MSE (cm) of 6 projected points. Baseline: DeepLabv3 segments 20 court lines, then `cv2.calibrateCamera` [4] |
| ↳ Player re-ID challenge (2023, 2nd ed.) | Crops of players, referees and coaches from 20-frame sequences, one camera view [5] | train 436 IDs / 8,569 imgs; test 50 query / 910 gallery; challenge 618 query / 9,505 gallery [5] | Code Apache-2.0 (data license not stated on page) [5] | [5] | Re-ID embedding benchmark. Baseline ~72.7 mAP (test); the 2022 winner, CLIP ViT-L/14, reached 98.44% mAP on the 2022 challenge set [5] |
| ↳ Ball 3D localization (2022) | Ball-diameter-in-pixels regression from an oracle 2D position; with calibration this gives 3D [6] | ball-centred thumbnails; "ballistic" set 233 samples; DeepSport extended set 364 samples [7] | data CC BY-NC-ND 4.0 per [7] | [6][7] | Ball height (z) estimation for shot arcs. Metric MADE (px). Test arenas KS-FR-CAEN / LIMOGES / ROANNE [6] |
| **SpaceJam** | Single-player action recognition: 16-frame RGB clips (.mp4) + 2D joints (.npy) [9] | ~32,560 examples [9]; 10 classes: Block, Pass, Run, Dribble, Shoot, Ball in Hand, Defence, Pick, No Action, Walk [10] | MIT (repo About) [9]; data license unclear (unverified) | github.com/simonefrancia/SpaceJam [9] | Pose-based per-player action head pre-training. Reference: R(2+1)D (K400-pretrained) reaches ~85% test accuracy after 25 epochs, lr 1e-4, batch 8 [10] |
| **NBA SportVU 2015-16 logs** | Player 2D (x, y) + ball (x, y, z) at 25 Hz, game/shot clock, `events` → `moments` [11][15][45] | 631 games [15][48]; files named `MM.DD.YYYY.AWAY.at.HOME.7z`, e.g. `01.01.2016.CHA.at.TOR.7z` (linouk23 folder listing linked under [13]) | No license stated [11][14]; MIT on the sealneaward mirror's code [13] | github.com/linouk23/NBA-Player-Movements [11]; HF `dcayton/nba_tracking_data_15_16` [14] | **Main pre-training corpus** for the event engine, shot model, pass-target model and EPV. Also gives "perfect-tracking" upper bounds |
| **NCAA Basketball** (Ramanathan et al., CVPR 2016) | 11 event classes = 5 shot types × {success, fail} + steal [16] | 257 games (~1.5 h each), ~14K event annotations; 212/12/33 train/val/test videos; 4 s clips at 6 fps → 11,436 / 856 / 2,256 clips; player boxes on 9,000 frames [16] | not stated (unverified) | No release URL in the paper: it only says "We plan to release our annotated data"; the one link is the source YouTube channel (youtube.com/user/ncaaondemand) [16]. Any dataset URL is (unverified) | Shot-type spotting benchmark. **Correction to the brief:** the paper has 11 classes and 14K annotations, not "14 action classes" [16] |
| **Basketball-51** | 8 classes: 2pt/3pt/FT/mid-range × {make, miss} [17] | 10,311 clips, ~6 s, 320×240, from 51 NBA broadcasts; best 3D-ConvNet accuracy 79% [17] | (unverified) | Kaggle `sarbagyashakya/basketball-51-dataset` [18] | Make/miss clip classifier warm start (low resolution) |
| **Rebound anticipation** (arXiv 2512.15386, Dec 2025) [19] | OREB vs DREB anticipation / classification / spotting | >100k clips (~300 h, 10 seasons, 1280×720 @ 60 fps), 1,916 valid frame-level rebound labels; splits D2K 1,500/250/250 and D25K 23k/1k/1k; ~73k unlabelled [19] | **Not released**: "upon NBA permission approval" [19] | code `github.com/arnalytics/labeled_plays_NBA` (returned HTTP 404 on 2026-10-20) | Protocol and baselines for our rebound-team Jev question: classification acc 0.881; spotting mAP 0.46 averaged over δ ∈ {1, 2, 3} s (val); anticipation acc 0.60 at 0.5 s vs human 0.71 [19] |
| **BasketEvent** (arXiv 2607.21267, Jul 2026) [20] | Player-centric "who did what when". 10 events (Missed Shot, Made Shot, Free Throw, Foul, Turnover, Rebound, Assist, Block, Steal, Jump Ball) + Background [20] | 226 games, 34,931 videos (90.6 h), 51,663 player-level samples; train 189 games, test 37; 1,000-sample temporal subset (100/class) [20] | "will be made publicly available"; no URL or license yet [20] | arXiv [20] | **Blueprint for our PBP→clip pipeline.** PlayNet F1 0.682 vs SlowFast 0.456 [20] |
| **PL-NBA** (arXiv 2608.19646, Aug 2026) [21] | Possession-level: event recognition, captioning, temporal action localization (TAL), anticipation | 60 games (2022–25), 11,000 possessions (avg 12.11 s), 31,567 sub-events, 13 sub-event types / 22 type+outcome labels, 1080p + 48 kHz audio [21]. The GitHub README instead says "40 official NBA games", "7,000+" possessions and "30,000+" events, so the public release may be a subset; count what you download | CC BY-NC 4.0, academic only [21] | github.com/holhouse/PL-NBA-Dataset [21] | Possession segmentation and outcome labels (e.g., PassBall.success 7,037; 3ptShot made 1,005 / missed 1,641) [21] |
| **NBA_Streaming** (arXiv 2608.09200, Aug 2026) [23] | Online commentary on continuous streams | 152 full games (307.5 h), ~35K aligned events, 121/15/16 game split [23] | Annotations CC BY-NC 4.0; videos only by request with a non-redistribution agreement [24] | github.com/wyy-081/NBA_Streaming [24] | **Narration eval protocol** (§5.4) and the OCR-on-game-clock alignment precedent |
| **BasketballBench** (arXiv 2608.23435, Aug 2026) [22] | 10 QA tasks incl. scorebug reading, temporal localization, jersey recognition | 7,980 questions, 2,501 possession clips (BasketEvent test split) [22] | paper CC BY 4.0; data license not stated [22] | arXiv [22] | Scorebug-OCR QC recipe; documents a 1–2 s PBP lag [22] |
| **BARD** (CVIU 2026) [25] | Multi-label clip actions. JSON fields `player, action, result, assisted, other_player, color` [25] | 60 games (2024–25), 14,676 clips (720p) from 24,692 raw [25] | CC BY 4.0 (project) [25]; video is NBA content (see [42]) | github.com/GabrieleGiudic/BARD, HF `GabrieleGiudici/BARD` [25] | Label-schema reference (who / what / result / assist) |
| **BASKET** (CVPR 2025; venue not stated on the arXiv page, unverified) [26] | Skill estimation, 20 skills × 5 levels | 4,477 h, 32,232 players [26] | paper CC BY 4.0; data license: check repo [26] | github.com/yulupan00/BASKET [26] | Not core. Possibly coach-facing skill profiling later |
| **SportsMOT** (ICCV 2023) | MOT in basketball / football / volleyball [29] | 240 videos, >150K frames, >1.6M boxes; split 45/45/150; 720p @ 25 fps; MOT17 format; 36.5 GB [27][28][29] | **CC BY-NC 4.0**, no redistribution without permission [27][28] | github.com/MCG-NJU/SportsMOT; HF `MCG-NJU/SportsMOT` [27][28] | Tracker benchmark (basketball subset via `splits_txt/basketball.txt`) [27] |
| **TeamTrack** (CVPRW 2024) | Full-court MOT; basketball fisheye + drone side + drone top [30] | Basketball 65.5 min: fisheye 6K 17.5 min / 346,500 boxes; drone side and top 4K 24 min / 475,200 boxes each [30] | (unverified; website is CC BY-SA 4.0) [30] | Kaggle `atomscott/teamtrack` [30] | Wide-angle / fixed-camera tracker tests. Baseline basketball side-view HOTA: BoT-SORT 75.2, ByteTrack 76.2 (Table 5, "SideView"); on the harder "SideView2" subset (Table 6) they score 47.3 / 42.9 [30] |
| **BasketHAR** (arXiv 2604.17065, 2026) | Multimodal HAR in training scenarios [81] | (unverified) | (unverified) | arXiv [81] | Watch-list only |

### 1.2 SportVU 2015-16 format (what our loader must handle)

- **Moment array** (from linouk23's parser) [12]:
  - `moment[0]` = quarter, `moment[2]` = game clock (s), `moment[3]` = shot clock.
  - `moment[5][0]` = ball, `moment[5][1:]` = 10 players. Indices 1 and 4 are unused by that parser.
- **HF re-packaging** [14]:
  - Top level: `gameid, gamedate, event_info{eventid,type,possession_team_id,desc_home,desc_away}, primary_info, secondary_info, visitor, home, moments`.
  - Moments: `quarter, game_clock (720→0), shot_clock (24→0), ball_coordinates (x,y,z), player_coordinates (teamid, playerid, x, y, z)`.
- **Caveats** [14]:
  - Some moments lack coordinates.
  - Event windows overlap, so the same frames recur across events. **Deduplicate by (quarter, game_clock)** before training.
- **Court frame:** the visualiser uses a 0–100 × 0–50 extent [11], i.e. feet on a 94 × 50 ft court plus margin (inference). The EPV paper confirms 25 Hz and 2D players with a 3D ball [45].
- **Shot-time heuristic** from the sealneaward mirror: shot time = the highest-acceleration point before the ball's peak [13]. This is a useful weak label for "release frame".

### 1.3 Roboflow Universe basketball datasets (class lists)

- **`roboflow-jvuqo/basketball-player-detection-3-ycjdo`** [31]
  - 10 classes [31]: `ball, ball-in-basket, number, player, player-in-possession, player-jump-shot, player-layup-dunk, player-shot-block, referee, rim`.
  - Image count: Roboflow's YOLO26 tutorial on this dataset (`workspace.project("basketball-player-detection-3-ycjdo")`, **version 10**) says "654 annotated high-resolution images" and trains `yolo26m.pt` for 20 epochs at imgsz 640 [34]. Whether version 10 carries all 10 classes is (unverified); pin the version number in the loader.
  - **Design note:** action-as-class labels (`player-jump-shot`, `ball-in-basket`) give cheap per-frame event cues. Keep them as auxiliary heads, not as the event engine.
- **`roboflow-jvuqo/basketball-court-detection-2`** [32][33]
  - 1 class (court), 33 keypoints, COCO-Keypoint export. The tutorial uses version 19.
  - Annotation rule: mark a keypoint `visible` when seen and `occluded` when hidden but geometrically known [32].
- **`roboflow-jvuqo/basketball-jersey-numbers-ocr`** [33][31]
  - ~3.6k jersey-number images ("The dataset contained 3.6k images") from the 2025 NBA Playoffs. They were auto-labelled with SmolVLM2, then manually refined, and are imbalanced ("8": 315 crops, "6": 8 crops) [31].
  - Results [31]:
    - SmolVLM2 zero-shot: 56%.
    - Fine-tuned SmolVLM2: 86%.
    - **ResNet-32 classifier: 93%.**
- **Other public sets** (class lists inferred from sample code; verify on the page):
  - `x-qfelz/x-basket-2`: ball, player, board, rim; CC BY 4.0 [35].
  - `machinelearning-6xoxa/basketball-jaawc`: Ball, Player, Hoop; CC BY 4.0 [36].
  - `dataset-baketball/baskball` v5: 2,583 images; player, basketball, hoop; CC BY 4.0 [37].
- `roboflow/sports` (MIT) collects these links [33].
- Universe licenses are set by the uploader. Re-check before commercial use.

---

## 2. Weak supervision from live games

### 2.1 Official PBP: exact fields

**`nba_api` stats endpoint `PlayByPlayV3`** [38]
- URL: `https://stats.nba.com/stats/playbyplayv3`.
- Parameters (all required, none nullable): `GameID` (regex `^\d{10}$`; Python `game_id`), `StartPeriod` (`start_period`), `EndPeriod` (`end_period`).
- The doc page says it was last validated 2023-09-14, so re-check field names against a live response before hard-coding them.
- Data sets:
  - `AvailableVideo` (`videoAvailable`).
  - `PlayByPlay`: `gameId, actionNumber, clock, period, teamId, teamTricode, personId, playerName, playerNameI, xLegacy, yLegacy, shotDistance, shotResult, isFieldGoal, scoreHome, scoreAway, pointsTotal, location, description, actionType, subType, videoAvailable, actionId`.

**`nba_api` live endpoint** [39]
- URL: `https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json` (`game_id` is 10 digits).
- The doc page says "Last validated 2020-08-16"; field names may have drifted, so validate against a live 2025-26 response before hard-coding.
- `actions[]` adds these fields, which matter for live alignment and labels:
  - `timeActual` (UTC, e.g. `2021-01-16T02:14:17.3Z`)
  - `edited` (UTC)
  - `orderNumber`
  - `clock` (ISO-8601 duration, e.g. `PT03M59.00S`)
  - `periodType`
  - `possession` (team id)
  - `qualifiers` (e.g. `["pointsinthepaint"]`)
  - `descriptor` (e.g. `pullup`)
  - `shotActionNumber`
  - `assistPersonId`, `blockPersonId`, `stealPersonId`, `foulDrawnPersonId`
  - `jumpBallWonPersonId`, `jumpBallLostPersonId`, `jumpBallRecoverdPersonId` (typo as spelled in the docs; the matching name field is `jumpBallRecoveredName`)
  - `x`, `y` (court-percent style floats, e.g. 90.2 / 45.9; semantics unverified)
  - `xLegacy`, `yLegacy`, `side`
  - `value` (corrections, e.g. "Corrected to Offensive Foul on …", seen on `instantreplay` actions with subType `challenge`).

**Coordinates.** `xLegacy`/`yLegacy` are basket-relative in tenths of a foot: per a dataset dictionary, `xLegacy = -99` means 9.9 ft left of the basket [41] (unverified: [41] now redirects to a Baselight shutdown notice and no longer shows the dictionary; verify by plotting).

**Label taxonomy.**
- BasketEvent derives 6 classes from structured fields: Missed Shot, Made Shot, Free Throw, Foul, Turnover, Rebound. It template-parses 4 more from `description`: Assist, Block, Steal, Jump Ball [20].
- In the live feed, `assistPersonId` / `stealPersonId` / `blockPersonId` replace the text parsing [39].
- **Use the live feed whenever available** (proposal).

**Per-event video.** hoopR's `nba_videoevents(game_id = "0021700807", game_event_id = "10", ...)` returns a list of tibbles, `videoUrls` and `playlist` [40]. The rebound dataset was built by "scraping the NBA Stats website" [19], and BARD's example clip links to an NBA Stats events page [25]; that they used this exact per-play endpoint is (unverified). BasketEvent does **not** say it used NBA Stats video: it retrieved "event-centered broadcast clips from the Internet" using game ID and game clock as anchors [20]. **R&D only**, because of [42].

**Corrections.** PBP is edited after the fact (`edited`, `value` [39]). Re-pull PBP after the final buzzer and the next morning, and version labels by `edited`. Final stats can change (e.g., assist and rebound re-credits) (proposal).

### 2.2 Video ↔ PBP alignment (step-by-step algorithm, proposal grounded in [22][23][43][44])

Prior art:
- 2016: Tesseract read score, period and clock per frame, which was matched to PBP [43].
- Stats Perform-style production pipelines use "fuzzy matching" over PBP + OCR + player/ball positions [44] (unverified: the fact-check could not extract text from the [44] PDF).
- BasketballBench sampled at 1 fps and read the scorebug with a VLM (Qwen3.6-27B). It kept clips only if every scorebug was readable and the clock counted down consistently [22].
- NBA_Streaming used OCR on the game clock (OCR model not named), then 3 annotators applied event-type boundary rules [23].

**Steps**

1. **Scorebug ROI.**
   - Detect the scorebug once per broadcast segment (template / layout detector), then crop.
   - Read `period`, `game_clock`, `shot_clock`, `score_home`, `score_away` at 5 Hz.
   - Formats: `MM:SS` above one minute, `SS.t` in the final minute (proposal).
   - OCR model: start with PP-OCRv5 (PaddleOCR 3.x per the model card's docs links, exact version unverified; Apache-2.0 [78]) or a tiny CRNN trained on synthetic scorebug digits. Keep VLM reading (as in [22]) as an offline fallback.
2. **Validity filter.**
   - Reject frames whose clock is not monotone non-increasing within a period.
   - Reject replay or graphics segments: no scorebug, frozen clock with camera cuts, or "REPLAY" overlays.
   - This mirrors the clip filter in [22].
3. **Fit `t_video = f(period, clock)`.**
   - For each run of *running-clock* frames, fit a robust line (RANSAC, slope ≈ −1 s/s).
   - Chain the runs with isotonic constraints.
   - Stopped-clock intervals become plateaus; many video seconds share one clock value.
4. **Place PBP events.**
   - Live-ball events (shots, rebounds, turnovers): `t = f(period, clock)`, then shift by the systematic lag. PBP lags the visible event by roughly 1–2 s [22]. Estimate the lag per league and season from the gold set.
   - Dead-ball events (fouls, FTs, subs, timeouts) all share one clock value. Order them by `actionNumber`/`orderNumber` within the plateau, and anchor them on:
     1. scorebug score deltas (FT made = +1 at a frame);
     2. `timeActual` differences (live feed);
     3. CV cues (ball at FT line, players lined up).
5. **Live mode (proposal).** For a stream we capture ourselves, estimate a constant broadcast offset `Δ = t_capture_utc − timeActual` as the median over made baskets. Each made basket must be detected both from a scorebug score delta and from PBP. Then map every PBP action to video with `t = timeActual + Δ`, which works even inside stoppages. That `timeActual` is the moment of the action and not the moment of data entry is **unverified**; check the residuals.
6. **Snap to CV.** Refine each aligned timestamp to a frame-precise anchor, then store both the PBP time and the snapped time:
   - shot release: ball leaves the shooter's hand with upward velocity;
   - rim contact or ball through the rim ROI;
   - rebound: first hand contact of the gaining player (the rebound paper's frame-level rule [19]);
   - possession change.
7. **Confidence.**
   - `align_conf = f(OCR conf, residual of fit, agreement of score-delta / CV snap)`.
   - Gold-check 5% of games by hand (proposal).

### 2.3 Auto-generating labelled event clips (proposal)

- **Window templates** around the aligned time `t` (tune on the gold set; PBP lag is 1–2 s [22]):
  - **shot (made/missed):** `[t−3.0 s, t+1.5 s]`, label from `shotResult`.
  - **rebound:** `[t_shot_end, t+2.0 s]`, team from `teamId`, OREB/DREB from `subType`.
  - **turnover / steal:** `[t−3.0 s, t+1.0 s]`.
  - **foul:** `[t−3.0 s, t+1.0 s]`.
  - **FT:** centred on the scorebug +1 delta (made) or the CV rim event (missed).
- **Rough yield.** BasketEvent yields ~155 event clips and ~229 player-event samples per game (34,931 clips and 51,663 samples over 226 games [20]). Expect O(10⁴) weakly labelled clips per 100 games.
- **Identity labels.**
  - `personId` comes from PBP.
  - The visual identity link is made with (a) team colour clustering and (b) jersey-number OCR constrained to the on-court roster.
  - Roboflow's recipe [31]:
    1. SigLIP embeddings → UMAP (3-D) → K-means (k = 2) for teams.
    2. Jersey OCR sampled every 5 frames; a number is accepted after 3 identical reads in a row.
    3. Number box matched to player mask with IoS ≥ 0.9.
  - BasketEvent assigned roster identities with SAM3 tracks plus Qwen2.5-VL jersey reading, without reported verification [20]. **Audit identity labels separately.**
- **Label-noise handling.**
  - Train with a bootstrapped loss: BasketEvent used bootstrapping β = 0.8 and a Background class weight of 0.1, with AdamW at lr 5e-5, batch 8, 10 epochs [20].
  - Drop clips whose `align_conf` is below a threshold.
  - Keep "ambiguous" clips for active learning (§2.5).

### 2.4 Self-training / pseudo-labelling

- **Teacher → student** (Noisy Student [64]): train a teacher on labelled data, pseudo-label unlabelled frames, then train an equal-or-larger student with noise (dropout, stochastic depth, augmentation).
- **For detection:**
  - Teacher: large offline models (RF-DETR-L/2XL, SAM 3 for masks).
  - Student: RF-DETR-S/N for real time.
  - End-to-end semi-supervised detection with a "Soft Teacher" also exists [65] (title verified only).
- **Basketball-specific pseudo-label filters (proposal):**
  1. temporal consistency: a box must persist ≥ 3 frames in a tracklet;
  2. physics: ball positions must fit a parabola during flight in court-3D, using calibration plus ball diameter (cf. DeepSportradar ball size → 3D [6]);
  3. roster constraints: ≤ 5 players per team on court, jersey numbers ∈ roster;
  4. PBP agreement: pseudo-events must match a PBP action within tolerance, otherwise they go to the review queue.
- **Ball detection** gains a lot from multi-frame input. TrackNet stacks 3 consecutive frames (Model II), regresses a heat-map whose target is "an amplified 2D Gaussian", and claims it "can even detect occluded ball by learning its trajectory patterns", though only 2 of 7 occluded balls were precisely detected in its tennis test [56]. Use it as a teacher or second-stage ball model (proposal).

### 2.5 Active learning (proposal)

Rank frames and clips for human review by these signals:

| Priority | Signal | Why |
|---|---|---|
| 1 | **PBP ↔ event-engine disagreement**: PBP says rebound by #23 but the engine says another player or no rebound | Highest value; directly fixes box-score errors |
| 2 | **Jev uncertainty**: Choice/Score `confidence` low or Noul p near 0.5 [75] (Noul returns no separate `confidence`, only the yes-probability, written p in the docs; its JSON field name is in `jev_api.md`, not [75]) | Jev's own uncertainty measure; the docs suggest top-probability and the top-to-second ratio from `probabilities` too [75] |
| 3 | Detector / tracker uncertainty: ball missing > N frames in live play, ID switches, team-cluster margin | Tracking failures cascade |
| 4 | Diversity: new arena, camera, uniform or lighting (embedding-space k-center sampling) | Domain shift |

Budget, a starting point: label ~300 frames per new venue (boxes + court keypoints) and ~200 events per game, each a PBP-disputed or low-confidence item.

### 2.6 SAM-assisted annotation and labelling tools

- **SAM 2.1** (Apache-2.0) [58]
  - Checkpoints and FPS: `sam2.1_hiera_tiny` 38.9M params / 91.2, `sam2.1_hiera_small` 46M / 84.8, `sam2.1_hiera_base_plus` 80.8M / 64.1, `sam2.1_hiera_large` 224.4M / 39.5. The README states "A100, torch 2.5.1, cuda 12.4" under the SAM 2 table; the SAM 2.1 table has no separate GPU note.
  - Video API: `build_sam2_video_predictor`, `init_state`, `add_new_points_or_box`, `propagate_in_video`.
  - The README targets CUDA, and MPS is not mentioned [58]. **Test on MPS (unverified)**, or run on a rented GPU.
- **SAM 3 / SAM 3.1** (SAM License) [57]
  - 848M params.
  - Text and exemplar prompts can detect, segment and track every instance of a concept, e.g. "basketball player", "basketball".
  - HF IDs `facebook/sam3` (gated) and `facebook/sam3.1`; SAM 3.1 (2026-03-27) adds Object Multiplex for faster multi-object tracking.
  - The **official repo** requires **Python ≥ 3.12, PyTorch ≥ 2.7 and a "CUDA-compatible GPU with CUDA 12.6 or higher"** [57]. So the official code path does not run on the M3.
  - **Correction:** this does not make SAM 3 unusable on the M3. Hugging Face Transformers ships a port (docs for v5.20.2):
    - images: `Sam3Model` / `Sam3Processor`, loaded with `Sam3Model.from_pretrained("facebook/sam3", device_map="auto")` [82];
    - video: `Sam3VideoModel` / `Sam3VideoProcessor`, with `processor.init_video_session(video=..., inference_device=...)` (default `"cpu"`), `processor.add_text_prompt(...)` and `model.propagate_in_video_iterator(...)`, plus a streaming mode [83].
    - Neither page states a CUDA requirement. MPS speed and numerical parity are (unverified). At 848M params the fp32 weights are ~3.4 GB, which fits in 18 GB.
    - **Plan:** prototype SAM 3 pseudo-labelling on the M3 via Transformers. Use a cloud GPU only for throughput.
  - Official-repo video API: `build_sam3_video_predictor()`, then `handle_request(request={"type": "start_session", ...})` / `"add_prompt"` [57].
  - Read the SAM License before commercial use (not reviewed here).
- **CVAT** (Community edition MIT [59])
  - Nuclio serverless auto-annotation functions; exports COCO / YOLO / MOT / CVAT XML [59].
  - **Correction: SAM 2 and SAM 3 are not in the self-hosted Community edition.**
    - SAM 2 video tracking: the December 2025 digest lists it under CVAT Enterprise, and says CVAT Online users reach it through AI Agents [80].
    - SAM 2 / SAM 3 built-in auto-labelling: the README places it in "CVAT Online paid plans (Solo, Team) and CVAT Enterprise" [59].
    - Community lists only the original SAM, as an interactor, and TransT as its tracker (Nuclio) [59].
  - SAM 3 is "available in all CVAT Online and Enterprise editions". The free Online tier gets demo mode only, and text-prompt annotation is "not wired into the UI yet" [60].
  - **Recommended for boxes, tracks and keypoints.** Self-hosting Community (MIT) means no SAM 2/3 tracker. Either pay for Online/Enterprise or run SAM 2.1/SAM 3 ourselves and import the masks.
- **Label Studio** (Apache-2.0 [61])
  - ML-backend SDK for pre-labelling, online learning and active learning [61].
  - SAM 2 video tutorial. Its limitations are dated "8/11/2024": GPU servers only, one tracked object at a time, no video segmentation, no Docker support [62].
  - VideoVector tag for SAM 2 is in Starter Cloud and above [63].
  - **Recommended for event/timeline QA and Jev or narration gold sets**: text and video side by side.
- **Roboflow Annotate:** hosted. Fastest path for the Roboflow-format datasets above; it does not change the licensing calculus.

---

## 3. Model training recipes

### 3.1 Detection (players, ball, rim, referee, jersey number)

**Class schema (proposal).** Use `player, referee, ball, rim, backboard, number`. Keep `ball-in-basket` and `player-in-possession` as auxiliary heads [31].

**RF-DETR** (preferred for production)
- Training API (`model.train(...)`) [50]:
  - Parameters: `dataset_dir, output_dir, epochs, batch_size, grad_accum_steps, lr, lr_encoder, resolution, weight_decay, device, use_ema, gradient_checkpointing, checkpoint_interval, resume, early_stopping, early_stopping_patience, early_stopping_min_delta, early_stopping_use_ema, tensorboard, wandb, project, run`. The page documents almost no defaults; the only stated one is that `checkpoint_interval` saves every 10 epochs.
  - Dataset: **COCO format** with `_annotations.coco.json` in each of `train/valid/test`. YOLO-format data must be converted to COCO first.
  - `resolution` must be divisible by 56.
  - Keep a total batch of 16 via `batch_size × grad_accum_steps`: 16 × 1 on an A100, 4 × 4 on a T4.
  - Outputs: `checkpoint.pth`, `checkpoint_<n>.pth`, `checkpoint_best_ema.pth`, `checkpoint_best_regular.pth`, `checkpoint_best_total.pth`.
- COCO AP50:95 by size, with native resolution and license [49]:

  | Size | AP50:95 | Native res. | License |
  |---|---|---|---|
  | Nano | 48.4 | 384 | Apache-2.0 |
  | Small | 53.0 | 512 | Apache-2.0 |
  | Medium | 54.7 | 576 | Apache-2.0 |
  | Large | 56.5 | 704 | Apache-2.0 |
  | XLarge | 58.6 | 700 | PML 1.0 |
  | 2XLarge | 60.1 | 880 | PML 1.0 |

- **M3 Pro recipe (proposal):**
  - `RFDETRSmall`/`RFDETRMedium`, `resolution` 728–1008 (multiples of 56) for ball recall.
  - `batch_size=2`, `grad_accum_steps=8`, `gradient_checkpointing=True`.
  - `epochs` 50–100 with `early_stopping=True`.
  - `device="mps"`: MPS support is not documented on the training page [50] (unverified; fall back to CPU or a rented GPU).
- Roboflow fine-tuned **RF-DETR-S** on the 10-class basketball set [31].

**Ultralytics YOLO26** (R&D)
- Sizes n/s/m/l/x; COCO mAP50-95 40.9 / 48.6 / 53.1 / 55.0 / 57.5 at 640 [51].
- Training pipeline: MuSGD (hybrid Muon + SGD) + Progressive Loss (the page now writes "Progressive Loss", not "ProgLoss") + STAL (Small-Target-Aware Label Assignment) [51].
- Tasks, as listed 2026-10-20: detect (`yolo26n.pt`…), instance segment (`-seg`), semantic segment (`-sem`), depth (`-depth`), classify (`-cls`), pose (`-pose`) and OBB (`-obb`) [51].
- MPS training: `model.train(data=..., epochs=100, imgsz=640, device="mps")`. MPS forces NCHW, and `channels_last=True` is ignored with a warning [52]. The YOLO26 model page itself does not mention MPS [51].
- Small-object knobs and their defaults [52]:
  - `imgsz` (640)
  - `rect` (False)
  - `mosaic` (1.0) with `close_mosaic` (10)
  - `multi_scale` (float, 0.0 = off)
  - `cache` (False)
  - `batch` (16; −1 = auto)
  - `patience` (100)
  - `epochs` (100)
- **Recipe (proposal):**
  - `imgsz=1280`, `multi_scale=0.25`, `close_mosaic=15`, `epochs=150`, `patience=30`, `cache="disk"`.
  - Use rectangular val (`rect=True`) for 16:9 broadcast.
- License AGPL-3.0 [51]: keep it out of the shipped service unless we hold an Enterprise license.

**Small-ball tricks**
1. **Higher input resolution** first: a ball in a wide broadcast shot is a few pixels.
2. **SAHI** (MIT [54]) for sliced inference and slice-aided fine-tuning.
   - Inference: `get_sliced_prediction(image, detection_model, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio, ...)`.
   - `AutoDetectionModel.from_pretrained(model_type="ultralytics" | "roboflow", ...)` [54].
   - Training data: `sahi coco slice` [54].
   - Paper gains on VisDrone/xView: +6.8 / +5.1 / +5.3 AP (FCOS / VFNet / TOOD) from sliced inference alone, and +12.7 / +13.4 / +14.5 AP cumulative with slice-aided fine-tuning [55]. The abstract writes these as "%", meaning AP points. These are aerial-imagery numbers, so expect a smaller gain on broadcast basketball.
   - `get_sliced_prediction` example from `docs/predict.md`: `slice_height=256, slice_width=256, overlap_height_ratio=0.2, overlap_width_ratio=0.2` [54]. RF-DETR loads with `model_type="roboflow"`; there is no `"rfdetr"` value [54].
   - Tiling multiplies latency, so use it **offline (teacher / pseudo-labelling)**. Real-time uses one high-res pass plus an ROI re-detect around the predicted ball position (proposal).
3. **Temporal ball model** (TrackNet-style 3-frame heat-map [56]) as a second stage on a crop around the last known ball.
4. **Augment with motion blur** and rim/net occlusions (proposal).

### 3.2 Court keypoints / homography

- **RF-DETR keypoints** [32]:
  - Model: `RFDETRKeypointPreview` (Apache-2.0; COCO-pose pretrained 71.8 AP [49]).
  - Data: fine-tuned on the 33-point `basketball-court-detection-2`.
  - Hyperparameters: epochs 50, resolution 576, batch 2, grad-accum 2, lr = lr_encoder = 1e-4.
  - Packages: `rfdetr[train,visual]>=1.8.1`, `supervision>=0.29.1` (the training extras are needed).
  - Select checkpoints on `val/keypoint_map_50_95`.
- **Evaluation:**
  - DeepSportradar calibration benchmark: MSE in cm of 6 reprojected points [4].
  - Our own metric: reprojection error of rim and FT-line points (proposal).
- A single-view homography on amateur footage degrades from poor line visibility, camera motion and domain shift [32] (via search summary of a Theseus thesis; treat as anecdotal).

### 3.3 Team / identity

- **Team:** SigLIP + UMAP + K-means(2) on central torso crops [31]. Refit per game and per half.
- **Jersey numbers:**
  - ResNet-32 classifier (93%) [31], constrained to the roster and to numbers on court.
  - Temporal voting: 3 consecutive identical reads [31].
- **Re-ID embedding:**
  - Pre-train on DeepSportradar re-ID (non-commercial) [5], fine-tune on own games.
  - The 2022 winner's CLIP ViT-L/14 reached 98.44% mAP on that benchmark [5] (too heavy for live; distil).

### 3.4 Event / action recognition: trajectories vs video

**Trajectory models (primary).**
- baller2vec, a multi-entity transformer on SportVU [47]:
  - 5 Hz, 4.2 s windows.
  - 6 layers, 8 heads, `dim_feedforward` 2048.
  - lr 1e-5, MIT license [47].
- CourtMotion (Dec 2025), which extends Baller2Vec++ [48]:
  - From the 631-game SportVU 2015-16 set, split 160 games for pre-training, 160 for downstream classifiers and 40 for evaluation. Positions are at 5 Hz; 30 Hz 3D skeletons are compressed to 5 Hz by a GNN.
  - 6 layers, 8 heads, d = 512, causal mask.
  - Event heads (BCE over past/current/future 2 s windows) for 9 labels: `no_action, free_throw, shot_attempt, pass, deflection, block, rebound, steal, dribble`.
  - Loss α·L_traj + (1−α)·L_events, α = 0.5; Adam, lr 5e-4, batch 32; trained ~100 h on 8× A100.
  - AP: shot-taker 0.91 at 0.8 s ahead; 2pt-vs-3pt 0.97 at 0.4 s; pick 0.85 at 0.0 s (0.82 at 0.6 s ahead) [48].
- **CourtVision recipe (proposal):**
  - Input `X ∈ ℝ^{T×11×F}`: T = 5 Hz × 4–6 s; 10 players + ball.
  - F = court x, y, vx, vy, team, is-ball-handler, ball z, distance to rim, nearest-defender distance, plus a player-ID embedding.
  - Model: 4–6-layer transformer, d = 256 (fits MPS). Use entity-set attention and a per-frame multi-label head.
  - Targets: Gaussian-smoothed per-frame event targets around snapped anchors.
  - Pre-train on SportVU. Labels come from SportVU events plus derived passes/dribbles: a ball-possession switch where the ball is within ~1 m of a player at < ~2 m height (heuristic, proposal).
  - Fine-tune on our CV tracks, whose noise distribution differs: jitter, ID switches.
  - Also train on SportVU with **injected tracking noise** that matches our measured tracker error.

**Video models (auxiliary).**
- PlayNet [20]:
  - TimeSformer (K400) with 12 clips × 8 frames at 224², ROIAlign entity tokens, entity/global cross-attention and gated clip pooling.
  - Results: F1 0.682 vs R3D 0.412, SlowFast 0.456, TimeSformer 0.329.
- Rebound baselines: X3D-m and TEAM (X3D + transformer) [19].
- Frame-precise spotting heads:
  - E2E-Spot (ECCV 2022, BSD-3; RegNet-Y + GSM; 1–2-frame tolerance) [67].
  - T-DEED (CVsports workshop at CVPR 2024). Its repo says the implementation took "1st place in the 2024 SNBAS Challenge". Code is **GPL-3.0**, so keep it out of shipped code [68][84].
- Use them for: make/miss confirmation, foul/whistle detection, and events in occlusion clusters (proposal).
- Train on our auto-clips with BasketEvent's settings as the starting point [20].

### 3.5 Shot-make probability model

**EPV shot model** (Cervone et al., JASA 111(514), 2016 [45])
- Form: `logit p_ℓ(t) = W_ℓs(t)'β_ℓs + ξ_ℓs(z_ℓ(t))`.
- Covariates W:
  - `log(1 + d_nearest_defender)`;
  - an indicator of having dribbled since gaining possession;
  - an intercept.
- Spatial effect ξ: Gaussian-process-like functional basis with d = 10 basis functions.
- Partial pooling: CAR priors across *similar* players. Similarity comes from NMF on court-occupancy histograms of 575 4-ft² offensive-half-court bins (461 × 575 matrix, square-root normalised, KL-type loss, rank r = 5); each player is linked to his 8 nearest players in the 5-D NMF space, then the adjacency matrix H is symmetrised (Appendix A.2).
- Fit with INLA on 2013-14 data (L = 461 players), validated on a 10% hold-out [45].
- "ESQ" (effective shot quality): a "learned least-squares regression function" on spatial inputs such as shot location and defender proximity [79]. The survey [79] does not name Second Spectrum or MIT Sloan 2014, so that attribution is (unverified).

**CourtVision recipe (proposal)**
- **Labels:** PBP `shotResult` (Made/Missed), `isFieldGoal`, plus the 3PT flag from `actionType`. Cross-check location with `xLegacy`/`yLegacy` [38][39].
- **Features at release frame:**
  - shot (x, y), distance and angle;
  - closest-defender distance, and whether that defender is between shooter and rim;
  - defender closing speed;
  - catch-and-shoot vs off-dribble (EPV's dribble flag [45]);
  - shot clock, game clock, score margin;
  - shooter ID (random effect / embedding).
- **Model:**
  - Start with a hierarchical logistic regression (player random intercepts, spatial spline basis), or LightGBM plus post-hoc calibration.
  - Pre-train on SportVU 2015-16, fine-tune on own games.
- **Evaluation:** log-loss, Brier, ECE (M = 15 bins [70]) and reliability diagrams. Split by game, never by shot.

### 3.6 Pass-target (receiver) model

- **EPV macro-transition hazards** [45]:
  - Competing-risks hazards: `log λ_ℓj(t) = W_ℓj(t)'β_ℓj + ξ_ℓj(z_ℓ(t)) + 1[j ≤ 4] ξ̃_ℓj(z_j(t))`.
  - Six types: 4 pass options, a shot attempt, a turnover.
  - Pass covariates: `dribble, ndef, closeness` (rank of teammate distance) and `open` (defender geometry relative to the passing lane).
  - Shot-attempt and turnover hazards use `dribble, ndef, ball_lastsec` (distance the ball travelled in the previous second). Every model has an intercept (Appendix A.1).
  - The exit model treats the receiver's state as their location at pass start.
  - Out-of-sample log-likelihood improves when covariates, spatial terms and hierarchical shrinkage are added (Table 1, e.g. Pass1 −29.4k → −26.4k) [45].
- **CourtVision recipe (proposal):**
  - Softmax over {4 teammates, shot, hold/drive, turnover}.
  - Per-candidate features: distance, the EPV `open` feature, passing-lane defender distance, receiver velocity toward the rim, receiver's own shot-make p.
  - Labels from tracked possession changes, confirmed by PBP assists where present.
  - Metrics: top-1 / top-3 accuracy, log-loss, ECE.
- **EPV itself (optional, later):**
  - Coarse states = ball-carrier × 7 court regions × defended (defender within 5 ft), plus transition and end states.
  - `ν_t = Σ_c E[X | C_δt = c] · P(C_δt = c | F_t)` [45].
  - Code and a full-game sample are in `dcervone/EPVDemo` (R; license not stated) [46].

### 3.7 Calibration (our models and Jev)

- **Metrics** [70]:
  - `ECE = Σ_m (|B_m|/n) · |acc(B_m) − conf(B_m)|` with M = 15 equal-width bins.
  - `MCE = max_m |acc(B_m) − conf(B_m)|`.
  - Reliability diagram = accuracy vs confidence per bin (plot bin counts too, since [70]'s diagrams omit them).
  - Temperature scaling: a single T fit on validation NLL; it leaves accuracy unchanged [70].
- **Jev:**
  - TypeSafe trains Jev with RLCD for calibrated probabilities [74][77].
  - The confidence docs do not describe measuring calibration or recalibrating on user data. They recommend domain-specific thresholds tested on your own data [75].
  - **So, per (question, pinned model version), we compute ECE and reliability on our gold set.** If ECE > 0.05 (proposal), fit isotonic regression on Jev's probability (Noul `p`; Choice/Score top-prob) and gate decisions on the recalibrated value.
  - Re-run this whenever the pinned version changes; aliases can change answers silently [74].
  - The Jev paper reports that a single generic question works well and that wording matters little on its benchmark [77]. Spend effort on `state` content (what the event engine tells Jev) rather than on prompt tweaking (to be validated in our domain).

---

## 4. Evaluation

### 4.1 Event spotting

- **mAP@δ**
  - A prediction is a TP if it falls within δ of a same-class GT timestamp. AP is computed per class, then mAP.
  - **Average-mAP** = area under mAP(δ):
    - SoccerNet "tight": δ ∈ 1…5 s.
    - SoccerNet "loose": δ ∈ 5…60 s [66].
  - Precedents: the rebound paper uses δ ∈ {1, 2, 3} s [19]; E2E-Spot uses 1–2 *frames* [67].
- **CourtVision set (proposal):**
  - `mAP@0.5s` for release / rim / rebound-touch anchors.
  - `mAP@1s`, `mAP@2s` for possession-level events.
  - Tight avg-mAP (1–5 s) for comparability.
- **F1 with tolerance window**, as in NBA_Streaming [23]:
  - One-to-one matching; a match needs both start and end within ±2 s.
  - Report **Event Response** (recall), **Correct Response** (precision), and Start Δt / End Δt MAE (s).
- **Who-did-it.** Report attributed F1, which requires class + time + `personId` (proposal). Also BasketEvent's Rec@1/3/5 and Hit@0.3/0.5 for temporal evidence (modified tIoU over the shorter segment) [20].
- **Live latency:** median and p95 time from the true event to emitted event, and to Jev decision (proposal).

### 4.2 Multi-object tracking

- **HOTA** [69]:
  - `HOTA_α = sqrt(DetA_α · AssA_α)`
  - `DetA_α = |TP| / (|TP| + |FN| + |FP|)`
  - `AssA_α = (1/|TP|) Σ_{c∈TP} A(c)`, with `A(c) = |TPA| / (|TPA| + |FNA| + |FPA|)`
  - Averaged over α ∈ {0.05, …, 0.95} (19 thresholds). LocA is average localisation similarity.
- **MOTA** = 1 − (FN + FP + IDSW) / GT [69].
- **IDF1** = IDTP / (IDTP + 0.5·IDFN + 0.5·IDFP) [69].
- **Tooling:** SportsMOT ships TrackEval-based scripts (`sportsmot_to_trackeval.py`) [27][28].
- **Baselines:** TeamTrack basketball side-view HOTA 75.2 (BoT-SORT) / 76.2 (ByteTrack) [30]. SportsMOT's top entries are around HOTA 74 (MixSort-OC, from a search summary; unverified).
- **CourtVision addition (proposal):**
  - Court-plane HOTA: replace IoU with a distance similarity `S = max(0, 1 − d/1.0 m)` on homography-projected foot points.
  - Report **identity accuracy** (fraction of frames where the track's `personId` is correct) separately from IDF1. Box-score attribution depends on it.

### 4.3 Box-score error (proposal; no standard benchmark found)

- **Ground truth:** the final official box score (league feed) after corrections. For own games, a human-scored sheet.
- **Metrics:**
  - per-player-game, per-stat **MAE** (PTS, FGM, FGA, 3PM, 3PA, FTM, FTA, OREB, DREB, AST, STL, BLK, TOV, PF);
  - **exact stat-line match rate**;
  - team totals MAE;
  - **points error** (must be ~0; scorebug deltas give an independent check);
  - **event-level attributed precision/recall** (which explains the MAE);
  - **time-to-correct** for live stat corrections.
- Report assists and rebounds separately, since scorer discretion makes them noisy. PBP itself gets edited post-game [39].

### 4.4 Narration factuality (Claude)

- **Atomic-claim precision** (FActScore [71]):
  - Split each utterance into atomic facts and report the % supported by the knowledge source.
  - FActScore's automated estimator had < 2% error vs humans on biographies [71].
  - Our knowledge source is the **frozen game-state log**: aligned PBP, event engine, box score, and the Jev answers given to Claude.
- **NBA_Streaming grounding metrics** [23]: **Identity Accuracy** (player names), **Event Type Accuracy** (15 types), **Fine-grained Action Accuracy** (71 fine-grained categories in 11 action families). The paper gives no formulas for these accuracies, so our implementation must define them (proposal).
- **Sports-commentary rubric precedents:**
  - SCBench's 6-dimension SCORES metric (Situation, taCtic, emOtion, backgRound, key Events, techniqueS), graded by GPT-4o-mini at temperature 0.1 [72].
  - TennisExpert's LLM judge with an accuracy-vs-ground-truth criterion (player identities, score states) [73] (unverified: the [73] PDF exceeded the fetch size limit during fact-checking).
- **CourtVision protocol (proposal):**
  1. Claude extracts claims offline, separate from narration.
  2. Each claim is typed: score, clock, player, event, stat, tactical.
  3. Deterministic checks for numeric and score/clock claims; target 100% for score/clock.
  4. A Jev `Noul` "is this claim supported by `state`?" check for semantic claims. It is cheap ($0.042/Mtok input, output free [74]). Calibrate it against ~500 human-labelled claims before trusting it.
  5. Report: contradiction rate, unsupported-claim rate, hallucinated-entity rate, and "stale fact" rate (claims true earlier but not at utterance time).
  6. Human audit of ≥ 100 utterances per evaluation game.

---

## 5. Step-by-step data flywheel for CourtVision

All phases write to one versioned store: Parquet tables plus video in object storage, keyed by `game_id` and `video_id`. Every row carries `source`, `rights_status ∈ {own, licensed, research_only}`, `label_origin ∈ {pbp, ocr, cv_auto, pseudo, jev, human}` and `label_conf` (proposal).

### Phase 0: Bootstrap (R&D-only data, weeks 1–3)
1. Load SportVU 2015-16 (631 games [15]) and dedupe moments [14].
   - Derive passes, dribbles and possessions.
   - Merge PBP (`event_info.type`) [14].
   - Train v0 of: event-engine transformer (§3.4), shot-make model (§3.5), pass-target model (§3.6).
   - Record "perfect-tracking" upper-bound metrics.
2. Train detector v0 (RF-DETR-S) on the Roboflow basketball sets [31][35]–[37]. Train court-keypoint v0 on the 33-point set [32]. Train jersey OCR v0 (ResNet-32) [31]. Treat licenses as research-only unless they are CC BY.
3. Benchmark (do not ship) on DeepSportradar calibration / segmentation / re-ID [3]–[5], SportsMOT basketball [27] and TeamTrack basketball [30].
4. Build the gold sets that will be **frozen for the life of the project** (proposal):
   - 10 games × full annotation of events with frame anchors and `personId`;
   - 2,000 frames of boxes/keypoints across ≥ 6 venues;
   - 500 Jev questions with human answers;
   - 300 narration utterances with claim labels.
   - Split by game and venue. DeepSportradar holds out whole arenas [4], SportsMOT never splits a game [29].

### Phase 1: Ingest live games
5. **Capture:** record video with frame PTS and capture wall-clock UTC.
   - Poll the live PBP JSON [39] (or the partner league feed) every 2–5 s.
   - Store every revision, keyed by `edited`.
6. **Scorebug OCR at 5 Hz** (§2.2, steps 1–2). Store `ocr_clock(video_id, pts_s, period, clock_s, shot_clock_s, score_home, score_away, conf)`.
7. **Alignment** (§2.2, steps 3–7). Store `alignment(game_id, actionNumber, pts_pbp_s, pts_snapped_s, method, residual_s, align_conf)`.
   - **Gate:** the game is usable only if median |residual| < 0.5 s on made baskets (proposal).

### Phase 2: Auto-label
8. Run the offline teacher stack on every ingested game:
   - high-res RF-DETR-L + SAHI tiles [54];
   - SAM 2.1 / SAM 3 masks: a cloud GPU for throughput, or the Transformers `Sam3VideoModel` port locally for small batches [57][58][83];
   - TrackNet-style ball [56];
   - tracker + team clustering + jersey OCR;
   - court homography.
   Then apply the §2.4 filters to produce pseudo-labels.
9. Generate event clips from aligned PBP with the §2.3 templates. Also produce scorebug-derived events: score deltas (made shots by team and value) and clock stops (whistles).
10. Run the **live (student) pipeline** on the same game and log every prediction: tracks, events, Jev calls (`model`, `state` hash, probabilities, confidence), narration.

### Phase 3: Disagreement mining and human QA
11. Build the review queue with the §2.5 priorities. The top of the queue is PBP ↔ engine disagreements plus Jev low-confidence items.
12. Annotate in CVAT (boxes, tracks, keypoints) [59] and Label Studio (event timelines, Jev answers, claim support) [61]. The SAM 2 tracker needs CVAT Enterprise or an Online paid plan, since self-hosted Community has none [59][80].
    - Use 2 annotators on 10% of items to measure agreement.
    - Write guidelines with frame-level rules, e.g. "rebound = first hand contact by the gaining player" [19].

### Phase 4: Train and distil
13. **Detector / keypoints:** fine-tune the student (RF-DETR-S/M) on human labels plus filtered pseudo-labels (Noisy-Student-style [64]).
14. **Event engine:** fine-tune on our tracks with PBP-aligned and human-corrected labels, using a bootstrapped loss for noisy rows [20].
15. **Shot / pass models:** refit with partial pooling [45]. Recalibrate with temperature or isotonic scaling on a held-out game set [70].
16. **Jev:** no training [74]. Instead:
    - regenerate per-question reliability diagrams and ECE;
    - refit the isotonic recalibration and thresholds;
    - revise `state` serialisation where reliability is poor;
    - keep the model version pinned (`jev-1.13.0` per `jev_api.md`).
17. **Claude:** no training. Refresh few-shot exemplars and system-prompt rules from the narration error log, and re-run the factuality suite (§4.4).

### Phase 5: Evaluate and gate (on frozen gold sets; proposal targets)

| Component | Metric | Promote if |
|---|---|---|
| Ball detector | Recall@IoU 0.5 (live-ball frames) | ≥ +2 pts vs current, no player-AP regression |
| Tracker | Court-plane HOTA, IDF1, identity accuracy | HOTA ≥ current; identity acc ≥ 95% |
| Event engine | mAP@1s, attributed F1 (±2 s) | ≥ current on all of shots / rebounds / TOs |
| Box score | per-stat MAE; PTS error | PTS error = 0 on ≥ 95% of games |
| Shot / pass models | log-loss, ECE (M = 15) | ECE ≤ 0.03 |
| Jev questions | ECE after recalibration; accuracy at operating threshold | ECE ≤ 0.05 |
| Narration | atomic-claim precision; Identity / Event / Action acc | ≥ 99% claim precision; 0 score/clock errors |

### Phase 6: Deploy, monitor, repeat
18. Ship the student models. Monitor drift live:
    - ball-missing rate;
    - team-cluster margin;
    - OCR failure rate;
    - Jev confidence histograms per question;
    - unsupported-claim rate.
    Any alarm sends that game segment to Phase 3.
19. Cadence: one weekly retrain per component, with a gold-set regression run on every candidate.
    - Per new venue: ~300 labelled frames before go-live.
    - Per season: refresh the uniform/jersey and roster data.

---

## 6. Hardware notes (M3 Pro, 18 GB, MPS)

- **Feasible locally:**
  - YOLO26 training (`device="mps"` documented [52]);
  - small RF-DETR fine-tunes (MPS undocumented; test first [50]). *Gap pass:* rfdetr 1.11.2's `TrainConfig` has no `device` field and uses PyTorch Lightning `accelerator="auto"` (§8.3);
  - trajectory transformers (d ≤ 256);
  - shot / pass models (CPU);
  - all OCR, alignment and evaluation code.
- **Rent a CUDA GPU for:**
  - SAM 3 / 3.1 at scale. The official repo requires CUDA ≥ 12.6 [57], but the HF Transformers port (`Sam3Model`, `Sam3VideoModel`) has no stated CUDA requirement and can be tried on MPS or CPU first; its performance there is (unverified) [82][83];
  - SAM 2.1 at scale [58];
  - large teachers (RF-DETR-L/2XL with SAHI);
  - video models (TimeSformer / X3D / T-DEED).
- CourtMotion-scale pre-training (8× A100, ~100 h [48]) is out of scope. Use a smaller transformer, or start from baller2vec-scale settings [47].

---

## 7. Legal and licensing checklist (proposal)

- NBA video and stats: R&D only without written permission [42]. Don't redistribute clips. Several 2025-26 datasets withhold video for this reason [19][24].
- Datasets: CC BY-NC(-ND/-SA) means no commercial training of shipped models (DeepSportradar [3][7], SportsMOT [27], PL-NBA [21], NBA_Streaming [24]).
- Models and tools:
  - Prefer Apache/MIT: RF-DETR N–L [49], SAM 2.1 [58], SAHI [54], CVAT Community [59], Label Studio [61].
  - Review AGPL (YOLO26 [51]), GPL-3.0 (T-DEED code [84]), PML 1.0 (RF-DETR XL/2XL/A/F/P [49]) and the SAM License (SAM 3 [57]) before production use.
- NBA statistics: Section 9(vii) of the ToU forbids any product or service featuring a database of "comprehensive, regularly updated statistics" without express prior consent [42]. A shipped live box-score feature built on NBA PBP needs a licence.

---

## Sources

1. DeepSportradar-v1 (arXiv 2208.08190) — https://arxiv.org/abs/2208.08190
2. DeepSportradar GitHub organisation — https://github.com/DeepSportradar
3. DeepSportradar instance-segmentation-challenge — https://github.com/DeepSportradar/instance-segmentation-challenge
4. DeepSportradar camera-calibration-challenge — https://github.com/DeepSportradar/camera-calibration-challenge
5. DeepSportradar player-reidentification-challenge — https://github.com/DeepSportradar/player-reidentification-challenge
6. DeepSportradar ball-3d-localization-challenge — https://github.com/DeepSportradar/ball-3d-localization-challenge
7. gabriel-vanzandycke/deepsport — https://github.com/gabriel-vanzandycke/deepsport
8. Dataset Ninja: DeepSportRadar. Lists CC BY-NC-SA 4.0 and 1,456 images / 4,104 objects, which conflicts with [3]; prefer [3]. — https://datasetninja.com/deep-sport-radar
9. SpaceJam — https://github.com/simonefrancia/SpaceJam
10. hkair Basketball-Action-Recognition (SpaceJam classes, R(2+1)D) — https://github.com/hkair/Basketball-Action-Recognition
11. linouk23/NBA-Player-Movements — https://github.com/linouk23/NBA-Player-Movements
12. NBA-Player-Movements `Moment.py` — https://raw.githubusercontent.com/linouk23/NBA-Player-Movements/master/Moment.py
13. sealneaward/nba-movement-data. The game-log folder listing is at https://github.com/linouk23/NBA-Player-Movements/tree/master/data/2016.NBA.Raw.SportVU.Game.Logs — https://github.com/sealneaward/nba-movement-data
14. HF dataset `dcayton/nba_tracking_data_15_16` — https://huggingface.co/datasets/dcayton/nba_tracking_data_15_16
15. NBA SportVU dataset page (631 games, 2015-16) — https://astro.paperswithcode.com/dataset/nba-sportvu
16. Ramanathan et al., "Detecting events and key actors in multi-person videos", CVPR 2016 (arXiv 1511.02917; PDF https://cs.stanford.edu/groups/vision/pdf/ramanathan2016cvpr.pdf) — https://arxiv.org/abs/1511.02917
17. Shakya et al., "Basketball-51", CS & IT 2021 — https://aircconline.com/csit/papers/vol11/csit110712.pdf
18. Basketball-51 on Kaggle — https://www.kaggle.com/datasets/sarbagyashakya/basketball-51-dataset
19. Barrera-Roy & Clapés, "See It Before You Grab It" (arXiv 2512.15386; HTML https://arxiv.org/html/2512.15386) — https://arxiv.org/abs/2512.15386
20. BasketEvent (arXiv 2607.21267) — https://arxiv.org/html/2607.21267v1
21. PL-NBA (arXiv 2608.19646; PDF https://arxiv.org/pdf/2608.19646) — https://arxiv.org/abs/2608.19646
22. "Towards Comprehensive Basketball Understanding" / BasketballBench (arXiv 2608.23435) — https://arxiv.org/html/2608.23435v1
23. NBA_Streaming (arXiv 2608.09200) — https://arxiv.org/html/2608.09200
24. NBA_Streaming repository — https://github.com/wyy-081/NBA_Streaming
25. BARD repository (project page https://bodai.unibs.it/?p=1198) — https://github.com/GabrieleGiudic/BARD
26. BASKET (arXiv 2503.20781) — https://arxiv.org/abs/2503.20781
27. SportsMOT repository — https://github.com/MCG-NJU/SportsMOT
28. SportsMOT on Hugging Face — https://huggingface.co/datasets/MCG-NJU/SportsMOT
29. SportsMOT paper (arXiv 2304.05170) — https://arxiv.org/abs/2304.05170
30. TeamTrack project page (arXiv 2404.13868) — https://atomscott.github.io/TeamTrack/
31. Roboflow, "How to Detect, Track, and Identify Basketball Players" — https://roboflow.com/blog/identify-basketball-players
32. Roboflow, "How to Fine-Tune RF-DETR Keypoints on Custom Data" — https://roboflow.com/blog/train-rf-detr-keypoint
33. roboflow/sports — https://github.com/roboflow/sports
34. Roboflow, "How to Train a YOLO26 Object Detection Model with Custom Data" — https://roboflow.com/blog/how-to-train-yolo26-custom-data
35. Roboflow Universe X-Basket 2 — https://universe.roboflow.com/x-qfelz/x-basket-2
36. Roboflow Universe basketball-jaawc — https://universe.roboflow.com/machinelearning-6xoxa/basketball-jaawc
37. Roboflow Universe baskball — https://universe.roboflow.com/dataset-baketball/baskball
38. nba_api PlayByPlayV3 docs — https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/playbyplayv3.md
39. nba_api live PlayByPlay docs — https://github.com/swar/nba_api/blob/master/docs/nba_api/live/endpoints/playbyplay.md
40. hoopR `nba_videoevents` — https://search.r-project.org/CRAN/refmans/hoopR/html/nba_videoevents.html
41. NBA Plays 2022-23 data dictionary (xLegacy/yLegacy semantics; secondary). Dead as of 2026-10-20: it 301-redirects to baselight.ai, which shows a shutdown notice — https://baselight.app/u/kaggle/dataset/brendankdata_nbaplays22_23
42. NBA.com Terms of Use (last updated 2026-07-13) — https://www.nba.com/termsofuse
43. "Leveraging Contextual Cues for Generating Basketball Highlights" (arXiv 1606.08955; OCR alignment) — https://arxiv.org/pdf/1606.08955
44. MIT Sloan SAC paper "Predicting NBA Talent from Enormous Amounts of College Basketball Tracking Data" (paper ID 91W8YT29R; authorship unverified) — https://global-uploads.webflow.com/5f1af76ed86d6771ad48324b/606dbe887622a74e9dc17f4b_91W8YT29R.pdf
45. Cervone, D'Amour, Bornn, Goldsberry, "A Multiresolution Stochastic Process Model for Predicting Basketball Possession Outcomes", JASA 111(514), 2016 (arXiv 1408.0777) — https://arxiv.org/abs/1408.0777
46. dcervone/EPVDemo — https://github.com/dcervone/EPVDemo
47. baller2vec (arXiv 2102.03291) — https://github.com/airalcorn2/baller2vec
48. CourtMotion (arXiv 2512.01478) — https://arxiv.org/html/2512.01478v2
49. RF-DETR repository — https://github.com/roboflow/rf-detr
50. RF-DETR training docs — https://rfdetr.roboflow.com/learn/train/
51. Ultralytics YOLO26 docs — https://docs.ultralytics.com/models/yolo26/
52. Ultralytics train mode docs — https://docs.ultralytics.com/modes/train/
53. Ultralytics YOLO Vision 2026 / YOLO27 announcement — https://www.ultralytics.com/blog/key-highlights-from-ultralytics-yolo-vision-2026
54. SAHI repository (MIT; docs/predict.md) — https://github.com/obss/sahi
55. Akyon et al., "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection", ICIP 2022 (arXiv 2202.06934) — https://arxiv.org/abs/2202.06934
56. TrackNet (arXiv 1907.03698) — https://arxiv.org/abs/1907.03698
57. SAM 3 repository — https://github.com/facebookresearch/sam3
58. SAM 2 repository — https://github.com/facebookresearch/sam2
59. CVAT repository — https://github.com/cvat-ai/cvat
60. CVAT blog: SAM 3 integration — https://www.cvat.ai/resources/blog/sam-3-image-segmentation
61. Label Studio repository — https://github.com/HumanSignal/label-studio
62. Label Studio SAM 2 video tutorial — https://labelstud.io/guide/ml_tutorials/segment_anything_2_video
63. Label Studio VideoVector + SAM 2 blog — https://labelstud.io/blog/video-object-segmentation-and-tracking-with-videovector-tag-for-sam-2-in-label-studio
64. Xie et al., Noisy Student (arXiv 1911.04252) — https://arxiv.org/abs/1911.04252
65. Soft Teacher (arXiv 2106.09018) — https://arxiv.org/abs/2106.09018
66. SoccerNet 2023 Challenges Results (tight/loose Average-mAP; also github.com/soccernet/sn-spotting) — https://arxiv.org/pdf/2309.06006
67. E2E-Spot (Hong et al., ECCV 2022) — https://github.com/jhong93/spot
68. T-DEED (arXiv 2404.05392, CVPRW 2024) — https://arxiv.org/abs/2404.05392
69. Luiten et al., HOTA (arXiv 2009.07736, IJCV 2021) — https://arxiv.org/abs/2009.07736
70. Guo et al., "On Calibration of Modern Neural Networks", ICML 2017 (arXiv 1706.04599) — https://arxiv.org/abs/1706.04599
71. Min et al., FActScore, EMNLP 2023 (arXiv 2305.14251; ACL Anthology 2023.emnlp-main.741) — https://arxiv.org/abs/2305.14251
72. SCBench (arXiv 2412.17637) — https://arxiv.org/html/2412.17637v1
73. TennisExpert (arXiv 2603.13397) — https://arxiv.org/pdf/2603.13397
74. TypeSafe docs: Models (versions, aliases, no fine-tuning, pricing) — https://docs.typesafe.ai/models
75. TypeSafe docs: Confidence — https://docs.typesafe.ai/confidence
76. OpenRouter, "What Is Jev?" — https://openrouter.ai/blog/insights/what-is-jev/
77. "Just Ask Jev: Reinforcement Learning for Calibrated Decisions…" (arXiv 2609.29429) — https://arxiv.org/abs/2609.29429
78. PaddlePaddle PP-OCRv5_server_rec model card (Apache-2.0) — https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec
79. "Spatio-Temporal Analysis of Team Sports — A Survey" (ESQ description; arXiv 1602.06994) — https://arxiv.org/pdf/1602.06994
80. CVAT digest, December 2025 — https://www.cvat.ai/resources/changelog/digest-dec-2025
81. BasketHAR (arXiv 2604.17065) — https://arxiv.org/pdf/2604.17065
82. Hugging Face Transformers docs: SAM3 (`Sam3Model`, `Sam3Processor`; v5.20.2) — https://huggingface.co/docs/transformers/model_doc/sam3
83. Hugging Face Transformers docs: SAM3 Video (`Sam3VideoModel`, `Sam3VideoProcessor`) — https://huggingface.co/docs/transformers/model_doc/sam3_video
84. T-DEED repository (GPL-3.0; CVsports @ CVPR 2024; SNBAS 2024 1st place) — https://github.com/arturxe2/T-DEED

---

## Verification log

Adversarial fact-check, 2026-10-20. Each implementation-critical claim was re-fetched from its primary source with WebFetch; many were also checked against a second source. Arxiv PDFs that the fetcher could not parse were read through ar5iv HTML, or through a local zlib text dump of the PDF.

The WebSearch budget for this run was already used up, so no new searches were made. Everything below comes from direct fetches of known URLs.

### Confirmed as written (no change needed)

**TypeSafe Jev [74][75][77]**
- "Jev is not fine-tuned or LoRA-adapted with customer data".
- `jev-latest` → `jev-1.13.0`; `jev-preview` → `jev-1.14.0`.
- $0.042/Mtok input; "Output tokens are free".
- Alias-drift warning; the response `model` field holds the versioned ID.
- 64k per request / 32k per question.
- RLCD training.
- Confidence docs: no calibration or recalibration guidance; thresholds are domain-specific.
- Jev paper: "Question wording matters little, while context matters more"; a generic question reaches median AUROC 0.886.

**nba_api [38][39]**
- PlayByPlayV3: URL; `GameID`/`StartPeriod`/`EndPeriod`; `GameID` regex `^\d{10}$`.
- The 23 PlayByPlay field names, in exact order, and `AvailableVideo.videoAvailable`.
- Live PBP: URL pattern and every listed `actions[]` field. Examples: `timeActual` `2021-01-16T02:14:17.3Z`, `clock` `PT03M59.00S`, x/y floats, `qualifiers`, `descriptor` `pullup`.

**Datasets**
- **BasketballBench [22]:** the 1-2 s PBP lag ("one or two seconds"); 1 fps sampling; Qwen3.6-27B; the readable-and-consistent-countdown filter; 7,980 examples, 10 tasks (including Scorebug Reading, Temporal Localization and Jersey-Number Recognition); 2,501 clips from the BasketEvent test split.
- **BasketEvent [20]:**
  - Counts: 226 games (189/37); 34,931 clips; 90.6 h; 51,663 samples; 10 classes + Background, with the 6 structured / 4 template split.
  - Results: PlayNet F1 0.682 vs SlowFast 0.456, R3D 0.412, TimeSformer 0.329; Rec@1/3/5; Hit@0.3/0.5.
  - Training: β 0.8, Background weight 0.1, AdamW at lr 5e-5, batch 8, 10 epochs.
  - Release: "will be made publicly available", with no data licence.
  - Identity: SAM3 + Qwen2.5-VL, with no reported verification.
- **Rebound paper [19]:** >100k clips; >300 h; 10 seasons; 1280×720 @ 60 fps; 1,916 valid labels (2,000 minus 84); D2K 1,500/250/250; D25K 23k/1k/1k; ~73k unlabelled; "upon NBA permission approval"; accuracy 0.881 (Table II); spotting mAP 0.46 on val (DREB 0.58, OREB 0.34); anticipation 0.60 vs human 0.71 at 0.5 s; first-hand-contact label rule. The code repo URL returns HTTP 404.
- **NCAA [16]** (via ar5iv): 257 games of ~1.5 h; 14K annotations; 11 classes (5 shot types × success/fail + steal); 212/12/33 videos; 4 s clips at 6 fps; 11,436 / 856 / 2,256 clips; 9,000 box frames.
- **DeepSportradar [3]–[7]:** calibration has 728 pairs, split 480/164/84, images 960×540, scored by MSE (cm) of 6 points, with a DeepLabv3 20-line baseline and `cv2.calibrateCamera`. Re-ID has 436 IDs / 8,569 images, 50/910 test, 618/9,505 challenge, ~72.7 mAP baseline and a CLIP ViT-L/14 winner at 98.44%. Instance segmentation has the OM metric, a Mask R-CNN baseline, Apache-2.0 code and CC BY-NC-ND 4.0 data. Ball-3D gives the KS-FR arenas; the deepsport repo states CC BY-NC-ND 4.0 and the 233/364 sample counts.
- **NBA_Streaming [23][24]:** 152 games (2025-26), 307.5 h, ~35K events, split 121/15/16; annotations CC BY-NC 4.0; videos by request with a "research-use and non-redistribution agreement"; Event Response = recall, Correct Response = precision; one-to-one matching with start and end within ±2 s; 15 event types.
- **PL-NBA paper [21]:** every number in the row matches the paper (60 games, 11,000 possessions, 31,567 sub-events, 13/22 labels, 7,037 / 1,005 / 1,641, 1080p, 48 kHz, CC BY-NC 4.0).
- **BARD [25]:** CC BY 4.0, 60 games, 24,692 raw → 14,676 clips, 720p, the JSON keys, CVIU 2026.
- **SportsMOT [27][28][29]:** CC BY-NC 4.0, no redistribution, 240 clips, >150K frames, >1.6M boxes, 45/45/150, 36.5 GB, 720p/25 fps, MOT17 format, `splits_txt/basketball.txt`, `sportsmot_to_trackeval.py`.
- **TeamTrack [30]:** basketball durations and box counts; side-view HOTA 75.2 / 76.2.
- **BASKET [26]:** 4,477 h, 32,232 players, 20 skills, 5 levels.
- **Basketball-51 [17]** (local PDF text dump): 10,311 clips, 51 games, 6 s, 320×240, 8 classes, 79% overall.
- **SpaceJam [9][10]:** ~32,560 examples; 10 classes; R(2+1)D on K400 at ~85%, 25 epochs, lr 1e-4, batch 8.
- **SportVU:** the `Moment.py` indices [12]; the shot-time heuristic [13]; the HF field names and overlap caveats [14]; 631 games, corroborated by CourtMotion [48].

**Models and tools**
- **RF-DETR [49]:** N/S/M/L are Apache-2.0; A/F/P and XL/2XL are PML 1.0; all Seg sizes (N…2XL) are Apache-2.0. The AP and resolution table is exact. Keypoint model `RFDETRKeypointPreview`: 71.8 AP, Apache-2.0.
- **RF-DETR training [50]:** COCO format; resolution divisible by 56; total batch 16 (4×4 on a T4); MPS not documented.
- **Ultralytics [51][52][53]:** YOLO26 released January 2026; AGPL-3.0/Enterprise; STAL = Small-Target-Aware Label Assignment; mAP 40.9 / 48.6 / 53.1 / 55.0 / 57.5. `device="mps"` example, NCHW on MPS, and the listed defaults. YOLO27 was announced at YV26 on 2026-09-13, "available later this year", with a "Coming soon" banner.
- **SAHI [54][55]:** MIT; the 6.8/5.1/5.3 and 12.7/13.4/14.5 gains; `model_type` values `ultralytics` and `roboflow`.
- **SAM 3 [57]:** 848M parameters; SAM License; `facebook/sam3` is gated; `facebook/sam3.1` released 2026-03-27 with Object Multiplex; Python ≥ 3.12, PyTorch ≥ 2.7, CUDA ≥ 12.6.
- **SAM 2.1 [58]:** Apache-2.0; parameter counts and FPS; video API names; no MPS mention.

**Models and formulas from papers**
- **EPV [45]** (ar5iv, including appendices): the shot logit form; covariates `ndef` = log(1+d), `dribble` and an intercept; d = 10 bases; 6 macro-transition types; pass covariates `dribble, ndef, closeness, open`; CAR prior over NMF similarity (575 bins of 4 ft²); INLA; 2013-14 data; L = 461; 90/10 hold-out; Pass1 −29.4k → −26.4k; 7 regions; defended = defender within 5 ft; 25 Hz.
- **baller2vec [47]:** MIT; 5 Hz; 4.2 s windows; 6 layers, 8 heads, ff 2048; lr 1e-5.
- **CourtMotion [48]:** 6 layers, 8 heads, d 512; the 9 labels; δ = 2 s; α = 0.5; ~100 h on 8× A100; AP 0.91 / 0.97 / 0.85.
- **Guo et al. [70]:** ECE and MCE formulas; M = 15 bins; diagrams omit bin counts; temperature scaling fit on validation NLL leaves accuracy unchanged.
- **FActScore [71]:** < 2% error on biographies.
- **SoccerNet [66]:** tight = 1–5 s, loose = 5–60 s.
- **E2E-Spot [67]:** BSD-3; 1–2 frame tolerance.
- **Roboflow [31][32][34]:** the 10-class list; RF-DETR-S; the 56/86/93% jersey results; SigLIP → UMAP (3-D) → K-means (k = 2); every 5 frames; 3 identical reads; IoS ≥ 0.9; the keypoint recipe (version 19, 33 points, epochs 50, resolution 576, batch 2, grad-accum 2, lr 1e-4, `val/keypoint_map_50_95`).
- **Tools [40][43][59][61]:** hoopR `nba_videoevents`; Tesseract OCR alignment in the 2016 highlights paper; CVAT Community is MIT with Nuclio; Label Studio is Apache-2.0 with its ML-backend features.

### Corrections made (error → fix)

1. **Jev quote [74].** "One set of weights serves all accounts" is not on the page. Replaced it with the actual wording: "the same weights serve every account" and "Jev is not fine-tuned or LoRA-adapted with customer data". Added the alias map and the 1.14 availability date (2026-10-14).
2. **NBA ToU [42].** The statistics clause reads "legitimate news reporting or private, non-commercial purposes", not news reporting alone. Added clause 9(vii), the comprehensive-statistics-database prohibition, to the TL;DR and §7. This is directly relevant to a live box-score product.
3. **SAM 3 on the M3 [57][82][83].** "Cannot run on the M3" was too strong. Only the official repo is CUDA-bound. Hugging Face Transformers provides `Sam3Model`/`Sam3Processor` and `Sam3VideoModel`/`Sam3VideoProcessor`, and their docs state no CUDA requirement (`init_video_session(inference_device=...)` defaults to `"cpu"`). Rewrote §2.6, Phase 2 step 8 and §6 to match. MPS performance remains (unverified).
4. **CVAT editions [59][60][80].** The note implied the self-hostable CVAT has a SAM 2 tracker and that SAM 3 is in "all editions". The sources say:
   - SAM 3 is in "all CVAT Online and Enterprise editions";
   - SAM 2 tracking is Enterprise (Online via AI Agents);
   - Community (MIT) lists only SAM v1 as an interactor and TransT as its tracker.

   Fixed §2.6 and Phase 3 step 12.
5. **BasketEvent clip source [20].** The note said BasketEvent scraped the NBA Stats per-play clip service. The paper says only that it retrieves "event-centered broadcast clips from the Internet", anchored on game ID and game clock. Fixed the TL;DR item 2 and §2.1 "Per-event video". The rebound set's "scraping the NBA Stats website" is kept [19]. That BARD used the same endpoint is marked (unverified).
6. **NCAA dataset URL [16].** `basketballattention.appspot.com` is not in the paper. Its only link is the YouTube source channel, and the paper says it "plan[s] to release" the data. Marked the URL (unverified).
7. **DeepSportradar licence citation [1].** The CC BY-NC-ND link on the arXiv page is the paper's licence. The data-licence citation now points to [3][7] only, including in the calibration row.
8. **DeepSportradar instance-segmentation splits [3].** Added a note that the repo contradicts itself: the script prints 223/37/64/84, while the prose says 246 = 211 + 35.
9. **PL-NBA [21].** The paper numbers are correct, but the GitHub README says "40 official NBA games", "7,000+" possessions and "30,000+" events. Added that discrepancy.
10. **TeamTrack [30].** Specified that 75.2 / 76.2 is the "SideView" subset (Table 5). "SideView2" (Table 6) gives 47.3 / 42.9.
11. **YOLO26 [51].** The page now says "Progressive Loss", not "ProgLoss". The task list now also includes semantic segmentation (`-sem`) and depth (`-depth`).
12. **RF-DETR training [50].** The parameter list was missing `project` and `run`. Added the A100 batch example (16×1), the full set of checkpoint filenames, and the fact that the page documents almost no defaults.
13. **RF-DETR keypoints [32].** The package pin is `rfdetr[train,visual]>=1.8.1`, not `rfdetr>=1.8.1`.
14. **Roboflow jersey set [31].** The blog says "3.6k images", not crops.
15. **YOLO26 tutorial [34].** It uses dataset version 10, so the 654-image count is tied to v10.
16. **TrackNet [56].** Added that it uses 3 input frames and a 2D-Gaussian heat-map target. Its occluded-ball claim rests on 2 of 7 occluded balls detected, so it is weak.
17. **T-DEED [68][84].** The venue is confirmed via the repo (CVsports @ CVPR 2024), as is the SNBAS 2024 first place. Added that its code is **GPL-3.0**, which matters for the licence checklist.
18. **SCBench [72].** The grader is GPT-4o-mini at temperature 0.1, not an unspecified "GPT grader".
19. **ESQ [79].** The survey does not name Second Spectrum or MIT Sloan 2014, so that attribution is marked (unverified).
20. **xLegacy semantics [41].** The source now redirects to a Baselight shutdown page. Marked (unverified).
21. **nba_api docs staleness [38][39].** Added that PlayByPlayV3 was "last validated 2023-09-14" and live PBP "Last validated 2020-08-16". Hard-coded field names must be checked against a live response.
22. **Smaller additions:**
    - EPV: shot and turnover hazards use `ball_lastsec`; NMF rank r = 5 with 8 nearest neighbours.
    - CourtMotion: the 160/160/40 game split, lr 5e-4 and batch 32.
    - SAHI: the abstract's "%" means AP points; the `get_sliced_prediction` example arguments.
    - SAM 2.1: the GPU is stated only for the SAM 2 FPS table.
    - Jev: Noul returns no separate `confidence`.
    - Label Studio: the full limitations list.
    - PaddleOCR: version stated as 3.x.
    - BASKET: venue marked (unverified).

### Could not verify (left marked "(unverified)")

- That [44] (the Sloan PDF) describes "fuzzy matching", because the PDF text could not be extracted.
- TennisExpert's judge criteria [73], because the PDF exceeded the fetch size limit.
- That paperswithcode lists 631 SportVU games [15], because of an SSL error. CourtMotion corroborates the figure.
- The BASKET (CVPR 2025) and TeamTrack (CVPRW 2024) venues.
- How MPS performs with the SAM 3 Transformers port, and whether RF-DETR trains on MPS.
- The SportsMOT leaderboard figure (HOTA ~74) and the BasketHAR details.
- Whether version 10 of `basketball-player-detection-3-ycjdo` has all 10 classes.
- The semantics of live-PBP `x`/`y` and whether `timeActual` is the action time or the data-entry time.

### Evidence URLs (primary, fetched 2026-10-20)

https://docs.typesafe.ai/models · https://docs.typesafe.ai/confidence · https://arxiv.org/abs/2609.29429 · https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/playbyplayv3.md · https://github.com/swar/nba_api/blob/master/docs/nba_api/live/endpoints/playbyplay.md · https://arxiv.org/html/2608.23435v1 · https://arxiv.org/html/2607.21267v1 · https://arxiv.org/html/2512.15386 · https://ar5iv.labs.arxiv.org/html/1511.02917 · https://github.com/DeepSportradar/camera-calibration-challenge · https://github.com/DeepSportradar/player-reidentification-challenge · https://github.com/DeepSportradar/instance-segmentation-challenge · https://github.com/DeepSportradar/ball-3d-localization-challenge · https://github.com/gabriel-vanzandycke/deepsport · https://arxiv.org/abs/2208.08190 · https://www.nba.com/termsofuse · https://github.com/roboflow/rf-detr · https://rfdetr.roboflow.com/learn/train/ · https://docs.ultralytics.com/models/yolo26/ · https://docs.ultralytics.com/modes/train/ · https://www.ultralytics.com/blog/key-highlights-from-ultralytics-yolo-vision-2026 · https://arxiv.org/abs/2202.06934 · https://raw.githubusercontent.com/obss/sahi/main/docs/predict.md · https://raw.githubusercontent.com/obss/sahi/main/LICENSE · https://github.com/facebookresearch/sam3 · https://github.com/facebookresearch/sam2 · https://huggingface.co/docs/transformers/model_doc/sam3 · https://huggingface.co/docs/transformers/model_doc/sam3_video · https://ar5iv.labs.arxiv.org/html/1408.0777 · https://arxiv.org/html/2608.09200 · https://github.com/wyy-081/NBA_Streaming · https://arxiv.org/html/2608.19646 · https://github.com/holhouse/PL-NBA-Dataset · https://github.com/GabrieleGiudic/BARD · https://github.com/MCG-NJU/SportsMOT · https://huggingface.co/datasets/MCG-NJU/SportsMOT · https://arxiv.org/abs/2304.05170 · https://atomscott.github.io/TeamTrack/ · https://huggingface.co/datasets/dcayton/nba_tracking_data_15_16 · https://raw.githubusercontent.com/linouk23/NBA-Player-Movements/master/Moment.py · https://github.com/linouk23/NBA-Player-Movements/tree/master/data/2016.NBA.Raw.SportVU.Game.Logs · https://github.com/sealneaward/nba-movement-data · https://search.r-project.org/CRAN/refmans/hoopR/html/nba_videoevents.html · https://roboflow.com/blog/identify-basketball-players · https://roboflow.com/blog/train-rf-detr-keypoint · https://roboflow.com/blog/how-to-train-yolo26-custom-data · https://arxiv.org/html/2512.01478v2 · https://github.com/airalcorn2/baller2vec · https://ar5iv.labs.arxiv.org/html/1706.04599 · https://arxiv.org/abs/2009.07736 · https://arxiv.org/abs/2305.14251 · https://ar5iv.labs.arxiv.org/html/2309.06006 · https://github.com/jhong93/spot · https://github.com/arturxe2/T-DEED · https://ar5iv.labs.arxiv.org/html/1907.03698 · https://arxiv.org/html/2412.17637v1 · https://ar5iv.labs.arxiv.org/html/1602.06994 · https://ar5iv.labs.arxiv.org/html/1606.08955 · https://github.com/cvat-ai/cvat · https://www.cvat.ai/resources/blog/sam-3-image-segmentation · https://www.cvat.ai/resources/changelog/digest-dec-2025 · https://github.com/HumanSignal/label-studio · https://labelstud.io/guide/ml_tutorials/segment_anything_2_video · https://labelstud.io/blog/video-object-segmentation-and-tracking-with-videovector-tag-for-sam-2-in-label-studio · https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec · https://github.com/simonefrancia/SpaceJam · https://github.com/hkair/Basketball-Action-Recognition · https://aircconline.com/csit/papers/vol11/csit110712.pdf · https://arxiv.org/abs/2503.20781 · https://github.com/arnalytics/labeled_plays_NBA (404)

---

## 8. Addendum: gap pass (2026-10-20): a shippable path to the first models

Cross-note decisions are in `docs/RESEARCH.md` §1A and §7A. Sources [G1]-[G6] are at the end.

### 8.1 Why bootstrap data cannot ship
Every bootstrap source in §1 is R&D-only, or its licence is unchecked: Roboflow Universe licences
returned 403, the jersey crops are NBA-playoff footage, the ball dataset comes from a shooting
robot, and SportVU states no licence. Partner platforms add a further block. Hudl's terms bar using
"any Hudl Content or data from the Services to train, fine-tune, or develop any machine learning"
model without written permission (§3.2.8) [G1]. So footage pulled from Hudl is not training data
unless the customer's own Hudl agreement allows it. Veo's terms let Veo itself train on customer
content (§6.3) [G2]. They do not grant third parties that right.

### 8.2 First-party data sprint (proposal; the sizes are planning targets, not findings)
| Item | Target | Why this size |
|---|---|---|
| Venues | ≥ 6 gyms (lighting, floor colour, camera height) | Matches the §5 gold-set spread |
| Games | ≥ 20 own or licensed games, with `training_allowed=True` per game | Separate consent from processing consent |
| Detector frames | ≈ 6,000 (≈ 1,000 per gym): player, referee, ball, rim, number | Base model; the "~300 frames per venue" figure covers adaptation only |
| In-flight ball boxes | ≥ 3,000 | Small-object recall is the bottleneck |
| Number crops | ≥ 5,000 legible crops across ≥ 100 jersey designs, pinnies included | Font and colour variety |
| Labelled shots (make/miss) | ≥ 1,000 | The isotonic-calibration threshold [G3][G4]. Platt scaling suffices from 300 (RESEARCH.md §7A.2) |

### 8.3 Training compute
- **MPS.** rfdetr 1.11.2's `TrainConfig` has no `device` field. A comment says PyTorch Lightning
  "auto-detects accelerator via `Trainer(accelerator="auto")`". `amp_dtype="fp16"` mentions
  "supported CUDA/MPS backends" [G5]. The training docs list only `cpu`/`cuda` devices, and give
  `batch_size=4`, `grad_accum_steps=4` on a T4 (effective batch 16), or `batch_size=16` on an A100
  [50]. **Action:** run a 1-hour smoke test with `rfdetr[train]==1.11.2` on MPS (loss falls, no CPU
  fallback storm) before relying on it.
- **Rented CUDA fallback.** Lambda on-demand prices, before tax: 1× H100 PCIe $3.29/h, H100 SXM
  $4.29/h, A10 $1.29/h, A6000 $1.09/h, GH200 $2.29/h [G6]. Example: a 24 h fine-tune on one H100 PCIe
  is about $79 (arithmetic, not a measured run time). Labelling time, not GPU time, dominates the
  cost of the first commercial model.

### 8.4 Release gate
CI refuses to export or package weights unless every training row's
`rights_status ∈ {own, licensed}` and its game has `training_allowed=True`. Each weights file
embeds the hash of its training manifest, so a shipped model can always be traced back to its rows.

### 8.5 OCR rate
This note's 5 Hz scorebug rate is now the project-wide rate (RESEARCH.md D14). A score change is
confirmed by 2 identical consecutive reads (≈ 400 ms).

Sources for this addendum:
[G1] Hudl Terms (updated 2026-08-11). https://www.hudl.com/terms
[G2] Veo General Terms and User Terms (2026-05-29). https://www.veo.com/terms
[G3] Niculescu-Mizil & Caruana, "Predicting Good Probabilities With Supervised Learning", ICML 2005. https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf
[G4] scikit-learn User Guide, "Probability calibration". https://scikit-learn.org/stable/modules/calibration.html
[G5] rf-detr source, `src/rfdetr/config.py`. https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/config.py
[G6] Lambda GPU Cloud pricing. https://lambda.ai/service/gpu-cloud
