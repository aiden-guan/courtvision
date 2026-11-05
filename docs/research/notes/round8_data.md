> Research notes compiled on 2026-11-05 by a research agent of the cloud session (round 8; docs/research/cross_game_detector.md §13), lightly reviewed. Figures marked as estimates or *unverified* were not checked; the sources are cited in place. Paths to that session's scratchpad (crop sheets, drawings) are not in the repository.

# New public data for the broadcast detector (players, referees, ball; crowd and benches left alone)

Researched on 2026-11-05 for `courtvision` (read only; nothing in the repository was changed). The goal was data for
the referee weakness (the light-grey shirts of the 2017 Finals broadcast) and the crowd/bench weakness, then the ball,
then unlabelled footage. Every number below was read at the source named next to it; anything I could not read
myself is marked **(unverified)** or **(snippet)**. "Snippet" means it came from a web-search result, not from the page.

## 0. How this was checked, and what could not be

* **Hugging Face:** API metadata, file trees and dataset cards, read with `curl` through the session proxy. WebFetch
  could not resolve `huggingface.co`. For E-BARD I read only the zip's central directory (1.09 MB) and one 53 kB
  annotation member, using HTTP Range requests. Nothing was extracted and no image was fetched.
* **Kaggle:** the public metadata API (`/api/v1/datasets/view/<owner>/<slug>` and `/api/v1/datasets/list/<owner>/<slug>`),
  plus three text files of at most 1.1 kB (the `README.roboflow.txt`, `README.dataset.txt` and `data.yaml` of one
  mirror).
* **Zenodo:** the REST API (`/api/records`).
* **GitHub:** `raw.githubusercontent.com` READMEs, configs and loader code.
* **arXiv:** the PDFs of WASB (2311.05237) and SAM (2007.09470), and the HTML of SportsHHI (2404.04565).
* **Could not read:**
  * Roboflow Universe pages: a Cloudflare challenge returned HTTP 403. Roboflow-only facts are therefore snippets.
  * `ruiyan1995.github.io/SAM.html`: HTTP 404.
  * The Wikimedia Commons API: HTTP 429, not retried.
  * Gated HF files (SportsHHI, SportsShot, SportsAction, FineVideo).
* **Downloads:** no dataset was downloaded. The largest single read was the WASB paper PDF (3.9 MB).

## 1. The rules applied (`docs/TRAINING.md` §1)

* **NBA video** is R&D only and quarantined, so every NBA-footage source below is `rights_status=research_only`. That
  holds even where the labels are CC BY or MIT.
* **NCAA championships** are prohibited. MultiSports-derived sets (SportsHHI, SportsGrounding, SportsMOT itself) contain
  NCAA tournament broadcasts. SportsMOT `val` already holds one: Kentucky v Wichita State on CBS, `v_00HRwkvvjtQ`
  (research note `cross_game_detector.md` §2). The Roboflow set `basketball-players-fy4c2` has frames named
  `Iowa_Nebraska-2023-03-06-college-men`, and Roboflow's `half-court-basketball` has Iowa and Nebraska player classes
  (snippet). Whether either is an NCAA championship game is unverified. The owner should decide whether to drop these.
* **YouTube: never download.** No candidate below needs a YouTube download. Several academic sets were sourced from
  YouTube by their authors and are hosted on HF or Zenodo. That is the same footing as SportsMOT, which the repository
  already uses. They are marked "YouTube-sourced" so the owner can confirm that reading of rule 3.
* **CC BY-NC(-ND/-SA) data** is for evaluation and pre-training only. The owner accepts research-only use.
* **Access barriers are flagged.** This covers anything behind a request, an e-mail, a signed agreement, a login with
  terms (HF "gated: auto") or an account (Kaggle, Roboflow, Baidu).

## 2. Top 5, ranked by expected value for the referee and crowd weaknesses

| # | Dataset | Why it ranks here | Labels that matter | Licence (labels) | Access | Main caveat |
|---|---|---|---|---|---|---|
| 1 | **E-BARD-detection** (HF `GabrieleGiudici/E-BARD-detection`) | Referees and players boxed on 1,800 frames of **60 NBA games / all 30 teams**, against 8 games now. On-court people only, so the crowd and benches are background. It also gives a 30-game held-out referee/crowd test. | 15,296 players, **3,853 referees**, 1,496 balls, 1,565 hoops; COCO JSON | CC BY 4.0 | Ungated HF; one 663 MB zip | Same era as now (2024-25), so it does not fix the 2017 shirts by itself. Its splits mix games: re-split by BARD game. |
| 2 | **DeepSport / DeepSportradar Basketball-Instants** (Kaggle, plus the instance-segmentation COCO files on GitHub) | The only verified labelled data from **2017-2018** with referees as positives. Every human on the court is masked; the crowd is not labelled. Calibration lets coaches be separated by court position. It also has the ball. | 324 images / 2,495 "human" instances (players, coaches, referees) in COCO; ball 3D points; 15 French LNB arenas | CC BY-NC-SA 4.0 (Kaggle listing) / CC BY-NC-ND 4.0 (v1 set, challenge README) | Free Kaggle account | Keemotion fixed-camera views, not TV broadcast. Small. Roles are not separated. |
| 3 | **"AI Sports Analytics System" v7** (Roboflow `asas-annotations`; Kaggle mirror `agrimrai29/basketball-yolo8`) | The largest referee-labelled NBA broadcast set found. Frames of NBA "FGM" clips, with team codes from 2014-15 or later. | `basketball, net, player, referee`; 16,486 exported images (3 augmented copies per source image) | MIT (Roboflow export); "Unknown" on Kaggle | Kaggle mirror (free account); Roboflow (account + API key) | Era, labelling policy and unique-frame count unverified. Images are stretched to 640x640. Check a sample first. |
| 4 | **NUST-NBA181 frames + WASB basketball ball labels** | The **2018-19 NBA** season, the closest era to the failing 2017 broadcast, at 1080p. **275,328 frames with human ball labels** (TrackNet-style). Referee and crowd value comes through a teacher plus `court-people`. | Per-frame ball centre and visibility (CVAT XML); no person labels | WASB repo MIT; footage "available upon request" | **Request to the SAM authors (flag)**; the annotation zip is public (5.8 MB) | The frames page now returns 404, and the set is about 135 GB (snippet). |
| 5 | **SportsHHI basketball keyframes** (HF `MCG-NJU/SportsHHI`) | Manual boxes of every player (about 9.5 a keyframe) on **6,791 keyframes of 80 broadcasts** (NBA/NCAA/FIBA). Possibly the same 80 videos as SportsMOT, which would label the 50 test sequences SportsMOT withholds (unverified). It strengthens the crowd-is-background signal. | 64,549 human boxes, players only | CC BY-NC 4.0 (HF tag); paper CC BY-NC-SA 4.0 | **HF login + auto-approved terms (flag)**; 24 GB with volleyball | Players only: use it with `mark-ignore --referees`. Remove SportsMOT `val` videos. YouTube-sourced; NCAA question. |

Runner-up: **SHOT** (HF `muyu111/basketball`, §3.6). It has older NBA eras (games with the New Jersey Nets, so 2012
or earlier) with player-only keyframe boxes, CC BY-NC 4.0 and no gate. It ranks under SportsHHI because only its
keyframe boxes are human labels (their number is unverified) and its tracks are tracker output. **Exclude its
`GSvsCLE` samples** (see §5).

**Not found anywhere:**

* a public basketball dataset that labels spectators or benches as a class;
* any public 2015-2019 NBA broadcast set with referee boxes.

The practical route to both is the repository's own teacher plus `court-people` (or a variant of `mark-ignore` that
promotes on-court non-players to positives) run on older footage: NBA181 2018-19, SHOT, SportsMOT `test`. See §6.

---

## 3. Candidates in detail

### 3.1 E-BARD-detection: rank 1

* **URL:** https://huggingface.co/datasets/GabrieleGiudici/E-BARD-detection
  * Code: https://github.com/GabrieleGiudic/E-BARD
  * Project page: https://bodai.unibs.it/basketball-action-recognition/
* **Licence: CC BY 4.0.**
  * Read on the HF card (`license: cc-by-4.0`).
  * Read in the GitHub README ("licensed under the Creative Commons Attribution 4.0 International License").
  * The footage is the NBA's, as for BARD.
* **Contents**
  * The card says: "Extracted from 60 full-game recordings from the 2024-2025 NBA season, covering all 30 teams";
    "1,800 frames, systematically sampled at 30 frames per game"; "22,210 manual, object-level annotations". The
    classes are 15,296 players, 3,853 referees, 1,565 hoops and 1,496 basketballs.
  * Broadcast frames at 1280x720 (read from the test annotations).
  * In the 180 test frames, players per frame are median 9, p95 10, max 12. Referees per frame are median 2, max 4.
    Person boxes are 94-354 px high (1st-99th percentile).
  * Those counts suggest only on-court people were boxed, so benches and the crowd are background. This is inferred;
    the card does not state the policy.
  * Same 60 games as BARD: names such as `bkn-vs-det-0022400861_000013.jpg` match the `bard.calib_games` (30) and
    `bard.check_games` (30) lists of `docs/research/cross_game_detector_splits.json`.
  * None of the 60 games is Lakers @ Warriors, so the README's 2024-12-25 test game is not touched.
* **Format and size**
  * `all.zip` is 662,567,020 bytes. Inside:
    * `coco/annotations/instances_{train,val,test}.json`, with categories `0 basketball, 1 hoop, 2 player, 3 referee`;
    * images in `coco/{train,valid,test}/` (1,440 / 180 / 180);
    * the same images again under `yolo/` and `yolo_rescaled/`;
    * Qwen grounding JSONs.
  * **E-BARD's own splits mix games.** The 180 test frames cover 56 of the 60 games.
  * A YOLO-only copy is on Kaggle (CC BY 4.0, 290 MB): https://www.kaggle.com/datasets/ilkinalis/ebard-basketball-detection
* **Rules:** fits as `research_only`. It is NBA footage (rule 3). The labels are CC BY 4.0 and need a credit to E-BARD
  (Giudici et al.). There is no form.
* **How to plug it in**
  1. **Fetch.** Add an `ebard` target to `tools/fetch_data.py`: one HTTPS file, one request, with `_retry` back-off,
     streamed to disk. To be gentler, Range-read only the `coco/` members. The central directory is about 1 MB, and
     `coco/` holds each image once, against three copies in the whole zip, so this cuts the transfer to roughly a third.
  2. **Rename the ball class.** Change category `basketball` to `ball`. `prep_coco` and `detector_bench.py --coco`
     both match the literal name `ball`; `player` and `referee` already match.
  3. **Merge the three image folders** into one; file names are unique. Then write two JSONs following the BARD split:
     `calib_ann.json` (30 calib games) and `check_ann.json` (30 check games).
  4. **Training folder:**
     `tools/student_finetune.py prep-coco data/ebard/calib_ann.json data/ebard/images data/ft/ebard_calib --referees`.
     The `--games` and `--exclude` prefix filters also work directly with the game IDs.
  5. **New held-out test:** `tools/detector_bench.py --coco data/ebard/check_ann.json ...`. This is about 900 frames and
     roughly 1,900 referees in 30 unseen games, if the 30 frames per game hold for every game. The current referee
     test has 506 referees in 5 games. Never train on the check games: they sit next to the BARD `check_clips` ball test.
  6. **Fix the licence text.** `prep_coco` writes `"license": "labels CC BY 4.0 (Roboflow Universe); footage NBA"` into
     `meta.json`, so set the right text for E-BARD.
* **Side uses**
  * **Refit the referee check.** `tools/referee_classifier.py --source data/ebard/calib_ann.json:data/ebard/images`
    would refit `nba/referee.py` on 30 games instead of 8. It needs a one-line change to its game parser, which
    expects `...-game-N` names.
  * **Gray jerseys.** `GabrieleGiudici/E-BARD-TeamAttribution` (CC BY 4.0) labels jersey colours of 15,295 players,
    including "Gray" 261. That helps check grey-jersey players against the referee check (the Thunder v Nuggets
    confusion in §11.1).
  * **A referee-aware teacher.** `GabrieleGiudici/E-BARD-detection-models` (CC BY 4.0; RF-DETR Nano and YOLOv8n with a
    `referee` class) could pseudo-label referees on older footage (§6). The YOLOv8n weights fall under the AGPL rule
    (R&D only). Both are research-only.

### 3.2 DeepSport / DeepSportradar Basketball-Instants: rank 2

* **URLs**
  * https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset (v4, 6,599,984,284 bytes, "700+
    high-resolution professional basketball images").
  * https://www.kaggle.com/datasets/gabrielvanzandycke/deepsport-dataset (v1, 2.7 GB, 300+ images).
  * COCO files: https://github.com/DeepSportradar/instance-segmentation-challenge/tree/master/annotations
* **Licence:** three sources disagree. All are non-commercial, so it is research-only either way.
  * The Kaggle listing of the Instants set says **CC BY-NC-SA 4.0** (read via
    `https://www.kaggle.com/api/v1/datasets/view/deepsportradar/basketball-instants-dataset`).
  * The v1 DeepSport set's Kaggle listing says **CC BY-NC-ND 4.0**.
  * The challenge README's License section says "The challenge data ... is available under the CC BY-NC-ND 4.0
    license".
  * The repository's training note records CC BY-NC-ND only. Treat the stricter ND terms as binding, and do not
    redistribute converted labels.
* **Contents**
  * "Acquired ... using the Keemotion system installed in multiple arenas ... the LNB for the rights on their images"
    (Kaggle description).
  * Pairs of images 40 ms apart, 2-5 Mpx, from cameras that "capture a fixed part of the basketball court". These are
    production-camera views, not the cut TV broadcast.
  * Masks per pixel: human = 1, ball = 3. Ball 3D centre with a `visible` flag, and camera calibration per image.
  * The challenge COCO files have one class, `human` (id 0): "individual humans (players, coaches and referees) on a
    basketball court". Spectators are unlabelled, so they are background.
  * Counts (README): train 223 images / 1,674 instances, val 37 / 344, test 64 / 477, plus 84 unlabelled `challenge`
    images.
  * **Arenas and era**, read from the COCO file names:
    * train and val: 12 arenas (Blois, Bourg-en-Bresse, Fos, Gravelines, Le Mans, Monaco, Nancy, Nantes, Poitiers,
      Saint-Chamond, Strasbourg, Vichy);
    * test: 3 arenas (Caen, Limoges, Roanne);
    * the Unix-ms timestamps in the names date the images to **2017 and 2018**, for example
      `KS-FR-BLOIS/24330/camcourt1_1513710529019` = 2017-12-19.
  * The officials are FIBA-rules referees. Their kit colour was not checked.
* **Size and download**
  * 6.6 GB from Kaggle with a free account and API token (`kaggle datasets download deepsportradar/basketball-instants-dataset`).
  * Flag: a Kaggle account is required; there is no form.
  * The COCO JSONs come from GitHub raw.
* **Rules:** fits as research-only (NC). ND forbids sharing adapted material, so keep converted labels local.
* **How to plug it in**
  1. In a copy of the challenge JSON, map `human` to `player`, or add a `--person-classes` option to `prep-coco`.
     `file_name`s are relative to the Kaggle root (`KS-FR-.../camcourt1_..._0.png`).
  2. Convert the PNGs to JPEG at the student's save width first. `prep_coco` byte-copies its inputs into
     `frames/NNNN/000001.jpg`.
  3. Coaches are positives in this label set. Use each instant's calibration to place a person's foot point on the
     court. Mark people beyond the lines as `ignore`, as `court_zone` does (the FIBA court is 28 x 15 m), so benches stay
     out of the person loss.
  4. Hold out by arena; the challenge's test arenas are the natural held-out set.
  5. For the ball, project the JSON's 3D `Ball` position with the calibration (`calib3d`, `deepsport_utilities`) and
     write `ball_labels.jsonl` rows (`xy`, `w` 1 if visible). This needs a small converter.
* **Also there:** `gabrielvanzandycke/ballistic-raw-sequences` (Kaggle, CC BY-NC-SA 4.0, 4.6 GB) holds raw Keemotion
  sequences with per-instant JSON. One example is `KS-FR-GRAVELINES-2018/...1572364075764`, dated 2019-10-29. It is
  useful for ball-in-flight checks.

### 3.3 "AI Sports Analytics System" v7 (Roboflow) via the Kaggle mirror: rank 3

* **URLs**
  * Mirror: https://www.kaggle.com/datasets/agrimrai29/basketball-yolo8 (1,235 MB; Kaggle licence field "Unknown").
  * Original: https://universe.roboflow.com/asas-annotations/ai-sports-analytics-system/dataset/7 (not readable here).
* **Licence: MIT** as declared by the Roboflow export. The mirror's `README.dataset.txt` says "License: MIT" and its
  `data.yaml` says `license: MIT`. Kaggle shows "Unknown". The footage is NBA broadcast, so `research_only`.
* **Contents**
  * The README says: "The dataset includes 16486 images", "Resize to 640x640 (Stretch)", "3 versions of each source
    image" (augmentation). The format is YOLOv8.
  * `data.yaml`: `names: ['basketball', 'net', 'player', 'referee']`.
  * File names are like `ATL_HOU_FGM_10_mp4-0005_jpg.rf.<hash>.jpg`, i.e. frames of per-game field-goal-made clips.
  * The sampled test page showed 6 matchups (ATL_HOU, BKN_NOP, BOS_PHI, CHA_NOP, CHI_CLE, CLE_WAS). The team codes BKN,
    NOP and CHA mean 2014-15 or later. **Season(s) unverified.**
  * The unique-frame count after removing augmented copies is **unverified**. If only the training split was tripled,
    it is roughly 5,500-6,500 (an estimate).
  * **Labelling policy unverified:** I could not check whether benches, coaches or fans were boxed as `player`, or
    whether every referee was boxed.
* **Rules:** research-only (NBA footage). Flags: the Kaggle mirror needs a free Kaggle account for the whole set, and
  Roboflow needs an account and API key.
* **How to plug it in**
  1. Convert YOLO to COCO. Un-stretch to the source aspect ratio, probably 16:9 (check one image).
  2. Keep one copy per source frame: the name before `.rf.` identifies it.
  3. Rename `basketball` to `ball`; drop `net`.
  4. `prep-coco ... --referees`, split by matchup prefix.
  5. Drop any `GSW_CLE` / `CLE_GSW` clips (§5).
  6. Inspect about 50 frames for the labelling policy before training.
* **Related:** `yampolal/basketball-data` (Kaggle, MIT, 2.2 GB, 31,696 images, classes `ball, player, ref`) is a
  "fusion of several datasets" whose test images include the same `ATL_HOU_FGM_*` frames. Its other sources are
  undocumented. It may contain Roboflow frames of the repository's held-out games, and its MIT tag may not match its
  sources' licences. **Not recommended** unless deduplicated against `roboflow_coco.heldout_games`.

### 3.4 NUST-NBA181 frames + WASB basketball ball labels: rank 4 (also the best ball set)

* **URLs**
  * WASB code and labels: https://github.com/nttcom/WASB-SBDT (paper arXiv 2311.05237, BMVC 2023).
  * Annotation zip: `https://drive.google.com/uc?id=1eH3n4uB4d8T-YKLRh46SshD0jYqQbnGF`. A HEAD request returned
    `basketball_annos.zip`, 5,793,468 bytes. This is the URL WASB's `setup_basketball.sh` fetches.
  * Frames: the SAM "NBA" dataset (Yan et al., ECCV 2020, arXiv 2007.09470).
* **Licence**
  * The WASB repo carries an MIT badge. No separate licence was found for the annotation zip.
  * The frames: the SAM paper says "Due to copyright restriction, this dataset is available upon request".
  * A snippet of the old SAM page said it is for academic research only and that you contact the author for the
    password. That page (https://ruiyan1995.github.io/SAM.html) returned 404 on 2026-11-05.
* **Contents**
  * The SAM paper says "181 NBA games of 2019 periods", 1920x1080, with 9,172 clips.
  * The WASB config's test video IDs (`21800974`, `21801076`, ...) are NBA game IDs `00218xxxxx`, i.e. the **2018-19
    regular season**.
  * The WASB paper says: "we manually annotated ball positions to 45% (81/181 games) of the whole video clips,
    resulting in 275,328 annotated images composed of 3,824 video clips".
  * WASB's split (paper, Table 1): train 70 games / 3,392 clips / 244,224 frames; test 11 games / 432 clips / 31,104
    frames. The mean ball displacement is 33.7 px per frame. The config `src/configs/dataset/basketball.yaml` lists 30
    test video IDs, which does not match the 11 test games; check this on the data.
  * The format is CVAT XML point tracks per clip with `outside` and `occluded` attributes (WASB
    `src/datasets/basketball.py`).
  * There are no person labels.
* **Size:** about 135 GB of frames (snippet; unverified) plus the 5.8 MB of labels.
* **Rules:** NBA footage, so `research_only`. **Flagged: the frames come by request (e-mail and password).** The SAM
  page is gone, so contact through the paper.
* **How to plug it in**
  1. A small `prep-wasb`, or a converter to a `ClipDataset` folder:
     * frames re-encoded with `_save`;
     * `detections.jsonl` rows with `persons: []`;
     * `ball_labels.jsonl` rows `{clip, frame, xy, w: 1.0, source: "human"}` on visible frames;
     * `meta.json` with `kind` human, `person_weight` 0 and `rights_status` research_only.
  2. Clips are consecutive frames, so the three-frame input works. The clip frame rate is unverified (a snippet says
     12 fps), so check it against `history_step`.
  3. Hold WASB's test games out as a ball benchmark next to `--ball-folder`.
  4. For referees and the crowd, run the teacher (`prep-teacher`, or the E-BARD RF-DETR Nano) and keep on-court people
     as positives (§6). These are 2018-19 referees: the nearest era to the 2017 failure that any source offers.

### 3.5 SportsHHI (basketball part): rank 5

* **URLs**
  * https://huggingface.co/datasets/MCG-NJU/SportsHHI (gated "auto"; files `frames.zip`,
    `annotations/sports_{train,val}_v1.csv`, `annotations/gt_dense_proposals_{train,val}.pkl`, ...).
  * Code: https://github.com/MCG-NJU/SportsHHI
* **Licence:** CC BY-NC 4.0 (HF tag). The arXiv page lists CC BY-NC-SA 4.0, which is the paper's licence.
* **Contents (paper)**
  * "We carefully selected 80 basketball and 80 volleyball videos from the MultiSports dataset ... average length of
    the videos is 603 frames ... 25FPS ... 720P".
  * Keyframes at 5 fps. The first stage, "person localization and tracking", was done manually "without using
    detection or tracking models".
  * Basketball: **6,791 keyframes, 64,549 human boxes, 9.51 per keyframe**, so every player and not the referees
    (inferred from the count).
  * Whether these are the same 80 basketball videos as SportsMOT's `basketball.txt` (also 80) is **unverified**: the
    files are gated.
* **Size:** 24.0 GB of storage, both sports.
* **Rules:** research-only (NC). YouTube-sourced (MultiSports). **Flagged: HF login and auto-approved terms.** NCAA
  tournament clips are likely included (§1).
* **How to plug it in**
  1. Write per-video MOT `gt.txt` at the keyframes and put the frames in `img1/`. Then run
     `prep-mot ... --sample-every 5`: `prep_mot` makes frames without gt into context rows.
  2. Drop the SportsMOT `val` videos (`v_<id>_c<NNN>`) to keep the NCAA/FIBA held-out test clean.
  3. Run `mark-ignore --referees` on the result.
  4. If the frames zip holds only keyframes, use one-frame clips as `prep-coco` does.

### 3.6 Other candidates (smaller value, or a supporting role)

| Dataset and URL | Licence (where read) | Contents | Size / download | Rules | Use |
|---|---|---|---|---|---|
| **SHOT** (ACM MM 2025) https://huggingface.co/datasets/muyu111/basketball (branch `shotdatasets`) | Annotations CC BY-NC 4.0; "Third-party basketball footage ... excluded" (the `LICENSE` file on that branch) | 1,979 NBA clips "sourced from NBA highlights and full-game replays" (https://xinyi-hu.github.io/SHOT_DATASET/). 5 camera views. Keyframe Pascal-VOC boxes of the 10 players (IDs 1-5 offence, 6-10 defence). MOT-style tracks (tracker output). Matchups include NJ (2012 or earlier) and **GSvsCLE** | 15.5 GB; ungated (`snapshot_download(..., revision="shotdatasets", allow_patterns=...)`) | Research-only (NBA) | Older-era player boxes. Keyframe XML to MOT gt or COCO, then `prep-mot`/`prep-coco`, then `mark-ignore`. Exclude GSvsCLE (§5). |
| **basketball-player-detection-3 v18** (Roboflow, 2026-04-16) mirrored at https://huggingface.co/datasets/koppolusameer/basketball-coco-20260416 | CC BY 4.0 (`README.dataset.txt`; no HF tag) | 654 images, 3 games of the 2025 playoffs: Celtics-Knicks G4 and Celtics-Magic G4 are **new**; Celtics-Knicks G1 is **a repository held-out game**. 6,105 players, 1,788 referees, 608 balls | About 170 MB; ungated, COCO | Research-only | Two more 2025 games of referees. Exclude the G1 clips. |
| **basketball-players-fy4c2 v25** (Roboflow) mirrored at https://huggingface.co/datasets/koppolusameer/basketball-coco-2025-04-02 | Upstream CC BY 4.0 (`README.dataset.txt`); HF tag says MIT (conflict) | Classes `Ball, Hoop, Period, Player, Ref, Shot Clock, Team Name, Team Points, Time Remaining`. 1,196 images but only about 157 unique frames (train augmented up to 20x). NCAA `Iowa_Nebraska-2023-03-06` plus `youtube-N` frames (broadcasts unknown) | About 205 MB; ungated, COCO | Research-only; NCAA question | Small extra referee set. Deduplicate augmented copies first. |
| **Roboflow-only referee sets** (snippet; pages not readable) | CC BY 4.0 per the snippets | `basketball-w376e/basketball-broadcast` (1,853 images, 11 classes incl. Player, Referee, Ball; a version split 3,245/311/148), `tumah/basketball_referee_detection` (1,050 images, v2, 2023), `big-th30a/basketball-project-sv6jj` (479 images, 1,313 referees, 640x360), `computer-vision-and-deep-learning/nba-players-detection` (ball, coach, player, referee, rim; 71 images), `workspace-mvcuz/basketball-courtside-detection` (ball, player, referee, coach), `mohamed-traore-2ekkp/half-court-basketball-kwk11` (NCAA Iowa v Nebraska; head coach, referee, players) | Roboflow account + API key (flag) | Research-only; sources and eras unverified | Only after checking each page. The `coach` classes are the nearest thing to "bench" labels found. |
| **SportsMOT `test` split** (larger SportsMOT, unlabelled) https://huggingface.co/datasets/MCG-NJU/SportsMOT `dataset/test.tar` | CC BY-NC 4.0 | 150 sequences, **50 of them basketball**, frames only (gt withheld) | 23,208,035,328 bytes; ungated; the existing `fetch_data.py sportsmot` streaming handles the split | Research-only; YouTube-sourced; NCAA question | Teacher plus on-court people. Unofficial unpacked copies (`Hellfire98/sportsmot`, `Lekim89/sportsmot`, about 36 GB) also lack test gt. SportsMOT bars redistribution, so use the official tar. |
| **SportsGrounding** https://huggingface.co/datasets/MCG-NJU/SportsGrounding | CC BY-NC 4.0 | 520 MultiSports basketball clips (374/146), about 20 s each, 720p. Boxes only for 4,243 caption targets | `train.zip` 10.6 GB, `val.zip` 4.1 GB; ungated | Research-only; YouTube-sourced; overlaps SportsMOT (its example is `v_00HRwkvvjtQ_c001`, a SportsMOT `val` video) | Unlabelled broadcast footage for the teacher. Drop SportsMOT ids first. |
| **SportsShot** https://huggingface.co/datasets/MCG-NJU/SportsShot | CC BY-NC 4.0 (HF tag); "no distribution without prior written permission" (snippet, CodaLab) | 1,200 YouTube videos of volleyball, basketball and football; 7 shot classes incl. **"audience"** (snippet, project page) | 183 GB total; gated (contact info) (flag) | Research-only; YouTube-sourced | Only as a crowd-shot mining source (frames with no player targets). No boxes. |
| **E-BARD-ObjectClassification** https://huggingface.co/datasets/GabrieleGiudici/E-BARD-ObjectClassification | CC BY 4.0 | 22,210 entities (basketball, hoop, player, referee) | About 165 MB | Research-only | Training and testing the referee check. |
| **UniqueData basketball_tracking** https://huggingface.co/datasets/UniqueData/basketball_tracking / Kaggle `trainingdatapro/basketball-tracking-dataset` | CC BY-NC-ND 4.0 | 70 screenshots with the ball boxed (vendor sample) | 191 MB / 210 MB | Research-only | Too small. |
| **TeamTrack** https://www.kaggle.com/datasets/atomscott/teamtrack | MIT (Kaggle listing) | University games: basketball fisheye 6K 17.5 min (346,500 boxes), drone side and top 4K 24 min (475,200 each). Players | 17.7 GB; Kaggle account | Amateur, not broadcast | Low. |
| **TrackID3x3** https://github.com/open-starlab/TrackID3x3 | No LICENSE file found (unverified) | 3x3; indoor and outdoor fixed cameras, drone | Google Drive | Amateur | Low. |

### 3.7 Checked and not recommended (access or rules)

* **FineSports** (CVPR 2024; 10,000 NBA videos, 123K spatio-temporal boxes): "please sign the Release Agreement and
  send it" by e-mail (https://github.com/PKU-ICST-MIPL/FineSports_CVPR2024). **Signed agreement required (flag).**
* **NSVA** (ECCV 2022; CC BY-NC per https://github.com/jackwu502/NSVA): its raw videos are fetched from NBA.com with
  the authors' download scripts. That is scraping, so **not allowed**.
* **PL-NBA**: CC BY-NC 4.0; the README says "40 official NBA games (2022–2025 season)". Annotations are on GitHub;
  videos come via Baidu Netdisk or "download the original NBA game videos on your own"
  (https://github.com/holhouse/PL-NBA-Dataset). The second route is not allowed; Baidu needs an account. It adds no
  new era.
* **NBA_Streaming**: videos by request with a non-redistribution agreement (already in the training note). Flag.
* **BasketLiDAR** (arXiv 2508.15299): "available upon request" (snippet). Fixed multi-view cameras; licence unverified.
* **Basketball-51**: Kaggle lists **Apache 2.0** (https://www.kaggle.com/datasets/sarbagyashakya/basketball-51-dataset);
  the training note had it unverified. 10,311 clips at 320x240: too small for detection.
* **FineVideo** (HF `HuggingFaceFV/finevideo`): gated, YouTube-sourced, licence tag "cc", 1.35 TB. Basketball content
  unknown.
* **Zenodo record 14021215** (https://doi.org/10.5281/zenodo.14021215, CC BY 4.0 as declared): two `.mov` files of
  about 500 MB, "raw video data from the Olympic competition" (3x3: Serbia-Latvia, Poland-Japan). The uploader's licence
  is unlikely to cover Olympic broadcast rights; unverified.
* **Zenodo UVY / MUVY / MUVS** (10.5281/zenodo.21303900, .13883315, .20708683; CC BY 4.0): user-generated phone video,
  partly YouTube-sourced. Not broadcast.
* **MultiSports / SportsAction** (HF `MCG-NJU/SportsAction`, gated): boxes only for the people performing actions,
  not every player.
* **Wikimedia Commons** "Videos of basketball": freely licensed, but the API answered 429, so the size is unverified.
* **Korea AI-Hub**: no KBL detection set found.
* **infactory-ai/referee-tracking** (HF): soccer, not basketball.

## 4. Licence discrepancies against the repository's notes

* **DeepSport.** The Kaggle Instants set says CC BY-NC-SA 4.0. The v1 set and the challenge README say CC BY-NC-ND
  4.0. `docs/research/training_data.md` has CC BY-NC-ND only. Both are non-commercial.
* **TeamTrack.** The Kaggle listing says MIT; the training note says "unverified; website CC BY-SA 4.0".
* **Basketball-51.** Kaggle says Apache 2.0; the training note says unverified.
* **`koppolusameer/basketball-coco-2025-04-02`.** The HF tag says MIT, but the Roboflow export inside says CC BY 4.0.
  Follow the upstream licence.
* **ASAS v7.** The Roboflow export says MIT; the Kaggle mirror says "Unknown".

## 5. Contamination of the repository's held-out tests

* **E-BARD.** Train only on the 30 `bard.calib_games`; the 30 `check_games` frames are a test set.
* **SportsHHI and SportsGrounding** share MultiSports videos with SportsMOT. Drop SportsMOT `val` ids before training.
* **basketball-player-detection-3 v18** contains `boston-celtics-new-york-knicks-game-1`, a held-out game in
  `roboflow_coco.heldout_games`.
* **SHOT** has a `GSvsCLE` sample (`view5/Shooting/GSvsCLE-46-view5-3`), and other views may hold more. The README's
  `cle_gsw_2017_06_12` test is the 2017 Finals Game 5, so drop every GSW-CLE sample. Do the same for any `GSW_CLE` clip
  in the ASAS set.
* **`yampolal/basketball-data`** has undocumented sources and may hold Roboflow held-out games.

## 6. Suggested order of work

1. **E-BARD first.** Fetch it, rename the ball class, and split by BARD game. Run `prep-coco --referees` on the calib
   games and build the 30-game held-out referee/crowd test from the check games. Refit the referee check on the calib
   games.
2. **DeepSport second.** Get it with a Kaggle account. Map `human` to people, ignore off-court humans using the
   calibration, and hold out by arena. Measure the 2017 game (§10 of the research note) before and after.
3. **Check the ASAS mirror** on a sample before using it: labelling policy, unique frames, era.
4. **Request NUST-NBA181**, pairing it with the WASB ball labels.
   * Ball: a converter to `ball_labels.jsonl`, plus a 31,104-frame ball test.
   * Referees and crowd: run the teacher on 2018-19 frames. Keep on-court people (players and referees) as positives
     and the crowd as background, with `court-people`. Where no camera track exists (short clips), use a variant of
     `mark-ignore` that *promotes* the on-court, non-player boxes the referee check calls referees, instead of ignoring
     them. This is the most direct public route to the 2017-era referee kit.
5. **SportsHHI** (gated) to widen player labels across broadcasts with `mark-ignore`. SHOT can be added for older NBA
   eras, without its GSvsCLE samples.

All of these stay `research_only`. Nothing here clears the commercial release gate of §1 rule 5.
