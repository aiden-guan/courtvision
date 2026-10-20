# CourtVision CV pipeline: real-time player, ball and rim tracking (fixed coaching camera and broadcast)

Researched 2026-10-20 for Python 3.12 on an Apple M3 Pro (18 GB unified memory, MPS, no CUDA).
Primary sources were read where they could be fetched. Roboflow Universe pages returned HTTP 403, so
dataset facts come from Roboflow's blog posts, docs and code. Claims I could not confirm against a
primary source are marked **(unverified)**. Claims that come only from a search-engine snippet,
because the page itself could not be fetched, are marked **(snippet)**. The web-search budget ran
out near the end of the session. The last few facts were checked only by fetching known URLs.

"Jev" in this note always means **TypeSafe AI's Jev**, the typed System One decision model (Choice,
Score, Noul and Box questions with calibrated probabilities). Its exact API is in
`docs/research/jev_api.md`. This note covers only where the CV layer hands narrow judgments to Jev.

---

## 0. TL;DR: decisions

| Decision | Recommendation | Why |
|---|---|---|
| Detector | **RF-DETR Small or Medium** (`rfdetr==1.11.2`, Apache-2.0), fine-tuned on our classes. Export to CoreML FP16 | Apache-2.0 for Nano to Large. Small: 53.0 COCO AP50:95 at 3.5 ms on a T4. Medium: 54.7 at 4.4 ms [13][14]. Roboflow's own basketball pipeline uses RF-DETR-S [1]. **On an M3 Pro, RF-DETR's own CoreML docs measure even `RFDETRNano` (384 px, fp16) at about 21 ms p50 per image [17]**, so S/M at full rate will not fit 33 ms with a ball-ROI pass on top; plan for 15 Hz detection or Nano (§9) |
| Alternative detector | **YOLO26s/m** (`ultralytics==8.4.183`), only with an Ultralytics Enterprise licence or if we open-source | AGPL-3.0, and §13 covers network use [10][11]. Strongest CoreML tooling, with an NMS-free head and STAL small-target assignment [8][59] |
| Tracker | **`trackers==2.6.1`** (Apache-2.0). `ByteTrackTracker` for the fixed camera; `BoTSORTTracker(enable_cmc=True)` for broadcast | `sv.ByteTrack` is **removed in supervision 0.31.0** (released 2026-10-16) [23][22]. BoT-SORT is the best non-SAM tracker on SportsMOT, at 73.8 HOTA [27] |
| Basketball association | Add Basketball-SORT ideas on top: associate in **court coordinates**, cap the roster at 10 players, reacquire long-lost IDs by ReID plus position | HOTA rose from 59.70 (BoT-SORT) to 63.35 and ID switches fell from 23.2 to 9.3 on a fixed basketball camera [32] |
| SAM2 / SAM 3 | **Not in the live loop.** Use them offline for auto-labelling, distillation and post-game re-tracking | Roboflow's SAM2 pipeline runs at about 1 to 2 FPS on a T4 [1]. SAM 3 needs CUDA 12.6+ and uses the restrictive SAM License [38] |
| Team assignment | `google/siglip-base-patch16-224` embeddings, then UMAP to 3 dimensions, then KMeans with 2 clusters, fitted at warm-up, followed by per-track voting. An HSV histogram is the fallback | This is the exact roboflow/sports `TeamClassifier` [3][1] |
| Jersey numbers | Legibility gate, then torso crop, then a CNN classifier (00 to 99 plus "illegible"), then tracklet-level confidence-weighted voting, then roster constraint | ResNet reached 93% and fine-tuned SmolVLM2 86% on Roboflow's NBA crops [1]. The Koshkina pipeline reached 87.45% on SoccerNet tracklets [40] |
| Court registration | Fixed camera: calibrate once (≥8 landmarks, RANSAC, undistort first), then check for drift every few seconds. Broadcast: run court keypoints every frame (33 landmarks, RF-DETR keypoint), then RANSAC, then temporal gating, then CMC fallback | [16][47][32] |
| Ball | Detector plus a dedicated **ball-only** constant-acceleration Kalman filter, with ROI crops at native resolution around the predicted position. Fit a parabola for shots | The ball is about 4 to 5 px wide after downscaling a wide 1080p view to 512 to 640 px (§6.1) |
| Make/miss | Fuse cues: the `ball-in-basket` class, an above-then-below rim-plane crossing inside the rim x-span, net-ROI motion, and drop speed after the net. Send ambiguous cases to Jev as a `Choice` | Roboflow's shot evaluator uses the drop-speed rule [50]. 2D views confuse front-rim and back-rim [50][54] |
| Pose | On demand only, around shots and possession: YOLO26n/s-pose or RTMPose-m via `rtmlib` | Release point, elbow and knee angles, feet relative to lines [57][56][49] |

---

## 1. The Roboflow "Basketball AI" pipeline (Sep 30 2025), read in full

Source: Piotr Skalski, "How to Detect, Track, and Identify Basketball Players with Computer
Vision", Roboflow blog, 30 Sep 2025 [1]. The old `blog.roboflow.com/identify-basketball-players/`
URL now 301-redirects to `roboflow.com/blog/identify-basketball-players`. The code lives in the Colab
notebook `basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb` in
`roboflow/notebooks`. That notebook is 45.3 MB and GitHub will not render it, so I could not read its
cells. Model ids and thresholds below come from the blog, the roboflow/sports source code, and a
third-party mirror of the post [7].

### 1.1 Components and every number stated

1. **Detection: RF-DETR-S**, fine-tuned for free on the Roboflow platform. The blog chose it for
   the best speed/accuracy balance and says it "outperforms larger models like YOLO11-L while
   running faster", without giving numbers or naming a benchmark [1].
   The current tables put RF-DETR-S at 53.0 AP50:95 / 3.5 ms [13] and YOLO11l at 53.4 / 6.2 ms [9].
   That makes RF-DETR-S faster at about equal accuracy, not strictly more accurate.
   - **Dataset:** `roboflow-jvuqo/basketball-player-detection-3-ycjdo`. The blog lists **10
     classes**: `ball`, `ball-in-basket`, `number`, `player`, `player-in-possession`,
     `player-jump-shot`, `player-layup-dunk`, `player-shot-block`, `referee`, `rim` [1]. A mirror
     links dataset **version 6** [7] (secondary; the mirror page was fetched and shows `/dataset/6`, and it names the tracker as "SAM2.1").
   - Player identity uses `number` plus the five `player*` classes. The other classes are kept for
     shot and made-basket analysis [1].
   - The blog gives no mAP, epoch count, image count or input resolution [1].
2. **Tracking: SAM2.** RF-DETR boxes from the first frame prompt SAM2. The pipeline assumes every
   player is visible in that frame. SAM2's memory bank re-identifies players after occlusion.
   Longer clips would need a component that watches detections and re-prompts SAM2. The checkpoint
   is not named in the blog. The mirror says SAM2.1 [7]. A mask clean-up step keeps the largest
   connected component and drops small, distant fragments [1]. supervision later shipped this
   as `sv.filter_segments_by_distance` in 0.27.0 [23].
3. **Team clustering:** one frame per second is sampled from a game. Players are cropped and only
   the **central part** of each box is kept, to cut out crowd and opponents. Each crop is
   embedded with SigLIP, reduced with UMAP to **3 dimensions**, and clustered with **K-means (k=2)**.
   The method is reused from Football AI [1]. The roboflow/sports implementation pins
   `SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'` and uses `SiglipVisionModel` and
   `AutoProcessor`. It mean-pools `last_hidden_state`, with `batch_size=32`,
   `umap.UMAP(n_components=3)` and `KMeans(n_clusters=2)`. `fit()` runs once and `predict()` uses
   `UMAP.transform` [3]. The basketball notebook probably uses the same class (unverified).
4. **Jersey numbers.**
   - **SmolVLM2**, used as OCR. Out of the box it scored **56%** test accuracy and produced
     impossible strings such as "011" and "3000". After fine-tuning on **3.6k** jersey crops from
     the **2025 NBA Playoffs** it reached **86%**. The labels were auto-annotated, then hand-fixed.
     The dataset is incomplete (not every number from 00 to 99 appears) and imbalanced: "8" appears
     315 times (8.7%) and "6" appears 8 times (0.2%) [1]. The OCR dataset is
     `roboflow-jvuqo/basketball-jersey-numbers-ocr`, version 3 [2][7]. The SmolVLM2 size and the
     LoRA settings are not stated (unverified).
   - **ResNet-32 classifier**, trained on the same data reformatted as a folder-per-class
     classification dataset. It reached **93%**, beating the VLM [1].
   - **Temporal confirmation:** numbers are read **every 5 frames**. A number is confirmed after
     **3 identical predictions in a row**, then mapped to a roster name via (number, team) [1].
5. **Player-to-number matching:** **IoS** (intersection over the smaller area) between the SAM2
   player mask and the number box rasterized to a mask. A pair is kept only when **IoS ≥ 0.9** [1].
6. **Throughput:** about **1 to 2 FPS on an NVIDIA T4**, which is not real-time. SAM2 is the
   bottleneck and slows down as the number of tracked objects grows (the blog does not state how
   many objects were tracked for the 1 to 2 FPS figure). Suggested fixes are lower
   resolution, a smaller SAM2 checkpoint, or distilling SAM2 into a lighter model [1].
7. **Stated limitations:** short clips only, with every player visible in frame 1. Numbers are weak
   for players far from the camera (for example a corner spot-up shooter). The VLM sometimes
   produces impossible numbers [1].

### 1.2 Related Roboflow basketball material

- **3-second violation detector** (22 Jul 2025) [49]:
  - Court keypoints from Universe `basketball-court-detection-2`, model and dataset **v13**,
    described as a "YOLOv11 keypoint model".
  - A homography rebuilds the paint polygon and reprojects missing court keypoints.
  - Players come from a `basketball-player-detection-2` model, v8 (a "YOLOv11 object detector";
    the dataset version is not stated), tracked by SAM2, with
    `remove_noisy_segments` mask clean-up.
  - **YOLO11-pose ankles** on player crops. The fallback is the bbox bottom-center.
  - `sv.PolygonZone(...).trigger` plus an `FPSBasedTimer` example utility, with a 3 s threshold.
  - Lesson learned: mask-based foot detection "was not robust", which is why the post moved to pose.
  - Licensing note from the post: YOLO11 is AGPL-3.0. Roboflow cloud includes a commercial
    licence; self-hosting needs an Enterprise licence or AGPL compliance [49].
- **Court keypoint dataset:** `roboflow-jvuqo/basketball-court-detection-2` has **33 court
  landmarks** on a single `court` class, marked visible or occluded, with off-screen points skipped.
  The RF-DETR keypoint cookbook uses **v19** [16].
- **AI shot evaluator** (14 Aug 2026) [50]:
  - RF-DETR-small detects ball and rim, trained on a fork of the University of Arizona "Basketball
    Shooting Robot" dataset (about 9.8k images): **mAP50 88.84, P 75.7, R 84.9**. The ball was
    found in **98.1%** of in-flight frames (759 of 774).
  - Pose: `RFDETRKeypointPreview` (rfdetr 1.9.2), zero-shot COCO-17 keypoints, found on 99% of frames.
  - The make/miss logic is in §6.4. No make/miss accuracy is reported.
  - Repo: `aarnavshah12/shot-tracker` [50].
- `roboflow/notebooks` also lists `basketball-ai-make-or-miss-jumpshot-detection.ipynb` and
  `basketball-ai-automatic-detection-of-3-second-violations.ipynb`. I could not read their
  contents (unverified).
- **roboflow/sports** (MIT; install with `pip install git+https://github.com/roboflow/sports.git`;
  no PyPI package) [2]. It names the open problems we will also face: fast and small ball,
  unreadable numbers, occlusion, re-identification, and camera calibration.

---

## 2. Detection

### 2.1 Classes CourtVision needs

| Class | Needed for | Notes |
|---|---|---|
| `player` | Everything | Keep a single class. Get team from the embedding, not from the detector |
| `referee` | Excluding refs from team, stats and possession | In Roboflow's 10-class set [1]. Practice footage may have none |
| `ball` | Possession, passes, shots | The hardest class. Usually needs a dedicated ball path (§6) |
| `rim` | Shot attempts, make/miss, pixels-per-metre (rim diameter is 18 in = 0.4572 m) [50] | Static in a fixed camera, so detect it once and refresh rarely |
| `number` | Jersey OCR crops, matched to players by IoS ≥ 0.9 [1] | Optional. A torso crop from pose (§5) works without this class |
| `ball-in-basket` | Make cue | In Roboflow's set [1]. Treat it as one vote, not the truth |
| `player-in-possession`, `player-jump-shot`, `player-layup-dunk`, `player-shot-block` | Optional priors for the event engine | Single-frame action labels are noisy. The event engine should decide from tracks over time |
| (optional) `backboard`, `net`, `scorebug` | Net-motion ROI, broadcast clock and score OCR | Not in the Roboflow set (unverified that any public set has them) |

### 2.2 Model options (verified numbers)

**RF-DETR** (Roboflow; ICLR 2026 paper arXiv 2511.09554; `pip install rfdetr`, latest **1.11.2**
from 2026-10-04, Python ≥ 3.10) [13][14]. Latency is T4, TensorRT FP16, batch 1 [13].

| Class | COCO AP50:95 | AP50 | Latency (ms) | Params (M) | Resolution | Licence |
|---|---|---|---|---|---|---|
| `RFDETRNano` | 48.4 | 67.6 | 2.3 | 30.5 | 384 | Apache-2.0 |
| `RFDETRSmall` | 53.0 | 72.1 | 3.5 | 32.1 | 512 | Apache-2.0 |
| `RFDETRMedium` | 54.7 | 73.6 | 4.4 | 33.7 | 576 | Apache-2.0 |
| `RFDETRLarge` | 56.5 | 75.1 | 6.8 | 33.9 | 704 | Apache-2.0 |
| `RFDETRXLarge` / `RFDETR2XLarge` | 58.6 / 60.1 | 77.4 / 78.5 | 11.5 / 17.2 | 126.4 / 126.9 | 700 / 880 | **PML 1.0** (`rfdetr[plus]`) |
| `RFDETRAtto`/`Femto`/`Pico` | 30.5 / 37.8 / 41.6 | | 1.0 / 1.4 / 1.7 | 7.4–8.4 | 380–560 | **PML 1.0** |

- Segmentation (`RFDETRSegNano` through `RFDETRSeg2XLarge`) and keypoints (`RFDETRKeypointPreview`:
  71.8 COCO AP, 9.7 ms, 40.7M params, 576×576) are all Apache-2.0 [13].
- API: `RFDETRMedium().predict(img, threshold=0.5)` returns `sv.Detections` directly. Class names
  are in `detections.data["class_name"]` for fine-tuned models. The alternative is
  `inference.get_model("rfdetr-medium")` followed by `sv.Detections.from_inference(...)` [13].
- RF100-VL (domain transfer) AP50:95: Small 60.2, Medium 61.2. The DINOv2 backbone is why it
  fine-tunes well on small domain datasets [13] (Atto, Femto and Pico use a PE-Core-T backbone instead [13]).
- **Apple:**
  - The PyPI page lists `coreml` and `coreai` extras [14].
  - The CoreML export guide (`pip install "rfdetr[coreml]"`; the extra pins
    `coremltools>=8.0,<10.0` [74]) exports a native `.mlpackage` via `model.export(format="coreml")`.
    It is marked **experimental**. The default is FP32; `coreml_precision="float16"` gives a
    half-size bundle that Core ML can schedule on the ANE (an FP32 bundle never reaches the ANE).
    FP16 kept COCO mAP identical for `RFDETRNano`, but raw tensors can differ, so validate
    post-processed detections. `dynamic_batch=True` is not supported, so we need one package per
    batch size. Compute units are chosen at load time (`MLModel` defaults to `ComputeUnit.ALL`) [17].
  - **Measured on an Apple M3 Pro** (pretrained `RFDETRNano`, 384×384, batch 1, p50, coremltools
    9.0): FP16 20.8 ms with `CPU_AND_NE`, 21.2 ms with `ALL`, 33.9 ms with `CPU_AND_GPU`, 44.5 ms
    CPU-only; FP32 about 34 ms on the GPU. FP16 on the ANE is "about 2.1x faster than CPU" [17].
    The develop-branch docs add an unreleased `coreml_neural_engine=True` flag and M4 Max numbers
    for Small/Medium (about 23 to 24 ms on the GPU) [17].
  - Eager MPS reports that `aten::_upsample_bicubic2d_aa.out` falls back to CPU [18]. **Prefer
    CoreML over eager MPS for RF-DETR.**

**Ultralytics YOLO26** (released January 2026; `ultralytics` **8.4.183**, 2026-10-20; YOLO27 is an
unreleased preview: "no launch date has been set") [8][10][69]. Numbers are COCO at 640 px [8].

| Model | mAP50-95 | e2e (NMS-free) | CPU ONNX ms | T4 TRT ms | Params M |
|---|---|---|---|---|---|
| YOLO26n | 40.9 | 40.1 | 38.9 | 1.7 | 2.4 |
| YOLO26s | 48.6 | 47.8 | 87.2 | 2.5 | 9.5 |
| YOLO26m | 53.1 | 52.5 | 220.0 | 4.7 | 20.4 |
| YOLO26l | 55.0 | 54.4 | 286.2 | 6.2 | 24.8 |
| YOLO26x | 57.5 | 56.9 | 525.8 | 11.8 | 55.7 |

- Changes from earlier YOLOs: an optional one-to-one **NMS-free** head, **DFL removed**, ProgLoss,
  **STAL** (Small-Target-Aware Label Assignment), and the MuSGD optimizer.
- P2 (small-object) heads exist only as YAML, with **no pretrained weights** [8].
- For comparison, YOLO11 (27 Sep 2024): n 39.5, s 47.0, m 51.5, l 53.4, x 54.7 [9].
- **Licence:**
  - AGPL-3.0 or an Enterprise licence [8][10].
  - AGPL §13 requires a modified version to offer its source to "all users interacting with it
    remotely through a computer network" [11]. Ultralytics says AGPL use requires that you
    "Open-source your entire project under AGPL-3.0", lists SaaS, API and cloud systems that use
    YOLO as needing an Enterprise licence, and sells that licence to remove the obligation [12].
  - Roboflow resells Ultralytics licences for models run on its platform [49].
  - **Implication:** importing `ultralytics` anywhere in a closed CourtVision service, including
    for training, is a licence decision to make before writing code. RF-DETR (Apache-2.0) avoids it.

**D-FINE** (Apache-2.0). COCO AP / T4 TRT-FP16 latency: N 42.8 / 2.12 ms, S 48.5 / 3.49 ms,
M 52.3 / 5.62 ms, L 54.0 / 8.07 ms, X 55.8 / 12.89 ms. With Objects365 pretraining: S 50.7,
M 55.1, L 57.3, X 59.3 [19]. The repo warns that Objects365-pretrained checkpoints "may be subject
to Objects365 dataset terms" [19]. DEIMv2 (Sep 2025, DINOv3 features) reports S 50.9 AP (9.71M
params) up to X 57.8 AP (50.3M params) in its abstract [20]. No latency GPU is stated. The code
licence is unverified: GitHub reports `NOASSERTION` for `Intellindust-AI-Lab/DEIMv2` (unverified).

**Verdict:**
- RF-DETR-S/M is the default. It is Apache-2.0, the backbone that fine-tunes best on small sports
  datasets, a basketball-tested recipe [1][50], and it returns `sv.Detections` natively.
- Keep YOLO26 as the benchmark challenger if CoreML throughput on the M3 Pro turns out to be the
  binding constraint and the licence is resolved.

### 2.3 Datasets to start from

| Dataset | Content | Source |
|---|---|---|
| `roboflow-jvuqo/basketball-player-detection-3-ycjdo` (v6) | 10 classes (above), NBA broadcast | [1][7] |
| `roboflow-jvuqo/basketball-court-detection-2` (v19) | 33 court keypoints | [16][2] |
| `roboflow-jvuqo/basketball-jersey-numbers-ocr` (v3) | About 3.6k number crops, 2025 NBA Playoffs, imbalanced | [1][2][7] |
| `basketball-player-detection-2` (model v8; dataset version not stated) | Player detector used in the 3-second post | [49] |
| U. Arizona "Basketball Shooting Robot" (fork) | About 9.8k images. Ball and rim used | [50] |
| TeamTrack basketball | Fixed, side-view fisheye, with tracking GT (162k train / 170k test frames used by Basketball-SORT) | [32][64] |
| SportsMOT | 240 videos and 150K+ frames covering basketball, volleyball and football. MOT GT | [33] |
| DeepSport / DeepSportRadar "Basketball Instants" | Calibrated images with 3D ball GT. The calibration challenge used 480/164/84/84 images | [47][52] |
| TrackID3x3 | 3x3 basketball, indoor/outdoor fixed and drone. Boxes, IDs, 10-keypoint pose | [36] |

Universe licences could not be checked (403). **Confirm each dataset's licence before any
commercial training run (unverified).** NBA broadcast frames carry their own rights questions.

---

## 3. Multi-object tracking

### 3.1 Library API as of Oct 2026 (this changed recently)

- **`supervision`** (MIT; latest **0.31.0**, 2026-10-16; Python ≥ 3.10) [22]:
  - The 0.31.0 changelog **removes `sv.ByteTrack`**, the `supervision.keypoint` module (use
    `supervision.key_points`), `sv.LMM` / `Detections.from_lmm` (use `sv.VLM` / `from_vlm`) and
    the legacy `MeanAveragePrecision` [23].
  - Many tutorials, including roboflow/sports' soccer example, still call
    `sv.ByteTrack(minimum_consecutive_frames=3)` [5]. **Pin versions or port.**
  - The old signature, for reference only:
    `sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=30, minimum_consecutive_frames=1)`,
    with `update_with_detections(detections)`, `update_with_tensors(...)` and `reset()`. It was
    deprecated in 0.28.0 [21].
- **`trackers`** (Roboflow; Apache-2.0; latest **2.6.1**, 2026-09-25; Python ≥ 3.10) [25].
  These are clean-room re-implementations. Every tracker exposes
  `update(detections, frame=None, timestamp=None) -> sv.Detections` (with `tracker_id`) and
  `reset()` [26]:
  - `SORTTracker(lost_track_buffer=30, frame_rate=30.0, track_activation_threshold=0.25, minimum_consecutive_frames=3, minimum_iou_threshold=0.3, ...)`
  - `ByteTrackTracker(lost_track_buffer=30, frame_rate=30.0, track_activation_threshold=0.7, minimum_consecutive_frames=2, minimum_iou_threshold=0.1, high_conf_det_threshold=0.6, ...)`
  - `OCSORTTracker(..., direction_consistency_weight=0.2, high_conf_det_threshold=0.6, delta_t=3, ...)`
  - `BoTSORTTracker(..., enable_cmc=True, cmc_method='sparseOptFlow', cmc_downscale=2, ...)`.
    The CMC methods are `"orb"`, `"sift"`, `"sparseOptFlow"` and `"ecc"`. CMC is skipped silently
    (no warning) when `frame=None`. BoT-SORT expects **BGR** frames of shape (H, W, 3) [26].
    Other BoT-SORT defaults: `track_activation_threshold=0.7`, `minimum_consecutive_frames=2`,
    `minimum_iou_threshold_first_assoc=0.2`, `..._second_assoc=0.5`, `..._unconfirmed_assoc=0.3`,
    `high_conf_det_threshold=0.6`, `instant_first_frame_activation=True` [26].
  - `CBIoUTracker(...)` (no CMC).
  - `McByteTracker(..., enable_cmc=True, enable_mask_manager=False, ...)`. Its mask pipeline
    (`enable_mask_manager=True`) needs SAM and Cutie, and its frames must be **RGB**. Device
    auto-selection picks CUDA or CPU; MPS is never auto-selected and must be requested [26].
  - Unmatched or immature detections come back with `tracker_id == -1`; filter them before
    the identity layer [26].
  - CLI: `trackers track | eval | download | tune` [25].
- **Ultralytics `model.track()`** has six built-in configs: `botsort.yaml`, `bytetrack.yaml`,
  `ocsort.yaml`, `deepocsort.yaml`, `fasttrack.yaml` and `tracktrack.yaml`. **TrackTrack is the
  default.** The last four need ≥ 8.4.63. ReID (`with_reid: True`, with `yolo26n-reid.onnx`
  through `yolo26x-reid.onnx` encoders) works for BoT-SORT, Deep OC-SORT and TrackTrack. Use
  `persist=True` only for consecutive frames of one stream [28].
- **`sv.Detections`** fields: `xyxy`, `mask`, `confidence`, `class_id`, `tracker_id`, `data`,
  `metadata`.
  - Constructors include `from_ultralytics`, `from_inference`, `from_transformers`, `from_sam`,
    **`from_sam3`** and `from_vlm`.
  - Methods include `get_anchors_coordinates(anchor)`, `with_nms`, `with_soft_nms`, `with_nmm`,
    `merge`, `empty` and `is_empty` [24].
  - Foot point: `detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)`. The enum value
    was not re-read on the docs page, but the soccer example projects bottom-center anchors [5].
- Other `sv` pieces we will use:
  - `sv.InferenceSlicer(..., overlap_wh=..., batch_size=...)`. `batch_size` arrived in 0.30.0;
    the ratio-based overlap was removed in 0.27.0.
  - `sv.PolygonZone(..., require_all_anchors=...)` (0.30.0), and `PolygonZone.trigger(...,
    coord_transform=...)` with the `sv.CoordinatesTransformation` protocol (0.31.0). This lets
    zones such as the paint be defined in court coordinates.
  - `sv.KeyPoints.merge`, `sv.get_video_frames_generator(prefetch=...)` and `sv.process_video`
    [23].
- **`ViewTransformer` is not part of `supervision`.** It is `sports.common.view.ViewTransformer`
  in roboflow/sports [4]:
  - It calls `cv2.findHomography(source, target)` with **no method argument**, so it uses plain
    least squares with **no RANSAC**, and it discards the inlier mask.
  - It provides `transform_points` (via `cv2.perspectiveTransform`) and `transform_image`
    (via `warpPerspective`).
  - In the soccer example it is **rebuilt every frame** from the detected keypoints [5].
  - **We should write our own version that uses RANSAC or MAGSAC and checks reprojection error
    (§5).**

### 3.2 Benchmark numbers

Roboflow `trackers` comparison: YOLOX detections for MOT17, SportsMOT and DanceTrack; oracle
detections for SoccerNet [27].

| Tracker | SportsMOT HOTA / IDF1 / MOTA (default) | SportsMOT tuned HOTA | DanceTrack HOTA |
|---|---|---|---|
| SORT | 70.8 / 68.9 / 95.5 | 72.9 | 47.2 |
| ByteTrack | 73.0 / 72.5 / 96.4 | 73.3 | 53.3 |
| OC-SORT | 71.7 / 71.4 / 95.0 | 74.0 | 54.1 |
| BoT-SORT | 73.8 / 73.4 / 96.9 | 74.1 | 57.8 |
| C-BIoU | 73.1 / 72.6 / 96.7 | 73.1 | 56.7 |
| McByte (SAM + Cutie masks) | 76.5 / 76.9 / 97.0 | — | 67.2 |

- Original papers:
  - ByteTrack: associates low-score boxes too; MOT17 80.3 MOTA, 77.3 IDF1, 63.1 HOTA at 30 FPS
    on a V100 [29].
  - BoT-SORT: adds camera-motion compensation, a better Kalman state and ReID fusion; MOT17 80.5
    MOTA, 80.2 IDF1, 65.0 HOTA [30].
  - Deep OC-SORT: adaptive appearance on top of OC-SORT; MOT17 64.9, MOT20 63.9, DanceTrack 61.3
    HOTA [31].
- **SportsMOT** (basketball, volleyball, football): the paper finds that association, not
  detection, is the main difficulty, because of fast, variable motion and "similar yet
  distinguishable" appearance [33].
- **Basketball-SORT** (arXiv 2406.19655; Hu, Scott, Yeung, Fujii; Jun 2024) [32]:
  - Target: complex multi-object occlusion (CMOO), where three or more similar players overlap.
  - Data: TeamTrack basketball, a fixed side-view **fisheye** camera calibrated in advance.
    Detector: fine-tuned YOLOv8l. ReID: FastReID.
  - **The homography comes from 20 manually marked court keypoints.** The player position is the
    bbox bottom-midpoint projected to the court, and the Kalman filter associates in **court
    coordinates and velocities, not IoU**.
  - **BGR** (basketball game restriction): at frame 100, keep the 10 longest trajectories as
    players and create no new tracks after that. This removes refs and spectators.
  - **RLLI** (reacquire long-lost IDs): tracks lost for more than B frames store their position and
    ReID feature, and are reassigned using a cosine-cost threshold α = 0.2 and distance β = 260 cm.
    **Caveats before hard-coding:** the paper never gives a value for B. Its Eq. 3 is printed as
    `Cost > α` and `Dist > β` (Cost = 1 − cosine similarity), while the text calls α a "similarity"
    threshold, so the inequality direction is ambiguous. We read it as `cost < 0.2` and
    `dist < 2.6 m` (our interpretation, unverified). Section 4.4 also says the distance was tuned
    over 170 to 250 cm, which does not contain 260 [32].
  - Results: **HOTA 63.35, AssA 59.15, IDS 9.3**, versus BoT-SORT 59.70 / 53.43 / 23.2 and
    Deep-EIoU 60.74 / 55.47 / 20.7. FP fell from 492.6 to 118.9 and FN rose from 270.8 to 387.7.
    The abstract says 63.48, which matches an ablation row. No code link was found [32].
- **TrackID3x3** (Mar 2025) [36]:
  - BoT-SORT-ReID with a non-fine-tuned YOLOX scored TI-HOTA **80.75% indoor** but **46.11%
    outdoor**.
  - On drone footage, HOTA was ByteTrack 47.92 versus BoT-SORT-ReID 50.64.
  - The authors identify occlusion-driven ID switches as the main challenge. Colour histograms
    (8×8×8 bins) are used; Jensen-Shannon divergence is mentioned only for merging Indoor
    tracklets. Indoor team roles come from starting positions, and Outdoor uses jersey-number
    recognition plus histogram similarity [36].
- **Sports-vmTracking** (Dec 2024): pose-based virtual markers on 3x3 basketball reached HOTA 72.3%
  with 0 ID switches. The abstract says this is more than 10 points above methods without virtual
  markers [35].
- **SAMIDARE** (Apr 2026), tracking-by-segmentation that extends SAM2MOT: +2.5 HOTA and +4.2 IDF1
  on the SportsMOT validation set. No speed was given [34].

### 3.3 SAM-family trackers

- **SAM 2.1** (Apache-2.0; Python ≥ 3.10, torch ≥ 2.5.1) [37]:
  - Checkpoints: `sam2.1_hiera_t` at 38.9M params and 91.2 FPS, `_s` at 46M and 84.8 FPS, `_b+`
    at 80.8M and 64.1 FPS, and `_l` at 224.4M and 39.5 FPS. Speeds are on an A100; the configs
    are `configs/sam2.1/sam2.1_hiera_*.yaml`.
  - API: `build_sam2_video_predictor(cfg, ckpt, vos_optimized=True)`, `init_state`,
    `add_new_points_or_box`, `propagate_in_video`. Per-object adds after tracking starts have been
    supported since Dec 2024.
  - Install tries to compile an optional CUDA extension (connected-components post-processing);
    the README says a build failure can be ignored and "doesn't affect the results in most cases".
    The README does not mention MPS (unverified on Apple) [37].
  - In practice, Roboflow's basketball pipeline (all players in frame 1) ran at about 1 to 2 FPS on
    a T4 [1].
- **SAM 3** (Nov 2025) and **SAM 3.1** (27 Mar 2026) [38][39]:
  - 848M params. Text and exemplar prompts give every instance of a concept, with IDs.
  - `build_sam3_video_predictor()` with `handle_request()` (`start_session`, `add_prompt`) [38].
  - SAM 3 runs in 30 ms per image with 100+ objects on an H200, and holds near real time for about
    five concurrent video objects. SAM 3.1's **Object Multiplex** tracks up to 16 objects per
    forward pass, raising throughput from 16 to 32 FPS on one H100 for videos with a medium number
    of objects [39]. The repo's SAM 3.1 release notes add "~7x speedup at 128 objects on a single
    H100" versus the Nov 2025 SAM 3 release [72].
  - **Needs Python 3.12+, PyTorch 2.7+ and CUDA 12.6+**, and is under the **SAM License** (not
    OSI). The HF checkpoints `facebook/sam3` and `facebook/sam3.1` are gated [38].
  - "Towards Comprehensive Basketball Understanding" (Aug 2026) tracks players and ball with **SAM
    3 + fine-tuned RF-DETR**, using RF-DETR to filter bad SAM trajectories [46].
- **Implication for an M3 Pro:** SAM 2.1-tiny might be usable offline for keyframe re-segmentation
  (unverified on MPS). SAM 3 is a cloud/CUDA tool. Use either to (a) auto-label tracking GT for
  tuning, (b) re-track post-game for "final" stats, and (c) distill into the live detector.

### 3.4 Identity switches in basketball, and our re-ID strategy

The failure pattern: 10 players in one-colour uniforms, hard contact in the paint, screens and
pick-and-rolls (two bodies merge for 10 to 30 frames), players jumping (the foot point leaves the
floor), and broadcast cuts. Appearance-only ReID is weak inside a team [33][36].

1. **Two-level identity.** `tracker_id` is a short-lived *tracklet*. The *player identity* is
   `(team, jersey_number)`, mapped to the roster. Only the identity layer feeds stats. A tracklet
   is attached to an identity once number votes are confirmed (§4.2). Until then it is
   `team:unknown#k`.
2. **Associate in court coordinates.** Run the Kalman filter on court x/y in metres (Basketball-SORT
   [32]). In the image plane a far player's 1 m step is a few pixels and a near player's is tens.
   Use a gating radius of about 1 to 1.5 m per frame at 30 fps (unverified; tune it). Sanity check
   (our arithmetic): even a 12 m/s sprint is only 0.4 m per frame at 30 fps, so most of a 1 to 1.5 m
   gate would be absorbing foot-point and homography noise. Prefer a Mahalanobis gate on the
   Kalman innovation covariance over a fixed radius.
3. **Roster cardinality constraints:**
   - Coach mode: at most 5 per team on court, plus refs.
   - Broadcast: at most 10 visible players.
   - When a new tracklet starts while a same-team identity is "lost", prefer reassigning that lost
     identity (RLLI-style) over minting a new one [32].
4. **Appearance:** use per-track SigLIP embedding exemplars (already computed for team assignment)
   as the ReID feature, together with the Basketball-SORT α/β gating. Add a dedicated ReID model
   (FastReID, or Ultralytics' ReID ONNX) only if measurements show it is needed [32][28].
5. **Occlusion clusters:** when three or more boxes have pairwise IoU > 0.3 (unverified threshold),
   freeze identity commits. When the cluster breaks up, re-solve the assignment with Hungarian
   matching on court position, team, number and appearance, rather than trusting the tracker.
6. **Broadcast cuts:** `reset()` the tracker on every detected shot boundary. Identities are then
   recovered from (team, number) and court position continuity.

---

## 4. Team assignment and jersey numbers

### 4.1 Team assignment

- **Primary:** the roboflow/sports `TeamClassifier` recipe [3][1]:
  - Embed the central crop of each player box with `google/siglip-base-patch16-224`
    (`SiglipVisionModel`, mean-pooled), reduce with UMAP (`n_components=3`), and cluster with
    `KMeans(n_clusters=2)`.
  - Fit once on warm-up crops. The soccer example samples every 60th frame (`STRIDE=60`) and uses
    players only, excluding keepers and refs [5]. Roboflow's basketball post samples at 1 fps [1].
  - `predict()` per track every about 15 frames, then a majority vote over a sliding window.
  - Team is a **track attribute with hysteresis**, not a per-frame label.
- **Referees:** use the detector's `referee` class [1] and keep refs out of the `fit()` set (as the
  soccer example does for refs and keepers [5]). Basketball-SORT's BGR removes refs and spectators
  structurally [32]. In practice footage there may be coaches on the floor, so keep a third
  "other" bucket. That means k=3 clustering with a rejection threshold on distance to centroid
  (unverified design).
- **Baseline and fallback:** an HSV colour histogram of the torso crop, compared to two reference
  histograms. TrackID3x3 used 8×8×8 colour histograms (with Jensen-Shannon divergence only for
  merging Indoor tracklets) [36].
  This is cheap and works for pinnies in practice. Use it when SigLIP is unavailable or for a
  sanity check.
- **Label permutation:** KMeans cluster 0 or 1 is arbitrary across sessions. Bind clusters to
  "home" and "away" once, from (a) the roster's declared colours, (b) one coach confirmation tap,
  or (c) for broadcast, the scorebug colours. This is a narrow judgment that suits Jev (a `Choice`
  over {home, away} given the colour descriptors, returning a calibrated probability), so the LLM
  never guesses team identity.
- **Apple:** `SiglipVisionModel` on `device="mps"` should work (unverified). The batch of 32 crops
  at 224 px is small. Run it in an async worker, not per frame.

### 4.2 Jersey number recognition

Reported accuracies:

| System | Data | Accuracy |
|---|---|---|
| SmolVLM2 base (zero-shot) | Roboflow NBA crops | 56% [1] |
| SmolVLM2 fine-tuned | 3.6k NBA Playoff crops | 86% [1] |
| ResNet-32 classifier | Same | **93%** [1] |
| Koshkina & Elder (CVPR-W 2024): legibility ResNet34, then ViTPose torso crop, then PARSeq, then tracklet aggregation | SoccerNet tracklets | **87.45% test / 79.31% challenge**. Hockey images 91.4% [40] |
| Legibility classifier (ResNet34, SAM optimizer, 20 epochs) | Hockey | 94.5% acc, 71.7% F1 [40] |
| SoccerNet 2023 challenge best | SoccerNet | 92.85% [45] (snippet; not in the arXiv abstract, unverified) |

Koshkina pipeline details [40]:
- Torso crop from shoulder and hip keypoints, padded 5 px.
- PARSeq fine-tuned with a maximum label length of 2.
- Outlier frames removed with a Centroid-ReID Gaussian (3.5σ, 3 iterations).
- Best aggregation: a **confidence-weighted majority vote** with an illegibility threshold and
  down-weighting of 1-digit votes when 2-digit votes exist. Only 39% of numbers in their data are single-digit.
- Cross-domain lesson: soccer-tuned PARSeq scored only 65.84% on hockey, versus 91.40% when tuned
  on hockey. **Fine-tune on basketball crops.**
- Code: `mkoshkina/jersey-number-pipeline` [40][41]. Published at the CVPR 2024 Workshops
  (CVsports) per the repo README [41]. **Licence: the repo's code, data and weights are under
  CC BY-NC 3.0 (non-commercial)** [41]. Re-implement the method from the paper; do not ship that
  code or those weights in a commercial CourtVision.

Other references:
- PARSeq: `torch.hub.load('baudm/parseq', 'parseq', pretrained=True)` (the README appends
  `.eval()`); the repo licence is Apache-2.0 per GitHub [44].
- SmolVLM2 ids: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
  and `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` (all three exist on the Hub, ungated,
  Apache-2.0). MLX builds exist, for example `mlx-community/SmolVLM2-500M-Video-Instruct-mlx` [42].
- The basketball paper from Aug 2026 used **Qwen3.5-9B** on **8 uniformly sampled frames per
  track**, outputting colour, number and confidence, and pairing colour with number to separate
  same-number opponents [46]. That suits offline use. It is too heavy for per-frame use on an M3 Pro.

**CourtVision design:**
1. **Gate:** read a track only when (a) bbox height ≥ about 120 px at native resolution
   (unverified threshold), (b) pose shows the torso facing the camera (shoulder width over torso
   height above a threshold), and (c) motion blur is low (Laplacian variance). Optionally add a
   ResNet34-style legibility classifier [40].
2. **Crop:** the torso from pose shoulders and hips [40], or the `number` box matched by IoS ≥ 0.9 [1].
3. **Recognise:** a small CNN classifier with 101 classes (00 to 99 plus illegible) exported to
   CoreML, as the primary [1]. Use PARSeq as the secondary for numbers missing from training (a
   classifier cannot produce an unseen class) [40][44]. Avoid VLM OCR in the hot path, because it
   produces impossible strings [1].
4. **Vote:** sample every 5 frames per track [1]. Accumulate confidence-weighted votes per tracklet
   [40]. Confirm when the top number leads with ≥ 3 agreeing reads (Roboflow's rule [1]) **and**
   the number exists on that team's roster. Partial reads ("2" of "23") count only toward
   two-digit candidates that are consistent with them (the Koshkina bias [40]).
5. **Constrain:** each roster number belongs to at most one live tracklet per team. Solve conflicts
   with Hungarian assignment over vote mass. Report "unknown #k" rather than guess.
6. **Practice footage:** pinnies usually have no numbers. Identity then comes from a one-time coach
   tag plus ReID, and number OCR is disabled.

---

## 5. Court registration and homography

### 5.1 Models and methods

- **Roboflow court keypoints:**
  - 33 landmarks (`roboflow-jvuqo/basketball-court-detection-2`, v19) [16].
  - The RF-DETR keypoint fine-tune recipe uses `RFDETRKeypointPreview`, 50 epochs, batch size 2,
    gradient accumulation 2, lr 1e-4, resolution 576. The best weights save to
    `checkpoint_best_total.pth`; model selection uses `val/keypoint_map_50_95`.
  - Load with `RFDETRKeypointPreview.from_checkpoint(...)`, passing the same `num_classes` and
    `num_keypoints_per_class`. Run `.predict(img, threshold=0.5).with_nms()` [16].
  - The 3-second post used a YOLO11 keypoint model on v13 [49].
- **KaliCalib** (MMSports 2022 calibration challenge) [47]:
  - 91 court keypoints. A U-Net with a ResNet-18 encoder (14.5M params, about 9 ms on a GTX 1080
    Ti) outputs heatmaps. Keypoints are heatmap centres of mass.
  - **DLT + RANSAC, with a reprojection threshold of 35 px.** OpenCV `calibrateCamera` turns the
    homography into a full camera. If the result is degenerate, the mean training homography is
    used.
  - Test MSE 126.61 cm, versus a 592.48 baseline (4.7 times better). 76.5% of images are under 1 m.
  - Fails when only a small part of the court is visible, or when lines from other sports are
    painted on the floor.
- **Central or zoomed views with few keypoints:** points and lines can be derived from circle
   correspondences, which complements keypoint detectors [48]. PnLCalib (points plus lines) is the
   general-field reference (not read).
- **Basketball-SORT** fixed camera: 20 manually clicked keypoints, then a homography [32].
  TrackID3x3 annotated keypoints on one frame [36].

### 5.2 Implementation (both modes)

```python
# Our own ViewTransformer: roboflow/sports' version has no RANSAC [4]
H, inliers = cv2.findHomography(img_pts, court_pts_m, cv2.RANSAC, ransacReprojThreshold=5.0)
# Gate: inliers.sum() >= 6, non-collinear inliers spanning both court axes,
# and median reprojection error < ~0.3 m (= 1.0 ft; code works in feet, RESEARCH.md D1) in court space (thresholds unverified; tune them).
feet_img = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
feet_court = cv2.perspectiveTransform(feet_img.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
```

- OpenCV `findHomography` method flags are `0`, `LMEDS` (4), `RANSAC` (8), `RHO` (16) and the
  `USAC_*` variants (`USAC_DEFAULT`=32 through `USAC_MAGSAC`=38). The signature is
  `findHomography(srcPoints, dstPoints, method=0, ransacReprojThreshold=3, mask=noArray(),
  maxIters=2000, confidence=0.995)`, confirmed in the OpenCV 4.x `calib3d.hpp` header [70]
  (docs.opencv.org itself returned 403 [66]). `cv2.USAC_MAGSAC` is the MAGSAC++ option.
- **Units:** court template in metres. FIBA court 28 × 15 m; NBA 94 × 50 ft (28.65 × 15.24 m).
  Rim height 10 ft (3.05 m) [71]. Rim inner diameter 18 in (0.4572 m) [50].
- **Symmetry trap:** the court is symmetric end to end and side to side. Keypoint models confuse
  left and right ends when only one half is visible. Resolve with (a) the fixed camera's known
  orientation, (b) a temporal prior (H changes smoothly), (c) a basket or scorer's-table side
  classifier, and (d) rejecting an H whose implied camera position flips sides.
- **Foot point ≠ ground contact:**
  - Jumping players: use the last grounded position or the ankle keypoints. Roboflow's paint
    logic uses ankles and falls back to bbox bottom-center [49].
  - Box bottoms clipped by occlusion: flag low-confidence positions when the box aspect ratio is
    abnormal.
- **Lens distortion:** wide coaching cameras and fisheyes break the planar homography. Calibrate
  intrinsics once (`cv2.calibrateCamera` or `cv2.fisheye.calibrate`) and run `cv2.undistortPoints`
  on foot points *before* applying H. Basketball-SORT's fisheye camera was pre-calibrated [32].
- **3D (shots and ball):** a ground homography alone cannot place the ball in 3D. Recover the full
  projection matrix P = K[R|t]:
  - KaliCalib-style, from the homography via `calibrateCamera` [47].
  - Or add non-planar points: rim centre at 3.05 m (10 ft), backboard corners.
  - With P and a known ball size (about 24 cm diameter, unverified exact value), monocular ball
    depth is feasible. Predict the ball's pixel diameter or height and back-project
    (Van Zandycke & De Vleeschouwer 2022; Caio et al. 2023; code `gabriel-vanzandycke/deepsport`)
    [53][52].

### 5.3 Fixed camera versus broadcast

| | Fixed coaching camera | Broadcast |
|---|---|---|
| H estimate | Once at setup, from manual clicks of ≥ 8 landmarks, or the median of keypoint-model H over the first about 60 frames | Every frame (or every 2nd frame, interpolating with CMC) |
| Validation | Reproject court landmarks every about 5 s. If the median error exceeds the threshold, re-run calibration (the camera was bumped) | Per frame: inlier count, reprojection error, and smoothness relative to H_{t-1} |
| Camera motion | None. Watch for slow drift from a tripod settling | Pan, tilt and zoom. Use BoT-SORT CMC (`sparseOptFlow`) to propagate H_{t-1} through the inter-frame global motion when keypoints are insufficient |
| Zoom | n/a | Close-ups have few lines, so registration fails. Mark frames `uncalibrated` and drop court-space stats for them |
| Cuts and replays | n/a | Detect shot boundaries and replays (§8). Reset H and the tracker |

---

## 6. Ball tracking and make/miss

### 6.1 Why the ball needs its own path

The numbers below are my arithmetic, based on an NBA ball of about 24 cm (unverified exact value)
and a wide view of a full court of about 28 m across 1920 px:
- About 68 px per metre, so the ball is **about 16 px** across at native 1080p. After resizing to
  512 px (RF-DETR-S) that is about **4 px**; at 640 px it is about 5 px.
- At 30 fps, a ball moving about 8 m/s (unverified typical pass or shot speed) travels about 27 cm,
  or about 18 px, per frame and blurs.
- Roboflow's own list of open problems puts ball tracking first, "due to its small size and rapid
  movements, especially in high-resolution videos" [2].

**Strategies:**
1. **ROI re-detection at native resolution:** crop a 640×640 window (about 9.3 m of court in a
   wide view, at about 68.6 px/m) around the Kalman-predicted ball position and run the detector on it every frame.
   When the ball is lost, sweep the frame with `sv.InferenceSlicer(slice_wh=(640,640),
   overlap_wh=(64,64), batch_size=...)` [23]. The soccer example detects the ball on 640×640
   slices with NMS 0.1 [5].
2. **Train ball and rim at higher input resolution** (RF-DETR-M at 576 or L at 704), or use YOLO26
   with STAL (P2 heads have no pretrained weights) [8][13].
3. **Temporal-input detectors** such as TrackNet (a heatmap model over several consecutive frames,
   640×360 input) help small, fast balls in racket sports: precision 99.7% / recall 97.3% on its
   main test video, but recall fell to 75.7% (precision 95.3%) in 10-fold cross-validation that
   added nine partially labelled videos [63]. Consider these only after measuring per-frame recall.
4. **Single-ball tracker:**
   - The roboflow/sports `BallTracker` keeps the detection nearest the centroid of the last
     `buffer_size=10` positions [6], or 20 in the soccer example [5]. That is a good minimal
     baseline.
   - Our version: a Kalman filter with state (x, y, vx, vy, ay) in image space, where ay absorbs
     projected gravity during flight. Gate by Mahalanobis distance. Allow at most one ball, and
     reject "balls" on heads, spare balls and rack balls with a court-ROI mask plus a motion prior.
   - States: `held(player_id)` → `dribble` → `pass/shot flight` → `loose`.

### 6.2 Possession and occlusion in hands

- A held ball is often hidden or merged with the hands. Possession is inferred **from the track,
  not the ball box**:
  - The ball is "held" by the player whose wrist keypoints (or box upper half) are within about
    0.5 m (unverified) for N frames.
  - When the ball vanishes, keep it "held by the last holder" until it reappears more than about
    1 m away or another holder appears.
- Roboflow's `player-in-possession` class [1] is a useful prior but is single-frame. Smooth it.

### 6.3 Shot detection and trajectory fitting

- **Release:** the ball separates from the shooter's wrist area (Roboflow uses about 0.35 m) while
  moving upward [50].
- **Attempt:** fit a parabola y = a·t² + b·t + c in image space (or the full 3D ballistic model
  when P is known, §5.2) over ≥ 5 post-release points (the point count is our choice; Roboflow does
  not state one, unverified). Roboflow records an attempt only if the projected arc points toward
  the rim bounding box [50].
- 3D ballistic fit: with P known, solve for (x0, v0) under gravity, minimizing reprojection error.
  This gives release height, release angle, entry angle and shot distance. Roboflow derives
  pixels-per-metre from the median rim width, using the 0.4572 m rim [50].

### 6.4 Make/miss determination

- **Rim-plane crossing:** a make requires the ball centre to go above the rim top and then below
  the rim plane, with its x inside the rim's x-span (shrunk by the ball radius) on the descending
  branch. Patent literature describes "above hoop in ROI then falls" [55] (snippet; patents only; unverified).
- **Net occlusion and drop speed** (Roboflow shot evaluator) [50]:
  - The net can hide the ball. In Roboflow's anecdote it was hidden for 2 frames, which made two
    makes log as misses. The post gives no general frame-count rule, so do not hard-code 2.
  - Drop speed on reappearance, as a fraction of free fall: 10% to 90% means a **make** (the net
    slowed it); 100% means an air ball behind the rim; 0% means a stationary court ball, rejected.
  - Rattles: a bounce within 1 rim-width that goes in is a "rattled make". A high bounce away from
    the hoop is a "rattled miss".
  - **Uncertain shots are tagged `rattled` or `occluded` in `shots.jsonl` rather than guessed.**
    This is the right pattern.
- **Net motion:** frame-difference energy in a net ROI below the rim, after the ball enters the rim
  ROI. Patents describe "substantial hoop net movement" as a make signal even with the ball
  occluded [55] (snippet; unverified).
- **Detector class `ball-in-basket`** [1]: one vote.
- **Reported accuracy:**
  - Roboflow reports no make/miss accuracy. In their tests two made shots were first logged as
    misses because of occlusion, and single-camera front-rim bounce-outs can look like rattle-in
    makes [50].
  - The nvan21 repo notes that the ball can hit the back rim while the 2D path looks like a make
    [54].
  - A 2025 BEEI paper (YOLOv8 hoop + ViT + DeepSORT) reports 94.55% accuracy and 94.5% F1 on 44
    scoring events from 5 broadcast videos [55] (snippet; page not fetched; unverified).
  - Roboflow's ball detection rate in flight was 98.1% [50].
  - A Theseus PoC found amateur footage (poor line visibility, camera motion, occlusion) hurt both
    homography and shot classification [65] (snippet; unverified).
- **CourtVision fusion:**
  - Compute the cues above as typed features: crossing (bool, margin in px), drop-speed fraction,
    net-motion z-score, `ball-in-basket` max confidence, rim-contact flag, view quality, and
    occlusion frames.
  - If the cues agree, commit the event.
  - If they disagree or quality is low, send the features (not frames) to **Jev** as a
    `Choice(make | miss | unclear)`. Commit only above a calibrated threshold. Otherwise emit
    `shot_outcome_pending`, and reconcile with the scorebug (broadcast) or the next inbound or
    rebound pattern (a make leads to an inbound under the basket).
  - Following `jev_api.md`, compute numbers in Python and send them as named buckets or text. Do
    not put live frames on the Jev hot path.
  - The LLM narrates only committed or explicitly hedged outcomes.

---

## 7. Pose estimation

| Model | COCO AP | Latency | Licence and notes |
|---|---|---|---|
| YOLO26n-pose / s / m | 57.2 / 63.0 / 68.8 mAP-pose (e2e) | T4 1.8 / 2.7 / 5.0 ms; CPU ONNX 40 / 85 / 218 ms | AGPL-3.0. Single-stage on the full frame (the docs page does not label it top-down or bottom-up). COCO 17 keypoints [57] |
| RTMPose-t / s / m (256×192) | 68.5 / 72.2 / 75.8 | i7 CPU ORT 3.20 / 4.48 / 11.06 ms per crop; GTX 1660Ti TRT 1.06 / 1.39 / 2.29 ms | Top-down on crops. `rtmlib` runs ONNX without torch or mmcv [56]. Code licence: `open-mmlab/mmpose` and `Tau-J/rtmlib` are Apache-2.0 per GitHub; weight licences not checked (unverified) |
| ViTPose (`usyd-community/vitpose-base-simple`, `vitpose-plus-base`/`-huge`) | (not re-verified) | heavy | HF `VitPoseForPoseEstimation`. Top-down, needs person boxes. ViTPose++ uses MoE dataset heads (`dataset_index`) [58] |
| `RFDETRKeypointPreview` | 71.8 | 9.7 ms T4 | Apache-2.0, COCO-17. Used zero-shot for shot form at 99% frame coverage [13][50] |

On TrackID3x3 basketball without fine-tuning, outdoor PDJ was RTMPose 89.43, SwinPose 89.27 and
HRNet 88.51. Indoor 720p video was lower (HRNet 77.26, RTMPose 73.28) [36].

**What pose adds:**
1. **Feet:** ankles give the position relative to the 3-point line, paint and baseline, which
   beats bbox bottom-center [49].
2. **Release point and timing:** wrist and ball separation [50].
3. **Shooting form:** elbow and knee angles at set point and release, and jump height from ankle
   y-trajectory. These give coach-facing metrics [50].
4. **Torso crop** for jersey OCR [40].
5. Facing direction and box-out posture, for later tactical features.

**Scheduling:** do not run pose on all 10 players every frame. Run it on (a) the ball holder and
nearest defender at 10 to 15 Hz, (b) the shooter at full rate from the gather until 0.5 s after
release, and (c) tracks queued for jersey reads. RTMPose-s or -m via `rtmlib` on crops is the
efficient option for selective crops. YOLO26n-pose is simpler if Ultralytics is already licensed.

---

## 8. Broadcast-specific concerns

- **Shot boundaries and replays:** broadcast includes cuts, slow-motion replays, close-ups, crowd
  shots, picture-in-picture and graphics wipes. **Replays duplicate events.** Plan:
  1. A shot-boundary detector (histogram or feature difference).
  2. A view classifier `{game-wide, close-up, replay, other}`, trainable from a few hundred labelled
     frames. Slow motion shows up as anomalous ball and player velocities, and replays are often
     bracketed by logo wipes (unverified heuristic).
  3. Run tracking, homography and the event engine only in `game-wide` views. In other views,
     freeze state.
- **Scorebug OCR:** read the game clock, shot clock and score from a fixed crop at **5 Hz** (gap pass: RESEARCH.md D14 supersedes the earlier 1 to 2 Hz, so two identical reads confirm a change in ≈ 400 ms).
  This is the authoritative source for score changes and lets us confirm makes. A VLM read of the
  scorebug is what the Aug 2026 basketball agent does (Qwen3.5-9B) [46]. On-device, a small digit
  classifier is far cheaper.
- **Off-screen players:** a broadcast view usually shows 6 to 10 players. Court-space stats
  (spacing, defensive shape) are **partial** and must carry a `coverage` flag.
- **Identity across cuts:** the global ID is (team, number). The tracker resets at each cut (§3.4).

---

## 9. Throughput on an Apple M3 Pro

### 9.1 What the evidence says

- **PyTorch MPS:** select with `device="mps"`, check `torch.backends.mps.is_available()`. Needs
  macOS 14.0+. The current docs are versioned 2.14 [61]. Setting `PYTORCH_ENABLE_MPS_FALLBACK=1`
  makes unsupported MPS ops run on the CPU [68]; do not assume fallback without it. RF-DETR's
  `aten::_upsample_bicubic2d_aa.out` is reported to "fall back to run on the CPU" with a
  performance warning [18].
- **Ultralytics on Apple** [59]:
  - Export with `model.export(format="coreml", quantize=16|8|"w8a16", imgsz=640, nms=...)`. The
    `half` and `int8` arguments are deprecated in favour of `quantize`. `nms=True` embeds NMS;
    `nms=False` selects the NMS-free head.
  - On a Mac host Ultralytics loads CoreML with **`CPU_AND_NE`** (macOS 13+, otherwise
    `CPU_ONLY`), "~3× faster than CPU". Its docs warn that `ALL` and `CPU_AND_GPU` can abort with
    `Error: MLIR pass manager failed` on coremltools 9.x [59]. RF-DETR's docs measured `ALL` and
    `CPU_AND_GPU` on an M3 Pro with coremltools 9.0 without reporting an abort [17], so this looks
    model-specific; test both.
  - `format="coreai"` targets macOS 27 and iOS 27.
  - YOLO26n INT8 on an iPhone 17 Pro: 3.2 ms with the ANE versus 9.2 ms CPU-only (burst; sustained
    use is slower).
- **Measured M-series numbers** (third-party, MacBook Air M3, single runs) [60]:

  | Model and backend | Inference ms | Total ms/frame |
  |---|---|---|
  | YOLO11n, Ultralytics MPS, 384×640 | 9.2 | about 15.9 (about 63 FPS) |
  | YOLO11m, Ultralytics MPS | 23.3 | about 36.6 (about 27 FPS) |
  | YOLO11n, Ultralytics CPU | 25.8 | about 27.5 |
  | YOLO11n, CoreML via ONNX/Ortex | 15.6 | about 20 |

- **RF-DETR on Apple (official, first-party) [17]:**
  - M3 Pro, `RFDETRNano` 384×384, batch 1, p50: FP16 20.8 ms (`CPU_AND_NE`), 21.2 ms (`ALL`),
    33.9 ms (`CPU_AND_GPU`), 44.5 ms (`CPU_ONLY`); FP32 about 34 ms on GPU and 73.6 ms on CPU.
  - First load of an FP16 bundle on the ANE compiles for about 5 s; later loads are cached.
  - The develop-branch docs (unreleased) list M4 Max FP16 numbers: `RFDETRSmall` 23.3 ms and
    `RFDETRMedium` 24.0 ms on `CPU_AND_GPU`. An M3 Pro will be slower than an M4 Max (unverified by
    how much).
- **SAM2:** about 1 to 2 FPS on a T4 for the players in a basketball clip [1]. Expect the same or worse on an M3 Pro
  (unverified). Keep it offline.

### 9.2 Realistic budget for 1080p30 live (estimates, to be measured)

The frame budget is 33 ms. A plausible M3 Pro plan, **all unverified until benchmarked**:

| Stage | Rate | Est. cost | Where |
|---|---|---|---|
| Decode 1080p H.264/HEVC | 30 Hz | a few ms with VideoToolbox HW decode (unverified) | Decode thread (PyAV/ffmpeg) |
| Detector (players, refs, rim, ball), RF-DETR-S CoreML FP16 at 512 or YOLO26s CoreML at 640 | 15 to 30 Hz | **at least about 21 ms**: RF-DETR's docs measure `RFDETRNano` at 384 px (FP16) at 20.8 to 21.2 ms p50 on an M3 Pro [17]. Small at 512 will be slower (unverified by how much). YOLO26s was not measured on a Mac [60][59] | ANE/GPU, main inference process |
| Ball ROI re-detect, 640 crop at native resolution | 30 Hz | +5 to 15 ms (unverified) | Same model or a smaller ball-only model |
| Tracker (`ByteTrackTracker` / `BoTSORTTracker` with CMC) | 30 Hz | < 2 ms for ~13 objects; CMC sparse optical flow at `cmc_downscale=2` costs a few ms (unverified) | CPU |
| Homography projection | 30 Hz | < 0.1 ms | CPU |
| Court keypoints (broadcast) | 10 to 30 Hz | about the cost of a detector pass | ANE/GPU |
| Court drift check (fixed camera) | 0.2 Hz | negligible | |
| SigLIP team embeddings | Per track every about 15 frames, batched | 32-crop batch per 0.5 s (unverified cost) | MPS, async worker |
| Jersey classifier | Per legible track every 5 frames [1] | tiny CNN, batched | CoreML, async |
| Pose | Shooter and ball handler only | RTMPose-m crops about 11 ms each on CPU ORT [56] | async |
| Event engine, then Jev, then Claude narration | Event-driven | network-bound | separate process |

**Engineering rules:**
- Run **detection at 15 Hz and let the Kalman filter fill the frames in between** if 30 Hz does not
  fit. Keep the **ball ROI path at 30 Hz**: shots and passes need it.
- Move pipeline stages between **processes** (multiprocessing or shared memory), not threads,
  because of the GIL.
- Keep the newest frame and drop stale ones: live narration needs bounded latency (glass-to-event
  target about 300 to 500 ms, unverified target), not every frame.
- Benchmark p50 and p95 latency **over 10+ minutes**, because thermal throttling on laptops shows
  up only in sustained runs (unverified for the M3 Pro). Ultralytics notes that sustained use is
  slower than burst numbers [59].
- If live throughput is insufficient, offer **near-live** mode (a 2 to 5 s delay with a lookahead
  window). This also lets the event engine use future frames to resolve make/miss and identity.

---

## 10. Recommended concrete stacks

### 10.1 Stack A: fixed single wide camera (coach use)

```
Camera (1080p30 or 4K30, high sideline mount, full court, fixed)
 ├─ setup: intrinsics (checkerboard) → undistort; court H from ≥8 clicked or auto keypoints (RANSAC)
 ├─ per frame (30 Hz):
 │    decode → RF-DETR-S/M (CoreML FP16, classes: player, referee, ball, rim, [number])
 │    → ball ROI re-detect at native resolution → BallKalman (single ball, gravity term)
 │    → trackers.ByteTrackTracker (no CMC) on players and refs
 │    → foot points → undistortPoints → H → court xy (m) → court-space association layer
 │       (Basketball-SORT style: ≤5 per team, RLLI with α=0.2, β=260 cm)
 ├─ async: SigLIP (google/siglip-base-patch16-224) + UMAP(3) + KMeans(2), fit at warm-up;
 │         per-track vote every ~15 frames; colour-histogram fallback for pinnies
 ├─ async: jersey CNN (00–99 + illegible) on legible torso crops every 5 frames; ≥3 agreeing votes
 │         + roster constraint (disabled for pinnies → coach tags + ReID)
 ├─ on demand: RTMPose-m (rtmlib) or YOLO26n-pose for the shooter and ball handler
 ├─ every ~5 s: reproject court landmarks to detect a bumped camera, then recalibrate
 └─ events: possession / pass / shot / make-miss (fused cues; ambiguous → Jev Choice) → stats → Claude
```

- The "per frame (30 Hz)" RF-DETR-S/M pass above is unlikely to fit on an M3 Pro: even
  `RFDETRNano` at 384 px measures about 21 ms per image there [17]. Start with full-frame
  detection at 15 Hz plus Kalman fill-in, and keep only the ball ROI at 30 Hz (§9.2).
- Rim: detect it over the first N frames and fix its box (it does not move). Re-detect every
  minute, and also on any H change.
- Calibrate P (§5.2) once using the rim height, for 3D shot metrics.
- Optional post-session pass: SAM 2.1 re-tracking plus a full-rate pose for "final" stats and form
  reports.

### 10.2 Stack B: broadcast

```
Broadcast feed (720p–1080p, 25/30/50/60 fps)
 ├─ shot-boundary + view classifier → only "game-wide" frames continue; reset on cut
 ├─ scorebug OCR (5 Hz, D14) → authoritative score and clock
 ├─ per frame: RF-DETR-M (players, referee, ball, rim, number) + court keypoints (33-pt RF-DETR keypoint)
 │    → H_t via RANSAC; gate on inliers and reprojection; else H_{t-1} ∘ CMC; else mark uncalibrated
 │    → trackers.BoTSORTTracker(enable_cmc=True, cmc_method="sparseOptFlow"), frame passed to update()
 │    → identity = (team, number) across cuts; tracklets are local
 ├─ async: team embedding (fit per game in the first minute of game-wide frames)
 ├─ async: jersey recognition (more important here: it is the only cross-cut identity cue)
 ├─ ball: ROI path + rim crossing; make confirmation from the scorebug delta within ~3 s
 └─ coverage flags on all court-space stats (off-screen players)
```

- Replays must never generate events. Use replay detection plus scorebug consistency.
- Higher frame rates (50 or 60 fps) help the ball but double the cost. Detect at 25 to 30 Hz and
  run the ball ROI at full rate.

---

## 11. Failure modes to engineer around

| # | Failure | Symptom | Mitigation |
|---|---|---|---|
| 1 | Paint pile-ups and screens (CMOO) | ID swaps between same-team players | Court-space association, cardinality caps, freeze-then-resolve in occlusion clusters, RLLI [32] |
| 2 | Same-team look-alikes | ReID useless within a team | Jersey votes plus position continuity. Never decide identity on appearance alone [33] |
| 3 | Unreadable numbers (back turned, blur, far, partial "23" read as "2") | Wrong roster mapping | Legibility gate, tracklet voting, 2-digit bias, roster constraint, "unknown" state [40][1] |
| 4 | Class imbalance and missing numbers in training data | Classifier cannot output an unseen number | Synthetic or augmented crops, a PARSeq fallback, a per-team roster prior [1][40] |
| 5 | Team clusters flip or merge (similar colours, alternates, lighting, refs) | Wrong team stats | Fit on players only, bind clusters to home/away once, hysteresis per track, an "other" bucket, histogram cross-check [3][5][36] |
| 6 | Homography: too few or collinear points, central or zoomed views, left/right end confusion | Teleporting players | RANSAC gating, a smoothness prior, CMC fallback, `uncalibrated` state, a side classifier [47][48] |
| 7 | Lens distortion in wide coaching cameras | Position errors growing toward the frame edges | Calibrate intrinsics and undistort points before H [32] |
| 8 | Foot point off the floor (jumps), or occluded box bottoms | False court positions and wrong 2PT/3PT calls | Ankle keypoints, last grounded position, confidence flags [49] |
| 9 | Tiny, blurred ball; ball in hands; spare balls and racks; heads detected as balls | Missed passes and shots, phantom possession | ROI native-res re-detect, single-ball Kalman with gating, court-ROI mask, holder-state inference [2][5][6] |
| 10 | Front-rim and back-rim ambiguity in 2D; net occlusion; rattles | Wrong make/miss | Cue fusion, drop-speed rule, net motion, scorebug confirmation, Jev `Choice` with an "unclear" option, a pending state [50][54] |
| 11 | Broadcast replays, cuts, PiP and graphics | Duplicate or phantom events | View classifier, tracker reset, scorebug consistency |
| 12 | Practice footage: more than 10 players on court, drills, no numbers, coaches on the floor | Cardinality assumptions break | Mode-specific config: no BGR cap in drills, coach tagging |
| 13 | MPS op fallbacks, GIL contention, thermal throttling | FPS collapses after minutes | CoreML export, multi-process pipeline, sustained benchmarks, frame dropping [18][59] |
| 14 | API drift (`sv.ByteTrack` removed in 0.31.0; Ultralytics' default tracker is now TrackTrack) | Breakage on upgrade, silently changed behaviour | Pin `supervision`, `trackers`, `rfdetr` and `ultralytics`; set the tracker config explicitly [23][28] |
| 15 | Licences: Ultralytics AGPL §13, PML for RF-DETR XL/2XL (and Atto/Femto/Pico), the SAM License for SAM 3, Objects365 terms for D-FINE O365 weights, **CC BY-NC 3.0 for `mkoshkina/jersey-number-pipeline` code and weights**, Universe dataset licences, NBA footage rights | Legal exposure | Default to Apache-2.0 components and audit before shipping [11][13][38][19][41] |
| 16 | LLM hallucinating events from noisy CV | Wrong live narration | The CV layer emits typed events with confidences. Narrow judgments go to Jev. Claude narrates only committed or hedged facts |

---

## 12. Pinned starting versions (Oct 2026)

| Package | Version | Licence | Note |
|---|---|---|---|
| `rfdetr` | 1.11.2 (2026-10-04) | Apache-2.0 (Plus: PML 1.0) | `rfdetr[coreml]` for export; the extra pins `coremltools>=8.0,<10.0` [14][17][74] |
| `supervision` | 0.31.0 (2026-10-16) | MIT | No `sv.ByteTrack` [22][23] |
| `trackers` | 2.6.1 (2026-09-25) | Apache-2.0 | `ByteTrackTracker`, `BoTSORTTracker` [25][26] |
| `ultralytics` | 8.4.183 (2026-10-20) | AGPL-3.0 / Enterprise | Optional, licence-gated [10] |
| `sam2` | from GitHub | Apache-2.0 | Offline only. torch ≥ 2.5.1 [37] |
| `transformers` | docs at v5.20.2 | Apache-2.0 (GitHub licence field) | SigLIP, ViTPose [58] |
| `umap-learn`, `scikit-learn` | latest | BSD-3-Clause (GitHub licence fields) | `TeamClassifier` deps [3] |
| `rtmlib` | latest | Apache-2.0 (GitHub licence field) | RTMPose ONNX without mmcv [56] |
| `coremltools` | 9.x (rfdetr allows 8.x to 9.x) | BSD-3-Clause (GitHub licence field) | Ultralytics warns of MLIR aborts with `ALL`/`CPU_AND_GPU` [59]; RF-DETR's M3 Pro runs did not report one [17] |
| `sports` (roboflow) | git main | MIT | Reference code only: `TeamClassifier`, `ViewTransformer`, `BallTracker` [2] |

---

## Sources

[1] Roboflow, "How to Detect, Track, and Identify Basketball Players with Computer Vision" (P. Skalski, 30 Sep 2025). https://roboflow.com/blog/identify-basketball-players (old URL https://blog.roboflow.com/identify-basketball-players/ redirects here)
[2] roboflow/sports README (datasets, challenges, MIT). https://github.com/roboflow/sports and https://raw.githubusercontent.com/roboflow/sports/main/README.md
[3] roboflow/sports `sports/common/team.py` (`TeamClassifier`, SigLIP id). https://raw.githubusercontent.com/roboflow/sports/main/sports/common/team.py
[4] roboflow/sports `sports/common/view.py` (`ViewTransformer`). https://raw.githubusercontent.com/roboflow/sports/main/sports/common/view.py
[5] roboflow/sports `examples/soccer/main.py`. https://raw.githubusercontent.com/roboflow/sports/main/examples/soccer/main.py
[6] roboflow/sports `sports/common/ball.py` (`BallTracker`). https://raw.githubusercontent.com/roboflow/sports/main/sports/common/ball.py
[7] Pixelsham mirror of the Roboflow basketball post (dataset versions; secondary). https://www.pixelsham.com/?p=25925
[8] Ultralytics YOLO26 docs. https://docs.ultralytics.com/models/yolo26/
[9] Ultralytics YOLO11 docs. https://docs.ultralytics.com/models/yolo11/
[10] ultralytics on PyPI. https://pypi.org/project/ultralytics/
[11] GNU AGPL-3.0 text, §13. https://www.gnu.org/licenses/agpl-3.0.en.html
[12] Ultralytics licensing page (fetched during verification). https://www.ultralytics.com/license
[13] roboflow/rf-detr README. https://github.com/roboflow/rf-detr
[14] rfdetr on PyPI. https://pypi.org/project/rfdetr/
[15] RF-DETR paper, arXiv 2511.09554 (ICLR 2026; cited from the README). https://arxiv.org/abs/2511.09554
[16] Roboflow, "How to Fine-Tune RF-DETR Keypoints on Custom Data" (26 Jun 2026). https://roboflow.com/blog/train-rf-detr-keypoint
[17] RF-DETR docs, "Native CoreML Export (.mlpackage)" (fetched; the old cookbook URL https://rfdetr.roboflow.com/latest/cookbooks/export-coreml/ returns 404). https://rfdetr.roboflow.com/latest/exports/coreml/ ; develop-branch source: https://github.com/roboflow/rf-detr/blob/develop/docs/exports/coreml.md
[18] Roboflow forum, "Can RF-DETR run with mac silicon?". https://discuss.roboflow.com/t/can-rf-detr-run-with-mac-silicon/10580
[19] D-FINE repo. https://github.com/Peterande/D-FINE
[20] DEIMv2, "Real-Time Object Detection Meets DINOv3" (abstract fetched). https://arxiv.org/abs/2509.20787
[21] supervision 0.29.0 trackers docs (`ByteTrack` deprecation). https://supervision.roboflow.com/0.29.0/trackers/
[22] supervision on PyPI (release history). https://pypi.org/project/supervision/
[23] supervision changelog. https://supervision.roboflow.com/latest/changelog/
[24] supervision `Detections` docs. https://supervision.roboflow.com/latest/detection/core/
[25] roboflow/trackers repo and PyPI. https://github.com/roboflow/trackers ; https://pypi.org/pypi/trackers/json
[26] trackers API reference. https://trackers.roboflow.com/latest/api/trackers/
[27] trackers comparison. https://trackers.roboflow.com/latest/trackers/comparison/
[28] Ultralytics track mode docs. https://docs.ultralytics.com/modes/track/
[29] ByteTrack, arXiv 2110.06864. https://arxiv.org/abs/2110.06864
[30] BoT-SORT, arXiv 2206.14651. https://arxiv.org/abs/2206.14651
[31] Deep OC-SORT, arXiv 2302.11813. https://arxiv.org/abs/2302.11813
[32] Basketball-SORT, arXiv 2406.19655. https://arxiv.org/html/2406.19655v1
[33] SportsMOT, arXiv 2304.05170. https://arxiv.org/abs/2304.05170
[34] SAMIDARE, arXiv 2604.22162. https://arxiv.org/abs/2604.22162
[35] Sports-vmTracking (3x3 basketball), arXiv 2412.06258. https://arxiv.org/abs/2412.06258
[36] TrackID3x3, arXiv 2503.18282. https://arxiv.org/html/2503.18282v1
[37] facebookresearch/sam2. https://github.com/facebookresearch/sam2
[38] facebookresearch/sam3. https://github.com/facebookresearch/sam3
[39] Meta AI blog, SAM 3 / SAM 3.1. https://ai.meta.com/blog/segment-anything-model-3
[40] Koshkina & Elder, "A General Framework for Jersey Number Recognition in Sports Video", arXiv 2405.13896. https://arxiv.org/html/2405.13896v1
[41] jersey-number-pipeline code (linked from [40]; README states CVPR 2024 Workshops and a CC BY-NC 3.0 licence). https://github.com/mkoshkina/jersey-number-pipeline
[42] Hugging Face, SmolVLM2 blog. https://huggingface.co/blog/smolvlm2
[43] Roboflow, "Comparing Base and Fine-Tuned SmolVLM2 for OCR" (14 Jul 2025). https://roboflow.com/blog/base-vs-fine-tuned-smolvlm2-ocr
[44] baudm/parseq (README and GitHub licence field checked). https://github.com/baudm/parseq
[45] SoccerNet 2023 Challenges Results, arXiv 2309.06006 (search snippet). https://arxiv.org/pdf/2309.06006
[46] "Towards Comprehensive Basketball Understanding", arXiv 2608.23435 (Aug 2026). https://arxiv.org/html/2608.23435v1
[47] KaliCalib, arXiv 2209.07795. https://ar5iv.labs.arxiv.org/html/2209.07795
[48] Magera et al., "Can Geometry Save Central Views for Sports Field Registration?", arXiv 2504.20052. https://arxiv.org/abs/2504.20052
[49] Roboflow, "Detect NBA 3 Second Violations with AI" (22 Jul 2025). https://roboflow.com/blog/detect-3-second-violation-ai-basketball
[50] Roboflow, "How to Build an AI Basketball Shot Evaluator" (14 Aug 2026). https://roboflow.com/blog/how-to-build-an-ai-basketball-shot-evaluator
[51] Roboflow, "How to Build Jersey Number Recognition for Sports" (4 May 2026). https://roboflow.com/blog/jersey-number-recognition-for-sports
[52] Caio, Van Zandycke, De Vleeschouwer, arXiv 2309.03640 (code: github.com/gabriel-vanzandycke/deepsport). https://arxiv.org/abs/2309.03640
[53] Van Zandycke & De Vleeschouwer, "3D Ball Localization From A Single Calibrated Image" (CVPR-W 2022; search snippet). https://ar5iv.labs.arxiv.org/html/2204.00003
[54] nvan21/Basketball-Shot-Detection. https://github.com/nvan21/Basketball-Shot-Detection
[55] "Scoring smarter: deep learning-based basketball scoring detection in real time", BEEI (search snippet; page fetch failed). Patent claims came from USPTO search snippets (e.g. https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11594029). https://beei.org/index.php/EEI/article/view/11374
[56] MMPose RTMPose project. https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
[57] Ultralytics pose docs (YOLO26-pose). https://docs.ultralytics.com/tasks/pose/
[58] Hugging Face Transformers ViTPose docs. https://huggingface.co/docs/transformers/model_doc/vitpose
[59] Ultralytics CoreML integration docs. https://docs.ultralytics.com/integrations/coreml/
[60] Elixir YOLO, MacBook Air M3 benchmarks (third-party). https://yolo.hexdocs.pm/macbook_air_m3.html
[61] PyTorch MPS backend notes. https://docs.pytorch.org/docs/2.14/notes/mps.html
[62] Roboflow, "Best iOS Object Detection Models" (10 Nov 2025). https://roboflow.com/blog/best-ios-object-detection-models
[63] TrackNet, arXiv 1907.03698 (abstract fetched). https://arxiv.org/abs/1907.03698
[64] TeamTrack dataset (linked from [32]). https://github.com/AtomScott/TeamTrack
[65] Theseus thesis, "Counting Basketball Shot Attempts with Computer Vision (PoC)" (search snippet). https://www.theseus.fi/handle/10024/909540
[66] OpenCV calib3d docs (`findHomography`; HTTP 403 on fetch). https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
[67] Roboflow Universe dataset page (HTTP 403 on fetch). https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo
[68] PyTorch 2.14 MPS environment variables (`PYTORCH_ENABLE_MPS_FALLBACK`). https://docs.pytorch.org/docs/2.14/mps_environment_variables.html
[69] Ultralytics YOLO27 preview page ("no launch date has been set"). https://docs.ultralytics.com/models/yolo27/
[70] OpenCV 4.x `calib3d.hpp` header (`findHomography` signature and method enum). https://raw.githubusercontent.com/opencv/opencv/4.x/modules/calib3d/include/opencv2/calib3d.hpp
[71] Wikipedia, "Basketball court" (FIBA/NBA dimensions, rim height; secondary). https://en.wikipedia.org/wiki/Basketball_court
[72] facebookresearch/sam3 SAM 3.1 release notes. https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md
[73] facebookresearch/sam3 README (Python 3.12+, PyTorch 2.7+, CUDA 12.6+, 848M params, SAM License). https://github.com/facebookresearch/sam3
[74] rf-detr `pyproject.toml` (`coreml` extra pins `coremltools>=8.0,<10.0`). https://github.com/roboflow/rf-detr/blob/develop/pyproject.toml

---

## Verification log

Adversarial fact-check run on 2026-10-20. Each implementation-critical claim was re-checked against
the cited primary source (fetched page, raw repo file, PyPI JSON, Hugging Face API or GitHub API),
plus a second source where one existed. The web-search budget was exhausted during the check, so
the last items were verified by fetching known URLs only. "Jev" throughout means TypeSafe AI's Jev;
its references here match `docs/research/jev_api.md` (Choice, Score, Noul, plus Box on
`jev-1.14.0`), and Jev's API was not re-checked on the web in this pass.

### Confirmed as written (no change)

- **Roboflow Basketball AI post** [1] (30 Sep 2025, P. Skalski; the old blog URL returns HTTP 301 to
  the new one): RF-DETR-S; the 10 classes; `roboflow-jvuqo/basketball-player-detection-3-ycjdo`;
  SAM2 prompted by first-frame RF-DETR boxes; "around 1–2 FPS" on a T4; SAM2 is the slowdown and
  degrades as tracked objects increase; SmolVLM2 56% base and 86% fine-tuned on 3.6k 2025 NBA
  Playoff crops; ResNet-32 at 93% (the post's "Models" list links a ResNet-18 article, but the body
  says ResNet-32); "8" 315 times (8.7%) and "6" 8 times (0.2%); numbers sampled every 5 frames and
  confirmed after 3 identical predictions in a row; IoS ≥ 0.9 between the SAM2 mask and the number
  box as a mask; team clustering at 1 frame/s with SigLIP, UMAP to 3 dimensions and k=2.
- **roboflow/sports** [2]-[6]: `team.py` (`google/siglip-base-patch16-224`, `SiglipVisionModel`,
  `AutoProcessor`, mean of `last_hidden_state`, `batch_size=32`, `umap.UMAP(n_components=3)`,
  `KMeans(n_clusters=2)`, `reducer.transform` in `predict`); `view.py` calls
  `cv2.findHomography(source, target)` with no method argument and discards the mask; `ball.py`
  `BallTracker(buffer_size=10)`; the soccer example uses `STRIDE = 60`, player-class crops only,
  `BallTracker(buffer_size=20)`, 640×640 slices with `with_nms(threshold=0.1)`, and
  `sv.ByteTrack(minimum_consecutive_frames=3)`. The repo licence is MIT (GitHub API), and the
  README says there is no Python package yet.
- **supervision** [22][23][21]: 0.31.0 on 2026-10-16 (PyPI); MIT; Python ≥ 3.10. 0.31.0 removes
  `sv.ByteTrack` ("use `ByteTrackTracker` from the `trackers` package"), `supervision.keypoint`,
  `sv.LMM`/`from_lmm` and the legacy `MeanAveragePrecision`, and adds `coord_transform` on
  `PolygonZone.trigger`/`LineZone.trigger` with `sv.CoordinatesTransformation`. 0.30.0 (Aug 4 2026):
  `InferenceSlicer(batch_size=...)`, `PolygonZone(require_all_anchors=True)`, `KeyPoints.merge`,
  `get_video_frames_generator(prefetch=0)`. 0.28.0 (Apr 30 2026): `sv.ByteTrack` deprecated and
  `Detections.from_sam3` added. 0.27.0 (Nov 16 2025): `filter_segments_by_distance` added and
  `overlap_ratio_wh` removed. The old `sv.ByteTrack` signature and methods match the 0.29.0 docs.
  Those docs said removal would be in 0.30.0, but it actually happened in 0.31.0.
- **trackers** [25][26][27]: 2.6.1 on 2026-09-25, Apache-2.0, Python ≥ 3.10 (PyPI). Every
  constructor default in §3.1 matches the API page, as do `update(detections, frame=None,
  timestamp=None)`, the CMC methods, and McByte's RGB requirement. The CLI has `track`, `eval`,
  `download` and `tune` subcommands (`src/trackers/cli/` on `develop`). The benchmark table in §3.2
  matches exactly, with YOLOX detections for MOT17, SportsMOT and DanceTrack and oracle boxes for
  SoccerNet.
- **RF-DETR** [13][14]: 1.11.2 on 2026-10-04, Apache-2.0, Python ≥ 3.10; `coreml` and `coreai`
  extras exist. Every row of the detection, segmentation and keypoint tables matches the README, as
  do the licences (PML 1.0 for Atto, Femto, Pico, XL and 2XL), RF100-VL S 60.2 and M 61.2, the
  `predict(...)` and `detections.data["class_name"]` usage, and `inference.get_model("rfdetr-medium")`.
- **YOLO26 / Ultralytics** [8][9][28][57][59][10]: released January 2026; every row of the
  detection and pose tables matches; NMS-free one-to-one head, DFL removed, ProgLoss, STAL, MuSGD;
  P2/P6 YAMLs with no released weights; AGPL-3.0 or Enterprise. `ultralytics` 8.4.183 was released
  on 2026-10-20 (PyPI). The six tracker YAMLs, TrackTrack as the default, ≥ 8.4.63 for the four
  newer trackers, the ReID ONNX names and the ReID-capable trackers all match. CoreML `quantize` is
  16, 8, "w8a16" or 32, replacing `half`/`int8`. `CPU_AND_NE` on a Mac host, "~3× faster than CPU",
  the MLIR abort warning and `format="coreai"` for the 27-generation OSes all match. The iPhone 17 Pro
  YOLO26n figures (3.2 vs 9.2 ms) are INT8 burst numbers. YOLO11 values and release date (27 Sep
  2024) match.
- **Basketball-SORT** [32]: 20 court keypoints, fixed side-view fisheye pre-calibrated camera,
  YOLOv8l, FastReID, BGR at frame 100 with 10 trajectories and no new tracks, α = 0.2 and β = 260,
  Table 1 values (63.35/59.15/9.3 vs BoT-SORT 59.70/53.43/23.2 and Deep-EIoU 60.74/55.47/20.7; FP
  492.6 to 118.9, FN 270.8 to 387.7), the abstract's 63.48 matching an ablation row, TeamTrack
  162,353/170,529 frames, and no code link.
- **KaliCalib** [47]: K = 91, U-Net with a ResNet-18 encoder, 14.5M params, about 9 ms on a GTX
  1080 Ti, centre-of-mass keypoints, DLT + RANSAC with 35 px, `calibrateCamera`, mean training
  homography as the fallback, test MSE 126.61 cm vs a 592.48 baseline (÷4.7; both on the test
  set), 76.5% under 1 m, and the 480/164/84/84 split.
- **RF-DETR keypoint recipe** [16] (26 Jun 2026): `basketball-court-detection-2` v19, 33 landmarks
  on one `court` class, v=2/1/0 flags, `RFDETRKeypointPreview`, 50 epochs, batch 2, grad-accum 2,
  lr 1e-4, resolution 576, `checkpoint_best_total.pth`, `val/keypoint_map_50_95`, `from_checkpoint`
  with the same `num_classes`/`num_keypoints_per_class`, `.predict(..., threshold=0.5).with_nms()`.
- **Shot evaluator** [50] (14 Aug 2026): RF-DETR-small for ball and rim; a fork of a ~9.8k-image
  dataset; mAP50 88.84, P 75.7, R 84.9; ball found in 98.1% (759/774) of in-flight frames;
  `RFDETRKeypointPreview` from rfdetr 1.9.2 on 99% of frames; drop-speed bands of 10–90% for a make,
  100% for an air ball and 0% for a stationary ball; rattle rules; `rattled`/`occluded` tags; 0.35 m
  release separation; 18 in (0.4572 m) rim and median rim width for px/m; no make/miss accuracy
  reported; `aarnavshah12/shot-tracker`.
- **3-second post** [49] (22 Jul 2025): court model and dataset v13, "YOLOv11 keypoint model",
  `remove_noisy_segments`, YOLOv11 pose ankles with a bbox bottom-centre fallback, `PolygonZone`
  trigger plus `FPSBasedTimer`, a 3 s threshold, "wasn't robust", and the AGPL/Enterprise wording.
- **SAM family** [37][38][39][72]: SAM 2.1 sizes and FPS (A100, torch 2.5.1), configs, the
  `vos_optimized=True` predictor API, per-object adds since 12/11/2024, Apache-2.0. SAM 3: Python
  3.12+, PyTorch 2.7+, CUDA 12.6+, 848M params, SAM License, `build_sam3_video_predictor` and
  `handle_request` with `start_session`/`add_prompt`; `facebook/sam3` and `facebook/sam3.1` are
  gated ("manual" per the HF API). SAM 3.1 was released on 27 Mar 2026; Object Multiplex handles up
  to 16 objects per pass, with 16 to 32 FPS on one H100; SAM 3 takes 30 ms per image with 100+
  objects on an H200, with near real time for about five objects.
- **Other papers**: ByteTrack 80.3/77.3/63.1 at 30 FPS on a V100 [29]; BoT-SORT 80.5/80.2/65.0 [30];
  Deep OC-SORT 64.9/63.9/61.3 [31]; SportsMOT 240 videos, 150K+ frames, association is the key
  challenge, "similar yet distinguishable" [33]; SAMIDARE +2.5 HOTA/+4.2 IDF1 on the SportsMOT val
  set [34]; Sports-vmTracking HOTA 72.3%, 0 IDS, >10 points [35]; TrackID3x3 80.75/46.11 TI-HOTA,
  drone 47.92/50.64, PDJ values, 10 keypoints [36]; Koshkina 87.45/79.31/91.4, legibility ResNet34
  94.5%/71.7% F1 with SAM for 20 epochs, 5 px padding, PARSeq max length 2, 3.5σ ×3, 39%
  single-digit, 65.84 vs 91.40 [40]; "Towards Comprehensive Basketball Understanding" (24 Aug 2026)
  with SAM 3 + RF-DETR, Qwen3.5-9B, 8 samples per track, and colour+number pairs [46]; Magera et al.
  derive points and lines from circle correspondences [48]; Caio et al. reconstruct 3D from a known
  projection matrix and link `gabriel-vanzandycke/deepsport` [52].
- **Misc**: D-FINE table and Objects365 warning [19]; the SmolVLM2 and MLX ids exist [42] (HF API);
  ViTPose ids, `VitPoseForPoseEstimation`, top-down, `dataset_index`, docs at v5.20.2 [58]; RTMPose
  AP and latency [56]; MacBook Air M3 Elixir/Ultralytics numbers [60]; PyTorch MPS needs macOS
  14.0+ and `torch.backends.mps.is_available()` [61]; the RF-DETR forum bicubic-AA fallback
  warning [18]; the Pixelsham mirror shows player dataset v6, OCR dataset v3 and "SAM2.1" [7].

### Corrected or tightened

1. **RF-DETR CoreML on Apple (§0, §2.2, §9.1, §9.2, §10.1, §12):** the note said the export guide
   could not be fetched and that no M-series measurement existed. The guide lives at
   `rfdetr.roboflow.com/latest/exports/coreml/`, and the old cookbook URL is a 404. It gives
   first-party **M3 Pro** numbers: `RFDETRNano` 384 px FP16 at 20.8 ms (`CPU_AND_NE`) and 21.2 ms
   (`ALL`) p50. The detector budget of "about 10 to 25 ms" for RF-DETR-S at 512 was therefore
   optimistic, and I rewrote it. I removed the unverified `>=1.9.0` install pin; the docs say
   `pip install "rfdetr[coreml]"` and the extra pins `coremltools>=8.0,<10.0` [17][74].
2. **"YOLO27 ... 'no launch date'"**: the quote is from the YOLO27 page, not [8]/[10]. I added
   [69] with the exact wording.
3. **TrackID3x3 team assignment (§3.2, §4.1):** the note said teams were assigned with colour
   histograms plus Jensen-Shannon divergence. In the paper, JS divergence is used only to merge
   Indoor tracklets. Teams come from starting position (Indoor) or jersey number plus histogram
   similarity (Outdoor) [36].
4. **Basketball-SORT RLLI (§3.2):** I added that B is never given, that Eq. 3 is printed as
   `Cost > α`, `Dist > β` (the direction is ambiguous), and that §4.4 says 170–250 cm [32].
5. **Roboflow README quote (§6.1):** "small and moves quickly" is not in the README. I replaced it
   with the actual wording [2].
6. **Shot evaluator details (§6.3, §6.4):** "net hides the ball for about 2 frames" is a single
   anecdote, not a rule. "≥ 5 points" is not from Roboflow (now marked as our choice), and
   "intersects" became "points toward the rim bounding box" [50].
7. **Koshkina jersey pipeline licence (§4.2, §11 #15):** the code, data and weights are **CC BY-NC
   3.0** (non-commercial) per the repo README. This was missing and matters for a commercial
   product [41]. The venue is CVPR 2024 Workshops (CVsports) [41].
8. **BoT-SORT/McByte (§3.1):** BoT-SORT expects BGR frames. CMC is skipped silently when
   `frame=None`. McByte needs SAM/Cutie only with `enable_mask_manager=True`, never auto-selects
   MPS, and defaults to `enable_cmc=True`. `tracker_id == -1` marks unmatched detections. I added the
   other BoT-SORT defaults [26].
9. **OpenCV `findHomography` (§5.2):** the defaults (`method=0`, `ransacReprojThreshold=3`,
   `maxIters=2000`, `confidence=0.995`) and the flag values are now confirmed from the OpenCV 4.x
   header [70]. The "(unverified)" tag was removed.
10. **PyTorch MPS fallback (§9.1):** generic CPU fallback needs `PYTORCH_ENABLE_MPS_FALLBACK=1` [68].
    The RF-DETR op warning is specific [18].
11. **SAM2 CUDA extension (§3.3):** the extension is optional, and a failed build "can be ignored"
    [37]. I removed "10-player" and "for 10 objects", which the blog does not state [1].
12. **Licences (§7, §12):** mmpose, rtmlib and transformers are Apache-2.0; umap-learn,
    scikit-learn and coremltools are BSD-3-Clause; PARSeq is Apache-2.0 (GitHub licence fields).
    SigLIP, SmolVLM2 and ViTPose checkpoints are Apache-2.0 per the HF API. The DEIMv2 code licence
    is `NOASSERTION` on GitHub (still unverified).
13. **Smaller fixes:** the 640 px ROI covers about 9.3 m, not 9.4 m (arithmetic); I added an
    arithmetic sanity note on the 1–1.5 m per-frame gating radius (§3.4); `basketball-player-
    detection-2` "v8" is the model version; YOLO26-pose is not labelled bottom-up on its page;
    TrackNet numbers are now from the abstract; the RF-DETR-S vs YOLO11-L wording is now quoted;
    court dimensions and rim height are cited [71]; the SAM 3.1 "~7x at 128 objects" figure was
    added [72]; snippet-only claims ([45], [55], [65]) are now marked (unverified).

### Still unverified

- The SoccerNet 2023 best jersey accuracy of 92.85% [45]; the BEEI 94.55% figure and the patent
  claims [55]; the Theseus PoC [65]; the Van Zandycke 2022 details [53].
- Universe dataset licences and contents (HTTP 403); the contents of the basketball notebooks; the
  exact SmolVLM2 size and LoRA settings in Roboflow's fine-tune.
- Every M3 Pro throughput figure other than RF-DETR Nano; SAM2/SAM 3 on MPS; the 24 cm ball
  diameter; every design threshold marked (unverified) in the body.

---

## 13. Addendum: gap pass (2026-10-20)

Cross-note decisions are in `docs/RESEARCH.md` §1A and §7A. This addendum gives the CV-specific
detail. Sources are listed at the end as [G1]-[G14].

### 13.1 CoreML runtime spec for RF-DETR (closes "no I/O spec")
- **Export.** `model.export(format="coreml", coreml_precision="float16")` writes
  `{variant}_fp16.mlpackage` (mlprogram, iOS 16+, built with `torch.export` + coremltools). The export
  is experimental, `dynamic_batch=True` is not supported, and shapes are fixed, so **export one
  package per input size**: one for the full frame (e.g. Small at 512) and one for the ball ROI (e.g.
  a 384 crop) [G1]. Default resolutions: Nano 384, Small 512, Medium 576; `num_queries=300`,
  `num_select=300` for all three [G2].
- **Load.** `ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)`. Compute units are not
  stored in the package and default to `ALL` on every load [G1][G3].
- **Input.** Name from `mlmodel.get_spec().description.input[0].name` (currently `tensors`); shape
  `(1, 3, H, W)`; float32. Preprocess: scale to [0, 1], resize to (H, W) with `antialias=False`,
  normalise with mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225) [G1].
- **Outputs.** Names are inferred by coremltools; match by position: `dets, labels` for detection
  [G1]. In export mode the model returns `(boxes, logits)`: boxes are sigmoid-normalised cxcywh per
  query [G4]. Whether a leading decoder-layer dimension appears is not settled by the source; read
  the shapes from the spec right after export **(unverified)**.
- **Post-processing**, ported from `rfdetr.models.postprocess.PostProcess` [G5]:
  `prob = sigmoid(logits)` → flatten (queries × classes) → top `num_select` scores →
  `query = idx // n_classes`, `label = idx % n_classes` → `box_cxcywh_to_xyxy` → multiply by
  `(W, H, W, H)` of the original frame → clamp. Then
  `sv.Detections(xyxy=boxes, confidence=scores, class_id=labels)` and per-class thresholds.
- **Parity test** (CI, on 100 frames of our footage): PyTorch fp32 vs CoreML fp16 after
  post-processing. Require ≥ 95% of PyTorch boxes above 0.5 confidence to have a CoreML match with
  IoU ≥ 0.95 and |Δconf| ≤ 0.05. Some fp16 drift is expected: Nano's COCO mAP drops 48.0 → 45.1 [G1].
- **ANE facts** [G1]: only fp16 reaches the ANE (`CPU_AND_NE` on fp32 equals `CPU_ONLY`); the first
  fp16 load compiles for the ANE in about 5 s (warm at boot); about 7-8 ops (two-stage query `topk`)
  stay on the CPU.

### 13.2 `trackers` at 15 Hz (closes "tracker parameters undocumented")
- `ByteTrackTracker(lost_track_buffer=30, frame_rate=30.0, track_activation_threshold=0.7,
  minimum_consecutive_frames=2, minimum_iou_threshold=0.1, high_conf_det_threshold=0.6,
  state_estimator_class=XYXYStateEstimator, iou=None)` [G6].
- `BoTSORTTracker(lost_track_buffer=30, frame_rate=30.0, track_activation_threshold=0.7,
  minimum_consecutive_frames=2, minimum_iou_threshold_first_assoc=0.2,
  minimum_iou_threshold_second_assoc=0.5, minimum_iou_threshold_unconfirmed_assoc=0.3,
  high_conf_det_threshold=0.6, enable_cmc=True, cmc_method="sparseOptFlow", cmc_downscale=2,
  instant_first_frame_activation=True, state_estimator_class=XCYCWHStateEstimator, iou=None)` [G6].
- `lost_track_buffer` is "number of 30 FPS frames" and is scaled by `frame_rate`. **At 15 Hz updates,
  pass `frame_rate=15.0`.** `update(detections, frame=None, timestamp=None)`: `timestamp` is "absolute
  time of the current frame in seconds", and `None` means fixed-rate mode (one frame step per call).
  **Pass the frame PTS** when frames are skipped. ByteTrack ignores `frame` and warns if one is given;
  BoT-SORT needs it for CMC [G6]. The exact scaling formula is not documented: add a unit test that
  feeds synthetic tracks at 15 Hz with 1-3 frame gaps and checks that IDs persist.
- **Decision.** Court-space association (≤ 5 players per team, Basketball-SORT reacquisition) lives
  in our own `ByteTracker` (ARCHITECTURE.md). `trackers.BoTSORTTracker` with CMC is used for broadcast
  and as the evaluation baseline. The `iou` similarity hook exists, but its callable signature is
  undocumented.
- `sv.Position.BOTTOM_CENTER` is **verified**. The enum is CENTER, CENTER_LEFT, CENTER_RIGHT,
  TOP_CENTER, TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_CENTER, BOTTOM_RIGHT, CENTER_OF_MASS [G7].
- **Packaging conflict.** `trackers` 2.6.1 requires `opencv-python>=4.8.0` [G8], and the headless
  wheel installs the same `cv2`. Use one wheel, `opencv-python` (RESEARCH.md D12). `supervision`
  0.31.0 (MIT) declares no OpenCV dependency [G9]. rfdetr 1.11.2 requires `torch>=2.2.0`,
  `transformers>=5.1.0,<6.0.0` and `supervision>=0.29.0` [G10].

### 13.3 M3 Pro throughput plan (closes "one measurement only")
- **Hardware.** The dev machine is the binned M3 Pro: 5 performance + 6 efficiency CPU cores, a 14-core
  GPU and 18 GB (`sysctl`, `system_profiler`, 2026-10-20). Apple's launch numbers (12-core CPU,
  18-core GPU) describe the top bin [G11].
- **Arithmetic.** Nano 384 fp16 costs 20.8 ms on `CPU_AND_NE` and 33.9 ms on `CPU_AND_GPU` [G1].
  Full frame at 15 Hz on the ANE = 0.32 s of accelerator time per second. A 30 Hz ROI pass at Nano cost
  adds 0.63 s/s. On the GPU the same ROI pass would need 30 × 33.9 ms = 1.02 s/s, which **does not
  fit**.
- **Placement (proposal).** ANE serial queue: full frame at 15 Hz plus the ROI pass at 15 Hz on the
  alternate frames, which gives 30 Hz ball observations at ≈ 0.63 s/s if both cost what Nano costs.
  GPU (MPS): SigLIP and OCR. CPU: tracker, homography, events. One process per CoreML model, because
  Core ML's synchronous prediction API is not thread-safe (the async API is) and concurrency gains are
  workload-dependent [G12].
- **Benchmark matrix (Phase A, first task).** ≥ 10 minutes per cell, p50/p95/p99, while SigLIP and OCR
  run: {Nano 384, Small 512, Medium 576, ball-only ROI 384 and 640} × {`CPU_AND_NE`, `CPU_AND_GPU`,
  `ALL`} × {fp16, fp32 on the GPU} × {1 model, 2 models in separate processes (ANE+ANE, ANE+GPU)}.
  Log `pmset -g therm` every 10 s (this Mac reports thermal and performance warning levels through it)
  and the frame-drop rate.
- **Degradation ladder (proposal).** On p95 capture→tracks latency over a 10 s window, 60 s hysteresis:
  L1 (> 50 ms or any thermal warning): pose off, SigLIP refresh 2 s, jersey reads every 10 frames.
  L2 (> 66 ms): full-frame detection 10 Hz. L3 (> 80 ms): ROI 10 Hz, Nano 384 for the full frame.
  L4 (> 100 ms for 30 s, or > 5% dropped frames): near-live mode, 3 s delay.

### 13.4 CV constants registry (closes "constants without status")
Every constant goes in `EventEngineConfig` as `(value, source, status)`:
- make drop band 10-90% of free fall: **sourced** [G13]; Roboflow reports no make/miss accuracy;
- ≈ 100% of free fall = an air ball behind the rim that never touched the net: **sourced** [G13]; rim
  bounces are handled by the `rattled` logic, not by this rule;
- release separation 0.35 m ≈ 1.15 ft with the ball rising: **sourced** [G13];
- 98.1% in-flight detection = 759 of 774 frames of Roboflow's own test footage, with a detector trained
  on a fork of the "Basketball Shooting Robot" dataset (~9.8k images); validation recall 84.9% [G13].
  It is **not** a game-footage figure;
- the `ball-in-basket` class comes from the player-identification post [1], not from the shot
  evaluator, which uses only ball and rim [G13];
- holder radius 0.5 m ≈ 1.6 ft: **unverified**; homography gate 1.0 ft: **unverified**; ≥ 5 parabola
  points: **unverified** (Roboflow gives no count) [G13].
These cues feed a per-event logistic model with hand-set weights, Platt/isotonic calibration and
hedge-only mode until certified (RESEARCH.md §7A.2).

### 13.5 Scoreboard reader (fixed camera)
At setup the operator clicks the scoreboard's 4 corners (homography rectification) and marks the digit
cells. An 11-class CNN (0-9, blank) trained on synthetic seven-segment and LED renders plus venue crops
reads each cell at 5 Hz. PP-OCRv5 remains an option for broadcast scorebugs: PaddlePaddle on macOS
supports Apple Silicon on the CPU only, for Python 3.9-3.13 (`paddlepaddle==3.3.0`) [G14]. Its
latency on the M3 Pro is unmeasured.

Sources for this addendum:
[G1] RF-DETR docs, "Native CoreML Export (.mlpackage)". https://rfdetr.roboflow.com/latest/exports/coreml/ and https://github.com/roboflow/rf-detr/blob/develop/docs/exports/coreml.md
[G2] rf-detr source, `src/rfdetr/config.py`. https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/config.py
[G3] coremltools guide, "Model prediction". https://apple.github.io/coremltools/docs-guides/source/model-prediction.html
[G4] rf-detr source, `src/rfdetr/models/lwdetr.py` (`forward_export`). https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/models/lwdetr.py
[G5] rf-detr source, `src/rfdetr/models/postprocess.py`. https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/models/postprocess.py
[G6] trackers API reference. https://trackers.roboflow.com/latest/api/trackers/
[G7] supervision source, `src/supervision/geometry/core.py`. https://github.com/roboflow/supervision/blob/develop/src/supervision/geometry/core.py
[G8] trackers on PyPI (JSON). https://pypi.org/pypi/trackers/json
[G9] supervision on PyPI (JSON). https://pypi.org/pypi/supervision/json
[G10] rfdetr on PyPI (JSON). https://pypi.org/pypi/rfdetr/json
[G11] Apple Newsroom, M3 family (2023-10). https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/
[G12] Apple, WWDC23 "Improve Core ML integration with async prediction". https://developer.apple.com/videos/play/wwdc2023/10049/
[G13] Roboflow, "How to Build an AI Basketball Shot Evaluator" (re-read 2026-10-20). https://roboflow.com/blog/how-to-build-an-ai-basketball-shot-evaluator
[G14] PaddlePaddle macOS pip install. https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/macos-pip_en.html
