> Research notes compiled on 2026-11-05 by a research agent of the cloud session (round 8; docs/research/cross_game_detector.md §13), lightly reviewed. Figures marked as estimates or *unverified* were not checked; the sources are cited in place. Paths to that session's scratchpad (crop sheets, drawings) are not in the repository.

# Making the broadcast detector more accurate: training and method ideas

Research for courtvision, 2026-11-05. Inputs: the code (`tools/student_finetune.py`, `courtvision/training/train_student.py`,
`courtvision/nba/referee.py`, `courtvision/nba/analyze.py`, `courtvision/perception/{student,tracker,teams,pipeline,broadcast_court,court_search}.py`,
`tools/detector_bench.py`); the research note `docs/research/cross_game_detector.md` (§5, §8-§13, including the round-8 plan committed this morning); papers and pages
(section 9). I also ran two light checks on the training labels in this container: a JSON count, and about 150 image crops viewed by eye.
No model was run, nothing was trained, and the repository is unchanged.

Every number either cites the research note's section, cites a paper, or comes from this session's checks (§2). Numbers marked
*estimate* are my own and are unverified.

---

## 0. Summary

| Rank | Experiment | Aim | Cost of one try | Main evidence |
|---|---|---|---|---|
| 1 | **E1. Referees of other eras become positives.** Promote SportsMOT's ignored and missed officials to person labels with a precision gate, and add a shirt-brightness augmentation for referee boxes | (b) boxed | relabel ~20 min CPU, 1 run | this session's label audit (§2); C's lineage (§10.15, §9, §11); CityPersons on ignore regions |
| 2 | **E2. Referees told apart within each game.** Over-cluster on-court people, take the leftover small group as the referees, decide per track by votes | (b) told apart | no training | Liu et al. 2007 (referee as the leftover cluster); Lu et al. 2013; SoccerNet-GSR tracklet votes; the learned check's failure (§10.10-10.12) |
| 3 | **E3. Round 8's court-geometry person labels (K2), with five label fixes and a negative-weight variant** | (a) | in flight (round 8) | court-people precedent (§10.9, §10.13); Hoiem et al. 2006; Vandeghen et al. 2022 doubt band |
| 4 | **E4. Ball pseudo-labels on SportsMOT** (older NBA at Oracle Arena, NCAA, FIBA; 720p, 25 fps) | (c), the 2017 game | ~20-45 min CPU, 1 run | M1 against M1-ctrl (§10.7.1); Noisy Student |
| 5 | **E5. Better ball pseudo-labels.** Flip ensembles, confidence-scaled weights (all pseudo labels now weigh exactly 0.5), hard negatives | (c) | ~1-2 h CPU, 1 run | Data Distillation; Jin et al. 2018; WASB's hard-sample mining; T2 (§8.2) |
| 6 | **E6. Weight EMA in every run; online Mean Teacher on the pseudo folders once K2 has reported** | all | EMA free; MT about +10 % per step (*estimate*) | Mean Teacher; Unbiased and Soft Teacher; SWA; frozen BN is the standard setting |
| 7 | **E7. Flicker mining** (detections isolated in time) for frames without a court fit | (a) | ~45 min CPU, 1 run | Jin et al. 2018; RoyChowdhury et al. 2019 |
| 8 | **E8. A court auxiliary head used only in training** (court-floor mask; same ONNX interface) | (a) | code change, 1 run | HyperLearner (Mao et al. 2017); SDS-RCNN (Brazil et al. 2017) |

Not recommended now (§6): a court-mask input channel; a stride-2 or HRNet-style ball head; motion-compensated history frames; a
referee class in the detector; adapting on the test game itself.

**Why E1 ranks first.** The training labels show a mechanism for v2's main cost, the 2017 game's light-grey referees falling from 84 % to 23 %
(§10.13). First, C started from T5, which was trained to treat every referee as background (26 of 506 referees found, §12). Second,
C's only referee *positives* are 2024-25 dark-shirted referees: the 3 Roboflow games and the Lakers @ Warriors teacher folder
(§10.9, §10.12). Third, the officials in SportsMOT are mostly *ignore* boxes, which teach nothing either way, and some are plain
background. Those officials include the light-grey-shirted referees at Oracle Arena on CSN Bay Area, the 2017 game's arena and
uniform era. So nothing in v2's data teaches light-grey referees as people. E1 fixes that with data that is already in the container.

**Coordination with round 8.** The K0, K1 and K2 runs reuse C's SportsMOT ignore regions, so on their own they will not bring back
the 2017 referees. E1 and E4 are independent additions; E3 refines K2; E6's EMA can be switched on in any run at no cost.

---

## 1. Constraints these ideas must fit

* **Network and interface.** `perception/student.py:build_net`: ResNet-18 with an FPN to stride 4, at a 544x960 input, which gives 136x240 cells. It takes
  9 input channels: frames t-2, t-1 and t, past only, so it runs live. It has five CenterNet outputs: `ball_hm`, `ball_off`,
  `person_hm`, `person_wh`, `person_off`. The ONNX export (`student_finetune.export`), its sparse-head runtime
  (`perception/student_onnx.py`), Core ML and the browser app (`browser/runtime.py`) all depend on these input and output names
  and shapes. Any idea that changes them costs work in four places.
* **Training recipe** (`student_finetune.train`). It starts from an exact copy of the main model (`training/student_from_onnx`).
  Batch norms are folded and frozen; the stem and `l1` are frozen. AdamW at 2e-4 with cosine decay, batch 4, 512-px crops, zoom 0.7-1.4,
  hue ±30° on samples without a ball label (`FineTuneDataset`), temporal dropout 0.25, folders sampled equally, 4,000 steps.
  Losses come from `train_student.focal_loss` and `masked_l1`: the CenterNet focal loss, with per-sample weights and a per-cell
  `pixel_weight`, which is how ignore regions work. Pseudo ball labels are capped at weight 0.5 (`PSEUDO_MAX_WEIGHT`).
* **Budget.** On the Mac's GPU (MPS) a 4,000-step run takes about 22 min (0.33 s a step, §10.6, §10.13). On this container's CPU the
  logs show 2,509 s for T5 and 2,179 s for R4b (`runs/models/ft-t5.log`, `ft-r4b.log`), so the 40-minute budget is already spent
  there. Extra compute per step therefore fits only on the Mac, about 1.8 times the current cost, unless the step count drops.
* **Data here and on the Mac.** In the container: `data/ft/{mot_train_ign, mot_val_ign}` (SportsMOT with ignore regions),
  `coco3r` (3 Roboflow games with referees), `bard_calib_r2` (390 BARD clips with ball labels), and
  `bard_check_teacher` with `bard_check2_teacher` (the 1,153 BARD check frames). Also `data/hfcoco/heldout*` (the 5 held-out
  Roboflow games), `runs/models/app-v1.onnx` (v1, which boxes everyone), v2 at `courtvision/server/static/nba-student.onnx`, and
  round 8's `bard_calib_r8` (being built). Only on the Mac: the Lakers @ Warriors teacher folder and camera track, and the
  Cavaliers @ Warriors test clips and camera track.
* **What already filters the crowd downstream.** When the court is registered, the pipeline drops players more than 4 ft (NBA analysis,
  `analyze.py`) or 6 ft (default, `PerceptionConfig.court_margin_ft`) outside the court. Detector-level crowd boxes therefore matter
  in four places: frames without a court fit (about half of the 2017 game, `made_shots_at_the_rim.md` §4); the overlay; the
  ball tracker's "people near the ball"; and the team clustering's first crops.
* **The referee path.** The shirt rule is the default; the learned check and its guard are options (`nba/referee.py`,
  `analyze.RefereeCheck`). Referee calls are made per detection, before tracking. The tracker then refuses to link a player
  detection to a referee track (`perception/tracker.py`: `cost[cls_t != cls_d] = _INF`), so a call that flips between frames
  splits the track.

---

## 2. New evidence from this session (light checks, no model run)

**(a) Ignore boxes per SportsMOT broadcast.** This is a JSON count over `mark-ignore`'s output. The ignore boxes came from the v1 app model at 0.3, with the
learned referee check or the players' foot hull, per the folder's `meta.json`. Script:
`scripts/ignore_by_broadcast.py`.

| Broadcast | Samples | Share of samples with ignore boxes | Ignore boxes per sample | Labelled players per sample |
|---|---:|---:|---:|---:|
| NBA, Warriors at Oracle Arena, CSN Bay Area (`v_2j7kLB-vEEk`) | 2,133 | 0.85 | **1.77** | 8.51 |
| NBA, Lakers at New Orleans, re-streamed local feed (`v_4LXTUim5anY`) | 1,234 | 0.84 | 1.62 | 8.73 |
| FIBA Women's Asia Cup (`v_-6Os86HzwCs`) | 758 | 0.87 | 1.51 | 9.65 |
| FIBA, USA v Russia (`v_BgwzTUxJaeU`) | 705 | 0.81 | 1.56 | 9.42 |
| FIBA Women's World Cup 2018 (`v_5ekaksddqrc`) | 1,405 | 0.71 | 1.01 | 9.43 |
| NCAA, Kentucky v Wichita State (`v_00HRwkvvjtQ`) | 1,775 | 0.66 | 0.87 | 9.55 |
| FIBA Asia, Philippines v Palestine (`v_4r8QL_wglzQ`) | 295 | 0.63 | 0.72 | 9.23 |

**(b) Who the ignore boxes are.** I viewed random crops by eye (seed 0). Script `.../scratchpad/scripts/ignore_sheet.py`; sheets in
`sheets/`.

* Oracle Arena (36 crops): about 28 officials in **light-grey shirts with black shoulders and black trousers**. About 6 are spectators or
  coaches (a man in a suit, seated fans in light shirts), and 2 are unclear. The network's branding is CSN Bay Area, which was renamed NBC Sports Bay Area on 2 April 2017
  ([NBC Sports press release](https://www.nbcsports.com/pressbox/press-releases/nbc-sports-regional-networks-to-rename-california-based-properties-to-feature-nbc-sports-brand)).
  So this footage predates April 2017: the same arena and, very likely, the same referee uniform as the 2017 Finals test game.
* Lakers at New Orleans (24 crops): about 14 officials in light shirts. About 10 are courtside fans, seated or standing, or bench people. In
  two whole frames of this broadcast, at least two people who look like officials are neither labelled nor ignored: light shirts and dark trousers, one at the left
  baseline and one at the right sideline. They were therefore taught as **background**. This feed is a low-angle camera with fans
  standing across the bottom of the frame.
* NCAA (24 crops): 24 striped referees. FIBA Women's World Cup (24 crops): about 23 FIBA referees in grey shirts with black shoulders.

**Reading.** The SportsMOT officials, in four uniform families (older NBA light grey, NCAA stripes, FIBA grey and black, and others), are
mostly *ignored*, sometimes *background*, never *positive*. CityPersons found that treating unsure regions as background hurts and
ignoring them helps ([Zhang et al. 2017](https://arxiv.org/abs/1702.05693), +1.33 MR points from ignore handling). But an ignore box
keeps an already-learned suppression in place. v2 descends from T5, which had learned that suppression. This is consistent with the
2017 referee numbers: C 23 %, from T5; A 35 % and round 4 49 %, both from the app model (§10.13). It is a hypothesis that E1 tests;
the crop counts are small samples.

---

## 3. Answers to the five questions

### Q1. Semi-supervised and self-training detection beyond plain pseudo-labels: what fits the budget?

What the literature offers, with what each costs here:

| Method | Idea | Cost per step | Fits on MPS (22 → 40 min)? | Fits on the container CPU (36-42 min now)? |
|---|---|---|---|---|
| Offline pseudo-labels: [Noisy Student](https://arxiv.org/abs/1911.04252), [STAC](https://arxiv.org/abs/2005.04757) (current) | teacher labels once, student trained with augmentation | 0 | yes | yes |
| [Data Distillation](https://arxiv.org/abs/1712.04440) (Radosavovic et al. 2018) | ensemble one model's predictions over several transforms of each unlabelled image (flip, scales) into pseudo-labels; beat COCO-only supervision for keypoints and detection | 0 (labelling 2-3 times offline) | yes | yes |
| Weight EMA: the [Mean Teacher](https://arxiv.org/abs/1703.01780) teacher; [SWA](https://arxiv.org/abs/1803.05407) | average weights along one training run | ~0 | yes | yes |
| Online teacher: [Unbiased Teacher](https://arxiv.org/abs/2102.09480), [Soft Teacher](https://arxiv.org/abs/2106.09018), [Dense Teacher](https://arxiv.org/abs/2207.02541), [Humble Teacher](https://arxiv.org/abs/2106.10456) | an EMA teacher labels weakly augmented views, the student learns on strongly augmented views; thresholds or soft scores | one forward pass without gradients on the unlabelled samples. Restricted to the pseudo folders' samples (about 1/5-1/3 of a balanced batch) that is about +10 % (*estimate*) | yes | borderline (~45 min, *estimate*) |
| [CSD](https://proceedings.neurips.cc/paper_files/paper/2019/file/d0f4dae80c3d0277922f8371d5827292-Paper.pdf) (Jeong et al. 2019) | consistency between an image's predictions and its flipped copy's, plus background elimination | +1 forward pass (stop-gradient branch), about +33 % (*estimate*) | yes | no |
| Classic SSOD with a separate unlabelled batch | student on labelled and unlabelled batches, teacher on unlabelled | ~2.3 times (*estimate*) | no (~50 min) | no |
| Domain-adaptive teachers: [UMT](https://arxiv.org/abs/2003.00707), [Adaptive Teacher](https://arxiv.org/abs/2111.13216), [Probabilistic Teacher](https://arxiv.org/abs/2206.06293) | teacher-student plus adversarial or feature alignment, or uncertainty modelling | discriminator and extra passes | borderline | no |

Reported effects worth knowing:

* Unbiased Teacher: +6.8 mAP over the previous best at 1 % COCO labels (abstract).
* Soft Teacher: +3.6 mAP even with full COCO labels plus 123K unlabelled images (40.9 to 44.5; abstract). An online teacher helps
  even when labels are plentiful.
* [Unbiased Teacher v2](https://arxiv.org/abs/2206.09500): anchor-free detectors gain much less from existing SSOD methods, and
  pseudo-labelling box regression misleads them. CenterNet is anchor-free. **Use pseudo-labels for the heatmaps; train `person_wh` and offsets only from
  confident peaks, or not at all.**
* Sports-specific: [Vandeghen et al. 2022](https://arxiv.org/abs/2204.06859) did teacher-student training for soccer players and the ball on
  SoccerNet-v3.
  * With all labels, the test mAP rose from the teacher's 51.0 to 52.3 using all their unlabelled data (52.0 with only 10 extra games).
  * With 10 % of the labels, validation mAP rose from 39.5 to 43.7-44.3, and to 45.1 after a second iteration.
  * Their best loss used a **doubt band**: teacher scores between a low and a high threshold are ignored, and in a variant the weight
    ramps up across the band. That maps one-to-one onto `ClipDataset`'s ignore regions.
  * Their best thresholds were very high (low 0.9, high 0.99-1). Their Faster R-CNN scores are not calibrated like our CenterNet
    peaks, so tune the band here rather than copy it.
* For video: [RoyChowdhury et al. 2019](https://arxiv.org/abs/1904.07305) adapt detectors with high-confidence detections plus
  hard examples found by a tracker, and soft labels through distillation. [Jin et al. 2018](https://arxiv.org/abs/1808.04285) mine
  hard negatives as detections *isolated in time* ("flickers") and hard positives from track gaps.
  [ARTHuS](https://openaccess.thecvf.com/content_CVPRW_2019/papers/CVSports/Cioppa_ARTHuS_Adaptive_Real-Time_Human_Segmentation_in_Sports_Through_Online_Distillation_CVPRW_2019_paper.pdf)
  distils a per-match network online in sports.

**Frozen BN is not an obstacle.** Unbiased Teacher's base config
([configs/Base-RCNN-FPN.yaml](https://github.com/facebookresearch/unbiased-teacher/blob/main/configs/Base-RCNN-FPN.yaml)) keeps
detectron2's default `MODEL.RESNETS.NORM = "FrozenBN"`
([defaults.py](https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py)), so the canonical SSOD
backbones run with frozen batch norms. Because v2's batch norms are folded, teacher and student see identical normalisation, and
strong augmentation cannot shift BN statistics (reasoning).

**Recommendation for this codebase:**
1. Keep offline self-training and improve its labels: flip ensembles, confidence-scaled weights and a doubt band (E5, E3).
2. Switch on a weight EMA in every run (E6).
3. Try an online Mean Teacher only on the pseudo-folder samples, and only after round 8's K2 shows whether geometry-gated person
   pseudo-labels help at all. If offline geometry labels do not help, an online version of them is unlikely to (reasoning).

The repo's own record supports label quality as the lever: a better ball labeller gave T2 +2.2 points on BARD (37 frames to 10,
p 9.9e-5, §8.2), and a further self-training round gave M1-ctrl a gain on BARD (22 to 6, p 0.0037, §10.7.1).
The weight soups failed (§12), but they averaged *different* runs that had moved in unrelated directions. An EMA along one run stays
in one basin, so §12 does not predict it (reasoning).

### Q2. Camera geometry as free supervision: prior work and pitfalls

**Prior work.** I found geometry used mainly as an inference-time filter or prior, and **no paper that turns a court homography into
detector training labels** (my searches may have missed one).

* [Lu et al. 2013 (TPAMI)](https://www.cs.ubc.ca/~murphyk/Papers/weilwun-pami12.pdf) track NBA broadcast players with a court
  homography. Their detector's false positives were mostly "spectators and referees", removed by a per-game colour classifier (§Q3).
* [Arbués-Sangüesa et al. 2019](https://arxiv.org/abs/1907.04637) restrict basketball detection to the court.
* The [SoccerNet-GSR](https://arxiv.org/abs/2404.11335) baseline places each box's bottom centre on the pitch through the calibration.
* [Theiner et al. 2022 (WACV)](https://arxiv.org/abs/2110.11107) chain field registration, detection and team assignment, and study
  how errors propagate between them.
* The camera-geometry prior itself, an object's image height given where its base stands and the camera, is
  [Hoiem et al. 2006, "Putting Objects in Perspective"](https://ri.cmu.edu/publications/putting-objects-in-perspective)
  (CVPR 2006; IJCV 2008).
* The repo already does this as training supervision: `student_finetune.court_people` keeps people within 3 ft, ignores 3-10 ft and
  drops beyond. Round 8's `court-persons` uses +2 ft positive, 2-8 ft ignore, background beyond 8 ft (§13).

**Pitfalls**, each with the codebase hook that handles it:

1. **Wrong fits.** At Oracle Arena in 2017 the floor mask fails because the crowd's shirts match the floor, and "the search fits
   the court into the stands" (`made_shots_at_the_rim.md` §4). Geometry labels on such a fit turn the crowd into positives. Use only
   frames with status `tracking` and coverage of at least ~0.5 (`BroadcastTrackerConfig.min_coverage` is 0.40,
   `reacquire_coverage` 0.5). `CameraTrack.load` already drops `held` frames.
2. **The foot point is not on the floor.**
   * Seated courtside fans: the box bottom is a seat or table.
   * Boxes truncated by the bottom of the image: near-side fans. Their foot point projects *into* the court.
   * Jumping players: the box bottom is above the floor and projects farther from the camera, so a player near the far sideline
     may land "off" the court.
   * Occlusion by the scorer's table.
   Fixes: ignore boxes that touch the image border. Compare the box height with the standing height the camera predicts at that
   foot point: `analyze.rectify_box` already computes a 6.6-ft standing box. Decide per track, by the median foot point over a
   ByteTrack track, not per frame.
3. **People who belong off the court.** The lead official works the baseline, out of bounds
   ([Official (basketball)](https://en.wikipedia.org/wiki/Official_(basketball))). Inbounders, players chasing loose balls, players
   waiting at the scorer's table, and coaches in the coaching box near the sideline are also off it. The 2-8 ft ignore band must
   stay ignore, not background, for **standing** people.
4. **People on or near the court who are not players or referees.** Photographers along the baseline; dancers, mascots and mop
   crews during timeouts; courtside fans a few feet from the sideline (I could not verify the NBA's minimum seat distance). Label
   only live-play frames. BARD clips are mostly live events; elsewhere use the pipeline's broadcast gate or scorebug clock.
5. **Seated against standing in the near band.** CityPersons labels sitting persons as ignore regions (paper above). For aim (a),
   though, seated courtside fans should be background, since we do not want them boxed. A standing-height ratio below about 0.75
   (*estimate*; check on crops) is a usable test.

### Q3. Referee against player in broadcasts: published approaches, and era and uniform shifts

* **Per-game colour classifier with an "other" class.** [Lu et al. 2013](https://www.cs.ubc.ca/~murphyk/Papers/weilwun-pami12.pdf):
  a logistic regression on RGB colour histograms, classes Team A, Team B and *other* (referees and spectators), trained from "a
  handful of examples" per game. On basketball it lifted detection precision from 73 % to 97 % while recall went from 78 % to 74 %.
* **Referees as the leftover cluster.** [Liu et al. 2007 (BMVC)](https://bmva-archive.org.uk/bmvc/2007/papers/paper-70.pdf),
  soccer, unsupervised per video. They note that referees "do not clearly form a cluster" because there are few of them, and that a
  direct clustering lets them be absorbed into a team cluster. So they over-cluster with EM (K large enough), merge to 4 groups,
  take the two largest as the teams, and name the remaining group farther from the team models "referee", the other "outlier".
  Referees were labelled correctly 82.9 % and 97.1 % of the time on two matches.
* **Supervised referee classifier plus unsupervised teams.** [Koshkina et al. 2021](https://arxiv.org/abs/2104.10068), hockey:
  a small CNN classifies referees "since the referee uniform is consistent across games" (98 % accuracy, 93 % precision, 96 %
  recall), then contrastive learning clusters the teams without labels. Their assumption, one referee uniform across games, is
  exactly what fails between 2017 and 2024-25 here.
* **Per-detection role plus tracklet vote.** The [SoccerNet-GSR](https://arxiv.org/abs/2404.11335) baseline predicts a role
  (player, goalkeeper, referee, other) for each detection, takes a majority vote per tracklet, and clusters the tracklets of the
  player role into two teams with K-means on averaged re-ID embeddings.
* **Referees as background.** [Istasse et al. 2019](https://arxiv.org/abs/1907.01058), basketball: associative embeddings for
  teams without per-game training, with referees treated as background. Their failure cases: a referee taken for a player, and a
  player dropped as a referee, "where referees shirts or pants are visually similar to a team".
* **Common practice for teams.** [roboflow/sports](https://github.com/roboflow/sports) (`sports/common/team.py`): SigLIP
  embeddings, UMAP to 3 dimensions, then K-means with 2 clusters. Referees come from a detector class, not from clustering. This repo
  has the same recipe as an option (`perception/teams.py:SiglipEmbedder`).

**Handling era and uniform shifts.** Make the decision *within the game* and *per track*: E2. Give the detector referee positives from
every era it can see (SportsMOT's older NBA, NCAA and FIBA officials) and augment shirt brightness: E1. Keep players in dark, grey or
white jerseys as negatives for any learned check: the `--players` refit of §10.12 already does this, but appearance alone did not pass
the 2017 game. The guard's count prior, about 3 referees a frame, is a good naming constraint for a cluster, but by itself it does not
tell a referee from a fan at the baseline (§10.11).

### Q4. The ball: what would improve a stride-4 CenterNet ball head on 3 stacked frames

**Facts about our errors.** The models almost never return *no* candidate above 0.08: missed 0.0 % for the fine-tunes and up to
1.8 % for the app model, against wrong 10-22 % (§5, §8.2, §10.2). The median error of a hit is about 2 px of 1080p. So the remaining errors are mostly **ranking errors**: the top
peak lands on something else. Resolution limits are not the main problem.

**From the literature:**

* [WASB, Tarashima et al. 2023 (BMVC)](https://arxiv.org/abs/2311.05237).
  * Setup: an HRNet with the stem's strides removed; 3 frames at 288x512 with outputs at input resolution; position-aware targets;
    online tracking that filters candidates by a constant-acceleration prediction.
  * Their own NBA ball set (from the NUST-NBA181 videos): removing stem strides raised F1 from 77.4 to 80.6.
  * Hard-to-Localize Sample Mining (real-valued targets and a quality focal loss, only for mined hard samples) added +0.3 to +0.6
    F1 on soccer, tennis and badminton.
  * Their online tracking did not help on soccer or badminton.
  * Our input is already 544x960 at stride 4, and our offline linker (`analyze.link_ball`) is a full dynamic programme.
* [TrackNetV3](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf) (Chen & Wang 2023): an estimated background frame as an
  extra input, a rectification network that inpaints missing points, and mixup. Fixed cameras.
* [TrackNetV4](https://arxiv.org/abs/2409.14543) (Raj et al. 2024): learnable motion-attention maps from frame differences, fused
  with features. Tennis and badminton; the search results report it as ICASSP 2025, which I did not verify. With a panning
  broadcast camera, raw frame differences carry the pan.
* [BlurBall](https://arxiv.org/abs/2509.18387) (2025, table tennis): labelling the ball at the *centre* of its blur streak improves
  detection. Our labels are box centres from RF-DETR or the student, which are probably streak centres already (reasoning).
* [TOTNet](https://arxiv.org/abs/2508.09650) (2025): occlusion-aware 3D convolutions, offline.
* [DeepBall](https://arxiv.org/abs/1902.07304) and [FootAndBall](https://arxiv.org/abs/1912.05445): FPN-style context for a small ball.
* [Vandeghen et al. 2022](https://arxiv.org/abs/2204.06859): semi-supervised player and ball training in soccer (Q1).

**For this head, in order** (expected value against cost, my ranking):

1. **Broader footage for the ball.** E4: SportsMOT has 7 broadcasts with no ball labels at all. The repo's evidence: footage from a
   broadcast game helped the 2017 ball, while more NBA.com clips did not (§10.7.1: M1 against M1-ctrl on the 2017 game, 21 frames to 7,
   p 0.0125; M1-ctrl against round 2 there, 5 to 16, p 0.027).
2. **Better and graded pseudo-labels** (E5). Every pseudo ball label currently weighs exactly 0.5: `label_folder` gives detected
   picks 0.75-1.0 and filled frames 0.5, then `ClipDataset` caps pseudo weights at 0.5. Add flip ensembles and a doubt band.
3. **Explicit hard negatives at distractor peaks** (E5): an extra focal-loss weight there, through `pixel_weight`.
4. WASB-style real-valued targets only for mined hard-to-localise frames. Small, and our metric is not localisation-bound.
5. Higher-resolution features for the ball head (§6). Worth it only if an error audit shows small or blurred balls missed.
6. Motion cues with camera compensation (§6). Speculative here.

### Q5. Fewer crowd false positives at the detector level

* **Label semantics first.** Crowd boxes are learned or suppressed by what each folder calls background, ignore or positive:
  SportsMOT's hull plus referee ignore, Roboflow's unlabelled crowd, Lakers @ Warriors' court-people bands. E1 must not promote fans
  (§2: up to ~40 % of ignore boxes in one broadcast are fans); E3 must not make seated courtside fans positive.
* **Hard-negative mining.** The CenterNet focal loss already concentrates on confident negatives
  ([Lin et al. 2017](https://arxiv.org/abs/1708.02002); [OHEM](https://arxiv.org/abs/1604.03540) is the region-sampling ancestor).
  Explicit mining adds negatives the labels lack: off-court people under geometry (E3), and flickers on frames without a fit (E7).
  An image-level version weights the balanced sampler by v2's false positives per sample (E7).
* **Focal-loss weighting.** A cheap variant is a ×3 `pixel_weight` inside geometry-certain "off" boxes (E3's K2b). It is cheap to test;
  its risk is suppressing bench-side players.
* **Context or court-mask input channel.** Not recommended (§6): it changes the 9-channel interface and the frozen stem, and it needs
  a court fit at inference, which is lost on about 20-50 % of frames. The pipeline already drops off-court players when a fit exists.
  The training-only auxiliary head (E8) gets most of the idea at no inference cost.

---

## 4. The ranked experiments

Each experiment starts from main model v2 (`student_from_onnx` of the app's export) with C's recipe (4,000 steps), changes one thing,
and is scored once.

**Standard measurement**, in this container:

```
.venv/bin/python tools/detector_bench.py --coco data/hfcoco/heldout_ann.json \
    --models courtvision/server/static/nba-student.onnx,runs/models/<cand>.onnx --sparse --json runs/bench/coco_heldout_<cand>.json
.venv/bin/python tools/detector_bench.py --ball-folder data/ft/bard_check_teacher,data/ft/bard_check2_teacher \
    --models courtvision/server/static/nba-student.onnx,runs/models/<cand>.onnx --sparse --json runs/bench/ball_check12_<cand>.json
    (and again with --link for the linked path)
```

On the Mac, §10.1's sets: the Lakers @ Warriors held-out clips, and the 60 Cavaliers @ Warriors clips with §10.2's split of people
on and off the court; then the pipeline for candidates that pass.

**v2's reference numbers.** The 5 Roboflow games: referees 426 of 506, players-and-referees AP50 0.892, boxes on nobody 0.19 a frame,
ball 68.0 %. BARD: 91.4 % on the Mac's labels. The 2017 game: referees on the court 23.3 % of 116, off the court 29.6 % of 226,
players 84.8 %, ball 79.2 % top candidate and 78.4 % linked. Lakers @ Warriors: players 97.9 %, referees 98.7 % (§10.13).

### E1. Referees of other eras become positives (aim b, boxed)

**What to change.**

* `tools/student_finetune.py`: a new `classify_unlabelled(...)` beside `foot_hull_ignore`, and `mark_ignore(..., promote=True)`.
  For each SportsMOT sample, run v1 (`runs/models/app-v1.onnx`, which boxes everyone, at 0.3). Take every person box that overlaps
  no labelled player (IoU < 0.3), whether it is now ignored or background. **Promote** it to a person label when all of these hold:
  1. *Standing on the floor.* Fit the labelled players' box height against their foot row in the same frame (robust line; under a
     pinhole camera on a flat floor, image height grows roughly linearly with the foot row, after Hoiem et al.), and require a ratio
     in [0.8, 1.25] (*estimate*). This rejects seated fans and the raised stands.
  2. Not touching the image border.
  3. Foot row no higher than the farthest labelled player's foot row minus 10 % of the frame height (*estimate*).
  4. The shirt rule **or** the learned check calls it a referee (`nba.referee.referee_calls`). On the 2017 game the rule was right
     on 66 of 80 hand-labelled boxes and the learned check on 32 (§10.12). The rule is the better cue for this era, but how many
     light-grey officials it catches is not measured; the audit counts it.

  Everything else keeps its current label: the change only promotes, never demotes. Write the promoted boxes into `persons` with a
  flag such as `"promoted": [indices]`, and record the counts in `meta.json`.
* **Precision gate before any training.** Draw 100 random promoted crops per broadcast, as in §2, and require ≥ 95 % officials by eye;
  tighten condition 1 until it holds.
* **Variant E1b: shirt-brightness augmentation.** In `courtvision/training/train_student.py:ClipDataset.__getitem__`, before
  `cv2.warpAffine`, with probability 0.5 per referee box and the same draw on all 3 frames: remap brightness
  `V' = 255 (V/255)^γ` with γ log-uniform in [0.5, 2] (*estimate*). Apply it to low-saturation pixels (S < 60) in the torso band
  (rows 15-55 % of the box). This spans dark and light grey without touching hue. It needs referee flags in the label rows:
  * `prep_coco --referees` must keep which boxes are referees (today it merges them into `persons`);
  * `court_people` must flag the boxes the shirt rule calls referees;
  * promoted boxes carry their flag.

  It is a targeted form of the colour randomisation the recipe already uses.

**Data.** All in the container: the SportsMOT folders, v1, and `coco3r` (re-run `prep-coco` from `data/hfcoco/train3_ann.json` with
flags). The Lakers @ Warriors folder is on the Mac. Relabelling cost: one v1 pass over 7,669 samples, about 106 ms each fp32
(§3.3), so about 15-20 min CPU. Training is 1-2 runs (E1a promotion only, E1b with the augmentation).

**Expected effect.**
* 2017 referees on the court should rise from 23.3 % toward v1's 83.6 %. *Estimate*: half or more of the gap, if the §2 mechanism holds.
* Roboflow referees (2024-25) unchanged at ~84 %; players unchanged.
* The basis is the lineage and data in §2, plus CityPersons' evidence that ignore regions only avoid harm. Nothing published measures
  this exact fix.

**Risk.**
* Promoted fans or staff. §2 found ~20-40 % fans among the ignore boxes of the two NBA broadcasts. They would push boxes on nobody up,
  against aim (a). The gate and the standing test control this.
* Light shirts with dark trousers are also common among courtside staff.
* More referees boxed means more work for the shirt rule downstream. That is v1's situation, under which the 2017 pipeline was 75.8 %.

**How to measure.**
* Container, no-regression checks: `--coco`. Referees found must not fall below 426, boxes on nobody must not rise beyond chance
  (sign test over the 211 frames), and players-and-referees AP50 is reported. `--ball-folder`: McNemar against v2, must not be worse.
* Mac, the target: 2017 referees on the court (116), paired per clip with a 95 % interval as in §10.13; 2017 off-court share (must not
  rise); Lakers @ Warriors players and referees.
* Optional container-side check: hold out the Lakers at New Orleans broadcast from training and hand-label its officials on ~100
  frames. That gives a light-grey referee test without the Mac, at the cost of one broadcast of training data.

### E2. Referees told apart within each game, per track (aim b, told apart)

**What to change.**

* `courtvision/nba/analyze.py`: `RefereeCheck(method="game")`, one per game.
  1. Buffer the on-court people: with a camera fit, foot within the court plus 3 ft; without one, boxes taller than 4.5 % of the
     frame height (the browser's referee size gate).
  2. Features: `nba.referee.referee_features_batch` (24 colour and stripe statistics) plus `perception.appearance.torso_feature`,
     standardised per game and reduced by PCA.
  3. **Over-cluster**, after Liu et al. 2007: `perception.teams.kmeans` with k = 6, merged to 4 groups. The two largest groups are the
     teams. Of the other two, the one farther from the team clusters is the referee candidate; the other is "outlier" (coaches, fans).
  4. Accept the candidate only if it averages 1 to 3.5 members a frame on frames with 8 or more people (the existing guard's numbers),
     with the shirt rule as a tie-breaker; otherwise fall back to the shirt rule.
  5. Refit as crops accumulate (`TeamConfig.refit_at`'s schedule, e.g. 150, 600, 1,500).
* **Per-track votes.** In `courtvision/perception/tracker.py`, drop or soften the hard class gate (`cost[cls_t != cls_d] = _INF`) so
  that one call flipping does not split a track. In `courtvision/perception/pipeline.py:process`, decide the referee class per track
  with decayed votes, as `perception.teams.TeamVoter` does for teams (SoccerNet-GSR uses a tracklet majority vote).
* **Step 1 alone (E2a):** per-track votes over the current shirt-rule calls. This is a smaller change and worth measuring on its own.

**Data.** None new. The tuning set is the Roboflow training games (`data/hfcoco/train3_ann.json`); the tests are the 5 held-out games,
the 160 hand-labelled boxes of the two README games (`docs/research/referee_check_two_games.json`, frames on the Mac), and the pipeline
on both games.

**Expected effect.**
* Era-agnostic referee calls: the shirt rule's behaviour on 2017 (66 of 80 hand-labelled boxes right, §10.12) and the learned
  check's on 2024-25 (F1 0.847 against 0.690 left out per game, §11.1). *Estimate*: per-game F1 ≥ 0.85 on Roboflow, and no
  possession or made-or-missed loss on 2017.
* Fewer split referee tracks.
* Literature: Liu et al. 2007 report 82.9 % and 97.1 % referee labelling with this scheme in soccer, where referees are fewer per
  frame than basketball's 3 of 13 people.

**Risk.**
* A team in grey or black near the referees' shade: the count prior and the fallback handle a whole team in the "referee" cluster.
* Coaches at the sideline absorbed into the referee group: the on-court restriction and the outlier group help.
* Early-game frames before the first fit.
* Roboflow's frames are sparse, about 40 per game, unlike video.

**How to measure.**
* Container: a `--referee-method game` option in `tools/detector_bench.py:coco_bench`, fitted per game on the matched detections and on
  the truth boxes. Report referee precision, recall and F1 per game, against the shirt rule; McNemar on boxes either gets right.
  The learned check is in-sample on these games (§11.1), so compare with its left-out-per-game figures.
* Mac: the 160 hand-labelled boxes; `nba_analyze.py events --referees game` on both games. Gate: possession timeline and made-or-missed
  not worse than the shirt rule beyond chance, as §10.12 required.

### E3. Court-geometry person labels at scale: round 8's K2, with fixes (aim a)

**What to change.** In round 8's `court-persons` (being written beside `student_finetune.court_people`):

1. **Seated or standing.** For people 0-8 ft outside the lines, compare the box height with `analyze.rectify_box`'s standing box at
   that foot point. Ratio below ~0.75 (*estimate*): background (seated fans, the bench). Otherwise: ignore (officials, inbounders,
   coaches standing).
2. Boxes touching the image border: ignore.
3. Decide per track: ByteTrack (`perception/tracker.py`) over the clip, median court position per track.
4. Use only frames with status `tracking` and coverage ≥ 0.5. Frames without a fit train no person loss, as planned.
5. **Doubt band on v1's scores**, after Vandeghen et al.: positives from v1 ≥ 0.5; 0.3-0.5 ignored, even on the court.
   These thresholds are *estimates*: their best band was much higher on Faster R-CNN scores. Set ours from a crop check of v1's
   on-court boxes by score.
6. **Variant K2b**, Q5's focal weighting: `ClipDataset` gets per-box negative weights (`neg` boxes, `pixel_weight` 3 in those cells;
   `focal_loss` already multiplies cells by `pixel_weight`).

**Data.** Round 8's 600 BARD clips with court tracks (per game: `perception.broadcast_court.estimate_center`, then
`tools/nba_track_court.py`) and v1. Being prepared.

**Expected effect.**
* Fewer crowd and bench boxes, especially on 2024-25 arenas.
* Precedent: with the Lakers @ Warriors teacher folder passed through `court-people`, A and C box 28.8 % and 29.6 % of the 2017
  game's off-court people, where M1, trained on the raw teacher folder, boxes 43 % (§10.9, §10.13). Other differences confound this.
* Fixes 1-5 reduce label noise; their size is unknown.

**Risk.** Positives on courtside fans within 2 ft; negatives on players out of bounds; wrong fits at difficult arenas (Q2); K2b
suppressing players near the bench.

**How to measure.** Container: `--coco` boxes on nobody (0.19 a frame), referees found, players-and-referees AP50; `--ball-folder`
(no change). Mac: 2017 off-court share (lower is better), players and referees on the court.

### E4. Ball pseudo-labels on SportsMOT (aim c, especially the 2017 game)

**What to change.**

* `tools/student_finetune.py`: a `prep-mot-ball` command, or `prep_video` over image sequences. Run v2 on each sequence's stored frames
  with the three-frame history: consecutive frames, since `history_step(25)` is 1. Keep candidates from 0.05 and link one path per
  sequence (`training.ball_labels.link_ball_path`, which rescales to 25 fps and 720 lines through `LinkConfig.for_clip`).
* Write the result as a **separate pseudo folder** whose frames are linked to SportsMOT's: `kind: pseudo`, `person_weight` 0. This
  gets the 0.5 cap from `folder_weights` for free.
  * Give it a `video` identity of its own, as `prep_mot` does. `plan_data`'s guard compares frame ranges, and every SportsMOT clip
    starts at frame 1, so a shared identity would guard out every clip.
  * Leave out the human folder's held-out clips (`clip % 10 == 5`) when writing it.
* The human SportsMOT folders keep no ball labels, so `FineTuneDataset` keeps their hue rotation for the person head. Writing the
  balls into the human folders instead would switch hue off on those samples and leave the weights uncapped, because `ClipDataset`
  treats human folders' ball weights as exact.

**Data.** SportsMOT frames (24,941, in the container) and v2. Labelling takes about 20 min with int8 (49.6 ms a frame, §3.3) or about
45 min fp32. Then 1 run.

**Expected effect.**
* A better 2017 ball: its broadcast shares the arena and probably the era with SportsMOT's Oracle footage, and both are 720p.
  *Estimate*: +1-3 points top candidate.
* BARD within chance.
* Evidence: M1 against M1-ctrl (§10.7.1). Footage outside NBA.com clips helped the 2017 ball; more NBA.com self-training did not
  (M1-ctrl against round 2: 5 frames to 16, p 0.027).

**Risk.**
* FIBA and NCAA balls differ in colour from the NBA ball `ball_color_likelihood` was tuned on (`BALL_LAB`). That should mean fewer
  labels rather than wrong ones (reasoning).
* SportsMOT's 25 fps spacing: inputs 40 ms apart instead of 33.
* Confirmation bias, as in all self-training.

**How to measure.** Container: `--ball-folder` per frame and `--link`, McNemar against v2; the Roboflow ball (194 frames). Mac: 2017
ball (606 frames, top and linked, McNemar) and Lakers @ Warriors (243).

### E5. Better ball pseudo-labels: flip ensembles, graded weights, hard negatives (aim c)

**What to change.**

* **Flip ensembles** (Data Distillation). In `student_finetune.prep_video`, run the labeller on the 3-frame input and on its
  horizontal flip, un-flip, and average the ball heatmaps before taking peaks. This needs dense outputs: `detector_bench.OnnxModel`
  returns them, while `OnnxStudent.detect` returns peaks. Then link as now.
* **Graded weights with a doubt band.** In `training/ball_labels.link_ball_path` and `label_folder`, map each label's evidence to a
  weight. Evidence: the raw candidate score, agreement between the flipped and plain passes, and on the Mac optionally agreement with
  the RF-DETR teacher path from `prep-teacher`. Scale: 0 at τl, 0.5 at τh, ignored in between (Vandeghen's progressive doubt).
  Today every pseudo label weighs exactly 0.5.
* **Hard negatives.** Candidates scoring ≥ 0.3 and more than ~45 px of 1080p (*estimate*) from the linked label are written as
  `neg` points. `ClipDataset` builds `ball_pw`: 1 everywhere, 3 within 2 cells of each `neg` point. `train` passes `tg["ball_pw"]`
  as `pixel_weight` to the ball `focal_loss`. These are Jin et al.'s temporally unsupported detections, applied to the ball.
* **Step 0, an error audit** (cheap). On the BARD check frames v2 gets wrong (~95 of 1,153), draw the label and the top candidate and
  sort them by eye: a second ball, a head, a hand, the rim, the crowd, a graphic, or a wrong label. This decides how much the metric can
  move at all.

**Data.** `data/ft/bard_calib_r2` (390 clips, in the container) and round 8's 600 clips. Labelling twice costs about 1-2 h CPU. The
RF-DETR agreement runs on the Mac (`prep-teacher --device mps`).

**Expected effect.** *Estimate*: +0.5-1.5 points per frame on BARD, mostly fewer wrong frames. Repo precedent: a better labeller gave
+2.2 points (§8.2). WASB's hard-sample mining gave +0.3-0.6 F1.

**Risk.**
* Confirmation bias.
* Agreement with RF-DETR biases labels toward RF-DETR's easy (held) balls. The BARD check labels are RF-DETR paths too, so the
  metric could rise for the wrong reason. Keep RF-DETR agreement as its own ablation, and look at the human-labelled Roboflow ball.

**How to measure.** `--ball-folder` per frame and `--link`, McNemar against v2 and against K1 (round 8's new ball labels); the
Roboflow ball; Mac: both games.

### E6. Weight EMA in every run; online Mean Teacher after K2 (all aims)

**What to change.**

* **EMA.** In `student_finetune.train`, add `--ema 0.999`: a float copy of the state dict updated after each `opt.step()`, saved as
  `ema.pt` beside `last.pt`; `_quick_mot_eval` both. Export and score both files.
* **Mean Teacher, later** (`--online-pseudo` for pseudo folders).
  * `ClipDataset` returns a weak view (no hue or colour jitter) and a strong view (hue ±30°, colour jitter) with identical geometry,
    so the teacher's heatmap aligns cell for cell.
  * The teacher is the EMA network, run without gradients on the weak view of pseudo-folder samples only.
  * Person peaks become targets with a doubt band: ≥ 0.5 positive, 0.25-0.5 ignored, below 0.25 background. They pass K2's geometry
    gate. `person_wh` and offsets are trained only from peaks ≥ 0.7, after Unbiased Teacher v2.
  * Ball targets come from the offline linked path, so temporal consistency is kept.

**Data.** As in K2. Nothing new.

**Expected effect.**
* EMA: small, possibly within chance (*estimate*). It is nearly free and has a mechanism (Mean Teacher; SWA).
* Mean Teacher: it updates pseudo-labels as the model improves (Unbiased Teacher +6.8 mAP over offline STAC at 1 % COCO; Soft Teacher
  +3.6 mAP with full labels). Here labels are plentiful, so expect less (*estimate*).

**Risk.** EMA lag over only 4,000 steps; the cosine schedule makes the end weights close anyway. Mean Teacher amplifies v2's own
errors (crowd boxes, missed referees) unless the geometry gate and doubt band are strict.

**Budget.** EMA: none. Mean Teacher: about +10 % per step (*estimate*), about 25 min on MPS; borderline on the container CPU.

**How to measure.** As K2, with `last.pt` against `ema.pt` paired on every set.

### E7. Flicker mining for frames without a court fit (aim a)

**What to change.**

* `tools/student_finetune.py mine-flickers`: run v2 over clips (BARD calibration clips, with stored frames) and track person detections
  ≥ 0.3 with `perception.tracker.ByteTracker`.
* On frames **without** a court fit, where K2 trains no person loss: a detection in a track of 2 frames or fewer, with no detection at
  IoU 0.5 in t±1 and t±2, becomes a weighted hard negative. Gaps of 1-2 frames inside long tracks are filled as person pseudo-labels at
  weight 0.5 (hard positives, after RoyChowdhury et al.).
* `ClipDataset`: per-box negative weights, the same mechanism as K2b.
* **Image level:** in `train`, multiply the balanced sampler's weights by (1 + k × v2's unmatched detections in the sample). This is
  OHEM at the image level.

**Data.** BARD clips (in the container and round 8) and v2. About 30-45 min CPU of inference. Then 1 run.

**Expected effect.** Unknown in size, modest by *estimate*. It targets crowd boxes on replays and close-ups, which neither geometry nor
the human labels cover. Jin et al. report gains for face and pedestrian detectors from such mined examples.

**Risk.** Stable crowd false positives (a fan detected on every frame) are not flickers and stay. Briefly occluded real players also
look like flickers; mitigate by applying it only to boxes not overlapping the players' foot-hull band.

**How to measure.** `--coco` boxes on nobody and referees found; Mac: 2017 off-court share. Also the frames without a fit: compare
the two models' person boxes on them against the teacher's people, as in §10.2's "no court fit" row.

### E8. A court auxiliary head, training only (aim a)

**What to change.**

* `courtvision/perception/student.py:build_net(..., aux_court=False)`: an optional head (3x3 conv 128→32, ReLU, 1x1 to 1 channel) on
  the FPN output at stride 4.
* `student_finetune.train`: a binary cross-entropy court-floor loss, weight ~0.2 (*estimate*), masked to frames with a fit.
* `export` and every `load_state_dict` caller: strip the `aux_*` keys. The `.pt` loaders (`detector_bench.TorchModel`,
  `StudentDetector`) load strictly, so this is needed.
* `ClipDataset`: rows carry the frame's image-to-court homography (round 8's `court-persons` computes it). Draw the court polygon,
  plus a 2-ft band, into a stride-4 mask under the same affine warp `M`.

**Data.** Lakers @ Warriors (Mac) and the BARD clips with court fits (round 8).

**Expected effect.** HyperLearner (CVPR 2017) and SDS-RCNN (ICCV 2017) improved pedestrian detection with auxiliary supervision at no
inference cost. Neither is in sports, so any gain on crowd boxes here is an *estimate* (unknown sign and size).

**Risk.** The person head may not use the cue. Jumping or out-of-bounds players. Fit quality.

**How to measure.** As E3.

---

## 5. Order of work, and what to run first

1. **Cheap audits (container, hours of light CPU).**
   * E1's relabel with the precision gate.
   * E5's ball error audit.
   * A crop check of K2's labels in the 0-3 ft band. In `court_people` on Lakers @ Warriors, 37,107 boxes were kept within 3 ft
     against 34 dropped beyond 10 ft (§10.9), so verify the "kept" side holds no seated fans.
2. **E1a, then E1b.** One run each, on top of whatever round 8 keeps.
3. **E2a, then E2.** No training; container scoring, then Mac pipeline runs.
4. **K2 results**, then E3's fixes as K2+. Switch on E6's EMA in the same runs.
5. **E4 and E5**, each one run against K1.
6. **E7 and E8** if aim (a) still needs work after K2+.

---

## 6. Not recommended now

* **A court-mask input channel.** It changes the 9-channel input, which needs the frozen stem retrained and touches `preprocess_frames`,
  ONNX, Core ML and the browser. It needs a court fit at inference (lost on about 20 % of Chase Center frames and about 50 % of the
  2017 game's), so it would have to be trained with the channel dropped out. The pipeline already drops off-court players when a fit
  exists. E8 is the training-only form.
* **A stride-2 or HRNet-style ball head** (WASB). It changes `ball_hm`'s shape (decoder, sparse heads, browser) and adds compute. Our
  ball errors are ranking errors, not misses (Q4). Revisit if E5's audit finds small or blurred balls lost. An interface-preserving
  middle way, untested: a space-to-depth of the stride-2 stem features added into the ball head's input at stride 4.
* **Motion-compensated history**: warp t−2 and t−1 into t with `CameraTrack.frame_to_frame`, or give a background frame as TrackNetV3
  does. It needs the camera at inference, and it changes what the frozen stem sees. Lower value than the label fixes.
* **A referee class in the detector** (a second person heatmap channel). It adds an output. The learned colour check's failure on dark
  jerseys (§10.10-10.12) shows that an appearance classifier trained on one era generalises badly; try E2 first.
* **Adapting on the test game** (ARTHuS-style online distillation on the game being measured). It conflicts with §10.1's rule that the
  2017 folder is never trained or tuned on, unless the owner changes the protocol to allow adaptation on unlabelled frames with
  held-out clips.
* **CSD flip consistency** as a separate experiment. It is cheap on MPS but likely small with plentiful labels; fold it into E6 if E6
  shows online consistency helps.

---

## 7. Measurement notes

* `coco_bench` keeps per-frame hit lists only for the ball. Two small additions would allow paired tests:
  * per-referee-box found flags, for McNemar on the 506 referees;
  * per-frame counts of boxes on nobody, for a sign or Wilcoxon test over the 211 frames.

  v2 puts about 40 boxes on nobody across the whole set (0.19 a frame), so only large changes in that count will be significant.
* Players-alone AP50 falls whenever referees are boxed (§11.2). For E1 and E2, use players-and-referees AP50 plus referees found.
* For people on the 2017 game, report per-clip paired intervals as §10.13 does; boxes of one clip are not independent.
* Every candidate goes through §10.8's replacement rule before it reaches the app.

---

## 8. Unverified or uncertain

* The §2 mechanism for the 2017 referees is a hypothesis built on the documented lineage plus small by-eye samples (36 + 24 + 24 + 24
  crops, and 5 whole frames). E1's ablation is the test.
* That SportsMOT's Oracle broadcast and the 2017 Finals share a referee uniform: inferred from the CSN branding (before April 2017)
  and the crops. I did not see the 2017 footage, which is on the Mac.
* Every number marked *estimate*: the thresholds (height ratio 0.8-1.25 and 0.75, γ range, 45 px, focal weight 3, EMA 0.999),
  the costs per step, and the expected gains.
* NBA courtside seat distances and an NBA-specific officiating manual: not found; I used Wikipedia's description of the lead official.
* TrackNetV4's venue (ICASSP 2025) comes from search results; TrackNetV3's venue is given only by its DOI.
* The NUST-NBA181 videos behind WASB's basketball ball labels are available on request for academic research only, per search
  results; the project page now returns 404. They would be a human-labelled NBA ball benchmark, but at 12 fps, which does not match
  the 3-frame, 33 ms input.

---

## 9. Sources

Semi-supervised and self-training detection:
* Tarvainen & Valpola 2017, Mean Teacher, NeurIPS: https://arxiv.org/abs/1703.01780
* Liu et al. 2021, Unbiased Teacher, ICLR: https://arxiv.org/abs/2102.09480 ; config: https://github.com/facebookresearch/unbiased-teacher/blob/main/configs/Base-RCNN-FPN.yaml
* Liu, Ma & Kira 2022, Unbiased Teacher v2, CVPR: https://arxiv.org/abs/2206.09500
* Xu et al. 2021, Soft Teacher, ICCV: https://arxiv.org/abs/2106.09018
* Sohn et al. 2020, STAC: https://arxiv.org/abs/2005.04757
* Tang et al. 2021, Humble Teacher, CVPR: https://arxiv.org/abs/2106.10456
* Zhou et al. 2022, Dense Teacher, ECCV: https://arxiv.org/abs/2207.02541
* Wang et al. 2023, Consistent-Teacher, CVPR: https://arxiv.org/abs/2209.01589
* Xu et al. 2023, Efficient Teacher: https://arxiv.org/abs/2302.07577
* Liu et al. 2023, ARSL, CVPR: https://arxiv.org/abs/2303.14960
* Li et al. 2022, PseCo, ECCV: https://arxiv.org/abs/2203.16317
* Jeong et al. 2019, CSD, NeurIPS: https://proceedings.neurips.cc/paper_files/paper/2019/file/d0f4dae80c3d0277922f8371d5827292-Paper.pdf
* Radosavovic et al. 2018, Data Distillation, CVPR: https://arxiv.org/abs/1712.04440
* Xie et al. 2020, Noisy Student, CVPR: https://arxiv.org/abs/1911.04252
* Deng et al. 2021, Unbiased Mean Teacher, CVPR: https://arxiv.org/abs/2003.00707
* Li et al. 2022, Adaptive Teacher, CVPR: https://arxiv.org/abs/2111.13216
* Chen et al. 2022, Probabilistic Teacher, ICML: https://arxiv.org/abs/2206.06293
* Izmailov et al. 2018, SWA, UAI: https://arxiv.org/abs/1803.05407
* detectron2 defaults (FrozenBN): https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py

Video, hard examples, and sports self-training:
* Jin et al. 2018, Unsupervised hard example mining from videos, ECCV: https://arxiv.org/abs/1808.04285
* RoyChowdhury et al. 2019, Automatic adaptation of object detectors to new domains using self-training, CVPR: https://arxiv.org/abs/1904.07305
* Vandeghen, Cioppa & Van Droogenbroeck 2022, Semi-supervised training to improve player and ball detection in soccer, CVPRW: https://arxiv.org/abs/2204.06859
* Cioppa et al. 2019, ARTHuS, CVPRW: https://openaccess.thecvf.com/content_CVPRW_2019/papers/CVSports/Cioppa_ARTHuS_Adaptive_Real-Time_Human_Segmentation_in_Sports_Through_Online_Distillation_CVPRW_2019_paper.pdf
* Shrivastava et al. 2016, OHEM, CVPR: https://arxiv.org/abs/1604.03540
* Lin et al. 2017, Focal loss, ICCV: https://arxiv.org/abs/1708.02002
* Zhou et al. 2019, CenterNet (Objects as Points): https://arxiv.org/abs/1904.07850
* Zhang et al. 2017, CityPersons (ignore regions), CVPR: https://arxiv.org/abs/1702.05693
* Ghiasi et al. 2021, Simple Copy-Paste, CVPR: https://arxiv.org/abs/2012.07177

Geometry and context:
* Hoiem, Efros & Hebert 2006, Putting Objects in Perspective, CVPR (IJCV 2008): https://ri.cmu.edu/publications/putting-objects-in-perspective
* Mao et al. 2017, What Can Help Pedestrian Detection? (HyperLearner), CVPR: https://arxiv.org/abs/1705.02757
* Brazil, Yin & Liu 2017, SDS-RCNN, ICCV: https://arxiv.org/abs/1706.08564
* Arbués-Sangüesa, Haro & Ballester 2019, multi-person tracking in basketball: https://arxiv.org/abs/1907.04637
* Theiner et al. 2022, Extraction of positional player data from broadcast soccer videos, WACV: https://arxiv.org/abs/2110.11107
* Official (basketball), Wikipedia: https://en.wikipedia.org/wiki/Official_(basketball)

Referees and teams:
* Lu, Ting, Little & Murphy 2013, TPAMI: https://www.cs.ubc.ca/~murphyk/Papers/weilwun-pami12.pdf
* Liu et al. 2007, Automatic player detection, labeling and tracking in broadcast soccer video, BMVC: https://bmva-archive.org.uk/bmvc/2007/papers/paper-70.pdf
* Koshkina, Pidaparthy & Elder 2021, Contrastive learning for sports video: https://arxiv.org/abs/2104.10068
* Istasse, Moreau & De Vleeschouwer 2019, Associative embedding for team discrimination, CVPRW: https://arxiv.org/abs/1907.01058
* Somers et al. 2024, SoccerNet Game State Reconstruction, CVPRW: https://arxiv.org/abs/2404.11335
* roboflow/sports team classifier: https://github.com/roboflow/sports (sports/common/team.py)
* CSN Bay Area renamed NBC Sports Bay Area, effective 2 April 2017: https://www.nbcsports.com/pressbox/press-releases/nbc-sports-regional-networks-to-rename-california-based-properties-to-feature-nbc-sports-brand

Ball:
* Tarashima et al. 2023, WASB, BMVC: https://arxiv.org/abs/2311.05237 ; code: https://github.com/nttcom/WASB-SBDT
* Chen & Wang 2023, TrackNetV3 (doi:10.1145/3595916.3626370): https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf
* Raj, Wang & Gedeon 2024, TrackNetV4: https://arxiv.org/abs/2409.14543
* Gossard et al. 2025, BlurBall (CVPRW 2026 per arXiv): https://arxiv.org/abs/2509.18387
* Xu et al. 2025, TOTNet: https://arxiv.org/abs/2508.09650
* Komorowski et al. 2019, DeepBall: https://arxiv.org/abs/1902.07304 ; FootAndBall: https://arxiv.org/abs/1912.05445
* Yan et al. 2020, NUST-NBA181 dataset (Social Adaptive Module), ECCV: https://arxiv.org/abs/2007.09470

Datasets used by the repo: SportsMOT https://arxiv.org/abs/2304.05170 ; DeepSportradar-v1 https://arxiv.org/abs/2208.08190
