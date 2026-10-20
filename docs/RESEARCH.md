# CourtVision research report

*2026-10-20. This report synthesises the seven verified notes in `docs/research/` and `sdk_ground_truth.md`. It is written for a technical founder building CourtVision in Python 3.12 on an Apple M3 Pro (18 GB, MPS, no CUDA).*

**Conventions.** **Jev** always means **TypeSafe AI's Jev**, the "System One" model. It answers typed `Choice`, `Score` and `Noul` questions (plus `Box` for images) with calibrated probabilities, and it writes no text [10][11]. **Claude** means `claude-opus-5-5`. Citations `[n]` point to §8. Sources [1]-[9] are the local notes, so a citation to a note marks that note's own finding or design proposal. "(unverified)" comes from the notes. Every threshold is a starting point to tune on labelled games, not a finding. Outside §1A and §7A the report adds no new facts. §1A and §7A were added in the 2026-10-20 gap pass; they add new facts with sources [449]-[482] (§8.4) and record each decision that resolves a conflict between documents.

**Precedence.** (1) §1A of this report decides configuration and design choices, and `docs/ARCHITECTURE.md` and `pyproject.toml` have been edited to match it. (2) `sdk_ground_truth.md` [8] still wins on SDK *API surface* facts (names, signatures, error types), but not on chosen values: its `max_tokens=400` example is superseded by D13. (3) The notes in `docs/research/` come next. (4) Where ARCHITECTURE.md still disagrees with (1)-(3), it is a bug in ARCHITECTURE.md.

---

## 1. Executive summary

**Bottom line.** A live system for **amateur basketball** (high school, small college, AAU, adult rec) can be built today on one M3 Pro, under four conditions:
1. Live calls are deterministic templates over confirmed events.
2. Claude writes only dead-ball colour, summaries and coach digests, behind a validator.
3. Jev makes narrow judgments, never on the sub-second path.
4. The box score is reconciled against the scoreboard, an official feed or an operator.

Pro and major-college rights are locked up. NBA tracking is Hawk-Eye plus Sportradar, with exclusive data rights through 2031 [12][13]. NCAA championship footage is NCAA copyright [14]. NBA.com terms bar commercial and play-by-play uses of NBA statistics [15].

**Feasible now.**
- **Team-level live box score from a fixed camera.** The market note ranks it #1 (value 5, feasibility 4) [5]. In Roboflow's tests, RF-DETR-small found the ball in 98.1% of in-flight frames: 759 of 774 frames of Roboflow's own test footage. The detector was trained on a fork of a shooting-robot dataset, and its validation recall was 84.9%. This is not a game-footage figure [16].
- **Template live calls**, about 0.3-0.7 s after an event resolves [4]. Warm AVSpeech produces first audio in 12-26 ms [9].
- **Post-game coach reports.** The alternatives are Hudl Assist human tagging at $900-$2,100 per team per season [17] and SportsVisio AI at $149-$189 a month for 5 games [18].
- **Jev at dead balls**, for about $0.006-0.019 per game [7]. Input costs $0.042 per million tokens, and output is free [19].

**Hard.**
- **Identity.** The best online system on a 2025-26 NBA benchmark names the right player 28.5% of the time [20].
- **Ball and make/miss.** The ball shrinks to about 4 px at 512 px input [2]. In 2D, a front-rim bounce can look like a make, and Roboflow publishes no make/miss accuracy [16][21].
- **Compute.** `RFDETRNano` alone takes about 21 ms per image on an M3 Pro [22].
- **LLM latency.** Opus 5.5 always thinks, and its time to first answer token is 6.52 s at `low` effort [23][24].
- **Jev latency.** TypeSafe claims "about 100 ms"; its cookbook measured a median of about 0.4 s; there is no SLA [25][26].
- **Licensing.** Almost every basketball dataset is non-commercial [27][28][29][30], and Ultralytics is AGPL [31][32].

**The gap.** The notes found no shipping product that produces live, grounded, spoken play-by-play plus a live box score from amateur video [5]. Pixellot has announced AI commentary without details [33][34]. WSC Sports voices finished NBA highlight clips, not live games [35].

**Recommended architecture** (merged from [2][4][7]):

```
 Camera / stream (RTSP·RTMP·HLS·file)               Scoreboard OCR · operator · official feed
        │ 1080p30                                    (authoritative for score and clock)
        ▼                                                          │
┌─ CV process (CoreML / MPS) ─────────────┐        ┌─ Events + stats process (asyncio) ───▼─────┐
│ RF-DETR-S/M @15 Hz + ball ROI @30 Hz    │ tracks │ Event engine: possession, shot, make/miss, │
│ trackers: ByteTrack | BoT-SORT + CMC    ├───────►│ rebound, TO, foul (holds + confidences)    │
│ team SigLIP→UMAP→KMeans · jersey CNN    │        │  ProvisionalEvent → Confirmed / Revision   │
│ court H (RANSAC) → court x,y in feet    │        │ Versioned fact store · stats (official vs  │
│ ball Kalman · pose on demand            │        │ provisional) · reconciliation              │
└─────────────────────────────────────────┘        └────────┬──────────────────────────┬────────┘
                                                   FactPacket│                 triggers │
                                                            ▼                          ▼
              ┌─ Narration director (deterministic, 50 Hz) ──────┐   ┌─ JevAdvisor (≤ 8 in flight) ─┐
              │ P0-P2: templates only (no LLM, no Jev)           │◄──┤ B1 possession_end    1.5 s   │
              │ P3: Claude opus-5-5, effort low, 2 speculative   │   │ B2 dead_ball_fan     1.0 s   │
              │     candidates → deterministic validator ────────┼──►│ B3 dead_ball_coach   1.0 s   │
              │     → Jev line guard (B4)                        │   │ B4 line_guard        0.8 s   │
              └───────┬──────────────────────────────┬───────────┘   │ B5 event_ambiguity   0.7 s   │
                      ▼                              ▼               └──────────────────────────────┘
        TTS: AVSpeech (Kokoro / ElevenLabs   Coach cards (≤ 12 words, Jev-ranked), dashboard,
        optional) → mixer, word-boundary     WebVTT captions, audio-description track,
        cut                                  JSON event stream (authenticated, optional delay)
```

---

## 1A. Decisions that supersede ARCHITECTURE.md and pyproject.toml

The module agents code against `docs/ARCHITECTURE.md`, `courtvision/types.py` and `pyproject.toml`. Those files disagreed with this report in the places below. Each row is now decided, and ARCHITECTURE.md and pyproject.toml were edited on 2026-10-20 to match. `types.py` already agrees with D1 and was not changed.

| # | Topic | Was (ARCHITECTURE / pyproject / a note) | Decision (wins) | Basis |
|---|---|---|---|---|
| D1 | Court units | ARCHITECTURE and `types.py`: feet. Rules note §6 and this report: a metre `CourtSpec` | **Feet are canonical everywhere in code** (`types.py` already says FEET). Source tables may stay in metres. FIBA metres are converted once, at the geometry boundary (`ft = m / 0.3048`). Metres appear only at the display edge (FIBA markets). The metre homography gate "0.3 m" becomes **1.0 ft**, which is ARCHITECTURE's `ransac_thresh_ft` default | `types.py` and the simulator are already in feet. NBA, NCAA and NFHS define court geometry in feet and inches, so their values stay exact [53][59][240]. pbpstats zone cut-offs are in feet [250]. Changing the contract would churn every module for no accuracy gain |
| D2 | Jev model | `JevConfig.model="jev-latest"` | `model="jev-1.13.0"`. Log `resp.model`. Treat any other version as untuned | The alias moves [19][7] |
| D3 | Jev on the live narration path | `EventAssessment.narrate`, `.interrupt`; `NarrationConfig.use_jev_gate=True` | **Removed.** The deterministic director owns speak-now, which-event and preemption (site 1a-c rejected). `EventAssessment` keeps only `stakes` (4-level `Score moment_stakes`) and `highlight` (`Noul highlight_play`), both from B1, for flavour | [7] §3 sites 1-2 |
| D4 | Jev line guard | `use_jev_validator=False` | `use_jev_validator=True`, matching `LLM_LINES_REQUIRE_JEV_GUARD=True`. Without a Jev verdict, LLM lines are not spoken; templates are | [7] §3 site 7 |
| D5 | TTS | `MacSayTTS` (`say` per line) | **`AVSpeechTTS`** (`AVSpeechSynthesizer` via `pyobjc-framework-AVFoundation` 12.2.2, warmed at boot). `say` is a dev-only fallback | `say` measured 0.78-2.81 s per line; warm AVSpeech gives first audio in 12-26 ms [9] |
| D6 | Claude deadline | `first_token_deadline_s=1.5` | No first-token abort. Claude runs only as speculative candidates, with `candidate_deadline_s` set from the measured p95 (§7A.3). Until measured: 12.0 s, and Claude only for timeouts, period breaks and state-independent topics | 6.52 s TTFT at `low` [24]; §7A.3 |
| D7 | Jev deadlines | `timeout_s=0.6, max_retries=0` for every call | Per-bundle deadlines (B1 1.5 s, B2/B3 1.0 s, B4 0.8 s, B5 0.7 s, B6 5 s) and per-bundle `RetryPolicy` as initial values. Reset each from the measured p95 (§7A.3). One circuit breaker per bundle (§7A.9) | [7] §4 |
| D8 | 2 vs 3 | Value from the "release location" | `is_three` is evaluated at the shooter's **last grounded foot position before the gather**. The line itself counts as 2 | [3][54][55] |
| D9 | YOLO adapter | `UltralyticsDetector` and `yolo=["ultralytics>=8.3"]` | Moved to an R&D-only extra, **`rnd-agpl`**. Shipped code must not import it. The production adapter is `RFDETRDetector` | AGPL §13 covers network use [31][32] |
| D10 | CV pins | `rfdetr>=1.2`; no `supervision`, no `trackers` | `rfdetr[coreml]==1.11.2`, `supervision==0.31.0`, `trackers==2.6.1` in extra `cv`. Core pins: `typesafe-sdk==0.7.5`, `anthropic==1.15.0`. Generate a lock file from these pins | `sv.ByteTrack` was removed [40]. rfdetr 1.11.2 needs `supervision>=0.29.0` and `transformers>=5.1,<6` [460] |
| D11 | Team-embedding deps | "No scikit-learn, no torch in core" | Still true **for core**. SigLIP→UMAP→KMeans lives in extra **`team`** (`transformers>=5.1,<6`, `torch`, `umap-learn`, `scikit-learn`), imported lazily. The numpy k-means `TeamClassifier` remains the core fallback | [44][460] |
| D12 | OpenCV wheel | `opencv-python-headless` (new finding) | One wheel only: **`opencv-python`**. `trackers` 2.6.1 requires `opencv-python>=4.8.0`, and both wheels install the same `cv2` package. The repo `.venv` currently has both (5.0.0.93). `supervision` 0.31.0 (MIT) declares no OpenCV dependency of its own | [461][462] |
| D13 | Claude `max_tokens` | 400 in `sdk_ground_truth.md`; 1024-2048 in the narration note | **1536.** Thinking always runs on `claude-opus-5-5` and counts toward `max_tokens`. Treat `stop_reason="max_tokens"` as a validator failure | [23] |
| D14 | Scoreboard OCR rate | 1-2 Hz (CV note) vs 5 Hz (training note); "2 identical reads, ~200 ms" (narration) | **5 Hz.** A score change is confirmed by 2 identical consecutive reads (≈ 400 ms after the first change) plus a clock-consistency check | §7A.5 |
| D15 | Clock and period rules | Inside the geometry `CourtSpec` | A separate **`RuleSpec`** loaded from per-league YAML with state and tournament overrides (§7A.6). `CourtSpec` keeps geometry only | §7A.6 |
| D16 | Names in cloud payloads | Real names in Jev states and Claude FactPackets | **Pseudonymised by default** (opaque per-game tokens). Names are re-inserted locally only after the deterministic validator passes (§7A.7) | [449]-[456] |

---

## 2. How it works end to end

The decision layer (Jev) is covered in §3.

### 2.1 Perception
- **Approach.**
  - Detector: RF-DETR Small or Medium (`rfdetr==1.11.2`, Apache-2.0), fine-tuned on `player`, `referee`, `ball`, `rim` and optionally `number` [36][37][2]. This matches Roboflow's basketball pipeline [38].
  - Export to CoreML FP16 (experimental). Eager MPS sends an RF-DETR op to the CPU [22][39].
  - Ball: re-detect on a 640×640 native-resolution crop around a single-ball Kalman prediction. Sweep with `sv.InferenceSlicer` when the ball is lost [40][2].
  - YOLO26 only with an Ultralytics Enterprise licence, because AGPL §13 covers network use [31][32][41].
- **Accuracy.**
  - COCO AP50:95: 53.0 for Small, 54.7 for Medium [36].
  - Basketball ball+rim with RF-DETR-small: mAP50 88.84 [16].
- **Latency.**
  - M3 Pro, `RFDETRNano` at 384 px, FP16: 20.8 ms p50 (`CPU_AND_NE`) and 21.2 ms (`ALL`) [22].
  - So run full-frame detection at 15 Hz with Kalman fill-in, keep the ball path at 30 Hz, and benchmark p95 over 10+ minutes for thermals [2].
- **Failure modes.**
  - The ball is about 16 px at native 1080p and blurs.
  - Heads and spare balls get detected as the ball.
  - A held ball vanishes into the hands, so possession comes from the track [2].

### 2.2 Tracking and identity
- **Tracker.**
  - `trackers==2.6.1` (Apache-2.0): `ByteTrackTracker` for a fixed camera, `BoTSORTTracker(enable_cmc=True)` for broadcast. Pass the BGR frame to `update()`, because CMC is silently skipped when `frame=None` [42].
  - `sv.ByteTrack` was removed in supervision 0.31.0 [40].
  - Associate in court coordinates with at most 5 players per team, and reacquire long-lost IDs (Basketball-SORT) [43][2].
- **Identity.**
  - Two levels: a short-lived tracklet, and a `(team, jersey)` identity mapped to the roster. Only the identity layer feeds stats [2].
  - Team: `google/siglip-base-patch16-224` → UMAP(3) → KMeans(2), fitted at warm-up and voted per track [38][44].
  - Jersey: legibility gate → torso crop → 101-class CNN (00-99 + illegible). Read every 5 frames, confirm after 3 agreeing reads, then apply the roster constraint [38][2].
  - SAM2 and SAM 3 stay offline: 1-2 FPS on a T4, and CUDA 12.6+ respectively [38][45].
- **Accuracy.**
  - SportsMOT HOTA: ByteTrack 73.0, BoT-SORT 73.8 [46].
  - On a fixed basketball camera, Basketball-SORT raised HOTA from 59.70 to 63.35 and cut ID switches from 23.2 to 9.3 [43].
  - Jersey reads: ResNet-32 93%, fine-tuned SmolVLM2 86% [38].
- **Failure modes.**
  - Screens merge bodies, and appearance ReID is weak within a team [47][48].
  - Partial numbers and broadcast cuts break tracks. Reset at each cut and recover identity from team, number and position [2].
  - Pinnies have no numbers, so they need a coach tag.
  - The Koshkina jersey pipeline is CC BY-NC 3.0, so re-implement it rather than reuse it [49][50].

### 2.3 Court mapping
- **Approach.**
  - Fixed camera: undistort, calibrate once from ≥ 8 landmarks with RANSAC, and reproject the landmarks about every 5 s to catch a bumped camera.
  - Broadcast: 33-point court keypoints per frame, then RANSAC, temporal gating and a CMC fallback. Close-ups are marked `uncalibrated` [2][51].
  - roboflow/sports' `ViewTransformer` has no RANSAC, so write our own [52].
  - One `CourtSpec` per league. Source values may be in metres; code stores feet (D1). The basket centre sits 1.600 m from the baseline in the NBA, NCAA and NFHS, and 1.575 m in FIBA [53][54].
  - 3-point test: arc radius plus corner offset plus junction (NBA: 4.328 m from the baseline). The line is 2-point territory, and the test uses the last grounded foot position [3][54][55].
- **Accuracy.** KaliCalib: test MSE 126.61 cm, with 76.5% of images under 1 m [56].
- **Failure modes.**
  - End-to-end symmetry confuses the two halves.
  - Collinear points and lens distortion.
  - Jumping players: use ankle keypoints [57].
  - College floors with several painted arcs [3].

### 2.4 Event inference
- **Approach.** A rule-based state machine over tracks, with confirmation holds and typed confidences.
  - Ball states: held → dribble → flight → loose. A release is the ball moving upward about 0.35 m from the wrist, followed by a parabola fit [16][2].
  - **Make/miss fuses cues:** rim-plane crossing; drop speed after the net (10-90% of free fall means a make; 100% means an air ball behind the rim that never touched the net; rim bounces are handled separately as `rattled`); net motion; and `ball-in-basket` detector votes (that class comes from the player-identification post, not the shot evaluator). Unclear shots are tagged `rattled` or `occluded`, never guessed [16][38]. How these cues become one calibrated P(make) is specified in §7A.2.
  - Shot-clock resets depend on rim contact: 24/14 in the NBA and FIBA, 30/20 in the NCAA, a full 35 reset in the NFHS [54][58][59][60][61]. That makes `ball_rim_contact` the most consequential CV event [3].
  - In broadcast, a view classifier keeps replays from generating events [2].
- **Accuracy references.**
  - Learned online NBA event localisation: about 2.1 s end error and 71.4% event accuracy [20].
  - PlayNet: F1 0.682 [62].
  - CourtMotion's trajectory model: shot-taker AP 0.91, 0.8 s ahead [63].
- **Latency.** Logic takes ≤ 10 ms. Holds: 150-300 ms for a make, 300 ms for a miss or rebound, 500 ms for a steal, 2-5 s for an attributed foul [4].
- **Failure modes.**
  - Front and back rim look alike in 2D [21].
  - The net hides the ball.
  - Replays duplicate events.
  - Drills break the 10-player assumption.
  - Fouls need a whistle, a clock stop and a referee signal or official data [4].

### 2.5 Stats
- **Canonical grammar.** Adopt the FIBA Statisticians' Manual 2024, the only complete public manual [64]:
  - A fouled shooter who misses gets no FGA.
  - A block requires an FGA.
  - Offensive goaltending counts as a turnover.
  - Team rebounds and team turnovers are first-class entities.
- **League flags.**
  - FIBA assists count free-throw trips; NBA official assists do not [64][65].
  - Foul-out is 6 in the NBA and 5 in FIBA and the NCAA [54][59][66].
- **Possessions.** Count them from events. The box-score estimate is `FGA − OREB + TOV + 0.44·FTA`, while NBA play-by-play implies an FT coefficient of 0.42-0.43 [67][68].
- **Two tiers.** `official` (operator-confirmed) and `provisional` [5].
- **Sanity priors (NBA 2025-26).** At the rim .673, corner 3 .387, above-the-break 3 .345, FT .783 [67][69]. These are NBA-only. xeFG coach cards ("eFG 61% on expected 49%") are suppressed outside the NBA until per-league priors exist from our own games; non-NBA cards show raw eFG with its sample size (§7A.2).
- **Failure modes.** Assist and rebound credits are noisy, and official play-by-play is edited after games [6][70].

### 2.6 Narration
- **The LLM never decides facts.** The versioned fact store produces a **FactPacket**: only the facts a line may use, with a pre-formatted clock, `allowed_numbers`, code-computed `derived` phrases, storylines and recent lines [4].
- **Live calls (P0-P2)** are templates with ≥ 8 variants each, rendered in under 1 ms [4].
- **Claude writes P3 colour, summaries and coach digests.** Every sentence must pass:
  1. A deterministic validator (target < 5 ms): entities, numbers, claims lexicon, score consistency, tense, length, repetition [4].
  2. Then the Jev guard (§3) [7].

  Precedents: MARIO's "one short, factual sentence" over machine-validated fields [71]; Mori et al.'s template-then-restyle [72]; IBM's regex plus multi-LLM validation [73].
- **Claude call.** `client.beta.messages.stream(model="claude-opus-5-5", output_config={"effort": "low"}, betas=["server-side-fallback-2026-07-01"], fallbacks="default", max_tokens=1536)`.
  - Thinking counts toward `max_tokens`, so 1536 wins over the 400 in `sdk_ground_truth.md` (D13).
  - Never send `thinking={"type": "disabled"}`, `budget_tokens`, `temperature` or a prefill.
  - Check `stop_reason` for `refusal` [8][23].
  - Cache a byte-stable system prefix of ≥ 512 tokens. Pre-warm it with a non-streaming `max_tokens: 0` request [23].
- **Hiding latency.**
  - Generate 2 speculative candidates per dead ball, in parallel with playback. In Kawamatsu et al. this cut silence from 9.5 s to 0.3-0.4 s [74].
  - Do **not** plan on the narration note's "1.5 s p50 / 3 s p95" (unverified [4]). Every Claude deadline comes from the latency harness in §7A.3. Until it runs, Claude is limited to timeouts, period breaks and state-independent topics (D6).
  - Haiku 5.5 is no automatic win: 8.39 s to first token at `low` [75]. Unlike Opus 5.5, it accepts `thinking: {"type": "disabled"}` at effort `low` to `high`, so the harness tests that configuration [23].
- **Pacing.**
  - Human gaps run 1.7-2.14 s [72][74], and NBA commentary sentences average 13.2 words [20].
  - Live calls are capped at 4-10 words, and about 20% of lines are colour [4][72].
- **Failure modes.**
  - Unbounded context repeats old claims [76].
  - Salience misses, like ESPN's recap that never mentioned Alex Morgan [77].
  - A server-side fallback can switch the serving model for about an hour, so log `response.model` [23].
- **Cost.** Input is about $0.04 (cached) to $0.72 (uncached) per game, excluding output and thinking at $20/MTok [4].

### 2.7 TTS, coach dashboard, accessibility
- **TTS.**
  - `AVSpeechSynthesizer` via PyObjC (`pyobjc-framework-AVFoundation` 12.2.2). First buffer 12-26 ms warm and 0.6-1.0 s cold, so warm it at boot. Calling `say` per line costs 0.78-2.81 s [9].
  - Word-boundary cuts: map the write-path range callbacks to sample offsets in our mixer [9].
  - Upgrades: Kokoro-82M (Apache-2.0, but it needs GPL espeak-ng) [78][79], or ElevenLabs `eleven_flash_v2_5` at about 75 ms. With ElevenLabs, send `flush: true` so short lines are not held back [80][81].
  - Use licensed stock voices only. The ELVIS Act covers simulated voices [82].
- **Coach channel.**
  - Cards of ≤ 12 words, at most 3 visible, each with its sample size. Audio only at dead balls [4].
  - Q+1 foul-trouble alerts [83].
  - Runs shown as information, not "call timeout", because timeout effects are weak or disputed [84][85][86].
  - Lineup +/- hidden below about 20 possessions [87].
  - NFHS Rule 1-19 allows devices for coaching and statistics [88]. The NCAA bench rule is unverified [89][90].
- **Accessibility.**
  - The same events drive WebVTT live captions (WCAG 2.1 SC 1.2.4) and an audio-description track [91]. The caption transport per ingest target, the timing rule and the AD vocabulary are in §7A.8.
  - US public schools face WCAG 2.1 AA deadlines in 2027-2028 [92].

### 2.8 Latency budget (design targets [4][9][26])

| Stage | P1 live call (template) | P3 colour (Claude) |
|---|---|---|
| Capture + decode | 33-70 ms | same |
| Detection + tracking | 30-80 ms (unverified; benchmark matrix and degradation ladder in §7A.4) | same |
| Event engine | ≤ 10 ms | same |
| Confirmation hold | 150-500 ms | n/a (the stoppage is the trigger) |
| Templates / validator | ≤ 5 ms | ≤ 5 ms per sentence |
| Jev | not on the path | ~0.4 s median, run concurrently |
| TTS first audio | 12-26 ms warm | same |
| **Total** | **≈ 0.3-0.7 s after the event resolves** | **Unknown until the §7A.3 harness runs. A fresh call at AA's 6.52 s TTFT [24] lands ≥ 6.5 s after the stoppage. ~0.05 s from a pre-generated candidate** |

- Streamed overlays can delay the video by the p95 event-to-audio latency (about 0.8-1.0 s) [74][4].
- If throughput falls short, run near-live with a 2-5 s delay [2].

---

## 3. Where Jev fits, and where it doesn't

**Jev's properties.**
- Questions in one request share a state, run independently and in parallel, and their IDs are never sent to the model [93].
- Jev is trained with RLCD for calibration and is not fine-tuned per customer. We adapt it only through `state`, `instructions` and `criteria` [19].

**SDK facts that shape the code.**
- **Package.** `typesafe-sdk==0.7.5` (MIT, Python ≥ 3.10, built on `httpx2`) [94][95]. The async call is `AsyncTypeSafeClient(...).system_one(state, questions, *, model, timeout, retry)` [96][97].
- **Questions** are keyword-only objects. Positional forms raise `TypeError` [98][99][8][4].
  - `Choice(instructions=, criteria={label: description})` takes 1-255 labels.
  - `Score(instructions=, criteria=[...])` takes 1-10 ordered levels.
  - `Noul(instructions=, criteria=NoulCriteria(true=, false=))`.
- **Answers.** [100][101]
  - `resp.choices[k]` has `.choice`, `.confidence` and `.probabilities`.
  - `resp.scores[k].score` is an expected, possibly fractional level, and its probabilities are keyed by int.
  - `resp.nouls[k].noul` is P(yes) with no confidence field. Use |2p−1| when a confidence is needed.
- **Pin `jev-1.13.0`.** The `jev-latest` alias moves. `jev-1.14.0` is needed only for `Box` [19][102]. Log `resp.model`, and treat any other version as untuned [7].
- **Hard deadlines.** The default `RetryPolicy` can block about 31.5 s, because its budget never cancels an attempt already in flight. Wrap every call in `asyncio.wait_for` [95][103].
- **One request per trigger.** One call with 13 questions took 0.27 s, against 2.71 s for 13 separate calls [104].
- **Known weaknesses.** Jev "does not count reliably", reads times as text, favours the first option and degrades on large states [105]. So Python computes every number and sends named buckets, and high-stakes Choices are asked twice in reversed option order and averaged [7].
- **Images** are "dreamer-ready" only: they need an opt-in header and 280-1568 px per side, and out-of-range images are never resized [106].

**Call-site verdicts** [7]:

| # | Site | Verdict | What Jev answers | Initial deadline (reset from measured p95, §7A.3) |
|---|---|---|---|---|
| 1a-c | Speak now / which event / interrupt | **Reject** | Nothing. The deterministic director handles priority, TTLs and preemption | — |
| 1d | Dead-ball line kind | Accept (flavour) | `Choice next_line_kind` (stat_note / tendency / storyline / silence), plus one `Noul` per flagged storyline | 1.0 s |
| 2 | Moment stakes | Accept (flavour) | 4-level `Score moment_stakes` and `Noul highlight_play`. Python still computes clutch, margin and runs | 1.5 s |
| 3 | Play type | Accept (hybrid) | Synergy-style `Choice`, order-ensembled, plus `Noul summary_is_clear`. Crisp Transition and Putback rules win | 1.5 s |
| 4 | Decision quality | **Reject** | Python buckets `xP_chosen − max xP_alt` | — |
| 5 | Ambiguous make/miss, shot vs pass | Accept, narrowly | Order-ensembled `Choice`. Commit as **provisional** only when p_max ≥ 0.85, both orders agree and the rule engine leans the same way. Fouls and 2-vs-3 are rejected | 0.7 s |
| 6 | Coach-card ranking | Accept | One `Noul relevant__<card>` per code-made card, plus `Choice top_card` with a `none` option | 1.0 s |
| 7 | Guard on Claude sentences | Accept, as the second check | Per sentence: `supported` ≥ 0.90, `unlisted` ≤ 0.20, `speculation` ≤ 0.30 | 0.8 s |
| 8 | Images / `Box` | Reject live; cold path behind a flag | Review of unresolved make/miss; rim and scorebug location at setup (`jev-1.14.0`) | 5 s |

**Never use Jev for** [7]:
- the P0/P1 path;
- arithmetic, counting or magnitude comparisons;
- identity;
- text generation;
- box-score changes on its answer alone;
- fouls, 2-vs-3 or rules interpretation;
- live frames;
- fan chat mixed with game state.

**Operating model** [7]:
- One `JevAdvisor` asyncio pool (≤ 8 requests in flight) writes annotations with `{site, value, p, model, request_id, state_version, latency_ms}`.
- Each consumer computes its deterministic fallback first, so a missing answer only makes output plainer, never wrong.
- Modes: `live`, `stub` (the real SDK over `httpx2.MockTransport`), `replay` and `off`.
- Rollout: shadow mode for ≥ 5 games, then promote one site at a time. Run an `off` chaos test.
- Every site has a kill criterion. For example, B5 is deleted once a calibrated classifier on the same cues matches it.
- Load: about 450-560 requests per game.

---

## 4. Accuracy strategy

### 4.1 Confidence gating (live path, CV confidences only [4])

| Event | Hold | Say it if | Otherwise |
|---|---|---|---|
| Release call | 0 ms | P(shot) ≥ 0.90 | No call |
| "Three" | 0 ms | P(3) ≥ 0.97 | "From the wing"; the scoreboard corrects it |
| Made FG | 150-300 ms | P(make) ≥ 0.95 | 0.80-0.95: wait up to 1.0 s for the scoreboard delta, else silence |
| Rebound / steal | 300 / 500 ms | ≥ 0.90 | Team-level line |
| Player name | — | ID ≥ 0.95 | ID 0.80-0.95: "Boston's number 0" if OCR ≥ 0.95; else the team name |
| Foul, attributed | 2-5 s | ≥ 0.97 or official data | "Whistle, foul on the floor." |

- Hedged lines are capped at 1 in 3.
- More than 2 audio corrections in 5 game minutes raises the gates by +0.03 and the holds by +200 ms [4].
- **What P(·) means.** Each P in this table is the output of that event's cue model in §7A.2, a logistic model over named CV cues. Until the model for an event is calibrated and its gate certified on held-out games, the gate is never met: that event runs **hedge-only** (no outcome words until the scoreboard, the next-possession pattern or the operator confirms).

### 4.2 Scoreboard and official-feed reconciliation
- **The scoreboard is authoritative.**
  - A score change needs 2 identical consecutive OCR reads at 5 Hz, i.e. about 400 ms, plus a clock-consistency check (D14) [4][6]. MARIO likewise fires goals only on an OCR score change [71].
  - In broadcast, the scorebug is read at 5 Hz (D14 supersedes the CV note's 1-2 Hz) and confirms a make within about 3 s [2]. The fixed-camera scoreboard reader and the audio-free dead-ball detector are specified in §7A.5.
  - Fixed-camera mode has no scorebug: read the venue scoreboard if visible, and require a manual score override [5].
- **Provisional (‡) results reconcile deterministically**, in order: scorebug delta; next possession (a baseline inbound means a make, a rebound means a miss); official feed; post-game re-tracking [7].
- **Official play-by-play is edited.** Re-pull it after the buzzer and the next morning, and version it by `edited` [6][70].
- **Spoken corrections** [4][107]:
  - A revision made before speech stays silent.
  - A wrong material fact that was already spoken gets a P0 "Correction:" line of ≤ 12 words.
  - Replay reviews get a hedge until they resolve.

### 4.3 Human in the loop
- **Operator controls from day one:** score override, roster mapping, mute hotkey, and correct-event with box-score recompute [5].
- **Review queue order** [6]:
  1. Play-by-play vs engine disagreements.
  2. Low Jev confidence.
  3. Tracker uncertainty.
  4. New venues.
- **Labelling budget:** about 300 frames per new venue and about 200 events per game [6].
- **Tools.** Label Studio (Apache-2.0) for timelines and gold sets [108]. CVAT Community (MIT) for boxes; it lacks the SAM 2/3 tracker, which is in paid editions [109][110].

### 4.4 Evaluation metrics and targets

| Area | Metric → target |
|---|---|
| Fan narration (U8) | p95 event-to-speech ≤ 3 s; **0 wrong scores spoken**; ≤ 1 wrong-player attribution and ≤ 2 hallucinated events per game [5] |
| Audio description (U9) | p95 ≤ 2 s; possession-change accuracy ≥ 97%; score 100% (operator-confirmed) [5] |
| Coach (U1) | Team points 100% after operator check; made-FG P ≥ 0.98 / R ≥ 0.95; foul attribution ≥ 0.95 before live foul alerts [5] |
| Event spotting | mAP@0.5/1/2 s; tight average-mAP over 1-5 s [111]; F1 with start and end within ±2 s [20] |
| Tracking | HOTA and IDF1 [112], plus a court-plane variant; identity accuracy ≥ 95% [6] |
| Box score | Per-stat MAE; points error = 0 on ≥ 95% of games [6] |
| Lineups (new) | Share of game seconds in which each team's recorded five is correct: ≥ 98% after operator entry; p95 sub-entry latency (§7A.8, proposal) |
| CV cue models (new) | 15-bin ECE ≤ 0.03 with bootstrap CI; each gate certified by 0 errors in ≥ 3/ε held-out accepted events (§7A.2) |
| Shot / pass models | Log-loss, Brier, 15-bin ECE ≤ 0.03 [113][6] |
| Jev questions | ECE ≤ 0.05 after isotonic recalibration, on ≥ 200 cases per site; flip rate ≤ 5% under option shuffle and paraphrase [7][6] |
| Jev guard | Catches ≥ 0.95 of synthetic and ≥ 0.85 of real bad lines, at ≤ 10% false rejects [7] |
| Narration facts | Atomic-claim precision ≥ 99% (FActScore-style) [114]; 0 score or clock errors; identity, event and action accuracy (NBA_Streaming identity baseline 28.5%) [20]; human audit of ≥ 100 lines per game [6] |
| Timing | Speaking/silence mIoU vs humans at 1 Hz [74]; per-stage p50/p95 from `t_occ` to `t_audio_out` [4] |

---

## 5. Training "off live games"

### 5.1 Ground rules
- **Shipped models train only on rights-cleared footage.** Public and NBA data are R&D only. Every row carries `rights_status ∈ {own, licensed, research_only}`, `label_origin` and `label_conf` [6].
- **Processing and training consent are separate.** Store a per-game `training_allowed` flag [5].
- **Jev cannot be trained:** the same weights serve every account [19]. Instead, keep gold sets, per-question ECE, isotonic recalibration and thresholds [6][101][113].
- **Claude improves through evaluation.** Refresh its exemplars and prompt rules from the error log [6].

### 5.2 Data flywheel [6]
0. **Bootstrap, R&D only (weeks 1-3).**
   - SportVU 2015-16 (631 games, 25 Hz, no licence stated) for event, shot and pass models v0 [115][116][117].
   - The Roboflow sets for the detector, court keypoints and jersey reader [38][51].
   - Benchmarks: DeepSportradar, SportsMOT, TeamTrack [27][28][118].
   - **Freeze gold sets:** 10 fully annotated games; 2,000 frames across ≥ 6 venues; 500 Jev questions; 300 narration lines.
1. **Ingest.** Video with frame PTS and capture UTC. Play-by-play polled every 2-5 s, keeping every revision. Scorebug OCR at 5 Hz with PP-OCRv5 (Apache-2.0) [119].
2. **Auto-label.**
   - Teacher stack: RF-DETR-L with SAHI [120], SAM 2.1 / SAM 3 [45][121] (SAM 3 also has an HF Transformers port; MPS speed unverified [122][123]), and a TrackNet-style ball model [124].
   - Filter pseudo-labels on ≥ 3-frame persistence, ballistic fit, roster caps and play-by-play agreement.
3. **Mine disagreements and run human QA** (§4.3).
4. **Distil.**
   - Noisy-Student RF-DETR-S/M [125].
   - The event engine on our tracks, with a bootstrapped loss (β 0.8) [62].
   - Recalibrated shot and pass models [113][126].
5. **Gate** on the frozen gold sets (§4.4).
6. **Deploy and monitor drift:** ball-missing rate, team-cluster margin, OCR failures, Jev confidence histograms, unsupported claims. Retrain weekly.

Train the event engine mainly on **trajectories**, which are cheap on MPS and well covered by public data. Use small video models only for make/miss, fouls and scrums [6].

### 5.3 Weak supervision via play-by-play alignment
- **Precedents.**
  - BasketEvent: 226 games, 34,931 clips located by game ID + clock [62].
  - NBA_Streaming: clock OCR [20].
  - BasketballBench: 1 fps scorebug reading with a countdown-consistency filter. It reports that play-by-play lags the video by 1-2 s [127].
- **Algorithm** (design proposal [6]):
  1. Read period, clock, shot clock and scores from the scorebug at 5 Hz.
  2. Drop frames where the clock is not monotone, and replays.
  3. Fit `t_video = f(period, clock)` with robust lines of slope ≈ −1, treating stopped clock as plateaus.
  4. Place live-ball events with a per-league lag. Order dead-ball events by `actionNumber`, anchored on score deltas.
  5. In live mode, use Δ = `t_capture_utc − timeActual`, the median over made baskets. Whether `timeActual` is the action time is unverified [70].
  6. Snap each event to a CV anchor: release, rim contact, or first hand contact on a rebound [128].
  7. Accept a game only if the median residual on made baskets is under 0.5 s.
- **Field names drift.** The nba_api field docs were last validated in 2023 and 2020, so check them against a live response [70][129].
- **Without play-by-play** (rec, high school, own games), the scorebug alone gives near-free labels [6]:
  - A +1, +2 or +3 score delta is a made FT, 2 or 3.
  - Clock stops, starts and shot-clock resets mark whistles and possession changes.
- **Expected yield.** About 155 clips per game, or around 10⁴ weak clips per 100 games [62][6].

### 5.4 Legal constraints

| Source | Constraint | Use |
|---|---|---|
| NBA video and stats | NBA.com terms bar commercial, gambling, fantasy and live play-by-play products, and comprehensive stats databases, without consent [15][130]. nba_api's MIT licence covers the code, not the data [131] | R&D only, quarantined |
| NCAA championships | NCAA copyright; licensed via Veritone / TNT Sports [14] | No |
| YouTube | Terms ban scraping and downloading [132] | Never |
| Fair use (US, 2025) | *Bartz*: training is "exceedingly transformative", but pirated copies are not covered. *Kadrey*: market-dilution warning. Copyright Office: market harm can defeat fair use [133][134][135] | How footage is acquired matters |
| Datasets | DeepSportradar CC BY-NC-ND [27]. SportsMOT, PL-NBA and NBA_Streaming annotations CC BY-NC [28][29][30]. BARD's video is NBA content [136]. SportVU has no licence [116][117]. Universe licences are unchecked [2] | Eval and pre-training only |
| Code and weights | RF-DETR N-L Apache-2.0; XL/2XL PML 1.0 [36]. YOLO26 AGPL [41]. SAM 3 under the SAM License [45]. T-DEED GPL-3.0 [137]. Koshkina pipeline CC BY-NC [50] | Ship Apache/MIT code only |
| High school | State associations control rights. NFHS Network has postseason exclusivity in KS, GA and NY [138][139][140] | Regular-season games the school films |
| Privacy | BIPA covers face geometry [141]. COPPA counts gait patterns and faceprints as biometric [142]. Basketball Australia's HoopsTV drew a backlash over opt-out filming of children [143][144]. 13 states require all-party consent to record [145] | No faces. Embeddings live in RAM for one game. No under-13 keypoints without parental consent. Mic off by default |

**Resolved default.** The narration note uses whistle *audio* [4], but the market note defaults audio capture to off [5]. **Audio capture stays off by default.** Dead balls come from the audio-free detector in §7A.5. An on-device, non-recording whistle-band detector is an opt-in option that needs counsel's sign-off first (§7A.13). Minors' data sent to cloud processors is covered in §7A.7.

---

## 6. Use cases ranked by value × feasibility

V and F are the market note's 1-5 scores [5]. V×F is our arithmetic. The priority is the note's own, which also weighs legal risk.

| Use case (users) | V | F | V×F | Legal | Priority |
|---|---|---|---|---|---|
| Live box score + event log (coaches, streams, fans, leagues) | 5 | 4 | 20 | M | **P0** |
| Post-game coach report (U1, U2) | 5 | 4 | 20 | L-M | **P0** |
| Audio description + live captions (U9, U8) | 4 | 4 | 16 | L | **P0** |
| Player reels + per-player stats (U11, U12) | 4 | 4 | 16 | **H** | P1, opt-in |
| Grounded live narration, commentator-less streams (U8) | 5 | 3 | 15 | M | **P0** |
| In-game coach alerts (U1) | 4 | 3 | 12 | L | P1 |
| Second-screen Q&A (U10) | 3 | 4 | 12 | M | P1 |
| Clip library (U2) | 4 | 3 | 12 | L | P1 |
| Scouting reports (U5) | 4 | 3 | 12 | M | P1 |
| Analytics export (U3) | 3 | 4 | 12 | L-M | P1 |
| League multi-court batch (U14) | 4 | 3 | 12 | M | P2 |
| Player development (U4) | 3 | 3 | 9 | M | P2 |
| Officiating clip packs (U7) | 2-3 | 3 | 6-9 | M | P2 |
| Trainer load (U6) | 3 | 2 | 6 | H | P2 |
| NBA/NCAA narration; betting feeds | 4-5 | 3-4 | — | **H** | **Not pursued** |

**Betting is excluded.** Sportradar holds official NBA data through 2031 [13], Genius holds NCAA postseason data through 2032 [146], and Virginia makes youth-game wagering a misdemeanor [147]. The defaults are anti-wagering terms and an optional 30-60 s stats delay [5].

**MVP features per user** [5]:
- **Stream operator (U8).** Ingest by default only from sources the customer controls: OBS, an iPhone, BallerCam or the school's own camera pushing RTMP to us. Pixellot/NFHS Network, Hudl Focus and Veo are integrations that need a written partner agreement before launch (§7A.12). Score/clock sync with manual override. Gated play-by-play, with names only when attribution passes. Box-score-only colour. A separate TTS track with a mute hotkey.
- **Blind and low-vision fans (U9).** A terse audio-description mode covering possession, ball zone, shot and result, and score every possession, plus WebVTT captions.
- **Head coach (U1).**
  - Post-game (P0): box score with player stats marked provisional, shot chart, runs, lineup +/- and minutes from **operator lineup entry** (tap-to-sub at dead balls, §7A.8), clips, a Claude summary citing event IDs, and a correction UI.
  - Live (P1): a tablet with fouls, bonus, timeouts and runs, with alerts ranked by Jev.
- **P1 group.** Clip search for video coordinators; Parquet tracks with confidences for analysts; Q&A that refuses anything outside the event log; consent-gated reels with no school marks.

---

## 7. Build roadmap and risks

The phase order synthesises the notes' priorities. Only Phase A has a duration in the notes (weeks 1-3 [6]), and the exit milestones are the §4.4 targets.

| Phase | Build | Exit milestone |
|---|---|---|
| **A. Foundations** | **First:** the sustained M3 Pro benchmark matrix (§7A.4) and the Claude/Jev latency harness (§7A.3). Pin `rfdetr` 1.11.2, `supervision` 0.31.0, `trackers` 2.6.1, `typesafe-sdk` 0.7.5 and `anthropic` 1.15.0 [2][8]. Add `RuleSpec` YAML (§7A.6), the CoreML parity test (§7A.10) and the first-party data sprint plan (§7A.11). Build the versioned fact store and an event schema with provenance, confidence and revisions first [4][5]. Add `CourtSpec`, SportVU v0 models, gold sets, and mock-transport tests for Jev and Claude [3][6][7] | Gold sets frozen; offline tests green |
| **B. Fixed-camera MVP** | Detection and ball-ROI rates chosen from the §7A.4 benchmark, ByteTrack, homography with drift check, make/miss cue model in hedge-only mode until calibrated (§7A.2), venue scoreboard reader and audio-free dead-ball detector (§7A.5), team-level stats, operator UI with tap-to-sub lineups (§7A.8), post-game report [2][5] | 10+ minutes sustained within budget; team points 100% after operator check; made-FG P ≥ 0.98 / R ≥ 0.95 |
| **C. Live voice** | Director, P0-P2 templates, AVSpeech, corrections, captions, AD mode [4] | p95 ≤ 3 s (≤ 2 s for AD); 0 wrong scores spoken; ≤ 2 hallucinated events per game |
| **D. Claude colour + Jev in shadow** | Speculative candidates, validator, B1-B4 in shadow for ≥ 5 games, then promotion site by site [7] | Claim precision ≥ 99%; each site passes its gate or is killed |
| **E. Player attribution + live tablet** | Jersey voting, roster constraints, Q+1 alerts, consent registry [2][5] | Identity ≥ 95%; foul attribution ≥ 0.95 |
| **F. Broadcast + flywheel** | Scorebug OCR, view classifier, BoT-SORT with CMC, play-by-play alignment on licensed feeds, weekly retraining, multi-court [2][6] | Alignment residual < 0.5 s; points error 0 on ≥ 95% of games |

**Top risks**

| Risk | Evidence | Mitigation |
|---|---|---|
| A competitor ships AI commentary first | Pixellot ($35M raise) announced it [33]; Superstat raised a $3.5M pre-seed [148] | Win on grounded, uncertainty-aware narration and accessibility; treat cameras as ingest partners [5] |
| M3 Pro throughput | Nano alone takes ~21 ms [22]; sustained use is slower than burst [149] | 15 Hz detection, process isolation, frame dropping, near-live mode [2] |
| Wrong player or make spoken aloud | 28.5% identity at the state of the art [20]; rim ambiguity [21] | Identity tiers, holds, hedges, provisional stats, reconciliation |
| Claude latency and fallback drift | 6.52 s TTFT [24]; the fallback model sticks for ~1 h [23] | Templates own the live path; pre-generation; log `response.model` |
| Jev drift or outage | No SLA; rate limits unpublished; aliases move [19][25] | Pin versions, hard deadlines, circuit breaker, `off` chaos test [7] |
| Licence contamination | AGPL, CC BY-NC, NBA terms [31][27][15] | `rights_status` on every row; ship Apache/MIT code only |
| Minors' privacy | COPPA biometrics; HoopsTV backlash [142][143] | On-device processing, no faces, opt-in per-player outputs |
| Library API drift | `sv.ByteTrack` removed; Ultralytics' default tracker changed [40][150] | Pin versions; set tracker configs explicitly |

**Still open** [7][5]: real Jev latency from the venue network; whether Jev answers are deterministic (which decides whether caching is safe); how often ambiguous makes occur; how good Pixellot's commentary is; the current NCAA bench-device rule; and whether one M3 Pro can handle two 1080p streams. The status of each, with the plans and the new gaps from the 2026-10-20 pass, is in §7A.13.

---

## 7A. Gap closures (2026-10-20 pass)

Each subsection answers one gap a reviewer found in this report and the notes. Design choices are marked **proposal**. Facts carry citations. The same material, in more detail, is in the addenda at the end of each affected note.

### 7A.1 Contract conflicts
Resolved in §1A (D1-D16). ARCHITECTURE.md and pyproject.toml were edited to match.

### 7A.2 From CV cues to calibrated probabilities

Every live gate in §4.1, and B5's trigger band `p_make_rule ∈ (0.05, 0.95)`, assumes a calibrated probability. The rule engine produces cues, not probabilities. The fix is one small logistic model per gated event, in `events/calibration.py` (**proposal**):

| Event | Named cue features (computed in Python) |
|---|---|
| Shot attempt | `release_sep_ft` (ball-wrist separation while rising; Roboflow uses 0.35 m ≈ 1.15 ft [16]), `upward_speed`, `parabola_rmse_px`, `parabola_n_pts`, `toward_rim_cos`, `in_frontcourt` |
| Make | `crossed_inside` (descending rim-plane crossing inside the rim span shrunk by the ball radius), `crossing_margin_px`, `drop_in_band` (10-90% of free fall [16]), `drop_is_freefall` (≈ 100%: an air ball behind the rim that never touched the net [16]), `net_motion_z`, `bib_conf_max` (`ball-in-basket` class [38]), `rim_contact`, `occluded_frames`, `view_quality` |
| 3-point | `foot_margin_ft` (signed distance of the last grounded foot from the line, D8), `foot_obs_quality`, `homography_rmse_ft`, `in_corner` |
| Rebound, steal, block | control duration, holder distance, track confidence, team-assignment margin |
| Player ID | jersey vote share, number of agreeing reads, roster constraint satisfied, team margin |

- **Cold start.** Initial weights are set by hand so a textbook case scores high and any conflict between cues lands mid-range. Example for make: `logit = −2.0 + 3.0·crossed_inside + 1.5·drop_in_band − 2.5·drop_is_freefall + 0.8·min(net_motion_z, 3) + 1.0·bib_conf_max − 1.0·[occluded_frames ≥ 3]`. These numbers are placeholders, not findings.
- **Status per model:** `uncalibrated` → `platt` → `isotonic`. Each event carries `data["p_status"]`.
  - `uncalibrated`: no gate is ever met, so the event runs **hedge-only**. Templates may say "shot from the wing" or "rebound, Home", with no outcome or point value. Outcome words wait for the scoreboard delta, the next-possession pattern or the operator. B5 is off, because its trigger band has no meaning yet.
  - `platt`: refit the weights (L2 logistic regression) and Platt-scale once ≥ 300 labelled events of that kind exist, at least 50 of them in the ambiguous band. Platt scaling beats isotonic regression when the calibration set has fewer than about 200-1,000 cases [480]. scikit-learn recommends isotonic only with "greater than ~1000 samples" [479].
  - `isotonic`: from ≥ 1,000 labelled events.
  - Report 15-bin ECE with a bootstrap 95% CI on held-out games. Target ≤ 0.03 (§4.4).
- **Gate certification.** A gate is switched on only after the held-out events above its threshold show 0 errors in n ≥ 3/ε cases. The "rule of three" gives a 95% upper bound of 3/n on the error rate when no error is observed in n trials [481]. So the made-FG precision target of 0.98 (ε = 2%) needs ≥ 150 accepted makes with no error, and ε = 1% needs ≥ 300.
- **Constant registry.** Every CV constant lives in `EventEngineConfig` as `(value, source, status)` with `status ∈ {sourced, unverified, tuned}`:

| Constant | Value | Status | Note |
|---|---|---|---|
| Make drop band | 10-90% of free-fall speed | sourced | Roboflow reports no make/miss accuracy [16] |
| Air-ball signature | ≈ 100% of free fall | sourced | Behind the rim, never touched the net. Rim bounces go to the `rattled` logic, not this rule [16] |
| Release separation | 0.35 m ≈ 1.15 ft, ball rising | sourced | [16] |
| In-flight ball detection | 98.1% (759/774 frames) | sourced, **not game footage** | Detector trained on a shooting-robot dataset fork; validation recall 84.9% [16] |
| Holder radius | 0.5 m ≈ 1.6 ft | unverified | tune |
| Homography reprojection gate | 1.0 ft (≈ 0.3 m) | unverified | D1; tune |
| Minimum parabola points | 5 | unverified | Roboflow gives no count [16] |
| Net-occlusion frames | not hard-coded | sourced anecdote (2 frames) | [16] |

- **xeFG cards.** Shot-quality priors exist only for the NBA [67][69]. NCAA and high school have no primary source (§2.5). `InsightEngine` suppresses expected-eFG cards when `league != "nba"` until we hold ≥ 2,000 labelled shots for that league tier from our own games (**proposal**). Until then, non-NBA cards show raw eFG with its sample size.

### 7A.3 Cloud latency, `max_tokens` and dead-ball windows

**Facts that bound the design.**
- `claude-opus-5-5` always thinks. `thinking: {"type": "disabled"}` and `budget_tokens` both return 400 at every effort. Effort defaults to `medium`, so set `low` explicitly. A system-prompt line such as "Answer directly without deliberating." can reduce thinking, TTFT and cost; measure quality before keeping it [23].
- Thinking counts toward `max_tokens` even though its text is omitted by default. Hence `max_tokens=1536` (D13) [23].
- Prices: Opus 5.5 is $4 input / $20 output per million tokens, with cache reads at $0.20. Haiku 5.5 is $0.10 / $0.50 for prompts of ≤ 100K tokens. The minimum cacheable prefix is 512 tokens for both [23].
- `claude-haiku-5-5` accepts `thinking: {"type": "disabled"}` at effort `low`, `medium` and `high` (400 at `xhigh`/`max`). It rejects non-default sampling parameters and prefill, and has no server-side refusal fallback [23][8].
- Artificial Analysis lists Opus 5.5 (low) at 6.52 s TTFT and 79.5 output tokens/s, against a cross-model median TTFT of 3.73 s. The page does not say whether its TTFT includes thinking time [24].

**Latency harness (Phase A task 0, proposal).** `tools/latency_harness.py` makes about 50 calls per configuration. It runs from the dev Mac and from a venue-like network (gym Wi-Fi and an LTE hotspot).

| Config | Model | Settings |
|---|---|---|
| C1 | `claude-opus-5-5` | effort `low`, `fallbacks="default"`, `max_tokens=1536` |
| C2 | `claude-opus-5-5` | C1 plus "Answer directly without deliberating." |
| C3 | `claude-haiku-5-5` | effort `low`, `thinking={"type": "disabled"}`, no `fallbacks` |
| C4 | `claude-haiku-5-5` | effort `low`, adaptive thinking |
| J1-J5 | `jev-1.13.0` | bundles B1-B5 with real-shaped states |

- **Input.** A real FactPacket (≈ 1.5k tokens) behind a byte-stable cached system prefix of ≥ 512 tokens, pre-warmed with a non-streaming `max_tokens: 0` request [23].
- **Recorded per Claude call:** time to `message_start`, time to the first text delta, total time, `usage.output_tokens` (includes thinking), `usage.cache_read_input_tokens`, `stop_reason` and `response.model`.
- **Recorded per Jev call:** wall time, `resp.model`, `usage.input_tokens` and question count.
- **Output.** p50/p95 per configuration and network. **Every deadline is set from the measured p95**: Claude's `candidate_deadline_s`, and each Jev bundle's `asyncio.wait_for`. Thinking cost per Opus call = `output_tokens × $20/MTok`.

**Dead-ball windows: rule-defined minimums** (real stoppages are longer):

| Window | NFHS | FIBA | NCAA men | NBA |
|---|---|---|---|---|
| Team timeout | 60 s or 30 s [241] | 1 min [54] | 75 s or 30 s; 60 s in the media format [59] | 1:15 for additional timeouts; mandatory 2:45 local / 3:15 national [55] |
| Between periods | 1 min [241] | 2 min [3] | — (halves) | 2:30 local / 3:30 national TV [3] |
| Halftime | 10 min, up to 15 [241] | 15 min [3] | 15 min [3] | 15 min [3] |
| Free throw | Unmeasured | Unmeasured; the shooter must release within 5 s of having the ball [54] | Unmeasured | Unmeasured |

- Free throws, quick inbounds and made baskets are short and unmeasured. In NFHS a made basket does not stop the clock at all [241]. Measure dead-ball durations on 3-5 own games, from the OCR clock-stopped intervals of §7A.5, and store the distribution per league.
- **Policy until measured:** request Claude candidates only when `expected_window_s ≥ p95_claude_total + tts_s + 1 s`. With p95 unknown, that means timeouts, period breaks and halftime only, plus pre-generated **state-independent** topics (season storylines, program facts) that no `state_version` change can invalidate. If the measured p95 exceeds the median free-throw window, keep this restriction permanently.

### 7A.4 M3 Pro throughput: benchmark matrix and degradation ladder

**Facts.**
- This dev machine is the binned M3 Pro: 5 performance and 6 efficiency CPU cores, a 14-core GPU and 18 GB [9]. Apple's launch figures (12-core CPU, 18-core GPU) describe the top bin [466].
- RF-DETR on CoreML [22]:
  - Only fp16 reaches the ANE. `CPU_AND_NE` on an fp32 package performs like `CPU_ONLY`.
  - fp16 compiles for the ANE in about 5 s on first load.
  - About 7-8 ops (the two-stage query `topk`) run on the CPU.
  - Nano fp16 loses COCO mAP 48.0 → 45.1.
  - Nano 384 fp16 p50 on an M3 Pro: `CPU_ONLY` 44.5 ms, `CPU_AND_NE` 20.8, `CPU_AND_GPU` 33.9, `ALL` 21.2.
  - Packages fix the input shape and batch size, so the full-frame model and the ROI model each need their own export.
- Core ML's synchronous prediction API is not thread-safe; the async API is. Gains from concurrency depend on the model, the compute units and other work on the device, and in-flight predictions should be capped [464].
- **Arithmetic (ours).** Full-frame Nano at 15 Hz uses 15 × 21 ms = 0.32 s of accelerator time per second. A 30 Hz ROI pass at Nano cost adds 0.63 s/s, for 0.95 s/s together. On the GPU, Nano fp16 costs 33.9 ms, so a 30 Hz ROI there would need 1.02 s/s. **A 30 Hz ROI pass on the GPU does not fit.**

**Proposed placement** (confirm by benchmark):
- **ANE, one serial queue:** the full-frame detector at 15 Hz, and the ball-ROI pass at 15 Hz on the alternate frames. Together they give 30 Hz ball observations, at ≈ 0.63 s/s if both cost what Nano costs.
- **GPU (MPS):** SigLIP team embeddings (batched every ~0.5 s) and OCR.
- **CPU:** trackers, homography and the event engine. Decode on VideoToolbox (cost unverified).
- **Processes:** one per CoreML model, so synchronous `predict` calls never share a model across threads [464]. Warm every model at boot to absorb the ~5 s ANE compile [22].

**Benchmark matrix** (Phase A; ≥ 10 min per cell; p50/p95/p99; SigLIP and OCR running at the same time):
- Models: RF-DETR Nano 384, Small 512, Medium 576; a ball-only ROI model at 384 and at 640.
- Compute units: `CPU_AND_NE`, `CPU_AND_GPU`, `ALL`. Precision fp16, plus fp32 on the GPU.
- Concurrency: 1 model; 2 models in separate processes (ANE + ANE, ANE + GPU).
- Every 10 s, log `pmset -g therm` (this Mac reports thermal and performance warning levels through it [9]) and the frame-drop rate.

**Degradation ladder (proposal).** Triggers use the p95 capture→tracks latency over a 10 s window. Each step reverts after 60 s below its threshold.

| Level | Trigger | Action |
|---|---|---|
| L0 | Nominal | Placement above |
| L1 | p95 > 50 ms, or any thermal warning | Pose off; SigLIP refresh 0.5 s → 2 s; jersey reads every 10 frames |
| L2 | p95 > 66 ms | Full-frame detection 15 → 10 Hz |
| L3 | p95 > 80 ms | ROI 15 → 10 Hz; switch the full-frame model to Nano 384 if it was S or M |
| L4 | p95 > 100 ms for 30 s, or > 5% dropped frames | Near-live mode: 3 s delay, with video and narration delayed together |

### 7A.5 Scoreboard, clock and dead balls

- **Rate and confirmation (D14).** OCR runs at 5 Hz. A score is confirmed by 2 identical consecutive reads (≈ 400 ms) whose delta is +1, +2 or +3 and whose clock is consistent with the previous reads.
- **Fixed-camera venue scoreboard (proposal).**
  - At setup, the operator clicks the 4 corners of the scoreboard in the frame. A homography rectifies the crop.
  - The operator marks the digit cells for home score, away score, period, game clock and, if present, shot clock.
  - Each digit cell goes to a small 11-class CNN (0-9 and blank), trained on synthetic seven-segment and LED-matrix renders plus crops from our venues, and exported to CoreML. This avoids general OCR on seven-segment fonts.
  - Benchmark it in the §7A.4 matrix.
- **Broadcast scorebug.** PP-OCRv5 is optional. On macOS, PaddlePaddle supports Apple Silicon (arm64) on the CPU only, for Python 3.9-3.13 (`paddlepaddle==3.3.0` from Paddle's own index) [467][468]. Its M3 Pro latency is unmeasured. Apple Vision's text recognizer via PyObjC is an alternative (unverified).
- **Make-wait band.** No source gives amateur scoreboard-operator lag. Until we measure it:
  - The 0.80-0.95 band waits up to 3.0 s for the scoreboard delta (the CV note's broadcast confirmation time [2]).
  - With no visible scoreboard, it waits for the next-possession pattern: an inbound under the basket means a make, a rebound means a miss.
  - Then measure the lag on own games and set the wait to its p90.
- **Audio-free dead-ball detector (proposal).** `dead_ball` fires on any of:
  1. The OCR game clock is unchanged for ≥ 2 consecutive reads while `RuleSpec` says it should be running.
  2. Free-throw formation: one player stationary at the FT line and ≥ 4 players set on the lane lines, held for ≥ 1.5 s.
  3. Global stillness: median player speed < 2 ft/s for ≥ 2 s, with the ball held by a referee, out of bounds, or not visible.
  4. The operator hotkey.
- **Clock inference without a scoreboard** uses `RuleSpec`'s clock-stop rules (§7A.6). In NFHS, a made basket does not stop the clock [241].

### 7A.6 `RuleSpec`: rules separate from geometry

```python
@dataclass(frozen=True)
class RuleSpec:
    ruleset: str                          # "nba" | "fiba" | "ncaa_m" | "ncaa_w" | "nfhs" | custom id
    period_s: int; n_periods: int; ot_s: int
    halves: bool                          # NCAA men and many AAU / rec formats
    running_clock: "RunningClock | None"  # e.g. clock stops only in the last N s, or mercy at a margin
    timeouts: "TimeoutRules"              # counts per half/game/OT, lengths, carry-over, late-game limits
    possession: Literal["arrow", "jump"]  # held balls and period starts
    bonus: "BonusRules"
    foul_out: int
    tech_ft: "TechRules"                  # number of FTs, who may shoot, where play resumes
    shot_clock_s: int | None; shot_clock_reset_s: int | None
    clock_stops_after_fg: "ClockStopRule | None"
```

It loads from `config/rules/<ruleset>.yaml`, with `overrides/<state or tournament>.yaml` chosen by the operator at setup. Values verified in this pass:

| Rule | NFHS | FIBA 2026 | NCAA men | NCAA women | NBA |
|---|---|---|---|---|---|
| Timeouts | Three 60-s and two 30-s per game; one more 60-s per extra period; unused ones carry over [241] | 2 in the first half; 3 in the second (max 2 once Q4 shows ≤ 2:00); 1 per OT; 1 min each; no carry-over [54] | Non-media format: four 75-s and two 30-s. With ≥ 3 media timeouts per half: three 30-s (up to two carry into the second half) and one 60-s. One more per extra period [59] | Three 30-s plus two or one full timeouts, depending on the media format [60] | 7 per game; at most 4 in Q4, dropping to 2 late in Q4; 2 per OT [55] |
| Held ball and period start | Alternating-possession arrow. The opening-jump winner sets the arrow toward its opponent's basket. Q2-Q4 start with an AP throw-in at the division line [472] | AP arrow. The team that does not gain first control gets the first AP throw-in [54] | AP procedure [59] | AP procedure [60] | Jump ball (unverified in this pass) |
| Clock stops after a made FG | No. Rule 5-8-1 lists fouls, held balls, violations and time-outs only [241] | When the game clock shows ≤ 2:00 in Q4 and in each OT [54] | With ≤ 59.9 s left in the game or any extra period (Rule 2-10, Art. 12) [59] | In the last 59.9 s of the fourth period and extra periods [60] | Last minute of Q1-Q3; last 2:00 of Q4 and each OT (Rule 5, Section V) [55] |
| Running clock / mercy | A state association "may institute a running clock when a specified point differential is reached" (5-5 Note) [241] | — | — | — | — |
| Foul-out | **5, personal and technical fouls counted together** (Case 4.14.1) [473]. This replaces the earlier "(unverified)" | 5 [54] | 5 [59] | 5 [60] | 6 [66] |
| Technical foul | 2 FTs by any member of the offended team, then a division-line throw-in (Cases 10.5.1, 10.6.5) [474] | 1 FT; then the team that had control, or was entitled to it, throws in from where play stopped (Art. 36.3.2) [54] | Not extracted | Not extracted | Transition take foul: 1 FT by any player, then sideline possession [3] |
| Flagrant / disruptive | Flagrant foul disqualifies (Case 10.6.2) [474] | FTs to the fouled player, then a frontcourt throw-in opposite the table (Art. 38.2). Disruptive foul: same structure (Art. 37.2) [54] | Flagrant 1 and flagrant 2: two FTs plus possession [59] | Not extracted | Not extracted |

**One event-engine unit test per edge case:**
- technical, flagrant and disruptive FT sequences (no rebound, possession retained);
- lane violations;
- clock-stop-after-FG in each league;
- `shot_clock=None`;
- held ball with arrow vs jump ball;
- out of bounds attributed from the next inbound (one camera cannot see the last touch; the inbounding team implies the other team touched it last, with `confidence < 1`);
- operator substitutions;
- running clock;
- foul-out at 5 vs 6;
- AAU halves.

AAU and adult rec formats default to `nfhs` plus a tournament override YAML that the operator fills in (halves, running clock, timeouts). No primary AAU source was retrieved (§7A.13).

### 7A.7 Minors' personal data in cloud payloads

**Facts.**
- **Anthropic retention.** API inputs and outputs are deleted from Anthropic's backend within 30 days of receipt or generation, unless agreed otherwise (e.g. under a ZDR agreement). Items flagged by trust-and-safety systems are kept for up to 2 years, with classifier scores kept for up to 7 years; feedback is kept for 5 years [449].
- **Anthropic Commercial Terms** (effective 17 June 2025):
  - "Anthropic may not train models on Customer Content from Services."
  - A DPA is incorporated by reference.
  - The Services are "not for consumer use" [450].
- **Anthropic ZDR.** Fable-class models require 30-day retention. The bundled Claude docs say to treat Opus 5.5 like Opus 5 on retention, and Opus 5 is available under ZDR [23]. This is not confirmed on a live Anthropic page (§7A.13). Under ZDR, server-side-fallback sticky routing is not recorded [23].
- **TypeSafe.**
  - TypeSafe offers ZDR to enterprise customers only [451].
  - Under its DPA, the customer is the controller and TypeSafe the processor/service provider.
  - Retention is "as long as necessary".
  - Subprocessors are listed on trust.typesafe.ai, which our fetcher could not read.
  - Customers have 15 days to object to a new subprocessor.
  - Breach notice is due within 72 h.
  - International transfers use EU SCC Modules 2 and 3.
  - Children's data is not addressed [452].
- **FERPA school-official exception.** An outside party qualifies only if it performs an institutional service, "is under the direct control" of the school with respect to the use and maintenance of education records, and is subject to the §99.33(a) redisclosure limits [455]. Our customer contract must flow those limits down to Anthropic and TypeSafe.
- **COPPA.**
  - "Personal information" includes a first and last name; "a photograph, video, or audio file where such file contains a child's image or voice"; and biometric identifiers, including gait patterns.
  - An "operator" includes one on whose behalf an agent or service provider collects or maintains information [456].
  - Whether CourtVision is an operator collecting information from a child online is a question for counsel.
- **GDPR.** Pseudonymised data is still personal data (Recital 26; Art. 4(5) defines pseudonymisation; not re-fetched in this pass) [375]. Pseudonymisation lowers risk but does not take payloads out of scope.

**Design (proposal).**
1. **Pseudonymise every person** in Jev states and Claude FactPackets. Players get per-game opaque tokens `P01`…`P30`, coaches `C1`/`C2`. Team names stay, since a school or club name is not a minor's personal data. Jersey numbers are sent only when a line needs one, and never together with a name.
2. **Claude writes tokens**, e.g. `"{P07} hits the corner three"`. The deterministic validator checks that every token is in the packet. Only then does the local renderer replace tokens with roster names, or with "number 23" under the identity tiers. Jev never needs names; its states use roles ("the shooter", "Home's ball handler").
3. **The name map stays on the device,** in RAM, per game.
4. **Consent registry gate.** `cloud_narration_ok` requires a signed customer DPA that lists Anthropic and TypeSafe as subprocessors, with the §99.33(a) flow-down for US schools. The output falls back to **template-only mode** (no cloud calls; local names still allowed on local output) when any of these holds:
   - a rostered player is under 13 without verifiable parental consent;
   - consent is missing;
   - the region is not cleared.
5. **Ask both vendors for ZDR** before selling to schools: Anthropic by agreement, TypeSafe through its enterprise tier.

### 7A.8 P0 features without a build path: lineups, captions, AD, Q&A

**Lineups (proposal).**
- The P0 mechanism is the operator. Starting fives are entered at setup; substitutions use tap-to-sub at each dead ball (two taps: out, in), mapped to the roster. CV suggestions come later, in Phase E.
- `PlayerLine.seconds`, `plus_minus` and `lineups` come only from operator entries. Without them, the report shows team stats only. Player box lines stay "provisional".
- **Lineup accuracy** is the share of game seconds in which each team's recorded five equals the truth; target ≥ 98% after operator entry. Also track sub-entry latency: p95 seconds from the real substitution to its entry.

**Caption transport per ingest target.**

| Target | Transport | Source |
|---|---|---|
| OBS (customer-run) | obs-websocket v5 request `SendStreamCaption` (`captionText`), which "Sends CEA-608 caption text over the stream output" (added in v5.0.0) | [469] |
| YouTube Live | HTTP POST to the event's signed Captions Ingestion URL, or EIA-608/CEA-708 embedded by the encoder. One caption track. YouTube asks users of caption software to add a 30 s or 60 s broadcast delay | [470] |
| Our HLS player and dashboard | A WebVTT subtitle rendition (`EXT-X-MEDIA TYPE=SUBTITLES`, URI required), with `X-TIMESTAMP-MAP=LOCAL:<cue time>,MPEGTS:<pts>` mapping cue times to the video's MPEG-2 timestamps | [471] |
| Any | Sidecar WebSocket JSON `{"type": "caption", …}` for our own player and the AD app | proposal |

**Timing rule.** Every caption cue and AD line carries the source video PTS of `t_occ`. It is shown at `pts(t_occ) + D`, where D is the stream's applied delay: the 0.8-1.0 s overlay delay, or a 30-60 s caption or anti-wagering delay. It is never shown before the event is confirmed. The AD "p95 ≤ 2 s" target is measured against the delayed video the listener actually gets.

**AD vocabulary (proposal).** One call per possession event, ≤ 6 words, from a fixed lexicon:
- possession ("Home ball") and progress ("across half court");
- zones: left corner, left wing, top, right wing, right corner, paint, backcourt;
- shot ("shot, left wing") and result ("good", "no good", "blocked");
- rebound ("rebound, Away") and turnover ("turnover, Home");
- the score after every possession ("Home 42, Away 40"), only from confirmed scoreboard or operator values.

Names are used only under the identity tiers of §4.1.

**Second-screen Q&A (U10, P1, proposal).**
- Claude (`claude-opus-5-5`, effort `low`) gets three read-only tools over our own event log: `get_events(filter)`, `get_box(entity)` and `get_score(t)`.
- Answers must cite event IDs. The deterministic validator checks every number against the tool outputs.
- Questions outside the log, about betting, or about minors' personal details get a fixed refusal.
- Latency target ≤ 10 s; never on the live narration path. The §7A.7 pseudonymisation applies.

### 7A.9 Jev question hygiene, breakers and determinism
- **B3 `top_card` labels** are neutral letter keys, `card_a` … `card_h`, each described as "The card stored at `cards.card_a`" (and so on). Card text lives only in the state, uses pseudonymised entities, and no label embeds a name. Letters keep the "no digit in any instruction or criterion" rule intact. Fixed in `jev_integration_design.md` §5 and §6.3.
- **One `Breaker` per bundle (B1-B6),** so a B6 image timeout or a B4 failure burst cannot silence the other bundles.
- **Determinism.**
  - TypeSafe's consistency cookbook made 15 `jev-latest` calls, all served by `jev-1.13.0`, each with a unique nonce field in the state. Raw label agreement was 90.8%, mean probability SD 0.0098 and maximum SD 0.0515. Agreement rose to 99.2% after a 0.60 confidence threshold.
  - The page itself says the design "cannot separate sensitivity to the irrelevant field from variation that would occur on identical requests" [453].
  - So the TTL cache stays **off** until a determinism test passes: 20 identical requests per bundle. Enable caching only if top-label agreement is ≥ 99% and per-label probability SD is ≤ 0.01 (**proposal**).
- **Server limits.**
  - The API returns 422 for malformed questions and 400 `max_tokens_exceeded` for context overflow. It documents no limit on the number of questions, labels or levels [454].
  - The SDK enforces 1-255 labels and 1-10 levels client-side [8].
  - The server's response to 256 labels or 11 levels is unverified. Make one live call for each and record the status.
- **Token scaling.** Log `usage.input_tokens` against question count per bundle; no doc describes the scaling.

### 7A.10 CV runtime spec: CoreML I/O, post-processing, trackers at 15 Hz

**CoreML I/O** (`rfdetr[coreml]` export) [22][465]:
- **Input.**
  - Name: read it from `mlmodel.get_spec().description.input[0].name` (currently `tensors`).
  - Shape: `(1, 3, H, W)`, with H and W fixed. Defaults are Nano 384, Small 512 and Medium 576 [458].
  - Dtype: float32 array.
  - Preprocessing: scale to [0, 1]; resize to (H, W) with `antialias=False`; normalise with mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225).
- **Load.** `ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)`. Compute units are not stored in the package and default to `ALL` at every load [22].
- **Outputs.**
  - Names are inferred by coremltools, so match by position: `dets, labels` for detection [22].
  - In export mode the model returns `(boxes, logits)`. Boxes are sigmoid-normalised cxcywh, one per query, and the query count is `num_queries` (default 300) [459][458].
  - A leading decoder-layer dimension may be present. Check the shapes in the spec at export (unverified).
- **Exports.** One package per input size: one for the full frame (e.g. Small 512) and one for the ball ROI (e.g. a 384 crop). Batch size is fixed as well [22].

**Post-processing**, ported from rfdetr's `PostProcess` [457]:
1. `prob = sigmoid(logits)`.
2. Flatten to queries × classes and take the top `num_select` (default 300) by score.
3. `query = idx // n_classes`; `label = idx % n_classes`.
4. Convert boxes from cxcywh to xyxy, multiply by (W, H, W, H) of the original frame, and clamp to the image.
5. Build `sv.Detections(xyxy=…, confidence=…, class_id=…)`, filter by per-class thresholds, and map class ids to `ObjClass`.

**Parity test.** Run PyTorch and CoreML fp16 on 100 frames of our footage. After post-processing, ≥ 95% of the PyTorch boxes above 0.5 confidence must have a CoreML match with IoU ≥ 0.95 and |Δconf| ≤ 0.05. Expect some fp16 drift (Nano mAP 48.0 → 45.1) [22].

**`trackers` at 15 Hz** [42]:
- `lost_track_buffer` counts frames at 30 FPS and is scaled by `frame_rate`. So pass `frame_rate=15.0` when the tracker is updated at 15 Hz.
- `update(detections, frame=None, timestamp=None)`: `timestamp` is the frame's absolute time in seconds; `None` means fixed-rate mode, one frame step per call. Pass the frame's PTS when frames are skipped.
- ByteTrack defaults: `track_activation_threshold=0.7`, `minimum_consecutive_frames=2`, `minimum_iou_threshold=0.1`, `high_conf_det_threshold=0.6`. Passing a frame to ByteTrack only emits a warning.
- BoT-SORT defaults: `enable_cmc=True`, `cmc_method="sparseOptFlow"`, `cmc_downscale=2`, and association thresholds 0.2 / 0.5 / 0.3.
- Both trackers take an `iou` similarity parameter (`None` means standard IoU).
- The docs give no exact formula for buffer scaling. A unit test should feed synthetic tracks at 15 Hz, with gaps and timestamps, and check that IDs persist.

**Decisions.**
- Court-space association (a cap of 5 per team, plus Basketball-SORT reacquisition [43]) lives in our own `ByteTracker` (ARCHITECTURE) for the fixed camera. `trackers.BoTSORTTracker` with CMC is used for broadcast and as the evaluation baseline.
- `sv.Position.BOTTOM_CENTER` exists: the enum runs from `CENTER` through `BOTTOM_CENTER` to `CENTER_OF_MASS` [463]. Verified.

### 7A.11 A legally shippable path to the first detector and jersey reader
- **No bootstrap source is shippable.** Every one is R&D-only (§5.4). Hudl's terms also bar using Hudl content or data "to train, fine-tune, or develop any machine learning" model without written permission (§3.2.8) [475]. So customer footage pulled from Hudl is not training data unless the customer's own Hudl agreement allows it.
- **First-party data sprint** (**proposal**; the sizes are planning targets, not findings):
  - ≥ 6 gyms and ≥ 20 games, filmed by us or under a licence that allows training, with `training_allowed=True` per game.
  - ≈ 6,000 labelled frames (≈ 1,000 per gym) for player, referee, ball, rim and number.
  - ≥ 3,000 in-flight ball boxes.
  - ≥ 5,000 legible number crops across ≥ 100 jersey designs and pinnies.
  - ≥ 1,000 shots labelled make/miss, the isotonic threshold of §7A.2.
- **Training on MPS.**
  - rfdetr's `TrainConfig` has no `device` field. It uses PyTorch Lightning's `accelerator="auto"`, and its fp16 AMP option mentions "supported CUDA/MPS backends" [458].
  - The training docs mention only `cpu` and `cuda`, and give T4 settings: `batch_size=4`, `grad_accum_steps=4`, for an effective batch of 16 [423].
  - Run a 1-hour `rfdetr[train]==1.11.2` smoke test on MPS. The fallback is a rented CUDA GPU.
- **Cost of the fallback.** Lambda on-demand prices, before tax: 1× H100 PCIe $3.29/h, H100 SXM $4.29/h, A10 $1.29/h, A6000 $1.09/h [482]. A 24 h fine-tune on one H100 PCIe costs ≈ $79 (our arithmetic). Labelling, not GPU time, dominates the budget.
- **Release gate.** CI refuses to package weights unless every training row has `rights_status ∈ {own, licensed}`. The weights file carries the manifest hash.

### 7A.12 Ingest partners' terms

| Source | What the terms say | MVP verdict |
|---|---|---|
| Hudl (Terms updated 11 Aug 2026) [475] | §3.2.8 bans "any automated means, including bots, scrapers, crawlers, or artificial intelligence tools" to collect or extract data, and bans training ML on Hudl content without permission. §3.2.3 prohibits "the unauthorized distribution of livestreams or broadcasts". §5.4: the game film stays the customer's, but "you may not export it or any full game recording" | **Deny** automated pulls and re-streams. Only through a written Hudl partnership or API agreement |
| NFHS Network (Terms last updated Nov 2024) [476] | "personal, non-commercial use only". No access "through any automated means, like scripts or web crawlers". No data mining. No reproducing, distributing, modifying or creating derivative works | **Deny.** Needs a PlayOn / NFHS Network agreement. Postseason exclusivity in KS, GA and NY also applies [138][139][140] |
| Veo (General Terms, 29 May 2026) [477] | Content belongs to the customer. Users may not "access, reproduce, download, distribute, transmit, broadcast" platform material except as permitted (User Terms §7.4). Veo licenses customer content for AI training (§6.3). Publishing through partner apps or the API is contemplated (§1.1, §7.1) | **Conditional.** Post-game files the customer exports, where Veo permits; live ingest only under Veo's API or partner terms |
| Pixellot [478] | No public terms of use found; the site footer links only privacy documents | **Unknown.** Partner agreement required |
| BallerCam | Not checked in this pass | **Unknown** |
| Customer-controlled push (OBS, iPhone, school camera over RTMP/SRT) | The customer owns or controls the stream | **Default MVP source** |

### 7A.13 Known gaps

These were not closed in this pass. The web-search budget ran out partway through, so the rest of the work used direct fetches of known primary URLs. Some items below may yield to a later search.

1. **Measured latency.** No Claude or Jev call was made, because no live keys were used in this pass. Opus 5.5, Haiku 5.5 and Jev B1-B5 p50/p95 from a venue network remain unmeasured, so every deadline is an initial value (§7A.3). Whether AA's 6.52 s TTFT includes thinking is not stated [24].
2. **Jev behaviour.** Determinism on identical requests, the server status for 256 labels or 11 levels, and how input tokens scale with question count are all unverified (§7A.9).
3. **M3 Pro throughput.** RF-DETR S/M, the ROI pass, SigLIP, OCR, RTMPose, ANE+GPU concurrency and thermal behaviour are unmeasured. `rfdetr` and `coremltools` are not installed in `.venv`, and no weights were downloaded (§7A.4).
4. **CoreML output shapes** for rfdetr exports (a possible leading decoder-layer dimension) are unverified until a real export. The exact `lost_track_buffer` scaling formula and the `timestamp` semantics in `trackers` are not documented (§7A.10).
5. **Field measurements with no source:** dead-ball window distributions and scoreboard-operator lag in amateur games. Both need 3-5 own games (§7A.3, §7A.5).
6. **Rules.** AAU and adult-rec rule sets (no primary source retrieved); the NBA held-ball jump rule; NCAA technical-foul sequences; and a purchased NFHS rules book to replace the third-party reproductions [241][472]-[474] (§7A.6).
7. **Vendors.** Anthropic's subprocessor list, and explicit confirmation that `claude-opus-5-5` is available under ZDR. TypeSafe's subprocessor list (its trust portal did not render for our fetcher) and its retention period ("as long as necessary") (§7A.7).
8. **Legal, for counsel:** COPPA operator status, the FERPA flow-down wording, an on-device whistle-band detector, and the ingest partner agreements (§7A.7, §7A.12).
9. **Unchecked terms:** Pixellot (none public) and BallerCam (§7A.12).
10. **Training on MPS.** `rfdetr` training on MPS is untested; no doc confirms it (§7A.11).
11. **Unverified APIs and priors.** Apple Vision's text-recognition API (Apple's docs pages did not render for our fetcher). NCAA and HS shooting priors (still no primary source).
12. **Untested proposals.** The data-sprint sizes, the degradation-ladder thresholds and the hand-set cue-model weights are untested.
13. **Lock file.** The pins are in `pyproject.toml`, but no lock file was generated in this pass, because resolving one needs package downloads. Run `uv lock` or `pip-compile` before Phase B.
14. **Code contract.** `courtvision/types.py` already uses feet (D1) and needed no change. Its comments still say nothing about `RuleSpec`, calibration status or pseudonymised tokens. The module agents add those fields when they implement §7A.2, §7A.6 and §7A.7.

---

## 8. Sources

All URLs from the eight notes, merged and deduplicated. arXiv abs/html/pdf/version variants are folded together, as are TypeSafe `.md` and plain pages. Each entry lists the notes that cite it, and titles come from the notes. Any "(unverified)" flag in a note still applies.

### 8.1 Local research notes and measurements

1. docs/research/jev_api.md — TypeSafe Jev: exact API reference for the CourtVision Python integration (local note; verified offline against typesafe-sdk 0.7.5)
2. docs/research/cv_pipeline.md — CV pipeline: real-time player, ball and rim tracking (local note)
3. docs/research/basketball_rules_stats.md — Basketball geometry, official stat definitions and hard-coded priors (local note)
4. docs/research/narration.md — Real-time factual commentary: research and latency engineering (local note)
5. docs/research/use_cases_market.md — Use cases, users, existing products, and legal/data constraints (local note)
6. docs/research/training_data.md — Training off live games: datasets, weak supervision, labelling, evaluation (local note)
7. docs/research/jev_integration_design.md — TypeSafe Jev inside CourtVision: where to use it, where not to, and exactly how (local note)
8. docs/research/sdk_ground_truth.md — SDK ground truth verified against installed typesafe-sdk 0.7.5 and anthropic 1.15.0 (local note)
9. Local measurements on the Apple M3 Pro (macOS 26.5.1, Python 3.12.3): AVSpeechSynthesizer via pyobjc-framework-AVFoundation 12.2.2, `say`, and offline mock-transport SDK tests, recorded as [M] in docs/research/narration.md. Gap pass, 2026-10-20: `sysctl` / `system_profiler` read this machine as an M3 Pro with 5 performance + 6 efficiency cores, a 14-core GPU and 18 GB (19,327,352,832 bytes), and `pmset -g therm` reported no thermal or performance warning level at idle

### 8.2 Web sources cited in this report (numbered in order of first citation)

10. TypeSafe docs: `/introduction`. <https://docs.typesafe.ai/introduction.md> *(jev_api)*
11. TypeSafe docs, System One concept (calibration over groups; no text generation). <https://docs.typesafe.ai/concepts/system-one.md> *(jev_api, jev_integration_design)*
12. ESPN, "NBA to use Hawk-Eye tracking system" (2023-03-09). <https://www.espn.com/nba/story/_/id/35818363/nba-use-hawk-eye-tracking-system-follow-players-ball> *(use_cases_market)*
13. iGaming Business, "NBA to receive equity in Sportradar with exclusive data deal". <https://igamingbusiness.com/nba-to-receive-equity-in-sportradar-with-exclusive-data-deal/> *(use_cases_market)*
14. NCAA, "NCAA Championships Digital Highlights Policy for Member Institutions" (updated 2026-09-01). <https://www.ncaa.org/championships/media-center/ncaa-championships-digital-highlights-policy-for-member-institutions/> *(use_cases_market)*
15. NBA.com, "Terms of Use". <https://www.nba.com/terms> *(use_cases_market)*
16. Roboflow, "How to Build an AI Basketball Shot Evaluator" (14 Aug 2026). <https://roboflow.com/blog/how-to-build-an-ai-basketball-shot-evaluator> *(cv_pipeline, jev_integration_design)*
17. Hudl, "Hudl Assist basketball pricing". <https://www.hudl.com/pricing/assist/basketball> *(use_cases_market)*
18. SportsVisio, "Pricing". <https://sportsvisio.com/pricing> *(use_cases_market)*
19. TypeSafe docs, "Models" (Jev 1.14 `jev-1.14.0` / Jev 1.13 `jev-1.13.0`; 64k per-request / 32k per-question context; `TypeSafe-Danger-Zone-Long-Context` header; aliases `jev-latest` and `jev-preview`). <https://docs.typesafe.ai/models.md> *(jev_api, use_cases_market, training_data, jev_integration_design)*
20. NBA_Streaming abstract (arXiv 2608.09200). <https://arxiv.org/abs/2608.09200> *(narration, training_data)*
21. nvan21/Basketball-Shot-Detection. <https://github.com/nvan21/Basketball-Shot-Detection> *(cv_pipeline)*
22. RF-DETR docs, "Native CoreML Export (.mlpackage)". <https://rfdetr.roboflow.com/latest/exports/coreml/> *(cv_pipeline)*
23. Anthropic Claude API reference (as bundled in the Claude Code claude-api skill: streaming, prompt caching, effort, Opus 5.5 migration notes); public docs root. <https://platform.claude.com/docs> *(narration)*
24. Artificial Analysis: Claude Opus 5.5 (low) — TTFT 6.52 s, 79.5 tok/s (Anthropic API, default 10k-input workload). <https://artificialanalysis.ai/models/claude-opus-5-5-low> *(narration, jev_integration_design)*
25. TypeSafe docs, How to build with System One ("about 100 ms"). <https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md> *(jev_api, jev_integration_design)*
26. TypeSafe docs, Twitch chat filter cookbook (median about 0.4 s, `jev-1.13.0`). <https://docs.typesafe.ai/cookbooks/twitch_chat_filter.md> *(jev_api, jev_integration_design)*
27. GitHub, DeepSportradar/instance-segmentation-challenge README (data CC BY-NC-ND 4.0; code Apache-2.0). <https://github.com/DeepSportradar/instance-segmentation-challenge> *(use_cases_market, training_data)*
28. SportsMOT repository. <https://github.com/MCG-NJU/SportsMOT> *(training_data)*
29. PL-NBA. <https://arxiv.org/abs/2608.19646> *(training_data)*
30. NBA_Streaming repository. <https://github.com/wyy-081/NBA_Streaming> *(training_data)*
31. GNU AGPL-3.0 text. <https://www.gnu.org/licenses/agpl-3.0.en.html> *(cv_pipeline)*
32. Ultralytics licensing page (fetched during verification). <https://www.ultralytics.com/license> *(cv_pipeline)*
33. Youth Sports Business Report, "Pixellot raises $35M". <https://youthsportsbusinessreport.com/pixellot-raises-35m-to-scale-ai-powered-game-capture-across-youth-and-high-school-sports/> *(use_cases_market)*
34. Techleap, "Pixellot raises $35M for growth plans" (2025-11). <https://finder.techleap.nl/news/feed/pixellot-raises-35m-for-growth-plans> *(use_cases_market)*
35. WSC Sports: NBA launches GenAI multilingual voice-over (Apr 2025). <https://wsc-sports.com/blog/customer-spotlight/nba-launches-genai-technology-with-wsc-sports-to-automate-multilingual-content/> *(narration, use_cases_market)*
36. roboflow/rf-detr README. <https://github.com/roboflow/rf-detr> *(cv_pipeline, training_data)*
37. rfdetr on PyPI. <https://pypi.org/project/rfdetr/> *(cv_pipeline)*
38. Roboflow, "How to Detect, Track, and Identify Basketball Players with Computer Vision" (P. Skalski, 30 Sep 2025). <https://roboflow.com/blog/identify-basketball-players> *(cv_pipeline, training_data)*
39. Roboflow forum, "Can RF-DETR run with mac silicon?". <https://discuss.roboflow.com/t/can-rf-detr-run-with-mac-silicon/10580> *(cv_pipeline)*
40. supervision changelog. <https://supervision.roboflow.com/latest/changelog/> *(cv_pipeline)*
41. Ultralytics YOLO26 docs. <https://docs.ultralytics.com/models/yolo26/> *(cv_pipeline, training_data)*
42. trackers API reference. <https://trackers.roboflow.com/latest/api/trackers/> *(cv_pipeline)*
43. Basketball-SORT, arXiv 2406.19655. <https://arxiv.org/html/2406.19655v1> *(cv_pipeline)*
44. roboflow/sports `sports/common/team.py` (`TeamClassifier`, SigLIP id). <https://raw.githubusercontent.com/roboflow/sports/main/sports/common/team.py> *(cv_pipeline)*
45. facebookresearch/sam3. <https://github.com/facebookresearch/sam3> *(cv_pipeline, training_data)*
46. trackers comparison. <https://trackers.roboflow.com/latest/trackers/comparison/> *(cv_pipeline)*
47. SportsMOT, arXiv 2304.05170. <https://arxiv.org/abs/2304.05170> *(cv_pipeline, training_data)*
48. TrackID3x3, arXiv 2503.18282. <https://arxiv.org/html/2503.18282v1> *(cv_pipeline)*
49. Koshkina & Elder, "A General Framework for Jersey Number Recognition in Sports Video", arXiv 2405.13896. <https://arxiv.org/html/2405.13896v1> *(cv_pipeline)*
50. jersey-number-pipeline code (linked from the Koshkina & Elder paper; README states CVPR 2024 Workshops and a CC BY-NC 3.0 licence). <https://github.com/mkoshkina/jersey-number-pipeline> *(cv_pipeline)*
51. Roboflow, "How to Fine-Tune RF-DETR Keypoints on Custom Data" (26 Jun 2026). <https://roboflow.com/blog/train-rf-detr-keypoint> *(cv_pipeline, training_data)*
52. roboflow/sports `sports/common/view.py` (`ViewTransformer`). <https://raw.githubusercontent.com/roboflow/sports/main/sports/common/view.py> *(cv_pipeline)*
53. NBA, 2026-27 Official NBA Playing Rules (PDF; p. 8 court-diagram labels). <https://ak-static.cms.nba.com/wp-content/uploads/sites/4/2026/10/2026-27-Official-NBA-Playing-Rules.pdf> *(basketball_rules_stats)*
54. FIBA, *Official Basketball Rules 2026 v1.1* (valid 1 Oct 2026). <https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2026-v1-1.pdf> *(basketball_rules_stats)*
55. NBA, *Rule No. 5: Scoring and Timing*. <https://official.nba.com/rule-no-5-scoring-and-timing/> *(basketball_rules_stats)*
56. KaliCalib, arXiv 2209.07795. <https://ar5iv.labs.arxiv.org/html/2209.07795> *(cv_pipeline)*
57. Roboflow, "Detect NBA 3 Second Violations with AI" (22 Jul 2025). <https://roboflow.com/blog/detect-3-second-violation-ai-basketball> *(cv_pipeline)*
58. NBA, *Rule No. 7: Shot Clock*. <https://official.nba.com/rule-no-7-24-second-clock/> *(basketball_rules_stats)*
59. NCAA, *2024-25 Men's Basketball Rules* (Aug 2024; copy hosted by NBC Sports; the current NCAA-hosted PDF exceeded the 10 MB fetch limit). <https://nbcsports.brightspotcdn.com/15/c2/e15320424538aa0c375182350c65/ncaa-2024-25-mens-rules.pdf> *(basketball_rules_stats)*
60. NCAA, *2025-26 and 2026-27 Women's Rules Book* (updated Sept. 1, 2026). <https://ncaaorg.s3.amazonaws.com/championships/sports/basketball/rules/women/PRWBB_RulesBook.pdf> *(basketball_rules_stats)*
61. NFHS, *2025-26 Shot Clock Guidelines* (hosted by KHSAA, updated 09/2025). <https://khsaa.org/common_documents/Rules-Basketball-Rules%20and%20NFHS%20Guidance/2025-26/NFHS%202025-26-NFHS%20Shot%20Clock%20Guidelines.pdf> *(basketball_rules_stats)*
62. BasketEvent (arXiv 2607.21267). <https://arxiv.org/html/2607.21267v1> *(training_data)*
63. CourtMotion (arXiv 2512.01478). <https://arxiv.org/html/2512.01478v2> *(training_data)*
64. FIBA, *Statisticians' Manual 2024 v1.0*. <https://assets.fiba.basketball/image/upload/documents-corporate-fiba-statisticians-manual-2024.pdf> *(basketball_rules_stats, jev_integration_design)*
65. NBA.com, *Stats Glossary*. <https://www.nba.com/stats/help/glossary> *(basketball_rules_stats)*
66. NBA, *Rule No. 3: Players, Substitutes and Coaches*. <https://official.nba.com/rule-no-3-players-substitutes-and-coaches/> *(basketball_rules_stats)*
67. pbpstats API, *League team totals* (2024-25 and 2025-26 Regular Season). <https://api.pbpstats.com/get-totals/nba?Season=2024-25&SeasonType=Regular%20Season&Type=Team> *(basketball_rules_stats)*
68. Poropudas & Halme, "Decomposing Basketball Team Efficiency Using the Four Factors and Shapley Values", arXiv 2305.13032 (v2, Oct 2026). <https://arxiv.org/abs/2305.13032> *(basketball_rules_stats, narration)*
69. Basketball-Reference, *NBA League Averages – Per Game*. <https://www.basketball-reference.com/leagues/NBA_stats_per_game.html> *(basketball_rules_stats)*
70. nba_api live PlayByPlay docs. <https://github.com/swar/nba_api/blob/master/docs/nba_api/live/endpoints/playbyplay.md> *(training_data)*
71. Petri et al., "An LLM-Based Automatic Sportscast Solution for Robot Soccer Matches" (MARIO), arXiv 2607.14809. <https://arxiv.org/html/2607.14809v1> *(narration, jev_integration_design)*
72. Mori et al., "Live Football Commentary System Providing Background Information", ACL 2025 Demos. <https://aclanthology.org/2025.acl-demo.38.pdf> *(narration, jev_integration_design)*
73. IBM case study: how the Masters uses watsonx to manage its AI lifecycle. <https://www.ibm.com/case-studies/blog/how-the-masters-uses-watsonx-to-manage-its-ai-lifecycle> *(narration)*
74. Kawamatsu et al., arXiv abstract 2606.13322. <https://arxiv.org/abs/2606.13322> *(narration, jev_integration_design)*
75. Artificial Analysis: Claude Haiku 5.5 (medium: 178.6 tok/s, 13.84 s; low: 171.2 tok/s, 8.39 s). <https://artificialanalysis.ai/models/claude-haiku-5-5-medium/providers> *(narration)*
76. StreamSoccer (arXiv 2608.19723). <https://arxiv.org/html/2608.19723v1> *(narration, jev_integration_design)*
77. AI Incident Database #785: ESPN AI recap omits Alex Morgan. <https://incidentdatabase.ai/cite/785/> *(narration)*
78. Kokoro-82M model card. <https://huggingface.co/hexgrad/Kokoro-82M> *(narration)*
79. espeak-ng GitHub (GPL-3.0-or-later). <https://github.com/espeak-ng/espeak-ng> *(narration)*
80. ElevenLabs models documentation. <https://elevenlabs.io/docs/models> *(narration)*
81. ElevenLabs WebSocket stream-input API reference. <https://elevenlabs.io/docs/api-reference/text-to-speech/v-1-text-to-speech-voice-id-stream-input> *(narration)*
82. Manatt, "Tennessee's ELVIS Act expands publicity rights for individuals' voices" (2024). <https://www.manatt.com/insights/newsletters/client-alert/tennessees-elvis-act-expands-publicity-rights-for> *(use_cases_market)*
83. Maymin, Maymin & Shen, "How Much Trouble is Early Foul Trouble?" (Sloan / IJSF 2012). <https://sloansportsconference.com/research-papers/how-much-trouble-is-early-foul-trouble> *(narration)*
84. "Stop the Clock: Are Timeout Effects Real?" (ECML-PKDD 2020). <https://arxiv.org/abs/2009.06750v1> *(narration)*
85. Gibbs et al., "The causal effect of a timeout at stopping an opposing run in the NBA". <https://arxiv.org/pdf/2011.11691> *(narration)*
86. Bocconi, "Do timeouts matter? A study of EuroLeague basketball" (2026). <https://iris.unibocconi.it/retrieve/befbdb69-1af2-4097-8e89-26fa894b51f4/Do_timeouts_matter_A_study_of_euroleague_Basketbal.pdf> *(narration)*
87. L-RAPM: Lineup Regularized Adjusted Plus-Minus (lineup sample sizes). <https://arxiv.org/pdf/2601.15000> *(narration)*
88. OHSAA (NFHS release), "2013-14 Basketball Rules Changes". <https://ohsaa.org/sports/bk/2013-14BasketballRulesPressRelease.pdf> *(use_cases_market)*
89. WTOP (AP), "March Madness meets the iPad era: coaches' challenges" (2026-03-25). <https://wtop.com/ncaa-basketball/2026/03/march-madness-meets-the-ipad-era-how-coaches-challenges-are-changing-the-game/> *(use_cases_market)*
90. NCAA, "Ten basketball conferences allowed to experiment with electronic devices" (2019-11-25). <https://www.ncaa.org/news/2019/11/25/ten-basketball-conferences-allowed-to-experiment-with-electronic-devices-during-league-games.aspx> *(use_cases_market)*
91. W3C, Web Content Accessibility Guidelines (WCAG) 2.1 (SC 1.2.4, 1.2.5). <https://www.w3.org/TR/WCAG21/> *(use_cases_market)*
92. JD Supra, "DOJ delays ADA web accessibility compliance deadlines" (2026-04). <https://www.jdsupra.com/legalnews/doj-delays-ada-web-accessibility-7295347/> *(use_cases_market)*
93. TypeSafe docs, Primitives (independent parallel questions; IDs not sent; backticked paths). <https://docs.typesafe.ai/primitives.md> *(jev_api, jev_integration_design)*
94. typesafe-sdk on PyPI (0.7.5, MIT). <https://pypi.org/project/typesafe-sdk/> *(jev_api)*
95. typesafe-ai/typesafe-sdk-python (SDK source, tag v0.7.5). <https://github.com/typesafe-ai/typesafe-sdk-python> *(jev_api, jev_integration_design)*
96. TypeSafe docs: `/sdk/python/api/clients/sync`. <https://docs.typesafe.ai/sdk/python/api/clients/sync.md> *(jev_api)*
97. TypeSafe docs: `/sdk/python/api/clients/async`. <https://docs.typesafe.ai/sdk/python/api/clients/async.md> *(jev_api)*
98. TypeSafe docs: `/sdk/python/api/types/questions`. <https://docs.typesafe.ai/sdk/python/api/types/questions.md> *(jev_api)*
99. TypeSafe API OpenAPI spec (info.version 0.2.0). <https://api.typesafe.ai/openapi.json> *(jev_api, narration)*
100. TypeSafe docs: `/sdk/python/api/types/responses`. <https://docs.typesafe.ai/sdk/python/api/types/responses.md> *(jev_api)*
101. TypeSafe docs: Confidence. <https://docs.typesafe.ai/confidence.md> *(jev_api, training_data, jev_integration_design)*
102. TypeSafe docs, Box. <https://docs.typesafe.ai/primitives/box.md> *(jev_api, jev_integration_design)*
103. TypeSafe docs, Python SDK retries. <https://docs.typesafe.ai/sdk/python/api/retries.md> *(jev_api, jev_integration_design)*
104. TypeSafe docs: `/cookbooks/parallel_questions`. <https://docs.typesafe.ai/cookbooks/parallel_questions.md> *(jev_api)*
105. TypeSafe docs, Jev 1.13 jaggedness. <https://docs.typesafe.ai/model-jaggedness/jev-1.13.md> *(jev_api, jev_integration_design)*
106. TypeSafe docs, "Images" (`TypeSafe-Danger-Zone-Images: true` header; images and Box questions labeled "Dreamer-ready", not production). <https://docs.typesafe.ai/images.md> *(jev_api, use_cases_market, jev_integration_design)*
107. NBA Official: replay triggers. <https://official.nba.com/replay/triggers/> *(narration)*
108. Label Studio repository. <https://github.com/HumanSignal/label-studio> *(training_data)*
109. CVAT repository. <https://github.com/cvat-ai/cvat> *(training_data)*
110. CVAT digest, December 2025. <https://www.cvat.ai/resources/changelog/digest-dec-2025> *(training_data)*
111. SoccerNet 2023 Challenges Results, arXiv 2309.06006 (search snippet). <https://arxiv.org/pdf/2309.06006> *(cv_pipeline, training_data)*
112. Luiten et al., HOTA (arXiv 2009.07736, IJCV 2021). <https://arxiv.org/abs/2009.07736> *(training_data)*
113. Guo et al., "On Calibration of Modern Neural Networks", ICML 2017 (arXiv 1706.04599). <https://arxiv.org/abs/1706.04599> *(training_data, jev_integration_design)*
114. Min et al., FActScore, EMNLP 2023 (arXiv 2305.14251; ACL Anthology 2023.emnlp-main.741). <https://arxiv.org/abs/2305.14251> *(training_data, jev_integration_design)*
115. NBA SportVU dataset page (631 games, 2015-16). <https://astro.paperswithcode.com/dataset/nba-sportvu> *(training_data)*
116. linouk23/NBA-Player-Movements. <https://github.com/linouk23/NBA-Player-Movements> *(training_data)*
117. HF dataset `dcayton/nba_tracking_data_15_16`. <https://huggingface.co/datasets/dcayton/nba_tracking_data_15_16> *(training_data)*
118. TeamTrack project page (arXiv 2404.13868). <https://atomscott.github.io/TeamTrack/> *(training_data)*
119. PaddlePaddle PP-OCRv5_server_rec model card (Apache-2.0). <https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec> *(training_data)*
120. SAHI repository (MIT; docs/predict.md). <https://github.com/obss/sahi> *(training_data)*
121. facebookresearch/sam2. <https://github.com/facebookresearch/sam2> *(cv_pipeline, training_data)*
122. Hugging Face Transformers docs: SAM3 (`Sam3Model`, `Sam3Processor`; v5.20.2). <https://huggingface.co/docs/transformers/model_doc/sam3> *(training_data)*
123. Hugging Face Transformers docs: SAM3 Video (`Sam3VideoModel`, `Sam3VideoProcessor`). <https://huggingface.co/docs/transformers/model_doc/sam3_video> *(training_data)*
124. TrackNet, arXiv 1907.03698 (abstract fetched). <https://arxiv.org/abs/1907.03698> *(cv_pipeline, training_data)*
125. Xie et al., Noisy Student (arXiv 1911.04252). <https://arxiv.org/abs/1911.04252> *(training_data)*
126. Cervone, D'Amour, Bornn, Goldsberry, "A Multiresolution Stochastic Process Model for Predicting Basketball Possession Outcomes", JASA 111(514), 2016 (arXiv 1408.0777). <https://arxiv.org/abs/1408.0777> *(training_data, jev_integration_design)*
127. "Towards Comprehensive Basketball Understanding", arXiv 2608.23435 (Aug 2026). <https://arxiv.org/html/2608.23435v1> *(cv_pipeline, training_data)*
128. Barrera-Roy & Clapés, "See It Before You Grab It". <https://arxiv.org/abs/2512.15386> *(training_data)*
129. nba_api PlayByPlayV3 docs. <https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/playbyplayv3.md> *(training_data)*
130. NBA.com Terms of Use (last updated 2026-07-13). <https://www.nba.com/termsofuse> *(use_cases_market, training_data)*
131. GitHub, swar/nba_api (MIT). <https://github.com/swar/nba_api> *(use_cases_market)*
132. The Register, "YouTube video subtitles used for AI training" (2024-07-17). <https://www.theregister.com/2024/07/17/youtube_video_subtitles_ai/> *(use_cases_market)*
133. Debevoise & Plimpton, "Anthropic and Meta decisions on fair use" (2025-06). <https://www.debevoise.com/insights/publications/2025/06/anthropic-and-meta-decisions-on-fair-use> *(use_cases_market)*
134. Manatt, "Making sense of the first two Gen AI fair use rulings" (2025). <https://www.manatt.com/insights/newsletters/client-alert/making-sense-of-the-first-two-often-contradictory-sometimes-confounding-gen-ai-fair-use-rulings> *(use_cases_market)*
135. Wiley, "Copyright Office issues key guidance on fair use in generative AI training" (2025-05). <https://www.wiley.law/alert-Copyright-Office-Issues-Key-Guidance-on-Fair-Use-in-Generative-AI-Training> *(use_cases_market)*
136. BARD repository. <https://github.com/GabrieleGiudic/BARD> *(training_data)*
137. T-DEED repository (GPL-3.0; CVsports @ CVPR 2024; SNBAS 2024 1st place). <https://github.com/arturxe2/T-DEED> *(training_data)*
138. Kansas Legislature testimony (KSHSAA / NFHS Network exclusive postseason rights, 2023). <https://kslegislature.gov/li/b2023_24/committees/ctte_s_ed_1/documents/testimony/20230123_11.pdf> *(use_cases_market)*
139. GHSA, "Reminder of GHSA live streaming rules". <https://www.ghsa.net/node/14814> *(use_cases_market)*
140. NFHS, "NFHS Network to stream New York State Association championships". <https://nfhs.org/stories/nfhs-network-to-stream-new-york-state-association-state-championships> *(use_cases_market)*
141. Illinois General Assembly, 740 ILCS 14/10 (BIPA definitions). <https://lig.ilga.gov/legislation/ilcs/documents/074000140K10.htm> *(use_cases_market)*
142. Hunton, "FTC publishes final COPPA rule amendments" (2025). <https://www.hunton.com/privacy-and-cybersecurity-law-blog/ftc-publishes-final-coppa-rule-amendments> *(use_cases_market)*
143. Wikipedia, "Pixellot" (Basketball Australia section, 2026). <https://en.wikipedia.org/wiki/Pixellot> *(use_cases_market)*
144. Youth Sports Business Report, "Basketball Australia and Pixellot launch HoopsTV". <https://youthsportsbusinessreport.com/basketball-australia-and-pixellot-launch-hoopstv-a-national-youth-basketball-video-platform/> *(use_cases_market)*
145. Kilpatrick Townsend, "Wiretap laws in the United States" (2024-07). <https://ktslaw.com/blog/globalprivacy-and-cybersecuritylaw/2024/7/wiretap-laws-in-the-united-states> *(use_cases_market)*
146. iGaming Business, "Genius Sports NCAA expanded partnership to 2032". <https://igamingbusiness.com/sports-betting/genius-sports-ncaa-expanded-partnership-2032/> *(use_cases_market)*
147. Code of Virginia. <https://law.lis.virginia.gov/vacode/58.1-4128> *(use_cases_market)*
148. Startup Daily, "AI amateur sport analysis startup raises $3.5 million pre-seed" (2026-07). <https://www.startupdaily.net/topic/funding/ai-amateur-sport-analysis-startup-raises-3-5-million-pre-seed/> *(use_cases_market)*
149. Ultralytics CoreML integration docs. <https://docs.ultralytics.com/integrations/coreml/> *(cv_pipeline)*
150. Ultralytics track mode docs. <https://docs.ultralytics.com/modes/track/> *(cv_pipeline)*

### 8.3 Other web sources consulted by the notes (not cited inline above)

151. TypeSafe docs: `/concepts/state`. <https://docs.typesafe.ai/concepts/state.md> *(jev_api)*
152. TypeSafe docs, Choice. <https://docs.typesafe.ai/primitives/choice.md> *(jev_api, jev_integration_design)*
153. TypeSafe docs, Score. <https://docs.typesafe.ai/primitives/score.md> *(jev_api, jev_integration_design)*
154. TypeSafe docs, Noul. <https://docs.typesafe.ai/primitives/noul.md> *(jev_api, jev_integration_design)*
155. TypeSafe docs: `/primitives/advanced`. <https://docs.typesafe.ai/primitives/advanced.md> *(jev_api)*
156. TypeSafe, "Speculative fan-out". <https://docs.typesafe.ai/patterns/fan-out.md> *(jev_api, jev_integration_design)*
157. TypeSafe docs, Confidence-gated routing. <https://docs.typesafe.ai/patterns/confidence-routing.md> *(jev_api, jev_integration_design)*
158. TypeSafe docs, Composite scoring (normalise `score/(n−1)`). <https://docs.typesafe.ai/patterns/composite-scoring.md> *(jev_api, jev_integration_design)*
159. TypeSafe docs: `/patterns/intent-routing`. <https://docs.typesafe.ai/patterns/intent-routing.md> *(jev_api)*
160. TypeSafe docs: `/api/system-one`. <https://docs.typesafe.ai/api/system-one.md> *(jev_api)*
161. TypeSafe docs: `/api/list-models`. <https://docs.typesafe.ai/api/list-models.md> *(jev_api)*
162. TypeSafe docs: `/api/errors`. <https://docs.typesafe.ai/api/errors.md> *(jev_api)*
163. TypeSafe docs: `/sdk/python`. <https://docs.typesafe.ai/sdk/python.md> *(jev_api)*
164. TypeSafe docs: `/sdk/python/usage/basic`. <https://docs.typesafe.ai/sdk/python/usage/basic.md> *(jev_api)*
165. TypeSafe docs, Python SDK advanced usage (logging, client reuse). <https://docs.typesafe.ai/sdk/python/usage/advanced.md> *(jev_api, jev_integration_design)*
166. TypeSafe docs: `/sdk/python/changelog`. <https://docs.typesafe.ai/sdk/python/changelog.md> *(jev_api)*
167. TypeSafe docs: `/sdk/python/api/types/assets`. <https://docs.typesafe.ai/sdk/python/api/types/assets.md> *(jev_api)*
168. TypeSafe docs: `/sdk/python/api/types/common`. <https://docs.typesafe.ai/sdk/python/api/types/common.md> *(jev_api)*
169. TypeSafe docs: `/sdk/python/api/exceptions`. <https://docs.typesafe.ai/sdk/python/api/exceptions.md> *(jev_api)*
170. TypeSafe docs: `/sdk/python/api/constants`. <https://docs.typesafe.ai/sdk/python/api/constants.md> *(jev_api)*
171. typesafe-sdk PyPI JSON metadata. <https://pypi.org/pypi/typesafe-sdk/json> *(jev_api)*
172. TypeSafe docs: `/introduction/quickstart`. <https://docs.typesafe.ai/introduction/quickstart.md> *(jev_api)*
173. TypeSafe docs index (no grounding or validation pattern page). <https://docs.typesafe.ai/llms.txt> *(jev_api, use_cases_market, jev_integration_design)*
174. <https://github.com/typesafe-ai/typesafe-sdk-python/blob/v0.7.5/src/typesafe_sdk/_core/questions.py> *(jev_api)*
175. <https://github.com/typesafe-ai/typesafe-sdk-python/blob/v0.7.5/src/typesafe_sdk/_core/schemas/base.py> *(jev_api)*
176. <https://console.typesafe.ai/rate-limits> *(jev_api)*
177. <https://api.typesafe.ai> *(jev_api, sdk_ground_truth)*
178. roboflow/sports README (datasets, challenges, MIT). <https://github.com/roboflow/sports> *(cv_pipeline, training_data)*
179. <https://raw.githubusercontent.com/roboflow/sports/main/README.md> *(cv_pipeline)*
180. roboflow/sports `examples/soccer/main.py`. <https://raw.githubusercontent.com/roboflow/sports/main/examples/soccer/main.py> *(cv_pipeline)*
181. roboflow/sports `sports/common/ball.py` (`BallTracker`). <https://raw.githubusercontent.com/roboflow/sports/main/sports/common/ball.py> *(cv_pipeline)*
182. Pixelsham mirror of the Roboflow basketball post (dataset versions; secondary). <https://www.pixelsham.com/?p=25925> *(cv_pipeline)*
183. Ultralytics YOLO11 docs. <https://docs.ultralytics.com/models/yolo11/> *(cv_pipeline)*
184. ultralytics on PyPI. <https://pypi.org/project/ultralytics/> *(cv_pipeline)*
185. RF-DETR paper, arXiv 2511.09554 (ICLR 2026; cited from the README). <https://arxiv.org/abs/2511.09554> *(cv_pipeline)*
186. <https://rfdetr.roboflow.com/latest/cookbooks/export-coreml/> *(cv_pipeline)*
187. <https://github.com/roboflow/rf-detr/blob/develop/docs/exports/coreml.md> *(cv_pipeline)*
188. D-FINE repo. <https://github.com/Peterande/D-FINE> *(cv_pipeline)*
189. DEIMv2, "Real-Time Object Detection Meets DINOv3" (abstract fetched). <https://arxiv.org/abs/2509.20787> *(cv_pipeline)*
190. supervision 0.29.0 trackers docs (`ByteTrack` deprecation). <https://supervision.roboflow.com/0.29.0/trackers/> *(cv_pipeline)*
191. supervision on PyPI (release history). <https://pypi.org/project/supervision/> *(cv_pipeline)*
192. supervision `Detections` docs. <https://supervision.roboflow.com/latest/detection/core/> *(cv_pipeline)*
193. roboflow/trackers repo and PyPI. <https://github.com/roboflow/trackers> *(cv_pipeline)*
194. <https://pypi.org/pypi/trackers/json> *(cv_pipeline)*
195. ByteTrack, arXiv 2110.06864. <https://arxiv.org/abs/2110.06864> *(cv_pipeline)*
196. BoT-SORT, arXiv 2206.14651. <https://arxiv.org/abs/2206.14651> *(cv_pipeline)*
197. Deep OC-SORT, arXiv 2302.11813. <https://arxiv.org/abs/2302.11813> *(cv_pipeline)*
198. SAMIDARE, arXiv 2604.22162. <https://arxiv.org/abs/2604.22162> *(cv_pipeline)*
199. Sports-vmTracking (3x3 basketball), arXiv 2412.06258. <https://arxiv.org/abs/2412.06258> *(cv_pipeline)*
200. Meta AI blog, SAM 3 / SAM 3.1. <https://ai.meta.com/blog/segment-anything-model-3> *(cv_pipeline)*
201. Hugging Face, SmolVLM2 blog. <https://huggingface.co/blog/smolvlm2> *(cv_pipeline)*
202. Roboflow, "Comparing Base and Fine-Tuned SmolVLM2 for OCR" (14 Jul 2025). <https://roboflow.com/blog/base-vs-fine-tuned-smolvlm2-ocr> *(cv_pipeline)*
203. baudm/parseq (README and GitHub licence field checked). <https://github.com/baudm/parseq> *(cv_pipeline)*
204. Magera et al., "Can Geometry Save Central Views for Sports Field Registration?", arXiv 2504.20052. <https://arxiv.org/abs/2504.20052> *(cv_pipeline)*
205. Roboflow, "How to Build Jersey Number Recognition for Sports" (4 May 2026). <https://roboflow.com/blog/jersey-number-recognition-for-sports> *(cv_pipeline)*
206. Caio, Van Zandycke, De Vleeschouwer, arXiv 2309.03640 (code: github.com/gabriel-vanzandycke/deepsport). <https://arxiv.org/abs/2309.03640> *(cv_pipeline)*
207. Van Zandycke & De Vleeschouwer, "3D Ball Localization From A Single Calibrated Image" (CVPR-W 2022; search snippet). <https://ar5iv.labs.arxiv.org/html/2204.00003> *(cv_pipeline)*
208. "Scoring smarter: deep learning-based basketball scoring detection in real time", BEEI (search snippet; page fetch failed). Patent claims came from USPTO search snippets. <https://beei.org/index.php/EEI/article/view/11374> *(cv_pipeline)*
209. <https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11594029> *(cv_pipeline)*
210. MMPose RTMPose project. <https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose> *(cv_pipeline)*
211. Ultralytics pose docs (YOLO26-pose). <https://docs.ultralytics.com/tasks/pose/> *(cv_pipeline)*
212. Hugging Face Transformers ViTPose docs. <https://huggingface.co/docs/transformers/model_doc/vitpose> *(cv_pipeline)*
213. Elixir YOLO, MacBook Air M3 benchmarks (third-party). <https://yolo.hexdocs.pm/macbook_air_m3.html> *(cv_pipeline)*
214. PyTorch MPS backend notes. <https://docs.pytorch.org/docs/2.14/notes/mps.html> *(cv_pipeline)*
215. Roboflow, "Best iOS Object Detection Models" (10 Nov 2025). <https://roboflow.com/blog/best-ios-object-detection-models> *(cv_pipeline)*
216. TeamTrack dataset (linked from Basketball-SORT). <https://github.com/AtomScott/TeamTrack> *(cv_pipeline)*
217. Theseus thesis, "Counting Basketball Shot Attempts with Computer Vision (PoC)" (search snippet). <https://www.theseus.fi/handle/10024/909540> *(cv_pipeline)*
218. OpenCV calib3d docs (`findHomography`; HTTP 403 on fetch). <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html> *(cv_pipeline)*
219. Roboflow Universe dataset page (HTTP 403 on fetch). <https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo> *(cv_pipeline)*
220. PyTorch 2.14 MPS environment variables (`PYTORCH_ENABLE_MPS_FALLBACK`). <https://docs.pytorch.org/docs/2.14/mps_environment_variables.html> *(cv_pipeline)*
221. Ultralytics YOLO27 preview page ("no launch date has been set"). <https://docs.ultralytics.com/models/yolo27/> *(cv_pipeline)*
222. OpenCV 4.x `calib3d.hpp` header (`findHomography` signature and method enum). <https://raw.githubusercontent.com/opencv/opencv/4.x/modules/calib3d/include/opencv2/calib3d.hpp> *(cv_pipeline)*
223. Wikipedia, "Basketball court" (FIBA/NBA dimensions, rim height; secondary). <https://en.wikipedia.org/wiki/Basketball_court> *(cv_pipeline, basketball_rules_stats)*
224. facebookresearch/sam3 SAM 3.1 release notes. <https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md> *(cv_pipeline)*
225. rf-detr `pyproject.toml` (`coreml` extra pins `coremltools>=8.0,<10.0`). <https://github.com/roboflow/rf-detr/blob/develop/pyproject.toml> *(cv_pipeline)*
226. NBA, *Rule No. 1: Court Dimensions – Equipment* (2026-27 rulebook web text). <https://official.nba.com/rule-no-1-court-dimensions-equipment/> *(basketball_rules_stats)*
227. NBA Official, *Rulebook index (2026-27)*. <https://official.nba.com/rulebook> *(basketball_rules_stats)*
228. NBA, *Rule No. 10: Violations and Penalties*. <https://official.nba.com/rule-no-10-violations-and-penalties/> *(basketball_rules_stats)*
229. NBA, *Rule No. 4: Definitions*. <https://official.nba.com/rule-no-4-definitions/> *(basketball_rules_stats)*
230. NBA, *Rule No. 12: Fouls and Penalties*. <https://official.nba.com/rule-no-12-fouls-and-penalties/> *(basketball_rules_stats)*
231. NBA, *Comments on the Rules* (Section L, Expiration of Time). <https://official.nba.com/comments-on-the-rules/> *(basketball_rules_stats)*
232. FIBA, *Official Basketball Rules Changes, valid as of 1 Oct 2026, v1.1* (July 2026). <https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2026-rule-changes-v1-1-en.pdf> *(basketball_rules_stats)*
233. FIBA, *Official Basketball Rules 2026 – Basketball Equipment v1.0*. <https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2026-equipment-v1-0.pdf> *(basketball_rules_stats)*
234. FIBA, *Resource hub downloads (Basketball Rules as of 1 Oct 2026)*. <https://about.fiba.basketball/en/services/resource-hub/downloads> *(basketball_rules_stats)*
235. NCAA, *Men's Basketball Playing Rules page*. <https://www.ncaa.org/championships/playing-rules/mens-basketball-playing-rules/> *(basketball_rules_stats)*
236. NCAA, *2025-26 and 2026-27 Men's Basketball Rules Changes*. <https://ncaaorg.s3.amazonaws.com/championships/sports/basketball/rules/men/2025-26PRMBB_RulesChanges.pdf> *(basketball_rules_stats)*
237. NCAA DI Men's Basketball Oversight Committee, *June 2026 report* (10-second count at shot clock 19). <https://ncaaorg.s3.amazonaws.com/championships/sports/basketball/d1/men/MBOC/Jun2026D1MBOC_Report.pdf> *(basketball_rules_stats)*
238. NCAA, *2023-24 and 2024-25 Women's Basketball Rules Changes*. <https://ncaaorg.s3.amazonaws.com/championships/sports/basketball/rules/women/2023-25PRWBB_RulesChanges.pdf> *(basketball_rules_stats)*
239. NCAA, *2025-26 and 2026-27 Women's Basketball Rules Changes*. <https://ncaaorg.s3.amazonaws.com/championships/sports/basketball/rules/women/2025-26PRWBB_RulesChanges.pdf> *(basketball_rules_stats)*
240. Abilene Basketball Referees (reproduction of NFHS Rule 1). <https://abilenebasketballrefs.com/nfhs-rules-book/nfhs-rules-book-rule-1-court-and-equipment/> *(basketball_rules_stats)*
241. Abilene Basketball Referees (reproduction of NFHS Rule 5). <https://abilenebasketballrefs.com/nfhs-rules-book/nfhs-rules-book-rule-5-scoring-and-timing-regulations/> *(basketball_rules_stats)*
242. NFHS, *Clock adjustments addressed … 2026-27 high school basketball rules changes*. <https://nfhs.org/stories/clock-adjustments-addressed-to-support-officials-in-2026-27-high-school-basketball-rules-changes> *(basketball_rules_stats)*
243. MaxPreps, *High school basketball: how many states use a shot clock* (May 7, 2026). <https://www.maxpreps.com/news/pYAZK_nH8ECtUCS5h883qQ/high-school-basketball-how-many-states-use-a-shot-clock.htm> *(basketball_rules_stats)*
244. NFHS, *Basketball Rules Changes 2021-22* (Rule 2-14 NEW: "effective with the 2022-2023 season, member state associations may establish a shot clock"; fetched by the verifier). <https://nfhs.org/resources/sports/basketball-rules-changes-2021-22> *(basketball_rules_stats)*
245. Basketball-Reference, *Glossary*. <https://www.basketball-reference.com/about/glossary.html> *(basketball_rules_stats)*
246. Basketball-Reference, *2024-25 NBA Season Summary*. <https://www.basketball-reference.com/leagues/NBA_2025.html> *(basketball_rules_stats)*
247. Basketball-Reference, *2025-26 NBA Season Summary*. <https://www.basketball-reference.com/leagues/NBA_2026.html> *(basketball_rules_stats)*
248. <https://api.pbpstats.com/get-totals/nba?Season=2025-26&SeasonType=Regular%20Season&Type=Team> *(basketball_rules_stats)*
249. pbpstats source, *field_goal.py* (shot zone, corner, distance, heave logic). <https://raw.githubusercontent.com/dblackrun/pbpstats/master/pbpstats/resources/enhanced_pbp/field_goal.py> *(basketball_rules_stats)*
250. pbpstats source, *__init__.py* (AT_RIM_CUTOFF=4, SHORT_MID_RANGE_CUTOFF=14, HEAVE 40 ft / 2 s). <https://raw.githubusercontent.com/dblackrun/pbpstats/master/pbpstats/__init__.py> *(basketball_rules_stats)*
251. KenPom, *Ratings glossary / blog* (possession formula with 0.475; page returned 403 to the fetcher, so quoted via search-result text). <https://kenpom.com/blog/ratings-glossary/> *(basketball_rules_stats)*
252. arXiv 2605.24056, J. Jacobs, *Possession-Level Player Impact in the Pre-Play-by-Play NBA Era: A Video-Reconstructed RAPM Database, 1984–1996* (22 May 2026). **Verifier note:** the full text does not contain a 0.44·FTA estimator, because possessions are tallied directly from video. It no longer supports any claim in this note. <https://arxiv.org/abs/2605.24056> *(basketball_rules_stats)*
253. nba_api, *parameters.py* (PlayType, ShotClockRange, DistanceRange enums). <https://raw.githubusercontent.com/swar/nba_api/master/src/nba_api/stats/library/parameters.py> *(basketball_rules_stats, jev_integration_design)*
254. nba_api docs, *leaguedashteamptshot*. <https://raw.githubusercontent.com/swar/nba_api/master/docs/nba_api/stats/endpoints/leaguedashteamptshot.md> *(basketball_rules_stats)*
255. nba_api docs, *synergyplaytypes*. <https://raw.githubusercontent.com/swar/nba_api/master/docs/nba_api/stats/endpoints/synergyplaytypes.md> *(basketball_rules_stats, jev_integration_design)*
256. Denver Stiffs, N. Timmons, *The NBA … where assists may or may not happen* (Apr 6, 2009; quotes the NBA statisticians' manual secondhand). <https://www.denverstiffs.com/the-nba-where-assists-may-or-may/> *(basketball_rules_stats)*
257. Wikipedia, *2026–27 NCAA Division I men's basketball season* (rule changes; corroborated by the June 2026 NCAA MBOC report). <https://en.wikipedia.org/wiki/2026%E2%80%9327_NCAA_Division_I_men's_basketball_season> *(basketball_rules_stats)*
258. NBA.com, *Teams Catch & Shoot* page (client-rendered; no data rows server-side). <https://www.nba.com/stats/teams/catch-shoot?Season=2024-25> *(basketball_rules_stats)*
259. FIBA Refereeing, *Official Basketball Rules page*. <https://refereeing.fiba.basketball/en/rules> *(basketball_rules_stats)*
260. Kubatko, Oliver, Pelton, Rosenbaum (2007), *A starting point for analyzing basketball statistics*, JQAS (bibliographic record). <https://sponet.de/sponet/Record/4013456> *(basketball_rules_stats)*
261. NFHS, *Basketball Rules Changes 2023-24* (Rule 4-8-1: one-and-one eliminated; two-FT bonus from the 5th team foul in each quarter; fouls reset each quarter). <https://nfhs.org/resources/sports/basketball-rules-changes-2023-24> *(basketball_rules_stats)*
262. IJCAI-ECAI 2026 proceedings entry (demo track, pp. 8427-8431). <https://www.ijcai.org/proceedings/2026/972> *(narration)*
263. LiveCC abstract (CVPR 2025). <https://arxiv.org/abs/2504.16030> *(narration)*
264. LiveCC GitHub repo. <https://github.com/showlab/livecc> *(narration)*
265. MatchTime full HTML (misalignment, MatchVoice). <https://arxiv.org/html/2406.18530> *(narration)*
266. MatchTime, EMNLP 2024 ACL Anthology. <https://aclanthology.org/2024.emnlp-main.99> *(narration)*
267. SoccerNet-Caption (CVPRW 2023). <https://arxiv.org/pdf/2304.04565> *(narration)*
268. SoccerNet-Echoes. <https://arxiv.org/abs/2405.07354> *(narration)*
269. Zheng et al., AI-generated game commentary survey, arXiv 2506.17294. <https://arxiv.org/abs/2506.17294> *(narration)*
270. IBM newsroom: US Open 2024 generative AI features (Granite 13B). <https://newsroom.ibm.com/2024-08-15-ibm-and-the-usta-serve-up-new-and-enhanced-generative-ai-features-for-2024-us-open-digital-platforms> *(narration)*
271. IBM newsroom: 90th Masters AI experiences (2026; Hole Insights third year). <https://newsroom.ibm.com/2026-03-23-ibm-debuts-new-ai-enabled-digital-experiences-for-the-90th-masters-tournament> *(narration)*
272. ESPN: Wimbledon and IBM AI-generated highlight commentary (2023). <https://espn.co.uk/tennis/story/_/id/37890817/wimbledon-ibm-team-ai-generated-highlight-commentary> *(narration)*
273. Awful Announcing: NBA/Amazon "Inside the Game" stats (Oct 1, 2025). <https://awfulannouncing.com/nba/inside-game-stats-platform-ai-aws-amazon.html> *(narration)*
274. Broadcast: Stats Perform Opta Stream for Urban Edge Network HBCU basketball (Dec 2024). <https://www.broadcastnow.co.uk/tech-innovation/stats-perform-unveils-ai-driven-stats-for-urban-edge-network/5199955.article> *(narration)*
275. CNN: NBC AI Al Michaels Olympic recaps (2024). <https://edition.cnn.com/2024/06/26/tech/al-michaels-ai-olympics> *(narration)*
276. GeekWire: Prime Video NBA "Prime 360" and AI tools (2026). <https://www.geekwire.com/2026/amazons-nba-coverage-adds-prime-360-ai-replays-that-fly-the-camera-through-the-action/> *(narration)*
277. <https://artificialanalysis.ai/models/claude-opus-5-5-low/providers> *(narration)*
278. <https://artificialanalysis.ai/models/claude-haiku-5-5-low> *(narration)*
279. OpenAI text-to-speech guide. <https://developers.openai.com/api/docs/guides/text-to-speech> *(narration)*
280. Cartesia TTS models documentation. <https://docs.cartesia.ai/build-with-cartesia/tts-models> *(narration)*
281. invideo review of Cartesia Sonic 3/3.5 latency (third-party). <https://invideo.io/blog/cartesia-sonic-ai-voice/> *(narration)*
282. hexgrad/kokoro GitHub. <https://github.com/hexgrad/kokoro> *(narration)*
283. HF discussion: Kokoro KPipeline ~11x realtime on M3 Max CPU (community). <https://huggingface.co/mlx-community/Kokoro-82M-bf16/discussions/5> *(narration)*
284. mlx-audio GitHub. <https://github.com/Blaizzy/mlx-audio> *(narration)*
285. piper-tts on PyPI (v1.8.0, GPL-3.0-or-later). <https://pypi.org/project/piper-tts/> *(narration)*
286. PromptQuorum Piper review (repo history / archival date; secondary). <https://www.promptquorum.com/power-local-llm/piper-tts-review> *(narration)*
287. kyutai-labs/pocket-tts GitHub. <https://github.com/kyutai-labs/pocket-tts> *(narration)*
288. Kyutai TTS page (220 ms, streaming text input). <https://kyutai.org/next/tts> *(narration)*
289. kyutai/tts-1.6b-en_fr model card. <https://huggingface.co/kyutai/tts-1.6b-en_fr> *(narration)*
290. sesame/csm-1b model card. <https://huggingface.co/sesame/csm-1b> *(narration)*
291. Apple AVSpeechSynthesizer documentation. <https://developer.apple.com/documentation/avfaudio/avspeechsynthesizer> *(narration)*
292. Ishchenko, fast speaking rate in TV football discourse (UEFA Euro 2012). <https://zenodo.org/records/19663708> *(narration)*
293. Wikipedia: Speech tempo. <https://en.wikipedia.org/wiki/Speech_tempo> *(narration)*
294. <https://www.espn.com/blog/truehoop/post/_/id/25832/how-much-trouble-is-early-foul-trouble> *(narration)*
295. NCAA: live statistics can be transmitted to benches in men's basketball (2021). <https://www.ncaa.org/news/2021/6/3/live-statistics-can-be-transmitted-to-benches-in-men-s-basketball.aspx> *(narration)*
296. Sports Illustrated: NBA will allow teams to review real-time video from the bench (2016). <https://www.si.com/media/2016/10/05/nba-real-time-video-review-bench> *(narration)*
297. ShotTracker: tech on the bench at the Hall of Fame Classic. <https://shottracker.com/the-next-frontier-of-coaching-tech-on-the-bench-at-the-hall-of-fame-classic/> *(narration)*
298. ESPN: Mountain West coaches to use real-time analytics (2019). <https://www.espn.com/mens-college-basketball/story/_/id/27475152/mwc-coaches-use-real-analytics> *(narration)*
299. Cleaning the Glass, "Ask Ben: Halftime, changing play calls..." (2017). <https://cleaningtheglass.com/ask-ben-halftime-changing-play-calls-and-technology-investment/> *(narration)*
300. Basketball-Reference: Four Factors. <https://www.inpredictable.com/2015/02/how-nba-games-are-won.html> *(narration)*
301. <https://basketball-reference.com/about/factors.html> *(narration)*
302. LiveCC-7B-Instruct model card (Apache-2.0, 8B params, BF16). <https://huggingface.co/chenjoya/LiveCC-7B-Instruct> *(narration)*
303. <https://github.com/rhasspy/piper> *(narration)*
304. <https://pypi.org/project/pyobjc-framework-AVFoundation/> *(narration)*
305. OpenRouter, "Jev" community guide (model ids `typesafe/jev-1.14`, `~typesafe/jev-latest`; Decisions API; 262,144-token context; long-context mode for >~32 KB text or images; no free-form text; thresholds per model). <https://openrouter.ai/docs/guides/community/jev.md> *(use_cases_market)*
306. OpenTweet, "Jev: Choice, Score, Noul" (secondary). <https://opentweet.io/jev/choice-score-noul> *(use_cases_market)*
307. Covers, "Sportradar unveils new cutting-edge NBA viewing experience" (2024-10-22). <https://www.covers.com/industry/nba-sportradar-new-player-tracking-technology> *(use_cases_market)*
308. NBA, "NBA and AWS announce new multi-year partnership" (2025-10-01). <https://pr.nba.com/nba-aws-partnership/> *(use_cases_market)*
309. NBA.com, "Inside the Game stat glossary". <https://nba.com/news/inside-the-game-stat-glossary> *(use_cases_market)*
310. NBA, "NBA and Genius Sports Second Spectrum expanded partnership" (2023). <https://pr.nba.com/nba-genius-sports-second-spectrum-expanded-partnership> *(use_cases_market)*
311. NBA/WNBA, "WNBA and Genius Sports bring tracking data to every WNBA arena" (2024). <https://pr.nba.com/wnba-genius-sports-tracking-data> *(use_cases_market)*
312. <https://www.cityam.com/?p=1804706> *(use_cases_market)*
313. iGaming Business, "NBA signs first data supply partnerships". <https://igamingbusiness.com/marketing-affiliates/nba-signs-first-data-supply-partnerships/> *(use_cases_market)*
314. Business Wire, "Sportradar Acquires Synergy Sports" (2021-03-22). <https://www.businesswire.com/news/home/20210322005437/en/> *(use_cases_market)*
315. Sportradar, "Synergy Basketball for High School". <https://sportradar.com/synergy-basketball-for-high-school/> *(use_cases_market)*
316. Hudl, "Hudl Assist for basketball". <https://www.hudl.com/products/assist/basketball> *(use_cases_market)*
317. Hudl, "Pricing for high school & middle school". <https://www.hudl.com/pricing/high-school/assist> *(use_cases_market)*
318. Hudl, "Club basketball plans & pricing". <https://www.hudl.com/pricing/club/hudl/basketball> *(use_cases_market)*
319. Hudl, "Hudl Focus cameras". <https://www.hudl.com/products/focus> *(use_cases_market)*
320. Utah Public Notice (procurement document with Hudl Focus Flex quote). <https://www.utah.gov/pmn/files/1476377.pdf> *(use_cases_market)*
321. Hudl, "Hudl expands volleyball focus through acquisition of Balltime" (2025-02). <https://www.hudl.com/blog/hudl-expands-volleyball-focus-through-game-changing-acquisition-of-balltime> *(use_cases_market)*
322. Silicon Prairie News, "Hudl acquires Titan Sports and Balltime" (2025-06). <https://siliconprairienews.com/2025/06/lincoln-based-hudl-acquires-titan-sports-and-balltime-expanding-access-to-performance-data-for-athletes/> *(use_cases_market)*
323. Pixellot homepage. <https://www.pixellot.tv/> *(use_cases_market)*
324. Youth Sports Business Report, "Pixellot and PlayOn extend partnership through 2030". <https://youthsportsbusinessreport.com/pixellot-and-playon-extend-high-school-streaming-partnership-through-2030/> *(use_cases_market)*
325. HoopBrief, "Pixellot cost 2026" (third-party). <https://hoopbrief.com/blog/pixellot-cost-2026> *(use_cases_market)*
326. NFHS Network / PlayOn, "Become an NFHS Network school". <https://nfhsnetwork.playonsports.com/overview> *(use_cases_market)*
327. DeepCut News, "Basketball Australia considers Israeli AI tech to record junior games" (2026). <https://www.deepcutnews.com/p/basketball-australia-considers-israeli> *(use_cases_market)*
328. Veo, "Pricing". <https://www.veo.com/en-us/pricing> *(use_cases_market)*
329. Veo, "Buyer's guide for parents & youth teams". <https://www.veo.com/en-us/article/buyers-guide-parents-youth-teams> *(use_cases_market)*
330. Pulse 2.0, "SportsVisio: $3.2 million raised" (2025-06). <https://pulse2.com/sportsvisio-3-2-million-raised-for-scaling-ai-sports-solution> *(use_cases_market)*
331. CNBC, "ShotTracker gets backing from Verizon in $11 million round" (2021-01-14). <https://www.cnbc.com/2021/01/14/shottracker-gets-backing-from-verizon-in-11-million-financing-round-.html> *(use_cases_market)*
332. Sportico, "Live basketball stats company ShotTracker / Hearst" (2021). <https://www.sportico.com/business/tech/2021/live-basketball-stats-company-shottracker-hearst-1234626143/> *(use_cases_market)*
333. Ubergizmo, "HomeCourt basketball app uses AI to train" (2018). <https://ubergizmo.com/2018/08/homecourt-basketball-app-uses-ai-to-train> *(use_cases_market)*
334. FitGap, "HomeCourt" (third-party). <https://us.fitgap.com/products/039841/homecourt> *(use_cases_market)*
335. Apple App Store, "Ball AI - Basketball Training". <https://apps.apple.com/us/app/-/id6450125219> *(use_cases_market)*
336. Signalbase, "Ball AI secures $475K". <https://www.trysignalbase.com/news/funding/ball-ai-secures-475k-pre> *(use_cases_market)*
337. Y Combinator, "Launch YC: PlayVision". <https://www.ycombinator.com/launches/OhD-playvision-ai-moneyball-for-sports> *(use_cases_market)*
338. Y Combinator, "PlayVision" company page (on 2026-10-20 this URL shows "Bron", Fall 2025, with the same founders). <https://www.ycombinator.com/companies/playvision> *(use_cases_market)*
339. Prep Network, "AI-powered BallerCam built for youth sports streaming" (2025-07-24). <https://prepnetwork.com/2025/07/24/ai-powered-ballercam-built-for-youth-sports-streaming/> *(use_cases_market)*
340. Hooper. <https://www.hooper.gg/> *(use_cases_market)*
341. IBM UK Newsroom, "IBM at Wimbledon 2025". <https://uk.newsroom.ibm.com/IBM-at-Wimbledon-2025> *(use_cases_market)*
342. Fox News, "AI-generated voice of former narrator Jim Fagan to be featured next NBA season, NBC says". <https://www.foxnews.com/sports/ai-generated-voice-former-narrator-jim-fagan-featured-next-nba-season-nbc-sports-says.amp> *(use_cases_market)*
343. Tech.eu, "Sportway raises €20M" (2026-06-23). <https://tech.eu/2026/06/23/from-the-sidelines-to-250000-matches-a-year-sportway-raises-eur20m-to-scale-its-ai-sports-media-platform/> *(use_cases_market)*
344. FootyHub TV. <https://footyhub.tv/> *(use_cases_market)*
345. Microsoft Accessibility Blog, "OneCourt: revolutionizing accessibility in live sports" (2025-01-14). <https://blogs.microsoft.com/accessibility/onecourt-revolutionizing-accessibility-in-live-sports/> *(use_cases_market)*
346. Axios Denver, "Denver Nuggets debut OneCourt tech for blind fans" (2026-01-13). <https://www.axios.com/local/denver/2026/01/13/denver-nuggets-onecourt-tech-blind-fans> *(use_cases_market)*
347. WRTV, "Pacers launch audio description service for visually impaired fans" (2025-10). <https://www.wrtv.com/sports/pacers/pacers-launch-audio-description-service-for-visually-impaired-fans> *(use_cases_market)*
348. Sony Europe, "Hawk-Eye Innovations and Action Audio debut inclusive feed for Wimbledon's Centre Court". <https://www.sony.eu/presscentre/hawk-eye-innovations-and-action-audio-debut-inclusive-feed-for-wimbledons-centre-court> *(use_cases_market)*
349. Digital Trends, "Action Audio soundscapes for sports". <https://www.digitaltrends.com/outdoors/action-audio-soundscapes-for-sports/> *(use_cases_market)*
350. Cyprus Mail (Reuters), "League to use AI for objective calls, Silver says" (2026-05-28). <https://cyprus-mail.com/2026/05/28/league-to-use-ai-for-objective-calls-silver-says> *(use_cases_market)*
351. ACC, "ACC selects RefMasters' RefScore for men's basketball officials" (URL date 2026-01-22; page dated 2026-01-23). <https://theacc.com/news/2026/1/22/acc-selects-refmasters-refscore-as-its-performance-development-platform-for-mens-basketball-officials.aspx> *(use_cases_market)*
352. NBA, "NBA referee training application named Innovation of the Year" (2022). <https://pr.nba.com/nba-referee-training-application-award> *(use_cases_market)*
353. Wareable, "Kinexon's wearable sensor is changing the way NBA teams prep". <https://www.wareable.com/sport/kinexon-wearable-sensor-nba-basketball-6679> *(use_cases_market)*
354. Vuckovic et al., Montenegrin Journal of Sports Science and Medicine (March 2026), Kinexon LPS. <https://www.mjssm.me/clanci/MJSSM_March_2026_Vuckovic.pdf> *(use_cases_market)*
355. FiveThirtyEight, "The NBA is turning to wearable sensors to prevent player injuries". <https://fivethirtyeight.com/features/the-nba-is-turning-to-wearable-sensors-to-prevent-player-injuries> *(use_cases_market)*
356. MobiHealthNews, "NBA bans use of wearable device data in contract negotiations" (2017). <https://www.mobihealthnews.com/news/nba-bans-use-wearable-device-data-contract-negotiations> *(use_cases_market)*
357. *NBA v. Motorola, Inc.*, 105 F.3d 841 (2d Cir. 1997). <https://law.justia.com/cases/federal/appellate-courts/F3/105/841/598844/> *(use_cases_market)*
358. Awful Announcing, "March Madness social media rules holding teams back". <https://awfulannouncing.com/ncaa/march-madness-social-media-rules-holding-teams-back.html> *(use_cases_market)*
359. Delaware DIAA, "Championship streaming". <https://education.delaware.gov/diaa/sport_championships/streaming> *(use_cases_market)*
360. Illinois Public Act 103-0769. <https://ilga.gov/legislation/publicacts/103/103-0769.htm> *(use_cases_market)*
361. Thompson Hine, "Illinois legislature amends BIPA to limit damages and expand consent options" (2024). <https://www.thompsonhine.com/insights/illinois-legislature-amends-bipa-to-limit-damages-and-expand-consent-options/> *(use_cases_market)*
362. Data Protection Report (Norton Rose Fulbright), "BIPA and the record retention requirement" (2022-12); Thompson Coburn on *Mora v. J&M Plating*. <https://www.dataprotectionreport.com/2022/12/bipa-and-the-record-retention-requirement/> *(use_cases_market)*
363. <https://www.thompsoncoburn.com/insights/publications/item/2022-12-15/illinois-appeals-court-holds-that-bipa-section-15(a)-requires-retention-and-destruction-policies-upon-possession-of-any-biometric-data> *(use_cases_market)*
364. Stinson, "The privacy playbook: how sports, media, entertainment and apparel companies became litigation targets" (2026). <https://www.stinson.com/newsroom-publications-the-privacy-playbook-how-sports-media-entertainment-and-apparel-companies-became-litigation-targets> *(use_cases_market)*
365. Reed Smith, "Chicago Blackhawks hit with Illinois BIPA class action". <https://www.reedsmith.com/our-insights/blogs/adlaw-by-request/102jzuc/chicago-blackhawks-hit-with-illinois-bipa-class-action-over-use-of-facial-recogni> *(use_cases_market)*
366. Blank Rome, "Meet CUBI: Texas' biometric privacy law". <https://blankrome.com/publications/meet-cubi-what-companies-need-know-about-texas-biometric-privacy-law> *(use_cases_market)*
367. Texas Attorney General, "$1.4 billion settlement with Meta" (2024-07-30). <https://www.texasattorneygeneral.gov/news/releases/attorney-general-ken-paxton-secures-14-billion-settlement-meta-over-its-unauthorized-capture> *(use_cases_market)*
368. ZwillGen, "Don't mess with Texas: CUBI settlement". <https://www.zwillgen.com/ftc-state-ag/dont-mess-with-texas-cubi-settlement-marks-state-tough-biometric-privacy-enforcer/> *(use_cases_market)*
369. Hunton, "Washington becomes third state to enact biometric privacy law". <https://www.hunton.com/privacy-and-cybersecurity-law-blog/washington-becomes-third-state-enact-biometric-privacy-law> *(use_cases_market)*
370. Colorado Legislative Council, HB24-1130 fiscal note. <https://leg.colorado.gov/sites/default/files/documents/2024A/bills/fn/2024a_hb1130_f1.pdf> *(use_cases_market)*
371. BBB National Programs, "Reminder: amended COPPA rule compliance date is approaching". <https://bbbprograms.org/media/insights/blog/coppa-amended> *(use_cases_market)*
372. US Dept. of Education, "FAQs on photos and videos under FERPA". <https://studentprivacy.ed.gov/faq/faqs-photos-and-videos-under-ferpa> *(use_cases_market)*
373. Capital News Service (Maryland), "Navigating NIL: a patchwork of policies governing high school athletes" (2025-05-06). <https://cnsmaryland.org/2025/05/06/navigating-nil-a-patchwork-of-policies-governing-high-school-athletes/> *(use_cases_market)*
374. EDPB, Guidelines 3/2019 on processing of personal data through video devices. <https://www.edpb.europa.eu/sites/default/files/files/file1/edpb_guidelines_201903_video_devices_en.pdf> *(use_cases_market)*
375. Regulation (EU) 2016/679 (GDPR), Arts. 6, 8, 9, 35. <https://eur-lex.europa.eu/eli/reg/2016/679/oj> *(use_cases_market)*
376. Orrick, "EU Commission publishes guidelines on prohibited AI practices under the AI Act" (2025-04). <https://www.orrick.com/en/insights/2025/04/eu-commission-publishes-guidelines-on-the-prohibited-ai-practices-under-the-ai-act> *(use_cases_market)*
377. Legal Sports Report, "How Project Red Card data fight could affect US sports betting"; Lewis Silkin, "Project Red Card". <https://www.legalsportsreport.com/?p=67676> *(use_cases_market)*
378. <https://sportsinsights.lewissilkin.com/post/102gcsz/project-red-card-footballers-to-sue-over-use-of-their-player-data-and-match-stat> *(use_cases_market)*
379. Society for Computers & Law, "Database right: ECJ judgment in Football Dataco v Sportradar". <https://www.scl.org/2595-database-right-ecj-judgment-in-football-data-co-v-sportradar/> *(use_cases_market)*
380. Van Zandycke et al., "DeepSportradar-v1" (arXiv 2208.08190). <https://arxiv.org/abs/2208.08190v1> *(use_cases_market, training_data)*
381. Dataset Ninja, "DeepSportradar" (lists CC BY-NC-SA 4.0; third-party). <https://datasetninja.com/deep-sport-radar> *(use_cases_market, training_data)*
382. Legal Sports Report, "Illinois 'commercially reasonable' pricing rules for official league data". <https://www.legalsportsreport.com/37240/illinois-sports-betting-rules-commercially-reasonable/> *(use_cases_market)*
383. Vixio, "Tennessee to drop data mandate". <https://www.vixio.com/insights/gc-tennessee-drop-data-mandate-part-sports-wagering-reshuffle> *(use_cases_market)*
384. Legal Sports Report, "NCAA grants sportsbooks first-time access to official data, logos". <https://www.legalsportsreport.com/231676/ncaa-grants-sportsbooks-first-time-access-to-official-data-logos/> *(use_cases_market)*
385. WSN, "States consider banning college player prop bets after NCAA request" (2026). <https://www.wsn.com/betting/states-consider-banning-college-player-prop-bets-after-ncaa-request/> *(use_cases_market)*
386. Boston Globe, "Chauncey Billups, Terry Rozier federal charges" (2025-10-23). <https://www.bostonglobe.com/2025/10/23/sports/chauncey-billups-terry-rozier-federal-charges> *(use_cases_market)*
387. Legal Sports Report, "NBA, players union back more limits on prop bets" (2025). <https://www.legalsportsreport.com/239853/nba-players-union-back-more-limits-on-prop-bets/> *(use_cases_market)*
388. NFHS, "Supreme Court's decision on sports gambling: effects on high school games". <https://nfhs.org/stories/supreme-court-s-decision-on-sports-gambling-effects-on-high-school-games> *(use_cases_market)*
389. Basketball Australia, "Basketball Australia introduces courtsiding policy". <https://australia.basketball/news/3337783/basketball-australia-introduces-courtsiding-policy> *(use_cases_market)*
390. NCAA, "Institutional responsibilities regarding recruiting/scouting service approvals". <https://www.ncaa.org/sports/2020/11/24/institutional-responsibilities-regarding-recruiting-scouting-service-approvals.aspx> *(use_cases_market)*
391. <http://www.ncaa.org/eligibility-center/recruiting/recruiting-and-scouting-service> *(use_cases_market)*
392. Northwestern University, "Recruiting service approval form" (school practice example). <https://www.northwestern.edu/athleticbusinessoffice/forms/recruiting-service-approval-form.pdf> *(use_cases_market)*
393. Hoopsalytics blog, "Hudl Assist alternative for basketball" (competitor marketing). <https://blog.hoopsalytics.com/hudl-assist-alternative-for-basketball/> *(use_cases_market)*
394. Basketball Australia, "Photography and filming of young people" (2026-05). <https://resources.basketball-australia.pulselive.com/basketball-australia/document/2026/05/28/030f58b4-3b13-482f-9c6d-2db0e87b2ce6/Photograhy_Filming_Young_People_Doc.pdf> *(use_cases_market)*
395. Level Up Coding, "Stop using generative LLMs for every decision: meet Jev" (secondary; vendor latency/pricing figures, unverified). <https://levelup.gitconnected.com/stop-using-generative-llms-for-every-decision-meet-jev-96f2ec183fc4> *(use_cases_market)*
396. DeepSportradar GitHub organisation. <https://github.com/DeepSportradar> *(training_data)*
397. DeepSportradar camera-calibration-challenge. <https://github.com/DeepSportradar/camera-calibration-challenge> *(training_data)*
398. DeepSportradar player-reidentification-challenge. <https://github.com/DeepSportradar/player-reidentification-challenge> *(training_data)*
399. DeepSportradar ball-3d-localization-challenge. <https://github.com/DeepSportradar/ball-3d-localization-challenge> *(training_data)*
400. gabriel-vanzandycke/deepsport. <https://github.com/gabriel-vanzandycke/deepsport> *(training_data)*
401. SpaceJam. <https://github.com/simonefrancia/SpaceJam> *(training_data)*
402. hkair Basketball-Action-Recognition (SpaceJam classes, R(2+1)D). <https://github.com/hkair/Basketball-Action-Recognition> *(training_data)*
403. NBA-Player-Movements `Moment.py`. <https://raw.githubusercontent.com/linouk23/NBA-Player-Movements/master/Moment.py> *(training_data)*
404. sealneaward/nba-movement-data (SportVU 2015-16 mirror; shot-time heuristic). <https://github.com/sealneaward/nba-movement-data> *(training_data)*
405. <https://github.com/linouk23/NBA-Player-Movements/tree/master/data/2016.NBA.Raw.SportVU.Game.Logs> *(training_data)*
406. Ramanathan et al., "Detecting events and key actors in multi-person videos", CVPR 2016. <https://arxiv.org/abs/1511.02917> *(training_data)*
407. <https://cs.stanford.edu/groups/vision/pdf/ramanathan2016cvpr.pdf> *(training_data)*
408. Shakya et al., "Basketball-51", CS & IT 2021. <https://aircconline.com/csit/papers/vol11/csit110712.pdf> *(training_data)*
409. Basketball-51 on Kaggle. <https://www.kaggle.com/datasets/sarbagyashakya/basketball-51-dataset> *(training_data)*
410. <https://bodai.unibs.it/?p=1198> *(training_data)*
411. BASKET (arXiv 2503.20781). <https://arxiv.org/abs/2503.20781> *(training_data)*
412. SportsMOT on Hugging Face. <https://huggingface.co/datasets/MCG-NJU/SportsMOT> *(training_data)*
413. Roboflow, "How to Train a YOLO26 Object Detection Model with Custom Data". <https://roboflow.com/blog/how-to-train-yolo26-custom-data> *(training_data)*
414. Roboflow Universe X-Basket 2. <https://universe.roboflow.com/x-qfelz/x-basket-2> *(training_data)*
415. Roboflow Universe basketball-jaawc. <https://universe.roboflow.com/machinelearning-6xoxa/basketball-jaawc> *(training_data)*
416. Roboflow Universe baskball. <https://universe.roboflow.com/dataset-baketball/baskball> *(training_data)*
417. hoopR `nba_videoevents`. <https://search.r-project.org/CRAN/refmans/hoopR/html/nba_videoevents.html> *(training_data)*
418. NBA Plays 2022-23 data dictionary (xLegacy/yLegacy semantics; secondary). Dead as of 2026-10-20: it 301-redirects to baselight.ai, which shows a shutdown notice. <https://baselight.app/u/kaggle/dataset/brendankdata_nbaplays22_23> *(training_data)*
419. "Leveraging Contextual Cues for Generating Basketball Highlights" (arXiv 1606.08955; OCR alignment). <https://arxiv.org/pdf/1606.08955> *(training_data)*
420. MIT Sloan SAC paper "Predicting NBA Talent from Enormous Amounts of College Basketball Tracking Data" (paper ID 91W8YT29R; authorship unverified). <https://global-uploads.webflow.com/5f1af76ed86d6771ad48324b/606dbe887622a74e9dc17f4b_91W8YT29R.pdf> *(training_data)*
421. dcervone/EPVDemo. <https://github.com/dcervone/EPVDemo> *(training_data)*
422. baller2vec (arXiv 2102.03291). <https://github.com/airalcorn2/baller2vec> *(training_data)*
423. RF-DETR training docs. <https://rfdetr.roboflow.com/learn/train/> *(training_data)*
424. Ultralytics train mode docs. <https://docs.ultralytics.com/modes/train/> *(training_data)*
425. Ultralytics YOLO Vision 2026 / YOLO27 announcement. <https://www.ultralytics.com/blog/key-highlights-from-ultralytics-yolo-vision-2026> *(training_data)*
426. Akyon et al., "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection", ICIP 2022 (arXiv 2202.06934). <https://arxiv.org/abs/2202.06934> *(training_data)*
427. CVAT blog: SAM 3 integration. <https://www.cvat.ai/resources/blog/sam-3-image-segmentation> *(training_data)*
428. Label Studio SAM 2 video tutorial. <https://labelstud.io/guide/ml_tutorials/segment_anything_2_video> *(training_data)*
429. Label Studio VideoVector + SAM 2 blog. <https://labelstud.io/blog/video-object-segmentation-and-tracking-with-videovector-tag-for-sam-2-in-label-studio> *(training_data)*
430. Soft Teacher (arXiv 2106.09018). <https://arxiv.org/abs/2106.09018> *(training_data)*
431. E2E-Spot (Hong et al., ECCV 2022). <https://github.com/jhong93/spot> *(training_data)*
432. T-DEED (arXiv 2404.05392, CVPRW 2024). <https://arxiv.org/abs/2404.05392> *(training_data)*
433. SCBench (arXiv 2412.17637). <https://arxiv.org/html/2412.17637v1> *(training_data)*
434. TennisExpert (arXiv 2603.13397). <https://arxiv.org/pdf/2603.13397> *(training_data)*
435. OpenRouter, "What Is Jev?". <https://openrouter.ai/blog/insights/what-is-jev/> *(training_data)*
436. "Just Ask Jev: Reinforcement Learning for Calibrated Decisions…" (arXiv 2609.29429). <https://arxiv.org/abs/2609.29429> *(training_data)*
437. "Spatio-Temporal Analysis of Team Sports — A Survey" (ESQ description; arXiv 1602.06994). <https://arxiv.org/pdf/1602.06994> *(training_data)*
438. BasketHAR (arXiv 2604.17065). <https://arxiv.org/pdf/2604.17065> *(training_data)*
439. <https://stats.nba.com/stats/playbyplayv3> *(training_data)*
440. <https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json> *(training_data)*
441. <https://raw.githubusercontent.com/obss/sahi/main/docs/predict.md> *(training_data)*
442. <https://raw.githubusercontent.com/obss/sahi/main/LICENSE> *(training_data)*
443. <https://github.com/holhouse/PL-NBA-Dataset> *(training_data)*
444. <https://github.com/arnalytics/labeled_plays_NBA> *(training_data)*
445. TypeSafe, "Guardrails for LLMs" cookbook (Noul batteries, review 0.35 / action 0.70-0.85, `jev-1.12`, safety-only). <https://docs.typesafe.ai/cookbooks/llm_guardrails.md> *(jev_integration_design)*
446. Tang, Laban, Durrett, "MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents", EMNLP 2024. <https://arxiv.org/abs/2404.10774> *(jev_integration_design)*
447. Geifman, El-Yaniv, "Selective Classification for Deep Neural Networks" (risk-coverage, reject option). <https://arxiv.org/abs/1705.08500> *(jev_integration_design)*
448. NBA.com Stats, Clutch Traditional ("Last 5 Minutes" and "5 Point Diff or Less" filters). <https://www.nba.com/stats/players/clutch-traditional> *(jev_integration_design)*

### 8.4 Sources added in the 2026-10-20 gap pass (§1A, §7A)

All fetched on 2026-10-20. Web pages are data; quotations are short.

449. Anthropic Privacy Center, "How long do you store my organization's data?" (dated 2026-07-01; API inputs/outputs deleted within 30 days; flagged items up to 2 years). <https://privacy.claude.com/en/articles/7996866-how-long-do-you-store-my-organization-s-data>
450. Anthropic, Commercial Terms of Service (effective 2025-06-17; no training on Customer Content; DPA incorporated by reference). <https://www.anthropic.com/legal/commercial-terms>
451. TypeSafe docs, "Legal" (ZDR offered to enterprise customers). <https://docs.typesafe.ai/legal.md>
452. TypeSafe, Data Processing Agreement (controller/processor roles, subprocessor notice and 15-day objection, 72 h breach notice, EU SCC Modules 2 and 3). <https://typesafe.ai/legal/data-processing>
453. TypeSafe docs, "Self-consistency: choices" cookbook (15 calls with a nonce: 90.8% raw agreement; mean probability SD 0.0098; max SD 0.0515). <https://docs.typesafe.ai/cookbooks/consistency_choice_cookbook.md>
454. TypeSafe docs, "Request rules and errors" (32 MiB body, 64-level nesting, 422 validation, 400 `max_tokens_exceeded`, 429, 529). <https://docs.typesafe.ai/api/errors.md>
455. 34 CFR § 99.31(a)(1)(i)(B), FERPA school-official exception (eCFR). <https://www.ecfr.gov/current/title-34/subtitle-A/part-99/subpart-D/section-99.31>
456. 16 CFR § 312.2, COPPA definitions (eCFR). <https://www.ecfr.gov/current/title-16/chapter-I/subchapter-C/part-312/section-312.2>
457. roboflow/rf-detr source, `src/rfdetr/models/postprocess.py` (`PostProcess`, `num_select=300`, sigmoid, top-k, cxcywh→xyxy, scaling). <https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/models/postprocess.py>
458. roboflow/rf-detr source, `src/rfdetr/config.py` (Nano 384 / Small 512 / Medium 576, `num_queries=300`, `TrainConfig` with PyTorch Lightning `accelerator="auto"`). <https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/config.py>
459. roboflow/rf-detr source, `src/rfdetr/models/lwdetr.py` (`forward_export` returns `(boxes, logits)`). <https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/models/lwdetr.py>
460. rfdetr on PyPI, JSON metadata (1.11.2 uploaded 2026-10-04; requires `torch>=2.2.0`, `transformers>=5.1.0,<6.0.0`, `supervision>=0.29.0`; extras `coreml`, `train`, `coreai`). <https://pypi.org/pypi/rfdetr/json>
461. trackers on PyPI, JSON metadata (2.6.1, 2026-09-25; requires `opencv-python>=4.8.0`, `supervision>=0.26.1`). <https://pypi.org/pypi/trackers/json>
462. supervision on PyPI, JSON metadata (0.31.0; MIT; no OpenCV in `requires_dist`). <https://pypi.org/pypi/supervision/json>
463. roboflow/supervision source, `src/supervision/geometry/core.py` (`Position` enum incl. `BOTTOM_CENTER`). <https://github.com/roboflow/supervision/blob/develop/src/supervision/geometry/core.py>
464. Apple, WWDC23 "Improve Core ML integration with async prediction" (synchronous prediction not thread-safe; async API thread-safe; workload-dependent concurrency gains). <https://developer.apple.com/videos/play/wwdc2023/10049/>
465. coremltools guide, "Model prediction" (`compute_units=` at load). <https://apple.github.io/coremltools/docs-guides/source/model-prediction.html>
466. Apple Newsroom, "Apple unveils M3, M3 Pro, and M3 Max" (2023-10). <https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/>
467. PaddlePaddle, macOS pip installation (Apple Silicon supported, CPU only, Python 3.9-3.13, `paddlepaddle==3.3.0`). <https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/macos-pip_en.html>
468. PaddleOCR 3.x installation. <http://www.paddleocr.ai/latest/en/version3.x/installation.html>
469. obs-websocket 5.x protocol, `SendStreamCaption`. <https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md>
470. YouTube Help, "Add live captions to a live stream". <https://support.google.com/youtube/answer/3068031>
471. IETF RFC 8216, HTTP Live Streaming, §3.5 WebVTT and `EXT-X-MEDIA`. <https://datatracker.ietf.org/doc/html/rfc8216>
472. Abilene Basketball Referees, reproduction of NFHS Rule 6 (alternating possession, 6-4). <https://abilenebasketballrefs.com/nfhs-rules-book/nfhs-rules-book-rule-6-live-ball-and-dead-ball/>
473. Abilene Basketball Referees, reproduction of the NFHS Case Book, Rule 4 (Case 4.14.1: fifth foul, personal and technical counted). <https://abilenebasketballrefs.com/nfhs-case-book/nfhs-case-book-rule-4-definitions/>
474. Abilene Basketball Referees, reproduction of the NFHS Case Book, Rule 10 (Cases 10.5.1, 10.5.5, 10.6.2, 10.6.5). <https://abilenebasketballrefs.com/nfhs-case-book/nfhs-case-book-rule-10-fouls-and-penalties/>
475. Hudl, Terms (updated 2026-08-11; §3.2.3, §3.2.8, §4.3.2, §5.4). <https://www.hudl.com/terms>
476. NFHS Network, Terms of Use (last updated November 2024). <https://www.nfhsnetwork.com/termsofuse>
477. Veo, General Terms and Conditions and User Terms (dated 2026-05-29). <https://www.veo.com/terms>
478. Pixellot, Privacy Policy page (no terms of use linked from the site footer). <https://www.pixellot.tv/privacy-policy>
479. scikit-learn User Guide, "Probability calibration". <https://scikit-learn.org/stable/modules/calibration.html>
480. Niculescu-Mizil & Caruana, "Predicting Good Probabilities With Supervised Learning", ICML 2005 (Platt beats isotonic below about 200-1,000 calibration cases). <https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf>
481. "Rule of three (statistics)" (95% upper bound 3/n with zero events; Hanley & Lippman-Hand, JAMA 1983). <https://en.wikipedia.org/wiki/Rule_of_three_(statistics)>
482. Lambda, GPU Cloud on-demand pricing. <https://lambda.ai/service/gpu-cloud>

Re-used sources with new facts in this pass: [3] (rules note tables), [9] (hardware readout), [16] (98.1% = 759/774 frames; 84.9% validation recall; no `ball-in-basket` class in that post), [22] (CoreML I/O, ANE notes), [23] (Opus 5.5 / Haiku 5.5 thinking, pricing, cache minimum, ZDR note), [24] (AA median 3.73 s; TTFT definition silent on thinking), [42] (trackers parameters), [54] (FIBA Arts. 12.6, 18.2, 36.3.2, 37.2, 38.2, 44, timer duties), [55] (NBA Rule 5 §V-VI), [59] (NCAA men Rule 2-10 Art. 12, Rule 5 timeouts), [60] (NCAA women timeouts and clock), [241] (NFHS Rule 5-5, 5-8-1, 5-11), [423] (RF-DETR training settings).
