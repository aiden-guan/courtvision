# Real-time, factual automated commentary: research and latency engineering for CourtVision narration

Researched 2026-10-19/20. Scope: how research systems and deployed products produce live sports
commentary, how to keep it factual, how to pace it, what the latency budget looks like on an
Apple M3 Pro, which TTS engines are usable from Python on macOS, and what coaches need from the
same data. Sections 8-9 give the recommended CourtVision narration design and its concrete parameters.

Conventions: `[n]` maps to the Sources list. `[M]` means I measured it on this machine (Apple M3 Pro,
macOS 26.5.1, Python 3.12.3). "(unverified)" marks claims I could not confirm from a primary
source. "(design proposal)" marks parameter values I am proposing. They are starting points to tune
on labelled games, not findings. "Jev" always means **Typesafe Jev**, TypeSafe's System One model
answering `Choice` / `Score` / `Noul` questions with calibrated probabilities. Its API facts come
from the sibling note `docs/research/jev_api.md` [61].

Search budget note: the web-search quota for this run ran out near the end. A few items (Piper
voice licences, Kokoro speed on MPS, OTT stream delays) are therefore marked unverified rather than
chased further.

---

## 0. TL;DR: what to build

1. **The LLM never decides facts.** CV → event engine → versioned **fact store**. Every utterance
   is rendered from a **FactPacket**: a JSON snapshot of only the facts the line may use. Live-ball
   calls (made shots, steals, blocks) are **deterministic templates**. They take under 1 ms to render
   and cannot hallucinate. Claude (`claude-opus-5-5`) writes only dead-ball colour, summaries and
   coach digests, and a validator checks every name and number against the FactPacket before audio
   plays. This matches how the most robust systems found here work: MARIO's "machine-validated
   fields" plus "one short factual sentence" [15], Mori et al.'s template event description followed
   by LLM styling [14], and IBM's stats-in, regex and multi-LLM validation [17].
2. **End-to-end video-LLMs cannot do identity.** On a 2025-26 NBA benchmark, the best online system
   gets player identity right **28.5%** of the time [13]. MatchTime anonymises players entirely [7].
   So CourtVision must get names from tracking plus jersey OCR plus the roster, never from a VLM.
3. **Claude Opus 5.5 is too slow for play-by-play.** Its thinking is always on. Artificial Analysis
   measures time to first answer token at **6.5 s** at `low` effort (Anthropic API, their default
   10,000-input-token workload, reasoning time included) [28]. Use Claude only where a dead ball leaves at least 3-4 s of slack,
   and pre-generate candidate lines in parallel with playback. That parallel buffering is the core
   idea of Kawamatsu et al., which cut inter-utterance silence from **9.5 s to 0.3-0.4 s** [2].
4. **Jev is not on the live-call hot path either.** Its documented median is about 0.4 s per
   request [61]. For live calls, gate with CV-derived confidences. Use Jev on dead-ball decisions:
   what kind of line to say, salience, candidate choice, coach-alert relevance.
5. **Local TTS default: macOS `AVSpeechSynthesizer` through PyObjC.** I measured a warm first audio
   buffer at **12-26 ms** (two runs), and a 6 s line fully synthesised in **67-75 ms** [M]. Do **not**
   shell out to `say` for each utterance: that cost 0.8-2.8 s per call for a 0.86 s clip (two runs) [M]. The quality upgrade
   path is Kokoro-82M (Apache-2.0) [35][36]. ElevenLabs `eleven_flash_v2_5` (~75 ms model latency) is
   the optional cloud option [30].
6. **Pace like a human.** Professional commentators pause about **1.7 s** between utterances
   (fighting-game commentary [2]) or **2.14 s** (soccer [14]). Utterances last about **2.8 s**
   [2] and run about **13 words** in NBA broadcast commentary [13]. Do not minimise silence to zero:
   the parallel system's 0.3 s gaps were shorter than the human 1.7 s [2].
7. **Corrections are first-class.** Every event carries a revision number. A revision before speech
   is silent. A revision to an already-spoken material fact (scorer, points, score) gets a short P0
   "Correction:" template line in the next gap. Replay-reviewable situations (2 vs 3, shot clock,
   out of bounds) are narrated with a hedge until resolved [60].

---

## 1. Research systems

### 1.1 Kawamatsu et al., "Low-Latency Real-Time Audio Game Commentary System via LLM-Based Parallel Text Generation" (arXiv 2606.13322, IJCAI-ECAI 2026 Demo)

Authors from the University of Tokyo, AIST, CMU and others (Kawamatsu, Afzal, Saito, Takamichi,
Neubig, Sudoh, Takamura, Ishigaki) [1][3].

**Architecture** [2]:
- Capture: Nintendo Switch → Elgato HD60 X → commentary server. Video arrives at **25 fps**.
- Frames are grouped into segments of **N = 32 frames** (about 1.28 s at 25 fps; my arithmetic),
  each sent as soon as it is complete.
- Vision-LLM: **GPT-4.1-mini**, given base64 frames. TTS follows Iura et al. 2025, an
  excitement-inducing commentary TTS for fighting games [2]. The paper does not name the model.
- Output: one audiovisual stream pushed to YouTube Live.

**When to speak.** Text generation **starts for every new segment, even while the current
utterance is still playing**. When playback ends, the system takes one buffered candidate and
synthesises it immediately [2]. The baselines show why this matters:
- *After-Audio*: the next generation waits for playback to finish.
- *After-Text*: the next generation starts when text generation finishes.

**Buffering.** Three selection policies: **Latest**, **Oldest** and **Random**. The authors chose
them "to avoid additional decision latency" [2]. The paper does not give a buffer size or an
explicit staleness rule [2].

**Video-delay control.** Video is buffered and starts playing only when the first utterance
starts. This absorbs the end-to-end generation latency with a **one-time fixed delay** [2].

**Numbers** (Table 1, `max_new_tokens = 20`; units are not stated in the table, and the abstract
gives mean silence in seconds) [2]:

| Method | Cumulative silence | Mean silence | Speaking/silence mIoU vs human |
|---|---|---|---|
| Human | 59.7 ± 11.3 | **1.7 ± 0.4** | – |
| After-Audio | 134.0 ± 5.2 | 9.5 ± 0.9 | 0.01 |
| After-Text | 125.5 ± 5.8 | 6.8 ± 0.9 | 0.10 |
| Latest | 24.7 ± 6.6 | 0.4 ± 0.1 | 0.59 |
| Oldest | 18.9 ± 3.1 | 0.3 ± 0.1 | 0.60 |
| Random | 19.1 ± 3.6 | 0.3 ± 0.1 | 0.60 |

Utterance length (Table 2). Human mean speaking time is **2.8 ± 0.5** with text length 25.9.
`max_new_tokens = 40` best matched human speaking duration (2.9). A value of 60 best matched human
text length [2]. The abstract quotes 9.6 s → 0.3 s and "over 40%" better timing similarity. That
40% figure cannot be traced directly to the tables [1][2].

Evaluation: 8 Super Smash Bros. Ultimate videos from the SMASH corpus. Timing similarity is mIoU over
1 Hz speaking/silence sequences. The user study had 120 crowdworkers rating 30 s clips [2].

**Lessons for CourtVision**
- Run generation in parallel with playback and keep a candidate buffer. This is the single biggest
  lever against silence.
- The paper's policies ignore freshness. Basketball changes state every few seconds, so
  CourtVision needs per-candidate **TTLs** and **state-version invalidation** (section 9.4). The
  paper itself notes that differences between policies were small [2].
- Its silences were shorter than human ones (0.3 s vs 1.7 s). Enforce a **minimum human-like gap**
  rather than minimising silence.
- A fixed video delay is a good trick for streamed output. It cannot help in-arena or on the bench.

### 1.2 Mori et al., "Live Football Commentary System Providing Background Information" (ACL 2025 System Demos, pp. 394-404)

Institute of Science Tokyo and AIST [14]. The pipeline has three modules: video analysis, spotting
(when to speak plus utterance type) and generation.

- **Video analysis.** Player ID uses Somers et al. 2024 (multi-object tracking, team
  classification, jersey OCR). Jersey number maps to name through the jersey-to-name mapping that
  SoccerNet-Caption provides (in effect a roster table). Ball candidates come
  from YOLOv8 at a low confidence threshold of **0.3**. A Dijkstra smoothing pass builds the
  trajectory, and gaps up to **25 frames (1 s)** are linearly interpolated. Events (pass, drive,
  out, ...) come from **T-DEED**, the winning model of the 2024 Ball Action Spotting challenge [14].
- **Play-by-play.** A template fills (action, player, team, ball zone), for example "Aubameyang
  passed from Left bottom corner". **If the jersey is not recognised, the template falls back to
  team level: "An Arsenal player passed..."** GPT-4o then restyles the template into commentary [14].
- **When to speak.** The gap after the previous utterance is sampled from the empirical
  distribution of inter-utterance gaps, with gaps **over 4 s excluded**. MSE was **15.50** versus
  **19.86** for a fixed **2.14 s** gap, the mean silence between utterances in their data [14].
- **Play-by-play or colour.** About **18%** of utterances contain background information. The rate
  rises after out-of-play events, measured within 15 s after the event: goal 32.1%, penalty 46.0%,
  yellow card 31.5%, offside 26.9%, foul 24.4% (Table 6). Across all the paper's out-of-play events
  the after-event range is 21.3% (ball out of play) to 46.0% (penalty) [14]. Rule: if an out-of-play
  event happened in the previous 15 s, choose background with p = 0.5, otherwise p = 0.18. Within
  15 s before or after kick-off, always choose background. **F1 was only 22.7% (predicted timing) /
  22.9% (gold timing)**, so this rule is weak [14].
- **RAG.** 3,257 Wikipedia player pages. Chunks are 1,000 characters with 100 overlap, embedded with
  `text-embedding-ada-002`. Retrieval returns the top 10 chunks with cosine above 0.7. The query
  holds match info, players visible in the previous 2 s, commentary from the previous 60 s and events
  from the previous 15 s. Generation uses GPT-4o, and the prompt says "Do not use information dated
  after this [game date]" [14].
- **Not real time.** Offline video analysis took 5:55 on a TITAN 24 GB, of which jersey OCR was
  3:54. The online stage took 27 s for a 30 s clip: 10 s retrieval plus GPT-4o and 17 s TTS (OpenAI
  TTS). Speech duration assumed **200 words per minute** [14].
- **Human evaluation.** Using ground-truth players raised relevance from 1.73 to 2.05. Adding
  ground-truth documents raised it to 2.55. Player-ID accuracy drives quality [14]. The factual
  accuracy of the background facts was **not measured** [14].

**Lessons:** use a team-level fallback when identity confidence is low. Concentrate colour in
dead-ball windows. Keep the colour share around 20%. Timing should not be a fixed metronome. RAG
facts need a date cutoff and a validator.

### 1.3 LiveCC (CVPR 2025, arXiv 2504.16030)

- LiveCC-7B starts from **Qwen2-VL-7B-Base**. It is trained on ASR words densely interleaved with
  frames at **2 FPS**, one-second interleave windows [5]. Datasets: Live-CC-5M (pre-training) and
  Live-WhisperX-526K (SFT). Checkpoints: `chenjoya/LiveCC-7B-Base` and `chenjoya/LiveCC-7B-Instruct`
  [4][6].
- Latency: **0.17 s per frame** in streaming mode, versus 5.62 s per clip for LLaVA-Video-7B [5]. The
  hardware is not stated. Evaluation used 8× 80 GB GPUs [6].
- LiveSports-3K CC track: 1,702 events, 416 videos, 49 sports, judged by GPT-4o win rate. LiveCC-7B-
  Instruct scores 41.5, above Qwen2.5-VL-72B at 30.4 [5].
- The paper does not evaluate hallucination [5]. The repo needs `flash-attn`, which is CUDA-only
  (common knowledge), and documents no licence on its front page [6]. The Hugging Face card for
  `chenjoya/LiveCC-7B-Instruct` lists Apache-2.0, 8B parameters, BF16, and loads the model with
  `attn_implementation='flash_attention_2'` [62].
- **For CourtVision:** not usable in the hot path on an 18 GB M3 Pro. At 8B parameters in bf16 the
  weights alone are about 16 GB (my arithmetic from [62]), and the model cannot ground names. At most, it is a research
  baseline for "style".

### 1.4 MatchTime (EMNLP 2024) and SoccerNet-Caption / SoccerNet-Echoes

- **SoccerNet-Caption** (CVPRW 2023): 36,894 timestamped commentaries over 715.9 h. The baseline
  scores 5.9% BLEU@4 (unverified: not found in the text extracted on re-check). The authors say
  commentary needs "rich factual" content, and their failure analysis shows the baseline's
  generated scores after a goal are "almost never accurate" [9].
- **SoccerNet-Echoes** (2024): Whisper ASR transcripts of SoccerNet broadcast commentary, translated
  with Google Translate, for spotting, captioning and summarisation [10].
- **MatchTime** [7][8]. The original caption timestamps are misaligned by a mean absolute **16.63 s**.
  Only 26.29% fall within 10 s of the key frame. The authors' WhisperX + LLaMA-3 + CLIP alignment
  pipeline reduces the error to 6.89 s, with 80.73% within 10 s. MatchVoice is a frozen visual
  encoder plus a Perceiver aggregator (32 queries) plus a LLaMA-3 decoder. The paper does not state
  a LoRA rank for the headline model: its ablation tries ranks 8/16/32/64 (16 is best on BLEU and
  METEOR, 32 on CIDEr), and the "Frozen LLM" row matches the main results. A 30 s input window
  works best. The commentary is anonymous (the GPT-score prompt states names are masked as
  `[PLAYER]` / `[TEAM]`), and the authors state that the model cannot describe player
  information. It confuses corner kicks with free kicks [7].
- **Lesson:** commentary text is a poor timing label, and generation without identity is a dead end
  for factual narration. If we ever train on broadcast audio, alignment must be fixed first.

### 1.5 StreamSoccer (arXiv 2608.19723, Aug 2026)

- Event memory with an "activate-update-close-consolidate-retrieve" lifecycle, 9 event tokens per
  clip, and a frozen Qwen3-VL-8B-Instruct with LoRA [11].
- **The scheduler is deterministic and rule-assisted.** It chooses silence, current-event,
  recent-window or historical-memory commentary with fixed priority **history >> event >> recent**
  [11]. A 24 s maximum event duration forces closure, which happens to 40.2% of events [11].
- Real-time factor p95 is **0.10-0.22 at 2 FPS** with no growth over a 90-minute match. Competitors
  degrade: TimeChat-Online hits its context limit at about 55 minutes [11].
- In the qualitative cases, StreamingVLM **repeats a goal claim** and TimeChat-Online contradicts the
  match time and score [11]. Unbounded-context streaming models drift. The authors add that metric
  scores "do not establish factual correctness" [11].
- **Lesson:** use bounded, stateless LLM calls over a structured state rather than one ever-growing
  conversation. Latency stays flat and the model cannot re-narrate old events.

### 1.6 NBA_Streaming (arXiv 2608.09200, Aug-Sep 2026), the only basketball benchmark found

- 152 full games from the **2025-26 NBA** season, 307.5 h, about 35K events, 50.4K commentary
  sentences, 539 player identities, 15 event types (`2pt_made`, `3pt_missed_rebound`,
  `putback_chain`, `shot_foul_chain`, `block_chain`, ...), and 71 fine-grained actions [12][13].
- **Average commentary sentence: 13.2 words** [13].
- The method is "completion-first" localisation. An event-ending detector fires at threshold 0.5,
  then 37 end-anchored candidate windows (2-20 s, in 0.5 s steps) are scored, using 6 FPS context
  of about 22 s. Stage II grounds the event on the ball using the WASB ball detector, with
  Qwen3-VL-8B as generator [13].
- Best results: start/end boundary error **2.47 s / 2.09 s**, **identity accuracy 28.5%**, event
  accuracy 71.4%, action accuracy 47.1%. Baselines reach 0.0% (HAT + Qwen3-VL) to 4.5% (MATR +
  MatchTime) identity accuracy. The identity, event and action accuracies are computed **only on
  events whose start and end boundary errors are both within 2 s**, so they overstate accuracy on all
  events [13].
- **Lesson:** even purpose-built 2026 basketball systems get names wrong about 70% of the time from
  pixels. Event-end timing error of about 2 s is the state of the art for learned online
  localisation. CourtVision's rule-based confirmation (ball through rim, possession change) should
  beat this for core events. This needs to be measured.

### 1.7 MARIO: LLM sportscaster for RoboCup soccer (arXiv 2607.14809, Jul 2026)

Naming note: in the paper, MARIO ("Modular and Extensible Architecture for Computing Visual
Statistics") is the existing RoboCup statistics framework that the sportscaster extends. "MARIO" in
this note means that sportscaster system [15].

This is the closest published architecture to what we need [15]:
- Symbolic event extraction (velocity, possession, direction cosines). Goals fire **only on an OCR
  score change**.
- Reactive prompts carry machine-validated fields (event class, acting team, score) and must
  produce "one short, factual sentence without introducing unsupported entities or actions".
- Priority goal > shot/pass > minor > periodic. **Higher priority preempts queued lower priority.**
  Duplicates are suppressed. A cooldown between similar-priority events keeps the earlier one.
  Periodic commentary runs only in quiet windows. Stale ball positions and scores are discarded
  after a freshness threshold [15].
- It names no LLM or TTS and reports no per-stage latency [15].

### 1.8 Survey taxonomy

Zheng et al.'s AI-generated game commentary survey sorts commentary into Live Observation →
Descriptive, Strategic Analysis → Analytical, and Historical Recall → Background [16]. CourtVision
maps these to P1 calls, coach/colour analysis and dead-ball background respectively.

---

## 2. Deployed AI commentary: what exists and what it teaches

| Deployment | What it is | Factuality / process facts | Source |
|---|---|---|---|
| IBM × Masters (2023-2026) | AI narration on highlight clips (20,000+ clips, Spanish added in 2024). Hole Insights: outcome probabilities per shot | Few-shot prompting, no fine-tuning during the tournament (they change the **input statistics** instead). HAP filter. **Multiple LLMs validate the main model's output, plus regex guardrails.** ROUGE/METEOR/perplexity gates before promotion. Re-validation after each round with ground truth. Response time measured because the system is "near real-time" | [17][19] |
| IBM × US Open 2024 | Granite 13B. AI Commentary audio + subtitles on highlights "just minutes after a match concludes". Match Reports for 254 singles matches | Match reports reviewed and edited by USTA editors before publishing | [18] |
| IBM × Wimbledon 2023 | Highlight narration from a foundation model trained on "the language of tennis" | Details (e.g., 3B parameters) only from secondary coverage (unverified) | [20] |
| WSC Sports × NBA (Apr 2025) | AI voice-over on 2-minute game highlights in French, Portuguese and Spanish, posted "within minutes of final whistle" | Pipeline: "large sports model" analysis → AI script with stats → sports-tuned TTS. 75% average view completion. No human review mentioned | [21] |
| NBA × AWS "Inside the Game" (Oct 2025) | Real-time Defensive Box Score, Shot Difficulty (29 body points) and Gravity, for teams, broadcasters and fans. Broadcasters get **alerts for historic or unusual plays** | Positioned as commentator *augmentation*. In-game on-air use described as "eventually" | [22] |
| Prime Video NBA (2025-26, 2026-27) | "Prime Insights" AI features, catch-up highlight reels (≤ 2 min), Prime 360 replays (unverified on re-check: the source returned HTTP 403) | No automated play-by-play voice reported | [26] |
| Stats Perform "Opta Stream" × Urban Edge (Dec 2024) | "AI-powered, automated, in-depth, play-by-play updates" for every SWAC/MEAC D1 HBCU basketball game | Text vs audio not stated. Opta editorial services involved | [23] |
| NBC "Your Daily Olympic Recap" (2024) | LLM script + AI Al Michaels voice, about 7M personalised variants (unverified on re-check: the source returned HTTP 451) | **Editors reviewed all content, audio included, before release** (unverified on re-check) | [24] |
| ESPN AI recaps (NWSL / PLL, 2024) | Templated or LLM game recaps | Alex Morgan's final match recap (215 words, unverified) **never mentioned her**: factually correct but blind to salience. ESPN updated it to mention her at 8:59 a.m. ET the next day, Sept 9, 2024 (AI Incident DB #785) | [25] |

**Finding:** I found **no deployment of fully automated, live audio play-by-play** on NBA, WNBA
or NCAA broadcasts as of Oct 2026. Deployed generative commentary is post-event (highlights,
recaps), text-only live updates (Opta Stream), or commentator augmentation (AWS alerts). This
reflects the searches I ran, not proof that none exist.

**Production lessons that carry over:**
1. Inputs are structured stats, not pixels [17][21][23].
2. Use layered validators: regex/HAP checks, secondary-model checks, metric gates [17].
3. Where latency allows, a human signs off on recaps [18][24]. CourtVision live narration cannot
   have that, so deterministic templates for core calls stand in for the human.
4. **Salience is a correctness property.** The ESPN miss is why the FactPacket carries
   `storylines` (milestones, farewells, records flagged by code) and why a Jev `Noul` checks
   whether a storyline is due (section 9.3).
5. Broadcasters want **"this is unusual"** alerts [22]. A cheap code-side milestone and outlier
   detector is high-value content for both channels.

---

## 3. Keeping narration factual: architecture patterns

### 3.1 The structured feed rule

Every system that can name players correctly gets names from a symbolic layer: jersey OCR plus
roster [14], GameController and OCR [15], editorial data [17][18]. CourtVision's rule: **the fact
store is the only source of names, numbers, clock and score**. The LLM gets a FactPacket and nothing
else. It never gets frames, and never gets free-text history beyond the last few utterances.

### 3.2 FactPacket (design proposal)

```python
@dataclass(frozen=True)
class FactPacket:
    packet_id: str
    state_version: int              # fact-store version; invalidates stale candidates
    game_clock: str                 # "Q4 1:52" (pre-formatted; LLM must not do clock math)
    score: dict[str, int]           # {"BOS": 98, "MIA": 95}
    event: dict | None              # the triggering confirmed event (type, actor_id, value, conf)
    entities: dict[str, dict]       # id -> {"name": "Jayson Tatum", "short": "Tatum",
                                    #        "aliases": ["JT"], "team": "BOS", "on_court": True}
    allowed_numbers: frozenset[int | float]   # every number the line may say (score, stats,
                                    # margin, run, clock components, jersey numbers in packet)
    derived: dict[str, str]         # code-computed phrases: {"margin": "three-point game",
                                    #  "run": "8-0 Boston run", "fouls": "Adebayo has 4"}
    storylines: list[str]           # code-flagged salience: "Tatum 2 pts from 30", "milestone"
    recent_utterances: list[str]    # last 6 spoken lines (for anti-repetition)
    utterance_type: str             # "color" | "summary" | "coach_digest" | ...
    max_words: int
    forbidden: list[str]            # e.g. ["career-high", "record", "first time"] unless flagged
```

All arithmetic (margins, runs, percentages, foul counts) happens in Python. The sibling Jev note
gives the same rule for Jev ("Jev is not a calculator") [61], and it applies equally to the LLM.

### 3.3 Validator (runs on every LLM sentence before TTS; target under 5 ms)

1. **Entities.** Extract capitalised spans and alias hits, and resolve them against
   `entities`. Any person or team name not in the packet means **reject**. A player with
   `on_court=False` described doing a live action means reject.
2. **Numbers.** Extract digits and number words ("twenty-one" → 21, "three" in "for three" maps to
   the shot value). Every number must be in `allowed_numbers`, or a jersey number or period number
   from the packet. Otherwise reject.
3. **Claims lexicon.** Superlatives and records ("career-high", "record", "first", "never",
   "all-time", "franchise") are allowed only if a matching storyline flag exists.
4. **Tense and recency.** No event verb about a player unless a packet event or a recent-event entry
   supports it. Block future-as-fact phrasing ("will win").
5. **Score consistency.** Any "X leads/trails by N" must match `score`.
6. **Length and form.** Word count must be ≤ `max_words`, and the text must end in terminal
   punctuation. A missing terminal mark suggests `max_tokens` truncation.
7. **Repetition.** Word-trigram Jaccard ≥ 0.5 against any of the last 10 utterances means reject.
   The same (fact_type, subject, value) already spoken within its cooldown also means reject
   (section 9.5).
8. **On failure:** if TTL remains above the p90 LLM latency, retry once with the failure reason
   appended. Otherwise fall back to the template for that slot, or stay silent. Log every rejection.
   The reject rate per rule is a key metric.

IBM's production stack is the deployed precedent for regex plus secondary checks [17]. MARIO's prompt
constraint ("without introducing unsupported entities or actions") goes into our system prompt
too [15]. The validator is what enforces it.

### 3.4 Templated fallbacks

- Each P0-P2 event type has **8 or more variants**. Selection is least-recently-used, and none of
  the last 3 used variants for that type may repeat.
- Use **identity-degradation tiers**. If shooter ID confidence is ≥ 0.95, say "Tatum". At 0.80-0.95,
  say "Boston's number 0" (only if the jersey OCR itself is ≥ 0.95). Otherwise say "Boston" or "the
  Celtics". This mirrors Mori et al.'s "An Arsenal player..." fallback [14].
- Templates can carry a **prosody tag** (calm / up / big) chosen by code (clutch, margin) or by a
  Jev `Score` on excitement level, off the hot path. The TTS layer maps the tag to rate and pitch.

### 3.5 Confidence gating

- **Live calls (P0/P1):** gate on CV-derived confidences only, such as make probability from rim
  geometry and track continuity, and ID confidence from OCR. Jev's ~0.4 s median [61] does not fit a
  sub-second call.
- **Dead-ball decisions:** use Jev. Ask a `Choice` for utterance type or topic (`stat_note` /
  `tendency` / `storyline` / `silence`), a `Noul` for "is this a clutch moment" or "is a storyline
  due" (thresholds of about 0.7-0.8, as the Jev note advises for narration-flavour judgments [61]),
  and a `Score` for excitement level. Batch all of them into one `system_one` call per dead-ball
  window [61].
- Use **hedge bands** rather than binary gates. Between the "say it" and "silence" thresholds, the
  director either waits up to the event's max hold for more evidence, or uses hedged wording ("looks
  like it's good", "they'll check that one"). Cap hedges at 1 per 3 utterances so the voice does
  not sound unsure all the time.

### 3.6 Correction ("scratch that") strategy (design proposal)

Basketball revisions are routine. The NBA reviews 2- vs 3-point scoring (trigger 6), shot-clock
violations (trigger 7), out-of-bounds (via the coach's challenge, trigger 15), goaltending/basket
interference (trigger 12, last 2:00 of Q4/OT only) and the correct free-throw shooter (trigger 9)
by replay [60]. CV will also
revise its own attributions.

| Case | Action |
|---|---|
| Revision arrives **before** the candidate started speaking | Silently re-render or drop. Candidates carry `state_version`, and a mismatch on any referenced fact invalidates them |
| Revision arrives **while** the wrong fact is being spoken | Cut at the next word boundary if the wrong token has not been voiced yet, then speak the correct template. Otherwise finish the word and queue a P0 correction. **How depends on the AVSpeech path:** with `speak_()` playback, use `stopSpeakingAtBoundary_(AVSpeechBoundaryWord)` (Swift `stopSpeaking(at: .word)`) and `speechSynthesizer:willSpeakRangeOfSpeechString:utterance:` for progress. With the recommended `writeUtterance_toBufferCallback_` + own mixer, those callbacks fire at **render** time (all 5 word ranges of a 2.07 s line arrived within 6 ms of each other, before playback) [M], so record the rendered frame count at each range callback as that word's sample offset, and cut in the mixer at the next offset after the playback cursor |
| **Material** fact already spoken (scorer, points value, score, who fouled, foul-out, timeout team) | P0 correction in the next gap, ≤ 10 s TTL, ≤ 12 words: "Correction: that bucket goes to Brown, not Tatum." "That's been changed to a three, Boston up five." |
| **Non-material** fact (assist credit, rebound credit, shot distance) | No audio correction. Fix the stats silently and use the correct value next time it is referenced |
| Replay review signalled (whistle + monitor, or officials' feed) | Narrate a hedge ("They're going to take a look"). Hold score-dependent statements until the review closes. Narrate the outcome as P1 |
| Correction storm (> 2 audio corrections in 5 game minutes) | Auto-raise the P1 confidence gates by +0.03 and widen holds by +200 ms for the rest of the period. Log for tuning |

Never correct the same event twice. If the second revision reverts to the original, say nothing
unless the reversal itself is material.

---

## 4. Turn-taking and pacing

### 4.1 Human rhythm numbers

| Quantity | Value | Source |
|---|---|---|
| Mean silence between utterances (skilled game commentators) | **1.7 ± 0.4 s** | [2] |
| Mean speaking time per utterance | **2.8 ± 0.5 s** | [2] |
| Mean silence between utterances (soccer broadcast, LFCBI) | **2.14 s** | [14] |
| Natural gap upper bound used for timing models | gaps > **4 s** excluded as unnatural | [14] |
| Words per commentary sentence (NBA 2025-26 broadcasts) | **13.2** | [13] |
| Share of utterances that are background/colour (soccer) | **18%** overall; 21-46% within 15 s after out-of-play events (foul 24.4%, goal 32.1%, penalty 46.0%) | [14] |
| Commentator articulation rate (Euro 2012 TV, Ukrainian) | 6.8-7.5 syllables/s average, peaks around 9 | [47] |
| Typical English speaking rate | about 4 syllables/s | [48] |
| Speech-rate assumption in a published commentary system | 200 wpm | [14] |
| macOS default voice (Samantha), measured | **3.5-3.7 words/s (≈ 210-220 wpm)**: 21 words → 5.99 s, 7 words → 1.91 s | [M] |

At about 3.5 words/s, an 8-word live call takes about 2.3 s, and a 13-word sentence about 3.7 s.
NBA teams average roughly 100 possessions each per 48 minutes (pace 98.5-99.4 in recent seasons per
the sibling note `basketball_rules_stats.md`), so with both teams about 200 possessions share 2,880 s
and a new possession starts about every 14 s of game clock. **Live-ball utterances must finish well inside
one possession.** That is the reason for the 4-10 word caps in section 9.2.

### 4.2 Pacing rules

- **Live ball:** speak on events only. Silence is fine. Never fill live play with colour.
  Commentators deliver background mainly in stoppages [14].
- **Dead ball:** a free throw, timeout, review, quarter break or substitution window ≥ 4 s opens P3
  colour. Target 1-2 colour lines per window, with the first starting 1.0-2.0 s after the stoppage
  call.
- **Inter-utterance gap:** at least 0.6 s live, target 1.0-2.0 s (human 1.7-2.1 s [2][14]). Sample
  each gap from a clipped distribution (Mori-style [14]) rather than a constant, to avoid a robotic
  cadence.
- **Preemption:** a P0 cuts the current line at the next word boundary. A P1 cuts P2/P3 at the
  next word boundary and never cuts another P1. Instead, the newer P1 replaces any *queued* P1 for
  the same possession (latest wins). MARIO preempts queued output the same way [15].
- **Anticipatory calls:** humans start a call before the outcome ("Curry, for three...") and finish
  it on the result. CourtVision can speak a 2-4 word **release call** when a shot attempt is detected
  with high confidence, then the result call after confirmation. This hides the confirmation hold
  inside natural speech.
- **Repetition:** see the cooldowns in section 9.5. StreamingVLM repeating a goal claim shows the
  failure mode of models without a "spoken facts" ledger [11].

---

## 5. Latency engineering

### 5.1 Budget: event occurrence → first audio (design targets, to be measured)

| Stage | P1 live call (template) | P3 dead-ball colour (Claude) | Notes |
|---|---|---|---|
| Capture + decode (30 fps) | 33-70 ms | same | about 1-2 frames |
| Detection + tracking + ball/rim | 30-80 ms | same | M3 Pro MPS, model-dependent (unverified, measure) |
| Event engine logic | ≤ 10 ms | same | — |
| **Confirmation hold** | 150-500 ms (shot/possession); 0.5 s generic foul | n/a (the trigger is the dead ball itself) | Section 9.3. The release call hides it |
| Director + template render + validator | ≤ 5 ms | ≤ 5 ms per sentence | — |
| Typesafe Jev decision | **not on path** | ~0.4 s median, run concurrently with the LLM call or at dead-ball start | [61] |
| Claude Opus 5.5 TTFT | n/a | **measure** with the §11 harness. Do not plan on 1.5 s p50 / 3 s p95 (unverified and superseded, RESEARCH.md D6); deadlines come from the measured p95 | AA benchmark: 6.52 s time to first answer token at `low` on Anthropic's API with AA's default 10,000-input-token workload (Google 7.20 s, Azure 8.62 s, Amazon 11.09 s) [28]. A search snapshot of a sibling AA page showed 12.35 s at `low` (unverified), so figures drift by date and workload |
| Claude output speed | n/a | ~80 tokens/s; 30 words ≈ 40 tokens ≈ 0.5 s | [28] |
| TTS first audio | **12-26 ms** warm (AVSpeech) [M] | same | Cold first call 0.6-1.0 s (612 / 708 / 982 ms in three processes): warm up at boot |
| CoreAudio output buffer | ~10-20 ms (unverified) | same | — |
| **Total** | **≈ 0.3-0.7 s** after the event resolves | **≈ 1.6-3.5 s** after the stoppage. **~0.05 s** when a pre-generated candidate is used | — |

Comparison points: learned online event localisation on NBA video has about 2.1 s end-boundary
error [13]. Sequential generate-then-speak pipelines leave about 9.5 s gaps [2]. A non-real-time
soccer pipeline needed 27 s per 30 s clip [14].

### 5.2 Claude-specific latency levers (from Anthropic's SDK reference [27])

- `claude-opus-5-5` **always thinks**. `thinking: {"type": "disabled"}` and `budget_tokens` return a
  400. **Effort is the latency control.** The default is `medium`, so set
  `output_config={"effort": "low"}` explicitly [27].
- Add a system-prompt line such as "Answer directly without deliberating." (documented in the
  Opus 5.5 migration guide as reducing thinking, TTFT and cost) or "Latency-sensitive; begin your
  visible answer immediately." (documented for Claude Opus 5, whose prompting patterns the guide
  says carry over to 5.5). Anthropic's prompt-audit guidance says to keep such a line only where a
  measurement on the latency-sensitive route shows it helps, since effort is the reliable control.
  Measure quality [27].
- **Thinking counts toward `max_tokens`.** Do not set `max_tokens` to the word cap: use 1024-2048
  and enforce length with the prompt and validator [27]. **Decision: 1536** (RESEARCH.md D13). This
  supersedes the `max_tokens=400` example once shown in `sdk_ground_truth.md`.
- **Prompt caching:** the minimum cacheable prefix on Opus 5.5 is **512 tokens**. Cache reads cost
  $0.20/MTok (0.05× input). The default TTL is 5 minutes, and `"ttl": "1h"` is available (1-hour
  writes cost 2× input, 5-minute writes 1.25×). **Pre-warm** the cache at tip-off with a
  `max_tokens: 0` request. That request must be **non-streaming** (`messages.create`, not
  `.stream`: `max_tokens: 0` with `stream: true` is an `invalid_request_error`), and it must send the
  same `system` blocks and `output_config.effort` as real traffic, since thinking/effort settings are
  part of the cached prefix. Whether `max_tokens: 0` combines with the `fallbacks` beta is not
  documented (unverified), so check `cache_read_input_tokens` on the first real call. Keep the system
  prompt and roster block byte-stable, and put volatile packet data after the breakpoint [27].
- **Fast mode** (`client.beta.messages...`, `betas=["fast-mode-2026-02-01"]`, `speed="fast"`):
  up to **2.5× output tokens/s** at $8/$40 per MTok, Claude API only. It has its own rate limit, and
  switching `speed` invalidates the prompt cache. It helps long summaries more than TTFT [27].
- Streaming: `AsyncAnthropic().messages.stream(...)` with `async for text in stream.text_stream`
  [27]. Cancel the stream (exit the context) when the candidate's TTL expires or it is preempted.
- Prefill is not supported. Use the system prompt and validator for format [27].
- Stateless calls with a bounded packet keep TTFT flat. StreamSoccer's results argue against an
  ever-growing session [11].
- The Claude Haiku 5.5 sibling (`claude-haiku-5-5`, $0.10/$0.50 for prompts ≤ 100K tokens, $0.50/$2.50
  above) accepts disabled thinking at effort `high` or below, and has **no server-side refusal
  fallback** (do not send `fallbacks` to it) [27]. Artificial Analysis lists it on Anthropic's API
  at **178.6 output tok/s and 13.84 s time to first answer token at `medium`**, and 171.2 tok/s with
  8.39 s TTFT at `low` [29]. Those benchmark TTFTs are *not* better than Opus 5.5 at `low` (6.52 s),
  so Haiku is not an automatic latency win with thinking on. It is a candidate to **benchmark**
  (including with thinking disabled) for P2 short lines if Opus TTFT proves too high. The project
  default stays `claude-opus-5-5`.

### 5.3 Streaming LLM output into streaming TTS

- **Sentence-gated streaming.** Accumulate Claude deltas until a sentence boundary (`.`, `!`, `?`,
  or `;` / `—` after at least 6 words). Validate that sentence, then hand it to TTS while the next
  sentence generates. Colour lines are 1-2 sentences, so first audio lands at about TTFT + first
  sentence generation (≈ 12-15 words at ~80 tok/s ≈ 0.25 s) + validation + TTS first buffer.
- **Do not stream token-by-token into TTS.** It defeats the validator, and local engines prosody
  better on whole clauses.
- **ElevenLabs WebSocket buffering.** `chunk_length_schedule` defaults to `[120, 160, 250, 290]`
  characters before each generation. A 60-character commentary line would **wait for `flush`**, so
  always send `flush: true` at the end of each line, or lower the schedule. The allowed range is
  50-500 [31].
- **OpenAI TTS:** request `response_format="pcm"` (24 kHz, 16-bit) or `wav` for the lowest latency
  [32].
- **Video delay for streamed output.** For a streamed overlay or alt-cast, delay outbound video by a
  fixed D ≈ p95 of event-to-P1-audio, roughly 0.8-1.0 s, so calls land in sync. This is Kawamatsu et
  al.'s fixed-delay trick [2]. In-arena and bench outputs cannot be delayed.

### 5.4 Speculative pre-generation (adapted from [2])

- At each **dead-ball start** (whistle confirmed, or clock stopped with no FT pending), fire
  **2 Claude requests in parallel** with different `utterance_type`/topic hints (for example
  `stat_note` and `tendency`), alongside one Jev call for type/salience.
- Results go into the candidate buffer with a `state_version` and a TTL. At the next playback
  boundary, the director takes the highest-priority fresh candidate, using the Jev `Choice` answer
  when it is available and freshest-first otherwise.
- During long stoppages (timeouts, quarter breaks), keep a rolling buffer of 2 candidates.
- Cost check: about 1.5k input tokens × 2 per stoppage × ~60 stoppages per game is about 0.18M
  input tokens, around $0.04 cached or $0.72 uncached per game at $4/MTok (my arithmetic). This
  excludes cache writes (1.25× or 2× input) and **output tokens, thinking included, at $20/MTok**,
  which with always-on thinking can be of the same order as the input cost. Measure
  `usage.output_tokens` per call.

---

## 6. TTS options usable from Python on macOS (facts only)

| Engine | Licence | Latency / speed | Integration facts | Notes |
|---|---|---|---|---|
| **macOS `AVSpeechSynthesizer`** (via PyObjC `pyobjc-framework-AVFoundation` 12.2.2, MIT) | Bundled with the OS | **First buffer 12-26 ms warm, 0.6-1.0 s cold. 6 s line fully rendered in 67-75 ms (~80-90× real time). 22,050 Hz** [M] | `writeUtterance_toBufferCallback_` gives PCM buffers for our own mixer. `stopSpeakingAtBoundary_(AVSpeechBoundaryWord / AVSpeechBoundaryImmediate)` (Swift `stopSpeaking(at: .word/.immediate)`) acts on the synthesizer's own `speak_()` playback. Delegate `speechSynthesizer:willSpeakRangeOfSpeechString:utterance:` also fires on the write path, but at render time, so map each range to the rendered frame count for word-boundary cuts in our mixer [M]. SSML init `speechUtteranceWithSSMLRepresentation_` (macOS 13+), Personal Voice (macOS 14+) (selectors confirmed present in PyObjC 12.2.2 [M]; OS versions from Apple docs, page text not retrievable [45]) | Only default-quality voices installed here (180 voices, all quality = 1) [M]. Enhanced/premium voices need a download in System Settings (unverified UI path). Needs an NSRunLoop on its thread |
| macOS `say` CLI | Bundled | **0.78-2.81 s per invocation** for a 0.86 s clip, across two measurement runs (process spawn + voice load) [M] | `-v voice`, `-r` rate in words per minute, `-o` file [M, man page] | Do not use per utterance. Fine for offline tests |
| **Kokoro-82M** (hexgrad) | **Apache-2.0** weights and code [35][36] | No official MPS benchmark. A community report gives ~11× real time on an **M3 Max CPU** with `KPipeline` (unverified) [37] | `pip install kokoro>=0.9.4 soundfile`. `KPipeline(lang_code='a')`, a generator yielding `(graphemes, phonemes, audio)` at 24 kHz. `PYTORCH_ENABLE_MPS_FALLBACK=1` for MPS. Needs **espeak-ng** for out-of-dictionary words. G2P via `misaki` [36]. 54 voices, v1.0 (Jan 2025) [35] | espeak-ng is GPL-3.0-or-later [63]: keep it a separate process or binary if licensing matters. MLX port: `mlx-community/Kokoro-82M-bf16` via `mlx-audio` (MIT) [38]. The same community report [37] used the PyTorch CPU path as a workaround because its MLX attempts (`kokoro-tts-mlx`, `mlx-audio`) failed, so verify the MLX path works before relying on it |
| **Piper** (`piper-tts`) | **GPL-3.0-or-later** (v1.8.0, 2026-09-04). The original `rhasspy/piper` was MIT and archived 2025-10-06 [39][40] | Fast VITS on CPU (no number verified) | macOS arm64 wheels (`macosx_11_0_arm64`), Python ≥ 3.9 [39] | GPL licence. Individual voice licences vary (unverified) |
| **Kyutai Pocket TTS** | MIT code. Weights licence not stated on the README [41] | **~200 ms to first chunk. ~6× real time on a MacBook Air M4 CPU** [41] | `pip install pocket-tts`, `TTSModel.load_model()`, `generate_audio(voice_state, text)`. 26 voices plus cloning. EN, FR, DE, PT, IT, ES, NL [41] | CPU-only fits beside MPS-heavy CV |
| Kyutai TTS 1.6B en_fr | **CC-BY-4.0** weights [43] | **220 ms** from first text token to first audio (official, GPU; unverified on re-check, the page rendered empty). Accepts **streaming text input** [42]. 1.28 s audio lag behind text (16 steps) [43] | PyTorch `moshi`, CUDA example. No MLX mentioned on the card [43] | The card says it is actually **1.8B** parameters (1B backbone + 600M depth transformer) [43]. Probably too heavy beside CV on 18 GB (unverified) |
| Sesame **CSM-1B** | Apache-2.0, gated on HF. "Research and educational" intended use [44] | No numbers. Speed-up path is CUDA graphs [44] | `transformers>=4.52.1`, `CsmForConditionalGeneration`. Llama backbone + Mimi codes. No preset voices. English-centric [44]. MLX: `mlx-community/csm-1b` [38] | Not a low-latency default on a Mac |
| **ElevenLabs** | Commercial API | `eleven_flash_v2_5`: **~75 ms** model latency (excludes network/app), 32 languages, 40k characters. `eleven_v4_turbo`: ~100 ms median inference, 90+ languages. `eleven_v3_conversational`: ~280 ms. `turbo_v2_5` deprecated [30] | WebSocket `wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input`. `chunk_length_schedule` default [120, 160, 250, 290]. `flush` [31] | Best cloud quality and latency combination found |
| **OpenAI** | Commercial API | `gpt-4o-mini-tts` recommended for realtime. `tts-1` lower latency, lower quality [32] | Chunked streaming. `pcm` 24 kHz s16le. `instructions` style prompt. 13 voices (`marin`, `cedar` best) [32] | — |
| **Cartesia** | Commercial API | `sonic-3.6` (snapshot `sonic-3.6-2026-08-27`), 44 languages. No latency number in the docs [33]. Third-party: vendor 40-90 ms, measured ~166-190 ms end-to-end [34] | WebSocket streaming | — |
| `mlx-audio` | MIT (repo). Model weights vary | No published latency numbers [38] | Runs Kokoro, KittenTTS, Qwen3-TTS, Chatterbox, CSM, Dia, Voxtral TTS, and MeloTTS (streaming) on Apple Silicon. `--stream` CLI [38] | Good hedge if PyTorch on MPS is slow |

**Recommendation for the local default:** `AVSpeechSynthesizer`. It is the lowest measured latency,
needs no model memory (leaving the GPU and RAM to CV) and supports word-boundary interruption.
Behind the same `TTSEngine` interface, benchmark Kokoro-82M (MPS, and MLX via `mlx-audio`) and
Pocket TTS on the M3 Pro **while the CV pipeline is running**, and switch if p95 first audio stays
≤ 150 ms. ElevenLabs Flash v2.5 is the opt-in cloud voice.

---

## 7. Coach-facing vs fan-facing outputs

### 7.1 What is permitted and actually used on benches

- The **NCAA** allowed live statistics to be transmitted to the bench in men's basketball starting
  in 2021-22 [53] (not re-verified: the cited NCAA URL now redirects to its news index). Earlier
  waivers allowed bench tablets with real-time stats: Hall of Fame Classic 2018 (ShotTracker's
  page does not mention video) [55], and Mountain West 2019-20, conference games only, where video
  was explicitly **not** allowed [56].
- The **NBA** has allowed real-time video on the bench over hard-wired connections since 2016-17
  [54].
- What staffs used in those pilots: lineup efficiencies, "best lineups", shot trends, per-player
  shot charts [56], and **zone maps**. One example: a first-half map showed a forward's makes
  clustered to one side, so staff forced him left. Another: **pass-count splits**, with a 0.6
  points-per-possession difference between 4-6 pass and 0-3 pass possessions [55]. Evidence here is
  mostly vendor-side.
- Halftime takes **a few minutes** of stats plus a short video edit before the head coach speaks.
  Box scores are handed out at timeouts. Changes are "pit-stop" tweaks (a coverage change), not
  overhauls [57].

### 7.2 Real-time content coaches can act on (with the evidence)

| Item | What to show | Evidence / caveat |
|---|---|---|
| **Foul trouble** | Alert when a player reaches **Q+1** fouls (one more than the current quarter): "Adebayo 3 PF, Q2, Q+1" | Maymin, Maymin & Shen (2006-09 NBA play-by-play): starters "ought to be idled" on a Q+1 basis. The ESPN summary reports benching raised first-half win probability by roughly 5-10% "with few exceptions" (not late in games), partly because players in foul trouble play tentatively [49]. NCAA halves need an adapted rule (design proposal). The foul-out limit is league config, not a constant: **6** in the NBA, **5** in FIBA and NCAA (sibling note `basketball_rules_stats.md`) |
| Team fouls / bonus | Fouls to give, bonus status, timeouts remaining (both teams) | League-specific: configure from the rulebook (verify per league) |
| **Opponent run** | "MIA 9-0 run, 2:14, BOS 3 TOs left. Last 4 poss: 2 TOV, 2 open 3s" | Timeouts' causal effect is weak or disputed: none after matching, attributed to regression to the mean (NBA) [50]; "slightly disadvantageous" during an opposing run (NBA) [51]; in the EuroLeague 2021-22 to 2023-24, a short-run improvement in point differential from −3.74 to −1.20 per window (about 2.5 points less opponent scoring), still negative after the timeout and with no season-level effect [52]. **Present as information, not "call timeout now"** |
| Lineup +/- | In-game net rating **with possession count**. Suppress below ~20 possessions (design proposal) | Average NBA lineup sees only ~25-30 possessions per *season*, from 600+ lineups per team [59]. In-game lineup numbers are noise unless shrunk |
| **Shot quality** | Expected eFG (location + defender distance + shot type) vs actual, both teams: "MIA eFG 61% on xeFG 49%" (luck vs process) | The NBA/AWS "Shot Difficulty" metric uses 29 body points [22]. Our xeFG is a simpler proxy (unverified accuracy until calibrated) |
| Four Factors (halftime / timeouts) | eFG%, TOV%, ORB%, FT rate, both teams, with deltas | Oliver's weights are about 40/25/20/15, rough. Regressions put shooting at 44-54% and FT at 6-10% [58] |
| Opponent tendencies | Drive direction / handedness, zone hot spots, PnR coverage used, transition frequency, pass-count PPP | ShotTracker pilot examples [55] |
| Substitution prompts | Stint length, foul state, matchup on-ball, fatigue proxy (speed/sprint decline from tracking) | Show as information. No evidence found that automatic sub suggestions help (unverified) |
| Unusual-event alerts | "Opp 7/9 from corner 3, season 36%" | Mirrors the AWS broadcaster alerts for unusual plays [22] |

### 7.3 Fan vs coach channel contract (design proposal)

| Dimension | Fan narration | Coach / staff channel |
|---|---|---|
| Modality | Audio first (TTS), captions | **Visual cards** on a tablet or web view. Optional earpiece only at dead balls |
| Tone | Energetic, names, storylines | Flat, numeric, no adjectives |
| Length | 4-10 words live, 20-35 dead ball | **≤ 12 words per card.** At most 3 cards visible. Halftime digest ≤ 6 bullets |
| Timing | Live + dead ball | Cards update live. Push or audio **only at dead balls** (timeouts, FTs, quarter breaks), except Q+1 foul alerts, which post immediately as a card |
| Uncertainty | Hedge wording or silence | Always show sample size and confidence (e.g., "n = 14 poss") |
| LLM role | Colour lines (validated) | One-line digest at timeouts and halftime from the same FactPacket and validator. Never invents recommendations, only summarises code-computed facts |
| Cooldown | Section 9.5 | Same alert type ≤ 1 per 120 s game clock. ≤ 1 audio alert per dead ball |
| Jev use | Type/topic/salience choice | `Noul` "is this alert decision-relevant now" (≥ 0.8) to rank cards. `Choice` for the top-3 card ordering [61] |

---

## 8. Recommended CourtVision narration architecture

```
 CV (tracks, ball, rim, jersey OCR, scoreboard OCR, whistle audio)
   │  frames 30 fps
   ▼
 Event Engine ──► ProvisionalEvent(id, rev, type, actors, conf, t_occ)
   │                 │ confirmation holds (9.3)
   │                 ▼
   │           ConfirmedEvent / Revision(id, rev+1, diff)
   ▼
 Fact Store (versioned GameState: score, clock, lineups, box score, fouls, runs, storylines)
   │                                            ▲
   │ snapshots (FactPacket)                     │ revisions
   ▼                                            │
 Narration Director (asyncio, 50 Hz tick) ──────┘
   │   priority queue P0-P3, TTLs, cooldown ledger, spoken-facts ledger, speaking-state machine
   ├─► Template Renderer (P0-P2, ~0 ms)  ─────────────┐
   ├─► Speculative LLM pool (P3/summary: claude-opus-5-5, effort low, streaming)
   │        └─► sentence splitter ─► Validator ───────┤
   ├─► Typesafe Jev advisor (dead-ball type/salience/candidate choice; async, never blocking)
   ▼                                                   ▼
 TTS Engine (AVSpeechSynthesizer warm; write-to-buffer + word→sample-offset map) ─► Audio mixer (word-boundary cut) ─► out
 Coach Channel (cards from Fact Store + Jev ranking + optional validated LLM digest)
```

Process layout on the M3 Pro: run CV in its own process so the GIL and MPS work stay isolated.
Event engine, fact store and director share one asyncio process. TTS gets a dedicated thread with
an NSRunLoop, or a separate process for Kokoro. Use an IPC queue (e.g. `multiprocessing` /
shared memory) for events.

---

## 9. Concrete parameters (all design proposals unless cited; tune on labelled games)

### 9.1 Priority levels

| Level | Name | Examples | Preempts | TTL if not started | Text source |
|---|---|---|---|---|---|
| **P0** | Critical | Correction of a spoken material fact. Made shot when Q4/OT ≤ 2:00 and margin ≤ 5 before the play. Lead change ≤ 2:00. Buzzer-beater. Foul-out | P1-P3 at the next word boundary | 3.0 s (corrections: 10 s) | Template |
| **P1** | Live call | Shot release call, made FG, and-one, dunk, block, steal, turnover, offensive rebound, whistle | P2-P3 at the next word boundary. Replaces queued P1 for the same possession | **2.0 s** after confirmation (release call: 0.8 s) | Template |
| **P2** | Context | Score/run update, timeout announcement, substitution, player stat milestone, foul attribution, FT result, replay hedge | P3 | 8 s, or until the next P1 | Template (LLM optional if TTL allows) |
| **P3** | Colour | Tendencies, matchup notes, storyline, background, quarter summary | none | End of the dead-ball window (≤ 20 s). Invalid on `state_version` change of any referenced fact | Claude (validated) |
| (C) | Coach | Cards and dead-ball digests (section 7) | separate channel | Until superseded | Code + optional LLM digest |

### 9.2 Max words per utterance by event type (≈ 3.5 words/s measured TTS rate [M])

| Event / slot | Max words | ≈ Duration | Notes |
|---|---|---|---|
| Shot release call ("Tatum, for three...") | 4 | 1.1 s | Only if P(shot) ≥ 0.90 |
| Made 2PT / 3PT result | 8 | 2.3 s | "...got it! Boston by five." The score tail only when the margin changed meaningfully |
| Dunk / and-one / block | 10 | 2.9 s | prosody = big |
| Missed shot (no other event within 1.5 s) | 5 | 1.4 s | Usually merged with the rebound call |
| Offensive rebound / putback chain | 6 | 1.7 s | Defensive rebounds: 4 words, spoken only if the slot is free |
| Steal / turnover | 8 | 2.3 s | — |
| Whistle (generic, unattributed) | 4 | 1.1 s | "Whistle, foul on the floor." |
| Foul with attribution / FT setup | 10 | 2.9 s | Only when the attribution gate passes |
| Free-throw result | 5 per FT | 1.4 s | Colour allowed between FTs |
| Timeout announcement | 10 | 2.9 s | Then colour |
| Substitution | 10 | 2.9 s | Dead ball only |
| Score / run update | 12 | 3.4 s | — |
| Correction | 12 | 3.4 s | Starts with "Correction:" |
| Colour between FTs | 20 | 5.7 s | 1 line |
| Colour in a timeout | 35 per line, ≤ 2 lines | 10 s each | — |
| Quarter-end summary | 60 | 17 s | LLM, validated |
| Halftime / final summary | 90 | 26 s | LLM, validated |

Human references: 2.8 s mean utterance [2], 13.2 words per NBA commentary sentence [13].

### 9.3 Confirmation delays and gates (CV-side, live path)

| Event | Provisional trigger | Confirm rule | Hold before narration | Gate to say it | Hedge band / fallback |
|---|---|---|---|---|---|
| Shot attempt | Release detected | — | 0 ms | P(shot) ≥ 0.90. Shooter ID ≥ 0.95, else team-level reference | Below 0.90: no release call |
| 3PT vs 2PT | Shooter foot position at release | Scoreboard delta (+2 vs +3) | 0 ms for the release call | Say "three" only if P(3) ≥ 0.97 | Otherwise "from the wing" (no point value). Correct via scoreboard |
| Made FG | Ball enters the cylinder | Ball below rim plane inside the cylinder for ≥ 2 frames, or a net-motion signal | **150-300 ms** | P(make) ≥ 0.95 | 0.80-0.95: wait up to 1.0 s for the scoreboard delta. Still unsure: silence |
| Missed FG | Rim/backboard contact without entry | Ball exits the cylinder region | 300 ms | ≥ 0.90 | — |
| Rebound | New possessor | Control ≥ 300 ms | 300 ms | ≥ 0.90 (ID ≥ 0.95 for a name) | Team-level fallback |
| Steal / turnover | Possession switch | New team control ≥ 500 ms | 500 ms | ≥ 0.90 | — |
| Block | Trajectory reversal near the release point + defender hand proximity | — | 300 ms | ≥ 0.92 | Otherwise narrate as a miss |
| Foul | Whistle audio + clock stop | Ref signal or official/scoreboard team-foul delta | 0.5 s generic. **2-5 s** attributed | Generic: P(whistle) ≥ 0.90. Attributed: ≥ 0.97 or official data | Generic line only |
| Timeout | Whistle + both benches converge, or scoreboard TO count | — | 2-3 s | ≥ 0.95 | "Stoppage on the floor." |
| Substitution | New track/jersey enters | ≥ 3 s on court + OCR ≥ 0.95 | dead ball only | ≥ 0.95 | Skip |
| Score | Scoreboard OCR change (OCR at 5 Hz, RESEARCH.md D14) | 2 consecutive identical reads with a +1/+2/+3 delta and a consistent clock | ≈ 400 ms (two reads 200 ms apart) | Authoritative over CV | Triggers a correction if it conflicts with the spoken line |
| Replay review | Officials at the monitor (CV/audio) or official feed | — | — | — | P2 hedge. Hold score claims until resolved [60] |

Jev's place (off the hot path, one batched `system_one` call per dead-ball window [61]). The question
name is the **mapping key** passed to `system_one(state, questions)`, not a constructor argument.
Question objects take keyword arguments only (`instructions=`, `criteria=`), and a positional form
such as `Choice("utterance_type", options=[...])` raises `TypeError` in `typesafe-sdk` 0.7.5
(checked against the repo's `.venv`) [61]. Question IDs are never sent to the model, so each
`instructions` string must be a full question [61]:

```python
from typesafe_sdk import Choice, Noul, NoulCriteria, Score

DEAD_BALL_QUESTIONS = {
    "utterance_type": Choice(
        instructions="Which kind of line should the commentator say next about `situation`?",
        criteria={"stat_note": "...", "tendency": "...", "storyline": "...",
                  "summary": "...", "silence": "Nothing is worth saying"},
    ),
    "clutch": Noul(instructions="Is the moment in `situation` a clutch moment?"),       # act at noul >= 0.8
    "storyline_due": Noul(instructions="Is a flagged storyline in `storylines` due to be mentioned now?",
                          criteria=NoulCriteria(true="...", false="...")),            # act at noul >= 0.75
    "excitement": Score(instructions="How exciting is the moment in `situation`?",
                        criteria=["calm", "rising", "big"]),                           # level 0..2 -> prosody tag
}
# resp = await aclient.system_one(state, DEAD_BALL_QUESTIONS)
# resp.choices["utterance_type"].choice / .confidence ; resp.nouls["clutch"].noul (P(yes), no confidence
# field) ; resp.scores["excitement"].score (expected level, may be fractional)
```

- Coach-card ranking uses the same call shape.

Thresholds and criteria texts are design proposals. Check against `jev_api.md` before coding [61].

### 9.4 Candidate buffer rules

- Candidate = `(priority, created_at, ttl, state_version, referenced_fact_ids, text|audio, source)`.
- On every fact-store revision, drop candidates whose `referenced_fact_ids` changed.
- Selection at a playback boundary:
  1. Highest priority first.
  2. Within a priority, the freshest valid candidate first. Exception: P3 uses the Jev `Choice` when
     it returned within its deadline.
  3. Ties go to the least-repeated topic.
- Buffer depth: P3 ≤ 2 per window (speculative pair, section 5.4). P1 = 1 per possession (latest
  wins). P2 ≤ 3.

### 9.5 Cooldowns and anti-repetition

| Rule | Value |
|---|---|
| Min gap between utterances | **0.6 s** live, **1.0 s** dead ball |
| Target gap (sampled) | 1.0-2.0 s, from an empirical-style distribution clipped at 4 s [2][14] |
| Score update | ≤ 1 per **60 s** game clock, unless lead change, tie, a run threshold, or ≤ 2:00 in Q4/OT |
| Run callout | At **8-0** (or 10-2) within 3:00 game clock. Update at +4 only |
| Player stat line | Per player ≥ **180 s** game clock. Only on milestones (10/20/30/40 pts, double-double, one foul from fouling out: 5th foul in the NBA (limit 6), 4th in FIBA/NCAA (limit 5); read the limit from league config) |
| Same template variant | Not within the last **3** uses of that event type (≥ 8 variants each) |
| Same fact (type, subject, value) | Never repeated within **5 min** game clock unless the value changed |
| Colour topic (same player/tendency) | ≥ **5 min**. Tendency lines ≤ 1 per 2 dead-ball windows |
| Colour share | Target ~20% of utterances, from the soccer 18% baseline [14] |
| Hedged phrasing | ≤ 1 in 3 consecutive utterances |
| Corrections | ≤ 1 per event. > 2 per 5 min triggers stricter gates (section 3.6) |
| Coach alerts | Same type ≤ 1 per 120 s game clock. ≤ 1 audio alert per dead ball |

### 9.6 Claude call configuration (colour, summaries, coach digest)

```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic()

async def stream_color(packet_json: str, system_blocks: list[dict]):
    # system_blocks: static style guide + roster/season facts, >=512 tokens, byte-stable,
    # last block carries {"cache_control": {"type": "ephemeral", "ttl": "1h"}}
    async with client.beta.messages.stream(
        model="claude-opus-5-5",
        max_tokens=1536,                      # thinking counts toward this; length enforced by validator
        output_config={"effort": "low"},      # Opus 5.5 always thinks; effort is the latency lever
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",                  # server-side refusal fallback (Anthropic default guidance)
        system=system_blocks,
        messages=[{"role": "user", "content": packet_json}],
    ) as stream:
        async for text in stream.text_stream:
            yield text                        # director's sentence splitter -> validator -> TTS
```

The calls and parameters come from Anthropic's SDK reference [27]. Check `stop_reason` on the final
message. Treat `refusal` or `max_tokens` as a validator failure and fall back to the template.
Verified offline against the repo's `anthropic` 1.15.0 with a mock transport: this function yields
the streamed text, and the request goes to `/v1/messages?beta=true` with header
`anthropic-beta: server-side-fallback-2026-07-01` and body fields `output_config`, `fallbacks:
"default"` and `stream: true` [M]. A server-side fallback can make later requests in the same
conversation stick to the fallback model for about an hour, and on a stream `message_start` names
the model that served it. Log `response.model` so a slower fallback is visible in latency
metrics [27].

System-prompt essentials:
- "You are a live basketball colour commentator. Use ONLY facts in the JSON packet. Do not introduce
  any player, team, number or event not in the packet. ≤ {max_words} words, 1-2 sentences, spoken
  style, no lists."
- "Answer directly without deliberating."
- "Do not repeat any phrasing from recent_utterances."
- "If nothing in the packet is worth saying, output exactly: SILENCE"

Per call, the input is a FactPacket of ≤ 1.5k tokens (design proposal). Cache the roster and style
guide.

---

## 10. Implementation implications (summary)

1. Build the fact store and event revision model first. Narration correctness depends on it.
2. P0-P2 output is template-only for v1. The LLM is a P3/summary add-on behind the validator.
3. Keep Typesafe Jev and Claude off the sub-second path. Run both concurrently at dead-ball start.
4. Warm everything at boot: one throwaway AVSpeech utterance (cold start 0.6-1.0 s [M]), a Claude
   cache pre-warm with a non-streaming `max_tokens: 0` request using the same system blocks and effort
   [27], and one Jev call.
5. Instrument every stage with `t_occ`, `t_confirm`, `t_enqueue`, `t_tts_first_buffer` and
   `t_audio_out`. Report p50/p95 per event type, plus validator reject rates and correction rates.
6. Evaluation: the Kawamatsu speaking/silence mIoU against human commentary at 1 Hz bins [2]; fact
   precision (every spoken claim checked against the official play-by-play); identity accuracy (vs
   NBA_Streaming's 28.5% baseline [13]); correction rate per game.

---

## Sources

1. Kawamatsu et al., arXiv abstract 2606.13322 — https://arxiv.org/abs/2606.13322
2. Kawamatsu et al., full HTML (architecture, Tables 1-2) — https://arxiv.org/html/2606.13322v1
3. IJCAI-ECAI 2026 proceedings entry (demo track, pp. 8427-8431) — https://www.ijcai.org/proceedings/2026/972
4. LiveCC abstract (CVPR 2025) — https://arxiv.org/abs/2504.16030
5. LiveCC full HTML (2 FPS, 0.17 s/frame, LiveSports-3K) — https://arxiv.org/html/2504.16030
6. LiveCC GitHub repo — https://github.com/showlab/livecc
7. MatchTime full HTML (misalignment, MatchVoice) — https://arxiv.org/html/2406.18530
8. MatchTime, EMNLP 2024 ACL Anthology — https://aclanthology.org/2024.emnlp-main.99
9. SoccerNet-Caption (CVPRW 2023) — https://arxiv.org/pdf/2304.04565
10. SoccerNet-Echoes — https://arxiv.org/abs/2405.07354
11. StreamSoccer (arXiv 2608.19723) — https://arxiv.org/html/2608.19723v1
12. NBA_Streaming abstract (arXiv 2608.09200) — https://arxiv.org/abs/2608.09200
13. NBA_Streaming full HTML v3 (Tables II/IV, dataset stats) — https://arxiv.org/html/2608.09200v3
14. Mori et al., "Live Football Commentary System Providing Background Information", ACL 2025 Demos — https://aclanthology.org/2025.acl-demo.38.pdf
15. Petri et al., "An LLM-Based Automatic Sportscast Solution for Robot Soccer Matches" (MARIO), arXiv 2607.14809 — https://arxiv.org/html/2607.14809v1
16. Zheng et al., AI-generated game commentary survey, arXiv 2506.17294 — https://arxiv.org/abs/2506.17294
17. IBM case study: how the Masters uses watsonx to manage its AI lifecycle — https://www.ibm.com/case-studies/blog/how-the-masters-uses-watsonx-to-manage-its-ai-lifecycle
18. IBM newsroom: US Open 2024 generative AI features (Granite 13B) — https://newsroom.ibm.com/2024-08-15-ibm-and-the-usta-serve-up-new-and-enhanced-generative-ai-features-for-2024-us-open-digital-platforms
19. IBM newsroom: 90th Masters AI experiences (2026; Hole Insights third year) — https://newsroom.ibm.com/2026-03-23-ibm-debuts-new-ai-enabled-digital-experiences-for-the-90th-masters-tournament
20. ESPN: Wimbledon and IBM AI-generated highlight commentary (2023) — https://espn.co.uk/tennis/story/_/id/37890817/wimbledon-ibm-team-ai-generated-highlight-commentary
21. WSC Sports: NBA launches GenAI multilingual voice-over (Apr 2025) — https://wsc-sports.com/blog/customer-spotlight/nba-launches-genai-technology-with-wsc-sports-to-automate-multilingual-content/
22. Awful Announcing: NBA/Amazon "Inside the Game" stats (Oct 1, 2025) — https://awfulannouncing.com/nba/inside-game-stats-platform-ai-aws-amazon.html
23. Broadcast: Stats Perform Opta Stream for Urban Edge Network HBCU basketball (Dec 2024) — https://www.broadcastnow.co.uk/tech-innovation/stats-perform-unveils-ai-driven-stats-for-urban-edge-network/5199955.article
24. CNN: NBC AI Al Michaels Olympic recaps (2024) — https://edition.cnn.com/2024/06/26/tech/al-michaels-ai-olympics
25. AI Incident Database #785: ESPN AI recap omits Alex Morgan — https://incidentdatabase.ai/cite/785/
26. GeekWire: Prime Video NBA "Prime 360" and AI tools (2026) — https://www.geekwire.com/2026/amazons-nba-coverage-adds-prime-360-ai-replays-that-fly-the-camera-through-the-action/
27. Anthropic Claude API reference, as bundled in the Claude Code `claude-api` skill (model table cached 2026-10-06; streaming, prompt caching, effort, fast mode, Opus 5.5 migration notes). Local: `/private/tmp/claude-501/bundled-skills/2.1.299/.../claude-api/` (SKILL, `python/claude-api/streaming.md`, `shared/prompt-caching.md`, `shared/model-migration.md`). Public docs root: https://platform.claude.com/docs
28. Artificial Analysis: Claude Opus 5.5 (low) — TTFT 6.52 s, 79.5 tok/s (Anthropic API, default 10k-input workload) — https://artificialanalysis.ai/models/claude-opus-5-5-low ; providers: https://artificialanalysis.ai/models/claude-opus-5-5-low/providers
29. Artificial Analysis: Claude Haiku 5.5 (medium: 178.6 tok/s, 13.84 s; low: 171.2 tok/s, 8.39 s) — https://artificialanalysis.ai/models/claude-haiku-5-5-medium/providers ; https://artificialanalysis.ai/models/claude-haiku-5-5-low
30. ElevenLabs models documentation — https://elevenlabs.io/docs/models
31. ElevenLabs WebSocket stream-input API reference — https://elevenlabs.io/docs/api-reference/text-to-speech/v-1-text-to-speech-voice-id-stream-input
32. OpenAI text-to-speech guide — https://developers.openai.com/api/docs/guides/text-to-speech
33. Cartesia TTS models documentation — https://docs.cartesia.ai/build-with-cartesia/tts-models
34. invideo review of Cartesia Sonic 3/3.5 latency (third-party) — https://invideo.io/blog/cartesia-sonic-ai-voice/
35. Kokoro-82M model card — https://huggingface.co/hexgrad/Kokoro-82M
36. hexgrad/kokoro GitHub — https://github.com/hexgrad/kokoro
37. HF discussion: Kokoro KPipeline ~11x realtime on M3 Max CPU (community) — https://huggingface.co/mlx-community/Kokoro-82M-bf16/discussions/5
38. mlx-audio GitHub — https://github.com/Blaizzy/mlx-audio
39. piper-tts on PyPI (v1.8.0, GPL-3.0-or-later) — https://pypi.org/project/piper-tts/
40. PromptQuorum Piper review (repo history / archival date; secondary) — https://www.promptquorum.com/power-local-llm/piper-tts-review
41. kyutai-labs/pocket-tts GitHub — https://github.com/kyutai-labs/pocket-tts
42. Kyutai TTS page (220 ms, streaming text input) — https://kyutai.org/next/tts
43. kyutai/tts-1.6b-en_fr model card — https://huggingface.co/kyutai/tts-1.6b-en_fr
44. sesame/csm-1b model card — https://huggingface.co/sesame/csm-1b
45. Apple AVSpeechSynthesizer documentation — https://developer.apple.com/documentation/avfaudio/avspeechsynthesizer
46. [M] Local measurements, 2026-10-20, Apple M3 Pro / macOS 26.5.1. `say -o` timings, and AVSpeechSynthesizer first-buffer timing via `pyobjc-framework-AVFoundation` 12.2.2 (script `avspeech_ttfb.py` in the session scratchpad). Re-run by the fact-checker the same day (second run's figures merged into the ranges above), plus a write-path word-range callback test and an offline mock-transport test of the section 9.6 code
47. Ishchenko, fast speaking rate in TV football discourse (UEFA Euro 2012) — https://zenodo.org/records/19663708
48. Wikipedia: Speech tempo — https://en.wikipedia.org/wiki/Speech_tempo
49. Maymin, Maymin & Shen, "How Much Trouble is Early Foul Trouble?" (Sloan / IJSF 2012) — https://sloansportsconference.com/research-papers/how-much-trouble-is-early-foul-trouble ; summary: https://www.espn.com/blog/truehoop/post/_/id/25832/how-much-trouble-is-early-foul-trouble
50. "Stop the Clock: Are Timeout Effects Real?" (ECML-PKDD 2020) — https://arxiv.org/abs/2009.06750v1
51. Gibbs et al., "The causal effect of a timeout at stopping an opposing run in the NBA" — https://arxiv.org/pdf/2011.11691
52. Bocconi, "Do timeouts matter? A study of EuroLeague basketball" (2026) — https://iris.unibocconi.it/retrieve/befbdb69-1af2-4097-8e89-26fa894b51f4/Do_timeouts_matter_A_study_of_euroleague_Basketbal.pdf
53. NCAA: live statistics can be transmitted to benches in men's basketball (2021) — https://www.ncaa.org/news/2021/6/3/live-statistics-can-be-transmitted-to-benches-in-men-s-basketball.aspx
54. Sports Illustrated: NBA will allow teams to review real-time video from the bench (2016) — https://www.si.com/media/2016/10/05/nba-real-time-video-review-bench
55. ShotTracker: tech on the bench at the Hall of Fame Classic — https://shottracker.com/the-next-frontier-of-coaching-tech-on-the-bench-at-the-hall-of-fame-classic/
56. ESPN: Mountain West coaches to use real-time analytics (2019) — https://www.espn.com/mens-college-basketball/story/_/id/27475152/mwc-coaches-use-real-analytics
57. Cleaning the Glass, "Ask Ben: Halftime, changing play calls..." (2017) — https://cleaningtheglass.com/ask-ben-halftime-changing-play-calls-and-technology-investment/
58. Basketball-Reference: Four Factors — https://basketball-reference.com/about/factors.html ; "Dean Oliver's Four Factors Revisited" — https://arxiv.org/pdf/2305.13032 ; Inpredictable, "How NBA Games Are Won" — https://www.inpredictable.com/2015/02/how-nba-games-are-won.html
59. L-RAPM: Lineup Regularized Adjusted Plus-Minus (lineup sample sizes) — https://arxiv.org/pdf/2601.15000
60. NBA Official: replay triggers — https://official.nba.com/replay/triggers/
61. Sibling note `docs/research/jev_api.md` (Typesafe Jev API: `system_one`, `Choice`/`Score`/`Noul`, latency ~100 ms claimed / ~0.4 s median measured in cookbook, "compute numbers in Python") — local file in this repo; its own sources include https://api.typesafe.ai/openapi.json . Constructor shapes re-checked against `typesafe-sdk` 0.7.5 in the repo `.venv` (see also `docs/research/sdk_ground_truth.md`)
62. LiveCC-7B-Instruct model card (Apache-2.0, 8B params, BF16) — https://huggingface.co/chenjoya/LiveCC-7B-Instruct
63. espeak-ng GitHub (GPL-3.0-or-later) — https://github.com/espeak-ng/espeak-ng

---

## Verification log

Adversarial fact-check, 2026-10-20. I re-read each cited primary source with WebFetch, or with
local PDF text extraction (pypdf) where WebFetch returned raw PDF bytes. I re-ran the local
measurements and tested the hard-coded SDK calls offline against the packages in the repo `.venv`.
The web-search quota for the run ran out partway through, so URLs that could not be fetched were not
replaced with search results. They are marked "(unverified)" or "(not re-verified)" in place.

### Confirmed (no change needed)

- **Kawamatsu et al.** [1][2][3]: title, authors, IJCAI-ECAI 2026 Demo Track pp. 8427-8431. 25 fps,
  N = 32, GPT-4.1-mini with base64 frames, Nintendo Switch → Elgato HD60 X, YouTube Live. Generation
  starts on each new segment during playback. Latest/Oldest/Random chosen "to avoid additional
  decision latency". No buffer size or staleness rule. The fixed video delay starts playback with the
  first utterance. Table 1 and Table 2 values match exactly (2.8 ± 0.5, 25.9; 40 tokens best matches
  duration, 60 best matches length). The abstract gives 9.6 s → 0.3 s and "over 40%". The paper says
  policy differences are "relatively small". 8 SMASH videos, 1 Hz mIoU, 120 crowdworkers.
- **Mori et al.** [14] (full text extracted from the ACL PDF): 18%. Out-of-play p = 0.5 within 15 s.
  Gaps > 4 s excluded. MSE 15.50 vs 19.86 at a fixed 2.14 s. The "An Arsenal player passed..."
  fallback. YOLOv8 at confidence 0.3, 25-frame interpolation, T-DEED. RAG: 3,257 pages, chunks of
  1,000/100, ada-002, top 10 above cosine 0.7, query windows 2 s / 60 s / 15 s, date-cutoff prompt.
  Timing: 5:55 offline on a TITAN 24 GB, jersey OCR 3:54, online 0:27 = 0:10 + 0:17. 200 wpm,
  OpenAI TTS. Relevance 1.73 / 2.05 / 2.55. Factual accuracy "unmeasured".
- **NBA_Streaming** [13]: 2025-26 season, 152 games, 307.5 h, 35.0K events, 50.4K sentences, 539
  players, 15 event types (including `3pt_missed_rebound`), 71 actions, 13.2 words per sentence.
  Threshold 0.5, 37 candidates, 6 FPS with about 22 s of context, WASB, Qwen3-VL-8B-Instruct.
  2.470 s / 2.094 s, 28.533%, 71.387%, 47.053%.
- **MARIO** [15]: goals fire only on an OCR score increase. The "one short, factual sentence..."
  contract over machine-validated fields. Higher priority preempts queued output. Duplicates are
  suppressed. The cooldown keeps "only the earlier one". Stale data has a freshness threshold (no
  number given). No LLM, TTS or per-stage latency named.
- **StreamSoccer** [11], **LiveCC** [5][6], **Zheng survey** [16]: all quoted numbers match.
- **IBM Masters** [17]: few-shot prompting with no fine-tuning during the tournament (input
  statistics change instead). HAP, multiple validating LLMs and regex guardrails. ROUGE/METEOR/
  perplexity gates. Re-validation per round. 20,000+ clips. Spanish in 2024. **IBM US Open** [18]:
  Granite 13B, "minutes after", editor review. The release gives "254" and also quotes "over 150"
  matches, which conflict. **IBM 2026** [19]: Hole Insights is in its third year.
- **WSC Sports × NBA** [21]: Apr 6, 2025. French/Portuguese/Spanish, 2-minute highlights "within
  minutes of final whistle". The body says 75% average completion (the subtitle says "more than 75%").
  No human review mentioned. **AWS Inside the Game** [22]: Oct 1, 2025, 29 body points, broadcaster
  alerts, "Eventually, it will be in-game augmentation". **Opta Stream** [23]: Dec 9, 2024, SWAC/MEAC,
  text vs audio not stated.
- **Claude API** [27] (claude-api skill files: SKILL, `shared/model-migration.md` § Migrating to
  Claude Opus 5.5, `shared/prompt-caching.md`, `python/claude-api/streaming.md`):
  - Opus 5.5 always thinks. `{type:"disabled"}` and `budget_tokens` return 400 at every effort.
  - Effort defaults to `medium` and is set via `output_config.effort`. Thinking counts toward
    `max_tokens`.
  - 512-token minimum cacheable prefix. Cache reads $0.20/MTok. 5 min default TTL, `ttl: "1h"`.
    `max_tokens: 0` pre-warm.
  - Fast mode: `fast-mode-2026-02-01`, `speed="fast"`, up to 2.5×, $8/$40, Claude API only.
  - No prefill. `fallbacks: "default"` + `server-side-fallback-2026-07-01`.
  - `messages.stream(...)` / `async for text in stream.text_stream`. Opus 5.5 $4/$20.
- **Artificial Analysis** [28]: Opus 5.5 low on Anthropic: 79.5 tok/s, 6.52 s time to first answer
  token (including thinking).
- **ElevenLabs** [30][31]: `eleven_flash_v2_5` ~75 ms excluding app and network, 32 languages, 40k
  characters. `eleven_v4_turbo` ~100 ms, 90+. `eleven_v3_conversational` ~280 ms. `eleven_turbo_v2_5`
  deprecated. WSS URL. `chunk_length_schedule` default [120,160,250,290], each item 50-500. `flush`
  forces generation of short buffers.
- **OpenAI TTS** [32]: `gpt-4o-mini-tts`; `wav`/`pcm` fastest; 24 kHz 16-bit LE; 13 voices listed
  (the intro says 11); `marin`/`cedar` best. **Cartesia** [33]: `sonic-3.6`,
  `sonic-3.6-2026-08-27`, 44 languages, no latency figures.
- **Kokoro** [35][36]: Apache-2.0, 82M, 54 voices, v1.0 on 27 Jan 2025.
  `pip install kokoro>=0.9.4 soundfile`, `KPipeline(lang_code='a')` yields `(gs, ps, audio)` at
  24 kHz. `PYTORCH_ENABLE_MPS_FALLBACK=1`, espeak-ng for OOD words, `misaki`. Community report [37]:
  ~11× real time on an M3 Max CPU via KPipeline.
- **Piper** [39]: 1.8.0 on Sep 4, 2026, GPL-3.0-or-later, Python ≥ 3.9,
  `cp39-abi3-macosx_11_0_arm64` wheel. `rhasspy/piper` is MIT and was archived Oct 6, 2025, and
  development moved to `OHF-Voice/piper1-gpl` (checked directly on GitHub:
  https://github.com/rhasspy/piper).
- **Pocket TTS** [41]: MIT code, weights licence not stated, ~100M, ~200 ms first chunk, ~6× on a
  MacBook Air M4 CPU, `TTSModel.load_model()` / `generate_audio(voice_state, text)`, 26 voices, 7
  languages, cloning.
- **CSM-1B** [44], **mlx-audio** [38], **Kyutai 1.6B card** [43] (CC-BY-4.0, 1.28 s lag): match.
- **Coach-side**: Q+1 [49] (Sloan page plus the ESPN summary). Timeouts: no effect after matching
  [50]; "slightly disadvantageous" [51] (PDF text extracted). NBA hard-wired bench video in 2016-17
  [54]. ShotTracker 2018 0.6 PPP pass-count split [55]. Mountain West 2019-20 tablets with no video
  [56]. Cleaning the Glass halftime "few minutes", box score at timeouts, pit-stop changes [57].
  Oliver weights 40/25/20/15 [58]. 600+ lineups, ~25-30 possessions per lineup [59] (PDF text
  extracted). Replay triggers [60]. Speaking rates [47][48]: 7.15 syll/s average, peaks 9, Ukrainian;
  ~4 syll/s English.
- **Local [M] re-run** (same M3 Pro, macOS 26.5.1):
  - AVSpeech: 708 ms cold, then 20.9 / 25.6 / 12.3 / 18.2 ms first buffer. The 21-word line rendered
    5.99 s of audio in 74.7 ms at 22,050 Hz. 180 voices, all quality 1.
  - `say -o`: 1.28-2.81 s per call for a 0.864 s, 22,050 Hz clip.
  - PyObjC selectors `stopSpeakingAtBoundary_`, `writeUtterance_toBufferCallback_`,
    `speechUtteranceWithSSMLRepresentation_` and
    `requestPersonalVoiceAuthorizationWithCompletionHandler_` all exist.
- **Section 9.6 code**: ran unmodified (async generator) against `anthropic` 1.15.0 with
  `httpx2.MockTransport`. It streamed text correctly. The request had the beta header
  `server-side-fallback-2026-07-01` and the body fields `output_config`, `fallbacks: "default"`
  and `stream: true`.

### Changed

1. **Haiku 5.5 speed**: "about 137 output tok/s at medium" was wrong. AA gives 178.6 tok/s and
   13.84 s time to first answer token at medium, and 171.2 tok/s and 8.39 s at low [29]. I added
   that this is not a TTFT win over Opus 5.5 low, the >100K-token price tier ($0.50/$2.50), and that
   there is no server-side fallback for Haiku 5.5 [27].
2. **Jev question constructors (section 9.3)**: `Choice("utterance_type", options=[...])` and
   `Noul("clutch")` raise `TypeError` in `typesafe-sdk` 0.7.5, which I tested in the repo `.venv`.
   I replaced them with the documented keyword form (`instructions=`, `criteria=`, names as mapping
   keys) and validated the new snippet offline [61].
3. **AVSpeech interruption mechanism (section 3.6, section 6 table, section 8 diagram)**:
   `stopSpeaking(at:)` and the `willSpeakRange` delegate were presented as the way to interrupt
   output from the write-to-buffer + own-mixer path. Stopping applies to `speak_()` playback. On the
   write path the range callbacks fire at render time: I measured all 5 words of a 2.07 s line within
   about 6 ms. I documented the frame-offset mapping approach and the PyObjC selector names.
4. **AVSpeech and `say` numbers**: widened to cover the re-run. Warm 12-26 ms (was 14-21), cold
   0.6-1.0 s (was 982 ms), full render 67-75 ms, `say` 0.78-2.81 s (was 0.78-1.89).
5. **Mori after-stoppage colour share**: "25-46%" changed to 21-46%. Table 6 has ball out of play
   at 21.3% and foul at 24.4%. F1 is now given as 22.7% (predicted timing) / 22.9% (gold timing). The
   kick-off rule now reads "within 15 s before or after". The jersey→name mapping source is now
   SoccerNet-Caption.
6. **NBA_Streaming**: the baseline identity range "0.5-4.5%" changed to 0.0-4.5%. I added that all
   accuracies are computed only on events with both boundary errors ≤ 2 s.
7. **MatchTime**: "LLaMA-3 with LoRA r = 16" is not supported by the paper. It ablates ranks
   8/16/32/64 and the headline numbers match the "Frozen LLM" row. The `[PLAYER]`/`[TEAM]` masking
   is stated for the GPT-score prompt.
8. **LiveCC memory**: "7B → about 14 GB" changed to 8B params → about 16 GB in bf16. I added the
   model-card licence (Apache-2.0) [62].
9. **SoccerNet-Caption**: the 5.9% BLEU@4 figure could not be found in the extracted text and is
   marked unverified. "Flags factual accuracy as a challenge" is rephrased to what the paper says.
10. **Bocconi EuroLeague timeouts**: "about −2.5 pts" was ambiguous. It is a +2.5-point short-run
    improvement in point differential (−3.74 → −1.20), with no season-level effect [52].
11. **Prompt-cache pre-warm**: `max_tokens: 0` must not be streamed and must match the
    system/effort settings. Whether it combines with `fallbacks` is undocumented (unverified).
    Section 10 is updated to match. Fast mode: added its separate rate limit and that switching speed
    invalidates the cache.
12. **Latency prompt lines**: added which model each line is documented for, and the prompt-audit
    guidance to keep one only when measurement shows it helps.
13. **Artificial Analysis context**: added the 10,000-input-token default workload and the
    per-provider TTFTs. "Long reasoning prompts" was inaccurate. The 12.35 s snapshot is marked
    unverified.
14. **Cost check**: added that output and thinking tokens ($20/MTok) and cache writes are excluded.
15. **League-dependent foul limits**: the "5th foul" milestone and the foul-out wording now read the
    limit from config (NBA 6; FIBA/NCAA 5, per the sibling rules note).
16. **Replay triggers**: added trigger numbers, the last-2:00 restriction for goaltending and
    out-of-bounds via the coach's challenge [60].
17. **Kyutai TTS**: the card says 1.8B parameters, not 1.6B. The 220 ms figure is unverified on
    re-check because the page rendered empty.
18. **Kokoro MLX**: added that the community report [37] fell back to the PyTorch CPU path because
    its MLX attempts failed. espeak-ng's licence is confirmed as GPL-3.0-or-later [63].
19. **MARIO naming**: clarified that MARIO is the framework the sportscaster extends.
20. **Pace**: replaced "common knowledge" with the sibling note's 98.5-99.4 and made the 14 s
    arithmetic explicit (about 200 total possessions).
21. **Mountain West / HoF Classic**: "no video" is now attributed only to the Mountain West waiver,
    because ShotTracker's 2018 page does not mention video.

### Could not verify (marked in place)

- NCAA 2021-22 live-stats rule [53]: the URL now redirects to the NCAA news index. The claim is
  consistent with the 2019-20 Mountain West waiver report [56], but it was not re-read.
- NBC Olympic recap details [24]: HTTP 451. Prime Video features [26]: HTTP 403. ESPN recap length
  "215 words" [25]: not on the AIID page. Kyutai 220 ms [42]: empty page. SoccerNet-Caption 5.9%
  BLEU@4 [9]. The AA 12.35 s snapshot.
- Already marked unverified by the author and not re-checked: Wimbledon 3B, Piper voice licences,
  OTT delays, the CoreAudio buffer, CV stage latencies, and the macOS enhanced-voice UI path.

### Evidence URLs used in this check (beyond the Sources list)

- https://github.com/rhasspy/piper (archived Oct 6, 2025; MIT; moved to OHF-Voice/piper1-gpl)
- https://artificialanalysis.ai/models/claude-opus-5-5-low/providers
- https://artificialanalysis.ai/models/claude-haiku-5-5-low
- https://huggingface.co/chenjoya/LiveCC-7B-Instruct
- https://github.com/espeak-ng/espeak-ng
- https://pypi.org/project/pyobjc-framework-AVFoundation/ (12.2.2, Aug 11, 2026, MIT)
- Local: claude-api skill files (`shared/model-migration.md`, `shared/prompt-caching.md`,
  `shared/prompt-audit.md`, `python/claude-api/streaming.md`); repo `.venv` (`typesafe-sdk` 0.7.5,
  `anthropic` 1.15.0)

---

## 11. Addendum: gap pass (2026-10-20)

Cross-note decisions are in `docs/RESEARCH.md` §1A (D1-D16) and §7A. This addendum records the
narration-specific parts.

### 11.1 What every gate's P(·) means (§9.3)
Each "P(make) ≥ 0.95"-style gate in §9.3 reads `Event.confidence` from the event's logistic cue
model (RESEARCH.md §7A.2). While a model's `p_status` is `uncalibrated`, its gate is never met,
and the director runs that event **hedge-only**: templates may name the attempt, zone and team
("shot from the wing", "rebound, Home") but no outcome or point value until the scoreboard, the
next-possession pattern or the operator confirms it. Gates switch on only after held-out
certification by the rule of three (0 errors in ≥ 3/ε accepted events, e.g. ≥ 150 for ε = 2%) [G6].
Platt scaling is used below about 1,000 labelled events, isotonic only above [G7][G8].

### 11.2 Make-wait band with no data on scoreboard lag
The "0.80-0.95: wait up to 1.0 s for the scoreboard delta" rule has no source for amateur scoreboard
operators. Until we measure the lag on our own games, wait up to 3.0 s (the CV note's broadcast
confirmation time). With no visible scoreboard, wait for the next-possession pattern (inbound under
the basket means a make, rebound means a miss). Set the wait to the measured p90 lag afterwards.

### 11.3 Latency harness and Claude policy (supersedes the §5.1 "1.5 s / 3 s" plan)
- `claude-opus-5-5` cannot disable thinking; `low` effort must be set explicitly because the
  default is `medium`. `claude-haiku-5-5` *can* run with `thinking={"type": "disabled"}` at effort
  `low`-`high`, has no server-side fallback, and costs $0.10 / $0.50 per MTok (prompts ≤ 100K) versus
  Opus 5.5's $4 / $20 [G1].
- Harness (RESEARCH.md §7A.3): about 50 calls each for C1 Opus `low`; C2 Opus `low` + "Answer
  directly without deliberating."; C3 Haiku `low` thinking disabled; C4 Haiku `low` adaptive. Real
  ≈ 1.5k-token FactPacket behind a pre-warmed ≥ 512-token cached prefix. Record time to
  `message_start`, to the first text delta, total time, `usage.output_tokens` (includes thinking),
  `cache_read_input_tokens`, `stop_reason` and `response.model`. Run from the dev Mac and from gym
  Wi-Fi / LTE.
- **Policy until measured:** request Claude candidates only when
  `expected_window_s ≥ p95_claude_total + tts_s + 1 s`. With p95 unknown, that limits Claude to
  timeouts, period breaks, halftime and pre-generated *state-independent* topics. Rule-defined
  window minimums: NFHS timeouts 60 s or 30 s, 1 min between quarters, 10-min halftime [G2]; FIBA
  timeouts 1 min [G3]; NBA additional timeouts 1:15, mandatory 2:45 / 3:15 [G4]. Free throws and
  inbounds are unmeasured, and in NFHS a made basket does not stop the clock [G2].
- No first-token abort (the old `first_token_deadline_s=1.5` would abort nearly every call at a
  6.52 s TTFT). Each candidate has `candidate_deadline_s` = measured p95 (initial placeholder 12 s).

### 11.4 Pseudonymised FactPackets
FactPackets carry per-game tokens (`P07`) instead of player names. Claude writes `{P07}`. The
deterministic validator checks every token against the packet; only then does the local renderer
substitute the roster name, or "number 23" under the identity tiers. The name map never leaves the
device. If consent is missing, any rostered player is under 13 without verifiable parental consent,
or the customer DPA does not yet list Anthropic and TypeSafe, narration runs template-only
(RESEARCH.md §7A.7).

### 11.5 Caption and audio-description transport
| Target | Transport |
|---|---|
| OBS | obs-websocket v5 `SendStreamCaption` (`captionText`), which "Sends CEA-608 caption text over the stream output" [G5] |
| YouTube Live | HTTP POST to the event's signed Captions Ingestion URL, or EIA-608/CEA-708 embedded by the encoder; one caption track; 30 s or 60 s broadcast delay advised for caption software [G9] |
| Our HLS player | WebVTT subtitle rendition (`EXT-X-MEDIA TYPE=SUBTITLES`, URI required) with `X-TIMESTAMP-MAP=LOCAL:…,MPEGTS:…` mapping cues to video timestamps [G10] |
| Dashboard / AD app | sidecar WebSocket `{"type": "caption", "pts": …, "text": …, "channel": "caption" \| "ad"}` |

Every cue carries the source video PTS of `t_occ` and is shown at `pts + D`, where D is the
stream's applied delay (overlay 0.8-1.0 s, or a 30-60 s caption or anti-wagering delay). The AD
p95 ≤ 2 s target is measured against the delayed video the listener receives.

**AD vocabulary (proposal).** ≤ 6 words per call, from a fixed lexicon: possession ("Home ball"),
progress ("across half court"), zones (left corner, left wing, top, right wing, right corner, paint,
backcourt), shot ("shot, left wing"), result ("good" / "no good" / "blocked"), rebound ("rebound,
Away"), turnover ("turnover, Home"), and the score after every possession, from confirmed values
only.

### 11.6 Second-screen Q&A (U10, P1, proposal)
`claude-opus-5-5` at `low` effort, with three read-only tools over our event log
(`get_events(filter)`, `get_box(entity)`, `get_score(t)`). Answers cite event IDs; the deterministic
validator checks every number against tool outputs; questions outside the log, about betting, or
about minors' personal details get a fixed refusal. Target ≤ 10 s; never on the live path.

### 11.7 TTS adapter
`ARCHITECTURE.md` now names `AVSpeechTTS` (PyObjC `AVSpeechSynthesizer`, warmed at boot) as the
production adapter. `say` stays dev-only because it measured 0.78-2.81 s per line [M].

Sources for this addendum:
[G1] Anthropic Claude API reference, bundled in the Claude Code `claude-api` skill (Opus 5.5 and Haiku 5.5 thinking, effort, pricing, cache minimum). https://platform.claude.com/docs
[G2] Abilene Basketball Referees, reproduction of NFHS Rule 5 (5-5, 5-8-1, 5-11). https://abilenebasketballrefs.com/nfhs-rules-book/nfhs-rules-book-rule-5-scoring-and-timing-regulations/
[G3] FIBA, Official Basketball Rules 2026 v1.1 (Art. 18.2: each time-out lasts 1 minute). https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2026-v1-1.pdf
[G4] NBA, Rule No. 5, Scoring and Timing (Section VI). https://official.nba.com/rule-no-5-scoring-and-timing/
[G5] obs-websocket 5.x protocol. https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md
[G6] "Rule of three (statistics)" (Hanley & Lippman-Hand 1983). https://en.wikipedia.org/wiki/Rule_of_three_(statistics)
[G7] Niculescu-Mizil & Caruana, ICML 2005. https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf
[G8] scikit-learn User Guide, "Probability calibration". https://scikit-learn.org/stable/modules/calibration.html
[G9] YouTube Help, "Add live captions to a live stream". https://support.google.com/youtube/answer/3068031
[G10] IETF RFC 8216, §3.5 and EXT-X-MEDIA. https://datatracker.ietf.org/doc/html/rfc8216
