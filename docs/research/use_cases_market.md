# CourtVision: use cases, users, existing products, and legal/data constraints

Research note, 2026-10-20. Scope: who would use a real-time AI basketball narrator and live-stats system, what already exists and what it costs, and what legal and data rules limit how we collect footage, train models and ship outputs. The last sections set priorities (value against feasibility) and a minimum-viable feature set for each user type.

System assumed throughout: CV tracking of players, ball and rim, an event engine that turns tracks into basketball events and a live box score, **TypeSafe AI's Jev** (a "System One" decision model that answers Choice, Score and Noul questions with probabilities and writes no prose) for narrow judgments, and **Claude (`claude-opus-5-5`)** for grounded narration and insight. Dev and edge box: Apple M3 Pro, 18 GB, MPS, Python 3.12.

> Legal content below is engineering research, not legal advice. Every "recommendation" is a design default to review with counsel before launch.

---

## 0. TL;DR (decisions this note supports)

1. **Target amateur basketball first** (high school, small college, AAU/club, adult rec), not NBA or NCAA broadcasts. Pro and major-college rights are locked up. NBA tracking and stats come from Sony Hawk-Eye cameras plus Sportradar, which is the NBA's exclusive data provider and global distributor of official betting data under an eight-year deal from 2023-24 to 2031 [4][5][10]. NCAA championship footage is NCAA copyright, licensed through Veritone and TNT Sports [68]. NBA.com terms bar using NBA statistics for gambling, fantasy, commercial products, or any live or archived play-by-play product [65].
2. **The gap we fill.** At the amateur level, capture is solved (Pixellot/NFHS Network, Hudl Focus, Veo, BallerCam) and post-game stats are partly solved: Hudl Assist uses human taggers at $900 to $2,100 per team per season [14][15], SportsVisio is AI with 5 games/month on its $149 to $189/month plans [33], and PlayVision is post-game AI [41] (its YC company page now shows a sports-betting app, "Bron", from the same founders, so PlayVision may have pivoted [42]). We found **no shipping product that produces live, grounded, spoken play-by-play plus a live box score directly from the video of an amateur basketball game**. Pixellot has announced "AI commentary" with no published details [24][25]. WSC Sports' AI voice-overs are for finished NBA highlight clips, not live games [45].
3. **The narration stack is also an accessibility product.** The same event stream can drive (a) a live caption track and (b) an audio-description mode for blind and low-vision fans. NBA teams already pay for human audio description (Pacers with MindsEye [52]) and for haptic and AI products (OneCourt with the Trail Blazers and Nuggets [50][51]). Public schools and universities will need WCAG 2.1 AA web content under the ADA Title II rule. The DOJ moved the deadlines to 2027/2028 [112].
4. **Privacy by architecture is a selling point.** Basketball Australia's 2026 Pixellot rollout (HoopsTV, announced 2026-10-12, go-live expected November 2026 [29]) drew a backlash over opt-out filming of children, footage reportedly transferred or stored overseas without explicit consent, and access controls [28][29][30]. We can run on-device on the M3 Pro: footage stays in the gym, there is no face recognition, players are identified by jersey number plus a coach-entered roster, and track embeddings are discarded at game end. This keeps us clear of most BIPA, CUBI, COPPA and GDPR Art. 9 triggers [78][84][89][94].
5. **Do not train on scraped NBA/NCAA/YouTube footage.** YouTube terms prohibit unauthorized downloading and scraping [74]. The 2025 fair-use rulings protected transformative training but not illegitimately acquired copies [75][76]. Build a consented "own footage" corpus, and use permissively licensed datasets only after checking their licenses. DeepSportradar data is non-commercial: the official instance-segmentation challenge README says CC BY-NC-ND 4.0 (code Apache-2.0) [119], and a third-party listing says CC BY-NC-SA 4.0 [100].
6. **Exclude betting and fantasy feeds.** Official-data exclusives (NBA to Sportradar through 2031 [10]; NCAA postseason to Genius Sports through 2032 [103]), betting bans on youth and high school games (for example, Virginia makes it a Class 1 misdemeanor [108]), state bans on college player props [105], and the October 2025 NBA gambling indictments [106] make this high-risk with low defensibility. Build anti-wagering terms of service and an optional output delay instead.

---

## 1. Users and use cases

Each row gives the job to be done, the latency the user will tolerate, the accuracy bar, and which CourtVision component carries the load. Latency and accuracy targets are our engineering proposals unless cited.

| # | User | Job to be done (in-game / post-game) | Latency tolerance | Accuracy bar | Main components |
|---|---|---|---|---|---|
| U1 | **Head coach** (HS / small college / AAU) | In-game: foul trouble, opponent runs, lineup +/-, timeouts left, shot profile ("they're 0-7 from the left corner"). Post-game: box score, shot chart, lineup report, key-possession clips | In-game 5-30 s; post-game minutes | High for counts (fouls, points). Coaches lose trust after one wrong foul count | Event engine and live box score; Claude for summaries; Jev for "is this alert worth interrupting for?" (Noul) |
| U2 | **Assistant coach / video coordinator** | Tag possessions (PnR, transition, ATO, zone vs man), build clip playlists, find plays fast. NCAA men's coaches now challenge calls using iPad review by video staff [58] | Clip lookup < 10 s in-game; full tagging same night | Medium for play types (human edits tags). High for timestamps | CV and event engine for possession segmentation; Jev Choice for play-type labels; Claude for search ("show me every baseline out-of-bounds play") |
| U3 | **Analytics staff** (college, top HS/prep) | Raw x,y tracks, possessions, shot-quality features, exports to their tools | Batch | Medium-high, with explicit uncertainty | Tracking export (CSV/Parquet/JSON); event log with confidences |
| U4 | **Player development coach** | Shooting % by zone, release timing, footwork (pose), workload in drills | Post-session | Medium | Pose (single camera), shot detection; HomeCourt and Ball AI already serve consumers [37][39] |
| U5 | **Scouts / recruiting** (college staff; recruiting services) | Opponent tendencies; prospect evaluation; highlight reels | Post-game | Medium-high on player ID | Player ID by jersey; clip extraction; Claude scouting summaries. Note NCAA DI basketball scouting-service approval rules [114][115] |
| U6 | **Athletic trainer / S&C** | Load: distance, high-speed running, accelerations/decelerations, jumps | Post-session; live alerts optional | High (load decisions). Pros use UWB wearables: Kinexon, sub-10 cm and 20 ms [61][62] | Optical tracking from 1-2 cameras is noisy for accelerations; low feasibility (see §6) |
| U7 | **Referees / officiating review** | Post-game clip packs of calls for assessor review; development. ACC adopted RefScore for this in 2025-26 [56]; NBA uses its REPS app [57] | Post-game | High; politically sensitive | Event engine (fouls, out-of-bounds, violations), clip export. Not "AI ref" calls |
| U8 | **Broadcasters with no commentator** (HS streams, small colleges, AAU events, rec leagues) | Live audio play-by-play and color, score bug, stat graphics | 1-3 s after the event (proposed) | Very high for facts said aloud (wrong scorer or score is embarrassing). Color can hedge | Event engine, Jev gating, Claude narration, TTS; Pixellot/NFHS streams [22][27] and BallerCam [43] as ingest |
| U9 | **Blind / low-vision fans (audio description)** | Where the ball is, who has it, court location, score/clock, outcome of each possession | ~1-2 s (proposed) | High for possession and score; spatial detail matters more than flair | Same events as U8 plus spatial phrasing ("left wing, three steps inside the arc"); optional haptics later (OneCourt model [50]) |
| U10 | **Fans / second screen** | Live box score, "what just happened?", Q&A ("how many threes does #23 have?"), recaps | Seconds | High for numbers | Claude Q&A grounded only on the event log (Wimbledon's Match Chat is a precedent [46]) |
| U11 | **Youth / AAU parents** | "My kid's" stats, highlights, shareable clips, stream | Post-game (same day) | Medium-high on player ID | Jersey-based player ID, per-player reels. Highest privacy sensitivity (minors) |
| U12 | **Recruiting highlight reels** (athletes, families, HS coaches) | 3-5 min reel of a player's best plays plus stat line | Days | High on attribution | Clip ranking (Jev Score: "highlight quality 0-4"), Claude captions; NIL and consent rules (§5.4) |
| U13 | **Fantasy / betting data consumers** | Low-latency event feed, player props | < 1 s | Very high | **Not pursued** (§5.6) |
| U14 | **League / tournament operators** (AAU events, rec leagues, state associations) | Stats for every game on every court, standings, leader boards, streaming upsell | Same day | Medium-high | Multi-court batch processing; SportsVisio has a league tier [33] |

Notes:

- **NFHS bench-device rule.** NFHS Rule 1-19 was revised for 2013-14 to allow electronic devices for coaching and gathering statistics. It bars voice communication with players on the court and using devices to review officials' decisions [59]. State associations may differ, and the current wording is unverified. This makes a coach-facing live dashboard on a tablet plausible for high school.
- **NCAA bench-device rule.** NCAA men's basketball historically did not allow video in the bench area, with conference experiments since 2019 [60]. By March 2026, video coordinators were reviewing plays on iPads for coach's challenges [58]. (Unverified: both cited URLs were dead when re-checked on 2026-10-20; [58] returns 404 and [60] redirects to the NCAA news index.) Confirm the current NCAA rule before selling in-game video to college benches.

---

## 2. Existing products: what they do and what they cost

### 2.1 Professional / league tier (not our market; shows what "great" looks like)

| Product | What it does | Capture | Cost / access | Notes for us |
|---|---|---|---|---|
| **Sony Hawk-Eye + Sportradar (NBA)** | League optical tracking from 2023-24. Replaced Second Spectrum's single-point "center of mass" tracking with 3D tracking [4]. 29 skeletal points per player, 14 cameras per arena, under 1 s latency (Sportradar, announced 2024-10-22 for 2024-25); the article says the advances extend to Sportradar's broadcast and sports-betting partnerships [5]. The NBA x AWS release also gives 29 data points per player [6] | 14 fixed cameras per arena [5] | League-wide contract; not for sale to amateurs | Shows that pose-level data is now the pro standard. We cannot match it with 1-2 cameras, so our product must be robust at "coarse" tracking |
| **NBA x AWS "Inside the Game" (2025-26)** | AWS became the NBA's "Official Cloud and Cloud AI Partner" (announced 2025-10-01). New stats: Defensive Box Score (primary defender, ball pressure, switches, doubles), Shot Difficulty / xFG%, Gravity (processes tracking 60x/s). Play Finder runs on Amazon Bedrock + SageMaker and includes a real-time alert system that gives commentators historical context [6]. Gravity is computed only when all 5 players and the ball are in the frontcourt, and not during rebound situations [7] | Hawk-Eye tracking (29 points/player) [6] | League/Prime only | **The closest pro analogue to our "insight for commentators" layer.** Our Claude color commentary is the amateur equivalent. Gravity and defensive attribution need all-10-player tracking, which is hard from one camera |
| **Second Spectrum (Genius Sports)** | Former NBA tracking provider. In 2023 named NBA League Pass augmentation provider and team analytics provider, serving all 30 teams [8]. WNBA in-arena 3D pose and ball tracking from 2024 [9]. Genius bought Second Spectrum in 2021 | Multi-camera | Enterprise | 2025-26 status of the NBA roles is unverified |
| **Sportradar official NBA data** | Deal announced 2021-11-17: eight years from 2023-24 to 2031. Sportradar is the exclusive data provider to the NBA, WNBA and G League and the authorized global distributor of official NBA/WNBA betting data, and it distributes live game video to international betting operators. The NBA received warrants for 3% of Sportradar at $0.01 per share [10]. The 2018 non-exclusive US betting-data deals with Sportradar and Genius Sports [11] came before this deal. Whether Genius keeps any NBA betting-data rights from 2023-24 is unverified | n/a | Enterprise | Why betting feeds are closed to us |
| **NBA automated out-of-bounds (announced May 2026)** | Commissioner Silver said on 2026-05-27 that a whole category of objective boundary calls will become automatic, like tennis line-calling. He gave no date beyond "fairly quickly". Fouls and contact stay with referees [55] | "Cameras lined around the court" [55]. That it will be Hawk-Eye is our inference | League | Officiating automation is coming at the top. At amateur level we should only produce review clips, not calls |
| **Kinexon** (wearables) | UWB local positioning: 15.4 g waistband sensor, about 12 RF receivers (unverified; [61] returned 403); sub-10 cm accuracy, 20 ms latency, 300+ metrics, positions at 20 Hz [62]. Over half of NBA teams use it in practice (unverified; [63] now redirects away) | Wearables | Enterprise | The bar for trainer-grade load. NBA CBA (2017): team-requested wearable data limited to health/performance and tactics, banned in contract negotiations, up to $250k fine (unverified; [64] returned 403) |

### 2.2 Team video and stats services (amateur and college: our real competitors and partners)

| Product | What it does | Price (as published) | AI or human | Live? |
|---|---|---|---|---|
| **Synergy Basketball (Sportradar)** | Possession-logged video libraries (college, HS; NBA/WNBA for an extra fee), 500+ breakdown categories, "Shot Quality" and "Player Roles" insights; breakdown in under 24 h [13]. Sportradar acquired Synergy in 2021; Synergy had league-wide relationships with the NBA and all NCAA DI basketball [12] | HS bundles: Access Only / High School / High School Pro. 12-month non-prorated contracts; **no public prices** [13] | Not stated on the page; whether logging is human or AI is unverified [13] | No |
| **Hudl Assist** | Analysts break down uploaded film into box score, shooting splits, lineup and plus-minus, points per possession, shot charts and similar. Hudl says it tagged 150k+ HS games last season [15] | **$900** (your games, Standard) / **$1,300** (Express) / **$1,500** (your games + scout games, Standard) / **$2,100** (Express) per team per season; max 3 games/day; one team; current season only [14] | Human analysts ("our team of professional analysts", staffed 24/7) [15]. The pricing page itself does not say [14] | No (queue) |
| **Hudl (school packages + Focus cameras)** | Video platform + Focus auto-tracking cameras (Focus Indoor: permanently mounted, AI player tracking, livestream [18]. The 180° panoramic view, ball tracking and single ethernet cable are not on the cited page and are unverified) | HS program: Silver+ $1,500, Gold+ $2,500, Platinum+ $4,000 per year [16]. Club basketball team: $400 / $1,000 / $1,600 per year [17]. One Utah district quote: $10,000 initial + $10,000/yr for Focus Flex, hardware rented [19] (unverified: the PDF's text could not be extracted on re-check) | Camera AI for framing only | Streams live; stats post-game |
| **Hudl x Balltime** | **Correction to the brief: Balltime is volleyball, not basketball.** Hudl completed the acquisition on 2025-02-06, for AI volleyball stats (indoor and beach; jump height, ball trajectory, 3D views) [20]. 125k athletes / 12k teams claimed [20][21] | n/a | AI | n/a |
| **SportsVisio** | Box score and auto highlight reels (all plans). Shot charts and "AI game summary" on Elite and Coach only, not Pro; player minutes on Elite. "AI Assistant Coach" and opponent scouting on Coach [33]. $3.2M round (2025-06-20) took the total raised to $9M; Sony Innovation Fund joined as an investor [34] | Pro $149/mo and Elite $189/mo (5 games/mo each); Coach $750/season (20 games). Overage per game: Pro $29.80, Elite $37.80, Coach $37.50. Leagues & Tournaments tier by quote (add-on games 10% off); API access appears to be on that tier only (inferred from the comparison table) [33] | AI company; the pricing page does not label the box score "AI" or state human verification [33] | No (post-game) |
| **PlayVision (YC Fall 2025)** | Upload practice and game film for advanced stats on own players, opponents and recruits; automated scouting reports; player-ranking indexes built from 40+ metrics; national player-ranking database covering 10 years; live for 2025-26 [41]. Co-founder Anish is a former Cal Athletics scout [41]. The "transfer-portal database" and "full game in about 1-2 h" claims are not on the YC launch page (unverified). **The YC company page at this URL now shows "Bron" ("TikTok for Sports Betting"), Fall 2025, with the same founders, so PlayVision has likely pivoted [42]** | Not public | AI | No |
| **Hoopsalytics** | Lower-cost tagging and stats; claims to be about half Hudl's price (competitor marketing) [116] | not verified | Self-tag + tools | No |
| **Superstat** (2026) | Footage to stats and event timelines for amateur sport; $3.5M pre-seed (July 2026) [118] | n/a | AI | No |

### 2.3 Capture and streaming (our ingest sources, not competitors to fight)

| Product | What it does | Price | Notes |
|---|---|---|---|
| **Pixellot** (+ PlayOn / NFHS Network) | AI auto-production cameras; "Show" streams 19 sports with graphics, ads, commentary and highlights [22]. PlayOn partnership extended 5 years to 2030 with a revenue-share "AI-as-a-Service" model (2025-11-06) [23]. Raised $35M in late 2025 ($15M from PSG Equity + $20M venture debt; reported 2025-11-23), with **AI-generated commentary "in the voice of a friend or family member"** among recent and upcoming releases [24][25]. NFHS Network streams basketball automatically with Pixellot [27] | Third-party estimate: about $199/mo per team for basketball (unverified) [26] | **Most direct threat** if Pixellot ships good AI commentary. No technical details or accuracy data published. Voice cloning of relatives raises the voice-likeness issues in §5.7. Also the cautionary privacy case (Basketball Australia HoopsTV: announced 2026-10-12, go-live expected November 2026) [28][29][30] |
| **Veo** | Veo Cam 3 (Wi-Fi / ethernet / 5G livestream), Veo Go (turns two iPhones into a match camera, 2025); Starter/Team/Club/Veo Go/Enterprise plans; Coach Assist chat over match data (requires Analytics 2, so effectively soccer-only); auto event detection **only for soccer, rugby, lacrosse**; Analytics 2 is soccer-only [31][32] | Prices not shown on the official page [31] | Basketball is a secondary sport for Veo |
| **BallerCam (BallerTV)** | iPhone AI camera (needs iPhone 13 Pro or newer; auto pan, zoom, reframe), live scoring, scoreboard overlays, play tagging, highlight clipping, streaming; launched the week of 2025-07-24; trained on data from over 2 million youth-sports games [43] | $299 hardware or $399 with tripod and sling bag; video storage subscription from $99/yr [43] | Strong in AAU; ingest candidate |
| **Hooper** | Phone-recorded games/shootarounds to auto stats, shot tracking, AI clips; full court uses 2 phones, one per half, linked into one session [44] | Free (2 h recording/mo); Pro $11.99/mo (10 h); Team $29.99/mo (10 shared h, unlimited seats) [44] | Consumer; SoftBank partnership for Japanese youth basketball [44] |

### 2.4 Consumer training apps

| Product | What it does | Price |
|---|---|---|
| **HomeCourt (NEX Team)** | Phone-camera shot tracking and drills; early tier was 300 shots/mo free, $7.99/mo unlimited [37]; team tier reported at $1.99/player/mo (third-party) [38] | Freemium |
| **Ball AI** (developer Automatic AI, LLC) | Phone-camera shot tracking: makes/misses, shooting %, court location, release angle, release time, dribble speed; Apple Watch listed as a compatible device. The App Store page says "Trusted by 300K+ hoopers" (not 50k+) [39]. $475k raised [40] | Free with in-app purchases (Pro monthly $5.99-$14.99, yearly $39.99-$79.99 per App Store listing) [39] |
| **ShotTracker** | Sensor ball + player sensors + court sensors for automatic real-time stats; installed in college conferences; installation about $45,000 (2021) [35][36] (unverified: both sources blocked automated fetch on re-check) | Hardware install |

### 2.5 AI commentary, narration and accessibility (closest to our core)

| Product / deployment | What it is | Live? | Grounded in data? |
|---|---|---|---|
| **WSC Sports x NBA (announced 2025-04-06)** | GenAI voice-over for a 2-minute highlight clip per game, posted within minutes of the final whistle, in French, Portuguese and Spanish (for France, Brazil, Spain); scripts from WSC's sports LLM plus fine-tuned TTS. Vendor completion figure: the headline says "more than 75%", the body says an average of 75% of each video is watched [45] | No (post-game clips) | Game context and stats |
| **IBM x Wimbledon "Match Chat" (2025)** | Q&A assistant during live ladies' and gentlemen's singles matches, built on watsonx Orchestrate with IBM Granite among its models [46]. "AI commentary since 2023" is not in the cited page (unverified) | Near-live Q&A | Yes (match data) |
| **NBC "AI Jim Fagan" (2025-26)** | AI recreation of a late narrator's voice for NBA coverage and promos; Fagan's family gave NBC Sports permission [47] | Scripted segments | n/a. Shows estate-permission practice for voices |
| **Pixellot AI commentary** | Announced in 2025 funding coverage; no technical or accuracy disclosures [24][25] | Unknown | Unknown |
| **Sportway** | €20M (2026-06-23) for AI-produced live video and data for grassroots and youth sport: about 2,500 AI camera systems in 21 countries, 250,000+ live events a year, event recognition for highlights [48] | Yes (production) | Commentary not mentioned [48] |
| **FootyHub TV** | Independent YouTube channel whose AI hosts call all 104 matches of the 2026 World Cup live [49] | Yes | Unknown; soccer |
| **OneCourt** | Haptic display + generative audio description from real-time league data for blind/low-vision fans. Trail Blazers pilot 2023-24, then at every home game. Fans rated their comprehension 74.3% higher than with radio alone; the blog does not say who measured this (Microsoft blog, 2025-01-14) [50]. Denver Nuggets debut Jan 2026 [51] (unverified: [51] returned 403) | Yes | Yes (league feed) |
| **Pacers x MindsEye (2025-26)** | **Human** live audio description at 20 home games via the LiveVoice app [52] | Yes | Human |
| **Action Audio (AKQA, Hawk-Eye)** | Hawk-Eye ball-tracking data rendered as sound for blind/low-vision tennis fans (AO since 2021; Wimbledon Centre Court feed with a visual overlay) [53][54] (unverified: [53] returned 403 and [54] returned no content on re-check) | Yes | Yes (tracking) |

### 2.6 Gap analysis: what CourtVision uniquely does

| Capability | Pro tier | Amateur incumbents | CourtVision |
|---|---|---|---|
| Video capture / streaming | yes | **yes** (Pixellot, Hudl Focus, Veo, BallerCam) | Ingest any RTSP/RTMP/file; no hardware business |
| Post-game box score | yes | yes: human (Hudl Assist, Synergy) or AI (SportsVisio, PlayVision) | yes |
| **Live box score from video alone** | yes (multi-camera) | Partial: ShotTracker uses hardware; nobody found doing it camera-only and live | **yes (core)** |
| **Live spoken play-by-play grounded in detected events** | Human broadcasters; AI assists commentators (NBA Play Finder alerts [6]) | Announced, not shown (Pixellot) | **yes (core differentiator)** |
| Uncertainty-aware narration (won't say what it isn't sure of) | n/a | none found | **yes (Jev probabilities + gating)** |
| Live audio description / captions | OneCourt and human AD in NBA arenas only | none found | **yes (same pipeline)** |
| On-device / footage stays local | n/a | Cloud-first (Pixellot footage reportedly transferred or stored overseas [28]) | **yes (M3 Pro edge)** |
| Coach-facing live tactical alerts | Team analysts | none found at amateur level | yes (P1) |

**Positioning:** we sit on top of existing cameras. Treat Pixellot/NFHS, Hudl Focus, Veo and BallerCam as ingest sources and partners. Our product is the **live intelligence and voice layer**: live stats, grounded narration, audio description and captions, and coach alerts.

---

## 3. How the components map to use cases

- **Event engine** carries U1, U2, U8-U11 and U14. Most important: shot attempt, make/miss, 2 vs 3, free throw, possession change, rebound (O/D), turnover, foul (from whistle and clock stop plus referee signal if visible), substitution, timeout, period boundaries.
- **Jev** (TypeSafe; per local SDK ground truth `docs/research/sdk_ground_truth.md`: `Choice` 1-255 labels with per-label probabilities and confidence, `Score` 1-10 ordered levels with probabilities, `Noul` = single P(yes) with no confidence field; consistent with secondary docs [2]) handles bounded judgments with thresholds:
  - Noul: "Did this attempt go in?" (as a check on the rim-zone detector), "Is this moment narration-worthy?", "Is the ball dead?"
  - Choice: event type among candidates; shooter among visible jersey numbers; play type for U2 tagging.
  - Score: shot contest level (0-4), highlight quality (0-4), alert urgency.
  - OpenRouter's community guide says Jev produces no free-form text or reasoning, and that on OpenRouter, text requests over about 32,000 bytes and every image request run in a slower, less accurate long-context mode. It also says probabilities differ between models, so thresholds should be re-picked per model [1].
  - **Correction from TypeSafe's own docs (docs.typesafe.ai "Models", which applies to the direct `typesafe-sdk` / `api.typesafe.ai` path we use) [121]:**
    - Context limits are **64k tokens per request and 32k tokens per question** (state plus any one question). These are token limits, not bytes.
    - Long context is **off by default**. It is enabled per request with the header `TypeSafe-Danger-Zone-Long-Context: true`, which raises both limits to 262k tokens and is slower and less accurate.
    - Listed input for Jev 1.14/1.13 is **text only**. Images and Box questions need the header `TypeSafe-Danger-Zone-Images: true` and are labeled not production-ready ("Dreamer-ready") [122].
    - Versioned ids are `jev-1.14.0` (available from 2026-10-14) and `jev-1.13.0`. Aliases: `jev-latest` points to `jev-1.13.0` (the SDK default) and `jev-preview` points to `jev-1.14.0`. TypeSafe says that if you tuned thresholds against one version, you should send that version's id instead of the alias [121].
  - **Implication:** keep live Jev state compact text, well under the 32k-token per-question limit (a 32 KB budget is a safe proxy). Never send the long-context or images headers on the live path. Pin a versioned id such as `jev-1.13.0`, not `jev-latest`. Store per-version thresholds in config. Vendor latency of 70-500 ms per call comes via secondary sources (unverified) [120].
- **Claude** narrates and summarizes only from the event log plus the live box score (no free recall of player facts). It also answers Q&A (U10) and writes coach summaries (U1) and scouting text (U5).

---

## 4. Value vs feasibility: prioritized use cases

Scoring (our assessment):
- **Value** (1-5): willingness to pay × frequency × how weak the alternatives are.
- **Feasibility** (1-5): accuracy reachable on M3 Pro with 1-2 fixed or panning cameras in an amateur gym, plus data availability.
- **Legal risk**: L/M/H.
- Priority: P0 = MVP, P1 = next, P2 = later, X = not pursued.

| Rank | Use case | Users | Value | Feasibility | Legal risk | Priority | Rationale |
|---|---|---|---|---|---|---|---|
| 1 | **Live auto box score + event log** (team-level scoring first, player attribution by jersey second) | U1, U8, U10, U11, U14 | 5 | 4 | M | **P0** | Foundation for everything else. Team-level made/missed/points is very feasible (rim is static, ball near rim is a strong cue). Player attribution is harder (jersey occlusion), so ship team-level first with player-level marked "provisional" |
| 2 | **Grounded live narration** (play-by-play + light color) for commentator-less streams | U8 | 5 | 3 | M | **P0** | Clearest gap (§2.6). Feasible only if narration is gated by event confidence: say "a three from the left wing... it's good" only above threshold, and fall back to neutral phrasing ("shot goes up, rebound Blue") otherwise |
| 3 | **Audio-description mode + live captions** | U9, U8 (compliance) | 4 | 4 | L | **P0** | Small extra cost on top of #2. A different prompt emphasizes spatial and possession detail. Supports WCAG 1.2.4 live captions (AA) for public-entity streams [113][112]. Strong mission and press value; precedents OneCourt, MindsEye, Action Audio [50][52][53] |
| 4 | **Post-game coach report** (box score, shot chart, runs, lineup +/-, possession clips) | U1, U2 | 5 | 4 | L-M | **P0** | Competes with $900-$2,100/season human tagging [14] and $149-$189/mo AI [33]. Batch processing allows heavier models and a human-correction UI |
| 5 | **In-game coach alerts** (foul trouble, opponent run, timeouts, hot zones) | U1 | 4 | 3 | L | P1 | Counts must be right, so we need player-level foul attribution, which is hard. NFHS allows devices for coaching/statistics [59]; NCAA bench rules vary [58][60]. Jev Noul gates interrupt-worthiness |
| 6 | **Second-screen Q&A + recap** | U10, U11 | 3 | 4 | M | P1 | Cheap once the event log exists; Wimbledon Match Chat precedent [46]. Must refuse questions the log cannot answer |
| 7 | **Auto-tagged clip library / video coordinator search** | U2 | 4 | 3 | L | P1 | Play-type classification (PnR, zone) is hard from one camera; start with possession segmentation + event tags |
| 8 | **Parent / player highlight reels + per-player stats** | U11, U12 | 4 | 4 | **H** | P1 (consent-gated) | High demand, but involves minors, consent, NIL and recruiting rules. Ship only behind explicit per-player opt-in (§5.4) |
| 9 | **Opponent scouting reports** | U5, U1 | 4 | 3 | M | P1 | Needs legitimate access to opponent film (exchange platforms). Claude writes tendencies from our event logs. Selling to NCAA DI programs triggers scouting-service approval rules [114][115] |
| 10 | **Analytics export** (tracks, possessions, shot features) | U3 | 3 | 4 | L-M | P1 | Cheap and builds credibility; include confidences |
| 11 | **Player development** (shot zones, release time, pose cues) | U4 | 3 | 3 | M (pose may be biometric under COPPA: "gait patterns" [89]) | P2 | Consumer apps own this [37][39] |
| 12 | **Officiating review clip packs** | U7 | 2-3 | 3 | M | P2 | Real but niche market (ACC RefScore [56]). Never present AI judgments as calls |
| 13 | **Athletic-trainer load metrics** | U6 | 3 | 2 | H (health data of minors) | P2 | 1-2 camera optical estimates of accelerations are noisy compared with UWB (sub-10 cm / 20 ms [62]). Offer only coarse distance/minutes with error bars |
| 14 | **League/tournament multi-court batch stats** | U14 | 4 | 3 | M | P2 | Business-development play after P0 quality is proven; compute scaling beyond one M3 Pro |
| 15 | Narration of NBA/NCAA broadcasts | fans | 4 | 4 | **H** | **X** | Rights (§5.1, §5.2) |
| 16 | Betting / fantasy live data feed | U13 | 5 ($) | 3 | **H** | **X** | §5.6 |

---

## 5. Legal and data constraints

### 5.1 Copyright and licensing of game footage ("training off live games")

- **Games are facts; broadcasts are copyrighted works.** In *NBA v. Motorola* (2d Cir. 1997), real-time transmission of scores and facts taken from broadcasts was not hot-news misappropriation. The underlying game is not copyrightable, but the broadcast is [67]. **Implication:** the events we detect (scores, makes, fouls) are facts we may describe. The pixels of someone else's broadcast are not ours to copy, store or train on.
- **NBA.com Terms of Use** (last updated 2026-07-13):
  - Video footage and game-action photographs are NBA-entity IP.
  - Downloads are allowed to a single computer for personal, non-commercial use only.
  - Commercial reuse needs written permission [65].
  - The terms do not mention AI training, so it falls under the general reuse restriction.
- **NCAA championships:**
  - Footage captured inside NCAA Championship venues is NCAA copyright.
  - Most digital-highlight uses must be licensed through Veritone (TNT Sports holds the rights), however the footage was obtained.
  - Member schools get only limited editorial, non-commercial use: during live play, ≤12 clips per game of ≤30 s each plus ≤8 GIFs totaling ≤10 s, only on official school social accounts and with a call to tune in, and no live streams; after live play, 6 min aggregate with no clip over 30 s; use allowed during the championship and for 30 days after.
  - The policy does **not apply** to DI Men's and Women's Basketball Championship games. Extra digital/social rights for those are available only through optional TNT Sports/NCAA/Veritone licensing [68]. Men's March Madness teams were restricted on X, TikTok and Snapchat [69].
- **High school streaming rights.** These are set by state associations, often with NFHS Network exclusivity for postseason:
  - Kansas: KSHSAA uses the NFHS Network as its exclusive postseason streaming partner; regular-season rights are set by individual schools [70]. The start year of 2018 is unverified.
  - Georgia: the NFHS Network has exclusive rights to GHSA postseason events. If it declines an event, other platforms need GHSA office approval and pay a per-game rights fee [71].
  - Delaware: no streaming without DIAA approval [72] (unverified: [72] returned 403 on re-check).
  - New York: NFHS Network is the exclusive streaming home of NYSPHSAA championships under a 10-year deal from 2018-19 [73].
  - **Implication:** regular-season games filmed by the school are the realistic data source. Postseason processing needs association permission.
- **YouTube.** YouTube's terms ban scrapers and other automated access (except public search engines that follow robots.txt or have permission) and appear to prohibit downloading content for AI training without permission [74]. The CEO's "clear violation" remark is not in [74] (unverified). **Do not build a training set by ripping YouTube highlight channels.**
- **Fair use for AI training (US, 2025).**
  - *Bartz v. Anthropic* (Alsup, 2025-06-23): training was "exceedingly transformative" and fair use, and scanning purchased print books was fair use. Anthropic had "no entitlement to use pirated copies for its central library", so a trial was set on those copies [75][76].
  - *Kadrey v. Meta* (Chhabria, 2025-06-25): fair use on that record, but the judge warned about market dilution [75][76].
  - The US Copyright Office's Part 3 pre-publication report (2025-05-09) found training is often transformative but that market harm and other factors can defeat fair use [77].
  - **Implication:** how footage is acquired matters as much as how it is used. Contract (terms of use) claims are separate from copyright.
- **Recommendation:**
  - (a) Build an **owned/consented footage corpus**: partner teams sign a data-use agreement; per-game releases cover recording and model training; minors' releases are signed by parents (§5.4).
  - (b) Use **public datasets only after checking their license.** DeepSportradar-v1 has basketball ball-3D, calibration, instance-segmentation and re-ID tasks [99]. The official instance-segmentation challenge README says its Kaggle data is **CC BY-NC-ND 4.0**, with code under Apache-2.0 [119]. A third-party listing says CC BY-NC-SA 4.0 [100]. Either way it is non-commercial (and under ND, no derivatives), so use it for research/eval only. Licenses for the other three challenge datasets were not found; the GitHub org shows no license on the ball-3D and calibration repos.
  - (c) Use NBA/NCAA broadcasts **at most for qualitative internal viewing**, never for stored training data or shipped weights.

### 5.2 League stats APIs (`nba_api` / stats.nba.com)

- `nba_api` is MIT-licensed client code [66]. The MIT license covers the code, **not the data**. Its README points to NBA.com's terms (https://www.nba.com/termsofuse) [66].
- NBA.com Terms §9 limits "NBA Statistics" to legitimate news reporting or private, non-commercial use [65]. It specifically bars use:
  - with any sponsorship or commercial identification;
  - with **any gambling activity (including legal)**;
  - with **fantasy games or any other commercial product or service**;
  - with products presenting **live, near-live, real-time or archived play-by-play**;
  - with a regularly updated statistics database without prior consent.
  - Any permitted use needs prominent attribution to NBA.com [65].
- **Implications:**
  - Do **not** use `nba_api` data in any shipped feature, including narrator "fun facts" about NBA players and evaluation sets used for commercial claims.
  - If internal R&D uses it (for example, sanity-checking event-engine logic against NBA play-by-play), keep it in a quarantined `research/` area and never in product databases. Get counsel's view on whether internal R&D at a for-profit counts as "commercial" (open question).
  - If we ever need NBA data, license it (Sportradar/Genius).
  - Separately, EU database right can protect live match-data collections. *Football Dataco v Sportradar* (CJEU 2012 + UK CoA) went against a data provider that reused another's live data [98].

### 5.3 Biometric privacy (BIPA, CUBI, others) and how tracking relates to it

- **Illinois BIPA:**
  - "Biometric identifier" means retina or iris scan, fingerprint, voiceprint, or scan of hand or **face geometry**. Photographs and physical descriptions (height, weight, etc.) are excluded [78].
  - §15(a): a public retention and destruction schedule; destroy data when the purpose is met or **within 3 years** of the last interaction. The policy must exist when data is first possessed (*Mora*) [81].
  - §15(b): written release before collection.
  - §15(c): no selling or profiting from the data [81].
  - Damages: $1,000 (or actual damages) per negligent violation / $5,000 per intentional or reckless violation [80][82]. The 2024 amendment (P.A. 103-0769, from SB 2979, signed 2024-08-02) limits recovery under §15(b) and §15(d) to one per person when collection uses the same method, and adds "electronic signature" to "written release" [79][80]. (The signing date was not re-confirmed: ilga.gov blocked automated fetches.)
  - The 9th Cir. (*Zellmer v. Meta*, 2024) held identifiers must be capable of identifying a person. (Unverified: [82] does not discuss Zellmer.)
  - Sports venues have been sued over facial recognition: the Blackhawks [83], and the Cubs (*Berta v. Chicago Cubs*, N.D. Ill. No. 25-CV-11138, which ended when the plaintiffs withdrew their complaints) [82].
- **Texas CUBI:**
  - Covers retina/iris, fingerprint, voiceprint, hand or face geometry captured for commercial purposes [84].
  - Enforced only by the Attorney General (no private right of action), with civil penalties of **up to $25,000 per violation** [86]. Meta settled for $1.4B (announced 2024-07-30) [85][86].
- **Washington:** excludes photographs, video or audio recordings and data generated from them [87].
- **Colorado:** HB24-1130 adds biometric duties to the Colorado Privacy Act. The final fiscal note says it was signed 2024-05-31 and takes effect 2024-08-07 [88]. The commonly cited operative date for the biometric obligations, 2025-07-01, is not in the fiscal note (unverified).
- **COPPA amendments** (published 2025-04-22; effective about 2025-06-21; compliance 2026-04-22): "personal information" now includes "a biometric identifier that can be used for the automated or semi-automated recognition of an individual", with examples including **fingerprints, handprints, retina and iris patterns, genetic data, voiceprints, gait patterns, facial templates and faceprints**. The final rule dropped the proposed "data derived from voice data, gait data, or facial data" [89][90]. This applies to services directed to, or knowingly collecting from, children under 13. AAU and youth leagues have many under-13 players.
- **Implications for CourtVision (design defaults):**
  1. **No face recognition, ever.** Do not compute or store face embeddings. If faces are blurred in stored training frames, use a face detector only; that produces boxes, not a geometry template.
  2. **Player identity** = jersey OCR + coach-entered roster mapping, not biometric re-ID.
  3. **Appearance / re-ID embeddings** (needed for tracking through occlusion) stay in RAM, are scoped to one game, are never persisted, and are destroyed at game end. Document this in a public retention policy anyway (cheap BIPA §15(a) hygiene).
  4. **Pose/skeleton sequences** may be "gait patterns" under COPPA if they can be used for automated recognition of an individual (the rule's qualifier). For under-13 users, store only court x,y tracks and derived events, not per-frame keypoints, unless there is verifiable parental consent.
  5. **No audio capture of people by default** (see wiretap, §5.4).

### 5.4 Minors, recording consent, NIL, schools

- **FERPA.**
  - A video is an education record only if it is directly related to a student **and** maintained by the school.
  - Footage of public events like games is generally not "directly related" unless it focuses on a student (injury, discipline, etc.).
  - Schools often designate such footage as directory information or get parental consent [91].
  - **Implication:** when a school is our customer, our DPA must say we are acting for the school ("school official"). Injury clips must be handled as potential education records (restricted access).
- **Consent culture for youth filming.**
  - Basketball Australia guidance calls for prior written parental approval covering what is captured, how it is used, storage and deletion, and withdrawal [117].
  - The 2026 Pixellot/HoopsTV rollout drew criticism over filming without explicit, informed consent (participants must opt out), reported transfer or storage of footage overseas without consent, children's game video reachable from a newly created account, and a new mandatory registration fee [28][29][30].
  - Basketball Australia says access is authenticated, participants can opt out of filming, and there is no facial recognition [29]. A pre-launch report says players would be matched by jersey number, with participant blurring planned for a later phase [30]. That is the same identity approach as ours.
  - **Implication:** opt-in for any per-player output (named stats, reels); team-level outputs default on; per-player "do not identify" flag; data stays on-site by default. *Gap pass:* the only exceptions are pseudonymised Jev states and Claude FactPackets (no names), under a customer DPA that lists both vendors as subprocessors (§9.1).
- **Audio / wiretap.** Thirteen US states are listed as all-party-consent states for recording conversations: CA, CT, DE, FL, IL, MD, MA, MT, NV, NH, OR, PA and WA [93]. **Implication:** disable microphone capture of benches and huddles. Use only crowd/ambient level if needed, or no audio. Our narration is generated, not recorded.
- **High-school NIL.** Associations in 41 states + DC allow some HS athlete NIL compensation. All bar naming the school or team in NIL deals. Nine require some notification, and five set deadlines: AZ, DE, GA, PA, VA [92]. **Implication:** recruiting reels must be the athlete's (or family's) own content, with no school marks in monetized versions. Do not monetize minors' likeness in our own marketing without separate releases.
- **Recruiting services.** NCAA DI basketball and football programs may subscribe only to services approved by ECAG (Bylaws 13.14.3.1 and 13.14.3.1.1 for video-only services). Approvals expire May 31 each year, and approved services must submit subscriber lists, invoices and sample reports [114]. The specific rules (one subscription per institution, the same publicly listed rate for all, video limited to regularly scheduled HS/prep/2-year contests not arranged by the institution) are unverified: the [114] URL now redirects to a summary page that does not reproduce the bylaw text, and [115] returns 404. **Implication:** if we sell a prospect database or reels to college coaches, apply for NCAA approval and price uniformly.

### 5.5 GDPR / EU (if we sell in Europe)

- Video is personal data when people are identifiable. **Article 9 special-category rules apply only when biometric processing is used to uniquely identify** a person; raw footage is not automatically biometric (EDPB Guidelines 3/2019) [94]. When Art. 9 applies, explicit consent is generally needed for private use, plus a DPIA and data minimization [94][95].
- Children's consent for information-society services: 16 by default, member states may lower it to 13 (GDPR Art. 8) [95].
- **EU AI Act** Art. 5 prohibitions (in force 2025-02-02) include emotion recognition in **workplaces and educational institutions**, and biometric categorization to infer sensitive traits [96]. **Implication:** do not ship "body language / frustration / effort" inference for school teams in the EU (feature-flag it off by region).
- Athletes have asserted GDPR rights over performance data commercialized by betting and data firms ("Project Red Card", UK; pre-litigation, 2,000+ players claimed by 2024) [97] (player count unverified: the Lewis Silkin page returns 404 and the LSR page was not re-checked). **Implication:** per-player data products in the EU need a lawful basis and data-subject request handling.

### 5.6 Fantasy and betting feeds

- **League exclusivity:**
  - NBA: from 2023-24 to 2031, Sportradar is the exclusive official data provider and authorized global distributor of official NBA/WNBA betting data [10]. The 2018 non-exclusive US deals with Sportradar and Genius [11] came before this.
  - NCAA: Genius Sports is the exclusive distributor of official NCAA data to licensed sportsbooks for March Madness and all other postseason tournaments through 2032 (announced 2025-04-25), with integrity protections through its Authorised Gaming Licensee programme [103]. The requirement that sportsbooks drop certain "risky" bet types is unverified: [103] does not list bet types and [104] returned 403.
- **Official-data mandates:** Illinois requires official league data for in-play bets, subject to "commercially reasonable" terms [101]. Tennessee dropped its mandate [102].
- **Integrity climate:**
  - October 2025: federal charges against Terry Rozier (alleged insider-info betting scheme) and Chauncey Billups (rigged poker games backed by Mafia families) [106]. The "30+ defendants" total is not in [106] (unverified).
  - The NBA asked sportsbooks to pull certain props, e.g., unders on two-way/10-day players [107] (unverified: [107] returned 403 on re-check).
  - The NCAA asked state regulators to ban college player props; several states (e.g., OH, MD, LA, VT) ban them [105] (unverified: [105] returned 403 on re-check).
- **Youth / high school:** Virginia (Va. Code § 58.1-4128(B)-(C)) makes wagering on, or accepting wagers on, a youth sports game a Class 1 misdemeanor [108]. The NFHS urges states to exclude high school contests from betting [109]. Leagues use entry and ticket terms to bar "courtsiders" (data scouts and similar in-venue collectors) to curb unauthorized betting markets. Basketball Australia's 2018 policy covers all basketball in Australia; the NBL is not named [110].
- **Recommendation:**
  - Do not sell or expose low-latency player-level feeds.
  - Add terms of service that prohibit using outputs for wagering.
  - Rate-limit and authenticate any API.
  - Offer an **optional broadcast delay** (e.g., 30-60 s) on public live stats for youth games.
  - Never show odds or win-probability "betting-style" overlays on youth content.

### 5.7 Narrator voice and likeness

- Tennessee's ELVIS Act (effective 2024-07-01) adds **voice, including simulated voice**, to publicity rights, and creates liability for tools whose primary purpose is producing an identified person's voice without authorization. Enforcement is civil (injunctions, damages, profits) [111]; criminal enforcement is not described in [111] (unverified). NBC's AI Jim Fagan voice was a deliberate, presented recreation made with the Fagan family's permission [47].
- Pixellot's planned AI commentary "in the voice of a friend or family member" [24] shows that voice cloning is entering amateur products. If we ever offer custom voices, require the voice owner's recorded consent.
- **Implication:** use licensed stock TTS voices. Never clone or imitate identifiable announcers. Do not synthesize player voices.

### 5.8 Accessibility law as a demand driver

- DOJ ADA Title II web rule: WCAG 2.1 AA for state and local governments, including **public schools and universities**. An interim final rule of 2026-04-20 moved compliance to **2027-04-26** (population ≥50k) and **2028-04-26** (smaller entities and special districts) [112].
- WCAG 2.1 AA includes **1.2.4 Captions (Live)** and **1.2.5 Audio Description (Prerecorded)** [113]. Live audio description is not an AA requirement, but it is a strong differentiator.
- **Implication:** emit (a) a WebVTT/CEA-608-compatible caption stream from narration text, and (b) a separate AD audio track. Archived game VODs can carry the AD track, which helps public-school customers meet 1.2.5.

### 5.9 Practical data and legal recommendations (checklist)

1. **Own footage first.** Begin with adult rec leagues (no minors; simple releases), then HS/club teams under a school or club DPA with parental opt-in for per-player features.
2. **Separate "processing" from "training" consent.** Many customers will agree to live processing but not training. Store a per-game `training_allowed: bool`.
3. **Store tracks, not pixels.** Default retention: raw video = customer's choice (default delete after N days on-device); training frames = face-blurred, audio stripped; tracks and events = retained.
4. **No biometric identification.** No face recognition, re-ID embeddings scoped to one game, jersey + roster mapping, and a published retention policy (BIPA §15(a)).
5. **Region flags:** EU (disable affect inference, Art. 8/9 consent flows); Illinois/Texas (no biometrics stored); under-13 (COPPA: no keypoint storage without verifiable parental consent).
6. **No league data or footage** in product: no `nba_api`, no NBA/NCAA broadcast frames, no YouTube rips.
7. **Anti-wagering terms + optional delay**; no prop-like per-player public feeds for minors.
8. **Licensed TTS voices only.**
9. **Postseason HS games:** check the state association's media policy before processing or streaming [70][71][72][73].
10. **Audio:** off by default (wiretap).

---

## 6. Minimum viable feature set per user type

Targets are proposed acceptance criteria for the MVP evaluation harness (our numbers, to be validated on our own labeled games).

### 6.1 Broadcaster / streamer with no commentator (U8): P0
- Ingest RTMP/RTSP/HLS or file. *Gap pass (2026-10-20):* by default only from sources the customer controls (OBS, iPhone, BallerCam, the school's own camera pushing to us). Pixellot/NFHS Network, Hudl Focus and Veo need a written partner agreement first; see §9.2.
- Live score and clock sync (manual scorebug override is required; OCR of the venue scoreboard if visible).
- **Grounded play-by-play**: Claude speaks only events above the Jev/event-engine confidence threshold. Low-confidence events use neutral templates. No player names unless the roster is mapped and attribution confidence passes the threshold.
- Light color commentary from the live box score (runs, streaks, team shooting) with no outside facts.
- TTS output as a separate audio track; operator "mute/override" hotkey; profanity and safety filter.
- **Targets:** p95 event-to-speech ≤ 3 s; **0 wrong scores spoken**; ≤ 1 wrong-player attribution per game; ≤ 2 "hallucinated" events (spoken but didn't happen) per game.

### 6.2 Blind / low-vision fans (U9): P0
- AD mode: possession, ball location by court zone, shot type and result, score/clock every possession; terse, consistent vocabulary (co-design with BLV users, as Action Audio did over two years [54]; the duration is unverified).
- Separate low-latency audio stream (web player / LiveVoice-style link [52]); live captions (WebVTT).
- **Targets:** p95 ≤ 2 s; possession-change accuracy ≥ 97%; score accuracy 100% (operator-confirmed).

### 6.3 Head coach (U1): P0 post-game, P1 live
- Post-game (P0): team and player box score (player stats flagged provisional until confirmed), shot chart, scoring runs, lineup +/- and minutes from **operator lineup entry** (tap-to-sub at dead balls; §9.3), possession clip list, Claude 1-page summary citing event IDs, and a **correction UI** (edit an event and everything recomputes).
- Live (P1): tablet dashboard with fouls by player, team fouls/bonus, timeouts (counts from the per-league `RuleSpec`), run alerts, opponent hot zones; alerts ranked by Jev's B3 card bundle (`relevant__card_x` Nouls plus `top_card`). Jev does not decide interruptions (RESEARCH.md D3).
- **Targets:** team points 100% after operator check; made-FG detection precision ≥ 0.98 / recall ≥ 0.95; foul attribution ≥ 0.95 before live foul alerts ship.

### 6.4 Video coordinator / assistant (U2): P1
- Possession segmentation, event tags, clip export (MP4 + XML/CSV for Hudl/Synergy import), natural-language clip search over the event log.
- Fast "last N seconds" replay clip for challenge decisions (where permitted [58]).

### 6.5 Analytics staff (U3): P1
- Per-frame court x,y (team, jersey, confidence), ball x,y(,z), possession table, shot table with features (distance, contest Score from Jev, shot clock if known). Parquet/CSV + JSON schema docs.

### 6.6 Parents / players / recruiting (U11, U12): P1, opt-in only
- Per-player stat line and auto reel (Jev Score ranks highlight quality); player-controlled sharing; no school logos in monetized reels [92]; parent consent for under-18 and verifiable parental consent for under-13 [89].

### 6.7 Fans / second screen (U10): P1
- Live box score page; "what just happened?" and Q&A grounded strictly in the event log (refuse when unknown); end-of-game recap.

### 6.8 Scouts (U5): P1
- Opponent tendency report from our event logs (shot zones, pace, play-type mix when available); Claude prose with event citations; NCAA scouting-service compliance if sold to DI programs [114].

### 6.9 Athletic trainers (U6) and referees (U7): P2
- Trainers: minutes played, approximate distance with stated error band. No acceleration-based load claims until validated against a wearable reference.
- Referees: post-game clip packs by call type for assessors; no automated call judgments.

### 6.10 League / tournament operators (U14): P2
- Multi-court batch processing queue, standings/leaders, per-league privacy defaults, API with anti-wagering terms.

---

## 7. Implementation implications (how this shapes the codebase)

1. **Event schema must carry provenance and confidence**: `event_id, t_start, t_end, type, team, player_jersey|null, confidence, sources[detector, jev_question_id, operator_override], frame_refs`. Narration, stats and Q&A all read only from this log.
2. **Confidence-gated narration policy**: a per-event-type threshold table, keyed by the versioned Jev model id (e.g., `jev-1.13.0`, never the moving `jev-latest` alias), since probabilities differ between models [1][121]. Log `SystemOneResponse.model` on every call and refuse to apply a threshold table whose version does not match. Below threshold, use a neutral template or say nothing.
3. **Two-tier stats**: `official` (operator-confirmed) vs `provisional` (model). The UI and narration distinguish them.
4. **Operator-in-the-loop controls** from day one: score override, roster mapping, mute narration, correct-event, which recomputes the box score.
5. **Privacy module**: consent registry (per player: identify yes/no, reels yes/no, training yes/no); region flags (EU, IL, TX, under-13); retention jobs; no-face-recognition guarantee enforced in code review (no face-embedding libraries in deps).
6. **Edge-first deployment**: footage processed on the M3 Pro in the gym, with only events/stats optionally synced to the cloud. Keep Jev payloads compact text, well inside the default 32k-token per-question / 64k-token per-request limits (target <32 KB). Do not send the `TypeSafe-Danger-Zone-Long-Context` or `TypeSafe-Danger-Zone-Images` headers on the live path [121][122]. Claude receives facts, not frames.
7. **Output adapters**: TTS audio track, AD audio track, WebVTT captions, JSON event stream (authenticated, rate-limited, optional delay), Hudl/Synergy-style XML/CSV clip export.
8. **Data pipeline guardrails**: training-set builder accepts only footage whose game record has `training_allowed=true`; blurs faces; strips audio; refuses sources tagged `broadcast/third_party`.
9. **Evaluation harness** built around the §6 targets, especially "spoken-fact error rate" and "wrong score spoken = 0".

---

## 8. Open questions

- Does Pixellot's announced AI commentary exist in production for basketball, and how good is it? This is the main competitive risk; we found no technical disclosures [24][25].
- What is the current (2026-27) NCAA men's basketball rule on bench-area video and devices, and does it allow a live stats tablet with clips? [58][60]
- What is the current NFHS Rule 1-19 wording, and are there state-specific limits on live analytics devices? [59]
- Does internal R&D use of `nba_api` data by a for-profit count as "commercial" under NBA.com Terms §9? (Counsel.) [65]
- What are the official licenses of the other DeepSportradar datasets? The instance-segmentation data is CC BY-NC-ND 4.0 per the official README [119]; a third party lists CC BY-NC-SA 4.0 [100]. Both are non-commercial.
- Is PlayVision still a basketball-analytics product, or did it pivot to the sports-betting app "Bron" shown on its YC page? [41][42]
- What are the 2025-26 status and terms of Second Spectrum's NBA League Pass and team-analytics roles? [8]
- Would courts treat short-lived appearance re-ID embeddings or pose sequences as "biometric identifiers" (BIPA) or "gait patterns" (COPPA)? We designed to avoid the question, but it is unresolved.
- What are actual prices for Pixellot (basketball), Veo and Synergy HS? Not published. (BallerCam: $299/$399 hardware plus storage from $99/yr [43].)
- Can a single M3 Pro handle two 1080p60 streams plus tracking plus event engine in real time? (Engineering, separate note.)

---

## Sources

- [1] OpenRouter, "Jev" community guide (model ids `typesafe/jev-1.14`, `~typesafe/jev-latest`; Decisions API; 262,144-token context; long-context mode for >~32 KB text or images; no free-form text; thresholds per model). https://openrouter.ai/docs/guides/community/jev.md (local SDK details: `docs/research/sdk_ground_truth.md`)
- [2] OpenTweet, "Jev: Choice, Score, Noul" (secondary). https://opentweet.io/jev/choice-score-noul
- [4] ESPN, "NBA to use Hawk-Eye tracking system" (2023-03-09). https://www.espn.com/nba/story/_/id/35818363/nba-use-hawk-eye-tracking-system-follow-players-ball
- [5] Covers, "Sportradar unveils new cutting-edge NBA viewing experience" (2024-10-22). https://www.covers.com/industry/nba-sportradar-new-player-tracking-technology
- [6] NBA, "NBA and AWS announce new multi-year partnership" (2025-10-01). https://pr.nba.com/nba-aws-partnership/
- [7] NBA.com, "Inside the Game stat glossary". https://nba.com/news/inside-the-game-stat-glossary
- [8] NBA, "NBA and Genius Sports Second Spectrum expanded partnership" (2023). https://pr.nba.com/nba-genius-sports-second-spectrum-expanded-partnership
- [9] NBA/WNBA, "WNBA and Genius Sports bring tracking data to every WNBA arena" (2024). https://pr.nba.com/wnba-genius-sports-tracking-data
- [10] iGaming Business, "NBA to receive equity in Sportradar with exclusive data deal". https://igamingbusiness.com/nba-to-receive-equity-in-sportradar-with-exclusive-data-deal/ (see also City A.M. https://www.cityam.com/?p=1804706)
- [11] iGaming Business, "NBA signs first data supply partnerships". https://igamingbusiness.com/marketing-affiliates/nba-signs-first-data-supply-partnerships/
- [12] Business Wire, "Sportradar Acquires Synergy Sports" (2021-03-22). https://www.businesswire.com/news/home/20210322005437/en/
- [13] Sportradar, "Synergy Basketball for High School". https://sportradar.com/synergy-basketball-for-high-school/
- [14] Hudl, "Hudl Assist basketball pricing". https://www.hudl.com/pricing/assist/basketball
- [15] Hudl, "Hudl Assist for basketball". https://www.hudl.com/products/assist/basketball
- [16] Hudl, "Pricing for high school & middle school". https://www.hudl.com/pricing/high-school/assist
- [17] Hudl, "Club basketball plans & pricing". https://www.hudl.com/pricing/club/hudl/basketball
- [18] Hudl, "Hudl Focus cameras". https://www.hudl.com/products/focus
- [19] Utah Public Notice (procurement document with Hudl Focus Flex quote). https://www.utah.gov/pmn/files/1476377.pdf
- [20] Hudl, "Hudl expands volleyball focus through acquisition of Balltime" (2025-02). https://www.hudl.com/blog/hudl-expands-volleyball-focus-through-game-changing-acquisition-of-balltime
- [21] Silicon Prairie News, "Hudl acquires Titan Sports and Balltime" (2025-06). https://siliconprairienews.com/2025/06/lincoln-based-hudl-acquires-titan-sports-and-balltime-expanding-access-to-performance-data-for-athletes/
- [22] Pixellot homepage. https://www.pixellot.tv/
- [23] Youth Sports Business Report, "Pixellot and PlayOn extend partnership through 2030". https://youthsportsbusinessreport.com/pixellot-and-playon-extend-high-school-streaming-partnership-through-2030/
- [24] Youth Sports Business Report, "Pixellot raises $35M". https://youthsportsbusinessreport.com/pixellot-raises-35m-to-scale-ai-powered-game-capture-across-youth-and-high-school-sports/
- [25] Techleap, "Pixellot raises $35M for growth plans" (2025-11). https://finder.techleap.nl/news/feed/pixellot-raises-35m-for-growth-plans
- [26] HoopBrief, "Pixellot cost 2026" (third-party). https://hoopbrief.com/blog/pixellot-cost-2026
- [27] NFHS Network / PlayOn, "Become an NFHS Network school". https://nfhsnetwork.playonsports.com/overview
- [28] Wikipedia, "Pixellot" (Basketball Australia section, 2026). https://en.wikipedia.org/wiki/Pixellot
- [29] Youth Sports Business Report, "Basketball Australia and Pixellot launch HoopsTV". https://youthsportsbusinessreport.com/basketball-australia-and-pixellot-launch-hoopstv-a-national-youth-basketball-video-platform/
- [30] DeepCut News, "Basketball Australia considers Israeli AI tech to record junior games" (2026). https://www.deepcutnews.com/p/basketball-australia-considers-israeli
- [31] Veo, "Pricing". https://www.veo.com/en-us/pricing
- [32] Veo, "Buyer's guide for parents & youth teams". https://www.veo.com/en-us/article/buyers-guide-parents-youth-teams
- [33] SportsVisio, "Pricing". https://sportsvisio.com/pricing
- [34] Pulse 2.0, "SportsVisio: $3.2 million raised" (2025-06). https://pulse2.com/sportsvisio-3-2-million-raised-for-scaling-ai-sports-solution
- [35] CNBC, "ShotTracker gets backing from Verizon in $11 million round" (2021-01-14). https://www.cnbc.com/2021/01/14/shottracker-gets-backing-from-verizon-in-11-million-financing-round-.html
- [36] Sportico, "Live basketball stats company ShotTracker / Hearst" (2021). https://www.sportico.com/business/tech/2021/live-basketball-stats-company-shottracker-hearst-1234626143/
- [37] Ubergizmo, "HomeCourt basketball app uses AI to train" (2018). https://ubergizmo.com/2018/08/homecourt-basketball-app-uses-ai-to-train
- [38] FitGap, "HomeCourt" (third-party). https://us.fitgap.com/products/039841/homecourt
- [39] Apple App Store, "Ball AI - Basketball Training". https://apps.apple.com/us/app/-/id6450125219
- [40] Signalbase, "Ball AI secures $475K". https://www.trysignalbase.com/news/funding/ball-ai-secures-475k-pre
- [41] Y Combinator, "Launch YC: PlayVision". https://www.ycombinator.com/launches/OhD-playvision-ai-moneyball-for-sports
- [42] Y Combinator, "PlayVision" company page (on 2026-10-20 this URL shows "Bron", Fall 2025, with the same founders). https://www.ycombinator.com/companies/playvision
- [43] Prep Network, "AI-powered BallerCam built for youth sports streaming" (2025-07-24). https://prepnetwork.com/2025/07/24/ai-powered-ballercam-built-for-youth-sports-streaming/
- [44] Hooper. https://www.hooper.gg/
- [45] WSC Sports, "NBA launches GenAI technology with WSC Sports to automate multilingual content" (2025). https://wsc-sports.com/blog/customer-spotlight/nba-launches-genai-technology-with-wsc-sports-to-automate-multilingual-content/
- [46] IBM UK Newsroom, "IBM at Wimbledon 2025". https://uk.newsroom.ibm.com/IBM-at-Wimbledon-2025
- [47] Fox News, "AI-generated voice of former narrator Jim Fagan to be featured next NBA season, NBC says". https://www.foxnews.com/sports/ai-generated-voice-former-narrator-jim-fagan-featured-next-nba-season-nbc-sports-says.amp
- [48] Tech.eu, "Sportway raises €20M" (2026-06-23). https://tech.eu/2026/06/23/from-the-sidelines-to-250000-matches-a-year-sportway-raises-eur20m-to-scale-its-ai-sports-media-platform/
- [49] FootyHub TV. https://footyhub.tv/
- [50] Microsoft Accessibility Blog, "OneCourt: revolutionizing accessibility in live sports" (2025-01-14). https://blogs.microsoft.com/accessibility/onecourt-revolutionizing-accessibility-in-live-sports/
- [51] Axios Denver, "Denver Nuggets debut OneCourt tech for blind fans" (2026-01-13). https://www.axios.com/local/denver/2026/01/13/denver-nuggets-onecourt-tech-blind-fans
- [52] WRTV, "Pacers launch audio description service for visually impaired fans" (2025-10). https://www.wrtv.com/sports/pacers/pacers-launch-audio-description-service-for-visually-impaired-fans
- [53] Sony Europe, "Hawk-Eye Innovations and Action Audio debut inclusive feed for Wimbledon's Centre Court". https://www.sony.eu/presscentre/hawk-eye-innovations-and-action-audio-debut-inclusive-feed-for-wimbledons-centre-court
- [54] Digital Trends, "Action Audio soundscapes for sports". https://www.digitaltrends.com/outdoors/action-audio-soundscapes-for-sports/
- [55] Cyprus Mail (Reuters), "League to use AI for objective calls, Silver says" (2026-05-28). https://cyprus-mail.com/2026/05/28/league-to-use-ai-for-objective-calls-silver-says
- [56] ACC, "ACC selects RefMasters' RefScore for men's basketball officials" (URL date 2026-01-22; page dated 2026-01-23). https://theacc.com/news/2026/1/22/acc-selects-refmasters-refscore-as-its-performance-development-platform-for-mens-basketball-officials.aspx
- [57] NBA, "NBA referee training application named Innovation of the Year" (2022). https://pr.nba.com/nba-referee-training-application-award
- [58] WTOP (AP), "March Madness meets the iPad era: coaches' challenges" (2026-03-25). https://wtop.com/ncaa-basketball/2026/03/march-madness-meets-the-ipad-era-how-coaches-challenges-are-changing-the-game/
- [59] OHSAA (NFHS release), "2013-14 Basketball Rules Changes". https://ohsaa.org/sports/bk/2013-14BasketballRulesPressRelease.pdf
- [60] NCAA, "Ten basketball conferences allowed to experiment with electronic devices" (2019-11-25). https://www.ncaa.org/news/2019/11/25/ten-basketball-conferences-allowed-to-experiment-with-electronic-devices-during-league-games.aspx
- [61] Wareable, "Kinexon's wearable sensor is changing the way NBA teams prep". https://www.wareable.com/sport/kinexon-wearable-sensor-nba-basketball-6679
- [62] Vuckovic et al., Montenegrin Journal of Sports Science and Medicine (March 2026), Kinexon LPS. https://www.mjssm.me/clanci/MJSSM_March_2026_Vuckovic.pdf
- [63] FiveThirtyEight, "The NBA is turning to wearable sensors to prevent player injuries". https://fivethirtyeight.com/features/the-nba-is-turning-to-wearable-sensors-to-prevent-player-injuries
- [64] MobiHealthNews, "NBA bans use of wearable device data in contract negotiations" (2017). https://www.mobihealthnews.com/news/nba-bans-use-wearable-device-data-contract-negotiations
- [65] NBA.com, "Terms of Use" (last updated 2026-07-13; §1, §7, §9). https://www.nba.com/terms
- [66] GitHub, swar/nba_api (MIT). https://github.com/swar/nba_api
- [67] *NBA v. Motorola, Inc.*, 105 F.3d 841 (2d Cir. 1997). https://law.justia.com/cases/federal/appellate-courts/F3/105/841/598844/
- [68] NCAA, "NCAA Championships Digital Highlights Policy for Member Institutions" (updated 2026-09-01). https://www.ncaa.org/championships/media-center/ncaa-championships-digital-highlights-policy-for-member-institutions/
- [69] Awful Announcing, "March Madness social media rules holding teams back". https://awfulannouncing.com/ncaa/march-madness-social-media-rules-holding-teams-back.html
- [70] Kansas Legislature testimony (KSHSAA / NFHS Network exclusive postseason rights, 2023). https://kslegislature.gov/li/b2023_24/committees/ctte_s_ed_1/documents/testimony/20230123_11.pdf
- [71] GHSA, "Reminder of GHSA live streaming rules". https://www.ghsa.net/node/14814
- [72] Delaware DIAA, "Championship streaming". https://education.delaware.gov/diaa/sport_championships/streaming
- [73] NFHS, "NFHS Network to stream New York State Association championships". https://nfhs.org/stories/nfhs-network-to-stream-new-york-state-association-state-championships
- [74] The Register, "YouTube video subtitles used for AI training" (2024-07-17). https://www.theregister.com/2024/07/17/youtube_video_subtitles_ai/
- [75] Debevoise & Plimpton, "Anthropic and Meta decisions on fair use" (2025-06). https://www.debevoise.com/insights/publications/2025/06/anthropic-and-meta-decisions-on-fair-use
- [76] Manatt, "Making sense of the first two Gen AI fair use rulings" (2025). https://www.manatt.com/insights/newsletters/client-alert/making-sense-of-the-first-two-often-contradictory-sometimes-confounding-gen-ai-fair-use-rulings
- [77] Wiley, "Copyright Office issues key guidance on fair use in generative AI training" (2025-05). https://www.wiley.law/alert-Copyright-Office-Issues-Key-Guidance-on-Fair-Use-in-Generative-AI-Training
- [78] Illinois General Assembly, 740 ILCS 14/10 (BIPA definitions). https://lig.ilga.gov/legislation/ilcs/documents/074000140K10.htm
- [79] Illinois Public Act 103-0769. https://ilga.gov/legislation/publicacts/103/103-0769.htm
- [80] Thompson Hine, "Illinois legislature amends BIPA to limit damages and expand consent options" (2024). https://www.thompsonhine.com/insights/illinois-legislature-amends-bipa-to-limit-damages-and-expand-consent-options/
- [81] Data Protection Report (Norton Rose Fulbright), "BIPA and the record retention requirement" (2022-12); Thompson Coburn on *Mora v. J&M Plating*. https://www.dataprotectionreport.com/2022/12/bipa-and-the-record-retention-requirement/ ; https://www.thompsoncoburn.com/insights/publications/item/2022-12-15/illinois-appeals-court-holds-that-bipa-section-15(a)-requires-retention-and-destruction-policies-upon-possession-of-any-biometric-data
- [82] Stinson, "The privacy playbook: how sports, media, entertainment and apparel companies became litigation targets" (2026). https://www.stinson.com/newsroom-publications-the-privacy-playbook-how-sports-media-entertainment-and-apparel-companies-became-litigation-targets
- [83] Reed Smith, "Chicago Blackhawks hit with Illinois BIPA class action". https://www.reedsmith.com/our-insights/blogs/adlaw-by-request/102jzuc/chicago-blackhawks-hit-with-illinois-bipa-class-action-over-use-of-facial-recogni
- [84] Blank Rome, "Meet CUBI: Texas' biometric privacy law". https://blankrome.com/publications/meet-cubi-what-companies-need-know-about-texas-biometric-privacy-law
- [85] Texas Attorney General, "$1.4 billion settlement with Meta" (2024-07-30). https://www.texasattorneygeneral.gov/news/releases/attorney-general-ken-paxton-secures-14-billion-settlement-meta-over-its-unauthorized-capture
- [86] ZwillGen, "Don't mess with Texas: CUBI settlement". https://www.zwillgen.com/ftc-state-ag/dont-mess-with-texas-cubi-settlement-marks-state-tough-biometric-privacy-enforcer/
- [87] Hunton, "Washington becomes third state to enact biometric privacy law". https://www.hunton.com/privacy-and-cybersecurity-law-blog/washington-becomes-third-state-enact-biometric-privacy-law
- [88] Colorado Legislative Council, HB24-1130 fiscal note. https://leg.colorado.gov/sites/default/files/documents/2024A/bills/fn/2024a_hb1130_f1.pdf
- [89] Hunton, "FTC publishes final COPPA rule amendments" (2025). https://www.hunton.com/privacy-and-cybersecurity-law-blog/ftc-publishes-final-coppa-rule-amendments
- [90] BBB National Programs, "Reminder: amended COPPA rule compliance date is approaching". https://bbbprograms.org/media/insights/blog/coppa-amended
- [91] US Dept. of Education, "FAQs on photos and videos under FERPA". https://studentprivacy.ed.gov/faq/faqs-photos-and-videos-under-ferpa
- [92] Capital News Service (Maryland), "Navigating NIL: a patchwork of policies governing high school athletes" (2025-05-06). https://cnsmaryland.org/2025/05/06/navigating-nil-a-patchwork-of-policies-governing-high-school-athletes/
- [93] Kilpatrick Townsend, "Wiretap laws in the United States" (2024-07). https://ktslaw.com/blog/globalprivacy-and-cybersecuritylaw/2024/7/wiretap-laws-in-the-united-states
- [94] EDPB, Guidelines 3/2019 on processing of personal data through video devices. https://www.edpb.europa.eu/sites/default/files/files/file1/edpb_guidelines_201903_video_devices_en.pdf
- [95] Regulation (EU) 2016/679 (GDPR), Arts. 6, 8, 9, 35. https://eur-lex.europa.eu/eli/reg/2016/679/oj
- [96] Orrick, "EU Commission publishes guidelines on prohibited AI practices under the AI Act" (2025-04). https://www.orrick.com/en/insights/2025/04/eu-commission-publishes-guidelines-on-the-prohibited-ai-practices-under-the-ai-act
- [97] Legal Sports Report, "How Project Red Card data fight could affect US sports betting"; Lewis Silkin, "Project Red Card". https://www.legalsportsreport.com/?p=67676 ; https://sportsinsights.lewissilkin.com/post/102gcsz/project-red-card-footballers-to-sue-over-use-of-their-player-data-and-match-stat
- [98] Society for Computers & Law, "Database right: ECJ judgment in Football Dataco v Sportradar". https://www.scl.org/2595-database-right-ecj-judgment-in-football-data-co-v-sportradar/
- [99] Van Zandycke et al., "DeepSportradar-v1" (arXiv 2208.08190). https://arxiv.org/abs/2208.08190v1
- [100] Dataset Ninja, "DeepSportradar" (lists CC BY-NC-SA 4.0; third-party). https://datasetninja.com/deep-sport-radar
- [101] Legal Sports Report, "Illinois 'commercially reasonable' pricing rules for official league data". https://www.legalsportsreport.com/37240/illinois-sports-betting-rules-commercially-reasonable/
- [102] Vixio, "Tennessee to drop data mandate". https://www.vixio.com/insights/gc-tennessee-drop-data-mandate-part-sports-wagering-reshuffle
- [103] iGaming Business, "Genius Sports NCAA expanded partnership to 2032". https://igamingbusiness.com/sports-betting/genius-sports-ncaa-expanded-partnership-2032/
- [104] Legal Sports Report, "NCAA grants sportsbooks first-time access to official data, logos". https://www.legalsportsreport.com/231676/ncaa-grants-sportsbooks-first-time-access-to-official-data-logos/
- [105] WSN, "States consider banning college player prop bets after NCAA request" (2026). https://www.wsn.com/betting/states-consider-banning-college-player-prop-bets-after-ncaa-request/
- [106] Boston Globe, "Chauncey Billups, Terry Rozier federal charges" (2025-10-23). https://www.bostonglobe.com/2025/10/23/sports/chauncey-billups-terry-rozier-federal-charges
- [107] Legal Sports Report, "NBA, players union back more limits on prop bets" (2025). https://www.legalsportsreport.com/239853/nba-players-union-back-more-limits-on-prop-bets/
- [108] Code of Virginia § 58.1-4128. https://law.lis.virginia.gov/vacode/58.1-4128
- [109] NFHS, "Supreme Court's decision on sports gambling: effects on high school games". https://nfhs.org/stories/supreme-court-s-decision-on-sports-gambling-effects-on-high-school-games
- [110] Basketball Australia, "Basketball Australia introduces courtsiding policy". https://australia.basketball/news/3337783/basketball-australia-introduces-courtsiding-policy
- [111] Manatt, "Tennessee's ELVIS Act expands publicity rights for individuals' voices" (2024). https://www.manatt.com/insights/newsletters/client-alert/tennessees-elvis-act-expands-publicity-rights-for
- [112] JD Supra, "DOJ delays ADA web accessibility compliance deadlines" (2026-04). https://www.jdsupra.com/legalnews/doj-delays-ada-web-accessibility-7295347/
- [113] W3C, Web Content Accessibility Guidelines (WCAG) 2.1 (SC 1.2.4, 1.2.5). https://www.w3.org/TR/WCAG21/
- [114] NCAA, "Institutional responsibilities regarding recruiting/scouting service approvals" (now redirects to "Recruiting and Scouting Service", http://www.ncaa.org/eligibility-center/recruiting/recruiting-and-scouting-service). https://www.ncaa.org/sports/2020/11/24/institutional-responsibilities-regarding-recruiting-scouting-service-approvals.aspx
- [115] Northwestern University, "Recruiting service approval form" (school practice example). https://www.northwestern.edu/athleticbusinessoffice/forms/recruiting-service-approval-form.pdf
- [116] Hoopsalytics blog, "Hudl Assist alternative for basketball" (competitor marketing). https://blog.hoopsalytics.com/hudl-assist-alternative-for-basketball/
- [117] Basketball Australia, "Photography and filming of young people" (2026-05). https://resources.basketball-australia.pulselive.com/basketball-australia/document/2026/05/28/030f58b4-3b13-482f-9c6d-2db0e87b2ce6/Photograhy_Filming_Young_People_Doc.pdf
- [118] Startup Daily, "AI amateur sport analysis startup raises $3.5 million pre-seed" (2026-07). https://www.startupdaily.net/topic/funding/ai-amateur-sport-analysis-startup-raises-3-5-million-pre-seed/
- [119] GitHub, DeepSportradar/instance-segmentation-challenge README (data CC BY-NC-ND 4.0; code Apache-2.0). https://github.com/DeepSportradar/instance-segmentation-challenge
- [120] Level Up Coding, "Stop using generative LLMs for every decision: meet Jev" (secondary; vendor latency/pricing figures, unverified). https://levelup.gitconnected.com/stop-using-generative-llms-for-every-decision-meet-jev-96f2ec183fc4
- [121] TypeSafe docs, "Models" (Jev 1.14 `jev-1.14.0` / Jev 1.13 `jev-1.13.0`; 64k per-request / 32k per-question context; `TypeSafe-Danger-Zone-Long-Context` header; aliases `jev-latest` and `jev-preview`). https://docs.typesafe.ai/models (index: https://docs.typesafe.ai/llms.txt)
- [122] TypeSafe docs, "Images" (`TypeSafe-Danger-Zone-Images: true` header; images and Box questions labeled "Dreamer-ready", not production). https://docs.typesafe.ai/images

---

## Verification log

Adversarial fact-check on 2026-10-20. Each cited primary source was fetched again, plus a second source where one was reachable. WebSearch was unavailable (search budget used up), so checks used WebFetch on the cited URLs and on known official URLs only. ilga.gov (BIPA statute and P.A. 103-0769) refused automated fetches (certificate error, then a "Bot Blocked" 403), so BIPA text was checked through secondary legal sources.

### Confirmed as written (no change, or wording tightened only)

- **NBA.com Terms of Use**: last updated 2026-07-13. §9 restricts NBA Statistics to legitimate news reporting or private non-commercial use. It bars sponsorship or commercial identification (added to the note), any gambling (including legal), fantasy or other commercial products, live, near-live, real-time or archived play-by-play, and regularly updated statistics databases without consent, and it requires prominent NBA.com attribution. §1: single-computer personal non-commercial downloads only; video footage and game-action photos are NBA IP. No AI, training or scraping language. https://www.nba.com/terms
- **nba_api**: MIT license; the README links NBA.com terms (https://www.nba.com/termsofuse). https://github.com/swar/nba_api
- **Hawk-Eye from 2023-24**, replacing Second Spectrum's single-point "center of mass" tracking (ESPN, 2023-03-09). https://www.espn.com/nba/story/_/id/35818363/nba-use-hawk-eye-tracking-system-follow-players-ball
- **29 skeletal points, 14 cameras per arena, under 1 s latency** (Covers, 2024-10-22). The article does not mention Hawk-Eye or Second Spectrum, and it describes the recipients as "broadcast and sports betting partnerships". The note's wording was adjusted. https://www.covers.com/industry/nba-sportradar-new-player-tracking-technology
- **NBA x AWS** (2025-10-01): "Official Cloud and Cloud AI Partner"; Defensive Box Score, Shot Difficulty (Expected FG%) and Gravity; tracking processed 60 times per second; 29 data points per player; Play Finder on Amazon Bedrock + SageMaker with a real-time alert system for commentators. https://pr.nba.com/nba-aws-partnership/
  - The Gravity condition (all 5 players and the ball in the frontcourt) is confirmed, and "not captured during rebounds" was added. https://nba.com/news/inside-the-game-stat-glossary
- **Hudl Assist basketball**: $900 / $1,300 / $1,500 / $2,100 per team per season; up to 3 games/day; one team; current season only. https://www.hudl.com/pricing/assist/basketball
  - The "human analysts" claim is confirmed on the product page ("Our team of professional analysts", 150,000+ HS games last season), not on the pricing page. https://www.hudl.com/products/assist/basketball
- **Hudl HS packages** (Silver+ $1,500, Gold+ $2,500, Platinum+ $4,000 per year) and **club basketball** ($400 / $1,000 / $1,600 per year) are confirmed. https://www.hudl.com/pricing/high-school/assist ; https://www.hudl.com/pricing/club/hudl/basketball
- **Balltime** is volleyball (indoor and beach), acquisition completed 2025-02-06, 12k+ teams / 125k+ athletes. https://www.hudl.com/blog/hudl-expands-volleyball-focus-through-game-changing-acquisition-of-balltime
- **Pixellot**: $35M ($15M PSG Equity + $20M venture debt), reported 2025-11-23. AI commentary is listed among recent and upcoming releases with no technical details. The phrase "in the voice of a friend or family member" was added. https://youthsportsbusinessreport.com/pixellot-raises-35m-to-scale-ai-powered-game-capture-across-youth-and-high-school-sports/
  - PlayOn extension: five years, through 2030, "AI-as-a-Service" revenue share, 2025-11-06. https://youthsportsbusinessreport.com/pixellot-and-playon-extend-high-school-streaming-partnership-through-2030/
  - Show S3: 19 sports with graphics, ads, commentary and highlights. https://www.pixellot.tv/
- **WSC Sports x NBA**: post-game 2-minute highlight voice-overs in French, Portuguese and Spanish, 2025-04-06. Not live. https://wsc-sports.com/blog/customer-spotlight/nba-launches-genai-technology-with-wsc-sports-to-automate-multilingual-content/
- **OneCourt**: haptics plus generative audio description from real-time league data; Trail Blazers pilot 2023-24 and then every home game; fans rated comprehension 74.3% higher than radio alone (who measured it is not stated). https://blogs.microsoft.com/accessibility/onecourt-revolutionizing-accessibility-in-live-sports/
- **Pacers x MindsEye**: human describers at 20 home games in 2025-26 via the LiveVoice app. https://www.wrtv.com/sports/pacers/pacers-launch-audio-description-service-for-visually-impaired-fans
- **BIPA**
  - §15(a) text, including "within 3 years of the individual's last interaction", and the *Mora* possession trigger. https://www.dataprotectionreport.com/2022/12/bipa-and-the-record-retention-requirement/
  - The list of biometric identifiers, $1,000 negligent / $5,000 reckless or intentional damages, SB 2979's one-recovery-per-person rule for §15(b)/(d), and electronic signature. https://www.thompsonhine.com/insights/illinois-legislature-amends-bipa-to-limit-damages-and-expand-consent-options/
  - The "photographs excluded" text and the 2024-08-02 signing date could not be re-fetched from ilga.gov.
- **COPPA**: published 2025-04-22, effective about 2025-06-21, compliance 2026-04-22. The biometric list includes gait patterns, facial templates and faceprints. The "automated or semi-automated recognition" qualifier and the dropped "derived data" language were added. https://www.hunton.com/privacy-and-cybersecurity-law-blog/ftc-publishes-final-coppa-rule-amendments ; https://bbbprograms.org/media/insights/blog/coppa-amended
- **NCAA digital highlights policy** (updated 2026-09-01): venue footage is NCAA copyright; licensing goes through Veritone, with TNT Sports holding digital highlight rights, "regardless of how such footage is acquired". https://www.ncaa.org/championships/media-center/ncaa-championships-digital-highlights-policy-for-member-institutions/
- **Genius Sports x NCAA**: exclusive official data to licensed sportsbooks for March Madness and all other postseason tournaments, through 2032 (2025-04-25). https://igamingbusiness.com/sports-betting/genius-sports-ncaa-expanded-partnership-2032/
- **Virginia § 58.1-4128**: no wagering on youth sports games; violation is a Class 1 misdemeanor. https://law.lis.virginia.gov/vacode/58.1-4128
- **DOJ ADA Title II**: interim final rule 2026-04-20; WCAG 2.1 AA; new deadlines 2027-04-26 (50k+) and 2028-04-26 (under 50k and special districts); covers K-12 districts and public colleges. https://www.jdsupra.com/legalnews/doj-delays-ada-web-accessibility-7295347/
- **WCAG 2.1**: SC 1.2.4 Captions (Live) is AA, 1.2.5 Audio Description (Prerecorded) is AA, and 1.2.9 Audio-only (Live) is AAA. There is no live-AD criterion. https://www.w3.org/TR/WCAG21/
- **Fair use**
  - *Bartz* (2025-06-23): "exceedingly transformative" confirmed; "no entitlement to use pirated copies for its central library". https://www.manatt.com/insights/newsletters/client-alert/making-sense-of-the-first-two-often-contradictory-sometimes-confounding-gen-ai-fair-use-rulings ; https://www.debevoise.com/insights/publications/2025/06/anthropic-and-meta-decisions-on-fair-use
  - *Kadrey* (2025-06-25): the market-dilution warning is confirmed (same two sources).
  - US Copyright Office Part 3 pre-publication report, 2025-05-09. https://www.wiley.law/alert-Copyright-Office-Issues-Key-Guidance-on-Fair-Use-in-Generative-AI-Training
- **NFHS Rule 1-19** (release dated 2013-05-09): devices allowed for coaching and gathering statistics; no voice communication with players on the court; no audio/video review of officials' decisions. https://ohsaa.org/sports/bk/2013-14BasketballRulesPressRelease.pdf
- **Kinexon**: sub-10 cm accuracy, 20 ms latency, 300+ metrics, 20 Hz (MJSSM, March 2026). https://www.mjssm.me/clanci/MJSSM_March_2026_Vuckovic.pdf
- **Second Spectrum**: 2023-03-09 roles (League Pass augmentation; team analytics for all 30 teams). https://pr.nba.com/nba-genius-sports-second-spectrum-expanded-partnership
- **WNBA tracking**: 2024-03-05, 3D posture and ball tracking in every arena. https://pr.nba.com/wnba-genius-sports-tracking-data
- **Synergy HS**: package names, 12-month non-prorated contracts, under-24 h turnaround, 500+ categories, NBA/WNBA video for an extra fee. https://sportradar.com/synergy-basketball-for-high-school/
- **Veo**: auto event detection for soccer, rugby and lacrosse; Analytics 2 is soccer-only; no prices shown. https://www.veo.com/en-us/pricing
- **Other legal sources**
  - CUBI: AG-only enforcement, up to $25,000 per violation, Meta $1.4B on 2024-07-30. https://www.zwillgen.com/ftc-state-ag/dont-mess-with-texas-cubi-settlement-marks-state-tough-biometric-privacy-enforcer/
  - Washington photo/video exclusion. https://www.hunton.com/privacy-and-cybersecurity-law-blog/washington-becomes-third-state-enact-biometric-privacy-law
  - EU AI Act Art. 5 applies from 2025-02-02 (emotion recognition at work and in education; biometric categorization). https://www.orrick.com/en/insights/2025/04/eu-commission-publishes-guidelines-on-the-prohibited-ai-practices-under-the-ai-act
  - FERPA video guidance. https://studentprivacy.ed.gov/faq/faqs-photos-and-videos-under-ferpa
  - HS NIL: 41 states + DC; all bar naming the school. https://cnsmaryland.org/2025/05/06/navigating-nil-a-patchwork-of-policies-governing-high-school-athletes/
  - Wiretap: 13 all-party states. https://ktslaw.com/blog/globalprivacy-and-cybersecuritylaw/2024/7/wiretap-laws-in-the-united-states
  - ELVIS Act effective 2024-07-01, covering simulated voice. https://www.manatt.com/insights/newsletters/client-alert/tennessees-elvis-act-expands-publicity-rights-for
  - GHSA (https://www.ghsa.net/node/14814), NYSPHSAA/NFHS (https://nfhs.org/stories/nfhs-network-to-stream-new-york-state-association-state-championships), the KSHSAA exclusive postseason (https://kslegislature.gov/li/b2023_24/committees/ctte_s_ed_1/documents/testimony/20230123_11.pdf), and the NFHS gambling statement (https://nfhs.org/stories/supreme-court-s-decision-on-sports-gambling-effects-on-high-school-games).
- **Other market facts**
  - NBA automated calls (Silver, 2026-05-27). https://cyprus-mail.com/2026/05/28/league-to-use-ai-for-objective-calls-silver-says
  - ACC RefScore for 2025-26. https://theacc.com/news/2026/1/22/acc-selects-refmasters-refscore-as-its-performance-development-platform-for-mens-basketball-officials.aspx
  - NBA REPS, Innovation of the Year 2022. https://pr.nba.com/nba-referee-training-application-award
  - IBM Match Chat in live singles. https://uk.newsroom.ibm.com/IBM-at-Wimbledon-2025
  - Sportway €20M on 2026-06-23. https://tech.eu/2026/06/23/from-the-sidelines-to-250000-matches-a-year-sportway-raises-eur20m-to-scale-its-ai-sports-media-platform/
  - FootyHub (all 104 World Cup matches). https://footyhub.tv/
  - Superstat $3.5M on 2026-07-28, basketball focus. https://www.startupdaily.net/topic/funding/ai-amateur-sport-analysis-startup-raises-3-5-million-pre-seed/
  - HomeCourt: 300 free shots/mo, $7.99/mo (2018). https://ubergizmo.com/2018/08/homecourt-basketball-app-uses-ai-to-train
  - SportsVisio: $9M total raised, Sony Innovation Fund. https://pulse2.com/sportsvisio-3-2-million-raised-for-scaling-ai-sports-solution

### Corrected

1. **Jev context and long-context mode (implementation-critical).** The note attributed to OpenRouter a ">~32,000 bytes / all images go to long-context mode" rule. TypeSafe's own docs, which govern the direct `typesafe-sdk` path, say something different:
   - default limits are 64k tokens per request and 32k tokens per question;
   - long context is opt-in with the `TypeSafe-Danger-Zone-Long-Context: true` header (262k tokens; slower and less accurate);
   - images need `TypeSafe-Danger-Zone-Images: true` and are not production-ready;
   - versioned ids are `jev-1.14.0` and `jev-1.13.0`, with `jev-latest` pointing to `jev-1.13.0` and `jev-preview` to `jev-1.14.0`;
   - TypeSafe recommends pinning the versioned id when thresholds are tuned.
   §3 and §7 items 2 and 6 were updated. OpenRouter's statement is kept and labeled as OpenRouter's. Evidence: https://docs.typesafe.ai/models and https://docs.typesafe.ai/images (read from copies fetched 2026-10-20); https://openrouter.ai/docs/guides/community/jev.md
2. **Sportradar/NBA.** "Exclusive international betting-data provider" was too narrow, and "US non-exclusive" (a 2018 deal) was stale. Corrected to: from 2023-24 to 2031, Sportradar is the exclusive data provider and authorized global distributor of official betting data; 3% warrants at $0.01. https://igamingbusiness.com/nba-to-receive-equity-in-sportradar-with-exclusive-data-deal/ ; https://www.cityam.com/?p=1804706 ; https://igamingbusiness.com/marketing-affiliates/nba-signs-first-data-supply-partnerships/
3. **SportsVisio.** Shot charts and the AI game summary are on Elite and Coach only, not Pro. Coach overage is $37.50. API access appears to be on the league tier. The pricing page does not label the box score "AI". https://sportsvisio.com/pricing
4. **PlayVision.** The YC company URL now shows "Bron", a sports-betting app with the same founders, so a pivot is likely. "Transfer-portal database" and "1-2 h per game" are not on the YC launch page and are marked unverified. https://www.ycombinator.com/companies/playvision ; https://www.ycombinator.com/launches/OhD-playvision-ai-moneyball-for-sports
5. **DeepSportradar license.** The official README says the instance-segmentation data is CC BY-NC-ND 4.0 (code Apache-2.0), not only the third-party CC BY-NC-SA. https://github.com/DeepSportradar/instance-segmentation-challenge ; https://datasetninja.com/deep-sport-radar
6. **BallerCam.** Price found: $299, or $399 with tripod and bag, plus storage from $99/yr. "2M+ livestreamed games" corrected to "trained on data from over 2 million youth-sports games". https://prepnetwork.com/2025/07/24/ai-powered-ballercam-built-for-youth-sports-streaming/
7. **Ball AI.** The App Store says "300K+ hoopers", not 50k+. "Automatic" is the developer name (Automatic AI, LLC), not an Apple Watch feature. Prices added. https://apps.apple.com/us/app/-/id6450125219
8. **Hooper.** The $11.99/mo Pro price is confirmed on the official site (Free / Pro / Team $29.99), so "(unverified)" was removed. SoftBank Japan partnership. https://www.hooper.gg/
9. **NBC AI Jim Fagan.** "That it was licensed is our inference" replaced with: Fagan's family gave NBC Sports permission. https://www.foxnews.com/sports/ai-generated-voice-former-narrator-jim-fagan-featured-next-nba-season-nbc-sports-says.amp
10. **Basketball Australia/HoopsTV dates.** "Oct-Nov 2026" clarified: announced 2026-10-12, go-live expected November 2026. "Pixellot stores abroad [30]" was re-cited to [28] as "reportedly transferred or stored overseas", because [30] does not say where footage is stored. https://youthsportsbusinessreport.com/basketball-australia-and-pixellot-launch-hoopstv-a-national-youth-basketball-video-platform/ ; https://en.wikipedia.org/wiki/Pixellot ; https://www.deepcutnews.com/p/basketball-australia-considers-israeli
11. **NCAA DI basketball.** "Carved out under even tighter rules" corrected: the highlights policy does not apply to DI basketball championship games, and extra rights come only through optional TNT/NCAA/Veritone licensing. Clip and GIF limits added. (NCAA policy URL above.)
12. **CUBI.** "$25,000 per violation" changed to "up to $25,000". (ZwillGen above.)
13. **Colorado HB24-1130.** The fiscal note says signed 2024-05-31 and effective 2024-08-07. The 2025-07-01 operative date is not in the cited source and is marked unverified. https://content.leg.colorado.gov/sites/default/files/documents/2024A/bills/fn/2024a_hb1130_f1.pdf
14. **Courtsiding.** "Basketball Australia/NBL" corrected to Basketball Australia's 2018 policy for all basketball in Australia; the NBL is not named. https://australia.basketball/news/3337783/basketball-australia-introduces-courtsiding-policy
15. **WSC completion figure.** Clarified: the headline says ">75%" and the body says an average of 75%. (WSC URL above.)
16. **Veo Coach Assist** is effectively soccer-only because it requires Analytics 2. Veo Go uses two iPhones. (Veo URL above.)
17. **Wiretap list** expanded to the 13 states the source names. **HS NIL** notification detail added. (Sources above.)
18. **Hudl Focus Indoor** specs (180°, ball tracking, single ethernet cable) are not on the cited page and are marked unverified. https://www.hudl.com/products/focus
19. **Synergy** "human logging" is not stated on the cited page and is marked unverified. (Synergy URL above.)
20. **Cubs BIPA case** detail added (*Berta v. Chicago Cubs*, withdrawn). **Zellmer** is not in its cited source [82] and is marked unverified. https://www.stinson.com/newsroom-publications-the-privacy-playbook-how-sports-media-entertainment-and-apparel-companies-became-litigation-targets

### Marked "(unverified)": source dead, blocked, or silent on the claim

- YouTube CEO "clear violation" quote: not in [74].
- IBM "AI commentary since 2023": not in [46].
- "30+ defendants": not in [106].
- ELVIS criminal enforcement: not in [111].
- NCAA scouting-service specifics: [114] redirects; [115] returns 404.
- NCAA iPad challenges [58] (404) and the 2019 device experiment [60] (redirects to the news index).
- Kinexon sensor weight and receiver count [61] (403); "over half of NBA teams" [63] (redirects away); CBA $250k fine [64] (403).
- ShotTracker $45,000 [35][36]: blocked.
- Action Audio [53][54]: 403 / empty.
- Nuggets OneCourt [51]: 403.
- Delaware DIAA [72]: 403.
- College prop bans [105] and NBA prop limits [107]: 403.
- NCAA "risky bet types" [104]: 403.
- Project Red Card 2,000+ [97]: 404.
- Utah Focus Flex quote [19]: PDF text not extractable.
- KSHSAA "since 2018" [70]: not in the text.
- Jev latency of 70-500 ms [120]: already marked unverified; no figure appears in the TypeSafe docs.

### Not changed but worth knowing

- The SDK ground truth (`docs/research/sdk_ground_truth.md`) says the SDK default model is `jev-latest`. TypeSafe's Models page says `jev-latest` currently resolves to `jev-1.13.0`, while `jev-1.14.0` (from 2026-10-14) is reachable via `jev-preview` or its versioned id. OpenRouter calls 1.14 "current", which matches TypeSafe's "newest release", not "latest stable".

---

## 9. Addendum: gap pass (2026-10-20)

Cross-note decisions are in `docs/RESEARCH.md` §1A and §7A. Sources [G1]-[G12] are at the end.

### 9.1 Minors' data sent to cloud processors (Jev, Claude)
**What the vendors' terms say.**
- **Anthropic retention.** API inputs and outputs are deleted within 30 days of receipt or
  generation unless otherwise agreed (e.g. ZDR). Trust-and-safety-flagged items are kept up to 2
  years (scores up to 7 years); feedback is kept 5 years [G1].
- **Anthropic Commercial Terms** (effective 17 June 2025): "Anthropic may not train models on
  Customer Content from Services". A DPA is incorporated by reference. The Services are "not for
  consumer use" [G2].
- **Anthropic ZDR.** The bundled Claude docs say to treat Opus 5.5 like Opus 5 on retention, and
  Opus 5 is available under ZDR. This is not confirmed on a live Anthropic page (unverified).
- **TypeSafe.** ZDR is available to enterprise customers only [G3]. Its DPA makes TypeSafe the
  processor/service provider and the customer the controller. Retention is "as long as necessary".
  Subprocessors are listed at trust.typesafe.ai (not readable by our fetcher). Customers get 15 days
  to object to a new subprocessor. Breach notice is due within 72 h. Transfers use EU SCC Modules 2
  and 3. Children's data is not addressed [G4].

**What the law requires.**
- **FERPA.** An outside party is a "school official" only if it performs an institutional service,
  "is under the direct control" of the school with respect to education records, and is subject to
  the §99.33(a) redisclosure limits [G5]. Our customer contract must name Anthropic and TypeSafe as
  subprocessors and flow those limits down to them.
- **COPPA.** Personal information includes a first and last name; "a photograph, video, or audio
  file where such file contains a child's image or voice"; and biometric identifiers, including gait
  patterns. An "operator" includes one on whose behalf an agent or service provider collects or
  maintains information [G6]. Whether we are an operator collecting from children online is for
  counsel.
- **GDPR.** Pseudonymised data remains personal data (Recital 26; Art. 4(5) defines pseudonymisation; not re-fetched in this pass) [95].

**Design (proposal).**
- Pseudonymise every person in cloud payloads with per-game tokens (`P07`). Keep the name map on the
  device. Re-insert names locally only after the deterministic validator passes.
- A consent registry gates cloud narration. Without a signed DPA, consent, or cleared region and age
  (any under-13 player without verifiable parental consent), the system runs **template-only**, with
  no cloud calls.
- Request ZDR from both vendors before school sales.

### 9.2 Ingest partners' terms (per-source allow/deny)
| Source | Terms (key clauses) | Verdict |
|---|---|---|
| Hudl (updated 11 Aug 2026) [G7] | §3.2.8: no "automated means, including bots, scrapers, crawlers, or artificial intelligence tools" to collect or extract data; no training ML on Hudl content without written permission. §3.2.3: no "unauthorized distribution of livestreams or broadcasts". §5.4: film is the customer's, but "you may not export it or any full game recording" | Deny automated ingest; partner agreement only |
| NFHS Network (Nov 2024) [G8] | "personal, non-commercial use only"; no "automated means, like scripts or web crawlers"; no data mining; no reproduction, distribution or derivative works | Deny; PlayOn agreement needed. Postseason exclusivity (KS, GA, NY) applies too |
| Veo (29 May 2026) [G9] | Customer owns content; no "access, reproduce, download, distribute, transmit, broadcast" of platform material except as permitted (User Terms §7.4); Veo may train AI on content (§6.3); API/partner apps contemplated (§1.1, §7.1) | Conditional: customer-exported files for post-game; live only via Veo's API/partner terms |
| Pixellot [G10] | No public terms of use found; the footer links only privacy documents | Unknown; partner agreement |
| BallerCam | Not checked | Unknown |
| OBS / iPhone / school camera pushing RTMP/SRT | Customer controls the stream | **Default MVP source** |

### 9.3 P0 features that lacked a build path
- **Lineups / minutes / +/-.** These come from operator entry: starting fives at setup, then
  tap-to-sub at dead balls. CV suggestions come later (Phase E). Metric: share of game seconds in
  which each team's recorded five is correct; target ≥ 98% after operator entry.
- **Captions.**
  - OBS: obs-websocket `SendStreamCaption` (CEA-608) [G11].
  - YouTube: HTTP POST captions to the event's signed ingestion URL, or embedded 608/708; one
    caption track; YouTube advises a 30 s or 60 s delay when caption software is used [G12].
  - Our own HLS player: WebVTT renditions with `X-TIMESTAMP-MAP` (RFC 8216).
  - Cues are keyed to source video PTS and shown at PTS plus the stream's applied delay.
- **AD vocabulary and Q&A design:** RESEARCH.md §7A.8 and `narration.md` §11.5-11.6.

Sources for this addendum:
[G1] Anthropic Privacy Center, "How long do you store my organization's data?" https://privacy.claude.com/en/articles/7996866-how-long-do-you-store-my-organization-s-data
[G2] Anthropic Commercial Terms of Service. https://www.anthropic.com/legal/commercial-terms
[G3] TypeSafe docs, Legal. https://docs.typesafe.ai/legal.md
[G4] TypeSafe Data Processing Agreement. https://typesafe.ai/legal/data-processing
[G5] 34 CFR § 99.31 (eCFR). https://www.ecfr.gov/current/title-34/subtitle-A/part-99/subpart-D/section-99.31
[G6] 16 CFR § 312.2 (eCFR). https://www.ecfr.gov/current/title-16/chapter-I/subchapter-C/part-312/section-312.2
[G7] Hudl Terms. https://www.hudl.com/terms
[G8] NFHS Network Terms of Use. https://www.nfhsnetwork.com/termsofuse
[G9] Veo General Terms and User Terms. https://www.veo.com/terms
[G10] Pixellot Privacy Policy page. https://www.pixellot.tv/privacy-policy
[G11] obs-websocket 5.x protocol. https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md
[G12] YouTube Help, live captions. https://support.google.com/youtube/answer/3068031
