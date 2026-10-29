# Possession and shots from the video: a decision model, and where Jev fits

Design with measurements, 2026-10-29. One game so far: Lakers @ Warriors, 2024-12-25 (ESPN), research use.
The probe that produced the Jev numbers is `tools/nba_jev_probe.py` (§5). Step 1, the possession timeline, is built
(§6), and so is step 3, made or missed from the video (§7), now also finding shots the engine misses and free
throws (§8).

## 1. What goes wrong today

The engine alone (without the scorebug fusion), whole game, `runs/full/report_events.md`. Its own made / missed
labels still lean on the scorebug (§7), so its makes here are better than vision alone would give:

| | Engine | Truth |
|---|---:|---:|
| Made baskets found (field goals and free throws) | 57 of 113 (recall 0.50) | 113 |
| Made free throws | 5 | 31 |
| Steals | 27 (F1 0.06) | 8 |
| Turnovers | 46 (F1 0.18) | 22 |
| Rebounds | 141 (F1 0.47) | 85 |
| Possession gains | 344 | about 200 (about 100 per team) |
| Player track ids | 944 | 10 players on court at a time |

With the scorebug fusion, points and makes become exact because they are read off the scoreboard, but steals,
turnovers and rebounds stay as they are. Four causes:

1. **Possession is decided greedily, frame by frame.** `events/possession.py` (`HolderTracker`) confirms and releases
   holders with votes and hysteresis, reading 67 hand-set `EventEngineConfig` settings. A single wrong flip of team
   becomes a steal and a turnover, or a rebound. At settled moments (3 s before a shot, no other play-by-play action
   in the 2 s before) the engine had the wrong team 9 times out of 40 (§2).
2. **A ball in a player's hands is never labelled.** Ball labels come from the teacher (RF-DETR) only where it saw
   the ball and one path linked it (`training/ball_labels.py`). Training skips unlabelled frames, which is right,
   but it means the student never sees a single held ball as a positive. `tools/nba_ball_bench.py` scores only
   labelled frames (`if not lab["xy"]: continue`), so its 95.7% measures balls in the open, not balls in hands.
3. **Player identity breaks every few seconds** (944 track ids), so "who did it" cannot be kept over a game.
4. **Makes.** The engine's rim cues find half the makes, and almost no free throws.

## 2. What Jev adds (measured)

Jev answers typed questions (yes/no, pick one, score) about a state, with calibrated probabilities. It is not a
tracker. Given only numbers the CV computed, its answer re-weighs the same evidence (jev_integration_design.md,
site 5). It adds new evidence only when it looks at pixels: image input, `jev-1.14.0`, labelled pre-production by
TypeSafe.

Truth for every row below is the ESPN play-by-play placed on video time (`nba_analyze.truth`).

| Probe (`nba_jev_probe.py`) | Jev | Engine, same moments | Per request |
|---|---|---|---|
| `makes`: 10 rim crops around the ball's arrival, 40 field goals (20 made, 20 missed) | 80% asked once (order-ensembled); 82.5% from "is the ball in the net?" per crop | 94% where it answers, no answer on 6 of 40 (80% of all 40) | 12k tokens, $0.0005, 0.5 s |
| `possession`: one full frame in settled play, 40 moments (20 per team), run twice | 80-85% (31 and 33 of 39 answered); confident (p ≥ 0.85) on 19 and 20, **all right in both runs** | 77.5% (31 of 40) | 4.3k tokens, $0.0002, 0.2 s |
| `turnovers`: all 42 engine turnovers with frames on the court, one frame 2 s before and one 2 s after | vetoes 21 of the 29 false ones; also vetoes 1 of 6 real turnovers and 4 of 7 other changes of possession; confident on 4 | all 42 claimed (6 real) | 7.9k tokens, $0.0003, 0.3 s |

- Settled play: Jev reads possession well, better than the engine, and its confidence means something.
- Scrambles: it is unsure, which is exactly where the false steals happen. Three frames a side (2.5, 4 and 6 s)
  caught more real changes (9 of 13 instead of 5) but vetoed fewer false ones (15 of 29 instead of 21).
- Make or miss: about the engine's level. When the two agree (27 of 34 shots) both were right every time.
- Identical requests do not always get identical answers: in the second `possession` run two unsure answers flipped
  to wrong. Only confident answers should count as evidence.
- These are small samples from one game. They show direction, not final rates.

## 3. Design: one decision over time instead of many decisions per frame

```
frames ──► detector, court, ball flights (exists)
              │
              ├─► CV evidence: the camera's aim, shot results, rebounds, flights at the rim
              ├─► Jev evidence where the CV is unsure: team in settled play, ball in the net
              └─► rules: the 8-second back court, what follows a make or a miss
                         │
                         ▼
              possession timeline (decoder over the whole game, or a few seconds behind live)
                         │
                         ▼
              steals, turnovers, rebounds, shots, makes = the boundaries of its segments
```

### 3.1 The possession timeline (the decision model)

As built (step 1, §6; `courtvision/nba/timeline.py`):

- **Hidden state** every quarter second: which team has the ball, and whether the ball is in that team's front
  court or its back court (with the time spent there).
- **Evidence**: where the broadcast camera aims (it follows the ball, so it shows which half the ball is in); the
  engine's shot results, rebounds, steals and turnovers, and 3-D flights that come down at a rim, as the moments
  where the ball may change hands. Jev's team answer in settled play is step 2. The plan named the engine's holder
  scores and the shot clock; the camera turned out far stronger than the holder scores (§6), and the shot clock is
  the scorebug.
- **Rules as transition costs**: possession changes rarely, and only at those moments; a team that gains the ball
  in its back court crosses half court within 8 seconds of live play.
- **Decoder**: Viterbi, a standard algorithm that finds the most likely sequence of states given all the evidence.
  Offline it sees the whole game; near-live it decides a few seconds behind. One bad frame cannot create a steal,
  because a flip must be supported by the evidence on both sides of it.
- **Fitting**: how often each change happens and how far to trust each cue are fitted on aligned play-by-play,
  replacing hand-set thresholds with about ten learned numbers.
- **Player level** later: the same decoder over the players of the team in possession, once identity holds.

### 3.2 Seeing the ball in hands: "focus on the player who has it"

- Once the timeline says who has the ball, run the ball detector on a full-resolution crop around that player. The
  detector sees 960×544 today, so the ball is about twice as large in a crop of the 1080p frame.
- Add a small "is this player holding the ball" classifier on player crops. Labels can come from Jev where it is
  confident, from the play-by-play (the shooter just before a shot), and from hand labels in the Studio.
- Add held-ball frames to `nba_ball_bench.py`, so the benchmark measures this case.

### 3.3 Shots and makes from the video

- A shot is a possession segment that ends in a ballistic flight to the rim (`ball_track.json` has the flights).
- Make or miss is decided from three sources together: the engine's rim cues, Jev's per-crop "ball in the net",
  and what happens next (after a make the other team inbounds from the baseline; after a miss someone rebounds).
  The timeline ties the shot's outcome to the possession that follows.
- Then train a small rim classifier on the rim crops. Every game gives about 180 field goals already labelled made
  or missed by the play-by-play. It takes over Jev's make/miss calls once it beats them on games it was not trained
  on.
- The scorebug becomes a check, used to evaluate and to train, not to produce the numbers.

### 3.4 Where Jev sits

- Offline and near-live only. Image input is pre-production: pin `jev-1.14.0` and rerun the probe on each new
  version.
- Cost estimate at one check per second of tracked play (about 3,300 s in this game): about 14 M tokens, roughly
  $0.60 a game.
- Teacher: its confident answers become labels for the small local models, so the number of Jev calls shrinks over
  time.
- Not used for per-frame tracking, player identity, or anything it would have to count or measure ("Jev is not a
  calculator", jev_api.md §6).

## 4. Build order (each step measured against the play-by-play before the next)

1. **Done (§6).** The team-level possession timeline on the camera and the engine's cues, fitted on the
   play-by-play, no Jev yet: steal, turnover and rebound F1, and team accuracy at settled moments.
2. Jev's settled-play team checks as extra evidence. Measure the change.
3. **Done (§7).** Make or miss from the engine's rim cues, what the timeline shows happens next and (optionally)
   Jev on rim crops. The trained rim classifier waits for more games.
4. Held-ball labels, the holder crop for the detector, and a held-ball benchmark.
5. Player identity (jersey numbers, re-identification). A separate track.

## 5. Running the probe

```bash
.venv/bin/python tools/nba_jev_probe.py runs/nba/<game> makes --n 40
.venv/bin/python tools/nba_jev_probe.py runs/nba/<game> possession --n 40 --jerseys white,blue
.venv/bin/python tools/nba_jev_probe.py runs/nba/<game> turnovers --jerseys white,blue [--offsets 2.5,4,6]
```

`--jerseys` gives the away and home jersey colours as Jev is told them. `--dry-run` writes the images only (no API
call). Results go to `runs/nba/<game>/jev_probe/` (`<task>.jsonl`, `<task>.summary.json`, and the images sent). The
key comes from `.env` (`Set up Jev.command`). The 2026-10-29 runs above cost about $0.09 in all.

## 6. Step 1: the possession timeline (built 2026-10-29)

`courtvision/nba/timeline.py`; on a game: `tools/nba_analyze.py runs/nba/<game> timeline --tag <run>`, then
`eval --tag <run> --file timeline.jsonl`.

In plain words: for every quarter second of the game the decoder holds one answer to "which team has the ball, and
is it in their front court or back court", and Viterbi picks the most likely story for the whole game at once. A
change of team is cheap at a moment where the ball can change hands (a shot result, a rebound, a steal, a flight at
the rim) and expensive anywhere else, so the camera decides whether the ball changed hands and those moments decide
when. Nothing is read from the scorebug: which basket each team attacks, and when they swap at half time, comes from
the video. The play-by-play only fits about ten numbers (how far to trust the camera, how often the ball changes
hands at each kind of moment, how long a team stays in its back court) and scores the result.

Lakers @ Warriors, the timeline fitted on one half of the game and scored on the other, both ways:

| | Engine alone | Engine + timeline |
|---|---|---|
| Team with the ball in settled live play (7,929 quarter-second steps) | 73.6% | **96.2%** |
| Changes of possession found within 2.5 s (195 true) | 206 found: precision 0.35, recall 0.37 | 156 found: precision 0.71, recall 0.56 |
| Turnovers F1 (22 true: 9 LAL, 13 GSW) | 0.18 (46: 25 / 21) | **0.42** (26: 10 / 16) |
| Steals F1 (8 true: 6 / 2) | 0.06 (27: 17 / 10) | **0.29** (20: 13 / 7) |
| Rebounds F1 (85 true) | 0.47 (141) | **0.52** (136) |
| Box-score error | 0.314 | **0.292** |

(Shot results keep the engine's labels here: `--shots off`. Events the engine inferred from the scorebug, a few
made shots and their attempts, are not used.) Decoding with the stored defaults (the whole game's fit, no fitting: how
a game without play-by-play is decoded) scores about the same here, but that is in-sample.

What the measurements on the way showed:

- On its own, the camera's half 3 s later named the team with the ball 97.7% of the time in settled live play; the
  engine's last possession gain 73.5%, the team of its nearest dribble or pass 76.4%.
- A team-only model put the change after a made basket 5-9 s late, when the other team brought the ball up, and those
  late changes became turnovers. Giving the state a back court fixed it: the other team now gains the ball at the
  basket, in its back court, which is the half the camera is already on.
- From the camera alone, home in its front court looks like away in its back court. The 8-second rule tells them
  apart: the back-court stay is gamma distributed (fitted mean 2 s) and capped at 12 s of live play.
- The engine's own possession gains are not used as change moments: they come when its holder logic sees the new team,
  seconds after the basket or the rebound.
- A turnover is a steal when the camera stays on the court for the next 8 s: 6 of the 8 true steals, 3 of the 14
  dead-ball turnovers (which usually cut to a replay or a stoppage).

What it does not fix:

- Shots and makes are the engine's (made-basket recall 0.50, 45 false shot results): step 3.
- Steals are still too many: about two thirds of the inferred turnovers sit where the play-by-play has no turnover,
  and the camera keeps rolling after most of them.
- Every false miss becomes a rebound, which caps rebound precision.
- One game. The settings are fitted per half; more games will show how well they carry over.

## 7. Step 3: made or missed from the video (built 2026-10-29)

`courtvision/nba/makes.py` (and `rim_jev.py` for Jev); part of `tools/nba_analyze.py runs/nba/<game> timeline`, which now
also decides every field-goal result (`--shots cross`, the default with the play-by-play), and asks Jev about each
one's rim crops with `--jev`.

The engine's own made / missed label leans on the scorebug: a make it sees waits for a score change and becomes a
miss without one, and a score change turns a miss it saw into a make (12 of the 19 real makes it called missed were
scorebug vetoes; 20 more events it simply inferred from score changes). So the new decision reads only what the
camera saw, in two small logistic models fitted on the play-by-play:

- **Real shot or not**, for each field-goal result the engine reports. 50 of its 198 camera-seen results are no shot
  at all (passes and loose balls near the rim); the ball usually does not change hands after them.
- **Made or missed**, for the real ones.

Their evidence: the engine's rim cues (the ball crossed the rim plane inside the ring, touched the rim, dropped at net
speed, the crossing margin); what the timeline shows happens next (did the ball change hands, how soon, how long the
other team stayed in its back court: an inbound after a make takes about 4 s, a rebound outlet about 2 s); and with
`--jev`, Jev's answers on ten rim crops (the in-net probability per crop, and made / missed). A make without an
assist takes the last pass of a teammate (the timeline's team) in the 3.5 s before the release. Results the engine
inferred from the scorebug are dropped.

Lakers @ Warriors, every model fitted on one half of the game and scored on the other:

| | Engine (labels lean on the scorebug) | Vision only | Vision + Jev |
|---|---|---|---|
| Made or missed, real shots | n/a | 81.8% | **89.9%** |
| Real shot or not, engine results | n/a | 79.8% | 81.8% |
| Field-goal makes: recall / precision (82 true) | 0.65 / 0.87 | 0.66 / 0.69 | **0.68 / 0.77** |
| Points (115 / 113 true) | 79 / 75 | **108 / 85** | 102 / 78 |
| Field goals made (41 / 41) | 32 / 29 | 44 / 34 | 42 / 31 |
| Assists (30 / 26) | 19 / 13 | 25 / 20 | 24 / 18 |
| Rebounds F1 (85 true) | 0.47 (141) | 0.53 (81) | 0.54 (84) |
| Box-score error | 0.314 | **0.249** | 0.269 |

(The possession timeline of §6 is under both new columns; on its own it gave 0.292.) Jev answered all 198 results:
2.4 M input tokens, $0.10, about 2.5 minutes; answers are kept in `runs/<tag>/jev_rim.jsonl`, so a re-run asks
nothing. It makes each call more accurate; the box error is a little higher with it only because it calls fewer
makes, and over- and under-calls cancel in the box counts.

What it does not fix:

- The engine never reports 30 of the 180 field goals, among them most put-backs: offensive rebounds are badly
  undercounted (0 / 6 for 11 / 12) and every lost make costs points. Finding those shots is the next gap.
- Free throws: 5 of 31 made ones are seen.
- Without Jev a made call is right 69% of the time, below the scorebug's 87%.
- With fewer false shots to explain them, more changes of possession become turnovers: turnovers F1 0.42 to 0.41,
  steals 0.29 to 0.26.
- Whether the engine saw a rebound right after a result, or a 3-D flight came down at a rim, lowered both models'
  held-out accuracy here, so neither is used.
- One game. `shot_models.json` (fitted on the whole game) lets another game be decided without play-by-play
  (`--fit none --shot-models ...`); how well it carries over is for the next game to show.

## 8. Shots the engine misses, and free throws (built 2026-10-29)

Still `courtvision/nba/makes.py` inside `nba_analyze.py timeline`: the decisions now run on **candidates**, not only on
the engine's results.

Where the shots were going missing (Lakers @ Warriors):

- The engine never reported 30 of the 180 field goals. Six were blocked (they never reach the rim), six were
  put-backs, and about half left a trace of the ball at the rim.
- It saw 10 of the 35 free throws, and reported 10 others as field goals (2 or 3 points each).
- The linked ball came within a rim width of a rim, over a few frames, 248 times. 44 of those visits had no engine
  result near them, and 27 of the 44 were real shots.

So the candidates are the engine's shot results the camera saw plus the **rim visits** it reported nothing at
(`rim_visits`, `candidates`). New evidence joins the old: the ball's path at the rim (how close it came, whether it
came down from above and left below), and the setting of a free throw: the camera off the court in the 5-20 s before
(the stoppage after the foul, shown as replays and close-ups) and almost no dribbling or passing around it. That pair
alone named 10 of 14 real free throws and only 3 of 162 real field goals. Three logistic models decide each candidate:
a real shot or not, a free throw or a field goal, made or missed. A shot added at a rim visit is a 2 (its distance is
unknown) of the timeline's team. A miss followed within 5 s by the same team's next attempt, the ball never changing
hands, now gets an offensive rebound in between (a put-back).

Lakers @ Warriors, every model fitted on one half and scored on the other:

| | Before (§7, vision only) | Vision only | Vision + Jev |
|---|---|---|---|
| Real shot or not, candidates | 79.8% | 78.8% | 79.6% |
| Free throw or field goal, real shots | n/a | **94.9%** | 94.9% |
| Made or missed, real shots | 81.8% | 84.8% | **87.6%** |
| Points (115 / 113 true) | 108 / 85 | **119 / 95** | 123 / 82 |
| Free throws made (18 / 13) | 3 / 2 | 11 / 6 | 11 / 5 |
| Made-shot F1 (field goals and free throws) | 0.63 | 0.68 | **0.71** |
| Shot-attempt F1 | 0.64 | 0.69 | **0.70** |
| Turnovers F1 / steals F1 | 0.41 / 0.26 | 0.44 / 0.32 | 0.45 / 0.32 |
| Offensive rebounds (11 / 12) | 0 / 6 | 4 / 2 | 4 / 2 |
| Box-score error | 0.249 | **0.203** | 0.235 |

Jev has now read all 245 candidates: 2.9 M input tokens, $0.12 in all, kept in `jev_rim.jsonl`. As in §7 it makes each
call more accurate, while the box counts happen to balance better without it.

What is left:

- Offensive rebounds: of the 23 real ones, 6 are credited; 13 follow a shot nothing detected (blocked shots, and shots
  the ball tracker lost near the rim), 3 led to a reset rather than a put-back.
- Free throws with the camera off the court (11 of 35) are not seen, and every free throw found is called made (31
  of 35 were).
- About 1 in 4 candidates is still wrongly kept or dropped: a ball through the hoop after the whistle, or in a
  stoppage, looks like a shot.
- The engine's blocks (1 right of 11) went with the shots they hung on: blocks are 0.
- One game.

