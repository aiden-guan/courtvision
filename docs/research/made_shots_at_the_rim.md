# Made shots at the rim, a second game, and a faster analysis ahead

Work of the night of 2026-10-29/30 on an M3 Pro (18 GB), research use. Two games: Lakers @ Warriors (2024-12-25,
ESPN, 1080p 29.97 fps) and, new, Cavaliers @ Warriors (2017-06-12, NBA Finals Game 5, ABC, 720p 59.94 fps). Every
number is held out: fitted on one half of a game and scored on the other, or fitted on one game and scored on the
other, as each table says. The scorebug decides nothing here; it places the play-by-play on video time to score
against (docs/research/possession_decision_model.md).

## 1. What was wrong

**The Studio called most makes missed.** Its replay of Lakers @ Warriors (the analysis ahead of 2026-10-29) had 17
makes and 89 misses in the first 36 minutes, against about 45 real makes. The engine's camera had seen many of them:
it waits for the scorebug to confirm a make, and 30 makes it saw were turned into misses because no score change came
within 3 s; 34 score changes matched no shot. The make or miss was effectively left to reading the broadcast graphic,
and that matching failed.

**Two smaller faults in the analysis ahead:**

* it stopped for good on the first frame that raised an error (unpaced, the pipeline re-raises a frame's error): the
  Lakers @ Warriors replay ended at 36 of 79 minutes;
* a 60 fps video was analysed at every frame, twice the work of the 29.97 fps the detector, tracker and engine are set
  up for, and the detector, which sees each frame with the two before it, saw them 17 ms apart instead of the 33 ms
  it was trained on.

## 2. Evidence at the rim

The camera track projects each rim into every frame (`analyze.rim_boxes`), so the evidence can be read where a make
happens:

* **Rim clips** (`nba/rim_clips.py`, `tools/nba_rim_clips.py`): 37 frames (every other frame from 1 s before to 1.4 s
  after the ball's arrival), each cut 4.8 rim widths square about a point half a rim width under the rim, at 96 px.
  The rim is the same size and place in every frame whatever the zoom; a 720p and a 1080p game give the same crop.
* **Ball zones** (`rim_clips.zone_scores`, `makes.zone_features`): where the detector's ball candidates are about the
  rim, frame by frame: above it, at the ring, in the net, under it, beside it, high. Features: the ball in the net
  after the arrival, above the ring and then in the net, in the net and then under it, beside the ring after it
  (off the rim), and so on. It uses every candidate, not only the one ball the tracker follows, so a ball the net
  hides from the tracker still counts.
* **The rim model** (`nba/rim_model.py`, `tools/nba_rim_model.py`): a ResNet-18 (ImageNet weights to `layer3`, the
  cells kept rather than averaged, so where the ball is stays) on each frame of a rim clip, two 1-D convolutions over
  the frames and the maximum over time (the moment the ball passes can fall anywhere in the clip). Input: the frame in
  grey and its change from the frames before and after (the ball and the net moving), mirrored, shifted, zoomed and
  re-coloured in training. Trained on the real shots (made or missed).

Published work points the same way (research brief of 2026-10-30): models that look at the whole broadcast frame
reach about 75-77% made or missed even with 10,000 clips (Basketball-51); the methods above 90% look at the hoop.

Lakers @ Warriors, each half scored by models fitted on the other half (195 real shots, 99 made):

| Made or missed, real shots | Accuracy | AUC |
|---|---:|---:|
| Ball zones alone (logistic) | 84.1% | 0.90 |
| Rim model alone, ImageNet layers kept as they are | 85.1% | 0.90 |
| Rim model alone, ImageNet layers fine-tuned gently | 86.2% | |
| Rim model with a third class "no shot" (the rim model cannot tell a real shot from a near-rim non-shot: AUC 0.63) | 82.6% | 0.89 |

In the shot models of `nba/makes.py` (`nba_analyze.py timeline`, the same halves; vision only):

| | Before | + ball zones | + ball zones and rim model |
|---|---:|---:|---:|
| Made or missed, real shots | 84.8% | **88.1%** | 87.6% |
| Made-shot F1 (precision / recall), 113 true | 0.68 (0.72 / 0.65) | 0.71 (0.77 / 0.66) | 0.72 (0.81 / 0.65) |
| Real shot or not, candidates | 78.8% | 78.8% (zones lowered it to 75.9% there, so they are left out) | 78.8% |

## 3. The shot judge: made or missed in the Studio

`nba/live_judge.py`. The engine still finds the shots and calls them; the scorebug no longer confirms, vetoes or
invents them in the Studio (`EventEngineConfig.score_reconcile = False`). 2.5 s after each of the engine's results the
judge finds the frame where the tracked ball passed nearest the rim, reads the ball zones about it and, with a rim
model, runs the rim model once on the rim's frames (`FrameState.rim_crops`, cut as the clips are), and a logistic
model over those and the engine's own rim cues gives P(made). The engine keeps or overturns its call
(`EventEngine.judge_shot`: a make flipped from a miss loses its rebound and gains its assist, as with a scorebug
flip). Fitted by `tools/nba_live_judge.py`; the Studio uses `runs/studio/models/shot-judge.json`.

Lakers @ Warriors, the engine's results that are real shots (162), each half scored by a judge fitted on the other:

| Made or missed | Accuracy |
|---|---:|
| The engine's own call (the ball crossed the rim's line inside it) | 83.3% |
| Judge: the engine's rim cues and the ball zones | 89.5% |
| Judge: the same and the rim model | **92.6%** |

**In the Studio's pipeline itself**, on a stretch the judge was not fitted on: Lakers @ Warriors 1589-2189 s (25
real shots, 15 made), the judge fitted on the shots after 2300 s only (ball zones and rim cues, no rim model):

| Studio, 1589-2189 s | Makes called (right) | Made-shot F1 | Made or missed, results on real shots |
|---|---:|---:|---:|
| Before (the replay of 2026-10-29: scorebug confirmation) | 4 (3) | 0.32 | 7 of 16 (44%) |
| The engine alone, no scorebug | 9 (8) | 0.67 | 11 of 15 (73%) |
| **The engine and the judge** | **12 (11)** | **0.81** | **14 of 15 (93%)** |

The same test on Cavaliers @ Warriors 3000-3600 s (27 real shots, 17 made), the 60 fps original read at 30 frames a
second, with a judge (and its rim model) fitted on Lakers @ Warriors only:

| Studio, 3000-3600 s | Makes called (right) | Made-shot F1 | Made or missed, results on real shots |
|---|---:|---:|---:|
| The engine alone (the scorebug's scores were not read on this broadcast, so "before" is the same) | 6 (5) | 0.43 | 11 of 13 (85%) |
| The engine and the judge | 6 (5) | 0.43 | 10 of 13 (77%) |

Here the judge cannot help: the engine reported a result for 14 of the 27 shots, though the court was tracked around
24 of them. The shots it never reported are mostly makes (threes and layups among them): the ball's flight is not
picked up well on this 720p 2017 footage (the detector was trained on the 2024 game), and the judge only re-decides
the shots the engine reports.

## 6. Across games, and what is left

Each game scored by models fitted on the other game only (the honest test for a new game; two very different
broadcasts: 1080p 2024 at Chase Center and 720p 2017 at Oracle Arena):

| Made or missed | Lakers @ Warriors | Cavaliers @ Warriors |
|---|---:|---:|
| Rim model alone (all real shots; within a game it scored 85-86%) | 69.4% of both games' 366 together (AUC 0.79) | |
| Shot models (`makes.py`), real shots among the candidates | 75.1% | 72.8% |
| Shot models with the ball zones | 75.1% | 71.1% |
| Shot models with the ball zones and the rim model | 79.7% | 69.3% |
| The engine's own call, its results on real shots | 83.3% | 68.9% |
| The judge (rim cues and ball zones) | 74.7% | 76.7% |
| **The judge with the rim model** (what the Studio uses, now fitted on both) | **84.0%** | **77.8%** |
| (within each game, halves: the judge with the rim model) | 88.9% | 78.9% |

What this says: the evidence at the rim is strong within a game, and with two games it carries over only partly; the
features shift between a 1080p and a 720p broadcast (the detector's scores, the size of a ball in the net in
pixels). The judge with the rim model is the one that does not fall below the engine's own call on either game, so
it is the one installed (`runs/studio/models/shot-judge.json`, `rim-model.pt`; the possession analysis's shot models
`shot-models.json`, with the ball zones, all fitted on both games). More games, different from these two, are now the
largest gain: the research brief's advice was to watch the held-out accuracy as games are added, and with two it has
not levelled off.

What is left, in order of what it would gain:

1. **More games**, a few from other arenas, years and broadcasters (`nba_new_game.py`; then the steps in NBA.md),
   and the rim model, the judge and the shot models fitted again on all of them.
2. **Shots the engine never reports** on footage unlike its training: the student detector trained on teacher labels
   of the new game as well (`nba_teacher_labels.py`, `train_student`; the learning loop runs it), then the games
   detected again.
3. **Court tracking at arenas whose crowd looks like the floor** (§4): a floor model learned per video from its
   tracked frames.
4. **Real shot or not** stays the weakest decision (52-79% within a game): the rim's pixels cannot tell a shot from a
   pass near the rim (AUC 0.63), the shooter's release and the ball's arc have to.
5. **Free throws shown in close-up** (8 of the 19 makes no candidate found on the Lakers game): a rim detector for
   frames with no court, and a free-throw scene classifier.
6. **The scorebug layout of a new broadcaster in the Studio**: discovered when the video is added
   (`nba_scan_scorebug.py --profile auto` does it in 30 s), so its clock and period are read; today the Studio falls
   back to ESPN's 2024 bug.

## 4. The second game: Cavaliers @ Warriors, 2017 Finals Game 5

Added with `tools/nba_new_game.py` (ESPN event 400954514; the NBA.com play-by-play of a 2017 game is refused, HTTP
403, so the ESPN play-by-play is the truth: 229 shots, 93 / 85 field goals made / missed, 38 / 13 free throws). What it
took, and what it showed about other games:

* **The video**: a 2-hour 720p recording at 59.94 fps with no sound, re-encoded at 29.97 fps for the research tools
  (`runs/nba/2017-06-12-CLE-GSW/source30.mp4`, VideoToolbox, 3.2 Mb/s; `game.json` keeps the original as
  `video_original`): every stage is set up for 29.97 fps, and half the frames is half the work.
* **The scorebug** (ABC's 2017 Finals bug) was found with no hand measurement (`nba_scan_scorebug.py --profile auto`):
  clock, period, scores and shot clock, home and away confirmed by 31 play-by-play scores.
* **Alignment**: this bug shows a basket about 1 s sooner after the play-by-play clock time than ESPN's 2024 bug (the
  made shots' residuals had a median of +1.0 s). `game.json` now takes `score_update_lag_s` (1.0 here); the residuals
  then centre at +0.3 s.
* **Court tracking** found the camera on about half the frames of the game (the tracker's counts per part are in
  `camera_parts/part*.log`), against about 80% at Chase Center. Many of the frames it lost are ordinary game-camera
  views: the floor mask (court lines are sparse on the floor and dense in the crowd) does not hold at Oracle Arena,
  where the crowd in matching yellow shirts is as smooth as the floor, so the evidence carries crowd edges, and the
  search fits the court into the stands (`court_search.court_evidence`; the fixed-centre fits of three lost
  game-camera frames covered 24-45% of the court's lines, under the 50% a re-acquisition needs). Re-estimating the
  camera centre from 60 frames did not change it. A floor model learned per video from its tracked frames is the
  fix to try; until then such arenas yield fewer tracked frames and fewer shots seen.
* **Teams.** The engine's two team clusters were fitted once, on the first 30 player crops of the game, and kept.
  Here that first fit went wrong: the shots' teams agreed with the play-by-play on 40%, 83% of the engine's team
  events went to one team, and the possession timeline, which learns which basket each team attacks from those
  events, found no half-time swap: 38.8% right in settled play, worse than a coin. Jersey colours in `game.json`
  (`home_color_bgr` white, `away_color_bgr` black) name the clusters but do not mend them (78%). Now the clusters are
  refitted as crops accumulate (`TeamConfig.refit_at`: 150, 600, 1,500, 3,000 crops; named by the colours, or as the
  clusters before, so they cannot swap; `nba_analyze.py events --team-refit`, and the Studio): the shots' teams agree
  on 83%, the swap is found at 3,286 s (the play-by-play's shots show it between 3,000 and 3,300 s), and the
  timeline is right on **75.8%** of settled play (the engine alone 61.0%). The camera, the timeline's main evidence,
  is lost on about half of this game's frames, which keeps it below the Lakers game's 96.2%. On Lakers @ Warriors,
  whose first fit was sound, the refits raise the team with the ball a little (the engine alone 73.6% to 78.3%, the
  timeline 96.2% to 97.3%) but the event engine's other calls move with the team labels: made-shot F1 0.68 to 0.65,
  turnovers 0.44 to 0.38, box-score error 0.203 to 0.227. So the research runs keep it a switch, and the Studio, where a
  bad first fit on a video nobody checked is the larger risk, has it on.

## 5. A faster and sturdier analysis ahead

* **About 30 frames a second.** A 50-60 fps video is read at every 2nd frame (`preanalysis.rate_stride`), in the
  analysis ahead and in Live: half the work, and the detector's input frames are 33 ms apart as in its training
  (at 60 fps it saw them 17 ms apart). The replay's choices now say 30, 15 or 10 frames a second.
* **The court tracked ahead in a process of its own** (`perception/court_worker.py`): the same fixes as in process,
  frame for frame (identical recordings), 1.46 times as fast on Lakers @ Warriors (docs/research/analysis_speed.md
  §5). Together with the frame rate, a 60 fps game takes about a third of the time it took.
* **One bad frame no longer ends it**: logged, counted (`failed_frames`, `first_error` in the status) and passed over;
  90 failures in a row stop it.
* **Watch for any video analysed ahead** (`nba/auto_analysis.py`): the analysis ahead keeps the camera and the
  detector's ball candidates of every frame, and when it ends builds the possession analysis a research game gets
  (linked ball, possession timeline with the stored settings, every shot decided by the shot models fitted on the
  research games): 4.6 s for 2 minutes of video, a few minutes for a game. Checked end to end through the Studio's
  own preparation of a 2-minute Lakers @ Warriors clip: the replay was done with no failed frame, the judge judged 6
  shots, and Watch found the analysis (its three misses matched the play-by-play's; the clip's last make fell at its
  end, with nothing after it to decide it by).
