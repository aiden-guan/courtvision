# NBA broadcasts: analysis and evaluation against the box score

CourtVision can analyse a full NBA TV broadcast and score itself against the official box score and play-by-play.
This page covers how it works, how to run it and what it measures. Everything here is **research use**: NBA footage
and statistics may not be shipped in a product (docs/TRAINING.md §1).

## What the broadcast pipeline does

| Stage | Module | What it solves |
|---|---|---|
| Scorebug | `perception/scorebug_ocr.py`, `perception/scorebug_layout.py`, `perception/vision_ocr.py` | Reads the bug (scores, period, game clock with tenths, shot clock) with Apple Vision text recognition. No glyph training. The ESPN/ABC layout is hand-measured; other broadcasters' layouts are discovered per video (see "Adding a game"). Its layout moves when the shot clock is off, so fields are told apart by pattern and order rather than by fixed boxes. |
| Court | `perception/court_search.py`, `perception/broadcast_court.py` | **No clicks.** The main NBA camera sits at half court, high behind the near sideline, and only pans, tilts and zooms. The camera centre is estimated once per game; each frame then needs only pan, tilt, roll and zoom. |
| Detection | `perception/student.py`, `training/train_student.py` | A ResNet-18 CenterNet trained on this footage. It sees the current and two previous frames (motion makes a blurred ball stand out), at 960x544. On a Mac it runs on the Neural Engine through Core ML (exported once per weights and cached; `nba_analyze.py detect --backend torch` gives the MPS detections of before bit for bit; [docs/research/analysis_speed.md](research/analysis_speed.md)). |
| Ball | `nba/analyze.py` (`link_ball`), `perception/ball3d.py` | One ball path through the per-frame candidates (dynamic programme; the camera's turn between frames is removed from the motion cost). Ballistic stretches are fitted in 3-D: gravity fixes the depth a single camera cannot see. Candidates centred in the scorebug can be dropped first (off by default; see "The scorebug filter"). |
| Players | `nba/analyze.py` (`rectify_box`, `is_referee`) | Boxes re-drawn as a standing player at the box centre through the camera (feet within a few px); referees (grey with black stripes) are kept out of the teams. |
| Events | `events/engine.py` | The existing monocular event engine on these frame states. |
| Scorebug fusion | `nba/scorebug_events.py`, `nba/fusion.py` | Score changes are made baskets of a known team and value (a shot the camera saw goes before the engine's own guess); shot-clock resets to 24 mark rim touches (a reset that starts counting at once is a change of possession), and an engine miss the shot clock shows no sign of is dropped; a shot takes the team attacking the half the camera aims at when the ball reaches the rim; after a miss, a reset to 14 (the offence kept the ball) or a count from a fresh 24 (the defence got it) decides the rebound, and an engine rebound with no miss just before it (after a make, or after no shot) is dropped; a steal or turnover needs a reset to 24 that no shot or score explains, and such a reset in play with no shot before it (the camera then leaving the half) is a steal; assists follow the corrected makes and take the last pass in the scorer's possession (no shot-clock reset since it), whatever team the engine gave the pass. Each pass has a switch in `FusionConfig`; `tools/nba_fusion_ab.py` measures each one on a game in seconds, and `tools/nba_bug_study.py --split` checks the shot-clock rules on the part of a game they were not tuned on. |
| Possession timeline | `nba/timeline.py` | Which team has the ball, decided once over the whole game (Viterbi) from where the camera aims, the engine's shot results, rebounds and steals, 3-D flights at the rim and the 8-second back-court rule. Nothing is read from the scorebug: which basket each team attacks, and the swap at half time, come from the video. The engine's steals, turnovers, rebounds and team labels are rebuilt from it (`nba_analyze.py timeline`); its settings are fitted on aligned play-by-play (docs/research/possession_decision_model.md §6). |
| Shots | `nba/makes.py`, `nba/rim_jev.py` | Every shot decided from the video alone: the engine's shot results and the linked ball's visits to a rim it reported nothing at, each a shot or not, a free throw or a field goal (the camera off the court before it, a stoppage, and no dribbling around it), made or missed, from the engine's rim cues, the ball's path at the rim, what the timeline shows happens next and, with `--jev`, TypeSafe Jev's answers on rim crops. Three logistic models fitted on the play-by-play (docs/research/possession_decision_model.md §7-8); the engine's own labels lean on the scorebug. |
| Rim | `nba/rim_clips.py`, `nba/rim_model.py`, `nba/live_judge.py` | What the camera saw at the rim: clips that follow each rim through pans and zooms (always the same size and place), where the detector's ball candidates are about it (above, in the net, beside), and a ResNet-18 on the rim's frames. `timeline --zones --rim NAME` adds them to the shot models; the **shot judge** uses them in the Studio to keep or overturn the engine's make or miss 2.5 s after it ([research note](research/made_shots_at_the_rim.md)). |
| Truth | `nba/espn.py`, `nba/evaluate.py` | ESPN play-by-play and box score, placed on video time through the scorebug clock. |

### Court registration in more detail

1. **Line evidence.** Court lines are thin and differ in brightness from the floor around them: white lines on
   dark paint, or the dark blue lines of Chase Center on light wood. A top-hat and a black-hat of the Lab lightness
   catch both. The crowd is dense with such structures while the floor carries only its lines, so the floor mask
   keeps pixels whose neighbourhood has few strong responses.
2. **Search.** Pan, tilt and focal length are searched over a grid. Candidates are scored at three pyramid levels
   by bidirectional, orientation-aware chamfer matching: model lines must land on line pixels of the same
   direction, and line pixels cast back onto the floor must land on court lines. One-way matching was fooled by a
   zoomed-out court squeezed onto the busy part of the image.
3. **Refinement.** A wide-basin pass on the line distance map, then matches along each projected line's normal
   with a robust scale that shrinks with the search radius.
4. **Tracking.** Each frame refines the previous camera, predicted with its velocity. On a flagged cut the last
   camera is refined first and kept when it still fits as well as a re-acquisition must (most flagged cuts in a
   game are fast pans or flashes); a real cut, or a frame that no longer fits, drops it. Re-acquisition first tries
   recently seen views, then a pan/tilt/zoom search at the known centre, and accepts only a camera the game camera
   can be (focal length in the prior's field of view, looking down into the aim box, nearly level, a tight fit).
   Close-ups and replays fail it and stay "lost", which the broadcast gate treats as not live.
   Tracked frames read line evidence only around the court projected through the predicted camera, grown by
   `track_margin_ft` (4 ft by default; `nba_track_court.py --track-margin-ft 0` restores the frame-wide floor mask),
   and after a loss in play re-acquisition is also tried 1, 4, 10 and 20 frames later
   (`--no-reacquire-after-loss` turns that off). Both were measured only on the fixed-seed synthetic arenas of
   `tools/court_bench.py`, not yet on broadcast footage.

## Adding a game

Any NBA game recording (any broadcaster, arena, resolution or frame rate) joins the learning loop with one command;
it finds the ESPN event from the date and the teams (guessed from the file name when not given), writes
`runs/nba/<date>-<AWY>-<HOM>/game.json` with the ESPN box score and the NBA.com shot locations, and names the next
command:

```bash
.venv/bin/python tools/nba_new_game.py "/path/to/recording.mp4" --check     # validates only
.venv/bin/python tools/nba_new_game.py "/path/to/recording.mp4" [--date YYYY-MM-DD --teams AWY,HOM]
.venv/bin/python tools/nba_new_game.py --batch games.csv                    # video,date,teams per row
.venv/bin/python tools/nba_loop.py run --game <YYYY-MM-DD-AWY-HOM>
```

What makes other games work without hand tuning:

* **Scorebug.** `nba_scan_scorebug.py --profile auto` (the default) finds the broadcast's scorebug from text
  recognition of whole frames at ~30 moments: the fields are told apart by how they behave (a game clock that counts
  down, scores that never decrease, a 0-24 shot clock beside the clock), home and away by the teams' codes and the
  play-by-play scores. The layout is saved as `scorebug_profile.json` (check its notes for "ASSUMED" or "GUESSED");
  the court tracker masks that bug (`nba_track_court.py --scorebug-profile` names another) and the studio reads with
  it. When discovery fails it falls back to the ESPN 2024 layout with a warning.
* **Frame rates and sizes.** Settings counted in frames are defined at 29.97 fps and converted (59.94 / 50 fps work
  alike); pixel settings scale with the frame height (720p to 4K). Variable-frame-rate recordings use decoder
  timestamps; converting them first is faster (`ffmpeg -i in.mp4 -fps_mode cfr -r 30000/1001 out.mp4`). Interlaced
  sources may decode wrongly in OpenCV: check `probe_video`'s warnings (printed by the tracker and the teacher tool).
* **Teams.** Without jersey colours in game.json (`home_color_bgr`, `away_color_bgr`, BGR) the engine's home / away
  clusters are arbitrary; the fusion's `team_map` swaps them when the scorebug's baskets say so. The clusters are
  fitted on the first 30 player crops; `nba_analyze.py events --team-refit` refits them as crops accumulate, which a
  game whose first fit splits on something else needs (Cavaliers @ Warriors 2017: the possession timeline went from
  38.8% to 75.8%; the Studio always refits).
* **Score lag.** The play-by-play's made shots are placed through the scorebug's score changes, which ESPN's 2024 bug
  shows about 2 s after the basket. A bug that is faster or slower sets `"score_update_lag_s"` in game.json (ABC's
  2017 Finals bug: 1.0); check it with the median of the made shots' residuals (`truth(...)[1].aligned`).

## Running it on a game

```bash
G=runs/nba/2024-12-25-LAL-GSW
VIDEO=runs/studio/<video id>/source.video
.venv/bin/python tools/nba_scan_scorebug.py $VIDEO --out $G/scorebug.jsonl
.venv/bin/python tools/nba_track_court.py $VIDEO --scorebug $G/scorebug.jsonl --out $G/camera_parts/part0.jsonl
.venv/bin/python tools/nba_analyze.py $G detect
.venv/bin/python tools/nba_analyze.py $G ball
.venv/bin/python tools/nba_analyze.py $G events --tag baseline
.venv/bin/python tools/nba_analyze.py $G eval --tag baseline
.venv/bin/python tools/nba_analyze.py $G timeline --tag baseline     # possession and makes, no scorebug [--jev]
.venv/bin/python tools/nba_analyze.py $G eval --tag baseline --file timeline.jsonl
.venv/bin/python tools/nba_diagnose.py $G --tag baseline
```

The rim evidence and the shot judge (research note: [made_shots_at_the_rim.md](research/made_shots_at_the_rim.md)):

```bash
.venv/bin/python tools/nba_rim_clips.py $G --tag baseline                    # rim clips of every candidate and shot
.venv/bin/python tools/nba_rim_model.py --games <g1>,<g2> --folds games --name rim-x --real-only \
    --config '{"motion": true, "backbone_lr": 3e-5, "epochs": 30, "lr": 1e-3}' [--save runs/models/rim/rim.pt]
.venv/bin/python tools/nba_analyze.py $G timeline --tag baseline --zones --rim rim-x
.venv/bin/python tools/nba_shot_models.py --runs <g1>:<tag>,<g2>:<tag> --zones --out runs/models/shot_models.json
.venv/bin/python tools/nba_live_judge.py --games <g1>,<g2> --tag baseline --rim rim-x --rim-model runs/models/rim/rim.pt
```

`nba_shot_models.py` and `nba_live_judge.py` score each game with models fitted on the others before fitting on all
of them. The Studio reads `runs/studio/models/shot-judge.json` (the judge; a `rim_model` path in it is taken from the
file's folder) and `runs/studio/models/shot-models.json` (the shot models of the possession analysis it builds after
an analysis ahead).

To watch the result, open the recording in the Studio ([STUDIO.md](STUDIO.md)): it finds the game folder whose
`game.json` names the video and draws the newest timeline run on the feed, and **Watch** plays it at full speed. To
render a clip instead, with the broadcast's sound, to `$G/videos/<tag>_<start>-<end>.mp4`:

```bash
.venv/bin/python tools/nba_overlay.py $G --tag baseline --auto 180              # the 3 minutes with the most going on
.venv/bin/python tools/nba_overlay.py $G --tag baseline --start 600 --end 780
```

Both show the team with the ball by the timeline and by the play-by-play, and each decided shot. The play-by-play
the timeline was scored against is kept in `timeline.json` (`reference`) for them.

`$G/game.json` names the video and the ESPN event id, plus optional team colours (BGR) for the team classifier.
`espn_summary.json` comes from `courtvision.nba.espn.fetch_summary(event_id, out=...)`. The court tracker can run
in parallel chunks (`--start` / `--duration`), one output file per chunk.

The court search scores its candidate cameras on several threads (`COURTVISION_THREADS`, by default one per core
but one, at most 8): set it to 2 or so when several `nba_track_court.py` parts run at once.

On a Mac, `tools/nba_mac_jobs.sh` runs the heavy steps one at a time (`q1`, `retrack-q1`, `train`, `retrack-full`,
`full`); it refuses to start while another CourtVision job runs or when little memory is free. Several heavy jobs
at once ran an 18 GB Mac out of memory.

To train the student detector on a game:

```bash
.venv/bin/python tools/nba_teacher_labels.py $VIDEO --scorebug $G/scorebug.jsonl --out $G/teacher
# ball labels: courtvision.training.ball_labels.link_ball_path over each clip
.venv/bin/python -m courtvision.training.train_student --data $G/teacher --out runs/models/student
# continue an interrupted run (same --epochs and data)
.venv/bin/python -m courtvision.training.train_student --data $G/teacher --out runs/models/student --resume
```

The teacher is RF-DETR Small (COCO) at 1312x736 on 150 clips of 36 frames. Its ball candidates become labels only
where one consistent, ball-coloured path links them. Fast gaps are filled, so the student also learns the blurred
balls the teacher misses.

### The scorebug filter

The scorebug is an opaque overlay, so no ball candidate centred in it is the game ball, but a team logo in it can
score higher than the real ball. When both stand still (a free throw, the ball in the shooter's hands) the linker can
follow the logo for as long as they do (a test in `tests/test_nba_analyze.py` shows it on synthetic candidates).
`ball --scorebug-filter on` drops the candidates centred in the bug before linking (`analyze.without_balls_in`). The
boxes are the ones `nba_track_court.py` keeps out of the court evidence (`bug_exclude`: the game's
`scorebug_profile.json`, else the ESPN 2024 bug; `--scorebug-profile` names another), in the frames' pixels, and the
ball track records them (`scorebug_boxes`, which `timeline --zones` and `nba_live_judge.py` then apply to their
candidates too). Two limits: the boxes carry the court mask's margins (10 px of a 1080-line frame left and right, half
a box's height above and below; ESPN's lower row), so a real ball centred in those margins, outside the bug itself, is
dropped as well; and offline the filter applies on every frame with the game camera, whether the bug is shown then or
not. In the Studio the same boxes apply only while the OCR reader sees its bug
(`perception.broadcast_config.drop_scorebug_balls`).

Both switches default to `scorebug_cover.DROP_BALLS_IN_BUG`, which is off until the filter is measured on the games.
For each game, with the `events` flags of its other runs (for example `--team-refit` on Cavaliers @ Warriors):

```bash
G=runs/nba/2024-12-25-LAL-GSW
.venv/bin/python tools/nba_ball_bench.py $G/teacher --out $G/ball_bench_scorebug.json   # off and on
for f in off on; do
  .venv/bin/python tools/nba_analyze.py $G ball --scorebug-filter $f --ball-track ball_track_bug-$f.json
  .venv/bin/python tools/nba_analyze.py $G events --tag bug-$f --ball-track ball_track_bug-$f.json
  .venv/bin/python tools/nba_analyze.py $G eval --tag bug-$f
  .venv/bin/python tools/nba_analyze.py $G timeline --tag bug-$f
  .venv/bin/python tools/nba_analyze.py $G eval --tag bug-$f --file timeline.jsonl
done
```

`--ball-track` keeps both ball tracks beside `ball_track.json`, which stays as it is; `events` records the ball
track in `run.json`, and `fuse` and `timeline` read the run's. `ball` prints how many candidates the filter dropped
and on how many frames one of them was the best candidate. The bench scores both from the same detections and counts
the held-out frames whose best candidate or linked ball lies in the bug, and the labels that do (a label there is one
only the filter can lose). It scores only frames where the teacher found the ball, and a ball held in a player's
hands is not labelled, so the filter's effect on held balls, such as a free throw's, can only show in the whole-game
`eval` and `timeline` numbers.

## The learning loop: better with every game

Every analysed game becomes training data for the next analysis. Three things learn:

| What learns | From | Tools |
|---|---|---|
| The ball and person detector (student) | Teacher labels (RF-DETR Small at high resolution) on random clips; teacher labels on clips around the shots, rebounds and turnovers an analysis missed, invented or gave the wrong team (hard examples); the student's own ball where a 3-D flight or a confident, consistent path pins it down (pseudo labels, weighted below the teacher's and never used to score a model) | `tools/nba_hard_examples.py`, `tools/nba_teacher_labels.py --times`, `courtvision.training.pseudo_labels`, `courtvision.training.train_student` |
| The event engine's decisions (shot, make, three, rebound, steal, block) | The engine's cues on each game labelled by the play-by-play placed on video time (and NBA.com shot locations); labels that hinge on the alignment error or a team mistake are left out | `courtvision.training.nba_cue_rows`, `courtvision.training.fit_cue_models` |
| The scorebug fusion settings (windows, thresholds, passes) | The box scores of the analysed games, with leave-one-game-out checks | `tools/nba_calibrate_fusion.py`, `nba_analyze.py fuse --fusion-file` |

`tools/nba_loop.py` runs it: `status` and `next` show the plan, `run` takes the missing steps one at a time (refused
while another job runs or with less than 25% memory free; stopped by a watchdog under 8%; resumable), and fits new
model versions when their inputs change. Every version keeps its provenance in `runs/models/registry.json`, every
evaluation goes to `runs/nba/leaderboard.jsonl`, and `promote` makes a version current only when its mean box error
over the games it was not fitted on is lower. With one game nothing is held out, so the loop starts to learn for
real from the second game: add a folder with a `game.json` (`video`, `espn_event`) and `run` again.

```bash
.venv/bin/python tools/nba_loop.py status
.venv/bin/python tools/nba_loop.py run --game <YYYY-MM-DD-AWY-HOM>   # one game's stages, resumable
.venv/bin/python tools/nba_loop.py run                               # then the fits and candidate evaluations
.venv/bin/python tools/nba_loop.py promote <version>
```

To check single pieces by hand: `tools/nba_fusion_ab.py` (each fusion switch flipped on a run, in seconds),
`tools/nba_bug_study.py` (what the play-by-play says happened at each shot-clock reset, to check the fusion's
shot-clock rules) and `tools/nba_jev_probe.py` (TypeSafe Jev's answers from frames on made shots, possession and
the engine's turnovers, against the play-by-play; see docs/research/possession_decision_model.md).

## In the studio

Copy the trained student to `runs/studio/models/nba-student.pt` and the studio uses it instead of the COCO starter.
With **Broadcast footage** on and no saved calibration, the court is registered automatically. Six frames, a second
apart, give the camera centre, which is saved next to the video's calibration. On macOS the scorebug is read with
Vision, using the `scorebug_profile.json` that `nba_scan_scorebug.py` saved for the video (beside it or in its game
folder), else the ESPN 2024 layout; with `perception.broadcast_config.drop_scorebug_balls` on, ball detections
centred in that bug are dropped while it is shown ("The scorebug filter"). The **Model lab** view shows the learning
loop above and can run its next step.

## Results

See the "NBA broadcast accuracy" section of the README for the current numbers on Lakers @ Warriors (2024-12-25).
