# Video workspace

Run `.venv/bin/python -m courtvision studio --open`, or double-click `Start CourtVision.command`.
The default address is `http://127.0.0.1:8000`. The server binds localhost; a non-loopback bind requires
`--auth-token`. Enter the token in dashboard settings. Detector setup is an explicit action from Start analysis.

Select a person under **People** in the Live side panel to highlight their brackets and dim other person tracks.
**Show all tracks** restores the full overlay; the **Tracking overlay** switch shows clean footage. The labels show
detector confidence, not identity confidence. The bottom marker is the bounding-box base, not a measured joint.
Simulator players are labelled separately and cannot be selected as camera detections.

## Finding your way around

The sidebar holds the views: **Live** (footage, the court lens, team comparison, and plays / people / narration /
insights in the side panel), **Box score** (player lines, shot chart, decision grades), **Board** (pinned answers),
**Game controls** (score and clock override, playback, substitutions), **Court mapping**, **Scoreboard** and
**Model lab**. Press `⌘K` (or `/`) to ask a question or run any command; `G` then `L`, `S`, `B`, `C`, `M`, `R` or `T`
opens a view; `?` lists every shortcut.

**Plays that need review.** The engine flags the plays it was unsure of: a change of possession it didn't see happen,
a rebound nobody caught, a shot whose result it lost, a basket only the scoreboard showed. Each flagged play says why
in plain words and has a **Check** button. Watch that moment, then choose **It's right**, change what is wrong and
**Save changes**, or choose **It didn't happen**. A play you have checked counts as confirmed by you and leaves the
list. The badges say how sure the engine was ("70% sure") and whether a play is confirmed yet.

The **court lens** draws the mapped floor in perspective. Players stand on their floor position with their number;
the ball is drawn at its tracked height (feet) with a floor shadow, so 3-D ball flights read as arcs. Drag to orbit,
double-click to reset, or pick the broadcast, overhead or baseline camera. Hover a player for their position and
speed; click for their card.

**Ask** answers on this Mac from the live game: team comparisons, leaders by stat, shot charts, player cards,
momentum, the review queue, tracking health and the training loop. Answers are cards built from live data blocks:
pin one to the Board and it keeps updating as the game runs. With Anthropic credentials (`scripts/mac/courtvision.sh
keys`), `⌘↵` sends a question to Claude (`claude-opus-5-5`, low effort), which chooses the blocks and writes a short
summary from a snapshot of the game; its numbers on the card still come from the game data. Questions the device
cannot place go to Claude automatically when a key is set. Each Claude answer is a billed API request.

The **Model lab** reads the NBA learning loop (the data of `tools/nba_loop.py status` and `next`). **Run next step**
starts `tools/nba_loop.py run --max-steps 1`; **Run every ready step** runs the loop until nothing is ready. Both apply
the loop's guards: a step does not start while another CourtVision job runs or with less than 25% of memory free,
and the watchdog stops it below 8%. **Stop** sends the loop its interrupt, which stops the step's whole process tree.
Logs go to `runs/nba/logs/studio-loop-*.log`. Closing the Studio stops a run it started.

## The possession analysis on the feed

When the open video has been analysed as an NBA game (`tools/nba_analyze.py <game> timeline`, [NBA.md](NBA.md)), the
Studio draws that analysis on the feed. It looks under `runs/nba` for a game folder whose `game.json` names this
video, or a copy of it (same size, same first and last MiB), and takes its newest run with a timeline. The panel at
the top left shows the team with the ball ("bringing it up" while it is still in its back court) and the score from
the shots decided from the video. When the game has a play-by-play, the panel also shows the play-by-play's team with
the ball, whether the two agree, and its latest line. Each decided shot stays on the feed for a few seconds: made or
missed, its value, P(made), and whether it was found at a rim visit the engine reported nothing at. Rebounds, steals
and turnovers show the same way. The band under the feed covers the whole game: who had the ball by the analysis
(upper) and by the play-by-play (lower), with a tick for each decided shot (tall: made). `B` hides or shows all of it.

**Watch** (at the top of the feed, in the Possession cell, or `W`) plays the recording itself at full speed with its
sound and draws the analysis for the time playing. The live analysis pauses meanwhile and resumes on **Live**. Click
the band to jump to any moment. The browser has to be able to play the file; H.264 MP4 works.

Everything shown comes from the video except the play-by-play. That is placed on video time through the scorebug
clock and is shown only to judge the analysis by. The held-out accuracies in the panel are the run's own: the team
with the ball in settled play, and made or missed on real shots. So far they come from one game. A video that no game
folder analyses shows "Not analysed": add it with `tools/nba_new_game.py` and run the steps in NBA.md through
`timeline`.

## Replay: the analysis done ahead

The switch at the top of the feed picks how it plays:

- **Live** analyses the video as it plays. With the NBA detector that is about 18 to 22 frames a second on an M3 Pro
  (the detector runs on the Neural Engine), so the feed runs at about two thirds of real time and cannot seek.
- **Watch** (`W`) plays the recording at full speed with the game's possession analysis (above).
- **Replay** (`R`) plays the recording at full speed with the live analysis done ahead of time: the tracking boxes,
  the plays, the box score, the court lens and the narration follow the video, and it seeks.

Replay needs the video analysed ahead. An upload starts that in the background when a detector is installed
(the NBA detector or a downloaded checkpoint; the first download of the starter detector still needs **Start
analysis**); any other video starts with **Analyse ahead** in the row under the status cells. The row shows how far
it has got and about how long is left; **Stop** ends it, and Replay works up to where it got. It runs the same pipeline
and settings as Live, unpaced, one video at a time, with Jev, Claude and spoken narration off: no cloud calls. It
takes a while, at the analysis's own pace: about 2 hours at every frame for a 79-minute game on an M3 Pro. **Every
2nd frame** or **every 3rd frame** finishes 2 or 3 times sooner; the boxes then update 15 or 10 times a second, and
the plays come from fewer frames. Replay plays up to what has been analysed so far and pauses there. Live analysis
at the same time slows both.

Each analysis ahead is kept in `runs/studio/<video id>/replays/<run>/`: `status.json`, the recording in minutes
(`chunks/NNNN.jsonl`: each analysed frame's boxes and every message the live view got, with its video time),
`sparse.jsonl` (events, narration, insights and captions, to rebuild the view at any moment for a seek) and
`analysis/` (the events and stats, as a live run saves them). A full game takes about 250 MB. Analysing again starts
a new run beside the old one. Closing the Studio stops a running analysis; Replay then works up to where it got.

## Jev setup

Double-click `Set up Jev.command`, or run `.venv/bin/python -m courtvision.credentials` in this folder.
Paste the TypeSafe key at the hidden prompt and press Return. It is saved to the ignored local `.env`
with owner-only permissions; existing settings are preserved. Studio checks this file for each new video
or simulator session, so start/restart a session after saving. Exported `TYPESAFE_API_KEY` takes precedence.
The footer distinguishes a configured key from an active live-mode session. Saving does not validate the
key or make an API call. Real requests use the existing decision layer; a rejected key falls back to rules.
Jev judges decisions; narration text still comes from local templates or separately configured Claude.

UI validation on 2026-10-28: 97 focused server, CLI and credential tests passed; the 15 studio/credential
tests passed again after the final backend adjustment. Browser checks covered real sample tracking,
person selection, clean-video mode, JSON download, simulator labels, and dark/light layouts at 1440 px
and 390 px. No horizontal page overflow was found at either width. The two-person sample is a UI and
detector smoke check, not a measured basketball accuracy benchmark. No live Jev request was made.

## What works

- Streamed file upload, decoder validation and an 8 GB upload cap, without buffering the file in RAM.
  Ordinary JSON/operator endpoints keep their 64 KB cap. Existing Origin, Host and token guards apply.
- A local RF-DETR Nano starter detector, warmed on a worker thread. Apple Silicon uses MPS when available.
  Preparation errors are shown in setup; repeated frame failures stop analysis with an actionable message.
- Image-space people/ball tracks before calibration. No court positions, basketball events or scores are
  invented from an uncalibrated image. Cut detection resets these preview track IDs between shots.
- A monitor built from the processed video frame, paired with its source timestamp. Bounding boxes are drawn
  over that same image; the time ruler shows processed source time. Overlays can be hidden.
- Pause/resume, speed, restart, full-screen monitor, recent footage, JSON export, and separate analysis folders.
- Court calibration using the existing landmark fitter, with a reference diagram and frozen source frame.
  Saved calibration and scoreboard regions are reused when reopening the same uploaded file. A recoverable
  pinhole camera provides projected rims; moving-camera calibration is handled by the broadcast wrapper.
- Existing event corrections, score/clock override, lineup controls, box scores, shot charts and narration.
- A simulator demo with explicit labelling. It uses synthetic state and is not presented as real CV evidence.

## Current limits

The starter checkpoint is trained on COCO. It detects generic people and sports balls, not NBA rosters,
referees, basketball rims or official basketball events. Names require validated identity evidence or manual
event attribution. Team clustering needs review. No NBA-specific accuracy or full-game throughput claim is made.
Pass `--config` with `perception.rfdetr.weights` or `perception.rfdetr.coreml_path` to use an existing local
checkpoint instead. The workspace honours that configuration rather than replacing it with the starter.

Spatial analysis needs an accurate court mapping. Four clicks fit mathematically even if the wrong landmarks
were selected, so fit RMSE alone is not evidence of correct calibration. Use additional well-spread points and
inspect the map. For fixed footage, camera geometry may be unrecoverable from noisy points; rim-based shot
events then need a trained rim detector or a supplied calibration with rim observations.

Broadcast mode tracks court lines and pauses spatial inference when the court is lost. Without an explicitly
configured scorebug, it cannot distinguish a replay that shows the court from live play. Scorebug fonts/layouts
must be configured and verified for the broadcast. Fouls, substitutions and timeouts require operator/feed input.

Every source frame is processed. The monitor reports measured inference rate; video can run behind wall time
when inference is slower than the input frame rate. The current monitor is a sequence of processed JPEGs and
does not play source audio. It supports pause/speed/restart, not random seeking of the stateful event engine.
Watch and Replay play the recording with sound and seeking; they show saved analysis, not the engine running now.

Footage persists at `runs/studio/<video id>/source.video`. Each run is saved independently under
`runs/studio/analyses/<run id>/`. There is no automatic retention deletion. Reopening a video starts a fresh
analysis, while previous output files remain available. JSON export also works during analysis.

## Validation record — 2026-10-26

- Imported round 5 (`425a779`) without discarding the newer README/banner commits on main.
- Ran one held-out aggregate check: truth3d **0.991**, mono **0.815**, synthetic video **0.792**.
  [Full results](bench/round5-heldout.md). The simulator's pump-fake changes alter the games, so round-4 deltas
  are not a controlled comparison. The held-out set was not used to tune this workspace.
- The existing full suite initially reported 2,104 passes, three expected failures and one short-clip failure.
  Round 5 changed the simulator RNG sequence; the integration fixture now uses a verified 4–7 second window
  containing settled possession. Its original event assertion remains intact.
- Exercised upload and RF-DETR inference on Roboflow's public
  [Supervision basketball sample](https://supervision.roboflow.com/latest/assets/): 1920×1080, 59.94 fps,
  approximately eight seconds. This is a two-person gym clip, not an NBA game. No labelled event score exists.
- A public NBA play-by-play URL returned a 15-second “Video not available” placeholder. Those files do
  not count as game validation. An NBA-hosted Raptors documentary also does not establish broadcast accuracy.
- Automated workspace tests exercise real video decoding, pre-calibration gating, pause, export, independent
  runs, saved calibration, upload validation, auth/origin/body limits and hub replay reset, using an injected
  detector. No model downloads or cloud calls occur in tests.
- Full suite: **2,111 passed, 3 expected failures**. Follow-up server/CLI/controller checks: **212 passed**;
  the six workspace tests also passed after adding the reconnect snapshot. Browser checks covered
  real-file upload, reopening footage, local tracking, pause, court setup and the simulator demo, at
  1440 px and 390 px viewport widths without horizontal page overflow.
- Jev/Claude credentials were absent. Their live APIs, billed narration and their production latency were
  not exercised. Browser video upload and local inference require neither key.

The README's latency table was re-measured with `simulate --mode truth3d --seconds 120 --seed 7`:
3,600 frames in 0.9 seconds processing wall time, complete-step p50/p95 0.1/0.5 ms, event-to-narration
p50/p95 650/3160 ms. These are offline synthetic timings, not real-footage throughput.
