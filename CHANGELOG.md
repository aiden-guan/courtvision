# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased] — Plays that need review say why, and what to do

- Each flagged play in the Studio says why the engine flagged it, in plain words (for example "The ball changed
  teams, but the engine didn't see how"), and has a **Check** button that is always visible. The check dialog asks
  one question, with **It's right**, **Save changes** and **It didn't happen**. Badges read "Unconfirmed", "70%
  sure" and "Needs review" instead of "provisional", "70%" and "Review". A change of possession says how the team
  got the ball ("Home ball off a defensive rebound").
- Fixed: a play confirmed or corrected in the Studio stayed in the review list and its count. Checking it now clears
  its flag.

---

## [Unreleased] — The possession analysis in the Studio, and watching it at full speed

- The Studio draws an analysed game on its feed (`courtvision/nba/results.py`, `static/possession.js`). Opening a
  recording that a game folder under `runs/nba` analyses (the same file, or a copy of it) shows the team with the
  ball, the score from the shots decided from the video, each decided shot (made or missed, its value, P(made),
  whether it was found at a rim visit), rebounds, steals and turnovers, and a band of the whole game under the feed.
  With a play-by-play, its team with the ball and latest line are shown beside them. **Watch** (`W`) plays the
  recording itself at full speed with sound and seeking (`GET /api/studio/video/<id>`, served in ranges; the token
  may come in the query, as for the websocket), with the live analysis paused meanwhile.
- Added `tools/nba_overlay.py`, which renders the same onto a clip of the game with ffmpeg. `--auto SECONDS` picks the
  stretch with the most going on.
- `nba_analyze.py timeline` now keeps the play-by-play it was scored against in `timeline.json` (`reference`), and
  the stretches the team with the ball spends in its back court (`back_court`).

---

## [Unreleased] — Possession timeline and shots from the video (no scorebug), and the Jev frame probe

- Added the possession timeline (`courtvision/nba/timeline.py`, `tools/nba_analyze.py <game> timeline`): which team
  has the ball is decided once over the whole game by Viterbi, from where the broadcast camera aims, the engine's
  shot results, rebounds and steals, 3-D flights at the rim and the 8-second back-court rule, with nothing read from
  the scorebug. Steals, turnovers, rebounds and team labels are rebuilt from it. On Lakers @ Warriors (fitted on one
  half, scored on the other): the right team in 96.2% of settled live play (the engine alone 73.6%), turnovers F1
  0.18 → 0.42, steals 0.06 → 0.29, rebounds 0.47 → 0.52, box-score error 0.314 → 0.292.
- Added shots from the video (`courtvision/nba/makes.py`, part of `nba_analyze.py timeline`): the engine's shot
  results and the linked ball's visits to a rim it reported nothing at are each a shot or not, a free throw or a
  field goal (told apart by their setting: the camera off the court before, no dribbling), made or missed, from the
  engine's rim cues, the ball's path at the rim and what the timeline shows happens next; `--jev` adds TypeSafe
  Jev's answers on rim crops (`courtvision/nba/rim_jev.py`). The engine's own labels lean on the scorebug. A miss
  followed by the same team's next attempt is a put-back after an offensive rebound. Made or missed right on 84.8%
  of real shots (87.6% with Jev), free throw or field goal on 94.9%, points 119 / 95 for 115 / 113 (the engine alone
  79 / 75), box-score error 0.314 → 0.203.
- Added `tools/nba_jev_probe.py`: TypeSafe Jev's answers from frames (made shots, the team with the ball, the
  engine's turnovers) scored against the play-by-play.
- See [the design note](docs/research/possession_decision_model.md).

---

## [Unreleased] — Studio redesign: court lens, Ask and the model lab

- Rebuilt the Studio dashboard as a keyboard-first workspace: a sidebar with seven views (Live, Box score, Board,
  Game controls, Court mapping, Scoreboard, Model lab), a broadcast-style scorebug with a draining shot clock, and a
  Live view with the camera monitor, a score-margin timeline, the court lens and a side panel for plays, people,
  narration and insights. Graphite chrome; colour comes only from the teams, the ball and status. Light and dark.
  Local Archivo and IBM Plex Mono type; `static/studio.css` is folded into `static/styles.css`.
- Added the court lens (`static/lens.js`): the tracked floor in perspective as a lit night arena, players as pins with
  trails, and the ball drawn at its tracked height with a shadow and flight trail. Drag to orbit; broadcast, overhead
  and baseline cameras. The empty stage runs the same lens on a scripted possession.
- Added generative cards (`static/genui.js`): ⌘K / `/` opens Ask, which runs commands or answers questions about the
  game as live cards built from typed blocks (team duel, leaders, shot chart, plays, player card, momentum, tracking,
  model lab). Answers are composed on the device; with Anthropic credentials, open questions can go to Claude
  (`POST /api/ask`, structured output on `claude-opus-5-5` at low effort). Data blocks name live sources, never
  numbers, so cards cannot show figures the game data does not hold. Cards pin to the Board and stay live. The Live
  side panel opens with a briefing composed from the workspace state (setup steps, review queue, runs, leaders).
- Added the model lab (`courtvision/server/lab.py`, `GET /api/lab`): the NBA learning loop's model versions, each
  game's stages, the next step and its command, free memory, other jobs and the leaderboard. It can start
  `tools/nba_loop.py run` (one step, or every ready step) with the loop's own memory guard and watchdog, stop it,
  show its log, and promote a candidate that passes the held-out rule.
- `G` then a letter switches views; `P`, `M`, `D`, `A`, `O`, `F`, `]` and `?` act anywhere outside text fields.
- See [the workspace guide](docs/STUDIO.md).

---

## [Unreleased] — NBA broadcasts: analysis, evaluation and the learning loop

- Added NBA broadcast analysis scored against the ESPN box score and play-by-play (research use;
  [docs/NBA.md](docs/NBA.md)): scorebug reading with Apple Vision, click-free court registration and tracking, a
  ResNet-18 CenterNet student detector trained on the footage, one ball path with 3-D ballistic fits, and scorebug
  fusion of the event engine's output.
- Added the learning loop (`tools/nba_loop.py`): every analysed game is scored and mined for training data, and a
  model becomes current only when it does better on games it was not fitted on. `nba_mac_jobs.sh` takes
  `DET_WORKERS` for parallel full-game detection.
- Any NBA game recording joins with one command (`tools/nba_new_game.py`, or `--batch` for a CSV), from the ESPN
  event found by date and teams. Scorebug layouts are discovered for any broadcaster (`scorebug_profile.json`, else
  the ESPN 2024 layout); frame rates (29.97 / 59.94 / 50 fps, variable-rate video) and sizes (720p to 4K) are
  converted.
- Scorebug fusion rules: shot-clock resets mark rim touches (a reset that counts at once is a change of possession),
  engine misses with no sign on the shot clock are dropped, a shot takes the team attacking the camera's half at the
  rim, rebounds follow the reset to 14 or 24, steals and turnovers need a reset no shot explains, assists take the
  last pass in the scorer's possession, and `team_map` swaps engine teams the scorebug contradicts. Team rebounds and
  dead-ball turnovers are off by default: their triggers failed on Q2-Q4.
- Court tracking masks the scorebug the scan found (`nba_track_court.py --scorebug-profile` names another), reads
  line evidence only around the projected court, and re-acquires sooner after a loss in play. `tools/court_bench.py`
  is a fixed-seed synthetic arena benchmark for it; those gains are synthetic only so far.
- Measured on one game, Lakers @ Warriors (2024-12-25): box-score error 0.305 before the fusion work, **0.068**
  after, on the first quarter (in-sample: the fusion rules were tuned on it); ball within 15 px on 95.7% of 256
  held-out labelled frames. See [NBA broadcast accuracy](README.md#nba-broadcast-accuracy).

---

## [Unreleased] — video studio and round 5

- Made the Mac launcher self-installing: `Start CourtVision.command` now finds or installs Python, builds
  `.venv`, stores optional API keys in the macOS Keychain and can add a CourtVision app to `~/Applications`
  (`scripts/mac/courtvision.sh`).
- Added `courtvision studio --open` and a Mac launcher: local video upload, actual processed-frame monitor,
  player/ball overlays, pause/speed/restart, saved court setup, recent footage and JSON export.
- Reworked the dashboard around the footage, with a restrained neutral palette, tracking status and a
  court-landmark guide. Existing operator/stats tools remain available.
- Added streamed uploads with decoder validation and existing auth/Origin/Host guards. Separate run folders
  preserve prior outputs; new analyses clear old hub snapshots, events and queued replay messages.
- Integrated round 5, including pump fakes, transition-confirmed mono steals and reviewed truth3d fixes.
  Held-out simulator F1: truth3d **0.991**, mono **0.815**, video **0.792**. NBA broadcast accuracy is reported in the NBA broadcasts entry.
- Re-pinned the rendered CLI fixture to a settled-possession window after the simulator flow changed;
  retained its original event assertion. Added workspace integration tests and refreshed timing documentation.
- See [workspace validation and limitations](docs/STUDIO.md).

---

## [Unreleased] — round 4

### Summary
Round 4 accuracy pass on the event engine (train seeds 1–6, independent review per change), verified once on the
held-out seeds 101–104: `truth3d` 0.941 → 0.970, `mono` 0.708 → 0.742 (both outside the round-start CI), `video`
0.636 → 0.650 (180 s games: noise). Targets (`mono` / `video` ≥ 0.85, `truth3d` ≥ 0.98) not met on held-out games.

### Detailed Changes

#### Event engine

- 3-D shot apex gate, throw-in restarts on the court, ring-contact step slack, reconciler flip hold (`7b1e0c6`)
- Monocular blocks: loose-ball gate, deferred in-flight blocks, image reversal 12 ft/s (`dc30c32`); the shooter left out of the vote during the loose-ball time (`de913e3`, `f47ea37`); image-speed gate, touch margin, head line at the touch, hold-time block path off by default (`ee94d0d`); a far teammate's phantom gain judged when his stint ends (`ffe5129`)
- Monocular steals: long-flight gains are recoveries, firm dribbles corroborate control (`38d1655`)
- Monocular passes: a pass crossing a teammate's box in flight is one pass (`f3e5e81`)
- Shots: a miss the ball went back to the rim after is not flipped (inferred putback); lost passes end at the clock stop (`c916956`)
- Monocular dribbles: bounce height above the dribbler's feet, a fall into each low point; no self-assist (`eae9c50`)
- Monocular rebounds: hidden holds after a neighbour's win, catches dated in the hidden stretch, loose balls re-credited (`3c806cd`)

#### Docs

- Round-4 train and held-out numbers, corrected round-3 video figure, limitations (`cc7e482`)

### Verification Proof
2,075 deterministic pytest tests passing offline at `ffe5129` (6 skipped, 3 strict expected failures).

---

## [v0.3.0] — 2026-10-22

### Summary
Round 3 accuracy verification pass, broadcast perception, learning loop, and presentation showcase transformation.

### Detailed Changes

#### Tooling & Hygiene

- Round-3 train / held-out numbers, learning loop, broadcast, limitations (`5e52881`)
- **General**: Learning loop (ingest / train / promote) and broadcast perception (`8b5dba8`)
- **General**: Monocular retro-block: retried during the flight, committed on a dead ball (`f65e092`)
- **General**: Review fixes: held manual scoreboard, speed-aware narrator clock, wording, stale clock (`78eef81`)
- **General**: Round 3 mono steals: robust reachability veto, contest-aware backdating (`269b806`)
- **General**: Round 3 mono: hidden-ball rebounds, raised-ball holder keep, reachability veto on steals (`04864c2`)
- **General**: Round 3: rebound lift catches, straddled 3-D blocks, tip-ins, OOB rebound touches (`3286983`)
- Accuracy pass rounds 1-2 (verified) + interrupted round 3 (`b212cb0`)
- **General**: Integrate the stack: config, live pipeline, CLI and end-to-end tests (`31e23cd`)
- **General**: Initial commit: research, architecture contract, and CourtVision modules (`50e9721`)

### Verification Proof
2,009 deterministic pytest suites passing offline; validate_readme.py passed in strict mode.
