# CourtVision

Real-time basketball computer vision → live stats → a TypeSafe **Jev** decision layer → grounded
live narration with **Claude**, plus coach insights and an operator dashboard.

```
video / camera / simulator ─▶ perception ─▶ event engine ─▶ stats ─▶ narration (templates + Claude) ─▶ TTS / captions
   (detect, track, team,        (possession, shots,   (box score, tracking,      ▲
    jersey, homography, ball)    make/miss, rebounds,  xFG, EPV-lite decisions,   │ Jev: play type, moment stakes,
                                  assists, TOs,         insights)                 │ coach-card ranking, Claude line guard
                                  scoreboard reconcile)
```

## Status

The whole chain runs end to end, from the `courtvision` command line and from Python. A simulated
game, a video file or a camera goes through perception, the event engine, stats, insights,
narration and the live dashboard. Every module has its own test suite, written against
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and then adversarially reviewed. On top of those,
the integration and CLI tests (`tests/test_integration_*.py`) drive the real stack. That includes
a serve smoke test that runs the dashboard app and a live game on one event loop. The suite has
1 890+ tests, all deterministic and network-free.

**All accuracy figures come from the seeded simulator.** No real game footage has been evaluated
yet. On real video the system runs, but its accuracy is unmeasured (see [Limitations](#limitations)).
Claude and Jev have not been called with live keys in this repo. Without keys, narration is
template-only and Jev uses its deterministic fallbacks, and nothing crashes.

```python
from courtvision.config import load_config
from courtvision.pipeline import run_offline

cfg = load_config(None, ["sim.seconds=120", "source.sim_mode=mono", "narration.tts=none"])
result = run_offline(cfg, out_dir="runs/demo")  # events.jsonl, transcript.jsonl, stats.json, report.md, ...
print(result.summary())
```

| Package | What it does |
|---|---|
| `courtvision/geometry` | Court specs for NBA / FIBA / NCAA / NFHS, rule sets (`RuleSpec` + YAML), homography (DLT + RANSAC), pinhole camera |
| `courtvision/perception` | Detectors (synthetic colour-blob, RF-DETR PyTorch/CoreML; Ultralytics R&D-only), ByteTrack, ball Kalman, team clustering, jersey voting, identity, scoreboard reader |
| `courtvision/events` | Deterministic event engine (3-D and monocular modes), cue-model confidences, scoreboard reconciliation, dead-ball detection |
| `courtvision/stats` | Box score, tracking stats, advanced metrics, operator lineups, shot-quality model, EPV-lite decision grades, coach insights |
| `courtvision/decision` | TypeSafe Jev client (per-bundle deadlines, breakers, stub/replay/off modes) and the advisor façade with deterministic fallbacks |
| `courtvision/narration` | Fact packets with pseudonymised players, templates, grounding validator, director, Claude candidates, TTS, captions |
| `courtvision/sim` | Seeded game simulator with ground truth, noise model, synthetic renderer, simulated operator |
| `courtvision/eval` | Event, box-score and tracking metrics |
| `courtvision/server` | FastAPI + WebSocket hub and the live dashboard |
| `courtvision/training` | Play-by-play alignment, shot-model and cue-model fitting, YOLO export with a rights gate |
| `courtvision/config.py`, `pipeline.py`, `cli.py`, `roster.py` | One `AppConfig` (defaults in `courtvision/config/default.yaml`), frame sources + `LiveGame` + offline runs, the command line, roster files |

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# 1. Simulate a 2-minute game, run the whole stack offline and score it against the simulator's truth
.venv/bin/courtvision simulate --seconds 120 --mode truth3d          # or --mode mono / --mode video

# 2. Live dashboard: a simulated game at 2x, then open http://127.0.0.1:8765/
.venv/bin/courtvision serve --speed 2 --tts none --open
```

`python -m courtvision ...` works the same as `.venv/bin/courtvision ...`. The dashboard shows a
top-down court, the score and clock, the narration transcript, the event feed with corrections,
box scores, the shot chart, coach insights and a calibration tab. The game runs as a task on the
server's own event loop, and Ctrl-C stops both cleanly.

**A video file.** `render` writes a synthetic broadcast video. Next to it go its calibration,
roster, scoreboard ROI and truth events, so the real-video path can be exercised and scored:

```bash
.venv/bin/courtvision render --seconds 60 --out runs/demo.mkv      # .mkv / .avi: lossless FFV1; .mp4: H.264
.venv/bin/courtvision run --video runs/demo.mkv --calibration runs/demo.calibration.json \
    --roster runs/demo.roster.yaml --scoreboard-roi runs/demo.scoreboard_roi.json --detector colorblob --out runs/demo-run
.venv/bin/courtvision eval runs/demo-run --truth runs/demo.truth_events.jsonl
```

**Your own footage** needs a fixed camera, a calibration and a detector:

```bash
# Click >= 4 floor landmarks in a frame and write them down (format: docs/examples/calibration_points.json)
.venv/bin/courtvision calibrate my_points.json --out venue.json        # prints the fit RMSE in court feet
.venv/bin/pip install -e ".[cv]"                                       # RF-DETR (Apache-2.0)
.venv/bin/courtvision run --video game.mp4 --calibration venue.json --roster docs/examples/roster.yaml \
    --detector rfdetr:weights/basketball.pth                           # or coreml:<model.mlpackage>
.venv/bin/courtvision serve --source camera --camera 0 --calibration venue.json --roster roster.yaml
```

The roster format is [`docs/examples/roster.yaml`](docs/examples/roster.yaml). Quote jersey numbers:
`"0"` and `"00"` are different numbers. You can also calibrate in the dashboard's calibration tab
(`POST /api/calibration`): `serve --source video` without `--calibration` holds the file at its first
frame (status `needs_calibration`) until you do. A camera keeps running while it waits.

Nothing is downloaded silently. RF-DETR needs a local weights file; without one, `run` and `serve`
stop and explain. `--allow-download` lets rfdetr fetch its pretrained COCO checkpoint (~370 MB),
which detects people and balls but not rims or referees, so use a fine-tuned checkpoint for games.
`--detector rnd-yolo:<weights>` uses Ultralytics, which is AGPL-3.0. It
is for R&D only, works only from a source checkout (through `tools/rnd_detectors.py`; shipped code
never imports it) and prints a warning (RESEARCH.md D9).

**Keys** are optional and detected by name only, never printed or logged:

| Variable | Enables | Without it |
|---|---|---|
| `ANTHROPIC_API_KEY` (or an `ant auth login` profile) | Claude colour lines in real-time `serve`, at timeouts and period breaks; the prompt cache uses a 1 h TTL, pre-warmed at start and every 55 min | template-only narration |
| `TYPESAFE_API_KEY` | live Jev (play type, stakes, coach-card ranking, the line guard) | deterministic fallbacks. Claude lines also stay off, because D4 speaks a Claude line only after a passing Jev guard |
| `COURTVISION_TOKEN` | bearer token for the dashboard API and WebSocket | `serve` refuses a non-loopback `--host` (the API carries rosters) unless `--insecure-no-auth` |

Offline commands (`simulate`, `run`) keep Jev off unless you pass `--jev live|stub`, so their
results are reproducible. Claude stays off in offline runs unless `--set llm.offline=true`.

### Commands

| Command | What it does |
|---|---|
| `simulate` | Simulator run, evaluated against truth: `--seconds --seed --league --mode truth3d\|mono\|video --out --jev --json` |
| `serve` | Live dashboard on uvicorn: `--source sim\|video\|camera --mode --video --camera --speed --host 127.0.0.1 --port 8765 --jev live\|stub\|off --no-llm --tts none\|say\|avspeech --auth-token --open` |
| `run` | Offline video file: `--video --calibration --detector colorblob\|rfdetr:<weights or size>\|coreml:<mlpackage>\|rnd-yolo:<weights> --roster --scoreboard-roi --start --duration --out` |
| `calibrate` | Calibration JSON from a points file (landmark names or court points), or `--from-sim`; prints the RMSE |
| `render` | Synthetic demo video (`.mp4` H.264, `.avi`/`.mkv` lossless) plus calibration / roster / ROI / truth sidecars |
| `eval` | `events.jsonl` (or a run directory) against `truth_events.jsonl`: per-kind P/R/F1, attribution, shots |
| `config` | The effective YAML config (`--section NAME`, `--default` for the commented defaults) |
| `latency` | `tools/latency_harness.py`: real, billed Claude / Jev calls that set every deadline from the measured p95. Refuses to run without the needed keys |

Every command that builds a config takes `--config FILE` and repeatable `--set section.key=value`
(keys as in `courtvision/config/default.yaml`). Explicit flags win over `--set`, which wins over the file.

## How accuracy is measured

Every simulated game has exact ground truth: positions, events and an independently tallied box
score. The simulator and the engine follow the same event definitions
(ARCHITECTURE "Shared event semantics"). Each run is scored by `courtvision/eval`:

- **Events.** Predictions are matched to truth one-to-one, per kind, greedily by time within ±1.0 s
  (±0.25 s for dribbles). Each kind gets precision, recall, F1, actor accuracy and an
  "attributed" F1 that also requires the right player. The **headline** is the macro-F1 over the
  core kinds present (shots, makes, misses, rebounds, assists, steals, turnovers, blocks, passes).
  Dribbles are reported apart. A shot-level join adds make/miss accuracy, 2-vs-3 accuracy and
  assist attribution.
- **Box score.** Per-stat MAE over all truth players, the share of exactly right stat lines, and
  the team-points error. It is computed only for whole-game runs.
- **Tracking** (video modes). MOTA, IDF1, position RMSE in feet, ID switches and identity accuracy.
- **Latency.** Each stage has p50 and p95: perception per frame, the frame step, event time →
  engine emission, emission → utterance, and event time → utterance.

The modes say what a number tests. `truth3d` feeds perfect 3-D observations, so it tests the event
logic. `mono` adds the noise model: position and box noise, ball dropouts and false positives, and
a monocular ball with no height. `video` renders frames and runs the real perception stack
(colour-blob detector, ByteTrack, template jersey reader, scoreboard OCR, homography). Offline runs
use source-time clocks, so they are deterministic: the same seed gives the same report.

**The benchmark.** `tools/accuracy_bench.py` is the one accuracy number every change reports. For
each (mode, seed) it runs exactly what `courtvision simulate` runs, then pools over seeds: per-kind
F1 from summed TP / predictions / truths, the headline macro-F1 (eval's definition) over those
pooled F1s, FG make/miss and 2-vs-3 accuracy, rebound and assist attribution, box-score MAE and
team-points error (`courtvision/eval/bench.py` defines each). Runs go to a process pool. Tune only on
the **train** seeds 1-6. The **held-out** seeds 101-104 are for the verifier; they produce aggregate
numbers only (`--keep-runs` is refused). Video mode renders every frame, so its games default to
180 s (`--video-seconds`); box scores of 180 s and 300 s games are not comparable across modes.

```bash
.venv/bin/python tools/accuracy_bench.py --modes truth3d,mono,video --split train --seconds 300 \
    --video-seconds 180 --video-seeds 1,2,3 --workers 8 --json runs/bench/train.json
```

Train baseline (2026-10-20, the command above, 2 min 47 s wall on 8 workers):

| Mode | Seeds | s/game | Headline F1 (pooled) | Seed mean ± sd | FG make/miss | 2 vs 3 | Rebound F1 / rebounder | Assist F1 / label | Box MAE | Team pts error |
|---|---|---|---|---|---|---|---|---|---|---|
| truth3d | 1-6 | 300 | **0.934** | 0.935 ± 0.063 | 0.967 (n=92) | 0.989 (n=95) | 0.964 / 0.963 | 1.000 / 0.946 | 0.022 | 0 |
| mono | 1-6 | 300 | **0.600** | 0.644 ± 0.060 | 0.967 (n=92) | 0.947 (n=94) | 0.807 / 0.543 | 0.931 / 0.730 | 0.119 | 0 |
| video | 1-3 | 180 | **0.707** | 0.761 ± 0.098 | 0.933 (n=30) | 0.935 (n=31) | 0.821 / 0.500 | 0.941 / 0.909 | 0.080 | 0 |

The pooled headline counts a kind once over all seeds, so a kind with only false positives (mono
blocks, video steals) costs a full 1/9 of it. The seed mean counts that kind only in the seeds where
it occurs. Since the round-1 simulator pass the report also gives the macro over the core kinds that
have truth, the pooled micro-F1, false positives per stream minute of core kinds without truth and 95%
seed-bootstrap intervals; with 0-15 truth events per rare kind, only pooled multi-seed numbers of
>= 180-300 s games are comparable.

**Simulator realism pass (round 1).** The simulator and the observation models were made more like real
basketball and a real single camera (see ARCHITECTURE "sim/" for the list): bodies no longer
interpenetrate, boxes include jumps and raised arms, passes avoid the rim, steals are followed by a
transition push, blocks / fouls / free throws / out-of-bounds follow the rules note's priors, rebounds
are contested, PERIOD_END sits at the horn, substitutes walk on; mono velocities come from the noisy
positions, ball misses are bursty and occlusion-driven, false positives persist, identities swap in
occlusions, and mono / video read an operator-run scoreboard at 5 Hz. The eval no longer scores
unresolvable events at a cut stream edge and matches operator substitutions with the operator's lag.
Re-measured train numbers (`--modes truth3d,mono`, 300 s, seeds 1-6; `runs/bench/r1_sim_after.md`):

| Mode | Headline F1 (pooled) [95% CI] | Micro F1 | FG make/miss | Rebound F1 / rebounder | Assist F1 / label | Box MAE |
|---|---|---|---|---|---|---|
| truth3d | **0.953** [0.869, 0.997] | 0.988 | 1.000 (n=83) | 1.000 / 0.945 | 0.976 / 1.000 | 0.015 |
| mono | **0.553** [0.536, 0.573] | 0.814 | 0.987 (n=78) | 0.642 / 0.600 | 0.800 / 0.655 | 0.113 |

The end-to-end table below predates that pass.

## Current end-to-end numbers

Measured on 2026-10-20 on the development Mac with `courtvision simulate` / `render` + `run` +
`eval`. All runs use seed 7 and no keys, so narration is template-only and Jev is off.

| Run | Headline F1 | Notes |
|---|---|---|
| `simulate --mode truth3d`, 120 s | **0.967** | Every kind 1.0 except shot_missed 0.80 (one extra miss). Box score exact: per-stat MAE 0, every line exact, team points error 0 |
| `simulate --mode truth3d`, 600 s | **0.988** | Pass 0.974, shot attempt 0.987, miss 0.971, rebound 0.970, the rest 1.0. Box MAE 0.019, player points MAE 0 |
| `simulate --mode mono`, 120 s | **0.687** | Possession 0.77, pass 0.89, shots 1.0, assist 0.80, plus spurious steal / turnover. Box MAE 0.075, points MAE 0.42 |
| `simulate --mode mono`, 600 s | **0.547** | Shot attempt 0.85, make 0.89, miss 0.86, rebound 0.69, steal / turnover / block 0.0. Box MAE 0.214, points MAE 0.58 |
| `simulate --mode video`, 60 s | **0.972** | Shots 1.0, pass 0.89, possession 0.50. MOTA 0.981, IDF1 0.977, RMSE 0.75 ft. Box exact |
| `simulate --mode video`, 300 s, seeds 7 / 11 | **0.630** / **0.648** | Longer games expose the same monocular gaps as `mono`: steal 0.0 / 0.31, turnover 0.0 / 0.27, rebound 0.71 / 0.59. MOTA 0.956 / 0.942, box MAE 0.12 / 0.16 |
| `render` (lossless .avi) → `run`, 60 s | **0.981** | Same perception from a file, with calibration / ROI / roster files and no simulated operator |
| `render` (H.264 .mp4) → `run`, 60 s | 0.633 | Every shot right, but compression adds false passes, a steal and turnovers |

Team points error is 0 in every simulator mode, because team points follow the confirmed
scoreboard. Substitution F1 is low (0.14 over 600 s) because the simulated operator enters subs
2 s late, as a real one would. Fouls appear only through operator entries. Mono scores vary a lot
by seed: earlier runs of seeds 11 and 23 at 120 s scored 0.62 and 0.30.

**Latency.**

| Measurement | Value |
|---|---|
| Event engine | 0.06 ms/frame mean (p95 0.11) |
| Whole frame step (truth3d) | 0.12 ms mean, p95 0.3 ms |
| Offline throughput | 120 s in about 0.5 s, about 230x real time |
| Event time → engine emission | p50 67 ms, p95 1.1 s. Shots resolve when the ball leaves the rim zone |
| Engine emission → utterance (offline, source time) | p50 283 ms, p95 1.6 s |
| Real-time `serve`, truth3d at 1x, 60 s | 0 dropped frames. Emission → utterance p50 245 ms, p95 509 ms (wall). Event → utterance p50 0.22 s, p95 1.39 s |
| Real-time `serve`, video mode at 1x, 40 s | Perception in the worker thread p50 24.5 ms, p95 46.7 ms per 1280×720 frame. Kept real time (factor 1.00, 0 dropped) |
| Offline video runs | about 1.4x real time |

## Limitations

- **Simulator only.** No real footage has been scored. The colour-blob detector and the template
  jersey reader work only on the synthetic renderer's palette and digits.
- **Real video is largely unbuilt beyond the plumbing.**
  - RF-DETR runs with COCO weights unless you pass fine-tuned ones. COCO has no referee or rim
    class, so referees count as players and rims must come from the calibration file.
  - No jersey reader for real footage ships. Players stay provisional (team + track), and
    narration names teams, not players.
  - Team assignment is colour k-means.
  - One fixed camera only. A panning camera breaks the floor homography.
- **Compression hurts.** Lossy video costs accuracy: on the same 60 s render, H.264 scores 0.63
  against 0.98 lossless, and OpenCV's mp4v loses the free throws entirely.
- **Monocular gaps.** With the ball's height unknown, steals, turnovers and blocks are not detected
  (F1 0.0 over 600 s). Possession sits near 0.75 and rebound attribution near 0.69.
- **The CV never infers some things.** Fouls, timeouts and substitutions are never inferred: they
  come from the operator (dashboard) or a feed. Minutes, +/- and lineups exist only after operator
  lineup entries. Without them, a whistle stoppage can turn into an inferred out-of-bounds turnover.
- **Uncalibrated confidences.** The cue models (shot, make, three, ...) are hand-set and
  uncalibrated outside the simulator. On real footage, narration uses hedge-only lines (no outcome,
  no point value) for any result the scoreboard or the operator has not confirmed, until the models
  are fitted on labelled games.
- **Cloud paths are unmeasured.** Claude and Jev have not been called with real keys. Every cloud
  deadline is an initial value until `courtvision latency` has been run on the venue network
  (RESEARCH.md §7A.13). Claude only voices dead-ball colour, and only with both keys (D4).
- **Shot-quality priors are NBA.** Other leagues show raw eFG% instead of expected eFG%.
- **No audio.** Audio capture is off by design. AVSpeech TTS is macOS-only; `--tts none` works everywhere.

## Docs

- [`docs/RESEARCH.md`](docs/RESEARCH.md) — the research report (state of the art, where Jev fits, accuracy strategy, legal constraints, roadmap). §1A lists the design decisions that bind the code.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — module contracts and shared event semantics.
- [`docs/TRAINING.md`](docs/TRAINING.md) — data flywheel and fine-tuning.
- [`docs/research/`](docs/research/) — the fact-checked research notes behind the report.
- [`docs/examples/`](docs/examples/) — a roster file and a calibration points file.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

Optional extras: `.[cv]` (RF-DETR, supervision, trackers), `.[team]` (SigLIP team embeddings),
`.[tts]` (AVSpeech), `.[rnd-agpl]` (Ultralytics, R&D only — AGPL-3.0).

Claude uses `ANTHROPIC_API_KEY` (or an `ant auth login` profile); Jev uses `TYPESAFE_API_KEY`.
Without keys, narration runs template-only and Jev falls back to deterministic rules.
