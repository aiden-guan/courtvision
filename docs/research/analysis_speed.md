# Faster analysis with the same results

Measurements of 2026-10-29 on an M3 Pro (18 GB), Lakers @ Warriors (ESPN, 1080p, 29.97 fps), research use. The
Studio's own analysis ahead of this game was running on the same machine throughout, so every number here was taken
under that load; "before" and "after" ran back to back on the same frames, "before" from a worktree at `78b82e2`.

## 1. Where the time went

**The Studio** (the analysis ahead, `server/preanalysis.py`, which runs Live's pipeline unpaced) took 164 ms a frame
on 20 s of the second quarter (1200-1220 s; 362 of the 600 frames live): 0.20 times real time, so a whole game
would have taken about 6.5 hours with the machine to itself. Its own run on this game was at 0.11 (ETA 10.5 hours).
From cProfile:

| Stage | Cost |
|---|---|
| Court re-acquisition while the game camera is lost (replays, close-ups): a full pan / tilt / zoom search, 2.2 s each, every 30 lost frames and a few frames after each cut and loss | about 70 ms a frame on average |
| Student detector: preprocessing (every frame was normalised three times, once for each stacked sample it is in) | 48 ms a live frame |
| Student detector: the network on MPS, sharing the GPU | about 60 ms a live frame (28 ms alone) |
| Torso colour histograms (`appearance._histogram`: per-pixel norms and `np.add.at`) | 17 ms a live frame |
| Scorebug OCR: a Vision request (another process) at 5 Hz, waited for | 80 ms a read, about 9 ms a frame |
| Court tracking frame to frame (refinement, line evidence) | about 23 ms a frame |

**Offline** (`tools/nba_analyze.py`): `detect` 56 ms a frame in one process, `events` 14.5 ms a frame (half of it the
same torso histograms, a tenth the referee test) and court tracking (`nba_track_court.py`) 30-80 ms a frame, more
where the camera is often lost. Decoding the 1080p H.264 video costs about 1 ms a frame and was never the problem.

## 2. What changed, and how each change was checked

The rule: a change gives the same numbers bit for bit, checked against the code before it. The one that cannot is
the Neural Engine, which rounds its fp16 arithmetic differently from MPS, and that one was checked on the held-out
labels.

| Change | Where | Check | Speed |
|---|---|---|---|
| The student runs on the Neural Engine through Core ML (an fp16 ML program, exported once per weights and cached by their hash; torch elsewhere or when the export fails) | `perception/student.py` | Held-out val (540 frames, 256 balls): ball F1 0.9608 with the same precision and recall at every threshold as MPS and as fp32; person recall 0.8039 (MPS and fp32 0.8037). Outputs within 0.002 of fp32 (MPS fp16: 0.0012). On 160 broadcast frames: ball points 0.002 px apart (median; max 0.009), person boxes IoU 0.9999, 3 of 588 ball candidates and 1 of 1,639 boxes found by one only (at the score threshold) | network 28 ms to 7.7 ms; the GPU is left free |
| Each frame normalised once (the detector keeps the normalised frames, not the raw ones), through per-channel lookup tables (`cv2.LUT`) | `perception/student.py` | Bit-identical to the float maths (test); with torch, 160 of 160 broadcast frames' detections identical | 9.1 ms to 0.5 ms a frame; a live detection 63 ms to 27 ms on torch, 10 ms on Core ML |
| Decoding and preprocessing on a background thread while the network runs | `nba/analyze.py` (`prefetch`, `detect_video`) | With `--backend torch`, 884 of 884 frames' detections bit-identical | see §3 |
| Torso histograms from a table of the bin of every 8-bit BGR colour (16 MB, built in 0.4 s) | `perception/appearance.py` | All 16,777,216 colours bin as the per-pixel maths; 427 of 427 real torso crops identical | 636 µs to 83 µs a crop |
| The referee test's medians and 20th percentile from value counts | `nba/analyze.py` (`is_referee`) | NumPy's float64 results for every sample size tried (test); 579 of 579 real boxes decided alike | 305 µs to 79 µs a box |
| Court search: the chamfer scores on up to 8 threads, the grids that each finer level searches around its candidates in one call per centre | `perception/court_search.py` | A candidate's score does not depend on the batch or thread it is scored in: 0 of 51,600 scores changed with their position; the same candidates and cameras on 4 frames; 900 of 900 tracked frames identical (status, camera, coverage, inliers, error) | a search 1.23 s to 0.37 s; a whole re-acquisition (search and refinements) 1.40 s to 0.54 s |
| Orientation distance maps (each bin's line pixels found once; the 8 distance transforms on threads) and the explained-lines test (the court drawn as polylines, its dense points cached) | `perception/court_search.py` | 0 of 12 maps and 0 of 600 random cameras differ (test) | 26.7 ms to 10.3 ms; 8.1 ms to 2.4 ms |
| The scorebug read on a thread of its own while the same frame's players are detected | `perception/broadcast.py` | The same reads on the same frames, given to the same frame states | 80 ms waited per read to about 20 |

Tried and dropped: detecting a frame while the court tracker reads it, on another thread (a look-ahead dropped at a
cut or a frame that is not live, so the detections are the same). It was no faster, 46.9 against 46.8 ms a frame:
the court tracker's least-squares loops and the detector's decoding both hold the GIL long enough that the two
barely overlap.

## 3. Results

**Studio**, Live's pipeline (measured through the analysis ahead):

| Clip | Before | After | With `backend="torch"` |
|---|---:|---:|---:|
| 20 s of Q2 (1200 s; 40 % of the frames not live) | 164 ms a frame, 0.20 x real time | 47-55 ms a frame, 0.6-0.7 x | 69-76 ms, recording identical to before |
| 60 s of Q3 (2400 s; 1,780 of 1,799 frames live) | 118 ms a frame, 0.28 x | 46 ms a frame, 0.72 x | |

With the torch backend the recording is the one of before in every frame and message: 600 frames' 5,209 boxes
(positions, track labels, teams), 300 court-lens frames, 75 vision, 11 stats, the 4 events, the caption and the
narration. With Core ML 94.9 % of the boxes are on the same pixel and 99.9 % within 1 px; 22 of 5,221 track-frames
are in one run only (detections at the score threshold); the events are the same plays (kind, time, team, player),
with court positions 0.00001 ft apart. On the 60 s of Q3 the 60 events are the same plays (kind, time, team, player),
their confidences and positions within the fourth decimal; 99.8 % of the boxes are within 1 px.

A whole game in the Studio (Live, or the analysis ahead for Replay) then takes about 2 hours instead of 5 to 7.

**Offline**, the same frames before and after:

| Stage | Before | After | Same results |
|---|---:|---:|---|
| `detect` (900 frames, one process) | 56.5 ms a frame | 20.6 ms (Core ML); 35.4 ms (`--backend torch`) | torch: 884 of 884 frames bit-identical |
| `events` (3,000 frames) | 14.5 ms a frame | 5.1 ms | `events.jsonl` byte-identical |
| `nba_track_court.py` (900 frames at 1200 s: 417 lost) | 82.7 ms a frame | 48.0 ms | 900 of 900 rows identical |

A whole game (142,387 frames): `events` about 34 to 12 minutes, `detect` on one process about 2.2 hours to 30-50
minutes (12.1 ms a frame with the machine otherwise quiet but for the Studio). Two detection processes
(`DET_WORKERS=2`) still give about 1.5 times one's frames per second: they share the Neural Engine, and decoding and
preprocessing run on the CPU.

## 4. Settings, and what is left

* `StudentConfig.backend`: "auto" (Core ML on Apple silicon where it loads, else torch), "coreml" or "torch";
  `nba_analyze.py detect --backend torch` reproduces the detections of before bit for bit. The exported model is
  cached in `$COURTVISION_CACHE` or `~/Library/Caches/courtvision` (`student-<hash>-...mlpackage`, about 25 MB).
* `COURTVISION_THREADS`: the court search's threads (by default one per core but one, at most 8). Several
  `nba_track_court.py` parts at once share the cores; 2 each is enough then.

The court tracker is now most of the Studio's time: its refinements are scipy least squares over a few hundred
points, and a re-acquisition still refines the best search candidates one after another. A Gauss-Newton with
analytic derivatives would be faster but would move the camera fits in their last digits, so the camera track would
no longer be the one of before; it was left for a change that is measured on its own.

## 5. The analysis ahead, 2026-10-30

Measured on the same M3 Pro, Lakers @ Warriors 2400-2460 s (1,799 frames), back to back and twice, with another
analysis (CPU and GPU) running meanwhile; research note: [made_shots_at_the_rim.md](made_shots_at_the_rim.md) §5.

| Change | Where | Check | Speed |
|---|---|---|---|
| The court tracked in a worker process that reads the video itself (the same seek and stride) and works up to 48 frames ahead; each answer carries its frame's checksum and cut flag, and any disagreement sends the tracking back in process | `perception/court_worker.py`, `PipelineSettings.court_worker` (on for the analysis ahead) | Identical recordings in process and with the worker: 43 of 43 events, 1,799 of 1,799 frames' boxes and 1,218 of 1,218 messages (timing fields aside), both repetitions; also 600 frames at 1200 s; no fallback | 56.2 / 55.0 ms a frame to **37.7 / 38.3** (1.46 times as fast) |
| A 50-60 fps video read at every 2nd frame (`preanalysis.rate_stride`; Live too) | `server/preanalysis.py`, `server/studio.py` | The analysis is of a 29.97 fps stream, as it is set up for; the detector's input frames are 33 ms apart again (it saw 17 ms at 60 fps) | half the frames |

Where the court is often lost (replays, close-ups, an arena the tracker struggles with), the worker is the slower of
the two and the analysis runs at its pace; where it tracks, the detector side is. Together, a 60 fps game's analysis
ahead does about a third of the work it did.

## 6. Less repeated work, 2026-10-31

Three further changes preserve the arithmetic and the evidence instead of changing detection thresholds,
sampling fewer frames, or loosening convergence:

* **Batched numerical probes.** The chamfer fit still takes the same central differences with the same absolute
  steps and uses the same SciPy solver and stopping rules. Its 6-14 perturbed cameras now project and sample the
  distance map together. The matrix products, bilinear interpolation and subtraction keep their original order;
  tests compare every Jacobian entry and the complete multi-pass fit against scalar differences, including
  off-image probes, with the centre fixed/free and roll fixed/free.
* **Court evidence on demand.** A tracked frame uses the floor around the predicted court, computed from
  the density and exclusion masks. Building and labelling its separate global floor mask was wasted work. It is
  now deferred until a cut check or re-acquisition actually needs it, then reused for the rest of the frame.
  Tests check the local and global masks, line pixels, distances and orientations exactly against eager evidence,
  including blank frames, scorebug exclusions and partly off-image people.
  While the court is lost, evidence is skipped entirely between the existing scheduled re-acquisition attempts;
  the counters still advance on every frame. Tests check the exact regular, after-cut and in-play-loss schedules.
* **Bounded camera factors.** Offline ball linking tries several earlier frames for each candidate. Their camera
  rotations and inverse intrinsics are now reused in a 256-entry LRU cache (three read-only 3x3 arrays per entry).
  The homography still multiplies its factors in the same order. Keys include the camera values and image size,
  so calibration edits cannot reuse an old fit; each returned homography is independent of the cache. Tests
  compare the transforms and complete linked paths at 29.97 and 59.94 fps, with empty detections and camera gaps.

### Measurements

M3 Pro, three alternating trials per implementation; medians below. The court clips use the cut detector and
decode the video in each trial. Raw trial timings are saved locally under `runs/bench/court-processing-*.json`.

| Court clip (300 frames each) | Before, ms/frame | After, ms/frame | Speed |
|---|---:|---:|---:|
| Lakers @ Warriors, 1200 s | 26.54 | 24.74 | 1.07x |
| Lakers @ Warriors, 2400 s | 28.35 | 24.30 | 1.17x |
| Cavaliers @ Warriors, 1200 s | 75.93 | 38.30 | timing unstable |

All **900 frames across both games agree exactly**, including lost/held/re-acquired states and all reported fit
values, in all three repetitions. The Cavaliers trials varied from 40-90 ms/frame before and 37-53 after, so their
apparent 1.98x median ratio is not used as a stable speed estimate. Timing gains depend on machine load and the
clip's mix of tracking, cuts and re-acquisition.

Offline ball linking of `detections-full/0000009-0017795.npz` using `camera_parts/part0.jsonl` for the Lakers game:
15,503 detection frames, **12,480 linked points identical**, 2.735 s to 1.673 s (**1.64x**). This compares the
baseline revision's `link_ball` and `CameraTrack` against the working tree, with the same detection arrays and
camera parameters, alternating order over three trials and clearing the new factor cache before each trial.
Loading files is excluded; no detection model or flight fitting runs in this measurement. Raw timings are saved
locally in `runs/bench/ball-processing-lal-q1.json`.

Validation: the full Python suite passed (2,577 passes, 3 expected failures); the final cache changes passed 320
NBA/integration tests, and the final court changes passed 48 court/broadcast/pipeline tests. The Node replay suite
passed all 4 tests.

### Reproducing the court comparison

The baseline for this change is `41c1dfafb79e13936b08fb709e60549e58fe42df`. The benchmark loads only that
revision's court-search and broadcast-tracker modules; the decoder, court geometry and other dependencies are
shared with the working tree. It warms both implementations, alternates trial order, and fails on any change
to a frame's status, camera parameters, coverage, inliers or RMSE. It also checks repeated trials for determinism.
Video decoding keeps one frame at a time; results are small records for the requested clip.

```bash
.venv/bin/python tools/court_processing_bench.py \
  runs/studio/72e1591f5d4d4fdfb639b20e56396fce/source.video \
  --start 1200 --frames 300 --repeats 3 \
  --reference-ref 41c1dfafb79e13936b08fb709e60549e58fe42df \
  --out runs/bench/court-processing-lal-q2.json
```

This measures decoding, cut detection and court processing. It does not establish a speedup for the complete
Studio analysis, where detection and court tracking overlap in separate processes. Quality agreement applies
to the tested clips and cases; no new held-out accuracy claim is made.

## 7. Live processing, 2026-10-31

The goal here is Live keeping up with a 29.97 fps broadcast while retaining all frames and the existing models.
The settings are automatic in Live (`pipeline.parallel_live: true`); no smaller detector input, lower confidence
threshold, frame-dropping shortcut or reduced OCR rate is involved.

### What changed

* **Parallel court work in Live.** File playback uses the existing checked court worker, which can work ahead of
  the detector. A camera/stream uses `LiveCourtTrackerProcess`: one shared-memory image, one submitted frame and
  one checked answer, with no second camera connection or video lookahead. The compatible student detector can
  infer the same frame while the worker fits its court. A cut waits for the gate before running inference; an
  unexpected non-live frame discards speculative inference and resets its history exactly as the serial path.
  Once the next frame's pixels have actually arrived, the causal worker can fit that one frame while its
  predecessor finishes. A non-mutating cut check predicts its cut flag; the real gate validates it again. The
  next camera is not exposed until consumed. If the frame/index/cut differs, the child restores a bounded snapshot
  of its full prior state, including velocity, loss/cut counters and remembered views, before fitting the actual
  frame. Shared pixels are replaced only after the outstanding answer is drained.
* **Bounded decoded frames.** Live overlaps decoding with current-frame processing. During an OCR wait,
  `StudentDetector.prefetch` can infer up to three already-decoded frames in source order. It checks for OCR
  completion before each inference, and stops at a missing/failed frame. Results are consumed only for the same
  image object/frame/time in FIFO order. Each result retains its prior history: a skip or replacement rolls the
  whole queue back to the last consumed frame, and a cut clears it. Normalisation buffers referenced by queued
  history snapshots cannot be reused. Rectification still uses each frame's eventual camera; events/stats consume
  frames in order. Decoding is serial even though reads are queued; stop/failure cancels all pending reads before
  closing the source generator. `pipeline.live_lookahead_frames` defaults to 3 and is bounded to 1–3.
* **OCR overlap without changing its evidence.** On the causal path, the existing recognition function can run
  alongside court fitting. Its fields are applied only after the live gate accepts that exact frame. Discarded
  recognition cannot learn a logo, change seen-score state or move the read schedule. Readings remain attached to
  their original timestamps.
* **Predictable model and memory work.** The rim CNN warms its normal evaluation shapes before playback; it
  observes no game evidence and its predictions are discarded. This moves first-use GPU compilation out of the
  shot path (one cold call paused a measured run for 2.46 s). The student reuses a bounded pool of normalised
  frames and its stacked input; single-sample Core ML results avoid a redundant concatenation. At the normal
  input size this avoids repeatedly allocating about 25 MB of normalisation/stack buffers per analysed frame.
  LUTs also write directly into the destination planes, avoiding another 6 MB of temporary output/copy traffic.
* **Less numerical-differencing overhead.** The line-match solver's 2-point probes are projected together, with
  SciPy's float64 step sizes and the actual representable `dx`. Tests compare every Jacobian entry and the final
  fit directly with the installed SciPy implementation, for fixed/free centre and roll. The robust loss, scales,
  tolerances and observed court points are unchanged.

`pipeline.opencv_threads: 1` avoids nested native parallel regions alongside the pipeline workers; `null` preserves
the host setting. OpenCV's GCD build reports the CPU count rather than the requested pool size, so the benchmark
records both the requested and reported values ([OpenCV implementation](https://github.com/opencv/opencv/blob/4.x/modules/core/src/parallel.cpp)).
The dedicated live threads request user-initiated QoS on macOS when their inherited priority is lower, without
changing other processes' priority.

The Studio's speed label now uses a two-second rolling count of completed frames, including waits between
frames, rather than the duration of one possibly cached inference. Its denominator is the analysed source rate,
so a 59.94 fps recording analysed at 29.97 fps is compared against 29.97 rather than incorrectly showing 0.5x.

### Results and their limits

The harness exercises `StudioGame`, its real student model, court registration, OCR, rim judge, events, stats and
Hub publication. It starts from the saved camera at the clip's time, so one-time venue registration is excluded.
Model loading is outside the processing timer; rim warm-up is reported separately. It does not measure browser
rendering or JPEG requests from a viewer. All runs below use the Lakers @ Warriors 1080p footage at 2400 s.

| Run | Frames | Whole-run fps | Steady fps after 3 source seconds | Dropped / errors |
|---|---:|---:|---:|---:|
| Live path, unpaced, before buffer reuse | 900 | 37.76 | 44.06 | 0 / 0 |
| One-frame buffer reuse, unpaced, frame audit on | 900 | 33.83 | 37.30 | 0 / 0 |
| One-frame buffer reuse, paced, 60 s | 1,799 | 23.22 | 23.97 | 0 / 0 |
| Later paced run, after host memory pressure eased | 1,799 | 28.93 | 29.63 | 0 / 0 |
| Repeat of that paced run | 1,799 | 22.28 | 23.45 | 0 / 0 |
| Three-frame buffer, unpaced, frame audit on | 900 | 35.94 | 39.20 | 0 / 0 |
| Causal input, three-frame detector buffer, paced, 60 s, before next-court overlap | 1,799 | 21.98 | 22.69 | 0 / 0 |
| Causal input with next-court overlap, unpaced, frame audit on | 600 | 31.88 | 36.63 | 0 / 0 |
| Causal input with next-court overlap, paced, 60 s | 1,799 | 20.64 | 20.95 | 0 / 0 |

The three 900-frame comparisons agree **exactly** with synchronous perception: all player/ball states, candidate
balls, scorebug readings, camera/rim data, rim-crop checksums and final events. Causal processing was also compared
on 600 frames, including the three-frame buffer: its states and events agree exactly with the serial reference.
The latest next-court overlap also matches that complete 600-frame reference exactly, including rim-crop checksums.

The first one-minute paced run accumulated 17.26 s of lag, so **consistent sustained real time is not yet verified**. At inspection
the M3 Pro had 18.76 GB of swap in use and four iPad simulators booted; earlier, a simulator media-analysis process
used about four CPU cores. The timing runs were sequential, with other apps left running. These are observed
rates under variable load, not a controlled speedup ratio or a guarantee for every clip. The pipeline can exceed
30 fps on the measured short segment, but the long-run resource constraint still needs a controlled test.

After two simulators were no longer booted, one paced run completed at **30.005 fps from its first output to its
last**, following 2.12 s of startup. Its final lag was 2.06 s, so it added no net lag after startup. The repeat,
however, processed 23.14 fps over the corresponding interval and added 17.71 s of lag. This establishes real-time
throughput in one observed run, but not consistent behaviour under the current host load. The benchmark records
both the fixed-three-source-second warm-up rate and the full first-output-to-last-output rate, plus lag growth,
to keep startup costs and later stalls visible rather than hiding them in a single fps number.

With the simulators left running as requested, the final causal one-minute sample processed 21.28 fps from its
first output to its last and added 24.51 s of lag. All three simulators were still booted; swap use was 17,276 MB.
The short causal audit reached 35.80 fps over the corresponding post-first-output interval. These samples do not
establish sustained 30 fps for incoming camera frames under this load. No physical camera, browser rendering or
cloud narration was included in the measurement.

A later 900-frame segment, starting at 2430 s, tested two court-search threads against the normal eight. Both used
the same candidate grid and produced identical complete states/events. Two threads took 34.40 s (27.60 fps after
first output); eight took 31.32 s (30.79 fps after first output). Fewer threads did not help in this sample, so the
production search-thread default is unchanged. `--court-threads` is available in the benchmark for investigation;
these sequential timings under variable host load are not a controlled speedup estimate.

### Reproduce

```bash
# Actual Live executor and bounded decoding, running unpaced to measure available throughput:
.venv/bin/python tools/live_processing_bench.py --worker --live-path --seconds 30 --audit \
  --out runs/bench/live-audit.json

# Normal 1x video cadence, preserving every frame:
.venv/bin/python tools/live_processing_bench.py --worker --paced --seconds 60 \
  --out runs/bench/live-paced.json

# Camera/stream worker with prerecorded pixels arriving at 1x (no physical camera claim):
.venv/bin/python tools/live_processing_bench.py --worker --causal --paced --seconds 60 \
  --out runs/bench/live-causal-paced.json
```

The harness writes frame-completion times, last-frame lag, stage summaries, worker fallback, error/drop counts,
and optional full state audits. Any processing error or dropped frame fails the sample. `--code-root` can select
a saved source tree for comparison. Set `pipeline.parallel_live: false` to use the original sequential Live
path (and leave `pipeline.court_worker: false`); this is useful for comparisons.

Final validation: **479 Python tests passed** across integration, pipeline, perception, Studio, NBA analysis and
the live shot judge; the replay client's **4 Node tests passed**, and the Studio JavaScript syntax check passed.
An earlier full-suite run passed 2,613 tests with 3 existing expected failures, before the final three-frame and
next-court extensions. That full suite was not repeated after those extensions. The new tests cover
exact SciPy Jacobians/fits, worker parity and shared-memory cleanup, spawn/crash fallback, bounded source
lookahead and stop, speculative-inference errors, OCR state/schedules, model warm-up without evidence, and exact
temporal inputs across cache reuse, cuts, skipped/replaced frames and failed prefetches at both history spacings.
They include FIFO consumption, rollback after partial consumption, bounded allocation, source failure and
cancelled reads, and exact court fits after speculative replacements, skipped frames and changed cut flags.
