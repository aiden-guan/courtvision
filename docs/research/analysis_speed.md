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
