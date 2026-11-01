# Public demo on Vercel

The root `Dockerfile` runs CourtVision Studio on Vercel's Container Runtime as a public,
showcase with private, on-device video detection (`courtvision studio --public-demo`).

## What visitors can do

- Start and watch the 3-minute simulated game ("Explore the demo"). Everyone shares one
  workspace: starting the demo while one is running joins it instead of restarting it.
- Read everything the dashboard shows (box score, shot chart, play-by-play, narration).
- Pause/resume their own demo playback, exit to the video picker, and export the visible results.
  Demo frames and messages are buffered in each browser; pausing never pauses another visitor.
- Choose a local video **up to 30 seconds**, play, pause, seek, change speed, and inspect
  people/ball detections from the **existing NBA-trained BallPersonNet**. Video metadata is
  checked before loading the model; longer, unreadable or unsupported clips are rejected.
  The file is opened through a blob URL and analysed in order before playback. A browser
  worker runs the original Python vision pipeline and ONNX Runtime Web. Neither footage
  nor detections are uploaded. MP4 (H.264) and WebM are recommended.
- Export recorded browser detections as JSON, including the source-checkpoint and Python
  bundle SHA-256 hashes, backend, source-video pixel boxes, timestamps and summary counts.

Server writes remain refused with 403: uploads, server detector setup, shared game controls,
manual events, calibration, scoreboard setup, the video library, replays, the model lab and
Ask. Browser inference uses no API keys. Scoring, roster identities and game statistics
remain desktop features, available through `courtvision studio --open`.

## The original NBA model and Python vision pipeline in the browser

`static/nba-student.onnx` is an fp32 ONNX export of `runs/studio/models/nba-student.pt`.
There is no retraining, replacement detector or quantization. ONNX Runtime Web runs the
neural forward pass on WebGPU, with WASM as a fallback. It uses the pinned 1.31.1 JSEP
bundle: the native WebGPU bundle gave incorrect outputs in matched-frame testing.
Before accepting a backend, the worker runs a deterministic input against 640 CPU reference
probes in `nba-validation.json`. A GPU failure falls back to CPU; a CPU mismatch stops
analysis. This checks numerical behavior as well as downloaded weight hashes.

Pyodide runs the **original Python modules** for OpenCV area resizing and normalization,
three-frame history, decoding, automatic court registration, box correction, ByteTrack,
team clustering and single-ball tracking. `courtvision/browser/runtime.py` adapts the
asynchronous forward pass; it does not replace those algorithms with JavaScript versions.
`python -m courtvision.browser.build` packages the current sources into `browser-python.zip`
and records every source hash in `browser-python.json`. The Docker build regenerates it.
The worker verifies both the ONNX and Python bundle hashes before running them.

The browser checks the 30-second duration limit before loading the model. It advances a
paused decoder in source order at approximately 30 analysis frames per second, retains
results for the clip, then replays them against their recorded timestamps. During analysis,
the displayed frame is the exact pixel snapshot sent to inference. Pause suspends further
frames; Close terminates an active worker and releases the video. Seeking during replay
reads recorded results instead of starting a new detector history. Footage never leaves the
device. Camera registration and analysis can take several minutes, depending on the device.
A setup panel on the video shows the current stage (model, vision tools, court), measured
progress and elapsed time. Court progress advances after each checked view and the final
map refinement; stages without a known total show an indeterminate bar. It disappears when
the first analysed frame arrives, and the video timeline then shows analysis progress.
If no camera can be registered, the same original image-space trackers provide a preview;
the interface reports that the court is not mapped. Track IDs and team groups are provisional.

`static/nba-student.json` records checkpoint/export SHA-256 hashes and raw neural output
checks. Reproduce the export with torch, torchvision, onnx and onnxruntime installed:

```sh
python tools/export_student_browser.py --video /path/to/clean-nba-clip.mp4
python -m courtvision.browser.build
```

ONNX Runtime Web 1.31.1 is vendored with its MIT licence. The Python runtime and its numpy,
scipy and OpenCV packages load from the official pinned Pyodide 314.0.7 distribution on
jsDelivr. The first visit downloads these assets as well as the 47 MB NBA model; later
visits can use the browser cache. The CDN receives runtime downloads, never video frames.
CSP permits this pinned runtime path, same-origin workers, blob video and WASM compilation
(`wasm-unsafe-eval`); it does not permit JavaScript `unsafe-eval` or inline scripts.

The browser does not run the desktop scorebug OCR, shot models, event/statistics engine,
roster identification or narration. Those remain desktop features; no synthetic game
statistics are presented as video analysis.

### Browser verification

On the clean 1080p Lakers–Warriors source, a matched-pixel 12-frame comparison against
the original CPU desktop pipeline preserved every detection count, track ID and team group.
Maximum confidence difference was 0.00000066; maximum box-coordinate difference was
1.59 source pixels (median 0.12 pixels). The neural input tensor was byte-identical and
WebGPU raw outputs differed by at most 0.000011 from CPU ONNX outputs. The replaced native
WebGPU bundle differed by up to 9.97 on the same ball logits despite identical input.

An 8.008-second clean NBA clip completed all 240 ordered analysis frames in the browser,
with a single ball identity, pause/resume and timestamped replay. A readable 31-second
clip was rejected before model loading. These checks verify the browser implementation;
they do not establish detection accuracy on arbitrary broadcasts.

## No API keys: the local engine only

The demo runs on the local engine alone: the simulated game, the event engine and template
narration, with no Claude and no Jev. `--public-demo` removes `ANTHROPIC_API_KEY`,
`ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_IDENTITY_TOKEN(_FILE)` and `TYPESAFE_API_KEY` from the
process at startup; every run forces `narration.llm_enabled=false` and `jev.mode=off` and is
given no keys, not even a Jev key saved in a `.env`; and Ask, the other feature that calls
Claude, is refused. Nothing is billed even if keys are still set in the Vercel project.
Deleting them from the project is still the tidiest option.

## Host check

The app's Host check accepts `*.vercel.app` (set in the Dockerfile `CMD`). Add another
`--set server.allowed_hosts=...` entry if you attach a custom domain.

## Running it privately with Claude instead

Drop `--public-demo` from the `CMD`, pass `--auth-token "$COURTVISION_TOKEN"`, set that
variable plus the API keys as Sensitive env vars, and turn Vercel Authentication back on
(Settings → Deployment Protection). Open `https://<deployment>/#token=<token>` once per tab.

## Platform limits (Hobby plan)

- Requests, including the WebSocket, end after 300 s; the 180 s demo fits and the dashboard
  reconnects after a cut.
- No persistent disk: state lives in `/tmp` and resets when the instance scales down
  (5 minutes without traffic).
- Demo runs count against the plan's Active CPU quota.
