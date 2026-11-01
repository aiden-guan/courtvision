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
  The file is opened through a blob URL. Pixels are processed in a browser worker using
  ONNX Runtime Web, with WebGPU when available and a WebAssembly CPU fallback. Neither
  footage nor detections are uploaded. MP4 (H.264) and WebM are recommended.
- Export sampled browser detections as JSON, including the source-checkpoint SHA-256,
  backend, source-video pixel boxes, and summary counts. Detection runs while viewing;
  seeking skips intervening frames. Exports retain the latest 3,600 sampled frames.

Server writes remain refused with 403: uploads, server detector setup, shared game controls,
manual events, calibration, scoreboard setup, the video library, replays, the model lab and
Ask. Browser inference uses no API keys. Teams, identities, scoring and court mapping still
use the full desktop pipeline, available through `courtvision studio --open`.

## The same NBA-trained model, exported for the browser

`static/nba-student.onnx` is an fp32 ONNX export of `runs/studio/models/nba-student.pt`.
There is no retraining, replacement detector or quantization. The model keeps its 960×544
input, three RGB frames approximately 33 ms apart, ImageNet normalization, five output
heads and desktop thresholds (ball .08, person .3). The browser retains decoded frame history
and runs inference in a worker so controls stay responsive. Seeking resets temporal history,
as it does in the desktop detector.

`static/nba-student.json` records checkpoint/export SHA-256 hashes and numerical parity checks
on the Lakers–Warriors NBA clip. To reproduce the export (requires torch, torchvision, onnx
and onnxruntime):

```sh
python tools/export_student_browser.py --video runs/nba/2024-12-25-LAL-GSW/videos/full_981-1001.mp4
```

ONNX Runtime Web 1.31.1 is vendored from its npm package, with its MIT licence. The worker
and model are served from this site; no third-party model download is used. First use loads
about 75 MB of model/runtime assets, which the browser can cache. The worker verifies the
model's SHA-256 before inference. CSP allows same-origin workers, blob video, and WASM
compilation (`wasm-unsafe-eval`); it does not allow JavaScript `unsafe-eval` or inline scripts.

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
