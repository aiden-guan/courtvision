# Public demo on Vercel

The root `Dockerfile` runs CourtVision Studio on Vercel's Container Runtime as a public,
watch-only showcase (`courtvision studio --public-demo`).

## What visitors can do

- Start and watch the 3-minute simulated game ("Explore the demo"). Everyone shares one
  workspace: starting the demo while one is running joins it instead of restarting it.
- Read everything the dashboard shows (box score, shot chart, play-by-play, narration).

Everything else is refused with 403: uploads, detector setup, game controls, manual events,
calibration and scoreboard setup. Real footage analysis stays on your own machine.

## No API spend

`--public-demo` removes `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` and `TYPESAFE_API_KEY`
from the process at startup and forces `narration.llm_enabled=false` and `jev.mode=off` for
every run, so narration is template-only and nothing is billed even if the keys are still set
in the Vercel project. Deleting them from the project is still the tidiest option.

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
