# Private deployment on Vercel

The root `Dockerfile` runs CourtVision Studio on Vercel's Container Runtime.

## Access control (two layers)

1. **Vercel Authentication** on all deployments, including the production domain: only
   signed-in members of the owning Vercel account can load any page.
2. **Bearer token**: every `/api/*` route and the WebSocket require `COURTVISION_TOKEN`.
   Open `https://<deployment>/#token=<token>` once per browser session (the hash never
   reaches the server or logs). If the variable is unset, the container refuses to start.
3. The app's Host check accepts `*.vercel.app` (set in the Dockerfile `CMD`); add a custom
   domain there with another `--set server.allowed_hosts=...` entry if you attach one.

## Environment variables (Project Settings → Environment Variables, type Sensitive)

| Variable | Purpose |
| --- | --- |
| `COURTVISION_TOKEN` | Required. Dashboard/API bearer token. |
| `ANTHROPIC_API_KEY` | Optional. Claude colour commentary. |
| `TYPESAFE_API_KEY` | Optional. Jev decision layer; spoken Claude lines also need it (line guard). |

Keys are read server-side only; the browser receives booleans saying whether each is set.

## Platform limits that shape what works (Hobby plan)

- Requests, including the WebSocket, end after 300 s. The 180 s Studio demo fits; the
  dashboard reconnects after a cut.
- Request bodies are capped at 4.5 MB, so footage uploads above that fail with 413.
- No persistent disk: uploads and analyses live in `/tmp` and vanish when the instance
  scales down (5 minutes without traffic in production).
- The image installs the core package only. Video analysis needs `courtvision[cv]`
  (PyTorch, RF-DETR), which is not in the image; run that locally.
