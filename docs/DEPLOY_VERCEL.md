# Public video showcase on Vercel

The public site at <https://courtvision-beta.vercel.app> is a static page with a prerecorded,
30-second Lakers–Warriors analysis video. Visitors press play, pause, seek or go fullscreen
using native video controls. The analysis is baked into the recording.

`vercel.json` selects the **Other** framework (`null`), skips installation and building,
and publishes **only `showcase/`**. The root Dockerfile is for optional local Studio use;
it is not used by the public deployment. A deployment should contain static files only,
with no Vercel Functions.

## What is shipped

- `showcase/index.html` and `styles.css`.
- Two local fonts with their licences.
- One H.264 MP4 recording and a WebP poster, with content-versioned filenames.

The video uses `preload="none"`, so the page loads its poster without downloading the film
until a visitor plays it. `/media/*` has a one-year immutable cache policy. Subsequent views
can use the browser cache; first views still count as Vercel CDN transfer and requests.

There are no scripts, uploads, API calls, polling, WebSockets, workers, model downloads,
browser inference, server inference or live simulated games in this site. Its CSP does not
allow JavaScript or network connections. The full Studio remains available locally:

```sh
.venv/bin/python -m courtvision studio --open
```

## Reproduce the recording

The exporter reads existing analysis rather than running inference. It requires the local
source recording, cached detections, linked ball path, camera registration and possession
timeline produced by the research tools. None of these inputs are included in `showcase/`.

```sh
.venv/bin/python tools/render_showcase.py runs/nba/2024-12-25-LAL-GSW \
  --start 981 --seconds 30 --out showcase/media
```

The output contains player track brackets, the linked ball location, and possession/event
labels from the saved `full` analysis. Track numbers are tracker identities, not player
names. The recording is silent. It is a demonstration of recorded model output, not a
claim that every track or event is correct.

The exporter writes content-versioned MP4/poster files and `docs/showcase-recording.json`
with the duration, detector/checkpoint provenance and output/cache/timeline hashes. Update
the source and poster references in `showcase/index.html` after regenerating, and remove
superseded public media from the next deployment. The short edited showcase recording is
the only MP4 exception in `.gitignore`; full source footage and training weights stay ignored.

## Check and deploy

```sh
.venv/bin/python -m pytest tests/test_public_showcase.py -q
python3 -m http.server 8766 --bind 127.0.0.1 --directory showcase
```

Open <http://127.0.0.1:8766>, confirm that play/pause/seek work, and check a narrow viewport.
The browser should request only the page, stylesheet, fonts, poster and selected MP4.
Deploy the repository to the existing CourtVision project:

```sh
vercel link --project courtvision --scope aiden-guans-projects
vercel --prod --scope aiden-guans-projects
```

Verify that the deployment reports static output, that `/api/studio` and old inference asset
paths return 404, and that video range requests return partial content. Historical containers
and retained function bundles can continue contributing storage until separately retired;
switching the production site does not delete old deployments or immediately erase past usage.
