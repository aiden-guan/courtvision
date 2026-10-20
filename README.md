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

Work in progress. Every module below is implemented with its own test suite, written against the
contract in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and then adversarially reviewed. The
integration layer (`config.py`, `pipeline.py`, `cli.py`, end-to-end tests) is the next step.

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

## Docs

- [`docs/RESEARCH.md`](docs/RESEARCH.md) — the research report (state of the art, where Jev fits, accuracy strategy, legal constraints, roadmap). §1A lists the design decisions that bind the code.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — module contracts and shared event semantics.
- [`docs/TRAINING.md`](docs/TRAINING.md) — data flywheel and fine-tuning.
- [`docs/research/`](docs/research/) — the fact-checked research notes behind the report.

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
