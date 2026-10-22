# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to [Semantic Versioning](https://semver.org/).

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
