# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased] — round 4

### Summary
Round 4 accuracy pass on the event engine (train seeds 1–6, independent review per change), verified once on the
held-out seeds 101–104: `truth3d` 0.941 → 0.970, `mono` 0.708 → 0.742 (both outside the round-start CI), `video`
0.636 → 0.650 (180 s games: noise). Targets (`mono` / `video` ≥ 0.85, `truth3d` ≥ 0.98) not met on held-out games.

### Detailed Changes

#### Event engine

- 3-D shot apex gate, throw-in restarts on the court, ring-contact step slack, reconciler flip hold (`7b1e0c6`)
- Monocular blocks: loose-ball gate, deferred in-flight blocks, image reversal 12 ft/s (`dc30c32`); the shooter left out of the vote during the loose-ball time (`de913e3`, `f47ea37`); image-speed gate, touch margin, head line at the touch, hold-time block path off by default (`ee94d0d`); a far teammate's phantom gain judged when his stint ends (`ffe5129`)
- Monocular steals: long-flight gains are recoveries, firm dribbles corroborate control (`38d1655`)
- Monocular passes: a pass crossing a teammate's box in flight is one pass (`f3e5e81`)
- Shots: a miss the ball went back to the rim after is not flipped (inferred putback); lost passes end at the clock stop (`c916956`)
- Monocular dribbles: bounce height above the dribbler's feet, a fall into each low point; no self-assist (`eae9c50`)
- Monocular rebounds: hidden holds after a neighbour's win, catches dated in the hidden stretch, loose balls re-credited (`3c806cd`)

#### Docs

- Round-4 train and held-out numbers, corrected round-3 video figure, limitations (`cc7e482`)

### Verification Proof
2,075 deterministic pytest tests passing offline at `ffe5129` (6 skipped, 3 strict expected failures).

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
