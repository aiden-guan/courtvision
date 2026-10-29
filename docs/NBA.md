# NBA broadcasts: analysis and evaluation against the box score

CourtVision can analyse a full NBA TV broadcast and score itself against the official box score and play-by-play.
This page covers how it works, how to run it and what it measures. Everything here is **research use**: NBA footage
and statistics may not be shipped in a product (docs/TRAINING.md §1).

## What the broadcast pipeline does

| Stage | Module | What it solves |
|---|---|---|
| Scorebug | `perception/scorebug_ocr.py`, `perception/vision_ocr.py` | Reads the ESPN/ABC bug (scores, period, game clock with tenths, shot clock) with Apple Vision text recognition. No glyph training. Its layout moves when the shot clock is off, so fields are told apart by pattern and order rather than by fixed boxes. |
| Court | `perception/court_search.py`, `perception/broadcast_court.py` | **No clicks.** The main NBA camera sits at half court, high behind the near sideline, and only pans, tilts and zooms. The camera centre is estimated once per game; each frame then needs only pan, tilt, roll and zoom. |
| Detection | `perception/student.py`, `training/train_student.py` | A ResNet-18 CenterNet trained on this footage. It sees the current and two previous frames (motion makes a blurred ball stand out), at 960x544. |
| Ball | `nba/analyze.py` (`link_ball`), `perception/ball3d.py` | One ball path through the per-frame candidates (dynamic programme; the camera's turn between frames is removed from the motion cost). Ballistic stretches are fitted in 3-D: gravity fixes the depth a single camera cannot see. |
| Players | `nba/analyze.py` (`rectify_box`, `is_referee`) | Boxes re-drawn as a standing player at the box centre through the camera (feet within a few px); referees (grey with black stripes) are kept out of the teams. |
| Events | `events/engine.py` | The existing monocular event engine on these frame states. |
| Scorebug fusion | `nba/scorebug_events.py`, `nba/fusion.py` | Score changes are made baskets of a known team and value; shot-clock resets to 24 mark rim touches; assists follow the corrected makes. |
| Truth | `nba/espn.py`, `nba/evaluate.py` | ESPN play-by-play and box score, placed on video time through the scorebug clock. |

### Court registration in more detail

1. **Line evidence.** Court lines are thin and differ in brightness from the floor around them: white lines on
   dark paint, or the dark blue lines of Chase Center on light wood. A top-hat and a black-hat of the Lab lightness
   catch both. The crowd is dense with such structures while the floor carries only its lines, so the floor mask
   keeps pixels whose neighbourhood has few strong responses.
2. **Search.** Pan, tilt and focal length are searched over a grid. Candidates are scored at three pyramid levels
   by bidirectional, orientation-aware chamfer matching: model lines must land on line pixels of the same
   direction, and line pixels cast back onto the floor must land on court lines. One-way matching was fooled by a
   zoomed-out court squeezed onto the busy part of the image.
3. **Refinement.** A wide-basin pass on the line distance map, then matches along each projected line's normal
   with a robust scale that shrinks with the search radius.
4. **Tracking.** Each frame refines the previous camera, predicted with its velocity. A cut, or a frame that no
   longer fits, drops it. Re-acquisition first tries recently seen views, then a pan/tilt/zoom search at the known
   centre. Close-ups and replays fail it and stay "lost", which the broadcast gate treats as not live.

## Running it on a game

```bash
G=runs/nba/2024-12-25-LAL-GSW
VIDEO=runs/studio/<video id>/source.video
.venv/bin/python tools/nba_scan_scorebug.py $VIDEO --out $G/scorebug.jsonl
.venv/bin/python tools/nba_track_court.py $VIDEO --scorebug $G/scorebug.jsonl --out $G/camera_parts/part0.jsonl
.venv/bin/python tools/nba_analyze.py $G detect
.venv/bin/python tools/nba_analyze.py $G ball
.venv/bin/python tools/nba_analyze.py $G events --tag baseline
.venv/bin/python tools/nba_analyze.py $G eval --tag baseline
.venv/bin/python tools/nba_diagnose.py $G --tag baseline
```

`$G/game.json` names the video and the ESPN event id, plus optional team colours (BGR) for the team classifier.
`espn_summary.json` comes from `courtvision.nba.espn.fetch_summary(event_id, out=...)`. The court tracker can run
in parallel chunks (`--start` / `--duration`), one output file per chunk.

To train the student detector on a game:

```bash
.venv/bin/python tools/nba_teacher_labels.py $VIDEO --scorebug $G/scorebug.jsonl --out $G/teacher
# ball labels: courtvision.training.ball_labels.link_ball_path over each clip
.venv/bin/python -m courtvision.training.train_student --data $G/teacher --out runs/models/student
```

The teacher is RF-DETR Small (COCO) at 1312x736 on 150 clips of 36 frames. Its ball candidates become labels only
where one consistent, ball-coloured path links them. Fast gaps are filled, so the student also learns the blurred
balls the teacher misses.

## In the studio

Copy the trained student to `runs/studio/models/nba-student.pt` and the studio uses it instead of the COCO starter.
With **broadcast** on and no saved calibration, the court is registered automatically. Six frames, a second apart,
give the camera centre, which is saved next to the video's calibration. On macOS the ESPN scorebug is read with
Vision.

## Results

See the "NBA broadcast accuracy" section of the README for the current numbers on Lakers @ Warriors (2024-12-25).
