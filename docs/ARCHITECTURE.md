# CourtVision architecture & module contracts

CourtVision turns basketball video into (1) a stream of basketball events, (2) live
box-score / tracking / shot-quality stats, (3) fast typed judgments from TypeSafe
**Jev**, and (4) grounded live narration from **Claude**, plus coach insights,
published to a live dashboard.

```
 video / camera / sim                                  scoreboard OCR / manual / feed
        │                                                          │
        ▼                                                          ▼
 ┌──────────────┐  FrameState  ┌──────────────┐  Event(s)  ┌──────────────┐
 │ perception   │─────────────▶│ events       │───────────▶│ stats        │
 │ detect→track │              │ possession,  │            │ box, tracking│
 │ team, jersey │              │ shots, make/ │            │ xFG, decision│
 │ identity     │              │ miss, reb,   │            │ (EPV-lite),  │
 │ homography   │              │ TO, assists, │            │ lineups,     │
 │ ball Kalman  │              │ reconcile    │            │ insights     │
 └──────────────┘              └──────────────┘            └──────┬───────┘
                                      │ events + GameStats         │
                                      ▼                            ▼
                               ┌──────────────┐  Judgement  ┌──────────────┐
                               │ decision/jev │◀───────────▶│ narration    │──▶ TTS
                               │ gate, import-│             │ facts→Claude │
                               │ ance, play   │             │ validator,   │
                               │ type, insight│             │ templates    │
                               └──────────────┘             └──────┬───────┘
                                                                   ▼
                                                     server: FastAPI + WebSocket dashboard
```

> **Precedence (2026-10-20).** `docs/RESEARCH.md` §1A ("Decisions that supersede
> ARCHITECTURE.md", rows D1-D16) is authoritative. This file was edited to match it. If you find
> a remaining disagreement, follow RESEARCH.md §1A and fix this file.

Everything shares `courtvision/types.py` (read it first). Python 3.12. Core deps are only:
numpy, scipy, opencv-python (one OpenCV wheel only: `trackers` 2.6.1 requires `opencv-python`,
and the headless wheel installs the same `cv2`; D12), pyyaml, fastapi, uvicorn,
anthropic==1.15.0, typesafe-sdk==0.7.5, pydantic (+ pytest). **No scikit-learn, no torch** in
the core. Heavy dependencies live in optional extras and are imported lazily inside functions:
`cv` (`rfdetr[coreml]==1.11.2`, `supervision==0.31.0`, `trackers==2.6.1`, torch), `team`
(transformers, torch, umap-learn, scikit-learn for SigLIP→UMAP→KMeans; D11), `tts`
(`pyobjc-framework-AVFoundation`), and `rnd-agpl` (ultralytics: R&D only, never imported by
shipped code; D9). Run tests with `.venv/bin/python -m pytest tests -q`. Tests must never touch
the network.

## Global conventions

* Units: court feet, seconds, ft/s. See `types.py` header for axes. **Feet are canonical in all
  code (D1).** Source tables in `basketball_rules_stats.md` may be in metres; FIBA metres are
  converted once at the geometry boundary (`ft = m / 0.3048`). Metres appear only at the display
  edge.
* Every module exposes a `@dataclass` config with sane defaults named `<Thing>Config`;
  constructors take `config: <Thing>Config | None = None`. The integration layer loads
  YAML into these dataclasses by field name (`courtvision/config.py`), so keep fields
  plain (float/int/str/bool/tuple/list/dict).
* IDs: event ids `"e000123"` style from a monotonically increasing counter owned by the
  producer (`EventEngine` uses prefix `e`, manual events `m`, reconciliation `r`).
  Shot ids `"s0001"`. Utterance ids `"u0001"`, insight ids `"i0001"`.
* No module except `server/` and `pipeline.py` does I/O beyond reading model files.
* Logging via `logging.getLogger(__name__)`; no prints in library code.

## Synthetic render palette (shared by `sim/render.py` and `perception/detector.py:ColorBlobDetector`)

Defined in `courtvision/sim/palette.py` (owned by sim agent; perception imports it):
```
FLOOR_BGR   = (140, 180, 215)   LINE_BGR  = (250, 250, 250)   PAINT_BGR = (110, 140, 190)
HOME_BGR    = (40, 160, 40)     # green jerseys
AWAY_BGR    = (150, 40, 30)     # blue jerseys
REF_BGR     = (25, 25, 25)      # black referee shirts
BALL_BGR    = (0, 110, 255)     # orange
RIM_BGR     = (0, 0, 220)       # red rim
BACKBOARD_BGR = (235, 235, 235)
SKIN_BGR    = (120, 150, 200)
NUMBER_BGR  = (255, 255, 255)
```
Players are drawn as a filled jersey-coloured rectangle (torso+shorts) with a skin head
circle on top and the jersey number in white; the image bbox of the drawn figure equals
`PlayerState.bbox` from the camera projection.

---------------------------------------------------------------------------------------

## geometry/ (owner: geometry agent)

`courtvision/geometry/court.py`
```python
@dataclass(frozen=True)
class CourtSpec:
    league: str                 # "nba" | "fiba" | "ncaa_m" | "ncaa_w" | "nfhs"
    length: float; width: float # ft
    rim_height: float           # 10.0
    rim_radius: float           # inner radius ft (18 in diameter -> 0.75)
    basket_offset: float        # baseline -> rim centre (ft)
    backboard_offset: float     # baseline -> backboard face (ft)
    three_pt_radius: float      # arc radius from rim centre (ft)
    corner_three_dist: float    # |y - basket_y| of the straight corner segments (ft)
    lane_width: float           # ft
    free_throw_dist: float      # backboard face -> free-throw line (ft)  (NBA 15)
    restricted_radius: float    # ft (0 if league has none)
    center_circle_radius: float
    # Clock, period, timeout, possession and foul rules moved to RuleSpec (D15, below).
    def basket_xy(self, side: Side) -> tuple[float, float]
    def other_side(self, side: Side) -> Side
    def nearest_basket(self, x: float, y: float) -> Side
    def distance_to_basket(self, x: float, y: float, side: Side) -> float
    def is_three(self, x: float, y: float, side: Side) -> bool   # strictly beyond the line; the line is 2.
        # Call it with the shooter's LAST GROUNDED FOOT position before the gather, not the
        # release point (D8).
    def zone(self, x: float, y: float, side: Side) -> str        # "restricted_area"|"paint"|"midrange"|"corner_three"|"above_break_three"|"backcourt"
    def in_bounds(self, x: float, y: float, margin: float = 0.0) -> bool
    def in_paint(self, x: float, y: float, side: Side) -> bool
    def free_throw_xy(self, side: Side) -> tuple[float, float]  # centre of FT line
    def half(self, x: float) -> Side                            # which half x is in
    def landmarks(self) -> dict[str, tuple[float, float]]       # named floor keypoints for calibration
    def line_segments(self, arc_steps: int = 48) -> list[list[tuple[float, float]]]  # polylines for drawing
def court_for(league: str) -> CourtSpec      # factory; raise ValueError on unknown league
```
Values must come from `docs/research/basketball_rules_stats.md` (verified). FIBA metres
converted to feet with `m / 0.3048` (= × 3.28084).

`courtvision/geometry/rules.py` (D15; spec in RESEARCH.md §7A.6 and basketball_rules_stats.md addendum)
```python
@dataclass(frozen=True)
class RuleSpec:
    ruleset: str                         # "nba" | "fiba" | "ncaa_m" | "ncaa_w" | "nfhs" | custom
    period_s: int; n_periods: int; ot_s: int; halves: bool
    running_clock: dict | None           # e.g. {"stop_last_s": 120} or {"mercy_margin": 35}
    timeouts: dict                       # counts per half/game/OT, lengths, carry-over, late-game limits
    possession: str                      # "arrow" | "jump"
    bonus: dict; foul_out: int
    tech_ft: dict                        # {"n_ft": 1|2, "shooter": "any", "resume": "poi"|"division_line"}
    shot_clock_s: int | None; shot_clock_reset_s: int | None   # None = no shot clock (many NFHS states)
    clock_stops_after_fg: dict | None    # e.g. NBA {"q1_3_last_s": 60, "q4_ot_last_s": 120}; NFHS None
def rules_for(ruleset: str, overrides: list[str] | None = None) -> RuleSpec   # loads config/rules/*.yaml
```

`courtvision/geometry/homography.py`
```python
class Homography:                    # maps image px -> court ft on the floor plane
    def __init__(self, H: np.ndarray)               # 3x3 image->court
    @classmethod
    def fit(cls, image_pts, court_pts, ransac_thresh_ft: float | None = 1.0) -> tuple["Homography", np.ndarray, float]
        # normalized DLT (Hartley) + optional RANSAC via cv2.findHomography; returns (H, inlier_mask, rmse_ft); >=4 pts
    def image_to_court(self, pts) -> np.ndarray     # (N,2) -> (N,2); accepts (2,) too
    def court_to_image(self, pts) -> np.ndarray
    def to_json(self) -> dict; @classmethod from_json(cls, d) -> "Homography"

@dataclass
class Calibration:                   # fixed-camera calibration file
    homography: Homography
    image_size: tuple[int, int]
    league: str = "nba"
    rims: dict[str, BBox] = {}       # side -> image bbox (fixed camera)
    def save(self, path) ; @classmethod load(cls, path)
```

`courtvision/geometry/camera.py`
```python
@dataclass
class PinholeCamera:                 # world = court ft (x, y, z up)
    K: np.ndarray; R: np.ndarray; t: np.ndarray; image_size: tuple[int, int]
    @classmethod
    def look_at(cls, position, target, image_size=(1280, 720), hfov_deg=70.0) -> "PinholeCamera"
    def project(self, xyz) -> np.ndarray            # (N,3)->(N,2) px; points behind camera -> nan
    def floor_homography(self) -> Homography        # image->court for z=0
    def bbox_of_cylinder(self, x, y, height, radius) -> BBox   # image bbox of an upright person-like cylinder
    def bbox_of_sphere(self, x, y, z, radius) -> BBox
```

## perception/ (owner: perception agent)

* `detector.py`: `class Detector(Protocol): def detect(self, frame_bgr, frame_idx: int, t: float) -> FrameDetections`.
  - `ColorBlobDetector(DetectorConfig)`: segments the synthetic palette (HSV/BGR distance,
    morphology, connected components). Players = jersey blobs extended upward to include the head;
    emits PLAYER (with `embedding` = mean jersey colour feature), REFEREE, BALL, RIM.
  - `RFDETRDetector(...)`: **the production adapter.** Lazy `rfdetr` / `coremltools` import; same
    contract. CoreML input/output spec, post-processing and the PyTorch-vs-CoreML parity test are in
    `docs/research/cv_pipeline.md` §13 and RESEARCH.md §7A.10. One `.mlpackage` per input size
    (full frame, ball ROI).
  - `UltralyticsDetector(weights: str, class_map: dict[str, ObjClass], conf=0.25, imgsz=1280, device=None)`:
    **R&D only (AGPL-3.0, extra `rnd-agpl`; D9).** Lives in `courtvision/perception/rnd/`; shipped
    code paths must never import it. Lazy `from ultralytics import YOLO`; COCO fallback map
    {"person": PLAYER, "sports ball": BALL}.
* `appearance.py`: `torso_feature(frame_bgr, bbox) -> np.ndarray` (HSV histogram of torso region,
  court-colour suppressed, L1-normalised).
* `tracker.py`: `ByteTracker(TrackerConfig).update(dets: list[Detection], frame_idx: int) -> list[Track]`
  — ByteTrack two-stage association (high/low score), constant-velocity Kalman on (cx, cy, aspect, h),
  IoU (+ optional appearance) cost, Hungarian (`scipy.optimize.linear_sum_assignment`),
  track buffer for lost tracks. Players and referees only (ball is separate).
* `ball.py`: `BallTracker(BallTrackerConfig).update(ball_dets, frame_idx, t, homography | None) -> BallState`
  — picks the most plausible ball detection (gating by Kalman prediction, score), image-space
  constant-acceleration Kalman, short-gap interpolation (`interpolated=True`), `court_xy` by floor homography.
* `teams.py`: `TeamClassifier(TeamConfig)`: `observe(embedding)`, `fit()` (numpy k-means, k=2,
  k-means++ init, fixed seed), `predict(embedding) -> tuple[Team | None, float]`, `is_fitted`.
  Cluster -> HOME/AWAY via `TeamConfig.home_color_bgr`/`away_color_bgr` if given, else cluster order.
* `jersey.py`: `JerseyReader(Protocol).read(frame_bgr, bbox) -> tuple[str | None, float]`; `NullJerseyReader`;
  `JerseyVoter(min_votes=3, min_share=0.6)`: per-track weighted temporal voting.
* `identity.py`: `IdentityManager(roster: Roster | None, IdentityConfig)`:
  `assign(tracks: list[Track], teams: dict[int, tuple[Team|None,float]], frame_idx) -> dict[int, tuple[str, Team|None, str|None]]`
  track_id -> (player_id, team, jersey). Stable roster ids when (team, jersey) is voted; provisional
  `"{team}-t{track_id}"` otherwise; re-links a new track to a recently lost identity by team + jersey + court
  proximity; never assigns one roster id to two live tracks.
* `scoreboard_reader.py`: `ScoreboardROI` (4 corners + named digit cells, JSON-serialisable; set via
  `POST /api/scoreboard_roi`), `rectify(frame, roi) -> np.ndarray`, `DigitClassifier` protocol
  (`classify(cell_bgr) -> tuple[str, float]`, "0"-"9" or "" blank), `SevenSegmentClassifier` (deterministic
  segment sampling; exact on the synthetic renderer's digits), `ScoreboardReader(roi, classifier).read(frame, t)
  -> ScoreboardReading | None` (source="ocr", confidence = min cell confidence).
* `rfdetr_adapter.py`: `RFDETRDetector` (PyTorch `rfdetr` predict path and CoreML `.mlpackage` path,
  post-processing per RESEARCH §7A.10). `trackers_adapter.py`: `TrackersAdapter` wrapping
  `trackers.ByteTrackTracker` / `trackers.BoTSORTTracker` behind the `ByteTracker.update` contract
  (pass `frame=` for BoT-SORT CMC and `timestamp=` when frames are skipped). Both import lazily.
* `pipeline.py`: `PerceptionPipeline(detector, calibration: Calibration, roster=None, config=None, jersey_reader=None)`
  `.process(frame_bgr, frame_idx, t) -> FrameState` — detect, track, team, identity, foot-point homography,
  court-space smoothing + velocity (per-player alpha-beta or Kalman, speed clamp 30 ft/s), ball, rims
  (from detector RIM class or `calibration.rims`), drop players far outside the court (margin 6 ft).

## events/ (owner: events agent)

`courtvision/events/engine.py`
```python
@dataclass
class EventEngineConfig: ... (all thresholds; document each)
class EventEngine:
    def __init__(self, court: CourtSpec, roster: Roster | None = None, config: EventEngineConfig | None = None)
    def process(self, frame: FrameState) -> list[Event]   # new events + revisions produced at this frame
    def flush(self) -> list[Event]                        # end of stream; resolve pending shots etc.
    def add_manual(self, event: Event) -> list[Event]     # operator insert/correct (source="manual", CONFIRMED)
    @property def events(self) -> list[Event]             # latest revision of each event, sorted by t
    @property def holder_id(self) -> str | None
    @property def possession_team(self) -> Team | None
    @property def attacking_side(self) -> dict[Team, Side]  # inferred from shots; config default before evidence
```
Must work in two observation modes, chosen per frame from the data:
1. **3-D mode** (`ball.z` known: simulation truth, 3-D tracking feeds): shot = ball above rim height near a
   basket after release; make = ball centre descends through the rim plane within `rim_radius` (+tolerance).
2. **Monocular mode** (`ball.z is None`, `ball.image_xy` and `frame.rims` present): shot = released ball enters
   the image "shot zone" around/above a rim; make = image trajectory crosses the rim's top line downward with the
   interpolated crossing inside the rim's horizontal extent (with tolerance), then appears below it.
Holder detection: image-space (ball inside an expanded player bbox) when bboxes exist, else court-space
distance from `ball.court_xy` (only trusted when `z` is None or low). Hysteresis to avoid flicker.
Event logic required: POSSESSION, PASS (completed/intercepted), DRIBBLE (bounce detection), SHOT_ATTEMPT
(release-time timestamp, release location, value via `court.is_three` at the shooter's **last grounded
foot position before the gather** on the target basket (D8), shot_type,
distance, closest defender, catch_and_shoot, dribbles_before), SHOT_MADE/SHOT_MISSED, BLOCK (released upward,
reversed by a defender before reaching the rim zone), REBOUND (offensive/defensive, team rebound when
the ball goes out/dead), ASSIST (last pass to scorer within `assist_window_s`, configurable max dribbles),
STEAL + TURNOVER, out-of-bounds turnovers, FREE THROW attempts (shot from FT line by a stationary shooter,
value 1), PERIOD_START/END from scoreboard. Scoreboard reconciliation (`reconcile.py`): score deltas confirm
or correct shot results (status CONFIRMED / revisions / SCORE_CORRECTION with needs_review).
Scoreboard OCR runs at 5 Hz; a score change needs 2 identical consecutive reads (D14).
`events/scoreboard.py`: `ScoreboardConfirmer` — feeds raw `ScoreboardReading`s (5 Hz OCR/feed/manual);
a score change is confirmed by 2 identical consecutive reads with delta +1/+2/+3 and a consistent clock;
manual/feed/sim sources confirm immediately. `events/deadball.py`: `DeadBallDetector` (§7A.5 rules 1-4,
plus `mark_dead_ball(t)` for the operator hotkey) exposing `is_dead(t)`, `expected_window_s(t)` (from
RuleSpec + stoppage type) and stoppage-type labels ("timeout", "free_throw", "period_break", "out_of_bounds",
"foul", "made_basket", "unknown").
`events/calibration.py`: one logistic cue model per gated event (shot, make, three, rebound, steal,
block, player ID) turns named cues into `Event.confidence`, with `attrs["p_status"]` in
{"uncalibrated","platt","isotonic"} (RESEARCH.md §7A.2). Narration gates treat `uncalibrated` as
hedge-only. Events with `source` in {"sim","manual"} or status CONFIRMED by the scoreboard/operator are
not subject to hedge-only; `EventEngineConfig.assume_calibrated=False` may be set True only in
simulator evaluations. Every CV constant in `EventEngineConfig` records its source and status
(sourced / unverified / tuned). The engine takes a `RuleSpec` for clock-stop, possession-arrow,
technical/flagrant FT sequences, foul-out and `shot_clock=None`. Dead balls come from the
audio-free detector (clock stopped on OCR, FT formation, stillness, operator hotkey; §7A.5).

### Shared event semantics (the simulator's truth and the engine MUST both follow these)

Stat grammar: FIBA Statisticians' Manual 2024 (RESEARCH §2.5) with NBA-league flags where noted.
| Kind | `t` (timestamp) | Actor fields | Rules |
|---|---|---|---|
| POSSESSION | first frame of confirmed control by the new team | team, player_id = first holder | not emitted for an offensive rebound (same team keeps possession); `attrs.cause` |
| PASS | release frame | player_id passer, other_player_id intended/actual receiver | completed = a teammate gains control; intercepted ⇒ STEAL+TURNOVER instead of a completed PASS (emit PASS with `completed=False` too) |
| DRIBBLE | floor-contact frame | player_id | not narrated |
| SHOT_ATTEMPT | release frame | player_id shooter, team | value 2/3 from `court.is_three` at the last grounded foot position before the gather (D8; in sim = shooter position at gather); FT attempts value 1, shot_type "free_throw". A shooter fouled before the release gets NO attempt (2 FTs, 3 on a three); an **and-one** is an attempt + SHOT_MADE with the FOUL (`attrs.and_one`) at the release frame (continuation), then 1 FT |
| SHOT_MADE / SHOT_MISSED | frame the ball centre crosses the rim plane (made) / the frame the miss is decided (ball leaves the rim zone downward or sideways, or a player gains control) | player_id shooter, shot_id | made `value` = points |
| BLOCK | defender contact frame | player_id blocker, other_player_id shooter, shot_id | only with an FGA (attempt + SHOT_MISSED also emitted). Any contact before the ball touches the rim, at the rim too (FIBA; rules note §2a): most blocks are on rim and short-mid attempts (pbpstats .095 / .102 of FGA) |
| REBOUND | first frame of confirmed control after a missed FGA/last FT | player_id rebounder (None = team rebound), `attrs.offensive` | team rebound when the ball goes out of bounds / dead before control; credited to the team that inbounds |
| ASSIST | the SHOT_MADE frame | player_id passer, other_player_id scorer, shot_id | last completed pass to the scorer from a teammate, catch→release ≤ `assist_window_s` = 5.0 s and scorer dribbles in between ≤ `assist_max_dribbles` = 2. Never on free throws (NBA flag; FIBA would count FT trips — configurable `assist_on_ft_trips=False`) |
| STEAL + TURNOVER | frame the stealer('s team) gains control | STEAL: player_id stealer, other_player_id victim; TURNOVER: player_id victim, `type` "bad_pass" (intercepted pass) or "steal" (stripped dribble/hold) | the first deflector gets the steal (rules note §2a): a ball poked loose and recovered by his teammate is his STEAL at the teammate's control frame (POSSESSION player_id = the recoverer); recovered by the offence = no steal, no turnover |
| TURNOVER (non-steal) | frame the ball goes out / the violation | player_id last holder, `type` "out_of_bounds" / "shot_clock" / "travel" / "offensive_foul" / "other" | out of bounds attributed from the next inbound (one camera can't see the last touch): confidence < 1. A ball the **defence** knocks out of bounds is no turnover and no steal: the offence throws in with the unexpired shot clock (rules note §2a) |
| FOUL | whistle frame | player_id fouler, other_player_id fouled | sim truth only + manual/feed; CV engine does not infer fouls. `attrs.type` "shooting" / "personal" (non-shooting: FTs only in the RuleSpec team-foul penalty, else a throw-in) / "offensive" (with a TURNOVER "offensive_foul"); `attrs.n_ft` the free throws awarded |
| SUBSTITUTION | entry frame | player_id in, other_player_id out | operator / sim truth. The substitute waits at the table (`SimResult.waiting`, rendered) and steps on after the outgoing player has left the floor. Operator entries arrive `operator.entry_delay_s` later and are matched within tolerance + that lag (`EventEvalConfig.operator_lag_s`) |
| PERIOD_START / PERIOD_END | first / last frame of the period (PERIOD_END = the frame the game clock reaches 0) | — | from scoreboard / RuleSpec. A shot released before the horn still plays out and counts after PERIOD_END; no rebound follows its miss |
Matching tolerance for evaluation: ±1.0 s (`eval.match_events(tol_s=1.0)`). A run cut mid-game
(`match_events(window=(start, end))`; the pipeline passes its stream edges) does not score unmatched
events within the kind's tolerance of a cut edge (a shot still in the air at the end, an event too late
to confirm): they are reported in `EventReport.boundary_excluded`, matched pairs always count.

## stats/ (owners: stats-core agent and stats-models agent)

stats-core owns `stats/engine.py`, `stats/boxscore.py`, `stats/tracking.py`, `stats/advanced.py`, `stats/lineups.py`.
stats-models owns `stats/shot_quality.py`, `stats/decision.py`, `stats/insights.py`.

```python
# stats/engine.py
@dataclass class PlayerLine:   # counting stats per player (+ derived properties efg, ts, fg_pct, ...)
    player_id: str; team: Team | None; pts=0; fgm=0; fga=0; fg3m=0; fg3a=0; ftm=0; fta=0
    oreb=0; dreb=0; ast=0; stl=0; blk=0; tov=0; pf=0; plus_minus=0; seconds=0.0
    xpts=0.0                      # sum of expected points of attempts
    @property reb, efg, ts, fg_pct ...
@dataclass class TeamLine:     # same counting stats + team_reb, possessions (counted), possessions_est,
    ...                           # ppp, pace, efg, ts, paint_pts, second_chance_pts, fast_break_pts,
                                  # pts_off_turnovers, xpts, avg_xfg
@dataclass class TrackingLine: player_id; distance_ft; top_speed_fts; avg_speed_fts; seconds_tracked;
    time_of_possession_s; touches; paint_touches; passes_made; passes_received; dribbles
@dataclass class ShotRecord: shot_id; player_id; team; t; x; y; value; made: bool | None; distance_ft;
    zone; shot_type; xfg; closest_defender_ft; assisted_by
@dataclass class GameStats:
    t: float; period: int | None; game_clock: float | None
    score: dict[Team, int]; players: dict[str, PlayerLine]; teams: dict[Team, TeamLine]
    tracking: dict[str, TrackingLine]; shots: list[ShotRecord]
    run: dict            # current scoring run {"team": Team|None, "points": int, "opp_points": int, "since_t": float}
    lineups: dict        # lineup id -> {"team","players","seconds","plus_minus","pts_for","pts_against"}
                         # built ONLY from operator lineup entries (POST /api/lineup); empty otherwise
    decisions: list[dict]  # DecisionGrade.to_dict() entries
    def to_dict(self) -> dict   # JSON-safe (enums -> values)
class StatsEngine:
    def __init__(self, court, roster, config=None, shot_model: ShotQualityModel | None = None,
                 decision_model: DecisionModel | None = None)
    def on_events(self, events: list[Event]) -> None   # accepts revisions/retractions (latest revision wins)
    def on_frame(self, frame: FrameState, holder_id: str | None, possession_team: Team | None) -> None
    def set_lineup(self, team: Team, player_ids: list[str], t: float) -> None      # operator: starting five / reset
    def substitute(self, team: Team, out_id: str, in_id: str, t: float) -> None    # operator tap-to-sub
    def snapshot(self) -> GameStats
```
Player seconds / plus_minus / lineups come ONLY from `set_lineup`/`substitute` (operator entries; in
simulation `sim/operator.py:SimOperator` replays the sim's true substitutions through this same API with a
configurable entry delay). Without entries those fields stay 0 and lineups empty. Every PlayerLine is
"provisional" unless its contributing events are CONFIRMED (`GameStats.to_dict()` exposes
`players[*].provisional`). Team points always equal the confirmed scoreboard when one is present.
`stats/advanced.py`: pure functions `efg(fgm, fg3m, fga)`, `ts(pts, fga, fta)`, `possessions_est(fga, oreb, tov, fta)`,
`pace(...)`, `ortg(...)` with the coefficients from `basketball_rules_stats.md`.

```python
# stats/shot_quality.py
@dataclass class ShotFeatures: distance_ft: float; value: int; shot_type: str = "jumper";
    closest_defender_ft: float | None = None; catch_and_shoot: bool | None = None; zone: str | None = None
class ShotQualityModel:
    def __init__(self, league: str = "nba", config=None)
    def xfg(self, f: ShotFeatures) -> float            # P(make) in (0,1); free throws use league FT%
    def expected_points(self, f: ShotFeatures) -> float
    def fit(self, features: list[ShotFeatures], made: list[bool], l2: float = 1.0) -> None  # refit on own data
    def to_json(self) -> dict; @classmethod from_json(cls, d)
# stats/decision.py  (EPV-lite)
@dataclass class DecisionOption: kind: str ("shoot"|"pass"); target_id: str | None; success_prob: float;
    xpts_if_success: float; value: float; detail: dict
@dataclass class DecisionGrade: t; player_id; chosen: DecisionOption; best: DecisionOption;
    options: list[DecisionOption]; delta: float; label: str ("good"|"neutral"|"poor"); def to_dict()
class DecisionModel:
    def __init__(self, court, shot_model, config=None)
    def options(self, frame: FrameState, holder_id: str, offense: Team, basket: Side) -> list[DecisionOption]
    def grade(self, frame: FrameState, event: Event, offense: Team, basket: Side) -> DecisionGrade | None  # for PASS / SHOT_ATTEMPT
# stats/insights.py
class InsightEngine:
    def __init__(self, roster, court, config=None)
    def update(self, stats: GameStats, new_events: list[Event], t: float) -> list[Insight]  # deduped, rate-limited
    # xeFG / expected-eFG cards only when league == "nba" until per-league priors exist from our own
    # games (>= 2,000 labelled shots per league tier); otherwise raw eFG with n (RESEARCH.md §7A.2).
    # Card labels sent to Jev are neutral (card_a..card_h) and card text is pseudonymised (§7A.9).
```

## decision/ (owner: jev agent) — TypeSafe Jev decision layer

Read `docs/research/sdk_ground_truth.md` (authoritative SDK facts), `docs/research/jev_api.md`,
and `docs/research/jev_integration_design.md` (§4 bundles, §5 questions VERBATIM, §6 state schemas,
§7 client, §7.4 decoders/thresholds/fallbacks, §11 gap-pass addendum). The design note's
`courtvision/jev/...` paths map to `courtvision/decision/...` here.

Files: `decision/client.py` (`JevClient`, `Bundle`, `BUNDLES`, `JevResult`, `Breaker`, `TTLCache`,
`StubAnswers`, modes live/stub/replay/off), `decision/questions.py` (B1-B6 question objects / builders,
verbatim from §5), `decision/states.py` (B1-B5 state builders + bucket helpers `time_bucket`,
`margin_text`, `clutch_text`, pseudonymised roles/tokens only), `decision/decoders.py`
(`Thresholds`, `usable`, `ensembled_choice`, `decide_shot_outcome`, ...), `decision/advisor.py`
(the façade below).
```python
@dataclass
class Judgement:
    label: str | None            # Choice label (or None)
    value: float | None          # Noul P(yes), or Score normalised to [0,1] (score / (levels-1))
    confidence: float            # Choice/Score confidence; for Noul: |2p-1|
    probabilities: dict[str, float]
    source: str                  # "jev" | "cache" | "stub" | "replay" | "fallback"
    latency_ms: float = 0.0
    model: str | None = None     # resp.model that answered (None for fallback)
@dataclass
class PossessionAssessment:      # B1 possession_end (flavour + play type; tiers A/D)
    play_type: Judgement         # Synergy-style label, order-ensembled (play_type + play_type_rev)
    summary_clear: Judgement     # Noul summary_is_clear
    stakes: Judgement            # 4-level Score moment_stakes, value normalised to [0,1]
    highlight: Judgement         # Noul highlight_play
@dataclass
class LineVerdict:               # B4 line_guard, one per sentence
    sentence: str; supported: float | None; unlisted: float | None; speculation: float | None
    passed: bool                 # thresholds: supported >= 0.90, unlisted <= 0.20, speculation <= 0.30
    source: str
@dataclass class JevConfig:
    enabled: bool = True; mode: str = "live"          # "live" | "stub" | "replay" | "off"
    model: str = "jev-1.13.0"                          # pinned (D2); log resp.model; other versions = untuned
    deadlines_s: dict = {"B1": 1.5, "B2": 1.0, "B3": 1.0, "B4": 0.8, "B5": 0.7, "B6": 5.0}  # initial (D7)
    max_in_flight: int = 8; cache_enabled: bool = False; images_enabled: bool = False
class JevDecider:                # a.k.a. JevAdvisor in the research notes
    def __init__(self, config: JevConfig | None = None, client: JevClient | None = None)
    @property def available(self) -> bool             # False when off / no key / session disabled (401/403)
    async def possession_end(self, state: dict) -> PossessionAssessment          # B1
    async def next_line_kind(self, state: dict, available_kinds: set[str],
                             storylines: list[str]) -> tuple[Judgement, list[Judgement]]  # B2 (+ storyline_due_i)
    async def pick_insight(self, candidates: list[Insight], ctx: dict) -> Insight | None   # B3; labels card_a..card_h
    async def guard_lines(self, facts: dict, sentences: list[str]) -> list[LineVerdict]   # B4 (<= 2 sentences/call)
    async def ambiguity(self, kind: str, cues: dict, rule_lean: str) -> Judgement          # B5 "shot_outcome"|"release_intent"
    def stats(self) -> dict                            # per-bundle calls, fallbacks, p50/p95 latency, models seen
    async def aclose(self) -> None
```
Every method computes its deterministic fallback FIRST and returns it (`source="fallback"`) when Jev is
off, unavailable, past its deadline, errors, breaker-open, answers from an unpinned model, or is below
threshold — a missing Jev answer only makes output plainer, never wrong. `guard_lines` fallback is
`passed=False` (D4: no Jev verdict ⇒ the LLM line is not spoken). Numbers are always computed in Python
and sent as named buckets; never ask Jev to count or do arithmetic. States are pseudonymised: no player
names, only per-game tokens or roles (D16; RESEARCH.md §7A.7). B5 is used only when the caller's cue
model is calibrated (the caller checks); B6 only when `images_enabled`.

## narration/ (owners: narration-core agent, then narration-io agent)

narration-core owns `facts.py`, `pseudonym.py`, `templates.py`, `validator.py`, `director.py`.
narration-io owns `llm.py`, `candidates.py`, `narrator.py`, `tts.py`, `captions.py` and
`tools/latency_harness.py`. Read `docs/research/narration.md` (director, priorities, TTLs, template
variants, validator checks, corrections, AD lexicon) and RESEARCH.md §2.6, §4.1, §7A.2, §7A.3,
§7A.7, §7A.8.

Core contracts (narration-core):
```python
# pseudonym.py — D16: cloud payloads carry tokens, never names
class PseudonymMap:              # per game, in RAM only
    def __init__(self, roster: Roster, seed: int = 0)       # players -> "P01".."P30" (shuffled, not jersey order)
    def token(self, player_id: str) -> str; def player(self, token: str) -> str | None
    def render(self, text: str, roster: Roster, id_conf: dict[str, float] | None = None,
               jersey_conf: dict[str, float] | None = None) -> str
        # "{P07}" -> name if id_conf >= 0.95; "Home's number 23" if 0.80 <= id_conf < 0.95 and jersey_conf >= 0.95;
        # else the team name (identity tiers, RESEARCH §4.1). Missing conf => 1.0 (sim/manual).
# facts.py
def build_facts(events: list[Event], stats: GameStats, roster: Roster, court: CourtSpec,
                frame: FrameState | None, pseudo: PseudonymMap, rules: RuleSpec | None = None,
                recent_lines: list[str] | None = None) -> dict
    # FactPacket: {"state_version", "event": {...tokens...}, "score", "clock" (pre-formatted strings),
    #  "players": {token: {...counting stats...}}, "teams", "run", "derived": [code-made phrases],
    #  "storylines": [...], "allowed_numbers": [str...], "allowed_tokens": [...], "recent_lines": [...],
    #  "hedge_only": bool, "confirmed": bool}
# templates.py — deterministic, always-correct fallback; >= 8 variants per (kind, outcome, channel)
def render(events: list[Event], facts: dict, channel: str, rng: random.Random, hedge: bool = False) -> str
    # returns text with {Pnn} tokens; hedge=True never states an outcome or point value
# validator.py
@dataclass class ValidationResult: ok: bool; problems: list[str]
class GroundingValidator:
    def validate(self, text: str, facts: dict) -> ValidationResult
        # tokens ⊆ allowed_tokens; numbers (digits + number words + ordinals) ⊆ allowed_numbers;
        # claims lexicon (career/season/record/streak/injury/... unless in facts); score consistency;
        # tense; max words; repetition vs recent_lines. Target < 5 ms.
# director.py — deterministic owner of speak-now / priority / preemption (D3)
@dataclass class Cue: cue_id; priority: int (0..3; 0 = highest); text_template: str (tokens); t_occ: float;
    ttl_s: float; event_ids: list[str]; channel: str; hedge: bool; state_version: int
class Director:
    def __init__(self, config: DirectorConfig | None = None)
    def offer(self, cue: Cue, now: float) -> None          # enqueue; may preempt lower priority
    def next(self, now: float) -> Cue | None                # pop the best non-expired cue respecting cooldowns
    def on_spoken(self, cue: Cue, now: float) -> None
    def preempt_requested(self, now: float) -> bool         # a higher-priority cue is waiting
    def invalidate(self, state_version: int | None = None, event_ids: list[str] | None = None) -> None
```
(narration-io):
```python
@dataclass class NarrationConfig: mode="broadcast" ("broadcast"|"coach"|"accessibility"); llm_enabled=True;
    model="claude-opus-5-5"; effort="low"; max_tokens=1536 (D13: thinking counts toward it);
    candidate_deadline_s=12.0 (D6: no first-token abort; reset from the measured p95; Claude only when
    expected_window_s >= p95 + tts + 1 s); max_words: dict[str,int]; cooldown_s: dict[str,float];
    use_jev_validator=True (D4: LLM lines require the B4 guard); pseudonymise=True (D16);
    tts="avspeech"|"none"|"say" ("say" is dev-only, D5)
class NarrationLLM(Protocol):
    def stream(self, system: str, prompt: str, max_tokens: int) -> AsyncIterator[str]
class ClaudeNarrationLLM:   # anthropic.AsyncAnthropic, beta stream, fallbacks="default" (see sdk_ground_truth.md);
                            # speculative candidates only; logs response.model and usage.output_tokens
facts.py:      build_facts(events: list[Event], stats: GameStats, roster: Roster, court: CourtSpec, frame: FrameState | None) -> dict
templates.py:  render(events, facts, channel, rng) -> str          # deterministic, always-correct fallback
validator.py:  GroundingValidator().validate(text: str, facts: dict) -> ValidationResult(ok: bool, problems: list[str])
narrator.py:   class Narrator:
    def __init__(self, roster, court, config=None, llm: NarrationLLM | None = None, jev: JevDecider | None = None,
                 on_utterance: Callable[[Utterance], Awaitable[None] | None] | None = None, tts: TTS | None = None)
    async def on_events(self, events: list[Event], stats: GameStats, frame: FrameState | None) -> None
    async def on_insights(self, insights: list[Insight], stats: GameStats) -> None
    async def tick(self, t: float, stats: GameStats) -> None   # dead-ball colour scheduling
    async def aclose(self) -> None
    transcript: list[Utterance]
tts.py: class TTS(Protocol): async speak(text, priority) ; async stop(); NullTTS;
        AVSpeechTTS (AVSpeechSynthesizer via pyobjc-framework-AVFoundation, warmed at boot, write-path
        buffers into our mixer, word-boundary cut; D5); MacSayTTS (subprocess `say`, dev-only fallback)
captions.py: caption/AD cue emitter keyed to source video PTS (OBS SendStreamCaption, YouTube HTTP POST,
        HLS WebVTT with X-TIMESTAMP-MAP, sidecar WebSocket; RESEARCH.md §7A.8)
```
Rules: the deterministic director decides speak-now, priority and preemption (no Jev gate, D3);
narrate only facts present in the fact packet; outcome words only when the event's cue model is
calibrated and its gate certified, otherwise hedge-only wording (RESEARCH.md §7A.2); provisional events may be narrated only for
low-risk kinds (attempts, passes) — outcomes wait for the engine's result; RETRACTED/revised events
produce a short correction on the "correction" channel; LLM output is streamed, cut into sentences,
each sentence validated; failures fall back to templates; never block the event loop.

## sim/ (owner: sim agent)

```python
@dataclass class SimConfig: league="nba"; seconds=600.0; fps=30.0; seed=7; image_size=(1280,720);
    camera_position=(47.0, -38.0, 26.0); camera_target=(47.0, 25.0, 0.0); hfov_deg=72.0; ...
class GameSimulator:
    def __init__(self, config: SimConfig | None = None)
    def run(self) -> SimResult
@dataclass class SimResult:
    config: SimConfig; court: CourtSpec; roster: Roster; camera: PinholeCamera
    frames: list[FrameState]     # TRUTH: exact court xy, ball court_xy + z, image_xy & bboxes via camera,
                                 # rims via camera, scoreboard each frame (source="sim")
    events: list[Event]          # TRUTH events, status CONFIRMED, source "sim"
    truth_box: dict[str, dict[str, int]]   # player_id -> counting stats, independently tallied
    truth_score: dict[Team, int]
class NoiseModel(NoiseConfig).apply(frames, seed) -> list[FrameState]
    # observed copy: position/bbox noise, ball dropouts & false positives, optional brief id swaps,
    # monocular=True removes ball z and replaces ball court_xy with the floor-homography projection of image_xy
render.py: SyntheticRenderer(camera, court).render(frame: FrameState, referees: list[tuple[float,float]] | None = None) -> np.ndarray
    # BGR, palette above; also draws a venue SCOREBOARD panel at a fixed image location with
    # seven-segment digits (home, away, period, game clock MM:SS, shot clock) from frame.scoreboard;
    # SCOREBOARD_ROI (corners + digit cell boxes) is exported from render.py for the reader.
palette.py: the colours above
names.py: fictional team & player names (never real people)
operator.py: SimOperator(result).entries() -> list[dict]  # starting fives + true subs as operator lineup
    # entries {"t","team","out","in"} delayed by config entry_delay_s; drives StatsEngine.set_lineup/substitute
SimResult also carries `referees: list[list[tuple[float,float]]]` (per frame) and `substitutions` truth.
```
Possession model: realistic flow — inbound / bring-up, 1–6 passes, dribbles with real ball bounces,
drives, shots with a league-realistic distance mix and make probability, offensive/defensive rebounds,
turnovers & steals, occasional blocks, shooting fouls with free throws, substitutions at dead balls,
halftime side switch, game/shot clock, scoreboard.
Physical realism (round-1 sim pass; constants in `SimConfig` with sources / statuses):
* bodies never interpenetrate: every pair is projected apart to `body_contact_ft` (1.5 ft, FIBA Art. 33.1
  cylinder principle), locked players included, and no other body centre comes within `ball_clear_ft`
  of a held / dribbled ball; on-ball defenders keep `on_ball_gap_ft` and slide laterally under the rim;
* person boxes follow the body: jumps lift the box off the floor (shots, blocks, high rebound catches,
  jump balls; `SimResult.feet_z`), raised arms and a ball held above the shoulders extend it;
* passes peak at most at `pass_apex_ft` and never cross the zone above a rim; overthrown lobs at most
  `oob_lob_apex_ft`; catches happen at their exact flight time;
* live-ball steals: the stealer pushes in transition at once, the robbed player reacts late; half of the
  strips are poke-aways recovered by the deflector;
* block rates per zone from pbpstats (~5.5% of FGA); fouls: pre-release 2- and 3-shot fouls, and-ones,
  non-shooting fouls with RuleSpec team-foul penalties (one-and-one where the ruleset has it), offensive
  fouls; out of bounds: overthrown lobs, fumbled catches, dribblers stepping out, defensive deflections
  (offence keeps the ball);
* rebounds: 1-3 players contest the landing spot, defenders box out crashing men, the rebounder has the
  inside position; tip putbacks go straight back up;
* PERIOD_END at the horn; substitutes walk on from the table as separate people.
`SimResult` also carries `waiting` (substitutes at the table per frame) and `feet_z` (per frame, each
player's feet height). `NoiseModel.apply(frames, seed, feet_z=...)` (mono): AR(1)-correlated position
error, alpha-beta velocities from the noisy positions (never the simulator's), foot-point lift of jumping
players, clustered occlusion drops, same-team identity swaps after image occlusions, Gilbert-Elliott
burst ball misses plus depth-ordered body occlusion and the net, persistent false positives (heads,
static blobs) with overlapping confidence. `ScoreboardModel` (noise.py): the operator-run venue board
(score / clock / shot-clock lags, MM:SS display, tenths in the last minute) read by 5 Hz OCR (source
"ocr"); `source.scoreboard="auto"` gives truth3d the exact board and mono / video the operator board
(`board.*` settings). The renderer draws the ball in depth order (only upper bodies hide it), with a
motion-blur streak, waiting substitutes (`render(frame, referees, extras)`) and optional hand cover /
spare balls.

## eval/ (owner: eval agent)
```python
events.py:   match_events(pred, truth, kinds=None, tol_s=1.0, match_player=True) -> EventReport
             # greedy-by-time one-to-one matching per kind; precision/recall/F1, player accuracy, mean |dt|
boxscore.py: compare_box(stats: GameStats, truth_box, truth_score, edge=None) -> BoxReport  # per-stat MAE, exact %, score match
             # edge = edge_contributions(truth events at the cut stream end): each truth value is taken with
             # or without that part, whichever is closer ("don't care", e.g. a last-second basket the
             # operator board shows only after the stop); the pipeline passes the last
             # max(tol, board lag + 2 OCR reads) seconds
tracking.py: tracking_report(pred_frames, truth_frames, match_radius_ft=3.0) -> TrackReport  # MOTA, IDF1-lite, RMSE
report.py:   format_report(...) -> str (markdown table)
bench.py:    BenchCase, run_case(case) -> aggregate record, pool_mode(records), build_report, format_markdown
             # the accuracy benchmark behind tools/accuracy_bench.py: each (mode, seed) runs the
             # `courtvision simulate` code path (cli.simulate_config -> pipeline.run_offline); per-kind
             # F1 pooled over seeds (sum TP / pred / truth), headline = eval's macro-F1 on the pooled
             # F1s. Splits: train seeds 1-6 (tune here), held-out 101-104 (verifier; aggregates only).
             # Next to the strict headline: the macro over core kinds with truth support, the pooled
             # micro-F1, false positives per stream minute of core kinds without truth, and 95%
             # seed-bootstrap intervals (bootstrap_ci). Only pooled multi-seed >= 180-300 s numbers are
             # comparable; single-seed <= 60 s headlines are not.
events.py also: match_events(..., window=(start, end)) ignores unmatched events at a cut stream edge;
             EventEvalConfig.operator_lag_s widens operator-entered kinds (substitution) by the entry lag;
             EventReport carries supported_macro_f1 / micro_f1 / unsupported_fp / boundary_excluded and
             the window / lag it was scored with (`courtvision eval RUN_DIR` re-scores with them)
```

## server/ (owner: server agent)
`server/hub.py`: `Hub` — async pub/sub with per-client queues (bounded, drop-oldest), replay of latest
`state`/`stats` + last 200 events/utterances to late joiners.
`server/app.py`: `create_app(hub: Hub, controller: Controller | None = None) -> FastAPI`
* `GET /` dashboard (static files in `server/static/`), `WS /ws`
* `GET /api/state`, `GET /api/events`, `GET /api/boxscore`, `GET /api/transcript`
* `POST /api/events` (manual event JSON -> controller.add_manual), `POST /api/scoreboard`
* `POST /api/lineup` (operator tap-to-sub: {"team","out","in","t"}; the only P0 source of lineups, minutes, +/-)
* `POST /api/scoreboard_roi` (4 corners + digit cells for the venue scoreboard reader, §7A.5)
* `GET /api/frame.jpg` (latest frame), `POST /api/calibration` ({"image_points": [[u,v]...], "court_points": [[x,y]...] or "landmarks": [names]})
* `POST /api/control` ({"action": "pause"|"resume"|"speed", "value": float})
`Controller` protocol (implemented by pipeline): `add_manual(dict)`, `set_scoreboard(dict)`, `calibrate(dict) -> dict`,
`control(dict)`, `latest_frame_jpeg() -> bytes | None`.

WebSocket message schema (JSON, every message has "type" and "t"):
```
{"type":"frame","t":..,"frame_idx":..,"players":[{"id","team","jersey","name","x","y","speed"}],
 "ball":{"x","y","z","visible","holder"},"possession":"home"|"away"|null,"clock":{"period","game","shot"}}
{"type":"event","t":..,"event":Event.to_dict()}                    # includes revisions
{"type":"utterance","t":..,"utterance":{...Utterance fields}}
{"type":"stats","t":..,"stats":GameStats.to_dict()}                # throttled ~1 Hz
{"type":"insight","t":..,"insight":{...Insight fields}}
{"type":"meta","t":..,"roster":[...],"teams":{"home":name,"away":name},"court":{...CourtSpec fields},"rules":{...RuleSpec fields}}
{"type":"caption","t":..,"pts":..,"text":..,"channel":"caption"|"ad"}   # cue keyed to source video PTS
```
Dashboard: live top-down court (canvas) with players/ball, score & clock bar, narration transcript,
event feed (with correction button per event), box score tables, shot chart (made/missed, xFG colour),
coach insights panel, decision-quality list, calibration tab (click image points + choose landmarks).

## training/ (owner: training agent)
`training/align_pbp.py` (play-by-play ↔ video alignment via clock timeline), `training/fit_shot_model.py`,
`training/export_yolo.py` (pseudo-label export), `docs/TRAINING.md`.

## Integration (owner: integration phase)
`courtvision/config.py`, `courtvision/config/default.yaml`, `courtvision/pipeline.py` (sources: video file,
camera, simulation; perception thread -> asyncio; engine -> stats -> insights -> narration -> hub),
`courtvision/cli.py` (`simulate`, `serve`, `run`, `calibrate`, `render`, `eval`, `config`, `latency`), README.

Implemented contracts (integration core):
```python
# config.py
load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> AppConfig
    # defaults <- YAML <- "a.b=value" overrides; unknown keys -> ConfigError with the dotted path;
    # nested sections partial, dict fields merge key by key, lists replace
dump_config(cfg) -> dict; config_to_yaml(cfg) -> str; render_default_yaml() -> str  # = config/default.yaml
# AppConfig sections: game, source, sim, noise, render, board, operator, perception, events, stats, shot_model,
# decision_model, insights, jev, narration, director, facts, templates, candidates, tts, captions, llm,
# server, hub, serve, pipeline, eval. game.* is authoritative over sim.league / ruleset / rule_overrides.

# pipeline.py
class FrameItem(NamedTuple): frame_idx: int; t: float; image: np.ndarray | None; state: FrameState | None
class FrameSource:  # async iterator of FrameItem; realtime / live / speed / fps / image_size / roster / court / rules / truth
SimSource(sim_result, mode="truth3d"|"mono"|"video", noise=None, realtime=False, speed=1.0, *, noise_seed,
          start_s, duration_s, scoreboard="truth"|"ocr_5hz"|"none", render_config)
VideoFileSource(path, realtime=True, speed=1.0, ...)   # cv2.VideoCapture, t = PTS
CameraSource(index_or_url, ...)                         # live, t = wall clock since the first frame
make_source(cfg, sim_result=None, realtime=None) -> FrameSource
class LiveGame:
    def __init__(self, config, source, hub=None, *, realtime=None, jev=None, llm=None, tts=None, env=None)
    async def run(self) / start() / process(item) / finish()
    controller: GameController          # implements server.app.Controller
    events / transcript / insights_emitted / snapshot() / latency_report() / status()
run_offline(cfg, source=None, out_dir=None, *, hub=None, jev=None, llm=None, write=True, env=None) -> RunResult
    # source-time (virtual) clocks => deterministic; writes events.jsonl, truth_events.jsonl,
    # transcript.jsonl, insights.jsonl, stats.json, report.json, report.md, config.yaml
```
Per frame: (perception in a worker thread for image sources in real time; inline offline) ->
`EventEngine.process` -> enrichment (`p_three` copied from the attempt onto its result; `attrs.id_conf` /
`attrs.jersey_conf` from `PlayerState.confidence` and the jersey vote share; always 1.0 for sim / manual /
feed, also over values an operator correction copied from the CV event it revises) ->
`StatsEngine.on_events` / `on_frame` -> `InsightEngine.observe_frame` (every frame) / `update` ->
`Narrator.on_events` / `on_insights` / `tick(t, stats, engine.deadball)` -> hub (`frame` at
`pipeline.frame_hz`, `stats` at `pipeline.stats_hz`, events / utterances / insights / captions at once).
The hub `stats` message is `pipeline.hub_stats_msg`: `GameStats.to_dict()` with only the newest
`pipeline.hub_decisions` decision grades, without per-option detail, plus `decisions_total` (the full list
grows ~5 KB per graded pass / shot; `stats.json` and `RunResult.stats` keep it).
The engine and stats see only fresh scoreboard reads; the last confirmed reading feeds the clock display,
the narrator's frame and the hub `frame` message. One `PseudonymMap` per game
(`narration.facts.pseudonyms_for(roster, seed)`), handed to `Narrator(pseudo=...)` wrapped in
`pipeline.TieredPseudonyms`: same tokens and state, but `render` gives every named player a confidence
(the cue event's own, else the latest perceived identity, else 0.0 on perception sources / 1.0 in sim
truth), so colour lines, Claude candidates and coach cards, which carry no events, also obey the 0.95
name tier. `director.*` settings are applied on top of the narrator's own director (`narration.cooldown_s`
/ `min_gap_s`, the accessibility FIFO policy) rather than replacing it (`pipeline.director_overrides`).
Operator writes (`GameController`): a corrected scorer on a make / miss also revises the shot's attempt
(the box score credits the attempt's shooter) and moves or retracts its assist / block; a new
SUBSTITUTION from `POST /api/events` is also a lineup entry; with no calibration, a file played in real
time holds (status `needs_calibration`, `paused`) until `POST /api/calibration`; `speed` is refused for a
live camera. `Insight.priority` is
higher = more important everywhere; `pipeline.cue_priority_for_insight` is the only mapping onto the
director's 0..3 levels. No Anthropic credentials => template-only narration; no `TYPESAFE_API_KEY` =>
Jev mode "off" (fallbacks). Claude's cached prefix uses a 1 h TTL and is pre-warmed at start and every
55 min (`llm.*`).

Implemented contracts (CLI, `courtvision <command>` / `python -m courtvision`):
```python
# cli.py: main(argv=None) -> int   (0 ok, 1 runtime failure, 2 usage / config error)
# every config-building command: --config FILE, --set section.key=value (repeatable); flags > --set > file
simulate  --seconds --seed --league --mode truth3d|mono|video --out --jev --json   # run_offline + eval summary
serve     --source sim|video|camera --mode --video --camera --speed --host 127.0.0.1 --port 8765
          --jev live|stub|off --no-llm --tts none|say|avspeech --auth-token --open [--calibration --roster
          --scoreboard-roi --detector --seconds --seed --league --insecure-no-auth]
run       --video --calibration --detector --roster --scoreboard-roi --start --duration --out --jev --json
calibrate POINTS | --from-sim  [--out --league --image-size W H --ransac-thresh --points-out --roi-out]
render    --seconds --seed --start --out (.mp4 = H.264, .avi/.mkv = lossless FFV1) [--no-sidecars]
eval      PRED(events.jsonl | run dir) [--truth --tol --kinds --no-player --json --out]
config    [--default | --section NAME]
latency   --configs C1..C4,J1..J5 --n --network --out --packet --pause --no-prewarm   # refuses without keys
create_serve_app(cfg, *, source=None, env=None, auth_token=None, jev=None, llm=None, tts=None, detector=None)
    -> ServeApp(app, game, hub)    # server.app.create_app + attach_game: LiveGame runs as a task on the
                                   # app's own loop for its lifespan; shutdown = request_stop(drain=False)
parse_detector("colorblob" | "rfdetr[:weights|nano|small|medium]" | "coreml:<mlpackage>" | "rnd-yolo:<weights>")
# roster.py: load_roster(path, home_name=None, away_name=None) -> Roster; parse_roster; save_roster
#   nested form teams: {home: {name, players: [{jersey, name, position, player_id?}]}, away: {...}}
#   (docs/examples/roster.yaml); the flat form {"teams": {side: name}, "players": [{team, ...}]} still loads;
#   jerseys are text ("0" != "00"); duplicate ids / (team, jersey) pairs are errors
```
Offline commands default Jev to "off" unless the file, `--set` or `--jev` picks a mode (source time runs
ahead of wall time). `serve` refuses a non-loopback `--host` without a token (`$COURTVISION_TOKEN` or
`--auth-token`) unless `--insecure-no-auth`; uvicorn's logs redact `token=` query values. It warns about
flags the chosen source never reads (`--roster` with the simulator, perception flags outside `--mode video`,
`--seconds` / `--seed` with a file or camera), and `gc.freeze()`s the objects that exist once the app is
built (a pre-simulated 48 min game is 2.3 M objects; each full collection stalled the shared event loop
~0.5 s). `render` writes
`<stem>.calibration.json`, `<stem>.roster.yaml`, `<stem>.scoreboard_roi.json` and `<stem>.truth_events.jsonl`
(re-timed to the file) next to the video. `--detector rnd-yolo:` prints the AGPL / R&D warning, then loads
`tools/rnd_detectors.py` by path (a source checkout only; `tools/` is not in the package), which is the one
bridge to `perception/rnd/`: shipped code never imports it (D9), so an installed wheel cannot reach AGPL code.
The detector is injected with `LiveGame(detector=...)` / `run_offline(detector=...)`.
