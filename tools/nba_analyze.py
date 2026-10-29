"""Analyse a whole NBA broadcast and score it against the official box score (research use).

    .venv/bin/python tools/nba_analyze.py runs/nba/<game> detect [--workers 2]
    .venv/bin/python tools/nba_analyze.py runs/nba/<game> ball
    .venv/bin/python tools/nba_analyze.py runs/nba/<game> events [--tag baseline] [--3d] [--start S --end S]
    .venv/bin/python tools/nba_analyze.py runs/nba/<game> eval --tag baseline
    .venv/bin/python tools/nba_analyze.py runs/nba/<game> fuse --tag baseline [--fusion-file calib.json] [--fusion '{}']
    .venv/bin/python tools/nba_analyze.py runs/nba/<game> timeline --tag baseline [--fit none] [--timeline '{}']

The game folder holds ``game.json`` (``video``, ``espn_event``, optional ``home_color_bgr`` / ``away_color_bgr``),
``scorebug.jsonl`` (tools/nba_scan_scorebug.py), ``camera_parts/*.jsonl`` (tools/nba_track_court.py) and
``espn_summary.json``. ``detect`` writes ``detections/*.npz`` (``--det-dir``: another folder, e.g. for new
weights; ``ball`` and ``events`` read the same one), ``ball`` writes ``ball_track.json``, ``events`` writes
``runs/<tag>/events.jsonl`` and ``eval`` writes ``runs/<tag>/report.json`` / ``report.md``. ``timeline`` decodes the
team with the ball over the run (``courtvision/nba/timeline.py``, no scorebug) and writes ``runs/<tag>/timeline.jsonl``
(the engine's events rebuilt from it; ``eval --file timeline.jsonl`` scores them) and ``timeline.json``; with the
play-by-play it is fitted on one half of the game and scored on the other, both ways.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from courtvision.nba import espn  # noqa: E402
from courtvision.nba.analyze import (  # noqa: E402
    BallTrack,
    CameraTrack,
    DetectionCache,
    PerceiveConfig,
    build_ball_track,
    detect_video,
    load_scorebug,
    perceive,
    run_engine,
)
from courtvision.nba.evaluate import compare  # noqa: E402
from courtvision.training.align_pbp import AlignConfig, ClockTimeline, align_pbp, score_changes  # noqa: E402

# ESPN's scorebug shows a basket about 2 s after the play-by-play clock time of the make (median over the made
# field goals of Lakers @ Warriors 2024-12-25); its play-by-play clock is the event's own time (no entry lag)
ESPN_ALIGN = AlignConfig(score_update_lag_s=2.0, lag_s={"nba": 0.0}, estimate_lag=False)
from courtvision.types import Event, Team  # noqa: E402


def load_game(root: Path) -> tuple[dict, CameraTrack]:
    game = json.loads((root / "game.json").read_text())
    camera = CameraTrack.load(sorted((root / "camera_parts").glob("part*.jsonl")))
    return game, camera


def cmd_detect(root: Path, args: argparse.Namespace) -> None:
    import multiprocessing as mp

    game, camera = load_game(root)
    out = root / args.det_dir
    out.mkdir(exist_ok=True)
    frames = sorted(camera.params)
    lo = max(frames[0], int(args.start * camera.fps)) if args.start else frames[0]
    hi = min(frames[-1] + 1, int(args.end * camera.fps)) if args.end else frames[-1] + 1
    n = max(args.workers, args.chunks)  # chunks already written are skipped, so an interrupted run resumes
    bounds = [lo + (hi - lo) * k // n for k in range(n + 1)]
    jobs = [(str(root), game["video"], args.weights, bounds[k], bounds[k + 1], str(out / f"{bounds[k]:07d}-{bounds[k + 1]:07d}.npz"))
            for k in range(n) if not (out / f"{bounds[k]:07d}-{bounds[k + 1]:07d}.npz").exists()]
    if args.workers == 1:
        for j in jobs:
            _detect_part(*j)
    else:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            pool.starmap(_detect_part, jobs)


def _detect_part(root: str, video: str, weights: str, start: int, end: int, out: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    _, camera = load_game(Path(root))
    t0 = time.time()
    cache = detect_video(Path(video), camera, Path(weights), start, end)
    cache.save(Path(out))
    print(f"{out}: {len(cache.balls)} frames in {time.time() - t0:.0f}s", flush=True)


def load_cache(root: Path, det_dir: str = "detections") -> DetectionCache:
    cache = DetectionCache()
    for p in sorted((root / det_dir).glob("*.npz")):
        cache.merge(DetectionCache.load(p))
    return cache


def cmd_ball(root: Path, args: argparse.Namespace) -> None:
    _, camera = load_game(root)
    t0 = time.time()
    track = build_ball_track(load_cache(root, args.det_dir), camera)
    track.save(root / "ball_track.json")
    print(f"ball path {len(track.path)} frames, {len(track.flights)} flights ({time.time() - t0:.0f}s)")


def cmd_events(root: Path, args: argparse.Namespace) -> None:
    game, camera = load_game(root)
    cache = load_cache(root, args.det_dir)
    ball = BallTrack.load(root / "ball_track.json")
    reads = load_scorebug(root / "scorebug.jsonl")
    cfg = PerceiveConfig(use_3d_flights=args.use_3d, home_color_bgr=tuple(game["home_color_bgr"]) if game.get("home_color_bgr") else None,
                         away_color_bgr=tuple(game["away_color_bgr"]) if game.get("away_color_bgr") else None)
    start = int(args.start * camera.fps) if args.start else 0
    end = int(args.end * camera.fps) if args.end else None
    t0 = time.time()
    frames = perceive(Path(game["video"]), camera, cache, ball, reads, cfg, start, end)
    events = run_engine(frames, config=json.loads(args.engine) if args.engine else None)
    out = root / "runs" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    with (out / "events.jsonl").open("w") as fh:
        for e in events:
            fh.write(json.dumps(e.to_dict()) + "\n")
    (out / "run.json").write_text(json.dumps({"start": args.start, "end": args.end, "use_3d": args.use_3d,
                                              "engine": args.engine, "seconds": time.time() - t0}))
    print(f"{len(events)} events in {time.time() - t0:.0f}s -> {out}")


def truth(root: Path):  # noqa: ANN201
    summary = espn.load_summary(root / "espn_summary.json")
    reads = load_scorebug(root / "scorebug.jsonl")
    tl = ClockTimeline.fit(reads)
    return summary, align_pbp(espn.pbp_actions(summary), tl, score_changes=score_changes(reads), config=ESPN_ALIGN)


def cmd_fuse(root: Path, args: argparse.Namespace) -> None:
    from courtvision.nba.fusion import PlayDirection, fuse
    from courtvision.nba.fusion_tune import fusion_config
    from courtvision.nba.scorebug_events import scorebug_events

    _, camera = load_game(root)
    out = root / "runs" / args.tag
    run = json.loads((out / "run.json").read_text())
    lo, hi = float(run.get("start") or 0.0), float(run.get("end") or 1e9)
    reads = load_scorebug(root / "scorebug.jsonl")
    bug = [b for b in scorebug_events(reads) if lo <= b.t <= hi]
    periods = [(r.t, r.period) for r in reads if r.period is not None]
    events = [Event.from_dict(json.loads(line)) for line in (out / "events.jsonl").open()]
    overrides: dict = {}
    if args.fusion_file:  # a calibration (tools/nba_calibrate_fusion.py --out): its "config" object
        doc = json.loads(Path(args.fusion_file).read_text())
        if not isinstance(doc, dict) or not isinstance(doc.get("config"), dict):
            raise SystemExit(f'{args.fusion_file}: no "config" object (FusionConfig overrides)')
        overrides.update(doc["config"])
    overrides.update(json.loads(args.fusion) if args.fusion else {})  # --fusion goes on top of the file
    cfg = fusion_config(overrides)
    if overrides:
        print(f"FusionConfig overrides: {json.dumps(overrides)}")
    flights = BallTrack.load(root / "ball_track.json").flights if (root / "ball_track.json").exists() else []
    from courtvision.geometry.court import court_for

    fused = fuse(events, bug, cfg, PlayDirection.from_camera(camera), periods, flights, court_for("nba"))
    with (out / "fused.jsonl").open("w") as fh:
        for e in fused:
            fh.write(json.dumps(e.to_dict()) + "\n")
    print(f"{len(events)} engine events -> {len(fused)} fused events")


def cmd_timeline(root: Path, args: argparse.Namespace) -> None:
    import numpy as np

    from courtvision.geometry.court import court_for
    from courtvision.nba import timeline as tlm
    from courtvision.nba.fusion import PlayDirection

    game, camera = load_game(root)
    out = root / "runs" / args.tag
    events = [Event.from_dict(json.loads(line)) for line in (out / "events.jsonl").open()]
    ball = BallTrack.load(root / "ball_track.json") if (root / "ball_track.json").exists() else None
    flights = ball.flights if ball is not None else []
    court = court_for("nba")
    direction = PlayDirection.from_camera(camera)
    base = tlm.TimelineConfig(**(json.loads(args.timeline) if args.timeline else {}))
    grid = tlm.grid_for(events, direction, base.dt_s)
    attack = tlm.attack_map(events, direction, margin_ft=base.margin_ft)
    report: dict = {"attack": attack.to_dict()}
    split = attack.switch_t if np.isfinite(attack.switch_t) else float(grid[len(grid) // 2])
    aligned = truth(root)[1] if "cross" in (args.fit, args.shots) else None
    if args.fit == "cross":
        marks = tlm.truth_marks(aligned.aligned)
        clock = aligned.timeline
        live = np.array([bool(cp is not None and cp.running) for cp in (clock.clock_at(float(t)) for t in grid)])
        first = grid < split
        state, phase = np.zeros(len(grid), np.int8), np.zeros(len(grid), np.int8)
        folds = []
        for name, test in (("first half", first), ("second half", ~first)):
            cfg = tlm.fit(events, direction, marks, grid, ~test, live, base, flights, court, attack=attack)
            tl = tlm.decode(events, direction, cfg, flights, court, attack, grid)
            state[test], phase[test] = tl.state[test], tl.phase[test]  # type: ignore[index]
            folds.append({"scored_on": name, "fitted_on": "the other half", "config": cfg.to_dict(),
                          "timeline": tlm.score(tl.teams(), grid, marks, test & live)})
        timeline = tlm.Timeline(grid, state, attack, tl.tracked, phase)
        gains = sorted((e for e in events if e.kind.value == "possession" and e.team is not None), key=lambda e: e.t)
        gt = np.array([e.t for e in gains])
        engine = [gains[j].team if j >= 0 else None for j in np.searchsorted(gt, grid, side="right") - 1]
        report.update(folds=folds, engine=tlm.score(engine, grid, marks, live),
                      timeline=tlm.score(timeline.teams(), grid, marks, live))
    else:
        timeline = tlm.decode(events, direction, base, flights, court, attack, grid)
        report["config"] = base.to_dict()
    if args.shots != "off":
        events = decide_shots(events, timeline, attack, aligned, split, args, report, game, camera, court, ball, out)
    derived = tlm.derive_events(events, timeline, tlm.DeriveConfig(**(json.loads(args.derive) if args.derive else {})),
                                flights, court)
    with (out / "timeline.jsonl").open("w") as fh:
        for e in derived:
            fh.write(json.dumps(e.to_dict()) + "\n")
    doc = {**report, **timeline.to_dict()}
    if aligned is not None:  # the play-by-play beside the timeline, to watch it by (tools/nba_overlay.py, the Studio)
        doc["reference"] = tlm.reference(aligned.aligned)
    (out / "timeline.json").write_text(json.dumps(doc, indent=1))
    for key in ("engine", "timeline"):
        if key in report:
            print(f"{key:9} team in settled live play {report[key]['accuracy']:.3f} "
                  f"({report[key]['settled_steps']} steps); changes of possession: {report[key]['changes_pred']} found, "
                  f"{report[key]['changes_true']} true, precision {report[key]['precision']:.2f}, "
                  f"recall {report[key]['recall']:.2f}")
    if "shots" in report:
        s = report["shots"]
        print(f"shots     {s['candidates']} candidates ({s['from_rim_visits']} rim visits with no engine result): "
              f"{s['kept']} shots kept ({s['added']} added at rim visits), {s['free_throws']} free throws, "
              f"{s['made']} made" + (f"; on held-out halves: real or not {s['real_accuracy']:.3f}, free throw or field "
                                     f"goal {s['ft_accuracy']:.3f}, made or missed {s['made_accuracy']:.3f}"
                                     if "made_accuracy" in s else "")
              + (f"; Jev asked about {s['jev_answers']}" if "jev_answers" in s else ""))
    print(f"{len(events)} events -> {len(derived)} timeline events -> {out / 'timeline.jsonl'}")


def decide_shots(events: list[Event], timeline, attack, aligned, split: float, args: argparse.Namespace,  # noqa: ANN001
                 report: dict, game: dict, camera: CameraTrack, court, ball, out: Path) -> list[Event]:  # noqa: ANN001
    """Every shot decided from the video (``courtvision/nba/makes.py``): the engine's shot results and the ball's rim
    visits it reported nothing at. With ``--shots cross`` the models are fitted on one half of the game and decide the
    other, both ways (and once on the whole game, written to ``shot_models.json`` for games without play-by-play);
    ``--shots none`` reads ``--shot-models``. ``--jev`` adds Jev's read of each candidate's rim crops (asked once,
    kept in ``jev_rim.jsonl``)."""
    from collections import Counter

    from courtvision.nba import makes as mk

    loaded = None
    if args.shots != "cross":
        if not args.shot_models:
            raise SystemExit("--shots none needs --shot-models (the shot_models.json of a run fitted with --shots cross)")
        loaded = mk.ShotModels.from_dict(json.loads(Path(args.shot_models).read_text()))
        if loaded.uses_jev != args.jev:
            raise SystemExit(f"--shot-models were fitted with{'' if loaded.uses_jev else 'out'} Jev's answers: run "
                             f"with{'' if loaded.uses_jev else 'out'} --jev")
    judge = rim_judge(game, camera, court, ball, out) if args.jev else None

    def jev_read(c: mk.Candidate) -> dict | None:  # the visit's rim, else the one the shooting team attacks
        assert judge is not None
        shooter = timeline.team_at(c.release_t - 0.25)
        side = c.side or (attack.side(shooter, c.t) if shooter is not None else None)
        return judge.judge(c.key, c.t, side)

    visits = mk.rim_visits(ball.path, camera, court) if ball is not None else []
    rows = mk.candidate_rows(events, timeline, visits, jev_read if judge is not None else None)
    shots: dict = {"candidates": len(rows), "from_rim_visits": sum(c.result is None for c, _ in rows),
                   "jev": judge is not None}
    if args.shots == "cross":
        truth_shots = [(float(al.t), bool(al.action.is_free_throw), bool(al.action.made)) for al in aligned.aligned
                       if al.t is not None and al.action.is_shot and not al.action.synthetic and al.action.made is not None]
        labels = mk.label_candidates([c for c, _ in rows], truth_shots)
        fits = {}
        right = Counter()
        n_real = 0
        for in_first in (True, False):
            train = [(c, f) for c, f in rows if (c.t < split) != in_first]
            fits[in_first] = mk.fit_shot_models([f for _, f in train], [labels[c.key] for c, _ in train],
                                                jev=judge is not None)
            m = fits[in_first]
            for c, f in ((c, f) for c, f in rows if (c.t < split) == in_first):
                lab = labels[c.key]
                right["real"] += (m.real.p(f) >= 0.5) == (lab != "false")
                if lab != "false":
                    n_real += 1
                    ft = m.ft.p(f) >= 0.5
                    right["ft"] += ft == lab.startswith("ft")
                    right["made"] += (m.made.p({**f, "is_ft": float(ft)}) >= 0.5) == lab.endswith("made")
        whole = mk.fit_shot_models([f for _, f in rows], [labels[c.key] for c, _ in rows], jev=judge is not None)
        (out / "shot_models.json").write_text(json.dumps(whole.to_dict(), indent=1))
        shots.update(labels=dict(Counter(labels.values())), real_accuracy=round(right["real"] / max(1, len(rows)), 4),
                     ft_accuracy=round(right["ft"] / max(1, n_real), 4),
                     made_accuracy=round(right["made"] / max(1, n_real), 4),
                     models={"first half": fits[True].to_dict(), "second half": fits[False].to_dict()})

        def models(c: mk.Candidate) -> mk.ShotModels:  # fitted on the other half
            return fits[c.t < split]
    else:
        models = loaded  # type: ignore[assignment]
    decided = mk.decide(events, timeline, models, visits, jev=jev_read if judge is not None else None)
    kept = [e for e in decided if e.kind.value in ("shot_made", "shot_missed") and e.attrs.get("makes")]
    shots.update(kept=len(kept), made=sum(e.kind.value == "shot_made" for e in kept),
                 free_throws=sum(e.value == 1 for e in kept),
                 added=sum(e.attrs.get("inferred") == "rim_visit" for e in kept))
    if judge is not None:
        shots["jev_answers"] = sum(1 for c, _ in rows if judge.judge(c.key, c.t) is not None)
        if judge.stopped:
            shots["jev_stopped"] = judge.stopped
    report["shots"] = shots
    return decided


def rim_judge(game: dict, camera: CameraTrack, court, ball, out: Path):  # noqa: ANN001, ANN201
    """Jev on rim crops (``courtvision/nba/rim_jev.py``), with the key from .env (Set up Jev.command)."""
    from courtvision.credentials import runtime_env
    from courtvision.decision.client import JevClient, JevConfig
    from courtvision.nba.rim_jev import RimJudge

    client = JevClient(JevConfig(mode="live", images_enabled=True), api_key=runtime_env().get("TYPESAFE_API_KEY"))
    path = {f: (p[0], p[1]) for f, p in ball.path.items()} if ball is not None else {}
    return RimJudge(game["video"], camera, court, path, client, cache=out / "jev_rim.jsonl")


def nba_checks(root: Path, events: list[Event], window: tuple[float, float] | None) -> dict:
    """Extra checks against NBA.com data when the game folder has it: shot locations (``nba_pbp.csv``) and
    passes per team (``nba_tracking.json``, Second Spectrum)."""
    from courtvision.eval.events import latest_revisions
    from courtvision.nba.nba_stats import load_shots, load_tracking, match_shot_locations
    from courtvision.types import EventKind, EventStatus

    out: dict = {}
    evs = [e for e in latest_revisions(events) if e.status is not EventStatus.RETRACTED]
    lo, hi = window if window is not None else (0.0, 1e9)
    if (root / "nba_pbp.csv").exists():
        tl = ClockTimeline.fit(load_scorebug(root / "scorebug.jsonl"))
        placed = []
        for s in load_shots(root / "nba_pbp.csv"):
            t = tl.video_t(s.period, s.clock)
            if t is not None and lo <= t <= hi:
                placed.append((t, s))
        att = [e for e in evs if e.kind is EventKind.SHOT_ATTEMPT and e.value in (2, 3)]  # the release point
        out["shot_locations"] = match_shot_locations(att, placed, tol_s=3.0)
    if (root / "nba_tracking.json").exists() and window is None:
        trk = load_tracking(root / "nba_tracking.json")
        passes = {t.value: sum(1 for e in evs if e.kind is EventKind.PASS and e.team is t) for t in (Team.HOME, Team.AWAY)}
        out["passes"] = {"pred": passes, "nba": {"home": trk["home"]["totals"]["passes"],
                                                 "away": trk["away"]["totals"]["passes"]}}
    return out


def cmd_eval(root: Path, args: argparse.Namespace) -> None:
    summary, aligned = truth(root)
    out = root / "runs" / args.tag
    run = json.loads((out / "run.json").read_text())
    events = [Event.from_dict(json.loads(line)) for line in (out / args.file).open()]
    window = None
    if run.get("start") or run.get("end"):
        window = (float(run.get("start") or 0.0), float(run.get("end") or 1e9))
    rep = compare(events, aligned.aligned, espn.box_score(summary), tol_s=args.tol, window=window)
    rep.update(nba_checks(root, events, window))
    stem = Path(args.file).stem
    (out / f"report_{stem}.json").write_text(json.dumps(rep, indent=1))
    abbr = {v["side"]: v["abbr"] for v in espn.teams(summary).values()}
    lines = [f"# {args.tag} ({stem})", ""]
    if "box" in rep:
        lines += [f"Box-score error (sum |pred - truth| / sum truth): **{rep['box']['error']:.3f}**, "
                  f"exact cells {rep['box']['exact_share']:.0%}", ""]
        lines.append("| stat | " + " | ".join(f"{abbr[t]} pred | {abbr[t]} truth" for t in (Team.AWAY, Team.HOME)) + " |")
        lines.append("|---|---:|---:|---:|---:|")
        by = {(r["team"], r["stat"]): r for r in rep["box"]["rows"]}
        for s in ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov"):
            lines.append(f"| {s} | " + " | ".join(f"{by[(t.value, s)]['pred']} | {by[(t.value, s)]['truth']}"
                                                for t in (Team.AWAY, Team.HOME)) + " |")
    else:
        lines.append(f"Window {window}: " + json.dumps(rep["box_window"]))
    if "shot_locations" in rep:
        lines.append("")
        lines.append(f"Shot locations vs NBA.com: {json.dumps(rep['shot_locations'])}")
    if "passes" in rep:
        lines.append(f"Passes vs NBA tracking: {json.dumps(rep['passes'])}")
    lines += ["", "| event | F1 | precision | recall | pred | truth |", "|---|---:|---:|---:|---:|---:|"]
    for k, r in rep["events"].items():
        f = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        lines.append(f"| {k} | {f(r['f1'])} | {f(r['precision'])} | {f(r['recall'])} | {r['n_pred']} | {r['n_truth']} |")
    (out / f"report_{stem}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("cmd", choices=("detect", "ball", "events", "fuse", "timeline", "eval"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=1, help="detect: split the range into this many files (resumable)")
    ap.add_argument("--det-dir", default="detections", help="detect / ball / events: detection folder in the game folder")
    ap.add_argument("--weights", default="runs/models/student/best.pt")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--3d", dest="use_3d", action="store_true")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    ap.add_argument("--engine", default=None, help="EventEngineConfig overrides (JSON)")
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--file", default="events.jsonl", help="eval: events file of the run (events.jsonl / fused.jsonl)")
    ap.add_argument("--fusion", default=None, help="fuse: FusionConfig overrides (JSON)")
    ap.add_argument("--fusion-file", default=None,
                    help='fuse: a JSON file whose "config" object holds FusionConfig overrides (the output of '
                         "tools/nba_calibrate_fusion.py); --fusion overrides go on top")
    ap.add_argument("--fit", choices=("cross", "none"), default="cross",
                    help="timeline: cross = fit on each half of the game and decode the other with it (needs the "
                         "play-by-play); none = decode with the TimelineConfig defaults and --timeline overrides")
    ap.add_argument("--timeline", default=None, help="timeline: TimelineConfig overrides (JSON)")
    ap.add_argument("--derive", default=None, help="timeline: DeriveConfig overrides (JSON)")
    ap.add_argument("--shots", choices=("cross", "none", "off"), default=None,
                    help="timeline: decide each field-goal result from the video: cross = models fitted on each half "
                         "(needs the play-by-play), none = the models of --shot-models, off = keep the engine's labels "
                         "(they lean on the scorebug); default: cross with --fit cross, else none")
    ap.add_argument("--shot-models", default=None, help="timeline: shot_models.json of a run fitted with --shots cross")
    ap.add_argument("--jev", action="store_true",
                    help="timeline: add Jev's read of each shot's rim crops (frames go to TypeSafe's API; answers kept in "
                         "runs/<tag>/jev_rim.jsonl)")
    args = ap.parse_args()
    if args.shots is None:
        args.shots = "cross" if args.fit == "cross" else "none"
    logging.basicConfig(level=logging.WARNING)
    root = Path(args.root)
    {"detect": cmd_detect, "ball": cmd_ball, "events": cmd_events, "fuse": cmd_fuse, "timeline": cmd_timeline,
     "eval": cmd_eval}[args.cmd](root, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
