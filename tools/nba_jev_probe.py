"""Can Jev see the game? TypeSafe Jev's image answers on real broadcast moments, scored against the play-by-play.

    .venv/bin/python tools/nba_jev_probe.py runs/nba/<game> makes --n 40 [--tag full] [--dry-run]
    .venv/bin/python tools/nba_jev_probe.py runs/nba/<game> possession --n 40 --jerseys white,blue
    .venv/bin/python tools/nba_jev_probe.py runs/nba/<game> turnovers --jerseys white,blue

Research use: the frames go to TypeSafe's API, and Jev's image input (``jev-1.14.0``) is pre-production
(docs/research/jev_api.md §5). ``makes`` and ``possession`` are a seeded random sample of field goals from the
aligned play-by-play (``nba_analyze.truth``), half of each class; ``turnovers`` takes every turnover of the engine
run. One request per moment:

* ``makes``: ten crops around the rim the ball reaches, in time order (0.1 s apart near the middle), centred on the
  frame where the linked ball (``ball_track.json``) passes nearest the rim, or on the play-by-play time when the
  ball is not seen there (``courtvision/nba/rim_jev.py``). Jev answers made / missed / unclear, asked twice with the
  options reversed and averaged (jev_integration_design.md §1.5), and for each crop whether the ball is in the net. Truth: the play-by-play
  result. The engine's answer: its field-goal result nearest in time (within ``--tol`` s).
* ``possession``: the full frame ``--lead`` s before the play-by-play time of the shot, when no other action happened
  in the ``--quiet`` s before that moment (the shooting team has had the ball since). Jev answers which team's
  player holds the ball and which team has possession (both order-ensembled). Truth: the shooting team.
  ``--jerseys AWAY,HOME`` names the jersey colours Jev is told about. The engine's answer: the team of its last
  possession gain at or before the moment.
* ``turnovers``: each turnover the engine reports (a steal's turnover included), with the full frames ``--offsets``
  s before and after it (one offset: 3 s when the camera is not on the court there; several: those on the court).
  Jev answers which team has possession in each frame (order-ensembled; a side's answer is the mean over its
  frames); different teams confirm the change, the same team vetoes it. Truth: a play-by-play turnover within
  ``--tol`` s, or any change of possession there (a turnover, a defensive rebound or a made basket; a made free
  throw only when it is the last of its trip).

Writes ``<game>/jev_probe/<task>.jsonl`` (one row per moment: truth, the engine, every probability, latency, tokens)
and ``<task>.summary.json``; the images stay in ``<game>/jev_probe/<task>/``. ``--dry-run`` writes the images only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import statistics
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nba_analyze import load_game, truth  # noqa: E402

from courtvision.credentials import runtime_env  # noqa: E402
from courtvision.decision.client import JevClient, JevConfig, _retry  # noqa: E402
from courtvision.eval.events import latest_revisions  # noqa: E402
from courtvision.geometry.court import court_for  # noqa: E402
from courtvision.nba.analyze import CameraTrack  # noqa: E402
from courtvision.nba.rim_jev import (  # noqa: E402
    FATAL,
    Frames,
    choice,
    decode_make,
    ensemble,
    jpeg,
    make_request,
    pick,
    reversed_options,
    rim_centre,
    rim_crops,
)
from courtvision.training.align_pbp import AlignedAction  # noqa: E402
from courtvision.types import Event, EventKind, EventStatus, Team  # noqa: E402

PRICE_PER_TOKEN = 0.042 / 1e6  # US$ per input token, jev-1.13 and 1.14 alike; output is free (jev_api.md §1)
FRAME_PX = 1568  # longest side of a full frame sent to Jev


@dataclass
class Moment:
    """One sampled field goal and what the probe sends and compares for it."""

    id: str
    t: float  # play-by-play video time of the shot
    team: Team
    made: bool
    description: str
    frames: list[int] = field(default_factory=list)  # the frames sent, in order
    centred_on: str = ""  # makes: "ball" (the linked ball's pass by the rim) or "pbp" (the play-by-play time)
    side: str = ""  # makes: the rim, "left" / "right"


# --------------------------------------------------------------------------- moments


def field_goals(aligned: Sequence[AlignedAction]) -> list[AlignedAction]:
    return [al for al in aligned if al.t is not None and al.action.is_field_goal and not al.action.synthetic
            and al.action.team is not None and al.action.made is not None]


def balanced(items: Sequence[AlignedAction], key: Any, n: int, seed: int) -> list[AlignedAction]:
    """Every item, shuffled with ``seed`` and interleaved by ``key`` so that any prefix is balanced; the caller
    takes the first ``n`` usable ones."""
    rng = random.Random(seed)
    groups: dict[Any, list[AlignedAction]] = {}
    for al in items:
        groups.setdefault(key(al), []).append(al)
    for g in groups.values():
        rng.shuffle(g)
    out: list[AlignedAction] = []
    for i in range(max((len(g) for g in groups.values()), default=0)):
        out += [g[i] for _, g in sorted(groups.items(), key=lambda kv: str(kv[0])) if i < len(g)]
    return out


def make_moment(al: AlignedAction, camera: CameraTrack, ball: Mapping[int, tuple[float, float]],
                search_s: float = 2.5) -> Moment | None:
    """The rim and the centre frame of a shot's crops (``rim_centre``), or None when no rim is in view around it."""
    found = rim_centre(float(al.t), camera, court_for("nba"), ball, search_s)  # type: ignore[arg-type]
    if found is None:
        return None
    centre, side, how = found
    a = al.action
    return Moment(f"fg{a.action_number}", float(al.t), a.team, bool(a.made), a.description,  # type: ignore[arg-type]
                  [centre], "pbp" if how == "time" else how, side)


def possession_moments(aligned: Sequence[AlignedAction], camera: CameraTrack, lead_s: float,
                       quiet_s: float) -> Iterator[tuple[AlignedAction, Moment]]:
    """Field goals with a moment ``lead_s`` before them in live play (the camera tracked) and no other placed action
    in the ``quiet_s`` before that moment or between it and the shot."""
    others = sorted(al.t for al in aligned if al.t is not None and al.action.kind is not None
                    and not al.action.synthetic)
    for al in field_goals(aligned):
        m = al.t - lead_s  # type: ignore[operator]
        if any(m - quiet_s < t < al.t - 0.5 for t in others):  # type: ignore[operator]
            continue
        f = camera.frame_at(m)
        if camera.get(f) is None:
            continue
        a = al.action
        yield al, Moment(f"fg{a.action_number}", float(al.t), a.team,  # type: ignore[arg-type]
                         bool(a.made), a.description, [f])


# --------------------------------------------------------------------------- the engine's answers


def engine_events(path: Path) -> list[Event]:
    events = [Event.from_dict(json.loads(line)) for line in path.open()]
    return [e for e in latest_revisions(events) if e.status is not EventStatus.RETRACTED]


def engine_result(events: Sequence[Event], t: float, tol: float) -> str | None:
    """The engine's field-goal result nearest ``t`` (within ``tol``): "made", "missed" or None (none found)."""
    near = [e for e in events if e.kind in (EventKind.SHOT_MADE, EventKind.SHOT_MISSED) and e.value in (2, 3)
            and abs(e.t - t) <= tol]
    if not near:
        return None
    e = min(near, key=lambda e: abs(e.t - t))
    return "made" if e.kind is EventKind.SHOT_MADE else "missed"


def engine_team(events: Sequence[Event], t: float) -> Team | None:
    """The team of the engine's last possession gain at or before ``t``."""
    gains = [e for e in events if e.kind is EventKind.POSSESSION and e.t <= t and e.team is not None]
    return max(gains, key=lambda e: e.t).team if gains else None


def changes_possession(al: AlignedAction) -> bool:
    """A play-by-play action after which the other team has the ball (or the rebounder's team, after a miss)."""
    a = al.action
    if a.kind in (EventKind.TURNOVER, EventKind.STEAL):
        return True
    if a.kind is EventKind.REBOUND:
        return not a.attrs.get("offensive")
    if a.kind is EventKind.SHOT_MADE:
        return not a.is_free_throw or int(a.attrs.get("ft_index") or 1) >= int(a.attrs.get("ft_total") or 1)
    return False


def turnover_moments(events: Sequence[Event], aligned: Sequence[AlignedAction], camera: CameraTrack,
                     offsets_s: Sequence[float], tol: float) -> Iterator[tuple[None, Moment, dict[str, Any]]]:
    """Every engine turnover with tracked frames at ``offsets_s`` before and after it (one offset: 3 s when the camera
    is not on the court there), and its truth. ``Moment.frames``: the frames before (farthest first), then after;
    the row's ``n_before`` says where they split."""
    placed = [al for al in aligned if al.t is not None and not al.action.synthetic and al.action.kind is not None]
    tries = [list(offsets_s)] if len(offsets_s) > 1 else [[g] for g in dict.fromkeys((offsets_s[0], 3.0))]
    for e in sorted((e for e in events if e.kind is EventKind.TURNOVER and e.team is not None), key=lambda e: e.t):
        sides = None
        for offs in tries:
            before = [f for f in (camera.frame_at(e.t - g) for g in sorted(offs, reverse=True)) if camera.get(f)]
            after = [f for f in (camera.frame_at(e.t + g) for g in sorted(offs)) if camera.get(f)]
            if before and after:
                sides = before, after
                break
        if sides is None:
            continue
        near = [al for al in placed if abs(al.t - e.t) <= tol]  # type: ignore[operator]
        truth = {"pbp_turnover": any(al.action.kind in (EventKind.TURNOVER, EventKind.STEAL) for al in near),
                 "pbp_change": any(changes_possession(al) for al in near),
                 "pbp_near": [al.action.description for al in near][:4], "n_before": len(sides[0])}
        reason = str(e.attrs.get("reason", ""))
        yield None, Moment(e.event_id, e.t, e.team, False, reason, [*sides[0], *sides[1]]), truth


# --------------------------------------------------------------------------- requests


def possession_options(jerseys: Mapping[Team, str]) -> tuple[dict[str, str], dict[str, str]]:
    a, h = jerseys[Team.AWAY], jerseys[Team.HOME]
    hold = {a: f"A player in a {a} jersey is holding or dribbling the ball.",
            h: f"A player in a {h} jersey is holding or dribbling the ball.",
            "nobody": "No player holds the ball: it is in the air, loose on the floor, or not visible."}
    team = {a: f"The team in {a} jerseys has the ball and is attacking.",
            h: f"The team in {h} jerseys has the ball and is attacking.",
            "unclear": "The image does not show which team has the ball."}
    return hold, team


def possession_request(frame: bytes, jerseys: Mapping[Team, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    from typesafe_sdk import Image

    hold, team = possession_options(jerseys)
    ih = "`frame` is a basketball broadcast image. Which player is holding the basketball?"
    it = "`frame` is a basketball broadcast image. Which team has possession of the ball?"
    q = {"holder": choice(ih, hold), "holder_rev": choice(ih, reversed_options(hold)),
         "team": choice(it, team), "team_rev": choice(it, reversed_options(team)),
         "ball_visible": {"type": "noul", "instructions": "Is the basketball visible in `frame`?"}}
    return {"frame": Image(frame)}, q


def turnover_request(before: Sequence[bytes], after: Sequence[bytes], jerseys: Mapping[Team, str]
                     ) -> tuple[dict[str, Any], dict[str, Any]]:
    """The possession question about each frame of one play: ``before_<i>`` and ``after_<i>`` (i from 1, nearest the
    turnover last before it and first after it)."""
    from typesafe_sdk import Image

    _, team = possession_options(jerseys)
    state: dict[str, Any] = {}
    q: dict[str, Any] = {}
    for side, imgs in (("before", before), ("after", after)):
        for i, img in enumerate(imgs, 1):
            key = f"{side}_{i}"
            state[key] = Image(img)
            it = f"`{key}` is a basketball broadcast image. Which team has possession of the ball in `{key}`?"
            q[f"team_{key}"], q[f"team_{key}_rev"] = choice(it, team), choice(it, reversed_options(team))
    return state, q


def probs_of(resp: Any, name: str) -> dict[str, float]:
    return {k: float(v) for k, v in resp.choices[name].probabilities.items()}


# --------------------------------------------------------------------------- scoring


def score_makes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Jev's two read-outs (the order-ensembled Choice, and any crop's in-net Noul >= 0.5) and the engine on the
    same shots; ``accuracy`` is over the shots answered (not unclear / not found), ``coverage`` their share."""
    def tally(preds: Sequence[str | None], truths: Sequence[str]) -> dict[str, Any]:
        ans = [(p, t) for p, t in zip(preds, truths, strict=True) if p in ("made", "missed")]
        right = sum(p == t for p, t in ans)
        per = {c: {"recall": round(sum(p == t == c for p, t in ans) / max(1, sum(t == c for t in truths)), 3)}
               for c in ("made", "missed")}
        return {"answered": len(ans), "right": right, "accuracy": round(right / max(1, len(ans)), 3),
                "coverage": round(len(ans) / max(1, len(truths)), 3), "per_class": per,
                "confusion": dict(Counter(f"{t}->{p}" for p, t in zip(preds, truths, strict=True)))}

    ok = [r for r in rows if r.get("jev")]
    truths = ["made" if r["made"] else "missed" for r in ok]
    out: dict[str, Any] = {
        "moments": len(rows), "answered_requests": len(ok),
        "jev_choice": tally([r["jev"]["outcome"] for r in ok], truths),
        "jev_in_net": tally([r["jev"]["in_net_pick"] for r in ok], truths),
        "engine": tally([r["engine"] for r in ok], truths),
        "by_centre": {c: tally([r["jev"]["outcome"] for r in ok if r["centred_on"] == c],
                               ["made" if r["made"] else "missed" for r in ok if r["centred_on"] == c])
                      for c in ("ball", "pbp")},
    }
    both = [(r["jev"]["outcome"], r["engine"], t) for r, t in zip(ok, truths, strict=True)
            if r["jev"]["outcome"] in ("made", "missed") and r["engine"] in ("made", "missed")]
    agree = [(j, t) for j, e, t in both if j == e]
    out["jev_and_engine"] = {"both_answered": len(both), "agree": len(agree),
                             "accuracy_when_agree": round(sum(j == t for j, t in agree) / max(1, len(agree)), 3),
                             "jev_right_when_disagree": sum(j == t for j, e, t in both if j != e),
                             "engine_right_when_disagree": sum(e == t for j, e, t in both if j != e)}
    out["calibration"] = calibration([(r["jev"]["p"], r["jev"]["outcome"] == t) for r, t in zip(ok, truths, strict=True)
                                      if r["jev"]["outcome"] in ("made", "missed")])
    return out


def score_possession(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def tally(preds: Sequence[str | None], truths: Sequence[str]) -> dict[str, Any]:
        ans = [(p, t) for p, t in zip(preds, truths, strict=True) if p in (Team.HOME.value, Team.AWAY.value)]
        right = sum(p == t for p, t in ans)
        return {"answered": len(ans), "right": right, "accuracy": round(right / max(1, len(ans)), 3),
                "coverage": round(len(ans) / max(1, len(truths)), 3)}

    ok = [r for r in rows if r.get("jev")]
    truths = [r["team"] for r in ok]
    return {"moments": len(rows), "answered_requests": len(ok),
            "jev_holder": tally([r["jev"]["holder"] for r in ok], truths),
            "jev_team": tally([r["jev"]["team"] for r in ok], truths),
            "engine": tally([r["engine"] for r in ok], truths),
            "ball_visible_mean": round(statistics.fmean(r["jev"]["ball_visible"] for r in ok), 3) if ok else None,
            "calibration_team": calibration([(r["jev"]["p_team"], r["jev"]["team"] == r["team"]) for r in ok
                                             if r["jev"]["team"] in (Team.HOME.value, Team.AWAY.value)])}


def score_turnovers(rows: Sequence[Mapping[str, Any]], confident: float = 0.85) -> dict[str, Any]:
    """Each engine turnover by its truth (a play-by-play turnover, another change of possession, or none) and Jev's
    verdict: "changed" (two different teams), "same" (one team: a veto), "unsure" (an unclear answer, or below
    ``confident`` in the confident view)."""
    def verdict(jv: Mapping[str, Any], floor: float) -> str:
        teams = (Team.HOME.value, Team.AWAY.value)
        if jv["before"] not in teams or jv["after"] not in teams or min(jv["p_before"], jv["p_after"]) < floor:
            return "unsure"
        return "same" if jv["before"] == jv["after"] else "changed"

    ok = [r for r in rows if r.get("jev")]
    truth = ["turnover" if r["pbp_turnover"] else "other_change" if r["pbp_change"] else "no_change" for r in ok]
    out: dict[str, Any] = {"moments": len(rows), "answered_requests": len(ok), "truth": dict(Counter(truth))}
    for name, floor in (("any", 0.0), (f"p>={confident}", confident)):
        table = Counter(f"{t}:{verdict(r['jev'], floor)}" for r, t in zip(ok, truth, strict=True))
        out[f"verdicts_{name}"] = {t: {v: table[f"{t}:{v}"] for v in ("changed", "same", "unsure")}
                                   for t in ("turnover", "other_change", "no_change")}
    return out


def calibration(pairs: Sequence[tuple[float, bool]], edges: Sequence[float] = (0.5, 0.7, 0.85, 0.95, 1.01)
                ) -> list[dict[str, Any]]:
    """Accuracy per band of the answer's own (ensembled) probability: does a confident Jev answer mean more?"""
    out = []
    for lo, hi in zip((0.0, *edges[:-1]), edges, strict=True):
        band = [ok for p, ok in pairs if lo <= p < hi]
        if band:
            out.append({"p": f"{lo:.2f}-{min(hi, 1.0):.2f}", "n": len(band),
                        "accuracy": round(sum(band) / len(band), 3)})
    return out


def usage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ms = [r["latency_ms"] for r in rows if r.get("jev")]
    tokens = sum(r.get("input_tokens") or 0 for r in rows)
    return {"requests": len(ms), "latency_ms_p50": round(statistics.median(ms)) if ms else None,
            "latency_ms_max": round(max(ms)) if ms else None, "input_tokens": tokens,
            "tokens_per_request": round(tokens / max(1, len(ms))), "cost_usd": round(tokens * PRICE_PER_TOKEN, 4),
            "models": dict(Counter(r.get("model") for r in rows if r.get("jev")))}


# --------------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game")
    ap.add_argument("task", choices=("makes", "possession", "turnovers"))
    ap.add_argument("--n", type=int, default=None, help="moments (makes / possession: 40, half of each class; "
                                                        "turnovers: every one)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="full", help="the engine run compared: runs/<tag>/events.jsonl")
    ap.add_argument("--tol", type=float, default=2.5, help="s between a play-by-play shot and the engine's result")
    ap.add_argument("--lead", type=float, default=3.0, help="possession: s before the shot's play-by-play time")
    ap.add_argument("--quiet", type=float, default=2.0, help="possession: s without another action before the moment")
    ap.add_argument("--offsets", default="2", help="turnovers: s from the turnover to the frames on each side, "
                                                   "comma-separated (e.g. 2.5,4,6)")
    ap.add_argument("--jerseys", default=None, help="possession / turnovers: jersey colours AWAY,HOME as Jev is told, "
                                                    "e.g. white,blue")
    ap.add_argument("--dry-run", action="store_true", help="write the images only, no API call")
    args = ap.parse_args(argv)

    jerseys: dict[Team, str] = {}
    if args.task != "makes":
        parts = [p.strip() for p in (args.jerseys or "").split(",")]
        if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
            raise SystemExit(f"{args.task} needs --jerseys AWAY,HOME: two different colours, e.g. white,blue")
        jerseys = {Team.AWAY: parts[0], Team.HOME: parts[1]}
    root = Path(args.game)
    game, camera = load_game(root)
    if camera.vfr:
        raise SystemExit("variable frame rate video: convert it to a constant rate first (docs/NBA.md)")
    n = args.n if args.n is not None else 40 if args.task != "turnovers" else 10**9
    _, aligned = truth(root)
    events = engine_events(root / "runs" / args.tag / "events.jsonl")
    out_dir = root / "jev_probe"
    img_dir = out_dir / args.task
    img_dir.mkdir(parents=True, exist_ok=True)
    frames = Frames(game["video"])
    w, h = camera.image_size

    picks: Iterator[tuple[Moment | None, dict[str, Any]]]
    if args.task == "makes":
        ball = {int(f): (p[0], p[1]) for f, p in json.loads((root / "ball_track.json").read_text())["path"].items()}
        order = balanced(field_goals(aligned.aligned), lambda al: bool(al.action.made), 10 * n, args.seed)
        picks = ((make_moment(al, camera, ball), {}) for al in order)
    elif args.task == "possession":
        cands = {id(al): m for al, m in possession_moments(aligned.aligned, camera, args.lead, args.quiet)}
        order = balanced([al for al in aligned.aligned if id(al) in cands], lambda al: al.action.team.value, 10 * n,
                         args.seed)
        picks = ((cands[id(al)], {}) for al in order)
    else:
        offsets = [float(x) for x in args.offsets.split(",") if x.strip()]
        picks = ((m, known) for _, m, known in turnover_moments(events, aligned.aligned, camera, offsets, args.tol))

    client = None
    bundle = None
    if not args.dry_run:
        client = JevClient(JevConfig(mode="live", images_enabled=True), api_key=runtime_env().get("TYPESAFE_API_KEY"))
        # offline: generous HTTP timeout and retries; the model stays pinned to the B6 image model
        bundle = dataclasses.replace(client.bundle("B6"), http_timeout_s=60.0,
                                     retry=_retry(2, backoff_initial=0.5, backoff_max=5.0, timeout=180.0))
    rows: list[dict[str, Any]] = []
    for m, extra in picks:
        if len(rows) >= n:
            break
        if m is None:
            continue
        if args.task == "makes":
            crops = rim_crops(frames, camera, court_for("nba"), m.frames[0], m.side)
            if crops is None:
                continue
            payload = crops
        else:
            imgs = [frames.get(f) for f in m.frames]
            if any(i is None for i in imgs):
                continue
            s = FRAME_PX / max(w, h)
            payload = [cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                       for img in imgs if img is not None]
        for i, img in enumerate(payload):
            cv2.imwrite(str(img_dir / f"{m.id}_{i:02d}.jpg"), img)
        row: dict[str, Any] = {"id": m.id, "t": round(m.t, 2), "team": m.team.value, "made": m.made,
                               "description": m.description, "frames": m.frames, **extra}
        if args.task == "makes":
            row.update(centred_on=m.centred_on, side=m.side, engine=engine_result(events, m.t, args.tol))
        elif args.task == "possession":
            et = engine_team(events, camera.time(m.frames[0]))
            row.update(engine=et.value if et is not None else None)
        else:
            row.update(engine=m.team.value)  # the team the engine says lost the ball
        if client is not None and bundle is not None:
            enc = [jpeg(img) for img in payload]
            if args.task == "makes":
                state, questions = make_request(enc)
            elif args.task == "possession":
                state, questions = possession_request(enc[0], jerseys)
            else:
                state, questions = turnover_request(enc[:row["n_before"]], enc[row["n_before"]:], jerseys)
            r = client.ask(bundle, state, questions, tag=m.id)
            row.update(latency_ms=round(r.latency_ms), input_tokens=r.input_tokens, model=r.model, error=r.error)
            if r.ok and r.response is not None:
                row["jev"] = decode(args.task, r.response, jerseys)
            elif r.error in FATAL:
                print(f"stopping: Jev request failed ({r.error})", file=sys.stderr)
                rows.append(row)
                break
        rows.append(row)
        print(progress(args.task, row), flush=True)
    if client is not None:
        client.close()

    if args.dry_run:  # the images only: a dry run never replaces a live run's results
        print(f"{len(rows)} moments, images in {img_dir}")
        return 0
    with (out_dir / f"{args.task}.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    score = {"makes": score_makes, "possession": score_possession, "turnovers": score_turnovers}[args.task]
    summary = {"task": args.task, "game": root.name, "seed": args.seed, "tag": args.tag, "results": score(rows),
               "usage": usage(rows)}
    if jerseys:
        summary["jerseys"] = {t.value: c for t, c in jerseys.items()}
    (out_dir / f"{args.task}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


def decode(task: str, resp: Any, jerseys: Mapping[Team, str]) -> dict[str, Any]:
    if task == "makes":
        return decode_make(resp)
    by_colour = {c: t.value for t, c in jerseys.items()}
    if task == "turnovers":
        out: dict[str, Any] = {}
        for side in ("before", "after"):
            keys = sorted((k[:-4] for k in resp.choices if k.startswith(f"team_{side}_") and k.endswith("_rev")),
                          key=lambda k: int(k.rsplit("_", 1)[1]))
            per = [ensemble(probs_of(resp, k), probs_of(resp, f"{k}_rev")) for k in keys]
            p = ensemble(*per)  # the side's evidence: the mean over its frames
            team = pick(p, "unclear")
            out.update({side: by_colour.get(team, team), f"p_{side}": round(p[team], 4),
                        f"probs_{side}": {by_colour.get(k, k): round(v, 4) for k, v in p.items()},
                        f"frames_{side}": [by_colour.get(pick(x, "unclear"), pick(x, "unclear")) for x in per]})
        return out
    ph = ensemble(probs_of(resp, "holder"), probs_of(resp, "holder_rev"))
    pt = ensemble(probs_of(resp, "team"), probs_of(resp, "team_rev"))
    holder, team = pick(ph, "nobody"), pick(pt, "unclear")
    return {"holder": by_colour.get(holder, holder), "team": by_colour.get(team, team), "p_team": round(pt[team], 4),
            "probs_holder": {by_colour.get(k, k): round(v, 4) for k, v in ph.items()},
            "probs_team": {by_colour.get(k, k): round(v, 4) for k, v in pt.items()},
            "ball_visible": round(float(resp.nouls["ball_visible"].noul), 3)}


def progress(task: str, row: Mapping[str, Any]) -> str:
    jv = row.get("jev") or {}
    if task == "makes":
        truth_, answer = "made" if row["made"] else "missed", jv.get("outcome")
    elif task == "possession":
        truth_, answer = row["team"], jv.get("team")
    else:
        truth_ = "turnover" if row["pbp_turnover"] else "change" if row["pbp_change"] else "none"
        answer = f"{jv.get('before')}->{jv.get('after')}" if jv else None
    return f"{row['id']} t={row['t']:7.1f} truth={truth_:8} jev={answer} engine={row['engine']}"


if __name__ == "__main__":
    raise SystemExit(main())
