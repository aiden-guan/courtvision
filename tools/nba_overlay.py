"""Watch the analysis: a game's video with the possession timeline and the shot decisions drawn on it.

    .venv/bin/python tools/nba_overlay.py runs/nba/<game> --tag full --auto 180
    .venv/bin/python tools/nba_overlay.py runs/nba/<game> --tag full --start 600 --end 780 [--out clip.mp4]

Reads the game's video (``game.json``), its camera track, the linked ball path (``ball_track.json``) and a run's
``timeline.jsonl`` / ``timeline.json`` (``nba_analyze.py <game> timeline``); with the play-by-play (the run's
``reference``, or ``espn_summary.json`` and ``scorebug.jsonl`` in the game folder) it shows the truth beside them. The
Studio draws the same on its feed (docs/STUDIO.md). On every frame:

* the bar across the top: the team with the ball by the timeline (in its colour; "bringing it up" while it is in its
  own back court) and, when the play-by-play knows, by the play-by-play, with whether they agree;
* the tracked ball (a ring) and the rims (boxes; green or red at a decided make or miss);
* each decided shot for a few seconds: made or missed, its value, a free throw, and "the engine had missed this one"
  for a shot added at a rim visit; changes of possession with their cause; rebounds, steals and turnovers;
* a strip of the 30 s around now (the timeline's team above, the play-by-play's below) and the latest play-by-play
  line, along the bottom.

``--auto SECONDS`` picks the stretch of that length with the most going on (shots, free throws, shots added at rim
visits, changes of possession), in live play. The clip keeps the broadcast's sound and is written as H.264 with
ffmpeg, ``--width`` pixels wide (default ``runs/nba/<game>/videos/<tag>_<start>-<end>.mp4``).
"""

from __future__ import annotations

import argparse
import bisect
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nba_analyze import load_game, truth  # noqa: E402

from courtvision.geometry.court import court_for  # noqa: E402
from courtvision.nba import espn  # noqa: E402
from courtvision.nba import timeline as tlm  # noqa: E402
from courtvision.nba.analyze import BallTrack, CameraTrack, rim_boxes  # noqa: E402
from courtvision.types import Event, EventKind, Team  # noqa: E402

K = EventKind
FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE, BLACK, GREY = (255, 255, 255), (0, 0, 0), (90, 90, 90)
GREEN, RED, AMBER = (60, 190, 60), (60, 60, 220), (0, 170, 255)
SHOW_S = 2.5  # how long an event's label stays on screen
STRIP_S = 30.0  # the possession strip's width in seconds


@dataclass
class Overlay:
    """What is drawn, read once for the clip."""

    names: dict[Team, str]
    colours: dict[Team, tuple[int, int, int]]
    segments: list[tuple[float, float, Team]]
    back: list[tuple[float, float]]
    events: list[Event]
    marks: list[tuple[float, Team | None]] = field(default_factory=list)  # the play-by-play's team with the ball
    pbp: list[tuple[float, str]] = field(default_factory=list)  # (time, description) of play-by-play actions

    def __post_init__(self) -> None:  # time indexes for the per-frame lookups
        self._seg = (np.array([a for a, _, _ in self.segments]), np.array([b for _, b, _ in self.segments]))
        self._back = (np.array([a for a, _ in self.back]), np.array([b for _, b in self.back]))
        self._mark_t = np.array([t for t, _ in self.marks])
        self._ev_t = [e.t for e in self.events]
        self._pbp_t = [t for t, _ in self.pbp]

    def teams_at(self, ts: np.ndarray) -> list[Team | None]:
        """The timeline's team with the ball at each of ``ts``."""
        i = np.searchsorted(self._seg[0], ts, side="right") - 1
        return [self.segments[k][2] if k >= 0 and t <= self._seg[1][k] else None for k, t in zip(i, ts, strict=True)]

    def truths_at(self, ts: np.ndarray) -> list[Team | None]:
        """The play-by-play's team with the ball at each of ``ts`` (None: loose, or no play-by-play)."""
        if not self.marks:
            return [None] * len(ts)
        i = np.searchsorted(self._mark_t, ts, side="right") - 1
        return [self.marks[k][1] if k >= 0 else None for k in i]

    def team_at(self, t: float) -> Team | None:
        return self.teams_at(np.array([t]))[0]

    def truth_at(self, t: float) -> Team | None:
        return self.truths_at(np.array([t]))[0]

    def bringing_up(self, t: float) -> bool:
        k = int(np.searchsorted(self._back[0], t, side="right")) - 1
        return k >= 0 and t <= self._back[1][k]

    def recent(self, t: float, window_s: float) -> list[Event]:
        """The events of the last ``window_s`` seconds up to ``t``."""
        return self.events[bisect.bisect_left(self._ev_t, t - window_s):bisect.bisect_right(self._ev_t, t)]

    def pbp_line(self, t: float, window_s: float = 4.0) -> str | None:
        k = bisect.bisect_right(self._pbp_t, t) - 1
        return self.pbp[k][1] if k >= 0 and t - self._pbp_t[k] <= window_s else None


def label(e: Event, names: Mapping[Team, str]) -> tuple[str, str, tuple[int, int, int]] | None:
    """(headline, detail, colour) of an event worth showing, or None."""
    who = names.get(e.team, "?") if e.team is not None else "?"
    if e.kind in (K.SHOT_MADE, K.SHOT_MISSED):
        made = e.kind is K.SHOT_MADE
        ft = e.value == 1 or e.attrs.get("shot_type") == "free_throw"
        what = "FREE THROW" if ft else f"{e.value}-POINTER"
        probs = e.attrs.get("makes") or {}
        detail = ("the engine had missed this one: found at the rim" if e.attrs.get("inferred") == "rim_visit"
                  else f"P(made) {probs['p_made']:.2f}" if isinstance(probs, dict) and "p_made" in probs else "")
        return f"{who} {what} {'MADE' if made else 'MISSED'}", detail, GREEN if made else RED
    if e.kind is K.POSSESSION:
        cause = str(e.attrs.get("cause", "")).replace("_", " ")
        return f"{who} BALL", cause, AMBER
    if e.kind is K.REBOUND:
        return f"{who} {'OFFENSIVE' if e.attrs.get('offensive') else 'DEFENSIVE'} REBOUND", "", AMBER
    if e.kind is K.STEAL:
        return f"{who} STEAL", "", AMBER
    if e.kind is K.TURNOVER:
        return f"{who} TURNOVER", "", AMBER
    return None


def pick_window(events: Sequence[Event], duration: float, tracked: Sequence[float], step: float = 5.0,
                min_tracked: float = 0.6) -> tuple[float, float]:
    """The ``duration``-second stretch with the most going on (a shot 1, a free throw or a shot added at a rim visit 3,
    a change of possession 1, a steal or a turnover 2) among those with the camera on the court at least
    ``min_tracked`` of the time (``tracked``: the sorted times of tracked frames)."""
    weight = []
    for e in events:
        if e.kind in (K.SHOT_MADE, K.SHOT_MISSED):
            special = e.value == 1 or e.attrs.get("inferred") == "rim_visit"
            weight.append((e.t, 3.0 if special else 1.0))
        elif e.kind is K.POSSESSION:
            weight.append((e.t, 1.0))
        elif e.kind in (K.STEAL, K.TURNOVER):
            weight.append((e.t, 2.0))
    if not weight:
        return (0.0, duration)
    ts, ws = np.array([t for t, _ in weight]), np.array([w for _, w in weight])
    trk = np.asarray(tracked, float)
    best = (-1.0, 0.0)
    for start in np.arange(max(0.0, ts.min() - duration / 2), ts.max(), step):
        end = start + duration
        if len(trk) and (np.searchsorted(trk, end) - np.searchsorted(trk, start)) < min_tracked * duration * 25:
            continue  # mostly replays or a break (frames at about 30 fps, tracked or not)
        score = float(ws[(ts >= start) & (ts < end)].sum())
        if score > best[0]:
            best = (score, float(start))
    return best[1], best[1] + duration


# --------------------------------------------------------------------------- drawing


def _text(img: np.ndarray, s: str, org: tuple[int, int], scale: float, colour: tuple[int, int, int],
          thick: int = 1) -> None:
    cv2.putText(img, s, org, FONT, scale, BLACK, thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, colour, thick, cv2.LINE_AA)


def _swatch(img: np.ndarray, x: int, y: int, w: int, h: int, colour: tuple[int, int, int], s: str,
            scale: float) -> int:
    """A filled box with ``s`` in it (black text on a light colour, white on a dark one); returns its right edge."""
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, 1)
    w = max(w, tw + 16)
    cv2.rectangle(img, (x, y), (x + w, y + h), colour, -1)
    ink = BLACK if sum(colour) > 380 else WHITE
    cv2.putText(img, s, (x + 8, y + (h + th) // 2), FONT, scale, ink, 1, cv2.LINE_AA)
    return x + w


def draw(img: np.ndarray, t: float, ov: Overlay, rims: Mapping[str, tuple[float, ...]],
         ball: tuple[float, float] | None, sx: float) -> None:
    """Draw the overlay for video time ``t`` on ``img`` (already scaled by ``sx`` from the source frame)."""
    h, w = img.shape[:2]
    u = w / 1280.0  # layout unit
    # the bar across the top
    cv2.rectangle(img, (0, 0), (w, int(46 * u)), (20, 20, 20), -1)
    team = ov.team_at(t)
    x = int(12 * u)
    _text(img, "TIMELINE", (x, int(30 * u)), 0.55 * u, (200, 200, 200))
    x += int(100 * u)
    if team is not None:
        what = f"{ov.names[team]} BALL" + (" - bringing it up" if ov.bringing_up(t) else "")
        x = _swatch(img, x, int(8 * u), int(120 * u), int(30 * u), ov.colours[team], what, 0.6 * u)
    if ov.marks:
        x += int(30 * u)
        _text(img, "PLAY-BY-PLAY", (x, int(30 * u)), 0.55 * u, (200, 200, 200))
        x += int(140 * u)
        real = ov.truth_at(t)
        if real is None:
            _text(img, "loose / unknown", (x, int(30 * u)), 0.6 * u, (200, 200, 200))
        else:
            x = _swatch(img, x, int(8 * u), int(90 * u), int(30 * u), ov.colours[real], f"{ov.names[real]} BALL", 0.6 * u)
            agree = team is real
            _text(img, "agrees" if agree else "differs", (x + int(12 * u), int(30 * u)), 0.6 * u,
                  GREEN if agree else RED)
    # the rims, lit at a decided shot
    for side, box in rims.items():
        shot = next((e for e in ov.recent(t, 1.5) if e.kind in (K.SHOT_MADE, K.SHOT_MISSED)), None)
        colour = (GREEN if shot.kind is K.SHOT_MADE else RED) if shot is not None else (220, 220, 220)
        x1, y1, x2, y2 = (int(v * sx) for v in box)
        pad = int(6 * u)
        cv2.rectangle(img, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), colour, 2 if shot is not None else 1)
    if ball is not None:
        cv2.circle(img, (int(ball[0] * sx), int(ball[1] * sx)), int(13 * u), (0, 230, 255), 2, cv2.LINE_AA)
    # the events of the last few seconds, newest on top
    recent = [(e, lab) for e in ov.recent(t, SHOW_S) if (lab := label(e, ov.names)) is not None]
    y = int(90 * u)
    for e, (head, detail, colour) in sorted(recent, key=lambda p: -p[0].t)[:3]:
        _text(img, head, (int(20 * u), y), 0.95 * u, colour, 2)
        if detail:
            _text(img, detail, (int(22 * u), y + int(28 * u)), 0.6 * u, WHITE)
        y += int(68 * u)
    # the strip of the 30 s around now, and the latest play-by-play line
    top = h - int(78 * u)
    cv2.rectangle(img, (0, top - int(6 * u)), (w, h), (20, 20, 20), -1)
    x0, x1 = int(12 * u), w - int(12 * u)
    xs = np.arange(x0, x1, 2)
    ts = t - STRIP_S / 2 + STRIP_S * (xs - x0) / (x1 - x0)
    rows = [ov.teams_at(ts)] + ([ov.truths_at(ts)] if ov.marks else [])
    for r, teams in enumerate(rows):
        yy = top + r * int(16 * u)
        a = 0
        for b in range(1, len(xs) + 1):  # one box per run of a team
            if b == len(xs) or teams[b] is not teams[a]:
                colour = ov.colours[teams[a]] if teams[a] is not None else GREY
                cv2.rectangle(img, (int(xs[a]), yy), (int(xs[b - 1]) + 2, yy + int(12 * u)), colour, -1)
                a = b
    mid = (x0 + x1) // 2
    cv2.line(img, (mid, top - int(4 * u)), (mid, top + len(rows) * int(16 * u)), (0, 230, 255), 2)
    line = ov.pbp_line(t)
    if line is not None:
        _text(img, f"PLAY-BY-PLAY: {line}"[:110], (x0, h - int(12 * u)), 0.55 * u, (230, 230, 230))


# --------------------------------------------------------------------------- main


def load_overlay(root: Path, tag: str, game: Mapping[str, object]) -> Overlay:
    run = root / "runs" / tag
    doc = json.loads((run / "timeline.json").read_text())
    events = [Event.from_dict(json.loads(line)) for line in (run / "timeline.jsonl").open()]
    names = {Team.HOME: "HOME", Team.AWAY: "AWAY"}
    if (root / "espn_summary.json").exists():
        names.update({v["side"]: v["abbr"] for v in espn.teams(espn.load_summary(root / "espn_summary.json")).values()})
    ref = doc.get("reference")  # written by the timeline since the Studio shows it; older runs: aligned here
    if ref is None and (root / "espn_summary.json").exists() and (root / "scorebug.jsonl").exists():
        ref = tlm.reference(truth(root)[1].aligned)
    marks = [(float(t), Team(team) if team else None) for t, team in ref["marks"]] if ref else []
    pbp = [(float(t), str(s)) for t, s in ref["plays"]] if ref else []
    colours = {Team.HOME: tuple(game.get("home_color_bgr") or (200, 120, 40)),  # type: ignore[arg-type]
               Team.AWAY: tuple(game.get("away_color_bgr") or (230, 230, 230))}  # type: ignore[arg-type]
    segments = [(float(a), float(b), Team(team)) for a, b, team in doc["segments"]]
    back = [(float(a), float(b)) for a, b in doc.get("back_court", [])]
    return Overlay(names, colours, segments, back, sorted(events, key=lambda e: e.t), marks, pbp)  # type: ignore[arg-type]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game")
    ap.add_argument("--tag", default="full", help="the run: runs/<tag>/timeline.jsonl and timeline.json")
    ap.add_argument("--start", type=float, default=None, help="video seconds")
    ap.add_argument("--end", type=float, default=None, help="video seconds")
    ap.add_argument("--auto", type=float, default=None, help="pick a stretch of this many seconds with the most going on")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.auto is None and (args.start is None or args.end is None or args.end <= args.start):
        raise SystemExit("give --start and --end (video seconds, end after start), or --auto SECONDS")
    root = Path(args.game)
    game, camera = load_game(root)
    if camera.vfr:
        raise SystemExit("variable frame rate video: convert it to a constant rate first (docs/NBA.md)")
    ov = load_overlay(root, args.tag, game)
    if args.auto is not None:
        start, end = pick_window(ov.events, args.auto, sorted(camera.time(f) for f in camera.params))
    else:
        start, end = float(args.start), float(args.end)
    ball = BallTrack.load(root / "ball_track.json").path if (root / "ball_track.json").exists() else {}
    out = Path(args.out) if args.out else root / "videos" / f"{args.tag}_{start:.0f}-{end:.0f}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(Path(str(game["video"])), camera, ov, ball, start, end, args.width, out)
    print(f"{out} ({start:.0f}-{end:.0f} s)")
    return 0


def render(video: Path, camera: CameraTrack, ov: Overlay, ball: Mapping[int, Sequence[float]], start: float,
           end: float, width: int, out: Path) -> None:
    court = court_for("nba")
    w0, h0 = camera.image_size
    sx = width / w0
    size = (width, int(round(h0 * sx / 2)) * 2)
    rate = "30000/1001" if abs(camera.fps - 29.97) < 0.01 else f"{camera.fps:.6f}"
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
           "-r", rate, "-i", "-", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video),
           "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(out)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(video))
    f0, f1 = camera.frame_at(start), camera.frame_at(end)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    try:
        for f in range(f0, f1):
            ok, img = cap.read()
            if not ok:
                break
            frame = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            cam = camera.get(f)
            rims = {s: (b.x1, b.y1, b.x2, b.y2) for s, b in rim_boxes(cam, court, (w0, h0)).items()} if cam else {}
            p = ball.get(f)
            draw(frame, camera.time(f), ov, rims, (p[0], p[1]) if p is not None else None, sx)
            assert enc.stdin is not None
            enc.stdin.write(frame.tobytes())
    finally:
        cap.release()
        if enc.stdin is not None:
            enc.stdin.close()
        enc.wait()
    if enc.returncode != 0:
        raise SystemExit(f"ffmpeg failed ({enc.returncode})")


if __name__ == "__main__":
    raise SystemExit(main())
