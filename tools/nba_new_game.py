"""Add an NBA game recording to the learning loop in one command: its folder ``runs/nba/<YYYY-MM-DD-AWY-HOM>/`` with
``game.json``, the ESPN summary and the NBA.com play-by-play (research use only: docs/NBA.md, docs/TRAINING.md §1).

    .venv/bin/python tools/nba_new_game.py VIDEO [--date YYYY-MM-DD] [--teams AWY,HOM | --espn-event ID]
        [--name FOLDER] [--nba-root runs/nba] [--broadcaster ESPN] [--nba-game-id ID] [--no-nba-pbp] [--force] [--yes]
    .venv/bin/python tools/nba_new_game.py VIDEO --check [...]          # validate only: writes nothing
    .venv/bin/python tools/nba_new_game.py --batch games.csv [--check] [--force]

**Finding the game.** The ESPN event is the game of the two teams on ESPN's scoreboard of the date
(``espn.fetch_scoreboard``), whatever order or spelling the teams have: NBA.com tricode, ESPN abbreviation, an alias
(GS / GSW, NY / NYK, SA / SAS, NO / NOP, UTAH / UTA, WSH / WAS, PHO / PHX, BRK / BKN, ...) or a name
(``espn.team_code``). ``--espn-event`` names the event directly; it is checked against its ESPN summary. The date is
the US Eastern date of the tip-off, the day ESPN lists the game on. Without ``--date`` it is guessed from the file
name (for a studio upload, ``source.video``, also from the original name in its ``video.json``; then from the folder
name), else from the container's ``creation_time`` (ffprobe), and confirmed: ESPN must list a game of these teams on
it (the day before and after are tried too), and an interactive run asks before writing (``--yes``: do not ask).
Teams not given are guessed from the same names; with none, an interactive run lists the day's games to pick from.

**What it writes.** ``runs/nba/<date>-<away>-<home>/`` (NBA.com tricodes; ``--name`` for another name, which should
still start with the date: the loop holds out the newest games by folder name) with ``game.json``: ``video`` (the
absolute path; the video is never copied and never downloaded), ``espn_event``, ``date``, ``teams`` (``away`` /
``home``), ``broadcaster`` (``--broadcaster``, else one named in the file name), ``national_tv`` (ESPN's list) and
``nba_game_id``. Then ``espn_summary.json`` (``espn.fetch_summary``; only a final game whose box score and
play-by-play parse) and, unless ``--no-nba-pbp``, ``nba_pbp.csv`` (``nba_stats.export_pbp_csv``; the NBA.com game id
from ``--nba-game-id``, else the cdn.nba.com season schedule (current season), else the stats.nba.com scoreboard of
the date). An NBA.com step that fails is a warning: the loop runs without it. Last, the loop command to run next.

**Safety.** An existing ``game.json`` is not overwritten without ``--force``; the same video and event again is
"already added" (game.json is left as it is; missing data files are fetched). ``--force`` is still refused when the
folder holds analysis outputs and the video or the event changes (the loop would take them for the new game's; to
point game.json at a moved copy of the same recording, edit ``video`` by hand), and when game.json is a link (a
variant folder). A game already in another folder (same ESPN event or same video) is refused (``--force``: add it
anyway; the loop treats a second folder of an event as a variant). The video must open with cv2, decode at its
start, middle and end, and last ``--min-minutes`` to ``--max-minutes``.

**Without network** ESPN cannot be searched: give the event id by hand, ``--espn-event`` (the number after gameId in
the game's espn.com page URL, ``espn.com/nba/game/_/gameId/<id>``), with ``--date`` and ``--teams``. game.json is
written unchecked and the loop's ``espn`` stage fetches the summary when the network is back.

**--batch CSV**: a header row, then one game per row: ``video`` (a relative path is taken from the CSV's folder),
``date``, ``teams`` (``AWY@HOM``, ``"AWY,HOM"`` in quotes, or ``away`` and ``home`` columns) and optionally
``espn_event``, ``name``, ``broadcaster``, ``nba_game_id``; blank lines and lines starting with ``#`` are skipped.
Rows are independent and never ask; a game already added is skipped, so a batch can be run again. **--check**
validates each game (the video, the ESPN event and its teams and date, the folder) and writes nothing.

Exit codes: 0 every game added or already there (--check: every game valid), 1 a game failed, 2 bad input.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import http.client
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import urllib.error
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from courtvision.nba import espn, nba_stats  # noqa: E402
from courtvision.types import Team  # noqa: E402

PYTHON = ".venv/bin/python"
DEFAULT_NBA_ROOT = REPO / "runs" / "nba"
# what a game folder holds besides analysis outputs: fetched data and step logs (hidden files are ignored too)
DATA_PREFIXES = ("game.json", "espn_summary", "nba_pbp", "nba_tracking", "logs")
GAME_KEYS = ("video", "espn_event", "date", "teams", "broadcaster", "national_tv", "nba_game_id")
SCOREBUG_TV = ("ESPN", "ABC")  # the broadcasts of perception/scorebug_ocr.py ESPN_2024, the hand-measured bug
BROADCASTER_WORDS = {"espn": "ESPN", "abc": "ABC", "tnt": "TNT", "nbatv": "NBA TV", "nbc": "NBC", "peacock": "Peacock",
                     "prime": "Prime Video", "amazon": "Prime Video", "trutv": "truTV"}
BATCH_COLUMNS = ("video", "date", "teams", "away", "home", "espn_event", "name", "broadcaster", "nba_game_id")
FOLDER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
URL_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
# abbreviations that are words too: in a file name they count only in capitals (as do two-letter ones: GS, NO, ...)
WORD_CODES = {"was", "min", "den", "mil", "ind", "por", "det", "tor", "sac", "cha", "chi", "mia"}


# --------------------------------------------------------------------------- network


class Offline(Exception):
    """A host could not be reached (no network, DNS, refused, timeout)."""


class FetchError(Exception):
    """A host answered, but not with the document asked for (an HTTP error, not JSON)."""


class Net:
    """Every network read of this tool, all through ``urllib.request.urlopen`` (the tests replace it): a document is
    fetched once per run, and a host that could not be reached is not asked again in the run."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout
        self.cache: dict[str, dict[str, Any]] = {}
        self.down: dict[str, str] = {}  # host -> why it could not be reached

    def _get(self, key: str, host: str, fetch: Callable[[], Any]) -> dict[str, Any]:
        if key in self.cache:
            return self.cache[key]
        if host in self.down:
            raise Offline(f"{host} unreachable ({self.down[host]})")
        try:
            data = fetch()
        except urllib.error.HTTPError as exc:
            raise FetchError(f"{host}: HTTP {exc.code} {exc.reason}") from exc
        except ValueError as exc:  # JSONDecodeError, UnicodeDecodeError
            raise FetchError(f"{host}: not a JSON document ({exc})") from exc
        except (OSError, http.client.HTTPException) as exc:  # URLError (no network, DNS), refused, reset, timeout,
            # a body cut short (IncompleteRead)
            why = str(getattr(exc, "reason", "") or exc) or type(exc).__name__
            self.down[host] = why
            raise Offline(f"{host} unreachable ({why})") from exc
        if not isinstance(data, dict):
            raise FetchError(f"{host}: not a JSON object ({type(data).__name__})")
        self.cache[key] = data
        return data

    def scoreboard(self, day: str) -> dict[str, Any]:
        return self._get(f"espn scoreboard {day}", "site.api.espn.com",
                         lambda: espn.fetch_scoreboard(day, timeout=self.timeout))

    def summary(self, event: str) -> dict[str, Any]:
        return self._get(f"espn summary {event}", "site.api.espn.com",
                         lambda: espn.fetch_summary(event, timeout=self.timeout))

    def nba_schedule(self) -> dict[str, Any]:
        return self._get("nba schedule", "cdn.nba.com", lambda: nba_stats.fetch_schedule(max(self.timeout, 60.0)))

    def nba_scoreboard(self, day: str, version: int) -> dict[str, Any]:
        return self._get(f"nba scoreboard v{version} {day}", "stats.nba.com",
                         lambda: nba_stats.fetch_scoreboard(day, version, min(self.timeout, 20.0)))

    def nba_pbp(self, game_id: str) -> dict[str, Any]:
        return self._get(f"nba pbp {game_id}", "cdn.nba.com",
                         lambda: nba_stats.fetch_pbp(game_id, timeout=self.timeout))


# --------------------------------------------------------------------------- what one game asks for and what it got


@dataclass
class Spec:
    """One game as asked: the command line or one row of a batch file."""

    video: str
    date: str | None = None
    teams: str | None = None
    espn_event: str | None = None
    name: str | None = None
    broadcaster: str | None = None
    nba_game_id: str | None = None
    where: str = ""  # "games.csv line 3"
    base: Path | None = None  # a relative video path is taken from here (the batch file's folder)
    problem: str | None = None  # a batch row that cannot be read


@dataclass
class Options:
    nba_root: Path
    check: bool = False
    force: bool = False
    nba_pbp: bool = True
    yes: bool = False
    interactive: bool = False  # may ask (a terminal; not --batch, --check or --yes)
    min_minutes: float = 40.0
    max_minutes: float = 360.0


@dataclass
class Hints:
    """What the video's names say about the game, each with where it was read."""

    dates: list[tuple[dt.date, str]] = field(default_factory=list)
    teams: tuple[str, str] | None = None
    teams_from: str = ""
    broadcaster: str | None = None
    broadcaster_from: str = ""


@dataclass
class Game:
    """The ESPN game a video is of."""

    event: str
    date: str  # YYYY-MM-DD: the US Eastern date ESPN lists the game on
    away: str  # NBA.com tricodes
    home: str
    names: dict[str, str] = field(default_factory=dict)  # "away" / "home" -> ESPN's display name
    completed: bool | None = None
    status: str = ""
    national_tv: list[str] = field(default_factory=list)
    summary: dict[str, Any] | None = None  # the ESPN summary, when it was fetched to find the game
    checked: bool = True  # False: given by hand while ESPN could not be reached
    date_from: str = ""  # where a guessed date was read ("" when given)
    teams_from: str = ""


@dataclass
class Plan:
    root: Path
    state: str  # new | exists (same video and event: game.json is kept) | overwrite (--force)
    old: dict[str, Any] = field(default_factory=dict)  # the game.json there


@dataclass
class Report:
    """What happened to one game: (key, text) lines in order, "warning" and "error" lines among them."""

    title: str
    items: list[tuple[str, str]] = field(default_factory=list)
    status: str = "failed"  # added | exists | valid | failed
    shown: int = -1  # items printed so far (-1: not even the title)

    def add(self, key: str, text: str) -> None:
        self.items.append((key, text))

    def warn(self, text: str) -> None:
        self.items.append(("warning", text))

    def error(self, text: str) -> None:
        self.items.append(("error", text))

    @property
    def errors(self) -> list[str]:
        return [t for k, t in self.items if k == "error"]

    @property
    def warnings(self) -> list[str]:
        return [t for k, t in self.items if k == "warning"]

    def flush(self) -> None:
        """Print what is not printed yet (before a question, and at the end)."""
        if self.shown < 0:
            print(self.title)
            self.shown = 0
        for k, t in self.items[self.shown:]:
            print(f"  {k:<8} {t}")
        self.shown = len(self.items)
        sys.stdout.flush()


# --------------------------------------------------------------------------- small helpers


def show(p: Path) -> str:
    """A path as printed: relative to the working directory when inside it."""
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def _real(p: str | Path) -> str:
    """A video path as the loop resolves it (relative to the repository), links followed: for comparisons."""
    return os.path.realpath(os.path.join(REPO, os.path.expanduser(str(p))))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_text_atomic(path: Path, text: str) -> None:
    """A temporary file renamed over ``path``: a crash never leaves half a file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _hms(seconds: float) -> str:
    s = round(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _matchup(g: dict[str, Any]) -> str:
    side = {k: (g[k]["code"] or g[k]["abbr"]) for k in ("away", "home")}
    return f"{side['away']} @ {side['home']} (ESPN {g['event']}{', ' + g['status'] if g['status'] else ''})"


def _interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, OSError, ValueError):
        return False


def _confirm(question: str, ask: Callable[[str], str]) -> bool:
    try:
        return ask(f"{question} [Y/n] ").strip().lower() in ("", "y", "yes")
    except EOFError:
        return False


# --------------------------------------------------------------------------- dates and teams from names

_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_DATE_RES = (
    ("ymd", re.compile(r"(?<!\d)((?:19|20)\d\d)([-_. ])(\d{1,2})\2(\d{1,2})(?!\d)")),
    ("ymd8", re.compile(r"(?<!\d)((?:19|20)\d\d)(\d\d)(\d\d)(?!\d)")),
    ("mdy", re.compile(r"(?<!\d)(\d{1,2})([-_. /])(\d{1,2})\2((?:19|20)\d\d|\d\d)(?!\d)")),
    ("mon", re.compile(rf"(?<![a-z]){_MON}[-_. ,]*(\d{{1,2}})(?:st|nd|rd|th)?[-_. ,]+((?:19|20)\d\d)(?!\d)",
                       re.IGNORECASE)),
    ("dmon", re.compile(rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th)?[-_. ,]*{_MON}[-_. ,]*((?:19|20)\d\d)(?!\d)",
                        re.IGNORECASE)),
)
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")


def _readings(kind: str, m: re.Match[str]) -> list[tuple[int, int, int]]:
    g = m.groups()
    if kind == "ymd":
        return [(int(g[0]), int(g[2]), int(g[3]))]
    if kind == "ymd8":
        return [(int(g[0]), int(g[1]), int(g[2]))]
    if kind == "mdy":
        year = int(g[3]) + (2000 if len(g[3]) == 2 else 0)
        a, b = int(g[0]), int(g[2])
        return [(year, a, b), (year, b, a)] if a != b else [(year, a, b)]  # US first, then day-month
    if kind == "mon":
        return [(int(g[2]), _MONTHS.index(g[0][:3].lower()) + 1, int(g[1]))]
    return [(int(g[2]), _MONTHS.index(g[1][:3].lower()) + 1, int(g[0]))]  # dmon


def parse_date(text: str) -> dt.date:
    """A date given on the command line or in a batch row: YYYY-MM-DD (or YYYYMMDD)."""
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"date {text!r}: write it as YYYY-MM-DD")


def guess_dates(text: str) -> list[dt.date]:
    """The dates a file name writes, in order: 2024-12-25, 20241225, 12-25-2024 / 12.25.24 (US; the day-month
    reading too when both are dates), Dec 25 2024, 25 December 2024."""
    found: list[tuple[int, int, dt.date]] = []
    for kind, rx in _DATE_RES:
        for m in rx.finditer(text):
            for y, mo, d in _readings(kind, m):
                try:
                    day = dt.date(y, mo, d)
                except ValueError:
                    continue
                if 1990 <= day.year <= 2100:
                    found.append((m.start(), len(found), day))
    return list(dict.fromkeys(d for _, _, d in sorted(found)))


def guess_teams(text: str) -> list[str]:
    """The NBA.com tricodes of the teams a file name names, in order: names (full, nickname, city; "Los Angeles"
    alone is left out), abbreviations (two-letter ones and those that are words, such as MIN or WAS, only in
    capitals) and two tricodes run together (LALGSW)."""
    words = re.findall(r"[A-Za-z0-9]+", text)
    out: list[str] = []
    i = 0
    while i < len(words):
        hit: tuple[int, set[str]] | None = None
        for n in (3, 2, 1):
            if i + n > len(words):
                continue
            key = " ".join(w.lower() for w in words[i:i + n])
            codes = espn.TEAM_NAMES.get(key)
            if codes is None and n == 1:
                codes = espn.TEAM_CODES.get(key)
                if codes and (len(key) < 3 or key in WORD_CODES) and not words[i].isupper():
                    codes = None
            if codes:
                hit = (n, codes)
                break
        if hit is None:
            w = words[i].upper()
            if len(w) == 6 and w[:3] in espn.TEAMS and w[3:] in espn.TEAMS:
                out += [w[:3], w[3:]]
            i += 1
            continue
        if len(hit[1]) == 1:
            out.append(next(iter(hit[1])))
        i += hit[0]
    return list(dict.fromkeys(out))


def guess_broadcaster(text: str) -> str | None:
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", text)]
    for i, w in enumerate(words):
        if w in BROADCASTER_WORDS:
            return BROADCASTER_WORDS[w]
        if w == "nba" and words[i + 1:i + 2] == ["tv"]:
            return "NBA TV"
    return None


def name_texts(video: Path) -> list[tuple[str, str]]:
    """The names that may say which game a video is: its file name, the original name of a studio upload
    (``runs/studio/<id>/source.video`` with ``video.json``) and the name of its folder."""
    out = [("the file name", video.name)]
    meta = video.parent / "video.json"
    if video.name == "source.video" and meta.is_file():
        doc = _read_json(meta)
        if isinstance(doc, dict) and doc.get("name"):
            out.append(("the studio upload's name", str(doc["name"])))
    if video.parent.name:
        out.append(("the folder name", video.parent.name))
    return out


def read_hints(video: Path) -> Hints:
    h = Hints()
    for src, text in name_texts(video):
        if not h.dates:
            h.dates = [(d, src) for d in guess_dates(text)]
        teams = guess_teams(text)
        if h.teams is None and len(teams) == 2:
            h.teams, h.teams_from = (teams[0], teams[1]), src
        tv = guess_broadcaster(text)
        if h.broadcaster is None and tv:
            h.broadcaster, h.broadcaster_from = tv, src
    return h


def ffprobe_format(video: Path) -> dict[str, Any]:
    """ffprobe's ``format`` section (``duration``, ``tags.creation_time``); {} without ffprobe or when it fails."""
    exe = shutil.which("ffprobe")
    if exe is None:
        return {}
    try:
        res = subprocess.run([exe, "-v", "error", "-show_entries", "format=duration:format_tags=creation_time",
                              "-of", "json", str(video)], capture_output=True, text=True, timeout=120, check=False)
        doc = json.loads(res.stdout or "{}")
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}
    fmt = doc.get("format") if isinstance(doc, dict) else None
    return fmt if isinstance(fmt, dict) else {}


def metadata_date(video: Path) -> dt.date | None:
    """The US Eastern date of the container's ``creation_time``: a recorder's start, but a re-encoded file's encoding
    time, so only a guess for ESPN to confirm."""
    tags = ffprobe_format(video).get("tags") or {}
    day = espn.us_date(str(tags["creation_time"])) if isinstance(tags, dict) and tags.get("creation_time") else None
    date = dt.date.fromisoformat(day) if day else None
    return date if date is not None and date.year >= 1990 else None


# --------------------------------------------------------------------------- the video


def local_video(spec: Spec, rep: Report) -> Path | None:
    """The video as an absolute path (links not resolved: game.json keeps the path given); None with an error."""
    raw = spec.video.strip()
    if URL_RE.match(raw):
        rep.error(f"{raw} is a URL: give the local video file (this tool never downloads video)")
        return None
    p = Path(os.path.expanduser(raw))
    p = Path(os.path.abspath(p if p.is_absolute() else (spec.base or Path.cwd()) / p))
    if not p.exists():
        rep.error(f"video not found: {p}")
        return None
    if not p.is_file():
        rep.error(f"not a video file: {p}")
        return None
    return p


def probe_video(video: Path) -> dict[str, Any]:
    """What cv2 sees of a video: whether it opens, fps, frame count, size, duration (frames / fps, else ffprobe's),
    and whether a frame decodes at the start, the middle and near the end (a cut-off download fails there)."""
    import cv2

    info: dict[str, Any] = {"opens": False, "fps": None, "frames": None, "size": None, "duration_s": None,
                            "decodes": {}}
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            return info
        info["opens"] = True
        fps, frames = float(cap.get(cv2.CAP_PROP_FPS) or 0.0), float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        info["size"] = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        info["fps"] = fps if math.isfinite(fps) and fps > 0 else None
        info["frames"] = int(frames) if math.isfinite(frames) and frames > 0 else None
        if info["fps"] and info["frames"]:
            info["duration_s"] = info["frames"] / info["fps"]
        else:
            try:
                dur = ffprobe_format(video).get("duration")
                info["duration_s"] = float(dur) if dur is not None else None
            except (TypeError, ValueError):
                pass
        for frac in (0.0, 0.5, 0.97):
            if frac > 0:
                if info["frames"]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frac * (info["frames"] - 1)))
                elif info["duration_s"]:
                    cap.set(cv2.CAP_PROP_POS_MSEC, frac * info["duration_s"] * 1000.0)
                else:
                    break
            ok, frame = cap.read()
            info["decodes"][frac] = bool(ok and frame is not None and getattr(frame, "size", 0))
    finally:
        cap.release()
    return info


def report_video(video: Path, info: dict[str, Any], opts: Options, rep: Report) -> None:
    if not info["opens"]:
        rep.add("video", str(video))
        rep.error("cv2 cannot open the video")
        return
    w, h = info["size"] or (0, 0)
    dur = info["duration_s"]
    desc = f"{_hms(dur) if dur else 'length unknown'}, {w}x{h}" + (f" at {info['fps']:.2f} fps" if info["fps"] else "")
    rep.add("video", f"{video} ({desc})")
    dec = info["decodes"]
    if not dec.get(0.0):
        rep.error("cv2 opens the video but decodes no frame at its start")
    bad = [f"{round(f * 100)}%" for f in (0.5, 0.97) if f in dec and not dec[f]]
    if bad:
        rep.error(f"no frame decodes at {' and '.join(bad)} of the video (a cut-off or damaged file?)")
    if dur is None:
        rep.error("its length cannot be read (no frame count from cv2, no duration from ffprobe)")
    elif not opts.min_minutes * 60.0 <= dur <= opts.max_minutes * 60.0:
        rep.error(f"{_hms(dur)} is not the length of a whole game ({opts.min_minutes:g} to {opts.max_minutes:g} "
                  "minutes; --min-minutes / --max-minutes)")
    if w and h and abs(w / h - 16.0 / 9.0) > 0.02:
        rep.warn(f"{w}x{h} is not 16:9: the scorebug and court stages expect a 16:9 broadcast frame")


# --------------------------------------------------------------------------- finding the ESPN game


def _game_of(g: dict[str, Any], day: str) -> Game:
    return Game(g["event"], g.get("date") or day, g["away"]["code"], g["home"]["code"],
                {"away": g["away"]["name"], "home": g["home"]["name"]}, g["completed"], g["status"], g["national_tv"])


def manual_fallback(spec: Spec, day: str, teams: tuple[str, str] | None) -> str:
    t = f"{teams[0]},{teams[1]}" if teams else "AWY,HOM"
    how = ("add an espn_event column to the row" if spec.where else
           f"{PYTHON} tools/nba_new_game.py {shlex.quote(spec.video)} --date {day} --teams {t} --espn-event <id>")
    return ("Without network, give the ESPN event id by hand (the number after gameId in the game's espn.com page "
            f"URL, espn.com/nba/game/_/gameId/<id>): {how}")


def find_game(spec: Spec, hints: Hints, video: Path | None, opts: Options, net: Net, rep: Report,
              ask: Callable[[str], str]) -> Game | None:
    try:
        date = parse_date(spec.date) if spec.date else None
        teams = espn.parse_matchup(spec.teams) if spec.teams else None
    except ValueError as exc:
        rep.error(str(exc))
        return None
    if date is not None and hints.dates and all(abs((date - d).days) > 1 for d, _ in hints.dates):
        rep.warn(f"{hints.dates[0][1]} says {hints.dates[0][0]}, not {date} (--date): is it this game's video?")
    if teams is not None and hints.teams and set(hints.teams) != set(teams):
        rep.warn(f"{hints.teams_from} names {hints.teams[0]}-{hints.teams[1]}, not {teams[0]}-{teams[1]} (--teams): "
                 "is it this game's video?")
    if spec.espn_event:
        return game_of_event(spec.espn_event.strip(), date, teams, hints, opts, net, rep, ask)
    if date is not None:
        days = [(date, "")]
    else:
        guessed = list(hints.dates)
        if not guessed and video is not None:
            md = metadata_date(video)
            guessed = [(md, "the video's creation time")] if md else []
        if not guessed:
            rep.error("no date: give --date YYYY-MM-DD (none in the file name or the video's metadata)")
            return None
        days = []
        for d, src in guessed + [(d + dt.timedelta(days=k), f"{src}, a day {'later' if k > 0 else 'earlier'}")
                                 for d, src in guessed for k in (-1, 1)]:
            if all(d != e for e, _ in days):  # the dates as written first, then the days around them
                days.append((d, src))
    teams_from = ""
    if teams is None and hints.teams is not None:
        teams, teams_from = hints.teams, hints.teams_from
    tried: list[str] = []
    found: tuple[dict[str, Any], str, str] | None = None
    for day, src in days:
        try:
            board = net.scoreboard(day.isoformat())
        except Offline as exc:
            rep.error(f"{exc}: cannot look the game up on ESPN. {manual_fallback(spec, day.isoformat(), teams)}")
            return None
        except FetchError as exc:
            tried.append(f"{day}: ESPN scoreboard {exc}")
            continue
        games = [g for g in espn.scoreboard_games(board) if g["away"] and g["home"]]
        if teams is None:
            pick = pick_game(games, day.isoformat(), src, opts, rep, ask)
            if pick is None:
                return None
            found = (pick, day.isoformat(), src)
            teams_from = "the list of the day's games"
            break
        hits = espn.find_games(games, teams)
        if len(hits) > 1:
            rep.error(f"ESPN lists {len(hits)} {teams[0]}-{teams[1]} games on {day}: give --espn-event")
            return None
        if hits:
            found = (dict(hits[0]), day.isoformat(), src)
            break
        tried.append(f"{day}: " + (", ".join(_matchup(g) for g in games) or "no games"))
    if found is None:
        what = f"{teams[0]}-{teams[1]} game" if teams else "game"
        rep.error(f"ESPN lists no {what} ({'; '.join(tried)})")
        return None
    g, on, src = found
    if not (g["away"]["code"] and g["home"]["code"]):
        rep.error(f"{_matchup(g)} is not a game between two NBA teams")
        return None
    game = _game_of(g, on)
    game.date_from, game.teams_from = src, teams_from
    if game.date != on:
        rep.warn(f"ESPN's scoreboard of {on} lists the game, which tipped off on {game.date} (US Eastern): the "
                 f"folder takes {game.date}")
    if src and ", a day " in src:
        rep.warn(f"no {game.away}-{game.home} game on the date {src.split(',')[0]} gives; ESPN has one on {game.date}")
    if spec.teams and teams is not None and teams != (game.away, game.home):
        rep.warn(f"ESPN has {game.away} at {game.home}: the folder follows ESPN's away and home")
    return game


def pick_game(games: list[dict[str, Any]], day: str, src: str, opts: Options, rep: Report,
              ask: Callable[[str], str]) -> dict[str, Any] | None:
    listing = [_matchup(g) for g in games]
    if not games:
        rep.error(f"ESPN lists no NBA game on {day}" + (f" (the date from {src})" if src else ""))
        return None
    if opts.interactive:
        rep.flush()
        print(f"Which game of {day} is the video?" + (f" (the date from {src})" if src else ""))
        for k, line in enumerate(listing, 1):
            print(f"  {k}) {line}")
        try:
            ans = ask(f"number 1-{len(games)} (Enter: none): ").strip()
        except EOFError:
            ans = ""
        if ans.isdigit() and 1 <= int(ans) <= len(games):
            return games[int(ans) - 1]
        rep.error("no game picked")
        return None
    rep.error(f"which game? give --teams AWY,HOM. ESPN lists on {day}: {'; '.join(listing)}")
    return None


def game_of_event(event: str, date: dt.date | None, teams: tuple[str, str] | None, hints: Hints, opts: Options,
                  net: Net, rep: Report, ask: Callable[[str], str]) -> Game | None:
    if not event.isdigit():
        rep.error(f"--espn-event {event!r}: an ESPN event id is a number (espn.com/nba/game/_/gameId/<id>)")
        return None
    try:
        summary = net.summary(event)
    except FetchError as exc:
        rep.error(f"ESPN event {event}: {exc}")
        return None
    except Offline as exc:
        return unchecked_game(event, date, teams, hints, opts, rep, ask, str(exc))
    g = espn.summary_game(summary)
    if not (g["away"] and g["home"] and g["away"]["code"] and g["home"]["code"]):
        rep.error(f"ESPN event {event} is not a game between two NBA teams")
        return None
    game = _game_of(g, "")
    game.summary = summary
    if date is not None and game.date and game.date != date.isoformat():
        rep.error(f"ESPN event {event} was on {game.date}, not {date} (--date)")
    if teams is not None and set(teams) != {game.away, game.home}:
        rep.error(f"ESPN event {event} is {game.away} @ {game.home}, not {teams[0]}-{teams[1]} (--teams)")
    if date is None and hints.dates and game.date:
        near = {d + dt.timedelta(days=k) for d, _ in hints.dates for k in (-1, 0, 1)}
        if dt.date.fromisoformat(game.date) not in near:
            said, src = hints.dates[0]
            rep.error(f"{src} says {said}, but ESPN event {event} was on {game.date}: check the event id (or give "
                      "--date)")
    if teams is None and hints.teams and set(hints.teams) != {game.away, game.home}:
        rep.error(f"{hints.teams_from} names {hints.teams[0]}-{hints.teams[1]}, but ESPN event {event} is "
                  f"{game.away} @ {game.home}: check the event id (or give --teams)")
    if not game.date:
        d = date or (hints.dates[0][0] if hints.dates else None)
        if d is None:
            rep.error(f"ESPN event {event} gives no date: give --date")
            return None
        game.date = d.isoformat()
        rep.warn(f"ESPN's summary gives no date: the folder takes {game.date}")
    if teams is not None and teams != (game.away, game.home) and set(teams) == {game.away, game.home}:
        rep.warn(f"ESPN has {game.away} at {game.home}: the folder follows ESPN's away and home")
    return game


def unchecked_game(event: str, date: dt.date | None, teams: tuple[str, str] | None, hints: Hints, opts: Options,
                   rep: Report, ask: Callable[[str], str], why: str) -> Game | None:
    """--espn-event while ESPN cannot be reached: the game as given (date and teams by hand or from the names)."""
    if opts.check:
        rep.error(f"{why}: ESPN event {event} cannot be checked without network")
        return None
    d = date or (hints.dates[0][0] if hints.dates else None)
    t = teams or hints.teams
    if d is None or t is None:
        rep.error(f"{why}. Without network give --date YYYY-MM-DD and --teams AWY,HOM with --espn-event: the folder "
                  "is named after them")
        return None
    game = Game(event, d.isoformat(), t[0], t[1], checked=False, date_from="" if date else hints.dates[0][1],
                teams_from="" if teams else hints.teams_from)
    if (game.date_from or game.teams_from) and not opts.yes:
        q = f"ESPN cannot be reached: add ESPN event {event} as {game.date} {game.away} @ {game.home}, as guessed?"
        if opts.interactive:
            rep.flush()
        if not (opts.interactive and _confirm(q, ask)):
            rep.error(f"{why}: the date or teams were guessed from the names and ESPN cannot confirm them: give --date "
                      "and --teams (or --yes)")
            return None
    rep.warn(f"{why}: ESPN event {event} not checked; the loop's espn stage fetches its summary later")
    return game


def describe(game: Game, spec: Spec, hints: Hints, rep: Report) -> None:
    rep.add("date", game.date + (f" (from {game.date_from})" if game.date_from else ""))
    names = f": {game.names['away']} at {game.names['home']}" if game.names else ""
    rep.add("teams", f"{game.away} @ {game.home}{names}" + (f" (from {game.teams_from})" if game.teams_from else ""))
    tv = f"; national TV {', '.join(game.national_tv)}" if game.national_tv else ""
    rep.add("event", f"ESPN {game.event}" + (f", {game.status}" if game.status else "")
            + ("" if game.checked else " (not checked: ESPN unreachable)") + tv)
    if game.completed is False:
        rep.error(f"the game is not final ({game.status or 'status unknown'}): add it once ESPN has its box score")
    tv_name = spec.broadcaster or hints.broadcaster
    measured = "the one hand-measured scorebug profile is the ESPN / ABC 2024-25 bug (scorebug_ocr.py ESPN_2024)"
    if tv_name:
        rep.add("tv", tv_name + ("" if spec.broadcaster else f" (from {hints.broadcaster_from})"))
        if not any(s in tv_name.upper() for s in SCOREBUG_TV):
            rep.warn(f"{measured}: check that the scorebug stage reads this {tv_name} broadcast")
    elif game.national_tv and not any(s in n.upper() for n in game.national_tv for s in SCOREBUG_TV):
        rep.warn(f"ESPN lists the national TV as {', '.join(game.national_tv)} and {measured}: check that the "
                 "scorebug stage reads this recording (--broadcaster names it)")


# --------------------------------------------------------------------------- the folder


def other_games(nba_root: Path, skip: str) -> Iterator[tuple[Path, dict[str, Any]]]:
    """The game folders other than ``skip`` (variants, whose game.json is a link, left out)."""
    for gj in sorted(nba_root.glob("*/game.json")):
        if gj.parent.name == skip or gj.is_symlink():
            continue
        meta = _read_json(gj)
        if isinstance(meta, dict):
            yield gj.parent, meta


def analysis_outputs(root: Path) -> list[str]:
    """What a folder holds besides fetched data and logs (scorebug.jsonl, camera_parts/, detections*/, runs/, ...)."""
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return []
    return [p.name + ("/" if p.is_dir() else "") for p in entries
            if not p.name.startswith(".") and not p.name.startswith(DATA_PREFIXES)]


def _few(names: list[str]) -> str:
    return ", ".join(names[:5]) + (f" and {len(names) - 5} more" if len(names) > 5 else "")


def plan_folder(game: Game, spec: Spec, video: Path, opts: Options, rep: Report,
                seen: dict[tuple[str, str], str]) -> Plan | None:
    name = spec.name or f"{game.date}-{game.away}-{game.home}"
    if not FOLDER_RE.match(name):
        rep.error(f"folder name {name!r}: letters, digits, '.', '_' and '-' only (one folder, no path)")
        return None
    if spec.name and not name.startswith(game.date):
        rep.warn(f"{name} does not start with the date: the loop holds out the newest games by folder name")
    root = opts.nba_root / name
    vkey = _real(video)
    dup = [f"the same {what} as {seen[key]}" for key, what in ((("folder", name), "folder"),
                                                                (("event", game.event), f"ESPN event ({game.event})"),
                                                                (("video", vkey), "video")) if key in seen]
    if dup:
        for d in dup:
            rep.error(d)
        return None
    for other, meta in other_games(opts.nba_root, name):
        same_event = str(meta.get("espn_event") or "") == game.event
        same_video = bool(meta.get("video")) and _real(meta["video"]) == vkey
        if same_event and same_video:
            rep.add("note", f"already added as {show(other)}")
            return Plan(other, "exists", meta)
        what = "ESPN event" if same_event else "video" if same_video else None
        if what is None:
            continue
        text = f"this {what} is already {show(other)}"
        if opts.force:
            rep.warn(f"{text}: adding a second folder (--force); the loop treats it as a variant")
        else:
            rep.error(f"{text} (--force to add a second folder, a variant the loop leaves alone)")
    gj = root / "game.json"
    if gj.is_symlink():
        rep.error(f"{show(gj)} is a link (a variant folder): change the game.json it links to")
        return None
    if gj.exists():
        old = _read_json(gj)
        old = old if isinstance(old, dict) else {}
        same_video = bool(old.get("video")) and _real(old["video"]) == vkey
        same_event = str(old.get("espn_event") or "") == game.event
        if same_video and same_event:
            return Plan(root, "exists", old)
        was = f"video {old.get('video')}, ESPN event {old.get('espn_event')}"
        if not opts.force:
            rep.error(f"{show(gj)} exists ({was}): --force to overwrite it")
            return None
        outputs = analysis_outputs(root)
        if outputs:
            changed = " and the ".join(w for w, same in (("video", same_video), ("ESPN event", same_event)) if not same)
            rep.error(f"--force refused: the {changed} would change under the analysis outputs in {show(root)} "
                      f"({_few(outputs)}); add the game under another --name or move them away (for a moved copy "
                      "of the same recording, edit \"video\" in game.json by hand)")
            return None
        rep.warn(f"overwriting {show(gj)} ({was}) (--force)")
        return Plan(root, "overwrite", old)
    outputs = analysis_outputs(root) if root.is_dir() else []
    if outputs:
        if not opts.force:
            rep.error(f"{show(root)} has no game.json but holds {_few(outputs)}: move them away, or --force to add "
                      "the game to it")
            return None
        rep.warn(f"{show(root)} has no game.json but holds {_few(outputs)}: kept (--force)")
    return Plan(root, "new")


# --------------------------------------------------------------------------- writing


def write_game_json(plan: Plan, game: Game, video: Path, spec: Spec, hints: Hints, nba_game_id: str | None,
                    rep: Report) -> None:
    doc: dict[str, Any] = {"video": str(video), "espn_event": game.event, "date": game.date,
                           "teams": {"away": game.away, "home": game.home}}
    tv_name = spec.broadcaster or hints.broadcaster
    if tv_name:
        doc["broadcaster"] = tv_name
    if game.national_tv:
        doc["national_tv"] = game.national_tv
    if nba_game_id:
        doc["nba_game_id"] = nba_game_id
    if plan.state == "overwrite":
        rest = {k: v for k, v in plan.old.items() if k not in GAME_KEYS}
        if str(plan.old.get("espn_event") or "") == game.event:
            doc.update(rest)  # team colours, "loop": false, ...: still this game's
        elif rest:
            rep.warn(f"dropped {', '.join(sorted(rest))} of the old game.json (another game's)")
    plan.root.mkdir(parents=True, exist_ok=True)
    write_text_atomic(plan.root / "game.json", json.dumps(doc, indent=1) + "\n")
    rep.add("wrote", show(plan.root / "game.json"))


def retire_stale(plan: Plan, game: Game, rep: Report) -> None:
    """--force onto another event: its espn_summary.json and nba_pbp.csv are renamed out of the loop's way."""
    old_event = str(plan.old.get("espn_event") or "")
    if plan.state != "overwrite" or old_event == game.event:
        return
    for f in ("espn_summary.json", "nba_pbp.csv"):
        p = plan.root / f
        if p.exists():
            stale = p.with_name(f"{f}.stale-{old_event or 'unknown'}")
            os.replace(p, stale)
            rep.add("moved", f"{show(p)} -> {stale.name} (the old event's)")


def summary_problem(summary: dict[str, Any], game: Game) -> str | None:
    """Why an ESPN summary must not become espn_summary.json (the loop never fetches it again once it exists)."""
    g = espn.summary_game(summary)
    if g["event"] and g["event"] != game.event:
        return f"it is the summary of event {g['event']}"
    codes = {s["code"] for s in (g["away"], g["home"]) if s}
    if codes != {game.away, game.home}:
        return f"its teams are {'/'.join(sorted(str(c) for c in codes))}, not {game.away}/{game.home}"
    if g["completed"] is False:
        return f"the game is not final ({g['status'] or 'status unknown'})"
    if not summary.get("plays"):
        return "it has no play-by-play yet"
    try:
        espn.box_score(summary)
        espn.pbp_actions(summary)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        return f"its box score or play-by-play does not parse ({type(exc).__name__}: {exc})"
    return None


def write_summary(plan: Plan, game: Game, net: Net, rep: Report) -> None:
    out = plan.root / "espn_summary.json"
    if out.is_file():
        old = _read_json(out)
        old_id = str(((old.get("header") or {}) if isinstance(old, dict) else {}).get("id") or "")
        if old_id and old_id != game.event:
            rep.warn(f"{show(out)} is the summary of ESPN event {old_id}, not {game.event}: kept; delete it and run "
                     "again to fetch this game's")
        else:
            rep.add("kept", show(out))
        return
    if not game.checked:
        rep.add("later", "espn_summary.json: the loop's espn stage fetches it (ESPN unreachable now)")
        return
    summary = game.summary
    if summary is None:
        try:
            summary = net.summary(game.event)
        except (Offline, FetchError) as exc:
            rep.warn(f"espn_summary.json not fetched ({exc}): the loop's espn stage fetches it")
            return
    problem = summary_problem(summary, game)
    if problem:
        rep.warn(f"espn_summary.json not written: {problem}")
        return
    write_text_atomic(out, json.dumps(summary))
    rep.add("wrote", f"{show(out)} ({len(summary.get('plays') or [])} plays)")


def find_nba_game_id(game: Game, given: str | None, net: Net, rep: Report) -> str | None:
    if given:
        if nba_stats.GAME_ID_RE.match(given.strip()):
            return given.strip()
        rep.warn(f"NBA.com game id {given.strip()!r} is not 10 digits: ignored")
    tried: list[str] = []
    sources: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
        ("the NBA.com schedule", net.nba_schedule),
        ("the stats.nba.com scoreboard (v3)", lambda: net.nba_scoreboard(game.date, 3)),
        ("the stats.nba.com scoreboard (v2)", lambda: net.nba_scoreboard(game.date, 2)))
    for what, get in sources:
        try:
            doc = get()
        except (Offline, FetchError) as exc:
            tried.append(f"{what}: {exc}")
            continue
        gid = nba_stats.find_game_id(doc, game.date, (game.away, game.home))
        if gid:
            rep.add("nba.com", f"game {gid} (from {what})")
            return gid
        tried.append(f"{what}: no {game.away}-{game.home} game on {game.date}")
    rep.warn(f"NBA.com game id not found ({'; '.join(tried)}): nba_pbp.csv not written; give --nba-game-id (the "
             "10-digit number at the end of the game's nba.com page URL)")
    return None


def write_nba_pbp(plan: Plan, game: Game, gid: str, net: Net, rep: Report) -> None:
    out = plan.root / "nba_pbp.csv"
    try:
        pbp = net.nba_pbp(gid)
    except (Offline, FetchError) as exc:
        rep.warn(f"nba_pbp.csv not written: NBA.com play-by-play of game {gid}: {exc}")
        return
    got = pbp.get("game")
    body: dict[str, Any] = got if isinstance(got, dict) else {}
    actions = [a for a in body.get("actions") or [] if isinstance(a, dict)]
    tricodes = {str(a["teamTricode"]) for a in actions if a.get("teamTricode")}
    if body.get("gameId") and str(body["gameId"]) != gid:
        rep.warn(f"nba_pbp.csv not written: NBA.com sent game {body['gameId']} for {gid}")
        return
    if not actions:
        rep.warn(f"nba_pbp.csv not written: the NBA.com play-by-play of game {gid} has no actions")
        return
    if tricodes != {game.away, game.home}:
        rep.warn(f"nba_pbp.csv not written: NBA.com game {gid} is {'/'.join(sorted(tricodes))}, not "
                 f"{game.away}/{game.home}")
        return
    rows = nba_stats.export_pbp_csv(pbp, out)
    located = sum(1 for r in rows if r["type"] in ("2", "3") and r["x"] != "")
    rep.add("wrote", f"{show(out)} (NBA.com game {gid}: {len(rows)} actions, {located} field goals with a location)")
    codes = nba_stats.csv_team_codes(rows)
    if codes.get(game.home) is not Team.HOME or codes.get(game.away) is not Team.AWAY:
        got = ", ".join(f"{c} {t.value}" for c, t in sorted(codes.items())) or "no clear sides"
        rep.warn(f"the running score of nba_pbp.csv does not give {game.home} as the home team ({got}): load it with "
                 f"load_shots(path, home_code={game.home!r}, away_code={game.away!r})")


def next_command(root: Path, opts: Options) -> str:
    cmd = [PYTHON, "tools/nba_loop.py", "run"]
    if os.path.realpath(opts.nba_root) != os.path.realpath(DEFAULT_NBA_ROOT):
        cmd += ["--runs", str(opts.nba_root.parent)]
    return " ".join(shlex.quote(c) for c in [*cmd, "--game", root.name])


# --------------------------------------------------------------------------- one game


def process(spec: Spec, opts: Options, net: Net, seen: dict[tuple[str, str], str],
            ask: Callable[[str], str]) -> Report:
    """Check and (unless --check) add one game; the report says what happened."""
    rep = Report(spec.where or spec.video)
    if spec.problem:
        rep.error(spec.problem)
        return rep
    video = local_video(spec, rep)
    hints = read_hints(video if video is not None else Path(os.path.expanduser(spec.video.strip())))
    if video is not None:
        report_video(video, probe_video(video), opts, rep)
    game = find_game(spec, hints, video, opts, net, rep, ask)
    if game is None:
        return rep
    describe(game, spec, hints, rep)
    if video is None or rep.errors:
        return rep
    plan = plan_folder(game, spec, video, opts, rep, seen)
    if plan is None or rep.errors:
        return rep
    rep.title = plan.root.name + (f" ({spec.where})" if spec.where else "")
    who = spec.where or plan.root.name
    for key in (("folder", plan.root.name), ("event", game.event), ("video", _real(video))):
        seen.setdefault(key, who)
    if opts.check:
        state = {"new": "new", "exists": "already added", "overwrite": "game.json would be overwritten (--force)"}
        rep.add("folder", f"{show(plan.root)}: {state[plan.state]}")
        rep.status = "valid"
        return rep
    if (game.date_from or game.teams_from) and game.checked and opts.interactive and plan.state != "exists":
        q = (f"Add {plan.root.name}: {game.names.get('away', game.away)} at {game.names.get('home', game.home)} on "
             f"{game.date}, ESPN event {game.event}?")
        rep.flush()
        if not _confirm(q, ask):
            rep.error("not added (not confirmed)")
            return rep
    retire_stale(plan, game, rep)
    gid = None
    if opts.nba_pbp and not (plan.root / "nba_pbp.csv").is_file():
        gid = find_nba_game_id(game, spec.nba_game_id or plan.old.get("nba_game_id"), net, rep)
    if plan.state == "exists":
        rep.add("kept", f"{show(plan.root / 'game.json')} (already added: same video and ESPN event)")
    else:
        write_game_json(plan, game, video, spec, hints, gid, rep)
    write_summary(plan, game, net, rep)
    if opts.nba_pbp:
        if (plan.root / "nba_pbp.csv").is_file():
            rep.add("kept", show(plan.root / "nba_pbp.csv"))
        elif gid:
            write_nba_pbp(plan, game, gid, net, rep)
    rep.add("next", next_command(plan.root, opts))
    rep.status = "exists" if plan.state == "exists" else "added"
    return rep


# --------------------------------------------------------------------------- batch file and command line


def read_batch(path: Path) -> list[Spec]:
    """The games of a --batch CSV (module docstring); ValueError when its header is not usable."""
    rows: list[tuple[int, list[str]]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip() and not line.lstrip().startswith("#"):
            rows.append((lineno, next(csv.reader([line]))))
    if not rows:
        raise ValueError(f"{path}: empty (a header row such as video,date,teams, then one game per row)")
    header = [h.strip().lower() for h in rows[0][1]]
    unknown = [h for h in header if h not in BATCH_COLUMNS]
    if "video" not in header or unknown:
        raise ValueError(f"{path}: the header names columns of {', '.join(BATCH_COLUMNS)} (video required)"
                         + (f"; unknown: {', '.join(unknown)}" if unknown else ""))
    specs = []
    base = path.parent.absolute()
    for lineno, cells in rows[1:]:
        r = {h: (cells[i].strip() if i < len(cells) else "") for i, h in enumerate(header)}
        teams = r.get("teams") or (f"{r['away']}@{r['home']}" if r.get("away") and r.get("home") else "")
        spec = Spec(r["video"], r.get("date") or None, teams or None, r.get("espn_event") or None,
                    r.get("name") or None, r.get("broadcaster") or None, r.get("nba_game_id") or None,
                    where=f"{path.name} line {lineno}", base=base)
        if len(cells) > len(header):
            spec.problem = (f"{len(cells)} cells for {len(header)} columns (put \"AWY,HOM\" in quotes, or write "
                            "AWY@HOM)")
        elif not spec.video:
            spec.problem = "no video"
        specs.append(spec)
    return specs


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", help="the game's video: a local file (never downloaded, never copied)")
    ap.add_argument("--date", help="the game's date, YYYY-MM-DD (US Eastern, as ESPN lists it); default: guessed")
    ap.add_argument("--teams", help="AWY,HOM: NBA.com or ESPN abbreviations or names (also AWY@HOM); default: guessed")
    ap.add_argument("--espn-event", help="the ESPN event id: no scoreboard search (the way in without network)")
    ap.add_argument("--name", help="the folder name (default <date>-<AWY>-<HOM>)")
    ap.add_argument("--nba-root", default=None, help="the game folders (default: <repo>/runs/nba)")
    ap.add_argument("--broadcaster",
                    help="who produced the broadcast (ESPN, ABC, TNT, ...); default: one the file name names")
    ap.add_argument("--nba-game-id", help="the NBA.com game id (10 digits, the end of the nba.com game page URL)")
    ap.add_argument("--no-nba-pbp", action="store_true", help="do not fetch the NBA.com play-by-play (nba_pbp.csv)")
    ap.add_argument("--batch", help="a CSV of games: video,date,teams[,espn_event,name,broadcaster,nba_game_id]")
    ap.add_argument("--check", action="store_true", help="validate only (video, ESPN event, folder): write nothing")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing game.json; add a game that another folder already has")
    ap.add_argument("--yes", "-y", action="store_true", help="do not ask before adding a game whose date or teams "
                                                             "were guessed")
    ap.add_argument("--min-minutes", type=float, default=40.0, help="shortest plausible video of a whole game")
    ap.add_argument("--max-minutes", type=float, default=360.0, help="longest plausible video of one game")
    ap.add_argument("--timeout", type=float, default=30.0, help="seconds per network request")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None, ask: Callable[[str], str] | None = None) -> int:
    args = parse_args(argv)
    nba_root = Path(os.path.abspath(os.path.expanduser(args.nba_root))) if args.nba_root else DEFAULT_NBA_ROOT
    single = [f"--{n.replace('_', '-')}" for n in ("date", "teams", "espn_event", "name", "broadcaster", "nba_game_id")
              if getattr(args, n)]
    if args.batch:
        if args.video or single:
            print(f"nba_new_game: --batch takes its games from the file: no VIDEO or {', '.join(single) or '--date'}",
                  file=sys.stderr)
            return 2
        try:
            specs = read_batch(Path(os.path.expanduser(args.batch)))
        except (OSError, ValueError) as exc:
            print(f"nba_new_game: {exc}", file=sys.stderr)
            return 2
    elif not args.video:
        print("nba_new_game: give a VIDEO (or --batch CSV); --help for more", file=sys.stderr)
        return 2
    else:
        specs = [Spec(args.video, args.date, args.teams, args.espn_event, args.name, args.broadcaster,
                      args.nba_game_id)]
    if nba_root.name != "nba":
        print(f"warning: the loop reads <runs>/nba; {nba_root} is not such a folder", file=sys.stderr)
    opts = Options(nba_root, check=args.check, force=args.force, nba_pbp=not args.no_nba_pbp, yes=args.yes,
                   interactive=not (args.batch or args.check or args.yes) and _interactive(),
                   min_minutes=args.min_minutes, max_minutes=args.max_minutes)
    net = Net(args.timeout)
    seen: dict[tuple[str, str], str] = {}
    reports = []
    for spec in specs:
        rep = process(spec, opts, net, seen, ask or input)
        rep.flush()
        reports.append(rep)
    if args.batch:
        counts = Counter(r.status for r in reports)
        print(f"{args.batch}: {len(reports)} games: "
              + ", ".join(f"{counts[s]} {s.replace('exists', 'already added')}" for s in
                          ("added", "exists", "valid", "failed") if counts[s]))
        if counts["added"] and not args.check:
            print(f"next: {PYTHON} tools/nba_loop.py run   (every game's missing stages, then the fits)")
    return 0 if all(r.status != "failed" for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
