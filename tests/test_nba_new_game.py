"""Adding a game to the NBA loop (tools/nba_new_game.py) and the lookups it uses (nba/espn.py, nba/nba_stats.py).

No network: ``urllib.request.urlopen`` is replaced by ``FakeWeb``, which answers with the hand-made fixtures in
tests/fixtures/nba (ESPN scoreboard and summary, NBA.com schedule, scoreboards and live play-by-play, in the shapes
of those APIs; the games, ids and players in them are made up). Videos are a few frames written with cv2."""

from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import pytest

from courtvision.nba import espn, nba_stats
from courtvision.nba import loop as nl
from courtvision.training.nba_cue_rows import infer_nba_codes
from courtvision.types import Team

TOOLS = Path(__file__).resolve().parents[1] / "tools"
FIX = Path(__file__).resolve().parent / "fixtures" / "nba"
G = "2024-12-25-LAL-GSW"
H, A = Team.HOME, Team.AWAY


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tool = _load("nba_new_game")


def load(path: Path) -> Any:
    return json.loads(path.read_text())


class FakeWeb:
    """``urllib.request.urlopen`` for the tests: the first route whose key is in the URL answers (a fixture file, a
    JSON-able object, or an exception to raise); any other URL is a 404. Records every URL and its headers."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = dict(routes)
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []

    def __call__(self, req: Any, timeout: float | None = None, **_: Any) -> io.BytesIO:
        url = req.full_url if isinstance(req, urllib.request.Request) else str(req)
        self.urls.append(url)
        self.headers.append(dict(req.header_items()) if isinstance(req, urllib.request.Request) else {})
        for key, answer in self.routes.items():
            if key in url:
                if isinstance(answer, BaseException):
                    raise answer
                if isinstance(answer, bytes):  # a body as sent (not JSON)
                    return io.BytesIO(answer)
                return io.BytesIO(answer.read_bytes() if isinstance(answer, Path) else json.dumps(answer).encode())
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)  # type: ignore[arg-type]


ROUTES = {
    "nba/scoreboard?dates=20241225": FIX / "espn_scoreboard_20241225.json",
    "nba/scoreboard?dates=": {"leagues": [], "events": []},  # ESPN: any other day, no games
    "summary?event=401705034": FIX / "espn_summary_401705034.json",
    "scheduleLeagueV2.json": FIX / "nba_schedule.json",
    "scoreboardv3?GameDate=2025-01-10": FIX / "nba_scoreboardv3_20250110.json",
    "scoreboardv2?GameDate=2025-01-10": FIX / "nba_scoreboardv2_20250110.json",
    "playbyplay_0022400447.json": FIX / "nba_pbp_0022400447.json",
}


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch) -> FakeWeb:
    fake = FakeWeb(ROUTES)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


def make_video(path: Path, frames: int = 30, fps: float = 10.0, size: tuple[int, int] = (64, 36)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert w.isOpened()
    for k in range(frames):
        w.write(np.full((size[1], size[0], 3), (k * 8) % 256, np.uint8))
    w.release()
    return path


def run(argv: list[Any], capsys: pytest.CaptureFixture[str], ask: Any = None) -> tuple[int, str]:
    rc = tool.main([str(a) for a in argv], ask=ask)
    return rc, capsys.readouterr().out


# --------------------------------------------------------------------------- teams


def test_every_team_by_tricode_espn_abbreviation_and_name() -> None:
    assert len(espn.TEAMS) == 30
    for tri, (abbr, city, nick, _) in espn.TEAMS.items():
        assert espn.team_code(tri) == espn.team_code(abbr) == espn.team_code(f"{city} {nick}") == tri
        assert espn.team_code(nick.lower()) == tri and espn.team_code(f" {tri.lower()} ") == tri


@pytest.mark.parametrize(("written", "tri"), [
    ("GS", "GSW"), ("gsw", "GSW"), ("Golden State", "GSW"), ("NY", "NYK"), ("Knicks", "NYK"), ("SA", "SAS"),
    ("NO", "NOP"), ("NOH", "NOP"), ("UTAH", "UTA"), ("utah jazz", "UTA"), ("WSH", "WAS"), ("PHO", "PHX"),
    ("BRK", "BKN"), ("CHO", "CHA"), ("Sixers", "PHI"), ("76ers", "PHI"), ("LA Clippers", "LAC"),
    ("L.A. Clippers", "LAC"), ("Los Angeles Clippers", "LAC"), ("LA Lakers", "LAL"), ("Trail Blazers", "POR"),
    ("Blazers", "POR"), ("Cavs", "CLE"), ("Wolves", "MIN"), ("Oklahoma City Thunder", "OKC")])
def test_team_aliases(written: str, tri: str) -> None:
    assert espn.team_code(written) == tri


def test_unknown_and_ambiguous_teams_are_refused() -> None:
    for text in ("Los Angeles", "LA", "XYZ", "", "Seattle"):
        with pytest.raises(ValueError):
            espn.team_code(text)


def test_matchups_in_any_spelling() -> None:
    for text in ("LAL,GSW", "LAL@GS", "LAL @ GSW", "Lakers at Warriors", "lal vs gsw", "LAL GSW", "LAL-GSW",
                 "Los Angeles Lakers @ Golden State Warriors", "Los Angeles Lakers Golden State Warriors"):
        assert espn.parse_matchup(text) == ("LAL", "GSW"), text
    assert espn.parse_matchup("NY,SA") == ("NYK", "SAS")
    for bad in ("LAL,LAL", "LAL", "Los Angeles,GSW", "LAL,XYZ", "LAL,GSW,BOS"):
        with pytest.raises(ValueError):
            espn.parse_matchup(bad)


# --------------------------------------------------------------------------- ESPN scoreboard and summary


def test_scoreboard_games_and_matching() -> None:
    games = espn.scoreboard_games(load(FIX / "espn_scoreboard_20241225.json"))
    assert [(g["event"], g["away"]["code"], g["home"]["code"]) for g in games] == [
        ("401705030", "SAS", "NYK"), ("401705031", "MIN", "DAL"), ("401705032", "PHI", "BOS"),
        ("401705034", "LAL", "GSW"), ("401705033", "DEN", "PHX")]
    # every game is on the US date of the scoreboard, also the 10:30 pm ET tip-off (03:30Z the next day)
    assert {g["date"] for g in games} == {"2024-12-25"} and all(g["completed"] for g in games)
    lal = espn.find_games(games, ("GSW", "LAL"))  # either order
    assert [g["event"] for g in lal] == ["401705034"] and lal[0]["national_tv"] == ["ABC", "ESPN"]
    assert lal[0]["home"] == {"abbr": "GS", "code": "GSW", "name": "Golden State Warriors", "score": "7"}
    assert [g["event"] for g in espn.find_games(games, ("NYK", "SAS"))] == ["401705030"]  # ESPN's NY and SA
    assert espn.find_games(games, ("BOS", "NYK")) == []


def test_summary_game_and_us_dates() -> None:
    g = espn.summary_game(load(FIX / "espn_summary_401705034.json"))
    assert (g["event"], g["date"], g["away"]["code"], g["home"]["code"], g["completed"], g["status"]) == (
        "401705034", "2024-12-25", "LAL", "GSW", True, "Final")
    assert g["national_tv"] == ["ABC", "ESPN"]  # the summary's broadcasts shape (market / media)
    # the summary shape tests/test_nba_truth.py uses (no header id, date or status) still reads
    bare = espn.summary_game({"header": {"competitions": [{"competitors": [
        {"homeAway": "away", "team": {"id": "1", "abbreviation": "NY"}},
        {"homeAway": "home", "team": {"id": "2", "abbreviation": "SA"}}]}]}})
    assert (bare["away"]["code"], bare["home"]["code"], bare["date"], bare["completed"]) == ("NYK", "SAS", None, None)
    assert espn.us_date("2024-12-26T03:30Z") == "2024-12-25" and espn.us_date("2024-12-25T17:00Z") == "2024-12-25"
    assert espn.us_date("2025-04-02T02:30Z") == "2025-04-01"  # daylight saving time (EDT)
    assert espn.us_date("not a date") is None


# --------------------------------------------------------------------------- names and dates


@pytest.mark.parametrize(("name", "dates"), [
    ("2024-12-25 LAL @ GSW.mp4", ["2024-12-25"]),
    ("NBA_20241225_LALGSW_1080p.mkv", ["2024-12-25"]),
    ("lakers-vs-warriors-12-25-2024.mp4", ["2024-12-25"]),
    ("LAL GSW 12.25.24.mp4", ["2024-12-25"]),
    ("Lakers at Warriors Dec 25, 2024.mp4", ["2024-12-25"]),
    ("Lakers Warriors 25th December 2024.ts", ["2024-12-25"]),
    ("2024_12_25 full game.mov", ["2024-12-25"]),
    ("05-12-2024 game.mp4", ["2024-05-12", "2024-12-05"]),  # US first, then the day-month reading
    ("game1.mp4", []), ("401705034.mp4", []), ("0022400447 29.97fps.mp4", []), ("1920x1080 x264.mkv", [])])
def test_dates_in_file_names(name: str, dates: list[str]) -> None:
    assert [d.isoformat() for d in tool.guess_dates(name)] == dates


@pytest.mark.parametrize(("name", "teams"), [
    ("2024-12-25 LAL @ GSW.mp4", ["LAL", "GSW"]),
    ("lakers-vs-warriors-12-25-2024.mp4", ["LAL", "GSW"]),
    ("NBA_20241225_LALGSW.mkv", ["LAL", "GSW"]),
    ("Los Angeles Lakers at Golden State Warriors.mp4", ["LAL", "GSW"]),
    ("Spurs @ Knicks (SA at NY).mp4", ["SAS", "NYK"]),
    ("no commentary BOS NYK.mp4", ["BOS", "NYK"]),  # a lower-case "no" is a word, not New Orleans
    ("NO at MIN.mp4", ["NOP", "MIN"]),
    ("it was 90 min of nothing.mp4", []), ("game1.mp4", []), ("Los Angeles 2024.mp4", [])])
def test_teams_in_file_names(name: str, teams: list[str]) -> None:
    assert tool.guess_teams(name) == teams


def test_explicit_dates_broadcasters_and_studio_uploads(tmp_path: Path) -> None:
    assert tool.parse_date("2024-12-25") == tool.parse_date("20241225") == dt.date(2024, 12, 25)
    with pytest.raises(ValueError):
        tool.parse_date("12/25/2024")
    assert tool.guess_broadcaster("LAL GSW ESPN 1080p") == "ESPN"
    assert tool.guess_broadcaster("nba tv replay") == "NBA TV"
    assert tool.guess_broadcaster("lakers warriors") is None
    # a studio upload keeps its original name in video.json next to source.video
    up = tmp_path / "runs" / "studio" / "0123abcd"
    up.mkdir(parents=True)
    (up / "source.video").write_bytes(b"")
    (up / "video.json").write_text(json.dumps({"id": "0123abcd", "name": "Lakers at Warriors 12-25-2024 ABC.mp4"}))
    h = tool.read_hints(up / "source.video")
    assert h.dates == [(dt.date(2024, 12, 25), "the studio upload's name")]
    assert (h.teams, h.teams_from, h.broadcaster) == (("LAL", "GSW"), "the studio upload's name", "ABC")


# --------------------------------------------------------------------------- adding a game


def test_add_a_game_from_its_file_name(tmp_path: Path, web: FakeWeb, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "runs" / "nba"
    video = make_video(tmp_path / "videos" / "2024-12-25 Lakers at Warriors ESPN.mp4")
    rc, out = run([video, "--nba-root", root, "--min-minutes", "0"], capsys)
    assert rc == 0, out
    folder = root / G
    assert load(folder / "game.json") == {
        "video": str(video), "espn_event": "401705034", "date": "2024-12-25", "teams": {"away": "LAL", "home": "GSW"},
        "broadcaster": "ESPN", "national_tv": ["ABC", "ESPN"], "nba_game_id": "0022400447"}
    assert load(folder / "espn_summary.json") == load(FIX / "espn_summary_401705034.json")
    with (folder / "nba_pbp.csv").open(newline="") as fh:
        assert tuple(csv.DictReader(fh).fieldnames or ()) == nba_stats.CSV_COLUMNS
    assert "2024-12-25 (from the file name)" in out
    assert f"next     .venv/bin/python tools/nba_loop.py run --runs {root.parent} --game {G}" in out
    # JSON APIs only (the video is never fetched), each once; NBA.com as a browser
    assert web.urls == [
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=20241225",
        "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json",
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=401705034",
        "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_0022400447.json"]
    assert "Mozilla" in web.headers[-1]["User-agent"]
    # the loop takes the folder as a game: its ESPN event, the summary there, the video readable
    games = nl.discover_games(root)
    assert [(g.name, g.game_id, g.primary) for g in games] == [(G, "401705034", True)]
    stages = nl.status(nl.Loop(tmp_path / "runs", tmp_path, nl.LoopConfig(python="py")))["games"][0]["stages"]
    assert stages["espn"] == "done" and stages["scorebug"] == "ready"
    # with the default root the next command needs no --runs
    assert tool.next_command(tool.DEFAULT_NBA_ROOT / G, tool.Options(tool.DEFAULT_NBA_ROOT)) == (
        f".venv/bin/python tools/nba_loop.py run --game {G}")


def test_teams_in_another_spelling_and_order(tmp_path: Path, web: FakeWeb, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "runs" / "nba"
    video = make_video(tmp_path / "videos" / "recording.mp4")
    rc, out = run([video, "--date", "2024-12-25", "--teams", "GS,Lakers", "--nba-root", root, "--min-minutes", "0",
                   "--no-nba-pbp", "--broadcaster", "TNT"], capsys)
    assert rc == 0, out
    doc = load(root / G / "game.json")  # ESPN's away and home name the folder
    assert doc["teams"] == {"away": "LAL", "home": "GSW"} and doc["broadcaster"] == "TNT" and "nba_game_id" not in doc
    assert "ESPN has LAL at GSW" in out and "check that the scorebug stage reads this TNT broadcast" in out
    assert not (root / G / "nba_pbp.csv").exists() and not any("nba.com" in u for u in web.urls)


def test_the_same_game_again_and_overwriting(tmp_path: Path, web: FakeWeb, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "runs" / "nba"
    v1, v2, v3 = (make_video(tmp_path / "videos" / f"{n}.mp4") for n in ("a", "b", "c"))
    base = ["--date", "2024-12-25", "--teams", "LAL,GSW", "--nba-root", root, "--min-minutes", "0", "--no-nba-pbp"]
    assert run([v1, *base], capsys)[0] == 0
    gj = root / G / "game.json"
    first = gj.read_text()
    rc, out = run([v1, *base], capsys)  # the same video and event: already there, nothing rewritten
    assert rc == 0 and "already added" in out and gj.read_text() == first
    rc, out = run([v2, *base], capsys)  # another video for this folder: refused
    assert rc == 1 and "exists" in out and "--force to overwrite" in out and gj.read_text() == first
    doc = json.loads(first)
    doc["home_color_bgr"] = [10, 20, 30]  # a key the tool does not write
    gj.write_text(json.dumps(doc))
    rc, out = run([v2, *base, "--force"], capsys)
    new = load(gj)
    assert rc == 0 and new["video"] == str(v2) and new["home_color_bgr"] == [10, 20, 30]  # same event: kept
    (root / G / "scorebug.jsonl").write_text("{}\n")  # an analysis output of the video v2
    rc, out = run([v1, *base, "--force"], capsys)
    assert rc == 1 and "--force refused" in out and "scorebug.jsonl" in out and load(gj)["video"] == str(v2)
    # a second folder of the same event: refused, or a variant with --force
    rc, out = run([v3, *base, "--name", f"{G}-copy"], capsys)
    assert rc == 1 and f"this ESPN event is already {root / G}" in out and not (root / f"{G}-copy").exists()
    rc, out = run([v3, *base, "--name", f"{G}-copy", "--force"], capsys)
    assert rc == 0 and (root / f"{G}-copy" / "game.json").exists() and "variant" in out
    # a variant folder whose game.json is a link (nba_mac_jobs.sh link_v2) is never written through
    (root / f"{G}-v2").mkdir()
    (root / f"{G}-v2" / "game.json").symlink_to(gj)
    rc, out = run([v1, *base, "--name", f"{G}-v2", "--force"], capsys)
    assert rc == 1 and "is a link" in out and load(gj)["video"] == str(v2)
    rc, out = run([v2, *base, "--name", f"{G}-v2"], capsys)  # the video and event of the primary folder
    assert rc == 0 and f"already added as {root / G}" in out
    variants = {g.name: g.variant_of for g in nl.discover_games(root)}
    assert variants == {G: None, f"{G}-copy": G, f"{G}-v2": G}
    # a folder name is one folder
    rc, out = run([v1, *base, "--name", "../elsewhere"], capsys)
    assert rc == 1 and "one folder" in out


def test_force_onto_another_event_moves_its_data_aside(tmp_path: Path, web: FakeWeb,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "runs" / "nba"
    folder = root / G
    folder.mkdir(parents=True)
    video = make_video(tmp_path / "videos" / "a.mp4")
    (folder / "game.json").write_text(json.dumps({"video": str(video), "espn_event": "401700000",
                                                  "away_color_bgr": [1, 2, 3]}))
    (folder / "espn_summary.json").write_text(json.dumps({"header": {"id": "401700000"}}))
    (folder / "nba_pbp.csv").write_text("n,period\n")
    args = [video, "--date", "2024-12-25", "--teams", "LAL@GSW", "--nba-root", root, "--min-minutes", "0"]
    rc, out = run(args, capsys)
    assert rc == 1 and "--force to overwrite" in out
    rc, out = run([*args, "--force"], capsys)
    assert rc == 0, out
    assert load(folder / "espn_summary.json.stale-401700000") == {"header": {"id": "401700000"}}
    assert (folder / "nba_pbp.csv.stale-401700000").read_text() == "n,period\n"
    assert load(folder / "espn_summary.json")["header"]["id"] == "401705034"
    assert (folder / "nba_pbp.csv").read_text().startswith("n,period,clock,team")
    doc = load(folder / "game.json")
    assert doc["espn_event"] == "401705034" and "away_color_bgr" not in doc  # another game's colours
    assert "dropped away_color_bgr" in out


def test_a_game_not_final_or_without_play_by_play(tmp_path: Path, web: FakeWeb,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    video = make_video(tmp_path / "videos" / "a.mp4")

    def args(root: Path) -> list[Any]:
        return [video, "--date", "2024-12-25", "--teams", "LAL,GSW", "--nba-root", root, "--min-minutes", "0",
                "--no-nba-pbp"]

    root = tmp_path / "runs" / "nba"
    summary = load(FIX / "espn_summary_401705034.json")
    summary["plays"] = []
    web.routes["summary?event=401705034"] = summary
    rc, out = run(args(root), capsys)  # added, but the summary waits (the loop fetches it once it exists)
    assert rc == 0 and (root / G / "game.json").exists() and not (root / G / "espn_summary.json").exists()
    assert "espn_summary.json not written: it has no play-by-play yet" in out
    board = load(FIX / "espn_scoreboard_20241225.json")
    for ev in board["events"]:
        for st in (ev.get("status"), ev["competitions"][0].get("status")):
            if st:
                st["type"].update(completed=False, state="pre", name="STATUS_SCHEDULED", description="Scheduled")
    web.routes["nba/scoreboard?dates=20241225"] = board
    rc, out = run(args(tmp_path / "other" / "nba"), capsys)
    assert rc == 1 and "the game is not final (Scheduled)" in out and not (tmp_path / "other").exists()


def test_a_url_or_a_missing_file_is_refused(tmp_path: Path, web: FakeWeb, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "runs" / "nba"
    rc, out = run(["https://example.com/game.mp4", "--date", "2024-12-25", "--teams", "LAL,GSW", "--nba-root", root],
                  capsys)
    assert rc == 1 and "never downloads video" in out and not any("example.com" in u for u in web.urls)
    rc, out = run([tmp_path / "nope.mp4", "--date", "2024-12-25", "--teams", "LAL,GSW", "--nba-root", root], capsys)
    assert rc == 1 and "video not found" in out and not root.exists()
    assert run(["--nba-root", root], capsys)[0] == 2  # no video at all


# --------------------------------------------------------------------------- --check


def test_check_writes_nothing_and_finds_problems(tmp_path: Path, web: FakeWeb,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "runs" / "nba"
    video = make_video(tmp_path / "videos" / "LAL @ GSW 2024-12-25.mp4")
    rc, out = run([video, "--check", "--nba-root", root, "--min-minutes", "0"], capsys)
    assert rc == 0 and f"folder   {root / G}: new" in out and "0:00:03, 64x36 at 10.00 fps" in out
    assert not root.exists() and not any("summary" in u or "nba.com" in u for u in web.urls)
    rc, out = run([video, "--check", "--nba-root", root], capsys)  # 3 s is no whole game
    assert rc == 1 and "is not the length of a whole game (40 to 360 minutes" in out
    junk = tmp_path / "videos" / "junk.mp4"
    junk.write_text("not a video")
    rc, out = run([junk, "--check", "--date", "2024-12-25", "--teams", "LAL,GSW", "--nba-root", root], capsys)
    assert rc == 1 and "cv2 cannot open the video" in out
    # an event that does not match what was given or what the file name says
    for extra, says in ((["--teams", "BOS,NYK"], "is LAL @ GSW, not BOS-NYK (--teams)"),
                        (["--date", "2024-12-26"], "was on 2024-12-25, not 2024-12-26 (--date)")):
        rc, out = run([make_video(tmp_path / "v2" / "game.mp4"), "--check", "--espn-event", "401705034", *extra,
                       "--nba-root", root, "--min-minutes", "0"], capsys)
        assert rc == 1 and says in out
    named = make_video(tmp_path / "v3" / "2025-01-10 BOS at NYK.mp4")
    rc, out = run([named, "--check", "--espn-event", "401705034", "--nba-root", root, "--min-minutes", "0"], capsys)
    assert rc == 1 and "the file name says 2025-01-10, but ESPN event 401705034 was on 2024-12-25" in out
    rc, out = run([named, "--check", "--nba-root", root, "--min-minutes", "0"], capsys)  # no such game on ESPN
    assert rc == 1 and "ESPN lists no BOS-NYK game" in out
    rc, out = run([named, "--check", "--date", "2024-12-25", "--teams", "LAL,GSW", "--nba-root", root,
                   "--min-minutes", "0"], capsys)  # given by hand: valid, but the name is pointed out
    assert rc == 0 and "the file name says 2025-01-10, not 2024-12-25 (--date)" in out
    assert "the file name names BOS-NYK, not LAL-GSW (--teams)" in out
    rc, out = run([make_video(tmp_path / "v4" / "christmas 2024-12-25.mp4"), "--check", "--nba-root", root,
                   "--min-minutes", "0"], capsys)  # no teams: which of the day's games?
    assert rc == 1 and "which game? give --teams AWY,HOM" in out and "LAL @ GSW (ESPN 401705034, Final)" in out
    assert not root.exists()


# --------------------------------------------------------------------------- --batch


def test_batch_file(tmp_path: Path, web: FakeWeb, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "runs" / "nba"
    vids = tmp_path / "videos"
    a = make_video(vids / "lal-gsw.mp4")
    make_video(vids / "copy.mp4")
    make_video(vids / "c.mp4")
    batch = vids / "games.csv"
    batch.write_text("video,date,teams\n"
                     "# Christmas 2024: relative paths are taken from this file's folder\n"
                     "lal-gsw.mp4,2024-12-25,LAL@GSW\n"
                     "copy.mp4,2024-12-25,Lakers@Warriors\n"  # the same game again
                     'c.mp4,2024-12-25,"LAL,XYZ"\n'  # an unknown team
                     "c.mp4,2024-12-25,LAL,GSW\n"  # an unquoted comma: one cell too many
                     "\n"
                     "missing.mp4,2024-12-25,PHI@BOS\n")
    rc, out = run(["--batch", batch, "--nba-root", root, "--min-minutes", "0", "--no-nba-pbp"], capsys)
    assert rc == 1
    assert "games.csv: 5 games: 1 added, 4 failed" in out
    assert "the same ESPN event (401705034) as games.csv line 3" in out
    assert "unknown NBA team 'XYZ'" in out and "4 cells for 3 columns" in out and "video not found" in out
    assert load(root / G / "game.json")["video"] == str(a)
    assert sorted(p.name for p in root.iterdir()) == [G]
    # run again with the good row only: already there
    again = vids / "again.csv"
    again.write_text("video,date,teams\nlal-gsw.mp4,2024-12-25,LAL@GSW\n")
    rc, out = run(["--batch", again, "--nba-root", root, "--min-minutes", "0", "--no-nba-pbp"], capsys)
    assert rc == 0 and "again.csv: 1 games: 1 already added" in out
    # --check of a batch with away / home / espn_event columns writes nothing
    fresh = tmp_path / "fresh" / "nba"
    other = vids / "check.csv"
    other.write_text("video,away,home,espn_event,name\n"
                     f"{a},LAL,GSW,401705034,\n"
                     "copy.mp4,,,401705034,2024-12-25-LAL-GSW-b\n")  # the same event as the row before
    rc, out = run(["--batch", other, "--check", "--nba-root", fresh, "--min-minutes", "0"], capsys)
    assert rc == 1 and "1 valid, 1 failed" in out and "the same ESPN event (401705034) as check.csv line 2" in out
    assert not fresh.exists()
    # bad input: an unknown column, or a video next to --batch
    bad = vids / "bad.csv"
    bad.write_text("vid,date\nx.mp4,2024-12-25\n")
    assert run(["--batch", bad, "--nba-root", root], capsys)[0] == 2
    assert run([a, "--batch", batch, "--nba-root", root], capsys)[0] == 2


# --------------------------------------------------------------------------- without network


def test_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    down = FakeWeb({"": urllib.error.URLError(OSError(8, "nodename nor servname provided, or not known"))})
    monkeypatch.setattr(urllib.request, "urlopen", down)
    root = tmp_path / "runs" / "nba"
    video = make_video(tmp_path / "videos" / "2024-12-25 LAL @ GSW.mp4")
    rc, out = run([video, "--nba-root", root, "--min-minutes", "0"], capsys)
    assert rc == 1 and "site.api.espn.com unreachable" in out and not root.exists()
    assert "espn.com/nba/game/_/gameId/<id>" in out and "--date 2024-12-25 --teams LAL,GSW --espn-event <id>" in out
    # the manual way in: the event id by hand, with the date and the teams
    rc, out = run([video, "--espn-event", "401705034", "--date", "2024-12-25", "--teams", "LAL,GSW",
                   "--nba-root", root, "--min-minutes", "0"], capsys)
    assert rc == 0, out
    assert load(root / G / "game.json") == {"video": str(video), "espn_event": "401705034", "date": "2024-12-25",
                                            "teams": {"away": "LAL", "home": "GSW"}}
    assert not (root / G / "espn_summary.json").exists() and "the loop's espn stage fetches it" in out
    assert sum("stats.nba.com" in u for u in down.urls) == 1  # a host that is down is asked once per run
    # a date and teams only guessed from the name are not taken unchecked (unless --yes)
    other = tmp_path / "other" / "nba"
    rc, out = run([video, "--espn-event", "401705034", "--nba-root", other, "--min-minutes", "0"], capsys)
    assert rc == 1 and "give --date and --teams (or --yes)" in out and not other.exists()
    rc, out = run([video, "--espn-event", "401705034", "--nba-root", other, "--min-minutes", "0", "--yes",
                   "--no-nba-pbp"], capsys)
    assert rc == 0 and load(other / G / "game.json")["teams"] == {"away": "LAL", "home": "GSW"}
    # --check cannot check an event without network
    rc, out = run([video, "--check", "--espn-event", "401705034", "--nba-root", other, "--min-minutes", "0"], capsys)
    assert rc == 1 and "cannot be checked without network" in out
    # a connection cut mid-body, or a page that is not JSON (a block page): reported, no traceback
    import http.client

    monkeypatch.setattr(urllib.request, "urlopen", FakeWeb({"scoreboard": http.client.IncompleteRead(b"{")}))
    rc, out = run([video, "--check", "--nba-root", other, "--min-minutes", "0"], capsys)
    assert rc == 1 and "site.api.espn.com unreachable" in out and "--espn-event <id>" in out
    monkeypatch.setattr(urllib.request, "urlopen", FakeWeb({"scoreboard": b"<html>Access Denied</html>"}))
    rc, out = run([video, "--check", "--nba-root", other, "--min-minutes", "0"], capsys)
    assert rc == 1 and "site.api.espn.com: not a JSON document" in out


# --------------------------------------------------------------------------- asking


def test_a_guessed_game_is_confirmed_and_a_missing_one_picked(tmp_path: Path, web: FakeWeb,
                                                              monkeypatch: pytest.MonkeyPatch,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(tool, "_interactive", lambda: True)
    root = tmp_path / "runs" / "nba"
    video = make_video(tmp_path / "videos" / "Lakers at Warriors 2024-12-25.mp4")
    args = [video, "--nba-root", root, "--min-minutes", "0", "--no-nba-pbp"]
    rc, out = run(args, capsys, ask=lambda q: "n")
    assert rc == 1 and "not added (not confirmed)" in out and not root.exists()
    asked: list[tuple[str, str]] = []

    def ask(question: str) -> str:
        asked.append((question, capsys.readouterr().out))  # what is on screen when asked
        return ""

    rc, out = run(args, capsys, ask=ask)
    assert rc == 0 and len(asked) == 1
    question, before = asked[0]
    assert "Los Angeles Lakers at Golden State Warriors on 2024-12-25, ESPN event 401705034?" in question
    assert f"{G}\n" in before and "teams    LAL @ GSW" in before and "wrote" not in before and "wrote" in out
    # no teams in the name: the day's games to pick from (the 4th is LAL @ GSW), then the confirmation
    answers = iter(["4", "y"])
    other = tmp_path / "other" / "nba"
    rc, out = run([make_video(tmp_path / "v2" / "christmas 2024-12-25.mp4"), "--nba-root", other, "--min-minutes", "0",
                   "--no-nba-pbp"], capsys, ask=lambda q: next(answers))
    assert rc == 0 and "  4) LAL @ GSW (ESPN 401705034, Final)" in out and (other / G / "game.json").exists()
    assert "(from the list of the day's games)" in out


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg and ffprobe")
def test_the_date_from_the_video_metadata(tmp_path: Path, web: FakeWeb, capsys: pytest.CaptureFixture[str]) -> None:
    raw = make_video(tmp_path / "raw.mp4")
    video = tmp_path / "videos" / "recording.mp4"
    video.parent.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(raw), "-c", "copy", "-metadata",
                    "creation_time=2024-12-26T01:05:00Z", str(video)], check=True)
    assert tool.metadata_date(video) == dt.date(2024, 12, 25)  # 8:05 pm US Eastern on the 25th
    root = tmp_path / "runs" / "nba"
    rc, out = run([video, "--teams", "LAL,GSW", "--nba-root", root, "--min-minutes", "0", "--no-nba-pbp"], capsys)
    assert rc == 0 and "2024-12-25 (from the video's creation time)" in out and (root / G / "game.json").exists()


# --------------------------------------------------------------------------- NBA.com


def test_nba_game_id_from_the_schedule_or_a_scoreboard(web: FakeWeb) -> None:
    sched = load(FIX / "nba_schedule.json")
    assert nba_stats.find_game_id(sched, "2024-12-25", ("GSW", "LAL")) == "0022400447"
    assert nba_stats.find_game_id(sched, "2024-12-26", ("LAL", "GSW")) == "0022400451"  # the date decides
    assert nba_stats.find_game_id(sched, "2024-12-25", ("BOS", "NYK")) is None
    for name in ("nba_scoreboardv3_20250110.json", "nba_scoreboardv2_20250110.json"):
        doc = load(FIX / name)
        assert nba_stats.find_game_id(doc, "2025-01-10", ("NYK", "BOS")) == "0022400555"
        games = {g["game_id"]: (g["date"], g["away"], g["home"]) for g in nba_stats.schedule_games(doc)}
        assert games == {"0022400555": ("2025-01-10", "BOS", "NYK"), "0022400556": ("2025-01-10", "MIA", "UTA")}
    # the tool: the season schedule lacks the game (another season), stats.nba.com's v3 scoreboard has it
    game = tool.Game("401705999", "2025-01-10", "BOS", "NYK")
    rep = tool.Report("t")
    assert tool.find_nba_game_id(game, None, tool.Net(), rep) == "0022400555"
    assert [u.split("?")[0].rsplit("/", 1)[1] for u in web.urls] == ["scheduleLeagueV2.json", "scoreboardv3"]
    assert web.headers[-1]["X-nba-stats-origin"] == "stats" and "Mozilla" in web.headers[-1]["User-agent"]
    forbidden = urllib.error.HTTPError("u", 403, "Forbidden", None, None)  # type: ignore[arg-type]
    web.routes["scoreboardv3?GameDate=2025-01-10"] = forbidden
    rep = tool.Report("t")
    assert tool.find_nba_game_id(game, None, tool.Net(), rep) == "0022400555"
    assert rep.items[-1] == ("nba.com", "game 0022400555 (from the stats.nba.com scoreboard (v2))")
    rep = tool.Report("t")
    assert tool.find_nba_game_id(tool.Game("1", "2025-02-01", "BOS", "NYK"), None, tool.Net(), rep) is None
    assert "give --nba-game-id" in rep.warnings[0] and "HTTP 404" in rep.warnings[0]
    assert tool.find_nba_game_id(game, "0022400999", tool.Net(), tool.Report("t")) == "0022400999"  # given
    with pytest.raises(ValueError):
        nba_stats.fetch_pbp("22400447")


def test_pbp_csv_round_trip_through_load_shots(tmp_path: Path) -> None:
    out = tmp_path / "nba_pbp.csv"
    rows = nba_stats.export_pbp_csv(load(FIX / "nba_pbp_0022400447.json"), out)
    with out.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames or ()) == nba_stats.CSV_COLUMNS
        back = {r["n"]: r for r in reader}
    assert len(back) == len(rows) == 16
    assert back["7"] == {"n": "7", "period": "1", "clock": "698.0", "team": "GSW", "person": "9001", "type": "2",
                         "sub": "Layup", "res": "M", "dist": "2.13", "x": "7.293035479632063",
                         "y": "52.20588235294118", "side": "left", "poss": "GSW", "sh": "2", "sa": "0",
                         "qual": "pointsinthepaint"}
    assert (back["9"]["type"], back["9"]["res"]) == ("3", "X") and back["30"]["clock"] == "5.3"
    assert (back["17"]["type"], back["17"]["res"], back["17"]["sub"]) == ("freethrow", "M", "1 of 1")
    assert back["15"]["poss"] == "LAL" and back["2"]["team"] == "" and back["2"]["clock"] == "720.0"
    shots = nba_stats.load_shots(out)  # no codes: the running score gives them
    assert [(s.period, s.clock, s.team, s.value, s.made) for s in shots] == [
        (1, 698.0, H, 2, True), (1, 680.0, A, 3, False), (1, 661.0, H, 3, True), (1, 644.0, A, 2, True),
        (1, 605.0, A, 3, True), (2, 5.3, H, 2, True)]  # the miss without a location (24) is left out
    assert shots[0].x == pytest.approx(94.0 * (1 - 7.293035479632063 / 100))
    assert shots[0].y == pytest.approx(50.0 * (1 - 52.20588235294118 / 100)) and shots[0].dist_ft == 2.13
    assert nba_stats.load_shots(out, home_code="GSW", away_code="LAL") == shots
    assert [s.team for s in nba_stats.load_shots(out, home_code="LAL", away_code="GSW")] == [A, H, A, H, H, A]
    assert nba_stats.csv_team_codes(back.values()) == {"GSW": H, "LAL": A}
    # the cue-row labeller's own code inference (shots matched to ESPN's) reads the exported codes
    assert infer_nba_codes(out, espn.pbp_actions(load(FIX / "espn_summary_401705034.json"))) == {"GSW": H, "LAL": A}
    # the first export's codes (G home, L away, as tools/nba_analyze.py reads them) still load the same way
    legacy = tmp_path / "legacy.csv"
    with legacy.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=nba_stats.CSV_COLUMNS)
        w.writeheader()
        for n, clock, team, value, res, x in ((1, 700.0, "G", 2, "M", 10), (2, 680.0, "L", 3, "X", 90)):
            w.writerow({"n": n, "period": 1, "clock": clock, "team": team, "type": value, "res": res, "x": x, "y": 50,
                        "dist": 3})
    assert [(s.team, s.value, s.made) for s in nba_stats.load_shots(legacy)] == [(H, 2, True), (A, 3, False)]
    assert nba_stats.csv_team_codes([]) == {}
