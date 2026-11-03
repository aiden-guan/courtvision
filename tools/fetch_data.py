"""Fetch the public research data of the detector fine-tunes (docs/research/cross_game_detector.md), politely.

    .venv/bin/python tools/fetch_data.py sportsmot --split val [--out data/sportsmot]
    .venv/bin/python tools/fetch_data.py bard --clips check_clips [--out data/bard/split/check]

``sportsmot``: SportsMOT (Cui et al., ICCV 2023; CC BY-NC 4.0) is published on Hugging Face as one tar per split
(``train`` 6.7 GB, ``val`` 6.6 GB) holding football, volleyball and basketball. This streams the split's tar once, in
order, and writes only the members of the sequences in ``splits_txt/basketball.txt`` (about half the archive) to
``<out>/<split>/<sequence>/``, the layout ``tools/detector_bench.py`` and ``tools/student_finetune.py prep-mot``
read; the list itself goes to ``<out>/basketball.txt``. A failed stream starts the archive again from its beginning
(a tar cannot be entered in the middle); files already written with the right size are not written again.

``bard``: the BARD clips (Giudici et al., CVIU 2026; CC BY 4.0, the footage is the NBA's) named by one list of
``cross_game_detector_splits.json``'s ``bard`` part (``check_clips``, ``calib_clips_round1``,
``calib_clips_round2``), one file at a time, to ``<out>/<game>/<clip>.mp4``; files already there are skipped.

Both are research and evaluation data only (docs/TRAINING.md §1). One HTTP request at a time; a rate-limited or
failed request (HTTP 429 or 5xx, a dropped connection) is retried after an exponential back-off.
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TypeVar

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "docs" / "research" / "cross_game_detector_splits.json"
SPORTSMOT = "https://huggingface.co/datasets/MCG-NJU/SportsMOT/resolve/main"
BARD = "https://huggingface.co/datasets/GabrieleGiudici/BARD/resolve/main"
RETRY_S = (30, 60, 120, 240, 480)
PAUSE_S = 0.5  # between two BARD files
RETRYABLE = (429, 500, 502, 503, 504)

T = TypeVar("T")


def _get(url: str) -> urllib.request.addinfourl:
    req = urllib.request.Request(url, headers={"User-Agent": "courtvision-fetch-data"})
    try:  # certifi's CA bundle where installed (python.org builds of Python ship without one configured)
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    return urllib.request.urlopen(req, timeout=120, context=ctx)  # noqa: S310 - fixed https URLs


def _retry(what: str, job: Callable[[], T]) -> T:
    """``job()``, retried after ``RETRY_S`` back-offs on a rate limit, a server error or a dropped connection."""
    for attempt, wait in enumerate((0, *RETRY_S)):
        if wait:
            print(f"{what}: retrying in {wait} s", file=sys.stderr, flush=True)
            time.sleep(wait)
        try:
            return job()
        except (urllib.error.URLError, OSError, tarfile.TarError) as e:
            code = getattr(e, "code", None)
            print(f"{what}: attempt {attempt + 1} failed: {e}", file=sys.stderr, flush=True)
            if code is not None and code not in RETRYABLE:
                raise
    raise RuntimeError(f"{what}: gave up after {len(RETRY_S) + 1} attempts")


# --------------------------------------------------------------------------- SportsMOT


def target(name: str, split: str, keep: set[str], out: Path) -> Path | None:
    """Where a tar member goes: ``<out>/<split>/<sequence>/...`` for a kept sequence, else None. macOS resource forks
    (``._*``), pax headers and anything that leaves the folder are skipped."""
    parts = PurePosixPath(name).parts
    if len(parts) < 3 or parts[0] != split or parts[1] not in keep:
        return None
    if any(p in ("", ".", "..") or p.startswith("._") or p == "PaxHeader" for p in parts):
        return None
    return out.joinpath(*parts)


def _sportsmot_pass(split: str, out: Path, keep: set[str]) -> tuple[int, int]:
    """One pass over the split's archive: (files written, bytes read)."""
    written = read = 0
    with _get(f"{SPORTSMOT}/dataset/{split}.tar") as resp:
        total = int(resp.headers.get("content-length") or 0)

        class _Counted:
            def read(self, n: int = -1) -> bytes:
                nonlocal read
                b = resp.read(n)
                read += len(b)
                return b

        report = 0
        with tarfile.open(fileobj=_Counted(), mode="r|") as tf:  # type: ignore[call-overload]
            for m in tf:
                if read - report >= 256 << 20:
                    report = read
                    print(f"{split}: {read / 1e9:.2f} of {total / 1e9:.2f} GB read, {written} files written",
                          file=sys.stderr, flush=True)
                if not m.isfile():
                    continue
                dest = target(m.name, split, keep, out)
                if dest is None or (dest.is_file() and dest.stat().st_size == m.size):
                    continue
                src = tf.extractfile(m)
                if src is None:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".part")
                tmp.write_bytes(src.read())
                tmp.replace(dest)
                written += 1
    return written, read


def fetch_sportsmot(split: str, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)

    def names() -> str:
        with _get(f"{SPORTSMOT}/splits_txt/basketball.txt") as resp:
            return resp.read().decode()

    text = _retry("basketball.txt", names)
    (out / "basketball.txt").write_text(text)
    keep = {s.strip() for s in text.split() if s.strip()}
    written, read = _retry(f"{split}.tar", lambda: _sportsmot_pass(split, out, keep))
    seqs = sorted(d.name for d in (out / split).iterdir() if d.is_dir()) if (out / split).is_dir() else []
    print(f"{split}: {written} files written ({read / 1e9:.2f} GB read); {len(seqs)} basketball sequences in "
          f"{out / split}", flush=True)
    return 0


# --------------------------------------------------------------------------- BARD


def bard_clips(listing: str, splits: Path = SPLITS) -> list[str]:
    """``<game>/<clip>.mp4`` names of one list of the splits file's ``bard`` part."""
    bard = json.loads(splits.read_text())["bard"]
    if listing not in bard or not isinstance(bard[listing], list) or not all(isinstance(c, str) for c in bard[listing]):
        names = sorted(k for k, v in bard.items() if isinstance(v, list) and v and isinstance(v[0], str))
        raise SystemExit(f"no clip list {listing!r} in {splits}; lists: {', '.join(names)}")
    clips = bard[listing]
    for c in clips:
        p = PurePosixPath(c)
        if len(p.parts) != 2 or p.suffix != ".mp4" or any(s in ("", ".", "..") for s in p.parts):
            raise SystemExit(f"unexpected clip name {c!r}")
    return clips


def fetch_bard(listing: str, out: Path) -> int:
    clips = bard_clips(listing)
    got = 0
    for i, name in enumerate(clips):
        dest = out.joinpath(*PurePosixPath(name).parts)
        if dest.is_file() and dest.stat().st_size > 0:
            continue

        def one(name: str = name, dest: Path = dest) -> None:
            with _get(f"{BARD}/{name}") as resp:
                data = resp.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)

        _retry(name, one)
        got += 1
        if (i + 1) % 10 == 0:
            print(f"{listing}: {i + 1} of {len(clips)}", file=sys.stderr, flush=True)
        time.sleep(PAUSE_S)
    print(f"{listing}: {got} clips fetched, {len(clips)} in {out}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sportsmot", help="the basketball sequences of one SportsMOT split")
    p.add_argument("--split", required=True, choices=("train", "val"), help="the splits with public labels")
    p.add_argument("--out", type=Path, default=Path("data/sportsmot"))
    p = sub.add_parser("bard", help="BARD clips named by the splits file")
    p.add_argument("--clips", required=True, help="a list of the splits file's bard part, e.g. check_clips")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "sportsmot":
        return fetch_sportsmot(a.split, a.out)
    return fetch_bard(a.clips, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
