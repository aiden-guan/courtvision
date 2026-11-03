"""Fetch the public research data of the detector fine-tunes (docs/research/cross_game_detector.md), politely.

    .venv/bin/python tools/fetch_data.py sportsmot --split val [--out data/sportsmot]
    .venv/bin/python tools/fetch_data.py bard --clips check_clips [--out data/bard/split/check]
    .venv/bin/python tools/fetch_data.py roboflow [--out data/roboflow]

``sportsmot``: SportsMOT (Cui et al., ICCV 2023; CC BY-NC 4.0) is published on Hugging Face as one tar per split
(``train`` 6.7 GB, ``val`` 6.6 GB) holding football, volleyball and basketball. This streams the split's tar once, in
order, and writes only the members of the sequences in ``splits_txt/basketball.txt`` (about half the archive) to
``<out>/<split>/<sequence>/``, the layout ``tools/detector_bench.py`` and ``tools/student_finetune.py prep-mot``
read; the list itself goes to ``<out>/basketball.txt``. A failed stream starts the archive again from its beginning
(a tar cannot be entered in the middle); files already written with the right size are not written again.

``bard``: the BARD clips (Giudici et al., CVIU 2026; CC BY 4.0, the footage is the NBA's) named by one list of
``cross_game_detector_splits.json``'s ``bard`` part (``check_clips``, ``calib_clips_round1``,
``calib_clips_round2``), one file at a time, to ``<out>/<game>/<clip>.mp4``; files already there are skipped.

``roboflow``: Roboflow Universe's "basketball-player-detection-2" (CC BY 4.0; human labels of NBA broadcast frames) from
Hugging Face: the images of the splits file's ``roboflow_coco`` clips (exact clip names), the training clips' from all
three folders of the copy to ``<out>/fit/`` with ``<out>/fit_ann.json``, the held-out clips' from its ``valid`` and
``test`` folders to ``<out>/heldout/`` with ``<out>/heldout_ann.json`` (1,190 and 211 images) (``detector_bench.py --coco data/roboflow/heldout_ann.json``; ``student_finetune.py
prep-coco data/roboflow/fit_ann.json data/roboflow/fit ...``).

All are research and evaluation data only (docs/TRAINING.md §1). One HTTP request at a time; a rate-limited or
failed request (HTTP 429 or 5xx, a dropped connection) is retried after an exponential back-off.
"""

from __future__ import annotations

import argparse
import json
import re
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
ROBOFLOW = "https://huggingface.co/datasets/koppolusameer/basketball-coco-player-detection/resolve/main"
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


def fetch_bard(listing: str, out: Path, first: int = 0, count: int | None = None) -> int:
    clips = bard_clips(listing)[first:None if count is None else first + count]
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


# --------------------------------------------------------------------------- Roboflow


RF_SPLITS = ("train", "valid", "test")  # the Hugging Face copy's folders


def rf_clip(file_name: str) -> str:
    """A Roboflow export image's clip: its name up to ``_mp4-NNNN_jpg`` (or ``_mov-``)."""
    return re.split(r"_(?:mp4|mov)-\d{4}_jpg", file_name)[0]


def roboflow_split(anns: dict[str, dict], clips: list[str], folders: tuple[str, ...]) -> dict:
    """One annotation file of the images of ``clips`` (exact clip names, ``rf_clip``) found in ``folders`` of the
    per-folder annotation files ``anns``, ids renumbered; each image keeps its folder as ``"folder"``."""
    keep = set(clips)
    images, labels = [], []
    for folder in folders:
        ann = anns[folder]
        ids = {}
        for im in ann["images"]:
            if rf_clip(im["file_name"]) in keep:
                ids[im["id"]] = len(images) + 1
                images.append({**im, "id": ids[im["id"]], "folder": folder})
        for a in ann["annotations"]:
            if a["image_id"] in ids:
                labels.append({**a, "id": len(labels) + 1, "image_id": ids[a["image_id"]]})
    first = anns[folders[0]]
    return {**{k: v for k, v in first.items() if k not in ("images", "annotations")}, "images": images,
            "annotations": labels}


def fetch_roboflow(out: Path, splits: Path = SPLITS, whole: bool = False) -> int:
    """The splits file's training games' images from every folder of the copy, the held-out games' from ``valid`` and
    ``test`` (the cloud round's sets: 1,190 and 211 images). With ``whole``, also every image of the copy as it is
    published, to ``<out>/whole/<folder>/`` beside its annotation file (the referee classifier's sources,
    ``tools/referee_classifier.py``); images already fetched are linked, not fetched again."""
    rf = json.loads(splits.read_text())["roboflow_coco"]

    def ann_file(folder: str) -> dict:
        with _get(f"{ROBOFLOW}/{folder}/_annotations.coco.json") as resp:
            return json.loads(resp.read().decode())

    anns = {f: _retry(f"{f}/_annotations.coco.json", lambda f=f: ann_file(f)) for f in RF_SPLITS}
    for name, clips, folders in (("fit", rf["train_prefixes"], RF_SPLITS),
                                 ("heldout", rf["heldout_prefixes"], ("valid", "test"))):
        part = roboflow_split(anns, clips, folders)
        dest = out / name
        dest.mkdir(parents=True, exist_ok=True)
        for im in part["images"]:
            fn = im["file_name"]
            if "/" in fn or fn.startswith(".") or (dest / fn).is_file():
                continue

            def one(fn: str = fn, folder: str = im["folder"]) -> None:
                with _get(f"{ROBOFLOW}/{folder}/{fn}") as resp:
                    data = resp.read()
                tmp = dest / (fn + ".part")
                tmp.write_bytes(data)
                tmp.replace(dest / fn)

            _retry(fn, one)
            time.sleep(PAUSE_S)
        (out / f"{name}_ann.json").write_text(json.dumps(part))
        print(f"{name}: {len(part['images'])} images, {len(part['annotations'])} labels in {dest}", flush=True)
    if whole:
        have = {p.name: p for d in ("fit", "heldout") if (out / d).is_dir() for p in (out / d).iterdir()}
        for folder in RF_SPLITS:
            dest = out / "whole" / folder
            dest.mkdir(parents=True, exist_ok=True)
            for im in anns[folder]["images"]:
                fn = im["file_name"]
                if "/" in fn or fn.startswith(".") or (dest / fn).is_file():
                    continue
                if fn in have:
                    (dest / fn).hardlink_to(have[fn])
                    continue

                def one(fn: str = fn, folder: str = folder, dest: Path = dest) -> None:
                    with _get(f"{ROBOFLOW}/{folder}/{fn}") as resp:
                        data = resp.read()
                    tmp = dest / (fn + ".part")
                    tmp.write_bytes(data)
                    tmp.replace(dest / fn)

                _retry(fn, one)
                time.sleep(PAUSE_S)
            (dest / "_annotations.coco.json").write_text(json.dumps(anns[folder]))
            print(f"whole/{folder}: {len(anns[folder]['images'])} images", flush=True)
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
    p.add_argument("--first", type=int, default=0, help="start at this clip of the list (fetch it in batches)")
    p.add_argument("--count", type=int, default=None, help="fetch at most this many clips")
    p = sub.add_parser("roboflow", help="the Roboflow NBA frames of the splits file's training and held-out games")
    p.add_argument("--out", type=Path, default=Path("data/roboflow"))
    p.add_argument("--whole", action="store_true", help="also the whole copy, by folder (the referee classifier's data)")
    a = ap.parse_args(argv)
    if a.cmd == "sportsmot":
        return fetch_sportsmot(a.split, a.out)
    if a.cmd == "roboflow":
        return fetch_roboflow(a.out, whole=a.whole)
    return fetch_bard(a.clips, a.out, a.first, a.count)


if __name__ == "__main__":
    raise SystemExit(main())
