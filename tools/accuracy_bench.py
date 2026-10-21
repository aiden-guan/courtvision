"""CourtVision accuracy benchmark: the one accuracy number every agent reports.

For each ``(mode, seed)`` it runs exactly what ``courtvision simulate --mode MODE --seed SEED
--seconds T`` runs (``cli.simulate_config`` -> ``pipeline.run_offline``), evaluates against the
simulator's truth with ``courtvision.eval`` and pools the counts over seeds
(``courtvision/eval/bench.py`` documents every metric). Each run is independent, so runs go to a
process pool (``--workers``). It prints a markdown table and writes JSON (``--json``).

Split protocol (anti-overfitting; ``courtvision.eval.bench``):

- ``--split train`` (default): seeds 1-6. The only seeds anyone tunes on. ``--keep-runs DIR``
  writes each run directory (events, truth, report.md) for debugging.
- ``--split heldout``: seeds 101-104, for the verifier only. Aggregate metrics only: ``--keep-runs``
  is refused, and no per-event output is written anywhere.

Video mode renders and perceives every frame (~30 ms/frame), so its game length defaults to
``--video-seconds 180`` (about 165 s wall per run) instead of ``--seconds``; ``--video-seeds``
limits it to fewer seeds. Box-score numbers of 180 s and 300 s games are not comparable across
modes; F1 and accuracies are rates and are.

    .venv/bin/python tools/accuracy_bench.py --modes truth3d,mono,video --split train \\
        --seconds 300 --video-seconds 180 --video-seeds 1,2,3 --workers 8 --json runs/bench/train.json

``--compare BEFORE.json`` appends the deltas against an earlier report (and ``--diff A.json B.json`` compares
two saved reports without running): every delta comes with the truth n and the 95% seed-bootstrap
intervals, and a 300 s headline change of <= 0.03 is flagged as noise (decide on 1200 s runs, where the
rare kinds have ~20-40 truth events). Each mode also prints its pooled headline per non-cumulative 60 s
window (a real degradation over time would show as a trend there).

Exit codes: 0 every run succeeded, 1 a run failed (the others are still reported), 2 usage error.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import shlex
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # run as a script from the repo root
    sys.path.insert(0, str(ROOT))

from courtvision.eval.bench import (DEFAULT_SECONDS, DEFAULT_VIDEO_SECONDS, MODES, SPLITS,  # noqa: E402
                                    BenchCase, build_report, compare_reports, format_comparison, format_markdown,
                                    quiet_worker_logging, run_case)

EXIT_OK, EXIT_ERROR, EXIT_USAGE = 0, 1, 2


class UsageError(Exception):
    pass


def _csv(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def _seeds(text: str) -> list[int]:
    try:
        out = [int(t) for t in _csv(text)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}") from exc
    if not out:
        raise argparse.ArgumentTypeError("no seeds given")
    return out


def _positive(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from exc
    if not value > 0 or value == float("inf"):
        raise argparse.ArgumentTypeError(f"expected a positive number, got {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="accuracy_bench.py", description=__doc__.split("\n\n")[0],
        epilog="Train seeds 1-6 (tune here); held-out seeds 101-104 (verifier only, aggregate output only).")
    p.add_argument("--modes", default="truth3d,mono,video", help="comma list of truth3d,mono,video (default: all)")
    p.add_argument("--split", choices=sorted(SPLITS), default="train", help="seed split (default train)")
    p.add_argument("--seeds", type=_seeds, default=None,
                   help="comma list, a subset of the split's seeds (default: every seed of the split)")
    p.add_argument("--seconds", type=_positive, default=DEFAULT_SECONDS,
                   help=f"simulated seconds per game (default {DEFAULT_SECONDS:g})")
    p.add_argument("--video-seconds", type=_positive, default=None,
                   help=f"seconds per game in video mode (default {DEFAULT_VIDEO_SECONDS:g}, or --seconds when "
                        f"that is shorter: render + perception run ~30 ms/frame)")
    p.add_argument("--video-seeds", type=_seeds, default=None, help="seeds for video mode (default: --seeds)")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel processes (default: min(runs, CPUs - 1)); 1 runs in this process")
    p.add_argument("--json", metavar="OUT.json", help="write the full report as JSON")
    p.add_argument("--md", metavar="OUT.md", help="also write the markdown table")
    p.add_argument("--config", metavar="FILE", help="config file passed to every run (as `simulate --config`)")
    p.add_argument("--set", metavar="KEY=VALUE", action="append", default=[],
                   help="config override passed to every run (as `simulate --set`; repeatable)")
    p.add_argument("--keep-runs", metavar="DIR", help="train split only: write each run directory under DIR")
    p.add_argument("--compare", metavar="BEFORE.json",
                   help="after the run, print deltas against an earlier --json report (truth n, 95%% CIs, noise rule)")
    p.add_argument("--diff", nargs=2, metavar=("BEFORE.json", "AFTER.json"),
                   help="compare two saved --json reports and exit (runs nothing)")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress lines on stderr")
    return p


def plan_cases(args: argparse.Namespace) -> list[BenchCase]:
    """The ``(mode, seed)`` cases of an invocation; :class:`UsageError` on a protocol violation."""
    modes = _csv(args.modes)
    unknown = [m for m in modes if m not in MODES]
    if not modes or unknown:
        raise UsageError(f"--modes: unknown or empty {unknown or modes!r} (choose from {', '.join(MODES)})")
    if len(set(modes)) != len(modes):
        raise UsageError(f"--modes: duplicate mode in {args.modes!r}")
    allowed = SPLITS[args.split]
    seeds = list(args.seeds) if args.seeds is not None else list(allowed)
    video_seeds = list(args.video_seeds) if args.video_seeds is not None else seeds
    for name, group in (("--seeds", seeds), ("--video-seeds", video_seeds)):
        outside = sorted(set(group) - set(allowed))
        if outside:
            raise UsageError(f"{name}: {outside} are not {args.split} seeds {list(allowed)} "
                             "(tune only on train seeds; held-out seeds are for the verifier)")
        if len(set(group)) != len(group):
            raise UsageError(f"{name}: duplicate seed in {group}")
    if args.keep_runs and args.split != "train":
        raise UsageError("--keep-runs is train-only: held-out runs keep aggregate metrics, never per-event output")
    for item in args.set:
        if "=" not in item or not item.split("=", 1)[0].strip():
            raise UsageError(f"--set: expected section.key=value, got {item!r}")
    video_seconds = args.video_seconds if args.video_seconds is not None else min(DEFAULT_VIDEO_SECONDS, args.seconds)
    cases = []
    for mode in modes:
        for seed in (video_seeds if mode == "video" else seeds):
            cases.append(BenchCase(mode=mode, seed=seed, seconds=video_seconds if mode == "video" else args.seconds,
                                   split=args.split, config=args.config, overrides=tuple(args.set),
                                   keep_dir=args.keep_runs))
    return cases


def execute(cases: Sequence[BenchCase], workers: int,
            progress: Callable[[str], None] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run every case (in a spawn process pool when ``workers > 1``); returns (records, errors)."""
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    total = len(cases)

    def done(case: BenchCase, rec: dict[str, Any] | None, exc: BaseException | None) -> None:
        n = len(records) + len(errors) + 1
        if exc is not None:
            errors.append({"mode": case.mode, "seed": case.seed, "seconds": case.seconds,
                           "error": f"{type(exc).__name__}: {exc}"})
            if progress:
                progress(f"[{n}/{total}] {case.label}: FAILED {type(exc).__name__}: {exc}")
            return
        assert rec is not None
        records.append(rec)
        if progress:
            head = rec.get("headline_f1")
            progress(f"[{n}/{total}] {case.label}: headline F1 "
                     f"{'n/a' if head is None else f'{head:.3f}'} ({rec.get('wall_s', 0.0):.1f} s)")

    # Longest first (video renders every frame), so the pool's tail is short.
    order = sorted(cases, key=lambda c: (c.mode != "video", -c.seconds))
    if workers <= 1:
        for case in order:
            try:
                rec = run_case(case)
            except Exception as exc:  # noqa: BLE001 - one failed run must not hide the others
                done(case, None, exc)
            else:
                done(case, rec, None)
        return records, errors
    # spawn: fork is unsafe on macOS once numpy / OpenCV threads exist. Everything a child unpickles
    # (run_case, the initializer, BenchCase) lives in courtvision.eval.bench, which a child can import.
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                initializer=quiet_worker_logging) as pool:
        futures = {pool.submit(run_case, case): case for case in order}
        for fut in cf.as_completed(futures):
            case = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                done(case, None, exc)
            else:
                done(case, rec, None)
    return records, errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.diff:
        try:
            before, after = (json.loads(Path(f).read_text()) for f in args.diff)
        except (OSError, ValueError) as exc:
            print(f"accuracy_bench: error: --diff: {exc}", file=sys.stderr)
            return EXIT_USAGE
        print(format_comparison(compare_reports(before, after)), end="")
        return EXIT_OK
    baseline = None
    if args.compare:
        try:
            baseline = json.loads(Path(args.compare).read_text())
        except (OSError, ValueError) as exc:
            print(f"accuracy_bench: error: --compare: {exc}", file=sys.stderr)
            return EXIT_USAGE
    try:
        cases = plan_cases(args)
    except UsageError as exc:
        print(f"accuracy_bench: error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if args.workers is not None and args.workers < 1:
        print("accuracy_bench: error: --workers must be >= 1", file=sys.stderr)
        return EXIT_USAGE
    workers = args.workers if args.workers is not None else max(1, min(len(cases), (os.cpu_count() or 2) - 1))
    raw = list(argv) if argv is not None else sys.argv[1:]
    command = shlex.join([".venv/bin/python", "tools/accuracy_bench.py", *raw])
    progress = None if args.quiet else (lambda line: print(line, file=sys.stderr, flush=True))
    if progress:
        progress(f"accuracy_bench: {len(cases)} runs ({args.split} split) on {min(workers, len(cases))} worker(s)")
    t0 = time.perf_counter()
    records, errors = execute(cases, workers, progress)
    report = build_report(records, split=args.split, command=command, errors=errors,
                          config={"file": args.config, "overrides": list(args.set)},
                          wall_s=time.perf_counter() - t0)
    text = format_markdown(report)
    if baseline is not None:
        text += "\n" + format_comparison(compare_reports(baseline, report))
    print(text, end="")
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2, default=str) + "\n")
    if args.md:
        out = Path(args.md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    return EXIT_ERROR if errors else EXIT_OK


if __name__ == "__main__":
    quiet_worker_logging()
    raise SystemExit(main())
