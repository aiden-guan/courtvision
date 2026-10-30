"""Model lab endpoints for the Studio: the NBA learning loop's state, its next step and one runner.

The reads are the data of ``tools/nba_loop.py status --json`` and ``next`` (``courtvision.nba.loop``), plus free
memory and other CourtVision jobs. The runner starts ``tools/nba_loop.py run`` as a child process, one at a time; the
loop applies its own guards (another job running, too little free memory) and watchdog. Stopping sends SIGTERM, which
the loop turns into an interrupt that stops the step's process tree (exit 130). Research use only (docs/NBA.md).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
EXIT_MEANING = {0: "done", 1: "a step failed (the others went on)", 2: "bad input", 3: "refused (another job, or too little memory)",
                4: "stopped by the memory watchdog", 130: "stopped", -15: "stopped"}
LOG_TAIL_BYTES = 48_000


class RunIn(BaseModel):
    max_steps: int | None = Field(None, ge=1, le=50)
    game: str | None = Field(None, max_length=120)


class PromoteIn(BaseModel):
    version: str = Field(..., max_length=16)


def _tail(path: Path | None, limit: int = LOG_TAIL_BYTES) -> str:
    if path is None or not path.is_file():
        return ""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - limit))
        data = f.read()
    text = data.decode("utf-8", "replace")
    return text if size <= limit else text.split("\n", 1)[-1]


class Lab:
    """The learning loop as the Studio sees it: one runner at a time, started from this server."""

    def __init__(self, runs: Path | None = None, repo: Path = REPO) -> None:
        self.repo = repo
        self.runs = (runs or repo / "runs").resolve()
        self.proc: subprocess.Popen[bytes] | None = None
        self.started: float | None = None
        self.args: list[str] = []
        self.log_path: Path | None = None
        self.exit: int | None = None
        self.lock = asyncio.Lock()

    @property
    def tool(self) -> Path:
        return self.repo / "tools" / "nba_loop.py"

    def available(self) -> str | None:
        """Why the lab cannot run here, or None."""
        if not self.tool.is_file():
            return "The model lab needs the CourtVision repository (tools/nba_loop.py was not found next to this package)."
        return None

    def _loop(self) -> Any:
        from courtvision.nba import loop as nl

        return nl, nl.Loop(self.runs, self.repo, nl.LoopConfig(python=self.python()))

    def python(self) -> str:
        # relative, as tools/nba_loop.py prints it: steps run from the repository
        return ".venv/bin/python" if (self.repo / ".venv" / "bin" / "python").exists() else sys.executable

    def runner(self) -> dict[str, Any]:
        running = self.proc is not None and self.proc.poll() is None
        if self.proc is not None and not running and self.exit is None:
            self.exit = self.proc.returncode
        return {"running": running, "pid": self.proc.pid if self.proc is not None else None,
                "args": self.args, "started": self.started, "elapsed_s": round(time.time() - self.started, 1) if self.started else None,
                "exit": self.exit, "exit_meaning": EXIT_MEANING.get(self.exit, f"exit {self.exit}") if self.exit is not None else None,
                "log": self._rel(self.log_path)}

    def _rel(self, p: Path | None) -> str | None:
        if p is None:
            return None
        try:
            return str(p.relative_to(self.repo))
        except ValueError:
            return str(p)

    def snapshot(self) -> dict[str, Any]:
        why = self.available()
        if why:
            return {"available": False, "reason": why, "runner": self.runner()}
        nl, loop = self._loop()
        status = nl.status(loop)
        step, steps = nl.next_step(loop)
        cmd_steps = [s for s in steps if s.cmd is not None]
        nxt = None
        if step is not None:
            nxt = {"label": step.label(), "stage": step.stage, "game": step.game, "heavy": step.heavy, "why": step.why,
                   "command": step.shell(), "log": loop.rel(step.log) if step.log else None,
                   "ready": len(cmd_steps), "heavy_ready": sum(1 for s in cmd_steps if s.heavy)}
        assessments = []
        for kind in nl.KINDS:
            for vid in nl.open_candidates(loop, kind):
                try:
                    d = nl.assess(loop, vid)
                    assessments.append({"candidate": vid, "kind": kind, "outcome": d.get("outcome"), "reason": d.get("reason")})
                except Exception as exc:  # noqa: BLE001 - one unreadable candidate must not hide the rest
                    assessments.append({"candidate": vid, "kind": kind, "outcome": "unknown", "reason": str(exc)})
        jobs = [{"pid": pid, "cmd": cmd[:240]} for pid, cmd in nl.other_jobs()]
        cfg = loop.cfg
        return {"available": True, "status": status, "next": nxt, "notes": dict(loop.fit_notes), "assessments": assessments,
                "stages": list(nl.STATUS_STAGES),
                "machine": {"free_pct": nl.free_memory_pct(), "min_free": cfg.min_free, "watch_free": cfg.watch_free, "jobs": jobs},
                "runner": self.runner()}

    def latest_log(self) -> Path | None:
        if self.log_path is not None and self.log_path.is_file():
            return self.log_path
        nba = self.runs / "nba"
        logs = [p for p in nba.glob("*/logs/*.log")] + [p for p in nba.glob("logs/*.log")]
        logs = [p for p in logs if p.is_file()]
        return max(logs, key=lambda p: p.stat().st_mtime) if logs else None

    def start(self, body: RunIn) -> dict[str, Any]:
        if self.proc is not None and self.proc.poll() is None:
            raise HTTPException(409, "A loop run is already going. Stop it or wait for it to finish.")
        why = self.available()
        if why:
            raise HTTPException(404, why)
        args = [self.python(), str(self.tool), "run"]
        if body.max_steps:
            args += ["--max-steps", str(body.max_steps)]
        if body.game:
            nl, loop = self._loop()
            if body.game not in {g.name for g in loop.games}:
                raise HTTPException(404, f"No game folder named {body.game!r} under runs/nba.")
            args += ["--game", body.game]
        logs = self.runs / "nba" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self.log_path = logs / f"studio-loop-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
        with self.log_path.open("ab") as out:
            self.proc = subprocess.Popen(args, cwd=self.repo, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.args = args[1:]
        self.started = time.time()
        self.exit = None
        log.info("model lab: started %s (pid %s)", " ".join(self.args), self.proc.pid)
        return self.runner()

    def stop(self) -> dict[str, Any]:
        if self.proc is None or self.proc.poll() is not None:
            raise HTTPException(409, "No loop run started here is going.")
        self.proc.send_signal(signal.SIGTERM)
        return self.runner()

    def promote(self, version: str) -> dict[str, Any]:
        if not re.fullmatch(r"[scf]\d{3,}", version):
            raise HTTPException(400, "A version looks like s002, c001 or f001.")
        if self.proc is not None and self.proc.poll() is None:
            raise HTTPException(409, "Wait for the loop run to finish before promoting.")
        nl, loop = self._loop()
        if version not in {v for k in nl.KINDS for v in nl.open_candidates(loop, k)}:
            raise HTTPException(404, f"{version} is not an open candidate.")
        d = nl.promote(loop, version)
        return {"promoted": bool(d.get("promoted")), "outcome": d.get("outcome"), "reason": d.get("reason")}

    def shutdown(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)


def register_lab(app: FastAPI, lab: Lab, read: Any, write: Any) -> None:
    @app.get("/api/lab", dependencies=[read])
    async def lab_state() -> Any:
        async with lab.lock:
            try:
                return await asyncio.to_thread(lab.snapshot)
            except (KeyError, ValueError) as exc:
                return {"available": False, "reason": f"The learning loop could not be read: {exc}", "runner": lab.runner()}

    @app.get("/api/lab/log", dependencies=[read])
    async def lab_log() -> Any:
        path = await asyncio.to_thread(lab.latest_log)
        return {"path": lab._rel(path), "text": await asyncio.to_thread(_tail, path), "runner": lab.runner()}

    @app.post("/api/lab/run", dependencies=[write])
    async def lab_run(body: RunIn) -> Any:
        async with lab.lock:
            return await asyncio.to_thread(lab.start, body)

    @app.post("/api/lab/stop", dependencies=[write])
    async def lab_stop() -> Any:
        return lab.stop()

    @app.post("/api/lab/promote", dependencies=[write])
    async def lab_promote(body: PromoteIn) -> Any:
        async with lab.lock:
            return await asyncio.to_thread(lab.promote, body.version)
