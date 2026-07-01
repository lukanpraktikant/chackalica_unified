"""Thin HTTP service wrapping the friendy_chachkalica training pipeline.

The trainer is a heavy torch/CUDA library that must run in *this* environment
(its own ``.venv``), so a separate process (e.g. a Django app) can't import it.
This service is the bridge: it accepts a request naming a config YAML already on
the shared filesystem, launches ``run.py`` against it as a subprocess, and
reports status. All real outputs (checkpoints, ``results.yaml``,
``run_summary.yaml``) are written by the trainer to the config's ``output_dir``
— this service only owns *launching* and *liveness*.

Run it from the repo root:

    .venv/bin/uvicorn service:app --host 0.0.0.0 --port 8200

Endpoints:
    GET  /health           -> {"status": "ok"}
    POST /train            -> {run_id, status, pid}    (body: {run_id, config_path, resume?})
    GET  /runs/{run_id}    -> {run_id, status, returncode, started_at, finished_at, log_tail}
    POST /eval             -> {eval_id, status, pid}   (body: {eval_id, request_path})
    GET  /evals/{eval_id}  -> {eval_id, status, returncode, started_at, finished_at, log_tail}
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
# Single-GPU safety: refuse a new launch while one is running unless overridden.
MAX_CONCURRENT = int(os.environ.get("FC_MAX_CONCURRENT", "1"))

app = FastAPI(title="friendy_chachkalica trainer")

_lock = threading.Lock()
_jobs: dict[int, dict] = {}  # run_id -> {proc, pid, started_at, finished_at, log_path, ...}


class TrainRequest(BaseModel):
    run_id: int
    config_path: str
    resume: bool = False


class EvalRequest(BaseModel):
    eval_id: int
    request_path: str


def _job_status(job: dict) -> str:
    proc: subprocess.Popen = job["proc"]
    rc = proc.poll()
    if rc is None:
        return "running"
    if job.get("finished_at") is None:
        job["finished_at"] = time.time()
    job["returncode"] = rc
    return "ok" if rc == 0 else "error"


def _active_count() -> int:
    return sum(1 for job in _jobs.values() if _job_status(job) == "running")


def _log_tail(log_path: Optional[str], limit: int = 4000) -> str:
    if not log_path or not Path(log_path).exists():
        return ""
    data = Path(log_path).read_text(encoding="utf-8", errors="replace")
    return data[-limit:]


@app.get("/health")
def health():
    return {"status": "ok", "active": _active_count()}


def _spawn(key: str, cmd: list[str], output_dir: Path) -> dict:
    """Launch ``cmd`` as a tracked subprocess logging into output_dir/service.log."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "service.log"
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(HERE), stdout=log_file, stderr=subprocess.STDOUT)
    job = {
        "proc": proc,
        "pid": proc.pid,
        "started_at": time.time(),
        "finished_at": None,
        "returncode": None,
        "log_path": str(log_path),
        "output_dir": str(output_dir),
        "_log_file": log_file,
    }
    _jobs[key] = job
    return job


def _status_payload(job: dict) -> dict:
    return {
        "status": _job_status(job),
        "pid": job["pid"],
        "returncode": job.get("returncode"),
        "started_at": job["started_at"],
        "finished_at": job.get("finished_at"),
        "output_dir": job["output_dir"],
        "log_tail": _log_tail(job.get("log_path")),
    }


@app.post("/train")
def train(req: TrainRequest):
    config_path = Path(req.config_path)
    if not config_path.exists():
        raise HTTPException(status_code=400, detail=f"config not found: {config_path}")

    with _lock:
        key = f"train-{req.run_id}"
        existing = _jobs.get(key)
        if existing and _job_status(existing) == "running":
            return {"run_id": req.run_id, "status": "running", "pid": existing["pid"]}
        if _active_count() >= MAX_CONCURRENT:
            raise HTTPException(status_code=409, detail="trainer busy: another run is active")

        try:
            output_dir = Path(yaml.safe_load(config_path.read_text())["output_dir"])
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"bad config: {exc}")

        cmd = [sys.executable, "run.py", str(config_path)]
        if req.resume:
            cmd.append("--resume")
        job = _spawn(key, cmd, output_dir)
        return {"run_id": req.run_id, "status": "running", "pid": job["pid"]}


@app.get("/runs/{run_id}")
def run_status(run_id: int):
    job = _jobs.get(f"train-{run_id}")
    if job is None:
        raise HTTPException(status_code=404, detail="unknown run_id")
    return {"run_id": run_id, **_status_payload(job)}


@app.post("/eval")
def evaluate(req: EvalRequest):
    request_path = Path(req.request_path)
    if not request_path.exists():
        raise HTTPException(status_code=400, detail=f"eval request not found: {request_path}")

    with _lock:
        key = f"eval-{req.eval_id}"
        existing = _jobs.get(key)
        if existing and _job_status(existing) == "running":
            return {"eval_id": req.eval_id, "status": "running", "pid": existing["pid"]}
        if _active_count() >= MAX_CONCURRENT:
            raise HTTPException(status_code=409, detail="trainer busy: another job is active")

        try:
            output_dir = Path(yaml.safe_load(request_path.read_text())["output_dir"])
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"bad eval request: {exc}")

        cmd = [sys.executable, "eval_checkpoint.py", str(request_path)]
        job = _spawn(key, cmd, output_dir)
        return {"eval_id": req.eval_id, "status": "running", "pid": job["pid"]}


@app.get("/evals/{eval_id}")
def eval_status(eval_id: int):
    job = _jobs.get(f"eval-{eval_id}")
    if job is None:
        raise HTTPException(status_code=404, detail="unknown eval_id")
    return {"eval_id": eval_id, **_status_payload(job)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("FC_PORT", "8200")))
