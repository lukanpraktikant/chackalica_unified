"""HTTP client for the friendy_chachkalica trainer service.

The trainer runs in its own torch/CUDA environment behind the small FastAPI
service (``friendy_chachkalica/service.py``). We talk to it over HTTP, pointing it at
config YAMLs we wrote to the shared filesystem and reading back run liveness.
"""

import os

import requests

from training.models import TrainingSettings

TIMEOUT = 30


def base_url(ts: TrainingSettings | None = None) -> str:
    # In the unified docker-compose stack the trainer is reachable by its
    # service name; TRAINING_SERVICE_URL lets the deployment override the DB
    # setting without an admin edit. Falls back to TrainingSettings otherwise.
    env_url = os.getenv("TRAINING_SERVICE_URL")
    if env_url:
        return env_url.rstrip("/")
    ts = ts or TrainingSettings.load()
    return ts.service_base_url.rstrip("/")


def health(ts: TrainingSettings | None = None) -> dict:
    resp = requests.get(f"{base_url(ts)}/health", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def launch(run, resume: bool = False, ts: TrainingSettings | None = None) -> dict:
    """Ask the service to start training ``run`` from its generated config."""
    payload = {"run_id": run.pk, "config_path": run.config_yaml_path, "resume": resume}
    resp = requests.post(f"{base_url(ts)}/train", json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_status(run, ts: TrainingSettings | None = None) -> dict:
    """Return the service's view of ``run``.

    A 404 means the service has no record (e.g. it was restarted) — reported as
    ``{"status": "unknown"}`` so the caller can fall back to the shared
    filesystem (a finished run leaves run_summary.yaml behind).
    """
    resp = requests.get(f"{base_url(ts)}/runs/{run.pk}", timeout=TIMEOUT)
    if resp.status_code == 404:
        return {"status": "unknown"}
    resp.raise_for_status()
    return resp.json()


def stop(run, grace: float = 10.0, ts: TrainingSettings | None = None) -> dict:
    """Ask the service to gracefully stop ``run``'s training process.

    A 404 means the service has no record of the run (never launched, or the
    service was restarted) — reported as ``{"status": "unknown"}`` so a kill can
    proceed to clean up the DB/filesystem regardless.
    """
    resp = requests.post(
        f"{base_url(ts)}/runs/{run.pk}/stop", params={"grace": grace}, timeout=TIMEOUT
    )
    if resp.status_code == 404:
        return {"status": "unknown"}
    resp.raise_for_status()
    return resp.json()


def launch_eval(eval_run, ts: TrainingSettings | None = None) -> dict:
    """Ask the service to evaluate a trained model from its generated request."""
    payload = {"eval_id": eval_run.pk, "request_path": eval_run.request_yaml_path}
    resp = requests.post(f"{base_url(ts)}/eval", json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_eval_status(eval_run, ts: TrainingSettings | None = None) -> dict:
    resp = requests.get(f"{base_url(ts)}/evals/{eval_run.pk}", timeout=TIMEOUT)
    if resp.status_code == 404:
        return {"status": "unknown"}
    resp.raise_for_status()
    return resp.json()


def launch_pipeline(pe, ts: TrainingSettings | None = None) -> dict:
    """Ask the service to run a chachak pipeline eval from its generated request."""
    payload = {"pipeline_id": pe.pk, "request_path": pe.request_yaml_path}
    resp = requests.post(f"{base_url(ts)}/pipeline", json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_pipeline_status(pe, ts: TrainingSettings | None = None) -> dict:
    resp = requests.get(f"{base_url(ts)}/pipelines/{pe.pk}", timeout=TIMEOUT)
    if resp.status_code == 404:
        return {"status": "unknown"}
    resp.raise_for_status()
    return resp.json()


# First predict call for a model loads it into GPU memory; allow well past the
# short module TIMEOUT. Warm calls return in well under a second.
PREDICT_TIMEOUT = 180


def predict_image(payload: dict, ts: TrainingSettings | None = None) -> dict:
    """Run one image through a trained model synchronously; return {boxes, classes}.

    Used by the interactive model preview — the service keeps the model warm, so
    only the first request per model/pipeline pays the load cost.
    """
    resp = requests.post(
        f"{base_url(ts)}/predict_image", json=payload, timeout=PREDICT_TIMEOUT)
    if resp.status_code >= 400:
        detail = resp.text
        try:
            body = resp.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and body.get("detail"):
            detail = str(body["detail"])
        raise RuntimeError(
            f"trainer /predict_image returned HTTP {resp.status_code}: {detail}"
        )
    return resp.json()
