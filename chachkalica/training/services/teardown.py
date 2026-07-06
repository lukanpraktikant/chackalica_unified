"""Graceful teardown of a training run.

Mirrors the fleet annotator teardown (``fleet.services.provisioning.remove_annotator``):
stop the running process, remove its artifacts from the shared filesystem, then
delete the DB row. ``RunResult`` rows cascade with the run; promoted
``TrainedModel`` rows survive (their ``source_run_result`` is SET_NULL), but note
their checkpoints live *inside* the run's output dir, so removing the files
orphans those checkpoints — the admin confirm page warns about this first.

File removal is delegated to :mod:`training.services.cleanup` — the same guarded
code the ``post_delete`` signal uses — so a blank/relative/rogue ``output_dir``
can never escalate into deleting something outside the configured roots. (The
signal fires again on ``run.delete()`` below; it's idempotent, so the files are
simply already gone.)
"""

from pathlib import Path

from django.conf import settings

from training.models import TrainingRun
from training.services import cleanup, runner


def _resolve(path: str) -> Path:
    """Absolute path: as-is if absolute, else relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else Path(settings.BASE_DIR) / p


def kill_run(run: TrainingRun) -> dict:
    """Stop ``run`` on the trainer, delete its files, then delete the row.

    Best-effort throughout: a trainer that no longer knows the run (404) or a
    file that's already gone doesn't block the teardown — anything non-fatal is
    collected into ``errors`` so the operator sees it but the row still goes.
    """
    outcome: dict = {"run_id": run.pk, "stopped": None, "removed_paths": [], "errors": []}

    # 1. Stop the training process on the trainer service.
    try:
        result = runner.stop(run)
        outcome["stopped"] = result.get("outcome") or result.get("status")
    except Exception as exc:  # noqa: BLE001 - never let a stop failure block cleanup
        outcome["errors"].append(f"stop failed: {exc}")

    # 2. Remove artifacts via the guarded cleanup service. Report only what was
    #    actually removed: a path the guard refused (outside the roots) still
    #    exists afterward and so is left out of removed_paths.
    candidates = [_resolve(raw) for raw in (run.output_dir, run.config_yaml_path) if raw]
    existed = [p for p in candidates if p.exists()]
    try:
        cleanup.remove_run_artifacts(run)
    except Exception as exc:  # noqa: BLE001 - never let cleanup block the row delete
        outcome["errors"].append(f"cleanup failed: {exc}")
    outcome["removed_paths"] = [str(p) for p in existed if not p.exists()]

    # 3. Delete the DB row (RunResult rows cascade; post_delete cleanup is a no-op
    #    now that the files are already gone).
    run.delete()
    outcome["deleted"] = True
    return outcome
