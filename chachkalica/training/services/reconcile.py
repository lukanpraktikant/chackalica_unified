"""Recover training/eval runs stranded at ``running``/``queued``.

A run's status is driven by a single long-lived RQ job (:mod:`training.jobs`)
that polls the trainer service until the run terminates. If that work-horse dies
mid-poll — killed by an RQ timeout, a worker restart, a deploy, or OOM — the
trainer keeps running to completion but nothing is left to finalize the row, so
it sticks at ``running`` forever (with the results only on disk).

This module re-derives the true state from the trainer service and the shared
filesystem and finalizes the row exactly as the lost job would have. It is:

* **idempotent** — safe to run repeatedly; and
* **conservative** — a run the trainer still reports as running, or one a live
  RQ job is still polling, is left untouched.

Run it via the ``reconcile_training`` management command (cron it for
hands-off recovery) or the "Reconcile status" admin action.
"""

import logging

import django_rq
from rq.registry import StartedJobRegistry

from training import jobs
from training.models import EvalRun, TrainingRun
from training.services import ingest, runner

logger = logging.getLogger(__name__)

_ACTIVE = (TrainingRun.RUNNING, TrainingRun.QUEUED)


def _has_live_job(func_suffix: str, pk: int) -> bool:
    """True if a queued or executing RQ job still owns ``(func, pk)``.

    Guards against declaring a still-queued or actively-polling run "lost".
    """
    q = django_rq.get_queue("default")
    job_ids = list(q.job_ids) + list(StartedJobRegistry(queue=q).get_job_ids())
    for jid in job_ids:
        job = q.fetch_job(jid)
        if job is None:
            continue
        if (job.func_name or "").endswith(func_suffix) and list(job.args or []) == [pk]:
            return True
    return False


def reconcile_run(run: TrainingRun) -> str:
    """Re-derive and finalize one stranded :class:`TrainingRun`. Returns an outcome."""
    try:
        status = runner.fetch_status(run)
    except Exception as exc:  # noqa: BLE001 - trainer unreachable; try again later
        logger.warning("reconcile: run #%s status unreachable: %s", run.pk, exc)
        return "unreachable"

    state = status.get("status")
    if state == "error":
        jobs._mark(run, TrainingRun.ERROR, error=status.get("log_tail", "")[-4000:],
                   finished=True)
        return "error"
    if state == "ok" or (state == "unknown" and ingest.is_complete(run.output_dir)):
        jobs.finalize_success(run)
        return "ok"
    if state == "running":
        # Genuinely training on the service; just make sure the row reflects it.
        if run.status != TrainingRun.RUNNING:
            run.status = TrainingRun.RUNNING
            run.save(update_fields=["status"])
        return "running"

    # state == "unknown" and nothing finished on disk: the trainer has no record.
    if _has_live_job("run_training", run.pk):
        return "pending"  # a poller (or queued job) still owns it — leave alone
    jobs._mark(run, TrainingRun.ERROR,
               error="run lost: trainer has no record and no summary on disk",
               finished=True)
    return "lost"


def reconcile_eval(eval_run: EvalRun) -> str:
    """Re-derive and finalize one stranded :class:`EvalRun`. Returns an outcome."""
    try:
        status = runner.fetch_eval_status(eval_run)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reconcile: eval #%s status unreachable: %s", eval_run.pk, exc)
        return "unreachable"

    state = status.get("status")
    if state == "error":
        jobs._mark_eval(eval_run, EvalRun.ERROR, error=status.get("log_tail", "")[-4000:],
                        finished=True)
        return "error"
    if state == "ok" or (state == "unknown" and ingest.eval_is_complete(eval_run.output_dir)):
        jobs.finalize_eval_success(eval_run)
        return "ok"
    if state == "running":
        if eval_run.status != EvalRun.RUNNING:
            eval_run.status = EvalRun.RUNNING
            eval_run.save(update_fields=["status"])
        return "running"

    if _has_live_job("run_eval", eval_run.pk):
        return "pending"
    jobs._mark_eval(eval_run, EvalRun.ERROR,
                    error="eval lost: trainer has no record and no result on disk",
                    finished=True)
    return "lost"


def reconcile_pipeline(pe) -> str:
    """Re-derive and finalize one stranded :class:`PipelineEvalRun`. Returns an outcome."""
    from eval_pipelines.models import PipelineEvalRun

    try:
        status = runner.fetch_pipeline_status(pe)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reconcile: pipeline eval #%s status unreachable: %s", pe.pk, exc)
        return "unreachable"

    state = status.get("status")
    if state == "error":
        jobs._mark_pipeline(pe, PipelineEvalRun.ERROR, error=status.get("log_tail", "")[-4000:],
                            finished=True)
        return "error"
    if state == "ok" or (state == "unknown" and ingest.pipeline_is_complete(pe.output_dir)):
        jobs.finalize_pipeline_success(pe)
        return "ok"
    if state == "running":
        if pe.status != PipelineEvalRun.RUNNING:
            pe.status = PipelineEvalRun.RUNNING
            pe.save(update_fields=["status"])
        return "running"

    if _has_live_job("run_pipeline_eval", pe.pk):
        return "pending"
    jobs._mark_pipeline(pe, PipelineEvalRun.ERROR,
                        error="pipeline eval lost: trainer has no record and no result on disk",
                        finished=True)
    return "lost"


def reconcile_all() -> dict:
    """Reconcile every training/eval run currently stuck at running/queued."""
    from eval_pipelines.models import PipelineEvalRun

    runs = {r.pk: reconcile_run(r) for r in TrainingRun.objects.filter(status__in=_ACTIVE)}
    evals = {e.pk: reconcile_eval(e) for e in EvalRun.objects.filter(status__in=_ACTIVE)}
    pipelines = {
        p.pk: reconcile_pipeline(p)
        for p in PipelineEvalRun.objects.filter(status__in=_ACTIVE)
    }
    return {"runs": runs, "evals": evals, "pipelines": pipelines}
