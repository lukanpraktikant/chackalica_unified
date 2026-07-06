"""Signal handlers wiring model deletion to on-disk cleanup.

Deleting a training/eval run in the admin should also remove its artifacts under
``data/`` (checkpoints, logs, generated config). ``post_delete`` fires for single
deletes, bulk "delete selected", and cascades alike, so this is the one place
that covers every deletion path. Cleanup never raises — a filesystem hiccup must
not roll back the row deletion.
"""

import logging

from django.db.models.signals import post_delete
from django.dispatch import receiver

from training.models import EvalRun, TrainingRun
from training.services import cleanup

logger = logging.getLogger(__name__)


@receiver(post_delete, sender=TrainingRun, dispatch_uid="training_run_cleanup")
def _training_run_deleted(sender, instance, **kwargs):
    try:
        cleanup.remove_run_artifacts(instance)
    except Exception:  # noqa: BLE001 - never let cleanup break the delete
        logger.exception("cleanup failed for TrainingRun #%s", instance.pk)


@receiver(post_delete, sender=EvalRun, dispatch_uid="eval_run_cleanup")
def _eval_run_deleted(sender, instance, **kwargs):
    try:
        cleanup.remove_eval_artifacts(instance)
    except Exception:  # noqa: BLE001
        logger.exception("cleanup failed for EvalRun #%s", instance.pk)
