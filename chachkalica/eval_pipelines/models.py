"""Chachak pipeline evals, and the "Eval Pipelines" admin section.

This app owns nothing but the eval-comparison corner of the admin: a concrete
:class:`PipelineEvalRun` (one run of a chachak pipeline against a dataset) plus a
:class:`BaseEval` proxy that re-homes the existing :class:`training.EvalRun` under
this section so the base and pipeline evals sit side by side.

A ``PipelineEvalRun`` deliberately mirrors ``EvalRun``: the admin action writes a
chachak request YAML and enqueues a job that drives the trainer service's
``/pipeline`` endpoint, then ingests the resulting metrics back here. The extra
fields describe *which* pipeline and its detector/tiling knobs.
"""

from django.db import models

from fleet.models import Annotator, Dataset
from training import pipelines
from training.models import EvalRun, ExperimentDataset, TrainedModel, default_iou_thresholds


class PipelineEvalRun(models.Model):
    """One evaluation of a :class:`TrainedModel` through a chachak pipeline."""

    # Status vocabulary matches EvalRun so the shared status badge/analytics work.
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    STATUS_CHOICES = [
        (CREATED, "created"),
        (QUEUED, "queued"),
        (RUNNING, "running"),
        (OK, "ok"),
        (ERROR, "error"),
    ]

    # Reuse the label-source vocabulary from ExperimentDataset (as EvalRun does).
    SOURCE = ExperimentDataset.SOURCE
    ANNOTATOR = ExperimentDataset.ANNOTATOR
    EXPLICIT = ExperimentDataset.EXPLICIT

    # Pipeline vocabulary lives in ``training.pipelines`` so the Experiment and
    # this model share one definition (kept in sync with chachak.config).
    BATCH_DETECT = pipelines.BATCH_DETECT
    PEOPLE_DETECT_FIRST = pipelines.PEOPLE_DETECT_FIRST
    BATCH_PEOPLE = pipelines.BATCH_PEOPLE
    CHAIN = pipelines.CHAIN
    PIPELINE_CHOICES = pipelines.PIPELINE_CHOICES
    # Pipelines that require a person detector checkpoint.
    DETECTOR_PIPELINES = pipelines.DETECTOR_PIPELINES

    trained_model = models.ForeignKey(
        TrainedModel, on_delete=models.CASCADE, related_name="pipeline_eval_runs"
    )
    dataset = models.ForeignKey(Dataset, on_delete=models.PROTECT, related_name="+")
    label_source = models.CharField(
        max_length=16,
        choices=ExperimentDataset.LABEL_SOURCE_CHOICES,
        default=ExperimentDataset.SOURCE,
    )
    annotator = models.ForeignKey(
        Annotator, on_delete=models.PROTECT, null=True, blank=True, related_name="+"
    )
    explicit_labels_path = models.CharField(max_length=1024, blank=True)
    map_score_threshold = models.FloatField(
        default=0.001,
        help_text="Minimum prediction confidence kept for AP/mAP. Keep this low so mAP can "
                  "sweep the precision-recall curve.",
    )
    score_threshold = models.FloatField(
        default=0.25,
        help_text="Operating-point confidence threshold used for precision, recall, F1, "
                  "prediction counts, and the confusion matrix.",
    )

    pipeline = models.CharField(max_length=32, choices=PIPELINE_CHOICES, default=BATCH_DETECT)
    detector_checkpoint = models.CharField(
        max_length=1024, blank=True,
        help_text="Person-detector checkpoint; required for people_detect_first / batch_people.",
    )
    tile_width_pct = models.FloatField(
        null=True, blank=True,
        help_text="Tile width as a percent (0–100] of each image's width.",
    )
    tile_height_pct = models.FloatField(
        null=True, blank=True,
        help_text="Tile height as a percent (0–100] of each image's height.",
    )
    overlap = models.FloatField(null=True, blank=True)
    chain = models.JSONField(
        default=list, blank=True,
        help_text="Ordered pipeline names for the 'chain' pipeline.",
    )

    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=CREATED)
    request_yaml_path = models.CharField(max_length=1024, blank=True)
    output_dir = models.CharField(max_length=1024, blank=True)
    metrics = models.JSONField(null=True, blank=True)
    last_error = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "All pipeline eval"
        verbose_name_plural = "All pipeline evals"

    def __str__(self) -> str:
        return (
            f"Pipeline Eval #{self.pk} — {self.trained_model.name} "
            f"[{self.pipeline}] on {self.dataset.name}"
        )

    def metric(self, key: str):
        return self.metrics.get(key) if isinstance(self.metrics, dict) else None

    def default_iou_thresholds(self):
        return default_iou_thresholds()


class BaseEval(EvalRun):
    """Proxy of :class:`training.EvalRun`, re-homed under "Eval Pipelines".

    Defined in this app, it inherits ``app_label="eval_pipelines"`` so the admin
    lists it in this section (renamed "Base Eval") while sharing EvalRun's table.
    """

    class Meta:
        proxy = True
        verbose_name = "Base Eval"
        verbose_name_plural = "Base Eval"


def _pipeline_manager(pipeline_value: str) -> models.Manager:
    """Manager that scopes a proxy to a single ``pipeline`` value.

    Each per-pipeline proxy below shares ``PipelineEvalRun``'s table; the manager
    filters it down so the proxy's admin list shows only that pipeline's runs.
    """

    class _Manager(models.Manager):
        def get_queryset(self):
            return super().get_queryset().filter(pipeline=pipeline_value)

    return _Manager()


# Per-pipeline proxies — one admin list per pipeline type. Each shares
# ``PipelineEvalRun``'s table but its manager scopes the queryset to a single
# pipeline, so a run created by the "Evaluate…" action appears under the list
# matching the pipeline chosen there.
class BatchDetectEval(PipelineEvalRun):
    objects = _pipeline_manager(PipelineEvalRun.BATCH_DETECT)

    class Meta:
        proxy = True
        verbose_name = "Batch detect eval"
        verbose_name_plural = "Batch detect"


class PeopleDetectFirstEval(PipelineEvalRun):
    objects = _pipeline_manager(PipelineEvalRun.PEOPLE_DETECT_FIRST)

    class Meta:
        proxy = True
        verbose_name = "People detect first eval"
        verbose_name_plural = "People detect first"


class BatchPeopleEval(PipelineEvalRun):
    objects = _pipeline_manager(PipelineEvalRun.BATCH_PEOPLE)

    class Meta:
        proxy = True
        verbose_name = "Batch people eval"
        verbose_name_plural = "Batch people"


class ChainEval(PipelineEvalRun):
    objects = _pipeline_manager(PipelineEvalRun.CHAIN)

    class Meta:
        proxy = True
        verbose_name = "Chain eval"
        verbose_name_plural = "Chain"
