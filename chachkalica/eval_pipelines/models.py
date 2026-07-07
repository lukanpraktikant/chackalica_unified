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

    # Keep in sync with chachak.config.PIPELINE_NAMES — we can't import chachak
    # here (its package/pipeline imports pull in torch, absent in the app env).
    BATCH_DETECT = "batch_detect"
    PEOPLE_DETECT_FIRST = "people_detect_first"
    BATCH_PEOPLE = "batch_people"
    CHAIN = "chain"
    PIPELINE_CHOICES = [
        (BATCH_DETECT, "batch_detect — tile, detect per tile, merge"),
        (PEOPLE_DETECT_FIRST, "people_detect_first — detect people, crop, detect per crop"),
        (BATCH_PEOPLE, "batch_people — tile, detect people, crop, detect per crop"),
        (CHAIN, "chain — run several pipelines and merge"),
    ]
    # Pipelines that require a person detector checkpoint.
    DETECTOR_PIPELINES = {PEOPLE_DETECT_FIRST, BATCH_PEOPLE}

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

    pipeline = models.CharField(max_length=32, choices=PIPELINE_CHOICES, default=BATCH_DETECT)
    detector_checkpoint = models.CharField(
        max_length=1024, blank=True,
        help_text="Person-detector checkpoint; required for people_detect_first / batch_people.",
    )
    tile_size = models.PositiveIntegerField(null=True, blank=True)
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
        verbose_name = "Pipeline Eval"
        verbose_name_plural = "Pipeline Eval"

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
