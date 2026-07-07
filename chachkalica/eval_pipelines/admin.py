"""The "Eval Pipelines" admin section: Base Eval + Pipeline Eval.

Base Eval is the existing :class:`training.EvalRun` re-homed here via the
:class:`~eval_pipelines.models.BaseEval` proxy (its admin was moved out of
``training/admin.py`` so it appears in this section only). Pipeline Eval is the
new :class:`~eval_pipelines.models.PipelineEvalRun`. Both share the read-only,
job-driven shape and the "Analyze" action that renders ``eval_analytics.compare``.
"""

import django_rq
from django.contrib import admin, messages
from django.template.response import TemplateResponse

from fleet.admin import _status_badge
from training import jobs

from eval_pipelines.models import BaseEval, PipelineEvalRun


def _queue():
    return django_rq.get_queue("default")


def _analyze(model_admin, request, queryset, title):
    """Shared "Analyze / compare" action body for base and pipeline evals."""
    from training.services import eval_analytics

    runs = [e for e in queryset if isinstance(e.metrics, dict) and e.metrics]
    skipped = [e for e in queryset if not (isinstance(e.metrics, dict) and e.metrics)]
    if skipped:
        model_admin.message_user(
            request,
            "Skipped eval(s) with no ingested metrics yet: "
            + ", ".join(f"#{e.pk}" for e in skipped),
            level=messages.WARNING,
        )
    if not runs:
        model_admin.message_user(request, "No evaluated metrics to analyze.",
                                 level=messages.WARNING)
        return None

    context = {
        **model_admin.admin_site.each_context(request),
        "title": title,
        **eval_analytics.compare(runs),
    }
    return TemplateResponse(request, "admin/training/eval_analytics.html", context)


@admin.register(BaseEval)
class BaseEvalAdmin(admin.ModelAdmin):
    list_display = ["__str__", "trained_model", "dataset", "status_badge",
                    "map50", "map50_95", "eval_time", "created_at"]
    list_filter = ["status", "trained_model"]
    actions = ["analyze_selected", "launch_selected", "reconcile_selected"]
    readonly_fields = [
        "trained_model", "dataset", "label_source", "annotator", "explicit_labels_path",
        "status", "request_yaml_path", "output_dir", "metrics", "last_error",
        "started_at", "finished_at", "created_at",
    ]

    def has_add_permission(self, request):
        return False

    @admin.action(description="Analyze / compare metrics of selected eval(s)…")
    def analyze_selected(self, request, queryset):
        return _analyze(self, request, queryset, "Eval metrics comparison")

    @admin.display(description="status", ordering="status")
    def status_badge(self, obj):
        return _status_badge(obj.status)

    @admin.display(description="mAP50")
    def map50(self, obj):
        return obj.metric("map50")

    @admin.display(description="mAP50-95")
    def map50_95(self, obj):
        return obj.metric("map50_95")

    @admin.display(description="eval time")
    def eval_time(self, obj):
        seconds = obj.metric("eval_seconds")
        return f"{seconds:.1f}s" if isinstance(seconds, (int, float)) else "—"

    @admin.action(description="Launch / relaunch on trainer service")
    def launch_selected(self, request, queryset):
        queue = _queue()
        for eval_run in queryset:
            if not eval_run.request_yaml_path:
                self.message_user(request, f"Eval #{eval_run.pk} has no request; skipped.",
                                  level=messages.WARNING)
                continue
            queue.enqueue(jobs.run_eval, eval_run.pk, job_timeout=jobs.JOB_TIMEOUT)
            eval_run.status = BaseEval.QUEUED
            eval_run.save(update_fields=["status"])
        self.message_user(request, "Eval job(s) queued — refresh to see progress.")

    @admin.action(description="Reconcile status from trainer / disk")
    def reconcile_selected(self, request, queryset):
        from training.services import reconcile

        for eval_run in queryset:
            outcome = reconcile.reconcile_eval(eval_run)
            self.message_user(request, f"Eval #{eval_run.pk}: {outcome}")


@admin.register(PipelineEvalRun)
class PipelineEvalRunAdmin(admin.ModelAdmin):
    list_display = ["__str__", "trained_model", "pipeline", "dataset", "status_badge",
                    "map50", "map50_95", "eval_time", "created_at"]
    list_filter = ["status", "pipeline", "trained_model"]
    actions = ["analyze_selected", "launch_selected", "reconcile_selected"]
    readonly_fields = [
        "trained_model", "dataset", "label_source", "annotator", "explicit_labels_path",
        "pipeline", "detector_checkpoint", "tile_size", "overlap", "chain",
        "status", "request_yaml_path", "output_dir", "metrics", "last_error",
        "started_at", "finished_at", "created_at",
    ]

    def has_add_permission(self, request):
        return False

    @admin.action(description="Analyze / compare metrics of selected pipeline eval(s)…")
    def analyze_selected(self, request, queryset):
        return _analyze(self, request, queryset, "Pipeline eval metrics comparison")

    @admin.display(description="status", ordering="status")
    def status_badge(self, obj):
        return _status_badge(obj.status)

    @admin.display(description="mAP50")
    def map50(self, obj):
        return obj.metric("map50")

    @admin.display(description="mAP50-95")
    def map50_95(self, obj):
        return obj.metric("map50_95")

    @admin.display(description="eval time")
    def eval_time(self, obj):
        seconds = obj.metric("eval_seconds")
        return f"{seconds:.1f}s" if isinstance(seconds, (int, float)) else "—"

    @admin.action(description="Launch / relaunch on trainer service")
    def launch_selected(self, request, queryset):
        queue = _queue()
        for pe in queryset:
            if not pe.request_yaml_path:
                self.message_user(request, f"Pipeline eval #{pe.pk} has no request; skipped.",
                                  level=messages.WARNING)
                continue
            queue.enqueue(jobs.run_pipeline_eval, pe.pk, job_timeout=jobs.JOB_TIMEOUT)
            pe.status = PipelineEvalRun.QUEUED
            pe.save(update_fields=["status"])
        self.message_user(request, "Pipeline eval job(s) queued — refresh to see progress.")

    @admin.action(description="Reconcile status from trainer / disk")
    def reconcile_selected(self, request, queryset):
        from training.services import reconcile

        for pe in queryset:
            outcome = reconcile.reconcile_pipeline(pe)
            self.message_user(request, f"Pipeline eval #{pe.pk}: {outcome}")
