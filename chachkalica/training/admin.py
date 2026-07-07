"""Admin = the training operator console.

The Experiment page composes a friendy_chachkalica config from inline dataset/model
rows; the "Generate config & create run" action validates the roster, writes the
YAML, and records a :class:`TrainingRun`. Executing that run against the
trainer service is a later phase — for now the action ends at a generated,
ready-to-run config.
"""

import django_rq
from django.contrib import admin, messages
from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.template.response import TemplateResponse
from django.utils import timezone
from django.utils.html import format_html_join

from fleet.admin import _status_badge
from fleet.models import Annotator, Dataset
from training import jobs
from training.models import (
    EvalRun,
    Experiment,
    ExperimentDataset,
    ExperimentModel,
    RunResult,
    TrainedModel,
    TrainingRun,
    TrainingSettings,
)
from training import model_specs
from training.forms import ExperimentModelForm
from training.services import config_gen, ingest, promote, teardown


def _queue():
    return django_rq.get_queue("default")


@admin.register(TrainingSettings)
class TrainingSettingsAdmin(admin.ModelAdmin):
    list_display = ["__str__", "configs_root", "runs_root", "default_device", "service_base_url"]

    def has_add_permission(self, request):
        return not TrainingSettings.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False


class ExperimentDatasetInline(admin.TabularInline):
    model = ExperimentDataset
    extra = 1
    autocomplete_fields = ["dataset", "annotator"]
    fields = [
        "dataset", "role", "label_source", "annotator", "explicit_labels_path",
        "aug_hflip", "aug_hflip_fraction", "aug_scale_crop", "aug_scale_crop_fraction",
    ]

    def formfield_for_dbfield(self, db_field, request, **kwargs):
        # Tabular inlines squeeze help_text into a 10px hover-only icon in the
        # column header, which reads as missing. Mirror it onto the widget so
        # hovering the checkbox/input itself shows the tooltip too.
        field = super().formfield_for_dbfield(db_field, request, **kwargs)
        if field is not None and field.help_text:
            field.widget.attrs.setdefault("title", field.help_text)
        return field

    class Media:
        # Shows each augmentation's fraction input only while its checkbox is
        # ticked (train rows). Pure progressive enhancement — with JS off the
        # inputs stay visible and model.clean() still validates them.
        js = ("training/experiment_dataset_aug.js",)


class ExperimentModelInline(admin.StackedInline):
    # Stacked (not tabular) so each model gets a vertical form: the form exposes
    # a field per builder option of every arch (size/variant, thresholds, backbone
    # weights …) and JS hides the ones not matching the selected arch. num_classes
    # stays optional — blank means "auto", resolved per train dataset.
    model = ExperimentModel
    form = ExperimentModelForm
    extra = 1

    def get_fields(self, request, obj=None):
        # arch first, then every arch's builder-option widgets (JS shows only the
        # selected arch's), then the shared knobs. The spec fields are declared on
        # the form, so listing them here is safe for the inline formset factory.
        return ["arch", *model_specs.spec_field_names(), "pretrained", "num_classes", "params"]

    class Media:
        js = ("training/experiment_model_form.js",)


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    list_display = ["name", "dataset_count", "model_count", "epochs", "device", "updated_at"]
    search_fields = ["name", "description"]
    inlines = [ExperimentDatasetInline, ExperimentModelInline]
    actions = ["generate_run"]
    fieldsets = [
        (None, {"fields": ["name", "description"]}),
        (
            "Training",
            {
                "fields": [
                    "epochs", "batch_size", "num_workers", "device", "seed",
                    "amp", "gradient_clip_norm", "early_stopping_patience",
                    "optimizer_name", "lr", "weight_decay", "optimizer_params",
                    "scheduler_name", "scheduler_params",
                ]
            },
        ),
        (
            "Evaluation",
            {"fields": ["eval_batch_size", "eval_num_workers", "eval_score_threshold", "iou_thresholds"]},
        ),
    ]

    @admin.display(description="datasets")
    def dataset_count(self, obj):
        return obj.datasets.count()

    @admin.display(description="models")
    def model_count(self, obj):
        return obj.models.count()

    @admin.action(description="Generate config & create run…")
    def generate_run(self, request, queryset):
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one experiment.", level=messages.WARNING)
            return None
        experiment = queryset.first()

        # Validate the roster up front so both the preview and the apply path
        # surface the same friendly error.
        ts = TrainingSettings.load()
        try:
            if request.POST.get("apply"):
                run = TrainingRun.objects.create(experiment=experiment)
                yaml_path, _text = config_gen.write_config(experiment, run)
                _queue().enqueue(jobs.run_training, run.pk, job_timeout=jobs.JOB_TIMEOUT)
                run.status = TrainingRun.QUEUED
                run.save(update_fields=["status"])
                self.message_user(
                    request,
                    f"Run #{run.pk} queued — config written to {yaml_path}. "
                    "Watch the Training runs page for progress.",
                )
                return None
            provisional_output = config_gen._resolve(ts.runs_root) / f"{experiment.name}-<run id>"
            yaml_text = config_gen.build_yaml(experiment, provisional_output)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            self.message_user(request, f"Cannot generate config: {exc}", level=messages.ERROR)
            return None

        context = {
            **self.admin_site.each_context(request),
            "title": "Generate training config",
            "experiment": experiment,
            "datasets": list(experiment.datasets.all()),
            "models": list(experiment.models.all()),
            "yaml_text": yaml_text,
            "configs_root": config_gen._resolve(ts.configs_root),
            "action": "generate_run",
            "selected": [str(experiment.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/training/launch_run.html", context)


class RunResultInline(admin.TabularInline):
    model = RunResult
    extra = 0
    can_delete = False
    fields = ["run_name", "model_arch", "train_dataset_name", "map50", "map50_95",
              "best_epoch", "best_checkpoint"]
    readonly_fields = fields

    def has_add_permission(self, request, obj=None):
        return False

    @admin.display(description="mAP50")
    def map50(self, obj):
        return obj.metric("map50")

    @admin.display(description="mAP50-95")
    def map50_95(self, obj):
        return obj.metric("map50_95")


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ["__str__", "experiment", "status_badge", "config_yaml_path", "created_at"]
    list_filter = ["status", "experiment"]
    inlines = [RunResultInline]
    actions = ["launch_selected", "ingest_selected", "reconcile_selected", "kill_run_gracefully"]
    readonly_fields = [
        "experiment", "status", "epoch_progress", "config_yaml_path", "output_dir",
        "last_error", "started_at", "finished_at", "results", "created_at",
    ]

    def has_add_permission(self, request):
        # Runs are created by the Experiment action, not by hand.
        return False

    @admin.display(description="status", ordering="status")
    def status_badge(self, obj):
        return _status_badge(obj.status)

    @admin.display(description="Epoch progress")
    def epoch_progress(self, obj):
        """Per-epoch lines read live from each run's history.yaml on disk.

        Reflects training as it happens (history.yaml is rewritten each epoch);
        reload the page to refresh. One block per internal run, since a run fans
        out into several train-dataset × model pairings.
        """
        from training.services import progress

        histories = progress.run_histories(obj.output_dir)
        if not histories:
            return "No epoch history yet — reload while the run is training."
        return format_html_join(
            "",
            "<div style='margin-bottom:1em'><strong>{}</strong>"
            "<pre style='max-height:24em;overflow:auto;margin:.3em 0;padding:.6em;"
            "background:#1e1e1e;color:#d4d4d4;border-radius:4px;font-size:12px;"
            "line-height:1.6'>{}</pre></div>",
            (
                (h["run_name"], "\n".join(progress.epoch_line(e) for e in h["epochs"]))
                for h in histories
            ),
        )

    @admin.action(description="Launch / relaunch on trainer service")
    def launch_selected(self, request, queryset):
        queue = _queue()
        for run in queryset:
            if not run.config_yaml_path:
                self.message_user(request, f"Run #{run.pk} has no config; skipped.",
                                  level=messages.WARNING)
                continue
            queue.enqueue(jobs.run_training, run.pk, job_timeout=jobs.JOB_TIMEOUT)
            run.status = TrainingRun.QUEUED
            run.save(update_fields=["status"])
        self.message_user(request, "Launch job(s) queued — refresh to see progress.")

    @admin.action(description="Ingest results from output dir (no rerun)")
    def ingest_selected(self, request, queryset):
        from training.services import ingest

        for run in queryset:
            if not run.output_dir or not ingest.is_complete(run.output_dir):
                self.message_user(request, f"Run #{run.pk}: no finished output to ingest.",
                                  level=messages.WARNING)
                continue
            summary = ingest.ingest_run(run)
            run.status = TrainingRun.OK
            run.save(update_fields=["status"])
            self.message_user(request, f"Run #{run.pk}: ingested {summary['run_results']} result(s).")

    @admin.action(description="Reconcile status from trainer / disk")
    def reconcile_selected(self, request, queryset):
        from training.services import reconcile

        for run in queryset:
            outcome = reconcile.reconcile_run(run)
            self.message_user(request, f"Run #{run.pk}: {outcome}")

    @admin.action(description="Kill run gracefully (stop, delete run + files)")
    def kill_run_gracefully(self, request, queryset):
        """Stop the training process, then delete the run's files and DB row.

        Two-step: the first click shows a confirm page (with a warning for any
        promoted models whose checkpoints would be removed); the form re-POSTs
        with ``apply=1`` to actually tear down.
        """
        if request.POST.get("apply"):
            for run in queryset:
                outcome = teardown.kill_run(run)
                self.message_user(
                    request,
                    f"Run #{outcome['run_id']}: stopped ({outcome['stopped']}); "
                    f"removed {len(outcome['removed_paths'])} path(s); row deleted.",
                    level=messages.WARNING if outcome["errors"] else messages.SUCCESS,
                )
                for err in outcome["errors"]:
                    self.message_user(request, f"Run #{outcome['run_id']}: {err}",
                                      level=messages.ERROR)
            return None

        runs = list(queryset)
        affected = [
            {"run": run,
             "models": list(TrainedModel.objects.filter(source_run_result__run=run))}
            for run in runs
        ]
        context = {
            **self.admin_site.each_context(request),
            "title": "Kill training run(s) gracefully",
            "affected": affected,
            "action": "kill_run_gracefully",
            "selected": [str(run.pk) for run in runs],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/training/kill_run.html", context)


@admin.register(RunResult)
class RunResultAdmin(admin.ModelAdmin):
    list_display = ["run_name", "run", "model_arch", "train_dataset_name",
                    "map50", "map50_95", "best_epoch"]
    list_filter = ["model_arch", "train_dataset_name"]
    search_fields = ["run_name", "train_dataset_name"]
    actions = ["show_best_epoch_stats", "promote_selected"]

    def has_add_permission(self, request):
        return False

    @admin.display(description="mAP50")
    def map50(self, obj):
        return obj.metric("map50")

    @admin.display(description="mAP50-95")
    def map50_95(self, obj):
        return obj.metric("map50_95")

    @admin.action(description="Show best-epoch statistics")
    def show_best_epoch_stats(self, request, queryset):
        import yaml

        from training.services import progress

        if queryset.count() != 1:
            self.message_user(request, "Select exactly one result.", level=messages.WARNING)
            return None
        rr = queryset.first()
        entry = progress.best_epoch_entry(rr.run_dir, rr.best_epoch)

        def _dump(value):
            return yaml.safe_dump(value, sort_keys=False, default_flow_style=False) if value else ""

        context = {
            **self.admin_site.each_context(request),
            "title": f"Best-epoch statistics — {rr.run_name}",
            "rr": rr,
            "entry": entry,
            "entry_yaml": _dump(entry),
            "val_yaml": _dump(rr.val_metrics),
            "test_yaml": _dump(rr.test_metrics),
        }
        return TemplateResponse(request, "admin/training/best_epoch_stats.html", context)

    @admin.action(description="Promote to model registry")
    def promote_selected(self, request, queryset):
        promoted = 0
        for rr in queryset:
            try:
                tm = promote.promote_run_result(rr)
            except ValueError as exc:
                self.message_user(request, f"{rr.run_name}: {exc}", level=messages.WARNING)
                continue
            promoted += 1
            self.message_user(request, f"Registered model {tm.name!r}.")
        if promoted:
            self.message_user(request, f"Promoted {promoted} model(s) — see Trained models.")


@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ["name", "stage", "arch", "num_classes", "map50", "map50_95", "created_at"]
    list_filter = ["stage", "arch"]
    search_fields = ["name", "description"]
    actions = ["evaluate_on_dataset", "evaluate_with_pipeline"]
    readonly_fields = ["source_run_result", "created_at", "updated_at"]

    @admin.display(description="mAP50")
    def map50(self, obj):
        return obj.metrics.get("map50") if isinstance(obj.metrics, dict) else None

    @admin.display(description="mAP50-95")
    def map50_95(self, obj):
        return obj.metrics.get("map50_95") if isinstance(obj.metrics, dict) else None

    @admin.action(description="Evaluate on a dataset…")
    def evaluate_on_dataset(self, request, queryset):
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one model to evaluate.",
                              level=messages.WARNING)
            return None
        model = queryset.first()

        if request.POST.get("apply"):
            dataset = Dataset.objects.filter(pk=request.POST.get("dataset")).first()
            if dataset is None:
                self.message_user(request, "Choose a dataset.", level=messages.WARNING)
                return None
            label_source = request.POST.get("label_source") or EvalRun.SOURCE
            annotator = Annotator.objects.filter(pk=request.POST.get("annotator")).first()
            eval_run = EvalRun.objects.create(
                trained_model=model, dataset=dataset, label_source=label_source,
                annotator=annotator, explicit_labels_path=request.POST.get("explicit_labels_path", ""),
            )
            try:
                config_gen.write_eval_request(eval_run)
            except (ValueError, FileNotFoundError, RuntimeError) as exc:
                eval_run.delete()
                self.message_user(request, f"Cannot build eval request: {exc}", level=messages.ERROR)
                return None
            _queue().enqueue(jobs.run_eval, eval_run.pk, job_timeout=jobs.JOB_TIMEOUT)
            eval_run.status = EvalRun.QUEUED
            eval_run.save(update_fields=["status"])
            self.message_user(request, f"Eval #{eval_run.pk} queued for {model.name} on {dataset.name}.")
            return None

        context = {
            **self.admin_site.each_context(request),
            "title": f"Evaluate {model.name}",
            "model": model,
            "datasets": Dataset.objects.all(),
            "annotators": Annotator.objects.filter(status=Annotator.ACTIVE).order_by("username"),
            "label_source_choices": EvalRun._meta.get_field("label_source").choices,
            "action": "evaluate_on_dataset",
            "selected": [str(model.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/training/eval_model.html", context)

    @admin.action(description="Evaluate with a pipeline…")
    def evaluate_with_pipeline(self, request, queryset):
        from eval_pipelines.models import PipelineEvalRun

        if queryset.count() != 1:
            self.message_user(request, "Select exactly one model to evaluate.",
                              level=messages.WARNING)
            return None
        model = queryset.first()

        if request.POST.get("apply"):
            dataset = Dataset.objects.filter(pk=request.POST.get("dataset")).first()
            if dataset is None:
                self.message_user(request, "Choose a dataset.", level=messages.WARNING)
                return None
            pipeline = request.POST.get("pipeline") or PipelineEvalRun.BATCH_DETECT
            label_source = request.POST.get("label_source") or PipelineEvalRun.SOURCE
            annotator = Annotator.objects.filter(pk=request.POST.get("annotator")).first()

            def _int(name):
                raw = (request.POST.get(name) or "").strip()
                return int(raw) if raw else None

            def _float(name):
                raw = (request.POST.get(name) or "").strip()
                return float(raw) if raw else None

            pe = PipelineEvalRun.objects.create(
                trained_model=model, dataset=dataset, label_source=label_source,
                annotator=annotator,
                explicit_labels_path=request.POST.get("explicit_labels_path", ""),
                pipeline=pipeline,
                detector_checkpoint=(request.POST.get("detector_checkpoint") or "").strip(),
                tile_size=_int("tile_size"), overlap=_float("overlap"),
            )
            try:
                config_gen.write_pipeline_request(pe)
            except (ValueError, FileNotFoundError, RuntimeError) as exc:
                pe.delete()
                self.message_user(request, f"Cannot build pipeline request: {exc}",
                                  level=messages.ERROR)
                return None
            _queue().enqueue(jobs.run_pipeline_eval, pe.pk, job_timeout=jobs.JOB_TIMEOUT)
            pe.status = PipelineEvalRun.QUEUED
            pe.save(update_fields=["status"])
            self.message_user(
                request,
                f"Pipeline eval #{pe.pk} ({pipeline}) queued for {model.name} on {dataset.name}.")
            return None

        context = {
            **self.admin_site.each_context(request),
            "title": f"Evaluate {model.name} with a pipeline",
            "model": model,
            "datasets": Dataset.objects.all(),
            "annotators": Annotator.objects.filter(status=Annotator.ACTIVE).order_by("username"),
            "label_source_choices": PipelineEvalRun._meta.get_field("label_source").choices,
            "pipeline_choices": PipelineEvalRun.PIPELINE_CHOICES,
            "action": "evaluate_with_pipeline",
            "selected": [str(model.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/training/pipeline_eval_model.html", context)


# EvalRun's admin lives in ``eval_pipelines.admin`` (as the "Base Eval" proxy)
# so the base and pipeline evals sit together under the "Eval Pipelines" section.
