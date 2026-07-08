"""Admin = the training operator console.

The Experiment page composes a friendy_chachkalica config from inline dataset/model
rows; the "Generate config & create run" action validates the roster, writes the
YAML, and records a :class:`TrainingRun`. Executing that run against the
trainer service is a later phase — for now the action ends at a generated,
ready-to-run config.
"""

from pathlib import Path

import django_rq
from django.contrib import admin, messages
from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils import timezone
from django.utils.html import format_html_join
from django.utils.http import urlencode

from fleet.admin import _status_badge
from fleet.models import Annotator, Dataset
from fleet.services import lsapi
from fleet.services.paths import source_root
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
from training.services import config_gen, ingest, promote, runner, teardown


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
              "best_epoch", "best_checkpoint", "error"]
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
                    "map50", "map50_95", "best_epoch", "error"]
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


def _float_or_none(raw):
    raw = (raw or "").strip()
    return float(raw) if raw else None


def _preview_index(request, count):
    """Parse a 0-based ``?index=`` and bound it to ``[0, count)`` or 404."""
    if count <= 0:
        raise Http404("dataset has no images")
    try:
        index = int(request.GET.get("index", 0))
    except (TypeError, ValueError):
        raise Http404("bad index")
    if not 0 <= index < count:
        raise Http404("index out of range")
    return index


def _preview_label_dir(request, dataset):
    """Resolve the ground-truth label dir from the preview query, or None.

    Returns None when no label source was chosen or the resolved dir is missing,
    so the viewer can disable the GT toggle instead of erroring.
    """
    label_source = request.GET.get("label_source") or ""
    if not label_source:
        return None
    annotator = Annotator.objects.filter(pk=request.GET.get("annotator")).first()
    try:
        label_dir = config_gen.resolve_label_dir(
            dataset, label_source, annotator,
            (request.GET.get("explicit_labels_path") or "").strip())
    except (ValueError, FileNotFoundError):
        return None
    return label_dir if Path(label_dir).is_dir() else None


def _read_gt_boxes(label_dir, image_path, class_names):
    """Parse a YOLO ``<stem>.txt`` into normalized center-xywh GT box dicts.

    Matches the pairing Friendy uses (``stem.txt`` or ``image.jpg.txt``); lines
    are ``class_id cx cy w h`` already normalized, so they draw on the same path
    as predictions. Extra values (polygons/OBB) beyond the first five are ignored.
    """
    label_dir = Path(label_dir)
    candidates = [label_dir / f"{image_path.stem}.txt", label_dir / f"{image_path.name}.txt"]
    label_file = next((c for c in candidates if c.exists()), None)
    if label_file is None:
        return []
    boxes = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        boxes.append({"cx": cx, "cy": cy, "w": w, "h": h,
                      "class_id": class_id, "class_name": name})
    return boxes


@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ["name", "stage", "arch", "num_classes", "map50", "map50_95", "created_at"]
    list_filter = ["stage", "arch"]
    search_fields = ["name", "description"]
    actions = ["evaluate", "preview_on_dataset"]
    readonly_fields = ["source_run_result", "created_at", "updated_at"]

    @admin.display(description="mAP50")
    def map50(self, obj):
        return obj.metrics.get("map50") if isinstance(obj.metrics, dict) else None

    @admin.display(description="mAP50-95")
    def map50_95(self, obj):
        return obj.metrics.get("map50_95") if isinstance(obj.metrics, dict) else None

    # Field visibility per pipeline (the template shows/hides these). A regular
    # eval (blank pipeline) shows none of them; keep this in sync with
    # ``config_gen.build_pipeline_request`` and ``PipelineEvalRun.DETECTOR_PIPELINES``.
    NO_PIPELINE = ("", "No pipeline — regular eval")

    @admin.action(description="Evaluate…")
    def evaluate(self, request, queryset):
        """Evaluate one model — optionally through a chachak pipeline.

        Merges the old "Evaluate on a dataset" and "Evaluate with a pipeline"
        actions: leave *Pipeline* blank for a plain :class:`EvalRun`, or pick one
        to create a :class:`PipelineEvalRun`. The form renders only the knobs the
        chosen pipeline uses (detector / tiling / chain).
        """
        from eval_pipelines.models import PipelineEvalRun

        if queryset.count() != 1:
            self.message_user(request, "Select exactly one model to evaluate.",
                              level=messages.WARNING)
            return None
        model = queryset.first()

        if request.POST.get("apply"):
            dataset = Dataset.objects.filter(pk=request.POST.get("dataset") or None).first()
            if dataset is None:
                self.message_user(request, "Choose a dataset.", level=messages.WARNING)
                return None
            label_source = request.POST.get("label_source") or EvalRun.SOURCE
            annotator = Annotator.objects.filter(pk=request.POST.get("annotator") or None).first()
            explicit = request.POST.get("explicit_labels_path", "")
            pipeline = (request.POST.get("pipeline") or "").strip()

            def _threshold(name, default):
                raw = (request.POST.get(name) or "").strip()
                try:
                    return float(raw) if raw else default
                except ValueError:
                    return default

            def _map_score_threshold():
                return _threshold("map_score_threshold", 0.001)

            def _score_threshold():
                return _threshold("score_threshold", 0.25)

            if not pipeline:
                # No pipeline chosen — a plain EvalRun (the old "evaluate on a dataset").
                eval_run = EvalRun.objects.create(
                    trained_model=model, dataset=dataset, label_source=label_source,
                    annotator=annotator, explicit_labels_path=explicit,
                    map_score_threshold=_map_score_threshold(),
                    score_threshold=_score_threshold(),
                )
                try:
                    config_gen.write_eval_request(eval_run)
                except (ValueError, FileNotFoundError, RuntimeError) as exc:
                    eval_run.delete()
                    self.message_user(request, f"Cannot build eval request: {exc}",
                                      level=messages.ERROR)
                    return None
                _queue().enqueue(jobs.run_eval, eval_run.pk, job_timeout=jobs.JOB_TIMEOUT)
                eval_run.status = EvalRun.QUEUED
                eval_run.save(update_fields=["status"])
                self.message_user(
                    request, f"Eval #{eval_run.pk} queued for {model.name} on {dataset.name}.")
                return None

            # A pipeline was chosen — a PipelineEvalRun.
            def _float(name):
                raw = (request.POST.get(name) or "").strip()
                return float(raw) if raw else None

            chain = [c.strip() for c in (request.POST.get("chain") or "").split(",") if c.strip()]
            pe = PipelineEvalRun.objects.create(
                trained_model=model, dataset=dataset, label_source=label_source,
                annotator=annotator, explicit_labels_path=explicit,
                pipeline=pipeline,
                detector_checkpoint=(request.POST.get("detector_checkpoint") or "").strip(),
                tile_width_pct=_float("tile_width_pct"),
                tile_height_pct=_float("tile_height_pct"),
                overlap=_float("overlap"),
                chain=chain,
                map_score_threshold=_map_score_threshold(),
                score_threshold=_score_threshold(),
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
            "title": f"Evaluate {model.name}",
            "model": model,
            "datasets": Dataset.objects.all(),
            "annotators": Annotator.objects.filter(status=Annotator.ACTIVE).order_by("username"),
            "label_source_choices": PipelineEvalRun._meta.get_field("label_source").choices,
            "pipeline_choices": [self.NO_PIPELINE, *PipelineEvalRun.PIPELINE_CHOICES],
            "detector_pipelines": " ".join(sorted(PipelineEvalRun.DETECTOR_PIPELINES)),
            "tiling_pipelines": " ".join([
                PipelineEvalRun.BATCH_DETECT, PipelineEvalRun.BATCH_PEOPLE, PipelineEvalRun.CHAIN]),
            "chain_pipelines": PipelineEvalRun.CHAIN,
            "action": "evaluate",
            "selected": [str(model.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/training/evaluate_model.html", context)

    # ------------------------------------------------------------------ preview
    RAW_PIPELINE = ("raw", "Raw model (no pipeline)")

    @admin.action(description="Preview model on selected dataset…")
    def preview_on_dataset(self, request, queryset):
        """Open the interactive box-preview viewer for one model on a dataset.

        Unlike the eval actions this persists nothing — on submit it just
        redirects to the viewer with the chosen settings as query params.
        """
        from eval_pipelines.models import PipelineEvalRun

        if queryset.count() != 1:
            self.message_user(request, "Select exactly one model to preview.",
                              level=messages.WARNING)
            return None
        model = queryset.first()

        if request.POST.get("apply"):
            dataset = Dataset.objects.filter(pk=request.POST.get("dataset") or None).first()
            if dataset is None:
                self.message_user(request, "Choose a dataset.", level=messages.WARNING)
                return None
            params = {
                "model": model.pk,
                "dataset": dataset.pk,
                "pipeline": request.POST.get("pipeline") or self.RAW_PIPELINE[0],
                "detector_checkpoint": (request.POST.get("detector_checkpoint") or "").strip(),
                "tile_width_pct": (request.POST.get("tile_width_pct") or "").strip(),
                "tile_height_pct": (request.POST.get("tile_height_pct") or "").strip(),
                "overlap": (request.POST.get("overlap") or "").strip(),
                "chain": (request.POST.get("chain") or "").strip(),
                "score": (request.POST.get("score") or "").strip(),
                "label_source": request.POST.get("label_source") or "",
                "annotator": request.POST.get("annotator") or "",
                "explicit_labels_path": (request.POST.get("explicit_labels_path") or "").strip(),
            }
            query = urlencode({k: v for k, v in params.items() if v not in ("", None)})
            return redirect(reverse("admin:training_trainedmodel_preview") + "?" + query)

        context = {
            **self.admin_site.each_context(request),
            "title": f"Preview {model.name} on a dataset",
            "model": model,
            "datasets": Dataset.objects.all(),
            "annotators": Annotator.objects.filter(status=Annotator.ACTIVE).order_by("username"),
            "label_source_choices": PipelineEvalRun._meta.get_field("label_source").choices,
            "pipeline_choices": [self.RAW_PIPELINE, *PipelineEvalRun.PIPELINE_CHOICES],
            "detector_pipelines": " ".join(sorted(PipelineEvalRun.DETECTOR_PIPELINES)),
            "tiling_pipelines": " ".join([
                PipelineEvalRun.BATCH_DETECT, PipelineEvalRun.BATCH_PEOPLE, PipelineEvalRun.CHAIN]),
            "chain_pipelines": PipelineEvalRun.CHAIN,
            "action": "preview_on_dataset",
            "selected": [str(model.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/training/preview_model.html", context)

    def get_urls(self):
        custom = [
            path("preview/", self.admin_site.admin_view(self.preview_view),
                 name="training_trainedmodel_preview"),
            path("preview/image/", self.admin_site.admin_view(self.preview_image),
                 name="training_trainedmodel_preview_image"),
            path("preview/data/", self.admin_site.admin_view(self.preview_data),
                 name="training_trainedmodel_preview_data"),
        ]
        return custom + super().get_urls()

    def preview_view(self, request):
        """Render the viewer shell; the browser pulls images + boxes per index."""
        model = TrainedModel.objects.filter(pk=request.GET.get("model")).first()
        dataset = Dataset.objects.filter(pk=request.GET.get("dataset")).first()
        if model is None or dataset is None:
            raise Http404("preview requires ?model= and ?dataset=")

        images = lsapi.list_dataset_images(source_root() / dataset.name)
        has_labels = _preview_label_dir(request, dataset) is not None

        context = {
            **self.admin_site.each_context(request),
            "title": f"Preview {model.name} on {dataset.name}",
            "model": model,
            "dataset": dataset,
            "pipeline": request.GET.get("pipeline") or self.RAW_PIPELINE[0],
            "image_count": len(images),
            "has_labels": has_labels,
            "query": request.GET.urlencode(),
        }
        return TemplateResponse(request, "admin/training/preview_viewer.html", context)

    def preview_image(self, request):
        """Stream the raw bytes of the dataset image at ``?index=``."""
        dataset = Dataset.objects.filter(pk=request.GET.get("dataset")).first()
        if dataset is None:
            raise Http404("unknown dataset")
        images = lsapi.list_dataset_images(source_root() / dataset.name)
        index = _preview_index(request, len(images))
        return FileResponse(open(images[index], "rb"))

    def preview_data(self, request):
        """Run inference for one image and return predictions (+ optional GT)."""
        model = TrainedModel.objects.filter(pk=request.GET.get("model")).first()
        dataset = Dataset.objects.filter(pk=request.GET.get("dataset")).first()
        if model is None or dataset is None:
            return JsonResponse({"error": "unknown model or dataset"}, status=400)

        images = lsapi.list_dataset_images(source_root() / dataset.name)
        if not images:
            return JsonResponse({"error": "dataset has no images"}, status=400)
        index = _preview_index(request, len(images))
        image_path = images[index]

        score = _float_or_none(request.GET.get("score"))
        chain = [
            c.strip() for c in (request.GET.get("chain") or "").split(",")
            if c.strip()
        ]
        try:
            payload = config_gen.build_preview_request(
                model,
                request.GET.get("pipeline") or self.RAW_PIPELINE[0],
                str(image_path),
                detector_checkpoint=(request.GET.get("detector_checkpoint") or "").strip(),
                tile_width_pct=_float_or_none(request.GET.get("tile_width_pct")),
                tile_height_pct=_float_or_none(request.GET.get("tile_height_pct")),
                overlap=_float_or_none(request.GET.get("overlap")),
                chain=chain,
                score_threshold=0.05 if score is None else score,
            )
            result = runner.predict_image(payload)
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001 — surface trainer/network errors to the viewer
            return JsonResponse({"error": f"inference failed: {exc}"}, status=502)

        label_dir = _preview_label_dir(request, dataset)
        ground_truth = (
            _read_gt_boxes(label_dir, image_path, config_gen.dataset_classes(dataset))
            if label_dir is not None else []
        )
        return JsonResponse({
            "predictions": result.get("boxes", []),
            "ground_truth": ground_truth,
            "classes": result.get("classes", {}),
            "image": image_path.name,
            "index": index,
            "count": len(images),
        })


# EvalRun's admin lives in ``eval_pipelines.admin`` (as the "Base Eval" proxy)
# so the base and pipeline evals sit together under the "Eval Pipelines" section.
