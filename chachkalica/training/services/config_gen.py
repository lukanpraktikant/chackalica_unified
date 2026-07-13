"""Turn an :class:`~training.models.Experiment` into friendy_chachkalica YAML.

friendy_chachkalica (``/home/luka/workspace/chachkalica/friendy_chachkalica``) is config-driven: it
reads an experiment YAML whose ``datasets.{train(list),val,test}`` entries each
point at an ``images`` dir and a ``labels`` dir with a ``classes`` list, plus
``models``/``training``/``evaluation`` blocks. We generate that YAML with
**absolute** paths so it resolves regardless of where the trainer runs, reusing
the same on-disk resolvers the fleet annotation side already uses.
"""

from pathlib import Path

import yaml
from django.conf import settings

from fleet.services import datasets as datasets_svc
from fleet.services import lsapi
from fleet.services.paths import source_root, target_root
from training.models import (
    Experiment,
    ExperimentDataset,
    ExperimentModel,
    TrainingSettings,
    default_iou_thresholds,
)


def _resolve(path: str) -> Path:
    """Absolute path: as-is if absolute, else relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else Path(settings.BASE_DIR) / p


def resolve_label_dir(dataset, label_source: str, annotator=None, explicit_path: str = "") -> Path:
    """Resolve a dataset's labels directory for a given label-source choice.

    Shared by training (:class:`ExperimentDataset`) and standalone eval
    (:class:`EvalRun`) so both pick labels the same way:
    ``source`` -> data/source/<name>/labels, ``annotator`` ->
    data/target/<name>/<username>, ``explicit`` -> a given path.
    """
    if label_source == ExperimentDataset.SOURCE:
        return datasets_svc.labels_source_dir(dataset)
    if label_source == ExperimentDataset.ANNOTATOR:
        if annotator is None:
            raise ValueError(f"{dataset.name}: annotator output selected but no annotator set.")
        return target_root() / dataset.name / annotator.username
    if label_source == ExperimentDataset.EXPLICIT:
        if not (explicit_path or "").strip():
            raise ValueError(f"{dataset.name}: explicit label path selected but empty.")
        return _resolve(explicit_path.strip())
    raise ValueError(f"Unknown label source {label_source!r}")


def label_dir(exp_dataset: ExperimentDataset) -> Path:
    """Resolve the labels directory feeding this dataset, per its label source."""
    return resolve_label_dir(
        exp_dataset.dataset, exp_dataset.label_source,
        exp_dataset.annotator, exp_dataset.explicit_labels_path,
    )


def dataset_classes(dataset) -> list[str]:
    """Class names for a dataset, from its on-disk classes.txt."""
    classes, _tools = lsapi.parse_classes_file(source_root() / dataset.name / "classes.txt")
    return classes


def images_dir(dataset) -> Path:
    return lsapi.image_source_dir(source_root() / dataset.name)


def dataset_entry(exp_dataset: ExperimentDataset) -> dict:
    """Build one YAML dataset entry: {name, images, labels, classes[, augmentation]}."""
    entry = {
        "name": exp_dataset.dataset.name,
        "images": str(images_dir(exp_dataset.dataset)),
        "labels": str(label_dir(exp_dataset)),
        "classes": dataset_classes(exp_dataset.dataset),
    }
    augmentation = augmentation_entry(exp_dataset)
    if augmentation:
        entry["augmentation"] = augmentation
    return entry


def augmentation_entry(exp_dataset: ExperimentDataset) -> dict:
    """The dataset's `augmentation` block: enabled checkboxes -> fractions.

    The trainer only augments train datasets (and rejects the key elsewhere),
    so flags on val/test rows are never emitted — model.clean() already blocks
    saving them, but rows predating that validation shouldn't break a run.
    """
    if exp_dataset.role != ExperimentDataset.TRAIN:
        return {}
    augmentation = {}
    if exp_dataset.aug_hflip and exp_dataset.aug_hflip_fraction:
        augmentation["hflip"] = exp_dataset.aug_hflip_fraction
    if exp_dataset.aug_scale_crop and exp_dataset.aug_scale_crop_fraction:
        augmentation["scale_crop"] = exp_dataset.aug_scale_crop_fraction
    return augmentation


def model_entry(exp_model: ExperimentModel) -> dict:
    """Build one YAML model entry; our name/num_classes win over params.

    The ``pretrained`` checkbox maps to ``weights: true`` — every adapter reads
    ``weights=True`` as "load the published COCO-pretrained weights" (retinanet,
    rtdetr, yolox, rfdetr). An explicit ``weights`` in ``params`` (e.g. a path or
    URL) is left untouched and wins over the checkbox.
    """
    params = dict(exp_model.params or {})
    entry = {
        **params,
        "name": exp_model.arch,
        "num_classes": exp_model.num_classes if exp_model.num_classes is not None else "auto",
    }
    if exp_model.pretrained and "weights" not in params:
        entry["weights"] = True
    return entry


def _scheduler(experiment: Experiment):
    if experiment.scheduler_name == "none":
        return None
    return {"name": experiment.scheduler_name, **(experiment.scheduler_params or {})}


def build_experiment_dict(experiment: Experiment, output_dir: Path | str) -> dict:
    """Assemble the full friendy_chachkalica experiment dict.

    Raises ``ValueError`` if the roster is invalid (no train dataset, more than
    one val/test, or no models) — surfaced by the admin action.
    """
    rows = list(experiment.datasets.all())
    train = [dataset_entry(r) for r in rows if r.role == ExperimentDataset.TRAIN]
    vals = [r for r in rows if r.role == ExperimentDataset.VAL]
    tests = [r for r in rows if r.role == ExperimentDataset.TEST]

    if not train:
        raise ValueError("Add at least one train dataset.")
    if len(vals) > 1:
        raise ValueError("At most one val dataset is allowed.")
    if len(tests) > 1:
        raise ValueError("At most one test dataset is allowed.")

    models = list(experiment.models.all())
    if not models:
        raise ValueError("Add at least one model architecture.")

    datasets: dict = {"train": train}
    if vals:
        datasets["val"] = dataset_entry(vals[0])
    if tests:
        datasets["test"] = dataset_entry(tests[0])

    return {
        "name": experiment.name,
        "output_dir": str(output_dir),
        "datasets": datasets,
        "models": [model_entry(m) for m in models],
        "training": {
            "epochs": experiment.epochs,
            "batch_size": experiment.batch_size,
            "num_workers": experiment.num_workers,
            "device": experiment.device,
            "seed": experiment.seed,
            "amp": experiment.amp,
            "gradient_clip_norm": experiment.gradient_clip_norm,
            "early_stopping_patience": experiment.early_stopping_patience,
            # "f1+map50" is one choice value meaning "average of both"; the
            # trainer takes a list and tracks the mean.
            "best_metric": experiment.best_metric.split("+"),
            "val_interval": experiment.val_interval,
            "optimizer": {
                "name": experiment.optimizer_name,
                "lr": experiment.lr,
                "weight_decay": experiment.weight_decay,
                "params": experiment.optimizer_params or {},
            },
            "scheduler": _scheduler(experiment),
        },
        "evaluation": {
            "batch_size": experiment.eval_batch_size,
            "num_workers": experiment.eval_num_workers,
            "map_score_threshold": 0.001,
            "score_threshold": experiment.eval_score_threshold,
            "iou_thresholds": experiment.iou_thresholds,
        },
    }


def build_yaml(experiment: Experiment, output_dir: Path | str) -> str:
    """Render the experiment dict to YAML text (no files written)."""
    data = build_experiment_dict(experiment, output_dir)
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def run_paths(experiment: Experiment, run_id: int, ts: TrainingSettings | None = None):
    """Return (config_yaml_path, output_dir) for a run."""
    ts = ts or TrainingSettings.load()
    stem = f"{experiment.name}-{run_id}"
    yaml_path = _resolve(ts.configs_root) / f"{stem}.yaml"
    output_dir = _resolve(ts.runs_root) / stem
    return yaml_path, output_dir


def classes_for_name(dataset_name: str) -> list[str]:
    """Class names for a dataset directory, by name (no DB row required)."""
    classes, _tools = lsapi.parse_classes_file(source_root() / dataset_name / "classes.txt")
    return classes


def eval_request_paths(eval_run, ts: TrainingSettings | None = None):
    ts = ts or TrainingSettings.load()
    stem = f"eval-{eval_run.pk}"
    return _resolve(ts.configs_root) / f"{stem}.yaml", _resolve(ts.runs_root) / stem


def build_eval_request(eval_run, output_dir: Path | str, ts: TrainingSettings | None = None) -> dict:
    """Assemble the eval request consumed by friendy_chachkalica's eval_checkpoint.py.

    ``classes`` is the *eval dataset's* class space (the target labels); the
    model's own train-class space is read from the checkpoint by the trainer.
    """
    ts = ts or TrainingSettings.load()
    tm = eval_run.trained_model
    ds = eval_run.dataset
    if not tm.checkpoint_path:
        raise ValueError(f"{tm.name}: no checkpoint path to evaluate.")
    return {
        "name": f"eval-{eval_run.pk}",
        "checkpoint_path": tm.checkpoint_path,
        "images": str(images_dir(ds)),
        "labels": str(resolve_label_dir(
            ds, eval_run.label_source, eval_run.annotator, eval_run.explicit_labels_path)),
        "classes": dataset_classes(ds),
        "output_dir": str(output_dir),
        "map_score_threshold": eval_run.map_score_threshold,
        "score_threshold": eval_run.score_threshold,
        "iou_thresholds": default_iou_thresholds(),
        "device": ts.default_device,
    }


def write_eval_request(eval_run, ts: TrainingSettings | None = None) -> tuple[Path, str]:
    """Generate the eval request YAML for ``eval_run`` and persist its paths."""
    ts = ts or TrainingSettings.load()
    request_path, output_dir = eval_request_paths(eval_run, ts)
    data = build_eval_request(eval_run, output_dir, ts)
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(text, encoding="utf-8")
    eval_run.request_yaml_path = str(request_path)
    eval_run.output_dir = str(output_dir)
    eval_run.save(update_fields=["request_yaml_path", "output_dir"])
    return request_path, text


def pipeline_request_paths(pe, ts: TrainingSettings | None = None):
    ts = ts or TrainingSettings.load()
    stem = f"pipeline-{pe.pk}"
    return _resolve(ts.configs_root) / f"{stem}.yaml", _resolve(ts.runs_root) / stem


def build_pipeline_request(pe, output_dir: Path | str, ts: TrainingSettings | None = None) -> dict:
    """Assemble the chachak request consumed by ``chachak/run.py``.

    ``classes`` is the *eval dataset's* class space (the target labels); the
    model's own train-class space is read from the checkpoint by chachak. Only
    non-default detector/tiling knobs are emitted so chachak's own defaults apply
    when the operator left a field blank. Raises ``ValueError`` when a
    detector-requiring pipeline has no detector checkpoint (mirrors
    ``chachak/config.py``'s own validation, but caught before we enqueue).
    """
    ts = ts or TrainingSettings.load()
    tm = pe.trained_model
    ds = pe.dataset
    if not tm.checkpoint_path:
        raise ValueError(f"{tm.name}: no checkpoint path to evaluate.")

    from eval_pipelines.models import PipelineEvalRun

    data = {
        "name": f"pipeline-{pe.pk}",
        "pipeline": pe.pipeline,
        "model_checkpoint": tm.checkpoint_path,
        "images": str(images_dir(ds)),
        "labels": str(resolve_label_dir(
            ds, pe.label_source, pe.annotator, pe.explicit_labels_path)),
        "classes": dataset_classes(ds),
        "output_dir": str(output_dir),
        "map_score_threshold": pe.map_score_threshold,
        "score_threshold": pe.score_threshold,
        "iou_thresholds": default_iou_thresholds(),
        "device": ts.default_device,
    }

    if pe.pipeline == PipelineEvalRun.CHAIN and pe.chain:
        data["chain"] = list(pe.chain)

    needs_detector = pe.pipeline in PipelineEvalRun.DETECTOR_PIPELINES or (
        pe.pipeline == PipelineEvalRun.CHAIN
        and any(c in PipelineEvalRun.DETECTOR_PIPELINES for c in (pe.chain or []))
    )
    if pe.detector_checkpoint:
        data["detector"] = {"checkpoint": pe.detector_checkpoint}
    elif needs_detector:
        raise ValueError(
            f"pipeline '{pe.pipeline}' requires a detector checkpoint.")

    tiling = {}
    if pe.tile_width_pct:
        tiling["tile_width_pct"] = pe.tile_width_pct
    if pe.tile_height_pct:
        tiling["tile_height_pct"] = pe.tile_height_pct
    if pe.overlap is not None:
        tiling["overlap"] = pe.overlap
    if tiling:
        data["tiling"] = tiling

    return data


def write_pipeline_request(pe, ts: TrainingSettings | None = None) -> tuple[Path, str]:
    """Generate the chachak request YAML for ``pe`` and persist its paths."""
    ts = ts or TrainingSettings.load()
    request_path, output_dir = pipeline_request_paths(pe, ts)
    data = build_pipeline_request(pe, output_dir, ts)
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(text, encoding="utf-8")
    pe.request_yaml_path = str(request_path)
    pe.output_dir = str(output_dir)
    pe.save(update_fields=["request_yaml_path", "output_dir"])
    return request_path, text


def build_preview_request(
    tm,
    pipeline: str,
    image_path: str,
    *,
    detector_checkpoint: str = "",
    tile_width_pct: float | None = None,
    tile_height_pct: float | None = None,
    overlap: float | None = None,
    chain: list[str] | None = None,
    score_threshold: float = 0.05,
    ts: TrainingSettings | None = None,
) -> dict:
    """Assemble the ``POST /predict_image`` payload for one preview image.

    Slim sibling of :func:`build_pipeline_request`: no labels/output_dir (preview
    reads GT locally and persists nothing) and no dataset ``classes`` (prediction
    class names come from the checkpoint). ``pipeline`` may be ``"raw"`` (run the
    model directly) or any chachak pipeline name.
    """
    ts = ts or TrainingSettings.load()
    if not tm.checkpoint_path:
        raise ValueError(f"{tm.name}: no checkpoint path to preview.")
    from eval_pipelines.models import PipelineEvalRun

    valid_pipelines = {"raw", *(value for value, _label in PipelineEvalRun.PIPELINE_CHOICES)}
    if pipeline not in valid_pipelines:
        raise ValueError(f"Unknown preview pipeline: {pipeline!r}.")

    chain = list(chain or [])
    if pipeline == PipelineEvalRun.CHAIN and not chain:
        raise ValueError("pipeline 'chain' requires at least one chain member.")

    needs_detector = pipeline in PipelineEvalRun.DETECTOR_PIPELINES or (
        pipeline == PipelineEvalRun.CHAIN
        and any(c in PipelineEvalRun.DETECTOR_PIPELINES for c in chain)
    )
    if needs_detector and not detector_checkpoint:
        raise ValueError(f"pipeline '{pipeline}' requires a detector checkpoint.")

    payload = {
        "model_checkpoint": tm.checkpoint_path,
        "image_path": image_path,
        "pipeline": pipeline,
        "score_threshold": score_threshold,
        "device": ts.default_device,
    }
    if detector_checkpoint:
        payload["detector_checkpoint"] = detector_checkpoint
    if tile_width_pct:
        payload["tile_width_pct"] = tile_width_pct
    if tile_height_pct:
        payload["tile_height_pct"] = tile_height_pct
    if overlap is not None:
        payload["overlap"] = overlap
    if chain:
        payload["chain"] = chain
    return payload


def write_config(experiment: Experiment, run) -> tuple[Path, str]:
    """Generate the YAML for ``run`` and persist its paths on the run.

    Returns (yaml_path, yaml_text).
    """
    yaml_path, output_dir = run_paths(experiment, run.pk)
    text = build_yaml(experiment, output_dir)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(text, encoding="utf-8")
    run.config_yaml_path = str(yaml_path)
    run.output_dir = str(output_dir)
    run.save(update_fields=["config_yaml_path", "output_dir"])
    return yaml_path, text
