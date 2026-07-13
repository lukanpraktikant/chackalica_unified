import argparse
import json
import random
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import yaml
from torch.utils.data import DataLoader

try:
    from .config import DatasetConfig, ExperimentConfig, ExperimentRun, build_experiment_runs, load_config
    from .data import build_eval_dataloader, build_train_dataloader
    from .device import resolve_device
    from .metrics import (
        HARD_IMAGE_METRIC,
        HARD_IMAGE_METRIC_DESCRIPTION,
        evaluate_detection,
        select_hard_images,
    )
    from .postprocess import apply_class_aware_nms
    from .registry import build_model
except ImportError:
    from config import DatasetConfig, ExperimentConfig, ExperimentRun, build_experiment_runs, load_config
    from data import build_eval_dataloader, build_train_dataloader
    from device import resolve_device
    from metrics import (
        HARD_IMAGE_METRIC,
        HARD_IMAGE_METRIC_DESCRIPTION,
        evaluate_detection,
        select_hard_images,
    )
    from postprocess import apply_class_aware_nms
    from registry import build_model


def resolve_operating_nms_threshold(
    config: ExperimentConfig,
    model_config: Any,
) -> Optional[float]:
    """NMS IoU for the operating-point val/test metrics of one run.

    A model entry's own ``nms_threshold`` param wins (for yolox that is also its
    internal predict-time NMS, so re-applying it is a no-op; for the DETRs it
    exists purely for this), falling back to the experiment-wide
    ``evaluation.operating_nms_threshold``. None disables it — mAP is never
    affected either way.
    """
    value = (getattr(model_config, "params", None) or {}).get("nms_threshold")
    if value is None:
        value = config.evaluation.operating_nms_threshold
    return None if value is None else float(value)


def _best_metric_name(config: ExperimentConfig) -> str:
    """Identifier of what best-checkpoint selection tracks, e.g. ``val_map50``
    or ``val_f1+map50`` (an average of both).

    Stamped into checkpoints so a resume can tell whether a stored best_score
    is comparable (older checkpoints tracked val loss, where lower was better,
    or a different metric selection).
    """
    return "val_" + "+".join(config.training.best_metric)


def _best_metric_score(
    config: ExperimentConfig,
    val_map_summary: Optional[Dict[str, Any]],
) -> Optional[float]:
    """The configured selection score for one epoch: the named val metric, or
    the mean when several are configured. All choices are higher-is-better."""
    if val_map_summary is None:
        return None
    values = [val_map_summary.get(metric) for metric in config.training.best_metric]
    if any(value is None for value in values):
        return None
    return float(sum(values)) / len(values)


def train_from_config(
    config_path: str | Path,
    evaluate_after_train: bool = True,
    resume: bool = False,
) -> List[Dict[str, Any]]:
    """Train every model declared in one Friendy Chachkalica YAML config."""
    print(f"[train] Starting from config: {config_path}")
    config = load_config(config_path)
    return train_experiment(
        config,
        evaluate_after_train=evaluate_after_train,
        resume=resume,
    )


def train_experiment(
    config: ExperimentConfig,
    evaluate_after_train: bool = True,
    resume: bool = False,
) -> List[Dict[str, Any]]:
    if config.training.seed is not None:
        print(f"[train] Setting random seed: {config.training.seed}")
        _set_seed(config.training.seed)

    device = resolve_device(config.training.device)
    print(f"[train] Using device: {device}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] Output directory: {config.output_dir}")
    _write_yaml(config.output_dir / "config.resolved.yaml", _to_builtin(config))
    print(f"[train] Wrote resolved config: {config.output_dir / 'config.resolved.yaml'}")

    train_loaders: Dict[tuple, DataLoader] = {}
    eval_loaders: Dict[tuple, DataLoader] = {}
    results = []
    runs = build_experiment_runs(config)
    print(f"[train] Training {len(runs)} run(s) (resume={resume})")
    for run in runs:
        result_path = config.output_dir / run.name / "result.yaml"
        if resume and result_path.exists():
            print(f"[train] Run {run.name} already complete, skipping (found {result_path})")
            results.append(_read_yaml(result_path))
            _write_yaml(config.output_dir / "results.yaml", _to_builtin(results))
            continue

        train_loader = _get_train_loader(config, run.train_dataset, train_loaders)
        val_loader = _get_eval_loader(config, run.val_dataset, eval_loaders)
        test_loader = _get_eval_loader(config, run.test_dataset, eval_loaders)

        try:
            result = train_model(
                config=config,
                run=run,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                evaluate_after_train=False,
                resume=resume,
            )
        except Exception as exc:
            print(f"[train] Run {run.name} FAILED: {exc}")
            result = {"run_index": run.index, "run_name": run.name, "error": str(exc)}
        results.append(result)
        _write_yaml(config.output_dir / "results.yaml", _to_builtin(results))

    if evaluate_after_train:
        _evaluate_all_runs_after_train(config)

    return results


def _evaluate_all_runs_after_train(config: ExperimentConfig) -> None:
    """Run the val/test evaluation phase across every run from its saved checkpoints.

    This is the consolidated "testing phase": it covers every run uniformly,
    including runs that were skipped by --resume because they were already
    trained. Evaluation reads each run's best/last checkpoint, so no retraining
    happens here.
    """
    if config.val_dataset is None and config.test_dataset is None:
        print("[train] No val/test dataset configured; skipping post-train evaluation")
        return

    # Imported lazily: val.py imports predict_dataset from this module, so a
    # top-level import here would be circular.
    try:
        from .val import val_experiment
    except ImportError:
        from val import val_experiment

    checkpoint = "best" if config.val_dataset is not None else "last"
    print(f"[train] Post-train evaluation phase start (checkpoint={checkpoint})")
    if config.val_dataset is not None:
        print("[train] Evaluating val split for all runs")
        val_experiment(config, split="val", checkpoint=checkpoint)
    if config.test_dataset is not None:
        print("[train] Evaluating test split for all runs")
        val_experiment(config, split="test", checkpoint=checkpoint)
    print("[train] Post-train evaluation phase done")


def _get_train_loader(
    config: ExperimentConfig,
    dataset_config: DatasetConfig,
    cache: Dict[tuple, DataLoader],
) -> DataLoader:
    cache_key = _dataset_cache_key(dataset_config)
    loader = cache.get(cache_key)
    if loader is None:
        print(f"[train] Creating train loader for dataset={dataset_config.name}")
        loader = build_train_dataloader(config, dataset_config)
        cache[cache_key] = loader
    else:
        print(f"[train] Reusing train loader for dataset={dataset_config.name}")
    return loader


def _get_eval_loader(
    config: ExperimentConfig,
    dataset_config: Optional[DatasetConfig],
    cache: Dict[tuple, DataLoader],
) -> Optional[DataLoader]:
    if dataset_config is None:
        return None

    cache_key = _dataset_cache_key(dataset_config)
    loader = cache.get(cache_key)
    if loader is None:
        print(f"[train] Creating eval loader for dataset={dataset_config.name} role={dataset_config.role}")
        loader = build_eval_dataloader(dataset_config, config)
        cache[cache_key] = loader
    else:
        print(f"[train] Reusing eval loader for dataset={dataset_config.name} role={dataset_config.role}")
    return loader


def train_model(
    config: ExperimentConfig,
    run: ExperimentRun,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    test_loader: Optional[DataLoader],
    device: torch.device,
    evaluate_after_train: bool = True,
    resume: bool = False,
) -> Dict[str, Any]:
    model_config = run.model
    train_dataset_config = run.train_dataset
    run_name = run.name
    run_dir = config.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[train] Run {run.index} start: name={run_name} model={model_config.name} "
        f"num_classes={model_config.num_classes} train_dataset={train_dataset_config.name}"
    )
    print(f"[train] Run directory: {run_dir}")

    print(f"[train] Building model adapter: {model_config.name}")
    adapter = build_model(
        model_config.name,
        num_classes=model_config.num_classes,
        **model_config.params,
    )
    adapter.to(device)
    print(f"[train] Model moved to device: {device}")

    optimizer = build_optimizer(adapter.model.parameters(), config)
    scheduler = build_scheduler(optimizer, config)
    scaler = _build_grad_scaler(config, device, adapter)
    print(
        f"[train] Optimizer={config.training.optimizer.name} lr={config.training.optimizer.lr} "
        f"scheduler={config.training.scheduler.name} amp={scaler is not None}"
    )

    history = []
    best_score = None
    best_epoch = None
    epochs_without_improvement = 0
    start_epoch = 1
    best_metric_name = _best_metric_name(config)
    val_interval = config.training.val_interval
    operating_nms_threshold = resolve_operating_nms_threshold(config, model_config)
    print(
        f"[train] Best-checkpoint selection metric: {best_metric_name} "
        f"(val_interval={val_interval} operating_nms_threshold={operating_nms_threshold})"
    )

    last_checkpoint = run_dir / "last.pt"
    if resume and last_checkpoint.exists():
        print(f"[train] Resuming run {run_name} from checkpoint: {last_checkpoint}")
        state = torch.load(last_checkpoint, map_location=device)
        adapter.model.load_state_dict(state["model_state_dict"])
        if state.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler is not None and state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])
        history = list(state.get("history") or [])
        best_score = state.get("best_score")
        best_epoch = _best_epoch_from_history(history)
        if best_score is not None and state.get("best_metric") != best_metric_name:
            # Scores on different metrics aren't comparable (older checkpoints
            # even tracked val loss, where lower was better), so a stale best
            # would never — or wrongly — be beaten.
            print(
                f"[train] Resume: checkpoint tracked best_score on a different "
                f"metric ({state.get('best_metric')!r}), now selecting on "
                f"{best_metric_name}; restarting best-model tracking"
            )
            best_score = None
            best_epoch = None
        # Restore the early-stopping counter, otherwise a resumed run gets up
        # to `patience` extra epochs before stopping.
        epochs_without_improvement = _epochs_since_best(history, best_epoch)
        start_epoch = int(state.get("epoch", 0)) + 1
        print(
            f"[train] Resumed run {run_name} at epoch {start_epoch}/{config.training.epochs} "
            f"(best_score={best_score} best_epoch={best_epoch} "
            f"epochs_without_improvement={epochs_without_improvement})"
        )

    for epoch in range(start_epoch, config.training.epochs + 1):
        print(f"[train] Run {run_name} epoch {epoch}/{config.training.epochs} start")
        train_summary = train_one_epoch(
            adapter=adapter,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            device=device,
        )

        val_summary = None
        val_map_summary = None
        # Validation runs every `val_interval` epochs; the final epoch always
        # validates so the run ends with a scored epoch (and thus a best.pt).
        run_val = val_loader is not None and (
            epoch % val_interval == 0 or epoch == config.training.epochs
        )
        if val_loader is not None and not run_val:
            print(
                f"[train] Run {run_name} epoch {epoch}: skipping validation "
                f"(val_interval={val_interval})"
            )
        if run_val:
            print(f"[train] Run {run_name} epoch {epoch}: evaluating validation loss")
            val_summary = evaluate_loss(
                adapter,
                val_loader,
                device,
                config=config,
                source_classes=run.val_dataset.classes if run.val_dataset is not None else None,
                model_classes=train_dataset_config.classes,
            )
            print(f"[train] Run {run_name} epoch {epoch}: evaluating validation mAP")
            val_map_summary = evaluate_map(
                adapter,
                val_loader,
                device,
                config=config,
                hard_images_path=run_dir / "val_predictions.pt",
                prediction_classes=train_dataset_config.classes,
                target_classes=(
                    run.val_dataset.classes
                    if run.val_dataset is not None
                    else train_dataset_config.classes
                ),
                operating_nms_threshold=operating_nms_threshold,
            )

        if scheduler is not None:
            scheduler.step()

        # The best checkpoint is selected on the configured val metric(s)
        # (training.best_metric, higher is better; several are averaged), not
        # val loss: the summed loss mixes objectness/cls/box terms and
        # routinely diverges from detection quality.
        score = _best_metric_score(config, val_map_summary)
        is_best = score is not None and (best_score is None or score > best_score)
        if is_best:
            best_score = score
            best_epoch = epoch
        if score is not None:
            epochs_without_improvement = 0 if is_best else epochs_without_improvement + 1

        epoch_summary = {
            "epoch": epoch,
            "train": train_summary,
            "val": val_summary,
            # Compact subset only: the full metrics dict (per_class etc.) would
            # bloat history.yaml and every checkpoint.
            "val_map": _compact_map_summary(val_map_summary),
            # What checkpoint selection tracked this epoch, so progress
            # displays can label the score without knowing the config.
            "best_metric": best_metric_name,
            "best_metric_score": score,
            "lr": _current_lr(optimizer),
            "is_best": is_best,
        }
        history.append(epoch_summary)

        checkpoint = {
            "epoch": epoch,
            "model_name": model_config.name,
            "model_config": _to_builtin(model_config),
            "train_dataset": _to_builtin(train_dataset_config),
            "model_state_dict": adapter.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_score": best_score,
            "best_metric": best_metric_name,
            "history": history,
        }
        save_checkpoint(checkpoint, run_dir / "last.pt")
        print(f"[train] Saved checkpoint: {run_dir / 'last.pt'}")
        if is_best:
            save_checkpoint(checkpoint, run_dir / "best.pt")
            print(f"[train] Saved new best checkpoint: {run_dir / 'best.pt'}")

        _write_yaml(run_dir / "history.yaml", _to_builtin(history))
        print(
            f"[train] Run {run_name} epoch {epoch} done: "
            f"train_loss={train_summary.get('loss')} "
            f"val_loss={val_summary.get('loss') if val_summary else None} "
            f"val_map50={val_map_summary.get('map50') if val_map_summary else None} "
            f"val_map50_95={val_map_summary.get('map50_95') if val_map_summary else None} "
            f"{best_metric_name}={score} "
            f"lr={_current_lr(optimizer)} best={is_best}"
        )

        # Early stopping: once the best metric hasn't improved for `patience`
        # *scored* epochs (val_interval > 1 skips epochs without scoring them),
        # stop — the best.pt checkpoint already holds the best epoch, so
        # continuing just overfits. Only active when a val set produced a score.
        patience = config.training.early_stopping_patience
        if patience is not None and score is not None and epochs_without_improvement >= patience:
            print(
                f"[train] Run {run_name} early stopping at epoch {epoch}: no val "
                f"improvement for {epochs_without_improvement} scored epoch(s) "
                f"(best epoch {best_epoch}, best {best_metric_name}={best_score})"
            )
            break

    if evaluate_after_train and test_loader is not None:
        print(f"[train] Run {run_name}: running post-train test prediction")
        best_checkpoint = run_dir / "best.pt"
        if val_loader is not None and best_checkpoint.exists():
            print(f"[train] Loading best checkpoint for test: {best_checkpoint}")
            state = torch.load(best_checkpoint, map_location=device)
            adapter.model.load_state_dict(state["model_state_dict"])
        prediction_path = run_dir / "test_predictions.pt"
        test_metrics = predict_dataset(
            adapter,
            test_loader,
            device,
            prediction_path,
            config,
            num_classes=model_config.num_classes,
            prediction_classes=train_dataset_config.classes,
            target_classes=run.test_dataset.classes if run.test_dataset is not None else None,
            eval_classes=run.test_dataset.classes if run.test_dataset is not None else None,
            operating_nms_threshold=operating_nms_threshold,
        )
    else:
        prediction_path = None
        test_metrics = None

    result = {
        "run_index": run.index,
        "model": model_config.name,
        "model_num_classes": model_config.num_classes,
        "train_dataset": train_dataset_config.name,
        "train_dataset_images": str(train_dataset_config.images),
        "train_dataset_labels": str(train_dataset_config.labels),
        "train_dataset_role": train_dataset_config.role,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_metric": best_metric_name,
        "best_score": best_score,
        "best_loss": _val_loss_at_epoch(history, best_epoch),
        "last_epoch": history[-1]["epoch"] if history else config.training.epochs,
        "last_train_loss": history[-1]["train"]["loss"] if history else None,
        "last_val_loss": history[-1]["val"]["loss"] if history and history[-1]["val"] else None,
        "best_checkpoint": str(run_dir / "best.pt") if best_epoch is not None else None,
        "last_checkpoint": str(run_dir / "last.pt"),
        "test_predictions": str(prediction_path) if prediction_path is not None else None,
        "test_metrics": test_metrics,
    }
    _write_yaml(run_dir / "result.yaml", _to_builtin(result))
    print(f"[train] Run {run_name} complete: result={run_dir / 'result.yaml'}")
    return result


def train_one_epoch(
    adapter: Any,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, Any]:
    adapter.train()
    total_loss = 0.0
    total_images = 0
    loss_totals: Dict[str, float] = {}

    for images, targets in loader:
        images, targets = _move_batch_to_device(images, targets, device)
        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(config, device, adapter):
            loss, loss_items = adapter.training_step(images, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            if config.training.gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    adapter.model.parameters(),
                    config.training.gradient_clip_norm,
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.training.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    adapter.model.parameters(),
                    config.training.gradient_clip_norm,
                )
            optimizer.step()

        batch_size = len(images)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_images += batch_size
        _accumulate_losses(loss_totals, loss_items, batch_size)

    return _summarize_losses(total_loss, total_images, loss_totals)


@torch.no_grad()
def evaluate_loss(
    adapter: Any,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    source_classes: Optional[Dict[int, str]] = None,
    model_classes: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    total_loss = 0.0
    total_images = 0
    loss_totals: Dict[str, float] = {}

    for images, targets in loader:
        images, targets = _move_batch_to_device(images, targets, device)
        targets = _remap_targets_to_model_classes(targets, source_classes, model_classes)
        loss_step = getattr(adapter, "validation_step", adapter.training_step)
        with _autocast_context(config, device, adapter):
            loss, loss_items = loss_step(images, targets)
        batch_size = len(images)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_images += batch_size
        _accumulate_losses(loss_totals, loss_items, batch_size)

    return _summarize_losses(total_loss, total_images, loss_totals)


@torch.no_grad()
def evaluate_map(
    adapter: Any,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    hard_images_path: Optional[str | Path] = None,
    prediction_classes: Optional[Dict[int, str]] = None,
    target_classes: Optional[Dict[int, str]] = None,
    operating_nms_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Predict over a loader and compute detection metrics.

    Used for per-epoch validation so best-checkpoint selection and early
    stopping can track val mAP instead of val loss. When ``hard_images_path`` is
    supplied, it also refreshes the live ``val_hard_images.json`` viewer artifact.
    Predictions are remapped by class name onto ``target_classes``, mirroring
    ``predict_dataset``.
    """
    was_training = adapter.model.training
    adapter.eval()
    all_predictions = []
    all_targets = []
    records = []
    try:
        for images, targets in loader:
            images, targets = _move_batch_to_device(images, targets, device)
            predictions = _predict_with_config(adapter, images, config)
            predictions = _apply_eval_nms(predictions, config)
            for target, prediction in zip(targets, predictions):
                prediction = prediction.detach().cpu()
                target_cpu = _target_to_cpu(target)
                all_predictions.append(prediction)
                all_targets.append(target_cpu)
                records.append(
                    {
                        "image_path": target.get("image_path"),
                        "label_path": target.get("label_path"),
                        "orig_size": _cpu_value(target.get("orig_size")),
                    }
                )
    finally:
        adapter.train(was_training)

    metrics = evaluate_detection(
        all_predictions,
        all_targets,
        iou_thresholds=config.evaluation.iou_thresholds,
        score_threshold=config.evaluation.score_threshold,
        map_score_threshold=config.evaluation.map_score_threshold,
        prediction_classes=prediction_classes,
        target_classes=target_classes,
        eval_classes=target_classes,
        operating_nms_threshold=operating_nms_threshold,
    )
    _print_eval_map_debug(
        metrics,
        all_predictions,
        all_targets,
        prediction_classes=prediction_classes,
        target_classes=target_classes,
    )
    if hard_images_path is not None:
        _write_hard_images(
            hard_images_path,
            all_predictions,
            all_targets,
            records,
            config=config,
            prediction_classes=prediction_classes,
            target_classes=target_classes,
            eval_classes=target_classes,
        )
    return metrics


@torch.no_grad()
def predict_dataset(
    adapter: Any,
    loader: DataLoader,
    device: torch.device,
    output_path: str | Path,
    config: Optional[ExperimentConfig] = None,
    num_classes: Optional[int] = None,
    prediction_classes: Optional[Dict[int, str]] = None,
    target_classes: Optional[Dict[int, str]] = None,
    eval_classes: Optional[Dict[int, str]] = None,
    operating_nms_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    adapter.eval()
    records = []
    all_predictions = []
    all_targets = []
    started_at = datetime.now(timezone.utc)
    start_perf = time.perf_counter()
    print(f"[train] Predicting dataset to: {output_path} (started {started_at.isoformat(timespec='seconds')})")
    for batch_index, (images, targets) in enumerate(loader, start=1):
        images, targets = _move_batch_to_device(images, targets, device)
        predictions = _predict_with_config(adapter, images, config)
        predictions = _apply_eval_nms(predictions, config)
        print(f"[train] Predicted batch {batch_index}: images={len(images)}")
        for target, prediction in zip(targets, predictions):
            prediction = prediction.detach().cpu()
            target_cpu = _target_to_cpu(target)
            all_predictions.append(prediction)
            all_targets.append(target_cpu)
            records.append(
                {
                    "image_path": target.get("image_path"),
                    "label_path": target.get("label_path"),
                    "orig_size": _cpu_value(target.get("orig_size")),
                    "predictions": prediction,
                }
            )
    torch.save(records, output_path)
    print(f"[train] Saved predictions: {output_path} records={len(records)}")

    metrics = evaluate_detection(
        all_predictions,
        all_targets,
        iou_thresholds=config.evaluation.iou_thresholds if config is not None else None,
        score_threshold=config.evaluation.score_threshold if config is not None else 0.001,
        map_score_threshold=config.evaluation.map_score_threshold if config is not None else None,
        num_classes=num_classes,
        prediction_classes=prediction_classes,
        target_classes=target_classes,
        eval_classes=eval_classes,
        operating_nms_threshold=operating_nms_threshold,
    )
    # Stamp the run with wall-clock timing so downstream tooling can show when the
    # eval ran and how long it took, alongside the quality metrics.
    metrics["evaluated_at"] = started_at.isoformat(timespec="seconds")
    metrics["eval_seconds"] = round(time.perf_counter() - start_perf, 3)
    print(
        f"[train] Metrics: map50={metrics.get('map50')} "
        f"map50_95={metrics.get('map50_95')} precision={metrics.get('precision')} "
        f"recall={metrics.get('recall')} eval_seconds={metrics.get('eval_seconds')}"
    )

    _write_hard_images(
        output_path,
        all_predictions,
        all_targets,
        records,
        config=config,
        prediction_classes=prediction_classes,
        target_classes=target_classes,
        eval_classes=eval_classes,
    )
    return metrics


def _write_hard_images(
    predictions_path: str | Path,
    all_predictions: List[torch.Tensor],
    all_targets: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    *,
    config: Optional[ExperimentConfig],
    prediction_classes: Optional[Dict[int, str]],
    target_classes: Optional[Dict[int, str]],
    eval_classes: Optional[Dict[int, str]],
    top_k: int = 50,
    iou_threshold: float = 0.5,
    score_threshold: Optional[float] = None,
    max_display_predictions: int = 20,
) -> None:
    """Persist the ``top_k`` hardest images alongside the predictions file.

    Writes ``<split>_hard_images.json`` next to ``<split>_predictions.pt`` (self-contained:
    image paths + normalized boxes + class names), which the admin viewer renders. Guarded so
    a split with no ground truth is skipped and any failure never sinks the eval that already
    produced its metrics. The saved artifact keeps low-confidence predictions so the browser
    confidence slider can decide what to display.
    """
    if not all_targets or not any(int(target['labels'].numel()) for target in all_targets):
        print("[train] Skipping hard-images artifact: no ground-truth labels in split")
        return

    if score_threshold is None:
        if config is not None and config.evaluation.map_score_threshold is not None:
            score_threshold = config.evaluation.map_score_threshold
        elif config is not None:
            score_threshold = config.evaluation.score_threshold
        else:
            score_threshold = 0.001

    predictions_path = Path(predictions_path)
    if predictions_path.name.endswith("_predictions.pt"):
        out_name = predictions_path.name[: -len("_predictions.pt")] + "_hard_images.json"
    else:
        out_name = predictions_path.stem + "_hard_images.json"
    output_path = predictions_path.with_name(out_name)

    try:
        images = select_hard_images(
            all_predictions,
            all_targets,
            records,
            top_k=top_k,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            prediction_classes=prediction_classes,
            target_classes=target_classes,
            eval_classes=eval_classes,
            max_display_predictions=max_display_predictions,
        )
        payload = {
            "metric": HARD_IMAGE_METRIC,
            "metric_description": HARD_IMAGE_METRIC_DESCRIPTION,
            "iou_threshold": float(iou_threshold),
            "score_threshold": float(score_threshold),
            "max_display_predictions": int(max_display_predictions),
            "top_k": int(top_k),
            "num_images_ranked": len(all_targets),
            "images": images,
        }
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"[train] Saved hard images: {output_path} count={len(images)}")
    except Exception as exc:  # noqa: BLE001 - artifact is best-effort; never break the eval
        print(f"[train] WARNING: failed to write hard images ({output_path}): {exc}")


def _print_eval_map_debug(
    metrics: Dict[str, Any],
    predictions: List[torch.Tensor],
    targets: List[Dict[str, Any]],
    *,
    prediction_classes: Optional[Dict[int, str]],
    target_classes: Optional[Dict[int, str]],
) -> None:
    raw_predictions = int(sum(prediction.shape[0] for prediction in predictions if prediction is not None))
    max_score = None
    score_parts = [prediction[:, 4].detach().cpu() for prediction in predictions if prediction is not None and prediction.numel()]
    if score_parts:
        max_score = round(float(torch.cat(score_parts).max().item()), 4)
    target_count = int(sum(int(target["labels"].numel()) for target in targets))
    pred_names = {str(name) for name in (prediction_classes or {}).values()}
    target_names = {str(name) for name in (target_classes or {}).values()}
    common_names = sorted(pred_names & target_names)
    print(
        "[train] Val mAP debug: "
        f"raw_predictions={raw_predictions} "
        f"operating_predictions={metrics.get('num_predictions')} "
        f"targets={target_count} max_score={max_score} "
        f"eval_classes_with_gt={metrics.get('num_eval_classes_with_gt')}/"
        f"{metrics.get('num_eval_classes')} "
        f"class_name_overlap={len(common_names)}"
    )
    if prediction_classes is not None and target_classes is not None and len(common_names) < len(target_names):
        missing = sorted(target_names - pred_names)
        if missing:
            print(f"[train] Val mAP debug: target classes not predicted by this model: {missing}")


def _apply_eval_nms(
    predictions: List[torch.Tensor],
    config: Optional[ExperimentConfig],
) -> List[torch.Tensor]:
    threshold = None if config is None else config.evaluation.nms_threshold
    return [apply_class_aware_nms(prediction, threshold) for prediction in predictions]


def _predict_with_config(
    adapter: Any,
    images: List[torch.Tensor],
    config: Optional[ExperimentConfig],
) -> List[torch.Tensor]:
    if config is None:
        return adapter.predict(images)

    try:
        threshold = config.evaluation.map_score_threshold
        if threshold is None:
            threshold = config.evaluation.score_threshold
        return adapter.predict(images, score_threshold=threshold)
    except TypeError:
        return adapter.predict(images)


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    config: ExperimentConfig,
) -> torch.optim.Optimizer:
    optimizer_config = config.training.optimizer
    name = optimizer_config.name.lower()
    params = list(parameters)
    kwargs = dict(optimizer_config.params)

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            **kwargs,
        )
    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            **kwargs,
        )
    if name == "sgd":
        kwargs.setdefault("momentum", 0.9)
        return torch.optim.SGD(
            params,
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            **kwargs,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_config.name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    scheduler_config = config.training.scheduler
    if scheduler_config.name is None:
        return None

    name = scheduler_config.name.lower()
    kwargs = dict(scheduler_config.params)
    if name in {"step", "step_lr", "steplr"}:
        kwargs.setdefault("step_size", 30)
        kwargs.setdefault("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    if name in {"multistep", "multi_step", "multi_step_lr", "multisteplr"}:
        kwargs.setdefault("milestones", [60, 80])
        kwargs.setdefault("gamma", 0.1)
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    if name in {"cosine", "cosine_annealing", "cosine_annealing_lr"}:
        kwargs.setdefault("T_max", config.training.epochs)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    if name in {"exponential", "exponential_lr"}:
        kwargs.setdefault("gamma", 0.95)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)

    raise ValueError(f"Unsupported scheduler: {scheduler_config.name}")


def save_checkpoint(checkpoint: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def _move_batch_to_device(
    images: List[torch.Tensor],
    targets: List[Dict[str, Any]],
    device: torch.device,
) -> tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    moved_images = [image.to(device, non_blocking=True) for image in images]
    moved_targets = []
    for target in targets:
        moved_targets.append(
            {
                key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
                for key, value in target.items()
            }
        )
    return moved_images, moved_targets


def _remap_targets_to_model_classes(
    targets: List[Dict[str, Any]],
    source_classes: Optional[Dict[int, str]],
    model_classes: Optional[Dict[int, str]],
) -> List[Dict[str, Any]]:
    if source_classes is None or model_classes is None:
        return targets

    model_name_to_id = {str(name): int(class_id) for class_id, name in model_classes.items()}
    source_to_model_id = {
        int(source_id): model_name_to_id[str(source_name)]
        for source_id, source_name in source_classes.items()
        if str(source_name) in model_name_to_id
    }

    return [
        _remap_target_to_model_classes(target, source_to_model_id)
        for target in targets
    ]


def _remap_target_to_model_classes(
    target: Dict[str, Any],
    source_to_model_id: Dict[int, int],
) -> Dict[str, Any]:
    labels = target["labels"].long()
    if labels.numel() == 0:
        return target

    remapped_labels = torch.full_like(labels, fill_value=-1)
    for source_id, model_id in source_to_model_id.items():
        remapped_labels[labels == source_id] = int(model_id)

    keep = remapped_labels >= 0
    remapped_target = dict(target)
    remapped_target["labels"] = remapped_labels[keep]
    remapped_target["boxes"] = target["boxes"][keep]

    if "area" in target and torch.is_tensor(target["area"]):
        remapped_target["area"] = target["area"][keep]
    if "iscrowd" in target and torch.is_tensor(target["iscrowd"]):
        remapped_target["iscrowd"] = target["iscrowd"][keep]

    return remapped_target


def _dataset_cache_key(dataset_config: DatasetConfig) -> tuple:
    return (
        dataset_config.name,
        str(dataset_config.images),
        str(dataset_config.labels),
        dataset_config.role,
        tuple(sorted((dataset_config.augmentation or {}).items())),
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_grad_scaler(
    config: ExperimentConfig,
    device: torch.device,
    adapter: Any,
) -> Optional[torch.amp.GradScaler]:
    if not _amp_enabled(config, device, adapter):
        return None
    return torch.amp.GradScaler("cuda")


def _autocast_context(config: ExperimentConfig, device: torch.device, adapter: Any):
    enabled = _amp_enabled(config, device, adapter)
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def _amp_enabled(config: ExperimentConfig, device: torch.device, adapter: Any) -> bool:
    return (
        config.training.amp
        and device.type == "cuda"
        and getattr(adapter, "supports_amp", True)
    )


def _accumulate_losses(
    loss_totals: Dict[str, float],
    loss_items: Dict[str, torch.Tensor],
    batch_size: int,
) -> None:
    for name, value in loss_items.items():
        if not torch.is_tensor(value):
            continue
        loss_totals[name] = loss_totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size


def _summarize_losses(
    total_loss: float,
    total_images: int,
    loss_totals: Dict[str, float],
) -> Dict[str, Any]:
    if total_images == 0:
        return {"loss": None, "num_images": 0, "loss_items": {}}
    return {
        "loss": total_loss / total_images,
        "num_images": total_images,
        "loss_items": {
            name: value / total_images
            for name, value in sorted(loss_totals.items())
        },
    }


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _write_yaml(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        yaml.safe_dump(value, file, sort_keys=False)


def _read_yaml(path: str | Path) -> Any:
    with open(path) as file:
        return yaml.safe_load(file)


def _best_epoch_from_history(history: List[Dict[str, Any]]) -> Optional[int]:
    best_epoch = None
    for entry in history:
        if entry.get("is_best"):
            best_epoch = entry.get("epoch", best_epoch)
    return best_epoch


def _epochs_since_best(history: List[Dict[str, Any]], best_epoch: Optional[int]) -> int:
    """Scored epochs after ``best_epoch`` — the resumed early-stopping counter."""
    if best_epoch is None:
        return 0
    return sum(
        1
        for entry in history
        if (entry.get("val_map") or entry.get("val")) is not None
        and entry.get("epoch", 0) > best_epoch
    )


def _val_loss_at_epoch(history: List[Dict[str, Any]], epoch: Optional[int]) -> Optional[float]:
    if epoch is None:
        return None
    for entry in history:
        if entry.get("epoch") == epoch and entry.get("val"):
            return entry["val"].get("loss")
    return None


def _compact_map_summary(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if metrics is None:
        return None
    keys = ("map50", "map50_95", "precision", "recall", "f1", "f1_confidence", "num_targets")
    return {key: metrics.get(key) for key in keys}


def _to_builtin(value: Any) -> Any:
    if is_dataclass(value):
        return _to_builtin(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def _target_to_cpu(target: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: _cpu_value(value)
        for key, value in target.items()
    }


def _cpu_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Friendy Chachkalica models from YAML config")
    parser.add_argument("config", help="Path to experiment YAML config")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs with a complete result.yaml and continue any run with a last.pt checkpoint from its next epoch",
    )
    args = parser.parse_args()
    results = train_from_config(args.config, resume=args.resume)
    print(yaml.safe_dump(_to_builtin(results), sort_keys=False))


if __name__ == "__main__":
    main()
