"""Standalone evaluation of a single trained checkpoint on an arbitrary dataset.

``val.py`` evaluates checkpoints *within* an experiment's run-directory layout.
This module instead evaluates one catalogued checkpoint (a registered model)
against a dataset chosen after the fact — the "test a trained model" path. The
checkpoint already carries everything needed to rebuild the model
(``model_name``, ``model_config``, and the training ``classes``), so the caller
only supplies the eval dataset (images, labels, classes) and where to write.

Driven by a small request YAML so the trainer service can launch it as a
subprocess, mirroring ``run.py``:

    .venv/bin/python eval_checkpoint.py request.yaml

Request YAML fields: checkpoint_path, images, labels, classes (list/mapping),
output_dir, and optional name, score_threshold, map_score_threshold, nms_threshold,
iou_thresholds, batch_size, num_workers, device.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import yaml

try:
    from .config import (
        DatasetConfig,
        EvaluationConfig,
        ExperimentConfig,
        ModelConfig,
        TrainingConfig,
    )
    from .data import build_eval_dataloader
    from .device import resolve_device
    from .registry import build_model
    from .train import predict_dataset, resolve_operating_nms_threshold
    from .val import _to_builtin, _write_yaml
except ImportError:
    from config import (
        DatasetConfig,
        EvaluationConfig,
        ExperimentConfig,
        ModelConfig,
        TrainingConfig,
    )
    from data import build_eval_dataloader
    from device import resolve_device
    from registry import build_model
    from train import predict_dataset, resolve_operating_nms_threshold
    from val import _to_builtin, _write_yaml


def _as_class_map(classes: Union[Dict, List]) -> Dict[int, str]:
    if isinstance(classes, dict):
        return {int(k): str(v) for k, v in classes.items()}
    return {index: str(name) for index, name in enumerate(classes)}


def eval_checkpoint(
    checkpoint_path: Union[str, Path],
    images: Union[str, Path],
    labels: Union[str, Path],
    classes: Union[Dict, List],
    output_dir: Union[str, Path],
    *,
    name: str = "eval",
    score_threshold: float = 0.001,
    map_score_threshold: Optional[float] = None,
    nms_threshold: Optional[float] = None,
    operating_nms_threshold: Optional[float] = None,
    iou_thresholds: Optional[List[float]] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    device: str = "auto",
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    print(f"[eval] Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    model_name = state["model_name"]
    model_config = state.get("model_config", {}) or {}
    num_classes = model_config.get("num_classes")
    params = dict(model_config.get("params", {}) or {})
    train_classes_raw = (state.get("train_dataset") or {}).get("classes")
    if not train_classes_raw:
        # Falling back to the eval class list would silently mislabel every
        # prediction whenever the model's training class order differs from it.
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not record its training class names "
            "(train_dataset.classes), so predictions cannot be remapped by name onto "
            "the eval classes. Re-train (or re-save the checkpoint) with class names."
        )
    train_classes = _as_class_map(train_classes_raw)
    eval_classes = _as_class_map(classes)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev = resolve_device(device)
    print(f"[eval] Building model adapter: {model_name} num_classes={num_classes} device={dev}")
    adapter = build_model(model_name, num_classes=num_classes, **params)
    adapter.to(dev)
    adapter.model.load_state_dict(state["model_state_dict"])

    dataset_config = DatasetConfig(
        name=f"{name}-data", images=Path(images), labels=Path(labels),
        classes=eval_classes, role="test",
    )
    evaluation = EvaluationConfig(
        batch_size=batch_size, num_workers=num_workers,
        score_threshold=score_threshold,
        map_score_threshold=map_score_threshold,
        nms_threshold=nms_threshold,
        operating_nms_threshold=operating_nms_threshold,
        iou_thresholds=iou_thresholds or EvaluationConfig().iou_thresholds,
    )
    training = TrainingConfig(batch_size=batch_size, num_workers=num_workers, device=device)
    config = ExperimentConfig(
        name=name, train_datasets=[dataset_config],
        models=[ModelConfig(name=model_name, num_classes=num_classes, params=params)],
        output_dir=output_dir, test_dataset=dataset_config,
        training=training, evaluation=evaluation,
    )

    loader = build_eval_dataloader(dataset_config, config)
    prediction_path = output_dir / "eval_predictions.pt"
    metrics = predict_dataset(
        adapter, loader, dev, prediction_path, config,
        num_classes=num_classes,
        prediction_classes=train_classes,
        target_classes=eval_classes,
        eval_classes=eval_classes,
        operating_nms_threshold=resolve_operating_nms_threshold(config, config.models[0]),
    )

    result = {
        "checkpoint": str(checkpoint_path),
        "model": model_name,
        "num_classes": num_classes,
        "eval_dataset": dataset_config.name,
        "images": str(images),
        "labels": str(labels),
        "metrics": metrics,
    }
    result_path = output_dir / "eval_result.yaml"
    _write_yaml(result_path, _to_builtin(result))
    print(f"[eval] Wrote eval result: {result_path}")
    return result


def eval_from_request(request_path: Union[str, Path]) -> Dict[str, Any]:
    request = yaml.safe_load(Path(request_path).read_text())
    return eval_checkpoint(
        checkpoint_path=request["checkpoint_path"],
        images=request["images"],
        labels=request["labels"],
        classes=request["classes"],
        output_dir=request["output_dir"],
        name=request.get("name", "eval"),
        score_threshold=request.get("score_threshold", 0.001),
        map_score_threshold=request.get("map_score_threshold"),
        nms_threshold=request.get("nms_threshold"),
        operating_nms_threshold=request.get("operating_nms_threshold"),
        iou_thresholds=request.get("iou_thresholds"),
        batch_size=request.get("batch_size", 4),
        num_workers=request.get("num_workers", 4),
        device=request.get("device", "auto"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a single checkpoint on a dataset")
    parser.add_argument("request", help="Path to an eval request YAML")
    args = parser.parse_args()
    result = eval_from_request(args.request)
    print(yaml.safe_dump(_to_builtin(result), sort_keys=False))


if __name__ == "__main__":
    main()
