"""CLI entrypoint for chachak pipelines.

    python chachak/run.py request.yaml

Loads a pipeline config, loads the trained model (and a detector when the
pipeline needs one), builds the eval dataloader with Friendy's loader, runs the
pipeline, and writes ``predictions.pt`` + ``result.yaml`` in the same shape as
``friendy_chachkalica/eval_checkpoint.py`` so results drop into the eval flow.
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

try:
    from ._friendy import (
        DatasetConfig,
        EvaluationConfig,
        ExperimentConfig,
        ModelConfig,
        TrainingConfig,
        _to_builtin,
        _write_yaml,
        build_eval_dataloader,
        resolve_device,
    )
    from .config import load_pipeline_config
    from .detector import load_detector
    from .infer import load_checkpoint_adapter
    from .registry import build_pipeline
except ImportError:  # run as a flat script
    from _friendy import (
        DatasetConfig,
        EvaluationConfig,
        ExperimentConfig,
        ModelConfig,
        TrainingConfig,
        _to_builtin,
        _write_yaml,
        build_eval_dataloader,
        resolve_device,
    )
    from config import load_pipeline_config
    from detector import load_detector
    from infer import load_checkpoint_adapter
    from registry import build_pipeline


def _needs_detector(config) -> bool:
    if config.pipeline in {"people_detect_first", "batch_people"}:
        return True
    if config.pipeline == "chain":
        return any(c in {"people_detect_first", "batch_people"} for c in config.chain)
    return False


def run_pipeline(config) -> Dict[str, Any]:
    device = resolve_device(config.device)

    model_adapter, info = load_checkpoint_adapter(config.model_checkpoint, device)
    num_classes = info["num_classes"]

    detector = None
    if _needs_detector(config):
        detector = load_detector(
            config.detector.checkpoint,
            device,
            person_class_name=config.detector.person_class_name,
            person_class_id=config.detector.person_class_id,
            score_threshold=config.detector.score_threshold,
            batch_size=config.infer_batch_size,
        )

    pipeline = build_pipeline(config, model_adapter, device, detector)

    # Build the eval dataloader through Friendy so datasets/labels load identically
    # to eval_checkpoint.py. Frames-per-batch is infer_batch_size; the model only
    # ever sees infer_batch_size tiles/crops at a time inside the pipeline.
    dataset_config = DatasetConfig(
        name=f"{config.name}-data",
        images=config.images,
        labels=config.labels,
        classes=config.classes,
        role="test",
    )
    experiment = ExperimentConfig(
        name=config.name,
        train_datasets=[dataset_config],
        models=[ModelConfig(name=info["model_name"], num_classes=num_classes)],
        output_dir=config.output_dir,
        test_dataset=dataset_config,
        training=TrainingConfig(
            batch_size=config.infer_batch_size,
            num_workers=config.num_workers,
            device=config.device,
        ),
        evaluation=EvaluationConfig(
            batch_size=config.infer_batch_size,
            num_workers=config.num_workers,
            score_threshold=config.score_threshold,
            iou_thresholds=config.iou_thresholds,
        ),
    )
    loader = build_eval_dataloader(dataset_config, experiment)

    result = pipeline.run(
        loader,
        config.output_dir,
        num_classes=num_classes,
        prediction_classes=info["train_classes"] or config.classes,
        target_classes=config.classes,
        eval_classes=config.classes,
    )

    output = {
        "pipeline": config.pipeline,
        "name": config.name,
        "model_checkpoint": str(config.model_checkpoint),
        "detector_checkpoint": (
            str(config.detector.checkpoint) if detector is not None else None
        ),
        "images": str(config.images),
        "labels": str(config.labels),
        "predictions": str(result["prediction_path"]),
        "metrics": result["metrics"],
    }
    result_path = Path(config.output_dir) / "result.yaml"
    _write_yaml(result_path, _to_builtin(output))
    print(f"[chachak] Wrote result: {result_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a chachak inference/eval pipeline")
    parser.add_argument("request", help="Path to a pipeline request YAML")
    args = parser.parse_args()
    config = load_pipeline_config(args.request)
    output = run_pipeline(config)
    print(yaml.safe_dump(_to_builtin(output), sort_keys=False))


if __name__ == "__main__":
    main()
