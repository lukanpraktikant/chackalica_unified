import argparse
import csv
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


METRIC_KEYS = [
    "map50_95",
    "map50",
    "precision",
    "recall",
    "f1",
    "num_images",
    "num_targets",
    "num_predictions",
]

UNIVERSAL_COLUMNS = [
    "run_index",
    "run_name",
    "train_dataset",
    "train_dataset_images",
    "train_dataset_labels",
    "train_dataset_role",
    "model",
    "model_num_classes",
    "best_epoch",
    "best_loss",
    "last_epoch",
    "last_train_loss",
    "last_val_loss",
    "best_checkpoint",
    "last_checkpoint",
    "val_dataset",
    "val_dataset_images",
    "val_dataset_labels",
    "val_dataset_role",
    "val_checkpoint",
    "val_predictions",
    "val_map50_95",
    "val_map50",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_num_images",
    "val_num_targets",
    "val_num_predictions",
    "test_dataset",
    "test_dataset_images",
    "test_dataset_labels",
    "test_dataset_role",
    "test_checkpoint",
    "test_predictions",
    "test_map50_95",
    "test_map50",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_num_images",
    "test_num_targets",
    "test_num_predictions",
]

DATASET_VALUE_COLUMNS = [
    "run_index",
    "run_name",
    "train_dataset_images",
    "train_dataset_labels",
    "train_dataset_role",
    "model_num_classes",
    "best_epoch",
    "best_loss",
    "last_epoch",
    "last_train_loss",
    "last_val_loss",
    "best_checkpoint",
    "last_checkpoint",
    "val_dataset",
    "val_dataset_images",
    "val_dataset_labels",
    "val_dataset_role",
    "val_checkpoint",
    "val_predictions",
    "val_map50_95",
    "val_map50",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_num_images",
    "val_num_targets",
    "val_num_predictions",
    "test_dataset",
    "test_dataset_images",
    "test_dataset_labels",
    "test_dataset_role",
    "test_checkpoint",
    "test_predictions",
    "test_map50_95",
    "test_map50",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_num_images",
    "test_num_targets",
    "test_num_predictions",
]

# Kept for exporting one YAML file directly. The full pipeline uses UNIVERSAL_COLUMNS.
DEFAULT_COLUMNS = [
    "run_index",
    "run_name",
    "train_dataset",
    "eval_dataset",
    "model",
    "model_num_classes",
    "map50_95",
    "map50",
    "precision",
    "recall",
    "f1",
    "num_images",
    "num_targets",
    "num_predictions",
    "best_epoch",
    "best_loss",
    "last_train_loss",
    "last_val_loss",
    "checkpoint",
    "predictions",
]


def export_universal_csv(
    train_results_path: str | Path,
    output_path: str | Path,
    val_results_path: str | Path | None = None,
    test_results_path: str | Path | None = None,
    columns: Optional[Iterable[str]] = None,
) -> Path:
    print(
        f"[export] Exporting universal CSV train_results={train_results_path} "
        f"val_results={val_results_path} test_results={test_results_path} output={output_path}"
    )
    rows_by_key: Dict[Any, Dict[str, Any]] = {}

    for result in _load_results(Path(train_results_path)):
        row = _flatten_train_result(result)
        rows_by_key[_result_key(result)] = row

    if val_results_path is not None and Path(val_results_path).exists():
        _merge_eval_results(rows_by_key, Path(val_results_path), prefix="val")

    if test_results_path is not None and Path(test_results_path).exists():
        _merge_eval_results(rows_by_key, Path(test_results_path), prefix="test")

    rows = [rows_by_key[key] for key in sorted(rows_by_key, key=_sort_key)]
    if columns is not None:
        return _write_csv(rows, output_path, list(columns))

    pivoted_rows, pivoted_columns = _pivot_datasets_to_columns(rows)
    return _write_csv(pivoted_rows, output_path, pivoted_columns)


def export_csv(
    results_path: str | Path,
    output_path: str | Path,
    columns: Optional[Iterable[str]] = None,
) -> Path:
    print(f"[export] Exporting CSV results={results_path} output={output_path}")
    results_path = Path(results_path)
    rows = [_flatten_result(result) for result in _load_results(results_path)]
    return _write_csv(rows, output_path, list(columns or DEFAULT_COLUMNS))


def _merge_eval_results(
    rows_by_key: Dict[Any, Dict[str, Any]],
    results_path: Path,
    prefix: str,
) -> None:
    for result in _load_results(results_path):
        key = _result_key(result)
        row = rows_by_key.setdefault(key, _base_row_from_eval(result))
        metrics = result.get("metrics") or result.get("test_metrics") or {}
        row[f"{prefix}_dataset"] = result.get("eval_dataset")
        row[f"{prefix}_dataset_images"] = result.get("eval_dataset_images")
        row[f"{prefix}_dataset_labels"] = result.get("eval_dataset_labels")
        row[f"{prefix}_dataset_role"] = result.get("eval_dataset_role")
        row[f"{prefix}_checkpoint"] = result.get("checkpoint")
        row[f"{prefix}_predictions"] = result.get("predictions") or result.get("test_predictions")
        for metric_key in METRIC_KEYS:
            row[f"{prefix}_{metric_key}"] = metrics.get(metric_key)


def _flatten_train_result(result: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "run_index": result.get("run_index"),
        "run_name": result.get("run_name"),
        "train_dataset": result.get("train_dataset"),
        "train_dataset_images": result.get("train_dataset_images"),
        "train_dataset_labels": result.get("train_dataset_labels"),
        "train_dataset_role": result.get("train_dataset_role"),
        "model": result.get("model"),
        "model_num_classes": result.get("model_num_classes"),
        "best_epoch": result.get("best_epoch"),
        "best_loss": result.get("best_loss"),
        "last_epoch": result.get("last_epoch"),
        "last_train_loss": result.get("last_train_loss"),
        "last_val_loss": result.get("last_val_loss"),
        "best_checkpoint": result.get("best_checkpoint"),
        "last_checkpoint": result.get("last_checkpoint"),
    }

    metrics = result.get("test_metrics")
    if metrics:
        row["test_predictions"] = result.get("test_predictions")
        for metric_key in METRIC_KEYS:
            row[f"test_{metric_key}"] = metrics.get(metric_key)

    return row


def _base_row_from_eval(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_index": result.get("run_index"),
        "run_name": result.get("run_name"),
        "train_dataset": result.get("train_dataset"),
        "model": result.get("model"),
        "model_num_classes": result.get("model_num_classes"),
    }


def _flatten_result(result: Dict[str, Any]) -> Dict[str, Any]:
    metrics = result.get("metrics") or result.get("test_metrics") or {}
    row = {
        "run_index": result.get("run_index"),
        "run_name": result.get("run_name"),
        "train_dataset": result.get("train_dataset"),
        "train_dataset_images": result.get("train_dataset_images"),
        "train_dataset_labels": result.get("train_dataset_labels"),
        "train_dataset_role": result.get("train_dataset_role"),
        "eval_dataset": result.get("eval_dataset") or _infer_eval_dataset(result),
        "eval_dataset_images": result.get("eval_dataset_images"),
        "eval_dataset_labels": result.get("eval_dataset_labels"),
        "eval_dataset_role": result.get("eval_dataset_role"),
        "model": result.get("model"),
        "model_num_classes": result.get("model_num_classes"),
        "best_epoch": result.get("best_epoch"),
        "best_loss": result.get("best_loss"),
        "last_epoch": result.get("last_epoch"),
        "last_train_loss": result.get("last_train_loss"),
        "last_val_loss": result.get("last_val_loss"),
        "checkpoint": result.get("checkpoint") or result.get("best_checkpoint") or result.get("last_checkpoint"),
        "predictions": result.get("predictions") or result.get("test_predictions"),
    }

    for key in METRIC_KEYS:
        row[key] = metrics.get(key)

    return row


def _load_results(path: Path) -> List[Dict[str, Any]]:
    with open(path) as file:
        value = yaml.safe_load(file) or []

    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        raise ValueError(f"Expected a YAML list or mapping in {path}")
    return value


def _write_csv(rows: List[Dict[str, Any]], output_path: str | Path, columns: List[str]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[export] Wrote CSV: {output_path} rows={len(rows)}")
    return output_path


def _pivot_datasets_to_columns(rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[str]]:
    grouped_rows: Dict[Any, Dict[str, Any]] = {}
    first_run_index_by_group: Dict[Any, Any] = {}
    dataset_columns: List[str] = []
    model_ordinals_by_dataset: Dict[Any, Dict[str, int]] = {}
    prefix_registry: Dict[str, Any] = {}

    for row in rows:
        dataset_key = row.get("train_dataset")
        model_name = str(row.get("model") or "model")
        ordinal = _model_ordinal_for_dataset(model_ordinals_by_dataset, dataset_key, model_name)
        group_key = (model_name, ordinal)

        output_row = grouped_rows.setdefault(group_key, _dataset_pivot_base_row(model_name, ordinal))
        first_run_index_by_group.setdefault(group_key, row.get("run_index"))

        prefix = _unique_prefix(_dataset_column_prefix(dataset_key), dataset_key, prefix_registry)
        for column in DATASET_VALUE_COLUMNS:
            pivot_column = f"{prefix}_{column}"
            output_row[pivot_column] = row.get(column)
            if pivot_column not in dataset_columns:
                dataset_columns.append(pivot_column)

    pivoted_rows = [
        grouped_rows[key]
        for key in sorted(grouped_rows, key=lambda key: _sort_key(first_run_index_by_group[key]))
    ]
    return pivoted_rows, ["model"] + dataset_columns


def _dataset_column_prefix(dataset: Any) -> str:
    prefix = re.sub(r"[^0-9a-zA-Z]+", "_", str(dataset or "dataset")).strip("_").lower()
    return prefix or "dataset"


def _unique_prefix(base_prefix: str, dataset_key: Any, registry: Dict[str, Any]) -> str:
    if base_prefix not in registry or registry[base_prefix] == dataset_key:
        registry[base_prefix] = dataset_key
        return base_prefix
    counter = 2
    while True:
        candidate = f"{base_prefix}_{counter}"
        if candidate not in registry or registry[candidate] == dataset_key:
            registry[candidate] = dataset_key
            return candidate
        counter += 1


def _dataset_pivot_base_row(model_name: str, ordinal: int) -> Dict[str, Any]:
    model = model_name if ordinal == 1 else f"{model_name}_{ordinal}"
    return {"model": model}


def _model_ordinal_for_dataset(
    ordinals_by_dataset: Dict[Any, Dict[str, int]],
    dataset_key: Any,
    model_name: str,
) -> int:
    model_counts = ordinals_by_dataset.setdefault(dataset_key, {})
    ordinal = model_counts.get(model_name, 0) + 1
    model_counts[model_name] = ordinal
    return ordinal


def _result_key(result: Dict[str, Any]) -> Any:
    if result.get("run_index") is not None:
        return int(result["run_index"])
    return result.get("run_name")


def _sort_key(key: Any) -> tuple[int, Any]:
    if isinstance(key, int):
        return (0, key)
    return (1, str(key))


def _infer_eval_dataset(result: Dict[str, Any]) -> Any:
    if result.get("test_predictions"):
        return "test"
    if result.get("predictions"):
        return "eval"
    return None


def default_output_path(results_path: str | Path) -> Path:
    results_path = Path(results_path)
    return results_path.with_suffix(".csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Friendy Chachkalica metrics to CSV")
    parser.add_argument(
        "results",
        help="Path to results YAML, for example runs/helmet-benchmark/results.yaml",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="CSV output path. Defaults to the results path with .csv suffix.",
    )
    parser.add_argument("--val-results", help="Optional val_results.yaml to merge into a universal CSV")
    parser.add_argument("--test-results", help="Optional test_results.yaml to merge into a universal CSV")
    args = parser.parse_args()

    results_path = Path(args.results)
    output_path = Path(args.output) if args.output else default_output_path(results_path)
    if args.val_results or args.test_results:
        written_path = export_universal_csv(
            results_path,
            output_path,
            val_results_path=args.val_results,
            test_results_path=args.test_results,
        )
    else:
        written_path = export_csv(results_path, output_path)
    print(written_path)


if __name__ == "__main__":
    main()
