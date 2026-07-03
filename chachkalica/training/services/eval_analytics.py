"""Compare the metrics of one or more finished :class:`~training.models.EvalRun`.

Read-only: it takes EvalRuns that already carry an ingested ``metrics`` dict
(see ``ingest.ingest_eval``) and reshapes them into a side-by-side comparison
the admin renders synchronously — one column per eval, one row per metric, with
the winning cell flagged per row. Mirrors fleet's "analyze dataset" action, but
across evaluated models rather than within one dataset.
"""

# (metric key, human label, higher_is_better). The count rows are context, not
# quality scores, so they get no winner highlight; eval time is lower-is-better.
_METRICS = [
    ("map50", "mAP@50", True),
    ("map50_95", "mAP@50-95", True),
    ("precision", "Precision", True),
    ("recall", "Recall", True),
    ("f1", "F1", True),
    ("num_predictions", "# predictions", None),
    ("num_targets", "# targets", None),
    ("num_images", "# images", None),
    ("eval_seconds", "Eval time (s)", False),
]


def _num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _column(eval_run) -> dict:
    metrics = eval_run.metrics if isinstance(eval_run.metrics, dict) else {}
    return {
        "id": eval_run.pk,
        "model": eval_run.trained_model.name,
        "arch": eval_run.trained_model.arch,
        "dataset": eval_run.dataset.name,
        "metrics": metrics,
        # When the eval ran (stamped by the trainer). Fall back to the row's
        # finished timestamp for evals predating the metric.
        "evaluated_at": metrics.get("evaluated_at") or (
            eval_run.finished_at.isoformat(timespec="seconds") if eval_run.finished_at else None
        ),
    }


def _winner(values, higher_is_better) -> float | None:
    """The best numeric value in ``values``, or None if no contest applies."""
    nums = [v for v in values if v is not None]
    if higher_is_better is None or len(nums) < 2 or len(set(nums)) < 2:
        return None  # counts, single column, or an all-tie row → nothing to flag
    return max(nums) if higher_is_better else min(nums)


def _overall_rows(columns) -> list[dict]:
    rows = []
    for key, label, higher_is_better in _METRICS:
        values = [_num(col["metrics"].get(key)) for col in columns]
        best = _winner(values, higher_is_better)
        rows.append({
            "label": label,
            "cells": [
                {"value": v, "is_best": best is not None and v == best}
                for v in values
            ],
        })
    return rows


def _class_rows(columns) -> list[dict]:
    """Per-class AP comparison, unioning the classes each eval reported."""
    names: dict = {}  # class_id -> class_name (first non-empty wins)
    for col in columns:
        for class_id, stats in (col["metrics"].get("per_class") or {}).items():
            names.setdefault(class_id, (stats or {}).get("class_name") or str(class_id))

    rows = []
    for class_id, class_name in names.items():
        ap50 = [_num((col["metrics"].get("per_class") or {}).get(class_id, {}).get("ap50")) for col in columns]
        ap = [_num((col["metrics"].get("per_class") or {}).get(class_id, {}).get("ap50_95")) for col in columns]
        best50, best = _winner(ap50, True), _winner(ap, True)
        rows.append({
            "class_name": class_name,
            "ap50": [{"value": v, "is_best": best50 is not None and v == best50} for v in ap50],
            "ap50_95": [{"value": v, "is_best": best is not None and v == best} for v in ap],
        })
    return rows


def compare(eval_runs) -> dict:
    """Build the comparison context for a list of EvalRuns (metrics required).

    Columns are ordered best-mAP@50 first so the strongest model reads left-to-right.
    """
    columns = sorted(
        (_column(e) for e in eval_runs),
        key=lambda c: (_num(c["metrics"].get("map50")) is not None, _num(c["metrics"].get("map50")) or 0),
        reverse=True,
    )
    return {
        "columns": columns,
        "overall_rows": _overall_rows(columns),
        "class_rows": _class_rows(columns),
    }
