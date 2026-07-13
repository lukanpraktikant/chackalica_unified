"""Apply operator-approved repairs for dataset analytics quality issues."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fleet.models import Dataset
from fleet.reconcile import writer
from fleet.reconcile.txt_format import _fmt, _is_axis_aligned_rectangle
from fleet.services import datasets as datasets_svc
from fleet.services import lsapi
from fleet.services.paths import source_root

ISSUE_ORPHAN_LABEL_FILES = "orphan_label_files"
ISSUE_INVALID_CLASS_REGIONS = "invalid_class_regions"
ISSUE_OUT_OF_BOUNDS_BOXES = "out_of_bounds_boxes"
ISSUE_ZERO_AREA_BOXES = "zero_area_boxes"

ACTION_DELETE = "delete"
ACTION_REMOVE = "remove"
ACTION_CLIP = "clip"

_BACKUP_DIR = ".quality_fix_backups"
_BOUNDS_TOL = 1e-3


@dataclass
class _ParsedRow:
    class_id: int
    bbox: tuple[float, float, float, float]
    polygon: list[float] | None
    is_polygon_source: bool
    app_format: bool


def solve_dataset_quality(dataset: Dataset, issue: str, action: str | None = None) -> dict:
    """Repair one quality issue for a dataset's source labels.

    Returns mutation counts for admin UI feedback. Every changed/deleted file is
    copied to ``labels/.quality_fix_backups/<timestamp>/`` before mutation.
    """
    labels_dir = datasets_svc.labels_source_dir(dataset)
    result = {
        "dataset": dataset.name,
        "issue": issue,
        "action": action or "",
        "changed_files": 0,
        "deleted_files": 0,
        "backed_up_files": 0,
        "removed_regions": 0,
        "clipped_regions": 0,
        "skipped_files": 0,
        "backup_dir": "",
    }
    if not labels_dir.is_dir():
        return result

    backup_dir = labels_dir / _BACKUP_DIR / datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    if issue == ISSUE_ORPHAN_LABEL_FILES:
        if action not in (None, "", ACTION_DELETE):
            raise ValueError("orphan label files only support delete")
        _delete_orphans(dataset, labels_dir, backup_dir, result)
    else:
        class_count = _class_count(dataset)
        if issue == ISSUE_INVALID_CLASS_REGIONS:
            _rewrite_image_labels(dataset, labels_dir, backup_dir, result, class_count, _fix_invalid_class)
        elif issue == ISSUE_OUT_OF_BOUNDS_BOXES:
            if action in (None, ""):
                action = ACTION_CLIP
            if action == ACTION_CLIP:
                _rewrite_image_labels(dataset, labels_dir, backup_dir, result, class_count, _clip_out_of_bounds)
            elif action == ACTION_REMOVE:
                _rewrite_image_labels(dataset, labels_dir, backup_dir, result, class_count, _remove_out_of_bounds)
            else:
                raise ValueError("out-of-bounds boxes support clip or remove")
        elif issue == ISSUE_ZERO_AREA_BOXES:
            if action not in (None, "", ACTION_REMOVE):
                raise ValueError("zero-area boxes only support remove")
            _rewrite_image_labels(dataset, labels_dir, backup_dir, result, class_count, _fix_zero_area)
        else:
            raise ValueError(f"unsupported data-quality issue: {issue}")

    if result["backed_up_files"]:
        result["backup_dir"] = str(backup_dir)
    return result


def _class_count(dataset: Dataset) -> int:
    classes_file = source_root() / dataset.name / "classes.txt"
    lsapi.require_path(classes_file, kind="Classes file")
    names, _tools = lsapi.parse_classes_file(classes_file)
    return len(names)


def _dataset_images(dataset: Dataset) -> list[Path]:
    image_dir = lsapi.image_source_dir(source_root() / dataset.name)
    return [
        p for p in sorted(image_dir.iterdir())
        if p.suffix.lower() in lsapi.IMAGE_EXTENSIONS
    ]


def _find_label_file(labels_dir: Path, image_filename: str) -> Path | None:
    for candidate in (
        labels_dir / f"{image_filename}.txt",
        labels_dir / f"{Path(image_filename).stem}.txt",
    ):
        if candidate.exists():
            return candidate
    return None


def _delete_orphans(dataset: Dataset, labels_dir: Path, backup_dir: Path, result: dict) -> None:
    images = _dataset_images(dataset)
    image_names = {p.name for p in images}
    image_stems = {p.stem for p in images}
    for path in sorted(labels_dir.iterdir()):
        if path.suffix.lower() != ".txt" or not path.is_file():
            continue
        base = path.name[: -len(".txt")]
        if base in image_names or base in image_stems:
            continue
        _backup(path, labels_dir, backup_dir, result)
        if writer.delete(path):
            result["deleted_files"] += 1


def _rewrite_image_labels(
    dataset: Dataset,
    labels_dir: Path,
    backup_dir: Path,
    result: dict,
    class_count: int,
    row_fix,
) -> None:
    seen: set[Path] = set()
    for image in _dataset_images(dataset):
        path = _find_label_file(labels_dir, image.name)
        if path is None or path in seen:
            continue
        seen.add(path)
        original = path.read_text(encoding="utf-8")
        fixed, changed, counts = _rewrite_text(original, class_count, row_fix)
        if not changed:
            continue
        _backup(path, labels_dir, backup_dir, result)
        writer.write_atomic(path, fixed)
        result["changed_files"] += 1
        result["removed_regions"] += counts["removed"]
        result["clipped_regions"] += counts["clipped"]


def _backup(path: Path, labels_dir: Path, backup_dir: Path, result: dict) -> None:
    relative = path.relative_to(labels_dir)
    target = backup_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    result["backed_up_files"] += 1


def _rewrite_text(text: str, class_count: int, row_fix) -> tuple[str, bool, dict]:
    out: list[str] = []
    counts = {"removed": 0, "clipped": 0}
    header_checked = False
    app_format = False
    changed = False
    kept_rows = 0

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        if not header_checked:
            header_checked = True
            if _is_header(tokens):
                app_format = True
                out.append(line)
                continue
        parsed = _parse_row(tokens, app_format)
        if parsed is None:
            out.append(line)
            kept_rows += 1
            continue
        replacement, outcome = row_fix(parsed, class_count, tokens)
        if outcome == "remove":
            changed = True
            counts["removed"] += 1
            continue
        if outcome == "clip":
            changed = True
            counts["clipped"] += 1
            out.append(replacement)
            kept_rows += 1
            continue
        out.append(line)
        kept_rows += 1

    if changed and kept_rows == 0:
        return "", changed, counts
    return ("\n".join(out) + "\n") if out else "", changed, counts


def _is_header(tokens: list[str]) -> bool:
    if len(tokens) != 2:
        return False
    try:
        return all(float(token).is_integer() for token in tokens)
    except ValueError:
        return False


def _parse_row(tokens: list[str], app_format: bool) -> _ParsedRow | None:
    if len(tokens) < 5:
        return None
    try:
        class_id = int(float(tokens[0]))
        coords = [float(token) for token in tokens[1:]]
    except ValueError:
        return None

    if app_format:
        cx, cy, w, h = coords[:4]
        polygon = coords[4:] if len(coords[4:]) >= 6 else None
        return _ParsedRow(class_id, (cx, cy, w, h), polygon, polygon is not None, True)

    if len(coords) >= 6 and len(coords) % 2 == 0:
        polygon = coords
        bbox = _bbox_from_polygon(polygon)
        if bbox is None:
            return None
        if _is_axis_aligned_rectangle(polygon):
            return _ParsedRow(class_id, bbox, None, True, False)
        return _ParsedRow(class_id, bbox, polygon, True, False)

    cx, cy, w, h = coords[:4]
    return _ParsedRow(class_id, (cx, cy, w, h), None, False, False)


def _valid_class(parsed: _ParsedRow, class_count: int) -> bool:
    return 0 <= parsed.class_id < class_count


def _fix_invalid_class(parsed: _ParsedRow, class_count: int, tokens: list[str]) -> tuple[str, str]:
    if not _valid_class(parsed, class_count):
        return "", "remove"
    return "", "keep"


def _fix_zero_area(parsed: _ParsedRow, class_count: int, tokens: list[str]) -> tuple[str, str]:
    if not _valid_class(parsed, class_count):
        return "", "keep"
    _cx, _cy, w, h = parsed.bbox
    if w <= 0 or h <= 0:
        return "", "remove"
    return "", "keep"


def _remove_out_of_bounds(parsed: _ParsedRow, class_count: int, tokens: list[str]) -> tuple[str, str]:
    if not _valid_class(parsed, class_count):
        return "", "keep"
    if _is_out_of_bounds(parsed.bbox):
        return "", "remove"
    return "", "keep"


def _clip_out_of_bounds(parsed: _ParsedRow, class_count: int, tokens: list[str]) -> tuple[str, str]:
    if not _valid_class(parsed, class_count):
        return "", "keep"
    if not _is_out_of_bounds(parsed.bbox):
        return "", "keep"
    if parsed.polygon is not None:
        polygon = [_clamp(coord) for coord in parsed.polygon]
        bbox = _bbox_from_polygon(polygon)
        if bbox is None or bbox[2] <= 0 or bbox[3] <= 0:
            return "", "remove"
        return _format_row(parsed.class_id, bbox, polygon, parsed.app_format), "clip"

    clipped = _clip_bbox(parsed.bbox)
    if clipped is None:
        return "", "remove"
    return _format_row(parsed.class_id, clipped, None, parsed.app_format), "clip"


def _is_out_of_bounds(bbox: tuple[float, float, float, float]) -> bool:
    cx, cy, w, h = bbox
    return (
        cx - w / 2 < -_BOUNDS_TOL
        or cy - h / 2 < -_BOUNDS_TOL
        or cx + w / 2 > 1 + _BOUNDS_TOL
        or cy + h / 2 > 1 + _BOUNDS_TOL
    )


def _clip_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    cx, cy, w, h = bbox
    x1 = _clamp(cx - w / 2)
    y1 = _clamp(cy - h / 2)
    x2 = _clamp(cx + w / 2)
    y2 = _clamp(cy + h / 2)
    new_w = x2 - x1
    new_h = y2 - y1
    if new_w <= 0 or new_h <= 0:
        return None
    return ((x1 + x2) / 2, (y1 + y2) / 2, new_w, new_h)


def _bbox_from_polygon(polygon: list[float]) -> tuple[float, float, float, float] | None:
    if len(polygon) < 6 or len(polygon) % 2 != 0:
        return None
    xs = polygon[0::2]
    ys = polygon[1::2]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return ((min_x + max_x) / 2, (min_y + max_y) / 2, max_x - min_x, max_y - min_y)


def _format_row(
    class_id: int,
    bbox: tuple[float, float, float, float],
    polygon: list[float] | None,
    app_format: bool,
) -> str:
    if polygon and not app_format:
        return " ".join([str(class_id), *(_fmt(value) for value in polygon)])

    parts = [str(class_id), *(_fmt(value) for value in bbox)]
    if polygon:
        parts.extend(_fmt(value) for value in polygon)
    return " ".join(parts)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))
