#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

# Expected source classes from labels_universal:
# 0=head, 1=glass, 2=helmet, 3=vest, 4=person
HEAD_CLASS_ID = 0
HELMET_CLASS_ID = 2
VEST_CLASS_ID = 3
PERSON_CLASS_ID = 4
NO_HELMET_CLASS_ID = 5
NO_VEST_CLASS_ID = 6

SOURCE_CLASSES = ["head", "glass", "helmet", "vest", "person"]
NEW_CLASSES = SOURCE_CLASSES + ["no_helmet", "no_vest"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create augmented CHVG YOLO labels by adding no_helmet boxes from "
            "unmatched head boxes, and no_vest boxes from unmatched person boxes."
        )
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/chvg_dataset/labels_universal"),
        help="Source remapped CHVG YOLO label directory.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data/chvg_dataset/labels_universal_with_missing_ppe"),
        help="Destination directory for augmented labels.",
    )
    parser.add_argument(
        "--head-y-shift",
        type=float,
        default=-0.15,
        help="Shift synthetic no_helmet boxes by this fraction of head height; negative moves up.",
    )
    parser.add_argument(
        "--helmet-iou-threshold",
        type=float,
        default=0.10,
        help="Treat a head as helmeted when its IoU with any helmet box is at least this value.",
    )
    parser.add_argument(
        "--vest-iou-threshold",
        type=float,
        default=0.10,
        help="Treat a person as vested when its IoU with any vest box is at least this value.",
    )
    parser.add_argument(
        "--image-level",
        action="store_true",
        help="Use image-level absence checks instead of per-box IoU matching.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting files in the destination directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"Source labels directory does not exist: {src}")
    if dst.exists() and any(dst.rglob("*.txt")) and not args.overwrite:
        raise FileExistsError(
            f"Destination already contains label files: {dst}. Use --overwrite to replace them."
        )

    dst.mkdir(parents=True, exist_ok=True)

    files = 0
    copied_objects = 0
    added_no_helmet = 0
    added_no_vest = 0
    class_counts: dict[int, int] = {}

    for label_path in sorted(src.rglob("*.txt")):
        relative_path = label_path.relative_to(src)
        output_path = dst / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = _read_label_records(label_path)
        output_lines = [_format_record(record) for record in records]

        for class_id, *_ in records:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        copied_objects += len(records)

        helmets = [record for record in records if record[0] == HELMET_CLASS_ID]
        vests = [record for record in records if record[0] == VEST_CLASS_ID]
        has_helmet = bool(helmets)
        has_vest = bool(vests)

        for record in records:
            if record[0] == HEAD_CLASS_ID and _should_add_missing_label(
                record,
                helmets,
                has_positive=has_helmet,
                threshold=args.helmet_iou_threshold,
                image_level=args.image_level,
            ):
                synthetic = _make_no_helmet_record(record, args.head_y_shift)
                output_lines.append(_format_record(synthetic))
                class_counts[NO_HELMET_CLASS_ID] = class_counts.get(NO_HELMET_CLASS_ID, 0) + 1
                added_no_helmet += 1

            if record[0] == PERSON_CLASS_ID and _should_add_missing_label(
                record,
                vests,
                has_positive=has_vest,
                threshold=args.vest_iou_threshold,
                image_level=args.image_level,
            ):
                synthetic = (NO_VEST_CLASS_ID, *record[1:])
                output_lines.append(_format_record(synthetic))
                class_counts[NO_VEST_CLASS_ID] = class_counts.get(NO_VEST_CLASS_ID, 0) + 1
                added_no_vest += 1

        output_path.write_text("\n".join(output_lines) + ("\n" if output_lines else ""))
        files += 1

    classes_path = dst.parent / "classes_universal_with_missing_ppe.txt"
    classes_path.write_text("\n".join(NEW_CLASSES) + "\n")

    print(f"source={src}")
    print(f"destination={dst}")
    print(f"files={files} copied_objects={copied_objects}")
    print(f"added_no_helmet={added_no_helmet} added_no_vest={added_no_vest}")
    print(f"classes={NEW_CLASSES}")
    print(f"class_counts={dict(sorted(class_counts.items()))}")
    print(f"wrote_classes={classes_path}")


def _read_label_records(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    records = []
    for line_number, raw_line in enumerate(label_path.read_text().splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO label line {label_path}:{line_number}: {raw_line!r}")

        class_id = int(float(parts[0]))
        if class_id < 0 or class_id >= len(SOURCE_CLASSES):
            raise ValueError(
                f"Unexpected source class ID {class_id} at {label_path}:{line_number}; "
                f"expected labels_universal classes 0-{len(SOURCE_CLASSES) - 1}"
            )

        x_center, y_center, width, height = (float(value) for value in parts[1:5])
        _validate_yolo_box(label_path, line_number, x_center, y_center, width, height)
        records.append((class_id, x_center, y_center, width, height))
    return records


def _should_add_missing_label(
    source_record: tuple[int, float, float, float, float],
    positive_records: list[tuple[int, float, float, float, float]],
    has_positive: bool,
    threshold: float,
    image_level: bool,
) -> bool:
    if image_level:
        return not has_positive
    return all(_iou(source_record, positive_record) < threshold for positive_record in positive_records)


def _make_no_helmet_record(
    head_record: tuple[int, float, float, float, float],
    head_y_shift: float,
) -> tuple[int, float, float, float, float]:
    _, x_center, y_center, width, height = head_record
    shifted_y = y_center + height * head_y_shift
    shifted_y = _clamp(shifted_y, height / 2.0, 1.0 - height / 2.0)
    return (NO_HELMET_CLASS_ID, x_center, shifted_y, width, height)


def _iou(
    left: tuple[int, float, float, float, float],
    right: tuple[int, float, float, float, float],
) -> float:
    left_x1, left_y1, left_x2, left_y2 = _xywh_to_xyxy(left)
    right_x1, right_y1, right_x2, right_y2 = _xywh_to_xyxy(right)

    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _xywh_to_xyxy(record: tuple[int, float, float, float, float]) -> tuple[float, float, float, float]:
    _, x_center, y_center, width, height = record
    return (
        x_center - width / 2.0,
        y_center - height / 2.0,
        x_center + width / 2.0,
        y_center + height / 2.0,
    )


def _validate_yolo_box(
    label_path: Path,
    line_number: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> None:
    values = (x_center, y_center, width, height)
    if any(value < 0.0 or value > 1.0 for value in values) or width <= 0.0 or height <= 0.0:
        raise ValueError(
            f"Invalid normalized YOLO box at {label_path}:{line_number}: "
            f"{x_center} {y_center} {width} {height}"
        )


def _format_record(record: tuple[int, float, float, float, float]) -> str:
    class_id, x_center, y_center, width, height = record
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


if __name__ == "__main__":
    main()
