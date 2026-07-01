#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

# Original CHVG classes:
# 0=head, 1=glass, 2=red helmet, 3=yellow helmet, 4=blue helmet,
# 5=vest, 6=person, 7=white helmet
DEFAULT_CLASS_MAP = {
    0: 0,  # head -> head
    1: 1,  # glass -> glass
    2: 2,  # red -> helmet
    3: 2,  # yellow -> helmet
    4: 2,  # blue -> helmet
    5: 3,  # vest -> vest
    6: 4,  # person -> person
    7: 2,  # white -> helmet
}
NEW_CLASSES = ["head", "glass", "helmet", "vest", "person"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create remapped CHVG YOLO labels with helmet color classes collapsed into one helmet class."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/chvg_dataset/labels"),
        help="Source CHVG YOLO label directory.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data/chvg_dataset/labels_universal"),
        help="Destination directory for remapped labels.",
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

    converted_files = 0
    converted_objects = 0
    class_counts: dict[int, int] = {}
    dst.mkdir(parents=True, exist_ok=True)

    for label_path in sorted(src.rglob("*.txt")):
        relative_path = label_path.relative_to(src)
        output_path = dst / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_lines = []
        for line_number, raw_line in enumerate(label_path.read_text().splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) < 5:
                raise ValueError(f"Invalid YOLO label line {label_path}:{line_number}: {raw_line!r}")

            old_class_id = int(float(parts[0]))
            if old_class_id not in DEFAULT_CLASS_MAP:
                raise ValueError(
                    f"Unexpected CHVG class ID {old_class_id} at {label_path}:{line_number}"
                )

            new_class_id = DEFAULT_CLASS_MAP[old_class_id]
            parts[0] = str(new_class_id)
            output_lines.append(" ".join(parts))
            converted_objects += 1
            class_counts[new_class_id] = class_counts.get(new_class_id, 0) + 1

        output_path.write_text("\n".join(output_lines) + ("\n" if output_lines else ""))
        converted_files += 1

    classes_path = dst.parent / "classes_universal.txt"
    classes_path.write_text("\n".join(NEW_CLASSES) + "\n")

    print(f"source={src}")
    print(f"destination={dst}")
    print(f"files={converted_files} objects={converted_objects}")
    print(f"classes={NEW_CLASSES}")
    print(f"class_counts={dict(sorted(class_counts.items()))}")
    print(f"wrote_classes={classes_path}")


if __name__ == "__main__":
    main()
