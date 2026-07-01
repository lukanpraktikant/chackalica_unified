#!/usr/bin/env python3
"""Convert normalized polygon label rows to YOLO xywh label rows.

Rows already in YOLO format are preserved. Polygon rows are expected to be:
    class_id x1 y1 x2 y2 ...
with normalized point coordinates in [0, 1]. The output is:
    class_id x_center y_center width height
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _convert_line(line: str) -> tuple[str, bool]:
    stripped = line.strip()
    if not stripped:
        return line, False

    parts = stripped.split()
    if len(parts) == 5:
        return " ".join(parts), False
    if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
        raise ValueError(f"expected YOLO row or polygon row, got {len(parts)} columns: {stripped}")

    class_id = parts[0]
    values = [float(value) for value in parts[1:]]
    xs = values[0::2]
    ys = values[1::2]

    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)

    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        raise ValueError(f"non-positive box after polygon conversion: {stripped}")
    if not all(0 <= value <= 1 for value in (x_center, y_center, width, height)):
        raise ValueError(f"converted YOLO box is outside normalized range: {stripped}")

    return " ".join(
        [class_id, *(_format_float(value) for value in (x_center, y_center, width, height))]
    ), True


def convert_file(path: Path, backup_suffix: str | None, dry_run: bool = False) -> tuple[int, int]:
    original_text = path.read_text()
    had_trailing_newline = original_text.endswith("\n")
    lines = original_text.splitlines()

    converted_lines = []
    converted_count = 0
    for line_number, line in enumerate(lines, start=1):
        try:
            converted_line, changed = _convert_line(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        converted_lines.append(converted_line)
        converted_count += int(changed)

    if converted_count == 0:
        return 0, len(lines)

    output_text = "\n".join(converted_lines)
    if had_trailing_newline:
        output_text += "\n"

    if not dry_run:
        if backup_suffix:
            backup_path = path.with_name(path.name + backup_suffix)
            if not backup_path.exists():
                backup_path.write_text(original_text)
        path.write_text(output_text)

    return converted_count, len(lines)


def convert_dir(labels_dir: Path, backup_suffix: str | None, dry_run: bool = False) -> dict[str, int]:
    stats = {"files_seen": 0, "files_changed": 0, "rows_seen": 0, "rows_converted": 0}
    for path in sorted(labels_dir.rglob("*.txt")):
        stats["files_seen"] += 1
        converted, rows = convert_file(path, backup_suffix=backup_suffix, dry_run=dry_run)
        stats["rows_seen"] += rows
        stats["rows_converted"] += converted
        if converted:
            stats["files_changed"] += 1
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels_dir", type=Path, help="Directory containing .txt label files")
    parser.add_argument(
        "--backup-suffix",
        default=".polygon.bak",
        help="Backup suffix for changed files; use empty string to disable",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_dir = args.labels_dir.resolve()
    if not labels_dir.is_dir():
        raise SystemExit(f"Labels directory not found: {labels_dir}")

    backup_suffix = args.backup_suffix or None
    stats = convert_dir(labels_dir, backup_suffix=backup_suffix, dry_run=args.dry_run)
    print(
        "converted polygon labels:",
        f"files_seen={stats['files_seen']}",
        f"files_changed={stats['files_changed']}",
        f"rows_seen={stats['rows_seen']}",
        f"rows_converted={stats['rows_converted']}",
        f"dry_run={args.dry_run}",
    )


if __name__ == "__main__":
    main()
