from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

try:
    from .config import DatasetConfig, ExperimentConfig
    from .formats import xywhn_to_xyxy
except ImportError:
    from config import DatasetConfig, ExperimentConfig
    from formats import xywhn_to_xyxy


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def _resolve_image_dir(images_root):
    images_root = Path(images_root).resolve()
    if not images_root.is_dir():
        raise FileNotFoundError(f"Could not resolve images directory: {images_root}")
    return sorted(
        path.resolve()
        for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _image_to_label_path(image_path, images_root, labels_root):
    relative_path = image_path.relative_to(images_root)
    # Two label-naming conventions occur in practice: "<stem>.txt" (drop the
    # image suffix, e.g. frame_000.txt) and "<image filename>.txt" (append,
    # e.g. frame_000.jpg.txt). Prefer the appended form when it exists on disk,
    # otherwise fall back to the stripped form (also the not-found default, so
    # callers that only test .exists() behave as before).
    stripped = (labels_root / relative_path).with_suffix(".txt")
    appended = labels_root / relative_path.parent / (relative_path.name + ".txt")
    if appended.exists() and not stripped.exists():
        return appended
    return stripped


def _read_yolo_label_file(label_path, image_width, image_height):
    boxes = []
    labels = []

    if not label_path.exists():
        return boxes, labels

    with open(label_path) as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
            if len(values) == 4:
                box = xywhn_to_xyxy(values, image_width, image_height)
            elif len(values) == 5:
                # YOLO OBB format (cx cy w h angle) — treat as axis-aligned bbox
                box = xywhn_to_xyxy(values[:4], image_width, image_height)
            elif len(values) >= 6 and len(values) % 2 == 0:
                box = _normalized_polygon_to_xyxy(values, image_width, image_height)
            else:
                continue

            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            boxes.append(box)
            labels.append(class_id)

    return boxes, labels


def _scan_label_file(label_path, valid_class_ids):
    """Count label lines with out-of-range class ids or clearly unnormalized coords.

    Mirrors ``_read_yolo_label_file``'s parsing so it only inspects lines that
    would actually produce a box. Coordinates are flagged when far outside
    [0, 1] (i.e. pixel coordinates in a file that must be normalized); for the
    5-value OBB form the trailing angle is exempt from the range check.
    """
    bad_class_lines = 0
    unnormalized_lines = 0
    valid_class_ids = set(valid_class_ids)
    with open(label_path) as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
                values = [float(value) for value in parts[1:]]
            except ValueError:
                continue
            if class_id not in valid_class_ids:
                bad_class_lines += 1
            coords = values[:4] if len(values) == 5 else values
            if any(value < -0.5 or value > 1.5 for value in coords):
                unnormalized_lines += 1
    return bad_class_lines, unnormalized_lines


def _normalized_polygon_to_xyxy(points, image_width, image_height):
    xs = points[0::2]
    ys = points[1::2]
    x1 = min(xs) * image_width
    y1 = min(ys) * image_height
    x2 = max(xs) * image_width
    y2 = max(ys) * image_height
    return [x1, y1, x2, y2]


class YoloDetectionDataset:
    def __init__(self, images_dir, labels_dir, classes, transforms=None, require_labels=False):
        self.transforms = transforms
        self.images_root = Path(images_dir).resolve()
        self.labels_root = Path(labels_dir).resolve()
        self.names = dict(classes)
        self.image_paths = _resolve_image_dir(self.images_root)

        if not self.labels_root.is_dir():
            raise FileNotFoundError(f"Could not resolve labels directory: {self.labels_root}")

        label_count = 0
        bad_class_lines = 0
        unnormalized_lines = 0
        for image_path in self.image_paths:
            label_path = _image_to_label_path(image_path, self.images_root, self.labels_root)
            if not label_path.exists():
                continue
            label_count += 1
            bad_classes, unnormalized = _scan_label_file(label_path, self.names)
            bad_class_lines += bad_classes
            unnormalized_lines += unnormalized
        print(
            f"[data] Dataset ready: images={len(self.image_paths)} labels_found={label_count} "
            f"classes={len(self.names)} images_root={self.images_root} labels_root={self.labels_root}"
        )
        # Both anomalies below corrupt training/eval without crashing, so call
        # them out once per dataset instead of failing silently per line.
        if bad_class_lines:
            print(
                f"[data] WARNING: {bad_class_lines} label line(s) use class ids outside the "
                f"{len(self.names)} configured classes. Those boxes will crash training or be "
                f"silently dropped from eval — check that the class list matches how the "
                f"labels were exported (e.g. 0-based vs 1-based ids)."
            )
        if unnormalized_lines:
            print(
                f"[data] WARNING: {unnormalized_lines} label line(s) have coordinates far "
                f"outside [0, 1]. Labels must be normalized YOLO xywh; pixel-coordinate "
                f"labels produce garbage boxes without crashing."
            )
        # Zero labels found almost always means a path/naming mismatch, not a
        # dataset that is genuinely all-background. For eval sets this silently
        # produces meaningless loss/mAP (every prediction scored against empty
        # ground truth), so fail loud. Train tolerates it with a warning.
        if label_count == 0:
            message = (
                f"No labels matched any of {len(self.image_paths)} images under "
                f"{self.labels_root} (checked both '<stem>.txt' and "
                f"'<image>.txt' naming). Check the labels directory and filename "
                f"convention."
            )
            if require_labels:
                raise FileNotFoundError(message)
            print(f"[data] WARNING: {message}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        import numpy as np
        import torch
        from PIL import Image

        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        label_path = _image_to_label_path(image_path, self.images_root, self.labels_root)
        boxes, labels = _read_yolo_label_file(label_path, width, height)

        image = np.asarray(image).copy()
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
        }

        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_dataset(dataset_config: DatasetConfig, require_labels: bool = False) -> YoloDetectionDataset:
    print(
        f"[data] Building dataset name={dataset_config.name} role={dataset_config.role} "
        f"classes={len(dataset_config.classes)}"
    )
    return YoloDetectionDataset(
        images_dir=dataset_config.images,
        labels_dir=dataset_config.labels,
        classes=dataset_config.classes,
        require_labels=require_labels,
    )


def build_train_dataloader(
    config: ExperimentConfig,
    dataset_config: DatasetConfig,
) -> DataLoader:
    dataset = build_dataset(dataset_config)
    print(
        f"[data] Building train dataloader dataset={dataset_config.name} "
        f"batch_size={config.training.batch_size} workers={config.training.num_workers} "
        f"shuffle=True"
    )
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=_cuda_is_available(),
    )


def build_eval_dataloader(
    dataset_config: Optional[DatasetConfig],
    config: ExperimentConfig,
) -> Optional[DataLoader]:
    if dataset_config is None:
        return None

    batch_size = config.evaluation.batch_size or config.training.batch_size
    num_workers = config.evaluation.num_workers
    if num_workers is None:
        num_workers = config.training.num_workers

    # Eval requires labels: a val/test set with none makes every loss and mAP
    # meaningless, so surface it as a hard error instead of "training complete".
    dataset = build_dataset(dataset_config, require_labels=True)
    print(
        f"[data] Building eval dataloader dataset={dataset_config.name} "
        f"batch_size={batch_size} workers={num_workers} shuffle=False"
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=_cuda_is_available(),
    )


def _cuda_is_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()
