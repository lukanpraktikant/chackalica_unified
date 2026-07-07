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
    return (labels_root / relative_path).with_suffix(".txt")


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


def _normalized_polygon_to_xyxy(points, image_width, image_height):
    xs = points[0::2]
    ys = points[1::2]
    x1 = min(xs) * image_width
    y1 = min(ys) * image_height
    x2 = max(xs) * image_width
    y2 = max(ys) * image_height
    return [x1, y1, x2, y2]


class YoloDetectionDataset:
    def __init__(self, images_dir, labels_dir, classes, transforms=None):
        self.transforms = transforms
        self.images_root = Path(images_dir).resolve()
        self.labels_root = Path(labels_dir).resolve()
        self.names = dict(classes)
        self.image_paths = _resolve_image_dir(self.images_root)

        if not self.labels_root.is_dir():
            raise FileNotFoundError(f"Could not resolve labels directory: {self.labels_root}")

        label_count = sum(
            1
            for image_path in self.image_paths
            if _image_to_label_path(image_path, self.images_root, self.labels_root).exists()
        )
        print(
            f"[data] Dataset ready: images={len(self.image_paths)} labels_found={label_count} "
            f"classes={len(self.names)} images_root={self.images_root} labels_root={self.labels_root}"
        )

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


class TrainAugmentations:
    """Random train-time augmentations, each applied to a fraction of samples.

    ``fractions`` maps augmentation name -> probability per sample per epoch
    (see ``config.AUGMENTATION_KEYS``). Boxes stay absolute xyxy on the
    (possibly transformed) image; images keep their original size, so nothing
    downstream of the dataset changes shape.
    """

    SCALE_RANGE = (0.6, 1.0)  # crop window size as a fraction of each image side
    MIN_BOX_VISIBILITY = 0.25  # drop boxes with less than this area left in the crop
    MIN_BOX_SIDE_PX = 2.0  # drop boxes thinner than this after the crop resize

    def __init__(self, fractions: dict):
        self.hflip = float(fractions.get("hflip", 0.0))
        self.scale_crop = float(fractions.get("scale_crop", 0.0))

    def __call__(self, image, target):
        import torch

        # torch.rand honors the per-worker seeding the DataLoader sets up, so
        # runs stay reproducible under training.seed.
        if self.hflip > 0 and torch.rand(1).item() < self.hflip:
            image, target = _horizontal_flip(image, target)
        if self.scale_crop > 0 and torch.rand(1).item() < self.scale_crop:
            image, target = _random_scale_crop(
                image, target, self.SCALE_RANGE, self.MIN_BOX_VISIBILITY, self.MIN_BOX_SIDE_PX
            )
        return image, target


def _horizontal_flip(image, target):
    import torch

    image = torch.flip(image, dims=[2])  # (C, H, W) -> flip width
    boxes = target["boxes"]
    if boxes.numel():
        width = image.shape[2]
        flipped = boxes.clone()
        flipped[:, 0] = width - boxes[:, 2]
        flipped[:, 2] = width - boxes[:, 0]
        target["boxes"] = flipped
    return image, target


def _random_scale_crop(image, target, scale_range, min_visibility, min_side_px):
    """Crop a random window of scale_range x the image, resize back to full size.

    Boxes are shifted into the crop, clipped at its edges, rescaled with the
    image, and dropped when the crop leaves too little of them (less than
    ``min_visibility`` of their area or a side under ``min_side_px``).
    """
    import torch

    _, height, width = image.shape
    low, high = scale_range
    scale = low + (high - low) * torch.rand(1).item()
    crop_h = max(1, min(height, round(height * scale)))
    crop_w = max(1, min(width, round(width * scale)))
    if crop_h == height and crop_w == width:
        return image, target

    top = int(torch.randint(0, height - crop_h + 1, (1,)).item())
    left = int(torch.randint(0, width - crop_w + 1, (1,)).item())

    cropped = image[:, top : top + crop_h, left : left + crop_w]
    image = torch.nn.functional.interpolate(
        cropped.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
    ).squeeze(0)

    boxes = target["boxes"]
    if boxes.numel():
        shifted = boxes - boxes.new_tensor([left, top, left, top])
        clipped = shifted.clone()
        clipped[:, 0::2] = clipped[:, 0::2].clamp(0, crop_w)
        clipped[:, 1::2] = clipped[:, 1::2].clamp(0, crop_h)

        # Rescale from crop coordinates back to the full-size image.
        scale_x = width / crop_w
        scale_y = height / crop_h
        clipped = clipped * clipped.new_tensor([scale_x, scale_y, scale_x, scale_y])

        widths = clipped[:, 2] - clipped[:, 0]
        heights = clipped[:, 3] - clipped[:, 1]
        original_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        visible_area = widths * heights / (scale_x * scale_y)
        keep = (
            (widths >= min_side_px)
            & (heights >= min_side_px)
            & (visible_area >= min_visibility * original_area.clamp(min=1e-6))
        )

        target["boxes"] = clipped[keep]
        target["labels"] = target["labels"][keep]
        target["iscrowd"] = target["iscrowd"][keep]
        target["area"] = (
            (target["boxes"][:, 2] - target["boxes"][:, 0])
            * (target["boxes"][:, 3] - target["boxes"][:, 1])
        )
    return image, target


def build_train_augmentations(dataset_config: DatasetConfig):
    """The transforms callable for a train dataset, or None when unconfigured."""
    fractions = {k: v for k, v in (dataset_config.augmentation or {}).items() if v > 0}
    if not fractions:
        return None
    return TrainAugmentations(fractions)


def build_dataset(dataset_config: DatasetConfig, transforms=None) -> YoloDetectionDataset:
    print(
        f"[data] Building dataset name={dataset_config.name} role={dataset_config.role} "
        f"classes={len(dataset_config.classes)} transforms={transforms is not None}"
    )
    return YoloDetectionDataset(
        images_dir=dataset_config.images,
        labels_dir=dataset_config.labels,
        classes=dataset_config.classes,
        transforms=transforms,
    )


def build_train_dataloader(
    config: ExperimentConfig,
    dataset_config: DatasetConfig,
) -> DataLoader:
    transforms = build_train_augmentations(dataset_config)
    if transforms is not None:
        print(
            f"[data] Train augmentations for {dataset_config.name}: "
            f"hflip={transforms.hflip} scale_crop={transforms.scale_crop}"
        )
    dataset = build_dataset(dataset_config, transforms=transforms)
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

    dataset = build_dataset(dataset_config)
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
