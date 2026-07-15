"""Tile a training batch for tiling pipelines (batch_detect).

When an experiment attaches a tiling pipeline, the model is trained on the same
tiled representation it is validated/tested on: each full-frame image is split
into overlapping tiles and its ground-truth boxes are re-mapped into each tile's
local coordinates, so every tile becomes an independent training sample with its
own targets. Loss is then computed per tile inside the adapter's ``training_step``
(the standard, differentiable tiled-training approach).

Image geometry reuses ``chachak.boxes.tile_frame``; re-tiling the *targets* lives
here because inference never needs it (the eval pipeline predicts per tile and
merges detections back to the frame — it doesn't split labels).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# chachak default tiling knobs (mirror chachak.config.TilingConfig) used when a
# knob is left blank on the experiment.
_DEFAULT_TILE_WIDTH_PCT = 50.0
_DEFAULT_TILE_HEIGHT_PCT = 50.0
_DEFAULT_OVERLAP = 0.2

# A re-tiled box is kept only if at least this fraction of its original area
# falls inside the tile — drops slivers clipped at a tile seam that carry too
# little of the object to be a useful positive.
_MIN_VISIBLE_FRACTION = 0.1


def _ensure_chachak_importable() -> None:
    import sys

    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


def _resolve_tiling(tiling: Any) -> Tuple[float, float, float]:
    """Return (width_frac, height_frac, overlap) from a TilingSpec, with defaults."""
    width_pct = getattr(tiling, "tile_width_pct", None) or _DEFAULT_TILE_WIDTH_PCT
    height_pct = getattr(tiling, "tile_height_pct", None) or _DEFAULT_TILE_HEIGHT_PCT
    overlap = getattr(tiling, "overlap", None)
    if overlap is None:
        overlap = _DEFAULT_OVERLAP
    return width_pct / 100.0, height_pct / 100.0, float(overlap)


def _retile_target(
    target: Dict[str, Any],
    x0: int,
    y0: int,
    tile_w: int,
    tile_h: int,
) -> Optional[Dict[str, Any]]:
    """Re-map a full-frame target's boxes into one tile's local coordinates.

    Returns a new target dict whose ``boxes`` (absolute xyxy) and ``labels`` are
    restricted and clipped to the tile ``[x0, y0, x0+tile_w, y0+tile_h]``, or
    ``None`` when no ground-truth box has enough area inside the tile (an
    all-background tile — dropped for now to avoid empty-target edge cases in the
    adapters; feeding hard negatives is a future improvement).
    """
    boxes = target.get("boxes")
    labels = target.get("labels")
    if boxes is None or boxes.numel() == 0:
        return None

    boxes = boxes.float()
    x1 = torch.clamp(boxes[:, 0], min=x0, max=x0 + tile_w)
    y1 = torch.clamp(boxes[:, 1], min=y0, max=y0 + tile_h)
    x2 = torch.clamp(boxes[:, 2], min=x0, max=x0 + tile_w)
    y2 = torch.clamp(boxes[:, 3], min=y0, max=y0 + tile_h)

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter_area = inter_w * inter_h

    orig_w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    orig_h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    orig_area = (orig_w * orig_h).clamp(min=1e-6)

    keep = (inter_area / orig_area) >= _MIN_VISIBLE_FRACTION
    if not bool(keep.any()):
        return None

    # Translate the clipped boxes into tile-local coords.
    local = torch.stack(
        [x1 - x0, y1 - y0, x2 - x0, y2 - y0], dim=1
    )[keep]

    new_target = dict(target)
    new_target["boxes"] = local
    if labels is not None:
        new_target["labels"] = labels[keep]
    new_target["area"] = (local[:, 2] - local[:, 0]) * (local[:, 3] - local[:, 1])
    orig_size = target.get("orig_size")
    if orig_size is not None:
        new_target["orig_size"] = torch.tensor(
            [tile_h, tile_w], dtype=orig_size.dtype, device=orig_size.device
        )
    if "iscrowd" in target and target["iscrowd"] is not None:
        new_target["iscrowd"] = target["iscrowd"][keep]
    return new_target


def tile_batch(
    images: List[torch.Tensor],
    targets: List[Dict[str, Any]],
    tiling: Any,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    """Expand a batch of full frames into per-tile ``(image, target)`` samples.

    Each frame is split by :func:`chachak.boxes.tile_frame`; its targets are
    re-mapped per tile by :func:`_retile_target`. Tiles with no ground-truth
    inside are dropped. Returns flat lists suitable for feeding to an adapter's
    ``training_step`` (optionally re-chunked into micro-batches by the caller).
    """
    _ensure_chachak_importable()
    try:
        from chachak.boxes import tile_frame
    except ImportError:
        from boxes import tile_frame  # pragma: no cover - flat-script fallback

    width_frac, height_frac, overlap = _resolve_tiling(tiling)

    tile_images: List[torch.Tensor] = []
    tile_targets: List[Dict[str, Any]] = []
    for image, target in zip(images, targets):
        for tile, (x0, y0), (tile_w, tile_h) in tile_frame(
            image, width_frac, height_frac, overlap
        ):
            new_target = _retile_target(target, x0, y0, tile_w, tile_h)
            if new_target is None:
                continue
            tile_images.append(tile)
            tile_targets.append(new_target)
    return tile_images, tile_targets
