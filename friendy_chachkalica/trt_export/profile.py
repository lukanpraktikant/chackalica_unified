"""Derive a TensorRT optimization profile (min / opt / max input H×W) from a
model's ``meta.json`` (Contract B). Batch is always 1, channels always 3 — only
the spatial dims vary, exactly matching the ONNX graph's dynamic ``pixel_values``
axes.

Per ``resize_mode``:

- ``square``       (RF-DETR): one static shape, ``size × size`` — the ONNX graph
  has fixed H/W, so min == opt == max.
- ``longest_side`` (RT-DETR): bounded above by ``max_size`` (the service never
  feeds a larger input); min side is one ``multiple``.
- ``none``         (RetinaNet, YOLOX): the service passes the image through
  unchanged (only padded up to ``multiple``), so the input size is genuinely
  open-ended. We pick a sensible default range and let the caller override it —
  an inference image larger than ``max`` needs an engine rebuilt with a bigger
  ``max_hw``.
"""

from __future__ import annotations

from typing import Optional, Tuple

HW = Tuple[int, int]

# Fallback spatial bounds (pixels) for open-ended (``resize_mode: none``) archs.
_DEFAULT_MIN_SIDE = 64
_DEFAULT_OPT_SIDE = 640
_DEFAULT_MAX_SIDE = 1024


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return int(value)
    return int((value + multiple - 1) // multiple * multiple)


def profile_from_meta(
    meta: dict,
    *,
    min_hw: Optional[HW] = None,
    opt_hw: Optional[HW] = None,
    max_hw: Optional[HW] = None,
) -> Tuple[HW, HW, HW]:
    """Return ``(min_hw, opt_hw, max_hw)`` for the graph's ``pixel_values`` input.

    Explicit ``*_hw`` overrides always win; anything left ``None`` is derived from
    the meta's ``input`` spec.
    """
    spec = meta.get("input", {}) or {}
    resize_mode = spec.get("resize_mode", "none")
    multiple = int(spec.get("multiple") or 0)
    step = multiple if multiple > 1 else 1

    if resize_mode == "square":
        size = int(spec["size"])
        static: HW = (size, size)
        return (min_hw or static, opt_hw or static, max_hw or static)

    if resize_mode == "longest_side":
        top = _round_up(int(spec.get("max_size") or _DEFAULT_MAX_SIDE), step)
        low = _round_up(_DEFAULT_MIN_SIDE, step)
        default_min: HW = (low, low)
        default_opt: HW = (top, top)
        default_max: HW = (top, top)
    else:  # "none" — open-ended; use overridable defaults.
        low = _round_up(_DEFAULT_MIN_SIDE, step)
        mid = _round_up(_DEFAULT_OPT_SIDE, step)
        high = _round_up(_DEFAULT_MAX_SIDE, step)
        default_min = (low, low)
        default_opt = (mid, mid)
        default_max = (high, high)

    return (min_hw or default_min, opt_hw or default_opt, max_hw or default_max)
