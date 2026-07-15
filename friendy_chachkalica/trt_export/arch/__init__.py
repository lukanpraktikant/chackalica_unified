"""Arch-specific TensorRT ONNX preparation.

Most archs (rtdetr, rfdetr) compile straight from their standard ONNX graph — the
DETR-family top-k is fixed-size, so TensorRT is happy. Those have NO entry here
and ``build_engine`` compiles the standard ``.onnx`` directly.

Archs whose standard ONNX bakes data-dependent NMS/decode (retinanet, yolox) DO
have an entry: a ``prep(adapter, meta, out_onnx_path)`` callable that re-exports a
raw-output graph and appends the ``EfficientNMS_TRT`` plugin (see
``efficientnms.py``). These need the torch adapter (hence the ``.pt`` checkpoint),
not just the standard ``.onnx``.
"""

from __future__ import annotations

try:
    from .retinanet import prep_retinanet
    from .yolox import prep_yolox
except ImportError:  # run flat (cwd on sys.path)
    from trt_export.arch.retinanet import prep_retinanet  # type: ignore
    from trt_export.arch.yolox import prep_yolox  # type: ignore

# arch name -> prep callable. Archs absent here compile from their standard ONNX.
TRT_PREP_REGISTRY = {
    "yolox": prep_yolox,
    "retinanet": prep_retinanet,
}


def get_trt_prep(arch: str):
    """Return the arch's TRT ONNX-prep callable, or ``None`` for passthrough archs."""
    return TRT_PREP_REGISTRY.get(arch)


__all__ = ["TRT_PREP_REGISTRY", "get_trt_prep"]
