"""Export-function registry — one exporter per architecture, keyed by the same
model name as ``friendy_chachkalica.registry.MODEL_REGISTRY``.

Each exporter has signature::

    export(adapter, *, num_classes, params, class_map, onnx_path) -> dict

It writes the ``.onnx`` at ``onnx_path`` and returns the ``meta.json`` dict
(Contract B). The service-side handler in ``onnx_infer/arch/<name>.py`` is the
matching half — the two must agree on Contracts A/B for that arch.
"""

from __future__ import annotations

try:
    from .arch.retinanet import export_retinanet
except ImportError:  # run as a flat script
    from arch.retinanet import export_retinanet

EXPORT_REGISTRY = {
    "retinanet": export_retinanet,
}


def get_exporter(model_name: str):
    try:
        return EXPORT_REGISTRY[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(EXPORT_REGISTRY)) or "(none)"
        raise ValueError(
            f"No ONNX exporter for arch {model_name!r}. Registered: {available}"
        ) from exc
