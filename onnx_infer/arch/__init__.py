"""Per-architecture service handlers, one file per arch, dispatched by name —
mirroring ``friendy_chachkalica/adapters`` + ``registry.py::MODEL_REGISTRY``.

Adding a 6th arch is a drop-in: add ``arch/<name>.py`` with an ``ArchHandler``
subclass and register it here.
"""

from __future__ import annotations

from ..errors import UnknownArchError
from .base import ArchHandler
from .retinanet import RetinaNetHandler

ARCH_REGISTRY: dict[str, ArchHandler] = {
    handler.name: handler
    for handler in (RetinaNetHandler(),)
}


def get_handler(arch: str) -> ArchHandler:
    try:
        return ARCH_REGISTRY[arch]
    except KeyError as exc:
        available = ", ".join(sorted(ARCH_REGISTRY)) or "(none)"
        raise UnknownArchError(
            f"No ONNX handler for arch {arch!r}. Registered: {available}"
        ) from exc


__all__ = ["ARCH_REGISTRY", "get_handler", "ArchHandler"]
