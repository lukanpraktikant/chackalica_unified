"""``ArchHandler`` — the one per-architecture seam on the service side.

The generic core (preprocess/postprocess) is arch-agnostic. The only thing that
genuinely varies between archs at inference time is how the raw onnxruntime
output list maps onto the canonical Contract-A 3-tensor
``(boxes_xyxy_input_px, scores, labels)``.

For archs exported through our own wrapper the graph already emits exactly those
three in order, so :class:`PassthroughHandler` is an identity. RF-DETR, exported
via the ``rfdetr`` package's native exporter, has a different output signature
and overrides :meth:`adapt_outputs` in ``arch/rfdetr.py``.
"""

from __future__ import annotations

import numpy as np


class ArchHandler:
    """Base handler: subclass and set ``name``; override ``adapt_outputs`` if the
    graph's raw output layout isn't already ``(boxes, scores, labels)``."""

    name: str = ""

    def adapt_outputs(
        self, outputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class PassthroughHandler(ArchHandler):
    """Graph outputs are already ``(boxes[N,4], scores[N], labels[N])`` in order."""

    def adapt_outputs(
        self, outputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(outputs) < 3:
            raise ValueError(
                f"{self.name}: expected 3 graph outputs (boxes, scores, labels), "
                f"got {len(outputs)}"
            )
        boxes, scores, labels = outputs[0], outputs[1], outputs[2]
        return (
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(scores, dtype=np.float32).reshape(-1),
            np.asarray(labels).reshape(-1).astype(np.int64),
        )
