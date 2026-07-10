"""RT-DETR service handler.

The export wrapper (``onnx_export/arch/rtdetr.py``) bakes sigmoid + top-k query
selection into the graph and emits ``(boxes, scores, labels)`` where boxes are
normalized ``[0,1]`` xyxy over the model input (``box_coords: "input_normalized"``
in meta). So the handler is a straight passthrough; ``postprocess`` maps the
normalized boxes to Friendy directly, with no resize/pad inverse. The service
still replicates the adapter's input pipeline (longest-side resize, ImageNet
normalize, pad to a multiple of 32) so the graph sees the same tensor.
"""

from .base import PassthroughHandler


class RTDetrHandler(PassthroughHandler):
    name = "rtdetr"
