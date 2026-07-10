"""RF-DETR service handler.

The export wrapper (``onnx_export/arch/rfdetr.py``) bakes the full head math —
sigmoid + top-k selection + ``cxcywh -> xyxy`` + clamp ``[0,1]`` + background-class
drop — into the graph and emits ``(boxes, scores, labels)`` where boxes are
normalized ``[0,1]`` xyxy over the square model input (``box_coords:
"input_normalized"`` in meta). So the handler is a straight passthrough;
``postprocess`` maps the normalized boxes to Friendy directly, with no resize/pad
inverse. The service still replicates the adapter's input pipeline (square resize
to the variant's resolution + ImageNet normalize) so the graph sees the same
tensor.
"""

from .base import PassthroughHandler


class RFDetrHandler(PassthroughHandler):
    name = "rfdetr"
