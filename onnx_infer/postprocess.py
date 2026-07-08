"""Generic post-processing (pure NumPy): the canonical 3-tensor graph output
plus the applied :class:`Transform` -> a Friendy ``(N, 6)`` prediction array.

Uniform for every architecture — no per-arch logic here. The graph already
decoded, thresholded (to its arch default), NMS'd, and emitted xyxy boxes in its
own input-pixel frame (Contract A). This module only:

  1. applies the caller's runtime ``score_threshold`` (on top of the graph's),
  2. inverts the preprocessing transform back to original-image pixels,
  3. converts to normalized ``(x_center, y_center, width, height)`` and appends
     ``(confidence, class_id)`` — matching ``friendy_chachkalica.formats``.

Deliberately does **not** clip boxes to image bounds: the training adapters'
``xyxy_prediction_to_friendy`` doesn't either (the models clip internally), so
clipping here would break parity.
"""

from __future__ import annotations

import numpy as np

from .preprocess import Transform

# Column order of the Friendy prediction tensor (mirrors formats.FRIENDY_PREDICTION_COLUMNS).
FRIENDY_COLUMNS = ("x_center", "y_center", "width", "height", "confidence", "class_id")


def to_friendy(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    transform: Transform,
    score_threshold: float,
) -> np.ndarray:
    """boxes ``[N,4]`` xyxy input-pixels, scores ``[N]``, labels ``[N]`` -> ``[M,6]``."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    if boxes.shape[0] == 0:
        return np.zeros((0, len(FRIENDY_COLUMNS)), dtype=np.float32)

    keep = scores >= score_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if boxes.shape[0] == 0:
        return np.zeros((0, len(FRIENDY_COLUMNS)), dtype=np.float32)

    # Invert preprocessing: input-pixel -> original-image pixel.
    boxes = boxes.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - transform.pad_x) / transform.scale_x
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - transform.pad_y) / transform.scale_y

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2
    y_center = y1 + height / 2

    xywhn = np.stack(
        [
            x_center / transform.orig_w,
            y_center / transform.orig_h,
            width / transform.orig_w,
            height / transform.orig_h,
        ],
        axis=1,
    )
    return np.concatenate(
        [xywhn, scores.reshape(-1, 1), labels.reshape(-1, 1).astype(np.float32)],
        axis=1,
    ).astype(np.float32)
