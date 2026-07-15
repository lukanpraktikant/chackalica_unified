"""``TrtAdapter`` — the torch-facing drop-in for the trained-model adapters,
backed by a TensorRT engine.

A near-verbatim twin of ``onnx_infer/adapter.py::OnnxAdapter``: same surface
(``predict`` / ``to`` / ``eval``) so it slots into ``load_checkpoint_adapter`` and
everything downstream with no caller changes. The only difference is the session
(``TrtModel`` instead of ``OnnxModel``); the meta-driven pre/post-processing and
the per-arch output handler are reused from ``onnx_infer`` unchanged, because a
TensorRT engine is just a recompilation of the same ONNX graph (same contracts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from onnx_infer.arch import get_handler
from onnx_infer.meta import ModelMeta
from onnx_infer.postprocess import to_friendy
from onnx_infer.preprocess import preprocess

from .session import TrtModel


class TrtAdapter:
    def __init__(self, engine_path: str | Path, meta: ModelMeta, device="cuda") -> None:
        self.meta = meta
        self.name = meta.arch
        self.num_classes = meta.num_classes
        self.score_threshold = meta.score_threshold
        self._handler = get_handler(meta.arch)
        self._model = TrtModel(engine_path, meta, device=device)

    def to(self, device) -> "TrtAdapter":
        self._model.to(device)
        return self

    def eval(self) -> "TrtAdapter":  # parity with nn.Module-style adapters
        return self

    def predict(self, images, score_threshold: Optional[float] = None):
        """List of CHW float [0,1] tensors -> list of Friendy ``(N,6)`` tensors."""
        import torch

        threshold = self.score_threshold if score_threshold is None else score_threshold
        results = []
        for image in images:
            chw = _to_chw_numpy(image)
            batched, transform = preprocess(chw, self.meta)
            raw = self._model.run(batched)
            boxes, scores, labels = self._handler.adapt_outputs(raw)
            friendy = to_friendy(
                boxes, scores, labels, transform, threshold,
                clip_boxes=self.meta.clip_boxes,
                box_coords=self.meta.box_coords,
            )
            results.append(torch.from_numpy(friendy))
        return results


def _to_chw_numpy(image) -> np.ndarray:
    """Coerce a CHW image (torch tensor or ndarray) to a float32 numpy array."""
    if hasattr(image, "detach"):  # torch.Tensor
        image = image.detach().cpu().numpy()
    return np.asarray(image, dtype=np.float32)


def load_trt_adapter(engine_path: str | Path, device="cuda") -> tuple[TrtAdapter, dict]:
    """Build a :class:`TrtAdapter` + an ``info`` dict shaped like
    ``chachak.infer.load_checkpoint_adapter``'s second return value.
    """
    engine_path = Path(engine_path)
    meta = ModelMeta.load(engine_path.with_suffix(".meta.json"))
    adapter = TrtAdapter(engine_path, meta, device=device)
    info = {
        "model_name": meta.arch,
        "num_classes": meta.num_classes,
        "params": {},
        "train_classes": dict(meta.class_map),
    }
    print(
        f"[trt_infer] Adapter ready: {meta.arch} num_classes={meta.num_classes} "
        f"classes={len(meta.class_map)} device={device}"
    )
    return adapter, info
