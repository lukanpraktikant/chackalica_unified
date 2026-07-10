"""Probe: is the f32 position-embedding twin equivalent to HF's original?

(a) compare the two embedding builders numerically for a few grid sizes;
(b) compare full torch model forward with original vs twin embedding (no ONNX).
"""
import sys
import numpy as np
import torch
sys.path.insert(0, "/app")
from friendy_chachkalica.registry import build_model
from friendy_chachkalica.onnx_export.arch.rtdetr import (
    _f32_sinusoidal_position_embedding, _float32_position_embedding,
)
from transformers.models.rt_detr.modeling_rt_detr import RTDetrSinePositionEmbedding

orig = RTDetrSinePositionEmbedding.__dict__["_cached_build_2d_sinusoidal_position_embedding"].__func__

print("== embedding builder: original vs twin ==")
for (h, w) in [(20, 20), (16, 20), (15, 15)]:
    a = orig(h, w)                                   # HF (default embed_dim?)
    print("orig sig ok, shape", tuple(a.shape), "dtype", a.dtype)
    b = _f32_sinusoidal_position_embedding(h, w, embed_dim=a.shape[-1])
    print(f"  h={h} w={w}: max abs diff = {(a.float()-b.float()).abs().max().item():.3e}  shapes {tuple(a.shape)} vs {tuple(b.shape)}")

print("\n== full forward: original vs twin embedding ==")
torch.manual_seed(0)
adapter = build_model("rtdetr", num_classes=3, weights=None); adapter.eval()
model = adapter.model
mean = torch.tensor(adapter.image_mean).view(3,1,1); std = torch.tensor(adapter.image_std).view(3,1,1)
torch.manual_seed(1)
norm = ((torch.rand(3,512,512)-mean)/std).unsqueeze(0)

with torch.no_grad():
    o1 = model(pixel_values=norm)
    with _float32_position_embedding():
        o2 = model(pixel_values=norm)
print("logits max diff:", (o1.logits-o2.logits).abs().max().item())
print("boxes  max diff:", (o1.pred_boxes-o2.pred_boxes).abs().max().item())
