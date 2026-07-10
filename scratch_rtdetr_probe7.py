"""Localize RT-DETR ONNX divergence: encoder query-selection (anchors) vs
decoder deformable-attention (grid_sample). Export intermediates and compare."""
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0, "/app")
from friendy_chachkalica.registry import build_model
from friendy_chachkalica.onnx_export.arch.rtdetr import _float32_position_embedding

torch.manual_seed(0)
adapter = build_model("rtdetr", num_classes=3, weights=None); adapter.eval()
inner = adapter.model.model  # RTDetrModel
mean = torch.tensor(adapter.image_mean).view(3,1,1); std = torch.tensor(adapter.image_std).view(3,1,1)
torch.manual_seed(1)
norm = ((torch.rand(3,640,640)-mean)/std).unsqueeze(0)

class Inter(nn.Module):
    def __init__(self, m): super().__init__(); self.m=m
    def forward(self, pv):
        o = self.m(pixel_values=pv, return_dict=True)
        # encoder-stage outputs (pre/at query selection) + final decoder ref points
        return (o.init_reference_points,          # [b,q,4] selected anchors+delta (encoder)
                o.enc_topk_bboxes,                 # [b,q,4] encoder top-k boxes
                o.intermediate_reference_points[-1])  # [b,q,4] final decoder ref (after deform attn)

with torch.no_grad():
    ta = Inter(inner)(norm)
names = ["init_reference_points","enc_topk_bboxes","final_dec_ref"]

out = Path("/tmp/rtdetr_inter"); out.mkdir(exist_ok=True)
p = out/"inter.onnx"
with _float32_position_embedding(), torch.no_grad():
    torch.onnx.export(Inter(inner), (norm,), str(p),
        input_names=["pixel_values"], output_names=names,
        dynamic_axes={"pixel_values":{0:"b",2:"h",3:"w"}},
        opset_version=17, do_constant_folding=True, dynamo=False)

import onnxruntime as ort
sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
oa = sess.run(None, {"pixel_values": norm.numpy()})
for name, t, o in zip(names, ta, oa):
    t = t.numpy()
    print(f"{name:24s} max|diff|={np.abs(t-o).max():.3e}  mean={np.abs(t-o).mean():.3e}  shape={tuple(t.shape)}")
