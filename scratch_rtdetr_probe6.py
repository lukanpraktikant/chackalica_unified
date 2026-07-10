"""Probe: (a) characterize the legacy-export divergence; (b) try dynamo export."""
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
model = adapter.model
mean = torch.tensor(adapter.image_mean).view(3,1,1); std = torch.tensor(adapter.image_std).view(3,1,1)
torch.manual_seed(1)
norm = ((torch.rand(3,640,640)-mean)/std).unsqueeze(0)

class Raw(nn.Module):
    def __init__(self, m): super().__init__(); self.m=m
    def forward(self, pv):
        o=self.m(pixel_values=pv); return o.logits[0], o.pred_boxes[0]

with torch.no_grad():
    lt, bt = Raw(model)(norm)
lt, bt = lt.numpy(), bt.numpy()

import onnxruntime as ort
out = Path("/tmp/rtdetr_dyn"); out.mkdir(exist_ok=True)

def parity(path, tag):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    lo, bo = sess.run(None, {sess.get_inputs()[0].name: norm.numpy()})
    dl = np.abs(lt-lo); db = np.abs(bt-bo)
    print(f"[{tag}] |logit| max={dl.max():.2e} mean={dl.mean():.2e}  frac>1e-2={np.mean(dl>1e-2):.2f}"
          f"   |box| max={db.max():.2e} mean={db.mean():.2e}")

# --- dynamo export attempt ---
try:
    with _float32_position_embedding(), torch.no_grad():
        torch.onnx.export(
            Raw(model), (norm,), str(out/"dyn.onnx"),
            input_names=["pixel_values"], output_names=["logits","pred_boxes"],
            dynamic_axes={"pixel_values":{0:"b",2:"h",3:"w"},"logits":{0:"q"},"pred_boxes":{0:"q"}},
            opset_version=18, dynamo=True,
        )
    print("dynamo export OK")
    parity(out/"dyn.onnx", "dynamo op18")
except Exception as e:
    print("dynamo export FAILED:", repr(e)[:300])
