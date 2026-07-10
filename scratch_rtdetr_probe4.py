"""Probe: raw forward parity torch-vs-ONNX for RT-DETR (no topk), across sizes.

Exports a bare graph emitting per-query (logits, pred_boxes) and compares to the
torch model on an identical normalized (+ padded) tensor. This measures export
numerical fidelity directly, free of topk tie instability.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, "/app")
from friendy_chachkalica.registry import build_model
from friendy_chachkalica.onnx_export.arch.rtdetr import _float32_position_embedding

torch.manual_seed(0)
adapter = build_model("rtdetr", num_classes=3, weights=None)
adapter.eval()
model = adapter.model


class Raw(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, pixel_values):
        o = self.m(pixel_values=pixel_values)
        return o.logits[0], o.pred_boxes[0]


out = Path("/tmp/rtdetr_raw"); out.mkdir(exist_ok=True)
onnx_path = out / "raw.onnx"
dummy = torch.rand(1, 3, 640, 640)
with _float32_position_embedding(), torch.no_grad():
    torch.onnx.export(
        Raw(model), (dummy,), str(onnx_path),
        input_names=["pixel_values"], output_names=["logits", "pred_boxes"],
        dynamic_axes={"pixel_values": {0: "b", 2: "h", 3: "w"},
                      "logits": {0: "q"}, "pred_boxes": {0: "q"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )

import onnxruntime as ort
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

mean = torch.tensor(adapter.image_mean).view(3,1,1)
std = torch.tensor(adapter.image_std).view(3,1,1)

def ceil32(v): return ((v + 31)//32)*32

for hw in [(640,640),(512,512),(480,640),(704,512)]:
    torch.manual_seed(1)
    image = torch.rand(3, *hw)
    # replicate adapter preprocessing: longest-side resize to 640, normalize, pad to 32
    img = adapter._resize_image(image)
    norm = (img - mean)/std
    H, W = norm.shape[-2:]
    ph, pw = ceil32(H)-H, ceil32(W)-W
    norm = F.pad(norm, (0, pw, 0, ph)).unsqueeze(0)
    with torch.no_grad():
        lt, bt = Raw(model)(norm)
    lo, bo = sess.run(None, {"pixel_values": norm.numpy()})
    dl = np.abs(lt.numpy()-lo).max()
    db = np.abs(bt.numpy()-bo).max()
    # sigmoid-score max diff (what actually matters post-threshold)
    ds = np.abs(torch.sigmoid(lt).numpy() - 1/(1+np.exp(-lo))).max()
    print(f"hw={hw} tensor={tuple(norm.shape[-2:])}  |logit|={dl:.2e}  |score|={ds:.2e}  |box|={db:.2e}")
