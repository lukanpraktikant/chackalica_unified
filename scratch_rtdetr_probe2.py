"""Probe: isolate torch-vs-ONNX RT-DETR divergence on the no-resize 512 case.

Compares (a) raw model forward parity on an identical normalized tensor, and
(b) the final friendy detection sets, printed sorted by score, to distinguish a
genuine forward-pass divergence from topk tie-instability on a random-init model.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/app")
from friendy_chachkalica.registry import build_model
from friendy_chachkalica.onnx_export.arch.rtdetr import export_rtdetr
from onnx_infer import load_onnx_adapter

torch.manual_seed(0)
adapter = build_model("rtdetr", num_classes=3, weights=None)
adapter.eval()

out = Path("/tmp/rtdetr_probe"); out.mkdir(exist_ok=True)
onnx_path = out / "model.onnx"
meta = export_rtdetr(adapter, num_classes=3, params={}, class_map={0:"a",1:"b",2:"c"}, onnx_path=onnx_path)
onnx_path.with_suffix(".meta.json").write_text(json.dumps(meta))

torch.manual_seed(1)
image = torch.rand(3, 512, 512)

# ---- raw forward parity on identical normalized tensor ----
mean = torch.tensor(adapter.image_mean).view(3,1,1)
std = torch.tensor(adapter.image_std).view(3,1,1)
norm = ((image - mean) / std).unsqueeze(0)  # [1,3,512,512], mult of 32 -> no pad

with torch.no_grad():
    o = adapter.model(pixel_values=norm)
logits_t = o.logits[0].numpy()      # [Q,C]
boxes_t = o.pred_boxes[0].numpy()   # [Q,4] cxcywh norm

import onnxruntime as ort
# raw ORT of full baked graph (post-topk); compare final sets instead for raw.
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
ob, os_, ol = sess.run(None, {"pixel_values": norm.numpy()})
print("ORT raw shapes:", ob.shape, os_.shape, ol.shape)

# Replicate torch topk to get torch's (box,score,label) pre-threshold, compare to ORT graph out.
scores = torch.sigmoid(torch.tensor(logits_t))
Q, C = logits_t.shape
top_scores, top_idx = torch.topk(scores.flatten(), Q)
labels = (top_idx % C).numpy()
box_idx = (top_idx // C).numpy()
cx,cy,w,h = [boxes_t[box_idx][:,k] for k in range(4)]
xyxy_t = np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1)
top_scores = top_scores.numpy()

print("\n-- top-8 torch (score, label, xyxy) --")
for i in range(8):
    print(f"{top_scores[i]:.4f} L{labels[i]}  {xyxy_t[i].round(4)}")
print("\n-- top-8 ORT (score, label, xyxy) --")
order = np.argsort(-os_)
for i in order[:8]:
    print(f"{os_[i]:.4f} L{ol[i]}  {ob[i].round(4)}")

# Direct box parity: ORT graph should equal torch topk box-for-box (same order).
db = np.abs(ob - xyxy_t).max()
ds = np.abs(os_ - top_scores).max()
print(f"\nmax |box diff| (graph vs torch-topk, same order): {db:.3e}")
print(f"max |score diff|: {ds:.3e}")
print("labels equal:", bool((ol == labels).all()))

# How many scores are within 1e-3 of the 0.5 threshold neighborhood / tie density
n_above = int((top_scores >= 0.5).sum())
print(f"\ndetections >=0.5: {n_above}")
gaps = np.diff(np.sort(top_scores[top_scores>=0.5])[::-1])
print("min score gap among >=0.5 dets:", float(np.abs(gaps).min()) if len(gaps) else "n/a")
