"""Probe 2: does pixel_mask change RT-DETR output? And where is center_to_corners?"""
import sys, torch
import torch.nn.functional as F
sys.path.insert(0, "/app")
from friendy_chachkalica.registry import build_model

torch.manual_seed(0)
adapter = build_model("rtdetr", num_classes=3, weights=None)
adapter.eval()
model = adapter.model

# padded input (content 512x512 -> pad to 544x544), with vs without a content mask
base = torch.rand(3, 512, 512)
padded = F.pad(base, (0, 32, 0, 32), value=0.0).unsqueeze(0)  # [1,3,544,544]
mask = torch.zeros(1, 544, 544, dtype=torch.long)
mask[:, :512, :512] = 1

with torch.no_grad():
    o_nomask = model(pixel_values=padded)
    o_mask = model(pixel_values=padded, pixel_mask=mask)

print("mask vs no-mask pred_boxes max abs diff:", (o_nomask.pred_boxes - o_mask.pred_boxes).abs().max().item())
print("mask vs no-mask logits    max abs diff:", (o_nomask.logits - o_mask.logits).abs().max().item())

# confirm center_to_corners location
try:
    from transformers.models.rt_detr.image_processing_rt_detr import center_to_corners_format
    print("center_to_corners_format importable from image_processing_rt_detr")
except Exception as e:
    print("import err:", repr(e)[:120])
