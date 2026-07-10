"""Probe: find a fixture surgery that yields a sparse, well-separated RT-DETR
detection set (large score gaps) so topk selection is deterministic."""
import sys
import numpy as np
import torch
sys.path.insert(0, "/app")
from friendy_chachkalica.registry import build_model

torch.manual_seed(0)
adapter = build_model("rtdetr", num_classes=3, weights=None)
adapter.eval()
model = adapter.model

HEAD = model.model.decoder.class_embed[-1]
print("head:", HEAD)

def score_stats(tag):
    torch.manual_seed(1)
    image = torch.rand(3, 512, 512)
    mean = torch.tensor(adapter.image_mean).view(3,1,1)
    std = torch.tensor(adapter.image_std).view(3,1,1)
    norm = ((image - mean)/std).unsqueeze(0)
    with torch.no_grad():
        o = model(pixel_values=norm)
    s = torch.sigmoid(o.logits[0]).flatten()
    top, _ = torch.topk(s, 300)
    top = top.numpy()
    n = int((top>=0.5).sum())
    gaps = np.abs(np.diff(np.sort(top[top>=0.5])[::-1])) if n>1 else np.array([0])
    print(f"{tag}: >=0.5 count={n:3d}  max={top.max():.4f}  min_gap={gaps.min():.2e}  top5={top[:5].round(3)}")

score_stats("baseline")

# Surgery: spread logits by scaling the last decoder-layer class head weight up and
# pushing bias negative so only a few queries poke high -> big gaps, sparse set.
with torch.no_grad():
    last = model.model.decoder.class_embed[-1]
    torch.nn.init.normal_(last.weight, mean=0.0, std=3.0)
    torch.nn.init.constant_(last.bias, -6.0)
score_stats("scaled w std3 bias-6")

with torch.no_grad():
    last = model.model.decoder.class_embed[-1]
    torch.nn.init.normal_(last.weight, mean=0.0, std=5.0)
    torch.nn.init.constant_(last.bias, -8.0)
score_stats("scaled w std5 bias-8")
