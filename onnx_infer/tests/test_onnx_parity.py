"""End-to-end parity: the torch training adapter vs the ONNX service adapter.

The gate for every architecture. Builds an adapter, exports it, then asserts the
:class:`OnnxAdapter` reproduces the adapter's ``predict`` output (boxes/scores
within tolerance, labels exact) on the same images.

Skips cleanly when torch / torchvision / onnx / onnxruntime aren't installed, so
the pure-numpy tests still run in a minimal environment.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from friendy_chachkalica.registry import build_model  # noqa: E402
from friendy_chachkalica.onnx_export.arch.retinanet import export_retinanet  # noqa: E402
from onnx_infer import load_onnx_adapter  # noqa: E402


def _assert_parity(torch_pred, onnx_pred, min_dets=1, atol=1e-3):
    """Every torch detection has a unique ONNX twin (same label, box+score within
    ``atol``), and counts match. Order-independent: the two runners can emit the
    same set in a different row order, and near-tied scores make row-align
    unreliable, so we greedily nearest-neighbour match instead.
    """
    torch_pred = np.asarray(torch_pred, dtype=np.float32)
    onnx_pred = np.asarray(onnx_pred, dtype=np.float32)
    assert torch_pred.shape[0] == onnx_pred.shape[0], (
        f"detection count differs: torch={torch_pred.shape[0]} onnx={onnx_pred.shape[0]}"
    )
    assert torch_pred.shape[0] >= min_dets, (
        f"trivial comparison: only {torch_pred.shape[0]} detections (expected >= {min_dets})"
    )

    used = set()
    for row in torch_pred:
        best_j, best_d = None, np.inf
        for j, cand in enumerate(onnx_pred):
            if j in used or int(cand[5]) != int(row[5]):
                continue
            dist = np.abs(row[:5] - cand[:5]).sum()  # box (4) + score
            if dist < best_d:
                best_d, best_j = dist, j
        assert best_j is not None, f"no ONNX match for torch detection {row}"
        assert best_d <= atol, f"closest ONNX match off by L1 {best_d:.2e} (> {atol}) for {row}"
        used.add(best_j)


@pytest.fixture(scope="module")
def retinanet_export(tmp_path_factory):
    torch.manual_seed(0)
    adapter = build_model("retinanet", num_classes=3)  # random weights, no download
    # RetinaNet's classification head is prior-init'd to ~0.01 scores, so random
    # weights would emit zero detections above the score floor — a trivial
    # (0 == 0) comparison. Widen the head bias so the model actually fires,
    # giving a real multi-detection set to check parity against.
    torch.nn.init.normal_(adapter.model.head.classification_head.cls_logits.bias, mean=-2.0, std=2.0)
    adapter.eval()
    out_dir = tmp_path_factory.mktemp("retinanet")
    onnx_path = out_dir / "model.onnx"
    meta = export_retinanet(
        adapter, num_classes=3, params={},
        class_map={0: "a", 1: "b", 2: "c"}, onnx_path=onnx_path,
    )
    onnx_path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return adapter, onnx_path


@pytest.mark.parametrize("hw", [(480, 640), (512, 512)])
def test_retinanet_parity(retinanet_export, hw):
    adapter, onnx_path = retinanet_export
    onnx_adapter, info = load_onnx_adapter(onnx_path, "cpu")
    assert info["num_classes"] == 3
    assert info["train_classes"] == {0: "a", 1: "b", 2: "c"}

    torch.manual_seed(1)
    image = torch.rand(3, *hw)
    threshold = 0.05
    torch_pred = adapter.predict([image])[0].detach().cpu().numpy()
    torch_pred = torch_pred[torch_pred[:, 4] >= threshold]  # match the graph's score floor
    onnx_pred = onnx_adapter.predict([image], score_threshold=threshold)[0].detach().cpu().numpy()

    _assert_parity(torch_pred, onnx_pred, min_dets=10)
