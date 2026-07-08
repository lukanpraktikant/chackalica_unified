"""Export a trained checkpoint to ``<name>.onnx`` + ``<name>.meta.json``.

Usage::

    .venv/bin/python -m friendy_chachkalica.onnx_export.cli runs/foo/best.pt
    .venv/bin/python friendy_chachkalica/onnx_export/cli.py runs/foo/best.pt -o out/foo.onnx

Rebuilds the adapter exactly like ``chachak/infer.py`` and ``eval_checkpoint.py``
(``build_model(model_name, num_classes, **params)`` + ``load_state_dict``), then
dispatches to the arch's exporter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Union

import torch

try:
    from ..registry import build_model
    from .registry import get_exporter
except ImportError:  # run as a flat script
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from registry import build_model  # type: ignore
    from onnx_export.registry import get_exporter  # type: ignore


def _as_class_map(classes: Union[Dict, List, None]) -> Dict[int, str]:
    if classes is None:
        return {}
    if isinstance(classes, dict):
        return {int(k): str(v) for k, v in classes.items()}
    return {index: str(name) for index, name in enumerate(classes)}


def export_checkpoint(
    checkpoint_path: Union[str, Path],
    onnx_path: Union[str, Path, None] = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    onnx_path = Path(onnx_path) if onnx_path else checkpoint_path.with_suffix(".onnx")
    meta_path = onnx_path.with_suffix(".meta.json")

    print(f"[export] Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    model_name = state["model_name"]
    model_config = state.get("model_config", {}) or {}
    num_classes = model_config.get("num_classes")
    params = dict(model_config.get("params", {}) or {})
    class_map = _as_class_map((state.get("train_dataset") or {}).get("classes"))

    adapter = build_model(model_name, num_classes=num_classes, **params)
    adapter.model.load_state_dict(state["model_state_dict"])
    adapter.eval()
    print(f"[export] Rebuilt adapter: {model_name} num_classes={num_classes} classes={len(class_map)}")

    exporter = get_exporter(model_name)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    meta = exporter(
        adapter,
        num_classes=num_classes,
        params=params,
        class_map=class_map,
        onnx_path=onnx_path,
    )

    with open(meta_path, "w") as file:
        json.dump(meta, file, indent=2)
    print(f"[export] Wrote {onnx_path}")
    print(f"[export] Wrote {meta_path}")
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Friendy checkpoint to ONNX + meta.json")
    parser.add_argument("checkpoint", help="Path to a checkpoint, e.g. runs/foo/best.pt")
    parser.add_argument("--output", "-o", help="ONNX output path (default: checkpoint with .onnx)")
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
