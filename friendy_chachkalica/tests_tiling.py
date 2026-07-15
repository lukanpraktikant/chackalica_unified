import unittest
from pathlib import Path

import torch

from friendy_chachkalica.config import (
    EvaluationConfig,
    ExperimentConfig,
    PipelineSpec,
    TilingSpec,
    _parse_pipeline,
)
from friendy_chachkalica.tiling import tile_batch
from friendy_chachkalica.train import _predict_with_config, build_run_pipeline


def _target(boxes, labels, size=100):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "orig_size": torch.tensor([size, size], dtype=torch.int64),
    }


# 50% tiles, no overlap -> four 50x50 tiles at offsets (0,0) (50,0) (0,50) (50,50).
_TILING = TilingSpec(tile_width_pct=50.0, tile_height_pct=50.0, overlap=0.0)


class TileBatchTests(unittest.TestCase):
    def test_box_confined_to_one_tile(self):
        img = torch.zeros(3, 100, 100)
        # A box fully inside the top-left tile, and an empty frame that yields nothing.
        images, targets = tile_batch(
            [img, img],
            [_target([[10, 10, 40, 40]], [1]), _target([], [])],
            _TILING,
        )
        # Only the top-left tile has ground truth; every other tile (and the whole
        # empty frame) is all-background and dropped.
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape, (3, 50, 50))
        self.assertTrue(
            torch.equal(targets[0]["boxes"], torch.tensor([[10.0, 10.0, 40.0, 40.0]]))
        )
        self.assertTrue(torch.equal(targets[0]["labels"], torch.tensor([1])))
        self.assertTrue(torch.equal(targets[0]["orig_size"], torch.tensor([50, 50])))

    def test_box_translated_into_bottom_right_tile(self):
        img = torch.zeros(3, 100, 100)
        images, targets = tile_batch([img], [_target([[60, 60, 90, 90]], [2])], _TILING)
        self.assertEqual(len(images), 1)
        # Bottom-right tile starts at (50, 50): [60,60,90,90] -> [10,10,40,40].
        self.assertTrue(
            torch.equal(targets[0]["boxes"], torch.tensor([[10.0, 10.0, 40.0, 40.0]]))
        )

    def test_seam_spanning_box_clipped_into_every_tile(self):
        img = torch.zeros(3, 100, 100)
        # Centered 20x20 box straddles all four tile seams; each tile gets a
        # quarter (area fraction 0.25 >= the 0.1 min-visible threshold).
        images, targets = tile_batch([img], [_target([[40, 40, 60, 60]], [0])], _TILING)
        self.assertEqual(len(images), 4)
        for target in targets:
            self.assertEqual(target["boxes"].shape, (1, 4))

    def test_sliver_below_min_visible_is_dropped(self):
        img = torch.zeros(3, 100, 100)
        # Box mostly in the top-left tile with only a 1px sliver crossing into the
        # top-right tile -> that sliver is below the 10% visibility floor.
        images, targets = tile_batch([img], [_target([[10, 10, 51, 40]], [0])], _TILING)
        # Top-left keeps it; top-right's sliver (1px wide of a 41px box) is dropped.
        self.assertEqual(len(images), 1)


class ParsePipelineTests(unittest.TestCase):
    def test_none_when_absent(self):
        self.assertIsNone(_parse_pipeline(None, Path("/tmp")))

    def test_batch_detect(self):
        spec = _parse_pipeline(
            {"name": "batch_detect", "tiling": {"tile_width_pct": 50, "overlap": 0.2}},
            Path("/tmp"),
        )
        self.assertIsInstance(spec, PipelineSpec)
        self.assertEqual(spec.name, "batch_detect")
        self.assertEqual(spec.tiling.tile_width_pct, 50.0)
        self.assertEqual(spec.tiling.overlap, 0.2)
        self.assertIsNone(spec.detector_checkpoint)

    def test_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            _parse_pipeline({"name": "nope"}, Path("/tmp"))


class _StubAdapter:
    """Minimal adapter: predicts one centered box per input image/tile."""

    def predict(self, images, score_threshold=None):
        out = []
        for _ in images:
            # (cx, cy, w, h, conf, class_id) normalized to the input.
            out.append(torch.tensor([[0.5, 0.5, 0.2, 0.2, 0.9, 0.0]]))
        return out


class RouteThroughPipelineTests(unittest.TestCase):
    def _config(self, pipeline):
        return ExperimentConfig(
            name="t",
            train_datasets=[],
            models=[],
            output_dir=Path("/tmp/out"),
            evaluation=EvaluationConfig(),
            pipeline=pipeline,
        )

    def test_no_pipeline_uses_adapter_predict(self):
        config = self._config(None)
        self.assertIsNone(build_run_pipeline(_StubAdapter(), config, torch.device("cpu")))

    def test_batch_detect_pipeline_routes_process_batch(self):
        spec = PipelineSpec(
            name="batch_detect",
            tiling=TilingSpec(tile_width_pct=50.0, tile_height_pct=50.0, overlap=0.0),
        )
        config = self._config(spec)
        adapter = _StubAdapter()
        pipeline = build_run_pipeline(adapter, config, torch.device("cpu"))
        self.assertIsNotNone(pipeline)

        images = [torch.rand(3, 64, 64)]
        targets = [{"orig_size": torch.tensor([64, 64])}]
        preds = _predict_with_config(adapter, images, config, pipeline=pipeline, targets=targets)
        self.assertEqual(len(preds), 1)
        # Full-frame-normalized (N, 6): tiling four 32x32 tiles then merging yields
        # multiple boxes remapped onto the frame.
        self.assertEqual(preds[0].shape[1], 6)
        self.assertGreater(preds[0].shape[0], 0)


if __name__ == "__main__":
    unittest.main()
