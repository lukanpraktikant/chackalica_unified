"""Tests for build_pipeline and the Detector person-class filtering.

Run from the chachak directory:  python -m unittest tests.test_registry
"""

import sys
import unittest
from pathlib import Path

_CHACHAK_DIR = Path(__file__).resolve().parent.parent
if str(_CHACHAK_DIR) not in sys.path:
    sys.path.insert(0, str(_CHACHAK_DIR))

import torch  # noqa: E402

from config import DetectorConfig, PipelineConfig, TilingConfig  # noqa: E402
from detector import Detector  # noqa: E402
from pipeline import (  # noqa: E402
    BatchDetectPipeline,
    BatchPeoplePipeline,
    ChainedPipeline,
    PeopleDetectFirstPipeline,
)
from registry import build_pipeline


def make_config(pipeline, chain=None):
    return PipelineConfig(
        name="t",
        pipeline=pipeline,
        model_checkpoint=Path("/dev/null"),
        images=Path("/dev/null"),
        labels=Path("/dev/null"),
        classes={0: "a"},
        output_dir=Path("/tmp/out"),
        tiling=TilingConfig(),
        detector=DetectorConfig(),
        chain=chain or [],
    )


class BuildPipelineTest(unittest.TestCase):
    def test_builds_each_concrete_pipeline(self):
        self.assertIsInstance(
            build_pipeline(make_config("batch_detect"), None, "cpu"), BatchDetectPipeline
        )
        self.assertIsInstance(
            build_pipeline(make_config("people_detect_first"), None, "cpu"),
            PeopleDetectFirstPipeline,
        )
        self.assertIsInstance(
            build_pipeline(make_config("batch_people"), None, "cpu"), BatchPeoplePipeline
        )

    def test_chain_builds_children(self):
        config = make_config("chain", chain=["batch_detect", "people_detect_first"])
        pipe = build_pipeline(config, None, "cpu")
        self.assertIsInstance(pipe, ChainedPipeline)
        self.assertEqual(len(pipe.pipelines), 2)
        self.assertIsInstance(pipe.pipelines[0], BatchDetectPipeline)
        self.assertIsInstance(pipe.pipelines[1], PeopleDetectFirstPipeline)


class StubDetectorAdapter:
    """Returns a fixed prediction set with mixed classes and scores."""

    def predict(self, images, score_threshold=None):
        preds = torch.tensor(
            [
                [0.5, 0.5, 0.2, 0.2, 0.90, 1.0],  # person, high score  -> keep
                [0.3, 0.3, 0.2, 0.2, 0.40, 1.0],  # person, low score   -> drop
                [0.7, 0.7, 0.2, 0.2, 0.95, 0.0],  # non-person          -> drop
            ]
        )
        return [preds for _ in images]


class DetectorFilterTest(unittest.TestCase):
    def test_keeps_only_person_above_threshold(self):
        det = Detector(StubDetectorAdapter(), person_class_id=1, score_threshold=0.5, batch_size=4)
        out = det.predict([torch.rand(3, 32, 32)])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].shape[0], 1)  # only the high-score person survives
        self.assertAlmostEqual(float(out[0][0, 4]), 0.90)
        self.assertEqual(int(out[0][0, 5]), 1)

    def test_no_person_yields_empty(self):
        det = Detector(StubDetectorAdapter(), person_class_id=5, score_threshold=0.5, batch_size=4)
        out = det.predict([torch.rand(3, 32, 32), torch.rand(3, 32, 32)])
        self.assertEqual(len(out), 2)
        self.assertTrue(all(p.shape == (0, 6) for p in out))


if __name__ == "__main__":
    unittest.main()
