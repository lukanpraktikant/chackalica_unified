"""Tests for chachak pipeline evals: request generation and metric ingest."""

import tempfile
from pathlib import Path

import yaml
from django.test import TestCase

from fleet.models import Dataset, FleetSettings
from training.models import Experiment, RunResult, TrainingRun
from training.services import config_gen, ingest, promote

from eval_pipelines.models import PipelineEvalRun


def _make_dataset_on_disk(source_root: Path, name: str, classes: list[str]) -> None:
    ds = source_root / name
    (ds / "images").mkdir(parents=True)
    (ds / "images" / "img1.jpg").write_bytes(b"")
    (ds / "labels").mkdir()
    (ds / "labels" / "img1.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (ds / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")


class PipelineRequestTests(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.source = root / "source"
        self.source.mkdir()
        fs = FleetSettings.load()
        fs.source_dir = str(self.source)
        fs.target_dir = str(root / "target")
        fs.save()

        _make_dataset_on_disk(self.source, "ds1", ["helmet", "head", "vest"])
        self.ds1 = Dataset.objects.create(name="ds1")

        exp = Experiment.objects.create(name="exp1")
        run = TrainingRun.objects.create(experiment=exp, output_dir=str(root / "out"))
        rr = RunResult.objects.create(
            run=run, run_name="00-ds1-00-retinanet", model_arch="retinanet",
            train_dataset_name="ds1", best_epoch=5, best_loss=0.3,
            best_checkpoint=str(root / "out/run0/best.pt"),
            last_checkpoint=str(root / "out/run0/last.pt"),
            test_metrics={"map50": 0.8, "map50_95": 0.5},
        )
        self.tm = promote.promote_run_result(rr, name="m1")

    def tearDown(self):
        self._tmp.cleanup()

    def _make(self, **kwargs):
        return PipelineEvalRun.objects.create(
            trained_model=self.tm, dataset=self.ds1,
            label_source=PipelineEvalRun.SOURCE, **kwargs,
        )

    def test_build_request_batch_detect(self):
        pe = self._make(pipeline=PipelineEvalRun.BATCH_DETECT)
        req = config_gen.build_pipeline_request(pe, "/out/pipeline-1")
        self.assertEqual(req["pipeline"], "batch_detect")
        self.assertEqual(req["model_checkpoint"], self.tm.checkpoint_path)
        self.assertTrue(req["images"].endswith("ds1/images"))
        self.assertTrue(req["labels"].endswith("ds1/labels"))
        self.assertEqual(req["classes"], ["helmet", "head", "vest"])
        self.assertEqual(req["output_dir"], "/out/pipeline-1")
        self.assertNotIn("detector", req)
        self.assertNotIn("tiling", req)

    def test_detector_pipeline_requires_checkpoint(self):
        pe = self._make(pipeline=PipelineEvalRun.PEOPLE_DETECT_FIRST)
        with self.assertRaises(ValueError):
            config_gen.build_pipeline_request(pe, "/out/pipeline-2")

    def test_detector_and_tiling_emitted(self):
        pe = self._make(
            pipeline=PipelineEvalRun.BATCH_PEOPLE,
            detector_checkpoint="/models/person.pt",
            tile_width_pct=25, tile_height_pct=40, overlap=0.25,
        )
        req = config_gen.build_pipeline_request(pe, "/out/pipeline-3")
        self.assertEqual(req["detector"], {"checkpoint": "/models/person.pt"})
        self.assertEqual(
            req["tiling"],
            {"tile_width_pct": 25, "tile_height_pct": 40, "overlap": 0.25},
        )

    def test_chain_pipeline_carries_children(self):
        pe = self._make(
            pipeline=PipelineEvalRun.CHAIN,
            chain=[PipelineEvalRun.BATCH_DETECT],
        )
        req = config_gen.build_pipeline_request(pe, "/out/pipeline-4")
        self.assertEqual(req["chain"], ["batch_detect"])

    def test_write_request_persists_paths(self):
        pe = self._make(pipeline=PipelineEvalRun.BATCH_DETECT)
        request_path, text = config_gen.write_pipeline_request(pe)
        pe.refresh_from_db()
        self.assertEqual(pe.request_yaml_path, str(request_path))
        self.assertTrue(pe.output_dir)
        self.assertTrue(Path(request_path).exists())
        self.assertEqual(yaml.safe_load(text)["pipeline"], "batch_detect")

    def test_ingest_pipeline_eval(self):
        out = Path(self._tmp.name) / "pipeout"
        out.mkdir()
        pe = self._make(pipeline=PipelineEvalRun.BATCH_DETECT, output_dir=str(out))
        with open(out / "result.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump({"metrics": {"map50": 0.66, "map50_95": 0.44}}, fh)

        self.assertTrue(ingest.pipeline_is_complete(out))
        ingest.ingest_pipeline_eval(pe)
        pe.refresh_from_db()
        self.assertEqual(pe.metric("map50_95"), 0.44)
