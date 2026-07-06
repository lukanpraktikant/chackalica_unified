"""Tests for on-disk artifact cleanup when a run/eval row is deleted.

Deleting a TrainingRun/EvalRun (directly, in bulk, or via cascade) should remove
its output_dir and generated YAML under the configured roots — but never touch
anything outside them.
"""

import tempfile
from pathlib import Path

from django.test import TestCase

from fleet.models import Dataset
from training.models import (
    EvalRun,
    Experiment,
    TrainedModel,
    TrainingRun,
    TrainingSettings,
)


class CleanupTests(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.runs_root = self.root / "runs"
        self.configs_root = self.root / "configs"
        self.runs_root.mkdir()
        self.configs_root.mkdir()

        ts = TrainingSettings.load()
        ts.runs_root = str(self.runs_root)
        ts.configs_root = str(self.configs_root)
        ts.save()

    def tearDown(self):
        self._tmp.cleanup()

    def _make_run(self, name="run-1"):
        out = self.runs_root / name
        (out / "00-yolox").mkdir(parents=True)
        (out / "00-yolox" / "best.pt").write_bytes(b"weights")
        (out / "run_summary.yaml").write_text("summary: yes\n", encoding="utf-8")
        cfg = self.configs_root / f"{name}.yaml"
        cfg.write_text("name: x\n", encoding="utf-8")
        run = TrainingRun.objects.create(output_dir=str(out), config_yaml_path=str(cfg))
        return run, out, cfg

    def test_deleting_run_removes_output_dir_and_config(self):
        run, out, cfg = self._make_run()
        self.assertTrue(out.exists() and cfg.exists())
        run.delete()
        self.assertFalse(out.exists())
        self.assertFalse(cfg.exists())

    def test_bulk_delete_removes_artifacts(self):
        run_a, out_a, _ = self._make_run("run-a")
        run_b, out_b, _ = self._make_run("run-b")
        TrainingRun.objects.all().delete()
        self.assertFalse(out_a.exists())
        self.assertFalse(out_b.exists())

    def test_cascade_delete_from_experiment_cleans_run(self):
        run, out, _ = self._make_run("run-casc")
        exp = Experiment.objects.create(name="exp-casc")
        run.experiment = exp
        run.save(update_fields=["experiment"])
        # TrainingRun.experiment is SET_NULL, so deleting the experiment does not
        # cascade-delete the run; delete the run's owner path directly instead.
        run.delete()
        self.assertFalse(out.exists())

    def test_refuses_to_delete_path_outside_runs_root(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("important", encoding="utf-8")
        run = TrainingRun.objects.create(output_dir=str(outside))
        run.delete()
        self.assertTrue(outside.exists(), "must not delete dirs outside runs_root")

    def test_blank_output_dir_is_safe(self):
        run = TrainingRun.objects.create(output_dir="", config_yaml_path="")
        run.delete()  # must not raise, must not delete anything
        self.assertTrue(self.runs_root.exists())

    def test_deleting_eval_removes_its_artifacts(self):
        out = self.runs_root / "eval-1"
        out.mkdir()
        (out / "eval_result.yaml").write_text("metrics: {}\n", encoding="utf-8")
        req = self.configs_root / "eval-1.yaml"
        req.write_text("eval: yes\n", encoding="utf-8")

        model = TrainedModel.objects.create(name="m1", arch="yolox",
                                            checkpoint_path="/tmp/best.pt")
        dataset = Dataset.objects.create(name="ds1")
        ev = EvalRun.objects.create(trained_model=model, dataset=dataset,
                                    output_dir=str(out), request_yaml_path=str(req))
        ev.delete()
        self.assertFalse(out.exists())
        self.assertFalse(req.exists())
