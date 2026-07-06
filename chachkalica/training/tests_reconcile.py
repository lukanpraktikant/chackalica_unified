"""Tests for stranded-run recovery (training.services.reconcile).

These exercise reconcile's *routing* — given the trainer service's view of a run,
does it finalize / error / leave-alone correctly — by mocking the trainer client
and the shared finalize helpers, so no real trainer or filesystem is needed.
"""

from unittest import mock

from django.test import TestCase

from training.models import EvalRun, TrainedModel, TrainingRun
from training.services import reconcile
from fleet.models import Dataset


class ReconcileRunTests(TestCase):
    def setUp(self):
        self.run = TrainingRun.objects.create(status=TrainingRun.RUNNING,
                                              output_dir="/tmp/does-not-matter")

    def _fetch(self, payload):
        return mock.patch.object(reconcile.runner, "fetch_status", return_value=payload)

    def test_ok_finalizes(self):
        with self._fetch({"status": "ok"}), \
             mock.patch.object(reconcile.jobs, "finalize_success") as fin:
            self.assertEqual(reconcile.reconcile_run(self.run), "ok")
        fin.assert_called_once_with(self.run)

    def test_unknown_but_complete_on_disk_finalizes(self):
        with self._fetch({"status": "unknown"}), \
             mock.patch.object(reconcile.ingest, "is_complete", return_value=True), \
             mock.patch.object(reconcile.jobs, "finalize_success") as fin:
            self.assertEqual(reconcile.reconcile_run(self.run), "ok")
        fin.assert_called_once()

    def test_error_marks_error_with_log_tail(self):
        with self._fetch({"status": "error", "log_tail": "boom traceback"}):
            self.assertEqual(reconcile.reconcile_run(self.run), "error")
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, TrainingRun.ERROR)
        self.assertIn("boom traceback", self.run.last_error)
        self.assertIsNotNone(self.run.finished_at)

    def test_running_is_left_alone(self):
        with self._fetch({"status": "running"}), \
             mock.patch.object(reconcile.jobs, "finalize_success") as fin:
            self.assertEqual(reconcile.reconcile_run(self.run), "running")
        fin.assert_not_called()
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, TrainingRun.RUNNING)

    def test_queued_run_reported_running_adopts_running(self):
        self.run.status = TrainingRun.QUEUED
        self.run.save(update_fields=["status"])
        with self._fetch({"status": "running"}):
            self.assertEqual(reconcile.reconcile_run(self.run), "running")
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, TrainingRun.RUNNING)

    def test_unknown_incomplete_with_live_job_stays_pending(self):
        with self._fetch({"status": "unknown"}), \
             mock.patch.object(reconcile.ingest, "is_complete", return_value=False), \
             mock.patch.object(reconcile, "_has_live_job", return_value=True):
            self.assertEqual(reconcile.reconcile_run(self.run), "pending")
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, TrainingRun.RUNNING)  # untouched

    def test_unknown_incomplete_without_live_job_is_lost(self):
        with self._fetch({"status": "unknown"}), \
             mock.patch.object(reconcile.ingest, "is_complete", return_value=False), \
             mock.patch.object(reconcile, "_has_live_job", return_value=False):
            self.assertEqual(reconcile.reconcile_run(self.run), "lost")
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, TrainingRun.ERROR)
        self.assertIn("lost", self.run.last_error)

    def test_unreachable_trainer_leaves_run_untouched(self):
        with mock.patch.object(reconcile.runner, "fetch_status",
                               side_effect=RuntimeError("connection refused")):
            self.assertEqual(reconcile.reconcile_run(self.run), "unreachable")
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, TrainingRun.RUNNING)

    def test_reconcile_all_only_scans_active_runs(self):
        TrainingRun.objects.create(status=TrainingRun.OK)      # finished — skip
        TrainingRun.objects.create(status=TrainingRun.ERROR)   # finished — skip
        with mock.patch.object(reconcile, "reconcile_run", return_value="running") as rr, \
             mock.patch.object(reconcile, "reconcile_eval"):
            results = reconcile.reconcile_all()
        # Only the single RUNNING run from setUp is scanned.
        self.assertEqual(rr.call_count, 1)
        self.assertEqual(results["runs"], {self.run.pk: "running"})


class ReconcileEvalTests(TestCase):
    def setUp(self):
        self.model = TrainedModel.objects.create(
            name="m1", arch="yolox", checkpoint_path="/tmp/best.pt",
        )
        self.dataset = Dataset.objects.create(name="ds-eval")
        self.eval = EvalRun.objects.create(
            trained_model=self.model, dataset=self.dataset,
            status=EvalRun.RUNNING, output_dir="/tmp/eval-out",
        )

    def test_ok_finalizes(self):
        with mock.patch.object(reconcile.runner, "fetch_eval_status", return_value={"status": "ok"}), \
             mock.patch.object(reconcile.jobs, "finalize_eval_success") as fin:
            self.assertEqual(reconcile.reconcile_eval(self.eval), "ok")
        fin.assert_called_once_with(self.eval)

    def test_error_marks_error(self):
        with mock.patch.object(reconcile.runner, "fetch_eval_status",
                               return_value={"status": "error", "log_tail": "kaboom"}):
            self.assertEqual(reconcile.reconcile_eval(self.eval), "error")
        self.eval.refresh_from_db()
        self.assertEqual(self.eval.status, EvalRun.ERROR)
        self.assertIn("kaboom", self.eval.last_error)
