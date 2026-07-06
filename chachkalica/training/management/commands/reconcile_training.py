"""Recover training/eval runs stranded at running/queued.

Re-derives each stuck run's true state from the trainer service and the shared
filesystem and finalizes the row (ingest + mark OK, or mark error/lost). Safe to
run repeatedly and while runs are genuinely training. Cron this for hands-off
recovery, e.g. hourly::

    */60 * * * *  python manage.py reconcile_training
"""

from django.core.management.base import BaseCommand

from training.services import reconcile


class Command(BaseCommand):
    help = "Recover training/eval runs stranded at running/queued."

    def handle(self, *args, **options):
        results = reconcile.reconcile_all()
        runs, evals = results["runs"], results["evals"]
        if not runs and not evals:
            self.stdout.write("No runs or evals were stuck at running/queued.")
            return
        for pk, outcome in runs.items():
            self.stdout.write(f"TrainingRun #{pk}: {outcome}")
        for pk, outcome in evals.items():
            self.stdout.write(f"EvalRun #{pk}: {outcome}")
        self.stdout.write(self.style.SUCCESS(
            f"Reconciled {len(runs)} run(s) and {len(evals)} eval(s)."
        ))
