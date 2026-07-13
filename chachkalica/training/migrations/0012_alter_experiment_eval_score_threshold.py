from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("training", "0011_experiment_best_metric_experiment_val_interval_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="experiment",
            name="eval_score_threshold",
            field=models.FloatField(
                default=0.25,
                help_text=(
                    "Operating-point confidence threshold used for precision, recall, and F1 "
                    "during per-epoch validation. map50/map50_95 are unaffected — they're always "
                    "computed at a fixed low threshold (0.001) so mAP can sweep the full "
                    "precision-recall curve."
                ),
            ),
        ),
    ]
