from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("training", "0009_runresult_error"),
    ]

    operations = [
        migrations.AddField(
            model_name="evalrun",
            name="map_score_threshold",
            field=models.FloatField(
                default=0.001,
                help_text=(
                    "Minimum prediction confidence kept for AP/mAP. Keep this low so mAP can "
                    "sweep the precision-recall curve."
                ),
            ),
        ),
        migrations.AddField(
            model_name="evalrun",
            name="score_threshold",
            field=models.FloatField(
                default=0.25,
                help_text=(
                    "Operating-point confidence threshold used for precision, recall, F1, "
                    "prediction counts, and the confusion matrix."
                ),
            ),
        ),
    ]
