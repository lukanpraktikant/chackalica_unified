from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("training", "0007_experiment_early_stopping_patience_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="runresult",
            name="error",
            field=models.TextField(blank=True, default=""),
        ),
    ]
