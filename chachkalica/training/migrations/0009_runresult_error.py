from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("training", "0008_experimentdataset_augmentations"),
    ]

    operations = [
        migrations.AddField(
            model_name="runresult",
            name="error",
            field=models.TextField(blank=True, default=""),
        ),
    ]
