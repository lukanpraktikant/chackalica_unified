from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('eval_pipelines', '0003_all_pipeline_evals_label'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='pipelineevalrun',
            name='tile_size',
        ),
        migrations.AddField(
            model_name='pipelineevalrun',
            name='tile_width_pct',
            field=models.FloatField(
                blank=True,
                null=True,
                help_text="Tile width as a percent (0–100] of each image's width.",
            ),
        ),
        migrations.AddField(
            model_name='pipelineevalrun',
            name='tile_height_pct',
            field=models.FloatField(
                blank=True,
                null=True,
                help_text="Tile height as a percent (0–100] of each image's height.",
            ),
        ),
    ]
