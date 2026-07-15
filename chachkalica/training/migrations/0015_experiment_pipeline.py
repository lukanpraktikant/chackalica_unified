from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('training', '0014_alter_experimentmodel_arch'),
    ]

    operations = [
        migrations.AddField(
            model_name='experiment',
            name='pipeline',
            field=models.CharField(blank=True, choices=[('batch_detect', 'batch_detect — tile, detect per tile, merge'), ('people_detect_first', 'people_detect_first — detect people, crop, detect per crop'), ('batch_people', 'batch_people — tile, detect people, crop, detect per crop'), ('chain', 'chain — run several pipelines and merge')], help_text="Run train (tiling only), val, and test through this chachak pipeline. Blank = plain full-frame training and eval.", max_length=32),
        ),
        migrations.AddField(
            model_name='experiment',
            name='detector_checkpoint',
            field=models.CharField(blank=True, help_text='Person-detector checkpoint; required for people_detect_first / batch_people (and any chain that includes them). Used for val/test only.', max_length=1024),
        ),
        migrations.AddField(
            model_name='experiment',
            name='tile_width_pct',
            field=models.FloatField(blank=True, help_text="Tile width as a percent (0–100] of each image's width. Blank = chachak's default.", null=True),
        ),
        migrations.AddField(
            model_name='experiment',
            name='tile_height_pct',
            field=models.FloatField(blank=True, help_text="Tile height as a percent (0–100] of each image's height. Blank = chachak's default.", null=True),
        ),
        migrations.AddField(
            model_name='experiment',
            name='overlap',
            field=models.FloatField(blank=True, help_text='Fraction (0–1) by which adjacent tiles overlap. Blank = default.', null=True),
        ),
        migrations.AddField(
            model_name='experiment',
            name='merge_nms_iou',
            field=models.FloatField(blank=True, help_text="Class-aware NMS IoU used to merge predictions across tiles/crops. Blank = chachak's default.", null=True),
        ),
        migrations.AddField(
            model_name='experiment',
            name='chain',
            field=models.JSONField(blank=True, default=list, help_text="Ordered pipeline names for the 'chain' pipeline."),
        ),
    ]
