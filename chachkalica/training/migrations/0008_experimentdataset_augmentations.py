from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("training", "0007_experiment_early_stopping_patience_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="experimentdataset",
            name="aug_hflip",
            field=models.BooleanField(
                default=False,
                help_text="Randomly mirror images (and their boxes) horizontally during "
                          "training. Applied on the fly, replacing the original for that "
                          "epoch — the dataset does not get bigger. Train datasets only.",
                verbose_name="hflip",
            ),
        ),
        migrations.AddField(
            model_name="experimentdataset",
            name="aug_hflip_fraction",
            field=models.FloatField(
                default=0.5,
                help_text="Chance (0-1) each image is flipped in a given epoch; 0.5 means "
                          "about half the images, a different random half every epoch.",
                verbose_name="hflip fraction",
            ),
        ),
        migrations.AddField(
            model_name="experimentdataset",
            name="aug_scale_crop",
            field=models.BooleanField(
                default=False,
                help_text="Randomly crop a 60-100% window and scale it back up; boxes are "
                          "clipped/dropped at crop edges. Applied on the fly, replacing the "
                          "original for that epoch — the dataset does not get bigger. "
                          "Train datasets only.",
                verbose_name="scale+crop",
            ),
        ),
        migrations.AddField(
            model_name="experimentdataset",
            name="aug_scale_crop_fraction",
            field=models.FloatField(
                default=0.5,
                help_text="Chance (0-1) each image is scale-cropped in a given epoch; 0.5 "
                          "means about half the images, a different random half every epoch.",
                verbose_name="scale+crop fraction",
            ),
        ),
    ]
