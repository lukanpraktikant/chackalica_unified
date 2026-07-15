from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('training', '0015_experiment_pipeline'),
    ]

    operations = [
        migrations.AlterField(
            model_name='trainingrun',
            name='status',
            field=models.CharField(choices=[('created', 'created'), ('queued', 'queued'), ('running', 'running'), ('paused', 'paused'), ('ok', 'ok'), ('error', 'error')], default='created', max_length=16),
        ),
    ]
