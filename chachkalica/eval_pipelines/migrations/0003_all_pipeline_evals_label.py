from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('eval_pipelines', '0002_pipeline_proxies'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='pipelineevalrun',
            options={
                'ordering': ['-created_at'],
                'verbose_name': 'All pipeline eval',
                'verbose_name_plural': 'All pipeline evals',
            },
        ),
    ]
