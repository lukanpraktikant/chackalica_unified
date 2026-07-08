from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('eval_pipelines', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='BatchDetectEval',
            fields=[],
            options={
                'verbose_name': 'Batch detect eval',
                'verbose_name_plural': 'Batch detect',
                'proxy': True,
                'indexes': [],
                'constraints': [],
            },
            bases=('eval_pipelines.pipelineevalrun',),
        ),
        migrations.CreateModel(
            name='PeopleDetectFirstEval',
            fields=[],
            options={
                'verbose_name': 'People detect first eval',
                'verbose_name_plural': 'People detect first',
                'proxy': True,
                'indexes': [],
                'constraints': [],
            },
            bases=('eval_pipelines.pipelineevalrun',),
        ),
        migrations.CreateModel(
            name='BatchPeopleEval',
            fields=[],
            options={
                'verbose_name': 'Batch people eval',
                'verbose_name_plural': 'Batch people',
                'proxy': True,
                'indexes': [],
                'constraints': [],
            },
            bases=('eval_pipelines.pipelineevalrun',),
        ),
        migrations.CreateModel(
            name='ChainEval',
            fields=[],
            options={
                'verbose_name': 'Chain eval',
                'verbose_name_plural': 'Chain',
                'proxy': True,
                'indexes': [],
                'constraints': [],
            },
            bases=('eval_pipelines.pipelineevalrun',),
        ),
    ]
