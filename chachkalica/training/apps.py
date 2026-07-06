from django.apps import AppConfig


class TrainingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "training"
    verbose_name = "Training & Eval"

    def ready(self):
        # Register post_delete handlers that clean up on-disk run artifacts.
        from training import signals  # noqa: F401
