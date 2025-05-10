from django.apps import AppConfig
import threading


class AslConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "asl"

    def ready(self):
        import asl.signals
        from .tasks import delete_expired_videos
        threading.Thread(target=delete_expired_videos, daemon=True).start()