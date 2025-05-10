import threading
import time
import os
from datetime import timedelta
from django.utils.timezone import now
from asl.models import ASLVideoHistory

def delete_expired_videos():
    while True:
        threshold = now() - timedelta(minutes=1)
        expired_videos = ASLVideoHistory.objects.filter(created_at__lt=threshold, is_deleted=False)

        for video in expired_videos:
            file_path = video.absolute_file_path
            if os.path.exists(file_path):
                os.remove(file_path)

            video.is_deleted = True
            video.deleted_at = now()
            video.save()

        time.sleep(300)  # Sleep 5 minutes before next run
