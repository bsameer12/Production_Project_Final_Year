from django.db import models
from django.contrib.auth.models import User
import os

class ASLPrediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    input_sequence = models.JSONField()  # stores the raw input landmark sequence
    top1_label = models.CharField(max_length=20)
    top1_confidence = models.FloatField()
    top2_label = models.CharField(max_length=20)
    top2_confidence = models.FloatField()
    top3_label = models.CharField(max_length=20)
    top3_confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.top1_label} ({self.top1_confidence})"


class AuditLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    action = models.CharField(max_length=255)  # e.g., 'Prediction', 'Login', 'Logout'
    description = models.TextField()           # e.g., 'Predicted sign A with 0.98 confidence'
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user} - {self.action} at {self.timestamp}"



class ASLSentenceGeneration(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    predictions = models.JSONField()  # e.g., ['H', 'E', 'L', 'L', 'O']
    generated_sentence = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.generated_sentence[:30]}..."


class ASLVideoHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    input_text = models.TextField()
    frame_count = models.IntegerField()
    video_size_kb = models.FloatField()
    video_duration_sec = models.FloatField()
    video_name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    video_url = models.URLField()

    # Optional additions
    is_deleted = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.user.username} - {self.video_name}"

    @property
    def absolute_file_path(self):
        from django.conf import settings
        return os.path.join(settings.MEDIA_ROOT, 'asl_videos', self.video_name)

