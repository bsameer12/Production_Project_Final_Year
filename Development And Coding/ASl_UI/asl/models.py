from django.db import models
from django.contrib.auth.models import User

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

