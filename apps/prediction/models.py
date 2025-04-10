from django.db import models

from apps.webcam.models import WebCam


class Snapshot(models.Model):
    webcam = models.ForeignKey(WebCam, on_delete=models.CASCADE)
    ts = models.DateTimeField()
    webcam_image = models.ImageField(null=True, blank=True, upload_to='img/originals/')
    webcam_video = models.FileField(null=True, blank=True, upload_to='stream/originals/')
    # Prediction data
    predicted_crowd_count = models.FloatField(null=True)
    predicted_image = models.ImageField(null=True, blank=True, upload_to='img/predictions/')

    def __str__(self):
        return f'Snapshot {self.webcam.beach_name} - {self.ts}'
