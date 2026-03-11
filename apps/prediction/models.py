from django.db import models

from apps.webcam.models import WebCam


class Snapshot(models.Model):
    webcam = models.ForeignKey(WebCam, on_delete=models.CASCADE, related_name='snapshots')
    ts = models.DateTimeField()
    webcam_image = models.ImageField(null=True, blank=True, upload_to='img/originals/')
    webcam_video = models.FileField(null=True, blank=True, upload_to='stream/originals/')
    filter_frozen_image = models.BooleanField(default=False)
    filter_blurry_image = models.BooleanField(default=False)
    filter_moving_camera = models.BooleanField(default=False)

    # Image masks
    mask_sand = models.ImageField(upload_to='masks/beach/', blank=True, null=True, help_text="Mask of the beach area (sand, areas with people, etc). For non-movable webcams only.")
    mask_swimming_water = models.ImageField(upload_to='masks/swimming', blank=True, null=True, help_text="Mask of the swimming area (near the beach, shallow waters). For non-movable webcams only.")
    mask_sand_and_water = models.ImageField(upload_to='masks/sand_and_water/', blank=True, null=True, help_text="Mask of the whole area of interest (beach + swimming water). For non-movable webcams only.")

    # Prediction data
    predicted_crowd_count = models.FloatField(null=True)
    predicted_image = models.ImageField(null=True, blank=True, upload_to='img/predictions/')

    def __str__(self):
        return f'Snapshot {self.webcam.beach.beach_name} - {self.ts}'
