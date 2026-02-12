from django.db import models
from django.utils import timezone


class Device(models.Model):
    """
    Model representing an ESP32 (or similar) device
    """
    description = models.CharField(max_length=255)

    class Meta:
        app_label = 'beachImage'
        verbose_name = 'Device'
        verbose_name_plural = 'Devices'

    def __str__(self):
        return f"Device {self.id}: {self.description}"


class BeachImage(models.Model):
    """
    Model for storing beach images from ESP32 devices with GPS coordinates
    """

    # Device reference
    device = models.ForeignKey(
        Device,
        on_delete=models.CASCADE,
        related_name='images'
    )

    # Image file
    image = models.ImageField(upload_to='beach_images/%Y/%m/%d/')
    
    # GPS coordinates
    longitude = models.DecimalField(max_digits=9, decimal_places=6)
    latitude = models.DecimalField(max_digits=9, decimal_places=6)
    
    # Timestamp
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    
    class Meta:
        app_label = 'beachImage'  # Explicitly set app label
        ordering = ['-timestamp']
        verbose_name = 'Beach Image'
        verbose_name_plural = 'Beach Images'
        indexes = [
            models.Index(fields=['-timestamp']),
            models.Index(fields=['latitude', 'longitude']),
        ]
    
    def __str__(self):
        return f"Beach Image {self.id} ({self.device_id}) - {self.timestamp}"