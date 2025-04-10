import io
import os

import requests
from django.conf import settings
from django.contrib.contenttypes.fields import GenericForeignKey, GenericRelation
from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.template.defaultfilters import slugify
from django.utils import timezone
from yt_dlp import YoutubeDL

from apps.webcam import utils


class WebCam(models.Model):
    # Beach info
    beach_name = models.CharField(max_length=200, unique=True)
    slug = models.CharField(max_length=200, unique=True, blank=True)
    beach_latitude = models.DecimalField(max_digits=9, decimal_places=6, blank=True, null=True)
    beach_longitude = models.DecimalField(max_digits=9, decimal_places=6, blank=True, null=True)
    description = models.CharField(max_length=255, blank=True, null=True)
    # Cam/probing info
    available = models.BooleanField(default=True)
    probe_freq_mins = models.IntegerField(default=60)
    max_crowd_count = models.IntegerField(default=0)
    # Cam provider
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.PositiveIntegerField()
    provider = GenericForeignKey("content_type", "object_id")

    def __str__(self):
        return self.beach_name

    def save(self, *args, **kwargs):
        self.slug = slugify(self.beach_name)
        super(WebCam, self).save(*args, **kwargs)

    def last_prediction(self):
        return self.snapshot_set.exclude(predicted_crowd_count__isnull=True).order_by('-ts').first()

    def relative_filepath(self, timestamp=None, subfolder=None, extension=None):
        timestamp = timestamp or timezone.now()
        path = f'{self.slug}_{timestamp.strftime("%Y%m%d%H%M%S")}'
        if subfolder:
            path = os.path.join(subfolder, path)
        if extension:
            path = path + ('' if extension.startswith('.') else '.') + extension
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def create_snapshot(self):
        from apps.prediction.models import Snapshot
        ts = timezone.now()
        video_path, image_path = self.provider.download_video_and_image(timestamp=ts)
        snapshot = Snapshot.objects.create(
            webcam=self,
            ts=ts,
            predicted_crowd_count=None,
        )
        snapshot.webcam_video.name = video_path
        snapshot.webcam_image.name = image_path
        snapshot.save()
        return snapshot


class ImageProvider(models.Model):
    webcam = GenericRelation(WebCam, related_query_name="image_provider")
    url = models.CharField(max_length=2048, help_text="Can use `%Y`, `%m`, `%d`, etc. as in python's strftime(...)")

    def __str__(self):
        return f"ImageProvider({self.webcam.first()}): {self.url}"

    def download_video_and_image(self, timestamp=None):
        timestamp = timestamp or timezone.now()
        image_path = self.webcam.first().relative_filepath(timestamp=timestamp, subfolder='img/originals/', extension='.mp4')
        url = timestamp.strftime(self.url)
        with open(os.path.join(settings.MEDIA_ROOT, image_path), 'wb') as f:
            f.write(requests.get(url, verify=False).content)
        return None, image_path


class StreamProvider(models.Model):
    webcam = GenericRelation(WebCam, related_query_name="stream_provider")
    url = models.CharField(max_length=2048, help_text="Use .m3u8 or .html that generates a .m3u8 after clicking on a html element.")
    clickable_element_xpath = models.CharField(max_length=2048)
    seconds = models.IntegerField(default=10, help_text="Seconds to record for the video", blank=True)

    def __str__(self):
        return f"StreamProvider({self.webcam.first()}): {self.url}"

    def download_video_and_image(self, timestamp=None):
        timestamp = timestamp or timezone.now()
        webcam = self.webcam.first()
        image_path = webcam.relative_filepath(timestamp=timestamp, subfolder='img/originals/', extension='.jpg')
        video_path = webcam.relative_filepath(timestamp=timestamp, subfolder='vid/originals/', extension='.mp4')
        stream_urls = utils.m3u8_from_url(self.url, self.clickable_element_xpath)
        utils.video_and_image_from_m3u8(stream_urls[0], self.seconds, os.path.join(settings.MEDIA_ROOT, video_path), os.path.join(settings.MEDIA_ROOT, image_path))
        return video_path, image_path


class YoutubeProvider(models.Model):
    webcam = GenericRelation(WebCam, related_query_name="stream_provider")
    url = models.CharField(max_length=2048, help_text="Only for https://www.youtube.com/watch?v=(...) links.")
    seconds = models.IntegerField(default=10, help_text="Seconds to record for the video", blank=True)

    def __str__(self):
        return f"YoutubeProvider({self.webcam.first()}): {self.url}"

    def download_video_and_image(self, timestamp=None):
        """ Provides the path to the stream, and the path to the image."""
        timestamp = timestamp or timezone.now()
        webcam = self.webcam.first()
        with YoutubeDL({'format': 'bestvideo/best'}) as ydl:
            result = ydl.extract_info(
                self.url,
                download=False  # We just want to extract the info
            )
        video = result['entries'][0] if 'entries' in result else result  # Can be a playlist or a list of videos
        stream_url = video['url']

        image_path = webcam.relative_filepath(timestamp=timestamp, subfolder='img/originals/', extension='.jpg')
        video_path = webcam.relative_filepath(timestamp=timestamp, subfolder='vid/originals/', extension='.mp4')
        utils.video_and_image_from_m3u8(stream_url, self.seconds, os.path.join(settings.MEDIA_ROOT, video_path), os.path.join(settings.MEDIA_ROOT, image_path))
        return video_path, image_path
