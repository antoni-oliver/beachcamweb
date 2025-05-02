import os
import re
from datetime import timedelta

import requests
from django.conf import settings
from django.db import models
from django.template.defaultfilters import slugify
from django.utils import timezone
from yt_dlp import YoutubeDL

from apps.webcam import utils


class WebCam(models.Model):
    # Beach info
    beach_name = models.CharField(max_length=200, unique=True)
    slug = models.CharField(max_length=200, unique=True, blank=True, null=True)
    beach_latitude = models.DecimalField(max_digits=9, decimal_places=6, blank=True, null=True)
    beach_longitude = models.DecimalField(max_digits=9, decimal_places=6, blank=True, null=True)
    description = models.CharField(max_length=255, blank=True, null=True)
    # Cam/probing info
    num_consecutive_failures = models.IntegerField(default=0)
    max_crowd_count = models.IntegerField(default=0)
    # Image masks
    mask_beach = models.ImageField(upload_to='masks/beach/', blank=True, null=True, help_text="Mask of the beach area (sand, areas with people, etc). For non-movable webcams only.")
    mask_swimming = models.ImageField(upload_to='masks/swimming', blank=True, null=True, help_text="Mask of the swimming area (near the beach, shallow waters). For non-movable webcams only.")
    mask_boats = models.ImageField(upload_to='masks/ocean', blank=True, null=True, help_text="Mask of the boat area (further from beach, deep waters). For non-movable webcams only.")
    # Webcam info
    public_url = models.CharField(max_length=2048, blank=True, null=True, help_text="URL to redirect viewers to original source.")
    video_seconds = models.IntegerField(default=10, help_text="Seconds to record for the video", blank=True)
    # Webcam info: if provider is static image
    provider_image_url = models.CharField(max_length=2048, help_text="Can use `%Y`, `%m`, `%d`, etc. as in python's strftime(...)", blank=True, null=True)
    # Webcam info: if provider is static image
    provider_stream_m3u8_url = models.CharField(max_length=2048, help_text=".m3u8 static url", blank=True, null=True)
    # Webcam info: if provider is m3u8 stream obtainable after parsing .html file (selenium not needed)
    provider_streamfromregex_url = models.CharField(max_length=2048, help_text=".html url that contains a dynamically-generated .m3u8 that can be located with a regex match.", blank=True, null=True)
    provider_streamfromregex_regex = models.CharField(max_length=2048, help_text="Regex match.", blank=True, null=True)
    provider_streamfromregex_strformat = models.CharField(max_length=2048, help_text="String to be formatted with regex's match[1].", blank=True, null=True)
    # Webcam info: if provider is m3u8 stream obtainable after clicking on element (requires selenium)
    provider_streamfromclick_url = models.CharField(max_length=2048, help_text=".html url that generates a .m3u8 after clicking on a html element.", blank=True, null=True)
    provider_streamfromclick_clickable_element_xpath = models.CharField(max_length=2048, help_text="XPath to the clickable element that generates the stream.", blank=True, null=True)
    # Webcam info: if provider is static image
    provider_youtube_url = models.CharField(max_length=2048, help_text="Only for https://www.youtube.com/watch?v=(...) links.", blank=True, null=True)

    def __str__(self):
        return self.beach_name

    def save(self, *args, **kwargs):
        slug_base = slugify(self.beach_name)
        slug_idx = 0
        slug_candidate = slug_base
        while WebCam.objects.filter(slug=slug_candidate).exists():
            slug_idx += 1
            slug_candidate = slug_base + f'_{slug_idx}'
        self.slug = slug_candidate
        if not self.public_url:
            self.public_url = self.provider_image_url or self.provider_stream_m3u8_url or self.provider_streamfromregex_url or self.provider_streamfromclick_url or self.provider_youtube_url
        super(WebCam, self).save(*args, **kwargs)

    def last_prediction(self):
        return self.snapshot_set.exclude(predicted_crowd_count__isnull=True).order_by('-ts').first()

    def history(self):
        from apps.prediction.models import Snapshot
        dt_since = timezone.now() - timedelta(days=100)
        return list(Snapshot.objects.filter(webcam=self).filter(ts__gt=dt_since).order_by('ts').all())

    def relative_filepath(self, timestamp=None, subfolder=None, extension=None):
        """ Returns a filepath relative to MEDIA_ROOT. """
        timestamp = timestamp or timezone.now()
        path = f'{self.slug}_{timestamp.strftime("%Y%m%d%H%M%S")}'
        if subfolder:
            path = os.path.join(subfolder, path)
        if extension:
            path = path + ('' if extension.startswith('.') else '.') + extension
        os.makedirs(os.path.join(settings.MEDIA_ROOT, os.path.dirname(path)), exist_ok=True)
        return path

    def create_snapshot(self):
        from apps.prediction.models import Snapshot
        ts = timezone.now()
        try:
            video_path, image_path = self.download_video_and_image(timestamp=ts)
            snapshot = Snapshot.objects.create(
                webcam=self,
                ts=ts,
                predicted_crowd_count=None,
            )
            snapshot.webcam_video.name = video_path
            snapshot.webcam_image.name = image_path
            snapshot.save()
            self.num_consecutive_failures = 0
            self.save()
            return snapshot
        except Exception as e:
            print(e)
            # TODO: log error
            self.num_consecutive_failures += 1
            self.save()

    def download_video_and_image(self, timestamp=None):
        """ Downloads the video and image from the provider. """
        timestamp = timestamp or timezone.now()
        image_path = self.relative_filepath(timestamp=timestamp, subfolder='img/originals/', extension='.jpg')
        video_path = self.relative_filepath(timestamp=timestamp, subfolder='vid/originals/', extension='.mp4')
        if self.provider_image_url:
            url = timestamp.strftime(self.provider_image_url)
            with open(os.path.join(settings.MEDIA_ROOT, image_path), 'wb') as f:
                f.write(requests.get(url, verify=False).content)
            return None, image_path
        elif self.provider_stream_m3u8_url:
            utils.video_and_image_from_m3u8(self.provider_stream_m3u8_url, self.video_seconds, os.path.join(settings.MEDIA_ROOT, video_path), os.path.join(settings.MEDIA_ROOT, image_path))
            return video_path, image_path
        elif self.provider_streamfromregex_url:
            response = requests.get(self.provider_streamfromregex_url)
            if response.status_code == 200:
                content = response.text
                regex = self.provider_streamfromregex_regex.encode().decode('unicode-escape')   # Avoid escaping
                match = re.search(regex, content)
                if match is None:
                    raise ValueError(f"Regex '{regex}' did not match any content.")
                m3u8_url = self.provider_streamfromregex_strformat.format(**match.groupdict())
                utils.video_and_image_from_m3u8(m3u8_url, self.video_seconds, os.path.join(settings.MEDIA_ROOT, video_path), os.path.join(settings.MEDIA_ROOT, image_path))
            return video_path, image_path
        elif self.provider_streamfromclick_url:
            stream_urls = utils.m3u8_from_clickable_element(self.provider_streamfromclick_url, self.provider_streamfromclick_clickable_element_xpath)
            utils.video_and_image_from_m3u8(stream_urls[0], self.video_seconds, os.path.join(settings.MEDIA_ROOT, video_path), os.path.join(settings.MEDIA_ROOT, image_path))
            return video_path, image_path
        elif self.provider_youtube_url:
            with YoutubeDL({'format': 'bestvideo/best'}) as ydl:
                result = ydl.extract_info(
                    self.provider_youtube_url,
                    download=False  # We just want to extract the info
                )
            video = result['entries'][0] if 'entries' in result else result  # Can be a playlist or a list of videos
            stream_url = video['url']
            utils.video_and_image_from_m3u8(stream_url, self.video_seconds, os.path.join(settings.MEDIA_ROOT, video_path), os.path.join(settings.MEDIA_ROOT, image_path))
            return video_path, image_path
        raise NotImplementedError()
