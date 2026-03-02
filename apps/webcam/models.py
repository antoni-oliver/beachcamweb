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
    beach = models.ForeignKey('Beach', on_delete=models.CASCADE, related_name='webcams')
    # Beach info
    camera_slug = models.CharField(max_length=200, unique=True, blank=True, null=True)
    camera_latitude = models.DecimalField(max_digits=9, decimal_places=6, blank=True, null=True)
    camera_longitude = models.DecimalField(max_digits=9, decimal_places=6, blank=True, null=True)
    camera_description = models.CharField(max_length=255, blank=True, null=True)
    # Cam/probing info
    num_consecutive_failures = models.IntegerField(default=0)
    max_crowd_count = models.IntegerField(default=0)
    # Image masks
    mask_beach = models.ImageField(upload_to='masks/beach/', blank=True, null=True,
                                   help_text="Mask of the beach area (sand, areas with people, etc). For non-movable webcams only.")
    mask_swimming = models.ImageField(upload_to='masks/swimming', blank=True, null=True,
                                      help_text="Mask of the swimming area (near the beach, shallow waters). For non-movable webcams only.")
    mask_boats = models.ImageField(upload_to='masks/ocean', blank=True, null=True,
                                   help_text="Mask of the boat area (further from beach, deep waters). For non-movable webcams only.")
    # Webcam info
    public_url = models.CharField(max_length=2048, blank=True, null=True, help_text="URL to redirect viewers to original source.")
    video_seconds = models.IntegerField(default=10, help_text="Seconds to record for the video", blank=True)
    # Webcam info: if provider is static image
    provider_image_url = models.CharField(max_length=2048, help_text="Can use `%Y`, `%m`, `%d`, etc. as in python's strftime(...)", blank=True, null=True)
    # Webcam info: if provider is static image
    provider_stream_m3u8_url = models.CharField(max_length=2048, help_text=".m3u8 static url", blank=True, null=True)
    # Webcam info: if provider is m3u8 stream obtainable after parsing .html file (selenium not needed)
    provider_streamfromregex_url = models.CharField(max_length=2048, help_text=".html url that contains a dynamically-generated .m3u8 that can be located with a regex match.",
                                                    blank=True, null=True)
    provider_streamfromregex_regex = models.CharField(max_length=2048, help_text="Regex match.", blank=True, null=True)
    provider_streamfromregex_strformat = models.CharField(max_length=2048, help_text="String to be formatted with regex's match[1].", blank=True, null=True)
    # Webcam info: if provider is m3u8 stream obtainable after clicking on element (requires selenium)
    provider_streamfromclick_url = models.CharField(max_length=2048, help_text=".html url that generates a .m3u8 after clicking on a html element.", blank=True, null=True)
    provider_streamfromclick_clickable_element_xpath = models.CharField(max_length=2048, help_text="XPath to the clickable element that generates the stream.", blank=True,
                                                                        null=True)
    # Webcam info: if provider is static image
    provider_youtube_url = models.CharField(max_length=2048, help_text="Only for https://www.youtube.com/watch?v=(...) links.", blank=True, null=True)

    def __str__(self):
        return f'WebCam {self.beach.beach_name} - {self.camera_slug}'

    def save(self, *args, **kwargs):
        # Only auto-generate on create (or if camera_slug was left blank)
        if not self.camera_slug:
            slug_base = slugify(self.beach.beach_name) if self.beach_id else "webcam"
            slug_candidate = slug_base
            slug_idx = 0
            qs = WebCam.objects.all()
            if self.pk:
                qs = qs.exclude(pk=self.pk)
            while qs.filter(camera_slug=slug_candidate).exists():
                slug_idx += 1
                slug_candidate = f"{slug_base}_{slug_idx}"
            self.camera_slug = slug_candidate
        if not self.public_url:
            self.public_url = self.provider_image_url or self.provider_stream_m3u8_url or self.provider_streamfromregex_url or self.provider_streamfromclick_url or self.provider_youtube_url
        super(WebCam, self).save(*args, **kwargs)

    def last_prediction(self):
        return self.snapshots.exclude(predicted_crowd_count__isnull=True).order_by('-ts').first()

    def history(self):
        from apps.prediction.models import Snapshot
        # dt_since = timezone.now() - timedelta(days=100)
        # return list(Snapshot.objects.filter(webcam=self).filter(ts__gt=dt_since).order_by('ts').all())
        return list(Snapshot.objects.filter(webcam=self).order_by('ts').all())

    def relative_filepath(self, timestamp=None, subfolder=None, extension=None):
        """ Returns a filepath relative to MEDIA_ROOT. """
        timestamp = timestamp or timezone.now()
        path = f'{self.camera_slug}_{timestamp.strftime("%Y%m%d%H%M%S")}'
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
            img_abs = os.path.join(settings.MEDIA_ROOT, image_path) if image_path else None
            vid_abs = os.path.join(settings.MEDIA_ROOT, video_path) if video_path else None

            if img_abs and not os.path.exists(img_abs):
                raise RuntimeError(f"Image not created: {img_abs}")
            if vid_abs and not os.path.exists(vid_abs):
                raise RuntimeError(f"Video not created: {vid_abs}")

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
            utils.video_and_image_from_m3u8(self.provider_stream_m3u8_url, self.video_seconds, os.path.join(settings.MEDIA_ROOT, video_path),
                                            os.path.join(settings.MEDIA_ROOT, image_path))
            return video_path, image_path
        elif self.provider_streamfromregex_url:
            response = requests.get(self.provider_streamfromregex_url)
            if response.status_code == 200:
                content = response.text
                regex = self.provider_streamfromregex_regex.encode().decode('unicode-escape')  # Avoid escaping
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


class Beach(models.Model):
    # Basic info
    beach_slug = models.CharField(max_length=200, unique=True, help_text="Provided by FundacioBit")
    beach_name = models.CharField(max_length=200)
    source_url = models.URLField(max_length=2048, blank=True, null=True)

    class Occupancy(models.TextChoices):
        HIGH = "HIGH", "Alto"
        MEDIUM = "MEDIUM", "Medio"
        LOW = "LOW", "Bajo"

    class Proximity(models.TextChoices):
        URBAN = "URBAN", "Urbana"
        SEMI_URBAN = "SEMI_URBAN", "Semiurbana"
        REMOTE = "REMOTE", "Alejada"

    class BeachType(models.TextChoices):
        NATURAL = "NATURAL", "Natural"
        ARTIFICIAL = "ARTIFICIAL", "Artificial"

    class Composition(models.TextChoices):
        SAND = "SAND", "Arena"
        PEBBLES = "PEBBLES", "Bolos"
        GRAVEL = "GRAVEL", "Graba"
        ROCKS = "ROCKS", "Rocas"

    class BathingConditions(models.TextChoices):
        CALM = "CALM", "Aguas tranquilas"
        MODERATE = "MODERATE", "Oleaje moderado"
        STRONG = "STRONG", "Oleaje fuerte"

    class DominantWinds(models.TextChoices):
        LIGHT_BREEZE = "LIGHT_BREEZE", "Brisa ligera"
        DOMINANT = "DOMINANT", "Dominante"
        STRONG = "STRONG", "Fuerte"

    class BeachSlope(models.TextChoices):
        GENTLE = "GENTLE", "Ángulo suave"
        NORMAL = "NORMAL", "Ángulo normal"
        STEEP = "STEEP", "Pendiente brusco"

    # Basic technical data
    longitud = models.CharField(max_length=100, blank=True, null=True)
    anchura = models.CharField(max_length=100, blank=True, null=True)
    limites = models.CharField(max_length=255, blank=True, null=True)
    localizacion = models.CharField(max_length=255, blank=True, null=True)
    coordenadas_geograficas = models.CharField(max_length=255, blank=True, null=True)

    # Single-choice fields 
    grado_de_ocupacion = models.CharField(max_length=20, choices=Occupancy.choices, blank=True, null=True)
    proximidad_al_nucleo_urbano = models.CharField(max_length=20, choices=Proximity.choices, blank=True, null=True)
    tipos_de_playas = models.CharField(max_length=20, choices=BeachType.choices, blank=True, null=True)
    composicion_de_la_playa = models.CharField(max_length=20, choices=Composition.choices, blank=True, null=True)
    condiciones_de_bano = models.CharField(max_length=20, choices=BathingConditions.choices, blank=True, null=True)
    vientos_dominantes = models.CharField(max_length=30, choices=DominantWinds.choices, blank=True, null=True)
    pendiente_de_la_playa = models.CharField(max_length=20, choices=BeachSlope.choices, blank=True, null=True)

    # Optional booleans 
    paseo_maritimo = models.BooleanField(blank=True, null=True)
    vegetacion = models.BooleanField(blank=True, null=True)
    proximidad_a_zona_residencial = models.BooleanField(blank=True, null=True)
    acceso_para_discapacitado = models.BooleanField(blank=True, null=True)
    localizacion_de_accesos_para_vehiculos_de_emergencias = models.BooleanField(blank=True, null=True)
    campos_de_dunas = models.BooleanField(blank=True, null=True)
    marismas = models.BooleanField(blank=True, null=True)
    vegetacion_protegida = models.BooleanField(blank=True, null=True)
    espacio_natural_protegido = models.BooleanField(blank=True, null=True)
    ondulaciones_de_fondo = models.BooleanField(blank=True, null=True)
    barras_sumergidas = models.BooleanField(blank=True, null=True)
    fosas = models.BooleanField(blank=True, null=True)
    obstaculos_semisumergidos_o_sumergidos = models.BooleanField(blank=True, null=True)

    # Numeric field from the source 
    espigones = models.IntegerField(blank=True, null=True)

    # Ports 
    puertos_mares = models.BooleanField(blank=True, null=True)
    puertos_islas = models.BooleanField(blank=True, null=True)
    puertos_baches = models.BooleanField(blank=True, null=True)
    canal = models.BooleanField(blank=True, null=True)
    arenosa = models.BooleanField(blank=True, null=True)
    rocosa_en_la_montana = models.BooleanField(blank=True, null=True)
    rocosa_en_costa_de_escalon = models.BooleanField(blank=True, null=True)
    playa_mixta_de_grava_y_arena = models.BooleanField(blank=True, null=True)

    # Accessibility
    accesos_a_la_playa_para_peatones = models.BooleanField(default=False)
    accesos_a_la_playa_para_vehiculos = models.BooleanField(default=False)
    accesos_a_la_playa_para_barcos = models.BooleanField(default=False)

    # User type booleans
    tipo_de_usuario_local = models.BooleanField(default=False)
    tipo_de_usuario_turista = models.BooleanField(default=False)

    # Wave direction booleans
    oleaje_dir_n = models.BooleanField(default=False)
    oleaje_dir_ne = models.BooleanField(default=False)
    oleaje_dir_e = models.BooleanField(default=False)
    oleaje_dir_se = models.BooleanField(default=False)
    oleaje_dir_s = models.BooleanField(default=False)
    oleaje_dir_so = models.BooleanField(default=False)
    oleaje_dir_o = models.BooleanField(default=False)
    oleaje_dir_no = models.BooleanField(default=False)

    # Wave type booleans
    tipos_de_olas_mar_de_fondo = models.BooleanField(default=False)
    tipos_de_olas_mar_de_viento = models.BooleanField(default=False)

    # Water flow booleans
    flujo_de_agua_centrales_termicas = models.BooleanField(default=False)
    flujo_de_agua_corriente_de_marea = models.BooleanField(default=False)
    flujo_de_agua_corriente_de_oleaje = models.BooleanField(default=False)
    flujo_de_agua_corrientes_de_resaca = models.BooleanField(default=False)
    flujo_de_agua_deriva_litoral = models.BooleanField(default=False)
    flujo_de_agua_desembocaduras = models.BooleanField(default=False)
    flujo_de_agua_otros = models.BooleanField(default=False)
    flujo_de_agua_rieras = models.BooleanField(default=False)

    def __str__(self) -> str:
        return f"{self.beach_name} ({self.beach_slug})"