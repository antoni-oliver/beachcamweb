from django.contrib import admin

from apps.webcam import models

admin.site.register(models.WebCam)
admin.site.register(models.ImageProvider)
admin.site.register(models.StreamProvider)
admin.site.register(models.YoutubeProvider)
