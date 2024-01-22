# Run this script to download and process the data contained in the django models:

# Before importing django, configure it
import os

from django.utils import timezone

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
import django
django.setup()

from datetime import datetime
import requests
from PIL import Image
from django.conf import settings
from io import BytesIO
from data.models import BeachCam, Prediction


for beachcam in BeachCam.objects.all():
    response = requests.get(beachcam.url_image)
    now = timezone.now()
    img_path = settings.MEDIA_ROOT / f"{beachcam.beach_name}_{now.strftime('%Y%m%d%H%M%S')}.jpg"
    # img = Image.open(BytesIO(response.content))
    # img.save(str(img_path))


    prediction = Prediction.objects.create(
        beachcam=beachcam,
        ts=now,
        crowd_count=0
    )
    prediction.image.save(
        f'{beachcam.beach_name}_{now.strftime("%Y%m%d%H%M%S")}.jpg',
        BytesIO(response.content))
    prediction.save()

    print(f"Downloaded and processed {beachcam.beach_name}.")
