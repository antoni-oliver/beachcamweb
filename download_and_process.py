# Run this script to download and process the data contained in the django models:

# Before importing django, configure it
import os

from django.utils import timezone

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
import django
django.setup()

import requests
from django.conf import settings
from data.models import BeachCam
from predictions.PredictionService import PredictionService
from predictions.prediction_strategies.BayesianPredictor import BayesianPredictor

for beachcam in BeachCam.objects.all():
    image = requests.get(beachcam.url_image)
    now = timezone.now()
    img_path = settings.MEDIA_ROOT / f"{beachcam.beach_name}_{now.strftime('%Y%m%d%H%M%S')}.jpg"
    # img = Image.open(BytesIO(response.content))
    # img.save(str(img_path))

    prediction_service = PredictionService()
    bayesian_predictor = BayesianPredictor()
    prediction_service.makePrediction(beachcam, bayesian_predictor, image)

    print(f"Downloaded and processed {beachcam.beach_name}.")
