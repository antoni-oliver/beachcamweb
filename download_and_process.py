# Run this script to download and process the data contained in the django models:

# Before importing django, configure it
import os

from django.utils import timezone

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
import django
django.setup()

import requests
from PIL import Image
from io import BytesIO
from django.conf import settings
from data.models import BeachCam, Prediction
from predictions.classes.BayesianPredictor import BayesianPredictor

predictors = [BayesianPredictor()]

#delete the outdated images of the file system from outdated predictions
outdatedPredictions = Prediction.getOutDatedPredictions()
for outdatePrediction in outdatedPredictions:
    if outdatePrediction.image:
        predictionImg = outdatePrediction.image.url
        outdatePrediction.deleteImg()
            
for beachcam in BeachCam.objects.all():
    
    response = requests.get(beachcam.url_image)
    img_path = settings.MEDIA_ROOT / beachcam.getNewFileName()
    
    image = Image.open(BytesIO(response.content))
    image.save(str(img_path))
    
    for predictor in predictors:
        predictionDTO = predictor.predict(img_path)
        Prediction.saveBeachCamPrediction(
            beachcam,
            predictionDTO.time_stamp,
            predictionDTO.crowd_count,
            predictionDTO.img_predict_content,
            predictor.__class__.__name__
        )
    
    #delete the tmp image
    if os.path.exists(img_path):
        os.remove(img_path)

    print(f"Downloaded and processed {beachcam.beach_name}.")
