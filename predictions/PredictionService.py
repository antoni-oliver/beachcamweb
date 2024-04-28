from django.utils import timezone
from predictions.interfaces.PredictorStrategy import PredictorStrategy
from data.models import BeachCam, Prediction
from io import BytesIO

class PredictionService:

    def makePrediction(self,beachcam: BeachCam, predictor: PredictorStrategy, img):
        predictionDTO = predictor.predict(img)
        now = timezone.now()
        prediction = Prediction.objects.create(
            beachcam=beachcam,
            ts=now,
            crowd_count= predictionDTO.crowd_count
        )
        """ prediction.image.save(
            f'{beachcam.beach_name}_{now.strftime("%Y%m%d%H%M%S")}.jpg',
            BytesIO(predictionDTO.img_predict_content)
        ) """
        prediction.save()
        #TODO mejorar la respuesta del servicio
        return "saved"
    