from predictions.interfaces.PredictorInterface import PredictorInterface
from predictions.DTO.PredictionDTO import PredictionDTO
from django.conf import settings
from django.utils import timezone
from django import forms
from PIL import Image
import os
import logging
logger = logging.getLogger(__name__)

class CustomerPredict:
    
    def handle(self, image: forms.ImageField, predictor: PredictorInterface) -> None|PredictionDTO:
        
        #save customer image
        image_path = self.getImagePath(image)
        image_class = Image.open(image)
        image_class.save(image_path)
        
        predictionDTO = None
        try:
            predictionDTO = predictor.predict(image_path)
        
        except Exception as e:
            logger.error(f"CustomerPredict::handle, error: {e}")
            
        #delete customer image
        if os.path.exists(image_path):
            os.remove(image_path)
        
        return predictionDTO
    
    def getImagePath(self, image: forms.ImageField):
        return settings.MEDIA_ROOT / f'{self.__class__.__name__}_{timezone.now().strftime("%Y%m%d%H%M%S")}.{image.image.format.lower()}'