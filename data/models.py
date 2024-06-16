from django.db import models
from django.utils import timezone
from datetime import timedelta

class BeachCam(models.Model):
    beach_name = models.CharField(max_length=200, unique=True)
    beach_latitude = models.DecimalField(max_digits=9, decimal_places=6)
    beach_longitude = models.DecimalField(max_digits=9, decimal_places=6)
    url_image = models.CharField(max_length=200, null=True)
    url_aemet = models.CharField(max_length=200, null=True, blank=True)
    url_platgesbalears = models.CharField(max_length=200, null=True, blank=True)
    probe_freq_mins = models.IntegerField(default=60)
    description = models.CharField(max_length=255, null=True)

    max_crowd_count = models.IntegerField()

    def __str__(self):
        return self.beach_name

    def last_prediction(self):
        return self.prediction_set.order_by('-ts').first()
    
    def getNewFileName(self):
        return f'{self.beach_name}_{timezone.now().strftime("%Y%m%d%H%M%S")}.jpg'


class Prediction(models.Model):
    beachcam = models.ForeignKey(BeachCam, on_delete=models.CASCADE)
    ts = models.DateTimeField()
    image = models.ImageField(null=True, blank=True, upload_to='count/')
    crowd_count = models.FloatField(default=0)
    algorithm = models.CharField(max_length=255, default="")

    def __str__(self):
        return f'{self.beachcam.beach_name} - {self.ts}'
    
    def getCrowdCount(self):
        return int(self.crowd_count)
    
    def getCrowdCountText(self):
        return f"Estimat de gent actual: {self.getCrowdCount()}"

    @staticmethod 
    def saveBeachCamPrediction(beachcam: BeachCam, time_stamp, crowd_count, img_content, algorithm):
        prediction = Prediction.objects.create(
            beachcam=beachcam,
            ts=time_stamp,
            crowd_count=crowd_count,
            algorithm=algorithm
        )
        prediction.image.save(
            beachcam.getNewFileName(),
            img_content
        )
        prediction.save()    
        return prediction
    
    def deleteImg(self):
        self.image.delete(save=True)
    
    def getOutDatedPredictions():
        return Prediction.objects.filter(ts__gte=timezone.now() + timedelta(days=1)).exclude(image__exact='')

        
