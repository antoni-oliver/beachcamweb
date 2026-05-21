# Before importing django, configure it
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django
django.setup()


from apps.prediction.models import Snapshot
from predictions.classes.BayesianPredictor import BayesianPredictor
predictor = BayesianPredictor()

total = Snapshot.objects.count()

for idx, snapshot in enumerate(Snapshot.objects.all()):
    if idx % 100 == 0:
        print(f'Processing {idx/total:%} - {idx}/{total}')
    try:
        if snapshot.mask_sand_and_water:
            mask_paths = [snapshot.mask_sand_and_water.path]
        else:
            mask_paths = []
        predictionDTO = predictor.predict(snapshot.webcam_image.path, mask_paths)
        snapshot.predicted_crowd_count = predictionDTO.crowd_count
        snapshot.save()
    except Exception as e:
        # Handle any exception
        print(f"download_and_process.py an error ocurred: {e}")
        # TODO: make it NOT available.