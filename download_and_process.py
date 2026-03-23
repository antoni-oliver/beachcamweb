# Run this script to download and process the data contained in the django models:

# Before importing django, configure it
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django
django.setup()

from datetime import timedelta

from django.conf import settings
from django.utils import timezone

from apps.prediction.models import Snapshot
from apps.webcam.models import WebCam

from predictions.classes.BayesianPredictor import BayesianPredictor

import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # To download pytorch model

predictors = [BayesianPredictor()]

def main():
    def _log(level: str, message: str, **context):
        ts = timezone.now().isoformat()
        ctx = " ".join(f"{k}={v}" for k, v in context.items())
        print(f"[{ts}] [{level}] {message}" + (f" | {ctx}" if ctx else ""))

    total_webcams = WebCam.objects.count()
    _log("INFO", "Starting download and process run", webcams=total_webcams, predictors=len(predictors))

    for index, beachcam in enumerate(WebCam.objects.all(), start=1):
        _log(
            "INFO",
            "Processing webcam",
            index=f"{index}/{total_webcams}",
            webcam_id=beachcam.pk,
            beach=beachcam.beach.beach_name,
            camera_slug=beachcam.camera_slug,
        )

        snapshot = beachcam.create_snapshot()
        if not snapshot:
            _log("ERROR", "Snapshot creation failed", webcam_id=beachcam.pk)
            continue

        _log(
            "INFO",
            "Snapshot created",
            webcam_id=beachcam.pk,
            snapshot_ts=snapshot.ts.isoformat() if snapshot.ts else "None",
            has_mask=bool(snapshot.mask_sand_and_water),
            image_path=snapshot.webcam_image.path if snapshot.webcam_image else "None",
        )

        for predictor in predictors:
            predictor_name = predictor.__class__.__name__
            try:
                if snapshot.mask_sand_and_water:
                    mask_paths = [snapshot.mask_sand_and_water.path]
                else:
                    mask_paths = []

                _log(
                    "DEBUG",
                    "Starting prediction",
                    webcam_id=beachcam.pk,
                    predictor=predictor_name,
                    masks=len(mask_paths),
                )

                predictionDTO = predictor.predict(snapshot.webcam_image.path, mask_paths)
                _log(
                    "INFO",
                    "Prediction completed",
                    webcam_id=beachcam.pk,
                    predictor=predictor_name,
                    crowd_count=predictionDTO.crowd_count,
                    prediction_bytes=len(predictionDTO.img_predict_content),
                )

                snapshot.predicted_crowd_count = predictionDTO.crowd_count
                prediction_image_path = beachcam.relative_filepath(
                    timestamp=snapshot.ts,
                    subfolder="img/predictions/",
                    extension=".jpg",
                )
                absolute_prediction_path = os.path.join(settings.MEDIA_ROOT, prediction_image_path)
                os.makedirs(os.path.dirname(absolute_prediction_path), exist_ok=True)

                with open(absolute_prediction_path, "wb") as f:
                    f.write(predictionDTO.img_predict_content)

                snapshot.predicted_image.name = prediction_image_path
                snapshot.save()

                previous_max = beachcam.max_crowd_count
                beachcam.max_crowd_count = max(beachcam.max_crowd_count, predictionDTO.crowd_count)
                beachcam.save()

                _log(
                    "INFO",
                    "Prediction saved",
                    webcam_id=beachcam.pk,
                    predictor=predictor_name,
                    predicted_image=prediction_image_path,
                    max_crowd_before=previous_max,
                    max_crowd_after=beachcam.max_crowd_count,
                )
            except Exception as e:
                _log(
                    "ERROR",
                    "Prediction pipeline failed",
                    webcam_id=beachcam.pk,
                    predictor=predictor_name,
                    error_type=type(e).__name__,
                    error=str(e),
                )
                import traceback
                print(traceback.format_exc())
    # print(f"Deleting old data.")
    # # Delete the outdated images of the file system from outdated predictions
    # for outdated_pred in Snapshot.objects.filter(ts__lte=timezone.now() - timedelta(days=1)):
    #     # TODO: Move `webcam_image` into a new storage server, rather than deleting it?
    #     if outdated_pred.webcam_image:
    #         outdated_pred.webcam_image.delete(save=True)
    #     if outdated_pred.webcam_video:
    #         outdated_pred.webcam_image.delete(save=True)
    #     if outdated_pred.predicted_image:
    #         outdated_pred.predicted_image.delete(save=True)


if __name__ == "__main__":
    main()
