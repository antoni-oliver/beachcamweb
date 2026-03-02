from django.shortcuts import get_object_or_404, render
from django.utils.safestring import mark_safe

from django.http import JsonResponse

from apps.webcam.models import WebCam
from apps.core.forms import ImageUploaderForm

from predictions.classes.BayesianPredictor import BayesianPredictor
from predictions.actions.CustomerPredict import CustomerPredict

from datetime import datetime, timedelta
from django.utils import timezone as tz


import logging
logger = logging.getLogger(__name__)


def home(request):
    """ Returns home page. """
    beachcams = list(WebCam.objects.filter(num_consecutive_failures__lte=10).all())
    #beachcams = list(WebCam.objects.filter(snapshots__ts__gte=now() - timedelta(hours=2)))
    return render(request, 'core/home.html', context={'cams': beachcams})


def webcam(request, camera_slug):
    """ Returns ajax_image of latest prediction overimposed on captured image. """
    beachcam = get_object_or_404(WebCam, camera_slug=camera_slug)
    other_beachcams = WebCam.objects.exclude(camera_slug=camera_slug)
    #other_beachcams = WebCam.objects.exclude(camera_slug=camera_slug).filter(snapshots__ts__gte=now() - timedelta(hours=2))
    # history_dates, history_counts = zip(*[[f"'{h.ts.isoformat()}'", round(h.predicted_crowd_count)] 
    #                                       for h in beachcam.history() if h.predicted_crowd_count is not None])
    # history_dates = f'[{",".join([str(a) for a in list(history_dates)])}]'
    # history_counts = f'[{",".join([str(a) for a in list(history_counts)])}]'
    history = [ [h.ts.timestamp() * 1000, h.predicted_crowd_count] 
               for h in beachcam.history() if h.predicted_crowd_count is not None]
    return render(request, 'core/beach.html', context={'cam': beachcam, 'other_cams': other_beachcams, 'prediction': beachcam.last_prediction, 'history': history})

def analyze_image(request):
    # https://docs.djangoproject.com/en/5.0/topics/forms/
    if request.method == "POST":
        form = ImageUploaderForm(request.POST, request.FILES)
        if form.is_valid():
            cleaned_data = form.cleaned_data
            image = cleaned_data.get('image')
            predictor = BayesianPredictor()
            action = CustomerPredict()
            predictionDTO = action.handle(image, predictor)
            if predictionDTO is None:
                return JsonResponse({'errors': "Internal Server Error"}, status=500)
            else:
                return JsonResponse(predictionDTO.to_dict(), status=200)
        else:
            return JsonResponse({'errors': form.errors}, status=400)
    else:
        form = ImageUploaderForm()
        return render(request, 'core/analyze_image.html',  context={'form': form})


def tft_forecast_json(request, camera_slug):
    """Serve TFT forecast as JSON. Accepts optional ?since=YYYY-MM-DD for debug."""
    from apps.prediction.tft_service import tft_service

    days = int(request.GET.get('days', 3))
    days = max(1, min(days, 15))

    since_str = request.GET.get('since')
    since = None
    if since_str:
        try:
            since = tz.make_aware(datetime.strptime(since_str, '%Y-%m-%d'))
        except ValueError:
            return JsonResponse({'error': 'since must be YYYY-MM-DD'}, status=400)

    try:
        webcam = WebCam.objects.select_related('beach').get(camera_slug=camera_slug)
    except WebCam.DoesNotExist:
        return JsonResponse({'error': 'Not found'}, status=404)

    if not tft_service.models:
        return JsonResponse({'error': 'Models not loaded'}, status=503)

    try:
        result = tft_service.predict(webcam, days=days, since=since)
        return JsonResponse(result)
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)
    except Exception as e:
        logger.exception(f"Forecast failed for {camera_slug}")
        return JsonResponse({'error': str(e)}, status=500)


def tft_actuals_json(request, camera_slug):
    """Return actual snapshot data for a date range (for hindcast comparison)."""
    from apps.prediction.tft_service import tft_service

    try:
        webcam = WebCam.objects.select_related('beach').get(camera_slug=camera_slug)
    except WebCam.DoesNotExist:
        return JsonResponse({'error': 'Not found'}, status=404)

    since_str = request.GET.get('since')
    days = int(request.GET.get('days', 3))

    if not since_str:
        return JsonResponse({'error': 'since is required (YYYY-MM-DD)'}, status=400)

    try:
        date_from = tz.make_aware(datetime.strptime(since_str, '%Y-%m-%d'))
    except ValueError:
        return JsonResponse({'error': 'since must be YYYY-MM-DD'}, status=400)

    date_to = date_from + timedelta(days=days)
    actuals = tft_service.get_actuals(webcam, date_from, date_to)

    return JsonResponse({
        'webcam': camera_slug,
        'since': since_str,
        'days': days,
        'actuals': actuals,
    })


def tft_metrics_json(request, camera_slug):
    """Return pre-computed model metrics for this webcam."""
    from apps.prediction.tft_service import tft_service

    if not tft_service.models:
        return JsonResponse({'error': 'Models not loaded'}, status=503)

    metrics = tft_service.get_metrics(webcam_slug=camera_slug)
    return JsonResponse({'webcam': camera_slug, 'models': metrics})