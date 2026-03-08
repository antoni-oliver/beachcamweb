import logging
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from datetime import datetime, timedelta
from django.utils import timezone as tz

from apps.webcam.models import WebCam
from .tft_service import tft_service

logger = logging.getLogger(__name__)


@require_GET
def tft_forecast_json(request, camera_slug):
    days = int(request.GET.get('days', 3))
    days = max(1, min(days, 15))
    model_set = request.GET.get('model_set', 'default')

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

    try:
        result = tft_service.predict(webcam, days=days, since=since, model_set=model_set)
        return JsonResponse(result)
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)
    except Exception as e:
        logger.exception(f"Forecast failed for {camera_slug}")
        return JsonResponse({'error': str(e)}, status=500)


@require_GET
def tft_actuals_json(request, camera_slug):
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
    return JsonResponse({'webcam': camera_slug, 'since': since_str, 'days': days, 'actuals': actuals})


@require_GET
def tft_metrics_json(request, camera_slug):
    model_set = request.GET.get('model_set', 'default')
    metrics = tft_service.get_metrics(webcam_slug=camera_slug, model_set=model_set)
    return JsonResponse({'webcam': camera_slug, 'model_set': model_set, 'models': metrics})


@require_GET
def tft_model_sets_json(request, camera_slug):
    sets = tft_service.list_model_sets()
    return JsonResponse({'model_sets': list(sets.keys())})