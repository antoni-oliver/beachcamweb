
import logging
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

from apps.webcam.models import WebCam
from .tft_service import tft_service

logger = logging.getLogger(__name__)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def tft_predict(request, camera_slug):
    """
    Run TFT forecast for a beach webcam.

    Query params:
        days (int): forecast horizon in days, 1-30. Default 3.

    Returns JSON:
        model, beach, webcam, horizon_days, horizon_hours, predictions[]
    """
    days = int(request.GET.get('days', 3))
    days = max(1, min(days, 15))

    try:
        webcam = WebCam.objects.select_related('beach').get(camera_slug=camera_slug)
    except WebCam.DoesNotExist:
        return JsonResponse({'error': f'Webcam {camera_slug} not found'}, status=404)

    if not tft_service.models:
        return JsonResponse({'error': 'TFT models not loaded'}, status=503)

    try:
        result = tft_service.predict(webcam, days=days)
        return JsonResponse(result)
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)
    except Exception as e:
        logger.exception(f"TFT prediction failed for {camera_slug}")
        return JsonResponse({'error': f'Prediction failed: {str(e)}'}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def tft_status(request):
    """Return loaded models and available beaches."""
    models_info = {}
    for key, m in tft_service.models.items():
        models_info[key] = {
            'horizon': m['horizon'],
            'max_days': m['max_days'],
            'input_size': m['input_size'],
            'futr_features': m['futr'],
            'hist_features': m['hist'],
            'n_static_beaches': len(m['static']),
        }

    return JsonResponse({
        'loaded': bool(tft_service.models),
        'models': models_info,
    })


@api_view(['GET'])
def tft_beaches(request):
    """List beaches available for TFT prediction (have enough snapshot history)."""
    from django.db.models import Count, Q

    webcams = (
        WebCam.objects
        .select_related('beach')
        .annotate(snapshot_count=Count(
            'snapshots',
            filter=Q(snapshots__predicted_crowd_count__isnull=False)
        ))
        .filter(snapshot_count__gte=12)
    )

    beaches = []
    for wc in webcams:
        beaches.append({
            'camera_slug': wc.camera_slug,
            'beach_name': wc.beach.beach_name,
            'latitude': float(wc.camera_latitude) if wc.camera_latitude else None,
            'longitude': float(wc.camera_longitude) if wc.camera_longitude else None,
            'snapshot_count': wc.snapshot_count,
        })

    return JsonResponse({'count': len(beaches), 'beaches': beaches})