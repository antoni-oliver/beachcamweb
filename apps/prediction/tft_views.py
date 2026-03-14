import logging
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from datetime import datetime, timedelta
from django.utils import timezone as tz
from apps.prediction.models import Snapshot

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

@require_GET
def tft_snapshot_image_json(request, camera_slug):

    ts_str = request.GET.get('ts')
    if not ts_str:
        return JsonResponse({'error': 'ts required'}, status=400)

    try:
        ts = tz.make_aware(datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S'))
    except ValueError:
        return JsonResponse({'error': 'ts must be YYYY-MM-DDTHH:MM:SS'}, status=400)

    try:
        webcam = WebCam.objects.get(camera_slug=camera_slug)
    except WebCam.DoesNotExist:
        return JsonResponse({'error': 'Not found'}, status=404)

    snapshot = (
        Snapshot.objects
        .filter(webcam=webcam, ts__gte=ts - timedelta(minutes=30), ts__lte=ts + timedelta(minutes=30))
        .order_by('ts')
        .first()
    )

    if snapshot:
        original_url = snapshot.webcam_image.url if snapshot.webcam_image else None
        prediction_url = snapshot.predicted_image.url if snapshot.predicted_image else None
        return JsonResponse({'url': original_url, 'prediction_url': prediction_url, 'ts': snapshot.ts.isoformat()})
    return JsonResponse({'url': None})

@require_GET
def tft_full_forecast_json(request, beach_id=None):
    """
    GET /beach/full-forecast/?days=15&since=YYYY-MM-DD
    GET /beach/<beach_id>/full-forecast/

    Returns per-webcam predictions + capacity-weighted mean summary per beach.
    The weighted mean uses: rel = sum(cc_i) / sum(mcc_i), abs = sum(cc_i * mcc_i) / sum(mcc_i)
    """
    days = int(request.GET.get('days', 15))
    days = max(1, min(days, 15))

    since_str = request.GET.get('since')
    since = None
    if since_str:
        try:
            since = tz.make_aware(datetime.strptime(since_str, '%Y-%m-%d'))
        except ValueError:
            return JsonResponse({'error': 'since must be YYYY-MM-DD'}, status=400)

    base_qs = WebCam.objects.select_related('beach').order_by('-max_crowd_count')

    if beach_id is not None:
        cams_qs = base_qs.filter(beach_id=int(beach_id))
    elif request.GET.get('id_beach'):
        cams_qs = base_qs.filter(beach__beach_slug=request.GET['id_beach'])
    else:
        cams_qs = base_qs

    # Group cameras by beach
    beaches: dict[int, list] = {}
    for cam in cams_qs:
        beaches.setdefault(cam.beach_id, []).append(cam)

    if not beaches:
        return JsonResponse({'error': 'Beach not found or has no cameras'}, status=404)

    def _weighted_mean(cam_results):
        """Merge per-camera predictions into one weighted mean series."""
        # Build lookup: camera_slug -> {timestamp_prefix: prediction}
        maps = []
        for res in cam_results:
            mcc = float(res.get('max_crowd_count') or 0)
            if mcc <= 0:
                continue
            pred_map = {
                p['timestamp'][:16]: p
                for p in res.get('predictions', [])
                if p.get('available') and p.get('crowd_count') is not None
            }
            maps.append((mcc, pred_map, res))

        if not maps:
            return None

        # Collect all timestamps
        all_keys = sorted(set(k for _, m, _ in maps for k in m))
        merged = []
        for key in all_keys:
            sum_cc = 0.0
            sum_cc_x_mcc = 0.0
            sum_mcc = 0.0
            temperature = None
            for mcc, pred_map, _ in maps:
                if key in pred_map:
                    p = pred_map[key]
                    cc = float(p['crowd_count'])
                    sum_cc += cc
                    sum_cc_x_mcc += cc * mcc
                    sum_mcc += mcc
                    if temperature is None:
                        temperature = p.get('temperature')
            if sum_mcc > 0:
                rel = sum_cc / sum_mcc
                abs_val = sum_cc_x_mcc / sum_mcc
                level = ('high' if rel >= 0.75 else 'medium' if rel >= 0.50 else 'low' if rel >= 0.25 else 'very_low')
                merged.append({
                    'timestamp': key + ':00',
                    'hour': int(key[11:13]),
                    'crowd_count': round(abs_val, 2),
                    'available': True,
                    'occupancy_level': level.upper(),
                    'occupancy_ratio': round(rel, 4),
                    'temperature': temperature,
                })
        return merged

    results = []
    for beach_id_key, cams in beaches.items():
        # Dedup cameras by slug (queryset may return duplicates via joins)
        seen_slugs = set()
        unique_cams = []
        for cam in cams:
            if cam.camera_slug not in seen_slugs:
                seen_slugs.add(cam.camera_slug)
                unique_cams.append(cam)

        cam_results = []
        for cam in unique_cams:
            try:
                res = tft_service.predict_mixed(cam, days=days, since=since)
                res['webcam'] = cam.camera_slug
                res['max_crowd_count'] = float(cam.max_crowd_count or 0)
                cam_results.append(res)
            except Exception as e:
                logger.warning(f"Full forecast skipped for {cam.camera_slug}: {e}")

        if not cam_results:
            continue

        first = cam_results[0]
        merged_predictions = _weighted_mean(cam_results) if len(cam_results) > 1 else first.get('predictions', [])

        results.append({
            'beach': first.get('beach'),
            'beach_id': beach_id_key,
            'horizon_days': first.get('horizon_days'),
            'segments': first.get('segments'),
            'predictions': merged_predictions,
            'cameras': cam_results,
        })

    if len(results) == 1:
        return JsonResponse(results[0])
    return JsonResponse({'forecasts': results})

@require_GET
def tft_eval_ranking_json(request, camera_slug):
    from django.conf import settings
    import json as _json
    from pathlib import Path

    raw = getattr(settings, 'TFT_EVAL_JSON', None)
    if not raw:
        return JsonResponse({'error': 'TFT_EVAL_JSON not configured in settings.'}, status=404)

    eval_path = Path(raw)
    if not eval_path.is_absolute():
        eval_path = Path(settings.BASE_DIR) / eval_path

    if not eval_path.exists():
        return JsonResponse({'error': f'Evaluation file not found: {eval_path}. Run evaluate_model_sets first.'}, status=404)

    data = _json.loads(eval_path.read_text())
    return JsonResponse({
        'ranking':         data.get('ranking', []),
        'best_by_horizon': data.get('best_by_horizon', {}),
        'evaluated_at':    data.get('evaluated_at'),
        'date_range':      data.get('date_range'),
    })