from __future__ import annotations

from datetime import datetime, timezone as dt_timezone

import jwt
from django.conf import settings
from django.db.models import OuterRef, Subquery, DateTimeField, FloatField
from django.http import JsonResponse, HttpRequest
from django.utils import timezone
from django.views.decorators.http import require_GET

from apps.webcam.models import WebCam
from apps.prediction.models import Snapshot
from apps.api import utils as api_utils


def _unix(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=dt_timezone.utc)
    return int(dt.timestamp())


def _get_token(request: HttpRequest) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return request.GET.get("key")


def _require_jwt(request: HttpRequest) -> JsonResponse | None:
    token = _get_token(request)
    if not token:
        return JsonResponse(
            {"detail": "Missing JWT (use Authorization: Bearer <token> or ?key=<token>)"},
            status=401,
        )

    secret = getattr(settings, "API_JWT_SECRET", None)
    algos = getattr(settings, "API_JWT_ALGOS", ["HS256"])

    if not secret:
        return JsonResponse(
            {"detail": "Server misconfigured: API_JWT_SECRET not set"},
            status=500,
        )

    try:
        jwt.decode(token, secret, algorithms=algos, options={"verify_aud": False})
        return None
    except jwt.PyJWTError as e:
        return JsonResponse({"detail": f"Invalid JWT: {str(e)}"}, status=401)


def _annotate_last_snapshot(qs):
    last_snap_qs = (
        Snapshot.objects
        .filter(webcam=OuterRef("pk"))
        .exclude(predicted_crowd_count__isnull=True)
        .order_by("-ts")
    )
    return qs.annotate(
        last_pred_ts=Subquery(last_snap_qs.values("ts")[:1], output_field=DateTimeField()),
        last_pred_count=Subquery(last_snap_qs.values("predicted_crowd_count")[:1], output_field=FloatField()),
    )


def _filter_cams(id_camera: str | None, id_beach: str | None):
    """
    id_camera: WebCam.id
    id_beach:  Beach.beach_slug
    """
    qs = WebCam.objects.select_related("beach").all()

    if id_camera:
        qs = qs.filter(id=int(id_camera))

    if id_beach:
        qs = qs.filter(beach__beach_slug=id_beach)

    return qs


@require_GET
def camera_info(request: HttpRequest):
    auth_err = _require_jwt(request)
    if auth_err:
        return auth_err

    id_camera = request.GET.get("id_camera")
    id_beach = request.GET.get("id_beach")

    qs = _filter_cams(id_camera, id_beach)
    qs = _annotate_last_snapshot(qs)

    out = []
    for cam in qs:
        out.append({
            "id_camera": cam.id,
            "playa_slug": cam.beach.beach_slug,
            "camera_slug": cam.camera_slug,
            "timestamp_last_prediction": _unix(cam.last_pred_ts) if cam.last_pred_ts else None,
            "latitude": float(cam.camera_latitude) if cam.camera_latitude is not None else None,
            "longitude": float(cam.camera_longitude) if cam.camera_longitude is not None else None,
        })

    return JsonResponse(out, safe=False)


@require_GET
def estimacio_actual(request: HttpRequest):
    auth_err = _require_jwt(request)
    if auth_err:
        return auth_err

    id_camera = request.GET.get("id_camera")
    id_beach = request.GET.get("id_beach")
    since_ts = request.GET.get("since_timestamp")

    cams_qs = _filter_cams(id_camera, id_beach)

    all_rows = []
    last_timestamp_dt = None

    if since_ts is None or since_ts == "":
        cams_qs = _annotate_last_snapshot(cams_qs)
        cams = list(cams_qs)

        for cam in cams:
            if cam.last_pred_ts is None or cam.last_pred_count is None:
                continue

            ts_val = cam.last_pred_ts
            abs_val = float(cam.last_pred_count)

            all_rows.append((cam.beach.beach_slug, cam, ts_val, abs_val))
            last_timestamp_dt = max(last_timestamp_dt or ts_val, ts_val)

    else:
        cams = list(cams_qs)
        since_dt = datetime.fromtimestamp(int(since_ts), tz=dt_timezone.utc)

        limit = 10000

        snaps_qs = (
            Snapshot.objects
            .filter(webcam__in=cams)
            .filter(ts__gt=since_dt)
            .exclude(predicted_crowd_count__isnull=True)
            .select_related("webcam", "webcam__beach")
            .order_by("ts", "id")
        )

        boundary_row = list(snaps_qs.values_list("ts", flat=True)[limit - 1:limit])

        if not boundary_row:
            snaps = list(snaps_qs)
        else:
            boundary_ts = boundary_row[0]
            snaps = list(snaps_qs.filter(ts__lt=boundary_ts))

        for s in snaps:
            cam = s.webcam
            ts_val = s.ts
            abs_val = float(s.predicted_crowd_count)

            all_rows.append((cam.beach.beach_slug, cam, ts_val, abs_val))
            last_timestamp_dt = max(last_timestamp_dt or ts_val, ts_val)

    by_beach = {}
    for beach_slug, cam, ts_val, abs_val in all_rows:
        entry = by_beach.setdefault(beach_slug, {"playa_slug": beach_slug, "cameras": []})

        max_val = float(cam.max_crowd_count or 0)
        rel_val = (abs_val / max_val) if max_val > 0 else None

        entry["cameras"].append({
            "id": cam.id,
            "camera_slug": cam.camera_slug,
            "timestamp": _unix(ts_val),
            "occupancy_estimation_absolute": round(abs_val, 3),
            "occupancy_estimation_relative": round(rel_val, 6) if rel_val is not None else None,
        })

    data = []
    # calculo global por playa (sum abs / sum max para cams de esa playa)
    cams_all = list(cams_qs) if "cams_qs" in locals() else []
    for beach_slug, entry in by_beach.items():
        last_abs_by_cam = {c["id"]: float(c["occupancy_estimation_absolute"]) for c in entry["cameras"]}

        sum_abs = 0.0
        sum_max = 0.0
        for cam in cams_all:
            if cam.beach.beach_slug != beach_slug:
                continue
            max_val = float(cam.max_crowd_count or 0)
            if max_val <= 0:
                continue
            if cam.id not in last_abs_by_cam:
                continue
            sum_abs += last_abs_by_cam[cam.id]
            sum_max += max_val

        entry["global_occupancy_estimation_relative"] = round(sum_abs / sum_max, 6) if sum_max > 0 else 0
        data.append(entry)

    return JsonResponse({
        "last_timestamp": _unix(last_timestamp_dt) if last_timestamp_dt else None,
        "data": data
    })


@require_GET
def prediccio_futura(request: HttpRequest):
    auth_err = _require_jwt(request)
    if auth_err:
        return auth_err

    id_beach = request.GET.get("id_beach")  # beach_slug
    prediction_time_raw = request.GET.getlist("prediction_time")

    max_times = 100
    max_days_ahead = 7
    cache_ttl = 300

    try:
        prediction_times = api_utils.parse_prediction_times(prediction_time_raw, max_items=max_times)
    except ValueError as e:
        return JsonResponse({"detail": str(e)}, status=400)
    except Exception:
        return JsonResponse({"detail": "Invalid prediction_time. Provide unix timestamps."}, status=400)

    if not prediction_times:
        return JsonResponse({"detail": "prediction_time is required (list of unix timestamps)."}, status=400)

    cams_qs = WebCam.objects.select_related("beach").all()
    if id_beach:
        cams_qs = cams_qs.filter(beach__beach_slug=id_beach)

    cams = list(cams_qs)

    cams_by_beach = {}
    for cam in cams:
        cams_by_beach.setdefault(cam.beach.beach_slug, []).append(cam)

    try:
        preds_by_beach = api_utils.build_future_predictions(
            cams_by_beach=cams_by_beach,
            prediction_times=prediction_times,
            max_days_ahead=max_days_ahead,
            cache_ttl_seconds=cache_ttl,
        )
    except ValueError as e:
        return JsonResponse({"detail": str(e)}, status=400)

    now_ts = int(timezone.now().timestamp())
    data = [{"playa_slug": slug, "predictions": preds} for slug, preds in preds_by_beach.items()]

    return JsonResponse({
        "timestamp_prediction_generated_at": now_ts,
        "data": data
    })
