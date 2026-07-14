from __future__ import annotations

from datetime import datetime, timezone as dt_timezone
import zoneinfo
_SPAIN_TZ = zoneinfo.ZoneInfo('Europe/Madrid')

import jwt
from django.conf import settings
from django.db.models import OuterRef, Subquery, DateTimeField, FloatField
from django.http import JsonResponse, HttpRequest
from django.utils import timezone
from django.views.decorators.http import require_GET

from apps.webcam.models import WebCam
from apps.prediction.models import Snapshot


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
    if not (request.user.is_authenticated and request.user.is_staff):
        auth_err = _require_jwt(request)
        if auth_err:
            return auth_err

    id_beach = request.GET.get("id_beach")
    prediction_time_raw = request.GET.getlist("prediction_time")

    MAX_TIMES      = 100
    MAX_DAYS_AHEAD = 15

    if not prediction_time_raw:
        return JsonResponse({"detail": "prediction_time is required (list of unix timestamps)."}, status=400)

    try:
        prediction_times = [int(t) for t in prediction_time_raw]
    except (ValueError, TypeError):
        return JsonResponse({"detail": "Invalid prediction_time. Provide unix timestamps."}, status=400)

    if len(prediction_times) > MAX_TIMES:
        return JsonResponse({"detail": f"Maximum {MAX_TIMES} prediction_time values allowed."}, status=400)

    now_utc = timezone.now()
    now_ts  = int(now_utc.timestamp())
    cutoff  = now_ts + MAX_DAYS_AHEAD * 86400

    prediction_times = [t for t in prediction_times if now_ts <= t <= cutoff]
    if not prediction_times:
        return JsonResponse(
            {"detail": f"All timestamps out of range. Must be within {MAX_DAYS_AHEAD} days from now."},
            status=400,
        )

    from apps.prediction.tft_service import tft_service

    cams_qs = WebCam.objects.select_related("beach")
    if id_beach:
        cams_qs = cams_qs.filter(beach__beach_slug=id_beach)

    cams_by_beach: dict[str, list] = {}
    for cam in cams_qs:
        cams_by_beach.setdefault(cam.beach.beach_slug, []).append(cam)

    if not cams_by_beach:
        return JsonResponse({"detail": "No cameras found for the requested beach(es)."}, status=404)

    local_tz = _SPAIN_TZ
    data = []

    for beach_slug, cams in cams_by_beach.items():
        cam_pred_maps: list[tuple[float, dict]] = []
        for cam in cams:
            mcc = float(cam.max_crowd_count or 0)
            if mcc <= 0:
                continue
            try:
                result = tft_service.predict_mixed(cam, days=MAX_DAYS_AHEAD)
            except Exception:
                continue
            pred_map: dict[str, float] = {
                p["timestamp"][:16]: float(p["crowd_count"])
                for p in result.get("predictions", [])
                if p.get("available") and p.get("crowd_count") is not None
            }
            cam_pred_maps.append((mcc, pred_map))

        if not cam_pred_maps:
            continue

        predictions = []
        for ts in sorted(prediction_times):
            # predict_mixed keys forecasts in Django localtime (TIME_ZONE='UTC'), so build the
            # lookup key in that same tz — converting to Europe/Madrid here shifts it by the
            # UTC offset (1–2 h) and every key misses, returning null estimations.
            dt  = timezone.localtime(datetime.fromtimestamp(ts, tz=dt_timezone.utc))
            key = dt.strftime("%Y-%m-%dT%H:00")

            sum_cc_x_max = 0.0
            sum_cc       = 0.0
            sum_max      = 0.0
            any_hit      = False
            for max_crowd, pred_map in cam_pred_maps:
                if key in pred_map:
                    cc            = pred_map[key]
                    sum_cc       += cc
                    sum_cc_x_max += cc * max_crowd
                    sum_max      += max_crowd
                    any_hit       = True

            rel_val = round(sum_cc / sum_max * 100, 2)  if (any_hit and sum_max > 0) else None
            abs_val = round(sum_cc_x_max / sum_max, 3)  if (any_hit and sum_max > 0) else None

            predictions.append({
                "timestamp_prediction_forecast": ts,
                "estimation_absolute":  abs_val,
                "estimation_relative":  rel_val,
            })

        data.append({"playa_slug": beach_slug, "predictions": predictions})

    return JsonResponse({
        "timestamp_prediction_generated_at": now_ts,
        "data": data,
    })