# apps/api/utils.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Iterable, List, Dict, Tuple

from django.core.cache import cache

from apps.webcam.models import WebCam
from apps.prediction.models import Snapshot


@dataclass(frozen=True)
class BeachBaseline:
    """Baseline (dummy) para una playa, calculada a partir del último snapshot de cada cámara."""
    beach_slug: str
    abs_total: float         # suma de abs por cámaras
    max_total: float         # suma de máximos de cámaras
    rel_total: float | None  # abs_total / max_total (si max_total > 0)


def unix_to_dt(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def dt_to_unix(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def parse_prediction_times(values: List[str], max_items: int = 100) -> List[int]:
    """
    Acepta prediction_time como:
      - repetido: ?prediction_time=1&prediction_time=2
      - o CSV:    ?prediction_time=1,2,3
    Devuelve lista de ints (unix timestamps).
    """
    if not values:
        return []

    items: List[str] = []
    for v in values:
        if v is None:
            continue
        v = v.strip()
        if not v:
            continue
        if "," in v:
            items.extend([a.strip() for a in v.split(",") if a.strip()])
        else:
            items.append(v)

    if len(items) > max_items:
        raise ValueError(f"prediction_time excede el máximo permitido ({max_items}).")

    out: List[int] = []
    for it in items:
        out.append(int(it))
    return out


def clamp_days_ahead(prediction_times: List[int], max_days_ahead: int) -> None:
    """
    Lanza ValueError si alguno de los timestamps está demasiado lejos en el futuro.
    """
    now = datetime.now(timezone.utc)
    max_dt = now + timedelta(days=max_days_ahead)
    for ts in prediction_times:
        dt = unix_to_dt(ts)
        if dt > max_dt:
            raise ValueError(
                f"timestamp {ts} supera el límite de {max_days_ahead} días de antelación."
            )


def _compute_beach_baseline(beach_slug: str, cams: List[WebCam]) -> BeachBaseline:
    """
    Predicción dummy:
      - coge el último predicted_crowd_count disponible por cada cámara
      - abs_total = suma(abs)
      - max_total = suma(max_crowd_count)
      - rel_total = abs_total / max_total
    """
    # Último snapshot por cámara de forma eficiente:
    # 1) traemos todos los últimos snapshots (por webcam) con una query por cámara sería N+1.
    # Aquí lo resolvemos con una query sobre Snapshot y luego nos quedamos con el último por webcam.
    snaps = (
        Snapshot.objects
        .filter(webcam__in=cams)
        .exclude(predicted_crowd_count__isnull=True)
        .order_by("-ts")
        .select_related("webcam")
    )

    # Nos quedamos con el primero (más reciente) por webcam_id
    last_abs_by_cam: Dict[int, float] = {}
    for s in snaps:
        cam_id = s.webcam_id
        if cam_id not in last_abs_by_cam:
            last_abs_by_cam[cam_id] = float(s.predicted_crowd_count)

    abs_total = 0.0
    max_total = 0.0
    for cam in cams:
        max_val = float(cam.max_crowd_count or 0)
        if max_val <= 0:
            continue
        if cam.id not in last_abs_by_cam:
            continue
        abs_total += last_abs_by_cam[cam.id]
        max_total += max_val

    rel_total = (abs_total / max_total) if max_total > 0 else None

    return BeachBaseline(
        beach_slug=beach_slug,
        abs_total=abs_total,
        max_total=max_total,
        rel_total=rel_total,
    )


def get_cached_baseline_for_beach(beach_slug: str, cams: List[WebCam], ttl_seconds: int = 300) -> BeachBaseline:
    """
    Cachea el baseline de la playa (dummy) para evitar recalcular en cada llamada.
    ttl_seconds: por defecto 5 minutos (ajustable).
    """
    cache_key = f"pred_futura:baseline:{beach_slug}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    baseline = _compute_beach_baseline(beach_slug, cams)
    cache.set(cache_key, baseline, ttl_seconds)
    return baseline


def build_future_predictions(
    cams_by_beach: Dict[str, List[WebCam]],
    prediction_times: List[int],
    max_days_ahead: int,
    cache_ttl_seconds: int = 300,
) -> Dict[str, List[Dict[str, float | int | None]]]:
    """
    Devuelve dict:
      beach_slug -> list(predictions)
    donde predictions = [{"timestamp_prediction_forecast": ts, "estimation_absolute": X, "estimation_relative": Y}, ...]
    """
    if prediction_times:
        clamp_days_ahead(prediction_times, max_days_ahead)

    out: Dict[str, List[Dict[str, float | int | None]]] = {}

    for beach_slug, cams in cams_by_beach.items():
        baseline = get_cached_baseline_for_beach(beach_slug, cams, ttl_seconds=cache_ttl_seconds)

        preds = []
        for ts in prediction_times:
            preds.append({
                "timestamp_prediction_forecast": ts,
                "estimation_absolute": round(baseline.abs_total, 3),
                "estimation_relative": round(baseline.rel_total, 6) if baseline.rel_total is not None else None,
            })
        out[beach_slug] = preds

    return out
