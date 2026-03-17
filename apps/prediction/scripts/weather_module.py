"""
Unified weather module — Open-Meteo only.

Two APIs:
  get_weather_at_point(lat, lon, dt)   -> dict  (used by build_dataset)
  get_for_period(lat, lon, start, end) -> DataFrame (used by tft_service)

Cache strategy:
  - Archive (past dates): monthly JSON chunks in cache/weather/archive/
  - Forecast (future):    in-memory, 1-hour TTL
  - Point results:        JSON files in cache/weather/results/
"""

import hashlib
import json
import logging
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from django.conf import settings as _settings
    CACHE_DIR = Path(_settings.WEATHER_CACHE_DIR)
except Exception:
    CACHE_DIR = Path(__file__).resolve().parent.parent / 'cache' / 'weather'

ARCHIVE_DIR = CACHE_DIR / 'archive'
RESULTS_DIR = CACHE_DIR / 'results'

for _d in [CACHE_DIR, ARCHIVE_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

OPENMETEO_VARS = [
    'temperature_2m', 'apparent_temperature', 'dewpoint_2m', 'relative_humidity_2m',
    'pressure_msl', 'precipitation', 'rain', 'wind_speed_10m', 'wind_direction_10m',
    'wind_gusts_10m', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high',
    'visibility', 'uv_index', 'uv_index_clear_sky', 'sunshine_duration',
    'evapotranspiration', 'vapour_pressure_deficit', 'direct_radiation', 'shortwave_radiation'
]

RATE_LIMIT_S = 1.5
FORECAST_TTL = 3600

_api_lock     = threading.Lock()
_last_call    = 0.0
_archive_lock = threading.Lock()
_forecast_lock = threading.Lock()
_archive_mem: dict = {}
_forecast_mem: dict = {}


def _parse_datetime(target_dt):
    if isinstance(target_dt, str):
        return pd.to_datetime(target_dt).to_pydatetime()
    elif isinstance(target_dt, date) and not isinstance(target_dt, datetime):
        return datetime.combine(target_dt, datetime.min.time())
    return target_dt


def _months_between(start: date, end: date) -> list:
    months, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


def _call_api(url: str) -> dict:
    global _last_call
    with _api_lock:
        elapsed = time.time() - _last_call
        if elapsed < RATE_LIMIT_S:
            time.sleep(RATE_LIMIT_S - elapsed)
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.loads(resp.read())
        _last_call = time.time()
    return data


def _hourly_to_df(data: dict) -> pd.DataFrame:
    hourly = data['hourly']
    df = pd.DataFrame({'ds_real': pd.to_datetime(hourly['time'])})
    for var in OPENMETEO_VARS:
        if var in hourly:
            df[f'om_{var}'] = hourly[var]
    df['hour']       = df['ds_real'].dt.hour
    df['day_of_week'] = df['ds_real'].dt.dayofweek
    df['month']      = df['ds_real'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_summer']  = df['month'].isin([6, 7, 8]).astype(int)
    return df


def _archive_path(lat_r: float, lon_r: float, y: int, m: int) -> Path:
    return ARCHIVE_DIR / f"{lat_r:.2f}_{lon_r:.2f}_{y}_{m:02d}.json"


def _load_archive_month(lat_r, lon_r, y, m):
    key = (lat_r, lon_r, y, m)
    with _archive_lock:
        if key in _archive_mem:
            return _archive_mem[key]
    path = _archive_path(lat_r, lon_r, y, m)
    if path.exists():
        try:
            df = _hourly_to_df(json.loads(path.read_text()))
            with _archive_lock:
                _archive_mem[key] = df
            return df
        except Exception as e:
            logger.warning(f"[Weather] Archive read error {path}: {e}")
    return None


def _fetch_archive_month(lat_r, lon_r, y, m):
    start = date(y, m, 1)
    next_m = date(y + (m // 12), (m % 12) + 1, 1)
    end = min(next_m - timedelta(days=1), date.today() - timedelta(days=2))
    if start > end:
        return pd.DataFrame()

    params = urllib.parse.urlencode({
        'latitude': lat_r, 'longitude': lon_r,
        'hourly': ','.join(OPENMETEO_VARS),
        'start_date': str(start), 'end_date': str(end),
        'timezone': 'Europe/Madrid',
    })
    url = f'https://archive-api.open-meteo.com/v1/archive?{params}'
    logger.info(f"[Weather] Fetch archive ({lat_r}, {lon_r}) {start} -> {end}")
    try:
        data = _call_api(url)
    except Exception as e:
        logger.error(f"[Weather] Archive fetch failed: {e}")
        return pd.DataFrame()

    if 'error' in data:
        logger.error(f"[Weather] API error: {data.get('reason')}")
        return pd.DataFrame()

    _archive_path(lat_r, lon_r, y, m).write_text(json.dumps(data))
    df = _hourly_to_df(data)
    with _archive_lock:
        _archive_mem[(lat_r, lon_r, y, m)] = df
    return df


def _get_archive(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    lat_r, lon_r = round(lat, 2), round(lon, 2)
    dfs = []
    for y, m in _months_between(start, end):
        cached = _load_archive_month(lat_r, lon_r, y, m)
        df = cached if cached is not None else _fetch_archive_month(lat_r, lon_r, y, m)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    df = (pd.concat(dfs, ignore_index=True)
          .drop_duplicates(subset='ds_real')
          .sort_values('ds_real')
          .reset_index(drop=True))
    s, e = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    return df[(df['ds_real'] >= s) & (df['ds_real'] < e)].reset_index(drop=True)


def _get_forecast(lat: float, lon: float, past_days: int = 10, forecast_days: int = 16) -> pd.DataFrame:
    lat_r, lon_r = round(lat, 2), round(lon, 2)
    key = (lat_r, lon_r, past_days, forecast_days)
    with _forecast_lock:
        if key in _forecast_mem:
            ts, df = _forecast_mem[key]
            if time.time() - ts < FORECAST_TTL:
                return df.copy()

    params = urllib.parse.urlencode({
        'latitude': lat_r, 'longitude': lon_r,
        'hourly': ','.join(OPENMETEO_VARS),
        'forecast_days': forecast_days, 'past_days': past_days,
        'timezone': 'Europe/Madrid',
    })
    url = f'https://api.open-meteo.com/v1/forecast?{params}'
    logger.info(f"[Weather] Fetch forecast ({lat_r}, {lon_r}) past={past_days} fc={forecast_days}")
    try:
        data = _call_api(url)
    except Exception as e:
        logger.error(f"[Weather] Forecast fetch failed: {e}")
        return pd.DataFrame()

    df = _hourly_to_df(data)
    with _forecast_lock:
        _forecast_mem[key] = (time.time(), df)
    return df


# ── Public: DataFrame API (tft_service) ───────────────────────────────────────

def get_for_period(lat: float, lon: float, start_date: date, end_date: date,
                   context_days: int = 10) -> pd.DataFrame:
    today = date.today()
    archive_cutoff = today - timedelta(days=2)
    full_start = start_date - timedelta(days=context_days)
    dfs = []

    if full_start <= archive_cutoff:
        dfs.append(_get_archive(lat, lon, full_start, min(end_date, archive_cutoff)))

    if end_date > archive_cutoff:
        past = max(0, min(10, (today - archive_cutoff).days + 1))
        fc_days = max(1, min(16, (end_date - today).days + 1))
        dfs.append(_get_forecast(lat, lon, past_days=past, forecast_days=fc_days))

    if not dfs:
        return pd.DataFrame()

    df = (pd.concat(dfs, ignore_index=True)
          .drop_duplicates(subset='ds_real')
          .sort_values('ds_real')
          .reset_index(drop=True))
    s = pd.Timestamp(full_start)
    e = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    return df[(df['ds_real'] >= s) & (df['ds_real'] < e)].reset_index(drop=True)


def prefetch(lat: float, lon: float, start_date: date, end_date: date, context_days: int = 15):
    archive_cutoff = date.today() - timedelta(days=2)
    full_start = start_date - timedelta(days=context_days)
    arch_end = min(end_date, archive_cutoff)
    if full_start <= arch_end:
        _get_archive(lat, lon, full_start, arch_end)


def archive_info() -> dict:
    with _archive_lock:
        keys = list(_archive_mem.keys())
    return {
        'mem_entries': len(keys),
        'disk_entries': len(list(ARCHIVE_DIR.glob('*.json'))),
        'coords': len(set((k[0], k[1]) for k in keys)),
    }


def clear_archive():
    with _archive_lock:
        _archive_mem.clear()
    for f in ARCHIVE_DIR.glob('*.json'):
        f.unlink()
    for f in RESULTS_DIR.glob('*.json'):
        f.unlink()
    logger.info("[Weather] Archive cleared")


# ── Public: dict API (build_dataset) ──────────────────────────────────────────

def get_weather_at_point(lat: float, lon: float, target_dt, use_cache: bool = True) -> dict:
    target_dt = _parse_datetime(target_dt)
    target_date = target_dt.date()
    target_hour = target_dt.hour

    if use_cache:
        key = hashlib.md5(f"{lat:.4f}_{lon:.4f}_{target_date}_{target_hour}".encode()).hexdigest()
        cache_file = RESULTS_DIR / f"{key}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                pass

    df = _get_archive(lat, lon, target_date - timedelta(days=1), target_date + timedelta(days=1))
    if df.empty:
        return {}

    mask = (df['ds_real'].dt.date == target_date) & (abs(df['ds_real'].dt.hour - target_hour) <= 1)
    if not mask.any():
        mask = df['ds_real'].dt.date == target_date

    subset = df[mask]
    if subset.empty:
        return {}

    result = {}
    for col in [c for c in subset.columns if c.startswith('om_')]:
        vals = subset[col].dropna().values
        if len(vals) > 0:
            result[col] = round(float(np.mean(vals)), 4)

    if use_cache and result:
        key = hashlib.md5(f"{lat:.4f}_{lon:.4f}_{target_date}_{target_hour}".encode()).hexdigest()
        (RESULTS_DIR / f"{key}.json").write_text(json.dumps(result))

    return result


def check_connectivity() -> dict:
    try:
        params = urllib.parse.urlencode({
            'latitude': 39.5, 'longitude': 2.5,
            'start_date': '2022-07-15', 'end_date': '2022-07-16',
            'hourly': 'temperature_2m'
        })
        with urllib.request.urlopen(
            f'https://archive-api.open-meteo.com/v1/archive?{params}', timeout=10
        ) as r:
            return {"status": "ok", "code": r.status}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == '__main__':
    lat, lon = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) >= 3 else (39.7325522, 3.2400431)
    t0 = time.time()
    result = get_weather_at_point(lat, lon, datetime(2022, 10, 26, 14, 0))
    print(json.dumps(result, indent=2))
    print(f"\nTime: {time.time() - t0:.3f}s | Archive: {archive_info()}")