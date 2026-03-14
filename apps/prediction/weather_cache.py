"""
apps/prediction/weather_cache.py

Two-tier weather cache:
  - Archive (historical): permanent disk cache, keyed by monthly chunks.
    Only re-fetched if explicitly cleared.
  - Forecast (future): in-memory only, 1-hour TTL.

Usage:
    from apps.prediction.weather_cache import WeatherCache
    wc = WeatherCache()
    df = wc.get_for_period(lat, lon, start_date, end_date)
    wc.clear_archive()   # only when user asks
"""

import json
import logging
import pickle
import threading
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from django.conf import settings

logger = logging.getLogger(__name__)

OM_VAR_MAP = {
    'om_temperature_2m':          'temperature_2m',
    'om_apparent_temperature':    'apparent_temperature',
    'om_cloud_cover':             'cloud_cover',
    'om_cloud_cover_low':         'cloud_cover_low',
    'om_cloud_cover_mid':         'cloud_cover_mid',
    'om_cloud_cover_high':        'cloud_cover_high',
    'om_shortwave_radiation':     'shortwave_radiation',
    'om_direct_radiation':        'direct_radiation',
    'om_vapour_pressure_deficit': 'vapour_pressure_deficit',
    'om_dewpoint_2m':             'dewpoint_2m',
    'om_relative_humidity_2m':    'relative_humidity_2m',
    'om_precipitation':           'precipitation',
    'om_rain':                    'rain',
    'om_wind_speed_10m':          'wind_speed_10m',
    'om_wind_direction_10m':      'wind_direction_10m',
    'om_wind_gusts_10m':          'wind_gusts_10m',
    'om_pressure_msl':            'pressure_msl',
    'om_sunshine_duration':       'sunshine_duration',
}

FORECAST_TTL = 3600   # 1 hour
RATE_LIMIT_S  = 1.5   # seconds between API calls


def _default_cache_path() -> Path:
    return Path(getattr(settings, 'TFT_WEATHER_CACHE_PATH', 'weather_archive_cache.pkl'))


class WeatherCache:
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._archive: dict = {}          # (lat_r, lon_r, year, month) -> DataFrame
        self._forecast: dict = {}         # (lat_r, lon_r, past_days, fc_days) -> (ts, DataFrame)
        self._archive_lock = threading.Lock()
        self._forecast_lock = threading.Lock()
        self._api_lock = threading.Lock()
        self._last_call = 0.0
        self._path = _default_cache_path()
        self._load_disk()
        self._initialized = True

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_for_period(self, lat: float, lon: float, start_date: date, end_date: date,
                       context_days: int = 10) -> pd.DataFrame:
        """
        Return weather DataFrame covering [start_date - context_days, end_date].
        Uses archive for past dates, forecast API for near-future/future.
        """
        today = date.today()
        archive_cutoff = today - timedelta(days=2)

        full_start = start_date - timedelta(days=context_days)
        dfs = []

        # Archive portion
        if full_start <= archive_cutoff:
            arch_end = min(end_date, archive_cutoff)
            dfs.append(self._get_archive(lat, lon, full_start, arch_end))

        # Forecast portion (near-future or today's data not yet in archive)
        if end_date > archive_cutoff:
            past = max(0, min(10, (today - archive_cutoff).days + 1))
            fc_days = max(1, min(16, (end_date - today).days + 1))
            dfs.append(self._get_forecast(lat, lon, past_days=past, forecast_days=fc_days))

        if not dfs:
            return pd.DataFrame()

        df = (
            pd.concat(dfs, ignore_index=True)
            .drop_duplicates(subset='ds_real')
            .sort_values('ds_real')
            .reset_index(drop=True)
        )
        # Slice to the requested window
        start_ts = pd.Timestamp(full_start)
        end_ts   = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        return df[(df['ds_real'] >= start_ts) & (df['ds_real'] < end_ts)].reset_index(drop=True)

    def prefetch(self, lat: float, lon: float, start_date: date, end_date: date,
                 context_days: int = 15):
        """
        Pre-fetch and cache all archive data for a coord + date range.
        Used by evaluate_model_sets to front-load all API calls.
        """
        archive_cutoff = date.today() - timedelta(days=2)
        full_start = start_date - timedelta(days=context_days)
        arch_end = min(end_date, archive_cutoff)
        if full_start <= arch_end:
            self._get_archive(lat, lon, full_start, arch_end)

    def clear_archive(self):
        with self._archive_lock:
            self._archive.clear()
        if self._path.exists():
            self._path.unlink()
        logger.info("WeatherCache: archive cleared")

    def ensure_columns(self, df: pd.DataFrame, lat: float, lon: float,
                       start_date: date, end_date: date, required_cols: list) -> pd.DataFrame:
        """
        Check if required om_* columns are present in df.
        If any are missing, delete the entire cache and re-fetch.
        """
        missing = [c for c in required_cols if c.startswith('om_') and c not in df.columns]
        if not missing:
            return df

        logger.info(f"WeatherCache: missing columns {missing}, clearing full cache and re-fetching")
        self.clear_archive()

        return self.get_for_period(lat, lon, start_date, end_date)


        with self._archive_lock:
            self._archive.clear()
        if self._path.exists():
            self._path.unlink()
        logger.info("WeatherCache: archive cleared")

    def archive_info(self) -> dict:
        with self._archive_lock:
            keys = list(self._archive.keys())
        coords = set((k[0], k[1]) for k in keys)
        return {"entries": len(keys), "coords": len(coords)}

    # ── Archive (permanent disk) ───────────────────────────────────────────────

    def _get_archive(self, lat: float, lon: float,
                     start_date: date, end_date: date) -> pd.DataFrame:
        lat_r, lon_r = round(lat, 2), round(lon, 2)
        months = _months_between(start_date, end_date)
        dfs = []
        missing = []

        with self._archive_lock:
            for y, m in months:
                key = (lat_r, lon_r, y, m)
                if key in self._archive:
                    dfs.append(self._archive[key])
                else:
                    missing.append((y, m))

        for y, m in missing:
            month_start = date(y, m, 1)
            next_month  = date(y + (m // 12), (m % 12) + 1, 1)
            month_end   = next_month - timedelta(days=1)
            fetch_end   = min(month_end, date.today() - timedelta(days=2))
            if month_start > fetch_end:
                continue

            df = self._fetch_archive_api(lat_r, lon_r, month_start, fetch_end)
            key = (lat_r, lon_r, y, m)
            with self._archive_lock:
                self._archive[key] = df
                dfs.append(df)
            self._save_disk()

        if not dfs:
            return pd.DataFrame()

        df = (
            pd.concat(dfs, ignore_index=True)
            .drop_duplicates(subset='ds_real')
            .sort_values('ds_real')
            .reset_index(drop=True)
        )
        start_ts = pd.Timestamp(start_date)
        end_ts   = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        return df[(df['ds_real'] >= start_ts) & (df['ds_real'] < end_ts)].reset_index(drop=True)

    def _fetch_archive_api(self, lat_r, lon_r, start_date, end_date) -> pd.DataFrame:
        needed = list(OM_VAR_MAP.values())
        params = urllib.parse.urlencode({
            'latitude': lat_r, 'longitude': lon_r,
            'hourly': ','.join(needed),
            'start_date': str(start_date),
            'end_date':   str(end_date),
            'timezone': 'Europe/Madrid',
        })
        url = f'https://archive-api.open-meteo.com/v1/archive?{params}'
        logger.info(f"WeatherCache: archive fetch ({lat_r}, {lon_r}) {start_date} → {end_date}")
        return self._call_api(url)

    # ── Forecast (1-hour in-memory TTL) ───────────────────────────────────────

    def _get_forecast(self, lat: float, lon: float,
                      past_days: int = 10, forecast_days: int = 16) -> pd.DataFrame:
        lat_r, lon_r = round(lat, 2), round(lon, 2)
        key = (lat_r, lon_r, past_days, forecast_days)

        with self._forecast_lock:
            if key in self._forecast:
                ts, df = self._forecast[key]
                if time.time() - ts < FORECAST_TTL:
                    return df.copy()

        needed = list(OM_VAR_MAP.values())
        params = urllib.parse.urlencode({
            'latitude': lat_r, 'longitude': lon_r,
            'hourly': ','.join(needed),
            'forecast_days': forecast_days,
            'past_days': past_days,
            'timezone': 'Europe/Madrid',
        })
        url = f'https://api.open-meteo.com/v1/forecast?{params}'
        logger.info(f"WeatherCache: forecast fetch ({lat_r}, {lon_r}) past={past_days} fc={forecast_days}")
        df = self._call_api(url)

        with self._forecast_lock:
            self._forecast[key] = (time.time(), df)
        return df

    # ── Shared API call ───────────────────────────────────────────────────────

    def _call_api(self, url: str) -> pd.DataFrame:
        with self._api_lock:
            elapsed = time.time() - self._last_call
            if elapsed < RATE_LIMIT_S:
                time.sleep(RATE_LIMIT_S - elapsed)
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
            self._last_call = time.time()

        hourly = data['hourly']
        df = pd.DataFrame({'ds_real': pd.to_datetime(hourly['time'])})
        for om_name, api_name in OM_VAR_MAP.items():
            if api_name in hourly:
                df[om_name] = hourly[api_name]
        df['hour']        = df['ds_real'].dt.hour
        df['day_of_week'] = df['ds_real'].dt.dayofweek
        df['month']       = df['ds_real'].dt.month
        df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
        df['is_summer']   = df['month'].isin([6, 7, 8]).astype(int)
        return df

    # ── Disk persistence ──────────────────────────────────────────────────────

    def _load_disk(self):
        if self._path.exists():
            try:
                with open(self._path, 'rb') as f:
                    self._archive = pickle.load(f)
                logger.info(f"WeatherCache: loaded {len(self._archive)} archive entries from {self._path}")
            except Exception as e:
                logger.warning(f"WeatherCache: could not load disk cache ({e}), starting fresh")
                self._archive = {}

    def _save_disk(self):
        with self._archive_lock:
            data = dict(self._archive)
        try:
            with open(self._path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"WeatherCache: could not save disk cache: {e}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _months_between(start: date, end: date) -> list[tuple[int, int]]:
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months

weather_cache = WeatherCache()