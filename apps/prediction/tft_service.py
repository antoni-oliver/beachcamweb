"""
TFT Prediction Service — loads 3 pre-trained models and serves predictions.

Models: 3-day (H=36), 14-day (H=168), 30-day (H=360)
Auto-selects model based on requested horizon.
"""

import json
import logging
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)

HOURS_PER_DAY = 12
HOUR_MIN, HOUR_MAX = 8, 19

OCCUPANCY_THRESHOLDS = [
    (0.75, 'HIGH'),
    (0.50, 'MEDIUM'),
    (0.25, 'LOW'),
    (0.0,  'VERY_LOW'),
]


def classify_occupancy(crowd_count, max_crowd_count):
    if not max_crowd_count or max_crowd_count <= 0:
        return None
    ratio = crowd_count / max_crowd_count
    for threshold, level in OCCUPANCY_THRESHOLDS:
        if ratio >= threshold:
            return level
    return 'VERY_LOW'

OM_VAR_MAP = {
    'om_temperature_2m': 'temperature_2m',
    'om_cloud_cover_low': 'cloud_cover_low',
    'om_shortwave_radiation': 'shortwave_radiation',
    'om_vapour_pressure_deficit': 'vapour_pressure_deficit',
}

MODEL_CONFIGS = {
    '3d':  {'dir': 'tft_model_3d',  'horizon': 36,  'max_days': 3},
    '10d': {'dir': 'tft_model_10d', 'horizon': 120, 'max_days': 10},
    '15d': {'dir': 'tft_model_15d', 'horizon': 180, 'max_days': 15},
}


class TFTService:
    _instance = None
    _lock = threading.Lock()
    _predict_lock = threading.Lock()
    _weather_lock = threading.Lock()
    _last_weather_call = 0
    _weather_cache = {}
    _prediction_cache = {}
    _prediction_cache_lock = threading.Lock()
    PREDICTION_TTL = 3600
    WEATHER_TTL = 300

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.models = {}          # default set — backward compat
        self.model_sets = {}      # name -> {3d: ..., 10d: ..., 15d: ...}
        self._initialized = True

    def _load_single_model(self, key, model_dir, max_days):
        from neuralforecast import NeuralForecast
        model_dir = Path(model_dir)

        with open(model_dir / 'config.json') as f:
            config = json.load(f)

        nf = NeuralForecast.load(str(model_dir / 'nf_model'))

        BAD_KEYS = ['training_data_availability_threshold']
        for model_obj in nf.models:
            for attr in ['trainer_kwargs', 'pred_trainer_kwargs']:
                d = getattr(model_obj, attr, None)
                if isinstance(d, dict):
                    for k in BAD_KEYS:
                        d.pop(k, None)
                    d['logger'] = False
                    d['enable_progress_bar'] = False
            if hasattr(model_obj, 'hparams'):
                for k in BAD_KEYS:
                    model_obj.hparams.pop(k, None)

        static = pd.read_csv(model_dir / 'static_features.csv')
        per_beach_path = model_dir / 'per_beach.csv'
        per_beach = pd.read_csv(per_beach_path) if per_beach_path.exists() else pd.DataFrame()

        all_features = config['futr_exog'] + config['hist_exog']
        fi_path = model_dir / 'feature_importance.json'
        if fi_path.exists():
            with open(fi_path) as f:
                raw_fi = json.load(f)
            first_key = next(iter(raw_fi))
            if isinstance(first_key, int) or (isinstance(first_key, str) and first_key.isdigit()):
                feature_importance = {
                    all_features[int(k)]: v
                    for k, v in raw_fi.items()
                    if int(k) < len(all_features)
                }
            else:
                feature_importance = raw_fi
        else:
            feature_importance = {f: round(1.0 / len(all_features), 4) for f in all_features}

        return {
            'nf': nf,
            'config': config,
            'static': static,
            'per_beach': per_beach,
            'horizon': config['horizon'],
            'input_size': config['input_size'],
            'futr': config['futr_exog'],
            'hist': config['hist_exog'],
            'max_days': max_days,
            'stat_cols': list(static.columns.drop('unique_id')),
            'feature_importance': feature_importance,
            'model_dir': str(model_dir),
        }

    def load_models(self, base_dir=None, set_name='default'):
        base_dir = Path(base_dir or getattr(settings, 'TFT_MODELS_DIR', 'tft_models'))

        # Auto-discover horizon dirs — tries known patterns then glob
        def find_dir(horizon_tag):
            patterns = [
                f'tft_model_{horizon_tag}',
                f'tft_grid_model_full_{horizon_tag}',
                f'tft_grid_model_cutoff_{horizon_tag}',
            ]
            for pattern in patterns:
                p = base_dir / pattern
                if (p / 'config.json').exists():
                    return p
            # glob fallback: any dir containing the horizon tag
            for p in sorted(base_dir.glob(f'*{horizon_tag}*')):
                if p.is_dir() and (p / 'config.json').exists():
                    return p
            return None

        max_days_map = {'3d': 3, '10d': 10, '15d': 15}
        loaded = {}
        for key in ['3d', '10d', '15d']:
            model_dir = find_dir(key)
            if model_dir is None:
                logger.warning(f"[{set_name}] Model {key} not found in {base_dir}, skipping")
                continue
            loaded[key] = self._load_single_model(key, model_dir, max_days_map[key])
            logger.info(f"[{set_name}] Loaded {key} (H={loaded[key]['horizon']}) from {model_dir.name}")

        self.model_sets[set_name] = loaded
        if set_name == 'default':
            self.models = loaded

        logger.info(f"TFT set '{set_name}': {len(loaded)} models loaded")
        return loaded

    def list_model_sets(self):
        loaded = {
            name: {k: {'horizon': m['horizon'], 'max_days': m['max_days'], 'dir': m['model_dir']}
                   for k, m in models.items()}
            for name, models in self.model_sets.items()
        }
        # Include configured but not yet loaded sets
        configured = getattr(settings, 'TFT_MODEL_SETS', {})
        for name in configured:
            if name not in loaded:
                loaded[name] = {}  # will lazy-load on first predict
        return loaded

    def select_model(self, days, model_set='default'):
        models = self.model_sets.get(model_set, self.models)
        for key in ['3d', '10d', '15d']:
            if key in models and days <= models[key]['max_days']:
                return key
        available = [k for k in ['15d', '10d', '3d'] if k in models]
        return available[0] if available else None

    def _ensure_loaded(self, model_set='default'):
        if model_set not in self.model_sets:
            if model_set == 'default':
                logger.info("TFT: lazy loading default models...")
                self.load_models()
            else:
                configured = getattr(settings, 'TFT_MODEL_SETS', {})
                if model_set in configured:
                    logger.info(f"TFT: lazy loading model set '{model_set}'...")
                    self.load_models(base_dir=configured[model_set], set_name=model_set)
                else:
                    raise RuntimeError(f"Model set '{model_set}' not found. Available: {list(self.model_sets.keys())}")

    def predict(self, webcam, days=3, since=None, model_set='default'):
        self._ensure_loaded(model_set)
        if model_set not in self.model_sets:
            raise RuntimeError(f"Model set '{model_set}' not loaded")
        models = self.model_sets.get(model_set, self.models)
        model_key = self.select_model(days, model_set=model_set)
        if model_key is None:
            raise RuntimeError(f"No TFT models loaded for set '{model_set}'")

        if not since:
            cache_key = (webcam.camera_slug, days, model_set)
            with self._prediction_cache_lock:
                if cache_key in self._prediction_cache:
                    cached_time, cached_result = self._prediction_cache[cache_key]
                    if time.time() - cached_time < self.PREDICTION_TTL:
                        logger.debug(f"Prediction cache hit: {cache_key}")
                        return cached_result

        m = models[model_key]
        H_model = m['horizon']                          # full model horizon for futr_df
        H_out = min(days * HOURS_PER_DAY, H_model)     # steps to include in output

        lat = float(webcam.camera_latitude or webcam.beach.coordenadas_geograficas.split(',')[0] if webcam.beach.coordenadas_geograficas else 39.5)
        lon = float(webcam.camera_longitude or webcam.beach.coordenadas_geograficas.split(',')[1] if webcam.beach.coordenadas_geograficas else 2.6)

        if since:
            since_naive = since.replace(tzinfo=None) if hasattr(since, 'tzinfo') and since.tzinfo else since
            weather_all = self._fetch_weather_for_period(lat, lon, since_naive, days, context_days=10)
        else:
            weather_all = self._fetch_weather_forecast(lat, lon, past_days=10, forecast_days=16)

        context = self._build_context(webcam, m, weather_all, since=since)

        if not since:
            context_start = context['ds_real'].iloc[0]
            weather_start = weather_all['ds_real'].min()
            if context_start < weather_start:
                context_weather = self._fetch_weather_for_period(
                    lat, lon, context_start, days, context_days=2
                )
                context = self._build_context(webcam, m, context_weather, since=since)
                weather_all = pd.concat([context_weather, weather_all], ignore_index=True)
                weather_all = weather_all.drop_duplicates(subset='ds_real').sort_values('ds_real').reset_index(drop=True)

        last_real = context['ds_real'].iloc[-1]

        weather_daytime = self._filter_daytime(weather_all)
        futr_df = self._build_future(context, weather_daytime, m, H_model)

        uid = webcam.camera_slug
        train_cols = ['unique_id', 'ds', 'y'] + m['futr'] + m['hist']
        context_df = context[train_cols].copy()

        static_df = self._make_static(uid, context, m['stat_cols'])

        with self._predict_lock:
            forecast = m['nf'].predict(
                df=context_df,
                static_df=static_df,
                futr_df=futr_df[['unique_id', 'ds'] + m['futr']],
            )
        forecast = forecast.reset_index()
        forecast['TFT'] = forecast['TFT'].clip(lower=0)

        max_cc = webcam.max_crowd_count or 0
        predictions = self._format_output(forecast, weather_all, last_real, H_out, days, max_cc)

        result = {
            'model': model_key,
            'model_set': model_set,
            'beach': webcam.beach.beach_name,
            'webcam': uid,
            'horizon_days': days,
            'horizon_hours': H_out,
            'max_crowd_count': max_cc,
            'last_data': last_real.strftime('%Y-%m-%dT%H:%M:%S'),
            'feature_importance': m['feature_importance'],
            'predictions': predictions,
        }

        if not since:
            with self._prediction_cache_lock:
                self._prediction_cache[(webcam.camera_slug, days, model_set)] = (time.time(), result)

        return result

    def _build_context(self, webcam, m, weather_all, since=None):
        from apps.prediction.models import Snapshot

        input_size = m['input_size']
        uid = webcam.camera_slug

        qs = Snapshot.objects.filter(webcam=webcam, predicted_crowd_count__isnull=False)
        if since:
            qs = qs.filter(ts__lt=since)
        snapshots = list(qs.order_by('-ts')[:input_size * 3])[::-1]

        if len(snapshots) < 12:
            raise ValueError(f"Not enough history for {uid}: {len(snapshots)} snapshots")

        rows = []
        for s in snapshots:
            ts = timezone.localtime(s.ts).replace(tzinfo=None)
            if ts.hour < HOUR_MIN or ts.hour > HOUR_MAX:
                continue
            rows.append({
                'ds_real': ts,
                'y': float(s.predicted_crowd_count),
                'hour': ts.hour,
                'day_of_week': ts.weekday(),
                'month': ts.month,
                'is_weekend': int(ts.weekday() >= 5),
                'is_summer': int(ts.month in (6, 7, 8)),
            })

        df = pd.DataFrame(rows)
        df['unique_id'] = uid
        df = df.sort_values('ds_real').tail(input_size).reset_index(drop=True)
        df['ds'] = range(len(df))

        df = self._merge_weather(df, self._filter_daytime(weather_all))

        all_features = m['futr'] + m['hist']
        for col in all_features:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].astype(float)
        df['ds'] = df['ds'].astype(int)

        return df

    def _fetch_weather_forecast(self, lat, lon, past_days=10, forecast_days=16):
        cache_key = ('forecast', round(lat, 2), round(lon, 2), past_days, forecast_days)
        with self._weather_lock:
            if cache_key in self._weather_cache:
                cached_time, cached_df = self._weather_cache[cache_key]
                if time.time() - cached_time < self.WEATHER_TTL:
                    return cached_df.copy()
            self._rate_limit()

        needed = list(OM_VAR_MAP.values())
        params = urllib.parse.urlencode({
            'latitude': lat, 'longitude': lon,
            'hourly': ','.join(needed),
            'forecast_days': forecast_days,
            'past_days': past_days,
            'timezone': 'Europe/Madrid',
        })
        url = f'https://api.open-meteo.com/v1/forecast?{params}'
        df = self._parse_weather_response(url)
        self._weather_cache[cache_key] = (time.time(), df)
        return df

    def _fetch_weather_archive(self, lat, lon, start_date, end_date):
        cache_key = ('archive', round(lat, 2), round(lon, 2), str(start_date), str(end_date))
        with self._weather_lock:
            if cache_key in self._weather_cache:
                cached_time, cached_df = self._weather_cache[cache_key]
                if time.time() - cached_time < self.WEATHER_TTL:
                    return cached_df.copy()
            self._rate_limit()

        needed = list(OM_VAR_MAP.values())
        params = urllib.parse.urlencode({
            'latitude': lat, 'longitude': lon,
            'hourly': ','.join(needed),
            'start_date': str(start_date),
            'end_date': str(end_date),
            'timezone': 'Europe/Madrid',
        })
        url = f'https://archive-api.open-meteo.com/v1/archive?{params}'
        df = self._parse_weather_response(url)
        self._weather_cache[cache_key] = (time.time(), df)
        return df

    def _fetch_weather_for_period(self, lat, lon, since_naive, forecast_days, context_days=10):
        from datetime import timedelta
        since_date = pd.Timestamp(since_naive).date()
        start_date = since_date - timedelta(days=context_days)
        end_date = since_date + timedelta(days=forecast_days)
        today = pd.Timestamp.now().date()

        archive_end = min(end_date, today - timedelta(days=2))
        dfs = []
        if start_date <= archive_end:
            dfs.append(self._fetch_weather_archive(lat, lon, start_date, archive_end))

        if end_date > archive_end:
            past = max(0, (today - archive_end).days + 1)
            past = min(past, 10)
            fc_days = max(1, (end_date - today).days + 1)
            fc_days = min(fc_days, 16)
            dfs.append(self._fetch_weather_forecast(lat, lon, past_days=past, forecast_days=fc_days))

        if not dfs:
            return pd.DataFrame()
        df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='ds_real').sort_values('ds_real').reset_index(drop=True)
        return df

    def _rate_limit(self):
        elapsed = time.time() - self._last_weather_call
        if elapsed < 1.5:
            time.sleep(1.5 - elapsed)
        self._last_weather_call = time.time()

    def _parse_weather_response(self, url):
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        hourly = data['hourly']
        df = pd.DataFrame({'ds_real': pd.to_datetime(hourly['time'])})
        for om_name, api_name in OM_VAR_MAP.items():
            if api_name in hourly:
                df[om_name] = hourly[api_name]
        df['hour'] = df['ds_real'].dt.hour
        df['day_of_week'] = df['ds_real'].dt.dayofweek
        df['month'] = df['ds_real'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        return df

    def _filter_daytime(self, weather_df):
        return weather_df[(weather_df['hour'] >= HOUR_MIN) & (weather_df['hour'] <= HOUR_MAX)]

    def _merge_weather(self, panel_df, weather_df):
        panel_df = panel_df.copy()
        weather_df = weather_df.sort_values('ds_real')

        for _, row in panel_df.iterrows():
            idx = row.name
            ts = row['ds_real']
            diffs = (weather_df['ds_real'] - ts).abs()
            nearest = diffs.idxmin()
            for col in OM_VAR_MAP.keys():
                if col in weather_df.columns:
                    panel_df.loc[idx, col] = weather_df.loc[nearest, col]

        return panel_df

    def _build_future(self, context, weather, m, H):
        uid = context['unique_id'].iloc[0]
        last_ds = int(context['ds'].max())
        last_real = context['ds_real'].iloc[-1]

        future_weather = weather[weather['ds_real'] > last_real].head(H)
        all_features = m['futr'] + m['hist']

        rows = []
        for step in range(1, H + 1):
            row = {'unique_id': uid, 'ds': last_ds + step}
            if step <= len(future_weather):
                w = future_weather.iloc[step - 1]
                for col in all_features:
                    row[col] = float(w[col]) if col in w.index and pd.notna(w[col]) else 0.0
            else:
                for col in all_features:
                    row[col] = rows[-1].get(col, 0.0) if rows else 0.0
            rows.append(row)

        futr_df = pd.DataFrame(rows)
        for col in all_features:
            futr_df[col] = futr_df[col].astype(float)
        futr_df['ds'] = futr_df['ds'].astype(int)
        return futr_df

    def _make_static(self, uid, context, stat_cols):
        y = context['y']
        row = {
            'unique_id': uid,
            'stat_mean_y': float(y.mean()),
            'stat_cv': float(y.std() / max(y.mean(), 1)),
        }
        for col in stat_cols:
            if col not in row:
                row[col] = 0.0
        return pd.DataFrame([row])

    def _format_output(self, forecast, weather, last_real, H, days, max_crowd_count):
        daytime_preds = [float(forecast.iloc[i]['TFT']) for i in range(min(H, len(forecast)))]

        last_real_floored = last_real.replace(minute=0, second=0, microsecond=0)

        if last_real_floored.hour >= HOUR_MAX:
            first_forecast = (last_real_floored + pd.Timedelta(days=1)).normalize().replace(hour=HOUR_MIN)
        else:
            first_forecast = last_real_floored + pd.Timedelta(hours=1)
            if first_forecast.hour < HOUR_MIN:
                first_forecast = first_forecast.normalize().replace(hour=HOUR_MIN)

        pred_map = {}
        ts = first_forecast
        for i, val in enumerate(daytime_preds):
            pred_map[ts.strftime('%Y-%m-%dT%H:00')] = val
            ts = ts + pd.Timedelta(hours=1)
            if ts.hour > HOUR_MAX:
                ts = (ts + pd.Timedelta(days=1)).normalize().replace(hour=HOUR_MIN)

        FEATURE_LABELS = {
            'om_temperature_2m': 'Temperature (°C)',
            'om_shortwave_radiation': 'Solar Radiation (W/m²)',
            'om_cloud_cover_low': 'Low Cloud Cover (%)',
            'om_vapour_pressure_deficit': 'Vapour Pressure Deficit (kPa)',
            'is_weekend': 'Weekend',
            'is_summer': 'Summer',
        }

        weather_map = {}
        for _, row in weather.iterrows():
            key = row['ds_real'].strftime('%Y-%m-%dT%H:00')
            entry = {}
            if pd.notna(row.get('om_temperature_2m')):
                entry['temperature'] = round(float(row['om_temperature_2m']), 1)
            for col in ['om_shortwave_radiation', 'om_cloud_cover_low', 'om_vapour_pressure_deficit']:
                if col in row and pd.notna(row[col]):
                    entry[col] = round(float(row[col]), 2)
            weather_map[key] = entry

        start_day = first_forecast.normalize()
        predictions = []

        for day_offset in range(days):
            current_day = start_day + pd.Timedelta(days=day_offset)
            for hour in range(24):
                ts = current_day.replace(hour=hour)
                key = ts.strftime('%Y-%m-%dT%H:00')
                w = weather_map.get(key, {})
                temp = w.get('temperature')

                if key in pred_map:
                    cc = round(pred_map[key], 1)
                    level = classify_occupancy(cc, max_crowd_count)
                    features = {FEATURE_LABELS.get(k, k): v for k, v in w.items()}
                    features['Weekend'] = int(ts.weekday() >= 5)
                    features['Summer'] = int(ts.month in (6, 7, 8))
                    predictions.append({
                        'timestamp': ts.isoformat(),
                        'hour': hour,
                        'crowd_count': cc,
                        'available': True,
                        'occupancy_level': level,
                        'occupancy_ratio': round(cc / max_crowd_count, 3) if max_crowd_count > 0 else None,
                        'temperature': temp,
                        'features': features,
                    })
                else:
                    predictions.append({
                        'timestamp': ts.isoformat(),
                        'hour': hour,
                        'crowd_count': 0,
                        'available': False,
                        'occupancy_level': None,
                        'occupancy_ratio': None,
                        'temperature': temp,
                        'features': {},
                    })

        return predictions

    def get_metrics(self, webcam_slug=None, model_set='default'):
        models = self.model_sets.get(model_set, self.models)
        result = {}
        for key, m in models.items():
            config = m['config']
            per_beach = m['per_beach']

            model_info = {
                'horizon': config.get('horizon'),
                'horizon_days': config.get('horizon', 0) // HOURS_PER_DAY,
                'input_size': config.get('input_size'),
                'overall_relMAE': config.get('overall_relMAE') or config.get('relMAE'),
                'season_relMAE': config.get('season_relMAE'),
            }

            if not per_beach.empty and webcam_slug:
                match = per_beach[per_beach['unique_id'].str.contains(webcam_slug, na=False)]
                if not match.empty:
                    row = match.iloc[0]
                    model_info['beach_relMAE'] = round(float(row.get('relMAE', row.get('season_relMAE', 0))), 4) if 'relMAE' in row or 'season_relMAE' in row else None
                    model_info['beach_MAE'] = round(float(row['MAE']), 2) if 'MAE' in row else None
                    model_info['beach_RMSE'] = round(float(row['RMSE']), 2) if 'RMSE' in row else None
                    model_info['beach_mean_y'] = round(float(row['mean_y']), 1) if 'mean_y' in row else None

            result[key] = model_info
        return result

    @staticmethod
    def get_actuals(webcam, date_from, date_to):
        from apps.prediction.models import Snapshot

        snapshots = (
            Snapshot.objects
            .filter(
                webcam=webcam,
                predicted_crowd_count__isnull=False,
                ts__gte=date_from,
                ts__lt=date_to,
            )
            .order_by('ts')
        )

        actuals = []
        for s in snapshots:
            ts = timezone.localtime(s.ts).replace(tzinfo=None)
            ts = ts.replace(minute=0, second=0, microsecond=0)
            if ts.hour < HOUR_MIN or ts.hour > HOUR_MAX:
                continue
            actuals.append({
                'timestamp': ts.isoformat(),
                'hour': ts.hour,
                'crowd_count': round(float(s.predicted_crowd_count), 1),
            })
        return actuals


tft_service = TFTService()