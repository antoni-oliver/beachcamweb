"""
TFT Prediction Service — loads 3 pre-trained models and serves predictions.

Models: 3-day (H=36), 14-day (H=168), 30-day (H=360)
Auto-selects model based on requested horizon.
"""

import json
import logging
import threading
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
from django.utils.timezone import make_aware

from apps.prediction.models import Snapshot

if not hasattr(pd.Series, 'is_nan'):
    pd.Series.is_nan = pd.Series.isna
if not hasattr(pd.Series, 'is_null'):
    pd.Series.is_null = pd.Series.isnull
if not hasattr(pd.DataFrame, 'is_nan'):
    pd.DataFrame.is_nan = lambda self: self.isna().squeeze()
if not hasattr(pd.DataFrame, 'is_null'):
    pd.DataFrame.is_null = lambda self: self.isnull().squeeze()
from django.conf import settings
from django.utils import timezone

from apps.prediction.weather_cache import weather_cache

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

# Temporal features computable from a timestamp — extend as needed
TEMPORAL_FEATURE_BUILDERS = {
    'hour':         lambda ts: ts.hour,
    'day_of_week':  lambda ts: ts.weekday(),
    'month':        lambda ts: ts.month,
    'day_of_year':  lambda ts: ts.timetuple().tm_yday,
    'week_of_year': lambda ts: ts.isocalendar()[1],
    'is_weekend':   lambda ts: int(ts.weekday() >= 5),
    'is_summer':    lambda ts: int(ts.month in (6, 7, 8)),
    'quarter':      lambda ts: (ts.month - 1) // 3 + 1,
}

# Static features computable from the y series — extend as needed
STATIC_FEATURE_BUILDERS = {
    'stat_mean_y': lambda y: float(y.mean()),
    'stat_cv':     lambda y: float(y.std() / max(float(y.mean()), 1)),
    'stat_max_y':  lambda y: float(y.max()),
    'stat_min_y':  lambda y: float(y.min()),
    'stat_median_y': lambda y: float(y.median()),
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
    _prediction_cache = {}
    _prediction_cache_lock = threading.Lock()
    PREDICTION_TTL = 3600

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.models = {}
        self.model_sets = {}
        self._best_for_horizon = {}
        self._discovered = {}     # name -> base_dir path, from auto-discovery
        self._load_best_models()
        self._auto_discover()
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

        model_type = None
        has_static = False
        for model_obj in nf.models:
            model_type = model_obj.__class__.__name__
            has_static = hasattr(model_obj, 'stat_exog_list') and bool(getattr(model_obj, 'stat_exog_list', None))
            break

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
            'model_type': model_type,
            'has_static': has_static,
        }

    def load_models(self, base_dir=None, set_name='default'):
        base_dir = Path(base_dir or getattr(settings, 'TFT_MODELS_DIR', 'tft_models'))

        # Auto-discover horizon dirs — tries known patterns then glob
        def find_dir(horizon_tag):
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
        # Add configured but not yet loaded sets
        for name in getattr(settings, 'TFT_MODEL_SETS', {}):
            if name not in loaded:
                loaded[name] = {}
        # Add auto-discovered but not yet loaded sets
        for name in self._discovered:
            if name not in loaded:
                loaded[name] = {}
        return loaded

    def select_model(self, days, model_set='default'):
        if model_set != 'default':
            models = self.model_sets.get(model_set, {})
            for key in ['3d', '10d', '15d']:
                if key in models and days <= models[key]['max_days']:
                    return key
            available = [k for k in ['15d', '10d', '3d'] if k in models]
            return available[0] if available else None

        for key in ['3d', '10d', '15d']:
            best = self._best_for_horizon.get(key)
            if best:
                set_name = best['model_set']
                models = self.model_sets.get(set_name, {})
                if key in models and days <= models[key]['max_days']:
                    return key

        for set_name, models in self.model_sets.items():
            for key in ['3d', '10d', '15d']:
                if key in models and days <= models[key]['max_days']:
                    return key
        return None


    def _load_best_models(self):
        raw = getattr(settings, 'TFT_EVAL_JSON', 'apps/prediction/model_evaluation.json')
        path = Path(raw)
        if not path.is_absolute():
            path = Path(settings.BASE_DIR) / path
        if not path.exists():
            return
        try:
            data = __import__('json').loads(path.read_text())
            self._best_for_horizon = data.get('best_by_horizon', {})
            logger.info(f"Best models: { {k: v['model_set'] for k, v in self._best_for_horizon.items()} }")
        except Exception as e:
            logger.warning(f"Could not load eval JSON: {e}")

    def _auto_discover(self):
        """Scan TFT_MODELS_DIR for run subfolders containing tft_model_Xd dirs."""
        models_dir = getattr(settings, 'TFT_MODELS_DIR', None)
        if not models_dir:
            return
        models_dir = Path(models_dir)
        if not models_dir.is_absolute():
            models_dir = Path(settings.BASE_DIR) / models_dir
        if not models_dir.exists():
            logger.warning(f"TFT_MODELS_DIR not found: {models_dir}")
            return

        configured = getattr(settings, 'TFT_MODEL_SETS', {})

        for candidate in sorted(models_dir.iterdir()):
            if not candidate.is_dir():
                continue
            # Valid run folder must contain at least one tft_model_Xd subdir with config.json
            has_model = any(
                (candidate / subdir / 'config.json').exists()
                for subdir in ['tft_model_3d', 'tft_model_10d', 'tft_model_15d']
            )
            if not has_model:
                continue
            name = candidate.name
            # TFT_MODEL_SETS takes priority — skip if already explicitly configured
            if name in configured:
                continue
            self._discovered[name] = candidate
            logger.info(f"Auto-discovered model set: {name}")

        logger.info(f"Auto-discovered {len(self._discovered)} model set(s): {list(self._discovered)}")

    def _horizon_key_for_days(self, days):
        if days <= 3:
            return '3d'
        if days <= 10:
            return '10d'
        return '15d'

    def _any_available_set(self):
        """Return any loaded set, else any discovered/configured set name."""
        if self.model_sets:
            return next(iter(self.model_sets))
        for name in self._discovered:
            return name
        for name in getattr(settings, 'TFT_MODEL_SETS', {}):
            return name
        return None

    def _any_loaded_set(self):
        """Return any already-loaded non-empty set, or load from discovered/configured sets until one works."""
        for name, models in self.model_sets.items():
            if models:
                return name
        candidates = list(self._discovered) + list(getattr(settings, 'TFT_MODEL_SETS', {}))
        for name in candidates:
            try:
                self._ensure_loaded(name)
                if self.model_sets.get(name):
                    return name
            except Exception:
                continue
        return None

    def _best_set_for_horizon(self, horizon_key) -> str:
        best = self._best_for_horizon.get(horizon_key)
        if best:
            name = best['model_set']
            try:
                self._ensure_loaded(name)
                if self.model_sets.get(name):
                    return name
            except Exception:
                pass
        # Fall back to any loaded set that has this horizon key
        for set_name, models in self.model_sets.items():
            if horizon_key in models:
                return set_name
        return None

    def _accuracy_for_horizon(self, horizon_key):
        best = self._best_for_horizon.get(horizon_key)
        if not best:
            return None
        return {
            'relMAE_all':    best.get('relMAE_all'),
            'relMAE_summer': best.get('relMAE_summer'),
            'relMAE_season': best.get('relMAE_season'),
        }

    def predict_mixed(self, webcam, days=15, since=None):
        SEGMENTS = [('3d', 1, 3), ('10d', 4, 10), ('15d', 11, 15)]
        combined = []
        seen_ts = set()

        for horizon_key, day_from, day_to in SEGMENTS:
            if day_from > days:
                break
            day_to_actual = min(day_to, days)
            set_name = self._best_set_for_horizon(horizon_key)
            if not set_name:
                set_name = self._any_loaded_set()
            if not set_name:
                logger.warning(f"predict_mixed: no set available for {horizon_key}, skipping")
                continue
            try:
                self._ensure_loaded(set_name)
            except Exception as e:
                logger.warning(f"predict_mixed: could not load {set_name}: {e}")
                continue
            if set_name not in self.model_sets or horizon_key not in self.model_sets.get(set_name, {}):
                logger.warning(f"predict_mixed: no model for {horizon_key} in {set_name}, skipping")
                continue
            try:
                result = self.predict(webcam, day_to_actual, since=since, model_set=set_name)
            except Exception as e:
                logger.warning(f"predict_mixed: {horizon_key} failed: {e}")
                continue

            accuracy = self._accuracy_for_horizon(horizon_key)
            day_preds = {}
            days_seen = []
            for p in result.get('predictions', []):
                if not p.get('available'):
                    continue
                d = p['timestamp'][:10]
                if d not in day_preds:
                    days_seen.append(d)
                    day_preds[d] = []
                day_preds[d].append(p)

            for d in days_seen[day_from - 1: day_to_actual]:
                for p in day_preds.get(d, []):
                    ts = p['timestamp'][:16]
                    if ts not in seen_ts:
                        seen_ts.add(ts)
                        combined.append({**p, 'model_key': horizon_key,
                                         'model_set': set_name, 'accuracy': accuracy})

        # Fill unavailable (night) slots from 3d model output
        first_set = self._best_set_for_horizon('3d') or self._any_loaded_set()
        try:
            full = self.predict(webcam, days, since=since, model_set=first_set)
            for p in full.get('predictions', []):
                if not p.get('available'):
                    ts = p['timestamp'][:16]
                    if ts not in seen_ts:
                        seen_ts.add(ts)
                        combined.append({**p, 'model_key': None, 'model_set': None, 'accuracy': None})
        except Exception:
            pass

        combined.sort(key=lambda p: p['timestamp'])
        return {
            'beach_id':        webcam.beach.id,
            'beach':           webcam.beach.beach_name,
            'webcam':          webcam.camera_slug,
            'horizon_days':    days,
            'max_crowd_count': webcam.max_crowd_count or 0,
            'segments':        {k: self._best_for_horizon.get(k) for k, _, _ in SEGMENTS},
            'predictions':     combined,
        }

    def _ensure_loaded(self, model_set='default'):
        if model_set == 'default' or model_set in self.model_sets:
            return  # default is virtual — resolved at predict time per horizon
        configured = getattr(settings, 'TFT_MODEL_SETS', {})
        if model_set in configured:
            logger.info(f"TFT: lazy loading configured set '{model_set}'...")
            self.load_models(base_dir=configured[model_set], set_name=model_set)
        elif model_set in self._discovered:
            logger.info(f"TFT: lazy loading discovered set '{model_set}'...")
            self.load_models(base_dir=self._discovered[model_set], set_name=model_set)
        else:
            available = list(self.model_sets) + list(configured) + list(self._discovered)
            raise RuntimeError(f"Model set '{model_set}' not found. Available: {available}")

    def predict(self, webcam, days=3, since=None, model_set='default'):
        # Resolve 'default' to the best real set for the requested horizon
        if model_set == 'default':
            model_key_hint = self._horizon_key_for_days(days)
            model_set = self._best_set_for_horizon(model_key_hint)
            if not model_set:
                model_set = self._any_loaded_set()
            if not model_set:
                raise RuntimeError("No model sets available. Check TFT_MODELS_DIR or TFT_MODEL_SETS.")

        self._ensure_loaded(model_set)

        # If the resolved set loaded nothing (e.g. stale eval JSON pointing to a legacy path),
        # fall back to any set that actually has models
        if model_set not in self.model_sets or not self.model_sets[model_set]:
            logger.warning(f"Set '{model_set}' has no models, falling back to any available set")
            model_set = self._any_loaded_set()
            if not model_set:
                raise RuntimeError("No model sets with loaded models available.")

        models = self.model_sets[model_set]
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

        # Re-fetch if model needs columns not present in cached weather
        required_om = [c for c in m['futr'] + m['hist'] if c.startswith('om_')]
        #print(f"[DEBUG predict] required_om={required_om}")
        #print(f"[DEBUG predict] weather_all cols={list(weather_all.columns)}")
        missing = [c for c in required_om if c not in weather_all.columns]
        #print(f"[DEBUG predict] missing weather cols={missing}")
        since_date = pd.Timestamp(since_naive if since else pd.Timestamp.now()).date()
        end_date = since_date + pd.Timedelta(days=days + 1)
        weather_all = weather_cache.ensure_columns(
            weather_all, lat, lon,
            since_date - pd.Timedelta(days=10),
            end_date.to_pydatetime().date() if hasattr(end_date, 'to_pydatetime') else end_date,
            required_om,
        )

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
        all_feature_cols = list(dict.fromkeys(m['futr'] + m['hist']))  # deduplicate, preserve order
        train_cols = ['unique_id', 'ds', 'y'] + all_feature_cols
        context_df = context[train_cols].copy()

        static_df = self._make_static(uid, context, m['stat_cols'])
        use_static = m.get('has_static', True)

        # Deduplicate columns — om_temperature_2m can appear in both futr and hist
        context_df = context_df.loc[:, ~context_df.columns.duplicated()]

        # Fill NaNs before predict — some model versions trigger is_nan() check
        # which fails on newer pandas; filling avoids the check path entirely
        context_df = context_df.fillna(0)
        futr_input = futr_df[['unique_id', 'ds'] + m['futr']].fillna(0)
        if use_static and static_df is not None:
            static_df = static_df.fillna(0)

        # Monkey-patch utilsforecast.processing.is_nan — newer pandas dropped
        # Series.is_nan() but utilsforecast calls it unconditionally (polars API)
        try:
            import utilsforecast.processing as _ufp
            import pandas as _pd
            def _is_nan_pandas(s):
                if isinstance(s, _pd.Series):
                    return s.isna()
                return s.is_nan()
            _ufp.is_nan = _is_nan_pandas
        except Exception:
            pass

        with self._predict_lock:
            forecast = m['nf'].predict(
                df=context_df,
                static_df=static_df if use_static else None,
                futr_df=futr_input,
            )
        forecast = forecast.reset_index()
        # detect column: TFT, LSTM, etc.
        model_col = m.get('model_type', 'TFT')
        if model_col not in forecast.columns:
            model_col = next((c for c in forecast.columns if c not in ('unique_id', 'ds')), 'TFT')
        forecast[model_col] = forecast[model_col].clip(lower=0)
        forecast['TFT'] = forecast[model_col]  # normalize to TFT key for _format_output

        max_cc = webcam.max_crowd_count or 0
        predictions = self._format_output(forecast, weather_all, last_real, H_out, days, max_cc, m['futr'], m['hist'])

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
            'futr_features': m['futr'],
            'hist_features': m['hist'],
            'predictions': predictions,
        }

        if not since:
            with self._prediction_cache_lock:
                self._prediction_cache[(webcam.camera_slug, days, model_set)] = (time.time(), result)

        return result

    def _build_context(self, webcam, m, weather_all, since=None):

        input_size = m['input_size']
        uid = webcam.camera_slug

        qs = Snapshot.objects.filter(webcam=webcam, predicted_crowd_count__isnull=False)
        if since:
            since_aware = since if (hasattr(since, "tzinfo") and since.tzinfo) else make_aware(since)
            qs = qs.filter(ts__lt=since_aware)
        snapshots = list(qs.order_by('-ts')[:input_size * 3])[::-1]

        #print(f"[DEBUG _build_context] uid={uid} snapshots={len(snapshots)} input_size={input_size}")

        if len(snapshots) < 1:
            raise ValueError(f"Not enough history for {uid}: {len(snapshots)} snapshots")

        rows = []
        for s in snapshots:
            ts = timezone.localtime(s.ts).replace(tzinfo=None)
            if ts.hour < HOUR_MIN or ts.hour > HOUR_MAX:
                continue
            row = {'ds_real': ts, 'y': float(s.predicted_crowd_count)}
            for col in m['futr'] + m['hist']:
                if col in TEMPORAL_FEATURE_BUILDERS:
                    row[col] = TEMPORAL_FEATURE_BUILDERS[col](ts)
            rows.append(row)

        if not rows:
            raise ValueError(f"Not enough daytime history for {uid}")

        df = pd.DataFrame(rows)
        df['unique_id'] = uid
        df = df.sort_values('ds_real').tail(input_size).reset_index(drop=True)

        #print(f"[DEBUG _build_context] rows_daytime={len(rows)} df after tail={len(df)}")

        # Pad to input_size by repeating the first row if needed
        if len(df) < input_size:
            pad = pd.concat([df.iloc[[0]]] * (input_size - len(df)), ignore_index=True)
            df = pd.concat([pad, df], ignore_index=True)
        df['ds'] = range(len(df))

        df = self._merge_weather(df, self._filter_daytime(weather_all), m['futr'] + m['hist'])

        #print(f"[DEBUG _build_context] after merge cols={list(df.columns)}")
        #print(f"[DEBUG _build_context] NaN counts:\n{df.isna().sum()}")

        all_features = m['futr'] + m['hist']
        for col in all_features:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].astype(float)
        df['ds'] = df['ds'].astype(int)

        return df

    def _fetch_weather_forecast(self, lat, lon, past_days=10, forecast_days=16):
        return weather_cache._get_forecast(lat, lon, past_days=past_days, forecast_days=forecast_days)

    def _fetch_weather_archive(self, lat, lon, start_date, end_date):
        return weather_cache._get_archive(lat, lon, start_date, end_date)

    def _fetch_weather_for_period(self, lat, lon, since_naive, forecast_days, context_days=10):
        since_date = pd.Timestamp(since_naive).date()
        end_date = since_date + timedelta(days=forecast_days)
        return weather_cache.get_for_period(lat, lon, since_date, end_date, context_days=context_days)

    def _filter_daytime(self, weather_df):
        return weather_df[(weather_df['hour'] >= HOUR_MIN) & (weather_df['hour'] <= HOUR_MAX)]

    def _merge_weather(self, panel_df, weather_df, feature_cols):
        panel_df = panel_df.copy()
        weather_df = weather_df.sort_values('ds_real')
        om_cols = [c for c in feature_cols if c.startswith('om_') and c in weather_df.columns]

        for idx, row in panel_df.iterrows():
            diffs = (weather_df['ds_real'] - row['ds_real']).abs()
            nearest = diffs.idxmin()
            for col in om_cols:
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
        row = {'unique_id': uid}
        for col in stat_cols:
            if col in STATIC_FEATURE_BUILDERS:
                row[col] = STATIC_FEATURE_BUILDERS[col](y)
            else:
                row[col] = 0.0
        return pd.DataFrame([row])

    def _format_output(self, forecast, weather, last_real, H, days, max_crowd_count, futr_cols, hist_cols):
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
        for val in daytime_preds:
            pred_map[ts.strftime('%Y-%m-%dT%H:00')] = val
            ts = ts + pd.Timedelta(hours=1)
            if ts.hour > HOUR_MAX:
                ts = (ts + pd.Timedelta(days=1)).normalize().replace(hour=HOUR_MIN)

        om_cols = [c for c in futr_cols + hist_cols if c.startswith('om_')]
        temporal_cols = [c for c in futr_cols + hist_cols if c in TEMPORAL_FEATURE_BUILDERS]
        temp_col = next((c for c in futr_cols + hist_cols if 'temperature' in c), None)

        weather_map = {}
        last_weather = {}
        for _, row in weather.sort_values('ds_real').iterrows():
            key = row['ds_real'].strftime('%Y-%m-%dT%H:00')
            entry = {}
            for col in om_cols:
                if col in row.index and pd.notna(row[col]):
                    entry[col] = round(float(row[col]), 2)
            if entry:
                last_weather = entry
            weather_map[key] = entry

        # Forward-fill: for any key in pred_map without weather, use last known values
        for key in pred_map:
            if not weather_map.get(key) and last_weather:
                weather_map[key] = last_weather

        start_day = first_forecast.normalize()
        predictions = []

        for day_offset in range(days):
            current_day = start_day + pd.Timedelta(days=day_offset)
            for hour in range(24):
                ts = current_day.replace(hour=hour)
                key = ts.strftime('%Y-%m-%dT%H:00')
                w = weather_map.get(key, {})
                temp = w.get(temp_col) if temp_col else None

                if key in pred_map:
                    cc = round(pred_map[key], 1)
                    level = classify_occupancy(cc, max_crowd_count)
                    features = dict(w)
                    for col in temporal_cols:
                        features[col] = TEMPORAL_FEATURE_BUILDERS[col](ts)
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