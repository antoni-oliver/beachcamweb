"""
TFT Prediction Service — loads 3 pre-trained models and serves predictions.

Models: 3-day (H=36), 14-day (H=168), 30-day (H=360)
Auto-selects model based on requested horizon.
"""

import hashlib
import json
import logging
import threading
from datetime import timedelta, date
from pathlib import Path

import holidays as _holidays
import numpy as np
import pandas as pd
from django.core.cache import cache
from django.utils.timezone import make_aware

from apps.prediction.models import Snapshot
from apps.prediction.scripts import weather_module

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

logger = logging.getLogger(__name__)

HOURS_PER_DAY = 12
HOUR_MIN, HOUR_MAX = 8, 20   # daytime window standardised to 8-20 (incl. the 20:00 bin), matching the dataset + research pipeline

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
_es_holidays_cache: dict[int, set] = {}

def _es_holidays(year: int) -> set:
    if year not in _es_holidays_cache:
        _es_holidays_cache[year] = set(_holidays.country_holidays('ES', subdiv='IB', years=year).keys())
    return _es_holidays_cache[year]

TEMPORAL_FEATURE_BUILDERS = {
    'hour':         lambda ts: ts.hour,
    'day_of_week':  lambda ts: ts.weekday(),
    'month':        lambda ts: ts.month,
    'day_of_year':  lambda ts: ts.timetuple().tm_yday,
    'week_of_year': lambda ts: ts.isocalendar()[1],
    'is_weekend':   lambda ts: int(ts.weekday() >= 5),
    'is_summer':    lambda ts: int(ts.month in (6, 7, 8)),
    'is_holiday':   lambda ts: int(ts.date() in _es_holidays(ts.year)),
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
    _prediction_cache_lock = threading.Lock()
    PREDICTION_TTL = 3600  # seconds — used for Django cache key TTL

    def _forecast_cache_key(self, slug, days, model_set, model_key):
        return f'tft_forecast:{slug}:{days}:{model_set}:{model_key}'

    def _hindcast_cache_path(self, slug, since, days, model_set, model_key):
        raw = f'{slug}:{since}:{days}:{model_set}:{model_key}'
        h = hashlib.md5(raw.encode()).hexdigest()
        base = Path(getattr(settings, 'TFT_HINDCAST_CACHE_DIR', settings.TFT_MODELS_DIR)) / 'hindcast_cache'
        base.mkdir(parents=True, exist_ok=True)
        return base / f'{h}.json'

    def _hindcast_read(self, path):
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def _hindcast_write(self, path, result):
        try:
            path.write_text(json.dumps(result))
        except Exception:
            pass

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
        self._set_kind = {}       # name -> 'tft' or 'xgb'
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
        base_dir = Path(base_dir or settings.TFT_MODELS_DIR)

        # Auto-discover horizon dirs — prefer dirs ending with _{tag}, then any containing {tag}
        def find_dir(horizon_tag):
            suffix = f'_{horizon_tag}'
            candidates = [
                p for p in base_dir.iterdir()
                if p.is_dir() and (p / 'config.json').exists()
            ]
            # exact suffix first (e.g. lstm_grid_model_3d, tft_model_3d)
            for p in sorted(candidates):
                if p.name.endswith(suffix):
                    return p
            # fallback: any dir containing the tag
            for p in sorted(candidates):
                if horizon_tag in p.name:
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

    def load_xgb_models(self, base_dir=None, set_name='xgb'):
        """Load <set>/xgb_model_<H>/{model.joblib, config.json} for each horizon."""
        import joblib
        base_dir = Path(base_dir)
        max_days_map = {'3d': 3, '10d': 10, '15d': 15}
        loaded = {}
        for key in ['3d', '10d', '15d']:
            candidates = [p for p in base_dir.iterdir()
                          if p.is_dir() and p.name.endswith(f'_{key}')
                          and (p / 'model.joblib').exists()
                          and (p / 'config.json').exists()]
            if not candidates:
                logger.warning(f"[{set_name}] xgb {key} not found in {base_dir}, skipping")
                continue
            model_dir = sorted(candidates)[0]
            with open(model_dir / 'config.json') as f:
                config = json.load(f)
            model = joblib.load(model_dir / 'model.joblib')
            per_beach_path = model_dir / 'per_beach.csv'
            per_beach = pd.read_csv(per_beach_path) if per_beach_path.exists() else pd.DataFrame()
            loaded[key] = {
                'model':       model,
                'config':      config,
                'features':    config.get('features', []),
                'hist_pool':   config.get('hist_pool', []),
                'horizon':     int(config.get('horizon', max_days_map[key] * HOURS_PER_DAY)),
                'max_days':    max_days_map[key],
                'per_beach':   per_beach,
                'model_dir':   str(model_dir),
                'model_type':  'XGBRegressor',
                'feature_importance': self._xgb_feature_importance(model, config.get('features', [])),
            }
            logger.info(f"[{set_name}] Loaded XGB {key} (H={loaded[key]['horizon']}) from {model_dir.name}")

        self.model_sets[set_name] = loaded
        self._set_kind[set_name] = 'xgb'
        logger.info(f"XGB set '{set_name}': {len(loaded)} models loaded")
        return loaded

    @staticmethod
    def _xgb_feature_importance(model, features):
        try:
            imp = model.feature_importances_
            return {features[i]: float(imp[i]) for i in range(min(len(features), len(imp)))}
        except Exception:
            return {f: round(1.0 / max(len(features), 1), 4) for f in features}

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
        path = Path(settings.TFT_EVAL_JSON)
        if not path.exists():
            return
        try:
            data = __import__('json').loads(path.read_text())
            self._best_for_horizon = data.get('best_by_horizon', {})
            logger.info(f"Best models: { {k: v['model_set'] for k, v in self._best_for_horizon.items()} }")
        except Exception as e:
            logger.warning(f"Could not load eval JSON: {e}")

    def _auto_discover(self):
        """Scan TFT_MODELS_DIR + XGB_MODELS_DIR for run subfolders.

        TFT/LSTM sets must have <model>_<H>/config.json + nf_model/ subdirs.
        XGB sets must have <model>_<H>/config.json + model.joblib.
        """
        configured = getattr(settings, 'TFT_MODEL_SETS', {})

        def _classify(candidate: Path) -> str | None:
            horizons = [s for s in candidate.iterdir()
                        if s.is_dir() and s.name.endswith(('_3d', '_10d', '_15d'))]
            if not horizons:
                return None
            if any((s / 'config.json').exists() and (s / 'nf_model').is_dir() for s in horizons):
                return 'tft'
            if any((s / 'config.json').exists() and (s / 'model.joblib').exists() for s in horizons):
                return 'xgb'
            return None

        scan_dirs = [Path(settings.TFT_MODELS_DIR)]
        xgb_dir = getattr(settings, 'XGB_MODELS_DIR', None)
        if xgb_dir:
            scan_dirs.append(Path(xgb_dir))

        for models_dir in scan_dirs:
            if not models_dir.exists():
                logger.warning(f"Models dir not found: {models_dir}")
                continue
            for candidate in sorted(models_dir.iterdir()):
                if not candidate.is_dir():
                    continue
                kind = _classify(candidate)
                if kind is None:
                    continue
                name = candidate.name
                if name in configured or name in self._discovered:
                    # An XGB set sharing a TFT set's name (same run, two model
                    # families) is kept under an `xgb_` prefix so both are servable.
                    if kind == 'xgb' and f"xgb_{name}" not in self._discovered \
                            and f"xgb_{name}" not in configured:
                        name = f"xgb_{name}"
                    else:
                        continue
                self._discovered[name] = candidate
                self._set_kind[name] = kind
                logger.info(f"Auto-discovered {kind} set: {name}")

        logger.info(f"Auto-discovered {len(self._discovered)} set(s): {list(self._discovered)}")

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

    def warm_forecast_cache(self, days=15):
        from apps.webcam.models import WebCam
        cams = WebCam.objects.select_related('beach').filter(max_crowd_count__gt=0)
        total = cams.count()
        ok = 0
        for cam in cams:
            try:
                self.predict_mixed(cam, days=days)
                ok += 1
            except Exception:
                logger.exception(f"Cache warm failed for {cam.camera_slug}")
        logger.info(f"Forecast cache warmed: {ok}/{total} cameras.")

    def predict_mixed(self, webcam, days=15, since=None, model_set=None):
        SEGMENTS = [('3d', 1, 3), ('10d', 4, 10), ('15d', 11, 15)]
        combined = []
        seen_ts = set()

        for horizon_key, day_from, day_to in SEGMENTS:
            if day_from > days:
                break
            day_to_actual = min(day_to, days)
            if model_set:
                set_name = model_set
            else:
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
        kind = self._set_kind.get(model_set, 'tft')
        if model_set in configured:
            logger.info(f"TFT: lazy loading configured set '{model_set}'...")
            self.load_models(base_dir=configured[model_set], set_name=model_set)
        elif model_set in self._discovered:
            base = self._discovered[model_set]
            logger.info(f"{kind.upper()}: lazy loading discovered set '{model_set}'...")
            if kind == 'xgb':
                self.load_xgb_models(base_dir=base, set_name=model_set)
            else:
                self.load_models(base_dir=base, set_name=model_set)
        else:
            available = list(self.model_sets) + list(configured) + list(self._discovered)
            raise RuntimeError(f"Model set '{model_set}' not found. Available: {available}")

    def predict(self, webcam, days=3, since=None, model_set='default', model_key=None):
        # Resolve 'default' to the best real set for the requested horizon
        if model_set == 'default':
            model_key_hint = self._horizon_key_for_days(days)
            model_set = self._best_set_for_horizon(model_key_hint)
            if not model_set:
                model_set = self._any_loaded_set()
            if not model_set:
                raise RuntimeError("No model sets available. Check TFT_MODELS_DIR or TFT_MODEL_SETS.")

        self._ensure_loaded(model_set)

        if self._set_kind.get(model_set) == 'xgb':
            return self._predict_xgb(webcam, days, since=since, model_set=model_set, model_key=model_key)

        # If the resolved set loaded nothing (e.g. stale eval JSON pointing to a legacy path),
        # fall back to any set that actually has models
        if model_set not in self.model_sets or not self.model_sets[model_set]:
            logger.warning(f"Set '{model_set}' has no models, falling back to any available set")
            model_set = self._any_loaded_set()
            if not model_set:
                raise RuntimeError("No model sets with loaded models available.")

        models = self.model_sets[model_set]
        model_key = model_key if model_key and model_key in models else self.select_model(days, model_set=model_set)
        if model_key is None:
            raise RuntimeError(f"No TFT models loaded for set '{model_set}'")

        if not since:
            cache_key = self._forecast_cache_key(webcam.camera_slug, days, model_set, model_key)
            cached = cache.get(cache_key)
            if cached is not None:
                logger.debug(f'Forecast cache hit: {cache_key}')
                return cached
        else:
            hc_path = self._hindcast_cache_path(webcam.camera_slug, since, days, model_set, model_key)
            cached = self._hindcast_read(hc_path)
            if cached is not None:
                logger.debug(f'Hindcast cache hit: {hc_path.name}')
                return cached

        m = models[model_key]
        H_model = m['horizon']                          # full model horizon for futr_df
        H_out = min(days * HOURS_PER_DAY, H_model)     # steps to include in output

        lat = float(webcam.camera_latitude or webcam.beach.coordenadas_geograficas.split(',')[0] if webcam.beach.coordenadas_geograficas else 39.5)
        lon = float(webcam.camera_longitude or webcam.beach.coordenadas_geograficas.split(',')[1] if webcam.beach.coordenadas_geograficas else 2.6)

        since_naive = None
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
        _since_ts = pd.Timestamp(since_naive) if since_naive is not None else pd.Timestamp.now()
        since_date = _since_ts.date() if not pd.isnull(_since_ts) else date.today()
        end_date = since_date + timedelta(days=days + 1)
        if missing:
            weather_module.clear_archive()
            weather_all = weather_module.get_for_period(
                lat, lon,
                since_date - timedelta(days=10),
                end_date,
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
            cache_key = self._forecast_cache_key(webcam.camera_slug, days, model_set, model_key)
            cache.set(cache_key, result, self.PREDICTION_TTL)
        else:
            self._hindcast_write(hc_path, result)

        return result

    def _predict_xgb(self, webcam, days, since=None, model_set=None, model_key=None):
        """Castelle-style XGB prediction: each of the H rows of feature history
        produces one prediction at offset H ahead, filling the daytime window
        last_real+1 .. last_real+H (daytime hours 8-20 only)."""
        models = self.model_sets[model_set]
        model_key = model_key if model_key and model_key in models else self.select_model(days, model_set=model_set)
        if not model_key or model_key not in models:
            raise RuntimeError(f"No XGB model for set '{model_set}'")
        m = models[model_key]

        if not since:
            cache_key = self._forecast_cache_key(webcam.camera_slug, days, model_set, model_key)
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        else:
            hc_path = self._hindcast_cache_path(webcam.camera_slug, since, days, model_set, model_key)
            cached = self._hindcast_read(hc_path)
            if cached is not None:
                return cached

        H = m['horizon']
        H_out = min(days * HOURS_PER_DAY, H)
        feat_cols = m['features']
        uid = webcam.camera_slug

        lat = float(webcam.camera_latitude or (webcam.beach.coordenadas_geograficas.split(',')[0]
                    if webcam.beach.coordenadas_geograficas else 39.5))
        lon = float(webcam.camera_longitude or (webcam.beach.coordenadas_geograficas.split(',')[1]
                    if webcam.beach.coordenadas_geograficas else 2.6))

        since_naive = None
        if since:
            since_naive = since.replace(tzinfo=None) if hasattr(since, 'tzinfo') and since.tzinfo else since
            weather_all = self._fetch_weather_for_period(lat, lon, since_naive, days, context_days=10)
        else:
            weather_all = self._fetch_weather_forecast(lat, lon, past_days=10, forecast_days=16)
        weather_dt = self._filter_daytime(weather_all).sort_values('ds_real').reset_index(drop=True)

        # ── Snapshot history (daytime only)
        qs = Snapshot.objects.filter(webcam=webcam, predicted_crowd_count__isnull=False)
        if since:
            since_aware = since if (hasattr(since, "tzinfo") and since.tzinfo) else make_aware(since_naive)
            qs = qs.filter(ts__lt=since_aware)
        snapshots = list(qs.order_by('-ts')[:(H + 60)])[::-1]
        rows = []
        for s in snapshots:
            ts = timezone.localtime(s.ts).replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
            if HOUR_MIN <= ts.hour <= HOUR_MAX:
                rows.append({'ds': ts, 'y': float(s.predicted_crowd_count)})
        if len(rows) < 5:
            raise ValueError(f"Not enough daytime history for {uid}: {len(rows)}")
        hist = pd.DataFrame(rows).drop_duplicates(subset='ds').sort_values('ds').reset_index(drop=True)

        last_real = hist['ds'].iloc[-1]

        # Build target daytime timestamps after last_real
        target_ts = []
        cur = last_real
        while len(target_ts) < H:
            cur = cur + pd.Timedelta(hours=1)
            if cur.hour > HOUR_MAX:
                cur = (cur + pd.Timedelta(days=1)).normalize().replace(hour=HOUR_MIN)
            if cur.hour < HOUR_MIN:
                cur = cur.replace(hour=HOUR_MIN)
            target_ts.append(cur)

        # Compute lag/roll features on the full daytime history, then take last H rows
        hist['y_lag1'] = hist['y'].shift(1)
        hist['y_roll12'] = hist['y'].shift(1).rolling(12, min_periods=1).mean()
        hist['y_roll36'] = hist['y'].shift(1).rolling(36, min_periods=1).mean()

        # Pad history if we don't have enough rows for H predictions
        if len(hist) < H:
            need = H - len(hist)
            pad = pd.concat([hist.iloc[[0]]] * need, ignore_index=True)
            hist = pd.concat([pad, hist], ignore_index=True).reset_index(drop=True)
        feat_rows = hist.tail(H).reset_index(drop=True)

        # Build feature matrix: feat_rows[i] predicts target_ts[i]
        X_rows = []
        for i in range(H):
            target = target_ts[i]
            # nearest weather row to target
            diffs = (weather_dt['ds_real'] - target).abs()
            w = weather_dt.loc[diffs.idxmin()] if len(weather_dt) else None

            row = {
                'y_lag1':   float(feat_rows['y_lag1'].iloc[i]) if pd.notna(feat_rows['y_lag1'].iloc[i]) else 0.0,
                'y_roll12': float(feat_rows['y_roll12'].iloc[i]) if pd.notna(feat_rows['y_roll12'].iloc[i]) else 0.0,
                'y_roll36': float(feat_rows['y_roll36'].iloc[i]) if pd.notna(feat_rows['y_roll36'].iloc[i]) else 0.0,
            }
            for col in feat_cols:
                if col in row:
                    continue
                if not col.endswith('_t_plus_h'):
                    row[col] = 0.0
                    continue
                raw = col[: -len('_t_plus_h')]
                if raw == 'hour':
                    row[col] = target.hour
                elif raw == 'day_of_week':
                    row[col] = target.weekday()
                elif raw == 'month':
                    row[col] = target.month
                elif raw == 'is_weekend':
                    row[col] = int(target.weekday() >= 5)
                elif w is not None and raw in w.index and pd.notna(w[raw]):
                    row[col] = float(w[raw])
                else:
                    row[col] = 0.0
            X_rows.append([row.get(c, 0.0) for c in feat_cols])

        X = np.asarray(X_rows, dtype=float)
        preds = m['model'].predict(X).clip(0, None)
        pred_map = {target_ts[i].strftime('%Y-%m-%dT%H:00'): float(preds[i]) for i in range(H_out)}

        max_cc = webcam.max_crowd_count or 0
        start_day = target_ts[0].normalize()
        om_cols = [c for c in feat_cols if c.endswith('_t_plus_h') and c[:-len('_t_plus_h')].startswith('om_')]
        weather_map = {}
        for _, wrow in weather_dt.iterrows():
            entry = {}
            for col in om_cols:
                raw = col[: -len('_t_plus_h')]
                if raw in wrow.index and pd.notna(wrow[raw]):
                    entry[col[: -len('_t_plus_h')]] = round(float(wrow[raw]), 2)
            if entry:
                weather_map[wrow['ds_real'].strftime('%Y-%m-%dT%H:00')] = entry

        temp_col = next((c for c in feat_cols if 'temperature' in c), None)
        predictions = []
        for day_offset in range(days):
            current_day = start_day + pd.Timedelta(days=day_offset)
            for hour in range(24):
                ts = current_day.replace(hour=hour)
                key = ts.strftime('%Y-%m-%dT%H:00')
                w = weather_map.get(key, {})
                if key in pred_map:
                    cc = round(pred_map[key], 1)
                    level = classify_occupancy(cc, max_cc)
                    predictions.append({
                        'timestamp': ts.isoformat(),
                        'hour': hour,
                        'crowd_count': cc,
                        'available': True,
                        'occupancy_level': level,
                        'occupancy_ratio': round(cc / max_cc, 3) if max_cc > 0 else None,
                        'temperature': w.get(temp_col[:-len('_t_plus_h')]) if temp_col else None,
                        'features': w,
                    })
                else:
                    predictions.append({
                        'timestamp': ts.isoformat(),
                        'hour': hour,
                        'crowd_count': 0,
                        'available': False,
                        'occupancy_level': None,
                        'occupancy_ratio': None,
                        'temperature': None,
                        'features': {},
                    })

        result = {
            'model': model_key,
            'model_set': model_set,
            'beach': webcam.beach.beach_name,
            'webcam': uid,
            'horizon_days': days,
            'horizon_hours': H_out,
            'max_crowd_count': max_cc,
            'last_data': last_real.strftime('%Y-%m-%dT%H:%M:%S'),
            'feature_importance': m.get('feature_importance', {}),
            'futr_features': feat_cols,
            'hist_features': [],
            'predictions': predictions,
        }
        if not since:
            cache_key = self._forecast_cache_key(webcam.camera_slug, days, model_set, model_key)
            cache.set(cache_key, result, self.PREDICTION_TTL)
        else:
            self._hindcast_write(hc_path, result)
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
        return weather_module._get_forecast(lat, lon, past_days=past_days, forecast_days=forecast_days)

    def _fetch_weather_archive(self, lat, lon, start_date, end_date):
        return weather_module._get_archive(lat, lon, start_date, end_date)

    def _fetch_weather_for_period(self, lat, lon, since_naive, forecast_days, context_days=10):
        ts = pd.Timestamp(since_naive) if since_naive is not None else pd.Timestamp.now()
        since_date = ts.date() if not pd.isnull(ts) else date.today()
        end_date = since_date + timedelta(days=forecast_days)
        return weather_module.get_for_period(lat, lon, since_date, end_date, context_days=context_days)

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