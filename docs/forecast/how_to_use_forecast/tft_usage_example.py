"""
TFT Beach Occupancy — Standalone Prediction Script
===================================================
Runs anywhere. No Django. No database. No webcam.

Requirements:
    pip install neuralforecast torch pandas numpy

Usage:
    python tft_usage_example.py

The script runs end-to-end with fake data by default.
To use real data, replace the three blocks marked:
    # <<< REPLACE: <what> >>>  ...  # <<< END REPLACE >>>
"""

import json
import math
try:
    import holidays as _holidays_lib
    _HAS_HOLIDAYS = True
except ImportError:
    _HAS_HOLIDAYS = False
import zoneinfo
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast


# =============================================================================
# CONFIGURATION — set these paths before running
# =============================================================================

# Absolute path to the trained model directory.
# Each set has three horizon subdirs: tft_model_3d / tft_model_10d / tft_model_15d
#
# Example layout on the server:
#   /home/user/beachcamweb/apps/prediction/tft_models/
#   └── tft_20260315_225143/
#       ├── tft_model_3d/
#       │   ├── nf_model/            ← NeuralForecast checkpoint
#       │   ├── config.json          ← hyperparameters + feature lists
#       │   └── static_features.csv  ← per-camera statistics from training
#       ├── tft_model_10d/
#       └── tft_model_15d/

MODEL_DIR = Path('tft_20260315_225143/tft_model_10d')   # ← SET THIS

# The horizon is read automatically from config.json inside MODEL_DIR.


# =============================================================================
# CONSTANTS — do not change (must match training)
# =============================================================================

_SPAIN_TZ     = zoneinfo.ZoneInfo('Europe/Madrid')
HOUR_MIN      = 8    # first daytime hour included in predictions
HOUR_MAX      = 20   # last  daytime hour included in predictions
HOURS_PER_DAY = HOUR_MAX - HOUR_MIN + 1   # 13 hours/day

OCCUPANCY_THRESHOLDS = [
    (0.75, 'HIGH'),
    (0.50, 'MEDIUM'),
    (0.25, 'LOW'),
    (0.00, 'VERY_LOW'),
]

# Temporal features computed from a timestamp — cover all features the models
# were trained with. The model reads which ones it needs from config.json.
TEMPORAL_BUILDERS = {
    'hour':         lambda ts: ts.hour,
    'day_of_week':  lambda ts: ts.weekday(),
    'month':        lambda ts: ts.month,
    'day_of_year':  lambda ts: ts.timetuple().tm_yday,
    'week_of_year': lambda ts: ts.isocalendar()[1],
    'is_weekend':   lambda ts: int(ts.weekday() >= 5),
    'is_summer':    lambda ts: int(ts.month in (6, 7, 8)),
    'quarter':      lambda ts: (ts.month - 1) // 3 + 1,
    # is_holiday: requires 'pip install holidays'
    # Returns 1 on Spanish public holidays, 0 otherwise.
    'is_holiday': lambda ts: int(
        _HAS_HOLIDAYS and
        ts.date() in _holidays_lib.Spain(years=ts.year)
    ),
}

# Static per-camera statistics derived from training data y-series.
STATIC_BUILDERS = {
    'stat_mean_y': lambda y: float(y.mean()),
    'stat_cv':     lambda y: float(y.std() / max(float(y.mean()), 1)),
}


# =============================================================================
# STEP 1 — Load model weights and config
# =============================================================================
# Reads nf_model/ (NeuralForecast checkpoint) and config.json.
# Feature lists (futr_exog, hist_exog) are read automatically from config —
# you never hardcode them.

def load_model(model_dir: Path):
    if not model_dir.exists():
        raise FileNotFoundError(
            f'Model directory not found: {model_dir}\n'
            f'Set MODEL_DIR at the top of this script.'
        )

    with open(model_dir / 'config.json') as f:
        config = json.load(f)

    nf = NeuralForecast.load(str(model_dir / 'nf_model'))

    # Remove keys that newer PyTorch Lightning versions no longer accept
    BAD_KEYS = ['training_data_availability_threshold']
    for m in nf.models:
        for attr in ['trainer_kwargs', 'pred_trainer_kwargs']:
            d = getattr(m, attr, None)
            if isinstance(d, dict):
                for k in BAD_KEYS:
                    d.pop(k, None)
                d['logger'] = False
                d['enable_progress_bar'] = False

    static_df = pd.read_csv(model_dir / 'static_features.csv')

    print('Model loaded')
    print(f'  Horizon      : H={config["horizon"]} steps ({config["horizon"] // HOURS_PER_DAY} days)')
    print(f'  Input size   : {config["input_size"]} steps (context history)')
    print(f'  futr_exog    : {config["futr_exog"]}')
    print(f'  hist_exog    : {config["hist_exog"]}')
    return nf, config, static_df


nf, config, static_df = load_model(MODEL_DIR)

# Derived from config — do not change manually
FUTR_FEATURES = config['futr_exog']          # known at forecast time (calendar + NWP temp)
HIST_FEATURES = config['hist_exog']          # observed historically only (solar, cloud, vpd)
ALL_FEATURES  = FUTR_FEATURES + HIST_FEATURES
INPUT_SIZE    = config['input_size']         # 48 steps = 4 days of daytime history
H             = config['horizon']            # total forecast steps
MAX_DAYS      = math.ceil(H / HOURS_PER_DAY)


# =============================================================================
# STEP 2 — Camera / beach metadata
# =============================================================================
# In BeachCamWeb (Django):
#   webcam    = WebCam.objects.select_related('beach').get(camera_slug='...')
#   uid       = webcam.camera_slug         → unique string per camera
#   max_crowd = webcam.max_crowd_count     → P99-calibrated maximum people
#   lat       = float(webcam.camera_latitude)
#   lon       = float(webcam.camera_longitude)
#
# Expected data types:
#   uid       : str   — must match unique_id values in static_features.csv
#   max_crowd : int   — used only for occupancy_ratio + level classification
#   lat, lon  : float — used only for weather fetching (not needed here)

# <<< REPLACE: camera metadata >>>
uid       = 'platja-de-muro'    # camera slug — must exist in static_features.csv
                                 # or be a new camera (handled in Step 5 below)
max_crowd = 80                   # P99 maximum people count for this camera
lat, lon  = 39.8028, 3.1256     # GPS coordinates (Mallorca north coast)
# <<< END REPLACE >>>

print(f'\nCamera  : {uid}')
print(f'Max cap : {max_crowd} people')


# =============================================================================
# STEP 3 — Historical crowd counts (context)
# =============================================================================
# The model needs INPUT_SIZE (48) daytime hourly crowd count readings as
# historical context. These come from the Bayesian VGG-19 crowd counting model
# running on webcam images, stored as Django Snapshot records.
#
# In BeachCamWeb (Django):
#   from apps.prediction.models import Snapshot
#   qs = (Snapshot.objects
#         .filter(webcam=webcam, predicted_crowd_count__isnull=False)
#         .order_by('-ts')[:INPUT_SIZE * 3])
#   snapshot_rows = [
#       {'ts':    s.ts.astimezone(_SPAIN_TZ).replace(tzinfo=None),
#        'crowd': float(s.predicted_crowd_count)}
#       for s in reversed(list(qs))
#       if HOUR_MIN <= s.ts.astimezone(_SPAIN_TZ).hour <= HOUR_MAX
#   ]
#
# Expected data structure:
#   snapshot_rows = [
#       {'ts': datetime(2025, 7, 14,  8, 0),  'crowd': 5.0},
#       {'ts': datetime(2025, 7, 14,  9, 0),  'crowd': 12.3},
#       {'ts': datetime(2025, 7, 14, 10, 0),  'crowd': 24.1},
#       ...   (one row per daytime hour, Spain local time, no tzinfo)
#       {'ts': datetime(2025, 7, 14, 20, 0),  'crowd': 8.7},
#       {'ts': datetime(2025, 7, 15,  8, 0),  'crowd': 6.2},   ← next day
#       ...
#   ]
#
# Rules:
#   - ts must be Spain local time (Europe/Madrid) with tzinfo stripped
#   - Only hours 8–20 are included (night hours filtered out)
#   - Chronological order (oldest first)
#   - At least INPUT_SIZE (48) rows needed; more is fine (script tails to 48)

# <<< REPLACE: crowd count history >>>
np.random.seed(42)
prediction_start = datetime(2025, 7, 15, 8, 0)   # when the forecast should start

def _fake_crowd(ts, max_count):
    """Simulate realistic occupancy: daily peak at 13:30, summer boost."""
    h_eff = np.exp(-((ts.hour - 13.5) ** 2) / 18)
    s_eff = 0.4 + 0.6 * np.sin(np.pi * (ts.month - 4) / 6)
    return max(0.0, max_count * 0.7 * s_eff * h_eff + np.random.normal(0, max_count * 0.08))

snapshot_rows = []
for i in range(INPUT_SIZE * 2):
    ts = prediction_start - timedelta(hours=INPUT_SIZE * 2 - i)
    if HOUR_MIN <= ts.hour <= HOUR_MAX:
        snapshot_rows.append({'ts': ts, 'crowd': _fake_crowd(ts, max_crowd)})
# <<< END REPLACE >>>

print(f'\n{len(snapshot_rows)} context snapshots')
print(f'  From : {snapshot_rows[0]["ts"]}')
print(f'  To   : {snapshot_rows[-1]["ts"]}')


# =============================================================================
# STEP 4 — Weather data
# =============================================================================
# The model uses two types of weather features:
#
#   futr_exog — known at forecast time (NWP forecast, e.g. om_temperature_2m)
#   hist_exog — observed historically only (e.g. solar radiation, cloud cover)
#
# In BeachCamWeb (Django):
#   import sys
#   sys.path.insert(0, '/path/to/beachcamweb/apps/prediction/scripts')
#   import weather_module
#   weather_start = prediction_start.date() - timedelta(days=5)
#   weather_end   = prediction_start.date() + timedelta(days=MAX_DAYS + 1)
#   weather_df    = weather_module.get_for_period(lat, lon, weather_start, weather_end)
#
# weather_module.get_for_period() calls the Open-Meteo API and returns a
# DataFrame covering both past (for context merge) and future (for futr_df).
#
# Expected data structure:
#   weather_df columns:
#     ds_real                    : datetime  — hourly timestamp, Spain local, no tzinfo
#     hour                       : int       — hour of day (0–23)
#     om_temperature_2m          : float     — air temperature °C
#     om_shortwave_radiation     : float     — solar radiation W/m²
#     om_cloud_cover_low         : float     — low-level cloud cover %
#     om_vapour_pressure_deficit : float     — vapour pressure deficit kPa
#     (+ any other om_* columns the model was trained with)
#
#   Required date coverage:
#     past  : prediction_start - 5 days  (merged into historical context)
#     future: prediction_start + MAX_DAYS (used to build futr_df)
#
# The exact om_* columns required are listed in:
#     config['futr_exog']  — weather features known at forecast time
#     config['hist_exog']  — weather features used only from history

# <<< REPLACE: weather data >>>
def _fake_weather(start_dt, n_hours):
    """Synthetic July weather for Mallorca.
    Generates all known om_* columns — script validates only required ones.
    Replace this entire block with a real weather API call in production.
    """
    rows = []
    for i in range(n_hours):
        ts   = start_dt + timedelta(hours=i)
        h    = ts.hour
        temp = 26 + 4 * np.sin(np.pi * max(h - 6, 0) / 14)
        rows.append({
            'ds_real':                     ts,
            'hour':                        h,
            'om_temperature_2m':           temp,
            'om_apparent_temperature':     temp - 1.5 + np.random.normal(0, 0.3),
            'om_shortwave_radiation':      max(0, 650 * np.sin(np.pi * max(h - 6, 0) / 12)),
            'om_cloud_cover':              max(0, min(100, 15 + np.random.normal(0, 10))),
            'om_cloud_cover_low':          max(0, min(100, 10 + np.random.normal(0, 8))),
            'om_vapour_pressure_deficit':  max(0, 1.2 + 0.8 * np.sin(np.pi * max(h - 6, 0) / 14)),
            'om_wind_speed_10m':           max(0, 8 + np.random.normal(0, 3)),
            'om_precipitation':            0.0,
            'om_rain':                     0.0,
        })
    return pd.DataFrame(rows)

weather_start = prediction_start - timedelta(days=5)
weather_df    = _fake_weather(weather_start, n_hours=5 * 24 + MAX_DAYS * 24 + 4)
# <<< END REPLACE >>>

weather_df['ds_real'] = pd.to_datetime(weather_df['ds_real'])
om_cols = [c for c in ALL_FEATURES if c.startswith('om_')]
missing_weather = [c for c in om_cols if c not in weather_df.columns]
if missing_weather:
    raise ValueError(f'Weather DataFrame missing columns: {missing_weather}')

print(f'\nWeather rows : {len(weather_df)}')
print(f'  om_* cols  : {[c for c in weather_df.columns if c.startswith("om_")]}')
print(f'  Range      : {weather_df["ds_real"].min()} → {weather_df["ds_real"].max()}')


# =============================================================================
# STEP 5 — Build context_df  [no changes needed]
# =============================================================================
# Combines snapshot history + weather into the NeuralForecast input panel.
#
# Output schema (context_df):
#   unique_id  : str    — camera slug
#   ds         : int    — step index (0 to INPUT_SIZE-1), NOT a real timestamp
#   y          : float  — crowd count at that step
#   hour       : int    — hour of day (temporal feature)
#   day_of_week: int    — 0=Mon … 6=Sun
#   month      : int    — 1–12
#   is_weekend : int    — 0 or 1
#   is_summer  : int    — 1 if month in {6,7,8}
#   om_*       : float  — weather values merged by nearest timestamp

ctx_rows = []
for snap in snapshot_rows:
    ts  = snap['ts']
    row = {'ds_real': ts, 'y': snap['crowd']}
    for col in ALL_FEATURES:
        if col in TEMPORAL_BUILDERS:
            row[col] = TEMPORAL_BUILDERS[col](ts)
    ctx_rows.append(row)

context_df = pd.DataFrame(ctx_rows)
context_df['unique_id'] = uid
context_df = context_df.sort_values('ds_real').tail(INPUT_SIZE).reset_index(drop=True)
context_df['ds'] = range(len(context_df))

# Merge weather into context (nearest-timestamp match)
for idx, row in context_df.iterrows():
    nearest = (weather_df['ds_real'] - row['ds_real']).abs().idxmin()
    for col in om_cols:
        context_df.loc[idx, col] = float(weather_df.loc[nearest, col])

context_df = context_df.fillna(0)

print(f'\ncontext_df : {context_df.shape}')
print(f'  ds range : {context_df["ds"].min()} → {context_df["ds"].max()}')
print(f'  y range  : {context_df["y"].min():.1f} → {context_df["y"].max():.1f}')


# =============================================================================
# STEP 6 — Build futr_df  [no changes needed]
# =============================================================================
# H rows of future exogenous features starting right after the last context step.
# Temporal features are computed from the timestamp.
# Weather features come from the forecast portion of weather_df.
# Steps beyond the available weather window are forward-filled.
#
# Output schema (futr_df):
#   unique_id  : str    — camera slug (same as context_df)
#   ds         : int    — continues from context_df ds (INPUT_SIZE to INPUT_SIZE+H)
#   hour       : int    — future hour of day
#   day_of_week: int
#   month      : int
#   is_weekend : int
#   is_summer  : int
#   om_temperature_2m : float  — NWP forecast (futr_exog only)

last_ds      = int(context_df['ds'].max())
last_real    = context_df['ds_real'].iloc[-1]
futr_weather = weather_df[weather_df['ds_real'] > last_real].head(H)

futr_rows = []
for step in range(1, H + 1):
    future_ts = last_real + timedelta(hours=step)
    row = {'unique_id': uid, 'ds': last_ds + step}

    for col in FUTR_FEATURES:
        if col in TEMPORAL_BUILDERS:
            row[col] = TEMPORAL_BUILDERS[col](future_ts)

    futr_om = [c for c in FUTR_FEATURES if c.startswith('om_')]
    if step <= len(futr_weather):
        w = futr_weather.iloc[step - 1]
        for col in futr_om:
            row[col] = float(w[col]) if col in w.index and pd.notna(w[col]) else 0.0
    else:
        for col in futr_om:
            row[col] = futr_rows[-1].get(col, 0.0) if futr_rows else 0.0

    futr_rows.append(row)

futr_df = pd.DataFrame(futr_rows)[['unique_id', 'ds'] + FUTR_FEATURES].fillna(0)

print(f'\nfutr_df    : {futr_df.shape}  (H={H} = {MAX_DAYS} days x {HOURS_PER_DAY} h/day)')


# =============================================================================
# STEP 7 — Static features  [no changes needed]
# =============================================================================
# Per-camera statistics saved during training in static_features.csv.
# For a camera not in training data, computed from its current context y-series.
#
# static_features.csv schema:
#   unique_id   : str   — camera slug
#   stat_mean_y : float — mean historical crowd count (helps model calibrate scale)
#   stat_cv     : float — coefficient of variation (helps model understand volatility)

stat_cols = [c for c in static_df.columns if c != 'unique_id']

if uid in static_df['unique_id'].values:
    cam_static = static_df[static_df['unique_id'] == uid].copy()
    print(f'\ncam_static : from static_features.csv')
else:
    y = context_df['y']
    stat_row = {'unique_id': uid}
    for col in stat_cols:
        stat_row[col] = STATIC_BUILDERS[col](y) if col in STATIC_BUILDERS else 0.0
    cam_static = pd.DataFrame([stat_row])
    print(f'\ncam_static : computed from context (new camera)')

print(f'  {cam_static.to_dict(orient="records")[0]}')


# =============================================================================
# STEP 8 — Inference  [no changes needed]
# =============================================================================

# Patch for pandas 2.x + older neuralforecast/utilsforecast combinations
try:
    import utilsforecast.processing as _ufp
    _orig_is_nan = _ufp.is_nan
    def _is_nan_compat(s):
        return s.isna() if isinstance(s, pd.Series) else _orig_is_nan(s)
    _ufp.is_nan = _is_nan_compat
except Exception:
    pass

has_static = any(
    hasattr(m, 'stat_exog_list') and bool(getattr(m, 'stat_exog_list', None))
    for m in nf.models
)

print('\nRunning inference...')
forecast = nf.predict(
    df        = context_df[['unique_id', 'ds', 'y'] + ALL_FEATURES].fillna(0),
    static_df = cam_static.fillna(0) if has_static else None,
    futr_df   = futr_df,
).reset_index(drop=True)

# Detect model output column name (TFT, LSTM, etc.)
# NeuralForecast may return unique_id/ds as index or as columns depending on version.
# We move them to columns if needed, then pick the remaining prediction column.
if 'unique_id' not in forecast.columns:
    forecast = forecast.reset_index()
SKIP = {'unique_id', 'ds', 'index'}
model_col = next(c for c in forecast.columns if c not in SKIP)
forecast[model_col] = forecast[model_col].clip(lower=0)

print(f'  Output column : {model_col}')
print(f'  Forecast rows : {len(forecast)}')


# =============================================================================
# STEP 9 — Format output  [no changes needed]
# =============================================================================
# Maps integer ds steps back to real Spain-local timestamps.
# Builds per-hour prediction dicts with occupancy classification.
# Night hours (outside HOUR_MIN–HOUR_MAX) are included as available=False.

def classify(crowd_count, max_cc):
    if not max_cc:
        return None
    r = crowd_count / max_cc
    for thresh, level in OCCUPANCY_THRESHOLDS:
        if r >= thresh:
            return level
    return 'VERY_LOW'


step_to_cc   = dict(zip(forecast['ds'], forecast[model_col]))
last_floored = last_real.replace(minute=0, second=0, microsecond=0)

if last_floored.hour >= HOUR_MAX:
    first_ts = (last_floored + timedelta(days=1)).replace(hour=HOUR_MIN, minute=0)
else:
    first_ts = last_floored + timedelta(hours=1)
    if first_ts.hour < HOUR_MIN:
        first_ts = first_ts.replace(hour=HOUR_MIN, minute=0)

pred_map = {}
ts = first_ts
for step in sorted(step_to_cc):
    pred_map[ts.strftime('%Y-%m-%dT%H:00')] = step_to_cc[step]
    ts += timedelta(hours=1)
    if ts.hour > HOUR_MAX:
        ts = (ts + timedelta(days=1)).replace(hour=HOUR_MIN, minute=0)

start_day   = first_ts.replace(hour=0, minute=0)
predictions = []

for day_offset in range(MAX_DAYS):
    current_day = start_day + timedelta(days=day_offset)
    for hour in range(24):
        t   = current_day.replace(hour=hour)
        key = t.strftime('%Y-%m-%dT%H:00')
        if key in pred_map:
            cc = round(pred_map[key], 1)
            predictions.append({
                'timestamp':       t.isoformat(),
                'hour':            hour,
                'crowd_count':     cc,
                'available':       True,
                'occupancy_level': classify(cc, max_crowd),
                'occupancy_ratio': round(cc / max_crowd, 3),
            })
        else:
            predictions.append({
                'timestamp':       t.isoformat(),
                'hour':            hour,
                'crowd_count':     0,
                'available':       False,   # night — no prediction
                'occupancy_level': None,
                'occupancy_ratio': None,
            })


# =============================================================================
# RESULT
# =============================================================================

result = {
    'model':           MODEL_DIR.name,   # e.g. tft_model_3d
    'webcam':          uid,
    'horizon_days':    MAX_DAYS,
    'max_crowd_count': max_crowd,
    'last_data':       last_real.isoformat(),
    'futr_features':   FUTR_FEATURES,
    'hist_features':   HIST_FEATURES,
    'predictions':     predictions,
}

daytime = [p for p in predictions if p['available']]

print(f'\n{"─" * 62}')
print(f'  RESULT  model={MODEL_DIR.name}  camera={uid}')
print(f'{"─" * 62}')
print(f'  Total slots   : {len(predictions)} ({MAX_DAYS} days x 24h)')
print(f'  Daytime slots : {len(daytime)} ({HOUR_MIN}:00-{HOUR_MAX}:00, available=True)')
print(f'  Last context  : {result["last_data"]}')
print()
print(f'  {"Timestamp":<22} {"Count":>6} {"Ratio":>7}  Level')
print(f'  {"─" * 48}')
current_day = None
for p in daytime:
    day = p['timestamp'][:10]
    if day != current_day:
        current_day = day
        print(f'  --- {day} ---')
    print(f'  {p["timestamp"]:<22} {p["crowd_count"]:>6.1f} {p["occupancy_ratio"]:>7.3f}  {p["occupancy_level"]}')
print()
print('Done.')