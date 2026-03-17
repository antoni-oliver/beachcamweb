# TFT Beach Occupancy Prediction — Usage Guide

**Script:** `tft_usage_example.py`  
**Runs anywhere** — no Django, no database, no webcam required.

```bash
pip install neuralforecast torch pandas numpy
python tft_usage_example.py
```

---

## What the script does

Loads a trained TFT model from disk and generates hourly beach occupancy forecasts. It runs end-to-end with **synthetic data by default**. To use real data, replace exactly **three marked blocks**.

```
MODEL WEIGHTS (nf_model/)
      +
CROWD HISTORY  → snapshot_rows   ─┐
      +                            ├─► context_df ─┐
WEATHER DATA   → weather_df      ─┘                ├─► nf.predict() → predictions[]
      +                                             │
STATIC STATS   → cam_static     ───────────────────┘
```

---

## Setup — two variables at the top

Open the script and set these before running:

```python
MODEL_DIR = Path('/absolute/path/to/tft_model_3d')
HORIZON   = '3d'    # '3d' | '10d' | '15d'
```

**Model directory structure** (what `MODEL_DIR` must contain):

```
tft_model_3d/                    ← point MODEL_DIR here
├── nf_model/                    NeuralForecast checkpoint files
│   ├── model.ckpt
│   └── ...
├── config.json                  hyperparameters + feature lists + metrics
└── static_features.csv          per-camera statistics from training
```

**Available horizons:**

| Value | Steps | Days | Best for |
|---|---|---|---|
| `3d` | H=36 | 3 | Short-term, highest accuracy |
| `10d` | H=120 | 10 | Weekly planning |
| `15d` | H=180 | 14 | Two-week outlook |

> `MAX_DAYS = ceil(H / 13)` — 13 daytime hours/day (08:00–20:00 inclusive). H=36→3d, H=120→10d, H=180→14d.

---

## The three replacements

Everything else in the script is boilerplate that never changes. Only these three blocks need real data.

---

### Replacement 1 — Camera metadata

**Search for:** `# <<< REPLACE: camera metadata >>>`

**Default (fake):**
```python
uid       = 'platja-de-muro'
max_crowd = 80
lat, lon  = 39.8028, 3.1256
```

**Production (Django):**
```python
from apps.webcam.models import WebCam

webcam    = WebCam.objects.select_related('beach').get(camera_slug='platja-de-muro')
uid       = webcam.camera_slug
max_crowd = webcam.max_crowd_count
lat       = float(webcam.camera_latitude)
lon       = float(webcam.camera_longitude)
```

| Variable | Type | Description |
|---|---|---|
| `uid` | `str` | Camera slug. Must match a row in `static_features.csv`, or be handled as a new camera (see Step 7 in the script). |
| `max_crowd` | `int` | P99-calibrated maximum occupancy. Used only for `occupancy_ratio` and level classification. |
| `lat`, `lon` | `float` | GPS coordinates. Only needed for the weather API call in Replacement 3. |

---

### Replacement 2 — Crowd count history (snapshots)

**Search for:** `# <<< REPLACE: crowd count history >>>`

**Default (fake):** A sine-wave crowd pattern generated for the last 96 hours.

**Production (Django):**
```python
from apps.prediction.models import Snapshot

qs = (Snapshot.objects
      .filter(webcam=webcam, predicted_crowd_count__isnull=False)
      .order_by('-ts')[:INPUT_SIZE * 3])

snapshot_rows = [
    {'ts':    s.ts.astimezone(_SPAIN_TZ).replace(tzinfo=None),
     'crowd': float(s.predicted_crowd_count)}
    for s in reversed(list(qs))
    if HOUR_MIN <= s.ts.astimezone(_SPAIN_TZ).hour <= HOUR_MAX
]
```

**Expected data structure:**
```python
snapshot_rows = [
    {'ts': datetime(2025, 7, 14,  8, 0),  'crowd': 5.0},
    {'ts': datetime(2025, 7, 14,  9, 0),  'crowd': 12.3},
    {'ts': datetime(2025, 7, 14, 10, 0),  'crowd': 24.1},
    {'ts': datetime(2025, 7, 14, 11, 0),  'crowd': 38.7},
    # ... one row per daytime hour (08:00–20:00)
    {'ts': datetime(2025, 7, 14, 20, 0),  'crowd': 8.7},
    {'ts': datetime(2025, 7, 15,  8, 0),  'crowd': 6.2},  # next day
    # ... continues
]
```

**Rules:**
- `ts` must be **Spain local time** (`Europe/Madrid`) with `tzinfo` stripped
- Only hours **08:00–20:00** are included — night hours are filtered out
- **Chronological order** (oldest first)
- Minimum `INPUT_SIZE` (48) rows; the script automatically takes the last 48

`predicted_crowd_count` values are produced by the Bayesian VGG-19 crowd counting model that processes webcam images. In BeachCamWeb, these are populated by:
```bash
python manage.py run_crowd_counting
python manage.py recount_snapshots --webcam platja-de-muro --only-null
```

---

### Replacement 3 — Weather data

**Search for:** `# <<< REPLACE: weather data >>>`

**Default (fake):** Synthetic July values for Mallorca (temperature, solar radiation, cloud cover, VPD).

**Production (BeachCamWeb weather_module):**
```python
import sys
sys.path.insert(0, '/path/to/beachcamweb/apps/prediction/scripts')
import weather_module

weather_start = prediction_start.date() - timedelta(days=5)
weather_end   = prediction_start.date() + timedelta(days=MAX_DAYS + 1)
weather_df    = weather_module.get_for_period(lat, lon, weather_start, weather_end)
```

`weather_module.get_for_period()` calls the **Open-Meteo API** (free, no key required) and caches results locally. It returns a DataFrame with the exact columns the model needs.

**You can also call Open-Meteo directly:**
```python
import requests

params = {
    'latitude':         lat,
    'longitude':        lon,
    'hourly':           'temperature_2m,apparent_temperature,shortwave_radiation,cloud_cover,cloud_cover_low,vapour_pressure_deficit',
    'timezone':         'Europe/Madrid',
    'past_days':        5,
    'forecast_days':    MAX_DAYS + 1,
}
r = requests.get('https://api.open-meteo.com/v1/forecast', params=params)
data = r.json()['hourly']

weather_df = pd.DataFrame({
    'ds_real':                     pd.to_datetime(data['time']),
    'hour':                        pd.to_datetime(data['time']).hour,
    'om_temperature_2m':           data['temperature_2m'],
    'om_apparent_temperature':     data['apparent_temperature'],
    'om_shortwave_radiation':      data['shortwave_radiation'],
    'om_cloud_cover':              data['cloud_cover'],
    'om_cloud_cover_low':          data['cloud_cover_low'],
    'om_vapour_pressure_deficit':  data['vapour_pressure_deficit'],
})
```

**Expected data structure:**

| Column | Type | Description |
|---|---|---|
| `ds_real` | `datetime` | Hourly timestamp, Spain local time, no `tzinfo` |
| `hour` | `int` | Hour of day (0–23) |
| `om_temperature_2m` | `float` | Air temperature °C — **futr_exog** |
| `om_apparent_temperature` | `float` | Feels-like temperature °C — **futr_exog** |
| `om_cloud_cover` | `float` | Total cloud cover % (0–100) — **futr_exog** |
| `om_shortwave_radiation` | `float` | Solar radiation W/m² — **hist_exog** |
| `om_cloud_cover_low` | `float` | Low-level cloud cover % (0–100) — **hist_exog** |
| `om_vapour_pressure_deficit` | `float` | VPD kPa — **hist_exog** |

**Required date range:**

```
prediction_start - 5 days ──────── prediction_start ──────── + MAX_DAYS
        ↑                                  ↑                      ↑
  merged into context_df             forecast starts          forecast ends
  (hist_exog features)                (futr_df)
```

The exact columns required depend on what the model was trained with. Always check:
```python
print('futr (need NWP forecast):', config['futr_exog'])
print('hist (historical only):',   config['hist_exog'])
```

The script validates that all required columns are present and raises a `ValueError` if any are missing.

---

## Output structure

```python
result = {
    'model':           '3d',
    'webcam':          'platja-de-muro',
    'horizon_days':    3,
    'max_crowd_count': 80,
    'last_data':       '2025-07-14T19:00:00',   # last context timestamp
    'futr_features':   ['hour', 'day_of_week', 'month', 'is_weekend', 'is_summer',
                        'om_temperature_2m', 'om_apparent_temperature', 'om_cloud_cover'],
    'hist_features':   ['om_cloud_cover_low', 'om_shortwave_radiation',
                        'om_vapour_pressure_deficit'],
    'predictions': [
        # daytime slot (available=True)
        {
            'timestamp':       '2025-07-15T13:00:00',
            'hour':            13,
            'crowd_count':     52.4,
            'available':       True,
            'occupancy_level': 'MEDIUM',     # >= 50% of max_crowd_count
            'occupancy_ratio': 0.655,        # crowd_count / max_crowd_count
        },
        # night slot (available=False)
        {
            'timestamp':       '2025-07-15T02:00:00',
            'hour':            2,
            'crowd_count':     0,
            'available':       False,
            'occupancy_level': None,
            'occupancy_ratio': None,
        },
    ]
}
```

**Filter to daytime only and iterate by day:**
```python
daytime = [p for p in result['predictions'] if p['available']]

current_day = None
for p in daytime:
    day = p['timestamp'][:10]
    if day != current_day:
        current_day = day
        print(f'--- {day} ---')
    print(p['timestamp'], p['crowd_count'], p['occupancy_level'])
```

**Occupancy levels:**

| Level | Threshold |
|---|---|
| `HIGH` | ≥ 75% of `max_crowd_count` |
| `MEDIUM` | ≥ 50% |
| `LOW` | ≥ 25% |
| `VERY_LOW` | < 25% |

---

## Full pipeline summary

| Step | What it does | Change needed? |
|---|---|---|
| 1 | Load model from `MODEL_DIR` | Set `MODEL_DIR` path |
| 2 | Read feature lists from `config.json` | Never |
| 3 | Camera metadata | **Replace** with Django or hardcode |
| 4 | Crowd count history | **Replace** with Django Snapshots |
| 5 | Weather data | **Replace** with API call |
| 6 | Build `context_df` | Never |
| 7 | Build `futr_df` | Never |
| 8 | Load static features | Never |
| 9 | Run `nf.predict()` | Never |
| 10 | Map steps to timestamps | Never |
| 11 | Print result | Never |
