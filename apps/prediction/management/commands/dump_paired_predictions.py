"""
Management command: dump_paired_predictions

Dumps PER-ROW predictions (not aggregates) for a list of model sets, mirroring
the actuals loading and walk-forward window iteration used by
`evaluate_model_sets`. Produces a tidy long CSV that pairs predictions across
model families on identical (base_beach, ds_real) daytime targets, so a
Diebold-Mariano test can be run downstream.

For each camera (max_crowd_count > 0 with Snapshot data), each horizon key in
[3d, 10d, 15d], each walk-forward window (stride = horizon_days) in [since, until]:
calls service.predict(cam, horizon_days, since=window_start_utc, model_set=set)
and aligns the returned daytime predictions (8 <= hour <= 20) to the Snapshot
actuals by hour-truncated timestamp ('YYYY-MM-DDTHH').

Emits one row per matched (model_set, horizon, cam, timestamp):
    scenario, horizon, model, unique_id, base_beach, cutoff, ds_real, lead, y, y_pred

Two scenario-tagged copies of each row are written:
    scenario='season' for window months in {4,5,6,7,8,9}
    scenario='summer' for window months in {6,7,8}  (additional copy)

y / y_pred are RAW counts (not normalized).

Usage:
    python manage.py dump_paired_predictions \
        --since 2025-04-01 --until 2025-09-30 \
        --sets tft_20260323_180247 lstm --out paired_summer.csv
"""

import csv
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone as dj_tz

from apps.prediction.models import Snapshot
from apps.prediction.scripts import weather_module
from apps.prediction.tft_service import TFTService
from apps.webcam.models import WebCam

SEASON_MONTHS = {4, 5, 6, 7, 8, 9}
SUMMER_MONTHS = {6, 7, 8}
HOUR_MIN, HOUR_MAX = 8, 20   # daytime window standardised to 8-20 (was 8-19)
HORIZON_DAYS = {'3d': 3, '10d': 10, '15d': 15}


def _cam_lat(cam) -> float:
    if cam.camera_latitude:
        return float(cam.camera_latitude)
    if cam.beach.coordenadas_geograficas:
        return float(cam.beach.coordenadas_geograficas.split(',')[0])
    return 39.5


def _cam_lon(cam) -> float:
    if cam.camera_longitude:
        return float(cam.camera_longitude)
    if cam.beach.coordenadas_geograficas:
        return float(cam.beach.coordenadas_geograficas.split(',')[1])
    return 2.6


def load_all_actuals(cam, since: date, until: date) -> list:
    date_from = datetime(since.year, since.month, since.day, tzinfo=timezone.utc)
    date_to = datetime(until.year, until.month, until.day, tzinfo=timezone.utc)
    rows = []
    for s in (
        Snapshot.objects
        .filter(webcam=cam, predicted_crowd_count__isnull=False,
                ts__gte=date_from, ts__lt=date_to)
        .order_by('ts')
    ):
        ts = dj_tz.localtime(s.ts).replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
        if HOUR_MIN <= ts.hour <= HOUR_MAX:
            rows.append({'timestamp': ts.isoformat(), 'crowd_count': float(s.predicted_crowd_count)})
    return rows


def _model_family(set_name: str) -> str:
    return 'TFT' if set_name.startswith('tft') or set_name.startswith('cross') else 'LSTM'


class Command(BaseCommand):
    help = 'Dump per-row paired predictions for a list of model sets (tidy long CSV)'

    def add_arguments(self, parser):
        parser.add_argument('--since', default=None)
        parser.add_argument('--until', default=None)
        parser.add_argument('--sets', nargs='*', default=None)
        parser.add_argument('--out', default='paired_predictions.csv')

    def handle(self, *args, **options):
        service = TFTService()

        configured_sets = getattr(settings, 'TFT_MODEL_SETS', {})
        all_set_names = list(configured_sets.keys()) + [
            n for n in service._discovered if n not in configured_sets
        ]
        set_names = options['sets'] or all_set_names
        if not set_names:
            self.stderr.write('No model sets found.')
            return

        cams = list(WebCam.objects.select_related('beach').filter(max_crowd_count__gt=0))
        self.stdout.write(f'{len(cams)} cameras with max_crowd_count set')

        if options['since']:
            global_start = date.fromisoformat(options['since'])
        else:
            first = Snapshot.objects.order_by('ts').values_list('ts', flat=True).first()
            global_start = first.date() if first else date(2024, 1, 1)

        if options['until']:
            global_end = date.fromisoformat(options['until'])
        else:
            last = Snapshot.objects.order_by('-ts').values_list('ts', flat=True).first()
            global_end = last.date() if last else date.today() - timedelta(days=1)

        self.stdout.write(f'Range: {global_start} -> {global_end}\n')

        # ── Pre-fetch weather ──────────────────────────────────────────────────
        self.stdout.write('Pre-fetching weather archive...')
        seen_coords = set()
        for cam in cams:
            lat_r, lon_r = round(_cam_lat(cam), 2), round(_cam_lon(cam), 2)
            if (lat_r, lon_r) in seen_coords:
                continue
            seen_coords.add((lat_r, lon_r))
            try:
                weather_module.prefetch(lat_r, lon_r, global_start, global_end)
            except Exception as e:
                self.stderr.write(f'  weather prefetch failed for ({lat_r}, {lon_r}): {e}')
        self.stdout.write('done\n')

        # ── Load model sets ────────────────────────────────────────────────────
        loaded_sets = []
        for set_name in set_names:
            try:
                service._ensure_loaded(set_name)
                if service.model_sets.get(set_name):
                    loaded_sets.append(set_name)
                    self.stdout.write(f'Loaded: {set_name} ({_model_family(set_name)})')
                else:
                    self.stderr.write(f'No models found for {set_name}, skipping')
            except Exception as e:
                self.stderr.write(f'Could not load {set_name}: {type(e).__name__}: {e}')

        if not loaded_sets:
            self.stderr.write('No model sets loaded.')
            return

        # ── Load actuals (slug -> day -> [rows]) ───────────────────────────────
        self.stdout.write('\nLoading actuals from DB...')
        cam_by_day: dict = {}
        for cam in cams:
            rows = load_all_actuals(cam, global_start, global_end + timedelta(days=1))
            if rows:
                by_day = defaultdict(list)
                for r in rows:
                    by_day[date.fromisoformat(r['timestamp'][:10])].append(r)
                cam_by_day[cam.camera_slug] = by_day
        self.stdout.write(f'{len(cam_by_day)} cameras have snapshot data\n')

        # ── Dump per-row predictions ───────────────────────────────────────────
        out_path = Path(options['out'])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ['scenario', 'horizon', 'model', 'unique_id', 'base_beach',
                  'cutoff', 'ds_real', 'lead', 'y', 'y_pred']

        total_rows = 0
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for set_name in loaded_sets:
                family = _model_family(set_name)
                self.stdout.write(f'\n-- {set_name} ({family})')

                for horizon_key, horizon_days in HORIZON_DAYS.items():
                    if horizon_key not in service.model_sets[set_name]:
                        self.stdout.write(f'  [{horizon_key}] not in set — skipping')
                        continue

                    stride = timedelta(days=horizon_days)
                    computed = 0
                    error_count = 0
                    error_types: dict = {}
                    set_rows = 0
                    self.stdout.write(f'  [{horizon_key}] horizon={horizon_days}d', ending=' ')
                    self.stdout.flush()

                    for cam in cams:
                        slug = cam.camera_slug
                        if slug not in cam_by_day:
                            continue
                        max_crowd = cam.max_crowd_count
                        by_day = cam_by_day[slug]
                        window_start = global_start

                        while window_start <= global_end:
                            month = window_start.month
                            in_season = month in SEASON_MONTHS
                            in_summer = month in SUMMER_MONTHS
                            if not in_season:
                                window_start += stride
                                continue

                            window_actuals = []
                            for i in range(horizon_days):
                                window_actuals.extend(by_day.get(window_start + timedelta(days=i), []))
                            if not window_actuals:
                                window_start += stride
                                continue

                            try:
                                since_dt = datetime(window_start.year, window_start.month,
                                                    window_start.day, tzinfo=timezone.utc)
                                pred = service.predict(cam, horizon_days,
                                                       since=since_dt, model_set=set_name)
                            except Exception as e:
                                key = f'{type(e).__name__}: {e}'
                                error_types[key] = error_types.get(key, 0) + 1
                                error_count += 1
                                window_start += stride
                                continue

                            pred_by_ts = {
                                p['timestamp'][:13]: p['crowd_count']
                                for p in pred.get('predictions', [])
                                if p.get('available') and p.get('crowd_count') is not None
                                and HOUR_MIN <= int(p['timestamp'][11:13]) <= HOUR_MAX
                            }

                            matched = []
                            for a in window_actuals:
                                key = a['timestamp'][:13]
                                if key not in pred_by_ts:
                                    continue
                                matched.append((a['timestamp'], a['crowd_count'], pred_by_ts[key]))

                            matched.sort(key=lambda r: r[0])
                            cutoff_iso = window_start.isoformat()
                            for lead, (ds_real, y, y_pred) in enumerate(matched):
                                base = {
                                    'horizon': horizon_key,
                                    'model': family,
                                    'unique_id': slug,
                                    'base_beach': slug,
                                    'cutoff': cutoff_iso,
                                    'ds_real': ds_real,
                                    'lead': lead,
                                    'y': y,
                                    'y_pred': y_pred,
                                }
                                writer.writerow({'scenario': 'season', **base})
                                set_rows += 1
                                if in_summer:
                                    writer.writerow({'scenario': 'summer', **base})
                                    set_rows += 1

                            computed += 1
                            window_start += stride

                    total_rows += set_rows
                    msg = f'(windows={computed}, rows={set_rows})'
                    if error_count:
                        msg += f' errors={error_count}'
                    self.stdout.write(msg)
                    for err_msg, count in sorted(error_types.items(), key=lambda x: -x[1]):
                        self.stderr.write(f'    [{count:>5}x] {err_msg}')

        self.stdout.write(f'\nSaved CSV: {out_path} ({total_rows} rows)')
