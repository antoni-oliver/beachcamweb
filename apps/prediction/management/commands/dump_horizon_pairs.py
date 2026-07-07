"""
Management command: dump_horizon_pairs

Dumps PER-ROW predictions for ONE TFT model set, issuing the 3d, 10d AND 15d
models from the SAME issue dates (a common issue grid). This lets a downstream
Diebold-Mariano test compare each dedicated short-horizon model against the
single 15d model on identical (base_beach, ds_real) daytime targets.

Common issue grid: stride = 15 days (non-overlapping). For each camera
(max_crowd_count > 0 with Snapshot data) and each issue_date in [since, until]
stepping 15 days (fast-forwarded to the camera's first snapshot), and for each
days in [3, 10, 15]:
    service.predict(cam, days, since=issue_date_utc, model_set=SET)
    -> days=3 uses the 3d model, days=10 the 10d model, days=15 the 15d model,
       all issued from the same issue_date.
Returned daytime predictions (8 <= hour <= 20) are aligned to Snapshot actuals
by hour-truncated timestamp ('YYYY-MM-DDTHH'); one row per matched timestamp.

Row columns:
    scenario, model_horizon, unique_id, base_beach, cutoff, ds_real,
    lead_hours, y, y_pred

Two scenario-tagged copies of each row are written, based on the ds_real MONTH:
    scenario='season' if ds_real month in {4,5,6,7,8,9}
    scenario='summer' if ds_real month in {6,7,8}  (additional copy)

y / y_pred are RAW counts (not normalized).

Usage:
    python manage.py dump_horizon_pairs \
        --since 2025-04-01 --until 2025-09-30 \
        --set tft_20260323_180247 --out horizon_pairs.csv
"""

import csv
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from django.core.management.base import BaseCommand
from django.utils import timezone as dj_tz

from apps.prediction.models import Snapshot
from apps.prediction.scripts import weather_module
from apps.prediction.tft_service import TFTService
from apps.webcam.models import WebCam

SEASON_MONTHS = {4, 5, 6, 7, 8, 9}
SUMMER_MONTHS = {6, 7, 8}
HOUR_MIN, HOUR_MAX = 8, 20   # daytime window standardised to 8-20 (was 8-19)
STRIDE_DAYS = 15
MODEL_DAYS = [3, 10, 15]
MODEL_HORIZON = {3: '3d', 10: '10d', 15: '15d'}


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


def issue_date_naive(d: date) -> datetime:
    return datetime(d.year, d.month, d.day)


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


class Command(BaseCommand):
    help = 'Dump per-row predictions for 3d/10d/15d models issued from a common grid (tidy long CSV)'

    def add_arguments(self, parser):
        parser.add_argument('--since', default=None)
        parser.add_argument('--until', default=None)
        parser.add_argument('--set', dest='set_name', default='tft_20260323_180247')
        parser.add_argument('--out', default='horizon_pairs.csv')

    def handle(self, *args, **options):
        service = TFTService()
        set_name = options['set_name']

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

        self.stdout.write(f'Range: {global_start} -> {global_end} (stride={STRIDE_DAYS}d, set={set_name})\n')

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

        # ── Load model set ─────────────────────────────────────────────────────
        try:
            service._ensure_loaded(set_name)
        except Exception as e:
            self.stderr.write(f'Could not load {set_name}: {type(e).__name__}: {e}')
            return
        models = service.model_sets.get(set_name) or {}
        if not models:
            self.stderr.write(f'No models found for {set_name}.')
            return
        self.stdout.write(f'Loaded: {set_name} (horizons: {sorted(models)})')

        # ── Load actuals (slug -> day -> [rows]) and first-snapshot day ────────
        self.stdout.write('\nLoading actuals from DB...')
        cam_by_day: dict = {}
        cam_first_day: dict = {}
        for cam in cams:
            rows = load_all_actuals(cam, global_start, global_end + timedelta(days=1))
            if rows:
                by_day = defaultdict(list)
                for r in rows:
                    by_day[date.fromisoformat(r['timestamp'][:10])].append(r)
                cam_by_day[cam.camera_slug] = by_day
                cam_first_day[cam.camera_slug] = min(by_day)
        self.stdout.write(f'{len(cam_by_day)} cameras have snapshot data\n')

        # ── Dump per-row predictions ───────────────────────────────────────────
        out_path = Path(options['out'])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ['scenario', 'model_horizon', 'unique_id', 'base_beach',
                  'cutoff', 'ds_real', 'lead_hours', 'y', 'y_pred']

        total_rows = 0
        error_count = 0
        error_types: dict = {}

        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for cam in cams:
                slug = cam.camera_slug
                if slug not in cam_by_day:
                    continue
                by_day = cam_by_day[slug]

                # Common issue grid: step STRIDE_DAYS, fast-forward to first snapshot.
                first_day = cam_first_day[slug]
                issue_date = max(global_start, first_day)
                cam_rows = 0
                cam_windows = 0
                self.stdout.write(f'-- {slug}', ending=' ')
                self.stdout.flush()

                while issue_date <= global_end:
                    issue_dt = datetime(issue_date.year, issue_date.month,
                                        issue_date.day, tzinfo=timezone.utc)
                    cutoff_iso = issue_date.isoformat()

                    for days in MODEL_DAYS:
                        model_horizon = MODEL_HORIZON[days]
                        if model_horizon not in models:
                            continue

                        # Daytime actuals over the days-window for this issue date.
                        window_actuals = []
                        for i in range(days):
                            window_actuals.extend(by_day.get(issue_date + timedelta(days=i), []))
                        if not window_actuals:
                            continue

                        try:
                            pred = service.predict(cam, days, since=issue_dt, model_set=set_name)
                        except Exception as e:
                            key = f'{type(e).__name__}: {e}'
                            error_types[key] = error_types.get(key, 0) + 1
                            error_count += 1
                            continue

                        pred_by_ts = {
                            p['timestamp'][:13]: p['crowd_count']
                            for p in pred.get('predictions', [])
                            if p.get('available') and p.get('crowd_count') is not None
                            and HOUR_MIN <= int(p['timestamp'][11:13]) <= HOUR_MAX
                        }

                        for a in window_actuals:
                            key = a['timestamp'][:13]
                            if key not in pred_by_ts:
                                continue
                            ds_real = a['timestamp']
                            ds_dt = datetime.fromisoformat(ds_real)
                            lead_hours = int((ds_dt - issue_date_naive(issue_date)).total_seconds() // 3600)
                            month = ds_dt.month
                            base = {
                                'model_horizon': model_horizon,
                                'unique_id': slug,
                                'base_beach': slug,
                                'cutoff': cutoff_iso,
                                'ds_real': ds_real,
                                'lead_hours': lead_hours,
                                'y': a['crowd_count'],
                                'y_pred': pred_by_ts[key],
                            }
                            if month in SEASON_MONTHS:
                                writer.writerow({'scenario': 'season', **base})
                                cam_rows += 1
                            if month in SUMMER_MONTHS:
                                writer.writerow({'scenario': 'summer', **base})
                                cam_rows += 1

                        cam_windows += 1

                    issue_date += timedelta(days=STRIDE_DAYS)

                total_rows += cam_rows
                self.stdout.write(f'(windows={cam_windows}, rows={cam_rows})')

        if error_count:
            self.stderr.write(f'\nPredict errors: {error_count}')
            for err_msg, count in sorted(error_types.items(), key=lambda x: -x[1]):
                self.stderr.write(f'    [{count:>5}x] {err_msg}')

        self.stdout.write(f'\nSaved CSV: {out_path} ({total_rows} rows)')
