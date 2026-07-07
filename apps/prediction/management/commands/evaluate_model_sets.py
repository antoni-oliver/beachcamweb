"""
Management command: evaluate_model_sets

Evaluates TFT_MODEL_SETS × all horizons. Outputs:
  - model_evaluation.json  (machine-readable, consumed by TFTService)
  - model_evaluation.csv   (human-readable)

Caches per-window errors so re-runs only compute new (model_set, horizon, camera, window) combos.
Adding a new model set only runs predictions for that set — existing results are reused.

Usage:
    python manage.py evaluate_model_sets
    python manage.py evaluate_model_sets --sets tft_v8 tft_v9
    python manage.py evaluate_model_sets --since 2024-04-01 --until 2024-09-30
    python manage.py evaluate_model_sets --clear-cache
    python manage.py evaluate_model_sets --clear-weather-cache
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import csv
import json
from collections import defaultdict
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone as dj_tz
import numpy as np

from apps.prediction.scripts import weather_module
from apps.webcam.models import WebCam
from apps.prediction.models import Snapshot
from apps.prediction.tft_service import TFTService
import sys as _sys
from pathlib import Path as _Path
from django.conf import settings as _settings

SEASON_MONTHS = {4, 5, 6, 7, 8, 9}
SUMMER_MONTHS = {6, 7, 8}   # standardised to match run_a2/retrain/train scripts (was {7,8})
HOUR_MIN, HOUR_MAX = 8, 20   # daytime window standardised to 8-20 (was 8-19)


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


def relmae(errors: list) -> float | None:
    return sum(errors) / len(errors) if errors else None


def load_all_actuals(cam, since: date, until: date) -> list:
    date_from = datetime(since.year, since.month, since.day, tzinfo=timezone.utc)
    date_to   = datetime(until.year, until.month, until.day, tzinfo=timezone.utc)
    rows = []
    for s in (
        Snapshot.objects
        .filter(webcam=cam, predicted_crowd_count__isnull=False, ts__gte=date_from, ts__lt=date_to)
        .order_by('ts')
    ):
        ts = dj_tz.localtime(s.ts).replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
        if HOUR_MIN <= ts.hour <= HOUR_MAX:
            rows.append({'timestamp': ts.isoformat(), 'crowd_count': float(s.predicted_crowd_count)})
    return rows


def p90(actuals: list) -> float:
    vals = [r['crowd_count'] for r in actuals]
    return float(np.percentile(vals, 90)) if vals else 0.0


def _cache_key(model_set, model_key, slug, window_start: date) -> str:
    return f"{model_set}|{model_key}|{slug}|{window_start}"


def _load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    return {}


def _save_cache(cache_path: Path, cache: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))


class Command(BaseCommand):
    help = 'Evaluate TFT model sets and save best-model JSON for dynamic selection'

    def add_arguments(self, parser):
        parser.add_argument('--since', default=None)
        parser.add_argument('--until', default=None)
        parser.add_argument('--sets', nargs='*', default=None)
        parser.add_argument('--out', default='model_set_evaluation')
        parser.add_argument('--clear-cache', action='store_true', help='Clear the window error cache')
        parser.add_argument('--clear-weather-cache', action='store_true')
        parser.add_argument('--cap-k', dest='cap_k', type=float, default=1.5,
                            help='Cap real actuals above k*P90 per camera before scoring '
                                 '(outlier removal; 0 disables). Predictions never capped.')

    def handle(self, *args, **options):
        if options['clear_weather_cache']:
            weather_module.clear_archive()
            self.stdout.write('Weather archive cache cleared.')

        # ── Cache path (alongside TFT_EVAL_JSON) ──────────────────────────────
        eval_json_setting = getattr(settings, 'TFT_EVAL_JSON', 'apps/prediction/model_evaluation.json')
        eval_json_path = Path(eval_json_setting)
        if not eval_json_path.is_absolute():
            eval_json_path = Path(settings.BASE_DIR) / eval_json_path
        cap_k = options['cap_k']
        # Cap changes the actuals, so capped/uncapped runs use separate caches.
        cache_name = (f'model_evaluation_cache_cap{cap_k}_p90.json' if cap_k and cap_k > 0
                      else 'model_evaluation_cache_p90.json')
        cache_path = eval_json_path.with_name(cache_name)

        if options['clear_cache']:
            if cache_path.exists():
                cache_path.unlink()
                self.stdout.write('Evaluation cache cleared.')
            else:
                self.stdout.write('No cache file found.')

        window_cache = _load_cache(cache_path)
        self.stdout.write(f'Cache: {len(window_cache)} cached windows from {cache_path.name}')

        # ── Model sets to evaluate ─────────────────────────────────────────────
        configured_sets = getattr(settings, 'TFT_MODEL_SETS', {})
        service = TFTService()
        all_set_names = list(configured_sets.keys()) + [
            n for n in service._discovered if n not in configured_sets
        ]
        set_names = options['sets'] or all_set_names
        if not set_names:
            self.stderr.write('No model sets found.')
            return

        cams = list(WebCam.objects.select_related('beach').filter(max_crowd_count__gt=0))
        self.stdout.write(f'{len(cams)} cameras with max_crowd_count set')

        # ── Date range ─────────────────────────────────────────────────────────
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
            self.stdout.write(f'  ({lat_r}, {lon_r}) ...', ending=' ')
            self.stdout.flush()
            weather_module.prefetch(lat_r, lon_r, global_start, global_end)
            self.stdout.write('done')

        info = weather_module.archive_info()
        self.stdout.write(f'Cache: {info["disk_entries"]} monthly chunks, {info["coords"]} coords\n')

        # ── Load model sets ────────────────────────────────────────────────────
        loaded_sets = []
        for set_name in set_names:
            try:
                service._ensure_loaded(set_name)
                if service.model_sets.get(set_name):
                    loaded_sets.append(set_name)
                    self.stdout.write(f'Loaded: {set_name}')
                else:
                    self.stderr.write(f'No models found for {set_name}, skipping')
            except Exception as e:
                self.stderr.write(f'Could not load {set_name}: {e}')

        if not loaded_sets:
            self.stderr.write('No model sets loaded.')
            return

        # ── Load actuals ───────────────────────────────────────────────────────
        self.stdout.write('\nLoading actuals from DB...')
        cam_by_day: dict = {}
        cam_p90:    dict = {}
        cam_first_snapshot: dict = {}  # slug -> date of first snapshot ever
        n_capped = 0

        for cam in cams:
            rows = load_all_actuals(cam, global_start, global_end + timedelta(days=1))
            if rows and cap_k and cap_k > 0:
                # Clamp the REAL actuals above k*P90 per camera (mirrors
                # crowd_outliers.cap_outliers). Predictions are never capped.
                pos = [r['crowd_count'] for r in rows if r['crowd_count'] > 0]
                if pos:
                    thr = cap_k * float(np.percentile(pos, 90))
                    for r in rows:
                        if r['crowd_count'] > thr:
                            r['crowd_count'] = thr
                            n_capped += 1
            if rows:
                by_day = defaultdict(list)
                for r in rows:
                    by_day[date.fromisoformat(r['timestamp'][:10])].append(r)
                cam_by_day[cam.camera_slug] = by_day
                cam_p90[cam.camera_slug] = p90(rows)

            first_ts = (
                Snapshot.objects
                .filter(webcam=cam, predicted_crowd_count__isnull=False)
                .order_by('ts')
                .values_list('ts', flat=True)
                .first()
            )
            if first_ts is not None:
                d = first_ts.date() if hasattr(first_ts, 'date') else date.fromisoformat(str(first_ts)[:10])
                cam_first_snapshot[cam.camera_slug] = d

        self.stdout.write(f'{len(cam_by_day)} cameras have snapshot data')
        if cap_k and cap_k > 0:
            self.stdout.write(f'Outlier cap k={cap_k}*P90 on actuals: clamped {n_capped:,} rows')
        self.stdout.write('')

        # ── Evaluate — skip cached windows ────────────────────────────────────
        all_errors: dict = {}
        new_cached = 0

        for set_name in loaded_sets:
            all_errors[set_name] = {}
            self.stdout.write(f'\n-- {set_name}')

            for model_key, m in service.model_sets[set_name].items():
                horizon_days = m['max_days']
                stride = timedelta(days=horizon_days)
                bucket = defaultdict(lambda: {'all': [], 'summer': [], 'season': []})
                cached_hits = 0
                computed = 0
                error_count = 0
                error_types: dict = {}  # "ExcType: msg" -> count
                self.stdout.write(f'  [{model_key}] horizon={horizon_days}d', ending=' ')
                self.stdout.flush()

                for cam in cams:
                    slug = cam.camera_slug
                    if slug not in cam_by_day:
                        continue
                    # STATISTICAL lens: normalise by the per-series P90 of (capped) actual
                    # daytime counts — the same denominator as the cross-family headline /
                    # thesis metric, so the ranking matches the reported relMAE. (Operational
                    # max_crowd_count is the alternative "which snapshot to serve" reading.)
                    denom = max(float(cam_p90.get(slug, 0.0)), 1.0)
                    by_day = cam_by_day[slug]
                    first_snap = cam_first_snapshot.get(slug)
                    window_start = global_start
                    # Fast-forward to first window where camera has history
                    if first_snap and first_snap > window_start:
                        window_start = first_snap

                    while window_start <= global_end:
                        ck = _cache_key(set_name, model_key, slug, window_start)

                        if ck in window_cache and window_cache[ck]['all']:
                            cached = window_cache[ck]
                            bucket[slug]['all'].extend(cached['all'])
                            bucket[slug]['summer'].extend(cached['summer'])
                            bucket[slug]['season'].extend(cached['season'])
                            cached_hits += 1
                            window_start += stride
                            continue

                        window_actuals = []
                        for i in range(horizon_days):
                            window_actuals.extend(by_day.get(window_start + timedelta(days=i), []))

                        if not window_actuals:
                            window_cache[ck] = {'all': [], 'summer': [], 'season': []}
                            window_start += stride
                            continue

                        try:
                            since_dt = datetime(window_start.year, window_start.month,
                                                window_start.day, tzinfo=timezone.utc)
                            pred = service.predict(cam, horizon_days,
                                                   since=since_dt,
                                                   model_set=set_name)
                        except Exception as e:
                            key = f"{type(e).__name__}: {e}"
                            error_types[key] = error_types.get(key, 0) + 1
                            error_count += 1
                            window_start += stride
                            continue

                        pred_by_ts = {
                            p['timestamp'][:16]: p['crowd_count'] / denom
                            for p in pred.get('predictions', [])
                            if p.get('available') and p.get('crowd_count') is not None
                        }

                        m_val = window_start.month
                        in_summer = m_val in SUMMER_MONTHS
                        in_season = m_val in SEASON_MONTHS
                        window_errors = {'all': [], 'summer': [], 'season': []}

                        for a in window_actuals:
                            key = a['timestamp'][:16]
                            if key not in pred_by_ts:
                                continue
                            err = abs(pred_by_ts[key] - a['crowd_count'] / denom)
                            window_errors['all'].append(err)
                            if in_summer:
                                window_errors['summer'].append(err)
                            if in_season:
                                window_errors['season'].append(err)

                        window_cache[ck] = window_errors
                        bucket[slug]['all'].extend(window_errors['all'])
                        bucket[slug]['summer'].extend(window_errors['summer'])
                        bucket[slug]['season'].extend(window_errors['season'])
                        computed += 1
                        new_cached += 1
                        window_start += stride

                all_errors[set_name][model_key] = bucket
                msg = f'(computed={computed}, cached={cached_hits})'
                if error_count:
                    msg += f' errors={error_count}'
                self.stdout.write(msg)
                for err_msg, count in sorted(error_types.items(), key=lambda x: -x[1]):
                    self.stderr.write(f'    [{count:>5}x] {err_msg}')

        _save_cache(cache_path, window_cache)
        self.stdout.write(f'\nCache saved ({len(window_cache)} total windows, {new_cached} new)')

        # ── Build results ──────────────────────────────────────────────────────
        def agg(bucket, slugs=None):
            merged = {'all': [], 'summer': [], 'season': []}
            for slug, errs in bucket.items():
                if slugs and slug not in slugs:
                    continue
                for p in ('all', 'summer', 'season'):
                    merged[p].extend(errs[p])
            return {p: round(relmae(merged[p]), 4) if relmae(merged[p]) is not None else None
                    for p in ('all', 'summer', 'season')}

        results = []
        for set_name, mk_dict in all_errors.items():
            for model_key, bucket in mk_dict.items():
                overall  = agg(bucket)
                per_beach = []
                for cam in cams:
                    slug = cam.camera_slug
                    if slug not in bucket or not bucket[slug]['all']:
                        continue
                    metrics = agg({slug: bucket[slug]})
                    per_beach.append({
                        'camera': slug,
                        'beach':  cam.beach.beach_name,
                        'island': getattr(cam.beach, 'island', ''),
                        'p90':    round(cam_p90.get(slug, 0), 1),
                        'relMAE_all':    metrics['all'],
                        'relMAE_summer': metrics['summer'],
                        'relMAE_season': metrics['season'],
                    })
                results.append({
                    'model_set': set_name,
                    'model_key': model_key,
                    'overall':   overall,
                    'per_beach': per_beach,
                })

        if not results:
            self.stderr.write('No results generated.')
            return

        # ── Pick best per horizon ──────────────────────────────────────────────
        best_by_horizon = {}
        for horizon_key in ['3d', '10d', '15d']:
            candidates = [
                r for r in results
                if r['model_key'] == horizon_key and r['overall']['season'] is not None
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda r: r['overall']['all'])
            best_by_horizon[horizon_key] = {
                'model_set':     best['model_set'],
                'relMAE_all':    best['overall']['all'],
                'relMAE_summer': best['overall']['summer'],
                'relMAE_season': best['overall']['season'],
            }
            self.stdout.write(
                f'\nBest {horizon_key}: {best["model_set"]}'
                f' (relMAE_all={best["overall"]["all"]:.4f})'
            )

        # ── Write JSON ─────────────────────────────────────────────────────────
        ranking = sorted(
            [{'model_set': r['model_set'], 'model_key': r['model_key'],
              'relMAE_all': r['overall']['all'],
              'relMAE_summer': r['overall']['summer'],
              'relMAE_season': r['overall']['season']}
             for r in results if r['overall']['all'] is not None],
            key=lambda x: x['relMAE_all']
        )
        best_all = ranking[0]['relMAE_all'] if ranking else None
        for row in ranking:
            row['pct_vs_best'] = round((row['relMAE_all'] - best_all) / best_all * 100, 2) if best_all else None

        output = {
            'evaluated_at':    datetime.utcnow().isoformat() + 'Z',
            'date_range':      {'since': str(global_start), 'until': str(global_end)},
            'best_by_horizon': best_by_horizon,
            'ranking':         ranking,
            'results':         results,
        }
        json_str = json.dumps(output, indent=2)

        eval_json_path.parent.mkdir(parents=True, exist_ok=True)
        eval_json_path.write_text(json_str)
        self.stdout.write(f'\nSaved JSON -> {eval_json_path}')

        out_base = options['out']
        json_path = Path(out_base + '.json')
        csv_path  = Path(out_base + '.csv')
        if eval_json_path != json_path:
            json_path.write_text(json_str)
            self.stdout.write(f'Saved JSON -> {json_path}')

        # ── Write CSV ──────────────────────────────────────────────────────────
        fields = ['model_set', 'model_key', 'scope', 'camera', 'beach', 'island', 'p90',
                  'relMAE_all', 'relMAE_summer', 'relMAE_season', 'is_best']
        best_keys = {(v['model_set'], k) for k, v in best_by_horizon.items()}
        csv_rows = []
        for r in results:
            is_best = (r['model_set'], r['model_key']) in best_keys
            csv_rows.append({
                'model_set': r['model_set'], 'model_key': r['model_key'],
                'scope': 'OVERALL', 'camera': 'ALL', 'beach': '', 'island': '', 'p90': '',
                'relMAE_all':    r['overall']['all'],
                'relMAE_summer': r['overall']['summer'],
                'relMAE_season': r['overall']['season'],
                'is_best': is_best,
            })
            for b in r['per_beach']:
                csv_rows.append({
                    'model_set': r['model_set'], 'model_key': r['model_key'],
                    'scope': 'per_beach',
                    'camera':  b['camera'], 'beach': b['beach'],
                    'island':  b['island'], 'p90': b['p90'],
                    'relMAE_all':    b['relMAE_all'],
                    'relMAE_summer': b['relMAE_summer'],
                    'relMAE_season': b['relMAE_season'],
                    'is_best': is_best,
                })

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_rows)
        self.stdout.write(f'Saved CSV: {csv_path} ({len(csv_rows)} rows)')

        # ── Console summary ────────────────────────────────────────────────────
        self.stdout.write('\n' + '-' * 72)
        self.stdout.write(f'{"SET":<20} {"KEY":<6} {"SCOPE":<14} {"ALL":>8} {"SUMMER":>8} {"SEASON":>8} BEST')
        self.stdout.write('-' * 72)
        for r in csv_rows:
            if r['scope'] == 'OVERALL':
                best_marker = ' <' if r['is_best'] else ''
                self.stdout.write(
                    f"{r['model_set']:<20} {r['model_key']:<6} {r['scope']:<14}"
                    f" {str(r['relMAE_all'] or ''):>8} {str(r['relMAE_summer'] or ''):>8}"
                    f" {str(r['relMAE_season'] or ''):>8}{best_marker}"
                )
        self.stdout.write('-' * 72)