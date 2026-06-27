"""
Cross-horizon evaluation: can a larger model replace a smaller one?

For each camera, runs 3d/10d/15d models over rolling windows and compares
predictions at the same day offsets across models.

Comparisons made:
  Days 1–3:   3d  vs  10d  vs  15d
  Days 4–10:  10d vs  15d
  Days 11–15: 15d only

Output: cross_horizon_eval.json + cross_horizon_eval.csv

Usage:
    python manage.py evaluate_cross_horizon
    python manage.py evaluate_cross_horizon --sets tft_a tft_b
    python manage.py evaluate_cross_horizon --since 2025-04-01 --until 2025-09-30
    python manage.py evaluate_cross_horizon --out results/my_eval
"""

import csv
import json
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone as dj_tz

from apps.prediction.models import Snapshot
from apps.prediction.scripts import weather_module
from apps.prediction.tft_service import TFTService
from apps.webcam.models import WebCam

import sys as _sys


SEASON_MONTHS = {4, 5, 6, 7, 8, 9}
SUMMER_MONTHS = {6, 7, 8}
HOUR_MIN, HOUR_MAX = 8, 20

# Which day offsets each model is evaluated on (1-indexed from window start)
MODEL_DAY_RANGES = {
    '3d':  (1, 3),
    '10d': (1, 10),
    '15d': (1, 15),
}

# Cross-horizon comparison groups: {label: [model_keys]}
COMPARISON_GROUPS = {
    'days_1_3':   ['3d', '10d', '15d'],
    'days_4_10':  ['10d', '15d'],
    'days_11_15': ['15d'],
}
COMPARISON_DAY_RANGES = {
    'days_1_3':   (1, 3),
    'days_4_10':  (4, 10),
    'days_11_15': (11, 15),
}


def _cam_lat(cam):
    if cam.camera_latitude:
        return float(cam.camera_latitude)
    if cam.beach.coordenadas_geograficas:
        return float(cam.beach.coordenadas_geograficas.split(',')[0])
    return 39.5


def _cam_lon(cam):
    if cam.camera_longitude:
        return float(cam.camera_longitude)
    if cam.beach.coordenadas_geograficas:
        return float(cam.beach.coordenadas_geograficas.split(',')[1])
    return 2.6


def _relmae(errors):
    return float(np.mean(errors)) if errors else None


def _load_actuals(cam, since, until):
    rows = {}
    qs = (
        Snapshot.objects
        .filter(webcam=cam, predicted_crowd_count__isnull=False,
                ts__gte=datetime(since.year, since.month, since.day, tzinfo=timezone.utc),
                ts__lt=datetime(until.year, until.month, until.day, tzinfo=timezone.utc))
        .order_by('ts')
    )
    for s in qs:
        ts = s.ts.astimezone(dj_tz.get_current_timezone()).replace(tzinfo=None)
        ts = ts.replace(minute=0, second=0, microsecond=0)
        if HOUR_MIN <= ts.hour <= HOUR_MAX:
            rows[ts.strftime('%Y-%m-%dT%H:00')] = float(s.predicted_crowd_count)
    return rows


class Command(BaseCommand):
    help = 'Cross-horizon evaluation: compare 3d/10d/15d predictions at the same day windows'

    def add_arguments(self, parser):
        parser.add_argument('--since', default=None)
        parser.add_argument('--until', default=None)
        parser.add_argument('--sets', nargs='*', default=None,
                            help='Model sets to evaluate (default: all discovered)')
        parser.add_argument('--out', default='cross_horizon_eval',
                            help='Output file base path (no extension)')
        parser.add_argument('--min-windows', type=int, default=3,
                            help='Min evaluation windows required per camera (default: 3)')

    def handle(self, *args, **options):
        service = TFTService()

        configured_sets = getattr(settings, 'TFT_MODEL_SETS', {})
        all_sets = list(configured_sets.keys()) + [
            n for n in service._discovered if n not in configured_sets
        ]
        set_names = options['sets'] or all_sets
        if not set_names:
            self.stderr.write('No model sets found.')
            return

        # Load + filter cameras
        cams = list(WebCam.objects.select_related('beach').filter(max_crowd_count__gt=0))
        self.stdout.write(f'{len(cams)} cameras')

        # Date range
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

        self.stdout.write(f'Range: {global_start} → {global_end}')

        # Pre-fetch weather
        self.stdout.write('Pre-fetching weather...')
        seen = set()
        for cam in cams:
            key = (round(_cam_lat(cam), 2), round(_cam_lon(cam), 2))
            if key not in seen:
                seen.add(key)
                weather_module.prefetch(key[0], key[1], global_start, global_end)

        # Load model sets
        loaded = []
        for set_name in set_names:
            try:
                service._ensure_loaded(set_name)
                if service.model_sets.get(set_name):
                    loaded.append(set_name)
                    self.stdout.write(f'  Loaded: {set_name}')
            except Exception as e:
                self.stderr.write(f'  Could not load {set_name}: {e}')

        if not loaded:
            self.stderr.write('No model sets loaded.')
            return

        # Load actuals
        self.stdout.write('Loading actuals...')
        cam_actuals = {}
        for cam in cams:
            rows = _load_actuals(cam, global_start, global_end + timedelta(days=1))
            if rows:
                cam_actuals[cam.camera_slug] = rows

        self.stdout.write(f'{len(cam_actuals)} cameras have actuals\n')

        # ── Evaluate ────────────────────────────────────────────────────────
        # Structure: results[set_name][model_key][cam_slug] = [relMAE per window]
        results = {s: {mk: defaultdict(list) for mk in MODEL_DAY_RANGES} for s in loaded}

        for set_name in loaded:
            self.stdout.write(f'\n── {set_name}')
            ms = service.model_sets[set_name]

            for model_key, (day_from, day_to) in MODEL_DAY_RANGES.items():
                if model_key not in ms:
                    self.stdout.write(f'  [{model_key}] not in set — skipping')
                    continue

                horizon_days = day_to
                stride = timedelta(days=horizon_days)
                total_windows = 0
                total_errors = 0
                self.stdout.write(f'  [{model_key}] horizon={horizon_days}d', ending=' ')
                self.stdout.flush()

                for cam in cams:
                    slug = cam.camera_slug
                    if slug not in cam_actuals:
                        continue
                    actuals = cam_actuals[slug]
                    max_crowd = cam.max_crowd_count
                    window_start = global_start

                    while window_start + timedelta(days=horizon_days) <= global_end:
                        try:
                            since_dt = datetime(window_start.year, window_start.month,
                                                window_start.day, tzinfo=timezone.utc)
                            pred = service.predict(cam, horizon_days,
                                                   since=since_dt, model_set=set_name)
                        except Exception:
                            window_start += stride
                            continue

                        # OPERATIONAL lens: crowd_count / max_crowd_count (deployment
                        # capacity), NOT the statistical per-series-P90 headline denominator.
                        pred_map = {
                            p['timestamp'][:16]: p['crowd_count'] / max_crowd
                            for p in pred.get('predictions', [])
                            if p.get('available') and p.get('crowd_count') is not None
                        }

                        # Compute relMAE per sub-window (days 1-3, 4-10, 11-15)
                        for group, (gfrom, gto) in COMPARISON_DAY_RANGES.items():
                            if gfrom > horizon_days:
                                continue
                            actual_gto = min(gto, horizon_days)
                            errs = []
                            for offset in range(gfrom - 1, actual_gto):
                                day = window_start + timedelta(days=offset)
                                for hour in range(HOUR_MIN, HOUR_MAX + 1):
                                    key = f"{day.strftime('%Y-%m-%d')}T{str(hour).zfill(2)}:00"
                                    if key in pred_map and key in actuals:
                                        err = abs(pred_map[key] - actuals[key] / max_crowd)
                                        errs.append(err)

                            if errs:
                                results[set_name][model_key][f'{slug}_{group}'].extend(errs)

                        total_windows += 1
                        total_errors += 1 if not pred_map else 0
                        window_start += stride

                self.stdout.write(f'(windows={total_windows})')

        # ── Build comparison summary ─────────────────────────────────────────
        # For each group, compare all models that cover those days
        summary = {}
        for set_name in loaded:
            summary[set_name] = {}
            for group, (gfrom, gto) in COMPARISON_DAY_RANGES.items():
                group_data = {}
                for model_key in MODEL_DAY_RANGES:
                    day_from, day_to = MODEL_DAY_RANGES[model_key]
                    if day_to < gfrom:
                        continue  # model doesn't reach this window
                    # Aggregate errors across all cameras for this group
                    all_errs = []
                    for slug in cam_actuals:
                        k = f'{slug}_{group}'
                        all_errs.extend(results[set_name][model_key].get(k, []))
                    if all_errs:
                        group_data[model_key] = {
                            'relMAE': round(_relmae(all_errs), 4),
                            'n_errors': len(all_errs),
                        }
                summary[set_name][group] = group_data

        # ── Print summary ───────────────────────────────────────────────────
        self.stdout.write('\n' + '=' * 70)
        self.stdout.write(f'{"GROUP":<14} {"MODEL":>6} {"relMAE":>8} {"N":>8}  SET')
        self.stdout.write('=' * 70)
        for set_name in loaded:
            for group in COMPARISON_GROUPS:
                gdata = summary[set_name].get(group, {})
                for model_key in COMPARISON_GROUPS[group]:
                    d = gdata.get(model_key)
                    if d:
                        mae_str = f'{d["relMAE"]:.4f}'
                        self.stdout.write(
                            f'{group:<14} {model_key:>6} {mae_str:>8} {d["n_errors"]:>8}  {set_name}'
                        )
        self.stdout.write('=' * 70)

        # Highlight if larger model beats smaller in short windows
        self.stdout.write('\nCan a larger model replace a smaller one?')
        for set_name in loaded:
            self.stdout.write(f'\n  {set_name}:')
            s = summary[set_name]
            checks = [
                ('days_1_3',  '3d',  '10d'),
                ('days_1_3',  '3d',  '15d'),
                ('days_1_3',  '10d', '15d'),
                ('days_4_10', '10d', '15d'),
            ]
            for group, smaller, larger in checks:
                sm = s.get(group, {}).get(smaller, {}).get('relMAE')
                lg = s.get(group, {}).get(larger, {}).get('relMAE')
                if sm is None or lg is None:
                    continue
                diff_pct = (lg - sm) / sm * 100
                verdict = '✓ YES' if diff_pct <= 2.0 else ('~' if diff_pct <= 5.0 else '✗ NO')
                self.stdout.write(
                    f'    {group} | {larger} vs {smaller}: '
                    f'Δ={diff_pct:+.1f}%  {verdict}'
                )

        # ── Write JSON ──────────────────────────────────────────────────────
        output = {
            'evaluated_at': datetime.utcnow().isoformat() + 'Z',
            'date_range': {'since': str(global_start), 'until': str(global_end)},
            'model_sets': loaded,
            'comparison_groups': {g: {'day_range': list(r), 'models': ms}
                                   for g, (r, ms) in zip(
                                       COMPARISON_GROUPS,
                                       [(COMPARISON_DAY_RANGES[g], COMPARISON_GROUPS[g])
                                        for g in COMPARISON_GROUPS]
                                   )},
            'summary': {
                set_name: {
                    group: {
                        model_key: data
                        for model_key, data in gdata.items()
                    }
                    for group, gdata in sdata.items()
                }
                for set_name, sdata in summary.items()
            },
        }

        out_base = options['out']
        json_path = Path(out_base + '.json')
        csv_path = Path(out_base + '.csv')
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(output, indent=2))
        self.stdout.write(f'\nSaved JSON → {json_path}')

        # ── Write CSV ───────────────────────────────────────────────────────
        fields = ['model_set', 'group', 'day_range', 'model_key', 'relMAE', 'n_errors',
                  'vs_smaller_model', 'delta_pct', 'can_replace']
        csv_rows = []
        for set_name in loaded:
            for group, models in COMPARISON_GROUPS.items():
                gfrom, gto = COMPARISON_DAY_RANGES[group]
                gdata = summary[set_name].get(group, {})
                # baseline = smallest model in the group
                baseline_key = models[0]
                baseline_mae = gdata.get(baseline_key, {}).get('relMAE')
                for model_key in models:
                    d = gdata.get(model_key)
                    if not d:
                        continue
                    delta = None
                    can_replace = None
                    if model_key != baseline_key and baseline_mae:
                        delta = round((d['relMAE'] - baseline_mae) / baseline_mae * 100, 2)
                        can_replace = delta <= 2.0
                    csv_rows.append({
                        'model_set': set_name,
                        'group': group,
                        'day_range': f'{gfrom}-{gto}',
                        'model_key': model_key,
                        'relMAE': d['relMAE'],
                        'n_errors': d['n_errors'],
                        'vs_smaller_model': baseline_key if model_key != baseline_key else '',
                        'delta_pct': delta if delta is not None else '',
                        'can_replace': can_replace if can_replace is not None else '',
                    })

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_rows)
        self.stdout.write(f'Saved CSV  → {csv_path} ({len(csv_rows)} rows)')
