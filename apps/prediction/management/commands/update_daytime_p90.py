"""
Update WebCam.daytime_p90: the per-series 90th percentile of actual daytime
(8-20 local) season (Apr-Sep) crowd counts, including zeros.

This is the statistical denominator used by evaluate_model_sets and the thesis,
and what the hindcast comparison widget normalises by. It does NOT touch
max_crowd_count (the operational denominator used for occupancy classification).

Usage:
    python manage.py update_daytime_p90
    python manage.py update_daytime_p90 --dry-run
"""

from django.core.management.base import BaseCommand
from django.utils import timezone as dj_tz

import numpy as np

from apps.prediction.models import Snapshot
from apps.webcam.models import WebCam

HOUR_MIN, HOUR_MAX = 8, 20
SEASON_MONTHS = {4, 5, 6, 7, 8, 9}


class Command(BaseCommand):
    help = 'Update WebCam.daytime_p90 (season daytime P90 of actual counts) for the hindcast metric'

    def add_arguments(self, parser):
        parser.add_argument('--percentile', type=float, default=90)
        parser.add_argument('--min-snapshots', type=int, default=50)
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args, **options):
        pct = options['percentile']
        min_snaps = options['min_snapshots']
        dry_run = options['dry_run']

        updated = 0
        for cam in WebCam.objects.all():
            vals = []
            for ts, cc in (Snapshot.objects
                           .filter(webcam=cam, predicted_crowd_count__isnull=False)
                           .values_list('ts', 'predicted_crowd_count')):
                local = dj_tz.localtime(ts)
                if HOUR_MIN <= local.hour <= HOUR_MAX and local.month in SEASON_MONTHS:
                    vals.append(float(cc))

            if len(vals) < min_snaps:
                continue

            dp90 = round(float(np.percentile(vals, pct)), 2)
            self.stdout.write(f"{cam.camera_slug}: daytime_p90={dp90} (old={cam.daytime_p90}, n={len(vals)})")

            if not dry_run:
                cam.daytime_p90 = dp90
                cam.save(update_fields=['daytime_p90'])
                updated += 1

        prefix = "[DRY RUN] " if dry_run else ""
        self.stdout.write(self.style.SUCCESS(f"{prefix}Updated {updated} webcams"))
