"""
Update WebCam.max_crowd_count based on P90 of historical predicted_crowd_count.

Usage:
    python manage.py update_max_crowd_count
    python manage.py update_max_crowd_count --percentile 95
    python manage.py update_max_crowd_count --dry-run
"""

from django.core.management.base import BaseCommand
from django.db.models import Count

import numpy as np

from apps.prediction.models import Snapshot
from apps.webcam.models import WebCam


class Command(BaseCommand):
    help = 'Update max_crowd_count on each WebCam using P90 of predicted_crowd_count'

    def add_arguments(self, parser):
        parser.add_argument('--percentile', type=float, default=90)
        parser.add_argument('--min-snapshots', type=int, default=50)
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args, **options):
        pct = options['percentile']
        min_snaps = options['min_snapshots']
        dry_run = options['dry_run']

        webcams = (
            WebCam.objects
            .annotate(snap_count=Count('snapshots'))
            .filter(snap_count__gte=min_snaps)
        )

        updated = 0
        for cam in webcams:
            values = list(
                Snapshot.objects
                .filter(webcam=cam, predicted_crowd_count__isnull=False, predicted_crowd_count__gt=0)
                .values_list('predicted_crowd_count', flat=True)
            )

            if len(values) < min_snaps:
                continue

            p_value = int(np.percentile(values, pct))
            old = cam.max_crowd_count

            self.stdout.write(
                f"{cam.camera_slug}: P{int(pct)}={p_value} "
                f"(old={old}, n={len(values)})"
            )

            if not dry_run:
                cam.max_crowd_count = p_value
                cam.save(update_fields=['max_crowd_count'])
                updated += 1

        prefix = "[DRY RUN] " if dry_run else ""
        self.stdout.write(self.style.SUCCESS(f"{prefix}Updated {updated} webcams"))
