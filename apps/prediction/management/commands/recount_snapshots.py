"""
Rerun crowd counting on Django snapshots and update predicted_crowd_count.

Usage:
    python manage.py recount_snapshots
    python manage.py recount_snapshots --webcam cala-vedella
    python manage.py recount_snapshots --webcam cala-vedella --since 2025-06-01
    python manage.py recount_snapshots --only-null        # skip already predicted
    python manage.py recount_snapshots --force-cpu
    python manage.py recount_snapshots --dry-run
"""
import sys
from pathlib import Path
from datetime import datetime

import django
from django.core.management.base import BaseCommand
from django.utils import timezone

from apps.prediction.scripts import crowd_module


class Command(BaseCommand):
    help = 'Rerun crowd counting on Snapshot images and update predicted_crowd_count'

    def add_arguments(self, parser):
        parser.add_argument('--webcam', type=str, help='Filter by camera_slug')
        parser.add_argument('--since', type=str, help='Only snapshots after YYYY-MM-DD')
        parser.add_argument('--until', type=str, help='Only snapshots before YYYY-MM-DD')
        parser.add_argument('--only-null', action='store_true', help='Skip snapshots already with a prediction')
        parser.add_argument('--force-cpu', action='store_true', help='Force CPU inference (avoids MPS memory issues)')
        parser.add_argument('--batch-size', type=int, default=None, help='Batch size for inference')
        parser.add_argument('--dry-run', action='store_true', help='Show what would be processed without updating DB')

    def handle(self, *args, **options):
        from apps.prediction.models import Snapshot

        # Load crowd module with optional CPU flag
        if options['force_cpu']:
            import os
            os.environ['FORCE_CPU_BATCH'] = '1'

        scripts_dir = Path(__file__).resolve().parents[3] / 'scripts'

        # Build queryset
        qs = Snapshot.objects.select_related('webcam', 'webcam__beach').filter(
            webcam_image__isnull=False
        ).exclude(webcam_image='')

        if options['webcam']:
            qs = qs.filter(webcam__camera_slug=options['webcam'])

        if options['since']:
            since_dt = timezone.make_aware(datetime.strptime(options['since'], '%Y-%m-%d'))
            qs = qs.filter(ts__gte=since_dt)

        if options['until']:
            until_dt = timezone.make_aware(datetime.strptime(options['until'], '%Y-%m-%d'))
            qs = qs.filter(ts__lt=until_dt)

        if options['only_null']:
            qs = qs.filter(predicted_crowd_count__isnull=True)

        total = qs.count()
        if total == 0:
            self.stdout.write('No snapshots matched the filters.')
            return

        self.stdout.write(f'Found {total} snapshots to process')

        if options['dry_run']:
            self.stdout.write(self.style.WARNING('Dry run — no DB updates will be made'))
            for s in qs[:5]:
                self.stdout.write(f'  {s.webcam.camera_slug} | {s.ts} | {s.webcam_image}')
            if total > 5:
                self.stdout.write(f'  ... and {total - 5} more')
            return

        # Collect image paths
        from django.conf import settings
        media_root = Path(settings.MEDIA_ROOT)

        snapshots = list(qs.order_by('ts'))
        image_paths = []
        valid_snapshots = []

        for s in snapshots:
            img_path = media_root / str(s.webcam_image)
            if img_path.exists():
                image_paths.append(str(img_path))
                valid_snapshots.append(s)
            else:
                self.stdout.write(self.style.WARNING(f'  Image not found: {img_path}'))

        if not image_paths:
            self.stdout.write(self.style.ERROR('No valid image files found on disk.'))
            return

        self.stdout.write(f'Running inference on {len(image_paths)} images...')
        crowd_module.load_model()
        self.stdout.write(str(crowd_module.get_model_info()))

        counts = crowd_module.predict_count_batch(
            image_paths,
            batch_size=options['batch_size']
        )

        # Update DB
        updated = 0
        failed = 0
        for s, count in zip(valid_snapshots, counts):
            if count is None:
                failed += 1
                continue
            s.predicted_crowd_count = round(count, 4)
            updated += 1

        # Bulk update
        Snapshot.objects.bulk_update(
            [s for s, c in zip(valid_snapshots, counts) if c is not None],
            ['predicted_crowd_count'],
            batch_size=500
        )

        self.stdout.write(self.style.SUCCESS(
            f'Done — updated {updated} snapshots, {failed} failed, {total - len(valid_snapshots)} images missing'
        ))
