"""
Run Bayesian crowd counting on /media_2022 images using beaches.json.

Usage:
    python manage.py run_crowd_counting
    python manage.py run_crowd_counting --force-reprocess
    python manage.py run_crowd_counting --media-dir /path/to/media_2022
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime

from django.core.management.base import BaseCommand

from apps.prediction import pipeline_config as cfg
from apps.prediction.scripts import crowd_module


def _load_beaches(beaches_json_path):
    with open(beaches_json_path) as f:
        return json.load(f)


def _parse_datetime_from_filename(filename):
    try:
        ts = int(Path(filename).stem)
        return datetime.fromtimestamp(ts)
    except (ValueError, OSError):
        return None


def _process_camera(camera_key, beach_info, media_dir, predictor, force_reprocess):

    # camera_key is like "ClubNauticSoller/webcam-mestral" → nested path
    image_dir = media_dir / camera_key

    if not image_dir.exists():
        return 0, 0

    lat = beach_info['lat']
    lon = beach_info['lon']
    beach_name = beach_info['name']
    folder_name = camera_key.replace('/', '_')

    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    ])

    processed = 0
    skipped = 0

    for img_path in image_files:
        cached = crowd_module.load_prediction_from_cache(
            img_path.name, lat, lon,
            model=predictor.name, name=folder_name
        )
        if cached is not None and not force_reprocess:
            skipped += 1
            continue

        dt = _parse_datetime_from_filename(img_path.name)
        try:
            count = predictor.predict(img_path)
            result = {
                'beach': beach_name,
                'beach_folder': folder_name,
                'lat': lat,
                'lon': lon,
                'filename': img_path.name,
                'image_path': str(img_path),
                'datetime': dt.isoformat() if dt else None,
                'count': round(count, 4),
            }
            crowd_module.save_prediction_to_cache(
                img_path.name, lat, lon, result,
                model=predictor.name, name=folder_name
            )
            processed += 1
        except Exception as e:
            print(f"    [WARN] {img_path.name}: {e}")

    return processed, skipped


class Command(BaseCommand):
    help = 'Run crowd counting on media_2022 images using beaches.json config'

    def add_arguments(self, parser):
        parser.add_argument('--media-dir', type=str, default=None)
        parser.add_argument('--beaches-json', type=str, default=None)
        parser.add_argument('--force-reprocess', action='store_true')
        parser.add_argument('--force-cpu', action='store_true')

    def handle(self, *args, **options):
        cfg.ensure_dirs()

        media_dir = Path(options['media_dir']) if options['media_dir'] else cfg.MEDIA_2022_DIR
        beaches_json = Path(options['beaches_json']) if options['beaches_json'] else cfg.BEACHES_JSON
        force_reprocess = options['force_reprocess']

        if not media_dir.exists():
            self.stderr.write(f'media_2022 dir not found: {media_dir}')
            return

        if not beaches_json.exists():
            self.stderr.write(f'beaches.json not found: {beaches_json}')
            return

        if options['force_cpu']:
            os.environ['FORCE_CPU_BATCH'] = '1'
            print(f"Enable Force CPU")

        predictor = crowd_module._registry.get_active()
        predictor.load()

        beaches = _load_beaches(beaches_json)
        self.stdout.write(f'Processing {len(beaches)} cameras from {media_dir}')

        total_processed = 0
        total_skipped = 0

        for camera_key, beach_info in beaches.items():
            self.stdout.write(self.style.NOTICE(f'Processing camera: {camera_key} ({beach_info})'))
            processed, skipped = _process_camera(
                camera_key, beach_info, media_dir, predictor, force_reprocess
            )
            if processed or skipped:
                self.stdout.write(f'  {camera_key}: +{processed} new, {skipped} cached')
            total_processed += processed
            total_skipped += skipped

        self.stdout.write(self.style.SUCCESS(
            f'Done — {total_processed} new predictions, {total_skipped} cached'
        ))