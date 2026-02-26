"""
Django management command: export all WebCam + Snapshot data to JSON.

Usage:
    python manage.py export_snapshots
    python manage.py export_snapshots --output /path/to/output.json
    python manage.py export_snapshots --since 2024-01-01
"""
import json
from datetime import datetime

from django.core.management.base import BaseCommand
from apps.webcam.models import WebCam
from apps.prediction.models import Snapshot


class Command(BaseCommand):
    help = "Export all webcam and snapshot data to JSON for dataset building"

    def add_arguments(self, parser):
        parser.add_argument("--output", type=str, default="django_export.json")
        parser.add_argument("--since", type=str, default=None, help="ISO date filter, e.g. 2024-01-01")

    def handle(self, *args, **options):
        output_path = options["output"]
        since = options.get("since")

        webcams = WebCam.objects.all()
        self.stdout.write(f"Found {webcams.count()} webcams")

        export = {"webcams": {}, "snapshots": []}

        for wc in webcams:
            export["webcams"][wc.id] = {
                "id": wc.id,
                "beach_name": wc.beach_name,
                "slug": wc.slug,
                "lat": float(wc.beach_latitude) if wc.beach_latitude else None,
                "lon": float(wc.beach_longitude) if wc.beach_longitude else None,
                "max_crowd_count": wc.max_crowd_count,
            }

        snapshots = Snapshot.objects.select_related("webcam").exclude(predicted_crowd_count__isnull=True)
        if since:
            snapshots = snapshots.filter(ts__gte=datetime.fromisoformat(since))
        snapshots = snapshots.order_by("ts")

        self.stdout.write(f"Exporting {snapshots.count()} snapshots...")

        for snap in snapshots.iterator():
            # Get image name
            image_path = snap.webcam_image.name if snap.webcam_image else None
            prediction_path = snap.predicted_image.name if snap.predicted_image else None
            export["snapshots"].append({
                "webcam_id": snap.webcam_id,
                "image_path": image_path,
                "prediction_path": prediction_path,
                "beach_name": snap.webcam.beach_name,
                "slug": snap.webcam.slug,
                "lat": float(snap.webcam.beach_latitude) if snap.webcam.beach_latitude else None,
                "lon": float(snap.webcam.beach_longitude) if snap.webcam.beach_longitude else None,
                "ts": snap.ts.isoformat(),
                "crowd_count": snap.predicted_crowd_count,
            })

        with open(output_path, "w") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)

        self.stdout.write(self.style.SUCCESS(
            f"Exported {len(export['snapshots'])} snapshots from {len(export['webcams'])} webcams → {output_path}"
        ))