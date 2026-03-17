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
from apps.webcam.models import WebCam, Beach
from apps.prediction.models import Snapshot


def serialize_beach(b):
    """Extract beach metadata relevant for TFT static exogenous features."""
    return {
        "beach_name": b.beach_name,
        "beach_slug": b.beach_slug,
        "grado_de_ocupacion": b.grado_de_ocupacion,
        "proximidad_al_nucleo_urbano": b.proximidad_al_nucleo_urbano,
        "composicion_de_la_playa": b.composicion_de_la_playa,
        "condiciones_de_bano": b.condiciones_de_bano,
        "paseo_maritimo": b.paseo_maritimo,
        "tipo_de_usuario_local": b.tipo_de_usuario_local,
        "tipo_de_usuario_turista": b.tipo_de_usuario_turista,
    }


class Command(BaseCommand):
    help = "Export all webcam and snapshot data to JSON for dataset building"

    def add_arguments(self, parser):
        parser.add_argument("--output", type=str, default="django_export.json")
        parser.add_argument("--since", type=str, default=None, help="ISO date filter, e.g. 2024-01-01")

    def handle(self, *args, **options):
        output_path = options["output"]
        since = options.get("since")

        webcams = WebCam.objects.select_related("beach").all()
        self.stdout.write(f"Found {webcams.count()} webcams")

        export = {"beaches": {}, "webcams": {}, "snapshots": []}

        for wc in webcams:
            b = wc.beach
            if b.beach_slug not in export["beaches"]:
                export["beaches"][b.beach_slug] = serialize_beach(b)

            export["webcams"][wc.id] = {
                "id": wc.id,
                "beach_slug": b.beach_slug,
                "beach_name": b.beach_name,
                "camera_slug": wc.camera_slug,
                "lat": float(wc.camera_latitude) if wc.camera_latitude else None,
                "lon": float(wc.camera_longitude) if wc.camera_longitude else None,
                "max_crowd_count": wc.max_crowd_count,
            }

        snapshots = (
            Snapshot.objects
            .select_related("webcam", "webcam__beach")
            .exclude(predicted_crowd_count__isnull=True, filter_blurry_image=False, filter_frozen_image=False, filter_moving_camera=False)
        )
        if since:
            snapshots = snapshots.filter(ts__gte=datetime.fromisoformat(since))
        snapshots = snapshots.order_by("ts")

        self.stdout.write(f"Exporting {snapshots.count()} snapshots...")

        for snap in snapshots.iterator():
            image_path = snap.webcam_image.name if snap.webcam_image else None
            prediction_path = snap.predicted_image.name if snap.predicted_image else None
            export["snapshots"].append({
                "webcam_id": snap.webcam_id,
                "image_path": image_path,
                "prediction_path": prediction_path,
                "beach_name": snap.webcam.beach.beach_name,
                "beach_slug": snap.webcam.beach.beach_slug,
                "camera_slug": snap.webcam.camera_slug,
                "lat": float(snap.webcam.camera_latitude) if snap.webcam.camera_latitude else None,
                "lon": float(snap.webcam.camera_longitude) if snap.webcam.camera_longitude else None,
                "ts": snap.ts.isoformat(),
                "crowd_count": snap.predicted_crowd_count,
            })

        with open(output_path, "w") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)

        self.stdout.write(self.style.SUCCESS(
            f"Exported {len(export['snapshots'])} snapshots, "
            f"{len(export['webcams'])} webcams, "
            f"{len(export['beaches'])} beaches → {output_path}"
        ))
