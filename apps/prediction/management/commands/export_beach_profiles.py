"""
Export beach metadata profiles with coordinates.
Usage: python manage.py export_beach_profiles --output beach_profiles.json
"""
import json
from django.core.management.base import BaseCommand
from apps.webcam.models import WebCam


class Command(BaseCommand):
    help = "Export beach metadata keyed by camera_slug"

    def add_arguments(self, parser):
        parser.add_argument("--output", type=str, default="beach_profiles.json")

    def handle(self, *args, **options):
        output_path = options["output"]
        profiles = {}

        for wc in WebCam.objects.select_related("beach").all():
            b = wc.beach
            profiles[wc.camera_slug] = {
                "beach_name": b.beach_name,
                "lat": float(wc.camera_latitude) if wc.camera_latitude else None,
                "lon": float(wc.camera_longitude) if wc.camera_longitude else None,
                "grado_de_ocupacion": b.grado_de_ocupacion,
                "proximidad_al_nucleo_urbano": b.proximidad_al_nucleo_urbano,
                "composicion_de_la_playa": b.composicion_de_la_playa,
                "condiciones_de_bano": b.condiciones_de_bano,
                "paseo_maritimo": b.paseo_maritimo,
                "tipo_de_usuario_local": b.tipo_de_usuario_local,
                "tipo_de_usuario_turista": b.tipo_de_usuario_turista,
            }

        with open(output_path, "w") as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)

        self.stdout.write(self.style.SUCCESS(f"Exported {len(profiles)} profiles → {output_path}"))
