"""
Management command: python manage.py load_tft_models [--dir path/to/models]
"""
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Load TFT models into memory'

    def add_arguments(self, parser):
        parser.add_argument('--dir', type=str, default=None, help='Models directory')

    def handle(self, *args, **options):
        from apps.prediction.tft_service import tft_service

        self.stdout.write("Loading TFT models...")
        tft_service.load_models(base_dir=options['dir'])

        for key, m in tft_service.models.items():
            self.stdout.write(self.style.SUCCESS(
                f"  {key}: H={m['horizon']} ({m['max_days']}d), "
                f"input_size={m['input_size']}, "
                f"beaches={len(m['static'])}"
            ))

        self.stdout.write(self.style.SUCCESS(f"Done: {len(tft_service.models)} models loaded"))
