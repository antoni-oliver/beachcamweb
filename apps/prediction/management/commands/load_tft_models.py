"""
Management command: python manage.py load_tft_models [--dir path] [--name set_name] [--list]
"""
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Load TFT models into memory'

    def add_arguments(self, parser):
        parser.add_argument('--dir', type=str, default=None, help='Models directory')
        parser.add_argument('--name', type=str, default='default', help='Model set name (default: default)')
        parser.add_argument('--list', action='store_true', help='List all loaded model sets')

    def handle(self, *args, **options):
        from apps.prediction.tft_service import tft_service

        if options['list']:
            sets = tft_service.list_model_sets()
            if not sets:
                self.stdout.write("No model sets loaded.")
                return
            for set_name, models in sets.items():
                self.stdout.write(f"\n  [{set_name}]")
                for key, info in models.items():
                    self.stdout.write(f"    {key}: H={info['horizon']}, max_days={info['max_days']}, dir={info['dir']}")
            return

        set_name = options['name']
        self.stdout.write(f"Loading TFT models (set: '{set_name}')...")
        loaded = tft_service.load_models(base_dir=options['dir'], set_name=set_name)

        for key, m in loaded.items():
            self.stdout.write(self.style.SUCCESS(
                f"  {key}: H={m['horizon']} ({m['max_days']}d), "
                f"input_size={m['input_size']}, "
                f"beaches={len(m['static'])}, "
                f"dir={m['model_dir']}"
            ))

        self.stdout.write(self.style.SUCCESS(f"Done: {len(loaded)} models loaded as '{set_name}'"))