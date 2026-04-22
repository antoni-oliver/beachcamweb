import os
import sys
import logging
import threading
from django.apps import AppConfig

logger = logging.getLogger(__name__)


class PredictionConfig(AppConfig):
    name = 'apps.prediction'
    default_auto_field = 'django.db.models.BigAutoField'

    def ready(self):
        argv = ' '.join(sys.argv)
        is_server = 'runserver' in argv or 'waitress' in argv
        is_autoreload_parent = 'runserver' in argv and os.environ.get('RUN_MAIN') != 'true'
        if not is_server or is_autoreload_parent:
            return

        def _warmup():
            try:
                from apps.prediction.tft_service import tft_service
                from django.conf import settings
                for name, base_dir in getattr(settings, 'TFT_MODEL_SETS', {}).items():
                    tft_service.load_models(base_dir=base_dir, set_name=name)
                for name, base_dir in tft_service._discovered.items():
                    if name not in tft_service.model_sets:
                        tft_service.load_models(base_dir=base_dir, set_name=name)
                logger.info("TFT models loaded.")
                tft_service.warm_forecast_cache()
            except Exception:
                logger.exception("TFT warmup failed.")

        threading.Thread(target=_warmup, name='tft-warmup', daemon=True).start()
