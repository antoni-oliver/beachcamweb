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
            from apps.prediction.tft_service import tft_service
            from django.conf import settings

            def _try_load(name, base_dir):
                # Load each set independently so one missing/broken set (e.g. an
                # XGB set with no nf_model dir) does not abort the whole warmup.
                if name in tft_service.model_sets:
                    return
                try:
                    if tft_service._set_kind.get(name) == 'xgb':
                        tft_service.load_xgb_models(base_dir=base_dir, set_name=name)
                    else:
                        tft_service.load_models(base_dir=base_dir, set_name=name)
                except Exception:
                    logger.exception("skipping model set '%s' (failed to load)", name)

            for name, base_dir in getattr(settings, 'TFT_MODEL_SETS', {}).items():
                _try_load(name, base_dir)
            for name, base_dir in list(tft_service._discovered.items()):
                _try_load(name, base_dir)
            logger.info("TFT warmup done: %d set(s) loaded.", len(tft_service.model_sets))
            try:
                tft_service.warm_forecast_cache()
            except Exception:
                logger.exception("forecast cache warmup failed.")

        threading.Thread(target=_warmup, name='tft-warmup', daemon=True).start()
