import logging
from django.apps import AppConfig

logger = logging.getLogger(__name__)


class PredictionConfig(AppConfig):
    name = 'apps.prediction'
    default_auto_field = 'django.db.models.BigAutoField'

    def ready(self):
        """Load TFT models when Django starts (runs once per process)."""
        import os
        if os.environ.get('RUN_MAIN') == 'true' or not os.environ.get('DJANGO_DEV'):
            try:
                from .tft_service import tft_service
                tft_service.load_models()
                logger.info("TFT models loaded at startup")
            except Exception as e:
                logger.warning(f"TFT models not loaded: {e}")