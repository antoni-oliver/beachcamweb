import logging
from django.apps import AppConfig

logger = logging.getLogger(__name__)


class PredictionConfig(AppConfig):
    name = 'apps.prediction'
    default_auto_field = 'django.db.models.BigAutoField'