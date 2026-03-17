from pathlib import Path
from django.conf import settings

PREDICTION_DIR = Path(settings.PREDICTION_DIR)
SCRIPTS_DIR    = PREDICTION_DIR / 'scripts'

PIPELINE_DIR       = PREDICTION_DIR / 'pipeline_workspace'
EXPORTS_DIR        = PIPELINE_DIR / 'exports'
DATASET_CSV        = PIPELINE_DIR / 'dataset.csv'
CLEAN_DATASETS_DIR = PIPELINE_DIR / 'clean_datasets'
LOGS_DIR           = PIPELINE_DIR / 'logs'
NOTEBOOKS_DIR      = PREDICTION_DIR / 'notebooks'

TFT_MODELS_DIR     = Path(settings.TFT_MODELS_DIR)
WEATHER_CACHE_DIR  = Path(settings.WEATHER_CACHE_DIR)
CROWD_CACHE_DIR    = PREDICTION_DIR / 'cache' / 'predictions'

BEACHES_JSON         = SCRIPTS_DIR / 'beaches.json'
MEDIA_2022_DIR       = Path(settings.PIPELINE_MEDIA_2022_DIR)
CUDA_DEVICE          = settings.PIPELINE_CUDA_DEVICE

NOTEBOOK_INPUT       = NOTEBOOKS_DIR / 'generate_train_data.ipynb'
NOTEBOOK_OUTPUT      = PIPELINE_DIR / 'generate_train_data_output.ipynb'
NOTEBOOK_HTML        = PIPELINE_DIR / 'generate_train_data_output.html'

TRAIN_SCRIPT         = SCRIPTS_DIR / 'tft_train_3_models.py'
BUILD_DATASET_SCRIPT = SCRIPTS_DIR / 'build_dataset_v4.py'

DJANGO_EXPORT_FILE   = EXPORTS_DIR / 'django_export.json'
BEACH_PROFILES_FILE  = EXPORTS_DIR / 'beach_profiles.json'


def ensure_dirs():
    for d in [PIPELINE_DIR, EXPORTS_DIR, CLEAN_DATASETS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)