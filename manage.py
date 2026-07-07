#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import warnings
import logging

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message=".*litlogger.*")
warnings.filterwarnings("ignore", message=".*ModelSummary.*")

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

# XGBoost and PyTorch each ship their own OpenMP runtime; loading both in one
# process segfaults on macOS unless OpenMP is pinned to a single thread. The web
# server serves both families (the forecast comparison loads TFT/LSTM and XGB
# together), so pin it for runserver only, before any numpy/torch/xgboost import.
# Management commands (training, evaluation) keep full threading.
if "runserver" in sys.argv:
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
