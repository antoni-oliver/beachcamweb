"""
ASGI config for beachcamweb project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""

import os

# XGBoost and PyTorch each ship their own OpenMP runtime; loading both in one
# process segfaults on macOS unless OpenMP is pinned to a single thread. The web
# server serves both families together (forecast comparison), so pin it before
# any numpy/torch/xgboost import.
os.environ.setdefault("OMP_NUM_THREADS", "1")

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

application = get_asgi_application()
