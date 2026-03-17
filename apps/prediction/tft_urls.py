from django.urls import path
from apps.prediction.tft_views import (
    tft_forecast_json, tft_actuals_json, tft_metrics_json,
    tft_model_sets_json, tft_snapshot_image_json,
    tft_full_forecast_json, tft_eval_ranking_json,
)

urlpatterns = [
    path('platja/<str:camera_slug>/forecast/', tft_forecast_json, name='tft-forecast'),
    path('platja/<str:camera_slug>/actuals/', tft_actuals_json, name='tft-actuals'),
    path('platja/<str:camera_slug>/metrics/', tft_metrics_json, name='tft-metrics'),
    path('platja/<str:camera_slug>/model-sets/', tft_model_sets_json, name='tft-model-sets'),
    path('platja/<str:camera_slug>/snapshot-image/', tft_snapshot_image_json, name='tft-snapshot-image'),
    path('platja/<str:camera_slug>/eval-ranking/', tft_eval_ranking_json, name='tft-eval-ranking'),
    path('beach/full-forecast/', tft_full_forecast_json),
    path('beach/<int:beach_id>/full-forecast/', tft_full_forecast_json),
]