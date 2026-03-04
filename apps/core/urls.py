from django.urls import path
from apps.core import views
from apps.prediction.tft_views import tft_forecast_json, tft_actuals_json, tft_metrics_json

urlpatterns = [
    path('', views.home, name='home'),
    path('platja/<str:camera_slug>', views.webcam, name='beach'),
    path('platja/<str:camera_slug>/forecast/', tft_forecast_json, name='tft-forecast'),
    path('platja/<str:camera_slug>/actuals/', tft_actuals_json, name='tft-actuals'),
    path('platja/<str:camera_slug>/metrics/', tft_metrics_json, name='tft-metrics'),
    path('analitza/', views.analyze_image, name='analyze-image'),
]
