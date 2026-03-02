from django.urls import path
from apps.core import views

urlpatterns = [
    path('', views.home, name='home'),
    path('platja/<str:camera_slug>', views.webcam, name='beach'),
    path('platja/<str:camera_slug>/forecast/', views.tft_forecast_json, name='beach-forecast'),
    path('platja/<str:camera_slug>/actuals/', views.tft_actuals_json, name='beach-actuals'),
    path('platja/<str:camera_slug>/metrics/', views.tft_metrics_json, name='beach-metrics'),
    path('analitza/', views.analyze_image, name='analyze-image'),
]
