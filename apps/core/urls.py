from django.urls import path
from apps.core.core_views import beach_overview, home, beach_history_api, webcam, analyze_image, all_beaches_forecast, fundaciobit_debug, pipeline_report, forecast_model_report, \
    admin_webcam_filters, admin_webcam_snapshots_filters
from apps.prediction.tft_urls import urlpatterns as prediction_urls

urlpatterns = [
    path('', home, name='home'),
    *prediction_urls,
    path('platja/<str:camera_slug>/', webcam, name='beach'),
    path('platja/<slug:camera_slug>/history/', beach_history_api, name='beach-history'),
    path('analitza/', analyze_image, name='analyze-image'),
    path('gestio/webcam/<int:webcam_id>/filtres/', admin_webcam_filters, name='admin-webcam-filters'),
    path('gestio/webcam/<int:webcam_id>/snapshots/filtres/', admin_webcam_snapshots_filters, name='admin-webcam-snapshots-filters'),
    path('overview/', beach_overview, name='beach-overview'),
    path('forecast/', all_beaches_forecast, name='all-beaches-forecast'),
    path('debug/fundaciobit/', fundaciobit_debug, name='fundaciobit-debug'),
    path('pipeline/report/', pipeline_report, name='pipeline-report'),
    path('forecast/report/', forecast_model_report, name='forecast-model-report'),

]