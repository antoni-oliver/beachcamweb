from django.urls import path
from apps.core.core_views import beach_overview, home, beach_history_api, webcam, analyze_image, all_beaches_forecast, fundaciobit_debug
from apps.prediction.tft_urls import urlpatterns as prediction_urls

urlpatterns = [
    path('', home, name='home'),
    *prediction_urls,
    path('platja/<str:camera_slug>/', webcam, name='beach'),
    path('platja/<slug:camera_slug>/history/', beach_history_api, name='beach-history'),
    path('analitza/', analyze_image, name='analyze-image'),
    path('overview/', beach_overview, name='beach-overview'),
    path('forecast/', all_beaches_forecast, name='all-beaches-forecast'),
    path('debug/fundaciobit/', fundaciobit_debug, name='fundaciobit-debug'),
]