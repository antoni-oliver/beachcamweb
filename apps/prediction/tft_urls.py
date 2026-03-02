from django.urls import path
from . import tft_views

tft_urlpatterns = [
    path('tft/status/', tft_views.tft_status, name='tft-status'),
    path('tft/beaches/', tft_views.tft_beaches, name='tft-beaches'),
    path('tft/<str:camera_slug>/', tft_views.tft_predict, name='tft-predict'),
]