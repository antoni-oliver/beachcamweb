# apps/api/urls.py
from django.urls import path
from . import api_views

urlpatterns = [
    path("camera_info/", api_views.camera_info, name="camera_info"),
    path("estimacio_actual/", api_views.estimacio_actual, name="estimacio_actual"),
    path("prediccio_futura/", api_views.prediccio_futura, name="prediccio_futura"),
]
