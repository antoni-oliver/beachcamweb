# apps/api/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("camera_info/", views.camera_info, name="camera_info"),
    path("estimacio_actual/", views.estimacio_actual, name="estimacio_actual"),
    path("prediccio_futura/", views.prediccio_futura, name="prediccio_futura"),
]
