from django.urls import path

from apps.core import views

urlpatterns = [
    path('', views.home, name='home'),
    path('platja/<str:camera_slug>', views.webcam, name='beach'),
    path('analitza/', views.analyze_image, name='analyze-image'),
    path('gestio/webcam/<int:webcam_id>/filtres/', views.admin_webcam_filters, name='admin-webcam-filters'),
    path('gestio/webcam/<int:webcam_id>/snapshots/filtres/', views.admin_webcam_snapshots_filters, name='admin-webcam-snapshots-filters'),
]