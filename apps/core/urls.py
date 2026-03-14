from django.urls import path

from apps.core import core_views

urlpatterns = [
    path('', core_views.home, name='home'),
    path('platja/<str:camera_slug>', core_views.webcam, name='beach'),
    path('analitza/', core_views.analyze_image, name='analyze-image'),
    path('gestio/webcam/<int:webcam_id>/filtres/', core_views.admin_webcam_filters, name='admin-webcam-filters'),
    path('gestio/webcam/<int:webcam_id>/snapshots/filtres/', core_views.admin_webcam_snapshots_filters, name='admin-webcam-snapshots-filters'),
]