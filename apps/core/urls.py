from django.urls import path

from apps.core import views

urlpatterns = [
    path('', views.home, name='home'),
    path('platja/<str:slug>', views.webcam, name='beach'),
    path('analitza/', views.analyze_image, name='analyze-image'),
]