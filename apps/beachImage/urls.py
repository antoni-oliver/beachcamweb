from django.urls import path
from . import views

urlpatterns = [
    # Upload endpoint
    path('upload/', views.upload_image, name='upload-image'),
    
    # Image management
    path('images/', views.get_images, name='list-images'),
    path('images/<int:pk>/', views.get_image, name='get-image'),
    path('images/<int:pk>/delete/', views.delete_image, name='delete-image'),
]

"""
Available endpoints:

POST   /api/upload/              - Upload beach image with GPS coordinates
GET    /api/images/              - List all beach images
GET    /api/images/{id}/         - Get specific beach image
DELETE /api/images/{id}/delete/  - Delete beach image
"""