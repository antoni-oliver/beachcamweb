from django.contrib import admin
from apps.prediction import models
from apps.webcam.models import WebCam


class WebCamFilter(admin.SimpleListFilter):
    title = 'webcam'
    parameter_name = 'webcam'

    def lookups(self, request, model_admin):
        webcams = WebCam.objects.order_by('camera_slug')
        return [(webcam.id, str(webcam)) for webcam in webcams]

    def queryset(self, request, queryset):
        if self.value():
            return queryset.filter(webcam_id=self.value())
        return queryset


@admin.register(models.Snapshot)
class SnapshotAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'webcam',
        'get_beach',
        'ts',
        'filter_frozen_image',
        'filter_blurry_image',
        'filter_moving_camera',
        'predicted_crowd_count',
    )
    list_filter = (
        WebCamFilter,
        'filter_frozen_image',
        'filter_blurry_image',
        'filter_moving_camera',
    )
    search_fields = (
        'webcam__camera_slug',
        'webcam__beach__beach_name',
    )
    date_hierarchy = 'ts'
    ordering = ('-ts',)

    @admin.display(description='Playa')
    def get_beach(self, obj):
        return obj.webcam.beach.beach_name