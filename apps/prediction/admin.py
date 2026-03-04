from django.contrib import admin
from django.db.models import Q, Count
from adminfilters.mixin import AdminFiltersMixin
from adminfilters.filters import (
    RelatedFieldComboFilter,
    DateInDateRangeFilter,
    NumberFilter,
)
from .models import Snapshot


class HasWebcamImageFilter(admin.SimpleListFilter):
    title = "Has webcam image"
    parameter_name = "has_webcam_image"

    def lookups(self, request, model_admin):
        return (("1", "Yes"), ("0", "No"))

    def queryset(self, request, queryset):
        val = self.value()
        if val == "1":
            return queryset.exclude(Q(webcam_image__isnull=True) | Q(webcam_image=""))
        if val == "0":
            return queryset.filter(Q(webcam_image__isnull=True) | Q(webcam_image=""))
        return queryset


class HasWebcamVideoFilter(admin.SimpleListFilter):
    title = "Has webcam video"
    parameter_name = "has_webcam_video"

    def lookups(self, request, model_admin):
        return (("1", "Yes"), ("0", "No"))

    def queryset(self, request, queryset):
        val = self.value()
        if val == "1":
            return queryset.exclude(Q(webcam_video__isnull=True) | Q(webcam_video=""))
        if val == "0":
            return queryset.filter(Q(webcam_video__isnull=True) | Q(webcam_video=""))
        return queryset


class HasPredictionFilter(admin.SimpleListFilter):
    title = "Has prediction"
    parameter_name = "has_prediction"

    def lookups(self, request, model_admin):
        return (("1", "Yes"), ("0", "No"))

    def queryset(self, request, queryset):
        if self.value() == "1":
            return queryset.exclude(predicted_crowd_count__isnull=True)
        if self.value() == "0":
            return queryset.filter(predicted_crowd_count__isnull=True)
        return queryset


@admin.register(Snapshot)
class SnapshotAdmin(AdminFiltersMixin, admin.ModelAdmin):
    list_display = ("get_beach", "webcam", "ts", "predicted_crowd_count", "webcam_image", "webcam_video")
    list_filter = (
        ("webcam__beach", RelatedFieldComboFilter),
        ("webcam", RelatedFieldComboFilter),
        ("ts", DateInDateRangeFilter),
        ("predicted_crowd_count", NumberFilter),
        HasPredictionFilter,
        HasWebcamImageFilter,
        HasWebcamVideoFilter,
    )
    search_fields = ("webcam__camera_slug", "webcam__beach__beach_name")
    date_hierarchy = "ts"
    list_select_related = ("webcam", "webcam__beach")
    list_per_page = 50

    @admin.display(description="Beach", ordering="webcam__beach__beach_name")
    def get_beach(self, obj):
        return obj.webcam.beach.beach_name