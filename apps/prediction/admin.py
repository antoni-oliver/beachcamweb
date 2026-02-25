from django.contrib import admin
from django.db.models import Q

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


@admin.register(Snapshot)
class SnapshotAdmin(AdminFiltersMixin, admin.ModelAdmin):
    list_display = ("webcam", "ts", "predicted_crowd_count", "webcam_image", "webcam_video")
    list_filter = (
        ("webcam", RelatedFieldComboFilter),
        ("ts", DateInDateRangeFilter),
        ("predicted_crowd_count", NumberFilter),
        HasWebcamImageFilter,
        HasWebcamVideoFilter,
    )
    search_fields = ("webcam__beach_name",)
    date_hierarchy = "ts"