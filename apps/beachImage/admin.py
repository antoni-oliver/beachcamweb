from django.contrib import admin
from .models import BeachImage


@admin.register(BeachImage)
class BeachImageAdmin(admin.ModelAdmin):
    """
    Admin interface for BeachImage model
    """
    list_display = ['id', 'latitude', 'longitude', 'timestamp', 'image']
    list_filter = ['timestamp']
    search_fields = ['id']
    readonly_fields = ['id']
    date_hierarchy = 'timestamp'
    
    fieldsets = (
        ('Image', {
            'fields': ('image',)
        }),
        ('GPS Coordinates', {
            'fields': ('latitude', 'longitude')
        }),
        ('Timestamp', {
            'fields': ('timestamp',)
        }),
        ('System', {
            'fields': ('id',),
            'classes': ('collapse',)
        }),
    )
