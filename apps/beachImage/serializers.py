from rest_framework import serializers
from .models import BeachImage


class BeachImageSerializer(serializers.ModelSerializer):
    """
    Serializer for BeachImage model
    """
    
    class Meta:
        model = BeachImage
        fields = ['id', 'device', 'image', 'longitude', 'latitude', 'timestamp']
        read_only_fields = ['id']
    
    def validate_image(self, value):
        """
        Validate image file size
        """
        max_size = 10 * 1024 * 1024  # 10MB
        if value.size > max_size:
            raise serializers.ValidationError(
                "Image too large. Maximum size is 10MB."
            )
        return value
    
    def validate_longitude(self, value):
        """
        Validate longitude is within valid range
        """
        if value < -180 or value > 180:
            raise serializers.ValidationError(
                "Longitude must be between -180 and 180."
            )
        return value
    
    def validate_latitude(self, value):
        """
        Validate latitude is within valid range
        """
        if value < -90 or value > 90:
            raise serializers.ValidationError(
                "Latitude must be between -90 and 90."
            )
        return value
