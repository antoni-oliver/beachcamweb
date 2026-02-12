from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser

from .models import BeachImage
from .serializers import BeachImageSerializer


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_image(request):
    """
    Upload a beach image with GPS coordinates
    
    POST /api/upload/
    
    Required fields:
    - device: Device id (foreign key)
    - image: Image file (multipart/form-data)
    - longitude: Longitude coordinate (decimal)
    - latitude: Latitude coordinate (decimal)
    - timestamp: ISO 8601 timestamp
    
    Response:
    {
        "success": true,
        "id": 123
    }
    """
    serializer = BeachImageSerializer(data=request.data)
    
    if serializer.is_valid():
        beach_image = serializer.save()
        return Response({
            'success': True,
            'id': beach_image.id
        }, status=status.HTTP_201_CREATED)
    
    return Response({
        'success': False,
        'errors': serializer.errors
    }, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
def get_images(request):
    """
    Get all beach images with optional filtering
    
    GET /api/images/
    
    Query parameters:
    - start_date: ISO 8601 date (YYYY-MM-DD)
    - end_date: ISO 8601 date (YYYY-MM-DD)
    - limit: Number of results (default 50, max 500)
    
    Response:
    {
        "success": true,
        "count": 10,
        "data": [
            {
                "id": 1,
                "image": "http://...",
                "longitude": -118.243683,
                "latitude": 34.052235,
                "timestamp": "2024-01-15T10:30:00Z"
            },
            ...
        ]
    }
    """
    queryset = BeachImage.objects.all()
    
    # Filter by date range
    start_date = request.query_params.get('start_date')
    end_date = request.query_params.get('end_date')
    
    if start_date:
        queryset = queryset.filter(timestamp__gte=start_date)
    if end_date:
        queryset = queryset.filter(timestamp__lte=end_date)
    
    # Limit results
    limit = int(request.query_params.get('limit', 50))
    limit = min(limit, 500)
    queryset = queryset[:limit]
    
    serializer = BeachImageSerializer(queryset, many=True, context={'request': request})
    
    return Response({
        'success': True,
        'count': len(serializer.data),
        'data': serializer.data
    })


@api_view(['GET'])
def get_image(request, pk):
    """
    Get a specific beach image by ID
    
    GET /api/images/{id}/
    
    Response:
    {
        "success": true,
        "data": {
            "id": 1,
            "image": "http://...",
            "longitude": -118.243683,
            "latitude": 34.052235,
            "timestamp": "2024-01-15T10:30:00Z"
        }
    }
    """
    try:
        beach_image = BeachImage.objects.get(pk=pk)
    except BeachImage.DoesNotExist:
        return Response({
            'success': False,
            'error': 'Image not found'
        }, status=status.HTTP_404_NOT_FOUND)
    
    serializer = BeachImageSerializer(beach_image, context={'request': request})
    return Response({
        'success': True,
        'data': serializer.data
    })


@api_view(['DELETE'])
def delete_image(request, pk):
    """
    Delete a beach image
    
    DELETE /api/images/{id}/
    
    Response:
    {
        "success": true,
        "message": "Image deleted"
    }
    """
    try:
        beach_image = BeachImage.objects.get(pk=pk)
        beach_image.delete()
        return Response({
            'success': True,
            'message': 'Image deleted'
        }, status=status.HTTP_200_OK)
    except BeachImage.DoesNotExist:
        return Response({
            'success': False,
            'error': 'Image not found'
        }, status=status.HTTP_404_NOT_FOUND)