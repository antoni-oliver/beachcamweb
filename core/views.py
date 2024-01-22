from django.shortcuts import render

from data.models import BeachCam


# Create your views here.


def home(request):
    """ Returns home page. """
    beachcams = list(BeachCam.objects.all())
    return render(request, 'core/home.html', context={'cams': beachcams})


def ajax_image(request, beach_name):
    """ Returns ajax_image of latest prediction overimposed on captured image. """
    return render(request, 'core/ajax_image.html')
