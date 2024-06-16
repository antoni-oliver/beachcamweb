from django.shortcuts import render

from data.models import BeachCam
from django.shortcuts import get_object_or_404, render

from django.http import JsonResponse
from .forms import ImageUploaderForm

from predictions.classes.BayesianPredictor import BayesianPredictor
from predictions.actions.CustomerPredict import CustomerPredict


# Create your views here.


def home(request):
    """ Returns home page. """
    beachcams = list(BeachCam.objects.all())
    return render(request, 'core/home.html', context={'cams': beachcams})


def show_image(request, beach_name):
    """ Returns ajax_image of latest prediction overimposed on captured image. """
    beachcam = get_object_or_404(BeachCam, beach_name=beach_name)
    other_beachcams = BeachCam.objects.exclude(beach_name=beach_name)
    return render(request, 'core/show_image.html', context={'cam': beachcam, 'other_cams': other_beachcams, 'prediction': beachcam.last_prediction})

def show_analyze_image(request):
    # https://docs.djangoproject.com/en/5.0/topics/forms/
    if request.method == "POST":
        form = ImageUploaderForm(request.POST, request.FILES)
        if form.is_valid():
            cleaned_data = form.cleaned_data
            image = cleaned_data.get('image')
            predictor = BayesianPredictor()
            action = CustomerPredict()
            predictionDTO = action.handle(image, predictor)
            if predictionDTO is None:
                return JsonResponse({'errors': "Internal Server Error"}, status=500)
            else:
                return JsonResponse(predictionDTO.to_dict(), status=200)
        else:
            return JsonResponse({'errors': form.errors}, status=400)
    else:
        form = ImageUploaderForm()
        return render(request, 'core/analyze_image.html',  context={'form': form})
