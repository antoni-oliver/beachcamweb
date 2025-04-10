from django.shortcuts import get_object_or_404, render

from django.http import JsonResponse

from apps.webcam.models import WebCam
from apps.core.forms import ImageUploaderForm

from predictions.classes.BayesianPredictor import BayesianPredictor
from predictions.actions.CustomerPredict import CustomerPredict


# Create your views here.


def home(request):
    """ Returns home page. """
    beachcams = list(WebCam.objects.all())
    return render(request, 'core/home.html', context={'cams': beachcams})


def webcam(request, slug):
    """ Returns ajax_image of latest prediction overimposed on captured image. """
    beachcam = get_object_or_404(WebCam, slug=slug)
    other_beachcams = WebCam.objects.exclude(slug=slug)
    return render(request, 'core/beach.html', context={'cam': beachcam, 'other_cams': other_beachcams, 'prediction': beachcam.last_prediction})

def analyze_image(request):
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
