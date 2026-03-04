from django.shortcuts import get_object_or_404, render
from django.utils.safestring import mark_safe

from django.http import JsonResponse

from apps.webcam.models import WebCam
from apps.core.forms import ImageUploaderForm

from predictions.classes.BayesianPredictor import BayesianPredictor
from predictions.actions.CustomerPredict import CustomerPredict


import logging
logger = logging.getLogger(__name__)


def home(request):
    """ Returns home page. """
    beachcams = list(WebCam.objects.filter(num_consecutive_failures__lte=10).all())
    #beachcams = list(WebCam.objects.filter(snapshots__ts__gte=now() - timedelta(hours=2)))
    return render(request, 'core/home.html', context={'cams': beachcams})


def webcam(request, camera_slug):
    """ Returns ajax_image of latest prediction overimposed on captured image. """
    beachcam = get_object_or_404(WebCam, camera_slug=camera_slug)
    other_beachcams = WebCam.objects.exclude(camera_slug=camera_slug)
    #other_beachcams = WebCam.objects.exclude(camera_slug=camera_slug).filter(snapshots__ts__gte=now() - timedelta(hours=2))
    # history_dates, history_counts = zip(*[[f"'{h.ts.isoformat()}'", round(h.predicted_crowd_count)] 
    #                                       for h in beachcam.history() if h.predicted_crowd_count is not None])
    # history_dates = f'[{",".join([str(a) for a in list(history_dates)])}]'
    # history_counts = f'[{",".join([str(a) for a in list(history_counts)])}]'
    history = [ [h.ts.timestamp() * 1000, h.predicted_crowd_count] 
               for h in beachcam.history() if h.predicted_crowd_count is not None]
    return render(request, 'core/beach.html', context={'cam': beachcam, 'other_cams': other_beachcams, 'prediction': beachcam.last_prediction, 'history': history})

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