from django.shortcuts import get_object_or_404, render
from django.utils.safestring import mark_safe

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
    history_dates, history_counts = zip(*[[f"'{h.ts.isoformat()}'", round(h.predicted_crowd_count)] for h in beachcam.history()])
    history_dates = f'[{",".join([str(a) for a in list(history_dates)])}]'
    history_counts = f'[{",".join([str(a) for a in list(history_counts)])}]'
    history = [[h.ts.timestamp() * 1000, h.predicted_crowd_count] for h in beachcam.history()]
    return render(request, 'core/beach.html', context={'cam': beachcam, 'other_cams': other_beachcams, 'prediction': beachcam.last_prediction, 'history': history, 'history_dates': mark_safe(history_dates), 'history_counts': mark_safe(history_counts)})

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
