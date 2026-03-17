from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.contrib.admin.views.decorators import staff_member_required

from django.http import JsonResponse

from apps.prediction.models import Snapshot
from apps.webcam.models import WebCam
from apps.webcam import utils as webcam_utils
from apps.core.forms import WebcamFiltersForm, SnapshotFiltersUpdate, WebcamMaskPolygonForm
from apps.core.forms import ImageUploaderForm
from config import settings
from django.http import HttpResponse
from apps.prediction import pipeline_config as cfg
from django.utils.timezone import now
from datetime import timedelta

from django.views.decorators.cache import cache_page

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

@staff_member_required
def admin_webcam_filters(request, webcam_id):
    webcam = get_object_or_404(WebCam, pk=webcam_id)
    latest_snapshot = webcam.snapshots.exclude(webcam_image='').order_by('-ts').first()
    all_webcams = WebCam.objects.select_related('beach').order_by('camera_slug')

    if request.method == 'POST':
        form_type = request.POST.get('form_type')

        if form_type == 'filters':
            form = WebcamFiltersForm(request.POST, instance=webcam)
            mask_form = WebcamMaskPolygonForm()

            if form.is_valid():
                form.save()
                return redirect(
                    f"{reverse('admin-webcam-filters', kwargs={'webcam_id': webcam.id})}?saved_filters=1"
                )

        elif form_type == 'mask_polygon':
            form = WebcamFiltersForm(instance=webcam)
            mask_form = WebcamMaskPolygonForm(request.POST)

            if not latest_snapshot or not latest_snapshot.webcam_image:
                mask_form.add_error(None, 'No hay una imagen base disponible para generar la máscara.')
            elif mask_form.is_valid():
                webcam_utils.save_polygon_mask(
                    webcam=webcam,
                    mask_field_name=mask_form.cleaned_data['mask_field'],
                    polygon_points=mask_form.cleaned_data['polygon_points'],
                    reference_image_path=latest_snapshot.webcam_image.path,
                )
                return redirect(
                    f"{reverse('admin-webcam-filters', kwargs={'webcam_id': webcam.id})}?saved_mask=1"
                )

        else:
            form = WebcamFiltersForm(instance=webcam)
            mask_form = WebcamMaskPolygonForm()
    else:
        form = WebcamFiltersForm(instance=webcam)
        mask_form = WebcamMaskPolygonForm()

    return render(
        request,
        'core/admin_webcam_filters.html',
        {
            'webcam': webcam,
            'latest_snapshot': latest_snapshot,
            'form': form,
            'mask_form': mask_form,
            'all_webcams': all_webcams,
            'current_view_name': 'admin-webcam-filters',
        }
    )


@staff_member_required
def admin_webcam_snapshots_filters(request, webcam_id):
    webcam = get_object_or_404(WebCam, pk=webcam_id)
    snapshots_qs = webcam.snapshots.exclude(webcam_image='').order_by('-ts')
    all_webcams = WebCam.objects.select_related('beach').order_by('camera_slug')

    updated_count = None

    if request.method == 'POST':
        form = SnapshotFiltersUpdate(request.POST)
        if form.is_valid():
            timestamp_since = form.cleaned_data.get('timestamp_since')
            timestamp_until = form.cleaned_data.get('timestamp_until')

            snapshots_to_update = webcam.snapshots.all()

            if timestamp_since:
                snapshots_to_update = snapshots_to_update.filter(ts__gte=timestamp_since)
            if timestamp_until:
                snapshots_to_update = snapshots_to_update.filter(ts__lte=timestamp_until)

            updated_count = snapshots_to_update.update(
                filter_frozen_image=form.cleaned_data['filter_frozen_image'],
                filter_blurry_image=form.cleaned_data['filter_blurry_image'],
                filter_moving_camera=form.cleaned_data['filter_moving_camera'],
            )

            snapshots_qs = webcam.snapshots.exclude(webcam_image='').order_by('-ts')
    else:
        form = SnapshotFiltersUpdate()

    return render(
        request,
        'core/admin_webcam_snapshots_filters.html',
        {
            'webcam': webcam,
            'form': form,
            'snapshots': snapshots_qs,
            'updated_count': updated_count,
            'all_webcams': all_webcams,
            'current_view_name': 'admin-webcam-snapshots-filters',
        }
    )


def beach_overview(request):
    cams = WebCam.objects.select_related('beach').filter(snapshots__isnull=False).distinct()
    return render(request, 'core/beach_overview.html', {'cams': cams})


def all_beaches_forecast(request):
    return render(request, 'core/all_beaches_forecast.html')

@cache_page(60 * 60)
def beach_history_api(request, camera_slug):
    cam = get_object_or_404(WebCam, camera_slug=camera_slug)
    since = now() - timedelta(days=365)

    base_qs = (
        Snapshot.objects
        .filter(webcam=cam, predicted_crowd_count__isnull=False, ts__gte=since)
        .order_by('ts')
    )
    rows = list(base_qs.values_list('ts', 'predicted_crowd_count'))

    data = [[ts.isoformat(), round(float(count))] for ts, count in rows]

    base = getattr(settings, 'MEDIA_URL', '')
    latest = (
        Snapshot.objects
        .filter(webcam=cam, predicted_image__isnull=False)
        .exclude(predicted_image='')
        .order_by('-ts')
        .values_list('predicted_image', flat=True)
        .first()
    )
    image_url = base + latest if latest else None

    return JsonResponse({'data': data, 'image_url': image_url, 'max_crowd_count': cam.max_crowd_count or 0})

@staff_member_required
def fundaciobit_debug(request):
    return render(request, 'core/fundaciobit_debug.html')

def pipeline_report(request):
    if not cfg.NOTEBOOK_HTML.exists():
        return HttpResponse('<h2>No report generated yet. Run the pipeline first.</h2>', status=404)
    return HttpResponse(cfg.NOTEBOOK_HTML.read_text(), content_type='text/html')

def forecast_model_report(request):
    cam = WebCam.objects.filter(max_crowd_count__gt=0).first()
    return render(request, 'core/forecast_model_report.html', {'default_slug': cam.camera_slug})