from django import forms
from django.core.exceptions import ValidationError
import json
from apps.webcam.models import WebCam

class ImageUploaderForm(forms.Form):
    image = forms.ImageField(
        label="Selecciona una imatge per analitzar",
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control input-image',
            'accept': 'image/*'
        })
    )
class WebcamFiltersForm(forms.ModelForm):
    class Meta:
        model = WebCam
        fields = [
            'filter_frozen_image',
            'filter_blurry_image',
            'filter_moving_camera',
        ]
        widgets = {
            'filter_frozen_image': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'filter_blurry_image': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'filter_moving_camera': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

class SnapshotFiltersUpdate(forms.Form):
    filter_frozen_image = forms.BooleanField(required=False, widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}))
    filter_blurry_image = forms.BooleanField(required=False, widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}))
    filter_moving_camera = forms.BooleanField(required=False, widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}))

    timestamp_since = forms.DateTimeField(
        required=False,
        widget=forms.DateTimeInput(attrs={'type': 'datetime-local', 'class': 'form-control'}),
    )
    timestamp_until = forms.DateTimeField(
        required=False,
        widget=forms.DateTimeInput(attrs={'type': 'datetime-local', 'class': 'form-control'}),
    )

    def clean(self):
        cleaned_data = super().clean()
        timestamp_since = cleaned_data.get('timestamp_since')
        timestamp_until = cleaned_data.get('timestamp_until')

        if timestamp_since and timestamp_until and timestamp_since > timestamp_until:
            raise forms.ValidationError('timestamp_since no puede ser posterior a timestamp_until.')

        return cleaned_data

class WebcamMaskPolygonForm(forms.Form):
    MASK_FIELD_CHOICES = (
        ('mask_sand', 'Mask sand'),
        ('mask_swimming_water', 'Mask swimming water'),
    )

    mask_field = forms.ChoiceField(
        choices=MASK_FIELD_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'}),
    )
    polygon_points = forms.CharField(
        widget=forms.HiddenInput(attrs={'id': 'coordinates'}),
    )

    def clean_polygon_points(self):
        raw_value = self.cleaned_data['polygon_points']

        if not raw_value:
            raise ValidationError('Debes dibujar un polígono antes de guardar la máscara.')

        try:
            points = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValidationError('Las coordenadas del polígono no son válidas.') from exc

        if not isinstance(points, list) or len(points) < 3:
            raise ValidationError('El polígono debe tener al menos 3 puntos.')

        normalized_points = []
        for point in points:
            if not isinstance(point, dict):
                raise ValidationError('Formato de punto no válido.')

            x = point.get('x')
            y = point.get('y')

            if x is None or y is None:
                raise ValidationError('Cada punto debe incluir x e y.')

            try:
                normalized_points.append({
                    'x': int(round(float(x))),
                    'y': int(round(float(y))),
                })
            except (TypeError, ValueError) as exc:
                raise ValidationError('Las coordenadas del polígono no son válidas.') from exc

        return normalized_points