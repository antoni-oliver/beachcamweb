from django import forms

class ImageUploaderForm(forms.Form):
    image = forms.ImageField(
        label="Selecciona una imatge per analitzar",
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control input-image',
            'accept': 'image/*'
        })
    )