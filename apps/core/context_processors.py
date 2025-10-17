from django.conf import settings
from django.utils import translation
import json

HOST_LANGUAGE = {
    'ocupacioplatges.uib.es': 'es-es',
    'ocupacioplatges.uib.cat': 'ca-es',
    'ocupacioplatges.uib.eu': 'en-us',
}

def parent_context(request):
    global HOST_LANGUAGE

    host = request.get_host()
    lang = HOST_LANGUAGE.get(host, settings.LANGUAGE_CODE)
    translation.activate(lang)
    request.LANGUAGE_CODE = translation.get_language()
    request.session['django_language'] = lang

    return {}