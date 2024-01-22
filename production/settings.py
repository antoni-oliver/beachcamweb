import os

try:
    from production.secrets import secrets
except ImportError:
    class DummySecrets:
        def __getattr__(self, name):
            return 'No secrets file found'
    secrets = DummySecrets()    # Use as 'secrets.ANYTHING_HERE'

# Development settings
SECRET_KEY = 'hola :)'
DEBUG = True

BASE_DIR_LOCAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR_LOCAL, 'db.sqlite3'),
    }
}
# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.postgresql_psycopg2',
#         'NAME': 'autoscrapper',
#         'USER': 'autoscrapper',
#         'PASSWORD': 'estrategias',
#         'HOST': 'localhost',
#         'PORT': '5432',
#     }
# }
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# Production settings (generated from this information)
class FABRIC:
    PROJECT_NAME = 'beachcamweb'
    MAIN_APP = 'core'
    SERVERS = [{
        'NAME': 'servidor_virtual_uib',
        'USER': 'beachcamweb',
        'PWD': secrets.REMOTE_ROOT_PWD,
        'KEYFILE_LOCALPATH': '/Users/pedro/.ssh/id_ed25519',
        'IP': '130.206.132.12',
    }]
    DOMAINS = ['ocupacioplatges.uib.es']   # The domain(s) used by your site
    REQUIREMENTS_PATH = 'requirements.txt'  # Project's pip requirements
    SECRET_KEY = secrets.DJANGO_SECRET_KEY

    LOCALE = 'en_US.UTF-8'  # Should end with ".UTF-8"
    DB_PWD = secrets.PRODUCTION_DB_PWD  # Live database pwd

    class SUPERUSER:
        NAME = 'pedro'
        EMAIL = 'p.bibiloni@uib.es'
        PWD = secrets.DJANGO_SUPERUSER_PWD

    class DJANGO_EMAIL:
        BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
        HOST = 'smtp.ionos.es'
        PORT = 587
        USE_TLS = True
        USER = 'p.bibiloni@uib.es'
        PWD = secrets.DJANGO_EMAIL_PWD

    class GITHUB:
        REPO_PATH = 'PBibiloni/beachcamweb.git'
        USER = 'PBibiloni'
        AUTH_TOKEN = secrets.GITHUB_AUTH_TOKEN

    class GUNICORN:
        NUM_WORKERS = 9
