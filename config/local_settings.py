# Production settings
class PRODUCTION_SETTINGS:
    DEBUG = False
    ALLOWED_HOSTS = ['.localhost', 'ocupacioplatges.uib.es']


# Development settings
class DEVELOPMENT_SETTINGS:
    DEBUG = True
    ALLOWED_HOSTS = ['*']


LOCAL_SETTINGS = PRODUCTION_SETTINGS
