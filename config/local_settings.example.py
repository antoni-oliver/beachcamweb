# Production settings
class PRODUCTION_SETTINGS:
    DEBUG = False
    ALLOWED_HOSTS = ['.localhost', 'ocupacioplatges.uib.es', 'ocupacioplatges.uib.cat', 'ocupacioplatges.uib.eu']
    SECRET_KEY = 'django-insecure-qh0123jn123jk12nk38!123ddwb820%y__#0z=v#6r5ot_2(^f(x6@ah-g'
    API_JWT_SECRET = "changeme123123123"
    API_JWT_ALGOS = ["HS256"]

# Development settings
class DEVELOPMENT_SETTINGS:
    DEBUG = True
    ALLOWED_HOSTS = ['*']
    SECRET_KEY = 'django-insecure-qh0123jn123jk12nk38!123ddwb820%y__#0z=v#6r5ot_2(^f(x6@ah-g'
    API_JWT_SECRET = "changeme123123123"
    API_JWT_ALGOS = ["HS256"]


LOCAL_SETTINGS = DEVELOPMENT_SETTINGS
