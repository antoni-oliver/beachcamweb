# Production settings
class PRODUCTION_SETTINGS:
    DEBUG = False
    ALLOWED_HOSTS = ['.localhost', 'ocupacioplatges.uib.es', 'ocupacioplatges.uib.cat', 'ocupacioplatges.uib.eu']
    SECRET_KEY = '4=1r*v^1a@%&vy31242x^e&jk123l1sddd2l3g5!yje3+g_ycwyn#6$j8y9aur3$'

# Development settings
class DEVELOPMENT_SETTINGS:
    DEBUG = True
    ALLOWED_HOSTS = ['*']
    SECRET_KEY = 'django-insecure-qh0123jn123jk12nk38!123ddwb820%y__#0z=v#6r5ot_2(^f(x6@ah-g'


LOCAL_SETTINGS = PRODUCTION_SETTINGS
