import sys, time, jwt
from config.local_settings import LOCAL_SETTINGS

now = int(time.time())
exp_in = int(sys.argv[1]) if len(sys.argv) > 1 else 3600
exp = exp_in if exp_in > now else now + exp_in

print(jwt.encode({"iat": now, "exp": exp}, LOCAL_SETTINGS.API_JWT_SECRET,
                 algorithm=(LOCAL_SETTINGS.API_JWT_ALGOS or ["HS256"])[0]))