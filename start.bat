call .venv\Scripts\Activate.bat
start python manage.py run_scheduler
waitress-serve --listen=127.0.0.1:8001 --threads=8 config.wsgi:application
