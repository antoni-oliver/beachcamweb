call .venv\Scripts\Activate.bat
waitress-serve --listen=127.0.0.1:8001 --threads=8 config.wsgi:application
