call .venv\Scripts\Activate.bat
start python manage.py run_scheduler
rem uvicorn main:app --host 0.0.0.0 --port 8001
python .\manage.py runserver 8001