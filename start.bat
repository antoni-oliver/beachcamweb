call .venv\Scripts\Activate.bat
rem uvicorn main:app --host 0.0.0.0 --port 8001
python .\manage.py crontab add
python .\manage.py crontab add
python .\manage.py runserver 8001