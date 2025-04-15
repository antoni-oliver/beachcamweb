call .venv\Scripts\Activate.bat
echo %DATE% %TIME% >> logs/download.log
python .\download_and_process.py >> logs/download.log