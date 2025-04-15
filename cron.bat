call .venv\Scripts\Activate.bat
echo %DATE% %TIME% >> download.log
python .\download_and_process.py >> download.log