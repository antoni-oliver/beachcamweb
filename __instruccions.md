# Activació del virtual environment:
.venv\Scripts\activate


# Traducció
## Per a detectar noves strings a traduir:
python.exe .\manage.py makemessages -all

## Per a compilar els .po a .mo
python.exe .\manage.py compilemessages --ignore .venv