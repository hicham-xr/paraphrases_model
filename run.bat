@echo off
REM Lance l'interface Streamlit dans l'environnement virtuel du projet.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Environnement introuvable. Cree-le avec :
    echo   py -3.12 -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py
pause
