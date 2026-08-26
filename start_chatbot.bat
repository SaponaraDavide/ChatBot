@echo off
REM Avvia UniBot dalla cartella in cui si trova questo file.
cd /d "%~dp0"

REM Attiva il virtualenv se presente.
if exist "venv\Scripts\activate.bat" call "venv\Scripts\activate.bat"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

streamlit run app.py
pause
