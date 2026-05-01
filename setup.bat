@echo off
REM Setup script for Windows

python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt

echo.
echo Setup complete. To get started:
echo   venv\Scripts\activate
echo   python run.py
