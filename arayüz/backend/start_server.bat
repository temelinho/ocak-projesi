@echo off
echo ========================================
echo   Ocak Ses Analiz - Backend Sunucusu
echo ========================================
echo.

REM ocak-projesi venv'indeki Python ile calistir
set PYTHON_PATH=c:\projelerim\ocak ses\ocak-projesi\.venv\Scripts\python.exe

echo Python: %PYTHON_PATH%
echo Sunucu: http://localhost:8000
echo.

"%PYTHON_PATH%" -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload

pause
