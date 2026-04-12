@echo off
echo ========================================
echo   Ocak Ses Analiz - Backend Sunucusu
echo ========================================
echo.

REM Egittigimiz modelin bulundugu venv (Python 3.13 + TensorFlow) ile calistir
set PYTHON_PATH="%~dp0..\..\ocak-projesi\venv_train\Scripts\python.exe"

echo Python: %PYTHON_PATH%
echo Sunucu: http://localhost:8000
echo.

%PYTHON_PATH% -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload

pause
