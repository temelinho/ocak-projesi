@echo off
setlocal

cd /d "%~dp0"

echo =========================================
echo   Ocak AI - Baslatiliyor
echo =========================================
echo.

set "BACKEND_DIR="
set "FRONTEND_DIR="

for /d %%D in ("*") do (
  if exist "%%~fD\backend\server.py" set "BACKEND_DIR=%%~fD\backend"
  if exist "%%~fD\react-frontend\package.json" set "FRONTEND_DIR=%%~fD\react-frontend"
)

if "%BACKEND_DIR%"=="" (
  echo [HATA] Backend klasoru bulunamadi. Beklenen: */backend/server.py
  pause
  exit /b 1
)

if "%FRONTEND_DIR%"=="" (
  echo [HATA] Frontend klasoru bulunamadi. Beklenen: */react-frontend/package.json
  pause
  exit /b 1
)

echo [1/2] Backend baslatiliyor...
start "Ocak AI Backend" cmd /k "cd /d \"%BACKEND_DIR%\" && python server.py"

echo [2/2] Frontend baslatiliyor...
start "Ocak AI Frontend" cmd /k "cd /d \"%FRONTEND_DIR%\" && npm run dev"

echo.
echo Tamam. Servisler aciliyor:
echo - Backend : http://localhost:8000
echo - Frontend: http://localhost:5173
echo.
echo Bu pencereyi kapatabilirsiniz.

endlocal
