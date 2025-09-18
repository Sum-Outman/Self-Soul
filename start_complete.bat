@echo off
chcp 65001 >nul
echo ========================================
echo    Self Soul  AGI System - Complete Startup
echo ========================================
echo.

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM 检查Node.js环境
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed or not in PATH
    pause
    exit /b 1
)

echo Starting Backend Server (FastAPI) on port 8000...
start "Self Soul  Backend" cmd /k "cd /d %~dp0 && python -m uvicorn core.main:app --host 0.0.0.0 --port 8000 --reload"

echo Waiting for backend to start...
timeout /t 3 /nobreak >nul

echo Starting Frontend Server (Vite) on port 5175...
start "Self Soul  Frontend" cmd /k "cd /d %~dp0\app && npm run dev"

echo.
echo ========================================
echo    System Starting...
echo    Backend: http://localhost:8000
echo    Frontend: http://localhost:5175
echo ========================================
echo.
echo Press any key to stop all services...
pause >nul

echo Stopping all services...
taskkill /f /im node.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1

echo All services stopped.
pause
