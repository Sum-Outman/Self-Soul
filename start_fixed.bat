@echo off
REM Self Soul  Complete Startup Script - Starts both frontend and backend services
REM Fixed version without Chinese characters to avoid encoding issues

echo ========================================
echo        Self Soul  System Starting
echo        AGI Brain System Starting
echo ========================================

REM Check current directory
if exist "app\" (
    echo Current directory: %cd%
    set "ROOT_DIR=%cd%"
) else if exist "..\app\" (
    echo Switching to parent directory
    cd ..
    set "ROOT_DIR=%cd%"
) else (
    echo Error: app directory not found
    echo Please ensure you are running this script from the project root directory (e:\Self Soul ) or its subdirectories
    pause
    exit /b 1
)

echo Project root directory: %ROOT_DIR%

REM Check if Python is installed
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python not found, please install Python 3.8+
    echo Visit https://www.python.org/downloads/ to download and install Python
    pause
    exit /b 1
)

REM Check if Node.js is installed
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Node.js not found, please install Node.js
    echo Visit https://nodejs.org/ to download and install Node.js
    pause
    exit /b 1
)

REM Check Python dependencies
echo Checking Python dependencies...
cd %ROOT_DIR%
if not exist "agi_env_311\" (
    echo Creating Python virtual environment...
    python -m venv agi_env_311
)

echo Activating virtual environment and installing dependencies...
call agi_env_311\Scripts\activate.bat
pip install --upgrade pip
if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo Warning: requirements.txt file not found
)

REM Check frontend dependencies
echo Checking frontend dependencies...
cd %ROOT_DIR%\app
if not exist "node_modules\" (
    echo Installing frontend dependencies...
    npm install
)

REM Start backend service
echo Starting backend service...
cd %ROOT_DIR%
start "Self Soul  Backend" cmd /k "call agi_env_311\Scripts\activate.bat && python -m core.main"

REM Wait for backend service to start
echo Waiting for backend service to start (5 seconds)...
timeout /t 5 >nul

REM Start frontend service
echo Starting frontend service...
cd %ROOT_DIR%\app
start "Self Soul  Frontend" npm run dev

REM Display access information
echo.
echo ========================================
echo        Services Started Successfully!
echo ========================================
echo Frontend: http://localhost:5175/
echo Backend API: http://localhost:5001/
echo API Docs: http://localhost:5001/docs
echo.
echo Opening browser in 3 seconds...
timeout /t 3 >nul
start "" "http://localhost:5175/"
echo.
echo If you encounter issues, please check:
echo 1. Ports 5001 and 5175 are not occupied
echo 2. All dependencies are installed correctly
echo 3. Check console output for error messages
pause
