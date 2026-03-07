@echo off
REM Start script for Unified Cognitive Architecture (Windows)

echo =========================================
echo Starting Unified Cognitive Architecture
echo =========================================

REM Check Python version
python --version
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check if in src directory
if not exist "src" (
    echo Error: Please run from new_agi_system directory
    pause
    exit /b 1
)

REM Check for virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM Check if torch is installed
python -c "import torch" 2>nul
if errorlevel 1 (
    echo Installing PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

REM Start the API server
echo Starting API server on port 9000...
cd src
python -m api.server --host 127.0.0.1 --port 9000

pause