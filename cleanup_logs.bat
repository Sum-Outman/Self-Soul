@echo off
echo ============================================
echo Self Soul AGI System - Log Cleanup Script
echo ============================================
echo.
echo This script will delete all log files from the system.
echo WARNING: This action cannot be undone!
echo.

set /p confirm="Are you sure you want to delete all log files? (y/n): "
if /i "%confirm%" neq "y" (
    echo Operation cancelled.
    pause
    exit /b 0
)

echo.
echo Deleting log files...

:: Delete logs from main logs directory
if exist "logs\*.log" (
    echo Deleting files from logs directory...
    del "logs\*.log" /q
    echo Deleted: logs\*.log
)

if exist "logs\*.log.*" (
    echo Deleting rotated log files...
    del "logs\*.log.*" /q
    echo Deleted: logs\*.log.*
)

:: Delete logs from core/logs directory
if exist "core\logs\*.log" (
    echo Deleting files from core\logs directory...
    del "core\logs\*.log" /q
    echo Deleted: core\logs\*.log
)

if exist "core\logs\*.log.*" (
    echo Deleting rotated log files from core\logs...
    del "core\logs\*.log.*" /q
    echo Deleted: core\logs\*.log.*
)

:: Delete any other log files in project root
if exist "*.log" (
    echo Deleting log files in root directory...
    del "*.log" /q
    echo Deleted: *.log
)

echo.
echo ============================================
echo Log cleanup completed!
echo.
echo To clear conversation history in the browser:
echo 1. Open http://localhost:5175/#/ in Chrome/Firefox
echo 2. Press F12 to open Developer Tools
echo 3. Go to Application tab (Chrome) or Storage tab (Firefox)
echo 4. Find Local Storage and delete "self_soul_conversation_history"
echo 5. Refresh the page
echo ============================================
echo.
pause