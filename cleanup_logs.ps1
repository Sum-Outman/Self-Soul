# Self Soul AGI System - Log Cleanup Script (PowerShell)
# This script deletes all log files from the system

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Self Soul AGI System - Log Cleanup Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will delete all log files from the system." -ForegroundColor Yellow
Write-Host "WARNING: This action cannot be undone!" -ForegroundColor Red
Write-Host ""

$confirm = Read-Host "Are you sure you want to delete all log files? (y/n)"
if ($confirm -ne 'y' -and $confirm -ne 'Y') {
    Write-Host "Operation cancelled." -ForegroundColor Gray
    exit
}

Write-Host ""
Write-Host "Deleting log files..." -ForegroundColor Green

# Delete logs from main logs directory
if (Test-Path "logs\*.log") {
    Write-Host "Deleting files from logs directory..." -ForegroundColor Gray
    Remove-Item "logs\*.log" -Force
    Write-Host "Deleted: logs\*.log" -ForegroundColor Green
}

if (Test-Path "logs\*.log.*") {
    Write-Host "Deleting rotated log files..." -ForegroundColor Gray
    Remove-Item "logs\*.log.*" -Force
    Write-Host "Deleted: logs\*.log.*" -ForegroundColor Green
}

# Delete logs from core/logs directory
if (Test-Path "core\logs\*.log") {
    Write-Host "Deleting files from core\logs directory..." -ForegroundColor Gray
    Remove-Item "core\logs\*.log" -Force
    Write-Host "Deleted: core\logs\*.log" -ForegroundColor Green
}

if (Test-Path "core\logs\*.log.*") {
    Write-Host "Deleting rotated log files from core\logs..." -ForegroundColor Gray
    Remove-Item "core\logs\*.log.*" -Force
    Write-Host "Deleted: core\logs\*.log.*" -ForegroundColor Green
}

# Delete any other log files in project root
if (Test-Path "*.log") {
    Write-Host "Deleting log files in root directory..." -ForegroundColor Gray
    Remove-Item "*.log" -Force
    Write-Host "Deleted: *.log" -ForegroundColor Green
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Log cleanup completed!" -ForegroundColor Green
Write-Host ""
Write-Host "To clear conversation history in the browser:" -ForegroundColor Yellow
Write-Host "1. Open http://localhost:5175/#/ in Chrome/Firefox" -ForegroundColor Gray
Write-Host "2. Press F12 to open Developer Tools" -ForegroundColor Gray
Write-Host "3. Go to Application tab (Chrome) or Storage tab (Firefox)" -ForegroundColor Gray
Write-Host "4. Find Local Storage and delete 'self_soul_conversation_history'" -ForegroundColor Gray
Write-Host "5. Refresh the page" -ForegroundColor Gray
Write-Host ""
Write-Host "Alternative: Run the JavaScript console command:" -ForegroundColor Yellow
Write-Host "localStorage.removeItem('self_soul_conversation_history'); location.reload();" -ForegroundColor Gray
Write-Host "============================================" -ForegroundColor Cyan